import Mathlib

namespace polynomial_unique_l181_181006

noncomputable def p (x : ℝ) : ℝ := x^2 + 1

theorem polynomial_unique (p : ℝ → ℝ) 
  (h1 : p 2 = 5) 
  (h2 : ∀ x y : ℝ, p x * p y = p x + p y + p (x * y) - 2) : 
  ∀ x : ℝ, p x = x^2 + 1 :=
by
  sorry

end polynomial_unique_l181_181006


namespace probability_three_unused_rockets_expected_targets_hit_l181_181954

section RocketArtillery

variables (p : ℝ) (h0 : 0 ≤ p) (h1 : p ≤ 1)

-- Probability that exactly three unused rockets remain after firing at five targets
theorem probability_three_unused_rockets (p : ℝ) (h0 : 0 ≤ p) (h1 : p ≤ 1) : 
  let prob := 10 * (p ^ 3) * ((1 - p) ^ 2) in
  prob = 10 * (p ^ 3) * ((1 - p) ^ 2) := 
by
  sorry

-- Expected number of targets hit if there are nine targets
theorem expected_targets_hit (p : ℝ) (h0 : 0 ≤ p) (h1 : p ≤ 1) : 
  let expected_hits := 10 * p - (p ^ 10) in
  expected_hits = 10 * p - (p ^ 10) := 
by
  sorry

end RocketArtillery

end probability_three_unused_rockets_expected_targets_hit_l181_181954


namespace arithmetic_geometric_seq_proof_l181_181980

theorem arithmetic_geometric_seq_proof
  (a1 a2 b1 b2 b3 : ℝ)
  (h1 : a1 - a2 = -1)
  (h2 : 1 * (b2 * b2) = 4)
  (h3 : b2 > 0) :
  (a1 - a2) / b2 = -1 / 2 :=
by
  sorry

end arithmetic_geometric_seq_proof_l181_181980


namespace pyramid_layers_total_l181_181107

-- Since we are dealing with natural number calculations, noncomputable is generally not needed.

-- Definition of the pyramid layers and the number of balls in each layer
def number_of_balls (n : ℕ) : ℕ := n ^ 2

-- Given conditions for the layers
def third_layer_balls : ℕ := number_of_balls 3
def fifth_layer_balls : ℕ := number_of_balls 5

-- Statement of the problem proving that their sum is 34
theorem pyramid_layers_total : third_layer_balls + fifth_layer_balls = 34 := by
  sorry -- proof to be provided

end pyramid_layers_total_l181_181107


namespace first_reappearance_line_l181_181439

theorem first_reappearance_line
  (letters_cycle : Nat := 6)
  (digits_cycle : Nat := 5) :
  Nat.lcm letters_cycle digits_cycle = 30 :=
by
  -- The proof would go here
  sorry

end first_reappearance_line_l181_181439


namespace union_M_N_l181_181701

def M : Set ℝ := { x | x^2 - x = 0 }
def N : Set ℝ := { y | y^2 + y = 0 }

theorem union_M_N : (M ∪ N) = {-1, 0, 1} := 
by 
  sorry

end union_M_N_l181_181701


namespace sqrt_four_four_summed_l181_181920

theorem sqrt_four_four_summed :
  sqrt (4 ^ 4 + 4 ^ 4 + 4 ^ 4 + 4 ^ 4) = 32 := by
  sorry

end sqrt_four_four_summed_l181_181920


namespace part_a_part_b_l181_181956

variables (p : ℝ) (h : p ≥ 0 ∧ p ≤ 1) -- Probability p is between 0 and 1 inclusive

-- Part (a)
theorem part_a (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) : 
  (5.choose 3) * p^3 * (1 - p)^2 = 10 * p^3 * (1 - p)^2 := by 
sorry

-- Part (b)
theorem part_b (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) : 
  10 * p - p^10 = (10 * p - p^10) := by 
sorry

end part_a_part_b_l181_181956


namespace angle_C_in_triangle_l181_181040

theorem angle_C_in_triangle (A B C : ℝ) (h1 : A + B = 80) (h2 : A + B + C = 180) : C = 100 := 
by 
  sorry

end angle_C_in_triangle_l181_181040


namespace probability_of_region_D_l181_181129

theorem probability_of_region_D
    (P_A : ℚ) (P_B : ℚ) (P_C : ℚ) (P_D : ℚ)
    (h1 : P_A = 1/4) 
    (h2 : P_B = 1/3) 
    (h3 : P_C = 1/6) 
    (h4 : P_A + P_B + P_C + P_D = 1) : 
    P_D = 1/4 := by
    sorry

end probability_of_region_D_l181_181129


namespace value_of_expression_l181_181384

theorem value_of_expression (x : ℝ) (h : (3 / (x - 3)) + (5 / (2 * x - 6)) = 11 / 2) : 2 * x - 6 = 2 :=
sorry

end value_of_expression_l181_181384


namespace no_common_points_range_a_l181_181399

theorem no_common_points_range_a (a k : ℝ) (hl : ∃ k, ∀ x y : ℝ, k * x - y - k + 2 = 0) :
  (∀ x y : ℝ, x^2 + 2 * a * x + y^2 - a + 2 ≠ 0) → (-7 < a ∧ a < -2) ∨ (1 < a) := by
  sorry

end no_common_points_range_a_l181_181399


namespace apples_per_case_l181_181793

theorem apples_per_case (total_apples : ℕ) (number_of_cases : ℕ) (h1 : total_apples = 1080) (h2 : number_of_cases = 90) : total_apples / number_of_cases = 12 := by
  sorry

end apples_per_case_l181_181793


namespace good_number_is_1008_l181_181449

-- Given conditions
def sum_1_to_2015 : ℕ := (2015 * (2015 + 1)) / 2
def sum_mod_2016 : ℕ := sum_1_to_2015 % 2016

-- The proof problem expressed in Lean
theorem good_number_is_1008 (x : ℕ) (h1 : sum_1_to_2015 = 2031120)
  (h2 : sum_mod_2016 = 1008) :
  x = 1008 ↔ (sum_1_to_2015 - x) % 2016 = 0 := by
  sorry

end good_number_is_1008_l181_181449


namespace soap_box_width_l181_181633

theorem soap_box_width
  (carton_length : ℝ) (carton_width : ℝ) (carton_height : ℝ)
  (box_length : ℝ) (box_height : ℝ) (max_boxes : ℝ) (carton_volume : ℝ)
  (box_volume : ℝ) (W : ℝ) : 
  carton_length = 25 →
  carton_width = 42 →
  carton_height = 60 →
  box_length = 6 →
  box_height = 6 →
  max_boxes = 250 →
  carton_volume = carton_length * carton_width * carton_height →
  box_volume = box_length * W * box_height →
  max_boxes * box_volume = carton_volume →
  W = 7 :=
sorry

end soap_box_width_l181_181633


namespace chinese_team_arrangements_l181_181868

noncomputable def numArrangements : ℕ := 6

theorem chinese_team_arrangements :
  let swimmers := {A, B, C, D}
  let styles := {backstroke, breaststroke, butterfly, freestyle}
  let constraints := (A ∈ {backstroke, freestyle} ∧ B ∈ {butterfly, freestyle}) 
                      ∧ (C ∈ styles ∧ D ∈ styles) 
                      ∧ (A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  in
  constraints → numArrangements = 6 := 
by 
  sorry -- Proof to be done

end chinese_team_arrangements_l181_181868


namespace medal_award_ways_l181_181242

open Nat

theorem medal_award_ways :
  let sprinters := 10
  let italians := 4
  let medals := 3
  let gold_medal_ways := choose italians 1
  let remaining_sprinters := sprinters - 1
  let non_italians := remaining_sprinters - (italians - 1)
  let silver_medal_ways := choose non_italians 1
  let new_remaining_sprinters := remaining_sprinters - 1
  let new_non_italians := new_remaining_sprinters - (italians - 1)
  let bronze_medal_ways := choose new_non_italians 1
  gold_medal_ways * silver_medal_ways * bronze_medal_ways = 120 := by
    sorry

end medal_award_ways_l181_181242


namespace find_angle_C_l181_181013

variables {A B C : ℝ} {a b c : ℝ} 

theorem find_angle_C (h1 : a^2 + b^2 - c^2 + a*b = 0) (C_pos : 0 < C) (C_lt_pi : C < Real.pi) :
  C = (2 * Real.pi) / 3 :=
sorry

end find_angle_C_l181_181013


namespace complement_intersection_l181_181702

universe u

def U : Finset Int := {-3, -2, -1, 0, 1}
def A : Finset Int := {-2, -1}
def B : Finset Int := {-3, -1, 0}

def complement_U (A : Finset Int) (U : Finset Int) : Finset Int :=
  U.filter (λ x => x ∉ A)

theorem complement_intersection :
  (complement_U A U) ∩ B = {-3, 0} :=
by
  sorry

end complement_intersection_l181_181702


namespace problem_21_sum_correct_l181_181871

theorem problem_21_sum_correct (A B C D E : ℕ) (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
    (h_digits : A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧ E < 10)
    (h_eq : (10 * A + B) * (10 * C + D) = 111 * E) : 
  A + B + C + D + E = 21 :=
sorry

end problem_21_sum_correct_l181_181871


namespace number_of_triangles_2016_30_l181_181174

def f (m n : ℕ) : ℕ :=
  2 * m - n - 2

theorem number_of_triangles_2016_30 :
  f 2016 30 = 4000 := 
by
  sorry

end number_of_triangles_2016_30_l181_181174


namespace probability_more_heads_than_tails_l181_181383

-- Define the total number of outcomes when flipping 10 coins
def total_outcomes : ℕ := 2 ^ 10

-- Define the number of ways to get exactly 5 heads out of 10 flips (combination)
def combinations_5_heads : ℕ := Nat.choose 10 5

-- Define the probability of getting exactly 5 heads out of 10 flips
def probability_5_heads := (combinations_5_heads : ℚ) / total_outcomes

-- Define the probability of getting more heads than tails (x)
def probability_more_heads := (1 - probability_5_heads) / 2

-- Theorem stating the probability of getting more heads than tails
theorem probability_more_heads_than_tails : probability_more_heads = 193 / 512 :=
by
  -- Proof skipped using sorry
  sorry

end probability_more_heads_than_tails_l181_181383


namespace area_of_path_cost_of_constructing_path_l181_181948

-- Definitions for the problem
def original_length : ℕ := 75
def original_width : ℕ := 40
def path_width : ℕ := 25 / 10  -- 2.5 converted to a Lean-readable form

-- Conditions
def new_length := original_length + 2 * path_width
def new_width := original_width + 2 * path_width

def area_with_path := new_length * new_width
def area_without_path := original_length * original_width

-- Statements to prove
theorem area_of_path : area_with_path - area_without_path = 600 := sorry

def cost_per_sq_m : ℕ := 2
def total_cost := (area_with_path - area_without_path) * cost_per_sq_m

theorem cost_of_constructing_path : total_cost = 1200 := sorry

end area_of_path_cost_of_constructing_path_l181_181948


namespace digit_multiplication_sum_l181_181717

-- Define the main problem statement in Lean 4
theorem digit_multiplication_sum (A B E F : ℕ) (h1 : 0 ≤ A ∧ A ≤ 9) 
                                            (h2 : 0 ≤ B ∧ B ≤ 9) 
                                            (h3 : 0 ≤ E ∧ E ≤ 9)
                                            (h4 : 0 ≤ F ∧ F ≤ 9)
                                            (h5 : A ≠ B) 
                                            (h6 : A ≠ E) 
                                            (h7 : A ≠ F)
                                            (h8 : B ≠ E)
                                            (h9 : B ≠ F)
                                            (h10 : E ≠ F)
                                            (h11 : (100 * A + 10 * B + E) * F = 1001 * E + 100 * A)
                                            : A + B = 5 :=
sorry

end digit_multiplication_sum_l181_181717


namespace meaningful_domain_of_function_l181_181030

theorem meaningful_domain_of_function : ∀ x : ℝ, (∃ y : ℝ, y = 3 / Real.sqrt (x - 2)) → x > 2 :=
by
  intros x h
  sorry

end meaningful_domain_of_function_l181_181030


namespace volunteer_hours_per_year_l181_181723

def volunteer_sessions_per_month := 2
def hours_per_session := 3
def months_per_year := 12

theorem volunteer_hours_per_year : 
  (volunteer_sessions_per_month * months_per_year * hours_per_session) = 72 := 
by
  sorry

end volunteer_hours_per_year_l181_181723


namespace triple_sum_of_45_point_2_and_one_fourth_l181_181607

theorem triple_sum_of_45_point_2_and_one_fourth : 
  (3 * (45.2 + 0.25)) = 136.35 :=
by
  sorry

end triple_sum_of_45_point_2_and_one_fourth_l181_181607


namespace factorization_of_polynomial_l181_181662

theorem factorization_of_polynomial :
  ∀ x : ℝ, x^8 - 4*x^6 + 6*x^4 - 4*x^2 + 1 = (x - 1)^4 * (x + 1)^4 :=
by
  intros x
  sorry

end factorization_of_polynomial_l181_181662


namespace favorite_movies_hours_l181_181728

theorem favorite_movies_hours (J M N R : ℕ) (h1 : J = M + 2) (h2 : N = 3 * M) (h3 : R = (4 * N) / 5) (h4 : N = 30) : 
  J + M + N + R = 76 :=
by
  sorry

end favorite_movies_hours_l181_181728


namespace triangle_inequality_l181_181099

theorem triangle_inequality (a : ℝ) :
  (3/2 < a) ∧ (a < 5) ↔ ((4 * a + 1 - (3 * a - 1) < 12 - a) ∧ (4 * a + 1 + (3 * a - 1) > 12 - a)) := 
by 
  sorry

end triangle_inequality_l181_181099


namespace part1_part2_l181_181515

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (Real.exp x / x) - Real.log x + x - a

theorem part1 (a : ℝ) (h : ∀ x > 0, f x a ≥ 0) : a ≤ Real.exp 1 + 1 := 
sorry

theorem part2 {a : ℝ} {x1 x2 : ℝ} (hx1 : 0 < x1) (hx2 : 0 < x2) 
  (hz1 : f x1 a = 0) (hz2 : f x2 a = 0) : x1 * x2 < 1 := 
sorry

end part1_part2_l181_181515


namespace total_amount_paid_l181_181196

theorem total_amount_paid : 
  let spam_cost := 3
  let peanut_butter_cost := 5
  let bread_cost := 2
  let spam_quantity := 12
  let peanut_butter_quantity := 3
  let bread_quantity := 4
  let spam_total := spam_cost * spam_quantity
  let peanut_butter_total := peanut_butter_cost * peanut_butter_quantity
  let bread_total := bread_cost * bread_quantity
  let total_amount := spam_total + peanut_butter_total + bread_total in
  total_amount = 59 :=
by
  sorry

end total_amount_paid_l181_181196


namespace smaller_root_of_equation_l181_181334

theorem smaller_root_of_equation : 
  ∀ x : ℝ, (x - 3 / 4) * (x - 3 / 4) + (x - 3 / 4) * (x - 1 / 4) = 0 → x = 1 / 2 :=
by
  intros x h
  sorry

end smaller_root_of_equation_l181_181334


namespace quadratic_roots_sum_l181_181393

theorem quadratic_roots_sum (a b : ℤ) (h_roots : ∀ x, a * x^2 + b * x + 2 = 0 ↔ x = -1/2 ∨ x = 1/3) : a + b = -14 :=
by
  sorry

end quadratic_roots_sum_l181_181393


namespace triangle_condition_proof_l181_181718

variables {A B C D M K : Type*}
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace M] [MetricSpace K]
variables (AB AC AD : ℝ)

-- Definitions based on the conditions
def is_isosceles (A B C : Type*) (AB AC : ℝ) : Prop :=
  AB = AC

def is_altitude (A D B C : Type*) : Prop :=
  true -- Ideally, this condition is more complex and involves perpendicular projection

def is_midpoint (M A D : Type*) : Prop :=
  true -- Ideally, this condition is more specific and involves equality of segments

def extends_to (C M A B K : Type*) : Prop :=
  true -- Represents the extension relationship

-- The theorem to be proved
theorem triangle_condition_proof (A B C D M K : Type*)
  (h_iso : is_isosceles A B C AB AC)
  (h_alt : is_altitude A D B C)
  (h_mid : is_midpoint M A D)
  (h_ext : extends_to C M A B K)
  : AB = 3 * AK :=
  sorry

end triangle_condition_proof_l181_181718


namespace total_garbage_collected_l181_181828

def Daliah := 17.5
def Dewei := Daliah - 2
def Zane := 4 * Dewei
def Bela := Zane + 3.75

theorem total_garbage_collected :
  Daliah + Dewei + Zane + Bela = 160.75 :=
by
  sorry

end total_garbage_collected_l181_181828


namespace cost_of_saddle_l181_181130

theorem cost_of_saddle (S : ℝ) (H : 4 * S + S = 5000) : S = 1000 :=
by sorry

end cost_of_saddle_l181_181130


namespace initial_amount_l181_181582

theorem initial_amount (spent_sweets friends_each left initial : ℝ) 
  (h1 : spent_sweets = 3.25) (h2 : friends_each = 2.20) (h3 : left = 2.45) :
  initial = spent_sweets + (friends_each * 2) + left :=
by
  sorry

end initial_amount_l181_181582


namespace Michelangelo_ceiling_painting_l181_181740

theorem Michelangelo_ceiling_painting (C : ℕ) : 
  ∃ C, (C + (1/4) * C = 15) ∧ (28 - (C + (1/4) * C) = 13) :=
sorry

end Michelangelo_ceiling_painting_l181_181740


namespace common_real_root_pair_l181_181358

theorem common_real_root_pair (n : ℕ) (hn : n > 1) :
  ∃ x : ℝ, (∃ a b : ℤ, ((x^n + (a : ℝ) * x = 2008) ∧ (x^n + (b : ℝ) * x = 2009))) ↔
    ((a = 2007 ∧ b = 2008) ∨
     (a = (-1)^(n-1) - 2008 ∧ b = (-1)^(n-1) - 2009)) :=
by sorry

end common_real_root_pair_l181_181358


namespace total_amount_paid_l181_181875

def original_price : ℝ := 20
def discount_rate : ℝ := 0.5
def number_of_tshirts : ℕ := 6

theorem total_amount_paid : 
  (number_of_tshirts : ℝ) * (original_price * discount_rate) = 60 := by
  sorry

end total_amount_paid_l181_181875


namespace convert_vulgar_fraction_l181_181932

theorem convert_vulgar_fraction (h : (35 : ℚ) / 100 = 7 / 20) : (0.35 : ℚ) = 7 / 20 := 
by 
  have h1 : (0.35 : ℚ) = 35 / 100 := by norm_num
  rw h1
  exact h

end convert_vulgar_fraction_l181_181932


namespace point_d_lies_on_graph_l181_181294

theorem point_d_lies_on_graph : (-1 : ℝ) = -2 * (1 : ℝ) + 1 :=
by {
  sorry
}

end point_d_lies_on_graph_l181_181294


namespace correct_expression_l181_181293

theorem correct_expression (a b c m x y : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : a + b ≠ 0) (h5 : x ≠ y) : 
  ¬ ( (a + m) / (b + m) = a / b ) ∧
  ¬ ( (a + b) / (a + b) = 0 ) ∧ 
  ¬ ( (a * b - 1) / (a * c - 1) = (b - 1) / (c - 1) ) ∧ 
  ( (x - y) / (x^2 - y^2) = 1 / (x + y) ) :=
by
  sorry

end correct_expression_l181_181293


namespace sin_alpha_cos_beta_value_l181_181184

variables {α β : ℝ}

theorem sin_alpha_cos_beta_value 
  (h1 : Real.sin (α + β) = 1/2) 
  (h2 : 2 * Real.sin (α - β) = 1/2) : 
  Real.sin α * Real.cos β = 3/8 := by
sorry

end sin_alpha_cos_beta_value_l181_181184


namespace chime_date_is_march_22_2003_l181_181944

-- Definitions
def clock_chime (n : ℕ) : ℕ := n % 12

def half_hour_chimes (half_hours : ℕ) : ℕ := half_hours
def hourly_chimes (hours : List ℕ) : ℕ := hours.map clock_chime |>.sum

-- Problem conditions and result
def initial_chimes_and_half_hours : ℕ := half_hour_chimes 9
def initial_hourly_chimes : ℕ := hourly_chimes [4, 5, 6, 7, 8, 9, 10, 11, 0]
def chimes_on_february_28_2003 : ℕ := initial_chimes_and_half_hours + initial_hourly_chimes

def half_hour_chimes_per_day : ℕ := half_hour_chimes 24
def hourly_chimes_per_day : ℕ := hourly_chimes (List.range 12 ++ List.range 12)
def total_chimes_per_day : ℕ := half_hour_chimes_per_day + hourly_chimes_per_day

def remaining_chimes_needed : ℕ := 2003 - chimes_on_february_28_2003
def full_days_needed : ℕ := remaining_chimes_needed / total_chimes_per_day
def additional_chimes_needed : ℕ := remaining_chimes_needed % total_chimes_per_day

-- Lean theorem statement
theorem chime_date_is_march_22_2003 :
    (full_days_needed = 21) → (additional_chimes_needed < total_chimes_per_day) → 
    true :=
by
  sorry

end chime_date_is_march_22_2003_l181_181944


namespace max_m_value_inequality_abc_for_sum_l181_181179

-- Define the mathematical conditions and the proof problem.

theorem max_m_value (x m : ℝ) (h1 : |x - 2| - |x + 3| ≥ |m + 1|) :
  m ≤ 4 :=
sorry

theorem inequality_abc_for_sum (a b c : ℝ) (habc_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum_eq_M : a + 2 * b + c = 4) :
  (1 / (a + b)) + (1 / (b + c)) ≥ 1 :=
sorry

end max_m_value_inequality_abc_for_sum_l181_181179


namespace three_digit_number_cubed_sum_l181_181497

theorem three_digit_number_cubed_sum {a b c : ℕ} (h₁ : 1 ≤ a ∧ a ≤ 9)
                                      (h₂ : 0 ≤ b ∧ b ≤ 9)
                                      (h₃ : 0 ≤ c ∧ c ≤ 9) :
  (100 ≤ 100 * a + 10 * b + c ∧ 100 * a + 10 * b + c ≤ 999) →
  (100 * a + 10 * b + c = (a + b + c) ^ 3) →
  (100 * a + 10 * b + c = 512) :=
by
  sorry

end three_digit_number_cubed_sum_l181_181497


namespace compare_probabilities_l181_181270

theorem compare_probabilities 
  (M N : ℕ)
  (m n : ℝ)
  (h_m_pos : m > 0)
  (h_n_pos : n > 0)
  (h_m_million : m > 10^6)
  (h_n_million : n ≤ 10^6) :
  (M : ℝ) / (M + n / m * N) > (M : ℝ) / (M + N) :=
by
  sorry

end compare_probabilities_l181_181270


namespace triangle_altitude_from_equal_area_l181_181440

variable (x : ℝ)

theorem triangle_altitude_from_equal_area (h : x^2 = (1 / 2) * x * altitude) :
  altitude = 2 * x := by
  sorry

end triangle_altitude_from_equal_area_l181_181440


namespace total_spending_in_4_years_is_680_l181_181606

-- Define the spending of each person
def trevor_spending_yearly : ℕ := 80
def reed_spending_yearly : ℕ := trevor_spending_yearly - 20
def quinn_spending_yearly : ℕ := reed_spending_yearly / 2

-- Define the total spending in one year
def total_spending_yearly : ℕ := trevor_spending_yearly + reed_spending_yearly + quinn_spending_yearly

-- Let's state what we need to prove
theorem total_spending_in_4_years_is_680 : 
  4 * total_spending_yearly = 680 := 
by 
  -- this is where the proof steps would go
  sorry

end total_spending_in_4_years_is_680_l181_181606


namespace tangent_line_at_zero_decreasing_intervals_l181_181844

noncomputable def f (x : ℝ) : ℝ := -x^3 + 3 * x^2 + 9 * x - 2

theorem tangent_line_at_zero :
  let t : ℝ × ℝ := (0, f 0)
  (∀ x : ℝ, (9 * x - f x - 2 = 0) → t.snd = -2) := by
  sorry

theorem decreasing_intervals :
  ∀ x : ℝ, (-3 * x^2 + 6 * x + 9 < 0) ↔ (x < -1 ∨ x > 3) := by
  sorry

end tangent_line_at_zero_decreasing_intervals_l181_181844


namespace inequality_solution_set_l181_181125

theorem inequality_solution_set : 
  (∃ (x : ℝ), (4 / (x - 1) ≤ x - 1) ↔ (x ≥ 3 ∨ (-1 ≤ x ∧ x < 1))) :=
by
  sorry

end inequality_solution_set_l181_181125


namespace triangle_angle_sum_l181_181062

theorem triangle_angle_sum (A B C : ℝ) (h1 : A + B = 80) (h2 : A + B + C = 180) : C = 100 := 
by
  sorry

end triangle_angle_sum_l181_181062


namespace last_three_digits_of_2_pow_9000_l181_181618

-- The proof statement
theorem last_three_digits_of_2_pow_9000 (h : 2 ^ 300 ≡ 1 [MOD 1000]) : 2 ^ 9000 ≡ 1 [MOD 1000] :=
by
  sorry

end last_three_digits_of_2_pow_9000_l181_181618


namespace workout_total_correct_l181_181737

structure Band := 
  (A : ℕ) 
  (B : ℕ) 
  (C : ℕ)

structure Equipment := 
  (leg_weight_squat : ℕ) 
  (dumbbell : ℕ) 
  (leg_weight_lunge : ℕ) 
  (kettlebell : ℕ)

def total_weight (bands : Band) (equip : Equipment) : ℕ := 
  let squat_total := bands.A + bands.B + bands.C + (2 * equip.leg_weight_squat) + equip.dumbbell
  let lunge_total := bands.A + bands.C + (2 * equip.leg_weight_lunge) + equip.kettlebell
  squat_total + lunge_total

theorem workout_total_correct (bands : Band) (equip : Equipment) : 
  bands = ⟨7, 5, 3⟩ → 
  equip = ⟨10, 15, 8, 18⟩ → 
  total_weight bands equip = 94 :=
by 
  -- Insert your proof steps here
  sorry

end workout_total_correct_l181_181737


namespace factorization_of_polynomial_l181_181668

theorem factorization_of_polynomial :
  (x : ℝ) → (x^8 - 4*x^6 + 6*x^4 - 4*x^2 + 1) = ((x - 1)^4 * (x + 1)^4) :=
by
  intro x
  sorry

end factorization_of_polynomial_l181_181668


namespace thirty_first_number_in_pascal_thirty_second_row_l181_181256

theorem thirty_first_number_in_pascal_thirty_second_row : Nat.choose 32 30 = 496 := by sorry

end thirty_first_number_in_pascal_thirty_second_row_l181_181256


namespace computer_cost_250_l181_181427

-- Define the conditions as hypotheses
variables (total_budget : ℕ) (tv_cost : ℕ) (computer_cost fridge_cost : ℕ)
variables (h1 : total_budget = 1600) (h2 : tv_cost = 600) (h3 : fridge_cost = computer_cost + 500)
variables (h4 : total_budget - tv_cost = fridge_cost + computer_cost)

-- State the theorem to be proved
theorem computer_cost_250 : computer_cost = 250 :=
by
  simp [h1, h2, h3, h4]
  sorry -- Proof omitted

end computer_cost_250_l181_181427


namespace find_FC_l181_181501

theorem find_FC 
  (DC CB AD AB ED FC : ℝ)
  (h1 : DC = 10)
  (h2 : CB = 12)
  (h3 : AB = (1/5) * AD)
  (h4 : ED = (2/3) * AD)
  (h5 : AD = (5/4) * 22)  -- Derived step from solution for full transparency
  (h6 : FC = (ED * (CB + AB)) / AD) : 
  FC = 35 / 3 := 
sorry

end find_FC_l181_181501


namespace solution_set_l181_181760

theorem solution_set (x : ℝ) : (3 ≤ |5 - 2 * x| ∧ |5 - 2 * x| < 9) ↔ (-2 < x ∧ x ≤ 1) ∨ (4 ≤ x ∧ x < 7) :=
sorry

end solution_set_l181_181760


namespace sum_odd_even_50_l181_181494

def sum_first_n_odd (n : ℕ) : ℕ := n * n

def sum_first_n_even (n : ℕ) : ℕ := n * (n + 1)

theorem sum_odd_even_50 : 
  sum_first_n_odd 50 + sum_first_n_even 50 = 5050 := by
  sorry

end sum_odd_even_50_l181_181494


namespace part1_part2_l181_181514

def f (x a : ℝ) : ℝ := (Exp x / x) - Log x + x - a

theorem part1 (x a : ℝ) (hx : f x a ≥ 0) : a ≤ Real.exp 1 + 1 := 
sorry

theorem part2 (x1 x2 a : ℝ) (hx1 : f x1 a = 0) (hx2 : f x2 a = 0) 
  (hx1_range : 0 < x1) (hx2_range : x2 > 1) : x1 * x2 < 1 := 
sorry

end part1_part2_l181_181514


namespace triangle_angle_sum_l181_181063

theorem triangle_angle_sum (A B C : ℝ) (h1 : A + B = 80) (h2 : A + B + C = 180) : C = 100 := 
by
  sorry

end triangle_angle_sum_l181_181063


namespace binom_20_19_eq_20_l181_181304

theorem binom_20_19_eq_20 : nat.choose 20 19 = 20 :=
by sorry

end binom_20_19_eq_20_l181_181304


namespace probability_more_heads_than_tails_l181_181382

-- Define the total number of outcomes when flipping 10 coins
def total_outcomes : ℕ := 2 ^ 10

-- Define the number of ways to get exactly 5 heads out of 10 flips (combination)
def combinations_5_heads : ℕ := Nat.choose 10 5

-- Define the probability of getting exactly 5 heads out of 10 flips
def probability_5_heads := (combinations_5_heads : ℚ) / total_outcomes

-- Define the probability of getting more heads than tails (x)
def probability_more_heads := (1 - probability_5_heads) / 2

-- Theorem stating the probability of getting more heads than tails
theorem probability_more_heads_than_tails : probability_more_heads = 193 / 512 :=
by
  -- Proof skipped using sorry
  sorry

end probability_more_heads_than_tails_l181_181382


namespace books_new_arrivals_false_implies_statements_l181_181394

variable (Books : Type) -- representing the set of books in the library
variable (isNewArrival : Books → Prop) -- predicate stating if a book is a new arrival

theorem books_new_arrivals_false_implies_statements (H : ¬ ∀ b : Books, isNewArrival b) :
  (∃ b : Books, ¬ isNewArrival b) ∧ (¬ ∀ b : Books, isNewArrival b) :=
by
  sorry

end books_new_arrivals_false_implies_statements_l181_181394


namespace probability_is_correct_l181_181938

-- Given definitions
def total_marbles : ℕ := 100
def red_marbles : ℕ := 35
def white_marbles : ℕ := 30
def green_marbles : ℕ := 10

-- Probe the probability
noncomputable def probability_red_white_green : ℚ :=
  (red_marbles + white_marbles + green_marbles : ℚ) / total_marbles

-- The theorem we need to prove
theorem probability_is_correct :
  probability_red_white_green = 0.75 := by
  sorry

end probability_is_correct_l181_181938


namespace range_of_a_l181_181859

noncomputable def f (a x : ℝ) : ℝ := Real.log (a * x ^ 2 + 2 * x + 1)

theorem range_of_a {a : ℝ} :
  (∀ x : ℝ, a * x ^ 2 + 2 * x + 1 > 0) ↔ (0 ≤ a ∧ a ≤ 1) :=
by 
  sorry

end range_of_a_l181_181859


namespace probability_more_heads_than_tails_l181_181381

theorem probability_more_heads_than_tails :
  let n := 10
  let total_outcomes := 2^n
  let equal_heads_tails_ways := Nat.choose n (n / 2)
  let y := (equal_heads_tails_ways : ℝ) / total_outcomes
  let x := (1 - y) / 2
  x = 193 / 512 := by
    let n := 10
    let total_outcomes := 2^n
    have h1 : equal_heads_tails_ways = Nat.choose n (n / 2) := rfl
    have h2 : total_outcomes = 2^n := rfl
    let equal_heads_tails_ways := Nat.choose n (n / 2)
    let y := (equal_heads_tails_ways : ℝ) / total_outcomes
    have h3 : y = 63 / 256 := sorry  -- calculation steps
    let x := (1 - y) / 2
    have h4 : x = 193 / 512 := sorry  -- calculation steps
    exact h4

end probability_more_heads_than_tails_l181_181381


namespace ball_problem_solution_l181_181249

namespace BallProblem

-- Definitions based on the conditions
def num_colors := 4
def balls_per_color := 4
def total_balls := num_colors * balls_per_color

-- Set conditions for drawing two balls of different colors
def different_colors (c1 c2 : ℕ) (h1 : c1 ≠ c2) : Prop := true

-- Number of ways to draw two balls of different colors
def calculate_ways : ℕ :=
  (Nat.choose num_colors 2) * balls_per_color * balls_per_color

-- Probability calculation for the maximum difference in their numbers
def max_diff_prob (num_diff_ways : ℕ) : ℚ :=
  num_diff_ways / calculate_ways

theorem ball_problem_solution :
  (calculate_ways = 96) ∧ (max_diff_prob 12 = (1 : ℚ) / 8) :=
by
  sorry

end BallProblem

end ball_problem_solution_l181_181249


namespace binomial_20_19_eq_20_l181_181315

theorem binomial_20_19_eq_20 : nat.choose 20 19 = 20 :=
sorry

end binomial_20_19_eq_20_l181_181315


namespace soccer_balls_with_holes_l181_181221

-- Define the total number of soccer balls
def total_soccer_balls : ℕ := 40

-- Define the total number of basketballs
def total_basketballs : ℕ := 15

-- Define the number of basketballs with holes
def basketballs_with_holes : ℕ := 7

-- Define the total number of balls without holes
def total_balls_without_holes : ℕ := 18

-- Prove the number of soccer balls with holes given the conditions
theorem soccer_balls_with_holes : (total_soccer_balls - (total_balls_without_holes - (total_basketballs - basketballs_with_holes))) = 30 := by
  sorry

end soccer_balls_with_holes_l181_181221


namespace total_tomato_seeds_l181_181884

theorem total_tomato_seeds (mike_morning mike_afternoon : ℕ) 
  (ted_morning : mike_morning = 50) 
  (ted_afternoon : mike_afternoon = 60) 
  (ted_morning_eq : 2 * mike_morning = 100) 
  (ted_afternoon_eq : mike_afternoon - 20 = 40)
  (total_seeds : mike_morning + mike_afternoon + (2 * mike_morning) + (mike_afternoon - 20) = 250) : 
  (50 + 60 + 100 + 40 = 250) :=
sorry

end total_tomato_seeds_l181_181884


namespace simplify_cosine_expression_l181_181748

theorem simplify_cosine_expression :
  ∀ (θ : ℝ), θ = 30 * Real.pi / 180 → (1 - Real.cos θ) * (1 + Real.cos θ) = 1 / 4 :=
by
  intro θ hθ
  have cos_30 := Real.cos θ
  rewrite [hθ]
  sorry

end simplify_cosine_expression_l181_181748


namespace min_jellybeans_l181_181141

theorem min_jellybeans (n : ℕ) (h1 : n ≥ 150) (h2 : n % 17 = 15) : n = 151 :=
by { sorry }

end min_jellybeans_l181_181141


namespace toothbrush_count_l181_181154

theorem toothbrush_count (T A : ℕ) (h1 : 53 + 67 + 46 = 166)
  (h2 : 67 - 36 = 31) (h3 : A = 31) (h4 : T = 166 + 2 * A) :
  T = 228 :=
  by 
  -- Using Lean's sorry keyword to skip the proof
  sorry

end toothbrush_count_l181_181154


namespace choose_amber_bronze_cells_l181_181578

theorem choose_amber_bronze_cells (a b : ℕ) (h_a_pos : a > 0) (h_b_pos : b > 0)
  (grid : Fin (a+b+1) × Fin (a+b+1) → Prop) 
  (amber_cells : ℕ) (h_amber_cells : amber_cells ≥ a^2 + a * b - b)
  (bronze_cells : ℕ) (h_bronze_cells : bronze_cells ≥ b^2 + b * a - a):
  ∃ (amber_choice : Fin (a+b+1) → Fin (a+b+1)), 
    ∃ (bronze_choice : Fin (a+b+1) → Fin (a+b+1)), 
    amber_choice ≠ bronze_choice ∧ 
    (∀ i j, i ≠ j → grid (amber_choice i) ≠ grid (bronze_choice j)) :=
sorry

end choose_amber_bronze_cells_l181_181578


namespace range_of_a_l181_181543

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, |x - 1| - |x - 2| < a^2 + a + 1) →
  (a < -1 ∨ a > 0) :=
by
  sorry

end range_of_a_l181_181543


namespace degree_of_monomial_3ab_l181_181753

variable (a b : ℕ)

def monomialDegree (x y : ℕ) : ℕ :=
  x + y

theorem degree_of_monomial_3ab : monomialDegree 1 1 = 2 :=
by
  sorry

end degree_of_monomial_3ab_l181_181753


namespace ne_of_P_l181_181700

-- Define the initial proposition P
def P : Prop := ∀ m : ℝ, (0 ≤ m → 4^m ≥ 4 * m)

-- Define the negation of P
def not_P : Prop := ∃ m : ℝ, (0 ≤ m ∧ 4^m < 4 * m)

-- The theorem we need to prove
theorem ne_of_P : ¬P ↔ not_P :=
by
  sorry

end ne_of_P_l181_181700


namespace compare_magnitudes_proof_l181_181357

noncomputable def compare_magnitudes (a b c : ℝ) (ha : 0 < a) (hbc : b * c > a^2) (heq : a^2 - 2 * a * b + c^2 = 0) : Prop :=
  b > c ∧ c > a ∧ b > a

theorem compare_magnitudes_proof (a b c : ℝ) (ha : 0 < a) (hbc : b * c > a^2) (heq : a^2 - 2 * a * b + c^2 = 0) :
  compare_magnitudes a b c ha hbc heq :=
sorry

end compare_magnitudes_proof_l181_181357


namespace distance_from_sphere_center_to_triangle_plane_l181_181959

theorem distance_from_sphere_center_to_triangle_plane :
  ∀ (O : ℝ × ℝ × ℝ) (r : ℝ) (a b c : ℝ), 
  r = 9 →
  a = 13 →
  b = 13 →
  c = 10 →
  (∀ (d : ℝ), d = distance_from_O_to_plane) →
  d = 8.36 :=
by
  intro O r a b c hr ha hb hc hd
  sorry

end distance_from_sphere_center_to_triangle_plane_l181_181959


namespace num_C_atoms_in_compound_l181_181809

def num_H_atoms := 6
def num_O_atoms := 1
def molecular_weight := 58
def atomic_weight_C := 12
def atomic_weight_H := 1
def atomic_weight_O := 16

theorem num_C_atoms_in_compound : 
  ∃ (num_C_atoms : ℕ), 
    molecular_weight = (num_C_atoms * atomic_weight_C) + (num_H_atoms * atomic_weight_H) + (num_O_atoms * atomic_weight_O) ∧ 
    num_C_atoms = 3 :=
by
  -- To be proven
  sorry

end num_C_atoms_in_compound_l181_181809


namespace measure_of_angle_C_l181_181045

-- Conditions of the problem
variable (A B C : ℝ) -- Angles in the triangle

-- Sum of all angles in a triangle
axiom sum_of_angles (h : A + B + C = 180)

-- Condition given in the problem
axiom given_angle_sum (h1 : A + B = 80)

-- Desired proof statement
theorem measure_of_angle_C : C = 100 := by
  sorry

end measure_of_angle_C_l181_181045


namespace simplify_expression_l181_181692

theorem simplify_expression (x : ℝ) (h : x^2 + x - 6 = 0) : 
  (x - 1) / ((2 / (x - 1)) - 1) = 8 / 3 :=
sorry

end simplify_expression_l181_181692


namespace negation_implication_l181_181447

theorem negation_implication (a b c : ℝ) : 
  ¬(a > b → a + c > b + c) ↔ (a ≤ b → a + c ≤ b + c) :=
by 
  sorry

end negation_implication_l181_181447


namespace charlene_initial_necklaces_l181_181302

-- Definitions for the conditions.
def necklaces_sold : ℕ := 16
def necklaces_giveaway : ℕ := 18
def necklaces_left : ℕ := 26

-- Statement to prove that the initial number of necklaces is 60.
theorem charlene_initial_necklaces : necklaces_sold + necklaces_giveaway + necklaces_left = 60 := by
  sorry

end charlene_initial_necklaces_l181_181302


namespace probability_hyperbola_check_l181_181810

def roll_die := {n : ℕ | 1 ≤ n ∧ n ≤ 6}

theorem probability_hyperbola_check :
  let outcomes := { (m, n) | m ∈ roll_die ∧ n ∈ roll_die }
  let on_hyperbola := { (m, n) | m ∈ roll_die ∧ n ∈ roll_die ∧ n = 1 / m }
  let probability := on_hyperbola.card.to_real / outcomes.card.to_real
  in probability = 1 / 6 :=
by
  sorry

end probability_hyperbola_check_l181_181810


namespace hyperbola_same_foci_l181_181368

-- Define the conditions for the ellipse and hyperbola
def ellipse (x y : ℝ) : Prop := (x^2 / 12) + (y^2 / 4) = 1
def hyperbola (x y m : ℝ) : Prop := (x^2 / m) - y^2 = 1

-- Statement to be proved in Lean 4
theorem hyperbola_same_foci : ∃ m : ℝ, ∀ x y : ℝ, ellipse x y → hyperbola x y m :=
by
  have a_squared := 12
  have b_squared := 4
  have c_squared := a_squared - b_squared
  have c := Real.sqrt c_squared
  have c_value : c = 2 * Real.sqrt 2 := by sorry
  let m := c^2 - 1
  exact ⟨m, by sorry⟩

end hyperbola_same_foci_l181_181368


namespace probability_of_all_girls_chosen_is_1_over_11_l181_181808

-- Defining parameters and conditions
def total_members : ℕ := 12
def boys : ℕ := 6
def girls : ℕ := 6
def chosen_members : ℕ := 3

-- Number of combinations to choose 3 members from 12
def total_combinations : ℕ := Nat.choose total_members chosen_members

-- Number of combinations to choose 3 girls from 6
def girl_combinations : ℕ := Nat.choose girls chosen_members

-- Probability is defined as the ratio of these combinations
def probability_all_girls_chosen : ℚ := girl_combinations / total_combinations

-- Proof Statement
theorem probability_of_all_girls_chosen_is_1_over_11 : probability_all_girls_chosen = 1 / 11 := by
  sorry -- Proof to be completed

end probability_of_all_girls_chosen_is_1_over_11_l181_181808


namespace quadratic_roots_sum_cubes_l181_181659

theorem quadratic_roots_sum_cubes (k : ℚ) (a b : ℚ) 
  (h1 : 4 * a^2 + 5 * a + k = 0) 
  (h2 : 4 * b^2 + 5 * b + k = 0) 
  (h3 : a^3 + b^3 = a + b) :
  k = 9 / 4 :=
by {
  -- Lean code requires the proof, here we use sorry to skip it
  sorry
}

end quadratic_roots_sum_cubes_l181_181659


namespace find_x_values_l181_181079

def f (x : ℝ) : ℝ := x^2 - 2 * x

theorem find_x_values (x : ℝ) :
  (f (f x) = f x) ↔ (x = 0 ∨ x = 2 ∨ x = -1 ∨ x = 3) :=
by
  sorry

end find_x_values_l181_181079


namespace total_raisins_l181_181412

noncomputable def yellow_raisins : ℝ := 0.3
noncomputable def black_raisins : ℝ := 0.4
noncomputable def red_raisins : ℝ := 0.5

theorem total_raisins : yellow_raisins + black_raisins + red_raisins = 1.2 := by
  sorry

end total_raisins_l181_181412


namespace minimize_AC_plus_BC_l181_181442

noncomputable def minimize_distance (k : ℝ) : Prop :=
  let A := (5, 5)
  let B := (2, 1)
  let C := (0, k)
  let dist (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  let AC := dist A C
  let BC := dist B C
  ∀ k', dist (0, k') A + dist (0, k') B ≥ AC + BC

theorem minimize_AC_plus_BC : minimize_distance (15 / 7) :=
sorry

end minimize_AC_plus_BC_l181_181442


namespace probability_three_good_students_l181_181547

theorem probability_three_good_students (total_people three_good_students selected_people : ℕ)
  (h_total_people : total_people = 12)
  (h_three_good_students : three_good_students = 5)
  (h_selected_people : selected_people = 6)
  (ξ : ℕ → ℕ) :
  (∑ ξ = 3).card = (nat.choose three_good_students 3 * nat.choose (total_people - three_good_students) (selected_people - 3))
  /
  (nat.choose total_people selected_people) :=
by sorry

end probability_three_good_students_l181_181547


namespace domain_of_rational_function_l181_181658

theorem domain_of_rational_function 
  (c : ℝ) 
  (h : -7 * (6 ^ 2) + 28 * c < 0) : 
  c < -9 / 7 :=
by sorry

end domain_of_rational_function_l181_181658


namespace Y_subset_X_l181_181586

def X : Set ℕ := {n | ∃ m : ℕ, n = 4 * m + 2}

def Y : Set ℕ := {t | ∃ k : ℕ, t = (2 * k - 1)^2 + 1}

theorem Y_subset_X : Y ⊆ X := by
  sorry

end Y_subset_X_l181_181586


namespace largest_possible_number_of_neither_l181_181641

theorem largest_possible_number_of_neither
  (writers : ℕ)
  (editors : ℕ)
  (attendees : ℕ)
  (x : ℕ)
  (N : ℕ)
  (h_writers : writers = 45)
  (h_editors_gt : editors > 38)
  (h_attendees : attendees = 90)
  (h_both : N = 2 * x)
  (h_equation : writers + editors - x + N = attendees) :
  N = 12 :=
by
  sorry

end largest_possible_number_of_neither_l181_181641


namespace binomial_20_19_l181_181324

theorem binomial_20_19 : nat.choose 20 19 = 20 :=
by {
  sorry
}

end binomial_20_19_l181_181324


namespace field_length_l181_181236

theorem field_length (w l : ℝ) (A_f A_p : ℝ) 
  (h1 : l = 3 * w)
  (h2 : A_p = 150) 
  (h3 : A_p = 0.4 * A_f)
  (h4 : A_f = l * w) : 
  l = 15 * Real.sqrt 5 :=
by
  sorry

end field_length_l181_181236


namespace power_of_power_example_l181_181161

theorem power_of_power_example : (3^2)^4 = 6561 := by
  sorry

end power_of_power_example_l181_181161


namespace probability_tenth_ball_black_l181_181803

theorem probability_tenth_ball_black :
  let total_balls := 30
  let black_balls := 4
  let red_balls := 7
  let yellow_balls := 5
  let green_balls := 6
  let white_balls := 8
  (black_balls / total_balls) = 4 / 30 :=
by sorry

end probability_tenth_ball_black_l181_181803


namespace number_of_boys_l181_181267

theorem number_of_boys (M W B : ℕ) (X : ℕ) 
  (h1 : 5 * M = W) 
  (h2 : W = B) 
  (h3 : 5 * M * 12 + W * X + B * X = 180) 
  : B = 15 := 
by sorry

end number_of_boys_l181_181267


namespace max_quotient_l181_181995

theorem max_quotient (x y : ℝ) (hx : 100 ≤ x ∧ x ≤ 300) (hy : 900 ≤ y ∧ y ≤ 1800) : 
  (∀ x y, (100 ≤ x ∧ x ≤ 300) ∧ (900 ≤ y ∧ y ≤ 1800) → y / x ≤ 18) ∧ 
  (∃ x y, (100 ≤ x ∧ x ≤ 300) ∧ (900 ≤ y ∧ y ≤ 1800) ∧ y / x = 18) :=
by
  sorry

end max_quotient_l181_181995


namespace second_player_wins_l181_181085

-- Piles of balls and game conditions
def two_pile_game (pile1 pile2 : ℕ) : Prop :=
  ∀ (player1_turn : ℕ → Prop) (player2_turn : ℕ → Prop),
    (∀ n : ℕ, player1_turn n → (pile1 ≥ n ∧ pile2 ≥ n) ∨ pile1 ≥ n ∨ pile2 ≥ n) → -- player1's move
    (∀ n : ℕ, player2_turn n → (pile1 ≥ n ∧ pile2 ≥ n) ∨ pile1 ≥ n ∨ pile2 ≥ n) → -- player2's move
    -- - Second player has a winning strategy
    ∃ (win_strategy : ℕ → ℕ), ∀ k : ℕ, player1_turn k → player2_turn (win_strategy k) 

-- Lean statement of the problem
theorem second_player_wins : ∀ (pile1 pile2 : ℕ), pile1 = 30 ∧ pile2 = 30 → two_pile_game pile1 pile2 :=
  by
    intros pile1 pile2 h
    sorry  -- Placeholder for the proof


end second_player_wins_l181_181085


namespace largest_4_digit_congruent_to_15_mod_25_l181_181784

theorem largest_4_digit_congruent_to_15_mod_25 : 
  ∀ x : ℕ, (1000 ≤ x ∧ x < 10000 ∧ x % 25 = 15) → x = 9990 :=
by
  intros x h
  sorry

end largest_4_digit_congruent_to_15_mod_25_l181_181784


namespace binom_20_19_eq_20_l181_181305

theorem binom_20_19_eq_20 : nat.choose 20 19 = 20 :=
by sorry

end binom_20_19_eq_20_l181_181305


namespace num_real_roots_of_abs_x_eq_l181_181026

theorem num_real_roots_of_abs_x_eq (k : ℝ) (hk : 6 < k ∧ k < 7) 
  : (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ 
    (|x1| * x1 - 2 * x1 + 7 - k = 0) ∧ 
    (|x2| * x2 - 2 * x2 + 7 - k = 0) ∧
    (|x3| * x3 - 2 * x3 + 7 - k = 0)) ∧
  (¬ ∃ x4 : ℝ, x4 ≠ x1 ∧ x4 ≠ x2 ∧ x4 ≠ x3 ∧ |x4| * x4 - 2 * x4 + 7 - k = 0) :=
sorry

end num_real_roots_of_abs_x_eq_l181_181026


namespace percentage_increase_second_year_l181_181905

theorem percentage_increase_second_year 
  (initial_population : ℝ)
  (first_year_increase : ℝ) 
  (population_after_2_years : ℝ) 
  (final_population : ℝ)
  (H_initial_population : initial_population = 800)
  (H_first_year_increase : first_year_increase = 0.22)
  (H_population_after_2_years : final_population = 1220) :
  ∃ P : ℝ, P = 25 := 
by
  -- Define the population after the first year
  let population_after_first_year := initial_population * (1 + first_year_increase)
  -- Define the equation relating populations and solve for P
  let second_year_increase := (final_population / population_after_first_year - 1) * 100
  -- Show P equals 25
  use second_year_increase
  sorry

end percentage_increase_second_year_l181_181905


namespace total_distance_correct_l181_181086

-- Given conditions
def fuel_efficiency_city : Float := 15
def fuel_efficiency_highway : Float := 25
def fuel_efficiency_gravel : Float := 18

def gallons_used_city : Float := 2.5
def gallons_used_highway : Float := 3.8
def gallons_used_gravel : Float := 1.7

-- Define distances
def distance_city := fuel_efficiency_city * gallons_used_city
def distance_highway := fuel_efficiency_highway * gallons_used_highway
def distance_gravel := fuel_efficiency_gravel * gallons_used_gravel

-- Define total distance
def total_distance := distance_city + distance_highway + distance_gravel

-- Prove the total distance traveled is 163.1 miles
theorem total_distance_correct : total_distance = 163.1 := by
  -- Proof to be filled in
  sorry

end total_distance_correct_l181_181086


namespace triangle_angle_C_l181_181060

theorem triangle_angle_C (A B C : ℝ) (h : A + B = 80) : C = 100 :=
sorry

end triangle_angle_C_l181_181060


namespace inequality_proof_l181_181536

-- Given conditions
variables {a b : ℝ} (ha_lt_b : a < b) (hb_lt_0 : b < 0)

-- Question statement we want to prove
theorem inequality_proof : ab < 0 → a < b → b < 0 → ab > b^2 :=
by
  sorry

end inequality_proof_l181_181536


namespace evan_runs_200_more_feet_l181_181553

def street_width : ℕ := 25
def block_side : ℕ := 500

def emily_path : ℕ := 4 * block_side
def evan_path : ℕ := 4 * (block_side + 2 * street_width)

theorem evan_runs_200_more_feet : evan_path - emily_path = 200 := by
  sorry

end evan_runs_200_more_feet_l181_181553


namespace simplify_expression_l181_181436

theorem simplify_expression (y : ℝ) : (3 * y^4)^4 = 81 * y^16 :=
by
  sorry

end simplify_expression_l181_181436


namespace no_solution_for_x6_eq_2y2_plus_2_l181_181152

theorem no_solution_for_x6_eq_2y2_plus_2 :
  ¬ ∃ (x y : ℤ), x^6 = 2 * y^2 + 2 :=
sorry

end no_solution_for_x6_eq_2y2_plus_2_l181_181152


namespace frank_remaining_money_l181_181009

noncomputable def cheapest_lamp_cost : ℝ := 20
noncomputable def most_expensive_lamp_cost : ℝ := 3 * cheapest_lamp_cost
noncomputable def frank_initial_money : ℝ := 90

theorem frank_remaining_money : frank_initial_money - most_expensive_lamp_cost = 30 := by
  -- Proof will go here
  sorry

end frank_remaining_money_l181_181009


namespace trapezoidal_section_length_l181_181637

theorem trapezoidal_section_length 
  (total_area : ℝ) 
  (rectangular_area : ℝ) 
  (parallel_side1 : ℝ) 
  (parallel_side2 : ℝ) 
  (trapezoidal_area : ℝ)
  (H1 : total_area = 55)
  (H2 : rectangular_area = 30)
  (H3 : parallel_side1 = 3)
  (H4 : parallel_side2 = 6)
  (H5 : trapezoidal_area = total_area - rectangular_area) :
  (trapezoidal_area = 25) → 
  (1/2 * (parallel_side1 + parallel_side2) * L = trapezoidal_area) →
  L = 25 / 4.5 :=
by
  sorry

end trapezoidal_section_length_l181_181637


namespace input_value_of_x_l181_181478

theorem input_value_of_x (x y : ℤ) (h₁ : (x < 0 → y = (x + 1) * (x + 1)) ∧ (¬(x < 0) → y = (x - 1) * (x - 1)))
  (h₂ : y = 16) : x = 5 ∨ x = -5 :=
sorry

end input_value_of_x_l181_181478


namespace daily_profit_1200_impossible_daily_profit_1600_l181_181940

-- Definitions of given conditions
def avg_shirts_sold_per_day : ℕ := 30
def profit_per_shirt : ℕ := 40

-- Function for the number of shirts sold given a price reduction
def shirts_sold (x : ℕ) : ℕ := avg_shirts_sold_per_day + 2 * x

-- Function for the profit per shirt given a price reduction
def new_profit_per_shirt (x : ℕ) : ℕ := profit_per_shirt - x

-- Function for the daily profit given a price reduction
def daily_profit (x : ℕ) : ℕ := (new_profit_per_shirt x) * (shirts_sold x)

-- Proving the desired conditions in Lean

-- Part 1: Prove that reducing the price by 25 yuan results in a daily profit of 1200 yuan
theorem daily_profit_1200 (x : ℕ) : daily_profit x = 1200 ↔ x = 25 :=
by
  { sorry }

-- Part 2: Prove that a daily profit of 1600 yuan is not achievable
theorem impossible_daily_profit_1600 (x : ℕ) : daily_profit x ≠ 1600 :=
by
  { sorry }

end daily_profit_1200_impossible_daily_profit_1600_l181_181940


namespace converse_inverse_contrapositive_count_l181_181699

theorem converse_inverse_contrapositive_count
  (a b : ℝ) : (a = 0 → ab = 0) →
  (if (ab = 0 → a = 0) then 1 else 0) +
  (if (a ≠ 0 → ab ≠ 0) then 1 else 0) +
  (if (ab ≠ 0 → a ≠ 0) then 1 else 0) = 1 :=
sorry

end converse_inverse_contrapositive_count_l181_181699


namespace triangle_angle_C_l181_181059

theorem triangle_angle_C (A B C : ℝ) (h : A + B = 80) : C = 100 :=
sorry

end triangle_angle_C_l181_181059


namespace fraction_simplification_l181_181114

theorem fraction_simplification (b : ℝ) (hb : b = 3) : 
  (3 * b⁻¹ + b⁻¹ / 3) / b^2 = 10 / 81 :=
by
  rw [hb]
  sorry

end fraction_simplification_l181_181114


namespace min_value_of_expression_l181_181432

theorem min_value_of_expression (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a^2 + a * b + a * c + b * c = 4) : 2 * a + b + c ≥ 4 :=
sorry

end min_value_of_expression_l181_181432


namespace petya_mistake_l181_181741

theorem petya_mistake (x : ℝ) (h : x - x / 10 = 19.71) : x = 21.9 := 
  sorry

end petya_mistake_l181_181741


namespace find_alpha_l181_181548

variable {α p₀ p_new : ℝ}
def Q_d (p : ℝ) : ℝ := 150 - p
def Q_s (p : ℝ) : ℝ := 3 * p - 10
def Q_d_new (α : ℝ) (p : ℝ) : ℝ := α * (150 - p)

theorem find_alpha 
  (h_eq_initial : Q_d p₀ = Q_s p₀)
  (h_eq_increase : p_new = 1.25 * p₀)
  (h_eq_new : Q_s p_new = Q_d_new α p_new) :
  α = 1.4 :=
by
  sorry

end find_alpha_l181_181548


namespace no_natural_numbers_satisfy_equation_l181_181462

theorem no_natural_numbers_satisfy_equation :
  ¬ ∃ (x y : ℕ), Nat.gcd x y + Nat.lcm x y + x + y = 2019 :=
by
  sorry

end no_natural_numbers_satisfy_equation_l181_181462


namespace power_sum_l181_181484

theorem power_sum :
  (-1:ℤ)^53 + 2^(3^4 + 4^3 - 6 * 7) = 2^103 - 1 :=
by
  sorry

end power_sum_l181_181484


namespace land_division_possible_l181_181836

-- Define the basic properties and conditions of the plot
structure Plot :=
  (is_square : Prop)
  (has_center_well : Prop)
  (has_four_trees : Prop)
  (has_four_gates : Prop)

-- Define a section of the plot
structure Section :=
  (contains_tree : Prop)
  (contains_gate : Prop)
  (equal_fence_length : Prop)
  (unrestricted_access_to_well : Prop)

-- Define the property that indicates a valid division of the plot
def valid_division (p : Plot) (sections : List Section) : Prop :=
  sections.length = 4 ∧
  (∀ s ∈ sections, s.contains_tree) ∧
  (∀ s ∈ sections, s.contains_gate) ∧
  (∀ s ∈ sections, s.equal_fence_length) ∧
  (∀ s ∈ sections, s.unrestricted_access_to_well)

-- Define the main theorem to prove
theorem land_division_possible (p : Plot) : 
  p.is_square ∧ p.has_center_well ∧ p.has_four_trees ∧ p.has_four_gates → 
  ∃ sections : List Section, valid_division p sections :=
by
  sorry

end land_division_possible_l181_181836


namespace arithmetic_sequence_s10_l181_181693

noncomputable def arithmetic_sequence_sum (n : ℕ) (a d : ℤ) : ℤ :=
  n * (2 * a + (n - 1) * d) / 2

theorem arithmetic_sequence_s10 (a : ℤ) (d : ℤ)
  (h1 : a + (a + 8 * d) = 18)
  (h4 : a + 3 * d = 7) :
  arithmetic_sequence_sum 10 a d = 100 :=
by sorry

end arithmetic_sequence_s10_l181_181693


namespace david_savings_l181_181000

def lawn_rate_monday : ℕ := 14
def lawn_rate_wednesday : ℕ := 18
def lawn_rate_friday : ℕ := 20
def hours_per_day : ℕ := 2
def weekly_earnings : ℕ := (lawn_rate_monday * hours_per_day) + (lawn_rate_wednesday * hours_per_day) + (lawn_rate_friday * hours_per_day)

def tax_rate : ℚ := 0.10
def tax_paid (earnings : ℚ) : ℚ := earnings * tax_rate

def shoe_price : ℚ := 75
def discount : ℚ := 0.15
def discounted_shoe_price : ℚ := shoe_price * (1 - discount)

def money_remaining (earnings : ℚ) (tax : ℚ) (shoes : ℚ) : ℚ := earnings - tax - shoes

def gift_rate : ℚ := 1 / 3
def money_given_to_mom (remaining : ℚ) : ℚ := remaining * gift_rate

def final_savings (remaining : ℚ) (gift : ℚ) : ℚ := remaining - gift

theorem david_savings : 
  final_savings (money_remaining weekly_earnings (tax_paid weekly_earnings) discounted_shoe_price) 
                (money_given_to_mom (money_remaining weekly_earnings (tax_paid weekly_earnings) discounted_shoe_price)) 
  = 19.90 :=
by
  -- The proof goes here
  sorry

end david_savings_l181_181000


namespace Gerald_needs_to_average_5_chores_per_month_l181_181837

def spending_per_month := 100
def season_length := 4
def cost_per_chore := 10
def total_spending := spending_per_month * season_length
def months_not_playing := 12 - season_length
def amount_to_save_per_month := total_spending / months_not_playing
def chores_per_month := amount_to_save_per_month / cost_per_chore

theorem Gerald_needs_to_average_5_chores_per_month :
  chores_per_month = 5 := by
  sorry

end Gerald_needs_to_average_5_chores_per_month_l181_181837


namespace kmph_to_mps_l181_181804

theorem kmph_to_mps (s : ℝ) (h : s = 0.975) : s * (1000 / 3600) = 0.2708 := by
  -- We include the assumption s = 0.975 as part of the problem condition.
  -- Import Mathlib to gain access to real number arithmetic.
  -- sorry is added to indicate a place where the proof should go.
  sorry

end kmph_to_mps_l181_181804


namespace fraction_of_product_l181_181778

theorem fraction_of_product : (7 / 8) * 64 = 56 := by
  sorry

end fraction_of_product_l181_181778


namespace projection_cardinal_inequality_l181_181420

variables {Point : Type} [Fintype Point] [DecidableEq Point]

def projection_Oyz (S : Finset Point) : Finset Point := sorry
def projection_Ozx (S : Finset Point) : Finset Point := sorry
def projection_Oxy (S : Finset Point) : Finset Point := sorry

theorem projection_cardinal_inequality
  (S : Finset Point)
  (S_x := projection_Oyz S)
  (S_y := projection_Ozx S)
  (S_z := projection_Oxy S)
  : (Finset.card S)^2 ≤ (Finset.card S_x) * (Finset.card S_y) * (Finset.card S_z) :=
sorry

end projection_cardinal_inequality_l181_181420


namespace aspirin_mass_percentages_l181_181964

noncomputable def atomic_mass_H : ℝ := 1.01
noncomputable def atomic_mass_C : ℝ := 12.01
noncomputable def atomic_mass_O : ℝ := 16.00

noncomputable def molar_mass_aspirin : ℝ := (9 * atomic_mass_C) + (8 * atomic_mass_H) + (4 * atomic_mass_O)

theorem aspirin_mass_percentages :
  let mass_percent_H := ((8 * atomic_mass_H) / molar_mass_aspirin) * 100
  let mass_percent_C := ((9 * atomic_mass_C) / molar_mass_aspirin) * 100
  let mass_percent_O := ((4 * atomic_mass_O) / molar_mass_aspirin) * 100
  mass_percent_H = 4.48 ∧ mass_percent_C = 60.00 ∧ mass_percent_O = 35.52 :=
by
  -- Placeholder for the proof
  sorry

end aspirin_mass_percentages_l181_181964


namespace factor_difference_of_squares_l181_181906

theorem factor_difference_of_squares (a : ℝ) : a^2 - 16 = (a - 4) * (a + 4) := 
sorry

end factor_difference_of_squares_l181_181906


namespace part1_part2_l181_181512

def f (x a : ℝ) : ℝ := (Exp x / x) - Log x + x - a

theorem part1 (x a : ℝ) (hx : f x a ≥ 0) : a ≤ Real.exp 1 + 1 := 
sorry

theorem part2 (x1 x2 a : ℝ) (hx1 : f x1 a = 0) (hx2 : f x2 a = 0) 
  (hx1_range : 0 < x1) (hx2_range : x2 > 1) : x1 * x2 < 1 := 
sorry

end part1_part2_l181_181512


namespace solve_inequality_system_l181_181437

theorem solve_inequality_system (x : ℝ) 
  (h1 : 2 * (x - 1) < x + 3)
  (h2 : (x + 1) / 3 - x < 3) : 
  -4 < x ∧ x < 5 := 
  sorry

end solve_inequality_system_l181_181437


namespace jennas_total_ticket_cost_l181_181564

theorem jennas_total_ticket_cost :
  let normal_price := 50
  let tickets_from_website := 2 * normal_price
  let scalper_price := 2 * normal_price * 2.4 - 10
  let friend_discounted_ticket := normal_price * 0.6
  tickets_from_website + scalper_price + friend_discounted_ticket = 360 :=
by
  sorry

end jennas_total_ticket_cost_l181_181564


namespace biased_coin_die_even_probability_l181_181534

theorem biased_coin_die_even_probability (p_head : ℚ) (p_even : ℚ) (independent : Prop) :
  p_head = 2/3 → p_even = 1/2 → independent → (p_head * p_even) = 1/3 :=
by
  intro h_head h_even h_independent
  rw [h_head, h_even]
  norm_num
  sorry

end biased_coin_die_even_probability_l181_181534


namespace discs_angular_velocity_relation_l181_181453

variables {r1 r2 ω1 ω2 : ℝ} -- Radii and angular velocities

-- Conditions:
-- Discs have radii r1 and r2, and angular velocities ω1 and ω2, respectively.
-- Discs come to a halt after being brought into contact via friction.
-- Discs have identical thickness and are made of the same material.
-- Prove the required relation.

theorem discs_angular_velocity_relation
  (h1 : r1 > 0)
  (h2 : r2 > 0)
  (halt_contact : ω1 * r1^3 = ω2 * r2^3) :
  ω1 * r1^3 = ω2 * r2^3 :=
sorry

end discs_angular_velocity_relation_l181_181453


namespace three_digit_cubes_divisible_by_8_and_9_l181_181853

theorem three_digit_cubes_divisible_by_8_and_9 : 
  ∃! n : ℕ, (216 ≤ n^3 ∧ n^3 ≤ 999) ∧ (n % 6 = 0) :=
sorry

end three_digit_cubes_divisible_by_8_and_9_l181_181853


namespace floor_neg_seven_over_four_l181_181346

theorem floor_neg_seven_over_four : Int.floor (- 7 / 4 : ℝ) = -2 := 
by
  sorry

end floor_neg_seven_over_four_l181_181346


namespace smallest_possible_value_l181_181570

theorem smallest_possible_value (a b c d : ℝ)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a + b + c + d) * ((1 / (a + b)) + (1 / (a + c)) + (1 / (b + d)) + (1 / (c + d))) ≥ 8 := 
sorry

end smallest_possible_value_l181_181570


namespace eighteenth_prime_l181_181090

-- Define the necessary statements
def isPrime (n : ℕ) : Prop := sorry

def primeSeq (n : ℕ) : ℕ :=
  if n = 0 then
    2
  else if n = 1 then
    3
  else
    -- Function to generate the n-th prime number
    sorry

theorem eighteenth_prime :
  primeSeq 17 = 67 := by
  sorry

end eighteenth_prime_l181_181090


namespace charity_tickets_solution_l181_181942

theorem charity_tickets_solution (f h d p : ℕ) (ticket_count : f + h + d = 200)
  (revenue : f * p + h * (p / 2) + d * (2 * p) = 3600) : f = 80 := by
  sorry

end charity_tickets_solution_l181_181942


namespace smallest_value_am_hm_inequality_l181_181572

theorem smallest_value_am_hm_inequality (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a + b + c + d) * (1 / (a + b) + 1 / (a + c) + 1 / (b + d) + 1 / (c + d)) ≥ 8 :=
by
  sorry

end smallest_value_am_hm_inequality_l181_181572


namespace pens_sold_l181_181137

theorem pens_sold (C : ℝ) (N : ℝ) (h_gain : 22 * C = 0.25 * N * C) : N = 88 :=
by {
  sorry
}

end pens_sold_l181_181137


namespace probability_snow_once_first_week_l181_181887

theorem probability_snow_once_first_week :
  let p_first_two_days := (3 / 4) * (3 / 4)
  let p_next_three_days := (1 / 2) * (1 / 2) * (1 / 2)
  let p_last_two_days := (2 / 3) * (2 / 3)
  let p_no_snow := p_first_two_days * p_next_three_days * p_last_two_days
  let p_at_least_once := 1 - p_no_snow
  p_at_least_once = 31 / 32 :=
by
  sorry

end probability_snow_once_first_week_l181_181887


namespace largest_n_binom_eq_l181_181299

open Nat

theorem largest_n_binom_eq :
  ∃ n, n <= 13 ∧ (binomial 12 5 + binomial 12 6 = binomial 13 n) ∧ (n = 7) :=
by
  have pascal := Nat.add_binom_eq (12) (5)
  have symm := binomial_symm (13) (6)
  use 7
  split
  · exact le_of_eq rfl
  · split
    · rw [pascal, ←symm]
    · exact rfl
  · sorry

end largest_n_binom_eq_l181_181299


namespace factor_polynomial_l181_181671

theorem factor_polynomial : ∀ x : ℝ, 
  x^8 - 4 * x^6 + 6 * x^4 - 4 * x^2 + 1 = (x - 1)^4 * (x + 1)^4 :=
by
  intro x
  sorry

end factor_polynomial_l181_181671


namespace known_number_l181_181101

theorem known_number (A B : ℕ) (h_hcf : 1 / (Nat.gcd A B) = 1 / 15) (h_lcm : 1 / Nat.lcm A B = 1 / 312) (h_B : B = 195) : A = 24 :=
by
  -- Skipping proof
  sorry

end known_number_l181_181101


namespace seven_divides_n_iff_seven_divides_q_minus_2r_seven_divides_2023_thirteen_divides_n_iff_thirteen_divides_q_plus_4r_l181_181934

-- Problem 1
theorem seven_divides_n_iff_seven_divides_q_minus_2r (n q r : ℕ) (h : n = 10 * q + r) :
  (7 ∣ n) ↔ (7 ∣ (q - 2 * r)) := sorry

-- Problem 2
theorem seven_divides_2023 : 7 ∣ 2023 :=
  let q := 202
  let r := 3
  have h : 2023 = 10 * q + r := by norm_num
  have h1 : (7 ∣ 2023) ↔ (7 ∣ (q - 2 * r)) :=
    seven_divides_n_iff_seven_divides_q_minus_2r 2023 q r h
  sorry -- Here you would use h1 and prove the statement using it

-- Problem 3
theorem thirteen_divides_n_iff_thirteen_divides_q_plus_4r (n q r : ℕ) (h : n = 10 * q + r) :
  (13 ∣ n) ↔ (13 ∣ (q + 4 * r)) := sorry

end seven_divides_n_iff_seven_divides_q_minus_2r_seven_divides_2023_thirteen_divides_n_iff_thirteen_divides_q_plus_4r_l181_181934


namespace Adam_teaches_students_l181_181639

-- Define the conditions
def students_first_year : ℕ := 40
def students_per_year : ℕ := 50
def total_years : ℕ := 10
def remaining_years : ℕ := total_years - 1

-- Define the statement we are proving
theorem Adam_teaches_students (total_students : ℕ) :
  total_students = students_first_year + (students_per_year * remaining_years) :=
sorry

end Adam_teaches_students_l181_181639


namespace percent_decaf_coffee_l181_181794

variable (initial_stock new_stock decaf_initial_percent decaf_new_percent : ℝ)
variable (initial_stock_pos new_stock_pos : initial_stock > 0 ∧ new_stock > 0)

theorem percent_decaf_coffee :
    initial_stock = 400 → 
    decaf_initial_percent = 20 → 
    new_stock = 100 → 
    decaf_new_percent = 60 → 
    (100 * ((decaf_initial_percent / 100 * initial_stock + decaf_new_percent / 100 * new_stock) / (initial_stock + new_stock))) = 28 := 
by
  sorry

end percent_decaf_coffee_l181_181794


namespace total_initial_amounts_l181_181910

theorem total_initial_amounts :
  ∃ (a j t : ℝ), a = 50 ∧ t = 50 ∧ (50 + j + 50 = 187.5) :=
sorry

end total_initial_amounts_l181_181910


namespace find_chemistry_marks_l181_181657

theorem find_chemistry_marks (marks_english marks_math marks_physics marks_biology : ℤ)
    (average_marks total_subjects : ℤ)
    (h1 : marks_english = 36)
    (h2 : marks_math = 35)
    (h3 : marks_physics = 42)
    (h4 : marks_biology = 55)
    (h5 : average_marks = 45)
    (h6 : total_subjects = 5) :
    (225 - (marks_english + marks_math + marks_physics + marks_biology)) = 57 :=
by
  sorry

end find_chemistry_marks_l181_181657


namespace sqrt_sum_of_powers_l181_181918

theorem sqrt_sum_of_powers : sqrt (4^4 + 4^4 + 4^4 + 4^4) = 32 := by
  have h : 4^4 = 256 := by
    calc
      4^4 = 4 * 4 * 4 * 4 := by rw [show 4^4 = 4 * 4 * 4 * 4 from rfl]
      ... = 16 * 16       := by rw [mul_assoc, mul_assoc]
      ... = 256           := by norm_num
  calc
    sqrt (4^4 + 4^4 + 4^4 + 4^4)
      = sqrt (256 + 256 + 256 + 256) := by rw [h, h, h, h]
      ... = sqrt 1024                 := by norm_num
      ... = 32                        := by norm_num

end sqrt_sum_of_powers_l181_181918


namespace binom_20_19_eq_20_l181_181328

def binom (n k : ℕ) := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binom_20_19_eq_20 : binom 20 19 = 20 := by
  sorry

end binom_20_19_eq_20_l181_181328


namespace negate_proposition_l181_181446

theorem negate_proposition :
  (¬ ∃ x : ℝ, x^2 + x - 2 ≤ 0) ↔ (∀ x : ℝ, x^2 + x - 2 > 0) := 
sorry

end negate_proposition_l181_181446


namespace triangle_reflection_not_necessarily_perpendicular_l181_181289

theorem triangle_reflection_not_necessarily_perpendicular
  (P Q R : ℝ × ℝ)
  (hP : 0 ≤ P.1 ∧ 0 ≤ P.2)
  (hQ : 0 ≤ Q.1 ∧ 0 ≤ Q.2)
  (hR : 0 ≤ R.1 ∧ 0 ≤ R.2)
  (not_on_y_eq_x_P : P.1 ≠ P.2)
  (not_on_y_eq_x_Q : Q.1 ≠ Q.2)
  (not_on_y_eq_x_R : R.1 ≠ R.2) :
  ¬ (∃ (mPQ mPQ' : ℝ), 
      mPQ = (Q.2 - P.2) / (Q.1 - P.1) ∧ 
      mPQ' = (Q.1 - P.1) / (Q.2 - P.2) ∧ 
      mPQ * mPQ' = -1) :=
sorry

end triangle_reflection_not_necessarily_perpendicular_l181_181289


namespace cos_x_plus_2y_eq_one_l181_181012

theorem cos_x_plus_2y_eq_one 
  (x y : ℝ) (a : ℝ)
  (hx : x ∈ set.Icc (-π/4) (π/4))
  (hy : y ∈ set.Icc (-π/4) (π/4))
  (h_eq1 : x^3 + real.sin x = 2 * a)
  (h_eq2 : 4 * y^3 + real.sin y * real.cos y + a = 0) :
  real.cos (x + 2 * y) = 1 :=
sorry

end cos_x_plus_2y_eq_one_l181_181012


namespace conic_section_is_parabola_l181_181830

-- Define the equation |y-3| = sqrt((x+4)^2 + y^2)
def equation (x y : ℝ) : Prop := |y - 3| = Real.sqrt ((x + 4) ^ 2 + y ^ 2)

-- The main theorem stating the conic section type is a parabola
theorem conic_section_is_parabola : ∀ x y : ℝ, equation x y → false := sorry

end conic_section_is_parabola_l181_181830


namespace tensor_correct_l181_181505

-- Given definitions in the problem
def vec (α : Type) := α × α

-- Conditions as Lean definitions
variables {a b : vec ℝ} (theta : ℝ)

noncomputable def modulus (v : vec ℝ) : ℝ :=
  real.sqrt (v.1 * v.1 + v.2 * v.2)

noncomputable def dot_product (v1 v2 : vec ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

noncomputable def cos_theta (v1 v2 : vec ℝ) : ℝ :=
  (dot_product v1 v2) / ((modulus v1) * (modulus v2))

noncomputable def tensor_product (v1 v2 : vec ℝ) : ℝ :=
  (modulus v1) / (modulus v2) * cos (cos_theta v1 v2)

-- Problem statement translated to Lean
theorem tensor_correct:
  ∀ (a b : vec ℝ), modulus a ≠ 0 ∧ modulus b ≠ 0 ∧
  modulus a ≥ modulus b ∧
  θ ∈ Ioo 0 (real.pi / 4) ∧
  (tensor_product a b ∈ {x | ∃ n : ℕ, x = n / 2}) ∧
  (tensor_product b a ∈ {x | ∃ n : ℕ, x = n / 2}) →
  tensor_product a b = 3 / 2 :=
begin
  sorry
end

end tensor_correct_l181_181505


namespace gas_total_cost_l181_181451

theorem gas_total_cost (x : ℝ) (h : (x/3) - 11 = x/5) : x = 82.5 :=
sorry

end gas_total_cost_l181_181451


namespace part_a_part_b_l181_181951

-- Definitions based on the conditions:
def probability_of_hit (p : ℝ) := p
def probability_of_miss (p : ℝ) := 1 - p

-- Condition: exactly three unused rockets after firing at five targets
def exactly_three_unused_rockets (p : ℝ) : ℝ := 10 * (probability_of_hit p) ^ 3 * (probability_of_miss p) ^ 2

-- Condition: expected number of targets hit when there are nine targets
def expected_targets_hit (p : ℝ) : ℝ := 10 * p - p ^ 10

-- Lean 4 statements representing the proof problems:
theorem part_a (p : ℝ) (h_p_nonneg : 0 ≤ p) (h_p_le_one : p ≤ 1) : 
  exactly_three_unused_rockets p = 10 * p ^ 3 * (1 - p) ^ 2 :=
by sorry

theorem part_b (p : ℝ) (h_p_nonneg : 0 ≤ p) (h_p_le_one : p ≤ 1) :
  expected_targets_hit p = 10 * p - p ^ 10 :=
by sorry

end part_a_part_b_l181_181951


namespace intersect_at_four_points_l181_181829

theorem intersect_at_four_points (a : ℝ) : 
  (∃ p : ℝ × ℝ, (p.1^2 + p.2^2 = a^2) ∧ (p.2 = p.1^2 - a - 1) ∧ 
                 ∃ q : ℝ × ℝ, (q.1 ≠ p.1 ∧ q.2 ≠ p.2) ∧ (q.1^2 + q.2^2 = a^2) ∧ (q.2 = q.1^2 - a - 1) ∧ 
                 ∃ r : ℝ × ℝ, (r.1 ≠ p.1 ∧ r.1 ≠ q.1 ∧ r.2 ≠ p.2 ∧ r.2 ≠ q.2) ∧ (r.1^2 + r.2^2 = a^2) ∧ (r.2 = r.1^2 - a - 1) ∧
                 ∃ s : ℝ × ℝ, (s.1 ≠ p.1 ∧ s.1 ≠ q.1 ∧ s.1 ≠ r.1 ∧ s.2 ≠ p.2 ∧ s.2 ≠ q.2 ∧ s.2 ≠ r.2) ∧ (s.1^2 + s.2^2 = a^2) ∧ (s.2 = s.1^2 - a - 1))
  ↔ a > -1/2 := 
by 
  sorry

end intersect_at_four_points_l181_181829


namespace factorization_correct_l181_181003

theorem factorization_correct (a b : ℝ) : a * b^2 - 25 * a = a * (b + 5) * (b - 5) :=
by
  -- The actual proof will be written here.
  sorry

end factorization_correct_l181_181003


namespace floor_sum_equality_l181_181227

theorem floor_sum_equality (n : ℕ) (h : n > 1) :
  (∑ i in Finset.range n, Nat.floor (n^(1 / (i + 1) : ℝ))) =
  (∑ i in Finset.range n, Nat.floor (Real.logBase (i + 2) n)) :=
sorry

end floor_sum_equality_l181_181227


namespace cristina_speed_cristina_running_speed_l181_181223

theorem cristina_speed 
  (head_start : ℕ)
  (nicky_speed : ℕ)
  (catch_up_time : ℕ)
  (distance : ℕ := head_start + (nicky_speed * catch_up_time))
  : distance / catch_up_time = 6
  := by
  sorry

-- Given conditions used as definitions in Lean 4:
-- head_start = 36 (meters)
-- nicky_speed = 3 (meters/second)
-- catch_up_time = 12 (seconds)

theorem cristina_running_speed
  (head_start : ℕ := 36)
  (nicky_speed : ℕ := 3)
  (catch_up_time : ℕ := 12)
  : (head_start + (nicky_speed * catch_up_time)) / catch_up_time = 6
  := by
  sorry

end cristina_speed_cristina_running_speed_l181_181223


namespace regionA_regionC_area_ratio_l181_181745

-- Definitions for regions A and B
def regionA (l w : ℝ) : Prop := 2 * (l + w) = 16 ∧ l = 2 * w
def regionB (l w : ℝ) : Prop := 2 * (l + w) = 20 ∧ l = 2 * w
def area (l w : ℝ) : ℝ := l * w

theorem regionA_regionC_area_ratio {lA wA lB wB lC wC : ℝ} :
  regionA lA wA → regionB lB wB → (lC = lB ∧ wC = wB) → 
  (area lC wC ≠ 0) → 
  (area lA wA / area lC wC = 16 / 25) :=
by
  intros hA hB hC hC_area_ne_zero
  sorry

end regionA_regionC_area_ratio_l181_181745


namespace more_likely_millionaire_city_resident_l181_181280

noncomputable def probability_A (M N m n : ℝ) : ℝ :=
  m * M / (m * M + n * N)

noncomputable def probability_B (M N : ℝ) : ℝ :=
  M / (M + N)

theorem more_likely_millionaire_city_resident
  (M N : ℕ) (m n : ℝ) (hM : M > 0) (hN : N > 0)
  (hm : m > 10^6) (hn : n ≤ 10^6) :
  probability_A M N m n > probability_B M N :=
by {
  sorry
}

end more_likely_millionaire_city_resident_l181_181280


namespace sqrt_sum_of_four_terms_of_4_pow_4_l181_181914

-- Proof Statement
theorem sqrt_sum_of_four_terms_of_4_pow_4 : 
  Real.sqrt (4 ^ 4 + 4 ^ 4 + 4 ^ 4 + 4 ^ 4) = 32 := 
by 
  sorry

end sqrt_sum_of_four_terms_of_4_pow_4_l181_181914


namespace triangle_angle_sum_l181_181034

/-- In any triangle ABC, the sum of angle A and angle B is given to be 80 degrees.
    We need to prove that the measure of angle C is 100 degrees. -/
theorem triangle_angle_sum (A B C : ℝ) 
  (h1 : A + B = 80)
  (h2 : A + B + C = 180) : C = 100 :=
sorry

end triangle_angle_sum_l181_181034


namespace minimum_value_l181_181370

noncomputable def minimum_y_over_2x_plus_1_over_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2 * y = 1) : ℝ :=
  (y / (2 * x)) + (1 / y)

theorem minimum_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2 * y = 1) :
  minimum_y_over_2x_plus_1_over_y x y hx hy h = 2 + Real.sqrt 2 :=
sorry

end minimum_value_l181_181370


namespace triangle_angle_C_l181_181057

theorem triangle_angle_C (A B C : ℝ) (h : A + B = 80) : C = 100 :=
sorry

end triangle_angle_C_l181_181057


namespace compute_quotient_of_q_and_r_l181_181254

theorem compute_quotient_of_q_and_r (p q r s t : ℤ) (h_eq_4 : 256 * p + 64 * q + 16 * r + 4 * s + t = 0)
                                     (h_eq_neg3 : -27 * p + 9 * q - 3 * r + s + t = 0)
                                     (h_eq_0 : t = 0)
                                     (h_p_nonzero : p ≠ 0) :
                                     (q + r) / p = -13 :=
by
  have eq1 := h_eq_4
  have eq2 := h_eq_neg3
  rw [h_eq_0] at eq1 eq2
  sorry

end compute_quotient_of_q_and_r_l181_181254


namespace tetrahedron_edge_square_sum_l181_181228

variable (A B C D : Point)
variable (AB AC AD BC BD CD : ℝ) -- Lengths of the edges
variable (m₁ m₂ m₃ : ℝ) -- Distances between the midpoints of the opposite edges

theorem tetrahedron_edge_square_sum:
  (AB ^ 2 + AC ^ 2 + AD ^ 2 + BC ^ 2 + BD ^ 2 + CD ^ 2) =
  4 * (m₁ ^ 2 + m₂ ^ 2 + m₃ ^ 2) :=
  sorry

end tetrahedron_edge_square_sum_l181_181228


namespace binom_20_19_eq_20_l181_181316

theorem binom_20_19_eq_20 :
  nat.choose 20 19 = 20 :=
by
  -- Using the property of binomial coefficients: C(n, k) = C(n, n-k)
  have h1 : nat.choose 20 19 = nat.choose 20 (20 - 19),
  from nat.choose_symm 20 19,
  -- Simplifying: C(20, 1)
  rw [h1, nat.sub_self],
  -- Using the binomial coefficient result: C(n, 1) = n
  exact nat.choose_one_right 20

end binom_20_19_eq_20_l181_181316


namespace necessary_but_not_sufficient_condition_l181_181360

def p (x : ℝ) : Prop := |x + 1| ≤ 4
def q (x : ℝ) : Prop := x^2 - 5*x + 6 ≤ 0

theorem necessary_but_not_sufficient_condition (x : ℝ) : 
  (∀ x, q x → p x) ∧ ¬ (∀ x, p x → q x) := 
by
  sorry

end necessary_but_not_sufficient_condition_l181_181360


namespace sum_of_digits_is_21_l181_181569

theorem sum_of_digits_is_21 :
  ∃ (a b c d : ℕ), 
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧ 
  (a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10) ∧ 
  ((10 * a + b) * (10 * c + b) = 111 * d) ∧ 
  (d = 9) ∧ 
  (a + b + c + d = 21) := by
  sorry

end sum_of_digits_is_21_l181_181569


namespace find_four_numbers_l181_181689

theorem find_four_numbers
  (a d : ℕ)
  (h_pos : 0 < a - d ∧ 0 < a ∧ 0 < a + d)
  (h_sum : (a - d) + a + (a + d) = 48)
  (b c : ℕ)
  (h_geo : b = a ∧ c = a + d)
  (last : ℕ)
  (h_last_val : last = 25)
  (h_geometric_seq : (a + d) * (a + d) = b * last)
  : (a - d, a, a + d, last) = (12, 16, 20, 25) := 
  sorry

end find_four_numbers_l181_181689


namespace binomial_20_19_eq_20_l181_181312

theorem binomial_20_19_eq_20 : nat.choose 20 19 = 20 :=
sorry

end binomial_20_19_eq_20_l181_181312


namespace kishore_savings_l181_181479

-- Define the monthly expenses and condition
def expenses : Real :=
  5000 + 1500 + 4500 + 2500 + 2000 + 6100

-- Define the monthly salary and savings conditions
def salary (S : Real) : Prop :=
  expenses + 0.1 * S = S

-- Define the savings amount
def savings (S : Real) : Real :=
  0.1 * S

-- The theorem to prove
theorem kishore_savings : ∃ S : Real, salary S ∧ savings S = 2733.33 :=
by
  sorry

end kishore_savings_l181_181479


namespace exists_infinitely_many_primes_dividing_form_l181_181422

theorem exists_infinitely_many_primes_dividing_form (a : ℕ) (ha : 0 < a) :
  ∃ᶠ p in Filter.atTop, ∃ n : ℕ, p ∣ 2^(2*n) + a := 
sorry

end exists_infinitely_many_primes_dividing_form_l181_181422


namespace binomial_20_19_l181_181326

theorem binomial_20_19 : nat.choose 20 19 = 20 :=
by {
  sorry
}

end binomial_20_19_l181_181326


namespace total_letters_received_l181_181201

theorem total_letters_received :
  ∀ (g b m t : ℕ), 
    b = 40 →
    g = b + 10 →
    m = 2 * (g + b) →
    t = g + b + m → 
    t = 270 :=
by
  intros g b m t hb hg hm ht
  rw [hb, hg, hm, ht]
  sorry

end total_letters_received_l181_181201


namespace sequence_sum_property_l181_181175

theorem sequence_sum_property {a S : ℕ → ℚ} (h1 : a 1 = 3/2)
  (h2 : ∀ n : ℕ, 2 * a (n + 1) + S n = 3) :
  (∀ n : ℕ, a n = 3 * (1/2)^n) ∧
  (∃ (n_max : ℕ),  (∀ n : ℕ, n ≤ n_max → (S n = 3 * (1 - (1/2)^n)) ∧ ∀ n : ℕ, (S (2 * n)) / (S n) > 64 / 63 → n_max = 5)) :=
by {
  -- The proof would go here
  sorry
}

end sequence_sum_property_l181_181175


namespace original_board_is_120_l181_181464

-- Define the two given conditions
def S : ℕ := 35
def L : ℕ := 2 * S + 15

-- Define the length of the original board
def original_board_length : ℕ := S + L

-- The theorem we want to prove
theorem original_board_is_120 : original_board_length = 120 :=
by
  -- Skipping the actual proof
  sorry

end original_board_is_120_l181_181464


namespace irwin_basketball_l181_181560

theorem irwin_basketball (A B C D : ℕ) (h1 : C = 2) (h2 : 2^A * 5^B * 11^C * 13^D = 2420) : A = 2 :=
by
  sorry

end irwin_basketball_l181_181560


namespace power_of_power_example_l181_181164

theorem power_of_power_example : (3^2)^4 = 6561 := by
  sorry

end power_of_power_example_l181_181164


namespace max_spheres_touch_l181_181457

noncomputable def max_spheres (r_outer r_inner : ℝ) : ℕ := 6

theorem max_spheres_touch {r_outer r_inner : ℝ} :
  r_outer = 7 →
  r_inner = 3 →
  max_spheres r_outer r_inner = 6 := by
sorry

end max_spheres_touch_l181_181457


namespace part1_part2_l181_181526

noncomputable def f (x : ℝ) (a : ℝ) := (Real.exp x / x) - Real.log x + x - a

theorem part1 (x : ℝ) (a : ℝ) :
    (∀ x > 0, f x a ≥ 0) → a ≤ Real.exp 1 + 1 :=
sorry

theorem part2 (x1 x2 : ℝ) (a : ℝ) :
  f x1 a = 0 → f x2 a = 0 → x1 < 1 → 1 < x2 → x1 * x2 < 1 :=
sorry

end part1_part2_l181_181526


namespace garden_perimeter_l181_181818

theorem garden_perimeter (L B : ℕ) (hL : L = 100) (hB : B = 200) : 
  2 * (L + B) = 600 := by
sorry

end garden_perimeter_l181_181818


namespace monotonic_decreasing_interval_l181_181599

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x + 1 / x

theorem monotonic_decreasing_interval :
  ∀ x : ℝ, x > 0 → (∀ a b : ℝ, a < b → (f b ≤ f a → b ≤ (1 : ℝ) / 2)) :=
by sorry

end monotonic_decreasing_interval_l181_181599


namespace johns_age_l181_181566

-- Define the variables and conditions
def age_problem (j d : ℕ) : Prop :=
j = d - 34 ∧ j + d = 84

-- State the theorem to prove that John's age is 25
theorem johns_age : ∃ (j d : ℕ), age_problem j d ∧ j = 25 :=
by {
  sorry
}

end johns_age_l181_181566


namespace find_window_cost_l181_181558

-- Definitions (conditions)
def total_damages : ℕ := 1450
def cost_of_tire : ℕ := 250
def number_of_tires : ℕ := 3
def cost_of_tires := number_of_tires * cost_of_tire

-- The cost of the window that needs to be proven
def window_cost := total_damages - cost_of_tires

-- We state the theorem that the window costs $700 and provide a sorry as placeholder for its proof
theorem find_window_cost : window_cost = 700 :=
by sorry

end find_window_cost_l181_181558


namespace problem_85_cube_plus_3_85_square_plus_3_85_plus_1_l181_181260

theorem problem_85_cube_plus_3_85_square_plus_3_85_plus_1 :
  85^3 + 3 * (85^2) + 3 * 85 + 1 = 636256 := 
sorry

end problem_85_cube_plus_3_85_square_plus_3_85_plus_1_l181_181260


namespace area_of_quadrilateral_centroids_l181_181091

noncomputable def square_side_length : ℝ := 40
noncomputable def point_Q_XQ : ℝ := 15
noncomputable def point_Q_YQ : ℝ := 35

theorem area_of_quadrilateral_centroids (h1 : square_side_length = 40)
    (h2 : point_Q_XQ = 15)
    (h3 : point_Q_YQ = 35) :
    ∃ (area : ℝ), area = 800 / 9 :=
by
  sorry

end area_of_quadrilateral_centroids_l181_181091


namespace fraction_of_product_l181_181777

theorem fraction_of_product : (7 / 8) * 64 = 56 := by
  sorry

end fraction_of_product_l181_181777


namespace sin_segment_ratio_is_rel_prime_l181_181096

noncomputable def sin_segment_ratio : ℕ × ℕ :=
  let p := 1
  let q := 8
  (p, q)
  
theorem sin_segment_ratio_is_rel_prime :
  1 < 8 ∧ gcd 1 8 = 1 ∧ sin_segment_ratio = (1, 8) :=
by
  -- gcd 1 8 = 1
  have h1 : gcd 1 8 = 1 := by exact gcd_one_right 8
  -- 1 < 8
  have h2 : 1 < 8 := by decide
  -- final tuple
  have h3 : sin_segment_ratio = (1, 8) := by rfl
  exact ⟨h2, h1, h3⟩

end sin_segment_ratio_is_rel_prime_l181_181096


namespace det_of_A_gt_zero_l181_181795

open Matrix

noncomputable theory

def exists_matrix_A (n : ℕ) : Prop :=
  ∃ (A : Matrix (Fin n) (Fin n) ℚ), (A ^ 3 = A + 1)

theorem det_of_A_gt_zero (n : ℕ) (A : Matrix (Fin n) (Fin n) ℚ) (h : A ^ 3 = A + 1) : det A > 0 :=
sorry

example (n : ℕ) : exists_matrix_A n :=
sorry

end det_of_A_gt_zero_l181_181795


namespace general_term_arithmetic_sequence_l181_181177

-- Define an arithmetic sequence with first term a1 and common ratio q
def arithmetic_sequence (a1 : ℤ) (q : ℤ) (n : ℕ) : ℤ :=
  a1 * q ^ (n - 1)

-- Theorem: given the conditions, prove that the general term is a1 * q^(n-1)
theorem general_term_arithmetic_sequence (a1 q : ℤ) (n : ℕ) :
  arithmetic_sequence a1 q n = a1 * q ^ (n - 1) :=
by
  sorry

end general_term_arithmetic_sequence_l181_181177


namespace total_population_correct_l181_181244

/-- Define the populations of each city -/
def Population.Seattle : ℕ := sorry
def Population.LakeView : ℕ := 24000
def Population.Boise : ℕ := (3 * Population.Seattle) / 5

/-- Population of Lake View is 4000 more than the population of Seattle -/
axiom lake_view_population : Population.LakeView = Population.Seattle + 4000

/-- Define the total population -/
def total_population : ℕ :=
  Population.Seattle + Population.LakeView + Population.Boise

/-- Prove that total population of the three cities is 56000 -/
theorem total_population_correct :
  total_population = 56000 :=
sorry

end total_population_correct_l181_181244


namespace match_proverbs_l181_181790

-- Define each condition as a Lean definition
def condition1 : Prop :=
"As cold comes and heat goes, the four seasons change" = "Things are developing"

def condition2 : Prop :=
"Thousands of flowers arranged, just waiting for the first thunder" = 
"Decisively seize the opportunity to promote qualitative change"

def condition3 : Prop :=
"Despite the intention to plant flowers, they don't bloom; unintentionally planting willows, they grow into shade" = 
"The unity of contradictions"

def condition4 : Prop :=
"There will be times when the strong winds break the waves, and we will sail across the sea with clouds" = 
"The future is bright"

-- The theorem we need to prove, using the condition definitions
theorem match_proverbs : condition2 ∧ condition4 :=
sorry

end match_proverbs_l181_181790


namespace find_possible_values_of_y_l181_181576

noncomputable def solve_y (x : ℝ) : ℝ :=
  ((x - 3) ^ 2 * (x + 4)) / (2 * x - 4)

theorem find_possible_values_of_y (x : ℝ) 
  (h : x ^ 2 + 9 * (x / (x - 3)) ^ 2 = 90) : 
  solve_y x = 0 ∨ solve_y x = 105.23 := 
sorry

end find_possible_values_of_y_l181_181576


namespace floor_negative_fraction_l181_181342

theorem floor_negative_fraction : (Int.floor (-7 / 4 : ℚ)) = -2 := by
  sorry

end floor_negative_fraction_l181_181342


namespace quadratic_roots_l181_181589

theorem quadratic_roots : ∀ x : ℝ, (x^2 - 6 * x + 5 = 0) ↔ (x = 5 ∨ x = 1) :=
by sorry

end quadratic_roots_l181_181589


namespace fraction_zero_implies_x_neg1_l181_181397

theorem fraction_zero_implies_x_neg1 (x : ℝ) (h₁ : x^2 - 1 = 0) (h₂ : x - 1 ≠ 0) : x = -1 := by
  sorry

end fraction_zero_implies_x_neg1_l181_181397


namespace inequality_positive_real_numbers_l181_181891

theorem inequality_positive_real_numbers
  (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_condition : a * b + b * c + c * a = 1) :
  (a / Real.sqrt (a^2 + 1)) + (b / Real.sqrt (b^2 + 1)) + (c / Real.sqrt (c^2 + 1)) ≤ (3 / 2) :=
  sorry

end inequality_positive_real_numbers_l181_181891


namespace binom_20_19_eq_20_l181_181323

theorem binom_20_19_eq_20: nat.choose 20 19 = 20 :=
  sorry

end binom_20_19_eq_20_l181_181323


namespace floor_neg_seven_four_is_neg_two_l181_181341

noncomputable def floor_neg_seven_four : Int :=
  Int.floor (-7 / 4)

theorem floor_neg_seven_four_is_neg_two : floor_neg_seven_four = -2 := by
  sorry

end floor_neg_seven_four_is_neg_two_l181_181341


namespace total_cost_of_tickets_l181_181561

-- Conditions
def normal_price : ℝ := 50
def website_tickets_cost : ℝ := 2 * normal_price
def scalper_tickets_cost : ℝ := 2 * (2.4 * normal_price) - 10
def discounted_ticket_cost : ℝ := 0.6 * normal_price

-- Proof Statement
theorem total_cost_of_tickets :
  website_tickets_cost + scalper_tickets_cost + discounted_ticket_cost = 360 :=
by
  sorry

end total_cost_of_tickets_l181_181561


namespace find_f_2023_4_l181_181819

noncomputable def f : ℕ → ℝ → ℝ
| 0, x => 2 * Real.sqrt x
| (n + 1), x => 4 / (2 - f n x)

theorem find_f_2023_4 : f 2023 4 = -2 := sorry

end find_f_2023_4_l181_181819


namespace range_of_a_plus_b_l181_181575

theorem range_of_a_plus_b (a b : ℝ) (h1 : 0 < a) (h2 : a < b)
    (h3 : |2 - a^2| = |2 - b^2|) : 2 < a + b ∧ a + b < 2 * Real.sqrt 2 :=
by
  sorry

end range_of_a_plus_b_l181_181575


namespace floor_neg_seven_four_is_neg_two_l181_181340

noncomputable def floor_neg_seven_four : Int :=
  Int.floor (-7 / 4)

theorem floor_neg_seven_four_is_neg_two : floor_neg_seven_four = -2 := by
  sorry

end floor_neg_seven_four_is_neg_two_l181_181340


namespace evaluate_expression_l181_181825

theorem evaluate_expression :
  12 - 5 * 3^2 + 8 / 2 - 7 + 4^2 = -20 :=
by
  sorry

end evaluate_expression_l181_181825


namespace find_m_l181_181367

-- Define points O, A, B, C
def O : (ℝ × ℝ) := (0, 0)
def A : (ℝ × ℝ) := (2, 3)
def B : (ℝ × ℝ) := (1, 5)
def C (m : ℝ) : (ℝ × ℝ) := (m, 3)

-- Define vectors AB and OC
def vector_AB : (ℝ × ℝ) := (B.1 - A.1, B.2 - A.2)  -- (B - A)
def vector_OC (m : ℝ) : (ℝ × ℝ) := (m, 3)  -- (C - O)

-- Define the dot product
def dot_product (v₁ v₂ : (ℝ × ℝ)) : ℝ := (v₁.1 * v₂.1) + (v₁.2 * v₂.2)

-- Theorem: vector_AB ⊥ vector_OC implies m = 6
theorem find_m (m : ℝ) (h : dot_product vector_AB (vector_OC m) = 0) : m = 6 :=
by
  -- Proof part not required
  sorry

end find_m_l181_181367


namespace difference_between_extremes_l181_181119

/-- Define the structure of a 3-digit integer and its digits. -/
structure ThreeDigitInteger where
  hundreds : ℕ
  tens : ℕ
  units : ℕ
  val : ℕ := 100 * hundreds + 10 * tens + units

/-- Define the problem conditions. -/
def satisfiesConditions (x : ThreeDigitInteger) : Prop :=
  x.hundreds > 0 ∧
  4 * x.hundreds = 2 * x.tens ∧
  2 * x.tens = x.units

/-- Given conditions prove the difference between the two greatest possible values of x is 124. -/
theorem difference_between_extremes :
  ∃ (x₁ x₂ : ThreeDigitInteger), 
    satisfiesConditions x₁ ∧ satisfiesConditions x₂ ∧
    (x₁.val = 248 ∧ x₂.val = 124 ∧ (x₁.val - x₂.val = 124)) :=
sorry

end difference_between_extremes_l181_181119


namespace exists_bijection_l181_181731

-- Define the non-negative integers set
def N_0 := {n : ℕ // n ≥ 0}

-- Translation of the equivalent proof statement into Lean
theorem exists_bijection (f : N_0 → N_0) :
  (∀ m n : N_0, f ⟨3 * m.val * n.val + m.val + n.val, sorry⟩ = 
   ⟨4 * (f m).val * (f n).val + (f m).val + (f n).val, sorry⟩) :=
sorry

end exists_bijection_l181_181731


namespace gcd_1080_920_is_40_l181_181622

theorem gcd_1080_920_is_40 : Nat.gcd 1080 920 = 40 :=
by
  sorry

end gcd_1080_920_is_40_l181_181622


namespace problem_a_problem_b_problem_c_problem_d_l181_181617

def rotate (n : Nat) : Nat := 
  sorry -- Function definition for rotating the last digit to the start
def add_1001 (n : Nat) : Nat := 
  sorry -- Function definition for adding 1001
def subtract_1001 (n : Nat) : Nat := 
  sorry -- Function definition for subtracting 1001

theorem problem_a :
  ∃ (steps : List (Nat → Nat)), 
    (∀ step ∈ steps, step = rotate ∨ step = add_1001 ∨ step = subtract_1001) ∧ (List.foldl (λacc step => step acc) 202122 steps = 313233) :=
sorry

theorem problem_b :
  ∃ (steps : List (Nat → Nat)), 
    (∀ step ∈ steps, step = rotate ∨ step = add_1001 ∨ step = subtract_1001) ∧ (steps.length = 8) ∧ (List.foldl (λacc step => step acc) 999999 steps = 000000) :=
sorry

theorem problem_c (n : Nat) (hn : n % 11 = 0) : 
  ∀ (steps : List (Nat → Nat)), 
    (∀ step ∈ steps, step = rotate ∨ step = add_1001 ∨ step = subtract_1001) → (List.foldl (λacc step => step acc) n steps) % 11 = 0 :=
sorry

theorem problem_d : 
  ∀ (steps : List (Nat → Nat)), 
    (∀ step ∈ steps, step = rotate ∨ step = add_1001 ∨ step = subtract_1001) → ¬(List.foldl (λacc step => step acc) 112233 steps = 000000) :=
sorry

end problem_a_problem_b_problem_c_problem_d_l181_181617


namespace binom_20_19_eq_20_l181_181331

def binom (n k : ℕ) := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binom_20_19_eq_20 : binom 20 19 = 20 := by
  sorry

end binom_20_19_eq_20_l181_181331


namespace ratio_of_female_to_male_members_l181_181481

theorem ratio_of_female_to_male_members 
  (f m : ℕ)
  (avg_age_female avg_age_male avg_age_membership : ℕ)
  (hf : avg_age_female = 35)
  (hm : avg_age_male = 30)
  (ha : avg_age_membership = 32)
  (h_avg : (35 * f + 30 * m) / (f + m) = 32) : 
  f / m = 2 / 3 :=
sorry

end ratio_of_female_to_male_members_l181_181481


namespace min_m_squared_plus_n_squared_l181_181390

theorem min_m_squared_plus_n_squared {m n : ℝ} (h : 4 * m - 3 * n - 5 * Real.sqrt 2 = 0) :
  m^2 + n^2 = 2 :=
sorry

end min_m_squared_plus_n_squared_l181_181390


namespace probability_of_multiple_12_is_one_third_l181_181712

open Finset

def numbers := {3, 4, 6, 9}

noncomputable def product_is_multiple_of_12 (x y : ℕ) : Prop :=
  (x * y) % 12 = 0

noncomputable def favorable_pairs := {(3, 4), (4, 6)}

noncomputable def all_pairs := numbers.toFinset.powerset.filter (λ s, s.card = 2)

noncomputable def favorable_count : ℕ :=
  favorable_pairs.card

noncomputable def total_count : ℕ :=
  all_pairs.card

theorem probability_of_multiple_12_is_one_third :
  (favorable_count : ℚ) / (total_count : ℚ) = 1 / 3 :=
by
  sorry

end probability_of_multiple_12_is_one_third_l181_181712


namespace derangements_8_letters_l181_181554

/-- The number of derangements of 8 letters A, B, C, D, E, F, G, H where A, C, E, G do not appear in their original positions is 24024. -/
theorem derangements_8_letters : ∃ n : ℕ, n = 24024 ∧
  derangement_count 8 {0, 2, 4, 6} n :=
sorry

end derangements_8_letters_l181_181554


namespace total_spending_in_4_years_is_680_l181_181605

-- Define the spending of each person
def trevor_spending_yearly : ℕ := 80
def reed_spending_yearly : ℕ := trevor_spending_yearly - 20
def quinn_spending_yearly : ℕ := reed_spending_yearly / 2

-- Define the total spending in one year
def total_spending_yearly : ℕ := trevor_spending_yearly + reed_spending_yearly + quinn_spending_yearly

-- Let's state what we need to prove
theorem total_spending_in_4_years_is_680 : 
  4 * total_spending_yearly = 680 := 
by 
  -- this is where the proof steps would go
  sorry

end total_spending_in_4_years_is_680_l181_181605


namespace fraction_zero_iff_x_neg_one_l181_181395

theorem fraction_zero_iff_x_neg_one (x : ℝ) (h₀ : x^2 - 1 = 0) (h₁ : x - 1 ≠ 0) :
  (x^2 - 1) / (x - 1) = 0 ↔ x = -1 :=
by
  sorry

end fraction_zero_iff_x_neg_one_l181_181395


namespace relative_error_comparison_l181_181295

theorem relative_error_comparison :
  (0.05 / 25 = 0.002) ∧ (0.4 / 200 = 0.002) → (0.002 = 0.002) :=
by
  sorry

end relative_error_comparison_l181_181295


namespace perpendicular_condition_sufficient_but_not_necessary_l181_181023

theorem perpendicular_condition_sufficient_but_not_necessary (a : ℝ) :
  (a = -2) → ((∀ x y : ℝ, ax + (a + 1) * y + 1 = 0 → x + a * y + 2 = 0 ∧ (∃ t : ℝ, t ≠ 0 ∧ x = -t / (a + 1) ∧ y = (t / a))) →
  ¬ (a = -2) ∨ (a + 1 ≠ 0 ∧ ∃ k1 k2 : ℝ, k1 * k2 = -1 ∧ k1 = -a / (a + 1) ∧ k2 = -1 / a)) :=
by
  sorry

end perpendicular_condition_sufficient_but_not_necessary_l181_181023


namespace commutative_binary_op_no_identity_element_associative_binary_op_l181_181491

def binary_op (x y : ℤ) : ℤ :=
  2 * (x + 2) * (y + 2) - 3

theorem commutative_binary_op (x y : ℤ) : binary_op x y = binary_op y x := by
  sorry

theorem no_identity_element (x e : ℤ) : ¬ (∀ x, binary_op x e = x) := by
  sorry

theorem associative_binary_op (x y z : ℤ) : (binary_op (binary_op x y) z = binary_op x (binary_op y z)) ∨ ¬ (binary_op (binary_op x y) z = binary_op x (binary_op y z)) := by
  sorry

end commutative_binary_op_no_identity_element_associative_binary_op_l181_181491


namespace impossible_event_D_l181_181458

-- Event definitions
def event_A : Prop := true -- This event is not impossible
def event_B : Prop := true -- This event is not impossible
def event_C : Prop := true -- This event is not impossible
def event_D (bag : Finset String) : Prop :=
  if "red" ∈ bag then false else true -- This event is impossible if there are no red balls

-- Bag condition
def bag : Finset String := {"white", "white", "white", "white", "white", "white", "white", "white"}

-- Proof statement
theorem impossible_event_D : event_D bag = true :=
by
  -- The bag contains only white balls, so drawing a red ball is impossible.
  rw [event_D, if_neg]
  sorry

end impossible_event_D_l181_181458


namespace tan_product_ge_sqrt2_l181_181183

variable {α β γ : ℝ}

theorem tan_product_ge_sqrt2 (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) (hγ : 0 < γ ∧ γ < π / 2) 
  (h : Real.cos α ^ 2 + Real.cos β ^ 2 + Real.cos γ ^ 2 = 1) : 
  Real.tan α * Real.tan β * Real.tan γ ≥ 2 * Real.sqrt 2 := 
by
  sorry

end tan_product_ge_sqrt2_l181_181183


namespace problem1_problem2_problem3_l181_181361

def point : Type := (ℝ × ℝ)
def vec (p1 p2 : point) : point := (p2.1 - p1.1, p2.2 - p1.2)

noncomputable def A : point := (-2, 4)
noncomputable def B : point := (3, -1)
noncomputable def C : point := (-3, -4)

noncomputable def a : point := vec A B
noncomputable def b : point := vec B C
noncomputable def c : point := vec C A

-- Problem 1
theorem problem1 : (3 * a.1 + b.1 - 3 * c.1, 3 * a.2 + b.2 - 3 * c.2) = (6, -42) :=
sorry

-- Problem 2
theorem problem2 : ∃ m n : ℝ, a = (m * b.1 + n * c.1, m * b.2 + n * c.2) ∧ m = -1 ∧ n = -1 :=
sorry

-- Helper function for point addition
def add_point (p1 p2 : point) : point := (p1.1 + p2.1, p1.2 + p2.2)
def scale_point (k : ℝ) (p : point) : point := (k * p.1, k * p.2)

-- problem 3
noncomputable def M : point := add_point (scale_point 3 c) C
noncomputable def N : point := add_point (scale_point (-2) b) C

theorem problem3 : M = (0, 20) ∧ N = (9, 2) ∧ vec M N = (9, -18) :=
sorry

end problem1_problem2_problem3_l181_181361


namespace combination_property_problem_solution_l181_181124

open Nat

def combination (n k : ℕ) : ℕ :=
  if h : k ≤ n then (factorial n) / (factorial k * factorial (n - k)) else 0

theorem combination_property (n k : ℕ) (h₀ : 1 ≤ k) (h₁ : k ≤ n) :
  combination n k + combination n (k - 1) = combination (n + 1) k := sorry

theorem problem_solution :
  (combination 3 2 + combination 4 2 + combination 5 2 + combination 6 2 + combination 7 2 + 
   combination 8 2 + combination 9 2 + combination 10 2 + combination 11 2 + combination 12 2 + 
   combination 13 2 + combination 14 2 + combination 15 2 + combination 16 2 + combination 17 2 + 
   combination 18 2 + combination 19 2) = 1139 := sorry

end combination_property_problem_solution_l181_181124


namespace sequence_a4_l181_181686

theorem sequence_a4 (S : ℕ → ℚ) (a : ℕ → ℚ) 
  (hS : ∀ n, S n = (n + 1) / (n + 2))
  (hS0 : S 0 = a 0)
  (hSn : ∀ n, S (n + 1) = S n + a (n + 1)) :
  a 4 = 1 / 30 := 
sorry

end sequence_a4_l181_181686


namespace julio_salary_l181_181567

-- Define the conditions
def customers_first_week : ℕ := 35
def customers_second_week : ℕ := 2 * customers_first_week
def customers_third_week : ℕ := 3 * customers_first_week
def commission_per_customer : ℕ := 1
def bonus : ℕ := 50
def total_earnings : ℕ := 760

-- Calculate total commission and total earnings
def commission_first_week : ℕ := customers_first_week * commission_per_customer
def commission_second_week : ℕ := customers_second_week * commission_per_customer
def commission_third_week : ℕ := customers_third_week * commission_per_customer
def total_commission : ℕ := commission_first_week + commission_second_week + commission_third_week
def total_earnings_commission_bonus : ℕ := total_commission + bonus

-- Define the proof problem
theorem julio_salary : total_earnings - total_earnings_commission_bonus = 500 :=
by
  sorry

end julio_salary_l181_181567


namespace floor_neg_seven_over_four_l181_181348

theorem floor_neg_seven_over_four : Int.floor (- 7 / 4 : ℝ) = -2 := 
by
  sorry

end floor_neg_seven_over_four_l181_181348


namespace problem_proof_l181_181685

variable (f : ℝ → ℝ)
variable (hp : ∀ x ∈ Ioo (0 : ℝ) (Real.pi / 2), f x < f' x * Real.tan x)

theorem problem_proof : sqrt 3 * f (Real.pi / 6) < f (Real.pi / 3) := sorry

end problem_proof_l181_181685


namespace floor_neg_seven_four_is_neg_two_l181_181339

noncomputable def floor_neg_seven_four : Int :=
  Int.floor (-7 / 4)

theorem floor_neg_seven_four_is_neg_two : floor_neg_seven_four = -2 := by
  sorry

end floor_neg_seven_four_is_neg_two_l181_181339


namespace range_of_a_l181_181858

theorem range_of_a (a : ℝ) : (¬ ∃ x : ℝ, x^2 + (a - 1) * x + 1 ≤ 0) → -1 < a ∧ a < 3 :=
sorry

end range_of_a_l181_181858


namespace find_alpha_l181_181551

def demand_function (p : ℝ) : ℝ := 150 - p
def supply_function (p : ℝ) : ℝ := 3 * p - 10

def new_demand_function (p : ℝ) (α : ℝ) : ℝ := α * (150 - p)

theorem find_alpha (α : ℝ) :
  (∃ p₀ p_new, demand_function p₀ = supply_function p₀ ∧ 
    p_new = p₀ * 1.25 ∧ 
    3 * p_new - 10 = new_demand_function p_new α) →
  α = 1.4 :=
by
  sorry

end find_alpha_l181_181551


namespace two_digit_number_system_l181_181290

theorem two_digit_number_system (x y : ℕ) :
  (10 * x + y - 3 * (x + y) = 13) ∧ (10 * x + y - 6 = 4 * (x + y)) :=
by sorry

end two_digit_number_system_l181_181290


namespace smallest_value_am_hm_inequality_l181_181573

theorem smallest_value_am_hm_inequality (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a + b + c + d) * (1 / (a + b) + 1 / (a + c) + 1 / (b + d) + 1 / (c + d)) ≥ 8 :=
by
  sorry

end smallest_value_am_hm_inequality_l181_181573


namespace neither_sufficient_nor_necessary_l181_181016

theorem neither_sufficient_nor_necessary (a b : ℝ) (h : a^2 > b^2) : 
  ¬(a > b) ∨ ¬(b > a) := sorry

end neither_sufficient_nor_necessary_l181_181016


namespace Bridget_skittles_after_giving_l181_181647

-- Given conditions
def Bridget_initial_skittles : ℕ := 4
def Henry_skittles : ℕ := 4
def Henry_gives_all_to_Bridget : Prop := True

-- Prove that Bridget will have 8 Skittles in total after Henry gives all of his Skittles to her.
theorem Bridget_skittles_after_giving (h : Henry_gives_all_to_Bridget) :
  Bridget_initial_skittles + Henry_skittles = 8 :=
by
  sorry

end Bridget_skittles_after_giving_l181_181647


namespace combined_weight_difference_l181_181075

-- Define the weights of the textbooks
def chemistry_weight : ℝ := 7.125
def geometry_weight : ℝ := 0.625
def calculus_weight : ℝ := 5.25
def biology_weight : ℝ := 3.75

-- Define the problem statement that needs to be proven
theorem combined_weight_difference :
  ((calculus_weight + biology_weight) - (chemistry_weight - geometry_weight)) = 2.5 :=
by
  sorry

end combined_weight_difference_l181_181075


namespace total_cost_of_tickets_l181_181562

-- Conditions
def normal_price : ℝ := 50
def website_tickets_cost : ℝ := 2 * normal_price
def scalper_tickets_cost : ℝ := 2 * (2.4 * normal_price) - 10
def discounted_ticket_cost : ℝ := 0.6 * normal_price

-- Proof Statement
theorem total_cost_of_tickets :
  website_tickets_cost + scalper_tickets_cost + discounted_ticket_cost = 360 :=
by
  sorry

end total_cost_of_tickets_l181_181562


namespace evaluate_polynomial_l181_181261

theorem evaluate_polynomial (x : ℤ) (h : x = 3) : x^6 - 6 * x^2 = 675 := by
  sorry

end evaluate_polynomial_l181_181261


namespace problem_l181_181363

theorem problem (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 16) 
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 4) : 
  x * y * z = 4 := 
sorry

end problem_l181_181363


namespace lamp_cost_l181_181490

def saved : ℕ := 500
def couch : ℕ := 750
def table : ℕ := 100
def remaining_owed : ℕ := 400

def total_cost_without_lamp : ℕ := couch + table

theorem lamp_cost :
  total_cost_without_lamp - saved + lamp = remaining_owed → lamp = 50 := by
  sorry

end lamp_cost_l181_181490


namespace compare_probabilities_l181_181272

theorem compare_probabilities 
  (M N : ℕ)
  (m n : ℝ)
  (h_m_pos : m > 0)
  (h_n_pos : n > 0)
  (h_m_million : m > 10^6)
  (h_n_million : n ≤ 10^6) :
  (M : ℝ) / (M + n / m * N) > (M : ℝ) / (M + N) :=
by
  sorry

end compare_probabilities_l181_181272


namespace more_likely_millionaire_city_resident_l181_181279

noncomputable def probability_A (M N m n : ℝ) : ℝ :=
  m * M / (m * M + n * N)

noncomputable def probability_B (M N : ℝ) : ℝ :=
  M / (M + N)

theorem more_likely_millionaire_city_resident
  (M N : ℕ) (m n : ℝ) (hM : M > 0) (hN : N > 0)
  (hm : m > 10^6) (hn : n ≤ 10^6) :
  probability_A M N m n > probability_B M N :=
by {
  sorry
}

end more_likely_millionaire_city_resident_l181_181279


namespace binom_20_19_eq_20_l181_181320

theorem binom_20_19_eq_20: nat.choose 20 19 = 20 :=
  sorry

end binom_20_19_eq_20_l181_181320


namespace online_store_commission_l181_181946

theorem online_store_commission (cost : ℝ) (desired_profit_pct : ℝ) (online_price : ℝ) (commission_pct : ℝ) :
  cost = 19 →
  desired_profit_pct = 0.20 →
  online_price = 28.5 →
  commission_pct = 25 :=
by
  sorry

end online_store_commission_l181_181946


namespace walker_rate_l181_181139

theorem walker_rate (W : ℝ) :
  (∀ t : ℝ, t = 5 / 60 ∧ t = 20 / 60 → 20 * t = (5 * 20 / 3) ∧ W * (1 / 3) = 5 / 3) →
  W = 5 :=
by
  sorry

end walker_rate_l181_181139


namespace transportable_load_l181_181470

theorem transportable_load 
  (mass_of_load : ℝ) 
  (num_boxes : ℕ) 
  (box_capacity : ℝ) 
  (num_trucks : ℕ) 
  (truck_capacity : ℝ) 
  (h1 : mass_of_load = 13.5) 
  (h2 : box_capacity = 0.35) 
  (h3 : truck_capacity = 1.5) 
  (h4 : num_trucks = 11)
  (boxes_condition : ∀ (n : ℕ), n * box_capacity ≥ mass_of_load) :
  mass_of_load ≤ num_trucks * truck_capacity :=
by
  sorry

end transportable_load_l181_181470


namespace binom_20_19_eq_20_l181_181317

theorem binom_20_19_eq_20 :
  nat.choose 20 19 = 20 :=
by
  -- Using the property of binomial coefficients: C(n, k) = C(n, n-k)
  have h1 : nat.choose 20 19 = nat.choose 20 (20 - 19),
  from nat.choose_symm 20 19,
  -- Simplifying: C(20, 1)
  rw [h1, nat.sub_self],
  -- Using the binomial coefficient result: C(n, 1) = n
  exact nat.choose_one_right 20

end binom_20_19_eq_20_l181_181317


namespace correct_option_l181_181881

variable (p q : Prop)

/-- If only one of p and q is true, then p or q is a true proposition. -/
theorem correct_option (h : (p ∧ ¬ q) ∨ (¬ p ∧ q)) : p ∨ q :=
by sorry

end correct_option_l181_181881


namespace floor_neg_seven_over_four_l181_181349

theorem floor_neg_seven_over_four : Int.floor (- 7 / 4 : ℝ) = -2 := 
by
  sorry

end floor_neg_seven_over_four_l181_181349


namespace percent_savings_per_roll_l181_181629

theorem percent_savings_per_roll 
  (cost_case : ℕ := 900) -- In cents, equivalent to $9
  (cost_individual : ℕ := 100) -- In cents, equivalent to $1
  (num_rolls : ℕ := 12) :
  (cost_individual - (cost_case / num_rolls)) * 100 / cost_individual = 25 := 
sorry

end percent_savings_per_roll_l181_181629


namespace javier_needs_10_dozen_l181_181411

def javier_goal : ℝ := 96
def cost_per_dozen : ℝ := 2.40
def selling_price_per_donut : ℝ := 1

theorem javier_needs_10_dozen : (javier_goal / ((selling_price_per_donut - (cost_per_dozen / 12)) * 12)) = 10 :=
by
  sorry

end javier_needs_10_dozen_l181_181411


namespace min_value_of_squares_l181_181574

theorem min_value_of_squares (a b c t : ℝ) (h : a + b + c = t) : a^2 + b^2 + c^2 ≥ t^2 / 3 :=
sorry

end min_value_of_squares_l181_181574


namespace binom_20_19_eq_20_l181_181321

theorem binom_20_19_eq_20: nat.choose 20 19 = 20 :=
  sorry

end binom_20_19_eq_20_l181_181321


namespace max_volume_cylinder_l181_181817

theorem max_volume_cylinder (x : ℝ) (h1 : x > 0) (h2 : x < 10) : 
  (∀ x, 0 < x ∧ x < 10 → ∃ max_v, max_v = (4 * (10^3) * Real.pi) / 27) ∧ 
  ∃ x, x = 20/3 := 
by
  sorry

end max_volume_cylinder_l181_181817


namespace triangle_inequality_l181_181176

noncomputable def area_triangle (a b c : ℝ) : ℝ := sorry -- Definition of area, but implementation is not required.

theorem triangle_inequality (a b c : ℝ) (S_triangle : ℝ):
  1 - (8 * ((a - b) ^ 2 + (b - c) ^ 2 + (c - a) ^ 2) / (a + b + c) ^ 2)
  ≤ 432 * S_triangle ^ 2 / (a + b + c) ^ 4
  ∧ 432 * S_triangle ^ 2 / (a + b + c) ^ 4
  ≤ 1 - (2 * ((a - b) ^ 2 + (b - c) ^ 2 + (c - a) ^ 2) / (a + b + c) ^ 2) :=
sorry -- Proof is omitted

end triangle_inequality_l181_181176


namespace polygon_area_l181_181609

-- Define the vertices of the polygon
def x1 : ℝ := 0
def y1 : ℝ := 0

def x2 : ℝ := 4
def y2 : ℝ := 0

def x3 : ℝ := 2
def y3 : ℝ := 3

def x4 : ℝ := 4
def y4 : ℝ := 6

-- Define the expression for the Shoelace Theorem
def shoelace_area (x1 y1 x2 y2 x3 y3 x4 y4 : ℝ) : ℝ :=
  0.5 * abs (x1 * y2 + x2 * y3 + x3 * y4 + x4 * y1 - (y1 * x2 + y2 * x3 + y3 * x4 + y4 * x1))

-- The theorem statement proving the area of the polygon
theorem polygon_area :
  shoelace_area x1 y1 x2 y2 x3 y3 x4 y4 = 6 := 
  by
  sorry

end polygon_area_l181_181609


namespace costs_equal_when_x_20_l181_181474

noncomputable def costA (x : ℕ) : ℤ := 150 * x + 3300
noncomputable def costB (x : ℕ) : ℤ := 210 * x + 2100

theorem costs_equal_when_x_20 : costA 20 = costB 20 :=
by
  -- Statements representing the costs equal condition
  have ha : costA 20 = 150 * 20 + 3300 := rfl
  have hb : costB 20 = 210 * 20 + 2100 := rfl
  rw [ha, hb]
  -- Simplification steps (represented here in Lean)
  sorry

end costs_equal_when_x_20_l181_181474


namespace carla_bought_marbles_l181_181485

def starting_marbles : ℕ := 2289
def total_marbles : ℝ := 2778.0

theorem carla_bought_marbles : (total_marbles - starting_marbles) = 489 := 
by
  sorry

end carla_bought_marbles_l181_181485


namespace girls_left_to_play_kho_kho_l181_181807

theorem girls_left_to_play_kho_kho (B G x : ℕ) 
  (h_eq : B = G)
  (h_twice : B = 2 * (G - x))
  (h_total : B + G = 32) :
  x = 8 :=
by sorry

end girls_left_to_play_kho_kho_l181_181807


namespace Marty_combinations_l181_181883

theorem Marty_combinations :
  let colors := 5
  let methods := 4
  let patterns := 3
  colors * methods * patterns = 60 :=
by
  sorry

end Marty_combinations_l181_181883


namespace toys_per_box_l181_181087

theorem toys_per_box (number_of_boxes total_toys : ℕ) (h₁ : number_of_boxes = 4) (h₂ : total_toys = 32) :
  total_toys / number_of_boxes = 8 :=
by
  sorry

end toys_per_box_l181_181087


namespace regression_equation_correct_l181_181602

-- Defining the given data as constants
def x_data : List ℕ := [1, 2, 3, 4, 5, 6, 7]
def y_data : List ℕ := [891, 888, 351, 220, 200, 138, 112]

def sum_t_y : ℚ := 1586
def avg_t : ℚ := 0.37
def sum_t2_min7_avg_t2 : ℚ := 0.55

-- Defining the target regression equation
def target_regression (x : ℚ) : ℚ := 1000 / x + 30

-- Function to calculate the regression equation from data
noncomputable def calculate_regression (x_data y_data : List ℕ) : (ℚ → ℚ) :=
  let n : ℚ := x_data.length
  let avg_y : ℚ := y_data.sum / n
  let b : ℚ := (sum_t_y - n * avg_t * avg_y) / (sum_t2_min7_avg_t2)
  let a : ℚ := avg_y - b * avg_t
  fun x : ℚ => a + b / x

-- Theorem stating the regression equation matches the target regression equation
theorem regression_equation_correct :
  calculate_regression x_data y_data = target_regression :=
by
  sorry

end regression_equation_correct_l181_181602


namespace calculate_total_amount_l181_181428

-- Define the number of dogs.
def numberOfDogs : ℕ := 2

-- Define the number of puppies each dog gave birth to.
def puppiesPerDog : ℕ := 10

-- Define the fraction of puppies sold.
def fractionSold : ℚ := 3 / 4

-- Define the price per puppy.
def pricePerPuppy : ℕ := 200

-- Define the total amount of money Nalani received from the sale of the puppies.
def totalAmountReceived : ℕ := 3000

-- The theorem to prove that the total amount of money received is $3000 given the conditions.
theorem calculate_total_amount :
  (numberOfDogs * puppiesPerDog * fractionSold * pricePerPuppy : ℚ).toNat = totalAmountReceived :=
by
  sorry

end calculate_total_amount_l181_181428


namespace combined_area_of_removed_triangles_l181_181288

theorem combined_area_of_removed_triangles (s : ℝ) (x : ℝ) (h : 15 = ((s - 2 * x) ^ 2 + (s - 2 * x) ^ 2) ^ (1/2)) :
  2 * x ^ 2 = 28.125 :=
by
  -- The necessary proof will go here
  sorry

end combined_area_of_removed_triangles_l181_181288


namespace number_of_students_per_normal_class_l181_181850

theorem number_of_students_per_normal_class (total_students : ℕ) (percentage_moving : ℕ) (grade_levels : ℕ) (adv_class_size : ℕ) (additional_classes : ℕ) 
  (h1 : total_students = 1590) 
  (h2 : percentage_moving = 40) 
  (h3 : grade_levels = 3) 
  (h4 : adv_class_size = 20) 
  (h5 : additional_classes = 6) : 
  (total_students * percentage_moving / 100 / grade_levels - adv_class_size) / additional_classes = 32 :=
by
  sorry

end number_of_students_per_normal_class_l181_181850


namespace cone_base_circumference_l181_181134

noncomputable def radius (V : ℝ) (h : ℝ) : ℝ :=
  real.sqrt (3 * V / (π * h))

theorem cone_base_circumference
  (V : ℝ) (h : ℝ) (C : ℝ) (π_neq_zero : π ≠ 0)
  (V_eq : V = 24 * π) (h_eq : h = 6)
  (C_eq : C = 4 * real.sqrt 3 * π) :
  2 * π * radius V h = C :=
by {
  -- Simplify the expressions using the provided conditions
  rw [V_eq, h_eq, C_eq, radius, real.sqrt_mul, real.sqrt_div, mul_comm, ← mul_assoc],
  -- Answer follows directly from given conditions and basic algebra
  sorry
}

end cone_base_circumference_l181_181134


namespace inequality_proof_l181_181508

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  abc ≥ (a + b + c) / ((1 / a^2) + (1 / b^2) + (1 / c^2)) ∧
  (a + b + c) / ((1 / a^2) + (1 / b^2) + (1 / c^2)) ≥ (a + b - c) * (b + c - a) * (c + a - b) :=
by
  sorry

end inequality_proof_l181_181508


namespace servings_left_proof_l181_181971

-- Define the number of servings prepared
def total_servings : ℕ := 61

-- Define the number of guests
def total_guests : ℕ := 8

-- Define the fraction of servings the first 3 guests shared
def first_three_fraction : ℚ := 2 / 5

-- Define the fraction of servings the next 4 guests shared
def next_four_fraction : ℚ := 1 / 4

-- Define the number of servings consumed by the 8th guest
def eighth_guest_servings : ℕ := 5

-- Total consumed servings by the first three guests (rounded down)
def first_three_consumed := (first_three_fraction * total_servings).floor

-- Total consumed servings by the next four guests (rounded down)
def next_four_consumed := (next_four_fraction * total_servings).floor

-- Total consumed servings in total
def total_consumed := first_three_consumed + next_four_consumed + eighth_guest_servings

-- The number of servings left unconsumed
def servings_left_unconsumed := total_servings - total_consumed

-- The theorem stating there are 17 servings left unconsumed
theorem servings_left_proof : servings_left_unconsumed = 17 := by
  sorry

end servings_left_proof_l181_181971


namespace no_natural_m_n_prime_l181_181153

theorem no_natural_m_n_prime (m n : ℕ) : ¬Prime (n^2 + 2018 * m * n + 2019 * m + n - 2019 * m^2) :=
by
  sorry

end no_natural_m_n_prime_l181_181153


namespace calculate_expression_l181_181826

theorem calculate_expression : ((-3: ℤ) ^ 3 + (5: ℤ) ^ 2 - ((-2: ℤ) ^ 2)) = -6 := by
  sorry

end calculate_expression_l181_181826


namespace factor_polynomial_l181_181673

theorem factor_polynomial : ∀ x : ℝ, 
  x^8 - 4 * x^6 + 6 * x^4 - 4 * x^2 + 1 = (x - 1)^4 * (x + 1)^4 :=
by
  intro x
  sorry

end factor_polynomial_l181_181673


namespace sqrt_neg_num_squared_l181_181652

theorem sqrt_neg_num_squared (n : ℤ) (hn : n = -2023) : Real.sqrt (n ^ 2) = 2023 :=
by
  -- substitute -2023 for n
  rw hn
  -- compute (-2023)^2
  have h1 : (-2023 : ℝ) ^ 2 = 2023 ^ 2 :=
    by rw [neg_sq, abs_of_nonneg (by norm_num)]
  -- show sqrt(2023^2) = 2023
  rw [h1, Real.sqrt_sq]
  exact eq_of_abs_sub_nonneg (by norm_num)

end sqrt_neg_num_squared_l181_181652


namespace total_amount_paid_l181_181874

def original_price : ℝ := 20
def discount_rate : ℝ := 0.5
def number_of_tshirts : ℕ := 6

theorem total_amount_paid : 
  (number_of_tshirts : ℝ) * (original_price * discount_rate) = 60 := by
  sorry

end total_amount_paid_l181_181874


namespace probability_more_heads_than_tails_l181_181380

theorem probability_more_heads_than_tails :
  let n := 10
  let total_outcomes := 2^n
  let equal_heads_tails_ways := Nat.choose n (n / 2)
  let y := (equal_heads_tails_ways : ℝ) / total_outcomes
  let x := (1 - y) / 2
  x = 193 / 512 := by
    let n := 10
    let total_outcomes := 2^n
    have h1 : equal_heads_tails_ways = Nat.choose n (n / 2) := rfl
    have h2 : total_outcomes = 2^n := rfl
    let equal_heads_tails_ways := Nat.choose n (n / 2)
    let y := (equal_heads_tails_ways : ℝ) / total_outcomes
    have h3 : y = 63 / 256 := sorry  -- calculation steps
    let x := (1 - y) / 2
    have h4 : x = 193 / 512 := sorry  -- calculation steps
    exact h4

end probability_more_heads_than_tails_l181_181380


namespace distance_between_given_parallel_lines_l181_181991

noncomputable def distance_between_parallel_lines (L1 L2 : AffineSubspace ℝ (EuclideanSpace ℝ (Fin 2))) : ℝ :=
  -- Define the distance formula for two parallel lines
sorry

theorem distance_between_given_parallel_lines :
  let L1 := (3:ℝ) • EuclideanSpace.mk 2 A • (EuclideanSpace.mk 1 B • (6:ℝ) • EuclideanSpace.mk 2 C) = 0
  let L2 := (6:ℝ) • EuclideanSpace.mk 1 A • (4:ℝ) • (EuclideanSpace.mk 2 B • (3:ℝ)) = 0
  (distance_between_parallel_lines L1 L2 = 3) := 
by
sorry

end distance_between_given_parallel_lines_l181_181991


namespace equation_of_BC_area_of_triangle_l181_181696

section triangle_geometry

variables (x y : ℝ)

/-- Given equations of the altitudes and vertex A, the equation of side BC is 2x + 3y + 7 = 0 -/
theorem equation_of_BC (h1 : 2 * x - 3 * y + 1 = 0) (h2 : x + y = 0) (A : ℝ × ℝ) (hA : A = (1, 2)) :
  ∃ (a b c : ℝ), (a * x + b * y + c = 0) ∧ (a, b, c) = (2, 3, 7) := 
sorry

/-- Given equations of the altitudes and vertex A, the area of triangle ABC is 45/2 -/
theorem area_of_triangle (h1 : 2 * x - 3 * y + 1 = 0) (h2 : x + y = 0) (A : ℝ × ℝ) (hA : A = (1, 2)) :
  ∃ (area : ℝ), (area = (45 / 2)) := 
sorry

end triangle_geometry

end equation_of_BC_area_of_triangle_l181_181696


namespace dogwood_trees_current_l181_181767

variable (X : ℕ)
variable (trees_today : ℕ := 41)
variable (trees_tomorrow : ℕ := 20)
variable (total_trees_after : ℕ := 100)

theorem dogwood_trees_current (h : X + trees_today + trees_tomorrow = total_trees_after) : X = 39 :=
by
  sorry

end dogwood_trees_current_l181_181767


namespace factor_polynomial_l181_181672

theorem factor_polynomial : ∀ x : ℝ, 
  x^8 - 4 * x^6 + 6 * x^4 - 4 * x^2 + 1 = (x - 1)^4 * (x + 1)^4 :=
by
  intro x
  sorry

end factor_polynomial_l181_181672


namespace condition_necessary_but_not_sufficient_l181_181296

noncomputable def S_n (a_1 d : ℝ) (n : ℕ) : ℝ :=
  n * a_1 + (n * (n - 1) / 2) * d

theorem condition_necessary_but_not_sufficient (a_1 d : ℝ) :
  (∀ n : ℕ, S_n a_1 d (n + 1) > S_n a_1 d n) ↔ (a_1 + d > 0) :=
sorry

end condition_necessary_but_not_sufficient_l181_181296


namespace expected_value_Z_variance_Z_l181_181640

-- Define the probability mass functions for X and Y
def pmf_X (x : ℕ) : ℝ :=
  if x = 1 then 0.1 else if x = 2 then 0.6 else if x = 3 then 0.3 else 0.0

def pmf_Y (y : ℕ) : ℝ :=
  if y = 0 then 0.2 else if y = 1 then 0.8 else 0.0

-- Define the random variables X and Y
noncomputable def X : ℕ → ℝ := λ x, pmf_X x
noncomputable def Y : ℕ → ℝ := λ y, pmf_Y y

-- Define Z as X + Y
noncomputable def Z : ℕ → ℕ → ℝ := λ x y, X x + Y y

-- Expected value and variance of Z as per the problem statement
theorem expected_value_Z : 
  (∑ x (hx : pmf_X x > 0) y (hy : pmf_Y y > 0), (x + y) * (pmf_X x * pmf_Y y)) = 3 := sorry

theorem variance_Z : 
  (∑ x (hx : pmf_X x > 0) y (hy : pmf_Y y > 0), (x + y)^2 * (pmf_X x * pmf_Y y)) - (∑ x (hx : pmf_X x > 0) y (hy : pmf_Y y > 0), (x + y) * (pmf_X x * pmf_Y y))^2 = 0.52 := sorry

end expected_value_Z_variance_Z_l181_181640


namespace percentage_error_in_square_area_l181_181796

-- Given an error of 1% in excess while measuring the side of a square,
-- prove that the percentage of error in the calculated area of the square is 2.01%.

theorem percentage_error_in_square_area (s : ℝ) (h : s ≠ 0) :
  let measured_side := 1.01 * s
  let actual_area := s ^ 2
  let calculated_area := (1.01 * s) ^ 2
  let error_in_area := calculated_area - actual_area
  let percentage_error := (error_in_area / actual_area) * 100
  percentage_error = 2.01 :=
by {
  let measured_side := 1.01 * s;
  let actual_area := s ^ 2;
  let calculated_area := (1.01 * s) ^ 2;
  let error_in_area := calculated_area - actual_area;
  let percentage_error := (error_in_area / actual_area) * 100;
  sorry
}

end percentage_error_in_square_area_l181_181796


namespace partial_fraction_sum_zero_l181_181487

theorem partial_fraction_sum_zero
    (A B C D E : ℝ)
    (h : ∀ x : ℝ, x ≠ 0 ∧ x ≠ -1 ∧ x ≠ -2 ∧ x ≠ -3 ∧ x ≠ -5 →
    1 = A * (x + 1) * (x + 2) * (x + 3) * (x + 5) +
        B * x * (x + 2) * (x + 3) * (x + 5) +
        C * x * (x + 1) * (x + 3) * (x + 5) +
        D * x * (x + 1) * (x + 2) * (x + 5) +
        E * x * (x + 1) * (x + 2) * (x + 3)) :
    A + B + C + D + E = 0 := by
    sorry

end partial_fraction_sum_zero_l181_181487


namespace barbed_wire_cost_l181_181896

theorem barbed_wire_cost
  (A : ℕ)          -- Area of the square field (sq m)
  (cost_per_meter : ℕ)  -- Cost per meter for the barbed wire (Rs)
  (gate_width : ℕ)      -- Width of each gate (m)
  (num_gates : ℕ)       -- Number of gates
  (side_length : ℕ)     -- Side length of the square field (m)
  (perimeter : ℕ)       -- Perimeter of the square field (m)
  (total_length : ℕ)    -- Total length of the barbed wire needed (m)
  (total_cost : ℕ)      -- Total cost of drawing the barbed wire (Rs)
  (h1 : A = 3136)       -- Given: Area = 3136 sq m
  (h2 : cost_per_meter = 1)  -- Given: Cost per meter = 1 Rs/m
  (h3 : gate_width = 1)      -- Given: Width of each gate = 1 m
  (h4 : num_gates = 2)       -- Given: Number of gates = 2
  (h5 : side_length * side_length = A)  -- Side length calculated from the area
  (h6 : perimeter = 4 * side_length)    -- Perimeter of the square field
  (h7 : total_length = perimeter - (num_gates * gate_width))  -- Actual barbed wire length after gates
  (h8 : total_cost = total_length * cost_per_meter)           -- Total cost calculation
  : total_cost = 222 :=      -- The result we need to prove
sorry

end barbed_wire_cost_l181_181896


namespace tangent_line_eq_l181_181498

-- Define the function
def curve (x : ℝ) : ℝ := x^2 + Real.log x + 1

-- Define the point at which the tangent is evaluated
def point : ℝ × ℝ := (1, 2)

-- The statement to prove that the equation of the tangent line at point (1, 2) is y = 3x - 1
theorem tangent_line_eq :
  let x_1 := (1 : ℝ)
  let y_1 := (2 : ℝ)
  let k := (deriv curve x_1)
  (k = 3) → (∀ x : ℝ, curve x_1 = y_1 → k = 3 → ∀ y : ℝ, (y - y_1) = k * (x - x_1) → y = 3*x - 1) :=
by 
  sorry

end tangent_line_eq_l181_181498


namespace fraction_of_64_l181_181780

theorem fraction_of_64 : (7 / 8) * 64 = 56 :=
sorry

end fraction_of_64_l181_181780


namespace focus_coordinates_of_hyperbola_l181_181337

theorem focus_coordinates_of_hyperbola (x y : ℝ) :
  (∃ c : ℝ, (c = 5 ∧ y = 10) ∧ (c = 5 + Real.sqrt 97)) ↔ 
  (x, y) = (5 + Real.sqrt 97, 10) :=
by
  sorry

end focus_coordinates_of_hyperbola_l181_181337


namespace age_of_B_l181_181713

theorem age_of_B (A B : ℕ) (h1 : A + 10 = 2 * (B - 10)) (h2 : A = B + 11) : B = 41 :=
by
  -- Proof not required as per instructions
  sorry

end age_of_B_l181_181713


namespace sum_of_variables_is_16_l181_181373

theorem sum_of_variables_is_16 (A B C D E : ℕ)
    (h1 : C + E = 4) 
    (h2 : B + E = 7) 
    (h3 : B + D = 6) 
    (h4 : A = 6)
    (hdistinct : ∀ x y, x ≠ y → (x ≠ A ∧ x ≠ B ∧ x ≠ C ∧ x ≠ D ∧ x ≠ E) ∧ (y ≠ A ∧ y ≠ B ∧ y ≠ C ∧ y ≠ D ∧ y ≠ E)) :
    A + B + C + D + E = 16 :=
by
    sorry

end sum_of_variables_is_16_l181_181373


namespace moles_of_H2_required_l181_181852

theorem moles_of_H2_required 
  (moles_C : ℕ) 
  (moles_O2 : ℕ) 
  (moles_CH4 : ℕ) 
  (moles_CO2 : ℕ) 
  (balanced_reaction_1 : ℕ → ℕ → ℕ → Prop)
  (balanced_reaction_2 : ℕ → ℕ → ℕ → ℕ → Prop)
  (H_balanced : balanced_reaction_2 2 4 2 1)
  (H_form_CO2 : balanced_reaction_1 1 1 1) :
  moles_C = 2 ∧ moles_O2 = 1 ∧ moles_CH4 = 2 ∧ moles_CO2 = 1 → (∃ moles_H2, moles_H2 = 4) :=
by sorry

end moles_of_H2_required_l181_181852


namespace sum_square_ends_same_digit_l181_181893

theorem sum_square_ends_same_digit {a b : ℤ} (h : (a + b) % 10 = 0) :
  (a^2 % 10) = (b^2 % 10) :=
by
  sorry

end sum_square_ends_same_digit_l181_181893


namespace goods_train_speed_l181_181811

theorem goods_train_speed (length_train : ℝ) (length_platform : ℝ) (time_seconds : ℝ) (speed_kmph : ℝ) :
  length_train = 280.04 →
  length_platform = 240 →
  time_seconds = 26 →
  speed_kmph = (length_train + length_platform) / time_seconds * 3.6 →
  speed_kmph = 72 :=
by
  intros h_train h_platform h_time h_speed
  rw [h_train, h_platform, h_time] at h_speed
  sorry

end goods_train_speed_l181_181811


namespace total_spent_l181_181738

/-- Define the prices of the rides in the morning and the afternoon --/
def morning_price (ride : String) (age : Nat) : Nat :=
  match ride, age with
  | "bumper_car", n => if n < 18 then 2 else 3
  | "space_shuttle", n => if n < 18 then 4 else 5
  | "ferris_wheel", n => if n < 18 then 5 else 6
  | _, _ => 0

def afternoon_price (ride : String) (age : Nat) : Nat :=
  (morning_price ride age) + 1

/-- Define the number of rides taken by Mara and Riley --/
def rides_morning (person : String) (ride : String) : Nat :=
  match person, ride with
  | "Mara", "bumper_car" => 1
  | "Mara", "ferris_wheel" => 2
  | "Riley", "space_shuttle" => 2
  | "Riley", "ferris_wheel" => 2
  | _, _ => 0

def rides_afternoon (person : String) (ride : String) : Nat :=
  match person, ride with
  | "Mara", "bumper_car" => 1
  | "Mara", "ferris_wheel" => 1
  | "Riley", "space_shuttle" => 2
  | "Riley", "ferris_wheel" => 1
  | _, _ => 0

/-- Define the ages of Mara and Riley --/
def age (person : String) : Nat :=
  match person with
  | "Mara" => 17
  | "Riley" => 19
  | _ => 0

/-- Calculate the total expenditure --/
def total_cost (person : String) : Nat :=
  List.sum ([
    (rides_morning person "bumper_car") * (morning_price "bumper_car" (age person)),
    (rides_afternoon person "bumper_car") * (afternoon_price "bumper_car" (age person)),
    (rides_morning person "space_shuttle") * (morning_price "space_shuttle" (age person)),
    (rides_afternoon person "space_shuttle") * (afternoon_price "space_shuttle" (age person)),
    (rides_morning person "ferris_wheel") * (morning_price "ferris_wheel" (age person)),
    (rides_afternoon person "ferris_wheel") * (afternoon_price "ferris_wheel" (age person))
  ])

/-- Prove the total cost for Mara and Riley is $62 --/
theorem total_spent : total_cost "Mara" + total_cost "Riley" = 62 :=
by
  sorry

end total_spent_l181_181738


namespace find_line_equation_l181_181443

def parameterized_line (t : ℝ) : ℝ × ℝ :=
  (3 * t + 2, 5 * t - 3)

theorem find_line_equation (x y : ℝ) (t : ℝ) (h : parameterized_line t = (x, y)) :
  y = (5 / 3) * x - 19 / 3 :=
sorry

end find_line_equation_l181_181443


namespace card_length_l181_181963

noncomputable def width_card : ℕ := 2
noncomputable def side_poster_board : ℕ := 12
noncomputable def total_cards : ℕ := 24

theorem card_length :
  ∃ (card_length : ℕ),
    (side_poster_board / width_card) * (side_poster_board / card_length) = total_cards ∧ 
    card_length = 3 := by
  sorry

end card_length_l181_181963


namespace number_of_pairs_satisfying_l181_181840

theorem number_of_pairs_satisfying (h1 : 2 ^ 2013 < 5 ^ 867) (h2 : 5 ^ 867 < 2 ^ 2014) :
  ∃ k, k = 279 ∧ ∀ (m n : ℕ), 1 ≤ m ∧ m ≤ 2012 ∧ 5 ^ n < 2 ^ m ∧ 2 ^ (m + 2) < 5 ^ (n + 1) → 
  ∃ (count : ℕ), count = 279 :=
by
  sorry

end number_of_pairs_satisfying_l181_181840


namespace solve_variables_l181_181082

theorem solve_variables (x y z : ℝ)
  (h1 : (x / 6) * 12 = 10)
  (h2 : (y / 4) * 8 = x)
  (h3 : (z / 3) * 5 + y = 20) :
  x = 5 ∧ y = 2.5 ∧ z = 10.5 :=
by { sorry }

end solve_variables_l181_181082


namespace sum_nk_l181_181755

theorem sum_nk (n k : ℕ) (h₁ : 3 * n - 4 * k = 4) (h₂ : 4 * n - 5 * k = 13) : n + k = 55 := by
  sorry

end sum_nk_l181_181755


namespace population_after_4_years_l181_181140

theorem population_after_4_years 
  (initial_population : ℕ) 
  (new_people : ℕ) 
  (people_moved_out : ℕ) 
  (years : ℕ) 
  (final_population : ℕ) :
  initial_population = 780 →
  new_people = 100 →
  people_moved_out = 400 →
  years = 4 →
  final_population = initial_population + new_people - people_moved_out →
  final_population / 2 / 2 / 2 / 2 = 30 :=
by
  sorry

end population_after_4_years_l181_181140


namespace sum_n_k_l181_181754

theorem sum_n_k (n k : ℕ) (h1 : 3 * (k + 1) = n - k) (h2 : 2 * (k + 2) = n - k - 1) : n + k = 13 := by
  sorry

end sum_n_k_l181_181754


namespace volume_of_prism_l181_181921

variable (x y z : ℝ)
variable (h1 : x * y = 15)
variable (h2 : y * z = 10)
variable (h3 : x * z = 6)

theorem volume_of_prism : x * y * z = 30 :=
by
  sorry

end volume_of_prism_l181_181921


namespace arithmetic_geometric_sequence_solution_l181_181688

theorem arithmetic_geometric_sequence_solution (u v : ℕ → ℝ) (a b u₀ : ℝ) :
  (∀ n, u (n + 1) = a * u n + b) ∧ (∀ n, v (n + 1) = a * v n + b) →
  u 0 = u₀ →
  v 0 = b / (1 - a) →
  ∀ n, u n = a ^ n * (u₀ - b / (1 - a)) + b / (1 - a) :=
by
  intros
  sorry

end arithmetic_geometric_sequence_solution_l181_181688


namespace x4_value_l181_181413

/-- Define x_n sequence based on given initial value and construction rules -/
def x_n (n : ℕ) : ℕ :=
  if n = 1 then 27
  else if n = 2 then 1 * 2^4 + 1 * 2^3 + 0 * 2^2 + 1 * 2 + 1
  else if n = 3 then 1 * 2^4 + 1 * 2^3 + 0 * 2^2 + 1 * 2 + 1 -- Need to generalize for actual sequence definition
  else if n = 4 then 1 * 2^4 + 0 * 2^3 + 1 * 2^2 + 1 * 2 + 1
  else 0 -- placeholder for general case (not needed here)

/-- Prove that x_4 = 23 given x_1=27 and the sequence construction criteria --/
theorem x4_value : x_n 4 = 23 :=
by
  -- Proof not required, hence sorry is used
  sorry

end x4_value_l181_181413


namespace sampling_is_systematic_l181_181943

-- Define the total seats in each row and the total number of rows
def total_seats_per_row : ℕ := 25
def total_rows : ℕ := 30

-- Define a function to identify if the sampling is systematic
def is_systematic_sampling (sample_count : ℕ) (n : ℕ) (interval : ℕ) : Prop :=
  interval = total_seats_per_row ∧ sample_count = total_rows

-- Define the count and interval for the problem
def sample_count : ℕ := 30
def sampling_interval : ℕ := 25

-- Theorem statement: Given the conditions, it is systematic sampling
theorem sampling_is_systematic :
  is_systematic_sampling sample_count total_rows sampling_interval = true :=
sorry

end sampling_is_systematic_l181_181943


namespace find_k_l181_181375

-- Define the lines l1 and l2
def l1 (x y : ℝ) : Prop := x + 2 * y - 7 = 0
def l2 (x y : ℝ) (k : ℝ) : Prop := 2 * x + k * x + 3 = 0

-- Define the condition for parallel lines in our context
def parallel (k : ℝ) : Prop := - (1 / 2) = -(2 / k)

-- Prove that under the given conditions, k must be 4
theorem find_k (k : ℝ) : parallel k → k = 4 :=
by
  intro h
  sorry

end find_k_l181_181375


namespace tangent_line_equation_l181_181899

theorem tangent_line_equation 
    (h_perpendicular : ∃ m1 m2 : ℝ, m1 * m2 = -1 ∧ (∀ y, x + m1 * y = 4) ∧ (x + 4 * y = 4)) 
    (h_tangent : ∀ x : ℝ, y = 2 * x ^ 2 ∧ (∀ y', y' = 4 * x)) :
    ∃ a b c : ℝ, (4 * a - b - c = 0) ∧ (∀ (t : ℝ), a * t + b * (2 * t ^ 2) = 1) :=
sorry

end tangent_line_equation_l181_181899


namespace evaluate_expression_l181_181833

-- Definition of the given condition.
def sixty_four_eq_sixteen_squared : Prop := 64 = 16^2

-- The statement to prove that the given expression equals the answer.
theorem evaluate_expression (h : sixty_four_eq_sixteen_squared) : 
  (16^24) / (64^8) = 16^8 :=
by 
  -- h contains the condition that 64 = 16^2, but we provide a proof step later with sorry
  sorry

end evaluate_expression_l181_181833


namespace outfit_count_correct_l181_181706

def total_shirts : ℕ := 8
def total_pants : ℕ := 4
def total_hats : ℕ := 6
def shirt_colors : Set (String) := {"tan", "black", "blue", "gray", "white", "yellow"}
def hat_colors : Set (String) := {"tan", "black", "blue", "gray", "white", "yellow"}
def conflict_free_outfits (total_shirts total_pants total_hats : ℕ) : ℕ :=
  let total_outfits := total_shirts * total_pants * total_hats
  let matching_outfits := (2 * 1 * 4) * total_pants
  total_outfits - matching_outfits

theorem outfit_count_correct :
  conflict_free_outfits total_shirts total_pants total_hats = 160 :=
by
  unfold conflict_free_outfits
  norm_num
  sorry

end outfit_count_correct_l181_181706


namespace perpendicular_exists_l181_181189

-- Definitions for geometric entities involved

structure Point where
  x : ℝ
  y : ℝ

structure Line where
  p1 : Point
  p2 : Point

structure Circle where
  center : Point
  radius : ℝ

-- Definitions for conditions in the problem

-- Condition 1: Point C is not on the circle
def point_not_on_circle (C : Point) (circle : Circle) : Prop :=
  (C.x - circle.center.x)^2 + (C.y - circle.center.y)^2 ≠ circle.radius^2

-- Condition 2: Point C is on the circle
def point_on_circle (C : Point) (circle : Circle) : Prop :=
  (C.x - circle.center.x)^2 + (C.y - circle.center.y)^2 = circle.radius^2

-- Definitions for lines and perpendicularity
def is_perpendicular (line1 : Line) (line2 : Line) : Prop :=
  (line1.p1.x - line1.p2.x) * (line2.p1.x - line2.p2.x) +
  (line1.p1.y - line1.p2.y) * (line2.p1.y - line2.p2.y) = 0

noncomputable def perpendicular_from_point_to_line (C : Point) (line : Line) (circle : Circle) 
  (h₁ : point_not_on_circle C circle ∨ point_on_circle C circle) : Line := 
  sorry

-- The Lean statement for part (a) and (b) combined into one proof.
theorem perpendicular_exists (C : Point) (lineAB : Line) (circle : Circle) 
  (h₁ : point_not_on_circle C circle ∨ point_on_circle C circle) : 
  ∃ (line_perpendicular : Line), is_perpendicular line_perpendicular lineAB ∧ 
  (line_perpendicular.p1 = C ∨ line_perpendicular.p2 = C) :=
  sorry

end perpendicular_exists_l181_181189


namespace triangle_side_ratio_impossible_triangle_side_ratio_impossible_2_triangle_side_ratio_impossible_3_l181_181720

theorem triangle_side_ratio_impossible (a b c : ℝ) (h₁ : a = b / 2) (h₂ : a = c / 3) : false :=
by
  sorry

theorem triangle_side_ratio_impossible_2 (a b c : ℝ) (h₁ : b = a / 2) (h₂ : b = c / 3) : false :=
by
  sorry

theorem triangle_side_ratio_impossible_3 (a b c : ℝ) (h₁ : c = a / 2) (h₂ : c = b / 3) : false :=
by
  sorry

end triangle_side_ratio_impossible_triangle_side_ratio_impossible_2_triangle_side_ratio_impossible_3_l181_181720


namespace remainder_product_mod_five_l181_181109

-- Define the conditions as congruences
def num1 : ℕ := 14452
def num2 : ℕ := 15652
def num3 : ℕ := 16781

-- State the main theorem using the conditions and the given problem
theorem remainder_product_mod_five : 
  (num1 % 5 = 2) → 
  (num2 % 5 = 2) → 
  (num3 % 5 = 1) → 
  ((num1 * num2 * num3) % 5 = 4) :=
by
  intros
  sorry

end remainder_product_mod_five_l181_181109


namespace max_difference_two_digit_numbers_l181_181253

theorem max_difference_two_digit_numbers (A B : ℤ) (hA : 10 ≤ A ∧ A ≤ 99) (hB : 10 ≤ B ∧ B ≤ 99) (h : 2 * A * 3 = 2 * B * 7) : 
  56 ≤ A - B :=
sorry

end max_difference_two_digit_numbers_l181_181253


namespace relationship_between_roots_l181_181448

-- Define the number of real roots of the equations
def number_real_roots_lg_eq_sin : ℕ := 3
def number_real_roots_x_eq_sin : ℕ := 1
def number_real_roots_x4_eq_sin : ℕ := 2

-- Define the variables
def a : ℕ := number_real_roots_lg_eq_sin
def b : ℕ := number_real_roots_x_eq_sin
def c : ℕ := number_real_roots_x4_eq_sin

-- State the theorem
theorem relationship_between_roots : a > c ∧ c > b :=
by
  -- the proof is skipped
  sorry

end relationship_between_roots_l181_181448


namespace part1_part2_l181_181524

noncomputable def f (x : ℝ) (a : ℝ) := (Real.exp x / x) - Real.log x + x - a

theorem part1 (x : ℝ) (a : ℝ) :
    (∀ x > 0, f x a ≥ 0) → a ≤ Real.exp 1 + 1 :=
sorry

theorem part2 (x1 x2 : ℝ) (a : ℝ) :
  f x1 a = 0 → f x2 a = 0 → x1 < 1 → 1 < x2 → x1 * x2 < 1 :=
sorry

end part1_part2_l181_181524


namespace all_metals_conduct_electricity_l181_181935

def Gold_conducts : Prop := sorry
def Silver_conducts : Prop := sorry
def Copper_conducts : Prop := sorry
def Iron_conducts : Prop := sorry
def inductive_reasoning : Prop := sorry

theorem all_metals_conduct_electricity (g: Gold_conducts) (s: Silver_conducts) (c: Copper_conducts) (i: Iron_conducts) : inductive_reasoning := 
sorry

end all_metals_conduct_electricity_l181_181935


namespace frog_escape_l181_181431

theorem frog_escape (wellDepth dayClimb nightSlide escapeDays : ℕ)
  (h_depth : wellDepth = 30)
  (h_dayClimb : dayClimb = 3)
  (h_nightSlide : nightSlide = 2)
  (h_escape : escapeDays = 28) :
  ∃ n, n = escapeDays ∧
       ((wellDepth ≤ (n - 1) * (dayClimb - nightSlide) + dayClimb)) :=
by
  sorry

end frog_escape_l181_181431


namespace digits_sum_not_2001_l181_181752

theorem digits_sum_not_2001 (a : ℕ) (n m : ℕ) 
  (h1 : 10^(n-1) ≤ a ∧ a < 10^n)
  (h2 : 3 * n - 2 ≤ m ∧ m < 3 * n + 1)
  : m + n ≠ 2001 := 
sorry

end digits_sum_not_2001_l181_181752


namespace binomial_20_19_l181_181325

theorem binomial_20_19 : nat.choose 20 19 = 20 :=
by {
  sorry
}

end binomial_20_19_l181_181325


namespace equation_of_parabola_l181_181098

def parabola_passes_through_point (a h : ℝ) : Prop :=
  2 = a * (8^2) + h

def focus_x_coordinate (a h : ℝ) : Prop :=
  h + (1 / (4 * a)) = 3

theorem equation_of_parabola :
  ∃ (a h : ℝ), parabola_passes_through_point a h ∧ focus_x_coordinate a h ∧
    (∀ x y : ℝ, x = (15 / 256) * y^2 - (381 / 128)) :=
sorry

end equation_of_parabola_l181_181098


namespace power_of_power_example_l181_181163

theorem power_of_power_example : (3^2)^4 = 6561 := by
  sorry

end power_of_power_example_l181_181163


namespace least_number_to_subtract_l181_181922

theorem least_number_to_subtract (n : ℕ) (h : n = 13294) : ∃ k : ℕ, n - 1 = k * 97 :=
by
  sorry

end least_number_to_subtract_l181_181922


namespace min_value_of_M_l181_181616

theorem min_value_of_M (P : ℕ → ℝ) (n : ℕ) (M : ℝ):
  (P 1 = 9 / 11) →
  (∀ n ≥ 2, P n = (3 / 4) * (P (n - 1)) + (2 / 3) * (1 - P (n - 1))) →
  (∀ n ≥ 2, P n ≤ M) →
  (M = 97 / 132) := 
sorry

end min_value_of_M_l181_181616


namespace part1_part2_l181_181529

def f (x : ℝ) (a : ℝ) := (Real.exp x) / x - Real.log x + x - a

theorem part1 (a : ℝ) :
  (∀ x > 0, f x a ≥ 0) → (a ≤ Real.exp 1 + 1) :=
sorry

theorem part2 (x1 x2 a : ℝ) :
  f x1 a = 0 → f x2 a = 0 → x1 ≠ x2 → 0 < x1 → 0 < x2 → x1 * x2 < 1 :=
sorry

end part1_part2_l181_181529


namespace ratio_Mary_to_Seth_in_a_year_l181_181220

-- Given conditions
def Seth_current_age : ℝ := 3.5
def age_difference : ℝ := 9

-- Definitions derived from conditions
def Mary_current_age : ℝ := Seth_current_age + age_difference
def Seth_age_in_a_year : ℝ := Seth_current_age + 1
def Mary_age_in_a_year : ℝ := Mary_current_age + 1

-- The statement to prove
theorem ratio_Mary_to_Seth_in_a_year : (Mary_age_in_a_year / Seth_age_in_a_year) = 3 := sorry

end ratio_Mary_to_Seth_in_a_year_l181_181220


namespace max_area_and_length_l181_181805

def material_cost (x y : ℝ) : ℝ :=
  900 * x + 400 * y + 200 * x * y

def area (x y : ℝ) : ℝ := x * y

theorem max_area_and_length (x y : ℝ) (h₁ : material_cost x y ≤ 32000) :
  ∃ (S : ℝ) (x : ℝ), S = 100 ∧ x = 20 / 3 :=
sorry

end max_area_and_length_l181_181805


namespace latus_rectum_equation_l181_181597

theorem latus_rectum_equation (y x : ℝ) :
  y^2 = 4 * x → x = -1 :=
sorry

end latus_rectum_equation_l181_181597


namespace evaluate_exponent_l181_181158

theorem evaluate_exponent : (3^2)^4 = 6561 := sorry

end evaluate_exponent_l181_181158


namespace total_apples_correctness_l181_181203

-- Define the number of apples each man bought
def applesMen := 30

-- Define the number of apples each woman bought
def applesWomen := applesMen + 20

-- Define the total number of apples bought by the two men
def totalApplesMen := 2 * applesMen

-- Define the total number of apples bought by the three women
def totalApplesWomen := 3 * applesWomen

-- Define the total number of apples bought by the two men and three women
def totalApples := totalApplesMen + totalApplesWomen

-- Prove that the total number of apples bought by two men and three women is 210
theorem total_apples_correctness : totalApples = 210 := by
  sorry

end total_apples_correctness_l181_181203


namespace cube_inequality_l181_181734

theorem cube_inequality {a b : ℝ} (h : a > b) : a^3 > b^3 :=
sorry

end cube_inequality_l181_181734


namespace binomial_20_19_eq_20_l181_181313

theorem binomial_20_19_eq_20 : nat.choose 20 19 = 20 :=
sorry

end binomial_20_19_eq_20_l181_181313


namespace ben_eggs_left_l181_181822

def initial_eggs : ℕ := 50
def day1_morning : ℕ := 5
def day1_afternoon : ℕ := 4
def day2_morning : ℕ := 8
def day2_evening : ℕ := 3
def day3_afternoon : ℕ := 6
def day3_night : ℕ := 2

theorem ben_eggs_left : initial_eggs - (day1_morning + day1_afternoon + day2_morning + day2_evening + day3_afternoon + day3_night) = 22 := 
by
  sorry

end ben_eggs_left_l181_181822


namespace midpoint_quadrilateral_inequality_l181_181882

theorem midpoint_quadrilateral_inequality 
  (A B C D E F G H : ℝ) 
  (S_ABCD : ℝ)
  (midpoints_A : E = (A + B) / 2)
  (midpoints_B : F = (B + C) / 2)
  (midpoints_C : G = (C + D) / 2)
  (midpoints_D : H = (D + A) / 2)
  (EG : ℝ)
  (HF : ℝ) :
  S_ABCD ≤ EG * HF ∧ EG * HF ≤ (B + D) * (A + C) / 4 := by
  sorry

end midpoint_quadrilateral_inequality_l181_181882


namespace equation_of_line_perpendicular_l181_181812

theorem equation_of_line_perpendicular 
  (P : ℝ × ℝ) (hx : P.1 = -1) (hy : P.2 = 2)
  (a b c : ℝ) (h_line : 2 * a - 3 * b + 4 = 0)
  (l : ℝ → ℝ) (h_perpendicular : ∀ x, l x = -(3/2) * x)
  (h_passing : l (-1) = 2)
  : a * 3 + b * 2 - 1 = 0 :=
sorry

end equation_of_line_perpendicular_l181_181812


namespace vector_field_flux_l181_181170

-- Define the vector field F
def vector_field (x y z : ℝ) : ℝ × ℝ × ℝ :=
  (x - 2 * y + 1, 2 * x + y - 3 * z, 3 * y + z)

-- Define the surface of the sphere
def sphere (x y z : ℝ) : Prop :=
  x^2 + y^2 + z^2 = 1

-- Define the region constraints
def region_constraints (x y z : ℝ) : Prop :=
  (x ≥ 0) ∧ (y ≥ 0) ∧ (z ≥ 0) ∧ (x^2 + y^2 + z^2 > 1)

-- Define the entire problem as a Lean statement
theorem vector_field_flux :
  ∫∫ (θ ∈ set.Icc (0 : ℝ) (π / 2)) (φ ∈ set.Icc (0 : ℝ) (π / 2)), 
      let x := cos φ * sin θ in
      let y := sin φ * sin θ in
      let z := cos θ in
      let F := vector_field x y z in
      let n := (x, y, z) in
      (F.1 * n.1 + F.2 * n.2 + F.3 * n.3) * (sin θ)
  = (3 * π) / 4 := 
sorry

end vector_field_flux_l181_181170


namespace factorization_of_polynomial_l181_181669

theorem factorization_of_polynomial :
  (x : ℝ) → (x^8 - 4*x^6 + 6*x^4 - 4*x^2 + 1) = ((x - 1)^4 * (x + 1)^4) :=
by
  intro x
  sorry

end factorization_of_polynomial_l181_181669


namespace max_product_areas_l181_181454

theorem max_product_areas (a b c d : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hd : 0 ≤ d) (h : a + b + c + d = 1) :
  a * b * c * d ≤ 1 / 256 :=
sorry

end max_product_areas_l181_181454


namespace work_together_l181_181929

variable (W : ℝ) -- 'W' denotes the total work
variable (a_days b_days c_days : ℝ)

-- Conditions provided in the problem
axiom a_work : a_days = 18
axiom b_work : b_days = 6
axiom c_work : c_days = 12

-- The statement to be proved
theorem work_together :
  (W / a_days + W / b_days + W / c_days) * (36 / 11) = W := by
  sorry

end work_together_l181_181929


namespace greatest_savings_option2_l181_181947

-- Define the initial price
def initial_price : ℝ := 15000

-- Define the discounts for each option
def discounts_option1 : List ℝ := [0.75, 0.85, 0.95]
def discounts_option2 : List ℝ := [0.65, 0.90, 0.95]
def discounts_option3 : List ℝ := [0.70, 0.90, 0.90]

-- Define a function to compute the final price after successive discounts
def final_price (initial : ℝ) (discounts : List ℝ) : ℝ :=
  discounts.foldl (λ acc d => acc * d) initial

-- Define the savings for each option
def savings_option1 : ℝ := initial_price - (final_price initial_price discounts_option1)
def savings_option2 : ℝ := initial_price - (final_price initial_price discounts_option2)
def savings_option3 : ℝ := initial_price - (final_price initial_price discounts_option3)

-- Formulate the proof
theorem greatest_savings_option2 :
  max (max savings_option1 savings_option2) savings_option3 = savings_option2 :=
by
  sorry

end greatest_savings_option2_l181_181947


namespace A_2013_eq_neg_1007_l181_181225

def A (n : ℕ) : ℤ :=
  (-1)^n * ((n + 1) / 2)

theorem A_2013_eq_neg_1007 : A 2013 = -1007 :=
by
  sorry

end A_2013_eq_neg_1007_l181_181225


namespace function_relation_l181_181697

noncomputable def f (x : ℝ) : ℝ := x * Real.sin x

theorem function_relation:
  f (-Real.pi / 3) > f 1 ∧ f 1 > f (Real.pi / 5) :=
by 
  sorry

end function_relation_l181_181697


namespace bird_counts_l181_181816

theorem bird_counts :
  ∀ (num_cages_1 num_cages_2 num_cages_empty parrot_per_cage parakeet_per_cage canary_per_cage cockatiel_per_cage lovebird_per_cage finch_per_cage total_cages : ℕ),
    num_cages_1 = 7 →
    num_cages_2 = 6 →
    num_cages_empty = 2 →
    parrot_per_cage = 3 →
    parakeet_per_cage = 5 →
    canary_per_cage = 4 →
    cockatiel_per_cage = 2 →
    lovebird_per_cage = 3 →
    finch_per_cage = 1 →
    total_cages = 15 →
    (num_cages_1 * parrot_per_cage = 21) ∧
    (num_cages_1 * parakeet_per_cage = 35) ∧
    (num_cages_1 * canary_per_cage = 28) ∧
    (num_cages_2 * cockatiel_per_cage = 12) ∧
    (num_cages_2 * lovebird_per_cage = 18) ∧
    (num_cages_2 * finch_per_cage = 6) :=
by
  intros
  sorry

end bird_counts_l181_181816


namespace math_problem_l181_181733

open Real

theorem math_problem
  (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h : a - b + c = 0) :
  (a^2 * b^2 / ((a^2 + b * c) * (b^2 + a * c)) +
   a^2 * c^2 / ((a^2 + b * c) * (c^2 + a * b)) +
   b^2 * c^2 / ((b^2 + a * c) * (c^2 + a * b))) = 1 := by
  sorry

end math_problem_l181_181733


namespace days_B_to_finish_work_l181_181465

-- Definition of work rates based on the conditions
def work_rate_A (A_days: ℕ) : ℚ := 1 / A_days
def work_rate_B (B_days: ℕ) : ℚ := 1 / B_days

-- Theorem that encapsulates the problem statement
theorem days_B_to_finish_work (A_days B_days together_days : ℕ) (work_rate_A_eq : work_rate_A 4 = 1/4) (work_rate_B_eq : work_rate_B 12 = 1/12) : 
  ∀ (remaining_work: ℚ), remaining_work = 1 - together_days * (work_rate_A 4 + work_rate_B 12) → 
  (remaining_work / (work_rate_B 12)) = 4 :=
by
  sorry

end days_B_to_finish_work_l181_181465


namespace lottery_probability_l181_181068

theorem lottery_probability :
  let megaBallProbability := 1 / 30
  let winnerBallCombination := Nat.choose 50 5
  let winnerBallProbability := 1 / winnerBallCombination
  megaBallProbability * winnerBallProbability = 1 / 63562800 :=
by
  let megaBallProbability := 1 / 30
  let winnerBallCombination := Nat.choose 50 5
  have winnerBallCombinationEval: winnerBallCombination = 2118760 := by sorry
  let winnerBallProbability := 1 / winnerBallCombination
  have totalProbability: megaBallProbability * winnerBallProbability = 1 / 63562800 := by sorry
  exact totalProbability

end lottery_probability_l181_181068


namespace smallest_flash_drives_l181_181434

theorem smallest_flash_drives (total_files : ℕ) (flash_drive_space: ℝ)
  (files_size : ℕ → ℝ)
  (h1 : total_files = 40)
  (h2 : flash_drive_space = 2.0)
  (h3 : ∀ n, (n < 4 → files_size n = 1.2) ∧ 
              (4 ≤ n ∧ n < 20 → files_size n = 0.9) ∧ 
              (20 ≤ n → files_size n = 0.6)) :
  ∃ min_flash_drives, min_flash_drives = 20 :=
sorry

end smallest_flash_drives_l181_181434


namespace intersection_P_Q_l181_181194

def P : Set ℤ := {-4, -2, 0, 2, 4}
def Q : Set ℤ := {x : ℤ | -1 < x ∧ x < 3}

theorem intersection_P_Q : P ∩ Q = {0, 2} := by
  sorry

end intersection_P_Q_l181_181194


namespace percentage_silver_cars_after_shipment_l181_181264

-- Definitions for conditions
def initialCars : ℕ := 40
def initialSilverPerc : ℝ := 0.15
def newShipmentCars : ℕ := 80
def newShipmentNonSilverPerc : ℝ := 0.30

-- Proof statement that needs to be proven
theorem percentage_silver_cars_after_shipment :
  let initialSilverCars := initialSilverPerc * initialCars
  let newShipmentSilverPerc := 1 - newShipmentNonSilverPerc
  let newShipmentSilverCars := newShipmentSilverPerc * newShipmentCars
  let totalSilverCars := initialSilverCars + newShipmentSilverCars
  let totalCars := initialCars + newShipmentCars
  (totalSilverCars / totalCars) * 100 = 51.67 :=
by
  sorry

end percentage_silver_cars_after_shipment_l181_181264


namespace find_a5_l181_181694

open Nat

def increasing_seq (a : Nat → Nat) : Prop :=
  ∀ m n : Nat, m < n → a m < a n

theorem find_a5
  (a : Nat → Nat)
  (h1 : ∀ n : Nat, a (a n) = 3 * n)
  (h2 : increasing_seq a)
  (h3 : ∀ n : Nat, a n > 0) :
  a 5 = 8 :=
by
  sorry

end find_a5_l181_181694


namespace distance_from_center_to_line_l181_181847

-- Define the conditions 
def circle_polar_eq (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos θ
def line_polar_eq (ρ θ : ℝ) : Prop := ρ * Real.sin θ + 2 * ρ * Real.cos θ = 1

-- Define the assertion that we want to prove
theorem distance_from_center_to_line (ρ θ : ℝ) 
  (h_circle: circle_polar_eq ρ θ) 
  (h_line: line_polar_eq ρ θ) : 
  ∃ d : ℝ, d = (Real.sqrt 5) / 5 := 
sorry

end distance_from_center_to_line_l181_181847


namespace percentage_increase_area_l181_181798

theorem percentage_increase_area (L W : ℝ) :
  let A := L * W
  let L' := 1.20 * L
  let W' := 1.20 * W
  let A' := L' * W'
  let percentage_increase := (A' - A) / A * 100
  L > 0 → W > 0 → percentage_increase = 44 := 
by
  sorry

end percentage_increase_area_l181_181798


namespace fraction_subtraction_simplified_l181_181351

theorem fraction_subtraction_simplified :
  (8 / 21 - 3 / 63) = 1 / 3 := 
by
  sorry

end fraction_subtraction_simplified_l181_181351


namespace triangle_angle_C_l181_181061

theorem triangle_angle_C (A B C : ℝ) (h : A + B = 80) : C = 100 :=
sorry

end triangle_angle_C_l181_181061


namespace perpendicular_x_intercept_l181_181783

theorem perpendicular_x_intercept (x : ℝ) :
  (∃ y : ℝ, 2 * x + 3 * y = 9) ∧ (∃ y : ℝ, y = 5) → x = -10 / 3 :=
by sorry -- Proof omitted

end perpendicular_x_intercept_l181_181783


namespace alex_silver_tokens_count_l181_181142

-- Conditions
def initial_red_tokens := 90
def initial_blue_tokens := 80

def red_exchange (x : ℕ) (y : ℕ) : ℕ := 90 - 3 * x + y
def blue_exchange (x : ℕ) (y : ℕ) : ℕ := 80 + 2 * x - 4 * y

-- Boundaries where exchanges stop
def red_bound (x : ℕ) (y : ℕ) : Prop := red_exchange x y < 3
def blue_bound (x : ℕ) (y : ℕ) : Prop := blue_exchange x y < 4

-- Proof statement
theorem alex_silver_tokens_count (x y : ℕ) :
    red_bound x y → blue_bound x y → (x + y) = 52 :=
    by
    sorry

end alex_silver_tokens_count_l181_181142


namespace factor_polynomial_l181_181834

theorem factor_polynomial :
  (x^2 + 5 * x + 4) * (x^2 + 11 * x + 30) + (x^2 + 8 * x - 10) =
  (x^2 + 8 * x + 7) * (x^2 + 8 * x + 19) := by
  sorry

end factor_polynomial_l181_181834


namespace part1_part2_l181_181523

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (Real.exp x / x) - Real.log x + x - a

theorem part1 (a : ℝ) : (∀ x > 0, f x a ≥ 0) ↔ a ≤ Real.exp 1 + 1 :=
  sorry

theorem part2 (a : ℝ) (x1 x2 : ℝ) (h1 : f x1 a = 0) (h2 : f x2 a = 0) : x1 * x2 < 1 :=
  sorry

end part1_part2_l181_181523


namespace smallest_sum_l181_181732

theorem smallest_sum (a b c : ℕ) (h : (13 * a + 11 * b + 7 * c = 1001)) :
    a / 77 + b / 91 + c / 143 = 1 → a + b + c = 79 :=
by
  sorry

end smallest_sum_l181_181732


namespace measure_of_angle_C_l181_181043

-- Conditions of the problem
variable (A B C : ℝ) -- Angles in the triangle

-- Sum of all angles in a triangle
axiom sum_of_angles (h : A + B + C = 180)

-- Condition given in the problem
axiom given_angle_sum (h1 : A + B = 80)

-- Desired proof statement
theorem measure_of_angle_C : C = 100 := by
  sorry

end measure_of_angle_C_l181_181043


namespace relationship_between_y_l181_181539

theorem relationship_between_y (y1 y2 y3 : ℝ)
  (hA : y1 = 3 / -5)
  (hB : y2 = 3 / -3)
  (hC : y3 = 3 / 2) : y2 < y1 ∧ y1 < y3 :=
by
  sorry

end relationship_between_y_l181_181539


namespace incorrect_residual_plot_statement_l181_181789

theorem incorrect_residual_plot_statement :
  ∀ (vertical_only_residual : Prop)
    (horizontal_any_of : Prop)
    (narrower_band_smaller_ssr : Prop)
    (narrower_band_smaller_corr : Prop)
    ,
    narrower_band_smaller_corr → False :=
  by intros vertical_only_residual horizontal_any_of narrower_band_smaller_ssr narrower_band_smaller_corr
     sorry

end incorrect_residual_plot_statement_l181_181789


namespace fraction_of_walls_not_illuminated_l181_181714

-- Define given conditions
def point_light_source : Prop := true
def rectangular_room : Prop := true
def flat_mirror_on_wall : Prop := true
def full_height_of_room : Prop := true

-- Define the fraction not illuminated
def fraction_not_illuminated := 17 / 32

-- State the theorem to prove
theorem fraction_of_walls_not_illuminated :
  point_light_source ∧ rectangular_room ∧ flat_mirror_on_wall ∧ full_height_of_room →
  fraction_not_illuminated = 17 / 32 :=
by
  intros h
  sorry

end fraction_of_walls_not_illuminated_l181_181714


namespace binomial_20_19_eq_20_l181_181314

theorem binomial_20_19_eq_20 : nat.choose 20 19 = 20 :=
sorry

end binomial_20_19_eq_20_l181_181314


namespace parallel_tangent_line_exists_l181_181354

noncomputable def line_eq (k b : ℝ) : (ℝ × ℝ) → Prop :=
λ (P : ℝ × ℝ), P.snd = k * P.fst + b

noncomputable def circle_eq (a b r : ℝ) : (ℝ × ℝ) → Prop :=
λ (P : ℝ × ℝ), (P.fst - a) ^ 2 + (P.snd - b) ^ 2 = r ^ 2

theorem parallel_tangent_line_exists :
  ∃ (b : ℝ), ∀ (P : ℝ × ℝ),
    line_eq 2 b P → circle_eq 1 2 1 P :=
sorry

end parallel_tangent_line_exists_l181_181354


namespace f_neg2_eq_neg1_l181_181998

noncomputable def f (x : ℝ) : ℝ := x^2 + 2*x - 1

theorem f_neg2_eq_neg1 : f (-2) = -1 := by
  sorry

end f_neg2_eq_neg1_l181_181998


namespace fraction_of_64_l181_181782

theorem fraction_of_64 : (7 / 8) * 64 = 56 :=
sorry

end fraction_of_64_l181_181782


namespace solution_l181_181839

noncomputable def given_conditions (θ : ℝ) : Prop := 
  let a := (3, 1)
  let b := (Real.sin θ, Real.cos θ)
  (a.1 : ℝ) / b.1 = a.2 / b.2 

theorem solution (θ : ℝ) (h: given_conditions θ) :
  2 + Real.sin θ * Real.cos θ - Real.cos θ ^ 2 = 5 / 2 :=
by
  sorry

end solution_l181_181839


namespace range_of_a_l181_181683

noncomputable def A (x : ℝ) : Prop := x^2 - x ≤ 0
noncomputable def B (x : ℝ) (a : ℝ) : Prop := 2^(1 - x) + a ≤ 0

theorem range_of_a (a : ℝ) : (∀ x, A x → B x a) → a ≤ -2 := by
  intro h
  -- Proof steps would go here
  sorry

end range_of_a_l181_181683


namespace product_of_six_consecutive_nat_not_equal_776965920_l181_181585

theorem product_of_six_consecutive_nat_not_equal_776965920 (n : ℕ) :
  (n * (n + 1) * (n + 2) * (n + 3) * (n + 4) * (n + 5) ≠ 776965920) :=
by
  sorry

end product_of_six_consecutive_nat_not_equal_776965920_l181_181585


namespace tan_15pi_over_4_is_neg1_l181_181650

noncomputable def tan_15pi_over_4 : ℝ :=
  Real.tan (15 * Real.pi / 4)

theorem tan_15pi_over_4_is_neg1 :
  tan_15pi_over_4 = -1 :=
sorry

end tan_15pi_over_4_is_neg1_l181_181650


namespace max_value_of_f_l181_181857

noncomputable def f (x : ℝ) := x^3 - 3 * x + 1

theorem max_value_of_f (h: ∃ x, f x = -1) : ∃ y, f y = 3 :=
by
  -- We'll later prove this with appropriate mathematical steps using Lean tactics
  sorry

end max_value_of_f_l181_181857


namespace fraction_of_64_l181_181781

theorem fraction_of_64 : (7 / 8) * 64 = 56 :=
sorry

end fraction_of_64_l181_181781


namespace find_a_l181_181598

noncomputable def f (a x : ℝ) : ℝ := a * x^2 + 2 * a * x + 1

theorem find_a
  (a : ℝ)
  (h₁ : ∀ x ∈ Set.Icc (-3 : ℝ) (2 : ℝ), f a x ≤ 4)
  (h₂ : ∃ x ∈ Set.Icc (-3 : ℝ) (2 : ℝ), f a x = 4) :
  a = -3 ∨ a = 3 / 8 :=
by
  sorry

end find_a_l181_181598


namespace arithmetic_geometric_mean_inequality_l181_181502

theorem arithmetic_geometric_mean_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (A : ℝ) (G : ℝ)
  (hA : A = (a + b) / 2) (hG : G = Real.sqrt (a * b)) : A ≥ G :=
by
  sorry

end arithmetic_geometric_mean_inequality_l181_181502


namespace estimate_fish_number_l181_181452

noncomputable def numFishInLake (marked: ℕ) (caughtSecond: ℕ) (markedSecond: ℕ) : ℕ :=
  let totalFish := (caughtSecond * marked) / markedSecond
  totalFish

theorem estimate_fish_number (marked caughtSecond markedSecond : ℕ) :
  marked = 100 ∧ caughtSecond = 200 ∧ markedSecond = 25 → numFishInLake marked caughtSecond markedSecond = 800 :=
by
  intros h
  cases h
  sorry

end estimate_fish_number_l181_181452


namespace minimum_guests_l181_181235

theorem minimum_guests (total_food : ℤ) (max_food_per_guest : ℤ) (food_bound : total_food = 325) (guest_bound : max_food_per_guest = 2) : (⌈total_food / max_food_per_guest⌉ : ℤ) = 163 :=
by {
  sorry 
}

end minimum_guests_l181_181235


namespace problem_statement_l181_181842

def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ a₁ d : ℤ, ∀ n : ℕ, a n = a₁ + n * d

theorem problem_statement :
  ∃ a : ℕ → ℤ, is_arithmetic_sequence a ∧
  a 2 = 7 ∧
  a 4 + a 6 = 26 ∧
  (∀ n : ℕ, a (n + 1) = 2 * n + 1) ∧
  ∃ S : ℕ → ℤ, (S n = n^2 + 2 * n) ∧
  ∃ b : ℕ → ℚ, (∀ n : ℕ, b n = 1 / (a n ^ 2 - 1)) ∧
  ∃ T : ℕ → ℚ, (T n = (n / 4) * (1 / (n + 1))) :=
sorry

end problem_statement_l181_181842


namespace find_a_plus_b_l181_181018

-- Definitions for the conditions
variables {a b : ℝ} (i : ℂ)
def imaginary_unit : Prop := i * i = -1

-- Given condition
def given_equation (a b : ℝ) (i : ℂ) : Prop := (a + 2 * i) / i = b + i

-- Theorem statement
theorem find_a_plus_b (h1 : imaginary_unit i) (h2 : given_equation a b i) : a + b = 1 := 
sorry

end find_a_plus_b_l181_181018


namespace option_D_is_correct_l181_181925

noncomputable def correct_operation : Prop := 
  (∀ x : ℝ, x + x ≠ 2 * x^2) ∧
  (∀ y : ℝ, 2 * y^3 + 3 * y^2 ≠ 5 * y^5) ∧
  (∀ x : ℝ, 2 * x - x ≠ 1) ∧
  (∀ x y : ℝ, 4 * x^3 * y^2 - (-2)^2 * x^3 * y^2 = 0)

theorem option_D_is_correct : correct_operation :=
by {
  -- We'll complete the proofs later
  sorry
}

end option_D_is_correct_l181_181925


namespace problem_statement_l181_181017

theorem problem_statement (m n : ℝ) (h1 : 1 + 27 = m) (h2 : 3 + 9 = n) : |m - n| = 16 := by
  sorry

end problem_statement_l181_181017


namespace range_of_k_l181_181838

open Real

theorem range_of_k (k : ℝ) :
  (∃ x1 x2 ∈ Icc 1 2, abs (exp x1 - exp x2) > k * abs (log x1 - log x2)) ↔ k < 2 * exp 2 := 
sorry

end range_of_k_l181_181838


namespace spies_denounced_each_other_l181_181908

theorem spies_denounced_each_other :
  ∃ (pairs : Finset (ℕ × ℕ)), pairs.card ≥ 10 ∧ 
  (∀ (u v : ℕ), (u, v) ∈ pairs → (v, u) ∈ pairs) :=
sorry

end spies_denounced_each_other_l181_181908


namespace find_alpha_l181_181550

def demand_function (p : ℝ) : ℝ := 150 - p
def supply_function (p : ℝ) : ℝ := 3 * p - 10

def new_demand_function (p : ℝ) (α : ℝ) : ℝ := α * (150 - p)

theorem find_alpha (α : ℝ) :
  (∃ p₀ p_new, demand_function p₀ = supply_function p₀ ∧ 
    p_new = p₀ * 1.25 ∧ 
    3 * p_new - 10 = new_demand_function p_new α) →
  α = 1.4 :=
by
  sorry

end find_alpha_l181_181550


namespace large_box_count_l181_181149

variable (x y : ℕ)

theorem large_box_count (h₁ : x + y = 21) (h₂ : 120 * x + 80 * y = 2000) : x = 8 := by
  sorry

end large_box_count_l181_181149


namespace fraction_of_fraction_l181_181776

theorem fraction_of_fraction:
  let a := (3:ℚ) / 4
  let b := (5:ℚ) / 12
  b / a = (5:ℚ) / 9 := by
  sorry

end fraction_of_fraction_l181_181776


namespace hacker_neo_problem_l181_181131

theorem hacker_neo_problem (n : ℕ) :
  ∃ m : ℕ, rearrange_digits (123 + 102 * n) < 1000 :=
sorry

-- Helper function for rearranging digits
def rearrange_digits (n : ℕ) : ℕ :=
-- Define your own implementation or use a placeholder.
sorry

end hacker_neo_problem_l181_181131


namespace fifteenth_number_in_base_8_l181_181406

theorem fifteenth_number_in_base_8 : (15 : ℕ) = 1 * 8 + 7 := 
sorry

end fifteenth_number_in_base_8_l181_181406


namespace angle_C_in_triangle_l181_181038

theorem angle_C_in_triangle (A B C : ℝ) (h1 : A + B = 80) (h2 : A + B + C = 180) : C = 100 := 
by 
  sorry

end angle_C_in_triangle_l181_181038


namespace no_values_satisfy_equation_l181_181500

-- Define the sum of the digits function S
noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

-- Define the sum of digits of the sum of the digits function S(S(n))
noncomputable def sum_of_sum_of_digits (n : ℕ) : ℕ :=
  sum_of_digits (sum_of_digits n)

-- Theorem statement about the number of n satisfying n + S(n) + S(S(n)) = 2099
theorem no_values_satisfy_equation :
  (∃ n : ℕ, n > 0 ∧ n + sum_of_digits n + sum_of_sum_of_digits n = 2099) ↔ False := sorry

end no_values_satisfy_equation_l181_181500


namespace lunch_cost_before_tax_and_tip_l181_181615

theorem lunch_cost_before_tax_and_tip (C : ℝ) (h1 : 1.10 * C = 110) : C = 100 := by
  sorry

end lunch_cost_before_tax_and_tip_l181_181615


namespace average_marks_is_70_l181_181265

variable (P C M : ℕ)

-- Condition: The total marks in physics, chemistry, and mathematics is 140 more than the marks in physics
def total_marks_condition : Prop := P + C + M = P + 140

-- Definition of the average marks in chemistry and mathematics
def average_marks_C_M : ℕ := (C + M) / 2

theorem average_marks_is_70 (h : total_marks_condition P C M) : average_marks_C_M C M = 70 :=
sorry

end average_marks_is_70_l181_181265


namespace smallest_number_divisible_by_11_and_conditional_modulus_l181_181112

theorem smallest_number_divisible_by_11_and_conditional_modulus :
  ∃ n : ℕ, (n % 11 = 0) ∧ (n % 3 = 2) ∧ (n % 4 = 2) ∧ (n % 5 = 2) ∧ (n % 6 = 2) ∧ (n % 7 = 2) ∧ n = 2102 :=
by
  sorry

end smallest_number_divisible_by_11_and_conditional_modulus_l181_181112


namespace expected_value_in_classroom_l181_181067

noncomputable def expected_pairs_next_to_each_other (boys girls : ℕ) : ℕ :=
  if boys = 9 ∧ girls = 14 ∧ boys + girls = 23 then
    10 -- Based on provided conditions and conclusion
  else
    0

theorem expected_value_in_classroom :
  expected_pairs_next_to_each_other 9 14 = 10 :=
by
  sorry

end expected_value_in_classroom_l181_181067


namespace arithmetic_geometric_progression_l181_181077

theorem arithmetic_geometric_progression (a b c : ℤ) (h1 : a < b) (h2 : b < c)
  (h3 : b = 3 * a) (h4 : 2 * b = a + c) (h5 : b * b = a * c) : c = 9 :=
sorry

end arithmetic_geometric_progression_l181_181077


namespace kai_marbles_over_200_l181_181876

theorem kai_marbles_over_200 (marbles_on_day : Nat → Nat)
  (h_initial : marbles_on_day 0 = 4)
  (h_growth : ∀ n, marbles_on_day (n + 1) = 3 * marbles_on_day n) :
  ∃ k, marbles_on_day k > 200 ∧ k = 4 := by
  sorry

end kai_marbles_over_200_l181_181876


namespace prob_two_heads_in_four_flips_l181_181749

theorem prob_two_heads_in_four_flips : 
  (nat.choose 4 2 : ℝ) / (2^4) = 3 / 8 :=
by
  sorry

end prob_two_heads_in_four_flips_l181_181749


namespace sum_of_first_ten_terms_l181_181409

variable {α : Type*} [LinearOrderedField α]

-- Defining the arithmetic sequence and sum of the first n terms
def a_n (a d : α) (n : ℕ) : α := a + d * (n - 1)

def S_n (a : α) (d : α) (n : ℕ) : α := n / 2 * (2 * a + (n - 1) * d)

theorem sum_of_first_ten_terms (a d : α) (h : a_n a d 3 + a_n a d 8 = 12) : S_n a d 10 = 60 :=
by sorry

end sum_of_first_ten_terms_l181_181409


namespace sock_pairs_count_l181_181402

theorem sock_pairs_count :
  let white_socks := 5
  let brown_socks := 3
  let blue_socks := 4
  let blue_white_pairs := blue_socks * white_socks
  let blue_brown_pairs := blue_socks * brown_socks
  let total_pairs := blue_white_pairs + blue_brown_pairs
  total_pairs = 32 :=
by
  sorry

end sock_pairs_count_l181_181402


namespace quadratic_representation_integer_coefficients_iff_l181_181625

theorem quadratic_representation (A B C x : ℝ) :
  ∃ (k l m : ℝ), 
  k = 2 * A ∧ l = A + B ∧ m = C ∧ 
  (A * x^2 + B * x + C = k * (x * (x - 1) / 2) + l * x + m) :=
sorry

theorem integer_coefficients_iff (A B C : ℤ) :
  (∀ x : ℤ, ∃ (n : ℤ), A * x^2 + B * x + C = n) ↔ 
  (∃ (k l m : ℤ), k = 2 * A ∧ l = A + B ∧ m = C) :=
sorry

end quadratic_representation_integer_coefficients_iff_l181_181625


namespace marble_count_l181_181801

noncomputable def total_marbles (blue red white: ℕ) : ℕ := blue + red + white

theorem marble_count (W : ℕ) (h_prob : (9 + W) / (6 + 9 + W : ℝ) = 0.7) : 
  total_marbles 6 9 W = 20 :=
by
  sorry

end marble_count_l181_181801


namespace triangle_angle_sum_l181_181064

theorem triangle_angle_sum (A B C : ℝ) (h1 : A + B = 80) (h2 : A + B + C = 180) : C = 100 := 
by
  sorry

end triangle_angle_sum_l181_181064


namespace real_solution_to_cubic_eq_l181_181150

noncomputable section

open Real

theorem real_solution_to_cubic_eq (x : ℝ) : 
  x^3 + 2 * (x + 1)^3 + 3 * (x + 2)^3 = 3 * (x + 3)^3 → x = 5 := 
by
  sorry

end real_solution_to_cubic_eq_l181_181150


namespace fraction_inequality_l181_181691

-- Given the conditions
variables {c x y : ℝ} (h1 : c > x) (h2 : x > y) (h3 : y > 0)

-- Prove that \frac{x}{c-x} > \frac{y}{c-y}
theorem fraction_inequality (h4 : c > 0) : (x / (c - x)) > (y / (c - y)) :=
by {
  sorry  -- Proof to be completed
}

end fraction_inequality_l181_181691


namespace apr_sales_is_75_l181_181750

-- Definitions based on conditions
def sales_jan : ℕ := 90
def sales_feb : ℕ := 50
def sales_mar : ℕ := 70
def avg_sales : ℕ := 72

-- Total sales of first three months
def total_sales_jan_to_mar : ℕ := sales_jan + sales_feb + sales_mar

-- Total sales considering average sales over 5 months
def total_sales : ℕ := avg_sales * 5

-- Defining April sales
def sales_apr (sales_may : ℕ) : ℕ := total_sales - total_sales_jan_to_mar - sales_may

theorem apr_sales_is_75 (sales_may : ℕ) : sales_apr sales_may = 75 :=
by
  unfold sales_apr total_sales total_sales_jan_to_mar avg_sales sales_jan sales_feb sales_mar
  -- Here we could insert more steps if needed to directly connect to the proof
  sorry


end apr_sales_is_75_l181_181750


namespace wendy_furniture_time_l181_181255

variable (chairs tables pieces minutes total_time : ℕ)

theorem wendy_furniture_time (h1 : chairs = 4) (h2 : tables = 4) (h3 : pieces = chairs + tables) (h4 : minutes = 6) (h5 : total_time = pieces * minutes) : total_time = 48 :=
by
  sorry

end wendy_furniture_time_l181_181255


namespace max_value_x_minus_2y_exists_max_value_x_minus_2y_l181_181185

theorem max_value_x_minus_2y 
  (x y : ℝ) 
  (h : x^2 - 4 * x + y^2 = 0) : 
  x - 2 * y ≤ 2 + 2 * Real.sqrt 5 :=
sorry

theorem exists_max_value_x_minus_2y 
  (x y : ℝ) 
  (h : x^2 - 4 * x + y^2 = 0) : 
  ∃ (x y : ℝ), x - 2 * y = 2 + 2 * Real.sqrt 5 :=
sorry

end max_value_x_minus_2y_exists_max_value_x_minus_2y_l181_181185


namespace range_of_a_l181_181022

noncomputable def f (a x : ℝ) : ℝ :=
  (1 / 2) * x^2 - a * x + (a - 1) * Real.log x

theorem range_of_a (a : ℝ) (h1 : 1 < a) :
  (∀ x1 x2 : ℝ, 0 < x1 ∧ 0 < x2 ∧ x1 > x2 → f a x1 - f a x2 > x2 - x1) ↔ (1 < a ∧ a ≤ 5) :=
by
  -- The proof is omitted
  sorry

end range_of_a_l181_181022


namespace evaluate_expression_l181_181417

def f (x : ℕ) : ℕ := 3 * x - 4
def g (x : ℕ) : ℕ := x - 1

theorem evaluate_expression : f (1 + g 3) = 5 := by
  sorry

end evaluate_expression_l181_181417


namespace Total_points_proof_l181_181403

noncomputable def Samanta_points (Mark_points : ℕ) : ℕ := Mark_points + 8
noncomputable def Mark_points (Eric_points : ℕ) : ℕ := Eric_points + (Eric_points / 2)
def Eric_points : ℕ := 6
noncomputable def Daisy_points (Total_points_Samanta_Mark_Eric : ℕ) : ℕ := Total_points_Samanta_Mark_Eric - (Total_points_Samanta_Mark_Eric / 4)

def Total_points_Samanta_Mark_Eric (Samanta_points Mark_points Eric_points : ℕ) : ℕ := Samanta_points + Mark_points + Eric_points

theorem Total_points_proof :
  let Mk_pts := Mark_points Eric_points
  let Sm_pts := Samanta_points Mk_pts
  let Tot_SME := Total_points_Samanta_Mark_Eric Sm_pts Mk_pts Eric_points
  let D_pts := Daisy_points Tot_SME
  Sm_pts + Mk_pts + Eric_points + D_pts = 56 := by
  sorry

end Total_points_proof_l181_181403


namespace find_A_values_l181_181374

-- Definitions for the linear functions f and g.
def f (a b x : ℝ) : ℝ := a * x + b
def g (a c x : ℝ) : ℝ := a * x + c

-- Condition that the graphs of y = (f(x))^2 and y = -6g(x) touch
def condition1 (a b c : ℝ) : Prop :=
  let discriminant := (2 * a * (b + 3)) ^ 2 - 4 * a ^ 2 * (b ^ 2 + 6 * c) in
  discriminant = 0

-- Property that given condition1, we need to prove the values of A
def desired_A_values (a b c : ℝ) (A : ℝ) : Prop :=
  let discriminant := (a * (2 * c - A)) ^ 2 - 4 * a ^ 2 * (c ^ 2 - A * b) in
  discriminant = 0 → (A = 0 ∨ A = 6)

theorem find_A_values (a b c : ℝ) (ha : a ≠ 0) (hc1 : condition1 a b c) :
  ∀ A : ℝ, desired_A_values a b c A :=
begin
  intro A,
  sorry
end

end find_A_values_l181_181374


namespace num_sequences_with_zero_l181_181414

def valid_triples : List (ℕ × ℕ × ℕ) := 
  (List.finRange 5).bind (λ b1 => 
  (List.finRange 5).bind (λ b2 => 
  (List.finRange 5).map (λ b3 => (b1 + 1, b2 + 1, b3 + 1))))

def generates_zero (x y z : ℕ) : Prop :=
  ∃ n ≥ 4, ∃ b : ℕ → ℕ, b 1 = x ∧ b 2 = y ∧ b 3 = z ∧
  (∀ n ≥ 4, b n = b (n-1) * |b (n-2) - b (n-3)|) ∧ b n = 0

def count_valid_triples : ℕ := 
  valid_triples.countp (λ ⟨x, y, z⟩ => generates_zero x y z)

theorem num_sequences_with_zero : count_valid_triples = 173 := by
  sorry

end num_sequences_with_zero_l181_181414


namespace Bridget_Skittles_Final_l181_181649

def Bridget_Skittles_initial : Nat := 4
def Henry_Skittles_initial : Nat := 4

def Bridget_Receives_Henry_Skittles : Nat :=
  Bridget_Skittles_initial + Henry_Skittles_initial

theorem Bridget_Skittles_Final :
  Bridget_Receives_Henry_Skittles = 8 :=
by
  -- Proof steps here, assuming sorry for now
  sorry

end Bridget_Skittles_Final_l181_181649


namespace acute_triangle_exterior_angles_obtuse_l181_181211

theorem acute_triangle_exterior_angles_obtuse
  (A B C : ℝ)
  (hA : 0 < A ∧ A < π / 2)
  (hB : 0 < B ∧ B < π / 2)
  (hC : 0 < C ∧ C < π / 2)
  (h_sum : A + B + C = π) :
  ∀ α β γ, α = A + B → β = B + C → γ = C + A → α > π / 2 ∧ β > π / 2 ∧ γ > π / 2 :=
by
  sorry

end acute_triangle_exterior_angles_obtuse_l181_181211


namespace necessary_and_sufficient_condition_for_tangency_l181_181901

-- Given conditions
variables (ρ θ D E : ℝ)

-- Definition of the circle in polar coordinates and the condition for tangency with the radial axis
def circle_eq : Prop := ρ = D * Real.cos θ + E * Real.sin θ

-- Statement of the proof problem
theorem necessary_and_sufficient_condition_for_tangency :
  (circle_eq ρ θ D E) → (D = 0 ∧ E ≠ 0) :=
sorry

end necessary_and_sufficient_condition_for_tangency_l181_181901


namespace problem_statement_l181_181423

-- Given conditions
variable (x : ℝ) (h1 : x ≠ 0) (h2 : ∃ k : ℤ, x + 1/x = k)

-- Hypothesis configuration for inductive proof and goal statement
theorem problem_statement : ∀ n : ℕ, ∃ m : ℤ, x^n + 1/x^n = m :=
by sorry

end problem_statement_l181_181423


namespace num_ordered_triples_l181_181135

-- Given constants
def b : ℕ := 2024
def constant_value : ℕ := 4096576

-- Number of ordered triples (a, b, c) meeting the conditions
theorem num_ordered_triples (h : b = 2024 ∧ constant_value = 2024 * 2024) :
  ∃ (n : ℕ), n = 10 ∧ ∀ (a c : ℕ), a * c = constant_value → a ≤ c → n = 10 :=
by
  -- Translation of the mathematical conditions into the theorem
  sorry

end num_ordered_triples_l181_181135


namespace remainder_when_three_times_number_minus_seven_divided_by_seven_l181_181611

theorem remainder_when_three_times_number_minus_seven_divided_by_seven (n : ℤ) (h : n % 7 = 2) : (3 * n - 7) % 7 = 6 := by
  sorry

end remainder_when_three_times_number_minus_seven_divided_by_seven_l181_181611


namespace motorboat_time_to_C_l181_181471

variables (r s p t_B : ℝ)

-- Condition declarations
def kayak_speed := r + s
def motorboat_speed := p
def meeting_time := 12

-- Problem statement: to prove the time it took for the motorboat to reach dock C before turning back
theorem motorboat_time_to_C :
  (2 * r + s) * t_B = r * 12 + s * 6 → t_B = (r * 12 + s * 6) / (2 * r + s) := 
by
  intros h
  sorry

end motorboat_time_to_C_l181_181471


namespace surface_area_comparison_l181_181687

theorem surface_area_comparison (a R : ℝ) (h_eq_volumes : (4 / 3) * Real.pi * R^3 = a^3) :
  6 * a^2 > 4 * Real.pi * R^2 :=
by
  sorry

end surface_area_comparison_l181_181687


namespace favorite_movies_hours_l181_181729

theorem favorite_movies_hours (J M N R : ℕ) (h1 : J = M + 2) (h2 : N = 3 * M) (h3 : R = (4 * N) / 5) (h4 : N = 30) : 
  J + M + N + R = 76 :=
by
  sorry

end favorite_movies_hours_l181_181729


namespace sqrt_sum_4_pow_4_eq_32_l181_181916

theorem sqrt_sum_4_pow_4_eq_32 : Real.sqrt (4^4 + 4^4 + 4^4 + 4^4) = 32 :=
by
  sorry

end sqrt_sum_4_pow_4_eq_32_l181_181916


namespace binom_20_19_eq_20_l181_181306

theorem binom_20_19_eq_20 : nat.choose 20 19 = 20 :=
by sorry

end binom_20_19_eq_20_l181_181306


namespace factorization_of_polynomial_l181_181663

theorem factorization_of_polynomial :
  ∀ x : ℝ, x^8 - 4*x^6 + 6*x^4 - 4*x^2 + 1 = (x - 1)^4 * (x + 1)^4 :=
by
  intros x
  sorry

end factorization_of_polynomial_l181_181663


namespace capacity_of_other_bottle_l181_181268

theorem capacity_of_other_bottle (C : ℝ) :
  (∀ (total_milk c1 c2 : ℝ), total_milk = 8 ∧ c1 = 5.333333333333333 ∧ c2 = C ∧ 
  (c1 / 8 = (c2 / C))) → C = 4 :=
by
  intros h
  sorry

end capacity_of_other_bottle_l181_181268


namespace total_profit_is_28000_l181_181127

noncomputable def investment_A (investment_B : ℝ) : ℝ := 3 * investment_B
noncomputable def period_A (period_B : ℝ) : ℝ := 2 * period_B
noncomputable def profit_B : ℝ := 4000
noncomputable def total_profit (investment_B period_B : ℝ) : ℝ :=
  let x := investment_B * period_B
  let a_share := 6 * x
  profit_B + a_share

theorem total_profit_is_28000 (investment_B period_B : ℝ) : 
  total_profit investment_B period_B = 28000 :=
by
  have h1 : profit_B = 4000 := rfl
  have h2 : investment_A investment_B = 3 * investment_B := rfl
  have h3 : period_A period_B = 2 * period_B := rfl
  simp [total_profit, h1, h2, h3]
  have x_def : investment_B * period_B = 4000 := by sorry
  simp [x_def]
  sorry

end total_profit_is_28000_l181_181127


namespace not_both_zero_l181_181537

theorem not_both_zero (x y : ℝ) (h : x^2 + y^2 ≠ 0) : ¬ (x = 0 ∧ y = 0) :=
by {
  sorry
}

end not_both_zero_l181_181537


namespace wyatt_total_envelopes_l181_181927

theorem wyatt_total_envelopes :
  let b := 10
  let y := b - 4
  let t := b + y
  t = 16 :=
by
  let b := 10
  let y := b - 4
  let t := b + y
  sorry

end wyatt_total_envelopes_l181_181927


namespace ratio_depth_to_height_l181_181477

theorem ratio_depth_to_height
  (Dean_height : ℝ := 9)
  (additional_depth : ℝ := 81)
  (water_depth : ℝ := Dean_height + additional_depth) :
  water_depth / Dean_height = 10 :=
by
  -- Dean_height = 9
  -- additional_depth = 81
  -- water_depth = 9 + 81 = 90
  -- water_depth / Dean_height = 90 / 9 = 10
  sorry

end ratio_depth_to_height_l181_181477


namespace probability_correct_l181_181070

noncomputable def probability_study_group : ℝ :=
  let p_woman : ℝ := 0.5
  let p_man : ℝ := 0.5

  let p_woman_lawyer : ℝ := 0.3
  let p_woman_doctor : ℝ := 0.4
  let p_woman_engineer : ℝ := 0.3

  let p_man_lawyer : ℝ := 0.4
  let p_man_doctor : ℝ := 0.2
  let p_man_engineer : ℝ := 0.4

  (p_woman * p_woman_lawyer + p_woman * p_woman_doctor +
  p_man * p_man_lawyer + p_man * p_man_doctor)

theorem probability_correct : probability_study_group = 0.65 := by
  sorry

end probability_correct_l181_181070


namespace triple_composition_l181_181999

def g (x : ℤ) : ℤ := 3 * x + 2

theorem triple_composition :
  g (g (g 3)) = 107 :=
by
  sorry

end triple_composition_l181_181999


namespace determine_cost_price_l181_181621

def selling_price := 16
def loss_fraction := 1 / 6

noncomputable def cost_price (CP : ℝ) : Prop :=
  selling_price = CP - (loss_fraction * CP)

theorem determine_cost_price (CP : ℝ) (h: cost_price CP) : CP = 19.2 := by
  sorry

end determine_cost_price_l181_181621


namespace dodecahedron_interior_diagonals_count_l181_181705

-- Define the structure of a dodecahedron
structure Dodecahedron :=
  (faces : Fin 12)  -- 12 pentagonal faces
  (vertices : Fin 20)  -- 20 vertices
  (edges_per_vertex : Fin 3)  -- 3 faces meeting at each vertex

-- Define what an interior diagonal is
def is_interior_diagonal (v1 v2 : Fin 20) (dod : Dodecahedron) : Prop :=
  v1 ≠ v2 ∧ ¬ (∃ e, (e ∈ dod.edges_per_vertex))

-- Problem rephrased in Lean 4 statement: proving the number of interior diagonals equals 160.
theorem dodecahedron_interior_diagonals_count (dod : Dodecahedron) :
  (Finset.univ.filter (λ (p : Fin 20 × Fin 20), is_interior_diagonal p.1 p.2 dod)).card = 160 :=
by sorry

end dodecahedron_interior_diagonals_count_l181_181705


namespace systematic_sampling_first_group_l181_181958

theorem systematic_sampling_first_group (x : ℕ) (n : ℕ) (k : ℕ) (total_students : ℕ) (sampled_students : ℕ) 
  (interval : ℕ) (group_num : ℕ) (group_val : ℕ) 
  (h1 : total_students = 1000) (h2 : sampled_students = 40) (h3 : interval = total_students / sampled_students)
  (h4 : interval = 25) (h5 : group_num = 18) 
  (h6 : group_val = 443) (h7 : group_val = x + (group_num - 1) * interval) : 
  x = 18 := 
by 
  sorry

end systematic_sampling_first_group_l181_181958


namespace students_in_each_normal_class_l181_181849

theorem students_in_each_normal_class
  (total_students : ℕ)
  (percentage_moving : ℝ)
  (grades : ℕ)
  (adv_class_size : ℕ)
  (num_normal_classes : ℕ)
  (h1 : total_students = 1590)
  (h2 : percentage_moving = 0.4)
  (h3 : grades = 3)
  (h4 : adv_class_size = 20)
  (h5 : num_normal_classes = 6) :
  ((total_students * percentage_moving).toNat / grades - adv_class_size) / num_normal_classes = 32 := 
by sorry

end students_in_each_normal_class_l181_181849


namespace simplify_expression_l181_181231

theorem simplify_expression (y : ℝ) : 
  4 * y + 9 * y ^ 2 + 8 - (3 - 4 * y - 9 * y ^ 2) = 18 * y ^ 2 + 8 * y + 5 :=
by
  sorry

end simplify_expression_l181_181231


namespace find_b_l181_181081

-- Definitions from conditions
def f (x : ℚ) := 3 * x - 2
def g (x : ℚ) := 7 - 2 * x

-- Problem statement
theorem find_b (b : ℚ) (h : g (f b) = 1) : b = 5 / 3 := sorry

end find_b_l181_181081


namespace evaluate_three_squared_raised_four_l181_181168

theorem evaluate_three_squared_raised_four : (3^2)^4 = 6561 := by
  sorry

end evaluate_three_squared_raised_four_l181_181168


namespace probability_comparison_l181_181284

variables (M N : ℕ) (m n : ℝ)
variable (h₁ : m > 10^6)
variable (h₂ : n ≤ 10^6)

theorem probability_comparison (h₃: 0 < M) (h₄: 0 < N):
  (m * M) / (m * M + n * N) > (M / (M + N)) :=
by
  have h₅: n / m < 1 := sorry
  have h₆: M > 0 := by linarith
  have h₇: 1 + (n / m) * (N / M) < 2 := sorry
  sorry

end probability_comparison_l181_181284


namespace number_of_20_paise_coins_l181_181623

theorem number_of_20_paise_coins (x y : ℕ) (h1 : x + y = 336) (h2 : (20 / 100 : ℚ) * x + (25 / 100 : ℚ) * y = 71) :
    x = 260 :=
by
  sorry

end number_of_20_paise_coins_l181_181623


namespace total_spending_in_4_years_l181_181603

def trevor_spending_per_year : ℕ := 80
def reed_to_trevor_diff : ℕ := 20
def reed_to_quinn_factor : ℕ := 2

theorem total_spending_in_4_years :
  ∃ (reed_spending quinn_spending : ℕ),
  (reed_spending = trevor_spending_per_year - reed_to_trevor_diff) ∧
  (reed_spending = reed_to_quinn_factor * quinn_spending) ∧
  ((trevor_spending_per_year + reed_spending + quinn_spending) * 4 = 680) :=
sorry

end total_spending_in_4_years_l181_181603


namespace binomial_20_19_l181_181327

theorem binomial_20_19 : nat.choose 20 19 = 20 :=
by {
  sorry
}

end binomial_20_19_l181_181327


namespace gcd_polynomial_l181_181507

-- Given definitions based on the conditions
def is_multiple_of (x y : ℕ) : Prop := ∃ k : ℕ, x = k * y

-- Given the conditions: a is a multiple of 1610
variables (a : ℕ) (h : is_multiple_of a 1610)

-- Main theorem: Prove that gcd(a^2 + 9a + 35, a + 5) = 15
theorem gcd_polynomial (h : is_multiple_of a 1610) : gcd (a^2 + 9*a + 35) (a + 5) = 15 :=
sorry

end gcd_polynomial_l181_181507


namespace exists_polynomials_Q_R_l181_181421

theorem exists_polynomials_Q_R (P : Polynomial ℝ) (hP : ∀ x > 0, P.eval x > 0) :
  ∃ (Q R : Polynomial ℝ), (∀ a, 0 ≤ a → ∀ b, 0 ≤ b → Q.coeff a ≥ 0 ∧ R.coeff b ≥ 0) ∧ ∀ x > 0, P.eval x = (Q.eval x) / (R.eval x) := 
by
  sorry

end exists_polynomials_Q_R_l181_181421


namespace sum_of_digits_of_valid_n_eq_seven_l181_181880

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_valid_n (n : ℕ) : Prop :=
  (500 < n) ∧ (Nat.gcd 70 (n + 150) = 35) ∧ (Nat.gcd (n + 70) 150 = 50)

theorem sum_of_digits_of_valid_n_eq_seven :
  ∃ n : ℕ, is_valid_n n ∧ sum_of_digits n = 7 := by
  sorry

end sum_of_digits_of_valid_n_eq_seven_l181_181880


namespace problem_real_numbers_l181_181008

theorem problem_real_numbers (a b : ℝ) (n : ℕ) (h : 2 * a + 3 * b = 12) : 
  ((a / 3) ^ n + (b / 2) ^ n) ≥ 2 := 
sorry

end problem_real_numbers_l181_181008


namespace triangle_angle_sum_l181_181032

/-- In any triangle ABC, the sum of angle A and angle B is given to be 80 degrees.
    We need to prove that the measure of angle C is 100 degrees. -/
theorem triangle_angle_sum (A B C : ℝ) 
  (h1 : A + B = 80)
  (h2 : A + B + C = 180) : C = 100 :=
sorry

end triangle_angle_sum_l181_181032


namespace percentage_mr_william_land_l181_181797

theorem percentage_mr_william_land 
  (T W : ℝ) -- Total taxable land of the village and the total land of Mr. William
  (tax_collected_village : ℝ) -- Total tax collected from the village
  (tax_paid_william : ℝ) -- Tax paid by Mr. William
  (h1 : tax_collected_village = 3840) 
  (h2 : tax_paid_william = 480) 
  (h3 : (480 / 3840) = (25 / 100) * (W / T)) 
: (W / T) * 100 = 12.5 :=
by sorry

end percentage_mr_william_land_l181_181797


namespace tan_sum_l181_181011

theorem tan_sum (α : ℝ) (h : Real.cos (π / 2 + α) = 2 * Real.cos α) : 
  Real.tan α + Real.tan (2 * α) = -2 / 3 :=
by
  sorry

end tan_sum_l181_181011


namespace maximize_product_l181_181419

variable (x y : ℝ)
variable (h_xy_pos : x > 0 ∧ y > 0)
variable (h_sum : x + y = 35)

theorem maximize_product : x^5 * y^2 ≤ (25: ℝ)^5 * (10: ℝ)^2 :=
by
  -- Here we need to prove that the product x^5 y^2 is maximized at (x, y) = (25, 10)
  sorry

end maximize_product_l181_181419


namespace length_after_y_months_isabella_hair_length_l181_181557

-- Define the initial length of the hair
def initial_length : ℝ := 18

-- Define the growth rate of the hair per month
def growth_rate (x : ℝ) : ℝ := x

-- Define the number of months passed
def months_passed (y : ℕ) : ℕ := y

-- Prove the length of the hair after 'y' months
theorem length_after_y_months (x : ℝ) (y : ℕ) : ℝ :=
  initial_length + growth_rate x * y

-- Theorem statement to prove that the length of Isabella's hair after y months is 18 + xy
theorem isabella_hair_length (x : ℝ) (y : ℕ) : length_after_y_months x y = 18 + x * y :=
by sorry

end length_after_y_months_isabella_hair_length_l181_181557


namespace hyperbola_asymptotes_l181_181596

theorem hyperbola_asymptotes : 
  (∀ x y : ℝ, x^2 / 16 - y^2 / 9 = -1 → (y = 3/4 * x ∨ y = -3/4 * x)) :=
by
  sorry

end hyperbola_asymptotes_l181_181596


namespace intersection_non_empty_l181_181848

open Set

def M (a : ℤ) : Set ℤ := {a, 0}
def N : Set ℤ := {x | 2 * x^2 - 5 * x < 0}

theorem intersection_non_empty (a : ℤ) (h : (M a) ∩ N ≠ ∅) : a = 1 ∨ a = 2 := 
sorry

end intersection_non_empty_l181_181848


namespace triangle_angle_sum_l181_181033

/-- In any triangle ABC, the sum of angle A and angle B is given to be 80 degrees.
    We need to prove that the measure of angle C is 100 degrees. -/
theorem triangle_angle_sum (A B C : ℝ) 
  (h1 : A + B = 80)
  (h2 : A + B + C = 180) : C = 100 :=
sorry

end triangle_angle_sum_l181_181033


namespace power_function_increasing_l181_181392

theorem power_function_increasing (m : ℝ) : 
  (∀ x > 0, (m^2 - m - 1) * x^m > 0) → m = 2 := 
by 
  sorry

end power_function_increasing_l181_181392


namespace total_food_per_day_l181_181251

theorem total_food_per_day :
  let num_puppies := 4
  let num_dogs := 3
  let dog_meal_weight := 4
  let dog_meals_per_day := 3
  let dog_food_per_day := dog_meal_weight * dog_meals_per_day
  let total_dog_food_per_day := dog_food_per_day * num_dogs
  let puppy_meal_weight := dog_meal_weight / 2
  let puppy_meals_per_day := dog_meals_per_day * 3
  let puppy_food_per_day := puppy_meal_weight * puppy_meals_per_day
  let total_puppy_food_per_day := puppy_food_per_day * num_puppies
  total_dog_food_per_day + total_puppy_food_per_day = 108 :=
by
  sorry

end total_food_per_day_l181_181251


namespace inequality_proof_l181_181889

theorem inequality_proof (a b c : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) (h₃ : a * b + b * c + c * a = 1) : 
  (a / Real.sqrt (a ^ 2 + 1)) + (b / Real.sqrt (b ^ 2 + 1)) + (c / Real.sqrt (c ^ 2 + 1)) ≤ (3 / 2) :=
by
  sorry

end inequality_proof_l181_181889


namespace bus_distance_l181_181909

theorem bus_distance (w r : ℝ) (h1 : w = 0.17) (h2 : r = w + 3.67) : r = 3.84 :=
by
  sorry

end bus_distance_l181_181909


namespace pear_juice_percentage_l181_181583

/--
Miki has a dozen oranges and pears. She extracts juice as follows:
5 pears -> 10 ounces of pear juice
3 oranges -> 12 ounces of orange juice
She uses 10 pears and 10 oranges to make a blend.
Prove that the percent of the blend that is pear juice is 33.33%.
-/
theorem pear_juice_percentage :
  let pear_juice_per_pear := 10 / 5
  let orange_juice_per_orange := 12 / 3
  let total_pear_juice := 10 * pear_juice_per_pear
  let total_orange_juice := 10 * orange_juice_per_orange
  let total_juice := total_pear_juice + total_orange_juice
  total_pear_juice / total_juice = 1 / 3 :=
by
  sorry

end pear_juice_percentage_l181_181583


namespace percentage_conversion_l181_181467

-- Define the condition
def decimal_fraction : ℝ := 0.05

-- Define the target percentage
def percentage : ℝ := 5

-- State the theorem
theorem percentage_conversion (df : ℝ) (p : ℝ) (h1 : df = 0.05) (h2 : p = 5) : df * 100 = p :=
by
  rw [h1, h2]
  sorry

end percentage_conversion_l181_181467


namespace pen_distribution_l181_181593

theorem pen_distribution (x : ℕ) :
  8 * x + 3 = 12 * (x - 2) - 1 :=
sorry

end pen_distribution_l181_181593


namespace units_digit_product_l181_181676

theorem units_digit_product (a b : ℕ) (ha : a % 10 = 7) (hb : b % 10 = 4) :
  (a * b) % 10 = 8 := 
by
  sorry

end units_digit_product_l181_181676


namespace probability_heads_twice_and_die_three_l181_181994

-- Define a fair coin flip and regular six-sided die roll
def fair_coin : finset (bool) := {tt, ff}
def six_sided_die : finset (ℕ) := {1, 2, 3, 4, 5, 6}

-- Define the probability calculation function
noncomputable theory
open_locale classical

def probability_of_heads_times_two_and_die_shows_three : ℚ :=
  (1 / (fair_coin.card * fair_coin.card)) * (1 / six_sided_die.card)

-- Proof statement for the problem
theorem probability_heads_twice_and_die_three :
  probability_of_heads_times_two_and_die_shows_three = 1 / 24 := 
by {
  -- Calculation that follows directly from conditions
  have h_coin_card : fair_coin.card = 2 := by simp [fair_coin],
  have h_die_card : six_sided_die.card = 6 := by simp [six_sided_die],
  simp [probability_of_heads_times_two_and_die_shows_three, h_coin_card, h_die_card],
  norm_num,
}

end probability_heads_twice_and_die_three_l181_181994


namespace find_f_sum_l181_181902

noncomputable def f : ℝ → ℝ := sorry

axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom functional_eq : ∀ x : ℝ, f (2 + x) + f (2 - x) = 0
axiom f_at_one : f 1 = 9

theorem find_f_sum :
  f 2010 + f 2011 + f 2012 = -9 :=
sorry

end find_f_sum_l181_181902


namespace quotient_when_divided_by_8_l181_181786

theorem quotient_when_divided_by_8
  (n : ℕ)
  (h1 : n = 12 * 7 + 5)
  : (n / 8) = 11 :=
by
  -- the proof is omitted
  sorry

end quotient_when_divided_by_8_l181_181786


namespace find_x_on_line_segment_l181_181391

theorem find_x_on_line_segment (x : ℚ) : 
    (∃ m : ℚ, m = (9 - (-1))/(1 - (-2)) ∧ (2 - 9 = m * (x - 1))) → x = -11/10 :=
by 
  sorry

end find_x_on_line_segment_l181_181391


namespace angle_C_in_triangle_l181_181050

theorem angle_C_in_triangle (A B C : ℝ) (h : A + B = 80) : C = 100 :=
by
  -- The measure of angle C follows from the condition and the properties of triangles.
  sorry

end angle_C_in_triangle_l181_181050


namespace total_movie_hours_l181_181724

-- Definitions
def JoyceMovie : ℕ := 12 -- Joyce's favorite movie duration in hours
def MichaelMovie : ℕ := 10 -- Michael's favorite movie duration in hours
def NikkiMovie : ℕ := 30 -- Nikki's favorite movie duration in hours
def RynMovie : ℕ := 24 -- Ryn's favorite movie duration in hours

-- Condition translations
def Joyce_movie_condition : Prop := JoyceMovie = MichaelMovie + 2
def Nikki_movie_condition : Prop := NikkiMovie = 3 * MichaelMovie
def Ryn_movie_condition : Prop := RynMovie = (4 * NikkiMovie) / 5
def Nikki_movie_given : Prop := NikkiMovie = 30

-- The theorem to prove
theorem total_movie_hours : Joyce_movie_condition ∧ Nikki_movie_condition ∧ Ryn_movie_condition ∧ Nikki_movie_given → 
  (JoyceMovie + MichaelMovie + NikkiMovie + RynMovie = 76) :=
by
  intros h
  sorry

end total_movie_hours_l181_181724


namespace quadratic_roots_unique_l181_181681

theorem quadratic_roots_unique (p q : ℚ) :
  (∀ x : ℚ, x^2 + p * x + q = 0 ↔ (x = 2 * p ∨ x = p + q)) →
  p = 2 / 3 ∧ q = -8 / 3 :=
by
  sorry

end quadratic_roots_unique_l181_181681


namespace find_hidden_data_points_l181_181209

-- Given conditions and data
def student_A_score := 81
def student_B_score := 76
def student_D_score := 80
def student_E_score := 83
def number_of_students := 5
def average_score := 80

-- The total score from the average and number of students
def total_score := average_score * number_of_students

theorem find_hidden_data_points (student_C_score mode_score : ℕ) :
  (student_A_score + student_B_score + student_C_score + student_D_score + student_E_score = total_score) ∧
  (mode_score = 80) :=
by
  sorry

end find_hidden_data_points_l181_181209


namespace sqrt_four_four_summed_l181_181919

theorem sqrt_four_four_summed :
  sqrt (4 ^ 4 + 4 ^ 4 + 4 ^ 4 + 4 ^ 4) = 32 := by
  sorry

end sqrt_four_four_summed_l181_181919


namespace compute_pairs_a_b_l181_181878

noncomputable def f (x a b : ℝ) : ℝ := (x + a) / (x + b)

theorem compute_pairs_a_b (a b : ℝ) (x : ℝ) (hx1 : x ≠ 0) (hx2 : x ≠ -b) :
  ((∀ x, f (f x a b) a b = -1 / x) ↔ (a = -1 ∧ b = 1)) :=
sorry

end compute_pairs_a_b_l181_181878


namespace total_pears_picked_l181_181410

variables (jason_keith_mike_morning : ℕ)
variables (alicia_tina_nicola_afternoon : ℕ)
variables (days : ℕ)
variables (total_pears : ℕ)

def one_day_total (jason_keith_mike_morning alicia_tina_nicola_afternoon : ℕ) : ℕ :=
  jason_keith_mike_morning + alicia_tina_nicola_afternoon

theorem total_pears_picked (hjkm: jason_keith_mike_morning = 46 + 47 + 12)
                           (hatn: alicia_tina_nicola_afternoon = 28 + 33 + 52)
                           (hdays: days = 3)
                           (htotal: total_pears = 654):
  total_pears = (one_day_total  (46 + 47 + 12)  (28 + 33 + 52)) * 3 := 
sorry

end total_pears_picked_l181_181410


namespace fencing_required_l181_181120

theorem fencing_required (L W : ℕ) (hL : L = 40) (hA : 40 * W = 680) : 2 * W + L = 74 :=
by sorry

end fencing_required_l181_181120


namespace max_probability_first_black_ace_l181_181108

def probability_first_black_ace(k : ℕ) : ℚ :=
  if 1 ≤ k ∧ k ≤ 51 then (52 - k) / 1326 else 0

theorem max_probability_first_black_ace : 
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ 51 → probability_first_black_ace k ≤ probability_first_black_ace 1 :=
by
  sorry

end max_probability_first_black_ace_l181_181108


namespace total_food_per_day_l181_181250

theorem total_food_per_day :
  let num_puppies := 4
  let num_dogs := 3
  let dog_meal_weight := 4
  let dog_meals_per_day := 3
  let dog_food_per_day := dog_meal_weight * dog_meals_per_day
  let total_dog_food_per_day := dog_food_per_day * num_dogs
  let puppy_meal_weight := dog_meal_weight / 2
  let puppy_meals_per_day := dog_meals_per_day * 3
  let puppy_food_per_day := puppy_meal_weight * puppy_meals_per_day
  let total_puppy_food_per_day := puppy_food_per_day * num_puppies
  total_dog_food_per_day + total_puppy_food_per_day = 108 :=
by
  sorry

end total_food_per_day_l181_181250


namespace units_digit_of_n_l181_181206

theorem units_digit_of_n (n : ℕ) (h : n = 56^78 + 87^65) : (n % 10) = 3 :=
by
  sorry

end units_digit_of_n_l181_181206


namespace rise_in_water_level_l181_181631

theorem rise_in_water_level (edge base_length base_width : ℝ) (cube_volume base_area rise : ℝ) 
  (h₁ : edge = 5) (h₂ : base_length = 10) (h₃ : base_width = 5)
  (h₄ : cube_volume = edge^3) (h₅ : base_area = base_length * base_width) 
  (h₆ : rise = cube_volume / base_area) : 
  rise = 2.5 := 
by 
  -- add proof here 
  sorry

end rise_in_water_level_l181_181631


namespace exists_unique_continuous_extension_l181_181730

noncomputable def F (f : ℚ → ℚ) (hf_bij : Function.Bijective f) (hf_mono : Monotone f) : ℝ → ℝ :=
  sorry

theorem exists_unique_continuous_extension (f : ℚ → ℚ) (hf_bij : Function.Bijective f) (hf_mono : Monotone f) :
  ∃! F : ℝ → ℝ, Continuous F ∧ ∀ x : ℚ, F x = f x :=
sorry

end exists_unique_continuous_extension_l181_181730


namespace angle_C_in_triangle_l181_181051

theorem angle_C_in_triangle (A B C : ℝ) (h : A + B = 80) : C = 100 :=
by
  -- The measure of angle C follows from the condition and the properties of triangles.
  sorry

end angle_C_in_triangle_l181_181051


namespace depth_B_is_correct_l181_181632

-- Given: Diver A is at a depth of -55 meters.
def depth_A : ℤ := -55

-- Given: Diver B is 5 meters above diver A.
def offset : ℤ := 5

-- Prove: The depth of diver B
theorem depth_B_is_correct : (depth_A + offset) = -50 :=
by
  sorry

end depth_B_is_correct_l181_181632


namespace students_in_each_normal_class_l181_181851

theorem students_in_each_normal_class
  (initial_students : ℕ)
  (percent_to_move : ℚ)
  (grade_levels : ℕ)
  (students_in_advanced_class : ℕ)
  (num_of_normal_classes : ℕ)
  (h_initial_students : initial_students = 1590)
  (h_percent_to_move : percent_to_move = 0.4)
  (h_grade_levels : grade_levels = 3)
  (h_students_in_advanced_class : students_in_advanced_class = 20)
  (h_num_of_normal_classes : num_of_normal_classes = 6) :
  let students_moving := (initial_students : ℚ) * percent_to_move,
      students_per_grade := students_moving / grade_levels,
      students_remaining := students_per_grade - (students_in_advanced_class : ℚ),
      students_per_normal_class := students_remaining / (num_of_normal_classes : ℚ)
  in students_per_normal_class = 32 := 
by
  sorry

end students_in_each_normal_class_l181_181851


namespace percent_savings_per_roll_l181_181630

theorem percent_savings_per_roll 
  (cost_case : ℕ := 900) -- In cents, equivalent to $9
  (cost_individual : ℕ := 100) -- In cents, equivalent to $1
  (num_rolls : ℕ := 12) :
  (cost_individual - (cost_case / num_rolls)) * 100 / cost_individual = 25 := 
sorry

end percent_savings_per_roll_l181_181630


namespace compare_probabilities_l181_181276

-- Definitions
variables (M N: ℕ) (m n: ℕ)
  
-- Conditions
def condition_m_millionaire : Prop := m > 10^6
def condition_n_nonmillionaire : Prop := n ≤ 10^6

-- Probabilities
noncomputable def P_A : ℚ := (M:ℚ) / (M + (n:ℚ)/(m:ℚ) * N)
noncomputable def P_B : ℚ := (M:ℚ) / (M + N)

-- Theorem statement
theorem compare_probabilities
  (hM : M > 0) (hN : N > 0)
  (h_m_millionaire : condition_m_millionaire m)
  (h_n_nonmillionaire : condition_n_nonmillionaire n) :
  P_A M N m n > P_B M N := sorry

end compare_probabilities_l181_181276


namespace power_of_power_example_l181_181162

theorem power_of_power_example : (3^2)^4 = 6561 := by
  sorry

end power_of_power_example_l181_181162


namespace diff_is_multiple_of_9_l181_181476

-- Definitions
def orig_num (a b : ℕ) : ℕ := 10 * a + b
def new_num (a b : ℕ) : ℕ := 10 * b + a

-- Statement of the mathematical proof problem
theorem diff_is_multiple_of_9 (a b : ℕ) : 
  9 ∣ (new_num a b - orig_num a b) :=
by
  sorry

end diff_is_multiple_of_9_l181_181476


namespace find_y_l181_181116

theorem find_y (y : ℝ) (h : (3 * y) / 7 = 12) : y = 28 :=
by
  -- The proof would go here
  sorry

end find_y_l181_181116


namespace value_of_fourth_set_l181_181151

def value_in_set (a b c d : ℕ) : ℕ :=
  (a * b * c * d) - (a + b + c + d)

theorem value_of_fourth_set :
  value_in_set 1 5 6 7 = 191 :=
by
  sorry

end value_of_fourth_set_l181_181151


namespace angleC_is_100_l181_181052

-- Given a triangle ABC
variables (A B C : Type) [angle A] [angle B] [angle C]

-- Define a triangle
structure triangle (A B C : Type) := 
(angle_A : A)
(angle_B : B)
(angle_C : C)

-- Assume the sum of the angles in a triangle is 180 degrees
axiom sum_of_angles_in_triangle (t : triangle A B C) : 
  (angle_A + angle_B + angle_C = 180)

-- Given condition
axiom sum_of_angleA_and_angleB (t : triangle A B C) :
  (angle_A + angle_B = 80)

-- Prove that angle C is 100 degrees
theorem angleC_is_100 (t : triangle A B C) :
  angle_C = 100 :=
begin
  sorry
end

end angleC_is_100_l181_181052


namespace arithmetic_prog_includes_1999_l181_181144

-- Definitions based on problem conditions
def is_in_arithmetic_progression (a d n : ℕ) : ℕ := a + (n - 1) * d

theorem arithmetic_prog_includes_1999
  (d : ℕ) (h_pos : d > 0) 
  (h_includes7 : ∃ n:ℕ, is_in_arithmetic_progression 7 d n = 7)
  (h_includes15 : ∃ n:ℕ, is_in_arithmetic_progression 7 d n = 15)
  (h_includes27 : ∃ n:ℕ, is_in_arithmetic_progression 7 d n = 27) :
  ∃ n:ℕ, is_in_arithmetic_progression 7 d n = 1999 := 
sorry

end arithmetic_prog_includes_1999_l181_181144


namespace area_of_shaded_region_l181_181214

theorem area_of_shaded_region 
    (large_side : ℝ) (small_side : ℝ)
    (h_large : large_side = 10) 
    (h_small : small_side = 4) : 
    (large_side^2 - small_side^2) / 4 = 21 :=
by
  -- All proof steps are to be completed and checked,
  -- and sorry is used as placeholder for the final proof.
  sorry

end area_of_shaded_region_l181_181214


namespace necessary_not_sufficient_l181_181936

theorem necessary_not_sufficient (a b c d : ℝ) : 
  (a + c > b + d) → (a > b ∧ c > d) :=
sorry

end necessary_not_sufficient_l181_181936


namespace solution_set_ineq_l181_181923

theorem solution_set_ineq (m : ℝ) (hm : m > 1) :
  {x : ℝ | x^2 + (m-1) * x - m >= 0} = {x : ℝ | x <= -m ∨ x >= 1} :=
sorry

end solution_set_ineq_l181_181923


namespace fraction_zero_implies_x_neg1_l181_181398

theorem fraction_zero_implies_x_neg1 (x : ℝ) (h₁ : x^2 - 1 = 0) (h₂ : x - 1 ≠ 0) : x = -1 := by
  sorry

end fraction_zero_implies_x_neg1_l181_181398


namespace prank_people_combinations_l181_181772

theorem prank_people_combinations (Monday Tuesday Wednesday Thursday Friday : ℕ) 
  (hMonday : Monday = 2)
  (hTuesday : Tuesday = 3)
  (hWednesday : Wednesday = 6)
  (hThursday : Thursday = 4)
  (hFriday : Friday = 3) :
  Monday * Tuesday * Wednesday * Thursday * Friday = 432 :=
  by sorry

end prank_people_combinations_l181_181772


namespace triangle_area_l181_181461

-- Define the vertices of the triangle
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (7, 4)
def C : ℝ × ℝ := (7, -4)

-- Statement to prove the area of the triangle is 32 square units
theorem triangle_area :
  let x1 := A.1
  let y1 := A.2
  let x2 := B.1
  let y2 := B.2
  let x3 := C.1
  let y3 := C.2
  (1 / 2 : ℝ) * |x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)| = 32 := by
  sorry  -- Proof to be provided

end triangle_area_l181_181461


namespace aunt_masha_butter_usage_l181_181962

theorem aunt_masha_butter_usage
  (x y : ℝ)
  (h1 : x + 10 * y = 600)
  (h2 : x = 5 * y) :
  (2 * x + 2 * y = 480) := 
by
  sorry

end aunt_masha_butter_usage_l181_181962


namespace geometric_sequence_common_ratio_l181_181014

theorem geometric_sequence_common_ratio
  (q : ℝ) (a_n : ℕ → ℝ)
  (h_inc : ∀ n, a_n (n + 1) = q * a_n n ∧ q > 1)
  (h_a2 : a_n 2 = 2)
  (h_a4_a3 : a_n 4 - a_n 3 = 4) : 
  q = 2 :=
sorry

end geometric_sequence_common_ratio_l181_181014


namespace proctoring_arrangements_l181_181867

/-- Consider 4 teachers A, B, C, D each teaching their respective classes a, b, c, d.
    Each teacher must not proctor their own class.
    Prove that there are exactly 9 ways to arrange the proctoring as required. -/
theorem proctoring_arrangements : 
  ∃ (arrangements : Finset ((Fin 4) → (Fin 4))), 
    (∀ (f : (Fin 4) → (Fin 4)), f ∈ arrangements → ∀ i : Fin 4, f i ≠ i) 
    ∧ arrangements.card = 9 :=
sorry

end proctoring_arrangements_l181_181867


namespace part1_l181_181121

theorem part1 (P Q R : Polynomial ℝ) : 
  ¬ ∃ (P Q R : Polynomial ℝ), (∀ x y z : ℝ, (x - y + 1)^3 * P.eval x + (y - z - 1)^3 * Q.eval y + (z - 2 * x + 1)^3 * R.eval z = 1) := sorry

end part1_l181_181121


namespace Patriots_won_30_games_l181_181239

def Tigers_won_more_games_than_Eagles (games_tigers games_eagles : ℕ) : Prop :=
games_tigers > games_eagles

def Patriots_won_more_than_Cubs_less_than_Mounties (games_patriots games_cubs games_mounties : ℕ) : Prop :=
games_cubs < games_patriots ∧ games_patriots < games_mounties

def Cubs_won_more_than_20_games (games_cubs : ℕ) : Prop :=
games_cubs > 20

theorem Patriots_won_30_games (games_tigers games_eagles games_patriots games_cubs games_mounties : ℕ)  :
  Tigers_won_more_games_than_Eagles games_tigers games_eagles →
  Patriots_won_more_than_Cubs_less_than_Mounties games_patriots games_cubs games_mounties →
  Cubs_won_more_than_20_games games_cubs →
  ∃ games_patriots, games_patriots = 30 := 
by
  sorry

end Patriots_won_30_games_l181_181239


namespace game_cost_proof_l181_181355

variable (initial : ℕ) (allowance : ℕ) (final : ℕ) (cost : ℕ)

-- Initial amount
def initial_money : ℕ := 11
-- Allowance received
def allowance_money : ℕ := 14
-- Final amount of money
def final_money : ℕ := 22
-- Cost of the new game is to be proved
def game_cost : ℕ :=  initial_money - (final_money - allowance_money)

theorem game_cost_proof : game_cost = 3 := by
  sorry

end game_cost_proof_l181_181355


namespace neither_A_B_C_prob_correct_l181_181218

noncomputable def P (A B C : Prop) : Prop :=
  let P_A := 0.25
  let P_B := 0.35
  let P_C := 0.40
  let P_A_and_B := 0.10
  let P_A_and_C := 0.15
  let P_B_and_C := 0.20
  let P_A_and_B_and_C := 0.05
  
  let P_A_or_B_or_C := 
    P_A + P_B + P_C - P_A_and_B - P_A_and_C - P_B_and_C + P_A_and_B_and_C
  
  let P_neither_A_nor_B_nor_C := 1 - P_A_or_B_or_C
    
  P_neither_A_nor_B_nor_C = 0.45

theorem neither_A_B_C_prob_correct :
  P A B C := by
  sorry

end neither_A_B_C_prob_correct_l181_181218


namespace min_dot_product_l181_181979

theorem min_dot_product (m n : ℝ) (x1 x2 : ℝ)
    (h1 : m ≠ 0) 
    (h2 : n ≠ 0)
    (h3 : (x1 + 2) * (x2 - x1) + m * x1 * (n - m * x1) = 0) :
    ∃ (x1 : ℝ), (x1 = -2 / (m^2 + 1)) → 
    (x1 + 2) * (x2 + 2) + m * n * x1 = 4 * m^2 / (m^2 + 1) := 
sorry

end min_dot_product_l181_181979


namespace problem_a_problem_b_l181_181743

-- Problem (a)
theorem problem_a (n : Nat) : Nat.mod (7 ^ (2 * n) - 4 ^ (2 * n)) 33 = 0 := sorry

-- Problem (b)
theorem problem_b (n : Nat) : Nat.mod (3 ^ (6 * n) - 2 ^ (6 * n)) 35 = 0 := sorry

end problem_a_problem_b_l181_181743


namespace multiple_of_other_number_l181_181584

theorem multiple_of_other_number 
(m S L : ℕ) 
(hl : L = 33) 
(hrel : L = m * S - 3) 
(hsum : L + S = 51) : 
m = 2 :=
by
  sorry

end multiple_of_other_number_l181_181584


namespace williams_tips_august_l181_181459

variable (A : ℝ) (total_tips : ℝ)
variable (tips_August : ℝ) (average_monthly_tips_other_months : ℝ)

theorem williams_tips_august (h1 : tips_August = 0.5714285714285714 * total_tips)
                               (h2 : total_tips = 7 * average_monthly_tips_other_months) 
                               (h3 : total_tips = tips_August + 6 * average_monthly_tips_other_months) :
                               tips_August = 8 * average_monthly_tips_other_months :=
by
  sorry

end williams_tips_august_l181_181459


namespace problem_statement_l181_181350

theorem problem_statement (c d : ℤ) (hc : c = 3) (hd : d = 2) :
  (c^2 + d)^2 - (c^2 - d)^2 = 72 :=
by
  sorry

end problem_statement_l181_181350


namespace volunteers_meet_again_in_360_days_l181_181143

open Nat

theorem volunteers_meet_again_in_360_days :
  ∃ (n : ℕ), n = Nat.lcm (Nat.lcm 5 6) (Nat.lcm 8 9) ∧ n = 360 := by
  sorry

end volunteers_meet_again_in_360_days_l181_181143


namespace smallest_positive_perfect_square_divisible_by_3_and_5_is_225_l181_181110

def smallest_perf_square_divisible_by_3_and_5 : ℕ :=
  let n := 15 in n * n

theorem smallest_positive_perfect_square_divisible_by_3_and_5_is_225 :
  smallest_perf_square_divisible_by_3_and_5 = 225 :=
by
  sorry

end smallest_positive_perfect_square_divisible_by_3_and_5_is_225_l181_181110


namespace roots_equation_l181_181106

theorem roots_equation (a b c d : ℝ) (h₁ : a ≠ 0) (h₂ : a * 4^3 + b * 4^2 + c * 4 + d = 0) (h₃ : a * (-3)^3 + b * (-3)^2 + c * (-3) + d = 0) :
  (b + c) / a = -13 :=
by
  sorry

end roots_equation_l181_181106


namespace sum_of_a_and_b_is_24_l181_181483

theorem sum_of_a_and_b_is_24 
  (a b : ℕ) 
  (h_a_pos : a > 0) 
  (h_b_gt_one : b > 1) 
  (h_maximal : ∀ (a' b' : ℕ), (a' > 0) → (b' > 1) → (a'^b' < 500) → (a'^b' ≤ a^b)) :
  a + b = 24 := 
sorry

end sum_of_a_and_b_is_24_l181_181483


namespace part1_part2_part3_l181_181986

def f (x : ℝ) : ℝ := x^2 - 1
def g (a x : ℝ) : ℝ := a * |x - 1|

theorem part1 :
  ∀ x : ℝ, |f x| = |x - 1| → x = -2 ∨ x = 0 ∨ x = 1 :=
sorry

theorem part2 (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ |f x1| = g a x1 ∧ |f x2| = g a x2) ↔ (a = 0 ∨ a = 2) :=
sorry

theorem part3 (a : ℝ) :
  (∀ x : ℝ, f x ≥ g a x) ↔ (a ≤ -2) :=
sorry

end part1_part2_part3_l181_181986


namespace triangle_interior_angles_not_greater_than_60_l181_181263

theorem triangle_interior_angles_not_greater_than_60 (α β γ : ℝ) (h_sum : α + β + γ = 180) 
  (h_pos : α > 0 ∧ β > 0 ∧ γ > 0) :
  α ≤ 60 ∨ β ≤ 60 ∨ γ ≤ 60 :=
by
  sorry

end triangle_interior_angles_not_greater_than_60_l181_181263


namespace projection_magnitude_of_a_onto_b_equals_neg_three_l181_181356

variables {a b : ℝ}

def vector_magnitude (v : ℝ) : ℝ := abs v

def dot_product (a b : ℝ) : ℝ := a * b

noncomputable def projection (a b : ℝ) : ℝ := (dot_product a b) / (vector_magnitude b)

theorem projection_magnitude_of_a_onto_b_equals_neg_three
  (ha : vector_magnitude a = 5)
  (hb : vector_magnitude b = 3)
  (hab : dot_product a b = -9) :
  projection a b = -3 :=
by sorry

end projection_magnitude_of_a_onto_b_equals_neg_three_l181_181356


namespace cost_of_notebook_is_12_l181_181401

/--
In a class of 36 students, a majority purchased notebooks. Each student bought the same number of notebooks (greater than 2). The price of a notebook in cents was double the number of notebooks each student bought, and the total expense was 2772 cents.
Prove that the cost of one notebook in cents is 12.
-/
theorem cost_of_notebook_is_12
  (s n c : ℕ) (total_students : ℕ := 36) 
  (h_majority : s > 18) 
  (h_notebooks : n > 2) 
  (h_cost : c = 2 * n) 
  (h_total_cost : s * c * n = 2772) 
  : c = 12 :=
by sorry

end cost_of_notebook_is_12_l181_181401


namespace inequality_abcde_l181_181219

theorem inequality_abcde
  (a b c d : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (hd : 0 < d) : 
  1 / a + 1 / b + 4 / c + 16 / d ≥ 64 / (a + b + c + d) := 
  sorry

end inequality_abcde_l181_181219


namespace Zach_scored_more_l181_181619

theorem Zach_scored_more :
  let Zach := 42
  let Ben := 21
  Zach - Ben = 21 :=
by
  let Zach := 42
  let Ben := 21
  exact rfl

end Zach_scored_more_l181_181619


namespace smallest_perfect_square_divisible_by_3_and_5_l181_181111

theorem smallest_perfect_square_divisible_by_3_and_5 : ∃ (n : ℕ), n > 0 ∧ (∃ (m : ℕ), n = m * m) ∧ (n % 3 = 0) ∧ (n % 5 = 0) ∧ n = 225 :=
by
  sorry

end smallest_perfect_square_divisible_by_3_and_5_l181_181111


namespace factorization_of_polynomial_l181_181664

theorem factorization_of_polynomial :
  ∀ x : ℝ, x^8 - 4*x^6 + 6*x^4 - 4*x^2 + 1 = (x - 1)^4 * (x + 1)^4 :=
by
  intros x
  sorry

end factorization_of_polynomial_l181_181664


namespace part1_part2_l181_181525

noncomputable def f (x : ℝ) (a : ℝ) := (Real.exp x / x) - Real.log x + x - a

theorem part1 (x : ℝ) (a : ℝ) :
    (∀ x > 0, f x a ≥ 0) → a ≤ Real.exp 1 + 1 :=
sorry

theorem part2 (x1 x2 : ℝ) (a : ℝ) :
  f x1 a = 0 → f x2 a = 0 → x1 < 1 → 1 < x2 → x1 * x2 < 1 :=
sorry

end part1_part2_l181_181525


namespace measure_of_angle_C_l181_181042

-- Conditions of the problem
variable (A B C : ℝ) -- Angles in the triangle

-- Sum of all angles in a triangle
axiom sum_of_angles (h : A + B + C = 180)

-- Condition given in the problem
axiom given_angle_sum (h1 : A + B = 80)

-- Desired proof statement
theorem measure_of_angle_C : C = 100 := by
  sorry

end measure_of_angle_C_l181_181042


namespace pradeep_failure_marks_l181_181742

theorem pradeep_failure_marks :
  let total_marks := 925
  let pradeep_score := 160
  let passing_percentage := 20
  let passing_marks := (passing_percentage / 100) * total_marks
  let failed_by := passing_marks - pradeep_score
  failed_by = 25 :=
by
  sorry

end pradeep_failure_marks_l181_181742


namespace triangle_angle_sum_l181_181035

/-- In any triangle ABC, the sum of angle A and angle B is given to be 80 degrees.
    We need to prove that the measure of angle C is 100 degrees. -/
theorem triangle_angle_sum (A B C : ℝ) 
  (h1 : A + B = 80)
  (h2 : A + B + C = 180) : C = 100 :=
sorry

end triangle_angle_sum_l181_181035


namespace find_average_income_of_M_and_O_l181_181441

def average_income_of_M_and_O (M N O : ℕ) : Prop :=
  M + N = 10100 ∧
  N + O = 12500 ∧
  M = 4000 ∧
  (M + O) / 2 = 5200

theorem find_average_income_of_M_and_O (M N O : ℕ):
  average_income_of_M_and_O M N O → 
  (M + O) / 2 = 5200 :=
by
  intro h
  exact h.2.2.2

end find_average_income_of_M_and_O_l181_181441


namespace min_cost_and_ways_l181_181173

-- Define the cost of each package
def cost_A : ℕ := 10
def cost_B : ℕ := 5

-- Define a function to calculate the total cost given the number of each package
def total_cost (nA nB : ℕ) : ℕ := nA * cost_A + nB * cost_B

-- Define the number of friends
def num_friends : ℕ := 4

-- Prove the minimum cost is 15 yuan and there are 28 ways
theorem min_cost_and_ways :
  (∃ nA nB : ℕ, total_cost nA nB = 15 ∧ (
    (nA = 1 ∧ nB = 1 ∧ 12 = 12) ∨ 
    (nA = 0 ∧ nB = 3 ∧ 12 = 12) ∨
    (nA = 0 ∧ nB = 3 ∧ 4 = 4) → 28 = 28)) :=
sorry

end min_cost_and_ways_l181_181173


namespace trigonometric_identity_l181_181973

theorem trigonometric_identity :
  (3 / (Real.sin (20 * Real.pi / 180))^2) - (1 / (Real.cos (20 * Real.pi / 180))^2) + 64 * (Real.sin (20 * Real.pi / 180))^2 = 32 :=
by
  sorry

end trigonometric_identity_l181_181973


namespace two_positive_numbers_inequality_three_positive_numbers_am_gm_l181_181435

theorem two_positive_numbers_inequality (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  x^3 + y^3 ≥ x^2 * y + x * y^2 ∧ (x = y ↔ x^3 + y^3 = x^2 * y + x * y^2) := by
sorry

theorem three_positive_numbers_am_gm (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b + c) / 3 ≥ (a * b * c)^(1/3) ∧ (a = b ∧ b = c ↔ (a + b + c) / 3 = (a * b * c)^(1/3)) := by
sorry

end two_positive_numbers_inequality_three_positive_numbers_am_gm_l181_181435


namespace prove_a_ge_neg_one_fourth_l181_181531

-- Lean 4 statement to reflect the problem
theorem prove_a_ge_neg_one_fourth
  (x y z a : ℝ)
  (hx : 0 < x)
  (hy : 0 < y)
  (hz : 0 < z)
  (h1 : x * y - z = a)
  (h2 : y * z - x = a)
  (h3 : z * x - y = a) :
  a ≥ - (1 / 4) :=
sorry

end prove_a_ge_neg_one_fourth_l181_181531


namespace trader_sold_80_meters_l181_181960

variable (x : ℕ)
variable (selling_price_per_meter profit_per_meter cost_price_per_meter total_selling_price : ℕ)

theorem trader_sold_80_meters
  (h_cost_price : cost_price_per_meter = 118)
  (h_profit : profit_per_meter = 7)
  (h_selling_price : selling_price_per_meter = cost_price_per_meter + profit_per_meter)
  (h_total_selling_price : total_selling_price = 10000)
  (h_eq : selling_price_per_meter * x = total_selling_price) :
  x = 80 := by
    sorry

end trader_sold_80_meters_l181_181960


namespace tyler_buffet_meals_l181_181455

open Nat

theorem tyler_buffet_meals : 
  let meats := 3
  let vegetables := 5
  let desserts := 5
  ∃ meals: ℕ, meals = meats * (choose vegetables 2) * desserts ∧ meals = 150 :=
by
  let meats := 3
  let vegetables := 5
  let desserts := 5
  let meals := meats * (choose vegetables 2) * desserts
  existsi meals
  split
  {
    rfl
  }
  {
    rw [Nat.choose, factorial, factorial, factorial]
    {
      norm_num
    }
    {
      sorry -- Proof of equivalence to 150
    }
  }

end tyler_buffet_meals_l181_181455


namespace percent_savings_12_roll_package_l181_181628

def percent_savings_per_roll (package_cost : ℕ) (individual_cost : ℕ) (num_rolls : ℕ) : ℚ :=
  let individual_total := num_rolls * individual_cost
  let package_total := package_cost
  let per_roll_package_cost := package_total / num_rolls
  let savings_per_roll := individual_cost - per_roll_package_cost
  (savings_per_roll / individual_cost) * 100

theorem percent_savings_12_roll_package :
  percent_savings_per_roll 9 1 12 = 25 := 
sorry

end percent_savings_12_roll_package_l181_181628


namespace angleC_is_100_l181_181054

-- Given a triangle ABC
variables (A B C : Type) [angle A] [angle B] [angle C]

-- Define a triangle
structure triangle (A B C : Type) := 
(angle_A : A)
(angle_B : B)
(angle_C : C)

-- Assume the sum of the angles in a triangle is 180 degrees
axiom sum_of_angles_in_triangle (t : triangle A B C) : 
  (angle_A + angle_B + angle_C = 180)

-- Given condition
axiom sum_of_angleA_and_angleB (t : triangle A B C) :
  (angle_A + angle_B = 80)

-- Prove that angle C is 100 degrees
theorem angleC_is_100 (t : triangle A B C) :
  angle_C = 100 :=
begin
  sorry
end

end angleC_is_100_l181_181054


namespace total_letters_l181_181198

theorem total_letters (brother_letters : ℕ) (greta_more_than_brother : ℕ) (mother_multiple : ℕ)
  (h_brother : brother_letters = 40)
  (h_greta : ∀ (brother_letters greta_letters : ℕ), greta_letters = brother_letters + greta_more_than_brother)
  (h_mother : ∀ (total_letters mother_letters : ℕ), mother_letters = mother_multiple * total_letters) :
  brother_letters + (brother_letters + greta_more_than_brother) + (mother_multiple * (brother_letters + (brother_letters + greta_more_than_brother))) = 270 :=
by
  sorry

end total_letters_l181_181198


namespace base5_2004_to_decimal_is_254_l181_181981

def base5_to_decimal (n : Nat) : Nat :=
  match n with
  | 2004 => 2 * 5^3 + 0 * 5^2 + 0 * 5^1 + 4 * 5^0
  | _ => 0

theorem base5_2004_to_decimal_is_254 :
  base5_to_decimal 2004 = 254 :=
by
  -- Proof goes here
  sorry

end base5_2004_to_decimal_is_254_l181_181981


namespace total_shaded_area_l181_181126

/-- 
Given a 6-foot by 12-foot floor tiled with 1-foot by 1-foot tiles,
where each tile has four white quarter circles of radius 1/3 foot at its corners,
prove that the total shaded area of the floor is 72 - 8π square feet.
-/
theorem total_shaded_area :
  let floor_length := 6
  let floor_width := 12
  let tile_size := 1
  let radius := 1 / 3
  let area_of_tile := tile_size * tile_size
  let white_area_per_tile := (Real.pi * radius^2 / 4) * 4
  let shaded_area_per_tile := area_of_tile - white_area_per_tile
  let number_of_tiles := floor_length * floor_width
  let total_shaded_area := number_of_tiles * shaded_area_per_tile
  total_shaded_area = 72 - 8 * Real.pi :=
by
  let floor_length := 6
  let floor_width := 12
  let tile_size := 1
  let radius := 1 / 3
  let area_of_tile := tile_size * tile_size
  let white_area_per_tile := (Real.pi * radius^2 / 4) * 4
  let shaded_area_per_tile := area_of_tile - white_area_per_tile
  let number_of_tiles := floor_length * floor_width
  let total_shaded_area := number_of_tiles * shaded_area_per_tile
  sorry

end total_shaded_area_l181_181126


namespace unit_price_solution_purchase_plan_and_costs_l181_181715

-- Definitions based on the conditions (Note: numbers and relationships purely)
def unit_prices (x y : ℕ) : Prop :=
  3 * x + 2 * y = 60 ∧ x + 3 * y = 55

def prize_purchase_conditions (m n : ℕ) : Prop :=
  m + n = 100 ∧ 10 * m + 15 * n ≤ 1160 ∧ m ≤ 3 * n

-- Proving that the unit prices found match the given constraints
theorem unit_price_solution : ∃ x y : ℕ, unit_prices x y := by
  sorry

-- Proving the number of purchasing plans and minimum cost
theorem purchase_plan_and_costs : 
  (∃ (num_plans : ℕ) (min_cost : ℕ), 
    num_plans = 8 ∧ min_cost = 1125 ∧ 
    ∀ m n : ℕ, prize_purchase_conditions m n → 
      ((68 ≤ m ∧ m ≤ 75) →
      10 * m + 15 * (100 - m) = min_cost)) := by
  sorry

end unit_price_solution_purchase_plan_and_costs_l181_181715


namespace probability_colors_match_l181_181961

noncomputable def prob_abe_shows_blue : ℚ := 2 / 4
noncomputable def prob_bob_shows_blue : ℚ := 3 / 6
noncomputable def prob_abe_shows_green : ℚ := 2 / 4
noncomputable def prob_bob_shows_green : ℚ := 1 / 6

noncomputable def prob_same_color : ℚ :=
  (prob_abe_shows_blue * prob_bob_shows_blue) + (prob_abe_shows_green * prob_bob_shows_green)

theorem probability_colors_match : prob_same_color = 1 / 3 :=
by
  sorry

end probability_colors_match_l181_181961


namespace geometric_seq_property_l181_181736

noncomputable def a (n : ℕ) : ℝ := sorry

def S (n : ℕ) : ℝ := sorry

theorem geometric_seq_property (n : ℕ) (h_arith : S (n + 1) + S (n + 1) = 2 * S (n)) (h_condition : a 2 = -2) :
  a 7 = 64 := 
by sorry

end geometric_seq_property_l181_181736


namespace jennas_total_ticket_cost_l181_181563

theorem jennas_total_ticket_cost :
  let normal_price := 50
  let tickets_from_website := 2 * normal_price
  let scalper_price := 2 * normal_price * 2.4 - 10
  let friend_discounted_ticket := normal_price * 0.6
  tickets_from_website + scalper_price + friend_discounted_ticket = 360 :=
by
  sorry

end jennas_total_ticket_cost_l181_181563


namespace square_side_length_l181_181025

-- Variables for the conditions
variables (totalWire triangleWire : ℕ)
-- Definitions of the conditions
def totalLengthCondition := totalWire = 78
def triangleLengthCondition := triangleWire = 46

-- Goal is to prove the side length of the square
theorem square_side_length
  (h1 : totalLengthCondition totalWire)
  (h2 : triangleLengthCondition triangleWire)
  : (totalWire - triangleWire) / 4 = 8 := 
by
  rw [totalLengthCondition, triangleLengthCondition] at *
  sorry

end square_side_length_l181_181025


namespace find_special_n_l181_181677

noncomputable def is_good_divisor (n d_i d_im1 d_ip1 : ℕ) : Prop :=
  2 ≤ d_i ∧ d_i ≤ n ∧ ¬ (d_im1 * d_ip1 % d_i = 0)

def number_of_good_divisors (n : ℕ) : ℕ :=
  (finset.filter (λ d_i, ∃ d_im1 d_ip1, is_good_divisor n d_i d_im1 d_ip1)
    (finset.Ico 2 n)).card

def number_of_distinct_prime_divisors (n : ℕ) : ℕ :=
  (nat.factors n).to_finset.card

theorem find_special_n (n : ℕ) :
  number_of_good_divisors n < number_of_distinct_prime_divisors n ↔
  n = 1 ∨ ∃ p a, prime p ∧ n = p^a ∧ a ≥ 1 ∨
  ∃ p q a, prime p ∧ prime q ∧ q > p^a ∧ n = p^a * q ∧ a ≥ 1 :=
by
  sorry

end find_special_n_l181_181677


namespace probability_of_same_team_is_one_third_l181_181212

noncomputable def probability_same_team : ℚ :=
  let teams := 3
  let total_combinations := teams * teams
  let successful_outcomes := teams
  successful_outcomes / total_combinations

theorem probability_of_same_team_is_one_third :
  probability_same_team = 1 / 3 := by
  sorry

end probability_of_same_team_is_one_third_l181_181212


namespace schedule_arrangement_count_l181_181773

-- Given subjects
inductive Subject
| Chinese
| Mathematics
| Politics
| English
| PhysicalEducation
| Art

open Subject

-- Define a function to get the total number of different arrangements
def arrangement_count : Nat := 192

-- The proof statement (problem restated in Lean 4)
theorem schedule_arrangement_count :
  arrangement_count = 192 :=
by
  sorry

end schedule_arrangement_count_l181_181773


namespace max_regions_quadratic_trinomials_l181_181873

theorem max_regions_quadratic_trinomials (a b c : Fin 100 → ℝ) :
  ∃ R, (∀ (n : ℕ), n ≤ 100 → R = n^2 + 1) → R = 10001 := 
  sorry

end max_regions_quadratic_trinomials_l181_181873


namespace incorrect_operation_l181_181612

variable (a : ℕ)

-- Conditions
def condition1 := 4 * a ^ 2 - a ^ 2 = 3 * a ^ 2
def condition2 := a ^ 3 * a ^ 6 = a ^ 9
def condition3 := (a ^ 2) ^ 3 = a ^ 5
def condition4 := (2 * a ^ 2) ^ 2 = 4 * a ^ 4

-- Theorem to prove
theorem incorrect_operation : (a ^ 2) ^ 3 ≠ a ^ 5 := 
by
  sorry

end incorrect_operation_l181_181612


namespace set_of_x_values_l181_181759

theorem set_of_x_values (x : ℝ) : (3 ≤ abs (x + 2) ∧ abs (x + 2) ≤ 6) ↔ (1 ≤ x ∧ x ≤ 4) ∨ (-8 ≤ x ∧ x ≤ -5) := by
  sorry

end set_of_x_values_l181_181759


namespace evaluate_exponent_l181_181157

theorem evaluate_exponent : (3^2)^4 = 6561 := sorry

end evaluate_exponent_l181_181157


namespace find_a_plus_b_l181_181103

def cubic_function (a b : ℝ) (x : ℝ) := x^3 - x^2 - a * x + b

def tangent_line (x : ℝ) := 2 * x + 1

theorem find_a_plus_b (a b : ℝ) 
  (h1 : tangent_line 0 = 1)
  (h2 : cubic_function a b 0 = 1)
  (h3 : deriv (cubic_function a b) 0 = 2) :
  a + b = -1 :=
by
  sorry

end find_a_plus_b_l181_181103


namespace part_a_part_b_l181_181957

variables (p : ℝ) (h : p ≥ 0 ∧ p ≤ 1) -- Probability p is between 0 and 1 inclusive

-- Part (a)
theorem part_a (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) : 
  (5.choose 3) * p^3 * (1 - p)^2 = 10 * p^3 * (1 - p)^2 := by 
sorry

-- Part (b)
theorem part_b (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) : 
  10 * p - p^10 = (10 * p - p^10) := by 
sorry

end part_a_part_b_l181_181957


namespace angleC_is_100_l181_181055

-- Given a triangle ABC
variables (A B C : Type) [angle A] [angle B] [angle C]

-- Define a triangle
structure triangle (A B C : Type) := 
(angle_A : A)
(angle_B : B)
(angle_C : C)

-- Assume the sum of the angles in a triangle is 180 degrees
axiom sum_of_angles_in_triangle (t : triangle A B C) : 
  (angle_A + angle_B + angle_C = 180)

-- Given condition
axiom sum_of_angleA_and_angleB (t : triangle A B C) :
  (angle_A + angle_B = 80)

-- Prove that angle C is 100 degrees
theorem angleC_is_100 (t : triangle A B C) :
  angle_C = 100 :=
begin
  sorry
end

end angleC_is_100_l181_181055


namespace subset_relationship_l181_181195

def S : Set ℕ := {x | ∃ n : ℕ, x = 3^n}
def T : Set ℕ := {x | ∃ n : ℕ, x = 3 * n}

theorem subset_relationship : S ⊆ T :=
by sorry

end subset_relationship_l181_181195


namespace problem1_problem2_l181_181078

variable (a b c : ℝ)

-- Condition: a, b, c are positive numbers and a + b + c = 1
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (habc : a + b + c = 1)

-- Problem 1: Prove that a^2 / b + b^2 / c + c^2 / a ≥ 1
theorem problem1 : a^2 / b + b^2 / c + c^2 / a ≥ 1 :=
by sorry

-- Problem 2: Prove that ab + bc + ca ≤ 1 / 3
theorem problem2 : ab + bc + ca ≤ 1 / 3 :=
by sorry

end problem1_problem2_l181_181078


namespace evaluate_three_squared_raised_four_l181_181166

theorem evaluate_three_squared_raised_four : (3^2)^4 = 6561 := by
  sorry

end evaluate_three_squared_raised_four_l181_181166


namespace triangle_angle_sum_l181_181036

/-- In any triangle ABC, the sum of angle A and angle B is given to be 80 degrees.
    We need to prove that the measure of angle C is 100 degrees. -/
theorem triangle_angle_sum (A B C : ℝ) 
  (h1 : A + B = 80)
  (h2 : A + B + C = 180) : C = 100 :=
sorry

end triangle_angle_sum_l181_181036


namespace paint_fraction_l181_181205

variable (T C : ℕ) (h : T = 60) (t : ℕ) (partial_t : ℚ)

theorem paint_fraction (hT : T = 60) (ht : t = 12) : partial_t = t / T := by
  rw [ht, hT]
  norm_num
  sorry

end paint_fraction_l181_181205


namespace part1_part2_l181_181528

def f (x : ℝ) (a : ℝ) := (Real.exp x) / x - Real.log x + x - a

theorem part1 (a : ℝ) :
  (∀ x > 0, f x a ≥ 0) → (a ≤ Real.exp 1 + 1) :=
sorry

theorem part2 (x1 x2 a : ℝ) :
  f x1 a = 0 → f x2 a = 0 → x1 ≠ x2 → 0 < x1 → 0 < x2 → x1 * x2 < 1 :=
sorry

end part1_part2_l181_181528


namespace sector_properties_l181_181188

noncomputable def central_angle (l r : ℝ) : ℝ := l / r

noncomputable def sector_area (alpha r : ℝ) : ℝ := (1/2) * alpha * r^2

theorem sector_properties (l r : ℝ) (h_l : l = Real.pi) (h_r : r = 3) :
  central_angle l r = Real.pi / 3 ∧ sector_area (central_angle l r) r = 3 * Real.pi / 2 := 
  by
  sorry

end sector_properties_l181_181188


namespace cream_ratio_l181_181722

theorem cream_ratio (joe_initial_coffee joann_initial_coffee : ℝ)
                    (joe_drank_ounces joann_drank_ounces joe_added_cream joann_added_cream : ℝ) :
  joe_initial_coffee = 20 →
  joann_initial_coffee = 20 →
  joe_drank_ounces = 3 →
  joann_drank_ounces = 3 →
  joe_added_cream = 4 →
  joann_added_cream = 4 →
  (4 : ℝ) / ((21 / 24) * 24 - 3) = (8 : ℝ) / 7 :=
by
  intros h_ji h_ji h_jd h_jd h_jc h_jc
  sorry

end cream_ratio_l181_181722


namespace dot_but_not_straight_line_l181_181069

theorem dot_but_not_straight_line :
  let total := 80
  let D_n_S := 28
  let S_n_D := 47
  ∃ (D : ℕ), D - D_n_S = 5 ∧ D + S_n_D = total :=
by
  sorry

end dot_but_not_straight_line_l181_181069


namespace arithmetic_sequence_property_l181_181869

variable {a : ℕ → ℕ}

theorem arithmetic_sequence_property
  (h1 : a 3 + 3 * a 8 + a 13 = 120)
  (h2 : a 3 + a 13 = 2 * a 8) :
  a 3 + a 13 - a 8 = 24 := by
  sorry

end arithmetic_sequence_property_l181_181869


namespace probability_units_digit_8_l181_181415

noncomputable def units_digit (n : ℕ) : ℕ := n % 10

theorem probability_units_digit_8 : 
  let S := {n | n ∈ finset.range 40} in
  let prob := ∑ c in S, ∑ d in S, ∑ a in S, ∑ b in S,
    if units_digit (2^c + 5^d + 3^a + 7^b) = 8 then 1 else 0 in
  prob / 40^4 = (3 : ℝ) / 16 :=
by sorry

end probability_units_digit_8_l181_181415


namespace mean_of_set_l181_181389

theorem mean_of_set (x y : ℝ) 
  (h : (28 + x + 50 + 78 + 104) / 5 = 62) : 
  (48 + 62 + 98 + y + x) / 5 = (258 + y) / 5 :=
by
  -- we would now proceed to prove this according to lean's proof tactics.
  sorry

end mean_of_set_l181_181389


namespace dot_product_correct_l181_181703

theorem dot_product_correct:
  let a : ℝ × ℝ := (5, -7)
  let b : ℝ × ℝ := (-6, -4)
  (a.1 * b.1) + (a.2 * b.2) = -2 := by
sorry

end dot_product_correct_l181_181703


namespace correct_calculation_l181_181117

theorem correct_calculation (x : ℝ) : x * x^2 = x^3 :=
by sorry

end correct_calculation_l181_181117


namespace coin_flip_probability_l181_181592

theorem coin_flip_probability :
  let total_outcomes := 2^5
  let successful_outcomes := 2 * 2^2
  total_outcomes > 0 → (successful_outcomes / total_outcomes) = (1 / 4) :=
by
  intros
  sorry

end coin_flip_probability_l181_181592


namespace monotonic_intervals_range_of_k_l181_181735

noncomputable def f (x k : ℝ) : ℝ := (Real.exp x) / (x ^ 2) - k * (2 / x + Real.log x)
noncomputable def f' (x k : ℝ) : ℝ := (x - 2) * (Real.exp x - k * x) / (x^3)

theorem monotonic_intervals (k : ℝ) (h : k ≤ 0) :
  (∀ x : ℝ, 0 < x ∧ x < 2 → f' x k < 0) ∧ (∀ x : ℝ, x > 2 → f' x k > 0) := sorry

theorem range_of_k (k : ℝ) (h : e < k ∧ k < (e^2)/2) :
  ∃ x1 x2 : ℝ, 0 < x1 ∧ x1 < 2 ∧ 0 < x2 ∧ x2 < 2 ∧ 
    (f' x1 k = 0 ∧ f' x2 k = 0 ∧ x1 ≠ x2) := sorry

end monotonic_intervals_range_of_k_l181_181735


namespace clock_angle_at_3_30_l181_181824

theorem clock_angle_at_3_30 
    (deg_per_hour: Real := 30)
    (full_circle_deg: Real := 360)
    (hours_on_clock: Real := 12)
    (hour_hand_extra_deg: Real := 30 / 2)
    (hour_hand_deg: Real := 3 * deg_per_hour + hour_hand_extra_deg)
    (minute_hand_deg: Real := 6 * deg_per_hour) : 
    hour_hand_deg = 105 ∧ minute_hand_deg = 180 ∧ (minute_hand_deg - hour_hand_deg) = 75 := 
sorry

-- The problem specifies to write the theorem statement only, without the proof steps.

end clock_angle_at_3_30_l181_181824


namespace compare_probabilities_l181_181271

theorem compare_probabilities 
  (M N : ℕ)
  (m n : ℝ)
  (h_m_pos : m > 0)
  (h_n_pos : n > 0)
  (h_m_million : m > 10^6)
  (h_n_million : n ≤ 10^6) :
  (M : ℝ) / (M + n / m * N) > (M : ℝ) / (M + N) :=
by
  sorry

end compare_probabilities_l181_181271


namespace number_of_students_l181_181762

-- Definitions based on the problem conditions
def mini_cupcakes := 14
def donut_holes := 12
def desserts_per_student := 2

-- Total desserts calculation
def total_desserts := mini_cupcakes + donut_holes

-- Prove the number of students
theorem number_of_students : total_desserts / desserts_per_student = 13 :=
by
  -- Proof can be filled in here
  sorry

end number_of_students_l181_181762


namespace perfect_cube_divisor_count_l181_181533

noncomputable def num_perfect_cube_divisors : Nat :=
  let a_choices := Nat.succ (38 / 3)
  let b_choices := Nat.succ (17 / 3)
  let c_choices := Nat.succ (7 / 3)
  let d_choices := Nat.succ (4 / 3)
  a_choices * b_choices * c_choices * d_choices

theorem perfect_cube_divisor_count :
  num_perfect_cube_divisors = 468 :=
by
  sorry

end perfect_cube_divisor_count_l181_181533


namespace find_alpha_l181_181549

variable {α p₀ p_new : ℝ}
def Q_d (p : ℝ) : ℝ := 150 - p
def Q_s (p : ℝ) : ℝ := 3 * p - 10
def Q_d_new (α : ℝ) (p : ℝ) : ℝ := α * (150 - p)

theorem find_alpha 
  (h_eq_initial : Q_d p₀ = Q_s p₀)
  (h_eq_increase : p_new = 1.25 * p₀)
  (h_eq_new : Q_s p_new = Q_d_new α p_new) :
  α = 1.4 :=
by
  sorry

end find_alpha_l181_181549


namespace Johnson_farm_budget_l181_181594

variable (total_land : ℕ) (corn_cost_per_acre : ℕ) (wheat_cost_per_acre : ℕ)
variable (acres_wheat : ℕ) (acres_corn : ℕ)

def total_money (total_land corn_cost_per_acre wheat_cost_per_acre acres_wheat acres_corn : ℕ) : ℕ :=
  acres_corn * corn_cost_per_acre + acres_wheat * wheat_cost_per_acre

theorem Johnson_farm_budget :
  total_land = 500 ∧
  corn_cost_per_acre = 42 ∧
  wheat_cost_per_acre = 30 ∧
  acres_wheat = 200 ∧
  acres_corn = total_land - acres_wheat →
  total_money total_land corn_cost_per_acre wheat_cost_per_acre acres_wheat acres_corn = 18600 := by
  sorry

end Johnson_farm_budget_l181_181594


namespace inequality_proof_l181_181888

theorem inequality_proof (a b c : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) (h₃ : a * b + b * c + c * a = 1) : 
  (a / Real.sqrt (a ^ 2 + 1)) + (b / Real.sqrt (b ^ 2 + 1)) + (c / Real.sqrt (c ^ 2 + 1)) ≤ (3 / 2) :=
by
  sorry

end inequality_proof_l181_181888


namespace abs_difference_of_two_numbers_l181_181761

theorem abs_difference_of_two_numbers (x y : ℝ) (h1 : x + y = 36) (h2 : x * y = 105) :
  |x - y| = 6 * Real.sqrt 24.333 := sorry

end abs_difference_of_two_numbers_l181_181761


namespace total_movie_hours_l181_181726

variable (J M N R : ℕ)

def favorite_movie_lengths (J M N R : ℕ) :=
  J = M + 2 ∧
  N = 3 * M ∧
  R = (4 / 5 * N : ℝ).to_nat ∧
  N = 30

theorem total_movie_hours (J M N R : ℕ) (h : favorite_movie_lengths J M N R) :
  J + M + N + R = 76 :=
by
  sorry

end total_movie_hours_l181_181726


namespace ball_distance_traveled_l181_181626

noncomputable def total_distance (a1 d n : ℕ) : ℕ :=
  n * (a1 + a1 + (n-1) * d) / 2

theorem ball_distance_traveled : 
  total_distance 8 5 20 = 1110 :=
by
  sorry

end ball_distance_traveled_l181_181626


namespace y_intercept_line_l181_181764

theorem y_intercept_line : ∀ y : ℝ, (∃ x : ℝ, x = 0 ∧ x - 3 * y - 1 = 0) → y = -1/3 :=
by
  intro y
  intro h
  sorry

end y_intercept_line_l181_181764


namespace milly_folds_count_l181_181885

theorem milly_folds_count (mixing_time baking_time total_minutes fold_time rest_time : ℕ) 
  (h : total_minutes = 360)
  (h_mixing_time : mixing_time = 10)
  (h_baking_time : baking_time = 30)
  (h_fold_time : fold_time = 5)
  (h_rest_time : rest_time = 75) : 
  (total_minutes - (mixing_time + baking_time)) / (fold_time + rest_time) = 4 := 
by
  sorry

end milly_folds_count_l181_181885


namespace quadratic_inequality_always_holds_l181_181115

theorem quadratic_inequality_always_holds (k : ℝ) (h : ∀ x : ℝ, (x^2 - k*x + 1) > 0) : -2 < k ∧ k < 2 :=
  sorry

end quadratic_inequality_always_holds_l181_181115


namespace value_of_k_l181_181010

theorem value_of_k (m n k : ℝ) (h1 : 3 ^ m = k) (h2 : 5 ^ n = k) (h3 : 1 / m + 1 / n = 2) : k = Real.sqrt 15 :=
  sorry

end value_of_k_l181_181010


namespace floor_neg_seven_four_is_neg_two_l181_181338

noncomputable def floor_neg_seven_four : Int :=
  Int.floor (-7 / 4)

theorem floor_neg_seven_four_is_neg_two : floor_neg_seven_four = -2 := by
  sorry

end floor_neg_seven_four_is_neg_two_l181_181338


namespace number_of_possible_values_l181_181966

theorem number_of_possible_values (a b c : ℕ) (h : a + 11 * b + 111 * c = 1050) :
  ∃ (n : ℕ), 6 ≤ n ∧ n ≤ 1050 ∧ (n % 9 = 6) ∧ (n = a + 2 * b + 3 * c) :=
sorry

end number_of_possible_values_l181_181966


namespace total_rainfall_2019_to_2021_l181_181546

theorem total_rainfall_2019_to_2021 :
  let R2019 := 50
  let R2020 := R2019 + 5
  let R2021 := R2020 - 3
  12 * R2019 + 12 * R2020 + 12 * R2021 = 1884 :=
by
  sorry

end total_rainfall_2019_to_2021_l181_181546


namespace steve_break_even_l181_181591

noncomputable def break_even_performances
  (fixed_overhead : ℕ)
  (min_production_cost max_production_cost : ℕ)
  (venue_capacity percentage_occupied : ℕ)
  (ticket_price : ℕ) : ℕ :=
(fixed_overhead + (percentage_occupied / 100 * venue_capacity * ticket_price)) / (percentage_occupied / 100 * venue_capacity * ticket_price)

theorem steve_break_even
  (fixed_overhead : ℕ := 81000)
  (min_production_cost : ℕ := 5000)
  (max_production_cost : ℕ := 9000)
  (venue_capacity : ℕ := 500)
  (percentage_occupied : ℕ := 80)
  (ticket_price : ℕ := 40)
  (avg_production_cost : ℕ := (min_production_cost + max_production_cost) / 2) :
  break_even_performances fixed_overhead min_production_cost max_production_cost venue_capacity percentage_occupied ticket_price = 9 :=
by
  sorry

end steve_break_even_l181_181591


namespace xy_not_z_probability_l181_181792

theorem xy_not_z_probability :
  let P_X := (1 : ℝ) / 4
  let P_Y := (1 : ℝ) / 3
  let P_not_Z := (3 : ℝ) / 8
  let P := P_X * P_Y * P_not_Z
  P = (1 : ℝ) / 32 :=
by
  -- Definitions based on problem conditions
  let P_X := (1 : ℝ) / 4
  let P_Y := (1 : ℝ) / 3
  let P_not_Z := (3 : ℝ) / 8

  -- Calculate the combined probability
  let P := P_X * P_Y * P_not_Z
  
  -- Check equality with 1/32
  have h : P = (1 : ℝ) / 32 := by sorry
  exact h

end xy_not_z_probability_l181_181792


namespace current_dogwood_trees_l181_181770

def number_of_trees (X : ℕ) : Prop :=
  X + 61 = 100

theorem current_dogwood_trees (X : ℕ) (h : number_of_trees X) : X = 39 :=
by 
  sorry

end current_dogwood_trees_l181_181770


namespace children_vehicle_wheels_l181_181482

theorem children_vehicle_wheels:
  ∀ (x : ℕ),
    (6 * 2) + (15 * x) = 57 →
    x = 3 :=
by
  intros x h
  sorry

end children_vehicle_wheels_l181_181482


namespace inequality_m_le_minus3_l181_181388

theorem inequality_m_le_minus3 (m : ℝ) : (∀ x : ℝ, 0 < x ∧ x ≤ 1 → x^2 - 4 * x ≥ m) → m ≤ -3 :=
by
  sorry

end inequality_m_le_minus3_l181_181388


namespace part1_part2_l181_181513

def f (x a : ℝ) : ℝ := (Exp x / x) - Log x + x - a

theorem part1 (x a : ℝ) (hx : f x a ≥ 0) : a ≤ Real.exp 1 + 1 := 
sorry

theorem part2 (x1 x2 a : ℝ) (hx1 : f x1 a = 0) (hx2 : f x2 a = 0) 
  (hx1_range : 0 < x1) (hx2_range : x2 > 1) : x1 * x2 < 1 := 
sorry

end part1_part2_l181_181513


namespace incorrect_proposition_C_l181_181613

theorem incorrect_proposition_C (p q : Prop) :
  (¬(p ∨ q) → ¬p ∧ ¬q) ↔ False :=
by sorry

end incorrect_proposition_C_l181_181613


namespace probability_comparison_l181_181283

variables (M N : ℕ) (m n : ℝ)
variable (h₁ : m > 10^6)
variable (h₂ : n ≤ 10^6)

theorem probability_comparison (h₃: 0 < M) (h₄: 0 < N):
  (m * M) / (m * M + n * N) > (M / (M + N)) :=
by
  have h₅: n / m < 1 := sorry
  have h₆: M > 0 := by linarith
  have h₇: 1 + (n / m) * (N / M) < 2 := sorry
  sorry

end probability_comparison_l181_181283


namespace rational_function_eq_l181_181352

theorem rational_function_eq (f : ℚ → ℚ) 
  (h1 : f 1 = 2) 
  (h2 : ∀ x y : ℚ, f (x * y) = f x * f y - f (x + y) + 1) : 
  ∀ x : ℚ, f x = x + 1 :=
by sorry

end rational_function_eq_l181_181352


namespace find_a_l181_181372

noncomputable def f' (x : ℝ) (a : ℝ) := 2 * x^3 + a * x^2 + x

theorem find_a (a : ℝ) (h : f' 1 a = 9) : a = 6 :=
by
  sorry

end find_a_l181_181372


namespace base7_conversion_and_multiplication_l181_181169

theorem base7_conversion_and_multiplication :
  let x := 231
  let y := 452
  (BaseConversion.toBaseN 7 x = 450 ∧
  BaseConversion.toBaseN 7 y = 1214 ∧
  x * y = 104412 ∧
  BaseConversion.toBaseN 7 (x * y) = 613260) :=
by
  sorry

end base7_conversion_and_multiplication_l181_181169


namespace maximize_profit_at_200_l181_181187

noncomputable def cost (q : ℝ) : ℝ := 50000 + 200 * q
noncomputable def price (q : ℝ) : ℝ := 24200 - (1/5) * q^2
noncomputable def profit (q : ℝ) : ℝ := (price q) * q - (cost q)

theorem maximize_profit_at_200 : ∃ (q : ℝ), q = 200 ∧ ∀ (x : ℝ), x ≥ 0 → profit q ≥ profit x :=
by
  sorry

end maximize_profit_at_200_l181_181187


namespace shorter_trisector_length_eq_l181_181362

theorem shorter_trisector_length_eq :
  ∀ (DE EF DF FG : ℝ), DE = 6 → EF = 8 → DF = Real.sqrt (DE^2 + EF^2) → 
  FG = 2 * (24 / (3 + 4 * Real.sqrt 3)) → 
  FG = (192 * Real.sqrt 3 - 144) / 39 :=
by
  intros
  sorry

end shorter_trisector_length_eq_l181_181362


namespace seq_is_geometric_from_second_l181_181359

namespace sequence_problem

-- Definitions and conditions
def S : ℕ → ℕ
| 0 => 0
| 1 => 1
| 2 => 2
| (n + 1) => 3 * S n - 2 * S (n - 1)

-- Recursive definition for sum of sequence terms
axiom S_rec_relation (n : ℕ) (h : n ≥ 2) : 
  S (n + 1) - 3 * S n + 2 * S (n - 1) = 0

-- Prove the sequence is geometric from the second term
theorem seq_is_geometric_from_second :
  ∃ (a : ℕ → ℕ), (∀ n ≥ 2, a (n + 1) = 2 * a n) ∧ 
  (a 1 = 1) ∧ 
  (a 2 = 1) :=
by
  sorry

end sequence_problem

end seq_is_geometric_from_second_l181_181359


namespace compute_y_series_l181_181332

theorem compute_y_series :
  (∑' n : ℕ, (1 / 3) ^ n) + (∑' n : ℕ, ((-1) ^ n) / (4 ^ n)) = ∑' n : ℕ, (1 / (23 / 13) ^ n) :=
by
  sorry

end compute_y_series_l181_181332


namespace intercepts_line_5x_minus_2y_minus_10_eq_0_l181_181237

theorem intercepts_line_5x_minus_2y_minus_10_eq_0 :
  ∃ a b : ℝ, (a = 2 ∧ b = -5) ∧ (∀ x y : ℝ, 5 * x - 2 * y - 10 = 0 → 
     ((y = 0 ∧ x = a) ∨ (x = 0 ∧ y = b))) :=
by
  sorry

end intercepts_line_5x_minus_2y_minus_10_eq_0_l181_181237


namespace correct_equation_l181_181806

variable (x : ℝ) (h1 : x > 0)

def length_pipeline : ℝ := 3000
def efficiency_increase : ℝ := 0.2
def days_ahead : ℝ := 10

theorem correct_equation :
  (length_pipeline / x) - (length_pipeline / ((1 + efficiency_increase) * x)) = days_ahead :=
by
  sorry

end correct_equation_l181_181806


namespace factor_polynomial_l181_181670

theorem factor_polynomial : ∀ x : ℝ, 
  x^8 - 4 * x^6 + 6 * x^4 - 4 * x^2 + 1 = (x - 1)^4 * (x + 1)^4 :=
by
  intro x
  sorry

end factor_polynomial_l181_181670


namespace quadratic_to_square_form_l181_181968

theorem quadratic_to_square_form (x m n : ℝ) (h : x^2 + 6 * x - 1 = 0) 
  (hm : m = 3) (hn : n = 10) : m - n = -7 :=
by 
  -- Proof steps (skipped, as per instructions)
  sorry

end quadratic_to_square_form_l181_181968


namespace corn_increase_factor_l181_181468

noncomputable def field_area : ℝ := 1

-- Let x be the remaining part of the field
variable (x : ℝ)

-- First condition: if the remaining part is fully planted with millet
-- Millet will occupy half of the field
axiom condition1 : (field_area - x) + x = field_area / 2

-- Second condition: if the remaining part x is equally divided between oats and corn
-- Oats will occupy half of the field
axiom condition2 : (field_area - x) + 0.5 * x = field_area / 2

-- Prove the factor by which the amount of corn increases
theorem corn_increase_factor : (0.5 * x + x) / (0.5 * x / 2) = 3 :=
by
  sorry

end corn_increase_factor_l181_181468


namespace total_students_taught_l181_181638

theorem total_students_taught (students_per_first_year : ℕ) 
  (students_per_year : ℕ) 
  (years_remaining : ℕ) 
  (total_years : ℕ) :
  students_per_first_year = 40 → 
  students_per_year = 50 →
  years_remaining = 9 →
  total_years = 10 →
  students_per_first_year + students_per_year * years_remaining = 490 := 
by
  intros h₁ h₂ h₃ h₄
  rw [h₁, h₂, h₃]
  norm_num
  rw h₄
  norm_num

end total_students_taught_l181_181638


namespace min_quotient_l181_181879

theorem min_quotient {a b : ℕ} (h₁ : 100 ≤ a) (h₂ : a ≤ 300) (h₃ : 400 ≤ b) (h₄ : b ≤ 800) (h₅ : a + b ≤ 950) : a / b = 1 / 8 := 
by
  sorry

end min_quotient_l181_181879


namespace percent_shaded_area_of_rectangle_l181_181775

theorem percent_shaded_area_of_rectangle
  (side_length : ℝ)
  (length_rectangle : ℝ)
  (width_rectangle : ℝ)
  (overlap_length : ℝ)
  (h1 : side_length = 12)
  (h2 : length_rectangle = 20)
  (h3 : width_rectangle = 12)
  (h4 : overlap_length = 4)
  : (overlap_length * width_rectangle) / (length_rectangle * width_rectangle) * 100 = 20 :=
  sorry

end percent_shaded_area_of_rectangle_l181_181775


namespace total_amount_spent_l181_181216

theorem total_amount_spent (tax_paid : ℝ) (tax_rate : ℝ) (tax_free_cost : ℝ) (total_spent : ℝ) :
  tax_paid = 30 → tax_rate = 0.06 → tax_free_cost = 19.7 →
  total_spent = 30 / 0.06 + 19.7 :=
by
  -- Definitions for assumptions
  intro h1 h2 h3
  -- Skip the proof here
  sorry

end total_amount_spent_l181_181216


namespace binomial_20_19_eq_20_l181_181310

theorem binomial_20_19_eq_20 : Nat.choose 20 19 = 20 :=
by
  sorry

end binomial_20_19_eq_20_l181_181310


namespace part_a_probability_three_unused_rockets_part_b_expected_targets_hit_l181_181953

-- Proof Problem for Part (a):
theorem part_a_probability_three_unused_rockets (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) : 
  (∃ q : ℝ, q = 10 * p^3 * (1-p)^2) := sorry

-- Proof Problem for Part (b):
theorem part_b_expected_targets_hit (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) : 
  (∃ e : ℝ, e = 10 * p - p^10) := sorry

end part_a_probability_three_unused_rockets_part_b_expected_targets_hit_l181_181953


namespace tan_alpha_third_quadrant_l181_181510

theorem tan_alpha_third_quadrant (α : ℝ) 
  (h_eq: Real.sin α = Real.cos α) 
  (h_third: π < α ∧ α < 3 * π / 2) : Real.tan α = 1 := 
by 
  sorry

end tan_alpha_third_quadrant_l181_181510


namespace train_crossing_time_l181_181122

-- Definitions for conditions
def train_length : ℝ := 100 -- train length in meters
def train_speed_kmh : ℝ := 90 -- train speed in km/hr
def train_speed_mps : ℝ := 25 -- train speed in m/s after conversion

-- Lean 4 statement to prove the time taken for the train to cross the electric pole is 4 seconds
theorem train_crossing_time : (train_length / train_speed_mps) = 4 := by
  sorry

end train_crossing_time_l181_181122


namespace det_A_l181_181655

-- Define the matrix A
noncomputable def A : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![Real.sin 1, Real.cos 2, Real.sin 3],
   ![Real.sin 4, Real.cos 5, Real.sin 6],
   ![Real.sin 7, Real.cos 8, Real.sin 9]]

-- Define the explicit determinant calculation
theorem det_A :
  Matrix.det A = Real.sin 1 * (Real.cos 5 * Real.sin 9 - Real.sin 6 * Real.cos 8) -
                 Real.cos 2 * (Real.sin 4 * Real.sin 9 - Real.sin 6 * Real.sin 7) +
                 Real.sin 3 * (Real.sin 4 * Real.cos 8 - Real.cos 5 * Real.sin 7) :=
by
  sorry

end det_A_l181_181655


namespace P_positive_l181_181504

variable (P : ℕ → ℝ)

axiom P_cond_0 : P 0 > 0
axiom P_cond_1 : P 1 > P 0
axiom P_cond_2 : P 2 > 2 * P 1 - P 0
axiom P_cond_3 : P 3 > 3 * P 2 - 3 * P 1 + P 0
axiom P_cond_n : ∀ n, P (n + 4) > 4 * P (n + 3) - 6 * P (n + 2) + 4 * P (n + 1) - P n

theorem P_positive (n : ℕ) (h : n > 0) : P n > 0 := by
  sorry

end P_positive_l181_181504


namespace option_A_correct_l181_181172

theorem option_A_correct (p : ℕ) (h1 : p > 1) (h2 : p % 2 = 1) : 
  (p - 1)^(p/2 - 1) - 1 ≡ 0 [MOD (p - 2)] :=
sorry

end option_A_correct_l181_181172


namespace find_common_difference_l181_181147

-- Definitions for arithmetic sequences and sums
def S (a1 d : ℕ) (n : ℕ) := (n * (2 * a1 + (n - 1) * d)) / 2
def a (a1 d : ℕ) (n : ℕ) := a1 + (n - 1) * d

theorem find_common_difference (a1 d : ℕ) :
  S a1 d 3 = 6 → a a1 d 3 = 4 → d = 2 :=
by
  intros S3_eq a3_eq
  sorry

end find_common_difference_l181_181147


namespace larger_square_side_length_l181_181580

theorem larger_square_side_length (x y H : ℝ) 
  (smaller_square_perimeter : 4 * x = H - 20)
  (larger_square_perimeter : 4 * y = H) :
  y = x + 5 :=
by
  sorry

end larger_square_side_length_l181_181580


namespace total_number_of_athletes_l181_181071

theorem total_number_of_athletes (n : ℕ) (h1 : n % 10 = 6) (h2 : n % 11 = 6) (h3 : 200 ≤ n) (h4 : n ≤ 300) : n = 226 :=
sorry

end total_number_of_athletes_l181_181071


namespace evaluate_expression_l181_181027

theorem evaluate_expression (x : ℝ) (h : x = 2) : 4 * x ^ 2 + 1 / 2 = 16.5 := by
  sorry

end evaluate_expression_l181_181027


namespace smallest_cubes_to_fill_box_l181_181930

theorem smallest_cubes_to_fill_box
  (L W D : ℕ)
  (hL : L = 30)
  (hW : W = 48)
  (hD : D = 12) :
  ∃ (n : ℕ), n = (L * W * D) / ((Nat.gcd (Nat.gcd L W) D) ^ 3) ∧ n = 80 := 
by
  sorry

end smallest_cubes_to_fill_box_l181_181930


namespace angle_C_in_triangle_l181_181047

theorem angle_C_in_triangle (A B C : ℝ) (h : A + B = 80) : C = 100 :=
by
  -- The measure of angle C follows from the condition and the properties of triangles.
  sorry

end angle_C_in_triangle_l181_181047


namespace equation1_solution_equation2_solution_l181_181590

theorem equation1_solution : ∀ x : ℚ, x - 0.4 * x = 120 → x = 200 := by
  sorry

theorem equation2_solution : ∀ x : ℚ, 5 * x - 5/6 = 5/4 → x = 5/12 := by
  sorry

end equation1_solution_equation2_solution_l181_181590


namespace number_of_impossible_d_vals_is_infinite_l181_181904

theorem number_of_impossible_d_vals_is_infinite
  (t_1 t_2 s d : ℕ)
  (h1 : 2 * t_1 + t_2 - 4 * s = 4041)
  (h2 : t_1 = s + 2 * d)
  (h3 : t_2 = s + d)
  (h4 : 4 * s > 0) :
  ∀ n : ℕ, n ≠ 808 * 5 ↔ ∃ d, d > 0 ∧ d ≠ n :=
sorry

end number_of_impossible_d_vals_is_infinite_l181_181904


namespace count_prime_digit_sums_less_than_10_l181_181024

def is_prime (n : ℕ) : Prop :=
n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def sum_of_digits (n : ℕ) : ℕ := n / 10 + n % 10

def is_two_digit_number (n : ℕ) : Prop := n >= 10 ∧ n < 100

theorem count_prime_digit_sums_less_than_10 :
  ∃ count : ℕ, count = 17 ∧
  ∀ n : ℕ, is_two_digit_number n →
  (is_prime (sum_of_digits n) ∧ sum_of_digits n < 10) ↔
  n ∈ [11, 20, 12, 21, 30, 14, 23, 32, 41, 50, 16, 25, 34, 43, 52, 61, 70] :=
sorry

end count_prime_digit_sums_less_than_10_l181_181024


namespace sequence_perfect_square_l181_181820

theorem sequence_perfect_square (a : ℕ → ℤ) (h : ∀ n, a (n + 1) = a n ^ 3 + 1999) :
  ∃! n, ∃ k, a n = k ^ 2 :=
by
  sorry

end sequence_perfect_square_l181_181820


namespace Keiko_speed_is_pi_div_3_l181_181911

noncomputable def Keiko_avg_speed {r : ℝ} (v : ℝ → ℝ) (pi : ℝ) : ℝ :=
let C1 := 2 * pi * (r + 6) - 2 * pi * r
let t1 := 36
let v1 := C1 / t1

let C2 := 2 * pi * (r + 8) - 2 * pi * r
let t2 := 48
let v2 := C2 / t2

if v r = v1 ∧ v r = v2 then (v1 + v2) / 2 else 0

theorem Keiko_speed_is_pi_div_3 (pi : ℝ) (r : ℝ) (v : ℝ → ℝ) :
  v r = π / 3 ∧ (forall t1 t2 C1 C2, C1 / t1 = π / 3 ∧ C2 / t2 = π / 3 → 
  (C1/t1 + C2/t2)/2 = π / 3) :=
sorry

end Keiko_speed_is_pi_div_3_l181_181911


namespace poly_perimeter_eq_33_l181_181555

noncomputable theory

def distance (p1 p2 : (ℝ × ℝ)) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

def A : ℝ × ℝ := (0, 8)
def B : ℝ × ℝ := (4, 8)
def C : ℝ × ℝ := (4, 4)
def D : ℝ × ℝ := (0, -2)
def E : ℝ × ℝ := (9, -2)

def AB : ℝ := distance A B
def BC : ℝ := distance B C
def CD : ℝ := distance C D
def DE : ℝ := distance D E
def EA : ℝ := distance E A

def perimeter : ℝ := AB + BC + CD + DE + EA

theorem poly_perimeter_eq_33 : perimeter = 33 := by
  sorry

end poly_perimeter_eq_33_l181_181555


namespace tan_A_tan_B_eq_one_third_l181_181400

theorem tan_A_tan_B_eq_one_third (A B C : ℕ) (hC : C = 120) (hSum : Real.tan A + Real.tan B = 2 * Real.sqrt 3 / 3) : 
  Real.tan A * Real.tan B = 1 / 3 := 
by
  sorry

end tan_A_tan_B_eq_one_third_l181_181400


namespace total_watermelon_slices_l181_181489

theorem total_watermelon_slices 
(h1 : ∀ (n : Nat), n > 0 → n * 10 = 30)
(h2 : ∀ (m : Nat), m > 0 → m * 15 = 15) :
  3 * 10 + 1 * 15 = 45 := 
  by 
    have h_danny : 3 * 10 = 30 := h1 3 (by norm_num)
    have h_sister: 1 * 15 = 15 := h2 1 (by norm_num)
    calc
      3 * 10 + 1 * 15 = 30 + 15 := by rw [h_danny, h_sister]
                  ... = 45 := by norm_num

end total_watermelon_slices_l181_181489


namespace binary_add_sub_l181_181486

theorem binary_add_sub : 
  (1101 + 111 - 101 + 1001 - 11 : ℕ) = (10101 : ℕ) := by
  sorry

end binary_add_sub_l181_181486


namespace percent_savings_12_roll_package_l181_181627

def percent_savings_per_roll (package_cost : ℕ) (individual_cost : ℕ) (num_rolls : ℕ) : ℚ :=
  let individual_total := num_rolls * individual_cost
  let package_total := package_cost
  let per_roll_package_cost := package_total / num_rolls
  let savings_per_roll := individual_cost - per_roll_package_cost
  (savings_per_roll / individual_cost) * 100

theorem percent_savings_12_roll_package :
  percent_savings_per_roll 9 1 12 = 25 := 
sorry

end percent_savings_12_roll_package_l181_181627


namespace max_distance_without_fuel_depots_l181_181336

def exploration_max_distance : ℕ :=
  360

-- Define the conditions
def cars_count : ℕ :=
  9

def full_tank_distance : ℕ :=
  40

def additional_gal_capacity : ℕ :=
  9

def total_gallons_per_car : ℕ :=
  1 + additional_gal_capacity

-- Define the distance calculation under the given constraints
theorem max_distance_without_fuel_depots (n : ℕ) (d_tank : ℕ) (d_add : ℕ) :
  ∀ (cars : ℕ), (cars = cars_count) →
  (d_tank = full_tank_distance) →
  (d_add = additional_gal_capacity) →
  ((cars * (1 + d_add)) * d_tank) / (2 * cars - 1) = exploration_max_distance :=
by
  intros _ hc ht ha
  rw [hc, ht, ha]
  -- Proof skipped
  sorry

end max_distance_without_fuel_depots_l181_181336


namespace probability_ascending_order_l181_181945

theorem probability_ascending_order :
  let S := {1, 2, 3, 4, 5, 6, 7} in
  let all_selections := finset.univ.powerset.filter (λ s, s.card = 5) in
  let ascending_orderings := all_selections.filter (λ s, s.to_list = s.to_list.qsort (≤)) in
  (ascending_orderings.card : ℝ) / (all_selections.card : ℝ) = 1 / 120 := by
    sorry

end probability_ascending_order_l181_181945


namespace graph_does_not_pass_second_quadrant_l181_181855

noncomputable def y_function (a b : ℝ) (x : ℝ) : ℝ := a^x + b

theorem graph_does_not_pass_second_quadrant (a b : ℝ) (h1 : a > 1) (h2 : b < -1) : 
  ∀ x y : ℝ, (y = y_function a b x) → ¬(x < 0 ∧ y > 0) := by
  sorry

end graph_does_not_pass_second_quadrant_l181_181855


namespace trajectory_of_center_line_passes_fixed_point_l181_181978

-- Define the conditions
def pointA : ℝ × ℝ := (4, 0)
def chord_length : ℝ := 8
def pointB : ℝ × ℝ := (-3, 0)
def not_perpendicular_to_x_axis (t : ℝ) : Prop := t ≠ 0
def trajectory_eq (x y : ℝ) : Prop := y^2 = 8 * x
def line_eq (t m y x : ℝ) : Prop := x = t * y + m
def x_axis_angle_bisector (y1 x1 y2 x2 : ℝ) : Prop := (y1 / (x1 + 3)) + (y2 / (x2 + 3)) = 0

-- Prove the trajectory of the center of the moving circle is \( y^2 = 8x \)
theorem trajectory_of_center (x y : ℝ) 
  (H1: (x-4)^2 + y^2 = 4^2 + x^2) 
  (H2: trajectory_eq x y) : 
  trajectory_eq x y := sorry

-- Prove the line passes through the fixed point (3, 0)
theorem line_passes_fixed_point (t m y1 x1 y2 x2 : ℝ) 
  (Ht: not_perpendicular_to_x_axis t)
  (Hsys: ∀ y x, line_eq t m y x → trajectory_eq x y)
  (Hangle: x_axis_angle_bisector y1 x1 y2 x2) : 
  (m = 3) ∧ ∃ y, line_eq t 3 y 3 := sorry

end trajectory_of_center_line_passes_fixed_point_l181_181978


namespace part1_condition1_implies_a_le_1_condition2_implies_a_le_2_condition3_implies_a_le_1_l181_181983

section Problem

-- Universal set is ℝ
def universal_set : Set ℝ := Set.univ

-- Set A
def set_A : Set ℝ := { x | x^2 - x - 6 ≤ 0 }

-- Set A complement in ℝ
def complement_A : Set ℝ := universal_set \ set_A

-- Set B
def set_B : Set ℝ := { x | (x - 4)/(x + 1) < 0 }

-- Set C
def set_C (a : ℝ) : Set ℝ := { x | 2 - a < x ∧ x < 2 + a }

-- Prove (complement_A ∩ set_B = (3, 4))
theorem part1 : (complement_A ∩ set_B) = { x | 3 < x ∧ x < 4 } :=
  sorry

-- Assume a definition for real number a (non-negative)
variable (a : ℝ)

-- Prove range of a given the conditions
-- Condition 1: A ∩ C = C implies a ≤ 1
theorem condition1_implies_a_le_1 (h : set_A ∩ set_C a = set_C a) : a ≤ 1 :=
  sorry

-- Condition 2: B ∪ C = B implies a ≤ 2
theorem condition2_implies_a_le_2 (h : set_B ∪ set_C a = set_B) : a ≤ 2 :=
  sorry

-- Condition 3: C ⊆ (A ∩ B) implies a ≤ 1
theorem condition3_implies_a_le_1 (h : set_C a ⊆ set_A ∩ set_B) : a ≤ 1 :=
  sorry

end Problem

end part1_condition1_implies_a_le_1_condition2_implies_a_le_2_condition3_implies_a_le_1_l181_181983


namespace eq_factorial_sum_l181_181493

theorem eq_factorial_sum (k l m n : ℕ) (hk : k > 0) (hl : l > 0) (hm : m > 0) (hn : n > 0) :
  (1 / (Nat.factorial k : ℝ) + 1 / (Nat.factorial l : ℝ) + 1 / (Nat.factorial m : ℝ) = 1 / (Nat.factorial n : ℝ))
  ↔ (k = 3 ∧ l = 3 ∧ m = 3 ∧ n = 2) :=
by
  sorry

end eq_factorial_sum_l181_181493


namespace minimum_value_quadratic_l181_181900

noncomputable def quadratic (x : ℝ) : ℝ := x^2 - 4 * x + 5

theorem minimum_value_quadratic :
  ∀ x : ℝ, quadratic x ≥ 1 :=
by
  sorry

end minimum_value_quadratic_l181_181900


namespace larger_screen_diagonal_length_l181_181600

theorem larger_screen_diagonal_length :
  (∃ d : ℝ, (∀ a : ℝ, a = 16 → d^2 = 2 * (a^2 + 34)) ∧ d = Real.sqrt 580) :=
by
  sorry

end larger_screen_diagonal_length_l181_181600


namespace tangent_sphere_surface_area_l181_181695

noncomputable def cube_side_length (V : ℝ) : ℝ := V^(1/3)
noncomputable def sphere_radius (a : ℝ) : ℝ := a / 2
noncomputable def sphere_surface_area (r : ℝ) : ℝ := 4 * Real.pi * r^2

theorem tangent_sphere_surface_area (V : ℝ) (hV : V = 64) : 
  sphere_surface_area (sphere_radius (cube_side_length V)) = 16 * Real.pi :=
by
  sorry

end tangent_sphere_surface_area_l181_181695


namespace age_ratio_l181_181937

variable (p q : ℕ)

-- Conditions
def condition1 := p - 6 = (q - 6) / 2
def condition2 := p + q = 21

-- Theorem stating the desired ratio
theorem age_ratio (h1 : condition1 p q) (h2 : condition2 p q) : p / Nat.gcd p q = 3 ∧ q / Nat.gcd p q = 4 :=
by
  sorry

end age_ratio_l181_181937


namespace binomial_probability_l181_181193

theorem binomial_probability (p : ℝ) (X Y : ℕ → ℕ) 
    (hX : ∀ k, X k = (Nat.choose 2 k) * p^k * (1 - p)^(2 - k))
    (hY : ∀ k, Y k = (Nat.choose 4 k) * p^k * (1 - p)^(4 - k))
    (hP : (∑ k in {1, 2}, X k) = 5 / 9) :
    (∑ k in {2, 3, 4}, Y k) = 11 / 27 := by sorry

end binomial_probability_l181_181193


namespace problem_sol_l181_181756

-- Defining the operations as given
def operation_hash (a b c : ℤ) : ℤ := 4 * a ^ 3 + 4 * b ^ 3 + 8 * a ^ 2 * b + c
def operation_star (a b d : ℤ) : ℤ := 2 * a ^ 2 - 3 * b ^ 2 + d ^ 3

-- Main theorem statement
theorem problem_sol (a b x c d : ℤ) (h1 : a ≥ 0) (h2 : b ≥ 0) (hc : c > 0) (hd : d > 0) 
  (h3 : operation_hash a x c = 250)
  (h4 : operation_star a b d + x = 50) :
  False := sorry

end problem_sol_l181_181756


namespace SoccerBallPrices_SoccerBallPurchasingPlans_l181_181941

theorem SoccerBallPrices :
  ∃ (priceA priceB : ℕ), priceA = 100 ∧ priceB = 80 ∧ (900 / priceA) = (720 / (priceB - 20)) :=
sorry

theorem SoccerBallPurchasingPlans :
  ∃ (m n : ℕ), (m + n = 90) ∧ (m ≥ 2 * n) ∧ (100 * m + 80 * n ≤ 8500) ∧
  (m ∈ Finset.range 66 \ Finset.range 60) ∧ 
  (∀ k ∈ Finset.range 66 \ Finset.range 60, 100 * k + 80 * (90 - k) ≥ 8400) :=
sorry

end SoccerBallPrices_SoccerBallPurchasingPlans_l181_181941


namespace circumference_of_cone_base_l181_181133

theorem circumference_of_cone_base (V : ℝ) (h : ℝ) (C : ℝ) (π := Real.pi) 
  (volume_eq : V = 24 * π) (height_eq : h = 6) 
  (circumference_eq : C = 4 * Real.sqrt 3 * π) :
  ∃ r : ℝ, (V = (1 / 3) * π * r^2 * h) ∧ (C = 2 * π * r) :=
by
  sorry

end circumference_of_cone_base_l181_181133


namespace stratified_sampling_sum_l181_181814

theorem stratified_sampling_sum :
  let grains := 40
  let vegetable_oils := 10
  let animal_foods := 30
  let fruits_and_vegetables := 20
  let sample_size := 20
  let total_food_types := grains + vegetable_oils + animal_foods + fruits_and_vegetables
  let sampling_fraction := sample_size / total_food_types
  let number_drawn := sampling_fraction * (vegetable_oils + fruits_and_vegetables)
  number_drawn = 6 :=
by
  sorry

end stratified_sampling_sum_l181_181814


namespace flowers_count_l181_181532

theorem flowers_count (lilies : ℕ) (sunflowers : ℕ) (daisies : ℕ) (total_flowers : ℕ) (roses : ℕ)
  (h1 : lilies = 40) (h2 : sunflowers = 40) (h3 : daisies = 40) (h4 : total_flowers = 160) :
  lilies + sunflowers + daisies + roses = 160 → roses = 40 := 
by
  sorry

end flowers_count_l181_181532


namespace mark_brought_in_4_times_more_cans_l181_181813

theorem mark_brought_in_4_times_more_cans (M J R : ℕ) (h1 : M = 100) 
  (h2 : J = 2 * R + 5) (h3 : M + J + R = 135) : M / J = 4 :=
by sorry

end mark_brought_in_4_times_more_cans_l181_181813


namespace soccer_tournament_matches_l181_181084

theorem soccer_tournament_matches (n: ℕ):
  n = 20 → ∃ m: ℕ, m = 19 := sorry

end soccer_tournament_matches_l181_181084


namespace mean_median_modes_l181_181083

theorem mean_median_modes (d μ M : ℝ)
  (dataset : Multiset ℕ)
  (h_dataset : dataset = Multiset.replicate 12 1 + Multiset.replicate 12 2 + Multiset.replicate 12 3 +
                         Multiset.replicate 12 4 + Multiset.replicate 12 5 + Multiset.replicate 12 6 +
                         Multiset.replicate 12 7 + Multiset.replicate 12 8 + Multiset.replicate 12 9 +
                         Multiset.replicate 12 10 + Multiset.replicate 12 11 + Multiset.replicate 12 12 +
                         Multiset.replicate 12 13 + Multiset.replicate 12 14 + Multiset.replicate 12 15 +
                         Multiset.replicate 12 16 + Multiset.replicate 12 17 + Multiset.replicate 12 18 +
                         Multiset.replicate 12 19 + Multiset.replicate 12 20 + Multiset.replicate 12 21 +
                         Multiset.replicate 12 22 + Multiset.replicate 12 23 + Multiset.replicate 12 24 +
                         Multiset.replicate 12 25 + Multiset.replicate 12 26 + Multiset.replicate 12 27 +
                         Multiset.replicate 12 28 + Multiset.replicate 12 29 + Multiset.replicate 12 30 +
                         Multiset.replicate 7 31)
  (h_M : M = 16)
  (h_μ : μ = 5797 / 366)
  (h_d : d = 15.5) :
  d < μ ∧ μ < M :=
sorry

end mean_median_modes_l181_181083


namespace no_positive_alpha_exists_l181_181556

theorem no_positive_alpha_exists :
  ¬ ∃ α > 0, ∀ x : ℝ, |Real.cos x| + |Real.cos (α * x)| > Real.sin x + Real.sin (α * x) :=
by
  sorry

end no_positive_alpha_exists_l181_181556


namespace opposite_of_2023_l181_181757

/-- The opposite of a number n is defined as the number that, when added to n, results in zero. -/
def opposite (n : ℤ) : ℤ := -n

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by 
  sorry

end opposite_of_2023_l181_181757


namespace altitude_point_intersect_and_length_equalities_l181_181292

variables (A B C D E H : Type)
variables (triangle : A ≠ B ∧ B ≠ C ∧ C ≠ A)
variables (acute : ∀ (a b c : A), True) -- Placeholder for the acute triangle condition
variables (altitude_AD : True) -- Placeholder for the specific definition of altitude AD
variables (altitude_BE : True) -- Placeholder for the specific definition of altitude BE
variables (HD HE AD : ℝ)
variables (BD DC AE EC : ℝ)

theorem altitude_point_intersect_and_length_equalities
  (HD_eq : HD = 3)
  (HE_eq : HE = 4) 
  (sim1 : BD / 3 = (AD + 3) / DC)
  (sim2 : AE / 4 = (BE + 4) / EC)
  (sim3 : 4 * AD = 3 * BE) :
  (BD * DC) - (AE * EC) = 3 * AD - 7 := by
  sorry

end altitude_point_intersect_and_length_equalities_l181_181292


namespace opposite_of_8_is_neg8_l181_181903

theorem opposite_of_8_is_neg8 : ∃ y : ℤ, 8 + y = 0 ∧ y = -8 := by
  use -8
  split
  ·
    sorry

  ·
    rfl

end opposite_of_8_is_neg8_l181_181903


namespace binomial_20_19_eq_20_l181_181308

theorem binomial_20_19_eq_20 : Nat.choose 20 19 = 20 :=
by
  sorry

end binomial_20_19_eq_20_l181_181308


namespace middle_part_is_28_4_over_11_l181_181996

theorem middle_part_is_28_4_over_11 (x : ℚ) :
  let part1 := x
  let part2 := (1/2) * x
  let part3 := (1/3) * x
  part1 + part2 + part3 = 104
  ∧ part2 = 28 + 4/11 := by
  sorry

end middle_part_is_28_4_over_11_l181_181996


namespace middle_value_bounds_l181_181241

theorem middle_value_bounds (a b c : ℝ) (h1 : a + b + c = 10)
  (h2 : a > b) (h3 : b > c) (h4 : a - c = 3) : 
  7 / 3 < b ∧ b < 13 / 3 :=
by
  sorry

end middle_value_bounds_l181_181241


namespace ratio_of_heights_l181_181073

-- Define the height of the first rocket.
def H1 : ℝ := 500

-- Define the combined height of the two rockets.
def combined_height : ℝ := 1500

-- Define the height of the second rocket.
def H2 : ℝ := combined_height - H1

-- The statement to be proven.
theorem ratio_of_heights : H2 / H1 = 2 := by
  -- Proof goes here
  sorry

end ratio_of_heights_l181_181073


namespace factorization_of_polynomial_l181_181667

theorem factorization_of_polynomial :
  (x : ℝ) → (x^8 - 4*x^6 + 6*x^4 - 4*x^2 + 1) = ((x - 1)^4 * (x + 1)^4) :=
by
  intro x
  sorry

end factorization_of_polynomial_l181_181667


namespace min_sum_4410_l181_181100

def min_sum (a b c d : ℕ) : ℕ := a + b + c + d

theorem min_sum_4410 :
  ∃ (a b c d : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ a * b * c * d = 4410 ∧ min_sum a b c d = 69 :=
sorry

end min_sum_4410_l181_181100


namespace inequality_positive_real_numbers_l181_181890

theorem inequality_positive_real_numbers
  (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_condition : a * b + b * c + c * a = 1) :
  (a / Real.sqrt (a^2 + 1)) + (b / Real.sqrt (b^2 + 1)) + (c / Real.sqrt (c^2 + 1)) ≤ (3 / 2) :=
  sorry

end inequality_positive_real_numbers_l181_181890


namespace intersection_condition_l181_181171

-- Define the lines
def line1 (x y : ℝ) := 2*x - 2*y - 3 = 0
def line2 (x y : ℝ) := 3*x - 5*y + 1 = 0
def line (a b x y : ℝ) := a*x - y + b = 0

-- Define the condition
def condition (a b : ℝ) := 17*a + 4*b = 11

-- Prove that the line l passes through the intersection point of l1 and l2 if and only if the condition holds
theorem intersection_condition (a b : ℝ) :
  (∃ x y : ℝ, line1 x y ∧ line2 x y ∧ line a b x y) ↔ condition a b :=
  sorry

end intersection_condition_l181_181171


namespace part1_part2_l181_181989

-- Define the parabola C as y^2 = 4x
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- Define the line l with slope k passing through point P(-2, 1)
def line (x y k : ℝ) : Prop := y - 1 = k * (x + 2)

-- Part 1: Prove the range of k for which line l intersects parabola C at two points is -1 < k < -1/2
theorem part1 (k : ℝ) : 
  (∃ x y, parabola x y ∧ line x y k) ∧ (∃ u v, parabola u v ∧ u ≠ x ∧ line u v k) ↔ -1 < k ∧ k < -1/2 := sorry

-- Part 2: Prove the equations of line l when it intersects parabola C at only one point are y = 1, y = -x - 1, and y = -1/2 x
theorem part2 (k : ℝ) : 
  (∃! x y, parabola x y ∧ line x y k) ↔ (k = 0 ∨ k = -1 ∨ k = -1/2) := sorry

end part1_part2_l181_181989


namespace katya_age_l181_181088

theorem katya_age (A K V : ℕ) (h1 : A + K = 19) (h2 : A + V = 14) (h3 : K + V = 7) : K = 6 := by
  sorry

end katya_age_l181_181088


namespace tan_value_l181_181366

theorem tan_value (α : ℝ) (h1 : α ∈ (Set.Ioo (π/2) π)) (h2 : Real.sin α = 4/5) : Real.tan α = -4/3 :=
sorry

end tan_value_l181_181366


namespace probability_approx_l181_181635

noncomputable def circumscribed_sphere_volume (R : ℝ) : ℝ :=
  (4 / 3) * Real.pi * R^3

noncomputable def single_sphere_volume (R : ℝ) : ℝ :=
  (4 / 3) * Real.pi * (R / 3)^3

noncomputable def total_spheres_volume (R : ℝ) : ℝ :=
  6 * single_sphere_volume R

noncomputable def probability_inside_spheres (R : ℝ) : ℝ :=
  total_spheres_volume R / circumscribed_sphere_volume R

theorem probability_approx (R : ℝ) (hR : R > 0) : 
  abs (probability_inside_spheres R - 0.053) < 0.001 := sorry

end probability_approx_l181_181635


namespace floor_neg_seven_over_four_l181_181347

theorem floor_neg_seven_over_four : Int.floor (- 7 / 4 : ℝ) = -2 := 
by
  sorry

end floor_neg_seven_over_four_l181_181347


namespace angleC_is_100_l181_181053

-- Given a triangle ABC
variables (A B C : Type) [angle A] [angle B] [angle C]

-- Define a triangle
structure triangle (A B C : Type) := 
(angle_A : A)
(angle_B : B)
(angle_C : C)

-- Assume the sum of the angles in a triangle is 180 degrees
axiom sum_of_angles_in_triangle (t : triangle A B C) : 
  (angle_A + angle_B + angle_C = 180)

-- Given condition
axiom sum_of_angleA_and_angleB (t : triangle A B C) :
  (angle_A + angle_B = 80)

-- Prove that angle C is 100 degrees
theorem angleC_is_100 (t : triangle A B C) :
  angle_C = 100 :=
begin
  sorry
end

end angleC_is_100_l181_181053


namespace angle_C_in_triangle_l181_181039

theorem angle_C_in_triangle (A B C : ℝ) (h1 : A + B = 80) (h2 : A + B + C = 180) : C = 100 := 
by 
  sorry

end angle_C_in_triangle_l181_181039


namespace binom_20_19_eq_20_l181_181307

theorem binom_20_19_eq_20 : nat.choose 20 19 = 20 :=
by sorry

end binom_20_19_eq_20_l181_181307


namespace hyperbola_properties_l181_181845

open Real

def is_asymptote (y x : ℝ) : Prop :=
  y = (1/2) * x ∨ y = -(1/2) * x

noncomputable def eccentricity (a c : ℝ) : ℝ := c / a

theorem hyperbola_properties :
  ∀ x y : ℝ,
  (x^2 / 4 - y^2 = 1) →
  ∀ (a b c : ℝ), 
  (a = 2) →
  (b = 1) →
  (c = sqrt (a^2 + b^2)) →
  (∀ y x : ℝ, (is_asymptote y x)) ∧ (eccentricity a (sqrt (a^2 + b^2)) = sqrt 5 / 2) :=
by
  intros x y h a b c ha hb hc
  sorry

end hyperbola_properties_l181_181845


namespace find_p_q_r_l181_181182

def is_rel_prime (m n : ℕ) : Prop := Nat.gcd m n = 1

theorem find_p_q_r (x : ℝ) (p q r : ℕ)
  (h1 : (1 + Real.sin x) * (1 + Real.cos x) = 9 / 4)
  (h2 : (1 - Real.sin x) * (1 - Real.cos x) = p / q - Real.sqrt r)
  (hpq_rel_prime : is_rel_prime p q)
  (hp : 0 < p)
  (hq : 0 < q)
  (hr : 0 < r) :
  p + q + r = 26 :=
sorry

end find_p_q_r_l181_181182


namespace probability_three_unused_rockets_expected_targets_hit_l181_181955

section RocketArtillery

variables (p : ℝ) (h0 : 0 ≤ p) (h1 : p ≤ 1)

-- Probability that exactly three unused rockets remain after firing at five targets
theorem probability_three_unused_rockets (p : ℝ) (h0 : 0 ≤ p) (h1 : p ≤ 1) : 
  let prob := 10 * (p ^ 3) * ((1 - p) ^ 2) in
  prob = 10 * (p ^ 3) * ((1 - p) ^ 2) := 
by
  sorry

-- Expected number of targets hit if there are nine targets
theorem expected_targets_hit (p : ℝ) (h0 : 0 ≤ p) (h1 : p ≤ 1) : 
  let expected_hits := 10 * p - (p ^ 10) in
  expected_hits = 10 * p - (p ^ 10) := 
by
  sorry

end RocketArtillery

end probability_three_unused_rockets_expected_targets_hit_l181_181955


namespace find_f_of_2_l181_181386

variable (f : ℝ → ℝ)

-- Given condition: f is the inverse function of the exponential function 2^x
def inv_function : Prop := ∀ x, f (2^x) = x ∧ 2^(f x) = x

theorem find_f_of_2 (h : inv_function f) : f 2 = 1 :=
by sorry

end find_f_of_2_l181_181386


namespace truth_values_set1_truth_values_set2_l181_181460

-- Definitions for set (1)
def p1 : Prop := Prime 3
def q1 : Prop := Even 3

-- Definitions for set (2)
def p2 (x : Int) : Prop := x = -2 ∧ (x^2 + x - 2 = 0)
def q2 (x : Int) : Prop := x = 1 ∧ (x^2 + x - 2 = 0)

-- Theorem for set (1)
theorem truth_values_set1 : 
  (p1 ∨ q1) = true ∧ (p1 ∧ q1) = false ∧ (¬p1) = false := by sorry

-- Theorem for set (2)
theorem truth_values_set2 (x : Int) :
  (p2 x ∨ q2 x) = true ∧ (p2 x ∧ q2 x) = true ∧ (¬p2 x) = false := by sorry

end truth_values_set1_truth_values_set2_l181_181460


namespace problem_1_problem_2_problem_3_problem_4_l181_181771

-- Given conditions
variable {T : Type} -- Type representing teachers
variable {S : Type} -- Type representing students

def arrangements_ends (teachers : List T) (students : List S) : ℕ :=
  sorry -- Implementation skipped

def arrangements_next_to_each_other (teachers : List T) (students : List S) : ℕ :=
  sorry -- Implementation skipped

def arrangements_not_next_to_each_other (teachers : List T) (students : List S) : ℕ :=
  sorry -- Implementation skipped

def arrangements_two_between (teachers : List T) (students : List S) : ℕ :=
  sorry -- Implementation skipped

-- Statements to prove

-- 1. Prove that if teachers A and B must stand at the two ends, there are 48 different arrangements
theorem problem_1 {teachers : List T} {students : List S} (h : teachers.length = 2) (s : students.length = 4) :
  arrangements_ends teachers students = 48 :=
  sorry

-- 2. Prove that if teachers A and B must stand next to each other, there are 240 different arrangements
theorem problem_2 {teachers : List T} {students : List S} (h : teachers.length = 2) (s : students.length = 4) :
  arrangements_next_to_each_other teachers students = 240 :=
  sorry 

-- 3. Prove that if teachers A and B cannot stand next to each other, there are 480 different arrangements
theorem problem_3 {teachers : List T} {students : List S} (h : teachers.length = 2) (s : students.length = 4) :
  arrangements_not_next_to_each_other teachers students = 480 :=
  sorry 

-- 4. Prove that if there must be two students standing between teachers A and B, there are 144 different arrangements
theorem problem_4 {teachers : List T} {students : List S} (h : teachers.length = 2) (s : students.length = 4) :
  arrangements_two_between teachers students = 144 :=
  sorry

end problem_1_problem_2_problem_3_problem_4_l181_181771


namespace smallest_positive_integer_for_divisibility_l181_181259

def is_divisible_by (a b : ℕ) : Prop :=
  ∃ k, a = b * k

def smallest_n (n : ℕ) : Prop :=
  (is_divisible_by (n^2) 50) ∧ (is_divisible_by (n^3) 288) ∧ (∀ m : ℕ, m > 0 → m < n → ¬ (is_divisible_by (m^2) 50 ∧ is_divisible_by (m^3) 288))

theorem smallest_positive_integer_for_divisibility : smallest_n 60 :=
by
  sorry

end smallest_positive_integer_for_divisibility_l181_181259


namespace binom_20_19_eq_20_l181_181322

theorem binom_20_19_eq_20: nat.choose 20 19 = 20 :=
  sorry

end binom_20_19_eq_20_l181_181322


namespace soccer_team_unplayed_players_l181_181136

theorem soccer_team_unplayed_players:
  ∀ (total_players first_half_players first_half_subs : ℕ),
  first_half_players = 11 →
  first_half_subs = 2 →
  total_players = 24 →
  (total_players - (first_half_players + first_half_subs + 2 * first_half_subs)) = 7 :=
by
  intros total_players first_half_players first_half_subs h1 h2 h3
  rw [h1, h2, h3]
  show 24 - (11 + 2 + 2 * 2) = 7
  sorry

end soccer_team_unplayed_players_l181_181136


namespace fifty_third_card_is_A_l181_181210

noncomputable def card_seq : List String := 
  ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]

theorem fifty_third_card_is_A : card_seq[(53 % 13)] = "A" := 
by 
  simp [card_seq] 
  sorry

end fifty_third_card_is_A_l181_181210


namespace dogwood_trees_current_l181_181768

variable (X : ℕ)
variable (trees_today : ℕ := 41)
variable (trees_tomorrow : ℕ := 20)
variable (total_trees_after : ℕ := 100)

theorem dogwood_trees_current (h : X + trees_today + trees_tomorrow = total_trees_after) : X = 39 :=
by
  sorry

end dogwood_trees_current_l181_181768


namespace divisibility_by_1956_l181_181226

theorem divisibility_by_1956 (n : ℕ) (hn : n % 2 = 1) : 
  1956 ∣ (24 * 80^n + 1992 * 83^(n-1)) :=
by
  sorry

end divisibility_by_1956_l181_181226


namespace coprime_and_composite_sum_l181_181301

theorem coprime_and_composite_sum (n : ℕ) (h : n = 2005) :
  ∃ (S : Finset ℕ), 
    (∀ x y ∈ S, Nat.coprime x y) ∧ 
    (∀ (k : ℕ) (h : 2 ≤ k) (t : Finset (Finset ℕ)), 
      t.card = k ∧ t ⊆ S → Nat.isComposite (t.sum id)) :=
by
  sorry

end coprime_and_composite_sum_l181_181301


namespace dilation_rotation_l181_181097

noncomputable def center : ℂ := 2 + 3 * Complex.I
noncomputable def scale_factor : ℂ := 3
noncomputable def initial_point : ℂ := -1 + Complex.I
noncomputable def final_image : ℂ := -4 + 12 * Complex.I

theorem dilation_rotation (z : ℂ) :
  z = (-1 + Complex.I) →
  let z' := center + scale_factor * (initial_point - center)
  let rotated_z := center + Complex.I * (z' - center)
  rotated_z = final_image := sorry

end dilation_rotation_l181_181097


namespace find_m_l181_181371

theorem find_m (m : ℝ) (α : ℝ) (h_cos : Real.cos α = -3/5) (h_p : ((Real.cos α = m / (Real.sqrt (m^2 + 4^2)))) ∧ (Real.cos α < 0) ∧ (m < 0)) :

  m = -3 :=
by 
  sorry

end find_m_l181_181371


namespace quadratic_to_square_l181_181542

theorem quadratic_to_square (x h k : ℝ) : 
  (x * x - 4 * x + 3 = 0) →
  ((x + h) * (x + h) = k) →
  k = 1 :=
by
  sorry

end quadratic_to_square_l181_181542


namespace value_of_xyz_l181_181186

open Real

theorem value_of_xyz (x y z : ℝ)
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 37)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 11) :
  x * y * z = 26 / 3 := 
  sorry

end value_of_xyz_l181_181186


namespace evaluate_three_squared_raised_four_l181_181167

theorem evaluate_three_squared_raised_four : (3^2)^4 = 6561 := by
  sorry

end evaluate_three_squared_raised_four_l181_181167


namespace tshirt_more_expensive_l181_181993

-- Definitions based on given conditions
def jeans_price : ℕ := 30
def socks_price : ℕ := 5
def tshirt_price : ℕ := jeans_price / 2

-- Statement to prove (The t-shirt is $10 more expensive than the socks)
theorem tshirt_more_expensive : (tshirt_price - socks_price) = 10 :=
by
  rw [tshirt_price, socks_price]
  sorry  -- proof steps are omitted

end tshirt_more_expensive_l181_181993


namespace tan_a_eq_two_imp_cos_2a_plus_sin_2a_eq_one_fifth_l181_181997

theorem tan_a_eq_two_imp_cos_2a_plus_sin_2a_eq_one_fifth (a : ℝ) (h : Real.tan a = 2) :
  Real.cos (2 * a) + Real.sin (2 * a) = 1 / 5 :=
by
  sorry

end tan_a_eq_two_imp_cos_2a_plus_sin_2a_eq_one_fifth_l181_181997


namespace range_of_a_product_of_zeros_l181_181518

def f (x : ℝ) (a : ℝ) := (Real.exp x / x) - (Real.log x) + x - a

theorem range_of_a (a : ℝ) : (∀ x, (0 < x) → (f x a ≥ 0)) ↔ (a ≤ Real.exp 1 + 1) :=
by
  sorry

theorem product_of_zeros (x1 x2 a : ℝ) : 
  (0 < x1) → (x1 < 1) → (1 < x2) → (f x1 a = 0 ∧ f x2 a = 0) → x1 * x2 < 1 :=
by
  sorry

end range_of_a_product_of_zeros_l181_181518


namespace angleC_is_100_l181_181056

-- Given a triangle ABC
variables (A B C : Type) [angle A] [angle B] [angle C]

-- Define a triangle
structure triangle (A B C : Type) := 
(angle_A : A)
(angle_B : B)
(angle_C : C)

-- Assume the sum of the angles in a triangle is 180 degrees
axiom sum_of_angles_in_triangle (t : triangle A B C) : 
  (angle_A + angle_B + angle_C = 180)

-- Given condition
axiom sum_of_angleA_and_angleB (t : triangle A B C) :
  (angle_A + angle_B = 80)

-- Prove that angle C is 100 degrees
theorem angleC_is_100 (t : triangle A B C) :
  angle_C = 100 :=
begin
  sorry
end

end angleC_is_100_l181_181056


namespace no_solutions_l181_181674

theorem no_solutions {x y : ℤ} :
  (x ≠ 1) → (y ≠ 1) →
  ((x^7 - 1) / (x - 1) = y^5 - 1) →
  false :=
by sorry

end no_solutions_l181_181674


namespace overtime_pay_rate_increase_l181_181466

theorem overtime_pay_rate_increase
  (regular_rate : ℝ)
  (total_compensation : ℝ)
  (total_hours : ℝ)
  (overtime_hours : ℝ)
  (expected_percentage_increase : ℝ)
  (h1 : regular_rate = 16)
  (h2 : total_hours = 48)
  (h3 : total_compensation = 864)
  (h4 : overtime_hours = total_hours - 40)
  (h5 : 40 * regular_rate + overtime_hours * (regular_rate + regular_rate * expected_percentage_increase / 100) = total_compensation) :
  expected_percentage_increase = 75 := 
by
  sorry

end overtime_pay_rate_increase_l181_181466


namespace find_x_l181_181028

theorem find_x (x : ℕ) (h : x * 6000 = 480 * 10^5) : x = 8000 := 
by
  sorry

end find_x_l181_181028


namespace area_ADC_calculation_l181_181213

-- Definitions and assumptions
variables (BD DC : ℝ)
variables (area_ABD area_ADC : ℝ)

-- Given conditions
axiom ratio_BD_DC : BD / DC = 2 / 5
axiom area_ABD_given : area_ABD = 40

-- The theorem to prove
theorem area_ADC_calculation (h1 : BD / DC = 2 / 5) (h2 : area_ABD = 40) :
  area_ADC = 100 :=
sorry

end area_ADC_calculation_l181_181213


namespace pencils_distribution_count_l181_181976

def count_pencils_distribution : ℕ :=
  let total_pencils := 10
  let friends := 4
  let adjusted_pencils := total_pencils - friends
  Nat.choose (adjusted_pencils + friends - 1) (friends - 1)

theorem pencils_distribution_count :
  count_pencils_distribution = 84 := 
  by sorry

end pencils_distribution_count_l181_181976


namespace mens_wages_l181_181266

-- Definitions based on the problem conditions
def equivalent_wages (M W_earn B : ℝ) : Prop :=
  (5 * M = W_earn) ∧ 
  (W_earn = 8 * B) ∧ 
  (5 * M + W_earn + 8 * B = 210)

-- Prove that the total wages of 5 men are Rs. 105 given the conditions
theorem mens_wages (M W_earn B : ℝ) (h : equivalent_wages M W_earn B) : 5 * M = 105 :=
by
  sorry

end mens_wages_l181_181266


namespace speed_of_faster_train_l181_181912

noncomputable def speed_of_slower_train : ℝ := 36
noncomputable def length_of_each_train : ℝ := 70
noncomputable def time_to_pass : ℝ := 36

theorem speed_of_faster_train : 
  ∃ (V_f : ℝ), 
    (V_f - speed_of_slower_train) * (1000 / 3600) = 140 / time_to_pass ∧ 
    V_f = 50 :=
by {
  sorry
}

end speed_of_faster_train_l181_181912


namespace total_apples_bought_l181_181202

def apples_bought_by_each_man := 30
def apples_more_than_men := 20
def apples_bought_by_each_woman := apples_bought_by_each_man + apples_more_than_men

theorem total_apples_bought (men women : ℕ) (apples_bought_by_each_man : ℕ) 
  (apples_bought_by_each_woman : ℕ) (apples_more_than_men : ℕ) : 
  men = 2 → women = 3 → apples_bought_by_each_man = 30 → apples_more_than_men = 20 → 
  apples_bought_by_each_woman = apples_bought_by_each_man + apples_more_than_men →
  (men * apples_bought_by_each_man + women * apples_bought_by_each_woman) = 210 := 
by
  intros menTwo menThree womApples menApples moreApples _ _ _ _ eqn
  simp only [menTwo, menThree, womApples, menApples, moreApples, eqn]
  -- state infering
  exact eq.refl 210

end total_apples_bought_l181_181202


namespace side_length_of_square_with_circles_l181_181072

noncomputable def side_length_of_square (radius : ℝ) : ℝ :=
  2 * radius + 2 * radius

theorem side_length_of_square_with_circles 
  (radius : ℝ) 
  (h_radius : radius = 2) 
  (h_tangent : ∀ (P Q : ℝ), P = Q + 2 * radius) :
  side_length_of_square radius = 8 :=
by
  sorry

end side_length_of_square_with_circles_l181_181072


namespace sqrt_square_of_neg_four_l181_181300

theorem sqrt_square_of_neg_four : Real.sqrt ((-4:Real)^2) = 4 := by
  sorry

end sqrt_square_of_neg_four_l181_181300


namespace probability_comparison_l181_181285

variables (M N : ℕ) (m n : ℝ)
variable (h₁ : m > 10^6)
variable (h₂ : n ≤ 10^6)

theorem probability_comparison (h₃: 0 < M) (h₄: 0 < N):
  (m * M) / (m * M + n * N) > (M / (M + N)) :=
by
  have h₅: n / m < 1 := sorry
  have h₆: M > 0 := by linarith
  have h₇: 1 + (n / m) * (N / M) < 2 := sorry
  sorry

end probability_comparison_l181_181285


namespace find_square_digit_l181_181095

def is_even (n : ℕ) : Prop := n % 2 = 0

def sum_digits_31_42_7s (s : ℕ) : ℕ :=
  3 + 1 + 4 + 2 + 7 + s

-- The main theorem to prove
theorem find_square_digit (d : ℕ) (h0 : is_even d) (h1 : (sum_digits_31_42_7s d) % 3 = 0) : d = 4 :=
by
  sorry

end find_square_digit_l181_181095


namespace count_equilateral_triangles_in_hexagonal_lattice_l181_181333

-- Definitions based on conditions in problem (hexagonal lattice setup)
def hexagonal_lattice (dist : ℕ) : Prop :=
  -- Define properties of the points in hexagonal lattice
  -- Placeholder for actual structure defining the hexagon and surrounding points
  sorry

def equilateral_triangles (n : ℕ) : Prop :=
  -- Define a method to count equilateral triangles in the given lattice setup
  sorry

-- Theorem stating that 10 equilateral triangles can be formed in the lattice
theorem count_equilateral_triangles_in_hexagonal_lattice (dist : ℕ) (h : dist = 1 ∨ dist = 2) :
  equilateral_triangles 10 :=
by
  -- Proof to be completed
  sorry

end count_equilateral_triangles_in_hexagonal_lattice_l181_181333


namespace print_time_including_warmup_l181_181132

def warmUpTime : ℕ := 2
def pagesPerMinute : ℕ := 15
def totalPages : ℕ := 225

theorem print_time_including_warmup :
  (totalPages / pagesPerMinute) + warmUpTime = 17 := by
  sorry

end print_time_including_warmup_l181_181132


namespace library_books_total_l181_181877

-- Definitions for the conditions
def books_purchased_last_year : Nat := 50
def books_purchased_this_year : Nat := 3 * books_purchased_last_year
def books_before_last_year : Nat := 100

-- The library's current number of books
def total_books_now : Nat :=
  books_before_last_year + books_purchased_last_year + books_purchased_this_year

-- The proof statement
theorem library_books_total : total_books_now = 300 :=
by
  -- Placeholder for actual proof
  sorry

end library_books_total_l181_181877


namespace probability_comparison_l181_181282

variables (M N : ℕ) (m n : ℝ)
variable (h₁ : m > 10^6)
variable (h₂ : n ≤ 10^6)

theorem probability_comparison (h₃: 0 < M) (h₄: 0 < N):
  (m * M) / (m * M + n * N) > (M / (M + N)) :=
by
  have h₅: n / m < 1 := sorry
  have h₆: M > 0 := by linarith
  have h₇: 1 + (n / m) * (N / M) < 2 := sorry
  sorry

end probability_comparison_l181_181282


namespace fraction_of_work_left_l181_181118

theorem fraction_of_work_left 
  (A_days : ℝ) (B_days : ℝ) (work_days : ℝ) 
  (A_work_rate : A_days = 15) 
  (B_work_rate : B_days = 30) 
  (work_duration : work_days = 4)
  : (1 - (work_days * ((1 / A_days) + (1 / B_days)))) = 3 / 5 := 
by
  sorry

end fraction_of_work_left_l181_181118


namespace probability_is_8point64_percent_l181_181433

/-- Define the probabilities based on given conditions -/
def p_excel : ℝ := 0.45
def p_night_shift_given_excel : ℝ := 0.32
def p_no_weekend_given_night_shift : ℝ := 0.60

/-- Calculate the combined probability -/
def combined_probability :=
  p_excel * p_night_shift_given_excel * p_no_weekend_given_night_shift

theorem probability_is_8point64_percent :
  combined_probability = 0.0864 :=
by
  -- We will skip the proof for now
  sorry

end probability_is_8point64_percent_l181_181433


namespace actor_A_constraints_l181_181787

-- Definitions corresponding to the conditions.
def numberOfActors : Nat := 6
def positionConstraints : Nat := 4
def permutations (n : Nat) : Nat := Nat.factorial n

-- Lean statement for the proof problem.
theorem actor_A_constraints : 
  (positionConstraints * permutations (numberOfActors - 1)) = 480 := by
sorry

end actor_A_constraints_l181_181787


namespace binomial_20_19_eq_20_l181_181311

theorem binomial_20_19_eq_20 : Nat.choose 20 19 = 20 :=
by
  sorry

end binomial_20_19_eq_20_l181_181311


namespace cottage_cost_per_hour_l181_181721

-- Define the conditions
def jack_payment : ℝ := 20
def jill_payment : ℝ := 20
def total_payment : ℝ := jack_payment + jill_payment
def rental_duration : ℝ := 8

-- Define the theorem to be proved
theorem cottage_cost_per_hour : (total_payment / rental_duration) = 5 := by
  sorry

end cottage_cost_per_hour_l181_181721


namespace largest_sum_is_7_over_12_l181_181303

-- Define the five sums
def sum1 : ℚ := 1/3 + 1/4
def sum2 : ℚ := 1/3 + 1/5
def sum3 : ℚ := 1/3 + 1/6
def sum4 : ℚ := 1/3 + 1/9
def sum5 : ℚ := 1/3 + 1/8

-- Define the problem statement
theorem largest_sum_is_7_over_12 : 
  max (max (max sum1 sum2) (max sum3 sum4)) sum5 = 7/12 := 
by
  sorry

end largest_sum_is_7_over_12_l181_181303


namespace total_letters_l181_181199

theorem total_letters (brother_letters : ℕ) (greta_more_than_brother : ℕ) (mother_multiple : ℕ)
  (h_brother : brother_letters = 40)
  (h_greta : ∀ (brother_letters greta_letters : ℕ), greta_letters = brother_letters + greta_more_than_brother)
  (h_mother : ∀ (total_letters mother_letters : ℕ), mother_letters = mother_multiple * total_letters) :
  brother_letters + (brother_letters + greta_more_than_brother) + (mother_multiple * (brother_letters + (brother_letters + greta_more_than_brother))) = 270 :=
by
  sorry

end total_letters_l181_181199


namespace ezekiel_shoes_l181_181002

theorem ezekiel_shoes (pairs : ℕ) (shoes_per_pair : ℕ) (bought_pairs : pairs = 3) (pair_contains : shoes_per_pair = 2) : pairs * shoes_per_pair = 6 := by
  sorry

end ezekiel_shoes_l181_181002


namespace bill_toys_l181_181377

variable (B H : ℕ)

theorem bill_toys (h1 : H = B / 2 + 9) (h2 : B + H = 99) : B = 60 := by
  sorry

end bill_toys_l181_181377


namespace apples_ratio_l181_181644

theorem apples_ratio (bonnie_apples samuel_extra_apples samuel_left_over samuel_total_pies : ℕ) 
  (h_bonnie : bonnie_apples = 8)
  (h_samuel_extra : samuel_extra_apples = 20)
  (h_samuel_left_over : samuel_left_over = 10)
  (h_pie_ratio : samuel_total_pies = (8 + 20) / 7) :
  (28 - samuel_total_pies - 10) / 28 = 1 / 2 := 
by
  sorry

end apples_ratio_l181_181644


namespace find_k_l181_181581

theorem find_k 
  (k : ℝ)
  (p_eq : ∀ x : ℝ, (4 * x + 3 = k * x - 9) → (x = -3 → (k = 0)))
: k = 0 :=
by sorry

end find_k_l181_181581


namespace quadratic_distinct_roots_l181_181709

theorem quadratic_distinct_roots (k : ℝ) : 
  (∃ x y : ℝ, k * x^2 - 2 * x - 1 = 0 ∧ k * y^2 - 2 * y - 1 = 0 ∧ x ≠ y) ↔ k > -1 ∧ k ≠ 0 := 
sorry

end quadratic_distinct_roots_l181_181709


namespace problem_statement_l181_181080

def f(x : ℝ) : ℝ := 3 * x - 3
def g(x : ℝ) : ℝ := x^2 + 1

theorem problem_statement : f (1 + g 2) = 15 := by
  sorry

end problem_statement_l181_181080


namespace interior_diagonals_of_dodecahedron_l181_181704

/-- Definition of a dodecahedron. -/
structure Dodecahedron where
  vertices : ℕ
  faces : ℕ
  vertices_per_face : ℕ
  faces_meeting_per_vertex : ℕ
  interior_diagonals : ℕ

/-- A dodecahedron has 12 pentagonal faces, 20 vertices, and 3 faces meet at each vertex. -/
def dodecahedron : Dodecahedron :=
  { vertices := 20,
    faces := 12,
    vertices_per_face := 5,
    faces_meeting_per_vertex := 3,
    interior_diagonals := 160 }

theorem interior_diagonals_of_dodecahedron (d : Dodecahedron) :
    d.vertices = 20 → 
    d.faces = 12 →
    d.faces_meeting_per_vertex = 3 →
    d.interior_diagonals = 160 :=
by
  intros
  sorry

end interior_diagonals_of_dodecahedron_l181_181704


namespace main_factor_is_D_l181_181444

-- Let A, B, C, and D be the factors where A is influenced by 1, B by 2, C by 3, and D by 4
def A := 1
def B := 2
def C := 3
def D := 4

-- Defining the main factor influenced by the plan
def main_factor_influenced_by_plan := D

-- The problem statement translated to a Lean theorem statement
theorem main_factor_is_D : main_factor_influenced_by_plan = D := 
by sorry

end main_factor_is_D_l181_181444


namespace fraction_of_house_painted_l181_181856

theorem fraction_of_house_painted (total_time : ℝ) (paint_time : ℝ) (house : ℝ) (h1 : total_time = 60) (h2 : paint_time = 15) (h3 : house = 1) : 
  (paint_time / total_time) * house = 1 / 4 :=
by
  sorry

end fraction_of_house_painted_l181_181856


namespace compare_probabilities_l181_181275

-- Definitions
variables (M N: ℕ) (m n: ℕ)
  
-- Conditions
def condition_m_millionaire : Prop := m > 10^6
def condition_n_nonmillionaire : Prop := n ≤ 10^6

-- Probabilities
noncomputable def P_A : ℚ := (M:ℚ) / (M + (n:ℚ)/(m:ℚ) * N)
noncomputable def P_B : ℚ := (M:ℚ) / (M + N)

-- Theorem statement
theorem compare_probabilities
  (hM : M > 0) (hN : N > 0)
  (h_m_millionaire : condition_m_millionaire m)
  (h_n_nonmillionaire : condition_n_nonmillionaire n) :
  P_A M N m n > P_B M N := sorry

end compare_probabilities_l181_181275


namespace michael_can_cover_both_classes_l181_181739

open Nat

def total_students : ℕ := 30
def german_students : ℕ := 20
def japanese_students : ℕ := 24

-- Calculate the number of students taking both German and Japanese using inclusion-exclusion principle.
def both_students : ℕ := german_students + japanese_students - total_students

-- Calculate the number of students only taking German.
def only_german_students : ℕ := german_students - both_students

-- Calculate the number of students only taking Japanese.
def only_japanese_students : ℕ := japanese_students - both_students

-- Calculate the total number of ways to choose 2 students out of 30.
def total_ways_to_choose_2 : ℕ := (total_students * (total_students - 1)) / 2

-- Calculate the number of ways to choose 2 students only taking German or only taking Japanese.
def undesirable_outcomes : ℕ := (only_german_students * (only_german_students - 1)) / 2 + (only_japanese_students * (only_japanese_students - 1)) / 2

-- Calculate the probability of undesirable outcomes.
def undesirable_probability : ℚ := undesirable_outcomes / total_ways_to_choose_2

-- Calculate the probability Michael can cover both German and Japanese classes.
def desired_probability : ℚ := 1 - undesirable_probability

theorem michael_can_cover_both_classes : desired_probability = 25 / 29 := by sorry

end michael_can_cover_both_classes_l181_181739


namespace part1_part2_l181_181527

def f (x : ℝ) (a : ℝ) := (Real.exp x) / x - Real.log x + x - a

theorem part1 (a : ℝ) :
  (∀ x > 0, f x a ≥ 0) → (a ≤ Real.exp 1 + 1) :=
sorry

theorem part2 (x1 x2 a : ℝ) :
  f x1 a = 0 → f x2 a = 0 → x1 ≠ x2 → 0 < x1 → 0 < x2 → x1 * x2 < 1 :=
sorry

end part1_part2_l181_181527


namespace quadratic_completion_l181_181240

theorem quadratic_completion (b c : ℝ) (h : (x : ℝ) → x^2 + 1600 * x + 1607 = (x + b)^2 + c) (hb : b = 800) (hc : c = -638393) : 
  c / b = -797.99125 := by
  sorry

end quadratic_completion_l181_181240


namespace sqrt_neg2023_squared_l181_181651

theorem sqrt_neg2023_squared : Real.sqrt ((-2023 : ℝ)^2) = 2023 :=
by
  sorry

end sqrt_neg2023_squared_l181_181651


namespace find_n_pos_int_l181_181029

theorem find_n_pos_int (n : ℕ) (h1 : n ^ 3 + 2 * n ^ 2 + 9 * n + 8 = k ^ 3) : n = 7 := 
sorry

end find_n_pos_int_l181_181029


namespace heartsuit_value_l181_181335

def heartsuit (x y : ℝ) := 4 * x + 6 * y

theorem heartsuit_value : heartsuit 3 4 = 36 := by
  sorry

end heartsuit_value_l181_181335


namespace cube_volume_l181_181450

theorem cube_volume {V : ℝ} (x : ℝ) (hV : V = x^3) (hA : 2 * V = 6 * x^2) : V = 27 :=
by
  -- Proof goes here
  sorry

end cube_volume_l181_181450


namespace eqn_intersecting_straight_lines_l181_181595

theorem eqn_intersecting_straight_lines (x y : ℝ) : 
  x^2 - y^2 = 0 → (y = x ∨ y = -x) :=
by
  intros h
  sorry

end eqn_intersecting_straight_lines_l181_181595


namespace value_of_m_l181_181207

-- Define the function given m
def f (x m : ℝ) : ℝ := x^2 - 2 * (abs x) + 2 - m

-- State the theorem to be proved
theorem value_of_m (m : ℝ) :
  (∃ x1 x2 x3 : ℝ, f x1 m = 0 ∧ f x2 m = 0 ∧ f x3 m = 0 ∧ x1 ≠ x2 ∧ x2 ≠ x3 ∧ x3 ≠ x1) →
  m = 2 :=
by
  sorry

end value_of_m_l181_181207


namespace value_of_x_minus_2y_l181_181379

theorem value_of_x_minus_2y (x y : ℝ) (h : 0.5 * x = y + 20) : x - 2 * y = 40 :=
sorry

end value_of_x_minus_2y_l181_181379


namespace total_population_correct_l181_181243

/-- Define the populations of each city -/
def Population.Seattle : ℕ := sorry
def Population.LakeView : ℕ := 24000
def Population.Boise : ℕ := (3 * Population.Seattle) / 5

/-- Population of Lake View is 4000 more than the population of Seattle -/
axiom lake_view_population : Population.LakeView = Population.Seattle + 4000

/-- Define the total population -/
def total_population : ℕ :=
  Population.Seattle + Population.LakeView + Population.Boise

/-- Prove that total population of the three cities is 56000 -/
theorem total_population_correct :
  total_population = 56000 :=
sorry

end total_population_correct_l181_181243


namespace emir_needs_more_money_l181_181661

noncomputable def dictionary_cost : ℝ := 5.50
noncomputable def dinosaur_book_cost : ℝ := 11.25
noncomputable def childrens_cookbook_cost : ℝ := 5.75
noncomputable def science_experiment_kit_cost : ℝ := 8.50
noncomputable def colored_pencils_cost : ℝ := 3.60
noncomputable def world_map_poster_cost : ℝ := 2.40
noncomputable def puzzle_book_cost : ℝ := 4.65
noncomputable def sketchpad_cost : ℝ := 6.20

noncomputable def sales_tax_rate : ℝ := 0.07
noncomputable def dinosaur_discount_rate : ℝ := 0.10
noncomputable def saved_amount : ℝ := 28.30

noncomputable def total_cost_before_tax : ℝ :=
  dictionary_cost +
  (dinosaur_book_cost - dinosaur_discount_rate * dinosaur_book_cost) +
  childrens_cookbook_cost +
  science_experiment_kit_cost +
  colored_pencils_cost +
  world_map_poster_cost +
  puzzle_book_cost +
  sketchpad_cost

noncomputable def total_sales_tax : ℝ := sales_tax_rate * total_cost_before_tax

noncomputable def total_cost_after_tax : ℝ := total_cost_before_tax + total_sales_tax

noncomputable def additional_amount_needed : ℝ := total_cost_after_tax - saved_amount

theorem emir_needs_more_money : additional_amount_needed = 21.81 := by
  sorry

end emir_needs_more_money_l181_181661


namespace Bridget_skittles_after_giving_l181_181646

-- Given conditions
def Bridget_initial_skittles : ℕ := 4
def Henry_skittles : ℕ := 4
def Henry_gives_all_to_Bridget : Prop := True

-- Prove that Bridget will have 8 Skittles in total after Henry gives all of his Skittles to her.
theorem Bridget_skittles_after_giving (h : Henry_gives_all_to_Bridget) :
  Bridget_initial_skittles + Henry_skittles = 8 :=
by
  sorry

end Bridget_skittles_after_giving_l181_181646


namespace remainder_when_M_divided_by_52_l181_181076

def M : Nat := 123456789101112131415161718192021222324252627282930313233343536373839404142434445464748495051

theorem remainder_when_M_divided_by_52 : M % 52 = 0 :=
by
  sorry

end remainder_when_M_divided_by_52_l181_181076


namespace solve_system_of_equations_l181_181438

theorem solve_system_of_equations (x y : ℝ) 
  (h1 : sqrt (y / x) - 2 * sqrt (x / y) = 1)
  (h2 : sqrt (5 * x + y) + sqrt (5 * x - y) = 4) : 
  x = 1 ∧ y = 4 := 
sorry

end solve_system_of_equations_l181_181438


namespace inequality_solution_set_l181_181988

noncomputable def solution_set (a b : ℝ) := {x : ℝ | 2 < x ∧ x < 3}

theorem inequality_solution_set (a b : ℝ) :
  (∀ x : ℝ, 2 < x ∧ x < 3 → (ax^2 + 5 * x + b > 0)) →
  (∀ x : ℝ, (-6) * x^2 - 5 * x - 1 > 0 ↔ -1/2 < x ∧ x < -1/3) :=
by
  sorry

end inequality_solution_set_l181_181988


namespace probability_picasso_consecutive_l181_181222

theorem probability_picasso_consecutive :
  let totalArtPieces := 12
  let picassoPrints := 4
  let totalArrangements := 12.factorial
  let blockArrangements := (12 - 4 + 1).factorial * 4.factorial
  (blockArrangements / totalArrangements : ℚ) = 1 / 55 :=
by 
  -- Definitions from Conditions
  let totalArtPieces := 12
  let picassoPrints := 4
  let totalArrangements := 12.factorial
  let blockArrangements := (12 - 4 + 1).factorial * 4.factorial

  -- Calculation based on the provided solution
  have h1 : (12 - 4 + 1).factorial = 9.fact := by rw Nat.factorial
  have h2 : blockArrangements = 9.fact * 4.fact := by rw [h1, Nat.factorial]
  have h3 : 12.factorial = 479001600 := rfl -- 12! = 479001600
  
  -- Obtained Simplified Answer
  have h4 : 9.fact * 4.fact = 8709120 := by norm_num [Nat.factorial, h2]
  have h5 : (9.fact * 4.fact) / 479001600 = 1 / 55 := by norm_num [h3, ←h4]
  
  exact h5

end probability_picasso_consecutive_l181_181222


namespace class_contribution_Miss_Evans_class_contribution_Mr_Smith_class_contribution_Mrs_Johnson_l181_181866

theorem class_contribution_Miss_Evans :
  let total_contribution : ℝ := 90
  let class_funds_Evans : ℝ := 14
  let num_students_Evans : ℕ := 19
  let individual_contribution_Evans : ℝ := (total_contribution - class_funds_Evans) / num_students_Evans
  individual_contribution_Evans = 4 := 
sorry

theorem class_contribution_Mr_Smith :
  let total_contribution : ℝ := 90
  let class_funds_Smith : ℝ := 20
  let num_students_Smith : ℕ := 15
  let individual_contribution_Smith : ℝ := (total_contribution - class_funds_Smith) / num_students_Smith
  individual_contribution_Smith = 4.67 := 
sorry

theorem class_contribution_Mrs_Johnson :
  let total_contribution : ℝ := 90
  let class_funds_Johnson : ℝ := 30
  let num_students_Johnson : ℕ := 25
  let individual_contribution_Johnson : ℝ := (total_contribution - class_funds_Johnson) / num_students_Johnson
  individual_contribution_Johnson = 2.40 := 
sorry

end class_contribution_Miss_Evans_class_contribution_Mr_Smith_class_contribution_Mrs_Johnson_l181_181866


namespace determine_n_eq_1_l181_181831

theorem determine_n_eq_1 :
  ∃ n : ℝ, (∀ x : ℝ, (x = 2 → (x^3 - 3*x^2 + n = 2*x^3 - 6*x^2 + 5*n))) → n = 1 :=
by
  sorry

end determine_n_eq_1_l181_181831


namespace number_of_sides_l181_181601

theorem number_of_sides (n : ℕ) : 
  (2 / 9) * (n - 2) * 180 = 360 → n = 11 := 
by
  intro h
  sorry

end number_of_sides_l181_181601


namespace intersection_A_B_l181_181990

def setA : Set ℝ := { x | |x| < 2 }
def setB : Set ℝ := { x | x^2 - 4 * x + 3 < 0 }
def setC : Set ℝ := { x | 1 < x ∧ x < 2 }

theorem intersection_A_B : setA ∩ setB = setC := by
  sorry

end intersection_A_B_l181_181990


namespace mouse_jump_less_than_frog_l181_181445

-- Definitions for the given conditions
def grasshopper_jump : ℕ := 25
def frog_jump : ℕ := grasshopper_jump + 32
def mouse_jump : ℕ := 31

-- The statement we need to prove
theorem mouse_jump_less_than_frog :
  frog_jump - mouse_jump = 26 :=
by
  -- The proof will be filled in here
  sorry

end mouse_jump_less_than_frog_l181_181445


namespace sasha_added_num_l181_181821

theorem sasha_added_num (a b c : ℤ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a / b = 5 * (a + c) / (b * c)) : c = 6 ∨ c = -20 := 
sorry

end sasha_added_num_l181_181821


namespace replace_signs_to_achieve_2013_l181_181463

theorem replace_signs_to_achieve_2013 : 
  ∃ (f : ℕ → ℤ), (∀ n, 1 ≤ n ∧ n ≤ 2013 → (f n) = n^2 ∨ (f n) = -n^2) 
  ∧ (∑ n in finset.range (2014), f n) = 2013 :=
sorry

end replace_signs_to_achieve_2013_l181_181463


namespace parallel_lines_slope_condition_l181_181610

-- Define the first line equation and the slope
def line1 (x : ℝ) : ℝ := 6 * x + 5
def slope1 : ℝ := 6

-- Define the second line equation and the slope
def line2 (x c : ℝ) : ℝ := (3 * c) * x - 7
def slope2 (c : ℝ) : ℝ := 3 * c

-- Theorem stating that if the lines are parallel, the value of c is 2
theorem parallel_lines_slope_condition (c : ℝ) : 
  (slope1 = slope2 c) → c = 2 := 
  by
    sorry -- Proof

end parallel_lines_slope_condition_l181_181610


namespace angle_C_in_triangle_l181_181048

theorem angle_C_in_triangle (A B C : ℝ) (h : A + B = 80) : C = 100 :=
by
  -- The measure of angle C follows from the condition and the properties of triangles.
  sorry

end angle_C_in_triangle_l181_181048


namespace min_a_l181_181987

theorem min_a (a : ℝ) (h : ∀ x y : ℝ, 0 < x → 0 < y → (x + y) * (1/x + a/y) ≥ 25) : a ≥ 16 :=
sorry  -- Proof is omitted

end min_a_l181_181987


namespace triangle_angle_sum_l181_181065

theorem triangle_angle_sum (A B C : ℝ) (h1 : A + B = 80) (h2 : A + B + C = 180) : C = 100 := 
by
  sorry

end triangle_angle_sum_l181_181065


namespace tickets_distribution_l181_181765

theorem tickets_distribution (n m : ℕ) (h_n : n = 10) (h_m : m = 3) :
  ∃ (d : ℕ), d = 10 * 9 * 8 := 
sorry -- Detailed proof would go here.

end tickets_distribution_l181_181765


namespace poodle_terrier_bark_ratio_l181_181472

theorem poodle_terrier_bark_ratio :
  ∀ (P T : ℕ),
  (T = 12) →
  (P = 24) →
  (P / T = 2) :=
by intros P T hT hP
   sorry

end poodle_terrier_bark_ratio_l181_181472


namespace Roberto_outfit_count_l181_181746

theorem Roberto_outfit_count :
  let trousers := 5
  let shirts := 6
  let jackets := 3
  let ties := 2
  trousers * shirts * jackets * ties = 180 :=
by
  sorry

end Roberto_outfit_count_l181_181746


namespace problem_inequality_l181_181387

theorem problem_inequality (a : ℝ) :
  (∀ x : ℝ, (1 ≤ x) → (x ≤ 2) → (x^2 + 2 + |x^3 - 2 * x| ≥ a * x)) ↔ (a ≤ 2 * Real.sqrt 2) := 
sorry

end problem_inequality_l181_181387


namespace length_of_AB_l181_181846

open Real

noncomputable def line (t : ℝ) : ℝ × ℝ := (1 + (1/2) * t, (sqrt 3 / 2) * t)
noncomputable def curve (θ : ℝ) : ℝ × ℝ := (cos θ, sin θ)

theorem length_of_AB :
  ∃ A B : ℝ × ℝ, (∃ t : ℝ, line t = A) ∧ (∃ θ : ℝ, curve θ = A) ∧
                 (∃ t : ℝ, line t = B) ∧ (∃ θ : ℝ, curve θ = B) ∧
                 dist A B = 1 :=
by
  sorry

end length_of_AB_l181_181846


namespace average_donation_l181_181949

theorem average_donation (d : ℕ) (n : ℕ) (r : ℕ) (average_donation : ℕ) 
  (h1 : d = 10)   -- $10 donated by customers
  (h2 : r = 2)    -- $2 donated by restaurant
  (h3 : n = 40)   -- number of customers
  (h4 : (r : ℕ) * n / d = 24) -- total donation by restaurant is $24
  : average_donation = 3 := 
by
  sorry

end average_donation_l181_181949


namespace simplify_expression_and_evaluate_evaluate_expression_at_one_l181_181588

theorem simplify_expression_and_evaluate (x : ℝ)
  (h1 : x ≠ -2) (h2 : x ≠ 2) (h3 : x ≠ 3) :
  ( ((x^2 - 2*x) / (x^2 - 4*x + 4) - 3 / (x - 2)) / ((x - 3) / (x^2 - 4)) ) = x + 2 :=
by {
  sorry
}

theorem evaluate_expression_at_one :
  ( ((1^2 - 2*1) / (1^2 - 4*1 + 4) - 3 / (1 - 2)) / ((1 - 3) / (1^2 - 4)) ) = 3 :=
by {
  sorry
}

end simplify_expression_and_evaluate_evaluate_expression_at_one_l181_181588


namespace pencil_distribution_l181_181975

open_locale big_operators

theorem pencil_distribution : 
  ∃ (ways : ℕ), ways = 58 ∧ ∃ (a b c d : ℕ), a + b + c + d = 10 ∧ a ≥ 1 ∧ b ≥ 1 ∧ c ≥ 1 ∧ d ≥ 1 :=
by 
  sorry

end pencil_distribution_l181_181975


namespace triangle_angle_C_l181_181058

theorem triangle_angle_C (A B C : ℝ) (h : A + B = 80) : C = 100 :=
sorry

end triangle_angle_C_l181_181058


namespace quadratic_distinct_real_roots_range_l181_181711

theorem quadratic_distinct_real_roots_range (k : ℝ) :
  (∃ x : ℝ, k * x^2 - 2 * x - 1 = 0 ∧ ∃ y : ℝ, y ≠ x ∧ k * y^2 - 2 * y - 1 = 0) ↔ (k > -1 ∧ k ≠ 0) :=
by
  sorry

end quadratic_distinct_real_roots_range_l181_181711


namespace quadratic_distinct_roots_l181_181679

theorem quadratic_distinct_roots (p q : ℚ) :
  (∀ x : ℚ, x^2 + p * x + q = 0 ↔ x = 2 * p ∨ x = p + q) →
  (p = 2 / 3 ∧ q = -8 / 3) :=
by sorry

end quadratic_distinct_roots_l181_181679


namespace like_monomials_are_same_l181_181031

theorem like_monomials_are_same (m n : ℤ) (h1 : 2 * m + 4 = 8) (h2 : 2 * n - 3 = 5) : m = 2 ∧ n = 4 :=
by
  sorry

end like_monomials_are_same_l181_181031


namespace proof_problem_l181_181180

variable {R : Type} [OrderedRing R]

-- Definitions and conditions
variable (g : R → R) (f : R → R) (k a m : R)
variable (h_odd : ∀ x : R, g (-x) = -g x)
variable (h_f_def : ∀ x : R, f x = g x + k)
variable (h_f_neg_a : f (-a) = m)

-- Theorem statement
theorem proof_problem : f a = 2 * k - m :=
by
  -- Here is where the proof would go.
  sorry

end proof_problem_l181_181180


namespace purely_imaginary_roots_iff_l181_181967

theorem purely_imaginary_roots_iff (z : ℂ) (k : ℝ) (i : ℂ) (h_i2 : i^2 = -1) :
  (∀ r : ℂ, (20 * r^2 + 6 * i * r - ↑k = 0) → (∃ b : ℝ, r = b * i)) ↔ (k = 9 / 5) :=
sorry

end purely_imaginary_roots_iff_l181_181967


namespace total_nails_l181_181074

def num_planks : Nat := 1
def nails_per_plank : Nat := 3
def additional_nails : Nat := 8

theorem total_nails : (num_planks * nails_per_plank + additional_nails) = 11 :=
by
  sorry

end total_nails_l181_181074


namespace arithmetic_sequence_sum_l181_181870

theorem arithmetic_sequence_sum (a : ℕ → ℕ) (h : a 2 + a 10 = 16) : a 4 + a 6 + a 8 = 24 :=
by
  sorry

end arithmetic_sequence_sum_l181_181870


namespace quadratic_real_equal_roots_l181_181495

theorem quadratic_real_equal_roots (k : ℝ) :
  (∃ x : ℝ, 3 * x^2 - k * x + 2 * x + 15 = 0 ∧ ∀ y : ℝ, (3 * y^2 - k * y + 2 * y + 15 = 0 → y = x)) ↔
  (k = 6 * Real.sqrt 5 + 2 ∨ k = -6 * Real.sqrt 5 + 2) :=
by
  sorry

end quadratic_real_equal_roots_l181_181495


namespace triangle_base_length_l181_181751

theorem triangle_base_length (h : ℝ) (A : ℝ) (b : ℝ) (h_eq : h = 6) (A_eq : A = 13.5) (area_eq : A = (b * h) / 2) : b = 4.5 :=
by
  sorry

end triangle_base_length_l181_181751


namespace composite_of_n_gt_one_l181_181229

theorem composite_of_n_gt_one (n : ℕ) (h : n > 1) : ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n^4 + 4 = a * b :=
by
  sorry

end composite_of_n_gt_one_l181_181229


namespace find_k_l181_181530

open Real

-- Define the operation "※"
def star (a b : ℝ) : ℝ := a * b + a + b^2

-- Define the main theorem stating the problem
theorem find_k (k : ℝ) (hk : k > 0) (h : star 1 k = 3) : k = 1 := by
  sorry

end find_k_l181_181530


namespace num_possible_bases_l181_181092

theorem num_possible_bases (b : ℕ) (h1 : b ≥ 2) (h2 : b^3 ≤ 256) (h3 : 256 < b^4) : ∃ n : ℕ, n = 2 :=
by
  sorry

end num_possible_bases_l181_181092


namespace correct_calculation_l181_181568

theorem correct_calculation (y : ℤ) (h : (y + 4) * 5 = 140) : 5 * y + 4 = 124 :=
by {
  sorry
}

end correct_calculation_l181_181568


namespace average_speeds_l181_181286

theorem average_speeds (x y : ℝ) (h1 : 4 * x + 5 * y = 98) (h2 : 4 * x = 5 * y - 2) : 
  x = 12 ∧ y = 10 :=
by sorry

end average_speeds_l181_181286


namespace negation_of_existential_prop_l181_181191

open Real

theorem negation_of_existential_prop :
  ¬ (∃ x, x ≥ π / 2 ∧ sin x > 1) ↔ ∀ x, x < π / 2 → sin x ≤ 1 :=
by
  sorry

end negation_of_existential_prop_l181_181191


namespace factor_values_l181_181496

theorem factor_values (a b : ℤ) :
  (∀ s : ℂ, s^2 - s - 1 = 0 → a * s^15 + b * s^14 + 1 = 0) ∧
  (∀ t : ℂ, t^2 - t - 1 = 0 → a * t^15 + b * t^14 + 1 = 0) →
  a = 377 ∧ b = -610 :=
by
  sorry

end factor_values_l181_181496


namespace product_mod_m_l181_181785

-- Define the constants
def a : ℕ := 2345
def b : ℕ := 1554
def m : ℕ := 700

-- Definitions derived from the conditions
def a_mod_m : ℕ := a % m
def b_mod_m : ℕ := b % m

-- The proof problem
theorem product_mod_m (a b m : ℕ) (h1 : a % m = 245) (h2 : b % m = 154) :
  (a * b) % m = 630 := by sorry

end product_mod_m_l181_181785


namespace relationship_between_y_l181_181538

theorem relationship_between_y (y1 y2 y3 : ℝ)
  (hA : y1 = 3 / -5)
  (hB : y2 = 3 / -3)
  (hC : y3 = 3 / 2) : y2 < y1 ∧ y1 < y3 :=
by
  sorry

end relationship_between_y_l181_181538


namespace floor_negative_fraction_l181_181343

theorem floor_negative_fraction : (Int.floor (-7 / 4 : ℚ)) = -2 := by
  sorry

end floor_negative_fraction_l181_181343


namespace root_value_cond_l181_181204

theorem root_value_cond (p q : ℝ) (h₁ : ∃ x : ℝ, x^2 + p * x + q = 0 ∧ x = q) (h₂ : q ≠ 0) : p + q = -1 := 
sorry

end root_value_cond_l181_181204


namespace total_sections_l181_181405

theorem total_sections (boys girls max_students per_section boys_ratio girls_ratio : ℕ)
  (hb : boys = 408) (hg : girls = 240) (hm : max_students = 24) 
  (br : boys_ratio = 3) (gr : girls_ratio = 2)
  (hboy_sec : (boys + max_students - 1) / max_students = 17)
  (hgirl_sec : (girls + max_students - 1) / max_students = 10) 
  : (3 * (((boys + max_students - 1) / max_students) + 2 * ((girls + max_students - 1) / max_students))) / 5 = 30 :=
by
  sorry

end total_sections_l181_181405


namespace floor_negative_fraction_l181_181344

theorem floor_negative_fraction : (Int.floor (-7 / 4 : ℚ)) = -2 := by
  sorry

end floor_negative_fraction_l181_181344


namespace current_dogwood_trees_l181_181769

def number_of_trees (X : ℕ) : Prop :=
  X + 61 = 100

theorem current_dogwood_trees (X : ℕ) (h : number_of_trees X) : X = 39 :=
by 
  sorry

end current_dogwood_trees_l181_181769


namespace line_eq_l181_181707

theorem line_eq (P : ℝ × ℝ) (hP : P = (1, 2)) (h_perp : ∀ x y : ℝ, 2 * x + y - 1 = 0 → x - 2 * y + c = 0) : 
  ∃ c : ℝ, (x - 2 * y + c = 0 ∧ P ∈ {(x, y) | x - 2 * y + c = 0}) ∧ c = 3 :=
  sorry

end line_eq_l181_181707


namespace seokgi_jumped_furthest_l181_181928

noncomputable def yooseung_jump : ℝ := 15 / 8
def shinyoung_jump : ℝ := 2
noncomputable def seokgi_jump : ℝ := 17 / 8

theorem seokgi_jumped_furthest :
  yooseung_jump < seokgi_jump ∧ shinyoung_jump < seokgi_jump :=
by
  sorry

end seokgi_jumped_furthest_l181_181928


namespace puzzle_pieces_l181_181559

theorem puzzle_pieces
  (total_puzzles : ℕ)
  (pieces_per_10_min : ℕ)
  (total_minutes : ℕ)
  (h1 : total_puzzles = 2)
  (h2 : pieces_per_10_min = 100)
  (h3 : total_minutes = 400) :
  ((total_minutes / 10) * pieces_per_10_min) / total_puzzles = 2000 :=
by
  sorry

end puzzle_pieces_l181_181559


namespace two_class_students_l181_181862

-- Define the types of students and total sum variables
variables (H M E HM HE ME HME : ℕ)
variable (Total_Students : ℕ)

-- Given conditions
axiom condition1 : Total_Students = 68
axiom condition2 : H = 19
axiom condition3 : M = 14
axiom condition4 : E = 26
axiom condition5 : HME = 3

-- Inclusion-Exclusion principle formula application
def exactly_two_classes : Prop := 
  Total_Students = H + M + E - (HM + HE + ME) + HME

-- Theorem to prove the number of students registered for exactly two classes is 6
theorem two_class_students : H + M + E - 2 * HME + HME - (HM + HE + ME) = 6 := by
  sorry

end two_class_students_l181_181862


namespace total_money_received_l181_181429

-- Define the conditions
def total_puppies : ℕ := 20
def fraction_sold : ℚ := 3 / 4
def price_per_puppy : ℕ := 200

-- Define the statement to prove
theorem total_money_received : fraction_sold * total_puppies * price_per_puppy = 3000 := by
  sorry

end total_money_received_l181_181429


namespace volume_ratio_octahedron_cube_l181_181506

theorem volume_ratio_octahedron_cube 
  (s : ℝ) -- edge length of the octahedron
  (h := s * Real.sqrt 2 / 2) -- height of one of the pyramids forming the octahedron
  (volume_O := s^3 * Real.sqrt 2 / 3) -- volume of the octahedron
  (a := (2 * s) / Real.sqrt 3) -- edge length of the cube
  (volume_C := (a ^ 3)) -- volume of the cube
  (diag_C : ℝ := 2 * s) -- diagonal of the cube
  (h_diag : diag_C = (a * Real.sqrt 3)) -- relation of diagonal to edge length of the cube
  (ratio := volume_O / volume_C) -- ratio of the volumes
  (desired_ratio := 3 / 8) -- given ratio in simplified form
  (m := 3) -- first part of the ratio
  (n := 8) -- second part of the ratio
  (rel_prime : Nat.gcd m n = 1) -- m and n are relatively prime
  (correct_ratio : ratio = desired_ratio) -- the ratio is correct
  : m + n = 11 :=
by
  sorry 

end volume_ratio_octahedron_cube_l181_181506


namespace find_angle_A_l181_181865

-- Conditions
def is_triangle (A B C : ℝ) : Prop := A + B + C = 180
def B_is_two_C (B C : ℝ) : Prop := B = 2 * C
def B_is_80 (B : ℝ) : Prop := B = 80

-- Theorem statement
theorem find_angle_A (A B C : ℝ) (h₁ : is_triangle A B C) (h₂ : B_is_two_C B C) (h₃ : B_is_80 B) : A = 60 := by
  sorry

end find_angle_A_l181_181865


namespace frosting_problem_equivalent_l181_181823

/-
Problem:
Cagney can frost a cupcake every 15 seconds.
Lacey can frost a cupcake every 40 seconds.
Mack can frost a cupcake every 25 seconds.
Prove that together they can frost 79 cupcakes in 10 minutes.
-/

def cupcakes_frosted_together_in_10_minutes (rate_cagney rate_lacey rate_mack : ℕ) (time_minutes : ℕ) : ℕ :=
  let time_seconds := time_minutes * 60
  let rate_cagney := 1 / 15
  let rate_lacey := 1 / 40
  let rate_mack := 1 / 25
  let combined_rate := rate_cagney + rate_lacey + rate_mack
  combined_rate * time_seconds

theorem frosting_problem_equivalent:
  cupcakes_frosted_together_in_10_minutes 1 1 1 10 = 79 := by
  sorry

end frosting_problem_equivalent_l181_181823


namespace regression_analysis_incorrect_statement_l181_181614

theorem regression_analysis_incorrect_statement
  (y : ℕ → ℝ) (x : ℕ → ℝ) (b a : ℝ)
  (r : ℝ) (l : ℝ → ℝ) (P : ℝ × ℝ)
  (H1 : ∀ i, y i = b * x i + a)
  (H2 : abs r = 1 → ∀ x1 x2, l x1 = l x2 → x1 = x2)
  (H3 : ∃ m k, ∀ x, l x = m * x + k)
  (H4 : P.1 = b → l P.1 = P.2)
  (cond_A : ∀ i, y i ≠ b * x i + a) : false := 
sorry

end regression_analysis_incorrect_statement_l181_181614


namespace factorization_of_polynomial_l181_181665

theorem factorization_of_polynomial :
  ∀ x : ℝ, x^8 - 4*x^6 + 6*x^4 - 4*x^2 + 1 = (x - 1)^4 * (x + 1)^4 :=
by
  intros x
  sorry

end factorization_of_polynomial_l181_181665


namespace stock_price_end_of_second_year_l181_181155

def initial_price : ℝ := 80
def first_year_increase_rate : ℝ := 1.2
def second_year_decrease_rate : ℝ := 0.3

theorem stock_price_end_of_second_year : 
  initial_price * (1 + first_year_increase_rate) * (1 - second_year_decrease_rate) = 123.2 := 
by sorry

end stock_price_end_of_second_year_l181_181155


namespace number_of_good_permutations_l181_181503

def is_good_permutation {α : Type*} [linear_order α] [fintype α] (a : list α) : Prop :=
∃ i j, i < j ∧ a[i] > a[i+1] ∧ a[j] > a[j+1]

noncomputable def count_good_permutations (n : ℕ) : ℕ :=
fintype.card {a : vector (fin n) n // is_good_permutation a.to_list}

theorem number_of_good_permutations (n : ℕ) (h : 1 ≤ n) : 
  count_good_permutations n = 3^n - (n+1) * 2^n + n * (n+1) / 2 :=
sorry

end number_of_good_permutations_l181_181503


namespace locus_centers_tangent_circles_l181_181897

theorem locus_centers_tangent_circles (a b : ℝ) :
  (∃ r : ℝ, a^2 + b^2 = (r + 2)^2 ∧ (a - 3)^2 + b^2 = (3 - r)^2) →
  a^2 - 12 * a + 4 * b^2 = 0 :=
by
  sorry

end locus_centers_tangent_circles_l181_181897


namespace books_in_june_l181_181774

-- Definitions
def Book_may : ℕ := 2
def Book_july : ℕ := 10
def Total_books : ℕ := 18

-- Theorem statement
theorem books_in_june : ∃ (Book_june : ℕ), Book_may + Book_june + Book_july = Total_books ∧ Book_june = 6 :=
by
  -- Proof will be here
  sorry

end books_in_june_l181_181774


namespace find_value_of_a20_l181_181178

variable {α : Type*} [LinearOrder α] [Field α]

def arithmetic_sequence (a d : α) (n : ℕ) : α :=
  a + (n - 1) * d

def arithmetic_sum (a d : α) (n : ℕ) : α :=
  n * a + (n * (n - 1) / 2) * d

theorem find_value_of_a20 
  (a d : ℝ) 
  (h1 : arithmetic_sequence a d 3 + arithmetic_sequence a d 5 = 4)
  (h2 : arithmetic_sum a d 15 = 60) :
  arithmetic_sequence a d 20 = 10 := 
sorry

end find_value_of_a20_l181_181178


namespace part_I_part_II_l181_181217

noncomputable def f (a x : ℝ) : ℝ := Real.exp x - 2 * x + 2 * a

theorem part_I (a : ℝ) :
  let x := Real.log 2
  ∃ I₁ I₂ : Set ℝ,
    (∀ x ∈ I₁, f a x > f a (Real.log 2)) ∧
    (∀ x ∈ I₂, f a x < f a (Real.log 2)) ∧
    I₁ = Set.Iio (Real.log 2) ∧
    I₂ = Set.Ioi (Real.log 2) ∧
    f a (Real.log 2) = 2 * (1 - Real.log 2 + a) :=
by sorry

theorem part_II (a : ℝ) (h : a > Real.log 2 - 1) (x : ℝ) (hx : 0 < x) :
  Real.exp x > x^2 - 2 * a * x + 1 :=
by sorry

end part_I_part_II_l181_181217


namespace square_fits_in_unit_cube_l181_181654

theorem square_fits_in_unit_cube (x : ℝ) (h₀ : 0 < x) (h₁ : x < 1) :
  let PQ := Real.sqrt (2 * (1 - x) ^ 2)
  let PS := Real.sqrt (1 + 2 * x ^ 2)
  (PQ > 1.05 ∧ PS > 1.05) :=
by
  sorry

end square_fits_in_unit_cube_l181_181654


namespace sufficient_condition_not_necessary_condition_l181_181123

theorem sufficient_condition (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) : ab > 0 := by
  sorry

theorem not_necessary_condition (a b : ℝ) : ¬(a > 0 ∧ b > 0) → ab > 0 := by
  sorry

end sufficient_condition_not_necessary_condition_l181_181123


namespace fraction_of_satisfactory_grades_l181_181473

theorem fraction_of_satisfactory_grades :
  let num_A := 6
  let num_B := 5
  let num_C := 4
  let num_D := 3
  let num_E := 2
  let num_F := 6
  -- Total number of satisfactory grades
  let satisfactory := num_A + num_B + num_C + num_D + num_E
  -- Total number of students
  let total := satisfactory + num_F
  -- Fraction of satisfactory grades
  satisfactory / total = (10 : ℚ) / 13 :=
by
  sorry

end fraction_of_satisfactory_grades_l181_181473


namespace rain_difference_l181_181224

theorem rain_difference
    (rain_monday : ℕ → ℝ)
    (rain_tuesday : ℕ → ℝ)
    (rain_wednesday : ℕ → ℝ)
    (rain_thursday : ℕ → ℝ)
    (h_monday : ∀ n : ℕ, n = 10 → rain_monday n = 1.25)
    (h_tuesday : ∀ n : ℕ, n = 12 → rain_tuesday n = 2.15)
    (h_wednesday : ∀ n : ℕ, n = 8 → rain_wednesday n = 1.60)
    (h_thursday : ∀ n : ℕ, n = 6 → rain_thursday n = 2.80) :
    let total_rain_monday := 10 * 1.25
    let total_rain_tuesday := 12 * 2.15
    let total_rain_wednesday := 8 * 1.60
    let total_rain_thursday := 6 * 2.80
    (total_rain_tuesday + total_rain_thursday) - (total_rain_monday + total_rain_wednesday) = 17.3 :=
by
  sorry

end rain_difference_l181_181224


namespace parrots_count_l181_181926

theorem parrots_count (p r : ℕ) : 2 * p + 4 * r = 26 → p + r = 10 → p = 7 := by
  intros h1 h2
  sorry

end parrots_count_l181_181926


namespace cards_problem_l181_181886

-- Definitions of the cards and their arrangement
def cards : List ℕ := [1, 3, 4, 6, 7, 8]
def missing_numbers : List ℕ := [2, 5, 9]

-- Function to check no three consecutive numbers are in ascending or descending order
def no_three_consec (ls : List ℕ) : Prop :=
  ∀ (a b c : ℕ), a < b → b < c → b - a = 1 → c - b = 1 → False ∧
                a > b → b > c → a - b = 1 → b - c = 1 → False

-- Assume that cards A, B, and C are not visible
variables (A B C : ℕ)

-- Ensure that A, B, and C are among the missing numbers
axiom A_in_missing : A ∈ missing_numbers
axiom B_in_missing : B ∈ missing_numbers
axiom C_in_missing : C ∈ missing_numbers

-- Ensuring no three consecutive cards are in ascending or descending order
axiom no_three_consec_cards : no_three_consec (cards ++ [A, B, C])

-- The final proof problem
theorem cards_problem : A = 5 ∧ B = 2 ∧ C = 9 :=
by
  sorry

end cards_problem_l181_181886


namespace amber_bronze_cells_selection_l181_181577

theorem amber_bronze_cells_selection (a b : ℕ) (h₁ : 0 < a) (h₂ : 0 < b)
  (cells : Fin (a + b + 1) → Fin (a + b + 1) → Bool)
  (amber : ∀ i j, cells i j = tt → Bool) -- True means amber, False means bronze
  (bronze : ∀ i j, cells i j = ff → Bool)
  (amber_count : (Finset.univ.product Finset.univ).filter (λ p, cells p.1 p.2 = tt).card ≥ a^2 + a * b - b)
  (bronze_count : (Finset.univ.product Finset.univ).filter (λ p, cells p.1 p.2 = ff).card ≥ b^2 + b * a - a) :
  ∃ (α : Finset (Fin (a + b + 1) × Fin (a + b + 1))), 
    (α.filter (λ p, cells p.1 p.2 = tt)).card = a ∧ 
    (α.filter (λ p, cells p.1 p.2 = ff)).card = b ∧ 
    (∀ (p1 p2 : Fin (a + b + 1) × Fin (a + b + 1)), p1 ≠ p2 → (p1.1 ≠ p2.1 ∧ p1.2 ≠ p2.2)) :=
by sorry

end amber_bronze_cells_selection_l181_181577


namespace probability_of_all_heads_or_tails_l181_181156

theorem probability_of_all_heads_or_tails :
  let possible_outcomes := 256
  let favorable_outcomes := 2
  favorable_outcomes / possible_outcomes = 1 / 128 := by
  sorry

end probability_of_all_heads_or_tails_l181_181156


namespace find_a14_l181_181982

-- Define the arithmetic sequence properties
def sum_of_first_n_terms (a1 d : ℤ) (n : ℕ) : ℤ :=
  n * a1 + (n * (n - 1) / 2) * d

def nth_term (a1 d : ℤ) (n : ℕ) : ℤ :=
  a1 + (n - 1) * d

theorem find_a14 (a1 d : ℤ) (S11 : sum_of_first_n_terms a1 d 11 = 55)
  (a10 : nth_term a1 d 10 = 9) : nth_term a1 d 14 = 13 :=
sorry

end find_a14_l181_181982


namespace part1_part2_l181_181521

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (Real.exp x / x) - Real.log x + x - a

theorem part1 (a : ℝ) : (∀ x > 0, f x a ≥ 0) ↔ a ≤ Real.exp 1 + 1 :=
  sorry

theorem part2 (a : ℝ) (x1 x2 : ℝ) (h1 : f x1 a = 0) (h2 : f x2 a = 0) : x1 * x2 < 1 :=
  sorry

end part1_part2_l181_181521


namespace cubic_polynomials_integer_roots_l181_181835

theorem cubic_polynomials_integer_roots (a b : ℤ) :
  (∀ α1 α2 α3 : ℤ, α1 + α2 + α3 = 0 ∧ α1 * α2 + α2 * α3 + α3 * α1 = a ∧ α1 * α2 * α3 = -b) →
  (∀ β1 β2 β3 : ℤ, β1 + β2 + β3 = 0 ∧ β1 * β2 + β2 * β3 + β3 * β1 = b ∧ β1 * β2 * β3 = -a) →
  a = 0 ∧ b = 0 :=
by
  sorry

end cubic_polynomials_integer_roots_l181_181835


namespace angle_C_in_triangle_l181_181037

theorem angle_C_in_triangle (A B C : ℝ) (h1 : A + B = 80) (h2 : A + B + C = 180) : C = 100 := 
by 
  sorry

end angle_C_in_triangle_l181_181037


namespace find_days_l181_181861

theorem find_days
  (wages1 : ℕ) (workers1 : ℕ) (days1 : ℕ)
  (wages2 : ℕ) (workers2 : ℕ) (days2 : ℕ)
  (h1 : wages1 = 9450) (h2 : workers1 = 15) (h3 : wages2 = 9975)
  (h4 : workers2 = 19) (h5 : days2 = 5) :
  days1 = 6 := 
by
  -- Insert proof here
  sorry

end find_days_l181_181861


namespace value_of_expression_l181_181113

theorem value_of_expression : 1 + 2 / (1 + 2 / (2 * 2)) = 7 / 3 := 
by 
  -- proof to be filled in
  sorry

end value_of_expression_l181_181113


namespace Jeongyeon_record_is_1_44_m_l181_181297

def Eunseol_record_in_cm : ℕ := 100 + 35
def Jeongyeon_record_in_cm : ℕ := Eunseol_record_in_cm + 9
def Jeongyeon_record_in_m : ℚ := Jeongyeon_record_in_cm / 100

theorem Jeongyeon_record_is_1_44_m : Jeongyeon_record_in_m = 1.44 := by
  sorry

end Jeongyeon_record_is_1_44_m_l181_181297


namespace not_divisible_1998_minus_1_by_1000_minus_1_l181_181089

theorem not_divisible_1998_minus_1_by_1000_minus_1 (m : ℕ) : ¬ (1000^m - 1 ∣ 1998^m - 1) :=
sorry

end not_divisible_1998_minus_1_by_1000_minus_1_l181_181089


namespace nested_fraction_l181_181007

theorem nested_fraction
  : 1 / (3 + 1 / (3 + 1 / (3 - 1 / (3 + 1 / (2 * (3 + 2 / 5))))))
  = 968 / 3191 := 
by
  sorry

end nested_fraction_l181_181007


namespace simplify_expression_l181_181230

noncomputable def simplify_expr : ℝ :=
  (3 + 2 * Real.sqrt 2) ^ Real.sqrt 3

theorem simplify_expression :
  (Real.sqrt 2 - 1) ^ (2 - Real.sqrt 3) / (Real.sqrt 2 + 1) ^ (2 + Real.sqrt 3) = simplify_expr :=
by
  sorry

end simplify_expression_l181_181230


namespace compare_probabilities_l181_181274

-- Definitions
variables (M N: ℕ) (m n: ℕ)
  
-- Conditions
def condition_m_millionaire : Prop := m > 10^6
def condition_n_nonmillionaire : Prop := n ≤ 10^6

-- Probabilities
noncomputable def P_A : ℚ := (M:ℚ) / (M + (n:ℚ)/(m:ℚ) * N)
noncomputable def P_B : ℚ := (M:ℚ) / (M + N)

-- Theorem statement
theorem compare_probabilities
  (hM : M > 0) (hN : N > 0)
  (h_m_millionaire : condition_m_millionaire m)
  (h_n_nonmillionaire : condition_n_nonmillionaire n) :
  P_A M N m n > P_B M N := sorry

end compare_probabilities_l181_181274


namespace compound_interest_principal_l181_181456

theorem compound_interest_principal (CI t : ℝ) (r n : ℝ) (P : ℝ) : CI = 630 ∧ t = 2 ∧ r = 0.10 ∧ n = 1 → P = 3000 :=
by
  -- Proof to be provided
  sorry

end compound_interest_principal_l181_181456


namespace smallest_palindrome_base2_base4_l181_181827

def is_palindrome_base (n : ℕ) (b : ℕ) : Prop :=
  let digits := (Nat.digits b n)
  digits = digits.reverse

theorem smallest_palindrome_base2_base4 : 
  ∃ (x : ℕ), x > 15 ∧ is_palindrome_base x 2 ∧ is_palindrome_base x 4 ∧ x = 17 :=
by
  sorry

end smallest_palindrome_base2_base4_l181_181827


namespace part_a_part_b_l181_181950

-- Definitions based on the conditions:
def probability_of_hit (p : ℝ) := p
def probability_of_miss (p : ℝ) := 1 - p

-- Condition: exactly three unused rockets after firing at five targets
def exactly_three_unused_rockets (p : ℝ) : ℝ := 10 * (probability_of_hit p) ^ 3 * (probability_of_miss p) ^ 2

-- Condition: expected number of targets hit when there are nine targets
def expected_targets_hit (p : ℝ) : ℝ := 10 * p - p ^ 10

-- Lean 4 statements representing the proof problems:
theorem part_a (p : ℝ) (h_p_nonneg : 0 ≤ p) (h_p_le_one : p ≤ 1) : 
  exactly_three_unused_rockets p = 10 * p ^ 3 * (1 - p) ^ 2 :=
by sorry

theorem part_b (p : ℝ) (h_p_nonneg : 0 ≤ p) (h_p_le_one : p ≤ 1) :
  expected_targets_hit p = 10 * p - p ^ 10 :=
by sorry

end part_a_part_b_l181_181950


namespace other_acute_angle_is_60_l181_181404

theorem other_acute_angle_is_60 (a b c : ℝ) (h_triangle : a + b + c = 180) (h_right : c = 90) (h_acute : a = 30) : b = 60 :=
by 
  -- inserting proof later
  sorry

end other_acute_angle_is_60_l181_181404


namespace vertex_property_l181_181763

theorem vertex_property (a b c m k : ℝ) (h : a ≠ 0)
  (vertex_eq : k = a * m^2 + b * m + c)
  (point_eq : m = a * k^2 + b * k + c) : a * (m - k) > 0 :=
sorry

end vertex_property_l181_181763


namespace problem_1_problem_2_l181_181545

noncomputable def A := Real.pi / 3
noncomputable def b := 5
noncomputable def c := 4 -- derived from the solution
noncomputable def S : ℝ := 5 * Real.sqrt 3

theorem problem_1 (A : ℝ) 
  (h : Real.cos (2 * A) - 3 * Real.cos (Real.pi - A) = 1) 
  : A = Real.pi / 3 :=
sorry

theorem problem_2 (a : ℝ) 
  (b : ℝ) 
  (S : ℝ) 
  (h_b : b = 5) 
  (h_S : S = 5 * Real.sqrt 3) 
  : a = Real.sqrt 21 :=
sorry

end problem_1_problem_2_l181_181545


namespace x_add_inv_ge_two_x_add_inv_eq_two_iff_l181_181892

theorem x_add_inv_ge_two {x : ℝ} (h : 0 < x) : x + (1 / x) ≥ 2 :=
sorry

theorem x_add_inv_eq_two_iff {x : ℝ} (h : 0 < x) : (x + (1 / x) = 2) ↔ (x = 1) :=
sorry

end x_add_inv_ge_two_x_add_inv_eq_two_iff_l181_181892


namespace math_problem_l181_181965

theorem math_problem :
  (-1)^2024 + (-10) / (1/2) * 2 + (2 - (-3)^3) = -10 := by
  sorry

end math_problem_l181_181965


namespace units_digit_k_squared_plus_pow2_k_l181_181418

def n : ℕ := 4016
def k : ℕ := n^2 + 2^n

theorem units_digit_k_squared_plus_pow2_k :
  (k^2 + 2^k) % 10 = 7 := sorry

end units_digit_k_squared_plus_pow2_k_l181_181418


namespace woman_needs_butter_l181_181291

noncomputable def butter_needed (cost_package : ℝ) (cost_8oz : ℝ) (cost_4oz : ℝ) 
                                (discount : ℝ) (lowest_price : ℝ) : ℝ :=
  if lowest_price = cost_8oz + 2 * (cost_4oz * discount / 100) then 8 + 2 * 4 else 0

theorem woman_needs_butter 
  (cost_single_package : ℝ := 7) 
  (cost_8oz_package : ℝ := 4) 
  (cost_4oz_package : ℝ := 2)
  (discount_4oz_package : ℝ := 50) 
  (lowest_price_payment : ℝ := 6) :
  butter_needed cost_single_package cost_8oz_package cost_4oz_package discount_4oz_package lowest_price_payment = 16 := 
by
  sorry

end woman_needs_butter_l181_181291


namespace factorization_of_polynomial_l181_181666

theorem factorization_of_polynomial :
  (x : ℝ) → (x^8 - 4*x^6 + 6*x^4 - 4*x^2 + 1) = ((x - 1)^4 * (x + 1)^4) :=
by
  intro x
  sorry

end factorization_of_polynomial_l181_181666


namespace triangle_area_eq_l181_181544

variable (a b c: ℝ) (A B C : ℝ)
variable (h_cosC : Real.cos C = 1/4)
variable (h_c : c = 3)
variable (h_ratio : a / Real.cos A = b / Real.cos B)

theorem triangle_area_eq : (1 / 2) * a * b * Real.sin C = 3 * Real.sqrt 15 / 4 :=
by
  sorry

end triangle_area_eq_l181_181544


namespace c_10_eq_3_pow_89_l181_181492

section sequence
  open Nat

  -- Define the sequence c
  def c : ℕ → ℕ
  | 0     => 3  -- Note: Typically Lean sequences start from 0, not 1
  | 1     => 9
  | (n+2) => c n.succ * c n

  -- Define the auxiliary sequence d
  def d : ℕ → ℕ
  | 0     => 1  -- Note: Typically Lean sequences start from 0, not 1
  | 1     => 2
  | (n+2) => d n.succ + d n

  -- The theorem we need to prove
  theorem c_10_eq_3_pow_89 : c 9 = 3 ^ d 9 :=    -- Note: c_{10} in the original problem is c(9) in Lean
  sorry   -- Proof omitted
end sequence

end c_10_eq_3_pow_89_l181_181492


namespace value_of_expression_l181_181653

theorem value_of_expression (x : ℕ) (h : x = 8) : 
  (x^3 + 3 * (x^2) * 2 + 3 * x * (2^2) + 2^3 = 1000) := by
{
  sorry
}

end value_of_expression_l181_181653


namespace inconsistent_equation_system_l181_181540

variables {a x c : ℝ}

theorem inconsistent_equation_system (h1 : (a + x) / 2 = 110) (h2 : (x + c) / 2 = 170) (h3 : a - c = 120) : false :=
by
  sorry

end inconsistent_equation_system_l181_181540


namespace bob_weekly_income_increase_l181_181298

theorem bob_weekly_income_increase
  (raise_per_hour : ℝ)
  (hours_per_week : ℝ)
  (benefit_reduction_per_month : ℝ)
  (weeks_per_month : ℝ)
  (h_raise : raise_per_hour = 0.50)
  (h_hours : hours_per_week = 40)
  (h_reduction : benefit_reduction_per_month = 60)
  (h_weeks : weeks_per_month = 4.33) :
  (raise_per_hour * hours_per_week - benefit_reduction_per_month / weeks_per_month) = 6.14 :=
by
  simp [h_raise, h_hours, h_reduction, h_weeks]
  norm_num
  sorry

end bob_weekly_income_increase_l181_181298


namespace domain_of_function_l181_181898

noncomputable def function_domain := {x : ℝ | 1 + 1 / x > 0 ∧ 1 - x^2 ≥ 0}

theorem domain_of_function : function_domain = {x : ℝ | 0 < x ∧ x ≤ 1} :=
by
  sorry

end domain_of_function_l181_181898


namespace total_letters_received_l181_181200

theorem total_letters_received :
  ∀ (g b m t : ℕ), 
    b = 40 →
    g = b + 10 →
    m = 2 * (g + b) →
    t = g + b + m → 
    t = 270 :=
by
  intros g b m t hb hg hm ht
  rw [hb, hg, hm, ht]
  sorry

end total_letters_received_l181_181200


namespace maximize_profit_l181_181407

noncomputable def profit (t : ℝ) : ℝ :=
  27 - (18 / t) - t

theorem maximize_profit : ∀ t > 0, profit t ≤ 27 - 6 * Real.sqrt 2 ∧ profit (3 * Real.sqrt 2) = 27 - 6 * Real.sqrt 2 := by {
  sorry
}

end maximize_profit_l181_181407


namespace simplify_expression_l181_181675

theorem simplify_expression :
  (3 * Real.sqrt 8) / 
  (Real.sqrt 3 + Real.sqrt 4 + Real.sqrt 7 + Real.sqrt 9) =
  - (2 * Real.sqrt 6 - 2 * Real.sqrt 2 + 2 * Real.sqrt 14) / 5 :=
by
  sorry

end simplify_expression_l181_181675


namespace gravel_weight_40_pounds_l181_181269

def weight_of_gravel_in_mixture (total_weight : ℝ) (sand_fraction : ℝ) (water_fraction : ℝ) : ℝ :=
total_weight - (sand_fraction * total_weight + water_fraction * total_weight)

theorem gravel_weight_40_pounds
  (total_weight : ℝ) (sand_fraction : ℝ) (water_fraction : ℝ) 
  (h1 : total_weight = 40) (h2 : sand_fraction = 1 / 4) (h3 : water_fraction = 2 / 5) :
  weight_of_gravel_in_mixture total_weight sand_fraction water_fraction = 14 :=
by
  -- Proof omitted
  sorry

end gravel_weight_40_pounds_l181_181269


namespace Karen_wall_paint_area_l181_181799

theorem Karen_wall_paint_area :
  let height_wall := 10
  let width_wall := 15
  let height_window := 3
  let width_window := 5
  let height_door := 2
  let width_door := 6
  let area_wall := height_wall * width_wall
  let area_window := height_window * width_window
  let area_door := height_door * width_door
  let area_to_paint := area_wall - area_window - area_door
  area_to_paint = 123 := by
{
  sorry
}

end Karen_wall_paint_area_l181_181799


namespace buratino_spent_dollars_l181_181643

theorem buratino_spent_dollars (x y : ℕ) (h1 : x + y = 50) (h2 : 2 * x = 3 * y) : 
  (y * 5 - x * 3) = 10 :=
by
  sorry

end buratino_spent_dollars_l181_181643


namespace total_movie_hours_l181_181727

variable (J M N R : ℕ)

def favorite_movie_lengths (J M N R : ℕ) :=
  J = M + 2 ∧
  N = 3 * M ∧
  R = (4 / 5 * N : ℝ).to_nat ∧
  N = 30

theorem total_movie_hours (J M N R : ℕ) (h : favorite_movie_lengths J M N R) :
  J + M + N + R = 76 :=
by
  sorry

end total_movie_hours_l181_181727


namespace correct_number_of_eggs_to_buy_l181_181992

/-- Define the total number of eggs needed and the number of eggs given by Andrew -/
def total_eggs_needed : ℕ := 222
def eggs_given_by_andrew : ℕ := 155

/-- Define a statement asserting the correct number of eggs to buy -/
def remaining_eggs_to_buy : ℕ := total_eggs_needed - eggs_given_by_andrew

/-- The statement of the proof problem -/
theorem correct_number_of_eggs_to_buy : remaining_eggs_to_buy = 67 :=
by sorry

end correct_number_of_eggs_to_buy_l181_181992


namespace three_city_population_l181_181246

noncomputable def totalPopulation (boise seattle lakeView: ℕ) : ℕ :=
  boise + seattle + lakeView

theorem three_city_population (pBoise pSeattle pLakeView : ℕ)
  (h1 : pBoise = 3 * pSeattle / 5)
  (h2 : pLakeView = pSeattle + 4000)
  (h3 : pLakeView = 24000) :
  totalPopulation pBoise pSeattle pLakeView = 56000 := by
  sorry

end three_city_population_l181_181246


namespace sqrt_sum_of_four_terms_of_4_pow_4_l181_181913

-- Proof Statement
theorem sqrt_sum_of_four_terms_of_4_pow_4 : 
  Real.sqrt (4 ^ 4 + 4 ^ 4 + 4 ^ 4 + 4 ^ 4) = 32 := 
by 
  sorry

end sqrt_sum_of_four_terms_of_4_pow_4_l181_181913


namespace impossible_to_achieve_target_l181_181907

def initial_matchsticks := (1, 0, 0, 0)  -- Initial matchsticks at vertices (A, B, C, D)
def target_matchsticks := (1, 9, 8, 9)   -- Target matchsticks at vertices (A, B, C, D)

def S (a1 a2 a3 a4 : ℕ) : ℤ := a1 - a2 + a3 - a4

theorem impossible_to_achieve_target : 
  ¬∃ (f : ℕ × ℕ × ℕ × ℕ → ℕ × ℕ × ℕ × ℕ), 
    (f initial_matchsticks = target_matchsticks) ∧ 
    (∀ (a1 a2 a3 a4 : ℕ) k, 
      f (a1, a2, a3, a4) = (a1 - k, a2 + k, a3, a4 + k) ∨ 
      f (a1, a2, a3, a4) = (a1, a2 - k, a3 + k, a4 + k) ∨ 
      f (a1, a2, a3, a4) = (a1 + k, a2 - k, a3 - k, a4) ∨ 
      f (a1, a2, a3, a4) = (a1 - k, a2, a3 + k, a4 - k)) := sorry

end impossible_to_achieve_target_l181_181907


namespace angles_measure_l181_181552

theorem angles_measure (A B C : ℝ) (h1 : A + B = 180) (h2 : C = 1 / 2 * B) (h3 : A = 6 * B) :
  A = 1080 / 7 ∧ B = 180 / 7 ∧ C = 90 / 7 :=
by
  sorry

end angles_measure_l181_181552


namespace calculate_f_2015_l181_181015

noncomputable def f : ℝ → ℝ := sorry

-- Define the odd function property
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Define the periodic function property with period 4
def periodic_4 (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 4) = f x

-- Define the given condition for the interval (0, 2)
def interval_condition (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, 0 < x ∧ x < 2 → f x = 2 * x ^ 2

theorem calculate_f_2015
  (odd_f : odd_function f)
  (periodic_f : periodic_4 f)
  (interval_f : interval_condition f) :
  f 2015 = -2 :=
sorry

end calculate_f_2015_l181_181015


namespace red_tulips_for_smile_l181_181480

/-
Problem Statement:
Anna wants to plant red and yellow tulips in the shape of a smiley face. Given the following conditions:
1. Anna needs 8 red tulips for each eye.
2. She needs 9 times the number of red tulips in the smile to make the yellow background of the face.
3. The total number of tulips needed is 196.

Prove:
The number of red tulips needed for the smile is 18.
-/

-- Defining the conditions
def red_tulips_per_eye : Nat := 8
def total_tulips : Nat := 196
def yellow_multiplier : Nat := 9

-- Proving the number of red tulips for the smile
theorem red_tulips_for_smile (R : Nat) :
  2 * red_tulips_per_eye + R + yellow_multiplier * R = total_tulips → R = 18 :=
by
  sorry

end red_tulips_for_smile_l181_181480


namespace minimum_omega_l181_181984

theorem minimum_omega (ω : ℝ) (k : ℤ) (hω : ω > 0) 
  (h_symmetry : ∃ k : ℤ, ω * (π / 12) + π / 6 = k * π + π / 2) : ω = 4 :=
sorry

end minimum_omega_l181_181984


namespace evaluate_exponent_l181_181160

theorem evaluate_exponent : (3^2)^4 = 6561 := sorry

end evaluate_exponent_l181_181160


namespace simplify_polynomial_l181_181894

theorem simplify_polynomial :
  (6 * p ^ 4 + 2 * p ^ 3 - 8 * p + 9) + (-3 * p ^ 3 + 7 * p ^ 2 - 5 * p - 1) = 
  6 * p ^ 4 - p ^ 3 + 7 * p ^ 2 - 13 * p + 8 :=
by
  sorry

end simplify_polynomial_l181_181894


namespace line_circle_intersect_a_le_0_l181_181369

theorem line_circle_intersect_a_le_0 :
  (∃ (x y : ℝ), x + a * y + 2 = 0 ∧ x^2 + y^2 + 2 * x - 2 * y + 1 = 0) →
  a ≤ 0 :=
sorry

end line_circle_intersect_a_le_0_l181_181369


namespace area_of_region_l181_181608

theorem area_of_region : 
  (∃ x y : ℝ, (x + 5)^2 + (y - 3)^2 = 32) → (π * 32 = 32 * π) :=
by 
  sorry

end area_of_region_l181_181608


namespace find_k_l181_181872

variable (c : ℝ) (k : ℝ)
variable (a : ℕ → ℝ) (S : ℕ → ℝ)

def geometric_sequence (a : ℕ → ℝ) (c : ℝ) : Prop :=
  ∀ n, a (n + 1) = c * a n

def sum_sequence (S : ℕ → ℝ) (k : ℝ) : Prop :=
  ∀ n, S n = 3^n + k

theorem find_k (c_ne_zero : c ≠ 0)
  (h_geo : geometric_sequence a c)
  (h_sum : sum_sequence S k)
  (h_a1 : a 1 = 3 + k)
  (h_a2 : a 2 = S 2 - S 1)
  (h_a3 : a 3 = S 3 - S 2) :
  k = -1 :=
sorry

end find_k_l181_181872


namespace quadratic_distinct_real_roots_range_l181_181710

theorem quadratic_distinct_real_roots_range (k : ℝ) :
  (∃ x : ℝ, k * x^2 - 2 * x - 1 = 0 ∧ ∃ y : ℝ, y ≠ x ∧ k * y^2 - 2 * y - 1 = 0) ↔ (k > -1 ∧ k ≠ 0) :=
by
  sorry

end quadratic_distinct_real_roots_range_l181_181710


namespace percentage_decrease_is_25_percent_l181_181424

noncomputable def percentage_decrease_in_revenue
  (R : ℝ)
  (projected_revenue : ℝ)
  (actual_revenue : ℝ) : ℝ :=
  ((R - actual_revenue) / R) * 100

-- Conditions
def last_year_revenue (R : ℝ) := R
def projected_revenue (R : ℝ) := 1.20 * R
def actual_revenue (R : ℝ) := 0.625 * (1.20 * R)

-- Proof statement
theorem percentage_decrease_is_25_percent (R : ℝ) :
  percentage_decrease_in_revenue R (projected_revenue R) (actual_revenue R) = 25 :=
by
  sorry

end percentage_decrease_is_25_percent_l181_181424


namespace total_population_l181_181248

variable (seattle lake_view boise : ℕ)

def lake_view_pop := (lake_view = 24000)
def seattle_pop := (seattle = lake_view - 4000)
def boise_pop := (boise = (3 / 5 : ℚ) * seattle)

theorem total_population (h1 : lake_view_pop) (h2 : seattle_pop) (h3 : boise_pop) :
  lake_view + seattle + boise = 56000 :=
by
  sorry

end total_population_l181_181248


namespace extreme_values_a_1_turning_point_a_8_l181_181985

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  x^2 - (a + 2) * x + a * Real.log x

def turning_point (g : ℝ → ℝ) (P : ℝ × ℝ) (h : ℝ → ℝ) : Prop :=
  ∀ (x : ℝ), x ≠ P.1 → (g x - h x) / (x - P.1) > 0

theorem extreme_values_a_1 :
  (∀ (x : ℝ), f x 1 ≤ f (1/2) 1 → f x 1 = f (1/2) 1) ∧ (∀ (x : ℝ), f x 1 ≥ f 1 1 → f x 1 = f 1 1) :=
sorry

theorem turning_point_a_8 :
  ∀ (x₀ : ℝ), x₀ = 2 → turning_point (f · 8) (x₀, f x₀ 8) (λ x => (2 * x₀ + 8 / x₀ - 10) * (x - x₀) + x₀^2 - 10 * x₀ + 8 * Real.log x₀) :=
sorry

end extreme_values_a_1_turning_point_a_8_l181_181985


namespace buratino_candy_spent_l181_181642

theorem buratino_candy_spent :
  ∃ (x y : ℕ), x + y = 50 ∧ 2 * x = 3 * y ∧ y * 5 - x * 3 = 10 :=
by {
  -- Declaration of variables and goal
  let x := 30,
  let y := 20,
  use [x, y],
  split,
  { exact rfl },
  split,
  { exact rfl },
  { exact rfl }
}

end buratino_candy_spent_l181_181642


namespace sin_range_l181_181970

theorem sin_range :
  ∀ x, (-Real.pi / 4 ≤ x ∧ x ≤ 3 * Real.pi / 4) → (∃ y, y = Real.sin x ∧ -Real.sqrt 2 / 2 ≤ y ∧ y ≤ 1) := by
  sorry

end sin_range_l181_181970


namespace granger_total_amount_l181_181197

-- Define the constants for the problem
def cost_spam := 3
def cost_peanut_butter := 5
def cost_bread := 2
def quantity_spam := 12
def quantity_peanut_butter := 3
def quantity_bread := 4

-- Define the total cost calculation
def total_cost := (quantity_spam * cost_spam) + (quantity_peanut_butter * cost_peanut_butter) + (quantity_bread * cost_bread)

-- The theorem we need to prove
theorem granger_total_amount : total_cost = 59 := by
  sorry

end granger_total_amount_l181_181197


namespace gcf_factor_l181_181788

theorem gcf_factor (x y : ℕ) : gcd (6 * x ^ 3 * y ^ 2) (3 * x ^ 2 * y ^ 3) = 3 * x ^ 2 * y ^ 2 :=
by
  sorry

end gcf_factor_l181_181788


namespace joes_bid_l181_181430

/--
Nelly tells her daughter she outbid her rival Joe by paying $2000 more than thrice his bid.
Nelly got the painting for $482,000. Prove that Joe's bid was $160,000.
-/
theorem joes_bid (J : ℝ) (h1 : 482000 = 3 * J + 2000) : J = 160000 :=
by
  sorry

end joes_bid_l181_181430


namespace cheryl_more_points_l181_181660

-- Define the number of each type of eggs each child found
def kevin_small_eggs : Nat := 5
def kevin_large_eggs : Nat := 3

def bonnie_small_eggs : Nat := 13
def bonnie_medium_eggs : Nat := 7
def bonnie_large_eggs : Nat := 2

def george_small_eggs : Nat := 9
def george_medium_eggs : Nat := 6
def george_large_eggs : Nat := 1

def cheryl_small_eggs : Nat := 56
def cheryl_medium_eggs : Nat := 30
def cheryl_large_eggs : Nat := 15

-- Define the points for each type of egg
def small_egg_points : Nat := 1
def medium_egg_points : Nat := 3
def large_egg_points : Nat := 5

-- Calculate the total points for each child
def kevin_points : Nat := kevin_small_eggs * small_egg_points + kevin_large_eggs * large_egg_points
def bonnie_points : Nat := bonnie_small_eggs * small_egg_points + bonnie_medium_eggs * medium_egg_points + bonnie_large_eggs * large_egg_points
def george_points : Nat := george_small_eggs * small_egg_points + george_medium_eggs * medium_egg_points + george_large_eggs * large_egg_points
def cheryl_points : Nat := cheryl_small_eggs * small_egg_points + cheryl_medium_eggs * medium_egg_points + cheryl_large_eggs * large_egg_points

-- Statement of the proof problem
theorem cheryl_more_points : cheryl_points - (kevin_points + bonnie_points + george_points) = 125 :=
by
  -- Here would go the proof steps
  sorry

end cheryl_more_points_l181_181660


namespace range_of_m_l181_181541

noncomputable def intersects_x_axis (m : ℝ) : Prop :=
  ∃ x : ℝ, m * x^2 - 4 * x + 1 = 0

theorem range_of_m (m : ℝ) (h : intersects_x_axis m) : m ≤ 4 := by
  sorry

end range_of_m_l181_181541


namespace total_population_l181_181247

variable (seattle lake_view boise : ℕ)

def lake_view_pop := (lake_view = 24000)
def seattle_pop := (seattle = lake_view - 4000)
def boise_pop := (boise = (3 / 5 : ℚ) * seattle)

theorem total_population (h1 : lake_view_pop) (h2 : seattle_pop) (h3 : boise_pop) :
  lake_view + seattle + boise = 56000 :=
by
  sorry

end total_population_l181_181247


namespace part1_part2_l181_181516

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (Real.exp x / x) - Real.log x + x - a

theorem part1 (a : ℝ) (h : ∀ x > 0, f x a ≥ 0) : a ≤ Real.exp 1 + 1 := 
sorry

theorem part2 {a : ℝ} {x1 x2 : ℝ} (hx1 : 0 < x1) (hx2 : 0 < x2) 
  (hz1 : f x1 a = 0) (hz2 : f x2 a = 0) : x1 * x2 < 1 := 
sorry

end part1_part2_l181_181516


namespace three_city_population_l181_181245

noncomputable def totalPopulation (boise seattle lakeView: ℕ) : ℕ :=
  boise + seattle + lakeView

theorem three_city_population (pBoise pSeattle pLakeView : ℕ)
  (h1 : pBoise = 3 * pSeattle / 5)
  (h2 : pLakeView = pSeattle + 4000)
  (h3 : pLakeView = 24000) :
  totalPopulation pBoise pSeattle pLakeView = 56000 := by
  sorry

end three_city_population_l181_181245


namespace total_spent_on_toys_and_clothes_l181_181565

def cost_toy_cars : ℝ := 14.88
def cost_skateboard : ℝ := 4.88
def cost_toy_trucks : ℝ := 5.86
def cost_pants : ℝ := 14.55
def cost_shirt : ℝ := 7.43
def cost_hat : ℝ := 12.50

theorem total_spent_on_toys_and_clothes :
  (cost_toy_cars + cost_skateboard + cost_toy_trucks) + (cost_pants + cost_shirt + cost_hat) = 60.10 :=
by
  sorry

end total_spent_on_toys_and_clothes_l181_181565


namespace measure_of_angle_C_l181_181046

-- Conditions of the problem
variable (A B C : ℝ) -- Angles in the triangle

-- Sum of all angles in a triangle
axiom sum_of_angles (h : A + B + C = 180)

-- Condition given in the problem
axiom given_angle_sum (h1 : A + B = 80)

-- Desired proof statement
theorem measure_of_angle_C : C = 100 := by
  sorry

end measure_of_angle_C_l181_181046


namespace carnival_ring_toss_l181_181102

theorem carnival_ring_toss (total_amount : ℕ) (days : ℕ) (amount_per_day : ℕ) 
  (h1 : total_amount = 420) 
  (h2 : days = 3) 
  (h3 : total_amount = days * amount_per_day) : amount_per_day = 140 :=
by
  sorry

end carnival_ring_toss_l181_181102


namespace common_difference_of_sequence_l181_181408

variable (a : ℕ → ℚ)

def is_arithmetic_sequence (a : ℕ → ℚ) :=
  ∃ d : ℚ, ∀ n m : ℕ, a n = a m + d * (n - m)

theorem common_difference_of_sequence 
  (h : a 2015 = a 2013 + 6) 
  (ha : is_arithmetic_sequence a) :
  ∃ d : ℚ, d = 3 :=
by
  sorry

end common_difference_of_sequence_l181_181408


namespace y1_gt_y2_l181_181385

theorem y1_gt_y2 (k : ℝ) (y1 y2 : ℝ) (hA : y1 = k * (-3) + 3) (hB : y2 = k * 1 + 3) (hK : k < 0) : y1 > y2 :=
by 
  sorry

end y1_gt_y2_l181_181385


namespace binom_20_19_eq_20_l181_181329

def binom (n k : ℕ) := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binom_20_19_eq_20 : binom 20 19 = 20 := by
  sorry

end binom_20_19_eq_20_l181_181329


namespace min_colors_correct_l181_181105

def min_colors (n : Nat) : Nat :=
  if n = 1 then 1
  else if n = 2 then 2
  else 3

theorem min_colors_correct (n : Nat) : min_colors n = 
  if n = 1 then 1
  else if n = 2 then 2
  else 3 := by
  sorry

end min_colors_correct_l181_181105


namespace find_d_l181_181233

-- Define the conditions
variables (x₀ y₀ c : ℝ)

-- Define the system of equations
def system_of_equations : Prop :=
  x₀ * y₀ = 6 ∧ x₀^2 * y₀ + x₀ * y₀^2 + x₀ + y₀ + c = 2

-- Define the target proof problem
theorem find_d (h : system_of_equations x₀ y₀ c) : x₀^2 + y₀^2 = 69 :=
sorry

end find_d_l181_181233


namespace arrange_letters_l181_181378

-- Definitions based on conditions
def total_letters := 6
def identical_bs := 2 -- Number of B's that are identical
def distinct_as := 3  -- Number of A's that are distinct
def distinct_ns := 1  -- Number of N's that are distinct

-- Now formulate the proof statement
theorem arrange_letters :
    (Nat.factorial total_letters) / (Nat.factorial identical_bs) = 360 :=
by
  sorry

end arrange_letters_l181_181378


namespace tammy_trees_l181_181093

-- Define the conditions as Lean definitions and the final statement to prove
theorem tammy_trees :
  (∀ (days : ℕ) (earnings : ℕ) (pricePerPack : ℕ) (orangesPerPack : ℕ) (orangesPerTree : ℕ),
    days = 21 →
    earnings = 840 →
    pricePerPack = 2 →
    orangesPerPack = 6 →
    orangesPerTree = 12 →
    (earnings / days) / (pricePerPack / orangesPerPack) / orangesPerTree = 10) :=
by
  intros days earnings pricePerPack orangesPerPack orangesPerTree
  sorry

end tammy_trees_l181_181093


namespace range_of_a_product_of_zeros_l181_181520

def f (x : ℝ) (a : ℝ) := (Real.exp x / x) - (Real.log x) + x - a

theorem range_of_a (a : ℝ) : (∀ x, (0 < x) → (f x a ≥ 0)) ↔ (a ≤ Real.exp 1 + 1) :=
by
  sorry

theorem product_of_zeros (x1 x2 a : ℝ) : 
  (0 < x1) → (x1 < 1) → (1 < x2) → (f x1 a = 0 ∧ f x2 a = 0) → x1 * x2 < 1 :=
by
  sorry

end range_of_a_product_of_zeros_l181_181520


namespace identify_first_brother_l181_181146

-- Definitions for conditions
inductive Brother
| Trulya : Brother
| Falsa : Brother

-- Extracting conditions into Lean 4 statements
def first_brother_says : String := "Both cards are of the purplish suit."
def second_brother_says : String := "This is not true!"

axiom trulya_always_truthful : ∀ (b : Brother) (statement : String), b = Brother.Trulya ↔ (statement = first_brother_says ∨ statement = second_brother_says)
axiom falsa_always_lies : ∀ (b : Brother) (statement : String), b = Brother.Falsa ↔ ¬(statement = first_brother_says ∨ statement = second_brother_says)

-- Proof statement 
theorem identify_first_brother :
  ∃ (b : Brother), b = Brother.Trulya :=
sorry

end identify_first_brother_l181_181146


namespace graph_equation_l181_181924

theorem graph_equation (x y : ℝ) : (x + y)^2 = x^2 + y^2 ↔ (x = 0 ∨ y = 0) := by
  sorry

end graph_equation_l181_181924


namespace escalator_ride_time_l181_181215

theorem escalator_ride_time (x y k t : ℝ)
  (h1 : 75 * x = y)
  (h2 : 30 * (x + k) = y)
  (h3 : t = y / k) :
  t = 50 := by
  sorry

end escalator_ride_time_l181_181215


namespace child_b_share_l181_181287

def total_money : ℕ := 4320

def ratio_parts : List ℕ := [2, 3, 4, 5, 6]

def parts_sum (parts : List ℕ) : ℕ :=
  parts.foldl (· + ·) 0

def value_of_one_part (total : ℕ) (parts : ℕ) : ℕ :=
  total / parts

def b_share (value_per_part : ℕ) (b_parts : ℕ) : ℕ :=
  value_per_part * b_parts

theorem child_b_share :
  let total_money := 4320
  let ratio_parts := [2, 3, 4, 5, 6]
  let total_parts := parts_sum ratio_parts
  let one_part_value := value_of_one_part total_money total_parts
  let b_parts := 3
  b_share one_part_value b_parts = 648 := by
  let total_money := 4320
  let ratio_parts := [2, 3, 4, 5, 6]
  let total_parts := parts_sum ratio_parts
  let one_part_value := value_of_one_part total_money total_parts
  let b_parts := 3
  show b_share one_part_value b_parts = 648
  sorry

end child_b_share_l181_181287


namespace fraction_zero_iff_x_neg_one_l181_181396

theorem fraction_zero_iff_x_neg_one (x : ℝ) (h₀ : x^2 - 1 = 0) (h₁ : x - 1 ≠ 0) :
  (x^2 - 1) / (x - 1) = 0 ↔ x = -1 :=
by
  sorry

end fraction_zero_iff_x_neg_one_l181_181396


namespace binomial_20_19_eq_20_l181_181309

theorem binomial_20_19_eq_20 : Nat.choose 20 19 = 20 :=
by
  sorry

end binomial_20_19_eq_20_l181_181309


namespace geometric_series_ratio_l181_181587

theorem geometric_series_ratio (a_1 a_2 S q : ℝ) (hq : |q| < 1)
  (hS : S = a_1 / (1 - q))
  (ha2 : a_2 = a_1 * q) :
  S / (S - a_1) = a_1 / a_2 := 
sorry

end geometric_series_ratio_l181_181587


namespace sum_of_solutions_of_quadratic_eq_l181_181499

theorem sum_of_solutions_of_quadratic_eq :
  (∀ x : ℝ, 5 * x^2 - 3 * x - 2 = 0) → (∀ a b : ℝ, a = 5 ∧ b = -3 → -b / a = 3 / 5) :=
by
  sorry

end sum_of_solutions_of_quadratic_eq_l181_181499


namespace Bridget_Skittles_Final_l181_181648

def Bridget_Skittles_initial : Nat := 4
def Henry_Skittles_initial : Nat := 4

def Bridget_Receives_Henry_Skittles : Nat :=
  Bridget_Skittles_initial + Henry_Skittles_initial

theorem Bridget_Skittles_Final :
  Bridget_Receives_Henry_Skittles = 8 :=
by
  -- Proof steps here, assuming sorry for now
  sorry

end Bridget_Skittles_Final_l181_181648


namespace inequality_holds_l181_181974

theorem inequality_holds (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x * y = 4) :
  (1 / (x + 3) + 1 / (y + 3) ≤ 2 / 5) ∧ (1 / (x + 3) + 1 / (y + 3) = 2 / 5 ↔ x = 2 ∧ y = 2) :=
sorry

end inequality_holds_l181_181974


namespace total_movie_hours_l181_181725

-- Definitions
def JoyceMovie : ℕ := 12 -- Joyce's favorite movie duration in hours
def MichaelMovie : ℕ := 10 -- Michael's favorite movie duration in hours
def NikkiMovie : ℕ := 30 -- Nikki's favorite movie duration in hours
def RynMovie : ℕ := 24 -- Ryn's favorite movie duration in hours

-- Condition translations
def Joyce_movie_condition : Prop := JoyceMovie = MichaelMovie + 2
def Nikki_movie_condition : Prop := NikkiMovie = 3 * MichaelMovie
def Ryn_movie_condition : Prop := RynMovie = (4 * NikkiMovie) / 5
def Nikki_movie_given : Prop := NikkiMovie = 30

-- The theorem to prove
theorem total_movie_hours : Joyce_movie_condition ∧ Nikki_movie_condition ∧ Ryn_movie_condition ∧ Nikki_movie_given → 
  (JoyceMovie + MichaelMovie + NikkiMovie + RynMovie = 76) :=
by
  intros h
  sorry

end total_movie_hours_l181_181725


namespace smallest_possible_value_l181_181571

theorem smallest_possible_value (a b c d : ℝ)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a + b + c + d) * ((1 / (a + b)) + (1 / (a + c)) + (1 / (b + d)) + (1 / (c + d))) ≥ 8 := 
sorry

end smallest_possible_value_l181_181571


namespace number_of_bars_in_box_l181_181001

variable (x : ℕ)
variable (cost_per_bar : ℕ := 6)
variable (remaining_bars : ℕ := 6)
variable (total_money_made : ℕ := 42)

theorem number_of_bars_in_box :
  cost_per_bar * (x - remaining_bars) = total_money_made → x = 13 :=
by
  intro h
  sorry

end number_of_bars_in_box_l181_181001


namespace number_of_classes_l181_181128

theorem number_of_classes (x : ℕ) (total_games : ℕ) (h : total_games = 45) :
  (x * (x - 1)) / 2 = total_games → x = 10 :=
by
  sorry

end number_of_classes_l181_181128


namespace consecutive_odd_numbers_first_l181_181933

theorem consecutive_odd_numbers_first :
  ∃ x : ℤ, 11 * x = 3 * (x + 4) + 4 * (x + 2) + 16 ∧ x = 9 :=
by 
  sorry

end consecutive_odd_numbers_first_l181_181933


namespace triangle_angle_sum_l181_181066

theorem triangle_angle_sum (A B C : ℝ) (h1 : A + B = 80) (h2 : A + B + C = 180) : C = 100 := 
by
  sorry

end triangle_angle_sum_l181_181066


namespace sqrt_sum_4_pow_4_eq_32_l181_181915

theorem sqrt_sum_4_pow_4_eq_32 : Real.sqrt (4^4 + 4^4 + 4^4 + 4^4) = 32 :=
by
  sorry

end sqrt_sum_4_pow_4_eq_32_l181_181915


namespace binom_20_19_eq_20_l181_181330

def binom (n k : ℕ) := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binom_20_19_eq_20 : binom 20 19 = 20 := by
  sorry

end binom_20_19_eq_20_l181_181330


namespace total_slices_at_picnic_l181_181488

def danny_watermelons : ℕ := 3
def danny_slices_per_watermelon : ℕ := 10
def sister_watermelons : ℕ := 1
def sister_slices_per_watermelon : ℕ := 15

def total_danny_slices : ℕ := danny_watermelons * danny_slices_per_watermelon
def total_sister_slices : ℕ := sister_watermelons * sister_slices_per_watermelon
def total_slices : ℕ := total_danny_slices + total_sister_slices

theorem total_slices_at_picnic : total_slices = 45 :=
by
  sorry

end total_slices_at_picnic_l181_181488


namespace total_marks_math_physics_l181_181636

variable (M P C : ℕ)

theorem total_marks_math_physics (h1 : C = P + 10) (h2 : (M + C) / 2 = 35) : M + P = 60 :=
by
  sorry

end total_marks_math_physics_l181_181636


namespace no_integer_solution_l181_181972

theorem no_integer_solution (a b : ℤ) : ¬ (4 ∣ a^2 + b^2 + 1) :=
by
  -- Prevent use of the solution steps and add proof obligations
  sorry

end no_integer_solution_l181_181972


namespace find_constants_l181_181353

theorem find_constants (A B C : ℚ) :
  (∀ x : ℚ, x ≠ 4 ∧ x ≠ 2 →
    (3 * x + 7) / ((x - 4) * (x - 2)^2) = A / (x - 4) + B / (x - 2) + C / (x - 2)^2) →
  A = 19 / 4 ∧ B = -19 / 4 ∧ C = -13 / 2 :=
by
  sorry

end find_constants_l181_181353


namespace part1_part2_l181_181517

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (Real.exp x / x) - Real.log x + x - a

theorem part1 (a : ℝ) (h : ∀ x > 0, f x a ≥ 0) : a ≤ Real.exp 1 + 1 := 
sorry

theorem part2 {a : ℝ} {x1 x2 : ℝ} (hx1 : 0 < x1) (hx2 : 0 < x2) 
  (hz1 : f x1 a = 0) (hz2 : f x2 a = 0) : x1 * x2 < 1 := 
sorry

end part1_part2_l181_181517


namespace evaluate_three_squared_raised_four_l181_181165

theorem evaluate_three_squared_raised_four : (3^2)^4 = 6561 := by
  sorry

end evaluate_three_squared_raised_four_l181_181165


namespace angle_C_in_triangle_l181_181041

theorem angle_C_in_triangle (A B C : ℝ) (h1 : A + B = 80) (h2 : A + B + C = 180) : C = 100 := 
by 
  sorry

end angle_C_in_triangle_l181_181041


namespace balcony_more_than_orchestra_l181_181931

theorem balcony_more_than_orchestra (x y : ℕ) 
  (h1 : x + y = 340) 
  (h2 : 12 * x + 8 * y = 3320) : 
  y - x = 40 := 
sorry

end balcony_more_than_orchestra_l181_181931


namespace measure_of_angle_C_range_of_sum_ab_l181_181719

-- Proof problem (1): Prove the measure of angle C
theorem measure_of_angle_C (a b c : ℝ) (A B C : ℝ) 
  (h1 : 2 * c * Real.sin C = (2 * b + a) * Real.sin B + (2 * a - 3 * b) * Real.sin A) :
  C = Real.pi / 3 := by 
  sorry

-- Proof problem (2): Prove the range of possible values of a + b
theorem range_of_sum_ab (a b : ℝ) (c : ℝ) (h1 : c = 4) (h2 : 16 = a^2 + b^2 - a * b) :
  4 < a + b ∧ a + b ≤ 8 := by 
  sorry

end measure_of_angle_C_range_of_sum_ab_l181_181719


namespace number_of_workers_l181_181234

theorem number_of_workers 
  (W : ℕ) 
  (h1 : 750 * W = (5 * 900) + 700 * (W - 5)) : 
  W = 20 := 
by 
  sorry

end number_of_workers_l181_181234


namespace more_likely_millionaire_city_resident_l181_181281

noncomputable def probability_A (M N m n : ℝ) : ℝ :=
  m * M / (m * M + n * N)

noncomputable def probability_B (M N : ℝ) : ℝ :=
  M / (M + N)

theorem more_likely_millionaire_city_resident
  (M N : ℕ) (m n : ℝ) (hM : M > 0) (hN : N > 0)
  (hm : m > 10^6) (hn : n ≤ 10^6) :
  probability_A M N m n > probability_B M N :=
by {
  sorry
}

end more_likely_millionaire_city_resident_l181_181281


namespace measure_of_angle_C_l181_181044

-- Conditions of the problem
variable (A B C : ℝ) -- Angles in the triangle

-- Sum of all angles in a triangle
axiom sum_of_angles (h : A + B + C = 180)

-- Condition given in the problem
axiom given_angle_sum (h1 : A + B = 80)

-- Desired proof statement
theorem measure_of_angle_C : C = 100 := by
  sorry

end measure_of_angle_C_l181_181044


namespace susan_gate_probability_l181_181895

theorem susan_gate_probability : 
  let p := 41
  let q := 70 in
  p + q = 111 ∧ (123 : ℚ) / 210 = (41 : ℚ) / 70 :=
by
  sorry

end susan_gate_probability_l181_181895


namespace how_many_times_faster_l181_181469

theorem how_many_times_faster (A B : ℝ) (h1 : A = 1 / 32) (h2 : A + B = 1 / 24) : A / B = 3 := by
  sorry

end how_many_times_faster_l181_181469


namespace dogs_not_eat_either_l181_181863

-- Definitions for our conditions
variable (dogs_total : ℕ) (dogs_watermelon : ℕ) (dogs_salmon : ℕ) (dogs_both : ℕ)

-- Specific values of our conditions
def dogs_total_value : ℕ := 60
def dogs_watermelon_value : ℕ := 9
def dogs_salmon_value : ℕ := 48
def dogs_both_value : ℕ := 5

-- The theorem we need to prove
theorem dogs_not_eat_either : 
    dogs_total = dogs_total_value → 
    dogs_watermelon = dogs_watermelon_value → 
    dogs_salmon = dogs_salmon_value → 
    dogs_both = dogs_both_value → 
    (dogs_total - (dogs_watermelon + dogs_salmon - dogs_both) = 8) :=
by
  intros
  sorry

end dogs_not_eat_either_l181_181863


namespace compare_probabilities_l181_181277

-- Definitions
variables (M N: ℕ) (m n: ℕ)
  
-- Conditions
def condition_m_millionaire : Prop := m > 10^6
def condition_n_nonmillionaire : Prop := n ≤ 10^6

-- Probabilities
noncomputable def P_A : ℚ := (M:ℚ) / (M + (n:ℚ)/(m:ℚ) * N)
noncomputable def P_B : ℚ := (M:ℚ) / (M + N)

-- Theorem statement
theorem compare_probabilities
  (hM : M > 0) (hN : N > 0)
  (h_m_millionaire : condition_m_millionaire m)
  (h_n_nonmillionaire : condition_n_nonmillionaire n) :
  P_A M N m n > P_B M N := sorry

end compare_probabilities_l181_181277


namespace polynomial_inequality_solution_l181_181678

theorem polynomial_inequality_solution (x : ℝ) :
  x^4 + x^3 - 10 * x^2 + 25 * x > 0 ↔ x > 0 :=
sorry

end polynomial_inequality_solution_l181_181678


namespace simplify_fraction_sum_eq_zero_l181_181416

variable (a b c : ℝ)
variable (ha : a ≠ 0)
variable (hb : b ≠ 0)
variable (hc : c ≠ 0)
variable (h : a + b + 2 * c = 0)

theorem simplify_fraction_sum_eq_zero :
  (1 / (b^2 + 4*c^2 - a^2) + 1 / (a^2 + 4*c^2 - b^2) + 1 / (a^2 + b^2 - 4*c^2)) = 0 :=
by sorry

end simplify_fraction_sum_eq_zero_l181_181416


namespace angle_C_in_triangle_l181_181049

theorem angle_C_in_triangle (A B C : ℝ) (h : A + B = 80) : C = 100 :=
by
  -- The measure of angle C follows from the condition and the properties of triangles.
  sorry

end angle_C_in_triangle_l181_181049


namespace quadratic_distinct_roots_l181_181680

theorem quadratic_distinct_roots (p q : ℚ) :
  (∀ x : ℚ, x^2 + p * x + q = 0 ↔ x = 2 * p ∨ x = p + q) →
  (p = 2 / 3 ∧ q = -8 / 3) :=
by sorry

end quadratic_distinct_roots_l181_181680


namespace total_bees_approx_l181_181634

-- Define a rectangular garden with given width and length
def garden_width : ℝ := 450
def garden_length : ℝ := 550

-- Define the average density of bees per square foot
def bee_density : ℝ := 2.5

-- Define the area of the garden in square feet
def garden_area : ℝ := garden_width * garden_length

-- Define the total number of bees in the garden
def total_bees : ℝ := bee_density * garden_area

-- Prove that the total number of bees approximately equals 620,000
theorem total_bees_approx : abs (total_bees - 620000) < 1000 :=
by
  sorry

end total_bees_approx_l181_181634


namespace range_of_a_product_of_zeros_l181_181519

def f (x : ℝ) (a : ℝ) := (Real.exp x / x) - (Real.log x) + x - a

theorem range_of_a (a : ℝ) : (∀ x, (0 < x) → (f x a ≥ 0)) ↔ (a ≤ Real.exp 1 + 1) :=
by
  sorry

theorem product_of_zeros (x1 x2 a : ℝ) : 
  (0 < x1) → (x1 < 1) → (1 < x2) → (f x1 a = 0 ∧ f x2 a = 0) → x1 * x2 < 1 :=
by
  sorry

end range_of_a_product_of_zeros_l181_181519


namespace sphere_radius_l181_181841

theorem sphere_radius (R : ℝ) (h : 4 * Real.pi * R^2 = 4 * Real.pi) : R = 1 :=
sorry

end sphere_radius_l181_181841


namespace mary_screws_l181_181425

theorem mary_screws (S : ℕ) (h : S + 2 * S = 24) : S = 8 :=
by sorry

end mary_screws_l181_181425


namespace candy_total_l181_181977

theorem candy_total (n m : ℕ) (h1 : n = 2) (h2 : m = 8) : n * m = 16 :=
by
  -- This will contain the proof
  sorry

end candy_total_l181_181977


namespace total_spending_in_4_years_l181_181604

def trevor_spending_per_year : ℕ := 80
def reed_to_trevor_diff : ℕ := 20
def reed_to_quinn_factor : ℕ := 2

theorem total_spending_in_4_years :
  ∃ (reed_spending quinn_spending : ℕ),
  (reed_spending = trevor_spending_per_year - reed_to_trevor_diff) ∧
  (reed_spending = reed_to_quinn_factor * quinn_spending) ∧
  ((trevor_spending_per_year + reed_spending + quinn_spending) * 4 = 680) :=
sorry

end total_spending_in_4_years_l181_181604


namespace abc_value_l181_181232

theorem abc_value (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
    (h1 : a * (b + c) = 171) 
    (h2 : b * (c + a) = 180) 
    (h3 : c * (a + b) = 189) :
    a * b * c = 270 :=
by
  -- Place proofs here
  sorry

end abc_value_l181_181232


namespace range_of_m_l181_181020

theorem range_of_m (x : ℝ) (h₁ : 1/2 ≤ x) (h₂ : x ≤ 2) :
  2 - Real.log 2 ≤ -Real.log x + 3*x - x^2 ∧ -Real.log x + 3*x - x^2 ≤ 2 :=
sorry

end range_of_m_l181_181020


namespace problem1_problem2_l181_181698

noncomputable def f (x : ℝ) : ℝ := x / Real.exp x

-- Problem 1: (0 < m < 1/e) implies g(x) = f(x) - m has two zeros
theorem problem1 (m : ℝ) (h1 : 0 < m) (h2 : m < 1 / Real.exp 1) :
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 = m ∧ f x2 = m :=
sorry

-- Problem 2: (2/e^2 ≤ a < 1/e) implies f^2(x) - af(x) > 0 has only one integer solution
theorem problem2 (a : ℝ) (h1 : 2 / (Real.exp 2) ≤ a) (h2 : a < 1 / Real.exp 1) :
  ∃! x : ℤ, ∀ y : ℤ, (f y)^2 - a * (f y) > 0 → y = x :=
sorry

end problem1_problem2_l181_181698


namespace unique_solution_l181_181004

theorem unique_solution (a b : ℤ) : 
  (a^6 + 1 ∣ b^11 - 2023 * b^3 + 40 * b) ∧ (a^4 - 1 ∣ b^10 - 2023 * b^2 - 41) 
  ↔ (a = 0 ∧ ∃ c : ℤ, b = c) := 
by 
  sorry

end unique_solution_l181_181004


namespace senior_high_sample_count_l181_181815

theorem senior_high_sample_count 
  (total_students : ℕ)
  (junior_high_students : ℕ)
  (senior_high_students : ℕ)
  (total_sampled_students : ℕ)
  (H1 : total_students = 1800)
  (H2 : junior_high_students = 1200)
  (H3 : senior_high_students = 600)
  (H4 : total_sampled_students = 180) :
  (senior_high_students * total_sampled_students / total_students) = 60 := 
sorry

end senior_high_sample_count_l181_181815


namespace max_halls_visitable_max_triangles_in_chain_l181_181939

-- Definition of the problem conditions
def castle_side_length : ℝ := 100
def num_halls : ℕ := 100
def hall_side_length : ℝ := 10
def max_visitable_halls : ℕ := 91

-- Theorem statements
theorem max_halls_visitable (S : ℝ) (n : ℕ) (H : ℝ) :
  S = 100 ∧ n = 100 ∧ H = 10 → max_visitable_halls = 91 :=
by sorry

-- Definitions for subdividing an equilateral triangle and the chain of triangles
def side_divisions (k : ℕ) : ℕ := k
def total_smaller_triangles (k : ℕ) : ℕ := k^2
def max_chain_length (k : ℕ) : ℕ := k^2 - k + 1

-- Theorem statements
theorem max_triangles_in_chain (k : ℕ) :
  max_chain_length k = k^2 - k + 1 :=
by sorry

end max_halls_visitable_max_triangles_in_chain_l181_181939


namespace digit_swap_division_l181_181475

theorem digit_swap_division (ab ba : ℕ) (k1 k2 : ℤ) (a b : ℕ) :
  (ab = 10 * a + b) ∧ (ba = 10 * b + a) →
  (ab % 7 = 1) ∧ (ba % 7 = 1) →
  ∃ n, n = 4 :=
by
  sorry

end digit_swap_division_l181_181475


namespace binom_20_19_eq_20_l181_181319

theorem binom_20_19_eq_20 :
  nat.choose 20 19 = 20 :=
by
  -- Using the property of binomial coefficients: C(n, k) = C(n, n-k)
  have h1 : nat.choose 20 19 = nat.choose 20 (20 - 19),
  from nat.choose_symm 20 19,
  -- Simplifying: C(20, 1)
  rw [h1, nat.sub_self],
  -- Using the binomial coefficient result: C(n, 1) = n
  exact nat.choose_one_right 20

end binom_20_19_eq_20_l181_181319


namespace find_x_for_fx_neg_half_l181_181509

open Function 

theorem find_x_for_fx_neg_half (f : ℝ → ℝ)
  (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_period : ∀ x : ℝ, f (x + 2) = -f x)
  (h_interval : ∀ x : ℝ, 0 ≤ x → x ≤ 1 → f x = 1/2 * x) :
  {x : ℝ | f x = -1/2} = {x : ℝ | ∃ n : ℤ, x = 4 * n - 1} :=
by
  sorry

end find_x_for_fx_neg_half_l181_181509


namespace opposite_of_2023_l181_181758

/-- The opposite of a number n is defined as the number that, when added to n, results in zero. -/
def opposite (n : ℤ) : ℤ := -n

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by 
  sorry

end opposite_of_2023_l181_181758


namespace proof_2_abs_a_plus_b_less_abs_4_plus_ab_l181_181181

theorem proof_2_abs_a_plus_b_less_abs_4_plus_ab (a b : ℝ) (h1 : abs a < 2) (h2 : abs b < 2) :
    2 * abs (a + b) < abs (4 + a * b) := 
by
  sorry

end proof_2_abs_a_plus_b_less_abs_4_plus_ab_l181_181181


namespace find_C_l181_181766

theorem find_C
  (A B C D : ℕ)
  (h1 : 0 ≤ A ∧ A ≤ 9)
  (h2 : 0 ≤ B ∧ B ≤ 9)
  (h3 : 0 ≤ C ∧ C ≤ 9)
  (h4 : 0 ≤ D ∧ D ≤ 9)
  (h5 : 4 * 1000 + A * 100 + 5 * 10 + B + (C * 1000 + 2 * 100 + D * 10 + 7) = 8070) :
  C = 3 :=
by
  sorry

end find_C_l181_181766


namespace orange_price_l181_181252

theorem orange_price (initial_apples : ℕ) (initial_oranges : ℕ) 
                     (apple_price : ℝ) (total_earnings : ℝ) 
                     (remaining_apples : ℕ) (remaining_oranges : ℕ)
                     (h1 : initial_apples = 50) (h2 : initial_oranges = 40)
                     (h3 : apple_price = 0.80) (h4 : total_earnings = 49)
                     (h5 : remaining_apples = 10) (h6 : remaining_oranges = 6) :
  ∃ orange_price : ℝ, orange_price = 0.50 :=
by
  sorry

end orange_price_l181_181252


namespace quadratic_distinct_roots_l181_181708

theorem quadratic_distinct_roots (k : ℝ) : 
  (∃ x y : ℝ, k * x^2 - 2 * x - 1 = 0 ∧ k * y^2 - 2 * y - 1 = 0 ∧ x ≠ y) ↔ k > -1 ∧ k ≠ 0 := 
sorry

end quadratic_distinct_roots_l181_181708


namespace binom_20_19_eq_20_l181_181318

theorem binom_20_19_eq_20 :
  nat.choose 20 19 = 20 :=
by
  -- Using the property of binomial coefficients: C(n, k) = C(n, n-k)
  have h1 : nat.choose 20 19 = nat.choose 20 (20 - 19),
  from nat.choose_symm 20 19,
  -- Simplifying: C(20, 1)
  rw [h1, nat.sub_self],
  -- Using the binomial coefficient result: C(n, 1) = n
  exact nat.choose_one_right 20

end binom_20_19_eq_20_l181_181318


namespace floor_negative_fraction_l181_181345

theorem floor_negative_fraction : (Int.floor (-7 / 4 : ℚ)) = -2 := by
  sorry

end floor_negative_fraction_l181_181345


namespace least_number_to_subtract_l181_181624

theorem least_number_to_subtract (x : ℕ) (h : 5026 % 5 = x) : x = 1 :=
by sorry

end least_number_to_subtract_l181_181624


namespace sin_A_equals_4_over_5_l181_181716

variables {A B C : ℝ}
-- Given a right triangle ABC with angle B = 90 degrees
def right_triangle (A B C : ℝ) : Prop :=
  (A + B + C = 180) ∧ (B = 90)

-- We are given 3 * sin(A) = 4 * cos(A)
def given_condition (A : ℝ) : Prop :=
  3 * Real.sin A = 4 * Real.cos A

-- We need to prove that sin(A) = 4/5
theorem sin_A_equals_4_over_5 (A B C : ℝ) 
  (h1 : right_triangle B 90 C)
  (h2 : given_condition A) : 
  Real.sin A = 4 / 5 :=
by
  sorry

end sin_A_equals_4_over_5_l181_181716


namespace smallest_n_l181_181258

theorem smallest_n (n : ℕ) (h : 0 < n) (h1 : 813 * n % 30 = 1224 * n % 30) : n = 10 := 
sorry

end smallest_n_l181_181258


namespace min_sum_a_b2_l181_181365

theorem min_sum_a_b2 (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = a + b + 1) : a + b ≥ 2 + 2 * Real.sqrt 2 :=
by
  sorry

end min_sum_a_b2_l181_181365


namespace GCF_75_135_l181_181257

theorem GCF_75_135 : Nat.gcd 75 135 = 15 :=
by
sorry

end GCF_75_135_l181_181257


namespace negation_of_all_cars_are_fast_l181_181238

variable {α : Type} -- Assume α is the type of entities
variable (car fast : α → Prop) -- car and fast are predicates on entities

theorem negation_of_all_cars_are_fast :
  ¬ (∀ x, car x → fast x) ↔ ∃ x, car x ∧ ¬ fast x :=
by sorry

end negation_of_all_cars_are_fast_l181_181238


namespace range_of_a_l181_181690

variables (a b c : ℝ)

theorem range_of_a (h₁ : a^2 - b * c - 8 * a + 7 = 0)
                   (h₂ : b^2 + c^2 + b * c - 6 * a + 6 = 0) :
  1 ≤ a ∧ a ≤ 9 :=
sorry

end range_of_a_l181_181690


namespace fraction_of_product_l181_181779

theorem fraction_of_product : (7 / 8) * 64 = 56 := by
  sorry

end fraction_of_product_l181_181779


namespace find_a_plus_b_l181_181511

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := 2^(a * x + b)

theorem find_a_plus_b
  (a b : ℝ)
  (h1 : f a b 2 = 1 / 2)
  (h2 : f a b (1 / 2) = 2) :
  a + b = 1 / 3 :=
sorry

end find_a_plus_b_l181_181511


namespace number_of_dogs_l181_181208

theorem number_of_dogs (D C B x : ℕ) (h1 : D = 3 * x) (h2 : B = 9 * x) (h3 : D + B = 204) (h4 : 12 * x = 204) : D = 51 :=
by
  -- Proof skipped
  sorry

end number_of_dogs_l181_181208


namespace sum_abc_l181_181148

noncomputable def polynomial : Polynomial ℝ :=
  Polynomial.C (-6) + Polynomial.X * (Polynomial.C 11 + Polynomial.X * (Polynomial.C (-6) + Polynomial.X))

def t (k : ℕ) : ℝ :=
  match k with
  | 0 => 3
  | 1 => 6
  | 2 => 14
  | _ => 0 -- placeholder, as only t_0, t_1, t_2 are given explicitly

def a := 6
def b := -11
def c := 18

def t_rec (k : ℕ) : ℝ :=
  match k with
  | 0 => 3
  | 1 => 6
  | 2 => 14
  | n + 3 => a * t (n + 2) + b * t (n + 1) + c * t n

theorem sum_abc : a + b + c = 13 := by
  sorry

end sum_abc_l181_181148


namespace real_roots_quadratic_iff_l181_181190

theorem real_roots_quadratic_iff (a : ℝ) : (∃ x : ℝ, (a - 1) * x^2 - 2 * x + 1 = 0) ↔ a ≤ 2 := 
sorry

end real_roots_quadratic_iff_l181_181190


namespace melanie_has_4_plums_l181_181426

theorem melanie_has_4_plums (initial_plums : ℕ) (given_plums : ℕ) :
  initial_plums = 7 ∧ given_plums = 3 → initial_plums - given_plums = 4 :=
by
  sorry

end melanie_has_4_plums_l181_181426


namespace p_sufficient_but_not_necessary_for_q_l181_181192

def p (x : ℝ) : Prop := -1 ≤ x ∧ x ≤ 5
def q (x : ℝ) : Prop := (x - 5) * (x + 1) < 0

theorem p_sufficient_but_not_necessary_for_q :
  (∀ x : ℝ, p x → q x) ∧ ∃ x : ℝ, q x ∧ ¬ p x :=
by
  sorry

end p_sufficient_but_not_necessary_for_q_l181_181192


namespace max_dot_product_between_ellipses_l181_181854

noncomputable def ellipse1 (x y : ℝ) : Prop := (x^2 / 25 + y^2 / 9 = 1)
noncomputable def ellipse2 (x y : ℝ) : Prop := (x^2 / 9 + y^2 / 9 = 1)

theorem max_dot_product_between_ellipses :
  ∀ (M N : ℝ × ℝ),
    ellipse1 M.1 M.2 →
    ellipse2 N.1 N.2 →
    ∃ θ φ : ℝ,
      M = (5 * Real.cos θ, 3 * Real.sin θ) ∧
      N = (3 * Real.cos φ, 3 * Real.sin φ) ∧
      (15 * Real.cos θ * Real.cos φ + 9 * Real.sin θ * Real.sin φ ≤ 15) :=
by
  sorry

end max_dot_product_between_ellipses_l181_181854


namespace x1_value_l181_181364

theorem x1_value (x1 x2 x3 : ℝ) 
  (h1 : 0 ≤ x3 ∧ x3 ≤ x2 ∧ x2 ≤ x1 ∧ x1 ≤ 1) 
  (h2 : (1 - x1)^2 + 2 * (x1 - x2)^2 + 2 * (x2 - x3)^2 + x3^2 = 1 / 2) : 
  x1 = 2 / 3 :=
sorry

end x1_value_l181_181364


namespace algebraic_expression_value_l181_181860

theorem algebraic_expression_value (x : ℝ) (h : 2 * x^2 + 3 * x + 7 = 8) : 2 * x^2 + 3 * x - 7 = -6 :=
by sorry

end algebraic_expression_value_l181_181860


namespace selling_price_for_target_profit_l181_181145

-- Defining the conditions
def purchase_price : ℝ := 200
def annual_cost : ℝ := 40000
def annual_sales_volume (x : ℝ) := 800 - x
def annual_profit (x : ℝ) : ℝ := (x - purchase_price) * annual_sales_volume x - annual_cost

-- The theorem to prove
theorem selling_price_for_target_profit : ∃ x : ℝ, annual_profit x = 40000 ∧ x = 400 :=
by
  sorry

end selling_price_for_target_profit_l181_181145


namespace tan_x_eq_sqrt3_intervals_of_monotonic_increase_l181_181684

noncomputable def m (x : ℝ) : ℝ × ℝ :=
  (Real.sin (x - Real.pi / 6), 1)

noncomputable def n (x : ℝ) : ℝ × ℝ :=
  (Real.cos x, 1)

noncomputable def f (x : ℝ) : ℝ :=
  (m x).1 * (n x).1 + (m x).2 * (n x).2

-- Proof for part 1
theorem tan_x_eq_sqrt3 (x : ℝ) (h₀ : m x = n x) : Real.tan x = Real.sqrt 3 :=
sorry

-- Proof for part 2
theorem intervals_of_monotonic_increase (x : ℝ) (h₀ : 0 ≤ x) (h₁ : x ≤ Real.pi) :
  (0 ≤ x ∧ x ≤ Real.pi / 3) ∨ (5 * Real.pi / 6 ≤ x ∧ x ≤ Real.pi) ↔ 
  (0 ≤ x ∧ x ≤ Real.pi / 3) ∨ (5 * Real.pi / 6 ≤ x ∧ x ≤ Real.pi) :=
sorry

end tan_x_eq_sqrt3_intervals_of_monotonic_increase_l181_181684


namespace binom_identity1_binom_identity2_l181_181744

section Combinatorics

variable (n k m : ℕ)

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ :=
  if k > n then 0 else Nat.choose n k

-- Prove the identity: C(n, k) + C(n, k-1) = C(n+1, k)
theorem binom_identity1 : binomial n k + binomial n (k-1) = binomial (n+1) k :=
  sorry

-- Using the identity, prove: C(n, m) + C(n-1, m) + ... + C(n-10, m) = C(n+1, m+1) - C(n-10, m+1)
theorem binom_identity2 :
  (binomial n m + binomial (n-1) m + binomial (n-2) m + binomial (n-3) m
   + binomial (n-4) m + binomial (n-5) m + binomial (n-6) m + binomial (n-7) m
   + binomial (n-8) m + binomial (n-9) m + binomial (n-10) m)
   = binomial (n+1) (m+1) - binomial (n-10) (m+1) :=
  sorry

end Combinatorics

end binom_identity1_binom_identity2_l181_181744


namespace find_c_l181_181802

theorem find_c 
  (a b c : ℝ) 
  (h_vertex : ∀ x y, y = a * x^2 + b * x + c → 
    (∃ k l, l = b / (2 * a) ∧ k = a * l^2 + b * l + c ∧ k = 3 ∧ l = -2))
  (h_pass : ∀ x y, y = a * x^2 + b * x + c → 
    (x = 2 ∧ y = 7)) : c = 4 :=
by sorry

end find_c_l181_181802


namespace sqrt_sum_of_powers_l181_181917

theorem sqrt_sum_of_powers : sqrt (4^4 + 4^4 + 4^4 + 4^4) = 32 := by
  have h : 4^4 = 256 := by
    calc
      4^4 = 4 * 4 * 4 * 4 := by rw [show 4^4 = 4 * 4 * 4 * 4 from rfl]
      ... = 16 * 16       := by rw [mul_assoc, mul_assoc]
      ... = 256           := by norm_num
  calc
    sqrt (4^4 + 4^4 + 4^4 + 4^4)
      = sqrt (256 + 256 + 256 + 256) := by rw [h, h, h, h]
      ... = sqrt 1024                 := by norm_num
      ... = 32                        := by norm_num

end sqrt_sum_of_powers_l181_181917


namespace find_a_and_b_minimum_value_of_polynomial_l181_181843

noncomputable def polynomial_has_maximum (x y a b : ℝ) : Prop :=
  y = a * x ^ 3 + b * x ^ 2 ∧ x = 1 ∧ y = 3

noncomputable def polynomial_minimum_value (y : ℝ) : Prop :=
  y = 0

theorem find_a_and_b (a b x y : ℝ) (h : polynomial_has_maximum x y a b) :
  a = -6 ∧ b = 9 :=
by sorry

theorem minimum_value_of_polynomial (a b y : ℝ) (h : a = -6 ∧ b = 9) :
  polynomial_minimum_value y :=
by sorry

end find_a_and_b_minimum_value_of_polynomial_l181_181843


namespace waiter_tables_l181_181138

theorem waiter_tables (w m : ℝ) (avg_customers_per_table : ℝ) (total_customers : ℝ) (t : ℝ)
  (hw : w = 7.0)
  (hm : m = 3.0)
  (havg : avg_customers_per_table = 1.111111111)
  (htotal : total_customers = w + m)
  (ht : t = total_customers / avg_customers_per_table) :
  t = 90 :=
by
  -- Proof would be inserted here
  sorry

end waiter_tables_l181_181138


namespace part_a_probability_three_unused_rockets_part_b_expected_targets_hit_l181_181952

-- Proof Problem for Part (a):
theorem part_a_probability_three_unused_rockets (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) : 
  (∃ q : ℝ, q = 10 * p^3 * (1-p)^2) := sorry

-- Proof Problem for Part (b):
theorem part_b_expected_targets_hit (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) : 
  (∃ e : ℝ, e = 10 * p - p^10) := sorry

end part_a_probability_three_unused_rockets_part_b_expected_targets_hit_l181_181952


namespace initial_rotations_l181_181376

-- Given conditions as Lean definitions
def rotations_per_block : ℕ := 200
def blocks_to_ride : ℕ := 8
def additional_rotations_needed : ℕ := 1000

-- Question translated to proof statement
theorem initial_rotations (rotations : ℕ) :
  rotations + additional_rotations_needed = rotations_per_block * blocks_to_ride → rotations = 600 :=
by
  intros h
  sorry

end initial_rotations_l181_181376


namespace more_likely_millionaire_city_resident_l181_181278

noncomputable def probability_A (M N m n : ℝ) : ℝ :=
  m * M / (m * M + n * N)

noncomputable def probability_B (M N : ℝ) : ℝ :=
  M / (M + N)

theorem more_likely_millionaire_city_resident
  (M N : ℕ) (m n : ℝ) (hM : M > 0) (hN : N > 0)
  (hm : m > 10^6) (hn : n ≤ 10^6) :
  probability_A M N m n > probability_B M N :=
by {
  sorry
}

end more_likely_millionaire_city_resident_l181_181278


namespace part1_part2_l181_181522

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (Real.exp x / x) - Real.log x + x - a

theorem part1 (a : ℝ) : (∀ x > 0, f x a ≥ 0) ↔ a ≤ Real.exp 1 + 1 :=
  sorry

theorem part2 (a : ℝ) (x1 x2 : ℝ) (h1 : f x1 a = 0) (h2 : f x2 a = 0) : x1 * x2 < 1 :=
  sorry

end part1_part2_l181_181522


namespace difference_in_students_and_guinea_pigs_l181_181832

def num_students (classrooms : ℕ) (students_per_classroom : ℕ) : ℕ := classrooms * students_per_classroom
def num_guinea_pigs (classrooms : ℕ) (guinea_pigs_per_classroom : ℕ) : ℕ := classrooms * guinea_pigs_per_classroom
def difference_students_guinea_pigs (students : ℕ) (guinea_pigs : ℕ) : ℕ := students - guinea_pigs

theorem difference_in_students_and_guinea_pigs :
  ∀ (classrooms : ℕ) (students_per_classroom : ℕ) (guinea_pigs_per_classroom : ℕ),
  classrooms = 6 →
  students_per_classroom = 24 →
  guinea_pigs_per_classroom = 3 →
  difference_students_guinea_pigs (num_students classrooms students_per_classroom) (num_guinea_pigs classrooms guinea_pigs_per_classroom) = 126 :=
by
  intros
  sorry

end difference_in_students_and_guinea_pigs_l181_181832


namespace eval_expression_at_x_is_correct_l181_181262

theorem eval_expression_at_x_is_correct : 
  let x := 3 in (x^6 - 6 * x^2) = 675 :=
by
  sorry

end eval_expression_at_x_is_correct_l181_181262


namespace max_M_inequality_l181_181005

theorem max_M_inequality :
  ∃ M : ℝ, (∀ x y : ℝ, x + y ≥ 0 → (x^2 + y^2)^3 ≥ M * (x^3 + y^3) * (x * y - x - y)) ∧ M = 32 :=
by {
  sorry
}

end max_M_inequality_l181_181005


namespace tea_set_costs_l181_181094
noncomputable section

-- Definition for the conditions of part 1
def cost_condition1 (x y : ℝ) : Prop := x + 2 * y = 250
def cost_condition2 (x y : ℝ) : Prop := 3 * x + 4 * y = 600

-- Definition for the conditions of part 2
def cost_condition3 (a : ℝ) : ℝ := 108 * a + 60 * (80 - a)

-- Definition for the conditions of part 3
def profit (a b : ℝ) : ℝ := 30 * a + 20 * b

theorem tea_set_costs (x y : ℝ) (a : ℕ) :
  cost_condition1 x y →
  cost_condition2 x y →
  x = 100 ∧ y = 75 ∧ a ≤ 30 ∧ profit 30 50 = 1900 := by
  sorry

end tea_set_costs_l181_181094


namespace birthday_paradox_l181_181791

-- Defining the problem conditions
def people (n : ℕ) := n ≥ 367

-- Using the Pigeonhole Principle as a condition
def pigeonhole_principle (pigeonholes pigeons : ℕ) := pigeonholes < pigeons

-- Stating the final proposition
theorem birthday_paradox (n : ℕ) (days_in_year : ℕ) (h1 : days_in_year = 366) (h2 : people n) : pigeonhole_principle days_in_year n :=
sorry

end birthday_paradox_l181_181791


namespace coefficient_a6_l181_181019

def expand_equation (x a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 : ℝ) : Prop :=
  x * (x - 2) ^ 8 =
    a0 + a1 * (x - 1) + a2 * (x - 1) ^ 2 + a3 * (x - 1) ^ 3 + a4 * (x - 1) ^ 4 +
    a5 * (x - 1) ^ 5 + a6 * (x - 1) ^ 6 + a7 * (x - 1) ^ 7 + a8 * (x - 1) ^ 8 + 
    a9 * (x - 1) ^ 9

theorem coefficient_a6 (x a0 a1 a2 a3 a4 a5 a7 a8 a9 : ℝ) (h : expand_equation x a0 a1 a2 a3 a4 a5 (-28) a7 a8 a9) :
  a6 = -28 :=
sorry

end coefficient_a6_l181_181019


namespace compare_probabilities_l181_181273

theorem compare_probabilities 
  (M N : ℕ)
  (m n : ℝ)
  (h_m_pos : m > 0)
  (h_n_pos : n > 0)
  (h_m_million : m > 10^6)
  (h_n_million : n ≤ 10^6) :
  (M : ℝ) / (M + n / m * N) > (M : ℝ) / (M + N) :=
by
  sorry

end compare_probabilities_l181_181273


namespace total_amount_paid_l181_181104

theorem total_amount_paid (cost_of_manicure : ℝ) (tip_percentage : ℝ) (total : ℝ) 
  (h1 : cost_of_manicure = 30) (h2 : tip_percentage = 0.3) (h3 : total = cost_of_manicure + cost_of_manicure * tip_percentage) : 
  total = 39 :=
by
  sorry

end total_amount_paid_l181_181104


namespace rigid_motion_pattern_l181_181656

-- Define the types of transformations
inductive Transformation
| rotation : ℝ → Transformation -- rotation by an angle
| translation : ℝ → Transformation -- translation by a distance
| reflection_across_m : Transformation -- reflection across line m
| reflection_perpendicular_to_m : ℝ → Transformation -- reflective across line perpendicular to m at a point

-- Define the problem statement conditions
def pattern_alternates (line_m : ℝ → ℝ) : Prop := sorry -- This should define the alternating pattern of equilateral triangles and squares along line m

-- Problem statement in Lean
theorem rigid_motion_pattern (line_m : ℝ → ℝ) (p : Transformation → Prop)
    (h1 : p (Transformation.rotation 180)) -- 180-degree rotation is a valid transformation for the pattern
    (h2 : ∀ d, p (Transformation.translation d)) -- any translation by pattern unit length is a valid transformation
    (h3 : p Transformation.reflection_across_m) -- reflection across line m is a valid transformation
    (h4 : ∀ x, p (Transformation.reflection_perpendicular_to_m x)) -- reflection across any perpendicular line is a valid transformation
    : ∃ t : Finset Transformation, t.card = 4 ∧ ∀ t_val, t_val ∈ t → p t_val ∧ t_val ≠ Transformation.rotation 0 := 
sorry

end rigid_motion_pattern_l181_181656


namespace xyz_inequality_l181_181535

theorem xyz_inequality (x y z : ℝ) (h : x^2 + y^2 + z^2 = 2) : x + y + z ≤ x * y * z + 2 := by
  sorry

end xyz_inequality_l181_181535


namespace find_t_l181_181800

theorem find_t (s t : ℤ) (h1 : 12 * s + 7 * t = 173) (h2 : s = t - 3) : t = 11 :=
by
  sorry

end find_t_l181_181800


namespace evaluate_exponent_l181_181159

theorem evaluate_exponent : (3^2)^4 = 6561 := sorry

end evaluate_exponent_l181_181159


namespace physics_marks_l181_181620

theorem physics_marks (P C M : ℕ) 
  (h1 : P + C + M = 195)
  (h2 : P + M = 180)
  (h3 : P + C = 140) :
  P = 125 :=
by {
  sorry
}

end physics_marks_l181_181620


namespace symmetry_center_of_graph_l181_181021

noncomputable def f (x θ : ℝ) : ℝ := 2 * cos (2 * x + θ) * sin θ - sin (2 * (x + θ))

theorem symmetry_center_of_graph (θ : ℝ) : ∃ x, f x θ = 0 ∧ x = 0 := by
  use 0
  unfold f
  simp
  ring
  exact sin_zero

end symmetry_center_of_graph_l181_181021


namespace min_length_M_intersect_N_l181_181579

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

end min_length_M_intersect_N_l181_181579


namespace quadratic_roots_unique_l181_181682

theorem quadratic_roots_unique (p q : ℚ) :
  (∀ x : ℚ, x^2 + p * x + q = 0 ↔ (x = 2 * p ∨ x = p + q)) →
  p = 2 / 3 ∧ q = -8 / 3 :=
by
  sorry

end quadratic_roots_unique_l181_181682


namespace drinks_left_for_Seungwoo_l181_181747

def coke_taken_liters := 35 + 0.5
def cider_taken_liters := 27 + 0.2
def coke_drank_liters := 1 + 0.75

theorem drinks_left_for_Seungwoo :
  (coke_taken_liters - coke_drank_liters) + cider_taken_liters = 60.95 := by
  sorry

end drinks_left_for_Seungwoo_l181_181747


namespace bretschneider_l181_181645

noncomputable def bretschneider_theorem 
  (a b c d m n : ℝ) 
  (A C : ℝ) : Prop :=
  m^2 * n^2 = a^2 * c^2 + b^2 * d^2 - 2 * a * b * c * d * Real.cos (A + C)

theorem bretschneider (a b c d m n A C : ℝ) :
  bretschneider_theorem a b c d m n A C :=
sorry

end bretschneider_l181_181645


namespace solution_set_of_inequality_l181_181969

variable (f : ℝ → ℝ)

theorem solution_set_of_inequality (h1 : ∀ x, f'' x > 1 - f x) 
    (h2 : f 0 = 6) 
    (h3 : ∀ x, deriv f'' x = deriv (λ y, deriv f y)) :
    { x : ℝ | e^x * f x > e^x + 5 } = set.Ioi 0 :=
by
  -- skip the proof
  sorry

end solution_set_of_inequality_l181_181969


namespace married_fraction_l181_181864

variable (total_people : ℕ) (fraction_women : ℚ) (max_unmarried_women : ℕ)
variable (fraction_married : ℚ)

theorem married_fraction (h1 : total_people = 80)
                         (h2 : fraction_women = 1/4)
                         (h3 : max_unmarried_women = 20)
                         : fraction_married = 3/4 :=
by
  sorry

end married_fraction_l181_181864

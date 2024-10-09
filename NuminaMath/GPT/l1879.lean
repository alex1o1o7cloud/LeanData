import Mathlib

namespace trigonometric_identity_l1879_187987

theorem trigonometric_identity (α : ℝ) :
  (2 * Real.sin (Real.pi - α) + Real.sin (2 * α)) / (Real.cos (α / 2) ^ 2) = 4 * Real.sin α :=
by
  sorry

end trigonometric_identity_l1879_187987


namespace newer_model_distance_l1879_187967

-- Given conditions
def older_model_distance : ℕ := 160
def newer_model_factor : ℝ := 1.25

-- The statement to be proved
theorem newer_model_distance :
  newer_model_factor * (older_model_distance : ℝ) = 200 := by
  sorry

end newer_model_distance_l1879_187967


namespace decimal_to_fraction_l1879_187914

theorem decimal_to_fraction :
  (3.56 : ℚ) = 89 / 25 := 
sorry

end decimal_to_fraction_l1879_187914


namespace fraction_order_l1879_187971

theorem fraction_order :
  (19 / 15 < 17 / 13) ∧ (17 / 13 < 15 / 11) :=
by
  sorry

end fraction_order_l1879_187971


namespace possible_values_for_D_l1879_187941

noncomputable def distinct_digit_values (A B C D : Nat) : Prop :=
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧ 
  B < 10 ∧ A < 10 ∧ D < 10 ∧ C < 10 ∧ C = 9 ∧ (B + A = 9 + D)

theorem possible_values_for_D :
  ∃ (Ds : Finset Nat), (∀ D ∈ Ds, ∃ A B C, distinct_digit_values A B C D) ∧
  Ds.card = 5 :=
sorry

end possible_values_for_D_l1879_187941


namespace mark_and_alice_probability_l1879_187958

def probability_sunny_days : ℚ := 51 / 250

theorem mark_and_alice_probability :
  (∀ (day : ℕ), day < 5 → (∃ rain_prob sun_prob : ℚ, rain_prob = 0.8 ∧ sun_prob = 0.2 ∧ rain_prob + sun_prob = 1))
  → probability_sunny_days = 51 / 250 :=
by sorry

end mark_and_alice_probability_l1879_187958


namespace negation_equivalence_l1879_187903

variable (x : ℝ)

def original_proposition := ∃ x : ℝ, x^2 - 3*x + 3 < 0

def negation_proposition := ∀ x : ℝ, x^2 - 3*x + 3 ≥ 0

theorem negation_equivalence : ¬ original_proposition ↔ negation_proposition :=
by 
  -- Lean doesn’t require the actual proof here
  sorry

end negation_equivalence_l1879_187903


namespace smallest_real_number_l1879_187911

theorem smallest_real_number (A B C D : ℝ) 
  (hA : A = |(-2 : ℝ)|) 
  (hB : B = -1) 
  (hC : C = 0) 
  (hD : D = -1 / 2) : 
  min A (min B (min C D)) = B := 
by
  sorry

end smallest_real_number_l1879_187911


namespace M_inter_N_eq_M_l1879_187961

def M : Set ℝ := {x | x > 1}
def N : Set ℝ := {y | y ≥ 1}

theorem M_inter_N_eq_M : M ∩ N = M := by
  sorry

end M_inter_N_eq_M_l1879_187961


namespace max_value_expression_l1879_187981

theorem max_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  ∃ (M : ℝ), M = (x^2 * y^2 * z^2 * (x^2 + y^2 + z^2)) / ((x + y)^3 * (y + z)^3) ∧ M = 1/24 := 
sorry

end max_value_expression_l1879_187981


namespace interior_angles_sum_l1879_187962

theorem interior_angles_sum (n : ℕ) (h : ∀ (k : ℕ), k = n → 60 * n = 360) : 
  180 * (n - 2) = 720 :=
by
  sorry

end interior_angles_sum_l1879_187962


namespace inequality_solution_set_l1879_187979

theorem inequality_solution_set (x : ℝ) : 
  (∃ x, (2 < x ∧ x < 3)) ↔ 
  ((x - 2) * (x - 3) / (x^2 + 1) < 0) :=
by sorry

end inequality_solution_set_l1879_187979


namespace Carrie_hourly_wage_l1879_187964

theorem Carrie_hourly_wage (hours_per_week : ℕ) (weeks_per_month : ℕ) (cost_bike : ℕ) (remaining_money : ℕ)
  (total_hours : ℕ) (total_savings : ℕ) (x : ℕ) :
  hours_per_week = 35 → 
  weeks_per_month = 4 → 
  cost_bike = 400 → 
  remaining_money = 720 → 
  total_hours = hours_per_week * weeks_per_month → 
  total_savings = cost_bike + remaining_money → 
  total_savings = total_hours * x → 
  x = 8 :=
by 
  intros h_hw h_wm h_cb h_rm h_th h_ts h_tx
  sorry

end Carrie_hourly_wage_l1879_187964


namespace salt_amount_evaporation_l1879_187937

-- Define the conditions as constants
def total_volume : ℕ := 2 -- 2 liters
def salt_concentration : ℝ := 0.2 -- 20%

-- The volume conversion factor from liters to milliliters.
def liter_to_ml : ℕ := 1000

-- Define the statement to prove
theorem salt_amount_evaporation : total_volume * (salt_concentration * liter_to_ml) = 400 := 
by 
  -- We'll skip the proof steps here
  sorry

end salt_amount_evaporation_l1879_187937


namespace butterfat_milk_mixture_l1879_187960

theorem butterfat_milk_mixture :
  ∃ (x : ℝ), 0.10 * x + 0.45 * 8 = 0.20 * (x + 8) ∧ x = 20 := by
  sorry

end butterfat_milk_mixture_l1879_187960


namespace find_remainder_l1879_187926

theorem find_remainder (y : ℕ) (hy : 7 * y % 31 = 1) : (17 + 2 * y) % 31 = 4 :=
sorry

end find_remainder_l1879_187926


namespace union_eq_C_l1879_187949

def A: Set ℝ := { x | x > 2 }
def B: Set ℝ := { x | x < 0 }
def C: Set ℝ := { x | x * (x - 2) > 0 }

theorem union_eq_C : (A ∪ B) = C :=
by
  sorry

end union_eq_C_l1879_187949


namespace probability_spade_heart_diamond_l1879_187985

-- Condition: Definition of probability functions and a standard deck
def probability_of_first_spade (deck : Finset ℕ) : ℚ := 13 / 52
def probability_of_second_heart (deck : Finset ℕ) (first_card_spade : Prop) : ℚ := 13 / 51
def probability_of_third_diamond (deck : Finset ℕ) (first_card_spade : Prop) (second_card_heart : Prop) : ℚ := 13 / 50

-- Combined probability calculation
def probability_sequence_spade_heart_diamond (deck : Finset ℕ) : ℚ := 
  probability_of_first_spade deck * 
  probability_of_second_heart deck (true) * 
  probability_of_third_diamond deck (true) (true)

-- Lean statement proving the problem
theorem probability_spade_heart_diamond :
  probability_sequence_spade_heart_diamond (Finset.range 52) = 2197 / 132600 :=
by
  -- Proof steps will go here
  sorry

end probability_spade_heart_diamond_l1879_187985


namespace smallest_real_number_among_minus3_minus2_0_2_is_minus3_l1879_187995

theorem smallest_real_number_among_minus3_minus2_0_2_is_minus3 :
  min (min (-3:ℝ) (-2)) (min 0 2) = -3 :=
by {
    sorry
}

end smallest_real_number_among_minus3_minus2_0_2_is_minus3_l1879_187995


namespace mitya_age_l1879_187986

theorem mitya_age {M S: ℕ} (h1 : M = S + 11) (h2 : S = 2 * (S - (M - S))) : M = 33 :=
by
  -- proof steps skipped
  sorry

end mitya_age_l1879_187986


namespace value_of_expression_l1879_187952

theorem value_of_expression (x y : ℕ) (h1 : x = 3) (h2 : y = 4) : 
  (x^4 + 3 * y^3 + 10) / 7 = 283 / 7 := by
  sorry

end value_of_expression_l1879_187952


namespace oranges_in_buckets_l1879_187966

theorem oranges_in_buckets :
  ∀ (x : ℕ),
  (22 + x + (x - 11) = 89) →
  (x - 22 = 17) :=
by
  intro x h
  sorry

end oranges_in_buckets_l1879_187966


namespace base_value_l1879_187939

theorem base_value (b : ℕ) : (b - 1)^2 * (b - 2) = 256 → b = 17 :=
by
  sorry

end base_value_l1879_187939


namespace king_william_probability_l1879_187953

theorem king_william_probability :
  let m := 2
  let n := 15
  m + n = 17 :=
by
  sorry

end king_william_probability_l1879_187953


namespace circle_proof_problem_l1879_187994

variables {P Q R : Type}
variables {p q r dPQ dPR dQR : ℝ}

-- Given Conditions
variables (hpq : p > q) (hqr : q > r)
variables (hdPQ : ℝ) (hdPR : ℝ) (hdQR : ℝ)

-- Statement of the problem: prove that all conditions can be true
theorem circle_proof_problem :
  (∃ hpq' : dPQ = p + q, true) ∧
  (∃ hqr' : dQR = q + r, true) ∧
  (∃ hpr' : dPR > p + r, true) ∧
  (∃ hpq_diff : dPQ > p - q, true) →
  false := 
sorry

end circle_proof_problem_l1879_187994


namespace a6_b6_gt_a4b2_ab4_l1879_187940

theorem a6_b6_gt_a4b2_ab4 {a b : ℝ} (h1 : 0 < a) (h2 : 0 < b) (h3 : a ≠ b) :
  a^6 + b^6 > a^4 * b^2 + a^2 * b^4 :=
sorry

end a6_b6_gt_a4b2_ab4_l1879_187940


namespace find_a_l1879_187912

variable {x a : ℝ}

def A (x : ℝ) : Prop := x ≤ -1 ∨ x > 2
def B (x a : ℝ) : Prop := x < a ∨ x > a + 1

theorem find_a (hA : ∀ x, (x + 1) / (x - 2) ≥ 0 ↔ A x)
                (hB : ∀ x, x^2 - (2 * a + 1) * x + a^2 + a > 0 ↔ B x a)
                (hSub : ∀ x, A x → B x a) :
  -1 < a ∧ a ≤ 1 :=
sorry

end find_a_l1879_187912


namespace sum_of_interior_angles_at_vertex_A_l1879_187954

-- Definitions of the interior angles for a square and a regular octagon.
def square_interior_angle : ℝ := 90
def octagon_interior_angle : ℝ := 135

-- Theorem that states the sum of the interior angles at vertex A formed by the square and octagon.
theorem sum_of_interior_angles_at_vertex_A : square_interior_angle + octagon_interior_angle = 225 := by
  sorry

end sum_of_interior_angles_at_vertex_A_l1879_187954


namespace fraction_of_remaining_supplies_used_l1879_187993

theorem fraction_of_remaining_supplies_used 
  (initial_food : ℕ)
  (food_used_first_day_fraction : ℚ)
  (food_remaining_after_three_days : ℕ) 
  (food_used_second_period_fraction : ℚ) :
  initial_food = 400 →
  food_used_first_day_fraction = 2 / 5 →
  food_remaining_after_three_days = 96 →
  (initial_food - initial_food * food_used_first_day_fraction) * (1 - food_used_second_period_fraction) = food_remaining_after_three_days →
  food_used_second_period_fraction = 3 / 5 :=
by
  intros h1 h2 h3 h4
  sorry

end fraction_of_remaining_supplies_used_l1879_187993


namespace permutations_with_exactly_one_descent_permutations_with_exactly_two_descents_l1879_187956

-- Part (a)
theorem permutations_with_exactly_one_descent (n : ℕ) : 
  ∃ (count : ℕ), count = 2^n - n - 1 := sorry

-- Part (b)
theorem permutations_with_exactly_two_descents (n : ℕ) : 
  ∃ (count : ℕ), count = 3^n - 2^n * (n + 1) + (n * (n + 1)) / 2 := sorry

end permutations_with_exactly_one_descent_permutations_with_exactly_two_descents_l1879_187956


namespace fraction_of_august_tips_l1879_187945

variable {A : ℝ} -- A denotes the average monthly tips for the other months.
variable {total_tips_6_months : ℝ} (h1 : total_tips_6_months = 6 * A)
variable {august_tips : ℝ} (h2 : august_tips = 6 * A)
variable {total_tips : ℝ} (h3 : total_tips = total_tips_6_months + august_tips)

theorem fraction_of_august_tips (h1 : total_tips_6_months = 6 * A)
                                (h2 : august_tips = 6 * A)
                                (h3 : total_tips = total_tips_6_months + august_tips) :
    (august_tips / total_tips) = 1 / 2 :=
by
    sorry

end fraction_of_august_tips_l1879_187945


namespace wire_length_from_sphere_volume_l1879_187983

theorem wire_length_from_sphere_volume
  (r_sphere : ℝ) (r_cylinder : ℝ) (h : ℝ)
  (h_sphere : r_sphere = 12)
  (h_cylinder : r_cylinder = 4)
  (volume_conservation : (4/3 * Real.pi * r_sphere^3) = (Real.pi * r_cylinder^2 * h)) :
  h = 144 :=
by {
  sorry
}

end wire_length_from_sphere_volume_l1879_187983


namespace contrapositive_even_addition_l1879_187982

theorem contrapositive_even_addition (a b : ℕ) :
  (¬((a % 2 = 0) ∧ (b % 2 = 0)) → (a + b) % 2 ≠ 0) :=
sorry

end contrapositive_even_addition_l1879_187982


namespace left_vertex_of_ellipse_l1879_187924

theorem left_vertex_of_ellipse :
  ∃ (a b c : ℝ), 
    (a > b) ∧ (b > 0) ∧ (b = 4) ∧ (c = 3) ∧ 
    (c^2 = a^2 - b^2) ∧ 
    (3^2 = a^2 - 4^2) ∧ 
    (a = 5) ∧ 
    (∀ x y : ℝ, (x, y) = (-5, 0)) := 
sorry

end left_vertex_of_ellipse_l1879_187924


namespace race_completion_times_l1879_187997

theorem race_completion_times :
  ∃ (Patrick Manu Amy Olivia Sophie Jack : ℕ),
  Patrick = 60 ∧
  Manu = Patrick + 12 ∧
  Amy = Manu / 2 ∧
  Olivia = (2 * Amy) / 3 ∧
  Sophie = Olivia - 10 ∧
  Jack = Sophie + 8 ∧
  Manu = 72 ∧
  Amy = 36 ∧
  Olivia = 24 ∧
  Sophie = 14 ∧
  Jack = 22 := 
by
  -- proof here
  sorry

end race_completion_times_l1879_187997


namespace scientific_notation_of_12400_l1879_187980

theorem scientific_notation_of_12400 :
  12400 = 1.24 * 10^4 :=
sorry

end scientific_notation_of_12400_l1879_187980


namespace zero_in_A_l1879_187909

def A : Set ℝ := {x | x * (x + 1) = 0}

theorem zero_in_A : 0 ∈ A := by
  sorry

end zero_in_A_l1879_187909


namespace inequality_part_1_inequality_part_2_l1879_187934

noncomputable def f (x : ℝ) := |x - 2| + 2
noncomputable def g (x : ℝ) (m : ℝ) := m * |x|

theorem inequality_part_1 (x : ℝ) : f x > 5 ↔ x < -1 ∨ x > 5 := by
  sorry

theorem inequality_part_2 (m : ℝ) : (∀ x, f x ≥ g x m) ↔ m ≤ 1 := by
  sorry

end inequality_part_1_inequality_part_2_l1879_187934


namespace rabbit_total_distance_l1879_187969

theorem rabbit_total_distance 
  (r₁ r₂ : ℝ) 
  (h1 : r₁ = 7) 
  (h2 : r₂ = 15) 
  (q : ∀ (x : ℕ), x = 4) 
  : (3.5 * π + 8 + 7.5 * π + 8 + 3.5 * π + 8) = 14.5 * π + 24 := 
by
  sorry

end rabbit_total_distance_l1879_187969


namespace bruno_initial_books_l1879_187935

theorem bruno_initial_books (X : ℝ)
  (h1 : X - 4.5 + 10.25 = 39.75) :
  X = 34 := by
  sorry

end bruno_initial_books_l1879_187935


namespace ant_crawling_routes_ratio_l1879_187921

theorem ant_crawling_routes_ratio 
  (m n : ℕ) 
  (h1 : m = 2) 
  (h2 : n = 6) : 
  n / m = 3 :=
by
  -- Proof is omitted (we only need the statement as per the instruction)
  sorry

end ant_crawling_routes_ratio_l1879_187921


namespace cube_remainder_l1879_187984

theorem cube_remainder (n : ℤ) (h : n % 13 = 5) : (n^3) % 17 = 6 :=
by
  sorry

end cube_remainder_l1879_187984


namespace mark_profit_l1879_187920

variable (initial_cost tripling_factor new_value profit : ℕ)

-- Conditions
def initial_card_cost := 100
def card_tripling_factor := 3

-- Calculations based on conditions
def card_new_value := initial_card_cost * card_tripling_factor
def card_profit := card_new_value - initial_card_cost

-- Proof Statement
theorem mark_profit (initial_card_cost tripling_factor card_new_value card_profit : ℕ) 
  (h1: initial_card_cost = 100)
  (h2: tripling_factor = 3)
  (h3: card_new_value = initial_card_cost * tripling_factor)
  (h4: card_profit = card_new_value - initial_card_cost) :
  card_profit = 200 :=
  by sorry

end mark_profit_l1879_187920


namespace problem1_l1879_187942

theorem problem1 : 20 + (-14) - (-18) + 13 = 37 :=
by
  sorry

end problem1_l1879_187942


namespace trigonometric_inequality_l1879_187931

theorem trigonometric_inequality (a b A B : ℝ) (h : ∀ x : ℝ, 1 - a * Real.cos x - b * Real.sin x - A * Real.cos 2 * x - B * Real.sin 2 * x ≥ 0) : 
  a ^ 2 + b ^ 2 ≤ 2 ∧ A ^ 2 + B ^ 2 ≤ 1 := 
sorry

end trigonometric_inequality_l1879_187931


namespace parabola_vertex_on_x_axis_l1879_187970

theorem parabola_vertex_on_x_axis (c : ℝ) : 
  (∃ x : ℝ, x^2 + 2 * x + c = 0) → c = 1 := by
  sorry

end parabola_vertex_on_x_axis_l1879_187970


namespace ellipse_major_axis_value_l1879_187900

theorem ellipse_major_axis_value (m : ℝ) (P : ℝ × ℝ) (F1 F2 : ℝ × ℝ)
  (h1 : ∀ {x y : ℝ}, (x, y) = P → (x^2 / m) + (y^2 / 16) = 1)
  (h2 : dist P F1 = 3)
  (h3 : dist P F2 = 7)
  : m = 25 :=
sorry

end ellipse_major_axis_value_l1879_187900


namespace proof_problem_l1879_187938

-- Define the function f(x) = -x - x^3
def f (x : ℝ) : ℝ := -x - x^3

-- Define the main theorem according to the conditions and the required proofs.
theorem proof_problem (x1 x2 : ℝ) (h : x1 + x2 ≤ 0) :
  (f x1) * (f (-x1)) ≤ 0 ∧ (f x1 + f x2) ≥ (f (-x1) + f (-x2)) :=
by
  sorry

end proof_problem_l1879_187938


namespace tips_fraction_of_salary_l1879_187992

theorem tips_fraction_of_salary (S T x : ℝ) (h1 : T = x * S) 
  (h2 : T / (S + T) = 1 / 3) : x = 1 / 2 := by
  sorry

end tips_fraction_of_salary_l1879_187992


namespace range_of_a_l1879_187925

def p (a : ℝ) : Prop := ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0
def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0

theorem range_of_a (a : ℝ) (hp : p a) (hq : q a) : a ≤ -2 ∨ a = 1 := 
by sorry

end range_of_a_l1879_187925


namespace difference_white_black_l1879_187915

def total_stones : ℕ := 928
def white_stones : ℕ := 713
def black_stones : ℕ := total_stones - white_stones

theorem difference_white_black :
  (white_stones - black_stones = 498) :=
by
  -- Leaving the proof for later
  sorry

end difference_white_black_l1879_187915


namespace abs_diff_60th_terms_arithmetic_sequences_l1879_187951

theorem abs_diff_60th_terms_arithmetic_sequences :
  let C : (ℕ → ℤ) := λ n => 25 + 15 * (n - 1)
  let D : (ℕ → ℤ) := λ n => 40 - 15 * (n - 1)
  |C 60 - D 60| = 1755 :=
by
  sorry

end abs_diff_60th_terms_arithmetic_sequences_l1879_187951


namespace age_problem_l1879_187973

open Classical

noncomputable def sum_cubes_ages (r j m : ℕ) : ℕ :=
  r^3 + j^3 + m^3

theorem age_problem (r j m : ℕ) (h1 : 5 * r + 2 * j = 3 * m)
    (h2 : 3 * m^2 + 2 * j^2 = 5 * r^2) (h3 : Nat.gcd r (Nat.gcd j m) = 1) :
    sum_cubes_ages r j m = 3 := by
  sorry

end age_problem_l1879_187973


namespace value_of_expression_l1879_187977

theorem value_of_expression (a b : ℤ) (h : a - 2 * b - 3 = 0) : 9 - 2 * a + 4 * b = 3 := 
by 
  sorry

end value_of_expression_l1879_187977


namespace angle_size_proof_l1879_187988

-- Define the problem conditions
def fifteen_points_on_circle (θ : ℕ) : Prop :=
  θ = 360 / 15 

-- Define the central angles
def central_angle_between_adjacent_points (θ : ℕ) : ℕ :=
  360 / 15  

-- Define the two required central angles
def central_angle_A1O_A3 (θ : ℕ) : ℕ :=
  2 * θ

def central_angle_A3O_A7 (θ : ℕ) : ℕ :=
  4 * θ

-- Define the problem using the given conditions and the proven answer
noncomputable def angle_A1_A3_A7 : ℕ :=
  108

-- Lean 4 statement of the math problem to prove
theorem angle_size_proof (θ : ℕ) (h1 : fifteen_points_on_circle θ) :
  central_angle_A1O_A3 θ = 48 ∧ central_angle_A3O_A7 θ = 96 → 
  angle_A1_A3_A7 = 108 :=
by sorry

#check angle_size_proof

end angle_size_proof_l1879_187988


namespace n_fraction_of_sum_l1879_187905

theorem n_fraction_of_sum (l : List ℝ) (h1 : l.length = 21) (n : ℝ) (h2 : n ∈ l)
  (h3 : ∃ m, l.erase n = m ∧ m.length = 20 ∧ n = 4 * (m.sum / 20)) :
  n = (l.sum) / 6 :=
by
  sorry

end n_fraction_of_sum_l1879_187905


namespace x_squared_plus_y_squared_l1879_187963

theorem x_squared_plus_y_squared (x y : ℝ) (h1 : x - y = 12) (h2 : x * y = 9) : x^2 + y^2 = 162 :=
by
  sorry

end x_squared_plus_y_squared_l1879_187963


namespace units_digit_char_of_p_l1879_187919

theorem units_digit_char_of_p (p : ℕ) (h_pos : 0 < p) (h_even : p % 2 = 0)
    (h_units_zero : (p^3 % 10) - (p^2 % 10) = 0) (h_units_eleven : (p + 5) % 10 = 1) :
    p % 10 = 6 :=
sorry

end units_digit_char_of_p_l1879_187919


namespace domain_of_function_l1879_187999

theorem domain_of_function :
  (∀ x : ℝ, (x + 1 ≥ 0) ∧ (x ≠ 0) ↔ (x ≥ -1) ∧ (x ≠ 0)) :=
sorry

end domain_of_function_l1879_187999


namespace triangle_ratio_l1879_187906

-- Define the conditions and the main theorem statement
theorem triangle_ratio (a b c A B C : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) 
  (h_eq : b * Real.cos C + c * Real.cos B = 2 * b) 
  (h_law_sines_a : a = 2 * b * Real.sin B / Real.sin A) 
  (h_angles : A + B + C = Real.pi) :
  b / a = 1 / 2 :=
by 
  sorry

end triangle_ratio_l1879_187906


namespace cost_per_student_admission_l1879_187946

-- Definitions based on the conditions.
def cost_to_rent_bus : ℕ := 100
def total_budget : ℕ := 350
def number_of_students : ℕ := 25

-- The theorem that we need to prove.
theorem cost_per_student_admission : (total_budget - cost_to_rent_bus) / number_of_students = 10 :=
by
  sorry

end cost_per_student_admission_l1879_187946


namespace more_students_suggested_bacon_than_mashed_potatoes_l1879_187913

-- Define the number of students suggesting each type of food
def students_suggesting_mashed_potatoes := 479
def students_suggesting_bacon := 489

-- State the theorem that needs to be proven
theorem more_students_suggested_bacon_than_mashed_potatoes :
  students_suggesting_bacon - students_suggesting_mashed_potatoes = 10 := 
  by
  sorry

end more_students_suggested_bacon_than_mashed_potatoes_l1879_187913


namespace recommended_sleep_hours_l1879_187947

theorem recommended_sleep_hours
  (R : ℝ)   -- The recommended number of hours of sleep per day
  (h1 : 2 * 3 + 5 * (0.60 * R) = 30) : R = 8 :=
sorry

end recommended_sleep_hours_l1879_187947


namespace circle_equation_l1879_187930

theorem circle_equation 
  (h k : ℝ) 
  (H_center : k = 2 * h)
  (H_tangent : ∃ (r : ℝ), (h - 1)^2 + (k - 0)^2 = r^2 ∧ r = k) :
  (x - 1)^2 + (y - 2)^2 = 4 := 
sorry

end circle_equation_l1879_187930


namespace peters_brother_read_percentage_l1879_187923

-- Definitions based on given conditions
def total_books : ℕ := 20
def peter_read_percentage : ℕ := 40
def difference_between_peter_and_brother : ℕ := 6

-- Statement to prove
theorem peters_brother_read_percentage :
  peter_read_percentage / 100 * total_books - difference_between_peter_and_brother = 2 → 
  2 / total_books * 100 = 10 := by
  sorry

end peters_brother_read_percentage_l1879_187923


namespace length_of_goods_train_l1879_187989

theorem length_of_goods_train 
  (speed_km_per_hr : ℕ) (platform_length_m : ℕ) (time_sec : ℕ) 
  (h1 : speed_km_per_hr = 72) (h2 : platform_length_m = 300) (h3 : time_sec = 26) : 
  ∃ length_of_train : ℕ, length_of_train = 220 :=
by
  sorry

end length_of_goods_train_l1879_187989


namespace solve_quadratic_eqn_l1879_187965

theorem solve_quadratic_eqn (x : ℝ) : 3 * x ^ 2 = 27 ↔ x = 3 ∨ x = -3 :=
by
  sorry

end solve_quadratic_eqn_l1879_187965


namespace distinct_real_roots_range_l1879_187929

theorem distinct_real_roots_range (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 - 4*x1 - a = 0) ∧ (x2^2 - 4*x2 - a = 0)) ↔ a > -4 :=
by
  sorry

end distinct_real_roots_range_l1879_187929


namespace min_value_of_function_l1879_187917

theorem min_value_of_function (x : ℝ) (hx : x > 4) : 
  ∃ y : ℝ, y = x + 1 / (x - 4) ∧ (∀ z : ℝ, z = x + 1 / (x - 4) → z ≥ 6) :=
sorry

end min_value_of_function_l1879_187917


namespace circle_center_coordinates_l1879_187932

theorem circle_center_coordinates :
  ∀ (x y : ℝ), x^2 + y^2 - 10 * x + 6 * y + 25 = 0 → (5, -3) = ((-(-10) / 2), (-6 / 2)) :=
by
  intros x y h
  have H : (5, -3) = ((-(-10) / 2), (-6 / 2)) := sorry
  exact H

end circle_center_coordinates_l1879_187932


namespace matrix_power_100_l1879_187936

def matrix_100_pow : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![1, 0], ![200, 1]]

def A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![1, 0], ![2, 1]]

theorem matrix_power_100 (A : Matrix (Fin 2) (Fin 2) ℤ) :
  A^100 = matrix_100_pow :=
by
  sorry

end matrix_power_100_l1879_187936


namespace divisible_by_117_l1879_187943

theorem divisible_by_117 (n : ℕ) (hn : 0 < n) :
  117 ∣ (3^(2*(n+1)) * 5^(2*n) - 3^(3*n+2) * 2^(2*n)) :=
sorry

end divisible_by_117_l1879_187943


namespace train_pass_platform_in_correct_time_l1879_187955

def length_of_train : ℝ := 2500
def time_to_cross_tree : ℝ := 90
def length_of_platform : ℝ := 1500

noncomputable def speed_of_train : ℝ := length_of_train / time_to_cross_tree
noncomputable def total_distance_to_cover : ℝ := length_of_train + length_of_platform
noncomputable def time_to_pass_platform : ℝ := total_distance_to_cover / speed_of_train

theorem train_pass_platform_in_correct_time :
  abs (time_to_pass_platform - 143.88) < 0.01 :=
sorry

end train_pass_platform_in_correct_time_l1879_187955


namespace usual_time_to_school_l1879_187908

variables (R T : ℝ)

theorem usual_time_to_school :
  (3 / 2) * R * (T - 4) = R * T -> T = 12 :=
by sorry

end usual_time_to_school_l1879_187908


namespace simplify_polynomial_l1879_187910

theorem simplify_polynomial (r : ℝ) :
  (2 * r ^ 3 + 5 * r ^ 2 - 4 * r + 8) - (r ^ 3 + 9 * r ^ 2 - 2 * r - 3)
  = r ^ 3 - 4 * r ^ 2 - 2 * r + 11 :=
by sorry

end simplify_polynomial_l1879_187910


namespace find_c_d_l1879_187927

def star (c d : ℕ) : ℕ := c^d + c*d

theorem find_c_d (c d : ℕ) (hc : 2 ≤ c) (hd : 2 ≤ d) (h_star : star c d = 28) : c + d = 7 :=
by
  sorry

end find_c_d_l1879_187927


namespace lowest_possible_price_l1879_187901

theorem lowest_possible_price 
  (MSRP : ℝ)
  (regular_discount_percentage additional_discount_percentage : ℝ)
  (h1 : MSRP = 40)
  (h2 : regular_discount_percentage = 0.30)
  (h3 : additional_discount_percentage = 0.20) : 
  (MSRP * (1 - regular_discount_percentage) * (1 - additional_discount_percentage) = 22.40) := 
by
  sorry

end lowest_possible_price_l1879_187901


namespace calculate_expression_l1879_187916

theorem calculate_expression :
  (-0.25) ^ 2014 * (-4) ^ 2015 = -4 :=
by
  sorry

end calculate_expression_l1879_187916


namespace number_of_fish_given_to_dog_l1879_187998

-- Define the conditions
def condition1 (D C : ℕ) : Prop := C = D / 2
def condition2 (D C : ℕ) : Prop := D + C = 60

-- Theorem to prove the number of fish given to the dog
theorem number_of_fish_given_to_dog (D : ℕ) (C : ℕ) (h1 : condition1 D C) (h2 : condition2 D C) : D = 40 :=
by
  sorry

end number_of_fish_given_to_dog_l1879_187998


namespace four_brothers_money_l1879_187922

theorem four_brothers_money 
  (a_1 a_2 a_3 a_4 : ℝ) 
  (x : ℝ)
  (h1 : a_1 + a_2 + a_3 + a_4 = 48)
  (h2 : a_1 + 3 = x)
  (h3 : a_2 - 3 = x)
  (h4 : 3 * a_3 = x)
  (h5 : a_4 / 3 = x) :
  a_1 = 6 ∧ a_2 = 12 ∧ a_3 = 3 ∧ a_4 = 27 :=
by
  sorry

end four_brothers_money_l1879_187922


namespace lindsey_owns_more_cars_than_cathy_l1879_187950

theorem lindsey_owns_more_cars_than_cathy :
  ∀ (cathy carol susan lindsey : ℕ),
    cathy = 5 →
    carol = 2 * cathy →
    susan = carol - 2 →
    cathy + carol + susan + lindsey = 32 →
    lindsey = cathy + 4 :=
by
  intros cathy carol susan lindsey h1 h2 h3 h4
  sorry

end lindsey_owns_more_cars_than_cathy_l1879_187950


namespace find_x_l1879_187957

noncomputable def e_squared := Real.exp 2

theorem find_x (x : ℝ) (h : Real.log (x^2 - 5*x + 10) = 2) :
  x = 4.4 ∨ x = 0.6 :=
sorry

end find_x_l1879_187957


namespace slope_of_line_through_origin_and_center_l1879_187959

def Point := (ℝ × ℝ)

def is_center (p : Point) : Prop :=
  p = (3, 1)

def is_dividing_line (l : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, l x = y → y / x = 1 / 3

theorem slope_of_line_through_origin_and_center :
  ∃ l : ℝ → ℝ, (∀ p1 p2 : Point,
  p1 = (0, 0) →
  p2 = (3, 1) →
  is_center p2 →
  is_dividing_line l) :=
sorry

end slope_of_line_through_origin_and_center_l1879_187959


namespace quadratic_has_two_distinct_real_roots_l1879_187907

theorem quadratic_has_two_distinct_real_roots (p : ℝ) :
  ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ (x1 - 3) * (x1 - 2) - p^2 = 0 ∧ (x2 - 3) * (x2 - 2) - p^2 = 0 :=
by
  -- This part will be replaced with the actual proof
  sorry

end quadratic_has_two_distinct_real_roots_l1879_187907


namespace find_m_value_l1879_187976

theorem find_m_value :
  ∃ (m : ℝ), (∃ (midpoint: ℝ × ℝ), midpoint = ((5 + m) / 2, 1) ∧ midpoint.1 - 2 * midpoint.2 = 0) -> m = -1 :=
by
  sorry

end find_m_value_l1879_187976


namespace genuine_coin_remains_l1879_187991

theorem genuine_coin_remains (n : ℕ) (g f : ℕ) (h : n = 2022) (h_g : g > n/2) (h_f : f = n - g) : 
  (after_moves : ℕ) -> after_moves = n - 1 -> ∃ remaining_g : ℕ, remaining_g > 0 :=
by
  intros
  sorry

end genuine_coin_remains_l1879_187991


namespace circumference_to_diameter_ratio_l1879_187944

theorem circumference_to_diameter_ratio (C D : ℝ) (hC : C = 94.2) (hD : D = 30) :
  C / D = 3.14 :=
by
  rw [hC, hD]
  norm_num

end circumference_to_diameter_ratio_l1879_187944


namespace cylinder_cone_surface_area_l1879_187928

theorem cylinder_cone_surface_area (r h : ℝ) (π : ℝ) (l : ℝ)
    (h_relation : h = Real.sqrt 3 * r)
    (l_relation : l = 2 * r)
    (cone_lateral_surface_area : π * r * l = 2 * π * r ^ 2) :
    (2 * π * r * h) / (π * r ^ 2) = 2 * Real.sqrt 3 :=
by
    sorry

end cylinder_cone_surface_area_l1879_187928


namespace evaluate_F_2_f_3_l1879_187933

def f (a : ℤ) : ℤ := a^2 - 1

def F (a b : ℤ) : ℤ := b^3 - a

theorem evaluate_F_2_f_3 : F 2 (f 3) = 510 := by
  sorry

end evaluate_F_2_f_3_l1879_187933


namespace JuliaPlayedTuesday_l1879_187974

variable (Monday : ℕ) (Wednesday : ℕ) (Total : ℕ)
variable (KidsOnTuesday : ℕ)

theorem JuliaPlayedTuesday :
  Monday = 17 →
  Wednesday = 2 →
  Total = 34 →
  KidsOnTuesday = Total - (Monday + Wednesday) →
  KidsOnTuesday = 15 :=
by
  intros hMon hWed hTot hTue
  rw [hTot, hMon, hWed] at hTue
  exact hTue

end JuliaPlayedTuesday_l1879_187974


namespace problem1_l1879_187918

theorem problem1 (x y z : ℝ) (h : x^2 + y^2 + z^2 = 1) : 
  |x - y| + |y - z| + |z - x| ≤ 2 * Real.sqrt 2 := 
sorry

end problem1_l1879_187918


namespace largest_valid_integer_l1879_187902

open Nat

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def satisfies_conditions (n : ℕ) : Prop :=
  (100 ≤ n ∧ n < 1000) ∧
  ∀ d ∈ n.digits 10, d ≠ 0 ∧ n % d = 0 ∧
  sum_of_digits n % 6 = 0

theorem largest_valid_integer : ∃ n : ℕ, satisfies_conditions n ∧ (∀ m : ℕ, satisfies_conditions m → m ≤ n) ∧ n = 936 :=
by
  sorry

end largest_valid_integer_l1879_187902


namespace age_of_youngest_child_l1879_187978

theorem age_of_youngest_child (mother_fee : ℝ) (child_fee_per_year : ℝ) 
  (total_fee : ℝ) (t : ℝ) (y : ℝ) (child_fee : ℝ)
  (h_mother_fee : mother_fee = 2.50)
  (h_child_fee_per_year : child_fee_per_year = 0.25)
  (h_total_fee : total_fee = 4.00)
  (h_child_fee : child_fee = total_fee - mother_fee)
  (h_y : y = 6 - 2 * t)
  (h_fee_eq : child_fee = y * child_fee_per_year) : y = 2 := 
by
  sorry

end age_of_youngest_child_l1879_187978


namespace problem_quadratic_roots_l1879_187996

theorem problem_quadratic_roots (m : ℝ) :
  (∀ x : ℝ, (m + 3) * x^2 - 4 * m * x + 2 * m - 1 = 0 →
    (∃ x₁ x₂ : ℝ, x₁ * x₂ < 0 ∧ |x₁| > x₂)) ↔ -3 < m ∧ m < 0 :=
sorry

end problem_quadratic_roots_l1879_187996


namespace sufficient_not_necessary_range_l1879_187968

theorem sufficient_not_necessary_range (x a : ℝ) : (∀ x, x < 1 → x < a) ∧ (∃ x, x < a ∧ ¬ (x < 1)) ↔ 1 < a := by
  sorry

end sufficient_not_necessary_range_l1879_187968


namespace part1_l1879_187904

def is_Xn_function (n : ℝ) (f : ℝ → ℝ) : Prop :=
  ∃ x1 x2, x1 ≠ x2 ∧ f x1 = f x2 ∧ x1 + x2 = 2 * n

theorem part1 : is_Xn_function 0 (fun x => abs x) ∧ is_Xn_function (1/2) (fun x => x^2 - x) :=
by
  sorry

end part1_l1879_187904


namespace sin_of_7pi_over_6_l1879_187948

theorem sin_of_7pi_over_6 : Real.sin (7 * Real.pi / 6) = -1 / 2 :=
by
  -- Conditions from the statement in a)
  -- Given conditions: \(\sin (180^\circ + \theta) = -\sin \theta\)
  -- \(\sin 30^\circ = \frac{1}{2}\)
  sorry

end sin_of_7pi_over_6_l1879_187948


namespace five_goats_choir_l1879_187972

theorem five_goats_choir 
  (total_members : ℕ)
  (num_rows : ℕ)
  (total_members_eq : total_members = 51)
  (num_rows_eq : num_rows = 4) :
  ∃ row_people : ℕ, row_people ≥ 13 :=
by 
  sorry

end five_goats_choir_l1879_187972


namespace eagles_points_l1879_187975

theorem eagles_points (s e : ℕ) (h1 : s + e = 52) (h2 : s - e = 6) : e = 23 :=
by
  sorry

end eagles_points_l1879_187975


namespace work_increase_percentage_l1879_187990

theorem work_increase_percentage (p w : ℕ) (hp : p > 0) : 
  (((4 / 3 : ℚ) * w) - w) / w * 100 = 33.33 := 
sorry

end work_increase_percentage_l1879_187990

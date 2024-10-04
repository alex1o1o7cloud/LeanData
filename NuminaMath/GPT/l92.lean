import Mathlib

namespace value_of_2_star_3_l92_92850

def star (a b : ℕ) : ℕ := a * b ^ 3 - b + 2

theorem value_of_2_star_3 : star 2 3 = 53 :=
by
  -- This is where the proof would go
  sorry

end value_of_2_star_3_l92_92850


namespace total_people_present_l92_92907

theorem total_people_present (A B : ℕ) 
  (h1 : 2 * A + B = 10) 
  (h2 : A + 2 * B = 14) :
  A + B = 8 :=
sorry

end total_people_present_l92_92907


namespace not_p_and_p_or_q_implies_q_l92_92529

theorem not_p_and_p_or_q_implies_q (p q : Prop) (h1 : ¬ p) (h2 : p ∨ q) : q :=
by
  have h3 : p := sorry
  have h4 : false := sorry
  exact sorry

end not_p_and_p_or_q_implies_q_l92_92529


namespace induction_base_case_l92_92072

theorem induction_base_case : (-1 : ℤ) + 3 - 5 + (-1)^2 * 1 = (-1 : ℤ) := sorry

end induction_base_case_l92_92072


namespace average_percentage_decrease_l92_92422

theorem average_percentage_decrease
  (original_price final_price : ℕ)
  (h_original_price : original_price = 2000)
  (h_final_price : final_price = 1280) :
  (original_price - final_price) / original_price * 100 / 2 = 18 :=
by 
  sorry

end average_percentage_decrease_l92_92422


namespace zander_stickers_l92_92466

theorem zander_stickers (S : ℕ) (h1 : 44 = (11 / 25) * S) : S = 100 :=
by
  sorry

end zander_stickers_l92_92466


namespace sam_drove_200_miles_l92_92823

-- Define the conditions
def marguerite_distance : ℕ := 150
def marguerite_time : ℕ := 3
def sam_time : ℕ := 4
def marguerite_speed := marguerite_distance / marguerite_time

-- Define the question
def sam_distance (speed : ℕ) (time : ℕ) : ℕ := speed * time

-- State the theorem to prove the answer
theorem sam_drove_200_miles :
  sam_distance marguerite_speed sam_time = 200 := by
  sorry

end sam_drove_200_miles_l92_92823


namespace factor_polynomial_l92_92275

-- Define the necessary polynomials
def p (x : ℝ) : ℝ := x^2 + 4*x + 3
def q (x : ℝ) : ℝ := x^2 + 8*x + 15
def r (x : ℝ) : ℝ := x^2 + 6*x - 8

-- State the main theorem
theorem factor_polynomial (x : ℝ) : 
  (p x * q x) + r x = (x^2 + 7*x + 1) * (x^2 + 4*x + 7) :=
by
  sorry

end factor_polynomial_l92_92275


namespace train_trip_length_l92_92596

theorem train_trip_length (v D : ℝ) :
  (3 + (3 * D - 6 * v) / (2 * v) = 4 + D / v) ∧ 
  (2.5 + 120 / v + (6 * D - 12 * v - 720) / (5 * v) = 3.5 + D / v) →
  (D = 420 ∨ D = 480 ∨ D = 540 ∨ D = 600 ∨ D = 660) :=
by
  sorry

end train_trip_length_l92_92596


namespace nature_reserve_birds_l92_92655

theorem nature_reserve_birds :
  ∀ N : ℕ, 
    (N > 0) →
    let hawks := 0.30 * N in
    let non_hawks := N - hawks in
    let paddyfield_warblers := 0.40 * non_hawks in
    let kingfishers := 0.25 * paddyfield_warblers in
    let non_hawks_paddyfield_warblers_kingfishers := hawks + paddyfield_warblers + kingfishers in
  N - non_hawks_paddyfield_warblers_kingfishers = 0.35 * N :=
by
  intros N hN hawks non_hawks paddyfield_warblers kingfishers non_hawks_paddyfield_warblers_kingfishers,
  sorry

end nature_reserve_birds_l92_92655


namespace factor_polynomial_l92_92305

theorem factor_polynomial :
  (x^2 + 4 * x + 3) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8) = 
  (x^2 + 6 * x + 7) * (x^2 + 6 * x + 8) :=
sorry

end factor_polynomial_l92_92305


namespace sequence_filling_l92_92788

theorem sequence_filling :
  ∃ (a : Fin 8 → ℕ), 
    a 0 = 20 ∧ 
    a 7 = 16 ∧ 
    (∀ i : Fin 6, a i + a (i+1) + a (i+2) = 100) ∧ 
    (a 1 = 16) ∧ 
    (a 2 = 64) ∧ 
    (a 3 = 20) ∧ 
    (a 4 = 16) ∧ 
    (a 5 = 64) ∧ 
    (a 6 = 20) := 
by
  sorry

end sequence_filling_l92_92788


namespace range_of_a_l92_92442

theorem range_of_a (a : ℝ) : (∃ x : ℝ, |x - 4| + |x - 3| < a) ↔ a > 1 := 
sorry

end range_of_a_l92_92442


namespace ratio_traditionalists_progressives_l92_92719

-- Define the given conditions
variables (T P C : ℝ)
variables (h1 : C = P + 4 * T)
variables (h2 : 4 * T = 0.75 * C)

-- State the theorem
theorem ratio_traditionalists_progressives (h1 : C = P + 4 * T) (h2 : 4 * T = 0.75 * C) : T / P = 3 / 4 :=
by {
  sorry
}

end ratio_traditionalists_progressives_l92_92719


namespace call_cost_inequalities_min_call_cost_correct_l92_92882

noncomputable def call_cost_before (x : ℝ) : ℝ :=
  if x ≤ 3 then 0.2 else 0.4

noncomputable def call_cost_after (x : ℝ) : ℝ :=
  if x ≤ 3 then 0.2
  else if x ≤ 4 then 0.2 + 0.1 * (x - 3)
  else 0.3 + 0.1 * (x - 4)

theorem call_cost_inequalities : 
  (call_cost_before 4 = 0.4 ∧ call_cost_after 4 = 0.3) ∧
  (call_cost_before 4.3 = 0.4 ∧ call_cost_after 4.3 = 0.4) ∧
  (call_cost_before 5.8 = 0.4 ∧ call_cost_after 5.8 = 0.5) ∧
  (∀ x, (0 < x ∧ x ≤ 3) ∨ x > 4 → call_cost_before x ≤ call_cost_after x) :=
by
  sorry

noncomputable def min_call_cost_plan (m : ℝ) (n : ℕ) : ℝ :=
  if 3 * n - 1 < m ∧ m ≤ 3 * n then 0.2 * n
  else if 3 * n < m ∧ m ≤ 3 * n + 1 then 0.2 * n + 0.1
  else if 3 * n + 1 < m ∧ m ≤ 3 * n + 2 then 0.2 * n + 0.2
  else 0.0  -- Fallback, though not necessary as per the conditions

theorem min_call_cost_correct (m : ℝ) (n : ℕ) (h : m > 5) :
  (3 * n - 1 < m ∧ m ≤ 3 * n → min_call_cost_plan m n = 0.2 * n) ∧
  (3 * n < m ∧ m ≤ 3 * n + 1 → min_call_cost_plan m n = 0.2 * n + 0.1) ∧
  (3 * n + 1 < m ∧ m ≤ 3 * n + 2 → min_call_cost_plan m n = 0.2 * n + 0.2) :=
by
  sorry

end call_cost_inequalities_min_call_cost_correct_l92_92882


namespace ivan_uses_more_paint_l92_92455

-- Conditions
def ivan_section_area : ℝ := 5 * 2
def petr_section_area (alpha : ℝ) : ℝ := 5 * 2 * Real.sin(alpha)
axiom alpha_lt_90 : ∀ α : ℝ, α < 90 → Real.sin(α) < 1

-- Assertion
theorem ivan_uses_more_paint (α : ℝ) (h1 : α < 90) : ivan_section_area > petr_section_area α :=
by
  sorry

end ivan_uses_more_paint_l92_92455


namespace least_integer_in_ratio_1_3_5_l92_92022

theorem least_integer_in_ratio_1_3_5 (a b c : ℕ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_sum : a + b + c = 90) (h_ratio : a * 3 = b ∧ a * 5 = c) : a = 10 :=
sorry

end least_integer_in_ratio_1_3_5_l92_92022


namespace conditional_probability_l92_92530

variables (A B : Prop)
variables (P : Prop → ℚ)
variables (h₁ : P A = 8 / 30) (h₂ : P (A ∧ B) = 7 / 30)

theorem conditional_probability : P (A → B) = 7 / 8 :=
by sorry

end conditional_probability_l92_92530


namespace birch_trees_count_l92_92940

-- Definitions based on the conditions
def total_trees : ℕ := 4000
def percentage_spruce : ℕ := 10
def percentage_pine : ℕ := 13

def count_spruce : ℕ := (percentage_spruce * total_trees) / 100
def count_pine : ℕ := (percentage_pine * total_trees) / 100
def count_oak : ℕ := count_spruce + count_pine

def count_birch : ℕ := total_trees - (count_spruce + count_pine + count_oak)

-- The theorem to be proven
theorem birch_trees_count :
  count_birch = 2160 := by
  sorry

end birch_trees_count_l92_92940


namespace probability_of_same_length_segments_l92_92155

-- Definitions directly from the conditions
def total_elements : ℕ := 6 + 9

-- Probability calculations
def probability_same_length : ℚ :=
  ((6 / total_elements) * (5 / (total_elements - 1))) + 
  ((9 / total_elements) * (8 / (total_elements - 1)))

-- Proof statement
theorem probability_of_same_length_segments : probability_same_length = 17 / 35 := by
  unfold probability_same_length
  sorry

end probability_of_same_length_segments_l92_92155


namespace solution_set_I_range_of_m_II_l92_92358

noncomputable def f (x : ℝ) : ℝ := |x - 1| + |x - 2|

theorem solution_set_I : {x : ℝ | 0 ≤ x ∧ x ≤ 3} = {x : ℝ | f x ≤ 3} :=
sorry

theorem range_of_m_II (x : ℝ) (hx : x > 0) : ∃ m : ℝ, ∀ (x : ℝ), f x ≤ m - x - 4 / x → m ≥ 5 :=
sorry

end solution_set_I_range_of_m_II_l92_92358


namespace find_b_of_parabola_axis_of_symmetry_l92_92004

theorem find_b_of_parabola_axis_of_symmetry (b : ℝ) :
  (∀ (x : ℝ), (x = 1) ↔ (x = - (b / (2 * 2))) ) → b = 4 :=
by
  intro h
  sorry

end find_b_of_parabola_axis_of_symmetry_l92_92004


namespace triangle_angle_contradiction_l92_92497

theorem triangle_angle_contradiction :
  ∀ (α β γ : ℝ), (α + β + γ = 180) →
  (α > 60) ∧ (β > 60) ∧ (γ > 60) →
  false :=
by
  intros α β γ h_sum h_angles
  sorry

end triangle_angle_contradiction_l92_92497


namespace find_x_given_y64_l92_92020

variable (x y k : ℝ)

def inversely_proportional (x y : ℝ) := (x^3 * y = k)

theorem find_x_given_y64
  (h_pos : x > 0 ∧ y > 0)
  (h_inversely : inversely_proportional x y)
  (h_given : inversely_proportional 2 8)
  (h_y64 : y = 64) :
  x = 1 := by
  sorry

end find_x_given_y64_l92_92020


namespace power_sum_l92_92239

theorem power_sum : (-2) ^ 2007 + (-2) ^ 2008 = 2 ^ 2007 := by
  sorry

end power_sum_l92_92239


namespace cos_comp_l92_92243

open Real

theorem cos_comp {a b c : ℝ} (h1 : a = cos (3 / 2)) (h2 : b = -cos (7 / 4)) (h3 : c = sin (1 / 10)) : 
  a < c ∧ c < b := 
by
  -- Assume the hypotheses
  sorry

end cos_comp_l92_92243


namespace complete_square_solution_l92_92203

theorem complete_square_solution (x : ℝ) :
  x^2 - 2*x - 3 = 0 → (x - 1)^2 = 4 :=
by
  sorry

end complete_square_solution_l92_92203


namespace averageSpeed_l92_92685

-- Define the total distance driven by Jane
def totalDistance : ℕ := 200

-- Define the total time duration from 6 a.m. to 11 a.m.
def totalTime : ℕ := 5

-- Theorem stating that the average speed is 40 miles per hour
theorem averageSpeed (h1 : totalDistance = 200) (h2 : totalTime = 5) : totalDistance / totalTime = 40 := 
by
  sorry

end averageSpeed_l92_92685


namespace intersection_subset_complement_l92_92665

open Set

variable (U A B : Set ℕ)

theorem intersection_subset_complement (U : Set ℕ) (A B : Set ℕ) 
  (hU: U = {1, 2, 3, 4, 5, 6}) 
  (hA: A = {1, 3, 5}) 
  (hB: B = {2, 4, 5}) : 
  A ∩ (U \ B) = {1, 3} := 
by
  sorry

end intersection_subset_complement_l92_92665


namespace expenditure_fraction_l92_92097

variable (B : ℝ)
def cost_of_book (x y : ℝ) (B : ℝ) := x = 0.30 * (B - 2 * y)
def cost_of_coffee (x y : ℝ) (B : ℝ) := y = 0.10 * (B - x)

theorem expenditure_fraction (x y : ℝ) (B : ℝ) 
  (hx : cost_of_book x y B) 
  (hy : cost_of_coffee x y B) : 
  (x + y) / B = 31 / 94 :=
sorry

end expenditure_fraction_l92_92097


namespace intersection_of_A_and_B_l92_92776

def I := {x : ℝ | true}
def A := {x : ℝ | x * (x - 1) ≥ 0}
def B := {x : ℝ | x > 1}
def C := {x : ℝ | x > 1}

theorem intersection_of_A_and_B : A ∩ B = C := by
  sorry

end intersection_of_A_and_B_l92_92776


namespace taxi_ride_cost_l92_92899

-- Definitions given in the conditions
def base_fare : ℝ := 2.00
def cost_per_mile : ℝ := 0.30
def distance_traveled : ℝ := 10

-- The theorem we need to prove
theorem taxi_ride_cost : base_fare + (cost_per_mile * distance_traveled) = 5.00 :=
by
  sorry

end taxi_ride_cost_l92_92899


namespace dressing_p_percentage_l92_92713

-- Define the percentages of vinegar and oil in dressings p and q
def vinegar_in_p : ℝ := 0.30
def vinegar_in_q : ℝ := 0.10

-- Define the desired percentage of vinegar in the new dressing
def vinegar_in_new_dressing : ℝ := 0.12

-- Define the total mass of the new dressing
def total_mass_new_dressing : ℝ := 100.0

-- Define the mass of dressing p in the new dressing
def mass_of_p (x : ℝ) : ℝ := x

-- Define the mass of dressing q in the new dressing
def mass_of_q (x : ℝ) : ℝ := total_mass_new_dressing - x

-- Define the amount of vinegar contributed by dressings p and q
def vinegar_from_p (x : ℝ) : ℝ := vinegar_in_p * mass_of_p x
def vinegar_from_q (x : ℝ) : ℝ := vinegar_in_q * mass_of_q x

-- Define the total vinegar in the new dressing
def total_vinegar (x : ℝ) : ℝ := vinegar_from_p x + vinegar_from_q x

-- Problem statement: prove the percentage of dressing p in the new dressing
theorem dressing_p_percentage (x : ℝ) (hx : total_vinegar x = vinegar_in_new_dressing * total_mass_new_dressing) :
  (mass_of_p x / total_mass_new_dressing) * 100 = 10 :=
by
  sorry

end dressing_p_percentage_l92_92713


namespace students_with_both_uncool_parents_l92_92373

theorem students_with_both_uncool_parents :
  let total_students := 35
  let cool_dads := 18
  let cool_moms := 22
  let both_cool := 11
  total_students - (cool_dads + cool_moms - both_cool) = 6 := by
sorry

end students_with_both_uncool_parents_l92_92373


namespace factor_polynomial_l92_92320

theorem factor_polynomial :
  ∀ x : ℝ, 
  (x^2 + 4 * x + 3) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8) 
  = (x^2 + 6 * x + 1) * (x^2 + 6 * x + 37) :=
by
  intros x
  sorry

end factor_polynomial_l92_92320


namespace inequality_solution_l92_92755

theorem inequality_solution (x : ℝ) :
  (x + 2) / (x^2 + 4) > 2 / x + 12 / 5 ↔ x < 0 :=
by
  sorry

end inequality_solution_l92_92755


namespace allocation_schemes_l92_92624

theorem allocation_schemes (volunteers events : ℕ) (h_vol : volunteers = 5) (h_events : events = 4) :
  (∃ allocation_scheme : ℕ, allocation_scheme = 10 * 24) :=
by
  use 240
  sorry

end allocation_schemes_l92_92624


namespace factorization_identity_l92_92331

-- We are asked to prove the mathematical equality under given conditions.
theorem factorization_identity (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = (x^2 + 6*x + 8)^2 :=
by
  sorry

end factorization_identity_l92_92331


namespace men_hours_per_day_l92_92878

theorem men_hours_per_day
  (H : ℕ)
  (men_days := 15 * 21 * H)
  (women_days := 21 * 20 * 9)
  (conversion_ratio := 3 / 2)
  (equivalent_man_hours := women_days * conversion_ratio)
  (same_work : men_days = equivalent_man_hours) :
  H = 8 :=
by
  sorry

end men_hours_per_day_l92_92878


namespace probability_of_not_all_same_number_l92_92045

theorem probability_of_not_all_same_number :
  ∀ (D : ℕ) (n : ℕ),
  D = 8 → n = 5 → 
  let total_outcomes := D^n in
  let same_outcome_probability := D / total_outcomes  in
  let not_all_same_probability := 1 - same_outcome_probability in
  (not_all_same_probability = 4095 / 4096) :=
begin
  intros D n D_eq n_eq,
  simp only [D_eq, n_eq],
  let total_outcomes := 8^5,
  let same_outcome_probability := 8 / total_outcomes,
  let not_all_same_probability := 1 - same_outcome_probability,
  have total_outcome_calc : total_outcomes = 8^5 := by sorry,
  have same_outcome_prob_calc : same_outcome_probability = 1 / 4096 := by sorry,
  have not_all_same_prob_calc : not_all_same_probability = 4095 / 4096 := by sorry,
  exact not_all_same_prob_calc,
end

end probability_of_not_all_same_number_l92_92045


namespace factor_polynomial_l92_92306

theorem factor_polynomial :
  (x^2 + 4 * x + 3) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8) = 
  (x^2 + 6 * x + 7) * (x^2 + 6 * x + 8) :=
sorry

end factor_polynomial_l92_92306


namespace find_other_endpoint_diameter_l92_92241

-- Define the given conditions
def center : ℝ × ℝ := (1, 2)
def endpoint_A : ℝ × ℝ := (4, 6)

-- Define a function to find the other endpoint
def other_endpoint (center endpoint_A : ℝ × ℝ) : ℝ × ℝ := 
  let vector_CA := (center.1 - endpoint_A.1, center.2 - endpoint_A.2)
  let vector_CB := (-vector_CA.1, -vector_CA.2)
  (center.1 + vector_CB.1, center.2 + vector_CB.2)

-- State the theorem
theorem find_other_endpoint_diameter : 
  ∀ center endpoint_A, other_endpoint center endpoint_A = (4, 6) :=
by
  intro center endpoint_A
  -- Proof would go here
  sorry

end find_other_endpoint_diameter_l92_92241


namespace max_prob_games_4_choose_best_of_five_l92_92847

-- Definitions of probabilities for Team A and Team B in different game scenarios
def prob_win_deciding_game : ℝ := 0.5
def prob_A_non_deciding : ℝ := 0.6
def prob_B_non_deciding : ℝ := 0.4

-- Definitions of probabilities for different number of games in the series
def prob_xi_3 : ℝ := (prob_A_non_deciding)^3 + (prob_B_non_deciding)^3
def prob_xi_4 : ℝ := 3 * (prob_A_non_deciding^2 * prob_B_non_deciding * prob_A_non_deciding + prob_B_non_deciding^2 * prob_A_non_deciding * prob_B_non_deciding)
def prob_xi_5 : ℝ := 6 * (prob_A_non_deciding^2 * prob_B_non_deciding^2) * (2 * prob_win_deciding_game)

-- The statement that a series of 4 games has the highest probability
theorem max_prob_games_4 : prob_xi_4 > prob_xi_5 ∧ prob_xi_4 > prob_xi_3 :=
by {
  sorry
}

-- Definitions of winning probabilities in the series for Team A
def prob_A_win_best_of_3 : ℝ := (prob_A_non_deciding)^2 + 2 * (prob_A_non_deciding * prob_B_non_deciding * prob_win_deciding_game)
def prob_A_win_best_of_5 : ℝ := (prob_A_non_deciding)^3 + 3 * (prob_A_non_deciding^2 * prob_B_non_deciding) + 6 * (prob_A_non_deciding^2 * prob_B_non_deciding^2 * prob_win_deciding_game)

-- The statement that Team A has a higher chance of winning in a best-of-five series
theorem choose_best_of_five : prob_A_win_best_of_5 > prob_A_win_best_of_3 :=
by {
  sorry
}

end max_prob_games_4_choose_best_of_five_l92_92847


namespace intersection_product_of_circles_l92_92064

theorem intersection_product_of_circles :
  (∀ x y : ℝ, (x^2 + 2 * x + y^2 + 4 * y + 5 = 0) ∧ (x^2 + 6 * x + y^2 + 4 * y + 9 = 0) →
  x * y = 2) :=
sorry

end intersection_product_of_circles_l92_92064


namespace transform_map_ABCD_to_A_l92_92514

structure Point :=
(x : ℤ)
(y : ℤ)

structure Rectangle :=
(A : Point)
(B : Point)
(C : Point)
(D : Point)

def transform180 (p : Point) : Point :=
  ⟨-p.x, -p.y⟩

def rect_transform180 (rect : Rectangle) : Rectangle :=
  { A := transform180 rect.A,
    B := transform180 rect.B,
    C := transform180 rect.C,
    D := transform180 rect.D }

def ABCD := Rectangle.mk ⟨-3, 2⟩ ⟨-1, 2⟩ ⟨-1, 5⟩ ⟨-3, 5⟩
def A'B'C'D' := Rectangle.mk ⟨3, -2⟩ ⟨1, -2⟩ ⟨1, -5⟩ ⟨3, -5⟩

theorem transform_map_ABCD_to_A'B'C'D' :
  rect_transform180 ABCD = A'B'C'D' :=
by
  -- This is where the proof would go.
  sorry

end transform_map_ABCD_to_A_l92_92514


namespace probability_not_all_dice_show_different_l92_92062

noncomputable def probability_not_all_same : ℚ :=
  let total_outcomes := 8^5
  let same_number_outcomes := 8
  (total_outcomes - same_number_outcomes) / total_outcomes

theorem probability_not_all_dice_show_different : 
  probability_not_all_same = 4095 / 4096 := by
  sorry

end probability_not_all_dice_show_different_l92_92062


namespace largest_x_satisfies_eq_l92_92028

theorem largest_x_satisfies_eq (x : ℝ) (h : sqrt (3 * x) = 5 * x) : x ≤ 3 / 25 :=
sorry

end largest_x_satisfies_eq_l92_92028


namespace garden_snake_is_10_inches_l92_92489

-- Define the conditions from the problem statement
def garden_snake_length (garden_snake boa_constrictor : ℝ) : Prop :=
  boa_constrictor = 7 * garden_snake

def boa_constrictor_length (boa_constrictor : ℝ) : Prop :=
  boa_constrictor = 70

-- Prove the length of the garden snake
theorem garden_snake_is_10_inches : ∃ (garden_snake : ℝ), garden_snake_length garden_snake 70 ∧ garden_snake = 10 :=
by {
  sorry
}

end garden_snake_is_10_inches_l92_92489


namespace books_difference_l92_92585

theorem books_difference (bobby_books : ℕ) (kristi_books : ℕ) (h1 : bobby_books = 142) (h2 : kristi_books = 78) : bobby_books - kristi_books = 64 :=
by {
  -- Placeholder for the proof
  sorry
}

end books_difference_l92_92585


namespace regular_hexagon_same_length_probability_l92_92169

theorem regular_hexagon_same_length_probability :
  let T : Finset (Finite_Field) := sorry, -- Define the actual set T una don the constructible geometry module in lean mathematical library
  let total_segments := 15,
  let sides := 6,
  let shorter_diagonals := 6,
  let longer_diagonals := 3,
  let prob_side := (6 / 15) * (5 / 14),
  let prob_shorter := (6 / 15) * (5 / 14),
  let prob_longer := (3 / 15) * (2 / 14),
  let total_prob := (prob_side + prob_shorter + prob_longer),
  (total_prob = (11 / 35)) := sorry

end regular_hexagon_same_length_probability_l92_92169


namespace profit_percentage_with_discount_l92_92483

theorem profit_percentage_with_discount
    (P M : ℝ)
    (h1 : M = 1.27 * P)
    (h2 : 0 < P) :
    ((0.95 * M - P) / P) * 100 = 20.65 :=
by
  sorry

end profit_percentage_with_discount_l92_92483


namespace factorization_identity_l92_92332

-- We are asked to prove the mathematical equality under given conditions.
theorem factorization_identity (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = (x^2 + 6*x + 8)^2 :=
by
  sorry

end factorization_identity_l92_92332


namespace regular_hexagon_same_length_probability_l92_92170

theorem regular_hexagon_same_length_probability :
  let T : Finset (Finite_Field) := sorry, -- Define the actual set T una don the constructible geometry module in lean mathematical library
  let total_segments := 15,
  let sides := 6,
  let shorter_diagonals := 6,
  let longer_diagonals := 3,
  let prob_side := (6 / 15) * (5 / 14),
  let prob_shorter := (6 / 15) * (5 / 14),
  let prob_longer := (3 / 15) * (2 / 14),
  let total_prob := (prob_side + prob_shorter + prob_longer),
  (total_prob = (11 / 35)) := sorry

end regular_hexagon_same_length_probability_l92_92170


namespace Aiyanna_has_more_cookies_l92_92732

theorem Aiyanna_has_more_cookies (Alyssa_cookies : ℕ) (Aiyanna_cookies : ℕ) (hAlyssa : Alyssa_cookies = 129) (hAiyanna : Aiyanna_cookies = 140) : Aiyanna_cookies - Alyssa_cookies = 11 := 
by sorry

end Aiyanna_has_more_cookies_l92_92732


namespace quadrilateral_impossible_l92_92186

theorem quadrilateral_impossible (a b c d : ℕ) (h1 : 2 * a ^ 2 - 18 * a + 36 = 0)
    (h2 : b ^ 2 - 20 * b + 75 = 0) (h3 : c ^ 2 - 20 * c + 75 = 0) (h4 : 2 * d ^ 2 - 18 * d + 36 = 0) :
    ¬(a + b > d ∧ a + c > d ∧ b + c > d ∧ a + d > c ∧ b + d > c ∧ c + d > b ∧
      a + b + c > d ∧ a + b + d > c ∧ a + c + d > b ∧ b + c + d > a) :=
by
  sorry

end quadrilateral_impossible_l92_92186


namespace edward_books_bought_l92_92914

def money_spent : ℕ := 6
def cost_per_book : ℕ := 3

theorem edward_books_bought : money_spent / cost_per_book = 2 :=
by
  sorry

end edward_books_bought_l92_92914


namespace inequality_part1_inequality_part2_l92_92846

section Proof

variable {x m : ℝ}
def f (x : ℝ) : ℝ := |2 * x + 2| + |2 * x - 3|

-- Part 1: Prove the solution set for the inequality f(x) > 7
theorem inequality_part1 (x : ℝ) :
  f x > 7 ↔ (x < -3 / 2 ∨ x > 2) := 
  sorry

-- Part 2: Prove the range of values for m such that the inequality f(x) ≤ |3m - 2| has a solution
theorem inequality_part2 (m : ℝ) :
  (∃ x, f x ≤ |3 * m - 2|) ↔ (m ≤ -1 ∨ m ≥ 7 / 3) := 
  sorry

end Proof

end inequality_part1_inequality_part2_l92_92846


namespace probability_of_not_all_same_number_l92_92046

theorem probability_of_not_all_same_number :
  ∀ (D : ℕ) (n : ℕ),
  D = 8 → n = 5 → 
  let total_outcomes := D^n in
  let same_outcome_probability := D / total_outcomes  in
  let not_all_same_probability := 1 - same_outcome_probability in
  (not_all_same_probability = 4095 / 4096) :=
begin
  intros D n D_eq n_eq,
  simp only [D_eq, n_eq],
  let total_outcomes := 8^5,
  let same_outcome_probability := 8 / total_outcomes,
  let not_all_same_probability := 1 - same_outcome_probability,
  have total_outcome_calc : total_outcomes = 8^5 := by sorry,
  have same_outcome_prob_calc : same_outcome_probability = 1 / 4096 := by sorry,
  have not_all_same_prob_calc : not_all_same_probability = 4095 / 4096 := by sorry,
  exact not_all_same_prob_calc,
end

end probability_of_not_all_same_number_l92_92046


namespace probability_not_all_dice_show_different_l92_92063

noncomputable def probability_not_all_same : ℚ :=
  let total_outcomes := 8^5
  let same_number_outcomes := 8
  (total_outcomes - same_number_outcomes) / total_outcomes

theorem probability_not_all_dice_show_different : 
  probability_not_all_same = 4095 / 4096 := by
  sorry

end probability_not_all_dice_show_different_l92_92063


namespace defective_pens_l92_92131

theorem defective_pens :
  ∃ D N : ℕ, (N + D = 9) ∧ (N / 9 * (N - 1) / 8 = 5 / 12) ∧ (D = 3) :=
by
  sorry

end defective_pens_l92_92131


namespace birches_count_l92_92942

-- Define the problem conditions
def total_trees : ℕ := 4000
def percentage_spruces : ℕ := 10
def percentage_pines : ℕ := 13
def number_spruces : ℕ := (percentage_spruces * total_trees) / 100
def number_pines : ℕ := (percentage_pines * total_trees) / 100
def number_oaks : ℕ := number_spruces + number_pines
def number_birches : ℕ := total_trees - number_oaks - number_pines - number_spruces

-- Prove the number of birches is 2160
theorem birches_count : number_birches = 2160 := by
  sorry

end birches_count_l92_92942


namespace coefficient_a2b2_in_expansion_l92_92458

theorem coefficient_a2b2_in_expansion :
  -- Combining the coefficients: \binom{4}{2} and \binom{6}{3}
  (Nat.choose 4 2) * (Nat.choose 6 3) = 120 :=
by
  -- No proof required, using sorry to indicate that.
  sorry

end coefficient_a2b2_in_expansion_l92_92458


namespace range_of_t_circle_largest_area_eq_point_P_inside_circle_l92_92355

open Real

-- Defining the given equation representing the trajectory of a point on a circle
def circle_eq (x y t : ℝ) : Prop :=
  x^2 + y^2 - 2 * (t + 3) * x + 2 * (1 - 4 * t^2) * y + 16 * t^4 + 9 = 0

-- Problem 1: Proving the range of t
theorem range_of_t : 
  ∀ t : ℝ, (∃ x y : ℝ, circle_eq x y t) → -1/7 < t ∧ t < 1 :=
sorry

-- Problem 2: Proving the equation of the circle with the largest area
theorem circle_largest_area_eq : 
  ∃ t : ℝ, t = 3/7 ∧ (∀ x y : ℝ, circle_eq x y (3/7)) → 
  ∀ x y : ℝ, (x - 24/7)^2 + (y + 13/49)^2 = 16/7 :=
sorry

-- Problem 3: Proving the range of t for point P to be inside the circle
theorem point_P_inside_circle : 
  ∀ t : ℝ, (∃ x y : ℝ, circle_eq x y t) → 
  (0 < t ∧ t < 3/4) :=
sorry

end range_of_t_circle_largest_area_eq_point_P_inside_circle_l92_92355


namespace cubic_and_quintic_values_l92_92388

theorem cubic_and_quintic_values (a : ℝ) (h : (a + 1/a)^2 = 11) : 
    (a^3 + 1/a^3 = 8 * Real.sqrt 11 ∧ a^5 + 1/a^5 = 71 * Real.sqrt 11) ∨ 
    (a^3 + 1/a^3 = -8 * Real.sqrt 11 ∧ a^5 + 1/a^5 = -71 * Real.sqrt 11) :=
by
  sorry

end cubic_and_quintic_values_l92_92388


namespace max_sum_condition_l92_92214

theorem max_sum_condition (a b : ℕ) (h1 : 10 ≤ a ∧ a ≤ 99) (h2 : 10 ≤ b ∧ b ≤ 99) (h3 : Nat.gcd a b = 6) : a + b ≤ 186 :=
sorry

end max_sum_condition_l92_92214


namespace ellipse_slope_ratio_l92_92380

theorem ellipse_slope_ratio (a b : ℝ) (k1 k2 : ℝ) 
  (h1 : a > b)
  (h2 : b > 0)
  (h3 : a > 2)
  (h4 : k2 = k1 * (a^2 + 5) / (a^2 - 1)) : 
  1 < (k2 / k1) ∧ (k2 / k1) < 3 :=
by
  sorry

end ellipse_slope_ratio_l92_92380


namespace log_sqrt_defined_l92_92634

theorem log_sqrt_defined (x : ℝ) : 2 < x ∧ x < 5 ↔ (∃ y z : ℝ, y = log (5 - x) ∧ z = sqrt (x - 2)) :=
by 
  sorry

end log_sqrt_defined_l92_92634


namespace log_eq_solution_l92_92868

theorem log_eq_solution (a x : ℝ) (h1 : 0 < a) (h2 : a ≠ 1)
  (h3 : -4 < x ∧ x < 4)
  (h4 : log a (sqrt (4 + x)) + 3 * log (a^2) (4 - x) - log (a^4) ((16 - x^2)^2) = 2)
  : a ∈ Ioo 0 1 ∨ a ∈ Ioo 1 (2 * Real.sqrt 2) :=
sorry

end log_eq_solution_l92_92868


namespace polynomial_factorization_l92_92316

theorem polynomial_factorization (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = (x^2 + 6*x + 1) * (x^2 + 6*x + 37) :=
  sorry

end polynomial_factorization_l92_92316


namespace yellow_chip_count_l92_92534

def point_values_equation (Y B G R : ℕ) : Prop :=
  2 ^ Y * 4 ^ B * 5 ^ G * 7 ^ R = 560000

theorem yellow_chip_count (Y B G R : ℕ) (h1 : B = 2 * G) (h2 : R = B / 2) (h3 : point_values_equation Y B G R) :
  Y = 2 :=
by
  sorry

end yellow_chip_count_l92_92534


namespace sweets_ratio_l92_92989

theorem sweets_ratio (number_orange_sweets : ℕ) (number_grape_sweets : ℕ) (max_sweets_per_tray : ℕ)
  (h1 : number_orange_sweets = 36) (h2 : number_grape_sweets = 44) (h3 : max_sweets_per_tray = 4) :
  (number_orange_sweets / max_sweets_per_tray) / (number_grape_sweets / max_sweets_per_tray) = 9 / 11 :=
by
  sorry

end sweets_ratio_l92_92989


namespace probability_of_same_length_l92_92165

-- Define the set T and its properties
def T : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
def sides : ℕ := 6
def diagonals : ℕ := 9
def total_elements : ℕ := 15
def probability_same_length : ℚ := 17 / 35

-- Lean 4 statement (theorem)
theorem probability_of_same_length: 
    ( ∃ (sides diagonals : ℕ), sides = 6 ∧ diagonals = 9 ∧ sides + diagonals = 15 →
                                              ∃ probability_same_length : ℚ, probability_same_length = 17 / 35) := 
sorry

end probability_of_same_length_l92_92165


namespace min_value_of_a_l92_92645

theorem min_value_of_a 
  {f : ℕ → ℝ} 
  (h : ∀ x : ℕ, 0 < x → f x = (x^2 + a * x + 11) / (x + 1)) 
  (ineq : ∀ x : ℕ, 0 < x → f x ≥ 3) : a ≥ -8 / 3 :=
sorry

end min_value_of_a_l92_92645


namespace Tyler_scissors_count_l92_92701

variable (S : ℕ)

def Tyler_initial_money : ℕ := 100
def cost_per_scissors : ℕ := 5
def number_of_erasers : ℕ := 10
def cost_per_eraser : ℕ := 4
def Tyler_remaining_money : ℕ := 20

theorem Tyler_scissors_count :
  Tyler_initial_money - (cost_per_scissors * S + number_of_erasers * cost_per_eraser) = Tyler_remaining_money →
  S = 8 :=
by
  sorry

end Tyler_scissors_count_l92_92701


namespace number_of_articles_l92_92653

-- Define the conditions
def gain := 1 / 9
def cp_one_article := 1  -- cost price of one article

-- Define the cost price for x articles
def cp (x : ℕ) := x * cp_one_article

-- Define the selling price for 45 articles
def sp (x : ℕ) := x / 45

-- Define the selling price equation considering gain
def sp_one_article := (cp_one_article * (1 + gain))

-- Main theorem to prove
theorem number_of_articles (x : ℕ) (h : sp x = sp_one_article) : x = 50 :=
by
  sorry

-- The theorem imports all necessary conditions and definitions and prepares the problem for proof.

end number_of_articles_l92_92653


namespace top_card_is_heartsuit_probability_l92_92731

-- Definitions of conditions
def total_ranks : ℕ := 13
def total_suits : ℕ := 4
def total_cards : ℕ := total_ranks * total_suits
def heartsuit_cards : ℕ := total_ranks
def probability_of_heartsuit : ℚ := heartsuit_cards / total_cards

-- Theorem to prove the question equals the answer given the conditions
theorem top_card_is_heartsuit_probability : probability_of_heartsuit = 1 / 4 := by
  -- Proof omitted
  sorry

end top_card_is_heartsuit_probability_l92_92731


namespace gnomes_telling_the_truth_l92_92419

-- Conditions
def gnome_height_claims : List ℕ := [60, 61, 62, 63, 64, 65, 66]

-- Question and Proof Problem in Lean 4
theorem gnomes_telling_the_truth :
  (∀ (actual_heights : List ℕ), 
    actual_heights.length = 7 →
    (∀ i, i < 7 → i > 0 → 
    actual_heights.get! i > actual_heights.get! (i - 1) → gnome_height_claims.get! i ≠ actual_heights.get! i)) →
  -- conclusion
  (∃ count, count = 1) :=
by
  sorry

end gnomes_telling_the_truth_l92_92419


namespace largest_x_satisfying_equation_l92_92032

theorem largest_x_satisfying_equation :
  ∃ x : ℝ, x = 3 / 25 ∧ (∀ y, (y : ℝ) ∈ {z | sqrt (3 * z) = 5 * z} → y ≤ x) :=
by
  sorry

end largest_x_satisfying_equation_l92_92032


namespace expression_value_l92_92994

theorem expression_value : (25 + 15)^2 - (25^2 + 15^2) = 750 := by
  sorry

end expression_value_l92_92994


namespace sum_of_three_numbers_l92_92443

theorem sum_of_three_numbers (x : ℝ) (a b c : ℝ) (h1 : a = 5 * x) (h2 : b = x) (h3 : c = 4 * x) (h4 : c = 400) :
  a + b + c = 1000 := by
  sorry

end sum_of_three_numbers_l92_92443


namespace restoration_of_axes_l92_92175

theorem restoration_of_axes (parabola : ℝ → ℝ) (h : ∀ x, parabola x = x^2) : 
  ∃ (origin : ℝ × ℝ) (x_axis y_axis : ℝ × ℝ → Prop), 
    (∀ x, x_axis (x, 0)) ∧ 
    (∀ y, y_axis (0, y)) ∧ 
    origin = (0, 0) := 
sorry

end restoration_of_axes_l92_92175


namespace hh_of_2_eq_91265_l92_92956

def h (x : ℝ) : ℝ := 3 * x^3 + 2 * x^2 - x + 1

theorem hh_of_2_eq_91265 : h (h 2) = 91265 := by
  sorry

end hh_of_2_eq_91265_l92_92956


namespace august_8th_is_saturday_l92_92372

-- Defining the conditions
def august_has_31_days : Prop := true

def august_has_5_mondays : Prop := true

def august_has_4_tuesdays : Prop := true

-- Statement of the theorem
theorem august_8th_is_saturday (h1 : august_has_31_days) (h2 : august_has_5_mondays) (h3 : august_has_4_tuesdays) : ∃ d : ℕ, d = 6 :=
by
  -- Translate the correct answer "August 8th is a Saturday" into the equivalent proposition
  -- Saturday is represented by 6 if we assume 0 = Sunday, 1 = Monday, ..., 6 = Saturday.
  sorry

end august_8th_is_saturday_l92_92372


namespace initial_population_of_town_l92_92978

theorem initial_population_of_town 
  (final_population : ℝ) 
  (growth_rate : ℝ) 
  (years : ℕ) 
  (initial_population : ℝ) 
  (h : final_population = initial_population * (1 + growth_rate) ^ years) : 
  initial_population = 297500 / (1 + 0.07) ^ 10 :=
by
  sorry

end initial_population_of_town_l92_92978


namespace consistent_values_l92_92101

theorem consistent_values (a x: ℝ) :
    (12 * x^2 + 48 * x - a + 36 = 0) ∧ ((a + 60) * x - 3 * (a - 20) = 0) ↔
    ((a = -12 ∧ x = -2) ∨ (a = 0 ∧ x = -1) ∨ (a = 180 ∧ x = 2)) := 
by
  -- proof steps should be filled here
  sorry

end consistent_values_l92_92101


namespace last_digit_x4_plus_inv_x4_l92_92555

theorem last_digit_x4_plus_inv_x4 (x : ℝ) (h : x^2 - 13 * x + 1 = 0) : (x^4 + (1 / x)^4) % 10 = 7 := 
by
  sorry

end last_digit_x4_plus_inv_x4_l92_92555


namespace probability_x_greater_8y_l92_92968

theorem probability_x_greater_8y :
  let rectangle := {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 3014 ∧ 0 ≤ p.2 ∧ p.2 ≤ 3015}
  let area_rectangle := (3014 : ℝ) * 3015
  let f : set (ℝ × ℝ) := {p | p.1 > 8 * p.2}
  let rectangle_triangle := {p | p.2 < p.1 / 8 ∧ p.1 ≤ 3014}
  let area_triangle := 1 / 2 * 3014 * 376.75
  (area_triangle / area_rectangle = 7535 / 120600) :=
by
  sorry

end probability_x_greater_8y_l92_92968


namespace birches_count_l92_92941

-- Define the problem conditions
def total_trees : ℕ := 4000
def percentage_spruces : ℕ := 10
def percentage_pines : ℕ := 13
def number_spruces : ℕ := (percentage_spruces * total_trees) / 100
def number_pines : ℕ := (percentage_pines * total_trees) / 100
def number_oaks : ℕ := number_spruces + number_pines
def number_birches : ℕ := total_trees - number_oaks - number_pines - number_spruces

-- Prove the number of birches is 2160
theorem birches_count : number_birches = 2160 := by
  sorry

end birches_count_l92_92941


namespace minimum_A2_minus_B2_l92_92957

noncomputable def A (x y z : ℝ) : ℝ := 
  Real.sqrt (x + 6) + Real.sqrt (y + 7) + Real.sqrt (z + 12)

noncomputable def B (x y z : ℝ) : ℝ :=
  Real.sqrt (x + 2) + Real.sqrt (y + 3) + Real.sqrt (z + 5)

theorem minimum_A2_minus_B2 (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) : 
  (A x y z)^2 - (B x y z)^2 = 49.25 := 
by 
  sorry 

end minimum_A2_minus_B2_l92_92957


namespace sheep_to_horses_ratio_l92_92908

-- Define the known quantities
def number_of_sheep := 32
def total_horse_food := 12880
def food_per_horse := 230

-- Calculate number of horses
def number_of_horses := total_horse_food / food_per_horse

-- Calculate and simplify the ratio of sheep to horses
def ratio_of_sheep_to_horses := (number_of_sheep : ℚ) / (number_of_horses : ℚ)

-- Define the expected simplified ratio
def expected_ratio_of_sheep_to_horses := (4 : ℚ) / (7 : ℚ)

-- The statement we want to prove
theorem sheep_to_horses_ratio : ratio_of_sheep_to_horses = expected_ratio_of_sheep_to_horses :=
by
  -- Proof will be here
  sorry

end sheep_to_horses_ratio_l92_92908


namespace find_divisor_l92_92398

theorem find_divisor (n d k : ℤ) (h1 : n = k * d + 3) (h2 : n^2 % d = 4) : d = 5 :=
by
  sorry

end find_divisor_l92_92398


namespace probability_not_all_same_l92_92038

theorem probability_not_all_same (n m : ℕ) (h₁ : n = 5) (h₂ : m = 8) 
  (fair_dice : ∀ (die : ℕ), 1 ≤ die ∧ die ≤ m)
  : (1 - (m / m^n) = 4095 / 4096) := by
  sorry

end probability_not_all_same_l92_92038


namespace correct_average_l92_92875

-- Define the conditions given in the problem
def avg_incorrect : ℕ := 46 -- incorrect average
def n : ℕ := 10 -- number of values
def incorrect_num : ℕ := 25
def correct_num : ℕ := 75
def diff : ℕ := correct_num - incorrect_num

-- Define the total sums
def total_incorrect : ℕ := avg_incorrect * n
def total_correct : ℕ := total_incorrect + diff

-- Define the correct average
def avg_correct : ℕ := total_correct / n

-- Statement in Lean 4
theorem correct_average :
  avg_correct = 51 :=
by
  -- We expect users to fill the proof here
  sorry

end correct_average_l92_92875


namespace cows_count_l92_92873

theorem cows_count (D C : ℕ) (h1 : 2 * (D + C) + 32 = 2 * D + 4 * C) : C = 16 :=
by
  sorry

end cows_count_l92_92873


namespace ivan_needs_more_paint_l92_92456

theorem ivan_needs_more_paint
  (section_count : ℕ)
  (α : ℝ)
  (hα : 0 < α ∧ α < π / 2) :
  let area_ivan := section_count * (5 * 2)
  let area_petr := section_count * (5 * 2 * sin α)
  area_ivan > area_petr := 
by
  simp only [area_ivan, area_petr, mul_assoc, mul_lt_mul_left, gt_iff_lt]
  exact sin_lt_one_iff.mpr hα

end ivan_needs_more_paint_l92_92456


namespace frog_jump_correct_l92_92433

def grasshopper_jump : ℤ := 25
def additional_distance : ℤ := 15
def frog_jump : ℤ := grasshopper_jump + additional_distance

theorem frog_jump_correct : frog_jump = 40 := by
  sorry

end frog_jump_correct_l92_92433


namespace remainder_of_division_l92_92578

theorem remainder_of_division (x r : ℕ) (h : 23 = 7 * x + r) : r = 2 :=
sorry

end remainder_of_division_l92_92578


namespace rectangle_area_constant_l92_92192

noncomputable def k (d : ℝ) : ℝ :=
  let x := d / Real.sqrt 29
  10 / 29

theorem rectangle_area_constant (d : ℝ) : 
  let k := 10 / 29
  let length := 5 * (d / Real.sqrt 29)
  let width := 2 * (d / Real.sqrt 29)
  let diagonal := d
  let area := length * width
  area = k * d^2 :=
by
  sorry

end rectangle_area_constant_l92_92192


namespace factor_poly_eq_factored_form_l92_92267

-- Defining the polynomial expressions
def poly1 (x : ℝ) := x^2 + 4 * x + 3
def poly2 (x : ℝ) := x^2 + 8 * x + 15
def poly3 (x : ℝ) := x^2 + 6 * x - 8

-- The main expression which needs to be factored
def main_expr (x : ℝ) := (poly1 x) * (poly2 x) + (poly3 x)

-- Stating the goal factored form
def factored_form (x : ℝ) := (x^2 + 6 * x + 19) * (x^2 + 6 * x - 2)

-- The theorem statement
theorem factor_poly_eq_factored_form (x : ℝ) : 
  main_expr x = factored_form x := 
by
  sorry

end factor_poly_eq_factored_form_l92_92267


namespace lines_region_division_l92_92377

theorem lines_region_division (f : ℕ → ℕ) (k : ℕ) (h : k ≥ 2) : 
  (∀ m, f m = m * (m + 1) / 2 + 1) → f (k + 1) = f k + (k + 1) :=
by
  intro h_f
  have h_base : f 1 = 2 := by sorry
  have h_ih : ∀ n, n ≥ 2 → f (n + 1) = f n + (n + 1) := by sorry
  exact h_ih k h

end lines_region_division_l92_92377


namespace f_increasing_on_positive_l92_92674

noncomputable def f (x : ℝ) : ℝ := - (1 / x) - 1

theorem f_increasing_on_positive (x1 x2 : ℝ) (h1 : 0 < x1) (h2 : 0 < x2) (h3 : x1 > x2) : f x1 > f x2 := by
  sorry

end f_increasing_on_positive_l92_92674


namespace birch_trees_count_l92_92939

-- Definitions based on the conditions
def total_trees : ℕ := 4000
def percentage_spruce : ℕ := 10
def percentage_pine : ℕ := 13

def count_spruce : ℕ := (percentage_spruce * total_trees) / 100
def count_pine : ℕ := (percentage_pine * total_trees) / 100
def count_oak : ℕ := count_spruce + count_pine

def count_birch : ℕ := total_trees - (count_spruce + count_pine + count_oak)

-- The theorem to be proven
theorem birch_trees_count :
  count_birch = 2160 := by
  sorry

end birch_trees_count_l92_92939


namespace prob_at_least_3_speak_l92_92862

-- Define the probability of any baby speaking
def prob_speaking := 1 / 3

-- Define the binomial coefficient
noncomputable def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Define the probability of at least 3 out of 6 babies speaking the next day
theorem prob_at_least_3_speak (P: ℝ) (n: ℕ) (k1: ℕ) (k2: ℕ):
  P = prob_speaking ∧ n = 6 ∧ k1 = 3 ∧ k2 = 2 ->
  (1 - ((2/3)^n + binom n 1 * (P)*(2/3)^(n-1) + binom n k2 * (P)^2 * (2/3)^(n-k2))) = 233 / 729 := sorry

end prob_at_least_3_speak_l92_92862


namespace binomial_8_4_eq_70_l92_92494

theorem binomial_8_4_eq_70 : Nat.binom 8 4 = 70 := by
  sorry

end binomial_8_4_eq_70_l92_92494


namespace power_function_nature_l92_92979

def f (x : ℝ) : ℝ := x ^ (1/2)

theorem power_function_nature:
  (f 3 = Real.sqrt 3) ∧
  (¬ (∀ x, f (-x) = f x)) ∧
  (¬ (∀ x, f (-x) = -f x)) ∧
  (∀ x, 0 < x → 0 < f x) := 
by
  sorry

end power_function_nature_l92_92979


namespace hawks_loss_percentage_is_30_l92_92444

-- Define the variables and the conditions
def matches_won (x : ℕ) : ℕ := 7 * x
def matches_lost (x : ℕ) : ℕ := 3 * x
def total_matches (x : ℕ) : ℕ := matches_won x + matches_lost x
def percent_lost (x : ℕ) : ℕ := (matches_lost x * 100) / total_matches x

-- The goal statement in Lean 4
theorem hawks_loss_percentage_is_30 (x : ℕ) (h : x > 0) : percent_lost x = 30 :=
by sorry

end hawks_loss_percentage_is_30_l92_92444


namespace min_value_expression_l92_92103

theorem min_value_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 1) :
  (∃ c : ℝ, c = (1 / (2 * x) + x / (y + 1)) ∧ c = 5 / 4) :=
sorry

end min_value_expression_l92_92103


namespace binom_eight_four_l92_92491

theorem binom_eight_four : (Nat.choose 8 4) = 70 :=
by
  sorry

end binom_eight_four_l92_92491


namespace inequality_xyz_geq_3_l92_92401

theorem inequality_xyz_geq_3
  (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (h_not_all_zero : x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0) :
  (2 * x^2 - x + y + z) / (x + y^2 + z^2) +
  (2 * y^2 + x - y + z) / (x^2 + y + z^2) +
  (2 * z^2 + x + y - z) / (x^2 + y^2 + z) ≥ 3 := 
sorry

end inequality_xyz_geq_3_l92_92401


namespace factor_polynomial_l92_92278

-- Define the necessary polynomials
def p (x : ℝ) : ℝ := x^2 + 4*x + 3
def q (x : ℝ) : ℝ := x^2 + 8*x + 15
def r (x : ℝ) : ℝ := x^2 + 6*x - 8

-- State the main theorem
theorem factor_polynomial (x : ℝ) : 
  (p x * q x) + r x = (x^2 + 7*x + 1) * (x^2 + 4*x + 7) :=
by
  sorry

end factor_polynomial_l92_92278


namespace direction_vector_is_3_1_l92_92429

-- Given the line equation x - 3y + 1 = 0
def line_equation : ℝ × ℝ → Prop :=
  λ p, p.1 - 3 * p.2 + 1 = 0

-- The direction vector of the line
def direction_vector_of_line (v : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v = (k * 3, k * 1)

theorem direction_vector_is_3_1 : direction_vector_of_line (3, 1) :=
by
  sorry

end direction_vector_is_3_1_l92_92429


namespace factor_polynomial_l92_92307

theorem factor_polynomial :
  (x^2 + 4 * x + 3) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8) = 
  (x^2 + 6 * x + 7) * (x^2 + 6 * x + 8) :=
sorry

end factor_polynomial_l92_92307


namespace factor_polynomial_l92_92297

theorem factor_polynomial :
  (x^2 + 4 * x + 3) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8) = 
  (x^2 + 6 * x + 7) * (x^2 + 6 * x + 8) :=
sorry

end factor_polynomial_l92_92297


namespace simplify_expression_l92_92576

-- Define the hypotheses and the expression.
variables (x : ℚ)
def expr := (1 + 1 / x) * (1 - 2 / (x + 1)) * (1 + 2 / (x - 1))

-- Define the conditions.
def valid_x : Prop := (x ≠ 0) ∧ (x ≠ -1) ∧ (x ≠ 1)

-- State the main theorem.
theorem simplify_expression (h : valid_x x) : expr x = (x + 1) / x := 
sorry

end simplify_expression_l92_92576


namespace find_term_ninth_term_l92_92447

variable (a_1 d a_k a_12 : ℤ)
variable (S_20 : ℤ := 200)

-- Definitions of the given conditions
def term_n (a_1 : ℤ) (d : ℤ) (n : ℤ) : ℤ := a_1 + (n - 1) * d

-- Problem Statement
theorem find_term_ninth_term :
  (∃ k, term_n a_1 d k + term_n a_1 d 12 = 20) ∧ 
  (S_20 = 10 * (2 * a_1 + 19 * d)) → 
  ∃ k, k = 9 :=
by sorry

end find_term_ninth_term_l92_92447


namespace max_and_min_of_z_in_G_l92_92758

def z (x y : ℝ) : ℝ := x^2 + y^2 - 2*x*y - x - 2*y

def G (x y : ℝ) : Prop := x ≥ 0 ∧ y ≥ 0 ∧ x + y ≤ 4

theorem max_and_min_of_z_in_G :
  (∃ (x y : ℝ), G x y ∧ z x y = 12) ∧ (∃ (x y : ℝ), G x y ∧ z x y = -1/4) :=
sorry

end max_and_min_of_z_in_G_l92_92758


namespace find_ab_l92_92116

theorem find_ab (a b : ℝ) (h1 : a + b = 5) (h2 : a^3 + b^3 = 125) : a * b = 0 :=
by
  sorry

end find_ab_l92_92116


namespace quadratic_function_min_value_l92_92106

theorem quadratic_function_min_value (a b c : ℝ) (h_a : a > 0) (h_b : b ≠ 0) 
(h_f0 : |c| = 1) (h_f1 : |a + b + c| = 1) (h_fn1 : |a - b + c| = 1) :
∃ f : ℝ → ℝ, (∀ x : ℝ, f x = a*x^2 + b*x + c) ∧
  (|f 0| = 1) ∧ (|f 1| = 1) ∧ (|f (-1)| = 1) ∧
  (f 0 = -(5/4) ∨ f 1 = -(5/4) ∨ f (-1) = -(5/4)) :=
by
  sorry

end quadratic_function_min_value_l92_92106


namespace f_derivative_at_1_intervals_of_monotonicity_l92_92925

def f (x : ℝ) := x^3 - 3 * x^2 + 10
def f' (x : ℝ) := 3 * x^2 - 6 * x

theorem f_derivative_at_1 : f' 1 = -3 := by
  sorry

theorem intervals_of_monotonicity :
  (∀ x : ℝ, x < 0 → f' x > 0) ∧
  (∀ x : ℝ, 0 < x ∧ x < 2 → f' x < 0) ∧
  (∀ x : ℝ, x > 2 → f' x > 0) := by
  sorry

end f_derivative_at_1_intervals_of_monotonicity_l92_92925


namespace product_is_correct_l92_92712

-- Define the numbers a and b
def a : ℕ := 72519
def b : ℕ := 9999

-- Theorem statement that proves the correctness of the product
theorem product_is_correct : a * b = 725117481 :=
by
  sorry

end product_is_correct_l92_92712


namespace nancy_total_money_l92_92671

theorem nancy_total_money (n : ℕ) (d : ℕ) (h1 : n = 9) (h2 : d = 5) : n * d = 45 := 
by
  sorry

end nancy_total_money_l92_92671


namespace sequence_filling_l92_92787

theorem sequence_filling :
  ∃ (a : Fin 8 → ℕ), 
    a 0 = 20 ∧ 
    a 7 = 16 ∧ 
    (∀ i : Fin 6, a i + a (i+1) + a (i+2) = 100) ∧ 
    (a 1 = 16) ∧ 
    (a 2 = 64) ∧ 
    (a 3 = 20) ∧ 
    (a 4 = 16) ∧ 
    (a 5 = 64) ∧ 
    (a 6 = 20) := 
by
  sorry

end sequence_filling_l92_92787


namespace number_of_snakes_l92_92779

-- Define the variables
variable (S : ℕ) -- Number of snakes

-- Define the cost constants
def cost_per_gecko := 15
def cost_per_iguana := 5
def cost_per_snake := 10

-- Define the number of each pet
def num_geckos := 3
def num_iguanas := 2

-- Define the yearly cost
def yearly_cost := 1140

-- Calculate the total monthly cost
def monthly_cost := num_geckos * cost_per_gecko + num_iguanas * cost_per_iguana + S * cost_per_snake

-- Calculate the total yearly cost
def total_yearly_cost := 12 * monthly_cost

-- Prove the number of snakes
theorem number_of_snakes : total_yearly_cost = yearly_cost → S = 4 := by
  sorry

end number_of_snakes_l92_92779


namespace probability_not_bought_by_Jim_l92_92893

open Finset

theorem probability_not_bought_by_Jim
  (total_pictures : ℕ) (bought_pictures : ℕ) (pick_pictures : ℕ)
  (h_total : total_pictures = 10) (h_bought : bought_pictures = 3) (h_pick : pick_pictures = 2) :
  (choose (total_pictures - bought_pictures) pick_pictures) / (choose total_pictures pick_pictures) = (7 / 15 : ℚ) :=
by
  sorry

end probability_not_bought_by_Jim_l92_92893


namespace pyramid_base_edge_length_l92_92683

theorem pyramid_base_edge_length
  (hemisphere_radius : ℝ) (pyramid_height : ℝ) (slant_height : ℝ) (is_tangent: Prop) :
  hemisphere_radius = 3 ∧ pyramid_height = 8 ∧ slant_height = 10 ∧ is_tangent →
  ∃ (base_edge_length : ℝ), base_edge_length = 6 * Real.sqrt 2 :=
by
  sorry

end pyramid_base_edge_length_l92_92683


namespace fraction_of_number_l92_92551

theorem fraction_of_number (x : ℕ) (f : ℚ) (h1 : x = 140) (h2 : 0.65 * x = f * x - 21) : f = 0.8 :=
sorry

end fraction_of_number_l92_92551


namespace exists_n_satisfying_conditions_l92_92548

open Nat

-- Define that n satisfies the given conditions
theorem exists_n_satisfying_conditions :
  ∃ (n : ℤ), (∃ (k : ℤ), 2 * n + 1 = (2 * k + 1) ^ 2) ∧ 
            (∃ (h : ℤ), 3 * n + 1 = (2 * h + 1) ^ 2) ∧ 
            (40 ∣ n) := by
  sorry

end exists_n_satisfying_conditions_l92_92548


namespace factorization_correct_l92_92286

noncomputable def polynomial_expr := 
  λ x : ℝ => (x^2 + 4 * x + 3) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8)

noncomputable def factored_expr := 
  λ x : ℝ => (x^2 + 6 * x + 9) * (x^2 + 6 * x + 1)

theorem factorization_correct : 
  ∀ x : ℝ, polynomial_expr x = factored_expr x :=
by
  intro x
  sorry

end factorization_correct_l92_92286


namespace curve_eq_and_max_distance_l92_92945

theorem curve_eq_and_max_distance
  (x y θ : ℝ)
  (h1 : x = sqrt 3 * Real.cos θ)
  (h2 : y = Real.sin θ)
  (h_line : ∀ ρ, ρ * Real.cos (θ - π / 4) = 2 * sqrt 2 → x = ρ * cos θ ∧ y = ρ * sin θ) :
  (x^2 / 3 + y^2 = 1) ∧
  (x + y - 4 = 0) ∧
  (∀ P : ℝ, 
    P = (sqrt 3 * Real.cos θ, Real.sin θ) → 
    let d := abs (sqrt 3 * Real.cos θ + Real.sin θ - 4) / sqrt 2 in 
    ∃ k : ℤ, d = 3 * sqrt 2) :=
  sorry

end curve_eq_and_max_distance_l92_92945


namespace cos_arcsin_l92_92745

theorem cos_arcsin (x : ℝ) (hx : x = 3 / 5) : Real.cos (Real.arcsin x) = 4 / 5 := by
  sorry

end cos_arcsin_l92_92745


namespace line_intersects_parabola_exactly_once_at_m_l92_92686

theorem line_intersects_parabola_exactly_once_at_m :
  (∃ y : ℝ, -3 * y^2 - 4 * y + 7 = m) → (∃! m : ℝ, m = 25 / 3) :=
by
  intro h
  sorry

end line_intersects_parabola_exactly_once_at_m_l92_92686


namespace max_irreducible_fractions_in_interval_l92_92350

open Real

-- The main theorem statement
theorem max_irreducible_fractions_in_interval (n : ℕ) (h_pos : n > 0) : 
  ∀ I : set ℝ, 
    (∃ a b : ℝ, I = {x | a < x ∧ x < b} ∧ b - a = 1 / (n : ℝ)) → 
    ∃ s : finset (ℚ), s.card ≤ (n + 1) / 2 ∧ ∀ q ∈ s, (q.den ∈ (set.Icc 1 n)) ∧ ((q : ℝ) ∈ I) :=
sorry

end max_irreducible_fractions_in_interval_l92_92350


namespace average_velocity_eq_l92_92122

noncomputable def motion_eq : ℝ → ℝ := λ t => 1 - t + t^2

theorem average_velocity_eq (Δt : ℝ) :
  (motion_eq (3 + Δt) - motion_eq 3) / Δt = 5 + Δt :=
by
  sorry

end average_velocity_eq_l92_92122


namespace car_return_point_l92_92728

theorem car_return_point (α : ℝ) (hα1 : 0 < α) (hα2 : α < 180) :
  (∀ n : ℕ, n = 5 → 
    let theta := n * α in ∃ k : ℤ, θ = k * 360) ↔ (α = 72 ∨ α = 144) := 
sorry

end car_return_point_l92_92728


namespace SamDrove200Miles_l92_92808

/-- Given conditions -/
def MargueriteDistance : ℝ := 150
def MargueriteTime : ℝ := 3
def SameRateTime : ℝ := 4

/-- Calculate Marguerite's average speed -/
def MargueriteSpeed : ℝ := MargueriteDistance / MargueriteTime

/-- Calculate distance Sam drove -/
def SamDistance : ℝ := MargueriteSpeed * SameRateTime

/-- The theorem statement: Sam drove 200 miles -/
theorem SamDrove200Miles : SamDistance = 200 := by
  sorry

end SamDrove200Miles_l92_92808


namespace coordinates_of_C_prime_l92_92134

-- Define the given vertices of the triangle
def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (2, 1)
def C : ℝ × ℝ := (3, 2)

-- Define the similarity ratio
def similarity_ratio : ℝ := 2

-- Define the function for the similarity transformation
def similarity_transform (center : ℝ × ℝ) (ratio : ℝ) (point : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := point
  (ratio * x, ratio * y)

-- Prove the coordinates of C'
theorem coordinates_of_C_prime :
  similarity_transform (0, 0) similarity_ratio C = (6, 4) ∨ 
  similarity_transform (0, 0) similarity_ratio C = (-6, -4) :=
by
  sorry

end coordinates_of_C_prime_l92_92134


namespace concert_ticket_revenue_l92_92857

theorem concert_ticket_revenue :
  let original_price := 20
  let first_group_discount := 0.40
  let second_group_discount := 0.15
  let third_group_premium := 0.10
  let first_group_size := 10
  let second_group_size := 20
  let third_group_size := 15
  (first_group_size * (original_price - first_group_discount * original_price)) +
  (second_group_size * (original_price - second_group_discount * original_price)) +
  (third_group_size * (original_price + third_group_premium * original_price)) = 790 :=
by
  simp
  sorry

end concert_ticket_revenue_l92_92857


namespace greatest_b_value_l92_92757

theorem greatest_b_value (b : ℝ) : 
  (-b^3 + b^2 + 7 * b - 10 ≥ 0) ↔ b ≤ 4 + Real.sqrt 6 :=
sorry

end greatest_b_value_l92_92757


namespace prove_x_eq_one_l92_92669

variables (x y : ℕ)

theorem prove_x_eq_one 
  (hx : x > 0) 
  (hy : y > 0) 
  (hdiv : ∀ n : ℕ, n > 0 → (2^n * y + 1) ∣ (x^2^n - 1)) : 
  x = 1 :=
sorry

end prove_x_eq_one_l92_92669


namespace solution_set_for_f_geq_zero_l92_92684

theorem solution_set_for_f_geq_zero (f : ℝ → ℝ) (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_f3 : f 3 = 0) (h_cond : ∀ x : ℝ, x < 0 → x * (deriv f x) < f x) :
  {x : ℝ | f x ≥ 0} = {x : ℝ | -3 < x ∧ x < 0} ∪ {x : ℝ | 3 < x} :=
by
  sorry

end solution_set_for_f_geq_zero_l92_92684


namespace cost_of_traveling_roads_is_2600_l92_92594

-- Define the lawn, roads, and the cost parameters
def width_lawn : ℝ := 80
def length_lawn : ℝ := 60
def road_width : ℝ := 10
def cost_per_sq_meter : ℝ := 2

-- Area calculations
def area_road_1 : ℝ := road_width * length_lawn
def area_road_2 : ℝ := road_width * width_lawn
def area_intersection : ℝ := road_width * road_width

def total_area_roads : ℝ := area_road_1 + area_road_2 - area_intersection

def total_cost : ℝ := total_area_roads * cost_per_sq_meter

theorem cost_of_traveling_roads_is_2600 :
  total_cost = 2600 :=
by
  sorry

end cost_of_traveling_roads_is_2600_l92_92594


namespace probability_of_not_all_same_number_l92_92047

theorem probability_of_not_all_same_number :
  ∀ (D : ℕ) (n : ℕ),
  D = 8 → n = 5 → 
  let total_outcomes := D^n in
  let same_outcome_probability := D / total_outcomes  in
  let not_all_same_probability := 1 - same_outcome_probability in
  (not_all_same_probability = 4095 / 4096) :=
begin
  intros D n D_eq n_eq,
  simp only [D_eq, n_eq],
  let total_outcomes := 8^5,
  let same_outcome_probability := 8 / total_outcomes,
  let not_all_same_probability := 1 - same_outcome_probability,
  have total_outcome_calc : total_outcomes = 8^5 := by sorry,
  have same_outcome_prob_calc : same_outcome_probability = 1 / 4096 := by sorry,
  have not_all_same_prob_calc : not_all_same_probability = 4095 / 4096 := by sorry,
  exact not_all_same_prob_calc,
end

end probability_of_not_all_same_number_l92_92047


namespace probability_ratio_l92_92764

noncomputable def total_slips : ℕ := 40
noncomputable def slips_per_number : ℕ := 5
noncomputable def numbers : ℕ := 8
noncomputable def draw : ℕ := 4

noncomputable def total_combinations : ℕ := (40.choose 4)
noncomputable def p' : ℚ := (8 * (5.choose 4)) / total_combinations
noncomputable def q' : ℚ := (28 * (10 * 10)) / total_combinations

theorem probability_ratio : q' / p' = 70 :=
by
  sorry

end probability_ratio_l92_92764


namespace inequality_range_of_a_l92_92189

theorem inequality_range_of_a (a : ℝ) :
  (∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0) ↔ a ∈ Set.Icc (-2: ℝ) 2 :=
by
  sorry

end inequality_range_of_a_l92_92189


namespace explicit_expression_l92_92936

variable {α : Type*} [LinearOrder α] {f : α → α}

/-- Given that the function satisfies a specific condition, prove the function's explicit expression. -/
theorem explicit_expression (f : ℝ → ℝ) (h : ∀ x, f (3 * x + 2) = 9 * x + 8) : 
  ∀ x, f x = 3 * x + 2 :=
by
  sorry

end explicit_expression_l92_92936


namespace factorization_identity_sum_l92_92848

theorem factorization_identity_sum (a b c : ℤ)
  (h1 : ∀ x : ℤ, x^2 + 15 * x + 36 = (x + a) * (x + b))
  (h2 : ∀ x : ℤ, x^2 + 7 * x - 60 = (x + b) * (x - c)) :
  a + b + c = 20 :=
sorry

end factorization_identity_sum_l92_92848


namespace stratified_sampling_third_year_students_l92_92221

theorem stratified_sampling_third_year_students 
  (N : ℕ) (N_1 : ℕ) (P_sophomore : ℝ) (n : ℕ) (N_2 : ℕ) :
  N = 2000 →
  N_1 = 760 →
  P_sophomore = 0.37 →
  n = 20 →
  N_2 = Nat.ceil (N - N_1 - P_sophomore * N) →
  Nat.floor ((n : ℝ) / (N : ℝ) * (N_2 : ℝ)) = 5 :=
by
  sorry

end stratified_sampling_third_year_students_l92_92221


namespace tens_digit_of_2023_pow_2024_minus_2025_l92_92577

theorem tens_digit_of_2023_pow_2024_minus_2025 : 
  ∀ (n : ℕ), n = 2023^2024 - 2025 → ((n % 100) / 10) = 0 :=
by
  intros n h
  sorry

end tens_digit_of_2023_pow_2024_minus_2025_l92_92577


namespace ellipse_hyperbola_eccentricities_l92_92695

theorem ellipse_hyperbola_eccentricities :
  ∃ x y : ℝ, (2 * x^2 - 5 * x + 2 = 0) ∧ (2 * y^2 - 5 * y + 2 = 0) ∧ 
  ((2 > 1) ∧ (0 < (1/2) ∧ (1/2 < 1))) :=
by
  sorry

end ellipse_hyperbola_eccentricities_l92_92695


namespace factor_polynomial_l92_92322

theorem factor_polynomial :
  ∀ x : ℝ, 
  (x^2 + 4 * x + 3) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8) 
  = (x^2 + 6 * x + 1) * (x^2 + 6 * x + 37) :=
by
  intros x
  sorry

end factor_polynomial_l92_92322


namespace probability_only_one_solves_l92_92376

theorem probability_only_one_solves :
  (1/2) * (1 - 1/3) * (1 - 1/4) + (1/3) * (1 - 1/2) * (1 - 1/4) + (1/4) * (1 - 1/2) * (1 - 1/3) = 11/24 :=
by
  rw [show 1 - 1/3 = 2/3, by norm_num, show 1 - 1/4 = 3/4, by norm_num]
  rw [show 1 - 1/2 = 1/2, by norm_num, show 1 - 1/4 = 3/4, by norm_num]
  rw [show 1 - 1/2 = 1/2, by norm_num, show 1 - 1/3 = 2/3, by norm_num]
  norm_num

end probability_only_one_solves_l92_92376


namespace true_or_false_is_true_l92_92773

theorem true_or_false_is_true (p q : Prop) (hp : p = true) (hq : q = false) : p ∨ q = true :=
by
  sorry

end true_or_false_is_true_l92_92773


namespace difference_max_min_planes_l92_92202

open Set

-- Defining the regular tetrahedron and related concepts
noncomputable def tetrahedron := Unit -- Placeholder for the tetrahedron

def union_faces (T : Unit) : Set Point := sorry -- Placeholder for union of faces definition

noncomputable def simple_trace (p : Plane) (T : Unit) : Set Point := sorry -- Placeholder for planes intersecting faces

-- Calculating number of planes
def maximum_planes (T : Unit) : Nat :=
  4 -- One for each face of the tetrahedron

def minimum_planes (T : Unit) : Nat :=
  2 -- Each plane covers traces on two adjacent faces if oriented appropriately

-- Statement of the problem
theorem difference_max_min_planes (T : Unit) :
  maximum_planes T - minimum_planes T = 2 :=
by
  -- Proof skipped
  sorry

end difference_max_min_planes_l92_92202


namespace sale_in_fourth_month_l92_92888

-- Given conditions
def sales_first_month : ℕ := 5266
def sales_second_month : ℕ := 5768
def sales_third_month : ℕ := 5922
def sales_sixth_month : ℕ := 4937
def required_average_sales : ℕ := 5600
def number_of_months : ℕ := 6

-- Sum of the first, second, third, and sixth month's sales
def total_sales_without_fourth_fifth : ℕ := sales_first_month + sales_second_month + sales_third_month + sales_sixth_month

-- Total sales required to achieve the average required
def required_total_sales : ℕ := required_average_sales * number_of_months

-- The sale in the fourth month should be calculated as follows
def sales_fourth_month : ℕ := required_total_sales - total_sales_without_fourth_fifth

-- Proof statement
theorem sale_in_fourth_month :
  sales_fourth_month = 11707 := by
  sorry

end sale_in_fourth_month_l92_92888


namespace bobs_password_probability_l92_92737

theorem bobs_password_probability :
  (5 / 10) * (5 / 10) * 1 * (9 / 10) = 9 / 40 :=
by
  sorry

end bobs_password_probability_l92_92737


namespace intersection_A_B_intersection_A_complementB_l92_92960

-- Definitions of the sets A and B
def setA : Set ℝ := { x | -5 ≤ x ∧ x ≤ 3 }
def setB : Set ℝ := { x | x < -2 ∨ x > 4 }

-- Proof problem 1: A ∩ B = { x | -5 ≤ x < -2 }
theorem intersection_A_B:
  setA ∩ setB = { x : ℝ | -5 ≤ x ∧ x < -2 } :=
sorry

-- Definition of the complement of B
def complB : Set ℝ := { x | -2 ≤ x ∧ x ≤ 4 }

-- Proof problem 2: A ∩ (complB) = { x | -2 ≤ x ≤ 3 }
theorem intersection_A_complementB:
  setA ∩ complB = { x : ℝ | -2 ≤ x ∧ x ≤ 3 } :=
sorry

end intersection_A_B_intersection_A_complementB_l92_92960


namespace nature_of_roots_l92_92750

noncomputable def P (x : ℝ) : ℝ := x^6 - 5 * x^5 - 7 * x^3 - 2 * x + 9

theorem nature_of_roots : 
  (∀ x < 0, P x > 0) ∧ ∃ x > 0, P 0 * P x < 0 := 
by {
  sorry
}

end nature_of_roots_l92_92750


namespace volunteer_comprehensive_score_is_92_l92_92217

noncomputable def written_score : ℝ := 90
noncomputable def trial_lecture_score : ℝ := 94
noncomputable def interview_score : ℝ := 90

noncomputable def written_weight : ℝ := 0.3
noncomputable def trial_lecture_weight : ℝ := 0.5
noncomputable def interview_weight : ℝ := 0.2

noncomputable def comprehensive_score : ℝ :=
  written_score * written_weight +
  trial_lecture_score * trial_lecture_weight +
  interview_score * interview_weight

theorem volunteer_comprehensive_score_is_92 :
  comprehensive_score = 92 := by
  sorry

end volunteer_comprehensive_score_is_92_l92_92217


namespace find_b_of_parabola_axis_of_symmetry_l92_92005

theorem find_b_of_parabola_axis_of_symmetry (b : ℝ) :
  (∀ (x : ℝ), (x = 1) ↔ (x = - (b / (2 * 2))) ) → b = 4 :=
by
  intro h
  sorry

end find_b_of_parabola_axis_of_symmetry_l92_92005


namespace number_of_math_books_l92_92991

theorem number_of_math_books (M H : ℕ) (h1 : M + H = 90) (h2 : 4 * M + 5 * H = 397) : M = 53 :=
by
  sorry

end number_of_math_books_l92_92991


namespace GCF_seven_eight_factorial_l92_92342

-- Given conditions
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Calculating 7! and 8!
def seven_factorial := factorial 7
def eight_factorial := factorial 8

-- Proof statement
theorem GCF_seven_eight_factorial : ∃ g, g = seven_factorial ∧ g = Nat.gcd seven_factorial eight_factorial ∧ g = 5040 :=
by sorry

end GCF_seven_eight_factorial_l92_92342


namespace abs_sum_sequence_l92_92564

def S (n : ℕ) : ℤ := n^2 - 4 * n

def a (n : ℕ) : ℤ := S n - S (n-1)

theorem abs_sum_sequence (h : ∀ n, S n = n^2 - 4 * n) :
  (|a 1| + |a 2| + |a 3| + |a 4| + |a 5| + |a 6| + |a 7| + |a 8| + |a 9| + |a 10|) = 68 :=
by
  sorry

end abs_sum_sequence_l92_92564


namespace sequence_of_8_numbers_l92_92786

theorem sequence_of_8_numbers :
  ∃ (a b c d e f g h : ℤ), 
    a + b + c = 100 ∧ b + c + d = 100 ∧ c + d + e = 100 ∧ 
    d + e + f = 100 ∧ e + f + g = 100 ∧ f + g + h = 100 ∧ 
    a = 20 ∧ h = 16 ∧ 
    (a, b, c, d, e, f, g, h) = (20, 16, 64, 20, 16, 64, 20, 16) :=
by
  sorry

end sequence_of_8_numbers_l92_92786


namespace find_base_l92_92512

theorem find_base 
  (k : ℕ) 
  (h : 1 * k^2 + 3 * k^1 + 2 * k^0 = 30) : 
  k = 4 :=
  sorry

end find_base_l92_92512


namespace ratio_s_t_l92_92569

variable {b s t : ℝ}
variable (hb : b ≠ 0)
variable (h1 : s = -b / 8)
variable (h2 : t = -b / 4)

theorem ratio_s_t : s / t = 1 / 2 :=
by
  sorry

end ratio_s_t_l92_92569


namespace factor_polynomial_l92_92258

theorem factor_polynomial (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = 
  (x^2 + 6*x + 4) * (x^2 + 6*x + 11) :=
by
  sorry

end factor_polynomial_l92_92258


namespace largest_sum_of_ABC_l92_92383

-- Define the variables and the conditions
def A := 533
def B := 5
def C := 1

-- Define the product condition
def product_condition : Prop := (A * B * C = 2665)

-- Define the distinct positive integers condition
def distinct_positive_integers_condition : Prop := (A > 0 ∧ B > 0 ∧ C > 0 ∧ A ≠ B ∧ B ≠ C ∧ A ≠ C)

-- State the theorem
theorem largest_sum_of_ABC : product_condition → distinct_positive_integers_condition → A + B + C = 539 := by
  intros _ _
  sorry

end largest_sum_of_ABC_l92_92383


namespace probability_of_same_length_segments_l92_92153

-- Definitions directly from the conditions
def total_elements : ℕ := 6 + 9

-- Probability calculations
def probability_same_length : ℚ :=
  ((6 / total_elements) * (5 / (total_elements - 1))) + 
  ((9 / total_elements) * (8 / (total_elements - 1)))

-- Proof statement
theorem probability_of_same_length_segments : probability_same_length = 17 / 35 := by
  unfold probability_same_length
  sorry

end probability_of_same_length_segments_l92_92153


namespace bernardo_wins_at_5_l92_92729

theorem bernardo_wins_at_5 :
  ∃ N : ℕ, 0 ≤ N ∧ N ≤ 499 ∧ 27 * N + 360 < 500 ∧ ∀ M : ℕ, (0 ≤ M ∧ M ≤ 499 ∧ 27 * M + 360 < 500 → N ≤ M) :=
by
  sorry

end bernardo_wins_at_5_l92_92729


namespace rectangle_area_l92_92592

def length : ℝ := 2
def width : ℝ := 4
def area := length * width

theorem rectangle_area : area = 8 := 
by
  -- Proof can be written here
  sorry

end rectangle_area_l92_92592


namespace average_speed_of_car_l92_92590

noncomputable def avgSpeed (Distance_uphill Speed_uphill Distance_downhill Speed_downhill : ℝ) : ℝ :=
  let Time_uphill := Distance_uphill / Speed_uphill
  let Time_downhill := Distance_downhill / Speed_downhill
  let Total_time := Time_uphill + Time_downhill
  let Total_distance := Distance_uphill + Distance_downhill
  Total_distance / Total_time

theorem average_speed_of_car:
  avgSpeed 100 30 50 60 = 36 := by
  sorry

end average_speed_of_car_l92_92590


namespace leak_empties_cistern_in_24_hours_l92_92720

theorem leak_empties_cistern_in_24_hours (F L : ℝ) (h1: F = 1 / 8) (h2: F - L = 1 / 12) :
  1 / L = 24 := 
by {
  sorry
}

end leak_empties_cistern_in_24_hours_l92_92720


namespace factor_polynomial_l92_92254

theorem factor_polynomial (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = 
  (x^2 + 6*x + 4) * (x^2 + 6*x + 11) :=
by
  sorry

end factor_polynomial_l92_92254


namespace distance_with_wind_l92_92225

-- Define constants
def distance_against_wind : ℝ := 320
def speed_wind : ℝ := 20
def speed_plane_still_air : ℝ := 180

-- Calculate effective speeds
def effective_speed_with_wind : ℝ := speed_plane_still_air + speed_wind
def effective_speed_against_wind : ℝ := speed_plane_still_air - speed_wind

-- Define the proof statement
theorem distance_with_wind :
  ∃ (D : ℝ), (D / effective_speed_with_wind) = (distance_against_wind / effective_speed_against_wind) ∧ D = 400 :=
by
  sorry

end distance_with_wind_l92_92225


namespace predicted_whales_l92_92197

theorem predicted_whales (num_last_year num_this_year num_next_year : ℕ)
  (h1 : num_this_year = 2 * num_last_year)
  (h2 : num_last_year = 4000)
  (h3 : num_next_year = 8800) :
  num_next_year - num_this_year = 800 :=
by
  sorry

end predicted_whales_l92_92197


namespace cos_arcsin_l92_92740

theorem cos_arcsin (h : real.sin θ = 3 / 5) : real.cos θ = 4 / 5 :=
sorry

end cos_arcsin_l92_92740


namespace polynomial_factorization_l92_92311

theorem polynomial_factorization (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = (x^2 + 6*x + 1) * (x^2 + 6*x + 37) :=
  sorry

end polynomial_factorization_l92_92311


namespace cubic_polynomial_roots_l92_92915

variables (a b c : ℚ)

theorem cubic_polynomial_roots (a b c : ℚ) :
  (c = 0 → ∃ x y z : ℚ, (x = 0 ∧ y = 1 ∧ z = -2) ∧ (-a = x + y + z) ∧ (b = x*y + y*z + z*x) ∧ (-c = x*y*z)) ∧
  (c ≠ 0 → ∃ x y z : ℚ, (x = 1 ∧ y = -1 ∧ z = -1) ∧ (-a = x + y + z) ∧ (b = x*y + y*z + z*x) ∧ (-c = x*y*z)) :=
by
  sorry

end cubic_polynomial_roots_l92_92915


namespace treasure_chest_l92_92597

theorem treasure_chest (n : ℕ) 
  (h1 : n % 8 = 2)
  (h2 : n % 7 = 6)
  (h3 : ∀ m : ℕ, (m % 8 = 2 → m % 7 = 6 → m ≥ n)) :
  n % 9 = 7 :=
sorry

end treasure_chest_l92_92597


namespace books_before_grant_correct_l92_92680

-- Definitions based on the given conditions
def books_purchased : ℕ := 2647
def total_books_now : ℕ := 8582

-- Definition and the proof statement
def books_before_grant : ℕ := 5935

-- Proof statement: The number of books before the grant plus the books purchased equals the total books now
theorem books_before_grant_correct :
  books_before_grant + books_purchased = total_books_now :=
by
  -- Predictably, no need to complete proof, 'sorry' is used.
  sorry

end books_before_grant_correct_l92_92680


namespace compute_difference_a_b_l92_92096

-- Define the initial amounts paid by Alex, Bob, and Carol
def alex_paid := 120
def bob_paid := 150
def carol_paid := 210

-- Define the total amount and equal share
def total_costs := alex_paid + bob_paid + carol_paid
def equal_share := total_costs / 3

-- Define the amounts Alex and Carol gave to Bob, satisfying their balances
def a := equal_share - alex_paid
def b := carol_paid - equal_share

-- Lean 4 statement to prove a - b = 30
theorem compute_difference_a_b : a - b = 30 := by
  sorry

end compute_difference_a_b_l92_92096


namespace solve_for_percentage_l92_92476

-- Define the constants and variables
variables (P : ℝ)

-- Define the given conditions
def condition : Prop := (P / 100 * 1600 = P / 100 * 650 + 190)

-- Formalize the conjecture: if the conditions hold, then P = 20
theorem solve_for_percentage (h : condition P) : P = 20 :=
sorry

end solve_for_percentage_l92_92476


namespace agnes_flight_cost_l92_92784

theorem agnes_flight_cost
  (booking_fee : ℝ) (cost_per_km : ℝ) (distance_XY : ℝ)
  (h1 : booking_fee = 120)
  (h2 : cost_per_km = 0.12)
  (h3 : distance_XY = 4500) :
  booking_fee + cost_per_km * distance_XY = 660 := 
by
  sorry

end agnes_flight_cost_l92_92784


namespace smallest_n_l92_92675

theorem smallest_n (n : ℕ) (h1 : n % 6 = 5) (h2 : n % 7 = 4) (h3 : n > 20) : n = 53 :=
sorry

end smallest_n_l92_92675


namespace determine_A_plus_B_l92_92094

theorem determine_A_plus_B :
  ∃ (A B : ℚ), ((∀ x : ℚ, x ≠ 4 ∧ x ≠ 5 → 
  (Bx - 23) / (x^2 - 9 * x + 20) = A / (x - 4) + 5 / (x - 5)) ∧
  (A + B = 11 / 9)) :=
sorry

end determine_A_plus_B_l92_92094


namespace necessary_not_sufficient_for_circle_l92_92184

theorem necessary_not_sufficient_for_circle (a : ℝ) :
  (a ≤ 2 → (x^2 + y^2 - 2*x + 2*y + a = 0 → ∃ r : ℝ, r > 0)) ∧
  (a ≤ 2 ∧ ∃ b, b < 2 → a = b) := sorry

end necessary_not_sufficient_for_circle_l92_92184


namespace length_AB_l92_92008

noncomputable def parabola_p := 3
def x1_x2_sum := 6

theorem length_AB (x1 x2 : ℝ) (y1 y2 : ℝ) 
  (h1 : x1 + x2 = x1_x2_sum)
  (h2 : (y1^2 = 6 * x1) ∧ (y2^2 = 6 * x2))
  : abs (x1 + parabola_p / 2 - (x2 + parabola_p / 2)) = 9 := by
  sorry

end length_AB_l92_92008


namespace f_decreasing_interval_triangle_abc_l92_92774

noncomputable def f (x : Real) : Real := 2 * (Real.sin x)^2 + Real.cos ((Real.pi) / 3 - 2 * x)

theorem f_decreasing_interval :
  ∃ (a b : Real), a = Real.pi / 3 ∧ b = 5 * Real.pi / 6 ∧ 
  ∀ x y, (a ≤ x ∧ x < y ∧ y ≤ b) → f y ≤ f x := 
sorry

variables {a b c : Real} (A B C : Real) 

theorem triangle_abc (h1 : A = Real.pi / 3) 
    (h2 : f A = 2)
    (h3 : a = 2 * b)
    (h4 : Real.sin C = 2 * Real.sin B):
  a / b = Real.sqrt 3 := 
sorry

end f_decreasing_interval_triangle_abc_l92_92774


namespace common_divisors_count_9240_10010_l92_92651

def divisors (n : Nat) : Nat :=
(n.primeFactors.map (λ p => p.2 + 1)).foldl (· * ·) 1

theorem common_divisors_count_9240_10010 :
  let gcd_value := Nat.gcd 9240 10010;
  let num_common_divisors := divisors gcd_value;
  gcd_value = 210 ∧ num_common_divisors = 16 :=
by
  have : 9240 = 2^3 * 3^1 * 5^1 * 7^2 := by norm_num
  have : 10010 = 2^1 * 3^1 * 5^1 * 7^1 * 11^1 := by norm_num
  have gcd_value_calc := Nat.gcd 9240 10010
  have : gcd_value_calc = 210 := by norm_num
  have num_common_divisors_calc := divisors gcd_value_calc
  have : num_common_divisors_calc = 16 := by norm_num
  exact ⟨this, by norm_num⟩

end common_divisors_count_9240_10010_l92_92651


namespace sam_drove_distance_l92_92817

theorem sam_drove_distance (marguerite_distance : ℝ) (marguerite_time : ℝ) (sam_time : ℝ) :
  marguerite_distance = 150 ∧ marguerite_time = 3 ∧ sam_time = 4 →
  (sam_time * (marguerite_distance / marguerite_time) = 200) :=
by
  sorry

end sam_drove_distance_l92_92817


namespace initial_tickets_l92_92095

-- Definitions of the conditions
def ferris_wheel_rides : ℕ := 2
def roller_coaster_rides : ℕ := 3
def log_ride_rides : ℕ := 7

def ferris_wheel_cost : ℕ := 2
def roller_coaster_cost : ℕ := 5
def log_ride_cost : ℕ := 1

def additional_tickets_needed : ℕ := 6

-- Calculate the total number of tickets needed
def total_tickets_needed : ℕ := 
  (ferris_wheel_rides * ferris_wheel_cost) +
  (roller_coaster_rides * roller_coaster_cost) +
  (log_ride_rides * log_ride_cost)

-- The proof statement
theorem initial_tickets : ∀ (initial_tickets : ℕ), 
  total_tickets_needed - additional_tickets_needed = initial_tickets → 
  initial_tickets = 20 :=
by
  intros initial_tickets h
  sorry

end initial_tickets_l92_92095


namespace probability_of_not_all_same_number_l92_92044

theorem probability_of_not_all_same_number :
  ∀ (D : ℕ) (n : ℕ),
  D = 8 → n = 5 → 
  let total_outcomes := D^n in
  let same_outcome_probability := D / total_outcomes  in
  let not_all_same_probability := 1 - same_outcome_probability in
  (not_all_same_probability = 4095 / 4096) :=
begin
  intros D n D_eq n_eq,
  simp only [D_eq, n_eq],
  let total_outcomes := 8^5,
  let same_outcome_probability := 8 / total_outcomes,
  let not_all_same_probability := 1 - same_outcome_probability,
  have total_outcome_calc : total_outcomes = 8^5 := by sorry,
  have same_outcome_prob_calc : same_outcome_probability = 1 / 4096 := by sorry,
  have not_all_same_prob_calc : not_all_same_probability = 4095 / 4096 := by sorry,
  exact not_all_same_prob_calc,
end

end probability_of_not_all_same_number_l92_92044


namespace base3_to_base10_l92_92612

theorem base3_to_base10 (d0 d1 d2 d3 d4 : ℕ)
  (h0 : d4 = 2)
  (h1 : d3 = 1)
  (h2 : d2 = 0)
  (h3 : d1 = 2)
  (h4 : d0 = 1) :
  d4 * 3^4 + d3 * 3^3 + d2 * 3^2 + d1 * 3^1 + d0 * 3^0 = 196 := by
  sorry

end base3_to_base10_l92_92612


namespace find_k_l92_92354

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (a b : V) (k : ℝ)

-- Conditions
def not_collinear (a b : V) : Prop := ¬ ∃ (m : ℝ), b = m • a
def collinear (u v : V) : Prop := ∃ (m : ℝ), u = m • v

theorem find_k (h1 : not_collinear a b) (h2 : collinear (2 • a + k • b) (a - b)) : k = -2 :=
by
  sorry

end find_k_l92_92354


namespace probability_same_length_segments_of_regular_hexagon_l92_92143

theorem probability_same_length_segments_of_regular_hexagon (sides diagonals : ℕ) 
  (total_segments : ℕ) (prob_same_length : ℚ) 
  (h_sides : sides = 6)
  (h_diagonals : diagonals = 9)
  (h_total_segments : total_segments = sides + diagonals)
  : prob_same_length = 17 / 35 :=
by
  have total_combinations := total_segments * (total_segments - 1) / 2
  have prob_sides := (sides * (sides - 1) / 2 : ℚ) / total_combinations
  have prob_diagonals := (diagonals * (diagonals - 1) / 2 : ℚ) / total_combinations
  have prob_sum := prob_sides + prob_diagonals
  have h_prob_sides : prob_sides = (6 * 5 / 2) / (15 * 14 / 2) :=
    by sorry
  have h_prob_diagonals : prob_diagonals = (9 * 8 / 2) / (15 * 14 / 2) :=
    by sorry
  calc
    prob_same_length = prob_sum : by sorry
    ... = 17 / 35 : by {rw [←h_prob_sides, ←h_prob_diagonals], sorry}

end probability_same_length_segments_of_regular_hexagon_l92_92143


namespace factors_of_12_factors_of_18_l92_92007

def is_factor (n k : ℕ) : Prop := k > 0 ∧ n % k = 0

theorem factors_of_12 : 
  {k : ℕ | is_factor 12 k} = {1, 12, 2, 6, 3, 4} :=
by
  sorry

theorem factors_of_18 : 
  {k : ℕ | is_factor 18 k} = {1, 18, 2, 9, 3, 6} :=
by
  sorry

end factors_of_12_factors_of_18_l92_92007


namespace problem1_problem2_problem3_l92_92010

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if 1 ≤ x then x^2 - 2 * a * x + a
  else if 0 < x then 2 * x + a / x
  else 0 -- Undefined for x ≤ 0

theorem problem1 (a : ℝ) :
  (∀ x y : ℝ, (0 < x ∧ x < y) → f a x < f a y) ↔ (a ≤ -1 / 2) :=
sorry
  
theorem problem2 (a : ℝ) :
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x3 ≠ x1 ∧ f a x1 = 1 ∧ f a x2 = 1 ∧ f a x3 = 1) ↔ (0 < a ∧ a < 1 / 8) :=
sorry

theorem problem3 (a : ℝ) :
  (∀ x : ℝ, f a x ≥ x - 2 * a) ↔ (0 ≤ a ∧ a ≤ 1 + Real.sqrt 3 / 2) :=
sorry

end problem1_problem2_problem3_l92_92010


namespace star_neg5_4_star_neg3_neg6_l92_92457

-- Definition of the new operation
def star (a b : ℤ) : ℤ := 2 * a * b - b / 2

-- The first proof problem
theorem star_neg5_4 : star (-5) 4 = -42 := by sorry

-- The second proof problem
theorem star_neg3_neg6 : star (-3) (-6) = 39 := by sorry

end star_neg5_4_star_neg3_neg6_l92_92457


namespace expression_evaluation_l92_92997

theorem expression_evaluation (a b : ℕ) (h1 : a = 25) (h2 : b = 15) : (a + b)^2 - (a^2 + b^2) = 750 :=
by
  sorry

end expression_evaluation_l92_92997


namespace solve_for_f_8_l92_92919

noncomputable def f (x : ℝ) : ℝ := (Real.logb 2 x)

theorem solve_for_f_8 {x : ℝ} (h : f (x^3) = Real.logb 2 x) : f 8 = 1 :=
by
sorry

end solve_for_f_8_l92_92919


namespace first_guinea_pig_food_l92_92402

theorem first_guinea_pig_food (x : ℕ) (h1 : ∃ x : ℕ, R = x + 2 * x + (2 * x + 3)) (hp : 13 = x + 2 * x + (2 * x + 3)) : x = 2 :=
by
  sorry

end first_guinea_pig_food_l92_92402


namespace polynomial_factorization_l92_92317

theorem polynomial_factorization (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = (x^2 + 6*x + 1) * (x^2 + 6*x + 37) :=
  sorry

end polynomial_factorization_l92_92317


namespace factor_polynomial_l92_92319

theorem factor_polynomial :
  ∀ x : ℝ, 
  (x^2 + 4 * x + 3) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8) 
  = (x^2 + 6 * x + 1) * (x^2 + 6 * x + 37) :=
by
  intros x
  sorry

end factor_polynomial_l92_92319


namespace birds_count_l92_92450

theorem birds_count (N B : ℕ) 
  (h1 : B = 5 * N)
  (h2 : B = N + 360) : 
  B = 450 := by
  sorry

end birds_count_l92_92450


namespace linda_age_l92_92682

theorem linda_age
  (j k l : ℕ)       -- Ages of Jane, Kevin, and Linda respectively
  (h1 : j + k + l = 36)    -- Condition 1: j + k + l = 36
  (h2 : l - 3 = j)         -- Condition 2: l - 3 = j
  (h3 : k + 4 = (1 / 2 : ℝ) * (l + 4))  -- Condition 3: k + 4 = 1/2 * (l + 4)
  : l = 16 := 
sorry

end linda_age_l92_92682


namespace probability_not_all_same_l92_92036

theorem probability_not_all_same (n m : ℕ) (h₁ : n = 5) (h₂ : m = 8) 
  (fair_dice : ∀ (die : ℕ), 1 ≤ die ∧ die ≤ m)
  : (1 - (m / m^n) = 4095 / 4096) := by
  sorry

end probability_not_all_same_l92_92036


namespace quadratic_solution_l92_92194

theorem quadratic_solution (x : ℝ) : (x^2 - 3 * x + 2 < 0) ↔ (1 < x ∧ x < 2) :=
by
  sorry

end quadratic_solution_l92_92194


namespace factor_polynomial_l92_92301

theorem factor_polynomial :
  (x^2 + 4 * x + 3) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8) = 
  (x^2 + 6 * x + 7) * (x^2 + 6 * x + 8) :=
sorry

end factor_polynomial_l92_92301


namespace sqrt_neg4_sq_eq_4_l92_92999

theorem sqrt_neg4_sq_eq_4 : Real.sqrt ((-4 : ℝ) ^ 2) = 4 := by
  sorry

end sqrt_neg4_sq_eq_4_l92_92999


namespace peaches_sold_to_friends_l92_92394

theorem peaches_sold_to_friends (x : ℕ) (total_peaches : ℕ) (peaches_to_relatives : ℕ) (peach_price_friend : ℕ) (peach_price_relative : ℝ) (total_earnings : ℝ) (peaches_left : ℕ) (total_peaches_sold : ℕ) 
  (h1 : total_peaches = 15) 
  (h2 : peaches_to_relatives = 4) 
  (h3 : peach_price_relative = 1.25) 
  (h4 : total_earnings = 25) 
  (h5 : peaches_left = 1)
  (h6 : total_peaches_sold = 14)
  (h7 : total_earnings = peach_price_friend * x + peach_price_relative * peaches_to_relatives)
  (h8 : total_peaches_sold = total_peaches - peaches_left) :
  x = 10 := 
sorry

end peaches_sold_to_friends_l92_92394


namespace discount_percentage_l92_92231

variable {P P_b P_s : ℝ}
variable {D : ℝ}

theorem discount_percentage (P_s_eq_bought : P_s = 1.60 * P_b)
  (P_s_eq_original : P_s = 1.52 * P)
  (P_b_eq_discount : P_b = P * (1 - D)) :
  D = 0.05 := by
sorry

end discount_percentage_l92_92231


namespace find_age_l92_92073

-- Define the age variables
variables (P Q : ℕ)

-- Define the conditions
def condition1 : Prop := (P - 3) * 3 = (Q - 3) * 4
def condition2 : Prop := (P + 6) * 6 = (Q + 6) * 7

-- Prove that, given the conditions, P equals 15
theorem find_age (h1 : condition1 P Q) (h2 : condition2 P Q) : P = 15 :=
sorry

end find_age_l92_92073


namespace mom_foster_dog_food_l92_92663

theorem mom_foster_dog_food
    (puppy_food_per_meal : ℚ := 1 / 2)
    (puppy_meals_per_day : ℕ := 2)
    (num_puppies : ℕ := 5)
    (total_food_needed : ℚ := 57)
    (days : ℕ := 6)
    (mom_meals_per_day : ℕ := 3) :
    (total_food_needed - (num_puppies * puppy_food_per_meal * ↑puppy_meals_per_day * ↑days)) / (↑days * ↑mom_meals_per_day) = 1.5 :=
by
  -- Definitions translation
  let puppy_total_food := num_puppies * puppy_food_per_meal * ↑puppy_meals_per_day * ↑days
  let mom_total_food := total_food_needed - puppy_total_food
  let mom_meals := ↑days * ↑mom_meals_per_day
  -- Proof starts with sorry to indicate that the proof part is not included
  sorry

end mom_foster_dog_food_l92_92663


namespace line_through_point_with_equal_intercepts_l92_92486

-- Define the point through which the line passes
def point : ℝ × ℝ := (3, -2)

-- Define the property of having equal absolute intercepts
def has_equal_absolute_intercepts (a b : ℝ) : Prop :=
  |a| = |b|

-- Define the general form of a line equation
def line_eq (a b c x y : ℝ) : Prop :=
  a * x + b * y + c = 0

-- Main theorem: Any line passing through (3, -2) with equal absolute intercepts satisfies the given equations
theorem line_through_point_with_equal_intercepts (a b : ℝ) :
  has_equal_absolute_intercepts a b
  → line_eq 2 3 0 3 (-2)
  ∨ line_eq 1 1 (-1) 3 (-2)
  ∨ line_eq 1 (-1) (-5) 3 (-2) :=
by {
  sorry
}

end line_through_point_with_equal_intercepts_l92_92486


namespace find_ab_l92_92115

theorem find_ab (a b : ℝ) (h1 : a + b = 5) (h2 : a^3 + b^3 = 125) : a * b = 0 :=
by
  sorry

end find_ab_l92_92115


namespace largest_x_satisfies_eq_l92_92025

theorem largest_x_satisfies_eq (x : ℝ) (hx : sqrt (3 * x) = 5 * x) : x ≤ 3 / 25 :=
sorry

end largest_x_satisfies_eq_l92_92025


namespace factor_polynomial_l92_92256

theorem factor_polynomial (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = 
  (x^2 + 6*x + 4) * (x^2 + 6*x + 11) :=
by
  sorry

end factor_polynomial_l92_92256


namespace factor_poly_eq_factored_form_l92_92268

-- Defining the polynomial expressions
def poly1 (x : ℝ) := x^2 + 4 * x + 3
def poly2 (x : ℝ) := x^2 + 8 * x + 15
def poly3 (x : ℝ) := x^2 + 6 * x - 8

-- The main expression which needs to be factored
def main_expr (x : ℝ) := (poly1 x) * (poly2 x) + (poly3 x)

-- Stating the goal factored form
def factored_form (x : ℝ) := (x^2 + 6 * x + 19) * (x^2 + 6 * x - 2)

-- The theorem statement
theorem factor_poly_eq_factored_form (x : ℝ) : 
  main_expr x = factored_form x := 
by
  sorry

end factor_poly_eq_factored_form_l92_92268


namespace equation_of_parallel_line_l92_92559

noncomputable def is_parallel (m₁ m₂ : ℝ) := m₁ = m₂

theorem equation_of_parallel_line (m : ℝ) (b : ℝ) (x₀ y₀ : ℝ) (a b1 c : ℝ) :
  is_parallel m (1 / 2) → y₀ = -1 → x₀ = 0 → 
  (a = 1 ∧ b1 = -2 ∧ c = -2) →
  a * x₀ + b1 * y₀ + c = 0 :=
by
  intros h_parallel hy hx habc
  sorry

end equation_of_parallel_line_l92_92559


namespace number_of_white_balls_l92_92790

theorem number_of_white_balls (total_balls yellow_frequency : ℕ) (h1 : total_balls = 10) (h2 : yellow_frequency = 60) :
  (total_balls - (total_balls * yellow_frequency / 100) = 4) :=
by
  sorry

end number_of_white_balls_l92_92790


namespace focus_of_parabola_l92_92098

def parabola_focus (a k : ℕ) : ℚ :=
  1 / (4 * a) + k

theorem focus_of_parabola :
  parabola_focus 9 6 = 217 / 36 :=
by
  sorry

end focus_of_parabola_l92_92098


namespace SamDrove200Miles_l92_92810

/-- Given conditions -/
def MargueriteDistance : ℝ := 150
def MargueriteTime : ℝ := 3
def SameRateTime : ℝ := 4

/-- Calculate Marguerite's average speed -/
def MargueriteSpeed : ℝ := MargueriteDistance / MargueriteTime

/-- Calculate distance Sam drove -/
def SamDistance : ℝ := MargueriteSpeed * SameRateTime

/-- The theorem statement: Sam drove 200 miles -/
theorem SamDrove200Miles : SamDistance = 200 := by
  sorry

end SamDrove200Miles_l92_92810


namespace weights_sum_l92_92250

theorem weights_sum (e f g h : ℕ) (h₁ : e + f = 280) (h₂ : f + g = 230) (h₃ : e + h = 300) : g + h = 250 := 
by 
  sorry

end weights_sum_l92_92250


namespace nate_total_distance_l92_92835

def length_field : ℕ := 168
def distance_8s : ℕ := 4 * length_field
def additional_distance : ℕ := 500
def total_distance : ℕ := distance_8s + additional_distance

theorem nate_total_distance : total_distance = 1172 := by
  sorry

end nate_total_distance_l92_92835


namespace solve_for_a_plus_b_l92_92119

theorem solve_for_a_plus_b (a b : ℝ) :
  (∀ x : ℝ, (-1 < x ∧ x < 1 / 3) → ax^2 + bx + 1 > 0) →
  a * (-3) + b = -5 :=
by
  intro h
  -- Here we can use the proofs provided in the solution steps.
  sorry

end solve_for_a_plus_b_l92_92119


namespace max_remainder_division_by_9_l92_92981

theorem max_remainder_division_by_9 : ∀ (r : ℕ), r < 9 → r ≤ 8 :=
by sorry

end max_remainder_division_by_9_l92_92981


namespace gcd_8917_4273_l92_92861

theorem gcd_8917_4273 : Int.gcd 8917 4273 = 1 :=
by
  sorry

end gcd_8917_4273_l92_92861


namespace not_difference_of_squares_10_l92_92705

theorem not_difference_of_squares_10 (a b : ℤ) : a^2 - b^2 ≠ 10 :=
sorry

end not_difference_of_squares_10_l92_92705


namespace sum_of_numerator_and_denominator_l92_92495

def repeating_decimal_to_fraction_sum (x : ℚ) := 
  let numerator := 710
  let denominator := 99
  numerator + denominator

theorem sum_of_numerator_and_denominator : repeating_decimal_to_fraction_sum (71/10 + 7/990) = 809 := by
  sorry

end sum_of_numerator_and_denominator_l92_92495


namespace cos_arcsin_l92_92741

theorem cos_arcsin (h : real.sin θ = 3 / 5) : real.cos θ = 4 / 5 :=
sorry

end cos_arcsin_l92_92741


namespace min_le_one_fourth_sum_max_ge_four_ninths_sum_l92_92360

variable (a b c : ℝ)

theorem min_le_one_fourth_sum
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
  (h_roots : b^2 - 4 * a * c ≥ 0) :
  min a (min b c) ≤ 1 / 4 * (a + b + c) :=
sorry

theorem max_ge_four_ninths_sum
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
  (h_roots : b^2 - 4 * a * c ≥ 0) :
  max a (max b c) ≥ 4 / 9 * (a + b + c) :=
sorry

end min_le_one_fourth_sum_max_ge_four_ninths_sum_l92_92360


namespace equivalent_after_eliminating_denominators_l92_92207

theorem equivalent_after_eliminating_denominators (x : ℝ) (h : 1 + 2 / (x - 1) = (x - 5) / (x - 3)) :
  (x - 1) * (x - 3) + 2 * (x - 3) = (x - 5) * (x - 1) :=
sorry

end equivalent_after_eliminating_denominators_l92_92207


namespace product_of_consecutive_integers_sqrt_73_l92_92019

theorem product_of_consecutive_integers_sqrt_73 : 
  ∃ (m n : ℕ), (m < n) ∧ ∃ (j k : ℕ), (j = 8) ∧ (k = 9) ∧ (m = j) ∧ (n = k) ∧ (m * n = 72) := by
  sorry

end product_of_consecutive_integers_sqrt_73_l92_92019


namespace root_in_interval_k_eq_2_l92_92926

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 5

theorem root_in_interval_k_eq_2
  (k : ℤ)
  (h1 : 0 < f 2)
  (h2 : Real.log 2 + 2 * 2 - 5 < 0)
  (h3 : Real.log 3 + 2 * 3 - 5 > 0) 
  (h4 : f (k : ℝ) * f (k + 1 : ℝ) < 0) :
  k = 2 := 
sorry

end root_in_interval_k_eq_2_l92_92926


namespace factor_polynomial_l92_92284

-- Define the necessary polynomials
def p (x : ℝ) : ℝ := x^2 + 4*x + 3
def q (x : ℝ) : ℝ := x^2 + 8*x + 15
def r (x : ℝ) : ℝ := x^2 + 6*x - 8

-- State the main theorem
theorem factor_polynomial (x : ℝ) : 
  (p x * q x) + r x = (x^2 + 7*x + 1) * (x^2 + 4*x + 7) :=
by
  sorry

end factor_polynomial_l92_92284


namespace log_sqrt_defined_in_interval_l92_92637

def defined_interval (x : ℝ) : Prop :=
  ∃ y, y = (5 - x) ∧ y > 0 ∧ (x - 2) ≥ 0

theorem log_sqrt_defined_in_interval {x : ℝ} :
  defined_interval x ↔ (2 < x ∧ x < 5) :=
sorry

end log_sqrt_defined_in_interval_l92_92637


namespace jerry_weighted_mean_l92_92954

noncomputable def weighted_mean (aunt uncle sister cousin friend1 friend2 friend3 friend4 friend5 : ℝ)
    (eur_to_usd gbp_to_usd cad_to_usd : ℝ) (family_weight friends_weight : ℝ) : ℝ :=
  let uncle_usd := uncle * eur_to_usd
  let friend3_usd := friend3 * eur_to_usd
  let friend4_usd := friend4 * gbp_to_usd
  let cousin_usd := cousin * cad_to_usd
  let family_sum := aunt + uncle_usd + sister + cousin_usd
  let friends_sum := friend1 + friend2 + friend3_usd + friend4_usd + friend5
  family_sum * family_weight + friends_sum * friends_weight

theorem jerry_weighted_mean : 
  weighted_mean 9.73 9.43 7.25 20.37 22.16 23.51 18.72 15.53 22.84 
               1.20 1.38 0.82 0.40 0.60 = 85.4442 := 
sorry

end jerry_weighted_mean_l92_92954


namespace factorization_correct_l92_92291

noncomputable def polynomial_expr := 
  λ x : ℝ => (x^2 + 4 * x + 3) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8)

noncomputable def factored_expr := 
  λ x : ℝ => (x^2 + 6 * x + 9) * (x^2 + 6 * x + 1)

theorem factorization_correct : 
  ∀ x : ℝ, polynomial_expr x = factored_expr x :=
by
  intro x
  sorry

end factorization_correct_l92_92291


namespace no_leopards_in_circus_l92_92238

theorem no_leopards_in_circus (L T : ℕ) (N : ℕ) (h₁ : L = N / 5) (h₂ : T = 5 * (N - T)) : 
  ∀ A, A = L + N → A = T + (N - T) → ¬ ∃ x, x ≠ L ∧ x ≠ T ∧ x ≠ (N - L - T) :=
by
  sorry

end no_leopards_in_circus_l92_92238


namespace cos_75_eq_l92_92608

theorem cos_75_eq : Real.cos (75 * Real.pi / 180) = (Real.sqrt 6 - Real.sqrt 2) / 4 := by
  sorry

end cos_75_eq_l92_92608


namespace total_number_of_employees_l92_92235

theorem total_number_of_employees (n : ℕ) (hm : ℕ) (hd : ℕ) 
  (h_ratio : 4 * hd = hm)
  (h_diff : hm = hd + 72) : n = 120 :=
by
  -- proof steps would go here
  sorry

end total_number_of_employees_l92_92235


namespace mike_planted_50_l92_92138

-- Definitions for conditions
def mike_morning (M : ℕ) := M
def ted_morning (M : ℕ) := 2 * M
def mike_afternoon := 60
def ted_afternoon := 40
def total_planted (M : ℕ) := mike_morning M + ted_morning M + mike_afternoon + ted_afternoon

-- Statement to prove
theorem mike_planted_50 (M : ℕ) (h : total_planted M = 250) : M = 50 :=
by
  sorry

end mike_planted_50_l92_92138


namespace polynomial_factorization_l92_92315

theorem polynomial_factorization (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = (x^2 + 6*x + 1) * (x^2 + 6*x + 37) :=
  sorry

end polynomial_factorization_l92_92315


namespace probability_of_same_length_segments_l92_92152

-- Definitions directly from the conditions
def total_elements : ℕ := 6 + 9

-- Probability calculations
def probability_same_length : ℚ :=
  ((6 / total_elements) * (5 / (total_elements - 1))) + 
  ((9 / total_elements) * (8 / (total_elements - 1)))

-- Proof statement
theorem probability_of_same_length_segments : probability_same_length = 17 / 35 := by
  unfold probability_same_length
  sorry

end probability_of_same_length_segments_l92_92152


namespace total_carriages_l92_92708

theorem total_carriages (Euston Norfolk Norwich FlyingScotsman : ℕ) 
  (h1 : Euston = 130)
  (h2 : Norfolk = Euston - 20)
  (h3 : Norwich = 100)
  (h4 : FlyingScotsman = Norwich + 20) :
  Euston + Norfolk + Norwich + FlyingScotsman = 460 :=
by 
  sorry

end total_carriages_l92_92708


namespace max_one_truthful_dwarf_l92_92404

def dwarf_height_claims : List (ℕ × ℕ) := [
  (1, 60),  -- First dwarf claims 60 cm
  (2, 61),  -- Second dwarf claims 61 cm
  (3, 62),  -- Third dwarf claims 62 cm
  (4, 63),  -- Fourth dwarf claims 63 cm
  (5, 64),  -- Fifth dwarf claims 64 cm
  (6, 65),  -- Sixth dwarf claims 65 cm
  (7, 66)   -- Seventh dwarf claims 66 cm
]

-- Proof problem statement: Prove that at most one dwarf is telling the truth about their height
theorem max_one_truthful_dwarf : ∀ (heights : List ℕ), heights.length = 7 → 
  (∀ i, i < heights.length → dwarf_height_claims.get? i ≠ none → (heights.get? i = dwarf_height_claims.get? i) → 
    ∑ i in finset.range heights.length, bool.to_num (heights.get? i = dwarf_height_claims.get? i) ≤ 1) :=
by
  intros heights len_heights valid_claims idx idx_in_range some_claim match_claim
  -- Add proof here 
  sorry

end max_one_truthful_dwarf_l92_92404


namespace projectile_first_reach_height_56_l92_92185

theorem projectile_first_reach_height_56 (t : ℝ) (h1 : ∀ t, y = -16 * t^2 + 60 * t) :
    (∃ t : ℝ, y = 56 ∧ t = 1.75 ∧ (∀ t', t' < 1.75 → y ≠ 56)) :=
by
  sorry

end projectile_first_reach_height_56_l92_92185


namespace positive_expressions_l92_92849

-- Define the approximate values for A, B, C, D, and E.
def A := 2.5
def B := -2.1
def C := -0.3
def D := 1.0
def E := -0.7

-- Define the expressions that we need to prove as positive numbers.
def exprA := A + B
def exprB := B * C
def exprD := E / (A * B)

-- The theorem states that expressions (A + B), (B * C), and (E / (A * B)) are positive.
theorem positive_expressions : exprA > 0 ∧ exprB > 0 ∧ exprD > 0 := 
by sorry

end positive_expressions_l92_92849


namespace range_f_l92_92987

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 2 * x + 1)

theorem range_f : (Set.range f) = Set.univ := by
  sorry

end range_f_l92_92987


namespace teams_points_l92_92535

-- Definitions of teams and points
inductive Team
| A | B | C | D | E
deriving DecidableEq

def points : Team → ℕ
| Team.A => 6
| Team.B => 5
| Team.C => 4
| Team.D => 3
| Team.E => 2

-- Conditions
axiom no_draws_A : ∀ t : Team, t ≠ Team.A → (points Team.A ≠ points t)
axiom no_loses_B : ∀ t : Team, t ≠ Team.B → (points Team.B > points t) ∨ (points Team.B = points t)
axiom no_wins_D : ∀ t : Team, t ≠ Team.D → (points Team.D < points t)
axiom unique_scores : ∀ (t1 t2 : Team), t1 ≠ t2 → points t1 ≠ points t2

-- Theorem
theorem teams_points :
  points Team.A = 6 ∧
  points Team.B = 5 ∧
  points Team.C = 4 ∧
  points Team.D = 3 ∧
  points Team.E = 2 :=
by
  sorry

end teams_points_l92_92535


namespace library_book_configurations_l92_92890

def number_of_valid_configurations (total_books : ℕ) (min_in_library : ℕ) (min_checked_out : ℕ) : ℕ :=
  (total_books - (min_in_library + min_checked_out + 1)) + 1

theorem library_book_configurations : number_of_valid_configurations 8 2 2 = 5 :=
by
  -- Here we would write the Lean proof, but since we are only interested in the statement:
  sorry

end library_book_configurations_l92_92890


namespace sam_driving_distance_l92_92803

-- Definitions based on the conditions
def marguerite_distance : ℝ := 150
def marguerite_time : ℝ := 3
def sam_time : ℝ := 4

-- Desired statement using the given conditions
theorem sam_driving_distance :
  let rate := marguerite_distance / marguerite_time in
  let sam_distance := rate * sam_time in
  sam_distance = 200 :=
by
  sorry

end sam_driving_distance_l92_92803


namespace find_salary_J_l92_92558

variables (J F M A May : ℝ)

def avg_salary_J_F_M_A (J F M A : ℝ) : Prop :=
  (J + F + M + A) / 4 = 8000

def avg_salary_F_M_A_May (F M A May : ℝ) : Prop :=
  (F + M + A + May) / 4 = 8700

def salary_May (May : ℝ) : Prop :=
  May = 6500

theorem find_salary_J (h1 : avg_salary_J_F_M_A J F M A) (h2 : avg_salary_F_M_A_May F M A May) (h3 : salary_May May) :
  J = 3700 :=
sorry

end find_salary_J_l92_92558


namespace three_digit_number_550_l92_92498

theorem three_digit_number_550 (N : ℕ) (a b c : ℕ) (h1 : N = 100 * a + 10 * b + c)
  (h2 : 1 ≤ a ∧ a ≤ 9) (h3 : b ≤ 9) (h4 : c ≤ 9) (h5 : 11 ∣ N)
  (h6 : N / 11 = a^2 + b^2 + c^2) : N = 550 :=
by
  sorry

end three_digit_number_550_l92_92498


namespace min_value_frac_l92_92768

theorem min_value_frac (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 3 * y = 1) :
  (1 / x + 1 / (3 * y)) = 4 :=
by
  sorry

end min_value_frac_l92_92768


namespace tan_plus_pi_over_4_l92_92770

variable (θ : ℝ)

-- Define the conditions
def condition_θ_interval : Prop := θ ∈ Set.Ioo (Real.pi / 2) Real.pi
def condition_sin_θ : Prop := Real.sin θ = 3 / 5

-- Define the theorem to be proved
theorem tan_plus_pi_over_4 (h1 : condition_θ_interval θ) (h2 : condition_sin_θ θ) :
  Real.tan (θ + Real.pi / 4) = 7 :=
sorry

end tan_plus_pi_over_4_l92_92770


namespace find_C_l92_92384

def A : ℝ × ℝ := (2, 8)
def M : ℝ × ℝ := (4, 11)
def L : ℝ × ℝ := (6, 6)

theorem find_C (C : ℝ × ℝ) (B : ℝ × ℝ) :
  -- Median condition: M is the midpoint of A and B
  M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  -- Given coordinates for A, M, L
  A = (2, 8) → M = (4, 11) → L = (6, 6) →
  -- Correct answer
  C = (14, 2) :=
by
  intros hmedian hA hM hL
  sorry

end find_C_l92_92384


namespace probability_of_exactly_one_defective_l92_92766

theorem probability_of_exactly_one_defective:
  let total_products := 6
  let genuine_products := 5
  let defective_products := 1
  -- selecting 2 products out of total_products
  let ways_to_choose_two := nat.choose total_products 2
  -- selecting 1 genuine and 1 defective product
  let ways_to_choose_one_genuine_one_defective := genuine_products * defective_products
  -- calculating probability
  (ways_to_choose_one_genuine_one_defective : ℚ) / ways_to_choose_two = 1 / 3 :=
by {
  sorry
}

end probability_of_exactly_one_defective_l92_92766


namespace juan_distance_l92_92386

def time : ℝ := 80.0
def speed : ℝ := 10.0
def distance (t : ℝ) (s : ℝ) : ℝ := t * s

theorem juan_distance : distance time speed = 800.0 := by
  sorry

end juan_distance_l92_92386


namespace probability_not_all_same_l92_92050

theorem probability_not_all_same (n : ℕ) (h : n = 5) : 
  let total_outcomes := 8^n in
  let same_number_outcomes := 8 in
  let prob_all_same_number := (same_number_outcomes : ℚ) / total_outcomes in
  let prob_not_all_same_number := 1 - prob_all_same_number in
  prob_not_all_same_number = 4095 / 4096 :=
by 
  sorry

end probability_not_all_same_l92_92050


namespace homogeneous_diff_eq_solution_l92_92661

open Real

theorem homogeneous_diff_eq_solution (C : ℝ) : 
  ∀ (x y : ℝ), (y^4 - 2 * x^3 * y) * (dx) + (x^4 - 2 * x * y^3) * (dy) = 0 ↔ x^3 + y^3 = C * x * y :=
by
  sorry

end homogeneous_diff_eq_solution_l92_92661


namespace factor_poly_eq_factored_form_l92_92265

-- Defining the polynomial expressions
def poly1 (x : ℝ) := x^2 + 4 * x + 3
def poly2 (x : ℝ) := x^2 + 8 * x + 15
def poly3 (x : ℝ) := x^2 + 6 * x - 8

-- The main expression which needs to be factored
def main_expr (x : ℝ) := (poly1 x) * (poly2 x) + (poly3 x)

-- Stating the goal factored form
def factored_form (x : ℝ) := (x^2 + 6 * x + 19) * (x^2 + 6 * x - 2)

-- The theorem statement
theorem factor_poly_eq_factored_form (x : ℝ) : 
  main_expr x = factored_form x := 
by
  sorry

end factor_poly_eq_factored_form_l92_92265


namespace factorization_correct_l92_92290

noncomputable def polynomial_expr := 
  λ x : ℝ => (x^2 + 4 * x + 3) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8)

noncomputable def factored_expr := 
  λ x : ℝ => (x^2 + 6 * x + 9) * (x^2 + 6 * x + 1)

theorem factorization_correct : 
  ∀ x : ℝ, polynomial_expr x = factored_expr x :=
by
  intro x
  sorry

end factorization_correct_l92_92290


namespace prove_expression_value_l92_92780

-- Define the conditions
variables {a b c d m : ℤ}
variable (h1 : a + b = 0)
variable (h2 : |m| = 2)
variable (h3 : c * d = 1)

-- State the theorem
theorem prove_expression_value : (a + b) / (4 * m) + 2 * m ^ 2 - 3 * c * d = 5 :=
by
  -- Proof goes here
  sorry

end prove_expression_value_l92_92780


namespace algebraic_expression_problem_l92_92128

-- Define the conditions and the target statement to verify.
theorem algebraic_expression_problem (x : ℝ) 
  (h : x^2 + 3 * x + 5 = 7) : 3 * x^2 + 9 * x - 2 = 4 :=
by 
  -- Add sorry to skip the proof.
  sorry

end algebraic_expression_problem_l92_92128


namespace find_f_65_l92_92009

theorem find_f_65 (f : ℝ → ℝ) (h_eq : ∀ x y : ℝ, f (x * y) = x * f y) (h_f1 : f 1 = 40) : f 65 = 2600 :=
by
  sorry

end find_f_65_l92_92009


namespace find_AD_l92_92400

noncomputable def A := 0
noncomputable def C := 3
noncomputable def B (x : ℝ) := C - x
noncomputable def D (x : ℝ) := A + 3 + x

-- conditions
def AC := 3
def BD := 4
def ratio_condition (x : ℝ) := (A + C - x - (A + 3)) / x = (A + 3 + x) / x

-- theorem statement
theorem find_AD (x : ℝ) (h1 : AC = 3) (h2 : BD = 4) (h3 : ratio_condition x) :
  D x = 6 :=
sorry

end find_AD_l92_92400


namespace shape_is_spiral_l92_92506

-- Assume cylindrical coordinates and constants.
variables (c : ℝ)
-- Define cylindrical coordinate properties.
variables (r θ z : ℝ)

-- Define the equation rθ = c.
def cylindrical_equation : Prop := r * θ = c

theorem shape_is_spiral (h : cylindrical_equation c r θ):
  ∃ f : ℝ → ℝ, ∀ θ > 0, r = f θ ∧ (∀ θ₁ θ₂, θ₁ < θ₂ ↔ f θ₁ > f θ₂) :=
sorry

end shape_is_spiral_l92_92506


namespace perfect_square_trinomial_m_l92_92522

theorem perfect_square_trinomial_m (m : ℤ) : (∀ x : ℤ, ∃ k : ℤ, x^2 + 2*m*x + 9 = (x + k)^2) ↔ m = 3 ∨ m = -3 :=
by
  sorry

end perfect_square_trinomial_m_l92_92522


namespace sum_of_reciprocals_of_squares_of_roots_l92_92761

noncomputable def reciprocal_squares_sum (p : Polynomial ℝ) : ℝ :=
  let roots := p.roots
  if h : roots.length = 4 then
    let r1, r2, r3, r4 := roots.nth_le 0 sorry, roots.nth_le 1 sorry, roots.nth_le 2 sorry, roots.nth_le 3 sorry
    (1 / (r1 ^ 2)) + (1 / (r2 ^ 2)) + (1 / (r3 ^ 2)) + (1 / (r4 ^ 2))
  else 0

theorem sum_of_reciprocals_of_squares_of_roots :
  let p : Polynomial ℝ := Polynomial.C (1 : ℝ) + Polynomial.X ^ 4 - 2 * Polynomial.C (1 : ℝ) * Polynomial.X ^ 3 + 
                          6 * Polynomial.C (1 : ℝ) * Polynomial.X ^ 2 - 2 * Polynomial.C (1 : ℝ) * Polynomial.X + 1
  reciprocal_squares_sum p = -8 := 
sorry

end sum_of_reciprocals_of_squares_of_roots_l92_92761


namespace slope_range_l92_92017

theorem slope_range (α : Real) (hα : -1 ≤ Real.cos α ∧ Real.cos α ≤ 1) :
  ∃ k ∈ Set.Icc (- Real.sqrt 3 / 3) (Real.sqrt 3 / 3), ∀ x y : Real, x * Real.cos α - Real.sqrt 3 * y - 2 = 0 → y = k * x - (2 / Real.sqrt 3) :=
by
  sorry

end slope_range_l92_92017


namespace next_correct_time_l92_92883

def clock_shows_correct_time (start_date : String) (start_time : String) (time_lost_per_hour : Int) : String :=
  if start_date = "March 21" ∧ start_time = "12:00 PM" ∧ time_lost_per_hour = 25 then
    "June 1, 12:00 PM"
  else
    "unknown"

theorem next_correct_time :
  clock_shows_correct_time "March 21" "12:00 PM" 25 = "June 1, 12:00 PM" :=
by sorry

end next_correct_time_l92_92883


namespace factor_polynomial_l92_92259

theorem factor_polynomial (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = 
  (x^2 + 6*x + 4) * (x^2 + 6*x + 11) :=
by
  sorry

end factor_polynomial_l92_92259


namespace boys_play_football_l92_92375

theorem boys_play_football (total_boys basketball_players neither_players both_players : ℕ)
    (h_total : total_boys = 22)
    (h_basketball : basketball_players = 13)
    (h_neither : neither_players = 3)
    (h_both : both_players = 18) : total_boys - neither_players - both_players + (both_players - basketball_players) = 19 :=
by
  sorry

end boys_play_football_l92_92375


namespace george_boxes_of_eggs_l92_92347

theorem george_boxes_of_eggs (boxes_eggs : Nat) (h1 : ∀ (eggs_per_box : Nat), eggs_per_box = 3 → boxes_eggs * eggs_per_box = 15) :
  boxes_eggs = 5 :=
by
  sorry

end george_boxes_of_eggs_l92_92347


namespace min_value_x_y_l92_92541

theorem min_value_x_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 19 / x + 98 / y = 1) : x + y ≥ 117 + 14 * Real.sqrt 38 := 
sorry

end min_value_x_y_l92_92541


namespace total_number_of_outfits_l92_92424

-- Definitions of the conditions as functions/values
def num_shirts : Nat := 8
def num_pants : Nat := 5
def num_ties_options : Nat := 4 + 1  -- 4 ties + 1 option for no tie
def num_belts_options : Nat := 2 + 1  -- 2 belts + 1 option for no belt

-- Lean statement to formulate the proof problem
theorem total_number_of_outfits : 
  num_shirts * num_pants * num_ties_options * num_belts_options = 600 := by
  sorry

end total_number_of_outfits_l92_92424


namespace fraction_same_ratio_l92_92467

theorem fraction_same_ratio (x : ℚ) : 
  (x / (2 / 5)) = (3 / 7) / (6 / 5) ↔ x = 1 / 7 :=
by
  sorry

end fraction_same_ratio_l92_92467


namespace Jane_age_l92_92178

theorem Jane_age (x : ℕ) 
  (h1 : ∃ n1 : ℕ, x - 1 = n1 ^ 2) 
  (h2 : ∃ n2 : ℕ, x + 1 = n2 ^ 3) : 
  x = 26 :=
sorry

end Jane_age_l92_92178


namespace division_result_l92_92065

def numerator : ℕ := 3 * 4 * 5
def denominator : ℕ := 2 * 3
def quotient : ℕ := numerator / denominator

theorem division_result : quotient = 10 := by
  sorry

end division_result_l92_92065


namespace eval_dagger_l92_92753

noncomputable def dagger (m n p q : ℕ) : ℚ := 
  (m * p) * (q / n)

theorem eval_dagger : dagger 5 16 12 5 = 75 / 4 := 
by 
  sorry

end eval_dagger_l92_92753


namespace range_of_m_l92_92679

def P (m : ℝ) : Prop :=
  ∃ (x1 x2 : ℝ), (x1 ≠ x2) ∧ (x1 ^ 2 + m * x1 + 1 = 0) ∧ (x2 ^ 2 + m * x2 + 1 = 0) ∧ (x1 < 0) ∧ (x2 < 0)

def Q (m : ℝ) : Prop :=
  ∀ (x : ℝ), 4 * x ^ 2 + 4 * (m - 2) * x + 1 ≠ 0

def P_or_Q (m : ℝ) : Prop :=
  P m ∨ Q m

def P_and_Q (m : ℝ) : Prop :=
  P m ∧ Q m

theorem range_of_m (m : ℝ) : P_or_Q m ∧ ¬P_and_Q m ↔ m < -2 ∨ (1 < m ∧ m ≤ 2) ∨ m ≥ 3 :=
by {
  sorry
}

end range_of_m_l92_92679


namespace shaded_area_10x12_floor_l92_92589

theorem shaded_area_10x12_floor :
  let tile_white_area := π + 1
  let tile_total_area := 4
  let tile_shaded_area := tile_total_area - tile_white_area
  let num_tiles := (10 / 2) * (12 / 2)
  let total_shaded_area := num_tiles * tile_shaded_area
  total_shaded_area = 90 - 30 * π :=
by
  let tile_white_area := π + 1
  let tile_total_area := 4
  let tile_shaded_area := tile_total_area - tile_white_area
  let num_tiles := (10 / 2) * (12 / 2)
  let total_shaded_area := num_tiles * tile_shaded_area
  show total_shaded_area = 90 - 30 * π
  sorry

end shaded_area_10x12_floor_l92_92589


namespace identify_worst_player_l92_92374

-- Define the participants
inductive Participant
| father
| sister
| son
| daughter

open Participant

-- Conditions
def participants : List Participant :=
  [father, sister, son, daughter]

def twins (p1 p2 : Participant) : Prop := 
  (p1 = father ∧ p2 = sister) ∨
  (p1 = sister ∧ p2 = father) ∨
  (p1 = son ∧ p2 = daughter) ∨
  (p1 = daughter ∧ p2 = son)

def not_same_sex (p1 p2 : Participant) : Prop :=
  (p1 = father ∧ p2 = sister) ∨
  (p1 = sister ∧ p2 = father) ∨
  (p1 = son ∧ p2 = daughter) ∨
  (p1 = daughter ∧ p2 = son)

def older_by_one_year (p1 p2 : Participant) : Prop :=
  (p1 = father ∧ p2 = sister) ∨
  (p1 = sister ∧ p2 = father)

-- Question: who is the worst player?
def worst_player : Participant := sister

-- Proof statement
theorem identify_worst_player
  (h_twins : ∃ p1 p2, twins p1 p2)
  (h_not_same_sex : ∀ p1 p2, twins p1 p2 → not_same_sex p1 p2)
  (h_age_diff : ∀ p1 p2, twins p1 p2 → older_by_one_year p1 p2) :
  worst_player = sister :=
sorry

end identify_worst_player_l92_92374


namespace area_of_triangle_with_given_medians_l92_92434

noncomputable def area_of_triangle (m1 m2 m3 : ℝ) : ℝ :=
sorry

theorem area_of_triangle_with_given_medians :
    area_of_triangle 3 4 5 = 8 :=
sorry

end area_of_triangle_with_given_medians_l92_92434


namespace comparison_of_exponential_values_l92_92918

theorem comparison_of_exponential_values : 
  let a := 0.3^3
  let b := 3^0.3
  let c := 0.2^3
  c < a ∧ a < b := 
by 
  let a := 0.3^3
  let b := 3^0.3
  let c := 0.2^3
  sorry

end comparison_of_exponential_values_l92_92918


namespace isosceles_triangle_area_l92_92911

theorem isosceles_triangle_area (p x : ℝ) 
  (h1 : 2 * p = 6 * x) 
  (h2 : 0 < p) 
  (h3 : 0 < x) :
  (1 / 2) * (2 * x) * (Real.sqrt (8 * p^2 / 9)) = (Real.sqrt 8 * p^2) / 3 :=
by
  sorry

end isosceles_triangle_area_l92_92911


namespace square_minus_self_divisible_by_2_l92_92969

theorem square_minus_self_divisible_by_2 (a : ℕ) : 2 ∣ (a^2 - a) :=
by sorry

end square_minus_self_divisible_by_2_l92_92969


namespace find_c_l92_92343

theorem find_c (c : ℝ) (h : ∀ x, 2 < x ∧ x < 6 → -x^2 + c * x + 8 > 0) : c = 8 := 
by
  sorry

end find_c_l92_92343


namespace smallest_and_second_smallest_four_digit_numbers_divisible_by_35_l92_92860

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999
def divisible_by_35 (n : ℕ) : Prop := n % 35 = 0

theorem smallest_and_second_smallest_four_digit_numbers_divisible_by_35 :
  ∃ a b : ℕ, 
    is_four_digit a ∧ 
    is_four_digit b ∧ 
    divisible_by_35 a ∧ 
    divisible_by_35 b ∧ 
    a < b ∧ 
    ∀ c : ℕ, is_four_digit c → divisible_by_35 c → a ≤ c → (c = a ∨ c = b) :=
by
  sorry

end smallest_and_second_smallest_four_digit_numbers_divisible_by_35_l92_92860


namespace three_equal_of_four_l92_92765

theorem three_equal_of_four (a b c d : ℕ) 
  (h1 : (a + b)^2 ∣ c * d) 
  (h2 : (a + c)^2 ∣ b * d) 
  (h3 : (a + d)^2 ∣ b * c) 
  (h4 : (b + c)^2 ∣ a * d) 
  (h5 : (b + d)^2 ∣ a * c) 
  (h6 : (c + d)^2 ∣ a * b) : 
  (a = b ∧ b = c) ∨ (a = b ∧ b = d) ∨ (a = c ∧ c = d) ∨ (b = c ∧ c = d) := 
sorry

end three_equal_of_four_l92_92765


namespace inequality_am_gm_l92_92782

theorem inequality_am_gm (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^3 / (a^2 + a * b + b^2) + b^3 / (b^2 + b * c + c^2) + c^3 / (c^2 + c * a + a^2)) ≥ (a + b + c) / 3 :=
by
  sorry

end inequality_am_gm_l92_92782


namespace parabola_translation_correct_l92_92533

-- Define the original parabola equation
def original_parabola (x : ℝ) : ℝ := 3 * x^2

-- Given vertex translation
def translated_vertex : ℝ × ℝ := (-2, -2)

-- Define the translated parabola equation
def translated_parabola (x : ℝ) : ℝ := 3 * (x + 2)^2 - 2

-- The proof statement
theorem parabola_translation_correct :
  ∀ x, translated_parabola x = 3 * (x + 2)^2 - 2 := by
  sorry

end parabola_translation_correct_l92_92533


namespace spinner_prime_probability_l92_92702

def spinner_labels : List ℕ := [3, 6, 1, 4, 5, 2]

def total_outcomes : ℕ := spinner_labels.length

def is_prime (n : ℕ) : Bool := n = 2 ∨ n = 3 ∨ n = 5

def prime_count : ℕ := spinner_labels.countp is_prime

def probability_of_prime : ℚ := prime_count / total_outcomes

theorem spinner_prime_probability :
  probability_of_prime = 1 / 2 := by
  sorry

end spinner_prime_probability_l92_92702


namespace rectangle_area_x_l92_92425

theorem rectangle_area_x (x : ℕ) (h1 : x > 0) (h2 : 5 * x = 45) : x = 9 := 
by
  -- proof goes here
  sorry

end rectangle_area_x_l92_92425


namespace find_b_l92_92947

theorem find_b (S : ℕ → ℝ) (a : ℕ → ℝ) (h1 : ∀ n, S n = 3^n + b)
  (h2 : ∀ n ≥ 2, a n = S n - S (n-1))
  (h_geometric : ∃ r, ∀ n ≥ 1, a n = a 1 * r^(n-1)) : b = -1 := 
sorry

end find_b_l92_92947


namespace Ivan_uses_more_paint_l92_92453

noncomputable def Ivan_section_area : ℝ := 10

noncomputable def Petr_section_area (α : ℝ) : ℝ := 10 * Real.sin α

theorem Ivan_uses_more_paint (α : ℝ) (hα : Real.sin α < 1) : 
  Ivan_section_area > Petr_section_area α := 
by 
  rw [Ivan_section_area, Petr_section_area]
  linarith [hα]

end Ivan_uses_more_paint_l92_92453


namespace min_value_a_plus_one_over_a_minus_one_l92_92125

theorem min_value_a_plus_one_over_a_minus_one (a : ℝ) (h : a > 1) : 
  a + 1 / (a - 1) ≥ 3 ∧ (a = 2 → a + 1 / (a - 1) = 3) :=
by
  -- Translate the mathematical proof problem into a Lean 4 theorem statement.
  sorry

end min_value_a_plus_one_over_a_minus_one_l92_92125


namespace maximize_profit_l92_92200

noncomputable def profit (x : ℝ) : ℝ :=
  16 - 4/(x+1) - x

theorem maximize_profit (a : ℝ) (h : 0 ≤ a) :
  ∃ x : ℝ, 0 ≤ x ∧ x ≤ a ∧ profit x = max 13 (16 - 4/(a+1) - a) := by
  sorry

end maximize_profit_l92_92200


namespace merchant_spent_initially_500_rubles_l92_92725

theorem merchant_spent_initially_500_rubles
  (x : ℕ)
  (h1 : x + 100 > x)
  (h2 : x + 220 > x + 100)
  (h3 : x * (x + 220) = (x + 100) * (x + 100))
  : x = 500 := sorry

end merchant_spent_initially_500_rubles_l92_92725


namespace solve_inequality_l92_92001

theorem solve_inequality (a : ℝ) : 
  (if a = 0 ∨ a = 1 then { x : ℝ | false }
   else if a < 0 ∨ a > 1 then { x : ℝ | a < x ∧ x < a^2 }
   else if 0 < a ∧ a < 1 then { x : ℝ | a^2 < x ∧ x < a }
   else ∅) = 
  { x : ℝ | (x - a) / (x - a^2) < 0 } :=
by sorry

end solve_inequality_l92_92001


namespace b_minus_a_l92_92853

theorem b_minus_a (a b : ℕ) : (a * b = 2 * (a + b) + 12) → (b = 10) → (b - a = 6) :=
by
  sorry

end b_minus_a_l92_92853


namespace total_carriages_proof_l92_92706

noncomputable def total_carriages (E N' F N : ℕ) : ℕ :=
  E + N + N' + F

theorem total_carriages_proof
  (E N N' F : ℕ)
  (h1 : E = 130)
  (h2 : E = N + 20)
  (h3 : N' = 100)
  (h4 : F = N' + 20) :
  total_carriages E N' F N = 460 := by
  sorry

end total_carriages_proof_l92_92706


namespace flowers_per_bouquet_l92_92204

theorem flowers_per_bouquet (total_flowers wilted_flowers : ℕ) (bouquets : ℕ) (remaining_flowers : ℕ)
    (h1 : total_flowers = 45)
    (h2 : wilted_flowers = 35)
    (h3 : bouquets = 2)
    (h4 : remaining_flowers = total_flowers - wilted_flowers)
    (h5 : bouquets * (remaining_flowers / bouquets) = remaining_flowers) :
  remaining_flowers / bouquets = 5 :=
by
  sorry

end flowers_per_bouquet_l92_92204


namespace min_value_of_f_l92_92521

noncomputable def f (x : ℝ) : ℝ := 4 * x + 2 / x

theorem min_value_of_f (x : ℝ) (hx : x > 0) : ∃ y : ℝ, (∀ z : ℝ, z > 0 → f z ≥ y) ∧ y = 4 * Real.sqrt 2 :=
sorry

end min_value_of_f_l92_92521


namespace chessboard_fraction_sum_l92_92218

theorem chessboard_fraction_sum (r s m n : ℕ) (h_r : r = 1296) (h_s : s = 204) (h_frac : (17 : ℚ) / 108 = (s : ℕ) / (r : ℕ)) : m + n = 125 :=
sorry

end chessboard_fraction_sum_l92_92218


namespace fraction_of_fifth_set_l92_92387

theorem fraction_of_fifth_set :
  let total_match_duration := 11 * 60 + 5
  let fifth_set_duration := 8 * 60 + 11
  (fifth_set_duration : ℚ) / total_match_duration = 3 / 4 := 
sorry

end fraction_of_fifth_set_l92_92387


namespace yellow_surface_area_min_fraction_l92_92078

/-- 
  Given a larger cube with 4-inch edges, constructed from 64 smaller cubes (each with 1-inch edge),
  where 50 cubes are colored blue, and 14 cubes are colored yellow. 
  If the large cube is crafted to display the minimum possible yellow surface area externally,
  then the fraction of the surface area of the large cube that is yellow is 7/48.
-/
theorem yellow_surface_area_min_fraction (n_smaller_cubes blue_cubes yellow_cubes : ℕ) 
  (edge_small edge_large : ℕ) (surface_area_larger_cube yellow_surface_min : ℕ) :
  edge_small = 1 → edge_large = 4 → n_smaller_cubes = 64 → 
  blue_cubes = 50 → yellow_cubes = 14 →
  surface_area_larger_cube = 96 → yellow_surface_min = 14 → 
  (yellow_surface_min : ℚ) / (surface_area_larger_cube : ℚ) = 7 / 48 := 
by 
  intros h_edge_small h_edge_large h_n h_blue h_yellow h_surface_area h_yellow_surface
  sorry

end yellow_surface_area_min_fraction_l92_92078


namespace birds_more_than_nests_l92_92449

theorem birds_more_than_nests : 
  let birds := 6 
  let nests := 3 
  (birds - nests) = 3 := 
by 
  sorry

end birds_more_than_nests_l92_92449


namespace f_zero_add_f_neg_three_l92_92171

noncomputable def f : ℝ → ℝ :=
  sorry

axiom f_add (x y : ℝ) : f x + f y = f (x + y)

axiom f_three : f 3 = 4

theorem f_zero_add_f_neg_three : f 0 + f (-3) = -4 :=
by
  sorry

end f_zero_add_f_neg_three_l92_92171


namespace factor_polynomial_l92_92282

-- Define the necessary polynomials
def p (x : ℝ) : ℝ := x^2 + 4*x + 3
def q (x : ℝ) : ℝ := x^2 + 8*x + 15
def r (x : ℝ) : ℝ := x^2 + 6*x - 8

-- State the main theorem
theorem factor_polynomial (x : ℝ) : 
  (p x * q x) + r x = (x^2 + 7*x + 1) * (x^2 + 4*x + 7) :=
by
  sorry

end factor_polynomial_l92_92282


namespace find_unique_positive_integers_l92_92341

theorem find_unique_positive_integers (x y : ℕ) (hx : x > 0) (hy : y > 0) :
  3 ^ x + 7 = 2 ^ y → x = 2 ∧ y = 4 :=
by
  -- Proof will go here
  sorry

end find_unique_positive_integers_l92_92341


namespace probability_same_length_segments_of_regular_hexagon_l92_92145

theorem probability_same_length_segments_of_regular_hexagon (sides diagonals : ℕ) 
  (total_segments : ℕ) (prob_same_length : ℚ) 
  (h_sides : sides = 6)
  (h_diagonals : diagonals = 9)
  (h_total_segments : total_segments = sides + diagonals)
  : prob_same_length = 17 / 35 :=
by
  have total_combinations := total_segments * (total_segments - 1) / 2
  have prob_sides := (sides * (sides - 1) / 2 : ℚ) / total_combinations
  have prob_diagonals := (diagonals * (diagonals - 1) / 2 : ℚ) / total_combinations
  have prob_sum := prob_sides + prob_diagonals
  have h_prob_sides : prob_sides = (6 * 5 / 2) / (15 * 14 / 2) :=
    by sorry
  have h_prob_diagonals : prob_diagonals = (9 * 8 / 2) / (15 * 14 / 2) :=
    by sorry
  calc
    prob_same_length = prob_sum : by sorry
    ... = 17 / 35 : by {rw [←h_prob_sides, ←h_prob_diagonals], sorry}

end probability_same_length_segments_of_regular_hexagon_l92_92145


namespace value_of_a_l92_92528

noncomputable def f (a x : ℝ) : ℝ := a ^ x

theorem value_of_a (a : ℝ) (h : abs ((a^2) - a) = a / 2) : a = 1 / 2 ∨ a = 3 / 2 := by
  sorry

end value_of_a_l92_92528


namespace probability_not_all_same_l92_92037

theorem probability_not_all_same (n m : ℕ) (h₁ : n = 5) (h₂ : m = 8) 
  (fair_dice : ∀ (die : ℕ), 1 ≤ die ∧ die ≤ m)
  : (1 - (m / m^n) = 4095 / 4096) := by
  sorry

end probability_not_all_same_l92_92037


namespace hydrogen_atoms_in_compound_l92_92478

theorem hydrogen_atoms_in_compound : 
  ∀ (C O H : ℕ) (molecular_weight : ℕ), 
  C = 1 → 
  O = 3 → 
  molecular_weight = 62 → 
  (12 * C + 16 * O + H = molecular_weight) → 
  H = 2 := 
by
  intros C O H molecular_weight hc ho hmw hcalc
  sorry

end hydrogen_atoms_in_compound_l92_92478


namespace factor_polynomial_l92_92329

theorem factor_polynomial :
  ∀ x : ℝ, 
  (x^2 + 4 * x + 3) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8) 
  = (x^2 + 6 * x + 1) * (x^2 + 6 * x + 37) :=
by
  intros x
  sorry

end factor_polynomial_l92_92329


namespace no_solution_l92_92613

noncomputable def problem_statement : Prop :=
  ∀ x : ℝ, ¬((85 + x = 3.5 * (15 + x)) ∧ (55 + x = 2 * (15 + x)))

theorem no_solution : problem_statement :=
by
  intro x
  have h₁ : ¬(85 + x = 3.5 * (15 + x)) -> ¬((85 + x = 3.5 * (15 + x)) ∧ (55 + x = 2 * (15 + x))) := sorry
  have h₂ : ¬(55 + x = 2 * (15 + x)) -> ¬((85 + x = 3.5 * (15 + x)) ∧ (55 + x = 2 * (15 + x))) := sorry
  exact sorry

end no_solution_l92_92613


namespace coeff_of_neg_5ab_l92_92427

theorem coeff_of_neg_5ab : coefficient (-5 * (a * b)) = -5 :=
by
  sorry

end coeff_of_neg_5ab_l92_92427


namespace at_most_one_dwarf_tells_truth_l92_92411

noncomputable def dwarfs_tell_truth : Prop :=
  ∀ heights: Fin 7 → ℕ,
  heights = fun i => i + 60 →
  ∀ k ∈ Finset.range 7, (heights k) ≠ (60 + k) → (∃ i ∈ Finset.range 7, i ≠ k)

theorem at_most_one_dwarf_tells_truth :
  dwarfs_tell_truth = 1 := 
sorry

end at_most_one_dwarf_tells_truth_l92_92411


namespace line_intersects_circle_l92_92123

noncomputable def point := ℝ × ℝ

def line (p : point) : Prop := p.1 + p.2 = 2

def circle (p : point) : Prop := (p.1 - 1) ^ 2 + p.2 ^ 2 = 1

theorem line_intersects_circle :
  ∃ (p : point), line p ∧ circle p :=
sorry

end line_intersects_circle_l92_92123


namespace probability_sum_less_than_9_is_7_over_9_l92_92700

def dice_rolls : List (ℕ × ℕ) := 
  [ (i, j) | i ← [1, 2, 3, 4, 5, 6], j ← [1, 2, 3, 4, 5, 6] ]

def favorable_outcomes : List (ℕ × ℕ) :=
  dice_rolls.filter (λ p => p.1 + p.2 < 9)

def probability_sum_less_than_9 := 
  favorable_outcomes.length.toRat / dice_rolls.length.toRat

theorem probability_sum_less_than_9_is_7_over_9 : 
  probability_sum_less_than_9 = 7 / 9 :=
by
  sorry

end probability_sum_less_than_9_is_7_over_9_l92_92700


namespace bananas_left_l92_92734

theorem bananas_left (dozen_bananas : ℕ) (eaten_bananas : ℕ) (h1 : dozen_bananas = 12) (h2 : eaten_bananas = 2) : dozen_bananas - eaten_bananas = 10 :=
sorry

end bananas_left_l92_92734


namespace katie_spending_l92_92583

theorem katie_spending :
  let price_per_flower : ℕ := 6
  let number_of_roses : ℕ := 5
  let number_of_daisies : ℕ := 5
  let total_number_of_flowers := number_of_roses + number_of_daisies
  let total_spending := total_number_of_flowers * price_per_flower
  total_spending = 60 :=
by
  sorry

end katie_spending_l92_92583


namespace new_average_age_l92_92681

theorem new_average_age (avg_age : ℕ) (num_students : ℕ) (teacher_age : ℕ) (new_num_individuals : ℕ) (new_avg_age : ℕ) :
  avg_age = 15 ∧ num_students = 20 ∧ teacher_age = 36 ∧ new_num_individuals = 21 →
  new_avg_age = (num_students * avg_age + teacher_age) / new_num_individuals → new_avg_age = 16 :=
by
  intros
  sorry

end new_average_age_l92_92681


namespace solve_x_eqns_solve_y_eqns_l92_92552

theorem solve_x_eqns : ∀ x : ℝ, 2 * x^2 = 8 * x ↔ (x = 0 ∨ x = 4) :=
by
  intro x
  sorry

theorem solve_y_eqns : ∀ y : ℝ, y^2 - 10 * y - 1 = 0 ↔ (y = 5 + Real.sqrt 26 ∨ y = 5 - Real.sqrt 26) :=
by
  intro y
  sorry

end solve_x_eqns_solve_y_eqns_l92_92552


namespace ellipse_equation_angle_AFB_constant_l92_92349

noncomputable def ellipse : Prop := 
  ∃ (a b c : ℝ), 
    a > b ∧ b > 0 ∧ 
    2 * a = 4 ∧ 
    c = a * (√3 / 2) ∧ 
    b = √(a^2 - c^2) ∧ 
    (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1)

theorem ellipse_equation :
  ellipse → ∀ x y : ℝ, x^2 / 4 + y^2 = 1 := 
by {
  assume h,
  rcases h with ⟨a, b, c, ha, hb, h1, h2, h3, hₑq⟩,
  rw h1 at h1 ⊢,
  rw h2,
  exact hₑq,
  sorry
}

theorem angle_AFB_constant :
  ellipse → ∀ P : ℝ × ℝ, ¬ (P.fst = 2 ∨ P.fst = -2) →
  ∀ A B F : ℝ × ℝ, 
    A = (2, (1 - (P.fst / 2)) / P.snd) →
    B = (-2, (1 + (P.fst / 2)) / P.snd) →
    F = (√3, 0) →
    (atan ((A.snd - F.snd) / (A.fst - F.fst)) -
     atan ((B.snd - F.snd) / (B.fst - F.fst))
     ) = π / 2 := 
by {
  assume h P hP A B F hA hB hF,
  sorry
}

end ellipse_equation_angle_AFB_constant_l92_92349


namespace unique_intersection_l92_92689

theorem unique_intersection {m : ℝ} :
  (∃! y : ℝ, m = -3 * y^2 - 4 * y + 7) ↔ m = 25 / 3 :=
by
  sorry

end unique_intersection_l92_92689


namespace crayons_total_l92_92208

def crayons_per_child := 6
def number_of_children := 12
def total_crayons := 72

theorem crayons_total :
  crayons_per_child * number_of_children = total_crayons := by
  sorry

end crayons_total_l92_92208


namespace functions_with_inverses_l92_92648

-- Definitions for the conditions
def passes_Horizontal_Line_Test_A : Prop := false
def passes_Horizontal_Line_Test_B : Prop := true
def passes_Horizontal_Line_Test_C : Prop := true
def passes_Horizontal_Line_Test_D : Prop := false
def passes_Horizontal_Line_Test_E : Prop := false

-- Proof statement
theorem functions_with_inverses :
  (passes_Horizontal_Line_Test_A = false) ∧
  (passes_Horizontal_Line_Test_B = true) ∧
  (passes_Horizontal_Line_Test_C = true) ∧
  (passes_Horizontal_Line_Test_D = false) ∧
  (passes_Horizontal_Line_Test_E = false) →
  ([B, C] = which_functions_have_inverses) :=
sorry

end functions_with_inverses_l92_92648


namespace factor_polynomial_l92_92326

theorem factor_polynomial :
  ∀ x : ℝ, 
  (x^2 + 4 * x + 3) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8) 
  = (x^2 + 6 * x + 1) * (x^2 + 6 * x + 37) :=
by
  intros x
  sorry

end factor_polynomial_l92_92326


namespace winner_is_Junsu_l92_92465

def Younghee_water_intake : ℝ := 1.4
def Jimin_water_intake : ℝ := 1.8
def Junsu_water_intake : ℝ := 2.1

theorem winner_is_Junsu : 
  Junsu_water_intake > Younghee_water_intake ∧ Junsu_water_intake > Jimin_water_intake :=
by sorry

end winner_is_Junsu_l92_92465


namespace running_distance_l92_92880

theorem running_distance (D : ℕ) 
  (hA_time : ∀ (A_time : ℕ), A_time = 28) 
  (hB_time : ∀ (B_time : ℕ), B_time = 32) 
  (h_lead : ∀ (lead : ℕ), lead = 28) 
  (hA_speed : ∀ (A_speed : ℚ), A_speed = D / 28) 
  (hB_speed : ∀ (B_speed : ℚ), B_speed = D / 32) 
  (hB_dist : ∀ (B_dist : ℚ), B_dist = D - 28) 
  (h_eq : ∀ (B_dist : ℚ), B_dist = D * (28 / 32)) :
  D = 224 :=
by 
  sorry

end running_distance_l92_92880


namespace share_of_a_120_l92_92215

theorem share_of_a_120 (A B C : ℝ) 
  (h1 : A = (2 / 3) * (B + C)) 
  (h2 : B = (6 / 9) * (A + C)) 
  (h3 : A + B + C = 300) : 
  A = 120 := 
by 
  sorry

end share_of_a_120_l92_92215


namespace max_truthful_gnomes_l92_92413

theorem max_truthful_gnomes :
  ∀ (heights : Fin 7 → ℝ), 
    heights 0 = 60 →
    heights 1 = 61 →
    heights 2 = 62 →
    heights 3 = 63 →
    heights 4 = 64 →
    heights 5 = 65 →
    heights 6 = 66 →
      (∃ i : Fin 7, ∀ j : Fin 7, (i ≠ j → heights j ≠ (60 + j.1)) ∧ (i = j → heights j = (60 + j.1))) :=
by
  intro heights h1 h2 h3 h4 h5 h6 h7
  sorry

end max_truthful_gnomes_l92_92413


namespace gum_pieces_in_each_packet_l92_92834

theorem gum_pieces_in_each_packet
  (packets : ℕ) (chewed_pieces : ℕ) (remaining_pieces : ℕ) (total_pieces : ℕ)
  (h1 : packets = 8) (h2 : chewed_pieces = 54) (h3 : remaining_pieces = 2) (h4 : total_pieces = chewed_pieces + remaining_pieces)
  (h5 : total_pieces = packets * (total_pieces / packets)) :
  total_pieces / packets = 7 :=
by
  sorry

end gum_pieces_in_each_packet_l92_92834


namespace unique_intersection_l92_92688

theorem unique_intersection {m : ℝ} :
  (∃! y : ℝ, m = -3 * y^2 - 4 * y + 7) ↔ m = 25 / 3 :=
by
  sorry

end unique_intersection_l92_92688


namespace cos_arcsin_l92_92744

theorem cos_arcsin (x : ℝ) (hx : x = 3 / 5) : Real.cos (Real.arcsin x) = 4 / 5 := by
  sorry

end cos_arcsin_l92_92744


namespace find_value_of_c_l92_92870

-- Mathematical proof problem in Lean 4 statement
theorem find_value_of_c (a b c d : ℝ)
  (h1 : a + c = 900)
  (h2 : b + c = 1100)
  (h3 : a + d = 700)
  (h4 : a + b + c + d = 2000) : 
  c = 200 :=
sorry

end find_value_of_c_l92_92870


namespace five_eight_sided_dice_not_all_same_l92_92056

noncomputable def probability_not_all_same : ℚ :=
  let total_outcomes := 8^5
  let same_number_outcomes := 8
  1 - (same_number_outcomes / total_outcomes)

theorem five_eight_sided_dice_not_all_same :
  probability_not_all_same = 4095 / 4096 :=
by
  sorry

end five_eight_sided_dice_not_all_same_l92_92056


namespace fraction_of_lollipops_given_to_emily_is_2_3_l92_92832

-- Given conditions as definitions
def initial_lollipops := 42
def kept_lollipops := 4
def lou_received := 10

-- The fraction of lollipops given to Emily
def fraction_given_to_emily : ℚ :=
  have emily_received : ℚ := initial_lollipops - (kept_lollipops + lou_received)
  have total_lollipops : ℚ := initial_lollipops
  emily_received / total_lollipops

-- The proof statement assert that fraction_given_to_emily is equal to 2/3
theorem fraction_of_lollipops_given_to_emily_is_2_3 : fraction_given_to_emily = 2 / 3 := by
  sorry

end fraction_of_lollipops_given_to_emily_is_2_3_l92_92832


namespace gnomes_telling_the_truth_l92_92418

-- Conditions
def gnome_height_claims : List ℕ := [60, 61, 62, 63, 64, 65, 66]

-- Question and Proof Problem in Lean 4
theorem gnomes_telling_the_truth :
  (∀ (actual_heights : List ℕ), 
    actual_heights.length = 7 →
    (∀ i, i < 7 → i > 0 → 
    actual_heights.get! i > actual_heights.get! (i - 1) → gnome_height_claims.get! i ≠ actual_heights.get! i)) →
  -- conclusion
  (∃ count, count = 1) :=
by
  sorry

end gnomes_telling_the_truth_l92_92418


namespace sample_size_is_fifteen_l92_92721

variable (total_employees : ℕ) (young_employees : ℕ) (middle_aged_employees : ℕ)
variable (elderly_employees : ℕ) (young_sample_count : ℕ) (sample_size : ℕ)

theorem sample_size_is_fifteen
  (h1 : total_employees = 750)
  (h2 : young_employees = 350)
  (h3 : middle_aged_employees = 250)
  (h4 : elderly_employees = 150)
  (h5 : 7 = young_sample_count)
  : sample_size = 15 := 
sorry

end sample_size_is_fifteen_l92_92721


namespace jonathan_fourth_task_completion_l92_92385

-- Conditions
def start_time : Nat := 9 * 60 -- 9:00 AM in minutes
def third_task_completion_time : Nat := 11 * 60 + 30 -- 11:30 AM in minutes
def number_of_tasks : Nat := 4
def number_of_completed_tasks : Nat := 3

-- Calculation of time duration
def total_time_first_three_tasks : Nat :=
  third_task_completion_time - start_time

def duration_of_one_task : Nat :=
  total_time_first_three_tasks / number_of_completed_tasks
  
-- Statement to prove
theorem jonathan_fourth_task_completion :
  (third_task_completion_time + duration_of_one_task) = (12 * 60 + 20) :=
  by
    -- We do not need to provide the proof steps as per instructions
    sorry

end jonathan_fourth_task_completion_l92_92385


namespace triangle_third_side_length_l92_92176

theorem triangle_third_side_length
  (A B C : ℝ)
  (a b c : ℝ)
  (h1 : b = 10)
  (h2 : c = 7)
  (h3 : A = 2 * B) :
  a = (50 + 5 * Real.sqrt 2) / 7 ∨ a = (50 - 5 * Real.sqrt 2) / 7 :=
sorry

end triangle_third_side_length_l92_92176


namespace num_divisors_630_l92_92691

theorem num_divisors_630 : ∃ d : ℕ, (d = 24) ∧ ∀ n : ℕ, (∃ (a b c d : ℕ), (n = 2^a * 3^b * 5^c * 7^d) ∧ a ≤ 1 ∧ b ≤ 2 ∧ c ≤ 1 ∧ d ≤ 1) ↔ (n ∣ 630) := sorry

end num_divisors_630_l92_92691


namespace find_x_l92_92641

-- Define the angles AXB, CYX, and XYB as given in the problem.
def angle_AXB : ℝ := 150
def angle_CYX : ℝ := 130
def angle_XYB : ℝ := 55

-- Define a function that represents the sum of angles in a triangle.
def sum_of_angles_in_triangle (a b c : ℝ) : Prop := a + b + c = 180

-- Define the angles.
def angle_XYZ : ℝ := angle_AXB - angle_XYB
def angle_YXZ : ℝ := 180 - angle_CYX
def angle_YXZ_proof (x : ℝ) : Prop := sum_of_angles_in_triangle angle_XYZ angle_YXZ x

-- State the theorem to be proved.
theorem find_x : angle_YXZ_proof 35 :=
sorry

end find_x_l92_92641


namespace missing_number_l92_92074

theorem missing_number (x : ℝ) : (306 / x) * 15 + 270 = 405 ↔ x = 34 := 
by
  sorry

end missing_number_l92_92074


namespace factor_polynomial_l92_92324

theorem factor_polynomial :
  ∀ x : ℝ, 
  (x^2 + 4 * x + 3) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8) 
  = (x^2 + 6 * x + 1) * (x^2 + 6 * x + 37) :=
by
  intros x
  sorry

end factor_polynomial_l92_92324


namespace intersection_complement_B_l92_92961

-- Define the sets A and B
def A : Set ℝ := { x | x^2 - 3 * x < 0 }
def B : Set ℝ := { x | abs x > 2 }

-- Complement of B
def complement_B : Set ℝ := { x | x ≥ -2 ∧ x ≤ 2 }

-- Final statement to prove the intersection equals the given set
theorem intersection_complement_B :
  A ∩ complement_B = { x : ℝ | 0 < x ∧ x ≤ 2 } := 
by 
  -- Proof omitted
  sorry

end intersection_complement_B_l92_92961


namespace product_of_third_side_l92_92201

/-- Two sides of a right triangle have lengths 5 and 7. The product of the possible lengths of 
the third side is exactly √1776. -/
theorem product_of_third_side :
  let a := 5
  let b := 7
  (Real.sqrt (a^2 + b^2) * Real.sqrt (b^2 - a^2)) = Real.sqrt 1776 := 
by 
  let a := 5
  let b := 7
  sorry

end product_of_third_side_l92_92201


namespace tan_alpha_eq_2_implies_sin_2alpha_eq_4_over_5_l92_92124

variable {α : Real}

theorem tan_alpha_eq_2_implies_sin_2alpha_eq_4_over_5 (h : Real.tan α = 2) : Real.sin (2 * α) = 4 / 5 := 
by
  sorry

end tan_alpha_eq_2_implies_sin_2alpha_eq_4_over_5_l92_92124


namespace jameson_total_medals_l92_92952

-- Define the number of track, swimming, and badminton medals
def track_medals := 5
def swimming_medals := 2 * track_medals
def badminton_medals := 5

-- Define the total number of medals
def total_medals := track_medals + swimming_medals + badminton_medals

-- Theorem statement
theorem jameson_total_medals : total_medals = 20 := 
by
  sorry

end jameson_total_medals_l92_92952


namespace simplify_frac_l92_92421

theorem simplify_frac : (5^4 + 5^2) / (5^3 - 5) = 65 / 12 :=
by 
  sorry

end simplify_frac_l92_92421


namespace sam_distance_traveled_l92_92813

-- Variables definition
variables (distance_marguerite : ℝ) (time_marguerite : ℝ) (time_sam : ℝ)

-- Given conditions
def marguerite_conditions : Prop :=
  distance_marguerite = 150 ∧
  time_marguerite = 3 ∧
  time_sam = 4

-- Statement to prove
theorem sam_distance_traveled (h : marguerite_conditions distance_marguerite time_marguerite time_sam) : 
  distance_marguerite / time_marguerite * time_sam = 200 :=
sorry

end sam_distance_traveled_l92_92813


namespace gcd_problem_l92_92523

theorem gcd_problem : ∃ b : ℕ, gcd (20 * b) (18 * 24) = 2 :=
by { sorry }

end gcd_problem_l92_92523


namespace problem1_problem2_l92_92639

-- Problem (1)
theorem problem1 (a : ℝ) (h : a = 1) (p q : ℝ → Prop) 
  (hp : ∀ x, p x ↔ x^2 - 4*a*x + 3*a^2 < 0) 
  (hq : ∀ x, q x ↔ (x - 3)^2 < 1) :
  (∀ x, (p x ∧ q x) ↔ (2 < x ∧ x < 3)) :=
by sorry

-- Problem (2)
theorem problem2 (a : ℝ) (p q : ℝ → Prop)
  (hp : ∀ x, p x ↔ x^2 - 4*a*x + 3*a^2 < 0)
  (hq : ∀ x, q x ↔ (x - 3)^2 < 1)
  (hnpc : ∀ x, ¬p x → ¬q x) 
  (hnpc_not_necessary : ∃ x, ¬p x ∧ q x) :
  (4 / 3 ≤ a ∧ a ≤ 2) :=
by sorry

end problem1_problem2_l92_92639


namespace encyclopedia_total_pages_l92_92547

noncomputable def totalPages : ℕ :=
450 + 3 * 90 +
650 + 5 * 68 +
712 + 4 * 75 +
820 + 6 * 120 +
530 + 2 * 110 +
900 + 7 * 95 +
680 + 4 * 80 +
555 + 3 * 180 +
990 + 5 * 53 +
825 + 6 * 150 +
410 + 2 * 200 +
1014 + 7 * 69

theorem encyclopedia_total_pages : totalPages = 13659 := by
  sorry

end encyclopedia_total_pages_l92_92547


namespace interval_of_defined_expression_l92_92633

theorem interval_of_defined_expression (x : ℝ) :
  (x > 2 ∧ x < 5) ↔ (x - 2 > 0 ∧ 5 - x > 0) :=
by
  sorry

end interval_of_defined_expression_l92_92633


namespace least_people_cheaper_second_caterer_l92_92177

noncomputable def cost_first_caterer (x : ℕ) : ℕ := 50 + 18 * x

noncomputable def cost_second_caterer (x : ℕ) : ℕ := 
  if x >= 30 then 150 + 15 * x else 180 + 15 * x

theorem least_people_cheaper_second_caterer : ∃ x : ℕ, x = 34 ∧ x >= 30 ∧ cost_second_caterer x < cost_first_caterer x :=
by
  sorry

end least_people_cheaper_second_caterer_l92_92177


namespace solve_for_b_l92_92000

theorem solve_for_b (b x : ℚ)
  (h₁ : 3 * x + 5 = 1)
  (h₂ : b * x + 6 = 0) :
  b = 9 / 2 :=
sorry   -- The proof is omitted as per instruction.

end solve_for_b_l92_92000


namespace proved_problem_l92_92511

theorem proved_problem (x y p n k : ℕ) (h_eq : x^n + y^n = p^k)
  (h1 : n > 1)
  (h2 : n % 2 = 1)
  (h3 : Nat.Prime p)
  (h4 : p % 2 = 1) :
  ∃ l : ℕ, n = p^l :=
by sorry

end proved_problem_l92_92511


namespace identify_urea_decomposing_bacteria_l92_92561

-- Definitions of different methods
def methodA (culture_medium : Type) := 
  ∃ (urea_only : culture_medium) (phenol_red : culture_medium), phenol_red = urea_only

def methodB (culture_medium : Type) := 
  ∃ (urea_only : culture_medium) (EMB_reagent : culture_medium), EMB_reagent = urea_only

def methodC (culture_medium : Type) := 
  ∃ (urea_only : culture_medium) (Sudan_III : culture_medium), Sudan_III = urea_only

def methodD (culture_medium : Type) := 
  ∃ (urea_only : culture_medium) (Biuret_reagent : culture_medium), Biuret_reagent = urea_only

-- The proof problem statement
theorem identify_urea_decomposing_bacteria (culture_medium : Type) :
  methodA culture_medium :=
sorry

end identify_urea_decomposing_bacteria_l92_92561


namespace distinct_complex_roots_A_eq_neg7_l92_92246

theorem distinct_complex_roots_A_eq_neg7 (x₁ x₂ : ℂ) (A : ℝ) (hx1: x₁ ≠ x₂)
  (h1 : x₁ * (x₁ + 1) = A)
  (h2 : x₂ * (x₂ + 1) = A)
  (h3 : x₁^4 + 3 * x₁^3 + 5 * x₁ = x₂^4 + 3 * x₂^3 + 5 * x₂) : A = -7 := 
sorry

end distinct_complex_roots_A_eq_neg7_l92_92246


namespace joan_kittens_remaining_l92_92537

def original_kittens : ℕ := 8
def kittens_given_away : ℕ := 2

theorem joan_kittens_remaining : original_kittens - kittens_given_away = 6 := by
  sorry

end joan_kittens_remaining_l92_92537


namespace product_of_geometric_sequence_l92_92950

theorem product_of_geometric_sequence (x y z : ℝ) 
  (h_seq : ∃ r, x = r * 1 ∧ y = r * x ∧ z = r * y ∧ 4 = r * z) : 
  1 * x * y * z * 4 = 32 :=
by
  sorry

end product_of_geometric_sequence_l92_92950


namespace factorization_correct_l92_92289

noncomputable def polynomial_expr := 
  λ x : ℝ => (x^2 + 4 * x + 3) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8)

noncomputable def factored_expr := 
  λ x : ℝ => (x^2 + 6 * x + 9) * (x^2 + 6 * x + 1)

theorem factorization_correct : 
  ∀ x : ℝ, polynomial_expr x = factored_expr x :=
by
  intro x
  sorry

end factorization_correct_l92_92289


namespace factorization_identity_l92_92337

-- We are asked to prove the mathematical equality under given conditions.
theorem factorization_identity (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = (x^2 + 6*x + 8)^2 :=
by
  sorry

end factorization_identity_l92_92337


namespace work_related_emails_count_l92_92600

-- Definitions based on the identified conditions and the question
def total_emails : ℕ := 1200
def spam_percentage : ℕ := 27
def promotional_percentage : ℕ := 18
def social_percentage : ℕ := 15

-- The statement to prove, indicated the goal
theorem work_related_emails_count :
  (total_emails * (100 - spam_percentage - promotional_percentage - social_percentage)) / 100 = 480 :=
by
  sorry

end work_related_emails_count_l92_92600


namespace anya_hair_growth_l92_92905

theorem anya_hair_growth (wash_loss : ℕ) (brush_loss : ℕ) (total_loss : ℕ) : wash_loss = 32 → brush_loss = wash_loss / 2 → total_loss = wash_loss + brush_loss → total_loss + 1 = 49 :=
by
  sorry

end anya_hair_growth_l92_92905


namespace no_valid_n_for_three_digit_conditions_l92_92344

theorem no_valid_n_for_three_digit_conditions :
  ∃ (n : ℕ) (h₁ : 100 ≤ n / 4 ∧ n / 4 ≤ 999) (h₂ : 100 ≤ 4 * n ∧ 4 * n ≤ 999), false :=
by sorry

end no_valid_n_for_three_digit_conditions_l92_92344


namespace factor_poly_eq_factored_form_l92_92270

-- Defining the polynomial expressions
def poly1 (x : ℝ) := x^2 + 4 * x + 3
def poly2 (x : ℝ) := x^2 + 8 * x + 15
def poly3 (x : ℝ) := x^2 + 6 * x - 8

-- The main expression which needs to be factored
def main_expr (x : ℝ) := (poly1 x) * (poly2 x) + (poly3 x)

-- Stating the goal factored form
def factored_form (x : ℝ) := (x^2 + 6 * x + 19) * (x^2 + 6 * x - 2)

-- The theorem statement
theorem factor_poly_eq_factored_form (x : ℝ) : 
  main_expr x = factored_form x := 
by
  sorry

end factor_poly_eq_factored_form_l92_92270


namespace sqrt_3x_eq_5x_largest_value_l92_92030

theorem sqrt_3x_eq_5x_largest_value (x : ℝ) (h : Real.sqrt (3 * x) = 5 * x) : x ≤ 3 / 25 := 
by
  sorry

end sqrt_3x_eq_5x_largest_value_l92_92030


namespace universal_inequality_l92_92549

theorem universal_inequality (x y : ℝ) : x^2 + y^2 ≥ 2 * x * y := 
by 
  sorry

end universal_inequality_l92_92549


namespace find_denominator_x_l92_92067

noncomputable def sum_fractions : ℝ := 
    3.0035428163476343

noncomputable def fraction1 (x : ℝ) : ℝ :=
    2007 / x

noncomputable def fraction2 : ℝ :=
    8001 / 5998

noncomputable def fraction3 : ℝ :=
    2001 / 3999

-- Problem statement in Lean
theorem find_denominator_x (x : ℝ) :
  sum_fractions = fraction1 x + fraction2 + fraction3 ↔ x = 1717 :=
by sorry

end find_denominator_x_l92_92067


namespace water_depth_is_12_feet_l92_92598

variable (Ron_height Dean_height Water_depth : ℕ)

-- Given conditions
axiom H1 : Ron_height = 14
axiom H2 : Dean_height = Ron_height - 8
axiom H3 : Water_depth = 2 * Dean_height

-- Prove that the water depth is 12 feet
theorem water_depth_is_12_feet : Water_depth = 12 :=
by
  sorry

end water_depth_is_12_feet_l92_92598


namespace parabola_directrix_eq_neg_2_l92_92617

noncomputable def parabola_directrix (a b c : ℝ) : ℝ :=
  (b^2 - 4 * a * c) / (4 * a)

theorem parabola_directrix_eq_neg_2 (x : ℝ) :
  parabola_directrix 1 (-4) 4 = -2 :=
by
  -- proof steps go here
  sorry

end parabola_directrix_eq_neg_2_l92_92617


namespace sum_difference_l92_92524

def sum_even (n : ℕ) : ℕ :=
  (n / 2) * (2 + n)

def sum_odd (n : ℕ) : ℕ :=
  (n / 2) * (1 + (n - 1))

theorem sum_difference : sum_even 100 - sum_odd 99 = 50 :=
by
  sorry

end sum_difference_l92_92524


namespace ratio_s_t_l92_92570

variable {b s t : ℝ}
variable (hb : b ≠ 0)
variable (h1 : s = -b / 8)
variable (h2 : t = -b / 4)

theorem ratio_s_t : s / t = 1 / 2 :=
by
  sorry

end ratio_s_t_l92_92570


namespace remainder_sum_first_150_div_11300_l92_92206

theorem remainder_sum_first_150_div_11300 :
  let n := 150 in
  let S := n * (n + 1) / 2 in
  S % 11300 = 25 :=
by
  let n := 150
  let S := n * (n + 1) / 2
  show S % 11300 = 25
  sorry

end remainder_sum_first_150_div_11300_l92_92206


namespace greatest_fourth_term_l92_92985

theorem greatest_fourth_term (a d : ℕ) (h1 : a > 0) (h2 : d > 0) 
  (h3 : 5 * a + 10 * d = 50) (h4 : a + 2 * d = 10) : 
  a + 3 * d = 14 :=
by {
  -- We introduced the given constraints and now need a proof
  sorry
}

end greatest_fourth_term_l92_92985


namespace ratio_is_one_half_l92_92571

noncomputable def ratio_of_intercepts (b : ℝ) (hb : b ≠ 0) : ℝ :=
  let s := -b / 8
  let t := -b / 4
  s / t

theorem ratio_is_one_half (b : ℝ) (hb : b ≠ 0) :
  ratio_of_intercepts b hb = 1 / 2 :=
by
  sorry

end ratio_is_one_half_l92_92571


namespace largest_x_value_satisfies_largest_x_value_l92_92027

theorem largest_x_value (x : ℚ) (h : Real.sqrt (3 * x) = 5 * x) : x ≤ 3 / 25 :=
sorry

theorem satisfies_largest_x_value : 
  ∃ x : ℚ, Real.sqrt (3 * x) = 5 * x ∧ x = 3 / 25 :=
sorry

end largest_x_value_satisfies_largest_x_value_l92_92027


namespace area_of_triangle_ABC_l92_92660

noncomputable def area_triangle_ABC (AF BE : ℝ) (angle_FGB : ℝ) : ℝ :=
  let FG := AF / 3
  let BG := (2 / 3) * BE
  let area_FGB := (1 / 2) * FG * BG * Real.sin angle_FGB
  6 * area_FGB

theorem area_of_triangle_ABC
  (AF BE : ℕ) (hAF : AF = 10) (hBE : BE = 15)
  (angle_FGB : ℝ) (h_angle_FGB : angle_FGB = Real.pi / 3) :
  area_triangle_ABC AF BE angle_FGB = 50 * Real.sqrt 3 :=
by
  simp [area_triangle_ABC, hAF, hBE, h_angle_FGB]
  sorry

end area_of_triangle_ABC_l92_92660


namespace circle_tangent_x_axis_at_origin_l92_92527

theorem circle_tangent_x_axis_at_origin (D E F : ℝ) :
  (∀ (x y : ℝ), x^2 + y^2 + D*x + E*y + F = 0 → (∃ r : ℝ, r^2 = x^2 + y^2) ∧ y = 0) →
  D = 0 ∧ E ≠ 0 ∧ F ≠ 0 := 
sorry

end circle_tangent_x_axis_at_origin_l92_92527


namespace anya_hair_growth_l92_92906

theorem anya_hair_growth (wash_loss : ℕ) (brush_loss : ℕ) (total_loss : ℕ) : wash_loss = 32 → brush_loss = wash_loss / 2 → total_loss = wash_loss + brush_loss → total_loss + 1 = 49 :=
by
  sorry

end anya_hair_growth_l92_92906


namespace polynomial_factorization_l92_92313

theorem polynomial_factorization (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = (x^2 + 6*x + 1) * (x^2 + 6*x + 37) :=
  sorry

end polynomial_factorization_l92_92313


namespace calculate_total_weight_AlBr3_l92_92091

-- Definitions for the atomic weights
def atomic_weight_Al : ℝ := 26.98
def atomic_weight_Br : ℝ := 79.90

-- Definition for the molecular weight of AlBr3
def molecular_weight_AlBr3 : ℝ := atomic_weight_Al + 3 * atomic_weight_Br

-- Number of moles
def number_of_moles : ℝ := 5

-- Total weight of 5 moles of AlBr3
def total_weight_5_moles_AlBr3 : ℝ := molecular_weight_AlBr3 * number_of_moles

-- Desired result
def expected_total_weight : ℝ := 1333.40

-- Statement to prove that total_weight_5_moles_AlBr3 equals the expected total weight
theorem calculate_total_weight_AlBr3 :
  total_weight_5_moles_AlBr3 = expected_total_weight :=
sorry

end calculate_total_weight_AlBr3_l92_92091


namespace f_inequality_l92_92420

variables {n1 n2 d : ℕ} (f : ℕ → ℕ → ℕ)

theorem f_inequality (hn1 : n1 > 0) (hn2 : n2 > 0) (hd : d > 0) :
  f (n1 * n2) d ≤ f n1 d + n1 * (f n2 d - 1) :=
sorry

end f_inequality_l92_92420


namespace sugar_cups_l92_92545

theorem sugar_cups (S : ℕ) (h1 : 21 = S + 8) : S = 13 := 
by { sorry }

end sugar_cups_l92_92545


namespace probability_not_all_same_l92_92055

theorem probability_not_all_same :
  (let total_outcomes := 8 ^ 5 in
   let same_number_outcomes := 8 in
   let probability_all_same := same_number_outcomes / total_outcomes in
   let probability_not_same := 1 - probability_all_same in
   probability_not_same = 4095 / 4096) :=
begin
  sorry
end

end probability_not_all_same_l92_92055


namespace base3_to_base10_l92_92611

theorem base3_to_base10 (d0 d1 d2 d3 d4 : ℕ)
  (h0 : d4 = 2)
  (h1 : d3 = 1)
  (h2 : d2 = 0)
  (h3 : d1 = 2)
  (h4 : d0 = 1) :
  d4 * 3^4 + d3 * 3^3 + d2 * 3^2 + d1 * 3^1 + d0 * 3^0 = 196 := by
  sorry

end base3_to_base10_l92_92611


namespace original_price_of_cycle_l92_92223

theorem original_price_of_cycle (P : ℝ) (h1 : P * 0.85 = 1190) : P = 1400 :=
by
  sorry

end original_price_of_cycle_l92_92223


namespace annual_interest_rate_l92_92224

-- Define the initial conditions
def P : ℝ := 5600
def A : ℝ := 6384
def t : ℝ := 2
def n : ℝ := 1

-- The theorem statement:
theorem annual_interest_rate : ∃ (r : ℝ), A = P * (1 + r / n) ^ (n * t) ∧ r = 0.067 :=
by 
  sorry -- proof goes here

end annual_interest_rate_l92_92224


namespace select_books_from_corner_l92_92183

def num_ways_to_select_books (n₁ n₂ k : ℕ) : ℕ :=
  if h₁ : k > n₁ ∧ k > n₂ then 0
  else if h₂ : k > n₂ then 1
  else if h₃ : k > n₁ then Nat.choose n₂ k
  else Nat.choose n₁ k + 2 * Nat.choose n₁ (k-1) * Nat.choose n₂ 1 + Nat.choose n₁ k * 0 +
    (Nat.choose n₂ 1 * Nat.choose n₂ (k-1)) + Nat.choose n₂ k * 1

theorem select_books_from_corner :
  num_ways_to_select_books 3 6 3 = 42 :=
by
  sorry

end select_books_from_corner_l92_92183


namespace calculate_expression_l92_92092

theorem calculate_expression :
  57.6 * (8 / 5) + 28.8 * (184 / 5) - 14.4 * 80 + 10.5 = 10.5 :=
by
  sorry

end calculate_expression_l92_92092


namespace celia_receives_correct_amount_of_aranha_l92_92749

def borboleta_to_tubarao (b : Int) : Int := 3 * b
def tubarao_to_periquito (t : Int) : Int := 2 * t
def periquito_to_aranha (p : Int) : Int := 3 * p
def macaco_to_aranha (m : Int) : Int := 4 * m
def cobra_to_periquito (c : Int) : Int := 3 * c

def celia_stickers_to_aranha (borboleta tubarao cobra periquito macaco : Int) : Int :=
  let borboleta_to_aranha := periquito_to_aranha (tubarao_to_periquito (borboleta_to_tubarao borboleta))
  let tubarao_to_aranha := periquito_to_aranha (tubarao_to_periquito tubarao)
  let cobra_to_aranha := periquito_to_aranha (cobra_to_periquito cobra)
  let periquito_to_aranha := periquito_to_aranha periquito
  let macaco_to_aranha := macaco_to_aranha macaco
  borboleta_to_aranha + tubarao_to_aranha + cobra_to_aranha + periquito_to_aranha + macaco_to_aranha

theorem celia_receives_correct_amount_of_aranha : 
  celia_stickers_to_aranha 4 5 3 6 6 = 171 := 
by
  simp only [celia_stickers_to_aranha, borboleta_to_tubarao, tubarao_to_periquito, periquito_to_aranha, cobra_to_periquito, macaco_to_aranha]
  -- Here we need to perform the arithmetic steps to verify the sum
  sorry -- This is the placeholder for the actual proof

end celia_receives_correct_amount_of_aranha_l92_92749


namespace asymptotes_of_hyperbola_l92_92012

-- Definitions
variables (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b)

-- Theorem: Equation of the asymptotes of the given hyperbola
theorem asymptotes_of_hyperbola (h_equiv : b = 2 * a) :
  ∀ x y : ℝ, 
    (x ≠ 0 ∧ y ≠ 0 ∧ (y = (2 : ℝ) * x ∨ y = - (2 : ℝ) * x)) ↔ (x, y) ∈ {p : ℝ × ℝ | (x^2 / a^2) - (y^2 / b^2) = 1} := 
sorry

end asymptotes_of_hyperbola_l92_92012


namespace max_truthful_dwarfs_l92_92408

theorem max_truthful_dwarfs :
  let heights := [60, 61, 62, 63, 64, 65, 66] in
  (∃ n, n = 1 ∧ ∀ i, i < 7 → i ≠ n → heights[i] ≠ (heights[i] + 1)) :=
by
  let heights := [60, 61, 62, 63, 64, 65, 66]
  existsi 1
  intro i
  intro h
  intro hi
  sorry

end max_truthful_dwarfs_l92_92408


namespace vehicle_capacity_rental_plans_l92_92379

variables (a b x y : ℕ)

/-- Conditions -/
axiom cond1 : 2*x + y = 11
axiom cond2 : x + 2*y = 13

/-- Resulting capacities for each vehicle type -/
theorem vehicle_capacity : 
  x = 3 ∧ y = 5 :=
by
  sorry

/-- Rental plans for transporting 33 tons of drugs -/
theorem rental_plans :
  3*a + 5*b = 33 ∧ ((a = 6 ∧ b = 3) ∨ (a = 1 ∧ b = 6)) :=
by
  sorry

end vehicle_capacity_rental_plans_l92_92379


namespace factorization_correct_l92_92287

noncomputable def polynomial_expr := 
  λ x : ℝ => (x^2 + 4 * x + 3) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8)

noncomputable def factored_expr := 
  λ x : ℝ => (x^2 + 6 * x + 9) * (x^2 + 6 * x + 1)

theorem factorization_correct : 
  ∀ x : ℝ, polynomial_expr x = factored_expr x :=
by
  intro x
  sorry

end factorization_correct_l92_92287


namespace John_sells_each_wig_for_five_dollars_l92_92538

theorem John_sells_each_wig_for_five_dollars
  (plays : ℕ)
  (acts_per_play : ℕ)
  (wigs_per_act : ℕ)
  (wig_cost : ℕ)
  (total_cost : ℕ)
  (sold_wigs_cost : ℕ)
  (remaining_wigs_cost : ℕ) :
  plays = 3 ∧
  acts_per_play = 5 ∧
  wigs_per_act = 2 ∧
  wig_cost = 5 ∧
  total_cost = 150 ∧
  remaining_wigs_cost = 110 ∧
  total_cost - remaining_wigs_cost = sold_wigs_cost →
  (sold_wigs_cost / (plays * acts_per_play * wigs_per_act - remaining_wigs_cost / wig_cost)) = wig_cost :=
by sorry

end John_sells_each_wig_for_five_dollars_l92_92538


namespace square_area_25_l92_92574

theorem square_area_25 (side_length : ℝ) (h_side_length : side_length = 5) : side_length * side_length = 25 := 
by
  rw [h_side_length]
  norm_num
  done

end square_area_25_l92_92574


namespace probability_of_picking_red_ball_l92_92988

theorem probability_of_picking_red_ball (w r : ℕ) 
  (h1 : r > w) 
  (h2 : r < 2 * w) 
  (h3 : 2 * w + 3 * r = 60) : 
  r / (w + r) = 7 / 11 :=
sorry

end probability_of_picking_red_ball_l92_92988


namespace factorization_identity_l92_92330

-- We are asked to prove the mathematical equality under given conditions.
theorem factorization_identity (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = (x^2 + 6*x + 8)^2 :=
by
  sorry

end factorization_identity_l92_92330


namespace perpendicular_tangents_at_x0_l92_92783

noncomputable def x0 := (36 : ℝ)^(1 / 3) / 6

theorem perpendicular_tangents_at_x0 :
  (∃ x0 : ℝ, (∃ f1 f2 : ℝ → ℝ,
    (∀ x, f1 x = x^2 - 1) ∧
    (∀ x, f2 x = 1 - x^3) ∧
    (2 * x0 * (-3 * x0^2) = -1)) ∧
    x0 = (36 : ℝ)^(1 / 3) / 6) := sorry

end perpendicular_tangents_at_x0_l92_92783


namespace interest_rate_of_second_part_l92_92082

theorem interest_rate_of_second_part 
  (total_sum : ℝ) (P2 : ℝ) (interest1_rate : ℝ) 
  (time1 : ℝ) (time2 : ℝ) (interest2_value : ℝ) : 
  (total_sum = 2704) → 
  (P2 = 1664) → 
  (interest1_rate = 0.03) → 
  (time1 = 8) → 
  (interest2_value = interest1_rate * (total_sum - P2) * time1) → 
  (time2 = 3) → 
  1664 * r * time2 = interest2_value → 
  r = 0.05 := 
by sorry

end interest_rate_of_second_part_l92_92082


namespace probability_of_same_length_l92_92163

-- Define the set T and its properties
def T : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
def sides : ℕ := 6
def diagonals : ℕ := 9
def total_elements : ℕ := 15
def probability_same_length : ℚ := 17 / 35

-- Lean 4 statement (theorem)
theorem probability_of_same_length: 
    ( ∃ (sides diagonals : ℕ), sides = 6 ∧ diagonals = 9 ∧ sides + diagonals = 15 →
                                              ∃ probability_same_length : ℚ, probability_same_length = 17 / 35) := 
sorry

end probability_of_same_length_l92_92163


namespace rounding_effect_l92_92357

/-- Given positive integers x, y, and z, and rounding scenarios, the
  approximation of x/y - z is necessarily less than its exact value
  when z is rounded up and x and y are rounded down. -/
theorem rounding_effect (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
(RoundXDown RoundYDown RoundZUp : ℕ → ℕ) 
(HRoundXDown : ∀ a, RoundXDown a ≤ a)
(HRoundYDown : ∀ a, RoundYDown a ≤ a)
(HRoundZUp : ∀ a, a ≤ RoundZUp a) :
  (RoundXDown x) / (RoundYDown y) - (RoundZUp z) < x / y - z :=
sorry

end rounding_effect_l92_92357


namespace find_a_l92_92542

open Set

def set_A (a : ℝ) : Set ℝ := {x | x^2 - a * x + a^2 - 19 = 0}
def set_B : Set ℝ := {2, 3}
def set_C : Set ℝ := {2, -4}

theorem find_a (a : ℝ) (haB : (set_A a) ∩ set_B ≠ ∅) (haC : (set_A a) ∩ set_C = ∅) : a = -2 :=
sorry

end find_a_l92_92542


namespace equal_phrases_impossible_l92_92565

-- Define the inhabitants and the statements they make.
def inhabitants : ℕ := 1234

-- Define what it means to be a knight or a liar.
inductive Person
| knight : Person
| liar : Person

-- Define the statements "He is a knight!" and "He is a liar!"
inductive Statement
| is_knight : Statement
| is_liar : Statement

-- Define the pairings and types of statements 
def pairings (inhabitant1 inhabitant2 : Person) : Statement :=
match inhabitant1, inhabitant2 with
| Person.knight, Person.knight => Statement.is_knight
| Person.liar, Person.liar => Statement.is_knight
| Person.knight, Person.liar => Statement.is_liar
| Person.liar, Person.knight => Statement.is_knight

-- Define the total number of statements
def total_statements (pairs : ℕ) : ℕ := 2 * pairs

-- Theorem stating the mathematical equivalent proof problem
theorem equal_phrases_impossible :
  ¬ ∃ n : ℕ, n = inhabitants / 2 ∧ total_statements n = inhabitants ∧
    (pairings Person.knight Person.liar = Statement.is_knight ∧
     pairings Person.liar Person.knight = Statement.is_knight ∧
     (pairings Person.knight Person.knight = Statement.is_knight ∧
      pairings Person.liar Person.liar = Statement.is_knight) ∨
      (pairings Person.knight Person.liar = Statement.is_liar ∧
       pairings Person.liar Person.knight = Statement.is_liar)) :=
sorry

end equal_phrases_impossible_l92_92565


namespace probability_of_same_length_l92_92162

-- Define the set T and its properties
def T : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
def sides : ℕ := 6
def diagonals : ℕ := 9
def total_elements : ℕ := 15
def probability_same_length : ℚ := 17 / 35

-- Lean 4 statement (theorem)
theorem probability_of_same_length: 
    ( ∃ (sides diagonals : ℕ), sides = 6 ∧ diagonals = 9 ∧ sides + diagonals = 15 →
                                              ∃ probability_same_length : ℚ, probability_same_length = 17 / 35) := 
sorry

end probability_of_same_length_l92_92162


namespace cost_of_cookies_l92_92248

theorem cost_of_cookies (diane_has : ℕ) (needs_more : ℕ) (cost : ℕ) :
  diane_has = 27 → needs_more = 38 → cost = 65 :=
by
  sorry

end cost_of_cookies_l92_92248


namespace maximum_possible_value_of_expression_l92_92946

theorem maximum_possible_value_of_expression :
  ∀ (a b c d : ℕ), (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧
  (a = 0 ∨ a = 1 ∨ a = 3 ∨ a = 4) ∧
  (b = 0 ∨ b = 1 ∨ b = 3 ∨ b = 4) ∧
  (c = 0 ∨ c = 1 ∨ c = 3 ∨ c = 4) ∧
  (d = 0 ∨ d = 1 ∨ d = 3 ∨ d = 4) ∧
  ¬ (a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0) →
  (c * a^b + d ≤ 196) :=
by sorry

end maximum_possible_value_of_expression_l92_92946


namespace coin_flip_probability_l92_92884

theorem coin_flip_probability :
  let total_flips := 8
  let num_heads := 6
  let total_outcomes := (2: ℝ) ^ total_flips
  let favorable_outcomes := (Nat.choose total_flips num_heads)
  let probability := favorable_outcomes / total_outcomes
  probability = (7 / 64 : ℝ) :=
by
  sorry

end coin_flip_probability_l92_92884


namespace sum_of_reciprocals_of_roots_l92_92543

theorem sum_of_reciprocals_of_roots :
  ∀ (c d : ℝ),
  (6 * c^2 + 5 * c + 7 = 0) → 
  (6 * d^2 + 5 * d + 7 = 0) → 
  (c + d = -5 / 6) → 
  (c * d = 7 / 6) → 
  (1 / c + 1 / d = -5 / 7) :=
by
  intros c d h₁ h₂ h₃ h₄
  sorry

end sum_of_reciprocals_of_roots_l92_92543


namespace max_true_statements_l92_92799

theorem max_true_statements (x : ℝ) :
  let stmt1 := (0 < x^3 ∧ x^3 < 1)
  let stmt2 := (x^3 > 1)
  let stmt3 := (-1 < x ∧ x < 0)
  let stmt4 := (1 < x ∧ x < 2)
  let stmt5 := (0 < 3*x - x^3 ∧ 3*x - x^3 < 2)
  max_true_statements stmt1 stmt2 stmt3 stmt4 stmt5 = 3 := sorry

end max_true_statements_l92_92799


namespace negation_of_proposition_l92_92958

theorem negation_of_proposition (p : ∀ (x : ℝ), x^2 + 1 > 0) :
  ∃ (x : ℝ), x^2 + 1 ≤ 0 ↔ ¬ (∀ (x : ℝ), x^2 + 1 > 0) :=
by
  sorry

end negation_of_proposition_l92_92958


namespace max_truthful_dwarfs_le_one_l92_92417

def dwarf_height_claims : List ℕ := [60, 61, 62, 63, 64, 65, 66]

theorem max_truthful_dwarfs_le_one (heights : List ℕ) (h_lengths : heights.length = 7) (h_sorted : heights.sorted (· > ·)) :
  ∃ n, (n ≤ 1) ∧ ∀ i, i ∈ (Finset.range 7).to_list → (heights[i] ≠ dwarf_height_claims[i] → (1 = n)) :=
  sorry

end max_truthful_dwarfs_le_one_l92_92417


namespace grasshopper_position_after_100_jumps_l92_92601

theorem grasshopper_position_after_100_jumps :
  let start_pos := 1
  let jumps (n : ℕ) := n
  let total_positions := 6
  let total_distance := (100 * (100 + 1)) / 2
  (start_pos + (total_distance % total_positions)) % total_positions = 5 :=
by
  sorry

end grasshopper_position_after_100_jumps_l92_92601


namespace probability_not_all_same_l92_92048

theorem probability_not_all_same (n : ℕ) (h : n = 5) : 
  let total_outcomes := 8^n in
  let same_number_outcomes := 8 in
  let prob_all_same_number := (same_number_outcomes : ℚ) / total_outcomes in
  let prob_not_all_same_number := 1 - prob_all_same_number in
  prob_not_all_same_number = 4095 / 4096 :=
by 
  sorry

end probability_not_all_same_l92_92048


namespace monotonic_range_of_a_l92_92352

noncomputable def f (a x : ℝ) : ℝ := -x^3 + a*x^2 - x - 1
noncomputable def f' (a x : ℝ) : ℝ := -3*x^2 + 2*a*x - 1

theorem monotonic_range_of_a (a : ℝ) : 
  (∀ x : ℝ, f' a x ≤ 0) ↔ (-Real.sqrt 3 ≤ a ∧ a ≤ Real.sqrt 3) :=
by 
  sorry

end monotonic_range_of_a_l92_92352


namespace orange_face_probability_correct_l92_92182

-- Define the number of faces
def total_faces : ℕ := 12
def green_faces : ℕ := 5
def orange_faces : ℕ := 4
def purple_faces : ℕ := 3

-- Define the probability of rolling an orange face
def probability_of_orange_face : ℚ := orange_faces / total_faces

-- Statement of the theorem
theorem orange_face_probability_correct :
  probability_of_orange_face = 1 / 3 :=
by
  sorry

end orange_face_probability_correct_l92_92182


namespace barney_no_clean_towels_days_l92_92236

theorem barney_no_clean_towels_days
  (wash_cycle_weeks : ℕ := 1)
  (total_towels : ℕ := 18)
  (towels_per_day : ℕ := 2)
  (days_per_week : ℕ := 7)
  (missed_laundry_weeks : ℕ := 1) :
  (days_per_week - (total_towels - (days_per_week * towels_per_day * missed_laundry_weeks)) / towels_per_day) = 5 :=
by
  sorry

end barney_no_clean_towels_days_l92_92236


namespace log_sqrt_defined_in_interval_l92_92636

def defined_interval (x : ℝ) : Prop :=
  ∃ y, y = (5 - x) ∧ y > 0 ∧ (x - 2) ≥ 0

theorem log_sqrt_defined_in_interval {x : ℝ} :
  defined_interval x ↔ (2 < x ∧ x < 5) :=
sorry

end log_sqrt_defined_in_interval_l92_92636


namespace paving_cost_l92_92581

-- Definitions based on conditions
def length : ℝ := 5.5
def width : ℝ := 3.75
def rate_per_sqm : ℝ := 600
def expected_cost : ℝ := 12375

-- The problem statement
theorem paving_cost :
  (length * width * rate_per_sqm = expected_cost) :=
sorry

end paving_cost_l92_92581


namespace R_l92_92389

variable (a d n : ℕ)

def arith_sum (k : ℕ) : ℕ :=
  k * (a + (k - 1) * d / 2)

def s1 := arith_sum n
def s2 := arith_sum (3 * n)
def s3 := arith_sum (5 * n)
def s4 := arith_sum (7 * n)

def R' := s4 - s3 - s2

theorem R'_depends_on_d_n : 
  R' = 2 * d * n^2 := 
by 
  sorry

end R_l92_92389


namespace percentage_difference_correct_l92_92437

noncomputable def percentage_difference (initial_price : ℝ) (increase_2012_percent : ℝ) (decrease_2013_percent : ℝ) : ℝ :=
  let price_end_2012 := initial_price * (1 + increase_2012_percent / 100)
  let price_end_2013 := price_end_2012 * (1 - decrease_2013_percent / 100)
  ((price_end_2013 - initial_price) / initial_price) * 100

theorem percentage_difference_correct :
  ∀ (initial_price : ℝ),
  percentage_difference initial_price 25 12 = 10 := 
by
  intros
  sorry

end percentage_difference_correct_l92_92437


namespace linear_function_is_C_l92_92066

theorem linear_function_is_C :
  ∀ (f : ℤ → ℤ), (f = (λ x => 2 * x^2 - 1) ∨ f = (λ x => -1/x) ∨ f = (λ x => (x+1)/3) ∨ f = (λ x => 3 * x + 2 * x^2 - 1)) →
  (f = (λ x => (x+1)/3)) ↔ 
  (∃ (m b : ℤ), ∀ x : ℤ, f x = m * x + b) :=
by
  sorry

end linear_function_is_C_l92_92066


namespace product_units_tens_not_divisible_by_8_l92_92673

theorem product_units_tens_not_divisible_by_8 :
  ¬ (1834 % 8 = 0) → (4 * 3 = 12) :=
by
  intro h
  exact (by norm_num : 4 * 3 = 12)

end product_units_tens_not_divisible_by_8_l92_92673


namespace total_income_by_nth_year_max_m_and_k_range_l92_92477

noncomputable def total_income (a : ℝ) (k : ℝ) (n : ℕ) : ℝ :=
  (6 - (n + 6) * 0.1 ^ n) * a

theorem total_income_by_nth_year (a : ℝ) (n : ℕ) :
  total_income a 0.1 n = (6 - (n + 6) * 0.1 ^ n) * a :=
sorry

theorem max_m_and_k_range (a : ℝ) (m : ℕ) :
  (m = 4 ∧ 1 ≤ 1) ∧ (∀ k, k ≥ 1 → m = 4) :=
sorry

end total_income_by_nth_year_max_m_and_k_range_l92_92477


namespace kiddie_scoop_cost_is_three_l92_92556

-- Define the parameters for the costs of different scoops and total payment
variable (k : ℕ)  -- cost of kiddie scoop
def cost_regular : ℕ := 4
def cost_double : ℕ := 6
def total_payment : ℕ := 32

-- Conditions: Mr. and Mrs. Martin each get a regular scoop
def regular_cost : ℕ := 2 * cost_regular

-- Their three teenage children each get double scoops
def double_cost : ℕ := 3 * cost_double

-- Total cost of regular and double scoops
def combined_cost : ℕ := regular_cost + double_cost

-- Total payment includes two kiddie scoops
def kiddie_total_cost : ℕ := total_payment - combined_cost

-- The cost of one kiddie scoop
def kiddie_cost : ℕ := kiddie_total_cost / 2

theorem kiddie_scoop_cost_is_three : kiddie_cost = 3 := by
  sorry

end kiddie_scoop_cost_is_three_l92_92556


namespace extreme_values_of_f_max_min_values_on_interval_l92_92775

noncomputable def f (x : ℝ) : ℝ := (2 * x) / (Real.exp x)

theorem extreme_values_of_f : 
  (∃ x_max : ℝ, f x_max = 2 / Real.exp 1 ∧ ∀ x : ℝ, f x ≤ 2 / Real.exp 1) :=
sorry

theorem max_min_values_on_interval : 
  (∀ x ∈ Set.Icc (1/2 : ℝ) 2, 
    (f 1 = 2 / Real.exp 1 ∧ ∀ x : ℝ, x ∈ Set.Icc (1/2) 2 → f x ≤ 2 / Real.exp 1)
     ∧ (f 2 = 4 / (Real.exp 2) ∧ ∀ x ∈ Set.Icc (1/2 : ℝ) 2, 4 / (Real.exp 2) ≤ f x)) :=
sorry

end extreme_values_of_f_max_min_values_on_interval_l92_92775


namespace factorization_identity_l92_92339

-- We are asked to prove the mathematical equality under given conditions.
theorem factorization_identity (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = (x^2 + 6*x + 8)^2 :=
by
  sorry

end factorization_identity_l92_92339


namespace sue_travel_time_correct_l92_92974

-- Define the flight and layover times as constants
def NO_to_ATL_flight_hours : ℕ := 2
def ATL_layover_hours : ℕ := 4
def ATL_to_CHI_flight_hours : ℕ := 5
def CHI_time_diff_hours : ℤ := -1
def CHI_layover_hours : ℕ := 3
def CHI_to_NY_flight_hours : ℕ := 3
def NY_time_diff_hours : ℤ := 1
def NY_layover_hours : ℕ := 16
def NY_to_DEN_flight_hours : ℕ := 6
def DEN_time_diff_hours : ℤ := -2
def DEN_layover_hours : ℕ := 5
def DEN_to_SF_flight_hours : ℕ := 4
def SF_time_diff_hours : ℤ := -1

-- Total time calculation including flights, layovers, and time zone changes
def total_travel_time_hours : ℕ :=
  NO_to_ATL_flight_hours +
  ATL_layover_hours +
  (ATL_to_CHI_flight_hours + CHI_time_diff_hours).toNat +  -- Handle time difference (ensure non-negative)
  CHI_layover_hours +
  (CHI_to_NY_flight_hours + NY_time_diff_hours).toNat +
  NY_layover_hours +
  (NY_to_DEN_flight_hours + DEN_time_diff_hours).toNat +
  DEN_layover_hours +
  (DEN_to_SF_flight_hours + SF_time_diff_hours).toNat

-- Statement to prove in Lean:
theorem sue_travel_time_correct : total_travel_time_hours = 45 :=
by {
  -- Skipping proof details since only the statement is required
  sorry
}

end sue_travel_time_correct_l92_92974


namespace grandmother_times_older_l92_92833

variables (M G Gr : ℕ)

-- Conditions
def MilenasAge : Prop := M = 7
def GrandfatherAgeRelation : Prop := Gr = G + 2
def AgeDifferenceRelation : Prop := Gr - M = 58

-- Theorem to prove
theorem grandmother_times_older (h1 : MilenasAge M) (h2 : GrandfatherAgeRelation G Gr) (h3 : AgeDifferenceRelation M Gr) :
  G / M = 9 :=
sorry

end grandmother_times_older_l92_92833


namespace odd_n_cube_plus_one_not_square_l92_92075

theorem odd_n_cube_plus_one_not_square (n : ℤ) (h : n % 2 = 1) : ¬ ∃ (x : ℤ), x^2 = n^3 + 1 :=
by
  sorry

end odd_n_cube_plus_one_not_square_l92_92075


namespace solution_correct_l92_92503

-- Conditions of the problem
variable (f : ℝ → ℝ)
variable (h_f_domain : ∀ (x : ℝ), 0 < x → 0 < f x)
variable (h_f_eq : ∀ (x y : ℝ), 0 < x → 0 < y → f x * f (y * f x) = f (x + y))

-- Correct answer to be proven
theorem solution_correct :
  ∃ b : ℝ, 0 ≤ b ∧ ∀ t : ℝ, 0 < t → f t = 1 / (1 + b * t) :=
sorry

end solution_correct_l92_92503


namespace find_pairs_of_natural_numbers_l92_92916

theorem find_pairs_of_natural_numbers (m n : ℕ) :
  (m + 1) % n = 0 ∧ (n^2 - n + 1) % m = 0 ↔ (m, n) = (1, 1) ∨ (m, n) = (1, 2) ∨ (m, n) = (3, 2) :=
by
  sorry

end find_pairs_of_natural_numbers_l92_92916


namespace factor_polynomial_l92_92321

theorem factor_polynomial :
  ∀ x : ℝ, 
  (x^2 + 4 * x + 3) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8) 
  = (x^2 + 6 * x + 1) * (x^2 + 6 * x + 37) :=
by
  intros x
  sorry

end factor_polynomial_l92_92321


namespace chessboard_not_divisible_by_10_l92_92396

theorem chessboard_not_divisible_by_10 :
  ∀ (B : ℕ × ℕ → ℕ), 
  (∀ x y, B (x, y) < 10) ∧ 
  (∀ x y, x ≥ 0 ∧ x < 8 ∧ y ≥ 0 ∧ y < 8) →
  ¬ ( ∃ k : ℕ, ∀ x y, (B (x, y) + k) % 10 = 0 ) :=
by
  intros
  sorry

end chessboard_not_divisible_by_10_l92_92396


namespace sum_of_probability_fractions_l92_92220

def total_tree_count := 15
def non_birch_count := 9
def birch_count := 6
def total_arrangements := Nat.choose 15 6
def non_adjacent_birch_arrangements := Nat.choose 10 6
def birch_probability := non_adjacent_birch_arrangements / total_arrangements
def simplified_probability_numerator := 6
def simplified_probability_denominator := 143
def answer := simplified_probability_numerator + simplified_probability_denominator

theorem sum_of_probability_fractions :
  answer = 149 := by
  sorry

end sum_of_probability_fractions_l92_92220


namespace max_truthful_gnomes_l92_92412

theorem max_truthful_gnomes :
  ∀ (heights : Fin 7 → ℝ), 
    heights 0 = 60 →
    heights 1 = 61 →
    heights 2 = 62 →
    heights 3 = 63 →
    heights 4 = 64 →
    heights 5 = 65 →
    heights 6 = 66 →
      (∃ i : Fin 7, ∀ j : Fin 7, (i ≠ j → heights j ≠ (60 + j.1)) ∧ (i = j → heights j = (60 + j.1))) :=
by
  intro heights h1 h2 h3 h4 h5 h6 h7
  sorry

end max_truthful_gnomes_l92_92412


namespace factor_polynomial_l92_92300

theorem factor_polynomial :
  (x^2 + 4 * x + 3) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8) = 
  (x^2 + 6 * x + 7) * (x^2 + 6 * x + 8) :=
sorry

end factor_polynomial_l92_92300


namespace interior_edges_sum_l92_92891

-- Definitions based on conditions
def frame_width : ℕ := 2
def frame_area : ℕ := 32
def outer_edge_length : ℕ := 8

-- Mathematically equivalent proof problem
theorem interior_edges_sum :
  ∃ (y : ℕ),  (frame_width * 2) * (y - frame_width * 2) = 32 ∧ (outer_edge_length * y - (outer_edge_length - 2 * frame_width) * (y - 2 * frame_width)) = 32 -> 4 + 4 + 0 + 0 = 8 :=
sorry

end interior_edges_sum_l92_92891


namespace minimum_time_to_serve_tea_equals_9_l92_92672

def boiling_water_time : Nat := 8
def washing_teapot_time : Nat := 1
def washing_teacups_time : Nat := 2
def fetching_tea_leaves_time : Nat := 2
def brewing_tea_time : Nat := 1

theorem minimum_time_to_serve_tea_equals_9 :
  boiling_water_time + brewing_tea_time = 9 := by
  sorry

end minimum_time_to_serve_tea_equals_9_l92_92672


namespace factorization_identity_l92_92336

-- We are asked to prove the mathematical equality under given conditions.
theorem factorization_identity (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = (x^2 + 6*x + 8)^2 :=
by
  sorry

end factorization_identity_l92_92336


namespace largest_product_is_168_l92_92912

open Set

noncomputable def largest_product_from_set (s : Set ℤ) (n : ℕ) (result : ℤ) : Prop :=
  ∃ (a b c : ℤ), a ∈ s ∧ b ∈ s ∧ c ∈ s ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  ∀ (x y z : ℤ), x ∈ s → y ∈ s → z ∈ s → x ≠ y → y ≠ z → x ≠ z →
  x * y * z ≤ a * b * c ∧ a * b * c = result

theorem largest_product_is_168 :
  largest_product_from_set {-4, -3, 1, 3, 7, 8} 3 168 :=
sorry

end largest_product_is_168_l92_92912


namespace interval_of_defined_expression_l92_92632

theorem interval_of_defined_expression (x : ℝ) :
  (x > 2 ∧ x < 5) ↔ (x - 2 > 0 ∧ 5 - x > 0) :=
by
  sorry

end interval_of_defined_expression_l92_92632


namespace rotations_needed_to_reach_goal_l92_92778

-- Define the given conditions
def rotations_per_block : ℕ := 200
def blocks_goal : ℕ := 8
def current_rotations : ℕ := 600

-- Define total_rotations_needed and more_rotations_needed
def total_rotations_needed : ℕ := blocks_goal * rotations_per_block
def more_rotations_needed : ℕ := total_rotations_needed - current_rotations

-- Theorem stating the solution
theorem rotations_needed_to_reach_goal : more_rotations_needed = 1000 := by
  -- proof steps are omitted
  sorry

end rotations_needed_to_reach_goal_l92_92778


namespace eval_expression_l92_92752

theorem eval_expression : (-3)^4 + (-3)^3 + (-3)^2 + 3^2 + 3^3 + 3^4 = 180 := by
  sorry

end eval_expression_l92_92752


namespace Jeffrey_steps_l92_92733

theorem Jeffrey_steps
  (Andrew_steps : ℕ) (Jeffrey_steps : ℕ) (h_ratio : Andrew_steps / Jeffrey_steps = 3 / 4)
  (h_Andrew : Andrew_steps = 150) :
  Jeffrey_steps = 200 :=
by
  sorry

end Jeffrey_steps_l92_92733


namespace garden_area_increase_l92_92473

-- Problem: Prove that changing a 40 ft by 10 ft rectangular garden into a square,
-- using the same fencing, increases the area by 225 sq ft.

theorem garden_area_increase :
  let length_orig := 40
  let width_orig := 10
  let perimeter := 2 * (length_orig + width_orig)
  let side_square := perimeter / 4
  let area_orig := length_orig * width_orig
  let area_square := side_square * side_square
  (area_square - area_orig) = 225 := 
sorry

end garden_area_increase_l92_92473


namespace total_weight_of_arrangement_l92_92595

def original_side_length : ℤ := 4
def original_weight : ℤ := 16
def larger_side_length : ℤ := 10

theorem total_weight_of_arrangement :
  let original_area := original_side_length ^ 2
  let larger_area := larger_side_length ^ 2
  let number_of_pieces := (larger_area / original_area : ℤ)
  let total_weight := (number_of_pieces * original_weight)
  total_weight = 96 :=
by
  let original_area := original_side_length ^ 2
  let larger_area := larger_side_length ^ 2
  let number_of_pieces := (larger_area / original_area : ℤ)
  let total_weight := (number_of_pieces * original_weight)
  sorry

end total_weight_of_arrangement_l92_92595


namespace geom_seq_a5_l92_92986

noncomputable def S3 (a1 q : ℚ) : ℚ := a1 + a1 * q^2
noncomputable def a (a1 q : ℚ) (n : ℕ) : ℚ := a1 * q^(n - 1)

theorem geom_seq_a5 (a1 q : ℚ) (hS3 : S3 a1 q = 5 * a1) (ha7 : a a1 q 7 = 2) :
  a a1 q 5 = 1 / 2 :=
by
  sorry

end geom_seq_a5_l92_92986


namespace minimum_of_quadratic_l92_92435

theorem minimum_of_quadratic : ∀ x : ℝ, 1 ≤ x^2 - 6 * x + 10 :=
by
  intro x
  have h : x^2 - 6 * x + 10 = (x - 3)^2 + 1 := by ring
  rw [h]
  have h_nonneg : (x - 3)^2 ≥ 0 := by apply sq_nonneg
  linarith

end minimum_of_quadratic_l92_92435


namespace ab_value_l92_92928

-- Define sets A and B
def A : Set ℝ := {-1.3}
def B (a b : ℝ) : Set ℝ := {x | x^2 + a * x + b = 0}

-- The proof statement: Given A = B, prove ab = 0.104
theorem ab_value (a b : ℝ) (h : A = B a b) : a * b = 0.104 :=
by
  sorry

end ab_value_l92_92928


namespace sam_distance_traveled_l92_92811

-- Variables definition
variables (distance_marguerite : ℝ) (time_marguerite : ℝ) (time_sam : ℝ)

-- Given conditions
def marguerite_conditions : Prop :=
  distance_marguerite = 150 ∧
  time_marguerite = 3 ∧
  time_sam = 4

-- Statement to prove
theorem sam_distance_traveled (h : marguerite_conditions distance_marguerite time_marguerite time_sam) : 
  distance_marguerite / time_marguerite * time_sam = 200 :=
sorry

end sam_distance_traveled_l92_92811


namespace allocation_schemes_l92_92623

theorem allocation_schemes (volunteers events : ℕ) (h_vol : volunteers = 5) (h_events : events = 4) :
  (∃ allocation_scheme : ℕ, allocation_scheme = 10 * 24) :=
by
  use 240
  sorry

end allocation_schemes_l92_92623


namespace factor_polynomial_l92_92279

-- Define the necessary polynomials
def p (x : ℝ) : ℝ := x^2 + 4*x + 3
def q (x : ℝ) : ℝ := x^2 + 8*x + 15
def r (x : ℝ) : ℝ := x^2 + 6*x - 8

-- State the main theorem
theorem factor_polynomial (x : ℝ) : 
  (p x * q x) + r x = (x^2 + 7*x + 1) * (x^2 + 4*x + 7) :=
by
  sorry

end factor_polynomial_l92_92279


namespace complete_square_solution_l92_92867

theorem complete_square_solution
  (x : ℝ)
  (h : x^2 + 4*x + 2 = 0):
  ∃ c : ℝ, (x + 2)^2 = c ∧ c = 2 :=
by
  sorry

end complete_square_solution_l92_92867


namespace factorization_identity_l92_92333

-- We are asked to prove the mathematical equality under given conditions.
theorem factorization_identity (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = (x^2 + 6*x + 8)^2 :=
by
  sorry

end factorization_identity_l92_92333


namespace strictly_increasing_interval_l92_92499

def f (x : ℝ) : ℝ := -x^3 + 3*x + 2

theorem strictly_increasing_interval : { x : ℝ | -1 < x ∧ x < 1 } = { x : ℝ | -3 * (x + 1) * (x - 1) > 0 } :=
sorry

end strictly_increasing_interval_l92_92499


namespace f_odd_solve_inequality_l92_92508

noncomputable def f : ℝ → ℝ := sorry

axiom f_additive : ∀ x y : ℝ, f (x + y) = f x + f y
axiom f_positive : ∀ x : ℝ, x > 0 → f x > 0

theorem f_odd : ∀ x : ℝ, f (-x) = -f x := 
by
  sorry

theorem solve_inequality : {a : ℝ | f (a-4) + f (2*a+1) < 0} = {a | a < 1} := 
by
  sorry

end f_odd_solve_inequality_l92_92508


namespace cost_of_pencil_pen_eraser_l92_92006

variables {p q r : ℝ}

theorem cost_of_pencil_pen_eraser 
  (h1 : 4 * p + 3 * q + r = 5.40)
  (h2 : 2 * p + 2 * q + 2 * r = 4.60) : 
  p + 2 * q + 3 * r = 4.60 := 
by sorry

end cost_of_pencil_pen_eraser_l92_92006


namespace percentage_increase_equal_price_l92_92480

/-
A merchant has selected two items to be placed on sale, one of which currently sells for 20 percent less than the other.
He wishes to raise the price of the cheaper item so that the two items are equally priced.
By what percentage must he raise the price of the less expensive item?
-/
theorem percentage_increase_equal_price (P: ℝ) : (P > 0) → 
  (∀ cheap_item, cheap_item = 0.80 * P → ((P - cheap_item) / cheap_item) * 100 = 25) :=
by
  intro P_pos
  intro cheap_item
  intro h
  sorry

end percentage_increase_equal_price_l92_92480


namespace roots_of_polynomial_l92_92616

theorem roots_of_polynomial :
  (∃ (r : List ℤ), r = [1, 3, 4] ∧ 
    (∀ x : ℤ, x ∈ r → x^3 - 8*x^2 + 19*x - 12 = 0)) ∧ 
  (∀ x, x^3 - 8*x^2 + 19*x - 12 = 0 → x ∈ [1, 3, 4]) := 
sorry

end roots_of_polynomial_l92_92616


namespace sum_opposite_signs_eq_zero_l92_92464

theorem sum_opposite_signs_eq_zero (x y : ℝ) (h : x * y < 0) : x + y = 0 :=
sorry

end sum_opposite_signs_eq_zero_l92_92464


namespace units_digit_of_product_l92_92460

def is_units_digit (n : ℕ) (d : ℕ) : Prop := n % 10 = d

theorem units_digit_of_product : 
  is_units_digit (6 * 8 * 9 * 10 * 12) 0 := 
by
  sorry

end units_digit_of_product_l92_92460


namespace probability_not_all_same_l92_92053

theorem probability_not_all_same :
  (let total_outcomes := 8 ^ 5 in
   let same_number_outcomes := 8 in
   let probability_all_same := same_number_outcomes / total_outcomes in
   let probability_not_same := 1 - probability_all_same in
   probability_not_same = 4095 / 4096) :=
begin
  sorry
end

end probability_not_all_same_l92_92053


namespace range_of_b_div_a_l92_92351

theorem range_of_b_div_a 
  (a b : ℝ)
  (h1 : 0 < a) 
  (h2 : a ≤ 2)
  (h3 : b ≥ 1)
  (h4 : b ≤ a^2) : 
  (1 / 2) ≤ b / a ∧ b / a ≤ 2 := 
sorry

end range_of_b_div_a_l92_92351


namespace probability_same_length_segments_of_regular_hexagon_l92_92141

theorem probability_same_length_segments_of_regular_hexagon (sides diagonals : ℕ) 
  (total_segments : ℕ) (prob_same_length : ℚ) 
  (h_sides : sides = 6)
  (h_diagonals : diagonals = 9)
  (h_total_segments : total_segments = sides + diagonals)
  : prob_same_length = 17 / 35 :=
by
  have total_combinations := total_segments * (total_segments - 1) / 2
  have prob_sides := (sides * (sides - 1) / 2 : ℚ) / total_combinations
  have prob_diagonals := (diagonals * (diagonals - 1) / 2 : ℚ) / total_combinations
  have prob_sum := prob_sides + prob_diagonals
  have h_prob_sides : prob_sides = (6 * 5 / 2) / (15 * 14 / 2) :=
    by sorry
  have h_prob_diagonals : prob_diagonals = (9 * 8 / 2) / (15 * 14 / 2) :=
    by sorry
  calc
    prob_same_length = prob_sum : by sorry
    ... = 17 / 35 : by {rw [←h_prob_sides, ←h_prob_diagonals], sorry}

end probability_same_length_segments_of_regular_hexagon_l92_92141


namespace floor_e_minus_3_eq_neg1_l92_92510

noncomputable def e : ℝ := 2.718

theorem floor_e_minus_3_eq_neg1 : Int.floor (e - 3) = -1 := by
  sorry

end floor_e_minus_3_eq_neg1_l92_92510


namespace similarity_coordinates_C_l92_92133

theorem similarity_coordinates_C (A B C : ℝ × ℝ) (ratio : ℝ) :
  A = (1,2) ∧ B = (2,1) ∧ C = (3,2) ∧ ratio = 2 →
  (exists C' : ℝ × ℝ, (C' = (6,4)) ∨ (C' = (-6,-4))) :=
by { intro h, sorry }

end similarity_coordinates_C_l92_92133


namespace probability_green_or_purple_l92_92863

theorem probability_green_or_purple
    (green purple orange : ℕ) 
    (h_green : green = 5) 
    (h_purple : purple = 4) 
    (h_orange : orange = 6) :
    (green + purple) / (green + purple + orange) = 3 / 5 :=
by
  sorry

end probability_green_or_purple_l92_92863


namespace distance_from_point_to_circle_center_l92_92381

noncomputable def polar_to_rect (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

noncomputable def circle_center : ℝ × ℝ := (1, 0)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

theorem distance_from_point_to_circle_center :
  distance (polar_to_rect 2 (Real.pi / 3)) circle_center = Real.sqrt 3 := sorry

end distance_from_point_to_circle_center_l92_92381


namespace sindbad_can_identify_eight_genuine_dinars_l92_92423

/--
Sindbad has 11 visually identical dinars in his purse, one of which may be counterfeit and differs in weight from the genuine ones. Using a balance scale twice without weights, it's possible to identify at least 8 genuine dinars.
-/
theorem sindbad_can_identify_eight_genuine_dinars (dinars : Fin 11 → ℝ) (is_genuine : Fin 11 → Prop) :
  (∃! i, ¬ is_genuine i) → 
  (∃ S : Finset (Fin 11), S.card = 8 ∧ S ⊆ (Finset.univ : Finset (Fin 11)) ∧ ∀ i ∈ S, is_genuine i) :=
sorry

end sindbad_can_identify_eight_genuine_dinars_l92_92423


namespace expression_value_l92_92995

theorem expression_value : (25 + 15)^2 - (25^2 + 15^2) = 750 := by
  sorry

end expression_value_l92_92995


namespace sam_distance_l92_92819

theorem sam_distance (m_distance m_time s_time : ℝ) (m_distance_eq : m_distance = 150) (m_time_eq : m_time = 3) (s_time_eq : s_time = 4) :
  let rate := m_distance / m_time,
      s_distance := rate * s_time
  in s_distance = 200 :=
by
  let rate := m_distance / m_time
  let s_distance := rate * s_time
  sorry

end sam_distance_l92_92819


namespace sufficient_no_x_axis_intersections_l92_92463

/-- Sufficient condition for no x-axis intersections -/
theorem sufficient_no_x_axis_intersections
    (a b c : ℝ)
    (h : a ≠ 0)
    (h_sufficient : b^2 - 4 * a * c < -1) :
    ∀ x : ℝ, ¬(a * x^2 + b * x + c = 0) :=
by
  sorry

end sufficient_no_x_axis_intersections_l92_92463


namespace taxi_ride_cost_l92_92898

-- Definitions given in the conditions
def base_fare : ℝ := 2.00
def cost_per_mile : ℝ := 0.30
def distance_traveled : ℝ := 10

-- The theorem we need to prove
theorem taxi_ride_cost : base_fare + (cost_per_mile * distance_traveled) = 5.00 :=
by
  sorry

end taxi_ride_cost_l92_92898


namespace james_muffins_baked_l92_92584

-- Define the number of muffins Arthur baked
def muffinsArthur : ℕ := 115

-- Define the multiplication factor
def multiplicationFactor : ℕ := 12

-- Define the number of muffins James baked
def muffinsJames : ℕ := muffinsArthur * multiplicationFactor

-- The theorem that needs to be proved
theorem james_muffins_baked : muffinsJames = 1380 :=
by
  sorry

end james_muffins_baked_l92_92584


namespace problem_statement_l92_92432

noncomputable def f (x : ℝ) : ℝ := if x ≥ 1 then Real.log x / Real.log 2 else sorry

theorem problem_statement : f (1 / 2) < f (1 / 3) ∧ f (1 / 3) < f 2 :=
by
  -- Definitions based on given conditions
  have h1 : ∀ x : ℝ, f (2 - x) = f x := sorry
  have h2 : ∀ x : ℝ, 1 ≤ x → f x = Real.log x / Real.log 2 := sorry
  -- Proof of the statement based on h1 and h2
  sorry

end problem_statement_l92_92432


namespace count_birches_in_forest_l92_92937

theorem count_birches_in_forest:
  ∀ (t p_s p_p : ℕ), t = 4000 → p_s = 10 → p_p = 13 →
  let n_s := (p_s * t) / 100 in
  let n_p := (p_p * t) / 100 in
  let n_o := n_s + n_p in 
  let n_b := t - (n_s + n_p + n_o) in 
  n_b = 2160 :=
by 
  intros t p_s p_p ht hps hpp
  let n_s := (p_s * t) / 100 
  let n_p := (p_p * t) / 100 
  let n_o := n_s + n_p 
  let n_b := t - (n_s + n_p + n_o) 
  exact sorry

end count_birches_in_forest_l92_92937


namespace distinct_pairs_count_l92_92247

theorem distinct_pairs_count :
  ∃ (n : ℕ), n = 2 ∧ 
  ∀ (x y : ℝ), (x = 3 * x^2 + y^2) ∧ (y = 3 * x * y) → 
    ((x = 0 ∧ y = 0) ∨ (x = 1 / 3 ∧ y = 0)) :=
by
  sorry

end distinct_pairs_count_l92_92247


namespace sam_drove_200_miles_l92_92825

-- Define the conditions
def marguerite_distance : ℕ := 150
def marguerite_time : ℕ := 3
def sam_time : ℕ := 4
def marguerite_speed := marguerite_distance / marguerite_time

-- Define the question
def sam_distance (speed : ℕ) (time : ℕ) : ℕ := speed * time

-- State the theorem to prove the answer
theorem sam_drove_200_miles :
  sam_distance marguerite_speed sam_time = 200 := by
  sorry

end sam_drove_200_miles_l92_92825


namespace tax_percentage_l92_92237

theorem tax_percentage (C T : ℝ) (h1 : C + 10 = 90) (h2 : 1 = 90 - C - T * 90) : T = 0.1 := 
by 
  -- We provide the conditions using sorry to indicate the steps would go here
  sorry

end tax_percentage_l92_92237


namespace number_is_4_l92_92588

theorem number_is_4 (x : ℕ) (h : x + 5 = 9) : x = 4 := 
by {
  sorry
}

end number_is_4_l92_92588


namespace min_A_max_B_l92_92791

-- Part (a): prove A = 15 is the smallest value satisfying the condition
theorem min_A (A B : ℕ) (h : 10 ≤ A ∧ A ≤ 99 ∧ 10 ≤ B ∧ B ≤ 99)
  (eq1 : (A - 5) / A + 4 / B = 1) : A = 15 := 
sorry

-- Part (b): prove B = 76 is the largest value satisfying the condition
theorem max_B (A B : ℕ) (h : 10 ≤ A ∧ A ≤ 99 ∧ 10 ≤ B ∧ B ≤ 99)
  (eq1 : (A - 5) / A + 4 / B = 1) : B = 76 := 
sorry

end min_A_max_B_l92_92791


namespace trading_organization_increase_price_l92_92232

theorem trading_organization_increase_price 
  (initial_moisture_content : ℝ)
  (final_moisture_content : ℝ)
  (solid_mass : ℝ)
  (initial_total_mass final_total_mass : ℝ) :
  initial_moisture_content = 0.99 → 
  final_moisture_content = 0.98 →
  initial_total_mass = 100 →
  solid_mass = initial_total_mass * (1 - initial_moisture_content) →
  final_total_mass = solid_mass / (1 - final_moisture_content) →
  (final_total_mass / initial_total_mass) = 0.5 →
  100 * (1 - (final_total_mass / initial_total_mass)) = 100 :=
by sorry

end trading_organization_increase_price_l92_92232


namespace total_germs_l92_92135

-- Define variables and constants
namespace BiologyLab

def petri_dishes : ℕ := 75
def germs_per_dish : ℕ := 48

-- The goal is to prove that the total number of germs is as expected.
theorem total_germs : (petri_dishes * germs_per_dish) = 3600 :=
by
  -- Proof is omitted for this example
  sorry

end BiologyLab

end total_germs_l92_92135


namespace factor_polynomial_l92_92280

-- Define the necessary polynomials
def p (x : ℝ) : ℝ := x^2 + 4*x + 3
def q (x : ℝ) : ℝ := x^2 + 8*x + 15
def r (x : ℝ) : ℝ := x^2 + 6*x - 8

-- State the main theorem
theorem factor_polynomial (x : ℝ) : 
  (p x * q x) + r x = (x^2 + 7*x + 1) * (x^2 + 4*x + 7) :=
by
  sorry

end factor_polynomial_l92_92280


namespace factorize_expression_l92_92490

theorem factorize_expression (x : ℝ) : 
  x^8 - 256 = (x^4 + 16) * (x^2 + 4) * (x + 2) * (x - 2) := 
by
  sorry

end factorize_expression_l92_92490


namespace gcd_lcm_product_l92_92099

theorem gcd_lcm_product (a b : ℕ) (h1 : a = 30) (h2 : b = 45) :
  Nat.gcd a b * Nat.lcm a b = 1350 :=
by
  rw [h1, h2]
  sorry

end gcd_lcm_product_l92_92099


namespace total_cakes_correct_l92_92077

-- Define the initial number of full-size cakes
def initial_cakes : ℕ := 350

-- Define the number of additional full-size cakes made
def additional_cakes : ℕ := 125

-- Define the number of half-cakes made
def half_cakes : ℕ := 75

-- Convert half-cakes to full-size cakes, considering only whole cakes
def half_to_full_cakes := (half_cakes / 2)

-- Total full-size cakes calculation
def total_cakes :=
  initial_cakes + additional_cakes + half_to_full_cakes

-- Prove the total number of full-size cakes
theorem total_cakes_correct : total_cakes = 512 :=
by
  -- Skip the proof
  sorry

end total_cakes_correct_l92_92077


namespace transformation_correct_l92_92367

variables {x y : ℝ}

theorem transformation_correct (h : x = y) : x - 2 = y - 2 := by
  sorry

end transformation_correct_l92_92367


namespace toms_investment_l92_92698

theorem toms_investment 
  (P : ℝ)
  (rA : ℝ := 0.06)
  (nA : ℝ := 1)
  (tA : ℕ := 4)
  (rB : ℝ := 0.08)
  (nB : ℕ := 2)
  (tB : ℕ := 4)
  (delta : ℝ := 100)
  (A_A := P * (1 + rA / nA) ^ (nA * tA))
  (A_B := P * (1 + rB / nB) ^ (nB * tB))
  (h : A_B - A_A = delta) : 
  P = 942.59 := by
sorry

end toms_investment_l92_92698


namespace general_term_l92_92923

noncomputable def S (n : ℕ) : ℕ := sorry
noncomputable def a (n : ℕ) : ℕ := sorry

axiom S2 : S 2 = 4
axiom a_recurrence : ∀ n : ℕ, n > 0 → a (n + 1) = 2 * S n + 1

theorem general_term (n : ℕ) : a n = 3 ^ (n - 1) :=
by
  sorry

end general_term_l92_92923


namespace trader_profit_l92_92083

noncomputable def profit_percentage (P : ℝ) : ℝ :=
  let purchased_price := 0.72 * P
  let market_increase := 1.05 * purchased_price
  let expenses := 0.08 * market_increase
  let net_price := market_increase - expenses
  let first_sale_price := 1.50 * net_price
  let final_sale_price := 1.25 * first_sale_price
  let profit := final_sale_price - P
  (profit / P) * 100

theorem trader_profit
  (P : ℝ) 
  (hP : 0 < P) :
  profit_percentage P = 30.41 :=
by
  sorry

end trader_profit_l92_92083


namespace least_possible_number_of_coins_in_jar_l92_92998

theorem least_possible_number_of_coins_in_jar (n : ℕ) : 
  (n % 7 = 3) → (n % 4 = 1) → (n % 6 = 5) → n = 17 :=
by
  sorry

end least_possible_number_of_coins_in_jar_l92_92998


namespace g_triple_application_l92_92126

def g (x : ℕ) : ℕ := 7 * x + 3

theorem g_triple_application : g (g (g 3)) = 1200 :=
by
  sorry

end g_triple_application_l92_92126


namespace factor_polynomial_l92_92323

theorem factor_polynomial :
  ∀ x : ℝ, 
  (x^2 + 4 * x + 3) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8) 
  = (x^2 + 6 * x + 1) * (x^2 + 6 * x + 37) :=
by
  intros x
  sorry

end factor_polynomial_l92_92323


namespace smallest_PR_minus_QR_l92_92568

theorem smallest_PR_minus_QR :
  ∃ (PQ QR PR : ℤ), 
    PQ + QR + PR = 2023 ∧ PQ ≤ QR ∧ QR < PR ∧ PR - QR = 13 :=
by
  sorry

end smallest_PR_minus_QR_l92_92568


namespace sunil_interest_l92_92554

-- Condition definitions
def A : ℝ := 3370.80
def r : ℝ := 0.06
def n : ℕ := 1
def t : ℕ := 2

-- Derived definition for principal P
noncomputable def P : ℝ := A / (1 + r/n)^(n * t)

-- Interest I calculation
noncomputable def I : ℝ := A - P

-- Proof statement
theorem sunil_interest : I = 370.80 :=
by
  -- Insert the mathematical proof steps here.
  sorry

end sunil_interest_l92_92554


namespace strawberries_weight_l92_92451

theorem strawberries_weight (total_weight apples_weight oranges_weight grapes_weight strawberries_weight : ℕ) 
  (h_total : total_weight = 10)
  (h_apples : apples_weight = 3)
  (h_oranges : oranges_weight = 1)
  (h_grapes : grapes_weight = 3) 
  (h_sum : total_weight = apples_weight + oranges_weight + grapes_weight + strawberries_weight) :
  strawberries_weight = 3 :=
by
  sorry

end strawberries_weight_l92_92451


namespace find_smaller_number_l92_92393

theorem find_smaller_number
  (x y : ℝ) (m : ℝ)
  (h1 : x - y = 9) 
  (h2 : x + y = 46)
  (h3 : x = m * y) : 
  min x y = 18.5 :=
by 
  sorry

end find_smaller_number_l92_92393


namespace polynomial_factorization_l92_92309

theorem polynomial_factorization (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = (x^2 + 6*x + 1) * (x^2 + 6*x + 37) :=
  sorry

end polynomial_factorization_l92_92309


namespace factorize_l92_92502

theorem factorize (x : ℝ) : 72 * x ^ 11 + 162 * x ^ 22 = 18 * x ^ 11 * (4 + 9 * x ^ 11) :=
by
  sorry

end factorize_l92_92502


namespace ab_zero_l92_92109

theorem ab_zero (a b : ℝ)
  (h1 : a + b = 5)
  (h2 : a^3 + b^3 = 125) : 
  a * b = 0 :=
by
  sorry

end ab_zero_l92_92109


namespace nikka_us_stamp_percentage_l92_92546

/-- 
Prove that 20% of Nikka's stamp collection are US stamps given the following conditions:
1. Nikka has a total of 100 stamps.
2. 35 of those stamps are Chinese.
3. 45 of those stamps are Japanese.
-/
theorem nikka_us_stamp_percentage
  (total_stamps : ℕ)
  (chinese_stamps : ℕ)
  (japanese_stamps : ℕ)
  (h1 : total_stamps = 100)
  (h2 : chinese_stamps = 35)
  (h3 : japanese_stamps = 45) :
  ((total_stamps - (chinese_stamps + japanese_stamps)) / total_stamps) * 100 = 20 := 
by
  sorry

end nikka_us_stamp_percentage_l92_92546


namespace chessboard_movement_l92_92180

-- Defining the problem as described in the transformed proof problem

theorem chessboard_movement (pieces : Nat) (adjacent_empty_square : Nat → Nat → Bool) (visited_all_squares : Nat → Bool)
  (returns_to_starting_square : Nat → Bool) :
  (∃ (moment : Nat), ∀ (piece : Nat), ¬ returns_to_starting_square piece) :=
by
  -- Here we state that there exists a moment when each piece (checker) is not on its starting square
  sorry

end chessboard_movement_l92_92180


namespace cos_arcsin_l92_92739

theorem cos_arcsin (h3: ℝ) (h5: ℝ) (h_op: h3 = 3) (h_hyp: h5 = 5) : 
  Real.cos (Real.arcsin (3 / 5)) = 4 / 5 := 
sorry

end cos_arcsin_l92_92739


namespace dwarfs_truth_claims_l92_92415

/-- Seven dwarfs lined up by height, starting with the tallest.
Each dwarf claims a specific height as follows.
Verify that the largest number of dwarfs telling the truth is exactly 1. -/
theorem dwarfs_truth_claims :
  let heights := [60, 61, 62, 63, 64, 65, 66] in
  ∀ (actual_height: Fin 7 → ℕ), 
    (∀ i, actual_height i > actual_height (Fin.mk i (by linarith))) →
    ∀ claim_truth: Fin 7 → Prop, 
    (∀ i, claim_truth i ↔ actual_height i = heights[i]) →
    (∃ k, ∀ j, claim_truth j → j = k) :=
by
  intro heights actual_height h_order claim_truth h_claim_match
  existsi Fin.mk 0 (by linarith) -- Claim that only one can be telling the truth (arbitrary choice)
  sorry  -- Proof steps are omitted

end dwarfs_truth_claims_l92_92415


namespace probability_of_X_eq_4_l92_92475

noncomputable def probability_X_eq_4 : ℝ :=
  let total_balls := 12
  let new_balls := 9
  let old_balls := 3
  let draw := 3
  -- Number of ways to choose 2 old balls from 3
  let choose_old := Nat.choose old_balls 2
  -- Number of ways to choose 1 new ball from 9
  let choose_new := Nat.choose new_balls 1
  -- Total number of ways to choose 3 balls from 12
  let total_ways := Nat.choose total_balls draw
  -- Probability calculation
  (choose_old * choose_new) / total_ways

theorem probability_of_X_eq_4 : probability_X_eq_4 = 27 / 220 := by
  sorry

end probability_of_X_eq_4_l92_92475


namespace evaluate_fraction_l92_92501

theorem evaluate_fraction :
  1 + 1 / (2 + 1 / (3 + 1 / (3 + 3))) = 63 / 44 := 
by
  -- Skipping the proof part with 'sorry'
  sorry

end evaluate_fraction_l92_92501


namespace smallest_positive_integer_l92_92459

theorem smallest_positive_integer (n : ℕ) (hn : 0 < n) (h : 19 * n ≡ 1456 [MOD 11]) : n = 6 :=
by
  sorry

end smallest_positive_integer_l92_92459


namespace set_A_membership_l92_92211

theorem set_A_membership (U : Finset ℕ) (A : Finset ℕ) (B : Finset ℕ)
  (hU : U.card = 193)
  (hB : B.card = 49)
  (hneither : (U \ (A ∪ B)).card = 59)
  (hAandB : (A ∩ B).card = 25) :
  A.card = 110 := sorry

end set_A_membership_l92_92211


namespace factor_poly_eq_factored_form_l92_92273

-- Defining the polynomial expressions
def poly1 (x : ℝ) := x^2 + 4 * x + 3
def poly2 (x : ℝ) := x^2 + 8 * x + 15
def poly3 (x : ℝ) := x^2 + 6 * x - 8

-- The main expression which needs to be factored
def main_expr (x : ℝ) := (poly1 x) * (poly2 x) + (poly3 x)

-- Stating the goal factored form
def factored_form (x : ℝ) := (x^2 + 6 * x + 19) * (x^2 + 6 * x - 2)

-- The theorem statement
theorem factor_poly_eq_factored_form (x : ℝ) : 
  main_expr x = factored_form x := 
by
  sorry

end factor_poly_eq_factored_form_l92_92273


namespace problem1_problem2_l92_92130

-- Define the total number of balls for clarity
def total_red_balls : ℕ := 4
def total_white_balls : ℕ := 6
def total_balls_drawn : ℕ := 4

-- Define binomial coefficient
noncomputable def binom (n k : ℕ) : ℕ := n.choose k

-- Problem 1: Prove that the number of ways to draw 4 balls that include both colors is 194
theorem problem1 :
  (binom total_red_balls 3 * binom total_white_balls 1) +
  (binom total_red_balls 2 * binom total_white_balls 2) +
  (binom total_red_balls 1 * binom total_white_balls 3) = 194 :=
  sorry

-- Problem 2: Prove that the number of ways to draw 4 balls where the number of red balls is at least the number of white balls is 115
theorem problem2 :
  (binom total_red_balls 4 * binom total_white_balls 0) +
  (binom total_red_balls 3 * binom total_white_balls 1) +
  (binom total_red_balls 2 * binom total_white_balls 2) = 115 :=
  sorry

end problem1_problem2_l92_92130


namespace exterior_angle_BAC_l92_92482

theorem exterior_angle_BAC (angle_octagon angle_rectangle : ℝ) (h_oct_135 : angle_octagon = 135) (h_rec_90 : angle_rectangle = 90) :
  360 - (angle_octagon + angle_rectangle) = 135 := 
by
  simp [h_oct_135, h_rec_90]
  sorry

end exterior_angle_BAC_l92_92482


namespace suit_cost_l92_92840

theorem suit_cost :
  let shirt_cost := 15
  let pants_cost := 40
  let sweater_cost := 30
  let shirts := 4
  let pants := 2
  let sweaters := 2 
  let total_cost := shirts * shirt_cost + pants * pants_cost + sweaters * sweater_cost
  let discount_store := 0.80
  let discount_coupon := 0.90
  ∃ S, discount_coupon * discount_store * (total_cost + S) = 252 → S = 150 :=
by
  let shirt_cost := 15
  let pants_cost := 40
  let sweater_cost := 30
  let shirts := 4
  let pants := 2
  let sweaters := 2 
  let total_cost := shirts * shirt_cost + pants * pants_cost + sweaters * sweater_cost
  let discount_store := 0.80
  let discount_coupon := 0.90
  exists 150
  intro h
  sorry

end suit_cost_l92_92840


namespace two_lines_parallel_same_plane_l92_92852

-- Defining the types for lines and planes
variable (Line : Type) (Plane : Type)

-- Defining the relationships similar to the mathematical conditions
variable (parallel_to_plane : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (intersect : Line → Line → Prop)
variable (skew : Line → Line → Prop)

-- Defining the non-overlapping relationships between lines (assuming these relations are mutually exclusive)
axiom parallel_or_intersect_or_skew : ∀ (a b: Line), 
  (parallel a b ∨ intersect a b ∨ skew a b)

-- The statement we want to prove
theorem two_lines_parallel_same_plane (a b: Line) (α: Plane) :
  parallel_to_plane a α → parallel_to_plane b α → (parallel a b ∨ intersect a b ∨ skew a b) :=
by
  intro ha hb
  apply parallel_or_intersect_or_skew

end two_lines_parallel_same_plane_l92_92852


namespace find_ab_l92_92117

theorem find_ab (a b : ℝ) (h1 : a + b = 5) (h2 : a^3 + b^3 = 125) : a * b = 0 :=
by
  sorry

end find_ab_l92_92117


namespace solve_for_x_l92_92468

theorem solve_for_x (x y : ℤ) (h1 : x + y = 10) (h2 : x - y = 18) : x = 14 := by
  sorry

end solve_for_x_l92_92468


namespace largest_x_satisfying_equation_l92_92033

theorem largest_x_satisfying_equation :
  ∃ x : ℝ, x = 3 / 25 ∧ (∀ y, (y : ℝ) ∈ {z | sqrt (3 * z) = 5 * z} → y ≤ x) :=
by
  sorry

end largest_x_satisfying_equation_l92_92033


namespace max_one_truthful_dwarf_l92_92405

def dwarf_height_claims : List (ℕ × ℕ) := [
  (1, 60),  -- First dwarf claims 60 cm
  (2, 61),  -- Second dwarf claims 61 cm
  (3, 62),  -- Third dwarf claims 62 cm
  (4, 63),  -- Fourth dwarf claims 63 cm
  (5, 64),  -- Fifth dwarf claims 64 cm
  (6, 65),  -- Sixth dwarf claims 65 cm
  (7, 66)   -- Seventh dwarf claims 66 cm
]

-- Proof problem statement: Prove that at most one dwarf is telling the truth about their height
theorem max_one_truthful_dwarf : ∀ (heights : List ℕ), heights.length = 7 → 
  (∀ i, i < heights.length → dwarf_height_claims.get? i ≠ none → (heights.get? i = dwarf_height_claims.get? i) → 
    ∑ i in finset.range heights.length, bool.to_num (heights.get? i = dwarf_height_claims.get? i) ≤ 1) :=
by
  intros heights len_heights valid_claims idx idx_in_range some_claim match_claim
  -- Add proof here 
  sorry

end max_one_truthful_dwarf_l92_92405


namespace sam_distance_l92_92821

theorem sam_distance (m_distance m_time s_time : ℝ) (m_distance_eq : m_distance = 150) (m_time_eq : m_time = 3) (s_time_eq : s_time = 4) :
  let rate := m_distance / m_time,
      s_distance := rate * s_time
  in s_distance = 200 :=
by
  let rate := m_distance / m_time
  let s_distance := rate * s_time
  sorry

end sam_distance_l92_92821


namespace positive_integer_solutions_l92_92245

theorem positive_integer_solutions (a b : ℕ) (h_pos_ab : 0 < a ∧ 0 < b) :
  (∃ k : ℕ, k = a^2 / (2 * a * b^2 - b^3 + 1) ∧ 0 < k) ↔
  ∃ n : ℕ, (a = 2 * n ∧ b = 1) ∨ (a = n ∧ b = 2 * n) ∨ (a = 8 * n^4 - n ∧ b = 2 * n) :=
by
  sorry

end positive_integer_solutions_l92_92245


namespace gardenia_to_lilac_ratio_l92_92507

-- Defining sales of flowers
def lilacs_sold : Nat := 10
def roses_sold : Nat := 3 * lilacs_sold
def total_flowers_sold : Nat := 45
def gardenias_sold : Nat := total_flowers_sold - (roses_sold + lilacs_sold)

-- The ratio of gardenias to lilacs as a fraction
def ratio_gardenias_to_lilacs (gardenias lilacs : Nat) : Rat := gardenias / lilacs

-- Stating the theorem to prove
theorem gardenia_to_lilac_ratio :
  ratio_gardenias_to_lilacs gardenias_sold lilacs_sold = 1 / 2 :=
by
  sorry

end gardenia_to_lilac_ratio_l92_92507


namespace necessary_condition_for_q_implies_m_bounds_necessary_but_not_sufficient_condition_for_not_q_l92_92777

-- Problem 1
theorem necessary_condition_for_q_implies_m_bounds (m : ℝ) :
  (∀ x : ℝ, x^2 - 8 * x - 20 ≤ 0 → 1 - m^2 ≤ x ∧ x ≤ 1 + m^2) → (- Real.sqrt 3 ≤ m ∧ m ≤ Real.sqrt 3) :=
sorry

-- Problem 2
theorem necessary_but_not_sufficient_condition_for_not_q (m : ℝ) :
  (∀ x : ℝ, ¬ (x^2 - 8 * x - 20 ≤ 0) → ¬ (1 - m^2 ≤ x ∧ x ≤ 1 + m^2)) → (m ≥ 3 ∨ m ≤ -3) :=
sorry

end necessary_condition_for_q_implies_m_bounds_necessary_but_not_sufficient_condition_for_not_q_l92_92777


namespace ice_cream_ratio_l92_92839

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

end ice_cream_ratio_l92_92839


namespace dogs_sold_correct_l92_92087

-- Definitions based on conditions
def ratio_cats_to_dogs (cats dogs : ℕ) := 2 * dogs = cats

-- Given conditions
def cats_sold := 16
def dogs_sold := 8

-- The theorem to prove
theorem dogs_sold_correct (h : ratio_cats_to_dogs cats_sold dogs_sold) : dogs_sold = 8 :=
by
  sorry

end dogs_sold_correct_l92_92087


namespace pqrs_l92_92366

theorem pqrs(p q r s t u : ℤ) :
  (729 * (x : ℤ) * x * x + 64 = (p * x * x + q * x + r) * (s * x * x + t * x + u)) →
  p = 9 → q = 4 → r = 0 → s = 81 → t = -36 → u = 16 →
  p^2 + q^2 + r^2 + s^2 + t^2 + u^2 = 8210 := by
  intros h1 hp hq hr hs ht hu
  sorry

end pqrs_l92_92366


namespace probability_not_all_dice_show_different_l92_92061

noncomputable def probability_not_all_same : ℚ :=
  let total_outcomes := 8^5
  let same_number_outcomes := 8
  (total_outcomes - same_number_outcomes) / total_outcomes

theorem probability_not_all_dice_show_different : 
  probability_not_all_same = 4095 / 4096 := by
  sorry

end probability_not_all_dice_show_different_l92_92061


namespace coin_flip_probability_l92_92885

theorem coin_flip_probability :
  let total_flips := 8
  let num_heads := 6
  let total_outcomes := (2: ℝ) ^ total_flips
  let favorable_outcomes := (Nat.choose total_flips num_heads)
  let probability := favorable_outcomes / total_outcomes
  probability = (7 / 64 : ℝ) :=
by
  sorry

end coin_flip_probability_l92_92885


namespace direction_vector_correct_l92_92430

open Real

def line_eq (x y : ℝ) : Prop := x - 3 * y + 1 = 0

noncomputable def direction_vector : ℝ × ℝ := (3, 1)

theorem direction_vector_correct (x y : ℝ) (h : line_eq x y) : 
    ∃ k : ℝ, direction_vector = (k * (1 : ℝ), k * (1 / 3)) :=
by
  use 3
  sorry

end direction_vector_correct_l92_92430


namespace multiply_fractions_l92_92573

theorem multiply_fractions :
  (1 / 3 : ℚ) * (3 / 5) * (5 / 6) = 1 / 6 :=
by
  sorry

end multiply_fractions_l92_92573


namespace factorization_correct_l92_92293

noncomputable def polynomial_expr := 
  λ x : ℝ => (x^2 + 4 * x + 3) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8)

noncomputable def factored_expr := 
  λ x : ℝ => (x^2 + 6 * x + 9) * (x^2 + 6 * x + 1)

theorem factorization_correct : 
  ∀ x : ℝ, polynomial_expr x = factored_expr x :=
by
  intro x
  sorry

end factorization_correct_l92_92293


namespace bakery_regular_price_l92_92724

theorem bakery_regular_price (y : ℝ) (h₁ : y / 4 * 0.4 = 2) : y = 20 :=
by {
  sorry
}

end bakery_regular_price_l92_92724


namespace combined_number_of_fasteners_l92_92544

def lorenzo_full_cans_total_fasteners
  (thumbtacks_cans : ℕ)
  (pushpins_cans : ℕ)
  (staples_cans : ℕ)
  (thumbtacks_per_board : ℕ)
  (pushpins_per_board : ℕ)
  (staples_per_board : ℕ)
  (boards_tested : ℕ)
  (thumbtacks_remaining : ℕ)
  (pushpins_remaining : ℕ)
  (staples_remaining : ℕ) :
  ℕ :=
  let thumbtacks_used := thumbtacks_per_board * boards_tested
  let pushpins_used := pushpins_per_board * boards_tested
  let staples_used := staples_per_board * boards_tested
  let thumbtacks_per_can := thumbtacks_used + thumbtacks_remaining
  let pushpins_per_can := pushpins_used + pushpins_remaining
  let staples_per_can := staples_used + staples_remaining
  let total_thumbtacks := thumbtacks_per_can * thumbtacks_cans
  let total_pushpins := pushpins_per_can * pushpins_cans
  let total_staples := staples_per_can * staples_cans
  total_thumbtacks + total_pushpins + total_staples

theorem combined_number_of_fasteners :
  lorenzo_full_cans_total_fasteners 5 3 2 3 2 4 150 45 35 25 = 4730 :=
  by
  sorry

end combined_number_of_fasteners_l92_92544


namespace six_times_six_l92_92871

-- Definitions based on the conditions
def pattern (n : ℕ) : ℕ := n * 6

-- Theorem statement to be proved
theorem six_times_six : pattern 6 = 36 :=
by {
  sorry
}

end six_times_six_l92_92871


namespace regular_hexagon_same_length_probability_l92_92168

theorem regular_hexagon_same_length_probability :
  let T : Finset (Finite_Field) := sorry, -- Define the actual set T una don the constructible geometry module in lean mathematical library
  let total_segments := 15,
  let sides := 6,
  let shorter_diagonals := 6,
  let longer_diagonals := 3,
  let prob_side := (6 / 15) * (5 / 14),
  let prob_shorter := (6 / 15) * (5 / 14),
  let prob_longer := (3 / 15) * (2 / 14),
  let total_prob := (prob_side + prob_shorter + prob_longer),
  (total_prob = (11 / 35)) := sorry

end regular_hexagon_same_length_probability_l92_92168


namespace f_m_plus_1_positive_l92_92515

def f (x : ℝ) (a : ℝ) : ℝ := x^2 + x + a

theorem f_m_plus_1_positive {m a : ℝ} (h_a_pos : a > 0) (h_f_m_neg : f m a < 0) : f (m + 1) a > 0 := by
  sorry

end f_m_plus_1_positive_l92_92515


namespace oil_layer_height_l92_92566

/-- Given a tank with a rectangular bottom measuring 16 cm in length and 12 cm in width, initially containing 6 cm deep water and 6 cm deep oil, and an iron block with dimensions 8 cm in length, 8 cm in width, and 12 cm in height -/

theorem oil_layer_height (volume_water volume_oil volume_iron base_area new_volume_water : ℝ) 
  (base_area_def : base_area = 16 * 12) 
  (volume_water_def : volume_water = base_area * 6) 
  (volume_oil_def : volume_oil = base_area * 6) 
  (volume_iron_def : volume_iron = 8 * 8 * 12) 
  (new_volume_water_def : new_volume_water = volume_water + volume_iron) 
  (new_water_height : new_volume_water / base_area = 10) 
  : (volume_water + volume_oil) / base_area - (new_volume_water / base_area - 6) = 7 :=
by 
  sorry

end oil_layer_height_l92_92566


namespace simple_interest_rate_l92_92504

open Rat

noncomputable def rate_of_interest_per_annum (P SI : ℚ) (T : ℚ) : ℚ :=
  (SI * 100) / (P * T)

theorem simple_interest_rate :
  let P : ℚ := 69600
  let SI : ℚ := 8625
  let T : ℚ := 3 / 4
  (rate_of_interest_per_annum P SI T) ≈ 22.04 := by
  sorry

end simple_interest_rate_l92_92504


namespace maxRegions_formula_l92_92481

-- Define the maximum number of regions in the plane given by n lines
def maxRegions (n: ℕ) : ℕ := (n^2 + n + 2) / 2

-- Main theorem to prove
theorem maxRegions_formula (n : ℕ) : maxRegions n = (n^2 + n + 2) / 2 := by 
  sorry

end maxRegions_formula_l92_92481


namespace find_y_value_l92_92368

theorem find_y_value (k : ℝ) (x y : ℝ) (h1 : y = k * x^(1/5)) (h2 : y = 4) (h3 : x = 32) :
  y = 6 := by
  sorry

end find_y_value_l92_92368


namespace binomial_8_4_eq_70_l92_92493

theorem binomial_8_4_eq_70 : Nat.binom 8 4 = 70 := by
  sorry

end binomial_8_4_eq_70_l92_92493


namespace solve_system1_solve_system2_l92_92843

theorem solve_system1 (x y : ℚ) (h1 : y = x - 5) (h2 : 3 * x - y = 8) :
  x = 3 / 2 ∧ y = -7 / 2 := 
sorry

theorem solve_system2 (x y : ℚ) (h1 : 3 * x - 2 * y = 1) (h2 : 7 * x + 4 * y = 11) :
  x = 1 ∧ y = 1 := 
sorry

end solve_system1_solve_system2_l92_92843


namespace log_over_sqrt_defined_l92_92630

theorem log_over_sqrt_defined (x : ℝ) : (2 < x ∧ x < 5) ↔ ∃ f : ℝ, f = (log (5 - x) / sqrt (x - 2)) :=
by
  sorry

end log_over_sqrt_defined_l92_92630


namespace chromosomes_mitosis_late_stage_l92_92881

/-- A biological cell with 24 chromosomes at the late stage of the second meiotic division. -/
def cell_chromosomes_meiosis_late_stage : ℕ := 24

/-- The number of chromosomes in this organism at the late stage of mitosis is double that at the late stage of the second meiotic division. -/
theorem chromosomes_mitosis_late_stage : cell_chromosomes_meiosis_late_stage * 2 = 48 :=
by
  -- We will add the necessary proof here.
  sorry

end chromosomes_mitosis_late_stage_l92_92881


namespace factor_polynomial_l92_92298

theorem factor_polynomial :
  (x^2 + 4 * x + 3) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8) = 
  (x^2 + 6 * x + 7) * (x^2 + 6 * x + 8) :=
sorry

end factor_polynomial_l92_92298


namespace taxi_ride_cost_l92_92896

theorem taxi_ride_cost (base_fare : ℚ) (cost_per_mile : ℚ) (distance : ℕ) :
  base_fare = 2 ∧ cost_per_mile = 0.30 ∧ distance = 10 →
  base_fare + cost_per_mile * distance = 5 :=
by
  sorry

end taxi_ride_cost_l92_92896


namespace factorization_identity_l92_92335

-- We are asked to prove the mathematical equality under given conditions.
theorem factorization_identity (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = (x^2 + 6*x + 8)^2 :=
by
  sorry

end factorization_identity_l92_92335


namespace probability_same_length_segments_l92_92147

theorem probability_same_length_segments (T : Finset ℕ) (sides : Finset ℕ) (diagonals : Finset ℕ) :
  sides.card = 6 →
  diagonals.card = 9 →
  (∃ d1 d2 : Finset ℕ, d1.card = 3 ∧ d2.card = 3 ∧ d1 ∩ d2 = ∅ ∧ diagonals = d1 ∪ d2) →
  T = sides ∪ diagonals →
  (∃ n : ℕ, (T.card.choose 2) = n * 5 ∧ ∃ f : ℕ, (∑ x in T, ∑ y in (T \ {x}), if x = y then 1 else 0) = f * 5) →
  ((∑ x in T, ∑ y in (T \ {x}), if x = y then 1 else 0).toRat / (T.card.choose 2).toRat = (1 : ℚ) / 5) :=
by
  sorry

end probability_same_length_segments_l92_92147


namespace cos_arcsin_l92_92747

theorem cos_arcsin (h : real.arcsin (3 / 5) = θ) : real.cos θ = 4 / 5 := 
by {
  have h1 : real.sin θ = 3 / 5 := by rwa [real.sin_arcsin],
  have hypo : (4 : real)^2 + (3 : real)^2 = (5 : real)^2 := by norm_num,
  have h2 : abs (real.cos θ) = 4 / 5,
  { rw [real.cos_eq_sqrt_one_sub_sin_sq, h1],
    simp only [sq, pow_two],
    rw [div_pow 3 5],
    norm_num, simp only [real.sqrt_sqr_eq_abs, sqr_pos],
  },
  rw abs_eq_self at h2,
  exact h2,
}

end cos_arcsin_l92_92747


namespace calculate_v3_l92_92606

def f (x : ℤ) : ℤ := 2*x^6 + 5*x^5 + 6*x^4 + 23*x^3 - 8*x^2 + 10*x - 3

def v0 : ℤ := 2
def v1 (x : ℤ) : ℤ := v0 * x + 5
def v2 (x : ℤ) : ℤ := v1 x * x + 6
def v3 (x : ℤ) : ℤ := v2 x * x + 23

theorem calculate_v3 : v3 (-4) = -49 :=
by
sorry

end calculate_v3_l92_92606


namespace starting_number_of_range_l92_92856

theorem starting_number_of_range (N : ℕ) : ∃ (start : ℕ), 
  (∀ n, n ≥ start ∧ n ≤ 200 → ∃ k, 8 * k = n) ∧ -- All numbers between start and 200 inclusive are multiples of 8
  (∃ k, k = (200 / 8) ∧ 25 - k = 13.5) ∧ -- There are 13.5 multiples of 8 in the range
  start = 84 := 
sorry

end starting_number_of_range_l92_92856


namespace directrix_of_parabola_l92_92619

theorem directrix_of_parabola :
  ∀ (x y : ℝ), (y = (x^2 - 4 * x + 4) / 8) → y = -2 :=
sorry

end directrix_of_parabola_l92_92619


namespace interest_rate_eq_five_percent_l92_92487

def total_sum : ℝ := 2665
def P2 : ℝ := 1332.5
def P1 : ℝ := total_sum - P2

theorem interest_rate_eq_five_percent :
  (3 * 0.03 * P1 = r * 0.03 * P2) → r = 5 :=
by
  sorry

end interest_rate_eq_five_percent_l92_92487


namespace triangle_side_lengths_m_range_l92_92525

theorem triangle_side_lengths_m_range (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (m : ℝ) :
  (2 - Real.sqrt 3) < m ∧ m < (2 + Real.sqrt 3) ↔
  (x + y) + Real.sqrt (x^2 + x * y + y^2) > m * Real.sqrt (x * y) ∧
  (x + y) + m * Real.sqrt (x * y) > Real.sqrt (x^2 + x * y + y^2) ∧
  Real.sqrt (x^2 + x * y + y^2) + m * Real.sqrt (x * y) > (x + y) :=
by sorry

end triangle_side_lengths_m_range_l92_92525


namespace andy_last_problem_l92_92903

theorem andy_last_problem (s t : ℕ) (start : s = 75) (total : t = 51) : (s + t - 1) = 125 :=
by
  sorry

end andy_last_problem_l92_92903


namespace intersection_A_B_l92_92797

def is_defined (x : ℝ) : Prop := x^2 - 1 ≥ 0

def range_of_y (y : ℝ) : Prop := y ≥ 0

def A_set : Set ℝ := { x | is_defined x }
def B_set : Set ℝ := { y | range_of_y y }

theorem intersection_A_B : A_set ∩ B_set = { x | 1 ≤ x } := 
sorry

end intersection_A_B_l92_92797


namespace sam_driving_distance_l92_92804

-- Definitions based on the conditions
def marguerite_distance : ℝ := 150
def marguerite_time : ℝ := 3
def sam_time : ℝ := 4

-- Desired statement using the given conditions
theorem sam_driving_distance :
  let rate := marguerite_distance / marguerite_time in
  let sam_distance := rate * sam_time in
  sam_distance = 200 :=
by
  sorry

end sam_driving_distance_l92_92804


namespace problem_conditions_l92_92771

noncomputable def f (x : ℝ) := x^2 - 2 * x * Real.log x
noncomputable def g (x : ℝ) := Real.exp x - (Real.exp 2 * x^2) / 4

theorem problem_conditions :
  (∀ x > 0, deriv f x > 0) ∧ 
  (∃! x, g x = 0) ∧ 
  (∃ x, f x = g x) :=
by
  sorry

end problem_conditions_l92_92771


namespace bertha_descendants_without_daughters_l92_92089

-- Definitions based on conditions
def num_daughters : ℕ := 6
def total_daughters_and_granddaughters : ℕ := 30
def daughters_with_daughters := (total_daughters_and_granddaughters - num_daughters) / 6

-- The number of Bertha's daughters who have no daughters:
def daughters_without_daughters := num_daughters - daughters_with_daughters
-- The number of Bertha's granddaughters:
def num_granddaughters := total_daughters_and_granddaughters - num_daughters
-- All granddaughters have no daughters:
def granddaughters_without_daughters := num_granddaughters

-- The total number of daughters and granddaughters without daughters
def total_without_daughters := daughters_without_daughters + granddaughters_without_daughters

-- Main theorem statement
theorem bertha_descendants_without_daughters :
  total_without_daughters = 26 :=
by
  sorry

end bertha_descendants_without_daughters_l92_92089


namespace opening_price_calculation_l92_92602

variable (Closing_Price : ℝ)
variable (Percent_Increase : ℝ)
variable (Opening_Price : ℝ)

theorem opening_price_calculation
    (H1 : Closing_Price = 28)
    (H2 : Percent_Increase = 0.1200000000000001) :
    Opening_Price = Closing_Price / (1 + Percent_Increase) := by
  sorry

end opening_price_calculation_l92_92602


namespace sum_of_digits_smallest_N_l92_92603

theorem sum_of_digits_smallest_N :
  ∃ (N : ℕ), N ≤ 999 ∧ 72 * N < 1000 ∧ (N = 13) ∧ (1 + 3 = 4) := by
  sorry

end sum_of_digits_smallest_N_l92_92603


namespace sam_distance_l92_92820

theorem sam_distance (m_distance m_time s_time : ℝ) (m_distance_eq : m_distance = 150) (m_time_eq : m_time = 3) (s_time_eq : s_time = 4) :
  let rate := m_distance / m_time,
      s_distance := rate * s_time
  in s_distance = 200 :=
by
  let rate := m_distance / m_time
  let s_distance := rate * s_time
  sorry

end sam_distance_l92_92820


namespace f_log2_9_l92_92922

def f (x : ℝ) : ℝ := sorry

theorem f_log2_9 : 
  (∀ x, f (x + 1) = 1 / f x) → 
  (∀ x, 0 < x ∧ x ≤ 1 → f x = 2^x) → 
  f (Real.log 9 / Real.log 2) = 8 / 9 :=
by
  intros h1 h2
  sorry

end f_log2_9_l92_92922


namespace more_cabbages_produced_l92_92479

theorem more_cabbages_produced
  (square_garden : ∀ n : ℕ, ∃ s : ℕ, s ^ 2 = n)
  (area_per_cabbage : ∀ cabbages : ℕ, cabbages = 11236 → ∃ s : ℕ, s ^ 2 = cabbages) :
  11236 - 105 ^ 2 = 211 := by
sorry

end more_cabbages_produced_l92_92479


namespace factorization_correct_l92_92294

noncomputable def polynomial_expr := 
  λ x : ℝ => (x^2 + 4 * x + 3) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8)

noncomputable def factored_expr := 
  λ x : ℝ => (x^2 + 6 * x + 9) * (x^2 + 6 * x + 1)

theorem factorization_correct : 
  ∀ x : ℝ, polynomial_expr x = factored_expr x :=
by
  intro x
  sorry

end factorization_correct_l92_92294


namespace scientific_notation_of_42000_l92_92965

theorem scientific_notation_of_42000 : 42000 = 4.2 * 10^4 := 
by 
  sorry

end scientific_notation_of_42000_l92_92965


namespace find_n_sequence_l92_92540

theorem find_n_sequence (n : ℕ) (b : ℕ → ℝ)
  (h0 : b 0 = 45) (h1 : b 1 = 80) (hn : b n = 0)
  (hrec : ∀ k, 1 ≤ k ∧ k ≤ n-1 → b (k+1) = b (k-1) - 4 / b k) :
  n = 901 :=
sorry

end find_n_sequence_l92_92540


namespace equilateral_triangle_intersections_l92_92249

-- Define the main theorem based on the conditions

theorem equilateral_triangle_intersections :
  let a_1 := (6 - 1) * (7 - 1) / 2
  let a_2 := (6 - 2) * (7 - 2) / 2
  let a_3 := (6 - 3) * (7 - 3) / 2
  let a_4 := (6 - 4) * (7 - 4) / 2
  let a_5 := (6 - 5) * (7 - 5) / 2
  a_1 + 2 * a_2 + 3 * a_3 + 4 * a_4 + 5 * a_5 = 70 := by
  sorry

end equilateral_triangle_intersections_l92_92249


namespace find_minimal_N_l92_92485

theorem find_minimal_N (N : ℕ) (l m n : ℕ) (h1 : (l - 1) * (m - 1) * (n - 1) = 252)
  (h2 : l ≥ 5 ∨ m ≥ 5 ∨ n ≥ 5) : N = l * m * n → N = 280 :=
by
  sorry

end find_minimal_N_l92_92485


namespace num_common_divisors_of_9240_and_10010_l92_92650

def prime_factors_9240 := {2^3, 3, 5, 7, 11}
def prime_factors_10010 := {2, 3, 5, 7, 11, 13}

theorem num_common_divisors_of_9240_and_10010 : 
  let gcd_9240_10010 := 2310 in
  (∏ p in (finset.filter prime (finset.range 14)), nat.divisors p).card = 32 :=
by
  sorry

end num_common_divisors_of_9240_and_10010_l92_92650


namespace jack_received_emails_in_the_morning_l92_92794

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

end jack_received_emails_in_the_morning_l92_92794


namespace vector_k_range_l92_92517

noncomputable def vector_length (v : (ℝ × ℝ)) : ℝ := (v.1 ^ 2 + v.2 ^ 2).sqrt

theorem vector_k_range :
  let a := (-2, 2)
  let b := (5, k)
  vector_length (a.1 + b.1, a.2 + b.2) ≤ 5 → -6 ≤ k ∧ k ≤ 2 := by
  sorry

end vector_k_range_l92_92517


namespace probability_of_not_all_8_sided_dice_rolling_the_same_number_l92_92040

def probability_not_all_same (total_faces : ℕ) (num_dice : ℕ) : ℚ :=
  let total_outcomes := total_faces ^ num_dice
  let same_number_outcomes := total_faces
  let p_same := same_number_outcomes / total_outcomes
  1 - p_same

theorem probability_of_not_all_8_sided_dice_rolling_the_same_number :
  probability_not_all_same 8 5 = 4095 / 4096 :=
by
  sorry

end probability_of_not_all_8_sided_dice_rolling_the_same_number_l92_92040


namespace polynomial_factorization_l92_92318

theorem polynomial_factorization (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = (x^2 + 6*x + 1) * (x^2 + 6*x + 37) :=
  sorry

end polynomial_factorization_l92_92318


namespace tee_shirts_with_60_feet_of_material_l92_92793

def tee_shirts (f t : ℕ) : ℕ := t / f

theorem tee_shirts_with_60_feet_of_material :
  tee_shirts 4 60 = 15 :=
by
  sorry

end tee_shirts_with_60_feet_of_material_l92_92793


namespace factor_polynomial_l92_92263

theorem factor_polynomial (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = 
  (x^2 + 6*x + 4) * (x^2 + 6*x + 11) :=
by
  sorry

end factor_polynomial_l92_92263


namespace smallest_number_l92_92461

theorem smallest_number (a b c d e: ℕ) (h1: a = 5) (h2: b = 8) (h3: c = 1) (h4: d = 2) (h5: e = 6) :
  min (min (min (min a b) c) d) e = 1 :=
by
  -- Proof skipped using sorry
  sorry

end smallest_number_l92_92461


namespace hyperbola_range_of_k_l92_92520

theorem hyperbola_range_of_k (x y k : ℝ) :
  (∃ x y : ℝ, (x^2 / (1 - 2 * k) - y^2 / (k - 2) = 1) ∧ (1 - 2 * k < 0) ∧ (k - 2 < 0)) →
  (1 / 2 < k ∧ k < 2) :=
by 
  sorry

end hyperbola_range_of_k_l92_92520


namespace area_ratio_l92_92191

variables (l w r : ℝ)

-- Define the conditions
def perimeter_eq_circumference : Prop := 2 * l + 2 * w = 2 * π * r
def length_eq_twice_width : Prop := l = 2 * w

-- Define the theorem to prove the ratio of the areas
theorem area_ratio (h1 : perimeter_eq_circumference l w r) (h2 : length_eq_twice_width l w) :
  (l * w) / (π * r^2) = 2 * π / 9 :=
sorry

end area_ratio_l92_92191


namespace total_candies_is_36_l92_92670

-- Defining the conditions
def candies_per_day (day : String) : Nat :=
  if day = "Monday" ∨ day = "Wednesday" then 2 else 1

def total_candies_per_week : Nat :=
  (candies_per_day "Monday" + candies_per_day "Tuesday"
  + candies_per_day "Wednesday" + candies_per_day "Thursday"
  + candies_per_day "Friday" + candies_per_day "Saturday"
  + candies_per_day "Sunday")

def total_candies_in_weeks (weeks : Nat) : Nat :=
  weeks * total_candies_per_week

-- Stating the theorem
theorem total_candies_is_36 : total_candies_in_weeks 4 = 36 :=
  sorry

end total_candies_is_36_l92_92670


namespace initial_forks_l92_92696

variables (forks knives spoons teaspoons : ℕ)
variable (F : ℕ)

-- Conditions as given
def num_knives := F + 9
def num_spoons := 2 * (F + 9)
def num_teaspoons := F / 2
def total_cutlery := (F + 2) + (F + 11) + (2 * (F + 9) + 2) + (F / 2 + 2)

-- Problem statement to prove
theorem initial_forks :
  (total_cutlery = 62) ↔ (F = 6) :=
by {
  sorry
}

end initial_forks_l92_92696


namespace probability_of_6_heads_in_8_flips_l92_92886

theorem probability_of_6_heads_in_8_flips :
  let n : ℕ := 8
  let k : ℕ := 6
  let total_outcomes := 2 ^ n
  let successful_outcomes := Nat.choose n k
  let probability := (successful_outcomes : ℚ) / total_outcomes
  probability = 7 / 64 := by
  sorry

end probability_of_6_heads_in_8_flips_l92_92886


namespace frisbee_sales_l92_92079

/-- A sporting goods store sold some frisbees, with $3 and $4 price points.
The total receipts from frisbee sales were $204. The fewest number of $4 frisbees that could have been sold is 24.
Prove the total number of frisbees sold is 60. -/
theorem frisbee_sales (x y : ℕ) (h1 : 3 * x + 4 * y = 204) (h2 : 24 ≤ y) : x + y = 60 :=
by {
  -- Proof skipped
  sorry
}

end frisbee_sales_l92_92079


namespace bunchkin_total_distance_l92_92629

theorem bunchkin_total_distance
  (a b c d e : ℕ)
  (ha : a = 17)
  (hb : b = 43)
  (hc : c = 56)
  (hd : d = 66)
  (he : e = 76) :
  (a + b + c + d + e) / 2 = 129 :=
by
  sorry

end bunchkin_total_distance_l92_92629


namespace Samantha_purse_value_l92_92550

def cents_per_penny := 1
def cents_per_nickel := 5
def cents_per_dime := 10
def cents_per_quarter := 25

def number_of_pennies := 2
def number_of_nickels := 1
def number_of_dimes := 3
def number_of_quarters := 2

def total_cents := 
  number_of_pennies * cents_per_penny + 
  number_of_nickels * cents_per_nickel + 
  number_of_dimes * cents_per_dime + 
  number_of_quarters * cents_per_quarter

def percent_of_dollar := (total_cents * 100) / 100

theorem Samantha_purse_value : percent_of_dollar = 87 := by
  sorry

end Samantha_purse_value_l92_92550


namespace sam_drove_200_miles_l92_92824

-- Define the conditions
def marguerite_distance : ℕ := 150
def marguerite_time : ℕ := 3
def sam_time : ℕ := 4
def marguerite_speed := marguerite_distance / marguerite_time

-- Define the question
def sam_distance (speed : ℕ) (time : ℕ) : ℕ := speed * time

-- State the theorem to prove the answer
theorem sam_drove_200_miles :
  sam_distance marguerite_speed sam_time = 200 := by
  sorry

end sam_drove_200_miles_l92_92824


namespace average_mark_of_excluded_students_l92_92557

theorem average_mark_of_excluded_students 
  (N A E A_remaining : ℕ) 
  (hN : N = 25) 
  (hA : A = 80) 
  (hE : E = 5) 
  (hA_remaining : A_remaining = 95) : 
  ∃ A_excluded : ℕ, A_excluded = 20 :=
by
  -- Use the conditions in the proof.
  sorry

end average_mark_of_excluded_students_l92_92557


namespace lower_percentage_increase_l92_92934

theorem lower_percentage_increase (E P : ℝ) (h1 : 1.26 * E = 693) (h2 : (1 + P) * E = 660) : P = 0.2 := by
  sorry

end lower_percentage_increase_l92_92934


namespace binomial_expansion_coeff_x10_sub_x5_eq_251_l92_92932

open BigOperators Polynomial

noncomputable def binomial_coeff (n k : ℕ) : ℕ := Nat.choose n k

theorem binomial_expansion_coeff_x10_sub_x5_eq_251 :
  ∀ (a : Fin 11 → ℤ), (fun (x : ℤ) =>
    x^10 - x^5 - (a 0 + a 1 * (x - 1) + a 2 * (x - 1)^2 + 
                  a 3 * (x - 1)^3 + a 4 * (x - 1)^4 + 
                  a 5 * (x - 1)^5 + a 6 * (x - 1)^6 + 
                  a 7 * (x - 1)^7 + a 8 * (x - 1)^8 + 
                  a 9 * (x - 1)^9 + a 10 * (x - 1)^10)) = 0 → 
  a 5 = 251 := 
by 
  sorry

end binomial_expansion_coeff_x10_sub_x5_eq_251_l92_92932


namespace probability_not_all_same_l92_92054

theorem probability_not_all_same :
  (let total_outcomes := 8 ^ 5 in
   let same_number_outcomes := 8 in
   let probability_all_same := same_number_outcomes / total_outcomes in
   let probability_not_same := 1 - probability_all_same in
   probability_not_same = 4095 / 4096) :=
begin
  sorry
end

end probability_not_all_same_l92_92054


namespace minimize_y_l92_92800

noncomputable def y (x a b : ℝ) : ℝ := (x - a)^2 + (x - b)^2 + 3 * x + 5

theorem minimize_y (a b : ℝ) : 
  ∃ x : ℝ, (∀ x' : ℝ, y x a b ≤ y x' a b) → x = (2 * a + 2 * b - 3) / 4 := by
  sorry

end minimize_y_l92_92800


namespace interest_payment_frequency_l92_92975

theorem interest_payment_frequency (i : ℝ) (EAR : ℝ) (n : ℕ)
  (h1 : i = 0.10) (h2 : EAR = 0.1025) :
  (1 + i / n)^n = 1 + EAR → n = 2 :=
by
  intros
  sorry

end interest_payment_frequency_l92_92975


namespace no_four_distinct_integers_with_product_plus_2006_perfect_square_l92_92086

theorem no_four_distinct_integers_with_product_plus_2006_perfect_square : 
  ¬ ∃ (a b c d : ℕ), 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
  (∃ k1 k2 k3 k4 k5 k6 : ℕ, a * b + 2006 = k1^2 ∧ 
                          a * c + 2006 = k2^2 ∧ 
                          a * d + 2006 = k3^2 ∧ 
                          b * c + 2006 = k4^2 ∧ 
                          b * d + 2006 = k5^2 ∧ 
                          c * d + 2006 = k6^2) := 
sorry

end no_four_distinct_integers_with_product_plus_2006_perfect_square_l92_92086


namespace disproving_rearranged_sum_l92_92539

noncomputable section

open scoped BigOperators

variable {a : ℕ → ℝ} {f : ℕ → ℕ}

-- Conditions
def summable_a (a : ℕ → ℝ) : Prop :=
  ∑' i, a i = 1

def strictly_decreasing_abs (a : ℕ → ℝ) : Prop :=
  ∀ n m, n < m → abs (a n) > abs (a m)

def bijection (f : ℕ → ℕ) : Prop :=
  ∀ n, ∃ m, f m = n

def limit_condition (a : ℕ → ℝ) (f : ℕ → ℕ) : Prop :=
  ∀ ε > 0, ∃ N, ∀ n ≥ N, abs ((f n : ℤ) - (n : ℤ)) * abs (a n) < ε

-- Statement
theorem disproving_rearranged_sum :
  summable_a a ∧
  strictly_decreasing_abs a ∧
  bijection f ∧
  limit_condition a f →
  ∑' i, a (f i) ≠ 1 :=
sorry

end disproving_rearranged_sum_l92_92539


namespace solution_set_of_inequality_l92_92446

theorem solution_set_of_inequality : {x : ℝ | x^2 + x - 6 ≤ 0} = {x : ℝ | -3 ≤ x ∧ x ≤ 2} :=
sorry

end solution_set_of_inequality_l92_92446


namespace problem_l92_92436

theorem problem (a b a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ : ℕ)
  (b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ b₁₀ b₁₁ : ℕ) 
  (h1 : a₁ < a₂) (h2 : a₂ < a₃) (h3 : a₃ < a₄) 
  (h4 : a₄ < a₅) (h5 : a₅ < a₆) (h6 : a₆ < a₇)
  (h7 : a₇ < a₈) (h8 : a₈ < a₉) (h9 : a₉ < a₁₀)
  (h10 : a₁₀ < a₁₁) (h11 : b₁ < b₂) (h12 : b₂ < b₃)
  (h13 : b₃ < b₄) (h14 : b₄ < b₅) (h15 : b₅ < b₆)
  (h16 : b₆ < b₇) (h17 : b₇ < b₈) (h18 : b₈ < b₉)
  (h19 : b₉ < b₁₀) (h20 : b₁₀ < b₁₁) 
  (h21 : a₁₀ + b₁₀ = a) (h22 : a₁₁ + b₁₁ = b) : 
  a = 1024 ∧ b = 2048 :=
sorry

end problem_l92_92436


namespace dice_sum_probability_l92_92699

theorem dice_sum_probability :
  let outcomes := 36 in
  let favorable := 26 in -- Total number of pairs where the sum is less than 9
  (favorable.toRat / outcomes.toRat) = (13 / 18) :=
by sorry

end dice_sum_probability_l92_92699


namespace ivan_uses_more_paint_l92_92454

-- Define the basic geometric properties
def rectangular_section_area (length width : ℝ) : ℝ := length * width
def parallelogram_section_area (side1 side2 : ℝ) (angle : ℝ) : ℝ := side1 * side2 * Real.sin angle

-- Define the areas for each neighbor's fences
def ivan_area : ℝ := rectangular_section_area 5 2
def petr_area (alpha : ℝ) : ℝ := parallelogram_section_area 5 2 alpha

-- Theorem stating that Ivan's total fence area is greater than Petr's total fence area provided the conditions
theorem ivan_uses_more_paint (α : ℝ) (hα : α ≠ Real.pi / 2) : ivan_area > petr_area α := by
  sorry

end ivan_uses_more_paint_l92_92454


namespace average_speed_correct_l92_92735

noncomputable def average_speed (initial_odometer : ℝ) (lunch_odometer : ℝ) (final_odometer : ℝ) (total_time : ℝ) : ℝ :=
  (final_odometer - initial_odometer) / total_time

theorem average_speed_correct :
  average_speed 212.3 372 467.2 6.25 = 40.784 :=
by
  unfold average_speed
  sorry

end average_speed_correct_l92_92735


namespace eq_implies_sq_eq_l92_92439

theorem eq_implies_sq_eq (a b : ℝ) (h : a = b) : a^2 = b^2 :=
sorry

end eq_implies_sq_eq_l92_92439


namespace problem_solution_l92_92471

-- Define the arithmetic sequence and its sum
def arith_seq_sum (n : ℕ) (a1 d : ℕ) : ℕ :=
  n * (2 * a1 + (n - 1) * d) / 2

-- Define the specific condition for our problem
def a1_a5_equal_six (a1 d : ℕ) : Prop :=
  a1 + (a1 + 4 * d) = 6

-- The target value of S5 that we want to prove
def S5 (a1 d : ℕ) : ℕ :=
  arith_seq_sum 5 a1 d

theorem problem_solution (a1 d : ℕ) (h : a1_a5_equal_six a1 d) : S5 a1 d = 15 :=
by
  sorry

end problem_solution_l92_92471


namespace increasing_exponential_function_range_l92_92644

theorem increasing_exponential_function_range (a : ℝ) (f : ℝ → ℝ) 
    (h1 : ∀ (x : ℝ), f x = a ^ x) 
    (h2 : a > 0)
    (h3 : a ≠ 1)
    (h4 : ∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2) : a > 1 := 
sorry

end increasing_exponential_function_range_l92_92644


namespace unaccounted_bottles_l92_92244

theorem unaccounted_bottles :
  let total_bottles := 254
  let football_bottles := 11 * 6
  let soccer_bottles := 53
  let lacrosse_bottles := football_bottles + 12
  let rugby_bottles := 49
  let team_bottles := football_bottles + soccer_bottles + lacrosse_bottles + rugby_bottles
  total_bottles - team_bottles = 8 :=
by
  rfl

end unaccounted_bottles_l92_92244


namespace maximum_value_existence_l92_92666

open Real

theorem maximum_value_existence (a b c : ℝ) (h : 9 * a^2 + 4 * b^2 + 25 * c^2 = 1) :
    8 * a + 3 * b + 5 * c ≤ sqrt (373 / 36) := by
  sorry

end maximum_value_existence_l92_92666


namespace cos_arcsin_l92_92746

theorem cos_arcsin (h : real.arcsin (3 / 5) = θ) : real.cos θ = 4 / 5 := 
by {
  have h1 : real.sin θ = 3 / 5 := by rwa [real.sin_arcsin],
  have hypo : (4 : real)^2 + (3 : real)^2 = (5 : real)^2 := by norm_num,
  have h2 : abs (real.cos θ) = 4 / 5,
  { rw [real.cos_eq_sqrt_one_sub_sin_sq, h1],
    simp only [sq, pow_two],
    rw [div_pow 3 5],
    norm_num, simp only [real.sqrt_sqr_eq_abs, sqr_pos],
  },
  rw abs_eq_self at h2,
  exact h2,
}

end cos_arcsin_l92_92746


namespace solve_first_equation_solve_second_equation_l92_92553

-- Statement for the first equation
theorem solve_first_equation : ∀ x : ℝ, x^2 - 3*x - 4 = 0 ↔ x = 4 ∨ x = -1 := by
  sorry

-- Statement for the second equation
theorem solve_second_equation : ∀ x : ℝ, x * (x - 2) = 1 ↔ x = 1 + Real.sqrt 2 ∨ x = 1 - Real.sqrt 2 := by
  sorry

end solve_first_equation_solve_second_equation_l92_92553


namespace isosceles_triangle_CBD_supplement_l92_92654

/-- Given an isosceles triangle ABC with AC = BC and angle C = 50 degrees,
    and point D such that angle CBD is supplementary to angle ABC,
    prove that angle CBD is 115 degrees. -/
theorem isosceles_triangle_CBD_supplement 
  (A B C D : Type) [DecidableEq A] [DecidableEq B] [DecidableEq C] [DecidableEq D]
  (AC BC : ℝ) (angleBAC angleABC angleC angleCBD : ℝ)
  (isosceles : AC = BC)
  (angle_C_eq : angleC = 50)
  (supplement : angleCBD = 180 - angleABC) :
  angleCBD = 115 :=
sorry

end isosceles_triangle_CBD_supplement_l92_92654


namespace factor_polynomial_l92_92260

theorem factor_polynomial (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = 
  (x^2 + 6*x + 4) * (x^2 + 6*x + 11) :=
by
  sorry

end factor_polynomial_l92_92260


namespace number_of_basketball_cards_l92_92736

theorem number_of_basketball_cards 
  (B : ℕ) -- Number of basketball cards in each box
  (H1 : 4 * B = 40) -- Given condition from equation 4B = 40
  
  (H2 : 4 * B + 40 - 58 = 22) -- Given condition from the total number of cards

: B = 10 := 
by 
  sorry

end number_of_basketball_cards_l92_92736


namespace factorize_expression_l92_92754

theorem factorize_expression (x y : ℝ) : 
  x^2 * (x + 1) - y * (x * y + x) = x * (x - y) * (x + y + 1) :=
by sorry

end factorize_expression_l92_92754


namespace sam_drove_200_miles_l92_92829

theorem sam_drove_200_miles
  (distance_m: ℝ)
  (time_m: ℝ)
  (distance_s: ℝ)
  (time_s: ℝ)
  (rate_m: ℝ)
  (rate_s: ℝ)
  (h1: distance_m = 150)
  (h2: time_m = 3)
  (h3: rate_m = distance_m / time_m)
  (h4: time_s = 4)
  (h5: rate_s = rate_m)
  (h6: distance_s = rate_s * time_s):
  distance_s = 200 :=
by
  sorry

end sam_drove_200_miles_l92_92829


namespace allocation_schemes_l92_92625

theorem allocation_schemes :
  ∃ (n : ℕ), n = 240 ∧ (∀ (volunteers : Fin 5 → Fin 4), 
    (∃ (assign : Fin 5 → Fin 4), 
      (∀ (i : Fin 4), ∃ (j : Fin 5), assign j = i)
      ∧ (∀ (k : Fin 5), assign k ≠ assign k)) 
    → true) := sorry

end allocation_schemes_l92_92625


namespace polynomial_base5_representation_l92_92710

-- Define the polynomials P and Q
def P(x : ℕ) : ℕ := 3 * 5^6 + 0 * 5^5 + 0 * 5^4 + 1 * 5^3 + 2 * 5^2 + 4 * 5 + 1
def Q(x : ℕ) : ℕ := 4 * 5^2 + 3 * 5 + 2

-- Define the representation of these polynomials in base-5
def base5_P : ℕ := 3001241
def base5_Q : ℕ := 432

-- Define the expected interpretation of the base-5 representation in decimal
def decimal_P : ℕ := P 0
def decimal_Q : ℕ := Q 0

-- The proof statement
theorem polynomial_base5_representation :
  decimal_P = base5_P ∧ decimal_Q = base5_Q :=
sorry

end polynomial_base5_representation_l92_92710


namespace ratio_of_red_to_blue_marbles_l92_92240

theorem ratio_of_red_to_blue_marbles (total_marbles yellow_marbles : ℕ) (green_marbles blue_marbles red_marbles : ℕ) 
  (odds_blue : ℚ) 
  (h1 : total_marbles = 60) 
  (h2 : yellow_marbles = 20) 
  (h3 : green_marbles = yellow_marbles / 2) 
  (h4 : red_marbles + blue_marbles = total_marbles - (yellow_marbles + green_marbles)) 
  (h5 : odds_blue = 0.25) 
  (h6 : blue_marbles = odds_blue * (red_marbles + blue_marbles)) : 
  red_marbles / blue_marbles = 11 / 4 := 
by 
  sorry

end ratio_of_red_to_blue_marbles_l92_92240


namespace initial_savings_amount_l92_92222

theorem initial_savings_amount (A : ℝ) (P : ℝ) (r1 r2 t1 t2 : ℝ) (hA : A = 2247.50) (hr1 : r1 = 0.08) (hr2 : r2 = 0.04) (ht1 : t1 = 0.25) (ht2 : t2 = 0.25) :
  P = 2181 :=
by
  sorry

end initial_savings_amount_l92_92222


namespace sum_of_ages_is_60_l92_92854

theorem sum_of_ages_is_60 (A B : ℕ) (h1 : A = 2 * B) (h2 : (A + 3) + (B + 3) = 66) : A + B = 60 :=
by sorry

end sum_of_ages_is_60_l92_92854


namespace cos_pi_minus_2alpha_l92_92767

theorem cos_pi_minus_2alpha {α : ℝ} (h : Real.sin α = 2 / 3) : Real.cos (π - 2 * α) = -1 / 9 :=
by
  sorry

end cos_pi_minus_2alpha_l92_92767


namespace daughter_age_l92_92567

theorem daughter_age (m d : ℕ) (h1 : m + d = 60) (h2 : m - 10 = 7 * (d - 10)) : d = 15 :=
sorry

end daughter_age_l92_92567


namespace factor_polynomial_l92_92304

theorem factor_polynomial :
  (x^2 + 4 * x + 3) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8) = 
  (x^2 + 6 * x + 7) * (x^2 + 6 * x + 8) :=
sorry

end factor_polynomial_l92_92304


namespace hexagon_probability_same_length_l92_92156

-- Define the problem
theorem hexagon_probability_same_length (T : Finset (Fin 15)) :
  let num_favorable_outcomes := 33
  let total_possible_outcomes := 105
  (num_favorable_outcomes / total_possible_outcomes) = (11 / 35) :=
by
  sorry

end hexagon_probability_same_length_l92_92156


namespace find_n_l92_92876

noncomputable def n (n : ℕ) : Prop :=
  lcm n 12 = 42 ∧ gcd n 12 = 6

theorem find_n (n : ℕ) (h : lcm n 12 = 42) (h1 : gcd n 12 = 6) : n = 21 :=
by sorry

end find_n_l92_92876


namespace chris_pounds_of_nuts_l92_92607

theorem chris_pounds_of_nuts :
  ∀ (R : ℝ) (x : ℝ),
  (∃ (N : ℝ), N = 4 * R) →
  (∃ (total_mixture_cost : ℝ), total_mixture_cost = 3 * R + 4 * R * x) →
  (3 * R = 0.15789473684210525 * total_mixture_cost) →
  x = 4 :=
by
  intros R x hN htotal_mixture_cost hRA
  sorry

end chris_pounds_of_nuts_l92_92607


namespace sector_area_eq_4cm2_l92_92526

variable (α : ℝ) (l : ℝ) (R : ℝ)
variable (h_alpha : α = 2) (h_l : l = 4) (h_R : R = l / α)

theorem sector_area_eq_4cm2
    (h_alpha : α = 2)
    (h_l : l = 4)
    (h_R : R = l / α) :
    (1/2 * l * R) = 4 := by
  sorry

end sector_area_eq_4cm2_l92_92526


namespace trevor_eggs_l92_92399

theorem trevor_eggs :
  let gertrude := 4
  let blanche := 3
  let nancy := 2
  let martha := 2
  let ophelia := 5
  let penelope := 1
  let quinny := 3
  let dropped := 2
  let gifted := 3
  let total_collected := gertrude + blanche + nancy + martha + ophelia + penelope + quinny
  let remaining_after_drop := total_collected - dropped
  let final_eggs := remaining_after_drop - gifted
  final_eggs = 15 := by
    sorry

end trevor_eggs_l92_92399


namespace percentage_increase_biographies_l92_92664

variable (B b n : ℝ)
variable (h1 : b = 0.20 * B)
variable (h2 : b + n = 0.32 * (B + n))

theorem percentage_increase_biographies (B b n : ℝ) (h1 : b = 0.20 * B) (h2 : b + n = 0.32 * (B + n)) :
  n / b * 100 = 88.24 := by
  sorry

end percentage_increase_biographies_l92_92664


namespace sqrt_3x_eq_5x_largest_value_l92_92031

theorem sqrt_3x_eq_5x_largest_value (x : ℝ) (h : Real.sqrt (3 * x) = 5 * x) : x ≤ 3 / 25 := 
by
  sorry

end sqrt_3x_eq_5x_largest_value_l92_92031


namespace find_a5_l92_92792

variable (a : ℕ → ℝ)
variable (q : ℝ)
variable (a1 : ℝ)

-- Geometric sequence definition
def geometric_sequence (a : ℕ → ℝ) (a1 : ℝ) (q : ℝ) : Prop :=
∀ (n : ℕ), a (n + 1) = a1 * q^n

-- Given conditions
def condition1 (a : ℕ → ℝ) (a1 : ℝ) (q : ℝ) : Prop :=
a 1 + a 3 = 10

def condition2 (a : ℕ → ℝ) (a1 : ℝ) (q : ℝ) : Prop :=
a 2 + a 4 = -30

-- Theorem to prove
theorem find_a5 (a : ℕ → ℝ) (a1 : ℝ) (q : ℝ)
  (h1 : geometric_sequence a a1 q)
  (h2 : condition1 a a1 q)
  (h3 : condition2 a a1 q) :
  a 5 = 81 := by
  sorry

end find_a5_l92_92792


namespace decrease_percent_in_revenue_l92_92582

-- Definitions based on the conditions
def original_tax (T : ℝ) := T
def original_consumption (C : ℝ) := C
def new_tax (T : ℝ) := 0.70 * T
def new_consumption (C : ℝ) := 1.20 * C

-- Theorem statement for the decrease percent in revenue
theorem decrease_percent_in_revenue (T C : ℝ) (hT : T > 0) (hC : C > 0) : 
  100 * ((original_tax T * original_consumption C - new_tax T * new_consumption C) / (original_tax T * original_consumption C)) = 16 :=
by
  sorry

end decrease_percent_in_revenue_l92_92582


namespace books_distribution_l92_92199

-- Definitions of conditions used in the math problem
def five_books : Finset ℕ := {1, 2, 3, 4, 5}
def four_students : Finset ℕ := {1, 2, 3, 4}

-- Statement encapsulating the proof problem
theorem books_distribution :
  (∃ f : fin 5 → fin 4, 
   (∀ i : fin 4, ∃ j : fin 5, f j = i)) → 
  fintype.card {σ : (fin 5 → fin 4) // ∀ i : fin 4, ∃ j : fin 5, σ j = i} = 240 :=
by
  -- Placeholder for proof
  sorry

end books_distribution_l92_92199


namespace hexagon_probability_same_length_l92_92157

-- Define the problem
theorem hexagon_probability_same_length (T : Finset (Fin 15)) :
  let num_favorable_outcomes := 33
  let total_possible_outcomes := 105
  (num_favorable_outcomes / total_possible_outcomes) = (11 / 35) :=
by
  sorry

end hexagon_probability_same_length_l92_92157


namespace hexagon_probability_same_length_l92_92158

-- Define the problem
theorem hexagon_probability_same_length (T : Finset (Fin 15)) :
  let num_favorable_outcomes := 33
  let total_possible_outcomes := 105
  (num_favorable_outcomes / total_possible_outcomes) = (11 / 35) :=
by
  sorry

end hexagon_probability_same_length_l92_92158


namespace factor_polynomial_l92_92283

-- Define the necessary polynomials
def p (x : ℝ) : ℝ := x^2 + 4*x + 3
def q (x : ℝ) : ℝ := x^2 + 8*x + 15
def r (x : ℝ) : ℝ := x^2 + 6*x - 8

-- State the main theorem
theorem factor_polynomial (x : ℝ) : 
  (p x * q x) + r x = (x^2 + 7*x + 1) * (x^2 + 4*x + 7) :=
by
  sorry

end factor_polynomial_l92_92283


namespace five_eight_sided_dice_not_all_same_l92_92059

noncomputable def probability_not_all_same : ℚ :=
  let total_outcomes := 8^5
  let same_number_outcomes := 8
  1 - (same_number_outcomes / total_outcomes)

theorem five_eight_sided_dice_not_all_same :
  probability_not_all_same = 4095 / 4096 :=
by
  sorry

end five_eight_sided_dice_not_all_same_l92_92059


namespace factorization_identity_l92_92340

-- We are asked to prove the mathematical equality under given conditions.
theorem factorization_identity (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = (x^2 + 6*x + 8)^2 :=
by
  sorry

end factorization_identity_l92_92340


namespace line_intersects_parabola_exactly_once_at_m_l92_92687

theorem line_intersects_parabola_exactly_once_at_m :
  (∃ y : ℝ, -3 * y^2 - 4 * y + 7 = m) → (∃! m : ℝ, m = 25 / 3) :=
by
  intro h
  sorry

end line_intersects_parabola_exactly_once_at_m_l92_92687


namespace sum_of_z_values_l92_92667

def f (x : ℚ) : ℚ := x^2 + x + 1

theorem sum_of_z_values : ∃ z₁ z₂ : ℚ, f (4 * z₁) = 12 ∧ f (4 * z₂) = 12 ∧ (z₁ + z₂ = - 1 / 12) :=
by
  sorry

end sum_of_z_values_l92_92667


namespace left_building_percentage_l92_92179

theorem left_building_percentage (L R : ℝ)
  (middle_building_height : ℝ := 100)
  (total_height : ℝ := 340)
  (condition1 : L + middle_building_height + R = total_height)
  (condition2 : R = L + middle_building_height - 20) :
  (L / middle_building_height) * 100 = 80 := by
  sorry

end left_building_percentage_l92_92179


namespace solve_fractional_equation_l92_92181

theorem solve_fractional_equation (x : ℝ) (hx1 : x ≠ 0) (hx2 : x ≠ 1) :
  (3 / (x^2 - x) + 1 = x / (x - 1)) → x = 3 :=
by
  sorry -- Placeholder for the actual proof

end solve_fractional_equation_l92_92181


namespace ab_is_zero_l92_92112

theorem ab_is_zero (a b : ℝ) (h₁ : a + b = 5) (h₂ : a^3 + b^3 = 125) : a * b = 0 :=
by
  -- Begin proof here
  sorry

end ab_is_zero_l92_92112


namespace find_equation_line_l92_92640

noncomputable def line_through_point_area (A : Real × Real) (S : Real) : Prop :=
  ∃ (k : Real), (k < 0) ∧ (2 * A.1 + A.2 - 4 = 0) ∧
    (1 / 2 * (2 - k) * (1 - 2 / k) = S)

theorem find_equation_line (A : ℝ × ℝ) (S : ℝ) (hA : A = (1, 2)) (hS : S = 4) :
  line_through_point_area A S →
  ∃ l : ℝ → ℝ, ∀ x y : ℝ, y = l x ↔ 2 * x + y - 4 = 0 :=
by
  sorry

end find_equation_line_l92_92640


namespace soap_box_length_l92_92889

def VolumeOfEachSoapBox (L : ℝ) := 30 * L
def VolumeOfCarton := 25 * 42 * 60
def MaximumSoapBoxes := 300

theorem soap_box_length :
  ∀ L : ℝ,
  MaximumSoapBoxes * VolumeOfEachSoapBox L = VolumeOfCarton → 
  L = 7 :=
by
  intros L h
  sorry

end soap_box_length_l92_92889


namespace problem_statement_l92_92668

theorem problem_statement (a b : ℕ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : ∀ n : ℕ, n ≥ 1 → 2^n * b + 1 ∣ a^(2^n) - 1) : a = 1 := by
  sorry

end problem_statement_l92_92668


namespace cross_colors_differ_l92_92085

-- Hypothesis: All cells of a grid are colored in 5 colors.
variables {Color : Type} [fintype Color] [decidable_eq Color] (A : ℕ → ℕ → Color)

-- Hypothesis: In any figure of the form 1x5 strip, all colors are different.
def valid_strip (i j : ℕ) : Prop :=
  (finset.univ.image (A i ∘ (j + .)).to_finset = finset.univ)

-- The target is to prove that in any figure of the 2x2 form, all colors are different.
theorem cross_colors_differ (i j : ℕ) (h_strip : ∀ i j, valid_strip A i j) : 
  ¬ (A i j = A (i + 1) j) ∧ 
  ¬ (A i j = A i (j + 1)) ∧ 
  ¬ (A i j = A (i + 1) (j + 1)) ∧ 
  ¬ (A (i + 1) j = A i (j + 1)) ∧ 
  ¬ (A (i + 1) j = A (i + 1) (j + 1)) ∧ 
  ¬ (A i (j + 1) = A (i + 1) (j + 1)) :=
begin
  sorry
end

end cross_colors_differ_l92_92085


namespace milk_water_mixture_initial_volume_l92_92726

theorem milk_water_mixture_initial_volume
  (M W : ℝ)
  (h1 : 2 * M = 3 * W)
  (h2 : 4 * M = 3 * (W + 58)) :
  M + W = 145 := by
  sorry

end milk_water_mixture_initial_volume_l92_92726


namespace factor_polynomial_l92_92253

theorem factor_polynomial (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = 
  (x^2 + 6*x + 4) * (x^2 + 6*x + 11) :=
by
  sorry

end factor_polynomial_l92_92253


namespace sequence_of_8_numbers_l92_92785

theorem sequence_of_8_numbers :
  ∃ (a b c d e f g h : ℤ), 
    a + b + c = 100 ∧ b + c + d = 100 ∧ c + d + e = 100 ∧ 
    d + e + f = 100 ∧ e + f + g = 100 ∧ f + g + h = 100 ∧ 
    a = 20 ∧ h = 16 ∧ 
    (a, b, c, d, e, f, g, h) = (20, 16, 64, 20, 16, 64, 20, 16) :=
by
  sorry

end sequence_of_8_numbers_l92_92785


namespace point_M_coordinates_l92_92120

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^2 + 1

-- Define the derivative of the function f
def f' (x : ℝ) : ℝ := 4 * x

-- Define the condition given in the problem: instantaneous rate of change
def rate_of_change (a : ℝ) : Prop := f' a = -4

-- Define the point on the curve
def point_M (a b : ℝ) : Prop := f a = b

-- Proof statement
theorem point_M_coordinates : 
  ∃ (a b : ℝ), rate_of_change a ∧ point_M a b ∧ a = -1 ∧ b = 3 :=  
by
  sorry

end point_M_coordinates_l92_92120


namespace arrangement_count_l92_92216

-- Definitions from the conditions
def people : Nat := 5
def valid_positions_for_A : Finset Nat := Finset.range 5 \ {0, 4}

-- The theorem that states the question equals the correct answer given the conditions
theorem arrangement_count (A_positions : Finset Nat := valid_positions_for_A) : 
  ∃ (total_arrangements : Nat), total_arrangements = 72 :=
by
  -- Placeholder for the proof
  sorry

end arrangement_count_l92_92216


namespace greatest_two_digit_multiple_of_7_l92_92023

theorem greatest_two_digit_multiple_of_7 : ∃ n, 10 ≤ n ∧ n < 100 ∧ n % 7 = 0 ∧ ∀ m, 10 ≤ m ∧ m < 100 ∧ m % 7 = 0 → n ≥ m := 
by
  sorry

end greatest_two_digit_multiple_of_7_l92_92023


namespace count_birches_in_forest_l92_92938

theorem count_birches_in_forest:
  ∀ (t p_s p_p : ℕ), t = 4000 → p_s = 10 → p_p = 13 →
  let n_s := (p_s * t) / 100 in
  let n_p := (p_p * t) / 100 in
  let n_o := n_s + n_p in 
  let n_b := t - (n_s + n_p + n_o) in 
  n_b = 2160 :=
by 
  intros t p_s p_p ht hps hpp
  let n_s := (p_s * t) / 100 
  let n_p := (p_p * t) / 100 
  let n_o := n_s + n_p 
  let n_b := t - (n_s + n_p + n_o) 
  exact sorry

end count_birches_in_forest_l92_92938


namespace average_marks_l92_92895

variable (M P C B : ℕ)

theorem average_marks (h1 : M + P = 20) (h2 : C = P + 20) 
  (h3 : B = 2 * M) (h4 : M ≤ 100) (h5 : P ≤ 100) (h6 : C ≤ 100) (h7 : B ≤ 100) :
  (M + C) / 2 = 20 := by
  sorry

end average_marks_l92_92895


namespace robin_piano_highest_before_lowest_l92_92970

def probability_reach_highest_from_middle_C : ℚ :=
  let p_k (k : ℕ) (p_prev : ℚ) (p_next : ℚ) : ℚ := (1/2 : ℚ) * p_prev + (1/2 : ℚ) * p_next
  let p_1 := 0
  let p_88 := 1
  let A := -1/87
  let B := 1/87
  A + B * 40

theorem robin_piano_highest_before_lowest :
  probability_reach_highest_from_middle_C = 13 / 29 :=
by
  sorry

end robin_piano_highest_before_lowest_l92_92970


namespace three_times_sum_first_35_odd_l92_92580

/-- 
The sum of the first n odd numbers --/
def sum_first_n_odd (n : ℕ) : ℕ := n * n

/-- Given that 69 is the 35th odd number --/
theorem three_times_sum_first_35_odd : 3 * sum_first_n_odd 35 = 3675 := by
  sorry

end three_times_sum_first_35_odd_l92_92580


namespace k_domain_all_reals_l92_92756

noncomputable def domain_condition (k : ℝ) : Prop :=
  9 + 28 * k < 0

noncomputable def k_values : Set ℝ :=
  {k : ℝ | domain_condition k}

theorem k_domain_all_reals :
  k_values = {k : ℝ | k < -9 / 28} :=
by
  sorry

end k_domain_all_reals_l92_92756


namespace solve_system_of_equations_l92_92677

variable (a x y z : ℝ)

theorem solve_system_of_equations (h1 : x^2 + y^2 - 2 * z^2 = 2 * a^2)
                                  (h2 : x + y + 2 * z = 4 * (a^2 + 1))
                                  (h3 : z^2 - x * y = a^2) :
                                  (x = a^2 + a + 1 ∧ y = a^2 - a + 1 ∧ z = a^2 + 1) ∨
                                  (x = a^2 - a + 1 ∧ y = a^2 + a + 1 ∧ z = a^2 + 1) :=
by
  sorry

end solve_system_of_equations_l92_92677


namespace total_carriages_proof_l92_92707

noncomputable def total_carriages (E N' F N : ℕ) : ℕ :=
  E + N + N' + F

theorem total_carriages_proof
  (E N N' F : ℕ)
  (h1 : E = 130)
  (h2 : E = N + 20)
  (h3 : N' = 100)
  (h4 : F = N' + 20) :
  total_carriages E N' F N = 460 := by
  sorry

end total_carriages_proof_l92_92707


namespace at_most_one_dwarf_tells_truth_l92_92410

noncomputable def dwarfs_tell_truth : Prop :=
  ∀ heights: Fin 7 → ℕ,
  heights = fun i => i + 60 →
  ∀ k ∈ Finset.range 7, (heights k) ≠ (60 + k) → (∃ i ∈ Finset.range 7, i ≠ k)

theorem at_most_one_dwarf_tells_truth :
  dwarfs_tell_truth = 1 := 
sorry

end at_most_one_dwarf_tells_truth_l92_92410


namespace evaluate_expression_l92_92090

theorem evaluate_expression : 10 * 0.2 * 5 * 0.1 + 5 = 6 :=
by
  -- transformed step-by-step mathematical proof goes here
  sorry

end evaluate_expression_l92_92090


namespace sum_of_prism_features_l92_92951

theorem sum_of_prism_features : (12 + 8 + 6 = 26) := by
  sorry

end sum_of_prism_features_l92_92951


namespace contractor_absent_days_l92_92722

theorem contractor_absent_days (x y : ℝ) 
  (h1 : x + y = 30) 
  (h2 : 25 * x - 7.5 * y = 685) : 
  y = 2 :=
by
  sorry

end contractor_absent_days_l92_92722


namespace student_marks_l92_92081

def max_marks : ℕ := 600
def passing_percentage : ℕ := 30
def fail_by : ℕ := 100

theorem student_marks :
  ∃ x : ℕ, x + fail_by = (passing_percentage * max_marks) / 100 :=
sorry

end student_marks_l92_92081


namespace original_proposition_false_implies_negation_true_l92_92927

-- Define the original proposition and its negation
def original_proposition (x y : ℝ) : Prop := (x + y > 0) → (x > 0 ∧ y > 0)
def negation (x y : ℝ) : Prop := ¬ original_proposition x y

-- Theorem statement
theorem original_proposition_false_implies_negation_true (x y : ℝ) : ¬ original_proposition x y → negation x y :=
by
  -- Since ¬ original_proposition x y implies the negation is true
  intro h
  exact h

end original_proposition_false_implies_negation_true_l92_92927


namespace consecutiveWhiteBallsProb_l92_92943

-- Definitions based on conditions
def totalBalls : ℕ := 9
def whiteBalls : ℕ := 5
def blackBalls : ℕ := 4
def firstDrawWhiteProb : ℚ := whiteBalls / totalBalls
def secondDrawWhiteProb : ℚ := (whiteBalls - 1) / (totalBalls - 1)

-- Proving that the probability of drawing two white balls consecutively is 5/18
theorem consecutiveWhiteBallsProb : firstDrawWhiteProb * secondDrawWhiteProb = 5 / 18 := by
  sorry

end consecutiveWhiteBallsProb_l92_92943


namespace min_value_frac_l92_92920

open Real

theorem min_value_frac (a b : ℝ) (h1 : a + b = 1/2) (h2 : a > 0) (h3 : b > 0) :
    (4 / a + 1 / b) = 18 :=
sorry

end min_value_frac_l92_92920


namespace problem1_sin_cos_problem2_linear_combination_l92_92513

/-- Problem 1: Prove that sin(α) * cos(α) = -2/5 given that the terminal side of angle α passes through (-1, 2) --/
theorem problem1_sin_cos (α : ℝ) (x y : ℝ) (h1 : x = -1) (h2 : y = 2) (h3 : Real.sqrt (x^2 + y^2) ≠ 0) :
  Real.sin α * Real.cos α = -2 / 5 :=
by
  sorry

/-- Problem 2: Prove that 10sin(α) + 3cos(α) = 0 given that the terminal side of angle α lies on the line y = -3x --/
theorem problem2_linear_combination (α : ℝ) (x y : ℝ) (h1 : y = -3 * x) (h2 : (x = -1 ∧ y = 3) ∨ (x = 1 ∧ y = -3)) (h3 : Real.sqrt (x^2 + y^2) ≠ 0) :
  10 * Real.sin α + 3 / Real.cos α = 0 :=
by
  sorry

end problem1_sin_cos_problem2_linear_combination_l92_92513


namespace e_exp_f_neg2_l92_92845

noncomputable def f : ℝ → ℝ := sorry

-- Conditions:
axiom h_odd : ∀ x : ℝ, f (-x) = -f x
axiom h_ln_pos : ∀ x : ℝ, x > 0 → f x = Real.log x

-- Theorem to prove:
theorem e_exp_f_neg2 : Real.exp (f (-2)) = 1 / 2 := by
  sorry

end e_exp_f_neg2_l92_92845


namespace factorization_identity_l92_92338

-- We are asked to prove the mathematical equality under given conditions.
theorem factorization_identity (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = (x^2 + 6*x + 8)^2 :=
by
  sorry

end factorization_identity_l92_92338


namespace evaluate_expression_l92_92252

theorem evaluate_expression : (1 / (3 - (1 / (3 - (1 / (3 - (1 / 3))))))) = (8 / 21) :=
by
  sorry

end evaluate_expression_l92_92252


namespace Taso_riddles_correct_l92_92662

-- Definitions based on given conditions
def Josh_riddles : ℕ := 8
def Ivory_riddles : ℕ := Josh_riddles + 4
def Taso_riddles : ℕ := 2 * Ivory_riddles

-- The theorem to prove
theorem Taso_riddles_correct : Taso_riddles = 24 := by
  sorry

end Taso_riddles_correct_l92_92662


namespace largest_angle_in_triangle_l92_92196

theorem largest_angle_in_triangle (A B C : ℝ) (h₁ : A + B = 126) (h₂ : A = B + 20) (h₃ : A + B + C = 180) :
  max A (max B C) = 73 := sorry

end largest_angle_in_triangle_l92_92196


namespace complement_of_A_with_respect_to_U_l92_92364

-- Definitions
def U : Set ℤ := {-2, -1, 1, 3, 5}
def A : Set ℤ := {-1, 3}

-- Statement of the problem
theorem complement_of_A_with_respect_to_U :
  (U \ A) = {-2, 1, 5} := 
by
  sorry

end complement_of_A_with_respect_to_U_l92_92364


namespace probability_not_all_same_l92_92049

theorem probability_not_all_same (n : ℕ) (h : n = 5) : 
  let total_outcomes := 8^n in
  let same_number_outcomes := 8 in
  let prob_all_same_number := (same_number_outcomes : ℚ) / total_outcomes in
  let prob_not_all_same_number := 1 - prob_all_same_number in
  prob_not_all_same_number = 4095 / 4096 :=
by 
  sorry

end probability_not_all_same_l92_92049


namespace hexagon_probability_same_length_l92_92160

-- Define the problem
theorem hexagon_probability_same_length (T : Finset (Fin 15)) :
  let num_favorable_outcomes := 33
  let total_possible_outcomes := 105
  (num_favorable_outcomes / total_possible_outcomes) = (11 / 35) :=
by
  sorry

end hexagon_probability_same_length_l92_92160


namespace initial_apples_l92_92431

-- Definitions based on the given conditions
def apples_given_away : ℕ := 88
def apples_left : ℕ := 39

-- Statement to prove
theorem initial_apples : apples_given_away + apples_left = 127 :=
by {
  -- Proof steps would go here
  sorry
}

end initial_apples_l92_92431


namespace rectangle_circle_area_ratio_l92_92190

theorem rectangle_circle_area_ratio (w r : ℝ) (h1 : 2 * 2 * w + 2 * w = 2 * pi * r) :
  ((2 * w) * w) / (pi * r^2) = 2 * pi / 9 :=
by
  sorry

end rectangle_circle_area_ratio_l92_92190


namespace dave_shirts_not_washed_l92_92877

variable (short_sleeve_shirts long_sleeve_shirts washed_shirts : ℕ)

theorem dave_shirts_not_washed (h1 : short_sleeve_shirts = 9) (h2 : long_sleeve_shirts = 27) (h3 : washed_shirts = 20) :
  (short_sleeve_shirts + long_sleeve_shirts - washed_shirts = 16) :=
by {
  -- sorry indicates the proof is omitted
  sorry
}

end dave_shirts_not_washed_l92_92877


namespace shortest_altitude_of_right_triangle_l92_92983

theorem shortest_altitude_of_right_triangle
  (a b c : ℝ)
  (ha : a = 9) 
  (hb : b = 12) 
  (hc : c = 15)
  (ht : a^2 + b^2 = c^2) :
  ∃ h : ℝ, (1 / 2) * c * h = (1 / 2) * a * b ∧ h = 7.2 := by
  sorry

end shortest_altitude_of_right_triangle_l92_92983


namespace number_of_routes_4x3_grid_l92_92609

def binomial_coefficient (n k : ℕ) : ℕ :=
  n.choose k

theorem number_of_routes_4x3_grid : binomial_coefficient 7 4 = 35 := by
  sorry

end number_of_routes_4x3_grid_l92_92609


namespace gcd_18_30_l92_92622

theorem gcd_18_30 : Int.gcd 18 30 = 6 := by
  sorry

end gcd_18_30_l92_92622


namespace sum_of_roots_of_polynomials_l92_92798

theorem sum_of_roots_of_polynomials :
  ∃ (a b : ℝ), (a^4 - 16 * a^3 + 40 * a^2 - 50 * a + 25 = 0) ∧ (b^4 - 24 * b^3 + 216 * b^2 - 720 * b + 625 = 0) ∧ (a + b = 7 ∨ a + b = 3) :=
by 
  sorry

end sum_of_roots_of_polynomials_l92_92798


namespace range_of_f_l92_92980

noncomputable def f (x : ℝ) : ℝ := (Real.cos x) ^ 2 - (Real.sin x) ^ 2 - 2 * (Real.sin x) * (Real.cos x)

theorem range_of_f : 
  (∀ x, 0 ≤ x ∧ x ≤ Real.pi / 2 → f x ≥ -Real.sqrt 2 ∧ f x ≤ 1) :=
sorry

end range_of_f_l92_92980


namespace cos_arcsin_l92_92742

theorem cos_arcsin (x : ℝ) (h : x = 3/5) : Real.cos (Real.arcsin x) = 4/5 := 
by
  rw h
  sorry

end cos_arcsin_l92_92742


namespace parabola_no_real_intersection_l92_92359

theorem parabola_no_real_intersection (a b c : ℝ) (h₁ : a = 1) (h₂ : b = -4) (h₃ : c = 5) :
  ∀ (x : ℝ), ¬ (a * x^2 + b * x + c = 0) :=
by
  sorry

end parabola_no_real_intersection_l92_92359


namespace quadratic_function_opens_downwards_l92_92370

theorem quadratic_function_opens_downwards (m : ℤ) (h1 : |m| = 2) (h2 : m + 1 < 0) : m = -2 := by
  sorry

end quadratic_function_opens_downwards_l92_92370


namespace eleventh_term_of_sequence_l92_92913

def inversely_proportional_sequence (a : ℕ → ℝ) (c : ℝ) : Prop :=
  ∀ n : ℕ, a n * a (n + 1) = c

theorem eleventh_term_of_sequence :
  ∃ a : ℕ → ℝ,
    (a 1 = 3) ∧
    (a 2 = 6) ∧
    inversely_proportional_sequence a 18 ∧
    a 11 = 3 :=
by
  sorry

end eleventh_term_of_sequence_l92_92913


namespace reflection_across_x_axis_l92_92944

theorem reflection_across_x_axis (x y : ℝ) : 
  (x, -y) = (-2, -4) ↔ (x, y) = (-2, 4) :=
by
  sorry

end reflection_across_x_axis_l92_92944


namespace probability_of_not_all_8_sided_dice_rolling_the_same_number_l92_92041

def probability_not_all_same (total_faces : ℕ) (num_dice : ℕ) : ℚ :=
  let total_outcomes := total_faces ^ num_dice
  let same_number_outcomes := total_faces
  let p_same := same_number_outcomes / total_outcomes
  1 - p_same

theorem probability_of_not_all_8_sided_dice_rolling_the_same_number :
  probability_not_all_same 8 5 = 4095 / 4096 :=
by
  sorry

end probability_of_not_all_8_sided_dice_rolling_the_same_number_l92_92041


namespace chessboard_not_divisible_by_10_l92_92397

theorem chessboard_not_divisible_by_10 (board : Fin 8 × Fin 8 → ℕ)
    (operation : (Fin 8 × Fin 8) → bool) -- an abstract representation of selecting a 3x3 or 4x4 square
    (increase : (Fin 8 × Fin 8) → (Fin 8 × Fin 8 → ℕ) → (Fin 8 × Fin 8 → ℕ)) -- an abstract representation of the increase operation
    (goal : (Fin 8 × Fin 8 → ℕ) → Prop) -- a representation of the goal of having all numbers divisible by 10
    : ¬(∃ op_seq : List (Fin 8 × Fin 8), (∀ op ∈ op_seq, operation op) ∧ goal (op_seq.foldl (λ b op, increase op b) board)) :=
by
  -- The proof will go here
  sorry

end chessboard_not_divisible_by_10_l92_92397


namespace possible_values_of_a_l92_92362

-- Declare the sets M and N based on given conditions.
def M : Set ℝ := {x | x^2 + x - 6 = 0}
def N (a : ℝ) : Set ℝ := {x | a * x + 2 = 0}

-- Define a proof where the set of possible values for a is {-1, 0, 2/3}
theorem possible_values_of_a : 
  {a : ℝ | N a ⊆ M} = {-1, 0, 2 / 3} := 
by 
  sorry

end possible_values_of_a_l92_92362


namespace number_is_seven_l92_92869

-- We will define the problem conditions and assert the answer
theorem number_is_seven (x : ℤ) (h : 3 * (2 * x + 9) = 69) : x = 7 :=
by 
  -- Proof will be filled in here
  sorry

end number_is_seven_l92_92869


namespace remainder_when_3_pow_2020_div_73_l92_92703

theorem remainder_when_3_pow_2020_div_73 :
  (3^2020 % 73) = 8 := 
sorry

end remainder_when_3_pow_2020_div_73_l92_92703


namespace max_truthful_dwarfs_l92_92406

theorem max_truthful_dwarfs :
  ∃ n : ℕ, n = 1 ∧ 
    ∀ heights : Fin 7 → ℕ, 
      (heights 0 = 60 ∨ heights 1 = 61 ∨ heights 2 = 62 ∨ heights 3 = 63 ∨ heights 4 = 64 ∨ heights 5 = 65 ∨ heights 6 = 66) → 
      (∀ i j : Fin 7, i < j → heights i > heights j) → 
      ( ∃ t : Fin 7 → bool, 
        ( ∀ i j : Fin 7, t i = tt → t j = tt → i = j) ∧ 
        ( ∀ i : Fin 7, t i = tt → heights i = 60 + i ) ∧
        #(t.to_list.count tt) = n ) :=
by
  use 1
  sorry

end max_truthful_dwarfs_l92_92406


namespace missed_questions_l92_92964

theorem missed_questions (F M : ℕ) (h1 : M = 5 * F) (h2 : M + F = 216) : M = 180 :=
by
  sorry

end missed_questions_l92_92964


namespace probability_not_all_same_l92_92039

theorem probability_not_all_same (n m : ℕ) (h₁ : n = 5) (h₂ : m = 8) 
  (fair_dice : ∀ (die : ℕ), 1 ≤ die ∧ die ≤ m)
  : (1 - (m / m^n) = 4095 / 4096) := by
  sorry

end probability_not_all_same_l92_92039


namespace eq_implies_sq_eq_l92_92438

theorem eq_implies_sq_eq (a b : ℝ) (h : a = b) : a^2 = b^2 :=
sorry

end eq_implies_sq_eq_l92_92438


namespace total_bike_cost_l92_92831

def marions_bike_cost : ℕ := 356
def stephanies_bike_cost : ℕ := 2 * marions_bike_cost

theorem total_bike_cost : marions_bike_cost + stephanies_bike_cost = 1068 := by
  sorry

end total_bike_cost_l92_92831


namespace first_number_positive_l92_92949

-- Define the initial condition
def initial_pair : ℕ × ℕ := (1, 1)

-- Define the allowable transformations
def transform1 (x y : ℕ) : Prop :=
(x, y - 1) = initial_pair ∨ (x + y, y + 1) = initial_pair

def transform2 (x y : ℕ) : Prop :=
(x, x * y) = initial_pair ∨ (1 / x, y) = initial_pair

-- Define discriminant function
def discriminant (a b : ℕ) : ℤ := b ^ 2 - 4 * a

-- Define the invariants maintained by the transformations
def invariant (a b : ℕ) : Prop :=
discriminant a b < 0

-- Statement to be proven
theorem first_number_positive :
(∀ (a b : ℕ), invariant a b → a > 0) :=
by
  sorry

end first_number_positive_l92_92949


namespace probability_of_cut_l92_92228

-- Given: A rope of length 4 meters
-- Cutting point C is uniformly distributed between 0 and 4 meters
-- Question: Prove that the probability that one of the pieces is at least 3 times as long as the other
--           and the shorter piece is at least 0.5 meters is 0.25

noncomputable def ropeProbability : ℝ :=
  let p := pmf.uniform (closedInterval (0, 4))
  p.toReal ((set.Icc (0.5, 1)).union (set.Icc (3, 3.5)))

theorem probability_of_cut :
  ropeProbability = 0.25 :=
by
  sorry

end probability_of_cut_l92_92228


namespace probability_at_least_one_boy_and_one_girl_l92_92193

open Finset

theorem probability_at_least_one_boy_and_one_girl :
  let total_members := 30
  let boys := 12
  let girls := 18
  let committee_size := 5
  let total_committees := choose total_members committee_size
  let all_boy_committees := choose boys committee_size
  let all_girl_committees := choose girls committee_size
  let favorable_committees := total_committees - (all_boy_committees + all_girl_committees)
  (favorable_committees / total_committees : ℚ) = 571 / 611 := 
by {
  let total_committees := choose total_members committee_size,
  let all_boy_committees := choose boys committee_size,
  let all_girl_committees := choose girls committee_size,
  let favorable_committees := total_committees - (all_boy_committees + all_girl_committees),
  exact eq.trans (div_eq_iff (ne_of_gt (nat.cast_pos.2 _)).symm) 
                  (eq.trans _ (eq.refl 571 / 611))
}

end probability_at_least_one_boy_and_one_girl_l92_92193


namespace polynomial_factorization_l92_92310

theorem polynomial_factorization (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = (x^2 + 6*x + 1) * (x^2 + 6*x + 37) :=
  sorry

end polynomial_factorization_l92_92310


namespace verify_addition_by_subtraction_l92_92859

theorem verify_addition_by_subtraction (a b c : ℤ) (h : a + b = c) : (c - a = b) ∧ (c - b = a) :=
by
  sorry

end verify_addition_by_subtraction_l92_92859


namespace lucas_age_correct_l92_92962

variable (Noah_age : ℕ) (Mia_age : ℕ) (Lucas_age : ℕ)

-- Conditions
axiom h1 : Noah_age = 12
axiom h2 : Mia_age = Noah_age + 5
axiom h3 : Lucas_age = Mia_age - 6

-- Goal
theorem lucas_age_correct : Lucas_age = 11 := by
  sorry

end lucas_age_correct_l92_92962


namespace train_speed_l92_92879

theorem train_speed (length : ℝ) (time : ℝ) (conversion_factor: ℝ)
  (h_length : length = 100) 
  (h_time : time = 5) 
  (h_conversion : conversion_factor = 3.6) :
  (length / time * conversion_factor) = 72 :=
by
  sorry

end train_speed_l92_92879


namespace pure_imaginary_a_zero_l92_92369

theorem pure_imaginary_a_zero (a : ℝ) (h : ∃ b : ℝ, (i : ℂ) * (1 + (a : ℂ) * i) = (b : ℂ) * i) : a = 0 :=
by
  sorry

end pure_imaginary_a_zero_l92_92369


namespace product_of_consecutive_integers_between_sqrt_29_l92_92751

-- Define that \(5 \lt \sqrt{29} \lt 6\)
lemma sqrt_29_bounds : 5 < Real.sqrt 29 ∧ Real.sqrt 29 < 6 :=
sorry

-- Main theorem statement
theorem product_of_consecutive_integers_between_sqrt_29 :
  (∃ (a b : ℤ), 5 < Real.sqrt 29 ∧ Real.sqrt 29 < 6 ∧ a = 5 ∧ b = 6 ∧ a * b = 30) := 
sorry

end product_of_consecutive_integers_between_sqrt_29_l92_92751


namespace translation_result_l92_92452

variables (P : ℝ × ℝ) (P' : ℝ × ℝ)

def translate_left (P : ℝ × ℝ) (units : ℝ) := (P.1 - units, P.2)
def translate_down (P : ℝ × ℝ) (units : ℝ) := (P.1, P.2 - units)

theorem translation_result :
    P = (-4, 3) -> P' = translate_down (translate_left P 2) 2 -> P' = (-6, 1) :=
by
  intros h1 h2
  sorry

end translation_result_l92_92452


namespace convert_sq_meters_to_hectares_convert_hours_to_hours_and_minutes_l92_92718

theorem convert_sq_meters_to_hectares :
  (123000 / 10000) = 12.3 :=
by
  sorry

theorem convert_hours_to_hours_and_minutes :
  (4 + 0.25 * 60) = 4 * 60 + 15 :=
by
  sorry

end convert_sq_meters_to_hectares_convert_hours_to_hours_and_minutes_l92_92718


namespace arithmetic_sequence_k_value_l92_92693

theorem arithmetic_sequence_k_value (a : ℕ → ℤ) (S: ℕ → ℤ)
    (h1 : ∀ n, S (n + 1) = S n + a (n + 1))
    (h2 : S 11 = S 4)
    (h3 : a 1 = 1)
    (h4 : ∃ k, a k + a 4 = 0) :
    ∃ k, k = 12 :=
by 
  sorry

end arithmetic_sequence_k_value_l92_92693


namespace cubic_root_equality_l92_92610

theorem cubic_root_equality (a b c : ℝ) (h1 : a + b + c = 12) (h2 : a * b + b * c + c * a = 14) (h3 : a * b * c = -3) :
  (1 / a^2) + (1 / b^2) + (1 / c^2) = 268 / 9 := 
by
  sorry

end cubic_root_equality_l92_92610


namespace sam_drove_200_miles_l92_92827

theorem sam_drove_200_miles
  (distance_m: ℝ)
  (time_m: ℝ)
  (distance_s: ℝ)
  (time_s: ℝ)
  (rate_m: ℝ)
  (rate_s: ℝ)
  (h1: distance_m = 150)
  (h2: time_m = 3)
  (h3: rate_m = distance_m / time_m)
  (h4: time_s = 4)
  (h5: rate_s = rate_m)
  (h6: distance_s = rate_s * time_s):
  distance_s = 200 :=
by
  sorry

end sam_drove_200_miles_l92_92827


namespace rectangle_length_width_l92_92593

theorem rectangle_length_width 
  (x y : ℚ)
  (h1 : x - 5 = y + 2)
  (h2 : x * y = (x - 5) * (y + 2)) :
  x = 25 / 3 ∧ y = 4 / 3 :=
by
  sorry

end rectangle_length_width_l92_92593


namespace tom_needs_495_boxes_l92_92990

-- Define the conditions
def total_chocolate_bars : ℕ := 3465
def chocolate_bars_per_box : ℕ := 7

-- Define the proof statement
theorem tom_needs_495_boxes : total_chocolate_bars / chocolate_bars_per_box = 495 :=
by
  sorry

end tom_needs_495_boxes_l92_92990


namespace cos_arcsin_l92_92738

theorem cos_arcsin (h3: ℝ) (h5: ℝ) (h_op: h3 = 3) (h_hyp: h5 = 5) : 
  Real.cos (Real.arcsin (3 / 5)) = 4 / 5 := 
sorry

end cos_arcsin_l92_92738


namespace minimum_a1_a2_sum_l92_92855

theorem minimum_a1_a2_sum (a : ℕ → ℕ)
  (h : ∀ n ≥ 1, a (n + 2) = (a n + 2017) / (1 + a (n + 1)))
  (positive_terms : ∀ n, a n > 0) :
  a 1 + a 2 = 2018 :=
sorry

end minimum_a1_a2_sum_l92_92855


namespace polynomial_factorization_l92_92314

theorem polynomial_factorization (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = (x^2 + 6*x + 1) * (x^2 + 6*x + 37) :=
  sorry

end polynomial_factorization_l92_92314


namespace factor_polynomial_l92_92261

theorem factor_polynomial (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = 
  (x^2 + 6*x + 4) * (x^2 + 6*x + 11) :=
by
  sorry

end factor_polynomial_l92_92261


namespace janice_initial_sentences_l92_92536

theorem janice_initial_sentences :
  ∀ (initial_sentences total_sentences erased_sentences: ℕ)
    (typed_rate before_break_minutes additional_minutes after_meeting_minutes: ℕ),
  typed_rate = 6 →
  before_break_minutes = 20 →
  additional_minutes = 15 →
  after_meeting_minutes = 18 →
  erased_sentences = 40 →
  total_sentences = 536 →
  (total_sentences - (before_break_minutes * typed_rate + (before_break_minutes + additional_minutes) * typed_rate + after_meeting_minutes * typed_rate - erased_sentences)) = initial_sentences →
  initial_sentences = 138 :=
by
  intros initial_sentences total_sentences erased_sentences typed_rate before_break_minutes additional_minutes after_meeting_minutes
  intros h_rate h_before h_additional h_after_meeting h_erased h_total h_eqn
  rw [h_rate, h_before, h_additional, h_after_meeting, h_erased, h_total] at h_eqn
  linarith

end janice_initial_sentences_l92_92536


namespace probability_same_length_segments_of_regular_hexagon_l92_92144

theorem probability_same_length_segments_of_regular_hexagon (sides diagonals : ℕ) 
  (total_segments : ℕ) (prob_same_length : ℚ) 
  (h_sides : sides = 6)
  (h_diagonals : diagonals = 9)
  (h_total_segments : total_segments = sides + diagonals)
  : prob_same_length = 17 / 35 :=
by
  have total_combinations := total_segments * (total_segments - 1) / 2
  have prob_sides := (sides * (sides - 1) / 2 : ℚ) / total_combinations
  have prob_diagonals := (diagonals * (diagonals - 1) / 2 : ℚ) / total_combinations
  have prob_sum := prob_sides + prob_diagonals
  have h_prob_sides : prob_sides = (6 * 5 / 2) / (15 * 14 / 2) :=
    by sorry
  have h_prob_diagonals : prob_diagonals = (9 * 8 / 2) / (15 * 14 / 2) :=
    by sorry
  calc
    prob_same_length = prob_sum : by sorry
    ... = 17 / 35 : by {rw [←h_prob_sides, ←h_prob_diagonals], sorry}

end probability_same_length_segments_of_regular_hexagon_l92_92144


namespace find_functions_l92_92615

theorem find_functions (f : ℝ → ℝ) :
  (∀ x y : ℝ, (f x + x * y) * f (x - 3 * y) + (f y + x * y) * f (3 * x - y) = (f (x + y))^2) →
  (∀ x : ℝ, f x = 0 ∨ f x = x ^ 2) :=
by
  sorry

end find_functions_l92_92615


namespace Nara_height_is_1_69_l92_92971

-- Definitions of the conditions
def SangheonHeight : ℝ := 1.56
def ChihoHeight : ℝ := SangheonHeight - 0.14
def NaraHeight : ℝ := ChihoHeight + 0.27

-- The statement to prove
theorem Nara_height_is_1_69 : NaraHeight = 1.69 :=
by {
  sorry
}

end Nara_height_is_1_69_l92_92971


namespace directrix_of_parabola_l92_92620

theorem directrix_of_parabola :
  ∀ (x y : ℝ), (y = (x^2 - 4 * x + 4) / 8) → y = -2 :=
sorry

end directrix_of_parabola_l92_92620


namespace tiling_tromino_l92_92139

theorem tiling_tromino (m n : ℕ) : (∀ t : ℕ, (t = 3) → (3 ∣ m * n)) → (m * n % 6 = 0) → (m * n % 6 = 0) :=
by
  sorry

end tiling_tromino_l92_92139


namespace largest_x_satisfies_eq_l92_92024

theorem largest_x_satisfies_eq (x : ℝ) (hx : sqrt (3 * x) = 5 * x) : x ≤ 3 / 25 :=
sorry

end largest_x_satisfies_eq_l92_92024


namespace triangle_perimeter_l92_92789

/-- Given a triangle with two sides of lengths 2 and 5, and the third side being a root of the equation
    x^2 - 8x + 12 = 0, the perimeter of the triangle is 13. --/
theorem triangle_perimeter
  (a b : ℕ) 
  (ha : a = 2) 
  (hb : b = 5)
  (c : ℕ)
  (h_c_root : c * c - 8 * c + 12 = 0)
  (h_triangle : a + b > c ∧ a + c > b ∧ b + c > a) :
  a + b + c = 13 := 
sorry

end triangle_perimeter_l92_92789


namespace if_a_eq_b_then_a_squared_eq_b_squared_l92_92441

theorem if_a_eq_b_then_a_squared_eq_b_squared (a b : ℝ) (h : a = b) : a^2 = b^2 :=
sorry

end if_a_eq_b_then_a_squared_eq_b_squared_l92_92441


namespace factor_poly_eq_factored_form_l92_92274

-- Defining the polynomial expressions
def poly1 (x : ℝ) := x^2 + 4 * x + 3
def poly2 (x : ℝ) := x^2 + 8 * x + 15
def poly3 (x : ℝ) := x^2 + 6 * x - 8

-- The main expression which needs to be factored
def main_expr (x : ℝ) := (poly1 x) * (poly2 x) + (poly3 x)

-- Stating the goal factored form
def factored_form (x : ℝ) := (x^2 + 6 * x + 19) * (x^2 + 6 * x - 2)

-- The theorem statement
theorem factor_poly_eq_factored_form (x : ℝ) : 
  main_expr x = factored_form x := 
by
  sorry

end factor_poly_eq_factored_form_l92_92274


namespace probability_exactly_two_heads_and_two_tails_l92_92127

noncomputable def probability_two_heads_two_tails (n : ℕ) (k : ℕ) (p : ℚ) : ℚ :=
  (Nat.choose n k : ℚ) * (p ^ n)

theorem probability_exactly_two_heads_and_two_tails
  (tosses : ℕ) (k : ℕ) (p : ℚ) (h_tosses : tosses = 4) (h_k : k = 2) (h_p : p = 1/2) :
  probability_two_heads_two_tails tosses k p = 3 / 8 := by
  sorry

end probability_exactly_two_heads_and_two_tails_l92_92127


namespace frog_jump_paths_l92_92723

noncomputable def φ : ℕ × ℕ → ℕ
| (0, 0) => 1
| (x, y) =>
  let φ_x1 := if x > 1 then φ (x - 1, y) else 0
  let φ_x2 := if x > 1 then φ (x - 2, y) else 0
  let φ_y1 := if y > 1 then φ (x, y - 1) else 0
  let φ_y2 := if y > 1 then φ (x, y - 2) else 0
  φ_x1 + φ_x2 + φ_y1 + φ_y2

theorem frog_jump_paths : φ (4, 4) = 556 := sorry

end frog_jump_paths_l92_92723


namespace polynomial_factorization_l92_92312

theorem polynomial_factorization (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = (x^2 + 6*x + 1) * (x^2 + 6*x + 37) :=
  sorry

end polynomial_factorization_l92_92312


namespace log_sqrt_defined_l92_92635

theorem log_sqrt_defined (x : ℝ) : 2 < x ∧ x < 5 ↔ (∃ y z : ℝ, y = log (5 - x) ∧ z = sqrt (x - 2)) :=
by 
  sorry

end log_sqrt_defined_l92_92635


namespace parabola_directrix_standard_eq_l92_92984

theorem parabola_directrix_standard_eq (p : ℝ) (h : p = 2) :
  ∀ y x : ℝ, (x = -1) → (y^2 = 4 * x) :=
by
  sorry

end parabola_directrix_standard_eq_l92_92984


namespace SamDrove200Miles_l92_92809

/-- Given conditions -/
def MargueriteDistance : ℝ := 150
def MargueriteTime : ℝ := 3
def SameRateTime : ℝ := 4

/-- Calculate Marguerite's average speed -/
def MargueriteSpeed : ℝ := MargueriteDistance / MargueriteTime

/-- Calculate distance Sam drove -/
def SamDistance : ℝ := MargueriteSpeed * SameRateTime

/-- The theorem statement: Sam drove 200 miles -/
theorem SamDrove200Miles : SamDistance = 200 := by
  sorry

end SamDrove200Miles_l92_92809


namespace sum_first_60_digits_of_1_div_9999_eq_15_l92_92865

theorem sum_first_60_digits_of_1_div_9999_eq_15 :
  let d := 1 / 9999 in
  let digits := (d.to_decimal 60).take 60 in
  digits.sum = 15 :=
by
  -- Lean code for expressing the decimal representation and summing the digits
  sorry

end sum_first_60_digits_of_1_div_9999_eq_15_l92_92865


namespace part1_part2_l92_92392

noncomputable def A : Set ℝ := {x | x^2 + 4 * x = 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | x^2 + 2 * (a + 1) * x + a^2 - 1 = 0}

theorem part1 (a : ℝ) : A ∪ B a = B a ↔ a = 1 :=
by
  sorry

theorem part2 (a : ℝ) : A ∩ B a = B a ↔ a ≤ -1 ∨ a = 1 :=
by
  sorry

end part1_part2_l92_92392


namespace isosceles_triangle_l92_92921
   
   theorem isosceles_triangle (a b c : ℝ) 
         (h_eqn: (a + c) * 1^2 - 2 * b * 1 - a + c = 0) : 
         c = b :=
   by
   simp at h_eqn,
   sorry
   
end isosceles_triangle_l92_92921


namespace quadratic_has_distinct_real_roots_l92_92500

theorem quadratic_has_distinct_real_roots :
  let a := 2
  let b := 3
  let c := -4
  (b^2 - 4 * a * c) > 0 := by
  sorry

end quadratic_has_distinct_real_roots_l92_92500


namespace ab_is_zero_l92_92114

theorem ab_is_zero (a b : ℝ) (h₁ : a + b = 5) (h₂ : a^3 + b^3 = 125) : a * b = 0 :=
by
  -- Begin proof here
  sorry

end ab_is_zero_l92_92114


namespace max_truthful_dwarfs_l92_92409

theorem max_truthful_dwarfs :
  let heights := [60, 61, 62, 63, 64, 65, 66] in
  (∃ n, n = 1 ∧ ∀ i, i < 7 → i ≠ n → heights[i] ≠ (heights[i] + 1)) :=
by
  let heights := [60, 61, 62, 63, 64, 65, 66]
  existsi 1
  intro i
  intro h
  intro hi
  sorry

end max_truthful_dwarfs_l92_92409


namespace probability_of_6_heads_in_8_flips_l92_92887

theorem probability_of_6_heads_in_8_flips :
  let n : ℕ := 8
  let k : ℕ := 6
  let total_outcomes := 2 ^ n
  let successful_outcomes := Nat.choose n k
  let probability := (successful_outcomes : ℚ) / total_outcomes
  probability = 7 / 64 := by
  sorry

end probability_of_6_heads_in_8_flips_l92_92887


namespace factor_polynomial_l92_92257

theorem factor_polynomial (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = 
  (x^2 + 6*x + 4) * (x^2 + 6*x + 11) :=
by
  sorry

end factor_polynomial_l92_92257


namespace intersection_A_B_l92_92959

-- Defining sets A and B
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 4}

-- Theorem stating the intersection of A and B
theorem intersection_A_B : A ∩ B = {2} := by
  sorry

end intersection_A_B_l92_92959


namespace tank_capacity_l92_92011

theorem tank_capacity (x : ℝ) (h : 0.50 * x = 75) : x = 150 :=
by sorry

end tank_capacity_l92_92011


namespace penelope_min_games_l92_92841

theorem penelope_min_games (m w l: ℕ) (h1: 25 * w - 13 * l = 2007) (h2: m = w + l) : m = 87 := by
  sorry

end penelope_min_games_l92_92841


namespace ratio_Theresa_Timothy_2010_l92_92697

def Timothy_movies_2009 : Nat := 24
def Timothy_movies_2010 := Timothy_movies_2009 + 7
def Theresa_movies_2009 := Timothy_movies_2009 / 2
def total_movies := 129
def Timothy_total_movies := Timothy_movies_2009 + Timothy_movies_2010
def Theresa_total_movies := total_movies - Timothy_total_movies
def Theresa_movies_2010 := Theresa_total_movies - Theresa_movies_2009

theorem ratio_Theresa_Timothy_2010 :
  (Theresa_movies_2010 / Timothy_movies_2010) = 2 :=
by
  sorry

end ratio_Theresa_Timothy_2010_l92_92697


namespace expected_value_coin_flip_l92_92599

-- Definitions based on conditions
def P_heads : ℚ := 2 / 3
def P_tails : ℚ := 1 / 3
def win_heads : ℚ := 4
def lose_tails : ℚ := -9

-- Expected value calculation
def expected_value : ℚ :=
  P_heads * win_heads + P_tails * lose_tails

-- Theorem statement to be proven
theorem expected_value_coin_flip : expected_value = -1 / 3 :=
by sorry

end expected_value_coin_flip_l92_92599


namespace remainder_3_pow_1503_mod_7_l92_92575

theorem remainder_3_pow_1503_mod_7 : 
  (3 ^ 1503) % 7 = 6 := 
by sorry

end remainder_3_pow_1503_mod_7_l92_92575


namespace average_of_expressions_l92_92093

theorem average_of_expressions (y : ℝ) :
  (1 / 3:ℝ) * ((2 * y + 5) + (3 * y + 4) + (7 * y - 2)) = 4 * y + 7 / 3 :=
by sorry

end average_of_expressions_l92_92093


namespace inequality_proof_l92_92772

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 / b) + (b^2 / c) + (c^2 / a) ≥ a + b + c + 4 * (a - b)^2 / (a + b + c) :=
by
  sorry

end inequality_proof_l92_92772


namespace factor_polynomial_l92_92325

theorem factor_polynomial :
  ∀ x : ℝ, 
  (x^2 + 4 * x + 3) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8) 
  = (x^2 + 6 * x + 1) * (x^2 + 6 * x + 37) :=
by
  intros x
  sorry

end factor_polynomial_l92_92325


namespace find_ab_l92_92872

theorem find_ab (a b : ℝ) (h₁ : a - b = 3) (h₂ : a^2 + b^2 = 29) : a * b = 10 :=
sorry

end find_ab_l92_92872


namespace probability_of_same_length_segments_l92_92151

-- Definitions directly from the conditions
def total_elements : ℕ := 6 + 9

-- Probability calculations
def probability_same_length : ℚ :=
  ((6 / total_elements) * (5 / (total_elements - 1))) + 
  ((9 / total_elements) * (8 / (total_elements - 1)))

-- Proof statement
theorem probability_of_same_length_segments : probability_same_length = 17 / 35 := by
  unfold probability_same_length
  sorry

end probability_of_same_length_segments_l92_92151


namespace painting_time_equation_l92_92649

theorem painting_time_equation
  (Hannah_rate : ℝ)
  (Sarah_rate : ℝ)
  (combined_rate : ℝ)
  (temperature_factor : ℝ)
  (break_time : ℝ)
  (t : ℝ)
  (condition1 : Hannah_rate = 1 / 6)
  (condition2 : Sarah_rate = 1 / 8)
  (condition3 : combined_rate = (Hannah_rate + Sarah_rate) * temperature_factor)
  (condition4 : temperature_factor = 0.9)
  (condition5 : break_time = 1.5) :
  (combined_rate * (t - break_time) = 1) ↔ (t = 1 + break_time + 1 / combined_rate) :=
by
  sorry

end painting_time_equation_l92_92649


namespace solution_set_a_neg5_solution_set_general_l92_92646

theorem solution_set_a_neg5 (x : ℝ) : (-5 * x^2 + 3 * x + 2 > 0) ↔ (-2/5 < x ∧ x < 1) := 
sorry

theorem solution_set_general (a x : ℝ) : 
  (ax^2 + (a + 3) * x + 3 > 0) ↔
  ((0 < a ∧ a < 3 ∧ (x < -1 ∨ x > -3/a)) ∨ 
   (a = 3 ∧ x ≠ -1) ∨ 
   (a > 3 ∧ (x < -1 ∨ x > -3/a)) ∨ 
   (a = 0 ∧ x > -1) ∨ 
   (a < 0 ∧ -1 < x ∧ x < -3/a)) := 
sorry

end solution_set_a_neg5_solution_set_general_l92_92646


namespace expected_number_of_draws_l92_92532

-- Given conditions
def redBalls : ℕ := 2
def blackBalls : ℕ := 5
def totalBalls : ℕ := redBalls + blackBalls

-- Definition of expected number of draws
noncomputable def expected_draws : ℚ :=
  (2 * (1/21) + 3 * (2/21) + 4 * (3/21) + 5 * (4/21) + 
   6 * (5/21) + 7 * (6/21))

-- The theorem statement to prove
theorem expected_number_of_draws :
  expected_draws = 16 / 3 := by
  sorry

end expected_number_of_draws_l92_92532


namespace altitude_change_correct_l92_92488

noncomputable def altitude_change (T_ground T_high : ℝ) (deltaT_per_km : ℝ) : ℝ :=
  (T_high - T_ground) / deltaT_per_km

theorem altitude_change_correct :
  altitude_change 18 (-48) (-6) = 11 :=
by 
  sorry

end altitude_change_correct_l92_92488


namespace max_marks_l92_92894

theorem max_marks (M : ℝ) (pass_percent : ℝ) (obtained_marks : ℝ) (failed_by : ℝ) (pass_marks : ℝ) 
  (h1 : pass_percent = 0.40) 
  (h2 : obtained_marks = 150) 
  (h3 : failed_by = 50) 
  (h4 : pass_marks = 200) 
  (h5 : pass_marks = obtained_marks + failed_by) 
  : M = 500 :=
by 
  -- Placeholder for the proof
  sorry

end max_marks_l92_92894


namespace probability_of_same_length_segments_l92_92154

-- Definitions directly from the conditions
def total_elements : ℕ := 6 + 9

-- Probability calculations
def probability_same_length : ℚ :=
  ((6 / total_elements) * (5 / (total_elements - 1))) + 
  ((9 / total_elements) * (8 / (total_elements - 1)))

-- Proof statement
theorem probability_of_same_length_segments : probability_same_length = 17 / 35 := by
  unfold probability_same_length
  sorry

end probability_of_same_length_segments_l92_92154


namespace smallest_value_of_y_undefined_l92_92993

-- Noncomputable means we expect some computations at some point that are not easy to guarantee as computable in Lean
noncomputable def find_smallest_y : Real :=
  let quadratic_formula (a b c : Real) : Real :=
    let discriminant := b * b - 4 * a * c in
    let sqrt_discriminant := Real.sqrt discriminant in
    let y1 := (-b + sqrt_discriminant) / (2 * a) in
    let y2 := (-b - sqrt_discriminant) / (2 * a) in
    if y1 < y2 then y1 else y2
  quadratic_formula 9 (-56) 7

theorem smallest_value_of_y_undefined :
  ∃ y : Real, y = find_smallest_y ∧ y ≈ 0.128 :=
by
  exists find_smallest_y
  unfold find_smallest_y
  sorry

end smallest_value_of_y_undefined_l92_92993


namespace difference_of_squares_l92_92902

theorem difference_of_squares (a b : ℝ) : -4 * a^2 + b^2 = (b + 2 * a) * (b - 2 * a) :=
by
  sorry

end difference_of_squares_l92_92902


namespace angle_K_is_72_l92_92948

variables {J K L M : ℝ}

/-- Given that $JKLM$ is a trapezoid with parallel sides $\overline{JK}$ and $\overline{LM}$,
and given $\angle J = 3\angle M$, $\angle L = 2\angle K$, $\angle J + \angle K = 180^\circ$,
and $\angle L + \angle M = 180^\circ$, prove that $\angle K = 72^\circ$. -/
theorem angle_K_is_72 {J K L M : ℝ}
  (h1 : J = 3 * M)
  (h2 : L = 2 * K)
  (h3 : J + K = 180)
  (h4 : L + M = 180) :
  K = 72 :=
by
  sorry

end angle_K_is_72_l92_92948


namespace find_difference_of_segments_l92_92226

theorem find_difference_of_segments 
  (a b c d x y : ℝ)
  (h1 : a + b = 70)
  (h2 : b + c = 90)
  (h3 : c + d = 130)
  (h4 : a + d = 110)
  (hx_y_sum : x + y = 130)
  (hx_c : x = c)
  (hy_d : y = d) : 
  |x - y| = 13 :=
sorry

end find_difference_of_segments_l92_92226


namespace sum_of_ages_l92_92209

variable (S F : ℕ)

theorem sum_of_ages (h1 : F - 18 = 3 * (S - 18)) (h2 : F = 2 * S) : S + F = 108 := by
  sorry

end sum_of_ages_l92_92209


namespace general_formula_arithmetic_sequence_l92_92643

variable (a : ℕ → ℤ)

def isArithmeticSequence (a : ℕ → ℤ) :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem general_formula_arithmetic_sequence :
  isArithmeticSequence a →
  a 5 = 9 →
  a 1 + a 7 = 14 →
  ∀ n : ℕ, a n = 2 * n - 1 :=
by
  intros h_seq h_a5 h_a17
  sorry

end general_formula_arithmetic_sequence_l92_92643


namespace larger_square_area_multiple_l92_92563

theorem larger_square_area_multiple (a b : ℕ) (h : a = 4 * b) :
  (a ^ 2) = 16 * (b ^ 2) :=
sorry

end larger_square_area_multiple_l92_92563


namespace factor_poly_eq_factored_form_l92_92272

-- Defining the polynomial expressions
def poly1 (x : ℝ) := x^2 + 4 * x + 3
def poly2 (x : ℝ) := x^2 + 8 * x + 15
def poly3 (x : ℝ) := x^2 + 6 * x - 8

-- The main expression which needs to be factored
def main_expr (x : ℝ) := (poly1 x) * (poly2 x) + (poly3 x)

-- Stating the goal factored form
def factored_form (x : ℝ) := (x^2 + 6 * x + 19) * (x^2 + 6 * x - 2)

-- The theorem statement
theorem factor_poly_eq_factored_form (x : ℝ) : 
  main_expr x = factored_form x := 
by
  sorry

end factor_poly_eq_factored_form_l92_92272


namespace sum_outer_equal_sum_inner_l92_92395

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def reverse_digits (n : ℕ) : ℕ :=
  let a := n / 1000
  let b := (n % 1000) / 100
  let c := (n % 100) / 10
  let d := n % 10
  1000 * d + 100 * c + 10 * b + a

theorem sum_outer_equal_sum_inner (M N : ℕ) (a b c d : ℕ) 
  (h1 : is_four_digit M)
  (h2 : M = 1000 * a + 100 * b + 10 * c + d) 
  (h3 : N = reverse_digits M) 
  (h4 : M + N % 101 = 0) 
  (h5 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0) : 
  a + d = b + c :=
  sorry

end sum_outer_equal_sum_inner_l92_92395


namespace xiaoxian_mistake_xiaoxuan_difference_l92_92356

-- Define the initial expressions and conditions
def original_expr := (-9) * 3 - 5
def xiaoxian_expr (x : Int) := (-9) * 3 - x
def xiaoxuan_expr := (-9) / 3 - 5

-- Given conditions
variable (result_xiaoxian : Int)
variable (result_original : Int)

-- Proof statement
theorem xiaoxian_mistake (hx : xiaoxian_expr 2 = -29) : 
  xiaoxian_expr 5 = result_xiaoxian := sorry

theorem xiaoxuan_difference : 
  abs (xiaoxuan_expr - original_expr) = 24 := sorry

end xiaoxian_mistake_xiaoxuan_difference_l92_92356


namespace largest_value_of_x_satisfying_sqrt3x_eq_5x_l92_92034

theorem largest_value_of_x_satisfying_sqrt3x_eq_5x : 
  ∃ (x : ℚ), sqrt (3 * x) = 5 * x ∧ (∀ y : ℚ, sqrt (3 * y) = 5 * y → y ≤ x) ∧ x = 3 / 25 :=
sorry

end largest_value_of_x_satisfying_sqrt3x_eq_5x_l92_92034


namespace shortest_altitude_right_triangle_l92_92982

theorem shortest_altitude_right_triangle (a b c : ℝ) (h : a = 9 ∧ b = 12 ∧ c = 15 ∧ a^2 + b^2 = c^2) :
  let area := 0.5 * a * b in
  let altitude := 2 * area / c in
  altitude = 7.2 :=
by
  cases h with h1 h_rest,
  cases h_rest with h2 h_rest,
  cases h_rest with h3 h4,
  rw [h1, h2, h3] at *,
  let area := 0.5 * 9 * 12,
  let altitude := 2 * area / 15,
  have : area = 54, by norm_num,
  rw this at *,
  have : altitude = 7.2, by norm_num,
  exact this

end shortest_altitude_right_triangle_l92_92982


namespace sufficient_not_necessary_condition_l92_92715

theorem sufficient_not_necessary_condition (a : ℝ)
  : (∃ x : ℝ, a * x^2 + x + 1 ≥ 0) ↔ (∃ x : ℝ, a ≥ 0 ∨ a * x^2 + x + 1 ≥ 0)
:= sorry

end sufficient_not_necessary_condition_l92_92715


namespace sam_distance_traveled_l92_92814

-- Variables definition
variables (distance_marguerite : ℝ) (time_marguerite : ℝ) (time_sam : ℝ)

-- Given conditions
def marguerite_conditions : Prop :=
  distance_marguerite = 150 ∧
  time_marguerite = 3 ∧
  time_sam = 4

-- Statement to prove
theorem sam_distance_traveled (h : marguerite_conditions distance_marguerite time_marguerite time_sam) : 
  distance_marguerite / time_marguerite * time_sam = 200 :=
sorry

end sam_distance_traveled_l92_92814


namespace probability_of_not_all_8_sided_dice_rolling_the_same_number_l92_92043

def probability_not_all_same (total_faces : ℕ) (num_dice : ℕ) : ℚ :=
  let total_outcomes := total_faces ^ num_dice
  let same_number_outcomes := total_faces
  let p_same := same_number_outcomes / total_outcomes
  1 - p_same

theorem probability_of_not_all_8_sided_dice_rolling_the_same_number :
  probability_not_all_same 8 5 = 4095 / 4096 :=
by
  sorry

end probability_of_not_all_8_sided_dice_rolling_the_same_number_l92_92043


namespace equidistant_points_eq_two_l92_92104

noncomputable def number_of_equidistant_points (O : Point) (r d : ℝ) 
  (h1 : d > r) : ℕ := 
2

theorem equidistant_points_eq_two (O : Point) (r d : ℝ) 
  (h1 : d > r) : number_of_equidistant_points O r d h1 = 2 :=
by
  sorry

end equidistant_points_eq_two_l92_92104


namespace negation_proof_l92_92977

theorem negation_proof (a b : ℝ) (h : a^2 + b^2 = 0) : ¬(a = 0 ∧ b = 0) :=
sorry

end negation_proof_l92_92977


namespace factor_poly_eq_factored_form_l92_92271

-- Defining the polynomial expressions
def poly1 (x : ℝ) := x^2 + 4 * x + 3
def poly2 (x : ℝ) := x^2 + 8 * x + 15
def poly3 (x : ℝ) := x^2 + 6 * x - 8

-- The main expression which needs to be factored
def main_expr (x : ℝ) := (poly1 x) * (poly2 x) + (poly3 x)

-- Stating the goal factored form
def factored_form (x : ℝ) := (x^2 + 6 * x + 19) * (x^2 + 6 * x - 2)

-- The theorem statement
theorem factor_poly_eq_factored_form (x : ℝ) : 
  main_expr x = factored_form x := 
by
  sorry

end factor_poly_eq_factored_form_l92_92271


namespace factor_polynomial_l92_92299

theorem factor_polynomial :
  (x^2 + 4 * x + 3) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8) = 
  (x^2 + 6 * x + 7) * (x^2 + 6 * x + 8) :=
sorry

end factor_polynomial_l92_92299


namespace taxi_ride_cost_l92_92897

theorem taxi_ride_cost (base_fare : ℚ) (cost_per_mile : ℚ) (distance : ℕ) :
  base_fare = 2 ∧ cost_per_mile = 0.30 ∧ distance = 10 →
  base_fare + cost_per_mile * distance = 5 :=
by
  sorry

end taxi_ride_cost_l92_92897


namespace avg_score_first_4_l92_92003

-- Definitions based on conditions
def average_score_all_7 : ℝ := 56
def total_matches : ℕ := 7
def average_score_last_3 : ℝ := 69.33333333333333
def matches_first : ℕ := 4
def matches_last : ℕ := 3

-- Calculation of total runs from average scores.
def total_runs_all_7 : ℝ := average_score_all_7 * total_matches
def total_runs_last_3 : ℝ := average_score_last_3 * matches_last

-- Total runs for the first 4 matches
def total_runs_first_4 : ℝ := total_runs_all_7 - total_runs_last_3

-- Prove the average score for the first 4 matches.
theorem avg_score_first_4 :
  (total_runs_first_4 / matches_first) = 46 := 
sorry

end avg_score_first_4_l92_92003


namespace sin_minus_cos_eq_pm_sqrt_b_l92_92935

open Real

/-- If θ is an acute angle such that cos(2θ) = b, then sin(θ) - cos(θ) = ±√b. -/
theorem sin_minus_cos_eq_pm_sqrt_b (θ b : ℝ) (hθ : 0 < θ ∧ θ < π / 2) (hcos2θ : cos (2 * θ) = b) :
  sin θ - cos θ = sqrt b ∨ sin θ - cos θ = -sqrt b :=
sorry

end sin_minus_cos_eq_pm_sqrt_b_l92_92935


namespace eraser_difference_l92_92930

theorem eraser_difference
  (hanna_erasers rachel_erasers tanya_erasers tanya_red_erasers : ℕ)
  (h1 : hanna_erasers = 2 * rachel_erasers)
  (h2 : rachel_erasers = tanya_red_erasers)
  (h3 : tanya_erasers = 20)
  (h4 : tanya_red_erasers = tanya_erasers / 2)
  (h5 : hanna_erasers = 4) :
  rachel_erasers - (tanya_red_erasers / 2) = 5 :=
sorry

end eraser_difference_l92_92930


namespace markup_percentage_l92_92714

variable (W R : ℝ) -- W for Wholesale Cost, R for Retail Cost

-- Conditions:
-- 1. The sweater is sold at a 40% discount.
-- 2. When sold at a 40% discount, the merchant nets a 30% profit on the wholesale cost.
def discount_price (R : ℝ) : ℝ := 0.6 * R
def profit_price (W : ℝ) : ℝ := 1.3 * W

-- Hypotheses
axiom wholesale_cost_is_positive : W > 0
axiom discount_condition : discount_price R = profit_price W

-- Question: Prove that the percentage markup from wholesale to retail price is 116.67%.
theorem markup_percentage (W R : ℝ) 
  (wholesale_cost_is_positive : W > 0)
  (discount_condition : discount_price R = profit_price W) :
  ((R - W) / W * 100) = 116.67 := by
  sorry

end markup_percentage_l92_92714


namespace volume_of_right_triangle_pyramid_l92_92426

noncomputable def pyramid_volume (H α β : ℝ) : ℝ :=
  (H^3 * Real.sin (2 * α)) / (3 * (Real.tan β)^2)

theorem volume_of_right_triangle_pyramid (H α β : ℝ) (alpha_acute : 0 < α ∧ α < π / 2) (H_pos : 0 < H) (beta_acute : 0 < β ∧ β < π / 2) :
  pyramid_volume H α β = (H^3 * Real.sin (2 * α)) / (3 * (Real.tan β)^2) := 
sorry

end volume_of_right_triangle_pyramid_l92_92426


namespace three_digit_number_ends_same_sequence_l92_92762

theorem three_digit_number_ends_same_sequence (N : ℕ) (a b c : ℕ) (h1 : 100 ≤ N ∧ N < 1000)
  (h2 : N % 10 = c)
  (h3 : (N / 10) % 10 = b)
  (h4 : (N / 100) % 10 = a)
  (h5 : a ≠ 0)
  (h6 : N^2 % 1000 = N) :
  N = 127 :=
by
  sorry

end three_digit_number_ends_same_sequence_l92_92762


namespace base_b_for_three_digits_l92_92716

theorem base_b_for_three_digits (b : ℕ) : b = 7 ↔ b^2 ≤ 256 ∧ 256 < b^3 := by
  sorry

end base_b_for_three_digits_l92_92716


namespace ab_zero_l92_92111

theorem ab_zero (a b : ℝ)
  (h1 : a + b = 5)
  (h2 : a^3 + b^3 = 125) : 
  a * b = 0 :=
by
  sorry

end ab_zero_l92_92111


namespace solve_abs_eq_l92_92933

theorem solve_abs_eq (x : ℝ) (h : |x - 1| = 2 * x) : x = 1 / 3 :=
by
  sorry

end solve_abs_eq_l92_92933


namespace proof_system_l92_92676

-- Define the system of equations
def system_of_equations (x y : ℝ) : Prop :=
  6 * x - 2 * y = 1 ∧ 2 * x + y = 2

-- Define the solution to the system of equations
def solution_equations (x y : ℝ) : Prop :=
  x = 0.5 ∧ y = 1

-- Define the system of inequalities
def system_of_inequalities (x : ℝ) : Prop :=
  2 * x - 10 < 0 ∧ (x + 1) / 3 < x - 1

-- Define the solution set for the system of inequalities
def solution_inequalities (x : ℝ) : Prop :=
  2 < x ∧ x < 5

-- The final theorem to be proved
theorem proof_system :
  ∃ x y : ℝ, system_of_equations x y ∧ solution_equations x y ∧ system_of_inequalities x ∧ solution_inequalities x :=
by
  sorry

end proof_system_l92_92676


namespace find_quotient_l92_92445

-- Definitions based on given conditions
def remainder : ℕ := 8
def dividend : ℕ := 997
def divisor : ℕ := 23

-- Hypothesis based on the division formula
def quotient_formula (q : ℕ) : Prop :=
  dividend = (divisor * q) + remainder

-- Statement of the problem
theorem find_quotient (q : ℕ) (h : quotient_formula q) : q = 43 :=
sorry

end find_quotient_l92_92445


namespace product_of_six_numbers_l92_92966

theorem product_of_six_numbers (x y : ℕ) (h1 : x ≠ 0) (h2 : y ≠ 0) 
  (h3 : x^3 * y^2 = 108) : 
  x * y * (x * y) * (x^2 * y) * (x^3 * y^2) * (x^5 * y^3) = 136048896 := 
by
  sorry

end product_of_six_numbers_l92_92966


namespace factor_polynomial_l92_92303

theorem factor_polynomial :
  (x^2 + 4 * x + 3) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8) = 
  (x^2 + 6 * x + 7) * (x^2 + 6 * x + 8) :=
sorry

end factor_polynomial_l92_92303


namespace find_x_when_y_64_l92_92021

theorem find_x_when_y_64 (x y k : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y)
  (h_inv_prop : x^3 * y = k) (h_given : x = 2 ∧ y = 8 ∧ k = 64) :
  y = 64 → x = 1 :=
by
  sorry

end find_x_when_y_64_l92_92021


namespace fraction_product_l92_92992

theorem fraction_product :
  (5 / 8) * (7 / 9) * (11 / 13) * (3 / 5) * (17 / 19) * (8 / 15) = 14280 / 1107000 :=
by sorry

end fraction_product_l92_92992


namespace factor_polynomial_l92_92302

theorem factor_polynomial :
  (x^2 + 4 * x + 3) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8) = 
  (x^2 + 6 * x + 7) * (x^2 + 6 * x + 8) :=
sorry

end factor_polynomial_l92_92302


namespace factorization_correct_l92_92295

noncomputable def polynomial_expr := 
  λ x : ℝ => (x^2 + 4 * x + 3) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8)

noncomputable def factored_expr := 
  λ x : ℝ => (x^2 + 6 * x + 9) * (x^2 + 6 * x + 1)

theorem factorization_correct : 
  ∀ x : ℝ, polynomial_expr x = factored_expr x :=
by
  intro x
  sorry

end factorization_correct_l92_92295


namespace sock_pairs_l92_92132

def total_ways (n_white n_brown n_blue n_red : ℕ) : ℕ :=
  n_blue * n_white + n_blue * n_brown + n_blue * n_red

theorem sock_pairs (n_white n_brown n_blue n_red : ℕ) (h_white : n_white = 5) (h_brown : n_brown = 4) (h_blue : n_blue = 2) (h_red : n_red = 1) :
  total_ways n_white n_brown n_blue n_red = 20 := by
  -- insert the proof steps here
  sorry

end sock_pairs_l92_92132


namespace probability_same_length_segments_l92_92148

theorem probability_same_length_segments (T : Finset ℕ) (sides : Finset ℕ) (diagonals : Finset ℕ) :
  sides.card = 6 →
  diagonals.card = 9 →
  (∃ d1 d2 : Finset ℕ, d1.card = 3 ∧ d2.card = 3 ∧ d1 ∩ d2 = ∅ ∧ diagonals = d1 ∪ d2) →
  T = sides ∪ diagonals →
  (∃ n : ℕ, (T.card.choose 2) = n * 5 ∧ ∃ f : ℕ, (∑ x in T, ∑ y in (T \ {x}), if x = y then 1 else 0) = f * 5) →
  ((∑ x in T, ∑ y in (T \ {x}), if x = y then 1 else 0).toRat / (T.card.choose 2).toRat = (1 : ℚ) / 5) :=
by
  sorry

end probability_same_length_segments_l92_92148


namespace framing_needed_l92_92474

def orig_width : ℕ := 5
def orig_height : ℕ := 7
def border_width : ℕ := 3
def doubling_factor : ℕ := 2
def inches_per_foot : ℕ := 12

-- Define the new dimensions after doubling
def new_width := orig_width * doubling_factor
def new_height := orig_height * doubling_factor

-- Define the dimensions after adding the border
def final_width := new_width + 2 * border_width
def final_height := new_height + 2 * border_width

-- Calculate the perimeter in inches
def perimeter := 2 * (final_width + final_height)

-- Convert perimeter to feet and round up if necessary
def framing_feet := (perimeter + inches_per_foot - 1) / inches_per_foot

theorem framing_needed : framing_feet = 6 := by
  sorry

end framing_needed_l92_92474


namespace probability_winning_probability_not_winning_l92_92657

section Lottery

variable (p1 p2 p3 : ℝ)
variable (h1 : p1 = 0.1)
variable (h2 : p2 = 0.2)
variable (h3 : p3 = 0.4)

theorem probability_winning (h1 : p1 = 0.1) (h2 : p2 = 0.2) (h3 : p3 = 0.4) :
  p1 + p2 + p3 = 0.7 :=
by
  rw [h1, h2, h3]
  norm_num
  done

theorem probability_not_winning (h1 : p1 = 0.1) (h2 : p2 = 0.2) (h3 : p3 = 0.4) :
  1 - (p1 + p2 + p3) = 0.3 :=
by
  rw [h1, h2, h3]
  norm_num
  done

end Lottery

end probability_winning_probability_not_winning_l92_92657


namespace sam_drove_200_miles_l92_92830

theorem sam_drove_200_miles
  (distance_m: ℝ)
  (time_m: ℝ)
  (distance_s: ℝ)
  (time_s: ℝ)
  (rate_m: ℝ)
  (rate_s: ℝ)
  (h1: distance_m = 150)
  (h2: time_m = 3)
  (h3: rate_m = distance_m / time_m)
  (h4: time_s = 4)
  (h5: rate_s = rate_m)
  (h6: distance_s = rate_s * time_s):
  distance_s = 200 :=
by
  sorry

end sam_drove_200_miles_l92_92830


namespace range_of_m_l92_92108

-- Define the discriminant of a quadratic equation
def discriminant(a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Proposition p: The equation x^2 - 2x + m = 0 has two distinct real roots
def p (m : ℝ) : Prop := discriminant 1 (-2) m > 0

-- Proposition q: The function y = (m + 2)x - 1 is monotonically increasing
def q (m : ℝ) : Prop := m + 2 > 0

-- The main theorem stating the conditions and proving the range of m
theorem range_of_m (m : ℝ) (hpq : p m ∨ q m) (hpnq : ¬(p m ∧ q m)) : m ≤ -2 ∨ m ≥ 1 := sorry

end range_of_m_l92_92108


namespace find_a_plus_b_l92_92562

theorem find_a_plus_b (a b : ℝ) (h1 : 2 * a = -6) (h2 : a^2 - b = 4) : a + b = 2 := 
by 
  sorry

end find_a_plus_b_l92_92562


namespace value_of_x_squared_plus_y_squared_l92_92781

theorem value_of_x_squared_plus_y_squared (x y : ℝ) (h : |x - 1/2| + (2*y + 1)^2 = 0) : 
  x^2 + y^2 = 1/2 :=
sorry

end value_of_x_squared_plus_y_squared_l92_92781


namespace largest_x_value_satisfies_largest_x_value_l92_92026

theorem largest_x_value (x : ℚ) (h : Real.sqrt (3 * x) = 5 * x) : x ≤ 3 / 25 :=
sorry

theorem satisfies_largest_x_value : 
  ∃ x : ℚ, Real.sqrt (3 * x) = 5 * x ∧ x = 3 / 25 :=
sorry

end largest_x_value_satisfies_largest_x_value_l92_92026


namespace calc_expression1_calc_expression2_l92_92586

theorem calc_expression1 : (1 / 3)^0 + Real.sqrt 27 - abs (-3) + Real.tan (Real.pi / 4) = 1 + 3 * Real.sqrt 3 - 2 :=
by
  sorry

theorem calc_expression2 (x : ℝ) : (x + 2)^2 - 2 * (x - 1) = x^2 + 2 * x + 6 :=
by
  sorry

end calc_expression1_calc_expression2_l92_92586


namespace superchess_no_attacks_l92_92560

open Finset

theorem superchess_no_attacks (board_size : ℕ) (num_pieces : ℕ)  (attack_limit : ℕ) 
  (h_board_size : board_size = 100) (h_num_pieces : num_pieces = 20) 
  (h_attack_limit : attack_limit = 20) : 
  ∃ (placements : Finset (ℕ × ℕ)), placements.card = num_pieces ∧
  ∀ {p1 p2 : ℕ × ℕ}, p1 ≠ p2 → p1 ∈ placements → p2 ∈ placements → 
  ¬(∃ (attack_positions : Finset (ℕ × ℕ)), attack_positions.card ≤ attack_limit ∧ 
  ∃ piece_pos : ℕ × ℕ, piece_pos ∈ placements ∧ attack_positions ⊆ placements ∧ p1 ∈ attack_positions ∧ p2 ∈ attack_positions) :=
sorry

end superchess_no_attacks_l92_92560


namespace remainder_division_l92_92730

theorem remainder_division {N : ℤ} (k : ℤ) (h : N = 125 * k + 40) : N % 15 = 10 :=
sorry

end remainder_division_l92_92730


namespace area_of_sheet_is_correct_l92_92346

noncomputable def area_of_rolled_sheet (length width height thickness : ℝ) : ℝ :=
  (length * width * height) / thickness

theorem area_of_sheet_is_correct :
  area_of_rolled_sheet 80 20 5 0.1 = 80000 :=
by
  -- The proof is omitted (sorry).
  sorry

end area_of_sheet_is_correct_l92_92346


namespace min_players_team_l92_92230

theorem min_players_team : Nat.lcm (Nat.lcm (Nat.lcm 8 9) 10) 11 = 7920 := 
by 
  -- The proof will be filled here.
  sorry

end min_players_team_l92_92230


namespace probability_same_length_segments_of_regular_hexagon_l92_92142

theorem probability_same_length_segments_of_regular_hexagon (sides diagonals : ℕ) 
  (total_segments : ℕ) (prob_same_length : ℚ) 
  (h_sides : sides = 6)
  (h_diagonals : diagonals = 9)
  (h_total_segments : total_segments = sides + diagonals)
  : prob_same_length = 17 / 35 :=
by
  have total_combinations := total_segments * (total_segments - 1) / 2
  have prob_sides := (sides * (sides - 1) / 2 : ℚ) / total_combinations
  have prob_diagonals := (diagonals * (diagonals - 1) / 2 : ℚ) / total_combinations
  have prob_sum := prob_sides + prob_diagonals
  have h_prob_sides : prob_sides = (6 * 5 / 2) / (15 * 14 / 2) :=
    by sorry
  have h_prob_diagonals : prob_diagonals = (9 * 8 / 2) / (15 * 14 / 2) :=
    by sorry
  calc
    prob_same_length = prob_sum : by sorry
    ... = 17 / 35 : by {rw [←h_prob_sides, ←h_prob_diagonals], sorry}

end probability_same_length_segments_of_regular_hexagon_l92_92142


namespace fraction_of_area_below_line_l92_92013

noncomputable def rectangle_area_fraction (x1 y1 x2 y2 : ℝ) (x3 y3 x4 y4 : ℝ) : ℝ :=
  let m := (y2 - y1) / (x2 - x1)
  let b := y1 - m * x1
  let y_intercept := b
  let base := x4 - x1
  let height := y4 - y3
  let triangle_area := 0.5 * base * height
  triangle_area / (base * height)

theorem fraction_of_area_below_line : 
  rectangle_area_fraction 1 3 5 1 1 0 5 4 = 1 / 8 := 
by
  sorry

end fraction_of_area_below_line_l92_92013


namespace no_valid_prime_angles_l92_92378

def is_prime (n : ℕ) : Prop := Prime n

theorem no_valid_prime_angles :
  ∀ (x : ℕ), (x < 30) ∧ is_prime x ∧ is_prime (3 * x) → False :=
by sorry

end no_valid_prime_angles_l92_92378


namespace probability_not_all_dice_show_different_l92_92060

noncomputable def probability_not_all_same : ℚ :=
  let total_outcomes := 8^5
  let same_number_outcomes := 8
  (total_outcomes - same_number_outcomes) / total_outcomes

theorem probability_not_all_dice_show_different : 
  probability_not_all_same = 4095 / 4096 := by
  sorry

end probability_not_all_dice_show_different_l92_92060


namespace total_carriages_l92_92709

theorem total_carriages (Euston Norfolk Norwich FlyingScotsman : ℕ) 
  (h1 : Euston = 130)
  (h2 : Norfolk = Euston - 20)
  (h3 : Norwich = 100)
  (h4 : FlyingScotsman = Norwich + 20) :
  Euston + Norfolk + Norwich + FlyingScotsman = 460 :=
by 
  sorry

end total_carriages_l92_92709


namespace tangent_line_through_point_l92_92621

theorem tangent_line_through_point (x y : ℝ) (tangent f : ℝ → ℝ) (M : ℝ × ℝ) :
  M = (1, 1) →
  f x = x^3 + 1 →
  tangent x = 3 * x^2 →
  (∃ a b c : ℝ, a * x + b * y + c = 0 ∧ ∀ x0 y0 : ℝ, (y0 = f x0) → (y - y0 = tangent x0 * (x - x0))) ∧
  (x, y) = M →
  (a = 0 ∧ b = 1 ∧ c = -1) ∨ (a = 27 ∧ b = -4 ∧ c = -23) :=
by
  sorry

end tangent_line_through_point_l92_92621


namespace factorization_identity_l92_92334

-- We are asked to prove the mathematical equality under given conditions.
theorem factorization_identity (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = (x^2 + 6*x + 8)^2 :=
by
  sorry

end factorization_identity_l92_92334


namespace complement_correct_l92_92929

universe u

-- We define sets A and B
def A : Set ℕ := {1, 2, 3, 4, 5}
def B : Set ℕ := {1, 3, 5}

-- Define the complement of B with respect to A
def complement (A B : Set ℕ) : Set ℕ := {x ∈ A | x ∉ B}

-- The theorem we need to prove
theorem complement_correct : complement A B = {2, 4} := 
  sorry

end complement_correct_l92_92929


namespace nate_total_run_l92_92838

def field_length := 168
def initial_run := 4 * field_length
def additional_run := 500
def total_run := initial_run + additional_run

theorem nate_total_run : total_run = 1172 := by
  sorry

end nate_total_run_l92_92838


namespace decrease_in_silver_coins_l92_92658

theorem decrease_in_silver_coins
  (a : ℕ) (h₁ : 2 * a = 3 * (50 - a))
  (h₂ : a + (50 - a) = 50) :
  (5 * (50 - a) - 3 * a = 10) :=
by
sorry

end decrease_in_silver_coins_l92_92658


namespace percentage_increase_l92_92484

theorem percentage_increase (total_capacity : ℝ) (additional_water : ℝ) (percentage_capacity : ℝ) (current_water : ℝ) : 
    additional_water + current_water = percentage_capacity * total_capacity →
    percentage_capacity = 0.70 →
    total_capacity = 1857.1428571428573 →
    additional_water = 300 →
    current_water = ((percentage_capacity * total_capacity) - additional_water) →
    (additional_water / current_water) * 100 = 30 :=
by
    sorry

end percentage_increase_l92_92484


namespace john_total_distance_l92_92874

theorem john_total_distance :
  let s₁ : ℝ := 45       -- Speed for the first part (mph)
  let t₁ : ℝ := 2        -- Time for the first part (hours)
  let s₂ : ℝ := 50       -- Speed for the second part (mph)
  let t₂ : ℝ := 3        -- Time for the second part (hours)
  let d₁ : ℝ := s₁ * t₁ -- Distance for the first part
  let d₂ : ℝ := s₂ * t₂ -- Distance for the second part
  d₁ + d₂ = 240          -- Total distance
:= by
  sorry

end john_total_distance_l92_92874


namespace factor_polynomial_l92_92328

theorem factor_polynomial :
  ∀ x : ℝ, 
  (x^2 + 4 * x + 3) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8) 
  = (x^2 + 6 * x + 1) * (x^2 + 6 * x + 37) :=
by
  intros x
  sorry

end factor_polynomial_l92_92328


namespace ratio_is_one_half_l92_92572

noncomputable def ratio_of_intercepts (b : ℝ) (hb : b ≠ 0) : ℝ :=
  let s := -b / 8
  let t := -b / 4
  s / t

theorem ratio_is_one_half (b : ℝ) (hb : b ≠ 0) :
  ratio_of_intercepts b hb = 1 / 2 :=
by
  sorry

end ratio_is_one_half_l92_92572


namespace if_a_eq_b_then_a_squared_eq_b_squared_l92_92440

theorem if_a_eq_b_then_a_squared_eq_b_squared (a b : ℝ) (h : a = b) : a^2 = b^2 :=
sorry

end if_a_eq_b_then_a_squared_eq_b_squared_l92_92440


namespace hexagon_probability_same_length_l92_92159

-- Define the problem
theorem hexagon_probability_same_length (T : Finset (Fin 15)) :
  let num_favorable_outcomes := 33
  let total_possible_outcomes := 105
  (num_favorable_outcomes / total_possible_outcomes) = (11 / 35) :=
by
  sorry

end hexagon_probability_same_length_l92_92159


namespace inequality_has_solutions_l92_92759

theorem inequality_has_solutions (a : ℝ) :
  (∃ x : ℝ, |x + 3| + |x - 1| < a^2 - 3 * a) ↔ (a < -1 ∨ 4 < a) := 
by
  sorry

end inequality_has_solutions_l92_92759


namespace max_p_pascal_distribution_l92_92531

open ProbabilityTheory

def pascalDistribution_prob (r x : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose (x - 1) (r - 1)) * p^r * (1 - p)^(x - r)

theorem max_p_pascal_distribution (p : ℝ) (h : 0 < p ∧ p < 1) :
  (pascalDistribution_prob 3 6 p) ≥ (pascalDistribution_prob 3 5 p) → p ≤ 2 / 5 := by
  sorry

end max_p_pascal_distribution_l92_92531


namespace sam_distance_traveled_l92_92812

-- Variables definition
variables (distance_marguerite : ℝ) (time_marguerite : ℝ) (time_sam : ℝ)

-- Given conditions
def marguerite_conditions : Prop :=
  distance_marguerite = 150 ∧
  time_marguerite = 3 ∧
  time_sam = 4

-- Statement to prove
theorem sam_distance_traveled (h : marguerite_conditions distance_marguerite time_marguerite time_sam) : 
  distance_marguerite / time_marguerite * time_sam = 200 :=
sorry

end sam_distance_traveled_l92_92812


namespace percentage_of_games_won_l92_92692

theorem percentage_of_games_won (games_won games_lost : ℕ) (h : games_won = 13 * games_lost / 7) : 
  (games_won : ℚ) / (games_won + games_lost) * 100 = 65 :=
by
  sorry

end percentage_of_games_won_l92_92692


namespace olivia_correct_answers_l92_92656

theorem olivia_correct_answers (c w : ℕ) 
  (h1 : c + w = 15) 
  (h2 : 6 * c - 3 * w = 45) : 
  c = 10 := 
  sorry

end olivia_correct_answers_l92_92656


namespace convert_to_scientific_notation_l92_92963

theorem convert_to_scientific_notation (N : ℕ) (h : 2184300000 = 2184.3 * 10^6) : 
    (2184300000 : ℝ) = 2.1843 * 10^7 :=
by 
  sorry

end convert_to_scientific_notation_l92_92963


namespace product_remainder_l92_92018

theorem product_remainder (a b c : ℕ) (h1 : a % 7 = 2) (h2 : b % 7 = 3) (h3 : c % 7 = 5) (h4 : (a + b + c) % 7 = 3) : 
  (a * b * c) % 7 = 2 := 
by sorry

end product_remainder_l92_92018


namespace ab_zero_l92_92110

theorem ab_zero (a b : ℝ)
  (h1 : a + b = 5)
  (h2 : a^3 + b^3 = 125) : 
  a * b = 0 :=
by
  sorry

end ab_zero_l92_92110


namespace largest_x_satisfies_eq_l92_92029

theorem largest_x_satisfies_eq (x : ℝ) (h : sqrt (3 * x) = 5 * x) : x ≤ 3 / 25 :=
sorry

end largest_x_satisfies_eq_l92_92029


namespace find_divisor_from_count_l92_92448

theorem find_divisor_from_count (h1 : ∃ n : ℕ, 11110 = n)
                                (h2 : ∀ k : ℕ, (10 ≤ k ∧ k ≤ 100000) → ∃ x : ℕ, k % x = 0)
                                (h3 : 10 % 9 = 0)
                                (h4 : 100000 % 9 = 0) :
  ∃ x : ℕ, (99990 / 11109) = x :=
by
  have gcd_99990_11109 : Int.gcd 99990 11109 = 9 := by sorry
  exact ⟨9, gcd_99990_11109⟩

end find_divisor_from_count_l92_92448


namespace alex_min_additional_coins_l92_92901

theorem alex_min_additional_coins (n m k : ℕ) (h_n : n = 15) (h_m : m = 120) :
  k = 0 ↔ m = (n * (n + 1)) / 2 :=
by
  sorry

end alex_min_additional_coins_l92_92901


namespace allocation_schemes_correct_l92_92627

def numWaysToAllocateVolunteers : ℕ :=
  let n := 5 -- number of volunteers
  let m := 4 -- number of events
  Nat.choose n 2 * Nat.factorial m

theorem allocation_schemes_correct :
  numWaysToAllocateVolunteers = 240 :=
by
  sorry

end allocation_schemes_correct_l92_92627


namespace vasya_has_more_fanta_l92_92967

-- Definitions based on the conditions:
def initial_fanta_vasya (a : ℝ) : ℝ := a
def initial_fanta_petya (a : ℝ) : ℝ := 1.1 * a
def remaining_fanta_vasya (a : ℝ) : ℝ := a * 0.98
def remaining_fanta_petya (a : ℝ) : ℝ := 1.1 * a * 0.89

-- The theorem to prove Vasya has more Fanta left than Petya.
theorem vasya_has_more_fanta (a : ℝ) (h : 0 < a) : remaining_fanta_vasya a > remaining_fanta_petya a := by
  sorry

end vasya_has_more_fanta_l92_92967


namespace factor_polynomial_l92_92255

theorem factor_polynomial (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = 
  (x^2 + 6*x + 4) * (x^2 + 6*x + 11) :=
by
  sorry

end factor_polynomial_l92_92255


namespace inequality_proof_l92_92638

open Real

theorem inequality_proof (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h_sum : a + b + c = 1) : 
  (a - b * c) / (a + b * c) + (b - c * a) / (b + c * a) + (c - a * b) / (c + a * b) ≤ 3 / 2 :=
sorry

end inequality_proof_l92_92638


namespace nate_total_distance_l92_92836

def length_field : ℕ := 168
def distance_8s : ℕ := 4 * length_field
def additional_distance : ℕ := 500
def total_distance : ℕ := distance_8s + additional_distance

theorem nate_total_distance : total_distance = 1172 := by
  sorry

end nate_total_distance_l92_92836


namespace binomial_variance_transformation_l92_92509

noncomputable def xi : ℝ → binomial 100 0.2 := sorry

lemma variance_linear_transformation (a b : ℝ) (hξ : random_variable ℝ xi)
  : D(a * ξ + b) = a^2 * D(ξ) := sorry

theorem binomial_variance_transformation :
  D(4*xi + 3) = 256 := by 
sorry

end binomial_variance_transformation_l92_92509


namespace edward_total_money_l92_92614

-- define the amounts made and spent
def money_made_spring : ℕ := 2
def money_made_summer : ℕ := 27
def money_spent_supplies : ℕ := 5

-- total money left is calculated by adding what he made and subtracting the expenses
def total_money_end (m_spring m_summer m_supplies : ℕ) : ℕ :=
  m_spring + m_summer - m_supplies

-- the theorem to prove
theorem edward_total_money :
  total_money_end money_made_spring money_made_summer money_spent_supplies = 24 :=
by
  sorry

end edward_total_money_l92_92614


namespace regular_hexagon_same_length_probability_l92_92167

theorem regular_hexagon_same_length_probability :
  let T : Finset (Finite_Field) := sorry, -- Define the actual set T una don the constructible geometry module in lean mathematical library
  let total_segments := 15,
  let sides := 6,
  let shorter_diagonals := 6,
  let longer_diagonals := 3,
  let prob_side := (6 / 15) * (5 / 14),
  let prob_shorter := (6 / 15) * (5 / 14),
  let prob_longer := (3 / 15) * (2 / 14),
  let total_prob := (prob_side + prob_shorter + prob_longer),
  (total_prob = (11 / 35)) := sorry

end regular_hexagon_same_length_probability_l92_92167


namespace collinear_dot_probability_computation_l92_92659

def collinear_dot_probability : ℚ := 12 / Nat.choose 25 5

theorem collinear_dot_probability_computation :
  collinear_dot_probability = 12 / 53130 :=
by
  -- This is where the proof steps would be if provided.
  sorry

end collinear_dot_probability_computation_l92_92659


namespace binom_eight_four_l92_92492

theorem binom_eight_four : (Nat.choose 8 4) = 70 :=
by
  sorry

end binom_eight_four_l92_92492


namespace probability_same_length_segments_l92_92149

theorem probability_same_length_segments (T : Finset ℕ) (sides : Finset ℕ) (diagonals : Finset ℕ) :
  sides.card = 6 →
  diagonals.card = 9 →
  (∃ d1 d2 : Finset ℕ, d1.card = 3 ∧ d2.card = 3 ∧ d1 ∩ d2 = ∅ ∧ diagonals = d1 ∪ d2) →
  T = sides ∪ diagonals →
  (∃ n : ℕ, (T.card.choose 2) = n * 5 ∧ ∃ f : ℕ, (∑ x in T, ∑ y in (T \ {x}), if x = y then 1 else 0) = f * 5) →
  ((∑ x in T, ∑ y in (T \ {x}), if x = y then 1 else 0).toRat / (T.card.choose 2).toRat = (1 : ℚ) / 5) :=
by
  sorry

end probability_same_length_segments_l92_92149


namespace general_term_formula_l92_92187

theorem general_term_formula :
  ∀ n : ℕ, (0 < n) → 
  (-1)^n * (2*n + 1) / (2*n) = ((-1) : ℝ)^n * ((2*n + 1) : ℝ) / (2*n) :=
by {
  sorry
}

end general_term_formula_l92_92187


namespace no_member_of_T_is_divisible_by_4_l92_92390

def sum_of_squares_of_four_consecutive_integers (n : ℤ) : ℤ :=
  (n-1)^2 + n^2 + (n+1)^2 + (n+2)^2

theorem no_member_of_T_is_divisible_by_4 : ∀ n : ℤ, ¬ (sum_of_squares_of_four_consecutive_integers n % 4 = 0) := by
  intro n
  sorry

end no_member_of_T_is_divisible_by_4_l92_92390


namespace hyperbola_eccentricity_proof_l92_92105

noncomputable def hyperbola_eccentricity (a b k1 k2 : ℝ) (ha : 0 < a) (hb : 0 < b) 
    (C_on_hyperbola : ∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1) 
    (slope_condition : k1 * k2 = b^2 / a^2)
    (minimized_expr : ∀ k1 k2: ℝ , (2 / (k1 * k2)) + Real.log k1 + Real.log k2 ≥ (2 / (b^2 / a^2)) + Real.log ((b^2 / a^2))) : 
    ℝ :=
  Real.sqrt (1 + (b^2 / a^2))

theorem hyperbola_eccentricity_proof (a b k1 k2 : ℝ) (ha : 0 < a) (hb : 0 < b) 
    (C_on_hyperbola : ∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1) 
    (slope_condition : k1 * k2 = b^2 / a^2)
    (minimized_expr : ∀ k1 k2: ℝ , (2 / (k1 * k2)) + Real.log k1 + Real.log k2 ≥ (2 / (b^2 / a^2)) + Real.log ((b^2 / a^2))) :    
    hyperbola_eccentricity a b k1 k2 ha hb C_on_hyperbola slope_condition minimized_expr = Real.sqrt 3 :=
sorry

end hyperbola_eccentricity_proof_l92_92105


namespace sam_drove_distance_l92_92816

theorem sam_drove_distance (marguerite_distance : ℝ) (marguerite_time : ℝ) (sam_time : ℝ) :
  marguerite_distance = 150 ∧ marguerite_time = 3 ∧ sam_time = 4 →
  (sam_time * (marguerite_distance / marguerite_time) = 200) :=
by
  sorry

end sam_drove_distance_l92_92816


namespace ab_is_zero_l92_92113

theorem ab_is_zero (a b : ℝ) (h₁ : a + b = 5) (h₂ : a^3 + b^3 = 125) : a * b = 0 :=
by
  -- Begin proof here
  sorry

end ab_is_zero_l92_92113


namespace five_eight_sided_dice_not_all_same_l92_92058

noncomputable def probability_not_all_same : ℚ :=
  let total_outcomes := 8^5
  let same_number_outcomes := 8
  1 - (same_number_outcomes / total_outcomes)

theorem five_eight_sided_dice_not_all_same :
  probability_not_all_same = 4095 / 4096 :=
by
  sorry

end five_eight_sided_dice_not_all_same_l92_92058


namespace needle_intersection_probability_l92_92014

noncomputable def needle_probability (a l : ℝ) (h : l < a) : ℝ :=
  (2 * l) / (a * Real.pi)

theorem needle_intersection_probability (a l : ℝ) (h : l < a) :
  needle_probability a l h = 2 * l / (a * Real.pi) :=
by
  -- This is the statement to be proved
  sorry

end needle_intersection_probability_l92_92014


namespace exponent_tower_divisibility_l92_92972

theorem exponent_tower_divisibility (h1 h2 : ℕ) (Hh1 : h1 ≥ 3) (Hh2 : h2 ≥ 3) : 
  (2 ^ (5 ^ (2 ^ (5 ^ h1))) + 4 ^ (5 ^ (4 ^ (5 ^ h2)))) % 2008 = 0 := by
  sorry

end exponent_tower_divisibility_l92_92972


namespace cos_arcsin_l92_92743

theorem cos_arcsin (x : ℝ) (h : x = 3/5) : Real.cos (Real.arcsin x) = 4/5 := 
by
  rw h
  sorry

end cos_arcsin_l92_92743


namespace sunland_more_plates_than_moonland_l92_92801

theorem sunland_more_plates_than_moonland :
  let sunland_plates := 26^5 * 10^2
  let moonland_plates := 26^3 * 10^3
  sunland_plates - moonland_plates = 1170561600 := by
  sorry

end sunland_more_plates_than_moonland_l92_92801


namespace find_m_l92_92363

theorem find_m (a b m : ℤ) (h1 : a - b = 6) (h2 : a + b = 0) : 2 * a + b = m → m = 3 :=
by
  sorry

end find_m_l92_92363


namespace bettys_herb_garden_l92_92604

theorem bettys_herb_garden :
  ∀ (basil oregano thyme rosemary total : ℕ),
    oregano = 2 * basil + 2 →
    thyme = 3 * basil - 3 →
    rosemary = (basil + thyme) / 2 →
    basil = 5 →
    total = basil + oregano + thyme + rosemary →
    total ≤ 50 →
    total = 37 :=
by
  intros basil oregano thyme rosemary total h_oregano h_thyme h_rosemary h_basil h_total h_le_total
  sorry

end bettys_herb_garden_l92_92604


namespace calculate_L_l92_92251

theorem calculate_L (T H K : ℝ) (hT : T = 2 * Real.sqrt 5) (hH : H = 10) (hK : K = 2) :
  L = 100 :=
by
  let L := 50 * T^4 / (H^2 * K)
  have : T = 2 * Real.sqrt 5 := hT
  have : H = 10 := hH
  have : K = 2 := hK
  sorry

end calculate_L_l92_92251


namespace allocation_schemes_correct_l92_92628

def numWaysToAllocateVolunteers : ℕ :=
  let n := 5 -- number of volunteers
  let m := 4 -- number of events
  Nat.choose n 2 * Nat.factorial m

theorem allocation_schemes_correct :
  numWaysToAllocateVolunteers = 240 :=
by
  sorry

end allocation_schemes_correct_l92_92628


namespace ordered_pairs_sum_reciprocal_l92_92518

theorem ordered_pairs_sum_reciprocal (a b : ℕ) (h1 : a > 0) (h2 : b > 0) :
  (∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ (1 / a + 1 / b : ℚ) = 1 / 6) → ∃ n : ℕ, n = 9 :=
by
  sorry

end ordered_pairs_sum_reciprocal_l92_92518


namespace sequence_relation_l92_92172

theorem sequence_relation
  (a b : ℝ)
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11) :
  a^10 + b^10 = 123 :=
sorry

end sequence_relation_l92_92172


namespace triangle_min_diff_l92_92858

variable (XY YZ XZ : ℕ) -- Declaring the side lengths as natural numbers

theorem triangle_min_diff (h1 : XY < YZ ∧ YZ ≤ XZ) -- Condition for side length relations
  (h2 : XY + YZ + XZ = 2010) -- Condition for the perimeter
  (h3 : XY + YZ > XZ)
  (h4 : XY + XZ > YZ)
  (h5 : YZ + XZ > XY) :
  (YZ - XY) = 1 := -- Statement that the smallest possible value of YZ - XY is 1
sorry

end triangle_min_diff_l92_92858


namespace factor_polynomial_l92_92327

theorem factor_polynomial :
  ∀ x : ℝ, 
  (x^2 + 4 * x + 3) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8) 
  = (x^2 + 6 * x + 1) * (x^2 + 6 * x + 37) :=
by
  intros x
  sorry

end factor_polynomial_l92_92327


namespace sqrt_a_minus_2_meaningful_l92_92365

theorem sqrt_a_minus_2_meaningful (a : ℝ) (h : 0 ≤ a - 2) : 2 ≤ a :=
by
  sorry

end sqrt_a_minus_2_meaningful_l92_92365


namespace geometric_series_sum_l92_92345

theorem geometric_series_sum (n : ℕ) : 
  let a₁ := 2
  let q := 2
  let S_n := a₁ * (1 - q^n) / (1 - q)
  S_n = 2 - 2^(n + 1) := 
by
  sorry

end geometric_series_sum_l92_92345


namespace probability_not_all_same_l92_92052

theorem probability_not_all_same :
  (let total_outcomes := 8 ^ 5 in
   let same_number_outcomes := 8 in
   let probability_all_same := same_number_outcomes / total_outcomes in
   let probability_not_same := 1 - probability_all_same in
   probability_not_same = 4095 / 4096) :=
begin
  sorry
end

end probability_not_all_same_l92_92052


namespace find_x_minus_y_l92_92851

def rotated_point (x y h k : ℝ) : ℝ × ℝ := (2 * h - x, 2 * k - y)

def reflected_point (x y : ℝ) : ℝ × ℝ := (y, x)

def transformed_point (x y : ℝ) : ℝ × ℝ :=
  reflected_point (rotated_point x y 2 3).1 (rotated_point x y 2 3).2

theorem find_x_minus_y (x y : ℝ) (h1 : transformed_point x y = (4, -1)) : x - y = 3 := 
by 
  sorry

end find_x_minus_y_l92_92851


namespace village_population_l92_92587

theorem village_population (P : ℕ) (h : 80 * P = 32000 * 100) : P = 40000 :=
sorry

end village_population_l92_92587


namespace factor_polynomial_l92_92281

-- Define the necessary polynomials
def p (x : ℝ) : ℝ := x^2 + 4*x + 3
def q (x : ℝ) : ℝ := x^2 + 8*x + 15
def r (x : ℝ) : ℝ := x^2 + 6*x - 8

-- State the main theorem
theorem factor_polynomial (x : ℝ) : 
  (p x * q x) + r x = (x^2 + 7*x + 1) * (x^2 + 4*x + 7) :=
by
  sorry

end factor_polynomial_l92_92281


namespace sam_drove_200_miles_l92_92826

-- Define the conditions
def marguerite_distance : ℕ := 150
def marguerite_time : ℕ := 3
def sam_time : ℕ := 4
def marguerite_speed := marguerite_distance / marguerite_time

-- Define the question
def sam_distance (speed : ℕ) (time : ℕ) : ℕ := speed * time

-- State the theorem to prove the answer
theorem sam_drove_200_miles :
  sam_distance marguerite_speed sam_time = 200 := by
  sorry

end sam_drove_200_miles_l92_92826


namespace range_of_a_l92_92519

theorem range_of_a 
  (a : ℝ)
  (H1 : ∀ x : ℝ, -2 < x ∧ x < 3 → -2 < x ∧ x < a)
  (H2 : ¬(∀ x : ℝ, -2 < x ∧ x < a → -2 < x ∧ x < 3)) :
  3 < a :=
by
  sorry

end range_of_a_l92_92519


namespace ants_species_A_count_l92_92904

theorem ants_species_A_count (a b : ℕ) (h1 : a + b = 30) (h2 : 2^5 * a + 3^5 * b = 3281) : 32 * a = 608 :=
by
  sorry

end ants_species_A_count_l92_92904


namespace within_acceptable_range_l92_92084

def flour_weight : ℝ := 25.18
def flour_label : ℝ := 25
def tolerance : ℝ := 0.25

theorem within_acceptable_range  :
  (flour_label - tolerance) ≤ flour_weight ∧ flour_weight ≤ (flour_label + tolerance) :=
by
  sorry

end within_acceptable_range_l92_92084


namespace find_second_dimension_l92_92227

variable (l h w : ℕ)
variable (cost_per_sqft total_cost : ℕ)
variable (surface_area : ℕ)

def insulation_problem_conditions (l : ℕ) (h : ℕ) (cost_per_sqft : ℕ) (total_cost : ℕ) (w : ℕ) (surface_area : ℕ) : Prop :=
  l = 4 ∧ h = 3 ∧ cost_per_sqft = 20 ∧ total_cost = 1880 ∧ surface_area = (2 * l * w + 2 * l * h + 2 * w * h)

theorem find_second_dimension (l h w : ℕ) (cost_per_sqft total_cost surface_area : ℕ) :
  insulation_problem_conditions l h cost_per_sqft total_cost w surface_area →
  surface_area = 94 →
  w = 5 :=
by
  intros
  simp [insulation_problem_conditions] at *
  sorry

end find_second_dimension_l92_92227


namespace base7_subtraction_correct_l92_92605

-- Define a function converting base 7 number to base 10
def base7_to_base10 (n : Nat) : Nat :=
  let d0 := n % 10
  let n1 := n / 10
  let d1 := n1 % 10
  let n2 := n1 / 10
  let d2 := n2 % 10
  let n3 := n2 / 10
  let d3 := n3 % 10
  d3 * 7^3 + d2 * 7^2 + d1 * 7^1 + d0 * 7^0

-- Define the numbers in base 7
def a : Nat := 2456
def b : Nat := 1234

-- Define the expected result in base 7
def result_base7 : Nat := 1222

-- State the theorem: The difference of a and b in base 7 should equal result_base7
theorem base7_subtraction_correct :
  let diff_base10 := (base7_to_base10 a) - (base7_to_base10 b)
  let result_base10 := base7_to_base10 result_base7
  diff_base10 = result_base10 :=
by
  sorry

end base7_subtraction_correct_l92_92605


namespace parabola_directrix_eq_neg_2_l92_92618

noncomputable def parabola_directrix (a b c : ℝ) : ℝ :=
  (b^2 - 4 * a * c) / (4 * a)

theorem parabola_directrix_eq_neg_2 (x : ℝ) :
  parabola_directrix 1 (-4) 4 = -2 :=
by
  -- proof steps go here
  sorry

end parabola_directrix_eq_neg_2_l92_92618


namespace matrix_sum_correct_l92_92505

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℚ :=
  ![
    [2 / 3, -1 / 2],
    [4, -5 / 2]
  ]

def B : Matrix (Fin 2) (Fin 2) ℚ :=
  ![
    [-5 / 6, 1 / 4],
    [3 / 2, -7 / 4]
  ]

def C : Matrix (Fin 2) (Fin 2) ℚ :=
  ![
    [-1 / 6, -1 / 4],
    [11 / 2, -17 / 4]
  ]

theorem matrix_sum_correct : (A + B) = C :=
by
  sorry

end matrix_sum_correct_l92_92505


namespace probability_at_least_9_heads_in_12_flips_l92_92711

theorem probability_at_least_9_heads_in_12_flips :
  let total_outcomes := 2^12
  let favorable_outcomes := (Nat.choose 12 9) + (Nat.choose 12 10) + (Nat.choose 12 11) + (Nat.choose 12 12)
  let probability := favorable_outcomes / total_outcomes
  probability = 299 / 4096 := 
by
  sorry

end probability_at_least_9_heads_in_12_flips_l92_92711


namespace cards_per_page_l92_92462

theorem cards_per_page 
  (new_cards old_cards pages : ℕ) 
  (h1 : new_cards = 8)
  (h2 : old_cards = 10)
  (h3 : pages = 6) : (new_cards + old_cards) / pages = 3 := 
by 
  sorry

end cards_per_page_l92_92462


namespace S_15_eq_1695_l92_92647

open Nat

/-- Sum of the nth set described in the problem -/
def S (n : ℕ) : ℕ :=
  let first := 1 + (n * (n - 1)) / 2
  let last := first + n - 1
  (n * (first + last)) / 2

theorem S_15_eq_1695 : S 15 = 1695 :=
by
  sorry

end S_15_eq_1695_l92_92647


namespace range_of_f_l92_92516

noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then 2^x else -Real.log x / Real.log 2

theorem range_of_f : Set.Iic 2 = Set.range f :=
  by sorry

end range_of_f_l92_92516


namespace expression_evaluation_l92_92996

theorem expression_evaluation (a b : ℕ) (h1 : a = 25) (h2 : b = 15) : (a + b)^2 - (a^2 + b^2) = 750 :=
by
  sorry

end expression_evaluation_l92_92996


namespace function_is_zero_l92_92795

variable (n : ℕ) (a : Fin n → ℤ) (f : ℤ → ℝ)

axiom condition : ∀ (k l : ℤ), l ≠ 0 → (Finset.univ.sum (λ i => f (k + a i * l)) = 0)

theorem function_is_zero : ∀ x : ℤ, f x = 0 := by
  sorry

end function_is_zero_l92_92795


namespace find_pairs_l92_92917

theorem find_pairs (m n : ℕ) :
  (m + 1) % n = 0 ∧ (n^2 - n + 1) % m = 0 ↔
  (m = 1 ∧ n = 1) ∨ (m = 1 ∧ n = 2) ∨ (m = 3 ∧ n = 2) := 
by
  sorry

end find_pairs_l92_92917


namespace plane_centroid_l92_92391

theorem plane_centroid (a b : ℝ) (h : 1 / a ^ 2 + 1 / b ^ 2 + 1 / 25 = 1 / 4) :
  let p := a / 3
  let q := b / 3
  let r := 5 / 3
  (1 / p ^ 2 + 1 / q ^ 2 + 1 / r ^ 2) = 369 / 400 :=
by
  sorry

end plane_centroid_l92_92391


namespace problem_1_l92_92472

noncomputable def f (a x : ℝ) : ℝ := abs (x + 2) + abs (x - a)

theorem problem_1 (a : ℝ) (h : ∀ x : ℝ, f a x ≥ 3) : a ≤ -5 ∨ a ≥ 1 :=
by
  sorry

end problem_1_l92_92472


namespace nylon_needed_is_192_l92_92953

-- Define the required lengths for the collars
def nylon_needed_for_dog_collar : ℕ := 18
def nylon_needed_for_cat_collar : ℕ := 10

-- Define the number of collars needed
def number_of_dog_collars : ℕ := 9
def number_of_cat_collars : ℕ := 3

-- Define the total nylon needed
def total_nylon_needed : ℕ :=
  (nylon_needed_for_dog_collar * number_of_dog_collars) + (nylon_needed_for_cat_collar * number_of_cat_collars)

-- State the theorem we need to prove
theorem nylon_needed_is_192 : total_nylon_needed = 192 := 
  by
    -- Simplification to match the complete statement for completeness
    sorry

end nylon_needed_is_192_l92_92953


namespace circle_equation_tangent_line1_tangent_line2_l92_92118

-- Definitions of points
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (1, 1)
def C : ℝ × ℝ := (2, 0)
def P : ℝ × ℝ := (2, 3)

-- Equation for the circle given the point constraints
def circle_eq : Prop := 
  ∀ x y : ℝ, ((x - 1)^2 + y^2 = 1) ↔ ((x = 0 ∧ y = 0) ∨ (x = 1 ∧ y = 1) ∨ (x = 2 ∧ y = 0))

-- Equations for the tangent lines passing through point P and tangent to the circle
def tangent_eq1 : Prop := 
  P.1 = 2

def tangent_eq2 : Prop :=
  4 * P.1 - 3 * P.2 + 1 = 0

-- Statements to be proven
theorem circle_equation : circle_eq := 
  sorry 

theorem tangent_line1 : tangent_eq1 := 
  sorry 

theorem tangent_line2 : tangent_eq2 := 
  sorry 

end circle_equation_tangent_line1_tangent_line2_l92_92118


namespace basic_computer_price_l92_92198

theorem basic_computer_price (C P : ℝ) 
  (h1 : C + P = 2500)
  (h2 : P = 1 / 8 * ((C + 500) + P)) :
  C = 2125 :=
by
  sorry

end basic_computer_price_l92_92198


namespace additional_rate_of_interest_l92_92678

variable (P A A' : ℝ) (T : ℕ) (R : ℝ)

-- Conditions
def principal_amount := (P = 8000)
def original_amount := (A = 9200)
def time_period := (T = 3)
def new_amount := (A' = 9440)

-- The Lean statement to prove the additional percentage of interest
theorem additional_rate_of_interest  (P A A' : ℝ) (T : ℕ) (R : ℝ)
    (h1 : principal_amount P)
    (h2 : original_amount A)
    (h3 : time_period T)
    (h4 : new_amount A') :
    (A' - P) / (P * T) * 100 - (A - P) / (P * T) * 100 = 1 :=
by
  sorry

end additional_rate_of_interest_l92_92678


namespace probability_not_all_same_l92_92051

theorem probability_not_all_same (n : ℕ) (h : n = 5) : 
  let total_outcomes := 8^n in
  let same_number_outcomes := 8 in
  let prob_all_same_number := (same_number_outcomes : ℚ) / total_outcomes in
  let prob_not_all_same_number := 1 - prob_all_same_number in
  prob_not_all_same_number = 4095 / 4096 :=
by 
  sorry

end probability_not_all_same_l92_92051


namespace bill_score_l92_92174

theorem bill_score (B J S E : ℕ)
                   (h1 : B = J + 20)
                   (h2 : B = S / 2)
                   (h3 : E = B + J - 10)
                   (h4 : B + J + S + E = 250) :
                   B = 50 := 
by sorry

end bill_score_l92_92174


namespace weight_of_each_hardcover_book_l92_92140

theorem weight_of_each_hardcover_book
  (weight_limit : ℕ := 80)
  (hardcover_books : ℕ := 70)
  (textbooks : ℕ := 30)
  (knick_knacks : ℕ := 3)
  (textbook_weight : ℕ := 2)
  (knick_knack_weight : ℕ := 6)
  (over_weight : ℕ := 33)
  (total_weight : ℕ := hardcover_books * x + textbooks * textbook_weight + knick_knacks * knick_knack_weight)
  (weight_eq : total_weight = weight_limit + over_weight) :
  x = 1 / 2 :=
by {
  sorry
}

end weight_of_each_hardcover_book_l92_92140


namespace blue_red_area_ratio_l92_92242

theorem blue_red_area_ratio (d1 d2 : ℝ) (h1 : d1 = 2) (h2 : d2 = 6) :
  let r1 := d1 / 2
  let r2 := d2 / 2
  let a_red := π * r1^2
  let a_large := π * r2^2
  let a_blue := a_large - a_red
  a_blue / a_red = 8 :=
by
  have r1 := d1 / 2
  have r2 := d2 / 2
  have a_red := π * r1^2
  have a_large := π * r2^2
  have a_blue := a_large - a_red
  sorry

end blue_red_area_ratio_l92_92242


namespace linear_func_passing_point_l92_92353

theorem linear_func_passing_point :
  ∃ k : ℝ, ∀ x y : ℝ, (y = k * x + 1) → (x = -1 ∧ y = 0) → k = 1 :=
by
  sorry

end linear_func_passing_point_l92_92353


namespace sam_driving_distance_l92_92805

-- Definitions based on the conditions
def marguerite_distance : ℝ := 150
def marguerite_time : ℝ := 3
def sam_time : ℝ := 4

-- Desired statement using the given conditions
theorem sam_driving_distance :
  let rate := marguerite_distance / marguerite_time in
  let sam_distance := rate * sam_time in
  sam_distance = 200 :=
by
  sorry

end sam_driving_distance_l92_92805


namespace contrapositive_example_l92_92428

theorem contrapositive_example (x : ℝ) :
  (x < -1 ∨ x ≥ 1) → (x^2 ≥ 1) :=
sorry

end contrapositive_example_l92_92428


namespace probability_same_length_segments_l92_92150

theorem probability_same_length_segments (T : Finset ℕ) (sides : Finset ℕ) (diagonals : Finset ℕ) :
  sides.card = 6 →
  diagonals.card = 9 →
  (∃ d1 d2 : Finset ℕ, d1.card = 3 ∧ d2.card = 3 ∧ d1 ∩ d2 = ∅ ∧ diagonals = d1 ∪ d2) →
  T = sides ∪ diagonals →
  (∃ n : ℕ, (T.card.choose 2) = n * 5 ∧ ∃ f : ℕ, (∑ x in T, ∑ y in (T \ {x}), if x = y then 1 else 0) = f * 5) →
  ((∑ x in T, ∑ y in (T \ {x}), if x = y then 1 else 0).toRat / (T.card.choose 2).toRat = (1 : ℚ) / 5) :=
by
  sorry

end probability_same_length_segments_l92_92150


namespace find_x_l92_92212

theorem find_x : ∃ x : ℤ, x + 3 * 10 = 33 → x = 3 := by
  sorry

end find_x_l92_92212


namespace solve_custom_operation_l92_92844

theorem solve_custom_operation (x : ℤ) (h : ((4 * 3 - (12 - x)) = 2)) : x = -2 :=
by
  sorry

end solve_custom_operation_l92_92844


namespace overall_percent_decrease_l92_92955

theorem overall_percent_decrease (trouser_price_italy : ℝ) (jacket_price_italy : ℝ) 
(trouser_price_uk : ℝ) (trouser_discount_uk : ℝ) (jacket_price_uk : ℝ) 
(jacket_discount_uk : ℝ) (exchange_rate : ℝ) 
(h1 : trouser_price_italy = 200) (h2 : jacket_price_italy = 150) 
(h3 : trouser_price_uk = 150) (h4 : trouser_discount_uk = 0.20) 
(h5 : jacket_price_uk = 120) (h6 : jacket_discount_uk = 0.30) 
(h7 : exchange_rate = 0.85) : 
((trouser_price_italy + jacket_price_italy) - 
 ((trouser_price_uk * (1 - trouser_discount_uk) / exchange_rate) + 
 (jacket_price_uk * (1 - jacket_discount_uk) / exchange_rate))) / 
 (trouser_price_italy + jacket_price_italy) * 100 = 31.43 := 
by 
  sorry

end overall_percent_decrease_l92_92955


namespace solve_m_value_l92_92188

-- Definitions for conditions
def hyperbola_eq (m : ℝ) : Prop := ∀ x y : ℝ, 3 * m * x^2 - m * y^2 = 3
def has_focus (m : ℝ) : Prop := (∃ f1 f2 : ℝ, f1 = 0 ∧ f2 = 2)

-- Statement of the problem to prove
theorem solve_m_value (m : ℝ) (h_eq : hyperbola_eq m) (h_focus : has_focus m) : m = -1 :=
sorry

end solve_m_value_l92_92188


namespace binomial_sum_mod_eq_l92_92219

open Nat

theorem binomial_sum_mod_eq :
  (∑ k in range 11, (Nat.choose 11 k) * 6^(11 - k)) % 8 = 5 := 
by
  sorry

end binomial_sum_mod_eq_l92_92219


namespace find_x_l92_92652

theorem find_x (x y : ℝ) (hx : 0 < x) (hy : 0 < y) 
(h_eq : 7 * x^3 + 14 * x^2 * y = x^4 + 2 * x^3 * y) :
  x = 7 :=
by
  sorry

end find_x_l92_92652


namespace no_non_square_number_with_triple_product_divisors_l92_92213

theorem no_non_square_number_with_triple_product_divisors (N : ℕ) (h_non_square : ∀ k : ℕ, k * k ≠ N) : 
  ¬ (∃ t : ℕ, ∃ d : Finset (Finset ℕ), (∀ s ∈ d, s.card = 3) ∧ (∀ s ∈ d, s.prod id = t)) := 
sorry

end no_non_square_number_with_triple_product_divisors_l92_92213


namespace remainder_sum_first_150_div_11300_l92_92205

theorem remainder_sum_first_150_div_11300 :
  let n := 150
  let S := n * (n + 1) / 2
  S % 11300 = 25 :=
by
  let n := 150
  let S := n * (n + 1) / 2
  show S % 11300 = 25
  sorry

end remainder_sum_first_150_div_11300_l92_92205


namespace factor_poly_eq_factored_form_l92_92269

-- Defining the polynomial expressions
def poly1 (x : ℝ) := x^2 + 4 * x + 3
def poly2 (x : ℝ) := x^2 + 8 * x + 15
def poly3 (x : ℝ) := x^2 + 6 * x - 8

-- The main expression which needs to be factored
def main_expr (x : ℝ) := (poly1 x) * (poly2 x) + (poly3 x)

-- Stating the goal factored form
def factored_form (x : ℝ) := (x^2 + 6 * x + 19) * (x^2 + 6 * x - 2)

-- The theorem statement
theorem factor_poly_eq_factored_form (x : ℝ) : 
  main_expr x = factored_form x := 
by
  sorry

end factor_poly_eq_factored_form_l92_92269


namespace cube_painting_l92_92233

theorem cube_painting (n : ℕ) (h₁ : n > 4) 
  (h₂ : (2 * (n - 2)) = (n^2 - 2*n + 1)) : n = 5 :=
sorry

end cube_painting_l92_92233


namespace factor_polynomial_l92_92285

-- Define the necessary polynomials
def p (x : ℝ) : ℝ := x^2 + 4*x + 3
def q (x : ℝ) : ℝ := x^2 + 8*x + 15
def r (x : ℝ) : ℝ := x^2 + 6*x - 8

-- State the main theorem
theorem factor_polynomial (x : ℝ) : 
  (p x * q x) + r x = (x^2 + 7*x + 1) * (x^2 + 4*x + 7) :=
by
  sorry

end factor_polynomial_l92_92285


namespace vectors_coplanar_l92_92470

def vector_a : ℝ × ℝ × ℝ := (3, 2, 1)
def vector_b : ℝ × ℝ × ℝ := (1, -3, -7)
def vector_c : ℝ × ℝ × ℝ := (1, 2, 3)

def scalar_triple_product (a b c : ℝ × ℝ × ℝ) : ℝ :=
  let (a1, a2, a3) := a
  let (b1, b2, b3) := b
  let (c1, c2, c3) := c
  a1 * (b2 * c3 - b3 * c2) - a2 * (b1 * c3 - b3 * c1) + a3 * (b1 * c2 - b2 * c1)

theorem vectors_coplanar : scalar_triple_product vector_a vector_b vector_c = 0 := 
by
  sorry

end vectors_coplanar_l92_92470


namespace sum_of_first_60_digits_l92_92864

-- Define the repeating sequence and the number of repetitions
def repeating_sequence : List ℕ := [0, 0, 0, 1]
def repetitions : ℕ := 15

-- Define the sum of first n elements of a repeating sequence
def sum_repeating_sequence (seq : List ℕ) (n : ℕ) : ℕ :=
  let len := seq.length
  let complete_cycles := n / len
  let remaining_digits := n % len
  let sum_complete_cycles := complete_cycles * seq.sum
  let sum_remaining_digits := (seq.take remaining_digits).sum
  sum_complete_cycles + sum_remaining_digits

-- Prove the specific case for 60 digits
theorem sum_of_first_60_digits : sum_repeating_sequence repeating_sequence 60 = 15 := 
by
  sorry

end sum_of_first_60_digits_l92_92864


namespace nate_total_run_l92_92837

def field_length := 168
def initial_run := 4 * field_length
def additional_run := 500
def total_run := initial_run + additional_run

theorem nate_total_run : total_run = 1172 := by
  sorry

end nate_total_run_l92_92837


namespace ninggao_intercity_project_cost_in_scientific_notation_l92_92694

theorem ninggao_intercity_project_cost_in_scientific_notation :
  let length_kilometers := 55
  let cost_per_kilometer_million := 140
  let total_cost_million := length_kilometers * cost_per_kilometer_million
  let total_cost_scientific := 7.7 * 10^6
  total_cost_million = total_cost_scientific := 
  sorry

end ninggao_intercity_project_cost_in_scientific_notation_l92_92694


namespace range_of_m_l92_92121

-- Definitions and the main problem statement
def f (m x : ℝ) : ℝ := m * x^2 - m * x - 1

theorem range_of_m (m : ℝ) : (∀ x : ℝ, f m x < 0) ↔ (-4 < m ∧ m ≤ 0) :=
by
  sorry

end range_of_m_l92_92121


namespace percentage_B_D_l92_92102

variables (A B C D : ℝ)

-- Conditions as hypotheses
theorem percentage_B_D
  (h1 : B = 1.71 * A)
  (h2 : C = 1.80 * A)
  (h3 : D = 1.90 * B)
  (h4 : B = 1.62 * C)
  (h5 : A = 0.65 * D)
  (h6 : C = 0.55 * D) : 
  B = 1.1115 * D :=
sorry

end percentage_B_D_l92_92102


namespace determine_p_l92_92137

theorem determine_p (p x1 x2 : ℝ) 
  (h_eq : ∀ x, x^2 + p * x + 3 = 0)
  (h_root_relation : x2 = 3 * x1)
  (h_vieta1 : x1 + x2 = -p)
  (h_vieta2 : x1 * x2 = 3) :
  p = 4 ∨ p = -4 := 
sorry

end determine_p_l92_92137


namespace sum_of_digits_of_gcd_l92_92210

def gcd_of_differences : ℕ := Int.gcd (Int.gcd 3360 2240) 5600

def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |>.sum

theorem sum_of_digits_of_gcd :
  sum_of_digits gcd_of_differences = 4 :=
by
  sorry

end sum_of_digits_of_gcd_l92_92210


namespace min_packs_needed_l92_92973

-- Define pack sizes
def pack_sizes : List ℕ := [6, 12, 24, 30]

-- Define the total number of cans needed
def total_cans : ℕ := 150

-- Define the minimum number of packs needed to buy exactly 150 cans of soda
theorem min_packs_needed : ∃ packs : List ℕ, (∀ p ∈ packs, p ∈ pack_sizes) ∧ List.sum packs = total_cans ∧ packs.length = 5 := by
  sorry

end min_packs_needed_l92_92973


namespace five_eight_sided_dice_not_all_same_l92_92057

noncomputable def probability_not_all_same : ℚ :=
  let total_outcomes := 8^5
  let same_number_outcomes := 8
  1 - (same_number_outcomes / total_outcomes)

theorem five_eight_sided_dice_not_all_same :
  probability_not_all_same = 4095 / 4096 :=
by
  sorry

end five_eight_sided_dice_not_all_same_l92_92057


namespace sum_of_x_coords_l92_92760

theorem sum_of_x_coords (x : ℝ) (y : ℝ) :
  y = abs (x^2 - 6*x + 8) ∧ y = 6 - x → (x = (5 + Real.sqrt 17) / 2 ∨ x = (5 - Real.sqrt 17) / 2 ∨ x = 2)
  →  ((5 + Real.sqrt 17) / 2 + (5 - Real.sqrt 17) / 2 + 2 = 7) :=
by
  intros h1 h2
  have H : ((5 + Real.sqrt 17) / 2 + (5 - Real.sqrt 17) / 2 + 2 = 7) := sorry
  exact H

end sum_of_x_coords_l92_92760


namespace max_metro_speed_l92_92136

variable (R S v : ℝ)

theorem max_metro_speed (h1 : S > 0) (h2 : R > 0)
    (yegor : 12 * (S / 24) > S / v)
    (nikita : 6 * (S / 2) < (S + R) / v) :
    v ≤ 23 :=
  sorry

end max_metro_speed_l92_92136


namespace fraction_value_l92_92496

def op_at (a b : ℤ) : ℤ := a * b - b ^ 2
def op_sharp (a b : ℤ) : ℤ := a + b - a * b ^ 2

theorem fraction_value : (op_at 7 3) / (op_sharp 7 3) = -12 / 53 :=
by
  sorry

end fraction_value_l92_92496


namespace min_A_cardinality_l92_92642

theorem min_A_cardinality {m a b : ℕ} (H : Nat.gcd a b = 1) (A : Set ℕ) (non_empty : A ≠ ∅) 
  (Ha : ∀ n : ℕ, n > 0 → a * n ∈ A ∨ b * n ∈ A) :
  ∃ c, c = max a b ∧
  min_value_of (A ∩ {x | x ∈ Finset.range (m + 1)}).card =
    if a = 1 ∧ b = 1 then m
    else ∑ i in Finset.range (m + 1), (-1) ^ (i + 1) * ⌊m / c ^ i⌋ :=
sorry

end min_A_cardinality_l92_92642


namespace factor_polynomial_l92_92262

theorem factor_polynomial (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = 
  (x^2 + 6*x + 4) * (x^2 + 6*x + 11) :=
by
  sorry

end factor_polynomial_l92_92262


namespace even_increasing_function_inequality_l92_92717

theorem even_increasing_function_inequality
  (f : ℝ → ℝ)
  (h_even : ∀ x : ℝ, f x = f (-x))
  (h_increasing : ∀ {x₁ x₂ : ℝ}, x₁ < x₂ ∧ x₂ < 0 → f x₁ < f x₂) :
  f 3 < f (-2) ∧ f (-2) < f 1 := by
  sorry

end even_increasing_function_inequality_l92_92717


namespace sequence_a113_l92_92382

theorem sequence_a113 {a : ℕ → ℝ} 
  (h1 : ∀ n, a n > 0)
  (h2 : a 1 = 1)
  (h3 : ∀ n, (a (n+1))^2 + (a n)^2 = 2 * n * ((a (n+1))^2 - (a n)^2)) :
  a 113 = 15 :=
sorry

end sequence_a113_l92_92382


namespace smallest_range_of_sample_l92_92892

open Real

theorem smallest_range_of_sample {a b c d e f g : ℝ}
  (h1 : (a + b + c + d + e + f + g) / 7 = 8)
  (h2 : d = 10)
  (h3 : a ≤ b ∧ b ≤ c ∧ c ≤ d ∧ d ≤ e ∧ e ≤ f ∧ f ≤ g) :
  ∃ r, r = g - a ∧ r = 8 :=
by
  sorry

end smallest_range_of_sample_l92_92892


namespace packets_of_sugar_per_week_l92_92080

theorem packets_of_sugar_per_week (total_grams : ℕ) (packet_weight : ℕ) (total_packets : ℕ) :
  total_grams = 2000 →
  packet_weight = 100 →
  total_packets = total_grams / packet_weight →
  total_packets = 20 := 
  by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3 

end packets_of_sugar_per_week_l92_92080


namespace sam_distance_l92_92822

theorem sam_distance (m_distance m_time s_time : ℝ) (m_distance_eq : m_distance = 150) (m_time_eq : m_time = 3) (s_time_eq : s_time = 4) :
  let rate := m_distance / m_time,
      s_distance := rate * s_time
  in s_distance = 200 :=
by
  let rate := m_distance / m_time
  let s_distance := rate * s_time
  sorry

end sam_distance_l92_92822


namespace increase_in_area_correct_l92_92076

-- Define the dimensions of the original rectangular garden
def length_rect := 60
def width_rect := 20

-- Define the perimeter of the rectangle
def perimeter_rect := 2 * (length_rect + width_rect)

-- Calculate the side length of the square garden using the same perimeter.
def side_square := perimeter_rect / 4

-- Define the area of the rectangular garden
def area_rect := length_rect * width_rect

-- Define the area of the square garden
def area_square := side_square * side_square

-- Define the increase in area after reshaping
def increase_in_area := area_square - area_rect

-- Prove that the increase in the area is 400 square feet
theorem increase_in_area_correct : increase_in_area = 400 := by
  -- The proof is omitted
  sorry

end increase_in_area_correct_l92_92076


namespace complex_fraction_simplification_l92_92910

theorem complex_fraction_simplification : 
  ∀ (i : ℂ), i^2 = -1 → (1 + i^2017) / (1 - i) = i :=
by
  intro i h_imag_unit
  sorry

end complex_fraction_simplification_l92_92910


namespace intersection_eq_l92_92371

def M : Set ℤ := {-1, 0, 1, 2}
def N : Set ℤ := {x | -1 ≤ x ∧ x < 2}

theorem intersection_eq : M ∩ N = {-1, 0, 1} :=
by
  sorry

end intersection_eq_l92_92371


namespace factor_poly_eq_factored_form_l92_92264

-- Defining the polynomial expressions
def poly1 (x : ℝ) := x^2 + 4 * x + 3
def poly2 (x : ℝ) := x^2 + 8 * x + 15
def poly3 (x : ℝ) := x^2 + 6 * x - 8

-- The main expression which needs to be factored
def main_expr (x : ℝ) := (poly1 x) * (poly2 x) + (poly3 x)

-- Stating the goal factored form
def factored_form (x : ℝ) := (x^2 + 6 * x + 19) * (x^2 + 6 * x - 2)

-- The theorem statement
theorem factor_poly_eq_factored_form (x : ℝ) : 
  main_expr x = factored_form x := 
by
  sorry

end factor_poly_eq_factored_form_l92_92264


namespace polynomial_factorization_l92_92308

theorem polynomial_factorization (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = (x^2 + 6*x + 1) * (x^2 + 6*x + 37) :=
  sorry

end polynomial_factorization_l92_92308


namespace max_truthful_dwarfs_le_one_l92_92416

def dwarf_height_claims : List ℕ := [60, 61, 62, 63, 64, 65, 66]

theorem max_truthful_dwarfs_le_one (heights : List ℕ) (h_lengths : heights.length = 7) (h_sorted : heights.sorted (· > ·)) :
  ∃ n, (n ≤ 1) ∧ ∀ i, i ∈ (Finset.range 7).to_list → (heights[i] ≠ dwarf_height_claims[i] → (1 = n)) :=
  sorry

end max_truthful_dwarfs_le_one_l92_92416


namespace popularity_order_l92_92195

def chess_popularity := 5 / 16
def drama_popularity := 7 / 24
def music_popularity := 11 / 32
def art_popularity := 13 / 48

theorem popularity_order :
  (31 / 96 < 34 / 96) ∧ (34 / 96 < 35 / 96) ∧ (35 / 96 < 36 / 96) ∧ 
  (chess_popularity < music_popularity) ∧ 
  (drama_popularity < music_popularity) ∧ 
  (music_popularity > art_popularity) ∧ 
  (chess_popularity > drama_popularity) ∧ 
  (drama_popularity > art_popularity) := 
sorry

end popularity_order_l92_92195


namespace largest_value_of_x_satisfying_sqrt3x_eq_5x_l92_92035

theorem largest_value_of_x_satisfying_sqrt3x_eq_5x : 
  ∃ (x : ℚ), sqrt (3 * x) = 5 * x ∧ (∀ y : ℚ, sqrt (3 * y) = 5 * y → y ≤ x) ∧ x = 3 / 25 :=
sorry

end largest_value_of_x_satisfying_sqrt3x_eq_5x_l92_92035


namespace zoo_children_tuesday_l92_92173

theorem zoo_children_tuesday 
  (x : ℕ) 
  (child_ticket_cost adult_ticket_cost : ℕ) 
  (children_monday adults_monday adults_tuesday : ℕ)
  (total_revenue : ℕ) : 
  child_ticket_cost = 3 → 
  adult_ticket_cost = 4 → 
  children_monday = 7 → 
  adults_monday = 5 → 
  adults_tuesday = 2 → 
  total_revenue = 61 → 
  7 * 3 + 5 * 4 + x * 3 + 2 * 4 = total_revenue → 
  x = 4 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end zoo_children_tuesday_l92_92173


namespace unique_x2_range_of_a_l92_92924

noncomputable def f (x : ℝ) (k a : ℝ) : ℝ :=
if x >= 0
then k*x + k*(1 - a^2)
else x^2 + (a^2 - 4*a)*x + (3 - a)^2

theorem unique_x2 (k a : ℝ) (x1 : ℝ) (hx1 : x1 ≠ 0) (hx2 : ∃ x2 : ℝ, x2 ≠ 0 ∧ x2 ≠ x1 ∧ f x2 k a = f x1 k a) :
f 0 k a = k*(1 - a^2) →
0 ≤ a ∧ a < 1 →
k = (3 - a)^2 / (1 - a^2) :=
sorry

variable (a : ℝ)

theorem range_of_a :
0 ≤ a ∧ a < 1 ↔ a^2 - 4*a ≤ 0 :=
sorry

end unique_x2_range_of_a_l92_92924


namespace probability_one_white_ball_conditional_probability_P_B_given_A_l92_92129

-- Definitions for Problem 1
def red_balls : Nat := 4
def white_balls : Nat := 2
def total_balls : Nat := red_balls + white_balls

def C (n k : ℕ) : ℕ := n.choose k

theorem probability_one_white_ball :
  (C 2 1 * C 4 2 : ℚ) / C 6 3 = 3 / 5 :=
by sorry

-- Definitions for Problem 2
def total_after_first_draw : Nat := total_balls - 1
def remaining_red_balls : Nat := red_balls - 1

theorem conditional_probability_P_B_given_A :
  (remaining_red_balls : ℚ) / total_after_first_draw = 3 / 5 :=
by sorry

end probability_one_white_ball_conditional_probability_P_B_given_A_l92_92129


namespace smallest_base_l92_92704

theorem smallest_base (b : ℕ) (n : ℕ) : (n = 512) → (b^3 ≤ n ∧ n < b^4) → ((n / b^3) % b + 1) % 2 = 0 → b = 6 := sorry

end smallest_base_l92_92704


namespace max_truthful_dwarfs_l92_92407

theorem max_truthful_dwarfs :
  ∃ n : ℕ, n = 1 ∧ 
    ∀ heights : Fin 7 → ℕ, 
      (heights 0 = 60 ∨ heights 1 = 61 ∨ heights 2 = 62 ∨ heights 3 = 63 ∨ heights 4 = 64 ∨ heights 5 = 65 ∨ heights 6 = 66) → 
      (∀ i j : Fin 7, i < j → heights i > heights j) → 
      ( ∃ t : Fin 7 → bool, 
        ( ∀ i j : Fin 7, t i = tt → t j = tt → i = j) ∧ 
        ( ∀ i : Fin 7, t i = tt → heights i = 60 + i ) ∧
        #(t.to_list.count tt) = n ) :=
by
  use 1
  sorry

end max_truthful_dwarfs_l92_92407


namespace dwarfs_truth_claims_l92_92414

/-- Seven dwarfs lined up by height, starting with the tallest.
Each dwarf claims a specific height as follows.
Verify that the largest number of dwarfs telling the truth is exactly 1. -/
theorem dwarfs_truth_claims :
  let heights := [60, 61, 62, 63, 64, 65, 66] in
  ∀ (actual_height: Fin 7 → ℕ), 
    (∀ i, actual_height i > actual_height (Fin.mk i (by linarith))) →
    ∀ claim_truth: Fin 7 → Prop, 
    (∀ i, claim_truth i ↔ actual_height i = heights[i]) →
    (∃ k, ∀ j, claim_truth j → j = k) :=
by
  intro heights actual_height h_order claim_truth h_claim_match
  existsi Fin.mk 0 (by linarith) -- Claim that only one can be telling the truth (arbitrary choice)
  sorry  -- Proof steps are omitted

end dwarfs_truth_claims_l92_92414


namespace correct_calculation_l92_92068

theorem correct_calculation (x : ℝ) (h : 3 * x - 12 = 60) : (x / 3) + 12 = 20 :=
by 
  sorry

end correct_calculation_l92_92068


namespace factor_polynomial_l92_92276

-- Define the necessary polynomials
def p (x : ℝ) : ℝ := x^2 + 4*x + 3
def q (x : ℝ) : ℝ := x^2 + 8*x + 15
def r (x : ℝ) : ℝ := x^2 + 6*x - 8

-- State the main theorem
theorem factor_polynomial (x : ℝ) : 
  (p x * q x) + r x = (x^2 + 7*x + 1) * (x^2 + 4*x + 7) :=
by
  sorry

end factor_polynomial_l92_92276


namespace michael_passes_donovan_l92_92069

theorem michael_passes_donovan
  (track_length : ℕ)
  (donovan_lap_time : ℕ)
  (michael_lap_time : ℕ)
  (start_time : ℕ)
  (L : ℕ)
  (h1 : track_length = 500)
  (h2 : donovan_lap_time = 45)
  (h3 : michael_lap_time = 40)
  (h4 : start_time = 0)
  : L = 9 :=
by
  sorry

end michael_passes_donovan_l92_92069


namespace sam_drove_200_miles_l92_92828

theorem sam_drove_200_miles
  (distance_m: ℝ)
  (time_m: ℝ)
  (distance_s: ℝ)
  (time_s: ℝ)
  (rate_m: ℝ)
  (rate_s: ℝ)
  (h1: distance_m = 150)
  (h2: time_m = 3)
  (h3: rate_m = distance_m / time_m)
  (h4: time_s = 4)
  (h5: rate_s = rate_m)
  (h6: distance_s = rate_s * time_s):
  distance_s = 200 :=
by
  sorry

end sam_drove_200_miles_l92_92828


namespace sam_drove_distance_l92_92818

theorem sam_drove_distance (marguerite_distance : ℝ) (marguerite_time : ℝ) (sam_time : ℝ) :
  marguerite_distance = 150 ∧ marguerite_time = 3 ∧ sam_time = 4 →
  (sam_time * (marguerite_distance / marguerite_time) = 200) :=
by
  sorry

end sam_drove_distance_l92_92818


namespace sum_of_numbers_is_twenty_l92_92071

-- Given conditions
variables {a b c : ℝ}

-- Prove that the sum of a, b, and c is 20 given the conditions
theorem sum_of_numbers_is_twenty (h1 : a^2 + b^2 + c^2 = 138) (h2 : ab + bc + ca = 131) :
  a + b + c = 20 :=
by
  sorry

end sum_of_numbers_is_twenty_l92_92071


namespace Caden_total_money_l92_92909

theorem Caden_total_money (p n d q : ℕ) (hp : p = 120)
    (hn : p = 3 * n) 
    (hd : n = 5 * d)
    (hq : q = 2 * d) :
    (p * 1 / 100 + n * 5 / 100 + d * 10 / 100 + q * 25 / 100) = 8 := 
by
  sorry

end Caden_total_money_l92_92909


namespace mushrooms_safe_to_eat_l92_92802

theorem mushrooms_safe_to_eat (S : ℕ) (Total_mushrooms Poisonous_mushrooms Uncertain_mushrooms : ℕ)
  (h1: Total_mushrooms = 32)
  (h2: Poisonous_mushrooms = 2 * S)
  (h3: Uncertain_mushrooms = 5)
  (h4: S + Poisonous_mushrooms + Uncertain_mushrooms = Total_mushrooms) :
  S = 9 :=
sorry

end mushrooms_safe_to_eat_l92_92802


namespace factor_polynomial_l92_92277

-- Define the necessary polynomials
def p (x : ℝ) : ℝ := x^2 + 4*x + 3
def q (x : ℝ) : ℝ := x^2 + 8*x + 15
def r (x : ℝ) : ℝ := x^2 + 6*x - 8

-- State the main theorem
theorem factor_polynomial (x : ℝ) : 
  (p x * q x) + r x = (x^2 + 7*x + 1) * (x^2 + 4*x + 7) :=
by
  sorry

end factor_polynomial_l92_92277


namespace intersection_eq_l92_92796

def setA (x : ℝ) : Prop := (x ≥ 1) ∨ (x ≤ -1)
def setB (y : ℝ) : Prop := (y ≥ 0)
def intersectionAB (x : ℝ) : Prop := (setA x) ∧ (setB (sqrt (x^2 - 1)))

theorem intersection_eq {x : ℝ} : setA x → setB (sqrt (x^2 - 1)) → (x ≥ 1) :=
begin
  intro hx,
  intro hy,
  cases hx,
  { exact hx },
  { exfalso,
    have : sqrt (x^2 - 1) < 0,
    { sorry },
    exact (not_lt_of_ge hy this)
  }
end

end intersection_eq_l92_92796


namespace graph_passes_through_point_l92_92015

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x-2) + 1

theorem graph_passes_through_point (a : ℝ) (h_pos : a > 0) (h_neq : a ≠ 1) : f a 2 = 2 :=
by
  sorry

end graph_passes_through_point_l92_92015


namespace flowers_given_l92_92403

theorem flowers_given (initial_flowers total_flowers flowers_given : ℝ)
  (h1 : initial_flowers = 67)
  (h2 : total_flowers = 157)
  (h3 : total_flowers = initial_flowers + flowers_given) :
  flowers_given = 90 :=
sorry

end flowers_given_l92_92403


namespace factor_poly_eq_factored_form_l92_92266

-- Defining the polynomial expressions
def poly1 (x : ℝ) := x^2 + 4 * x + 3
def poly2 (x : ℝ) := x^2 + 8 * x + 15
def poly3 (x : ℝ) := x^2 + 6 * x - 8

-- The main expression which needs to be factored
def main_expr (x : ℝ) := (poly1 x) * (poly2 x) + (poly3 x)

-- Stating the goal factored form
def factored_form (x : ℝ) := (x^2 + 6 * x + 19) * (x^2 + 6 * x - 2)

-- The theorem statement
theorem factor_poly_eq_factored_form (x : ℝ) : 
  main_expr x = factored_form x := 
by
  sorry

end factor_poly_eq_factored_form_l92_92266


namespace factorization_correct_l92_92288

noncomputable def polynomial_expr := 
  λ x : ℝ => (x^2 + 4 * x + 3) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8)

noncomputable def factored_expr := 
  λ x : ℝ => (x^2 + 6 * x + 9) * (x^2 + 6 * x + 1)

theorem factorization_correct : 
  ∀ x : ℝ, polynomial_expr x = factored_expr x :=
by
  intro x
  sorry

end factorization_correct_l92_92288


namespace excluded_avg_mark_l92_92002

theorem excluded_avg_mark (N A A_remaining excluded_count : ℕ)
  (hN : N = 15)
  (hA : A = 80)
  (hA_remaining : A_remaining = 90) 
  (h_excluded : excluded_count = 5) :
  (A * N - A_remaining * (N - excluded_count)) / excluded_count = 60 := sorry

end excluded_avg_mark_l92_92002


namespace exists_unique_n_digit_number_with_one_l92_92748

def n_digit_number (n : ℕ) : Type := {l : List ℕ // l.length = n ∧ ∀ x ∈ l, x = 1 ∨ x = 2 ∨ x = 3}

theorem exists_unique_n_digit_number_with_one (n : ℕ) (hn : n > 0) :
  ∃ x : n_digit_number n, x.val.count 1 = 1 ∧ ∀ y : n_digit_number n, y ≠ x → x.val.append [1] ≠ y.val.append [1] :=
sorry

end exists_unique_n_digit_number_with_one_l92_92748


namespace volume_of_polyhedron_l92_92229

open Real

-- Define the conditions
def square_side : ℝ := 100  -- in cm, equivalent to 1 meter
def rectangle_length : ℝ := 40  -- in cm
def rectangle_width : ℝ := 20  -- in cm
def trapezoid_leg_length : ℝ := 130  -- in cm

-- Define the question as a theorem statement
theorem volume_of_polyhedron :
  ∃ V : ℝ, V = 552 :=
sorry

end volume_of_polyhedron_l92_92229


namespace car_returns_to_start_after_5_operations_l92_92727

theorem car_returns_to_start_after_5_operations (α : ℝ) (h1 : 0 < α) (h2 : α < 180) : α = 72 ∨ α = 144 :=
sorry

end car_returns_to_start_after_5_operations_l92_92727


namespace sam_driving_distance_l92_92806

-- Definitions based on the conditions
def marguerite_distance : ℝ := 150
def marguerite_time : ℝ := 3
def sam_time : ℝ := 4

-- Desired statement using the given conditions
theorem sam_driving_distance :
  let rate := marguerite_distance / marguerite_time in
  let sam_distance := rate * sam_time in
  sam_distance = 200 :=
by
  sorry

end sam_driving_distance_l92_92806


namespace regular_hexagon_same_length_probability_l92_92166

theorem regular_hexagon_same_length_probability :
  let T : Finset (Finite_Field) := sorry, -- Define the actual set T una don the constructible geometry module in lean mathematical library
  let total_segments := 15,
  let sides := 6,
  let shorter_diagonals := 6,
  let longer_diagonals := 3,
  let prob_side := (6 / 15) * (5 / 14),
  let prob_shorter := (6 / 15) * (5 / 14),
  let prob_longer := (3 / 15) * (2 / 14),
  let total_prob := (prob_side + prob_shorter + prob_longer),
  (total_prob = (11 / 35)) := sorry

end regular_hexagon_same_length_probability_l92_92166


namespace find_number_lemma_l92_92100

theorem find_number_lemma (x : ℝ) (a b c d : ℝ) (h₁ : x = 5) 
  (h₂ : a = 0.47 * 1442) (h₃ : b = 0.36 * 1412) 
  (h₄ : c = a - b) (h₅ : d + c = x) : 
  d = -164.42 :=
by
  sorry

end find_number_lemma_l92_92100


namespace quadratic_inequality_solution_l92_92016

theorem quadratic_inequality_solution :
  ∀ x : ℝ, (x^2 - 4*x + 3) < 0 ↔ 1 < x ∧ x < 3 :=
by
  sorry

end quadratic_inequality_solution_l92_92016


namespace factorization_correct_l92_92292

noncomputable def polynomial_expr := 
  λ x : ℝ => (x^2 + 4 * x + 3) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8)

noncomputable def factored_expr := 
  λ x : ℝ => (x^2 + 6 * x + 9) * (x^2 + 6 * x + 1)

theorem factorization_correct : 
  ∀ x : ℝ, polynomial_expr x = factored_expr x :=
by
  intro x
  sorry

end factorization_correct_l92_92292


namespace SamDrove200Miles_l92_92807

/-- Given conditions -/
def MargueriteDistance : ℝ := 150
def MargueriteTime : ℝ := 3
def SameRateTime : ℝ := 4

/-- Calculate Marguerite's average speed -/
def MargueriteSpeed : ℝ := MargueriteDistance / MargueriteTime

/-- Calculate distance Sam drove -/
def SamDistance : ℝ := MargueriteSpeed * SameRateTime

/-- The theorem statement: Sam drove 200 miles -/
theorem SamDrove200Miles : SamDistance = 200 := by
  sorry

end SamDrove200Miles_l92_92807


namespace triangle_d_not_right_l92_92234

noncomputable def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

theorem triangle_d_not_right :
  ¬is_right_triangle 7 8 13 :=
by sorry

end triangle_d_not_right_l92_92234


namespace probability_of_not_all_8_sided_dice_rolling_the_same_number_l92_92042

def probability_not_all_same (total_faces : ℕ) (num_dice : ℕ) : ℚ :=
  let total_outcomes := total_faces ^ num_dice
  let same_number_outcomes := total_faces
  let p_same := same_number_outcomes / total_outcomes
  1 - p_same

theorem probability_of_not_all_8_sided_dice_rolling_the_same_number :
  probability_not_all_same 8 5 = 4095 / 4096 :=
by
  sorry

end probability_of_not_all_8_sided_dice_rolling_the_same_number_l92_92042


namespace train_cross_time_approx_l92_92931

noncomputable def length_of_train : ℝ := 100
noncomputable def speed_of_train_km_hr : ℝ := 80
noncomputable def length_of_bridge : ℝ := 142
noncomputable def total_distance : ℝ := length_of_train + length_of_bridge
noncomputable def speed_of_train_m_s : ℝ := speed_of_train_km_hr * 1000 / 3600
noncomputable def time_to_cross_bridge : ℝ := total_distance / speed_of_train_m_s

theorem train_cross_time_approx :
  abs (time_to_cross_bridge - 10.89) < 0.01 :=
by
  sorry

end train_cross_time_approx_l92_92931


namespace max_annual_profit_at_x_9_l92_92591

noncomputable def annual_profit (x : ℝ) : ℝ :=
if h : (0 < x ∧ x ≤ 10) then
  8.1 * x - x^3 / 30 - 10
else
  98 - 1000 / (3 * x) - 2.7 * x

theorem max_annual_profit_at_x_9 (x : ℝ) (h1 : 0 < x) (h2 : x ≤ 10) :
  annual_profit x ≤ annual_profit 9 :=
sorry

end max_annual_profit_at_x_9_l92_92591


namespace power_comparison_l92_92866

theorem power_comparison :
  2 ^ 16 = 256 * 16 ^ 2 := 
by
  sorry

end power_comparison_l92_92866


namespace probability_same_length_segments_l92_92146

theorem probability_same_length_segments (T : Finset ℕ) (sides : Finset ℕ) (diagonals : Finset ℕ) :
  sides.card = 6 →
  diagonals.card = 9 →
  (∃ d1 d2 : Finset ℕ, d1.card = 3 ∧ d2.card = 3 ∧ d1 ∩ d2 = ∅ ∧ diagonals = d1 ∪ d2) →
  T = sides ∪ diagonals →
  (∃ n : ℕ, (T.card.choose 2) = n * 5 ∧ ∃ f : ℕ, (∑ x in T, ∑ y in (T \ {x}), if x = y then 1 else 0) = f * 5) →
  ((∑ x in T, ∑ y in (T \ {x}), if x = y then 1 else 0).toRat / (T.card.choose 2).toRat = (1 : ℚ) / 5) :=
by
  sorry

end probability_same_length_segments_l92_92146


namespace intersection_complement_B_and_A_l92_92361

open Set Real

def A : Set ℝ := { x | x^2 - 4 * x + 3 < 0 }
def B : Set ℝ := { x | x > 2 }
def CR_B : Set ℝ := { x | x ≤ 2 }

theorem intersection_complement_B_and_A : CR_B ∩ A = { x | 1 < x ∧ x ≤ 2 } :=
by
  sorry

end intersection_complement_B_and_A_l92_92361


namespace petes_average_speed_l92_92088

theorem petes_average_speed
    (map_distance : ℝ := 5) 
    (time_taken : ℝ := 1.5) 
    (map_scale : ℝ := 0.05555555555555555) :
    (map_distance / map_scale) / time_taken = 60 := 
by
    sorry

end petes_average_speed_l92_92088


namespace max_students_seated_l92_92070

/-- Problem statement:
There are a total of 8 rows of desks.
The first row has 10 desks.
Each subsequent row has 2 more desks than the previous row.
We need to prove that the maximum number of students that can be seated in the class is 136.
-/
theorem max_students_seated : 
  let n := 8      -- number of rows
  let a1 := 10    -- desks in the first row
  let d := 2      -- common difference
  let an := a1 + (n - 1) * d  -- desks in the n-th row
  let S := n / 2 * (a1 + an)  -- sum of the arithmetic series
  S = 136 :=
by
  sorry

end max_students_seated_l92_92070


namespace probability_of_same_length_l92_92164

-- Define the set T and its properties
def T : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
def sides : ℕ := 6
def diagonals : ℕ := 9
def total_elements : ℕ := 15
def probability_same_length : ℚ := 17 / 35

-- Lean 4 statement (theorem)
theorem probability_of_same_length: 
    ( ∃ (sides diagonals : ℕ), sides = 6 ∧ diagonals = 9 ∧ sides + diagonals = 15 →
                                              ∃ probability_same_length : ℚ, probability_same_length = 17 / 35) := 
sorry

end probability_of_same_length_l92_92164


namespace tank_emptying_time_correct_l92_92900

noncomputable def tank_emptying_time : ℝ :=
  let initial_volume := 1 / 5
  let fill_rate := 1 / 15
  let empty_rate := 1 / 6
  let combined_rate := fill_rate - empty_rate
  initial_volume / combined_rate

theorem tank_emptying_time_correct :
  tank_emptying_time = 2 :=
by
  -- Proof will be provided here
  sorry

end tank_emptying_time_correct_l92_92900


namespace sequence_geometric_and_general_formula_find_minimum_n_l92_92107

theorem sequence_geometric_and_general_formula 
  (a : ℕ → ℕ)
  (S : ℕ → ℕ)
  (h1 : ∀ n : ℕ, n > 0 → S n + n = 2 * a n) :
  (∀ n : ℕ, n ≥ 1 → a (n + 1) + 1 = 2 * (a n + 1)) ∧ (∀ n : ℕ, n ≥ 1 → a n = 2^n - 1) :=
sorry

theorem find_minimum_n 
  (a : ℕ → ℕ) 
  (S : ℕ → ℕ) 
  (b T : ℕ → ℕ)
  (h1 : ∀ n : ℕ, n > 0 → S n + n = 2 * a n)
  (h2 : ∀ n : ℕ, b n = (2 * n + 1) * a n + (2 * n + 1))
  (h3 : T 0 = 0)
  (h4 : ∀ n : ℕ, T (n + 1) = T n + b (n + 1)) :
  ∃ n : ℕ, n ≥ 1 ∧ (T n - 2) / (2 * n - 1) > 2010 :=
sorry

end sequence_geometric_and_general_formula_find_minimum_n_l92_92107


namespace problem1_problem2_l92_92348

def setA (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 3}
def setB : Set ℝ := {x | x > 1 ∨ x < -6}

theorem problem1 (a : ℝ) : (setA a ∩ setB = ∅) → (-6 ≤ a ∧ a ≤ -2) := by
  intro h
  sorry

theorem problem2 (a : ℝ) : (setA a ∪ setB = setB) → (a < -9 ∨ a > 1) := by
  intro h
  sorry

end problem1_problem2_l92_92348


namespace allocation_schemes_l92_92626

theorem allocation_schemes :
  ∃ (n : ℕ), n = 240 ∧ (∀ (volunteers : Fin 5 → Fin 4), 
    (∃ (assign : Fin 5 → Fin 4), 
      (∀ (i : Fin 4), ∃ (j : Fin 5), assign j = i)
      ∧ (∀ (k : Fin 5), assign k ≠ assign k)) 
    → true) := sorry

end allocation_schemes_l92_92626


namespace log_over_sqrt_defined_l92_92631

theorem log_over_sqrt_defined (x : ℝ) : (2 < x ∧ x < 5) ↔ ∃ f : ℝ, f = (log (5 - x) / sqrt (x - 2)) :=
by
  sorry

end log_over_sqrt_defined_l92_92631


namespace fraction_cost_of_raisins_l92_92579

variable (cost_raisins cost_nuts total_cost_raisins total_cost_nuts total_cost : ℝ)

theorem fraction_cost_of_raisins (h1 : cost_nuts = 3 * cost_raisins)
                                 (h2 : total_cost_raisins = 4 * cost_raisins)
                                 (h3 : total_cost_nuts = 4 * cost_nuts)
                                 (h4 : total_cost = total_cost_raisins + total_cost_nuts) :
                                 (total_cost_raisins / total_cost) = (1 / 4) :=
by
  sorry

end fraction_cost_of_raisins_l92_92579


namespace manager_salary_l92_92469

theorem manager_salary 
    (avg_salary_18 : ℕ)
    (new_avg_salary : ℕ)
    (num_employees : ℕ)
    (num_employees_with_manager : ℕ)
    (old_total_salary : ℕ := num_employees * avg_salary_18)
    (new_total_salary : ℕ := num_employees_with_manager * new_avg_salary) :
    (new_avg_salary = avg_salary_18 + 200) →
    (old_total_salary = 18 * 2000) →
    (new_total_salary = 19 * (2000 + 200)) →
    new_total_salary - old_total_salary = 5800 :=
by
  intros h1 h2 h3
  sorry

end manager_salary_l92_92469


namespace sam_drove_distance_l92_92815

theorem sam_drove_distance (marguerite_distance : ℝ) (marguerite_time : ℝ) (sam_time : ℝ) :
  marguerite_distance = 150 ∧ marguerite_time = 3 ∧ sam_time = 4 →
  (sam_time * (marguerite_distance / marguerite_time) = 200) :=
by
  sorry

end sam_drove_distance_l92_92815


namespace find_n_l92_92763

variable (n : ℚ)

theorem find_n (h : (2 / (n + 2) + 3 / (n + 2) + n / (n + 2) + 1 / (n + 2) = 4)) : 
  n = -2 / 3 :=
by
  sorry

end find_n_l92_92763


namespace probability_of_same_length_l92_92161

-- Define the set T and its properties
def T : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
def sides : ℕ := 6
def diagonals : ℕ := 9
def total_elements : ℕ := 15
def probability_same_length : ℚ := 17 / 35

-- Lean 4 statement (theorem)
theorem probability_of_same_length: 
    ( ∃ (sides diagonals : ℕ), sides = 6 ∧ diagonals = 9 ∧ sides + diagonals = 15 →
                                              ∃ probability_same_length : ℚ, probability_same_length = 17 / 35) := 
sorry

end probability_of_same_length_l92_92161


namespace factorization_correct_l92_92296

noncomputable def polynomial_expr := 
  λ x : ℝ => (x^2 + 4 * x + 3) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8)

noncomputable def factored_expr := 
  λ x : ℝ => (x^2 + 6 * x + 9) * (x^2 + 6 * x + 1)

theorem factorization_correct : 
  ∀ x : ℝ, polynomial_expr x = factored_expr x :=
by
  intro x
  sorry

end factorization_correct_l92_92296


namespace diff_implies_continuous_l92_92976

def differentiable_imp_continuous (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  DifferentiableAt ℝ f x₀ → ContinuousAt f x₀

-- Problem statement: if f is differentiable at x₀, then it is continuous at x₀.
theorem diff_implies_continuous (f : ℝ → ℝ) (x₀ : ℝ) : differentiable_imp_continuous f x₀ :=
by
  sorry

end diff_implies_continuous_l92_92976


namespace negation_proof_l92_92690

theorem negation_proof :
  (¬ ∀ x : ℝ, x < 0 → 1 - x > Real.exp x) ↔ (∃ x_0 : ℝ, x_0 < 0 ∧ 1 - x_0 ≤ Real.exp x_0) :=
by
  sorry

end negation_proof_l92_92690


namespace Sarah_books_in_8_hours_l92_92842

theorem Sarah_books_in_8_hours (pages_per_hour: ℕ) (pages_per_book: ℕ) (hours_available: ℕ) 
  (h_pages_per_hour: pages_per_hour = 120) (h_pages_per_book: pages_per_book = 360) (h_hours_available: hours_available = 8) :
  hours_available * pages_per_hour / pages_per_book = 2 := by
  sorry

end Sarah_books_in_8_hours_l92_92842


namespace weight_labels_correct_l92_92769

-- Noncomputable because we're dealing with theoretical weight comparisons
noncomputable section

-- Defining the weights and their properties
variables {x1 x2 x3 x4 x5 x6 : ℕ}

-- Given conditions as stated
axiom h1 : x1 + x2 + x3 = 6
axiom h2 : x6 = 6
axiom h3 : x1 + x6 < x3 + x5

theorem weight_labels_correct :
  x1 = 1 ∧ x2 = 2 ∧ x3 = 3 ∧ x4 = 4 ∧ x5 = 5 ∧ x6 = 6 :=
sorry

end weight_labels_correct_l92_92769

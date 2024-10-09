import Mathlib

namespace product_of_all_n_satisfying_quadratic_l1641_164118

theorem product_of_all_n_satisfying_quadratic :
  (∃ n : ℕ, n^2 - 40 * n + 399 = 3) ∧
  (∀ p : ℕ, Prime p → ((∃ n : ℕ, n^2 - 40 * n + 399 = p) → p = 3)) →
  ∃ n1 n2 : ℕ, (n1^2 - 40 * n1 + 399 = 3) ∧ (n2^2 - 40 * n2 + 399 = 3) ∧ n1 ≠ n2 ∧ (n1 * n2 = 396) :=
by
  sorry

end product_of_all_n_satisfying_quadratic_l1641_164118


namespace sum_of_fraction_equiv_l1641_164126

theorem sum_of_fraction_equiv : 
  let x := 3.714714714
  let num := 3711
  let denom := 999
  3711 + 999 = 4710 :=
by 
  sorry

end sum_of_fraction_equiv_l1641_164126


namespace nth_term_of_sequence_99_l1641_164105

def sequence_rule (n : ℕ) : ℕ :=
  if n < 20 then n * 9
  else if n % 2 = 0 then n / 2
  else if n > 19 ∧ n % 7 ≠ 0 then n - 5
  else n + 7

noncomputable def sequence_nth_term (start : ℕ) (n : ℕ) : ℕ :=
  Nat.repeat sequence_rule n start

theorem nth_term_of_sequence_99 :
  sequence_nth_term 65 98 = 30 :=
sorry

end nth_term_of_sequence_99_l1641_164105


namespace george_initial_socks_l1641_164115

theorem george_initial_socks (S : ℕ) (h : S - 4 + 36 = 60) : S = 28 :=
by
  sorry

end george_initial_socks_l1641_164115


namespace quadratic_has_two_distinct_real_roots_l1641_164123

theorem quadratic_has_two_distinct_real_roots :
  ∀ (x : ℝ), ∃ (r1 r2 : ℝ), (x^2 - 2*x - 1 = 0) → r1 ≠ r2 :=
by
  sorry

end quadratic_has_two_distinct_real_roots_l1641_164123


namespace original_quadrilateral_area_l1641_164145

theorem original_quadrilateral_area :
  let deg45 := (Real.pi / 4)
  let h := 1 * Real.sin deg45
  let base_bottom := 1 + 2 * h
  let area_perspective := 0.5 * (1 + base_bottom) * h
  let area_original := area_perspective * (2 * Real.sqrt 2)
  area_original = 2 + Real.sqrt 2 := by
  sorry

end original_quadrilateral_area_l1641_164145


namespace sum_of_money_l1641_164113

noncomputable def Patricia : ℕ := 60
noncomputable def Jethro : ℕ := Patricia / 3
noncomputable def Carmen : ℕ := 2 * Jethro - 7

theorem sum_of_money : Patricia + Jethro + Carmen = 113 := by
  sorry

end sum_of_money_l1641_164113


namespace car_travel_distance_l1641_164188

-- Define the original gas mileage as x
variable (x : ℝ) (D : ℝ)

-- Define the conditions
def initial_condition : Prop := D = 12 * x
def revised_condition : Prop := D = 10 * (x + 2)

-- The proof goal
theorem car_travel_distance
  (h1 : initial_condition x D)
  (h2 : revised_condition x D) :
  D = 120 := by
  sorry

end car_travel_distance_l1641_164188


namespace triangle_is_isosceles_l1641_164152

theorem triangle_is_isosceles (a b c : ℝ) (A B C : ℝ) 
  (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) 
  (hABC_sum : A + B + C = π) 
  (cos_rule : a * Real.cos B + b * Real.cos A = a) :
  a = c :=
by
  sorry

end triangle_is_isosceles_l1641_164152


namespace pages_for_thirty_dollars_l1641_164198

-- Problem Statement Definitions
def costPerCopy := 4 -- cents
def pagesPerCopy := 2 -- pages
def totalCents := 3000 -- cents
def totalPages := 1500 -- pages

-- Theorem: Calculating the number of pages for a given cost.
theorem pages_for_thirty_dollars (c_per_copy : ℕ) (p_per_copy : ℕ) (t_cents : ℕ) (t_pages : ℕ) : 
  c_per_copy = 4 → p_per_copy = 2 → t_cents = 3000 → t_pages = 1500 := by
  intros h_cpc h_ppc h_tc
  sorry

end pages_for_thirty_dollars_l1641_164198


namespace lucas_payment_l1641_164161

noncomputable def payment (windows_per_floor : ℕ) (floors : ℕ) (days : ℕ) 
  (earn_per_window : ℝ) (delay_penalty : ℝ) (period : ℕ) : ℝ :=
  let total_windows := windows_per_floor * floors
  let earnings := total_windows * earn_per_window
  let penalty_periods := days / period
  let total_penalty := penalty_periods * delay_penalty
  earnings - total_penalty

theorem lucas_payment :
  payment 3 3 6 2 1 3 = 16 := by
  sorry

end lucas_payment_l1641_164161


namespace simplify_subtracted_terms_l1641_164140

theorem simplify_subtracted_terms (r : ℝ) : 180 * r - 88 * r = 92 * r := 
by 
  sorry

end simplify_subtracted_terms_l1641_164140


namespace cannot_be_combined_with_sqrt2_l1641_164108

def can_be_combined (x y : ℝ) : Prop := ∃ k : ℝ, k * x = y

theorem cannot_be_combined_with_sqrt2 :
  let a := Real.sqrt (1 / 2)
  let b := Real.sqrt 8
  let c := Real.sqrt 12
  let d := -Real.sqrt 18
  ¬ can_be_combined c (Real.sqrt 2) := 
by
  sorry

end cannot_be_combined_with_sqrt2_l1641_164108


namespace double_theta_acute_l1641_164177

theorem double_theta_acute (θ : ℝ) (h : 0 < θ ∧ θ < 90) : 0 < 2 * θ ∧ 2 * θ < 180 :=
by
  sorry

end double_theta_acute_l1641_164177


namespace product_in_third_quadrant_l1641_164116

def z1 : ℂ := 1 - 3 * Complex.I
def z2 : ℂ := 3 - 2 * Complex.I
def z := z1 * z2

theorem product_in_third_quadrant : z.re < 0 ∧ z.im < 0 := 
sorry

end product_in_third_quadrant_l1641_164116


namespace average_speed_with_stoppages_l1641_164174

theorem average_speed_with_stoppages
    (D : ℝ) -- distance the train travels
    (T_no_stop : ℝ := D / 250) -- time taken to cover the distance without stoppages
    (T_with_stop : ℝ := 2 * T_no_stop) -- total time with stoppages
    : (D / T_with_stop) = 125 := 
by sorry

end average_speed_with_stoppages_l1641_164174


namespace technicians_in_workshop_l1641_164131

theorem technicians_in_workshop :
  (∃ T R: ℕ, T + R = 42 ∧ 8000 * 42 = 18000 * T + 6000 * R) → ∃ T: ℕ, T = 7 :=
by
  sorry

end technicians_in_workshop_l1641_164131


namespace chi_square_test_l1641_164125

-- Conditions
def n : ℕ := 100
def a : ℕ := 5
def b : ℕ := 55
def c : ℕ := 15
def d : ℕ := 25

-- Critical chi-square value for alpha = 0.001
def chi_square_critical : ℝ := 10.828

-- Calculated chi-square value
noncomputable def chi_square_value : ℝ :=
  (n * (a * d - b * c)^2) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Statement to prove
theorem chi_square_test : chi_square_value > chi_square_critical :=
by sorry

end chi_square_test_l1641_164125


namespace value_of_expression_l1641_164121

theorem value_of_expression (m n : ℝ) (h : m + n = 4) : 2 * m^2 + 4 * m * n + 2 * n^2 - 5 = 27 :=
  sorry

end value_of_expression_l1641_164121


namespace min_cards_for_certain_event_l1641_164183

-- Let's define the deck configuration
structure DeckConfig where
  spades : ℕ
  clubs : ℕ
  hearts : ℕ
  total : ℕ

-- Define the given condition of the deck
def givenDeck : DeckConfig := { spades := 5, clubs := 4, hearts := 6, total := 15 }

-- Predicate to check if m cards drawn guarantees all three suits are present
def is_certain_event (m : ℕ) (deck : DeckConfig) : Prop :=
  m >= deck.spades + deck.hearts + 1

-- The main theorem to prove the minimum number of cards m
theorem min_cards_for_certain_event : ∀ m, is_certain_event m givenDeck ↔ m = 12 :=
by
  sorry

end min_cards_for_certain_event_l1641_164183


namespace time_to_school_l1641_164138

theorem time_to_school (total_distance walk_speed run_speed distance_ran : ℕ) (h_total : total_distance = 1800)
    (h_walk_speed : walk_speed = 70) (h_run_speed : run_speed = 210) (h_distance_ran : distance_ran = 600) :
    total_distance / walk_speed + distance_ran / run_speed = 20 := by
  sorry

end time_to_school_l1641_164138


namespace incorrect_parallel_m_n_l1641_164175

variables {l m n : Type} [LinearOrder m] [LinearOrder n] {α β : Type}

-- Assumptions for parallelism and orthogonality
def parallel (x y : Type) : Prop := sorry
def orthogonal (x y : Type) : Prop := sorry

-- Conditions
axiom parallel_m_l : parallel m l
axiom parallel_n_l : parallel n l
axiom orthogonal_m_α : orthogonal m α
axiom parallel_m_β : parallel m β
axiom parallel_m_α : parallel m α
axiom parallel_n_α : parallel n α
axiom orthogonal_m_β : orthogonal m β
axiom orthogonal_α_β : orthogonal α β

-- The theorem to prove
theorem incorrect_parallel_m_n : parallel m α ∧ parallel n α → ¬ parallel m n := sorry

end incorrect_parallel_m_n_l1641_164175


namespace lock_rings_l1641_164192

theorem lock_rings (n : ℕ) (h : 6 ^ n - 1 ≤ 215) : n = 3 :=
sorry

end lock_rings_l1641_164192


namespace problem1_problem2_l1641_164164

-- Problem 1:
theorem problem1 (α : ℝ) (h : Real.tan α = 2) : 
  (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 3 :=
sorry

-- Problem 2:
theorem problem2 (α : ℝ) : 
  (Real.tan (2 * Real.pi - α) * Real.cos (2 * Real.pi - α) * Real.sin (-α + 3 * Real.pi / 2)) /
  (Real.cos (-α + Real.pi) * Real.sin (-Real.pi + α)) = 1 :=
sorry

end problem1_problem2_l1641_164164


namespace mia_bought_more_pencils_l1641_164190

theorem mia_bought_more_pencils (p : ℝ) (n1 n2 : ℕ) 
  (price_pos : p > 0.01)
  (liam_spent : 2.10 = p * n1)
  (mia_spent : 2.82 = p * n2) :
  (n2 - n1) = 12 := 
by
  sorry

end mia_bought_more_pencils_l1641_164190


namespace complex_expr_simplify_l1641_164156

noncomputable def complex_demo : Prop :=
  let i := Complex.I
  7 * (4 + 2 * i) - 2 * i * (7 + 3 * i) = (34 : ℂ)

theorem complex_expr_simplify : 
  complex_demo :=
by
  -- proof skipped
  sorry

end complex_expr_simplify_l1641_164156


namespace remainder_of_349_divided_by_17_l1641_164112

theorem remainder_of_349_divided_by_17 : 
  (349 % 17 = 9) := 
by
  sorry

end remainder_of_349_divided_by_17_l1641_164112


namespace not_consecutive_l1641_164104

theorem not_consecutive (a b c : ℕ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a) : 
  ¬ (∃ n : ℕ, (2023 + a - b = n ∧ 2023 + b - c = n + 1 ∧ 2023 + c - a = n + 2) ∨ 
    (2023 + a - b = n ∧ 2023 + b - c = n - 1 ∧ 2023 + c - a = n - 2)) :=
by
  sorry

end not_consecutive_l1641_164104


namespace graphs_intersect_at_one_point_l1641_164100

theorem graphs_intersect_at_one_point (a : ℝ) : 
  (∀ x : ℝ, (a * x^2 + 3 * x + 1 = -x - 1) ↔ a = 2) :=
by
  sorry

end graphs_intersect_at_one_point_l1641_164100


namespace eve_walked_distance_l1641_164134

-- Defining the distances Eve ran and walked
def distance_ran : ℝ := 0.7
def distance_walked : ℝ := distance_ran - 0.1

-- Proving that the distance Eve walked is 0.6 mile
theorem eve_walked_distance : distance_walked = 0.6 := by
  -- The proof is omitted.
  sorry

end eve_walked_distance_l1641_164134


namespace number_of_female_officers_l1641_164144

theorem number_of_female_officers (total_on_duty : ℕ) (female_on_duty : ℕ) (percentage_on_duty : ℚ) : 
  total_on_duty = 500 → 
  female_on_duty = 250 → 
  percentage_on_duty = 1/4 → 
  (female_on_duty : ℚ) = percentage_on_duty * (total_on_duty / 2 : ℚ) →
  (total_on_duty : ℚ) = 4 * female_on_duty →
  total_on_duty = 1000 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end number_of_female_officers_l1641_164144


namespace ordered_pairs_1806_l1641_164196

theorem ordered_pairs_1806 :
  (∃ (xy_list : List (ℕ × ℕ)), xy_list.length = 12 ∧ ∀ (xy : ℕ × ℕ), xy ∈ xy_list → xy.1 * xy.2 = 1806) :=
sorry

end ordered_pairs_1806_l1641_164196


namespace find_number_l1641_164195

theorem find_number (x : ℤ) (h : 2 * x + 5 = 17) : x = 6 := 
by
  sorry

end find_number_l1641_164195


namespace simplify_expression_l1641_164197

theorem simplify_expression (x : ℝ) : 
  (3 * x - 4) * (x + 8) - (x + 6) * (3 * x - 2) = 4 * x - 20 := 
by
  sorry

end simplify_expression_l1641_164197


namespace smallest_part_proportional_l1641_164184

/-- If we divide 124 into three parts proportional to 2, 1/2, and 1/4,
    prove that the smallest part is 124 / 11. -/
theorem smallest_part_proportional (x : ℝ) 
  (h : 2 * x + (1 / 2) * x + (1 / 4) * x = 124) : 
  (1 / 4) * x = 124 / 11 :=
sorry

end smallest_part_proportional_l1641_164184


namespace sixth_ninth_grader_buddy_fraction_l1641_164181

theorem sixth_ninth_grader_buddy_fraction
  (s n : ℕ)
  (h_fraction_pairs : n / 4 = s / 3)
  (h_buddy_pairing : (∀ i, i < n -> ∃ j, j < s) 
     ∧ (∀ j, j < s -> ∃ i, i < n) -- each sixth grader paired with one ninth grader and vice versa
  ) :
  (n / 4 + s / 3) / (n + s) = 2 / 7 :=
by 
  sorry

end sixth_ninth_grader_buddy_fraction_l1641_164181


namespace equilibrium_constant_relationship_l1641_164101

def given_problem (K1 K2 : ℝ) : Prop :=
  K2 = (1 / K1)^(1 / 2)

theorem equilibrium_constant_relationship (K1 K2 : ℝ) (h : given_problem K1 K2) :
  K1 = 1 / K2^2 :=
by sorry

end equilibrium_constant_relationship_l1641_164101


namespace distinct_solutions_abs_eq_l1641_164114

theorem distinct_solutions_abs_eq : ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ (|2 * x1 - 14| = |x1 + 4| ∧ |2 * x2 - 14| = |x2 + 4|) ∧ (∀ x, |2 * x - 14| = |x + 4| → (x = x1 ∨ x = x2)) :=
by {
  sorry
}

end distinct_solutions_abs_eq_l1641_164114


namespace fixed_point_of_transformed_exponential_l1641_164124

variable (a : ℝ)
variable (h_pos : 0 < a)
variable (h_ne_one : a ≠ 1)

theorem fixed_point_of_transformed_exponential :
    (∃ x y : ℝ, (y = a^(x-2) + 2) ∧ (y = x) ∧ (x = 2) ∧ (y = 3)) :=
by {
    sorry -- Proof goes here
}

end fixed_point_of_transformed_exponential_l1641_164124


namespace sum_even_probability_l1641_164170

def probability_even_sum_of_wheels : ℚ :=
  let prob_wheel1_odd := 3 / 5
  let prob_wheel1_even := 2 / 5
  let prob_wheel2_odd := 2 / 3
  let prob_wheel2_even := 1 / 3
  (prob_wheel1_odd * prob_wheel2_odd) + (prob_wheel1_even * prob_wheel2_even)

theorem sum_even_probability :
  probability_even_sum_of_wheels = 8 / 15 :=
by
  -- Goal statement with calculations showed in the equivalent problem
  sorry

end sum_even_probability_l1641_164170


namespace distinct_paths_l1641_164102

def binom (n k : ℕ) : ℕ :=
  if k > n then 0
  else (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem distinct_paths (right_steps up_steps : ℕ) : right_steps = 7 → up_steps = 3 →
  binom (right_steps + up_steps) up_steps = 120 := 
by
  intros h1 h2
  rw [h1, h2]
  unfold binom
  simp
  norm_num
  sorry

end distinct_paths_l1641_164102


namespace r_exceeds_s_l1641_164139

theorem r_exceeds_s (x y : ℚ) (h1 : x + 2 * y = 16 / 3) (h2 : 5 * x + 3 * y = 26) :
  x - y = 106 / 21 :=
sorry

end r_exceeds_s_l1641_164139


namespace solution_concentration_l1641_164186

theorem solution_concentration (y z : ℝ) :
  let x_vol := 300
  let y_vol := 2 * z
  let z_vol := z
  let total_vol := x_vol + y_vol + z_vol
  let alcohol_x := 0.10 * x_vol
  let alcohol_y := 0.30 * y_vol
  let alcohol_z := 0.40 * z_vol
  let total_alcohol := alcohol_x + alcohol_y + alcohol_z
  total_vol = 600 ∧ y_vol = 2 * z_vol ∧ y_vol + z_vol = 300 → 
  total_alcohol / total_vol = 21.67 / 100 :=
by
  sorry

end solution_concentration_l1641_164186


namespace parallelogram_area_l1641_164163

theorem parallelogram_area (b : ℝ) (h : ℝ) (A : ℝ) (base_condition : b = 8) (altitude_condition : h = 2 * b) : 
  A = 128 :=
by 
  sorry

end parallelogram_area_l1641_164163


namespace line_third_quadrant_l1641_164111

theorem line_third_quadrant (A B C : ℝ) (h_origin : C = 0)
  (h_third_quadrant : ∃ (x y : ℝ), x < 0 ∧ y < 0 ∧ A * x - B * y = 0) :
  A * B < 0 :=
by
  sorry

end line_third_quadrant_l1641_164111


namespace prove_expression_value_l1641_164185

theorem prove_expression_value (x : ℕ) (h : x = 3) : x + x * (x ^ (x + 1)) = 246 := by
  rw [h]
  sorry

end prove_expression_value_l1641_164185


namespace separation_of_homologous_chromosomes_only_in_meiosis_l1641_164171

-- We start by defining the conditions extracted from the problem.
def chromosome_replication (phase: String) : Prop :=  
  phase = "S phase"

def separation_of_homologous_chromosomes (process: String) : Prop := 
  process = "meiosis I"

def separation_of_chromatids (process: String) : Prop := 
  process = "mitosis anaphase" ∨ process = "meiosis II anaphase II"

def cytokinesis (end_phase: String) : Prop := 
  end_phase = "end mitosis" ∨ end_phase = "end meiosis"

-- Now, we state that the separation of homologous chromosomes does not occur during mitosis.
theorem separation_of_homologous_chromosomes_only_in_meiosis :
  ∀ (process: String), ¬ separation_of_homologous_chromosomes "mitosis" := 
sorry

end separation_of_homologous_chromosomes_only_in_meiosis_l1641_164171


namespace adjacent_side_length_l1641_164199

-- Given the conditions
variables (a b : ℝ)
-- Area of the rectangular flower bed
def area := 6 * a * b - 2 * b
-- One side of the rectangular flower bed
def side1 := 2 * b

-- Prove the length of the adjacent side
theorem adjacent_side_length : 
  (6 * a * b - 2 * b) / (2 * b) = 3 * a - 1 :=
by sorry

end adjacent_side_length_l1641_164199


namespace find_x_l1641_164149

theorem find_x (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 101) : x = 50 :=
sorry

end find_x_l1641_164149


namespace max_value_of_expressions_l1641_164166

theorem max_value_of_expressions (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : a + b = 1) :
  b > 1/2 ∧ b > 2 * a * b ∧ b > a^2 + b^2 :=
by
  sorry

end max_value_of_expressions_l1641_164166


namespace jim_taxi_total_charge_l1641_164122

noncomputable def total_charge (initial_fee : ℝ) (per_mile_fee : ℝ) (mile_chunk : ℝ) (distance : ℝ) : ℝ :=
  initial_fee + (distance / mile_chunk) * per_mile_fee

theorem jim_taxi_total_charge :
  total_charge 2.35 0.35 (2/5) 3.6 = 5.50 :=
by
  sorry

end jim_taxi_total_charge_l1641_164122


namespace find_group_2018_l1641_164160

theorem find_group_2018 :
  ∃ n : ℕ, 2 ≤ n ∧ 2018 ≤ 2 * n * (n + 1) ∧ 2018 > 2 * (n - 1) * n :=
by
  sorry

end find_group_2018_l1641_164160


namespace cab_time_l1641_164157

theorem cab_time (d t : ℝ) (v : ℝ := d / t)
    (v1 : ℝ := (5 / 6) * v)
    (t1 : ℝ := d / v1)
    (v2 : ℝ := (2 / 3) * v)
    (t2 : ℝ := d / v2)
    (T : ℝ := t1 + t2)
    (delay : ℝ := 5) :
    let total_time := 2 * t + delay
    t * d ≠ 0 → T = total_time → t = 50 / 7 := by
    sorry

end cab_time_l1641_164157


namespace roger_candies_left_l1641_164143

theorem roger_candies_left (initial_candies : ℕ) (to_stephanie : ℕ) (to_john : ℕ) (to_emily : ℕ) : 
  initial_candies = 350 ∧ to_stephanie = 45 ∧ to_john = 25 ∧ to_emily = 18 → 
  initial_candies - (to_stephanie + to_john + to_emily) = 262 :=
by
  sorry

end roger_candies_left_l1641_164143


namespace negative_solution_iff_sum_zero_l1641_164153

theorem negative_solution_iff_sum_zero (a b c : ℝ) :
  (∃ x y : ℝ, x < 0 ∧ y < 0 ∧ (a * x + b * y = c) ∧ (b * x + c * y = a) ∧ (c * x + a * y = b)) ↔
  a + b + c = 0 :=
by
  sorry

end negative_solution_iff_sum_zero_l1641_164153


namespace not_partitionable_1_to_15_l1641_164141

theorem not_partitionable_1_to_15 :
  ∀ (A B : Finset ℕ), (∀ x ∈ A, x ∈ Finset.range 16) →
    (∀ x ∈ B, x ∈ Finset.range 16) →
    A.card = 2 → B.card = 13 →
    A ∪ B = Finset.range 16 →
    ¬(A.sum id = B.prod id) :=
by
  -- To be proved
  sorry

end not_partitionable_1_to_15_l1641_164141


namespace ricciana_jump_distance_l1641_164110

theorem ricciana_jump_distance (R : ℕ) :
  let Ricciana_run := 20
  let Margarita_run := 18
  let Margarita_jump := 2 * R - 1
  let Total_distance_Margarita := Margarita_run + Margarita_jump
  let Total_distance_Ricciana := Ricciana_run + R + 1
  Total_distance_Margarita = Total_distance_Ricciana → R = 22 :=
by
  -- Definitions
  let Ricciana_run := 20
  let Margarita_run := 18
  let Margarita_jump := 2 * R - 1
  let Total_distance_Margarita := Margarita_run + Margarita_jump
  let Total_distance_Ricciana := Ricciana_run + R + 1

  -- Given condition
  intro h
  sorry

end ricciana_jump_distance_l1641_164110


namespace find_highway_speed_l1641_164193

def car_local_distance := 40
def car_local_speed := 20
def car_highway_distance := 180
def average_speed := 44
def speed_of_car_on_highway := 60

theorem find_highway_speed :
  car_local_distance / car_local_speed + car_highway_distance / speed_of_car_on_highway = (car_local_distance + car_highway_distance) / average_speed :=
by
  sorry

end find_highway_speed_l1641_164193


namespace average_of_distinct_s_values_l1641_164162

theorem average_of_distinct_s_values : 
  (1 + 5 + 2 + 4 + 3 + 3 + 4 + 2 + 5 + 1) / 3 = 7.33 :=
by
  sorry

end average_of_distinct_s_values_l1641_164162


namespace find_positive_integral_solution_l1641_164119

theorem find_positive_integral_solution :
  ∃ n : ℕ, n > 0 ∧ (n - 1) * 101 = (n + 1) * 100 := by
sorry

end find_positive_integral_solution_l1641_164119


namespace compound_interest_eq_440_l1641_164129

-- Define the conditions
variables (P R T SI CI : ℝ)
variables (H_SI : SI = P * R * T / 100)
variables (H_R : R = 20)
variables (H_T : T = 2)
variables (H_given : SI = 400)
variables (H_question : CI = P * (1 + R / 100)^T - P)

-- Define the goal to prove
theorem compound_interest_eq_440 : CI = 440 :=
by
  -- Conditions and the result should be proved here, but we'll use sorry to skip the proof step.
  sorry

end compound_interest_eq_440_l1641_164129


namespace directrix_of_parabola_l1641_164168

theorem directrix_of_parabola (x y : ℝ) : (y^2 = 8*x) → (x = -2) :=
by
  sorry

end directrix_of_parabola_l1641_164168


namespace percentage_of_work_day_in_meetings_is_25_l1641_164150

-- Define the conditions
def workDayHours : ℕ := 9
def firstMeetingMinutes : ℕ := 45
def secondMeetingMinutes : ℕ := 2 * firstMeetingMinutes
def totalMeetingMinutes : ℕ := firstMeetingMinutes + secondMeetingMinutes
def workDayMinutes : ℕ := workDayHours * 60

-- Define the percentage calculation
def percentageOfWorkdaySpentInMeetings : ℕ := (totalMeetingMinutes * 100) / workDayMinutes

-- The theorem to be proven
theorem percentage_of_work_day_in_meetings_is_25 :
  percentageOfWorkdaySpentInMeetings = 25 :=
sorry

end percentage_of_work_day_in_meetings_is_25_l1641_164150


namespace find_d_not_unique_solution_l1641_164172

variable {x y k d : ℝ}

-- Definitions of the conditions
def eq1 (d : ℝ) (x y : ℝ) := 4 * (3 * x + 4 * y) = d
def eq2 (k : ℝ) (x y : ℝ) := k * x + 12 * y = 30

-- The theorem we need to prove
theorem find_d_not_unique_solution (h1: eq1 d x y) (h2: eq2 k x y) (h3 : ¬ ∃! (x y : ℝ), eq1 d x y ∧ eq2 k x y) : d = 40 := 
by
  sorry

end find_d_not_unique_solution_l1641_164172


namespace solution_set_interval_l1641_164103

theorem solution_set_interval (a : ℝ) : 
  {x : ℝ | x^2 - 2*a*x + a^2 - 1 < 0} = {x : ℝ | a - 1 < x ∧ x < a + 1} :=
sorry

end solution_set_interval_l1641_164103


namespace determine_k_l1641_164109

variables (x y z k : ℝ)

theorem determine_k (h1 : (5 / (x - z)) = (k / (y + z))) 
                    (h2 : (k / (y + z)) = (12 / (x + y))) 
                    (h3 : y + z = 2 * x) : 
                    k = 17 := 
by 
  sorry

end determine_k_l1641_164109


namespace problem_I_problem_II_l1641_164189

-- Declaration of function f(x)
def f (x a b : ℝ) := |x + a| - |x - b|

-- Proof 1: When a = 1, b = 1, solve the inequality f(x) > 1
theorem problem_I (x : ℝ) : (f x 1 1) > 1 ↔ x > 1/2 := by
  sorry

-- Proof 2: If the maximum value of the function f(x) is 2, prove that (1/a) + (1/b) ≥ 2
theorem problem_II (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_max_f : ∀ x, f x a b ≤ 2) : 1 / a + 1 / b ≥ 2 := by
  sorry

end problem_I_problem_II_l1641_164189


namespace holiday_price_correct_l1641_164179

-- Define the problem parameters
def original_price : ℝ := 250
def first_discount_rate : ℝ := 0.40
def second_discount_rate : ℝ := 0.10

-- Define the calculation for the first discount
def price_after_first_discount (original: ℝ) (rate: ℝ) : ℝ :=
  original * (1 - rate)

-- Define the calculation for the second discount
def price_after_second_discount (intermediate: ℝ) (rate: ℝ) : ℝ :=
  intermediate * (1 - rate)

-- The final Lean statement to prove
theorem holiday_price_correct : 
  price_after_second_discount (price_after_first_discount original_price first_discount_rate) second_discount_rate = 135 :=
by
  sorry

end holiday_price_correct_l1641_164179


namespace pq_sum_equals_4_l1641_164176

theorem pq_sum_equals_4 (p q : ℝ) (h : (Polynomial.C 1 + Polynomial.C q * Polynomial.X + Polynomial.C p * Polynomial.X^2 + Polynomial.X^4).eval (2 + I) = 0) :
  p + q = 4 :=
sorry

end pq_sum_equals_4_l1641_164176


namespace find_parallel_line_through_point_l1641_164182

-- Definition of a point in Cartesian coordinates
structure Point :=
(x : ℝ)
(y : ℝ)

-- Definition of a line in slope-intercept form
def line (a b c : ℝ) : Prop := ∀ p : Point, a * p.x + b * p.y + c = 0

-- Conditions provided in the problem
def P : Point := ⟨-1, 3⟩
def line1 : Prop := line 1 (-2) 3
def parallel_line (c : ℝ) : Prop := line 1 (-2) c

-- Theorem to prove
theorem find_parallel_line_through_point : parallel_line 7 :=
sorry

end find_parallel_line_through_point_l1641_164182


namespace subset_M_union_N_l1641_164167

theorem subset_M_union_N (M N P : Set ℝ) (f g : ℝ → ℝ)
  (hM : M = {x | f x = 0} ∧ M ≠ ∅)
  (hN : N = {x | g x = 0} ∧ N ≠ ∅)
  (hP : P = {x | f x * g x = 0} ∧ P ≠ ∅) :
  P ⊆ (M ∪ N) := 
sorry

end subset_M_union_N_l1641_164167


namespace solve_log_eq_l1641_164117

noncomputable def log3 (x : ℝ) := Real.log x / Real.log 3

theorem solve_log_eq :
  (∃ x : ℝ, log3 ((5 * x + 15) / (7 * x - 5)) + log3 ((7 * x - 5) / (2 * x - 3)) = 3 ∧ x = 96 / 49) :=
by
  sorry

end solve_log_eq_l1641_164117


namespace equal_share_payments_l1641_164146

theorem equal_share_payments (j n : ℝ) 
  (jack_payment : ℝ := 80) 
  (emma_payment : ℝ := 150) 
  (noah_payment : ℝ := 120)
  (liam_payment : ℝ := 200) 
  (total_cost := jack_payment + emma_payment + noah_payment + liam_payment) 
  (individual_share := total_cost / 4) 
  (jack_due := individual_share - jack_payment) 
  (emma_due := emma_payment - individual_share) 
  (noah_due := individual_share - noah_payment) 
  (liam_due := liam_payment - individual_share) 
  (j := jack_due) 
  (n := noah_due) : 
  j - n = 40 := 
by 
  sorry

end equal_share_payments_l1641_164146


namespace caleb_grandfather_age_l1641_164154

theorem caleb_grandfather_age :
  let yellow_candles := 27
  let red_candles := 14
  let blue_candles := 38
  yellow_candles + red_candles + blue_candles = 79 :=
by
  sorry

end caleb_grandfather_age_l1641_164154


namespace find_expression_value_l1641_164158

theorem find_expression_value 
  (x y : ℝ) 
  (h1 : 4 * x + y = 10) 
  (h2 : x + 4 * y = 18) : 
  16 * x^2 + 24 * x * y + 16 * y^2 = 424 := 
by 
  sorry

end find_expression_value_l1641_164158


namespace john_can_see_jane_for_45_minutes_l1641_164132

theorem john_can_see_jane_for_45_minutes :
  ∀ (john_speed : ℝ) (jane_speed : ℝ) (initial_distance : ℝ) (final_distance : ℝ),
  john_speed = 7 →
  jane_speed = 3 →
  initial_distance = 1 →
  final_distance = 2 →
  (initial_distance / (john_speed - jane_speed) + final_distance / (john_speed - jane_speed)) * 60 = 45 :=
by
  intros john_speed jane_speed initial_distance final_distance
  sorry

end john_can_see_jane_for_45_minutes_l1641_164132


namespace expected_value_of_coins_is_95_5_l1641_164165

-- Define the individual coin values in cents
def penny_value : ℕ := 1
def nickel_value : ℕ := 5
def dime_value : ℕ := 10
def quarter_value : ℕ := 25
def fifty_cent_value : ℕ := 50
def dollar_value : ℕ := 100

-- Expected value function with 1/2 probability 
def expected_value (coin_value : ℕ) : ℚ := (coin_value : ℚ) / 2

-- Calculate the total expected value of all coins flipped
noncomputable def total_expected_value : ℚ :=
  expected_value penny_value +
  expected_value nickel_value +
  expected_value dime_value +
  expected_value quarter_value +
  expected_value fifty_cent_value +
  expected_value dollar_value

-- Prove that the expected total value is 95.5
theorem expected_value_of_coins_is_95_5 :
  total_expected_value = 95.5 := by
  sorry

end expected_value_of_coins_is_95_5_l1641_164165


namespace brenda_age_l1641_164137

variables (A B J : ℝ)

-- Conditions
def condition1 : Prop := A = 4 * B
def condition2 : Prop := J = B + 7
def condition3 : Prop := A = J

-- Target to prove
theorem brenda_age (h1 : condition1 A B) (h2 : condition2 B J) (h3 : condition3 A J) : B = 7 / 3 :=
by
  sorry

end brenda_age_l1641_164137


namespace polygon_interior_angles_540_implies_5_sides_l1641_164142

theorem polygon_interior_angles_540_implies_5_sides (n : ℕ) :
  (n - 2) * 180 = 540 → n = 5 :=
by
  sorry

end polygon_interior_angles_540_implies_5_sides_l1641_164142


namespace parabola_translation_correct_l1641_164178

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

end parabola_translation_correct_l1641_164178


namespace each_baby_worms_per_day_l1641_164136

variable (babies : Nat) (worms_papa : Nat) (worms_mama_caught : Nat) (worms_mama_stolen : Nat) (worms_needed : Nat)
variable (days : Nat)

theorem each_baby_worms_per_day 
  (h1 : babies = 6) 
  (h2 : worms_papa = 9) 
  (h3 : worms_mama_caught = 13) 
  (h4 : worms_mama_stolen = 2)
  (h5 : worms_needed = 34) 
  (h6 : days = 3) :
  (worms_papa + (worms_mama_caught - worms_mama_stolen) + worms_needed) / babies / days = 3 :=
by
  sorry

end each_baby_worms_per_day_l1641_164136


namespace Crimson_Valley_skirts_l1641_164151

theorem Crimson_Valley_skirts
  (Azure_Valley_skirts : ℕ)
  (Seafoam_Valley_skirts : ℕ)
  (Purple_Valley_skirts : ℕ)
  (Crimson_Valley_skirts : ℕ)
  (h1 : Azure_Valley_skirts = 90)
  (h2 : Seafoam_Valley_skirts = (2/3 : ℚ) * Azure_Valley_skirts)
  (h3 : Purple_Valley_skirts = (1/4 : ℚ) * Seafoam_Valley_skirts)
  (h4 : Crimson_Valley_skirts = (1/3 : ℚ) * Purple_Valley_skirts)
  : Crimson_Valley_skirts = 5 := 
sorry

end Crimson_Valley_skirts_l1641_164151


namespace side_length_of_square_l1641_164127

theorem side_length_of_square (P : ℝ) (h1 : P = 12 / 25) : 
  P / 4 = 0.12 := 
by
  sorry

end side_length_of_square_l1641_164127


namespace number_of_correct_answers_l1641_164107

theorem number_of_correct_answers (c w : ℕ) (h1 : c + w = 60) (h2 : 4 * c - w = 110) : c = 34 :=
by
  -- placeholder for proof
  sorry

end number_of_correct_answers_l1641_164107


namespace arun_age_proof_l1641_164135

theorem arun_age_proof {A G M : ℕ} 
  (h1 : (A - 6) / 18 = G)
  (h2 : G = M - 2)
  (h3 : M = 5) :
  A = 60 :=
by
  sorry

end arun_age_proof_l1641_164135


namespace distance_to_fourth_side_l1641_164147

theorem distance_to_fourth_side (s : ℕ) (d1 d2 d3 : ℕ) (x : ℕ) 
  (cond1 : d1 = 4) (cond2 : d2 = 7) (cond3 : d3 = 12)
  (h : d1 + d2 + d3 + x = s) : x = 9 ∨ x = 15 :=
  sorry

end distance_to_fourth_side_l1641_164147


namespace weight_gain_ratio_l1641_164106

variable (J O F : ℝ)

theorem weight_gain_ratio :
  O = 5 ∧ F = (1/2) * J - 3 ∧ 5 + J + F = 20 → J / O = 12 / 5 :=
by
  intros h
  cases' h with hO h'
  cases' h' with hF hTotal
  sorry

end weight_gain_ratio_l1641_164106


namespace sticker_height_enlarged_l1641_164128

theorem sticker_height_enlarged (orig_width orig_height new_width : ℝ)
    (h1 : orig_width = 3) (h2 : orig_height = 2) (h3 : new_width = 12) :
    new_width / orig_width * orig_height = 8 :=
by
  rw [h1, h2, h3]
  norm_num

end sticker_height_enlarged_l1641_164128


namespace largest_possible_c_l1641_164187

theorem largest_possible_c (c : ℝ) (hc : (3 * c + 4) * (c - 2) = 9 * c) : c ≤ 4 :=
sorry

end largest_possible_c_l1641_164187


namespace ordered_pairs_count_l1641_164194

theorem ordered_pairs_count :
  (∃ (A B : ℕ), 0 < A ∧ 0 < B ∧ A % 2 = 0 ∧ B % 2 = 0 ∧ (A / 8) = (8 / B))
  → (∃ (n : ℕ), n = 5) :=
by {
  sorry
}

end ordered_pairs_count_l1641_164194


namespace find_second_quadrant_point_l1641_164130

def is_second_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 > 0

theorem find_second_quadrant_point :
  (is_second_quadrant (2, 3) = false) ∧
  (is_second_quadrant (2, -3) = false) ∧
  (is_second_quadrant (-2, -3) = false) ∧
  (is_second_quadrant (-2, 3) = true) := 
sorry

end find_second_quadrant_point_l1641_164130


namespace multiply_res_l1641_164120

theorem multiply_res (
  h : 213 * 16 = 3408
) : 1.6 * 213 = 340.8 :=
sorry

end multiply_res_l1641_164120


namespace angle_R_values_l1641_164173

theorem angle_R_values (P Q : ℝ) (h1: 5 * Real.sin P + 2 * Real.cos Q = 5) (h2: 2 * Real.sin Q + 5 * Real.cos P = 3) : 
  ∃ R : ℝ, R = Real.arcsin (1/20) ∨ R = 180 - Real.arcsin (1/20) :=
by
  sorry

end angle_R_values_l1641_164173


namespace accounting_major_students_count_l1641_164180

theorem accounting_major_students_count (p q r s: ℕ) (h1: p * q * r * s = 1365) (h2: 1 < p) (h3: p < q) (h4: q < r) (h5: r < s):
  p = 3 :=
sorry

end accounting_major_students_count_l1641_164180


namespace dice_sum_probability_l1641_164133

def four_dice_probability_sum_to_remain_die : ℚ :=
  let total_outcomes : ℚ := 6^4
  let favorable_outcomes : ℚ := 4 * 120
  favorable_outcomes / total_outcomes

theorem dice_sum_probability : four_dice_probability_sum_to_remain_die = 10 / 27 :=
  sorry

end dice_sum_probability_l1641_164133


namespace carpet_dimensions_l1641_164148
open Real

theorem carpet_dimensions (x y : ℝ) 
  (h1 : x > 0)
  (h2 : y > 0)
  (h3 : ∃ k: ℝ, y = k * x)
  (h4 : ∃ α β: ℝ, α + k * β = 50 ∧ k * α + β = 55)
  (h5 : ∃ γ δ: ℝ, γ + k * δ = 38 ∧ k * γ + δ = 55) :
  x = 25 ∧ y = 50 :=
by sorry

end carpet_dimensions_l1641_164148


namespace sufficient_condition_l1641_164169

theorem sufficient_condition (a b : ℝ) (h : b > a ∧ a > 0) : (a + 2) / (b + 2) > a / b :=
by sorry

end sufficient_condition_l1641_164169


namespace inscribed_circle_radius_l1641_164159

theorem inscribed_circle_radius
  (A p s : ℝ) (h1 : A = p) (h2 : s = p / 2) (r : ℝ) (h3 : A = r * s) :
  r = 2 :=
sorry

end inscribed_circle_radius_l1641_164159


namespace bert_ernie_ratio_l1641_164155

theorem bert_ernie_ratio (berts_stamps ernies_stamps peggys_stamps : ℕ) 
  (h1 : peggys_stamps = 75) 
  (h2 : ernies_stamps = 3 * peggys_stamps) 
  (h3 : berts_stamps = peggys_stamps + 825) : 
  berts_stamps / ernies_stamps = 4 := 
by sorry

end bert_ernie_ratio_l1641_164155


namespace min_num_cuboids_l1641_164191

/-
Definitions based on the conditions:
- Dimensions of the cuboid are given as 3 cm, 4 cm, and 5 cm.
- We need to find the Least Common Multiple (LCM) of these dimensions.
- Calculate the volume of the smallest cube.
- Calculate the volume of the given cuboid.
- Find the number of such cuboids needed to form the cube.
-/
def cuboid_length : ℤ := 3
def cuboid_width : ℤ := 4
def cuboid_height : ℤ := 5

noncomputable def lcm_3_4_5 : ℤ := Int.lcm (Int.lcm cuboid_length cuboid_width) cuboid_height

noncomputable def cube_side_length : ℤ := lcm_3_4_5
noncomputable def cube_volume : ℤ := cube_side_length * cube_side_length * cube_side_length
noncomputable def cuboid_volume : ℤ := cuboid_length * cuboid_width * cuboid_height

noncomputable def num_cuboids : ℤ := cube_volume / cuboid_volume

theorem min_num_cuboids :
  num_cuboids = 3600 := by
  sorry

end min_num_cuboids_l1641_164191

import Mathlib

namespace transportation_inverse_proportion_l1264_126458

theorem transportation_inverse_proportion (V t : ℝ) (h: V * t = 10^5) : V = 10^5 / t :=
by
  sorry

end transportation_inverse_proportion_l1264_126458


namespace remainder_17_pow_2037_mod_20_l1264_126466

theorem remainder_17_pow_2037_mod_20:
      (17^1) % 20 = 17 ∧
      (17^2) % 20 = 9 ∧
      (17^3) % 20 = 13 ∧
      (17^4) % 20 = 1 → 
      (17^2037) % 20 = 17 := sorry

end remainder_17_pow_2037_mod_20_l1264_126466


namespace chemistry_class_size_l1264_126472

theorem chemistry_class_size
  (total_students : ℕ)
  (chem_bio_both : ℕ)
  (bio_students : ℕ)
  (chem_students : ℕ)
  (both_students : ℕ)
  (H1 : both_students = 8)
  (H2 : bio_students + chem_students + both_students = total_students)
  (H3 : total_students = 70)
  (H4 : chem_students = 2 * (bio_students + both_students)) :
  chem_students + both_students = 52 :=
by
  sorry

end chemistry_class_size_l1264_126472


namespace min_value_at_2_l1264_126457

noncomputable def min_value (x : ℝ) := x + 4 / x + 5

theorem min_value_at_2 (x : ℝ) (h : x > 0) : min_value x ≥ 9 :=
sorry

end min_value_at_2_l1264_126457


namespace cos_beta_eq_neg_16_over_65_l1264_126401

variable (α β : ℝ)
variable (h1 : 0 < α ∧ α < π)
variable (h2 : 0 < β ∧ β < π)
variable (h3 : Real.sin β = 5 / 13)
variable (h4 : Real.tan (α / 2) = 1 / 2)

theorem cos_beta_eq_neg_16_over_65 : Real.cos β = -16 / 65 := by
  sorry

end cos_beta_eq_neg_16_over_65_l1264_126401


namespace sufficient_p_wages_l1264_126476

variable (S P Q : ℕ)

theorem sufficient_p_wages (h1 : S = 40 * Q) (h2 : S = 15 * (P + Q))  :
  ∃ D : ℕ, S = D * P ∧ D = 24 := 
by
  use 24
  sorry

end sufficient_p_wages_l1264_126476


namespace tetrad_does_not_have_four_chromosomes_l1264_126496

noncomputable def tetrad_has_two_centromeres : Prop := -- The condition: a tetrad has two centromeres
  sorry

noncomputable def tetrad_contains_four_dna_molecules : Prop := -- The condition: a tetrad contains four DNA molecules
  sorry

noncomputable def tetrad_consists_of_two_pairs_of_sister_chromatids : Prop := -- The condition: a tetrad consists of two pairs of sister chromatids
  sorry

theorem tetrad_does_not_have_four_chromosomes 
  (h1: tetrad_has_two_centromeres)
  (h2: tetrad_contains_four_dna_molecules)
  (h3: tetrad_consists_of_two_pairs_of_sister_chromatids) 
  : ¬ (tetrad_has_four_chromosomes : Prop) :=
sorry

end tetrad_does_not_have_four_chromosomes_l1264_126496


namespace min_negative_numbers_l1264_126471

theorem min_negative_numbers (a b c d : ℤ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : d ≠ 0) 
  (h5 : a + b + c < d) (h6 : a + b + d < c) (h7 : a + c + d < b) (h8 : b + c + d < a) :
  3 ≤ (if a < 0 then 1 else 0) + (if b < 0 then 1 else 0) + (if c < 0 then 1 else 0) + (if d < 0 then 1 else 0) := 
sorry

end min_negative_numbers_l1264_126471


namespace three_liters_to_gallons_l1264_126427

theorem three_liters_to_gallons :
  (0.5 : ℝ) * 3 * 0.1319 = 0.7914 := by
  sorry

end three_liters_to_gallons_l1264_126427


namespace solution_set_l1264_126469

-- Defining the condition and inequalities:
variable (a x : Real)

-- Condition that a < 0
def condition_a : Prop := a < 0

-- Inequalities in the system
def inequality1 : Prop := x > -2 * a
def inequality2 : Prop := x > 3 * a

-- The solution set we need to prove
theorem solution_set (h : condition_a a) : (inequality1 a x) ∧ (inequality2 a x) ↔ x > -2 * a :=
by
  sorry

end solution_set_l1264_126469


namespace find_positive_number_l1264_126487

-- The definition to state the given condition
def condition1 (n : ℝ) : Prop := n > 0 ∧ n^2 + n = 245

-- The theorem stating the problem and its solution
theorem find_positive_number (n : ℝ) (h : condition1 n) : n = 14 :=
by sorry

end find_positive_number_l1264_126487


namespace negq_sufficient_but_not_necessary_for_p_l1264_126447

variable (p q : Prop)

theorem negq_sufficient_but_not_necessary_for_p
  (h1 : ¬p → q)
  (h2 : ¬(¬q → p)) :
  (¬q → p) ∧ ¬(p → ¬q) :=
sorry

end negq_sufficient_but_not_necessary_for_p_l1264_126447


namespace not_sqrt2_rational_union_eq_intersection_implies_equal_intersection_eq_b_subset_a_element_in_both_implies_in_intersection_l1264_126453

-- Definitions
def is_rational (x : ℝ) : Prop := ∃ a b : ℤ, b ≠ 0 ∧ (x = a / b)
def union (A B : Set α) : Set α := {x | x ∈ A ∨ x ∈ B}
def intersection (A B : Set α) : Set α := {x | x ∈ A ∧ x ∈ B}
def subset (A B : Set α) : Prop := ∀ x, x ∈ A → x ∈ B

-- Statement A
theorem not_sqrt2_rational : ¬ is_rational (Real.sqrt 2) :=
sorry

-- Statement B
theorem union_eq_intersection_implies_equal {α : Type*} {A B : Set α}
  (h : union A B = intersection A B) : A = B :=
sorry

-- Statement C
theorem intersection_eq_b_subset_a {α : Type*} {A B : Set α}
  (h : intersection A B = B) : subset B A :=
sorry

-- Statement D
theorem element_in_both_implies_in_intersection {α : Type*} {A B : Set α} {a : α}
  (haA : a ∈ A) (haB : a ∈ B) : a ∈ intersection A B :=
sorry

end not_sqrt2_rational_union_eq_intersection_implies_equal_intersection_eq_b_subset_a_element_in_both_implies_in_intersection_l1264_126453


namespace find_value_of_k_l1264_126404

def line_equation_holds (m n : ℤ) : Prop := m = 2 * n + 5
def second_point_condition (m n k : ℤ) : Prop := m + 4 = 2 * (n + k) + 5

theorem find_value_of_k (m n k : ℤ) 
  (h1 : line_equation_holds m n) 
  (h2 : second_point_condition m n k) : 
  k = 2 :=
by sorry

end find_value_of_k_l1264_126404


namespace revenue_increase_l1264_126416

theorem revenue_increase
  (P Q : ℝ)
  (h : 0 < P)
  (hQ : 0 < Q)
  (price_decrease : 0.90 = 0.90)
  (unit_increase : 2 = 2) :
  (0.90 * P) * (2 * Q) = 1.80 * (P * Q) :=
by
  sorry

end revenue_increase_l1264_126416


namespace solve_for_n_l1264_126467

theorem solve_for_n (n : ℤ) (h : (1 : ℤ) / (n + 2) + 2 / (n + 2) + n / (n + 2) = 2) : n = 2 :=
sorry

end solve_for_n_l1264_126467


namespace negation_example_l1264_126438

theorem negation_example :
  ¬(∀ x : ℝ, x > 0 → x^2 - x ≤ 0) ↔ (∃ x₀ : ℝ, x₀ > 0 ∧ x₀^2 - x₀ > 0) :=
by
  sorry

end negation_example_l1264_126438


namespace arithmetic_expression_equals_47_l1264_126442

-- Define the arithmetic expression
def arithmetic_expression : ℕ :=
  2 + 5 * 3^2 - 4 + 6 * 2 / 3

-- The proof goal: arithmetic_expression equals 47
theorem arithmetic_expression_equals_47 : arithmetic_expression = 47 := 
by
  sorry

end arithmetic_expression_equals_47_l1264_126442


namespace inequality_proof_l1264_126407

theorem inequality_proof (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : ¬ (a + d > b + c) := sorry

end inequality_proof_l1264_126407


namespace subset_intersection_exists_l1264_126499

theorem subset_intersection_exists {n : ℕ} (A : Fin (n + 1) → Finset (Fin n)) 
    (h_distinct : ∀ i j : Fin (n + 1), i ≠ j → A i ≠ A j)
    (h_size : ∀ i : Fin (n + 1), (A i).card = 3) : 
    ∃ (i j : Fin (n + 1)), i ≠ j ∧ (A i ∩ A j).card = 1 :=
by
  sorry

end subset_intersection_exists_l1264_126499


namespace volume_diff_proof_l1264_126421

def volume_difference (x y z x' y' z' : ℝ) : ℝ := x * y * z - x' * y' * z'

theorem volume_diff_proof : 
  (∃ (x y z x' y' z' : ℝ),
    2 * (x + y) = 12 ∧ 2 * (x + z) = 16 ∧ 2 * (y + z) = 24 ∧
    2 * (x' + y') = 12 ∧ 2 * (x' + z') = 16 ∧ 2 * (y' + z') = 20 ∧
    volume_difference x y z x' y' z' = -13) :=
by {
  sorry
}

end volume_diff_proof_l1264_126421


namespace frank_eats_each_day_l1264_126403

theorem frank_eats_each_day :
  ∀ (cookies_per_tray cookies_per_day days ted_eats remaining_cookies : ℕ),
  cookies_per_tray = 12 →
  cookies_per_day = 2 →
  days = 6 →
  ted_eats = 4 →
  remaining_cookies = 134 →
  (2 * cookies_per_tray * days) - (ted_eats + remaining_cookies) / days = 1 :=
  by
    intros cookies_per_tray cookies_per_day days ted_eats remaining_cookies ht hc hd hted hr
    sorry

end frank_eats_each_day_l1264_126403


namespace delivery_driver_stops_l1264_126451

theorem delivery_driver_stops (initial_stops more_stops total_stops : ℕ)
  (h_initial : initial_stops = 3)
  (h_more : more_stops = 4)
  (h_total : total_stops = initial_stops + more_stops) : total_stops = 7 := by
  sorry

end delivery_driver_stops_l1264_126451


namespace line_equation_l1264_126450

theorem line_equation :
  ∃ m b, m = 1 ∧ b = 5 ∧ (∀ x y, y = m * x + b ↔ x - y + 5 = 0) :=
by
  sorry

end line_equation_l1264_126450


namespace max_expr_on_circle_l1264_126494

noncomputable def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 + 4 * x - 6 * y + 4 = 0

noncomputable def expr (x y : ℝ) : ℝ :=
  3 * x - 4 * y

theorem max_expr_on_circle : 
  ∃ (x y : ℝ), circle_eq x y ∧ ∀ (x' y' : ℝ), circle_eq x' y' → expr x y ≤ expr x' y' :=
sorry

end max_expr_on_circle_l1264_126494


namespace geometric_seq_a7_l1264_126433

theorem geometric_seq_a7 (a : ℕ → ℤ) (q : ℤ)
  (h_geom : ∀ n, a (n + 1) = a n * q)
  (h1 : a 2 * a 4 * a 5 = a 3 * a 6)
  (h2 : a 9 * a 10 = -8) : a 7 = -2 :=
by
  sorry

end geometric_seq_a7_l1264_126433


namespace divisibility_by_37_l1264_126405

theorem divisibility_by_37 (a b c : ℕ) :
  (100 * a + 10 * b + c) % 37 = 0 → 
  (100 * b + 10 * c + a) % 37 = 0 ∧
  (100 * c + 10 * a + b) % 37 = 0 :=
by
  sorry

end divisibility_by_37_l1264_126405


namespace cows_C_grazed_l1264_126410

/-- Define the conditions for each milkman’s cow-months. -/
def A_cow_months := 24 * 3
def B_cow_months := 10 * 5
def D_cow_months := 21 * 3
def C_cow_months (x : ℕ) := x * 4

/-- Define the cost per cow-month based on A's share. -/
def cost_per_cow_month := 720 / A_cow_months

/-- Define the total rent. -/
def total_rent := 3250

/-- Define the total cow-months including C's cow-months as a variable. -/
def total_cow_months (x : ℕ) := A_cow_months + B_cow_months + C_cow_months x + D_cow_months

/-- Lean 4 statement to prove the number of cows C grazed. -/
theorem cows_C_grazed (x : ℕ) :
  total_rent = total_cow_months x * cost_per_cow_month → x = 35 := by {
  sorry
}

end cows_C_grazed_l1264_126410


namespace find_prices_max_sets_of_go_compare_options_l1264_126489

theorem find_prices (x y : ℕ) (h1 : 2 * x + 3 * y = 140) (h2 : 4 * x + y = 130) :
  x = 25 ∧ y = 30 :=
by sorry

theorem max_sets_of_go (m : ℕ) (h3 : 25 * (80 - m) + 30 * m ≤ 2250) :
  m ≤ 50 :=
by sorry

theorem compare_options (a : ℕ) :
  (a < 10 → 27 * a < 21 * a + 60) ∧ (a = 10 → 27 * a = 21 * a + 60) ∧ (a > 10 → 27 * a > 21 * a + 60) :=
by sorry

end find_prices_max_sets_of_go_compare_options_l1264_126489


namespace compare_f_values_max_f_value_l1264_126402

noncomputable def f (x : ℝ) : ℝ := -2 * Real.sin x - Real.cos (2 * x)

theorem compare_f_values :
  f (Real.pi / 4) > f (Real.pi / 6) :=
sorry

theorem max_f_value :
  ∃ x : ℝ, f x = 3 :=
sorry

end compare_f_values_max_f_value_l1264_126402


namespace probability_of_real_roots_is_correct_l1264_126420

open Real

def has_real_roots (m : ℝ) : Prop :=
  2 * m^2 - 8 ≥ 0 

def favorable_set : Set ℝ := {m | has_real_roots m}

def interval_length (a b : ℝ) : ℝ := b - a

noncomputable def probability_of_real_roots : ℝ :=
  interval_length (-4) (-2) + interval_length 2 3 / interval_length (-4) 3

theorem probability_of_real_roots_is_correct : probability_of_real_roots = 3 / 7 :=
by
  sorry

end probability_of_real_roots_is_correct_l1264_126420


namespace total_matches_l1264_126498

theorem total_matches (home_wins home_draws home_losses rival_wins rival_draws rival_losses : ℕ)
  (H_home_wins : home_wins = 3)
  (H_home_draws : home_draws = 4)
  (H_home_losses : home_losses = 0)
  (H_rival_wins : rival_wins = 2 * home_wins)
  (H_rival_draws : rival_draws = 4)
  (H_rival_losses : rival_losses = 0) :
  home_wins + home_draws + home_losses + rival_wins + rival_draws + rival_losses = 17 :=
by
  sorry

end total_matches_l1264_126498


namespace ken_pencils_kept_l1264_126475

-- Define the known quantities and conditions
def initial_pencils : ℕ := 250
def manny_pencils : ℕ := 25
def nilo_pencils : ℕ := manny_pencils * 2
def carlos_pencils : ℕ := nilo_pencils / 2
def tina_pencils : ℕ := carlos_pencils + 10
def rina_pencils : ℕ := tina_pencils - 20

-- Formulate the total pencils given away
def total_given_away : ℕ :=
  manny_pencils + nilo_pencils + carlos_pencils + tina_pencils + rina_pencils

-- Prove the final number of pencils Ken kept.
theorem ken_pencils_kept : initial_pencils - total_given_away = 100 :=
by
  sorry

end ken_pencils_kept_l1264_126475


namespace cannon_hit_probability_l1264_126452

theorem cannon_hit_probability
  (P1 P2 P3 : ℝ)
  (h1 : P1 = 0.2)
  (h3 : P3 = 0.3)
  (h_none_hit : (1 - P1) * (1 - P2) * (1 - P3) = 0.27999999999999997) :
  P2 = 0.5 :=
by
  sorry

end cannon_hit_probability_l1264_126452


namespace cube_property_l1264_126423

theorem cube_property (x : ℝ) (s : ℝ) 
  (h1 : s^3 = 8 * x)
  (h2 : 6 * s^2 = 4 * x) :
  x = 5400 :=
by
  sorry

end cube_property_l1264_126423


namespace determinant_inequality_l1264_126461

variable (x : ℝ)

def determinant (a b c d : ℝ) : ℝ :=
  a * d - b * c

theorem determinant_inequality (h : determinant 2 (3 - x) 1 x > 0) : x > 1 := by
  sorry

end determinant_inequality_l1264_126461


namespace arithmetic_sequence_l1264_126492

theorem arithmetic_sequence (n : ℕ) (a : ℕ → ℤ) (h : ∀ n, a n = 3 * n + 1) : 
  ∀ n, a (n + 1) - a n = 3 := by
  sorry

end arithmetic_sequence_l1264_126492


namespace reciprocal_expression_l1264_126435

theorem reciprocal_expression :
  (1 / ((1 / 4 : ℚ) + (1 / 5 : ℚ)) / (1 / 3)) = (20 / 27 : ℚ) :=
by
  sorry

end reciprocal_expression_l1264_126435


namespace scott_monthly_miles_l1264_126468

theorem scott_monthly_miles :
  let miles_per_mon_wed := 3
  let mon_wed_days := 3
  let thur_fri_factor := 2
  let thur_fri_days := 2
  let weeks_per_month := 4
  let miles_mon_wed := miles_per_mon_wed * mon_wed_days
  let miles_thur_fri_per_day := thur_fri_factor * miles_per_mon_wed
  let miles_thur_fri := miles_thur_fri_per_day * thur_fri_days
  let miles_per_week := miles_mon_wed + miles_thur_fri
  let total_miles_in_month := miles_per_week * weeks_per_month
  total_miles_in_month = 84 := 
  by
    sorry

end scott_monthly_miles_l1264_126468


namespace toothpick_problem_l1264_126428

theorem toothpick_problem : 
  ∃ (N : ℕ), N > 5000 ∧ 
            N % 10 = 9 ∧ 
            N % 9 = 8 ∧ 
            N % 8 = 7 ∧ 
            N % 7 = 6 ∧ 
            N % 6 = 5 ∧ 
            N % 5 = 4 ∧ 
            N = 5039 :=
by
  sorry

end toothpick_problem_l1264_126428


namespace create_proper_six_sided_figure_l1264_126473

-- Definition of a matchstick configuration
structure MatchstickConfig where
  sides : ℕ
  matchsticks : ℕ

-- Initial configuration: a regular hexagon with 6 matchsticks
def initialConfig : MatchstickConfig := ⟨6, 6⟩

-- Condition: Cannot lay any stick on top of another, no free ends
axiom no_overlap (cfg : MatchstickConfig) : Prop
axiom no_free_ends (cfg : MatchstickConfig) : Prop

-- New configuration after adding 3 matchsticks
def newConfig : MatchstickConfig := ⟨6, 9⟩

-- Theorem stating the possibility to create a proper figure with six sides
theorem create_proper_six_sided_figure : no_overlap newConfig → no_free_ends newConfig → newConfig.sides = 6 :=
by
  sorry

end create_proper_six_sided_figure_l1264_126473


namespace min_ab_min_inv_a_plus_2_inv_b_max_sqrt_2a_plus_sqrt_b_not_max_a_plus_1_times_b_plus_1_l1264_126477

-- Condition definitions
variable {a b : ℝ}
variable (h1 : 0 < a) (h2 : 0 < b) (h3 : 2 * a + b = 1)

-- Minimum value of ab is 1/8
theorem min_ab (h1 : 0 < a) (h2 : 0 < b) (h3 : 2 * a + b = 1) : ∃ y, y = (a * b) ∧ y = 1 / 8 := by
  sorry

-- Minimum value of 1/a + 2/b is 8
theorem min_inv_a_plus_2_inv_b (h1 : 0 < a) (h2 : 0 < b) (h3 : 2 * a + b = 1) : ∃ y, y = (1 / a + 2 / b) ∧ y = 8 := by
  sorry

-- Maximum value of sqrt(2a) + sqrt(b) is sqrt(2)
theorem max_sqrt_2a_plus_sqrt_b (h1 : 0 < a) (h2 : 0 < b) (h3 : 2 * a + b = 1) : ∃ y, y = (Real.sqrt (2 * a) + Real.sqrt b) ∧ y = Real.sqrt 2 := by
  sorry

-- Maximum value of (a+1)(b+1) is not 2
theorem not_max_a_plus_1_times_b_plus_1 (h1 : 0 < a) (h2 : 0 < b) (h3 : 2 * a + b = 1) : ∃ y, y = ((a + 1) * (b + 1)) ∧ y ≠ 2 := by
  sorry


end min_ab_min_inv_a_plus_2_inv_b_max_sqrt_2a_plus_sqrt_b_not_max_a_plus_1_times_b_plus_1_l1264_126477


namespace inequality_proof_l1264_126432

theorem inequality_proof (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  (x / Real.sqrt y + y / Real.sqrt x) ≥ (Real.sqrt x + Real.sqrt y) := 
sorry

end inequality_proof_l1264_126432


namespace parabola_c_value_l1264_126431

theorem parabola_c_value (a b c : ℝ) (h1 : 3 = a * (-1)^2 + b * (-1) + c)
  (h2 : 1 = a * (-2)^2 + b * (-2) + c) : c = 1 :=
sorry

end parabola_c_value_l1264_126431


namespace intersection_A_B_l1264_126439

variable (A : Set ℤ) (B : Set ℤ)

-- Define the set A and B
def set_A : Set ℤ := {0, 1, 2}
def set_B : Set ℤ := {x | 1 < x ∧ x < 4}

theorem intersection_A_B :
  set_A ∩ set_B = {2} :=
by
  sorry

end intersection_A_B_l1264_126439


namespace initial_kittens_l1264_126460

theorem initial_kittens (x : ℕ) (h : x + 3 = 9) : x = 6 :=
by {
  sorry
}

end initial_kittens_l1264_126460


namespace square_area_l1264_126485

theorem square_area (side_length : ℝ) (h : side_length = 10) : side_length * side_length = 100 := by
  sorry

end square_area_l1264_126485


namespace value_this_year_l1264_126417

def last_year_value : ℝ := 20000
def depreciation_factor : ℝ := 0.8

theorem value_this_year :
  last_year_value * depreciation_factor = 16000 :=
by
  sorry

end value_this_year_l1264_126417


namespace price_per_gaming_chair_l1264_126445

theorem price_per_gaming_chair 
  (P : ℝ)
  (price_per_organizer : ℝ := 78)
  (num_organizers : ℕ := 3)
  (num_chairs : ℕ := 2)
  (total_paid : ℝ := 420)
  (delivery_fee_rate : ℝ := 0.05) 
  (cost_organizers : ℝ := num_organizers * price_per_organizer)
  (cost_gaming_chairs : ℝ := num_chairs * P)
  (total_sales : ℝ := cost_organizers + cost_gaming_chairs)
  (delivery_fee : ℝ := delivery_fee_rate * total_sales) :
  total_paid = total_sales + delivery_fee → P = 83 := 
sorry

end price_per_gaming_chair_l1264_126445


namespace value_of_af_over_cd_l1264_126481

variable (a b c d e f : ℝ)

theorem value_of_af_over_cd :
  a * b * c = 130 ∧
  b * c * d = 65 ∧
  c * d * e = 500 ∧
  d * e * f = 250 →
  (a * f) / (c * d) = 1 :=
by
  sorry

end value_of_af_over_cd_l1264_126481


namespace number_of_bonnies_l1264_126424

theorem number_of_bonnies (B blueberries apples : ℝ) 
  (h1 : blueberries = 3 / 4 * B) 
  (h2 : apples = 3 * blueberries)
  (h3 : B + blueberries + apples = 240) : 
  B = 60 :=
by
  sorry

end number_of_bonnies_l1264_126424


namespace eleanor_distance_between_meetings_l1264_126414

-- Conditions given in the problem
def track_length : ℕ := 720
def eric_time : ℕ := 4
def eleanor_time : ℕ := 5
def eric_speed : ℕ := track_length / eric_time
def eleanor_speed : ℕ := track_length / eleanor_time
def relative_speed : ℕ := eric_speed + eleanor_speed
def time_to_meet : ℚ := track_length / relative_speed

-- Proof task: prove that the distance Eleanor runs between consective meetings is 320 meters.
theorem eleanor_distance_between_meetings : eleanor_speed * time_to_meet = 320 := by
  sorry

end eleanor_distance_between_meetings_l1264_126414


namespace taxi_ride_cost_l1264_126455

-- Definitions
def initial_fee : Real := 2
def distance_in_miles : Real := 4
def cost_per_mile : Real := 2.5

-- Theorem statement
theorem taxi_ride_cost (initial_fee : Real) (distance_in_miles : Real) (cost_per_mile : Real) : 
  initial_fee + distance_in_miles * cost_per_mile = 12 := 
by
  sorry

end taxi_ride_cost_l1264_126455


namespace color_of_face_opposite_blue_l1264_126443

/-- Assume we have a cube with each face painted in distinct colors. -/
structure Cube where
  top : String
  front : String
  right_side : String
  back : String
  left_side : String
  bottom : String

/-- Given three views of a colored cube, determine the color of the face opposite the blue face. -/
theorem color_of_face_opposite_blue (c : Cube)
  (h_top : c.top = "R")
  (h_right : c.right_side = "G")
  (h_view1 : c.front = "W")
  (h_view2 : c.front = "O")
  (h_view3 : c.front = "Y") :
  c.back = "Y" :=
sorry

end color_of_face_opposite_blue_l1264_126443


namespace smallest_n_for_quadratic_factorization_l1264_126470

theorem smallest_n_for_quadratic_factorization :
  ∃ (n : ℤ), (∀ A B : ℤ, A * B = 50 → n = 5 * B + A) ∧ (∀ m : ℤ, 
    (∀ A B : ℤ, A * B = 50 → m ≤ 5 * B + A) → n ≤ m) :=
by
  sorry

end smallest_n_for_quadratic_factorization_l1264_126470


namespace total_weight_loss_l1264_126465

def seth_loss : ℝ := 17.53
def jerome_loss : ℝ := 3 * seth_loss
def veronica_loss : ℝ := seth_loss + 1.56
def seth_veronica_loss : ℝ := seth_loss + veronica_loss
def maya_loss : ℝ := seth_veronica_loss - 0.25 * seth_veronica_loss
def total_loss : ℝ := seth_loss + jerome_loss + veronica_loss + maya_loss

theorem total_weight_loss : total_loss = 116.675 := by
  sorry

end total_weight_loss_l1264_126465


namespace solution_set_eq_2m_add_2_gt_zero_l1264_126409

theorem solution_set_eq_2m_add_2_gt_zero {m : ℝ} (h : ∀ x : ℝ, mx + 2 > 0 ↔ x < 2) : m = -1 :=
sorry

end solution_set_eq_2m_add_2_gt_zero_l1264_126409


namespace wang_pens_purchase_l1264_126482

theorem wang_pens_purchase :
  ∀ (total_money spent_on_albums pen_cost : ℝ)
  (number_of_pens : ℕ),
  total_money = 80 →
  spent_on_albums = 45.6 →
  pen_cost = 2.5 →
  number_of_pens = 13 →
  (total_money - spent_on_albums) / pen_cost ≥ number_of_pens ∧ 
  (total_money - spent_on_albums) / pen_cost < number_of_pens + 1 :=
by
  intros
  sorry

end wang_pens_purchase_l1264_126482


namespace segment_length_C_C_l1264_126446

-- Define the points C and C''.
def C : ℝ × ℝ := (-3, 2)
def C'' : ℝ × ℝ := (-3, -2)

-- State the theorem that the length of the segment from C to C'' is 4.
theorem segment_length_C_C'' : dist C C'' = 4 := by
  sorry

end segment_length_C_C_l1264_126446


namespace remainder_of_sum_l1264_126479

theorem remainder_of_sum :
  ((88134 + 88135 + 88136 + 88137 + 88138 + 88139) % 9) = 6 :=
by
  sorry

end remainder_of_sum_l1264_126479


namespace hyperbola_asymptotes_l1264_126495

theorem hyperbola_asymptotes (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
                              (h3 : 2 * a = 4) (h4 : 2 * b = 6) : 
                              ∀ x y : ℝ, (y = (3 / 2) * x) ∨ (y = - (3 / 2) * x) := by
  sorry

end hyperbola_asymptotes_l1264_126495


namespace triangle_distance_bisectors_l1264_126448

noncomputable def distance_between_bisectors {a b c : ℝ} (h₁: a > 0) (h₂: b > 0) (h₃: c > 0) : ℝ :=
  (2 * a * b * c) / (b^2 - c^2)

theorem triangle_distance_bisectors 
  (a b c : ℝ) (h₁: a > 0) (h₂: b > 0) (h₃: c > 0) :
  ∀ (DD₁ : ℝ), 
  DD₁ = distance_between_bisectors h₁ h₂ h₃ → 
  DD₁ = (2 * a * b * c) / (b^2 - c^2) := by 
  sorry

end triangle_distance_bisectors_l1264_126448


namespace factorize_expression_l1264_126437

theorem factorize_expression (x : ℝ) : x * (x - 3) - x + 3 = (x - 1) * (x - 3) :=
by
  sorry

end factorize_expression_l1264_126437


namespace goose_eggs_count_l1264_126486

theorem goose_eggs_count (E : ℕ) (h1 : E % 3 = 0) 
(h2 : ((4 / 5) * (1 / 3) * E) * (2 / 5) = 120) : E = 1125 := 
sorry

end goose_eggs_count_l1264_126486


namespace polynomial_coeffs_sum_l1264_126419

theorem polynomial_coeffs_sum (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ) (x : ℝ) :
  (2*x - 3)^5 = a_0 + a_1*x + a_2*x^2 + a_3*x^3 + a_4*x^4 + a_5*x^5 →
  a_1 + 2*a_2 + 3*a_3 + 4*a_4 + 5*a_5 = 10 :=
by
  sorry

end polynomial_coeffs_sum_l1264_126419


namespace closest_point_on_line_is_correct_l1264_126490

theorem closest_point_on_line_is_correct :
  ∃ (p : ℝ × ℝ), p = (-0.04, -0.28) ∧
  ∃ x : ℝ, p = (x, (3 * x - 1) / 4) ∧
  ∀ q : ℝ × ℝ, (q = (x, (3 * x - 1) / 4) → 
  (dist (2, -3) p) ≤ (dist (2, -3) q)) :=
sorry

end closest_point_on_line_is_correct_l1264_126490


namespace tutors_meet_in_lab_l1264_126434

theorem tutors_meet_in_lab (c a j t : ℕ)
  (hC : c = 5) (hA : a = 6) (hJ : j = 8) (hT : t = 9) :
  Nat.lcm (Nat.lcm (Nat.lcm c a) j) t = 360 :=
by
  rw [hC, hA, hJ, hT]
  rfl

end tutors_meet_in_lab_l1264_126434


namespace socks_total_l1264_126429

def socks_lisa (initial: Nat) := initial + 0

def socks_sandra := 20

def socks_cousin (sandra: Nat) := sandra / 5

def socks_mom (initial: Nat) := 8 + 3 * initial

theorem socks_total (initial: Nat) (sandra: Nat) :
  initial = 12 → sandra = 20 → 
  socks_lisa initial + socks_sandra + socks_cousin sandra + socks_mom initial = 80 :=
by
  intros h_initial h_sandra
  rw [h_initial, h_sandra]
  sorry

end socks_total_l1264_126429


namespace sufficient_condition_for_inequality_l1264_126459

theorem sufficient_condition_for_inequality (m : ℝ) (h : m ≠ 0) : (m > 2) → (m + 4 / m > 4) :=
by
  sorry

end sufficient_condition_for_inequality_l1264_126459


namespace find_q_l1264_126463

theorem find_q (p q : ℚ) (h1 : 5 * p + 7 * q = 20) (h2 : 7 * p + 5 * q = 26) : q = 5 / 12 := by
  sorry

end find_q_l1264_126463


namespace find_extrema_of_f_l1264_126400

noncomputable def f (x : ℝ) := x^2 - 4 * x - 2

theorem find_extrema_of_f : 
  (∀ x, (1 ≤ x ∧ x ≤ 4) → f x ≤ -2) ∧ 
  (∃ x, (1 ≤ x ∧ x ≤ 4 ∧ f x = -6)) :=
by sorry

end find_extrema_of_f_l1264_126400


namespace Sam_and_Tina_distance_l1264_126415

theorem Sam_and_Tina_distance (marguerite_distance : ℕ) (marguerite_time : ℕ)
  (sam_time : ℕ) (tina_time : ℕ) (sam_distance : ℕ) (tina_distance : ℕ)
  (h1 : marguerite_distance = 150) (h2 : marguerite_time = 3)
  (h3 : sam_time = 4) (h4 : tina_time = 2)
  (h5 : sam_distance = (marguerite_distance / marguerite_time) * sam_time)
  (h6 : tina_distance = (marguerite_distance / marguerite_time) * tina_time) :
  sam_distance = 200 ∧ tina_distance = 100 :=
by
  sorry

end Sam_and_Tina_distance_l1264_126415


namespace possible_values_of_m_l1264_126430

theorem possible_values_of_m (a b : ℤ) (h1 : a * b = -14) :
  ∃ m : ℤ, m = a + b ∧ (m = 5 ∨ m = -5 ∨ m = 13 ∨ m = -13) :=
by
  sorry

end possible_values_of_m_l1264_126430


namespace sum_of_sides_eq_l1264_126497

open Real

theorem sum_of_sides_eq (a h : ℝ) (α : ℝ) (ha : a > 0) (hh : h > 0) (hα : 0 < α ∧ α < π) :
  ∃ b c : ℝ, b + c = sqrt (a^2 + 2 * a * h * (cos (α / 2) / sin (α / 2))) :=
by
  sorry

end sum_of_sides_eq_l1264_126497


namespace vector_AB_to_vector_BA_l1264_126444

theorem vector_AB_to_vector_BA (z : ℂ) (hz : z = -3 + 2 * Complex.I) : -z = 3 - 2 * Complex.I :=
by
  rw [hz]
  sorry

end vector_AB_to_vector_BA_l1264_126444


namespace range_of_a_l1264_126456

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 0 ≤ x ∧ x ≤ 5 → x^2 - 2*x + a ≥ 0) → a ≥ 1 :=
by
  sorry

end range_of_a_l1264_126456


namespace enrolled_percentage_l1264_126411

theorem enrolled_percentage (total_students : ℝ) (non_bio_students : ℝ)
    (h_total : total_students = 880)
    (h_non_bio : non_bio_students = 440.00000000000006) : 
    ((total_students - non_bio_students) / total_students) * 100 = 50 := 
by
  rw [h_total, h_non_bio]
  norm_num
  sorry

end enrolled_percentage_l1264_126411


namespace part1_part2_l1264_126488

variable (f : ℝ → ℝ)

-- Conditions
axiom h1 : ∀ x y : ℝ, f (x - y) = f x / f y
axiom h2 : ∀ x : ℝ, f x > 0
axiom h3 : ∀ x y : ℝ, x < y → f x > f y

-- First part: f(0) = 1 and proving f(x + y) = f(x) * f(y)
theorem part1 : f 0 = 1 ∧ (∀ x y : ℝ, f (x + y) = f x * f y) :=
sorry

-- Second part: Given f(-1) = 3, solve the inequality
axiom h4 : f (-1) = 3

theorem part2 : {x : ℝ | (x ≤ 3) ∨ (x ≥ 4)} = {x : ℝ | f (x^2 - 7*x + 10) ≤ f (-2)} :=
sorry

end part1_part2_l1264_126488


namespace total_cost_l1264_126449

-- Definitions:
def amount_beef : ℕ := 1000
def price_per_pound_beef : ℕ := 8
def amount_chicken := amount_beef * 2
def price_per_pound_chicken : ℕ := 3

-- Theorem: The total cost of beef and chicken is $14000.
theorem total_cost : (amount_beef * price_per_pound_beef) + (amount_chicken * price_per_pound_chicken) = 14000 :=
by
  sorry

end total_cost_l1264_126449


namespace kristin_reading_time_l1264_126483

-- Definitions
def total_books : Nat := 20
def peter_time_per_book : ℕ := 18
def reading_speed_ratio : Nat := 3

-- Derived Definitions
def kristin_time_per_book : ℕ := peter_time_per_book * reading_speed_ratio
def kristin_books_to_read : Nat := total_books / 2
def kristin_total_time : ℕ := kristin_time_per_book * kristin_books_to_read

-- Statement to be proved
theorem kristin_reading_time :
  kristin_total_time = 540 :=
  by 
    -- Proof would go here, but we are only required to state the theorem
    sorry

end kristin_reading_time_l1264_126483


namespace expand_expression_l1264_126406

theorem expand_expression : ∀ (x : ℝ), (17 * x + 21) * 3 * x = 51 * x^2 + 63 * x :=
by
  intro x
  sorry

end expand_expression_l1264_126406


namespace point_equidistant_x_axis_y_axis_line_l1264_126425

theorem point_equidistant_x_axis_y_axis_line (x y : ℝ) (h1 : abs y = abs x) (h2 : abs (x + y - 2) / Real.sqrt 2 = abs x) :
  x = 1 :=
  sorry

end point_equidistant_x_axis_y_axis_line_l1264_126425


namespace train_length_l1264_126426

-- Definitions based on conditions
def faster_train_speed := 46 -- speed in km/hr
def slower_train_speed := 36 -- speed in km/hr
def time_to_pass := 72 -- time in seconds
def relative_speed_kmph := faster_train_speed - slower_train_speed
def relative_speed_mps : ℚ := (relative_speed_kmph * 1000) / 3600

theorem train_length :
  ∃ L : ℚ, (2 * L = relative_speed_mps * time_to_pass / 1) ∧ L = 100 := 
by
  sorry

end train_length_l1264_126426


namespace range_x_l1264_126484

variable {R : Type*} [LinearOrderedField R]

def monotone_increasing_on (f : R → R) (s : Set R) := ∀ ⦃a b⦄, a ≤ b → f a ≤ f b

theorem range_x 
    (f : R → R) 
    (h_mono : monotone_increasing_on f Set.univ) 
    (h_zero : f 1 = 0) 
    (h_ineq : ∀ x, f (x^2 + 3 * x - 3) < 0) :
  ∀ x, -4 < x ∧ x < 1 :=
by 
  sorry

end range_x_l1264_126484


namespace integer_roots_polynomial_l1264_126413

theorem integer_roots_polynomial (a : ℤ) :
  (∃ x : ℤ, x^3 + 3 * x^2 + a * x + 9 = 0) ↔ 
  (a = -109 ∨ a = -21 ∨ a = -13 ∨ a = 3 ∨ a = 11 ∨ a = 53) :=
by
  sorry

end integer_roots_polynomial_l1264_126413


namespace darius_age_is_8_l1264_126464

def age_of_darius (jenna_age darius_age : ℕ) : Prop :=
  jenna_age = darius_age + 5

theorem darius_age_is_8 (jenna_age darius_age : ℕ) (h1 : jenna_age = darius_age + 5) (h2: jenna_age = 13) : 
  darius_age = 8 :=
by
  sorry

end darius_age_is_8_l1264_126464


namespace value_range_of_sum_difference_l1264_126478

theorem value_range_of_sum_difference (a b c : ℝ) (h₁ : a < b)
  (h₂ : a + b = b / a) (h₃ : a * b = c / a) (h₄ : a + b > c)
  (h₅ : a + c > b) (h₆ : b + c > a) : 
  ∃ x y, x = 7 / 8 ∧ y = Real.sqrt 5 - 1 ∧ x < a + b - c ∧ a + b - c < y := sorry

end value_range_of_sum_difference_l1264_126478


namespace banana_count_l1264_126408

theorem banana_count : (2 + 7) = 9 := by
  rfl

end banana_count_l1264_126408


namespace solution_of_modified_system_l1264_126440

theorem solution_of_modified_system
  (a b x y : ℝ)
  (h1 : 2*a*3 + 3*4 = 18)
  (h2 : -3 + 5*b*4 = 17)
  : (x + y = 7 ∧ x - y = -1) → (2*a*(x+y) + 3*(x-y) = 18 ∧ (x+y) - 5*b*(x-y) = -17) → (x = (7 / 2) ∧ y = (-1 / 2)) :=
by
sorry

end solution_of_modified_system_l1264_126440


namespace correct_population_growth_pattern_statement_l1264_126491

-- Definitions based on the conditions provided
def overall_population_growth_modern (world_population : ℕ) : Prop :=
  -- The overall pattern of population growth worldwide is already in the modern stage
  sorry

def transformation_synchronized (world_population : ℕ) : Prop :=
  -- The transformation of population growth patterns in countries or regions around the world is synchronized
  sorry

def developed_countries_transformed (world_population : ℕ) : Prop :=
  -- Developed countries have basically completed the transformation of population growth patterns
  sorry

def transformation_determined_by_population_size (world_population : ℕ) : Prop :=
  -- The process of transformation in population growth patterns is determined by the population size of each area
  sorry

-- The statement to be proven
theorem correct_population_growth_pattern_statement (world_population : ℕ) :
  developed_countries_transformed world_population := sorry

end correct_population_growth_pattern_statement_l1264_126491


namespace calculator_press_count_l1264_126441

theorem calculator_press_count : 
  ∃ n : ℕ, n ≥ 4 ∧ (2 ^ (2 ^ n)) > 500 := 
by
  sorry

end calculator_press_count_l1264_126441


namespace side_length_correct_l1264_126436

noncomputable def find_side_length (b : ℝ) (angleB : ℝ) (sinA : ℝ) : ℝ :=
  let sinB := Real.sin angleB
  let a := b * sinA / sinB
  a

theorem side_length_correct (b : ℝ) (angleB : ℝ) (sinA : ℝ) (a : ℝ) 
  (hb : b = 4)
  (hangleB : angleB = Real.pi / 6)
  (hsinA : sinA = 1 / 3)
  (ha : a = 8 / 3) : 
  find_side_length b angleB sinA = a :=
by
  sorry

end side_length_correct_l1264_126436


namespace number_of_eighth_graders_l1264_126454

theorem number_of_eighth_graders (x y : ℕ) :
  (x > 0) ∧ (y > 0) ∧ (8 + x * y = (x * (x + 3) - 14) / 2) →
  x = 7 ∨ x = 14 :=
by
  sorry

end number_of_eighth_graders_l1264_126454


namespace best_is_man_l1264_126474

structure Competitor where
  name : String
  gender : String
  age : Int
  is_twin : Bool

noncomputable def participants : List Competitor := [
  ⟨"man", "male", 30, false⟩,
  ⟨"sister", "female", 30, true⟩,
  ⟨"son", "male", 30, true⟩,
  ⟨"niece", "female", 25, false⟩
]

def are_different_gender (c1 c2 : Competitor) : Bool := c1.gender ≠ c2.gender
def has_same_age (c1 c2 : Competitor) : Bool := c1.age = c2.age

noncomputable def best_competitor : Competitor :=
  let best_candidate := participants[0] -- assuming "man" is the best for example's sake
  let worst_candidate := participants[2] -- assuming "son" is the worst for example's sake
  best_candidate

theorem best_is_man : best_competitor.name = "man" :=
by
  have h1 : are_different_gender (participants[0]) (participants[2]) := by sorry
  have h2 : has_same_age (participants[0]) (participants[2]) := by sorry
  exact sorry

end best_is_man_l1264_126474


namespace sum_of_coordinates_of_other_endpoint_of_segment_l1264_126462

theorem sum_of_coordinates_of_other_endpoint_of_segment {x y : ℝ}
  (h1 : (6 + x) / 2 = 3)
  (h2 : (1 + y) / 2 = 7) :
  x + y = 13 := by
  sorry

end sum_of_coordinates_of_other_endpoint_of_segment_l1264_126462


namespace eccentricity_range_l1264_126422

-- Definitions and conditions
variable (a b c e : ℝ) (A B: ℝ × ℝ)
variable (d1 d2 : ℝ)

variable (a_pos : a > 2)
variable (b_pos : b > 0)
variable (c_pos : c > 0)
variable (c_eq : c = Real.sqrt (a ^ 2 + b ^ 2))
variable (A_def : A = (a, 0))
variable (B_def : B = (0, b))
variable (d1_def : d1 = abs (b * 2 + a * 0 - a * b ) / Real.sqrt (a^2 + b^2))
variable (d2_def : d2 = abs (b * (-2) + a * 0 - a * b) / Real.sqrt (a^2 + b^2))
variable (d_ineq : d1 + d2 ≥ (4 / 5) * c)
variable (eccentricity : e = c / a)

-- Theorem statement
theorem eccentricity_range : (Real.sqrt 5 / 2 ≤ e) ∧ (e ≤ Real.sqrt 5) :=
by sorry

end eccentricity_range_l1264_126422


namespace prod_three_consec_cubemultiple_of_504_l1264_126412

theorem prod_three_consec_cubemultiple_of_504 (a : ℤ) : (a^3 - 1) * a^3 * (a^3 + 1) % 504 = 0 := by
  sorry

end prod_three_consec_cubemultiple_of_504_l1264_126412


namespace g_1200_value_l1264_126480

noncomputable def g : ℝ → ℝ := sorry

-- Assume the given condition as a definition
axiom functional_eq (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : g (x * y) = g x / y

-- Assume the given value of g(1000)
axiom g_1000_value : g 1000 = 4

-- Prove that g(1200) = 10/3
theorem g_1200_value : g 1200 = 10 / 3 := by
  sorry

end g_1200_value_l1264_126480


namespace total_crayons_l1264_126418

noncomputable def original_crayons : ℝ := 479.0
noncomputable def additional_crayons : ℝ := 134.0

theorem total_crayons : original_crayons + additional_crayons = 613.0 := by
  sorry

end total_crayons_l1264_126418


namespace comparison_of_square_roots_l1264_126493

theorem comparison_of_square_roots (P Q : ℝ) (hP : P = Real.sqrt 2) (hQ : Q = Real.sqrt 6 - Real.sqrt 2) : P > Q :=
by
  sorry

end comparison_of_square_roots_l1264_126493

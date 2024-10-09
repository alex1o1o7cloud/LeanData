import Mathlib

namespace greatest_multiple_of_5_and_6_lt_1000_l475_47530

theorem greatest_multiple_of_5_and_6_lt_1000 : 
  ∃ n, n % 5 = 0 ∧ n % 6 = 0 ∧ n < 1000 ∧ (∀ m, m % 5 = 0 ∧ m % 6 = 0 ∧ m < 1000 → m ≤ n) :=
  sorry

end greatest_multiple_of_5_and_6_lt_1000_l475_47530


namespace impossibility_of_equal_sum_selection_l475_47504

theorem impossibility_of_equal_sum_selection :
  ¬ ∃ (selected non_selected : Fin 10 → ℕ),
    (∀ i, selected i = 1 ∨ selected i = 36 ∨ selected i = 2 ∨ selected i = 35 ∨ 
              selected i = 3 ∨ selected i = 34 ∨ selected i = 4 ∨ selected i = 33 ∨ 
              selected i = 5 ∨ selected i = 32 ∨ selected i = 6 ∨ selected i = 31 ∨ 
              selected i = 7 ∨ selected i = 30 ∨ selected i = 8 ∨ selected i = 29 ∨ 
              selected i = 9 ∨ selected i = 28 ∨ selected i = 10 ∨ selected i = 27) ∧ 
    (∀ i, non_selected i = 1 ∨ non_selected i = 36 ∨ non_selected i = 2 ∨ non_selected i = 35 ∨ 
              non_selected i = 3 ∨ non_selected i = 34 ∨ non_selected i = 4 ∨ non_selected i = 33 ∨ 
              non_selected i = 5 ∨ non_selected i = 32 ∨ non_selected i = 6 ∨ non_selected i = 31 ∨ 
              non_selected i = 7 ∨ non_selected i = 30 ∨ non_selected i = 8 ∨ non_selected i = 29 ∨ 
              non_selected i = 9 ∨ non_selected i = 28 ∨ non_selected i = 10 ∨ non_selected i = 27) ∧ 
    (selected ≠ non_selected) ∧ 
    (Finset.univ.sum selected = Finset.univ.sum non_selected) :=
sorry

end impossibility_of_equal_sum_selection_l475_47504


namespace equation_of_circle_center_0_4_passing_through_3_0_l475_47533

noncomputable def circle_radius (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

theorem equation_of_circle_center_0_4_passing_through_3_0 :
  ∃ (r : ℝ), (r = circle_radius 0 4 3 0) ∧ (r = 5) ∧ ((x y : ℝ) → ((x - 0) ^ 2 + (y - 4) ^ 2 = r ^ 2) ↔ (x ^ 2 + (y - 4) ^ 2 = 25)) :=
by
  sorry

end equation_of_circle_center_0_4_passing_through_3_0_l475_47533


namespace original_fraction_2_7_l475_47590

theorem original_fraction_2_7 (N D : ℚ) : 
  (1.40 * N) / (0.50 * D) = 4 / 5 → N / D = 2 / 7 :=
by
  intro h
  sorry

end original_fraction_2_7_l475_47590


namespace jon_coffee_spending_in_april_l475_47562

def cost_per_coffee : ℕ := 2
def coffees_per_day : ℕ := 2
def days_in_april : ℕ := 30

theorem jon_coffee_spending_in_april :
  (coffees_per_day * cost_per_coffee) * days_in_april = 120 :=
by
  sorry

end jon_coffee_spending_in_april_l475_47562


namespace distribute_stickers_l475_47522

-- Definitions based on conditions
def stickers : ℕ := 10
def sheets : ℕ := 5

-- Theorem stating the equivalence of distributing the stickers onto sheets
theorem distribute_stickers :
  (Nat.choose (stickers + sheets - 1) (sheets - 1)) = 1001 :=
by 
  -- Here is where the proof would go, but we skip it with sorry for the purpose of this task
  sorry

end distribute_stickers_l475_47522


namespace factorize_polynomial_l475_47546

theorem factorize_polynomial (x y : ℝ) : 3 * x^2 - 3 * y^2 = 3 * (x + y) * (x - y) := 
by sorry

end factorize_polynomial_l475_47546


namespace mark_peters_pond_depth_l475_47544

theorem mark_peters_pond_depth :
  let mark_depth := 19
  let peter_depth := 5
  let three_times_peter_depth := 3 * peter_depth
  mark_depth - three_times_peter_depth = 4 :=
by
  sorry

end mark_peters_pond_depth_l475_47544


namespace sin_double_angle_l475_47578

theorem sin_double_angle (x : ℝ) (h : Real.tan (π / 4 - x) = 2) : Real.sin (2 * x) = -3 / 5 :=
by
  sorry

end sin_double_angle_l475_47578


namespace percent_neither_filler_nor_cheese_l475_47547

-- Define the given conditions as constants
def total_weight : ℕ := 200
def filler_weight : ℕ := 40
def cheese_weight : ℕ := 30

-- Definition of the remaining weight that is neither filler nor cheese
def neither_weight : ℕ := total_weight - filler_weight - cheese_weight

-- Calculation of the percentage of the burger that is neither filler nor cheese
def percentage_neither : ℚ := (neither_weight : ℚ) / (total_weight : ℚ) * 100

-- The theorem to prove
theorem percent_neither_filler_nor_cheese :
  percentage_neither = 65 := by
  sorry

end percent_neither_filler_nor_cheese_l475_47547


namespace five_equal_angles_72_degrees_l475_47568

theorem five_equal_angles_72_degrees
  (five_rays : ℝ)
  (equal_angles : ℝ) 
  (sum_angles : five_rays * equal_angles = 360) :
  equal_angles = 72 :=
by
  sorry

end five_equal_angles_72_degrees_l475_47568


namespace train_speed_is_correct_l475_47505

-- Conditions
def train_length := 190.0152  -- in meters
def crossing_time := 17.1     -- in seconds

-- Convert units
def train_length_km := train_length / 1000  -- in kilometers
def crossing_time_hr := crossing_time / 3600  -- in hours

-- Statement of the proof problem
theorem train_speed_is_correct :
  (train_length_km / crossing_time_hr) = 40 :=
sorry

end train_speed_is_correct_l475_47505


namespace family_members_l475_47557

theorem family_members (N : ℕ) (income : ℕ → ℕ) (average_income : ℕ) :
  average_income = 10000 ∧
  income 0 = 8000 ∧
  income 1 = 15000 ∧
  income 2 = 6000 ∧
  income 3 = 11000 ∧
  (income 0 + income 1 + income 2 + income 3) = 4 * average_income →
  N = 4 :=
by {
  sorry
}

end family_members_l475_47557


namespace prime_sum_divisible_l475_47589

theorem prime_sum_divisible (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (h : q = p + 2) :
  (p ^ q + q ^ p) % (p + q) = 0 :=
by
  sorry

end prime_sum_divisible_l475_47589


namespace topic_preference_order_l475_47503

noncomputable def astronomy_fraction := (8 : ℚ) / 21
noncomputable def botany_fraction := (5 : ℚ) / 14
noncomputable def chemistry_fraction := (9 : ℚ) / 28

theorem topic_preference_order :
  (astronomy_fraction > botany_fraction) ∧ (botany_fraction > chemistry_fraction) :=
by
  sorry

end topic_preference_order_l475_47503


namespace positive_integer_solutions_l475_47577

theorem positive_integer_solutions:
  ∀ (x y : ℕ), (5 * x + y = 11) → (x > 0) → (y > 0) → (x = 1 ∧ y = 6) ∨ (x = 2 ∧ y = 1) :=
by
  sorry

end positive_integer_solutions_l475_47577


namespace jewelry_store_total_cost_l475_47569

theorem jewelry_store_total_cost :
  let necklaces_needed := 7
  let rings_needed := 12
  let bracelets_needed := 7
  let necklace_price := 4
  let ring_price := 10
  let bracelet_price := 5
  let necklace_discount := if necklaces_needed >= 6 then 0.15 else if necklaces_needed >= 4 then 0.10 else 0
  let ring_discount := if rings_needed >= 20 then 0.10 else if rings_needed >= 10 then 0.05 else 0
  let bracelet_discount := if bracelets_needed >= 10 then 0.12 else if bracelets_needed >= 7 then 0.08 else 0
  let necklace_cost := necklaces_needed * (necklace_price * (1 - necklace_discount))
  let ring_cost := rings_needed * (ring_price * (1 - ring_discount))
  let bracelet_cost := bracelets_needed * (bracelet_price * (1 - bracelet_discount))
  let total_cost := necklace_cost + ring_cost + bracelet_cost
  total_cost = 170 := by
  -- calculation details omitted
  sorry

end jewelry_store_total_cost_l475_47569


namespace minimum_value_of_xy_l475_47599

theorem minimum_value_of_xy (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 2 * y = x * y) :
  x * y ≥ 8 :=
sorry

end minimum_value_of_xy_l475_47599


namespace find_angle_B_find_area_of_ABC_l475_47566

noncomputable def angle_B (a b c : ℝ) (C : ℝ) (h1 : 2 * b * Real.cos C = 2 * a + c) : ℝ := 
  if b * Real.cos C = -a then Real.pi - 2 * Real.arctan (a / c)
  else 2 * Real.pi / 3

theorem find_angle_B (a b c : ℝ) (C : ℝ) (h1 : 2 * b * Real.cos C = 2 * a + c) :
  angle_B a b c C h1 = 2 * Real.pi / 3 := 
sorry

noncomputable def area_of_ABC (a b c : ℝ) (C B : ℝ) (d : ℝ) (position : ℕ) (h1 : 2 * b * Real.cos C = 2 * a + c) (h2 : b = 2 * Real.sqrt 3) (h3 : d = 1) : ℝ :=
  if position = 1 then /- calculation for BD bisector case -/ (a * c / 2) * Real.sin (2 * Real.pi / 3)
  else /- calculation for midpoint case -/ (a * c / 2) * Real.sin (2 * Real.pi / 3)

theorem find_area_of_ABC (a b c : ℝ) (C B : ℝ) (d : ℝ) (position : ℕ) (h1 : 2 * b * Real.cos C = 2 * a + c) (h2 : b = 2 * Real.sqrt 3) (h3 : d = 1) (hB : angle_B a b c C h1 = 2 * Real.pi / 3) :
  area_of_ABC a b c C (2 * Real.pi / 3) d position h1 h2 h3 = Real.sqrt 3 := 
sorry

end find_angle_B_find_area_of_ABC_l475_47566


namespace domain_f_log2_x_to_domain_f_x_l475_47552

variable {f : ℝ → ℝ}

-- Condition: The domain of y = f(log₂ x) is [1/2, 4]
def domain_f_log2_x : Set ℝ := Set.Icc (1 / 2) 4

-- Proof statement
theorem domain_f_log2_x_to_domain_f_x
  (h : ∀ x, x ∈ domain_f_log2_x → f (Real.log x / Real.log 2) = f x) :
  Set.Icc (-1) 2 = {x : ℝ | ∃ y ∈ domain_f_log2_x, Real.log y / Real.log 2 = x} :=
by
  sorry

end domain_f_log2_x_to_domain_f_x_l475_47552


namespace sum_of_remainders_l475_47516

theorem sum_of_remainders (n : ℤ) (h : n % 18 = 11) :
  (n % 2 + n % 9) = 3 :=
sorry

end sum_of_remainders_l475_47516


namespace find_coefficients_l475_47508

theorem find_coefficients (a b p q : ℝ) :
    (∀ x : ℝ, (2 * x - 1) ^ 20 - (a * x + b) ^ 20 = (x^2 + p * x + q) ^ 10) →
    a = -2 * b ∧ (b = 1 ∨ b = -1) ∧ p = -1 ∧ q = 1 / 4 :=
by 
    sorry

end find_coefficients_l475_47508


namespace incorrect_statement_A_l475_47517

-- Definitions based on conditions
def equilibrium_shifts (condition: Type) : Prop := sorry
def value_K_changes (condition: Type) : Prop := sorry

-- The incorrect statement definition
def statement_A (condition: Type) : Prop := equilibrium_shifts condition → value_K_changes condition

-- The final theorem stating that 'statement_A' is incorrect
theorem incorrect_statement_A (condition: Type) : ¬ statement_A condition :=
sorry

end incorrect_statement_A_l475_47517


namespace express_f12_in_terms_of_a_l475_47500

variable {f : ℝ → ℝ}
variable {a : ℝ}
variable (f_add : ∀ x y : ℝ, f (x + y) = f x + f y)
variable (f_neg_three : f (-3) = a)

theorem express_f12_in_terms_of_a : f 12 = -4 * a := sorry

end express_f12_in_terms_of_a_l475_47500


namespace distinct_digit_sums_l475_47575

theorem distinct_digit_sums (A B C E D : ℕ) (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ E ∧ A ≠ D ∧ B ≠ C ∧ B ≠ E ∧ B ≠ D ∧ C ≠ E ∧ C ≠ D ∧ E ≠ D)
 (h_ab : A + B = D) (h_ab_lt_10 : A + B < 10) (h_ce : C + E = D) :
  ∃ (x : ℕ), x = 8 := 
sorry

end distinct_digit_sums_l475_47575


namespace arithmetic_sequence_ratio_l475_47553

theorem arithmetic_sequence_ratio
  (x y a1 a2 a3 b1 b2 b3 b4 : ℝ)
  (h1 : x ≠ y)
  (h2 : a1 = x + (1 * (a2 - a1)))
  (h3 : a2 = x + (2 * (a2 - a1)))
  (h4 : a3 = x + (3 * (a2 - a1)))
  (h5 : y = x + (4 * (a2 - a1)))
  (h6 : x = x)
  (h7 : b2 = x + (1 * (b3 - x)))
  (h8 : b3 = x + (2 * (b3 - x)))
  (h9 : y = x + (3 * (b3 - x)))
  (h10 : b4 = x + (4 * (b3 - x))) :
  (b4 - b3) / (a2 - a1) = 8 / 3 := by
  sorry

end arithmetic_sequence_ratio_l475_47553


namespace line_equation_through_two_points_l475_47524

noncomputable def LineEquation (x0 y0 x1 y1 x y : ℝ) : Prop :=
  (x1 ≠ x0) → (y1 ≠ y0) → 
  (y - y0) / (y1 - y0) = (x - x0) / (x1 - x0)

theorem line_equation_through_two_points 
  (x0 y0 x1 y1 : ℝ) 
  (h₁ : x1 ≠ x0) 
  (h₂ : y1 ≠ y0) : 
  ∀ (x y : ℝ), LineEquation x0 y0 x1 y1 x y :=  
by
  sorry

end line_equation_through_two_points_l475_47524


namespace probability_colors_match_l475_47521

noncomputable def prob_abe_shows_blue : ℚ := 2 / 4
noncomputable def prob_bob_shows_blue : ℚ := 3 / 6
noncomputable def prob_abe_shows_green : ℚ := 2 / 4
noncomputable def prob_bob_shows_green : ℚ := 1 / 6

noncomputable def prob_same_color : ℚ :=
  (prob_abe_shows_blue * prob_bob_shows_blue) + (prob_abe_shows_green * prob_bob_shows_green)

theorem probability_colors_match : prob_same_color = 1 / 3 :=
by
  sorry

end probability_colors_match_l475_47521


namespace option_c_is_not_equal_l475_47582

theorem option_c_is_not_equal :
  let A := 14 / 12
  let B := 1 + 1 / 6
  let C := 1 + 1 / 2
  let D := 1 + 7 / 42
  let E := 1 + 14 / 84
  A = 7 / 6 ∧ B = 7 / 6 ∧ D = 7 / 6 ∧ E = 7 / 6 ∧ C ≠ 7 / 6 :=
by
  sorry

end option_c_is_not_equal_l475_47582


namespace sufficient_not_necessary_condition_l475_47520

theorem sufficient_not_necessary_condition (x y : ℝ) (h1 : x ≥ 1) (h2 : y ≥ 2) : 
  x + y ≥ 3 ∧ (¬ (∀ x y : ℝ, x + y ≥ 3 → x ≥ 1 ∧ y ≥ 2)) := 
by {
  sorry -- The actual proof goes here.
}

end sufficient_not_necessary_condition_l475_47520


namespace height_difference_l475_47567

variable (h_A h_B h_D h_E h_F h_G : ℝ)

theorem height_difference :
  (h_A - h_D = 4.5) →
  (h_E - h_D = -1.7) →
  (h_F - h_E = -0.8) →
  (h_G - h_F = 1.9) →
  (h_B - h_G = 3.6) →
  (h_A - h_B > 0) :=
by
  intro h_AD h_ED h_FE h_GF h_BG
  sorry

end height_difference_l475_47567


namespace joan_balloons_l475_47592

def initial_balloons : ℕ := 72
def additional_balloons : ℕ := 23
def total_balloons : ℕ := initial_balloons + additional_balloons

theorem joan_balloons : total_balloons = 95 := by
  sorry

end joan_balloons_l475_47592


namespace parallel_vectors_m_l475_47560

theorem parallel_vectors_m (m : ℝ) :
  let a := (1, 2)
  let b := (m, m + 1)
  a.1 * b.2 = a.2 * b.1 → m = 1 :=
by
  intros a b h
  dsimp at *
  sorry

end parallel_vectors_m_l475_47560


namespace fraction_identity_l475_47529

theorem fraction_identity (a b : ℝ) (h : (1/a + 1/b) / (1/a - 1/b) = 1009) : (a + b) / (a - b) = -1009 :=
by
  sorry

end fraction_identity_l475_47529


namespace tournament_committee_count_l475_47597

theorem tournament_committee_count :
  let teams := 6
  let members_per_team := 8
  let host_team_choices := Nat.choose 8 3
  let regular_non_host_choices := Nat.choose 8 2
  let special_non_host_choices := Nat.choose 8 3
  let total_regular_non_host_choices := regular_non_host_choices ^ 4 
  let combined_choices_non_host := total_regular_non_host_choices * special_non_host_choices
  let combined_choices_host_non_host := combined_choices_non_host * host_team_choices
  let total_choices := combined_choices_host_non_host * teams
  total_choices = 11568055296 := 
by {
  let teams := 6
  let members_per_team := 8
  let host_team_choices := Nat.choose 8 3
  let regular_non_host_choices := Nat.choose 8 2
  let special_non_host_choices := Nat.choose 8 3
  let total_regular_non_host_choices := regular_non_host_choices ^ 4 
  let combined_choices_non_host := total_regular_non_host_choices * special_non_host_choices
  let combined_choices_host_non_host := combined_choices_non_host * host_team_choices
  let total_choices := combined_choices_host_non_host * teams
  have h_total_choices_eq : total_choices = 11568055296 := sorry
  exact h_total_choices_eq
}

end tournament_committee_count_l475_47597


namespace smallest_fraction_l475_47554

theorem smallest_fraction {a b c d e : ℚ}
  (ha : a = 7/15)
  (hb : b = 5/11)
  (hc : c = 16/33)
  (hd : d = 49/101)
  (he : e = 89/183) :
  (b < a) ∧ (b < c) ∧ (b < d) ∧ (b < e) := 
sorry

end smallest_fraction_l475_47554


namespace number_represented_by_B_l475_47540

theorem number_represented_by_B (b : ℤ) : 
  (abs (b - 3) = 5) -> (b = 8 ∨ b = -2) :=
by
  intro h
  sorry

end number_represented_by_B_l475_47540


namespace chuck_total_playable_area_l475_47506

noncomputable def chuck_roaming_area (shed_length shed_width leash_length : ℝ) : ℝ :=
  let larger_arc_area := (3 / 4) * Real.pi * leash_length ^ 2
  let additional_sector_area := (1 / 4) * Real.pi * (leash_length - shed_length) ^ 2
  larger_arc_area + additional_sector_area

theorem chuck_total_playable_area :
  chuck_roaming_area 3 4 5 = 19 * Real.pi :=
  by
  sorry

end chuck_total_playable_area_l475_47506


namespace solve_problem_l475_47593

open Complex

noncomputable def problem_statement (a : ℝ) : Prop :=
  abs ((a : ℂ) + I) / abs I = 2
  
theorem solve_problem {a : ℝ} : problem_statement a → a = Real.sqrt 3 :=
by
  sorry

end solve_problem_l475_47593


namespace equation_solution_l475_47591

noncomputable def solve_equation (x : ℝ) : Prop :=
  (4 / (x - 1) + 1 / (1 - x) = 1) → x = 4

theorem equation_solution (x : ℝ) (h : 4 / (x - 1) + 1 / (1 - x) = 1) : x = 4 := by
  sorry

end equation_solution_l475_47591


namespace man_l475_47584

theorem man's_speed_upstream :
  ∀ (R : ℝ), (R + 1.5 = 11) → (R - 1.5 = 8) :=
by
  intros R h
  sorry

end man_l475_47584


namespace parabola_vertex_coordinates_l475_47570

theorem parabola_vertex_coordinates {a b c : ℝ} (h_eq : ∀ x : ℝ, a * x^2 + b * x + c = 3)
  (h_root : a * 2^2 + b * 2 + c = 3) (h_symm : ∀ x : ℝ, a * (2 - x)^2 + b * (2 - x) + c = a * x^2 + b * x + c) :
  (2, 3) = (2, 3) :=
by
  sorry

end parabola_vertex_coordinates_l475_47570


namespace sector_radius_l475_47545

theorem sector_radius (α S r : ℝ) (h1 : α = 3/4 * Real.pi) (h2 : S = 3/2 * Real.pi) :
  S = 1/2 * r^2 * α → r = 2 :=
by
  sorry

end sector_radius_l475_47545


namespace black_shirts_in_pack_l475_47595

-- defining the conditions
variables (B : ℕ) -- the number of black shirts in each pack
variable (total_shirts : ℕ := 21)
variable (yellow_shirts_per_pack : ℕ := 2)
variable (black_packs : ℕ := 3)
variable (yellow_packs : ℕ := 3)

-- ensuring the conditions are met, the total shirts equals 21
def total_black_shirts := black_packs * B
def total_yellow_shirts := yellow_packs * yellow_shirts_per_pack

-- the proof problem
theorem black_shirts_in_pack : total_black_shirts + total_yellow_shirts = total_shirts → B = 5 := by
  sorry

end black_shirts_in_pack_l475_47595


namespace relationship_y1_y2_l475_47556

theorem relationship_y1_y2 (y1 y2 : ℝ) 
  (h1 : y1 = 3 / -1) 
  (h2 : y2 = 3 / -3) : 
  y1 < y2 :=
by
  sorry

end relationship_y1_y2_l475_47556


namespace max_true_statements_l475_47542

theorem max_true_statements 
  (a b : ℝ) 
  (cond1 : a > 0) 
  (cond2 : b > 0) : 
  ( 
    ( (1 / a > 1 / b) ∧ (a^2 < b^2) 
      ∧ (a > b) ∧ (a > 0) ∧ (b > 0) ) 
    ∨ 
    ( (1 / a > 1 / b) ∧ ¬(a^2 < b^2) 
      ∧ (a > b) ∧ (a > 0) ∧ (b > 0) ) 
    ∨ 
    ( ¬(1 / a > 1 / b) ∧ (a^2 < b^2) 
      ∧ (a > b) ∧ (a > 0) ∧ (b > 0) ) 
    ∨ 
    ( ¬(1 / a > 1 / b) ∧ ¬(a^2 < b^2) 
      ∧ (a > b) ∧ (a > 0) ∧ (b > 0) ) 
  ) 
→ 
  (true ∧ true ∧ true ∧ true → 4 = 4) :=
sorry

end max_true_statements_l475_47542


namespace square_pattern_1111111_l475_47580

theorem square_pattern_1111111 :
  11^2 = 121 ∧ 111^2 = 12321 ∧ 1111^2 = 1234321 → 1111111^2 = 1234567654321 :=
by
  sorry

end square_pattern_1111111_l475_47580


namespace multiple_with_digits_l475_47585

theorem multiple_with_digits (n : ℕ) (h : n > 0) :
  ∃ (m : ℕ), (m % n = 0) ∧ (m < 10 ^ n) ∧ (∀ d ∈ m.digits 10, d = 0 ∨ d = 1) :=
by
  sorry

end multiple_with_digits_l475_47585


namespace sqrt_x_eq_0_123_l475_47565

theorem sqrt_x_eq_0_123 (x : ℝ) (h1 : Real.sqrt 15129 = 123) (h2 : Real.sqrt x = 0.123) : x = 0.015129 := by
  -- proof goes here, but it is omitted
  sorry

end sqrt_x_eq_0_123_l475_47565


namespace percent_area_square_in_rectangle_l475_47574

theorem percent_area_square_in_rectangle 
  (s : ℝ) (rect_width : ℝ) (rect_length : ℝ) (h1 : rect_width = 2 * s) (h2 : rect_length = 2 * rect_width) : 
  (s^2 / (rect_length * rect_width)) * 100 = 12.5 :=
by
  sorry

end percent_area_square_in_rectangle_l475_47574


namespace compare_subtract_one_l475_47550

theorem compare_subtract_one (a b : ℝ) (h : a < b) : a - 1 < b - 1 :=
sorry

end compare_subtract_one_l475_47550


namespace michael_exceeds_suresh_l475_47501

theorem michael_exceeds_suresh (P M S : ℝ) 
  (h_total : P + M + S = 2400)
  (h_p_m_ratio : P / 5 = M / 7)
  (h_m_s_ratio : M / 3 = S / 2) : M - S = 336 :=
by
  sorry

end michael_exceeds_suresh_l475_47501


namespace cars_per_client_l475_47558

-- Define the conditions
def num_cars : ℕ := 18
def selections_per_car : ℕ := 3
def num_clients : ℕ := 18

-- Define the proof problem as a theorem
theorem cars_per_client :
  (num_cars * selections_per_car) / num_clients = 3 :=
sorry

end cars_per_client_l475_47558


namespace man_double_son_in_years_l475_47596

-- Definitions of conditions
def son_age : ℕ := 18
def man_age : ℕ := son_age + 20

-- The proof problem statement
theorem man_double_son_in_years :
  ∃ (X : ℕ), (man_age + X = 2 * (son_age + X)) ∧ X = 2 :=
by
  sorry

end man_double_son_in_years_l475_47596


namespace remainder_correct_l475_47588

noncomputable def P : Polynomial ℝ := Polynomial.C 1 * Polynomial.X^6 
                                  + Polynomial.C 2 * Polynomial.X^5 
                                  - Polynomial.C 3 * Polynomial.X^4 
                                  + Polynomial.C 1 * Polynomial.X^3 
                                  - Polynomial.C 2 * Polynomial.X^2
                                  + Polynomial.C 5 * Polynomial.X 
                                  - Polynomial.C 1

noncomputable def D : Polynomial ℝ := (Polynomial.X - Polynomial.C 1) * 
                                      (Polynomial.X + Polynomial.C 2) * 
                                      (Polynomial.X - Polynomial.C 3)

noncomputable def R : Polynomial ℝ := 17 * Polynomial.X^2 - 52 * Polynomial.X + 38

theorem remainder_correct :
    ∀ (q : Polynomial ℝ), P = D * q + R :=
by sorry

end remainder_correct_l475_47588


namespace number_that_multiplies_x_l475_47525

variables (n x y : ℝ)

theorem number_that_multiplies_x :
  n * x = 3 * y → 
  x * y ≠ 0 → 
  (1 / 5 * x) / (1 / 6 * y) = 0.72 →
  n = 5 :=
by
  intros h1 h2 h3
  sorry

end number_that_multiplies_x_l475_47525


namespace prob_score_5_points_is_three_over_eight_l475_47543

noncomputable def probability_of_scoring_5_points : ℚ :=
  let total_events := 2^3
  let favorable_events := 3 -- Calculated from combinatorial logic.
  favorable_events / total_events

theorem prob_score_5_points_is_three_over_eight :
  probability_of_scoring_5_points = 3 / 8 :=
by
  sorry

end prob_score_5_points_is_three_over_eight_l475_47543


namespace length_of_side_b_l475_47571

theorem length_of_side_b (B C : ℝ) (c b : ℝ) (hB : B = 45 * Real.pi / 180) (hC : C = 60 * Real.pi / 180) (hc : c = 1) :
  b = Real.sqrt 6 / 3 :=
by
  sorry

end length_of_side_b_l475_47571


namespace greatest_integer_value_of_x_l475_47512

theorem greatest_integer_value_of_x :
  ∃ x : ℤ, (3 * |2 * x + 1| + 10 > 28) ∧ (∀ y : ℤ, 3 * |2 * y + 1| + 10 > 28 → y ≤ x) :=
sorry

end greatest_integer_value_of_x_l475_47512


namespace kara_total_water_intake_l475_47555

-- Define dosages and water intake per tablet
def medicationA_doses_per_day := 3
def medicationB_doses_per_day := 4
def medicationC_doses_per_day := 2
def medicationD_doses_per_day := 1

def water_per_tablet_A := 4
def water_per_tablet_B := 5
def water_per_tablet_C := 6
def water_per_tablet_D := 8

-- Compute weekly water intake
def weekly_water_intake_medication (doses_per_day water_per_tablet : ℕ) (days : ℕ) : ℕ :=
  doses_per_day * water_per_tablet * days

-- Total water intake for two weeks if instructions are followed perfectly
def total_water_no_errors :=
  2 * (weekly_water_intake_medication medicationA_doses_per_day water_per_tablet_A 7 +
       weekly_water_intake_medication medicationB_doses_per_day water_per_tablet_B 7 +
       weekly_water_intake_medication medicationC_doses_per_day water_per_tablet_C 7 +
       weekly_water_intake_medication medicationD_doses_per_day water_per_tablet_D 7)

-- Missed doses in second week
def missed_water_second_week :=
  3 * water_per_tablet_A +
  2 * water_per_tablet_B +
  2 * water_per_tablet_C +
  1 * water_per_tablet_D

-- Total water actually drunk over two weeks
def total_water_real :=
  total_water_no_errors - missed_water_second_week

-- Proof statement
theorem kara_total_water_intake :
  total_water_real = 686 :=
by
  sorry

end kara_total_water_intake_l475_47555


namespace circle_Q_radius_l475_47537

theorem circle_Q_radius
  (radius_P : ℝ := 2)
  (radius_S : ℝ := 4)
  (u v : ℝ)
  (h1: (2 + v)^2 = (2 + u)^2 + v^2)
  (h2: (4 - v)^2 = u^2 + v^2)
  (h3: v = u + u^2 / 2)
  (h4: v = 2 - u^2 / 4) :
  v = 16 / 9 :=
by
  /- Proof goes here. -/
  sorry

end circle_Q_radius_l475_47537


namespace logs_needed_l475_47598

theorem logs_needed (needed_woodblocks : ℕ) (current_logs : ℕ) (woodblocks_per_log : ℕ) 
  (H1 : needed_woodblocks = 80) 
  (H2 : current_logs = 8) 
  (H3 : woodblocks_per_log = 5) : 
  current_logs * woodblocks_per_log < needed_woodblocks → 
  (needed_woodblocks - current_logs * woodblocks_per_log) / woodblocks_per_log = 8 := by
  sorry

end logs_needed_l475_47598


namespace correct_statements_l475_47509

theorem correct_statements (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + 2 * b = 1) :
  (∀ b, a = 1 - 2 * b → a^2 + b^2 ≥ 1/5) ∧
  (∀ a b, a + 2 * b = 1 → ab ≤ 1/8) ∧
  (∀ a b, a + 2 * b = 1 → 3 + 2 * Real.sqrt 2 ≤ (1 / a + 1 / b)) :=
by
  sorry

end correct_statements_l475_47509


namespace trains_cross_time_l475_47548

noncomputable def timeToCrossEachOther (L : ℝ) (T1 : ℝ) (T2 : ℝ) : ℝ :=
  let V1 := L / T1
  let V2 := L / T2
  let Vr := V1 + V2
  let totalDistance := L + L
  totalDistance / Vr

theorem trains_cross_time (L T1 T2 : ℝ) (hL : L = 120) (hT1 : T1 = 10) (hT2 : T2 = 15) :
  timeToCrossEachOther L T1 T2 = 12 :=
by
  simp [timeToCrossEachOther, hL, hT1, hT2]
  sorry

end trains_cross_time_l475_47548


namespace maximize_profit_l475_47536

theorem maximize_profit 
  (cost_per_product : ℝ)
  (initial_price : ℝ)
  (initial_sales : ℝ)
  (price_increase_effect : ℝ)
  (daily_sales_decrease : ℝ)
  (max_profit_price : ℝ)
  (max_profit : ℝ)
  :
  cost_per_product = 8 ∧ initial_price = 10 ∧ initial_sales = 100 ∧ price_increase_effect = 1 ∧ daily_sales_decrease = 10 → 
  max_profit_price = 14 ∧
  max_profit = 360 :=
by 
  intro h
  have h_cost := h.1
  have h_initial_price := h.2.1
  have h_initial_sales := h.2.2.1
  have h_price_increase_effect := h.2.2.2.1
  have h_daily_sales_decrease := h.2.2.2.2
  sorry

end maximize_profit_l475_47536


namespace allocation_methods_count_l475_47564

def number_of_allocation_methods (doctors nurses : ℕ) (hospitals : ℕ) (nurseA nurseB : ℕ) :=
  if (doctors = 3) ∧ (nurses = 6) ∧ (hospitals = 3) ∧ (nurseA = 1) ∧ (nurseB = 1) then 684 else 0

theorem allocation_methods_count :
  number_of_allocation_methods 3 6 3 2 2 = 684 :=
by
  sorry

end allocation_methods_count_l475_47564


namespace inverse_mod_187_l475_47594

theorem inverse_mod_187 : ∃ (x : ℤ), 0 ≤ x ∧ x ≤ 186 ∧ (2 * x) % 187 = 1 :=
by
  use 94
  sorry

end inverse_mod_187_l475_47594


namespace slices_left_for_Era_l475_47563

def total_burgers : ℕ := 5
def slices_per_burger : ℕ := 8

def first_friend_slices : ℕ := 3
def second_friend_slices : ℕ := 8
def third_friend_slices : ℕ := 5
def fourth_friend_slices : ℕ := 11
def fifth_friend_slices : ℕ := 6

def total_slices : ℕ := total_burgers * slices_per_burger
def slices_given_to_friends : ℕ := first_friend_slices + second_friend_slices + third_friend_slices + fourth_friend_slices + fifth_friend_slices

theorem slices_left_for_Era : total_slices - slices_given_to_friends = 7 :=
by
  rw [total_slices, slices_given_to_friends]
  exact Eq.refl 7

#reduce slices_left_for_Era

end slices_left_for_Era_l475_47563


namespace Chemistry_marks_l475_47572

theorem Chemistry_marks (english_marks mathematics_marks physics_marks biology_marks : ℕ) (avg_marks : ℝ) (num_subjects : ℕ) (total_marks : ℕ)
  (h1 : english_marks = 72)
  (h2 : mathematics_marks = 60)
  (h3 : physics_marks = 35)
  (h4 : biology_marks = 84)
  (h5 : avg_marks = 62.6)
  (h6 : num_subjects = 5)
  (h7 : total_marks = avg_marks * num_subjects) :
  (total_marks - (english_marks + mathematics_marks + physics_marks + biology_marks) = 62) :=
by
  sorry

end Chemistry_marks_l475_47572


namespace smallest_angle_l475_47535

noncomputable def smallest_angle_in_triangle (a b c : ℝ) : ℝ :=
  if h : 0 <= a ∧ 0 <= b ∧ 0 <= c ∧ a + b > c ∧ a + c > b ∧ b + c > a then
    Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))
  else
    0

theorem smallest_angle (a b c : ℝ) (h₁ : a = 4) (h₂ : b = 3) (h₃ : c = 2) :
  smallest_angle_in_triangle a b c = Real.arccos (7 / 8) :=
sorry

end smallest_angle_l475_47535


namespace fraction_simplification_l475_47515

theorem fraction_simplification : (145^2 - 121^2) / 24 = 266 := by
  sorry

end fraction_simplification_l475_47515


namespace area_of_square_on_RS_l475_47583

theorem area_of_square_on_RS (PQ QR PS PS_square PQ_square QR_square : ℝ)
  (hPQ : PQ_square = 25) (hQR : QR_square = 49) (hPS : PS_square = 64)
  (hPQ_eq : PQ_square = PQ^2) (hQR_eq : QR_square = QR^2) (hPS_eq : PS_square = PS^2)
  : ∃ RS_square : ℝ, RS_square = 138 := by
  let PR_square := PQ^2 + QR^2
  let RS_square := PR_square + PS^2
  use RS_square
  sorry

end area_of_square_on_RS_l475_47583


namespace surjective_injective_eq_l475_47561

theorem surjective_injective_eq (f g : ℕ → ℕ) 
  (hf : Function.Surjective f) 
  (hg : Function.Injective g) 
  (h : ∀ n : ℕ, f n ≥ g n) : 
  ∀ n : ℕ, f n = g n := 
by
  sorry

end surjective_injective_eq_l475_47561


namespace points_among_transformations_within_square_l475_47559

def projection_side1 (A : ℝ × ℝ) : ℝ × ℝ := (A.1, 2 - A.2)
def projection_side2 (A : ℝ × ℝ) : ℝ × ℝ := (-A.1, A.2)
def projection_side3 (A : ℝ × ℝ) : ℝ × ℝ := (A.1, -A.2)
def projection_side4 (A : ℝ × ℝ) : ℝ × ℝ := (2 - A.1, A.2)

def within_square (A : ℝ × ℝ) : Prop := 
  0 ≤ A.1 ∧ A.1 ≤ 1 ∧ 0 ≤ A.2 ∧ A.2 ≤ 1

theorem points_among_transformations_within_square (A : ℝ × ℝ)
  (H1 : within_square A)
  (H2 : within_square (projection_side1 A))
  (H3 : within_square (projection_side2 (projection_side1 A)))
  (H4 : within_square (projection_side3 (projection_side2 (projection_side1 A))))
  (H5 : within_square (projection_side4 (projection_side3 (projection_side2 (projection_side1 A))))) :
  A = (1 / 3, 1 / 3) := sorry

end points_among_transformations_within_square_l475_47559


namespace unique_fish_total_l475_47528

-- Define the conditions as stated in the problem
def Micah_fish : ℕ := 7
def Kenneth_fish : ℕ := 3 * Micah_fish
def Matthias_fish : ℕ := Kenneth_fish - 15
def combined_fish : ℕ := Micah_fish + Kenneth_fish + Matthias_fish
def Gabrielle_fish : ℕ := 2 * combined_fish

def shared_fish_Micah_Matthias : ℕ := 4
def shared_fish_Kenneth_Gabrielle : ℕ := 6

-- Define the total unique fish computation
def total_unique_fish : ℕ := (Micah_fish + Kenneth_fish + Matthias_fish + Gabrielle_fish) - (shared_fish_Micah_Matthias + shared_fish_Kenneth_Gabrielle)

-- State the theorem
theorem unique_fish_total : total_unique_fish = 92 := by
  -- Proof omitted
  sorry

end unique_fish_total_l475_47528


namespace solve_cubic_equation_l475_47510

theorem solve_cubic_equation :
  ∀ x : ℝ, (x^3 + 2 * (x + 1)^3 + (x + 2)^3 = (x + 4)^3) → x = 3 :=
by
  intro x
  sorry

end solve_cubic_equation_l475_47510


namespace smallest_n_for_divisibility_property_l475_47526

theorem smallest_n_for_divisibility_property :
  ∃ n : ℕ, 0 < n ∧ (∀ k : ℕ, 1 ≤ k ∧ k ≤ n → n^2 + n % k = 0) ∧ 
  (∃ k : ℕ, 1 ≤ k ∧ k ≤ n ∧ n^2 + n % k ≠ 0) ∧ 
  ∀ m : ℕ, 0 < m ∧ m < n → ¬ ((∀ k : ℕ, 1 ≤ k ∧ k ≤ m → m^2 + m % k = 0) ∧ 
  (∃ k : ℕ, 1 ≤ k ∧ k ≤ m ∧ m^2 + m % k ≠ 0)) := sorry

end smallest_n_for_divisibility_property_l475_47526


namespace problem1_problem2_l475_47576

theorem problem1 : ∃ (m : ℝ) (b : ℝ), ∀ (x y : ℝ),
  3 * x + 4 * y - 2 = 0 ∧ x - y + 4 = 0 →
  y = m * x + b ∧ (1 / m = -2) ∧ (y = - (2 * x + 2)) :=
sorry

theorem problem2 : ∀ (x y a : ℝ), (x = -1) ∧ (y = 3) → 
  (x + y = a) →
  a = 2 ∧ (x + y - 2 = 0) :=
sorry

end problem1_problem2_l475_47576


namespace find_ordered_triple_l475_47586

theorem find_ordered_triple (a b c : ℝ) (h₁ : 2 < a) (h₂ : 2 < b) (h₃ : 2 < c)
    (h_eq : (a + 1)^2 / (b + c - 1) + (b + 2)^2 / (c + a - 3) + (c + 3)^2 / (a + b - 5) = 32) :
    (a = 8 ∧ b = 6 ∧ c = 5) :=
sorry

end find_ordered_triple_l475_47586


namespace Jamie_minimum_4th_quarter_score_l475_47539

theorem Jamie_minimum_4th_quarter_score (q1 q2 q3 : ℤ) (avg : ℤ) (minimum_score : ℤ) :
  q1 = 84 → q2 = 80 → q3 = 83 → avg = 85 → minimum_score = 93 → 4 * avg - (q1 + q2 + q3) = minimum_score :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  sorry

end Jamie_minimum_4th_quarter_score_l475_47539


namespace isosceles_right_triangle_hypotenuse_length_l475_47502

theorem isosceles_right_triangle_hypotenuse_length (A B C : ℝ) (h1 : (A = 0) ∧ (B = 0) ∧ (C = 1)) (h2 : AC = 5) (h3 : BC = 5) : 
  AB = 5 * Real.sqrt 2 := 
sorry

end isosceles_right_triangle_hypotenuse_length_l475_47502


namespace systematic_sampling_works_l475_47518

def missiles : List ℕ := List.range' 1 60 

-- Define the systematic sampling function
def systematic_sampling (start interval n : ℕ) : List ℕ :=
  List.range' 0 n |>.map (λ i => start + i * interval)

-- Stating the proof problem.
theorem systematic_sampling_works :
  systematic_sampling 5 12 5 = [5, 17, 29, 41, 53] :=
sorry

end systematic_sampling_works_l475_47518


namespace num_two_digit_numbers_l475_47551

-- Define the set of given digits
def digits : Finset ℕ := {0, 2, 5}

-- Define the function that counts the number of valid two-digit numbers
def count_two_digit_numbers (d : Finset ℕ) : ℕ :=
  (d.erase 0).card * (d.card - 1)

theorem num_two_digit_numbers : count_two_digit_numbers digits = 4 :=
by {
  -- sorry placeholder for the proof
  sorry
}

end num_two_digit_numbers_l475_47551


namespace shifted_parabola_equation_l475_47532

-- Define the original parabola function
def original_parabola (x : ℝ) : ℝ := -2 * x^2

-- Define the shifted parabola function
def shifted_parabola (x : ℝ) : ℝ := -2 * (x + 1)^2 + 3

-- Proposition to prove that the given parabola equation is correct after transformations
theorem shifted_parabola_equation : 
  ∀ x : ℝ, shifted_parabola x = -2 * (x + 1)^2 + 3 :=
by
  sorry

end shifted_parabola_equation_l475_47532


namespace side_length_of_square_l475_47527

theorem side_length_of_square (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
  (a b c : ℝ) (h_leg1 : a = 12) (h_leg2 : b = 9) (h_right : c^2 = a^2 + b^2) :
  ∃ s : ℝ, s = 45/8 :=
by 
  -- Given the right triangle with legs 12 cm and 9 cm, the length of the side of the square is 45/8 cm
  let s := 45/8
  use s
  sorry

end side_length_of_square_l475_47527


namespace pq_implications_l475_47507

theorem pq_implications (p q : Prop) (hpq_or : p ∨ q) (hpq_and : p ∧ q) : p ∧ q :=
by
  sorry

end pq_implications_l475_47507


namespace parabola_standard_equation_oa_dot_ob_value_line_passes_fixed_point_l475_47514

-- Definitions for the problem conditions
def parabola_symmetry_axis := "coordinate axis"
def parabola_vertex := (0, 0)
def directrix_equation := "x = -1"
def intersects_at_two_points (l : ℝ → ℝ) (P : ℝ × ℝ) (Q : ℝ × ℝ) := (l P.1 = P.2) ∧ (l Q.1 = Q.2) ∧ (P ≠ Q)

-- Main theorem statements
theorem parabola_standard_equation : 
  (parabola_symmetry_axis = "coordinate axis") ∧ 
  (parabola_vertex = (0, 0)) ∧ 
  (directrix_equation = "x = -1") → 
  ∃ p, 0 < p ∧ ∀ y x, y^2 = 4 * p * x := 
  sorry

theorem oa_dot_ob_value (l : ℝ → ℝ) (focus : ℝ × ℝ) (P : ℝ × ℝ) (Q : ℝ × ℝ) : 
  (parabola_symmetry_axis = "coordinate axis") ∧ 
  (parabola_vertex = (0, 0)) ∧ 
  (directrix_equation = "x = -1") ∧ 
  intersects_at_two_points l P Q ∧ 
  l focus.1 = focus.2 → 
  (P.1 * Q.1 + P.2 * Q.2 = -3) := 
  sorry

theorem line_passes_fixed_point (l : ℝ → ℝ) (P : ℝ × ℝ) (Q : ℝ × ℝ) : 
  (parabola_symmetry_axis = "coordinate axis") ∧ 
  (parabola_vertex = (0, 0)) ∧ 
  (directrix_equation = "x = -1") ∧ 
  intersects_at_two_points l P Q ∧ 
  (P.1 * Q.1 + P.2 * Q.2 = -4) → 
  ∃ fp, fp = (2,0) := 
  sorry

end parabola_standard_equation_oa_dot_ob_value_line_passes_fixed_point_l475_47514


namespace seven_balls_expected_positions_l475_47519

theorem seven_balls_expected_positions :
  let n := 7
  let swaps := 4
  let p_stay := (1 - 2/7)^4 + 6 * (2/7)^2 * (5/7)^2 + (2/7)^4
  let expected_positions := n * p_stay
  expected_positions = 3.61 :=
by
  let n := 7
  let swaps := 4
  let p_stay := (1 - 2/7)^4 + 6 * (2/7)^2 * (5/7)^2 + (2/7)^4
  let expected_positions := n * p_stay
  exact sorry

end seven_balls_expected_positions_l475_47519


namespace distance_to_nearest_edge_of_picture_l475_47523

def wall_width : ℕ := 26
def picture_width : ℕ := 4
def distance_from_end (wall picture : ℕ) : ℕ := (wall - picture) / 2

theorem distance_to_nearest_edge_of_picture :
  distance_from_end wall_width picture_width = 11 :=
sorry

end distance_to_nearest_edge_of_picture_l475_47523


namespace triangle_area_290_l475_47511

theorem triangle_area_290 
  (P Q R : ℝ × ℝ)
  (h1 : (R.1 - P.1) * (R.1 - Q.1) + (R.2 - P.2) * (R.2 - Q.2) = 0) -- Right triangle condition
  (h2 : (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = 50^2) -- Length of hypotenuse PQ
  (h3 : ∀ x: ℝ, (x, x - 2) = P) -- Median through P
  (h4 : ∀ x: ℝ, (x, 3 * x + 3) = Q) -- Median through Q
  :
  ∃ (area : ℝ), area = 290 := 
sorry

end triangle_area_290_l475_47511


namespace polynomial_divisibility_l475_47534

theorem polynomial_divisibility (a b : ℕ) (h1 : a > 0) (h2 : b > 0) 
    (h3 : (a + b^3) % (a^2 + 3 * a * b + 3 * b^2 - 1) = 0) : 
    ∃ k : ℕ, k ≥ 1 ∧ (a^2 + 3 * a * b + 3 * b^2 - 1) % k^3 = 0 :=
    sorry

end polynomial_divisibility_l475_47534


namespace geometric_series_sum_l475_47541

/-- 
The series is given as 1/2^2 + 1/2^3 + 1/2^4 + 1/2^5 + 1/2^6 + 1/2^7 + 1/2^8.
First term a = 1/4 and common ratio r = 1/2 and number of terms n = 7. 
The sum should be 127/256.
-/
theorem geometric_series_sum :
  let a := 1 / 4
  let r := 1 / 2
  let n := 7
  let S := (a * (1 - r^n)) / (1 - r)
  S = 127 / 256 :=
by
  sorry

end geometric_series_sum_l475_47541


namespace calculation_correct_l475_47579

theorem calculation_correct : -2 + 3 = 1 :=
by
  sorry

end calculation_correct_l475_47579


namespace max_consecutive_semi_primes_l475_47549

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_semi_prime (n : ℕ) : Prop := 
  n > 25 ∧ ∃ (p q : ℕ), is_prime p ∧ is_prime q ∧ p ≠ q ∧ n = p + q

theorem max_consecutive_semi_primes : ∃ (N : ℕ), N = 5 ∧
  ∀ (a b : ℕ), (a > 25) ∧ (b = a + 4) → 
  (∀ n, a ≤ n ∧ n ≤ b → is_semi_prime n) ↔ N = 5 := sorry

end max_consecutive_semi_primes_l475_47549


namespace volume_of_box_ground_area_of_box_l475_47531

-- Given conditions
variable (l w h : ℕ)
variable (hl : l = 20)
variable (hw : w = 15)
variable (hh : h = 5)

-- Define volume and ground area
def volume (l w h : ℕ) : ℕ := l * w * h
def ground_area (l w : ℕ) : ℕ := l * w

-- Theorem to prove the correct volume
theorem volume_of_box : volume l w h = 1500 := by
  rw [hl, hw, hh]
  sorry

-- Theorem to prove the correct ground area
theorem ground_area_of_box : ground_area l w = 300 := by
  rw [hl, hw]
  sorry

end volume_of_box_ground_area_of_box_l475_47531


namespace find_a_8_l475_47513

-- Define the arithmetic sequence and its sum formula.
def arithmetic_sequence (a : ℕ → ℕ) (d : ℕ) :=
  ∀ n, a (n + 1) = a n + d

-- Define the sum of the first 'n' terms in the arithmetic sequence.
def sum_of_first_n_terms (S : ℕ → ℕ) (a : ℕ → ℕ) (n : ℕ) :=
  S n = n * (a 1 + a n) / 2

-- Given conditions
def S_15_eq_90 (S : ℕ → ℕ) : Prop := S 15 = 90

-- Prove that a_8 is 6
theorem find_a_8 (S : ℕ → ℕ) (a : ℕ → ℕ) (d : ℕ)
  (h1 : arithmetic_sequence a d) (h2 : sum_of_first_n_terms S a 15)
  (h3 : S_15_eq_90 S) : a 8 = 6 :=
sorry

end find_a_8_l475_47513


namespace Zachary_sold_40_games_l475_47573

theorem Zachary_sold_40_games 
  (R J Z : ℝ)
  (games_Zachary_sold : ℕ)
  (h1 : R = J + 50)
  (h2 : J = 1.30 * Z)
  (h3 : Z = 5 * games_Zachary_sold)
  (h4 : Z + J + R = 770) :
  games_Zachary_sold = 40 :=
by
  sorry

end Zachary_sold_40_games_l475_47573


namespace find_t_l475_47581

noncomputable def g (x : ℝ) (p q s t : ℝ) : ℝ :=
  x^4 + p*x^3 + q*x^2 + s*x + t

theorem find_t {p q s t : ℝ}
  (h1 : ∀ r : ℝ, g r p q s t = 0 → r < 0 ∧ Int.mod (round r) 2 = 1)
  (h2 : p + q + s + t = 2047) :
  t = 5715 :=
sorry

end find_t_l475_47581


namespace club_members_l475_47538

variable (x : ℕ)

theorem club_members (h1 : 2 * x + 5 = x + 15) : x = 10 := by
  sorry

end club_members_l475_47538


namespace inequality_problem_l475_47587

theorem inequality_problem (x a : ℝ) (h1 : x < a) (h2 : a < 0) : x^2 > ax ∧ ax > a^2 :=
by {
  sorry
}

end inequality_problem_l475_47587

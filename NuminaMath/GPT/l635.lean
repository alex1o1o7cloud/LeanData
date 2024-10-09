import Mathlib

namespace passes_through_point_l635_63545

theorem passes_through_point (a : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1) : (1, 1) ∈ {p : ℝ × ℝ | ∃ x, p = (x, a^(x-1))} :=
by
  sorry

end passes_through_point_l635_63545


namespace Jungkook_has_the_largest_number_l635_63526

theorem Jungkook_has_the_largest_number :
  let Yoongi := 4
  let Yuna := 5
  let Jungkook := 6 + 3
  Jungkook > Yoongi ∧ Jungkook > Yuna := by
    sorry

end Jungkook_has_the_largest_number_l635_63526


namespace drive_time_is_eleven_hours_l635_63578

-- Define the distances and speed as constants
def distance_salt_lake_to_vegas : ℕ := 420
def distance_vegas_to_los_angeles : ℕ := 273
def average_speed : ℕ := 63

-- Calculate the total distance
def total_distance : ℕ := distance_salt_lake_to_vegas + distance_vegas_to_los_angeles

-- Calculate the total time required
def total_time : ℕ := total_distance / average_speed

-- Theorem stating Andy wants to complete the drive in 11 hours
theorem drive_time_is_eleven_hours : total_time = 11 := sorry

end drive_time_is_eleven_hours_l635_63578


namespace complex_equation_solution_l635_63533

theorem complex_equation_solution (x : ℝ) (i : ℂ) (h_imag_unit : i * i = -1) (h_eq : (x + 2 * i) * (x - i) = 6 + 2 * i) : x = 2 :=
by
  sorry

end complex_equation_solution_l635_63533


namespace savings_calculation_l635_63562

theorem savings_calculation (income expenditure savings : ℕ) (ratio_income ratio_expenditure : ℕ)
  (h_ratio : ratio_income = 10) (h_ratio2 : ratio_expenditure = 7) (h_income : income = 10000)
  (h_expenditure : 10 * expenditure = 7 * income) :
  savings = income - expenditure :=
by
  sorry

end savings_calculation_l635_63562


namespace soccer_games_per_month_l635_63505

theorem soccer_games_per_month (total_games : ℕ) (months : ℕ) (h1 : total_games = 27) (h2 : months = 3) : total_games / months = 9 :=
by 
  sorry

end soccer_games_per_month_l635_63505


namespace total_time_proof_l635_63574

variable (mow_time : ℕ) (fertilize_time : ℕ) (total_time : ℕ)

-- Based on the problem conditions.
axiom mow_time_def : mow_time = 40
axiom fertilize_time_def : fertilize_time = 2 * mow_time
axiom total_time_def : total_time = mow_time + fertilize_time

-- The proof goal
theorem total_time_proof : total_time = 120 := by
  sorry

end total_time_proof_l635_63574


namespace total_number_of_animals_l635_63583

theorem total_number_of_animals 
  (rabbits ducks chickens : ℕ)
  (h1 : chickens = 5 * ducks)
  (h2 : ducks = rabbits + 12)
  (h3 : rabbits = 4) : 
  chickens + ducks + rabbits = 100 :=
by
  sorry

end total_number_of_animals_l635_63583


namespace find_m_plus_n_l635_63567

noncomputable def overlapping_points (A B: ℝ × ℝ) (C D: ℝ × ℝ) : Prop :=
  let k_AB := (B.2 - A.2) / (B.1 - A.1)
  let M_AB := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  let axis_slope := - 1 / k_AB
  let k_CD := (D.2 - C.2) / (D.1 - C.1)
  let M_CD := ((C.1 + D.1) / 2, (C.2 + D.2) / 2)
  k_CD = axis_slope ∧ (M_CD.2 - M_AB.2) = axis_slope * (M_CD.1 - M_AB.1)

theorem find_m_plus_n : 
  ∃ (m n: ℝ), overlapping_points (0, 2) (4, 0) (7, 3) (m, n) ∧ m + n = 34 / 5 :=
sorry

end find_m_plus_n_l635_63567


namespace biased_coin_probability_l635_63590

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- Define the probability mass function for a binomial distribution
def binomial_pmf (n k : ℕ) (p : ℝ) : ℝ :=
  binomial n k * p^k * (1 - p)^(n - k)

-- Define the problem conditions
def problem_conditions : Prop :=
  let p := 1 / 3
  binomial_pmf 5 1 p = binomial_pmf 5 2 p ∧ p ≠ 0 ∧ (1 - p) ≠ 0

-- The target probability to prove
def target_probability := 40 / 243

-- The theorem statement
theorem biased_coin_probability : problem_conditions → binomial_pmf 5 3 (1 / 3) = target_probability :=
by
  intro h
  sorry

end biased_coin_probability_l635_63590


namespace y_coordinate_of_point_l635_63572

theorem y_coordinate_of_point (x y : ℝ) (m : ℝ)
  (h₁ : x = 10)
  (h₂ : y = m * x + -2)
  (m_def : m = (0 - (-4)) / (4 - (-4)))
  (h₃ : y = 3) : y = 3 :=
sorry

end y_coordinate_of_point_l635_63572


namespace probability_of_events_l635_63580

-- Define the sets of tiles in each box
def boxA : Set ℕ := {n | 1 ≤ n ∧ n ≤ 25}
def boxB : Set ℕ := {n | 15 ≤ n ∧ n ≤ 40}

-- Define the specific conditions
def eventA (tile : ℕ) : Prop := tile ≤ 20
def eventB (tile : ℕ) : Prop := (Odd tile ∨ tile > 35)

-- Define the probabilities as calculations
def prob_eventA : ℚ := 20 / 25
def prob_eventB : ℚ := 15 / 26

-- The final probability given independence
def combined_prob : ℚ := prob_eventA * prob_eventB

-- The theorem statement we want to prove
theorem probability_of_events :
  combined_prob = 6 / 13 := 
by 
  -- proof details would go here
  sorry

end probability_of_events_l635_63580


namespace probability_of_graduate_degree_l635_63565

-- Define the conditions as Lean statements
variable (k m : ℕ)
variable (G := 1 * k) 
variable (C := 2 * m) 
variable (N1 := 8 * k) -- from the ratio G:N = 1:8
variable (N2 := 3 * m) -- from the ratio C:N = 2:3

-- Least common multiple (LCM) of 8 and 3 is 24
-- Therefore, determine specific values for G, C, and N
-- Given these updates from solution steps we set:
def G_scaled : ℕ := 3
def C_scaled : ℕ := 16
def N_scaled : ℕ := 24

-- Total number of college graduates
def total_college_graduates : ℕ := G_scaled + C_scaled

-- Probability q of picking a college graduate with a graduate degree
def q : ℚ := G_scaled / total_college_graduates

-- Lean proof statement for equivalence
theorem probability_of_graduate_degree : 
  q = 3 / 19 := by
sorry

end probability_of_graduate_degree_l635_63565


namespace simplify_and_evaluate_expression_l635_63599

theorem simplify_and_evaluate_expression :
  (2 * (-1/2) + 3 * 1)^2 - (2 * (-1/2) + 1) * (2 * (-1/2) - 1) = 4 :=
by
  sorry

end simplify_and_evaluate_expression_l635_63599


namespace largest_possible_average_l635_63582

noncomputable def ten_test_scores (a b c d e f g h i j : ℤ) : ℤ :=
  a + b + c + d + e + f + g + h + i + j

theorem largest_possible_average
  (a b c d e f g h i j : ℤ)
  (h1 : 0 ≤ a ∧ a ≤ 100)
  (h2 : 0 ≤ b ∧ b ≤ 100)
  (h3 : 0 ≤ c ∧ c ≤ 100)
  (h4 : 0 ≤ d ∧ d ≤ 100)
  (h5 : 0 ≤ e ∧ e ≤ 100)
  (h6 : 0 ≤ f ∧ f ≤ 100)
  (h7 : 0 ≤ g ∧ g ≤ 100)
  (h8 : 0 ≤ h ∧ h ≤ 100)
  (h9 : 0 ≤ i ∧ i ≤ 100)
  (h10 : 0 ≤ j ∧ j ≤ 100)
  (h11 : a + b + c + d ≤ 190)
  (h12 : b + c + d + e ≤ 190)
  (h13 : c + d + e + f ≤ 190)
  (h14 : d + e + f + g ≤ 190)
  (h15 : e + f + g + h ≤ 190)
  (h16 : f + g + h + i ≤ 190)
  (h17 : g + h + i + j ≤ 190)
  : ((ten_test_scores a b c d e f g h i j : ℚ) / 10) ≤ 44.33 := sorry

end largest_possible_average_l635_63582


namespace distance_between_foci_of_ellipse_l635_63559

-- Define the three given points
structure Point where
  x : ℝ
  y : ℝ

def p1 : Point := ⟨1, 3⟩
def p2 : Point := ⟨5, -1⟩
def p3 : Point := ⟨10, 3⟩

-- Define the statement that the distance between the foci of the ellipse they define is 2 * sqrt(4.25)
theorem distance_between_foci_of_ellipse : 
  ∃ (c : ℝ) (f : ℝ), f = 2 * Real.sqrt 4.25 ∧ 
  (∃ (ellipse : Point → Prop), ellipse p1 ∧ ellipse p2 ∧ ellipse p3) :=
sorry

end distance_between_foci_of_ellipse_l635_63559


namespace units_digit_17_pow_2007_l635_63521

theorem units_digit_17_pow_2007 : (17^2007) % 10 = 3 :=
by sorry

end units_digit_17_pow_2007_l635_63521


namespace notepad_last_duration_l635_63577

def note_duration (folds_per_paper : ℕ) (pieces_of_paper : ℕ) (notes_per_day : ℕ) : ℕ :=
  let note_size_papers_per_letter_paper := 2 ^ folds_per_paper
  let total_note_size_papers := pieces_of_paper * note_size_papers_per_letter_paper
  total_note_size_papers / notes_per_day

theorem notepad_last_duration :
  note_duration 3 5 10 = 4 := by
  sorry

end notepad_last_duration_l635_63577


namespace quincy_more_stuffed_animals_l635_63519

theorem quincy_more_stuffed_animals (thor_sold jake_sold quincy_sold : ℕ) 
  (h1 : jake_sold = thor_sold + 10) 
  (h2 : quincy_sold = 10 * thor_sold) 
  (h3 : quincy_sold = 200) : 
  quincy_sold - jake_sold = 170 :=
by sorry

end quincy_more_stuffed_animals_l635_63519


namespace white_clothing_probability_l635_63551

theorem white_clothing_probability (total_athletes sample_size k_min k_max : ℕ) 
  (red_upper_bound white_upper_bound yellow_upper_bound sampled_start_interval : ℕ)
  (h_total : total_athletes = 600)
  (h_sample : sample_size = 50)
  (h_intervals : total_athletes / sample_size = 12)
  (h_group_start : sampled_start_interval = 4)
  (h_red_upper : red_upper_bound = 311)
  (h_white_upper : white_upper_bound = 496)
  (h_yellow_upper : yellow_upper_bound = 600)
  (h_k_min : k_min = 26)   -- Calculated from 312 <= 12k + 4
  (h_k_max : k_max = 41)  -- Calculated from 12k + 4 <= 496
  : (k_max - k_min + 1) / sample_size = 8 / 25 := 
by
  sorry

end white_clothing_probability_l635_63551


namespace journey_speed_l635_63537

theorem journey_speed (v : ℝ) 
  (h1 : 3 * v + 60 * 2 = 240)
  (h2 : 3 + 2 = 5) :
  v = 40 :=
by
  sorry

end journey_speed_l635_63537


namespace sum_of_cubes_of_roots_l635_63598

theorem sum_of_cubes_of_roots :
  ∀ (x1 x2 : ℝ), (2 * x1^2 - 5 * x1 + 1 = 0) ∧ (2 * x2^2 - 5 * x2 + 1 = 0) →
  (x1 + x2 = 5 / 2) ∧ (x1 * x2 = 1 / 2) →
  (x1^3 + x2^3 = 95 / 8) :=
by
  sorry

end sum_of_cubes_of_roots_l635_63598


namespace solve_inequality_l635_63538

theorem solve_inequality (x : ℝ) :
  (4 ≤ x^2 - 3 * x - 6 ∧ x^2 - 3 * x - 6 ≤ 2 * x + 8) ↔ (5 ≤ x ∧ x ≤ 7 ∨ x = -2) :=
by
  sorry

end solve_inequality_l635_63538


namespace increase_to_restore_l635_63506

noncomputable def percentage_increase_to_restore (P : ℝ) : ℝ :=
  let reduced_price := 0.9 * P
  let restore_factor := P / reduced_price
  (restore_factor - 1) * 100

theorem increase_to_restore :
  percentage_increase_to_restore 100 = 100 / 9 :=
by
  sorry

end increase_to_restore_l635_63506


namespace inequalities_hold_l635_63556

variable {a b c x y z : ℝ}

theorem inequalities_hold 
  (h1 : x ≤ a)
  (h2 : y ≤ b)
  (h3 : z ≤ c) :
  x * y + y * z + z * x ≤ a * b + b * c + c * a ∧
  x^2 + y^2 + z^2 ≤ a^2 + b^2 + c^2 ∧
  x * y * z ≤ a * b * c :=
sorry

end inequalities_hold_l635_63556


namespace solution_set_of_gx_lt_0_l635_63530

noncomputable def f (x : ℝ) : ℝ := 2 ^ x

noncomputable def f_inv (x : ℝ) : ℝ := Real.log x / Real.log 2

noncomputable def g (x : ℝ) : ℝ := f_inv (1 - x) - f_inv (1 + x)

theorem solution_set_of_gx_lt_0 : { x : ℝ | g x < 0 } = Set.Ioo 0 1 := by
  sorry

end solution_set_of_gx_lt_0_l635_63530


namespace total_cost_correct_l635_63504

def football_cost : ℝ := 5.71
def marbles_cost : ℝ := 6.59
def total_cost : ℝ := 12.30

theorem total_cost_correct : football_cost + marbles_cost = total_cost := 
by
  sorry

end total_cost_correct_l635_63504


namespace prove_values_of_a_and_b_prove_range_of_k_l635_63542

variable {f : ℝ → ℝ}

-- (1) Prove values of a and b
theorem prove_values_of_a_and_b (h_odd : ∀ x : ℝ, f (-x) = -f x) :
  (∀ x, f x = 2 * x - 1) := by
sorry

-- (2) Prove range of k
theorem prove_range_of_k (h_fx_2x_minus_1 : ∀ x : ℝ, f x = 2 * x - 1) :
  (∀ t : ℝ, f (t^2 - 2 * t) + f (2 * t^2 - k) < 0) ↔ k < -1 / 3 := by
sorry

end prove_values_of_a_and_b_prove_range_of_k_l635_63542


namespace total_cost_of_tickets_l635_63529

def number_of_adults := 2
def number_of_children := 3
def cost_of_adult_ticket := 19
def cost_of_child_ticket := cost_of_adult_ticket - 6

theorem total_cost_of_tickets :
  let total_cost := number_of_adults * cost_of_adult_ticket + number_of_children * cost_of_child_ticket
  total_cost = 77 :=
by
  sorry

end total_cost_of_tickets_l635_63529


namespace avg_price_of_pencil_l635_63534

theorem avg_price_of_pencil 
  (total_pens : ℤ) (total_pencils : ℤ) (total_cost : ℤ)
  (avg_cost_pen : ℤ) (avg_cost_pencil : ℤ) :
  total_pens = 30 → 
  total_pencils = 75 → 
  total_cost = 690 → 
  avg_cost_pen = 18 → 
  (total_cost - total_pens * avg_cost_pen) / total_pencils = avg_cost_pencil → 
  avg_cost_pencil = 2 :=
by
  intros
  sorry

end avg_price_of_pencil_l635_63534


namespace rachel_picked_2_apples_l635_63591

def apples_picked (initial_apples picked_apples final_apples : ℕ) : Prop :=
  initial_apples - picked_apples = final_apples

theorem rachel_picked_2_apples (initial_apples final_apples : ℕ)
  (h_initial : initial_apples = 9)
  (h_final : final_apples = 7) :
  apples_picked initial_apples 2 final_apples :=
by
  rw [h_initial, h_final]
  sorry

end rachel_picked_2_apples_l635_63591


namespace tory_needs_to_sell_more_packs_l635_63558

theorem tory_needs_to_sell_more_packs 
  (total_goal : ℤ) (packs_grandmother : ℤ) (packs_uncle : ℤ) (packs_neighbor : ℤ) 
  (total_goal_eq : total_goal = 50)
  (packs_grandmother_eq : packs_grandmother = 12)
  (packs_uncle_eq : packs_uncle = 7)
  (packs_neighbor_eq : packs_neighbor = 5) :
  total_goal - (packs_grandmother + packs_uncle + packs_neighbor) = 26 :=
by
  rw [total_goal_eq, packs_grandmother_eq, packs_uncle_eq, packs_neighbor_eq]
  norm_num

end tory_needs_to_sell_more_packs_l635_63558


namespace arithmetic_sequence_problem_l635_63589

theorem arithmetic_sequence_problem
  (a : ℕ → ℕ)
  (h1 : a 2 + a 3 = 15)
  (h2 : a 3 + a 4 = 20) :
  a 4 + a 5 = 25 :=
sorry

end arithmetic_sequence_problem_l635_63589


namespace smallest_mu_ineq_l635_63592

theorem smallest_mu_ineq (a b c d : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hd : 0 ≤ d) :
    a^2 + b^2 + c^2 + d^2 + 2 * a * d ≥ 2 * (a * b + b * c + c * d) := by {
    sorry
}

end smallest_mu_ineq_l635_63592


namespace intersection_of_lines_l635_63548

-- Define the first and second lines
def line1 (x : ℚ) : ℚ := 3 * x + 1
def line2 (x : ℚ) : ℚ := -7 * x - 5

-- Statement: Prove that the intersection of the lines given by
-- y = 3x + 1 and y + 5 = -7x is (-3/5, -4/5).

theorem intersection_of_lines :
  ∃ x y : ℚ, y = line1 x ∧ y = line2 x ∧ x = -3 / 5 ∧ y = -4 / 5 :=
by
  sorry

end intersection_of_lines_l635_63548


namespace total_surface_area_of_pyramid_l635_63536

noncomputable def base_length_ab : ℝ := 8 -- Length of side AB
noncomputable def base_length_ad : ℝ := 6 -- Length of side AD
noncomputable def height_pf : ℝ := 15 -- Perpendicular height from peak P to the base's center F

noncomputable def base_area : ℝ := base_length_ab * base_length_ad
noncomputable def fm_distance : ℝ := Real.sqrt ((base_length_ab / 2)^2 + (base_length_ad / 2)^2)
noncomputable def slant_height_pm : ℝ := Real.sqrt (height_pf^2 + fm_distance^2)

noncomputable def lateral_area_ab : ℝ := 2 * (0.5 * base_length_ab * slant_height_pm)
noncomputable def lateral_area_ad : ℝ := 2 * (0.5 * base_length_ad * slant_height_pm)
noncomputable def total_surface_area : ℝ := base_area + lateral_area_ab + lateral_area_ad

theorem total_surface_area_of_pyramid :
  total_surface_area = 48 + 55 * Real.sqrt 10 := by
  sorry

end total_surface_area_of_pyramid_l635_63536


namespace polynomial_zero_iff_divisibility_l635_63532

theorem polynomial_zero_iff_divisibility (P : Polynomial ℤ) :
  (∀ n : ℕ, n > 0 → ∃ k : ℤ, P.eval (2^n) = n * k) ↔ P = 0 :=
by sorry

end polynomial_zero_iff_divisibility_l635_63532


namespace inequality_proof_l635_63576

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (habc : a * b * (1 / (a * b)) = 1) :
  a^2 + b^2 + (1 / (a * b))^2 + 3 ≥ 2 * (1 / a + 1 / b + a * b) := 
by sorry

end inequality_proof_l635_63576


namespace sum_midpoint_x_coords_l635_63516

theorem sum_midpoint_x_coords (a b c : ℝ) (h1 : a + b + c = 15) (h2 : a - b = 3) :
    (a + (a - 3)) / 2 + (a + c) / 2 + ((a - 3) + c) / 2 = 15 := 
by 
  sorry

end sum_midpoint_x_coords_l635_63516


namespace equal_sum_seq_value_at_18_l635_63588

-- Define what it means for a sequence to be an equal-sum sequence with a common sum
def equal_sum_seq (a : ℕ → ℤ) (c : ℤ) : Prop :=
  ∀ n : ℕ, a n + a (n + 1) = c

theorem equal_sum_seq_value_at_18
  (a : ℕ → ℤ)
  (h1 : a 1 = 2)
  (h2 : equal_sum_seq a 5) :
  a 18 = 3 :=
sorry

end equal_sum_seq_value_at_18_l635_63588


namespace supplement_of_complement_of_35_degree_angle_l635_63535

def complement (α : ℝ) : ℝ := 90 - α
def supplement (α : ℝ) : ℝ := 180 - α

theorem supplement_of_complement_of_35_degree_angle : 
  supplement (complement 35) = 125 := 
by sorry

end supplement_of_complement_of_35_degree_angle_l635_63535


namespace problem1_l635_63520

theorem problem1 :
  (15 * (-3 / 4) + (-15) * (3 / 2) + 15 / 4) = -30 :=
by
  sorry

end problem1_l635_63520


namespace number_of_total_flowers_l635_63585

theorem number_of_total_flowers :
  let n_pots := 141
  let flowers_per_pot := 71
  n_pots * flowers_per_pot = 10011 :=
by
  sorry

end number_of_total_flowers_l635_63585


namespace alex_shirts_l635_63566

theorem alex_shirts (shirts_joe shirts_alex shirts_ben : ℕ) 
  (h1 : shirts_joe = shirts_alex + 3) 
  (h2 : shirts_ben = shirts_joe + 8) 
  (h3 : shirts_ben = 15) : shirts_alex = 4 :=
by
  sorry

end alex_shirts_l635_63566


namespace inequality_for_positive_reals_l635_63568

theorem inequality_for_positive_reals (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (1 / (a + b) + 1 / (b + c) + 1 / (c + a)) ≥ ((a + b + c) ^ 2 / (a * b * (a + b) + b * c * (b + c) + c * a * (c + a))) :=
by
  sorry

end inequality_for_positive_reals_l635_63568


namespace max_lateral_surface_area_of_pyramid_l635_63524

theorem max_lateral_surface_area_of_pyramid (a h : ℝ) (r : ℝ) (h_eq : 2 * a^2 + h^2 = 4) (r_eq : r = 1) :
  ∃ (a : ℝ), (a = 1) :=
by
sorry

end max_lateral_surface_area_of_pyramid_l635_63524


namespace parallel_lines_intersect_hyperbola_l635_63550

noncomputable def point_A : (ℝ × ℝ) := (0, 14)
noncomputable def point_B : (ℝ × ℝ) := (0, 4)
noncomputable def hyperbola (x : ℝ) : ℝ := 1 / x

theorem parallel_lines_intersect_hyperbola (k : ℝ)
  (x_K x_L x_M x_N : ℝ) 
  (hAK : hyperbola x_K = k * x_K + 14) (hAL : hyperbola x_L = k * x_L + 14)
  (hBM : hyperbola x_M = k * x_M + 4) (hBN : hyperbola x_N = k * x_N + 4)
  (vieta1 : x_K + x_L = -14 / k) (vieta2 : x_M + x_N = -4 / k) :
  (AL - AK) / (BN - BM) = 3.5 :=
by
  sorry

end parallel_lines_intersect_hyperbola_l635_63550


namespace set_difference_is_single_element_l635_63563

-- Define the sets M and N based on the given conditions
def M : Set ℕ := {x | 1 ≤ x ∧ x ≤ 2002}
def N : Set ℕ := {y | 2 ≤ y ∧ y ≤ 2003}

-- State the theorem that we need to prove
theorem set_difference_is_single_element : (N \ M) = {2003} :=
sorry

end set_difference_is_single_element_l635_63563


namespace river_and_building_geometry_l635_63527

open Real

theorem river_and_building_geometry (x y : ℝ) :
  (tan 60 * x = y) ∧ (tan 30 * (x + 30) = y) → x = 15 ∧ y = 15 * sqrt 3 :=
by
  sorry

end river_and_building_geometry_l635_63527


namespace compute_expression_l635_63543

theorem compute_expression : 12 * (1 / 7) * 14 * 2 = 48 := 
sorry

end compute_expression_l635_63543


namespace selection_ways_l635_63553

def ways_to_select_president_and_secretary (n : Nat) : Nat :=
  n * (n - 1)

theorem selection_ways :
  ways_to_select_president_and_secretary 5 = 20 :=
by
  sorry

end selection_ways_l635_63553


namespace possible_values_of_n_l635_63552

open Nat

noncomputable def a (n : ℕ) : ℕ := 2 * n - 1

noncomputable def b (n : ℕ) : ℕ := 2 ^ (n - 1)

noncomputable def c (n : ℕ) : ℕ := a (b n)

noncomputable def T (n : ℕ) : ℕ := (Finset.range n).sum (λ i => c (i + 1))

theorem possible_values_of_n (n : ℕ) :
  T n < 2021 → n = 8 ∨ n = 9 := by
  sorry

end possible_values_of_n_l635_63552


namespace problem1_calculation_l635_63581

theorem problem1_calculation :
  (2 * Real.tan (Real.pi / 4) + (-1 / 2) ^ 0 + |Real.sqrt 3 - 1|) = 2 + Real.sqrt 3 :=
by
  sorry

end problem1_calculation_l635_63581


namespace average_of_data_set_l635_63587

theorem average_of_data_set :
  (7 + 5 + (-2) + 5 + 10) / 5 = 5 :=
by sorry

end average_of_data_set_l635_63587


namespace percentage_of_ginger_is_correct_l635_63539

noncomputable def teaspoons_per_tablespoon : ℕ := 3
noncomputable def ginger_tablespoons : ℕ := 3
noncomputable def cardamom_teaspoons : ℕ := 1
noncomputable def mustard_teaspoons : ℕ := 1
noncomputable def garlic_tablespoons : ℕ := 2
noncomputable def chile_powder_factor : ℕ := 4

theorem percentage_of_ginger_is_correct :
  let ginger_teaspoons := ginger_tablespoons * teaspoons_per_tablespoon
  let garlic_teaspoons := garlic_tablespoons * teaspoons_per_tablespoon
  let chile_teaspoons := chile_powder_factor * mustard_teaspoons
  let total_teaspoons := ginger_teaspoons + cardamom_teaspoons + mustard_teaspoons + garlic_teaspoons + chile_teaspoons
  let percentage_ginger := (ginger_teaspoons * 100) / total_teaspoons
  percentage_ginger = 43 :=
by
  sorry

end percentage_of_ginger_is_correct_l635_63539


namespace probability_correct_l635_63517
noncomputable def probability_no_2_in_id : ℚ :=
  let total_ids := 5000
  let valid_ids := 2916
  valid_ids / total_ids

theorem probability_correct : probability_no_2_in_id = 729 / 1250 := by
  sorry

end probability_correct_l635_63517


namespace final_position_correct_l635_63546

structure Position :=
(base : ℝ × ℝ)
(stem : ℝ × ℝ)

def initial_position : Position :=
{ base := (0, -1),
  stem := (1, 0) }

def reflect_x (p : Position) : Position :=
{ base := (p.base.1, -p.base.2),
  stem := (p.stem.1, -p.stem.2) }

def rotate_90_ccw (p : Position) : Position :=
{ base := (-p.base.2, p.base.1),
  stem := (-p.stem.2, p.stem.1) }

def half_turn (p : Position) : Position :=
{ base := (-p.base.1, -p.base.2),
  stem := (-p.stem.1, -p.stem.2) }

def reflect_y (p : Position) : Position :=
{ base := (-p.base.1, p.base.2),
  stem := (-p.stem.1, p.stem.2) }

def final_position : Position :=
reflect_y (half_turn (rotate_90_ccw (reflect_x initial_position)))

theorem final_position_correct : final_position = { base := (1, 0), stem := (0, 1) } :=
sorry

end final_position_correct_l635_63546


namespace maximize_det_l635_63557

theorem maximize_det (a b : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_sum : a + b = 2) : 
  (Matrix.det ![
    ![a, 1],
    ![1, b]
  ]) ≤ 0 :=
sorry

end maximize_det_l635_63557


namespace total_animals_received_l635_63596

-- Define the conditions
def cats : ℕ := 40
def additionalCats : ℕ := 20
def dogs : ℕ := cats - additionalCats

-- Prove the total number of animals received
theorem total_animals_received : (cats + dogs) = 60 := by
  -- The proof itself is not required in this task
  sorry

end total_animals_received_l635_63596


namespace selling_price_of_article_l635_63549

theorem selling_price_of_article (CP : ℝ) (L_percent : ℝ) (SP : ℝ) 
  (h1 : CP = 600) 
  (h2 : L_percent = 50) 
  : SP = 300 := 
by
  sorry

end selling_price_of_article_l635_63549


namespace number_of_sets_B_l635_63564

def A : Set ℕ := {1, 2, 3}

theorem number_of_sets_B :
  ∃ B : Set ℕ, (A ∪ B = A ∧ 1 ∈ B ∧ (∃ n : ℕ, n = 4)) :=
by
  sorry

end number_of_sets_B_l635_63564


namespace sara_total_spent_l635_63540

def cost_of_tickets := 2 * 10.62
def cost_of_renting := 1.59
def cost_of_buying := 13.95
def total_spent := cost_of_tickets + cost_of_renting + cost_of_buying

theorem sara_total_spent : total_spent = 36.78 := by
  sorry

end sara_total_spent_l635_63540


namespace food_requirement_l635_63584

/-- Peter has six horses. Each horse eats 5 pounds of oats, three times a day, and 4 pounds of grain twice a day. -/
def totalFoodRequired (horses : ℕ) (days : ℕ) (oatsMeal : ℕ) (oatsMealsPerDay : ℕ) (grainMeal : ℕ) (grainMealsPerDay : ℕ) : ℕ :=
  let dailyOats := oatsMeal * oatsMealsPerDay
  let dailyGrain := grainMeal * grainMealsPerDay
  let dailyFood := dailyOats + dailyGrain
  let totalDailyFood := dailyFood * horses
  totalDailyFood * days

theorem food_requirement :
  totalFoodRequired 6 5 5 3 4 2 = 690 :=
by sorry

end food_requirement_l635_63584


namespace power_division_l635_63571

theorem power_division : 3^18 / (27^3) = 19683 := by
  have h1 : 27 = 3^3 := by sorry
  have h2 : (3^3)^3 = 3^(3*3) := by sorry
  have h3 : 27^3 = 3^9 := by
    rw [h1]
    exact h2
  rw [h3]
  have h4 : 3^18 / 3^9 = 3^(18 - 9) := by sorry
  rw [h4]
  norm_num

end power_division_l635_63571


namespace number_of_elements_in_set_l635_63522

theorem number_of_elements_in_set 
  (S : ℝ) (n : ℝ) 
  (h_avg : S / n = 6.8) 
  (a : ℝ) (h_a : a = 6) 
  (h_new_avg : (S + 2 * a) / n = 9.2) : 
  n = 5 := 
  sorry

end number_of_elements_in_set_l635_63522


namespace domain_of_function_l635_63579

theorem domain_of_function : 
  {x : ℝ | 0 < x ∧ 4 - x^2 > 0} = {x : ℝ | 0 < x ∧ x < 2} :=
sorry

end domain_of_function_l635_63579


namespace total_wheels_in_parking_lot_l635_63501

theorem total_wheels_in_parking_lot :
  let cars := 5
  let trucks := 3
  let bikes := 2
  let three_wheelers := 4
  let wheels_per_car := 4
  let wheels_per_truck := 6
  let wheels_per_bike := 2
  let wheels_per_three_wheeler := 3
  (cars * wheels_per_car + trucks * wheels_per_truck + bikes * wheels_per_bike + three_wheelers * wheels_per_three_wheeler) = 54 := by
  sorry

end total_wheels_in_parking_lot_l635_63501


namespace percentage_microphotonics_l635_63528

noncomputable def percentage_home_electronics : ℝ := 24
noncomputable def percentage_food_additives : ℝ := 20
noncomputable def percentage_GMO : ℝ := 29
noncomputable def percentage_industrial_lubricants : ℝ := 8
noncomputable def angle_basic_astrophysics : ℝ := 18

theorem percentage_microphotonics : 
  ∀ (home_elec food_additives GMO industrial_lub angle_bas_astro : ℝ),
  home_elec = 24 →
  food_additives = 20 →
  GMO = 29 →
  industrial_lub = 8 →
  angle_bas_astro = 18 →
  (100 - (home_elec + food_additives + GMO + industrial_lub + ((angle_bas_astro / 360) * 100))) = 14 :=
by
  intros _ _ _ _ _
  sorry

end percentage_microphotonics_l635_63528


namespace minimum_radius_third_sphere_l635_63544

-- Definitions for the problem
def height_cone := 4
def base_radius_cone := 3
def cos_alpha := 4 / 5
def radius_identical_sphere := 4 / 3
def cos_beta := 1 -- since beta is maximized

-- Define the required minimum radius for the third sphere based on the given conditions
theorem minimum_radius_third_sphere :
  ∃ x : ℝ, x = 27 / 35 ∧
    (height_cone = 4) ∧ 
    (base_radius_cone = 3) ∧ 
    (cos_alpha = 4 / 5) ∧ 
    (radius_identical_sphere = 4 / 3) ∧ 
    (cos_beta = 1) :=
sorry

end minimum_radius_third_sphere_l635_63544


namespace arithmetic_sequence_sum_l635_63500

theorem arithmetic_sequence_sum (S : ℕ → ℤ) (m : ℕ) 
  (h1 : S (m - 1) = -2) 
  (h2 : S m = 0) 
  (h3 : S (m + 1) = 3) : 
  m = 5 :=
by sorry

end arithmetic_sequence_sum_l635_63500


namespace double_inequality_pos_reals_equality_condition_l635_63573

theorem double_inequality_pos_reals (x y z : ℝ) (x_pos: 0 < x) (y_pos: 0 < y) (z_pos: 0 < z):
  0 < (1 / (x + y + z + 1)) - (1 / ((x + 1) * (y + 1) * (z + 1))) ∧
  (1 / (x + y + z + 1)) - (1 / ((x + 1) * (y + 1) * (z + 1))) ≤ (1 / 8) :=
  sorry

theorem equality_condition (x y z : ℝ) :
  ((1 / (x + y + z + 1)) - (1 / ((x + 1) * (y + 1) * (z + 1))) = (1 / 8)) ↔ x = 1 ∧ y = 1 ∧ z = 1 :=
  sorry

end double_inequality_pos_reals_equality_condition_l635_63573


namespace reduced_price_l635_63515

variable (original_price : ℝ) (final_amount : ℝ)

noncomputable def sales_tax (price : ℝ) : ℝ :=
  if price <= 2500 then price * 0.04
  else if price <= 4500 then 2500 * 0.04 + (price - 2500) * 0.07
  else 2500 * 0.04 + 2000 * 0.07 + (price - 4500) * 0.09

noncomputable def discount (price : ℝ) : ℝ :=
  if price <= 2000 then price * 0.02
  else if price <= 4000 then 2000 * 0.02 + (price - 2000) * 0.05
  else 2000 * 0.02 + 2000 * 0.05 + (price - 4000) * 0.10

theorem reduced_price (P : ℝ) (original_price := 5000) (final_amount := 2468) :
  P = original_price - discount original_price + sales_tax original_price → P = 2423 :=
by
  sorry

end reduced_price_l635_63515


namespace determine_a_l635_63593

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 3 * x^2 + 2
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 6 * x

theorem determine_a (a : ℝ) (h : f' a (-1) = 4) : a = 10 / 3 :=
by {
  sorry
}

end determine_a_l635_63593


namespace find_m_l635_63595

-- Define the lines l1 and l2
def line1 (x y : ℝ) (m : ℝ) : Prop := x + m^2 * y + 6 = 0
def line2 (x y : ℝ) (m : ℝ) : Prop := (m - 2) * x + 3 * m * y + 2 * m = 0

-- The statement that two lines are parallel
def lines_parallel (m : ℝ) : Prop :=
  ∀ (x y : ℝ), line1 x y m → line2 x y m

-- The mathematically equivalent proof problem
theorem find_m (m : ℝ) (H_parallel : lines_parallel m) : m = 0 ∨ m = -1 :=
sorry

end find_m_l635_63595


namespace count_solutions_l635_63531

theorem count_solutions : 
  (∃ (n : ℕ), ∀ (x : ℕ), (x + 17) % 43 = 71 % 43 ∧ x < 150 → n = 4) := 
sorry

end count_solutions_l635_63531


namespace gcd_two_powers_l635_63518

def m : ℕ := 2 ^ 1998 - 1
def n : ℕ := 2 ^ 1989 - 1

theorem gcd_two_powers :
  Nat.gcd (2 ^ 1998 - 1) (2 ^ 1989 - 1) = 511 := 
sorry

end gcd_two_powers_l635_63518


namespace at_least_one_not_less_than_two_l635_63575

theorem at_least_one_not_less_than_two (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + 1 / b ≥ 2) ∨ (b + 1 / c ≥ 2) ∨ (c + 1 / a ≥ 2) :=
sorry

end at_least_one_not_less_than_two_l635_63575


namespace part1_part2_l635_63594

noncomputable def f (a x : ℝ) : ℝ := a * x^2 + x - a

-- Part 1
theorem part1 (a x : ℝ) (h1 : |a| ≤ 1) (h2 : |x| ≤ 1) : |f a x| ≤ 5/4 :=
by
  sorry

-- Part 2
theorem part2 (a : ℝ) (h : ∃ x ∈ Set.Icc (-1:ℝ) (1:ℝ), f a x = 17/8) : a = -2 :=
by
  sorry

end part1_part2_l635_63594


namespace perimeter_of_figure_composed_of_squares_l635_63511

theorem perimeter_of_figure_composed_of_squares
  (n : ℕ)
  (side_length : ℝ)
  (square_perimeter : ℝ := 4 * side_length)
  (total_squares : ℕ := 7)
  (total_perimeter_if_independent : ℝ := square_perimeter * total_squares)
  (meet_at_vertices : ∀ i j : ℕ, i ≠ j → ∀ (s1 s2 : ℝ × ℝ), s1 ≠ s2 → ¬(s1 = s2))
  : total_perimeter_if_independent = 28 :=
by sorry

end perimeter_of_figure_composed_of_squares_l635_63511


namespace max_bees_in_largest_beehive_l635_63514

def total_bees : ℕ := 2000000
def beehives : ℕ := 7
def min_ratio : ℚ := 0.7

theorem max_bees_in_largest_beehive (B_max : ℚ) : 
  (6 * (min_ratio * B_max) + B_max = total_bees) → 
  B_max <= 2000000 / 5.2 ∧ B_max.floor = 384615 :=
by
  sorry

end max_bees_in_largest_beehive_l635_63514


namespace larger_to_smaller_ratio_l635_63541

theorem larger_to_smaller_ratio (x y : ℝ) (h1 : 0 < y) (h2 : y < x) (h3 : x + y = 7 * (x - y)) :
  x / y = 4 / 3 :=
by
  sorry

end larger_to_smaller_ratio_l635_63541


namespace obtain_100_using_fewer_than_ten_threes_example1_obtain_100_using_fewer_than_ten_threes_example2_l635_63503

-- The main theorem states that 100 can be obtained using fewer than ten 3's.

theorem obtain_100_using_fewer_than_ten_threes_example1 :
  100 = (333 / 3) - (33 / 3) :=
by
  sorry

theorem obtain_100_using_fewer_than_ten_threes_example2 :
  100 = (33 * 3) + (3 / 3) :=
by
  sorry

end obtain_100_using_fewer_than_ten_threes_example1_obtain_100_using_fewer_than_ten_threes_example2_l635_63503


namespace sin_4theta_l635_63597

theorem sin_4theta (θ : ℝ) (h : Complex.exp (Complex.I * θ) = (4 + Complex.I * Real.sqrt 7) / 5) :
  Real.sin (4 * θ) = (144 * Real.sqrt 7) / 625 := by
  sorry

end sin_4theta_l635_63597


namespace rate_of_current_l635_63569

variable (c : ℝ)

-- Define the given conditions
def speed_still_water : ℝ := 4.5
def time_ratio : ℝ := 2

-- Define the effective speeds
def speed_downstream : ℝ := speed_still_water + c
def speed_upstream : ℝ := speed_still_water - c

-- Define the condition that it takes twice as long to row upstream as downstream
def rowing_equation : Prop := 1 / speed_upstream = 2 * (1 / speed_downstream)

-- The Lean theorem stating the problem we need to prove
theorem rate_of_current (h : rowing_equation) : c = 1.5 := by
  sorry

end rate_of_current_l635_63569


namespace time_between_last_two_rings_l635_63525

variable (n : ℕ) (x y : ℝ)

noncomputable def timeBetweenLastTwoRings : ℝ :=
  x + (n - 3) * y

theorem time_between_last_two_rings :
  timeBetweenLastTwoRings n x y = x + (n - 3) * y :=
by
  sorry

end time_between_last_two_rings_l635_63525


namespace cos_double_angle_l635_63508

theorem cos_double_angle (θ : ℝ) (h : Real.cos θ = 1/2) : Real.cos (2 * θ) = -1/2 := by
  sorry

end cos_double_angle_l635_63508


namespace find_d_l635_63586

noncomputable def d_value (a b c : ℝ) := (2 * a + 2 * b + 2 * c - (3 / 4)^2) / 3

theorem find_d (a b c d : ℝ) (h : 2 * a^2 + 2 * b^2 + 2 * c^2 + 3 = 2 * d + (2 * a + 2 * b + 2 * c - 3 * d)^(1/2)) : 
  d = 23 / 48 :=
sorry

end find_d_l635_63586


namespace reach_14_from_458_l635_63560

def double (n : ℕ) : ℕ :=
  n * 2

def erase_last_digit (n : ℕ) : ℕ :=
  n / 10

def can_reach (start target : ℕ) (ops : List (ℕ → ℕ)) : Prop :=
  ∃ seq : List (ℕ → ℕ), seq = ops ∧
    seq.foldl (fun acc f => f acc) start = target

-- The proof problem statement
theorem reach_14_from_458 : can_reach 458 14 [double, erase_last_digit, double, double, erase_last_digit, double, double, erase_last_digit] :=
  sorry

end reach_14_from_458_l635_63560


namespace ratio_ac_l635_63554

variable {a b c d : ℝ}

-- Given the conditions
axiom ratio_ab : a / b = 5 / 4
axiom ratio_cd : c / d = 4 / 3
axiom ratio_db : d / b = 1 / 5

-- The statement to prove
theorem ratio_ac : a / c = 75 / 16 :=
  by sorry

end ratio_ac_l635_63554


namespace unoccupied_cylinder_volume_l635_63561

theorem unoccupied_cylinder_volume (r h : ℝ) (V_cylinder V_cone : ℝ) :
  r = 15 ∧ h = 30 ∧ V_cylinder = π * r^2 * h ∧ V_cone = (1/3) * π * r^2 * (r / 2) →
  V_cylinder - 2 * V_cone = 4500 * π :=
by
  intros h1
  sorry

end unoccupied_cylinder_volume_l635_63561


namespace divisible_l635_63512

def P (x : ℝ) : ℝ := 6 * x^3 + x^2 - 1
def Q (x : ℝ) : ℝ := 2 * x - 1

theorem divisible : ∃ R : ℝ → ℝ, ∀ x : ℝ, P x = Q x * R x :=
sorry

end divisible_l635_63512


namespace weekly_earnings_l635_63507

-- Definition of the conditions
def hourly_rate : ℕ := 20
def hours_per_day : ℕ := 8
def days_per_week : ℕ := 4

-- Theorem that conforms to the problem statement
theorem weekly_earnings : hourly_rate * hours_per_day * days_per_week = 640 := by
  sorry

end weekly_earnings_l635_63507


namespace gross_profit_percentage_l635_63570

theorem gross_profit_percentage (sales_price gross_profit cost : ℝ) 
  (h1 : sales_price = 81) 
  (h2 : gross_profit = 51) 
  (h3 : cost = sales_price - gross_profit) : 
  (gross_profit / cost) * 100 = 170 := 
by
  simp [h1, h2, h3]
  sorry

end gross_profit_percentage_l635_63570


namespace a_2016_value_l635_63547

def S (n : ℕ) : ℕ := n^2 - 1

def a (n : ℕ) : ℕ := S n - S (n - 1)

theorem a_2016_value : a 2016 = 4031 := by
  sorry

end a_2016_value_l635_63547


namespace translation_min_point_correct_l635_63513

-- Define the original equation
def original_eq (x : ℝ) := |x| - 5

-- Define the translation function
def translate_point (p : ℝ × ℝ) (tx ty : ℝ) : ℝ × ℝ := (p.1 + tx, p.2 + ty)

-- Define the minimum point of the original equation
def original_min_point : ℝ × ℝ := (0, original_eq 0)

-- Translate the original minimum point three units right and four units up
def new_min_point := translate_point original_min_point 3 4

-- Prove that the new minimum point is (3, -1)
theorem translation_min_point_correct : new_min_point = (3, -1) :=
by
  sorry

end translation_min_point_correct_l635_63513


namespace quad_vertex_transform_l635_63510

theorem quad_vertex_transform :
  ∀ (x y : ℝ) (h : y = -2 * x^2) (new_x new_y : ℝ) (h_translation : new_x = x + 3 ∧ new_y = y - 2),
  new_y = -2 * (new_x - 3)^2 + 2 :=
by
  intros x y h new_x new_y h_translation
  sorry

end quad_vertex_transform_l635_63510


namespace unique_three_digit_numbers_count_l635_63509

theorem unique_three_digit_numbers_count :
  ∃ l : List Nat, (∀ n ∈ l, 100 ≤ n ∧ n < 1000) ∧ 
    l = [230, 203, 302, 320] ∧ l.length = 4 := 
by
  sorry

end unique_three_digit_numbers_count_l635_63509


namespace surface_area_of_cross_shape_with_five_unit_cubes_l635_63555

noncomputable def unit_cube_surface_area : ℕ := 6
noncomputable def num_cubes : ℕ := 5
noncomputable def total_surface_area_iso_cubes : ℕ := num_cubes * unit_cube_surface_area
noncomputable def central_cube_exposed_faces : ℕ := 2
noncomputable def surrounding_cubes_exposed_faces : ℕ := 5
noncomputable def surrounding_cubes_count : ℕ := 4
noncomputable def cross_shape_surface_area : ℕ := 
  central_cube_exposed_faces + (surrounding_cubes_count * surrounding_cubes_exposed_faces)

theorem surface_area_of_cross_shape_with_five_unit_cubes : cross_shape_surface_area = 22 := 
by sorry

end surface_area_of_cross_shape_with_five_unit_cubes_l635_63555


namespace solve_equation_l635_63502

theorem solve_equation : 
  ∀ x : ℝ, (x - 3 ≠ 0) → (x + 6) / (x - 3) = 4 → x = 6 :=
by
  intros x h1 h2
  sorry

end solve_equation_l635_63502


namespace first_train_travels_more_l635_63523

-- Define the conditions
def velocity_first_train := 50 -- speed of the first train in km/hr
def velocity_second_train := 40 -- speed of the second train in km/hr
def distance_between_P_and_Q := 900 -- distance between P and Q in km

-- Problem statement
theorem first_train_travels_more :
  ∃ t : ℝ, (velocity_first_train * t + velocity_second_train * t = distance_between_P_and_Q)
          → (velocity_first_train * t - velocity_second_train * t = 100) :=
by sorry

end first_train_travels_more_l635_63523

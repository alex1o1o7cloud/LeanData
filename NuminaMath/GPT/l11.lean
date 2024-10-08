import Mathlib

namespace complex_division_l11_11917

theorem complex_division (i : ℂ) (hi : i^2 = -1) : (2 * i) / (1 + i) = 1 + i :=
by
  sorry

end complex_division_l11_11917


namespace standard_equation_of_parabola_l11_11238

theorem standard_equation_of_parabola (focus : ℝ × ℝ): 
  (focus.1 - 2 * focus.2 - 4 = 0) → 
  ((focus = (4, 0) → (∃ a : ℝ, ∀ x y : ℝ, y^2 = 4 * a * x)) ∨
   (focus = (0, -2) → (∃ b : ℝ, ∀ x y : ℝ, x^2 = 4 * b * y))) :=
by
  sorry

end standard_equation_of_parabola_l11_11238


namespace weight_of_5_moles_BaO_molar_concentration_BaO_l11_11750

-- Definitions based on conditions
def atomic_mass_Ba : ℝ := 137.33
def atomic_mass_O : ℝ := 16.00
def molar_mass_BaO : ℝ := atomic_mass_Ba + atomic_mass_O
def moles_BaO : ℝ := 5
def volume_solution : ℝ := 3

-- Theorem statements
theorem weight_of_5_moles_BaO : moles_BaO * molar_mass_BaO = 766.65 := by
  sorry

theorem molar_concentration_BaO : moles_BaO / volume_solution = 1.67 := by
  sorry

end weight_of_5_moles_BaO_molar_concentration_BaO_l11_11750


namespace algebraic_expression_evaluation_l11_11496

theorem algebraic_expression_evaluation (x : ℝ) (h : x^2 + x - 3 = 0) : x^3 + 2 * x^2 - 2 * x + 2 = 5 :=
by
  sorry

end algebraic_expression_evaluation_l11_11496


namespace range_of_a_l11_11014

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = Real.exp x) :
  (∀ x : ℝ, f x ≥ Real.exp x + a) ↔ a ≤ 0 :=
by
  sorry

end range_of_a_l11_11014


namespace third_median_length_l11_11095

variable (a b : ℝ) (A : ℝ)

def two_medians (m₁ m₂ : ℝ) : Prop :=
  m₁ = 4.5 ∧ m₂ = 7.5

def triangle_area (area : ℝ) : Prop :=
  area = 6 * Real.sqrt 20

theorem third_median_length (m₁ m₂ m₃ : ℝ) (area : ℝ) (h₁ : two_medians m₁ m₂)
  (h₂ : triangle_area area) : m₃ = 3 * Real.sqrt 5 := by
  sorry

end third_median_length_l11_11095


namespace relationship_among_a_b_c_l11_11594

noncomputable def a : ℝ := Real.sin (2 * Real.pi / 5)
noncomputable def b : ℝ := Real.cos (5 * Real.pi / 6)
noncomputable def c : ℝ := Real.tan (7 * Real.pi / 5)

theorem relationship_among_a_b_c : c > a ∧ a > b := by
  sorry

end relationship_among_a_b_c_l11_11594


namespace loss_percent_l11_11730

theorem loss_percent (CP SP Loss : ℝ) (h1 : CP = 600) (h2 : SP = 450) (h3 : Loss = CP - SP) : (Loss / CP) * 100 = 25 :=
by
  sorry

end loss_percent_l11_11730


namespace sequence_geometric_l11_11031

theorem sequence_geometric {a_n : ℕ → ℕ} (S : ℕ → ℕ) (a1 a2 a3 : ℕ) 
(hS : ∀ n, S n = 2 * a_n n - a_n 1) 
(h_arith : 2 * (a_n 2 + 1) = a_n 3 + a_n 1) : 
  ∀ n, a_n n = 2 ^ n :=
sorry

end sequence_geometric_l11_11031


namespace mean_of_other_four_l11_11809

theorem mean_of_other_four (a b c d e : ℕ) (h_mean : (a + b + c + d + e + 90) / 6 = 75)
  (h_max : max a (max b (max c (max d (max e 90)))) = 90)
  (h_twice : b = 2 * a) :
  (a + c + d + e) / 4 = 60 :=
by
  sorry

end mean_of_other_four_l11_11809


namespace ashu_complete_job_in_20_hours_l11_11187

/--
  Suresh can complete a job in 15 hours.
  Ashutosh alone can complete the same job in some hours.
  Suresh works for 9 hours and then the remaining job is completed by Ashutosh in 8 hours.
  We need to prove that the number of hours it takes for Ashutosh to complete the job alone is 20.
-/
theorem ashu_complete_job_in_20_hours :
  let A : ℝ := 20
  let suresh_work_rate : ℝ := 1 / 15
  let suresh_completed_work_in_9_hours : ℝ := (9 * suresh_work_rate)
  let remaining_work : ℝ := 1 - suresh_completed_work_in_9_hours
  (8 * (1 / A)) = remaining_work → A = 20 :=
by
  sorry

end ashu_complete_job_in_20_hours_l11_11187


namespace profit_function_l11_11302

def cost_per_unit : ℝ := 8

def daily_sales_quantity (x : ℝ) : ℝ := -x + 30

def profit_per_unit (x : ℝ) : ℝ := x - cost_per_unit

def total_profit (x : ℝ) : ℝ := (profit_per_unit x) * (daily_sales_quantity x)

theorem profit_function (x : ℝ) : total_profit x = -x^2 + 38*x - 240 :=
  sorry

end profit_function_l11_11302


namespace sufficient_not_necessary_example_l11_11448

lemma sufficient_but_not_necessary_condition (x y : ℝ) (hx : x >= 2) (hy : y >= 2) : x^2 + y^2 >= 4 :=
by
  -- We only need to state the lemma, so the proof is omitted.
  sorry

theorem sufficient_not_necessary_example :
  ¬(∀ x y : ℝ, (x^2 + y^2 >= 4) -> (x >= 2) ∧ (y >= 2)) :=
by 
  -- We only need to state the theorem, so the proof is omitted.
  sorry

end sufficient_not_necessary_example_l11_11448


namespace chips_needed_per_console_l11_11463

-- Definitions based on the conditions
def chips_per_day : ℕ := 467
def consoles_per_day : ℕ := 93

-- The goal is to prove that each video game console needs 5 computer chips
theorem chips_needed_per_console : chips_per_day / consoles_per_day = 5 :=
by sorry

end chips_needed_per_console_l11_11463


namespace shaded_area_l11_11958

theorem shaded_area (whole_squares partial_squares : ℕ) (area_whole area_partial : ℝ)
  (h1 : whole_squares = 5)
  (h2 : partial_squares = 6)
  (h3 : area_whole = 1)
  (h4 : area_partial = 0.5) :
  (whole_squares * area_whole + partial_squares * area_partial) = 8 :=
by
  sorry

end shaded_area_l11_11958


namespace find_y_l11_11613

theorem find_y (x y : ℝ) (h1 : x + y = 30) (h2 : x - y = 10) (h3 : ∃ C, x * y = C) (hx : x = 4) : y = 50 :=
sorry

end find_y_l11_11613


namespace pencils_across_diameter_l11_11836

def radius_feet : ℝ := 14
def pencil_length_inches : ℝ := 6

theorem pencils_across_diameter : 
  (2 * radius_feet * 12 / pencil_length_inches) = 56 := 
by
  sorry

end pencils_across_diameter_l11_11836


namespace solution_set_of_f_inequality_l11_11050

variable {f : ℝ → ℝ}
variable (h1 : f 1 = 1)
variable (h2 : ∀ x, f' x < 1/2)

theorem solution_set_of_f_inequality :
  {x : ℝ | f (x^2) < x^2 / 2 + 1 / 2} = {x : ℝ | x < -1 ∨ 1 < x} :=
sorry

end solution_set_of_f_inequality_l11_11050


namespace shaded_area_fraction_l11_11271

theorem shaded_area_fraction :
  let A := (0, 0)
  let B := (4, 0)
  let C := (4, 4)
  let D := (0, 4)
  let P := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  let Q := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
  let R := ((C.1 + D.1) / 2, (C.2 + D.2) / 2)
  let S := ((D.1 + A.1) / 2, (D.2 + A.2) / 2)
  let area_triangle := 1 / 2 * 2 * 2
  let shaded_area := 2 * area_triangle
  let total_area := 4 * 4
  shaded_area / total_area = 1 / 4 :=
by
  sorry

end shaded_area_fraction_l11_11271


namespace cot_trig_identity_l11_11188

noncomputable def cot (x : Real) : Real :=
  Real.cos x / Real.sin x

theorem cot_trig_identity (a b c α β γ : Real) 
  (habc : a^2 + b^2 = 2021 * c^2) 
  (hα : α = Real.arcsin (a / c)) 
  (hβ : β = Real.arcsin (b / c)) 
  (hγ : γ = Real.arccos ((2021 * c^2 - a^2 - b^2) / (2 * 2021 * c^2))) 
  (h_triangle : a^2 = b^2 + c^2 - 2 * b * c * Real.cos α) :
  cot α / (cot β + cot γ) = 1010 :=
by
  sorry

end cot_trig_identity_l11_11188


namespace ratio_of_volumes_l11_11021

def cone_radius_X := 10
def cone_height_X := 15
def cone_radius_Y := 15
def cone_height_Y := 10

noncomputable def volume_cone (r h : ℝ) := (1 / 3) * Real.pi * r^2 * h

noncomputable def volume_X := volume_cone cone_radius_X cone_height_X
noncomputable def volume_Y := volume_cone cone_radius_Y cone_height_Y

theorem ratio_of_volumes : volume_X / volume_Y = 2 / 3 := sorry

end ratio_of_volumes_l11_11021


namespace problem_solution_l11_11082

theorem problem_solution (x : ℝ) (h : (18 / 100) * 42 = (27 / 100) * x) : x = 28 :=
sorry

end problem_solution_l11_11082


namespace number_is_two_l11_11298

theorem number_is_two 
  (N : ℝ)
  (h1 : N = 4 * 1 / 2)
  (h2 : (1 / 2) * N = 1) :
  N = 2 :=
sorry

end number_is_two_l11_11298


namespace point_K_is_intersection_of_diagonals_l11_11071

variable {K A B C D : Type}

/-- A quadrilateral is circumscribed if there exists a circle within which all four vertices lie. -/
noncomputable def is_circumscribed (A B C D : Type) : Prop :=
sorry

/-- Distances from point K to the sides of the quadrilateral ABCD are proportional to the lengths of those sides. -/
noncomputable def proportional_distances (K A B C D : Type) : Prop :=
sorry

/-- A point is the intersection point of the diagonals AC and BD of quadrilateral ABCD. -/
noncomputable def intersection_point_of_diagonals (K A C B D : Type) : Prop :=
sorry

theorem point_K_is_intersection_of_diagonals 
  (K A B C D : Type) 
  (circumQ : is_circumscribed A B C D) 
  (propDist : proportional_distances K A B C D) 
  : intersection_point_of_diagonals K A C B D :=
sorry

end point_K_is_intersection_of_diagonals_l11_11071


namespace find_width_fabric_width_is_3_l11_11922

variable (Area Length : ℝ)
variable (Width : ℝ)

theorem find_width (h1 : Area = 24) (h2 : Length = 8) :
  Width = Area / Length :=
sorry

theorem fabric_width_is_3 (h1 : Area = 24) (h2 : Length = 8) :
  (Area / Length) = 3 :=
by
  have h : Area / Length = 3 := by sorry
  exact h

end find_width_fabric_width_is_3_l11_11922


namespace prism_pyramid_sum_l11_11279

theorem prism_pyramid_sum :
  let faces := 6
  let edges := 12
  let vertices := 8
  let new_faces := faces - 1 + 4
  let new_edges := edges + 4
  let new_vertices := vertices + 1
  new_faces + new_edges + new_vertices = 34 :=
by
  sorry

end prism_pyramid_sum_l11_11279


namespace determine_y_l11_11603

theorem determine_y (y : ℝ) (h1 : 0 < y) (h2 : y * (⌊y⌋ : ℝ) = 90) : y = 10 :=
sorry

end determine_y_l11_11603


namespace super12_teams_l11_11294

theorem super12_teams :
  ∃ n : ℕ, (n * (n - 1) = 132) ∧ n = 12 := by
  sorry

end super12_teams_l11_11294


namespace Alyssa_spending_correct_l11_11840

def cost_per_game : ℕ := 20

def last_year_in_person_games : ℕ := 13
def this_year_in_person_games : ℕ := 11
def this_year_streaming_subscription : ℕ := 120
def next_year_in_person_games : ℕ := 15
def next_year_streaming_subscription : ℕ := 150
def friends_count : ℕ := 2
def friends_join_games : ℕ := 5

def Alyssa_total_spending : ℕ :=
  (last_year_in_person_games * cost_per_game) +
  (this_year_in_person_games * cost_per_game) + this_year_streaming_subscription +
  (next_year_in_person_games * cost_per_game) + next_year_streaming_subscription -
  (friends_join_games * friends_count * cost_per_game)

theorem Alyssa_spending_correct : Alyssa_total_spending = 850 := by
  sorry

end Alyssa_spending_correct_l11_11840


namespace final_value_of_A_l11_11075

theorem final_value_of_A (A : ℤ) (h₁ : A = 15) (h₂ : A = -A + 5) : A = -10 := 
by 
  sorry

end final_value_of_A_l11_11075


namespace pebbles_ratio_l11_11903

variable (S : ℕ)

theorem pebbles_ratio :
  let initial_pebbles := 18
  let skipped_pebbles := 9
  let additional_pebbles := 30
  let final_pebbles := 39
  initial_pebbles - skipped_pebbles + additional_pebbles = final_pebbles →
  (skipped_pebbles : ℚ) / initial_pebbles = 1 / 2 :=
by
  intros
  sorry

end pebbles_ratio_l11_11903


namespace combined_work_time_l11_11678

theorem combined_work_time (man_rate : ℚ := 1/5) (wife_rate : ℚ := 1/7) (son_rate : ℚ := 1/15) :
  (man_rate + wife_rate + son_rate)⁻¹ = 105 / 43 :=
by
  sorry

end combined_work_time_l11_11678


namespace smallest_number_h_divisible_8_11_24_l11_11760

theorem smallest_number_h_divisible_8_11_24 : 
  ∃ h : ℕ, (h + 5) % 8 = 0 ∧ (h + 5) % 11 = 0 ∧ (h + 5) % 24 = 0 ∧ h = 259 :=
by
  sorry

end smallest_number_h_divisible_8_11_24_l11_11760


namespace expenditure_recording_l11_11658

theorem expenditure_recording (income expense : ℤ) (h1 : income = 100) (h2 : expense = -100)
  (h3 : income = -expense) : expense = -100 :=
by
  sorry

end expenditure_recording_l11_11658


namespace bill_annual_healthcare_cost_l11_11383

def hourly_wage := 25
def weekly_hours := 30
def weeks_per_month := 4
def months_per_year := 12
def normal_monthly_price := 500
def annual_income := hourly_wage * weekly_hours * weeks_per_month * months_per_year
def subsidy (income : ℕ) : ℕ :=
  if income < 10000 then 90
  else if income ≤ 40000 then 50
  else if income > 50000 then 20
  else 0
def monthly_cost_after_subsidy := (normal_monthly_price * (100 - subsidy annual_income)) / 100
def annual_cost := monthly_cost_after_subsidy * months_per_year

theorem bill_annual_healthcare_cost : annual_cost = 3000 := by
  sorry

end bill_annual_healthcare_cost_l11_11383


namespace number_of_games_l11_11647

-- Definitions based on the conditions
def initial_money : ℕ := 104
def cost_of_blades : ℕ := 41
def cost_per_game : ℕ := 9

-- Lean 4 statement asserting the number of games Will can buy is 7
theorem number_of_games : (initial_money - cost_of_blades) / cost_per_game = 7 := by
  sorry

end number_of_games_l11_11647


namespace geometric_series_q_and_S6_l11_11025

theorem geometric_series_q_and_S6 (a : ℕ → ℝ) (q : ℝ) (S_6 : ℝ) 
  (ha_pos : ∀ n, a n > 0)
  (ha2 : a 2 = 3)
  (ha4 : a 4 = 27) :
  q = 3 ∧ S_6 = 364 :=
by
  sorry

end geometric_series_q_and_S6_l11_11025


namespace towel_area_decrease_l11_11415

theorem towel_area_decrease (L B : ℝ) :
  let A_original := L * B
  let L_new := 0.8 * L
  let B_new := 0.9 * B
  let A_new := L_new * B_new
  let percentage_decrease := ((A_original - A_new) / A_original) * 100
  percentage_decrease = 28 := 
by
  sorry

end towel_area_decrease_l11_11415


namespace arithmetic_expression_eval_l11_11971

theorem arithmetic_expression_eval : 
  5 * 7.5 + 2 * 12 + 8.5 * 4 + 7 * 6 = 137.5 :=
by
  sorry

end arithmetic_expression_eval_l11_11971


namespace largest_frog_weight_l11_11847

theorem largest_frog_weight (S L : ℕ) (h1 : L = 10 * S) (h2 : L = S + 108): L = 120 := by
  sorry

end largest_frog_weight_l11_11847


namespace resulting_shape_is_cone_l11_11166

-- Assume we have a right triangle
structure right_triangle (α β γ : ℝ) : Prop :=
  (is_right : γ = π / 2)
  (sum_of_angles : α + β + γ = π)
  (acute_angles : α < π / 2 ∧ β < π / 2)

-- Assume we are rotating around one of the legs
def rotate_around_leg (α β : ℝ) : Prop := sorry

theorem resulting_shape_is_cone (α β γ : ℝ) (h : right_triangle α β γ) :
  ∃ (shape : Type), rotate_around_leg α β → shape = cone :=
by
  sorry

end resulting_shape_is_cone_l11_11166


namespace no_integers_exist_l11_11767

theorem no_integers_exist :
  ¬ (∃ x y : ℤ, (x + 2019) * (x + 2020) + (x + 2020) * (x + 2021) + (x + 2019) * (x + 2021) = y^2) :=
by
  sorry

end no_integers_exist_l11_11767


namespace melissa_trip_total_time_l11_11998

theorem melissa_trip_total_time :
  ∀ (freeway_dist rural_dist : ℕ) (freeway_speed_factor : ℕ) 
  (rural_time : ℕ),
  freeway_dist = 80 →
  rural_dist = 20 →
  freeway_speed_factor = 4 →
  rural_time = 40 →
  (rural_dist * freeway_speed_factor / rural_time + freeway_dist / (rural_dist * freeway_speed_factor / rural_time)) = 80 :=
by
  intros freeway_dist rural_dist freeway_speed_factor rural_time hd1 hd2 hd3 hd4
  sorry

end melissa_trip_total_time_l11_11998


namespace calc_expression_l11_11060

theorem calc_expression : 112 * 5^4 * 3^2 = 630000 := by
  sorry

end calc_expression_l11_11060


namespace cats_to_dogs_l11_11651

theorem cats_to_dogs (c d : ℕ) (h1 : c = 24) (h2 : 4 * d = 5 * c) : d = 30 :=
by
  sorry

end cats_to_dogs_l11_11651


namespace weight_of_brand_b_l11_11085

theorem weight_of_brand_b (w_a w_b : ℕ) (vol_a vol_b : ℕ) (total_volume total_weight : ℕ) 
  (h1 : w_a = 950) 
  (h2 : vol_a = 3) 
  (h3 : vol_b = 2) 
  (h4 : total_volume = 4) 
  (h5 : total_weight = 3640) 
  (h6 : vol_a + vol_b = total_volume) 
  (h7 : vol_a * w_a + vol_b * w_b = total_weight) : 
  w_b = 395 := 
by {
  sorry
}

end weight_of_brand_b_l11_11085


namespace prime_divisor_of_ones_l11_11134

theorem prime_divisor_of_ones (p : ℕ) (hp : Nat.Prime p ∧ p ≠ 2 ∧ p ≠ 5) :
  ∃ k : ℕ, p ∣ (10^k - 1) / 9 :=
by
  sorry

end prime_divisor_of_ones_l11_11134


namespace simplify_fraction_l11_11918

theorem simplify_fraction :
  (1 / ((1 / (Real.sqrt 2 + 1)) + (2 / (Real.sqrt 3 - 1)) + (3 / (Real.sqrt 5 + 2)))) =
  (1 / (Real.sqrt 2 + 2 * Real.sqrt 3 + 3 * Real.sqrt 5 - 5)) :=
by
  sorry

end simplify_fraction_l11_11918


namespace sum_of_longest_altitudes_l11_11133

-- Define the sides of the triangle
def a : ℕ := 6
def b : ℕ := 8
def c : ℕ := 10

-- Define the sides are the longest altitudes in the right triangle
def longest_altitude1 : ℕ := a
def longest_altitude2 : ℕ := b

-- Define the main theorem to prove
theorem sum_of_longest_altitudes : longest_altitude1 + longest_altitude2 = 14 := 
by
  -- The proof goes here
  sorry

end sum_of_longest_altitudes_l11_11133


namespace quadratic_real_roots_range_l11_11904

theorem quadratic_real_roots_range (m : ℝ) :
  (∀ x : ℝ, (m - 1) * x^2 + 2 * x + 1 = 0 → 
    (∃ x1 x2 : ℝ, x = x1 ∧ x = x2 ∧ x1 = x2 → true)) → 
    m ≤ 2 ∧ m ≠ 1 :=
by
  sorry

end quadratic_real_roots_range_l11_11904


namespace probability_of_drawing_orange_marble_second_l11_11213

noncomputable def probability_second_marble_is_orange (total_A white_A black_A : ℕ) (total_B orange_B green_B blue_B : ℕ) (total_C orange_C green_C blue_C : ℕ) : ℚ := 
  let p_white := (white_A : ℚ) / total_A
  let p_black := (black_A : ℚ) / total_A
  let p_orange_B := (orange_B : ℚ) / total_B
  let p_orange_C := (orange_C : ℚ) / total_C
  (p_white * p_orange_B) + (p_black * p_orange_C)

theorem probability_of_drawing_orange_marble_second :
  probability_second_marble_is_orange 9 4 5 15 7 5 3 10 4 4 2 = 58 / 135 :=
by
  sorry

end probability_of_drawing_orange_marble_second_l11_11213


namespace problem_a_problem_b_problem_c_problem_d_l11_11626

-- a) Proof problem for \(x^2 + 5x + 6 < 0\)
theorem problem_a (x : ℝ) : x^2 + 5*x + 6 < 0 → -3 < x ∧ x < -2 := by
  sorry

-- b) Proof problem for \(-x^2 + 9x - 20 < 0\)
theorem problem_b (x : ℝ) : -x^2 + 9*x - 20 < 0 → x < 4 ∨ x > 5 := by
  sorry

-- c) Proof problem for \(x^2 + x - 56 < 0\)
theorem problem_c (x : ℝ) : x^2 + x - 56 < 0 → -8 < x ∧ x < 7 := by
  sorry

-- d) Proof problem for \(9x^2 + 4 < 12x\) (No solutions)
theorem problem_d (x : ℝ) : ¬ 9*x^2 + 4 < 12*x := by
  sorry

end problem_a_problem_b_problem_c_problem_d_l11_11626


namespace calculation_correctness_l11_11643

theorem calculation_correctness : 15 - 14 * 3 + 11 / 2 - 9 * 4 + 18 = -39.5 := by
  sorry

end calculation_correctness_l11_11643


namespace smallest_x_l11_11154

theorem smallest_x (x y : ℝ) (h1 : 4 < x) (h2 : x < 8) (h3 : 8 < y) (h4 : y < 12) (h5 : y - x = 7) :
  ∃ ε > 0, x = 4 + ε :=
by
  sorry

end smallest_x_l11_11154


namespace battery_difference_l11_11353

def flashlights_batteries := 2
def toys_batteries := 15
def difference := 13

theorem battery_difference : toys_batteries - flashlights_batteries = difference :=
by
  sorry

end battery_difference_l11_11353


namespace major_axis_length_proof_l11_11127

-- Define the conditions
def radius : ℝ := 3
def minor_axis_length : ℝ := 2 * radius
def major_axis_length : ℝ := minor_axis_length + 0.75 * minor_axis_length

-- State the proof problem
theorem major_axis_length_proof : major_axis_length = 10.5 := 
by
  -- Proof goes here
  sorry

end major_axis_length_proof_l11_11127


namespace ratio_of_trees_l11_11628

theorem ratio_of_trees (plums pears apricots : ℕ) (h_plums : plums = 3) (h_pears : pears = 3) (h_apricots : apricots = 3) :
  plums = pears ∧ pears = apricots :=
by
  sorry

end ratio_of_trees_l11_11628


namespace ava_planted_more_trees_l11_11017

theorem ava_planted_more_trees (L : ℕ) (h1 : 9 + L = 15) : 9 - L = 3 := 
by
  sorry

end ava_planted_more_trees_l11_11017


namespace find_a_l11_11744

-- Define the constants b and the asymptote equation
def asymptote_eq (x y : ℝ) := 3 * x + 2 * y = 0

-- Define the hyperbola equation and the condition
def hyperbola_eq (x y a : ℝ) := x^2 / a^2 - y^2 / 9 = 1
def hyperbola_condition (a : ℝ) := a > 0

-- Theorem stating the value of a given the conditions
theorem find_a (a : ℝ) (hcond : hyperbola_condition a) 
  (h_asymp : ∀ x y : ℝ, asymptote_eq x y → y = -(3/2) * x) :
  a = 2 := 
sorry

end find_a_l11_11744


namespace sum_of_two_even_numbers_is_even_l11_11107

  theorem sum_of_two_even_numbers_is_even (a b : ℤ) (ha : ∃ k : ℤ, a = 2 * k) (hb : ∃ m : ℤ, b = 2 * m) : ∃ n : ℤ, a + b = 2 * n := by
    sorry
  
end sum_of_two_even_numbers_is_even_l11_11107


namespace prize_distribution_l11_11527

/--
In a best-of-five competition where two players of equal level meet in the final, 
with a score of 2:1 after the first three games and the total prize money being 12,000 yuan, 
the prize awarded to the player who has won 2 games should be 9,000 yuan.
-/
theorem prize_distribution (prize_money : ℝ) 
  (A_wins : ℕ) (B_wins : ℕ) (prob_A : ℝ) (prob_B : ℝ) (total_games : ℕ) : 
  total_games = 5 → 
  prize_money = 12000 → 
  A_wins = 2 → 
  B_wins = 1 → 
  prob_A = 1/2 → 
  prob_B = 1/2 → 
  ∃ prize_for_A : ℝ, prize_for_A = 9000 :=
by
  intros
  sorry

end prize_distribution_l11_11527


namespace shifted_line_does_not_pass_through_third_quadrant_l11_11811

-- The condition: The original line is y = -2x - 1
def original_line (x : ℝ) : ℝ := -2 * x - 1

-- The condition: The line is shifted 3 units to the right
def shifted_line (x : ℝ) : ℝ := -2 * (x - 3) - 1

theorem shifted_line_does_not_pass_through_third_quadrant :
  ¬(∃ (x y : ℝ), y = shifted_line x ∧ x < 0 ∧ y < 0) :=
sorry

end shifted_line_does_not_pass_through_third_quadrant_l11_11811


namespace andy_correct_answer_l11_11726

-- Let y be the number Andy is using
def y : ℕ := 13  -- Derived from the conditions

-- Given condition based on Andy's incorrect operation
def condition : Prop := 4 * y + 5 = 57

-- Statement of the proof problem
theorem andy_correct_answer : condition → ((y + 5) * 4 = 72) := by
  intros h
  sorry

end andy_correct_answer_l11_11726


namespace determine_a_value_l11_11746

-- Define the initial equation and conditions
def fractional_equation (x a : ℝ) : Prop :=
  (x - a) / (x - 1) - 3 / x = 1

-- Define the existence of a positive root
def has_positive_root (a : ℝ) : Prop :=
  ∃ x : ℝ, x > 0 ∧ fractional_equation x a

-- The main theorem stating the correct value of 'a' for the given condition
theorem determine_a_value (x : ℝ) : has_positive_root 1 :=
sorry

end determine_a_value_l11_11746


namespace greatest_x_integer_l11_11498

theorem greatest_x_integer (x : ℤ) : 
  (∃ k : ℤ, (x^2 + 4 * x + 9) = k * (x - 4)) ↔ x ≤ 5 :=
by
  sorry

end greatest_x_integer_l11_11498


namespace phase_shift_of_sine_l11_11808

theorem phase_shift_of_sine :
  let a := 3
  let b := 4
  let c := - (Real.pi / 4)
  let phase_shift := -(c / b)
  phase_shift = Real.pi / 16 :=
by
  sorry

end phase_shift_of_sine_l11_11808


namespace bathroom_square_footage_l11_11362

theorem bathroom_square_footage
  (tiles_width : ℕ)
  (tiles_length : ℕ)
  (tile_size_inches : ℕ)
  (inches_per_foot : ℕ)
  (h1 : tiles_width = 10)
  (h2 : tiles_length = 20)
  (h3 : tile_size_inches = 6)
  (h4 : inches_per_foot = 12)
: (tiles_length * tile_size_inches / inches_per_foot) * (tiles_width * tile_size_inches / inches_per_foot) = 50 := 
by
  sorry

end bathroom_square_footage_l11_11362


namespace focus_on_negative_y_axis_l11_11987

-- Definition of the condition: equation of the parabola
def parabola (x y : ℝ) := x^2 + y = 0

-- Statement of the problem
theorem focus_on_negative_y_axis (x y : ℝ) (h : parabola x y) : 
  -- The focus of the parabola lies on the negative half of the y-axis
  ∃ y, y < 0 :=
sorry

end focus_on_negative_y_axis_l11_11987


namespace necessary_but_not_sufficient_l11_11968

-- Define the geometric mean condition between 2 and 8
def is_geometric_mean (m : ℝ) := m = 4 ∨ m = -4

-- Prove that m = 4 is a necessary but not sufficient condition for is_geometric_mean
theorem necessary_but_not_sufficient (m : ℝ) :
  (is_geometric_mean m) ↔ (m = 4) :=
sorry

end necessary_but_not_sufficient_l11_11968


namespace terminal_side_in_quadrant_l11_11950

theorem terminal_side_in_quadrant (k : ℤ) (α : ℝ)
  (h: π + 2 * k * π < α ∧ α < (3 / 2) * π + 2 * k * π) :
  (π / 2) + k * π < α / 2 ∧ α / 2 < (3 / 4) * π + k * π :=
sorry

end terminal_side_in_quadrant_l11_11950


namespace smallest_B_l11_11162

-- Definitions and conditions
def known_digit_sum : Nat := 4 + 8 + 3 + 9 + 4 + 2
def divisible_by_3 (n : Nat) : Bool := n % 3 = 0

-- Statement to prove
theorem smallest_B (B : Nat) (h : B < 10) (hdiv : divisible_by_3 (B + known_digit_sum)) : B = 0 :=
sorry

end smallest_B_l11_11162


namespace negate_prop_l11_11695

theorem negate_prop (p : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 * Real.pi → |Real.sin x| ≤ 1) :
  ¬ (∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 * Real.pi → |Real.sin x| ≤ 1) ↔ ∃ x_0 : ℝ, 0 ≤ x_0 ∧ x_0 ≤ 2 * Real.pi ∧ |Real.sin x_0| > 1 :=
by sorry

end negate_prop_l11_11695


namespace find_h_for_expression_l11_11073

theorem find_h_for_expression (a k : ℝ) (h : ℝ) :
  (∃ a k : ℝ, ∀ x : ℝ, x^2 - 6*x + 1 = a*(x - h)^3 + k) ↔ h = 2 :=
by
  sorry

end find_h_for_expression_l11_11073


namespace mirror_side_length_l11_11001

theorem mirror_side_length
  (width_wall : ℝ)
  (length_wall : ℝ)
  (area_wall : ℝ)
  (area_mirror : ℝ)
  (side_length_mirror : ℝ)
  (h1 : width_wall = 32)
  (h2 : length_wall = 20.25)
  (h3 : area_wall = width_wall * length_wall)
  (h4 : area_mirror = area_wall / 2)
  (h5 : side_length_mirror * side_length_mirror = area_mirror)
  : side_length_mirror = 18 := by
  sorry

end mirror_side_length_l11_11001


namespace part1_monotonicity_part2_minimum_range_l11_11028

noncomputable def f (k x : ℝ) : ℝ := (k + x) / (x - 1) * Real.log x

theorem part1_monotonicity (x : ℝ) (h : x ≠ 1) :
    k = 0 → f k x = (x / (x - 1)) * Real.log x ∧ 
    (0 < x ∧ x < 1 ∨ 1 < x) → Monotone (f k) :=
sorry

theorem part2_minimum_range (k : ℝ) :
    (∃ x ∈ Set.Ioi 1, IsLocalMin (f k) x) ↔ k ∈ Set.Ioi 1 :=
sorry

end part1_monotonicity_part2_minimum_range_l11_11028


namespace find_m_value_l11_11002

noncomputable def hyperbola_m_value (m : ℝ) : Prop :=
  let a := 1
  let b := 2 * a
  m = -(1/4)

theorem find_m_value :
  (∀ x y : ℝ, x^2 + m * y^2 = 1 → b = 2 * a) → hyperbola_m_value m :=
by
  intro h
  sorry

end find_m_value_l11_11002


namespace paper_clips_in_two_cases_l11_11945

theorem paper_clips_in_two_cases (c b : ℕ) : 
  2 * c * b * 200 = 2 * (c * b * 200) :=
by
  sorry

end paper_clips_in_two_cases_l11_11945


namespace oil_in_Tank_C_is_982_l11_11511

-- Definitions of tank capacities and oil amounts
def capacity_A := 80
def capacity_B := 120
def capacity_C := 160
def capacity_D := 240

def total_oil_bought := 1387

def oil_in_A := 70
def oil_in_B := 95
def oil_in_D := capacity_D  -- Since Tank D is 100% full

-- Statement of the problem
theorem oil_in_Tank_C_is_982 :
  oil_in_A + oil_in_B + oil_in_D + (total_oil_bought - (oil_in_A + oil_in_B + oil_in_D)) = total_oil_bought :=
by
  sorry

end oil_in_Tank_C_is_982_l11_11511


namespace inheritance_amount_l11_11667

theorem inheritance_amount (x : ℝ) (total_taxes_paid : ℝ) (federal_tax_rate : ℝ) (state_tax_rate : ℝ) 
  (federal_tax_paid : ℝ) (state_tax_base : ℝ) (state_tax_paid : ℝ) 
  (federal_tax_eq : federal_tax_paid = federal_tax_rate * x)
  (state_tax_base_eq : state_tax_base = x - federal_tax_paid)
  (state_tax_eq : state_tax_paid = state_tax_rate * state_tax_base)
  (total_taxes_eq : total_taxes_paid = federal_tax_paid + state_tax_paid) 
  (total_taxes_val : total_taxes_paid = 18000)
  (federal_tax_rate_val : federal_tax_rate = 0.25)
  (state_tax_rate_val : state_tax_rate = 0.15)
  : x = 50000 :=
sorry

end inheritance_amount_l11_11667


namespace total_fencing_cost_l11_11044

def side1 : ℕ := 34
def side2 : ℕ := 28
def side3 : ℕ := 45
def side4 : ℕ := 50
def side5 : ℕ := 55

def cost1_per_meter : ℕ := 2
def cost2_per_meter : ℕ := 2
def cost3_per_meter : ℕ := 3
def cost4_per_meter : ℕ := 3
def cost5_per_meter : ℕ := 4

def total_cost : ℕ :=
  side1 * cost1_per_meter +
  side2 * cost2_per_meter +
  side3 * cost3_per_meter +
  side4 * cost4_per_meter +
  side5 * cost5_per_meter

theorem total_fencing_cost : total_cost = 629 := by
  sorry

end total_fencing_cost_l11_11044


namespace coordinates_of_A_in_second_quadrant_l11_11314

noncomputable def coordinates_A (m : ℤ) : ℤ × ℤ :=
  (7 - 2 * m, 5 - m)

theorem coordinates_of_A_in_second_quadrant (m : ℤ) (h1 : 7 - 2 * m < 0) (h2 : 5 - m > 0) :
  coordinates_A m = (-1, 1) := 
sorry

end coordinates_of_A_in_second_quadrant_l11_11314


namespace incenter_coordinates_l11_11163

-- Define lengths of the sides of the triangle
def a : ℕ := 8
def b : ℕ := 10
def c : ℕ := 6

-- Define the incenter formula components
def sum_of_sides : ℕ := a + b + c
def x : ℚ := a / (sum_of_sides : ℚ)
def y : ℚ := b / (sum_of_sides : ℚ)
def z : ℚ := c / (sum_of_sides : ℚ)

-- Prove the result
theorem incenter_coordinates :
  (x, y, z) = (1 / 3, 5 / 12, 1 / 4) :=
by 
  -- Proof skipped
  sorry

end incenter_coordinates_l11_11163


namespace exist_x_y_satisfy_condition_l11_11587

theorem exist_x_y_satisfy_condition (f g : ℝ → ℝ) (h1 : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x ≥ 0) (h2 : ∀ y, 0 ≤ y ∧ y ≤ 1 → g y ≥ 0) :
  ∃ (x : ℝ), ∃ (y : ℝ), (0 ≤ x ∧ x ≤ 1) ∧ (0 ≤ y ∧ y ≤ 1) ∧ |f x + g y - x * y| ≥ 1 / 4 :=
by
  sorry

end exist_x_y_satisfy_condition_l11_11587


namespace pups_more_than_adults_l11_11512

-- Define the counts of dogs
def H := 5  -- number of huskies
def P := 2  -- number of pitbulls
def G := 4  -- number of golden retrievers

-- Define the number of pups each type of dog had
def pups_per_husky_and_pitbull := 3
def additional_pups_per_golden_retriever := 2
def pups_per_golden_retriever := pups_per_husky_and_pitbull + additional_pups_per_golden_retriever

-- Calculate the total number of pups
def total_pups := H * pups_per_husky_and_pitbull + P * pups_per_husky_and_pitbull + G * pups_per_golden_retriever

-- Calculate the total number of adult dogs
def total_adult_dogs := H + P + G

-- Prove that the number of pups is 30 more than the number of adult dogs
theorem pups_more_than_adults : total_pups - total_adult_dogs = 30 :=
by
  -- fill in the proof later
  sorry

end pups_more_than_adults_l11_11512


namespace Adam_spent_21_dollars_l11_11430

-- Define the conditions as given in the problem
def initial_money : ℕ := 91
def spent_money (x : ℕ) : Prop := (initial_money - x) * 3 = 10 * x

-- The theorem we want to prove: Adam spent 21 dollars on new books
theorem Adam_spent_21_dollars : spent_money 21 :=
by sorry

end Adam_spent_21_dollars_l11_11430


namespace area_relationship_l11_11179

theorem area_relationship (x β : ℝ) (hβ : 0.60 * x^2 = β) : α = (4 / 3) * β :=
by
  -- conditions and goal are stated
  let α := 0.80 * x^2
  sorry

end area_relationship_l11_11179


namespace line_through_A_with_zero_sum_of_intercepts_l11_11801

-- Definitions
def passesThroughPoint (A : ℝ × ℝ) (l : ℝ → ℝ → Prop) : Prop :=
  l A.1 A.2

def sumInterceptsZero (l : ℝ → ℝ → Prop) : Prop :=
  ∃ a b : ℝ, l a 0 ∧ l 0 b ∧ a + b = 0

-- Theorem statement
theorem line_through_A_with_zero_sum_of_intercepts (l : ℝ → ℝ → Prop) :
  passesThroughPoint (1, 4) l ∧ sumInterceptsZero l →
  (∀ x y, l x y ↔ 4 * x - y = 0) ∨ (∀ x y, l x y ↔ x - y + 3 = 0) :=
sorry

end line_through_A_with_zero_sum_of_intercepts_l11_11801


namespace three_x_plus_four_l11_11742

theorem three_x_plus_four (x : ℕ) (h : x = 5) : 3 * x + 4 = 19 :=
by
  sorry

end three_x_plus_four_l11_11742


namespace largest_divisor_of_consecutive_odd_product_l11_11121

theorem largest_divisor_of_consecutive_odd_product (n : ℕ) (h_even : n % 2 = 0) (h_pos : n > 0) :
  315 ∣ (n + 1) * (n + 3) * (n + 5) * (n + 7) * (n + 9) * (n + 11) * (n + 13) := 
sorry

end largest_divisor_of_consecutive_odd_product_l11_11121


namespace total_gallons_in_tanks_l11_11872

def tank1_capacity : ℕ := 7000
def tank2_capacity : ℕ := 5000
def tank3_capacity : ℕ := 3000

def tank1_filled : ℚ := 3/4
def tank2_filled : ℚ := 4/5
def tank3_filled : ℚ := 1/2

theorem total_gallons_in_tanks :
  (tank1_capacity * tank1_filled + tank2_capacity * tank2_filled + tank3_capacity * tank3_filled : ℚ) = 10750 := 
by 
  sorry

end total_gallons_in_tanks_l11_11872


namespace geometric_sequence_sixth_term_l11_11260

theorem geometric_sequence_sixth_term (a r : ℕ) (h₁ : a = 8) (h₂ : a * r ^ 3 = 64) : a * r ^ 5 = 256 :=
by
  -- to be filled (proof skipped)
  sorry

end geometric_sequence_sixth_term_l11_11260


namespace johns_original_earnings_l11_11595

-- Define the conditions
def raises (original : ℝ) (percentage : ℝ) := original + original * percentage

-- The theorem stating the equivalent problem proof
theorem johns_original_earnings :
  ∃ (x : ℝ), raises x 0.375 = 55 ↔ x = 40 :=
sorry

end johns_original_earnings_l11_11595


namespace amount_subtracted_is_15_l11_11623

theorem amount_subtracted_is_15 (n x : ℕ) (h1 : 7 * n - x = 2 * n + 10) (h2 : n = 5) : x = 15 :=
by 
  sorry

end amount_subtracted_is_15_l11_11623


namespace card_area_after_one_inch_shortening_l11_11579

def initial_length := 5
def initial_width := 7
def new_area_shortened_side_two := 21
def shorter_side_reduction := 2
def longer_side_reduction := 1

theorem card_area_after_one_inch_shortening :
  (initial_length - shorter_side_reduction) * initial_width = new_area_shortened_side_two →
  initial_length * (initial_width - longer_side_reduction) = 30 :=
by
  intro h
  sorry

end card_area_after_one_inch_shortening_l11_11579


namespace frac_e_a_l11_11242

variable (a b c d e : ℚ)

theorem frac_e_a (h1 : a / b = 5) (h2 : b / c = 1 / 4) (h3 : c / d = 7) (h4 : d / e = 1 / 2) :
  e / a = 8 / 35 :=
sorry

end frac_e_a_l11_11242


namespace balance_pots_l11_11165

theorem balance_pots 
  (w1 : ℕ) (w2 : ℕ) (m : ℕ)
  (h_w1 : w1 = 645)
  (h_w2 : w2 = 237)
  (h_m : m = 1000) :
  ∃ (m1 m2 : ℕ), 
  (w1 + m1 = w2 + m2) ∧ 
  (m1 + m2 = m) ∧ 
  (m1 = 296) ∧ 
  (m2 = 704) := by
  sorry

end balance_pots_l11_11165


namespace hyperbola_asymptotes_l11_11105

theorem hyperbola_asymptotes (x y : ℝ) (E : x^2 / 4 - y^2 = 1) :
  y = (1 / 2) * x ∨ y = -(1 / 2) * x :=
sorry

end hyperbola_asymptotes_l11_11105


namespace tangent_line_eq_max_min_values_l11_11718

noncomputable def f (x : ℝ) : ℝ := (1 / (3:ℝ)) * x^3 - 4 * x + 4

theorem tangent_line_eq (x y : ℝ) : 
    y = f 1 → 
    y = -3 * (x - 1) + f 1 → 
    3 * x + y - 10 / 3 = 0 := 
sorry

theorem max_min_values (x : ℝ) (h : 0 ≤ x ∧ x ≤ 3) : 
    (∀ x, (0 ≤ x ∧ x ≤ 3) → f x ≤ 4) ∧ 
    (∀ x, (0 ≤ x ∧ x ≤ 3) → f x ≥ -4 / 3) := 
sorry

end tangent_line_eq_max_min_values_l11_11718


namespace prove_solutions_l11_11059

noncomputable def solution1 (x : ℝ) : Prop :=
  3 * x^2 + 6 = abs (-25 + x)

theorem prove_solutions :
  solution1 ( (-1 + Real.sqrt 229) / 6 ) ∧ solution1 ( (-1 - Real.sqrt 229) / 6 ) :=
by
  sorry

end prove_solutions_l11_11059


namespace chimes_in_a_day_l11_11785

-- Definitions for the conditions
def strikes_in_12_hours : ℕ :=
  (1 + 12) * 12 / 2

def strikes_in_24_hours : ℕ :=
  2 * strikes_in_12_hours

def half_hour_strikes : ℕ :=
  24 * 2

def total_chimes_in_a_day : ℕ :=
  strikes_in_24_hours + half_hour_strikes

-- Statement to prove
theorem chimes_in_a_day : total_chimes_in_a_day = 204 :=
by 
  -- The proof would be placed here
  sorry

end chimes_in_a_day_l11_11785


namespace prasanna_speed_l11_11208

variable (L_speed P_speed time apart : ℝ)
variable (h1 : L_speed = 40)
variable (h2 : time = 1)
variable (h3 : apart = 78)

theorem prasanna_speed :
  P_speed = apart - (L_speed * time) / time := 
by
  rw [h1, h2, h3]
  simp
  sorry

end prasanna_speed_l11_11208


namespace pyramid_base_edge_length_l11_11519

theorem pyramid_base_edge_length 
(radius_hemisphere height_pyramid : ℝ)
(h_radius : radius_hemisphere = 4)
(h_height : height_pyramid = 10)
(h_tangent : ∀ face : ℝ, True) : 
∃ s : ℝ, s = 2 * Real.sqrt 42 :=
by
  sorry

end pyramid_base_edge_length_l11_11519


namespace day_crew_fraction_l11_11680

-- Definitions of number of boxes per worker for day crew, and workers for day crew
variables (D : ℕ) (W : ℕ)

-- Definitions of night crew loading rate and worker ratio based on given conditions
def night_boxes_per_worker := (3 / 4 : ℚ) * D
def night_workers := (2 / 3 : ℚ) * W

-- Definition of total boxes loaded by each crew
def day_crew_total := D * W
def night_crew_total := night_boxes_per_worker D * night_workers W

-- The proof problem shows fraction loaded by day crew equals 2/3
theorem day_crew_fraction : (day_crew_total D W) / (day_crew_total D W + night_crew_total D W) = (2 / 3 : ℚ) := by
  sorry

end day_crew_fraction_l11_11680


namespace find_k_l11_11652

theorem find_k (k : ℝ) (h : (3:ℝ)^4 + k * (3:ℝ)^2 - 26 = 0) : k = -55 / 9 := 
by sorry

end find_k_l11_11652


namespace ratio_d_e_l11_11745

theorem ratio_d_e (a b c d e f : ℝ)
  (h1 : a / b = 1 / 3)
  (h2 : b / c = 2)
  (h3 : c / d = 1 / 2)
  (h4 : e / f = 1 / 6)
  (h5 : a * b * c / (d * e * f) = 0.25) :
  d / e = 1 / 4 :=
sorry

end ratio_d_e_l11_11745


namespace set_membership_l11_11429

theorem set_membership :
  {m : ℤ | ∃ k : ℤ, 10 = k * (m + 1)} = {-11, -6, -3, -2, 0, 1, 4, 9} :=
by sorry

end set_membership_l11_11429


namespace binom_eq_fraction_l11_11416

open Nat

theorem binom_eq_fraction (n : ℕ) (h_pos : 0 < n) : choose n 2 = n * (n - 1) / 2 :=
by
  sorry

end binom_eq_fraction_l11_11416


namespace mary_pays_fifteen_l11_11418

def apple_cost : ℕ := 1
def orange_cost : ℕ := 2
def banana_cost : ℕ := 3
def discount_per_5_fruits : ℕ := 1

def apples_bought : ℕ := 5
def oranges_bought : ℕ := 3
def bananas_bought : ℕ := 2

def total_cost_before_discount : ℕ :=
  apples_bought * apple_cost +
  oranges_bought * orange_cost +
  bananas_bought * banana_cost

def total_fruits : ℕ :=
  apples_bought + oranges_bought + bananas_bought

def total_discount : ℕ :=
  (total_fruits / 5) * discount_per_5_fruits

def final_amount_to_pay : ℕ :=
  total_cost_before_discount - total_discount

theorem mary_pays_fifteen : final_amount_to_pay = 15 := by
  sorry

end mary_pays_fifteen_l11_11418


namespace diophantine_solution_range_l11_11256

theorem diophantine_solution_range {a b c n : ℤ} (coprime_ab : Int.gcd a b = 1) :
  (∃ (x y : ℕ), a * x + b * y = c ∧ ∀ k : ℤ, k ≥ 1 → ∃ (x y : ℕ), a * (x + k * b) + b * (y - k * a) = c) → 
  ((n - 1) * a * b + a + b ≤ c ∧ c ≤ (n + 1) * a * b) :=
sorry

end diophantine_solution_range_l11_11256


namespace star_value_l11_11203

-- Define the operation &
def and_operation (a b : ℕ) : ℕ := (a + b) * (a - b)

-- Define the operation star
def star_operation (c d : ℕ) : ℕ := and_operation c d + 2 * (c + d)

-- The proof problem
theorem star_value : star_operation 8 4 = 72 :=
by
  sorry

end star_value_l11_11203


namespace smaller_of_two_numbers_l11_11642

theorem smaller_of_two_numbers (a b : ℕ) (h1 : a * b = 4761) (h2 : 10 ≤ a ∧ a < 100) (h3 : 10 ≤ b ∧ b < 100) : min a b = 53 :=
by {
  sorry -- proof skips as directed
}

end smaller_of_two_numbers_l11_11642


namespace molecular_weight_CaO_is_56_l11_11454

def atomic_weight_Ca : ℕ := 40
def atomic_weight_O : ℕ := 16
def molecular_weight_CaO : ℕ := atomic_weight_Ca + atomic_weight_O

theorem molecular_weight_CaO_is_56 :
  molecular_weight_CaO = 56 := by
  sorry

end molecular_weight_CaO_is_56_l11_11454


namespace sum_and_ratio_implies_difference_l11_11524

theorem sum_and_ratio_implies_difference (a b : ℚ) (h1 : a + b = 500) (h2 : a / b = 0.8) : b - a = 55.55555555555556 := by
  sorry

end sum_and_ratio_implies_difference_l11_11524


namespace fuel_tank_ethanol_l11_11761

theorem fuel_tank_ethanol (x : ℝ) (H : 0.12 * x + 0.16 * (208 - x) = 30) : x = 82 := 
by
  sorry

end fuel_tank_ethanol_l11_11761


namespace root_exists_in_interval_l11_11455

noncomputable def f (x : ℝ) := (1 / 2) ^ x - x + 1

theorem root_exists_in_interval :
  (0 < f 1) ∧ (f 1.5 < 0) ∧ (f 2 < 0) ∧ (f 3 < 0) → ∃ x, 1 < x ∧ x < 1.5 ∧ f x = 0 :=
by
  -- use the intermediate value theorem and bisection method here
  sorry

end root_exists_in_interval_l11_11455


namespace broker_wealth_increase_after_two_years_l11_11936

theorem broker_wealth_increase_after_two_years :
  let initial_investment : ℝ := 100
  let first_year_increase : ℝ := 0.75
  let second_year_decrease : ℝ := 0.30
  let end_first_year := initial_investment * (1 + first_year_increase)
  let end_second_year := end_first_year * (1 - second_year_decrease)
  end_second_year - initial_investment = 22.50 :=
by
  sorry

end broker_wealth_increase_after_two_years_l11_11936


namespace tangent_parabola_points_l11_11444

theorem tangent_parabola_points (a b : ℝ) (h_circle : a^2 + b^2 = 1) (h_discriminant : a^2 - 4 * b * (b - 1) = 0) :
    (a = 0 ∧ b = 1) ∨ 
    (a = 2 * Real.sqrt 6 / 5 ∧ b = -1 / 5) ∨ 
    (a = -2 * Real.sqrt 6 / 5 ∧ b = -1 / 5) := sorry

end tangent_parabola_points_l11_11444


namespace pascal_row_20_fifth_sixth_sum_l11_11369

-- Conditions from the problem
def pascal_element (n k : ℕ) : ℕ := Nat.choose n k

-- Question translated to a Lean theorem
theorem pascal_row_20_fifth_sixth_sum :
  pascal_element 20 4 + pascal_element 20 5 = 20349 :=
by
  sorry

end pascal_row_20_fifth_sixth_sum_l11_11369


namespace prime_ge_7_not_divisible_by_40_l11_11032

theorem prime_ge_7_not_divisible_by_40 (p : ℕ) (hp_prime : Nat.Prime p) (hp_ge_7 : p ≥ 7) : ¬ (40 ∣ (p^3 - 1)) :=
sorry

end prime_ge_7_not_divisible_by_40_l11_11032


namespace binom_np_n_mod_p2_l11_11795

   theorem binom_np_n_mod_p2 (p n : ℕ) (hp : Nat.Prime p) : (Nat.choose (n * p) n) % (p ^ 2) = n % (p ^ 2) :=
   by
     sorry
   
end binom_np_n_mod_p2_l11_11795


namespace solve_for_xy_l11_11344

theorem solve_for_xy (x y : ℝ) 
  (h1 : 0.05 * x + 0.07 * (30 + x) = 14.9)
  (h2 : 0.03 * y - 5.6 = 0.07 * x) : 
  x = 106.67 ∧ y = 435.567 := 
  by 
  sorry

end solve_for_xy_l11_11344


namespace negation_statement_l11_11443

theorem negation_statement (h : ∀ x : ℝ, |x - 2| + |x - 4| > 3) : 
  ∃ x0 : ℝ, |x0 - 2| + |x0 - 4| ≤ 3 :=
sorry

end negation_statement_l11_11443


namespace children_difference_l11_11359

-- Axiom definitions based on conditions
def initial_children : ℕ := 36
def first_stop_got_off : ℕ := 45
def first_stop_got_on : ℕ := 25
def second_stop_got_off : ℕ := 68
def final_children : ℕ := 12

-- Mathematical formulation of the problem and its proof statement
theorem children_difference :
  ∃ (x : ℕ), 
    initial_children - first_stop_got_off + first_stop_got_on - second_stop_got_off + x = final_children ∧ 
    (first_stop_got_off + second_stop_got_off) - (first_stop_got_on + x) = 24 :=
by 
  sorry

end children_difference_l11_11359


namespace circles_intersect_l11_11020

theorem circles_intersect (m c : ℝ) (h1 : (1:ℝ) = (5 + (-m))) (h2 : (3:ℝ) = (5 + (c - (-2)))) :
  m + c = 3 :=
sorry

end circles_intersect_l11_11020


namespace margaret_age_in_12_years_l11_11694

theorem margaret_age_in_12_years
  (brian_age : ℝ)
  (christian_age : ℝ)
  (margaret_age : ℝ)
  (h1 : christian_age = 3.5 * brian_age)
  (h2 : brian_age + 12 = 45)
  (h3 : margaret_age = christian_age - 10) :
  margaret_age + 12 = 117.5 :=
by
  sorry

end margaret_age_in_12_years_l11_11694


namespace sufficient_but_not_necessary_condition_l11_11513

theorem sufficient_but_not_necessary_condition (x : ℝ) (p : -1 < x ∧ x < 3) (q : x^2 - 5 * x - 6 < 0) : 
  (-1 < x ∧ x < 3) → (x^2 - 5 * x - 6 < 0) ∧ ¬((x^2 - 5 * x - 6 < 0) → (-1 < x ∧ x < 3)) :=
by
  sorry

end sufficient_but_not_necessary_condition_l11_11513


namespace John_ASMC_score_l11_11788

def ASMC_score (c w : ℕ) : ℕ := 25 + 5 * c - 2 * w

theorem John_ASMC_score (c w : ℕ) (h1 : ASMC_score c w = 100) (h2 : c + w ≤ 25) :
  c = 19 ∧ w = 10 :=
by {
  sorry
}

end John_ASMC_score_l11_11788


namespace tiffany_first_level_treasures_l11_11143

-- Conditions
def treasure_points : ℕ := 6
def treasures_second_level : ℕ := 5
def total_points : ℕ := 48

-- Definition for the number of treasures on the first level
def points_from_second_level : ℕ := treasures_second_level * treasure_points
def points_from_first_level : ℕ := total_points - points_from_second_level
def treasures_first_level : ℕ := points_from_first_level / treasure_points

-- The theorem to prove
theorem tiffany_first_level_treasures : treasures_first_level = 3 :=
by
  sorry

end tiffany_first_level_treasures_l11_11143


namespace find_a2018_l11_11919

-- Definitions based on given conditions
def seq (a : ℕ → ℝ) : Prop :=
  a 1 = 0.5 ∧ ∀ n, a (n + 1) = 1 - 1 / (a n)

-- The statement to prove
theorem find_a2018 (a : ℕ → ℝ) (h : seq a) : a 2018 = -1 := by
  sorry

end find_a2018_l11_11919


namespace wilson_buys_3_bottles_of_cola_l11_11234

theorem wilson_buys_3_bottles_of_cola
    (num_hamburgers : ℕ := 2) 
    (cost_per_hamburger : ℕ := 5) 
    (cost_per_cola : ℕ := 2) 
    (discount : ℕ := 4) 
    (total_paid : ℕ := 12) :
    num_hamburgers * cost_per_hamburger - discount + x * cost_per_cola = total_paid → x = 3 :=
by
  sorry

end wilson_buys_3_bottles_of_cola_l11_11234


namespace avg_waiting_time_is_1_point_2_minutes_l11_11387

/--
Assume that a Distracted Scientist immediately pulls out and recasts the fishing rod upon a bite,
doing so instantly. After this, he waits again. Consider a 6-minute interval.
During this time, the first rod receives 3 bites on average, and the second rod receives 2 bites
on average. Therefore, on average, there are 5 bites on both rods together in these 6 minutes.

We need to prove that the average waiting time for the first bite is 1.2 minutes.
-/
theorem avg_waiting_time_is_1_point_2_minutes :
  let first_rod_bites := 3
  let second_rod_bites := 2
  let total_time := 6 -- in minutes
  let total_bites := first_rod_bites + second_rod_bites
  let avg_rate := total_bites / total_time
  let avg_waiting_time := 1 / avg_rate
  avg_waiting_time = 1.2 := by
  sorry

end avg_waiting_time_is_1_point_2_minutes_l11_11387


namespace sum_of_cube_edges_l11_11299

/-- A cube has 12 edges. Each edge of a cube is of equal length. Given the length of one
edge as 15 cm, the sum of the lengths of all the edges of the cube is 180 cm. -/
theorem sum_of_cube_edges (edge_length : ℝ) (num_edges : ℕ) (h1 : edge_length = 15) (h2 : num_edges = 12) :
  num_edges * edge_length = 180 :=
by
  sorry

end sum_of_cube_edges_l11_11299


namespace cosine_120_eq_neg_one_half_l11_11963

theorem cosine_120_eq_neg_one_half : Real.cos (120 * Real.pi / 180) = -1/2 :=
by
-- Proof omitted
sorry

end cosine_120_eq_neg_one_half_l11_11963


namespace angle_complement_30_l11_11772

def complement_angle (x : ℝ) : ℝ := 90 - x

theorem angle_complement_30 (x : ℝ) (h : x = complement_angle x - 30) : x = 30 :=
by
  sorry

end angle_complement_30_l11_11772


namespace average_age_when_youngest_born_l11_11911

theorem average_age_when_youngest_born (n : ℕ) (current_average_age youngest age_difference total_ages : ℝ)
  (hc1 : n = 7)
  (hc2 : current_average_age = 30)
  (hc3 : youngest = 6)
  (hc4 : age_difference = youngest * 6)
  (hc5 : total_ages = n * current_average_age - age_difference) :
  total_ages / n = 24.857
:= sorry

end average_age_when_youngest_born_l11_11911


namespace preimage_of_3_1_is_2_1_l11_11192

-- Definition of the mapping f
def f (x y : ℝ) : ℝ × ℝ := (x + y, x - y)

-- The Lean theorem statement
theorem preimage_of_3_1_is_2_1 : ∃ (x y : ℝ), f x y = (3, 1) ∧ (x = 2 ∧ y = 1) :=
by
  sorry

end preimage_of_3_1_is_2_1_l11_11192


namespace find_focus_with_larger_x_coordinate_l11_11253

noncomputable def focus_of_hyperbola_with_larger_x_coordinate : ℝ × ℝ :=
  let h := 5
  let k := 20
  let a := 7
  let b := 9
  let c := Real.sqrt (a^2 + b^2)
  (h + c, k)

theorem find_focus_with_larger_x_coordinate :
  focus_of_hyperbola_with_larger_x_coordinate = (5 + Real.sqrt 130, 20) := by
  sorry

end find_focus_with_larger_x_coordinate_l11_11253


namespace find_flag_count_l11_11910

-- Definitions of conditions
inductive Color
| purple
| gold
| silver

-- Function to count valid flags
def countValidFlags : Nat :=
  let first_stripe_choices := 3
  let second_stripe_choices := 2
  let third_stripe_choices := 2
  first_stripe_choices * second_stripe_choices * third_stripe_choices

-- Statement to prove
theorem find_flag_count : countValidFlags = 12 := by
  sorry

end find_flag_count_l11_11910


namespace number_of_cages_l11_11321

-- Definitions based on the conditions
def parrots_per_cage := 2
def parakeets_per_cage := 6
def total_birds := 72

-- Goal: Prove the number of cages
theorem number_of_cages : 
  (parrots_per_cage + parakeets_per_cage) * x = total_birds → x = 9 :=
by
  sorry

end number_of_cages_l11_11321


namespace solution_set_of_inequality_l11_11655

theorem solution_set_of_inequality :
  ∀ x : ℝ, (x-50)*(60-x) > 0 ↔ 50 < x ∧ x < 60 :=
by
  sorry

end solution_set_of_inequality_l11_11655


namespace scenario1_scenario2_scenario3_l11_11348

noncomputable def scenario1_possible_situations : Nat :=
  12

noncomputable def scenario2_possible_situations : Nat :=
  144

noncomputable def scenario3_possible_situations : Nat :=
  50

theorem scenario1 (shots : Nat) (hits : Nat) (consecutive_hits : Nat) (remaining_hits : Nat) (not_consecutive : Prop) :
  shots = 10 ∧ hits = 7 ∧ consecutive_hits = 5 ∧ remaining_hits = 2 ∧ not_consecutive → 
  scenario1_possible_situations = 12 := by
  sorry

theorem scenario2 (shots : Nat) (hits : Nat) (consecutive_hits : Nat) (remaining_hits : Nat) :
  shots = 10 ∧ hits = 7 ∧ consecutive_hits = 4 ∧ remaining_hits = 3 → 
  scenario2_possible_situations = 144 := by
  sorry

theorem scenario3 (shots : Nat) (hits : Nat) (consecutive_hits : Nat) (remaining_hits : Nat) :
  shots = 10 ∧ hits = 6 ∧ consecutive_hits = 4 ∧ remaining_hits = 2 → 
  scenario3_possible_situations = 50 := by
  sorry

end scenario1_scenario2_scenario3_l11_11348


namespace min_detectors_correct_l11_11447

noncomputable def min_detectors (M N : ℕ) : ℕ :=
  ⌈(M : ℝ) / 2⌉₊ + ⌈(N : ℝ) / 2⌉₊

theorem min_detectors_correct (M N : ℕ) (hM : 2 ≤ M) (hN : 2 ≤ N) :
  min_detectors M N = ⌈(M : ℝ) / 2⌉₊ + ⌈(N : ℝ) / 2⌉₊ :=
by {
  -- The proof goes here
  sorry
}

end min_detectors_correct_l11_11447


namespace g_675_eq_42_l11_11583

noncomputable def g : ℕ → ℕ := sorry

axiom gxy : ∀ (x y : ℕ), g (x * y) = g x + g y
axiom g15 : g 15 = 18
axiom g45 : g 45 = 24

theorem g_675_eq_42 : g 675 = 42 :=
sorry

end g_675_eq_42_l11_11583


namespace coin_flip_probability_l11_11403

theorem coin_flip_probability :
  let total_flips := 8
  let num_heads := 6
  let total_outcomes := (2: ℝ) ^ total_flips
  let favorable_outcomes := (Nat.choose total_flips num_heads)
  let probability := favorable_outcomes / total_outcomes
  probability = (7 / 64 : ℝ) :=
by
  sorry

end coin_flip_probability_l11_11403


namespace pow_two_sub_one_not_square_l11_11735

theorem pow_two_sub_one_not_square (n : ℕ) (h : n > 1) : ¬ ∃ k : ℕ, 2^n - 1 = k^2 := by
  sorry

end pow_two_sub_one_not_square_l11_11735


namespace amount_paid_out_l11_11618

theorem amount_paid_out 
  (amount : ℕ) 
  (h1 : amount % 50 = 0) 
  (h2 : ∃ (n : ℕ), n ≥ 15 ∧ amount = n * 5000 ∨ amount = n * 1000)
  (h3 : ∃ (n : ℕ), n ≥ 35 ∧ amount = n * 1000) : 
  amount = 29950 :=
by 
  sorry

end amount_paid_out_l11_11618


namespace total_length_correct_l11_11366

-- Definitions for the first area's path length and scale.
def first_area_scale : ℕ := 500
def first_area_path_length_inches : ℕ := 6
def first_area_path_length_feet : ℕ := first_area_scale * first_area_path_length_inches

-- Definitions for the second area's path length and scale.
def second_area_scale : ℕ := 1000
def second_area_path_length_inches : ℕ := 3
def second_area_path_length_feet : ℕ := second_area_scale * second_area_path_length_inches

-- Total length represented by both paths in feet.
def total_path_length_feet : ℕ :=
  first_area_path_length_feet + second_area_path_length_feet

-- The Lean theorem proving that the total length is 6000 feet.
theorem total_length_correct : total_path_length_feet = 6000 := by
  sorry

end total_length_correct_l11_11366


namespace total_cherry_tomatoes_l11_11764

-- Definitions based on the conditions
def cherryTomatoesPerJar : Nat := 8
def numberOfJars : Nat := 7

-- The statement we want to prove
theorem total_cherry_tomatoes : cherryTomatoesPerJar * numberOfJars = 56 := by
  sorry

end total_cherry_tomatoes_l11_11764


namespace no_conditions_satisfy_l11_11609

-- Define the conditions
def condition1 (a b c : ℤ) : Prop := a = 1 ∧ b = 1 ∧ c = 1
def condition2 (a b c : ℤ) : Prop := a = b - 1 ∧ b = c - 1
def condition3 (a b c : ℤ) : Prop := a = b ∧ b = c
def condition4 (a b c : ℤ) : Prop := a > c ∧ c = b - 1 

-- Define the equations
def equation1 (a b c : ℤ) : ℤ := a * (a - b)^3 + b * (b - c)^3 + c * (c - a)^3
def equation2 (a b c : ℤ) : Prop := a + b + c = 3

-- Proof statement for the original problem
theorem no_conditions_satisfy (a b c : ℤ) :
  ¬ (condition1 a b c ∧ equation1 a b c = 3 ∧ equation2 a b c) ∧
  ¬ (condition2 a b c ∧ equation1 a b c = 3 ∧ equation2 a b c) ∧
  ¬ (condition3 a b c ∧ equation1 a b c = 3 ∧ equation2 a b c) ∧
  ¬ (condition4 a b c ∧ equation1 a b c = 3 ∧ equation2 a b c) :=
sorry

end no_conditions_satisfy_l11_11609


namespace problem1_problem2_problem3_l11_11725

-- Definitions of sets A, B, and C as per given conditions
def set_A (a : ℝ) : Set ℝ :=
  {x | x^2 - a * x + a^2 - 19 = 0}

def set_B : Set ℝ :=
  {x | x^2 - 5 * x + 6 = 0}

def set_C : Set ℝ :=
  {x | x^2 + 2 * x - 8 = 0}

-- Questions reformulated as proof problems
theorem problem1 (a : ℝ) (h : set_A a = set_B) : a = 5 :=
sorry

theorem problem2 (a : ℝ) (h1 : ∃ x, x ∈ set_A a ∧ x ∈ set_B) (h2 : ∀ x, x ∈ set_A a → x ∉ set_C) : a = -2 :=
sorry

theorem problem3 (a : ℝ) (h1 : ∃ x, x ∈ set_A a ∧ x ∈ set_B) (h2 : set_A a ∩ set_B = set_A a ∩ set_C) : a = -3 :=
sorry

end problem1_problem2_problem3_l11_11725


namespace least_integer_square_condition_l11_11099

theorem least_integer_square_condition (x : ℤ) (h : x^2 = 3 * x + 36) : x = -6 :=
by sorry

end least_integer_square_condition_l11_11099


namespace factorize_expression_l11_11329

theorem factorize_expression (a : ℝ) : 
  a^3 - 16 * a = a * (a + 4) * (a - 4) :=
sorry

end factorize_expression_l11_11329


namespace find_k_l11_11375

theorem find_k (k : ℕ) : 5 ^ k = 5 * 25^2 * 125^3 → k = 14 := by
  sorry

end find_k_l11_11375


namespace hcl_formed_l11_11518

-- Define the balanced chemical equation as a relationship between reactants and products
def balanced_equation (m_C2H6 m_Cl2 m_CCl4 m_HCl : ℝ) :=
  m_C2H6 + 4 * m_Cl2 = m_CCl4 + 6 * m_HCl

-- Define the problem-specific values
def reaction_given (m_C2H6 m_Cl2 m_CCl4 m_HCl : ℝ) :=
  m_C2H6 = 3 ∧ m_Cl2 = 21 ∧ m_CCl4 = 6 ∧ balanced_equation m_C2H6 m_Cl2 m_CCl4 m_HCl

-- Prove the number of moles of HCl formed
theorem hcl_formed : ∃ (m_HCl : ℝ), reaction_given 3 21 6 m_HCl ∧ m_HCl = 18 :=
by
  sorry

end hcl_formed_l11_11518


namespace bricks_required_l11_11636

-- Courtyard dimensions in meters
def length_courtyard_m := 23
def width_courtyard_m := 15

-- Brick dimensions in centimeters
def length_brick_cm := 17
def width_brick_cm := 9

-- Conversion from meters to centimeters
def meter_to_cm (m : Int) : Int :=
  m * 100

-- Area of courtyard in square centimeters
def area_courtyard_cm2 : Int :=
  meter_to_cm length_courtyard_m * meter_to_cm width_courtyard_m

-- Area of a single brick in square centimeters
def area_brick_cm2 : Int :=
  length_brick_cm * width_brick_cm

-- Calculate the number of bricks needed, ensuring we round up to the nearest whole number
def total_bricks_needed : Int :=
  (area_courtyard_cm2 + area_brick_cm2 - 1) / area_brick_cm2

-- The theorem stating the total number of bricks needed
theorem bricks_required :
  total_bricks_needed = 22550 := by
  sorry

end bricks_required_l11_11636


namespace marble_probability_l11_11572

theorem marble_probability
  (total_marbles : ℕ)
  (blue_marbles : ℕ)
  (green_marbles : ℕ)
  (draws : ℕ)
  (prob_first_green : ℚ)
  (prob_second_blue_given_green : ℚ)
  (total_prob : ℚ)
  (h_total : total_marbles = 10)
  (h_blue : blue_marbles = 4)
  (h_green : green_marbles = 6)
  (h_draws : draws = 2)
  (h_prob_first_green : prob_first_green = 3 / 5)
  (h_prob_second_blue_given_green : prob_second_blue_given_green = 4 / 9)
  (h_total_prob : total_prob = 4 / 15) :
  prob_first_green * prob_second_blue_given_green = total_prob := sorry

end marble_probability_l11_11572


namespace books_more_than_movies_l11_11100

theorem books_more_than_movies (books_count movies_count read_books watched_movies : ℕ) 
  (h_books : books_count = 10)
  (h_movies : movies_count = 6)
  (h_read_books : read_books = 10) 
  (h_watched_movies : watched_movies = 6) : 
  read_books - watched_movies = 4 := by
  sorry

end books_more_than_movies_l11_11100


namespace wind_velocity_determination_l11_11391

theorem wind_velocity_determination (ρ : ℝ) (P1 P2 : ℝ) (A1 A2 : ℝ) (V1 V2 : ℝ) (k : ℝ) :
  ρ = 1.2 →
  P1 = 0.75 →
  A1 = 2 →
  V1 = 12 →
  P1 = ρ * k * A1 * V1^2 →
  P2 = 20.4 →
  A2 = 10.76 →
  P2 = ρ * k * A2 * V2^2 →
  V2 = 27 := 
by sorry

end wind_velocity_determination_l11_11391


namespace class_has_24_students_l11_11422

theorem class_has_24_students (n S : ℕ) 
  (h1 : (S - 91 + 19) / n = 87)
  (h2 : S / n = 90) : 
  n = 24 :=
by sorry

end class_has_24_students_l11_11422


namespace solve_for_t_l11_11831

theorem solve_for_t (s t : ℤ) (h1 : 11 * s + 7 * t = 160) (h2 : s = 2 * t + 4) : t = 4 :=
by
  sorry

end solve_for_t_l11_11831


namespace treaty_signed_on_saturday_l11_11334

-- Define the start day and the total days until the treaty.
def start_day_of_week : Nat := 4 -- Thursday is the 4th day (0 = Sunday, ..., 6 = Saturday)
def days_until_treaty : Nat := 919

-- Calculate the final day of the week after 919 days since start_day_of_week.
def treaty_day_of_week : Nat := (start_day_of_week + days_until_treaty) % 7

-- The goal is to prove that the treaty was signed on a Saturday.
theorem treaty_signed_on_saturday : treaty_day_of_week = 6 :=
by
  -- Implement the proof steps
  sorry

end treaty_signed_on_saturday_l11_11334


namespace find_divisor_l11_11672

theorem find_divisor (d : ℕ) : (55 / d) + 10 = 21 → d = 5 :=
by 
  sorry

end find_divisor_l11_11672


namespace find_b_l11_11221

def f (x : ℝ) : ℝ := 5 * x - 7

theorem find_b : ∃ (b : ℝ), f b = 3 :=
by
  use 2
  show f 2 = 3
  sorry

end find_b_l11_11221


namespace man_speed_approx_l11_11453

noncomputable def speed_of_man : ℝ :=
  let L := 700    -- Length of the train in meters
  let u := 63 / 3.6  -- Speed of the train in meters per second (converted)
  let t := 41.9966402687785 -- Time taken to cross the man in seconds
  let v := (u * t - L) / t  -- Speed of the man
  v

-- The main theorem to prove that the speed of the man is approximately 0.834 m/s.
theorem man_speed_approx : abs (speed_of_man - 0.834) < 1e-3 :=
by
  -- Simplification and exact calculations will be handled by the Lean prover or could be manually done.
  sorry

end man_speed_approx_l11_11453


namespace cost_of_each_adult_meal_is_8_l11_11243

/- Define the basic parameters and conditions -/
def total_people : ℕ := 11
def kids : ℕ := 2
def total_cost : ℕ := 72
def kids_eat_free (k : ℕ) := k = 0

/- The number of adults is derived from the total people minus kids -/
def num_adults : ℕ := total_people - kids

/- The cost per adult meal can be defined and we need to prove it equals to $8 -/
def cost_per_adult (total_cost : ℕ) (num_adults : ℕ) : ℕ := total_cost / num_adults

/- The statement to prove that the cost per adult meal is $8 -/
theorem cost_of_each_adult_meal_is_8 : cost_per_adult total_cost num_adults = 8 := by
  sorry

end cost_of_each_adult_meal_is_8_l11_11243


namespace division_of_repeating_decimals_l11_11738

noncomputable def repeating_to_fraction (r : ℚ) : ℚ := 
  if r == 0.36 then 4 / 11 
  else if r == 0.12 then 4 / 33 
  else 0

theorem division_of_repeating_decimals :
  (repeating_to_fraction 0.36) / (repeating_to_fraction 0.12) = 3 :=
by
  sorry

end division_of_repeating_decimals_l11_11738


namespace solve_a_plus_b_l11_11191

theorem solve_a_plus_b (a b : ℚ) (h1 : 2 * a + 5 * b = 47) (h2 : 7 * a + 2 * b = 54) : a + b = -103 / 31 :=
by
  sorry

end solve_a_plus_b_l11_11191


namespace no_such_function_exists_l11_11779

def satisfies_condition (f : ℤ → ℤ) : Prop :=
  ∀ x y z : ℤ, f (x * y) + f (x * z) - f x * f (y * z) ≤ -1

theorem no_such_function_exists : (∃ f : ℤ → ℤ, satisfies_condition f) = false :=
by
  sorry

end no_such_function_exists_l11_11779


namespace diagonals_in_nonagon_l11_11363

-- Define the properties of the polygon
def convex : Prop := true
def sides (n : ℕ) : Prop := n = 9
def right_angles (count : ℕ) : Prop := count = 2

-- Define the formula for the number of diagonals in a polygon with 'n' sides
def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- The theorem definition
theorem diagonals_in_nonagon :
  convex →
  (sides 9) →
  (right_angles 2) →
  number_of_diagonals 9 = 27 :=
by
  sorry

end diagonals_in_nonagon_l11_11363


namespace shortest_side_of_right_triangle_l11_11327

theorem shortest_side_of_right_triangle 
  (a b : ℕ) (ha : a = 7) (hb : b = 10) (c : ℝ) (hright : a^2 + b^2 = c^2) :
  min a b = 7 :=
by
  sorry

end shortest_side_of_right_triangle_l11_11327


namespace proof_x_squared_plus_y_squared_l11_11296

def problem_conditions (x y : ℝ) :=
  x - y = 18 ∧ x*y = 9

theorem proof_x_squared_plus_y_squared (x y : ℝ) 
  (h : problem_conditions x y) : 
  x^2 + y^2 = 342 :=
by
  sorry

end proof_x_squared_plus_y_squared_l11_11296


namespace farmer_apples_count_l11_11862

theorem farmer_apples_count (initial : ℕ) (given : ℕ) (remaining : ℕ) 
  (h1 : initial = 127) (h2 : given = 88) : remaining = initial - given := 
by
  sorry

end farmer_apples_count_l11_11862


namespace expression_evaluation_l11_11799

theorem expression_evaluation : (2 - (-3) - 4 + (-5) + 6 - (-7) - 8 = 1) := 
by 
  sorry

end expression_evaluation_l11_11799


namespace safer_four_engine_airplane_l11_11328

theorem safer_four_engine_airplane (P : ℝ) (hP : 0 < P ∧ P < 1):
  (∃ p : ℝ, p = 1 - P ∧ (p^4 + 4 * p^3 * (1 - p) + 6 * p^2 * (1 - p)^2 > p^2 + 2 * p * (1 - p) ↔ P > 2 / 3)) :=
sorry

end safer_four_engine_airplane_l11_11328


namespace evaluate_expression_l11_11810

lemma pow_mod_four_cycle (n : ℕ) : (n % 4) = 1 → (i : ℂ)^n = i :=
by sorry

lemma pow_mod_four_cycle2 (n : ℕ) : (n % 4) = 2 → (i : ℂ)^n = -1 :=
by sorry

lemma pow_mod_four_cycle3 (n : ℕ) : (n % 4) = 3 → (i : ℂ)^n = -i :=
by sorry

lemma pow_mod_four_cycle4 (n : ℕ) : (n % 4) = 0 → (i : ℂ)^n = 1 :=
by sorry

theorem evaluate_expression : 
  (i : ℂ)^(2021) + (i : ℂ)^(2022) + (i : ℂ)^(2023) + (i : ℂ)^(2024) = 0 :=
by sorry

end evaluate_expression_l11_11810


namespace diagonal_length_l11_11046

noncomputable def rectangle_diagonal (p : ℝ) (r : ℝ) (d : ℝ) : Prop :=
  ∃ k : ℝ, p = 2 * ((5 * k) + (2 * k)) ∧ r = 5 / 2 ∧ 
           d = Real.sqrt (((5 * k)^2 + (2 * k)^2)) 

theorem diagonal_length 
  (p : ℝ) (r : ℝ) (d : ℝ)
  (h₁ : p = 72) 
  (h₂ : r = 5 / 2)
  : rectangle_diagonal p r d ↔ d = 194 / 7 := 
sorry

end diagonal_length_l11_11046


namespace price_of_cashew_nuts_l11_11320

theorem price_of_cashew_nuts 
  (C : ℝ)  -- price per kilo of cashew nuts
  (P_p : ℝ := 130)  -- price per kilo of peanuts
  (cashew_kilos : ℝ := 3)  -- kilos of cashew nuts bought
  (peanut_kilos : ℝ := 2)  -- kilos of peanuts bought
  (total_kilos : ℝ := 5)  -- total kilos of nuts bought
  (total_price_per_kilo : ℝ := 178)  -- total price per kilo of all nuts
  (h_total_cost : cashew_kilos * C + peanut_kilos * P_p = total_kilos * total_price_per_kilo) :
  C = 210 :=
sorry

end price_of_cashew_nuts_l11_11320


namespace intersection_M_N_l11_11068

def M (x : ℝ) : Prop := (x - 3) / (x + 1) > 0
def N (x : ℝ) : Prop := 3 * x + 2 > 0

theorem intersection_M_N :
  {x : ℝ | M x} ∩ {x : ℝ | N x} = {x : ℝ | 3 < x} :=
by
  sorry

end intersection_M_N_l11_11068


namespace focus_of_parabola_l11_11114

-- Define the given parabola equation
def parabola_eq (x y : ℝ) : Prop := y = (1/4) * x^2

-- Define the conditions about the focus and the parabola direction
def focus_on_y_axis : Prop := True -- Given condition
def opens_upwards : Prop := True -- Given condition

theorem focus_of_parabola (x y : ℝ) 
  (h1 : parabola_eq x y) 
  (h2 : focus_on_y_axis) 
  (h3 : opens_upwards) : 
  (x = 0 ∧ y = 1) :=
by
  sorry

end focus_of_parabola_l11_11114


namespace arithmetic_sequence_20th_term_l11_11586

theorem arithmetic_sequence_20th_term :
  let a := 2
  let d := 3
  let n := 20
  (a + (n - 1) * d) = 59 :=
by 
  sorry

end arithmetic_sequence_20th_term_l11_11586


namespace num_friends_bought_robots_l11_11376

def robot_cost : Real := 8.75
def tax_charged : Real := 7.22
def change_left : Real := 11.53
def initial_amount : Real := 80.0
def friends_bought_robots : Nat := 7

theorem num_friends_bought_robots :
  (initial_amount - (change_left + tax_charged)) / robot_cost = friends_bought_robots := sorry

end num_friends_bought_robots_l11_11376


namespace coats_count_l11_11893

def initial_minks : Nat := 30
def babies_per_mink : Nat := 6
def minks_per_coat : Nat := 15

def total_minks : Nat := initial_minks + (initial_minks * babies_per_mink)
def remaining_minks : Nat := total_minks / 2

theorem coats_count : remaining_minks / minks_per_coat = 7 := by
  -- Proof goes here
  sorry

end coats_count_l11_11893


namespace charlie_fraction_l11_11485

theorem charlie_fraction (J B C : ℕ) (f : ℚ) (hJ : J = 12) (hB : B = 10) 
  (h1 : B = (2 / 3) * C) (h2 : C = f * J + 9) : f = (1 / 2) := by
  sorry

end charlie_fraction_l11_11485


namespace fencing_required_l11_11696

theorem fencing_required (L W : ℝ) (hL : L = 20) (hA : 20 * W = 60) : 2 * W + L = 26 :=
by
  sorry

end fencing_required_l11_11696


namespace standard_deviation_calculation_l11_11212

theorem standard_deviation_calculation : 
  let mean := 16.2 
  let stddev := 2.3 
  mean - 2 * stddev = 11.6 :=
by
  sorry

end standard_deviation_calculation_l11_11212


namespace value_of_f_at_3_l11_11528

def f (x : ℝ) : ℝ := 9 * x^3 - 5 * x^2 - 3 * x + 7

theorem value_of_f_at_3 : f 3 = 196 := by
  sorry

end value_of_f_at_3_l11_11528


namespace ratio_of_a_to_c_l11_11997

variable {a b c d : ℚ}

theorem ratio_of_a_to_c (h₁ : a / b = 5 / 4) (h₂ : c / d = 4 / 3) (h₃ : d / b = 1 / 5) : 
  a / c = 75 / 16 := 
sorry

end ratio_of_a_to_c_l11_11997


namespace ellipse_problem_l11_11898

theorem ellipse_problem :
  (∃ (k : ℝ) (a θ : ℝ), 
    (∀ x y : ℝ, y = k * (x + 3) → (x^2 / 25 + y^2 / 16 = 1)) ∧
    (a > -3) ∧
    (∃ x y : ℝ, (x = - (25 / 3) ∧ y = k * (x + 3)) ∧ 
                 (x = D_fst ∧ y = D_snd) ∧ -- Point D(a, θ)
                 (x = M_fst ∧ y = M_snd) ∧ -- Point M
                 (x = N_fst ∧ y = N_snd)) ∧ -- Point N
    (∃ x y : ℝ, (x = -3 ∧ y = 0))) → 
    a = 5 :=
sorry

end ellipse_problem_l11_11898


namespace trig_identity_l11_11520

theorem trig_identity : 
  ( 4 * Real.sin (40 * Real.pi / 180) * Real.cos (40 * Real.pi / 180) / Real.cos (20 * Real.pi / 180) 
   - Real.tan (20 * Real.pi / 180) ) = Real.sqrt 3 := 
by
  sorry

end trig_identity_l11_11520


namespace expression_evaluation_l11_11219

theorem expression_evaluation:
  ( (1/3)^2000 * 27^669 + Real.sin (60 * Real.pi / 180) * Real.tan (60 * Real.pi / 180) + (2009 + Real.sin (25 * Real.pi / 180))^0 ) = 
  (2 + 29/54) := by
  sorry

end expression_evaluation_l11_11219


namespace probability_of_scoring_l11_11030

theorem probability_of_scoring :
  ∀ (p : ℝ), (p + (1 / 3) * p = 1) → (p = 3 / 4) → (p * (1 - p) = 3 / 16) :=
by
  intros p h1 h2
  sorry

end probability_of_scoring_l11_11030


namespace photos_ratio_l11_11559

theorem photos_ratio (L R C : ℕ) (h1 : R = L) (h2 : C = 12) (h3 : R = C + 24) :
  L / C = 3 :=
by 
  sorry

end photos_ratio_l11_11559


namespace num_possible_sums_l11_11641

theorem num_possible_sums (s : Finset ℕ) (hs : s.card = 80) (hsub: s ⊆ Finset.range 121) : 
  ∃ (n : ℕ), (n = 3201) ∧ ∀ U, U = s.sum id → ∃ (U_min U_max : ℕ), U_min = 3240 ∧ U_max = 6440 ∧ (U_min ≤ U ∧ U ≤ U_max) :=
sorry

end num_possible_sums_l11_11641


namespace notebooks_bought_l11_11850

def dan_total_spent : ℕ := 32
def backpack_cost : ℕ := 15
def pens_cost : ℕ := 1
def pencils_cost : ℕ := 1
def notebook_cost : ℕ := 3

theorem notebooks_bought :
  ∃ x : ℕ, dan_total_spent - (backpack_cost + pens_cost + pencils_cost) = x * notebook_cost ∧ x = 5 := 
by
  sorry

end notebooks_bought_l11_11850


namespace exam_fail_percentage_l11_11860

theorem exam_fail_percentage
  (total_candidates : ℕ := 2000)
  (girls : ℕ := 900)
  (pass_percent : ℝ := 0.32) :
  ((total_candidates - ((pass_percent * (total_candidates - girls)) + (pass_percent * girls))) / total_candidates) * 100 = 68 :=
by
  sorry

end exam_fail_percentage_l11_11860


namespace probability_of_selecting_cooking_l11_11466

theorem probability_of_selecting_cooking :
  let courses := ["planting", "cooking", "pottery", "carpentry"]
  let total_courses := courses.length
  (1 / total_courses) = 1 / 4 :=
by
  sorry

end probability_of_selecting_cooking_l11_11466


namespace roots_eq_solution_l11_11446

noncomputable def roots_eq (a b c x : ℝ) : Prop :=
  a * x^2 + b * x + c = 0

noncomputable def quadratic_roots (m n : ℝ) : Prop :=
  roots_eq 1 (-2) (-2025) m ∧ roots_eq 1 (-2) (-2025) n

theorem roots_eq_solution (m n : ℝ) (hm : roots_eq 1 (-2) (-2025) m) (hn : roots_eq 1 (-2) (-2025) n) : 
  m^2 - 3 * m - n = 2023 := 
sorry

end roots_eq_solution_l11_11446


namespace smallest_number_among_l11_11716

theorem smallest_number_among
  (π : ℝ) (Hπ_pos : π > 0) :
  ∀ (a b c d : ℝ), 
    (a = 0) → 
    (b = -1) → 
    (c = -1.5) → 
    (d = π) → 
    (∀ (x y : ℝ), (x > 0) → (y > 0) → (x > y) ↔ x - y > 0) → 
    (∀ (x : ℝ), x < 0 → x < 0) → 
    (∀ (x y : ℝ), (x > 0) → (y < 0) → x > y) → 
    (∀ (x y : ℝ), (x < 0) → (y < 0) → (|x| > |y|) → x < y) → 
  c = -1.5 := 
by
  intros a b c d Ha Hb Hc Hd Hpos Hneg HposNeg Habs
  sorry

end smallest_number_among_l11_11716


namespace cos_540_eq_neg1_l11_11326

theorem cos_540_eq_neg1 : Real.cos (540 * Real.pi / 180) = -1 := by
  sorry

end cos_540_eq_neg1_l11_11326


namespace f_at_3_l11_11491

theorem f_at_3 (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x + 2) = x + 3) : f 3 = 4 := 
sorry

end f_at_3_l11_11491


namespace distance_ratio_gt_9_l11_11458

theorem distance_ratio_gt_9 (points : Fin 1997 → ℝ × ℝ × ℝ) (M m : ℝ) :
  (∀ i j, i ≠ j → dist (points i) (points j) ≤ M) →
  (∀ i j, i ≠ j → dist (points i) (points j) ≥ m) →
  m ≠ 0 →
  M / m > 9 :=
by
  sorry

end distance_ratio_gt_9_l11_11458


namespace ellipse_fence_cost_is_correct_l11_11823

noncomputable def ellipse_perimeter (a b : ℝ) : ℝ :=
  Real.pi * (3 * (a + b) - Real.sqrt ((3 * a + b) * (a + 3 * b)))

noncomputable def fence_cost_per_meter (rate : ℝ) (a b : ℝ) : ℝ :=
  rate * ellipse_perimeter a b

theorem ellipse_fence_cost_is_correct :
  fence_cost_per_meter 3 16 12 = 265.32 :=
by
  sorry

end ellipse_fence_cost_is_correct_l11_11823


namespace arithmetic_sequence_sum_neq_l11_11687

theorem arithmetic_sequence_sum_neq (a : ℕ → ℤ) (d : ℤ) (S : ℕ → ℤ)
    (h_arith : ∀ n, a (n + 1) = a n + d)
    (h_sum : ∀ n, S n = (n * (a 1 + a n)) / 2)
    (h_abs_eq : abs (a 3) = abs (a 9))
    (h_d_neg : d < 0) : S 5 ≠ S 6 := by
  sorry

end arithmetic_sequence_sum_neq_l11_11687


namespace find_water_in_sport_formulation_l11_11671

noncomputable def standard_formulation : ℚ × ℚ × ℚ := (1, 12, 30)
noncomputable def sport_flavoring_to_corn : ℚ := 3 * (1 / 12)
noncomputable def sport_flavoring_to_water : ℚ := (1 / 2) * (1 / 30)
noncomputable def sport_formulation (f : ℚ) (c : ℚ) (w : ℚ) : Prop :=
  f / c = sport_flavoring_to_corn ∧ f / w = sport_flavoring_to_water

noncomputable def given_corn_syrup : ℚ := 8

theorem find_water_in_sport_formulation :
  ∀ (f c w : ℚ), sport_formulation f c w → c = given_corn_syrup → w = 120 :=
by
  sorry

end find_water_in_sport_formulation_l11_11671


namespace sequence_subsequence_l11_11057

theorem sequence_subsequence :
  ∃ (a : Fin 101 → ℕ), 
  (∀ i, a i = i + 1) ∧ 
  ∃ (b : Fin 11 → ℕ), 
  (b 0 < b 1 ∧ b 1 < b 2 ∧ b 2 < b 3 ∧ b 3 < b 4 ∧ b 4 < b 5 ∧ 
  b 5 < b 6 ∧ b 6 < b 7 ∧ b 7 < b 8 ∧ b 8 < b 9 ∧ b 9 < b 10) ∨ 
  (b 0 > b 1 ∧ b 1 > b 2 ∧ b 2 > b 3 ∧ b 3 > b 4 ∧ b 4 > b 5 ∧ 
  b 5 > b 6 ∧ b 6 > b 7 ∧ b 7 > b 8 ∧ b 8 > b 9 ∧ b 9 > b 10) :=
by {
  sorry
}

end sequence_subsequence_l11_11057


namespace car_travel_distance_l11_11041

noncomputable def car_distance_in_30_minutes : ℝ := 
  let train_speed : ℝ := 96
  let car_speed : ℝ := (5 / 8) * train_speed
  let travel_time : ℝ := 0.5  -- 30 minutes is 0.5 hours
  car_speed * travel_time

theorem car_travel_distance : car_distance_in_30_minutes = 30 := by
  sorry

end car_travel_distance_l11_11041


namespace measure_angle_PQR_is_55_l11_11980

noncomputable def measure_angle_PQR (POQ QOR : ℝ) : ℝ :=
  let POQ := 120
  let QOR := 130
  let POR := 360 - (POQ + QOR)
  let OPR := (180 - POR) / 2
  let OPQ := (180 - POQ) / 2
  let OQR := (180 - QOR) / 2
  OPQ + OQR

theorem measure_angle_PQR_is_55 : measure_angle_PQR 120 130 = 55 := by
  sorry

end measure_angle_PQR_is_55_l11_11980


namespace find_k_intersect_lines_l11_11564

theorem find_k_intersect_lines :
  ∃ (k : ℚ), ∀ (x y : ℚ), 
  (2 * x + 3 * y + 8 = 0) → (x - y - 1 = 0) → (x + k * y = 0) → k = -1/2 :=
by sorry

end find_k_intersect_lines_l11_11564


namespace points_lie_on_hyperbola_l11_11275

theorem points_lie_on_hyperbola (s : ℝ) :
  let x := 2 * (Real.exp s + Real.exp (-s))
  let y := 4 * (Real.exp s - Real.exp (-s))
  (x^2) / 16 - (y^2) / 64 = 1 :=
by
  sorry

end points_lie_on_hyperbola_l11_11275


namespace driver_schedule_l11_11550

-- Definitions based on the conditions
def one_way_trip_time := 160 -- in minutes (2 hours 40 minutes)
def round_trip_time := 320  -- in minutes (5 hours 20 minutes)
def rest_time := 60         -- in minutes (1 hour)

def Driver := ℕ

def A := 1
def B := 2
def C := 3
def D := 4

noncomputable def return_time_A := 760 -- 12:40 PM in minutes from day start (12 * 60 + 40)
noncomputable def earliest_departure_A := 820 -- 13:40 PM in minutes from day start (13 * 60 + 40)
noncomputable def departure_time_D := 785 -- 13:05 PM in minutes from day start (13 * 60 + 5)
noncomputable def second_trip_departure_time := 640 -- 10:40 AM in minutes from day start (10 * 60 + 40)

-- Problem statement
theorem driver_schedule : 
  ∃ (n : ℕ), n = 4 ∧ (∀ i : Driver, i = B → second_trip_departure_time = 640) :=
by
  -- Adding sorry to skip proof
  sorry

end driver_schedule_l11_11550


namespace sum_of_digits_eleven_l11_11724

-- Definitions for the problem conditions
def distinct_digits (p q r : Nat) : Prop :=
  p ≠ q ∧ p ≠ r ∧ q ≠ r ∧ p > 0 ∧ q > 0 ∧ r > 0 ∧ p < 10 ∧ q < 10 ∧ r < 10

def is_two_digit_prime (n : Nat) : Prop :=
  10 ≤ n ∧ n < 100 ∧ n.Prime

def concat_digits (x y : Nat) : Nat :=
  10 * x + y

def problem_conditions (p q r : Nat) : Prop :=
  distinct_digits p q r ∧
  is_two_digit_prime (concat_digits p q) ∧
  is_two_digit_prime (concat_digits p r) ∧
  is_two_digit_prime (concat_digits q r) ∧
  (concat_digits p q) * (concat_digits p r) = 221

-- Lean 4 statement to prove the sum of p, q, r is 11
theorem sum_of_digits_eleven (p q r : Nat) (h : problem_conditions p q r) : p + q + r = 11 :=
sorry

end sum_of_digits_eleven_l11_11724


namespace cube_point_problem_l11_11419
open Int

theorem cube_point_problem (n : ℤ) (x y z u : ℤ)
  (hx : x = 0 ∨ x = 8)
  (hy : y = 0 ∨ y = 12)
  (hz : z = 0 ∨ z = 6)
  (hu : 24 ∣ u)
  (hn : n = x + y + z + u) :
  (n ≠ 100) ∧ (n = 200) ↔ (n % 6 = 0 ∨ (n - 8) % 6 = 0) :=
by sorry

end cube_point_problem_l11_11419


namespace count_primes_5p2p1_minus_1_perfect_square_l11_11411

-- Define the predicate for a prime number
def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Predicate for perfect square
def is_perfect_square (n : ℕ) : Prop := 
  ∃ m : ℕ, m * m = n

-- The main theorem statement
theorem count_primes_5p2p1_minus_1_perfect_square :
  (∀ p : ℕ, is_prime p → is_perfect_square (5 * p * (2^(p + 1) - 1))) → ∃! p : ℕ, is_prime p ∧ is_perfect_square (5 * p * (2^(p + 1) - 1)) :=
sorry

end count_primes_5p2p1_minus_1_perfect_square_l11_11411


namespace sequence_sum_l11_11561

noncomputable def a₁ : ℝ := sorry
noncomputable def a₂ : ℝ := sorry
noncomputable def a₃ : ℝ := sorry
noncomputable def a₄ : ℝ := sorry
noncomputable def a₅ : ℝ := sorry
noncomputable def a₆ : ℝ := sorry
noncomputable def a₇ : ℝ := sorry
noncomputable def a₈ : ℝ := sorry
noncomputable def q : ℝ := sorry

axiom condition_1 : a₁ + a₂ + a₃ + a₄ = 1
axiom condition_2 : a₅ + a₆ + a₇ + a₈ = 2
axiom condition_3 : q^4 = 2

theorem sequence_sum : q = (2:ℝ)^(1/4) → a₁ + a₂ + a₃ + a₄ = 1 → 
  (a₁ * q^16 + a₂ * q^17 + a₃ * q^18 + a₄ * q^19) = 16 := 
by
  intros hq hsum_s4
  sorry

end sequence_sum_l11_11561


namespace work_days_A_l11_11994

theorem work_days_A (x : ℝ) (h1 : ∀ y : ℝ, y = 20) (h2 : ∀ z : ℝ, z = 5) 
  (h3 : ∀ w : ℝ, w = 0.41666666666666663) :
  x = 15 :=
  sorry

end work_days_A_l11_11994


namespace max_books_per_student_l11_11067

theorem max_books_per_student
  (total_students : ℕ)
  (students_0_books : ℕ)
  (students_1_book : ℕ)
  (students_2_books : ℕ)
  (students_at_least_3_books : ℕ)
  (avg_books_per_student : ℕ)
  (max_books_limit : ℕ)
  (total_books_available : ℕ) :
  total_students = 20 →
  students_0_books = 2 →
  students_1_book = 10 →
  students_2_books = 5 →
  students_at_least_3_books = total_students - students_0_books - students_1_book - students_2_books →
  avg_books_per_student = 2 →
  max_books_limit = 5 →
  total_books_available = 60 →
  avg_books_per_student * total_students = 40 →
  total_books_available = 60 →
  max_books_limit = 5 :=
by sorry

end max_books_per_student_l11_11067


namespace digit_multiplication_sum_l11_11079

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

end digit_multiplication_sum_l11_11079


namespace find_n_l11_11697

theorem find_n (n : ℕ) (h : 2^6 * 3^3 * n = Nat.factorial 10) : n = 2100 :=
by
sorry

end find_n_l11_11697


namespace k_lt_zero_l11_11598

noncomputable def k_negative (k : ℝ) : Prop :=
  (∃ x : ℝ, x < 0 ∧ k * x > 0) ∧ (∃ x : ℝ, x > 0 ∧ k * x < 0)

theorem k_lt_zero (k : ℝ) : k_negative k → k < 0 :=
by
  intros h
  sorry

end k_lt_zero_l11_11598


namespace range_of_3a_minus_b_l11_11310

theorem range_of_3a_minus_b (a b : ℝ) (ha : -5 < a) (ha' : a < 2) (hb : 1 < b) (hb' : b < 4) : 
  -19 < 3 * a - b ∧ 3 * a - b < 5 :=
by
  sorry

end range_of_3a_minus_b_l11_11310


namespace lucy_l11_11654

-- Define rounding function to nearest ten
def round_to_nearest_ten (x : Int) : Int :=
  if x % 10 < 5 then x - x % 10 else x + (10 - x % 10)

-- Define the problem with given conditions
def lucy_problem : Prop :=
  let sum := 68 + 57
  round_to_nearest_ten sum = 130

-- Statement of proof problem
theorem lucy's_correct_rounded_sum : lucy_problem := by
  sorry

end lucy_l11_11654


namespace findAngleC_findPerimeter_l11_11529

noncomputable def triangleCondition (a b c : ℝ) (A B C : ℝ) : Prop :=
  let m := (b+c, Real.sin A)
  let n := (a+b, Real.sin C - Real.sin B)
  m.1 * n.2 = m.2 * n.1 -- m parallel to n

noncomputable def lawOfSines (a b c A B C : ℝ) : Prop :=
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C

noncomputable def areaOfTriangle (a b c : ℝ) (A B C : ℝ) : ℝ :=
  0.5 * a * b * Real.sin C -- Area calculation by a, b, and angle between them

theorem findAngleC (a b c A B C : ℝ) : 
  triangleCondition a b c A B C ∧ lawOfSines a b c A B C → 
  Real.cos C = -1/2 :=
sorry

theorem findPerimeter (a b c A B C : ℝ) : 
  b = 4 ∧ areaOfTriangle a b c A B C = 4 * Real.sqrt 3 → 
  a = 4 ∧ b = 4 ∧ c = 4 * Real.sqrt 3 ∧ a + b + c = 8 + 4 * Real.sqrt 3 :=
sorry

end findAngleC_findPerimeter_l11_11529


namespace claire_photos_l11_11990

-- Define the number of photos taken by Claire, Lisa, and Robert
variables (C L R : ℕ)

-- Conditions based on the problem
def Lisa_photos (C : ℕ) := 3 * C
def Robert_photos (C : ℕ) := C + 24

-- Prove that C = 12 given the conditions
theorem claire_photos : 
  (L = Lisa_photos C) ∧ (R = Robert_photos C) ∧ (L = R) → C = 12 := 
by
  sorry

end claire_photos_l11_11990


namespace parabola_and_length_ef_l11_11650

theorem parabola_and_length_ef :
  ∃ a b : ℝ, (∀ x : ℝ, (x + 1) * (x - 3) = 0 → a * x^2 + b * x + 3 = 0) ∧ 
            (∀ x : ℝ, -a * x^2 + b * x + 3 = 7 / 4 → 
              ∃ x1 x2 : ℝ, x1 = -1 / 2 ∧ x2 = 5 / 2 ∧ abs (x2 - x1) = 3) := 
sorry

end parabola_and_length_ef_l11_11650


namespace solve_quadratic_l11_11786

theorem solve_quadratic (x : ℝ) : x^2 = x ↔ (x = 0 ∨ x = 1) :=
by
  sorry

end solve_quadratic_l11_11786


namespace simplify_expression_l11_11231

theorem simplify_expression (i : ℂ) (h : i^2 = -1) : 3 * (2 - i) + i * (3 + 2 * i) = 4 :=
by
  sorry

end simplify_expression_l11_11231


namespace four_clique_exists_in_tournament_l11_11709

open Finset

/-- Given a graph G with 9 vertices and 28 edges, prove that G contains a 4-clique. -/
theorem four_clique_exists_in_tournament 
  (V : Finset ℕ) (E : Finset (ℕ × ℕ)) 
  (hV : V.card = 9) 
  (hE : E.card = 28) :
  ∃ (S : Finset ℕ), S.card = 4 ∧ ∀ (v₁ v₂ : ℕ), v₁ ∈ S → v₂ ∈ S → v₁ ≠ v₂ → (v₁, v₂) ∈ E ∨ (v₂, v₁) ∈ E :=
sorry

end four_clique_exists_in_tournament_l11_11709


namespace polynomial_perfect_square_l11_11701

theorem polynomial_perfect_square (m : ℤ) : (∃ a : ℤ, a^2 = 25 ∧ x^2 + m*x + 25 = (x + a)^2) ↔ (m = 10 ∨ m = -10) :=
by sorry

end polynomial_perfect_square_l11_11701


namespace set_P_equals_set_interval_l11_11019

def A : Set ℝ := {x | x < 5}
def B : Set ℝ := {x | x <= 1 ∨ x >= 3}
def P : Set ℝ := {x | x ∈ A ∧ ¬ (x ∈ A ∧ x ∈ B)}

theorem set_P_equals_set_interval :
  P = {x | 1 < x ∧ x < 3} :=
sorry

end set_P_equals_set_interval_l11_11019


namespace employees_in_factory_l11_11346

theorem employees_in_factory (initial_total : ℕ) (init_prod : ℕ) (init_admin : ℕ)
  (increase_prod_frac : ℚ) (increase_admin_frac : ℚ) :
  initial_total = 1200 →
  init_prod = 800 →
  init_admin = 400 →
  increase_prod_frac = 0.35 →
  increase_admin_frac = 3 / 5 →
  init_prod + init_prod * increase_prod_frac +
  init_admin + init_admin * increase_admin_frac = 1720 := by
  intros h_total h_prod h_admin h_inc_prod h_inc_admin
  sorry

end employees_in_factory_l11_11346


namespace fred_earnings_l11_11360

-- Conditions as definitions
def initial_amount : ℕ := 23
def final_amount : ℕ := 86

-- Theorem to prove
theorem fred_earnings : final_amount - initial_amount = 63 := by
  sorry

end fred_earnings_l11_11360


namespace twentieth_common_number_l11_11355

theorem twentieth_common_number : 
  (∃ (m n : ℤ), (4 * m - 1) = (3 * n + 2) ∧ 20 * 12 - 1 = 239) := 
by
  sorry

end twentieth_common_number_l11_11355


namespace regular_octagon_interior_angle_l11_11723

-- Define the number of sides of a regular octagon
def num_sides : ℕ := 8

-- Define the formula for the sum of interior angles of a polygon
def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

-- Define the value of the sum of interior angles for an octagon
def sum_of_interior_angles_of_octagon : ℕ := sum_of_interior_angles num_sides

-- Define the measure of each interior angle of a regular polygon
def interior_angle_of_regular_polygon (n : ℕ) : ℕ := sum_of_interior_angles n / n

-- Define the value for each interior angle of the regular octagon
def interior_angle_of_regular_octagon : ℕ := interior_angle_of_regular_polygon num_sides

-- Prove that each interior angle of a regular octagon is 135 degrees
theorem regular_octagon_interior_angle :
  interior_angle_of_regular_octagon = 135 :=
by
  sorry

end regular_octagon_interior_angle_l11_11723


namespace part1_part2_l11_11541

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - a * x^2 - Real.log x

theorem part1 (a : ℝ) (h : 0 < a) (hf'1 : (1 - 2 * a * 1 - 1) = -2) :
  a = 1 ∧ (∀ x y : ℝ, y = -2 * (x - 1) → 2 * x + y - 2 = 0) :=
by
  sorry

theorem part2 {a : ℝ} (ha : a ≥ 1 / 8) :
  ∀ x : ℝ, (1 - 2 * a * x - 1 / x) ≤ 0 :=
by
  sorry

end part1_part2_l11_11541


namespace C_pays_228_for_cricket_bat_l11_11468

def CostPriceA : ℝ := 152

def ProfitA (price : ℝ) : ℝ := 0.20 * price

def SellingPriceA (price : ℝ) : ℝ := price + ProfitA price

def ProfitB (price : ℝ) : ℝ := 0.25 * price

def SellingPriceB (price : ℝ) : ℝ := price + ProfitB price

theorem C_pays_228_for_cricket_bat :
  SellingPriceB (SellingPriceA CostPriceA) = 228 :=
by
  sorry

end C_pays_228_for_cricket_bat_l11_11468


namespace functional_equation_f2023_l11_11878

theorem functional_equation_f2023 (f : ℝ → ℝ) (h_add : ∀ x y : ℝ, f (x + y) = f x + f y) (h_one : f 1 = 1) :
  f 2023 = 2023 := sorry

end functional_equation_f2023_l11_11878


namespace derivative_of_f_tangent_line_at_pi_l11_11600

noncomputable def f (x : ℝ) : ℝ := (Real.sin x) / x

theorem derivative_of_f (x : ℝ) (h : x ≠ 0) : deriv f x = (x * Real.cos x - Real.sin x) / (x ^ 2) :=
  sorry

theorem tangent_line_at_pi : 
  let M := (Real.pi, 0)
  let slope := -1 / Real.pi
  let tangent_line (x : ℝ) : ℝ := -x / Real.pi + 1
  ∀ (x y : ℝ), (x, y) = M → y = tangent_line x :=
  sorry

end derivative_of_f_tangent_line_at_pi_l11_11600


namespace top_z_teams_l11_11086

theorem top_z_teams (n : ℕ) (h : (n * (n - 1)) / 2 = 45) : n = 10 := 
sorry

end top_z_teams_l11_11086


namespace base_16_zeros_in_15_factorial_l11_11381

-- Definition of the factorial function
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Definition of the power function to generalize \( a^b \)
def power (a b : ℕ) : ℕ :=
  if b = 0 then 1 else a * power a (b - 1)

-- The constraints of the problem
def k_zeros_base_16 (n : ℕ) (k : ℕ) : Prop :=
  ∃ p, factorial n = p * power 16 k ∧ ¬ (∃ q, factorial n = q * power 16 (k + 1))

-- The main theorem we want to prove
theorem base_16_zeros_in_15_factorial : ∃ k, k_zeros_base_16 15 k ∧ k = 3 :=
by 
  sorry -- Proof to be found

end base_16_zeros_in_15_factorial_l11_11381


namespace probability_of_other_girl_l11_11531

theorem probability_of_other_girl (A B : Prop) (P : Prop → ℝ) 
    (hA : P A = 3 / 4) 
    (hAB : P (A ∧ B) = 1 / 4) : 
    P (B ∧ A) / P A = 1 / 3 := by 
  -- The proof is skipped using the sorry keyword.
  sorry

end probability_of_other_girl_l11_11531


namespace algae_coverage_double_l11_11090

theorem algae_coverage_double (algae_cov : ℕ → ℝ) (h1 : ∀ n : ℕ, algae_cov (n + 2) = 2 * algae_cov n)
  (h2 : algae_cov 24 = 1) : algae_cov 18 = 0.125 :=
by
  sorry

end algae_coverage_double_l11_11090


namespace quotient_A_div_B_l11_11937

-- Define A according to the given conditions
def A : ℕ := (8 * 10) + (13 * 1)

-- Define B according to the given conditions
def B : ℕ := 30 - 9 - 9 - 9

-- Prove that the quotient of A divided by B is 31
theorem quotient_A_div_B : (A / B) = 31 := by
  sorry

end quotient_A_div_B_l11_11937


namespace quotient_of_division_l11_11342

theorem quotient_of_division (dividend divisor remainder : ℕ) (h_dividend : dividend = 127) (h_divisor : divisor = 14) (h_remainder : remainder = 1) :
  (dividend - remainder) / divisor = 9 :=
by 
  -- Proof follows
  sorry

end quotient_of_division_l11_11342


namespace difference_between_local_and_face_value_of_7_in_65793_l11_11130

theorem difference_between_local_and_face_value_of_7_in_65793 :
  let numeral := 65793
  let digit := 7
  let place := 100
  let local_value := digit * place
  let face_value := digit
  local_value - face_value = 693 := 
by
  sorry

end difference_between_local_and_face_value_of_7_in_65793_l11_11130


namespace apples_number_l11_11874

def num_apples (A O B : ℕ) : Prop :=
  A = O + 27 ∧ O = B + 11 ∧ A + O + B = 301 → A = 122

theorem apples_number (A O B : ℕ) : num_apples A O B := by
  sorry

end apples_number_l11_11874


namespace sin_315_degree_l11_11984

theorem sin_315_degree : Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_315_degree_l11_11984


namespace find_k_of_sequence_l11_11473

theorem find_k_of_sequence (S : ℕ → ℤ) (a : ℕ → ℤ) (hS : ∀ n, S n = n^2 - 9 * n)
  (hS_recurr : ∀ n ≥ 2, a n = S n - S (n-1)) (h_a_k : ∃ k, 5 < a k ∧ a k < 8) : ∃ k, k = 8 :=
by
  sorry

end find_k_of_sequence_l11_11473


namespace triangle_side_length_l11_11392

   theorem triangle_side_length
   (A B C D E F : Type)
   (angle_bac angle_edf : Real)
   (AB AC DE DF : Real)
   (h1 : angle_bac = angle_edf)
   (h2 : AB = 5)
   (h3 : AC = 4)
   (h4 : DE = 2.5)
   (area_eq : (1 / 2) * AB * AC * Real.sin angle_bac = (1 / 2) * DE * DF * Real.sin angle_edf):
   DF = 8 :=
   by
   sorry
   
end triangle_side_length_l11_11392


namespace count_ordered_pairs_l11_11467

theorem count_ordered_pairs (x y : ℕ) (px : 0 < x) (py : 0 < y) (h : 2310 = 2 * 3 * 5 * 7 * 11) :
  (x * y = 2310 → ∃ n : ℕ, n = 32) :=
by
  sorry

end count_ordered_pairs_l11_11467


namespace find_num_terms_in_AP_l11_11425

-- Define the necessary conditions and prove the final result
theorem find_num_terms_in_AP
  (a d : ℝ) (n : ℕ)
  (h_even : n % 2 = 0)
  (h_last_term_difference : (n - 1 : ℝ) * d = 7.5)
  (h_sum_odd_terms : n * (a + (n - 2 : ℝ) / 2 * d) = 60)
  (h_sum_even_terms : n * (a + ((n - 1 : ℝ) / 2) * d + d) = 90) :
  n = 12 := 
sorry

end find_num_terms_in_AP_l11_11425


namespace calc_value_l11_11244

theorem calc_value : 2 + 3 * 4 - 5 + 6 = 15 := 
by 
  sorry

end calc_value_l11_11244


namespace product_of_two_numbers_l11_11992

theorem product_of_two_numbers (x y : ℝ) (h_diff : x - y = 12) (h_sum_of_squares : x^2 + y^2 = 245) : x * y = 50.30 :=
sorry

end product_of_two_numbers_l11_11992


namespace minimum_red_chips_l11_11514

theorem minimum_red_chips (w b r : ℕ) (h1 : b ≥ (1 / 3) * w) (h2 : b ≤ (1 / 4) * r) (h3 : w + b ≥ 70) : r ≥ 72 := by
  sorry

end minimum_red_chips_l11_11514


namespace flower_shop_february_roses_l11_11834

theorem flower_shop_february_roses (roses_oct : ℕ) (roses_nov : ℕ) (roses_dec : ℕ) (roses_jan : ℕ) (d : ℕ) :
  roses_oct = 108 →
  roses_nov = 120 →
  roses_dec = 132 →
  roses_jan = 144 →
  roses_nov - roses_oct = d →
  roses_dec - roses_nov = d →
  roses_jan - roses_dec = d →
  (roses_jan + d = 156) :=
by
  intros h_oct h_nov h_dec h_jan h_diff1 h_diff2 h_diff3
  rw [h_jan, h_diff1] at *
  sorry

end flower_shop_february_roses_l11_11834


namespace h_at_neg_eight_l11_11733

noncomputable def f (x : ℝ) : ℝ := x^3 + 3 * x^2 + x + 1

noncomputable def h (x : ℝ) (a b c : ℝ) : ℝ := (x - a^3) * (x - b^3) * (x - c^3)

theorem h_at_neg_eight (a b c : ℝ) (hf : f a = 0) (hf_b : f b = 0) (hf_c : f c = 0) :
  h (-8) a b c = -115 :=
  sorry

end h_at_neg_eight_l11_11733


namespace radius_of_two_equal_circles_eq_16_l11_11978

noncomputable def radius_of_congruent_circles : ℝ := 16

theorem radius_of_two_equal_circles_eq_16 :
  ∃ x : ℝ, 
    (∀ r1 r2 r3 : ℝ, r1 = 4 ∧ r2 = r3 ∧ r2 = x ∧ 
    ∃ line : ℝ → ℝ → Prop, 
    (line 0 r1) ∧ (line 0 r2)  ∧ 
    (line 0 r3) ∧ 
    (line r2 r3) ∧
    (line r1 r2)  ∧ (line r1 r3) ∧ (line (r1 + r2) r2) ) 
    → x = 16 := sorry

end radius_of_two_equal_circles_eq_16_l11_11978


namespace integer_points_on_line_l11_11340

/-- Given a line that passes through points C(3, 3) and D(150, 250),
prove that the number of other points with integer coordinates
that lie strictly between C and D is 48. -/
theorem integer_points_on_line {C D : ℝ × ℝ} (hC : C = (3, 3)) (hD : D = (150, 250)) :
  ∃ (n : ℕ), n = 48 ∧ 
  ∀ p : ℝ × ℝ, C.1 < p.1 ∧ p.1 < D.1 ∧ 
  C.2 < p.2 ∧ p.2 < D.2 → 
  (∃ (k : ℤ), p.1 = ↑k ∧ p.2 = (5/3) * p.1 - 2) :=
sorry

end integer_points_on_line_l11_11340


namespace sequence_a_n_l11_11286

theorem sequence_a_n {n : ℕ} (S : ℕ → ℚ) (a : ℕ → ℚ)
  (hS : ∀ n, S n = (2/3 : ℚ) * n^2 - (1/3 : ℚ) * n)
  (ha : ∀ n, a n = if n = 1 then S n else S n - S (n - 1)) :
  ∀ n, a n = (4/3 : ℚ) * n - 1 := 
by
  sorry

end sequence_a_n_l11_11286


namespace june_eggs_count_l11_11324

theorem june_eggs_count :
  (2 * 5) + 3 + 4 = 17 := 
by 
  sorry

end june_eggs_count_l11_11324


namespace find_a_plus_b_l11_11829

theorem find_a_plus_b (a b : ℝ) (h : |a - 2| + |b + 3| = 0) : a + b = -1 :=
by {
  sorry
}

end find_a_plus_b_l11_11829


namespace balloon_count_l11_11821

theorem balloon_count (gold_balloon silver_balloon black_balloon blue_balloon green_balloon total_balloon : ℕ) (h1 : gold_balloon = 141) 
                      (h2 : silver_balloon = (gold_balloon / 3) * 5) 
                      (h3 : black_balloon = silver_balloon / 2) 
                      (h4 : blue_balloon = black_balloon / 2) 
                      (h5 : green_balloon = (blue_balloon / 4) * 3) 
                      (h6 : total_balloon = gold_balloon + silver_balloon + black_balloon + blue_balloon + green_balloon): 
                      total_balloon = 593 :=
by 
  sorry

end balloon_count_l11_11821


namespace number_of_erasers_l11_11597

theorem number_of_erasers (P E : ℕ) (h1 : P + E = 240) (h2 : P = E - 2) : E = 121 := by
  sorry

end number_of_erasers_l11_11597


namespace arithmetic_sequence_common_difference_l11_11171

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h1 : a 1 = 3)
  (h2 : S 5 = 35)
  (h3 : ∀ n, S n = n * a 1 + n * (n - 1) / 2 * d) :
  d = 2 :=
by
  sorry

end arithmetic_sequence_common_difference_l11_11171


namespace find_AC_l11_11008

theorem find_AC (A B C : ℝ) (r1 r2 : ℝ) (AB : ℝ) (AC : ℝ) 
  (h_rad1 : r1 = 1) (h_rad2 : r2 = 3) (h_AB : AB = 2 * Real.sqrt 5) 
  (h_AC : AC = AB / 4) :
  AC = Real.sqrt 5 / 2 :=
by
  sorry

end find_AC_l11_11008


namespace intersection_of_A_and_B_l11_11891

def setA : Set ℝ := {-1, 1, 2, 4}
def setB : Set ℝ := {x | abs (x - 1) ≤ 1}

theorem intersection_of_A_and_B : setA ∩ setB = {1, 2} :=
by
  sorry

end intersection_of_A_and_B_l11_11891


namespace Eva_arts_marks_difference_l11_11560

noncomputable def marks_difference_in_arts : ℕ := 
  let M1 := 90
  let A2 := 90
  let S1 := 60
  let M2 := 80
  let A1 := A2 - 75
  let S2 := 90
  A2 - A1

theorem Eva_arts_marks_difference : marks_difference_in_arts = 75 := by
  sorry

end Eva_arts_marks_difference_l11_11560


namespace initial_books_l11_11305

theorem initial_books (sold_books : ℕ) (given_books : ℕ) (remaining_books : ℕ) 
                      (h1 : sold_books = 11)
                      (h2 : given_books = 35)
                      (h3 : remaining_books = 62) :
  (sold_books + given_books + remaining_books = 108) :=
by
  -- Proof skipped
  sorry

end initial_books_l11_11305


namespace binary_111_is_7_l11_11635

def binary_to_decimal (b0 b1 b2 : ℕ) : ℕ :=
  b0 * (2^0) + b1 * (2^1) + b2 * (2^2)

theorem binary_111_is_7 : binary_to_decimal 1 1 1 = 7 :=
by
  -- We will provide the proof here.
  sorry

end binary_111_is_7_l11_11635


namespace area_inside_rectangle_outside_circles_is_4_l11_11543

-- Specify the problem in Lean 4
theorem area_inside_rectangle_outside_circles_is_4 :
  let CD := 3
  let DA := 5
  let radius_A := 1
  let radius_B := 2
  let radius_C := 3
  let area_rectangle := CD * DA
  let area_circles := (radius_A^2 + radius_B^2 + radius_C^2) * Real.pi / 4
  abs (area_rectangle - area_circles - 4) < 1 :=
by
  repeat { sorry }

end area_inside_rectangle_outside_circles_is_4_l11_11543


namespace sum_of_digits_of_greatest_prime_divisor_l11_11553

-- Define the number 32767
def number : ℕ := 32767

-- Assert that 32767 is 2^15 - 1
lemma number_def : number = 2^15 - 1 := by
  sorry

-- State that 151 is the greatest prime divisor of 32767
lemma greatest_prime_divisor : Nat.Prime 151 ∧ ∀ p : ℕ, Nat.Prime p → p ∣ number → p ≤ 151 := by
  sorry

-- Calculate the sum of the digits of 151
def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |>.sum

-- Conclude the sum of the digits of the greatest prime divisor is 7
theorem sum_of_digits_of_greatest_prime_divisor : sum_of_digits 151 = 7 := by
  sorry

end sum_of_digits_of_greatest_prime_divisor_l11_11553


namespace sphere_volume_l11_11930

theorem sphere_volume (r : ℝ) (h : 4 * Real.pi * r^2 = 36 * Real.pi) : (4/3) * Real.pi * r^3 = 36 * Real.pi := 
sorry

end sphere_volume_l11_11930


namespace unique_integer_solution_l11_11780

theorem unique_integer_solution (a b : ℤ) : 
  ∀ x₁ x₂ : ℤ, (x₁ - a) * (x₁ - b) * (x₁ - 3) + 1 = 0 ∧ (x₂ - a) * (x₂ - b) * (x₂ - 3) + 1 = 0 → x₁ = x₂ :=
by
  sorry

end unique_integer_solution_l11_11780


namespace circles_tangent_internally_l11_11367

theorem circles_tangent_internally 
  (x y : ℝ) 
  (h : x^4 - 16 * x^2 + 2 * x^2 * y^2 - 16 * y^2 + y^4 = 4 * x^3 + 4 * x * y^2 - 64 * x) :
  ∃ c₁ c₂ : ℝ × ℝ, 
    (c₁ = (0, 0)) ∧ (c₂ = (2, 0)) ∧ 
    ((x - c₁.1)^2 + (y - c₁.2)^2 = 16) ∧ 
    ((x - c₂.1)^2 + (y - c₂.2)^2 = 4) ∧
    dist c₁ c₂ = 2 := 
sorry

end circles_tangent_internally_l11_11367


namespace pet_store_cages_l11_11368

theorem pet_store_cages (initial_puppies sold_puppies puppies_per_cage : ℕ) 
  (h1 : initial_puppies = 78) (h2 : sold_puppies = 30) (h3 : puppies_per_cage = 8) : 
  (initial_puppies - sold_puppies) / puppies_per_cage = 6 := 
by 
  sorry

end pet_store_cages_l11_11368


namespace ratio_length_to_width_l11_11228

theorem ratio_length_to_width
  (w l : ℕ)
  (pond_length : ℕ)
  (field_length : ℕ)
  (pond_area : ℕ)
  (field_area : ℕ)
  (pond_to_field_area_ratio : ℚ)
  (field_length_given : field_length = 28)
  (pond_length_given : pond_length = 7)
  (pond_area_def : pond_area = pond_length * pond_length)
  (pond_to_field_area_ratio_def : pond_to_field_area_ratio = 1 / 8)
  (field_area_def : field_area = pond_area * 8)
  (field_area_calc : field_area = field_length * w) :
  (field_length / w) = 2 :=
by
  sorry

end ratio_length_to_width_l11_11228


namespace total_animals_in_shelter_l11_11230

def initial_cats : ℕ := 15
def adopted_cats := initial_cats / 3
def replacement_cats := 2 * adopted_cats
def current_cats := initial_cats - adopted_cats + replacement_cats
def additional_dogs := 2 * current_cats
def total_animals := current_cats + additional_dogs

theorem total_animals_in_shelter : total_animals = 60 := by
  sorry

end total_animals_in_shelter_l11_11230


namespace plantable_area_l11_11393

noncomputable def flowerbed_r := 10
noncomputable def path_w := 4
noncomputable def full_area := 100 * Real.pi
noncomputable def segment_area := 20.67 * Real.pi * 2 -- each path affects two segments

theorem plantable_area :
  full_area - segment_area = 58.66 * Real.pi := 
by sorry

end plantable_area_l11_11393


namespace tan_half_angle_product_l11_11181

theorem tan_half_angle_product (a b : ℝ) (h : 3 * (Real.cos a + Real.cos b) + 6 * (Real.cos a * Real.cos b + 1) = 0) :
  ∃ (x : ℝ), x = Real.tan (a / 2) * Real.tan (b / 2) ∧ (x = Real.sqrt 2 ∨ x = -Real.sqrt 2) := 
sorry

end tan_half_angle_product_l11_11181


namespace maximize_operation_l11_11693

-- Definitions from the conditions
def is_three_digit_integer (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

-- The proof statement
theorem maximize_operation : ∃ n, is_three_digit_integer n ∧ (∀ m, is_three_digit_integer m → 3 * (300 - m) ≤ 600) :=
by {
  -- Placeholder for the actual proof
  sorry
}

end maximize_operation_l11_11693


namespace least_five_digit_congruent_to_6_mod_17_l11_11621

theorem least_five_digit_congruent_to_6_mod_17 :
  ∃ (x : ℕ), 10000 ≤ x ∧ x ≤ 99999 ∧ x % 17 = 6 ∧
  ∀ (y : ℕ), 10000 ≤ y ∧ y ≤ 99999 ∧ y % 17 = 6 → x ≤ y := 
sorry

end least_five_digit_congruent_to_6_mod_17_l11_11621


namespace cone_slant_height_l11_11612

theorem cone_slant_height (r l : ℝ) (h1 : r = 1)
  (h2 : 2 * r * Real.pi = (1 / 2) * 2 * l * Real.pi) :
  l = 2 :=
by
  -- Proof steps go here
  sorry

end cone_slant_height_l11_11612


namespace miles_per_gallon_city_l11_11620

theorem miles_per_gallon_city
  (T : ℝ) -- tank size
  (h c : ℝ) -- miles per gallon on highway 'h' and in the city 'c'
  (h_eq : h = (462 / T))
  (c_eq : c = (336 / T))
  (relation : c = h - 9)
  (solution : c = 24) : c = 24 := 
sorry

end miles_per_gallon_city_l11_11620


namespace intersection_in_fourth_quadrant_l11_11865

theorem intersection_in_fourth_quadrant :
  (∃ x y : ℝ, y = -x ∧ y = 2 * x - 1 ∧ x = 1 ∧ y = -1) ∧ (1 > 0 ∧ -1 < 0) :=
by
  sorry

end intersection_in_fourth_quadrant_l11_11865


namespace quadratic_inverse_sum_roots_l11_11351

theorem quadratic_inverse_sum_roots (x1 x2 : ℝ) (h1 : x1^2 - 2023 * x1 + 1 = 0) (h2 : x2^2 - 2023 * x2 + 1 = 0) : 
  (1/x1 + 1/x2) = 2023 :=
by
  -- We outline the proof steps that should be accomplished.
  -- These will be placeholders and not part of the actual statement.
  -- sorry allows us to skip the proof.
  sorry

end quadratic_inverse_sum_roots_l11_11351


namespace rectangle_area_error_l11_11250

theorem rectangle_area_error (A B : ℝ) :
  let A' := 1.08 * A
  let B' := 1.08 * B
  let actual_area := A * B
  let measured_area := A' * B'
  let percentage_error := ((measured_area - actual_area) / actual_area) * 100
  percentage_error = 16.64 :=
by
  sorry

end rectangle_area_error_l11_11250


namespace Ben_hits_7_l11_11773

def regions : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
def Alice_score : ℕ := 18
def Ben_score : ℕ := 13
def Cindy_score : ℕ := 19
def Dave_score : ℕ := 16
def Ellen_score : ℕ := 20
def Frank_score : ℕ := 5

def hit_score (name : String) (region1 region2 : ℕ) (score : ℕ) : Prop :=
  region1 ∈ regions ∧ region2 ∈ regions ∧ region1 ≠ region2 ∧ region1 + region2 = score

theorem Ben_hits_7 :
  ∃ r1 r2, hit_score "Ben" r1 r2 Ben_score ∧ (r1 = 7 ∨ r2 = 7) :=
sorry

end Ben_hits_7_l11_11773


namespace nicky_run_time_l11_11830

-- Define the constants according to the conditions in the problem
def head_start : ℕ := 100 -- Nicky's head start (meters)
def cr_speed : ℕ := 8 -- Cristina's speed (meters per second)
def ni_speed : ℕ := 4 -- Nicky's speed (meters per second)

-- Define the event of Cristina catching up to Nicky
def meets_at_time (t : ℕ) : Prop :=
  cr_speed * t = head_start + ni_speed * t

-- The proof statement
theorem nicky_run_time : ∃ t : ℕ, meets_at_time t ∧ t = 25 :=
by
  sorry

end nicky_run_time_l11_11830


namespace notebook_and_pencil_cost_l11_11704

theorem notebook_and_pencil_cost :
  ∃ (x y : ℝ), 6 * x + 4 * y = 9.2 ∧ 3 * x + y = 3.8 ∧ x + y = 1.8 :=
by
  sorry

end notebook_and_pencil_cost_l11_11704


namespace calculate_roots_l11_11170

noncomputable def cube_root (x : ℝ) := x^(1/3 : ℝ)
noncomputable def square_root (x : ℝ) := x^(1/2 : ℝ)

theorem calculate_roots : cube_root (-8) + square_root 9 = 1 :=
by
  sorry

end calculate_roots_l11_11170


namespace find_group_2018_l11_11713

-- Definition of the conditions
def group_size (n : Nat) : Nat := 3 * n - 2

def total_numbers (n : Nat) : Nat := 
  (3 * n * n - n) / 2

theorem find_group_2018 : ∃ n : Nat, total_numbers (n - 1) < 1009 ∧ total_numbers n ≥ 1009 ∧ n = 27 :=
  by
  -- This forms the structure for the proof
  sorry

end find_group_2018_l11_11713


namespace prime_between_40_50_largest_prime_lt_100_l11_11198

def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def between (n m k : ℕ) : Prop := n < k ∧ k < m

theorem prime_between_40_50 :
  {x : ℕ | between 40 50 x ∧ isPrime x} = {41, 43, 47} :=
sorry

theorem largest_prime_lt_100 :
  ∃ p : ℕ, isPrime p ∧ p < 100 ∧ ∀ q : ℕ, isPrime q ∧ q < 100 → q ≤ p :=
sorry

end prime_between_40_50_largest_prime_lt_100_l11_11198


namespace find_weight_of_second_square_l11_11120

-- Define the initial conditions
def uniform_density_thickness (density : ℝ) (thickness : ℝ) : Prop :=
  ∀ (l₁ l₂ : ℝ), l₁ = l₂ → density = thickness

-- Define the first square properties
def first_square (side_length₁ weight₁ : ℝ) : Prop :=
  side_length₁ = 4 ∧ weight₁ = 16

-- Define the second square properties
def second_square (side_length₂ : ℝ) : Prop :=
  side_length₂ = 6

-- Define the proportional relationship between the area and weight
def proportional_weight (side_length₁ weight₁ side_length₂ weight₂ : ℝ) : Prop :=
  (side_length₁^2 / weight₁) = (side_length₂^2 / weight₂)

-- Lean statement to prove the weight of the second square
theorem find_weight_of_second_square (density thickness side_length₁ weight₁ side_length₂ weight₂ : ℝ)
  (h_density_thickness : uniform_density_thickness density thickness)
  (h_first_square : first_square side_length₁ weight₁)
  (h_second_square : second_square side_length₂)
  (h_proportional_weight : proportional_weight side_length₁ weight₁ side_length₂ weight₂) : 
  weight₂ = 36 :=
by 
  sorry

end find_weight_of_second_square_l11_11120


namespace area_of_trapezium_l11_11092

variables (x : ℝ) (h : x > 0)

def shorter_base := 2 * x
def altitude := 2 * x
def longer_base := 6 * x

theorem area_of_trapezium (hx : x > 0) :
  (1 / 2) * (shorter_base x + longer_base x) * altitude x = 8 * x^2 := 
sorry

end area_of_trapezium_l11_11092


namespace sequence_sum_l11_11706

theorem sequence_sum (r : ℝ) (x y : ℝ)
  (a : ℕ → ℝ)
  (h1 : a 1 = 4096)
  (h2 : a 2 = 1024)
  (h3 : a 3 = 256)
  (h4 : a 6 = 4)
  (h5 : a 7 = 1)
  (h6 : a 8 = 0.25)
  (h_sequence : ∀ n, a (n + 1) = r * a n)
  (h_r : r = 1 / 4) :
  x + y = 80 :=
sorry

end sequence_sum_l11_11706


namespace minimum_value_sum_l11_11752

noncomputable def min_value (a b c : ℝ) : ℝ :=
  (a / (2 * b)) + (b / (4 * c)) + (c / (8 * a))

theorem minimum_value_sum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  min_value a b c >= 3/4 :=
by
  sorry

end minimum_value_sum_l11_11752


namespace largest_spherical_ball_radius_in_torus_l11_11692

theorem largest_spherical_ball_radius_in_torus 
    (inner_radius outer_radius : ℝ) 
    (circle_center : ℝ × ℝ × ℝ) 
    (circle_radius : ℝ) 
    (r : ℝ)
    (h0 : inner_radius = 2)
    (h1 : outer_radius = 4)
    (h2 : circle_center = (3, 0, 1))
    (h3 : circle_radius = 1)
    (h4 : 3^2 + (r - 1)^2 = (r + 1)^2) :
    r = 9 / 4 :=
by
  sorry

end largest_spherical_ball_radius_in_torus_l11_11692


namespace find_n_value_l11_11098

theorem find_n_value (AB AC n m : ℕ) (h1 : AB = 33) (h2 : AC = 21) (h3 : AD = m) (h4 : DE = m) (h5 : EC = m) (h6 : BC = n) : 
  ∃ m : ℕ, m > 7 ∧ m < 21 ∧ n = 30 := 
by sorry

end find_n_value_l11_11098


namespace find_m_from_arithmetic_sequence_l11_11158

theorem find_m_from_arithmetic_sequence (S : ℕ → ℤ) (m : ℕ) 
  (h1 : S (m - 1) = -4) (h2 : S m = 0) (h3 : S (m + 1) = 6) : m = 5 := by
  sorry

end find_m_from_arithmetic_sequence_l11_11158


namespace average_tomatoes_per_day_l11_11501

theorem average_tomatoes_per_day :
  let t₁ := 120
  let t₂ := t₁ + 50
  let t₃ := 2 * t₂
  let t₄ := t₁ / 2
  (t₁ + t₂ + t₃ + t₄) / 4 = 172.5 := by
  sorry

end average_tomatoes_per_day_l11_11501


namespace leif_fruit_weight_difference_l11_11494

theorem leif_fruit_weight_difference :
  let apples_ounces := 27.5
  let grams_per_ounce := 28.35
  let apples_grams := apples_ounces * grams_per_ounce
  let dozens_oranges := 5.5
  let oranges_per_dozen := 12
  let total_oranges := dozens_oranges * oranges_per_dozen
  let weight_per_orange := 45
  let oranges_grams := total_oranges * weight_per_orange
  let weight_difference := oranges_grams - apples_grams
  weight_difference = 2190.375 := by
{
  sorry
}

end leif_fruit_weight_difference_l11_11494


namespace uncle_taller_than_james_l11_11390

def james_initial_height (uncle_height : ℕ) : ℕ := (2 * uncle_height) / 3

def james_final_height (initial_height : ℕ) (growth_spurt : ℕ) : ℕ := initial_height + growth_spurt

theorem uncle_taller_than_james (uncle_height : ℕ) (growth_spurt : ℕ) :
  uncle_height = 72 →
  growth_spurt = 10 →
  uncle_height - (james_final_height (james_initial_height uncle_height) growth_spurt) = 14 :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end uncle_taller_than_james_l11_11390


namespace f_eq_f_at_neg_one_f_at_neg_500_l11_11224

noncomputable def f : ℝ → ℝ := sorry

theorem f_eq : ∀ x y : ℝ, f (x * y) + x = x * f y + f x := sorry
theorem f_at_neg_one : f (-1) = 1 := sorry

theorem f_at_neg_500 : f (-500) = 999 := sorry

end f_eq_f_at_neg_one_f_at_neg_500_l11_11224


namespace find_sum_of_perimeters_l11_11686

variables (x y : ℝ)
noncomputable def sum_of_perimeters := 4 * x + 4 * y

theorem find_sum_of_perimeters (h1 : x^2 + y^2 = 65) (h2 : x^2 - y^2 = 33) :
  sum_of_perimeters x y = 44 :=
sorry

end find_sum_of_perimeters_l11_11686


namespace find_inverse_l11_11940

noncomputable def inverse_matrix_2x2 (a b c d : ℝ) : ℝ × ℝ × ℝ × ℝ :=
  if ad_bc : (a * d - b * c) = 0 then (0, 0, 0, 0)
  else (d / (a * d - b * c), -b / (a * d - b * c), -c / (a * d - b * c), a / (a * d - b * c))

theorem find_inverse :
  inverse_matrix_2x2 5 7 2 3 = (3, -7, -2, 5) :=
by 
  sorry

end find_inverse_l11_11940


namespace r_daily_earnings_l11_11248

-- Given conditions as definitions
def daily_earnings (P Q R : ℕ) : Prop :=
(P + Q + R) * 9 = 1800 ∧ (P + R) * 5 = 600 ∧ (Q + R) * 7 = 910

-- Theorem statement corresponding to the problem
theorem r_daily_earnings : ∃ R : ℕ, ∀ P Q : ℕ, daily_earnings P Q R → R = 50 :=
by sorry

end r_daily_earnings_l11_11248


namespace distance_between_centers_of_tangent_circles_l11_11915

theorem distance_between_centers_of_tangent_circles
  (R r d : ℝ) (h1 : R = 8) (h2 : r = 3) (h3 : d = R + r) : d = 11 :=
by
  -- Insert proof here
  sorry

end distance_between_centers_of_tangent_circles_l11_11915


namespace variance_transformation_l11_11960

theorem variance_transformation (a1 a2 a3 : ℝ) 
  (h1 : (a1 + a2 + a3) / 3 = 4) 
  (h2 : ((a1 - 4)^2 + (a2 - 4)^2 + (a3 - 4)^2) / 3 = 3) : 
  ((3 * a1 - 2 - (3 * 4 - 2))^2 + (3 * a2 - 2 - (3 * 4 - 2))^2 + (3 * a3 - 2 - (3 * 4 - 2))^2) / 3 = 27 := 
sorry

end variance_transformation_l11_11960


namespace range_of_a_l11_11977

def new_operation (x y : ℝ) : ℝ := x * (1 - y)

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, new_operation x (x - a) > 1) ↔ (a < -3 ∨ 1 < a) := 
by
  sorry

end range_of_a_l11_11977


namespace total_marbles_l11_11877

def Mary_marbles : ℕ := 9
def Joan_marbles : ℕ := 3
def Peter_marbles : ℕ := 7

theorem total_marbles : Mary_marbles + Joan_marbles + Peter_marbles = 19 := by
  sorry

end total_marbles_l11_11877


namespace union_of_sets_l11_11378

def setA : Set ℕ := {0, 1}
def setB : Set ℕ := {0, 2}

theorem union_of_sets : setA ∪ setB = {0, 1, 2} := 
sorry

end union_of_sets_l11_11378


namespace inequality_proof_l11_11802

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (1 + 4 * a / (b + c)) * (1 + 4 * b / (a + c)) * (1 + 4 * c / (a + b)) > 25 :=
sorry

end inequality_proof_l11_11802


namespace number_of_newborn_members_l11_11999

theorem number_of_newborn_members (N : ℝ) (h : (9/10 : ℝ) ^ 3 * N = 291.6) : N = 400 :=
sorry

end number_of_newborn_members_l11_11999


namespace probability_two_win_one_lose_l11_11690

noncomputable def p_A : ℚ := 1 / 5
noncomputable def p_B : ℚ := 3 / 8
noncomputable def p_C : ℚ := 2 / 7

noncomputable def P_two_win_one_lose : ℚ :=
  p_A * p_B * (1 - p_C) +
  p_A * p_C * (1 - p_B) +
  p_B * p_C * (1 - p_A)

theorem probability_two_win_one_lose :
  P_two_win_one_lose = 49 / 280 :=
by
  sorry

end probability_two_win_one_lose_l11_11690


namespace total_hangers_l11_11552

theorem total_hangers (pink green blue yellow orange purple red : ℕ) 
  (h_pink : pink = 7)
  (h_green : green = 4)
  (h_blue : blue = green - 1)
  (h_yellow : yellow = blue - 1)
  (h_orange : orange = 2 * pink)
  (h_purple : purple = yellow + 3)
  (h_red : red = purple / 2) :
  pink + green + blue + yellow + orange + purple + red = 37 :=
sorry

end total_hangers_l11_11552


namespace additional_people_needed_l11_11258

-- Define the conditions
def num_people_initial := 9
def work_done_initial := 3 / 5
def days_initial := 14
def days_remaining := 4

-- Calculated values based on conditions
def work_rate_per_person : ℚ :=
  work_done_initial / (num_people_initial * days_initial)

def work_remaining : ℚ := 1 - work_done_initial

def total_people_needed : ℚ :=
  work_remaining / (work_rate_per_person * days_remaining)

-- Formulate the statement to prove
theorem additional_people_needed :
  total_people_needed - num_people_initial = 12 :=
by
  sorry

end additional_people_needed_l11_11258


namespace consecutive_days_without_meeting_l11_11261

/-- In March 1987, there are 31 days, starting on a Sunday.
There are 11 club meetings to be held, and no meetings are on Saturdays or Sundays.
This theorem proves that there will be at least three consecutive days without a meeting. -/
theorem consecutive_days_without_meeting (meetings : Finset ℕ) :
  (∀ x ∈ meetings, 1 ≤ x ∧ x ≤ 31 ∧ ¬ ∃ k, x = 7 * k + 1 ∨ x = 7 * k + 2) →
  meetings.card = 11 →
  ∃ i, 1 ≤ i ∧ i + 2 ≤ 31 ∧ ¬ (i ∈ meetings ∨ (i + 1) ∈ meetings ∨ (i + 2) ∈ meetings) :=
by
  sorry

end consecutive_days_without_meeting_l11_11261


namespace tiffany_total_lives_l11_11616

-- Define the conditions
def initial_lives : Float := 43.0
def hard_part_won : Float := 14.0
def next_level_won : Float := 27.0

-- State the theorem
theorem tiffany_total_lives : 
  initial_lives + hard_part_won + next_level_won = 84.0 :=
by 
  sorry

end tiffany_total_lives_l11_11616


namespace problem_inequality_l11_11924

theorem problem_inequality {n : ℕ} {a : ℕ → ℕ} (h : ∀ i j : ℕ, 1 ≤ i → i < j → j ≤ n → a i < a j → (a j - a i) ∣ a i) 
  (h_sorted : ∀ i j : ℕ, 1 ≤ i → i < j → j ≤ n → a i < a j)
  (h_pos : ∀ i : ℕ, 1 ≤ i → i ≤ n → 0 < a i) 
  (i j : ℕ) (hi : 1 ≤ i) (hij : i < j) (hj : j ≤ n) : i * a j ≤ j * a i := 
sorry

end problem_inequality_l11_11924


namespace second_train_speed_l11_11902

theorem second_train_speed (v : ℝ) :
  (∃ t : ℝ, 20 * t = v * t + 75 ∧ 20 * t + v * t = 675) → v = 16 :=
by
  sorry

end second_train_speed_l11_11902


namespace hamburgers_served_l11_11659

-- Definitions for the conditions
def hamburgers_made : ℕ := 9
def hamburgers_left_over : ℕ := 6

-- The main statement to prove
theorem hamburgers_served : hamburgers_made - hamburgers_left_over = 3 := by
  sorry

end hamburgers_served_l11_11659


namespace train_crossing_time_l11_11235

noncomputable def length_train : ℝ := 250
noncomputable def length_bridge : ℝ := 150
noncomputable def speed_train_kmh : ℝ := 57.6
noncomputable def speed_train_ms : ℝ := speed_train_kmh * (1000 / 3600)

theorem train_crossing_time : 
  let total_length := length_train + length_bridge 
  let time := total_length / speed_train_ms 
  time = 25 := 
by 
  -- Convert all necessary units and parameters
  let length_train := (250 : ℝ)
  let length_bridge := (150 : ℝ)
  let speed_train_ms := (57.6 * (1000 / 3600) : ℝ)
  
  -- Compute the total length and time
  let total_length := length_train + length_bridge
  let time := total_length / speed_train_ms
  
  -- State the proof
  show time = 25
  { sorry }

end train_crossing_time_l11_11235


namespace work_days_l11_11405

theorem work_days (A B C : ℝ)
  (h1 : A + B = 1 / 20)
  (h2 : B + C = 1 / 30)
  (h3 : A + C = 1 / 30) :
  (1 / (A + B + C)) = 120 / 7 := 
by 
  sorry

end work_days_l11_11405


namespace distance_origin_to_line_l11_11955

theorem distance_origin_to_line : 
  let A := 1
  let B := Real.sqrt 3
  let C := -2
  let x1 := 0
  let y1 := 0
  let distance := |A*x1 + B*y1 + C| / Real.sqrt (A^2 + B^2)
  distance = 1 :=
by 
  let A := 1
  let B := Real.sqrt 3
  let C := -2
  let x1 := 0
  let y1 := 0
  let distance := |A*x1 + B*y1 + C| / Real.sqrt (A^2 + B^2)
  sorry

end distance_origin_to_line_l11_11955


namespace min_ab_square_is_four_l11_11345

noncomputable def min_ab_square : Prop :=
  ∃ a b : ℝ, (a^2 + b^2 = 4 ∧ ∃ x : ℝ, x^4 + a * x^3 + b * x^2 + a * x + 1 = 0)

theorem min_ab_square_is_four : min_ab_square :=
  sorry

end min_ab_square_is_four_l11_11345


namespace angle_measure_l11_11952

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
sorry

end angle_measure_l11_11952


namespace no_solution_l11_11115

theorem no_solution : ¬∃ x : ℝ, x^3 - 8*x^2 + 16*x - 32 / (x - 2) < 0 := by
  sorry

end no_solution_l11_11115


namespace num_initial_pairs_of_shoes_l11_11483

theorem num_initial_pairs_of_shoes (lost_shoes remaining_pairs : ℕ)
  (h1 : lost_shoes = 9)
  (h2 : remaining_pairs = 20) :
  (initial_pairs : ℕ) = 25 :=
sorry

end num_initial_pairs_of_shoes_l11_11483


namespace find_angle_x_l11_11881

def angle_ABC := 124
def angle_BAD := 30
def angle_BDA := 28
def angle_ABD := 180 - angle_ABC
def angle_x := 180 - (angle_BAD + angle_ABD)

theorem find_angle_x : angle_x = 94 :=
by
  repeat { sorry }

end find_angle_x_l11_11881


namespace range_a_l11_11155

theorem range_a (x a : ℝ) (h1 : x^2 - 8 * x - 33 > 0) (h2 : |x - 1| > a) (h3 : a > 0) :
  0 < a ∧ a ≤ 4 :=
by
  sorry

end range_a_l11_11155


namespace num_bad_oranges_l11_11712

theorem num_bad_oranges (G B : ℕ) (hG : G = 24) (ratio : G / B = 3) : B = 8 :=
by
  sorry

end num_bad_oranges_l11_11712


namespace smallest_third_term_GP_l11_11871

theorem smallest_third_term_GP : 
  ∃ d : ℝ, 
    (11 + d) ^ 2 = 9 * (29 + 2 * d) ∧
    min (29 + 2 * 10) (29 + 2 * -14) = 1 :=
by
  sorry

end smallest_third_term_GP_l11_11871


namespace second_player_wins_l11_11981

theorem second_player_wins : 
  ∀ (a b c : ℝ), (a ≠ 0) → 
  (∃ (first_choice: ℝ), ∃ (second_choice: ℝ), 
    ∃ (third_choice: ℝ), 
    ((first_choice ≠ 0) → (b^2 + 4 * first_choice^2 > 0)) ∧ 
    ((first_choice = 0) → (b ≠ 0)) ∧ 
    first_choice * (first_choice * b + a) = 0 ↔ ∃ x : ℝ, a * x^2 + (first_choice + second_choice) * x + third_choice = 0) :=
by sorry

end second_player_wins_l11_11981


namespace renu_suma_work_together_l11_11358

-- Define the time it takes for Renu to do the work by herself
def renu_days : ℕ := 6

-- Define the time it takes for Suma to do the work by herself
def suma_days : ℕ := 12

-- Define the work rate for Renu
def renu_work_rate : ℚ := 1 / renu_days

-- Define the work rate for Suma
def suma_work_rate : ℚ := 1 / suma_days

-- Define the combined work rate
def combined_work_rate : ℚ := renu_work_rate + suma_work_rate

-- Define the days it takes for both Renu and Suma to complete the work together
def days_to_complete_together : ℚ := 1 / combined_work_rate

-- The theorem stating that Renu and Suma can complete the work together in 4 days
theorem renu_suma_work_together : days_to_complete_together = 4 :=
by
  have h1 : renu_days = 6 := rfl
  have h2 : suma_days = 12 := rfl
  have h3 : renu_work_rate = 1 / 6 := by simp [renu_work_rate, h1]
  have h4 : suma_work_rate = 1 / 12 := by simp [suma_work_rate, h2]
  have h5 : combined_work_rate = 1 / 6 + 1 / 12 := by simp [combined_work_rate, h3, h4]
  have h6 : combined_work_rate = 1 / 4 := by norm_num [h5]
  have h7 : days_to_complete_together = 1 / (1 / 4) := by simp [days_to_complete_together, h6]
  have h8 : days_to_complete_together = 4 := by norm_num [h7]
  exact h8

end renu_suma_work_together_l11_11358


namespace time_left_to_use_exerciser_l11_11556

-- Definitions based on the conditions
def total_time : ℕ := 2 * 60  -- Total time in minutes (120 minutes)
def piano_time : ℕ := 30  -- Time spent on piano
def writing_music_time : ℕ := 25  -- Time spent on writing music
def history_time : ℕ := 38  -- Time spent on history

-- The theorem statement that Joan has 27 minutes left
theorem time_left_to_use_exerciser : 
  total_time - (piano_time + writing_music_time + history_time) = 27 :=
by {
  sorry
}

end time_left_to_use_exerciser_l11_11556


namespace circle_through_ABC_l11_11574

-- Define points A, B, and C
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (3, 0)
def C : ℝ × ℝ := (1, 4)

-- Define the circle equation components to be proved
def circle_eqn (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 3*y - 3 = 0

-- The theorem statement that we need to prove
theorem circle_through_ABC : 
  ∃ (D E F : ℝ), (∀ x y, (x, y) = A ∨ (x, y) = B ∨ (x, y) = C → x^2 + y^2 + D*x + E*y + F = 0) 
  → circle_eqn x y :=
sorry

end circle_through_ABC_l11_11574


namespace probability_of_non_defective_pens_l11_11063

-- Define the number of total pens, defective pens, and pens to be selected
def total_pens : ℕ := 15
def defective_pens : ℕ := 5
def selected_pens : ℕ := 3

-- Define the number of non-defective pens
def non_defective_pens : ℕ := total_pens - defective_pens

-- Define the combination function
def combination (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Define the total ways to choose 3 pens from 15 pens
def total_ways : ℕ := combination total_pens selected_pens

-- Define the ways to choose 3 non-defective pens from the non-defective pens
def non_defective_ways : ℕ := combination non_defective_pens selected_pens

-- Define the probability
def probability : ℚ := non_defective_ways / total_ways

-- Statement we need to prove
theorem probability_of_non_defective_pens : probability = 120 / 455 := by
  -- Proof to be completed
  sorry

end probability_of_non_defective_pens_l11_11063


namespace sum_of_coordinates_A_l11_11178

-- Define the problem settings and the required conditions
theorem sum_of_coordinates_A (a b : ℝ) (A B C : ℝ × ℝ) :
  -- Point B lies on the Ox axis
  B.snd = 0 →
  -- Point C lies on the Oy axis
  C.fst = 0 →
  -- Equations of lines given in some order
  (A.snd = a * A.fst + 4 ∧ C.snd = 2 * C.fst + b ∧ B.snd = (a / 2) * B.fst + 8) ∨ 
  (B.snd = a * A.fst + 4 ∧ A.snd = 2 * C.fst + b ∧ C.snd = (a / 2) * B.fst + 8) ∨
  (C.snd = a * A.fst + 4 ∧ B.snd = 2 * C.fst + b ∧ A.snd = (a / 2) * B.fst + 8) →
  -- Prove the sum of the coordinates of point A
  A.fst + A.snd = 13 ∨ A.fst + A.snd = 20 :=
by
  sorry

end sum_of_coordinates_A_l11_11178


namespace whale_sixth_hour_consumption_l11_11670

-- Definitions based on the given conditions
def consumption (x : ℕ) (hour : ℕ) : ℕ := x + 3 * (hour - 1)

def total_consumption (x : ℕ) : ℕ := 
  (consumption x 1) + (consumption x 2) + (consumption x 3) +
  (consumption x 4) + (consumption x 5) + (consumption x 6) + 
  (consumption x 7) + (consumption x 8) + (consumption x 9)

-- Given problem translated to Lean
theorem whale_sixth_hour_consumption (x : ℕ) (h1 : total_consumption x = 270) :
  consumption x 6 = 33 :=
sorry

end whale_sixth_hour_consumption_l11_11670


namespace marshmallow_challenge_l11_11197

noncomputable def haley := 8
noncomputable def michael := 3 * haley
noncomputable def brandon := (1 / 2) * michael
noncomputable def sofia := 2 * (haley + brandon)
noncomputable def total := haley + michael + brandon + sofia

theorem marshmallow_challenge : total = 84 :=
by
  sorry

end marshmallow_challenge_l11_11197


namespace train_length_l11_11849

theorem train_length (speed_km_hr : ℝ) (time_sec : ℝ) (length_m : ℝ) 
  (h1 : speed_km_hr = 90) 
  (h2 : time_sec = 11) 
  (h3 : length_m = 275) :
  length_m = (speed_km_hr * 1000 / 3600) * time_sec :=
sorry

end train_length_l11_11849


namespace rectangle_area_l11_11489

-- Definitions
def perimeter (l w : ℝ) : ℝ := 2 * (l + w)
def length (w : ℝ) : ℝ := 2 * w
def area (l w : ℝ) : ℝ := l * w

-- Main Statement
theorem rectangle_area (w l : ℝ) (h_p : perimeter l w = 120) (h_l : l = length w) :
  area l w = 800 :=
by
  sorry

end rectangle_area_l11_11489


namespace graph_is_point_l11_11241

theorem graph_is_point : ∀ x y : ℝ, x^2 + 3 * y^2 - 4 * x - 6 * y + 7 = 0 ↔ (x = 2 ∧ y = 1) :=
by
  sorry

end graph_is_point_l11_11241


namespace stone_statue_cost_l11_11934

theorem stone_statue_cost :
  ∃ S : Real, 
    let total_earnings := 10 * S + 20 * 5
    let earnings_after_taxes := 0.9 * total_earnings
    earnings_after_taxes = 270 ∧ S = 20 :=
sorry

end stone_statue_cost_l11_11934


namespace square_division_rectangles_l11_11251

theorem square_division_rectangles (k l : ℕ) (h_square : exists s : ℝ, 0 < s) 
(segment_division : ∀ (p q : ℝ), exists r : ℕ, r = s * k ∧ r = s * l) :
  ∃ n : ℕ, n = k * l :=
sorry

end square_division_rectangles_l11_11251


namespace route_comparison_l11_11106

-- Definitions
def distance (P Z C : Type) : Type := ℝ

variables {P Z C : Type} -- P: Park, Z: Zoo, C: Circus
variables (x y C : ℝ)     -- x: direct distance from Park to Zoo, y: direct distance from Circus to Zoo, C: total circumference

-- Conditions
axiom h1 : x + 3 * x = C -- distance from Park to Zoo via Circus is three times longer than not via Circus
axiom h2 : y = (C - x) / 2 -- distance from Circus to Zoo directly is y
axiom h3 : 2 * y = C - x -- distance from Circus to Zoo via Park is twice as short as not via Park

-- Proof statement
theorem route_comparison (P Z C : Type) (x y C : ℝ) (h1 : x + 3 * x = C) (h2 : y = (C - x) / 2) (h3 : 2 * y = C - x) :
  let direct_route := x
  let via_zoo_route := 3 * x - x
  via_zoo_route = 11 * direct_route := 
sorry

end route_comparison_l11_11106


namespace find_2nd_month_sales_l11_11064

def sales_of_1st_month : ℝ := 2500
def sales_of_3rd_month : ℝ := 9855
def sales_of_4th_month : ℝ := 7230
def sales_of_5th_month : ℝ := 7000
def sales_of_6th_month : ℝ := 11915
def average_sales : ℝ := 7500
def months : ℕ := 6
def total_required_sales : ℝ := average_sales * months
def total_known_sales : ℝ := sales_of_1st_month + sales_of_3rd_month + sales_of_4th_month + sales_of_5th_month + sales_of_6th_month

theorem find_2nd_month_sales : 
  ∃ (sales_of_2nd_month : ℝ), total_required_sales = sales_of_1st_month + sales_of_2nd_month + sales_of_3rd_month + sales_of_4th_month + sales_of_5th_month + sales_of_6th_month ∧ sales_of_2nd_month = 10500 := by
  sorry

end find_2nd_month_sales_l11_11064


namespace fraction_not_on_time_l11_11183

theorem fraction_not_on_time (n : ℕ) (h1 : ∃ (k : ℕ), 3 * k = 5 * n) 
(h2 : ∃ (k : ℕ), 4 * k = 5 * m) 
(h3 : ∃ (k : ℕ), 5 * k = 6 * f) 
(h4 : m + f = n) 
(h5 : r = rm + rf) 
(h6 : rm = 4/5 * m) 
(h7 : rf = 5/6 * f) :
  (not_on_time : ℚ) = 1/5 := 
by
  sorry

end fraction_not_on_time_l11_11183


namespace sequence_formula_l11_11240

theorem sequence_formula (a : ℕ → ℤ) (h1 : a 1 = 1)
  (h2 : ∀ n: ℕ, a (n + 1) = 2 * a n + n * (1 + 2^n)) :
  ∀ n : ℕ, a n = 2^(n - 2) * (n^2 - n + 6) - n - 1 :=
by intro n; sorry

end sequence_formula_l11_11240


namespace pow_modulus_l11_11456

theorem pow_modulus : (5 ^ 2023) % 11 = 3 := by
  sorry

end pow_modulus_l11_11456


namespace solution_correct_l11_11866

noncomputable def satisfies_conditions (f : ℤ → ℝ) : Prop :=
  (f 1 = 5 / 2) ∧ (f 0 ≠ 0) ∧ (∀ m n : ℤ, f m * f n = f (m + n) + f (m - n))

theorem solution_correct (f : ℤ → ℝ) :
  satisfies_conditions f → ∀ n : ℤ, f n = 2^n + (1/2)^n :=
by sorry

end solution_correct_l11_11866


namespace sum_of_lengths_of_edges_l11_11173

theorem sum_of_lengths_of_edges (s h : ℝ) 
(volume_eq : s^2 * h = 576) 
(surface_area_eq : 4 * s * h = 384) : 
8 * s + 4 * h = 112 := 
by
  sorry

end sum_of_lengths_of_edges_l11_11173


namespace xy_sum_l11_11445

-- Define the problem conditions
variable (x y : ℚ)
variable (h1 : 1 / x + 1 / y = 4)
variable (h2 : 1 / x - 1 / y = -8)

-- Define the theorem to prove
theorem xy_sum : x + y = -1 / 3 := by
  sorry

end xy_sum_l11_11445


namespace number_of_men_in_first_group_l11_11066

-- Define the conditions
def condition1 (M : ℕ) : Prop := M * 80 = 20 * 40

-- State the main theorem to be proved
theorem number_of_men_in_first_group (M : ℕ) (h : condition1 M) : M = 10 := by
  sorry

end number_of_men_in_first_group_l11_11066


namespace minimum_value_of_f_l11_11839

noncomputable def f (x : ℝ) : ℝ :=
  x - 1 - (Real.log x) / x

theorem minimum_value_of_f : (∀ x > 0, f x ≥ 0) ∧ (∃ x > 0, f x = 0) :=
by
  sorry

end minimum_value_of_f_l11_11839


namespace sequence_term_10_l11_11081

theorem sequence_term_10 : ∃ n : ℕ, (1 / (n * (n + 2)) = 1 / 120) ∧ n = 10 := by
  sorry

end sequence_term_10_l11_11081


namespace circleII_area_l11_11657

noncomputable def area_of_circle (r : ℝ) : ℝ := Real.pi * r^2

theorem circleII_area (r₁ : ℝ) (h₁ : area_of_circle r₁ = 9) (h₂ : r₂ = 3 * 2 * r₁) : 
  area_of_circle r₂ = 324 :=
by
  sorry

end circleII_area_l11_11657


namespace range_of_a_l11_11633

noncomputable def set_A : Set ℝ := { x | x^2 - 3 * x - 10 < 0 }
noncomputable def set_B : Set ℝ := { x | x^2 + 2 * x - 8 > 0 }
def set_C (a : ℝ) : Set ℝ := { x | 2 * a < x ∧ x < a + 3 }

theorem range_of_a (a : ℝ) :
  (A ∩ B) ∩ set_C a = set_C a → 1 ≤ a := 
sorry

end range_of_a_l11_11633


namespace joes_speed_l11_11939

theorem joes_speed (pete_speed : ℝ) (joe_speed : ℝ) (time_minutes : ℝ) (distance : ℝ) (h1 : joe_speed = 2 * pete_speed) (h2 : time_minutes = 40) (h3 : distance = 16) : joe_speed = 16 :=
by
  sorry

end joes_speed_l11_11939


namespace gingerbread_to_bagels_l11_11316

theorem gingerbread_to_bagels (gingerbread drying_rings bagels : ℕ) 
  (h1 : gingerbread = 1 → drying_rings = 6) 
  (h2 : drying_rings = 9 → bagels = 4) 
  (h3 : gingerbread = 3) : bagels = 8 :=
by
  sorry

end gingerbread_to_bagels_l11_11316


namespace find_n_l11_11331

theorem find_n (n : ℤ) (h₁ : 50 ≤ n ∧ n ≤ 120)
               (h₂ : n % 8 = 0)
               (h₃ : n % 12 = 4)
               (h₄ : n % 7 = 4) : 
  n = 88 :=
sorry

end find_n_l11_11331


namespace min_value_of_M_l11_11814

noncomputable def f (p q x : ℝ) : ℝ := x^2 + p * x + q

theorem min_value_of_M (p q M : ℝ) :
  (M = max (|f p q 1|) (max (|f p q (-1)|) (|f p q 0|))) →
  (0 > f p q 1 → 0 > f p q (-1) → 0 > f p q 0 → M = 1 / 2) :=
sorry

end min_value_of_M_l11_11814


namespace find_number_l11_11138

theorem find_number (n : ℕ) (h : n / 3 = 10) : n = 30 := by
  sorry

end find_number_l11_11138


namespace pure_alcohol_addition_l11_11923

theorem pure_alcohol_addition (x : ℝ) (h1 : 3 / 10 * 10 = 3)
    (h2 : 60 / 100 * (10 + x) = (3 + x) ) : x = 7.5 :=
sorry

end pure_alcohol_addition_l11_11923


namespace problem_solution_l11_11437

theorem problem_solution (a b : ℝ) (h1 : a^3 - 15 * a^2 + 25 * a - 75 = 0) (h2 : 8 * b^3 - 60 * b^2 - 310 * b + 2675 = 0) :
  a + b = 15 / 2 :=
sorry

end problem_solution_l11_11437


namespace abc_inequality_l11_11045

theorem abc_inequality (a b c : ℝ) : a^2 + b^2 + c^2 ≥ ab + ac + bc :=
by
  sorry

end abc_inequality_l11_11045


namespace increasing_C_l11_11223

theorem increasing_C (e R r : ℝ) (n : ℕ) (h₁ : 0 < e) (h₂ : 0 < R) (h₃ : 0 < r) (h₄ : 0 < n) :
    ∀ n1 n2 : ℕ, n1 < n2 → (e^2 * n1) / (R + n1 * r) < (e^2 * n2) / (R + n2 * r) :=
by
  sorry

end increasing_C_l11_11223


namespace gym_membership_cost_l11_11818

theorem gym_membership_cost 
    (cheap_monthly_fee : ℕ := 10)
    (cheap_signup_fee : ℕ := 50)
    (expensive_monthly_multiplier : ℕ := 3)
    (months_in_year : ℕ := 12)
    (expensive_signup_multiplier : ℕ := 4) :
    let cheap_gym_cost := cheap_monthly_fee * months_in_year + cheap_signup_fee
    let expensive_monthly_fee := cheap_monthly_fee * expensive_monthly_multiplier
    let expensive_gym_cost := expensive_monthly_fee * months_in_year + expensive_monthly_fee * expensive_signup_multiplier
    let total_cost := cheap_gym_cost + expensive_gym_cost
    total_cost = 650 :=
by
  sorry -- Proof is omitted because the focus is on the statement equivalency.

end gym_membership_cost_l11_11818


namespace ratio_of_original_to_reversed_l11_11205

def original_number : ℕ := 21
def reversed_number : ℕ := 12

theorem ratio_of_original_to_reversed : 
  (original_number : ℚ) / (reversed_number : ℚ) = 7 / 4 := by
  sorry

end ratio_of_original_to_reversed_l11_11205


namespace expected_value_eight_sided_die_win_l11_11037

/-- The expected value of winning with a fair 8-sided die, where the win is \( n^3 \) dollars if \( n \) is rolled, is 162 dollars. -/
theorem expected_value_eight_sided_die_win :
  (1 / 8) * (1^3) + (1 / 8) * (2^3) + (1 / 8) * (3^3) + (1 / 8) * (4^3) +
  (1 / 8) * (5^3) + (1 / 8) * (6^3) + (1 / 8) * (7^3) + (1 / 8) * (8^3) = 162 := 
by
  -- Simplification and calculation here
  sorry

end expected_value_eight_sided_die_win_l11_11037


namespace age_problem_l11_11906

variable (A B : ℕ)

theorem age_problem (h1 : A + 10 = 2 * (B - 10)) (h2 : A = B + 5) : B = 35 := by
  sorry

end age_problem_l11_11906


namespace MountainRidgeAcademy_l11_11007

theorem MountainRidgeAcademy (j s : ℕ) 
  (h1 : 3/4 * j = 1/2 * s) : s = 3/2 * j := 
by 
  sorry

end MountainRidgeAcademy_l11_11007


namespace contrapositive_abc_l11_11201

theorem contrapositive_abc (a b c : ℝ) : (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) → (abc ≠ 0) := 
sorry

end contrapositive_abc_l11_11201


namespace cricketer_average_score_l11_11012

variable {A : ℤ} -- A represents the average score after 18 innings

theorem cricketer_average_score
  (h1 : (19 * (A + 4) = 18 * A + 98)) :
  A + 4 = 26 := by
  sorry

end cricketer_average_score_l11_11012


namespace lily_calculation_l11_11330

theorem lily_calculation (a b c : ℝ) (h1 : a - 2 * b - 3 * c = 2) (h2 : a - 2 * (b - 3 * c) = 14) :
  a - 2 * b = 6 :=
by
  sorry

end lily_calculation_l11_11330


namespace train_crossing_time_l11_11989

noncomputable def train_length : ℝ := 385
noncomputable def train_speed_kmph : ℝ := 90
noncomputable def bridge_length : ℝ := 1250

noncomputable def convert_speed_to_mps (speed_kmph : ℝ) : ℝ :=
  speed_kmph * (1000 / 3600)

noncomputable def time_to_cross_bridge (train_length bridge_length train_speed_kmph : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let speed_mps := convert_speed_to_mps train_speed_kmph
  total_distance / speed_mps

theorem train_crossing_time :
  time_to_cross_bridge train_length bridge_length train_speed_kmph = 65.4 :=
by
  sorry

end train_crossing_time_l11_11989


namespace grandfather_older_than_xiaoming_dad_age_when_twenty_times_xiaoming_l11_11548

-- Definition of the conditions
def grandfather_age (gm_age dad_age : ℕ) := gm_age = 2 * dad_age
def dad_age_eight_times_xiaoming (dad_age xm_age : ℕ) := dad_age = 8 * xm_age
def grandfather_age_61 (gm_age : ℕ) := gm_age = 61
def twenty_times_xiaoming (gm_age xm_age : ℕ) := gm_age = 20 * xm_age

-- Question 1: Proof that Grandpa is 57 years older than Xiaoming 
theorem grandfather_older_than_xiaoming (gm_age dad_age xm_age : ℕ) 
  (h1 : grandfather_age gm_age dad_age) (h2 : dad_age_eight_times_xiaoming dad_age xm_age)
  (h3 : grandfather_age_61 gm_age)
  : gm_age - xm_age = 57 := 
sorry

-- Question 2: Proof that Dad is 31 years old when Grandpa's age is twenty times Xiaoming's age
theorem dad_age_when_twenty_times_xiaoming (gm_age dad_age xm_age : ℕ) 
  (h1 : twenty_times_xiaoming gm_age xm_age)
  (hm : grandfather_age gm_age dad_age)
  : dad_age = 31 :=
sorry

end grandfather_older_than_xiaoming_dad_age_when_twenty_times_xiaoming_l11_11548


namespace solve_for_d_l11_11210

theorem solve_for_d (n k c d : ℝ) (h₁ : n = 2 * k * c * d / (c + d)) (h₂ : 2 * k * c ≠ n) :
  d = n * c / (2 * k * c - n) :=
by
  sorry

end solve_for_d_l11_11210


namespace brandon_skittles_loss_l11_11407

theorem brandon_skittles_loss (original final : ℕ) (H1 : original = 96) (H2 : final = 87) : original - final = 9 :=
by sorry

end brandon_skittles_loss_l11_11407


namespace nadine_total_cleaning_time_l11_11857

-- Conditions
def time_hosing_off := 10 -- minutes
def shampoos := 3
def time_per_shampoo := 15 -- minutes

-- Total cleaning time calculation
def total_cleaning_time := time_hosing_off + (shampoos * time_per_shampoo)

-- Theorem statement
theorem nadine_total_cleaning_time : total_cleaning_time = 55 := by
  sorry

end nadine_total_cleaning_time_l11_11857


namespace Johnson_farm_budget_l11_11912

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

end Johnson_farm_budget_l11_11912


namespace complex_expression_identity_l11_11585

open Complex

theorem complex_expression_identity
  (x y : ℂ)
  (hx : x ≠ 0)
  (hy : y ≠ 0)
  (hxy : x^2 + x * y + y^2 = 0) :
  (x / (x + y))^1990 + (y / (x + y))^1990 = -1 :=
by
  sorry

end complex_expression_identity_l11_11585


namespace arithmetic_sequence_ninth_term_l11_11480

-- Define the terms in the arithmetic sequence
def sequence_term (a d : ℚ) (n : ℕ) : ℚ :=
  a + (n - 1) * d

-- Given conditions
def a1 : ℚ := 2 / 3
def a17 : ℚ := 5 / 6
def d : ℚ := 1 / 96 -- Calculated common difference

-- Prove the ninth term is 3/4
theorem arithmetic_sequence_ninth_term :
  sequence_term a1 d 9 = 3 / 4 :=
sorry

end arithmetic_sequence_ninth_term_l11_11480


namespace problem_statement_l11_11199

theorem problem_statement : 3.5 * 2.5 + 6.5 * 2.5 = 25 := by
  sorry

end problem_statement_l11_11199


namespace minimum_value_of_xy_l11_11202

theorem minimum_value_of_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 * x + y + 6 = x * y) : 
  18 ≤ x * y :=
sorry

end minimum_value_of_xy_l11_11202


namespace birds_on_fence_l11_11470

theorem birds_on_fence (B : ℕ) : ∃ B, (∃ S, S = 6 ∧ S = (B + 3) + 1) → B = 2 :=
by
  sorry

end birds_on_fence_l11_11470


namespace parabola_equation_max_slope_OQ_l11_11895

-- Definition of the problem for part (1)
theorem parabola_equation (p : ℝ) (hp : p = 2) : (∀ x y : ℝ, y^2 = 2 * p * x ↔ y^2 = 4 * x) :=
by {
  sorry
}

-- Definition of the problem for part (2)
theorem max_slope_OQ (m n : ℝ) (hp : y^2 = 4 * x)
  (h_relate : ∀ P Q F : (ℝ × ℝ), P.1 * Q.1 + P.2 * Q.2 = 9 * (Q.1 - F.1) * (Q.2 - F.2))
  : (∀ Q : (ℝ × ℝ), max (Q.2 / Q.1) = 1/3) :=
by {
  sorry
}

end parabola_equation_max_slope_OQ_l11_11895


namespace no_real_solutions_to_equation_l11_11959

theorem no_real_solutions_to_equation :
  ¬ ∃ y : ℝ, (3 * y - 4)^2 + 4 = -(y + 3) :=
by
  sorry

end no_real_solutions_to_equation_l11_11959


namespace min_value_of_expression_l11_11452

theorem min_value_of_expression (x y : ℝ) : 
  ∃ x y, 2 * x^2 + 3 * y^2 - 8 * x + 12 * y + 40 = 20 := 
sorry

end min_value_of_expression_l11_11452


namespace problem_l11_11979

theorem problem (a : ℕ) (h1 : a = 444) : (444 ^ 444) % 13 = 1 :=
by
  have h444 : 444 % 13 = 3 := by sorry
  have h3_pow3 : 3 ^ 3 % 13 = 1 := by sorry
  sorry

end problem_l11_11979


namespace people_from_second_row_joined_l11_11481

theorem people_from_second_row_joined
  (initial_first_row : ℕ) (initial_second_row : ℕ) (initial_third_row : ℕ) (people_waded : ℕ) (remaining_people : ℕ)
  (H1 : initial_first_row = 24)
  (H2 : initial_second_row = 20)
  (H3 : initial_third_row = 18)
  (H4 : people_waded = 3)
  (H5 : remaining_people = 54) :
  initial_second_row - (initial_first_row + initial_second_row + initial_third_row - initial_first_row - people_waded - remaining_people) = 5 :=
by
  sorry

end people_from_second_row_joined_l11_11481


namespace profit_difference_l11_11532

theorem profit_difference
  (p1 p2 : ℝ)
  (h1 : p1 > p2)
  (h2 : p1 + p2 = 3635000)
  (h3 : p2 = 442500) :
  p1 - p2 = 2750000 :=
by 
  sorry

end profit_difference_l11_11532


namespace binary_operation_l11_11554

def b11001 := 25  -- binary 11001 is 25 in decimal
def b1101 := 13   -- binary 1101 is 13 in decimal
def b101 := 5     -- binary 101 is 5 in decimal
def b100111010 := 314 -- binary 100111010 is 314 in decimal

theorem binary_operation : (b11001 * b1101 - b101) = b100111010 := by
  -- provide implementation details to prove the theorem
  sorry

end binary_operation_l11_11554


namespace avg_age_of_new_persons_l11_11606

-- We define the given conditions
def initial_persons : ℕ := 12
def initial_avg_age : ℝ := 16
def new_persons : ℕ := 12
def new_avg_age : ℝ := 15.5

-- Define the total initial age
def total_initial_age : ℝ := initial_persons * initial_avg_age

-- Define the total number of persons after new persons join
def total_persons_after_join : ℕ := initial_persons + new_persons

-- Define the total age after new persons join
def total_age_after_join : ℝ := total_persons_after_join * new_avg_age

-- We wish to prove that the average age of the new persons who joined is 15
theorem avg_age_of_new_persons : 
  (total_initial_age + new_persons * 15) = total_age_after_join :=
sorry

end avg_age_of_new_persons_l11_11606


namespace find_line_equation_l11_11536

theorem find_line_equation :
  ∃ (m : ℝ), ∃ (b : ℝ), (∀ x y : ℝ,
  (x + 3 * y - 2 = 0 → y = -1/3 * x + 2/3) ∧
  (x = 3 → y = 0) →
  y = m * x + b) ∧
  (m = 3 ∧ b = -9) :=
  sorry

end find_line_equation_l11_11536


namespace profit_share_of_B_l11_11129

-- Defining the initial investments
def a : ℕ := 8000
def b : ℕ := 10000
def c : ℕ := 12000

-- Given difference between profit shares of A and C
def diff_AC : ℕ := 680

-- Define total profit P
noncomputable def P : ℕ := (diff_AC * 15) / 2

-- Calculate B's profit share
noncomputable def B_share : ℕ := (5 * P) / 15

-- The theorem stating B's profit share
theorem profit_share_of_B : B_share = 1700 :=
by sorry

end profit_share_of_B_l11_11129


namespace characterize_set_A_l11_11504

open Int

noncomputable def A : Set ℤ := { x | x^2 - 3 * x - 4 < 0 }

theorem characterize_set_A : A = {0, 1, 2, 3} :=
by
  sorry

end characterize_set_A_l11_11504


namespace find_triplets_of_real_numbers_l11_11673

theorem find_triplets_of_real_numbers (x y z : ℝ) :
  (x^2 + y^2 + 25 * z^2 = 6 * x * z + 8 * y * z) ∧ 
  (3 * x^2 + 2 * y^2 + z^2 = 240) → 
  (x = 6 ∧ y = 8 ∧ z = 2) ∨ (x = -6 ∧ y = -8 ∧ z = -2) := 
sorry

end find_triplets_of_real_numbers_l11_11673


namespace triangle_perimeter_l11_11660

theorem triangle_perimeter (r : ℝ) (A B C P Q R S T : ℝ)
  (triangle_isosceles : A = C)
  (circle_tangent : P = A ∧ Q = B ∧ R = B ∧ S = C ∧ T = C)
  (center_dist : P + Q = 2 ∧ Q + R = 2 ∧ R + S = 2 ∧ S + T = 2) :
  2 * (A + B + C) = 6 := by
  sorry

end triangle_perimeter_l11_11660


namespace intersection_correct_l11_11769

def A : Set ℕ := {1, 2, 3}

def B : Set ℕ := { y | ∃ x ∈ A, y = 2 * x - 1 }

def intersection : Set ℕ := { x | x ∈ A ∧ x ∈ B }

theorem intersection_correct : intersection = {1, 3} := by
  sorry

end intersection_correct_l11_11769


namespace fifteenth_number_with_digit_sum_15_is_294_l11_11948

def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def numbers_with_digit_sum (s : ℕ) : List ℕ :=
  List.filter (λ n => digit_sum n = s) (List.range (10 ^ 3)) -- Assume a maximum of 3-digit numbers

def fifteenth_number_with_digit_sum (s : ℕ) : ℕ :=
  (numbers_with_digit_sum s).get! 14 -- Get the 15th element (0-indexed)

theorem fifteenth_number_with_digit_sum_15_is_294 : fifteenth_number_with_digit_sum 15 = 294 :=
by
  sorry -- Proof is omitted

end fifteenth_number_with_digit_sum_15_is_294_l11_11948


namespace squares_have_consecutive_digits_generalized_squares_have_many_consecutive_digits_l11_11581

theorem squares_have_consecutive_digits (n : ℕ) (h : ∃ j : ℕ, n = 33330 + j ∧ j < 10) :
    ∃ (a b : ℕ), n ^ 2 / 10 ^ a % 10 = n ^ 2 / 10 ^ (a + 1) % 10 :=
by
  sorry

theorem generalized_squares_have_many_consecutive_digits (k : ℕ) (n : ℕ)
  (h1 : k ≥ 4)
  (h2 : ∃ j : ℕ, n = 33333 * 10 ^ (k - 4) + j ∧ j < 10 ^ (k - 4)) :
    ∃ m, ∃ l : ℕ, ∀ i < m, n^2 / 10 ^ (l + i) % 10 = n^2 / 10 ^ l % 10 :=
by
  sorry

end squares_have_consecutive_digits_generalized_squares_have_many_consecutive_digits_l11_11581


namespace square_inequality_not_sufficient_nor_necessary_for_cube_inequality_l11_11506

variable {a b : ℝ}

theorem square_inequality_not_sufficient_nor_necessary_for_cube_inequality (a b : ℝ) :
  (a^2 > b^2) ↔ (a^3 > b^3) = false :=
sorry

end square_inequality_not_sufficient_nor_necessary_for_cube_inequality_l11_11506


namespace library_visitors_on_sunday_l11_11927

def avg_visitors_sundays (S : ℕ) : Prop :=
  let total_days := 30
  let avg_other_days := 240
  let avg_total := 285
  let sundays := 5
  let other_days := total_days - sundays
  (S * sundays) + (avg_other_days * other_days) = avg_total * total_days

theorem library_visitors_on_sunday (S : ℕ) (h : avg_visitors_sundays S) : S = 510 :=
by
  sorry

end library_visitors_on_sunday_l11_11927


namespace find_x1_l11_11551

theorem find_x1 (x1 x2 x3 x4 : ℝ) 
  (h1 : 0 ≤ x4 ∧ x4 ≤ x3 ∧ x3 ≤ x2 ∧ x2 ≤ x1 ∧ x1 ≤ 1)
  (h2 : (1 - x1)^2 + (x1 - x2)^2 + (x2 - x3)^2 + (x3 - x4)^2 + x4^2 = 1 / 5) :
  x1 = 4 / 5 := 
sorry

end find_x1_l11_11551


namespace remainder_sum_div_7_l11_11604

theorem remainder_sum_div_7 :
  (8145 + 8146 + 8147 + 8148 + 8149) % 7 = 4 :=
by
  sorry

end remainder_sum_div_7_l11_11604


namespace coach_class_seats_l11_11109

variable (F C : ℕ)

-- Define the conditions
def totalSeats := F + C = 387
def coachSeats := C = 4 * F + 2

-- State the theorem
theorem coach_class_seats : totalSeats F C → coachSeats F C → C = 310 :=
by sorry

end coach_class_seats_l11_11109


namespace parallel_lines_slope_eq_l11_11217

theorem parallel_lines_slope_eq (k : ℝ) : 
  (∀ x : ℝ, k * x - 1 = 3 * x) → k = 3 :=
by sorry

end parallel_lines_slope_eq_l11_11217


namespace problem_ab_plus_a_plus_b_l11_11938

noncomputable def polynomial := fun x : ℝ => x^4 - 6 * x - 2

theorem problem_ab_plus_a_plus_b :
  ∀ (a b : ℝ), polynomial a = 0 → polynomial b = 0 → (a * b + a + b) = 4 :=
by
  intros a b ha hb
  sorry

end problem_ab_plus_a_plus_b_l11_11938


namespace expected_turns_formula_l11_11584

noncomputable def expected_turns (n : ℕ) : ℝ :=
  n + 0.5 - (n - 0.5) * (1 / (Real.sqrt (Real.pi * (n - 1))))

theorem expected_turns_formula (n : ℕ) (h : n > 1) :
  expected_turns n = n + 0.5 - (n - 0.5) * (1 / (Real.sqrt (Real.pi * (n - 1)))) :=
by
  unfold expected_turns
  sorry

end expected_turns_formula_l11_11584


namespace kyle_caught_fish_l11_11827

def total_fish := 36
def fish_carla := 8
def fish_total := total_fish - fish_carla

-- kelle and tasha same number of fish means they equally divide the total fish left after deducting carla's
def fish_each_kt := fish_total / 2

theorem kyle_caught_fish :
  fish_each_kt = 14 :=
by
  -- Placeholder for the proof
  sorry

end kyle_caught_fish_l11_11827


namespace triangle_inequality_shortest_side_l11_11051

theorem triangle_inequality_shortest_side (a b c : ℝ) (h_triangle: a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a) 
  (h_inequality : a^2 + b^2 > 5 * c^2) : c ≤ a ∧ c ≤ b :=
sorry

end triangle_inequality_shortest_side_l11_11051


namespace find_base_solve_inequality_case1_solve_inequality_case2_l11_11734

noncomputable def log_function (a x : ℝ) : ℝ := Real.log x / Real.log a

theorem find_base (a : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1) : log_function a 8 = 3 → a = 2 :=
by sorry

theorem solve_inequality_case1 (a : ℝ) (h₁ : 1 < a) :
  ∀ x : ℝ, log_function a x ≤ log_function a (2 - 3 * x) → 0 < x ∧ x ≤ 1 / 2 :=
by sorry

theorem solve_inequality_case2 (a : ℝ) (h₁ : 0 < a) (h₂ : a < 1) :
  ∀ x : ℝ, log_function a x ≤ log_function a (2 - 3 * x) → 1 / 2 ≤ x ∧ x < 2 / 3 :=
by sorry

end find_base_solve_inequality_case1_solve_inequality_case2_l11_11734


namespace john_days_ran_l11_11270

theorem john_days_ran 
  (total_distance : ℕ) (daily_distance : ℕ) 
  (h1 : total_distance = 10200) (h2 : daily_distance = 1700) :
  total_distance / daily_distance = 6 :=
by
  sorry

end john_days_ran_l11_11270


namespace rain_probability_tel_aviv_l11_11364

open scoped Classical

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

noncomputable def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (binomial_coefficient n k) * (p^k) * ((1 - p)^(n - k))

theorem rain_probability_tel_aviv :
  binomial_probability 6 4 0.5 = 0.234375 :=
by 
  sorry

end rain_probability_tel_aviv_l11_11364


namespace find_central_angle_l11_11748

theorem find_central_angle
  (θ r : ℝ)
  (h1 : r * θ = 2 * π)
  (h2 : (1 / 2) * r^2 * θ = 3 * π) :
  θ = 2 * π / 3 := 
sorry

end find_central_angle_l11_11748


namespace sum_primes_less_than_20_l11_11450

theorem sum_primes_less_than_20 : 
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19) = 77 :=
by sorry

end sum_primes_less_than_20_l11_11450


namespace math_and_science_students_l11_11719

theorem math_and_science_students (x y : ℕ) 
  (h1 : x + y + 2 = 30)
  (h2 : y = 3 * x + 4) :
  y - 2 = 20 :=
by {
  sorry
}

end math_and_science_students_l11_11719


namespace triangle_inequality_l11_11036

theorem triangle_inequality (a b c : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) :
  a / (b + c - a) + b / (c + a - b) + c / (a + b - c) ≥ 3 :=
sorry

end triangle_inequality_l11_11036


namespace contrapositive_even_sum_l11_11589

theorem contrapositive_even_sum (a b : ℕ) :
  (¬(a % 2 = 0 ∧ b % 2 = 0) → ¬(a + b) % 2 = 0) ↔ (¬((a + b) % 2 = 0) → ¬(a % 2 = 0 ∧ b % 2 = 0)) :=
by
  sorry

end contrapositive_even_sum_l11_11589


namespace problem_conditions_l11_11237

variables (a b : ℝ)
open Real

theorem problem_conditions (ha : a < 0) (hb : 0 < b) (hab : a + b > 0) :
  (a / b > -1) ∧ (abs a < abs b) ∧ (1 / a + 1 / b ≤ 0) ∧ ((a - 1) * (b - 1) < 1) := sorry

end problem_conditions_l11_11237


namespace find_numbers_l11_11408

theorem find_numbers :
  ∃ a b : ℕ, a + b = 60 ∧ Nat.gcd a b + Nat.lcm a b = 84 :=
by
  sorry

end find_numbers_l11_11408


namespace chestnut_picking_l11_11868

theorem chestnut_picking 
  (P : ℕ)
  (h1 : 12 + P + (P + 2) = 26) :
  12 / P = 2 :=
sorry

end chestnut_picking_l11_11868


namespace pounds_per_ton_l11_11985

theorem pounds_per_ton (packet_count : ℕ) (packet_weight_pounds : ℚ) (packet_weight_ounces : ℚ) (ounces_per_pound : ℚ) (total_weight_tons : ℚ) (total_weight_pounds : ℚ) :
  packet_count = 1760 →
  packet_weight_pounds = 16 →
  packet_weight_ounces = 4 →
  ounces_per_pound = 16 →
  total_weight_tons = 13 →
  total_weight_pounds = (packet_count * (packet_weight_pounds + (packet_weight_ounces / ounces_per_pound))) →
  total_weight_pounds / total_weight_tons = 2200 :=
by
  intro h1 h2 h3 h4 h5 h6
  sorry

end pounds_per_ton_l11_11985


namespace driving_time_constraint_l11_11214

variable (x y z : ℝ)

theorem driving_time_constraint (h₁ : x > 0) (h₂ : y > 0) (h₃ : z > 0) :
  3 + (60 / x) + (90 / y) + (200 / z) ≤ 10 :=
sorry

end driving_time_constraint_l11_11214


namespace factor_x6_plus_8_l11_11737

theorem factor_x6_plus_8 : (x^2 + 2) ∣ (x^6 + 8) :=
by
  sorry

end factor_x6_plus_8_l11_11737


namespace original_cost_l11_11886

theorem original_cost (SP : ℝ) (C : ℝ) (h1 : SP = 540) (h2 : SP = C + 0.35 * C) : C = 400 :=
by {
  sorry
}

end original_cost_l11_11886


namespace point_in_first_quadrant_l11_11741

theorem point_in_first_quadrant (m : ℝ) (h : m < 0) : 
  (-m > 0) ∧ (-m + 1 > 0) :=
by 
  sorry

end point_in_first_quadrant_l11_11741


namespace range_of_a_l11_11883

theorem range_of_a (a x : ℝ) (p : 0.5 ≤ x ∧ x ≤ 1) (q : (x - a) * (x - a - 1) > 0) :
  (0 ≤ a ∧ a ≤ 0.5) :=
by 
  sorry

end range_of_a_l11_11883


namespace find_third_side_of_triangle_l11_11058

noncomputable def area_triangle_given_sides_angle {a b c : ℝ} (A : ℝ) : Prop :=
  A = 1/2 * a * b * Real.sin c

noncomputable def cosine_law_third_side {a b c : ℝ} (cosα : ℝ) : Prop :=
  c^2 = a^2 + b^2 - 2 * a * b * cosα

theorem find_third_side_of_triangle (a b : ℝ) (Area : ℝ) (h_a : a = 2 * Real.sqrt 2) (h_b : b = 3) (h_Area : Area = 3) :
  ∃ c : ℝ, (c = Real.sqrt 5 ∨ c = Real.sqrt 29) :=
by
  sorry

end find_third_side_of_triangle_l11_11058


namespace floor_of_neg_five_thirds_l11_11153

theorem floor_of_neg_five_thirds : Int.floor (-5/3 : ℝ) = -2 := 
by 
  sorry

end floor_of_neg_five_thirds_l11_11153


namespace time_to_fill_tank_with_two_pipes_simultaneously_l11_11682

def PipeA : ℝ := 30
def PipeB : ℝ := 45

theorem time_to_fill_tank_with_two_pipes_simultaneously :
  let A := 1 / PipeA
  let B := 1 / PipeB
  let combined_rate := A + B
  let time_to_fill_tank := 1 / combined_rate
  time_to_fill_tank = 18 := 
by
  sorry

end time_to_fill_tank_with_two_pipes_simultaneously_l11_11682


namespace solution_set_of_inequality_l11_11662

theorem solution_set_of_inequality :
  {x : ℝ | (x - 3) / x ≥ 0} = {x : ℝ | x < 0 ∨ x ≥ 3} :=
sorry

end solution_set_of_inequality_l11_11662


namespace jon_awake_hours_per_day_l11_11239

def regular_bottle_size : ℕ := 16
def larger_bottle_size : ℕ := 20
def weekly_fluid_intake : ℕ := 728
def larger_bottle_daily_intake : ℕ := 40
def larger_bottle_weekly_intake : ℕ := 280
def regular_bottle_weekly_intake : ℕ := 448
def regular_bottles_per_week : ℕ := 28
def regular_bottles_per_day : ℕ := 4
def hours_per_bottle : ℕ := 4

theorem jon_awake_hours_per_day
  (h1 : jon_drinks_regular_bottle_every_4_hours)
  (h2 : jon_drinks_two_larger_bottles_daily)
  (h3 : jon_drinks_728_ounces_per_week) :
  jon_is_awake_hours_per_day = 16 :=
by
  sorry

def jon_drinks_regular_bottle_every_4_hours : Prop :=
  ∀ hours : ℕ, hours * regular_bottle_size / hours_per_bottle = 1

def jon_drinks_two_larger_bottles_daily : Prop :=
  larger_bottle_size = (regular_bottle_size * 5) / 4 ∧ 
  larger_bottle_daily_intake = 2 * larger_bottle_size

def jon_drinks_728_ounces_per_week : Prop :=
  weekly_fluid_intake = 728

def jon_is_awake_hours_per_day : ℕ :=
  regular_bottles_per_day * hours_per_bottle

end jon_awake_hours_per_day_l11_11239


namespace in_range_p_1_to_100_l11_11229

def p (m n : ℤ) : ℤ :=
  2 * m^2 - 6 * m * n + 5 * n^2

-- Predicate that asserts k is in the range of p
def in_range_p (k : ℤ) : Prop :=
  ∃ m n : ℤ, p m n = k

-- Lean statement for the theorem
theorem in_range_p_1_to_100 :
  {k : ℕ | 1 ≤ k ∧ k ≤ 100 ∧ in_range_p k} = 
  {1, 2, 4, 5, 8, 9, 10, 13, 16, 17, 18, 20, 25, 26, 29, 32, 34, 36, 37, 40, 41, 45, 49, 50, 52, 53, 58, 61, 64, 65, 68, 72, 73, 74, 80, 81, 82, 85, 89, 90, 97, 98, 100} :=
  by
    sorry

end in_range_p_1_to_100_l11_11229


namespace only_A_can_form_triangle_l11_11015

/--
Prove that from the given sets of lengths, only the set {5cm, 8cm, 12cm} can form a valid triangle.

Given:
- A: 5 cm, 8 cm, 12 cm
- B: 2 cm, 3 cm, 6 cm
- C: 3 cm, 3 cm, 6 cm
- D: 4 cm, 7 cm, 11 cm

We need to show that only Set A satisfies the triangle inequality theorem.
-/
theorem only_A_can_form_triangle :
  (∀ (a b c : ℕ), a = 5 ∧ b = 8 ∧ c = 12 → a + b > c ∧ a + c > b ∧ b + c > a) ∧
  (∀ (a b c : ℕ), a = 2 ∧ b = 3 ∧ c = 6 → ¬(a + b > c ∧ a + c > b ∧ b + c > a)) ∧
  (∀ (a b c : ℕ), a = 3 ∧ b = 3 ∧ c = 6 → ¬(a + b > c ∧ a + c > b ∧ b + c > a)) ∧
  (∀ (a b c : ℕ), a = 4 ∧ b = 7 ∧ c = 11 → ¬(a + b > c ∧ a + c > b ∧ b + c > a)) :=
by
  sorry -- Proof to be provided

end only_A_can_form_triangle_l11_11015


namespace parabola_focus_distance_l11_11926

theorem parabola_focus_distance (x y : ℝ) (h1 : y^2 = 4 * x) (h2 : (x - 1)^2 + y^2 = 100) : x = 9 :=
sorry

end parabola_focus_distance_l11_11926


namespace ripe_oranges_l11_11436

theorem ripe_oranges (U : ℕ) (hU : U = 25) (hR : R = U + 19) : R = 44 := by
  sorry

end ripe_oranges_l11_11436


namespace quadratic_points_relationship_l11_11449

theorem quadratic_points_relationship (c y1 y2 y3 : ℝ) 
  (hA : y1 = (-3)^2 + 2*(-3) + c)
  (hB : y2 = (1/2)^2 + 2*(1/2) + c)
  (hC : y3 = 2^2 + 2*2 + c) : y2 < y1 ∧ y1 < y3 := 
sorry

end quadratic_points_relationship_l11_11449


namespace period_length_divisor_l11_11608

theorem period_length_divisor (p d : ℕ) (hp_prime : Nat.Prime p) (hd_period : ∀ n : ℕ, n ≥ 1 → 10^n % p = 1 ↔ n = d) :
  d ∣ (p - 1) :=
sorry

end period_length_divisor_l11_11608


namespace intersection_M_N_l11_11727

def M := { x : ℝ | -1 < x ∧ x < 2 }
def N := { x : ℝ | x ≤ 1 }
def expectedIntersection := { x : ℝ | -1 < x ∧ x ≤ 1 }

theorem intersection_M_N :
  M ∩ N = expectedIntersection :=
by
  sorry

end intersection_M_N_l11_11727


namespace room_width_l11_11797

theorem room_width (w : ℝ) (h1 : 21 > 0) (h2 : 2 > 0) 
  (h3 : (25 * (w + 4) - 21 * w = 148)) : w = 12 :=
by {
  sorry
}

end room_width_l11_11797


namespace inequality_proof_l11_11894

/-- Given a and b are positive and satisfy the inequality ab > 2007a + 2008b,
    prove that a + b > (sqrt 2007 + sqrt 2008)^2 -/
theorem inequality_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a * b > 2007 * a + 2008 * b) :
  a + b > (Real.sqrt 2007 + Real.sqrt 2008) ^ 2 :=
by
  sorry

end inequality_proof_l11_11894


namespace tank_capacity_l11_11225

theorem tank_capacity (C : ℝ) (h : (3 / 4) * C + 9 = (7 / 8) * C) : C = 72 :=
sorry

end tank_capacity_l11_11225


namespace ratio_of_visible_spots_l11_11705

theorem ratio_of_visible_spots (S S1 : ℝ) (h1 : ∀ (fold_type : ℕ), 
  (fold_type = 1 ∨ fold_type = 2 ∨ fold_type = 3) → 
  (if fold_type = 1 ∨ fold_type = 2 then S1 else S) = S1) : S1 / S = 2 / 3 := 
sorry

end ratio_of_visible_spots_l11_11705


namespace train_passes_pole_in_10_seconds_l11_11774

theorem train_passes_pole_in_10_seconds :
  let L := 150 -- length of the train in meters
  let S_kmhr := 54 -- speed in kilometers per hour
  let S_ms := S_kmhr * 1000 / 3600 -- speed in meters per second
  (L / S_ms = 10) := 
by
  sorry

end train_passes_pole_in_10_seconds_l11_11774


namespace B_more_than_C_l11_11622

variables (A B C : ℕ)
noncomputable def total_subscription : ℕ := 50000
noncomputable def total_profit : ℕ := 35000
noncomputable def A_profit : ℕ := 14700
noncomputable def A_subscr : ℕ := B + 4000

theorem B_more_than_C (B_subscr C_subscr : ℕ) (h1 : A_subscr + B_subscr + C_subscr = total_subscription)
    (h2 : 14700 * 50000 = 35000 * A_subscr) :
    B_subscr - C_subscr = 5000 :=
sorry

end B_more_than_C_l11_11622


namespace plane_equation_l11_11807

-- Define the point and the normal vector
def point : ℝ × ℝ × ℝ := (8, -2, 2)
def normal_vector : ℝ × ℝ × ℝ := (8, -2, 2)

-- Define integers A, B, C, D such that the plane equation satisfies the conditions
def A : ℤ := 4
def B : ℤ := -1
def C : ℤ := 1
def D : ℤ := -18

-- Prove the equation of the plane
theorem plane_equation (x y z : ℝ) :
  A * x + B * y + C * z + D = 0 ↔ 4 * x - y + z - 18 = 0 :=
by
  sorry

end plane_equation_l11_11807


namespace range_of_a_l11_11717

theorem range_of_a (a : ℝ) : 
  (∀ x, x ∈ {x | x^2 ≤ 1} ∪ {a} ↔ x ∈ {x | x^2 ≤ 1}) → (-1 ≤ a ∧ a ≤ 1) :=
by
  intro h
  sorry

end range_of_a_l11_11717


namespace polygon_area_l11_11441

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

end polygon_area_l11_11441


namespace max_profit_is_4sqrt6_add_21_l11_11148

noncomputable def profit (x : ℝ) : ℝ :=
  let y1 : ℝ := -2 * (3 - x)^2 + 14 * (3 - x)
  let y2 : ℝ := - (1 / 3) * x^3 + 2 * x^2 + 5 * x
  let F : ℝ := y1 + y2 - 3
  F

theorem max_profit_is_4sqrt6_add_21 : 
  ∃ x : ℝ, 0 ≤ x ∧ x ≤ 3 ∧ profit x = 4 * Real.sqrt 6 + 21 :=
sorry

end max_profit_is_4sqrt6_add_21_l11_11148


namespace track_length_eq_900_l11_11533

/-- 
Bruce and Bhishma are running on a circular track. 
The speed of Bruce is 30 m/s and that of Bhishma is 20 m/s.
They start from the same point at the same time in the same direction.
They meet again for the first time after 90 seconds. 
Prove that the length of the track is 900 meters.
-/
theorem track_length_eq_900 :
  let speed_bruce := 30 -- [m/s]
  let speed_bhishma := 20 -- [m/s]
  let time_meet := 90 -- [s]
  let distance_bruce := speed_bruce * time_meet
  let distance_bhishma := speed_bhishma * time_meet
  let track_length := distance_bruce - distance_bhishma
  track_length = 900 :=
by
  let speed_bruce := 30
  let speed_bhishma := 20
  let time_meet := 90
  let distance_bruce := speed_bruce * time_meet
  let distance_bhishma := speed_bhishma * time_meet
  let track_length := distance_bruce - distance_bhishma
  have : track_length = 900 := by
    sorry
  exact this

end track_length_eq_900_l11_11533


namespace original_price_is_135_l11_11246

-- Problem Statement:
variable (P : ℝ)  -- Let P be the original price of the potion

-- Conditions
axiom potion_cost : (1 / 15) * P = 9

-- Proof Goal
theorem original_price_is_135 : P = 135 :=
by
  sorry

end original_price_is_135_l11_11246


namespace students_in_front_of_Yuna_l11_11900

-- Defining the total number of students
def total_students : ℕ := 25

-- Defining the number of students behind Yuna
def students_behind_Yuna : ℕ := 9

-- Defining Yuna's position from the end of the line
def Yuna_position_from_end : ℕ := students_behind_Yuna + 1

-- Statement to prove the number of students in front of Yuna
theorem students_in_front_of_Yuna : (total_students - Yuna_position_from_end) = 15 := by
  sorry

end students_in_front_of_Yuna_l11_11900


namespace monotonic_decreasing_interval_l11_11122

noncomputable def func (x : ℝ) : ℝ :=
  x * Real.log x

noncomputable def derivative (x : ℝ) : ℝ :=
  Real.log x + 1

theorem monotonic_decreasing_interval :
  { x : ℝ | 0 < x ∧ x < Real.exp (-1) } ⊆ { x : ℝ | derivative x < 0 } :=
by
  sorry

end monotonic_decreasing_interval_l11_11122


namespace minimum_cost_l11_11080

noncomputable def total_cost (x : ℝ) : ℝ :=
  (1800 / (x + 5)) + 0.5 * x

theorem minimum_cost : 
  (∃ x : ℝ, x = 55 ∧ total_cost x = 57.5) :=
  sorry

end minimum_cost_l11_11080


namespace solution1_solution2_l11_11033

open Real

noncomputable def problem1 (a b : ℝ) : Prop :=
a = 2 ∧ b = 2

noncomputable def problem2 (b : ℝ) : Prop :=
b = (2 * (sqrt 3 + sqrt 2)) / 3

theorem solution1 (a b : ℝ) (c : ℝ) (C : ℝ) (area : ℝ)
  (h1 : c = 2)
  (h2 : C = π / 3)
  (h3 : area = sqrt 3)
  (h4 : (1 / 2) * a * b * sin C = area) :
  problem1 a b :=
by sorry

theorem solution2 (a b : ℝ) (c : ℝ) (C : ℝ) (cosA : ℝ)
  (h1 : c = 2)
  (h2 : C = π / 3)
  (h3 : cosA = sqrt 3 / 3)
  (h4 : sin (arccos (sqrt 3 / 3)) = sqrt 6 / 3)
  (h5 : (a / (sqrt 6 / 3)) = (2 / (sqrt 3 / 2)))
  (h6 : ((b / ((3 + sqrt 6) / 6)) = (2 / (sqrt 3 / 2)))) :
  problem2 b :=
by sorry

end solution1_solution2_l11_11033


namespace john_spent_on_sweets_l11_11640

def initial_amount := 7.10
def amount_given_per_friend := 1.00
def amount_left := 4.05
def amount_spent_on_friends := 2 * amount_given_per_friend
def amount_remaining_after_friends := initial_amount - amount_spent_on_friends
def amount_spent_on_sweets := amount_remaining_after_friends - amount_left

theorem john_spent_on_sweets : amount_spent_on_sweets = 1.05 := 
by
  sorry

end john_spent_on_sweets_l11_11640


namespace ratio_S7_S3_l11_11681

variable {a_n : ℕ → ℕ} -- Arithmetic sequence {a_n}
variable (S_n : ℕ → ℕ) -- Sum of the first n terms of the arithmetic sequence

-- Conditions
def ratio_a2_a4 (a_2 a_4 : ℕ) : Prop := a_2 = 7 * (a_4 / 6)
def sum_formula (n a_1 d : ℕ) : ℕ := n * (2 * a_1 + (n - 1) * d) / 2

-- Proof goal
theorem ratio_S7_S3 (a_1 d : ℕ) (h : ratio_a2_a4 (a_1 + d) (a_1 + 3 * d)): 
  (S_n 7 = sum_formula 7 a_1 d) ∧ (S_n 3 = sum_formula 3 a_1 d) →
  (S_n 7 / S_n 3 = 2) :=
by
  sorry

end ratio_S7_S3_l11_11681


namespace a_share_is_approx_560_l11_11946

noncomputable def investment_share (a_invest b_invest c_invest total_months b_share : ℕ) : ℝ :=
  let total_invest := a_invest + b_invest + c_invest
  let total_profit := (b_share * total_invest) / b_invest
  let a_share_ratio := a_invest / total_invest
  (a_share_ratio * total_profit)

theorem a_share_is_approx_560 
  (a_invest : ℕ := 7000) 
  (b_invest : ℕ := 11000) 
  (c_invest : ℕ := 18000) 
  (total_months : ℕ := 8) 
  (b_share : ℕ := 880) : 
  ∃ (a_share : ℝ), abs (a_share - 560) < 1 :=
by
  let a_share := investment_share a_invest b_invest c_invest total_months b_share
  existsi a_share
  sorry

end a_share_is_approx_560_l11_11946


namespace correct_removal_of_parentheses_C_incorrect_removal_of_parentheses_A_incorrect_removal_of_parentheses_B_incorrect_removal_of_parentheses_D_l11_11653

theorem correct_removal_of_parentheses_C (a : ℝ) :
    -(2 * a - 1) = -2 * a + 1 :=
by sorry

theorem incorrect_removal_of_parentheses_A (a : ℝ) :
    -(7 * a - 5) ≠ -7 * a - 5 :=
by sorry

theorem incorrect_removal_of_parentheses_B (a : ℝ) :
    -(-1 / 2 * a + 2) ≠ -1 / 2 * a - 2 :=
by sorry

theorem incorrect_removal_of_parentheses_D (a : ℝ) :
    -(-3 * a + 2) ≠ 3 * a + 2 :=
by sorry

end correct_removal_of_parentheses_C_incorrect_removal_of_parentheses_A_incorrect_removal_of_parentheses_B_incorrect_removal_of_parentheses_D_l11_11653


namespace finite_steps_iff_power_of_2_l11_11233

-- Define the conditions of the problem
def S (k n : ℕ) : ℕ := (k * (k + 1) / 2) % n

-- Define the predicate to check if the game finishes in finite number of steps
def game_completes (n : ℕ) : Prop :=
  ∃ k : ℕ, ∀ i : ℕ, i < n → S (k + i) n ≠ S k n

-- The main statement to prove
theorem finite_steps_iff_power_of_2 (n : ℕ) : game_completes n ↔ ∃ t : ℕ, n = 2^t :=
sorry  -- Placeholder for the proof

end finite_steps_iff_power_of_2_l11_11233


namespace gold_problem_proof_l11_11668

noncomputable def solve_gold_problem : Prop :=
  ∃ (a : ℕ → ℝ), 
  (a 1) + (a 2) + (a 3) = 4 ∧ 
  (a 8) + (a 9) + (a 10) = 3 ∧
  (a 5) + (a 6) = 7 / 3

theorem gold_problem_proof : solve_gold_problem := 
  sorry

end gold_problem_proof_l11_11668


namespace find_x_l11_11421

-- Define the percentages and multipliers as constants
def percent_47 := 47.0 / 100.0
def percent_36 := 36.0 / 100.0

-- Define the given quantities
def quantity1 := 1442.0
def quantity2 := 1412.0

-- Calculate the percentages of the quantities
def part1 := percent_47 * quantity1
def part2 := percent_36 * quantity2

-- Calculate the expression
def expression := (part1 - part2) + 63.0

-- Define the value of x given
def x := 232.42

-- Theorem stating the proof problem
theorem find_x : expression = x := by
  -- proof goes here
  sorry

end find_x_l11_11421


namespace train_cross_pole_in_5_seconds_l11_11756

/-- A train 100 meters long traveling at 72 kilometers per hour 
    will cross an electric pole in 5 seconds. -/
theorem train_cross_pole_in_5_seconds (L : ℝ) (v : ℝ) (t : ℝ) : 
  L = 100 → v = 72 * (1000 / 3600) → t = L / v → t = 5 :=
by
  sorry

end train_cross_pole_in_5_seconds_l11_11756


namespace complement_U_A_l11_11928

open Set

-- Definitions of the universal set U and the set A
def U : Set ℕ := {1, 2, 3}
def A : Set ℕ := {1, 2}

-- Proof statement: the complement of A with respect to U is {3}
theorem complement_U_A : U \ A = {3} :=
by
  sorry

end complement_U_A_l11_11928


namespace area_of_shaded_region_l11_11637

noncomputable def area_shaded (side : ℝ) : ℝ :=
  let area_square := side * side
  let radius := side / 2
  let area_circle := Real.pi * radius * radius
  area_square - area_circle

theorem area_of_shaded_region :
  let perimeter := 28
  let side := perimeter / 4
  area_shaded side = 49 - π * 12.25 :=
by
  sorry

end area_of_shaded_region_l11_11637


namespace factor_difference_of_squares_l11_11424

theorem factor_difference_of_squares (t : ℝ) : t^2 - 64 = (t - 8) * (t + 8) :=
by
  sorry

end factor_difference_of_squares_l11_11424


namespace line_through_two_points_l11_11882

theorem line_through_two_points :
  ∃ (m b : ℝ), (∀ x y : ℝ, (x, y) = (-2, 4) ∨ (x, y) = (-1, 3) → y = m * x + b) ∧ b = 2 ∧ m = -1 :=
by
  sorry

end line_through_two_points_l11_11882


namespace gasoline_distribution_impossible_l11_11352

theorem gasoline_distribution_impossible
  (x1 x2 x3 : ℝ)
  (h1 : x1 + x2 + x3 = 50)
  (h2 : x1 = x2 + 10)
  (h3 : x3 + 26 = x2) : false :=
by {
  sorry
}

end gasoline_distribution_impossible_l11_11352


namespace radius_of_large_circle_l11_11076

/-- Five circles are described with the given properties. -/
def small_circle_radius : ℝ := 2

/-- The angle between any centers of the small circles is 72 degrees due to equal spacing. -/
def angle_between_centers : ℝ := 72

/-- The final theorem states that the radius of the larger circle is as follows. -/
theorem radius_of_large_circle (number_of_circles : ℕ)
        (radius_small : ℝ)
        (angle : ℝ)
        (internally_tangent : ∀ (i : ℕ), i < number_of_circles → Prop)
        (externally_tangent : ∀ (i j : ℕ), i ≠ j → i < number_of_circles → j < number_of_circles → Prop) :
  number_of_circles = 5 →
  radius_small = small_circle_radius →
  angle = angle_between_centers →
  (∃ R : ℝ, R = 4 * Real.sqrt 5 - 2) 
:= by
  -- mathematical proof goes here
  sorry

end radius_of_large_circle_l11_11076


namespace bags_production_l11_11822

def machines_bags_per_minute (n : ℕ) : ℕ :=
  if n = 15 then 45 else 0 -- this definition is constrained by given condition

def bags_produced (machines : ℕ) (minutes : ℕ) : ℕ :=
  machines * (machines_bags_per_minute 15 / 15) * minutes

theorem bags_production (machines minutes : ℕ) (h : machines = 150 ∧ minutes = 8):
  bags_produced machines minutes = 3600 :=
by
  cases h with
  | intro h_machines h_minutes =>
    sorry

end bags_production_l11_11822


namespace additional_male_students_l11_11400

variable (a : ℕ)

theorem additional_male_students (h : a > 20) : a - 20 = (a - 20) := 
by 
  sorry

end additional_male_students_l11_11400


namespace binom_squared_l11_11277

theorem binom_squared :
  (Nat.choose 12 11) ^ 2 = 144 := 
by
  -- Mathematical steps would go here.
  sorry

end binom_squared_l11_11277


namespace triangle_problems_l11_11663

open Real

variables {A B C a b c : ℝ}
variables {m n : ℝ × ℝ}

def triangle_sides_and_angles (a b c : ℝ) (A B C : ℝ) : Prop := a > 0 ∧ b > 0 ∧ c > 0 ∧ A + B + C = π

def perpendicular (m n : ℝ × ℝ) : Prop := m.1 * n.1 + m.2 * n.2 = 0

noncomputable def area_of_triangle (a b c : ℝ) (A : ℝ) : ℝ :=
  1 / 2 * b * c * sin A

theorem triangle_problems
  (h1 : triangle_sides_and_angles a b c A B C)
  (h2 : m = (1, 1))
  (h3 : n = (sqrt 3 / 2 - sin B * sin C, cos B * cos C))
  (h4 : perpendicular m n)
  (h5 : a = 1)
  (h6 : b = sqrt 3 * c) :
  A = π / 6 ∧ area_of_triangle a b c A = sqrt 3 / 4 :=
by
  sorry

end triangle_problems_l11_11663


namespace interest_rate_decrease_l11_11257

theorem interest_rate_decrease (initial_rate final_rate : ℝ) (x : ℝ) 
  (h_initial_rate : initial_rate = 2.25 * 0.01)
  (h_final_rate : final_rate = 1.98 * 0.01) :
  final_rate = initial_rate * (1 - x)^2 := 
  sorry

end interest_rate_decrease_l11_11257


namespace necessary_condition_l11_11313

theorem necessary_condition (x : ℝ) : x = 1 → x^2 = 1 :=
by
  sorry

end necessary_condition_l11_11313


namespace cost_per_person_is_correct_l11_11200

-- Define the given conditions
def fee_per_30_minutes : ℕ := 4000
def bikes : ℕ := 4
def hours : ℕ := 3
def people : ℕ := 6

-- Calculate the correct answer based on the given conditions
noncomputable def cost_per_person : ℕ :=
  let fee_per_hour := 2 * fee_per_30_minutes
  let fee_per_3_hours := hours * fee_per_hour
  let total_cost := bikes * fee_per_3_hours
  total_cost / people

-- The theorem to be proved
theorem cost_per_person_is_correct : cost_per_person = 16000 := sorry

end cost_per_person_is_correct_l11_11200


namespace common_ratio_of_geometric_series_l11_11490

theorem common_ratio_of_geometric_series (a S r : ℝ) (h1 : a = 500) (h2 : S = 2500) (h3 : a / (1 - r) = S) : r = 4 / 5 :=
by
  rw [h1, h2] at h3
  sorry

end common_ratio_of_geometric_series_l11_11490


namespace ellipse_focus_distance_l11_11078

theorem ellipse_focus_distance :
  ∀ {x y : ℝ},
    (x^2) / 25 + (y^2) / 16 = 1 →
    (dist (x, y) (3, 0) = 8) →
    dist (x, y) (-3, 0) = 2 :=
by
  intro x y h₁ h₂
  sorry

end ellipse_focus_distance_l11_11078


namespace royalties_amount_l11_11113

/--
Given the following conditions:
1. No tax for royalties up to 800 yuan.
2. For royalties exceeding 800 yuan but not exceeding 4000 yuan, tax is levied at 14% on the amount exceeding 800 yuan.
3. For royalties exceeding 4000 yuan, tax is levied at 11% of the total royalties.

If someone has paid 420 yuan in taxes for publishing a book, prove that their royalties amount to 3800 yuan.
-/
theorem royalties_amount (r : ℝ) (h₁ : ∀ r, r ≤ 800 → 0 = r * 0 / 100)
  (h₂ : ∀ r, 800 < r ∧ r ≤ 4000 → 0.14 * (r - 800) = r * 0.14 / 100)
  (h₃ : ∀ r, r > 4000 → 0.11 * r = 420) : r = 3800 := sorry

end royalties_amount_l11_11113


namespace train_length_l11_11644

theorem train_length (time : ℕ) (speed_kmh : ℕ) (conversion_factor : ℚ) (speed_ms : ℚ) (length : ℚ) :
  time = 50 ∧ speed_kmh = 36 ∧ conversion_factor = 5 / 18 ∧ speed_ms = speed_kmh * conversion_factor ∧ length = speed_ms * time →
  length = 500 :=
by
  sorry

end train_length_l11_11644


namespace zero_sum_of_squares_eq_zero_l11_11417

theorem zero_sum_of_squares_eq_zero {a b : ℝ} (h : a^2 + b^2 = 0) : a = 0 ∧ b = 0 :=
sorry

end zero_sum_of_squares_eq_zero_l11_11417


namespace photo_album_requirement_l11_11590

-- Definition of the conditions
def pages_per_album : ℕ := 32
def photos_per_page : ℕ := 5
def total_photos : ℕ := 900

-- Calculation of photos per album
def photos_per_album := pages_per_album * photos_per_page

-- Calculation of required albums
noncomputable def albums_needed := (total_photos + photos_per_album - 1) / photos_per_album

-- Theorem to prove the required number of albums is 6
theorem photo_album_requirement : albums_needed = 6 :=
  by sorry

end photo_album_requirement_l11_11590


namespace pieces_count_l11_11951

def pieces_after_n_tears (n : ℕ) : ℕ :=
  3 * n + 1

theorem pieces_count (n : ℕ) : pieces_after_n_tears n = 3 * n + 1 :=
by
  sorry

end pieces_count_l11_11951


namespace class_b_students_l11_11935

theorem class_b_students (total_students : ℕ) (sample_size : ℕ) (class_a_sample : ℕ) :
  total_students = 100 → sample_size = 10 → class_a_sample = 4 → 
  (total_students - total_students * class_a_sample / sample_size = 60) :=
by
  intros
  sorry

end class_b_students_l11_11935


namespace find_f_at_2_l11_11991

def f (x : ℝ) (a b c : ℝ) : ℝ := a * x^5 + b * x^3 + c * x - 8

theorem find_f_at_2 (a b c : ℝ) (h : f (-2) a b c = 10) : f 2 a b c = -26 :=
by
  sorry

end find_f_at_2_l11_11991


namespace set_intersection_l11_11804

noncomputable def A : Set ℝ := { x | x / (x - 1) < 0 }
noncomputable def B : Set ℝ := { x | 0 < x ∧ x < 3 }
noncomputable def expected_intersection : Set ℝ := { x | 0 < x ∧ x < 1 }

theorem set_intersection (x : ℝ) : (x ∈ A ∧ x ∈ B) ↔ x ∈ expected_intersection :=
by
  sorry

end set_intersection_l11_11804


namespace new_paint_intensity_l11_11844

theorem new_paint_intensity (V : ℝ) (h1 : V > 0) :
    let initial_intensity := 0.5
    let replaced_fraction := 0.4
    let replaced_intensity := 0.25
    let new_intensity := (0.3 + 0.1 * replaced_fraction)  -- derived from (0.6 * 0.5 + 0.4 * 0.25)
    new_intensity = 0.4 :=
by
    sorry

end new_paint_intensity_l11_11844


namespace total_money_shared_l11_11102

def A_share (B : ℕ) : ℕ := B / 2
def B_share (C : ℕ) : ℕ := C / 2
def C_share : ℕ := 400

theorem total_money_shared (A B C : ℕ) (h1 : A = A_share B) (h2 : B = B_share C) (h3 : C = C_share) : A + B + C = 700 :=
by
  sorry

end total_money_shared_l11_11102


namespace quadratic_eqns_mod_7_l11_11907

/-- Proving the solutions for quadratic equations in arithmetic modulo 7. -/
theorem quadratic_eqns_mod_7 :
  (¬ ∃ x : ℤ, (5 * x^2 + 3 * x + 1) % 7 = 0) ∧
  (∃! x : ℤ, (x^2 + 3 * x + 4) % 7 = 0 ∧ x % 7 = 2) ∧
  (∃ x1 x2 : ℤ, (x1 ^ 2 - 2 * x1 - 3) % 7 = 0 ∧ (x2 ^ 2 - 2 * x2 - 3) % 7 = 0 ∧ 
              x1 % 7 = 3 ∧ x2 % 7 = 6) :=
by
  sorry

end quadratic_eqns_mod_7_l11_11907


namespace calculate_square_add_subtract_l11_11875

theorem calculate_square_add_subtract (a b : ℤ) :
  (41 : ℤ)^2 = (40 : ℤ)^2 + 81 ∧ (39 : ℤ)^2 = (40 : ℤ)^2 - 79 :=
by
  sorry

end calculate_square_add_subtract_l11_11875


namespace geometric_series_seventh_term_l11_11669

theorem geometric_series_seventh_term (a₁ a₁₀ : ℝ) (n : ℝ) (r : ℝ) :
  a₁ = 4 →
  a₁₀ = 93312 →
  n = 10 →
  a₁₀ = a₁ * r^(n-1) →
  (∃ (r : ℝ), r = 6) →
  4 * 6^(7-1) = 186624 := by
  intros a1_eq a10_eq n_eq an_eq exists_r
  sorry

end geometric_series_seventh_term_l11_11669


namespace find_AB_l11_11684

theorem find_AB
  (r R : ℝ)
  (h : r < R) :
  ∃ AB : ℝ, AB = (4 * r * (Real.sqrt (R * r))) / (R + r) :=
by
  sorry

end find_AB_l11_11684


namespace vector_on_plane_l11_11460

-- Define the vectors w and the condition for proj_w v
def w : ℝ × ℝ × ℝ := (3, -3, 3)
def v (x y z : ℝ) : ℝ × ℝ × ℝ := (x, y, z)
def projection_condition (x y z : ℝ) : Prop :=
  ((3 * x - 3 * y + 3 * z) / 27) * 3 = 6 ∧ ((3 * x - 3 * y + 3 * z) / 27) * (-3) = -6 ∧ ((3 * x - 3 * y + 3 * z) / 27) * 3 = 6

-- Define the plane equation
def plane_eq (x y z : ℝ) : Prop := x - y + z - 18 = 0

-- Prove that the set of vectors v lies on the plane
theorem vector_on_plane (x y z : ℝ) (h : projection_condition x y z) : plane_eq x y z :=
  sorry

end vector_on_plane_l11_11460


namespace four_digit_numbers_thousands_digit_5_div_by_5_l11_11144

theorem four_digit_numbers_thousands_digit_5_div_by_5 :
  ∃ (s : Finset ℕ), (∀ x ∈ s, 5000 ≤ x ∧ x ≤ 5999 ∧ x % 5 = 0) ∧ s.card = 200 :=
by
  sorry

end four_digit_numbers_thousands_digit_5_div_by_5_l11_11144


namespace average_nat_series_l11_11190

theorem average_nat_series : 
  let a := 12  -- first term
  let l := 53  -- last term
  let n := (l - a) / 1 + 1  -- number of terms
  let sum := n / 2 * (a + l)  -- sum of the arithmetic series
  let average := sum / n  -- average of the series
  average = 32.5 :=
by
  let a := 12
  let l := 53
  let n := (l - a) / 1 + 1
  let sum := n / 2 * (a + l)
  let average := sum / n
  sorry

end average_nat_series_l11_11190


namespace factorize_equivalence_l11_11356

-- declaring that the following definition may not be computable
noncomputable def factorize_expression (x y : ℝ) : Prop :=
  x^2 * y + x * y^2 = x * y * (x + y)

-- theorem to state the proof problem
theorem factorize_equivalence (x y : ℝ) : factorize_expression x y :=
sorry

end factorize_equivalence_l11_11356


namespace closest_point_on_parabola_l11_11259

/-- The coordinates of the point on the parabola y^2 = x that is closest to the line x - 2y + 4 = 0 are (1,1). -/
theorem closest_point_on_parabola (y : ℝ) (x : ℝ) (h_parabola : y^2 = x) (h_line : x - 2*y + 4 = 0) :
  (x = 1 ∧ y = 1) :=
sorry

end closest_point_on_parabola_l11_11259


namespace find_A_l11_11976

axiom power_eq_A (A : ℝ) (x y : ℝ) : 2^x = A ∧ 7^(2*y) = A
axiom reciprocal_sum_eq_2 (x y : ℝ) : (1/x) + (1/y) = 2

theorem find_A (A x y : ℝ) : 
  (2^x = A) ∧ (7^(2*y) = A) ∧ ((1/x) + (1/y) = 2) -> A = 7*Real.sqrt 2 :=
by 
  sorry

end find_A_l11_11976


namespace parallel_vectors_l11_11859

variable (a b : ℝ × ℝ)
variable (m : ℝ)

theorem parallel_vectors (h₁ : a = (-6, 2)) (h₂ : b = (m, -3)) (h₃ : a.1 * b.2 = a.2 * b.1) : m = 9 :=
by
  sorry

end parallel_vectors_l11_11859


namespace sum_of_remainders_l11_11563

theorem sum_of_remainders (n : ℤ) (h : n % 20 = 11) : (n % 4) + (n % 5) = 4 :=
by
  -- sorry is here to skip the actual proof as per instructions
  sorry

end sum_of_remainders_l11_11563


namespace problem_l11_11648

noncomputable def p (k : ℝ) (x : ℝ) := k * (x - 5) * (x - 2)
noncomputable def q (x : ℝ) := (x - 5) * (x + 3)

theorem problem {p q : ℝ → ℝ} (k : ℝ) :
  (∀ x, q x = (x - 5) * (x + 3)) →
  (∀ x, p x = k * (x - 5) * (x - 2)) →
  (∀ x ≠ 5, (p x) / (q x) = (3 * (x - 2)) / (x + 3)) →
  p 3 / q 3 = 1 / 2 :=
by
  sorry

end problem_l11_11648


namespace masha_more_cakes_l11_11438

theorem masha_more_cakes (S : ℝ) (m n : ℝ) (H1 : S > 0) (H2 : m > 0) (H3 : n > 0) 
  (H4 : 2 * S * (m + n) ≤ S * m + (1/3) * S * n) :
  m > n := 
by 
  sorry

end masha_more_cakes_l11_11438


namespace average_birds_seen_correct_l11_11396

-- Define the number of birds seen by each person
def birds_seen_by_marcus : ℕ := 7
def birds_seen_by_humphrey : ℕ := 11
def birds_seen_by_darrel : ℕ := 9

-- Define the number of people
def number_of_people : ℕ := 3

-- Calculate the total number of birds seen
def total_birds_seen : ℕ := birds_seen_by_marcus + birds_seen_by_humphrey + birds_seen_by_darrel

-- Calculate the average number of birds seen
def average_birds_seen : ℕ := total_birds_seen / number_of_people

-- Proof statement
theorem average_birds_seen_correct :
  average_birds_seen = 9 :=
by
  -- Leaving the proof out as instructed
  sorry

end average_birds_seen_correct_l11_11396


namespace cells_count_at_day_8_l11_11720

theorem cells_count_at_day_8 :
  let initial_cells := 3
  let common_ratio := 2
  let days := 8
  let interval := 2
  ∃ days_intervals, days_intervals = days / interval ∧ initial_cells * common_ratio ^ days_intervals = 48 :=
by
  sorry

end cells_count_at_day_8_l11_11720


namespace find_m_l11_11638

noncomputable def a_seq (a d : ℝ) (n : ℕ) : ℝ :=
  a + (n - 1 : ℝ) * d

noncomputable def S_n (a d : ℝ) (n : ℕ) : ℝ :=
  (n / 2) * (2 * a + (n - 1 : ℝ) * d)

theorem find_m (a d : ℝ) (m : ℕ) 
  (h1 : a_seq a d (m-1) + a_seq a d (m+1) - a = 0)
  (h2 : S_n a d (2*m - 1) = 38) : 
  m = 10 := 
sorry

end find_m_l11_11638


namespace average_rate_of_change_l11_11295

noncomputable def f (x : ℝ) : ℝ := x^2 + x

theorem average_rate_of_change :
  (f 2 - f 1) / (2 - 1) = 4 :=
by
  sorry

end average_rate_of_change_l11_11295


namespace inradius_circumradius_l11_11096

variables {T : Type} [MetricSpace T]

theorem inradius_circumradius (K k : ℝ) (d r rho : ℝ) (triangle : T)
  (h1 : (k / K) = (rho / r))
  (h2 : k ≤ K / 2)
  (h3 : 2 * r * rho = r^2 - d^2)
  (h4 : d ≥ 0) :
  r ≥ 2 * rho :=
sorry

end inradius_circumradius_l11_11096


namespace tino_jellybeans_l11_11427

theorem tino_jellybeans (Tino Lee Arnold Joshua : ℕ)
  (h1 : Tino = Lee + 24)
  (h2 : Arnold = Lee / 2)
  (h3 : Joshua = 3 * Arnold)
  (h4 : Arnold = 5) : Tino = 34 := by
sorry

end tino_jellybeans_l11_11427


namespace number_triangle_value_of_n_l11_11484

theorem number_triangle_value_of_n:
  ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ x * y = 2022 ∧ (∃ n : ℕ, n > 0 ∧ n^2 ∣ 2022 ∧ n = 1) :=
by sorry

end number_triangle_value_of_n_l11_11484


namespace symmetric_circle_equation_l11_11459

theorem symmetric_circle_equation :
  (∀ x y : ℝ, (x - 1) ^ 2 + y ^ 2 = 1 ↔ x ^ 2 + (y + 1) ^ 2 = 1) :=
by sorry

end symmetric_circle_equation_l11_11459


namespace midpoint_range_l11_11588

variable {x0 y0 : ℝ}

-- Conditions
def point_on_line1 (P : ℝ × ℝ) := P.1 + 2 * P.2 - 1 = 0
def point_on_line2 (Q : ℝ × ℝ) := Q.1 + 2 * Q.2 + 3 = 0
def is_midpoint (P Q M : ℝ × ℝ) := P.1 + Q.1 = 2 * M.1 ∧ P.2 + Q.2 = 2 * M.2
def midpoint_condition (M : ℝ × ℝ) := M.2 > M.1 + 2

-- Theorem
theorem midpoint_range
  (P Q M : ℝ × ℝ)
  (hP : point_on_line1 P)
  (hQ : point_on_line2 Q)
  (hM : is_midpoint P Q M)
  (h_cond : midpoint_condition M)
  (hx0 : x0 = M.1)
  (hy0 : y0 = M.2)
  : - (1 / 2) < y0 / x0 ∧ y0 / x0 < - (1 / 5) :=
sorry

end midpoint_range_l11_11588


namespace smallest_positive_perfect_square_divisible_by_2_3_5_l11_11196

theorem smallest_positive_perfect_square_divisible_by_2_3_5 : 
  ∃ (n : ℕ), (∃ k : ℕ, n = k^2) ∧ n % 2 = 0 ∧ n % 3 = 0 ∧ n % 5 = 0 ∧ 
  ∀ m : ℕ, (∃ k : ℕ, m = k^2) ∧ m % 2 = 0 ∧ m % 3 = 0 ∧ m % 5 = 0 → n ≤ m :=
sorry

end smallest_positive_perfect_square_divisible_by_2_3_5_l11_11196


namespace necessary_and_sufficient_condition_l11_11209

theorem necessary_and_sufficient_condition (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  a + b > a * b ↔ (a = 1 ∨ b = 1) :=
sorry

end necessary_and_sufficient_condition_l11_11209


namespace g_2023_eq_0_l11_11796

noncomputable def g (x : ℕ) : ℝ := sorry

axiom g_defined (x : ℕ) : ∃ y : ℝ, g x = y

axiom g_initial : g 1 = 1

axiom g_functional (a b : ℕ) : g (a + b) = g a + g b - 2 * g (a * b + 1)

theorem g_2023_eq_0 : g 2023 = 0 :=
sorry

end g_2023_eq_0_l11_11796


namespace children_tickets_l11_11838

theorem children_tickets (A C : ℝ) (h1 : A + C = 200) (h2 : 3 * A + 1.5 * C = 510) : C = 60 := by
  sorry

end children_tickets_l11_11838


namespace value_of_expression_l11_11740

theorem value_of_expression (x y z : ℕ) (h1 : x = 3) (h2 : y = 2) (h3 : z = 1) : 
  3 * x - 2 * y + 4 * z = 9 := 
by
  sorry

end value_of_expression_l11_11740


namespace sequence_a4_eq_5_over_3_l11_11743

theorem sequence_a4_eq_5_over_3 :
  ∀ (a : ℕ → ℚ), a 1 = 1 → (∀ n > 1, a n = 1 / a (n - 1) + 1) → a 4 = 5 / 3 :=
by
  intro a ha1 H
  sorry

end sequence_a4_eq_5_over_3_l11_11743


namespace car_speed_conversion_l11_11207

theorem car_speed_conversion (V_kmph : ℕ) (h : V_kmph = 36) : (V_kmph * 1000 / 3600) = 10 := by
  sorry

end car_speed_conversion_l11_11207


namespace sum_of_first_nine_terms_l11_11855

theorem sum_of_first_nine_terms (a : ℕ → ℤ) (d : ℤ) 
  (h1 : ∀ n, a (n + 1) = a n + d) 
  (h2 : a 1 = 3 * a 3 - 6) : 
  (9 * (a 0 + a 8)) / 2 = 27 := 
sorry

end sum_of_first_nine_terms_l11_11855


namespace mystical_swamp_l11_11547

/-- 
In a mystical swamp, there are two species of talking amphibians: toads, whose statements are always true, and frogs, whose statements are always false. 
Five amphibians: Adam, Ben, Cara, Dan, and Eva make the following statements:
Adam: "Eva and I are different species."
Ben: "Cara is a frog."
Cara: "Dan is a frog."
Dan: "Of the five of us, at least three are toads."
Eva: "Adam is a toad."
Given these statements, prove that the number of frogs is 3.
-/
theorem mystical_swamp :
  (∀ α β : Prop, α ∨ ¬β) ∧ -- Adam's statement: "Eva and I are different species."
  (Cara = "frog") ∧          -- Ben's statement: "Cara is a frog."
  (Dan = "frog") ∧         -- Cara's statement: "Dan is a frog."
  (∃ t, t = nat → t ≥ 3) ∧ -- Dan's statement: "Of the five of us, at least three are toads."
  (Adam = "toad")               -- Eva's statement: "Adam is a toad."
  → num_frogs = 3 := sorry       -- Number of frogs is 3.

end mystical_swamp_l11_11547


namespace power_of_expression_l11_11757

theorem power_of_expression (a b c d e : ℝ)
  (h1 : a - b - c + d = 18)
  (h2 : a + b - c - d = 6)
  (h3 : c + d - e = 5) :
  (2 * b - d + e) ^ 3 = 13824 :=
by
  sorry

end power_of_expression_l11_11757


namespace length_of_ladder_l11_11281

theorem length_of_ladder (a b : ℝ) (ha : a = 20) (hb : b = 15) : 
  ∃ c : ℝ, c^2 = a^2 + b^2 ∧ c = 25 := by
  sorry

end length_of_ladder_l11_11281


namespace circle_equation_with_focus_center_and_tangent_directrix_l11_11675

theorem circle_equation_with_focus_center_and_tangent_directrix :
  ∃ (x y : ℝ), (∃ k : ℝ, y^2 = -8 * x ∧ k = 2 ∧ (x = -2 ∧ y = 0) ∧ (x + 2)^2 + y^2 = 16) :=
by
  sorry

end circle_equation_with_focus_center_and_tangent_directrix_l11_11675


namespace solution_set_l11_11601

def op (a b : ℝ) : ℝ := -2 * a + b

theorem solution_set (x : ℝ) : (op x 4 > 0) ↔ (x < 2) :=
by {
  -- proof required here
  sorry
}

end solution_set_l11_11601


namespace count_solid_circles_among_first_2006_l11_11747

-- Definition of the sequence sum for location calculation
def sequence_sum (n : ℕ) : ℕ := (n + 1) * (n + 2) / 2 - 1

-- Main theorem
theorem count_solid_circles_among_first_2006 : 
  ∃ n : ℕ, sequence_sum (n - 1) < 2006 ∧ 2006 ≤ sequence_sum n ∧ n = 62 :=
by {
  sorry
}

end count_solid_circles_among_first_2006_l11_11747


namespace alex_chairs_l11_11038

theorem alex_chairs (x y z : ℕ) (h : x + y + z = 74) : z = 74 - x - y :=
by
  sorry

end alex_chairs_l11_11038


namespace value_of_f_l11_11306

variable {x t : ℝ}

noncomputable def f (x : ℝ) : ℝ :=
  if x = 0 ∨ x = 1 then 0
  else (1 : ℝ) / x

theorem value_of_f (h1 : ∀ x, x ≠ 0 → x ≠ 1 → f (x / (x - 1)) = 1 / x)
                   (h2 : 0 ≤ t ∧ t ≤ Real.pi / 2) :
  f (Real.tan t ^ 2 + 1) = Real.sin (2 * t) ^ 2 / 4 :=
sorry

end value_of_f_l11_11306


namespace total_apples_proof_l11_11475

-- Define the quantities Adam bought each day
def apples_monday := 15
def apples_tuesday := apples_monday * 3
def apples_wednesday := apples_tuesday * 4

-- The total quantity of apples Adam bought over these three days
def total_apples := apples_monday + apples_tuesday + apples_wednesday

-- Theorem stating that the total quantity of apples bought is 240
theorem total_apples_proof : total_apples = 240 := by
  sorry

end total_apples_proof_l11_11475


namespace sandy_age_l11_11141

variable (S M : ℕ)

-- Conditions
def condition1 := M = S + 12
def condition2 := S * 9 = M * 7

theorem sandy_age : condition1 S M → condition2 S M → S = 42 := by
  intros h1 h2
  sorry

end sandy_age_l11_11141


namespace estimate_fish_number_l11_11426

noncomputable def numFishInLake (marked: ℕ) (caughtSecond: ℕ) (markedSecond: ℕ) : ℕ :=
  let totalFish := (caughtSecond * marked) / markedSecond
  totalFish

theorem estimate_fish_number (marked caughtSecond markedSecond : ℕ) :
  marked = 100 ∧ caughtSecond = 200 ∧ markedSecond = 25 → numFishInLake marked caughtSecond markedSecond = 800 :=
by
  intros h
  cases h
  sorry

end estimate_fish_number_l11_11426


namespace monthly_growth_rate_optimal_selling_price_l11_11753

-- Conditions
def april_sales : ℕ := 150
def june_sales : ℕ := 216
def cost_price_per_unit : ℕ := 30
def initial_selling_price : ℕ := 40
def initial_sales_vol : ℕ := 300
def sales_decrease_rate : ℕ := 10
def desired_profit : ℕ := 3960

-- Questions (Proof statements)
theorem monthly_growth_rate :
  ∃ (x : ℝ), (1 + x) ^ 2 = (june_sales:ℝ) / (april_sales:ℝ) ∧ x = 0.2 := by
  sorry

theorem optimal_selling_price :
  ∃ (y : ℝ), (y - cost_price_per_unit) * (initial_sales_vol - sales_decrease_rate * (y - initial_selling_price)) = desired_profit ∧ y = 48 := by
  sorry

end monthly_growth_rate_optimal_selling_price_l11_11753


namespace chef_earns_less_than_manager_l11_11465

theorem chef_earns_less_than_manager :
  let manager_wage := 7.50
  let dishwasher_wage := manager_wage / 2
  let chef_wage := dishwasher_wage * 1.20
  (manager_wage - chef_wage) = 3.00 := by
    sorry

end chef_earns_less_than_manager_l11_11465


namespace fraction_of_termite_ridden_homes_collapsing_l11_11371

variable (T : ℕ) -- T represents the total number of homes
variable (termiteRiddenFraction : ℚ := 1/3) -- Fraction of homes that are termite-ridden
variable (termiteRiddenNotCollapsingFraction : ℚ := 1/7) -- Fraction of homes that are termite-ridden but not collapsing

theorem fraction_of_termite_ridden_homes_collapsing :
  termiteRiddenFraction - termiteRiddenNotCollapsingFraction = 4/21 :=
by
  -- Proof goes here
  sorry

end fraction_of_termite_ridden_homes_collapsing_l11_11371


namespace car_trip_time_difference_l11_11186

theorem car_trip_time_difference
  (average_speed : ℝ)
  (distance1 distance2 : ℝ)
  (speed_60_mph : average_speed = 60)
  (dist1_540 : distance1 = 540)
  (dist2_510 : distance2 = 510) :
  ((distance1 - distance2) / average_speed) * 60 = 30 := by
  sorry

end car_trip_time_difference_l11_11186


namespace fraction_orange_juice_in_large_container_l11_11472

-- Definitions according to the conditions
def pitcher1_capacity : ℕ := 800
def pitcher2_capacity : ℕ := 500
def pitcher1_fraction_orange_juice : ℚ := 1 / 4
def pitcher2_fraction_orange_juice : ℚ := 3 / 5

-- Prove the fraction of orange juice
theorem fraction_orange_juice_in_large_container :
  ( (pitcher1_capacity * pitcher1_fraction_orange_juice + pitcher2_capacity * pitcher2_fraction_orange_juice) / 
    (pitcher1_capacity + pitcher2_capacity) ) = 5 / 13 :=
by
  sorry

end fraction_orange_juice_in_large_container_l11_11472


namespace smallest_constant_for_triangle_sides_l11_11842

theorem smallest_constant_for_triangle_sides (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (triangle_condition : a + b > c ∧ b + c > a ∧ c + a > b) :
  ∃ N, (∀ a b c, (a + b > c ∧ b + c > a ∧ c + a > b) → (a^2 + b^2) / (a * b) < N) ∧ N = 2 := by
  sorry

end smallest_constant_for_triangle_sides_l11_11842


namespace largest_integer_less_than_100_with_remainder_5_l11_11029

theorem largest_integer_less_than_100_with_remainder_5 :
  ∃ n, (n < 100 ∧ n % 8 = 5) ∧ ∀ m, (m < 100 ∧ m % 8 = 5) → m ≤ n :=
sorry

end largest_integer_less_than_100_with_remainder_5_l11_11029


namespace find_value_of_expression_l11_11929

noncomputable def quadratic_function (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 2

theorem find_value_of_expression (a b : ℝ) (h : quadratic_function a b (-1) = 0) :
  2 * a - 2 * b = -4 :=
sorry

end find_value_of_expression_l11_11929


namespace cyclic_sum_inequality_l11_11070

variable {a b c x y z : ℝ}

-- Define the conditions
axiom h1 : a > 0
axiom h2 : b > 0
axiom h3 : c > 0
axiom h4 : x = a + 1 / b - 1
axiom h5 : y = b + 1 / c - 1
axiom h6 : z = c + 1 / a - 1
axiom h7 : x > 0
axiom h8 : y > 0
axiom h9 : z > 0

-- The statement we need to prove
theorem cyclic_sum_inequality : (x * y) / (Real.sqrt (x * y) + 2) + (y * z) / (Real.sqrt (y * z) + 2) + (z * x) / (Real.sqrt (z * x) + 2) ≥ 1 :=
sorry

end cyclic_sum_inequality_l11_11070


namespace find_a_l11_11607

noncomputable def polynomial1 (x : ℝ) : ℝ := x^3 + 3 * x^2 - x - 3
noncomputable def polynomial2 (x : ℝ) (a : ℝ) : ℝ := x^2 - 2 * a * x - 1

theorem find_a (a : ℝ) (x : ℝ) (hx1 : polynomial1 x > 0)
  (hx2 : polynomial2 x a ≤ 0) (ha : a > 0) : 
  3 / 4 ≤ a ∧ a < 4 / 3 :=
sorry

end find_a_l11_11607


namespace cuboid_edge_sum_l11_11350

-- Define the properties of a cuboid
structure Cuboid (α : Type) [LinearOrderedField α] where
  length : α
  width : α
  height : α

-- Define the volume of a cuboid
def volume {α : Type} [LinearOrderedField α] (c : Cuboid α) : α :=
  c.length * c.width * c.height

-- Define the surface area of a cuboid
def surface_area {α : Type} [LinearOrderedField α] (c : Cuboid α) : α :=
  2 * (c.length * c.width + c.width * c.height + c.height * c.length)

-- Define the sum of all edges of a cuboid
def edge_sum {α : Type} [LinearOrderedField α] (c : Cuboid α) : α :=
  4 * (c.length + c.width + c.height)

-- Given a geometric progression property
def gp_property {α : Type} [LinearOrderedField α] (c : Cuboid α) (q a : α) : Prop :=
  c.length = q * a ∧ c.width = a ∧ c.height = a / q

-- The main problem to be stated in Lean
theorem cuboid_edge_sum (α : Type) [LinearOrderedField α] (c : Cuboid α) (a q : α)
  (h1 : volume c = 8)
  (h2 : surface_area c = 32)
  (h3 : gp_property c q a) :
  edge_sum c = 32 := by
    sorry

end cuboid_edge_sum_l11_11350


namespace rain_in_both_areas_l11_11293

variable (P1 P2 : ℝ)
variable (hP1 : 0 < P1 ∧ P1 < 1)
variable (hP2 : 0 < P2 ∧ P2 < 1)

theorem rain_in_both_areas :
  ∀ P1 P2, (0 < P1 ∧ P1 < 1) → (0 < P2 ∧ P2 < 1) → (1 - P1) * (1 - P2) = (1 - P1) * (1 - P2) :=
by
  intros P1 P2 hP1 hP2
  sorry

end rain_in_both_areas_l11_11293


namespace percentage_decrease_l11_11413

theorem percentage_decrease (x y : ℝ) : 
  (xy^2 - (0.7 * x) * (0.6 * y)^2) / xy^2 = 0.748 :=
by
  sorry

end percentage_decrease_l11_11413


namespace remaining_area_exclude_smaller_rectangles_l11_11385

-- Conditions from part a)
variables (x : ℕ)
def large_rectangle_area := (x + 8) * (x + 6)
def small1_rectangle_area := (2 * x - 1) * (x - 1)
def small2_rectangle_area := (x - 3) * (x - 5)

-- Proof statement from part c)
theorem remaining_area_exclude_smaller_rectangles :
  large_rectangle_area x - (small1_rectangle_area x - small2_rectangle_area x) = 25 * x + 62 :=
by
  sorry

end remaining_area_exclude_smaller_rectangles_l11_11385


namespace bucket_weight_full_l11_11591

variable (c d : ℝ)

theorem bucket_weight_full (h1 : ∃ x y, x + (1 / 4) * y = c)
                           (h2 : ∃ x y, x + (3 / 4) * y = d) :
  ∃ x y, x + y = (3 * d - c) / 2 :=
by
  sorry

end bucket_weight_full_l11_11591


namespace no_rational_roots_l11_11365

theorem no_rational_roots {p q : ℤ} (hp : p % 2 = 1) (hq : q % 2 = 1) :
  ¬ ∃ x : ℚ, x^2 + (2 * p) * x + (2 * q) = 0 :=
by
  -- proof using contradiction technique
  sorry

end no_rational_roots_l11_11365


namespace max_min_f_l11_11172

noncomputable def f (x : ℝ) : ℝ :=
  if 6 ≤ x ∧ x ≤ 8 then
    (Real.sqrt (8 * x - x^2) - Real.sqrt (114 * x - x^2 - 48))
  else
    0

theorem max_min_f :
  ∀ x, 6 ≤ x ∧ x ≤ 8 → f x ≤ 2 * Real.sqrt 3 ∧ 0 ≤ f x :=
by
  intros
  sorry

end max_min_f_l11_11172


namespace grade_assignment_ways_l11_11370

theorem grade_assignment_ways (n_students : ℕ) (n_grades : ℕ) (h_students : n_students = 12) (h_grades : n_grades = 4) :
  (n_grades ^ n_students) = 16777216 := by
  rw [h_students, h_grades]
  rfl

end grade_assignment_ways_l11_11370


namespace inequality_holds_if_and_only_if_c_lt_0_l11_11569

theorem inequality_holds_if_and_only_if_c_lt_0 (a b c : ℝ) :
  (∀ x : ℝ, a * Real.sin x + b * Real.cos x + c > 0) ↔ (c < 0) :=
sorry

end inequality_holds_if_and_only_if_c_lt_0_l11_11569


namespace box_filling_possibilities_l11_11813

def possible_numbers : List ℕ := [2015, 2016, 2017, 2018, 2019]

def fill_the_boxes (D O G C W : ℕ) : Prop :=
  D + O + G = C + O + W

theorem box_filling_possibilities :
  (∃ D O G C W : ℕ, 
    D ∈ possible_numbers ∧
    O ∈ possible_numbers ∧
    G ∈ possible_numbers ∧
    C ∈ possible_numbers ∧
    W ∈ possible_numbers ∧
    D ≠ O ∧ D ≠ G ∧ D ≠ C ∧ D ≠ W ∧
    O ≠ G ∧ O ≠ C ∧ O ≠ W ∧
    G ≠ C ∧ G ≠ W ∧
    C ≠ W ∧
    fill_the_boxes D O G C W) → 
    ∃ ways : ℕ, ways = 24 :=
  sorry

end box_filling_possibilities_l11_11813


namespace age_difference_l11_11252

theorem age_difference :
  ∃ a b : ℕ, (a < 10) ∧ (b < 10) ∧
    (∀ x y : ℕ, (x = 10 * a + b) ∧ (y = 10 * b + a) → 
    (x + 5 = 2 * (y + 5)) ∧ ((10 * a + b) - (10 * b + a) = 18)) :=
by
  sorry

end age_difference_l11_11252


namespace quadratic_function_property_l11_11943

theorem quadratic_function_property
    (a b c : ℝ)
    (f : ℝ → ℝ)
    (h_f_def : ∀ x, f x = a * x^2 + b * x + c)
    (h_vertex : f (-2) = a^2)
    (h_point : f (-1) = 6)
    (h_vertex_condition : -b / (2 * a) = -2)
    (h_a_neg : a < 0) :
    (a + c) / b = 1 / 2 :=
by
  sorry

end quadratic_function_property_l11_11943


namespace total_gold_coins_l11_11040

theorem total_gold_coins (n c : ℕ) 
  (h1 : n = 11 * (c - 3))
  (h2 : n = 7 * c + 5) : 
  n = 75 := 
by 
  sorry

end total_gold_coins_l11_11040


namespace new_triangle_area_l11_11632

theorem new_triangle_area (a b : ℝ) (x y : ℝ) (hypotenuse : x = a ∧ y = b ∧ x^2 + y^2 = (a + b)^2) : 
    (3  * (1 / 2) * a * b) = (3 / 2) * a * b :=
by
  sorry

end new_triangle_area_l11_11632


namespace calculate_lassis_from_nine_mangoes_l11_11043

variable (mangoes_lassis_ratio : ℕ → ℕ → Prop)
variable (cost_per_mango : ℕ)

def num_lassis (mangoes : ℕ) : ℕ :=
  5 * mangoes
  
theorem calculate_lassis_from_nine_mangoes
  (h1 : mangoes_lassis_ratio 15 3)
  (h2 : cost_per_mango = 2) :
  num_lassis 9 = 45 :=
by
  sorry

end calculate_lassis_from_nine_mangoes_l11_11043


namespace complement_M_eq_interval_l11_11285

-- Definition of the set M
def M : Set ℝ := { x | x * (x - 3) > 0 }

-- Universal set is ℝ
def U : Set ℝ := Set.univ

-- Theorem to prove the complement of M in ℝ is [0, 3]
theorem complement_M_eq_interval :
  U \ M = { x | 0 ≤ x ∧ x ≤ 3 } :=
by
  sorry

end complement_M_eq_interval_l11_11285


namespace solve_system_l11_11156

theorem solve_system (x y z : ℝ) (h1 : (x + 1) * y * z = 12) 
                               (h2 : (y + 1) * z * x = 4) 
                               (h3 : (z + 1) * x * y = 4) : 
  (x = 1 / 3 ∧ y = 3 ∧ z = 3) ∨ (x = 2 ∧ y = -2 ∧ z = -2) :=
sorry

end solve_system_l11_11156


namespace effective_average_speed_l11_11890

def rowing_speed_with_stream := 16 -- km/h
def rowing_speed_against_stream := 6 -- km/h
def stream1_effect := 2 -- km/h
def stream2_effect := -1 -- km/h
def stream3_effect := 3 -- km/h
def opposing_wind := 1 -- km/h

theorem effective_average_speed :
  ((rowing_speed_with_stream + stream1_effect - opposing_wind) + 
   (rowing_speed_against_stream + stream2_effect - opposing_wind) + 
   (rowing_speed_with_stream + stream3_effect - opposing_wind)) / 3 = 13 := 
by
  sorry

end effective_average_speed_l11_11890


namespace women_exceed_men_l11_11843

variable (M W : ℕ)

theorem women_exceed_men (h1 : M + W = 24) (h2 : (M : ℚ) / (W : ℚ) = 0.6) : W - M = 6 :=
sorry

end women_exceed_men_l11_11843


namespace long_jump_record_l11_11323

theorem long_jump_record 
  (standard_distance : ℝ)
  (jump1 : ℝ)
  (jump2 : ℝ)
  (record1 : ℝ)
  (record2 : ℝ)
  (h1 : standard_distance = 4.00)
  (h2 : jump1 = 4.22)
  (h3 : jump2 = 3.85)
  (h4 : record1 = jump1 - standard_distance)
  (h5 : record2 = jump2 - standard_distance)
  : record2 = -0.15 := 
sorry

end long_jump_record_l11_11323


namespace dictionary_prices_and_max_A_l11_11431

-- Definitions for the problem
def price_A := 70
def price_B := 50

-- Conditions from the problem
def condition1 := (price_A + 2 * price_B = 170)
def condition2 := (2 * price_A + 3 * price_B = 290)

-- The proof problem statement
theorem dictionary_prices_and_max_A (h1 : price_A + 2 * price_B = 170) (h2 : 2 * price_A + 3 * price_B = 290) :
  price_A = 70 ∧ price_B = 50 ∧ (∀ (x y : ℕ), x + y = 30 → 70 * x + 50 * y ≤ 1600 → x ≤ 5) :=
by
  sorry

end dictionary_prices_and_max_A_l11_11431


namespace probability_of_winning_second_lawsuit_l11_11282

theorem probability_of_winning_second_lawsuit
  (P_W1 P_L1 P_W2 P_L2 : ℝ)
  (h1 : P_W1 = 0.30)
  (h2 : P_L1 = 0.70)
  (h3 : P_L1 * P_L2 = P_W1 * P_W2 + 0.20)
  (h4 : P_L2 = 1 - P_W2) :
  P_W2 = 0.50 :=
by
  sorry

end probability_of_winning_second_lawsuit_l11_11282


namespace sets_equality_l11_11440

variables {α : Type*} (A B C : Set α)

theorem sets_equality (h1 : A ∪ B ⊆ C) (h2 : A ∪ C ⊆ B) (h3 : B ∪ C ⊆ A) : A = B ∧ B = C :=
by
  sorry

end sets_equality_l11_11440


namespace sum_of_interior_angles_of_special_regular_polygon_l11_11711

theorem sum_of_interior_angles_of_special_regular_polygon (n : ℕ) (h1 : n = 4 ∨ n = 5) :
  ((n - 2) * 180 = 360 ∨ (n - 2) * 180 = 540) :=
by sorry

end sum_of_interior_angles_of_special_regular_polygon_l11_11711


namespace students_without_favorite_subject_l11_11649

theorem students_without_favorite_subject
  (total_students : ℕ)
  (students_like_math : ℕ)
  (students_like_english : ℕ)
  (remaining_students : ℕ)
  (students_like_science : ℕ)
  (students_without_favorite : ℕ)
  (h1 : total_students = 30)
  (h2 : students_like_math = total_students * (1 / 5))
  (h3 : students_like_english = total_students * (1 / 3))
  (h4 : remaining_students = total_students - (students_like_math + students_like_english))
  (h5 : students_like_science = remaining_students * (1 / 7))
  (h6 : students_without_favorite = remaining_students - students_like_science) :
  students_without_favorite = 12 := by
  sorry

end students_without_favorite_subject_l11_11649


namespace three_digit_number_addition_l11_11732

theorem three_digit_number_addition (a b : ℕ) (ha : a < 10) (hb : b < 10) (h1 : 307 + 294 = 6 * 100 + b * 10 + 1)
  (h2 : (6 * 100 + b * 10 + 1) % 7 = 0) : a + b = 8 :=
by {
  sorry  -- Proof steps not needed
}

end three_digit_number_addition_l11_11732


namespace valid_starting_day_count_l11_11539

-- Defining the structure of the 30-day month and conditions
def days_in_month : Nat := 30

-- A function to determine the number of each weekday in a month which also checks if the given day is valid as per conditions
def valid_starting_days : List Nat :=
  [1] -- '1' represents Tuesday being the valid starting day corresponding to equal number of Tuesdays and Thursdays

-- The theorem we want to prove
-- The goal is to prove that there is only 1 valid starting day for the 30-day month to have equal number of Tuesdays and Thursdays
theorem valid_starting_day_count (days : Nat) (valid_days : List Nat) : 
  days = days_in_month → valid_days = valid_starting_days :=
by
  -- Sorry to skip full proof implementation
  sorry

end valid_starting_day_count_l11_11539


namespace number_of_square_tiles_l11_11624

theorem number_of_square_tiles (a b : ℕ) (h1 : a + b = 32) (h2 : 3 * a + 4 * b = 110) : b = 14 :=
by
  -- the proof steps are skipped
  sorry

end number_of_square_tiles_l11_11624


namespace max_value_of_f_l11_11751

noncomputable def f (x : ℝ) : ℝ := x * (4 - x)

theorem max_value_of_f : ∃ y, ∀ x ∈ Set.Ioo 0 4, f x ≤ y ∧ y = 4 :=
by
  sorry

end max_value_of_f_l11_11751


namespace probability_at_least_one_first_class_part_l11_11395

-- Define the problem constants
def total_parts : ℕ := 6
def first_class_parts : ℕ := 4
def second_class_parts : ℕ := 2
def choose (n k : ℕ) : ℕ := Nat.choose n k

-- Define the target probability
def target_probability : ℚ := 14 / 15

-- Statement of the problem as a Lean theorem
theorem probability_at_least_one_first_class_part :
  (1 - (choose second_class_parts 2 : ℚ) / (choose total_parts 2 : ℚ)) = target_probability :=
by
  -- the proof is omitted
  sorry

end probability_at_least_one_first_class_part_l11_11395


namespace find_f_of_neg_1_l11_11227

-- Define the conditions
variables (a b c : ℝ)
variables (g f : ℝ → ℝ)
axiom g_definition : ∀ x, g x = x^3 + a*x^2 + 2*x + 15
axiom f_definition : ∀ x, f x = x^4 + x^3 + b*x^2 + 150*x + c

axiom g_has_distinct_roots : ∃ r1 r2 r3 : ℝ, r1 ≠ r2 ∧ r2 ≠ r3 ∧ r1 ≠ r3 ∧ ∀ x, g x = 0 ↔ (x = r1 ∨ x = r2 ∨ x = r3)
axiom roots_of_g_are_roots_of_f : ∀ x, g x = 0 → f x = 0

-- Prove the value of f(-1) given the conditions
theorem find_f_of_neg_1 (a : ℝ) (b : ℝ) (c : ℝ) (g f : ℝ → ℝ)
  (h_g_def : ∀ x, g x = x^3 + a*x^2 + 2*x + 15)
  (h_f_def : ∀ x, f x = x^4 + x^3 + b*x^2 + 150*x + c)
  (h_g_has_distinct_roots : ∃ r1 r2 r3 : ℝ, r1 ≠ r2 ∧ r2 ≠ r3 ∧ r1 ≠ r3 ∧ ∀ x, g x = 0 ↔ (x = r1 ∨ x = r2 ∨ x = r3))
  (h_roots : ∀ x, g x = 0 → f x = 0) :
  f (-1) = 3733.25 := 
by {
  sorry
}

end find_f_of_neg_1_l11_11227


namespace fibers_below_20_count_l11_11721

variable (fibers : List ℕ)

-- Conditions
def total_fibers := fibers.length = 100
def length_interval (f : ℕ) := 5 ≤ f ∧ f ≤ 40
def fibers_within_interval := ∀ f ∈ fibers, length_interval f

-- Question
def fibers_less_than_20 (fibers : List ℕ) : Nat :=
  (fibers.filter (λ f => f < 20)).length

theorem fibers_below_20_count (h_total : total_fibers fibers)
  (h_interval : fibers_within_interval fibers)
  (histogram_data : fibers_less_than_20 fibers = 30) :
  fibers_less_than_20 fibers = 30 :=
by
  sorry

end fibers_below_20_count_l11_11721


namespace t_f_3_equals_sqrt_44_l11_11661

noncomputable def t (x : ℝ) : ℝ := Real.sqrt (4 * x + 4)
noncomputable def f (x : ℝ) : ℝ := 6 + t x

theorem t_f_3_equals_sqrt_44 : t (f 3) = Real.sqrt 44 := by
  sorry

end t_f_3_equals_sqrt_44_l11_11661


namespace arithmetic_sequence_150th_term_l11_11126

theorem arithmetic_sequence_150th_term :
  let a₁ := 3
  let d := 5
  let n := 150
  (a₁ + (n - 1) * d) = 748 :=
by
  let a₁ := 3
  let d := 5
  let n := 150
  show a₁ + (n - 1) * d = 748
  sorry

end arithmetic_sequence_150th_term_l11_11126


namespace translate_graph_cos_l11_11722

/-- Let f(x) = cos(2x). 
    Translate f(x) to the left by π/6 units to get g(x), 
    then translate g(x) upwards by 1 unit to get h(x). 
    Prove that h(x) = cos(2x + π/3) + 1. -/
theorem translate_graph_cos :
  let f (x : ℝ) := Real.cos (2 * x)
  let g (x : ℝ) := f (x + Real.pi / 6)
  let h (x : ℝ) := g x + 1
  ∀ (x : ℝ), h x = Real.cos (2 * x + Real.pi / 3) + 1 :=
by
  sorry

end translate_graph_cos_l11_11722


namespace find_number_l11_11856

theorem find_number :
  ∃ x : Int, x - (28 - (37 - (15 - 20))) = 59 ∧ x = 45 :=
by
  sorry

end find_number_l11_11856


namespace bill_needs_paint_cans_l11_11957

theorem bill_needs_paint_cans :
  let bedrooms := 3
  let other_rooms := 2 * bedrooms
  let gallons_per_room := 2
  let color_paint_cans := 6 -- (bedrooms * gallons_per_room) / 1-gallon per can
  let white_paint_cans := 4 -- (other_rooms * gallons_per_room) / 3-gallons per can
  (color_paint_cans + white_paint_cans) = 10 := sorry

end bill_needs_paint_cans_l11_11957


namespace arrangements_A_and_B_together_arrangements_A_not_head_B_not_tail_arrangements_A_and_B_not_next_arrangements_one_person_between_A_and_B_l11_11789

open Nat

axiom students : Fin 7 → Type -- Define students indexed by their position in the line.

noncomputable def fact : ℕ → ℕ
| 0       => 1
| (n + 1) => (n + 1) * fact n

theorem arrangements_A_and_B_together :
  (2 * fact 6) = 1440 := 
by 
  sorry

theorem arrangements_A_not_head_B_not_tail :
  (fact 7 - 2 * fact 6 + fact 5) = 3720 := 
by 
  sorry

theorem arrangements_A_and_B_not_next :
  (3600) = 3600 := 
by 
  sorry

theorem arrangements_one_person_between_A_and_B :
  (fact 5 * 2) = 1200 := 
by 
  sorry

end arrangements_A_and_B_together_arrangements_A_not_head_B_not_tail_arrangements_A_and_B_not_next_arrangements_one_person_between_A_and_B_l11_11789


namespace b_value_rational_polynomial_l11_11914

theorem b_value_rational_polynomial (a b : ℚ) :
  (Polynomial.aeval (2 + Real.sqrt 3) (Polynomial.C (-15) + Polynomial.C b * X + Polynomial.C a * X^2 + X^3 : Polynomial ℚ) = 0) →
  b = -44 :=
by
  sorry

end b_value_rational_polynomial_l11_11914


namespace find_a_minus_b_plus_c_l11_11048

def a_n (n : ℕ) : ℕ := 4 * n - 3

def S_n (a b c n : ℕ) : ℕ := 2 * a * n ^ 2 + b * n + c

theorem find_a_minus_b_plus_c
  (a b c : ℕ)
  (h : ∀ n : ℕ, n > 0 → S_n a b c n = 2 * n ^ 2 - n)
  : a - b + c = 2 :=
by
  sorry

end find_a_minus_b_plus_c_l11_11048


namespace rectangle_area_l11_11332

theorem rectangle_area (AB AC : ℝ) (h_AB : AB = 15) (h_AC : AC = 17) : ∃ Area : ℝ, Area = 120 :=
by
  sorry

end rectangle_area_l11_11332


namespace water_ratio_horse_pig_l11_11702

-- Definitions based on conditions
def num_pigs : ℕ := 8
def water_per_pig : ℕ := 3
def num_horses : ℕ := 10
def water_for_chickens : ℕ := 30
def total_water : ℕ := 114

-- Statement of the problem
theorem water_ratio_horse_pig : 
  (total_water - (num_pigs * water_per_pig) - water_for_chickens) / num_horses / water_per_pig = 2 := 
by sorry

end water_ratio_horse_pig_l11_11702


namespace functional_equation_solution_l11_11557

-- Define ℕ* (positive integers) as a subtype of ℕ
def Nat.star := {n : ℕ // n > 0}

-- Define the problem statement
theorem functional_equation_solution (f : Nat.star → Nat.star) :
  (∀ m n : Nat.star, m.val ^ 2 + (f n).val ∣ m.val * (f m).val + n.val) →
  (∀ n : Nat.star, f n = n) :=
by
  intro h
  sorry

end functional_equation_solution_l11_11557


namespace sum_of_x_coordinates_mod_20_l11_11683

theorem sum_of_x_coordinates_mod_20 (y x : ℤ) (h1 : y ≡ 7 * x + 3 [ZMOD 20]) (h2 : y ≡ 13 * x + 17 [ZMOD 20]) 
: ∃ (x1 x2 : ℤ), (0 ≤ x1 ∧ x1 < 20) ∧ (0 ≤ x2 ∧ x2 < 20) ∧ x1 ≡ 1 [ZMOD 10] ∧ x2 ≡ 11 [ZMOD 10] ∧ x1 + x2 = 12 := sorry

end sum_of_x_coordinates_mod_20_l11_11683


namespace rational_solutions_quadratic_l11_11333

theorem rational_solutions_quadratic (k : ℕ) (h_pos : 0 < k) :
  (∃ (x : ℚ), k * x^2 + 24 * x + k = 0) ↔ k = 12 :=
by
  sorry

end rational_solutions_quadratic_l11_11333


namespace base_conversion_unique_b_l11_11578

theorem base_conversion_unique_b (b : ℕ) (h_b_pos : 0 < b) :
  (1 * 5^2 + 3 * 5^1 + 2 * 5^0) = (2 * b^2 + b) → b = 4 :=
by
  sorry

end base_conversion_unique_b_l11_11578


namespace quadratic_function_m_value_l11_11301

theorem quadratic_function_m_value :
  ∃ m : ℝ, (m - 3 ≠ 0) ∧ (m^2 - 7 = 2) ∧ m = -3 :=
by
  sorry

end quadratic_function_m_value_l11_11301


namespace mean_of_remaining_number_is_2120_l11_11077

theorem mean_of_remaining_number_is_2120 (a1 a2 a3 a4 a5 a6 : ℕ) 
    (h1 : a1 = 1451) (h2 : a2 = 1723) (h3 : a3 = 1987) (h4 : a4 = 2056) 
    (h5 : a5 = 2191) (h6 : a6 = 2212) 
    (mean_five : (a1 + a2 + a3 + a4 + a5) = 9500):
-- Prove that the mean of the remaining number a6 is 2120
  (a6 = 2120) :=
by
  -- Placeholder for proof
  sorry

end mean_of_remaining_number_is_2120_l11_11077


namespace cubic_root_equality_l11_11169

theorem cubic_root_equality (a b c : ℝ) (h1 : a + b + c = 12) (h2 : a * b + b * c + c * a = 14) (h3 : a * b * c = -3) :
  (1 / a^2) + (1 / b^2) + (1 / c^2) = 268 / 9 := 
by
  sorry

end cubic_root_equality_l11_11169


namespace part1_relationship_range_part2_maximize_profit_l11_11272

variables {x y a : ℝ}
noncomputable def zongzi_profit (x : ℝ) : ℝ := -5 * x + 6000

-- Given conditions
def conditions (x : ℝ) : Prop :=
  100 ≤ x ∧ x ≤ 150

-- Part 1: Prove the functional relationship and range of x
theorem part1_relationship_range (x : ℝ) (h : conditions x) :
  zongzi_profit x = -5 * x + 6000 :=
  sorry

-- Part 2: Profit maximization given modified purchase price condition
noncomputable def modified_zongzi_profit (x : ℝ) (a : ℝ) : ℝ :=
  (a - 5) * x + 6000

def maximize_strategy (x a : ℝ) : Prop :=
  (0 < a ∧ a < 5 → x = 100) ∧ (5 ≤ a ∧ a < 10 → x = 150)

theorem part2_maximize_profit (a : ℝ) (ha : 0 < a ∧ a < 10) :
  ∃ x, conditions x ∧ maximize_strategy x a :=
  sorry

end part1_relationship_range_part2_maximize_profit_l11_11272


namespace given_eqn_simplification_l11_11664

theorem given_eqn_simplification (x : ℝ) (h : 6 * x^2 - 4 * x - 3 = 0) : 
  (x - 1)^2 + x * (x + 2 / 3) = 2 :=
by
  sorry

end given_eqn_simplification_l11_11664


namespace minimum_value_omega_l11_11023

variable (f : ℝ → ℝ) (ω ϕ T : ℝ) (x : ℝ)
variable (h_zero : 0 < ω) (h_phi_range : 0 < ϕ ∧ ϕ < π)
variable (h_period : T = 2 * π / ω)
variable (h_f_period : f T = sqrt 3 / 2)
variable (h_zero_of_f : f (π / 9) = 0)
variable (h_f_def : ∀ x, f x = cos (ω * x + ϕ))

theorem minimum_value_omega : ω = 3 := by sorry

end minimum_value_omega_l11_11023


namespace abs_inequality_solution_l11_11284

theorem abs_inequality_solution (x : ℝ) : (|2 * x - 1| - |x - 2| < 0) ↔ (-1 < x ∧ x < 1) := 
sorry

end abs_inequality_solution_l11_11284


namespace percentage_failing_both_l11_11787

-- Define the conditions as constants
def percentage_failing_hindi : ℝ := 0.25
def percentage_failing_english : ℝ := 0.48
def percentage_passing_both : ℝ := 0.54

-- Define the percentage of students who failed in at least one subject
def percentage_failing_at_least_one : ℝ := 1 - percentage_passing_both

-- The main theorem statement we want to prove
theorem percentage_failing_both :
  percentage_failing_at_least_one = percentage_failing_hindi + percentage_failing_english - 0.27 := by
sorry

end percentage_failing_both_l11_11787


namespace max_minus_min_depends_on_a_not_b_l11_11318

def quadratic_function (a b x : ℝ) : ℝ := x^2 + a * x + b

theorem max_minus_min_depends_on_a_not_b (a b : ℝ) :
  let f := quadratic_function a b
  let M := max (f 0) (f 1)
  let m := min (f 0) (f 1)
  M - m == |a| :=
sorry

end max_minus_min_depends_on_a_not_b_l11_11318


namespace simplify_and_evaluate_expression_l11_11245

variable (x y : ℚ)

theorem simplify_and_evaluate_expression :
    x = 2 / 15 → y = 3 / 2 → 
    (2 * x + y)^2 - (3 * x - y)^2 + 5 * x * (x - y) = 1 :=
by 
  intros h1 h2
  subst h1
  subst h2
  sorry

end simplify_and_evaluate_expression_l11_11245


namespace angle_equivalence_l11_11784

theorem angle_equivalence : (2023 % 360 = -137 % 360) := 
by 
  sorry

end angle_equivalence_l11_11784


namespace problem_statement_l11_11749

variable (n : ℕ)
variable (op : ℕ → ℕ → ℕ)
variable (h1 : op 1 1 = 1)
variable (h2 : ∀ n, op (n+1) 1 = 3 * op n 1)

theorem problem_statement : op 5 1 - op 2 1 = 78 := by
  sorry

end problem_statement_l11_11749


namespace Tod_drove_time_l11_11247

section
variable (distance_north: ℕ) (distance_west: ℕ) (speed: ℕ)

theorem Tod_drove_time :
  distance_north = 55 → distance_west = 95 → speed = 25 → 
  (distance_north + distance_west) / speed = 6 :=
by
  intros
  sorry
end

end Tod_drove_time_l11_11247


namespace factor_polynomial_l11_11492

theorem factor_polynomial (a : ℝ) : 74 * a^2 + 222 * a + 148 * a^3 = 74 * a * (2 * a^2 + a + 3) :=
by
  sorry

end factor_polynomial_l11_11492


namespace right_triangle_area_l11_11336

/-- Given a right triangle with hypotenuse 13 meters and one side 5 meters,
prove that the area of the triangle is 30 square meters. -/
theorem right_triangle_area (a b c : ℝ) (h : a^2 + b^2 = c^2) (hc : c = 13) (ha : a = 5) :
  1/2 * a * b = 30 :=
by sorry

end right_triangle_area_l11_11336


namespace intersection_A_B_l11_11846

def A : Set (ℝ × ℝ) := {p | ∃ x y, p = (x, y) ∧ y = 3^x}
def B : Set (ℝ × ℝ) := {p | ∃ x y, p = (x, y) ∧ y = 2^(-x)}

theorem intersection_A_B :
  A ∩ B = {p | p = (0, 1)} :=
by
  sorry

end intersection_A_B_l11_11846


namespace triangle_cosine_identity_l11_11265

theorem triangle_cosine_identity
  (a b c : ℝ)
  (α β γ : ℝ)
  (hα : α = Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c)))
  (hβ : β = Real.arccos ((a^2 + c^2 - b^2) / (2 * a * c)))
  (hγ : γ = Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b)))
  (habc_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  (b / c + c / b) * Real.cos α + 
  (c / a + a / c) * Real.cos β + 
  (a / b + b / a) * Real.cos γ = 3 := 
sorry

end triangle_cosine_identity_l11_11265


namespace circumscribed_circle_radius_l11_11953

theorem circumscribed_circle_radius (b c : ℝ) (cosA : ℝ)
  (hb : b = 2) (hc : c = 3) (hcosA : cosA = 1 / 3) : 
  R = 9 * Real.sqrt 2 / 8 :=
by
  sorry

end circumscribed_circle_radius_l11_11953


namespace gum_needed_l11_11765

-- Definitions based on problem conditions
def num_cousins : ℕ := 4
def gum_per_cousin : ℕ := 5

-- Proposition that we need to prove
theorem gum_needed : num_cousins * gum_per_cousin = 20 := by
  sorry

end gum_needed_l11_11765


namespace find_a_m_18_l11_11629

variable (a : ℕ → ℝ)
variable (r : ℝ)
variable (a1 : ℝ)
variable (m : ℕ)

noncomputable def geometric_sequence (a : ℕ → ℝ) (a1 : ℝ) (r : ℝ) :=
  ∀ n : ℕ, a n = a1 * r^n

def problem_conditions (a : ℕ → ℝ) (r : ℝ) (a1 : ℝ) (m : ℕ) :=
  (geometric_sequence a a1 r) ∧
  a m = 3 ∧
  a (m + 6) = 24

theorem find_a_m_18 (a : ℕ → ℝ) (r : ℝ) (a1 : ℝ) (m : ℕ) :
  problem_conditions a r a1 m → a (m + 18) = 1536 :=
by
  sorry

end find_a_m_18_l11_11629


namespace multiplication_72515_9999_l11_11803

theorem multiplication_72515_9999 : 72515 * 9999 = 725077485 :=
by
  sorry

end multiplication_72515_9999_l11_11803


namespace tan_alpha_implies_fraction_l11_11004

theorem tan_alpha_implies_fraction (α : ℝ) (h : Real.tan α = -3/2) : 
  (Real.sin α + 2 * Real.cos α) / (Real.cos α - Real.sin α) = 1 / 5 := 
sorry

end tan_alpha_implies_fraction_l11_11004


namespace car_fuel_tanks_l11_11388

theorem car_fuel_tanks {x X p : ℝ}
  (h1 : x + X = 70)            -- Condition: total capacity is 70 liters
  (h2 : x * p = 45)            -- Condition: cost to fill small car's tank
  (h3 : X * (p + 0.29) = 68)   -- Condition: cost to fill large car's tank
  : x = 30 ∧ X = 40            -- Conclusion: capacities of the tanks
  :=
by {
  sorry
}

end car_fuel_tanks_l11_11388


namespace tunnel_length_correct_l11_11142

noncomputable def tunnel_length (truck_length : ℝ) (time_to_exit : ℝ) (speed_mph : ℝ) (mile_to_feet : ℝ) : ℝ :=
let speed_fps := (speed_mph * mile_to_feet) / 3600
let total_distance := speed_fps * time_to_exit
total_distance - truck_length

theorem tunnel_length_correct :
  tunnel_length 66 6 45 5280 = 330 :=
by
  sorry

end tunnel_length_correct_l11_11142


namespace correct_multiplication_result_l11_11631

theorem correct_multiplication_result :
  0.08 * 3.25 = 0.26 :=
by
  -- This is to ensure that the theorem is well-formed and logically connected
  sorry

end correct_multiplication_result_l11_11631


namespace binom_coefficient_largest_l11_11022

theorem binom_coefficient_largest (n : ℕ) (h : (n / 2) + 1 = 7) : n = 12 :=
by
  sorry

end binom_coefficient_largest_l11_11022


namespace correct_statement_is_C_l11_11942

theorem correct_statement_is_C :
  (∃ x : ℚ, ∀ y : ℚ, x < y) = false ∧
  (∃ x : ℚ, x < 0 ∧ ∀ y : ℚ, y < 0 → x < y) = false ∧
  (∃ x : ℝ, ∀ y : ℝ, abs x ≤ abs y) ∧
  (∃ x : ℝ, 0 < x ∧ ∀ y : ℝ, 0 < y → x ≤ y) = false :=
sorry

end correct_statement_is_C_l11_11942


namespace point_A_coords_l11_11954

theorem point_A_coords (x y : ℝ) (h : ∀ t : ℝ, (t + 1) * x - (2 * t + 5) * y - 6 = 0) : x = -4 ∧ y = -2 := by
  sorry

end point_A_coords_l11_11954


namespace least_value_q_minus_p_l11_11731

def p : ℝ := 2
def q : ℝ := 5

theorem least_value_q_minus_p (y : ℝ) (h : p < y ∧ y < q) : q - p = 3 :=
by
  sorry

end least_value_q_minus_p_l11_11731


namespace value_of_expression_l11_11194

theorem value_of_expression : 
  ∀ (x y : ℤ), x = -5 → y = -10 → (y - x) * (y + x) = 75 :=
by
  intros x y hx hy
  rw [hx, hy]
  sorry

end value_of_expression_l11_11194


namespace mathematician_correctness_l11_11104

theorem mathematician_correctness (box1_white1 box1_total1 box2_white1 box2_total1 : ℕ)
                                  (prob1 prob2 : ℚ) :
  (box1_white1 = 4 ∧ box1_total1 = 7 ∧ box2_white1 = 3 ∧ box2_total1 = 5 ∧ prob1 = (4:ℚ) / 7 ∧ prob2 = (3:ℚ) / 5) →
  (box1_white2 = 8 ∧ box1_total2 = 14 ∧ box2_white2 = 3 ∧ box2_total2 = 5 ∧ prob1 = (8:ℚ) / 14 ∧ prob2 = (3:ℚ) / 5) →
  (prob1 < (7:ℚ) / 12 ∧ prob2 > (11:ℚ)/19) →
  ¬((19:ℚ)/35 > (4:ℚ)/7 ∧ (19:ℚ)/35 < (3:ℚ)/5) :=
by
  sorry

end mathematician_correctness_l11_11104


namespace find_sum_of_common_ratios_l11_11873

-- Definition of the problem conditions
def is_geometric_sequence (a b c : ℕ) (k : ℕ) (r : ℕ) : Prop :=
  b = k * r ∧ c = k * r * r

-- Main theorem statement
theorem find_sum_of_common_ratios (k p r a_2 a_3 b_2 b_3 : ℕ) 
  (hk : k ≠ 0)
  (hp_neq_r : p ≠ r)
  (hp_seq : is_geometric_sequence k a_2 a_3 k p)
  (hr_seq : is_geometric_sequence k b_2 b_3 k r)
  (h_eq : a_3 - b_3 = 3 * (a_2 - b_2)) :
  p + r = 3 :=
sorry

end find_sum_of_common_ratios_l11_11873


namespace rainfall_second_week_l11_11108

theorem rainfall_second_week (r1 r2 : ℝ) (h1 : r1 + r2 = 40) (h2 : r2 = 1.5 * r1) : r2 = 24 :=
by
  sorry

end rainfall_second_week_l11_11108


namespace wendy_dentist_bill_l11_11754

theorem wendy_dentist_bill : 
  let cost_cleaning := 70
  let cost_filling := 120
  let num_fillings := 3
  let cost_root_canal := 400
  let cost_dental_crown := 600
  let total_bill := 9 * cost_root_canal
  let known_costs := cost_cleaning + (num_fillings * cost_filling) + cost_root_canal + cost_dental_crown
  let cost_tooth_extraction := total_bill - known_costs
  cost_tooth_extraction = 2170 := by
  sorry

end wendy_dentist_bill_l11_11754


namespace num_accompanying_year_2022_l11_11851

theorem num_accompanying_year_2022 : 
  ∃ N : ℤ, (N = 2) ∧ 
    (∀ n : ℤ, (100 * n + 22) % n = 0 ∧ 10 ≤ n ∧ n < 100 → n = 11 ∨ n = 22) :=
by 
  sorry

end num_accompanying_year_2022_l11_11851


namespace distance_between_A_and_B_l11_11222

theorem distance_between_A_and_B (x : ℝ) (boat_speed : ℝ) (flow_speed : ℝ) (dist_AC : ℝ) (total_time : ℝ) :
  (boat_speed = 8) →
  (flow_speed = 2) →
  (dist_AC = 2) →
  (total_time = 3) →
  (x = 10 ∨ x = 12.5) :=
by {
  sorry
}

end distance_between_A_and_B_l11_11222


namespace minimum_value_of_expression_l11_11204

theorem minimum_value_of_expression (x A B C : ℝ) (hx : x > 0) 
  (hA : A = x^2 + 1/x^2) (hB : B = x - 1/x) (hC : C = B * (A + 1)) : 
  ∃ m : ℝ, m = 6.4 ∧ m = A^3 / C :=
by {
  sorry
}

end minimum_value_of_expression_l11_11204


namespace incorrect_statement_for_proportional_function_l11_11026

theorem incorrect_statement_for_proportional_function (x y : ℝ) : y = -5 * x →
  ¬ (∀ x, (x > 0 → y > 0) ∧ (x < 0 → y < 0)) :=
by
  sorry

end incorrect_statement_for_proportional_function_l11_11026


namespace total_coffee_blend_cost_l11_11377

-- Define the cost per pound of coffee types A and B
def cost_per_pound_A := 4.60
def cost_per_pound_B := 5.95

-- Given the pounds of coffee for Type A and the blend condition for Type B
def pounds_A := 67.52
def pounds_B := 2 * pounds_A

-- Total cost calculation
def total_cost := (pounds_A * cost_per_pound_A) + (pounds_B * cost_per_pound_B)

-- Theorem statement: The total cost of the coffee blend is $1114.08
theorem total_coffee_blend_cost : total_cost = 1114.08 := by
  -- Proof omitted
  sorry

end total_coffee_blend_cost_l11_11377


namespace merck_hourly_rate_l11_11889

-- Define the relevant data from the problem
def hours_donaldsons : ℕ := 7
def hours_merck : ℕ := 6
def hours_hille : ℕ := 3
def total_earnings : ℕ := 273

-- Define the total hours based on the conditions
def total_hours : ℕ := hours_donaldsons + hours_merck + hours_hille

-- Define what we want to prove:
def hourly_rate := total_earnings / total_hours

theorem merck_hourly_rate : hourly_rate = 273 / (7 + 6 + 3) := by
  sorry

end merck_hourly_rate_l11_11889


namespace find_a_in_triangle_l11_11885

theorem find_a_in_triangle (a b c : ℝ) (A B C : ℝ) 
  (h1 : b^2 - c^2 + 2 * a = 0) 
  (h2 : Real.tan C / Real.tan B = 3) 
  : a = 4 :=
  sorry

end find_a_in_triangle_l11_11885


namespace vector_decomposition_l11_11354

def x : ℝ×ℝ×ℝ := (8, 0, 5)
def p : ℝ×ℝ×ℝ := (2, 0, 1)
def q : ℝ×ℝ×ℝ := (1, 1, 0)
def r : ℝ×ℝ×ℝ := (4, 1, 2)

theorem vector_decomposition :
  x = (1:ℝ) • p + (-2:ℝ) • q + (2:ℝ) • r :=
by
  sorry

end vector_decomposition_l11_11354


namespace find_n_l11_11768

theorem find_n (n : ℕ) (h1 : Nat.lcm n 14 = 56) (h2 : Nat.gcd n 14 = 10) : n = 40 :=
by
  sorry

end find_n_l11_11768


namespace fee_difference_l11_11535

-- Defining the given conditions
def stadium_capacity : ℕ := 2000
def fraction_full : ℚ := 3 / 4
def entry_fee : ℚ := 20

-- Statement to prove
theorem fee_difference :
  let people_at_three_quarters := stadium_capacity * fraction_full
  let total_fees_at_three_quarters := people_at_three_quarters * entry_fee
  let total_fees_full := stadium_capacity * entry_fee
  total_fees_full - total_fees_at_three_quarters = 10000 :=
by
  sorry

end fee_difference_l11_11535


namespace y_intercept_of_line_l11_11462

def line_equation (x y : ℝ) : Prop := x - 2 * y + 4 = 0

theorem y_intercept_of_line : ∀ y : ℝ, line_equation 0 y → y = 2 :=
by 
  intro y h
  unfold line_equation at h
  sorry

end y_intercept_of_line_l11_11462


namespace European_to_American_swallow_ratio_l11_11482

theorem European_to_American_swallow_ratio (a e : ℝ) (n_E : ℕ) 
  (h1 : a = 5)
  (h2 : 2 * n_E + n_E = 90)
  (h3 : 60 * a + 30 * e = 600) :
  e / a = 2 := 
by
  sorry

end European_to_American_swallow_ratio_l11_11482


namespace boys_neither_happy_nor_sad_correct_l11_11410

def total_children : ℕ := 60
def happy_children : ℕ := 30
def sad_children : ℕ := 10
def neither_happy_nor_sad_children : ℕ := 20
def total_boys : ℕ := 16
def total_girls : ℕ := 44
def happy_boys : ℕ := 6
def sad_girls : ℕ := 4

-- The number of boys who are neither happy nor sad
def boys_neither_happy_nor_sad : ℕ :=
  total_boys - happy_boys - (sad_children - sad_girls)

theorem boys_neither_happy_nor_sad_correct : boys_neither_happy_nor_sad = 4 := by
  sorry

end boys_neither_happy_nor_sad_correct_l11_11410


namespace largest_natural_S_n_gt_zero_l11_11112

noncomputable def S_n (n : ℕ) : ℤ :=
  let a1 := 9
  let d := -2
  n * (2 * a1 + (n - 1) * d) / 2

theorem largest_natural_S_n_gt_zero
  (a_2 : ℤ) (a_4 : ℤ)
  (h1 : a_2 = 7) (h2 : a_4 = 3) :
  ∃ n : ℕ, S_n n > 0 ∧ ∀ m : ℕ, m > n → S_n m ≤ 0 := 
sorry

end largest_natural_S_n_gt_zero_l11_11112


namespace factorization_of_z6_minus_64_l11_11812

theorem factorization_of_z6_minus_64 :
  ∀ (z : ℝ), (z^6 - 64) = (z - 2) * (z^2 + 2*z + 4) * (z + 2) * (z^2 - 2*z + 4) := 
by
  intros z
  sorry

end factorization_of_z6_minus_64_l11_11812


namespace arithmetic_geometric_sequence_l11_11534

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) (d : ℝ) (h1 : a 1 = 1)
  (h2 : ∀ n, a (n + 1) = a n + d)
  (h3 : a 2 * a 5 = (a 4) ^ 2)
  (h4 : d ≠ 0) : d = -1 / 5 :=
by
  sorry

end arithmetic_geometric_sequence_l11_11534


namespace mikey_jelly_beans_l11_11908

theorem mikey_jelly_beans :
  let napoleon_jelly_beans := 17
  let sedrich_jelly_beans := napoleon_jelly_beans + 4
  let total_jelly_beans := napoleon_jelly_beans + sedrich_jelly_beans
  let twice_sum := 2 * total_jelly_beans
  ∃ mikey_jelly_beans, 4 * mikey_jelly_beans = twice_sum → mikey_jelly_beans = 19 :=
by
  intro napoleon_jelly_beans
  intro sedrich_jelly_beans
  intro total_jelly_beans
  intro twice_sum
  use 19
  sorry

end mikey_jelly_beans_l11_11908


namespace train_speed_in_kmph_l11_11177

-- Definitions based on the conditions
def train_length : ℝ := 280 -- in meters
def time_to_pass_tree : ℝ := 28 -- in seconds

-- Conversion factor from meters/second to kilometers/hour
def mps_to_kmph : ℝ := 3.6

-- The speed of the train in kilometers per hour
theorem train_speed_in_kmph : (train_length / time_to_pass_tree) * mps_to_kmph = 36 := 
sorry

end train_speed_in_kmph_l11_11177


namespace value_depletion_rate_l11_11349

theorem value_depletion_rate (V_initial V_final : ℝ) (t : ℝ) (r : ℝ) :
  V_initial = 900 → V_final = 729 → t = 2 → V_final = V_initial * (1 - r)^t → r = 0.1 :=
by sorry

end value_depletion_rate_l11_11349


namespace common_chord_l11_11116

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + x - 2*y - 20 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 = 25

-- The common chord is the line where both circle equations are satisfied
theorem common_chord (x y : ℝ) : circle1 x y ∧ circle2 x y → x - 2*y + 5 = 0 :=
sorry

end common_chord_l11_11116


namespace digging_project_length_l11_11770

theorem digging_project_length (L : ℝ) (V1 V2 : ℝ) (depth1 length1 depth2 breadth1 breadth2 : ℝ) 
  (h1 : depth1 = 100) (h2 : length1 = 25) (h3 : breadth1 = 30) (h4 : V1 = depth1 * length1 * breadth1)
  (h5 : depth2 = 75) (h6 : breadth2 = 50) (h7 : V2 = depth2 * L * breadth2) (h8 : V1 / V2 = 1) :
  L = 20 :=
by
  sorry

end digging_project_length_l11_11770


namespace fewest_coach_handshakes_l11_11688

theorem fewest_coach_handshakes (n m1 m2 : ℕ) 
  (handshakes_total : (n * (n - 1)) / 2 + m1 + m2 = 465) 
  (m1_m2_eq_n : m1 + m2 = n) : 
  n * (n - 1) / 2 = 465 → m1 + m2 = 0 :=
by 
  sorry

end fewest_coach_handshakes_l11_11688


namespace average_calculation_l11_11525

def average_two (a b : ℚ) : ℚ := (a + b) / 2
def average_three (a b c : ℚ) : ℚ := (a + b + c) / 3

theorem average_calculation :
  average_three (average_three 2 2 0) (average_two 1 2) 1 = 23 / 18 :=
by sorry

end average_calculation_l11_11525


namespace tan_of_angle_in_second_quadrant_l11_11262

theorem tan_of_angle_in_second_quadrant (α : ℝ) (hα1 : π / 2 < α ∧ α < π) (hα2 : Real.cos (π / 2 - α) = 4 / 5) : Real.tan α = -4 / 3 :=
by
  sorry

end tan_of_angle_in_second_quadrant_l11_11262


namespace average_weight_of_children_l11_11758

theorem average_weight_of_children 
  (average_weight_boys : ℝ)
  (number_of_boys : ℕ)
  (average_weight_girls : ℝ)
  (number_of_girls : ℕ)
  (total_children : ℕ)
  (average_weight_children : ℝ) :
  average_weight_boys = 160 →
  number_of_boys = 8 →
  average_weight_girls = 130 →
  number_of_girls = 6 →
  total_children = number_of_boys + number_of_girls →
  average_weight_children = 
    (number_of_boys * average_weight_boys + number_of_girls * average_weight_girls) / total_children →
  average_weight_children = 147 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end average_weight_of_children_l11_11758


namespace mutually_exclusive_A_B_head_l11_11861

variables (A_head B_head B_end : Prop)

def mut_exclusive (P Q : Prop) : Prop := ¬(P ∧ Q)

theorem mutually_exclusive_A_B_head (A_head B_head : Prop) :
  mut_exclusive A_head B_head :=
sorry

end mutually_exclusive_A_B_head_l11_11861


namespace ratio_of_areas_of_triangles_l11_11815

theorem ratio_of_areas_of_triangles 
  (a b c d e f : ℕ)
  (h1 : a = 7) (h2 : b = 24) (h3 : c = 25)
  (h4 : d = 9) (h5 : e = 40) (h6 : f = 41) : 
  (84 : ℚ) / (180 : ℚ) = 7 / 15 := by
  have hPQR : a^2 + b^2 = c^2 := by
    rw [h1, h2, h3]
    norm_num
  have hSTU : d^2 + e^2 = f^2 := by
    rw [h4, h5, h6]
    norm_num
  have areaPQR : (1/2 : ℚ) * a * b = 84 := by
    rw [h1, h2]
    norm_num
  have areaSTU : (1/2 : ℚ) * d * e = 180 := by
    rw [h4, h5]
    norm_num
  sorry

end ratio_of_areas_of_triangles_l11_11815


namespace total_carrots_l11_11580

theorem total_carrots (sally_carrots fred_carrots mary_carrots : ℕ)
  (h_sally : sally_carrots = 6)
  (h_fred : fred_carrots = 4)
  (h_mary : mary_carrots = 10) :
  sally_carrots + fred_carrots + mary_carrots = 20 := 
by sorry

end total_carrots_l11_11580


namespace hyperbola_eqn_correct_l11_11517

def parabola_focus : ℝ × ℝ := (1, 0)

def hyperbola_vertex := parabola_focus

def hyperbola_eccentricity : ℝ := 2

def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 - (y^2 / 3) = 1

theorem hyperbola_eqn_correct (x y : ℝ) :
  hyperbola_equation x y :=
sorry

end hyperbola_eqn_correct_l11_11517


namespace parabola_equation_trajectory_midpoint_l11_11193

-- Given data and conditions
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2 * p * x
def point_on_parabola_x3 (p : ℝ) : Prop := ∃ y, parabola p 3 y
def distance_point_to_line (x d : ℝ) : Prop := x + d = 5

-- Prove that given these conditions, the parabola equation is y^2 = 8x
theorem parabola_equation (p : ℝ) (h1 : point_on_parabola_x3 p) (h2 : distance_point_to_line (3 + p / 2) 2) : p = 4 :=
sorry

-- Prove the equation of the trajectory for the midpoint of the line segment FP
def point_on_parabola (p x y : ℝ) : Prop := y^2 = 8 * x
theorem trajectory_midpoint (p x y : ℝ) (h1 : parabola 4 x y) : y^2 = 4 * (x - 1) :=
sorry

end parabola_equation_trajectory_midpoint_l11_11193


namespace distinct_three_digit_numbers_count_l11_11347

theorem distinct_three_digit_numbers_count : 
  ∃! n : ℕ, n = 5 * 4 * 3 :=
by
  use 60
  sorry

end distinct_three_digit_numbers_count_l11_11347


namespace ellipse_equation_hyperbola_equation_l11_11476

/-- Ellipse problem -/
def ellipse_eq (x y a b : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

theorem ellipse_equation (e a c b : ℝ) (h_c : c = 3) (h_e : e = 0.5) (h_a : a = 6) (h_b : b^2 = 27) :
  ellipse_eq x y a b := 
sorry

/-- Hyperbola problem -/
def hyperbola_eq (x y a b : ℝ) : Prop :=
  (x^2 / a^2) - (y^2 / b^2) = 1

theorem hyperbola_equation (a b c : ℝ) 
  (h_c : c = 6) 
  (h_A : ∀ (x y : ℝ), (x, y) = (-5, 2) → hyperbola_eq x y a b) 
  (h_eq1 : a^2 + b^2 = 36) 
  (h_eq2 : 25 / (a^2) - 4 / (b^2) = 1) :
  hyperbola_eq x y a b :=
sorry

end ellipse_equation_hyperbola_equation_l11_11476


namespace range_of_m_for_hyperbola_l11_11728

theorem range_of_m_for_hyperbola (m : ℝ) :
  (∃ u v : ℝ, (∀ x y : ℝ, x^2/(m+2) + y^2/(m+1) = 1) → (m > -2) ∧ (m < -1)) := by
  sorry

end range_of_m_for_hyperbola_l11_11728


namespace probability_of_white_ball_l11_11944

theorem probability_of_white_ball (red_balls white_balls : ℕ) (draws : ℕ)
    (h_red : red_balls = 4) (h_white : white_balls = 2) (h_draws : draws = 2) :
    ((4 * 2 + 1) / 15 : ℚ) = 3 / 5 := by sorry

end probability_of_white_ball_l11_11944


namespace figure_100_squares_l11_11373

theorem figure_100_squares :
  ∀ (f : ℕ → ℕ),
    (f 0 = 1) →
    (f 1 = 6) →
    (f 2 = 17) →
    (f 3 = 34) →
    f 100 = 30201 :=
by
  intros f h0 h1 h2 h3
  sorry

end figure_100_squares_l11_11373


namespace xiaomings_possible_score_l11_11913

def average_score_class_A : ℤ := 87
def average_score_class_B : ℤ := 82

theorem xiaomings_possible_score (x : ℤ) :
  (average_score_class_B < x ∧ x < average_score_class_A) → x = 85 :=
by sorry

end xiaomings_possible_score_l11_11913


namespace largest_non_summable_composite_l11_11962

def is_composite (n : ℕ) : Prop :=
  ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

def can_be_sum_of_two_composites (n : ℕ) : Prop :=
  ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b

theorem largest_non_summable_composite : ∀ m : ℕ, (m < 11 → ¬ can_be_sum_of_two_composites m) ∧ (m ≥ 11 → can_be_sum_of_two_composites m) :=
by sorry

end largest_non_summable_composite_l11_11962


namespace min_value_circles_tangents_l11_11806

theorem min_value_circles_tangents (a b : ℝ) (h1 : (∃ x y : ℝ, x^2 + y^2 + 2 * a * x + a^2 - 4 = 0) ∧ 
  (∃ x y : ℝ, x^2 + y^2 - 4 * b * y - 1 + 4 * b^2 = 0))
  (h2 : ∃ k : ℕ, k = 3) (h3 : a ≠ 0) (h4 : b ≠ 0) : 
  (∃ m : ℝ, m = 1 ∧  ∀ x : ℝ, (x = (1 / a^2) + (1 / b^2)) → x ≥ m) :=
  sorry

end min_value_circles_tangents_l11_11806


namespace car_speeds_l11_11009

noncomputable def distance_between_places : ℝ := 135
noncomputable def departure_time_diff : ℝ := 4 -- large car departs 4 hours before small car
noncomputable def arrival_time_diff : ℝ := 0.5 -- small car arrives 30 minutes earlier than large car
noncomputable def speed_ratio : ℝ := 5 / 2 -- ratio of speeds (small car : large car)

theorem car_speeds (v_small v_large : ℝ) (h1 : v_small / v_large = speed_ratio) :
    v_small = 45 ∧ v_large = 18 :=
sorry

end car_speeds_l11_11009


namespace cos_double_angle_l11_11089

theorem cos_double_angle (α : ℝ) (h : Real.sin (Real.pi + α) = 1 / 3) : Real.cos (2 * α) = 7 / 9 := 
by 
  sorry

end cos_double_angle_l11_11089


namespace gcf_252_96_l11_11845

theorem gcf_252_96 : Int.gcd 252 96 = 12 := by
  sorry

end gcf_252_96_l11_11845


namespace trisect_angle_l11_11538

noncomputable def can_trisect_with_ruler_and_compasses (n : ℕ) : Prop :=
  ¬(3 ∣ n) → ∃ a b : ℤ, 3 * a + n * b = 1

theorem trisect_angle (n : ℕ) (h : ¬(3 ∣ n)) :
  can_trisect_with_ruler_and_compasses n :=
sorry

end trisect_angle_l11_11538


namespace ten_times_average_letters_l11_11439

-- Define the number of letters Elida has
def letters_Elida : ℕ := 5

-- Define the number of letters Adrianna has
def letters_Adrianna : ℕ := 2 * letters_Elida - 2

-- Define the average number of letters in both names
def average_letters : ℕ := (letters_Elida + letters_Adrianna) / 2

-- Define the final statement for 10 times the average number of letters
theorem ten_times_average_letters : 10 * average_letters = 65 := by
  sorry

end ten_times_average_letters_l11_11439


namespace bike_owners_without_car_l11_11549

variable (T B C : ℕ) (H1 : T = 500) (H2 : B = 450) (H3 : C = 200)

theorem bike_owners_without_car (total bike_owners car_owners : ℕ) 
  (h_total : total = 500) (h_bike_owners : bike_owners = 450) (h_car_owners : car_owners = 200) : 
  (bike_owners - (bike_owners + car_owners - total)) = 300 := by
  sorry

end bike_owners_without_car_l11_11549


namespace remainder_23_2057_mod_25_l11_11707

theorem remainder_23_2057_mod_25 : (23^2057) % 25 = 16 := 
by
  sorry

end remainder_23_2057_mod_25_l11_11707


namespace scientific_notation_correct_l11_11124

theorem scientific_notation_correct :
  0.00000164 = 1.64 * 10^(-6) :=
sorry

end scientific_notation_correct_l11_11124


namespace room_width_is_12_l11_11477

variable (w : ℝ)

def length_of_room : ℝ := 20
def width_of_veranda : ℝ := 2
def area_of_veranda : ℝ := 144

theorem room_width_is_12 :
  24 * (w + 4) - 20 * w = 144 → w = 12 := by
  sorry

end room_width_is_12_l11_11477


namespace history_paper_pages_l11_11018

theorem history_paper_pages (p d : ℕ) (h1 : p = 11) (h2 : d = 3) : p * d = 33 :=
by
  sorry

end history_paper_pages_l11_11018


namespace ab_value_l11_11932

theorem ab_value (a b : ℝ) (h1 : a + b = 8) (h2 : a^3 + b^3 = 107) : a * b = 25.3125 :=
by
  sorry

end ab_value_l11_11932


namespace rectangle_perimeter_l11_11084

theorem rectangle_perimeter (l d : ℝ) (h_l : l = 8) (h_d : d = 17) :
  ∃ w : ℝ, (d^2 = l^2 + w^2) ∧ (2*l + 2*w = 46) :=
by
  sorry

end rectangle_perimeter_l11_11084


namespace rick_books_total_l11_11401

theorem rick_books_total 
  (N : ℕ)
  (h : N / 16 = 25) : 
  N = 400 := 
  sorry

end rick_books_total_l11_11401


namespace new_average_daily_production_l11_11003

theorem new_average_daily_production 
  (n : ℕ) 
  (avg_past_n_days : ℕ) 
  (today_production : ℕ)
  (new_avg_production : ℕ)
  (hn : n = 5) 
  (havg : avg_past_n_days = 60) 
  (htoday : today_production = 90) 
  (hnew_avg : new_avg_production = 65)
  : (n + 1 = 6) ∧ ((n * 60 + today_production) = 390) ∧ (390 / 6 = 65) :=
by
  sorry

end new_average_daily_production_l11_11003


namespace fraction_of_second_year_given_not_third_year_l11_11931

theorem fraction_of_second_year_given_not_third_year (total_students : ℕ) 
  (third_year_students : ℕ) (second_year_students : ℕ) :
  third_year_students = total_students * 30 / 100 →
  second_year_students = total_students * 10 / 100 →
  ↑second_year_students / (total_students - third_year_students) = (1 : ℚ) / 7 :=
by
  -- Proof omitted
  sorry

end fraction_of_second_year_given_not_third_year_l11_11931


namespace reflection_eq_l11_11000

theorem reflection_eq (x y : ℝ) : 
    let line_eq (x y : ℝ) := 2 * x + 3 * y - 5 = 0 
    let reflection_eq (x y : ℝ) := 3 * x + 2 * y - 5 = 0 
    (∀ (x y : ℝ), line_eq x y ↔ reflection_eq y x) →
    reflection_eq x y :=
by
    sorry

end reflection_eq_l11_11000


namespace largest_hole_leakage_rate_l11_11700

theorem largest_hole_leakage_rate (L : ℝ) (h1 : 600 = (L + L / 2 + L / 6) * 120) : 
  L = 3 :=
sorry

end largest_hole_leakage_rate_l11_11700


namespace final_price_correct_l11_11117

def original_price : Float := 100
def store_discount_rate : Float := 0.20
def promo_discount_rate : Float := 0.10
def sales_tax_rate : Float := 0.05
def handling_fee : Float := 5

def final_price (original_price : Float) 
                (store_discount_rate : Float) 
                (promo_discount_rate : Float) 
                (sales_tax_rate : Float) 
                (handling_fee : Float) 
                : Float :=
  let price_after_store_discount := original_price * (1 - store_discount_rate)
  let price_after_promo := price_after_store_discount * (1 - promo_discount_rate)
  let price_after_tax := price_after_promo * (1 + sales_tax_rate)
  let total_price := price_after_tax + handling_fee
  total_price

theorem final_price_correct : final_price original_price store_discount_rate promo_discount_rate sales_tax_rate handling_fee = 80.60 :=
by
  simp only [
    original_price,
    store_discount_rate,
    promo_discount_rate,
    sales_tax_rate,
    handling_fee
  ]
  norm_num
  sorry

end final_price_correct_l11_11117


namespace fifth_friend_payment_l11_11010

def contributions (a b c d e : ℕ) : Prop :=
  a + b + c + d + e = 120 ∧
  a = (1 / 3 : ℕ) * (b + c + d + e) ∧
  b = (1 / 4 : ℕ) * (a + c + d + e) ∧
  c = (1 / 5 : ℕ) * (a + b + d + e)

theorem fifth_friend_payment (a b c d e : ℕ) (h : contributions a b c d e) : e = 13 :=
sorry

end fifth_friend_payment_l11_11010


namespace penguins_remaining_to_get_fish_l11_11398

def total_penguins : Nat := 36
def fed_penguins : Nat := 19

theorem penguins_remaining_to_get_fish : (total_penguins - fed_penguins = 17) :=
by
  sorry

end penguins_remaining_to_get_fish_l11_11398


namespace zeros_indeterminate_in_interval_l11_11599

noncomputable def f : ℝ → ℝ := sorry

variables (a b : ℝ) (ha : a < b) (hf : f a * f b < 0)

-- The theorem statement
theorem zeros_indeterminate_in_interval :
  (∀ (f : ℝ → ℝ), f a * f b < 0 → (∃ (x : ℝ), a < x ∧ x < b ∧ f x = 0) ∨ (∀ (x : ℝ), a < x ∧ x < b → f x ≠ 0) ∨ (∃ (x1 x2 : ℝ), a < x1 ∧ x1 < x2 ∧ x2 < b ∧ f x1 = 0 ∧ f x2 = 0)) :=
by sorry

end zeros_indeterminate_in_interval_l11_11599


namespace smallest_factorization_c_l11_11423

theorem smallest_factorization_c : ∃ (c : ℤ), (∀ (r s : ℤ), r * s = 2016 → r + s = c) ∧ c > 0 ∧ c = 108 :=
by 
  sorry

end smallest_factorization_c_l11_11423


namespace oranges_worth_as_much_as_bananas_l11_11570

-- Define the given conditions
def worth_same_bananas_oranges (bananas oranges : ℕ) : Prop :=
  (3 / 4 * 12 : ℝ) = 9 ∧ 9 = 6

/-- Prove how many oranges are worth as much as (2 / 3) * 9 bananas,
    given that (3 / 4) * 12 bananas are worth 6 oranges. -/
theorem oranges_worth_as_much_as_bananas :
  worth_same_bananas_oranges 12 6 →
  (2 / 3 * 9 : ℝ) = 4 :=
by
  sorry

end oranges_worth_as_much_as_bananas_l11_11570


namespace shaded_area_l11_11049

-- Let A be the length of the side of the smaller square
def A : ℝ := 4

-- Let B be the length of the side of the larger square
def B : ℝ := 12

-- The problem is to prove that the area of the shaded region is 10 square inches
theorem shaded_area (A B : ℝ) (hA : A = 4) (hB : B = 12) :
  (A * A) - (1/2 * (B / (B + A)) * A * B) = 10 := by
  sorry

end shaded_area_l11_11049


namespace factor_expression_l11_11464

theorem factor_expression (x : ℝ) : 
  3 * x * (x - 5) + 4 * (x - 5) = (3 * x + 4) * (x - 5) :=
by
  sorry

end factor_expression_l11_11464


namespace solve_for_m_l11_11269

-- Define the operation ◎ for real numbers a and b
def op (a b : ℝ) : ℝ := (a + b)^2 - (a - b)^2

-- Lean statement for the proof problem
theorem solve_for_m (m : ℝ) (h : op (m + 1) (m - 2) = 16) : m = 3 ∨ m = -2 :=
sorry

end solve_for_m_l11_11269


namespace total_fish_correct_l11_11625

-- Define the number of pufferfish
def num_pufferfish : ℕ := 15

-- Define the number of swordfish as 5 times the number of pufferfish
def num_swordfish : ℕ := 5 * num_pufferfish

-- Define the total number of fish as the sum of pufferfish and swordfish
def total_num_fish : ℕ := num_pufferfish + num_swordfish

-- Theorem stating the total number of fish
theorem total_fish_correct : total_num_fish = 90 := by
  -- Proof is omitted
  sorry

end total_fish_correct_l11_11625


namespace C_investment_value_is_correct_l11_11495

noncomputable def C_investment_contribution 
  (A_investment B_investment total_profit A_profit_share : ℝ) : ℝ :=
  let C_investment := 
    (A_profit_share * (A_investment + B_investment) - A_investment * total_profit) / 
    (total_profit - A_profit_share)
  C_investment

theorem C_investment_value_is_correct : 
  C_investment_contribution 6300 4200 13600 4080 = 10500 := 
by
  unfold C_investment_contribution
  norm_num
  sorry

end C_investment_value_is_correct_l11_11495


namespace remaining_amount_to_pay_l11_11853

-- Define the constants and conditions
def total_cost : ℝ := 1300
def first_deposit : ℝ := 0.10 * total_cost
def second_deposit : ℝ := 2 * first_deposit
def promotional_discount : ℝ := 0.05 * total_cost
def interest_rate : ℝ := 0.02

-- Define the function to calculate the final payment
def final_payment (total_cost first_deposit second_deposit promotional_discount interest_rate : ℝ) : ℝ :=
  let total_paid := first_deposit + second_deposit
  let remaining_balance := total_cost - total_paid
  let remaining_after_discount := remaining_balance - promotional_discount
  remaining_after_discount * (1 + interest_rate)

-- Define the theorem to be proven
theorem remaining_amount_to_pay :
  final_payment total_cost first_deposit second_deposit promotional_discount interest_rate = 861.90 :=
by
  -- The proof goes here
  sorry

end remaining_amount_to_pay_l11_11853


namespace find_k_and_a_l11_11783

noncomputable def polynomial_P : Polynomial ℝ := Polynomial.C 5 + Polynomial.X * (Polynomial.C (-18) + Polynomial.X * (Polynomial.C 13 + Polynomial.X * (Polynomial.C (-4) + Polynomial.X)))
noncomputable def polynomial_D (k : ℝ) : Polynomial ℝ := Polynomial.C k + Polynomial.X * (Polynomial.C (-1) + Polynomial.X)
noncomputable def polynomial_R (a : ℝ) : Polynomial ℝ := Polynomial.C a + (Polynomial.C 2 * Polynomial.X)

theorem find_k_and_a : 
  ∃ k a : ℝ, polynomial_P = polynomial_D k * Polynomial.C 1 + polynomial_R a ∧ k = 10 ∧ a = 5 :=
sorry

end find_k_and_a_l11_11783


namespace solve_system_l11_11507

theorem solve_system (x y z a b c : ℝ)
  (h1 : x * (x + y + z) = a^2)
  (h2 : y * (x + y + z) = b^2)
  (h3 : z * (x + y + z) = c^2) :
  (x = a^2 / Real.sqrt (a^2 + b^2 + c^2) ∨ x = -a^2 / Real.sqrt (a^2 + b^2 + c^2)) ∧
  (y = b^2 / Real.sqrt (a^2 + b^2 + c^2) ∨ y = -b^2 / Real.sqrt (a^2 + b^2 + c^2)) ∧
  (z = c^2 / Real.sqrt (a^2 + b^2 + c^2) ∨ z = -c^2 / Real.sqrt (a^2 + b^2 + c^2)) :=
by
  sorry

end solve_system_l11_11507


namespace encyclopedia_total_pages_l11_11762

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

end encyclopedia_total_pages_l11_11762


namespace buffaloes_added_l11_11993

-- Let B be the daily fodder consumption of one buffalo in units
noncomputable def daily_fodder_buffalo (B : ℝ) := B
noncomputable def daily_fodder_cow (B : ℝ) := (3 / 4) * B
noncomputable def daily_fodder_ox (B : ℝ) := (3 / 2) * B

-- Initial conditions
def initial_buffaloes := 15
def initial_cows := 24
def initial_oxen := 8
def initial_days := 24
noncomputable def total_initial_fodder (B : ℝ) := (initial_buffaloes * daily_fodder_buffalo B) + (initial_oxen * daily_fodder_ox B) + (initial_cows * daily_fodder_cow B)
noncomputable def total_fodder (B : ℝ) := total_initial_fodder B * initial_days

-- New conditions after adding cows and buffaloes
def additional_cows := 60
def new_days := 9
noncomputable def total_new_daily_fodder (B : ℝ) (x : ℝ) := ((initial_buffaloes + x) * daily_fodder_buffalo B) + (initial_oxen * daily_fodder_ox B) + ((initial_cows + additional_cows) * daily_fodder_cow B)

-- Proof statement: Prove that given the conditions, the number of additional buffaloes, x, is 30.
theorem buffaloes_added (B : ℝ) : 
  (total_fodder B = total_new_daily_fodder B 30 * new_days) :=
by sorry

end buffaloes_added_l11_11993


namespace rain_at_house_l11_11781

/-- Define the amounts of rain on the three days Greg was camping. -/
def rain_day1 : ℕ := 3
def rain_day2 : ℕ := 6
def rain_day3 : ℕ := 5

/-- Define the total rain experienced by Greg while camping. -/
def total_rain_camping := rain_day1 + rain_day2 + rain_day3

/-- Define the difference in the rain experienced by Greg while camping and at his house. -/
def rain_difference : ℕ := 12

/-- Define the total amount of rain at Greg's house. -/
def total_rain_house := total_rain_camping + rain_difference

/-- Prove that the total rain at Greg's house is 26 mm. -/
theorem rain_at_house : total_rain_house = 26 := by
  /- We know that total_rain_camping = 14 mm and rain_difference = 12 mm -/
  /- Therefore, total_rain_house = 14 mm + 12 mm = 26 mm -/
  sorry

end rain_at_house_l11_11781


namespace fraction_addition_l11_11273

theorem fraction_addition : (3/4) / (5/8) + (1/2) = 17/10 := by
  sorry

end fraction_addition_l11_11273


namespace average_calls_per_day_l11_11335

/-- Conditions: Jean's calls per day -/
def calls_mon : ℕ := 35
def calls_tue : ℕ := 46
def calls_wed : ℕ := 27
def calls_thu : ℕ := 61
def calls_fri : ℕ := 31

/-- Assertion: The average number of calls Jean answers per day -/
theorem average_calls_per_day :
  (calls_mon + calls_tue + calls_wed + calls_thu + calls_fri) / 5 = 40 :=
by sorry

end average_calls_per_day_l11_11335


namespace find_side_length_of_largest_square_l11_11736

theorem find_side_length_of_largest_square (A : ℝ) (hA : A = 810) :
  ∃ a : ℝ, (5 / 8) * a ^ 2 = A ∧ a = 36 := by
  sorry

end find_side_length_of_largest_square_l11_11736


namespace value_of_X_is_one_l11_11386

-- Problem: Given the numbers 28 at the start of a row, 17 in the middle, and -15 in the same column as X,
-- we show the value of X must be 1 because the sequences are arithmetic.

theorem value_of_X_is_one (d : ℤ) (X : ℤ) :
  -- Conditions
  (17 - X = d) ∧ 
  (X - (-15) = d) ∧ 
  (d = 16) →
  -- Conclusion: X must be 1
  X = 1 :=
by 
  sorry

end value_of_X_is_one_l11_11386


namespace impossible_to_have_same_number_of_each_color_l11_11137

-- Define the initial number of coins Laura has
def initial_green : Nat := 1

-- Define the net gain in coins per transaction
def coins_gain_per_transaction : Nat := 4

-- Define a function that calculates the total number of coins after n transactions
def total_coins (n : Nat) : Nat :=
  initial_green + n * coins_gain_per_transaction

-- Define the theorem to prove that it's impossible for Laura to have the same number of red and green coins
theorem impossible_to_have_same_number_of_each_color :
  ¬ ∃ n : Nat, ∃ red green : Nat, red = green ∧ total_coins n = red + green := by
  sorry

end impossible_to_have_same_number_of_each_color_l11_11137


namespace fraction_of_male_fish_l11_11566

def total_fish : ℕ := 45
def female_fish : ℕ := 15
def male_fish := total_fish - female_fish

theorem fraction_of_male_fish : (male_fish : ℚ) / total_fish = 2 / 3 := by
  sorry

end fraction_of_male_fish_l11_11566


namespace probability_same_gate_l11_11817

open Finset

-- Definitions based on the conditions
def num_gates : ℕ := 3
def total_combinations : ℕ := num_gates * num_gates -- total number of combinations for both persons
def favorable_combinations : ℕ := num_gates         -- favorable combinations (both choose same gate)

-- Problem statement
theorem probability_same_gate : 
  ∃ (p : ℚ), p = (favorable_combinations : ℚ) / (total_combinations : ℚ) ∧ p = (1 / 3 : ℚ) := 
by
  sorry

end probability_same_gate_l11_11817


namespace total_cost_l11_11072

def num_of_rings : ℕ := 2

def cost_per_ring : ℕ := 12

theorem total_cost : num_of_rings * cost_per_ring = 24 :=
by sorry

end total_cost_l11_11072


namespace solve_inequality_l11_11973

noncomputable def f : ℝ → ℝ := sorry

def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def increasing_on_nonnegatives (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y

def f_at_one_third (f : ℝ → ℝ) : Prop :=
  f (1/3) = 0

theorem solve_inequality (f : ℝ → ℝ) (x : ℝ) :
  even_function f →
  increasing_on_nonnegatives f →
  f_at_one_third f →
  (0 < x ∧ x < 1/2) ∨ (x > 2) ↔ f (Real.logb (1/8) x) > 0 :=
by
  -- the proof will be filled in here
  sorry

end solve_inequality_l11_11973


namespace width_of_roads_l11_11399

-- Definitions for the conditions
def length_of_lawn := 80 
def breadth_of_lawn := 60 
def total_cost := 5200 
def cost_per_sq_m := 4 

-- Derived condition: total area based on cost
def total_area_by_cost := total_cost / cost_per_sq_m 

-- Statement to prove: width of each road w is 65/7
theorem width_of_roads (w : ℚ) : (80 * w) + (60 * w) = total_area_by_cost → w = 65 / 7 :=
by
  sorry

end width_of_roads_l11_11399


namespace max_value_7a_9b_l11_11500

theorem max_value_7a_9b 
    (r_1 r_2 r_3 a b : ℝ) 
    (h_eq : ∀ x, x^3 - x^2 + a * x - b = 0 → (x = r_1 ∨ x = r_2 ∨ x = r_3))
    (h_root_sum : r_1 + r_2 + r_3 = 1)
    (h_root_prod : r_1 * r_2 * r_3 = b)
    (h_root_sumprod : r_1 * r_2 + r_2 * r_3 + r_3 * r_1 = a)
    (h_bounds : ∀ i, i = r_1 ∨ i = r_2 ∨ i = r_3 → 0 < i ∧ i < 1) :
        7 * a - 9 * b ≤ 2 := 
sorry

end max_value_7a_9b_l11_11500


namespace center_determines_position_l11_11729

structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define what it means for the Circle's position being determined by its center.
theorem center_determines_position (c : Circle) : c.center = c.center :=
by
  sorry

end center_determines_position_l11_11729


namespace negation_proof_l11_11435

theorem negation_proof :
  (¬ (∀ x : ℝ, x^2 + 1 ≥ 2 * x)) ↔ (∃ x : ℝ, x^2 + 1 < 2 * x) :=
by
  sorry

end negation_proof_l11_11435


namespace factorization_sum_l11_11145

variable {a b c : ℤ}

theorem factorization_sum 
  (h1 : ∀ x : ℤ, x^2 + 17 * x + 52 = (x + a) * (x + b))
  (h2 : ∀ x : ℤ, x^2 + 7 * x - 60 = (x + b) * (x - c)) : 
  a + b + c = 27 :=
sorry

end factorization_sum_l11_11145


namespace number_of_people_l11_11964

theorem number_of_people (total_bowls : ℕ) (bowls_per_person : ℚ) : total_bowls = 55 ∧ bowls_per_person = 1 + 1/2 + 1/3 → total_bowls / bowls_per_person = 30 :=
by
  sorry

end number_of_people_l11_11964


namespace correct_calculation_l11_11195

theorem correct_calculation (m n : ℤ) :
  (m^2 * m^3 ≠ m^6) ∧
  (- (m - n) = -m + n) ∧
  (m * (m + n) ≠ m^2 + n) ∧
  ((m + n)^2 ≠ m^2 + n^2) :=
by sorry

end correct_calculation_l11_11195


namespace four_digit_number_8802_l11_11016

theorem four_digit_number_8802 (x : ℕ) (a b c d : ℕ) (h1 : 1000 ≤ x ∧ x ≤ 9999)
  (h2 : x = 1000 * a + 100 * b + 10 * c + d)
  (h3 : a ≠ 0)  -- since a 4-digit number cannot start with 0
  (h4 : a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10) : 
  x + 8802 = 1099 + 8802 :=
by
  sorry

end four_digit_number_8802_l11_11016


namespace scaled_multiplication_l11_11826

theorem scaled_multiplication (h : 213 * 16 = 3408) : 0.016 * 2.13 = 0.03408 :=
by
  sorry

end scaled_multiplication_l11_11826


namespace irreducible_fractions_properties_l11_11892

theorem irreducible_fractions_properties : 
  let f1 := 11 / 2
  let f2 := 11 / 6
  let f3 := 11 / 3
  let reciprocal_sum := (2 / 11) + (6 / 11) + (3 / 11)
  (f1 + f2 + f3 = 11) ∧ (reciprocal_sum = 1) :=
by
  sorry

end irreducible_fractions_properties_l11_11892


namespace pyramid_height_l11_11394

-- Define the heights of individual blocks and the structure of the pyramid.
def block_height := 10 -- in centimeters
def num_layers := 3

-- Define the total height of the pyramid as the sum of the heights of all blocks.
def total_height (block_height : Nat) (num_layers : Nat) := block_height * num_layers

-- The theorem stating that the total height of the stack is 30 cm given the conditions.
theorem pyramid_height : total_height block_height num_layers = 30 := by
  sorry

end pyramid_height_l11_11394


namespace least_possible_value_l11_11925

theorem least_possible_value (x y : ℝ) : (3 * x * y - 1)^2 + (x - y)^2 ≥ 1 := sorry

end least_possible_value_l11_11925


namespace train_crossing_time_l11_11605

theorem train_crossing_time (length_of_train : ℝ) (speed_of_train : ℝ) (speed_of_man : ℝ) :
  length_of_train = 1500 → speed_of_train = 95 → speed_of_man = 5 → 
  (length_of_train / ((speed_of_train - speed_of_man) * (1000 / 3600))) = 60 :=
by
  intros h1 h2 h3
  have h_rel_speed : ((speed_of_train - speed_of_man) * (1000 / 3600)) = 25 := by
    rw [h2, h3]
    norm_num
  rw [h1, h_rel_speed]
  norm_num

end train_crossing_time_l11_11605


namespace percentage_of_boys_from_schoolA_study_science_l11_11011

variable (T : ℝ) -- Total number of boys in the camp
variable (schoolA_boys : ℝ)
variable (science_boys : ℝ)

noncomputable def percentage_science_boys := (science_boys / schoolA_boys) * 100

theorem percentage_of_boys_from_schoolA_study_science 
  (h1 : schoolA_boys = 0.20 * T)
  (h2 : science_boys = schoolA_boys - 56)
  (h3 : T = 400) :
  percentage_science_boys science_boys schoolA_boys = 30 := 
by sorry

end percentage_of_boys_from_schoolA_study_science_l11_11011


namespace cost_of_individual_roll_is_correct_l11_11185

-- Definitions given in the problem's conditions
def cost_per_case : ℝ := 9
def number_of_rolls : ℝ := 12
def percent_savings : ℝ := 0.25

-- The cost of one roll sold individually
noncomputable def individual_roll_cost : ℝ := 0.9375

-- The theorem to prove
theorem cost_of_individual_roll_is_correct :
  individual_roll_cost = (cost_per_case * (1 + percent_savings)) / number_of_rolls :=
by
  sorry

end cost_of_individual_roll_is_correct_l11_11185


namespace loss_percentage_is_25_l11_11699

variables (C S : ℝ)
variables (h : 30 * C = 40 * S)

theorem loss_percentage_is_25 (h : 30 * C = 40 * S) : ((C - S) / C) * 100 = 25 :=
by
  -- proof skipped
  sorry

end loss_percentage_is_25_l11_11699


namespace verify_incorrect_operation_l11_11276

theorem verify_incorrect_operation (a : ℝ) :
  ¬ ((-a^2)^3 = -a^5) :=
by
  sorry

end verify_incorrect_operation_l11_11276


namespace total_vacations_and_classes_l11_11013

def kelvin_classes := 90
def grant_vacations := 4 * kelvin_classes
def total := grant_vacations + kelvin_classes

theorem total_vacations_and_classes :
  total = 450 :=
by
  sorry

end total_vacations_and_classes_l11_11013


namespace rug_shorter_side_l11_11897

theorem rug_shorter_side (x : ℝ) :
  (64 - x * 7) / 64 = 0.78125 → x = 2 :=
by
  sorry

end rug_shorter_side_l11_11897


namespace simplify_expr_l11_11782

noncomputable def expr1 : ℝ := 3 * Real.sqrt 8 / (Real.sqrt 3 + Real.sqrt 2 + Real.sqrt 7)
noncomputable def expr2 : ℝ := -3.6 * (1 + Real.sqrt 2 - 2 * Real.sqrt 7)

theorem simplify_expr : expr1 = expr2 := by
  sorry

end simplify_expr_l11_11782


namespace average_age_increase_l11_11708

variable (A : ℝ) -- Original average age of 8 men
variable (age1 age2 : ℝ) -- The ages of the two men being replaced
variable (avg_women : ℝ) -- The average age of the two women

-- Conditions as hypotheses
def conditions : Prop :=
  8 * A - age1 - age2 + avg_women * 2 = 8 * (A + 2)

-- The theorem that needs to be proved
theorem average_age_increase (h1 : age1 = 20) (h2 : age2 = 28) (h3 : avg_women = 32) (h4 : conditions A age1 age2 avg_women) : (8 * A + 16) / 8 - A = 2 :=
by
  sorry

end average_age_increase_l11_11708


namespace examine_points_l11_11474

variable (Bryan Jen Sammy mistakes : ℕ)

def problem_conditions : Prop :=
  Bryan = 20 ∧ Jen = Bryan + 10 ∧ Sammy = Jen - 2 ∧ mistakes = 7

theorem examine_points (h : problem_conditions Bryan Jen Sammy mistakes) : ∃ total_points : ℕ, total_points = Sammy + mistakes :=
by {
  sorry
}

end examine_points_l11_11474


namespace peter_age_l11_11128

variable (x y : ℕ)

theorem peter_age : 
  (x = (3 * y) / 2) ∧ ((4 * y - x) + 2 * y = 54) → x = 18 :=
by
  intro h
  cases h
  sorry

end peter_age_l11_11128


namespace average_price_per_pen_l11_11146

def total_cost_pens_pencils : ℤ := 690
def number_of_pencils : ℕ := 75
def price_per_pencil : ℤ := 2
def number_of_pens : ℕ := 30

theorem average_price_per_pen :
  (total_cost_pens_pencils - number_of_pencils * price_per_pencil) / number_of_pens = 18 :=
by
  sorry

end average_price_per_pen_l11_11146


namespace adult_tickets_sold_l11_11562

theorem adult_tickets_sold (A S : ℕ) (h1 : S = 3 * A) (h2 : A + S = 600) : A = 150 :=
by
  sorry

end adult_tickets_sold_l11_11562


namespace nancy_total_spending_l11_11646

theorem nancy_total_spending :
  let crystal_bead_price := 9
  let metal_bead_price := 10
  let nancy_crystal_beads := 1
  let nancy_metal_beads := 2
  nancy_crystal_beads * crystal_bead_price + nancy_metal_beads * metal_bead_price = 29 := by
sorry

end nancy_total_spending_l11_11646


namespace split_enthusiasts_into_100_sections_l11_11341

theorem split_enthusiasts_into_100_sections :
  ∃ (sections : Fin 100 → Set ℕ),
    (∀ i, sections i ≠ ∅) ∧
    (∀ i j, i ≠ j → sections i ∩ sections j = ∅) ∧
    (⋃ i, sections i) = {n : ℕ | n < 5000} :=
sorry

end split_enthusiasts_into_100_sections_l11_11341


namespace wendy_albums_l11_11961

theorem wendy_albums (total_pictures remaining_pictures pictures_per_album : ℕ) 
    (h1 : total_pictures = 79)
    (h2 : remaining_pictures = total_pictures - 44)
    (h3 : pictures_per_album = 7) :
    remaining_pictures / pictures_per_album = 5 := by
  sorry

end wendy_albums_l11_11961


namespace distance_between_trees_l11_11970

-- Define the conditions
def yard_length : ℝ := 325
def number_of_trees : ℝ := 26
def number_of_intervals : ℝ := number_of_trees - 1

-- Define what we need to prove
theorem distance_between_trees:
  (yard_length / number_of_intervals) = 13 := 
  sorry

end distance_between_trees_l11_11970


namespace ellipse_equation_range_of_M_x_coordinate_l11_11776

-- Proof 1: Proving the equation of the ellipse
theorem ellipse_equation {a b : ℝ} (h_ab : a > b) (h_b0 : b > 0) (e : ℝ)
  (h_e : e = (Real.sqrt 3) / 3) (vertex : ℝ × ℝ) (h_vertex : vertex = (Real.sqrt 3, 0)) :
  (∃ (a b : ℝ), a > b ∧ b > 0 ∧ e = (Real.sqrt 3) / 3 ∧ vertex = (Real.sqrt 3, 0) ∧ (∀ (x y : ℝ), (x^2) / 3 + (y^2) / 2 = 1)) :=
sorry

-- Proof 2: Proving the range of x-coordinate of point M
theorem range_of_M_x_coordinate (k : ℝ) (h_k : k ≠ 0) :
  (∃ M_x : ℝ, by sorry) :=
sorry


end ellipse_equation_range_of_M_x_coordinate_l11_11776


namespace Duke_three_pointers_impossible_l11_11307

theorem Duke_three_pointers_impossible (old_record : ℤ)
  (points_needed_to_tie : ℤ)
  (points_broken_record : ℤ)
  (free_throws : ℕ)
  (regular_baskets : ℕ)
  (three_pointers : ℕ)
  (normal_three_pointers_per_game : ℕ)
  (max_attempts : ℕ)
  (last_minutes : ℕ)
  (points_per_free_throw : ℤ)
  (points_per_regular_basket : ℤ)
  (points_per_three_pointer : ℤ) :
  free_throws = 5 → regular_baskets = 4 → normal_three_pointers_per_game = 2 → max_attempts = 10 → 
  points_per_free_throw = 1 → points_per_regular_basket = 2 → points_per_three_pointer = 3 →
  old_record = 257 → points_needed_to_tie = 17 → points_broken_record = 5 →
  (free_throws + regular_baskets + three_pointers ≤ max_attempts) →
  last_minutes = 6 → 
  ¬(free_throws + regular_baskets + (points_needed_to_tie + points_broken_record - 
  (free_throws * points_per_free_throw + regular_baskets * points_per_regular_basket)) / points_per_three_pointer ≤ max_attempts) := sorry

end Duke_three_pointers_impossible_l11_11307


namespace solve_equation_l11_11343

theorem solve_equation (x : ℝ) : (x + 4)^2 = 5 * (x + 4) ↔ (x = -4 ∨ x = 1) :=
by sorry

end solve_equation_l11_11343


namespace no_solution_for_x4_plus_y4_eq_z4_l11_11503

theorem no_solution_for_x4_plus_y4_eq_z4 :
  ∀ (x y z : ℤ), x ≠ 0 → y ≠ 0 → z ≠ 0 → gcd (gcd x y) z = 1 → x^4 + y^4 ≠ z^4 :=
sorry

end no_solution_for_x4_plus_y4_eq_z4_l11_11503


namespace olivia_paper_count_l11_11404

-- State the problem conditions and the final proof statement.
theorem olivia_paper_count :
  let math_initial := 220
  let science_initial := 150
  let math_used := 95
  let science_used := 68
  let math_received := 30
  let science_given := 15
  let math_remaining := math_initial - math_used + math_received
  let science_remaining := science_initial - science_used - science_given
  let total_pieces := math_remaining + science_remaining
  total_pieces = 222 :=
by
  -- Placeholder for the proof
  sorry

end olivia_paper_count_l11_11404


namespace five_g_speeds_l11_11384

theorem five_g_speeds (m : ℝ) :
  (1400 / 50) - (1400 / (50 * m)) = 24 → m = 7 :=
by
  sorry

end five_g_speeds_l11_11384


namespace min_value_frac_sum_l11_11828

theorem min_value_frac_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + 3 * b = 2) : 
  (∃ a b : ℝ, 0 < a ∧ 0 < b ∧ a + 3 * b = 2 ∧ (2 / a + 4 / b) = 14) :=
by
  sorry

end min_value_frac_sum_l11_11828


namespace arithmetic_sequence_sum_l11_11047

theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h_arith : ∀ n, a (n + 1) - a n = a 1 - a 0) 
  (h_sum : a 3 + a 9 + a 15 + a 21 = 8) :
  a 1 + a 23 = 4 :=
sorry

end arithmetic_sequence_sum_l11_11047


namespace distance_travelled_by_gavril_l11_11024

noncomputable def smartphoneFullyDischargesInVideoWatching : ℝ := 3
noncomputable def smartphoneFullyDischargesInPlayingTetris : ℝ := 5
noncomputable def speedForHalfDistanceFirst : ℝ := 80
noncomputable def speedForHalfDistanceSecond : ℝ := 60
noncomputable def averageSpeed (distance speed time : ℝ) :=
  distance / time = speed

theorem distance_travelled_by_gavril : 
  ∃ S : ℝ, 
    (∃ t : ℝ, 
      (t / 2 / smartphoneFullyDischargesInVideoWatching + t / 2 / smartphoneFullyDischargesInPlayingTetris = 1) ∧ 
      (S / 2 / t / 2 = speedForHalfDistanceFirst) ∧
      (S / 2 / t / 2 = speedForHalfDistanceSecond)) ∧
     S = 257 := 
sorry

end distance_travelled_by_gavril_l11_11024


namespace solve_for_x_l11_11739

theorem solve_for_x (x : ℝ) (h : (x + 4) / (x - 3) = (x - 2) / (x + 2)) : x = -(2 / 11) :=
by
  sorry

end solve_for_x_l11_11739


namespace no_integer_solutions_other_than_zero_l11_11975

theorem no_integer_solutions_other_than_zero (x y z : ℤ) :
  x^2 + y^2 + z^2 = x^2 * y^2 → x = 0 ∧ y = 0 ∧ z = 0 :=
by
  intro h
  sorry

end no_integer_solutions_other_than_zero_l11_11975


namespace arithmetic_sequence_general_term_find_n_given_sum_l11_11698

theorem arithmetic_sequence_general_term 
  (a : ℕ → ℕ)
  (S : ℕ → ℕ)
  (h1 : a 10 = 30)
  (h2 : a 15 = 40)
  : ∃ a1 d, (∀ n, a n = a1 + (n - 1) * d) ∧ a 10 = 30 ∧ a 15 = 40 :=
by {
  sorry
}

theorem find_n_given_sum
  (a : ℕ → ℕ)
  (S : ℕ → ℕ)
  (a1 d : ℕ)
  (h_gen : ∀ n, a n = a1 + (n - 1) * d)
  (h_sum : ∀ n, S n = n * a1 + (n * (n - 1) * d) / 2)
  (h_a1 : a1 = 12)
  (h_d : d = 2)
  (h_Sn : S 14 = 210)
  : ∃ n, S n = 210 ∧ n = 14 :=
by {
  sorry
}

end arithmetic_sequence_general_term_find_n_given_sum_l11_11698


namespace general_formula_sum_of_first_n_terms_max_sum_of_S_n_sum_of_absolute_values_l11_11297

section ArithmeticSequence

-- Given conditions
def a1 : Int := 13
def a4 : Int := 7
def d : Int := (a4 - a1) / 3

-- General formula for a_n
def a_n (n : Int) : Int := a1 + (n - 1) * d

-- Sum of the first n terms S_n
def S_n (n : Int) : Int := n * (a1 + a_n n) / 2

-- Maximum value of S_n and corresponding term
def S_max : Int := 49
def n_max_S : Int := 7

-- Sum of the absolute values of the first n terms T_n
def T_n (n : Int) : Int :=
  if n ≤ 7 then n^2 + 12 * n
  else 98 - 12 * n - n^2

-- Statements to prove
theorem general_formula (n : Int) : a_n n = 15 - 2 * n := sorry

theorem sum_of_first_n_terms (n : Int) : S_n n = 14 * n - n^2 := sorry

theorem max_sum_of_S_n : (S_n n_max_S = S_max) := sorry

theorem sum_of_absolute_values (n : Int) : T_n n = 
  if n ≤ 7 then n^2 + 12 * n else 98 - 12 * n - n^2 := sorry

end ArithmeticSequence

end general_formula_sum_of_first_n_terms_max_sum_of_S_n_sum_of_absolute_values_l11_11297


namespace number_of_arrangements_l11_11139

theorem number_of_arrangements (n : ℕ) (h_n : n = 6) : 
  ∃ total : ℕ, total = 90 := 
sorry

end number_of_arrangements_l11_11139


namespace pages_read_on_saturday_l11_11869

namespace BookReading

def total_pages : ℕ := 93
def pages_read_sunday : ℕ := 20
def pages_remaining : ℕ := 43

theorem pages_read_on_saturday :
  total_pages - (pages_read_sunday + pages_remaining) = 30 :=
by
  sorry

end BookReading

end pages_read_on_saturday_l11_11869


namespace car_speed_624km_in_2_2_5_hours_l11_11056

theorem car_speed_624km_in_2_2_5_hours : 
  ∀ (distance time_in_hours : ℝ), distance = 624 → time_in_hours = 2 + (2/5) → distance / time_in_hours = 260 :=
by
  intros distance time_in_hours h_dist h_time
  sorry

end car_speed_624km_in_2_2_5_hours_l11_11056


namespace lori_marbles_l11_11159

theorem lori_marbles (friends marbles_per_friend : ℕ) (h_friends : friends = 5) (h_marbles_per_friend : marbles_per_friend = 6) : friends * marbles_per_friend = 30 := sorry

end lori_marbles_l11_11159


namespace ducks_remaining_after_three_nights_l11_11471

def initial_ducks : ℕ := 320
def first_night_ducks (initial_ducks : ℕ) : ℕ := initial_ducks - (initial_ducks / 4)
def second_night_ducks (first_night_ducks : ℕ) : ℕ := first_night_ducks - (first_night_ducks / 6)
def third_night_ducks (second_night_ducks : ℕ) : ℕ := second_night_ducks - (second_night_ducks * 30 / 100)

theorem ducks_remaining_after_three_nights : 
  third_night_ducks (second_night_ducks (first_night_ducks initial_ducks)) = 140 :=
by
  -- Proof goes here
  sorry

end ducks_remaining_after_three_nights_l11_11471


namespace fraction_of_income_from_tips_l11_11280

variable (S T I : ℝ)

-- Conditions
def tips_as_fraction_of_salary : Prop := T = (3/4) * S
def total_income : Prop := I = S + T

-- Theorem stating the proof problem
theorem fraction_of_income_from_tips 
  (h1 : tips_as_fraction_of_salary S T)
  (h2 : total_income S T I) : (T / I) = 3 / 7 := by
  sorry

end fraction_of_income_from_tips_l11_11280


namespace general_admission_tickets_l11_11568

-- Define the number of student tickets and general admission tickets
variables {S G : ℕ}

-- Define the conditions
def tickets_sold (S G : ℕ) : Prop := S + G = 525
def amount_collected (S G : ℕ) : Prop := 4 * S + 6 * G = 2876

-- The theorem to prove that the number of general admission tickets is 388
theorem general_admission_tickets : 
  ∀ (S G : ℕ), tickets_sold S G → amount_collected S G → G = 388 :=
by
  sorry -- Proof to be provided

end general_admission_tickets_l11_11568


namespace spherical_coords_standard_form_l11_11982

theorem spherical_coords_standard_form :
  ∀ (ρ θ φ : ℝ), ρ > 0 → 0 ≤ θ ∧ θ < 2 * Real.pi → 0 ≤ φ ∧ φ ≤ Real.pi →
  (5, (5 * Real.pi) / 7, (11 * Real.pi) / 6) = (ρ, θ, φ) →
  (ρ, (12 * Real.pi) / 7, Real.pi / 6) = (ρ, θ, φ) :=
by 
  intros ρ θ φ hρ hθ hφ h_eq
  sorry

end spherical_coords_standard_form_l11_11982


namespace solve_equation_l11_11131

theorem solve_equation (x : ℝ) (h : (2 / (x - 3) = 3 / (x - 6))) : x = -3 :=
sorry

end solve_equation_l11_11131


namespace max_value_of_a_l11_11679
noncomputable def f (a x : ℝ) : ℝ :=
  if x < a then -a * x + 1 else (x - 2)^2

theorem max_value_of_a (a : ℝ) :
  (∃ (x : ℝ), ∀ (y : ℝ), f a x ≤ f a y) → a ≤ 1 := 
sorry

end max_value_of_a_l11_11679


namespace Balint_claim_impossible_l11_11069

-- Declare the lengths of the ladders and the vertical projection distance
def AC : ℝ := 3
def BD : ℝ := 2
def E_proj : ℝ := 1

-- State the problem conditions and what we need to prove
theorem Balint_claim_impossible (h1 : AC = 3) (h2 : BD = 2) (h3 : E_proj = 1) :
  False :=
  sorry

end Balint_claim_impossible_l11_11069


namespace compute_expression_l11_11592

theorem compute_expression : 6^2 + 2 * 5 - 4^2 = 30 :=
by sorry

end compute_expression_l11_11592


namespace b_share_l11_11088

theorem b_share (a b c : ℕ) (h1 : a + b + c = 120) (h2 : a = b + 20) (h3 : a = c - 20) : b = 20 :=
by
  sorry

end b_share_l11_11088


namespace commute_times_abs_difference_l11_11093

theorem commute_times_abs_difference (x y : ℝ)
  (h_avg : (x + y + 10 + 11 + 9) / 5 = 10)
  (h_var : ((x - 10)^2 + (y - 10)^2 + 0^2 + 1^2 + (-1)^2) / 5 = 2) 
  : |x - y| = 4 :=
sorry

end commute_times_abs_difference_l11_11093


namespace min_value_fraction_l11_11544

theorem min_value_fraction (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + 2 * b = 1) : 
  ∃ (m : ℝ), m = 3 + 2 * Real.sqrt 2 ∧ (∀ (x : ℝ) (hx : x = 1 / a + 1 / b), x ≥ m) := 
by
  sorry

end min_value_fraction_l11_11544


namespace find_q_l11_11848

noncomputable def common_ratio_of_geometric_sequence
  (a : ℕ → ℝ) (q : ℝ) : Prop :=
  a 4 = 27 ∧ a 7 = -729 ∧ ∀ n m, a n = a m * q ^ (n - m)

theorem find_q {a : ℕ → ℝ} {q : ℝ} (h : common_ratio_of_geometric_sequence a q) :
  q = -3 :=
by {
  sorry
}

end find_q_l11_11848


namespace problem_statement_l11_11537

theorem problem_statement (m n : ℕ) (h1 : 0 < m) (h2 : 0 < n) 
    (h3 : m + 5 < n) 
    (h4 : (m + 3 + m + 7 + m + 13 + n + 4 + n + 5 + 2 * n + 3) / 6 = n + 3)
    (h5 : (↑((m + 13) + (n + 4)) / 2 : ℤ) = n + 3) : 
  m + n = 37 :=
by
  sorry

end problem_statement_l11_11537


namespace visitors_answered_questionnaire_l11_11714

theorem visitors_answered_questionnaire (V : ℕ) (h : (3 / 4 : ℝ) * V = (V : ℝ) - 110) : V = 440 :=
sorry

end visitors_answered_questionnaire_l11_11714


namespace table_price_l11_11428

theorem table_price
  (C T : ℝ)
  (h1 : 2 * C + T = 0.6 * (C + 2 * T))
  (h2 : C + T = 60) :
  T = 52.5 :=
by
  sorry

end table_price_l11_11428


namespace tolu_pencils_l11_11791

theorem tolu_pencils (price_per_pencil : ℝ) (robert_pencils : ℕ) (melissa_pencils : ℕ) (total_money_spent : ℝ) (tolu_pencils : ℕ) :
  price_per_pencil = 0.20 →
  robert_pencils = 5 →
  melissa_pencils = 2 →
  total_money_spent = 2.00 →
  tolu_pencils * price_per_pencil = 2.00 - (5 * 0.20 + 2 * 0.20) →
  tolu_pencils = 3 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end tolu_pencils_l11_11791


namespace right_triangle_hypotenuse_l11_11619

theorem right_triangle_hypotenuse 
  (shorter_leg longer_leg hypotenuse : ℝ)
  (h1 : longer_leg = 2 * shorter_leg - 1)
  (h2 : 1 / 2 * shorter_leg * longer_leg = 60) :
  hypotenuse = 17 :=
by
  sorry

end right_triangle_hypotenuse_l11_11619


namespace sequence_ineq_l11_11147

theorem sequence_ineq (a : ℕ → ℝ) (h1 : a 1 = 15) 
  (h2 : ∀ n, a (n + 1) = a n - 2 / 3) 
  (hk : a k * a (k + 1) < 0) : k = 23 :=
sorry

end sequence_ineq_l11_11147


namespace employee_payment_proof_l11_11611

-- Define the wholesale cost
def wholesale_cost : ℝ := 200

-- Define the retail price as 20 percent above the wholesale cost
def retail_price (C_w : ℝ) : ℝ := C_w + 0.2 * C_w

-- Define the employee discount on the retail price
def employee_discount (C_r : ℝ) : ℝ := 0.15 * C_r

-- Define the amount paid by the employee
def amount_paid_by_employee (C_w : ℝ) : ℝ :=
  let C_r := retail_price C_w
  let D_e := employee_discount C_r
  C_r - D_e

-- Main theorem to prove the employee paid $204
theorem employee_payment_proof : amount_paid_by_employee wholesale_cost = 204 :=
by
  sorry

end employee_payment_proof_l11_11611


namespace intersection_of_A_and_B_solve_inequality_l11_11864

-- Definitions based on conditions
def U : Set ℝ := Set.univ
def A : Set ℝ := {x : ℝ | x^2 - 16 < 0}
def B : Set ℝ := {x : ℝ | x^2 - 4 * x + 3 ≥ 0}

-- Proof problem 1: Find A ∩ B
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | (-4 < x ∧ x < 1) ∨ (3 < x ∧ x < 4)} :=
sorry

-- Proof problem 2: Solve the inequality with respect to x
theorem solve_inequality (a : ℝ) :
  if a = 1 then
    {x : ℝ | x^2 - (a+1) * x + a < 0} = ∅
  else if a > 1 then
    {x : ℝ | x^2 - (a+1) * x + a < 0} = {x : ℝ | 1 < x ∧ x < a}
  else
    {x : ℝ | x^2 - (a+1) * x + a < 0} = {x : ℝ | a < x ∧ x < 1} :=
sorry

end intersection_of_A_and_B_solve_inequality_l11_11864


namespace problem_statement_l11_11676

theorem problem_statement (a b : ℕ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : ∀ n : ℕ, n ≥ 1 → 2^n * b + 1 ∣ a^(2^n) - 1) : a = 1 := by
  sorry

end problem_statement_l11_11676


namespace determine_p_l11_11268

variable (x y z p : ℝ)

theorem determine_p (h1 : 8 / (x + y) = p / (x + z)) (h2 : p / (x + z) = 12 / (z - y)) : p = 20 :=
sorry

end determine_p_l11_11268


namespace sqrt_rational_rational_l11_11896

theorem sqrt_rational_rational 
  (a b : ℚ) 
  (h : ∃ r : ℚ, r = (a : ℝ).sqrt + (b : ℝ).sqrt) : 
  (∃ p : ℚ, p = (a : ℝ).sqrt) ∧ (∃ q : ℚ, q = (b : ℝ).sqrt) := 
sorry

end sqrt_rational_rational_l11_11896


namespace necessary_and_sufficient_condition_l11_11216

variable (m n : ℕ)
def positive_integers (m n : ℕ) := m > 0 ∧ n > 0
def at_least_one_is_1 (m n : ℕ) : Prop := m = 1 ∨ n = 1
def sum_gt_product (m n : ℕ) : Prop := m + n > m * n

theorem necessary_and_sufficient_condition (h : positive_integers m n) : 
  sum_gt_product m n ↔ at_least_one_is_1 m n :=
by sorry

end necessary_and_sufficient_condition_l11_11216


namespace points_per_vegetable_correct_l11_11596

-- Given conditions
def total_points_needed : ℕ := 200
def number_of_students : ℕ := 25
def number_of_weeks : ℕ := 2
def veggies_per_student_per_week : ℕ := 2

-- Derived values
def total_veggies_eaten_by_class : ℕ :=
  number_of_students * number_of_weeks * veggies_per_student_per_week

def points_per_vegetable : ℕ :=
  total_points_needed / total_veggies_eaten_by_class

-- Theorem to be proven
theorem points_per_vegetable_correct :
  points_per_vegetable = 2 := by
sorry

end points_per_vegetable_correct_l11_11596


namespace quadratic_two_distinct_real_roots_l11_11510

theorem quadratic_two_distinct_real_roots
  (a1 a2 a3 a4 : ℝ)
  (h : a1 > a2 ∧ a2 > a3 ∧ a3 > a4) :
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 - (a1 + a2 + a3 + a4) * x1 + (a1 * a3 + a2 * a4) = 0)
  ∧ (x2^2 - (a1 + a2 + a3 + a4) * x2 + (a1 * a3 + a2 * a4) = 0) :=
by 
  sorry

end quadratic_two_distinct_real_roots_l11_11510


namespace taller_building_height_l11_11949

theorem taller_building_height
  (H : ℕ) -- H is the height of the taller building
  (h_ratio : (H - 36) / H = 5 / 7) -- heights ratio condition
  (h_diff : H > 36) -- height difference must respect physics
  : H = 126 := sorry

end taller_building_height_l11_11949


namespace possible_values_of_sum_of_reciprocals_l11_11325

theorem possible_values_of_sum_of_reciprocals {a b : ℝ} (h1 : a * b > 0) (h2 : a + b = 1) : 
  1 / a + 1 / b = 4 := 
by 
  sorry

end possible_values_of_sum_of_reciprocals_l11_11325


namespace intersection_eq_set_l11_11322

def M : Set ℤ := { x | -4 < (x : Int) ∧ x < 2 }
def N : Set Int := { x | (x : ℝ) ^ 2 < 4 }
def intersection := M ∩ N

theorem intersection_eq_set : intersection = {-1, 0, 1} := 
sorry

end intersection_eq_set_l11_11322


namespace average_expression_l11_11790

-- Define a theorem to verify the given problem
theorem average_expression (E a : ℤ) (h1 : a = 34) (h2 : (E + (3 * a - 8)) / 2 = 89) : E = 84 :=
by
  -- Proof goes here
  sorry

end average_expression_l11_11790


namespace parabola_vertex_y_l11_11665

theorem parabola_vertex_y (x : ℝ) : (∃ (h k : ℝ), (4 * (x - h)^2 + k = 4 * x^2 + 16 * x + 11) ∧ k = -5) := 
  sorry

end parabola_vertex_y_l11_11665


namespace consignment_shop_total_items_l11_11052

variable (x y z t n : ℕ)

noncomputable def totalItems (n : ℕ) := n + n + n + 3 * n

theorem consignment_shop_total_items :
  ∃ (x y z t n : ℕ), 
    3 * n * y + n * x + n * z + n * t = 240 ∧
    t = 10 * n ∧
    z + x = y + t + 4 ∧
    x + y + 24 = t + z ∧
    y ≤ 6 ∧
    totalItems n = 18 :=
by
  sorry

end consignment_shop_total_items_l11_11052


namespace number_of_children_l11_11479

-- Definitions of given conditions
def total_passengers := 170
def men := 90
def women := men / 2
def adults := men + women
def children := total_passengers - adults

-- Theorem statement
theorem number_of_children : children = 35 :=
by
  sorry

end number_of_children_l11_11479


namespace polynomial_divisibility_l11_11289

theorem polynomial_divisibility (n : ℕ) : 120 ∣ (n^5 - 5*n^3 + 4*n) :=
sorry

end polynomial_divisibility_l11_11289


namespace quadratic_real_roots_range_l11_11182

theorem quadratic_real_roots_range (k : ℝ) (h : ∀ x : ℝ, (k - 1) * x^2 - 2 * x + 1 = 0) : k ≤ 2 ∧ k ≠ 1 :=
by
  sorry

end quadratic_real_roots_range_l11_11182


namespace maximize_profit_l11_11879

def cups_sold (p : ℝ) : ℝ :=
  150 - 4 * p

def revenue (p : ℝ) : ℝ :=
  p * cups_sold p

def cost : ℝ :=
  200

def profit (p : ℝ) : ℝ :=
  revenue p - cost

theorem maximize_profit (p : ℝ) (h : p ≤ 30) : p = 19 → profit p = 1206.25 :=
by
  sorry

end maximize_profit_l11_11879


namespace intersection_P_Q_l11_11521

-- Define set P
def P : Set ℕ := {x | 1 ≤ x ∧ x ≤ 10}

-- Define set Q (using real numbers, but we will be interested in natural number intersections)
def Q : Set ℝ := {x | x^2 + x - 6 ≤ 0}

-- The intersection of P with Q in the natural numbers should be {1, 2}
theorem intersection_P_Q :
  {x : ℕ | x ∈ P ∧ (x : ℝ) ∈ Q} = {1, 2} :=
by
  sorry

end intersection_P_Q_l11_11521


namespace sum_is_correct_l11_11820

-- Define the five prime numbers with units digit 3
def prime1 := 3
def prime2 := 13
def prime3 := 23
def prime4 := 43
def prime5 := 53

-- Define the sum of these five primes
def sum_of_five_primes : Nat :=
  prime1 + prime2 + prime3 + prime4 + prime5

-- Theorem statement
theorem sum_is_correct : sum_of_five_primes = 123 :=
  by
    -- Proof placeholder
    sorry

end sum_is_correct_l11_11820


namespace keiko_walking_speed_l11_11412

theorem keiko_walking_speed (r : ℝ) (t : ℝ) (width : ℝ) 
   (time_diff : ℝ) (h0 : width = 8) (h1 : time_diff = 48) 
   (h2 : t = (2 * (2 * (r + 8) * Real.pi) / (r + 8) + 2 * (0 * Real.pi))) 
   (h3 : 2 * (2 * r * Real.pi) / r + 2 * (0 * Real.pi) = t - time_diff) :
   t = 48 -> 
   (v : ℝ) →
   v = (16 * Real.pi) / time_diff →
   v = Real.pi / 3 :=
by
  sorry

end keiko_walking_speed_l11_11412


namespace student_A_more_stable_l11_11065

-- Given conditions
def average_score (n : ℕ) (score : ℕ) := score = 110
def variance_A := 3.6
def variance_B := 4.4

-- Prove that student A has more stable scores than student B
theorem student_A_more_stable : variance_A < variance_B :=
by
  -- Skipping the actual proof
  sorry

end student_A_more_stable_l11_11065


namespace ratio_of_increase_to_current_l11_11039

-- Define the constants for the problem
def current_deductible : ℝ := 3000
def increase_deductible : ℝ := 2000

-- State the theorem that needs to be proven
theorem ratio_of_increase_to_current : 
  (increase_deductible / current_deductible) = (2 / 3) :=
by sorry

end ratio_of_increase_to_current_l11_11039


namespace length_of_greater_segment_l11_11966

theorem length_of_greater_segment (x : ℤ) (h1 : (x + 2)^2 - x^2 = 32) : x + 2 = 9 := by
  sorry

end length_of_greater_segment_l11_11966


namespace graphene_scientific_notation_l11_11317

def scientific_notation (n : ℝ) (a : ℝ) (exp : ℤ) : Prop :=
  n = a * 10 ^ exp ∧ 1 ≤ abs a ∧ abs a < 10

theorem graphene_scientific_notation :
  scientific_notation 0.00000000034 3.4 (-10) :=
by {
  sorry
}

end graphene_scientific_notation_l11_11317


namespace dino_remaining_money_l11_11338

-- Definitions of the conditions
def hours_gig_1 : ℕ := 20
def hourly_rate_gig_1 : ℕ := 10

def hours_gig_2 : ℕ := 30
def hourly_rate_gig_2 : ℕ := 20

def hours_gig_3 : ℕ := 5
def hourly_rate_gig_3 : ℕ := 40

def expenses : ℕ := 500

-- The theorem to be proved: Dino's remaining money at the end of the month
theorem dino_remaining_money : 
  (hours_gig_1 * hourly_rate_gig_1 + hours_gig_2 * hourly_rate_gig_2 + hours_gig_3 * hourly_rate_gig_3) - expenses = 500 := by
  sorry

end dino_remaining_money_l11_11338


namespace planted_fraction_correct_l11_11110

-- Define the vertices of the triangle
def A : (ℝ × ℝ) := (0, 0)
def B : (ℝ × ℝ) := (5, 0)
def C : (ℝ × ℝ) := (0, 12)

-- Define the length of the legs
def leg1 := 5
def leg2 := 12

-- Define the shortest distance from the square to the hypotenuse
def distance_to_hypotenuse := 3

-- Define the area of the triangle
def triangle_area := (1 / 2) * (leg1 * leg2)

-- Assume the side length of the square
def s := 6 / 13

-- Define the area of the square
def square_area := s^2

-- Define the fraction of the field that is unplanted
def unplanted_fraction := square_area / triangle_area

-- Define the fraction of the field that is planted
def planted_fraction := 1 - unplanted_fraction

theorem planted_fraction_correct :
  planted_fraction = 5034 / 5070 :=
sorry

end planted_fraction_correct_l11_11110


namespace S15_eq_l11_11571

-- Definitions in terms of the geometric sequence and given conditions
variable (a : ℕ → ℝ) (S : ℕ → ℝ)

-- Conditions given in the problem
axiom geom_seq (n : ℕ) : S n = (a 0) * (1 - (a 1) ^ n) / (1 - (a 1))
axiom S5_eq : S 5 = 10
axiom S10_eq : S 10 = 50

-- The problem statement to prove
theorem S15_eq : S 15 = 210 :=
by sorry

end S15_eq_l11_11571


namespace trajectory_of_M_ellipse_trajectory_l11_11414

variable {x y : ℝ}

theorem trajectory_of_M (hx : x ≠ 5) (hnx : x ≠ -5)
  (h : (y / (x + 5)) * (y / (x - 5)) = -2) : 
  (2 * x^2 + y^2 = 50) :=
by
  -- Proof is omitted.
  sorry

theorem ellipse_trajectory (hx : x ≠ 5) (hnx : x ≠ -5) 
  (h : (y / (x + 5)) * (y / (x - 5)) = -2) : 
  (x^2 / 25 + y^2 / 50 = 1) :=
by
  -- Using the previous theorem to derive.
  have h1 : (2 * x^2 + y^2 = 50) := trajectory_of_M hx hnx h
  -- Proof of transformation is omitted.
  sorry

end trajectory_of_M_ellipse_trajectory_l11_11414


namespace two_digit_factors_count_l11_11884

-- Definition of the expression 10^8 - 1
def expr : ℕ := 10^8 - 1

-- Factorization of 10^8 - 1
def factored_expr : List ℕ := [73, 137, 101, 11, 3^2]

-- Define the condition for being a two-digit factor
def is_two_digit (n : ℕ) : Bool := n > 9 ∧ n < 100

-- Count the number of positive two-digit factors in the factorization of 10^8 - 1
def num_two_digit_factors : ℕ := List.length (factored_expr.filter is_two_digit)

-- The theorem stating our proof problem
theorem two_digit_factors_count : num_two_digit_factors = 2 := by
  sorry

end two_digit_factors_count_l11_11884


namespace find_x_l11_11965

theorem find_x (x : ℝ) (h : 121 * x^4 = 75625) : x = 5 :=
sorry

end find_x_l11_11965


namespace falcon_speed_correct_l11_11103

-- Definitions based on conditions
def eagle_speed : ℕ := 15
def pelican_speed : ℕ := 33
def hummingbird_speed : ℕ := 30
def total_distance : ℕ := 248
def time_hours : ℕ := 2

-- Variables representing the unknown falcon speed
variable {falcon_speed : ℕ}

-- The Lean statement to prove
theorem falcon_speed_correct 
  (h : 2 * falcon_speed + (eagle_speed * time_hours) + (pelican_speed * time_hours) + (hummingbird_speed * time_hours) = total_distance) :
  falcon_speed = 46 :=
sorry

end falcon_speed_correct_l11_11103


namespace scientific_notation_conversion_l11_11150

theorem scientific_notation_conversion :
  0.000037 = 3.7 * 10^(-5) :=
by
  sorry

end scientific_notation_conversion_l11_11150


namespace no_even_threes_in_circle_l11_11291

theorem no_even_threes_in_circle (arr : ℕ → ℕ) (h1 : ∀ i, 1 ≤ arr i ∧ arr i ≤ 2017)
  (h2 : ∀ i, (arr i + arr ((i + 1) % 2017) + arr ((i + 2) % 2017)) % 2 = 0) : false :=
sorry

end no_even_threes_in_circle_l11_11291


namespace condition_inequality_l11_11837

theorem condition_inequality (x y : ℝ) :
  (¬ (x ≤ y → |x| ≤ |y|)) ∧ (¬ (|x| ≤ |y| → x ≤ y)) :=
by
  sorry

end condition_inequality_l11_11837


namespace sum_9_to_12_l11_11909

variable {a : ℕ → ℝ} -- Define the arithmetic sequence
variables {S : ℕ → ℝ} -- Define the sum function of the sequence

-- Define the conditions given in the problem
def S_4 : ℝ := 8
def S_8 : ℝ := 20

-- The goal is to show that the sum of the 9th to 12th terms is 16
theorem sum_9_to_12 : (a 9) + (a 10) + (a 11) + (a 12) = 16 :=
by
  sorry

end sum_9_to_12_l11_11909


namespace smallest_A_l11_11006

theorem smallest_A (A B C D E : ℕ) 
  (hA_even : A % 2 = 0)
  (hB_even : B % 2 = 0)
  (hC_even : C % 2 = 0)
  (hD_even : D % 2 = 0)
  (hE_even : E % 2 = 0)
  (hA_three_digit : 100 ≤ A ∧ A < 1000)
  (hB_three_digit : 100 ≤ B ∧ B < 1000)
  (hC_three_digit : 100 ≤ C ∧ C < 1000)
  (hD_three_digit : 100 ≤ D ∧ D < 1000)
  (hE_three_digit : 100 ≤ E ∧ E < 1000)
  (h_sorted : A < B ∧ B < C ∧ C < D ∧ D < E)
  (h_sum : A + B + C + D + E = 4306) :
  A = 326 :=
sorry

end smallest_A_l11_11006


namespace least_area_of_square_l11_11478

theorem least_area_of_square :
  ∀ (s : ℝ), (3.5 ≤ s ∧ s < 4.5) → (s * s ≥ 12.25) :=
by
  intro s
  intro hs
  sorry

end least_area_of_square_l11_11478


namespace bill_can_buy_donuts_in_35_ways_l11_11457

def different_ways_to_buy_donuts : ℕ :=
  5 + 20 + 10  -- Number of ways to satisfy the conditions

theorem bill_can_buy_donuts_in_35_ways :
  different_ways_to_buy_donuts = 35 :=
by
  -- Proof steps
  -- The problem statement and the solution show the calculation to be correct.
  sorry

end bill_can_buy_donuts_in_35_ways_l11_11457


namespace production_value_decreased_by_10_percent_l11_11304

variable (a : ℝ)

def production_value_in_January : ℝ := a

def production_value_in_February (a : ℝ) : ℝ := 0.9 * a

theorem production_value_decreased_by_10_percent (a : ℝ) :
  production_value_in_February a = 0.9 * production_value_in_January a := 
by
  sorry

end production_value_decreased_by_10_percent_l11_11304


namespace perfect_square_trinomial_m_l11_11267

theorem perfect_square_trinomial_m (m : ℝ) :
  (∃ (a b : ℝ), ∀ x : ℝ, x^2 + (m-1)*x + 9 = (a*x + b)^2) ↔ (m = 7 ∨ m = -5) :=
sorry

end perfect_square_trinomial_m_l11_11267


namespace geometric_sequence_sum_l11_11986

-- Defining the geometric sequence related properties and conditions
theorem geometric_sequence_sum (a : ℕ → ℝ) (r : ℝ) (S : ℕ → ℝ) :
  (∀ n, a (n + 1) = a n * r) → 
  S 3 = a 0 + a 1 + a 2 →
  S 6 = a 3 + a 4 + a 5 →
  S 12 = a 0 + a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 + a 11 →
  S 3 = 3 →
  S 6 = 6 →
  S 12 = 45 :=
by
  sorry

end geometric_sequence_sum_l11_11986


namespace necessary_but_not_sufficient_condition_l11_11091

theorem necessary_but_not_sufficient_condition (x : ℝ) :
  ((0 < x ∧ x < 5) → (|x - 2| < 3)) ∧ ¬ ((|x - 2| < 3) → (0 < x ∧ x < 5)) :=
by
  sorry

end necessary_but_not_sufficient_condition_l11_11091


namespace max_b_value_l11_11627

theorem max_b_value (a b c : ℕ) (h1 : 1 < c) (h2 : c < b) (h3 : b < a) (h4 : a * b * c = 360) : b ≤ 10 :=
sorry

end max_b_value_l11_11627


namespace total_unique_items_l11_11905

-- Define the conditions
def shared_albums : ℕ := 12
def total_andrew_albums : ℕ := 23
def exclusive_andrew_memorabilia : ℕ := 5
def exclusive_john_albums : ℕ := 8

-- Define the number of unique items in Andrew's and John's collection 
def unique_andrew_albums : ℕ := total_andrew_albums - shared_albums
def unique_total_items : ℕ := unique_andrew_albums + exclusive_john_albums + exclusive_andrew_memorabilia

-- The proof goal
theorem total_unique_items : unique_total_items = 24 := by
  -- Proof steps would go here
  sorry

end total_unique_items_l11_11905


namespace number_of_solutions_5x_plus_10y_eq_50_l11_11149

theorem number_of_solutions_5x_plus_10y_eq_50 : 
  (∃! (n : ℕ), ∃ (xy : ℕ × ℕ), xy.1 + 2 * xy.2 = 10 ∧ n = 6) :=
by
  sorry

end number_of_solutions_5x_plus_10y_eq_50_l11_11149


namespace find_y_l11_11442

-- Suppose C > A > B > 0
-- and A is y% smaller than C.
-- Also, C = 2B.
-- We need to show that y = 100 - 50 * (A / B).

variable (A B C : ℝ)
variable (y : ℝ)

-- Conditions
axiom h1 : C > A
axiom h2 : A > B
axiom h3 : B > 0
axiom h4 : C = 2 * B
axiom h5 : A = (1 - y / 100) * C

-- Goal
theorem find_y : y = 100 - 50 * (A / B) :=
by
  sorry

end find_y_l11_11442


namespace sheila_weekly_earnings_l11_11226

-- Definitions for conditions
def hours_per_day_on_MWF : ℕ := 8
def days_worked_on_MWF : ℕ := 3
def hours_per_day_on_TT : ℕ := 6
def days_worked_on_TT : ℕ := 2
def hourly_rate : ℕ := 10

-- Total weekly hours worked
def total_weekly_hours : ℕ :=
  (hours_per_day_on_MWF * days_worked_on_MWF) + (hours_per_day_on_TT * days_worked_on_TT)

-- Total weekly earnings
def weekly_earnings : ℕ :=
  total_weekly_hours * hourly_rate

-- Lean statement for the proof
theorem sheila_weekly_earnings : weekly_earnings = 360 :=
  sorry

end sheila_weekly_earnings_l11_11226


namespace simplify_expression_l11_11274

theorem simplify_expression (x : ℝ) : (2 * x)^5 + (3 * x) * x^4 + 2 * x^3 = 35 * x^5 + 2 * x^3 :=
by
  sorry

end simplify_expression_l11_11274


namespace find_ab_l11_11901

theorem find_ab (a b : ℝ) : 
  (∀ x : ℝ, -1 < x ∧ x < 2 →
  (3 * x - 2 < a + 1 ∧ 6 - 2 * x < b + 2)) →
  a = 3 ∧ b = 6 :=
by
  sorry

end find_ab_l11_11901


namespace difference_of_squares_multiple_of_20_l11_11691

theorem difference_of_squares_multiple_of_20 (a b : ℕ) (h1 : a > b) (h2 : a + b = 10) (hb : b = 10 - a) : 
  ∃ k : ℕ, (9 * a + 10)^2 - (100 - 9 * a)^2 = 20 * k :=
by
  sorry

end difference_of_squares_multiple_of_20_l11_11691


namespace solve_y_l11_11800

theorem solve_y (y : ℝ) : (12 - y)^2 = 4 * y^2 ↔ y = 4 ∨ y = -12 := by
  sorry

end solve_y_l11_11800


namespace same_color_points_distance_2004_l11_11540

noncomputable def exists_same_color_points_at_distance_2004 (color : ℝ × ℝ → ℕ) : Prop :=
  ∃ (p q : ℝ × ℝ), (p ≠ q) ∧ (color p = color q) ∧ (dist p q = 2004)

/-- The plane is colored in two colors. Prove that there exist two points of the same color at a distance of 2004 meters. -/
theorem same_color_points_distance_2004 {color : ℝ × ℝ → ℕ}
  (hcolor : ∀ p, color p = 1 ∨ color p = 2) :
  exists_same_color_points_at_distance_2004 color :=
sorry

end same_color_points_distance_2004_l11_11540


namespace volume_ratio_l11_11880

def volume_of_cube (side_length : ℕ) : ℕ :=
  side_length * side_length * side_length

theorem volume_ratio 
  (hyungjin_side_length_cm : ℕ)
  (kyujun_side_length_m : ℕ)
  (h1 : hyungjin_side_length_cm = 100)
  (h2 : kyujun_side_length_m = 2) :
  volume_of_cube (kyujun_side_length_m * 100) = 8 * volume_of_cube hyungjin_side_length_cm :=
by
  sorry

end volume_ratio_l11_11880


namespace find_g_l11_11140

def nabla (g h : ℤ) : ℤ := g ^ 2 - h ^ 2

theorem find_g (g : ℤ) (h : ℤ)
  (H1 : 0 < g)
  (H2 : nabla g 6 = 45) :
  g = 9 :=
by
  sorry

end find_g_l11_11140


namespace combination_8_choose_2_l11_11852

theorem combination_8_choose_2 : Nat.choose 8 2 = 28 := sorry

end combination_8_choose_2_l11_11852


namespace base7_divisible_by_5_l11_11087

theorem base7_divisible_by_5 :
  ∃ (d : ℕ), (0 ≤ d ∧ d < 7) ∧ (344 * d + 56) % 5 = 0 ↔ d = 1 :=
by
  sorry

end base7_divisible_by_5_l11_11087


namespace sqrt_14_range_l11_11593

theorem sqrt_14_range : 3 < Real.sqrt 14 ∧ Real.sqrt 14 < 4 :=
by
  -- We know that 9 < 14 < 16, so we can take the square root of all parts to get 3 < sqrt(14) < 4.
  sorry

end sqrt_14_range_l11_11593


namespace sequence_term_a1000_l11_11290

theorem sequence_term_a1000 :
  ∃ (a : ℕ → ℕ), a 1 = 1007 ∧ a 2 = 1008 ∧
  (∀ n, n ≥ 1 → a n + a (n + 1) + a (n + 2) = 2 * n) ∧
  a 1000 = 1673 :=
by
  sorry

end sequence_term_a1000_l11_11290


namespace kaleb_cherries_left_l11_11420

theorem kaleb_cherries_left (initial_cherries eaten_cherries remaining_cherries : ℕ) (h1 : initial_cherries = 67) (h2 : eaten_cherries = 25) : remaining_cherries = initial_cherries - eaten_cherries → remaining_cherries = 42 :=
by
  intros h3
  rw [h1, h2] at h3
  exact h3

end kaleb_cherries_left_l11_11420


namespace bridget_initial_skittles_l11_11389

theorem bridget_initial_skittles (b : ℕ) (h : b + 4 = 8) : b = 4 :=
by {
  sorry
}

end bridget_initial_skittles_l11_11389


namespace purple_chip_count_l11_11283

theorem purple_chip_count :
  ∃ (x : ℕ), (x > 5) ∧ (x < 11) ∧
  (∃ (blue green purple red : ℕ),
    (2^6) * (5^2) * 11 * 7 = (blue * 1) * (green * 5) * (purple * x) * (red * 11) ∧ purple = 1) :=
sorry

end purple_chip_count_l11_11283


namespace remainder_2048_mod_13_l11_11136

theorem remainder_2048_mod_13 : 2048 % 13 = 7 := by
  sorry

end remainder_2048_mod_13_l11_11136


namespace find_k_l11_11841

theorem find_k (k : ℤ) (h1 : |k| = 1) (h2 : k - 1 ≠ 0) : k = -1 :=
by
  sorry

end find_k_l11_11841


namespace circle_C_equation_l11_11175

/-- Definitions of circles C1 and C2 -/
def circle_C1 := ∀ (x y : ℝ), (x - 4) ^ 2 + (y - 8) ^ 2 = 1
def circle_C2 := ∀ (x y : ℝ), (x - 6) ^ 2 + (y + 6) ^ 2 = 9

/-- Condition that the center of circle C is on the x-axis -/
def center_on_x_axis (x : ℝ) : Prop := ∃ y : ℝ, y = 0

/-- Bisection condition circle C bisects circumferences of circles C1 and C2 -/
def bisects_circumferences (x : ℝ) : Prop := 
  (∀ (y1 y2 : ℝ), ((x - 4) ^ 2 + (y1 - 8) ^ 2 + 1 = (x - 6) ^ 2 + (y2 + 6) ^ 2 + 9)) ∧ 
  center_on_x_axis x

/-- Statement to prove -/
theorem circle_C_equation : ∃ x y : ℝ, bisects_circumferences x ∧ (x^2 + y^2 = 81) := 
sorry

end circle_C_equation_l11_11175


namespace total_players_l11_11337

-- Definitions based on problem conditions.
def players_kabadi : Nat := 10
def players_kho_kho_only : Nat := 20
def players_both_games : Nat := 5

-- Proof statement for the total number of players.
theorem total_players : (players_kabadi + players_kho_kho_only - players_both_games) = 25 := by
  sorry

end total_players_l11_11337


namespace Coe_speed_theorem_l11_11798

-- Define the conditions
def Teena_speed : ℝ := 55
def initial_distance_behind : ℝ := 7.5
def time_hours : ℝ := 1.5
def distance_ahead : ℝ := 15

-- Define Coe's speed
def Coe_speed := 50

-- State the theorem
theorem Coe_speed_theorem : 
  let distance_Teena_covers := Teena_speed * time_hours
  let total_relative_distance := distance_Teena_covers + initial_distance_behind
  let distance_Coe_covers := total_relative_distance - distance_ahead
  let computed_Coe_speed := distance_Coe_covers / time_hours
  computed_Coe_speed = Coe_speed :=
by sorry

end Coe_speed_theorem_l11_11798


namespace same_last_k_digits_pow_l11_11486

theorem same_last_k_digits_pow (A B : ℤ) (k n : ℕ) 
  (h : A % 10^k = B % 10^k) : 
  (A^n % 10^k = B^n % 10^k) := 
by
  sorry

end same_last_k_digits_pow_l11_11486


namespace find_pairs_l11_11206

theorem find_pairs (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a^2 / b + b^2 / a = (a + b)^2 / (a + b)) ↔ (a = b) := by
  sorry

end find_pairs_l11_11206


namespace dragon_jewels_end_l11_11555

-- Given conditions
variables (D : ℕ) (jewels_taken_by_king jewels_taken_from_king new_jewels final_jewels : ℕ)

-- Conditions corresponding to the problem
axiom h1 : jewels_taken_by_king = 3
axiom h2 : jewels_taken_from_king = 2 * jewels_taken_by_king
axiom h3 : new_jewels = jewels_taken_from_king
axiom h4 : new_jewels = D / 3

-- Equation derived from the problem setting
def number_of_jewels_initial := D
def number_of_jewels_after_king_stole := number_of_jewels_initial - jewels_taken_by_king
def number_of_jewels_final := number_of_jewels_after_king_stole + jewels_taken_from_king

-- Final proof obligation
theorem dragon_jewels_end : ∃ (D : ℕ), number_of_jewels_final D 3 6 = 21 :=
by
  sorry

end dragon_jewels_end_l11_11555


namespace find_c_l11_11119

open Real

theorem find_c (c : ℝ) (h : ∀ x, (x ∈ Set.Iio 2 ∨ x ∈ Set.Ioi 7) → -x^2 + c * x - 9 < -4) : 
  c = 9 :=
sorry

end find_c_l11_11119


namespace series_converges_l11_11309

theorem series_converges (u : ℕ → ℝ) (h : ∀ n, u n = n / (3 : ℝ)^n) :
  ∃ l, 0 ≤ l ∧ l < 1 ∧ ∑' n, u n = l := by
  sorry

end series_converges_l11_11309


namespace required_moles_h2so4_l11_11755

-- Defining chemical equation conditions
def balanced_reaction (nacl h2so4 hcl nahso4 : ℕ) : Prop :=
  nacl = h2so4 ∧ hcl = nacl ∧ nahso4 = nacl

-- Theorem statement
theorem required_moles_h2so4 (nacl_needed moles_h2so4 : ℕ) (hcl_produced nahso4_produced : ℕ)
  (h : nacl_needed = 2 ∧ balanced_reaction nacl_needed moles_h2so4 hcl_produced nahso4_produced) :
  moles_h2so4 = 2 :=
  sorry

end required_moles_h2so4_l11_11755


namespace parallel_vectors_cosine_identity_l11_11824

-- Defining the problem in Lean 4

theorem parallel_vectors_cosine_identity :
  ∀ α : ℝ, (∃ k : ℝ, (1 / 3, Real.tan α) = (k * Real.cos α, k)) →
  Real.cos (Real.pi / 2 + α) = -1 / 3 :=
by
  sorry

end parallel_vectors_cosine_identity_l11_11824


namespace sufficient_condition_for_inequality_l11_11151

open Real

theorem sufficient_condition_for_inequality (a : ℝ) (h : 0 < a ∧ a < 1 / 5) : 1 / a > 3 :=
by
  sorry

end sufficient_condition_for_inequality_l11_11151


namespace minimum_value_y_is_2_l11_11983

noncomputable def minimum_value_y (x : ℝ) : ℝ :=
  x + (1 / x)

theorem minimum_value_y_is_2 (x : ℝ) (hx : 0 < x) : 
  (∀ y, y = minimum_value_y x → y ≥ 2) :=
by
  sorry

end minimum_value_y_is_2_l11_11983


namespace christina_age_fraction_l11_11211

theorem christina_age_fraction {C : ℕ} (h1 : ∃ C : ℕ, (6 + 15) = (3/5 : ℚ) * C)
  (h2 : C + 5 = 40) : (C + 5) / 80 = 1 / 2 :=
by
  sorry

end christina_age_fraction_l11_11211


namespace time_to_finish_by_p_l11_11876

theorem time_to_finish_by_p (P_rate Q_rate : ℝ) (worked_together_hours remaining_job_rate : ℝ) :
    P_rate = 1/3 ∧ Q_rate = 1/9 ∧ worked_together_hours = 2 ∧ remaining_job_rate = 1 - (worked_together_hours * (P_rate + Q_rate)) → 
    (remaining_job_rate / P_rate) * 60 = 20 := 
by
  sorry

end time_to_finish_by_p_l11_11876


namespace income_left_at_end_of_year_l11_11409

variable (I : ℝ) -- Monthly income at the beginning of the year
variable (food_expense : ℝ := 0.35 * I) 
variable (education_expense : ℝ := 0.25 * I)
variable (transportation_expense : ℝ := 0.15 * I)
variable (medical_expense : ℝ := 0.10 * I)
variable (initial_expenses : ℝ := food_expense + education_expense + transportation_expense + medical_expense)
variable (remaining_income : ℝ := I - initial_expenses)
variable (house_rent : ℝ := 0.80 * remaining_income)

variable (annual_income : ℝ := 12 * I)
variable (annual_expenses : ℝ := 12 * (initial_expenses + house_rent))

variable (increased_food_expense : ℝ := food_expense * 1.05)
variable (increased_education_expense : ℝ := education_expense * 1.05)
variable (increased_transportation_expense : ℝ := transportation_expense * 1.05)
variable (increased_medical_expense : ℝ := medical_expense * 1.05)
variable (total_increased_expenses : ℝ := increased_food_expense + increased_education_expense + increased_transportation_expense + increased_medical_expense)

variable (new_income : ℝ := 1.10 * I)
variable (new_remaining_income : ℝ := new_income - total_increased_expenses)

variable (new_house_rent : ℝ := 0.80 * new_remaining_income)

variable (final_remaining_income : ℝ := new_income - (total_increased_expenses + new_house_rent))

theorem income_left_at_end_of_year : 
  final_remaining_income / new_income * 100 = 2.15 := 
  sorry

end income_left_at_end_of_year_l11_11409


namespace pool_buckets_l11_11035

theorem pool_buckets (buckets_george_per_round buckets_harry_per_round rounds : ℕ) 
  (h_george : buckets_george_per_round = 2) 
  (h_harry : buckets_harry_per_round = 3) 
  (h_rounds : rounds = 22) : 
  buckets_george_per_round + buckets_harry_per_round * rounds = 110 := 
by 
  sorry

end pool_buckets_l11_11035


namespace problem_solution_l11_11218

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4}

-- Define set A
def A : Set ℕ := {1, 2}

-- Define set B
def B : Set ℕ := {2}

-- Define the complement function specific to our universal set U
def complement (U A : Set ℕ) : Set ℕ := {x ∈ U | x ∉ A}

-- Lean theorem to prove the given problem's correctness
theorem problem_solution : complement U (A ∪ B) = {3, 4} :=
by
  sorry -- Proof is omitted as per the instructions

end problem_solution_l11_11218


namespace factor_expression_l11_11763

theorem factor_expression (x : ℕ) : 75 * x + 45 = 15 * (5 * x + 3) :=
by
  sorry

end factor_expression_l11_11763


namespace money_distribution_l11_11380

variable (A B C : ℕ)

theorem money_distribution :
  A + B + C = 500 →
  B + C = 360 →
  C = 60 →
  A + C = 200 :=
by
  intros h1 h2 h3
  sorry

end money_distribution_l11_11380


namespace midpoint_quadrilateral_inequality_l11_11516

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

end midpoint_quadrilateral_inequality_l11_11516


namespace number_of_girls_sampled_in_third_grade_l11_11805

-- Number of total students in the high school
def total_students : ℕ := 3000

-- Number of students in each grade
def first_grade_students : ℕ := 800
def second_grade_students : ℕ := 1000
def third_grade_students : ℕ := 1200

-- Number of boys and girls in each grade
def first_grade_boys : ℕ := 500
def first_grade_girls : ℕ := 300

def second_grade_boys : ℕ := 600
def second_grade_girls : ℕ := 400

def third_grade_boys : ℕ := 800
def third_grade_girls : ℕ := 400

-- Total number of students sampled
def total_sampled_students : ℕ := 150

-- Hypothesis: stratified sampling method according to grade proportions
theorem number_of_girls_sampled_in_third_grade :
  third_grade_girls * (total_sampled_students / total_students) = 20 :=
by
  -- We will add the proof here
  sorry

end number_of_girls_sampled_in_third_grade_l11_11805


namespace inequality_one_solution_l11_11546

theorem inequality_one_solution (a : ℝ) :
  (∀ x : ℝ, |x^2 + 2 * a * x + 4 * a| ≤ 4 → x = -a) ↔ a = 2 :=
by sorry

end inequality_one_solution_l11_11546


namespace find_quadratic_function_l11_11339

noncomputable def quadratic_function (a b : ℝ) (x : ℝ) : ℝ := x^2 + a * x + b

theorem find_quadratic_function (a b : ℝ) :
  (∀ x, (quadratic_function a b (quadratic_function a b x - x)) / (quadratic_function a b x) = x^2 + 2023 * x + 1777) →
  a = 2025 ∧ b = 249 :=
by
  intro h
  sorry

end find_quadratic_function_l11_11339


namespace max_cars_quotient_div_10_l11_11097

theorem max_cars_quotient_div_10 (n : ℕ) (h1 : ∀ v : ℕ, v ≥ 20 * n) (h2 : ∀ d : ℕ, d = 5* (n + 1)) :
  (4000 / 10 = 400) := by
  sorry

end max_cars_quotient_div_10_l11_11097


namespace total_marbles_l11_11715

-- Definitions based on the given conditions
def jars : ℕ := 16
def pots : ℕ := jars / 2
def marbles_in_jar : ℕ := 5
def marbles_in_pot : ℕ := 3 * marbles_in_jar

-- Main statement to be proved
theorem total_marbles : 
  5 * jars + marbles_in_pot * pots = 200 := 
by
  sorry

end total_marbles_l11_11715


namespace acute_triangle_sec_csc_inequality_l11_11921

theorem acute_triangle_sec_csc_inequality (A B C : ℝ) (h : A + B + C = π) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) (hA90 : A < π / 2) (hB90 : B < π / 2) (hC90 : C < π / 2) :
  (1 / Real.cos A) + (1 / Real.cos B) + (1 / Real.cos C) ≥
  (1 / Real.sin (A / 2)) + (1 / Real.sin (B / 2)) + (1 / Real.sin (C / 2)) :=
by sorry

end acute_triangle_sec_csc_inequality_l11_11921


namespace sum_of_ages_in_5_years_l11_11522

noncomputable def age_will_three_years_ago := 4
noncomputable def years_elapsed := 3
noncomputable def age_will_now := age_will_three_years_ago + years_elapsed
noncomputable def age_diane_now := 2 * age_will_now
noncomputable def years_into_future := 5
noncomputable def age_will_in_future := age_will_now + years_into_future
noncomputable def age_diane_in_future := age_diane_now + years_into_future

theorem sum_of_ages_in_5_years :
  age_will_in_future + age_diane_in_future = 31 := by
  sorry

end sum_of_ages_in_5_years_l11_11522


namespace possible_distances_between_andrey_and_gleb_l11_11602

theorem possible_distances_between_andrey_and_gleb (A B V G : Point) 
  (d_AB : ℝ) (d_VG : ℝ) (d_BV : ℝ) (d_AG : ℝ)
  (h1 : d_AB = 600) 
  (h2 : d_VG = 600) 
  (h3 : d_AG = 3 * d_BV) : 
  d_AG = 900 ∨ d_AG = 1800 :=
by {
  sorry
}

end possible_distances_between_andrey_and_gleb_l11_11602


namespace new_average_weight_l11_11575

theorem new_average_weight (avg_weight_19_students : ℝ) (new_student_weight : ℝ) (num_students_initial : ℕ) : 
  avg_weight_19_students = 15 → new_student_weight = 7 → num_students_initial = 19 → 
  let total_weight_with_new_student := (avg_weight_19_students * num_students_initial + new_student_weight) 
  let new_num_students := num_students_initial + 1 
  let new_avg_weight := total_weight_with_new_student / new_num_students 
  new_avg_weight = 14.6 :=
by
  intros h1 h2 h3
  let total_weight := avg_weight_19_students * num_students_initial
  let total_weight_with_new_student := total_weight + new_student_weight
  let new_num_students := num_students_initial + 1
  let new_avg_weight := total_weight_with_new_student / new_num_students
  have h4 : total_weight = 285 := by sorry
  have h5 : total_weight_with_new_student = 292 := by sorry
  have h6 : new_num_students = 20 := by sorry
  have h7 : new_avg_weight = 292 / 20 := by sorry
  have h8 : new_avg_weight = 14.6 := by sorry
  exact h8

end new_average_weight_l11_11575


namespace rightmost_three_digits_of_7_pow_1987_l11_11266

theorem rightmost_three_digits_of_7_pow_1987 :
  7^1987 % 1000 = 543 :=
by
  sorry

end rightmost_three_digits_of_7_pow_1987_l11_11266


namespace total_money_given_by_father_is_100_l11_11819

-- Define the costs and quantities given in the problem statement.
def cost_per_sharpener := 5
def cost_per_notebook := 5
def cost_per_eraser := 4
def money_spent_on_highlighters := 30

def heaven_sharpeners := 2
def heaven_notebooks := 4
def brother_erasers := 10

-- Calculate the total amount of money given by their father.
def total_money_given : ℕ :=
  heaven_sharpeners * cost_per_sharpener +
  heaven_notebooks * cost_per_notebook +
  brother_erasers * cost_per_eraser +
  money_spent_on_highlighters

-- Lean statement to prove
theorem total_money_given_by_father_is_100 :
  total_money_given = 100 := by
  sorry

end total_money_given_by_father_is_100_l11_11819


namespace trigon_expr_correct_l11_11361

noncomputable def trigon_expr : ℝ :=
  1 / Real.sin (Real.pi / 6) - 4 * Real.sin (Real.pi / 3)

theorem trigon_expr_correct : trigon_expr = 2 - 2 * Real.sqrt 3 := by
  sorry

end trigon_expr_correct_l11_11361


namespace countColorings_l11_11101

-- Defining the function that counts the number of valid colorings
def validColorings (n : ℕ) : ℕ :=
  if n = 0 then 1
  else 3 * 2^n - 2

-- Theorem specifying the number of colorings of the grid of length n
theorem countColorings (n : ℕ) : validColorings n = 3 * 2^n - 2 :=
by
  sorry

end countColorings_l11_11101


namespace time_to_clear_l11_11573

def length_train1 := 121 -- in meters
def length_train2 := 153 -- in meters
def speed_train1 := 80 * 1000 / 3600 -- converting km/h to meters/s
def speed_train2 := 65 * 1000 / 3600 -- converting km/h to meters/s

def total_distance := length_train1 + length_train2
def relative_speed := speed_train1 + speed_train2

theorem time_to_clear : 
  (total_distance / relative_speed : ℝ) = 6.80 :=
by
  sorry

end time_to_clear_l11_11573


namespace quadratic_maximum_or_minimum_l11_11164

open Real

noncomputable def quadratic_function (a b x : ℝ) : ℝ := a * x^2 + b * x - b^2 / (3 * a)

theorem quadratic_maximum_or_minimum (a b : ℝ) (h : a ≠ 0) :
  (a > 0 → ∃ x₀, ∀ x, quadratic_function a b x₀ ≤ quadratic_function a b x) ∧
  (a < 0 → ∃ x₀, ∀ x, quadratic_function a b x₀ ≥ quadratic_function a b x) :=
by
  -- Proof will go here
  sorry

end quadratic_maximum_or_minimum_l11_11164


namespace inequality_may_not_hold_l11_11565

theorem inequality_may_not_hold (a b c : ℝ) (h : a > b) : (c < 0) → ¬ (a/c > b/c) := 
sorry

end inequality_may_not_hold_l11_11565


namespace find_number_l11_11674

theorem find_number 
  (x y n : ℝ)
  (h1 : n * x = 0.04 * y)
  (h2 : (y - x) / (y + x) = 0.948051948051948) :
  n = 37.5 :=
sorry  -- proof omitted

end find_number_l11_11674


namespace tangent_line_at_origin_tangent_line_passing_through_neg1_neg3_l11_11174

noncomputable def f (x : ℝ) : ℝ := x^3 + 2 * x

noncomputable def f' (x : ℝ) : ℝ := 3 * x^2 + 2

theorem tangent_line_at_origin :
  (∀ x y : ℝ, y = f x → x = 0 → y = 2 * x) := by
  sorry

theorem tangent_line_passing_through_neg1_neg3 :
  (∀ x y : ℝ, y = f x → (x, y) ≠ (-1, -3) → y = 5 * x + 2) := by
  sorry

end tangent_line_at_origin_tangent_line_passing_through_neg1_neg3_l11_11174


namespace even_integers_in_form_3k_plus_4_l11_11835

theorem even_integers_in_form_3k_plus_4 (n : ℕ) :
  (20 ≤ n ∧ n ≤ 180 ∧ ∃ k : ℕ, n = 3 * k + 4) → 
  (∃ s : ℕ, s = 27) :=
by
  sorry

end even_integers_in_form_3k_plus_4_l11_11835


namespace necessary_but_not_sufficient_condition_l11_11710

variable {a b : ℝ}

theorem necessary_but_not_sufficient_condition
    (h1 : a ≠ 0)
    (h2 : b ≠ 0) :
    (a^2 + b^2 ≥ 2 * a * b) → 
    (¬(a^2 + b^2 ≥ 2 * a * b) → ¬(a / b + b / a ≥ 2)) ∧ 
    ((a / b + b / a ≥ 2) → (a^2 + b^2 ≥ 2 * a * b)) :=
sorry

end necessary_but_not_sufficient_condition_l11_11710


namespace tanya_total_sticks_l11_11054

theorem tanya_total_sticks (n : ℕ) (h : n = 11) : 3 * (n * (n + 1) / 2) = 198 :=
by
  have H : n = 11 := h
  sorry

end tanya_total_sticks_l11_11054


namespace isosceles_triangle_median_length_l11_11061

noncomputable def median_length (b h : ℝ) : ℝ :=
  let a := Real.sqrt ((b / 2) ^ 2 + h ^ 2)
  let m_a := Real.sqrt ((2 * a ^ 2 + 2 * b ^ 2 - a ^ 2) / 4)
  m_a

theorem isosceles_triangle_median_length :
  median_length 16 10 = Real.sqrt 146 :=
by
  sorry

end isosceles_triangle_median_length_l11_11061


namespace value_of_f_at_2_l11_11374

def f (x : ℝ) := x^2 + 2 * x - 3

theorem value_of_f_at_2 : f 2 = 5 :=
by
  sorry

end value_of_f_at_2_l11_11374


namespace elder_age_is_33_l11_11576

-- Define the conditions
variables (y e : ℕ)

def age_difference_condition : Prop :=
  e = y + 20

def age_reduced_condition : Prop :=
  e - 8 = 5 * (y - 8)

-- State the theorem to prove the age of the elder person
theorem elder_age_is_33 (h1 : age_difference_condition y e) (h2 : age_reduced_condition y e): e = 33 :=
  sorry

end elder_age_is_33_l11_11576


namespace option_D_functions_same_l11_11656

theorem option_D_functions_same (x : ℝ) : (x^2) = (x^6)^(1/3) :=
by 
  sorry

end option_D_functions_same_l11_11656


namespace quadratic_minimum_value_interval_l11_11833

theorem quadratic_minimum_value_interval (k : ℝ) :
  (∀ (x : ℝ), 0 ≤ x ∧ x < 2 → (x^2 - 4*k*x + 4*k^2 + 2*k - 1) ≥ (2*k^2 + 2*k - 1)) → (0 ≤ k ∧ k < 1) :=
by {
  sorry
}

end quadratic_minimum_value_interval_l11_11833


namespace range_of_a_l11_11055

theorem range_of_a (x a : ℝ) (p : Prop) (q : Prop) (H₁ : p ↔ (x < -3 ∨ x > 1))
  (H₂ : q ↔ (x > a))
  (H₃ : ¬p → ¬q) (H₄ : ¬q → ¬p → false) : a ≥ 1 :=
sorry

end range_of_a_l11_11055


namespace tank_capacity_l11_11094

theorem tank_capacity (T : ℚ) (h1 : 0 ≤ T)
  (h2 : 9 + (3 / 4) * T = (9 / 10) * T) : T = 60 :=
sorry

end tank_capacity_l11_11094


namespace sin_45_deg_l11_11508

theorem sin_45_deg : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := 
by 
  -- placeholder for the actual proof
  sorry

end sin_45_deg_l11_11508


namespace total_opponent_score_l11_11034

-- Definitions based on the conditions
def team_scores : List ℕ := [2, 3, 4, 5, 6, 7, 8, 9]

def lost_by_one_point (scores : List ℕ) : Bool :=
  scores = [3, 4, 5]

def scored_twice_as_many (scores : List ℕ) : Bool :=
  scores = [2, 3]

def scored_three_times_as_many (scores : List ℕ) : Bool :=
  scores = [2, 3, 3]

-- Proof problem:
theorem total_opponent_score :
  ∀ (lost_scores twice_scores thrice_scores : List ℕ),
    lost_by_one_point lost_scores →
    scored_twice_as_many twice_scores →
    scored_three_times_as_many thrice_scores →
    (lost_scores.sum + twice_scores.sum + thrice_scores.sum) = 25 :=
by
  intros
  sorry

end total_opponent_score_l11_11034


namespace inverse_value_ratio_l11_11972

noncomputable def g (x : ℚ) : ℚ := (3 * x + 1) / (x - 4)

theorem inverse_value_ratio :
  (∃ (a b c d : ℚ), ∀ x, g ((a * x + b) / (c * x + d)) = x) → ∃ a c : ℚ, a / c = -4 :=
by
  sorry

end inverse_value_ratio_l11_11972


namespace divisibility_polynomial_l11_11870

variables {a m x n : ℕ}

theorem divisibility_polynomial (a m x n : ℕ) :
  m ∣ n ↔ (x^m - a^m) ∣ (x^n - a^n) :=
by
  sorry

end divisibility_polynomial_l11_11870


namespace range_of_a_l11_11816

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x < 0 ∧ 5^x = (a + 3) / (a - 3)) → a < -3 :=
by
  sorry

end range_of_a_l11_11816


namespace total_revenue_full_price_tickets_l11_11461

theorem total_revenue_full_price_tickets (f q : ℕ) (p : ℝ) :
  f + q = 170 ∧ f * p + q * (p / 4) = 2917 → f * p = 1748 := by
  sorry

end total_revenue_full_price_tickets_l11_11461


namespace rectangle_error_percent_deficit_l11_11027

theorem rectangle_error_percent_deficit (L W : ℝ) (p : ℝ) 
    (h1 : L > 0) (h2 : W > 0)
    (h3 : 1.05 * (1 - p) = 1.008) :
    p = 0.04 :=
by
  sorry

end rectangle_error_percent_deficit_l11_11027


namespace age_difference_l11_11278

theorem age_difference (A B C : ℕ) (h : A + B = B + C + 18) : A = C + 18 :=
by
  sorry

end age_difference_l11_11278


namespace five_fourths_of_twelve_fifths_eq_three_l11_11523

theorem five_fourths_of_twelve_fifths_eq_three : (5 : ℝ) / 4 * (12 / 5) = 3 := 
by 
  sorry

end five_fourths_of_twelve_fifths_eq_three_l11_11523


namespace simplify_expression_l11_11074

variable (x : ℝ)

def expr := (5*x^10 + 8*x^8 + 3*x^6) + (2*x^12 + 3*x^10 + x^8 + 4*x^6 + 2*x^2 + 7)

theorem simplify_expression : expr x = 2*x^12 + 8*x^10 + 9*x^8 + 7*x^6 + 2*x^2 + 7 :=
by
  sorry

end simplify_expression_l11_11074


namespace canned_food_total_bins_l11_11005

theorem canned_food_total_bins :
  let soup_bins := 0.125
  let vegetable_bins := 0.125
  let pasta_bins := 0.5
  soup_bins + vegetable_bins + pasta_bins = 0.75 := 
by
  sorry

end canned_food_total_bins_l11_11005


namespace determinant_real_root_unique_l11_11406

theorem determinant_real_root_unique {a b c : ℝ} (ha : 0 < a ∧ a ≠ 1) (hb : 0 < b ∧ b ≠ 1) (hc : 0 < c ∧ c ≠ 1) :
  ∃! x : ℝ, (Matrix.det ![
    ![x - 1, c - 1, -(b - 1)],
    ![-(c - 1), x - 1, a - 1],
    ![b - 1, -(a - 1), x - 1]
  ]) = 0 :=
by
  sorry

end determinant_real_root_unique_l11_11406


namespace scarf_cost_is_10_l11_11974

-- Define the conditions as given in the problem statement
def initial_amount : ℕ := 53
def cost_per_toy_car : ℕ := 11
def num_toy_cars : ℕ := 2
def cost_of_beanie : ℕ := 14
def remaining_after_beanie : ℕ := 7

-- Calculate the cost of the toy cars
def total_cost_toy_cars : ℕ := num_toy_cars * cost_per_toy_car

-- Calculate the amount left after buying the toy cars
def amount_after_toys : ℕ := initial_amount - total_cost_toy_cars

-- Calculate the amount left after buying the beanie
def amount_after_beanie : ℕ := amount_after_toys - cost_of_beanie

-- Define the cost of the scarf
def cost_of_scarf : ℕ := amount_after_beanie - remaining_after_beanie

-- The theorem stating that cost_of_scarf is 10 dollars
theorem scarf_cost_is_10 : cost_of_scarf = 10 := by
  sorry

end scarf_cost_is_10_l11_11974


namespace solve_inequality_l11_11292

theorem solve_inequality (x : ℝ) : abs ((3 - x) / 4) < 1 ↔ 2 < x ∧ x < 7 :=
by {
  sorry
}

end solve_inequality_l11_11292


namespace ann_top_cost_l11_11858

noncomputable def cost_per_top (T : ℝ) := 75 = (5 * 7) + (2 * 10) + (4 * T)

theorem ann_top_cost : cost_per_top 5 :=
by {
  -- statement: prove cost per top given conditions
  sorry
}

end ann_top_cost_l11_11858


namespace Xiao_Ming_min_steps_l11_11264

-- Problem statement: Prove that the minimum number of steps Xiao Ming needs to move from point A to point B is 5,
-- given his movement pattern and the fact that he can reach eight different positions from point C.

def min_steps_from_A_to_B : ℕ :=
  5

theorem Xiao_Ming_min_steps (A B C : Type) (f : A → B → C) : 
  (min_steps_from_A_to_B = 5) :=
by
  sorry

end Xiao_Ming_min_steps_l11_11264


namespace substance_volume_proportional_l11_11184

theorem substance_volume_proportional (k : ℝ) (V₁ V₂ : ℝ) (W₁ W₂ : ℝ) 
  (h1 : V₁ = k * W₁) 
  (h2 : V₂ = k * W₂) 
  (h3 : V₁ = 48) 
  (h4 : W₁ = 112) 
  (h5 : W₂ = 84) 
  : V₂ = 36 := 
  sorry

end substance_volume_proportional_l11_11184


namespace regular_polygon_sides_l11_11832

theorem regular_polygon_sides (D : ℕ) (h : D = 30) :
  ∃ n : ℕ, D = n * (n - 3) / 2 ∧ n = 9 :=
by
  use 9
  rw [h]
  norm_num
  sorry

end regular_polygon_sides_l11_11832


namespace warehouse_capacity_l11_11689

theorem warehouse_capacity (total_bins num_20_ton_bins cap_20_ton_bin cap_15_ton_bin : Nat) 
  (h1 : total_bins = 30) 
  (h2 : num_20_ton_bins = 12) 
  (h3 : cap_20_ton_bin = 20) 
  (h4 : cap_15_ton_bin = 15) : 
  total_bins * cap_20_ton_bin + (total_bins - num_20_ton_bins) * cap_15_ton_bin = 510 := 
by
  sorry

end warehouse_capacity_l11_11689


namespace runners_meetings_on_track_l11_11825

def number_of_meetings (speed1 speed2 laps : ℕ) : ℕ := ((speed1 + speed2) * laps) / (2 * (speed2 - speed1))

theorem runners_meetings_on_track 
  (speed1 speed2 : ℕ) 
  (start_laps : ℕ)
  (speed1_spec : speed1 = 4) 
  (speed2_spec : speed2 = 10) 
  (laps_spec : start_laps = 28) : 
  number_of_meetings speed1 speed2 start_laps = 77 := 
by
  rw [speed1_spec, speed2_spec, laps_spec]
  -- Add further necessary steps or lemmas if required to reach the final proving statement
  sorry

end runners_meetings_on_track_l11_11825


namespace unique_pairs_pos_int_satisfy_eq_l11_11863

theorem unique_pairs_pos_int_satisfy_eq (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) :
  a^(b^2) = b^a ↔ (a = 1 ∧ b = 1) ∨ (a = 16 ∧ b = 2) ∨ (a = 27 ∧ b = 3) := 
by
  sorry

end unique_pairs_pos_int_satisfy_eq_l11_11863


namespace amoeba_population_after_ten_days_l11_11792

-- Definitions based on the conditions
def initial_population : ℕ := 3
def amoeba_growth (n : ℕ) : ℕ := initial_population * 2^n

-- Lean statement for the proof problem
theorem amoeba_population_after_ten_days : amoeba_growth 10 = 3072 :=
by 
  sorry

end amoeba_population_after_ten_days_l11_11792


namespace binomial_expansion_of_110_minus_1_l11_11582

theorem binomial_expansion_of_110_minus_1:
  110^5 - 5 * 110^4 + 10 * 110^3 - 10 * 110^2 + 5 * 110 - 1 = 109^5 :=
by
  -- We will use the binomial theorem: (a - b)^n = ∑ (k in range(n+1)), C(n, k) * a^(n-k) * (-b)^k
  -- where C(n, k) are the binomial coefficients.
  sorry

end binomial_expansion_of_110_minus_1_l11_11582


namespace tan_ratio_l11_11933

-- Definitions of the problem conditions
variables {A B C : ℝ} -- Angles of the triangle
variables {a b c : ℝ} -- Sides opposite to the angles

-- The given equation condition
axiom h : a * Real.cos B - b * Real.cos A = (4 / 5) * c

-- The goal is to prove the value of tan(A) / tan(B)
theorem tan_ratio (A B C : ℝ) (a b c : ℝ) (h : a * Real.cos B - b * Real.cos A = (4 / 5) * c) :
  Real.tan A / Real.tan B = 9 :=
sorry

end tan_ratio_l11_11933


namespace am_gm_inequality_l11_11254

theorem am_gm_inequality (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) : 
  (a^2 * b + b^2 * c + c^2 * a) * (a * b^2 + b * c^2 + c * a^2) ≥ 9 * (a * b * c)^2 :=
by
  sorry

end am_gm_inequality_l11_11254


namespace six_circles_distance_relation_l11_11703

/--
Prove that for any pair of non-touching circles (among six circles where each touches four of the remaining five),
their radii \( r_1 \) and \( r_2 \) and the distance \( d \) between their centers satisfy 

\[ d^{2}=r_{1}^{2}+r_{2}^{2} \pm 6r_{1}r_{2} \]

("plus" if the circles do not lie inside one another, "minus" otherwise).
-/
theorem six_circles_distance_relation 
  (r1 r2 d : ℝ) 
  (h : ∀ i : Fin 6, i < 6 → ∃ c : ℝ, (c = r1 ∨ c = r2) ∧ ∀ j : Fin 6, j ≠ i → abs (c - j) ≠ d ) :
  d^2 = r1^2 + r2^2 + 6 * r1 * r2 ∨ d^2 = r1^2 + r2^2 - 6 * r1 * r2 := 
  sorry

end six_circles_distance_relation_l11_11703


namespace parallel_lines_d_l11_11434

theorem parallel_lines_d (d : ℝ) : (∀ x : ℝ, -3 * x + 5 = (-6 * d) * x + 10) → d = 1 / 2 :=
by sorry

end parallel_lines_d_l11_11434


namespace unique_triplet_satisfying_conditions_l11_11542

theorem unique_triplet_satisfying_conditions :
  ∃! (a b c: ℕ), 1 < a ∧ 1 < b ∧ 1 < c ∧
                 (c ∣ a * b + 1) ∧
                 (b ∣ c * a + 1) ∧
                 (a ∣ b * c + 1) ∧
                 a = 2 ∧ b = 3 ∧ c = 7 :=
by
  sorry

end unique_triplet_satisfying_conditions_l11_11542


namespace find_a_plus_d_l11_11888

theorem find_a_plus_d (a b c d : ℕ)
  (h1 : a + b = 14)
  (h2 : b + c = 9)
  (h3 : c + d = 3) : 
  a + d = 2 :=
by sorry

end find_a_plus_d_l11_11888


namespace trivia_team_missing_members_l11_11487

theorem trivia_team_missing_members 
  (total_members : ℕ)
  (points_per_member : ℕ)
  (total_points : ℕ)
  (showed_up_members : ℕ)
  (missing_members : ℕ) 
  (h1 : total_members = 15) 
  (h2 : points_per_member = 3) 
  (h3 : total_points = 27) 
  (h4 : showed_up_members = total_points / points_per_member) 
  (h5 : missing_members = total_members - showed_up_members) : 
  missing_members = 6 :=
by
  sorry

end trivia_team_missing_members_l11_11487


namespace cannot_determine_red_marbles_l11_11312

variable (Jason_blue : ℕ) (Tom_blue : ℕ) (Total_blue : ℕ)

-- Conditions
axiom Jason_has_44_blue : Jason_blue = 44
axiom Tom_has_24_blue : Tom_blue = 24
axiom Together_have_68_blue : Total_blue = 68

theorem cannot_determine_red_marbles (Jason_blue Tom_blue Total_blue : ℕ) : ¬ ∃ (Jason_red : ℕ), True := by
  sorry

end cannot_determine_red_marbles_l11_11312


namespace girls_in_school_play_l11_11988

theorem girls_in_school_play (G : ℕ) (boys : ℕ) (total_parents : ℕ)
  (h1 : boys = 8) (h2 : total_parents = 28) (h3 : 2 * boys + 2 * G = total_parents) : 
  G = 6 :=
sorry

end girls_in_school_play_l11_11988


namespace problem_statement_l11_11288

variable { a b c x y z : ℝ }

theorem problem_statement 
  (h1 : (a + b + c) * (x + y + z) = 3)
  (h2 : (a^2 + b^2 + c^2) * (x^2 + y^2 + z^2) = 4) : 
  a * x + b * y + c * z ≥ 0 :=
by 
  sorry

end problem_statement_l11_11288


namespace boots_cost_5_more_than_shoes_l11_11125

variable (S B : ℝ)

-- Conditions based on the problem statement
axiom h1 : 22 * S + 16 * B = 460
axiom h2 : 8 * S + 32 * B = 560

/-- Theorem to prove that the difference in cost between pairs of boots and pairs of shoes is $5 --/
theorem boots_cost_5_more_than_shoes : B - S = 5 :=
by
  sorry

end boots_cost_5_more_than_shoes_l11_11125


namespace custom_op_equality_l11_11630

def custom_op (x y : Int) : Int :=
  x * y - 2 * x

theorem custom_op_equality : custom_op 5 3 - custom_op 3 5 = -4 := by
  sorry

end custom_op_equality_l11_11630


namespace find_original_number_l11_11610

-- Given definitions and conditions
def doubled_add_nine (x : ℝ) : ℝ := 2 * x + 9
def trebled (y : ℝ) : ℝ := 3 * y

-- The proof problem we need to solve
theorem find_original_number (x : ℝ) (h : trebled (doubled_add_nine x) = 69) : x = 7 := 
by sorry

end find_original_number_l11_11610


namespace total_earnings_l11_11053

noncomputable def daily_wage_a (C : ℝ) := (3 * C) / 5
noncomputable def daily_wage_b (C : ℝ) := (4 * C) / 5
noncomputable def daily_wage_c (C : ℝ) := C

noncomputable def earnings_a (C : ℝ) := daily_wage_a C * 6
noncomputable def earnings_b (C : ℝ) := daily_wage_b C * 9
noncomputable def earnings_c (C : ℝ) := daily_wage_c C * 4

theorem total_earnings (C : ℝ) (h : C = 115) : 
  earnings_a C + earnings_b C + earnings_c C = 1702 :=
by
  sorry

end total_earnings_l11_11053


namespace factor_expression_l11_11577

theorem factor_expression (z : ℤ) : 55 * z^17 + 121 * z^34 = 11 * z^17 * (5 + 11 * z^17) := 
by sorry

end factor_expression_l11_11577


namespace converse_proposition_true_l11_11167

theorem converse_proposition_true (x y : ℝ) (h : x > abs y) : x > y := 
by
sorry

end converse_proposition_true_l11_11167


namespace parts_per_hour_l11_11639

variables {x y : ℕ}

-- Condition 1: The time it takes for A to make 90 parts is the same as the time it takes for B to make 120 parts.
def time_ratio (x y : ℕ) := (x:ℚ) / y = 90 / 120

-- Condition 2: A and B together make 35 parts per hour.
def total_parts_per_hour (x y : ℕ) := x + y = 35

-- Given the conditions, prove the number of parts A and B each make per hour.
theorem parts_per_hour (x y : ℕ) (h1 : time_ratio x y) (h2 : total_parts_per_hour x y) : x = 15 ∧ y = 20 :=
by
  sorry

end parts_per_hour_l11_11639


namespace hyperbola_equation_l11_11180

theorem hyperbola_equation (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) 
  (eccentricity : Real.sqrt 2 = b / a)
  (line_through_FP_parallel_to_asymptote : ∃ c : ℝ, c = Real.sqrt 2 * a ∧ ∀ P : ℝ × ℝ, P = (0, 4) → (P.2 - 0) / (P.1 + c) = 1) :
  (∃ (a b : ℝ), a = b ∧ (a = 2 * Real.sqrt 2 ∧ b = 2 * Real.sqrt 2)) ∧
  (a = 2 * Real.sqrt 2 ∧ b = 2 * Real.sqrt 2) → 
  (∃ x y : ℝ, ((x^2 / 8) - (y^2 / 8) = 1)) :=
by
  sorry

end hyperbola_equation_l11_11180


namespace simplify_expression_l11_11775

theorem simplify_expression (x y : ℤ) (h1 : x = -2) (h2 : y = -1) :
  (2 * (x - 2 * y) * (2 * x + y) - (x + 2 * y)^2 + x * (8 * y - 3 * x)) / (6 * y) = 2 :=
by sorry

end simplify_expression_l11_11775


namespace number_of_triangles_with_perimeter_27_l11_11634

theorem number_of_triangles_with_perimeter_27 : 
  ∃ (n : ℕ), (∀ (a b c : ℕ), a ≤ b ∧ b ≤ c ∧ a + b + c = 27 → a + b > c ∧ a + c > b ∧ b + c > a → 
  n = 19 ) :=
  sorry

end number_of_triangles_with_perimeter_27_l11_11634


namespace arithmetic_sequence_15th_term_l11_11956

/-- 
The arithmetic sequence with first term 1 and common difference 3.
The 15th term of this sequence is 43.
-/
theorem arithmetic_sequence_15th_term :
  ∀ (a1 d n : ℕ), a1 = 1 → d = 3 → n = 15 → (a1 + (n - 1) * d) = 43 :=
by
  sorry

end arithmetic_sequence_15th_term_l11_11956


namespace negation_of_universal_proposition_l11_11249
open Real

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, sin x ≤ 1) ↔ ∃ x : ℝ, sin x > 1 :=
by
  sorry

end negation_of_universal_proposition_l11_11249


namespace abs_eq_2_iff_l11_11451

theorem abs_eq_2_iff (a : ℚ) : abs a = 2 ↔ a = 2 ∨ a = -2 :=
by
  sorry

end abs_eq_2_iff_l11_11451


namespace no_real_solution_l11_11189

-- Define the equation
def equation (a b : ℝ) : Prop := a^2 + 3 * b^2 + 2 = 3 * a * b

-- Prove that there do not exist real numbers a and b such that equation a b holds
theorem no_real_solution : ¬ ∃ a b : ℝ, equation a b :=
by
  -- Proof placeholder
  sorry

end no_real_solution_l11_11189


namespace range_of_a_l11_11558

theorem range_of_a (a : ℝ) (h1 : 2 * a + 1 < 17) (h2 : 2 * a + 1 > 7) : 3 < a ∧ a < 8 := by
  sorry

end range_of_a_l11_11558


namespace ordered_quadruple_solution_exists_l11_11916

theorem ordered_quadruple_solution_exists (a b c d : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) : 
  a^2 * b = c ∧ b * c^2 = a ∧ c * a^2 = b ∧ a + b + c = d → (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 3) :=
by
  sorry

end ordered_quadruple_solution_exists_l11_11916


namespace simplify_expr_correct_l11_11432

-- Define the expression
def simplify_expr (z : ℝ) : ℝ := (3 - 5 * z^2) - (5 + 7 * z^2)

-- Prove the simplified form
theorem simplify_expr_correct (z : ℝ) : simplify_expr z = -2 - 12 * z^2 := by
  sorry

end simplify_expr_correct_l11_11432


namespace minimum_value_of_sum_l11_11123

open Real

theorem minimum_value_of_sum {a b : ℝ} (h₁ : 0 < a) (h₂ : 0 < b) (h : log a / log 2 + log b / log 2 ≥ 6) :
  a + b ≥ 16 :=
sorry

end minimum_value_of_sum_l11_11123


namespace simplify_expression_correct_l11_11168

noncomputable def simplify_expression (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0) : ℝ :=
  let expr1 := (a^2 - b^2) / (a^2 + 2 * a * b + b^2)
  let expr2 := (2 : ℝ) / (a * b)
  let expr3 := ((1 : ℝ) / a + (1 : ℝ) / b)^2
  let expr4 := (2 : ℝ) / (a^2 - b^2 + 2 * a * b)
  expr1 + expr2 / expr3 * expr4

theorem simplify_expression_correct (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0) :
  simplify_expression a b h = 2 / (a + b)^2 := by
  sorry

end simplify_expression_correct_l11_11168


namespace factorization_example_l11_11502

open Function

theorem factorization_example (a b : ℤ) :
  (a - 1) * (b - 1) = ab - a - b + 1 :=
by
  sorry

end factorization_example_l11_11502


namespace decreasing_interval_for_function_l11_11614

theorem decreasing_interval_for_function :
  ∀ (f : ℝ → ℝ) (ϕ : ℝ),
  (∀ x, f x = -2 * Real.tan (2 * x + ϕ)) →
  |ϕ| < Real.pi →
  f (Real.pi / 16) = -2 →
  ∃ a b : ℝ, 
  a = 3 * Real.pi / 16 ∧ 
  b = 11 * Real.pi / 16 ∧ 
  ∀ x, a < x ∧ x < b → ∀ y, x < y ∧ y < b → f y < f x :=
by sorry

end decreasing_interval_for_function_l11_11614


namespace marcie_cups_coffee_l11_11220

theorem marcie_cups_coffee (S M T : ℕ) (h1 : S = 6) (h2 : S + M = 8) : M = 2 :=
by
  sorry

end marcie_cups_coffee_l11_11220


namespace remainder_of_x_pow_150_div_by_x_minus_1_cubed_l11_11118

theorem remainder_of_x_pow_150_div_by_x_minus_1_cubed :
  (x : ℤ) → (x^150 % (x - 1)^3) = (11175 * x^2 - 22200 * x + 11026) :=
by
  intro x
  sorry

end remainder_of_x_pow_150_div_by_x_minus_1_cubed_l11_11118


namespace cost_of_outfit_l11_11794

theorem cost_of_outfit (P T J : ℝ) 
  (h1 : 4 * P + 8 * T + 2 * J = 2400)
  (h2 : 2 * P + 14 * T + 3 * J = 2400)
  (h3 : 3 * P + 6 * T = 1500) :
  P + 4 * T + J = 860 := 
sorry

end cost_of_outfit_l11_11794


namespace gcd_two_powers_l11_11236

noncomputable def gcd_expression (m n : ℕ) : ℕ :=
  Int.gcd (2^m + 1) (2^n - 1)

theorem gcd_two_powers (m n : ℕ) (hm : m > 0) (hn : n > 0) (odd_n : n % 2 = 1) : 
  gcd_expression m n = 1 :=
by
  sorry

end gcd_two_powers_l11_11236


namespace log_term_evaluation_l11_11308

theorem log_term_evaluation : (Real.log 2)^2 + (Real.log 5)^2 + 2 * (Real.log 2) * (Real.log 5) = 1 := by
  sorry

end log_term_evaluation_l11_11308


namespace total_children_l11_11505

theorem total_children {x y : ℕ} (h₁ : x = 18) (h₂ : y = 12) 
  (h₃ : x + y = 30) (h₄ : x = 18) (h₅ : y = 12) : 2 * x + 3 * y = 72 := 
by
  sorry

end total_children_l11_11505


namespace set_elements_l11_11379

def is_divisor (a b : ℤ) : Prop := ∃ k : ℤ, b = k * a

theorem set_elements:
  {x : ℤ | ∃ d : ℤ, is_divisor d 12 ∧ d = 6 - x ∧ x ≥ 0} = 
  {0, 2, 3, 4, 5, 7, 8, 9, 10, 12, 18} :=
by {
  sorry
}

end set_elements_l11_11379


namespace standard_deviation_upper_bound_l11_11777

theorem standard_deviation_upper_bound (Mean StdDev : ℝ) (h : Mean = 54) (h2 : 54 - 3 * StdDev > 47) : StdDev < 2.33 :=
by
  sorry

end standard_deviation_upper_bound_l11_11777


namespace fraction_in_jug_x_after_pouring_water_l11_11666

-- Define capacities and initial fractions
def initial_fraction_x := 1 / 4
def initial_fraction_y := 2 / 3
def fill_needed_y := 1 - initial_fraction_y -- 1/3

-- Define capacity of original jugs
variable (C : ℚ) -- We can assume capacities are rational for simplicity

-- Define initial water amounts in jugs x and y
def initial_water_x := initial_fraction_x * C
def initial_water_y := initial_fraction_y * C

-- Define the water needed to fill jug y
def additional_water_needed_y := fill_needed_y * C

-- Define the final fraction of water in jug x
def final_fraction_x := initial_fraction_x / 2 -- since half of the initial water is poured out

theorem fraction_in_jug_x_after_pouring_water :
  final_fraction_x = 1 / 8 := by
  sorry

end fraction_in_jug_x_after_pouring_water_l11_11666


namespace initial_matchsticks_l11_11469

-- Define the problem conditions
def matchsticks_elvis := 4
def squares_elvis := 5
def matchsticks_ralph := 8
def squares_ralph := 3
def matchsticks_left := 6

-- Calculate the total matchsticks used by Elvis and Ralph
def total_used_elvis := matchsticks_elvis * squares_elvis
def total_used_ralph := matchsticks_ralph * squares_ralph
def total_used := total_used_elvis + total_used_ralph

-- The proof statement
theorem initial_matchsticks (matchsticks_elvis squares_elvis matchsticks_ralph squares_ralph matchsticks_left : ℕ) : total_used + matchsticks_left = 50 := 
by
  sorry

end initial_matchsticks_l11_11469


namespace max_area_quadrilateral_sum_opposite_angles_l11_11759

theorem max_area_quadrilateral (a b c d : ℝ) (h₁ : a = 3) (h₂ : b = 3) (h₃ : c = 4) (h₄ : d = 4) :
  ∃ (area : ℝ), area = 12 :=
by {
  sorry
}

theorem sum_opposite_angles (a b c d : ℝ) (θ₁ θ₂ θ₃ θ₄ : ℝ) 
  (h₁ : a = 3) (h₂ : b = 3) (h₃ : c = 4) (h₄ : d = 4) 
  (h_area : ∃ (area : ℝ), area = 12) 
  (h_opposite1 : θ₁ + θ₃ = 180) (h_opposite2 : θ₂ + θ₄ = 180) :
  ∃ θ, θ = 180 :=
by {
  sorry
}

end max_area_quadrilateral_sum_opposite_angles_l11_11759


namespace minimum_a_l11_11433

theorem minimum_a (x : ℝ) (h : ∀ x ≥ 0, x * Real.exp x + a * Real.exp x * Real.log (x + 1) + 1 ≥ Real.exp x * (x + 1) ^ a) : 
    a ≥ -1 := by
  sorry

end minimum_a_l11_11433


namespace product_is_eight_l11_11152

noncomputable def compute_product (r : ℂ) (hr : r ≠ 1) (hr7 : r^7 = 1) : ℂ :=
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) * (r^6 - 1)

theorem product_is_eight (r : ℂ) (hr : r ≠ 1) (hr7 : r^7 = 1) : compute_product r hr hr7 = 8 :=
by
  sorry

end product_is_eight_l11_11152


namespace sign_of_x_minus_y_l11_11867

theorem sign_of_x_minus_y (x y a : ℝ) (h1 : x + y > 0) (h2 : a < 0) (h3 : a * y > 0) : x - y > 0 := 
by 
  sorry

end sign_of_x_minus_y_l11_11867


namespace sum_of_digits_1197_l11_11617

theorem sum_of_digits_1197 : (1 + 1 + 9 + 7 = 18) := by sorry

end sum_of_digits_1197_l11_11617


namespace three_digit_integers_211_421_l11_11357

def is_one_more_than_multiple_of (n k : ℕ) : Prop :=
  ∃ m : ℕ, n = m * k + 1

theorem three_digit_integers_211_421
  (n : ℕ) (h1 : (100 ≤ n) ∧ (n ≤ 999))
  (h2 : is_one_more_than_multiple_of n 2)
  (h3 : is_one_more_than_multiple_of n 3)
  (h4 : is_one_more_than_multiple_of n 5)
  (h5 : is_one_more_than_multiple_of n 7) :
  n = 211 ∨ n = 421 :=
sorry

end three_digit_integers_211_421_l11_11357


namespace calculate_expression_l11_11062

open Complex

def B : Complex := 5 - 2 * I
def N : Complex := -3 + 2 * I
def T : Complex := 2 * I
def Q : ℂ := 3

theorem calculate_expression : B - N + T - 2 * Q = 2 - 2 * I := by
  sorry

end calculate_expression_l11_11062


namespace highest_score_runs_l11_11677

theorem highest_score_runs 
  (avg : ℕ) (innings : ℕ) (total_runs : ℕ) (H L : ℕ)
  (diff_HL : ℕ) (excl_avg : ℕ) (excl_innings : ℕ) (excl_total_runs : ℕ) :
  avg = 60 → innings = 46 → total_runs = avg * innings →
  diff_HL = 180 → excl_avg = 58 → excl_innings = 44 → 
  excl_total_runs = excl_avg * excl_innings →
  H - L = diff_HL →
  total_runs = excl_total_runs + H + L →
  H = 194 :=
by
  intros h_avg h_innings h_total_runs h_diff_HL h_excl_avg h_excl_innings h_excl_total_runs h_H_minus_L h_total_eq
  sorry

end highest_score_runs_l11_11677


namespace felicity_used_5_gallons_less_l11_11899

def adhesion_gas_problem : Prop :=
  ∃ A x : ℕ, (A + 23 = 30) ∧ (4 * A - x = 23) ∧ (x = 5)
  
theorem felicity_used_5_gallons_less :
  adhesion_gas_problem :=
by
  sorry

end felicity_used_5_gallons_less_l11_11899


namespace odd_positive_int_divisible_by_24_l11_11499

theorem odd_positive_int_divisible_by_24 (n : ℕ) (hn : n % 2 = 1 ∧ n > 0) : 24 ∣ (n ^ n - n) :=
sorry

end odd_positive_int_divisible_by_24_l11_11499


namespace domain_of_c_x_l11_11263

theorem domain_of_c_x (k : ℝ) :
  (∀ x : ℝ, -5 * x ^ 2 + 3 * x + k ≠ 0) ↔ k < -9 / 20 := 
sorry

end domain_of_c_x_l11_11263


namespace k_of_neg7_l11_11493

noncomputable def h (x : ℝ) : ℝ := 4 * x - 9
noncomputable def k (x : ℝ) : ℝ := 3 * x^2 + 4 * x - 2

theorem k_of_neg7 : k (-7) = 3 / 4 :=
by
  sorry

end k_of_neg7_l11_11493


namespace quadratic_greatest_value_and_real_roots_l11_11996

theorem quadratic_greatest_value_and_real_roots :
  (∀ x : ℝ, -x^2 + 9 * x - 20 ≥ 0 → x ≤ 5)
  ∧ (∃ x : ℝ, -x^2 + 9 * x - 20 = 0)
  :=
sorry

end quadratic_greatest_value_and_real_roots_l11_11996


namespace thomas_worked_hours_l11_11315

theorem thomas_worked_hours (Toby Thomas Rebecca : ℕ) 
  (h_total : Thomas + Toby + Rebecca = 157) 
  (h_toby : Toby = 2 * Thomas - 10) 
  (h_rebecca_1 : Rebecca = Toby - 8) 
  (h_rebecca_2 : Rebecca = 56) : Thomas = 37 :=
by
  sorry

end thomas_worked_hours_l11_11315


namespace stephan_cannot_afford_laptop_l11_11042

noncomputable def initial_laptop_price : ℝ := sorry

theorem stephan_cannot_afford_laptop (P₀ : ℝ) (h_rate : 0 < 0.06) (h₁ : initial_laptop_price = P₀) : 
  56358 < P₀ * (1.06)^2 :=
by 
  sorry

end stephan_cannot_afford_laptop_l11_11042


namespace midpoint_square_sum_l11_11300

theorem midpoint_square_sum (x y : ℝ) :
  (4, 1) = ((2 + x) / 2, (6 + y) / 2) → x^2 + y^2 = 52 :=
by
  sorry

end midpoint_square_sum_l11_11300


namespace original_number_of_professors_l11_11232

theorem original_number_of_professors (p : ℕ) 
  (h1 : 6480 % p = 0) 
  (h2 : 11200 % (p + 3) = 0) 
  (h3 : 6480 / p < 11200 / (p + 3))
  (h4 : 5 ≤ p) : 
  p = 5 :=
by {
  -- The body of the proof goes here.
  sorry
}

end original_number_of_professors_l11_11232


namespace area_enclosed_by_line_and_curve_l11_11685

theorem area_enclosed_by_line_and_curve :
  ∃ area, ∀ (x : ℝ), x^2 = 4 * (x - 4/2) → 
    area = ∫ (t : ℝ) in Set.Icc (-1 : ℝ) 2, (1/4 * t + 1/2 - 1/4 * t^2) :=
sorry

end area_enclosed_by_line_and_curve_l11_11685


namespace system_has_infinitely_many_solutions_l11_11509

theorem system_has_infinitely_many_solutions :
  ∃ (S : Set (ℝ × ℝ × ℝ)), (∀ x y z : ℝ, (x + y = 2 ∧ xy - z^2 = 1) ↔ (x, y, z) ∈ S) ∧ S.Infinite :=
by
  sorry

end system_has_infinitely_many_solutions_l11_11509


namespace product_of_consecutive_multiples_of_4_divisible_by_768_l11_11303

theorem product_of_consecutive_multiples_of_4_divisible_by_768 (n : ℤ) :
  (4 * n) * (4 * (n + 1)) * (4 * (n + 2)) % 768 = 0 :=
by
  sorry

end product_of_consecutive_multiples_of_4_divisible_by_768_l11_11303


namespace longest_sequence_positive_integer_x_l11_11526

theorem longest_sequence_positive_integer_x :
  ∃ x : ℤ, 0 < x ∧ 34 * x - 10500 > 0 ∧ 17000 - 55 * x > 0 ∧ x = 309 :=
by
  use 309
  sorry

end longest_sequence_positive_integer_x_l11_11526


namespace triangular_difference_l11_11311

/-- Definition of triangular numbers -/
def triangular (n : ℕ) : ℕ :=
  n * (n + 1) / 2

/-- Main theorem: the difference between the 30th and 29th triangular numbers is 30 -/
theorem triangular_difference : triangular 30 - triangular 29 = 30 :=
by
  sorry

end triangular_difference_l11_11311


namespace three_digit_number_l11_11766

/-- 
Prove there exists three-digit number N such that 
1. N is of form 100a + 10b + c
2. 1 ≤ a ≤ 9
3. 0 ≤ b, c ≤ 9
4. N = 11 * (a + b + c)
--/
theorem three_digit_number (N a b c : ℕ) 
  (hN: N = 100 * a + 10 * b + c) 
  (h_a: 1 ≤ a ∧ a ≤ 9)
  (h_b_c: 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9)
  (h_condition: N = 11 * (a + b + c)) :
  N = 198 := 
sorry

end three_digit_number_l11_11766


namespace is_linear_equation_D_l11_11255

theorem is_linear_equation_D :
  (∀ (x y : ℝ), 2 * x + 3 * y = 7 → false) ∧
  (∀ (x : ℝ), 3 * x ^ 2 = 3 → false) ∧
  (∀ (x : ℝ), 6 = 2 / x - 1 → false) ∧
  (∀ (x : ℝ), 2 * x - 1 = 20 → true) 
:= by {
  sorry
}

end is_linear_equation_D_l11_11255


namespace smallest_even_piece_to_stop_triangle_l11_11488

-- Define a predicate to check if an integer is even
def even (x : ℕ) : Prop := x % 2 = 0

-- Define the conditions for triangle inequality to hold
def triangle_inequality_violated (a b c : ℕ) : Prop :=
  a + b ≤ c ∨ a + c ≤ b ∨ b + c ≤ a

-- Define the main theorem
theorem smallest_even_piece_to_stop_triangle
  (x : ℕ) (hx : even x) (len1 len2 len3 : ℕ)
  (h_len1 : len1 = 7) (h_len2 : len2 = 24) (h_len3 : len3 = 25) :
  6 ≤ x → triangle_inequality_violated (len1 - x) (len2 - x) (len3 - x) :=
by
  sorry

end smallest_even_piece_to_stop_triangle_l11_11488


namespace quadratic_eq_proof_l11_11967

noncomputable def quadratic_eq := ∀ (a b : ℝ), 
  (a ≠ 0 → (∃ (x : ℝ), a * x^2 + b * x + 1/4 = 0) →
    (a = b^2 ∧ a = 1 ∧ b = 1) ∨ (a > 1 ∧ 0 < b ∧ b < 1 → ¬ ∃ (x : ℝ), a * x^2 + b * x + 1/4 = 0))

theorem quadratic_eq_proof : quadratic_eq := 
by
  sorry

end quadratic_eq_proof_l11_11967


namespace cos_double_angle_l11_11530

theorem cos_double_angle (α : ℝ) (h : Real.sin α = Real.sqrt 3 / 3) : 
  Real.cos (2 * α) = 1 / 3 :=
by
  sorry

end cos_double_angle_l11_11530


namespace find_m_value_l11_11645

variable (m : ℝ)
noncomputable def a : ℝ × ℝ := (2 * Real.sqrt 2, 2)
noncomputable def b : ℝ × ℝ := (0, 2)
noncomputable def c (m : ℝ) : ℝ × ℝ := (m, Real.sqrt 2)

theorem find_m_value (h : (a.1 + 2 * b.1) * (m) + (a.2 + 2 * b.2) * (Real.sqrt 2) = 0) : m = -3 :=
by
  sorry

end find_m_value_l11_11645


namespace p_twice_q_in_future_years_l11_11887

-- We define the ages of p and q
def p_current_age : ℕ := 33
def q_current_age : ℕ := 11

-- Third condition that is redundant given the values we already defined
def age_relation : Prop := (p_current_age = 3 * q_current_age)

-- Number of years in the future when p will be twice as old as q
def future_years_when_twice : ℕ := 11

-- Prove that in future_years_when_twice years, p will be twice as old as q
theorem p_twice_q_in_future_years :
  ∀ t : ℕ, t = future_years_when_twice → (p_current_age + t = 2 * (q_current_age + t)) := by
  sorry

end p_twice_q_in_future_years_l11_11887


namespace quadrilateral_with_three_right_angles_is_rectangle_l11_11793

-- Define a quadrilateral with angles
structure Quadrilateral :=
  (a1 a2 a3 a4 : ℝ)
  (sum_angles : a1 + a2 + a3 + a4 = 360)

-- Define a right angle
def is_right_angle (angle : ℝ) : Prop :=
  angle = 90

-- Define a rectangle
def is_rectangle (q : Quadrilateral) : Prop :=
  is_right_angle q.a1 ∧ is_right_angle q.a2 ∧ is_right_angle q.a3 ∧ is_right_angle q.a4

-- The main theorem: if a quadrilateral has three right angles, it is a rectangle
theorem quadrilateral_with_three_right_angles_is_rectangle 
  (q : Quadrilateral) 
  (h1 : is_right_angle q.a1) 
  (h2 : is_right_angle q.a2) 
  (h3 : is_right_angle q.a3) 
  : is_rectangle q :=
sorry

end quadrilateral_with_three_right_angles_is_rectangle_l11_11793


namespace first_term_of_geometric_series_l11_11941

theorem first_term_of_geometric_series (a r : ℝ) 
    (h1 : a / (1 - r) = 18) 
    (h2 : a^2 / (1 - r^2) = 72) : 
    a = 72 / 11 := 
  sorry

end first_term_of_geometric_series_l11_11941


namespace store_paid_price_l11_11215

-- Definition of the conditions
def selling_price : ℕ := 34
def difference_price : ℕ := 8

-- Statement that needs to be proven.
theorem store_paid_price : (selling_price - difference_price) = 26 :=
by
  sorry

end store_paid_price_l11_11215


namespace percentage_of_discount_l11_11132

variable (C : ℝ) -- Cost Price of the Book

-- Conditions
axiom profit_with_discount (C : ℝ) : ∃ S_d : ℝ, S_d = C * 1.235
axiom profit_without_discount (C : ℝ) : ∃ S_nd : ℝ, S_nd = C * 2.30

-- Theorem to prove
theorem percentage_of_discount (C : ℝ) : 
  ∃ discount_percentage : ℝ, discount_percentage = 46.304 := by
  sorry

end percentage_of_discount_l11_11132


namespace blake_initial_money_l11_11397

theorem blake_initial_money (amount_spent_oranges amount_spent_apples amount_spent_mangoes change_received initial_amount : ℕ)
  (h1 : amount_spent_oranges = 40)
  (h2 : amount_spent_apples = 50)
  (h3 : amount_spent_mangoes = 60)
  (h4 : change_received = 150)
  (h5 : initial_amount = (amount_spent_oranges + amount_spent_apples + amount_spent_mangoes) + change_received) :
  initial_amount = 300 :=
by
  sorry

end blake_initial_money_l11_11397


namespace students_with_average_age_of_16_l11_11778

theorem students_with_average_age_of_16
  (N : ℕ) (A : ℕ) (N14 : ℕ) (A15 : ℕ) (N16 : ℕ)
  (h1 : N = 15) (h2 : A = 15) (h3 : N14 = 5) (h4 : A15 = 11) :
  N16 = 9 :=
sorry

end students_with_average_age_of_16_l11_11778


namespace Paige_team_players_l11_11382

/-- Paige's team won their dodgeball game and scored 41 points total.
    If Paige scored 11 points and everyone else scored 6 points each,
    prove that the total number of players on the team was 6. -/
theorem Paige_team_players (total_points paige_points other_points : ℕ) (x : ℕ) (H1 : total_points = 41) (H2 : paige_points = 11) (H3 : other_points = 6) (H4 : paige_points + other_points * x = total_points) : x + 1 = 6 :=
by {
  sorry
}

end Paige_team_players_l11_11382


namespace acute_angles_in_triangle_l11_11083

theorem acute_angles_in_triangle (α β γ : ℝ) (A_ext B_ext C_ext : ℝ) 
  (h_sum : α + β + γ = 180) 
  (h_ext1 : A_ext = 180 - β) 
  (h_ext2 : B_ext = 180 - γ) 
  (h_ext3 : C_ext = 180 - α) 
  (h_ext_acute1 : A_ext < 90 → β > 90) 
  (h_ext_acute2 : B_ext < 90 → γ > 90) 
  (h_ext_acute3 : C_ext < 90 → α > 90) : 
  ((α < 90 ∧ β < 90) ∨ (α < 90 ∧ γ < 90) ∨ (β < 90 ∧ γ < 90)) ∧ 
  ((A_ext < 90 → ¬ (B_ext < 90 ∨ C_ext < 90)) ∧ 
   (B_ext < 90 → ¬ (A_ext < 90 ∨ C_ext < 90)) ∧ 
   (C_ext < 90 → ¬ (A_ext < 90 ∨ B_ext < 90))) :=
sorry

end acute_angles_in_triangle_l11_11083


namespace total_seashells_l11_11157

-- Definitions of the initial number of seashells and the number found
def initial_seashells : Nat := 19
def found_seashells : Nat := 6

-- Theorem stating the total number of seashells in the collection
theorem total_seashells : initial_seashells + found_seashells = 25 := by
  sorry

end total_seashells_l11_11157


namespace rectangle_area_x_l11_11920

theorem rectangle_area_x (x : ℕ) (h1 : x > 0) (h2 : 5 * x = 45) : x = 9 := 
by
  -- proof goes here
  sorry

end rectangle_area_x_l11_11920


namespace interval_length_l11_11372

theorem interval_length (a b : ℝ) (h : ∀ x : ℝ, a + 1 ≤ 3 * x + 6 ∧ 3 * x + 6 ≤ b - 2) :
  (b - a = 57) :=
sorry

end interval_length_l11_11372


namespace distance_between_Petrovo_and_Nikolaevo_l11_11615

theorem distance_between_Petrovo_and_Nikolaevo :
  ∃ S : ℝ, (10 + (S - 10) / 4) + (20 + (S - 20) / 3) = S ∧ S = 50 := by
    sorry

end distance_between_Petrovo_and_Nikolaevo_l11_11615


namespace proportional_distribution_ratio_l11_11771

theorem proportional_distribution_ratio (B : ℝ) (r : ℝ) (S : ℝ) 
  (h1 : B = 80) 
  (h2 : S = 164)
  (h3 : S = (B / (1 - r)) + (B * (1 - r))) : 
  r = 0.2 := 
sorry

end proportional_distribution_ratio_l11_11771


namespace find_x_l11_11161

theorem find_x (x : ℕ) :
  (3 * x > 91 ∧ x < 120 ∧ x < 27 ∧ ¬(4 * x > 37) ∧ ¬(2 * x ≥ 21) ∧ ¬(x > 7)) ∨
  (¬(3 * x > 91) ∧ x < 120 ∧ x < 27 ∧ 4 * x > 37 ∧ ¬(2 * x ≥ 21) ∧ ¬(x > 7)) ∨
  (¬(3 * x > 91) ∧ ¬(x < 120) ∧ x < 27 ∧ 4 * x > 37 ∧ 2 * x ≥ 21 ∧ x > 7) ∨
  (¬(3 * x > 91) ∧ x < 120 ∧ ¬(x < 27) ∧ 4 * x > 37 ∧ 2 * x ≥ 21 ∧ x > 7) ∨
  (¬(3 * x > 91) ∧ x < 120 ∧ x < 27 ∧ ¬(4 * x > 37) ∧ 2 * x ≥ 21 ∧ x > 7) ∨
  (¬(3 * x > 91) ∧ x < 120 ∧ x < 27 ∧ 4 * x > 37 ∧ 2 * x ≥ 21 ∧ ¬(x > 7)) →
  x = 9 :=
sorry

end find_x_l11_11161


namespace rem_fraction_of_66_l11_11319

noncomputable def n : ℝ := 22.142857142857142
noncomputable def s : ℝ := n + 5
noncomputable def p : ℝ := s * 7
noncomputable def q : ℝ := p / 5
noncomputable def r : ℝ := q - 5

theorem rem_fraction_of_66 : r = 33 ∧ r / 66 = 1 / 2 := by 
  sorry

end rem_fraction_of_66_l11_11319


namespace probability_all_three_blue_l11_11995

theorem probability_all_three_blue :
  let total_jellybeans := 20
  let initial_blue := 10
  let initial_red := 10
  let prob_first_blue := initial_blue / total_jellybeans
  let prob_second_blue := (initial_blue - 1) / (total_jellybeans - 1)
  let prob_third_blue := (initial_blue - 2) / (total_jellybeans - 2)
  prob_first_blue * prob_second_blue * prob_third_blue = 2 / 19 := 
by
  sorry

end probability_all_three_blue_l11_11995


namespace fgf_3_equals_108_l11_11515

def f (x : ℕ) : ℕ := 2 * x + 4
def g (x : ℕ) : ℕ := 5 * x + 2

theorem fgf_3_equals_108 : f (g (f 3)) = 108 := 
by
  sorry

end fgf_3_equals_108_l11_11515


namespace lateral_surface_area_of_pyramid_inscribed_in_sphere_l11_11160
-- Importing the entire Mathlib library to ensure all necessary definitions and theorems are available.

-- Formulate the problem as a Lean statement.

theorem lateral_surface_area_of_pyramid_inscribed_in_sphere :
  let R := (1 : ℝ)
  let theta := (45 : ℝ) * Real.pi / 180 -- Convert degrees to radians.
  -- Assuming the pyramid is regular and quadrilateral, inscribed in a sphere of radius 1
  ∃ S : ℝ, S = 4 :=
  sorry

end lateral_surface_area_of_pyramid_inscribed_in_sphere_l11_11160


namespace lottery_probability_prizes_l11_11854

theorem lottery_probability_prizes :
  let total_tickets := 3
  let first_prize_tickets := 1
  let second_prize_tickets := 1
  let non_prize_tickets := 1
  let person_a_wins_first := (2 / 3 : ℝ)
  let person_b_wins_from_remaining := (1 / 2 : ℝ)
  (2 / 3 * 1 / 2) = (1 / 3 : ℝ) := sorry

end lottery_probability_prizes_l11_11854


namespace relationship_between_vars_l11_11567

variable {α : Type*} [LinearOrderedAddCommGroup α]

theorem relationship_between_vars (a b : α) 
  (h1 : a + b < 0) 
  (h2 : b > 0) : a < -b ∧ -b < b ∧ b < -a :=
by
  sorry

end relationship_between_vars_l11_11567


namespace algebraic_expression_value_l11_11287

theorem algebraic_expression_value
  (a : ℝ) 
  (h : a^2 + 2 * a - 1 = 0) : 
  -a^2 - 2 * a + 8 = 7 :=
by 
  sorry

end algebraic_expression_value_l11_11287


namespace alpha_sufficient_not_necessary_l11_11969

def A := {x : ℝ | 2 < x ∧ x < 3}

def B (α : ℝ) := {x : ℝ | (x + 2) * (x - α) < 0}

theorem alpha_sufficient_not_necessary (α : ℝ) : 
  (α = 1 → A ∩ B α = ∅) ∧ (∃ β : ℝ, β ≠ 1 ∧ A ∩ B β = ∅) :=
by
  sorry

end alpha_sufficient_not_necessary_l11_11969


namespace petya_time_spent_l11_11135

theorem petya_time_spent :
  (1 / 3) + (1 / 5) + (1 / 6) + (1 / 70) + (1 / 3) > 1 :=
by
  sorry

end petya_time_spent_l11_11135


namespace probability_at_least_four_girls_l11_11545

noncomputable def binomial_probability (n : ℕ) (k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k) * (p^k) * ((1 - p)^(n - k))

theorem probability_at_least_four_girls
  (n : ℕ)
  (p : ℝ)
  (q : ℝ)
  (h_pq : p + q = 1)
  (h_p : p = 0.55)
  (h_q : q = 0.45)
  (h_n : n = 7) :
  (binomial_probability n 4 p) + (binomial_probability n 5 p) + (binomial_probability n 6 p) + (binomial_probability n 7 p) = 0.59197745 :=
sorry

end probability_at_least_four_girls_l11_11545


namespace find_numerator_l11_11947

variable {y : ℝ} (hy : y > 0) (n : ℝ)

theorem find_numerator (h: (2 * y / 10) + n = 1 / 2 * y) : n = 3 :=
sorry

end find_numerator_l11_11947


namespace annual_profit_growth_rate_l11_11111

variable (a : ℝ)

theorem annual_profit_growth_rate (ha : a > -1) : 
  (1 + a) ^ 12 - 1 = (1 + a) ^ 12 - 1 := 
by 
  sorry

end annual_profit_growth_rate_l11_11111


namespace locus_of_intersection_l11_11176

theorem locus_of_intersection
  (a b : ℝ) (h_a_nonzero : a ≠ 0) (h_b_nonzero : b ≠ 0) (h_neq : a ≠ b) :
  ∃ (x y : ℝ), 
    (∃ c : ℝ, y = (a/c)*x ∧ (x/b + y/c = 1)) 
    ∧ 
    ( (x - b/2)^2 / (b^2/4) + y^2 / (ab/4) = 1 ) :=
sorry

end locus_of_intersection_l11_11176


namespace positive_number_decreased_by_4_is_21_times_reciprocal_l11_11402

theorem positive_number_decreased_by_4_is_21_times_reciprocal (x : ℝ) (h1 : x > 0) (h2 : x - 4 = 21 * (1 / x)) : x = 7 := 
sorry

end positive_number_decreased_by_4_is_21_times_reciprocal_l11_11402


namespace factorize_x_squared_minus_1_l11_11497

theorem factorize_x_squared_minus_1 (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_x_squared_minus_1_l11_11497

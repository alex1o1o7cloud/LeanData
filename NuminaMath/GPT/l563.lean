import Mathlib

namespace eden_stuffed_bears_l563_56371

theorem eden_stuffed_bears 
  (initial_bears : ℕ) 
  (percentage_kept : ℝ) 
  (sisters : ℕ) 
  (eden_initial_bears : ℕ)
  (h1 : initial_bears = 65) 
  (h2 : percentage_kept = 0.40) 
  (h3 : sisters = 4) 
  (h4 : eden_initial_bears = 20) :
  ∃ eden_bears : ℕ, eden_bears = 29 :=
by
  sorry

end eden_stuffed_bears_l563_56371


namespace solve_equation1_solve_equation2_l563_56317

-- Define the equations
def equation1 (x : ℝ) : Prop := x^2 + 2 * x - 4 = 0
def equation2 (x : ℝ) : Prop := 2 * x - 6 = x * (3 - x)

-- State the first proof problem
theorem solve_equation1 (x : ℝ) :
  equation1 x ↔ (x = -1 + Real.sqrt 5 ∨ x = -1 - Real.sqrt 5) := by
  sorry

-- State the second proof problem
theorem solve_equation2 (x : ℝ) :
  equation2 x ↔ (x = 3 ∨ x = -2) := by
  sorry

end solve_equation1_solve_equation2_l563_56317


namespace triangle_angles_correct_l563_56359

open Real

noncomputable def angle_triple (A B C : ℝ) : Prop :=
  ∃ (a b c : ℝ),
    a = 2 * b * cos C ∧ 
    sin A * sin (B / 2 + C) = sin C * (sin (B / 2) + sin A)

theorem triangle_angles_correct (A B C : ℝ) (h : angle_triple A B C) :
  A = 5 * π / 9 ∧ B = 2 * π / 9 ∧ C = 2 * π / 9 := 
sorry

end triangle_angles_correct_l563_56359


namespace calculate_expression_l563_56372

def smallest_positive_two_digit_multiple_of_7 : ℕ := 14
def smallest_positive_three_digit_multiple_of_5 : ℕ := 100

theorem calculate_expression : 
  let c := smallest_positive_two_digit_multiple_of_7
  let d := smallest_positive_three_digit_multiple_of_5
  (c * d) - 100 = 1300 :=
by 
  let c := smallest_positive_two_digit_multiple_of_7
  let d := smallest_positive_three_digit_multiple_of_5
  sorry

end calculate_expression_l563_56372


namespace find_a_l563_56327

noncomputable def point_on_hyperbola (x y : ℝ) : Prop :=
  (x^2 / 4) - y^2 = 1 ∧ x ≥ 2

theorem find_a (a : ℝ) : 
  (∃ x y : ℝ, point_on_hyperbola x y ∧ (min ((x - a)^2 + y^2) = 3)) → 
  (a = -1 ∨ a = 2 * Real.sqrt 5) :=
by
  sorry

end find_a_l563_56327


namespace remainder_is_five_l563_56377

theorem remainder_is_five (A : ℕ) (h : 17 = 6 * 2 + A) : A = 5 :=
sorry

end remainder_is_five_l563_56377


namespace fraction_multiplication_l563_56321

theorem fraction_multiplication :
  (7 / 8) * 48 = 42 := 
sorry

end fraction_multiplication_l563_56321


namespace five_fourths_of_fifteen_fourths_l563_56329

theorem five_fourths_of_fifteen_fourths :
  (5 / 4) * (15 / 4) = 75 / 16 := by
  sorry

end five_fourths_of_fifteen_fourths_l563_56329


namespace option_D_correct_l563_56323

noncomputable def y1 (x : ℝ) : ℝ := 1 / x
noncomputable def y2 (x : ℝ) : ℝ := x^2
noncomputable def y3 (x : ℝ) : ℝ := (1 / 2)^x
noncomputable def y4 (x : ℝ) : ℝ := 1 / x^2

theorem option_D_correct :
  (∀ x : ℝ, y4 x = y4 (-x)) ∧ (∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → y4 x₁ > y4 x₂) :=
by
  sorry

end option_D_correct_l563_56323


namespace unique_solution_inequality_l563_56300

theorem unique_solution_inequality (a : ℝ) : 
  (∀ x : ℝ, -3 ≤ x^2 - 2 * a * x + a ∧ x^2 - 2 * a * x + a ≤ -2 → ∃! x : ℝ, x^2 - 2 * a * x + a = -2) ↔ (a = 2 ∨ a = -1) :=
sorry

end unique_solution_inequality_l563_56300


namespace days_worked_per_week_l563_56313

theorem days_worked_per_week
  (hourly_wage : ℕ) (hours_per_day : ℕ) (total_earnings : ℕ) (weeks : ℕ)
  (H_wage : hourly_wage = 12) (H_hours : hours_per_day = 9) (H_earnings : total_earnings = 3780) (H_weeks : weeks = 7) :
  (total_earnings / weeks) / (hourly_wage * hours_per_day) = 5 :=
by 
  sorry

end days_worked_per_week_l563_56313


namespace difference_of_scores_l563_56344

variable {x y : ℝ}

theorem difference_of_scores (h : x / y = 4) : x - y = 3 * y := by
  sorry

end difference_of_scores_l563_56344


namespace opposite_of_negative_rational_l563_56340

theorem opposite_of_negative_rational : - (-(4/3)) = (4/3) :=
by
  sorry

end opposite_of_negative_rational_l563_56340


namespace vector_subtraction_l563_56338

open Real

def vector_a : (ℝ × ℝ) := (3, 2)
def vector_b : (ℝ × ℝ) := (0, -1)

theorem vector_subtraction : 
  3 • vector_b - vector_a = (-3, -5) :=
by 
  -- Proof needs to be written here.
  sorry

end vector_subtraction_l563_56338


namespace candy_cost_proof_l563_56347

theorem candy_cost_proof (x : ℝ) (h1 : 10 ≤ 30) (h2 : 0 ≤ 5) (h3 : 0 ≤ 6) 
(h4 : 10 * x + 20 * 5 = 6 * 30) : x = 8 := by
  sorry

end candy_cost_proof_l563_56347


namespace intersection_with_complement_l563_56394

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 1, 2, 3}
def B : Set ℕ := {0, 2, 4}

theorem intersection_with_complement (hU : U = {0, 1, 2, 3, 4})
                                     (hA : A = {0, 1, 2, 3})
                                     (hB : B = {0, 2, 4}) :
  A ∩ (U \ B) = {1, 3} :=
by sorry

end intersection_with_complement_l563_56394


namespace emma_reaches_jack_after_33_minutes_l563_56374

-- Definitions from conditions
def distance_initial : ℝ := 30  -- 30 km apart initially
def combined_speed : ℝ := 2     -- combined speed is 2 km/min
def time_before_breakdown : ℝ := 6 -- Jack biked for 6 minutes before breaking down

-- Assume speeds
def v_J (v_E : ℝ) : ℝ := 2 * v_E  -- Jack's speed is twice Emma's speed

-- Assertion to prove
theorem emma_reaches_jack_after_33_minutes :
  ∀ v_E : ℝ, ((v_J v_E + v_E = combined_speed) → 
              (distance_initial - combined_speed * time_before_breakdown = 18) → 
              (v_E > 0) → 
              (time_before_breakdown + 18 / v_E = 33)) :=
by 
  intro v_E 
  intros h1 h2 h3 
  have h4 : v_J v_E = 2 * v_E := rfl
  sorry

end emma_reaches_jack_after_33_minutes_l563_56374


namespace correct_operation_l563_56354

-- Definitions based on conditions
def exprA (a b : ℤ) : ℤ := 3 * a * b - a * b
def exprB (a : ℤ) : ℤ := -3 * a^2 - 5 * a^2
def exprC (x : ℤ) : ℤ := -3 * x - 2 * x

-- Statement to prove that exprB is correct
theorem correct_operation (a : ℤ) : exprB a = -8 * a^2 := by
  sorry

end correct_operation_l563_56354


namespace not_taking_ship_probability_l563_56312

-- Real non-negative numbers as probabilities
variables (P_train P_ship P_car P_airplane : ℝ)

-- Conditions
axiom h_train : 0 ≤ P_train ∧ P_train ≤ 1 ∧ P_train = 0.3
axiom h_ship : 0 ≤ P_ship ∧ P_ship ≤ 1 ∧ P_ship = 0.1
axiom h_car : 0 ≤ P_car ∧ P_car ≤ 1 ∧ P_car = 0.4
axiom h_airplane : 0 ≤ P_airplane ∧ P_airplane ≤ 1 ∧ P_airplane = 0.2

-- Prove that the probability of not taking a ship is 0.9
theorem not_taking_ship_probability : 1 - P_ship = 0.9 :=
by
  sorry

end not_taking_ship_probability_l563_56312


namespace fraction_finding_l563_56389

theorem fraction_finding (x : ℝ) (h : (3 / 4) * x * (2 / 3) = 0.4) : x = 0.8 :=
sorry

end fraction_finding_l563_56389


namespace carlos_cycles_more_than_diana_l563_56361

theorem carlos_cycles_more_than_diana :
  let slope_carlos := 1
  let slope_diana := 0.75
  let rate_carlos := slope_carlos * 20
  let rate_diana := slope_diana * 20
  let distance_carlos_after_3_hours := 3 * rate_carlos
  let distance_diana_after_3_hours := 3 * rate_diana
  distance_carlos_after_3_hours - distance_diana_after_3_hours = 15 :=
sorry

end carlos_cycles_more_than_diana_l563_56361


namespace find_principal_l563_56383

noncomputable def principal_amount (P : ℝ) (r : ℝ) : Prop :=
  (800 = (P * r * 2) / 100) ∧ (820 = P * (1 + r / 100)^2 - P)

theorem find_principal (P : ℝ) (r : ℝ) (h : principal_amount P r) : P = 8000 :=
by
  sorry

end find_principal_l563_56383


namespace exists_large_absolute_value_solutions_l563_56325

theorem exists_large_absolute_value_solutions : 
  ∃ (x1 x2 y1 y2 y3 y4 : ℤ), 
    x1 + x2 = y1 + y2 + y3 + y4 ∧ 
    x1^2 + x2^2 = y1^2 + y2^2 + y3^2 + y4^2 ∧ 
    x1^3 + x2^3 = y1^3 + y2^3 + y3^3 + y4^3 ∧ 
    abs x1 > 2020 ∧ abs x2 > 2020 ∧ abs y1 > 2020 ∧ abs y2 > 2020 ∧ abs y3 > 2020 ∧ abs y4 > 2020 :=
  by
  sorry

end exists_large_absolute_value_solutions_l563_56325


namespace max_distance_from_circle_to_line_l563_56332

theorem max_distance_from_circle_to_line :
  ∀ (P : ℝ × ℝ), (P.1 - 1)^2 + P.2^2 = 9 →
  ∀ (x y : ℝ), 5 * x + 12 * y + 8 = 0 →
  ∃ (d : ℝ), d = 4 :=
by
  -- Proof is omitted as instructed.
  sorry

end max_distance_from_circle_to_line_l563_56332


namespace disease_cases_linear_decrease_l563_56379

theorem disease_cases_linear_decrease (cases_1970 cases_2010 cases_1995 cases_2005 : ℕ)
  (year_1970 year_2010 year_1995 year_2005 : ℕ)
  (h_cases_1970 : cases_1970 = 800000)
  (h_cases_2010 : cases_2010 = 200)
  (h_year_1970 : year_1970 = 1970)
  (h_year_2010 : year_2010 = 2010)
  (h_year_1995 : year_1995 = 1995)
  (h_year_2005 : year_2005 = 2005)
  (linear_decrease : ∀ t, cases_1970 - (cases_1970 - cases_2010) * (t - year_1970) / (year_2010 - year_1970) = cases_1970 - t * (cases_1970 - cases_2010) / (year_2010 - year_1970))
  : cases_1995 = 300125 ∧ cases_2005 = 100175 := sorry

end disease_cases_linear_decrease_l563_56379


namespace length_of_larger_sheet_l563_56302

theorem length_of_larger_sheet : 
  ∃ L : ℝ, 2 * (L * 11) = 2 * (5.5 * 11) + 100 ∧ L = 10 :=
by
  sorry

end length_of_larger_sheet_l563_56302


namespace ab_cd_divisible_eq_one_l563_56352

theorem ab_cd_divisible_eq_one (a b c d : ℕ) (h1 : ∃ e : ℕ, e = ab - cd ∧ (e ∣ a) ∧ (e ∣ b) ∧ (e ∣ c) ∧ (e ∣ d)) : ab - cd = 1 :=
sorry

end ab_cd_divisible_eq_one_l563_56352


namespace degree_of_p_x2_q_x4_l563_56334

-- Definitions to capture the given problem conditions
def is_degree_3 (p : Polynomial ℝ) : Prop := p.degree = 3
def is_degree_6 (q : Polynomial ℝ) : Prop := q.degree = 6

-- Statement of the proof problem
theorem degree_of_p_x2_q_x4 (p q : Polynomial ℝ) (hp : is_degree_3 p) (hq : is_degree_6 q) :
  (p.comp (Polynomial.X ^ 2) * q.comp (Polynomial.X ^ 4)).degree = 30 :=
sorry

end degree_of_p_x2_q_x4_l563_56334


namespace number_of_correct_conclusions_l563_56339

-- Define the conditions given in the problem
def conclusion1 (x : ℝ) : Prop := x > 0 → x > Real.sin x
def conclusion2 (x : ℝ) : Prop := (x - Real.sin x = 0 → x = 0) → (x ≠ 0 → x - Real.sin x ≠ 0)
def conclusion3 (p q : Prop) : Prop := (p ∧ q → p ∨ q) ∧ ¬(p ∨ q → p ∧ q)
def conclusion4 : Prop := ¬(∀ x : ℝ, x - Real.log x > 0) = ∃ x : ℝ, x - Real.log x ≤ 0

-- Prove the number of correct conclusions is 3
theorem number_of_correct_conclusions : 
  (∃ x1 : ℝ, conclusion1 x1) ∧
  (∃ x1 : ℝ, conclusion2 x1) ∧
  (∃ p q : Prop, conclusion3 p q) ∧
  ¬conclusion4 →
  3 = 3 :=
by
  intros
  sorry

end number_of_correct_conclusions_l563_56339


namespace tetrahedron_sum_eq_14_l563_56358

theorem tetrahedron_sum_eq_14 :
  let edges := 6
  let corners := 4
  let faces := 4
  edges + corners + faces = 14 :=
by
  let edges := 6
  let corners := 4
  let faces := 4
  show edges + corners + faces = 14
  sorry

end tetrahedron_sum_eq_14_l563_56358


namespace percentage_markup_l563_56369

theorem percentage_markup 
  (selling_price : ℝ) 
  (cost_price : ℝ) 
  (h1 : selling_price = 8215)
  (h2 : cost_price = 6625)
  : ((selling_price - cost_price) / cost_price) * 100 = 24 := 
  by
    sorry

end percentage_markup_l563_56369


namespace combined_resistance_l563_56301

theorem combined_resistance (x y r : ℝ) (hx : x = 5) (hy : y = 7) (h_parallel : 1 / r = 1 / x + 1 / y) : 
  r = 35 / 12 := 
by 
  sorry

end combined_resistance_l563_56301


namespace rate_of_descent_correct_l563_56397

def depth := 3500 -- in feet
def time := 100 -- in minutes

def rate_of_descent : ℕ := depth / time

theorem rate_of_descent_correct : rate_of_descent = 35 := by
  -- We intentionally skip the proof part as per the requirement
  sorry

end rate_of_descent_correct_l563_56397


namespace frustum_volume_correct_l563_56318

-- Define the base edge of the original pyramid
def base_edge_pyramid := 16

-- Define the height (altitude) of the original pyramid
def height_pyramid := 10

-- Define the base edge of the smaller pyramid after the cut
def base_edge_smaller_pyramid := 8

-- Define the function to calculate the volume of a square pyramid
def volume_square_pyramid (base_edge : ℕ) (height : ℕ) : ℚ :=
  (1 / 3) * (base_edge ^ 2) * height

-- Calculate the volume of the original pyramid
def V := volume_square_pyramid base_edge_pyramid height_pyramid

-- Calculate the volume of the smaller pyramid
def V_small := volume_square_pyramid base_edge_smaller_pyramid (height_pyramid / 2)

-- Calculate the volume of the frustum
def V_frustum := V - V_small

-- Prove that the volume of the frustum is 213.33 cubic centimeters
theorem frustum_volume_correct : V_frustum = 213.33 := by
  sorry

end frustum_volume_correct_l563_56318


namespace diagonal_of_rectangular_prism_l563_56324

theorem diagonal_of_rectangular_prism
  (width height depth : ℕ)
  (h1 : width = 15)
  (h2 : height = 20)
  (h3 : depth = 25) : 
  (width ^ 2 + height ^ 2 + depth ^ 2).sqrt = 25 * (2 : ℕ).sqrt :=
by {
  sorry
}

end diagonal_of_rectangular_prism_l563_56324


namespace initial_average_mark_l563_56330

-- Define the conditions
def total_students := 13
def average_mark := 72
def excluded_students := 5
def excluded_students_average := 40
def remaining_students := total_students - excluded_students
def remaining_students_average := 92

-- Define the total marks calculations
def initial_total_marks (A : ℕ) : ℕ := total_students * A
def excluded_total_marks : ℕ := excluded_students * excluded_students_average
def remaining_total_marks : ℕ := remaining_students * remaining_students_average

-- Prove the initial average mark
theorem initial_average_mark : 
  initial_total_marks average_mark = excluded_total_marks + remaining_total_marks →
  average_mark = 72 :=
by
  sorry

end initial_average_mark_l563_56330


namespace find_b_skew_lines_l563_56346

def line1 (b : ℝ) (t : ℝ) : ℝ × ℝ × ℝ :=
  (2 + 3*t, 3 + 4*t, b + 5*t)

def line2 (u : ℝ) : ℝ × ℝ × ℝ :=
  (5 + 6*u, 6 + 3*u, 1 + 2*u)

noncomputable def lines_are_skew (b : ℝ) : Prop :=
  ∀ t u : ℝ, line1 b t ≠ line2 u

theorem find_b_skew_lines (b : ℝ) : b ≠ -12 / 5 → lines_are_skew b :=
by
  sorry

end find_b_skew_lines_l563_56346


namespace average_age_of_team_l563_56351

theorem average_age_of_team 
  (n : ℕ) 
  (captain_age : ℕ) 
  (wicket_keeper_age : ℕ) 
  (remaining_avg : ℕ → ℕ) 
  (h1 : n = 11)
  (h2 : captain_age = 27)
  (h3 : wicket_keeper_age = 28)
  (h4 : ∀ A, remaining_avg A = A - 1)
  (h5 : ∀ A, 11 * A = 9 * (remaining_avg A) + captain_age + wicket_keeper_age) : 
  ∃ A, A = 32 :=
by
  sorry

end average_age_of_team_l563_56351


namespace Sandy_tokens_more_than_siblings_l563_56391

theorem Sandy_tokens_more_than_siblings :
  let total_tokens := 1000000
  let siblings := 4
  let Sandy_tokens := total_tokens / 2
  let sibling_tokens := (total_tokens - Sandy_tokens) / siblings
  Sandy_tokens - sibling_tokens = 375000 :=
by
  -- Definitions as per conditions
  let total_tokens := 1000000
  let siblings := 4
  let Sandy_tokens := total_tokens / 2
  let sibling_tokens := (total_tokens - Sandy_tokens) / siblings
  -- Conclusion
  show Sandy_tokens - sibling_tokens = 375000
  sorry

end Sandy_tokens_more_than_siblings_l563_56391


namespace sandy_total_puppies_l563_56303

-- Definitions based on conditions:
def original_puppies : ℝ := 8.0
def additional_puppies : ℝ := 4.0

-- Theorem statement: total_puppies should be 12.0
theorem sandy_total_puppies : original_puppies + additional_puppies = 12.0 := 
by
  sorry

end sandy_total_puppies_l563_56303


namespace children_got_off_bus_l563_56387

theorem children_got_off_bus :
  ∀ (initial_children final_children new_children off_children : ℕ),
    initial_children = 21 → final_children = 16 → new_children = 5 →
    initial_children - off_children + new_children = final_children →
    off_children = 10 :=
by
  intro initial_children final_children new_children off_children
  intros h_init h_final h_new h_eq
  sorry

end children_got_off_bus_l563_56387


namespace deg_to_rad_neg_630_l563_56305

theorem deg_to_rad_neg_630 :
  (-630 : ℝ) * (Real.pi / 180) = -7 * Real.pi / 2 := by
  sorry

end deg_to_rad_neg_630_l563_56305


namespace analogical_reasoning_correctness_l563_56310

theorem analogical_reasoning_correctness 
  (a b c : ℝ)
  (va vb vc : ℝ) :
  (a + b) * c = (a * c + b * c) ↔ 
  (va + vb) * vc = (va * vc + vb * vc) := 
sorry

end analogical_reasoning_correctness_l563_56310


namespace find_x_l563_56388

theorem find_x (x : ℝ) :
  let P1 := (2, 10)
  let P2 := (6, 2)
  
  -- Slope of the line joining (2, 10) and (6, 2)
  let slope12 := (P2.2 - P1.2) / (P2.1 - P1.1)
  
  -- Slope of the line joining (2, 10) and (x, -3)
  let P3 := (x, -3)
  let slope13 := (P3.2 - P1.2) / (P3.1 - P1.1)
  
  -- Condition that both slopes are equal
  slope12 = slope13
  
  -- To Prove: x must be 8.5
  → x = 8.5 :=
sorry

end find_x_l563_56388


namespace double_meat_sandwich_bread_count_l563_56343

theorem double_meat_sandwich_bread_count (x : ℕ) :
  14 * 2 + 12 * x = 64 → x = 3 := by
  intro h
  sorry

end double_meat_sandwich_bread_count_l563_56343


namespace Ingrid_cookie_percentage_l563_56398

theorem Ingrid_cookie_percentage : 
  let irin_ratio := 9.18
  let ingrid_ratio := 5.17
  let nell_ratio := 2.05
  let kim_ratio := 3.45
  let linda_ratio := 4.56
  let total_cookies := 800
  let total_ratio := irin_ratio + ingrid_ratio + nell_ratio + kim_ratio + linda_ratio
  let ingrid_share := ingrid_ratio / total_ratio
  let ingrid_cookies := ingrid_share * total_cookies
  let ingrid_percentage := (ingrid_cookies / total_cookies) * 100
  ingrid_percentage = 21.25 :=
by
  sorry

end Ingrid_cookie_percentage_l563_56398


namespace carbon_copies_after_folding_l563_56364

def initial_sheets : ℕ := 6
def initial_carbons (sheets : ℕ) : ℕ := sheets - 1
def final_copies (sheets : ℕ) : ℕ := sheets - 1

theorem carbon_copies_after_folding :
  (final_copies initial_sheets) =
  initial_carbons initial_sheets :=
by {
    -- sorry is a placeholder for the proof
    sorry
}

end carbon_copies_after_folding_l563_56364


namespace product_modulo_7_l563_56370

theorem product_modulo_7 : 
  (2007 % 7 = 4) ∧ (2008 % 7 = 5) ∧ (2009 % 7 = 6) ∧ (2010 % 7 = 0) →
  (2007 * 2008 * 2009 * 2010) % 7 = 0 :=
by
  intros h
  rcases h with ⟨h1, h2, h3, h4⟩
  sorry

end product_modulo_7_l563_56370


namespace max_value_of_f_l563_56335

noncomputable def f (x : ℝ) : ℝ :=
  2022 * x ^ 2 * Real.log (x + 2022) / ((Real.log (x + 2022)) ^ 3 + 2 * x ^ 3)

theorem max_value_of_f : ∃ x : ℝ, 0 < x ∧ f x ≤ 674 :=
by
  sorry

end max_value_of_f_l563_56335


namespace powerThreeExpression_l563_56314

theorem powerThreeExpression (x : ℝ) (h : x - (1 / x) = 5) : x^3 - (1 / (x^3)) = 140 := sorry

end powerThreeExpression_l563_56314


namespace sum_of_consecutive_integers_l563_56308

theorem sum_of_consecutive_integers (n : ℕ) (h : n * (n + 1) / 2 ≤ 500) :
  n = 31 ∨ ((32 * (32 + 1) / 2) ≤ 500) := 
sorry

end sum_of_consecutive_integers_l563_56308


namespace negation_proposition_l563_56355

variable (a : ℝ)

theorem negation_proposition :
  (¬ ∃ x : ℝ, x^2 + 2 * a * x + a ≤ 0) ↔ (∀ x : ℝ, x^2 + 2 * a * x + a > 0) :=
by
  sorry

end negation_proposition_l563_56355


namespace percent_decrease_apr_to_may_l563_56350

theorem percent_decrease_apr_to_may (P : ℝ) 
  (h1 : ∀ P : ℝ, P > 0 → (1.35 * P = P + 0.35 * P))
  (h2 : ∀ x : ℝ, P * (1.35 * (1 - x / 100) * 1.5) = 1.62000000000000014 * P)
  (h3 : 0 < x ∧ x < 100)
  : x = 20 :=
  sorry

end percent_decrease_apr_to_may_l563_56350


namespace rashmi_late_time_is_10_l563_56337

open Real

noncomputable def rashmi_late_time : ℝ :=
  let d : ℝ := 9.999999999999993
  let v1 : ℝ := 5 / 60 -- km per minute
  let v2 : ℝ := 6 / 60 -- km per minute
  let time1 := d / v1 -- time taken at 5 kmph
  let time2 := d / v2 -- time taken at 6 kmph
  let difference := time1 - time2
  let T := difference / 2 -- The time she was late or early
  T

theorem rashmi_late_time_is_10 : rashmi_late_time = 10 := by
  simp [rashmi_late_time]
  sorry

end rashmi_late_time_is_10_l563_56337


namespace multiple_of_remainder_l563_56368

theorem multiple_of_remainder (R V D Q k : ℤ) (h1 : R = 6) (h2 : V = 86) (h3 : D = 5 * Q) 
  (h4 : D = k * R + 2) (h5 : V = D * Q + R) : k = 3 := by
  sorry

end multiple_of_remainder_l563_56368


namespace inequality_for_positive_reals_l563_56320

theorem inequality_for_positive_reals 
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (1 / (a^3 + b^3 + a * b * c)) + (1 / (b^3 + c^3 + a * b * c)) + 
  (1 / (c^3 + a^3 + a * b * c)) ≤ 1 / (a * b * c) := 
sorry

end inequality_for_positive_reals_l563_56320


namespace percentage_of_bottle_danny_drank_l563_56386

theorem percentage_of_bottle_danny_drank
    (x : ℝ)  -- percentage of the first bottle Danny drinks, represented as a real number
    (b1 b2 b3 : ℝ)  -- volumes of the three bottles, represented as real numbers
    (h_b1 : b1 = 1)  -- first bottle is full (1 bottle)
    (h_b2 : b2 = 1)  -- second bottle is full (1 bottle)
    (h_b3 : b3 = 1)  -- third bottle is full (1 bottle)
    (h_given_away1 : b2 * 0.7 = 0.7)  -- gave away 70% of the second bottle
    (h_given_away2 : b3 * 0.7 = 0.7)  -- gave away 70% of the third bottle
    (h_soda_left : b1 * (1 - x) + b2 * 0.3 + b3 * 0.3 = 0.7)  -- 70% of bottle left
    : x = 0.9 :=
by
  sorry

end percentage_of_bottle_danny_drank_l563_56386


namespace percentage_increase_twice_l563_56304

theorem percentage_increase_twice (P : ℝ) (x : ℝ) :
  P * (1 + x)^2 = P * 1.3225 → x = 0.15 :=
by
  intro h
  have h1 : (1 + x)^2 = 1.3225 := by sorry
  have h2 : x^2 + 2 * x = 0.3225 := by sorry
  have h3 : x = (-2 + Real.sqrt 5.29) / 2 := by sorry
  have h4 : x = -2 / 2 + Real.sqrt 5.29 / 2 := by sorry
  have h5 : x = 0.15 := by sorry
  exact h5

end percentage_increase_twice_l563_56304


namespace average_infection_per_round_l563_56384

theorem average_infection_per_round (x : ℝ) (h1 : 1 + x + x * (1 + x) = 100) : x = 9 :=
sorry

end average_infection_per_round_l563_56384


namespace speedster_convertibles_l563_56319

noncomputable def total_inventory (not_speedsters : Nat) (fraction_not_speedsters : ℝ) : ℝ :=
  (not_speedsters : ℝ) / fraction_not_speedsters

noncomputable def number_speedsters (total_inventory : ℝ) (fraction_speedsters : ℝ) : ℝ :=
  total_inventory * fraction_speedsters

noncomputable def number_convertibles (number_speedsters : ℝ) (fraction_convertibles : ℝ) : ℝ :=
  number_speedsters * fraction_convertibles

theorem speedster_convertibles : (not_speedsters = 30) ∧ (fraction_not_speedsters = 2 / 3) ∧ (fraction_speedsters = 1 / 3) ∧ (fraction_convertibles = 4 / 5) →
  number_convertibles (number_speedsters (total_inventory not_speedsters fraction_not_speedsters) fraction_speedsters) fraction_convertibles = 12 :=
by
  intros h
  sorry

end speedster_convertibles_l563_56319


namespace S_5_equals_31_l563_56390

-- Define the sequence sum function S
def S (n : Nat) : Nat := 2^n - 1

-- The theorem to prove that S(5) = 31
theorem S_5_equals_31 : S 5 = 31 :=
by
  rw [S]
  sorry

end S_5_equals_31_l563_56390


namespace factor_b_value_l563_56356

theorem factor_b_value (a b : ℤ) (h : ∀ x : ℂ, (x^2 - x - 1) ∣ (a*x^3 + b*x^2 + 1)) : b = -2 := 
sorry

end factor_b_value_l563_56356


namespace polynomial_evaluation_l563_56399

def polynomial_at (x : ℝ) : ℝ :=
  let f := (7 : ℝ) * x^5 + 12 * x^4 - 5 * x^3 - 6 * x^2 + 3 * x - 5
  f

theorem polynomial_evaluation : polynomial_at 3 = 2488 :=
by
  sorry

end polynomial_evaluation_l563_56399


namespace students_taking_neither_l563_56309

theorem students_taking_neither (total students_cs students_electronics students_both : ℕ)
  (h1 : total = 60) (h2 : students_cs = 42) (h3 : students_electronics = 35) (h4 : students_both = 25) :
  total - (students_cs - students_both + students_electronics - students_both + students_both) = 8 :=
by {
  sorry
}

end students_taking_neither_l563_56309


namespace b_alone_days_l563_56342

theorem b_alone_days {a b : ℝ} (h1 : a + b = 1/6) (h2 : a = 1/11) : b = 1/(66/5) :=
by sorry

end b_alone_days_l563_56342


namespace probability_of_X_l563_56381

variable (P : Prop → ℝ)
variable (event_X event_Y : Prop)

-- Defining the conditions
variable (hYP : P event_Y = 2 / 3)
variable (hXYP : P (event_X ∧ event_Y) = 0.13333333333333333)

-- Proving that the probability of selection of X is 0.2
theorem probability_of_X : P event_X = 0.2 := by
  sorry

end probability_of_X_l563_56381


namespace convert_octal_127_to_binary_l563_56366

def octal_to_binary (n : ℕ) : ℕ :=
  match n with
  | 1 => 3  -- 001 in binary
  | 2 => 2  -- 010 in binary
  | 7 => 7  -- 111 in binary
  | _ => 0  -- No other digits are used in this example

theorem convert_octal_127_to_binary :
  octal_to_binary 1 * 1000000 + octal_to_binary 2 * 1000 + octal_to_binary 7 = 1010111 :=
by
  -- Proof would go here
  sorry

end convert_octal_127_to_binary_l563_56366


namespace range_of_a_l563_56326

-- Define the inequality problem
def inequality_always_true (a : ℝ) : Prop :=
  ∀ x, a * x^2 + 3 * a * x + a - 2 < 0

-- Define the range condition for "a"
def range_condition (a : ℝ) : Prop :=
  (a = 0 ∧ (-2 < 0)) ∨
  (a ≠ 0 ∧ a < 0 ∧ a * (5 * a + 8) < 0)

-- The main theorem stating the equivalence
theorem range_of_a (a : ℝ) : inequality_always_true a ↔ a ∈ Set.Icc (- (8 / 5)) 0 := by
  sorry

end range_of_a_l563_56326


namespace cannot_achieve_61_cents_with_six_coins_l563_56396

theorem cannot_achieve_61_cents_with_six_coins :
  ¬ ∃ (p n d q : ℕ), 
      p + n + d + q = 6 ∧ 
      p + 5 * n + 10 * d + 25 * q = 61 :=
by
  sorry

end cannot_achieve_61_cents_with_six_coins_l563_56396


namespace dhoni_remaining_earnings_l563_56395

theorem dhoni_remaining_earnings :
  let rent := 0.20
  let dishwasher := 0.15
  let bills := 0.10
  let car := 0.08
  let grocery := 0.12
  let tax := 0.05
  let expenses := rent + dishwasher + bills + car + grocery + tax
  let remaining_after_expenses := 1.0 - expenses
  let savings := 0.40 * remaining_after_expenses
  let remaining_after_savings := remaining_after_expenses - savings
  remaining_after_savings = 0.18 := by
sorry

end dhoni_remaining_earnings_l563_56395


namespace profit_per_box_type_A_and_B_maximize_profit_l563_56307

-- Condition definitions
def total_boxes : ℕ := 600
def profit_type_A : ℕ := 40000
def profit_type_B : ℕ := 160000
def profit_difference : ℕ := 200

-- Question 1: Proving the profit per box for type A and B
theorem profit_per_box_type_A_and_B (x : ℝ) :
  (profit_type_A / x + profit_type_B / (x + profit_difference) = total_boxes)
  → (x = 200) ∧ (x + profit_difference = 400) :=
sorry

-- Condition definitions for question 2
def price_reduction_per_box_A (a : ℕ) : ℕ := 5 * a
def price_increase_per_box_B (a : ℕ) : ℕ := 5 * a

-- Initial number of boxes sold for type A and B
def initial_boxes_sold_A : ℕ := 200
def initial_boxes_sold_B : ℕ := 400

-- General profit function
def profit (a : ℕ) : ℝ :=
  (initial_boxes_sold_A + 2 * a) * (200 - price_reduction_per_box_A a) +
  (initial_boxes_sold_B - 2 * a) * (400 + price_increase_per_box_B a)

-- Question 2: Proving the price reduction and maximum profit
theorem maximize_profit (a : ℕ) :
  ((price_reduction_per_box_A a = 75) ∧ (profit a = 204500)) :=
sorry

end profit_per_box_type_A_and_B_maximize_profit_l563_56307


namespace problem_l563_56345

noncomputable def f (x a : ℝ) : ℝ := (x - a) * Real.exp x

theorem problem (a : ℝ) (x : ℝ) (hx : x ∈ Set.Ici (-5)) (ha : a = 1) : 
  f x a + x + 5 ≥ -6 / Real.exp 5 := 
sorry

end problem_l563_56345


namespace eastville_to_westpath_travel_time_l563_56373

theorem eastville_to_westpath_travel_time :
  ∀ (d t₁ t₂ : ℝ) (s₁ s₂ : ℝ), 
  t₁ = 6 → s₁ = 80 → s₂ = 50 → d = s₁ * t₁ → t₂ = d / s₂ → t₂ = 9.6 := 
by
  intros d t₁ t₂ s₁ s₂ ht₁ hs₁ hs₂ hd ht₂
  sorry

end eastville_to_westpath_travel_time_l563_56373


namespace algebra_1_algebra_2_l563_56393

variable (x1 x2 : ℝ)
variable (h_root1 : x1^2 - 2*x1 - 1 = 0)
variable (h_root2 : x2^2 - 2*x2 - 1 = 0)
variable (h_sum : x1 + x2 = 2)
variable (h_prod : x1 * x2 = -1)

theorem algebra_1 : (x1 + x2) * (x1 * x2) = -2 := by
  -- Proof here
  sorry

theorem algebra_2 : (x1 - x2)^2 = 8 := by
  -- Proof here
  sorry

end algebra_1_algebra_2_l563_56393


namespace restaurant_total_cost_l563_56349

def total_cost
  (adults kids : ℕ)
  (adult_meal_cost adult_drink_cost adult_dessert_cost kid_drink_cost kid_dessert_cost : ℝ) : ℝ :=
  let num_adults := adults
  let num_kids := kids
  let adult_total := num_adults * (adult_meal_cost + adult_drink_cost + adult_dessert_cost)
  let kid_total := num_kids * (kid_drink_cost + kid_dessert_cost)
  adult_total + kid_total

theorem restaurant_total_cost :
  total_cost 4 9 7 4 3 2 1.5 = 87.5 :=
by
  sorry

end restaurant_total_cost_l563_56349


namespace ducks_and_geese_meeting_l563_56360

theorem ducks_and_geese_meeting:
  ∀ x : ℕ, ( ∀ ducks_speed : ℚ, ducks_speed = (1/7) ) → 
         ( ∀ geese_speed : ℚ, geese_speed = (1/9) ) → 
         (ducks_speed * x + geese_speed * x = 1) :=
by
  sorry

end ducks_and_geese_meeting_l563_56360


namespace gym_membership_count_l563_56378

theorem gym_membership_count :
  let charge_per_time := 18
  let times_per_month := 2
  let total_monthly_income := 10800
  let amount_per_member := charge_per_time * times_per_month
  let number_of_members := total_monthly_income / amount_per_member
  number_of_members = 300 :=
by
  let charge_per_time := 18
  let times_per_month := 2
  let total_monthly_income := 10800
  let amount_per_member := charge_per_time * times_per_month
  let number_of_members := total_monthly_income / amount_per_member
  sorry

end gym_membership_count_l563_56378


namespace find_a_l563_56328

theorem find_a (a x : ℝ) (h1 : 2 * (x - 1) - 6 = 0) (h2 : 1 - (3 * a - x) / 3 = 0) (h3 : x = 4) : a = -1 / 3 :=
by
  sorry

end find_a_l563_56328


namespace simplify_T_l563_56311

variable (x : ℝ)

theorem simplify_T :
  9 * (x + 2)^2 - 12 * (x + 2) + 4 = 4 * (1.5 * x + 2)^2 :=
by
  sorry

end simplify_T_l563_56311


namespace rihanna_money_left_l563_56353

-- Definitions of the item costs
def cost_of_mangoes : ℝ := 6 * 3
def cost_of_apple_juice : ℝ := 4 * 3.50
def cost_of_potato_chips : ℝ := 2 * 2.25
def cost_of_chocolate_bars : ℝ := 3 * 1.75

-- Total cost computation
def total_cost : ℝ := cost_of_mangoes + cost_of_apple_juice + cost_of_potato_chips + cost_of_chocolate_bars

-- Initial amount of money Rihanna has
def initial_money : ℝ := 50

-- Remaining money after the purchases
def remaining_money : ℝ := initial_money - total_cost

-- The theorem stating that the remaining money is $8.25
theorem rihanna_money_left : remaining_money = 8.25 := by
  -- Lean will require the proof here.
  sorry

end rihanna_money_left_l563_56353


namespace complex_z_pow_l563_56316

open Complex

theorem complex_z_pow {z : ℂ} (h : (1 + z) / (1 - z) = (⟨0, 1⟩ : ℂ)) : z ^ 2019 = -⟨0, 1⟩ := by
  sorry

end complex_z_pow_l563_56316


namespace polynomial_multiplication_equiv_l563_56362

theorem polynomial_multiplication_equiv (x : ℝ) : 
  (x^4 + 50*x^2 + 625)*(x^2 - 25) = x^6 - 75*x^4 + 1875*x^2 - 15625 := 
by 
  sorry

end polynomial_multiplication_equiv_l563_56362


namespace ball_hits_ground_at_t_l563_56365

theorem ball_hits_ground_at_t (t : ℝ) : 
  (∃ t, -8 * t^2 - 12 * t + 64 = 0 ∧ 0 ≤ t) → t = 2 :=
by
  sorry

end ball_hits_ground_at_t_l563_56365


namespace jessica_attended_games_l563_56376

/-- 
Let total_games be the total number of soccer games.
Let initially_planned be the number of games Jessica initially planned to attend.
Let commitments_skipped be the number of games skipped due to other commitments.
Let rescheduled_games be the rescheduled games during the season.
Let additional_missed be the additional games missed due to rescheduling.
-/
theorem jessica_attended_games
    (total_games initially_planned commitments_skipped rescheduled_games additional_missed : ℕ)
    (h1 : total_games = 12)
    (h2 : initially_planned = 8)
    (h3 : commitments_skipped = 3)
    (h4 : rescheduled_games = 2)
    (h5 : additional_missed = 4) :
    (initially_planned - commitments_skipped) - additional_missed = 1 := by
  sorry

end jessica_attended_games_l563_56376


namespace final_price_is_correct_l563_56392

-- Define the original price
def original_price : ℝ := 10

-- Define the first reduction percentage
def first_reduction_percentage : ℝ := 0.30

-- Define the second reduction percentage
def second_reduction_percentage : ℝ := 0.50

-- Define the price after the first reduction
def price_after_first_reduction : ℝ := original_price * (1 - first_reduction_percentage)

-- Define the final price after the second reduction
def final_price : ℝ := price_after_first_reduction * (1 - second_reduction_percentage)

-- Theorem to prove the final price is $3.50
theorem final_price_is_correct : final_price = 3.50 := by
  sorry

end final_price_is_correct_l563_56392


namespace point_D_coordinates_l563_56385

theorem point_D_coordinates 
  (F : (ℕ × ℕ)) 
  (coords_F : F = (5,5)) 
  (D : (ℕ × ℕ)) 
  (coords_D : D = (2,4)) :
  (D = (2,4)) :=
by 
  sorry

end point_D_coordinates_l563_56385


namespace pencil_distribution_l563_56375

theorem pencil_distribution (n : ℕ) (friends : ℕ): 
  (friends = 4) → (n = 8) → 
  (∃ A B C D : ℕ, A ≥ 2 ∧ B ≥ 1 ∧ C ≥ 1 ∧ D ≥ 1 ∧ A + B + C + D = n) →
  (∃! k : ℕ, k = 20) :=
by
  intros friends_eq n_eq h
  use 20
  sorry

end pencil_distribution_l563_56375


namespace kareem_largest_l563_56382

def jose_final : ℕ :=
  let start := 15
  let minus_two := start - 2
  let triple := minus_two * 3
  triple + 5

def thuy_final : ℕ :=
  let start := 15
  let triple := start * 3
  let minus_two := triple - 2
  minus_two + 5

def kareem_final : ℕ :=
  let start := 15
  let minus_two := start - 2
  let add_five := minus_two + 5
  add_five * 3

theorem kareem_largest : kareem_final > jose_final ∧ kareem_final > thuy_final := by
  sorry

end kareem_largest_l563_56382


namespace factor_polynomial_l563_56336

theorem factor_polynomial (a b c : ℝ) : 
  a^3 * (b^2 - c^2) + b^3 * (c^2 - b^2) + c^3 * (a^2 - b^2) = (a - b) * (b - c) * (c - a) * (a * b + a * c + b * c) :=
by 
  sorry

end factor_polynomial_l563_56336


namespace smallest_a_l563_56348

theorem smallest_a (x a : ℝ) (hx : x > 0) (ha : a > 0) (hineq : x + a / x ≥ 4) : a ≥ 4 :=
sorry

end smallest_a_l563_56348


namespace decrease_percent_revenue_l563_56306

theorem decrease_percent_revenue (T C : ℝ) (hT : T > 0) (hC : C > 0) :
  let original_revenue := T * C
  let new_tax := 0.80 * T
  let new_consumption := 1.05 * C
  let new_revenue := new_tax * new_consumption
  let decrease_in_revenue := original_revenue - new_revenue
  let decrease_percent := (decrease_in_revenue / original_revenue) * 100
  decrease_percent = 16 := 
by
  sorry

end decrease_percent_revenue_l563_56306


namespace total_games_attended_l563_56315

def games_in_months (this_month previous_month next_month following_month fifth_month : ℕ) : ℕ :=
  this_month + previous_month + next_month + following_month + fifth_month

theorem total_games_attended :
  games_in_months 24 32 29 19 34 = 138 :=
by
  -- Proof will be provided, but ignored for this problem
  sorry

end total_games_attended_l563_56315


namespace plane_difference_correct_l563_56380

noncomputable def max_planes : ℕ := 27
noncomputable def min_planes : ℕ := 7
noncomputable def diff_planes : ℕ := max_planes - min_planes

theorem plane_difference_correct : diff_planes = 20 := by
  sorry

end plane_difference_correct_l563_56380


namespace jake_fewer_peaches_undetermined_l563_56322

theorem jake_fewer_peaches_undetermined 
    (steven_peaches : ℕ) 
    (steven_apples : ℕ) 
    (jake_fewer_peaches : steven_peaches > jake_peaches) 
    (jake_more_apples : jake_apples = steven_apples + 3) 
    (steven_peaches_val : steven_peaches = 9) 
    (steven_apples_val : steven_apples = 8) : 
    ∃ n : ℕ, jake_peaches = n ∧ ¬(∃ m : ℕ, steven_peaches - jake_peaches = m) := 
sorry

end jake_fewer_peaches_undetermined_l563_56322


namespace equation_one_solution_equation_two_solution_l563_56367

variables (x : ℝ)

theorem equation_one_solution (h : 2 * (x + 3) = 5 * x) : x = 2 :=
sorry

theorem equation_two_solution (h : (x - 3) / 0.5 - (x + 4) / 0.2 = 1.6) : x = -9.2 :=
sorry

end equation_one_solution_equation_two_solution_l563_56367


namespace min_value_expression_l563_56341

theorem min_value_expression (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + 2 * b = 1) : 
  ∃ (c : ℝ), c = 2 * Real.sqrt 10 + 6 ∧ (∀ x y : ℝ, (x > 0) → (y > 0) → (a + 2 * y = 1) → ( (y^2 + a + 1) / (a * y)  ≥  c )) :=
sorry

end min_value_expression_l563_56341


namespace fence_length_l563_56333

theorem fence_length {w l : ℕ} (h1 : l = 2 * w) (h2 : 30 = 2 * l + 2 * w) : l = 10 := by
  sorry

end fence_length_l563_56333


namespace find_smallest_integer_l563_56363

/-- There exists an integer n such that:
   n ≡ 1 [MOD 3],
   n ≡ 2 [MOD 4],
   n ≡ 3 [MOD 5],
   and the smallest such n is 58. -/
theorem find_smallest_integer :
  ∃ n : ℕ, n % 3 = 1 ∧ n % 4 = 2 ∧ n % 5 = 3 ∧ n = 58 :=
by
  -- Proof goes here (not provided as per the instructions)
  sorry

end find_smallest_integer_l563_56363


namespace twenty_four_point_solution_l563_56331

theorem twenty_four_point_solution : (5 - (1 / 5)) * 5 = 24 := 
by 
  sorry

end twenty_four_point_solution_l563_56331


namespace sum_and_product_of_radical_l563_56357

theorem sum_and_product_of_radical (a b : ℝ) (h1 : 2 * a = -4) (h2 : a^2 - b = 1) :
  a + b = 1 :=
sorry

end sum_and_product_of_radical_l563_56357

import Mathlib

namespace carnival_tickets_l26_2643

theorem carnival_tickets (total_tickets friends : ℕ) (equal_share : ℕ)
  (h1 : friends = 6)
  (h2 : total_tickets = 234)
  (h3 : total_tickets % friends = 0)
  (h4 : equal_share = total_tickets / friends) : 
  equal_share = 39 := 
by
  sorry

end carnival_tickets_l26_2643


namespace find_larger_number_l26_2662

-- Definitions based on the conditions
def larger_number (L S : ℕ) : Prop :=
  L - S = 1365 ∧ L = 6 * S + 20

-- The theorem to prove
theorem find_larger_number (L S : ℕ) (h : larger_number L S) : L = 1634 :=
by
  sorry  -- Proof would go here

end find_larger_number_l26_2662


namespace graph_is_hyperbola_l26_2659

def graph_equation (x y : ℝ) : Prop := x^2 - 16 * y^2 - 8 * x + 64 = 0

theorem graph_is_hyperbola : ∃ (a b : ℝ), ∀ x y : ℝ, graph_equation x y ↔ (x - a)^2 / 48 - y^2 / 3 = -1 :=
by
  sorry

end graph_is_hyperbola_l26_2659


namespace remainder_of_8_pow_6_plus_1_mod_7_l26_2624

theorem remainder_of_8_pow_6_plus_1_mod_7 :
  (8^6 + 1) % 7 = 2 := by
  sorry

end remainder_of_8_pow_6_plus_1_mod_7_l26_2624


namespace more_balloons_allan_l26_2618

theorem more_balloons_allan (allan_balloons : ℕ) (jake_initial_balloons : ℕ) (jake_bought_balloons : ℕ) 
  (h1 : allan_balloons = 6) (h2 : jake_initial_balloons = 2) (h3 : jake_bought_balloons = 3) :
  allan_balloons = jake_initial_balloons + jake_bought_balloons + 1 := 
by 
  -- Assuming Jake's total balloons after purchase
  let jake_total_balloons := jake_initial_balloons + jake_bought_balloons
  -- The proof would involve showing that Allan's balloons are one more than Jake's total balloons
  sorry

end more_balloons_allan_l26_2618


namespace oranges_left_to_be_sold_l26_2648

-- Defining the initial conditions
def seven_dozen_oranges : ℕ := 7 * 12
def reserved_for_friend (total : ℕ) : ℕ := total / 4
def remaining_after_reserve (total reserved : ℕ) : ℕ := total - reserved
def sold_yesterday (remaining : ℕ) : ℕ := 3 * remaining / 7
def remaining_after_sale (remaining sold : ℕ) : ℕ := remaining - sold
def remaining_after_rotten (remaining : ℕ) : ℕ := remaining - 4

-- Statement to prove
theorem oranges_left_to_be_sold (total reserved remaining sold final : ℕ) :
  total = seven_dozen_oranges →
  reserved = reserved_for_friend total →
  remaining = remaining_after_reserve total reserved →
  sold = sold_yesterday remaining →
  final = remaining_after_sale remaining sold - 4 →
  final = 32 :=
by
  sorry

end oranges_left_to_be_sold_l26_2648


namespace smallest_possible_sum_l26_2637

theorem smallest_possible_sum (A B C D : ℤ) 
  (h1 : A + B = 2 * C)
  (h2 : B * D = C * C)
  (h3 : 3 * C = 7 * B)
  (h4 : 0 < A ∧ 0 < B ∧ 0 < C ∧ 0 < D) : 
  A + B + C + D = 76 :=
sorry

end smallest_possible_sum_l26_2637


namespace cheapest_salon_option_haily_l26_2611

theorem cheapest_salon_option_haily : 
  let gustran_haircut := 45
  let gustran_facial := 22
  let gustran_nails := 30
  let gustran_foot_spa := 15
  let gustran_massage := 50
  let gustran_total := gustran_haircut + gustran_facial + gustran_nails + gustran_foot_spa + gustran_massage
  let gustran_discount := 0.20
  let gustran_final := gustran_total * (1 - gustran_discount)

  let barbara_nails := 40
  let barbara_haircut := 30
  let barbara_facial := 28
  let barbara_foot_spa := 18
  let barbara_massage := 45
  let barbara_total :=
      barbara_nails + barbara_haircut + (barbara_facial * 0.5) + barbara_foot_spa + (barbara_massage * 0.5)

  let fancy_haircut := 34
  let fancy_facial := 30
  let fancy_nails := 20
  let fancy_foot_spa := 25
  let fancy_massage := 60
  let fancy_total := fancy_haircut + fancy_facial + fancy_nails + fancy_foot_spa + fancy_massage
  let fancy_discount := 15
  let fancy_final := fancy_total - fancy_discount

  let avg_haircut := (gustran_haircut + barbara_haircut + fancy_haircut) / 3
  let avg_facial := (gustran_facial + barbara_facial + fancy_facial) / 3
  let avg_nails := (gustran_nails + barbara_nails + fancy_nails) / 3
  let avg_foot_spa := (gustran_foot_spa + barbara_foot_spa + fancy_foot_spa) / 3
  let avg_massage := (gustran_massage + barbara_massage + fancy_massage) / 3

  let luxury_haircut := avg_haircut * 1.10
  let luxury_facial := avg_facial * 1.10
  let luxury_nails := avg_nails * 1.10
  let luxury_foot_spa := avg_foot_spa * 1.10
  let luxury_massage := avg_massage * 1.10
  let luxury_total := luxury_haircut + luxury_facial + luxury_nails + luxury_foot_spa + luxury_massage
  let luxury_discount := 20
  let luxury_final := luxury_total - luxury_discount

  gustran_final > barbara_total ∧ barbara_total < fancy_final ∧ barbara_total < luxury_final := 
by 
  sorry

end cheapest_salon_option_haily_l26_2611


namespace number_is_correct_l26_2661

theorem number_is_correct : (1 / 8) + 0.675 = 0.800 := 
by
  sorry

end number_is_correct_l26_2661


namespace f_pos_for_all_x_g_le_ax_plus_1_for_a_eq_1_l26_2642

noncomputable def f (x : ℝ) : ℝ := Real.exp x - (x + 1)^2 / 2
noncomputable def g (x : ℝ) : ℝ := 2 * Real.log (x + 1) + Real.exp (-x)

theorem f_pos_for_all_x (x : ℝ) (hx : x > -1) : f x > 0 := by
  sorry

theorem g_le_ax_plus_1_for_a_eq_1 (a : ℝ) (ha : a > 0) : (∀ x : ℝ, -1 < x → g x ≤ a * x + 1) ↔ a = 1 := by
  sorry

end f_pos_for_all_x_g_le_ax_plus_1_for_a_eq_1_l26_2642


namespace largest_four_digit_number_divisible_by_six_l26_2676

theorem largest_four_digit_number_divisible_by_six : 
  ∃ n : ℕ, (1000 ≤ n ∧ n ≤ 9999) ∧ (n % 2 = 0) ∧ (n % 3 = 0) ∧ 
  (∀ m : ℕ, (1000 ≤ m ∧ m ≤ 9999) ∧ (m % 2 = 0) ∧ (m % 3 = 0) → m ≤ n) ∧ n = 9960 := 
by { sorry }

end largest_four_digit_number_divisible_by_six_l26_2676


namespace restoration_of_axes_l26_2602

theorem restoration_of_axes (parabola : ℝ → ℝ) (h : ∀ x, parabola x = x^2) : 
  ∃ (origin : ℝ × ℝ) (x_axis y_axis : ℝ × ℝ → Prop), 
    (∀ x, x_axis (x, 0)) ∧ 
    (∀ y, y_axis (0, y)) ∧ 
    origin = (0, 0) := 
sorry

end restoration_of_axes_l26_2602


namespace teacher_engineer_ratio_l26_2622

-- Define the context with the given conditions
variable (t e : ℕ)

-- Conditions
def avg_age (t e : ℕ) : Prop := (40 * t + 55 * e) / (t + e) = 45

-- The statement to be proved
theorem teacher_engineer_ratio
  (h : avg_age t e) :
  t / e = 2 := sorry

end teacher_engineer_ratio_l26_2622


namespace valid_k_l26_2681

theorem valid_k (k : ℕ) (h_pos : k ≥ 1) (h : 10^k - 1 = 9 * k^2) : k = 1 := by
  sorry

end valid_k_l26_2681


namespace greatest_whole_number_difference_l26_2632

theorem greatest_whole_number_difference (x y : ℤ) (hx1 : 7 < x) (hx2 : x < 9) (hy1 : 9 < y) (hy2 : y < 15) : y - x = 6 :=
by
  sorry

end greatest_whole_number_difference_l26_2632


namespace length_of_rod_l26_2670

theorem length_of_rod (w1 w2 l1 l2 : ℝ) (h_uniform : ∀ m n, m * w1 = n * w2) (h1 : w1 = 42.75) (h2 : l1 = 11.25) : 
  l2 = 6 := 
  by
  have wpm := w1 / l1
  have h3 : 22.8 / wpm = l2 := by sorry
  rw [h1, h2] at *
  simp at *
  sorry

end length_of_rod_l26_2670


namespace Victor_can_carry_7_trays_at_a_time_l26_2668

-- Define the conditions
def trays_from_first_table : Nat := 23
def trays_from_second_table : Nat := 5
def number_of_trips : Nat := 4

-- Define the total number of trays
def total_trays : Nat := trays_from_first_table + trays_from_second_table

-- Prove that the number of trays Victor can carry at a time is 7
theorem Victor_can_carry_7_trays_at_a_time :
  total_trays / number_of_trips = 7 :=
by
  sorry

end Victor_can_carry_7_trays_at_a_time_l26_2668


namespace S_rational_iff_divides_l26_2650

-- Definition of "divides" for positive integers
def divides (m k : ℕ) : Prop := ∃ j : ℕ, k = m * j

-- Definition of the series S(m, k)
noncomputable def S (m k : ℕ) : ℝ := 
  ∑' n, 1 / (n * (m * n + k))

-- Proof statement
theorem S_rational_iff_divides (m k : ℕ) (hm : 0 < m) (hk : 0 < k) : 
  (∃ r : ℚ, S m k = r) ↔ divides m k :=
sorry

end S_rational_iff_divides_l26_2650


namespace cost_per_minute_l26_2698

theorem cost_per_minute (monthly_fee cost total_bill : ℝ) (minutes : ℕ) :
  monthly_fee = 2 ∧ total_bill = 23.36 ∧ minutes = 178 → 
  cost = (total_bill - monthly_fee) / minutes → 
  cost = 0.12 :=
by
  intros h1 h2
  sorry

end cost_per_minute_l26_2698


namespace boys_in_class_l26_2625

theorem boys_in_class (g b : ℕ) 
  (h_ratio : 4 * g = 3 * b) (h_total : g + b = 28) : b = 16 :=
by
  sorry

end boys_in_class_l26_2625


namespace proof_supplies_proof_transportation_cost_proof_min_cost_condition_l26_2640

open Real

noncomputable def supplies_needed (a b : ℕ) := a = 200 ∧ b = 300

noncomputable def transportation_cost (x : ℝ) := 60 ≤ x ∧ x ≤ 260 ∧ ∀ w : ℝ, w = 10 * x + 10200

noncomputable def min_cost_condition (m x : ℝ) := 
  (0 < m ∧ m ≤ 8) ∧ (∀ w : ℝ, (10 - m) * x + 10200 ≥ 10320)

theorem proof_supplies : ∃ a b : ℕ, supplies_needed a b := 
by
  use 200, 300
  sorry

theorem proof_transportation_cost : ∃ x : ℝ, transportation_cost x := 
by
  use 60
  sorry

theorem proof_min_cost_condition : ∃ m x : ℝ, min_cost_condition m x := 
by
  use 8, 60
  sorry

end proof_supplies_proof_transportation_cost_proof_min_cost_condition_l26_2640


namespace sphere_radius_eq_three_of_volume_eq_surface_area_l26_2621

theorem sphere_radius_eq_three_of_volume_eq_surface_area
  (r : ℝ) 
  (h1 : (4 / 3) * Real.pi * r^3 = 4 * Real.pi * r^2) : 
  r = 3 :=
sorry

end sphere_radius_eq_three_of_volume_eq_surface_area_l26_2621


namespace john_total_skateboarded_distance_l26_2620

noncomputable def total_skateboarded_distance (to_park: ℕ) (back_home: ℕ) : ℕ :=
  to_park + back_home

theorem john_total_skateboarded_distance :
  total_skateboarded_distance 10 10 = 20 :=
by
  sorry

end john_total_skateboarded_distance_l26_2620


namespace similar_iff_condition_l26_2617

-- Define the similarity of triangles and the necessary conditions.
variables {α : Type*} [LinearOrderedField α]
variables (a b c a' b' c' : α)

-- Statement of the problem in Lean 4
theorem similar_iff_condition : 
  (∃ z w : α, a' = a * z + w ∧ b' = b * z + w ∧ c' = c * z + w) ↔ 
  (a' * (b - c) + b' * (c - a) + c' * (a - b) = 0) :=
sorry

end similar_iff_condition_l26_2617


namespace find_annual_interest_rate_l26_2647

theorem find_annual_interest_rate (P0 P1 P2 : ℝ) (r1 r : ℝ) :
  P0 = 12000 →
  r1 = 10 →
  P1 = P0 * (1 + (r1 / 100) / 2) →
  P1 = 12600 →
  P2 = 13260 →
  P1 * (1 + (r / 200)) = P2 →
  r = 10.476 :=
by
  intros hP0 hr1 hP1 hP1val hP2 hP1P2
  sorry

end find_annual_interest_rate_l26_2647


namespace percentage_reduction_in_price_l26_2633

theorem percentage_reduction_in_price (P R : ℝ) (hR : R = 2.953846153846154)
  (h_condition : ∃ P, 65 / 12 * R = 40 - 24 / P) :
  ((P - R) / P) * 100 = 33.3 := by
  sorry

end percentage_reduction_in_price_l26_2633


namespace triangle_perimeter_l26_2613

variable (y : ℝ)

theorem triangle_perimeter (h₁ : 2 * y > y) (h₂ : y > 0) :
  ∃ (P : ℝ), P = 2 * y + y * Real.sqrt 2 :=
sorry

end triangle_perimeter_l26_2613


namespace green_light_probability_l26_2683

def red_duration : ℕ := 30
def green_duration : ℕ := 25
def yellow_duration : ℕ := 5

def total_cycle : ℕ := red_duration + green_duration + yellow_duration
def green_probability : ℚ := green_duration / total_cycle

theorem green_light_probability :
  green_probability = 5 / 12 := by
  sorry

end green_light_probability_l26_2683


namespace initial_eggs_ben_l26_2658

-- Let's define the conditions from step a):
def eggs_morning := 4
def eggs_afternoon := 3
def eggs_left := 13

-- Define the total eggs Ben ate
def eggs_eaten := eggs_morning + eggs_afternoon

-- Now we define the initial eggs Ben had
def initial_eggs := eggs_left + eggs_eaten

-- The theorem that states the initial number of eggs
theorem initial_eggs_ben : initial_eggs = 20 :=
  by sorry

end initial_eggs_ben_l26_2658


namespace questions_for_second_project_l26_2629

open Nat

theorem questions_for_second_project (days_per_week : ℕ) (first_project_q : ℕ) (questions_per_day : ℕ) 
  (total_questions : ℕ) (second_project_q : ℕ) 
  (h1 : days_per_week = 7)
  (h2 : first_project_q = 518)
  (h3 : questions_per_day = 142)
  (h4 : total_questions = days_per_week * questions_per_day)
  (h5 : second_project_q = total_questions - first_project_q) :
  second_project_q = 476 :=
by
  -- we assume the solution steps as correct
  sorry

end questions_for_second_project_l26_2629


namespace fraction_of_draws_is_two_ninths_l26_2675

-- Define the fraction of games that Ben wins and Tom wins
def BenWins : ℚ := 4 / 9
def TomWins : ℚ := 1 / 3

-- Definition of the fraction of games ending in a draw
def fraction_of_draws (BenWins TomWins : ℚ) : ℚ :=
  1 - (BenWins + TomWins)

-- The theorem to be proved
theorem fraction_of_draws_is_two_ninths : fraction_of_draws BenWins TomWins = 2 / 9 :=
by
  sorry

end fraction_of_draws_is_two_ninths_l26_2675


namespace termite_ridden_not_collapsing_l26_2697

theorem termite_ridden_not_collapsing
  (total_homes : ℕ)
  (termite_ridden_fraction : ℚ)
  (collapsing_fraction_of_termite_ridden : ℚ)
  (h1 : termite_ridden_fraction = 1/3)
  (h2 : collapsing_fraction_of_termite_ridden = 1/4) :
  (termite_ridden_fraction - (termite_ridden_fraction * collapsing_fraction_of_termite_ridden)) = 1/4 := 
by {
  sorry
}

end termite_ridden_not_collapsing_l26_2697


namespace find_range_of_m_l26_2606

def has_two_distinct_negative_real_roots (m : ℝ) : Prop := 
  let Δ := m^2 - 4
  Δ > 0 ∧ -m > 0

def inequality_holds_for_all_real (m : ℝ) : Prop :=
  let Δ := (4 * (m - 2))^2 - 16
  Δ < 0

def problem_statement (m : ℝ) : Prop :=
  (has_two_distinct_negative_real_roots m ∨ inequality_holds_for_all_real m) ∧ 
  ¬(has_two_distinct_negative_real_roots m ∧ inequality_holds_for_all_real m)

theorem find_range_of_m (m : ℝ) : problem_statement m ↔ ((1 < m ∧ m ≤ 2) ∨ (3 ≤ m)) :=
by
  sorry

end find_range_of_m_l26_2606


namespace strictly_decreasing_interval_l26_2687

noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 - Real.log x

theorem strictly_decreasing_interval :
  ∀ x y : ℝ, (0 < x ∧ x < 1) ∧ (0 < y ∧ y < 1) ∧ y < x → f y < f x :=
by
  sorry

end strictly_decreasing_interval_l26_2687


namespace expression_value_l26_2649

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f_add_prop (a b : ℝ) : f (a + b) = f a * f b
axiom f_one_val : f 1 = 2

theorem expression_value : 
  (f 1 ^ 2 + f 2) / f 1 + 
  (f 2 ^ 2 + f 4) / f 3 +
  (f 3 ^ 2 + f 6) / f 5 + 
  (f 4 ^ 2 + f 8) / f 7 
  = 16 := 
sorry

end expression_value_l26_2649


namespace find_f_at_75_l26_2673

variables (f : ℝ → ℝ) (h₀ : ∀ x, f (x + 2) = -f x)
variables (h₁ : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = x)
variables (h₂ : ∀ x, f (-x) = -f x)

theorem find_f_at_75 : f 7.5 = -0.5 := by
  sorry

end find_f_at_75_l26_2673


namespace credit_extended_by_automobile_finance_companies_l26_2692

def percentage_of_automobile_installment_credit : ℝ := 0.36
def total_consumer_installment_credit : ℝ := 416.66667
def fraction_extended_by_finance_companies : ℝ := 0.5

theorem credit_extended_by_automobile_finance_companies :
  fraction_extended_by_finance_companies * (percentage_of_automobile_installment_credit * total_consumer_installment_credit) = 75 :=
by
  sorry

end credit_extended_by_automobile_finance_companies_l26_2692


namespace sin_ratio_in_triangle_l26_2636

theorem sin_ratio_in_triangle
  {A B C : ℝ} {a b c : ℝ}
  (h : (b + c) / (c + a) = 4 / 5 ∧ (c + a) / (a + b) = 5 / 6) :
  (Real.sin A + Real.sin C) / Real.sin B = 2 :=
sorry

end sin_ratio_in_triangle_l26_2636


namespace four_times_angle_triangle_l26_2641

theorem four_times_angle_triangle (A B C : ℕ) 
  (h1 : A + B + C = 180) 
  (h2 : A = 40)
  (h3 : (A = 4 * C) ∨ (B = 4 * C) ∨ (C = 4 * A)) : 
  (B = 130 ∧ C = 10) ∨ (B = 112 ∧ C = 28) :=
by
  sorry

end four_times_angle_triangle_l26_2641


namespace solution_set_x2_f_x_positive_l26_2689

noncomputable def f : ℝ → ℝ := sorry
axiom f_odd : ∀ x, f (-x) = -f x
axiom f_at_2 : f 2 = 0
axiom derivative_condition : ∀ x, x > 0 → ((x * (deriv f x) - f x) / x^2) > 0

theorem solution_set_x2_f_x_positive :
  {x : ℝ | x^2 * f x > 0} = {x : ℝ | -2 < x ∧ x < 0} ∪ {x : ℝ | x > 2} :=
sorry

end solution_set_x2_f_x_positive_l26_2689


namespace functional_equation_solution_l26_2688

theorem functional_equation_solution (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (x + f (x + y)) + f (x * y) = x + f (x + y) + y * f x) :
  (∀ x : ℝ, f x = x) ∨ (∀ x : ℝ, f x = 2 - x) :=
sorry

end functional_equation_solution_l26_2688


namespace sally_fries_count_l26_2630

theorem sally_fries_count (sally_initial_fries mark_initial_fries : ℕ) 
  (mark_gave_fraction : ℤ) 
  (h_sally_initial : sally_initial_fries = 14) 
  (h_mark_initial : mark_initial_fries = 36) 
  (h_mark_give : mark_gave_fraction = 1 / 3) :
  sally_initial_fries + (mark_initial_fries * mark_gave_fraction).natAbs = 26 :=
by
  sorry

end sally_fries_count_l26_2630


namespace gwen_math_problems_l26_2601

-- Problem statement
theorem gwen_math_problems (m : ℕ) (science_problems : ℕ := 11) (problems_finished_at_school : ℕ := 24) (problems_left_for_homework : ℕ := 5) 
  (h1 : m + science_problems = problems_finished_at_school + problems_left_for_homework) : m = 18 := 
by {
  sorry
}

end gwen_math_problems_l26_2601


namespace swimmer_path_min_time_l26_2666

theorem swimmer_path_min_time (k : ℝ) :
  (k > Real.sqrt 2 → ∀ x y : ℝ, x = 0 ∧ y = 0 ∧ t = 2/k) ∧
  (k < Real.sqrt 2 → x = 1 ∧ y = 1 ∧ t = Real.sqrt 2) ∧
  (k = Real.sqrt 2 → ∀ x y : ℝ, x = y ∧ t = (1 / Real.sqrt 2) + Real.sqrt 2 + (1 / Real.sqrt 2)) :=
by sorry

end swimmer_path_min_time_l26_2666


namespace triangle_right_angled_solve_system_quadratic_roots_real_l26_2627

-- Problem 1
theorem triangle_right_angled (a b c : ℝ) (h : a^2 + b^2 + c^2 - 6 * a - 8 * b - 10 * c + 50 = 0) :
  (a = 3) ∧ (b = 4) ∧ (c = 5) ∧ (a^2 + b^2 = c^2) :=
sorry

-- Problem 2
theorem solve_system (x y : ℝ) (h1 : 3 * x + 4 * y = 30) (h2 : 5 * x + 3 * y = 28) :
  (x = 2) ∧ (y = 6) :=
sorry

-- Problem 3
theorem quadratic_roots_real (m : ℝ) :
  (∃ x : ℝ, ∃ y : ℝ, 3 * x^2 + 4 * x + m = 0 ∧ 3 * y^2 + 4 * y + m = 0) ↔ (m ≤ 4 / 3) :=
sorry

end triangle_right_angled_solve_system_quadratic_roots_real_l26_2627


namespace total_chestnuts_weight_l26_2623

def eunsoo_kg := 2
def eunsoo_g := 600
def mingi_g := 3700

theorem total_chestnuts_weight :
  (eunsoo_kg * 1000 + eunsoo_g + mingi_g) = 6300 :=
by
  sorry

end total_chestnuts_weight_l26_2623


namespace jenny_house_value_l26_2656

/-- Jenny's property tax rate is 2% -/
def property_tax_rate : ℝ := 0.02

/-- Her house's value increases by 25% due to the new high-speed rail project -/
noncomputable def house_value_increase_rate : ℝ := 0.25

/-- Jenny can afford to spend $15,000/year on property tax -/
def max_affordable_tax : ℝ := 15000

/-- Jenny can make improvements worth $250,000 to her house -/
def improvement_value : ℝ := 250000

/-- Current worth of Jenny's house -/
noncomputable def current_house_worth : ℝ := 500000

theorem jenny_house_value :
  property_tax_rate * (current_house_worth + improvement_value) = max_affordable_tax :=
by
  sorry

end jenny_house_value_l26_2656


namespace fifth_number_in_21st_row_l26_2603

theorem fifth_number_in_21st_row : 
  let nth_odd_number (n : ℕ) := 2 * n - 1 
  let sum_first_n_rows (n : ℕ) := n * (n + (n - 1))
  nth_odd_number 405 = 809 := 
by
  sorry

end fifth_number_in_21st_row_l26_2603


namespace shaded_area_z_shape_l26_2631

theorem shaded_area_z_shape (L W s1 s2 : ℕ) (hL : L = 6) (hW : W = 4) (hs1 : s1 = 2) (hs2 : s2 = 1) :
  (L * W - (s1 * s1 + s2 * s2)) = 19 := by
  sorry

end shaded_area_z_shape_l26_2631


namespace border_area_correct_l26_2667

-- Definition of the dimensions of the photograph
def photo_height := 8
def photo_width := 10
def frame_border := 3

-- Definition of the areas of the photograph and the framed area
def photo_area := photo_height * photo_width
def frame_height := photo_height + 2 * frame_border
def frame_width := photo_width + 2 * frame_border
def frame_area := frame_height * frame_width

-- Theorem stating that the area of the border is 144 square inches
theorem border_area_correct : (frame_area - photo_area) = 144 := 
by
  sorry

end border_area_correct_l26_2667


namespace line_through_diameter_l26_2645

theorem line_through_diameter (P : ℝ × ℝ) (hP : P = (2, 1)) (h_circle : ∀ x y : ℝ, (x - 1)^2 + y^2 = 4) :
  ∃ a b c : ℝ, a * P.1 + b * P.2 + c = 0 ∧ a = 1 ∧ b = -1 ∧ c = -1 :=
by
  exists 1, -1, -1
  sorry

end line_through_diameter_l26_2645


namespace units_digit_same_units_and_tens_digit_same_l26_2682

theorem units_digit_same (n : ℕ) : 
  (∃ a : ℕ, a ∈ [0, 1, 5, 6] ∧ n % 10 = a ∧ n^2 % 10 = a) := 
sorry

theorem units_and_tens_digit_same (n : ℕ) : 
  n ∈ [0, 1, 25, 76] ↔ (n % 100 = n^2 % 100) := 
sorry

end units_digit_same_units_and_tens_digit_same_l26_2682


namespace find_x7_l26_2600

-- Definitions for the conditions
def seq (x : ℕ → ℕ) : Prop :=
  (x 6 = 144) ∧ ∀ n, (n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4) → (x (n + 3) = x (n + 2) * (x (n + 1) + x n))

-- Theorem statement to prove x_7 = 3456
theorem find_x7 (x : ℕ → ℕ) (h : seq x) : x 7 = 3456 := sorry

end find_x7_l26_2600


namespace pet_shop_ways_l26_2651

theorem pet_shop_ways (puppies : ℕ) (kittens : ℕ) (turtles : ℕ)
  (h_puppies : puppies = 10) (h_kittens : kittens = 8) (h_turtles : turtles = 5) : 
  (puppies * kittens * turtles = 400) :=
by
  sorry

end pet_shop_ways_l26_2651


namespace positive_solution_for_y_l26_2669

theorem positive_solution_for_y (x y z : ℝ) 
  (h1 : x * y = 4 - x - 2 * y)
  (h2 : y * z = 8 - 3 * y - 2 * z)
  (h3 : x * z = 40 - 5 * x - 2 * z) : y = 2 := 
sorry

end positive_solution_for_y_l26_2669


namespace quadratic_factorization_value_of_a_l26_2652

theorem quadratic_factorization_value_of_a (a : ℝ) :
  (∀ x : ℝ, 2 * x^2 - 8 * x + a = 0 ↔ 2 * (x - 2)^2 = 4) → a = 4 :=
by
  intro h
  sorry

end quadratic_factorization_value_of_a_l26_2652


namespace initial_concentration_is_40_l26_2626

noncomputable def initial_concentration_fraction : ℝ := 1 / 3
noncomputable def replaced_solution_concentration : ℝ := 25
noncomputable def resulting_concentration : ℝ := 35
noncomputable def initial_concentration := 40

theorem initial_concentration_is_40 (C : ℝ) (h1 : C = (3 / 2) * (resulting_concentration - (initial_concentration_fraction * replaced_solution_concentration))) :
  C = initial_concentration :=
by sorry

end initial_concentration_is_40_l26_2626


namespace find_k_l26_2616

theorem find_k (a b c k : ℤ)
  (g : ℤ → ℤ)
  (h1 : ∀ x, g x = a * x^2 + b * x + c)
  (h2 : g 2 = 0)
  (h3 : 60 < g 6 ∧ g 6 < 70)
  (h4 : 90 < g 9 ∧ g 9 < 100)
  (h5 : 10000 * k < g 50 ∧ g 50 < 10000 * (k + 1)) :
  k = 0 :=
sorry

end find_k_l26_2616


namespace chocolates_problem_l26_2690

theorem chocolates_problem 
  (N : ℕ)
  (h1 : ∃ R C : ℕ, N = R * C)
  (h2 : ∃ r1, r1 = 3 * C - 1)
  (h3 : ∃ c1, c1 = R * 5 - 1)
  (h4 : ∃ r2 c2 : ℕ, r2 = 3 ∧ c2 = 5 ∧ r2 * c2 = r1 - 1)
  (h5 : N = 3 * (3 * C - 1))
  (h6 : N / 3 = (12 * 5) / 3) : 
  N = 60 ∧ (N - (3 * C - 1)) = 25 :=
by 
  sorry

end chocolates_problem_l26_2690


namespace evaluate_expression_l26_2628

theorem evaluate_expression : 5 - 7 * (8 - 3^2) * 4 = 33 :=
by
  sorry

end evaluate_expression_l26_2628


namespace arithmetic_sequence_transformation_l26_2612

theorem arithmetic_sequence_transformation (a : ℕ → ℝ) (d c : ℝ) (h : ∀ n, a (n + 1) = a n + d) (hc : c ≠ 0) :
  ∀ n, (c * a (n + 1)) - (c * a n) = c * d := 
by
  sorry

end arithmetic_sequence_transformation_l26_2612


namespace sum_of_missing_digits_l26_2654

-- Define the problem's conditions
def add_digits (a b c d e f g h : ℕ) := 
a + b = 18 ∧ b + c + d = 21

-- Prove the sum of the missing digits equals 7
theorem sum_of_missing_digits (a b c d e f g h : ℕ) (h1 : add_digits a b c d e f g h) : a + c = 7 := 
sorry

end sum_of_missing_digits_l26_2654


namespace tank_filling_time_l26_2665

noncomputable def fill_time (R1 R2 R3 : ℚ) : ℚ :=
  1 / (R1 + R2 + R3)

theorem tank_filling_time :
  let R1 := 1 / 18
  let R2 := 1 / 30
  let R3 := -1 / 45
  fill_time R1 R2 R3 = 15 :=
by
  intros
  unfold fill_time
  sorry

end tank_filling_time_l26_2665


namespace initial_birds_count_l26_2639

theorem initial_birds_count (B : ℕ) (h1 : 6 = B + 3 + 1) : B = 2 :=
by
  -- Placeholder for the proof, we are not required to provide it here.
  sorry

end initial_birds_count_l26_2639


namespace binomial_133_133_l26_2614

theorem binomial_133_133 : @Nat.choose 133 133 = 1 := by   
sorry

end binomial_133_133_l26_2614


namespace cubic_root_expression_l26_2663

theorem cubic_root_expression (u v w : ℂ) (huvwx : u * v * w ≠ 0)
  (h1 : u^3 - 6 * u^2 + 11 * u - 6 = 0)
  (h2 : v^3 - 6 * v^2 + 11 * v - 6 = 0)
  (h3 : w^3 - 6 * w^2 + 11 * w - 6 = 0) :
  (u * v / w) + (v * w / u) + (w * u / v) = 49 / 6 :=
sorry

end cubic_root_expression_l26_2663


namespace farmer_plough_rate_l26_2699

-- Define the problem statement and the required proof 

theorem farmer_plough_rate :
  ∀ (x y : ℕ),
  90 * x = 3780 ∧ y * (x + 2) = 3740 → y = 85 :=
by
  sorry

end farmer_plough_rate_l26_2699


namespace prove_collinear_prove_perpendicular_l26_2660

noncomputable def vec_a : ℝ × ℝ := (1, 3)
noncomputable def vec_b : ℝ × ℝ := (3, -4)

def collinear (k : ℝ) : Prop :=
  let v1 := (k * 1 - 3, k * 3 + 4)
  let v2 := (1 + 3, 3 - 4)
  v1.1 * v2.2 = v1.2 * v2.1

def perpendicular (k : ℝ) : Prop :=
  let v1 := (k * 1 - 3, k * 3 + 4)
  let v2 := (1 + 3, 3 - 4)
  v1.1 * v2.1 + v1.2 * v2.2 = 0

theorem prove_collinear : collinear (-1) :=
by
  sorry

theorem prove_perpendicular : perpendicular (16) :=
by
  sorry

end prove_collinear_prove_perpendicular_l26_2660


namespace total_dots_not_visible_eq_54_l26_2644

theorem total_dots_not_visible_eq_54 :
  let die_sum := 21
  let num_dice := 4
  let total_sum := num_dice * die_sum
  let visible_sum := 1 + 2 + 3 + 4 + 4 + 5 + 5 + 6
  total_sum - visible_sum = 54 :=
by
  let die_sum := 21
  let num_dice := 4
  let total_sum := num_dice * die_sum
  let visible_sum := 1 + 2 + 3 + 4 + 4 + 5 + 5 + 6
  show total_sum - visible_sum = 54
  sorry

end total_dots_not_visible_eq_54_l26_2644


namespace solution_set_f_gt_5_range_m_f_ge_abs_2m1_l26_2653

noncomputable def f (x : ℝ) : ℝ := abs (2 * x - 1) + abs (x + 3)

theorem solution_set_f_gt_5 :
  {x : ℝ | f x > 5} = {x : ℝ | x < -1} ∪ {x : ℝ | x > 1} :=
by sorry

theorem range_m_f_ge_abs_2m1 :
  (∀ x : ℝ, f x ≥ abs (2 * m + 1)) ↔ -9/4 ≤ m ∧ m ≤ 5/4 :=
by sorry

end solution_set_f_gt_5_range_m_f_ge_abs_2m1_l26_2653


namespace problem_1_problem_2_l26_2694

open Real

noncomputable def f (x : ℝ) : ℝ := 3^x
noncomputable def g (x : ℝ) : ℝ := log x / log 3

theorem problem_1 : g 4 + g 8 - g (32 / 9) = 2 := 
by
  sorry

theorem problem_2 (x : ℝ) (h : 0 < x ∧ x < 1) : g (x / (1 - x)) < 1 ↔ 0 < x ∧ x < 3 / 4 :=
by
  sorry

end problem_1_problem_2_l26_2694


namespace jack_additional_sweets_is_correct_l26_2619

/-- Initial number of sweets --/
def initial_sweets : ℕ := 22

/-- Sweets taken by Paul --/
def sweets_taken_by_paul : ℕ := 7

/-- Jack's total sweets taken --/
def jack_total_sweets_taken : ℕ := initial_sweets - sweets_taken_by_paul

/-- Half of initial sweets --/
def half_initial_sweets : ℕ := initial_sweets / 2

/-- Additional sweets taken by Jack --/
def additional_sweets_taken_by_jack : ℕ := jack_total_sweets_taken - half_initial_sweets

theorem jack_additional_sweets_is_correct : additional_sweets_taken_by_jack = 4 := by
  sorry

end jack_additional_sweets_is_correct_l26_2619


namespace least_number_to_add_l26_2657

-- Definition of LCM for given primes
def lcm_of_primes : ℕ := 5 * 7 * 11 * 13 * 17 * 19

theorem least_number_to_add (n : ℕ) : 
  (5432 + n) % 5 = 0 ∧ 
  (5432 + n) % 7 = 0 ∧ 
  (5432 + n) % 11 = 0 ∧ 
  (5432 + n) % 13 = 0 ∧ 
  (5432 + n) % 17 = 0 ∧ 
  (5432 + n) % 19 = 0 ↔ 
  n = 1611183 :=
by sorry

end least_number_to_add_l26_2657


namespace p_necessary_for_q_l26_2678

variable (x : ℝ)

def p := (x - 3) * (|x| + 1) < 0
def q := |1 - x| < 2

theorem p_necessary_for_q : (∀ x, q x → p x) ∧ (∃ x, q x) ∧ (∃ x, ¬(p x ∧ q x)) := by
  sorry

end p_necessary_for_q_l26_2678


namespace c_positive_when_others_negative_l26_2638

variables {a b c d e f : ℤ}

theorem c_positive_when_others_negative (h_ab_cdef_lt_0 : a * b + c * d * e * f < 0)
  (h_a_neg : a < 0) (h_b_neg : b < 0) (h_d_neg : d < 0) (h_e_neg : e < 0) (h_f_neg : f < 0) 
  : c > 0 :=
sorry

end c_positive_when_others_negative_l26_2638


namespace monica_studied_32_67_hours_l26_2664

noncomputable def monica_total_study_time : ℚ :=
  let monday := 1
  let tuesday := 2 * monday
  let wednesday := 2
  let thursday := 3 * wednesday
  let friday := thursday / 2
  let total_weekday := monday + tuesday + wednesday + thursday + friday
  let saturday := total_weekday
  let sunday := saturday / 3
  total_weekday + saturday + sunday

theorem monica_studied_32_67_hours :
  monica_total_study_time = 32.67 := by
  sorry

end monica_studied_32_67_hours_l26_2664


namespace sum_a_n_eq_2014_l26_2679

def f (n : ℕ) : ℤ :=
  if n % 2 = 1 then (n : ℤ)^2 else - (n : ℤ)^2

def a (n : ℕ) : ℤ :=
  f n + f (n + 1)

theorem sum_a_n_eq_2014 : (Finset.range 2014).sum a = 2014 :=
by
  sorry

end sum_a_n_eq_2014_l26_2679


namespace class_size_l26_2686

theorem class_size :
  ∃ (N : ℕ), (20 ≤ N) ∧ (N ≤ 30) ∧ (∃ (x : ℕ), N = 3 * x + 1) ∧ (∃ (y : ℕ), N = 4 * y + 1) ∧ (N = 25) :=
by { sorry }

end class_size_l26_2686


namespace combination_identity_l26_2609

-- Lean statement defining the proof problem
theorem combination_identity : Nat.choose 12 5 + Nat.choose 12 6 = Nat.choose 13 6 :=
  sorry

end combination_identity_l26_2609


namespace temperature_reaches_90_at_17_l26_2674

def temperature (t : ℝ) : ℝ := -t^2 + 14 * t + 40

theorem temperature_reaches_90_at_17 :
  ∃ t : ℝ, temperature t = 90 ∧ t = 17 :=
by
  exists 17
  dsimp [temperature]
  norm_num
  sorry

end temperature_reaches_90_at_17_l26_2674


namespace relationship_p_q_no_linear_term_l26_2634

theorem relationship_p_q_no_linear_term (p q : ℝ) :
  (∀ x : ℝ, (x^2 - p * x + q) * (x - 3) = x^3 + (-p - 3) * x^2 + (3 * p + q) * x - 3 * q) 
  → (3 * p + q = 0) → (q + 3 * p = 0) :=
by
  intro h_expansion coeff_zero
  sorry

end relationship_p_q_no_linear_term_l26_2634


namespace value_of_x_when_y_is_six_l26_2680

theorem value_of_x_when_y_is_six 
  (k : ℝ) -- The constant of variation
  (h1 : ∀ y : ℝ, x = k / y^2) -- The inverse relationship
  (h2 : y = 2)
  (h3 : x = 1)
  : x = 1 / 9 :=
by
  sorry

end value_of_x_when_y_is_six_l26_2680


namespace total_seeds_gray_sections_combined_l26_2691

noncomputable def total_seeds_first_circle : ℕ := 87
noncomputable def seeds_white_first_circle : ℕ := 68
noncomputable def total_seeds_second_circle : ℕ := 110
noncomputable def seeds_white_second_circle : ℕ := 68

theorem total_seeds_gray_sections_combined :
  (total_seeds_first_circle - seeds_white_first_circle) +
  (total_seeds_second_circle - seeds_white_second_circle) = 61 :=
by
  sorry

end total_seeds_gray_sections_combined_l26_2691


namespace unique_mod_inverse_l26_2608

theorem unique_mod_inverse (a n : ℤ) (coprime : Int.gcd a n = 1) : 
  ∃! b : ℤ, (a * b) % n = 1 % n := 
sorry

end unique_mod_inverse_l26_2608


namespace product_of_solutions_of_abs_equation_l26_2693

theorem product_of_solutions_of_abs_equation :
  (∃ x₁ x₂ : ℚ, |5 * x₁ - 2| + 7 = 52 ∧ |5 * x₂ - 2| + 7 = 52 ∧ x₁ ≠ x₂ ∧ (x₁ * x₂ = -2021 / 25)) :=
sorry

end product_of_solutions_of_abs_equation_l26_2693


namespace gcd_of_8_and_12_l26_2672

theorem gcd_of_8_and_12 :
  let a := 8
  let b := 12
  let lcm_ab := 24
  Nat.lcm a b = lcm_ab → Nat.gcd a b = 4 :=
by
  intros
  sorry

end gcd_of_8_and_12_l26_2672


namespace cooks_in_restaurant_l26_2671

theorem cooks_in_restaurant
  (C W : ℕ) 
  (h1 : C * 8 = 3 * W) 
  (h2 : C * 4 = (W + 12)) :
  C = 9 :=
by
  sorry

end cooks_in_restaurant_l26_2671


namespace skating_probability_given_skiing_l26_2605

theorem skating_probability_given_skiing (P_A P_B P_A_or_B : ℝ)
    (h1 : P_A = 0.6) (h2 : P_B = 0.5) (h3 : P_A_or_B = 0.7) : 
    (P_A_or_B = P_A + P_B - P_A * P_B) → 
    ((P_A * P_B) / P_B = 0.8) := 
    by
        intros
        sorry

end skating_probability_given_skiing_l26_2605


namespace problem_solution_l26_2696

def p (x : ℝ) : ℝ := x^2 - 4*x + 3
def tilde_p (x : ℝ) : ℝ := p (p x)

-- Proof problem: Prove tilde_p 2 = -4 
theorem problem_solution : tilde_p 2 = -4 := sorry

end problem_solution_l26_2696


namespace outer_boundary_diameter_l26_2646

theorem outer_boundary_diameter (fountain_diameter garden_width path_width : ℝ) 
(h1 : fountain_diameter = 12) 
(h2 : garden_width = 10) 
(h3 : path_width = 6) : 
2 * ((fountain_diameter / 2) + garden_width + path_width) = 44 :=
by
  -- Sorry, proof not needed for this statement
  sorry

end outer_boundary_diameter_l26_2646


namespace problem1_problem2_l26_2615

theorem problem1 : ((- (5 : ℚ) / 6) + 2 / 3) / (- (7 / 12)) * (7 / 2) = 1 := 
sorry

theorem problem2 : ((1 - 1 / 6) * (-3) - (- (11 / 6)) / (- (22 / 3))) = - (11 / 4) := 
sorry

end problem1_problem2_l26_2615


namespace find_a_l26_2607

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 3 * x^2 + 2

theorem find_a (a : ℝ) (h : deriv (f a) (-1) = 4) : a = 10 / 3 :=
by {
  sorry
}

end find_a_l26_2607


namespace unique_value_of_W_l26_2604

theorem unique_value_of_W (T O W F U R : ℕ) (h1 : T = 8) (h2 : O % 2 = 0) (h3 : ∀ x y, x ≠ y → x = O → y = T → x ≠ O) :
  (T + T) * 10^2 + (W + W) * 10 + (O + O) = F * 10^3 + O * 10^2 + U * 10 + R → W = 3 :=
by
  sorry

end unique_value_of_W_l26_2604


namespace percentage_A_to_B_l26_2635

variable (A B : ℕ)
variable (total : ℕ := 570)
variable (B_amount : ℕ := 228)

theorem percentage_A_to_B :
  (A + B = total) →
  B = B_amount →
  (A = total - B_amount) →
  ((A / B_amount : ℚ) * 100 = 150) :=
sorry

end percentage_A_to_B_l26_2635


namespace puppy_sleep_duration_l26_2655

-- Definitions based on the given conditions
def connor_sleep_hours : ℕ := 6
def luke_sleep_hours : ℕ := connor_sleep_hours + 2
def puppy_sleep_hours : ℕ := 2 * luke_sleep_hours

-- Theorem stating the puppy's sleep duration
theorem puppy_sleep_duration : puppy_sleep_hours = 16 :=
by
  -- ( Proof goes here )
  sorry

end puppy_sleep_duration_l26_2655


namespace car_travel_time_l26_2685

-- Definitions
def speed : ℝ := 50
def miles_per_gallon : ℝ := 30
def tank_capacity : ℝ := 15
def fraction_used : ℝ := 0.5555555555555556

-- Theorem statement
theorem car_travel_time : (fraction_used * tank_capacity * miles_per_gallon / speed) = 5 :=
sorry

end car_travel_time_l26_2685


namespace sum_of_real_solutions_l26_2610

theorem sum_of_real_solutions:
  (∃ (s : ℝ), ∀ x : ℝ, 
    (x - 3) / (x^2 + 6 * x + 2) = (x - 6) / (x^2 - 12 * x) → 
    s = 106 / 9) :=
  sorry

end sum_of_real_solutions_l26_2610


namespace value_of_M_l26_2695

theorem value_of_M :
  let row_seq := [25, 25 + (8 - 25) / 3, 25 + 2 * (8 - 25) / 3, 8, 8 + (8 - 25) / 3, 8 + 2 * (8 - 25) / 3, -9]
  let col_seq1 := [25, 25 - 4, 25 - 8]
  let col_seq2 := [16, 20, 20 + 4]
  let col_seq3 := [-9, -9 - 11/4, -9 - 2 * 11/4, -20]
  let M := -9 - (-11/4)
  M = -6.25 :=
by
  sorry

end value_of_M_l26_2695


namespace negation_exists_l26_2677

theorem negation_exists :
  (¬ ∀ x : ℝ, x^2 + 1 ≥ x) ↔ ∃ x : ℝ, x^2 + 1 < x :=
sorry

end negation_exists_l26_2677


namespace average_speed_car_l26_2684

theorem average_speed_car (speed_first_hour ground_speed_headwind speed_second_hour : ℝ) (time_first_hour time_second_hour : ℝ) (h1 : speed_first_hour = 90) (h2 : ground_speed_headwind = 10) (h3 : speed_second_hour = 55) (h4 : time_first_hour = 1) (h5 : time_second_hour = 1) : 
(speed_first_hour + ground_speed_headwind) * time_first_hour + speed_second_hour * time_second_hour / (time_first_hour + time_second_hour) = 77.5 :=
sorry

end average_speed_car_l26_2684

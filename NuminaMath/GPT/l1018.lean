import Mathlib

namespace part1_solution_set_part2_range_of_a_l1018_101858

-- Part 1
theorem part1_solution_set (x : ℝ) : (|x - 1| + |x + 3|) ≥ 6 ↔ (x ≤ -4 ∨ x ≥ 2) :=
sorry

-- Part 2
theorem part2_range_of_a (a : ℝ) : (∀ x : ℝ, |x - a| + |x + 3| > -a) ↔ (a > -3 / 2) :=
sorry

end part1_solution_set_part2_range_of_a_l1018_101858


namespace percent_students_both_correct_l1018_101833

def percent_answered_both_questions (total_students first_correct second_correct neither_correct : ℕ) : ℕ :=
  let at_least_one_correct := total_students - neither_correct
  let total_individual_correct := first_correct + second_correct
  total_individual_correct - at_least_one_correct

theorem percent_students_both_correct
  (total_students : ℕ)
  (first_question_correct : ℕ)
  (second_question_correct : ℕ)
  (neither_question_correct : ℕ) 
  (h_total_students : total_students = 100)
  (h_first_correct : first_question_correct = 80)
  (h_second_correct : second_question_correct = 55)
  (h_neither_correct : neither_question_correct = 20) :
  percent_answered_both_questions total_students first_question_correct second_question_correct neither_question_correct = 55 :=
by
  rw [h_total_students, h_first_correct, h_second_correct, h_neither_correct]
  sorry


end percent_students_both_correct_l1018_101833


namespace part_one_part_two_l1018_101832

noncomputable def f (x : ℝ) : ℝ := (x + 1) * Real.log x - x + 1

noncomputable def f' (x : ℝ) : ℝ := Real.log x + 1 / x

theorem part_one (x a : ℝ) (hx : x > 0) (ineq : x * f' x ≤ x^2 + a * x + 1) : a ∈ Set.Ici (-1) :=
by sorry

theorem part_two (x : ℝ) (hx : x > 0) : (x - 1) * f x ≥ 0 :=
by sorry

end part_one_part_two_l1018_101832


namespace grace_walks_distance_l1018_101880

theorem grace_walks_distance
  (south_blocks west_blocks : ℕ)
  (block_length_in_miles : ℚ)
  (h_south_blocks : south_blocks = 4)
  (h_west_blocks : west_blocks = 8)
  (h_block_length : block_length_in_miles = 1 / 4)
  : ((south_blocks + west_blocks) * block_length_in_miles = 3) :=
by 
  sorry

end grace_walks_distance_l1018_101880


namespace acme_profit_calculation_l1018_101877

theorem acme_profit_calculation :
  let initial_outlay := 12450
  let cost_per_set := 20.75
  let selling_price := 50
  let number_of_sets := 950
  let total_revenue := number_of_sets * selling_price
  let total_manufacturing_costs := initial_outlay + cost_per_set * number_of_sets
  let profit := total_revenue - total_manufacturing_costs 
  profit = 15337.50 := 
by
  sorry

end acme_profit_calculation_l1018_101877


namespace sum_of_x_and_y_l1018_101892

theorem sum_of_x_and_y (x y : ℤ) (h : 2 * x * y + x + y = 83) : x + y = 83 ∨ x + y = -85 := 
sorry

end sum_of_x_and_y_l1018_101892


namespace probability_of_both_selected_l1018_101846

variable (P_ram : ℚ) (P_ravi : ℚ) (P_both : ℚ)

def selection_probability (P_ram : ℚ) (P_ravi : ℚ) : ℚ :=
  P_ram * P_ravi

theorem probability_of_both_selected (h1 : P_ram = 3/7) (h2 : P_ravi = 1/5) :
  selection_probability P_ram P_ravi = P_both :=
by
  sorry

end probability_of_both_selected_l1018_101846


namespace chess_players_swim_not_every_swimmer_plays_tennis_tennis_players_play_chess_l1018_101861

variables (Bodyguards : Type)
variables (U S T : Bodyguards → Prop)

-- Conditions
axiom cond1 : ∀ x, (T x → (S x → U x))
axiom cond2 : ∀ x, (S x → (U x ∨ T x))
axiom cond3 : ∀ x, (¬ U x ∧ T x → S x)

-- To prove
theorem chess_players_swim : ∀ x, (S x → U x) := by
  sorry

theorem not_every_swimmer_plays_tennis : ¬ ∀ x, (U x → T x) := by
  sorry

theorem tennis_players_play_chess : ∀ x, (T x → S x) := by
  sorry

end chess_players_swim_not_every_swimmer_plays_tennis_tennis_players_play_chess_l1018_101861


namespace platform_length_l1018_101842

theorem platform_length (train_speed_kmph : ℕ) (train_time_man_seconds : ℕ) (train_time_platform_seconds : ℕ) (train_speed_mps : ℕ) : 
  train_speed_kmph = 54 →
  train_time_man_seconds = 20 →
  train_time_platform_seconds = 30 →
  train_speed_mps = (54 * 1000 / 3600) →
  (54 * 5 / 18) = 15 →
  ∃ (P : ℕ), (train_speed_mps * train_time_platform_seconds) = (train_speed_mps * train_time_man_seconds) + P ∧ P = 150 :=
by
  sorry

end platform_length_l1018_101842


namespace subtract_one_from_solution_l1018_101843

theorem subtract_one_from_solution (x : ℝ) (h : 15 * x = 45) : (x - 1) = 2 := 
by {
  sorry
}

end subtract_one_from_solution_l1018_101843


namespace part1_part2_l1018_101807

def pointA : (ℝ × ℝ) := (1, 2)
def pointB : (ℝ × ℝ) := (-2, 3)
def pointC : (ℝ × ℝ) := (8, -5)

-- Definitions of the vectors
def OA : (ℝ × ℝ) := pointA
def OB : (ℝ × ℝ) := pointB
def OC : (ℝ × ℝ) := pointC
def AB : (ℝ × ℝ) := (pointB.1 - pointA.1, pointB.2 - pointA.2)

-- Part 1: Proving the values of x and y
theorem part1 : ∃ (x y : ℝ), OC = (x * OA.1 + y * OB.1, x * OA.2 + y * OB.2) ∧ x = 2 ∧ y = -3 :=
by
  sorry

-- Part 2: Proving the value of m when vectors are parallel
theorem part2 : ∃ (m : ℝ), ∃ k : ℝ, AB = (k * (m + 8), k * (2 * m - 5)) ∧ m = 1 :=
by
  sorry

end part1_part2_l1018_101807


namespace sufficient_condition_not_necessary_condition_l1018_101840

variables (p q : Prop)
def φ := ¬p ∧ ¬q
def ψ := ¬p

theorem sufficient_condition : φ p q → ψ p := 
sorry

theorem not_necessary_condition : ψ p → ¬ (φ p q) :=
sorry

end sufficient_condition_not_necessary_condition_l1018_101840


namespace room_total_space_l1018_101864

-- Definitions based on the conditions
def bookshelf_space : ℕ := 80
def reserved_space : ℕ := 160
def number_of_shelves : ℕ := 3

-- The theorem statement
theorem room_total_space : 
  (number_of_shelves * bookshelf_space) + reserved_space = 400 := 
by
  sorry

end room_total_space_l1018_101864


namespace find_a_for_even_function_l1018_101859

theorem find_a_for_even_function (f : ℝ → ℝ) (a : ℝ) 
  (h_even : ∀ x, f x = f (-x)) 
  (h_neg : ∀ x, x < 0 → f x = x^2 + a * x) 
  (h_value : f 3 = 3) : a = 2 :=
sorry

end find_a_for_even_function_l1018_101859


namespace unique_k_for_triangle_inequality_l1018_101818

theorem unique_k_for_triangle_inequality (k : ℕ) (h : 0 < k) :
  (∀ (a b c : ℝ), 0 < a → 0 < b → 0 < c → k * (a * b + b * c + c * a) > 5 * (a * b + b * b + c * c) → a + b > c ∧ b + c > a ∧ c + a > b) ↔ (k = 6) :=
by
  sorry

end unique_k_for_triangle_inequality_l1018_101818


namespace average_speed_round_trip_l1018_101811

theorem average_speed_round_trip (d : ℝ) (h_d_pos : d > 0) : 
  let t1 := d / 80
  let t2 := d / 120
  let d_total := 2 * d
  let t_total := t1 + t2
  let v_avg := d_total / t_total
  v_avg = 96 :=
by
  sorry

end average_speed_round_trip_l1018_101811


namespace eccentricity_of_ellipse_l1018_101812

theorem eccentricity_of_ellipse 
  (a b : ℝ) (e : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : ∃ (x y : ℝ), x = 0 ∧ y > 0 ∧ (9 * b^2 = 16/7 * a^2)) :
  e = Real.sqrt (10) / 6 :=
sorry

end eccentricity_of_ellipse_l1018_101812


namespace julia_played_with_34_kids_l1018_101853

-- Define the number of kids Julia played with on each day
def kidsMonday : Nat := 17
def kidsTuesday : Nat := 15
def kidsWednesday : Nat := 2

-- Define the total number of kids Julia played with
def totalKids : Nat := kidsMonday + kidsTuesday + kidsWednesday

-- Prove given conditions
theorem julia_played_with_34_kids :
  totalKids = 34 :=
by
  sorry

end julia_played_with_34_kids_l1018_101853


namespace factorize_quadratic_l1018_101809

theorem factorize_quadratic (x : ℝ) : x^2 - 6 * x + 9 = (x - 3)^2 :=
by {
  sorry  -- Proof goes here
}

end factorize_quadratic_l1018_101809


namespace average_output_l1018_101899

theorem average_output (t1 t2 t_total : ℝ) (c1 c2 c_total : ℕ) 
                        (h1 : c1 = 60) (h2 : c2 = 60) 
                        (rate1 : ℝ := 15) (rate2 : ℝ := 60) :
  t1 = c1 / rate1 ∧ t2 = c2 / rate2 ∧ t_total = t1 + t2 ∧ c_total = c1 + c2 → 
  (c_total / t_total = 24) := 
by 
  sorry

end average_output_l1018_101899


namespace cone_generatrix_length_theorem_l1018_101827

noncomputable def cone_generatrix_length 
  (diameter : ℝ) 
  (unfolded_side_area : ℝ) 
  (h_diameter : diameter = 6)
  (h_area : unfolded_side_area = 18 * Real.pi) : 
  ℝ :=
6

theorem cone_generatrix_length_theorem 
  (diameter : ℝ) 
  (unfolded_side_area : ℝ) 
  (h_diameter : diameter = 6)
  (h_area : unfolded_side_area = 18 * Real.pi) :
  cone_generatrix_length diameter unfolded_side_area h_diameter h_area = 6 :=
sorry

end cone_generatrix_length_theorem_l1018_101827


namespace consecutive_sum_l1018_101844

theorem consecutive_sum (m k : ℕ) (h : (k + 1) * (2 * m + k) = 2000) :
  (m = 1000 ∧ k = 0) ∨ 
  (m = 198 ∧ k = 4) ∨ 
  (m = 28 ∧ k = 24) ∨ 
  (m = 55 ∧ k = 15) :=
by sorry

end consecutive_sum_l1018_101844


namespace elaineExpenseChanges_l1018_101855

noncomputable def elaineIncomeLastYear : ℝ := 20000 + 5000
noncomputable def elaineExpensesLastYearRent := 0.10 * elaineIncomeLastYear
noncomputable def elaineExpensesLastYearGroceries := 0.20 * elaineIncomeLastYear
noncomputable def elaineExpensesLastYearHealthcare := 0.15 * elaineIncomeLastYear
noncomputable def elaineTotalExpensesLastYear := elaineExpensesLastYearRent + elaineExpensesLastYearGroceries + elaineExpensesLastYearHealthcare
noncomputable def elaineSavingsLastYear := elaineIncomeLastYear - elaineTotalExpensesLastYear

noncomputable def elaineIncomeThisYear : ℝ := 23000 + 10000
noncomputable def elaineExpensesThisYearRent := 0.30 * elaineIncomeThisYear
noncomputable def elaineExpensesThisYearGroceries := 0.25 * elaineIncomeThisYear
noncomputable def elaineExpensesThisYearHealthcare := (0.15 * elaineIncomeThisYear) * 1.10
noncomputable def elaineTotalExpensesThisYear := elaineExpensesThisYearRent + elaineExpensesThisYearGroceries + elaineExpensesThisYearHealthcare
noncomputable def elaineSavingsThisYear := elaineIncomeThisYear - elaineTotalExpensesThisYear

theorem elaineExpenseChanges :
  ( ((elaineExpensesThisYearRent - elaineExpensesLastYearRent) / elaineExpensesLastYearRent) * 100 = 296)
  ∧ ( ((elaineExpensesThisYearGroceries - elaineExpensesLastYearGroceries) / elaineExpensesLastYearGroceries) * 100 = 65)
  ∧ ( ((elaineExpensesThisYearHealthcare - elaineExpensesLastYearHealthcare) / elaineExpensesLastYearHealthcare) * 100 = 45.2)
  ∧ ( (elaineSavingsLastYear / elaineIncomeLastYear) * 100 = 55)
  ∧ ( (elaineSavingsThisYear / elaineIncomeThisYear) * 100 = 28.5)
  ∧ ( (elaineTotalExpensesLastYear / elaineIncomeLastYear) = 0.45 )
  ∧ ( (elaineTotalExpensesThisYear / elaineIncomeThisYear) = 0.715 )
  ∧ ( (elaineSavingsLastYear - elaineSavingsThisYear) = 4345 ∧ ( (55 - ((elaineSavingsThisYear / elaineIncomeThisYear) * 100)) = 26.5 ))
:= by sorry

end elaineExpenseChanges_l1018_101855


namespace quadratic_roots_solution_l1018_101816

noncomputable def quadratic_roots_differ_by_2 (p q : ℝ) (hq_pos : 0 < q) (hp_pos : 0 < p) : Prop :=
  let root1 := (-p + Real.sqrt (p^2 - 4*q)) / 2
  let root2 := (-p - Real.sqrt (p^2 - 4*q)) / 2
  abs (root1 - root2) = 2

theorem quadratic_roots_solution (p q : ℝ) (hq_pos : 0 < q) (hp_pos : 0 < p) :
  quadratic_roots_differ_by_2 p q hq_pos hp_pos →
  p = 2 * Real.sqrt (q + 1) :=
sorry

end quadratic_roots_solution_l1018_101816


namespace slant_asymptote_sum_l1018_101885

theorem slant_asymptote_sum (x : ℝ) (hx : x ≠ 5) :
  (5 : ℝ) + (21 : ℝ) = 26 :=
by
  sorry

end slant_asymptote_sum_l1018_101885


namespace race_positions_l1018_101876

theorem race_positions :
  ∀ (M J T R H D : ℕ),
    (M = J + 3) →
    (J = T + 1) →
    (T = R + 3) →
    (H = R + 5) →
    (D = H + 4) →
    (M = 9) →
    H = 7 :=
by sorry

end race_positions_l1018_101876


namespace coefficient_a_eq_2_l1018_101895

theorem coefficient_a_eq_2 (a : ℝ) (h : (a^3 * (4 : ℝ)) = 32) : a = 2 :=
by {
  -- Proof will need to be filled in here
  sorry
}

end coefficient_a_eq_2_l1018_101895


namespace find_interest_rate_of_initial_investment_l1018_101821

def initial_investment : ℝ := 1400
def additional_investment : ℝ := 700
def total_investment : ℝ := 2100
def additional_interest_rate : ℝ := 0.08
def target_total_income_rate : ℝ := 0.06
def target_total_income : ℝ := target_total_income_rate * total_investment

theorem find_interest_rate_of_initial_investment (r : ℝ) :
  (initial_investment * r + additional_investment * additional_interest_rate = target_total_income) → 
  (r = 0.05) :=
by
  sorry

end find_interest_rate_of_initial_investment_l1018_101821


namespace total_spending_correct_l1018_101874

-- Define the costs and number of children for each ride and snack
def cost_ferris_wheel := 5 * 5
def cost_roller_coaster := 7 * 3
def cost_merry_go_round := 3 * 8
def cost_bumper_cars := 4 * 6

def cost_ice_cream := 8 * 2 * 5
def cost_hot_dog := 6 * 4
def cost_pizza := 4 * 3

-- Calculate the total cost
def total_cost_rides := cost_ferris_wheel + cost_roller_coaster + cost_merry_go_round + cost_bumper_cars
def total_cost_snacks := cost_ice_cream + cost_hot_dog + cost_pizza
def total_spent := total_cost_rides + total_cost_snacks

-- The statement to prove
theorem total_spending_correct : total_spent = 170 := by
  sorry

end total_spending_correct_l1018_101874


namespace annie_total_spent_l1018_101868

-- Define cost of a single television
def cost_per_tv : ℕ := 50
-- Define number of televisions bought
def number_of_tvs : ℕ := 5
-- Define cost of a single figurine
def cost_per_figurine : ℕ := 1
-- Define number of figurines bought
def number_of_figurines : ℕ := 10

-- Define total cost calculation
noncomputable def total_cost : ℕ :=
  number_of_tvs * cost_per_tv + number_of_figurines * cost_per_figurine

theorem annie_total_spent : total_cost = 260 := by
  sorry

end annie_total_spent_l1018_101868


namespace cubic_sum_l1018_101873

theorem cubic_sum (p q r : ℝ) (h1 : p + q + r = 4) (h2 : p * q + q * r + r * p = 7) (h3 : p * q * r = -10) :
  p ^ 3 + q ^ 3 + r ^ 3 = 154 := 
by sorry

end cubic_sum_l1018_101873


namespace maximum_positive_factors_l1018_101887

theorem maximum_positive_factors (b n : ℕ) (hb : 0 < b ∧ b ≤ 20) (hn : 0 < n ∧ n ≤ 15) :
  ∃ k, (k = b^n) ∧ (∀ m, m = b^n → m.factors.count ≤ 61) :=
sorry

end maximum_positive_factors_l1018_101887


namespace parakeets_in_each_cage_l1018_101884

variable (num_cages : ℕ) (parrots_per_cage : ℕ) (total_birds : ℕ)

-- Given conditions
def total_parrots (num_cages parrots_per_cage : ℕ) : ℕ := num_cages * parrots_per_cage
def total_parakeets (total_birds total_parrots : ℕ) : ℕ := total_birds - total_parrots
def parakeets_per_cage (total_parakeets num_cages : ℕ) : ℕ := total_parakeets / num_cages

-- Theorem: Number of parakeets in each cage is 7
theorem parakeets_in_each_cage (h1 : num_cages = 8) (h2 : parrots_per_cage = 2) (h3 : total_birds = 72) : 
  parakeets_per_cage (total_parakeets total_birds (total_parrots num_cages parrots_per_cage)) num_cages = 7 :=
by
  sorry

end parakeets_in_each_cage_l1018_101884


namespace sum_of_a_for_repeated_root_l1018_101886

theorem sum_of_a_for_repeated_root :
  ∀ a : ℝ, (∀ x : ℝ, 2 * x^2 + a * x + 10 * x + 16 = 0 → 
               (a + 10 = 8 * Real.sqrt 2 ∨ a + 10 = -8 * Real.sqrt 2)) → 
               (a = -10 + 8 * Real.sqrt 2 ∨ a = -10 - 8 * Real.sqrt 2) → 
               ((-10 + 8 * Real.sqrt 2) + (-10 - 8 * Real.sqrt 2) = -20) := by
sorry

end sum_of_a_for_repeated_root_l1018_101886


namespace simplify_expression_l1018_101831

theorem simplify_expression (b : ℝ) (hb : b = -1) : 
  (3 * b⁻¹ + (2 * b⁻¹) / 3) / b = 11 / 3 :=
by
  sorry

end simplify_expression_l1018_101831


namespace river_width_l1018_101847

theorem river_width (w : ℕ) (speed_const : ℕ) 
(meeting1_from_nearest_shore : ℕ) (meeting2_from_other_shore : ℕ)
(h1 : speed_const = 1) 
(h2 : meeting1_from_nearest_shore = 720) 
(h3 : meeting2_from_other_shore = 400)
(h4 : 3 * w = 3 * meeting1_from_nearest_shore)
(h5 : 2160 = 2 * w - meeting2_from_other_shore) :
w = 1280 :=
by
  {
      sorry
  }

end river_width_l1018_101847


namespace product_identity_l1018_101830

theorem product_identity :
  (1 + 1 / Nat.factorial 1) * (1 + 1 / Nat.factorial 2) * (1 + 1 / Nat.factorial 3) *
  (1 + 1 / Nat.factorial 4) * (1 + 1 / Nat.factorial 5) * (1 + 1 / Nat.factorial 6) *
  (1 + 1 / Nat.factorial 7) = 5041 / 5040 := sorry

end product_identity_l1018_101830


namespace no_such_pairs_l1018_101865

theorem no_such_pairs :
  ¬ ∃ (b c : ℕ), b > 0 ∧ c > 0 ∧ (b^2 - 4 * c < 0) ∧ (c^2 - 4 * b < 0) := sorry

end no_such_pairs_l1018_101865


namespace learning_hours_difference_l1018_101841

/-- Define the hours Ryan spends on each language. -/
def hours_learned (lang : String) : ℝ :=
  if lang = "English" then 2 else
  if lang = "Chinese" then 5 else
  if lang = "Spanish" then 4 else
  if lang = "French" then 3 else
  if lang = "German" then 1.5 else 0

/-- Prove that Ryan spends 2.5 more hours learning Chinese and French combined
    than he does learning German and Spanish combined. -/
theorem learning_hours_difference :
  hours_learned "Chinese" + hours_learned "French" - (hours_learned "German" + hours_learned "Spanish") = 2.5 :=
by
  sorry

end learning_hours_difference_l1018_101841


namespace largest_divisor_of_odd_product_l1018_101828

theorem largest_divisor_of_odd_product (n : ℕ) (h : Even n ∧ n > 0) :
  ∃ m, m > 0 ∧ (∀ k, (n+1)*(n+3)*(n+7)*(n+9)*(n+11) % k = 0 ↔ k ≤ 15) := by
  -- Proof goes here
  sorry

end largest_divisor_of_odd_product_l1018_101828


namespace point_on_line_has_correct_y_l1018_101800

theorem point_on_line_has_correct_y (a : ℝ) : (2 * 3 + a - 7 = 0) → a = 1 :=
by 
  sorry

end point_on_line_has_correct_y_l1018_101800


namespace initial_men_in_fort_l1018_101870

theorem initial_men_in_fort (M : ℕ) 
  (h1 : ∀ N : ℕ, M * 35 = (N - 25) * 42) 
  (h2 : 10 + 42 = 52) : M = 150 :=
sorry

end initial_men_in_fort_l1018_101870


namespace tangent_line_y_intercept_l1018_101894

noncomputable def y_intercept_tangent_line (R1_center R2_center : ℝ × ℝ)
  (R1_radius R2_radius : ℝ) : ℝ :=
if R1_center = (3,0) ∧ R2_center = (8,0) ∧ R1_radius = 3 ∧ R2_radius = 2
then 15 * Real.sqrt 26 / 26
else 0

theorem tangent_line_y_intercept : 
  y_intercept_tangent_line (3,0) (8,0) 3 2 = 15 * Real.sqrt 26 / 26 :=
by
  -- proof goes here
  sorry

end tangent_line_y_intercept_l1018_101894


namespace comparison_of_abc_l1018_101856

noncomputable def a : ℝ := (4 - Real.log 4) / Real.exp 2
noncomputable def b : ℝ := Real.log 2 / 2
noncomputable def c : ℝ := 1 / Real.exp 1

theorem comparison_of_abc : b < a ∧ a < c :=
by
  sorry

end comparison_of_abc_l1018_101856


namespace dice_product_sum_impossible_l1018_101898

theorem dice_product_sum_impossible (d1 d2 d3 d4 : ℕ) (h1 : 1 ≤ d1 ∧ d1 ≤ 6) (h2 : 1 ≤ d2 ∧ d2 ≤ 6) (h3 : 1 ≤ d3 ∧ d3 ≤ 6) (h4 : 1 ≤ d4 ∧ d4 ≤ 6) (hprod : d1 * d2 * d3 * d4 = 180) :
  (d1 + d2 + d3 + d4 ≠ 14) ∧ (d1 + d2 + d3 + d4 ≠ 17) :=
by
  sorry

end dice_product_sum_impossible_l1018_101898


namespace fraction_changes_l1018_101860

theorem fraction_changes (x y : ℝ) (h : 0 < x ∧ 0 < y) :
  (x + y) / (x * y) = 2 * ((2 * x + 2 * y) / (2 * x * 2 * y)) :=
by
  sorry

end fraction_changes_l1018_101860


namespace limit_of_sequence_z_l1018_101825

open Nat Real

noncomputable def sequence_z (n : ℕ) : ℝ :=
  -3 + (-1)^n / (n^2 : ℝ)

theorem limit_of_sequence_z :
  ∀ ε > 0, ∃ N : ℕ, ∀ n > N, abs (sequence_z n + 3) < ε :=
by
  sorry

end limit_of_sequence_z_l1018_101825


namespace sufficient_and_necessary_condition_l1018_101890

theorem sufficient_and_necessary_condition (a : ℝ) :
  (∀ x : ℝ, a * x^2 - 2 * a * x + 1 > 0) ↔ (0 < a ∧ a < 1) :=
by 
  sorry

end sufficient_and_necessary_condition_l1018_101890


namespace perpendicular_os_bc_l1018_101817

variable {A B C O S : Type}

noncomputable def acute_triangle (A B C : Type) := true -- Placeholder definition for acute triangle.

noncomputable def circumcenter (O : Type) (A B C : Type) := true -- Placeholder definition for circumcenter.

noncomputable def line_intersects_circumcircle_second_time (AC : Type) (circ : Type) (S : Type) := true -- Placeholder def.

-- Define the problem in Lean
theorem perpendicular_os_bc
  (ABC_is_acute : acute_triangle A B C)
  (O_is_circumcenter : circumcenter O A B C)
  (AC_intersects_AOB_circumcircle_at_S : line_intersects_circumcircle_second_time (A → C) (A → B → O) S) :
  true := -- Place for the proof that OS ⊥ BC
sorry

end perpendicular_os_bc_l1018_101817


namespace find_pairs_l1018_101891

open Nat

-- m and n are odd natural numbers greater than 2009
def is_odd_gt_2009 (x : ℕ) : Prop := (x % 2 = 1) ∧ (x > 2009)

-- condition: m divides n^2 + 8
def divides_m_n_squared_plus_8 (m n : ℕ) : Prop := m ∣ (n ^ 2 + 8)

-- condition: n divides m^2 + 8
def divides_n_m_squared_plus_8 (m n : ℕ) : Prop := n ∣ (m ^ 2 + 8)

-- Final statement
theorem find_pairs :
  ∃ m n : ℕ, is_odd_gt_2009 m ∧ is_odd_gt_2009 n ∧ divides_m_n_squared_plus_8 m n ∧ divides_n_m_squared_plus_8 m n ∧ ((m, n) = (881, 89) ∨ (m, n) = (3303, 567)) :=
sorry

end find_pairs_l1018_101891


namespace solve_fractional_equation_1_solve_fractional_equation_2_l1018_101889

-- Proof Problem 1
theorem solve_fractional_equation_1 (x : ℝ) (h : 6 * x - 2 ≠ 0) :
  (3 / 2 - 1 / (3 * x - 1) = 5 / (6 * x - 2)) ↔ (x = 10 / 9) :=
sorry

-- Proof Problem 2
theorem solve_fractional_equation_2 (x : ℝ) (h1 : 3 * x - 6 ≠ 0) :
  (5 * x - 4) / (x - 2) = (4 * x + 10) / (3 * x - 6) - 1 → false :=
sorry

end solve_fractional_equation_1_solve_fractional_equation_2_l1018_101889


namespace sum_first_40_terms_l1018_101803

-- Given: The sum of the first 10 terms of a geometric sequence is 9
axiom S_10 : ℕ → ℕ
axiom sum_S_10 : S_10 10 = 9 

-- Given: The sum of the terms from the 11th to the 20th is 36
axiom S_20 : ℕ → ℕ
axiom sum_S_20 : S_20 20 - S_10 10 = 36

-- Let Sn be the sum of the first n terms in the geometric sequence
def Sn (n : ℕ) : ℕ := sorry

-- Prove: The sum of the first 40 terms is 144
theorem sum_first_40_terms : Sn 40 = 144 := sorry

end sum_first_40_terms_l1018_101803


namespace solve_for_b_l1018_101849

theorem solve_for_b (b : ℝ) (h : b + b / 4 = 10 / 4) : b = 2 :=
sorry

end solve_for_b_l1018_101849


namespace total_cost_of_items_l1018_101869

theorem total_cost_of_items (m n : ℕ) : (8 * m + 5 * n) = 8 * m + 5 * n := 
by sorry

end total_cost_of_items_l1018_101869


namespace discounted_price_l1018_101872

variable (marked_price : ℝ) (discount_rate : ℝ)
variable (marked_price_def : marked_price = 150)
variable (discount_rate_def : discount_rate = 20)

theorem discounted_price (hmp : marked_price = 150) (hdr : discount_rate = 20) : 
  marked_price - (discount_rate / 100) * marked_price = 120 := by
  rw [hmp, hdr]
  sorry

end discounted_price_l1018_101872


namespace triangle_side_ratio_triangle_area_l1018_101839

-- Definition of Problem 1
theorem triangle_side_ratio {A B C a b c : ℝ} 
  (h1 : 4 * Real.sin A = 3 * Real.sin B)
  (h2 : 2 * a * Real.cos C + 2 * c * Real.cos A = a + c)
  (h3 : a / b = Real.sin A / Real.sin B)
  (h4 : b / c = Real.sin B / Real.sin C)
  : c / b = 5 / 4 :=
sorry

-- Definition of Problem 2
theorem triangle_area {A B C a b c : ℝ} 
  (h1 : C = 2 * Real.pi / 3)
  (h2 : c - a = 8)
  (h3 : 2 * a * Real.cos C + 2 * c * Real.cos A = a + c)
  (h4 : a + c = 2 * b)
  : (1 / 2) * a * b * Real.sin C = 15 * Real.sqrt 3 :=
sorry

end triangle_side_ratio_triangle_area_l1018_101839


namespace percentage_decrease_l1018_101862

theorem percentage_decrease (purchase_price selling_price decrease gross_profit : ℝ)
  (h_purchase : purchase_price = 81)
  (h_markup : selling_price = purchase_price + 0.25 * selling_price)
  (h_gross_profit : gross_profit = 5.40)
  (h_decrease : decrease = 108 - 102.60) :
  (decrease / 108) * 100 = 5 :=
by sorry

end percentage_decrease_l1018_101862


namespace deputy_more_enemies_than_friends_l1018_101836

theorem deputy_more_enemies_than_friends (deputies : Type) 
  (friendship hostility indifference : deputies → deputies → Prop)
  (h_symm_friend : ∀ (a b : deputies), friendship a b → friendship b a)
  (h_symm_hostile : ∀ (a b : deputies), hostility a b → hostility b a)
  (h_symm_indiff : ∀ (a b : deputies), indifference a b → indifference b a)
  (h_enemy_exists : ∀ (d : deputies), ∃ (e : deputies), hostility d e)
  (h_principle : ∀ (a b c : deputies), hostility a b → friendship b c → hostility a c) :
  ∃ (d : deputies), ∃ (f e : ℕ), f < e :=
sorry

end deputy_more_enemies_than_friends_l1018_101836


namespace chocoBites_mod_l1018_101882

theorem chocoBites_mod (m : ℕ) (hm : m % 8 = 5) : (4 * m) % 8 = 4 :=
by
  sorry

end chocoBites_mod_l1018_101882


namespace right_triangle_x_value_l1018_101808

theorem right_triangle_x_value (BM MA BC CA: ℝ) (M_is_altitude: BM + MA = BC + CA)
  (x: ℝ) (h: ℝ) (d: ℝ) (M: BM = x) (CB: BC = h) (CA: CA = d) :
  x = (2 * h * d - d ^ 2 / 4) / (2 * d + 2 * h) := by
  sorry

end right_triangle_x_value_l1018_101808


namespace convert_spherical_to_rectangular_l1018_101893

noncomputable def spherical_to_rectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

theorem convert_spherical_to_rectangular :
  spherical_to_rectangular 15 (3 * Real.pi / 4) (Real.pi / 2) = 
    (-15 * Real.sqrt 2 / 2, 15 * Real.sqrt 2 / 2, 0) :=
by 
  sorry

end convert_spherical_to_rectangular_l1018_101893


namespace math_problem_l1018_101802

theorem math_problem :
    3 * 3^4 - (27 ^ 63 / 27 ^ 61) = -486 :=
by
  sorry

end math_problem_l1018_101802


namespace solve_for_x_l1018_101897

theorem solve_for_x (m n : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) (h_neq : m ≠ n) :
  ∃ x : ℝ, (x + 2 * m)^2 - (x - 3 * n)^2 = 9 * (m + n)^2 ↔
  x = (5 * m^2 + 18 * m * n + 18 * n^2) / (10 * m + 6 * n) := sorry

end solve_for_x_l1018_101897


namespace paula_bought_fewer_cookies_l1018_101822
-- Import the necessary libraries

-- Definitions
def paul_cookies : ℕ := 45
def total_cookies : ℕ := 87

-- Theorem statement
theorem paula_bought_fewer_cookies : ∃ (paula_cookies : ℕ), paul_cookies + paula_cookies = total_cookies ∧ paul_cookies - paula_cookies = 3 := by
  sorry

end paula_bought_fewer_cookies_l1018_101822


namespace amount_received_by_A_is_4_over_3_l1018_101826

theorem amount_received_by_A_is_4_over_3
  (a d : ℚ)
  (h1 : a - 2 * d + a - d = a + (a + d) + (a + 2 * d))
  (h2 : 5 * a = 5) :
  a - 2 * d = 4 / 3 :=
by
  sorry

end amount_received_by_A_is_4_over_3_l1018_101826


namespace problem1_l1018_101851

theorem problem1 : abs (-3) + (-1: ℤ)^2021 * (Real.pi - 3.14)^0 - (- (1/2: ℝ))⁻¹ = 4 := 
  sorry

end problem1_l1018_101851


namespace lemonade_quart_calculation_l1018_101848

-- Define the conditions
def water_parts := 5
def lemon_juice_parts := 3
def total_parts := water_parts + lemon_juice_parts

def gallons := 2
def quarts_per_gallon := 4
def total_quarts := gallons * quarts_per_gallon
def quarts_per_part := total_quarts / total_parts

-- Proof problem
theorem lemonade_quart_calculation :
  let water_quarts := water_parts * quarts_per_part
  let lemon_juice_quarts := lemon_juice_parts * quarts_per_part
  water_quarts = 5 ∧ lemon_juice_quarts = 3 :=
by
  let water_quarts := water_parts * quarts_per_part
  let lemon_juice_quarts := lemon_juice_parts * quarts_per_part
  have h_w : water_quarts = 5 := sorry
  have h_l : lemon_juice_quarts = 3 := sorry
  exact ⟨h_w, h_l⟩

end lemonade_quart_calculation_l1018_101848


namespace factor_expression_l1018_101819

theorem factor_expression (y : ℝ) : 16 * y^3 + 8 * y^2 = 8 * y^2 * (2 * y + 1) :=
by
  sorry

end factor_expression_l1018_101819


namespace remainder_of_S_mod_1000_l1018_101813

def digit_contribution (d pos : ℕ) : ℕ := (d * d) * pos

def sum_of_digits_with_no_repeats : ℕ :=
  let thousands := (16 + 25 + 36 + 49 + 64 + 81) * (9 * 8 * 7) * 1000
  let hundreds := (16 + 25 + 36 + 49 + 64 + 81) * (8 * 7 * 6) * 100
  let tens := (0 + 1 + 4 + 9 + 16 + 25 + 36 + 49 + 64 + 81 - (16 + 25 + 36 + 49 + 64 + 81)) * 6 * 5 * 10
  let units := (0 + 1 + 4 + 9 + 16 + 25 + 36 + 49 + 64 + 81 - (16 + 25 + 36 + 49 + 64 + 81)) * 6 * 5 * 1
  thousands + hundreds + tens + units

theorem remainder_of_S_mod_1000 : (sum_of_digits_with_no_repeats % 1000) = 220 :=
  by
  sorry

end remainder_of_S_mod_1000_l1018_101813


namespace calculate_correctly_l1018_101888

theorem calculate_correctly (x : ℕ) (h : 2 * x = 22) : 20 * x + 3 = 223 :=
by
  sorry

end calculate_correctly_l1018_101888


namespace distance_travelled_l1018_101810

variables (D : ℝ) (h_eq : D / 10 = (D + 20) / 14)

theorem distance_travelled : D = 50 :=
by sorry

end distance_travelled_l1018_101810


namespace smallest_n_l1018_101804

theorem smallest_n (n : ℕ) : 17 * n ≡ 136 [MOD 5] → n = 3 := 
by sorry

end smallest_n_l1018_101804


namespace total_initial_seashells_l1018_101820

-- Definitions for the conditions
def Henry_seashells := 11
def Paul_seashells := 24

noncomputable def Leo_initial_seashells (total_seashells : ℕ) :=
  (total_seashells - (Henry_seashells + Paul_seashells)) * 4 / 3

theorem total_initial_seashells 
  (total_seashells_now : ℕ)
  (leo_shared_fraction : ℕ → ℕ)
  (h : total_seashells_now = 53) : 
  Henry_seashells + Paul_seashells + leo_shared_fraction 53 = 59 :=
by
  let L := Leo_initial_seashells 53
  have L_initial : L = 24 := by sorry
  exact sorry

end total_initial_seashells_l1018_101820


namespace sum_of_consecutive_negatives_l1018_101815

theorem sum_of_consecutive_negatives (n : ℤ) (h1 : n * (n + 1) = 2720) (h2 : n < 0) : 
  n + (n + 1) = -103 :=
by
  sorry

end sum_of_consecutive_negatives_l1018_101815


namespace evaluate_seventy_five_squared_minus_twenty_five_squared_l1018_101801

theorem evaluate_seventy_five_squared_minus_twenty_five_squared :
  75^2 - 25^2 = 5000 :=
by
  sorry

end evaluate_seventy_five_squared_minus_twenty_five_squared_l1018_101801


namespace right_triangle_bc_is_3_l1018_101814

-- Define the setup: a right triangle with given side lengths
structure RightTriangle :=
  (AB AC BC : ℝ)
  (right_angle : AB^2 = AC^2 + BC^2)
  (AB_val : AB = 5)
  (AC_val : AC = 4)

-- The goal is to prove that BC = 3 given the conditions
theorem right_triangle_bc_is_3 (T : RightTriangle) : T.BC = 3 :=
  sorry

end right_triangle_bc_is_3_l1018_101814


namespace sum_of_fractions_eq_decimal_l1018_101878

theorem sum_of_fractions_eq_decimal :
  (3 / 100) + (5 / 1000) + (7 / 10000) = 0.0357 :=
by
  sorry

end sum_of_fractions_eq_decimal_l1018_101878


namespace find_b_d_l1018_101863

theorem find_b_d (b d : ℕ) (h1 : b + d = 41) (h2 : b < d) : 
  (∃! x, b * x * x + 24 * x + d = 0) → (b = 9 ∧ d = 32) :=
by 
  sorry

end find_b_d_l1018_101863


namespace devin_initial_height_l1018_101881

theorem devin_initial_height (h : ℝ) (p : ℝ) (p' : ℝ) :
  (p = 10 / 100) →
  (p' = (h - 66) / 100) →
  (h + 3 = 68) →
  (p + p' * (h + 3 - 66) = 30 / 100) →
  h = 68 :=
by
  intros hp hp' hg pt
  sorry

end devin_initial_height_l1018_101881


namespace greatest_three_digit_multiple_of_17_l1018_101838

def is_three_digit (n : Int) : Prop := (n ≥ 100 ∧ n ≤ 999)
def is_multiple_of_17 (n : Int) : Prop := (∃ k : Int, n = 17 * k)

theorem greatest_three_digit_multiple_of_17 : ∀ n : Int, is_three_digit n ∧ is_multiple_of_17 n → n ≤ 986 :=
sorry

end greatest_three_digit_multiple_of_17_l1018_101838


namespace perfect_square_trinomial_l1018_101823

theorem perfect_square_trinomial (x : ℝ) : (x + 9)^2 = x^2 + 18 * x + 81 := by
  sorry

end perfect_square_trinomial_l1018_101823


namespace smallest_s_for_347_l1018_101896

open Nat

theorem smallest_s_for_347 (r s : ℕ) (hr_pos : 0 < r) (hs_pos : 0 < s) 
  (h_rel_prime : Nat.gcd r s = 1) (h_r_lt_s : r < s) 
  (h_contains_347 : ∃ k : ℕ, ∃ y : ℕ, 10 ^ k * r - s * y = 347): 
  s = 653 := 
by sorry

end smallest_s_for_347_l1018_101896


namespace find_original_volume_l1018_101879

theorem find_original_volume
  (V : ℝ)
  (h1 : V - (3 / 4) * V = (1 / 4) * V)
  (h2 : (1 / 4) * V - (3 / 4) * ((1 / 4) * V) = (1 / 16) * V)
  (h3 : (1 / 16) * V = 0.2) :
  V = 3.2 :=
by 
  -- Proof skipped, as the assistant is instructed to provide only the statement 
  sorry

end find_original_volume_l1018_101879


namespace cos_two_thirds_pi_l1018_101883

theorem cos_two_thirds_pi : Real.cos (2 / 3 * Real.pi) = -1 / 2 :=
by sorry

end cos_two_thirds_pi_l1018_101883


namespace abs_value_difference_l1018_101834

theorem abs_value_difference (x y : ℤ) (h1 : |x| = 7) (h2 : |y| = 9) (h3 : |x + y| = -(x + y)) :
  x - y = 16 ∨ x - y = -16 :=
sorry

end abs_value_difference_l1018_101834


namespace cylinder_height_relationship_l1018_101875

variables (π r₁ r₂ h₁ h₂ : ℝ)

theorem cylinder_height_relationship
  (h_volume_eq : π * r₁^2 * h₁ = π * r₂^2 * h₂)
  (h_radius_rel : r₂ = 1.2 * r₁) :
  h₁ = 1.44 * h₂ :=
by {
  sorry -- proof not required as per instructions
}

end cylinder_height_relationship_l1018_101875


namespace Mrs_Heine_treats_l1018_101805

theorem Mrs_Heine_treats :
  let dogs := 2
  let cats := 1
  let parrots := 3
  let biscuits_per_dog := 3
  let treats_per_cat := 2
  let sticks_per_parrot := 1
  let total_treats := dogs * biscuits_per_dog + cats * treats_per_cat + parrots * sticks_per_parrot
  total_treats = 11 :=
by
  let dogs := 2
  let cats := 1
  let parrots := 3
  let biscuits_per_dog := 3
  let treats_per_cat := 2
  let sticks_per_parrot := 1
  let total_treats := dogs * biscuits_per_dog + cats * treats_per_cat + parrots * sticks_per_parrot
  show total_treats = 11
  sorry

end Mrs_Heine_treats_l1018_101805


namespace remainder_when_7n_div_by_3_l1018_101835

theorem remainder_when_7n_div_by_3 (n : ℤ) (h : n % 3 = 2) : (7 * n) % 3 = 2 := 
sorry

end remainder_when_7n_div_by_3_l1018_101835


namespace profit_amount_l1018_101824

-- Conditions: Selling Price and Profit Percentage
def SP : ℝ := 850
def P_percent : ℝ := 37.096774193548384

-- Theorem: The profit amount is $230
theorem profit_amount : (SP / (1 + P_percent / 100)) * P_percent / 100 = 230 := by
  -- sorry will be replaced with the proof
  sorry

end profit_amount_l1018_101824


namespace Matilda_fathers_chocolate_bars_l1018_101806

theorem Matilda_fathers_chocolate_bars
  (total_chocolates : ℕ) (sisters : ℕ) (chocolates_each : ℕ) (given_to_father_each : ℕ) 
  (chocolates_given : ℕ) (given_to_mother : ℕ) (eaten_by_father : ℕ) : 
  total_chocolates = 20 → 
  sisters = 4 → 
  chocolates_each = total_chocolates / (sisters + 1) → 
  given_to_father_each = chocolates_each / 2 → 
  chocolates_given = (sisters + 1) * given_to_father_each → 
  given_to_mother = 3 → 
  eaten_by_father = 2 → 
  chocolates_given - given_to_mother - eaten_by_father = 5 :=
by
  intros h_total h_sisters h_chocolates_each h_given_to_father_each h_chocolates_given h_given_to_mother h_eaten_by_father
  sorry

end Matilda_fathers_chocolate_bars_l1018_101806


namespace ratio_13_2_l1018_101850

def initial_mahogany_trees : ℕ := 50
def initial_narra_trees : ℕ := 30
def total_trees_that_fell : ℕ := 5
def current_total_trees : ℕ := 88

def number_narra_trees_that_fell (N : ℕ) : Prop := N + (N + 1) = total_trees_that_fell
def total_trees_before_typhoon : ℕ := initial_mahogany_trees + initial_narra_trees

def ratio_of_planted_trees_to_narra_fallen (planted : ℕ) (N : ℕ) : Prop := 
  88 - (total_trees_before_typhoon - total_trees_that_fell) = planted ∧ 
  planted / N = 13 / 2

theorem ratio_13_2 : ∃ (planted N : ℕ), 
  number_narra_trees_that_fell N ∧ 
  ratio_of_planted_trees_to_narra_fallen planted N :=
sorry

end ratio_13_2_l1018_101850


namespace find_cubic_sum_l1018_101845

theorem find_cubic_sum
  {a b : ℝ}
  (h1 : a^5 - a^4 * b - a^4 + a - b - 1 = 0)
  (h2 : 2 * a - 3 * b = 1) :
  a^3 + b^3 = 9 :=
by
  sorry

end find_cubic_sum_l1018_101845


namespace partitions_equal_l1018_101837

namespace MathProof

-- Define the set of natural numbers
def nat := ℕ

-- Define the partition functions (placeholders)
def num_distinct_partitions (n : nat) : nat := sorry
def num_odd_partitions (n : nat) : nat := sorry

-- Statement of the theorem
theorem partitions_equal (n : nat) : 
  num_distinct_partitions n = num_odd_partitions n :=
sorry

end MathProof

end partitions_equal_l1018_101837


namespace number_of_people_got_on_train_l1018_101829

theorem number_of_people_got_on_train (initial_people : ℕ) (people_got_off : ℕ) (final_people : ℕ) (x : ℕ) 
  (h_initial : initial_people = 78) 
  (h_got_off : people_got_off = 27) 
  (h_final : final_people = 63) 
  (h_eq : final_people = initial_people - people_got_off + x) : x = 12 :=
by 
  sorry

end number_of_people_got_on_train_l1018_101829


namespace max_n_for_neg_sum_correct_l1018_101854

noncomputable def max_n_for_neg_sum (S : ℕ → ℤ) : ℕ :=
  if h₁ : S 19 > 0 then
    if h₂ : S 20 < 0 then
      11
    else 0  -- default value
  else 0  -- default value

theorem max_n_for_neg_sum_correct (S : ℕ → ℤ) (h₁ : S 19 > 0) (h₂ : S 20 < 0) : max_n_for_neg_sum S = 11 :=
by
  sorry

end max_n_for_neg_sum_correct_l1018_101854


namespace trip_to_Atlanta_equals_Boston_l1018_101866

def distance_to_Boston : ℕ := 840
def daily_distance : ℕ := 40
def num_days (distance : ℕ) (daily : ℕ) : ℕ := distance / daily
def distance_to_Atlanta (days : ℕ) (daily : ℕ) : ℕ := days * daily

theorem trip_to_Atlanta_equals_Boston :
  distance_to_Atlanta (num_days distance_to_Boston daily_distance) daily_distance = distance_to_Boston :=
by
  -- Here we would insert the proof.
  sorry

end trip_to_Atlanta_equals_Boston_l1018_101866


namespace max_green_beads_l1018_101852

theorem max_green_beads (n : ℕ) (red blue green : ℕ) 
    (total_beads : ℕ)
    (h_total : total_beads = 100)
    (h_colors : n = red + blue + green)
    (h_blue_condition : ∀ i : ℕ, i ≤ total_beads → ∃ j, j ≤ 4 ∧ (i + j) % total_beads = blue)
    (h_red_condition : ∀ i : ℕ, i ≤ total_beads → ∃ k, k ≤ 6 ∧ (i + k) % total_beads = red) :
    green ≤ 65 :=
by
  sorry

end max_green_beads_l1018_101852


namespace mean_of_combined_sets_l1018_101857

theorem mean_of_combined_sets
  (S₁ : Finset ℝ) (S₂ : Finset ℝ)
  (h₁ : S₁.card = 7) (h₂ : S₂.card = 8)
  (mean_S₁ : (S₁.sum id) / S₁.card = 15)
  (mean_S₂ : (S₂.sum id) / S₂.card = 26)
  : (S₁.sum id + S₂.sum id) / (S₁.card + S₂.card) = 20.8667 := 
by
  sorry

end mean_of_combined_sets_l1018_101857


namespace inequality_proof_l1018_101867

variable (a b c : ℝ)

theorem inequality_proof (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (1 / (2 * a) + 1 / (2 * b) + 1 / (2 * c)) ≥ (1 / (b + c) + 1 / (c + a) + 1 / (a + b)) :=
by sorry

end inequality_proof_l1018_101867


namespace constant_term_expansion_l1018_101871

theorem constant_term_expansion (n : ℕ) (hn : n = 9) :
  y^3 * (x + 1 / (x^2 * y))^n = 84 :=
by sorry

end constant_term_expansion_l1018_101871

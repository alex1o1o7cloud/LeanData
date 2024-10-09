import Mathlib

namespace quadratic_complete_square_l125_12535

theorem quadratic_complete_square :
  ∀ x : ℝ, (x^2 - 7 * x + 6) = (x - 7 / 2) ^ 2 - 25 / 4 :=
by
  sorry

end quadratic_complete_square_l125_12535


namespace fraction_to_decimal_l125_12547

theorem fraction_to_decimal :
  (17 : ℚ) / (2^2 * 5^4) = 0.0068 :=
by
  sorry

end fraction_to_decimal_l125_12547


namespace neg_four_fifth_less_neg_two_third_l125_12593

theorem neg_four_fifth_less_neg_two_third : (-4 : ℚ) / 5 < (-2 : ℚ) / 3 :=
  sorry

end neg_four_fifth_less_neg_two_third_l125_12593


namespace train_is_late_l125_12544

theorem train_is_late (S : ℝ) (T : ℝ) (T' : ℝ) (h1 : T = 2) (h2 : T' = T * 5 / 4) :
  (T' - T) * 60 = 30 :=
by
  sorry

end train_is_late_l125_12544


namespace abc_unique_l125_12537

theorem abc_unique (n : ℕ) (hn : 0 < n) (p : ℕ) (hp : Nat.Prime p) 
                   (a b c : ℤ) 
                   (h : a^n + p * b = b^n + p * c ∧ b^n + p * c = c^n + p * a) 
                   : a = b ∧ b = c :=
by
  sorry

end abc_unique_l125_12537


namespace number_of_boys_l125_12570

-- Definitions of the conditions
def total_students : ℕ := 30
def ratio_girls_parts : ℕ := 1
def ratio_boys_parts : ℕ := 2
def total_parts : ℕ := ratio_girls_parts + ratio_boys_parts

-- Statement of the problem
theorem number_of_boys :
  ∃ (boys : ℕ), boys = (total_students / total_parts) * ratio_boys_parts ∧ boys = 20 :=
by
  sorry

end number_of_boys_l125_12570


namespace smallest_solution_eq_sqrt_104_l125_12538

theorem smallest_solution_eq_sqrt_104 :
  ∃ x : ℝ, (⌊x^2⌋ - ⌊x⌋^2 = 19) ∧ (∀ y : ℝ, ⌊y^2⌋ - ⌊y⌋^2 = 19 → x ≤ y) := sorry

end smallest_solution_eq_sqrt_104_l125_12538


namespace set_representation_l125_12530

theorem set_representation : 
  { x : ℕ | x < 5 } = {0, 1, 2, 3, 4} :=
sorry

end set_representation_l125_12530


namespace apples_given_by_nathan_l125_12508

theorem apples_given_by_nathan (initial_apples : ℕ) (total_apples : ℕ) (given_by_nathan : ℕ) :
  initial_apples = 6 → total_apples = 12 → given_by_nathan = (total_apples - initial_apples) → given_by_nathan = 6 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end apples_given_by_nathan_l125_12508


namespace part_I_part_II_l125_12579

noncomputable def general_term (a : ℕ → ℤ) (d : ℤ) : Prop :=
  (a 2 = 1 ∧ ∀ n, a (n + 1) - a n = d) ∧
  (d ≠ 0 ∧ (a 3)^2 = (a 2) * (a 6))

theorem part_I (a : ℕ → ℤ) (d : ℤ) : general_term a d → 
  ∀ n, a n = 2 * n - 3 := 
sorry

noncomputable def sum_of_first_n_terms (a : ℕ → ℤ) (d : ℤ) (S : ℕ → ℤ) : Prop :=
  (∀ n, S n = n * (a 1 + a n) / 2) ∧ 
  (general_term a d)

theorem part_II (a : ℕ → ℤ) (d : ℤ) (S : ℕ → ℤ) : sum_of_first_n_terms a d S → 
  ∃ n, n > 7 ∧ S n > 35 :=
sorry

end part_I_part_II_l125_12579


namespace tractors_planting_rate_l125_12511

theorem tractors_planting_rate (total_acres : ℕ) (total_days : ℕ) 
    (tractors_first_team : ℕ) (days_first_team : ℕ)
    (tractors_second_team : ℕ) (days_second_team : ℕ)
    (total_tractor_days : ℕ) :
    total_acres = 1700 →
    total_days = 5 →
    tractors_first_team = 2 →
    days_first_team = 2 →
    tractors_second_team = 7 →
    days_second_team = 3 →
    total_tractor_days = (tractors_first_team * days_first_team) + (tractors_second_team * days_second_team) →
    total_acres / total_tractor_days = 68 :=
by
  -- proof can be filled in later
  intros
  sorry

end tractors_planting_rate_l125_12511


namespace greatest_distance_centers_of_circles_in_rectangle_l125_12571

/--
Two circles are drawn in a 20-inch by 16-inch rectangle,
each circle with a diameter of 8 inches.
Prove that the greatest possible distance between 
the centers of the two circles without extending beyond the 
rectangular region is 4 * sqrt 13 inches.
-/
theorem greatest_distance_centers_of_circles_in_rectangle :
  let diameter := 8
  let width := 20
  let height := 16
  let radius := diameter / 2
  let reduced_width := width - 2 * radius
  let reduced_height := height - 2 * radius
  let distance := Real.sqrt ((reduced_width ^ 2) + (reduced_height ^ 2))
  distance = 4 * Real.sqrt 13 := by
    sorry

end greatest_distance_centers_of_circles_in_rectangle_l125_12571


namespace division_of_negatives_example_div_l125_12596

theorem division_of_negatives (a b : Int) (ha : a < 0) (hb : b < 0) (hb_neq : b ≠ 0) : 
  (-a) / (-b) = a / b :=
by sorry

theorem example_div : (-300) / (-50) = 6 :=
by
  apply division_of_negatives
  repeat { sorry }

end division_of_negatives_example_div_l125_12596


namespace prime_power_of_n_l125_12520

theorem prime_power_of_n (n : ℕ) (h : Nat.Prime (4^n + 2^n + 1)) : ∃ k : ℕ, n = 3^k := 
sorry

end prime_power_of_n_l125_12520


namespace evaluate_expression_l125_12559

theorem evaluate_expression : Int.ceil (7 / 3 : ℚ) + Int.floor (-7 / 3 : ℚ) = 0 := by
  -- The proof part is omitted as per instructions.
  sorry

end evaluate_expression_l125_12559


namespace rhombus_diagonals_not_always_equal_l125_12516

structure Rhombus where
  all_four_sides_equal : Prop
  symmetrical : Prop
  centrally_symmetrical : Prop

theorem rhombus_diagonals_not_always_equal (R : Rhombus) :
  ¬ (∀ (d1 d2 : ℝ), d1 = d2) :=
sorry

end rhombus_diagonals_not_always_equal_l125_12516


namespace count_president_vp_secretary_l125_12522

theorem count_president_vp_secretary (total_members boys girls : ℕ) (total_members_eq : total_members = 30) 
(boys_eq : boys = 18) (girls_eq : girls = 12) :
  ∃ (ways : ℕ), 
  ways = (boys * girls * (boys - 1) + girls * boys * (girls - 1)) ∧
  ways = 6048 :=
by
  sorry

end count_president_vp_secretary_l125_12522


namespace fraction_zero_implies_x_zero_l125_12510

theorem fraction_zero_implies_x_zero (x : ℝ) (h : (x^2 - x) / (x - 1) = 0) (h₁ : x ≠ 1) : x = 0 := by
  sorry

end fraction_zero_implies_x_zero_l125_12510


namespace consecutive_integers_exist_l125_12550

def good (n : ℕ) : Prop :=
∃ (k : ℕ) (a : ℕ → ℕ), 
  (∀ i j, 1 ≤ i → i < j → j ≤ k → a i < a j) ∧ 
  (∀ i j i' j', 1 ≤ i → i < j → j ≤ k → 1 ≤ i' → i' < j' → j' ≤ k → a i + a j = a i' + a j' → i = i' ∧ j = j') ∧ 
  (∃ (t : ℕ), ∀ m, 0 ≤ m → m < n → ∃ i j, 1 ≤ i → i < j → j ≤ k → a i + a j = t + m)

theorem consecutive_integers_exist (n : ℕ) (h : n = 1000) : good n :=
sorry

end consecutive_integers_exist_l125_12550


namespace find_intersecting_lines_l125_12506

theorem find_intersecting_lines (x y : ℝ) : 
  (2 * x - y)^2 - (x + 3 * y)^2 = 0 ↔ x = 4 * y ∨ x = - (2 / 3) * y :=
by
  sorry

end find_intersecting_lines_l125_12506


namespace sum_of_digits_inequality_l125_12541

-- Assume that S(x) represents the sum of the digits of x in its decimal representation.
axiom sum_of_digits (x : ℕ) : ℕ

-- Given condition: for any natural numbers a and b, the sum of digits function satisfies the inequality
axiom sum_of_digits_add (a b : ℕ) : sum_of_digits (a + b) ≤ sum_of_digits a + sum_of_digits b

-- Theorem statement we want to prove
theorem sum_of_digits_inequality (k : ℕ) : sum_of_digits k ≤ 8 * sum_of_digits (8 * k) := 
  sorry

end sum_of_digits_inequality_l125_12541


namespace least_third_side_length_l125_12589

theorem least_third_side_length (a b : ℕ) (h_a : a = 8) (h_b : b = 15) : 
  ∃ c : ℝ, (c = Real.sqrt (a^2 + b^2) ∨ c = Real.sqrt (b^2 - a^2)) ∧ c = Real.sqrt 161 :=
by
  sorry

end least_third_side_length_l125_12589


namespace advertising_time_l125_12567

-- Define the conditions
def total_duration : ℕ := 30
def national_news : ℕ := 12
def international_news : ℕ := 5
def sports : ℕ := 5
def weather_forecasts : ℕ := 2

-- Calculate total content time
def total_content_time : ℕ := national_news + international_news + sports + weather_forecasts

-- Define the proof problem
theorem advertising_time (h : total_duration - total_content_time = 6) : (total_duration - total_content_time) = 6 :=
by
sorry

end advertising_time_l125_12567


namespace odds_against_C_l125_12532

def odds_against_winning (p : ℚ) : ℚ := (1 - p) / p

theorem odds_against_C (pA pB pC : ℚ) (hA : pA = 1 / 3) (hB : pB = 1 / 5) (hC : pC = 7 / 15) :
  odds_against_winning pC = 8 / 7 :=
by
  -- Definitions based on the conditions provided in a)
  have h1 : odds_against_winning (1/3) = 2 := by sorry
  have h2 : odds_against_winning (1/5) = 4 := by sorry

  -- Odds against C
  have h3 : 1 - (pA + pB) = pC := by sorry
  have h4 : pA + pB = 8 / 15 := by sorry

  -- Show that odds against C winning is 8/7
  have h5 : odds_against_winning pC = 8 / 7 := by sorry
  exact h5

end odds_against_C_l125_12532


namespace investment_time_ratio_l125_12531

theorem investment_time_ratio (x t : ℕ) (h_inv : 7 * x = t * 5) (h_prof_ratio : 7 / 10 = 70 / (5 * t)) : 
  t = 20 := sorry

end investment_time_ratio_l125_12531


namespace sum_of_squares_of_non_zero_digits_from_10_to_99_l125_12563

-- Definition of the sum of squares of digits from 1 to 9
def P : ℕ := (1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2 + 7^2 + 8^2 + 9^2)

-- Definition of the sum of squares of the non-zero digits of the integers from 10 to 99
def T : ℕ := 20 * P

-- Theorem stating that T equals 5700
theorem sum_of_squares_of_non_zero_digits_from_10_to_99 : T = 5700 :=
by
  sorry

end sum_of_squares_of_non_zero_digits_from_10_to_99_l125_12563


namespace largest_amount_received_back_l125_12505

theorem largest_amount_received_back 
  (x y x_lost y_lost : ℕ) 
  (h1 : 20 * x + 100 * y = 3000) 
  (h2 : x_lost + y_lost = 16) 
  (h3 : x_lost = y_lost + 2 ∨ x_lost = y_lost - 2) 
  : (3000 - (20 * x_lost + 100 * y_lost) = 2120) :=
sorry

end largest_amount_received_back_l125_12505


namespace connie_total_markers_l125_12572

theorem connie_total_markers (red_markers : ℕ) (blue_markers : ℕ) 
                              (h1 : red_markers = 41)
                              (h2 : blue_markers = 64) : 
                              red_markers + blue_markers = 105 := by
  sorry

end connie_total_markers_l125_12572


namespace sum_factors_of_30_l125_12586

theorem sum_factors_of_30 : (1 + 2 + 3 + 5 + 6 + 10 + 15 + 30) = 72 :=
by
  sorry

end sum_factors_of_30_l125_12586


namespace xy_zero_l125_12576

theorem xy_zero (x y : ℝ) (h1 : x - y = 6) (h2 : x^3 - y^3 = 162) : x * y = 0 :=
by
  sorry

end xy_zero_l125_12576


namespace ratio_sum_l125_12580

variable (x y z : ℝ)

-- Conditions
axiom geometric_sequence : 16 * y^2 = 15 * x * z
axiom arithmetic_sequence : 2 / y = 1 / x + 1 / z

-- Theorem to prove
theorem ratio_sum : x ≠ 0 → y ≠ 0 → z ≠ 0 → 
  (16 * y^2 = 15 * x * z) → (2 / y = 1 / x + 1 / z) → (x / z + z / x = 34 / 15) :=
by
  -- proof goes here
  sorry

end ratio_sum_l125_12580


namespace lines_through_origin_l125_12564

-- Define that a, b, c are in geometric progression
def geo_prog (a b c : ℝ) : Prop :=
  ∃ r : ℝ, b = a * r ∧ c = a * r^2

-- Define the property of the line passing through the common point (0, 0)
def passes_through_origin (a b c : ℝ) : Prop :=
  ∀ x y, (a * x + b * y = c) → (x = 0 ∧ y = 0)

theorem lines_through_origin (a b c : ℝ) (h : geo_prog a b c) : passes_through_origin a b c :=
by
  sorry

end lines_through_origin_l125_12564


namespace water_wasted_in_one_hour_l125_12584

theorem water_wasted_in_one_hour:
  let drips_per_minute : ℕ := 10
  let drop_volume : ℝ := 0.05 -- volume in mL
  let minutes_in_hour : ℕ := 60
  drips_per_minute * drop_volume * minutes_in_hour = 30 := by
  sorry

end water_wasted_in_one_hour_l125_12584


namespace quadratic_distinct_real_roots_l125_12588

theorem quadratic_distinct_real_roots (m : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 - 4*x1 + m - 1 = 0) ∧ (x2^2 - 4*x2 + m - 1 = 0)) → m < 5 := sorry

end quadratic_distinct_real_roots_l125_12588


namespace find_value_of_x_squared_plus_y_squared_l125_12553

theorem find_value_of_x_squared_plus_y_squared (x y : ℝ) (h : (x^2 + y^2 + 1)^2 - 4 = 0) : x^2 + y^2 = 1 :=
by
  sorry

end find_value_of_x_squared_plus_y_squared_l125_12553


namespace mr_zander_total_payment_l125_12566

noncomputable def total_cost (cement_bags : ℕ) (price_per_bag : ℝ) (sand_lorries : ℕ) 
(tons_per_lorry : ℝ) (price_per_ton : ℝ) (discount_rate : ℝ) (tax_rate : ℝ) : ℝ :=
  let cement_cost_before_discount := cement_bags * price_per_bag
  let discount := cement_cost_before_discount * discount_rate
  let cement_cost_after_discount := cement_cost_before_discount - discount
  let sand_cost_before_tax := sand_lorries * tons_per_lorry * price_per_ton
  let tax := sand_cost_before_tax * tax_rate
  let sand_cost_after_tax := sand_cost_before_tax + tax
  cement_cost_after_discount + sand_cost_after_tax

theorem mr_zander_total_payment :
  total_cost 500 10 20 10 40 0.05 0.07 = 13310 := 
sorry

end mr_zander_total_payment_l125_12566


namespace tetrahedron_formable_l125_12524

theorem tetrahedron_formable (x : ℝ) (hx_pos : 0 < x) (hx_bound : x < (Real.sqrt 6 + Real.sqrt 2) / 2) :
  true := 
sorry

end tetrahedron_formable_l125_12524


namespace no_solution_part_a_no_solution_part_b_l125_12568

theorem no_solution_part_a 
  (x y z : ℕ) :
  ¬(x^2 + y^2 + z^2 = 2 * x * y * z) := 
sorry

theorem no_solution_part_b 
  (x y z u : ℕ) :
  ¬(x^2 + y^2 + z^2 + u^2 = 2 * x * y * z * u) := 
sorry

end no_solution_part_a_no_solution_part_b_l125_12568


namespace quadratic_inequality_solution_l125_12509

theorem quadratic_inequality_solution :
  {x : ℝ | 3 * x^2 + 5 * x < 8} = {x : ℝ | -4 < x ∧ x < 2 / 3} :=
sorry

end quadratic_inequality_solution_l125_12509


namespace students_taking_neither_l125_12545

theorem students_taking_neither (total_students music_students art_students dance_students music_art music_dance art_dance music_art_dance : ℕ) :
  total_students = 2500 →
  music_students = 200 →
  art_students = 150 →
  dance_students = 100 →
  music_art = 75 →
  art_dance = 50 →
  music_dance = 40 →
  music_art_dance = 25 →
  total_students - ((music_students + art_students + dance_students) - (music_art + art_dance + music_dance) + music_art_dance) = 2190 :=
by
  intros
  sorry

end students_taking_neither_l125_12545


namespace fourth_leg_length_l125_12540

theorem fourth_leg_length (a b c : ℕ) (h₁ : a = 8) (h₂ : b = 9) (h₃ : c = 10) :
  (∃ x : ℕ, x ≠ a ∧ x ≠ b ∧ x ≠ c ∧ (a + x = b + c ∨ b + x = a + c ∨ c + x = a + b) ∧ (x = 7 ∨ x = 11)) :=
by sorry

end fourth_leg_length_l125_12540


namespace haley_total_expenditure_l125_12527

-- Definition of conditions
def ticket_cost : ℕ := 4
def tickets_bought_for_self_and_friends : ℕ := 3
def tickets_bought_for_others : ℕ := 5
def total_tickets : ℕ := tickets_bought_for_self_and_friends + tickets_bought_for_others

-- Proof statement
theorem haley_total_expenditure : total_tickets * ticket_cost = 32 := by
  sorry

end haley_total_expenditure_l125_12527


namespace smallest_area_2020th_square_l125_12512

theorem smallest_area_2020th_square (n : ℕ) :
  (∃ n : ℕ, n^2 > 2019 ∧ ∃ A : ℕ, A = n^2 - 2019 ∧ A ≠ 1) →
  (∃ A : ℕ, A = n^2 - 2019 ∧ A ≠ 1 ∧ A = 6) :=
sorry

end smallest_area_2020th_square_l125_12512


namespace trouser_sale_price_l125_12599

theorem trouser_sale_price 
  (original_price : ℝ) 
  (percent_decrease : ℝ) 
  (sale_price : ℝ) 
  (h : original_price = 100) 
  (p : percent_decrease = 0.25) 
  (s : sale_price = original_price * (1 - percent_decrease)) : 
  sale_price = 75 :=
by 
  sorry

end trouser_sale_price_l125_12599


namespace problem_solution_l125_12582

theorem problem_solution (x1 x2 x3 : ℝ) (h1: x1 < x2) (h2: x2 < x3)
(h3 : 10 * x1^3 - 201 * x1^2 + 3 = 0)
(h4 : 10 * x2^3 - 201 * x2^2 + 3 = 0)
(h5 : 10 * x3^3 - 201 * x3^2 + 3 = 0) :
x2 * (x1 + x3) = 398 :=
sorry

end problem_solution_l125_12582


namespace max_value_of_f_l125_12533

noncomputable def f (x : ℝ) : ℝ := 3 * x^3 - 18 * x^2 + 27 * x

theorem max_value_of_f (x : ℝ) (h : 0 ≤ x) : ∃ M, M = 12 ∧ ∀ y, 0 ≤ y → f y ≤ M :=
sorry

end max_value_of_f_l125_12533


namespace malingerers_exposed_l125_12503

theorem malingerers_exposed (a b c : Nat) (ha : a > b) (hc : c = b + 9) :
  let aabbb := 10000 * a + 1000 * a + 100 * b + 10 * b + b
  let abccc := 10000 * a + 1000 * b + 100 * c + 10 * c + c
  (aabbb - 1 = abccc) -> abccc = 10999 :=
by
  sorry

end malingerers_exposed_l125_12503


namespace inscribed_circle_radius_l125_12513

noncomputable def radius_inscribed_circle (O1 O2 D : ℝ × ℝ) (r1 r2 : ℝ) :=
  if (r1 = 2 ∧ r2 = 6) ∧ ((O1.fst - O2.fst)^2 + (O1.snd - O2.snd)^2 = 64) then
    2 * (Real.sqrt 3 - 1)
  else
    0

theorem inscribed_circle_radius (O1 O2 D : ℝ × ℝ) (r1 r2 : ℝ)
  (h1 : r1 = 2) (h2 : r2 = 6)
  (h3 : (O1.fst - O2.fst)^2 + (O1.snd - O2.snd)^2 = 64) :
  radius_inscribed_circle O1 O2 D r1 r2 = 2 * (Real.sqrt 3 - 1) :=
by
  sorry

end inscribed_circle_radius_l125_12513


namespace min_colors_needed_correct_l125_12554

-- Define the 5x5 grid as a type
def Grid : Type := Fin 5 × Fin 5

-- Define a coloring as a function from Grid to a given number of colors
def Coloring (colors : Type) : Type := Grid → colors

-- Define the property where in any row, column, or diagonal, no three consecutive cells have the same color
def valid_coloring (colors : Type) (C : Coloring colors) : Prop :=
  ∀ i : Fin 5, ∀ j : Fin 3, ( C (i, j) ≠ C (i, j + 1) ∧ C (i, j + 1) ≠ C (i, j + 2) ) ∧
  ∀ i : Fin 3, ∀ j : Fin 5, ( C (i, j) ≠ C (i + 1, j) ∧ C (i + 1, j) ≠ C (i + 2, j) ) ∧
  ∀ i : Fin 3, ∀ j : Fin 3, ( C (i, j) ≠ C (i + 1, j + 1) ∧ C (i + 1, j + 1) ≠ C (i + 2, j + 2) )

-- Define the minimum number of colors required
def min_colors_needed : Nat := 5

-- Prove the statement
theorem min_colors_needed_correct : ∃ C : Coloring (Fin min_colors_needed), valid_coloring (Fin min_colors_needed) C :=
sorry

end min_colors_needed_correct_l125_12554


namespace replaced_person_weight_l125_12592

theorem replaced_person_weight :
  ∀ (old_avg_weight new_person_weight incr_weight : ℕ),
    old_avg_weight * 8 + incr_weight = new_person_weight →
    incr_weight = 16 →
    new_person_weight = 81 →
    (old_avg_weight - (new_person_weight - incr_weight) / 8) = 65 :=
by
  intros old_avg_weight new_person_weight incr_weight h1 h2 h3
  -- TODO: Proof goes here
  sorry

end replaced_person_weight_l125_12592


namespace gretchen_fewest_trips_l125_12552

def fewestTrips (total_objects : ℕ) (max_carry : ℕ) : ℕ :=
  (total_objects + max_carry - 1) / max_carry

theorem gretchen_fewest_trips : fewestTrips 17 3 = 6 := 
  sorry

end gretchen_fewest_trips_l125_12552


namespace scientific_notation_of_1650000_l125_12536

theorem scientific_notation_of_1650000 : (1650000 : ℝ) = 1.65 * 10^6 := 
by {
  -- Proof goes here
  sorry
}

end scientific_notation_of_1650000_l125_12536


namespace total_amount_received_l125_12502

theorem total_amount_received (h1 : 12 = 12)
                              (h2 : 10 = 10)
                              (h3 : 8 = 8)
                              (h4 : 14 = 14)
                              (rate : 15 = 15) :
  (3 * (12 + 10 + 8 + 14) * 15) = 1980 :=
by sorry

end total_amount_received_l125_12502


namespace total_number_of_animals_is_650_l125_12528

def snake_count : Nat := 100
def arctic_fox_count : Nat := 80
def leopard_count : Nat := 20
def bee_eater_count : Nat := 10 * leopard_count
def cheetah_count : Nat := snake_count / 2
def alligator_count : Nat := 2 * (arctic_fox_count + leopard_count)

def total_animal_count : Nat :=
  snake_count + arctic_fox_count + leopard_count + bee_eater_count + cheetah_count + alligator_count

theorem total_number_of_animals_is_650 :
  total_animal_count = 650 :=
by
  sorry

end total_number_of_animals_is_650_l125_12528


namespace floor_T_equals_150_l125_12549

variable {p q r s : ℝ}

theorem floor_T_equals_150
  (hpq_sum_of_squares : p^2 + q^2 = 2500)
  (hrs_sum_of_squares : r^2 + s^2 = 2500)
  (hpq_product : p * q = 1225)
  (hrs_product : r * s = 1225)
  (hp_plus_s : p + s = 75) :
  ∃ T : ℝ, T = p + q + r + s ∧ ⌊T⌋ = 150 :=
by
  sorry

end floor_T_equals_150_l125_12549


namespace profit_15000_l125_12534

theorem profit_15000
  (P : ℝ)
  (invest_mary : ℝ := 550)
  (invest_mike : ℝ := 450)
  (total_invest := invest_mary + invest_mike)
  (share_ratio_mary := invest_mary / total_invest)
  (share_ratio_mike := invest_mike / total_invest)
  (effort_share := P / 6)
  (invest_share_mary := share_ratio_mary * (2 * P / 3))
  (invest_share_mike := share_ratio_mike * (2 * P / 3))
  (mary_total := effort_share + invest_share_mary)
  (mike_total := effort_share + invest_share_mike)
  (condition : mary_total - mike_total = 1000) :
  P = 15000 :=  
sorry

end profit_15000_l125_12534


namespace sampling_is_stratified_l125_12501

-- Given Conditions
def number_of_male_students := 500
def number_of_female_students := 400
def sampled_male_students := 25
def sampled_female_students := 20

-- Definition of stratified sampling according to the problem context
def is_stratified_sampling (N_M F_M R_M R_F : ℕ) : Prop :=
  (R_M > 0 ∧ R_F > 0 ∧ R_M < N_M ∧ R_F < N_M ∧ N_M > 0 ∧ N_M > 0)

-- Proving that the sampling method is stratified sampling
theorem sampling_is_stratified : 
  is_stratified_sampling number_of_male_students number_of_female_students sampled_male_students sampled_female_students = true :=
by
  sorry

end sampling_is_stratified_l125_12501


namespace find_fraction_value_l125_12581

variable {x y : ℂ}

theorem find_fraction_value
    (h1 : (x^2 + y^2) / (x + y) = 4)
    (h2 : (x^4 + y^4) / (x^3 + y^3) = 2) :
    (x^6 + y^6) / (x^5 + y^5) = 4 := by
  sorry

end find_fraction_value_l125_12581


namespace product_of_roots_eq_neg30_l125_12546

theorem product_of_roots_eq_neg30 (x : ℝ) (h : (x + 3) * (x - 4) = 18) : 
  (∃ (a b : ℝ), (x = a ∨ x = b) ∧ a * b = -30) :=
sorry

end product_of_roots_eq_neg30_l125_12546


namespace proof_equivalence_l125_12507

variables {a b c d e f : Prop}

theorem proof_equivalence (h₁ : (a ≥ b) → (c > d)) 
                        (h₂ : (c > d) → (a ≥ b)) 
                        (h₃ : (a < b) ↔ (e ≤ f)) :
  (c ≤ d) ↔ (e ≤ f) :=
sorry

end proof_equivalence_l125_12507


namespace rahul_work_days_l125_12597

theorem rahul_work_days
  (R : ℕ)
  (Rajesh_days : ℕ := 2)
  (total_payment : ℕ := 170)
  (rahul_share : ℕ := 68)
  (combined_work_rate : ℚ := 1) :
  (∃ R : ℕ, (1 / (R : ℚ) + 1 / (Rajesh_days : ℚ) = combined_work_rate) ∧ (68 / (total_payment - rahul_share) = 2 / R) ∧ R = 3) :=
sorry

end rahul_work_days_l125_12597


namespace all_elements_rational_l125_12548

open Set

def finite_set_in_interval (n : ℕ) : Set ℝ :=
  {x | ∃ i, i ∈ Finset.range (n + 1) ∧ (x = 0 ∨ x = 1 ∨ 0 < x ∧ x < 1)}

def unique_distance_condition (S : Set ℝ) : Prop :=
  ∀ d, d ≠ 1 → ∃ x_i x_j x_k x_l, x_i ∈ S ∧ x_j ∈ S ∧ x_k ∈ S ∧ x_l ∈ S ∧ 
        abs (x_i - x_j) = d ∧ abs (x_k - x_l) = d ∧ (x_i = x_k → x_j ≠ x_l)

theorem all_elements_rational
  (n : ℕ)
  (S : Set ℝ)
  (hS1 : ∀ x ∈ S, 0 ≤ x ∧ x ≤ 1)
  (hS2 : 0 ∈ S)
  (hS3 : 1 ∈ S)
  (hS4 : unique_distance_condition S) :
  ∀ x ∈ S, ∃ q : ℚ, (x : ℝ) = q := 
sorry

end all_elements_rational_l125_12548


namespace function_machine_output_is_17_l125_12557

def functionMachineOutput (input : ℕ) : ℕ :=
  let step1 := input * 3
  if step1 <= 22 then step1 + 10 else step1 - 7

theorem function_machine_output_is_17 : functionMachineOutput 8 = 17 := by
  sorry

end function_machine_output_is_17_l125_12557


namespace find_y_intercept_l125_12591

theorem find_y_intercept (m x y b : ℤ) (h_slope : m = 2) (h_point : (x, y) = (259, 520)) :
  y = m * x + b → b = 2 :=
by {
  sorry
}

end find_y_intercept_l125_12591


namespace magnitude_of_z_l125_12578

namespace ComplexNumberProof

open Complex

noncomputable def z (b : ℝ) : ℂ := (3 - b * Complex.I) / Complex.I

theorem magnitude_of_z (b : ℝ) (h : (z b).re = (z b).im) : Complex.abs (z b) = 3 * Real.sqrt 2 :=
by
  sorry

end ComplexNumberProof

end magnitude_of_z_l125_12578


namespace quadratic_equation_root_conditions_quadratic_equation_distinct_real_roots_l125_12574

theorem quadratic_equation_root_conditions
  (k : ℝ)
  (h_discriminant : 4 * k - 3 > 0)
  (h_sum_product : ∀ (x1 x2 : ℝ),
    x1 + x2 = -(2 * k + 1) ∧ 
    x1 * x2 = k^2 + 1 →
    x1 + x2 + 2 * (x1 * x2) = 1) :
  k = 1 :=
by
  sorry

theorem quadratic_equation_distinct_real_roots
  (k : ℝ) :
  (∃ (x1 x2 : ℝ),
    x1 ≠ x2 ∧
    x1^2 + (2 * k + 1) * x1 + (k^2 + 1) = 0 ∧
    x2^2 + (2 * k + 1) * x2 + (k^2 + 1) = 0) ↔
  k > 3 / 4 :=
by
  sorry

end quadratic_equation_root_conditions_quadratic_equation_distinct_real_roots_l125_12574


namespace intersecting_lines_l125_12598

theorem intersecting_lines (c d : ℝ) 
  (h1 : 3 = (1/3 : ℝ) * 0 + c)
  (h2 : 0 = (1/3 : ℝ) * 3 + d) :
  c + d = 2 := 
by {
  sorry
}

end intersecting_lines_l125_12598


namespace radius_scientific_notation_l125_12560

theorem radius_scientific_notation :
  696000 = 6.96 * 10^5 :=
sorry

end radius_scientific_notation_l125_12560


namespace largest_fraction_is_D_l125_12562

-- Define the fractions as Lean variables
def A : ℚ := 2 / 6
def B : ℚ := 3 / 8
def C : ℚ := 4 / 12
def D : ℚ := 7 / 16
def E : ℚ := 9 / 24

-- Define a theorem to prove the largest fraction is D
theorem largest_fraction_is_D : max (max (max A B) (max C D)) E = D :=
by
  sorry

end largest_fraction_is_D_l125_12562


namespace parking_space_length_l125_12525

theorem parking_space_length {L W : ℕ} 
  (h1 : 2 * W + L = 37) 
  (h2 : L * W = 126) : 
  L = 9 := 
sorry

end parking_space_length_l125_12525


namespace find_halls_per_floor_l125_12555

theorem find_halls_per_floor
  (H : ℤ)
  (floors_first_wing : ℤ := 9)
  (rooms_per_hall_first_wing : ℤ := 32)
  (floors_second_wing : ℤ := 7)
  (halls_per_floor_second_wing : ℤ := 9)
  (rooms_per_hall_second_wing : ℤ := 40)
  (total_rooms : ℤ := 4248) :
  9 * H * 32 + 7 * 9 * 40 = 4248 → H = 6 :=
by
  sorry

end find_halls_per_floor_l125_12555


namespace highest_prob_of_red_card_l125_12514

theorem highest_prob_of_red_card :
  let deck_size := 52
  let num_aces := 4
  let num_hearts := 13
  let num_kings := 4
  let num_reds := 26
  -- Event probabilities
  let prob_ace := num_aces / deck_size
  let prob_heart := num_hearts / deck_size
  let prob_king := num_kings / deck_size
  let prob_red := num_reds / deck_size
  prob_red > prob_heart ∧ prob_heart > prob_ace ∧ prob_ace = prob_king :=
sorry

end highest_prob_of_red_card_l125_12514


namespace problem_statement_l125_12521

theorem problem_statement (a b : ℝ) (h : a^2 > b^2) : a > b → a > 0 :=
sorry

end problem_statement_l125_12521


namespace cubic_sum_identity_l125_12515

theorem cubic_sum_identity (a b c : ℝ) (h1 : a + b + c = 1) (h2 : a * b + a * c + b * c = -6) (h3 : a * b * c = -3) :
  a^3 + b^3 + c^3 = 27 :=
by
  sorry

end cubic_sum_identity_l125_12515


namespace linear_equation_a_zero_l125_12504

theorem linear_equation_a_zero (a : ℝ) : 
  ((a - 2) * x ^ (abs (a - 1)) + 3 = 9) ∧ (abs (a - 1) = 1) → a = 0 := by
  sorry

end linear_equation_a_zero_l125_12504


namespace cities_real_distance_l125_12561

def map_scale := 7 -- number of centimeters representing 35 kilometers
def real_distance_equiv := 35 -- number of kilometers that corresponds to map_scale

def centimeters_per_kilometer := real_distance_equiv / map_scale -- kilometers per centimeter

def distance_on_map := 49 -- number of centimeters cities are separated by on the map

theorem cities_real_distance : distance_on_map * centimeters_per_kilometer = 245 :=
by
  sorry

end cities_real_distance_l125_12561


namespace geometric_seq_tenth_term_l125_12558

theorem geometric_seq_tenth_term :
  let a := 12
  let r := (1 / 2 : ℝ)
  (a * r^9) = (3 / 128 : ℝ) :=
by
  let a := 12
  let r := (1 / 2 : ℝ)
  show a * r^9 = 3 / 128
  sorry

end geometric_seq_tenth_term_l125_12558


namespace tangent_intersection_product_l125_12583

theorem tangent_intersection_product (R r : ℝ) (A B C : ℝ) :
  (AC * CB = R * r) :=
sorry

end tangent_intersection_product_l125_12583


namespace inequality_m_2n_l125_12526

noncomputable def f (x : ℝ) : ℝ := abs (x - 1) - 2 * abs (x + 1)

lemma max_f : ∃ x : ℝ, f x = 2 :=
sorry

theorem inequality_m_2n (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : 1/m + 1/(2*n) = 2) : m + 2*n ≥ 2 :=
sorry

end inequality_m_2n_l125_12526


namespace courier_speeds_correctness_l125_12551

noncomputable def courier_speeds : Prop :=
  ∃ (s1 s2 : ℕ), (s1 * 8 + s2 * 8 = 176) ∧ (s1 = 60 / 5) ∧ (s2 = 60 / 6)

theorem courier_speeds_correctness : courier_speeds :=
by
  sorry

end courier_speeds_correctness_l125_12551


namespace omega_range_l125_12573

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x)

theorem omega_range (ω : ℝ) (h1 : ω > 0)
  (h2 : ∀ x ∈ Set.Icc (-Real.pi / 3) (Real.pi / 4), f ω x ≥ -2) :
  0 < ω ∧ ω ≤ 3 / 2 :=
by
  sorry

end omega_range_l125_12573


namespace felix_trees_chopped_l125_12569

-- Given conditions
def cost_per_sharpening : ℕ := 8
def total_spent : ℕ := 48
def trees_per_sharpening : ℕ := 25

-- Lean statement of the problem
theorem felix_trees_chopped (h : total_spent / cost_per_sharpening * trees_per_sharpening >= 150) : True :=
by {
  -- This is just a placeholder for the proof.
  sorry
}

end felix_trees_chopped_l125_12569


namespace number_of_houses_built_l125_12542

def original_houses : ℕ := 20817
def current_houses : ℕ := 118558
def houses_built : ℕ := current_houses - original_houses

theorem number_of_houses_built :
  houses_built = 97741 := by
  sorry

end number_of_houses_built_l125_12542


namespace sum_of_possible_values_of_N_l125_12577

theorem sum_of_possible_values_of_N :
  (∃ N : ℝ, N * (N - 7) = 12) → (∃ N₁ N₂ : ℝ, (N₁ * (N₁ - 7) = 12 ∧ N₂ * (N₂ - 7) = 12) ∧ N₁ + N₂ = 7) :=
by
  sorry

end sum_of_possible_values_of_N_l125_12577


namespace exist_interval_l125_12543

noncomputable def f (x : ℝ) := Real.log x + x - 4

theorem exist_interval (x₀ : ℝ) (h₀ : f x₀ = 0) : 2 < x₀ ∧ x₀ < 3 :=
by
  sorry

end exist_interval_l125_12543


namespace kite_area_overlap_l125_12517

theorem kite_area_overlap (beta : Real) (h_beta : beta ≠ 0 ∧ beta ≠ π) : 
  ∃ (A : Real), A = 1 / Real.sin beta := by
  sorry

end kite_area_overlap_l125_12517


namespace number_of_people_and_price_l125_12500

theorem number_of_people_and_price 
  (x y : ℤ) 
  (h1 : 8 * x - y = 3) 
  (h2 : y - 7 * x = 4) : 
  x = 7 ∧ y = 53 :=
by
  sorry

end number_of_people_and_price_l125_12500


namespace andrea_needs_to_buy_sod_squares_l125_12519

theorem andrea_needs_to_buy_sod_squares :
  let area_section1 := 30 * 40
  let area_section2 := 60 * 80
  let total_area := area_section1 + area_section2
  let area_of_sod_square := 2 * 2
  1500 = total_area / area_of_sod_square :=
by
  let area_section1 := 30 * 40
  let area_section2 := 60 * 80
  let total_area := area_section1 + area_section2
  let area_of_sod_square := 2 * 2
  sorry

end andrea_needs_to_buy_sod_squares_l125_12519


namespace nigel_gave_away_l125_12529

theorem nigel_gave_away :
  ∀ (original : ℕ) (gift_from_mother : ℕ) (final : ℕ) (money_given_away : ℕ),
    original = 45 →
    gift_from_mother = 80 →
    final = 2 * original + 10 →
    final = original - money_given_away + gift_from_mother →
    money_given_away = 25 :=
by
  intros original gift_from_mother final money_given_away
  sorry

end nigel_gave_away_l125_12529


namespace units_digit_of_power_17_l125_12590

theorem units_digit_of_power_17 (n : ℕ) (k : ℕ) (h_n4 : n % 4 = 3) : (17^n) % 10 = 3 :=
  by
  -- Since units digits of powers repeat every 4
  sorry

-- Specific problem instance
example : (17^1995) % 10 = 3 := units_digit_of_power_17 1995 17 (by norm_num)

end units_digit_of_power_17_l125_12590


namespace students_taking_music_l125_12518

theorem students_taking_music
  (total_students : Nat)
  (students_taking_art : Nat)
  (students_taking_both : Nat)
  (students_taking_neither : Nat)
  (total_eq : total_students = 500)
  (art_eq : students_taking_art = 20)
  (both_eq : students_taking_both = 10)
  (neither_eq : students_taking_neither = 440) :
  ∃ M : Nat, M = 50 := by
  sorry

end students_taking_music_l125_12518


namespace at_least_one_le_one_l125_12575

theorem at_least_one_le_one (x y z : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z) (h_sum : x + y + z = 3) : 
  x * (x + y - z) ≤ 1 ∨ y * (y + z - x) ≤ 1 ∨ z * (z + x - y) ≤ 1 :=
sorry

end at_least_one_le_one_l125_12575


namespace andrea_still_needs_rhinestones_l125_12594

def total_rhinestones_needed : ℕ := 45
def rhinestones_bought : ℕ := total_rhinestones_needed / 3
def rhinestones_found : ℕ := total_rhinestones_needed / 5
def rhinestones_total_have : ℕ := rhinestones_bought + rhinestones_found
def rhinestones_still_needed : ℕ := total_rhinestones_needed - rhinestones_total_have

theorem andrea_still_needs_rhinestones : rhinestones_still_needed = 21 := by
  rfl

end andrea_still_needs_rhinestones_l125_12594


namespace sum_of_squares_of_roots_l125_12585

theorem sum_of_squares_of_roots :
  (∃ (x₁ x₂ : ℝ), 5 * x₁^2 + 3 * x₁ - 7 = 0 ∧ 5 * x₂^2 + 3 * x₂ - 7 = 0 ∧ x₁ ≠ x₂) →
  (∃ (x₁ x₂ : ℝ), 5 * x₁^2 + 3 * x₁ - 7 = 0 ∧ 5 * x₂^2 + 3 * x₂ - 7 = 0 ∧ x₁ ≠ x₂ ∧ x₁^2 + x₂^2 = 79 / 25) :=
by
  sorry

end sum_of_squares_of_roots_l125_12585


namespace winnie_keeps_balloons_l125_12587

theorem winnie_keeps_balloons (red white green chartreuse friends total remainder : ℕ) (hRed : red = 17) (hWhite : white = 33) (hGreen : green = 65) (hChartreuse : chartreuse = 83) (hFriends : friends = 10) (hTotal : total = red + white + green + chartreuse) (hDiv : total % friends = remainder) : remainder = 8 :=
by
  have hTotal_eq : total = 198 := by
    sorry -- This would be the computation of 17 + 33 + 65 + 83
  have hRemainder_eq : 198 % 10 = remainder := by
    sorry -- This would involve the computation of the remainder
  exact sorry -- This would be the final proof that remainder = 8, tying all parts together

end winnie_keeps_balloons_l125_12587


namespace basketball_club_members_l125_12556

theorem basketball_club_members :
  let sock_cost := 6
  let tshirt_additional_cost := 8
  let total_cost := 4440
  let cost_per_member := sock_cost + 2 * (sock_cost + tshirt_additional_cost)
  total_cost / cost_per_member = 130 :=
by
  sorry

end basketball_club_members_l125_12556


namespace fruit_problem_l125_12565

theorem fruit_problem :
  let apples_initial := 7
  let oranges_initial := 8
  let mangoes_initial := 15
  let grapes_initial := 12
  let strawberries_initial := 5
  let apples_taken := 3
  let oranges_taken := 4
  let mangoes_taken := 4
  let grapes_taken := 7
  let strawberries_taken := 3
  let apples_remaining := apples_initial - apples_taken
  let oranges_remaining := oranges_initial - oranges_taken
  let mangoes_remaining := mangoes_initial - mangoes_taken
  let grapes_remaining := grapes_initial - grapes_taken
  let strawberries_remaining := strawberries_initial - strawberries_taken
  let total_remaining := apples_remaining + oranges_remaining + mangoes_remaining + grapes_remaining + strawberries_remaining
  let total_taken := apples_taken + oranges_taken + mangoes_taken + grapes_taken + strawberries_taken
  total_remaining = 26 ∧ total_taken = 21 := by
    sorry

end fruit_problem_l125_12565


namespace gcd_lcm_product_l125_12595

theorem gcd_lcm_product (a b: ℕ) (h1 : a = 36) (h2 : b = 210) :
  Nat.gcd a b * Nat.lcm a b = 7560 := 
by 
  sorry

end gcd_lcm_product_l125_12595


namespace marbles_per_pack_l125_12523

theorem marbles_per_pack (total_marbles : ℕ) (leo_packs manny_packs neil_packs total_packs : ℕ) 
(h1 : total_marbles = 400) 
(h2 : leo_packs = 25) 
(h3 : manny_packs = total_packs / 4) 
(h4 : neil_packs = total_packs / 8) 
(h5 : leo_packs + manny_packs + neil_packs = total_packs) : 
total_marbles / total_packs = 10 := 
by sorry

end marbles_per_pack_l125_12523


namespace rhombus_other_diagonal_length_l125_12539

theorem rhombus_other_diagonal_length (area_square : ℝ) (side_length_square : ℝ) (d1_rhombus : ℝ) (d2_expected: ℝ) 
  (h1 : area_square = side_length_square^2) 
  (h2 : side_length_square = 8) 
  (h3 : d1_rhombus = 16) 
  (h4 : (d1_rhombus * d2_expected) / 2 = area_square) :
  d2_expected = 8 := 
by
  sorry

end rhombus_other_diagonal_length_l125_12539

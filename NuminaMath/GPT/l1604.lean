import Mathlib

namespace area_of_triangle_DOE_l1604_160495

-- Definitions of points D, O, and E
def D (p : ℝ) : ℝ × ℝ := (0, p)
def O : ℝ × ℝ := (0, 0)
def E : ℝ × ℝ := (15, 0)

-- Theorem statement
theorem area_of_triangle_DOE (p : ℝ) : 
  let base := 15
  let height := p
  let area := (1/2) * base * height
  area = (15 * p) / 2 :=
by sorry

end area_of_triangle_DOE_l1604_160495


namespace students_juice_count_l1604_160406

theorem students_juice_count (students chose_water chose_juice : ℕ) 
  (h1 : chose_water = 140) 
  (h2 : (25 : ℚ) / 100 * (students : ℚ) = chose_juice)
  (h3 : (70 : ℚ) / 100 * (students : ℚ) = chose_water) : 
  chose_juice = 50 :=
by 
  sorry

end students_juice_count_l1604_160406


namespace probability_heads_9_tails_at_least_2_l1604_160434

noncomputable def probability_exactly_nine_heads : ℚ :=
  let total_outcomes := 2 ^ 12
  let successful_outcomes := Nat.choose 12 9
  successful_outcomes / total_outcomes

theorem probability_heads_9_tails_at_least_2 (n : ℕ) (h : n = 12) :
  n = 12 → probability_exactly_nine_heads = 55 / 1024 := by
  intros h
  sorry

end probability_heads_9_tails_at_least_2_l1604_160434


namespace sum_of_even_numbers_l1604_160427

-- Define the sequence of even numbers between 1 and 1001
def even_numbers_sequence (n : ℕ) := 2 * n

-- Conditions
def first_term := 2
def last_term := 1000
def common_difference := 2
def num_terms := 500
def sum_arithmetic_series (n : ℕ) (a l : ℕ) := n * (a + l) / 2

-- Main statement to be proved
theorem sum_of_even_numbers : 
  sum_arithmetic_series num_terms first_term last_term = 250502 := 
by
  sorry

end sum_of_even_numbers_l1604_160427


namespace problem_correct_l1604_160418

noncomputable def problem := 
  1 - (1 / 2)⁻¹ * Real.sin (60 * Real.pi / 180) + abs (2^0 - Real.sqrt 3) = 0

theorem problem_correct : problem := by
  sorry

end problem_correct_l1604_160418


namespace cost_of_other_disc_l1604_160432

theorem cost_of_other_disc (x : ℝ) (total_spent : ℝ) (num_discs : ℕ) (num_850_discs : ℕ) (price_850 : ℝ) 
    (total_cost : total_spent = 93) (num_bought : num_discs = 10) (num_850 : num_850_discs = 6) (price_per_850 : price_850 = 8.50) 
    (total_cost_850 : num_850_discs * price_850 = 51) (remaining_discs_cost : total_spent - 51 = 42) (remaining_discs : num_discs - num_850_discs = 4) :
    total_spent = num_850_discs * price_850 + (num_discs - num_850_discs) * x → x = 10.50 :=
by
  sorry

end cost_of_other_disc_l1604_160432


namespace parabola_directrix_l1604_160482

theorem parabola_directrix (x y : ℝ) (h : x^2 = 12 * y) : y = -3 :=
sorry

end parabola_directrix_l1604_160482


namespace pastry_problem_minimum_n_l1604_160477

theorem pastry_problem_minimum_n (fillings : Finset ℕ) (n : ℕ) : 
    fillings.card = 10 →
    (∃ pairs : Finset (ℕ × ℕ), pairs.card = 45 ∧ ∀ (p : ℕ × ℕ), p ∈ pairs → p.1 ≠ p.2 ∧ p.1 ∈ fillings ∧ p.2 ∈ fillings) →
    (∀ (remaining_pies : Finset (ℕ × ℕ)), remaining_pies.card = 45 - n → 
     ∃ f1 f2, (f1, f2) ∈ remaining_pies → (f1 ∈ fillings ∧ f2 ∈ fillings)) →
    n = 36 :=
by
  intros h_fillings h_pairs h_remaining_pies
  sorry

end pastry_problem_minimum_n_l1604_160477


namespace total_apples_in_stack_l1604_160421

theorem total_apples_in_stack:
  let base_layer := 6 * 9
  let layer_2 := 5 * 8
  let layer_3 := 4 * 7
  let layer_4 := 3 * 6
  let layer_5 := 2 * 5
  let layer_6 := 1 * 4
  let top_layer := 2
  base_layer + layer_2 + layer_3 + layer_4 + layer_5 + layer_6 + top_layer = 156 :=
by sorry

end total_apples_in_stack_l1604_160421


namespace solution_system_inequalities_l1604_160466

theorem solution_system_inequalities (x : ℝ) : 
  (x - 4 ≤ 0 ∧ 2 * (x + 1) < 3 * x) ↔ (2 < x ∧ x ≤ 4) := 
sorry

end solution_system_inequalities_l1604_160466


namespace box_volume_l1604_160407

theorem box_volume (x y z : ℕ) 
  (h1 : 2 * x + 2 * y = 26)
  (h2 : x + z = 10)
  (h3 : y + z = 7) :
  x * y * z = 80 :=
by
  sorry

end box_volume_l1604_160407


namespace angle_sum_l1604_160476

theorem angle_sum (A B C : ℝ) (h1 : A > 0) (h2 : B > 0) (h3 : C > 0) (h_triangle : A + B + C = 180) (h_complement : 180 - C = 130) :
  A + B = 130 :=
by
  sorry

end angle_sum_l1604_160476


namespace lower_limit_of_a_l1604_160430

theorem lower_limit_of_a (a b : ℤ) (h_a : a < 26) (h_b1 : b > 14) (h_b2 : b < 31) (h_ineq : (4 : ℚ) / 3 ≤ a / b) : 
  20 ≤ a :=
by
  sorry

end lower_limit_of_a_l1604_160430


namespace arithmetic_sequence_common_difference_l1604_160446

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ) (d : ℝ) 
  (h1 : a 2 = 1)
  (h2 : a 3 + a 4 = 8)
  (h3 : ∀ n, a (n + 1) = a n + d) : 
  d = 2 :=
by
  sorry

end arithmetic_sequence_common_difference_l1604_160446


namespace rainfall_on_tuesday_is_correct_l1604_160414

-- Define the total days in a week
def days_in_week : ℕ := 7

-- Define the average rainfall for the whole week
def avg_rainfall : ℝ := 3.0

-- Define the total rainfall for the week
def total_rainfall : ℝ := avg_rainfall * days_in_week

-- Define a proposition that states rainfall on Tuesday equals 10.5 cm
def rainfall_on_tuesday (T : ℝ) : Prop :=
  T = 10.5

-- Prove that the rainfall on Tuesday is 10.5 cm given the conditions
theorem rainfall_on_tuesday_is_correct : rainfall_on_tuesday (total_rainfall / 2) :=
by
  sorry

end rainfall_on_tuesday_is_correct_l1604_160414


namespace no_such_A_exists_l1604_160413

def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem no_such_A_exists :
  ¬ ∃ A : ℕ, 0 < A ∧ digit_sum A = 16 ∧ digit_sum (2 * A) = 17 :=
by 
  sorry

end no_such_A_exists_l1604_160413


namespace determine_set_of_integers_for_ratio_l1604_160450

def arithmetic_sequences (a : ℕ → ℕ) (b : ℕ → ℕ) (S : ℕ → ℕ) (T : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, S n / T n = (31 * n + 101) / (n + 3)

def ratio_is_integer (a : ℕ → ℕ) (b : ℕ → ℕ) (n : ℕ) : Prop :=
  ∃ k : ℕ, a n / b n = k

theorem determine_set_of_integers_for_ratio (a b : ℕ → ℕ) (S T : ℕ → ℕ) :
  arithmetic_sequences a b S T →
  {n : ℕ | ratio_is_integer a b n} = {1, 3} :=
sorry

end determine_set_of_integers_for_ratio_l1604_160450


namespace triangle_side_relation_l1604_160496

theorem triangle_side_relation
  (A B C : ℝ)
  (a b c : ℝ)
  (h : 3 * (Real.sin (A / 2)) * (Real.sin (B / 2)) * (Real.cos (C / 2)) + (Real.sin (3 * A / 2)) * (Real.sin (3 * B / 2)) * (Real.cos (3 * C / 2)) = 0)
  (law_of_sines : a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C) :
  a^3 + b^3 = c^3 :=
by
  sorry

end triangle_side_relation_l1604_160496


namespace truck_capacity_l1604_160455

-- Definitions based on conditions
def initial_fuel : ℕ := 38
def total_money : ℕ := 350
def change : ℕ := 14
def cost_per_liter : ℕ := 3

-- Theorem statement
theorem truck_capacity :
  initial_fuel + (total_money - change) / cost_per_liter = 150 := by
  sorry

end truck_capacity_l1604_160455


namespace range_of_m_l1604_160437

noncomputable def p (x : ℝ) : Prop := |x - 3| ≤ 2
noncomputable def q (x m : ℝ) : Prop := (x - m + 1) * (x - m - 1) ≤ 0

theorem range_of_m {m : ℝ} (H : ∀ (x : ℝ), ¬p x → ¬q x m) :
  2 ≤ m ∧ m ≤ 4 :=
sorry

end range_of_m_l1604_160437


namespace speed_increase_l1604_160499

theorem speed_increase (v_initial: ℝ) (t_initial: ℝ) (t_new: ℝ) :
  v_initial = 60 → t_initial = 1 → t_new = 0.5 →
  v_new = (1 / (t_new / 60)) →
  v_increase = v_new - v_initial →
  v_increase = 60 :=
by
  sorry

end speed_increase_l1604_160499


namespace percentage_of_truth_speakers_l1604_160497

theorem percentage_of_truth_speakers
  (L : ℝ) (hL: L = 0.2)
  (B : ℝ) (hB: B = 0.1)
  (prob_truth_or_lies : ℝ) (hProb: prob_truth_or_lies = 0.4)
  (T : ℝ)
: T = prob_truth_or_lies - L + B :=
sorry

end percentage_of_truth_speakers_l1604_160497


namespace positive_difference_of_perimeters_l1604_160480

noncomputable def perimeter_figure1 : ℕ :=
  let outer_rectangle := 2 * (5 + 1)
  let inner_extension := 2 * (2 + 1)
  outer_rectangle + inner_extension

noncomputable def perimeter_figure2 : ℕ :=
  2 * (5 + 2)

theorem positive_difference_of_perimeters :
  (perimeter_figure1 - perimeter_figure2 = 4) :=
by
  let perimeter1 := perimeter_figure1
  let perimeter2 := perimeter_figure2
  sorry

end positive_difference_of_perimeters_l1604_160480


namespace find_unit_prices_minimize_cost_l1604_160469

-- Definitions for the given prices and conditions
def cypress_price := 200
def pine_price := 150

def cost_eq1 (x y : ℕ) : Prop := 2 * x + 3 * y = 850
def cost_eq2 (x y : ℕ) : Prop := 3 * x + 2 * y = 900

-- Proving the unit prices of cypress and pine trees
theorem find_unit_prices (x y : ℕ) (h1 : cost_eq1 x y) (h2 : cost_eq2 x y) :
  x = cypress_price ∧ y = pine_price :=
sorry

-- Definitions for the number of trees and their costs
def total_trees := 80
def cypress_min (a : ℕ) : Prop := a ≥ 2 * (total_trees - a)
def total_cost (a : ℕ) : ℕ := 200 * a + 150 * (total_trees - a)

-- Conditions given for minimizing the cost
theorem minimize_cost (a : ℕ) (h1 : cypress_min a) : 
  a = 54 ∧ (total_trees - a) = 26 ∧ total_cost a = 14700 :=
sorry

end find_unit_prices_minimize_cost_l1604_160469


namespace cubic_has_exactly_one_real_solution_sum_b_eq_three_fourths_l1604_160410

theorem cubic_has_exactly_one_real_solution_sum_b_eq_three_fourths 
  (b : ℝ)
  (h : ∀ x : ℝ, 4 * x^3 + 3 * x^2 + b * x + 27 = 0 → ∃! r : ℝ, r = x) :
  b = 3 / 4 := 
by
  sorry

end cubic_has_exactly_one_real_solution_sum_b_eq_three_fourths_l1604_160410


namespace problem_1_problem_2_l1604_160479

open Real

noncomputable def f (omega : ℝ) (x : ℝ) : ℝ := 
  (cos (omega * x) * cos (omega * x) + sqrt 3 * cos (omega * x) * sin (omega * x) - 1/2)

theorem problem_1 (ω : ℝ) (hω : ω > 0):
 (f ω x = sin (2 * x + π / 6)) ∧ 
 (∀ k : ℤ, ∀ x : ℝ, (-π / 3 + ↑k * π) ≤ x ∧ x ≤ (π / 6 + ↑k * π) → f ω x = sin (2 * x + π / 6)) :=
sorry

theorem problem_2 (A b S a : ℝ) (hA : A / 2 = π / 3)
  (hb : b = 1) (hS: S = sqrt 3) :
  a = sqrt 13 :=
sorry

end problem_1_problem_2_l1604_160479


namespace map_length_scale_l1604_160489

theorem map_length_scale (len1 len2 : ℕ) (dist1 dist2 : ℕ) (h1 : len1 = 15) (h2 : dist1 = 90) (h3 : len2 = 20) :
  dist2 = 120 :=
by
  sorry

end map_length_scale_l1604_160489


namespace binomial_expansion_equality_l1604_160422

theorem binomial_expansion_equality (x : ℝ) : 
  (x-1)^4 - 4*x*(x-1)^3 + 6*(x^2)*(x-1)^2 - 4*(x^3)*(x-1)*x^4 = 1 := 
by 
  sorry 

end binomial_expansion_equality_l1604_160422


namespace ratio_of_boys_to_girls_l1604_160431

-- Definitions based on the initial conditions
def G : ℕ := 135
def T : ℕ := 351

-- Noncomputable because it involves division which is not always computable
noncomputable def B : ℕ := T - G

-- Main theorem to prove the ratio
theorem ratio_of_boys_to_girls : (B : ℚ) / G = 8 / 5 :=
by
  -- Here would be the proof, skipped with sorry.
  sorry

end ratio_of_boys_to_girls_l1604_160431


namespace recurring_decimal_sum_is_13_over_33_l1604_160459

noncomputable def recurring_decimal_sum : ℚ :=
  let x := 1/3 -- 0.\overline{3}
  let y := 2/33 -- 0.\overline{06}
  x + y

theorem recurring_decimal_sum_is_13_over_33 : recurring_decimal_sum = 13/33 := by
  sorry

end recurring_decimal_sum_is_13_over_33_l1604_160459


namespace find_constants_C_and_A_l1604_160404

theorem find_constants_C_and_A :
  ∃ (C A : ℚ), (C * x + 7 - 17)/(x^2 - 9 * x + 20) = A / (x - 4) + 2 / (x - 5) ∧ B = 7 ∧ C = 12/5 ∧ A = 2/5 := sorry

end find_constants_C_and_A_l1604_160404


namespace desired_alcohol_percentage_l1604_160417

def initial_volume := 6.0
def initial_percentage := 35.0 / 100.0
def added_alcohol := 1.8
def final_volume := initial_volume + added_alcohol
def initial_alcohol := initial_volume * initial_percentage
def final_alcohol := initial_alcohol + added_alcohol
def desired_percentage := (final_alcohol / final_volume) * 100.0

theorem desired_alcohol_percentage : desired_percentage = 50.0 := 
by
  -- Proof would go here, but is omitted as per the instructions
  sorry

end desired_alcohol_percentage_l1604_160417


namespace zero_is_multiple_of_all_primes_l1604_160484

theorem zero_is_multiple_of_all_primes :
  ∀ (x : ℕ), (∀ p : ℕ, Prime p → ∃ n : ℕ, x = n * p) ↔ x = 0 := by
sorry

end zero_is_multiple_of_all_primes_l1604_160484


namespace total_cost_sean_bought_l1604_160467

theorem total_cost_sean_bought (cost_soda cost_soup cost_sandwich : ℕ) 
  (h_soda : cost_soda = 1)
  (h_soup : cost_soup = 3 * cost_soda)
  (h_sandwich : cost_sandwich = 3 * cost_soup) :
  3 * cost_soda + 2 * cost_soup + cost_sandwich = 18 := 
by
  sorry

end total_cost_sean_bought_l1604_160467


namespace div_246_by_73_sum_9999_999_99_9_prod_25_29_4_l1604_160444

-- Define the division of 246 by 73
theorem div_246_by_73 :
  246 / 73 = 3 + 27 / 73 :=
sorry

-- Define the sum calculation
theorem sum_9999_999_99_9 :
  9999 + 999 + 99 + 9 = 11106 :=
sorry

-- Define the product calculation
theorem prod_25_29_4 :
  25 * 29 * 4 = 2900 :=
sorry

end div_246_by_73_sum_9999_999_99_9_prod_25_29_4_l1604_160444


namespace largest_fraction_l1604_160442

theorem largest_fraction :
  (∀ (a b : ℚ), a = 2 / 5 → b = 1 / 3 → a < b) ∧  
  (∀ (a c : ℚ), a = 2 / 5 → c = 7 / 15 → a < c) ∧ 
  (∀ (a d : ℚ), a = 2 / 5 → d = 5 / 12 → a < d) ∧ 
  (∀ (a e : ℚ), a = 2 / 5 → e = 3 / 8 → a < e) ∧ 
  (∀ (b c : ℚ), b = 1 / 3 → c = 7 / 15 → b < c) ∧
  (∀ (b d : ℚ), b = 1 / 3 → d = 5 / 12 → b < d) ∧ 
  (∀ (b e : ℚ), b = 1 / 3 → e = 3 / 8 → b < e) ∧ 
  (∀ (c d : ℚ), c = 7 / 15 → d = 5 / 12 → c > d) ∧
  (∀ (c e : ℚ), c = 7 / 15 → e = 3 / 8 → c > e) ∧
  (∀ (d e : ℚ), d = 5 / 12 → e = 3 / 8 → d > e) :=
sorry

end largest_fraction_l1604_160442


namespace ramon_3_enchiladas_4_tacos_cost_l1604_160460

theorem ramon_3_enchiladas_4_tacos_cost :
  ∃ (e t : ℝ), 2 * e + 3 * t = 2.50 ∧ 3 * e + 2 * t = 2.70 ∧ 3 * e + 4 * t = 3.54 :=
by {
  sorry
}

end ramon_3_enchiladas_4_tacos_cost_l1604_160460


namespace Uki_earnings_l1604_160439

theorem Uki_earnings (cupcake_price cookie_price biscuit_price : ℝ) 
                     (cupcake_count cookie_count biscuit_count : ℕ)
                     (days : ℕ) :
  cupcake_price = 1.50 →
  cookie_price = 2 →
  biscuit_price = 1 →
  cupcake_count = 20 →
  cookie_count = 10 →
  biscuit_count = 20 →
  days = 5 →
  (days : ℝ) * (cupcake_price * (cupcake_count : ℝ) + cookie_price * (cookie_count : ℝ) + biscuit_price * (biscuit_count : ℝ)) = 350 := 
by
  sorry

end Uki_earnings_l1604_160439


namespace stationery_problem_l1604_160498

variables (S E : ℕ)

theorem stationery_problem
  (h1 : S - E = 30)
  (h2 : 4 * E = S) :
  S = 40 :=
by
  sorry

end stationery_problem_l1604_160498


namespace infection_average_l1604_160488

theorem infection_average (x : ℕ) (h : 1 + x + x * (1 + x) = 196) : x = 13 :=
sorry

end infection_average_l1604_160488


namespace complex_number_equality_l1604_160412

open Complex

theorem complex_number_equality (u v : ℂ) 
  (h1 : 3 * abs (u + 1) * abs (v + 1) ≥ abs (u * v + 5 * u + 5 * v + 1))
  (h2 : abs (u + v) = abs (u * v + 1)) : 
  u = 1 ∨ v = 1 :=
sorry

end complex_number_equality_l1604_160412


namespace polyhedron_has_triangular_face_l1604_160494

-- Let's define the structure of a polyhedron, its vertices, edges, and faces.
structure Polyhedron :=
(vertices : ℕ)
(edges : ℕ)
(faces : ℕ)

-- Let's assume a function that indicates if a polyhedron is convex.
def is_convex (P : Polyhedron) : Prop := sorry  -- Convexity needs a rigorous formal definition.

-- Define a face of a polyhedron as an n-sided polygon.
structure Face :=
(sides : ℕ)

-- Predicate to check if a face is triangular.
def is_triangle (F : Face) : Prop := F.sides = 3

-- Predicate to check if each vertex has at least four edges meeting at it.
def each_vertex_has_at_least_four_edges (P : Polyhedron) : Prop := 
  sorry  -- This would need a more intricate definition involving the degrees of vertices.

-- We state the theorem using the defined concepts.
theorem polyhedron_has_triangular_face 
(P : Polyhedron) 
(h1 : is_convex P) 
(h2 : each_vertex_has_at_least_four_edges P) :
∃ (F : Face), is_triangle F :=
sorry

end polyhedron_has_triangular_face_l1604_160494


namespace work_completion_time_l1604_160486

theorem work_completion_time (A B C : ℝ) (hA : A = 1 / 4) (hB : B = 1 / 12) (hAC : A + C = 1 / 2) :
  1 / (B + C) = 3 :=
by
  -- The proof goes here
  sorry

end work_completion_time_l1604_160486


namespace inequality_solution_set_l1604_160468

theorem inequality_solution_set {a : ℝ} (x : ℝ) :
  (∀ x, (x - a) / (x^2 - 3 * x + 2) ≥ 0 ↔ (1 < x ∧ x ≤ a) ∨ (2 < x)) → (1 < a ∧ a < 2) :=
by 
  -- We would fill in the proof here. 
  sorry

end inequality_solution_set_l1604_160468


namespace magician_hat_probability_l1604_160493

def total_arrangements : ℕ := Nat.choose 6 2
def favorable_arrangements : ℕ := Nat.choose 5 1
def probability_red_chips_drawn_first : ℚ := favorable_arrangements / total_arrangements

theorem magician_hat_probability :
  probability_red_chips_drawn_first = 1 / 3 :=
by
  sorry

end magician_hat_probability_l1604_160493


namespace Heather_delay_l1604_160461

noncomputable def find_start_time : ℝ :=
  let d := 15 -- Initial distance between Stacy and Heather in miles
  let H := 5 -- Heather's speed in miles/hour
  let S := H + 1 -- Stacy's speed in miles/hour
  let d_H := 5.7272727272727275 -- Distance Heather walked when they meet
  let t_H := d_H / H -- Time Heather walked till they meet in hours
  let d_S := S * t_H -- Distance Stacy walked till they meet in miles
  let total_distance := d_H + d_S -- Total distance covered when they meet in miles
  let remaining_distance := d - total_distance -- Remaining distance Stacy covers alone before Heather starts in miles
  let t_S := remaining_distance / S -- Time Stacy walked alone in hours
  let minutes := t_S * 60 -- Convert time Stacy walked alone to minutes
  minutes -- Result in minutes

theorem Heather_delay : find_start_time = 24 := by
  sorry -- Proof of the theorem

end Heather_delay_l1604_160461


namespace train_length_l1604_160429

theorem train_length (L V : ℝ) 
  (h1 : L = V * 110) 
  (h2 : L + 700 = V * 180) : 
  L = 1100 :=
by
  sorry

end train_length_l1604_160429


namespace find_salary_of_january_l1604_160416

variables (J F M A May : ℝ)

theorem find_salary_of_january
  (h1 : (J + F + M + A) / 4 = 8000)
  (h2 : (F + M + A + May) / 4 = 8600)
  (h3 : May = 6500) :
  J = 4100 := 
sorry

end find_salary_of_january_l1604_160416


namespace problem_a_problem_b_l1604_160402

variable (α : ℝ)

theorem problem_a (hα : 0 < α ∧ α < π) :
  Real.sqrt ((2 * Real.sin α - Real.sin (2 * α)) / (2 * Real.sin α + Real.sin (2 * α))) = Real.tan (α / 2) :=
sorry

theorem problem_b (hα : π < α ∧ α < 2 * π) :
  Real.sqrt ((2 * Real.sin α - Real.sin (2 * α)) / (2 * Real.sin α + Real.sin (2 * α))) = -Real.tan (α / 2) :=
sorry

end problem_a_problem_b_l1604_160402


namespace A_and_C_mutually_exclusive_l1604_160463

/-- Definitions for the problem conditions. -/
def A (all_non_defective : Prop) : Prop := all_non_defective
def B (all_defective : Prop) : Prop := all_defective
def C (at_least_one_defective : Prop) : Prop := at_least_one_defective

/-- Theorem stating that A and C are mutually exclusive. -/
theorem A_and_C_mutually_exclusive (all_non_defective at_least_one_defective : Prop) :
  A all_non_defective ∧ C at_least_one_defective → false :=
  sorry

end A_and_C_mutually_exclusive_l1604_160463


namespace tenth_term_geometric_sequence_l1604_160452

def a := 5
def r := Rat.ofInt 3 / 4
def n := 10

theorem tenth_term_geometric_sequence :
  a * r^(n-1) = Rat.ofInt 98415 / Rat.ofInt 262144 := sorry

end tenth_term_geometric_sequence_l1604_160452


namespace sequence_properties_sum_Tn_l1604_160487

noncomputable def a_n (n : ℕ) : ℤ := 2 * n - 1
noncomputable def b_n (n : ℕ) : ℤ := 2^(n - 1)
noncomputable def c_n (n : ℕ) : ℤ := (2 * n - 1) / 2^(n - 1)
noncomputable def T_n (n : ℕ) : ℤ := 6 - (2 * n + 3) / 2^(n - 1)

theorem sequence_properties : (d = 2) → (S₁₀ = 100) → 
  (∀ n : ℕ, a_n n = 2 * n - 1) ∧ (∀ n : ℕ, b_n n = 2^(n - 1)) := by
  sorry

theorem sum_Tn : (d > 1) → 
  (∀ n : ℕ, T_n n = 6 - (2 * n + 3) / 2^(n - 1)) := by
  sorry

end sequence_properties_sum_Tn_l1604_160487


namespace yvette_sundae_cost_l1604_160490

noncomputable def cost_friends : ℝ := 7.50 + 10.00 + 8.50
noncomputable def final_bill : ℝ := 42.00
noncomputable def tip_percentage : ℝ := 0.20
noncomputable def tip_amount : ℝ := tip_percentage * final_bill

theorem yvette_sundae_cost : 
  final_bill - (cost_friends + tip_amount) = 7.60 := by
  sorry

end yvette_sundae_cost_l1604_160490


namespace find_a_l1604_160415

theorem find_a (a x y : ℤ) (h1 : x = 1) (h2 : y = -2) (h3 : a * x + y = 1) : a = 3 :=
by
  sorry

end find_a_l1604_160415


namespace sqrt_49_mul_sqrt_25_l1604_160453

theorem sqrt_49_mul_sqrt_25 : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 :=
by
  sorry

end sqrt_49_mul_sqrt_25_l1604_160453


namespace abscissa_of_A_is_5_l1604_160441

theorem abscissa_of_A_is_5
  (A : ℝ × ℝ) (B : ℝ × ℝ)
  (hA : A.1 = A.2 ∧ A.1 > 0)
  (hB : B = (5, 0))
  (C : ℝ × ℝ) (D : ℝ × ℝ)
  (hC : C = ((A.1 + 5) / 2, A.2 / 2))
  (hD : D = (5 / 2, 5 / 2))
  (dot_product_eq : (B.1 - A.1, B.2 - A.2) • (D.1 - C.1, D.2 - C.2) = 0) :
  A.1 = 5 :=
sorry

end abscissa_of_A_is_5_l1604_160441


namespace increased_percentage_l1604_160471

theorem increased_percentage (P : ℝ) (N : ℝ) (hN : N = 80) 
  (h : (N + (P / 100) * N) - (N - (25 / 100) * N) = 30) : P = 12.5 := 
by 
  sorry

end increased_percentage_l1604_160471


namespace apple_price_equals_oranges_l1604_160419

theorem apple_price_equals_oranges (A O : ℝ) (H1 : A = 28 * O) (H2 : 45 * A + 60 * O = 1350) (H3 : 30 * A + 40 * O = 900) : A = 28 * O :=
by
  sorry

end apple_price_equals_oranges_l1604_160419


namespace denmark_pizza_combinations_l1604_160447

theorem denmark_pizza_combinations :
  (let cheese_options := 3
   let meat_options := 4
   let vegetable_options := 5
   let invalid_combinations := 1
   let total_combinations := cheese_options * meat_options * vegetable_options
   let valid_combinations := total_combinations - invalid_combinations
   valid_combinations = 59) :=
by
  sorry

end denmark_pizza_combinations_l1604_160447


namespace upstream_distance_l1604_160411

theorem upstream_distance
  (man_speed : ℝ)
  (downstream_distance : ℝ)
  (downstream_time : ℝ)
  (effective_downstream_speed: ℝ)
  (stream_speed : ℝ)
  (upstream_time : ℝ)
  (upstream_distance : ℝ):
  man_speed = 7 ∧ downstream_distance = 45 ∧ downstream_time = 5 ∧ effective_downstream_speed = man_speed + stream_speed 
  ∧ effective_downstream_speed * downstream_time = downstream_distance 
  ∧ upstream_time = 5 ∧ upstream_distance = (man_speed - stream_speed) * upstream_time 
  → upstream_distance = 25 :=
by
  sorry

end upstream_distance_l1604_160411


namespace part_1_solution_set_part_2_a_range_l1604_160445

noncomputable def f (x a : ℝ) : ℝ := |x - 1| + |x - a|

theorem part_1_solution_set (a : ℝ) (h : a = 4) :
  {x : ℝ | f x a ≥ 5} = {x : ℝ | x ≤ 0 ∨ x ≥ 5} :=
by
  sorry

theorem part_2_a_range :
  {a : ℝ | ∀ x : ℝ, f x a ≥ 4} = {a : ℝ | a ≤ -3 ∨ a ≥ 5} :=
by
  sorry

end part_1_solution_set_part_2_a_range_l1604_160445


namespace find_x_l1604_160425

def myOperation (x y : ℝ) : ℝ := 2 * x * y

theorem find_x (x : ℝ) (h : myOperation 9 (myOperation 4 x) = 720) : x = 5 :=
by
  sorry

end find_x_l1604_160425


namespace algebraic_expression_value_l1604_160475

theorem algebraic_expression_value (x : ℝ) (h : x^2 + 3 * x + 5 = 7) : 3 * x^2 + 9 * x - 11 = -5 :=
by
  sorry

end algebraic_expression_value_l1604_160475


namespace area_of_triangle_ABC_l1604_160451

variable (A : ℝ) -- Area of the triangle ABC
variable (S_heptagon : ℝ) -- Area of the heptagon ADECFGH
variable (S_overlap : ℝ) -- Overlapping area after folding

-- Given conditions
axiom ratio_condition : S_heptagon = (5 / 7) * A
axiom overlap_condition : S_overlap = 8

-- Proof statement
theorem area_of_triangle_ABC :
  A = 28 := by
  sorry

end area_of_triangle_ABC_l1604_160451


namespace sequence_general_term_l1604_160409

noncomputable def sequence_condition (a : ℕ → ℝ) : Prop :=
  (∀ n, 2 * a n = 3 * a (n + 1)) ∧ 
  (a 2 * a 5 = 8 / 27) ∧ 
  (∀ n, 0 < a n)

theorem sequence_general_term (a : ℕ → ℝ) (h : sequence_condition a) : 
  ∀ n, a n = (2 / 3)^(n - 2) :=
by 
  sorry

end sequence_general_term_l1604_160409


namespace range_of_m_F_x2_less_than_x2_minus_1_l1604_160403

noncomputable def f (x : ℝ) : ℝ := x + Real.log x
noncomputable def g (x : ℝ) : ℝ := 3 - 2 / x
noncomputable def T (x m : ℝ) : ℝ := Real.log x - x - 2 * m
noncomputable def F (x m : ℝ) : ℝ := x - m / x - 2 * Real.log x
noncomputable def h (t : ℝ) : ℝ := t - 2 * Real.log t - 1

-- (1)
theorem range_of_m (m : ℝ) (h_intersections : ∃ x y : ℝ, T x m = 0 ∧ T y m = 0 ∧ x ≠ y) :
  m < -1 / 2 := sorry

-- (2)
theorem F_x2_less_than_x2_minus_1 {m : ℝ} (h₀ : 0 < m ∧ m < 1) {x₁ x₂ : ℝ} (h₁ : 0 < x₁ ∧ x₁ < x₂)
  (h₂ : F x₁ m = 0 ∧ F x₂ m = 0) :
  F x₂ m < x₂ - 1 := sorry

end range_of_m_F_x2_less_than_x2_minus_1_l1604_160403


namespace closest_point_on_parabola_to_line_is_l1604_160400

-- Definitions of the parabola and the line
def parabola (x : ℝ) : ℝ := 4 * x^2
def line (x : ℝ) : ℝ := 4 * x - 5

-- Prove that the point on the parabola that is closest to the line is (1/2, 1)
theorem closest_point_on_parabola_to_line_is (x y : ℝ) :
  parabola x = y ∧ (∀ (x' y' : ℝ), parabola x' = y' -> (line x - y)^2 >= (line x' - y')^2) ->
  (x, y) = (1/2, 1) :=
by
  sorry

end closest_point_on_parabola_to_line_is_l1604_160400


namespace determine_k_l1604_160424

variable (x y z k : ℝ)

theorem determine_k (h1 : 7 / (x + y) = k / (x + z)) (h2 : k / (x + z) = 11 / (z - y)) : k = 18 := 
by 
  sorry

end determine_k_l1604_160424


namespace root_of_quadratic_l1604_160492

theorem root_of_quadratic :
  (∀ x : ℝ, 2 * x^2 + 3 * x - 65 = 0 → x = 5 ∨ x = -6.5) :=
sorry

end root_of_quadratic_l1604_160492


namespace base8_base13_to_base10_sum_l1604_160449

-- Definitions for the base 8 and base 13 numbers
def base8_to_base10 (a b c : ℕ) : ℕ := a * 64 + b * 8 + c
def base13_to_base10 (d e f : ℕ) : ℕ := d * 169 + e * 13 + f

-- Constants for the specific numbers in the problem
def num1 := base8_to_base10 5 3 7
def num2 := base13_to_base10 4 12 5

-- The theorem to prove
theorem base8_base13_to_base10_sum : num1 + num2 = 1188 := by
  sorry

end base8_base13_to_base10_sum_l1604_160449


namespace total_cost_correct_l1604_160473

-- Condition C1: There are 13 hearts in a deck of 52 playing cards. 
def hearts_in_deck : ℕ := 13

-- Condition C2: The number of cows is twice the number of hearts.
def cows_in_Devonshire : ℕ := 2 * hearts_in_deck

-- Condition C3: Each cow is sold at $200.
def cost_per_cow : ℕ := 200

-- Question Q1: Calculate the total cost of the cows.
def total_cost_of_cows : ℕ := cows_in_Devonshire * cost_per_cow

-- Final statement we need to prove
theorem total_cost_correct : total_cost_of_cows = 5200 := by
  -- This will be proven in the proof body
  sorry

end total_cost_correct_l1604_160473


namespace average_velocity_instantaneous_velocity_l1604_160454

noncomputable def s (t : ℝ) : ℝ := 8 - 3 * t^2

theorem average_velocity {Δt : ℝ} (h : Δt ≠ 0) :
  (s (1 + Δt) - s 1) / Δt = -6 - 3 * Δt :=
sorry

theorem instantaneous_velocity :
  deriv s 1 = -6 :=
sorry

end average_velocity_instantaneous_velocity_l1604_160454


namespace angle_measure_l1604_160474

theorem angle_measure (x : ℝ) (h : 90 - x = 3 * (180 - x)) : x = 45 := by
  sorry

end angle_measure_l1604_160474


namespace find_locus_of_T_l1604_160485

section Locus

variables {x y m : ℝ}
variable (M : ℝ × ℝ)

-- Condition: The equation of the ellipse
def ellipse (x y : ℝ) := (x^2 / 4) + y^2 = 1

-- Condition: Point P
def P := (1, 0)

-- Condition: M is any point on the ellipse, except A and B
def on_ellipse (M : ℝ × ℝ) := ellipse M.1 M.2 ∧ M ≠ (-2, 0) ∧ M ≠ (2, 0)

-- Condition: The intersection point N of line MP with the ellipse
def line_eq (m y : ℝ) := m * y + 1

-- Proposition: Locus of intersection point T of lines AM and BN
theorem find_locus_of_T 
  (hM : on_ellipse M)
  (hN : line_eq m M.2 = M.1)
  (hT : M.2 ≠ 0) :
  M.1 = 4 :=
sorry

end Locus

end find_locus_of_T_l1604_160485


namespace exists_a_squared_congruent_neg1_iff_p_mod_4_eq_1_l1604_160433

theorem exists_a_squared_congruent_neg1_iff_p_mod_4_eq_1 (p : ℕ) [Fact (Nat.Prime p)] (hp : p % 2 = 1) :
  (∃ a : ℤ, a^2 ≡ -1 [ZMOD p]) ↔ p % 4 = 1 :=
sorry

end exists_a_squared_congruent_neg1_iff_p_mod_4_eq_1_l1604_160433


namespace first_day_is_wednesday_l1604_160420

theorem first_day_is_wednesday (day22_wednesday : ∀ n, n = 22 → (n = 22 → "Wednesday" = "Wednesday")) :
  ∀ n, n = 1 → (n = 1 → "Wednesday" = "Wednesday") :=
by
  sorry

end first_day_is_wednesday_l1604_160420


namespace crimson_valley_skirts_l1604_160465

theorem crimson_valley_skirts (e : ℕ) (a : ℕ) (s : ℕ) (p : ℕ) (c : ℕ) 
  (h1 : e = 120) 
  (h2 : a = 2 * e) 
  (h3 : s = 3 * a / 5) 
  (h4 : p = s / 4) 
  (h5 : c = p / 3) : 
  c = 12 := 
by 
  sorry

end crimson_valley_skirts_l1604_160465


namespace algorithm_must_have_sequential_structure_l1604_160408

-- Definitions for types of structures used in algorithm definitions.
inductive Structure
| Logical
| Selection
| Loop
| Sequential

-- Predicate indicating whether a given Structure is necessary for any algorithm.
def necessary (s : Structure) : Prop :=
  match s with
  | Structure.Logical => False
  | Structure.Selection => False
  | Structure.Loop => False
  | Structure.Sequential => True

-- The theorem statement to prove that the sequential structure is necessary for any algorithm.
theorem algorithm_must_have_sequential_structure :
  necessary Structure.Sequential :=
by
  sorry

end algorithm_must_have_sequential_structure_l1604_160408


namespace plane_through_point_contains_line_l1604_160426

-- Definitions from conditions
structure Point := (x : ℝ) (y : ℝ) (z : ℝ)

def passes_through (p : Point) (plane : Point → Prop) : Prop :=
  plane p

def contains_line (line : ℝ → Point) (plane : Point → Prop) : Prop :=
  ∀ t, plane (line t)

def line_eq (t : ℝ) : Point :=
  ⟨4 * t + 2, -6 * t - 3, 2 * t + 4⟩

def plane_eq (A B C D : ℝ) (p : Point) : Prop :=
  A * p.x + B * p.y + C * p.z + D = 0

theorem plane_through_point_contains_line :
  ∃ (A B C D : ℝ), 1 < A ∧ gcd (abs A) (gcd (abs B) (gcd (abs C) (abs D))) = 1 ∧
  passes_through ⟨1, 2, -3⟩ (plane_eq A B C D) ∧
  contains_line line_eq (plane_eq A B C D) ∧ 
  (∃ (k : ℝ), 3 * k = A ∧ k = 1 / 3 ∧ B = k * 1 ∧ C = k * (-3) ∧ D = k * 2) :=
sorry

end plane_through_point_contains_line_l1604_160426


namespace sum_of_squares_l1604_160458

theorem sum_of_squares (x y : ℝ) (h1 : y + 6 = (x - 3)^2) (h2 : x + 6 = (y - 3)^2) (hxy : x ≠ y) : x^2 + y^2 = 43 :=
sorry

end sum_of_squares_l1604_160458


namespace letters_identity_l1604_160483

def identity_of_letters (first second third : ℕ) : Prop :=
  (first, second, third) = (1, 0, 1)

theorem letters_identity (first second third : ℕ) :
  first + second + third = 1 →
  (first = 1 → 1 ≠ first + second) →
  (second = 0 → first + second < 2) →
  (third = 0 → first + second = 1) →
  identity_of_letters first second third :=
by sorry

end letters_identity_l1604_160483


namespace exists_infinitely_many_triples_l1604_160428

theorem exists_infinitely_many_triples :
  ∀ n : ℕ, ∃ (a b c : ℕ), a^2 + b^2 + c^2 + 2016 = a * b * c :=
sorry

end exists_infinitely_many_triples_l1604_160428


namespace find_b_minus_c_l1604_160448

theorem find_b_minus_c (a b c : ℤ) (h : (x^2 + a * x - 3) * (x + 1) = x^3 + b * x^2 + c * x - 3) : b - c = 4 := by
  -- We would normally construct the proof here.
  sorry

end find_b_minus_c_l1604_160448


namespace total_weight_of_bottles_l1604_160436

variables (P G : ℕ) -- P stands for the weight of a plastic bottle, G stands for the weight of a glass bottle

-- Condition 1: The weight of 3 glass bottles is 600 grams
axiom glass_bottle_weight : 3 * G = 600

-- Condition 2: A glass bottle is 150 grams heavier than a plastic bottle
axiom glass_bottle_heavier : G = P + 150

-- The statement to prove: The total weight of 4 glass bottles and 5 plastic bottles is 1050 grams
theorem total_weight_of_bottles :
  4 * G + 5 * P = 1050 :=
sorry

end total_weight_of_bottles_l1604_160436


namespace probability_of_selecting_cooking_l1604_160472

-- Define the context where Xiao Ming has to choose a course randomly
def courses := ["planting", "cooking", "pottery", "carpentry"]

-- Define a statement to prove the probability of selecting "cooking" is 1/4
theorem probability_of_selecting_cooking :
  (1 / (courses.length : ℝ)) = 1 / 4 :=
by
  sorry

end probability_of_selecting_cooking_l1604_160472


namespace product_three_power_l1604_160405

theorem product_three_power (w : ℕ) (hW : w = 132) (hProd : ∃ (k : ℕ), 936 * w = 2^5 * 11^2 * k) : 
  ∃ (n : ℕ), (936 * w) = (2^5 * 11^2 * (3^3 * n)) :=
by 
  sorry

end product_three_power_l1604_160405


namespace arithmetic_sequence_has_correct_number_of_terms_l1604_160478

theorem arithmetic_sequence_has_correct_number_of_terms :
  ∀ (a₁ d : ℤ) (n : ℕ), a₁ = 1 ∧ d = -2 ∧ (n : ℤ) = (a₁ + (n - 1 : ℕ) * d) → n = 46 := by
  intros a₁ d n
  sorry

end arithmetic_sequence_has_correct_number_of_terms_l1604_160478


namespace HCF_48_99_l1604_160481

-- definitions and theorem stating the problem
def HCF (a b : ℕ) : ℕ := Nat.gcd a b

theorem HCF_48_99 : HCF 48 99 = 3 :=
by
  sorry

end HCF_48_99_l1604_160481


namespace area_of_square_with_diagonal_l1604_160464

theorem area_of_square_with_diagonal (c : ℝ) : 
  (∃ (s : ℝ), 2 * s^2 = c^4) → (∃ (A : ℝ), A = (c^4 / 2)) :=
  by
    sorry

end area_of_square_with_diagonal_l1604_160464


namespace math_problem_l1604_160462

theorem math_problem
  (x : ℝ)
  (h : (1/2) * x - 300 = 350) :
  (x + 200) * 2 = 3000 :=
by
  sorry

end math_problem_l1604_160462


namespace Trisha_total_distance_l1604_160470

theorem Trisha_total_distance :
  let d1 := 0.11  -- hotel to postcard shop
  let d2 := 0.11  -- postcard shop back to hotel
  let d3 := 1.52  -- hotel to T-shirt shop
  let d4 := 0.45  -- T-shirt shop to hat shop
  let d5 := 0.87  -- hat shop to purse shop
  let d6 := 2.32  -- purse shop back to hotel
  d1 + d2 + d3 + d4 + d5 + d6 = 5.38 :=
by
  sorry

end Trisha_total_distance_l1604_160470


namespace investment_amount_l1604_160491

noncomputable def calculate_principal (A : ℕ) (r t : ℝ) (n : ℕ) : ℝ :=
  A / (1 + r / n) ^ (n * t)

theorem investment_amount (A : ℕ) (r t : ℝ) (n P : ℕ) :
  A = 70000 → r = 0.08 → t = 5 → n = 12 →
  P = 46994 →
  calculate_principal A r t n = P :=
by
  intros hA hr ht hn hP
  rw [hA, hr, ht, hn, hP]
  sorry

end investment_amount_l1604_160491


namespace molecular_weight_NaClO_l1604_160443

theorem molecular_weight_NaClO :
  let Na := 22.99
  let Cl := 35.45
  let O := 16.00
  Na + Cl + O = 74.44 :=
by
  let Na := 22.99
  let Cl := 35.45
  let O := 16.00
  sorry

end molecular_weight_NaClO_l1604_160443


namespace battery_current_l1604_160435

variable (R : ℝ) (I : ℝ)

theorem battery_current (h1 : I = 48 / R) (h2 : R = 12) : I = 4 :=
by
  sorry

end battery_current_l1604_160435


namespace lcm_hcf_relationship_l1604_160457

theorem lcm_hcf_relationship (a b : ℕ) (h_prod : a * b = 84942) (h_hcf : Nat.gcd a b = 33) : Nat.lcm a b = 2574 :=
by
  sorry

end lcm_hcf_relationship_l1604_160457


namespace part1_answer1_part1_answer2_part2_answer1_part2_answer2_l1604_160440

open Set

def A : Set ℕ := {x | 1 ≤ x ∧ x < 11}
def B : Set ℕ := {1, 2, 3, 4}
def C : Set ℕ := {3, 4, 5, 6, 7}

theorem part1_answer1 : A ∩ C = {3, 4, 5, 6, 7} :=
by
  sorry

theorem part1_answer2 : A \ B = {5, 6, 7, 8, 9, 10} :=
by
  sorry

theorem part2_answer1 : A \ (B ∪ C) = {8, 9, 10} :=
by 
  sorry

theorem part2_answer2 : A ∪ (B ∩ C) = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} :=
by 
  sorry

end part1_answer1_part1_answer2_part2_answer1_part2_answer2_l1604_160440


namespace mps_to_kmph_conversion_l1604_160423

/-- Define the conversion factor from meters per second to kilometers per hour. -/
def mps_to_kmph : ℝ := 3.6

/-- Define the speed in meters per second. -/
def speed_mps : ℝ := 5

/-- Define the converted speed in kilometers per hour. -/
def speed_kmph : ℝ := 18

/-- Statement asserting the conversion from meters per second to kilometers per hour. -/
theorem mps_to_kmph_conversion : speed_mps * mps_to_kmph = speed_kmph := by 
  sorry

end mps_to_kmph_conversion_l1604_160423


namespace parabola_c_value_l1604_160438

theorem parabola_c_value :
  ∃ a b c : ℝ, (∀ y : ℝ, 4 = a * (3 : ℝ)^2 + b * 3 + c ∧ 2 = a * 5^2 + b * 5 + c ∧ c = -1 / 2) :=
by
  sorry

end parabola_c_value_l1604_160438


namespace find_f_l1604_160456

theorem find_f (f : ℤ → ℤ) (h : ∀ x : ℤ, f (x + 1) = x^2 + x) :
  ∀ x : ℤ, f x = x^2 - x :=
by
  intro x
  sorry

end find_f_l1604_160456


namespace max_k_consecutive_sum_2_times_3_pow_8_l1604_160401

theorem max_k_consecutive_sum_2_times_3_pow_8 :
  ∃ k : ℕ, 0 < k ∧ 
           (∃ n : ℕ, 2 * 3^8 = (k * (2 * n + k + 1)) / 2) ∧
           (∀ k' : ℕ, (∃ n' : ℕ, 0 < k' ∧ 2 * 3^8 = (k' * (2 * n' + k' + 1)) / 2) → k' ≤ 81) :=
sorry

end max_k_consecutive_sum_2_times_3_pow_8_l1604_160401

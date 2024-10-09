import Mathlib

namespace max_quarters_l1317_131756

theorem max_quarters (q : ℕ) (h1 : q + q + q / 2 = 20): q ≤ 11 :=
by
  sorry

end max_quarters_l1317_131756


namespace find_value_of_f2_l1317_131717

noncomputable def f (x : ℝ) : ℝ := (x + 1) / (x - 1)

theorem find_value_of_f2 : f 2 = 101 / 99 :=
  sorry

end find_value_of_f2_l1317_131717


namespace smallest_n_divisibility_l1317_131721

theorem smallest_n_divisibility (n : ℕ) (h : n = 5) : 
  ∃ k, (1 ≤ k ∧ k ≤ n + 1) ∧ (n^2 - n) % k = 0 ∧ ∃ i, (1 ≤ i ∧ i ≤ n + 1) ∧ (n^2 - n) % i ≠ 0 :=
by
  sorry

end smallest_n_divisibility_l1317_131721


namespace investment_amount_l1317_131759

theorem investment_amount (P: ℝ) (q_investment: ℝ) (ratio_pq: ℝ) (ratio_qp: ℝ) 
  (h1: ratio_pq = 4) (h2: ratio_qp = 6) (q_investment: ℝ) (h3: q_investment = 90000): 
  P = 60000 :=
by 
  -- Sorry is used here to skip the actual proof
  sorry

end investment_amount_l1317_131759


namespace solve_for_x_l1317_131708

theorem solve_for_x (y : ℝ) (x : ℝ) (h1 : y = 432) (h2 : 12^2 * x^4 / 432 = y) : x = 6 := by
  sorry

end solve_for_x_l1317_131708


namespace problem_l1317_131748

theorem problem (x : ℝ) (h : x^2 - 1 = 0) : x = -1 ∨ x = 1 :=
sorry

end problem_l1317_131748


namespace fraction_of_number_is_three_quarters_l1317_131706

theorem fraction_of_number_is_three_quarters 
  (f : ℚ) 
  (h1 : 76 ≠ 0) 
  (h2 : f * 76 = 76 - 19) : 
  f = 3 / 4 :=
by
  sorry

end fraction_of_number_is_three_quarters_l1317_131706


namespace number_of_pieces_l1317_131758

def area_of_pan (length : ℕ) (width : ℕ) : ℕ := length * width

def area_of_piece (side : ℕ) : ℕ := side * side

theorem number_of_pieces (length width side : ℕ) (h_length : length = 24) (h_width : width = 15) (h_side : side = 3) :
  (area_of_pan length width) / (area_of_piece side) = 40 :=
by
  rw [h_length, h_width, h_side]
  sorry

end number_of_pieces_l1317_131758


namespace value_of_polynomial_l1317_131718

theorem value_of_polynomial (x y : ℝ) (h : x - y = 5) : (x - y)^2 + 2 * (x - y) - 10 = 25 :=
by sorry

end value_of_polynomial_l1317_131718


namespace f_x_plus_f_neg_x_eq_seven_l1317_131791

variable (f : ℝ → ℝ)

-- Given conditions: 
axiom cond1 : ∀ x : ℝ, f x + f (1 - x) = 10
axiom cond2 : ∀ x : ℝ, f (1 + x) = 3 + f x

-- Prove statement:
theorem f_x_plus_f_neg_x_eq_seven : ∀ x : ℝ, f x + f (-x) = 7 := 
by
  sorry

end f_x_plus_f_neg_x_eq_seven_l1317_131791


namespace largest_multiple_of_15_less_than_500_is_495_l1317_131733

-- Define the necessary conditions
def is_multiple_of_15 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 15 * k

def is_less_than_500 (n : ℕ) : Prop :=
  n < 500

-- Problem statement: Prove that the largest positive multiple of 15 less than 500 is 495
theorem largest_multiple_of_15_less_than_500_is_495 : 
  ∀ n : ℕ, is_multiple_of_15 n → is_less_than_500 n → n ≤ 495 := 
by
  sorry

end largest_multiple_of_15_less_than_500_is_495_l1317_131733


namespace maximize_profit_l1317_131797

noncomputable def selling_price_to_maximize_profit (original_price selling_price : ℝ) (units units_sold_decrease : ℝ) : ℝ :=
  let x := 5
  let optimal_selling_price := selling_price + x
  optimal_selling_price

theorem maximize_profit :
  selling_price_to_maximize_profit 80 90 400 20 = 95 :=
by
  sorry

end maximize_profit_l1317_131797


namespace Xiaokang_position_l1317_131781

theorem Xiaokang_position :
  let east := 150
  let west := 100
  let total_walks := 3
  (east - west - west = -50) :=
sorry

end Xiaokang_position_l1317_131781


namespace polar_to_rectangular_correct_l1317_131751

noncomputable def polar_to_rectangular (rho theta x y : ℝ) : Prop :=
  rho = 4 * Real.sin theta + 2 * Real.cos theta ∧
  rho * Real.sin theta = y ∧
  rho * Real.cos theta = x ∧
  (x - 1) ^ 2 + (y - 2) ^ 2 = 5

theorem polar_to_rectangular_correct {rho theta x y : ℝ} :
  (rho = 4 * Real.sin theta + 2 * Real.cos theta) →
  (rho * Real.sin theta = y) →
  (rho * Real.cos theta = x) →
  (x - 1) ^ 2 + (y - 2) ^ 2 = 5 :=
by
  sorry

end polar_to_rectangular_correct_l1317_131751


namespace radius_intersection_xy_plane_l1317_131780

noncomputable def center_sphere : ℝ × ℝ × ℝ := (3, 3, 3)

def radius_xz_circle : ℝ := 2

def xz_center : ℝ × ℝ × ℝ := (3, 0, 3)

def xy_center : ℝ × ℝ × ℝ := (3, 3, 0)

theorem radius_intersection_xy_plane (r : ℝ) (s : ℝ) 
(h_center : center_sphere = (3, 3, 3)) 
(h_xz : xz_center = (3, 0, 3))
(h_r_xz : radius_xz_circle = 2)
(h_xy : xy_center = (3, 3, 0)):
s = 3 := 
sorry

end radius_intersection_xy_plane_l1317_131780


namespace eval_polynomial_positive_root_l1317_131796

theorem eval_polynomial_positive_root : 
  ∃ x : ℝ, (x^2 - 3 * x - 10 = 0 ∧ 0 < x ∧ (x^3 - 3 * x^2 - 9 * x + 7 = 12)) :=
sorry

end eval_polynomial_positive_root_l1317_131796


namespace cost_comparison_compare_cost_when_x_30_more_cost_effective_30_l1317_131714

-- Definitions for the conditions in the problem
def cost_of_suit : ℕ := 1000
def cost_of_tie : ℕ := 200

-- Definitions for Option 1 and Option 2 calculations
def option1_total_cost (x : ℕ) (h : x > 20) : ℕ := 200 * x + 16000
def option2_total_cost (x : ℕ) (h : x > 20) : ℕ := 180 * x + 18000

-- Case x=30 for comparison
def x : ℕ := 30
def option1_cost_when_x_30 : ℕ := 200 * x + 16000
def option2_cost_when_x_30 : ℕ := 180 * x + 18000

-- More cost-effective plan when x=30
def more_cost_effective_plan_for_x_30 : ℕ := 21800

theorem cost_comparison (x : ℕ) (h1 : x > 20) :
  option1_total_cost x h1 = 200 * x + 16000 ∧
  option2_total_cost x h1 = 180 * x + 18000 := 
by
  sorry

theorem compare_cost_when_x_30 :
  option1_cost_when_x_30 = 22000 ∧
  option2_cost_when_x_30 = 23400 ∧
  option1_cost_when_x_30 < option2_cost_when_x_30 := 
by
  sorry

theorem more_cost_effective_30 :
  more_cost_effective_plan_for_x_30 = 21800 := 
by
  sorry

end cost_comparison_compare_cost_when_x_30_more_cost_effective_30_l1317_131714


namespace digits_same_l1317_131790

theorem digits_same (k : ℕ) (hk : k ≥ 2) :
  (∃ n : ℕ, (10^(10^n) - 9^(9^n)) % (10^k) = 0) ↔ (k = 2 ∨ k = 3 ∨ k = 4) :=
sorry

end digits_same_l1317_131790


namespace concentric_circle_area_ratio_l1317_131728

theorem concentric_circle_area_ratio (r R : ℝ) (h_ratio : (π * R^2) / (π * r^2) = 16 / 3) :
  R - r = 1.309 * r :=
by
  sorry

end concentric_circle_area_ratio_l1317_131728


namespace number_of_zeros_is_one_l1317_131702

noncomputable def f (x : ℝ) : ℝ := Real.exp x + 3 * x

theorem number_of_zeros_is_one : 
  ∃! x : ℝ, f x = 0 :=
sorry

end number_of_zeros_is_one_l1317_131702


namespace total_tickets_correct_l1317_131799

-- Let's define the conditions given in the problem
def student_tickets (adult_tickets : ℕ) := 2 * adult_tickets
def adult_tickets := 122
def total_tickets := adult_tickets + student_tickets adult_tickets

-- We now state the theorem to be proved
theorem total_tickets_correct : total_tickets = 366 :=
by 
  sorry

end total_tickets_correct_l1317_131799


namespace shortest_handspan_is_Doyoon_l1317_131729

def Sangwon_handspan_cm : ℝ := 19.8
def Doyoon_handspan_cm : ℝ := 18.9
def Changhyeok_handspan_cm : ℝ := 19.3

theorem shortest_handspan_is_Doyoon :
  Doyoon_handspan_cm < Sangwon_handspan_cm ∧ Doyoon_handspan_cm < Changhyeok_handspan_cm :=
by
  sorry

end shortest_handspan_is_Doyoon_l1317_131729


namespace charlie_received_495_l1317_131763

theorem charlie_received_495 : 
  ∃ (A B C x : ℕ), 
    A + B + C = 1105 ∧ 
    A - 10 = 11 * x ∧ 
    B - 20 = 18 * x ∧ 
    C - 15 = 24 * x ∧ 
    C = 495 := 
by
  sorry

end charlie_received_495_l1317_131763


namespace football_team_birthday_collision_moscow_birthday_collision_l1317_131767

theorem football_team_birthday_collision (n : ℕ) (k : ℕ) (h1 : n ≥ 11) (h2 : k = 7) : 
  ∃ (d : ℕ) (p1 p2 : ℕ), p1 ≠ p2 ∧ p1 ≤ n ∧ p2 ≤ n ∧ d ≤ k :=
by sorry

theorem moscow_birthday_collision (population : ℕ) (days : ℕ) (h1 : population > 10000000) (h2 : days = 366) :
  ∃ (day : ℕ) (count : ℕ), count ≥ 10000 ∧ count ≤ population / days :=
by sorry

end football_team_birthday_collision_moscow_birthday_collision_l1317_131767


namespace intersection_point_of_lines_l1317_131769

theorem intersection_point_of_lines :
  ∃ x y : ℝ, (x - 2 * y - 4 = 0) ∧ (x + 3 * y + 6 = 0) ∧ (x = 0) ∧ (y = -2) :=
by
  sorry

end intersection_point_of_lines_l1317_131769


namespace infinite_solutions_x2_plus_2y2_eq_z2_no_nontrivial_solns_x2_plus_2y2_eq_z2_and_2x2_plus_y2_eq_t2_l1317_131746

-- Define x, y, z to be natural numbers
def has_infinitely_many_solutions : Prop :=
  ∃ (x y z : ℕ), x^2 + 2 * y^2 = z^2

-- Prove that there are infinitely many such x, y, z
theorem infinite_solutions_x2_plus_2y2_eq_z2 : has_infinitely_many_solutions :=
  sorry

-- Define x, y, z, t to be integers and non-zero
def no_nontrivial_integer_quadruplets : Prop :=
  ∀ (x y z t : ℤ), 
    (x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ t ≠ 0) → 
    ¬((x^2 + 2 * y^2 = z^2) ∧ (2 * x^2 + y^2 = t^2))

-- Prove that no nontrivial integer quadruplets exist
theorem no_nontrivial_solns_x2_plus_2y2_eq_z2_and_2x2_plus_y2_eq_t2 : no_nontrivial_integer_quadruplets :=
  sorry

end infinite_solutions_x2_plus_2y2_eq_z2_no_nontrivial_solns_x2_plus_2y2_eq_z2_and_2x2_plus_y2_eq_t2_l1317_131746


namespace total_games_played_l1317_131761

theorem total_games_played (months : ℕ) (games_per_month : ℕ) (h1 : months = 17) (h2 : games_per_month = 19) : 
  months * games_per_month = 323 :=
by
  sorry

end total_games_played_l1317_131761


namespace combined_molecular_weight_l1317_131719

def atomic_weight_C : ℝ := 12.01
def atomic_weight_Cl : ℝ := 35.45
def atomic_weight_S : ℝ := 32.07
def atomic_weight_F : ℝ := 19.00

def molecular_weight_CCl4 : ℝ := atomic_weight_C + 4 * atomic_weight_Cl
def molecular_weight_SF6 : ℝ := atomic_weight_S + 6 * atomic_weight_F

def weight_moles_CCl4 (moles : ℝ) : ℝ := moles * molecular_weight_CCl4
def weight_moles_SF6 (moles : ℝ) : ℝ := moles * molecular_weight_SF6

theorem combined_molecular_weight : weight_moles_CCl4 9 + weight_moles_SF6 5 = 2114.64 := by
  sorry

end combined_molecular_weight_l1317_131719


namespace alok_total_payment_l1317_131760

theorem alok_total_payment :
  let chapatis_cost := 16 * 6
  let rice_cost := 5 * 45
  let mixed_vegetable_cost := 7 * 70
  chapatis_cost + rice_cost + mixed_vegetable_cost = 811 :=
by
  sorry

end alok_total_payment_l1317_131760


namespace fraction_computation_l1317_131792

theorem fraction_computation : 
  (3 + 6 - 12 + 24 + 48 - 96) / (6 + 12 - 24 + 48 + 96 - 192) = 1 / 2 :=
by 
  sorry

end fraction_computation_l1317_131792


namespace sculpt_cost_in_mxn_l1317_131784

variable (usd_to_nad usd_to_mxn cost_nad cost_mxn : ℝ)

theorem sculpt_cost_in_mxn (h1 : usd_to_nad = 8) (h2 : usd_to_mxn = 20) (h3 : cost_nad = 160) : cost_mxn = 400 :=
by
  sorry

end sculpt_cost_in_mxn_l1317_131784


namespace men_with_all_attributes_le_l1317_131709

theorem men_with_all_attributes_le (total men_with_tv men_with_radio men_with_ac: ℕ) (married_men: ℕ) 
(h_total: total = 100) 
(h_married_men: married_men = 84) 
(h_men_with_tv: men_with_tv = 75) 
(h_men_with_radio: men_with_radio = 85) 
(h_men_with_ac: men_with_ac = 70) : 
  ∃ x, x ≤ men_with_ac ∧ x ≤ married_men ∧ x ≤ men_with_tv ∧ x ≤ men_with_radio ∧ (x ≤ total) := 
sorry

end men_with_all_attributes_le_l1317_131709


namespace final_hair_length_l1317_131703

theorem final_hair_length (x y z : ℕ) (hx : x = 16) (hy : y = 11) (hz : z = 12) : 
  (x - y) + z = 17 :=
by
  sorry

end final_hair_length_l1317_131703


namespace min_S_l1317_131779

-- Define the arithmetic sequence
def a (n : ℕ) (a1 d : ℤ) : ℤ :=
  a1 + (n - 1) * d

-- Define the sum of the first n terms
def S (n : ℕ) (a1 : ℤ) (d : ℤ) : ℤ :=
  (n * (a1 + a n a1 d)) / 2

-- Conditions
def a4 : ℤ := -15
def d : ℤ := 3

-- Found a1 from a4 and d
def a1 : ℤ := -24

-- Theorem stating the minimum value of the sum
theorem min_S : ∃ n, S n a1 d = -108 :=
  sorry

end min_S_l1317_131779


namespace arrange_digits_l1317_131720

theorem arrange_digits (A B C D E F : ℕ)
  (h1 : A ≠ B) (h2 : A ≠ C) (h3 : A ≠ D) (h4 : A ≠ E) (h5 : A ≠ F)
  (h6 : B ≠ C) (h7 : B ≠ D) (h8 : B ≠ E) (h9 : B ≠ F)
  (h10 : C ≠ D) (h11 : C ≠ E) (h12 : C ≠ F)
  (h13 : D ≠ E) (h14 : D ≠ F) (h15 : E ≠ F)
  (range_A : 1 ≤ A ∧ A ≤ 6) (range_B : 1 ≤ B ∧ B ≤ 6) (range_C : 1 ≤ C ∧ C ≤ 6)
  (range_D : 1 ≤ D ∧ D ≤ 6) (range_E : 1 ≤ E ∧ E ≤ 6) (range_F : 1 ≤ F ∧ F ≤ 6)
  (sum_line1 : A + D + E = 15) (sum_line2 : A + C + 9 = 15) 
  (sum_line3 : B + D + 9 = 15) (sum_line4 : 7 + C + E = 15) 
  (sum_line5 : 9 + C + A = 15) (sum_line6 : A + 8 + F = 15) 
  (sum_line7 : 7 + D + F = 15) : 
  (A = 4) ∧ (B = 1) ∧ (C = 2) ∧ (D = 5) ∧ (E = 6) ∧ (F = 3) :=
sorry

end arrange_digits_l1317_131720


namespace triangle_identity_l1317_131742

variables (a b c h_a h_b h_c x y z : ℝ)

-- Define the given conditions
def condition1 := a / h_a = x
def condition2 := b / h_b = y
def condition3 := c / h_c = z

-- Statement of the theorem to be proved
theorem triangle_identity 
  (h1 : condition1 a h_a x) 
  (h2 : condition2 b h_b y) 
  (h3 : condition3 c h_c z) : 
  x^2 + y^2 + z^2 - 2 * x * y - 2 * y * z - 2 * z * x + 4 = 0 := 
  by 
    sorry

end triangle_identity_l1317_131742


namespace floor_e_sub_6_eq_neg_4_l1317_131731

theorem floor_e_sub_6_eq_neg_4 :
  (⌊(e:Real) - 6⌋ = -4) :=
by
  let h₁ : 2 < (e:Real) := sorry -- assuming e is the base of natural logarithms
  let h₂ : (e:Real) < 3 := sorry
  sorry

end floor_e_sub_6_eq_neg_4_l1317_131731


namespace remainder_x_plus_1_pow_2025_mod_x_square_plus_1_eq_zero_l1317_131788

theorem remainder_x_plus_1_pow_2025_mod_x_square_plus_1_eq_zero (x : ℝ) :
  (x + 1) ^ 2025 % (x ^ 2 + 1) = 0 :=
  sorry

end remainder_x_plus_1_pow_2025_mod_x_square_plus_1_eq_zero_l1317_131788


namespace candy_box_price_l1317_131789

theorem candy_box_price (c s : ℝ) 
  (h1 : 1.50 * s = 6) 
  (h2 : c + s = 16) 
  (h3 : ∀ c, 1.25 * c = 1.25 * 12) : 
  (1.25 * c = 15) :=
by
  sorry

end candy_box_price_l1317_131789


namespace range_of_a_l1317_131775

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a ^ (-x)

theorem range_of_a (a : ℝ) (h_pos : 0 < a) (h_neq : a ≠ 1) (h_ineq : f a (-2) > f a (-3)) : 0 < a ∧ a < 1 :=
by {
  sorry
}

end range_of_a_l1317_131775


namespace abs_ineq_real_solution_range_l1317_131773

theorem abs_ineq_real_solution_range (a : ℝ) : 
  (∃ x : ℝ, |x - 4| + |x + 3| < a) ↔ a > 7 :=
sorry

end abs_ineq_real_solution_range_l1317_131773


namespace temperature_drop_l1317_131745

theorem temperature_drop (initial_temperature drop: ℤ) (h1: initial_temperature = 3) (h2: drop = 5) : initial_temperature - drop = -2 :=
by {
  sorry
}

end temperature_drop_l1317_131745


namespace abs_x_minus_2y_is_square_l1317_131772

theorem abs_x_minus_2y_is_square (x y : ℕ) (h : ∃ k : ℤ, x^2 - 4 * y + 1 = (x - 2 * y) * (1 - 2 * y) * k) : ∃ m : ℕ, x - 2 * y = m ^ 2 := by
  sorry

end abs_x_minus_2y_is_square_l1317_131772


namespace percentage_boys_playing_soccer_is_correct_l1317_131755

-- Definition of conditions 
def total_students := 420
def boys := 312
def soccer_players := 250
def girls_not_playing_soccer := 73

-- Calculated values based on conditions
def girls := total_students - boys
def girls_playing_soccer := girls - girls_not_playing_soccer
def boys_playing_soccer := soccer_players - girls_playing_soccer

-- Percentage of boys playing soccer
def percentage_boys_playing_soccer := (boys_playing_soccer / soccer_players) * 100

-- We assert the percentage of boys playing soccer is 86%
theorem percentage_boys_playing_soccer_is_correct : percentage_boys_playing_soccer = 86 := 
by
  -- Placeholder proof (use sorry as the proof is not required)
  sorry

end percentage_boys_playing_soccer_is_correct_l1317_131755


namespace least_positive_integer_satisfying_conditions_l1317_131713

theorem least_positive_integer_satisfying_conditions :
  ∃ b : ℕ, b > 0 ∧ (b % 7 = 6) ∧ (b % 11 = 10) ∧ (b % 13 = 12) ∧ b = 1000 :=
by
  sorry

end least_positive_integer_satisfying_conditions_l1317_131713


namespace birches_count_l1317_131738

-- Define the problem conditions
def total_trees : ℕ := 4000
def percentage_spruces : ℕ := 10
def percentage_pines : ℕ := 13
def number_spruces : ℕ := (percentage_spruces * total_trees) / 100
def number_pines : ℕ := (percentage_pines * total_trees) / 100
def number_oaks : ℕ := number_spruces + number_pines
def number_birches : ℕ := total_trees - number_oaks - number_pines - number_spruces

-- Prove the number of birches is 2160
theorem birches_count : number_birches = 2160 := by
  sorry

end birches_count_l1317_131738


namespace smallest_possible_l_l1317_131740

theorem smallest_possible_l (a b c L : ℕ) (h1 : a * b = 7) (h2 : a * c = 27) (h3 : b * c = L) (h4 : ∃ k, a * b * c = k * k) : L = 21 := sorry

end smallest_possible_l_l1317_131740


namespace systematic_sampling_probabilities_l1317_131766

-- Define the total number of students
def total_students : ℕ := 1005

-- Define the sample size
def sample_size : ℕ := 50

-- Define the number of individuals removed
def individuals_removed : ℕ := 5

-- Define the probability of an individual being removed
def probability_removed : ℚ := individuals_removed / total_students

-- Define the probability of an individual being selected in the sample
def probability_selected : ℚ := sample_size / total_students

-- The statement we need to prove
theorem systematic_sampling_probabilities :
  probability_removed = 5 / 1005 ∧ probability_selected = 50 / 1005 :=
sorry

end systematic_sampling_probabilities_l1317_131766


namespace total_students_in_school_l1317_131700

theorem total_students_in_school : 
  let C1 := 32
  let C2 := C1 - 2
  let C3 := C2 - 2
  let C4 := C3 - 2
  let C5 := C4 - 2
  C1 + C2 + C3 + C4 + C5 = 140 :=
by
  let C1 := 32
  let C2 := C1 - 2
  let C3 := C2 - 2
  let C4 := C3 - 2
  let C5 := C4 - 2
  sorry

end total_students_in_school_l1317_131700


namespace inverse_proportion_quadrants_l1317_131768

theorem inverse_proportion_quadrants (k : ℝ) (h : k ≠ 0) :
  (∃ (x y : ℝ), x = 1 ∧ y = 1 ∧ y = k / x) →
  (∀ x : ℝ, x ≠ 0 → ( (x > 0 → k / x > 0) ∧ (x < 0 → k / x < 0) ) ) :=
by
  sorry

end inverse_proportion_quadrants_l1317_131768


namespace relation_between_incircle_radius_perimeter_area_l1317_131712

theorem relation_between_incircle_radius_perimeter_area (r p S : ℝ) (h : S = (1 / 2) * r * p) : S = (1 / 2) * r * p :=
by {
  sorry
}

end relation_between_incircle_radius_perimeter_area_l1317_131712


namespace find_f_difference_l1317_131783

variable {α : Type*}
variable (f : ℝ → ℝ)
variable (h_odd : ∀ x, f (-x) = -f x)
variable (h_period : ∀ x, f (x + 5) = f x)
variable (h_value : f (-2) = 2)

theorem find_f_difference : f 2012 - f 2010 = -2 :=
by {
  sorry
}

end find_f_difference_l1317_131783


namespace is_correct_functional_expression_l1317_131705

variable (x : ℝ)

def is_isosceles_triangle (x : ℝ) (y : ℝ) : Prop :=
  2*x + y = 20

theorem is_correct_functional_expression (h1 : 5 < x) (h2 : x < 10) : 
  ∃ y, y = 20 - 2*x :=
by
  sorry

end is_correct_functional_expression_l1317_131705


namespace quadratic_roots_max_value_l1317_131744

theorem quadratic_roots_max_value (t q u₁ u₂ : ℝ)
  (h1 : u₁ + u₂ = t)
  (h2 : u₁ * u₂ = q)
  (h3 : u₁ + u₂ = u₁^2 + u₂^2)
  (h4 : u₁ + u₂ = u₁^4 + u₂^4) :
  (1 / u₁^2009 + 1 / u₂^2009) ≤ 2 :=
sorry

-- Explaination: 
-- This theorem states that given the conditions on the roots u₁ and u₂ of the quadratic equation, 
-- the maximum possible value of the expression (1 / u₁^2009 + 1 / u₂^2009) is 2.

end quadratic_roots_max_value_l1317_131744


namespace length_of_train_l1317_131754

-- We state the conditions as definitions.
def length_of_train_equals_length_of_platform (l_train l_platform : ℝ) : Prop :=
l_train = l_platform

def speed_of_train (s : ℕ) : Prop :=
s = 216

def crossing_time (t : ℕ) : Prop :=
t = 1

-- Defining the goal according to the problem statement.
theorem length_of_train (l_train l_platform : ℝ) (s t : ℕ) 
  (h1 : length_of_train_equals_length_of_platform l_train l_platform) 
  (h2 : speed_of_train s) 
  (h3 : crossing_time t) : 
  l_train = 1800 :=
by
  sorry

end length_of_train_l1317_131754


namespace f_x_f_2x_plus_1_l1317_131736

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x - 3

theorem f_x (x : ℝ) : f x = x^2 - 2 * x - 3 := 
by sorry

theorem f_2x_plus_1 (x : ℝ) : f (2 * x + 1) = 4 * x^2 - 4 := 
by sorry

end f_x_f_2x_plus_1_l1317_131736


namespace second_race_length_l1317_131771

variable (T L : ℝ)
variable (V_A V_B V_C : ℝ)

variables (h1 : V_A * T = 100)
variables (h2 : V_B * T = 90)
variables (h3 : V_C * T = 87)
variables (h4 : L / V_B = (L - 6) / V_C)

theorem second_race_length :
  L = 180 :=
sorry

end second_race_length_l1317_131771


namespace matthew_egg_rolls_l1317_131723

theorem matthew_egg_rolls (A P M : ℕ) 
  (h1 : M = 3 * P) 
  (h2 : P = A / 2) 
  (h3 : A = 4) : 
  M = 6 :=
by
  sorry

end matthew_egg_rolls_l1317_131723


namespace tree_drops_leaves_on_fifth_day_l1317_131701

def initial_leaves := 340
def daily_drop_fraction := 1 / 10

noncomputable def leaves_after_day (n: ℕ) : ℕ :=
  match n with
  | 0 => initial_leaves
  | 1 => initial_leaves - Nat.floor (initial_leaves * daily_drop_fraction)
  | 2 => leaves_after_day 1 - Nat.floor (leaves_after_day 1 * daily_drop_fraction)
  | 3 => leaves_after_day 2 - Nat.floor (leaves_after_day 2 * daily_drop_fraction)
  | 4 => leaves_after_day 3 - Nat.floor (leaves_after_day 3 * daily_drop_fraction)
  | _ => 0  -- beyond the 4th day

theorem tree_drops_leaves_on_fifth_day : leaves_after_day 4 = 225 := by
  -- We'll skip the detailed proof here, focusing on the statement
  sorry

end tree_drops_leaves_on_fifth_day_l1317_131701


namespace arithmetic_seq_problem_l1317_131753

-- Conditions and definitions for the arithmetic sequence
def arithmetic_seq (a_n : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a_n (n+1) = a_n n + d

def sum_seq (a_n S_n : ℕ → ℝ) : Prop :=
  ∀ n, S_n n = (n * (a_n 1 + a_n n)) / 2

def T_plus_K_eq_19 (T K : ℕ) : Prop :=
  T + K = 19

-- The given problem to prove
theorem arithmetic_seq_problem (a_n S_n : ℕ → ℝ) (d : ℝ) (h1 : d > 0)
  (h2 : arithmetic_seq a_n d) (h3 : sum_seq a_n S_n)
  (h4 : ∀ T K, T_plus_K_eq_19 T K → S_n T = S_n K) :
  ∃! n, a_n n - S_n n ≥ 0 := sorry

end arithmetic_seq_problem_l1317_131753


namespace john_must_study_4_5_hours_l1317_131727

-- Let "study_time" be the amount of time John needs to study for the second exam.

noncomputable def study_time_for_avg_score (hours1 score1 target_avg total_exams : ℝ) (direct_relation : Prop) :=
  2 * target_avg - score1 / (score1 / hours1)

theorem john_must_study_4_5_hours :
  study_time_for_avg_score 3 60 75 2 (60 / 3 = 90 / study_time_for_avg_score 3 60 75 2 (60 / 3 = 90 / study_time_for_avg_score 3 60 75 2 (sorry))) = 4.5 :=
sorry

end john_must_study_4_5_hours_l1317_131727


namespace prob_6_higher_than_3_after_10_shuffles_l1317_131710

def p_k (k : Nat) : ℚ := (3^k - 2^k) / (2 * 3^k)

theorem prob_6_higher_than_3_after_10_shuffles :
  p_k 10 = (3^10 - 2^10) / (2 * 3^10) :=
by
  sorry

end prob_6_higher_than_3_after_10_shuffles_l1317_131710


namespace number_of_friends_l1317_131711

theorem number_of_friends (total_bottle_caps : ℕ) (bottle_caps_per_friend : ℕ) (h1 : total_bottle_caps = 18) (h2 : bottle_caps_per_friend = 3) :
  total_bottle_caps / bottle_caps_per_friend = 6 :=
by
  sorry

end number_of_friends_l1317_131711


namespace geometric_sequence_a6a7_l1317_131747

theorem geometric_sequence_a6a7 (a : ℕ → ℝ) (q : ℝ)
  (h_geometric : ∀ n, a (n+1) = q * a n)
  (h1 : a 4 * a 5 = 1)
  (h2 : a 8 * a 9 = 16) : a 6 * a 7 = 4 :=
sorry

end geometric_sequence_a6a7_l1317_131747


namespace solve_for_x_l1317_131764

theorem solve_for_x (x : ℝ) (h : x ≠ -2) :
  (4 * x) / (x + 2) - 2 / (x + 2) = 3 / (x + 2) → x = 5 / 4 := by
  sorry

end solve_for_x_l1317_131764


namespace age_difference_l1317_131725

-- Defining the current age of the son
def S : ℕ := 26

-- Defining the current age of the man
def M : ℕ := 54

-- Defining the condition that in two years, the man's age is twice the son's age
def condition : Prop := (M + 2) = 2 * (S + 2)

-- The theorem that states how much older the man is than the son
theorem age_difference : condition → M - S = 28 := by
  sorry

end age_difference_l1317_131725


namespace minimum_box_value_l1317_131735

theorem minimum_box_value :
  ∃ (a b : ℤ), a * b = 36 ∧ (a^2 + b^2 = 72 ∧ ∀ (a' b' : ℤ), a' * b' = 36 → a'^2 + b'^2 ≥ 72) :=
by
  sorry

end minimum_box_value_l1317_131735


namespace num_values_of_a_l1317_131716

def A : Set ℝ := {-1, 1, 3}
def B (a : ℝ) : Set ℝ := {1, a^2 - 2 * a}

theorem num_values_of_a : ∃v : Finset ℝ, (∀ a ∈ v, B a ⊆ A) ∧ v.card = 3 :=
by
  sorry

end num_values_of_a_l1317_131716


namespace y_at_x8_l1317_131743

theorem y_at_x8 (k : ℝ) (y : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, y x = k * x^(1/3))
  (h2 : y 64 = 4 * Real.sqrt 3) :
  y 8 = 2 * Real.sqrt 3 :=
by
  sorry

end y_at_x8_l1317_131743


namespace tan_600_eq_neg_sqrt_3_l1317_131732

theorem tan_600_eq_neg_sqrt_3 : Real.tan (600 * Real.pi / 180) = -Real.sqrt 3 := by
  sorry

end tan_600_eq_neg_sqrt_3_l1317_131732


namespace option_A_correct_option_B_incorrect_option_C_incorrect_option_D_incorrect_l1317_131757

theorem option_A_correct (a : ℝ) : a ^ 2 * a ^ 3 = a ^ 5 := by {
  -- Here, we would provide the proof if required,
  -- but we are only stating the theorem.
  sorry
}

-- You may optionally add definitions of incorrect options for completeness.
theorem option_B_incorrect (a : ℝ) : ¬(a + 2 * a = 3 * a ^ 2) := by {
  sorry
}

theorem option_C_incorrect (a b : ℝ) : ¬((a * b) ^ 3 = a * b ^ 3) := by {
  sorry
}

theorem option_D_incorrect (a : ℝ) : ¬((-a ^ 3) ^ 2 = -a ^ 6) := by {
  sorry
}

end option_A_correct_option_B_incorrect_option_C_incorrect_option_D_incorrect_l1317_131757


namespace functional_eq_unique_solution_l1317_131749

theorem functional_eq_unique_solution (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (y - f x) = f x - 2 * x + f (f y)) :
  ∀ x : ℝ, f x = x :=
by
  sorry

end functional_eq_unique_solution_l1317_131749


namespace least_whole_number_clock_equivalent_l1317_131737

theorem least_whole_number_clock_equivalent :
  ∃ h : ℕ, h > 6 ∧ h ^ 2 % 24 = h % 24 ∧ ∀ k : ℕ, k > 6 ∧ k ^ 2 % 24 = k % 24 → h ≤ k := sorry

end least_whole_number_clock_equivalent_l1317_131737


namespace find_c_l1317_131724

-- Defining the variables and conditions given in the problem
variables (a b c : ℝ)

-- Conditions
def vertex_condition : Prop := (2, -3) = (a * (-3)^2 + b * (-3) + c, -3)
def point_condition : Prop := (7, -1) = (a * (-1)^2 + b * (-1) + c, -1)

-- Problem Statement
theorem find_c 
  (h_vertex : vertex_condition a b c)
  (h_point : point_condition a b c) :
  c = 53 / 4 :=
sorry

end find_c_l1317_131724


namespace find_c_l1317_131750

-- Define the polynomials f and g
def f (x : ℝ) : ℝ := 1 - 12 * x + 3 * x^2 - 4 * x^3 + 5 * x^4
def g (x : ℝ) : ℝ := 3 - 2 * x - 6 * x^3 + 7 * x^4

-- Define the main theorem stating that c = -5/7 makes f(x) + c*g(x) have degree 3
theorem find_c (c : ℝ) (h : ∀ x : ℝ, f x + c * g x = 0) : c = -5 / 7 := by
  sorry

end find_c_l1317_131750


namespace Francine_not_working_days_l1317_131762

-- Conditions
variables (d : ℕ) -- Number of days Francine works each week
def distance_per_day : ℕ := 140 -- Distance Francine drives each day
def total_distance_4_weeks : ℕ := 2240 -- Total distance in 4 weeks
def days_per_week : ℕ := 7 -- Days in a week

-- Proving that the number of days she does not go to work every week is 3
theorem Francine_not_working_days :
  (4 * distance_per_day * d = total_distance_4_weeks) →
  ((days_per_week - d) = 3) :=
by sorry

end Francine_not_working_days_l1317_131762


namespace parity_of_magazines_and_celebrities_l1317_131765

-- Define the main problem statement using Lean 4

theorem parity_of_magazines_and_celebrities {m c : ℕ}
  (h1 : ∀ i, i < m → ∃ d_i, d_i % 2 = 1)
  (h2 : ∀ j, j < c → ∃ e_j, e_j % 2 = 1) :
  (m % 2 = c % 2) ∧ (∃ ways, ways = 2 ^ ((m - 1) * (c - 1))) :=
by
  sorry

end parity_of_magazines_and_celebrities_l1317_131765


namespace gcd_sum_of_cubes_l1317_131777

-- Define the problem conditions
variables (n : ℕ) (h_pos : n > 27)

-- Define the goal to prove
theorem gcd_sum_of_cubes (h : n > 27) : 
  gcd (n^3 + 27) (n + 3) = n + 3 :=
by sorry

end gcd_sum_of_cubes_l1317_131777


namespace slope_of_line_det_by_two_solutions_l1317_131778

theorem slope_of_line_det_by_two_solutions (x y : ℝ) (h : 3 / x + 4 / y = 0) :
  (y = -4 * x / 3) → 
  ∀ x1 x2 y1 y2, (y1 = -4 * x1 / 3) ∧ (y2 = -4 * x2 / 3) → 
  (y2 - y1) / (x2 - x1) = -4 / 3 :=
sorry

end slope_of_line_det_by_two_solutions_l1317_131778


namespace prob_correct_l1317_131770

-- Define the individual probabilities.
def prob_first_ring := 1 / 10
def prob_second_ring := 3 / 10
def prob_third_ring := 2 / 5
def prob_fourth_ring := 1 / 10

-- Define the total probability of answering within the first four rings.
def prob_answer_within_four_rings := 
  prob_first_ring + prob_second_ring + prob_third_ring + prob_fourth_ring

-- State the theorem.
theorem prob_correct : prob_answer_within_four_rings = 9 / 10 :=
by
  -- We insert a placeholder for the proof.
  sorry

end prob_correct_l1317_131770


namespace volume_frustum_l1317_131787

noncomputable def volume_pyramid (base_edge height : ℝ) : ℝ :=
  (1/3) * (base_edge ^ 2) * height

theorem volume_frustum (original_base_edge original_height small_base_edge small_height : ℝ)
  (h_orig : original_base_edge = 10) (h_orig_height : original_height = 10)
  (h_small : small_base_edge = 5) (h_small_height : small_height = 5) :
  volume_pyramid original_base_edge original_height - volume_pyramid small_base_edge small_height
  = 875 / 3 := by
    simp [volume_pyramid, h_orig, h_orig_height, h_small, h_small_height]
    sorry

end volume_frustum_l1317_131787


namespace score_87_not_possible_l1317_131798

def max_score := 15 * 6
def score (correct unanswered incorrect : ℕ) := 6 * correct + unanswered

theorem score_87_not_possible :
  ¬∃ (correct unanswered incorrect : ℕ), 
    correct + unanswered + incorrect = 15 ∧
    6 * correct + unanswered = 87 := 
sorry

end score_87_not_possible_l1317_131798


namespace semicircle_radius_l1317_131730

noncomputable def radius_of_semicircle (P : ℝ) (h : P = 144) : ℝ :=
  144 / (Real.pi + 2)

theorem semicircle_radius (P : ℝ) (h : P = 144) : radius_of_semicircle P h = 144 / (Real.pi + 2) :=
  by sorry

end semicircle_radius_l1317_131730


namespace mika_jogging_speed_l1317_131793

theorem mika_jogging_speed 
  (s : ℝ)  -- Mika's constant jogging speed in meters per second.
  (r : ℝ)  -- Radius of the inner semicircle.
  (L : ℝ)  -- Length of each straight section.
  (h1 : 8 > 0) -- Overall width of the track is 8 meters.
  (h2 : (2 * L + 2 * π * (r + 8)) / s = (2 * L + 2 * π * r) / s + 48) -- Time difference equation.
  : s = π / 3 := 
sorry

end mika_jogging_speed_l1317_131793


namespace number_of_integer_solutions_Q_is_one_l1317_131734

def Q (x : ℤ) : ℤ := x^4 + 6 * x^3 + 13 * x^2 + 3 * x - 19

theorem number_of_integer_solutions_Q_is_one : 
    (∃! x : ℤ, ∃ k : ℤ, Q x = k^2) := 
sorry

end number_of_integer_solutions_Q_is_one_l1317_131734


namespace largest_three_digit_multiple_of_9_with_digits_sum_27_l1317_131722

theorem largest_three_digit_multiple_of_9_with_digits_sum_27 :
  ∃ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ (n % 9 = 0) ∧ ((n / 100) + ((n % 100) / 10) + (n % 10) = 27) ∧ n = 999 :=
by
  sorry

end largest_three_digit_multiple_of_9_with_digits_sum_27_l1317_131722


namespace length_of_longest_side_l1317_131782

variable (a b c p x l : ℝ)

-- conditions of the original problem
def original_triangle_sides (a b c : ℝ) : Prop := a = 8 ∧ b = 15 ∧ c = 17

def is_right_triangle (a b c : ℝ) : Prop := a^2 + b^2 = c^2

def similar_triangle_perimeter (a b c p x : ℝ) : Prop := (a * x) + (b * x) + (c * x) = p

-- proof target
theorem length_of_longest_side (h1: original_triangle_sides a b c) 
                               (h2: is_right_triangle a b c) 
                               (h3: similar_triangle_perimeter a b c p x) 
                               (h4: x = 4)
                               (h5: p = 160): (c * x) = 68 := by
  -- to complete the proof
  sorry

end length_of_longest_side_l1317_131782


namespace ratio_snakes_to_lions_is_S_per_100_l1317_131715

variables {S G : ℕ}

/-- Giraffe count in Safari National Park is 10 fewer than snakes -/
def safari_giraffes_minus_ten (S G : ℕ) : Prop := G = S - 10

/-- The number of lions in Safari National Park -/
def safari_lions : ℕ := 100

/-- The ratio of number of snakes to number of lions in Safari National Park -/
def ratio_snakes_to_lions (S : ℕ) : ℕ := S / safari_lions

/-- Prove the ratio of the number of snakes to the number of lions in Safari National Park -/
theorem ratio_snakes_to_lions_is_S_per_100 :
  ∀ S G, safari_giraffes_minus_ten S G → (ratio_snakes_to_lions S = S / 100) :=
by
  intros S G h
  sorry

end ratio_snakes_to_lions_is_S_per_100_l1317_131715


namespace sarah_pencils_on_tuesday_l1317_131786

theorem sarah_pencils_on_tuesday 
    (x : ℤ)
    (h1 : 20 + x + 3 * x = 92) : 
    x = 18 := 
by 
    sorry

end sarah_pencils_on_tuesday_l1317_131786


namespace fifteen_percent_minus_70_l1317_131704

theorem fifteen_percent_minus_70 (a : ℝ) : 0.15 * a - 70 = (15 / 100) * a - 70 :=
by sorry

end fifteen_percent_minus_70_l1317_131704


namespace parallelogram_area_l1317_131752

theorem parallelogram_area (s : ℝ) (ratio : ℝ) (A : ℝ) :
  s = 3 → ratio = 2 * Real.sqrt 2 → A = 9 → 
  (A * ratio = 18 * Real.sqrt 2) :=
by
  sorry

end parallelogram_area_l1317_131752


namespace domain_of_g_eq_7_infty_l1317_131785

noncomputable def domain_function (x : ℝ) : Prop := (2 * x + 1 ≥ 0) ∧ (x - 7 > 0)

theorem domain_of_g_eq_7_infty : 
  (∀ x : ℝ, domain_function x ↔ x > 7) :=
by 
  -- We declare the structure of our proof problem here.
  -- The detailed proof steps would follow.
  sorry

end domain_of_g_eq_7_infty_l1317_131785


namespace find_2a_plus_b_l1317_131707

theorem find_2a_plus_b (a b : ℝ) (h1 : 3 * a + 2 * b = 18) (h2 : 5 * a + 4 * b = 31) :
  2 * a + b = 11.5 :=
sorry

end find_2a_plus_b_l1317_131707


namespace total_packs_of_groceries_l1317_131726

-- Definitions for the conditions
def packs_of_cookies : ℕ := 2
def packs_of_cake : ℕ := 12

-- Theorem stating the total packs of groceries
theorem total_packs_of_groceries : packs_of_cookies + packs_of_cake = 14 :=
by sorry

end total_packs_of_groceries_l1317_131726


namespace map_distance_8_cm_l1317_131776

-- Define the conditions
def scale : ℕ := 5000000
def actual_distance_km : ℕ := 400
def actual_distance_cm : ℕ := 40000000
def map_distance_cm (x : ℕ) : Prop := x * scale = actual_distance_cm

-- The theorem to be proven
theorem map_distance_8_cm : ∃ x : ℕ, map_distance_cm x ∧ x = 8 :=
by
  use 8
  unfold map_distance_cm
  norm_num
  sorry

end map_distance_8_cm_l1317_131776


namespace payment_to_y_l1317_131741

theorem payment_to_y (X Y : ℝ) (h1 : X = 1.2 * Y) (h2 : X + Y = 580) : Y = 263.64 :=
by
  sorry

end payment_to_y_l1317_131741


namespace find_x_l1317_131774

-- Definitions from the conditions
def isPositiveMultipleOf7 (x : ℕ) : Prop := ∃ k : ℕ, x = 7 * k ∧ x > 0
def xSquaredGreaterThan150 (x : ℕ) : Prop := x^2 > 150
def xLessThan40 (x : ℕ) : Prop := x < 40

-- Main problem statement
theorem find_x (x : ℕ) (h1 : isPositiveMultipleOf7 x) (h2 : xSquaredGreaterThan150 x) (h3 : xLessThan40 x) : x = 14 :=
sorry

end find_x_l1317_131774


namespace circular_permutation_divisible_41_l1317_131795

theorem circular_permutation_divisible_41 (N : ℤ) (a b c d e : ℤ) (h : N = 10^4 * a + 10^3 * b + 10^2 * c + 10 * d + e)
  (h41 : 41 ∣ N) :
  ∀ (k : ℕ), 41 ∣ (10^((k % 5) * (4 - (k / 5))) * a + 10^((k % 5) * 3 + (k / 5) * 4) * b + 10^((k % 5) * 2 + (k / 5) * 3) * c + 10^((k % 5) + (k / 5) * 2) * d + 10^(k / 5) * e) :=
sorry

end circular_permutation_divisible_41_l1317_131795


namespace sum_of_possible_values_l1317_131739

theorem sum_of_possible_values 
  (x y : ℝ) 
  (h : x * y - x / y^2 - y / x^2 = 3) :
  (x = 0 ∨ y = 0 → False) → 
  ((x - 1) * (y - 1) = 1 ∨ (x - 1) * (y - 1) = 4) → 
  ((x - 1) * (y - 1) = 1 → (x - 1) * (y - 1) = 1) → 
  ((x - 1) * (y - 1) = 4 → (x - 1) * (y - 1) = 4) → 
  (1 + 4 = 5) := 
by 
  sorry

end sum_of_possible_values_l1317_131739


namespace total_value_proof_l1317_131794

def total_bills : ℕ := 126
def five_dollar_bills : ℕ := 84
def ten_dollar_bills : ℕ := total_bills - five_dollar_bills
def value_five_dollar_bills : ℕ := five_dollar_bills * 5
def value_ten_dollar_bills : ℕ := ten_dollar_bills * 10
def total_value : ℕ := value_five_dollar_bills + value_ten_dollar_bills

theorem total_value_proof : total_value = 840 := by
  unfold total_value value_five_dollar_bills value_ten_dollar_bills
  unfold five_dollar_bills ten_dollar_bills total_bills
  -- Calculation steps to show that value_five_dollar_bills + value_ten_dollar_bills = 840
  sorry

end total_value_proof_l1317_131794

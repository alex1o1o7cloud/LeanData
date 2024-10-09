import Mathlib

namespace mary_balloon_count_l2208_220811

theorem mary_balloon_count (n m : ℕ) (hn : n = 7) (hm : m = 4 * n) : m = 28 :=
by
  sorry

end mary_balloon_count_l2208_220811


namespace number_divided_by_five_is_same_as_three_added_l2208_220877

theorem number_divided_by_five_is_same_as_three_added :
  ∃ x : ℚ, x / 5 = x + 3 ∧ x = -15 / 4 :=
by
  sorry

end number_divided_by_five_is_same_as_three_added_l2208_220877


namespace sequence_product_l2208_220837

-- Definitions for the arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a1 d : ℝ), ∀ n, a n = a1 + (n - 1) * d

-- Definitions for the geometric sequence
def is_geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ (b1 r : ℝ), ∀ n, b n = b1 * r ^ (n - 1)

-- Defining the main proposition
theorem sequence_product (a b : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a)
  (h_geom  : is_geometric_sequence b)
  (h_eq    : b 7 = a 7)
  (h_cond  : 2 * a 2 - (a 7) ^ 2 + 2 * a 12 = 0) :
  b 3 * b 11 = 16 :=
sorry

end sequence_product_l2208_220837


namespace tan_alpha_plus_pi_div_four_l2208_220875

theorem tan_alpha_plus_pi_div_four
  (α : ℝ)
  (a : ℝ × ℝ := (3, 4))
  (b : ℝ × ℝ := (Real.sin α, Real.cos α))
  (h_parallel : ∃ k : ℝ, b = (k * a.1, k * a.2)) :
  Real.tan (α + Real.pi / 4) = 7 := by
  sorry

end tan_alpha_plus_pi_div_four_l2208_220875


namespace stacy_paper_shortage_l2208_220867

theorem stacy_paper_shortage:
  let bought_sheets : ℕ := 240 + 320
  let daily_mwf : ℕ := 60
  let daily_tt : ℕ := 100
  -- Calculate sheets used in a week
  let used_one_week : ℕ := (daily_mwf * 3) + (daily_tt * 2)
  -- Calculate sheets used in two weeks
  let used_two_weeks : ℕ := used_one_week * 2
  -- Remaining sheets at the end of two weeks
  let remaining_sheets : Int := bought_sheets - used_two_weeks
  remaining_sheets = -200 :=
by sorry

end stacy_paper_shortage_l2208_220867


namespace value_of_expression_l2208_220817

theorem value_of_expression (x y : ℚ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (x / |x| + |y| / y = 2) ∨ (x / |x| + |y| / y = 0) ∨ (x / |x| + |y| / y = -2) :=
by
  sorry

end value_of_expression_l2208_220817


namespace expand_binomials_l2208_220856

theorem expand_binomials (x : ℝ) : (3 * x + 4) * (2 * x + 7) = 6 * x^2 + 29 * x + 28 := 
by 
  sorry

end expand_binomials_l2208_220856


namespace intersection_of_M_and_N_l2208_220886

-- Define sets M and N
def M := {x : ℝ | (x + 2) * (x - 1) < 0}
def N := {x : ℝ | x + 1 < 0}

-- State the theorem for the intersection M ∩ N
theorem intersection_of_M_and_N :
  M ∩ N = {x : ℝ | -2 < x ∧ x < -1} :=
sorry

end intersection_of_M_and_N_l2208_220886


namespace average_payment_l2208_220806

-- Each condition from part a) is used as a definition here
variable (n : Nat) (p1 p2 first_payment remaining_payment : Nat)

-- Conditions given in natural language
def payments_every_year : Prop :=
  n = 52 ∧
  first_payment = 410 ∧
  remaining_payment = first_payment + 65 ∧
  p1 = 8 * first_payment ∧
  p2 = 44 * remaining_payment ∧
  p2 = 44 * (first_payment + 65) ∧
  p1 + p2 = 24180

-- The theorem to prove based on the conditions
theorem average_payment 
  (h : payments_every_year n p1 p2 first_payment remaining_payment) 
  : (p1 + p2) / n = 465 := 
sorry  -- Proof is omitted intentionally

end average_payment_l2208_220806


namespace todd_savings_l2208_220821

-- Define the initial conditions
def original_price : ℝ := 125
def sale_discount : ℝ := 0.20
def coupon : ℝ := 10
def card_discount : ℝ := 0.10

-- Define the resulting values after applying discounts
def sale_price := original_price * (1 - sale_discount)
def after_coupon := sale_price - coupon
def final_price := after_coupon * (1 - card_discount)

-- Define the total savings
def savings := original_price - final_price

-- The proof statement
theorem todd_savings : savings = 44 := by
  sorry

end todd_savings_l2208_220821


namespace negation_equivalence_l2208_220834

-- Define the propositions
def proposition (a b : ℝ) : Prop := a > b → a + 1 > b

def negation_proposition (a b : ℝ) : Prop := a ≤ b → a + 1 ≤ b

-- Statement to prove
theorem negation_equivalence (a b : ℝ) : ¬(proposition a b) ↔ negation_proposition a b := 
sorry

end negation_equivalence_l2208_220834


namespace solve_for_y_l2208_220855

theorem solve_for_y (y : ℕ) : 8^4 = 2^y → y = 12 :=
by
  sorry

end solve_for_y_l2208_220855


namespace solve_equation_1_solve_equation_2_l2208_220872

theorem solve_equation_1 :
  ∀ x : ℝ, 2 * x^2 - 4 * x = 0 ↔ (x = 0 ∨ x = 2) :=
by
  intro x
  sorry

theorem solve_equation_2 :
  ∀ x : ℝ, x^2 - 6 * x - 6 = 0 ↔ (x = 3 + Real.sqrt 15 ∨ x = 3 - Real.sqrt 15) :=
by
  intro x
  sorry

end solve_equation_1_solve_equation_2_l2208_220872


namespace chocolate_bar_cost_l2208_220831

-- Define the quantities Jessica bought
def chocolate_bars := 10
def gummy_bears_packs := 10
def chocolate_chips_bags := 20

-- Define the costs
def total_cost := 150
def gummy_bears_pack_cost := 2
def chocolate_chips_bag_cost := 5

-- Define what we want to prove (the cost of one chocolate bar)
theorem chocolate_bar_cost : 
  ∃ chocolate_bar_cost, 
    chocolate_bars * chocolate_bar_cost + 
    gummy_bears_packs * gummy_bears_pack_cost + 
    chocolate_chips_bags * chocolate_chips_bag_cost = total_cost ∧
    chocolate_bar_cost = 3 :=
by
  -- Proof goes here
  sorry

end chocolate_bar_cost_l2208_220831


namespace player_b_wins_l2208_220890

theorem player_b_wins : 
  ∃ B_strategy : (ℕ → ℕ → Prop), (∀ A_turn : ℕ → Prop, 
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ 2019 → (A_turn i ↔ ¬ A_turn (i + 1))) → 
  ((B_strategy 1 2019) ∨ ∃ k : ℕ, 1 ≤ k ∧ k ≤ 2019 ∧ B_strategy k (k + 1) ∧ ¬ A_turn k)) :=
sorry

end player_b_wins_l2208_220890


namespace solve_for_n_l2208_220858

theorem solve_for_n (n : ℕ) (h : 2 * 2 * 3 * 3 * 5 * 6 = 5 * 6 * n * n) : n = 6 := by
  sorry

end solve_for_n_l2208_220858


namespace find_x_l2208_220893

theorem find_x (x : ℝ) (h : ⌊x⌋ + x = 15/4) : x = 7/4 :=
sorry

end find_x_l2208_220893


namespace proof_part1_proof_part2_l2208_220854

variable {a b c : ℝ}

def pos_and_sum_of_powers (a b c : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ a^(3/2) + b^(3/2) + c^(3/2) = 1

theorem proof_part1 (h : pos_and_sum_of_powers a b c) : abc ≤ 1 / 9 :=
sorry

theorem proof_part2 (h : pos_and_sum_of_powers a b c) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end proof_part1_proof_part2_l2208_220854


namespace polygon_angle_ratio_pairs_count_l2208_220869

theorem polygon_angle_ratio_pairs_count :
  ∃ (m n : ℕ), (∃ (k : ℕ), (k > 0) ∧ (180 - 360 / ↑m) / (180 - 360 / ↑n) = 4 / 3
  ∧ Prime n ∧ (m - 6) * (n + 8) = 48 ∧ 
  ∃! (m n : ℕ), (180 - 360 / ↑m = (4 * (180 - 360 / ↑n)) / 3)) :=
sorry  -- Proof omitted, providing only the statement

end polygon_angle_ratio_pairs_count_l2208_220869


namespace area_between_hexagon_and_square_l2208_220830

noncomputable def circleRadius : ℝ := 6

noncomputable def centralAngleSquare : ℝ := Real.pi / 2

noncomputable def centralAngleHexagon : ℝ := Real.pi / 3

noncomputable def areaSegment (r α : ℝ) : ℝ :=
  0.5 * r^2 * (α - Real.sin α)

noncomputable def areaBetweenArcs : ℝ :=
  let r := circleRadius
  let T_AB := areaSegment r centralAngleSquare
  let T_CD := areaSegment r centralAngleHexagon
  2 * (T_AB - T_CD)

theorem area_between_hexagon_and_square :
  abs (areaBetweenArcs - 14.03) < 0.01 :=
by
  sorry

end area_between_hexagon_and_square_l2208_220830


namespace largest_x_l2208_220835

def largest_x_with_condition_eq_7_over_8 (x : ℝ) : Prop :=
  ⌊x⌋ / x = 7 / 8

theorem largest_x (x : ℝ) (h : largest_x_with_condition_eq_7_over_8 x) :
  x = 48 / 7 :=
sorry

end largest_x_l2208_220835


namespace a_investment_l2208_220859

theorem a_investment
  (b_investment : ℝ) (c_investment : ℝ) (c_share_profit : ℝ) (total_profit : ℝ)
  (h1 : b_investment = 45000)
  (h2 : c_investment = 50000)
  (h3 : c_share_profit = 36000)
  (h4 : total_profit = 90000) :
  ∃ A : ℝ, A = 30000 :=
by {
  sorry
}

end a_investment_l2208_220859


namespace y_intercept_probability_l2208_220823

theorem y_intercept_probability (b : ℝ) (hb : b ∈ Set.Icc (-2 : ℝ) 3 ) :
  (∃ P : ℚ, P = (2 / 5)) := 
by 
  sorry

end y_intercept_probability_l2208_220823


namespace correct_result_l2208_220899

-- Define the conditions
variables (x : ℤ)
axiom condition1 : (x - 27 + 19 = 84)

-- Define the goal
theorem correct_result : x - 19 + 27 = 100 :=
  sorry

end correct_result_l2208_220899


namespace total_apples_correctness_l2208_220860

-- Define the number of apples each man bought
def applesMen := 30

-- Define the number of apples each woman bought
def applesWomen := applesMen + 20

-- Define the total number of apples bought by the two men
def totalApplesMen := 2 * applesMen

-- Define the total number of apples bought by the three women
def totalApplesWomen := 3 * applesWomen

-- Define the total number of apples bought by the two men and three women
def totalApples := totalApplesMen + totalApplesWomen

-- Prove that the total number of apples bought by two men and three women is 210
theorem total_apples_correctness : totalApples = 210 := by
  sorry

end total_apples_correctness_l2208_220860


namespace simplify_expression_l2208_220845

theorem simplify_expression (x y z : ℝ) :
  3 * (x - (2 * y - 3 * z)) - 2 * ((3 * x - 2 * y) - 4 * z) = -3 * x - 2 * y + 17 * z :=
by
  sorry

end simplify_expression_l2208_220845


namespace rationalization_sum_l2208_220846

noncomputable def rationalize_denominator : ℚ :=
  let A := 5
  let B := 49
  let C := 21
  A + B + C

theorem rationalization_sum : rationalize_denominator = 75 := by
  sorry

end rationalization_sum_l2208_220846


namespace leo_score_l2208_220898

-- Definitions for the conditions
def caroline_score : ℕ := 13
def anthony_score : ℕ := 19
def winning_score : ℕ := 21

-- Lean statement for the proof problem
theorem leo_score : ∃ (leo_score : ℕ), leo_score = winning_score := by
  have h_caroline := caroline_score
  have h_anthony := anthony_score
  have h_winning := winning_score
  use 21
  sorry

end leo_score_l2208_220898


namespace hexagon_chord_problem_l2208_220829

-- Define the conditions of the problem
structure Hexagon :=
  (circumcircle : Type*)
  (inscribed : Prop)
  (AB BC CD : ℕ)
  (DE EF FA : ℕ)
  (chord_length_fraction_form : ℚ) 

-- Define the unique problem from given conditions and correct answer
theorem hexagon_chord_problem (hex : Hexagon) 
  (h1 : hex.inscribed)
  (h2 : hex.AB = 3) (h3 : hex.BC = 3) (h4 : hex.CD = 3)
  (h5 : hex.DE = 5) (h6 : hex.EF = 5) (h7 : hex.FA = 5)
  (h8 : hex.chord_length_fraction_form = 360 / 49) :
  let m := 360
  let n := 49
  m + n = 409 :=
by
  sorry

end hexagon_chord_problem_l2208_220829


namespace determine_f_2048_l2208_220894

theorem determine_f_2048 (f : ℕ → ℝ)
  (A1 : ∀ a b n : ℕ, a > 0 → b > 0 → a * b = 2^n → f a + f b = n^2)
  : f 2048 = 121 := by
  sorry

end determine_f_2048_l2208_220894


namespace greatest_visible_unit_cubes_from_one_point_12_l2208_220863

def num_unit_cubes (n : ℕ) : ℕ := n * n * n

def face_count (n : ℕ) : ℕ := n * n

def edge_count (n : ℕ) : ℕ := n

def visible_unit_cubes_from_one_point (n : ℕ) : ℕ :=
  let faces := 3 * face_count n
  let edges := 3 * (edge_count n - 1)
  let corner := 1
  faces - edges + corner

theorem greatest_visible_unit_cubes_from_one_point_12 :
  visible_unit_cubes_from_one_point 12 = 400 :=
  by
  sorry

end greatest_visible_unit_cubes_from_one_point_12_l2208_220863


namespace edward_original_amount_l2208_220857

theorem edward_original_amount (spent left total : ℕ) (h1 : spent = 13) (h2 : left = 6) (h3 : total = spent + left) : total = 19 := by 
  sorry

end edward_original_amount_l2208_220857


namespace bronchitis_option_D_correct_l2208_220818

noncomputable def smoking_related_to_bronchitis : Prop :=
  -- Conclusion that "smoking is related to chronic bronchitis"
sorry

noncomputable def confidence_level : ℝ :=
  -- Confidence level in the conclusion
  0.99

theorem bronchitis_option_D_correct :
  smoking_related_to_bronchitis →
  (confidence_level > 0.99) →
  -- Option D is correct: "Among 100 smokers, it is possible that not a single person has chronic bronchitis"
  ∃ (P : ℕ → Prop), (∀ n : ℕ, n ≤ 100 → P n = False) :=
by sorry

end bronchitis_option_D_correct_l2208_220818


namespace fair_coin_flip_probability_difference_l2208_220841

noncomputable def choose : ℕ → ℕ → ℕ
| _, 0 => 1
| 0, _ => 0
| n + 1, r + 1 => choose n r + choose n (r + 1)

theorem fair_coin_flip_probability_difference :
  let p := 1 / 2
  let p_heads_4_out_of_5 := choose 5 4 * p^4 * p
  let p_heads_5_out_of_5 := p^5
  p_heads_4_out_of_5 - p_heads_5_out_of_5 = 1 / 8 :=
by
  sorry

end fair_coin_flip_probability_difference_l2208_220841


namespace dimension_proof_l2208_220816

noncomputable def sports_field_dimensions (x y: ℝ) : Prop :=
  -- Given conditions
  x^2 + y^2 = 185^2 ∧
  (x - 4) * (y - 4) = x * y - 1012 ∧
  -- Seeking to prove dimensions
  ((x = 153 ∧ y = 104) ∨ (x = 104 ∧ y = 153))

theorem dimension_proof : ∃ x y: ℝ, sports_field_dimensions x y := by
  sorry

end dimension_proof_l2208_220816


namespace petya_can_force_difference_2014_l2208_220889

theorem petya_can_force_difference_2014 :
  ∀ (p q r : ℚ), ∃ (a b c : ℚ), ∀ (x : ℝ), (x^3 + a * x^2 + b * x + c = 0) → 
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 - x2 = 2014 :=
by sorry

end petya_can_force_difference_2014_l2208_220889


namespace max_arithmetic_sequence_terms_l2208_220888

theorem max_arithmetic_sequence_terms
  (n : ℕ)
  (a1 : ℝ)
  (d : ℝ) 
  (sum_sq_term_cond : (a1 + (n - 1) * d / 2)^2 + (n - 1) * (a1 + d * (n - 1) / 2) ≤ 100)
  (common_diff : d = 4)
  : n ≤ 8 := 
sorry

end max_arithmetic_sequence_terms_l2208_220888


namespace product_factors_eq_l2208_220897

theorem product_factors_eq :
  (1 - 1/2) * (1 - 1/3) * (1 - 1/4) * (1 - 1/5) * (1 - 1/6) * (1 - 1/7) * (1 - 1/8) * (1 - 1/9) * (1 - 1/10) * (1 - 1/11) = 1 / 11 := 
by
  sorry

end product_factors_eq_l2208_220897


namespace efficiency_ratio_l2208_220828

theorem efficiency_ratio (r : ℚ) (work_B : ℚ) (work_AB : ℚ) (B_alone : ℚ) (AB_together : ℚ) (efficiency_A : ℚ) (B_efficiency : ℚ) :
  B_alone = 30 ∧ AB_together = 20 ∧ B_efficiency = (1/B_alone) ∧ efficiency_A = (r * B_efficiency) ∧ (efficiency_A + B_efficiency) = (1 / AB_together) → r = 1 / 2 :=
by
  sorry

end efficiency_ratio_l2208_220828


namespace ratio_a7_b7_l2208_220822

-- Definitions of the conditions provided in the problem
variables {a b : ℕ → ℝ}   -- Arithmetic sequences {a_n} and {b_n}
variables {S T : ℕ → ℝ}   -- Sums of the first n terms of {a_n} and {b_n}

-- Condition: For any positive integer n, S_n / T_n = (3n + 5) / (2n + 3)
axiom condition_S_T (n : ℕ) (hn : 0 < n) : S n / T n = (3 * n + 5) / (2 * n + 3)

-- Goal: Prove that a_7 / b_7 = 44 / 29
theorem ratio_a7_b7 : a 7 / b 7 = 44 / 29 := 
sorry

end ratio_a7_b7_l2208_220822


namespace average_speed_ratio_l2208_220891

theorem average_speed_ratio 
  (jack_marathon_distance : ℕ) (jack_marathon_time : ℕ) 
  (jill_marathon_distance : ℕ) (jill_marathon_time : ℕ)
  (h1 : jack_marathon_distance = 40) (h2 : jack_marathon_time = 45) 
  (h3 : jill_marathon_distance = 40) (h4 : jill_marathon_time = 40) :
  (889 : ℕ) / 1000 = (jack_marathon_distance / jack_marathon_time) / 
                      (jill_marathon_distance / jill_marathon_time) :=
by
  sorry

end average_speed_ratio_l2208_220891


namespace product_of_two_numbers_l2208_220820

theorem product_of_two_numbers (x y : ℝ) (h1 : x - y = 12) (h2 : x^2 + y^2 = 340) : x * y = 97.9450625 :=
by
  sorry

end product_of_two_numbers_l2208_220820


namespace sales_tax_difference_l2208_220814

theorem sales_tax_difference :
  let price : ℝ := 50
  let tax_rate1 : ℝ := 0.075
  let tax_rate2 : ℝ := 0.0625
  let tax1 := price * tax_rate1
  let tax2 := price * tax_rate2
  let difference := tax1 - tax2
  difference = 0.625 :=
by
  sorry

end sales_tax_difference_l2208_220814


namespace trays_needed_l2208_220825

theorem trays_needed (cookies_classmates cookies_teachers cookies_per_tray : ℕ) 
  (hc1 : cookies_classmates = 276) 
  (hc2 : cookies_teachers = 92) 
  (hc3 : cookies_per_tray = 12) : 
  (cookies_classmates + cookies_teachers + cookies_per_tray - 1) / cookies_per_tray = 31 :=
by
  sorry

end trays_needed_l2208_220825


namespace q_can_be_true_or_false_l2208_220851

theorem q_can_be_true_or_false (p q : Prop) (h1 : ¬(p ∧ q)) (h2 : ¬p) : q ∨ ¬q :=
by
  sorry

end q_can_be_true_or_false_l2208_220851


namespace arithmetic_expressions_correctness_l2208_220879

theorem arithmetic_expressions_correctness :
  ((∀ (a b c : ℚ), (a + b) + c = a + (b + c)) ∧
   (∃ (a b c : ℚ), (a - b) - c ≠ a - (b - c)) ∧
   (∀ (a b c : ℚ), (a * b) * c = a * (b * c)) ∧
   (∃ (a b c : ℚ), a / b / c ≠ a / (b / c))) :=
by
  sorry

end arithmetic_expressions_correctness_l2208_220879


namespace time_for_B_alone_to_paint_l2208_220847

noncomputable def rate_A := 1 / 4
noncomputable def rate_BC := 1 / 3
noncomputable def rate_AC := 1 / 2
noncomputable def rate_DB := 1 / 6

theorem time_for_B_alone_to_paint :
  (1 / (rate_BC - (rate_AC - rate_A))) = 12 := by
  sorry

end time_for_B_alone_to_paint_l2208_220847


namespace model_price_and_schemes_l2208_220880

theorem model_price_and_schemes :
  ∃ (x y : ℕ), 3 * x = 2 * y ∧ x + 2 * y = 80 ∧ x = 20 ∧ y = 30 ∧ 
  ∃ (count m : ℕ), 468 ≤ m ∧ m ≤ 480 ∧ 
                   (20 * m + 30 * (800 - m) ≤ 19320) ∧ 
                   (800 - m ≥ 2 * m / 3) ∧ 
                   count = 13 ∧ 
                   800 - 480 = 320 :=
sorry

end model_price_and_schemes_l2208_220880


namespace max_distance_increases_l2208_220892

noncomputable def largest_n_for_rearrangement (C : ℕ) (marked_points : ℕ) : ℕ :=
  670

theorem max_distance_increases (C : ℕ) (marked_points : ℕ) (n : ℕ) (dist : ℕ → ℕ → ℕ) :
  ∀ i j, i < marked_points → j < marked_points →
    dist i j ≤ n → 
    (∃ rearrangement : ℕ → ℕ, 
    ∀ i j, i < marked_points → j < marked_points → 
      dist (rearrangement i) (rearrangement j) > dist i j) → 
    n ≤ largest_n_for_rearrangement C marked_points := 
by
  sorry

end max_distance_increases_l2208_220892


namespace range_of_a_l2208_220826

def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 2 * a * x + 4 > 0
def q (a : ℝ) : Prop := ∀ x : ℝ, (3 - 2 * a) ^ x > 0 -- using our characterization for 'increasing'

theorem range_of_a (a : ℝ) : (p a ∨ q a) ∧ ¬ (p a ∧ q a) ↔ (a ≤ -2 ∨ (1 ≤ a ∧ a < 2)) :=
by
  sorry

end range_of_a_l2208_220826


namespace determine_b_l2208_220807

noncomputable def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem determine_b (a b c m1 m2 : ℝ) (h1 : a > b) (h2 : b > c) (h3 : f a b c 1 = 0)
  (h4 : a^2 + (f a b c m1 + f a b c m2) * a + (f a b c m1) * (f a b c m2) = 0) : 
  b ≥ 0 := 
by
  -- Proof logic goes here
  sorry

end determine_b_l2208_220807


namespace chocolate_candies_total_cost_l2208_220881

-- Condition 1: A box of 30 chocolate candies costs $7.50
def box_cost : ℝ := 7.50
def candies_per_box : ℕ := 30

-- Condition 2: The local sales tax rate is 10%
def sales_tax_rate : ℝ := 0.10

-- Total number of candies to be bought
def total_candy_count : ℕ := 540

-- Calculate the number of boxes needed
def number_of_boxes (total_candies : ℕ) (candies_per_box : ℕ) : ℕ :=
  total_candies / candies_per_box

-- Calculate the cost without tax
def cost_without_tax (num_boxes : ℕ) (cost_per_box : ℝ) : ℝ :=
  num_boxes * cost_per_box

-- Calculate the total cost including tax
def total_cost_with_tax (cost : ℝ) (tax_rate : ℝ) : ℝ :=
  cost * (1 + tax_rate)

-- The main statement
theorem chocolate_candies_total_cost :
  total_cost_with_tax 
    (cost_without_tax (number_of_boxes total_candy_count candies_per_box) box_cost)
    sales_tax_rate = 148.50 :=
by
  sorry

end chocolate_candies_total_cost_l2208_220881


namespace downstream_distance_l2208_220866

theorem downstream_distance
  (time_downstream : ℝ) (time_upstream : ℝ)
  (distance_upstream : ℝ) (speed_still_water : ℝ)
  (h1 : time_downstream = 3) (h2 : time_upstream = 3)
  (h3 : distance_upstream = 15) (h4 : speed_still_water = 10) :
  ∃ d : ℝ, d = 45 :=
by
  sorry

end downstream_distance_l2208_220866


namespace arithmetic_sqrt_of_49_l2208_220844

theorem arithmetic_sqrt_of_49 : ∃ x : ℕ, x^2 = 49 ∧ x = 7 :=
by
  sorry

end arithmetic_sqrt_of_49_l2208_220844


namespace second_multiple_of_three_l2208_220883

theorem second_multiple_of_three (n : ℕ) (h : 3 * (n - 1) + 3 * (n + 1) = 150) : 3 * n = 75 :=
sorry

end second_multiple_of_three_l2208_220883


namespace triangle_with_altitudes_is_obtuse_l2208_220895

theorem triangle_with_altitudes_is_obtuse (h1 h2 h3 : ℝ) (h_pos1 : h1 > 0) (h_pos2 : h2 > 0) (h_pos3 : h3 > 0)
    (h_triangle_ineq1 : 1 / h2 + 1 / h3 > 1 / h1)
    (h_triangle_ineq2 : 1 / h1 + 1 / h3 > 1 / h2)
    (h_triangle_ineq3 : 1 / h1 + 1 / h2 > 1 / h3) : 
    ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    a * h1 = b * h2 ∧ b * h2 = c * h3 ∧
    (a^2 + b^2 < c^2 ∨ b^2 + c^2 < a^2 ∨ c^2 + a^2 < b^2) :=
sorry

end triangle_with_altitudes_is_obtuse_l2208_220895


namespace jon_weekly_speed_gain_l2208_220849

-- Definitions based on the conditions
def initial_speed : ℝ := 80
def speed_increase_percentage : ℝ := 0.20
def training_sessions : ℕ := 4
def weeks_per_session : ℕ := 4
def total_training_duration : ℕ := training_sessions * weeks_per_session

-- The calculated final speed
def final_speed : ℝ := initial_speed + initial_speed * speed_increase_percentage

theorem jon_weekly_speed_gain : 
  (final_speed - initial_speed) / total_training_duration = 1 :=
by
  -- This is the statement we want to prove
  sorry

end jon_weekly_speed_gain_l2208_220849


namespace factorization_of_expression_l2208_220873

-- Define variables
variables {a x y : ℝ}

-- State the problem
theorem factorization_of_expression : a * x^2 - 4 * a * y^2 = a * (x + 2 * y) * (x - 2 * y) :=
  sorry

end factorization_of_expression_l2208_220873


namespace train_speed_l2208_220864

theorem train_speed (length : ℝ) (time : ℝ) (length_eq : length = 400) (time_eq : time = 16) :
  (length / time) * (3600 / 1000) = 90 :=
by 
  rw [length_eq, time_eq]
  sorry

end train_speed_l2208_220864


namespace interest_rate_correct_l2208_220839

theorem interest_rate_correct :
  let SI := 155
  let P := 810
  let T := 4
  let R := SI * 100 / (P * T)
  R = 155 * 100 / (810 * 4) := 
sorry

end interest_rate_correct_l2208_220839


namespace systematic_sampling_employee_l2208_220896

theorem systematic_sampling_employee {x : ℕ} (h1 : 1 ≤ 6 ∧ 6 ≤ 52) (h2 : 1 ≤ 32 ∧ 32 ≤ 52) (h3 : 1 ≤ 45 ∧ 45 ≤ 52) (h4 : 6 + 45 = x + 32) : x = 19 :=
  by
    sorry

end systematic_sampling_employee_l2208_220896


namespace exponent_of_four_l2208_220871

theorem exponent_of_four (n : ℕ) (k : ℕ) (h : n = 21) 
  (eq : (↑(4 : ℕ) * 2 ^ (2 * n) = 4 ^ k)) : k = 22 :=
by
  sorry

end exponent_of_four_l2208_220871


namespace fraction_power_multiply_l2208_220882

theorem fraction_power_multiply :
  ((1 : ℚ) / 3)^4 * ((1 : ℚ) / 5) = (1 / 405 : ℚ) :=
by sorry

end fraction_power_multiply_l2208_220882


namespace solve_for_m_l2208_220887

theorem solve_for_m (m : ℝ) : 
  (∀ x : ℝ, (x = 2) → ((m - 2) * x = 5 * (x + 1))) → (m = 19 / 2) :=
by
  intro h
  have h1 := h 2
  sorry  -- proof can be filled in later

end solve_for_m_l2208_220887


namespace train_length_proof_l2208_220850

def speed_kmph : ℝ := 54
def time_seconds : ℝ := 54.995600351971845
def bridge_length_m : ℝ := 660
def train_length_approx : ℝ := 164.93

noncomputable def speed_m_s : ℝ := speed_kmph * (1000 / 3600)
noncomputable def total_distance : ℝ := speed_m_s * time_seconds
noncomputable def train_length : ℝ := total_distance - bridge_length_m

theorem train_length_proof :
  abs (train_length - train_length_approx) < 0.01 :=
by
  sorry

end train_length_proof_l2208_220850


namespace gary_money_after_sale_l2208_220838

theorem gary_money_after_sale :
  let initial_money := 73.0
  let sale_amount := 55.0
  initial_money + sale_amount = 128.0 :=
by
  let initial_money := 73.0
  let sale_amount := 55.0
  show initial_money + sale_amount = 128.0
  sorry

end gary_money_after_sale_l2208_220838


namespace assignment_plans_l2208_220815

theorem assignment_plans (students locations : ℕ) (library science_museum nursing_home : ℕ) 
  (students_eq : students = 5) (locations_eq : locations = 3) 
  (lib_gt0 : library > 0) (sci_gt0 : science_museum > 0) (nur_gt0 : nursing_home > 0) 
  (lib_science_nursing : library + science_museum + nursing_home = students) : 
  ∃ (assignments : ℕ), assignments = 150 :=
by
  sorry

end assignment_plans_l2208_220815


namespace total_first_half_points_l2208_220800

-- Define the sequences for Tigers and Lions
variables (a ar b d : ℕ)
-- Defining conditions
def tied_first_quarter : Prop := a = b
def geometric_tigers : Prop := ∃ r : ℕ, ar = a * r ∧ ar^2 = a * r^2 ∧ ar^3 = a * r^3
def arithmetic_lions : Prop := b+d = b + d ∧ b+2*d = b + 2*d ∧ b+3*d = b + 3*d
def tigers_win_by_four : Prop := (a + ar + ar^2 + ar^3) = (b + (b + d) + (b + 2*d) + (b + 3*d)) + 4
def score_limit : Prop := (a + ar + ar^2 + ar^3) ≤ 120 ∧ (b + (b + d) + (b + 2*d) + (b + 3*d)) ≤ 120

-- Goal: The total number of points scored by the two teams in the first half is 23
theorem total_first_half_points : tied_first_quarter a b ∧ geometric_tigers a ar ∧ arithmetic_lions b d ∧ tigers_win_by_four a ar b d ∧ score_limit a ar b d → 
(a + ar) + (b + d) = 23 := 
by {
  sorry
}

end total_first_half_points_l2208_220800


namespace sum_of_first_15_terms_l2208_220874

open scoped BigOperators

-- Define the sequence as an arithmetic sequence
def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ :=
  a + n * d

-- Define the condition given in the problem
def condition (a d : ℤ) : Prop :=
  3 * (arithmetic_sequence a d 2 + arithmetic_sequence a d 4) + 
  2 * (arithmetic_sequence a d 6 + arithmetic_sequence a d 11 + arithmetic_sequence a d 16) = 180

-- Prove that the sum of the first 15 terms is 225
theorem sum_of_first_15_terms (a d : ℤ) (h : condition a d) :
  ∑ i in Finset.range 15, arithmetic_sequence a d i = 225 :=
  sorry

end sum_of_first_15_terms_l2208_220874


namespace inequality_for_positive_integers_l2208_220870

theorem inequality_for_positive_integers (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a * b + b * c + a * c ≤ 3 * a * b * c :=
sorry

end inequality_for_positive_integers_l2208_220870


namespace identity_proof_l2208_220885

theorem identity_proof (n : ℝ) (h1 : n^2 ≥ 4) (h2 : n ≠ 0) :
    (n^3 - 3*n + (n^2 - 1)*Real.sqrt (n^2 - 4) - 2) / 
    (n^3 - 3*n + (n^2 - 1)*Real.sqrt (n^2 - 4) + 2)
    = ((n + 1) * Real.sqrt (n - 2)) / ((n - 1) * Real.sqrt (n + 2)) := by
  sorry

end identity_proof_l2208_220885


namespace inequality_system_solution_l2208_220884

theorem inequality_system_solution (x : ℤ) :
  (5 * x - 1 > 3 * (x + 1)) ∧ ((1 + 2 * x) / 3 ≥ x - 1) ↔ (x = 3 ∨ x = 4) := sorry

end inequality_system_solution_l2208_220884


namespace extra_pieces_of_gum_l2208_220840

theorem extra_pieces_of_gum (total_packages : ℕ) (pieces_per_package : ℕ) (total_pieces : ℕ) : ℕ :=
  if total_packages = 43 ∧ pieces_per_package = 23 ∧ total_pieces = 997 then
    997 - (43 * 23)
  else
    0  -- This is a dummy value for other cases, as they do not satisfy our conditions.

#print extra_pieces_of_gum

end extra_pieces_of_gum_l2208_220840


namespace composite_exists_for_x_64_l2208_220810

-- Define the conditions
def is_composite (n : ℕ) : Prop := ∃ m k : ℕ, m > 1 ∧ k > 1 ∧ n = m * k

-- Main statement
theorem composite_exists_for_x_64 :
  ∃ n : ℕ, is_composite (n^4 + 64) :=
sorry

end composite_exists_for_x_64_l2208_220810


namespace minimum_value_of_f_on_interval_l2208_220862

noncomputable def f (a x : ℝ) := Real.log x + a * x

theorem minimum_value_of_f_on_interval (a : ℝ) (h : a < 0) :
  ( ( -Real.log 2 ≤ a ∧ a < 0 ∧ ∀ x ∈ Set.Icc 1 2, f a x ≥ a ) ∧
    ( a < -Real.log 2 ∧ ∀ x ∈ Set.Icc 1 2, f a x ≥ (Real.log 2 + 2 * a) )
  ) :=
by
  sorry

end minimum_value_of_f_on_interval_l2208_220862


namespace function_is_linear_l2208_220868

theorem function_is_linear (f : ℝ → ℝ) :
  (∀ a b c d : ℝ,
    a ≠ b → b ≠ c → c ≠ d → d ≠ a →
    (a ≠ c ∧ b ≠ d ∧ a ≠ d ∧ b ≠ c) →
    (a - b) / (b - c) + (a - d) / (d - c) = 0 →
    (f a ≠ f b ∧ f b ≠ f c ∧ f c ≠ f d ∧ f a ≠ f c ∧ f a ≠ f d ∧ f b ≠ f d) →
    (f a - f b) / (f b - f c) + (f a - f d) / (f d - f c) = 0) →
  ∃ m c : ℝ, ∀ x : ℝ, f x = m * x + c :=
by
  sorry

end function_is_linear_l2208_220868


namespace eval_expression_l2208_220812

theorem eval_expression : 0.5 * 0.8 - 0.2 = 0.2 := by
  sorry

end eval_expression_l2208_220812


namespace total_pokemon_cards_l2208_220805

def pokemon_cards (sam dan tom keith : Nat) : Nat :=
  sam + dan + tom + keith

theorem total_pokemon_cards :
  pokemon_cards 14 14 14 14 = 56 := by
  sorry

end total_pokemon_cards_l2208_220805


namespace square_side_length_l2208_220801

/-- If the area of a square is 9m^2 + 24mn + 16n^2, then the length of the side of the square is |3m + 4n|. -/
theorem square_side_length (m n : ℝ) (a : ℝ) (h : a^2 = 9 * m^2 + 24 * m * n + 16 * n^2) : a = |3 * m + 4 * n| :=
sorry

end square_side_length_l2208_220801


namespace distance_between_points_A_and_B_l2208_220865

theorem distance_between_points_A_and_B :
  ∃ (d : ℝ), 
    -- Distance must be non-negative
    d ≥ 0 ∧
    -- Condition 1: Car 3 reaches point A at 10:00 AM (3 hours after 7:00 AM)
    (∃ V3 : ℝ, V3 = d / 6) ∧ 
    -- Condition 2: Car 2 reaches point A at 10:30 AM (3.5 hours after 7:00 AM)
    (∃ V2 : ℝ, V2 = 2 * d / 7) ∧ 
    -- Condition 3: When Car 1 and Car 3 meet, Car 2 has traveled exactly 3/8 of d
    (∃ V1 : ℝ, V1 = (d - 84) / 7 ∧ 2 * V1 + 2 * V3 = 8 * V2 / 3) ∧ 
    -- Required: The distance between A and B is 336 km
    d = 336 :=
by
  sorry

end distance_between_points_A_and_B_l2208_220865


namespace rectangular_field_length_l2208_220853

theorem rectangular_field_length {w l : ℝ} (h1 : l = 2 * w) (h2 : (8 : ℝ) * 8 = 1 / 18 * (l * w)) : l = 48 :=
by sorry

end rectangular_field_length_l2208_220853


namespace cans_of_soda_l2208_220809

variable (T R E : ℝ)

theorem cans_of_soda (hT: T > 0) (hR: R > 0) (hE: E > 0) : 5 * E * T / R = (5 * E) / R * T :=
by
  sorry

end cans_of_soda_l2208_220809


namespace spend_money_l2208_220803

theorem spend_money (n : ℕ) (h : n > 7) : ∃ a b : ℕ, 3 * a + 5 * b = n :=
by
  sorry

end spend_money_l2208_220803


namespace syllogism_sequence_correct_l2208_220876

-- Definitions based on conditions
def square_interior_angles_equal : Prop := ∀ (A B C D : ℝ), A = B ∧ B = C ∧ C = D
def rectangle_interior_angles_equal : Prop := ∀ (A B C D : ℝ), A = B ∧ B = C ∧ C = D
def square_is_rectangle : Prop := ∀ (S : Type), S = S

-- Final Goal
theorem syllogism_sequence_correct : (rectangle_interior_angles_equal → square_is_rectangle → square_interior_angles_equal) :=
by
  sorry

end syllogism_sequence_correct_l2208_220876


namespace population_doubles_in_35_years_l2208_220824

noncomputable def birth_rate : ℝ := 39.4 / 1000
noncomputable def death_rate : ℝ := 19.4 / 1000
noncomputable def natural_increase_rate : ℝ := birth_rate - death_rate
noncomputable def doubling_time (r: ℝ) : ℝ := 70 / (r * 100)

theorem population_doubles_in_35_years :
  doubling_time natural_increase_rate = 35 := by sorry

end population_doubles_in_35_years_l2208_220824


namespace ten_percent_of_n_l2208_220827

variable (n f : ℝ)

theorem ten_percent_of_n (h : n - (1 / 4 * 2) - (1 / 3 * 3) - f * n = 27) : 
  0.10 * n = 0.10 * (28.5 / (1 - f)) :=
by
  simp only [*, mul_one_div_cancel, mul_sub, sub_eq_add_neg, add_div, div_self, one_div, mul_add]
  sorry

end ten_percent_of_n_l2208_220827


namespace alex_not_read_probability_l2208_220832

def probability_reads : ℚ := 5 / 8
def probability_not_reads : ℚ := 3 / 8

theorem alex_not_read_probability : (1 - probability_reads) = probability_not_reads := 
by
  sorry

end alex_not_read_probability_l2208_220832


namespace new_fish_received_l2208_220819

def initial_fish := 14
def added_fish := 2
def eaten_fish := 6
def final_fish := 11

def current_fish := initial_fish + added_fish - eaten_fish
def returned_fish := 2
def exchanged_fish := final_fish - current_fish

theorem new_fish_received : exchanged_fish = 1 := by
  sorry

end new_fish_received_l2208_220819


namespace sqrt_k_kn_eq_k_sqrt_kn_l2208_220836

theorem sqrt_k_kn_eq_k_sqrt_kn (k n : ℕ) (h : k = Nat.sqrt (n + 1)) : 
  Real.sqrt (k * (k / n)) = k * Real.sqrt (k / n) := 
sorry

end sqrt_k_kn_eq_k_sqrt_kn_l2208_220836


namespace union_of_M_N_l2208_220802

-- Definitions of sets M and N
def M : Set ℕ := {0, 1}
def N : Set ℕ := {1, 2}

-- The theorem to prove
theorem union_of_M_N : M ∪ N = {0, 1, 2} :=
  by sorry

end union_of_M_N_l2208_220802


namespace find_g2_l2208_220808

theorem find_g2 (g : ℝ → ℝ) (h : ∀ x : ℝ, x ≠ 0 → g x - 3 * g (1 / x) = 3 ^ x) : 
  g 2 = (9 - 3 * Real.sqrt 3) / 8 := 
sorry

end find_g2_l2208_220808


namespace number_of_girls_l2208_220861

theorem number_of_girls (n : ℕ) (h : 2 * 25 * (n * (n - 1)) = 3 * 25 * 24) : (25 - n) = 16 := by
  sorry

end number_of_girls_l2208_220861


namespace no_arrangement_of_1_to_1978_coprime_l2208_220843

theorem no_arrangement_of_1_to_1978_coprime :
  ¬ ∃ (a : Fin 1978 → ℕ), 
    (∀ i : Fin 1977, Nat.gcd (a i) (a (i + 1)) = 1) ∧ 
    (∀ i : Fin 1976, Nat.gcd (a i) (a (i + 2)) = 1) ∧ 
    (∀ i : Fin 1978, 1 ≤ a i ∧ a i ≤ 1978 ∧ ∀ j : Fin 1978, (i ≠ j → a i ≠ a j)) :=
sorry

end no_arrangement_of_1_to_1978_coprime_l2208_220843


namespace jose_marks_difference_l2208_220842

theorem jose_marks_difference (M J A : ℕ) 
  (h1 : M = J - 20)
  (h2 : J + M + A = 210)
  (h3 : J = 90) : (J - A) = 40 :=
by
  sorry

end jose_marks_difference_l2208_220842


namespace second_divisor_is_340_l2208_220848

theorem second_divisor_is_340 
  (n : ℕ)
  (h1 : n = 349)
  (h2 : n % 13 = 11)
  (h3 : n % D = 9) : D = 340 :=
by
  sorry

end second_divisor_is_340_l2208_220848


namespace fraction_addition_l2208_220813

variable {w x y : ℝ}

theorem fraction_addition (h1 : w / x = 1 / 3) (h2 : w / y = 3 / 4) : (x + y) / y = 13 / 4 := by
  sorry

end fraction_addition_l2208_220813


namespace find_largest_number_l2208_220804

noncomputable def largest_of_three_numbers (x y z : ℝ) : ℝ :=
  if x ≥ y ∧ x ≥ z then x
  else if y ≥ x ∧ y ≥ z then y
  else z

theorem find_largest_number (x y z : ℝ) (h1 : x + y + z = 3) (h2 : xy + xz + yz = -11) (h3 : xyz = 15) :
  largest_of_three_numbers x y z = Real.sqrt 5 := by
  sorry

end find_largest_number_l2208_220804


namespace percent_calculation_l2208_220833

theorem percent_calculation (x : ℝ) (h : 0.30 * 0.40 * x = 24) : 0.20 * 0.60 * x = 24 := 
by
  sorry

end percent_calculation_l2208_220833


namespace largest_digit_divisible_by_6_l2208_220852

theorem largest_digit_divisible_by_6 : ∃ N : ℕ, N ≤ 9 ∧ (∃ d : ℕ, 3456 * 10 + N = 6 * d) ∧ (∀ M : ℕ, M ≤ 9 ∧ (∃ d : ℕ, 3456 * 10 + M = 6 * d) → M ≤ N) :=
sorry

end largest_digit_divisible_by_6_l2208_220852


namespace daps_equivalent_to_dips_l2208_220878

-- Definitions from conditions
def daps (n : ℕ) : ℕ := n
def dops (n : ℕ) : ℕ := n
def dips (n : ℕ) : ℕ := n

-- Given conditions
def equivalence_daps_dops : daps 8 = dops 6 := sorry
def equivalence_dops_dips : dops 3 = dips 11 := sorry

-- Proof problem
theorem daps_equivalent_to_dips (n : ℕ) (h1 : daps 8 = dops 6) (h2 : dops 3 = dips 11) : daps 24 = dips 66 :=
sorry

end daps_equivalent_to_dips_l2208_220878

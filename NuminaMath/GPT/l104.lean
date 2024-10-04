import Mathlib

namespace juju_juice_bar_l104_104658

theorem juju_juice_bar (M P : ℕ) 
  (h₁ : 6 * P = 54)
  (h₂ : 5 * M + 6 * P = 94) : 
  M + P = 17 := 
sorry

end juju_juice_bar_l104_104658


namespace comparison_of_negatives_l104_104154

theorem comparison_of_negatives : -2 < - (3 / 2) :=
by
  sorry

end comparison_of_negatives_l104_104154


namespace find_angle_OD_base_l104_104585

noncomputable def angle_between_edge_and_base (α β : ℝ): ℝ :=
  Real.arctan ((Real.sin α * Real.sin β) / Real.sqrt (Real.sin (α - β) * Real.sin (α + β)))

theorem find_angle_OD_base (α β : ℝ) :
  ∃ γ : ℝ, γ = angle_between_edge_and_base α β :=
sorry

end find_angle_OD_base_l104_104585


namespace sequence_formula_l104_104305

theorem sequence_formula (x : ℕ → ℤ) :
  x 1 = 1 →
  x 2 = -1 →
  (∀ n, n ≥ 2 → x (n-1) + x (n+1) = 2 * x n) →
  ∀ n, x n = -2 * n + 3 :=
by
  sorry

end sequence_formula_l104_104305


namespace number_in_marked_square_is_10_l104_104296

theorem number_in_marked_square_is_10 : 
  ∃ f : ℕ × ℕ → ℕ, 
    (f (0,0) = 5 ∧ f (0,1) = 6 ∧ f (0,2) = 7) ∧ 
    (∀ r c, r > 0 → 
      f (r,c) = f (r-1,c) + f (r-1,c+1)) 
    ∧ f (1, 1) = 13 
    ∧ f (2, 1) = 10 :=
    sorry

end number_in_marked_square_is_10_l104_104296


namespace match_end_time_is_17_55_l104_104006

-- Definitions corresponding to conditions
def start_time : ℕ := 15 * 60 + 30  -- Convert 15:30 to minutes past midnight
def duration : ℕ := 145  -- Duration in minutes

-- Definition corresponding to the question
def end_time : ℕ := start_time + duration 

-- Assertion corresponding to the correct answer
theorem match_end_time_is_17_55 : end_time = 17 * 60 + 55 :=
by
  -- Proof steps and actual proof will go here
  sorry

end match_end_time_is_17_55_l104_104006


namespace possible_degrees_of_remainder_l104_104628

theorem possible_degrees_of_remainder (p : Polynomial ℝ) :
  ∃ r q : Polynomial ℝ, p = q * (3 * X^3 - 4 * X^2 + 5 * X - 6) + r ∧ r.degree < 3 :=
sorry

end possible_degrees_of_remainder_l104_104628


namespace theater_rows_l104_104401

theorem theater_rows (R : ℕ) (h1 : R < 30 → ∃ r : ℕ, r < R ∧ r * 2 ≥ 30) (h2 : R ≥ 29 → 26 + 3 ≤ R) : R = 29 :=
by
  sorry

end theater_rows_l104_104401


namespace perimeter_of_semi_circle_region_l104_104489

theorem perimeter_of_semi_circle_region (side_length : ℝ) (h : side_length = 1/π) : 
  let radius := side_length / 2
  let circumference_of_half_circle := (1 / 2) * π * side_length
  3 * circumference_of_half_circle = 3 / 2
  := by
  sorry

end perimeter_of_semi_circle_region_l104_104489


namespace total_present_ages_l104_104782

variables (P Q : ℕ)

theorem total_present_ages :
  (P - 8 = (Q - 8) / 2) ∧ (P * 4 = Q * 3) → (P + Q = 28) :=
by
  sorry

end total_present_ages_l104_104782


namespace lottery_not_guaranteed_to_win_l104_104342

theorem lottery_not_guaranteed_to_win (total_tickets : ℕ) (winning_rate : ℚ) (num_purchased : ℕ) :
  total_tickets = 100000 ∧ winning_rate = 1 / 1000 ∧ num_purchased = 2000 → 
  ∃ (outcome : ℕ), outcome = 0 := by
  sorry

end lottery_not_guaranteed_to_win_l104_104342


namespace smallest_range_mean_2017_l104_104230

theorem smallest_range_mean_2017 :
  ∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  (a + b + c + d) / 4 = 2017 ∧ (max (max a b) (max c d) - min (min a b) (min c d)) = 4 := 
sorry

end smallest_range_mean_2017_l104_104230


namespace example_solution_l104_104065

variable (x y θ : Real)
variable (h1 : 0 < x) (h2 : 0 < y)
variable (h3 : θ ∈ Set.Ioo (Real.pi / 4) (Real.pi / 2))
variable (h4 : Real.sin θ / x = Real.cos θ / y)
variable (h5 : Real.cos θ ^ 2 / x ^ 2 + Real.sin θ ^ 2 / y ^ 2 = 10 / (3 * (x ^ 2 + y ^ 2)))

theorem example_solution : x / y = Real.sqrt 3 :=
by
  sorry

end example_solution_l104_104065


namespace compute_one_plus_i_power_four_l104_104811

theorem compute_one_plus_i_power_four (i : ℂ) (h : i^2 = -1) : (1 + i)^4 = -4 :=
by
  sorry

end compute_one_plus_i_power_four_l104_104811


namespace jason_earns_88_dollars_l104_104554

theorem jason_earns_88_dollars (earn_after_school: ℝ) (earn_saturday: ℝ)
  (total_hours: ℝ) (saturday_hours: ℝ) (after_school_hours: ℝ) (total_earn: ℝ)
  (h1 : earn_after_school = 4.00)
  (h2 : earn_saturday = 6.00)
  (h3 : total_hours = 18)
  (h4 : saturday_hours = 8)
  (h5 : after_school_hours = total_hours - saturday_hours)
  (h6 : total_earn = after_school_hours * earn_after_school + saturday_hours * earn_saturday) :
  total_earn = 88.00 :=
by
  sorry

end jason_earns_88_dollars_l104_104554


namespace democrats_ratio_l104_104922

theorem democrats_ratio (F M: ℕ) 
  (h_total_participants : F + M = 810)
  (h_female_democrats : 135 * 2 = F)
  (h_male_democrats : (1 / 4) * M = 135) : 
  (270 / 810 = 1 / 3) :=
by 
  sorry

end democrats_ratio_l104_104922


namespace evaluate_neg2012_l104_104055

def func (a b c x : ℝ) : ℝ := a * x^5 + b * x^3 + c * x + 1

theorem evaluate_neg2012 (a b c : ℝ) (h : func a b c 2012 = 3) : func a b c (-2012) = -1 :=
by
  sorry

end evaluate_neg2012_l104_104055


namespace problem2_l104_104693

theorem problem2 (x y : ℝ) (h1 : x^2 + x * y = 3) (h2 : x * y + y^2 = -2) : 
  2 * x^2 - x * y - 3 * y^2 = 12 := 
by 
  sorry

end problem2_l104_104693


namespace relative_value_ex1_max_value_of_m_plus_n_l104_104381

-- Definition of relative relationship value
def relative_relationship_value (a b n : ℚ) : ℚ := abs (a - n) + abs (b - n)

-- First problem statement
theorem relative_value_ex1 : relative_relationship_value 2 (-5) 2 = 7 := by
  sorry

-- Second problem statement: maximum value of m + n given the relative relationship value is 2
theorem max_value_of_m_plus_n (m n : ℚ) (h : relative_relationship_value m n 2 = 2) : m + n ≤ 6 := by
  sorry

end relative_value_ex1_max_value_of_m_plus_n_l104_104381


namespace find_number_l104_104623

theorem find_number (x : ℝ) (h : x / 5 = 30 + x / 6) : x = 900 :=
  sorry

end find_number_l104_104623


namespace compute_t_minus_s_l104_104492

noncomputable def t : ℚ := (40 + 30 + 30 + 20) / 4

noncomputable def s : ℚ := (40 * (40 / 120) + 30 * (30 / 120) + 30 * (30 / 120) + 20 * (20 / 120))

theorem compute_t_minus_s : t - s = -1.67 := by
  sorry

end compute_t_minus_s_l104_104492


namespace zero_x_intersections_l104_104864

theorem zero_x_intersections 
  (a b c : ℝ) 
  (h_geom_seq : b^2 = a * c) 
  (h_ac_pos : a * c > 0) : 
  ∀ x : ℝ, ¬(ax^2 + bx + c = 0) := 
by 
  sorry

end zero_x_intersections_l104_104864


namespace sin_of_7pi_over_6_l104_104204

theorem sin_of_7pi_over_6 : Real.sin (7 * Real.pi / 6) = -1 / 2 :=
by
  sorry

end sin_of_7pi_over_6_l104_104204


namespace probability_red_black_l104_104495

theorem probability_red_black :
  let cards := {c | c ∈ (Range 1 53)} -- Cards represented as numbers 1 to 52
  let suits := {1, 2, 3, 4} -- Suits represented as set of suits {1:hearts, 2:diamonds, 3:spades, 4:clubs}
  let number_of_cards := 52
  let number_of_red_cards := 26
  let number_of_black_cards := 26
  let probability_red_first := number_of_red_cards / number_of_cards
  let probability_black_second_given_red_first := number_of_black_cards / (number_of_cards - 1)
  let total_probability := probability_red_first * probability_black_second_given_red_first
  total_probability = 13 / 51 := by
sorry

end probability_red_black_l104_104495


namespace problem_statement_l104_104998

def a : ℝ × ℝ := (0, 2)
def b : ℝ × ℝ := (2, 2)

def vector_sub (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 - v.1, u.2 - v.2)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem problem_statement : dot_product (vector_sub a b) a = 0 := 
by 
  -- The proof would go here
  sorry

end problem_statement_l104_104998


namespace perpendicular_line_through_point_l104_104373

theorem perpendicular_line_through_point 
 {x y : ℝ}
 (p : (ℝ × ℝ)) 
 (point : p = (-2, 1)) 
 (perpendicular : ∀ x y, 2 * x - y + 4 = 0) : 
 (∀ x y, x + 2 * y = 0) ∧ (p.fst = -2 ∧ p.snd = 1) :=
by
  sorry

end perpendicular_line_through_point_l104_104373


namespace John_max_tests_under_B_l104_104023

theorem John_max_tests_under_B (total_tests first_tests tests_with_B goal_percentage B_tests_first_half : ℕ) :
  total_tests = 60 →
  first_tests = 40 → 
  tests_with_B = 32 → 
  goal_percentage = 75 →
  B_tests_first_half = 32 →
  let needed_B_tests := (goal_percentage * total_tests) / 100
  let remaining_tests := total_tests - first_tests
  let remaining_needed_B_tests := needed_B_tests - B_tests_first_half
  remaining_tests - remaining_needed_B_tests ≤ 7 := sorry

end John_max_tests_under_B_l104_104023


namespace mathe_matics_equals_2014_l104_104432

/-- 
Given the following mappings for characters in the word "MATHEMATICS":
M = 1, A = 8, T = 3, E = '+', I = 9, K = '-',
verify that the resulting numerical expression 183 + 1839 - 8 equals 2014.
-/
theorem mathe_matics_equals_2014 :
  183 + 1839 - 8 = 2014 :=
by
  sorry

end mathe_matics_equals_2014_l104_104432


namespace eliza_iron_total_l104_104191

-- Definition of the problem conditions in Lean
def blouse_time := 15 -- time to iron a blouse in minutes
def dress_time := 20 -- time to iron a dress in minutes
def blouse_hours := 2 -- hours spent ironing blouses
def dress_hours := 3 -- hours spent ironing dresses

-- Definition to convert hours to minutes
def hours_to_minutes (hours: Int) : Int :=
  hours * 60

-- Definition of the total number of pieces of clothes ironed by Eliza
def total_pieces_iron (blouse_time dress_time blouse_hours dress_hours: Int) : Int :=
  let blouses := hours_to_minutes(blouse_hours) / blouse_time
  let dresses := hours_to_minutes(dress_hours) / dress_time
  blouses + dresses

-- The proof statement
theorem eliza_iron_total : total_pieces_iron blouse_time dress_time blouse_hours dress_hours = 17 :=
by 
  -- To be filled in with the actual proof
  sorry

end eliza_iron_total_l104_104191


namespace gcd_paving_courtyard_l104_104157

theorem gcd_paving_courtyard :
  Nat.gcd 378 595 = 7 :=
by
  sorry

end gcd_paving_courtyard_l104_104157


namespace total_bins_used_l104_104515

def bins_of_soup : ℝ := 0.12
def bins_of_vegetables : ℝ := 0.12
def bins_of_pasta : ℝ := 0.5

theorem total_bins_used : bins_of_soup + bins_of_vegetables + bins_of_pasta = 0.74 :=
by
  sorry

end total_bins_used_l104_104515


namespace range_of_a_l104_104546

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 0 < x → 2 * x + 1 / x - a > 0) → a < 2 * Real.sqrt 2 :=
by
  intro h
  sorry

end range_of_a_l104_104546


namespace lcm_of_36_and_105_l104_104219

theorem lcm_of_36_and_105 : Nat.lcm 36 105 = 1260 := by
  sorry

end lcm_of_36_and_105_l104_104219


namespace arrangeable_sequence_l104_104929

theorem arrangeable_sequence (n : Fin 2017 → ℤ) :
  (∀ i : Fin 2017, ∃ (perm : Fin 5 → Fin 5),
    let a := n ((i + perm 0) % 2017)
    let b := n ((i + perm 1) % 2017)
    let c := n ((i + perm 2) % 2017)
    let d := n ((i + perm 3) % 2017)
    let e := n ((i + perm 4) % 2017)
    a - b + c - d + e = 29) →
  (∀ i : Fin 2017, n i = 29) :=
by
  sorry

end arrangeable_sequence_l104_104929


namespace simplify_product_l104_104115

theorem simplify_product (b : ℤ) : (1 * 2 * b * 3 * b^2 * 4 * b^3 * 5 * b^4 * 6 * b^5) = 720 * b^15 := by
  sorry

end simplify_product_l104_104115


namespace cost_price_decrease_proof_l104_104871

theorem cost_price_decrease_proof (x y : ℝ) (a : ℝ) (h1 : y - x = x * a / 100)
    (h2 : y = (1 + a / 100) * x)
    (h3 : y - 0.9 * x = (0.9 * x * a / 100) + 0.9 * x * 20 / 100) : a = 80 :=
  sorry

end cost_price_decrease_proof_l104_104871


namespace polygon_sides_sum_l104_104595

theorem polygon_sides_sum (n : ℕ) (h : (n - 2) * 180 = 1260) : n = 9 :=
by
  sorry

end polygon_sides_sum_l104_104595


namespace girl_speed_l104_104655

theorem girl_speed (distance time : ℝ) (h₁ : distance = 128) (h₂ : time = 32) : distance / time = 4 := 
by 
  rw [h₁, h₂]
  norm_num

end girl_speed_l104_104655


namespace expressionEquals243_l104_104808

noncomputable def calculateExpression : ℕ :=
  (1 / 3) * 9 * (1 / 27) * 81 * (1 / 243) * 729 * (1 / 2187) * 6561 *
  (1 / 19683) * 59049

theorem expressionEquals243 : calculateExpression = 243 := by
  sorry

end expressionEquals243_l104_104808


namespace first_term_of_geometric_series_l104_104917

theorem first_term_of_geometric_series (a r : ℝ) 
    (h1 : a / (1 - r) = 18) 
    (h2 : a^2 / (1 - r^2) = 72) : 
    a = 72 / 11 := 
  sorry

end first_term_of_geometric_series_l104_104917


namespace range_of_a_l104_104991

def quadratic_function (a x : ℝ) : ℝ := x^2 - 2 * a * x + 1

theorem range_of_a (a : ℝ) (h : ∀ x y : ℝ, 1 ≤ x ∧ x ≤ y → quadratic_function a x ≤ quadratic_function a y) : a ≤ 1 :=
sorry

end range_of_a_l104_104991


namespace extinction_prob_l104_104797

-- Define the probabilities
def p : ℝ := 0.6
def q : ℝ := 0.4

-- Define the base case condition
def v_0 : ℝ := 1

-- Define the extinction probability function
def v : ℕ → ℝ
| 0 => v_0
| k => p * v (k + 1) + q * v (k - 1)

-- Define the target value for v_1
def v_1 : ℝ := 2 / 3

-- The main theorem: Prove that v 1 = 2 / 3 given the conditions
theorem extinction_prob : v 1 = v_1 := by
  -- Proof will be provided here
  sorry

end extinction_prob_l104_104797


namespace arithmetic_mean_of_pq_is_10_l104_104125

variable (p q r : ℝ)

theorem arithmetic_mean_of_pq_is_10
  (H_mean_qr : (q + r) / 2 = 20)
  (H_r_minus_p : r - p = 20) :
  (p + q) / 2 = 10 := by
  sorry

end arithmetic_mean_of_pq_is_10_l104_104125


namespace probability_N_14_mod_5_is_1_l104_104020

theorem probability_N_14_mod_5_is_1 :
  let total := 1950
  let favorable := 2
  let outcomes := 5
  (favorable / outcomes) = (2 / 5) := by
  sorry

end probability_N_14_mod_5_is_1_l104_104020


namespace linear_or_large_derivative_l104_104887

noncomputable def continuous_differentiable_function (f : ℝ → ℝ) : Prop :=
  ∀ x ∈ Icc (0 : ℝ) 1, ContinuousOn f (Icc (0 : ℝ) 1) ∧ DifferentiableOn ℝ f (Ioo (0 : ℝ) 1)

theorem linear_or_large_derivative (f : ℝ → ℝ) (hf : continuous_differentiable_function f) :
  (∃ a b : ℝ, ∀ x ∈ Icc (0 : ℝ) 1, f x = a * x + b) ∨ (∃ t ∈ Ioo (0 : ℝ) 1, |f 1 - f 0| < |f' t|) :=
sorry

end linear_or_large_derivative_l104_104887


namespace max_distinct_fans_l104_104643

theorem max_distinct_fans : 
  let sectors := 6,
      initial_configs := 2^sectors,
      unchanged_configs := 8,
      unique_configs := (initial_configs - unchanged_configs) / 2 + unchanged_configs
  in unique_configs = 36 :=
by
  let sectors := 6
  let initial_configs := 2^sectors
  let unchanged_configs := 8
  let unique_configs := (initial_configs - unchanged_configs) / 2 + unchanged_configs
  have h : unique_configs = 36 := by sorry
  exact h

end max_distinct_fans_l104_104643


namespace problem2_l104_104692

theorem problem2 (x y : ℝ) (h1 : x^2 + x * y = 3) (h2 : x * y + y^2 = -2) : 
  2 * x^2 - x * y - 3 * y^2 = 12 := 
by 
  sorry

end problem2_l104_104692


namespace each_car_has_4_wheels_l104_104190
-- Import necessary libraries

-- Define the conditions
def number_of_guests := 40
def number_of_parent_cars := 2
def wheels_per_parent_car := 4
def number_of_guest_cars := 10
def total_wheels := 48
def parent_car_wheels := number_of_parent_cars * wheels_per_parent_car
def guest_car_wheels := total_wheels - parent_car_wheels

-- Define the proposition to prove
theorem each_car_has_4_wheels : (guest_car_wheels / number_of_guest_cars) = 4 :=
by
  sorry

end each_car_has_4_wheels_l104_104190


namespace calculate_fraction_pow_l104_104959

theorem calculate_fraction_pow :
  (0.5 ^ 4) / (0.05 ^ 3) = 500 := 
  sorry

end calculate_fraction_pow_l104_104959


namespace hilt_books_transaction_difference_l104_104108

noncomputable def total_cost_paid (original_price : ℝ) (num_first_books : ℕ) (discount1 : ℝ) (num_second_books : ℕ) (discount2 : ℝ) : ℝ :=
  let cost_first_books := num_first_books * original_price * (1 - discount1)
  let cost_second_books := num_second_books * original_price * (1 - discount2)
  cost_first_books + cost_second_books

noncomputable def total_sale_amount (sale_price : ℝ) (interest_rate : ℝ) (num_books : ℕ) : ℝ :=
  let compounded_price := sale_price * (1 + interest_rate) ^ 1
  compounded_price * num_books

theorem hilt_books_transaction_difference : 
  let original_price := 11
  let num_first_books := 10
  let discount1 := 0.20
  let num_second_books := 5
  let discount2 := 0.25
  let sale_price := 25
  let interest_rate := 0.05
  let num_books := 15
  total_sale_amount sale_price interest_rate num_books - total_cost_paid original_price num_first_books discount1 num_second_books discount2 = 264.50 :=
by
  sorry

end hilt_books_transaction_difference_l104_104108


namespace max_value_of_a_plus_b_l104_104331

theorem max_value_of_a_plus_b (a b : ℕ) 
  (h : 5 * a + 19 * b = 213) : a + b ≤ 37 :=
  sorry

end max_value_of_a_plus_b_l104_104331


namespace perp_bisector_b_value_l104_104760

theorem perp_bisector_b_value : ∃ b : ℝ, (∀ (x y : ℝ), x + y = b) ∧ (x + y = b) ∧ (x = (-1) ∧ y = 2) ∧ (x = 3 ∧ y = 8) := sorry

end perp_bisector_b_value_l104_104760


namespace derek_dogs_l104_104819

theorem derek_dogs (d c : ℕ) (h1 : d = 90) 
  (h2 : c = d / 3) 
  (h3 : c + 210 = 2 * (d + 120 - d)) : 
  d + 120 - d = 120 :=
by
  sorry

end derek_dogs_l104_104819


namespace faye_earned_total_money_l104_104197

def bead_necklaces : ℕ := 3
def gem_necklaces : ℕ := 7
def price_per_necklace : ℕ := 7

theorem faye_earned_total_money :
  (bead_necklaces + gem_necklaces) * price_per_necklace = 70 :=
by
  sorry

end faye_earned_total_money_l104_104197


namespace dorothy_money_left_l104_104513

-- Define the conditions
def annual_income : ℝ := 60000
def tax_rate : ℝ := 0.18

-- Define the calculation of the amount of money left after paying taxes
def money_left (income : ℝ) (rate : ℝ) : ℝ :=
  income - (rate * income)

-- State the main theorem to prove
theorem dorothy_money_left :
  money_left annual_income tax_rate = 49200 := 
by
  sorry

end dorothy_money_left_l104_104513


namespace common_ratio_of_gp_l104_104632

noncomputable def geometric_sum (a r : ℝ) (n : ℕ) : ℝ :=
if r = 1 then a * n else a * (1 - r^n) / (1 - r)

theorem common_ratio_of_gp (a : ℝ) (r : ℝ) (h : geometric_sum a r 6 / geometric_sum a r 3 = 28) : r = 3 :=
by
  sorry

end common_ratio_of_gp_l104_104632


namespace findYears_l104_104152

def totalInterest (n : ℕ) : ℕ :=
  24 * n + 70 * n

theorem findYears (n : ℕ) : totalInterest n = 350 → n = 4 := 
sorry

end findYears_l104_104152


namespace simplify_product_l104_104116

theorem simplify_product (b : ℤ) : (1 * 2 * b * 3 * b^2 * 4 * b^3 * 5 * b^4 * 6 * b^5) = 720 * b^15 := by
  sorry

end simplify_product_l104_104116


namespace range_of_a_mono_increase_l104_104391

def f (x : ℝ) := (1 / 2) * x^2 + 2 * x - 2 * log x

theorem range_of_a_mono_increase :
  (∀ x > 0, derivative ℝ f x ≥ 0) ↔ (a ≤ 0) := sorry

end range_of_a_mono_increase_l104_104391


namespace find_angle_A_l104_104407

variables {A B C a b c : ℝ}
variables {triangle_ABC : (2 * b - c) * (Real.cos A) = a * (Real.cos C)}

theorem find_angle_A (h : (2 * b - c) * (Real.cos A) = a * (Real.cos C)) : A = Real.pi / 3 :=
by
  sorry

end find_angle_A_l104_104407


namespace simplify_fraction_l104_104899

theorem simplify_fraction (a : ℝ) (h : a ≠ 2) : 
  (a^2 / (a - 2) - (4 * a - 4) / (a - 2)) = a - 2 :=
  sorry

end simplify_fraction_l104_104899


namespace neg_p_equiv_l104_104995

theorem neg_p_equiv (p : Prop) : 
  (¬ ∃ n : ℕ, n^2 > 2^n) ↔ (∀ n : ℕ, n^2 ≤ 2^n) :=
by sorry

end neg_p_equiv_l104_104995


namespace sum_of_3x3_matrix_arithmetic_eq_45_l104_104057

-- Statement: Prove that the sum of all nine elements of a 3x3 matrix, where each row and each column forms an arithmetic sequence and the middle element a_{22} = 5, is 45
theorem sum_of_3x3_matrix_arithmetic_eq_45 
  (matrix : ℤ → ℤ → ℤ)
  (arithmetic_row : ∀ i, matrix i 0 + matrix i 1 + matrix i 2 = 3 * matrix i 1)
  (arithmetic_col : ∀ j, matrix 0 j + matrix 1 j + matrix 2 j = 3 * matrix 1 j)
  (middle_elem : matrix 1 1 = 5) : 
  (matrix 0 0 + matrix 0 1 + matrix 0 2 + matrix 1 0 + matrix 1 1 + matrix 1 2 + matrix 2 0 + matrix 2 1 + matrix 2 2) = 45 :=
by
  sorry -- proof to be provided

end sum_of_3x3_matrix_arithmetic_eq_45_l104_104057


namespace brian_video_watching_time_l104_104504

/--
Brian watches a 4-minute video of cats.
Then he watches a video twice as long as the cat video involving dogs.
Finally, he watches a video on gorillas that's twice as long as the combined duration of the first two videos.
Prove that Brian spends a total of 36 minutes watching animal videos.
-/
theorem brian_video_watching_time (cat_video dog_video gorilla_video : ℕ) 
  (h₁ : cat_video = 4) 
  (h₂ : dog_video = 2 * cat_video) 
  (h₃ : gorilla_video = 2 * (cat_video + dog_video)) : 
  cat_video + dog_video + gorilla_video = 36 := by
  sorry

end brian_video_watching_time_l104_104504


namespace modulo_power_l104_104271

theorem modulo_power (a n : ℕ) (p : ℕ) (hn_pos : 0 < n) (hp_odd : p % 2 = 1)
  (hp_prime : Nat.Prime p) (h : a^p ≡ 1 [MOD p^n]) : a ≡ 1 [MOD p^(n-1)] :=
by
  sorry

end modulo_power_l104_104271


namespace hyperbola_condition_l104_104229

-- Definitions and hypotheses
def is_hyperbola (m n : ℝ) (x y : ℝ) : Prop := m * x^2 - n * y^2 = 1

-- Statement of the problem
theorem hyperbola_condition (m n : ℝ) : (∃ x y : ℝ, is_hyperbola m n x y) ↔ m * n > 0 :=
by sorry

end hyperbola_condition_l104_104229


namespace small_bottle_sold_percentage_l104_104016

-- Definitions for initial conditions
def small_bottles_initial : ℕ := 6000
def large_bottles_initial : ℕ := 15000
def large_bottle_sold_percentage : ℝ := 0.14
def total_remaining_bottles : ℕ := 18180

-- The statement we need to prove
theorem small_bottle_sold_percentage :
  ∃ k : ℝ, (0 ≤ k ∧ k ≤ 100) ∧
  (small_bottles_initial - (k / 100) * small_bottles_initial + 
   large_bottles_initial - large_bottle_sold_percentage * large_bottles_initial = total_remaining_bottles) ∧
  (k = 12) :=
sorry

end small_bottle_sold_percentage_l104_104016


namespace geometric_series_first_term_l104_104919

theorem geometric_series_first_term (a r : ℝ) 
  (h1 : a / (1 - r) = 18) 
  (h2 : a^2 / (1 - r^2) = 72) : 
  a = 7.2 :=
by
  sorry

end geometric_series_first_term_l104_104919


namespace contractor_laborers_l104_104932

theorem contractor_laborers (x : ℕ) (h1 : 15 * x = 20 * (x - 5)) : x = 20 :=
by sorry

end contractor_laborers_l104_104932


namespace four_integers_sum_product_odd_impossible_l104_104094

theorem four_integers_sum_product_odd_impossible (a b c d : ℤ) :
  ¬ (a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1 ∧ d % 2 = 1 ∧ 
     (a + b + c + d) % 2 = 1) :=
by
  sorry

end four_integers_sum_product_odd_impossible_l104_104094


namespace sprockets_produced_by_machines_l104_104568

noncomputable def machine_sprockets (t : ℝ) : Prop :=
  let machineA_hours := t + 10
  let machineA_rate := 4
  let machineA_sprockets := machineA_hours * machineA_rate
  let machineB_hours := t
  let machineB_rate := 4.4
  let machineB_sprockets := machineB_hours * machineB_rate
  machineA_sprockets = 440 ∧ machineB_sprockets = 440

theorem sprockets_produced_by_machines (t : ℝ) (h : machine_sprockets t) : t = 100 :=
  sorry

end sprockets_produced_by_machines_l104_104568


namespace dentist_age_l104_104285

theorem dentist_age (x : ℝ) (h : (x - 8) / 6 = (x + 8) / 10) : x = 32 :=
  by
  sorry

end dentist_age_l104_104285


namespace find_functional_form_l104_104829

theorem find_functional_form (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (f x + y) = 2 * x + f (f y - x)) →
  ∃ c : ℝ, ∀ x : ℝ, f x = x + c :=
by
  sorry

end find_functional_form_l104_104829


namespace swimming_speed_in_still_water_l104_104013

theorem swimming_speed_in_still_water :
  ∀ (speed_of_water person's_speed time distance: ℝ),
  speed_of_water = 8 →
  time = 1.5 →
  distance = 12 →
  person's_speed - speed_of_water = distance / time →
  person's_speed = 16 :=
by
  intro speed_of_water person's_speed time distance hw ht hd heff
  rw [hw, ht, hd] at heff
  -- steps to isolate person's_speed should be done here, but we leave it as sorry
  sorry

end swimming_speed_in_still_water_l104_104013


namespace businesses_can_apply_l104_104807

-- Define conditions
def total_businesses : ℕ := 72
def businesses_fired : ℕ := 36 -- Half of total businesses (72 / 2)
def businesses_quit : ℕ := 24 -- One third of total businesses (72 / 3)

-- Theorem: Number of businesses Brandon can still apply to
theorem businesses_can_apply : (total_businesses - (businesses_fired + businesses_quit)) = 12 := 
by
  sorry

end businesses_can_apply_l104_104807


namespace total_surface_area_l104_104766

theorem total_surface_area (a b c : ℝ) 
  (h1 : a + b + c = 45) 
  (h2 : a^2 + b^2 + c^2 = 625) : 
  2 * (a * b + b * c + c * a) = 1400 :=
sorry

end total_surface_area_l104_104766


namespace veronica_cans_of_food_is_multiple_of_4_l104_104318

-- Definitions of the given conditions
def number_of_water_bottles : ℕ := 20
def number_of_kits : ℕ := 4

-- Proof statement
theorem veronica_cans_of_food_is_multiple_of_4 (F : ℕ) :
  F % number_of_kits = 0 :=
sorry

end veronica_cans_of_food_is_multiple_of_4_l104_104318


namespace fraction_of_jam_eaten_for_dinner_l104_104882

-- Define the problem
theorem fraction_of_jam_eaten_for_dinner :
  ∃ (J : ℝ) (x : ℝ), 
  J > 0 ∧
  (1 / 3) * J + (x * (2 / 3) * J) + (4 / 7) * J = J ∧
  x = 1 / 7 :=
by
  sorry

end fraction_of_jam_eaten_for_dinner_l104_104882


namespace original_price_of_wand_l104_104999

-- Definitions as per the conditions
def price_paid (paid : Real) := paid = 8
def fraction_of_original (fraction : Real) := fraction = 1 / 8

-- Question and correct answer put as a theorem to prove
theorem original_price_of_wand (paid : Real) (fraction : Real) 
  (h1 : price_paid paid) (h2 : fraction_of_original fraction) : 
  (paid / fraction = 64) := 
by
  -- This 'sorry' indicates where the actual proof would go.
  sorry

end original_price_of_wand_l104_104999


namespace speed_increase_l104_104002

theorem speed_increase (v_initial: ℝ) (t_initial: ℝ) (t_new: ℝ) :
  v_initial = 60 → t_initial = 1 → t_new = 0.5 →
  v_new = (1 / (t_new / 60)) →
  v_increase = v_new - v_initial →
  v_increase = 60 :=
by
  sorry

end speed_increase_l104_104002


namespace tan_alpha_is_three_halves_l104_104979

theorem tan_alpha_is_three_halves (α : ℝ) (h : Real.tan (α - 5 * Real.pi / 4) = 1 / 5) : 
  Real.tan α = 3 / 2 :=
by
  sorry

end tan_alpha_is_three_halves_l104_104979


namespace transformed_independence_l104_104273

variables {E : Type*} {G : Type*} {n : ℕ}
variables (X : Fin n → E) (g : Fin n → E → G)
variables (measurable_spaces_E : Fin n → MeasurableSpace E)
variables (measurable_spaces_G : Fin n → MeasurableSpace G)
variables (independent_X : independent (λ i, σ X i))
variables (measurability : ∀ i, Measurable (g i))

theorem transformed_independence :
  independent (λ i, σ (g i (X i))) :=
sorry

end transformed_independence_l104_104273


namespace judy_pencil_cost_l104_104883

theorem judy_pencil_cost 
  (pencils_per_week : ℕ)
  (days_per_week : ℕ)
  (pack_cost : ℕ)
  (pack_size : ℕ)
  (total_days : ℕ)
  (pencil_usage : pencils_per_week = 10)
  (school_days : days_per_week = 5)
  (cost_per_pack : pack_cost = 4)
  (pencils_per_pack : pack_size = 30)
  (duration : total_days = 45) : 
  ∃ (total_cost : ℕ), total_cost = 12 :=
sorry

end judy_pencil_cost_l104_104883


namespace range_of_a_l104_104704

def p (x : ℝ) : Prop := 1 / 2 ≤ x ∧ x ≤ 1

def q (x a : ℝ) : Prop := (x - a) * (x - a - 1) ≤ 0

theorem range_of_a (a x : ℝ) 
  (hp : ∀ x, ¬ (1 / 2 ≤ x ∧ x ≤ 1) → (x < 1 / 2 ∨ x > 1))
  (hq : ∀ x, ¬ ((x - a) * (x - a - 1) ≤ 0) → (x < a ∨ x > a + 1))
  (h : ∀ x, (q x a) → (p x)) :
  0 ≤ a ∧ a ≤ 1 / 2 := 
sorry

end range_of_a_l104_104704


namespace arrange_abc_l104_104534

theorem arrange_abc (a b c : ℝ) (h1 : a = Real.log 2 / Real.log 2)
                               (h2 : b = Real.sqrt 2)
                               (h3 : c = Real.cos ((3 / 4) * Real.pi)) :
  c < a ∧ a < b :=
by
  sorry

end arrange_abc_l104_104534


namespace range_of_a_l104_104062

noncomputable def A (x : ℝ) : Prop := (3 * x) / (x + 1) ≤ 2
noncomputable def B (x a : ℝ) : Prop := a - 2 < x ∧ x < 2 * a + 1

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, A x ↔ B x a) ↔ (1 / 2 < a ∧ a ≤ 1) := by
sorry

end range_of_a_l104_104062


namespace oliver_final_amount_l104_104742

variable (initial_amount saved_amount spent_on_frisbee spent_on_puzzle gift : ℕ)

def final_amount_after_transactions (initial_amount saved_amount spent_on_frisbee spent_on_puzzle gift : ℕ) : ℕ :=
  initial_amount + saved_amount - (spent_on_frisbee + spent_on_puzzle) + gift

theorem oliver_final_amount :
  final_amount_after_transactions 9 5 4 3 8 = 15 :=
by
  -- We can fill in the exact calculations here to provide the proof.
  sorry

end oliver_final_amount_l104_104742


namespace lattice_points_in_region_l104_104010

theorem lattice_points_in_region :
  ∃ n : ℕ, n = 12 ∧ 
  ( ∀ x y : ℤ, (y = x ∨ y = -x ∨ y = -x^2 + 4) → n = 12) :=
by
  sorry

end lattice_points_in_region_l104_104010


namespace minimum_degree_g_l104_104559

-- Define the degree function for polynomials
noncomputable def degree (p : Polynomial ℤ) : ℕ := p.natDegree

-- Declare the variables and conditions for the proof
variables (f g h : Polynomial ℤ)
variables (deg_f : degree f = 10) (deg_h : degree h = 12)
variable (eqn : 2 * f + 5 * g = h)

-- State the main theorem for the problem
theorem minimum_degree_g : degree g ≥ 12 :=
    by sorry -- Proof to be provided

end minimum_degree_g_l104_104559


namespace work_done_by_A_alone_l104_104712

theorem work_done_by_A_alone (Wb : ℝ) (Wa : ℝ) (D : ℝ) :
  Wa = 3 * Wb →
  (Wb + Wa) * 18 = D →
  D = 72 → 
  (D / Wa) = 24 := 
by
  intros h1 h2 h3
  sorry

end work_done_by_A_alone_l104_104712


namespace total_corn_cobs_l104_104787

-- Definitions for the conditions
def rows_first_field : ℕ := 13
def rows_second_field : ℕ := 16
def cobs_per_row : ℕ := 4

-- Statement to prove
theorem total_corn_cobs : (rows_first_field * cobs_per_row + rows_second_field * cobs_per_row) = 116 :=
by sorry

end total_corn_cobs_l104_104787


namespace arith_seq_sum_20_l104_104059

theorem arith_seq_sum_20 (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h_arith : ∀ n, a n = a 0 + n * a 1)
  (h_sum : ∀ n, S n = n / 2 * (2 * a 0 + (n - 1) * a 1))
  (h2 : a 1 = 0)
  (h3 : a 3 * a 5 = 12)
  (ha0_pos : a 0 > 0) :
  S 20 = -340 :=
sorry

end arith_seq_sum_20_l104_104059


namespace total_students_sum_is_90_l104_104718

theorem total_students_sum_is_90:
  ∃ (x y z : ℕ), 
  (80 * x - 100 = 92 * (x - 5)) ∧
  (75 * y - 150 = 85 * (y - 6)) ∧
  (70 * z - 120 = 78 * (z - 4)) ∧
  (x + y + z = 90) :=
by
  sorry

end total_students_sum_is_90_l104_104718


namespace describe_set_T_l104_104888

theorem describe_set_T:
  ( ∀ (x y : ℝ), ((x + 2 = 4 ∧ y - 5 ≤ 4) ∨ (y - 5 = 4 ∧ x + 2 ≤ 4) ∨ (x + 2 = y - 5 ∧ 4 ≤ x + 2)) →
    ( ∃ (x y : ℝ), x = 2 ∧ y ≤ 9 ∨ y = 9 ∧ x ≤ 2 ∨ y = x + 7 ∧ x ≥ 2 ∧ y ≥ 9) ) :=
sorry

end describe_set_T_l104_104888


namespace calc_fraction_product_l104_104960

theorem calc_fraction_product : 
  (7 / 4) * (8 / 14) * (14 / 8) * (16 / 40) * (35 / 20) * (18 / 45) * (49 / 28) * (32 / 64) = 49 / 200 := 
by sorry

end calc_fraction_product_l104_104960


namespace great_white_shark_teeth_l104_104019

theorem great_white_shark_teeth :
  let tiger_shark_teeth := 180
  let hammerhead_shark_teeth := tiger_shark_teeth / 6
  let great_white_shark_teeth := 2 * (tiger_shark_teeth + hammerhead_shark_teeth)
  in great_white_shark_teeth = 420 := by
  sorry

end great_white_shark_teeth_l104_104019


namespace factorize_2mn_cube_arithmetic_calculation_l104_104941

-- Problem 1: Factorization problem
theorem factorize_2mn_cube (m n : ℝ) : 
  2 * m^3 * n - 8 * m * n^3 = 2 * m * n * (m + 2 * n) * (m - 2 * n) :=
by sorry

-- Problem 2: Arithmetic calculation problem
theorem arithmetic_calculation : 
  |1 - Real.sqrt 3| + 3 * Real.tan (Real.pi / 6) - ((Real.pi - 3)^0) + (-1/3)⁻¹ = 2 * Real.sqrt 3 - 5 :=
by sorry

end factorize_2mn_cube_arithmetic_calculation_l104_104941


namespace min_ratio_of_integers_l104_104566

theorem min_ratio_of_integers (x y : ℕ) (hx : 50 < x) (hy : 50 < y) (h_mean : x + y = 130) : 
  x = 51 → y = 79 → x / y = 51 / 79 := by
  sorry

end min_ratio_of_integers_l104_104566


namespace sufficient_but_not_necessary_decreasing_l104_104674

def is_decreasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ x ∈ I, ∀ y ∈ I, x ≤ y → f y ≤ f x

def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 6 * m * x + 6

theorem sufficient_but_not_necessary_decreasing (m : ℝ) :
  m = 1 → is_decreasing_on (f m) (Set.Iic 3) :=
by
  intros h
  rw [h]
  sorry

end sufficient_but_not_necessary_decreasing_l104_104674


namespace contrapositive_x_squared_eq_one_l104_104298

theorem contrapositive_x_squared_eq_one (x : ℝ) : 
  (x^2 = 1 → x = 1 ∨ x = -1) ↔ (x ≠ 1 ∧ x ≠ -1 → x^2 ≠ 1) := by
  sorry

end contrapositive_x_squared_eq_one_l104_104298


namespace Delaney_missed_bus_by_l104_104178

def time_in_minutes (hours : ℕ) (minutes : ℕ) : ℕ := hours * 60 + minutes

def Delaney_start_time : ℕ := time_in_minutes 7 50
def bus_departure_time : ℕ := time_in_minutes 8 0
def travel_duration : ℕ := 30

theorem Delaney_missed_bus_by :
  Delaney_start_time + travel_duration - bus_departure_time = 20 :=
by
  sorry

end Delaney_missed_bus_by_l104_104178


namespace convert_base10_to_base9_l104_104671

theorem convert_base10_to_base9 : 
  (2 * 9^3 + 6 * 9^2 + 7 * 9^1 + 7 * 9^0) = 2014 :=
by
  sorry

end convert_base10_to_base9_l104_104671


namespace oliver_final_amount_l104_104743

variable (initial_amount saved_amount spent_on_frisbee spent_on_puzzle gift : ℕ)

def final_amount_after_transactions (initial_amount saved_amount spent_on_frisbee spent_on_puzzle gift : ℕ) : ℕ :=
  initial_amount + saved_amount - (spent_on_frisbee + spent_on_puzzle) + gift

theorem oliver_final_amount :
  final_amount_after_transactions 9 5 4 3 8 = 15 :=
by
  -- We can fill in the exact calculations here to provide the proof.
  sorry

end oliver_final_amount_l104_104743


namespace factor_polynomial_l104_104195

theorem factor_polynomial (x : ℝ) : 
  54 * x ^ 5 - 135 * x ^ 9 = 27 * x ^ 5 * (2 - 5 * x ^ 4) :=
by 
  sorry

end factor_polynomial_l104_104195


namespace fraction_evaluation_l104_104962

theorem fraction_evaluation :
  (2 + 4 - 8 + 16 + 32 - 64 + 128 : ℚ) / (4 + 8 - 16 + 32 + 64 - 128 + 256) = 1 / 2 :=
by
  sorry

end fraction_evaluation_l104_104962


namespace folded_triangle_sqrt_equals_l104_104790

noncomputable def folded_triangle_length_squared (s : ℕ) (d : ℕ) : ℚ :=
  let x := (2 * s * s - 2 * d * s)/(2 * d)
  let y := (2 * s * s - 2 * (s - d) * s)/(2 * (s - d))
  x * x - x * y + y * y

theorem folded_triangle_sqrt_equals :
  folded_triangle_length_squared 15 11 = (60118.9025 / 1681 : ℚ) := sorry

end folded_triangle_sqrt_equals_l104_104790


namespace prove_x_minus_y_squared_l104_104243

noncomputable section

variables {x y a b : ℝ}

theorem prove_x_minus_y_squared (h1 : x * y = b) (h2 : x / y + y / x = a) : (x - y) ^ 2 = a * b - 2 * b := 
  sorry

end prove_x_minus_y_squared_l104_104243


namespace math_problem_l104_104411

noncomputable def students_scoring_above_110 
  (n : ℕ)  -- number of students
  (mu sigma : ℝ)  -- mean and standard deviation of the normal distribution
  (P_90_100 : ℝ)  -- Probability that a student's score is between 90 and 100
  (h_distribution : ∀ x, (λ ξ, P (ξ ≤ x)) = x ↦ (1/2) * (1 + erf (x - mu) / (sigma * sqrt 2))) : ℝ :=
  let P_100_110 := P_90_100 in
  let P_above_110 := 1 - (P_90_100 + P_100_110) in
  n * P_above_110

theorem math_problem 
  (h1 : ∀ x, (λ ξ, P (ξ ≤ x)) = x ↦ (1/2) * (1 + erf (x - 100) / (10 * sqrt 2))) 
  (h2 : P (λ ξ, 90 ≤ ξ ∧ ξ ≤ 100) = 0.3) :
  students_scoring_above_110 50 100 10 0.3 h1 = 10 :=
sorry

end math_problem_l104_104411


namespace words_memorized_on_fourth_day_l104_104708

-- Definitions for the conditions
def first_three_days_words (k : ℕ) : ℕ := 3 * k
def last_four_days_words (k : ℕ) : ℕ := 4 * k
def fourth_day_words (k : ℕ) (a : ℕ) : ℕ := a
def last_three_days_words (k : ℕ) (a : ℕ) : ℕ := last_four_days_words k - a

-- Problem Statement
theorem words_memorized_on_fourth_day {k a : ℕ} (h1 : first_three_days_words k + last_four_days_words k > 100)
    (h2 : first_three_days_words k * 6 = 5 * (4 * k - a))
    (h3 : 21 * (2 * k / 3) = 100) : 
    a = 10 :=
by 
  sorry

end words_memorized_on_fourth_day_l104_104708


namespace prove_intersection_area_is_correct_l104_104490

noncomputable def octahedron_intersection_area 
  (side_length : ℝ) (cut_height_factor : ℝ) : ℝ :=
  have height_triangular_face := Real.sqrt (side_length^2 - (side_length / 2)^2)
  have plane_height := cut_height_factor * height_triangular_face
  have proportional_height := plane_height / height_triangular_face
  let new_side_length := proportional_height * side_length
  have hexagon_area := (3 * Real.sqrt 3 / 2) * (new_side_length^2) / 2 
  (3 * Real.sqrt 3 / 2) * (new_side_length^2)

theorem prove_intersection_area_is_correct 
  : 
  octahedron_intersection_area 2 (3 / 4) = 9 * Real.sqrt 3 / 8 :=
  sorry 

example : 9 + 3 + 8 = 20 := 
  by rfl

end prove_intersection_area_is_correct_l104_104490


namespace correct_linear_regression_statement_l104_104881

-- Definitions based on the conditions:
def linear_regression (b a e : ℝ) (x : ℝ) : ℝ := b * x + a + e

def statement_A (b a e : ℝ) (x : ℝ) : Prop := linear_regression b a e x = b * x + a + e

def statement_B (b a e : ℝ) (x : ℝ) : Prop := ∀ x1 x2, (linear_regression b a e x1 ≠ linear_regression b a e x2) → (x1 ≠ x2)

def statement_C (b a e : ℝ) (x : ℝ) : Prop := ∃ (other_factors : ℝ), linear_regression b a e x = b * x + a + other_factors + e

def statement_D (b a e : ℝ) (x : ℝ) : Prop := (e ≠ 0) → false

-- The proof statement
theorem correct_linear_regression_statement (b a e : ℝ) (x : ℝ) :
  (statement_C b a e x) :=
sorry

end correct_linear_regression_statement_l104_104881


namespace Delaney_missed_bus_by_l104_104179

def time_in_minutes (hours : ℕ) (minutes : ℕ) : ℕ := hours * 60 + minutes

def Delaney_start_time : ℕ := time_in_minutes 7 50
def bus_departure_time : ℕ := time_in_minutes 8 0
def travel_duration : ℕ := 30

theorem Delaney_missed_bus_by :
  Delaney_start_time + travel_duration - bus_departure_time = 20 :=
by
  sorry

end Delaney_missed_bus_by_l104_104179


namespace max_ab_value_l104_104535

noncomputable def max_ab (a b : ℝ) : ℝ :=
  if (a > 0 ∧ b > 0 ∧ 2 * a + b = 1) then a * b else 0

theorem max_ab_value (a b : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_sum : 2 * a + b = 1) :
  max_ab a b = 1 / 8 := sorry

end max_ab_value_l104_104535


namespace donna_fully_loaded_truck_weight_l104_104967

-- Define conditions
def empty_truck_weight : ℕ := 12000
def soda_crate_weight : ℕ := 50
def soda_crate_count : ℕ := 20
def dryer_weight : ℕ := 3000
def dryer_count : ℕ := 3

-- Calculate derived weights
def soda_total_weight : ℕ := soda_crate_weight * soda_crate_count
def fresh_produce_weight : ℕ := 2 * soda_total_weight
def dryer_total_weight : ℕ := dryer_weight * dryer_count

-- Define target weight of fully loaded truck
def fully_loaded_truck_weight : ℕ :=
  empty_truck_weight + soda_total_weight + fresh_produce_weight + dryer_total_weight

-- State and prove the theorem
theorem donna_fully_loaded_truck_weight :
  fully_loaded_truck_weight = 24000 :=
by
  -- Provide necessary calculations and proof steps if needed
  sorry

end donna_fully_loaded_truck_weight_l104_104967


namespace largest_number_among_given_l104_104822

theorem largest_number_among_given (
  A B C D E : ℝ
) (hA : A = 0.936)
  (hB : B = 0.9358)
  (hC : C = 0.9361)
  (hD : D = 0.935)
  (hE : E = 0.921):
  C = max A (max B (max C (max D E))) :=
by
  sorry

end largest_number_among_given_l104_104822


namespace count_congruent_to_4_mod_7_l104_104861

theorem count_congruent_to_4_mod_7 : 
  ∃ (n : ℕ), 
  n = 71 ∧ 
  ∀ k : ℕ, 0 ≤ k ∧ k ≤ 70 → ∃ m : ℕ, m = 4 + 7 * k ∧ m < 500 := 
by
  sorry

end count_congruent_to_4_mod_7_l104_104861


namespace sqrt_sqrt_16_eq_pm_2_l104_104140

theorem sqrt_sqrt_16_eq_pm_2 : sqrt (sqrt 16) = 2 ∨ sqrt (sqrt 16) = -2 :=
by
  sorry

end sqrt_sqrt_16_eq_pm_2_l104_104140


namespace ten_row_geometric_figure_has_286_pieces_l104_104343

noncomputable def rods (rows : ℕ) : ℕ := 3 * rows * (rows + 1) / 2
noncomputable def connectors (rows : ℕ) : ℕ := (rows +1) * (rows + 2) / 2
noncomputable def squares (rows : ℕ) : ℕ := rows * (rows + 1) / 2

theorem ten_row_geometric_figure_has_286_pieces :
    rods 10 + connectors 10 + squares 10 = 286 := by
  sorry

end ten_row_geometric_figure_has_286_pieces_l104_104343


namespace f_neg_a_l104_104911

noncomputable def f (x : ℝ) : ℝ := x^3 + Real.sin x + 1

theorem f_neg_a (a : ℝ) (h : f a = 2) : f (-a) = 0 := by
  sorry

end f_neg_a_l104_104911


namespace find_the_number_l104_104946

theorem find_the_number :
  ∃ N : ℝ, ((4/5 : ℝ) * 25 = 20) ∧ (0.40 * N = 24) ∧ (N = 60) :=
by
  sorry

end find_the_number_l104_104946


namespace odd_difference_even_odd_l104_104400

theorem odd_difference_even_odd (a b : ℤ) (ha : a % 2 = 0) (hb : b % 2 = 1) : (a - b) % 2 = 1 :=
sorry

end odd_difference_even_odd_l104_104400


namespace curve_defined_by_r_eq_4_is_circle_l104_104518

theorem curve_defined_by_r_eq_4_is_circle : ∀ θ : ℝ, ∃ r : ℝ, r = 4 → ∀ θ : ℝ, r = 4 :=
by
  sorry

end curve_defined_by_r_eq_4_is_circle_l104_104518


namespace barbara_removed_114_sheets_l104_104347

/-- Given conditions: -/
def bundles (n : ℕ) := 2 * n
def bunches (n : ℕ) := 4 * n
def heaps (n : ℕ) := 20 * n

/-- Barbara removed certain amounts of paper from the chest of drawers. -/
def total_sheets_removed := bundles 3 + bunches 2 + heaps 5

theorem barbara_removed_114_sheets : total_sheets_removed = 114 := by
  -- proof will be inserted here
  sorry

end barbara_removed_114_sheets_l104_104347


namespace consecutive_numbers_count_l104_104584

theorem consecutive_numbers_count (n : ℕ) 
(avg : ℝ) 
(largest : ℕ) 
(h_avg : avg = 20) 
(h_largest : largest = 23) 
(h_eq : (largest + (largest - (n - 1))) / 2 = avg) : 
n = 7 := 
by 
  sorry

end consecutive_numbers_count_l104_104584


namespace binomial_coeff_ratio_l104_104910

open Nat

theorem binomial_coeff_ratio (n k : ℕ) (h1 : choose n k * 3 = choose n (k + 1))
                             (h2 : choose n (k + 1) * 2 = choose n (k + 2)) :
                             n + k = 13 := by
  sorry

end binomial_coeff_ratio_l104_104910


namespace find_t1_t2_l104_104249

-- Define the vectors a and b
def a (t : ℝ) : ℝ × ℝ := (2, t)
def b : ℝ × ℝ := (1, 2)

-- Define the conditions for t1 and t2
def t1_condition (t1 : ℝ) : Prop := (2 / 1) = (t1 / 2)
def t2_condition (t2 : ℝ) : Prop := (2 * 1 + t2 * 2 = 0)

-- The statement to prove
theorem find_t1_t2 (t1 t2 : ℝ) (h1 : t1_condition t1) (h2 : t2_condition t2) : (t1 = 4) ∧ (t2 = -1) :=
by
  sorry

end find_t1_t2_l104_104249


namespace simplify_expression_l104_104900

theorem simplify_expression (x y : ℝ) (h : x ≠ y) : (x^2 - x * y) / (x - y)^2 = x / (x - y) :=
by sorry

end simplify_expression_l104_104900


namespace delaney_missed_bus_time_l104_104183

def busDepartureTime : Nat := 480 -- 8:00 a.m. = 8 * 60 minutes
def travelTime : Nat := 30 -- 30 minutes
def departureFromHomeTime : Nat := 470 -- 7:50 a.m. = 7 * 60 + 50 minutes

theorem delaney_missed_bus_time :
  (departureFromHomeTime + travelTime - busDepartureTime) = 20 :=
by
  -- proof would go here
  sorry

end delaney_missed_bus_time_l104_104183


namespace equivalent_form_l104_104867

theorem equivalent_form (p q : ℝ) (hp₁ : p ≠ 0) (hp₂ : p ≠ 5) (hq₁ : q ≠ 0) (hq₂ : q ≠ 7) :
  (3/p + 4/q = 1/3) ↔ (p = 9*q/(q - 12)) :=
by
  sorry

end equivalent_form_l104_104867


namespace percentage_difference_l104_104779

variable (p : ℝ) (j : ℝ) (t : ℝ)

def condition_1 := j = 0.75 * p
def condition_2 := t = 0.9375 * p

theorem percentage_difference : (j = 0.75 * p) → (t = 0.9375 * p) → ((t - j) / t * 100 = 20) :=
by
  intros h1 h2
  rw [h1, h2]
  -- This will use the derived steps from the solution, and ultimately show 20
  sorry

end percentage_difference_l104_104779


namespace number_of_matching_pages_l104_104785

theorem number_of_matching_pages : 
  ∃ (n : Nat), n = 13 ∧ ∀ x, 1 ≤ x ∧ x ≤ 63 → (x % 10 = (64 - x) % 10) ↔ x % 10 = 2 ∨ x % 10 = 7 :=
by
  sorry

end number_of_matching_pages_l104_104785


namespace sufficient_not_necessary_l104_104060

variables {a : ℕ → ℝ} {S : ℕ → ℝ}
variables (h_seq : ∀ n, a (n + 1) = a n + (a 1 - a 0))
variables (h_sum : ∀ n, S n = n * (a 1 + a n) / 2)
variables (h_condition : 3 * a 2 = a 5 + 4)

theorem sufficient_not_necessary (h1 : a 1 < 1) : S 4 < 10 :=
sorry

end sufficient_not_necessary_l104_104060


namespace sum_powers_mod_5_l104_104431

theorem sum_powers_mod_5 (n : ℕ) (h : ¬ (n % 4 = 0)) : 
  (1^n + 2^n + 3^n + 4^n) % 5 = 0 :=
by
  sorry

end sum_powers_mod_5_l104_104431


namespace value_of_f_log3_54_l104_104698

noncomputable def f : ℝ → ℝ := sorry

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem value_of_f_log3_54
  (h1 : is_odd f)
  (h2 : ∀ x, f (x + 2) = -1 / f x)
  (h3 : ∀ x, 0 < x ∧ x < 1 → f x = 3 ^ x) :
  f (Real.log 54 / Real.log 3) = -3 / 2 := sorry

end value_of_f_log3_54_l104_104698


namespace max_distinct_fans_l104_104642

theorem max_distinct_fans : 
  let sectors := 6,
      initial_configs := 2^sectors,
      unchanged_configs := 8,
      unique_configs := (initial_configs - unchanged_configs) / 2 + unchanged_configs
  in unique_configs = 36 :=
by
  let sectors := 6
  let initial_configs := 2^sectors
  let unchanged_configs := 8
  let unique_configs := (initial_configs - unchanged_configs) / 2 + unchanged_configs
  have h : unique_configs = 36 := by sorry
  exact h

end max_distinct_fans_l104_104642


namespace find_fraction_value_l104_104691

theorem find_fraction_value (m n : ℝ) (h : 1/m - 1/n = 6) : (m * n) / (m - n) = -1/6 :=
sorry

end find_fraction_value_l104_104691


namespace sqrt_sqrt_sixteen_l104_104138

theorem sqrt_sqrt_sixteen : sqrt (sqrt 16) = 2 ∨ sqrt (sqrt 16) = -2 := by
  sorry

end sqrt_sqrt_sixteen_l104_104138


namespace color_swap_rectangle_l104_104823

theorem color_swap_rectangle 
  (n : ℕ) 
  (square_size : ℕ := 2*n - 1) 
  (colors : Finset ℕ := Finset.range n) 
  (vertex_colors : Fin (square_size + 1) × Fin (square_size + 1) → ℕ) 
  (h_vertex_colors : ∀ v, vertex_colors v ∈ colors) :
  ∃ row, ∃ (v₁ v₂ : Fin (square_size + 1) × Fin (square_size + 1)),
    (v₁.1 = row ∧ v₂.1 = row ∧ v₁ ≠ v₂ ∧
    (∃ r₀ r₁ r₂, r₀ ≠ r₁ ∧ r₁ ≠ r₂ ∧ r₂ ≠ r₀ ∧
    vertex_colors v₁ = vertex_colors (r₀, v₁.2) ∧
    vertex_colors v₂ = vertex_colors (r₀, v₂.2) ∧
    vertex_colors (r₁, v₁.2) = vertex_colors (r₂, v₂.2))) := 
sorry

end color_swap_rectangle_l104_104823


namespace lcm_36_105_l104_104213

theorem lcm_36_105 : Nat.lcm 36 105 = 1260 := by
  sorry

end lcm_36_105_l104_104213


namespace number_of_three_cell_shapes_l104_104648

theorem number_of_three_cell_shapes (x y : ℕ) (h : 3 * x + 4 * y = 22) : x = 6 :=
sorry

end number_of_three_cell_shapes_l104_104648


namespace cube_volume_is_64_l104_104572

theorem cube_volume_is_64 (a : ℕ) (h : (a - 2) * (a + 3) * a = a^3 + 12) : a^3 = 64 := 
  sorry

end cube_volume_is_64_l104_104572


namespace Pam_bags_count_l104_104428

theorem Pam_bags_count :
  ∀ (apples_in_geralds_bag : ℕ) (multiple_factor : ℕ) (total_apples_pam_has : ℕ) (expected_pam_bags : ℕ),
  (apples_in_geralds_bag = 40) →
  (multiple_factor = 3) →
  (total_apples_pam_has = 1200) →
  (expected_pam_bags = 10) →
  let apples_in_pams_bag := apples_in_geralds_bag * multiple_factor in
  let pam_bags := total_apples_pam_has / apples_in_pams_bag in
  pam_bags = expected_pam_bags :=
by
  intros apples_in_geralds_bag multiple_factor total_apples_pam_has expected_pam_bags
  intros h1 h2 h3 h4
  let apples_in_pams_bag := apples_in_geralds_bag * multiple_factor
  let pam_bags := total_apples_pam_has / apples_in_pams_bag
  rw [h1, h2, h3, h4]
  sorry

end Pam_bags_count_l104_104428


namespace oliver_final_amount_l104_104741

variable (initial_amount saved_amount spent_on_frisbee spent_on_puzzle gift : ℕ)

def final_amount_after_transactions (initial_amount saved_amount spent_on_frisbee spent_on_puzzle gift : ℕ) : ℕ :=
  initial_amount + saved_amount - (spent_on_frisbee + spent_on_puzzle) + gift

theorem oliver_final_amount :
  final_amount_after_transactions 9 5 4 3 8 = 15 :=
by
  -- We can fill in the exact calculations here to provide the proof.
  sorry

end oliver_final_amount_l104_104741


namespace value_of_N_l104_104052

theorem value_of_N (N : ℕ) (h : Nat.choose N 5 = 231) : N = 11 := sorry

end value_of_N_l104_104052


namespace derek_dogs_count_l104_104820

theorem derek_dogs_count
  (initial_dogs : ℕ)
  (initial_cars : ℕ)
  (cars_after_10_years : ℕ)
  (dogs_after_10_years : ℕ)
  (h1 : initial_dogs = 90)
  (h2 : initial_dogs = 3 * initial_cars)
  (h3 : cars_after_10_years = initial_cars + 210)
  (h4 : cars_after_10_years = 2 * dogs_after_10_years) :
  dogs_after_10_years = 120 :=
by
  sorry

end derek_dogs_count_l104_104820


namespace hannah_strawberries_l104_104855

-- Definitions for the conditions
def daily_harvest : ℕ := 5
def days_in_april : ℕ := 30
def strawberries_given_away : ℕ := 20
def strawberries_stolen : ℕ := 30

-- The statement we need to prove
theorem hannah_strawberries (harvested_strawberries : ℕ)
  (total_harvest := daily_harvest * days_in_april)
  (total_lost := strawberries_given_away + strawberries_stolen)
  (final_count := total_harvest - total_lost) :
  harvested_strawberries = final_count :=
sorry

end hannah_strawberries_l104_104855


namespace sum_of_number_and_square_is_306_l104_104778

theorem sum_of_number_and_square_is_306 : ∃ x : ℤ, x + x^2 = 306 ∧ x = 17 :=
by
  sorry

end sum_of_number_and_square_is_306_l104_104778


namespace convert_base_10_to_base_7_l104_104030

def base10_to_base7 (n : ℕ) : ℕ := 
  match n with
  | 5423 => 21545
  | _ => 0

theorem convert_base_10_to_base_7 : base10_to_base7 5423 = 21545 := by
  rfl

end convert_base_10_to_base_7_l104_104030


namespace maximize_area_l104_104609

noncomputable def max_area : ℝ :=
  let l := 100
  let w := 100
  l * w

theorem maximize_area (l w : ℝ) (h1 : 2 * l + 2 * w = 400) (h2 : l ≥ 100) (h3 : w ≥ 50) :
  (l * w ≤ 10000) :=
sorry

end maximize_area_l104_104609


namespace reciprocal_of_neg2_l104_104453

-- Define the number
def num : ℤ := -2

-- Define the reciprocal function
def reciprocal (x : ℤ) : ℚ := 1 / x

-- State the theorem
theorem reciprocal_of_neg2 : reciprocal num = -1 / 2 :=
by
  -- Proof is skipped
  sorry

end reciprocal_of_neg2_l104_104453


namespace total_exercise_time_l104_104726

-- Definitions based on given conditions
def javier_daily : ℕ := 50
def javier_days : ℕ := 7
def sanda_daily : ℕ := 90
def sanda_days : ℕ := 3

-- Proof problem to verify the total exercise time for both Javier and Sanda
theorem total_exercise_time : javier_daily * javier_days + sanda_daily * sanda_days = 620 := by
  sorry

end total_exercise_time_l104_104726


namespace sqrt81_minus_sqrt77_plus1_approx_l104_104527

noncomputable def approximate_sqrt (x : ℝ) : ℝ :=
  if x = 81 then 9 else if x = 77 then 8.78 else √x

theorem sqrt81_minus_sqrt77_plus1_approx :
  (approximate_sqrt 81 - approximate_sqrt 77) + 1 ≈ 1.23 :=
by
  have h_sqrt81 : approximate_sqrt 81 = 9 := if_pos rfl
  have h_sqrt77 : approximate_sqrt 77 = 8.78 := if_neg (by norm_num)

  rw [h_sqrt81, h_sqrt77]
  norm_num
  sorry

end sqrt81_minus_sqrt77_plus1_approx_l104_104527


namespace arrangement_meeting_ways_l104_104653

-- For convenience, define the number of members per school and the combination function.
def num_members_per_school : ℕ := 6
def num_schools : ℕ :=  4
def combination (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

theorem arrangement_meeting_ways : 
  let host_ways := num_schools
  let host_reps_ways := combination num_members_per_school 2
  let non_host_schools := num_schools - 1
  let non_host_reps_ways := combination num_members_per_school 2
  let total_non_host_reps_ways := non_host_reps_ways ^ non_host_schools
  let total_ways := host_ways * host_reps_ways * total_non_host_reps_ways
  total_ways = 202500 :=
by 
  -- Definitions and computation is deferred to the steps,
  -- which are to be filled during the proof.
  sorry

end arrangement_meeting_ways_l104_104653


namespace Erik_money_left_l104_104679

theorem Erik_money_left 
  (init_money : ℝ)
  (loaf_of_bread : ℝ) (n_loaves_of_bread : ℝ)
  (carton_of_orange_juice : ℝ) (n_cartons_of_orange_juice : ℝ)
  (dozen_eggs : ℝ) (n_dozens_of_eggs : ℝ)
  (chocolate_bar : ℝ) (n_chocolate_bars : ℝ)
  (pound_apples : ℝ) (n_pounds_apples : ℝ)
  (pound_grapes : ℝ) (n_pounds_grapes : ℝ)
  (discount_bread_and_eggs : ℝ) (discount_other_items : ℝ)
  (sales_tax : ℝ) :
  n_loaves_of_bread = 3 →
  loaf_of_bread = 3 →
  n_cartons_of_orange_juice = 3 →
  carton_of_orange_juice = 6 →
  n_dozens_of_eggs = 2 →
  dozen_eggs = 4 →
  n_chocolate_bars = 5 →
  chocolate_bar = 2 →
  n_pounds_apples = 4 →
  pound_apples = 1.25 →
  n_pounds_grapes = 1.5 →
  pound_grapes = 2.5 →
  discount_bread_and_eggs = 0.1 →
  discount_other_items = 0.05 →
  sales_tax = 0.06 →
  init_money = 86 →
  (init_money - 
     (n_loaves_of_bread * loaf_of_bread * (1 - discount_bread_and_eggs) + 
      n_cartons_of_orange_juice * carton_of_orange_juice * (1 - discount_other_items) + 
      n_dozens_of_eggs * dozen_eggs * (1 - discount_bread_and_eggs) + 
      n_chocolate_bars * chocolate_bar * (1 - discount_other_items) + 
      n_pounds_apples * pound_apples * (1 - discount_other_items) + 
      n_pounds_grapes * pound_grapes * (1 - discount_other_items)) * (1 + sales_tax)) = 32.78 :=
by
  sorry

end Erik_money_left_l104_104679


namespace polynomial_product_is_square_l104_104749

theorem polynomial_product_is_square (x a : ℝ) :
  (x + a) * (x + 2 * a) * (x + 3 * a) * (x + 4 * a) + a^4 = (x^2 + 5 * a * x + 5 * a^2)^2 :=
by
  sorry

end polynomial_product_is_square_l104_104749


namespace supermarket_spent_more_than_collected_l104_104608

-- Given conditions
def initial_amount : ℕ := 53
def collected_amount : ℕ := 91
def amount_left : ℕ := 14

-- Finding the total amount before shopping and amount spent in supermarket
def total_amount : ℕ := initial_amount + collected_amount
def spent_amount : ℕ := total_amount - amount_left

-- Prove that the difference between spent amount and collected amount is 39
theorem supermarket_spent_more_than_collected : (spent_amount - collected_amount) = 39 := by
  -- The proof will go here
  sorry

end supermarket_spent_more_than_collected_l104_104608


namespace functional_relationship_maximum_profit_desired_profit_l104_104035

-- Conditions
def cost_price := 80
def y (x : ℝ) : ℝ := -2 * x + 320
def w (x : ℝ) : ℝ := (x - cost_price) * y x

-- Functional relationship
theorem functional_relationship (x : ℝ) (hx : 80 ≤ x ∧ x ≤ 160) :
  w x = -2 * x^2 + 480 * x - 25600 :=
by sorry

-- Maximizing daily profit
theorem maximum_profit :
  ∃ x, 80 ≤ x ∧ x ≤ 160 ∧ w x = 3200 ∧ (∀ y, 80 ≤ y ∧ y ≤ 160 → w y ≤ 3200) ∧ x = 120 :=
by sorry

-- Desired profit of 2400 dollars
theorem desired_profit (w_desired : ℝ) (hw : w_desired = 2400) :
  ∃ x, 80 ≤ x ∧ x ≤ 160 ∧ w x = w_desired ∧ x = 100 :=
by sorry

end functional_relationship_maximum_profit_desired_profit_l104_104035


namespace find_f_1_div_2007_l104_104816

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_1_div_2007 :
  f 0 = 0 ∧
  (∀ x, f x + f (1 - x) = 1) ∧
  (∀ x, f (x / 5) = f x / 2) ∧
  (∀ x1 x2, 0 ≤ x1 → x1 < x2 → x2 ≤ 1 → f x1 ≤ f x2) →
  f (1 / 2007) = 1 / 32 :=
sorry

end find_f_1_div_2007_l104_104816


namespace height_of_pole_l104_104613

theorem height_of_pole (pole_shadow tree_shadow tree_height : ℝ) 
                       (ratio_equal : pole_shadow = 84 ∧ tree_shadow = 32 ∧ tree_height = 28) : 
                       round (tree_height * (pole_shadow / tree_shadow)) = 74 :=
by
  sorry

end height_of_pole_l104_104613


namespace cost_of_paving_l104_104935

theorem cost_of_paving (L W R : ℝ) (hL : L = 6.5) (hW : W = 2.75) (hR : R = 600) : 
  L * W * R = 10725 := by
  rw [hL, hW, hR]
  -- To solve the theorem successively
  -- we would need to verify the product of the values
  -- given by the conditions.
  sorry

end cost_of_paving_l104_104935


namespace hardey_fitness_center_ratio_l104_104902

theorem hardey_fitness_center_ratio
  (f m : ℕ)
  (avg_female_weight : ℕ := 140)
  (avg_male_weight : ℕ := 180)
  (avg_overall_weight : ℕ := 160)
  (h1 : avg_female_weight * f + avg_male_weight * m = avg_overall_weight * (f + m)) :
  f = m :=
by
  sorry

end hardey_fitness_center_ratio_l104_104902


namespace possible_values_of_r_l104_104303

noncomputable def r : ℝ := sorry

def is_four_place_decimal (x : ℝ) : Prop := 
  ∃ (a b c d : ℕ), a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ x = a / 10 + b / 100 + c / 1000 + d / 10000

def is_closest_fraction (x : ℝ) : Prop := 
  abs (x - 3 / 11) < abs (x - 3 / 10) ∧ abs (x - 3 / 11) < abs (x - 1 / 4)

theorem possible_values_of_r :
  (0.2614 <= r ∧ r <= 0.2864) ∧ is_four_place_decimal r ∧ is_closest_fraction r →
  ∃ n : ℕ, n = 251 := 
sorry

end possible_values_of_r_l104_104303


namespace n_squared_plus_2n_plus_3_mod_50_l104_104865

theorem n_squared_plus_2n_plus_3_mod_50 (n : ℤ) (hn : n % 50 = 49) : (n^2 + 2 * n + 3) % 50 = 2 := 
sorry

end n_squared_plus_2n_plus_3_mod_50_l104_104865


namespace proof_problem_l104_104540

open Set

noncomputable def U : Set ℝ := Icc (-5 : ℝ) 4

noncomputable def A : Set ℝ := {x : ℝ | -3 ≤ 2 * x + 1 ∧ 2 * x + 1 < 1}

noncomputable def B : Set ℝ := {x : ℝ | x^2 - 2 * x ≤ 0}

-- Definition of the complement of A in U
noncomputable def complement_U_A : Set ℝ := U \ A

-- The final proof statement
theorem proof_problem : (complement_U_A ∩ B) = Icc 0 2 :=
by
  sorry

end proof_problem_l104_104540


namespace find_number_l104_104625

theorem find_number (x : ℝ) (h : x / 5 = 30 + x / 6) : x = 900 :=
by
  -- Proof goes here
  sorry

end find_number_l104_104625


namespace number_of_members_l104_104776

theorem number_of_members (n : ℕ) (h : n^2 = 9801) : n = 99 :=
sorry

end number_of_members_l104_104776


namespace volume_of_prism_l104_104124

theorem volume_of_prism (a b c : ℝ) (h1 : a * b = 30) (h2 : a * c = 45) (h3 : b * c = 54) : 
  a * b * c = 270 :=
by
  sorry

end volume_of_prism_l104_104124


namespace complement_is_correct_l104_104247

def U : Set ℝ := {x | x^2 ≤ 4}
def A : Set ℝ := {x | abs (x + 1) ≤ 1}
def complement_U_A : Set ℝ := U \ A

theorem complement_is_correct :
  complement_U_A = {x | 0 < x ∧ x ≤ 2} := by
  sorry

end complement_is_correct_l104_104247


namespace solution_set_of_inequality_l104_104136

theorem solution_set_of_inequality (x : ℝ) (h : (2 * x - 1) / x < 0) : 0 < x ∧ x < 1 / 2 :=
by
  sorry

end solution_set_of_inequality_l104_104136


namespace middle_circle_radius_l104_104721

theorem middle_circle_radius 
  (r1 r3 : ℝ) 
  (geometric_sequence: ∃ r2 : ℝ, r2 ^ 2 = r1 * r3) 
  (r1_val : r1 = 5) 
  (r3_val : r3 = 20) 
  : ∃ r2 : ℝ, r2 = 10 := 
by
  sorry

end middle_circle_radius_l104_104721


namespace crabapple_sequences_count_l104_104957

theorem crabapple_sequences_count (n m : ℕ) (h1 : n = 12) (h2 : m = 5) :
  n^m = 248832 := by
  rw [h1, h2]
  norm_num

end crabapple_sequences_count_l104_104957


namespace fraction_comparison_l104_104631

theorem fraction_comparison :
  (2 : ℝ) * (4 : ℝ) > (7 : ℝ) → (4 / 7 : ℝ) > (1 / 2 : ℝ) :=
by
  sorry

end fraction_comparison_l104_104631


namespace f_1982_eq_660_l104_104446

def f : ℕ → ℕ := sorry

axiom h1 : ∀ m n : ℕ, f (m + n) - f m - f n = 0 ∨ f (m + n) - f m - f n = 1
axiom h2 : f 2 = 0
axiom h3 : f 3 > 0
axiom h4 : f 9999 = 3333

theorem f_1982_eq_660 : f 1982 = 660 := sorry

end f_1982_eq_660_l104_104446


namespace nadia_probability_condition_l104_104736

noncomputable def probability_no_favorite_track :: (tracks : List ℕ) (duration : ℕ) (favorite_track_length : ℕ) : ℚ :=
  let n := tracks.length
  let total_permutations := nat.factorial n
  let valid_permutations := -- Calculate the number of valid permutations where the favorite track is in the first 360 seconds
     -- Implementation of valid permutation calculation goes here
  1 - (valid_permutations / total_permutations)

theorem nadia_probability_condition (tracks : List ℕ) (duration : ℕ) (favorite_track_length : ℕ) :
  tracks.length = 12 →
  ∀ i, (i < 12 → tracks.nth i = some (20 + 20 * i)) →
  favorite_track_length = 280 →
  duration = 360 →
  let prob := probability_no_favorite_track tracks duration favorite_track_length 
  prob = 1 - -- Computed value
 := by 
  -- skip proof
  sorry

end nadia_probability_condition_l104_104736


namespace lcm_36_105_l104_104210

theorem lcm_36_105 : Int.lcm 36 105 = 1260 := by
  sorry

end lcm_36_105_l104_104210


namespace find_k_arithmetic_progression_l104_104081

theorem find_k_arithmetic_progression :
  ∃ k : ℤ, (2 * real.sqrt (225 + k) = real.sqrt (49 + k) + real.sqrt (441 + k)) → k = 255 :=
by
  sorry

end find_k_arithmetic_progression_l104_104081


namespace gretchen_total_money_l104_104853

def charge_per_drawing : ℕ := 20
def sold_saturday : ℕ := 24
def sold_sunday : ℕ := 16
def total_caricatures := sold_saturday + sold_sunday

theorem gretchen_total_money : charge_per_drawing * total_caricatures = 800 := by
  have total_caricatures_eq : total_caricatures = 40 := by
    unfold total_caricatures
    simp
  rw [total_caricatures_eq]
  calc
    20 * 40 = 800 := by norm_num

end gretchen_total_money_l104_104853


namespace members_not_playing_either_l104_104412

variable (total_members badminton_players tennis_players both_players : ℕ)

theorem members_not_playing_either (h1 : total_members = 40)
                                   (h2 : badminton_players = 20)
                                   (h3 : tennis_players = 18)
                                   (h4 : both_players = 3) :
  total_members - (badminton_players + tennis_players - both_players) = 5 := by
  sorry

end members_not_playing_either_l104_104412


namespace division_problem_l104_104619

theorem division_problem (x : ℕ) (h : x / 5 = 30 + x / 6) : x = 900 :=
sorry

end division_problem_l104_104619


namespace phone_purchase_initial_max_profit_additional_purchase_l104_104441

-- Definitions for phone purchase prices and selling prices
def purchase_price_A : ℕ := 3000
def selling_price_A : ℕ := 3400
def purchase_price_B : ℕ := 3500
def selling_price_B : ℕ := 4000

-- Definitions for total expenditure and profit
def total_spent : ℕ := 32000
def total_profit : ℕ := 4400

-- Definitions for initial number of units purchased
def initial_units_A : ℕ := 6
def initial_units_B : ℕ := 4

-- Definitions for the additional purchase constraints and profit calculation
def max_additional_units : ℕ := 30
def additional_units_A : ℕ := 10
def additional_units_B : ℕ := max_additional_units - additional_units_A 
def max_profit : ℕ := 14000

theorem phone_purchase_initial:
  3000 * initial_units_A + 3500 * initial_units_B = total_spent ∧
  (selling_price_A - purchase_price_A) * initial_units_A + (selling_price_B - purchase_price_B) * initial_units_B = total_profit := by
  sorry 

theorem max_profit_additional_purchase:
  additional_units_A + additional_units_B = max_additional_units ∧
  additional_units_B ≤ 2 * additional_units_A ∧
  (selling_price_A - purchase_price_A) * additional_units_A + (selling_price_B - purchase_price_B) * additional_units_B = max_profit := by
  sorry

end phone_purchase_initial_max_profit_additional_purchase_l104_104441


namespace find_number_l104_104620

theorem find_number (x : ℝ) (h : x / 5 = 30 + x / 6) : x = 900 :=
  sorry

end find_number_l104_104620


namespace george_initial_socks_l104_104531

theorem george_initial_socks (S : ℕ) (h : S - 4 + 36 = 60) : S = 28 :=
by
  sorry

end george_initial_socks_l104_104531


namespace trig_identity_example_l104_104237

theorem trig_identity_example (α : Real) (h : Real.cos α = 3 / 5) : Real.cos (2 * α) + Real.sin α ^ 2 = 9 / 25 := by
  sorry

end trig_identity_example_l104_104237


namespace fully_loaded_truck_weight_l104_104968

def empty_truck : ℕ := 12000
def weight_soda_crate : ℕ := 50
def num_soda_crates : ℕ := 20
def weight_dryer : ℕ := 3000
def num_dryers : ℕ := 3
def weight_fresh_produce : ℕ := 2 * (weight_soda_crate * num_soda_crates)

def total_loaded_truck_weight : ℕ :=
  empty_truck + (weight_soda_crate * num_soda_crates) + weight_fresh_produce + (weight_dryer * num_dryers)

theorem fully_loaded_truck_weight : total_loaded_truck_weight = 24000 := by
  sorry

end fully_loaded_truck_weight_l104_104968


namespace coaches_meet_together_l104_104826

theorem coaches_meet_together (e s n a : ℕ)
  (h₁ : e = 5) (h₂ : s = 3) (h₃ : n = 9) (h₄ : a = 8) :
  Nat.lcm (Nat.lcm e s) (Nat.lcm n a) = 360 :=
by
  sorry

end coaches_meet_together_l104_104826


namespace swimmer_distance_l104_104017

theorem swimmer_distance :
  let swimmer_speed : ℝ := 3
  let current_speed : ℝ := 1.7
  let time : ℝ := 2.3076923076923075
  let effective_speed := swimmer_speed - current_speed
  let distance := effective_speed * time
  distance = 3 := by
sorry

end swimmer_distance_l104_104017


namespace range_of_a_l104_104913

theorem range_of_a :
  (∀ t : ℝ, 0 < t ∧ t ≤ 2 → (t / (t^2 + 9) ≤ a ∧ a ≤ (t + 2) / t^2)) →
  (2 / 13 ≤ a ∧ a ≤ 1) :=
by
  intro h
  -- Proof of the theorem goes here
  sorry

end range_of_a_l104_104913


namespace shaded_area_is_14_percent_l104_104769

def side_length : ℕ := 20
def rectangle_width : ℕ := 35
def rectangle_height : ℕ := side_length
def rectangle_area : ℕ := rectangle_width * rectangle_height
def overlap_length : ℕ := 2 * side_length - rectangle_width
def shaded_area : ℕ := overlap_length * side_length
def shaded_percentage : ℚ := (shaded_area : ℚ) / rectangle_area * 100

theorem shaded_area_is_14_percent : shaded_percentage = 14 := by
  sorry

end shaded_area_is_14_percent_l104_104769


namespace final_result_is_8_l104_104012

theorem final_result_is_8 (n : ℕ) (h1 : n = 2976) (h2 : (n / 12) - 240 = 8) : (n / 12) - 240 = 8 :=
by {
  -- Proof steps would go here
  sorry
}

end final_result_is_8_l104_104012


namespace not_right_triangle_angle_ratio_l104_104773

theorem not_right_triangle_angle_ratio (A B C : ℝ) (h₁ : A / B = 3 / 4) (h₂ : B / C = 4 / 5) (h₃ : A + B + C = 180) : ¬ (A = 90 ∨ B = 90 ∨ C = 90) :=
sorry

end not_right_triangle_angle_ratio_l104_104773


namespace division_problem_l104_104616

theorem division_problem (x : ℕ) (h : x / 5 = 30 + x / 6) : x = 900 :=
sorry

end division_problem_l104_104616


namespace find_t_l104_104851

-- Definitions of the vectors and parallel condition
def a : ℝ × ℝ := (-1, 1)
def b (t : ℝ) : ℝ × ℝ := (3, t)
def is_parallel (u v : ℝ × ℝ) : Prop := ∃ k : ℝ, u = k • v ∨ v = k • u

-- The theorem statement
theorem find_t (t : ℝ) (h : is_parallel (b t) (a + b t)) : t = -3 := by
  sorry

end find_t_l104_104851


namespace oliver_money_left_l104_104744

-- Definitions based on the given conditions
def initial_amount : ℕ := 9
def allowance_saved : ℕ := 5
def frisbee_cost : ℕ := 4
def puzzle_cost : ℕ := 3
def birthday_gift : ℕ := 8

-- The proof problem statement
theorem oliver_money_left : 
  initial_amount + allowance_saved - (frisbee_cost + puzzle_cost) + birthday_gift = 15 := 
by
  sorry

end oliver_money_left_l104_104744


namespace praveen_initial_investment_l104_104111

theorem praveen_initial_investment
  (H : ℝ) (P : ℝ)
  (h_H : H = 9000.000000000002)
  (h_profit_ratio : (P * 12) / (H * 7) = 2 / 3) :
  P = 3500 := by
  sorry

end praveen_initial_investment_l104_104111


namespace fred_speed_l104_104384

variable {F : ℝ} -- Fred's speed
variable {T : ℝ} -- Time in hours

-- Conditions
def initial_distance : ℝ := 35
def sam_speed : ℝ := 5
def sam_distance : ℝ := 25
def fred_distance := initial_distance - sam_distance

-- Theorem to prove
theorem fred_speed (h1 : T = sam_distance / sam_speed) (h2 : fred_distance = F * T) :
  F = 2 :=
by
  sorry

end fred_speed_l104_104384


namespace division_problem_l104_104617

theorem division_problem (x : ℕ) (h : x / 5 = 30 + x / 6) : x = 900 :=
sorry

end division_problem_l104_104617


namespace no_solutions_exist_l104_104367

theorem no_solutions_exist (m n : ℤ) : ¬(m^2 = n^2 + 1954) :=
by sorry

end no_solutions_exist_l104_104367


namespace area_of_shaded_region_l104_104172

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

end area_of_shaded_region_l104_104172


namespace dorothy_money_left_l104_104512

def annual_income : ℝ := 60000
def tax_rate : ℝ := 0.18
def tax_amount : ℝ := annual_income * tax_rate
def money_left : ℝ := annual_income - tax_amount

theorem dorothy_money_left : money_left = 49200 := 
by
  sorry

end dorothy_money_left_l104_104512


namespace only_positive_integer_x_l104_104043

theorem only_positive_integer_x (x : ℕ) (k : ℕ) (h1 : 2 * x + 1 = k^2) (h2 : x > 0) :
  ¬ (∃ y : ℕ, (y >= 2 * x + 2 ∧ y <= 3 * x + 2 ∧ ∃ m : ℕ, y = m^2)) → x = 4 := 
by sorry

end only_positive_integer_x_l104_104043


namespace drums_per_day_l104_104856

theorem drums_per_day (total_drums : Nat) (days : Nat) (total_drums_eq : total_drums = 6264) (days_eq : days = 58) :
  total_drums / days = 108 :=
by
  sorry

end drums_per_day_l104_104856


namespace tangent_line_eq_l104_104377

noncomputable def f (x : ℝ) := x / (2 * x - 1)

def tangentLineAtPoint (x : ℝ) : ℝ := -x + 2

theorem tangent_line_eq {x y : ℝ} (hxy : y = f 1) (f_deriv : deriv f 1 = -1) :
  y = 1 → tangentLineAtPoint x = -x + 2 :=
by
  intros
  sorry

end tangent_line_eq_l104_104377


namespace smallest_integer_remainder_l104_104771

theorem smallest_integer_remainder :
  ∃ n : ℕ, n > 1 ∧
           (n % 3 = 2) ∧
           (n % 4 = 2) ∧
           (n % 5 = 2) ∧
           (n % 7 = 2) ∧
           n = 422 :=
by
  sorry

end smallest_integer_remainder_l104_104771


namespace segment_length_l104_104897

theorem segment_length (AB BC AC : ℝ) (hAB : AB = 4) (hBC : BC = 3) :
  AC = 7 ∨ AC = 1 :=
sorry

end segment_length_l104_104897


namespace nancy_coffee_spending_l104_104284

theorem nancy_coffee_spending :
  let daily_cost := 3.00 + 2.50
  let total_days := 20
  let total_cost := total_days * daily_cost
  total_cost = 110.00 := by
    let daily_cost := 3.00 + 2.50
    let total_days := 20
    let total_cost := total_days * daily_cost
    have h1 : daily_cost = 5.50 := by norm_num
    have h2 : total_cost = 20 * 5.50 := by rw [total_cost, total_days, h1]
    have h3 : total_cost = 110.00 := by norm_num
    exact h3

end nancy_coffee_spending_l104_104284


namespace find_t_l104_104843

noncomputable def f (x t k : ℝ): ℝ := (1/3) * x^3 - (t/2) * x^2 + k * x

theorem find_t (a b t k : ℝ) (h1 : t > 0) (h2 : k > 0) 
  (h3 : a + b = t) (h4 : a * b = k)
  (h5 : 2 * a = b - 2)
  (h6 : (-2) ^ 2 = a * b) : 
  t = 5 :=
by 
  sorry

end find_t_l104_104843


namespace reciprocal_of_neg_two_l104_104451

theorem reciprocal_of_neg_two : ∃ x : ℝ, (-2) * x = 1 ∧ x = -1/2 :=
by
  use -1/2
  split
  · exact (by norm_num : (-2) * (-1/2) = 1)
  · rfl

end reciprocal_of_neg_two_l104_104451


namespace contrapositive_iff_l104_104587

theorem contrapositive_iff (a b : ℤ) : (a > b → a - 5 > b - 5) ↔ (a - 5 ≤ b - 5 → a ≤ b) :=
by sorry

end contrapositive_iff_l104_104587


namespace reduced_price_is_60_l104_104477

variable (P R: ℝ) -- Declare the variables P and R as real numbers.

-- Define the conditions as hypotheses.
axiom h1 : R = 0.7 * P
axiom h2 : 1800 / R = 1800 / P + 9

-- The theorem stating the problem to prove.
theorem reduced_price_is_60 (P R : ℝ) (h1 : R = 0.7 * P) (h2 : 1800 / R = 1800 / P + 9) : R = 60 :=
by sorry

end reduced_price_is_60_l104_104477


namespace number_of_members_after_four_years_l104_104337

theorem number_of_members_after_four_years (b : ℕ → ℕ) (initial_condition : b 0 = 21) 
    (yearly_update : ∀ k, b (k + 1) = 4 * b k - 9) : 
    b 4 = 4611 :=
    sorry

end number_of_members_after_four_years_l104_104337


namespace train_length_l104_104953

theorem train_length (L V : ℝ) (h1 : L = V * 120) (h2 : L + 1000 = V * 220) : L = 1200 := 
by
  sorry

end train_length_l104_104953


namespace max_members_choir_l104_104588

variable (m k n : ℕ)

theorem max_members_choir :
  (∃ k, m = k^2 + 6) ∧ (∃ n, m = n * (n + 6)) → m = 294 :=
by
  sorry

end max_members_choir_l104_104588


namespace min_workers_for_profit_l104_104654

def revenue (n : ℕ) : ℕ := 240 * n
def cost (n : ℕ) : ℕ := 600 + 200 * n

theorem min_workers_for_profit (n : ℕ) (h : 240 * n > 600 + 200 * n) : n >= 16 :=
by {
  -- Placeholder for the proof steps (which are not required per instructions)
  sorry
}

end min_workers_for_profit_l104_104654


namespace largest_square_area_l104_104755

theorem largest_square_area (a b c : ℝ) 
  (h1 : c^2 = a^2 + b^2) 
  (h2 : a = b - 5) 
  (h3 : a^2 + b^2 + c^2 = 450) : 
  c^2 = 225 :=
by 
  sorry

end largest_square_area_l104_104755


namespace most_significant_action_for_sustainable_utilization_l104_104260

def investigate_population_dynamics (most_significant_action : String) :=
  most_significant_action = "Investigate the population dynamics of the fish species"

theorem most_significant_action_for_sustainable_utilization :
  investigate_population_dynamics ("Investigate the population dynamics of the fish species") :=
by
  -- We'll be skipping the proof using sorry, as instructed
  sorry

end most_significant_action_for_sustainable_utilization_l104_104260


namespace ellipse_foci_distance_l104_104519

theorem ellipse_foci_distance 
  (h : ∀ x y : ℝ, 9 * x^2 + y^2 = 144) : 
  ∃ c : ℝ, c = 16 * Real.sqrt 2 :=
  sorry

end ellipse_foci_distance_l104_104519


namespace cos_pi_div_12_value_l104_104687

noncomputable def cos_pi_div_12 : ℝ :=
  cos (π / 12)

theorem cos_pi_div_12_value : cos_pi_div_12 = (√6 + √2) / 4 :=
by
  sorry

end cos_pi_div_12_value_l104_104687


namespace num_sides_polygon_l104_104600

theorem num_sides_polygon (n : ℕ) (h : (n - 2) * 180 = 1260) : n = 9 := by
  sorry

end num_sides_polygon_l104_104600


namespace range_of_m_l104_104996

theorem range_of_m (m : ℝ) :
  let A := {x : ℝ | -2 < x ∧ x ≤ 5},
      B := {x : ℝ | -m + 1 ≤ x ∧ x ≤ 2 * m - 1}
  in B ⊆ A → m < 3 :=
by
  intros A B hB
  sorry

end range_of_m_l104_104996


namespace circles_tangent_l104_104192

theorem circles_tangent
  (rA rB rC rD rF : ℝ) (rE : ℚ) (m n : ℕ)
  (m_n_rel_prime : Int.gcd m n = 1)
  (rA_pos : 0 < rA) (rB_pos : 0 < rB)
  (rC_pos : 0 < rC) (rD_pos : 0 < rD)
  (rF_pos : 0 < rF)
  (inscribed_triangle_in_A : True)  -- Triangle T is inscribed in circle A
  (B_tangent_A : True)  -- Circle B is internally tangent to circle A
  (C_tangent_A : True)  -- Circle C is internally tangent to circle A
  (D_tangent_A : True)  -- Circle D is internally tangent to circle A
  (B_externally_tangent_E : True)  -- Circle B is externally tangent to circle E
  (C_externally_tangent_E : True)  -- Circle C is externally tangent to circle E
  (D_externally_tangent_E : True)  -- Circle D is externally tangent to circle E
  (F_tangent_A : True)  -- Circle F is internally tangent to circle A at midpoint of side opposite to B's tangency
  (F_externally_tangent_E : True)  -- Circle F is externally tangent to circle E
  (rA_eq : rA = 12) (rB_eq : rB = 5)
  (rC_eq : rC = 3) (rD_eq : rD = 2)
  (rF_eq : rF = 1)
  (rE_eq : rE = m / n)
  : m + n = 23 :=
by
  sorry

end circles_tangent_l104_104192


namespace binomial_coefficient_example_l104_104360

theorem binomial_coefficient_example :
  2 * (Nat.choose 7 4) = 70 := 
sorry

end binomial_coefficient_example_l104_104360


namespace find_integer_triplets_l104_104207

theorem find_integer_triplets (x y z : ℤ) : 
  x^3 + y^3 + z^3 - 3 * x * y * z = 2003 ↔ (x = 668 ∧ y = 668 ∧ z = 667) ∨ (x = 668 ∧ y = 667 ∧ z = 668) ∨ (x = 667 ∧ y = 668 ∧ z = 668) :=
by
  sorry

end find_integer_triplets_l104_104207


namespace find_a_l104_104084

noncomputable def givenConditions (a b c R : ℝ) : Prop :=
  (a^2 / (b * c) - c / b - b / c = Real.sqrt 3) ∧ (R = 3)

theorem find_a (a b c : ℝ) (R : ℝ) (h : givenConditions a b c R) : a = 3 :=
by
  sorry

end find_a_l104_104084


namespace simplify_expression_l104_104889

def real_numbers (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a^3 + b^3 = a^2 + b^2

theorem simplify_expression (a b : ℝ) (h : real_numbers a b) :
  (a^2 / b + b^2 / a - 1 / (a * a * b * b)) = (a^4 + 2 * a * b + b^4 - 1) / (a * b) :=
by
  sorry

end simplify_expression_l104_104889


namespace total_gallons_in_tanks_l104_104107

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

end total_gallons_in_tanks_l104_104107


namespace expand_polynomial_l104_104041

theorem expand_polynomial :
  (x^2 - 3 * x + 3) * (x^2 + 3 * x + 1) = x^4 - 5 * x^2 + 6 * x + 3 :=
by
  sorry

end expand_polynomial_l104_104041


namespace train_speed_problem_l104_104486

open Real

/-- Given specific conditions about the speeds and lengths of trains, prove the speed of the third train is 99 kmph. -/
theorem train_speed_problem
  (man_train_speed_kmph : ℝ)
  (man_train_speed : ℝ)
  (goods_train_length : ℝ)
  (goods_train_time : ℝ)
  (third_train_length : ℝ)
  (third_train_time : ℝ) :
  man_train_speed_kmph = 45 →
  man_train_speed = 45 * 1000 / 3600 →
  goods_train_length = 340 →
  goods_train_time = 8 →
  third_train_length = 480 →
  third_train_time = 12 →
  (third_train_length / third_train_time - man_train_speed) * 3600 / 1000 = 99 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end train_speed_problem_l104_104486


namespace geometric_seq_value_l104_104386

theorem geometric_seq_value (a : ℕ → ℝ) (h : a 4 + a 8 = -2) :
  a 6 * (a 2 + 2 * a 6 + a 10) = 4 :=
sorry

end geometric_seq_value_l104_104386


namespace log_monotonic_increasing_l104_104274

noncomputable def f (a x : ℝ) := Real.log x / Real.log a

theorem log_monotonic_increasing (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) (h3 : 1 < a) :
  f a (a + 1) > f a 2 := 
by
  -- Here the actual proof will be added.
  sorry

end log_monotonic_increasing_l104_104274


namespace smallest_distance_proof_l104_104731

noncomputable def smallest_distance (z w : ℂ) : ℝ :=
  Complex.abs (z - w)

theorem smallest_distance_proof (z w : ℂ) 
  (h1 : Complex.abs (z - (2 - 4*Complex.I)) = 2)
  (h2 : Complex.abs (w - (-5 + 6*Complex.I)) = 4) :
  smallest_distance z w ≥ Real.sqrt 149 - 6 :=
by
  sorry

end smallest_distance_proof_l104_104731


namespace lcm_36_105_l104_104216

theorem lcm_36_105 : Nat.lcm 36 105 = 1260 := by
  sorry

end lcm_36_105_l104_104216


namespace time_with_cat_total_l104_104277

def time_spent_with_cat (petting combing brushing playing feeding cleaning : ℕ) : ℕ :=
  petting + combing + brushing + playing + feeding + cleaning

theorem time_with_cat_total :
  let petting := 12
  let combing := 1/3 * petting
  let brushing := 1/4 * combing
  let playing := 1/2 * petting
  let feeding := 5
  let cleaning := 2/5 * feeding
  time_spent_with_cat petting combing brushing playing feeding cleaning = 30 := by
  sorry

end time_with_cat_total_l104_104277


namespace parabola_equation_l104_104841

open Real

theorem parabola_equation (vertex focus : ℝ × ℝ) (h_vertex : vertex = (0, 0)) (h_focus : focus = (0, 3)) :
  ∃ a : ℝ, x^2 = 12 * y := by
  sorry

end parabola_equation_l104_104841


namespace quotient_of_division_l104_104110

theorem quotient_of_division (dividend divisor remainder quotient : ℕ)
  (h_dividend : dividend = 15)
  (h_divisor : divisor = 3)
  (h_remainder : remainder = 3)
  (h_relation : dividend = divisor * quotient + remainder) :
  quotient = 4 :=
by sorry

end quotient_of_division_l104_104110


namespace point_always_outside_circle_l104_104529

theorem point_always_outside_circle (a : ℝ) : a^2 + (2 - a)^2 > 1 :=
by sorry

end point_always_outside_circle_l104_104529


namespace mens_wages_l104_104942

variable (M : ℕ) (wages_of_men : ℕ)

-- Conditions based on the problem
axiom eq1 : 15 * M = 90
axiom def_wages_of_men : wages_of_men = 5 * M

-- Prove that the total wages of the men are Rs. 30
theorem mens_wages : wages_of_men = 30 :=
by
  -- The proof would go here
  sorry

end mens_wages_l104_104942


namespace hyperbola_foci_condition_l104_104939

theorem hyperbola_foci_condition (m n : ℝ) (h : m * n > 0) :
    (m > 0 ∧ n > 0) ↔ ((∃ (x y : ℝ), m * x^2 - n * y^2 = 1) ∧ (∃ (x y : ℝ), m * x^2 - n * y^2 = 1)) :=
sorry

end hyperbola_foci_condition_l104_104939


namespace polygon_sides_sum_l104_104597

theorem polygon_sides_sum (n : ℕ) (h : (n - 2) * 180 = 1260) : n = 9 :=
by
  sorry

end polygon_sides_sum_l104_104597


namespace gcd_3570_4840_l104_104770

-- Define the numbers
def num1 : Nat := 3570
def num2 : Nat := 4840

-- Define the problem statement
theorem gcd_3570_4840 : Nat.gcd num1 num2 = 10 := by
  sorry

end gcd_3570_4840_l104_104770


namespace bead_necklaces_count_l104_104282

-- Define the conditions
def cost_per_necklace : ℕ := 9
def gemstone_necklaces_sold : ℕ := 3
def total_earnings : ℕ := 90

-- Define the total earnings from gemstone necklaces
def earnings_from_gemstone_necklaces : ℕ := gemstone_necklaces_sold * cost_per_necklace

-- Define the total earnings from bead necklaces
def earnings_from_bead_necklaces : ℕ := total_earnings - earnings_from_gemstone_necklaces

-- Define the number of bead necklaces sold
def bead_necklaces_sold : ℕ := earnings_from_bead_necklaces / cost_per_necklace

-- The statement to be proved
theorem bead_necklaces_count : bead_necklaces_sold = 7 := by
  sorry

end bead_necklaces_count_l104_104282


namespace area_of_square_diagonal_l104_104545

variable (a b : ℝ)
variable (a_gt_b : a > b) (b_pos : b > 0)

theorem area_of_square_diagonal :
  let d := 2 * a - b in
  let area := d^2 / 2 in
  area = (2 * a - b)^2 / 2 :=
by
  sorry

end area_of_square_diagonal_l104_104545


namespace math_expression_equals_2014_l104_104433

-- Define the mapping for each letter
def M : Nat := 1
def A : Nat := 8
def T : Nat := 3
def I : Nat := 9
def K : Nat := 0 -- K corresponds to 'minus', to be used in expression

-- Verification that the expression evaluates to 2014
theorem math_expression_equals_2014 : (100 * M + 10 * A + T) + (1000 * (M + 10 * A + 100 * T) + 100 * M + 10 * A + T + I) - A = 2014 := by
  calc
    (100 * M + 10 * A + T) + (1000 * (M + 10 * A + 100 * T) + 100 * M + 10 * A + T + I) - A
        = (100 * 1 + 10 * 8 + 3) + (1000 * (1 + 10 * 8 + 100 * 3) + 100 * 1 + 10 * 8 + 3 + 9) - 8 : by rfl
    ... = 183 + 1839 - 8 : by rfl
    ... = 2014 : by rfl

end math_expression_equals_2014_l104_104433


namespace prob_statement_l104_104567

open Set

-- Definitions from the conditions
def A : Set ℝ := {-2, -1, 0, 1, 2}
def B : Set ℝ := {x | x^2 + 2 * x < 0}

-- Proposition to be proved
theorem prob_statement : A ∩ (Bᶜ) = {-2, 0, 1, 2} :=
by
  sorry

end prob_statement_l104_104567


namespace sum_of_coefficients_l104_104067

/-- Given the coefficient of the second term in the binomial expansion of (x + 2y)^n is 8,
    prove that the sum of the coefficients of all terms in the expansion of (1 + x) + (1 + x)^2 + ... + (1 + x)^n is 30. -/
theorem sum_of_coefficients (n : ℕ) (h : 2 * n = 8) :
  let S := (1 + x) + (1 + x)^2 + (1 + x)^3 + (1 + x)^4 in
  ∑ k in (finset.range (n + 1)), 2^k = 30 :=
by
  have n_eq : n = 4 := by linarith, 
  sorry

end sum_of_coefficients_l104_104067


namespace order_b_gt_c_gt_a_l104_104834

noncomputable def a : ℝ := Real.log 2.6
def b : ℝ := 0.5 * 1.8^2
noncomputable def c : ℝ := 1.1^5

theorem order_b_gt_c_gt_a : b > c ∧ c > a := by
  sorry

end order_b_gt_c_gt_a_l104_104834


namespace inequality_A_only_inequality_B_not_always_l104_104898

theorem inequality_A_only (a b c : ℝ) (h1 : 2 * b > c) (h2 : c > a) (h3 : c > b) :
  a < c / 3 := 
sorry

theorem inequality_B_not_always (a b c : ℝ) (h1 : 2 * b > c) (h2 : c > a) (h3 : c > b) :
  ¬ (b < c / 3) := 
sorry

end inequality_A_only_inequality_B_not_always_l104_104898


namespace three_7_faced_dice_sum_18_prob_l104_104256

theorem three_7_faced_dice_sum_18_prob :
  let probability := 1 / 7 ^ 3
  in probability * 4 = 4 / 343 :=
by
  sorry

end three_7_faced_dice_sum_18_prob_l104_104256


namespace bobbit_worm_fish_count_l104_104464

theorem bobbit_worm_fish_count 
  (initial_fish : ℕ)
  (fish_eaten_per_day : ℕ)
  (days_before_adding_fish : ℕ)
  (additional_fish : ℕ)
  (days_after_adding_fish : ℕ) :
  days_before_adding_fish = 14 →
  days_after_adding_fish = 7 →
  fish_eaten_per_day = 2 →
  initial_fish = 60 →
  additional_fish = 8 →
  (initial_fish - days_before_adding_fish * fish_eaten_per_day + additional_fish - days_after_adding_fish * fish_eaten_per_day) = 26 :=
by
  intros 
  -- sorry proof goes here
  sorry

end bobbit_worm_fish_count_l104_104464


namespace smallest_z_value_l104_104880

-- Definitions: w, x, y, and z as consecutive even positive integers
def consecutive_even_cubes (w x y z : ℤ) : Prop :=
  w % 2 = 0 ∧ x % 2 = 0 ∧ y % 2 = 0 ∧ z % 2 = 0 ∧
  w < x ∧ x < y ∧ y < z ∧
  x = w + 2 ∧ y = x + 2 ∧ z = y + 2

-- Problem statement: Smallest possible value of z
theorem smallest_z_value :
  ∃ w x y z : ℤ, consecutive_even_cubes w x y z ∧ w^3 + x^3 + y^3 = z^3 ∧ z = 12 :=
by
  sorry

end smallest_z_value_l104_104880


namespace modulo_inverse_product_l104_104563

open Int 

theorem modulo_inverse_product (n : ℕ) (a b c : ℤ) 
  (hn : 0 < n) 
  (ha : a * a.gcd n = 1) 
  (hb : b * b.gcd n = 1) 
  (hc : c * c.gcd n = 1) 
  (hab : (a * b) % n = 1) 
  (hac : (c * a) % n = 1) : 
  ((a * b) * c) % n = c % n :=
by
  sorry

end modulo_inverse_product_l104_104563


namespace approx_val_l104_104903

variable (x : ℝ) (y : ℝ)

-- Definitions based on rounding condition
def approx_0_000315 : ℝ := 0.0003
def approx_7928564 : ℝ := 8000000

-- Main theorem statement
theorem approx_val (h1: x = approx_0_000315) (h2: y = approx_7928564) :
  x * y = 2400 := by
  sorry

end approx_val_l104_104903


namespace base_conversion_l104_104813

theorem base_conversion (b2_to_b10_step : 101101 = 1 * 2 ^ 5 + 0 * 2 ^ 4 + 1 * 2 ^ 3 + 1 * 2 ^ 2 + 0 * 2 + 1)
  (b10_to_b7_step1 : 45 / 7 = 6) (b10_to_b7_step2 : 45 % 7 = 3) (b10_to_b7_step3 : 6 / 7 = 0) (b10_to_b7_step4 : 6 % 7 = 6) :
  101101 = 45 ∧ 45 = 63 :=
by {
  -- Conversion steps from the proof will be filled in here
  sorry
}

end base_conversion_l104_104813


namespace find_second_number_l104_104467

theorem find_second_number 
    (lcm : ℕ) (gcf : ℕ) (num1 : ℕ) (num2 : ℕ)
    (h_lcm : lcm = 56) (h_gcf : gcf = 10) (h_num1 : num1 = 14) 
    (h_product : lcm * gcf = num1 * num2) : 
    num2 = 40 :=
by
  sorry

end find_second_number_l104_104467


namespace least_total_bananas_is_1128_l104_104053

noncomputable def least_total_bananas : ℕ :=
  let b₁ := 252
  let b₂ := 252
  let b₃ := 336
  let b₄ := 288
  b₁ + b₂ + b₃ + b₄

theorem least_total_bananas_is_1128 :
  least_total_bananas = 1128 :=
by
  sorry

end least_total_bananas_is_1128_l104_104053


namespace division_problem_l104_104618

theorem division_problem (x : ℕ) (h : x / 5 = 30 + x / 6) : x = 900 :=
sorry

end division_problem_l104_104618


namespace two_same_color_probability_l104_104410

-- Definitions based on the given conditions
def total_balls := 5
def black_balls := 3
def red_balls := 2

-- Definition for drawing two balls at random
def draw_two_same_color_probability : ℚ :=
  let total_ways := Nat.choose total_balls 2
  let black_pairs := Nat.choose black_balls 2
  let red_pairs := Nat.choose red_balls 2
  (black_pairs + red_pairs) / total_ways

-- Statement of the theorem
theorem two_same_color_probability :
  draw_two_same_color_probability = 2 / 5 :=
  sorry

end two_same_color_probability_l104_104410


namespace regular_polygons_enclosing_hexagon_l104_104951

theorem regular_polygons_enclosing_hexagon (m n : ℕ) 
  (hm : m = 6)
  (h_exterior_angle_central : 180 - ((m - 2) * 180 / m) = 60)
  (h_exterior_angle_enclosing : 2 * 60 = 120): 
  n = 3 := sorry

end regular_polygons_enclosing_hexagon_l104_104951


namespace area_of_circle_l104_104025

noncomputable def calculate_circle_area (center : ℝ×ℝ) (point : ℝ×ℝ) : ℝ :=
  let (x₁, y₁) := center;
  let (x₂, y₂) := point;
  let radius := Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2);
  π * radius^2

theorem area_of_circle (R S : ℝ × ℝ) (hR : R = (-2, 5)) (hS : S = (8, -4)) :
  calculate_circle_area R S = 181 * π := by
  sorry

end area_of_circle_l104_104025


namespace tile_floor_multiple_of_seven_l104_104150

theorem tile_floor_multiple_of_seven (n : ℕ) (a : ℕ)
  (h1 : n * n = 7 * a)
  (h2 : 4 * a / 7 + 3 * a / 7 = a) :
  ∃ k : ℕ, n = 7 * k := by
  sorry

end tile_floor_multiple_of_seven_l104_104150


namespace complex_solution_l104_104536

theorem complex_solution (z : ℂ) (i : ℂ) (h : i^2 = -1) (hz : (3 - 4 * i) * z = 5 * i) : z = (4 / 5) + (3 / 5) * i :=
by {
  sorry
}

end complex_solution_l104_104536


namespace find_a4_l104_104091

variable {a : ℕ → ℝ}

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ (n : ℕ), a (n + 1) - a n = a 1 - a 0

theorem find_a4 (h1 : arithmetic_sequence a) (h2 : a 2 + a 6 = 2) : a 4 = 1 :=
by
  sorry

end find_a4_l104_104091


namespace percentage_failed_in_Hindi_l104_104259

-- Define the percentage of students failed in English
def percentage_failed_in_English : ℝ := 56

-- Define the percentage of students failed in both Hindi and English
def percentage_failed_in_both : ℝ := 12

-- Define the percentage of students passed in both subjects
def percentage_passed_in_both : ℝ := 24

-- Define the total percentage of students
def percentage_total : ℝ := 100

-- Define what we need to prove
theorem percentage_failed_in_Hindi:
  ∃ (H : ℝ), H + percentage_failed_in_English - percentage_failed_in_both + percentage_passed_in_both = percentage_total ∧ H = 32 :=
  by 
    sorry

end percentage_failed_in_Hindi_l104_104259


namespace correct_option_l104_104630

-- Define the options as propositions
def OptionA (a : ℕ) := a ^ 3 * a ^ 5 = a ^ 15
def OptionB (a : ℕ) := a ^ 8 / a ^ 2 = a ^ 4
def OptionC (a : ℕ) := a ^ 2 + a ^ 3 = a ^ 5
def OptionD (a : ℕ) := 3 * a - a = 2 * a

-- Prove that Option D is the only correct statement
theorem correct_option (a : ℕ) : OptionD a ∧ ¬OptionA a ∧ ¬OptionB a ∧ ¬OptionC a :=
by
  sorry

end correct_option_l104_104630


namespace find_matrix_triples_elements_l104_104378

theorem find_matrix_triples_elements (M A : Matrix (Fin 2) (Fin 2) ℝ)
  (h : ∀ (a b c d : ℝ), A = ![![a, b], ![c, d]] -> M * A = ![![3 * a, 3 * b], ![3 * c, 3 * d]]) :
  M = ![![3, 0], ![0, 3]] :=
by
  sorry

end find_matrix_triples_elements_l104_104378


namespace lcm_of_36_and_105_l104_104217

theorem lcm_of_36_and_105 : Nat.lcm 36 105 = 1260 := by
  sorry

end lcm_of_36_and_105_l104_104217


namespace cost_of_used_cd_l104_104727

theorem cost_of_used_cd (N U : ℝ) 
    (h1 : 6 * N + 2 * U = 127.92) 
    (h2 : 3 * N + 8 * U = 133.89) :
    U = 9.99 :=
by 
  sorry

end cost_of_used_cd_l104_104727


namespace divisibility_2_pow_a_plus_1_l104_104955

theorem divisibility_2_pow_a_plus_1 (a b : ℕ) (h_b_pos : 0 < b) (h_b_ge_2 : 2 ≤ b) 
  (h_div : (2^a + 1) % (2^b - 1) = 0) : b = 2 := by
  sorry

end divisibility_2_pow_a_plus_1_l104_104955


namespace reflection_across_y_axis_coordinates_l104_104906

def coordinates_after_reflection (x y : ℤ) : ℤ × ℤ :=
  (-x, y)

theorem reflection_across_y_axis_coordinates :
  coordinates_after_reflection (-3) 4 = (3, 4) :=
by
  sorry

end reflection_across_y_axis_coordinates_l104_104906


namespace bob_and_bill_same_class_probability_l104_104350

-- Definitions based on the conditions mentioned in the original problem
def total_people : ℕ := 32
def allowed_per_class : ℕ := 30
def number_chosen : ℕ := 2
def number_of_classes : ℕ := 2
def bob_and_bill_pair : ℕ := 1

-- Binomial coefficient calculation (32 choose 2)
def binomial_coefficient (n k : ℕ) : ℕ := Nat.choose n k
def total_ways := binomial_coefficient total_people number_chosen

-- Probability that Bob and Bill are chosen
def probability_chosen : ℚ := bob_and_bill_pair / total_ways

-- Probability that Bob and Bill are placed in the same class
def probability_same_class : ℚ := 1 / number_of_classes

-- Total combined probability
def combined_probability : ℚ := probability_chosen * probability_same_class

-- Statement of the theorem
theorem bob_and_bill_same_class_probability :
  combined_probability = 1 / 992 := 
sorry

end bob_and_bill_same_class_probability_l104_104350


namespace polynomial_remainder_l104_104353

theorem polynomial_remainder (y : ℂ) (h1 : y^5 + y^4 + y^3 + y^2 + y + 1 = 0) (h2 : y^6 = 1) :
  (y^55 + y^40 + y^25 + y^10 + 1) % (y^5 + y^4 + y^3 + y^2 + y + 1) = 2 * y + 3 :=
sorry

end polynomial_remainder_l104_104353


namespace minimum_roots_in_interval_l104_104815

noncomputable def g : ℝ → ℝ := sorry

lemma symmetry_condition_1 (x : ℝ) : g (3 + x) = g (3 - x) := sorry
lemma symmetry_condition_2 (x : ℝ) : g (8 + x) = g (8 - x) := sorry
lemma initial_condition : g 1 = 0 := sorry

theorem minimum_roots_in_interval : 
  ∃ k, ∀ x, -1000 ≤ x ∧ x ≤ 1000 → g x = 0 ∧ 
  (2 * k) = 286 := sorry

end minimum_roots_in_interval_l104_104815


namespace sequence_is_geometric_l104_104069

def is_sequence_sum (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, n > 0 → S n = 3 * a n - 3

def geometric_sequence (a : ℕ → ℝ) (r : ℝ) (a₁ : ℝ) : Prop :=
∀ n : ℕ, a n = a₁ * r ^ n

theorem sequence_is_geometric (S : ℕ → ℝ) (a : ℕ → ℝ) :
  is_sequence_sum S a →
  (∃ a₁ : ℝ, ∃ r : ℝ, geometric_sequence a r a₁ ∧ a₁ = 3 / 2 ∧ r = 3 / 2) :=
by
  sorry

end sequence_is_geometric_l104_104069


namespace moles_of_NaHCO3_combined_l104_104372

theorem moles_of_NaHCO3_combined (n_HNO3 n_NaHCO3 : ℕ) (mass_H2O : ℝ) : 
  n_HNO3 = 2 ∧ mass_H2O = 36 ∧ n_HNO3 = n_NaHCO3 → n_NaHCO3 = 2 := by
  sorry

end moles_of_NaHCO3_combined_l104_104372


namespace gcf_75_135_l104_104145

theorem gcf_75_135 : Nat.gcd 75 135 = 15 :=
  by sorry

end gcf_75_135_l104_104145


namespace L_shaped_region_area_l104_104362

noncomputable def area_L_shaped_region (length full_width : ℕ) (sub_length sub_width : ℕ) : ℕ :=
  let area_full_rect := length * full_width
  let small_width := length - sub_length
  let small_height := full_width - sub_width
  let area_small_rect := small_width * small_height
  area_full_rect - area_small_rect

theorem L_shaped_region_area :
  area_L_shaped_region 10 7 3 4 = 49 :=
by sorry

end L_shaped_region_area_l104_104362


namespace tire_circumference_is_one_meter_l104_104868

-- Definitions for the given conditions
def car_speed : ℕ := 24 -- in km/h
def tire_rotations_per_minute : ℕ := 400

-- Conversion factors
def km_to_m : ℕ := 1000
def hour_to_min : ℕ := 60

-- The equivalent proof problem
theorem tire_circumference_is_one_meter 
  (hs : car_speed * km_to_m / hour_to_min = 400 * tire_rotations_per_minute)
  : 400 = 400 * 1 := 
by
  sorry

end tire_circumference_is_one_meter_l104_104868


namespace profit_percent_l104_104780

variable (C S : ℝ)
variable (h : (1 / 3) * S = 0.8 * C)

theorem profit_percent (h : (1 / 3) * S = 0.8 * C) : 
  ((S - C) / C) * 100 = 140 := 
by
  sorry

end profit_percent_l104_104780


namespace problem_statement_l104_104733

variables {a b c : ℝ}

theorem problem_statement 
  (h1 : a^2 + a * b + b^2 = 9)
  (h2 : b^2 + b * c + c^2 = 52)
  (h3 : c^2 + c * a + a^2 = 49) : 
  (49 * b^2 - 33 * b * c + 9 * c^2) / a^2 = 52 :=
by
  sorry

end problem_statement_l104_104733


namespace handrail_length_approximation_l104_104802

noncomputable def handrail_length_of_spiral_staircase 
  (radius : ℝ) (height : ℝ) (angle : ℝ) : ℝ :=
  let circumference := 2 * Real.pi * radius
  let arc_length := (angle / 360) * circumference
  let diagonal := Real.sqrt (height^2 + arc_length^2)
  diagonal

theorem handrail_length_approximation :
  handrail_length_of_spiral_staircase 3 12 180 ≈ 15.3 :=
sorry

end handrail_length_approximation_l104_104802


namespace find_x_l104_104326

variables (a b c d x : ℕ)

theorem find_x (h1 : (a + x) / (b + x) = 4 * a / 3 * b)
               (h2 : a ≠ b)
               (h3 : b ≠ 0)
               (h4 : c = 4 * a)
               (h5 : d = 3 * b) :
x = (a * b) / (3 * b - 4 * a) :=
sorry

end find_x_l104_104326


namespace average_price_of_cow_l104_104335

theorem average_price_of_cow (total_price_cows_and_goats rs: ℕ) (num_cows num_goats: ℕ)
    (avg_price_goat: ℕ) (total_price: total_price_cows_and_goats = 1400)
    (num_cows_eq: num_cows = 2) (num_goats_eq: num_goats = 8)
    (avg_price_goat_eq: avg_price_goat = 60) :
    let total_price_goats := avg_price_goat * num_goats
    let total_price_cows := total_price_cows_and_goats - total_price_goats
    let avg_price_cow := total_price_cows / num_cows
    avg_price_cow = 460 :=
by
  sorry

end average_price_of_cow_l104_104335


namespace quadratic_roots_difference_l104_104361

theorem quadratic_roots_difference (p q : ℝ) (hp : 0 < p) (hq : 0 < q) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 - x2 = 2 ∧ x1 * x2 = q ∧ x1 + x2 = -p) → p = 2 * Real.sqrt (q + 1) :=
by
  sorry

end quadratic_roots_difference_l104_104361


namespace derek_initial_lunch_cost_l104_104817

-- Definitions based on conditions
def derek_initial_money : ℕ := 40
def derek_dad_lunch_cost : ℕ := 11
def derek_more_lunch_cost : ℕ := 5
def dave_initial_money : ℕ := 50
def dave_mom_lunch_cost : ℕ := 7
def dave_difference : ℕ := 33

-- Variable X to represent Derek's initial lunch cost
variable (X : ℕ)

-- Definitions based on conditions
def derek_total_spending (X : ℕ) := X + derek_dad_lunch_cost + derek_more_lunch_cost
def derek_remaining_money (X : ℕ) := derek_initial_money - derek_total_spending X
def dave_remaining_money := dave_initial_money - dave_mom_lunch_cost

-- The main theorem to prove Derek spent $14 initially
theorem derek_initial_lunch_cost (h : dave_remaining_money = derek_remaining_money X + dave_difference) : X = 14 := by
  sorry

end derek_initial_lunch_cost_l104_104817


namespace range_of_b_not_strictly_decreasing_l104_104845

def f (b x : ℝ) : ℝ := -x^3 + b*x^2 - (2*b + 3)*x + 2 - b

theorem range_of_b_not_strictly_decreasing :
  {b : ℝ | ¬(∀ (x1 x2 : ℝ), x1 < x2 → f b x1 > f b x2)} = {b | b < -1 ∨ b > 3} :=
by
  sorry

end range_of_b_not_strictly_decreasing_l104_104845


namespace count_congruent_to_4_mod_7_l104_104862

theorem count_congruent_to_4_mod_7 : 
  ∃ (n : ℕ), 
  n = 71 ∧ 
  ∀ k : ℕ, 0 ≤ k ∧ k ≤ 70 → ∃ m : ℕ, m = 4 + 7 * k ∧ m < 500 := 
by
  sorry

end count_congruent_to_4_mod_7_l104_104862


namespace distance_between_foci_l104_104521

-- Define the given ellipse equation.
def ellipse_eq (x y : ℝ) : Prop := 9 * x^2 + y^2 = 144

-- Provide the values of the semi-major and semi-minor axes.
def a : ℝ := 12
def b : ℝ := 4

-- Define the equation for calculating the distance between the foci.
noncomputable def focal_distance (a b : ℝ) : ℝ :=
  Real.sqrt (a^2 - b^2)

-- The theorem we need to prove.
theorem distance_between_foci : focal_distance a b = 8 * Real.sqrt 2 :=
by sorry

end distance_between_foci_l104_104521


namespace sequence_formula_l104_104093

theorem sequence_formula (a : ℕ → ℕ) (h₀ : a 1 = 1)
  (h₁ : ∀ n : ℕ, a (n + 1) - 2 * a n + 3 = 0) :
  ∀ n : ℕ, a n = 3 - 2^n :=
by
  sorry

end sequence_formula_l104_104093


namespace gretchen_total_earnings_l104_104852

-- Define the conditions
def price_per_drawing : ℝ := 20.0
def caricatures_sold_saturday : ℕ := 24
def caricatures_sold_sunday : ℕ := 16

-- The total caricatures sold
def total_caricatures_sold : ℕ := caricatures_sold_saturday + caricatures_sold_sunday

-- The total amount of money made
def total_money_made : ℝ := total_caricatures_sold * price_per_drawing

-- The theorem to be proven
theorem gretchen_total_earnings : total_money_made = 800.0 := by
  sorry

end gretchen_total_earnings_l104_104852


namespace sufficient_not_necessary_not_necessary_l104_104333

theorem sufficient_not_necessary (x : ℝ) (h1: x > 2) : x^2 - 3 * x + 2 > 0 :=
sorry

theorem not_necessary (x : ℝ) (h2: x^2 - 3 * x + 2 > 0) : (x > 2 ∨ x < 1) :=
sorry

end sufficient_not_necessary_not_necessary_l104_104333


namespace unique_number_outside_range_f_l104_104235

noncomputable def f (a b c d : ℝ) (x : ℝ) : ℝ := (a * x + b) / (c * x + d)

theorem unique_number_outside_range_f (a b c d : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : d ≠ 0)
  (h5 : f a b c d 19 = 19) (h6 : f a b c d 97 = 97)
  (h7 : ∀ x, x ≠ -d / c → f a b c d (f a b c d x) = x) : 
  ∀ y : ℝ, y ≠ 58 → ∃ x : ℝ, f a b c d x ≠ y :=
sorry

end unique_number_outside_range_f_l104_104235


namespace dividend_rate_correct_l104_104652

-- Define the stock's yield and market value
def stock_yield : ℝ := 0.08
def market_value : ℝ := 175

-- Dividend rate definition based on given yield and market value
def dividend_rate (yield market_value : ℝ) : ℝ :=
  (yield * market_value)

-- The problem statement to be proven in Lean
theorem dividend_rate_correct :
  dividend_rate stock_yield market_value = 14 := by
  sorry

end dividend_rate_correct_l104_104652


namespace probability_diagonals_intersect_inside_decagon_l104_104314

/-- Two diagonals of a regular decagon are chosen. 
  What is the probability that their intersection lies inside the decagon?
-/
theorem probability_diagonals_intersect_inside_decagon : 
  let num_diagonals := 35
  let num_pairs := num_diagonals * (num_diagonals - 1) / 2
  let num_intersecting_pairs := 210
  let probability := num_intersecting_pairs / num_pairs
  probability = 42 / 119 :=
by
  -- Definitions based on the conditions
  let num_diagonals := (10 * (10 - 3)) / 2
  let num_pairs := num_diagonals * (num_diagonals - 1) / 2
  let num_intersecting_pairs := 210

  -- Simplified probability
  let probability := num_intersecting_pairs / num_pairs

  -- Sorry used to skip the proof
  sorry

end probability_diagonals_intersect_inside_decagon_l104_104314


namespace find_x_satisfying_condition_l104_104893

def A (x : ℝ) : Set ℝ := {1, 4, x}
def B (x : ℝ) : Set ℝ := {1, x^2}

theorem find_x_satisfying_condition : ∀ x : ℝ, (A x ∪ B x = A x) ↔ (x = 2 ∨ x = -2 ∨ x = 0) := by
  sorry

end find_x_satisfying_condition_l104_104893


namespace min_sum_abc_l104_104915

theorem min_sum_abc (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hprod : a * b * c = 2550) : a + b + c ≥ 48 :=
by sorry

end min_sum_abc_l104_104915


namespace geometric_series_first_term_l104_104918

theorem geometric_series_first_term (a r : ℝ) 
  (h1 : a / (1 - r) = 18) 
  (h2 : a^2 / (1 - r^2) = 72) : 
  a = 7.2 :=
by
  sorry

end geometric_series_first_term_l104_104918


namespace parallelogram_properties_l104_104339

noncomputable def length_adjacent_side_and_area (base height : ℝ) (angle : ℕ) : ℝ × ℝ :=
  let hypotenuse := height / Real.sin (angle * Real.pi / 180)
  let area := base * height
  (hypotenuse, area)

theorem parallelogram_properties :
  ∀ (base height : ℝ) (angle : ℕ),
  base = 12 → height = 6 → angle = 30 →
  length_adjacent_side_and_area base height angle = (12, 72) :=
by
  intros
  sorry

end parallelogram_properties_l104_104339


namespace problem_statement_l104_104073

variable {a b c d : ℚ}

-- Conditions
axiom h1 : a / b = 3
axiom h2 : b / c = 3 / 4
axiom h3 : c / d = 2 / 3

-- Goal
theorem problem_statement : d / a = 2 / 3 := by
  sorry

end problem_statement_l104_104073


namespace percentage_loss_calculation_l104_104589

theorem percentage_loss_calculation
  (initial_cost_euro : ℝ)
  (retail_price_dollars : ℝ)
  (exchange_rate_initial : ℝ)
  (discount1 : ℝ)
  (discount2 : ℝ)
  (sales_tax : ℝ)
  (exchange_rate_new : ℝ)
  (final_sale_price_dollars : ℝ) :
  initial_cost_euro = 800 ∧
  retail_price_dollars = 900 ∧
  exchange_rate_initial = 1.1 ∧
  discount1 = 0.10 ∧
  discount2 = 0.15 ∧
  sales_tax = 0.10 ∧
  exchange_rate_new = 1.5 ∧
  final_sale_price_dollars = (retail_price_dollars * (1 - discount1) * (1 - discount2) * (1 + sales_tax)) →
  ((initial_cost_euro - final_sale_price_dollars / exchange_rate_new) / initial_cost_euro) * 100 = 36.89 := by
  sorry

end percentage_loss_calculation_l104_104589


namespace donna_fully_loaded_truck_weight_l104_104966

-- Define conditions
def empty_truck_weight : ℕ := 12000
def soda_crate_weight : ℕ := 50
def soda_crate_count : ℕ := 20
def dryer_weight : ℕ := 3000
def dryer_count : ℕ := 3

-- Calculate derived weights
def soda_total_weight : ℕ := soda_crate_weight * soda_crate_count
def fresh_produce_weight : ℕ := 2 * soda_total_weight
def dryer_total_weight : ℕ := dryer_weight * dryer_count

-- Define target weight of fully loaded truck
def fully_loaded_truck_weight : ℕ :=
  empty_truck_weight + soda_total_weight + fresh_produce_weight + dryer_total_weight

-- State and prove the theorem
theorem donna_fully_loaded_truck_weight :
  fully_loaded_truck_weight = 24000 :=
by
  -- Provide necessary calculations and proof steps if needed
  sorry

end donna_fully_loaded_truck_weight_l104_104966


namespace gcd_24_36_l104_104045

theorem gcd_24_36 : Int.gcd 24 36 = 12 := by
  sorry

end gcd_24_36_l104_104045


namespace domain_of_function_l104_104976

-- Definitions of the conditions

def sqrt_condition (x : ℝ) : Prop := -x^2 - 3*x + 4 ≥ 0
def log_condition (x : ℝ) : Prop := x + 1 > 0 ∧ x + 1 ≠ 1

-- Statement of the problem

theorem domain_of_function :
  {x : ℝ | sqrt_condition x ∧ log_condition x} = { x | -1 < x ∧ x < 0 ∨ 0 < x ∧ x ≤ 1 } :=
sorry

end domain_of_function_l104_104976


namespace probability_of_intersection_in_decagon_is_fraction_l104_104316

open_locale big_operators

noncomputable def probability_intersecting_diagonals_in_decagon : ℚ :=
let num_points := 10 in
let diagonals := (num_points.choose 2) - num_points in
let total_diagonal_pairs := (diagonals.choose 2) in
let valid_intersections := (num_points.choose 4) in
valid_intersections / total_diagonal_pairs

theorem probability_of_intersection_in_decagon_is_fraction :
  probability_intersecting_diagonals_in_decagon = 42 / 119 :=
by {
  unfold probability_intersecting_diagonals_in_decagon,
  sorry
}

end probability_of_intersection_in_decagon_is_fraction_l104_104316


namespace billboard_shorter_side_length_l104_104076

theorem billboard_shorter_side_length
  (L W : ℝ)
  (h1 : L * W = 91)
  (h2 : 2 * L + 2 * W = 40) :
  L = 7 ∨ W = 7 :=
by sorry

end billboard_shorter_side_length_l104_104076


namespace steve_halfway_longer_than_danny_l104_104814

theorem steve_halfway_longer_than_danny :
  let T_d : Float := 31
  let T_s : Float := 2 * T_d
  (T_s / 2) - (T_d / 2) = 15.5 :=
by
  let T_d : Float := 31
  let T_s : Float := 2 * T_d
  show (T_s / 2) - (T_d / 2) = 15.5
  sorry

end steve_halfway_longer_than_danny_l104_104814


namespace negation_equivalence_l104_104762

variable (U : Type) (S R : U → Prop)

-- Original statement: All students of this university are non-residents, i.e., ∀ x, S(x) → ¬ R(x)
def original_statement : Prop := ∀ x, S x → ¬ R x

-- Negation of the original statement: ∃ x, S(x) ∧ R(x)
def negated_statement : Prop := ∃ x, S x ∧ R x

-- Lean statement to prove that the negation of the original statement is equivalent to some students are residents
theorem negation_equivalence : ¬ original_statement U S R = negated_statement U S R :=
sorry

end negation_equivalence_l104_104762


namespace smallest_range_possible_l104_104491

-- Definition of the problem conditions
def seven_observations (x1 x2 x3 x4 x5 x6 x7 : ℝ) :=
  (x1 + x2 + x3 + x4 + x5 + x6 + x7) / 7 = 9 ∧
  x4 = 10

noncomputable def smallest_range : ℝ :=
  5

-- Lean statement asserting the proof problem
theorem smallest_range_possible (x1 x2 x3 x4 x5 x6 x7 : ℝ) (h : seven_observations x1 x2 x3 x4 x5 x6 x7) :
  ∃ x1' x2' x3' x4' x5' x6' x7', seven_observations x1' x2' x3' x4' x5' x6' x7' ∧ (x7' - x1') = smallest_range :=
sorry

end smallest_range_possible_l104_104491


namespace perpendicular_to_plane_l104_104544

theorem perpendicular_to_plane (Line : Type) (Plane : Type) (triangle : Plane) (circle : Plane)
  (perpendicular1 : Line → Plane → Prop)
  (perpendicular2 : Line → Plane → Prop) :
  (∀ l, ∃ t, perpendicular1 l t ∧ t = triangle) ∧ (∀ l, ∃ c, perpendicular2 l c ∧ c = circle) →
  (∀ l, ∃ p, (perpendicular1 l p ∨ perpendicular2 l p) ∧ (p = triangle ∨ p = circle)) :=
by
  sorry

end perpendicular_to_plane_l104_104544


namespace max_min_values_l104_104685

def f (x : ℝ) : ℝ := 6 - 12 * x + x^3

theorem max_min_values :
  let a := (-1 / 3 : ℝ)
  let b := (1 : ℝ)
  max (f a) (f b) = f a ∧ f a = 269 / 27 ∧ min (f a) (f b) = f b ∧ f b = -5 :=
by
  let a := (-1 / 3 : ℝ)
  let b := (1 : ℝ)
  have ha : f a = 269 / 27 := sorry
  have hb : f b = -5 := sorry
  have max_eq : max (f a) (f b) = f a := by sorry
  have min_eq : min (f a) (f b) = f b := by sorry
  exact ⟨max_eq, ha, min_eq, hb⟩

end max_min_values_l104_104685


namespace daily_profit_functional_relationship_daily_profit_maximizes_at_120_selling_price_for_2400_profit_l104_104033

-- Given conditions
def cost_price : ℝ := 80
def daily_sales_quantity (x : ℝ) : ℝ := -2 * x + 320
def daily_profit (x : ℝ) : ℝ := (x - cost_price) * daily_sales_quantity x

-- Part 1: Functional relationship
theorem daily_profit_functional_relationship (x : ℝ) (hx : 80 ≤ x ∧ x ≤ 160) : daily_profit x = -2 * x^2 + 480 * x - 25600 :=
by sorry

-- Part 2: Maximizing daily profit
theorem daily_profit_maximizes_at_120 (hx : 80 ≤ 120 ∧ 120 ≤ 160) : daily_profit 120 = 3200 :=
by sorry

-- Part 3: Selling price for a daily profit of $2400
theorem selling_price_for_2400_profit (hx : 80 ≤ 100 ∧ 100 ≤ 160) : daily_profit 100 = 2400 :=
by sorry

end daily_profit_functional_relationship_daily_profit_maximizes_at_120_selling_price_for_2400_profit_l104_104033


namespace lana_total_winter_clothing_l104_104417

-- Define the number of boxes, scarves per box, and mittens per box as given in the conditions
def num_boxes : ℕ := 5
def scarves_per_box : ℕ := 7
def mittens_per_box : ℕ := 8

-- The total number of pieces of winter clothing is calculated as total scarves plus total mittens
def total_winter_clothing : ℕ := num_boxes * scarves_per_box + num_boxes * mittens_per_box

-- State the theorem that needs to be proven
theorem lana_total_winter_clothing : total_winter_clothing = 75 := by
  sorry

end lana_total_winter_clothing_l104_104417


namespace reciprocal_of_neg_two_l104_104457

noncomputable theory

def is_reciprocal (a x : ℝ) := a * x = 1

theorem reciprocal_of_neg_two : 
  ∃ x : ℝ, is_reciprocal (-2) x ∧ x = -1 / 2 := 
sorry

end reciprocal_of_neg_two_l104_104457


namespace total_kids_on_soccer_field_l104_104463

theorem total_kids_on_soccer_field (initial_kids : ℕ) (joining_kids : ℕ) (total_kids : ℕ)
  (h₁ : initial_kids = 14)
  (h₂ : joining_kids = 22)
  (h₃ : total_kids = initial_kids + joining_kids) :
  total_kids = 36 :=
by
  sorry

end total_kids_on_soccer_field_l104_104463


namespace randy_trip_length_l104_104288

theorem randy_trip_length (x : ℝ) (h : x / 2 + 30 + x / 4 = x) : x = 120 :=
by
  sorry

end randy_trip_length_l104_104288


namespace find_number_l104_104621

theorem find_number (x : ℝ) (h : x / 5 = 30 + x / 6) : x = 900 :=
  sorry

end find_number_l104_104621


namespace remainder_7_mul_12_pow_24_add_2_pow_24_mod_13_l104_104227

theorem remainder_7_mul_12_pow_24_add_2_pow_24_mod_13 :
  (7 * 12^24 + 2^24) % 13 = 8 := by
  sorry

end remainder_7_mul_12_pow_24_add_2_pow_24_mod_13_l104_104227


namespace scalene_triangle_not_unique_by_two_non_opposite_angles_l104_104329

theorem scalene_triangle_not_unique_by_two_non_opposite_angles
  (α β : ℝ) (h1 : α > 0) (h2 : β > 0) (h3 : α + β < π) :
  ∃ (γ δ : ℝ), γ ≠ δ ∧ γ + α + β = δ + α + β :=
sorry

end scalene_triangle_not_unique_by_two_non_opposite_angles_l104_104329


namespace square_area_is_100_l104_104487

-- Define the point and its distances from the closest sides of the square
variables (P : Type) [metric_space P] 
(inside_square : P)
(distance1 distance2 : ℝ)
(distance_to_side1 distance_to_side2 : inside_square = 1 ∧ inside_square = 2)

-- Define the radius of the inscribed circle
def radius := 5

-- Define the side length of the square as twice the radius of the circle
def side_length := 2 * radius

-- Define the area of the square
def area_of_square := side_length * side_length

-- Prove that given the conditions, the area of the square is 100
theorem square_area_is_100 : 
  area_of_square = 100 :=
by 
  sorry

end square_area_is_100_l104_104487


namespace nth_term_arithmetic_seq_l104_104345

theorem nth_term_arithmetic_seq (a : ℕ → ℝ) (d : ℝ)
  (h_arithmetic : ∀ n : ℕ, ∃ m : ℝ, a (n + 1) = a n + m)
  (h_d_neg : d < 0)
  (h_condition1 : a 2 * a 4 = 12)
  (h_condition2 : a 2 + a 4 = 8):
  ∀ n : ℕ, a n = -2 * n + 10 :=
by
  sorry

end nth_term_arithmetic_seq_l104_104345


namespace club_truncator_equal_wins_losses_l104_104358

noncomputable def probability_equal_wins_losses : ℚ :=
  252 * (1/4)^(10 : ℚ)

theorem club_truncator_equal_wins_losses :
  probability_equal_wins_losses = 63 / 262144 :=
by
  sorry

end club_truncator_equal_wins_losses_l104_104358


namespace max_distance_bicycle_l104_104650

theorem max_distance_bicycle (front_tire_last : ℕ) (rear_tire_last : ℕ) :
  front_tire_last = 5000 ∧ rear_tire_last = 3000 →
  ∃ (max_distance : ℕ), max_distance = 3750 :=
by
  sorry

end max_distance_bicycle_l104_104650


namespace football_team_total_players_l104_104768

theorem football_team_total_players (P : ℕ) (throwers : ℕ) (left_handed : ℕ) (right_handed : ℕ) :
  throwers = 49 →
  right_handed = 63 →
  left_handed = (1/3) * (P - 49) →
  (P - 49) - left_handed = (2/3) * (P - 49) →
  70 = P :=
by
  intros h_throwers h_right_handed h_left_handed h_remaining
  sorry

end football_team_total_players_l104_104768


namespace roof_length_width_difference_l104_104763

theorem roof_length_width_difference (w l : ℝ) 
  (h1 : l = 5 * w) 
  (h2 : l * w = 720) : l - w = 48 := 
sorry

end roof_length_width_difference_l104_104763


namespace lcm_36_105_l104_104223

noncomputable def factorize_36 : List (ℕ × ℕ) := [(2, 2), (3, 2)]
noncomputable def factorize_105 : List (ℕ × ℕ) := [(3, 1), (5, 1), (7, 1)]

theorem lcm_36_105 : Nat.lcm 36 105 = 1260 :=
by
  have h_36 : 36 = 2^2 * 3^2 := by norm_num
  have h_105 : 105 = 3^1 * 5^1 * 7^1 := by norm_num
  sorry

end lcm_36_105_l104_104223


namespace colton_stickers_final_count_l104_104028

-- Definitions based on conditions
def initial_stickers := 200
def stickers_given_to_7_friends := 6 * 7
def stickers_given_to_mandy := stickers_given_to_7_friends + 8
def remaining_after_mandy := initial_stickers - stickers_given_to_7_friends - stickers_given_to_mandy
def stickers_distributed_to_4_friends := remaining_after_mandy / 2
def remaining_after_4_friends := remaining_after_mandy - stickers_distributed_to_4_friends
def given_to_justin := 2 * remaining_after_4_friends / 3
def remaining_after_justin := remaining_after_4_friends - given_to_justin
def given_to_karen := remaining_after_justin / 5
def final_stickers := remaining_after_justin - given_to_karen

-- Theorem to state the proof problem
theorem colton_stickers_final_count : final_stickers = 15 := by
  sorry

end colton_stickers_final_count_l104_104028


namespace reciprocal_of_neg_two_l104_104456

noncomputable theory

def is_reciprocal (a x : ℝ) := a * x = 1

theorem reciprocal_of_neg_two : 
  ∃ x : ℝ, is_reciprocal (-2) x ∧ x = -1 / 2 := 
sorry

end reciprocal_of_neg_two_l104_104456


namespace plant_supplier_money_left_correct_l104_104661

noncomputable def plant_supplier_total_earnings : ℕ :=
  35 * 52 + 30 * 32 + 20 * 77 + 25 * 22 + 40 * 15

noncomputable def plant_supplier_total_expenses : ℕ :=
  3 * 65 + 2 * 45 + 280 + 150 + 100 + 125 + 225 + 550

noncomputable def plant_supplier_money_left : ℕ :=
  plant_supplier_total_earnings - plant_supplier_total_expenses

theorem plant_supplier_money_left_correct :
  plant_supplier_money_left = 3755 :=
by
  sorry

end plant_supplier_money_left_correct_l104_104661


namespace slope_angle_at_point_l104_104526

def f (x : ℝ) : ℝ := 2 * x^3 - 7 * x + 2

theorem slope_angle_at_point :
  let deriv_f := fun x : ℝ => 6 * x^2 - 7
  let slope := deriv_f 1
  let angle := Real.arctan slope
  angle = (3 * Real.pi) / 4 :=
by
  sorry

end slope_angle_at_point_l104_104526


namespace geometric_sequence_tenth_term_l104_104363

theorem geometric_sequence_tenth_term :
  let a := 4
  let r := (12 / 3) / 4
  let nth_term (n : ℕ) := a * r^(n-1)
  nth_term 10 = 4 :=
  by sorry

end geometric_sequence_tenth_term_l104_104363


namespace sqrt_arithmetic_identity_l104_104809

theorem sqrt_arithmetic_identity : 4 * (Real.sqrt 2) * (Real.sqrt 3) - (Real.sqrt 12) / (Real.sqrt 2) + (Real.sqrt 24) = 5 * (Real.sqrt 6) := by
  sorry

end sqrt_arithmetic_identity_l104_104809


namespace number_of_positive_terms_up_to_100_l104_104419

noncomputable def a (n : ℕ) : ℝ := if n = 0 then 0 else (1 / n) * Real.sin (n * Real.pi / 25)

noncomputable def S (n : ℕ) : ℝ := ∑ i in Finset.range (n + 1), a i

theorem number_of_positive_terms_up_to_100 : 
  ∀ n, 1 ≤ n ∧ n ≤ 100 -> S n > 0 := 
by 
  intro n 
  intro h 
  sorry

end number_of_positive_terms_up_to_100_l104_104419


namespace probability_intersecting_diagonals_l104_104309

def number_of_vertices := 10

def number_of_diagonals : ℕ := Nat.choose number_of_vertices 2 - number_of_vertices

def number_of_ways_choose_two_diagonals := Nat.choose number_of_diagonals 2

def number_of_sets_of_intersecting_diagonals : ℕ := Nat.choose number_of_vertices 4

def intersection_probability : ℚ :=
  (number_of_sets_of_intersecting_diagonals : ℚ) / (number_of_ways_choose_two_diagonals : ℚ)

theorem probability_intersecting_diagonals :
  intersection_probability = 42 / 119 :=
by
  sorry

end probability_intersecting_diagonals_l104_104309


namespace permutations_containing_substring_l104_104541

open Nat

/-- Prove that the number of permutations of the string "000011112222" that contain the substring "2020" is equal to 3575. -/
theorem permutations_containing_substring :
  let total_permutations := factorial 8 / (factorial 2 * factorial 4 * factorial 2)
  let num_positions := 9
  let non_overlap_count := total_permutations * num_positions
  let overlap_subtract := 7 * (factorial 6 / (factorial 1 * factorial 4 * factorial 1))
  let add_back := 5 * (factorial 4 / factorial 4)
  non_overlap_count - overlap_subtract + add_back = 3575 := 
by
  let total_permutations := factorial 8 / (factorial 2 * factorial 4 * factorial 2)
  let num_positions := 9
  let non_overlap_count := total_permutations * num_positions
  let overlap_subtract := 7 * (factorial 6 / (factorial 1 * factorial 4 * factorial 1))
  let add_back := 5 * (factorial 4 / factorial 4)
  have h: non_overlap_count - overlap_subtract + add_back = 3575 := by sorry
  exact h

end permutations_containing_substring_l104_104541


namespace H_H_H_one_eq_three_l104_104340

noncomputable def H : ℝ → ℝ := sorry

theorem H_H_H_one_eq_three :
  H 1 = -3 ∧ H (-3) = 3 ∧ H 3 = 3 → H (H (H 1)) = 3 :=
by
  sorry

end H_H_H_one_eq_three_l104_104340


namespace percentage_of_adult_men_l104_104905

theorem percentage_of_adult_men (total_members : ℕ) (children : ℕ) (p : ℕ) :
  total_members = 2000 → children = 200 → 
  (∀ adult_men_percentage : ℕ, adult_women_percentage = 2 * adult_men_percentage) → 
  (100 - p) = 3 * (p - 10) →  p = 30 :=
by sorry

end percentage_of_adult_men_l104_104905


namespace average_speed_l104_104478

theorem average_speed
  (distance1 : ℝ)
  (time1 : ℝ)
  (distance2 : ℝ)
  (time2 : ℝ)
  (total_distance : ℝ)
  (total_time : ℝ)
  (average_speed : ℝ)
  (h1 : distance1 = 90)
  (h2 : time1 = 1)
  (h3 : distance2 = 50)
  (h4 : time2 = 1)
  (h5 : total_distance = distance1 + distance2)
  (h6 : total_time = time1 + time2)
  (h7 : average_speed = total_distance / total_time) :
  average_speed = 70 := 
sorry

end average_speed_l104_104478


namespace common_divisor_of_differences_l104_104713

theorem common_divisor_of_differences 
  (a1 a2 b1 b2 c1 c2 d : ℤ) 
  (h1: d ∣ (a1 - a2)) 
  (h2: d ∣ (b1 - b2)) 
  (h3: d ∣ (c1 - c2)) : 
  d ∣ (a1 * b1 * c1 - a2 * b2 * c2) := 
by sorry

end common_divisor_of_differences_l104_104713


namespace reciprocal_of_neg_two_l104_104448

variable a : ℤ

theorem reciprocal_of_neg_two (h : a = -2) : ∃ b : ℚ, a * b = 1 ∧ b = -1 / 2 := 
by
  exists (-1 / 2)
  rw [h]
  split
  · norm_num
  · refl

end reciprocal_of_neg_two_l104_104448


namespace white_balls_probability_l104_104481

noncomputable def probability_all_white (total_balls white_balls draw_count : ℕ) : ℚ :=
  if h : total_balls >= draw_count ∧ white_balls >= draw_count then
    (Nat.choose white_balls draw_count : ℚ) / (Nat.choose total_balls draw_count : ℚ)
  else
    0

theorem white_balls_probability :
  probability_all_white 11 5 5 = 1 / 462 :=
by
  sorry

end white_balls_probability_l104_104481


namespace no_four_points_with_equal_tangents_l104_104722

theorem no_four_points_with_equal_tangents :
  ∀ (A B C D : ℝ × ℝ),
    A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A ∧
    A ≠ C ∧ B ≠ D →
    ¬ (∀ (P Q : ℝ × ℝ), (P = A ∧ Q = B) ∨ (P = C ∧ Q = D) →
      ∃ (M : ℝ × ℝ) (r : ℝ), M ≠ P ∧ M ≠ Q ∧
      (dist A M = dist C M ∧ dist B M = dist D M ∧
       dist P M > r ∧ dist Q M > r)) :=
by sorry

end no_four_points_with_equal_tangents_l104_104722


namespace base_conversion_and_operations_l104_104612

-- Definitions to convert numbers from bases 7, 5, and 6 to base 10
def base7_to_nat (n : ℕ) : ℕ := 
  8 * 7^0 + 6 * 7^1 + 4 * 7^2 + 2 * 7^3

def base5_to_nat (n : ℕ) : ℕ := 
  1 * 5^0 + 2 * 5^1 + 1 * 5^2

def base6_to_nat (n : ℕ) : ℕ := 
  1 * 6^0 + 5 * 6^1 + 4 * 6^2 + 3 * 6^3

def base7_to_nat2 (n : ℕ) : ℕ := 
  1 * 7^0 + 9 * 7^1 + 8 * 7^2 + 7 * 7^3

-- Problem statement: Perform the arithmetical operations
theorem base_conversion_and_operations : 
  (base7_to_nat 2468 / base5_to_nat 121) - base6_to_nat 3451 + base7_to_nat2 7891 = 2059 := 
by
  sorry

end base_conversion_and_operations_l104_104612


namespace multiple_of_fair_tickets_l104_104302

theorem multiple_of_fair_tickets (fair_tickets_sold : ℕ) (game_tickets_sold : ℕ) (h : fair_tickets_sold = game_tickets_sold * x + 6) :
  25 = 56 * x + 6 → x = 19 / 56 := by
  sorry

end multiple_of_fair_tickets_l104_104302


namespace max_value_f_compare_magnitude_l104_104244

open Real

def f (x : ℝ) : ℝ := abs (x - 1) - 2 * abs (x + 1)

-- 1. Prove that the maximum value of f(x) is 2.
theorem max_value_f : ∃ x : ℝ, f x = 2 :=
sorry

-- 2. Given the condition, prove 2m + n > 2.
theorem compare_magnitude (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : (1 / m) + (1 / (2 * n)) = 2) : 
  2 * m + n > 2 :=
sorry

end max_value_f_compare_magnitude_l104_104244


namespace max_distinct_fans_l104_104644

-- Define the problem conditions and the main statement
theorem max_distinct_fans : 
  let n := 6  -- Number of sectors per fan
  let configurations := 2^n  -- Total configurations without considering symmetry
  let unchanged_flips := 8  -- Number of configurations unchanged by flipping
  let distinct_configurations := (configurations - unchanged_flips) / 2 + unchanged_flips 
  in 
  distinct_configurations = 36 := by sorry  # We state the answer based on the provided steps


end max_distinct_fans_l104_104644


namespace option_C_is_quadratic_l104_104629

-- Define the conditions
def option_A (x : ℝ) : Prop := 2 * x = 3
def option_B (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0
def option_C (x : ℝ) : Prop := (4 * x - 3) * (3 * x + 1) = 0
def option_D (x : ℝ) : Prop := (x + 3) * (x - 2) = (x - 2) * (x + 1)

-- Define what it means to be a quadratic equation
def is_quadratic (f : ℝ → Prop) : Prop :=
  ∃ a b c : ℝ, (∀ x, f x = (a * x^2 + b * x + c = 0)) ∧ a ≠ 0

-- The main theorem statement
theorem option_C_is_quadratic : is_quadratic option_C :=
sorry

end option_C_is_quadratic_l104_104629


namespace sum_of_5_and_8_l104_104772

theorem sum_of_5_and_8 : 5 + 8 = 13 := by
  rfl

end sum_of_5_and_8_l104_104772


namespace sin_of_7pi_over_6_l104_104198

theorem sin_of_7pi_over_6 : Real.sin (7 * Real.pi / 6) = -1 / 2 :=
by
  -- Conditions from the statement in a)
  -- Given conditions: \(\sin (180^\circ + \theta) = -\sin \theta\)
  -- \(\sin 30^\circ = \frac{1}{2}\)
  sorry

end sin_of_7pi_over_6_l104_104198


namespace superhero_movies_l104_104672

theorem superhero_movies (d h a together : ℕ) (H1: d = 7) (H2: h = 12) (H3: a = 15) (H4: together = 2) :
  (d + h + a - together) = 32 :=
by
  rw [H1, H2, H3, H4]
  norm_num

end superhero_movies_l104_104672


namespace f_range_x_range_l104_104234

noncomputable def f (x : ℝ) : ℝ := (Real.log x / Real.log 2 - 3) * (1 + Real.log x / Real.log 2)
noncomputable def g (x : ℝ) : ℝ := 4^x - 2^(x + 1) - 3

theorem f_range : ∀ y ∈ Set.Icc (-4:ℝ) (Real.infinity), ∃ x : ℝ, f x = y := 
sorry

theorem x_range (x : ℝ) : 
  (∀ a ∈ Set.Icc (1 / 2) 2, f x - g a ≤ 0) ↔ x ∈ Set.Icc (2^(2 - Real.sqrt 2)) (2^(Real.sqrt 2)) :=
sorry

end f_range_x_range_l104_104234


namespace Donovan_percentage_correct_l104_104678

-- Definitions based on conditions from part a)
def fullyCorrectAnswers : ℕ := 35
def incorrectAnswers : ℕ := 13
def partiallyCorrectAnswers : ℕ := 7
def pointPerFullAnswer : ℝ := 1
def pointPerPartialAnswer : ℝ := 0.5

-- Lean 4 statement to prove the problem mathematically
theorem Donovan_percentage_correct : 
  (fullyCorrectAnswers * pointPerFullAnswer + partiallyCorrectAnswers * pointPerPartialAnswer) / 
  (fullyCorrectAnswers + incorrectAnswers + partiallyCorrectAnswers) * 100 = 70.00 :=
by
  sorry

end Donovan_percentage_correct_l104_104678


namespace min_value_of_m_l104_104709

theorem min_value_of_m : (2 ∈ {x | ∃ (m : ℤ), x * (x - m) < 0}) → ∃ (m : ℤ), m = 3 :=
by
  sorry

end min_value_of_m_l104_104709


namespace KimFridayToMondayRatio_l104_104556

variable (MondaySweaters : ℕ) (TuesdaySweaters : ℕ) (WednesdaySweaters : ℕ) (ThursdaySweaters : ℕ) (FridaySweaters : ℕ)

def KimSweaterKnittingConditions (MondaySweaters TuesdaySweaters WednesdaySweaters ThursdaySweaters FridaySweaters : ℕ) : Prop :=
  MondaySweaters = 8 ∧
  TuesdaySweaters = MondaySweaters + 2 ∧
  WednesdaySweaters = TuesdaySweaters - 4 ∧
  ThursdaySweaters = TuesdaySweaters - 4 ∧
  MondaySweaters + TuesdaySweaters + WednesdaySweaters + ThursdaySweaters + FridaySweaters = 34

theorem KimFridayToMondayRatio 
  (MondaySweaters TuesdaySweaters WednesdaySweaters ThursdaySweaters FridaySweaters : ℕ)
  (h : KimSweaterKnittingConditions MondaySweaters TuesdaySweaters WednesdaySweaters ThursdaySweaters FridaySweaters) :
  FridaySweaters / MondaySweaters = 1/2 :=
  sorry

end KimFridayToMondayRatio_l104_104556


namespace real_number_a_l104_104241

theorem real_number_a (a : ℝ) (ha : ∃ b : ℝ, z = 0 + bi) : a = 1 :=
sorry

end real_number_a_l104_104241


namespace calculate_expression_is_correct_l104_104355

noncomputable def calculate_expression : ℝ :=
  -(-2) + 2 * Real.cos (Real.pi / 3) + (-1 / 8)⁻¹ + (Real.pi - 3.14) ^ 0

theorem calculate_expression_is_correct :
  calculate_expression = -4 :=
by
  -- the conditions as definitions
  have h1 : Real.cos (Real.pi / 3) = 1 / 2 := by sorry
  have h2 : (Real.pi - 3.14) ^ 0 = 1 := by sorry
  -- use these conditions to prove the main statement
  sorry

end calculate_expression_is_correct_l104_104355


namespace monotonic_increasing_implies_range_l104_104392

open Real

noncomputable def f (x : ℝ) : ℝ := 1 / 2 * x ^ 2 + 2 * x - 2 * log x

theorem monotonic_increasing_implies_range (a : ℝ) :
  (∀ x > (0 : ℝ), deriv f x ≥ 0) → a ≤ 1 :=
  by 
  sorry

end monotonic_increasing_implies_range_l104_104392


namespace determine_constants_l104_104188

theorem determine_constants :
  ∃ P Q R : ℚ, (∀ x : ℚ, x ≠ 1 → x ≠ 4 → x ≠ 6 → (x^2 - 4 * x + 8) / ((x - 1) * (x - 4) * (x - 6)) = P / (x - 1) + Q / (x - 4) + R / (x - 6)) ∧ 
  P = 1 / 3 ∧ Q = - 4 / 3 ∧ R = 2 :=
by
  -- Proof is left as a placeholder
  sorry

end determine_constants_l104_104188


namespace boy_scouts_signed_slips_l104_104657

-- Definitions for the problem conditions have only been used; solution steps are excluded.

theorem boy_scouts_signed_slips (total_scouts : ℕ) (signed_slips : ℕ) (boy_scouts : ℕ) (girl_scouts : ℕ)
  (boy_scouts_signed : ℕ) (girl_scouts_signed : ℕ)
  (h1 : signed_slips = 4 * total_scouts / 5)  -- 80% of the scouts arrived with signed permission slips
  (h2 : boy_scouts = 2 * total_scouts / 5)  -- 40% of the scouts were boy scouts
  (h3 : girl_scouts = total_scouts - boy_scouts)  -- Rest are girl scouts
  (h4 : girl_scouts_signed = 8333 * girl_scouts / 10000)  -- 83.33% of girl scouts with permission slips
  (h5 : signed_slips = boy_scouts_signed + girl_scouts_signed)  -- Total signed slips by both boy and girl scouts
  : (boy_scouts_signed * 100 / boy_scouts = 75) :=    -- 75% of boy scouts with permission slips
by
  -- Proof to be filled in.
  sorry

end boy_scouts_signed_slips_l104_104657


namespace range_of_fx1_l104_104246

noncomputable def f (x a : ℝ) : ℝ := x^2 - 2*x + 1 + a * Real.log x

theorem range_of_fx1 (a x1 x2 : ℝ) (h1 : 0 < x1) (h2 : x1 < x2) (h3 : f x1 a = 0) (h4 : f x2 a = 0) :
    f x1 a > (1 - 2 * Real.log 2) / 4 :=
sorry

end range_of_fx1_l104_104246


namespace number_of_tiles_per_row_l104_104442

-- Definitions of conditions
def area : ℝ := 320
def length : ℝ := 16
def tile_size : ℝ := 1

-- Theorem statement
theorem number_of_tiles_per_row : (area / length) / tile_size = 20 := by
  sorry

end number_of_tiles_per_row_l104_104442


namespace inequality_solution_l104_104983

-- Definitions
variables {a b : ℝ}

-- Hypothesis
variable (h : a > b)

-- Theorem
theorem inequality_solution : -2 * a < -2 * b :=
sorry

end inequality_solution_l104_104983


namespace sin_of_7pi_over_6_l104_104199

theorem sin_of_7pi_over_6 : Real.sin (7 * Real.pi / 6) = -1 / 2 :=
by
  -- Conditions from the statement in a)
  -- Given conditions: \(\sin (180^\circ + \theta) = -\sin \theta\)
  -- \(\sin 30^\circ = \frac{1}{2}\)
  sorry

end sin_of_7pi_over_6_l104_104199


namespace shaded_square_probability_l104_104000

theorem shaded_square_probability :
  let length := 2024 in
  let middle := length / 2 in
  let n_rectangles := (length + 1) * length / 2 in
  let m_shaded := middle * middle in
  let prob_shaded := m_shaded / n_rectangles in
  1 - prob_shaded = 1 / 2 :=
sorry

end shaded_square_probability_l104_104000


namespace min_lcm_leq_six_floor_l104_104637

theorem min_lcm_leq_six_floor (n : ℕ) (h : n ≠ 4) (a : Fin n → ℕ) 
  (h1 : ∀ i, 0 < a i ∧ a i ≤ 2 * n) : 
  ∃ i j, i < j ∧ Nat.lcm (a i) (a j) ≤ 6 * (n / 2 + 1) :=
by
  sorry

end min_lcm_leq_six_floor_l104_104637


namespace inequality_order_l104_104980

theorem inequality_order (a b : ℝ) (h1 : b > a) (h2 : a > 0) (h3 : a + b = 1) : 
  b > (a^4 - b^4) / (a - b) ∧ (a^4 - b^4) / (a - b) > (a + b) / 2 ∧ (a + b) / 2 > 2 * a * b :=
by 
  sorry

end inequality_order_l104_104980


namespace part1_part2_part3_l104_104245

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  x + a / x + Real.log x

noncomputable def f' (x : ℝ) (a : ℝ) : ℝ :=
  1 - a / x^2 + 1 / x

noncomputable def g (x : ℝ) (a : ℝ) : ℝ :=
  f' x a - x

theorem part1 (a : ℝ) (h : f' 1 a = 0) : a = 2 :=
  sorry

theorem part2 {a : ℝ} (h : ∀ x, 1 < x → x < 2 → f' x a ≥ 0) : a ≤ 2 :=
  sorry

theorem part3 (a : ℝ) :
  ((a > 1 → ∀ x, g x a ≠ 0) ∧ 
  (a = 1 ∨ a ≤ 0 → ∃ x, g x a = 0 ∧ ∀ y, g y a = 0 → y = x) ∧ 
  (0 < a ∧ a < 1 → ∃ x y, x ≠ y ∧ g x a = 0 ∧ g y a = 0)) :=
  sorry

end part1_part2_part3_l104_104245


namespace tangent_line_at_1_1_l104_104375

noncomputable def f (x : ℝ) : ℝ := x / (2 * x - 1)

theorem tangent_line_at_1_1 :
  let m := -((2 * 1 - 1 - 2 * 1) / (2 * 1 - 1)^2) -- Derivative evaluated at x = 1
  let tangent_line (x y : ℝ) := x + y - 2
  ∀ x y : ℝ, tangent_line x y = 0 → (f x = y ∧ x = 1 → y = 1 → m = -1) :=
by
  sorry

end tangent_line_at_1_1_l104_104375


namespace number_of_lamps_bought_l104_104886

-- Define the given conditions
def price_of_lamp : ℕ := 7
def price_of_bulb : ℕ := price_of_lamp - 4
def bulbs_bought : ℕ := 6
def total_spent : ℕ := 32

-- Define the statement to prove
theorem number_of_lamps_bought : 
  ∃ (L : ℕ), (price_of_lamp * L + price_of_bulb * bulbs_bought = total_spent) ∧ (L = 2) :=
sorry

end number_of_lamps_bought_l104_104886


namespace simplify_product_l104_104114

theorem simplify_product (b : ℤ) : (1 * 2 * b * 3 * b^2 * 4 * b^3 * 5 * b^4 * 6 * b^5) = 720 * b^15 := by
  sorry

end simplify_product_l104_104114


namespace dice_prob_l104_104143

noncomputable def probability_four_twos : ℝ :=
  let total_ways := Nat.choose 12 4
  let prob_each_arrangement := (1 / 8)^4 * (7 / 8)^8
  in total_ways * prob_each_arrangement

theorem dice_prob : probability_four_twos = 0.089 := by
  sorry

end dice_prob_l104_104143


namespace probability_diagonals_intersect_inside_decagon_l104_104313

/-- Two diagonals of a regular decagon are chosen. 
  What is the probability that their intersection lies inside the decagon?
-/
theorem probability_diagonals_intersect_inside_decagon : 
  let num_diagonals := 35
  let num_pairs := num_diagonals * (num_diagonals - 1) / 2
  let num_intersecting_pairs := 210
  let probability := num_intersecting_pairs / num_pairs
  probability = 42 / 119 :=
by
  -- Definitions based on the conditions
  let num_diagonals := (10 * (10 - 3)) / 2
  let num_pairs := num_diagonals * (num_diagonals - 1) / 2
  let num_intersecting_pairs := 210

  -- Simplified probability
  let probability := num_intersecting_pairs / num_pairs

  -- Sorry used to skip the proof
  sorry

end probability_diagonals_intersect_inside_decagon_l104_104313


namespace service_cleaning_fee_percentage_is_correct_l104_104885

noncomputable def daily_rate : ℝ := 125
noncomputable def pet_fee : ℝ := 100
noncomputable def duration : ℕ := 14
noncomputable def security_deposit_percentage : ℝ := 0.5
noncomputable def security_deposit : ℝ := 1110

noncomputable def total_expected_cost : ℝ := (daily_rate * duration) + pet_fee
noncomputable def entire_bill : ℝ := security_deposit / security_deposit_percentage
noncomputable def service_cleaning_fee : ℝ := entire_bill - total_expected_cost

theorem service_cleaning_fee_percentage_is_correct : 
  (service_cleaning_fee / entire_bill) * 100 = 16.67 :=
by 
  sorry

end service_cleaning_fee_percentage_is_correct_l104_104885


namespace quadratic_inequality_solution_set_l104_104833

variable (a b c : ℝ) (α β : ℝ)

theorem quadratic_inequality_solution_set
  (hαβ : α < β)
  (hα_lt_0 : α < 0) 
  (hβ_lt_0 : β < 0)
  (h_sol_set : ∀ x : ℝ, a * x^2 + b * x + c < 0 ↔ (x < α ∨ x > β)) :
  (∀ x : ℝ, c * x^2 - b * x + a > 0 ↔ (-(1 / α) < x ∧ x < -(1 / β))) :=
  sorry

end quadratic_inequality_solution_set_l104_104833


namespace distance_between_foci_l104_104522

-- Define the given ellipse equation.
def ellipse_eq (x y : ℝ) : Prop := 9 * x^2 + y^2 = 144

-- Provide the values of the semi-major and semi-minor axes.
def a : ℝ := 12
def b : ℝ := 4

-- Define the equation for calculating the distance between the foci.
noncomputable def focal_distance (a b : ℝ) : ℝ :=
  Real.sqrt (a^2 - b^2)

-- The theorem we need to prove.
theorem distance_between_foci : focal_distance a b = 8 * Real.sqrt 2 :=
by sorry

end distance_between_foci_l104_104522


namespace find_d_minus_r_l104_104252

theorem find_d_minus_r :
  ∃ (d r : ℕ), d > 1 ∧ 1083 % d = r ∧ 1455 % d = r ∧ 2345 % d = r ∧ d - r = 1 := by
  sorry

end find_d_minus_r_l104_104252


namespace probability_diagonals_intersect_inside_decagon_l104_104312

/-- Two diagonals of a regular decagon are chosen. 
  What is the probability that their intersection lies inside the decagon?
-/
theorem probability_diagonals_intersect_inside_decagon : 
  let num_diagonals := 35
  let num_pairs := num_diagonals * (num_diagonals - 1) / 2
  let num_intersecting_pairs := 210
  let probability := num_intersecting_pairs / num_pairs
  probability = 42 / 119 :=
by
  -- Definitions based on the conditions
  let num_diagonals := (10 * (10 - 3)) / 2
  let num_pairs := num_diagonals * (num_diagonals - 1) / 2
  let num_intersecting_pairs := 210

  -- Simplified probability
  let probability := num_intersecting_pairs / num_pairs

  -- Sorry used to skip the proof
  sorry

end probability_diagonals_intersect_inside_decagon_l104_104312


namespace cities_drawn_from_group_b_l104_104135

def group_b_cities : ℕ := 8
def selection_probability : ℝ := 0.25

theorem cities_drawn_from_group_b : 
  group_b_cities * selection_probability = 2 :=
by
  sorry

end cities_drawn_from_group_b_l104_104135


namespace arithmetic_sequence_a12_l104_104090

theorem arithmetic_sequence_a12 (a : ℕ → ℝ)
    (h1 : a 3 + a 4 + a 5 = 3)
    (h2 : a 8 = 8)
    (h_arith : ∃ d, ∀ n, a (n + 1) = a n + d) :
    a 12 = 15 :=
by
  -- Since we aim to ensure the statement alone compiles, we leave the proof with 'sorry'.
  sorry

end arithmetic_sequence_a12_l104_104090


namespace valid_quadratic_polynomials_l104_104683

theorem valid_quadratic_polynomials (b c : ℤ)
  (h₁ : ∃ x₁ x₂ : ℤ, b = -(x₁ + x₂) ∧ c = x₁ * x₂)
  (h₂ : 1 + b + c = 10) :
  (b = -13 ∧ c = 22) ∨ (b = -9 ∧ c = 18) ∨ (b = 9 ∧ c = 0) ∨ (b = 5 ∧ c = 4) := sorry

end valid_quadratic_polynomials_l104_104683


namespace oliver_total_money_l104_104739

-- Define the initial conditions
def initial_amount : ℕ := 9
def allowance_saved : ℕ := 5
def cost_frisbee : ℕ := 4
def cost_puzzle : ℕ := 3
def birthday_money : ℕ := 8

-- Define the problem statement in Lean
theorem oliver_total_money : 
  (initial_amount + allowance_saved - (cost_frisbee + cost_puzzle) + birthday_money) = 15 := 
by 
  sorry

end oliver_total_money_l104_104739


namespace reciprocal_of_neg_two_l104_104450

theorem reciprocal_of_neg_two : ∃ x : ℝ, (-2) * x = 1 ∧ x = -1/2 :=
by
  use -1/2
  split
  · exact (by norm_num : (-2) * (-1/2) = 1)
  · rfl

end reciprocal_of_neg_two_l104_104450


namespace robbie_weight_l104_104576

theorem robbie_weight (R P : ℝ) 
  (h1 : P = 4.5 * R - 235)
  (h2 : P = R + 115) :
  R = 100 := 
by 
  sorry

end robbie_weight_l104_104576


namespace bicycle_price_l104_104424

theorem bicycle_price (P : ℝ) (h : 0.2 * P = 200) : P = 1000 := 
by
  sorry

end bicycle_price_l104_104424


namespace part_a_answer_part_b_answer_l104_104156

noncomputable def part_a_problem : Prop :=
  ∃! (x k : ℕ), x > 0 ∧ k > 0 ∧ 3^k - 1 = x^3

noncomputable def part_b_problem (n : ℕ) : Prop :=
  n > 1 ∧ n ≠ 3 → ∀ (x k : ℕ), ¬ (x > 0 ∧ k > 0 ∧ 3^k - 1 = x^n)

theorem part_a_answer : part_a_problem :=
  sorry

theorem part_b_answer (n : ℕ) : part_b_problem n :=
  sorry

end part_a_answer_part_b_answer_l104_104156


namespace lily_lemonade_calories_l104_104276

def total_weight (lemonade_lime_juice lemonade_honey lemonade_water : ℕ) : ℕ :=
  lemonade_lime_juice + lemonade_honey + lemonade_water

def total_calories (weight_lime_juice weight_honey : ℕ) : ℚ :=
  (30 * weight_lime_juice / 100) + (305 * weight_honey / 100)

def calories_in_portion (total_weight total_calories portion_weight : ℚ) : ℚ :=
  (total_calories * portion_weight) / total_weight

theorem lily_lemonade_calories :
  let lemonade_lime_juice := 150
  let lemonade_honey := 150
  let lemonade_water := 450
  let portion_weight := 300
  let total_weight := total_weight lemonade_lime_juice lemonade_honey lemonade_water
  let total_calories := total_calories lemonade_lime_juice lemonade_honey
  calories_in_portion total_weight total_calories portion_weight = 201 := 
by
  sorry

end lily_lemonade_calories_l104_104276


namespace f_neg2_eq_neg4_l104_104890

noncomputable def f (x : ℝ) : ℝ :=
  if hx : x >= 0 then 3^x - 2*x - 1
  else - (3^(-x) - 2*(-x) - 1)

theorem f_neg2_eq_neg4
: f (-2) = -4 :=
by
  sorry

end f_neg2_eq_neg4_l104_104890


namespace no_negative_roots_l104_104189

theorem no_negative_roots (x : ℝ) :
  x^4 - 4 * x^3 - 6 * x^2 - 3 * x + 9 = 0 → 0 ≤ x :=
by
  sorry

end no_negative_roots_l104_104189


namespace inequality_problem_l104_104565

theorem inequality_problem
  (a b c : ℝ)
  (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ( ( (2 * a + b + c) ^ 2 ) / ( 2 * a ^ 2 + (b + c) ^ 2 ) ) +
  ( ( (a + 2 * b + c) ^ 2 ) / ( 2 * b ^ 2 + (c + a) ^ 2 ) ) +
  ( ( (a + b + 2 * c) ^ 2 ) / ( 2 * c ^ 2 + (a + b) ^ 2 ) ) ≤ 8 :=
by
  sorry

end inequality_problem_l104_104565


namespace sides_of_polygon_l104_104601

-- Define the conditions
def polygon_sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

-- State the mathematically equivalent proof problem
theorem sides_of_polygon (n : ℕ) (h : polygon_sum_interior_angles n = 1260) : n = 9 := by
  sorry

end sides_of_polygon_l104_104601


namespace class_mean_calculation_correct_l104_104409

variable (s1 s2 : ℕ) (mean1 mean2 : ℕ)
variable (n : ℕ) (mean_total : ℕ)

def overall_class_mean (s1 s2 mean1 mean2 : ℕ) : ℕ :=
  let total_score := (s1 * mean1) + (s2 * mean2)
  total_score / (s1 + s2)

theorem class_mean_calculation_correct
  (h1 : s1 = 40)
  (h2 : s2 = 10)
  (h3 : mean1 = 80)
  (h4 : mean2 = 90)
  (h5 : n = 50)
  (h6 : mean_total = 82) :
  overall_class_mean s1 s2 mean1 mean2 = mean_total :=
  sorry

end class_mean_calculation_correct_l104_104409


namespace best_scrap_year_limit_l104_104908

theorem best_scrap_year_limit
    (purchase_cost : ℝ)
    (annual_expenses : ℝ)
    (base_maintenance_cost : ℝ)
    (annual_maintenance_increase : ℝ)
    (n : ℕ)
    (n_min_avg : ℝ) :
    purchase_cost = 150000 ∧
    annual_expenses = 15000 ∧
    base_maintenance_cost = 3000 ∧
    annual_maintenance_increase = 3000 ∧
    n = 10 →
    n_min_avg = 10 := by
  sorry

end best_scrap_year_limit_l104_104908


namespace numerator_of_fraction_l104_104549

-- Define the conditions
def y_pos (y : ℝ) : Prop := y > 0

-- Define the equation
def equation (x y : ℝ) : Prop := x + (3 * y) / 10 = (1 / 2) * y

-- Prove that x = (1/5) * y given the conditions
theorem numerator_of_fraction {y x : ℝ} (h1 : y_pos y) (h2 : equation x y) : x = (1/5) * y :=
  sorry

end numerator_of_fraction_l104_104549


namespace range_of_a_l104_104399

theorem range_of_a (a : ℝ) (x : ℝ) : (|x + a| < 3) ↔ (2 < x ∧ x < 3) →  a ∈ Icc (-5 : ℝ) (0 : ℝ) := 
sorry

end range_of_a_l104_104399


namespace total_exercise_time_l104_104724

theorem total_exercise_time :
  let javier_minutes_per_day := 50
  let javier_days := 7
  let sanda_minutes_per_day := 90
  let sanda_days := 3
  (javier_minutes_per_day * javier_days + sanda_minutes_per_day * sanda_days) = 620 :=
by
  sorry

end total_exercise_time_l104_104724


namespace inclination_angle_of_line_l104_104132

theorem inclination_angle_of_line (θ : Real) 
  (h : θ = Real.tan 45) : θ = 90 :=
sorry

end inclination_angle_of_line_l104_104132


namespace compound_interest_difference_l104_104444

variable (P r : ℝ)

theorem compound_interest_difference :
  (P * 9 * r^2 = 360) → (P * r^2 = 40) :=
by
  sorry

end compound_interest_difference_l104_104444


namespace algebraic_expression_value_l104_104248

theorem algebraic_expression_value 
  (θ : ℝ)
  (a := (Real.cos θ, Real.sin θ))
  (b := (1, -2))
  (parallel : ∃ k : ℝ, a = (k * 1, k * -2)) :
  (2 * Real.sin θ - Real.cos θ) / (Real.sin θ + Real.cos θ) = 5 := 
by 
  -- proof goes here 
  sorry

end algebraic_expression_value_l104_104248


namespace common_point_of_geometric_progression_l104_104499

theorem common_point_of_geometric_progression (a b c x y : ℝ) (r : ℝ) 
  (h1 : b = a * r) (h2 : c = a * r^2) 
  (h3 : a * x + b * y = c) : 
  x = 1 / 2 ∧ y = -1 / 2 := 
sorry

end common_point_of_geometric_progression_l104_104499


namespace math_proof_problem_l104_104838

noncomputable def problemStatement : Prop :=
  ∃ (α : ℝ), 5 * Real.sin (2 * α) = Real.sin (2 * Real.pi / 180) ∧ 
  (Real.tan (α + Real.pi / 180) / Real.tan (α - Real.pi / 180) = -3/2)

theorem math_proof_problem : problemStatement := 
by 
  sorry

end math_proof_problem_l104_104838


namespace polynomial_transformation_l104_104423

theorem polynomial_transformation (g : Polynomial ℝ) (x : ℝ)
  (h : g.eval (x^2 + 2) = x^4 + 6 * x^2 + 8 * x) : 
  g.eval (x^2 - 1) = x^4 - 1 := by
  sorry

end polynomial_transformation_l104_104423


namespace sum_eq_two_l104_104575

theorem sum_eq_two (x y : ℝ) (h : x^2 + y^2 = 10 * x - 6 * y - 34) : x + y = 2 :=
by
  sorry

end sum_eq_two_l104_104575


namespace min_h_condition_l104_104701

open Real

noncomputable def ellipse (a b : ℝ) : set (ℝ × ℝ) :=
{ p | let x := p.1, let y := p.2 in y^2 / a^2 + x^2 / b^2 = 1 }

def parabola (h : ℝ) : set (ℝ × ℝ) :=
{ p | let x := p.1, let y := p.2 in y = x^2 + h }

def midpoint_x (A B : ℝ × ℝ) : ℝ := (A.1 + B.1) / 2

theorem min_h_condition (a b : ℝ)  (h : ℝ) :
  a > b ∧ b > 0 ∧
  ellipse a b = { p : ℝ × ℝ | p.2^2 / a^2 + p.1^2 / b^2 = 1 } ∧
  ellipse a b ⊆ {(-a, 0), (a, 0), (0, -b), (0, b)} ∧
  right_vertex (1, 0) (ellipse a b) ∧
  chord_perpendicular_to_major_axis a b = 1 ∧
  let P : ℝ × ℝ := (t, t^2 + h) in
  tangent_to_parabola P parabola intersects_ellipse M N ∧
  let midpoint_AP_x := midpoint_x A P in
  let midpoint_MN_x := midpoint_x M N in
  midpoint_AP_x = midpoint_MN_x
→ h = 1 :=
sorry

end min_h_condition_l104_104701


namespace algebraic_expression_equals_one_l104_104533

variable (m n : ℝ)

theorem algebraic_expression_equals_one
  (hm : m ≠ 0)
  (hn : n ≠ 0)
  (h_eq : m - n = 1 / 2) :
  (m^2 - n^2) / (2 * m^2 + 2 * m * n) / (m - (2 * m * n - n^2) / m) = 1 :=
by
  sorry

end algebraic_expression_equals_one_l104_104533


namespace cost_per_serving_l104_104435

def pasta_cost : ℝ := 1.0
def sauce_cost : ℝ := 2.0
def meatballs_cost : ℝ := 5.0
def total_servings : ℝ := 8.0

theorem cost_per_serving : (pasta_cost + sauce_cost + meatballs_cost) / total_servings = 1.0 :=
by sorry

end cost_per_serving_l104_104435


namespace Luca_weight_loss_per_year_l104_104170

def Barbi_weight_loss_per_month : Real := 1.5
def months_in_a_year : Nat := 12
def Luca_years : Nat := 11
def extra_weight_Luca_lost : Real := 81

theorem Luca_weight_loss_per_year :
  (Barbi_weight_loss_per_month * months_in_a_year + extra_weight_Luca_lost) / Luca_years = 9 := by
  sorry

end Luca_weight_loss_per_year_l104_104170


namespace unit_prices_max_colored_tiles_l104_104735

-- Define the given conditions
def condition1 (x y : ℝ) := 40 * x + 60 * y = 5600
def condition2 (x y : ℝ) := 50 * x + 50 * y = 6000

-- Prove the solution for part 1
theorem unit_prices (x y : ℝ) (h1 : condition1 x y) (h2 : condition2 x y) :
  x = 80 ∧ y = 40 := 
sorry

-- Define the condition for the kitchen tiles
def condition3 (a : ℝ) := 80 * a + 40 * (60 - a) ≤ 3400

-- Prove the maximum number of colored tiles for the kitchen
theorem max_colored_tiles (a : ℝ) (h3 : condition3 a) :
  a ≤ 25 := 
sorry

end unit_prices_max_colored_tiles_l104_104735


namespace degree_to_radian_l104_104508

theorem degree_to_radian (h : 1 = (π / 180)) : 60 = π * (1 / 3) := 
sorry

end degree_to_radian_l104_104508


namespace lucas_fib_relation_l104_104320

noncomputable def α := (1 + Real.sqrt 5) / 2
noncomputable def β := (1 - Real.sqrt 5) / 2
def Fib : ℕ → ℝ
| 0       => 0
| 1       => 1
| (n + 2) => Fib n + Fib (n + 1)

def Lucas : ℕ → ℝ
| 0       => 2
| 1       => 1
| (n + 2) => Lucas n + Lucas (n + 1)

theorem lucas_fib_relation (n : ℕ) (hn : 1 ≤ n) :
  Lucas (2 * n + 1) + (-1)^(n+1) = Fib (2 * n) * Fib (2 * n + 1) := sorry

end lucas_fib_relation_l104_104320


namespace lcm_36_105_l104_104211

theorem lcm_36_105 : Int.lcm 36 105 = 1260 := by
  sorry

end lcm_36_105_l104_104211


namespace remainder_of_17_power_1801_mod_28_l104_104471

theorem remainder_of_17_power_1801_mod_28 : (17 ^ 1801) % 28 = 17 := 
by
  sorry

end remainder_of_17_power_1801_mod_28_l104_104471


namespace average_age_across_rooms_l104_104257

theorem average_age_across_rooms :
  let room_a_people := 8
  let room_a_average_age := 35
  let room_b_people := 5
  let room_b_average_age := 30
  let room_c_people := 7
  let room_c_average_age := 25
  let total_people := room_a_people + room_b_people + room_c_people
  let total_age := (room_a_people * room_a_average_age) + (room_b_people * room_b_average_age) + (room_c_people * room_c_average_age)
  let average_age := total_age / total_people
  average_age = 30.25 := by
{
  sorry
}

end average_age_across_rooms_l104_104257


namespace smallest_x_value_l104_104050

theorem smallest_x_value : 
  ∃ x : ℝ, 
  (∀ y : ℝ, (y^2 - 5 * y - 84) / (y - 9) = 4 / (y + 6) → y >= (x)) ∧ 
  (x^2 - 5 * x - 84) / (x - 9) = 4 / (x + 6) ∧ 
  x = ( - 13 - Real.sqrt 17 ) / 2 := 
sorry

end smallest_x_value_l104_104050


namespace find_b_value_l104_104543

theorem find_b_value (a b c : ℝ)
  (h1 : a * b * c = (Real.sqrt ((a + 2) * (b + 3))) / (c + 1))
  (h2 : 6 * b * 7 = 1.5) : b = 15 := by
  sorry

end find_b_value_l104_104543


namespace total_exercise_time_l104_104723

theorem total_exercise_time :
  let javier_minutes_per_day := 50
  let javier_days := 7
  let sanda_minutes_per_day := 90
  let sanda_days := 3
  (javier_minutes_per_day * javier_days + sanda_minutes_per_day * sanda_days) = 620 :=
by
  sorry

end total_exercise_time_l104_104723


namespace find_number_l104_104627

theorem find_number (x : ℝ) (h : x / 5 = 30 + x / 6) : x = 900 :=
by
  -- Proof goes here
  sorry

end find_number_l104_104627


namespace overall_gain_is_2_89_l104_104011

noncomputable def overall_gain_percentage : ℝ :=
  let cost1 := 500000
  let gain1 := 0.10
  let sell1 := cost1 * (1 + gain1)

  let cost2 := 600000
  let loss2 := 0.05
  let sell2 := cost2 * (1 - loss2)

  let cost3 := 700000
  let gain3 := 0.15
  let sell3 := cost3 * (1 + gain3)

  let cost4 := 800000
  let loss4 := 0.12
  let sell4 := cost4 * (1 - loss4)

  let cost5 := 900000
  let gain5 := 0.08
  let sell5 := cost5 * (1 + gain5)

  let total_cost := cost1 + cost2 + cost3 + cost4 + cost5
  let total_sell := sell1 + sell2 + sell3 + sell4 + sell5
  let overall_gain := total_sell - total_cost
  (overall_gain / total_cost) * 100

theorem overall_gain_is_2_89 :
  overall_gain_percentage = 2.89 :=
by
  -- Proof goes here
  sorry

end overall_gain_is_2_89_l104_104011


namespace common_points_intervals_l104_104758

noncomputable def h (x : ℝ) : ℝ := (2 * Real.log x) / x

theorem common_points_intervals (a : ℝ) (h₀ : 1 < a) : 
  (∀ f g : ℝ → ℝ, (f x = a ^ x) → (g x = x ^ 2) → 
  ∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ f x₁ = g x₁ ∧ f x₂ = g x₂ ∧ f x₃ = g x₃) → 
  a < Real.exp (2 / Real.exp 1) :=
by
  sorry

end common_points_intervals_l104_104758


namespace compare_f_g_l104_104878

def R (m n : ℕ) : ℕ := sorry
def L (m n : ℕ) : ℕ := sorry

def f (m n : ℕ) : ℕ := R m n + L m n - sorry
def g (m n : ℕ) : ℕ := R m n + L m n - sorry

theorem compare_f_g (m n : ℕ) : f m n ≤ g m n := sorry

end compare_f_g_l104_104878


namespace total_movies_seen_l104_104673

theorem total_movies_seen (d h a c : ℕ) (hd : d = 7) (hh : h = 12) (ha : a = 15) (hc : c = 2) :
  (c + (d - c) + (h - c) + (a - c)) = 30 :=
by
  sorry

end total_movies_seen_l104_104673


namespace number_of_friends_l104_104961

theorem number_of_friends (total_bottle_caps : ℕ) (bottle_caps_per_friend : ℕ) (h1 : total_bottle_caps = 18) (h2 : bottle_caps_per_friend = 3) :
  total_bottle_caps / bottle_caps_per_friend = 6 :=
by
  sorry

end number_of_friends_l104_104961


namespace break_even_machines_l104_104364

def cost_parts : ℤ := 3600
def cost_patent : ℤ := 4500
def selling_price : ℤ := 180

def total_costs : ℤ := cost_parts + cost_patent

def machines_to_break_even : ℤ := total_costs / selling_price

theorem break_even_machines :
  machines_to_break_even = 45 := by
  sorry

end break_even_machines_l104_104364


namespace solve_eq_sqrt_exp_l104_104208

theorem solve_eq_sqrt_exp :
  (∀ x : ℝ, (Real.sqrt ((3 + 2 * Real.sqrt 2) ^ x) + Real.sqrt ((3 - 2 * Real.sqrt 2) ^ x) = 6) → (x = 2 ∨ x = -1)) :=
by
  -- Prove that the solutions are x = 2 and x = -1
  sorry

end solve_eq_sqrt_exp_l104_104208


namespace rational_sum_zero_cube_nonzero_fifth_power_zero_l104_104863

theorem rational_sum_zero_cube_nonzero_fifth_power_zero
  (a b c : ℚ) 
  (h_sum : a + b + c = 0)
  (h_cube_nonzero : a^3 + b^3 + c^3 ≠ 0) 
  : a^5 + b^5 + c^5 = 0 :=
sorry

end rational_sum_zero_cube_nonzero_fifth_power_zero_l104_104863


namespace isosceles_triangle_no_obtuse_l104_104089

theorem isosceles_triangle_no_obtuse (A B C : ℝ) 
  (h1 : A = 70) 
  (h2 : B = 70) 
  (h3 : A + B + C = 180) 
  (h_iso : A = B) 
  : (A ≤ 90) ∧ (B ≤ 90) ∧ (C ≤ 90) :=
by
  sorry

end isosceles_triangle_no_obtuse_l104_104089


namespace decrease_in_profit_due_to_looms_breakdown_l104_104662

theorem decrease_in_profit_due_to_looms_breakdown :
  let num_looms := 70
  let month_days := 30
  let total_sales := 1000000
  let total_expenses := 150000
  let daily_sales_per_loom := total_sales / (num_looms * month_days)
  let daily_expenses_per_loom := total_expenses / (num_looms * month_days)
  let loom1_days := 10
  let loom2_days := 5
  let loom3_days := 15
  let loom_repair_cost := 2000
  let loom1_loss := daily_sales_per_loom * loom1_days
  let loom2_loss := daily_sales_per_loom * loom2_days
  let loom3_loss := daily_sales_per_loom * loom3_days
  let total_loss_sales := loom1_loss + loom2_loss + loom3_loss
  let total_repair_cost := loom_repair_cost * 3
  let decrease_in_profit := total_loss_sales + total_repair_cost
  decrease_in_profit = 20285.70 := by
  sorry

end decrease_in_profit_due_to_looms_breakdown_l104_104662


namespace sales_tax_difference_l104_104958

theorem sales_tax_difference (rate1 rate2 : ℝ) (price : ℝ) (h1 : rate1 = 0.075) (h2 : rate2 = 0.0625) (hprice : price = 50) : 
  rate1 * price - rate2 * price = 0.625 :=
by
  sorry

end sales_tax_difference_l104_104958


namespace min_varphi_symmetry_l104_104912

theorem min_varphi_symmetry (ϕ : ℝ) (hϕ : ϕ > 0) :
  (∃ k : ℤ, ϕ = (4 * Real.pi) / 3 - k * Real.pi ∧ ϕ > 0 ∧ (∀ x : ℝ, Real.cos (x - ϕ + (4 * Real.pi) / 3) = Real.cos (-x - ϕ + (4 * Real.pi) / 3))) 
  → ϕ = Real.pi / 3 :=
sorry

end min_varphi_symmetry_l104_104912


namespace delaney_missed_bus_time_l104_104182

def busDepartureTime : Nat := 480 -- 8:00 a.m. = 8 * 60 minutes
def travelTime : Nat := 30 -- 30 minutes
def departureFromHomeTime : Nat := 470 -- 7:50 a.m. = 7 * 60 + 50 minutes

theorem delaney_missed_bus_time :
  (departureFromHomeTime + travelTime - busDepartureTime) = 20 :=
by
  -- proof would go here
  sorry

end delaney_missed_bus_time_l104_104182


namespace derivative_exp_l104_104074

theorem derivative_exp (f : ℝ → ℝ) (h : ∀ x, f x = Real.exp x) : 
    ∀ x, deriv f x = Real.exp x :=
by 
  sorry

end derivative_exp_l104_104074


namespace range_of_a_for_decreasing_f_l104_104537

theorem range_of_a_for_decreasing_f :
  (∀ x : ℝ, (-3) * x^2 + 2 * a * x - 1 ≤ 0) ↔ (-Real.sqrt 3 ≤ a ∧ a ≤ Real.sqrt 3) :=
by
  -- The proof goes here
  sorry

end range_of_a_for_decreasing_f_l104_104537


namespace log_base_half_cuts_all_horizontal_lines_l104_104507

theorem log_base_half_cuts_all_horizontal_lines (x : ℝ) (k : ℝ) (h_pos : x > 0) (h_eq : y = Real.logb 0.5 x) : ∃ x, ∀ k, k = Real.logb 0.5 x ↔ x > 0 := 
sorry

end log_base_half_cuts_all_horizontal_lines_l104_104507


namespace seq_an_general_term_and_sum_l104_104100

theorem seq_an_general_term_and_sum
  (a_n : ℕ → ℕ)
  (S : ℕ → ℕ)
  (T : ℕ → ℕ)
  (H1 : ∀ n, S n = 2 * a_n n - a_n 1)
  (H2 : ∃ d : ℕ, a_n 1 = d ∧ a_n 2 + 1 = a_n 1 + d ∧ a_n 3 = a_n 2 + d) :
  (∀ n, a_n n = 2^n) ∧ (∀ n, T n = n * 2^(n + 1) + 2 - 2^(n + 1)) := 
  by
  sorry

end seq_an_general_term_and_sum_l104_104100


namespace quadratic_equation_transformation_l104_104289

theorem quadratic_equation_transformation (x : ℝ) :
  (-5 * x ^ 2 = 2 * x + 10) →
  (x ^ 2 + (2 / 5) * x + 2 = 0) :=
by
  intro h
  sorry

end quadratic_equation_transformation_l104_104289


namespace find_unit_vector_l104_104561

noncomputable def unit_vector : ℝ × ℝ × ℝ := 
  (5 / Real.sqrt 14, 0.5 / Real.sqrt 14, 2.5 / Real.sqrt 14)

def a : ℝ × ℝ × ℝ := (2, -3, 1)
def b : ℝ × ℝ × ℝ := (4, -2, 2)

theorem find_unit_vector (v : ℝ × ℝ × ℝ) : 
  (a, b, v) = ((2, -3, 1), (4, -2, 2), (5 / Real.sqrt 14, 0.5 / Real.sqrt 14, 2.5 / Real.sqrt 14)) → 
  v = unit_vector := 
sorry

end find_unit_vector_l104_104561


namespace part1_part2_l104_104670

noncomputable def f (x a : ℝ) : ℝ := |x - 1| + |2 * x + a|

theorem part1 (x : ℝ) : f x 1 + |x - 1| ≥ 3 :=
  sorry

theorem part2 (a : ℝ) (h : ∃ x : ℝ, f x a = 2) : a = 2 ∨ a = -6 :=
  sorry

end part1_part2_l104_104670


namespace total_distance_traveled_l104_104151

noncomputable def row_speed_still_water : ℝ := 8
noncomputable def river_speed : ℝ := 2

theorem total_distance_traveled (h : (3.75 / (row_speed_still_water - river_speed)) + (3.75 / (row_speed_still_water + river_speed)) = 1) : 
  2 * 3.75 = 7.5 :=
by
  sorry

end total_distance_traveled_l104_104151


namespace avg_of_six_is_3_9_l104_104128

noncomputable def avg_of_six_numbers 
  (avg1 avg2 avg3 : ℝ) 
  (h1 : avg1 = 3.4) 
  (h2 : avg2 = 3.85) 
  (h3 : avg3 = 4.45) : ℝ :=
  (2 * avg1 + 2 * avg2 + 2 * avg3) / 6

theorem avg_of_six_is_3_9 
  (avg1 avg2 avg3 : ℝ) 
  (h1 : avg1 = 3.4) 
  (h2 : avg2 = 3.85) 
  (h3 : avg3 = 4.45) : 
  avg_of_six_numbers avg1 avg2 avg3 h1 h2 h3 = 3.9 := 
by {
  sorry
}

end avg_of_six_is_3_9_l104_104128


namespace age_difference_l104_104604

variable (A B C D : ℕ)

theorem age_difference (h1 : A + B > B + C) (h2 : C = A - 16) : (A + B) - (B + C) = 16 :=
by
  sorry

end age_difference_l104_104604


namespace lcm_36_105_l104_104215

theorem lcm_36_105 : Nat.lcm 36 105 = 1260 := by
  sorry

end lcm_36_105_l104_104215


namespace volume_formula_l104_104660

noncomputable def volume_of_parallelepiped
  (a b : ℝ) (h : ℝ) (θ : ℝ) 
  (θ_eq : θ = Real.pi / 3) 
  (base_diagonal : ℝ)
  (base_diagonal_eq : base_diagonal = Real.sqrt (a ^ 2 + b ^ 2)) : ℝ :=
  a * b * h 

theorem volume_formula 
  (a b : ℝ) (h : ℝ) (θ : ℝ)
  (area_base : ℝ) 
  (area_of_base_eq : area_base = a * b) 
  (θ_eq : θ = Real.pi / 3) 
  (base_diagonal : ℝ) 
  (base_diagonal_eq : base_diagonal = Real.sqrt (a ^ 2 + b ^ 2))
  (height_eq : h = (base_diagonal / 2) * (Real.sqrt 3)): 
  volume_of_parallelepiped a b h θ θ_eq base_diagonal base_diagonal_eq 
  = (144 * Real.sqrt 3) / 5 :=
by {
  sorry
}

end volume_formula_l104_104660


namespace line_perpendicular_exists_k_line_intersects_circle_l104_104848

theorem line_perpendicular_exists_k (k : ℝ) :
  ∃ k, (k * (1 / 2)) = -1 :=
sorry

theorem line_intersects_circle (k : ℝ) :
  ∃ x y : ℝ, (k * x - y + 2 * k = 0) ∧ (x^2 + y^2 = 8) :=
sorry

end line_perpendicular_exists_k_line_intersects_circle_l104_104848


namespace solve_fractional_eq_l104_104119

theorem solve_fractional_eq {x : ℚ} : (3 / (x - 1)) = (1 / x) ↔ x = -1/2 :=
by sorry

end solve_fractional_eq_l104_104119


namespace largest_distance_l104_104322

noncomputable def max_distance_between_spheres 
  (c1 : ℝ × ℝ × ℝ) (r1 : ℝ) 
  (c2 : ℝ × ℝ × ℝ) (r2 : ℝ) : ℝ :=
dist c1 c2 + r1 + r2

theorem largest_distance 
  (c1 : ℝ × ℝ × ℝ) (r1 : ℝ) 
  (c2 : ℝ × ℝ × ℝ) (r2 : ℝ) 
  (h₁ : c1 = (-3, -15, 10))
  (h₂ : r1 = 24)
  (h₃ : c2 = (20, 18, -30))
  (h₄ : r2 = 95) : 
  max_distance_between_spheres c1 r1 c2 r2 = Real.sqrt 3218 + 119 := 
by
  sorry

end largest_distance_l104_104322


namespace GreatWhiteSharkTeeth_l104_104018

-- Definition of the number of teeth for a tiger shark
def tiger_shark_teeth : ℕ := 180

-- Definition of the number of teeth for a hammerhead shark
def hammerhead_shark_teeth : ℕ := tiger_shark_teeth / 6

-- Definition of the number of teeth for a great white shark
def great_white_shark_teeth : ℕ := 2 * (tiger_shark_teeth + hammerhead_shark_teeth)

-- Statement to prove
theorem GreatWhiteSharkTeeth : great_white_shark_teeth = 420 :=
by
  -- Proof omitted
  sorry

end GreatWhiteSharkTeeth_l104_104018


namespace sn_geq_mnplus1_l104_104500

namespace Polysticks

def n_stick (n : ℕ) : Type := sorry -- formalize the definition of n-stick
def n_mino (n : ℕ) : Type := sorry -- formalize the definition of n-mino

def S (n : ℕ) : ℕ := sorry -- define the number of n-sticks
def M (n : ℕ) : ℕ := sorry -- define the number of n-minos

theorem sn_geq_mnplus1 (n : ℕ) : S n ≥ M (n+1) := sorry

end Polysticks

end sn_geq_mnplus1_l104_104500


namespace tobias_swimming_distance_l104_104925

def swimming_time_per_100_meters : ℕ := 5
def pause_time : ℕ := 5
def swimming_period : ℕ := 25
def total_visit_hours : ℕ := 3

theorem tobias_swimming_distance :
  let total_visit_minutes := total_visit_hours * 60
  let sequence_time := swimming_period + pause_time
  let number_of_sequences := total_visit_minutes / sequence_time
  let total_pause_time := number_of_sequences * pause_time
  let total_swimming_time := total_visit_minutes - total_pause_time
  let number_of_100m_lengths := total_swimming_time / swimming_time_per_100_meters
  let total_distance := number_of_100m_lengths * 100
  total_distance = 3000 :=
by
  sorry

end tobias_swimming_distance_l104_104925


namespace range_of_a_l104_104405

theorem range_of_a (a : ℝ) : (∀ x : ℝ, abs (x + 2) - abs (x - 1) ≥ a^3 - 4 * a^2 - 3) → a ≤ 4 :=
sorry

end range_of_a_l104_104405


namespace compound_interest_two_years_l104_104975

/-- Given the initial amount, and year-wise interest rates, 
     we want to find the amount in 2 years and prove it equals to a specific value. -/
theorem compound_interest_two_years 
  (P : ℝ) (R1 : ℝ) (R2 : ℝ) (T1 : ℝ) (T2 : ℝ) 
  (initial_amount : P = 7644) 
  (interest_rate_first_year : R1 = 0.04) 
  (interest_rate_second_year : R2 = 0.05) 
  (time_first_year : T1 = 1) 
  (time_second_year : T2 = 1) : 
  (P + (P * R1 * T1) + ((P + (P * R1 * T1)) * R2 * T2) = 8347.248) := 
by 
  sorry

end compound_interest_two_years_l104_104975


namespace absolute_value_simplification_l104_104985

theorem absolute_value_simplification (a b : ℝ) (ha : a < 0) (hb : b > 0) : |a - b| + |b - a| = -2 * a + 2 * b := 
by 
  sorry

end absolute_value_simplification_l104_104985


namespace missed_bus_time_by_l104_104186

def bus_departure_time : Time := Time.mk 8 0 0
def travel_time_minutes : Int := 30
def departure_time_home : Time := Time.mk 7 50 0
def arrival_time_pickup_point : Time := 
  departure_time_home.addMinutes travel_time_minutes

theorem missed_bus_time_by :
  arrival_time_pickup_point.diff bus_departure_time = 20 * 60 :=
by
  sorry

end missed_bus_time_by_l104_104186


namespace degrees_for_basic_astrophysics_correct_l104_104775

-- Definitions for conditions
def percentage_allocations : List ℚ := [13, 24, 15, 29, 8]
def total_percentage : ℚ := percentage_allocations.sum
def remaining_percentage : ℚ := 100 - total_percentage

-- The question to answer
def total_degrees : ℚ := 360
def degrees_for_basic_astrophysics : ℚ := remaining_percentage / 100 * total_degrees

-- Prove that the degrees for basic astrophysics is 39.6
theorem degrees_for_basic_astrophysics_correct :
  degrees_for_basic_astrophysics = 39.6 :=
by
  sorry

end degrees_for_basic_astrophysics_correct_l104_104775


namespace Delaney_missed_bus_by_l104_104180

def time_in_minutes (hours : ℕ) (minutes : ℕ) : ℕ := hours * 60 + minutes

def Delaney_start_time : ℕ := time_in_minutes 7 50
def bus_departure_time : ℕ := time_in_minutes 8 0
def travel_duration : ℕ := 30

theorem Delaney_missed_bus_by :
  Delaney_start_time + travel_duration - bus_departure_time = 20 :=
by
  sorry

end Delaney_missed_bus_by_l104_104180


namespace most_compliant_expression_l104_104473

-- Define the expressions as algebraic terms.
def OptionA : String := "1(1/2)a"
def OptionB : String := "b/a"
def OptionC : String := "3a-1 个"
def OptionD : String := "a * 3"

-- Define a property that represents compliance with standard algebraic notation.
def is_compliant (expr : String) : Prop :=
  expr = OptionB

-- The theorem to prove.
theorem most_compliant_expression :
  is_compliant OptionB :=
by
  sorry

end most_compliant_expression_l104_104473


namespace smallest_leading_coefficient_l104_104614

theorem smallest_leading_coefficient :
  ∀ (P : ℤ → ℤ), (∃ (a b c : ℚ), ∀ (x : ℤ), P x = a * (x^2 : ℚ) + b * (x : ℚ) + c) →
  (∀ x : ℤ, ∃ k : ℤ, P x = k) →
  (∃ a : ℚ, (∀ x : ℤ, ∃ k : ℤ, a * (x^2 : ℚ) + b * (x : ℚ) + c = k) ∧ a > 0 ∧ (∀ a' : ℚ, (∀ x : ℤ, ∃ k : ℤ, a' * (x^2 : ℚ) + b * (x : ℚ) + c = k) → a' ≥ a) ∧ a = 1 / 2) := 
sorry

end smallest_leading_coefficient_l104_104614


namespace probability_heads_even_60_tosses_l104_104001

noncomputable def P_n (n : ℕ) : ℝ :=
  if n = 0 then 1 else 0.6 * P_n (n - 1) + 0.4 * (1 - P_n (n - 1))

theorem probability_heads_even_60_tosses :
  P_n 60 = 1 / 2 * (1 + 1 / (5 : ℝ)^60) :=
by sorry

end probability_heads_even_60_tosses_l104_104001


namespace part_one_part_two_l104_104396

variable {x : ℝ}

def setA (a : ℝ) : Set ℝ := {x | 0 < a * x + 1 ∧ a * x + 1 ≤ 5}
def setB : Set ℝ := {x | -1 / 2 < x ∧ x ≤ 2}

theorem part_one (a : ℝ) (h : a = 1) : setB ⊆ setA a :=
by
  sorry

theorem part_two (a : ℝ) : (setA a ⊆ setB) ↔ (a < -8 ∨ a ≥ 2) :=
by
  sorry

end part_one_part_two_l104_104396


namespace sugar_cone_count_l104_104667

theorem sugar_cone_count (ratio_sugar_waffle : ℕ → ℕ → Prop) (sugar_waffle_ratio : ratio_sugar_waffle 5 4) 
(w : ℕ) (h_w : w = 36) : ∃ s : ℕ, ratio_sugar_waffle s w ∧ s = 45 :=
by
  sorry

end sugar_cone_count_l104_104667


namespace max_students_can_participate_l104_104824

theorem max_students_can_participate (max_funds rent cost_per_student : ℕ) (h_max_funds : max_funds = 800) (h_rent : rent = 300) (h_cost_per_student : cost_per_student = 15) :
  ∃ x : ℕ, x ≤ (max_funds - rent) / cost_per_student ∧ x = 33 :=
by
  sorry

end max_students_can_participate_l104_104824


namespace minimum_value_x_plus_y_l104_104839

theorem minimum_value_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x * y * (x - y)^2 = 1) : x + y ≥ 2 :=
sorry

end minimum_value_x_plus_y_l104_104839


namespace sequence_unbounded_l104_104502

theorem sequence_unbounded 
  (a : ℕ → ℝ)
  (h1 : ∀ n, a n = |a (n + 1) - a (n + 2)|)
  (h2 : 0 < a 0)
  (h3 : 0 < a 1)
  (h4 : a 0 ≠ a 1) :
  ¬ ∃ M : ℝ, ∀ n, |a n| ≤ M := 
sorry

end sequence_unbounded_l104_104502


namespace solve_system_l104_104101

theorem solve_system :
  ∃ (x y : ℤ), (x * (1/7 : ℚ)^2 = 7^3) ∧ (x + y = 7^2) ∧ (x = 16807) ∧ (y = -16758) :=
by
  sorry

end solve_system_l104_104101


namespace line_passes_through_fixed_point_range_of_k_no_second_quadrant_min_area_triangle_l104_104847

-- Problem 1: The line passes through a fixed point
theorem line_passes_through_fixed_point (k : ℝ) : ∃ P : ℝ × ℝ, P = (1, -2) ∧ (∀ x y, k * x - y - 2 - k = 0 → P = (x, y)) :=
by
  sorry

-- Problem 2: Range of values for k if the line does not pass through the second quadrant
theorem range_of_k_no_second_quadrant (k : ℝ) : ¬ (∃ x y : ℝ, x < 0 ∧ y > 0 ∧ k * x - y - 2 - k = 0) → k ∈ Set.Ici (0) :=
by
  sorry

-- Problem 3: Minimum area of triangle AOB
theorem min_area_triangle (k : ℝ) :
  let A := (2 + k) / k
  let B := -2 - k
  (∀ x y, k * x - y - 2 - k = 0 ↔ (x = A ∧ y = 0) ∨ (x = 0 ∧ y = B)) →
  ∃ S : ℝ, S = 4 ∧ (∀ x y : ℝ, (k = 2 ∧ k * x - y - 4 = 0) → S = 4) :=
by
  sorry

end line_passes_through_fixed_point_range_of_k_no_second_quadrant_min_area_triangle_l104_104847


namespace number_of_subsets_B_l104_104070

def A : Set ℕ := {1, 3}
def C : Set ℕ := {1, 3, 5}

theorem number_of_subsets_B : ∃ (n : ℕ), (∀ B : Set ℕ, A ∪ B = C → n = 4) :=
sorry

end number_of_subsets_B_l104_104070


namespace range_of_a_l104_104700

theorem range_of_a (f : ℝ → ℝ)
  (h_odd : ∀ x, f (-x) = -f x)
  (h_mono : ∀ ⦃a b⦄, 0 ≤ a → a ≤ b → f a ≤ f b)
  (h_cond : ∀ a, f a < f (2 * a - 1) → a > 1) :
  ∀ a, f a < f (2 * a - 1) → 1 < a := 
sorry

end range_of_a_l104_104700


namespace number_is_40_l104_104949

theorem number_is_40 (N : ℝ) (h : N = (3/8) * N + (1/4) * N + 15) : N = 40 :=
by
  sorry

end number_is_40_l104_104949


namespace boys_in_class_l104_104127

noncomputable def number_of_boys_in_class : ℕ :=
  let avg_height : ℕ := 185
  let wrong_height : ℕ := 166
  let actual_wrong_height : ℕ := 106
  let actual_avg_height : ℕ := 183
  let difference : ℕ := wrong_height - actual_wrong_height
  -- Derived from the given equation: 185 * n - difference = 183 * n
  let equation := (avg_height * n - difference = actual_avg_height * n)
  -- From equation we have: (185 - 183) * n = difference
  -- Which leads to: 2 * n = 60
  let result : ℕ := 30

theorem boys_in_class : number_of_boys_in_class = 30 := 
by
  sorry

end boys_in_class_l104_104127


namespace ferris_wheel_capacity_l104_104440

-- Define the conditions
def number_of_seats : ℕ := 14
def people_per_seat : ℕ := 6

-- Theorem to prove the total capacity is 84
theorem ferris_wheel_capacity : number_of_seats * people_per_seat = 84 := sorry

end ferris_wheel_capacity_l104_104440


namespace exist_indices_for_sequences_l104_104707

open Nat

theorem exist_indices_for_sequences 
  (a b c : ℕ → ℕ) : 
  ∃ p q, p ≠ q ∧ p > q ∧ a p ≥ a q ∧ b p ≥ b q ∧ c p ≥ c q := by
  sorry

end exist_indices_for_sequences_l104_104707


namespace quadratic_function_min_value_l104_104503

theorem quadratic_function_min_value :
  ∃ x, ∀ y, 5 * x^2 - 15 * x + 2 ≤ 5 * y^2 - 15 * y + 2 ∧ (5 * x^2 - 15 * x + 2 = -9.25) :=
by
  sorry

end quadratic_function_min_value_l104_104503


namespace annie_weeks_off_sick_l104_104666

-- Define the conditions and the question
def weekly_hours_chess : ℕ := 2
def weekly_hours_drama : ℕ := 8
def weekly_hours_glee : ℕ := 3
def semester_weeks : ℕ := 12
def total_hours_before_midterms : ℕ := 52

-- Define the proof problem
theorem annie_weeks_off_sick :
  let total_weekly_hours := weekly_hours_chess + weekly_hours_drama + weekly_hours_glee
  let attended_weeks := total_hours_before_midterms / total_weekly_hours
  semester_weeks - attended_weeks = 8 :=
by
  -- Automatically prove by computation of above assumptions.
  sorry

end annie_weeks_off_sick_l104_104666


namespace extinction_prob_l104_104796

-- Define the probabilities
def p : ℝ := 0.6
def q : ℝ := 0.4

-- Define the base case condition
def v_0 : ℝ := 1

-- Define the extinction probability function
def v : ℕ → ℝ
| 0 => v_0
| k => p * v (k + 1) + q * v (k - 1)

-- Define the target value for v_1
def v_1 : ℝ := 2 / 3

-- The main theorem: Prove that v 1 = 2 / 3 given the conditions
theorem extinction_prob : v 1 = v_1 := by
  -- Proof will be provided here
  sorry

end extinction_prob_l104_104796


namespace find_number_l104_104624

theorem find_number (x : ℝ) (h : x / 5 = 30 + x / 6) : x = 900 :=
by
  -- Proof goes here
  sorry

end find_number_l104_104624


namespace negation_statement_contrapositive_statement_l104_104542

variable (x y : ℝ)

theorem negation_statement :
  (¬ ((x-1) * (y+2) ≠ 0 → x ≠ 1 ∧ y ≠ -2)) ↔ ((x-1) * (y+2) = 0 → x = 1 ∨ y = -2) :=
by sorry

theorem contrapositive_statement :
  (x = 1 ∨ y = -2) → ((x-1) * (y+2) = 0) :=
by sorry

end negation_statement_contrapositive_statement_l104_104542


namespace set_intersection_l104_104268

theorem set_intersection (A B : Set ℝ) 
  (hA : A = { x : ℝ | 0 < x ∧ x < 5 }) 
  (hB : B = { x : ℝ | -1 ≤ x ∧ x < 4 }) : 
  (A ∩ B) = { x : ℝ | 0 < x ∧ x < 4 } :=
by
  sorry

end set_intersection_l104_104268


namespace dice_probability_abs_diff_2_l104_104328

theorem dice_probability_abs_diff_2 :
  let total_outcomes := 36
  let favorable_outcomes := 8
  let probability := favorable_outcomes / total_outcomes
  probability = 2 / 9 :=
by
  sorry

end dice_probability_abs_diff_2_l104_104328


namespace reciprocal_of_neg2_l104_104452

-- Define the number
def num : ℤ := -2

-- Define the reciprocal function
def reciprocal (x : ℤ) : ℚ := 1 / x

-- State the theorem
theorem reciprocal_of_neg2 : reciprocal num = -1 / 2 :=
by
  -- Proof is skipped
  sorry

end reciprocal_of_neg2_l104_104452


namespace incorrect_statements_l104_104989

-- Definitions based on conditions from the problem.

def quadratic_inequality (a b c : ℝ) (x : ℝ) : Prop := a * x^2 + b * x + c < 0
def solution_set (a b c : ℝ) : Set ℝ := {x | quadratic_inequality a b c x}
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Lean statements of the conditions and the final proof problem.
theorem incorrect_statements (a b c : ℝ) (M : Set ℝ) :
  (M = ∅ → (a < 0 ∧ discriminant a b c < 0) → false) ∧
  (M = {x | x ≠ x0} → a < b → (a + 4 * c) / (b - a) = 2 + 2 * Real.sqrt 2 → false) := sorry

end incorrect_statements_l104_104989


namespace gcd_squares_example_l104_104931

noncomputable def gcd_of_squares : ℕ :=
  Nat.gcd (101 ^ 2 + 202 ^ 2 + 303 ^ 2) (100 ^ 2 + 201 ^ 2 + 304 ^ 2)

theorem gcd_squares_example : gcd_of_squares = 3 :=
by
  sorry

end gcd_squares_example_l104_104931


namespace nancy_coffee_expense_l104_104283

-- Definitions corresponding to the conditions
def cost_double_espresso : ℝ := 3.00
def cost_iced_coffee : ℝ := 2.50
def days : ℕ := 20

-- The statement of the problem
theorem nancy_coffee_expense :
  (days * (cost_double_espresso + cost_iced_coffee)) = 110.00 := by
  sorry

end nancy_coffee_expense_l104_104283


namespace volleyball_team_starters_l104_104748

theorem volleyball_team_starters :
  nat.choose 16 7 = 11440 :=
by
  sorry

end volleyball_team_starters_l104_104748


namespace unique_sum_of_cubes_lt_1000_l104_104250

theorem unique_sum_of_cubes_lt_1000 : 
  let max_cube := 11 
  let max_val := 1000 
  ∃ n : ℕ, n = 35 ∧ ∀ (a b : ℕ), 1 ≤ a ∧ a ≤ max_cube → 1 ≤ b ∧ b ≤ max_cube → a^3 + b^3 < max_val :=
sorry

end unique_sum_of_cubes_lt_1000_l104_104250


namespace distinct_sets_count_l104_104379

noncomputable def num_distinct_sets : ℕ :=
  let product : ℕ := 11 * 21 * 31 * 41 * 51 * 61
  728

theorem distinct_sets_count : 
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a * b * c = 11 * 21 * 31 * 41 * 51 * 61 ∧ num_distinct_sets = 728 :=
sorry

end distinct_sets_count_l104_104379


namespace strawberries_count_l104_104854

theorem strawberries_count (harvest_per_day : ℕ) (days_in_april : ℕ) (given_away : ℕ) (stolen : ℕ) :
  (harvest_per_day = 5) →
  (days_in_april = 30) →
  (given_away = 20) →
  (stolen = 30) →
  (harvest_per_day * days_in_april - given_away - stolen = 100) :=
by {
  intros h1 h2 h3 h4,
  rw [h1, h2, h3, h4],
  norm_num,
  sorry,
}

end strawberries_count_l104_104854


namespace simplify_proof_l104_104754

def simplify_fractions_product : ℚ :=
  (27 / 25) * (20 / 33) * (55 / 54)

theorem simplify_proof :
  simplify_fractions_product = 25 / 3 :=
by
  sorry

end simplify_proof_l104_104754


namespace dorothy_money_left_l104_104514

-- Define the conditions
def annual_income : ℝ := 60000
def tax_rate : ℝ := 0.18

-- Define the calculation of the amount of money left after paying taxes
def money_left (income : ℝ) (rate : ℝ) : ℝ :=
  income - (rate * income)

-- State the main theorem to prove
theorem dorothy_money_left :
  money_left annual_income tax_rate = 49200 := 
by
  sorry

end dorothy_money_left_l104_104514


namespace bus_probability_l104_104482

/-- A bus arrives randomly between 3:00 and 4:00, waits for 15 minutes, and then leaves. 
Sarah also arrives randomly between 3:00 and 4:00. Prove the probability that the bus 
will be there when Sarah arrives is 4275/7200. -/
theorem bus_probability : (4275 : ℚ) / 7200 = (4275 / 7200) :=
by 
  sorry

end bus_probability_l104_104482


namespace find_t_l104_104842

variable (g V V0 c S t : ℝ)
variable (h1 : V = g * t + V0 + c)
variable (h2 : S = (1/2) * g * t^2 + V0 * t + c * t^2)

theorem find_t
  (h1 : V = g * t + V0 + c)
  (h2 : S = (1/2) * g * t^2 + V0 * t + c * t^2) :
  t = 2 * S / (V + V0 - c) :=
sorry

end find_t_l104_104842


namespace power_function_pass_through_point_l104_104394

theorem power_function_pass_through_point (a : ℝ) (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = x ^ a) (h_point : f 2 = 16) : a = 4 :=
sorry

end power_function_pass_through_point_l104_104394


namespace find_x_in_equation_l104_104936

theorem find_x_in_equation :
  ∃ x : ℝ, x / 18 * (x / 162) = 1 ∧ x = 54 :=
by
  sorry

end find_x_in_equation_l104_104936


namespace smallest_n_l104_104368

theorem smallest_n (o y v : ℕ) (h1 : 18 * o = 21 * y) (h2 : 21 * y = 10 * v) (h3 : 10 * v = 30 * n) : 
  n = 21 := by
  sorry

end smallest_n_l104_104368


namespace lcm_36_105_l104_104222

noncomputable def factorize_36 : List (ℕ × ℕ) := [(2, 2), (3, 2)]
noncomputable def factorize_105 : List (ℕ × ℕ) := [(3, 1), (5, 1), (7, 1)]

theorem lcm_36_105 : Nat.lcm 36 105 = 1260 :=
by
  have h_36 : 36 = 2^2 * 3^2 := by norm_num
  have h_105 : 105 = 3^1 * 5^1 * 7^1 := by norm_num
  sorry

end lcm_36_105_l104_104222


namespace find_first_prime_l104_104304

theorem find_first_prime (p1 p2 z : ℕ) 
  (prime_p1 : Nat.Prime p1)
  (prime_p2 : Nat.Prime p2)
  (z_eq : z = p1 * p2)
  (z_val : z = 33)
  (p2_range : 8 < p2 ∧ p2 < 24)
  : p1 = 3 := 
sorry

end find_first_prime_l104_104304


namespace find_number_l104_104626

theorem find_number (x : ℝ) (h : x / 5 = 30 + x / 6) : x = 900 :=
by
  -- Proof goes here
  sorry

end find_number_l104_104626


namespace sum_of_scores_l104_104571

/-- Prove that given the conditions on Bill, John, and Sue's scores, the total sum of the scores of the three students is 160. -/
theorem sum_of_scores (B J S : ℕ) (h1 : B = J + 20) (h2 : B = S / 2) (h3 : B = 45) : B + J + S = 160 :=
sorry

end sum_of_scores_l104_104571


namespace division_reciprocal_multiplication_l104_104828

theorem division_reciprocal_multiplication : (4 / (8 / 13 : ℚ)) = (13 / 2 : ℚ) := 
by
  sorry

end division_reciprocal_multiplication_l104_104828


namespace no_cubic_term_l104_104403

noncomputable def p1 (a b k : ℝ) : ℝ := -2 * a * b + (1 / 3) * k * a^2 * b + 5 * b^2
noncomputable def p2 (a b : ℝ) : ℝ := b^2 + 3 * a^2 * b - 5 * a * b + 1
noncomputable def diff (a b k : ℝ) : ℝ := p1 a b k - p2 a b
noncomputable def cubic_term_coeff (a b k : ℝ) : ℝ := (1 / 3) * k - 3

theorem no_cubic_term (a b : ℝ) : ∀ k, (cubic_term_coeff a b k = 0) → k = 9 :=
by
  intro k h
  sorry

end no_cubic_term_l104_104403


namespace find_int_solutions_l104_104517

theorem find_int_solutions (x y : ℤ) (h : x^4 - 2 * y^2 = 1) : (x = 1 ∧ y = 0) ∨ (x = -1 ∧ y = 0) :=
sorry

end find_int_solutions_l104_104517


namespace proof_P_l104_104082

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define the complement of P in U
def CU_P : Set ℕ := {4, 5}

-- Define the set P as the difference between U and CU_P
def P : Set ℕ := U \ CU_P

-- Prove that P = {1, 2, 3}
theorem proof_P :
  P = {1, 2, 3} :=
by
  sorry

end proof_P_l104_104082


namespace simplify_expression_l104_104579

theorem simplify_expression (x : ℝ) (h1 : x ≠ 0) (h2 : 1 - x ≠ 0) :
  (1 - x) / x / ((1 - x) / x^2) = x := 
by 
  sorry

end simplify_expression_l104_104579


namespace printingTime_l104_104014

def printerSpeed : ℝ := 23
def pauseTime : ℝ := 2
def totalPages : ℝ := 350

theorem printingTime : (totalPages / printerSpeed) + ((totalPages / 50 - 1) * pauseTime) = 27 := by 
  sorry

end printingTime_l104_104014


namespace num_pos_integers_congruent_to_4_mod_7_l104_104857

theorem num_pos_integers_congruent_to_4_mod_7 (n : ℕ) (h1 : n < 500) (h2 : ∃ k : ℕ, n = 7 * k + 4) : 
  ∃ total : ℕ, total = 71 :=
sorry

end num_pos_integers_congruent_to_4_mod_7_l104_104857


namespace arithmetic_sequence_n_equals_8_l104_104068

theorem arithmetic_sequence_n_equals_8 :
  (∀ (a b c : ℕ), a + (1 / 4) * c = 2 * (1 / 2) * b) → ∃ n : ℕ, n = 8 :=
by 
  sorry

end arithmetic_sequence_n_equals_8_l104_104068


namespace union_of_sets_eq_A_l104_104734

noncomputable def A : Set ℝ := {x | x / ((x + 1) * (x - 4)) < 0}
noncomputable def B : Set ℝ := {x | Real.log x < 1}

theorem union_of_sets_eq_A: A ∪ B = A := by
  sorry

end union_of_sets_eq_A_l104_104734


namespace pizza_problem_l104_104573

theorem pizza_problem
  (pizza_slices : ℕ)
  (total_pizzas : ℕ)
  (total_people : ℕ)
  (pepperoni_only_friend : ℕ)
  (remaining_pepperoni : ℕ)
  (equal_distribution : Prop)
  (h_cond1 : pizza_slices = 16)
  (h_cond2 : total_pizzas = 2)
  (h_cond3 : total_people = 4)
  (h_cond4 : pepperoni_only_friend = 1)
  (h_cond5 : remaining_pepperoni = 1)
  (h_cond6 : equal_distribution ∧ (pepperoni_only_friend ≤ total_people)) :
  ∃ cheese_slices_left : ℕ, cheese_slices_left = 7 := by
  sorry

end pizza_problem_l104_104573


namespace roots_quadratic_expression_l104_104389

theorem roots_quadratic_expression (α β : ℝ) (hα : α^2 - 3 * α - 2 = 0) (hβ : β^2 - 3 * β - 2 = 0) :
    7 * α^4 + 10 * β^3 = 544 := 
sorry

end roots_quadratic_expression_l104_104389


namespace first_tv_cost_is_672_l104_104425

-- width and height of the first TV
def width_first_tv : ℕ := 24
def height_first_tv : ℕ := 16
-- width and height of the new TV
def width_new_tv : ℕ := 48
def height_new_tv : ℕ := 32
-- cost of the new TV
def cost_new_tv : ℕ := 1152
-- extra cost per square inch for the first TV
def extra_cost_per_square_inch : ℕ := 1

noncomputable def cost_first_tv : ℕ :=
  let area_first_tv := width_first_tv * height_first_tv
  let area_new_tv := width_new_tv * height_new_tv
  let cost_per_square_inch_new_tv := cost_new_tv / area_new_tv
  let cost_per_square_inch_first_tv := cost_per_square_inch_new_tv + extra_cost_per_square_inch
  cost_per_square_inch_first_tv * area_first_tv

theorem first_tv_cost_is_672 : cost_first_tv = 672 := by
  sorry

end first_tv_cost_is_672_l104_104425


namespace plastering_cost_l104_104952

variable (l w d : ℝ) (c : ℝ)

theorem plastering_cost :
  l = 60 → w = 25 → d = 10 → c = 0.90 →
    let A_bottom := l * w;
    let A_long_walls := 2 * (l * d);
    let A_short_walls := 2 * (w * d);
    let A_total := A_bottom + A_long_walls + A_short_walls;
    let C_total := A_total * c;
    C_total = 2880 :=
by sorry

end plastering_cost_l104_104952


namespace opposite_of_neg_6_l104_104914

theorem opposite_of_neg_6 : ∀ (n : ℤ), n = -6 → -n = 6 :=
by
  intro n h
  rw [h]
  sorry

end opposite_of_neg_6_l104_104914


namespace P_plus_Q_eq_14_l104_104552

variable (P Q : Nat)

-- Conditions:
axiom single_digit_P : P < 10
axiom single_digit_Q : Q < 10
axiom three_P_ends_7 : 3 * P % 10 = 7
axiom two_Q_ends_0 : 2 * Q % 10 = 0

theorem P_plus_Q_eq_14 : P + Q = 14 :=
by
  sorry

end P_plus_Q_eq_14_l104_104552


namespace pam_bags_count_l104_104427

noncomputable def geralds_bag_apples : ℕ := 40

noncomputable def pams_bag_apples := 3 * geralds_bag_apples

noncomputable def pams_total_apples : ℕ := 1200

theorem pam_bags_count : pams_total_apples / pams_bag_apples = 10 := by 
  sorry

end pam_bags_count_l104_104427


namespace train_crosses_bridge_in_30_seconds_l104_104497

/--
A train 155 metres long, travelling at 45 km/hr, can cross a bridge with length 220 metres in 30 seconds.
-/
theorem train_crosses_bridge_in_30_seconds
  (length_train : ℕ)
  (length_bridge : ℕ)
  (speed_km_per_hr : ℕ)
  (total_distance : ℕ)
  (speed_m_per_s : ℚ)
  (time_seconds : ℚ) 
  (h1 : length_train = 155)
  (h2 : length_bridge = 220)
  (h3 : speed_km_per_hr = 45)
  (h4 : total_distance = length_train + length_bridge)
  (h5 : speed_m_per_s = (speed_km_per_hr * 1000) / 3600)
  (h6 : time_seconds = total_distance / speed_m_per_s) :
  time_seconds = 30 :=
sorry

end train_crosses_bridge_in_30_seconds_l104_104497


namespace max_three_digit_numbers_divisible_by_4_in_sequence_l104_104159

theorem max_three_digit_numbers_divisible_by_4_in_sequence (n : ℕ) (a : ℕ → ℕ)
  (h_n : n ≥ 3)
  (h_increasing : ∀ i j, i < j → a i < a j)
  (h_recurrence : ∀ k, k ≤ n - 2 → a (k + 2) = 3 * a (k + 1) - 2 * a k - 2)
  (h_contains_2022 : ∃ k, a k = 2022) :
  ∀ k, a k = 2 * k → 
  (λ count_4 : ℕ, 
    (∀ m, 25 ≤ m ∧ m ≤ 249 → a (2 * m) = 4 * m) → 
    count_4 = 225) :=
begin
  sorry
end

end max_three_digit_numbers_divisible_by_4_in_sequence_l104_104159


namespace continuity_at_2_l104_104937

theorem continuity_at_2 (f : ℝ → ℝ) (x0 : ℝ) (hf : ∀ x, f x = -4 * x ^ 2 - 8) :
  x0 = 2 → ∀ ε > 0, ∃ δ > 0, ∀ x, |x - x0| < δ → |f x + 24| < ε := by
  sorry

end continuity_at_2_l104_104937


namespace volume_of_circumscribed_polyhedron_l104_104113

theorem volume_of_circumscribed_polyhedron (R : ℝ) (V : ℝ) (S_n : ℝ) (h : Π (F_i : ℝ), V = (1/3) * S_n * R) : V = (1/3) * S_n * R :=
sorry

end volume_of_circumscribed_polyhedron_l104_104113


namespace find_x_coordinate_l104_104162

noncomputable def point_on_plane (x y : ℝ) :=
  (|x + y - 1| / Real.sqrt 2 = |x| ∧
   |x| = |y - 3 * x| / Real.sqrt 10)

theorem find_x_coordinate (x y : ℝ) (h : point_on_plane x y) : 
  x = 1 / (4 + Real.sqrt 10 - Real.sqrt 2) :=
sorry

end find_x_coordinate_l104_104162


namespace fractional_equation_root_l104_104079

theorem fractional_equation_root (k : ℚ) (x : ℚ) (h : (2 * k) / (x - 1) - 3 / (1 - x) = 1) : k = -3 / 2 :=
sorry

end fractional_equation_root_l104_104079


namespace totalWatermelons_l104_104434

def initialWatermelons : ℕ := 4
def additionalWatermelons : ℕ := 3

theorem totalWatermelons : initialWatermelons + additionalWatermelons = 7 := by
  sorry

end totalWatermelons_l104_104434


namespace minimum_value_of_reciprocal_sum_l104_104873

theorem minimum_value_of_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h : 2 * a * (-1) - b * 2 + 2 = 0) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 2 * a * (-1) - b * 2 + 2 = 0 ∧ (a + b = 1) ∧ (a = 1/2 ∧ b = 1/2) ∧ (1/a + 1/b = 4) :=
by
  sorry

end minimum_value_of_reciprocal_sum_l104_104873


namespace oliver_total_money_l104_104740

-- Define the initial conditions
def initial_amount : ℕ := 9
def allowance_saved : ℕ := 5
def cost_frisbee : ℕ := 4
def cost_puzzle : ℕ := 3
def birthday_money : ℕ := 8

-- Define the problem statement in Lean
theorem oliver_total_money : 
  (initial_amount + allowance_saved - (cost_frisbee + cost_puzzle) + birthday_money) = 15 := 
by 
  sorry

end oliver_total_money_l104_104740


namespace reflex_angle_at_T_l104_104690

-- Assume points P, Q, R, and S are aligned
def aligned (P Q R S : ℝ × ℝ) : Prop :=
  ∃ a b, ∀ x, x = 0 * a + b + (P.1, Q.1, R.1, S.1)

-- Angles given in the problem
def PQT_angle : ℝ := 150
def RTS_angle : ℝ := 70

-- definition of the reflex angle at T
def reflex_angle (angle : ℝ) : ℝ := 360 - angle

theorem reflex_angle_at_T (P Q R S T : ℝ × ℝ) :
  aligned P Q R S → PQT_angle = 150 → RTS_angle = 70 →
  reflex_angle 40 = 320 :=
by
  sorry

end reflex_angle_at_T_l104_104690


namespace find_m_l104_104123

variables (a b m : ℝ)

def f (x : ℝ) : ℝ := x^3 + a*x^2 + b*x

def f' (x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem find_m (h1 : f m = 0) (h2 : f' m = 0) (h3 : m ≠ 0)
    (h4 : ∃ x, f' x = 0 ∧ ∀ y, x ≤ y → f x ≥ f y ∧ f x = 1/2) :
    m = 3/2 :=
sorry

end find_m_l104_104123


namespace complex_multiplication_l104_104586

theorem complex_multiplication (i : ℂ) (h : i^2 = -1) : i * (2 - i) = 1 + 2 * i :=
by
  sorry

end complex_multiplication_l104_104586


namespace sum_of_tangents_l104_104174

noncomputable def g (x : ℝ) : ℝ :=
  max (max (-7 * x - 25) (2 * x + 5)) (5 * x - 7)

theorem sum_of_tangents (a b c : ℝ) (q : ℝ → ℝ) (hq₁ : ∀ x, q x = k * (x - a) ^ 2 + (-7 * x - 25))
  (hq₂ : ∀ x, q x = k * (x - b) ^ 2 + (2 * x + 5))
  (hq₃ : ∀ x, q x = k * (x - c) ^ 2 + (5 * x - 7)) :
  a + b + c = -34 / 3 := 
sorry

end sum_of_tangents_l104_104174


namespace most_followers_is_sarah_l104_104439

def initial_followers_susy : ℕ := 100
def initial_followers_sarah : ℕ := 50

def susy_week1_new : ℕ := 40
def susy_week2_new := susy_week1_new / 2
def susy_week3_new := susy_week2_new / 2
def susy_total_new := susy_week1_new + susy_week2_new + susy_week3_new
def susy_final_followers := initial_followers_susy + susy_total_new

def sarah_week1_new : ℕ := 90
def sarah_week2_new := sarah_week1_new / 3
def sarah_week3_new := sarah_week2_new / 3
def sarah_total_new := sarah_week1_new + sarah_week2_new + sarah_week3_new
def sarah_final_followers := initial_followers_sarah + sarah_total_new

theorem most_followers_is_sarah : 
    sarah_final_followers ≥ susy_final_followers := by
  sorry

end most_followers_is_sarah_l104_104439


namespace train_pass_man_in_16_seconds_l104_104165

noncomputable def speed_km_per_hr := 54
noncomputable def speed_m_per_s := (speed_km_per_hr * 1000) / 3600
noncomputable def time_to_pass_platform := 16
noncomputable def length_platform := 90.0072
noncomputable def length_train := speed_m_per_s * time_to_pass_platform
noncomputable def time_to_pass_man := length_train / speed_m_per_s

theorem train_pass_man_in_16_seconds :
  time_to_pass_man = 16 :=
by sorry

end train_pass_man_in_16_seconds_l104_104165


namespace probability_of_exactly_two_dice_showing_3_l104_104228

-- Definition of the problem conditions
def n_dice : ℕ := 5
def sides : ℕ := 5
def prob_showing_3 : ℚ := 1/5
def prob_not_showing_3 : ℚ := 4/5
def way_to_choose_2_of_5 : ℕ := Nat.choose 5 2

-- Lean proof problem statement
theorem probability_of_exactly_two_dice_showing_3 : 
  (10 : ℚ) * (prob_showing_3 ^ 2) * (prob_not_showing_3 ^ 3) = 640 / 3125 := 
by sorry

end probability_of_exactly_two_dice_showing_3_l104_104228


namespace remainder_of_M_mod_210_l104_104047

def M : ℤ := 1234567891011

theorem remainder_of_M_mod_210 :
  (M % 210) = 31 :=
by
  have modulus1 : M % 6 = 3 := by sorry
  have modulus2 : M % 5 = 1 := by sorry
  have modulus3 : M % 7 = 2 := by sorry
  -- Using Chinese Remainder Theorem
  sorry

end remainder_of_M_mod_210_l104_104047


namespace sum_a_b_eq_4_l104_104590

-- Define the problem conditions
variables (a b : ℝ)

-- State the conditions
def condition1 : Prop := 2 * a = 8
def condition2 : Prop := a^2 - b = 16

-- State the theorem
theorem sum_a_b_eq_4 (h1 : condition1 a) (h2 : condition2 a b) : a + b = 4 :=
by sorry

end sum_a_b_eq_4_l104_104590


namespace sides_of_polygon_l104_104603

-- Define the conditions
def polygon_sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

-- State the mathematically equivalent proof problem
theorem sides_of_polygon (n : ℕ) (h : polygon_sum_interior_angles n = 1260) : n = 9 := by
  sorry

end sides_of_polygon_l104_104603


namespace spending_total_march_to_july_l104_104759

/-- Given the conditions:
  1. Total amount spent by the beginning of March is 1.2 million,
  2. Total amount spent by the end of July is 5.4 million,
  Prove that the total amount spent during March, April, May, June, and July is 4.2 million. -/
theorem spending_total_march_to_july
  (spent_by_end_of_feb : ℝ)
  (spent_by_end_of_july : ℝ)
  (h1 : spent_by_end_of_feb = 1.2)
  (h2 : spent_by_end_of_july = 5.4) :
  spent_by_end_of_july - spent_by_end_of_feb = 4.2 :=
by
  sorry

end spending_total_march_to_july_l104_104759


namespace johns_total_spending_l104_104166

theorem johns_total_spending
    (online_phone_price : ℝ := 2000)
    (phone_price_increase : ℝ := 0.02)
    (phone_case_price : ℝ := 35)
    (screen_protector_price : ℝ := 15)
    (accessories_discount : ℝ := 0.05)
    (sales_tax : ℝ := 0.06) :
    let store_phone_price := online_phone_price * (1 + phone_price_increase)
    let regular_accessories_price := phone_case_price + screen_protector_price
    let discounted_accessories_price := regular_accessories_price * (1 - accessories_discount)
    let pre_tax_total := store_phone_price + discounted_accessories_price
    let total_spending := pre_tax_total * (1 + sales_tax)
    total_spending = 2212.75 :=
by
    sorry

end johns_total_spending_l104_104166


namespace school_profit_calc_l104_104806

-- Definitions based on the conditions provided
def pizza_slices : Nat := 8
def slices_per_pizza : ℕ := 8
def slice_price : ℝ := 1.0 -- Defining price per slice
def pizzas_bought : ℕ := 55
def cost_per_pizza : ℝ := 6.85
def total_revenue : ℝ := pizzas_bought * slices_per_pizza * slice_price
def total_cost : ℝ := pizzas_bought * cost_per_pizza

-- The lean mathematical statement we need to prove
theorem school_profit_calc :
  total_revenue - total_cost = 63.25 := by
  sorry

end school_profit_calc_l104_104806


namespace find_k_l104_104325

theorem find_k {k : ℝ} (h : (∃ α β : ℝ, α ≠ 0 ∧ β ≠ 0 ∧ α / β = 3 / 1 ∧ α + β = -10 ∧ α * β = k)) : k = 18.75 :=
sorry

end find_k_l104_104325


namespace lcm_of_36_and_105_l104_104218

theorem lcm_of_36_and_105 : Nat.lcm 36 105 = 1260 := by
  sorry

end lcm_of_36_and_105_l104_104218


namespace base6_addition_unique_solution_l104_104965

theorem base6_addition_unique_solution : 
  ∃ (triangle square : ℕ), triangle < 6 ∧ square < 6 ∧
  (43 * 6^2 + triangle * 6 + square) + (1 * 6^2 + triangle * 6 + 5) + (0 * 6^2 + square * 6 + 4) = 45 * 6^2 + square * 6 + 2 ∧
  triangle = 4 ∧ square = 4 :=
by
  sorry

end base6_addition_unique_solution_l104_104965


namespace divides_or_l104_104097

-- Definitions
variables {m n : ℕ} -- using natural numbers (non-negative integers) for simplicity in Lean

-- Hypothesis: m ∨ n + m ∧ n = m + n
theorem divides_or (h : Nat.lcm m n + Nat.gcd m n = m + n) : m ∣ n ∨ n ∣ m :=
sorry

end divides_or_l104_104097


namespace find_constants_for_matrix_condition_l104_104728

noncomputable section

def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![1, 2, 3], ![0, 1, 2], ![1, 0, 1]]

def I : Matrix (Fin 3) (Fin 3) ℝ :=
  1

theorem find_constants_for_matrix_condition :
  ∃ p q r : ℝ, B^3 + p • B^2 + q • B + r • I = 0 :=
by
  use -5, 3, -6
  sorry

end find_constants_for_matrix_condition_l104_104728


namespace log_diff_decreases_l104_104757

-- Define the natural number n
variable (n : ℕ)

-- Proof statement
theorem log_diff_decreases (hn : 0 < n) : 
  (Real.log (n + 1) - Real.log n) = Real.log (1 + 1 / n) ∧ 
  ∀ m : ℕ, ∀ hn' : 0 < m, m > n → Real.log (m + 1) - Real.log m < Real.log (n + 1) - Real.log n := by
  sorry

end log_diff_decreases_l104_104757


namespace probability_intersecting_diagonals_l104_104310

def number_of_vertices := 10

def number_of_diagonals : ℕ := Nat.choose number_of_vertices 2 - number_of_vertices

def number_of_ways_choose_two_diagonals := Nat.choose number_of_diagonals 2

def number_of_sets_of_intersecting_diagonals : ℕ := Nat.choose number_of_vertices 4

def intersection_probability : ℚ :=
  (number_of_sets_of_intersecting_diagonals : ℚ) / (number_of_ways_choose_two_diagonals : ℚ)

theorem probability_intersecting_diagonals :
  intersection_probability = 42 / 119 :=
by
  sorry

end probability_intersecting_diagonals_l104_104310


namespace missed_bus_time_by_l104_104185

def bus_departure_time : Time := Time.mk 8 0 0
def travel_time_minutes : Int := 30
def departure_time_home : Time := Time.mk 7 50 0
def arrival_time_pickup_point : Time := 
  departure_time_home.addMinutes travel_time_minutes

theorem missed_bus_time_by :
  arrival_time_pickup_point.diff bus_departure_time = 20 * 60 :=
by
  sorry

end missed_bus_time_by_l104_104185


namespace small_branches_per_branch_l104_104947

theorem small_branches_per_branch (x : ℕ) (h1 : 1 + x + x^2 = 57) : x = 7 :=
by {
  sorry
}

end small_branches_per_branch_l104_104947


namespace line_canonical_eqn_l104_104479

theorem line_canonical_eqn 
  (x y z : ℝ)
  (h1 : x - y + z - 2 = 0)
  (h2 : x - 2*y - z + 4 = 0) :
  ∃ a : ℝ, ∃ b : ℝ, ∃ c : ℝ,
    (a = (x - 8)/3) ∧ (b = (y - 6)/2) ∧ (c = z/(-1)) ∧ (a = b) ∧ (b = c) ∧ (c = a) :=
by sorry

end line_canonical_eqn_l104_104479


namespace geometric_sequence_a_formula_l104_104233

noncomputable def a (n : ℕ) : ℤ :=
  if n = 1 then 1
  else if n = 2 then 2
  else n - 2

noncomputable def b (n : ℕ) : ℤ :=
  a (n + 1) - a n

theorem geometric_sequence (n : ℕ) (h : n ≥ 2) : 
  b n = (-1) * b (n - 1) := 
  sorry

theorem a_formula (n : ℕ) : 
  a n = (-1) ^ (n - 1) := 
  sorry

end geometric_sequence_a_formula_l104_104233


namespace math_club_team_selection_l104_104109

theorem math_club_team_selection :
  let boys := 10
  let girls := 12
  let total := boys + girls
  let team_size := 8
  (Nat.choose total team_size - Nat.choose girls team_size - Nat.choose boys team_size = 319230) :=
by
  sorry

end math_club_team_selection_l104_104109


namespace circle_radius_l104_104583

theorem circle_radius (A : ℝ) (k : ℝ) (r : ℝ) (h : A = k * π * r^2) (hA : A = 225 * π) (hk : k = 4) : 
  r = 7.5 :=
by 
  sorry

end circle_radius_l104_104583


namespace count_five_digit_numbers_with_digit_8_l104_104226

theorem count_five_digit_numbers_with_digit_8 : 
    let total_numbers := 99999 - 10000 + 1
    let without_8 := 8 * (9 ^ 4)
    90000 - without_8 = 37512 := by
    let total_numbers := 99999 - 10000 + 1 -- Total number of five-digit numbers
    let without_8 := 8 * (9 ^ 4) -- Number of five-digit numbers without any '8'
    show total_numbers - without_8 = 37512
    sorry

end count_five_digit_numbers_with_digit_8_l104_104226


namespace problem_l104_104705

-- Definitions and conditions
def seq (a : ℕ → ℚ) : Prop :=
  a 1 = 1 ∧ (∀ n, 2 ≤ n → 2 * a n / (a n * (Finset.sum (Finset.range n) a) - (Finset.sum (Finset.range n) a) ^ 2) = 1)

-- Sum of the first n terms
def S (a : ℕ → ℚ) (n : ℕ) : ℚ := Finset.sum (Finset.range n) a

-- The proof statement
theorem problem (a : ℕ → ℚ) (h : seq a) : S a 2017 = 1 / 1009 := sorry

end problem_l104_104705


namespace least_number_subtracted_l104_104636

theorem least_number_subtracted (x : ℕ) (y : ℕ) (h : 2590 - x = y) : 
  y % 9 = 6 ∧ y % 11 = 6 ∧ y % 13 = 6 → x = 10 := 
by
  sorry

end least_number_subtracted_l104_104636


namespace cost_of_2000_pieces_of_gum_l104_104907

theorem cost_of_2000_pieces_of_gum
  (cost_per_piece_in_cents : Nat)
  (pieces_of_gum : Nat)
  (conversion_rate_cents_to_dollars : Nat)
  (h1 : cost_per_piece_in_cents = 5)
  (h2 : pieces_of_gum = 2000)
  (h3 : conversion_rate_cents_to_dollars = 100) :
  (cost_per_piece_in_cents * pieces_of_gum) / conversion_rate_cents_to_dollars = 100 := 
by
  sorry

end cost_of_2000_pieces_of_gum_l104_104907


namespace overall_percent_change_l104_104102

theorem overall_percent_change (x : ℝ) : 
  (0.85 * x * 1.25 * 0.9 / x - 1) * 100 = -4.375 := 
by 
  sorry

end overall_percent_change_l104_104102


namespace speed_difference_l104_104158

theorem speed_difference (h_cyclist : 88 / 8 = 11) (h_car : 48 / 8 = 6) :
  (11 - 6 = 5) :=
by
  sorry

end speed_difference_l104_104158


namespace necessary_but_not_sufficient_l104_104781

theorem necessary_but_not_sufficient (a b : ℝ) : 
 (a > b) ↔ (a-1 > b+1) :=
by {
  sorry
}

end necessary_but_not_sufficient_l104_104781


namespace smallest_base_conversion_l104_104155

theorem smallest_base_conversion :
  let n1 := 8 * 9 + 5 -- 85 in base 9
  let n2 := 2 * 6^2 + 1 * 6 -- 210 in base 6
  let n3 := 1 * 4^3 -- 1000 in base 4
  let n4 := 1 * 2^7 - 1 -- 1111111 in base 2
  n3 < n1 ∧ n3 < n2 ∧ n3 < n4 :=
by
  let n1 := 8 * 9 + 5
  let n2 := 2 * 6^2 + 1 * 6
  let n3 := 1 * 4^3
  let n4 := 1 * 2^7 - 1
  sorry

end smallest_base_conversion_l104_104155


namespace delaney_missed_bus_time_l104_104181

def busDepartureTime : Nat := 480 -- 8:00 a.m. = 8 * 60 minutes
def travelTime : Nat := 30 -- 30 minutes
def departureFromHomeTime : Nat := 470 -- 7:50 a.m. = 7 * 60 + 50 minutes

theorem delaney_missed_bus_time :
  (departureFromHomeTime + travelTime - busDepartureTime) = 20 :=
by
  -- proof would go here
  sorry

end delaney_missed_bus_time_l104_104181


namespace max_min_values_l104_104675

-- Define the function f(x) = x^2 - 2ax + 1
def f (x a : ℝ) : ℝ := x ^ 2 - 2 * a * x + 1

-- Define the interval [0, 2]
def interval : Set ℝ := Set.Icc 0 2

theorem max_min_values (a : ℝ) : 
  (a > 2 → (∀ x ∈ interval, f x a ≤ 1) ∧ (∃ x ∈ interval, f x a = 5 - 4 * a))
  ∧ (1 ≤ a ∧ a ≤ 2 → (∀ x ∈ interval, f x a ≤ 5 - 4 * a) ∧ (∃ x ∈ interval, f x a = -a^2 + 1))
  ∧ (0 ≤ a ∧ a < 1 → (∀ x ∈ interval, f x a ≤ 1) ∧ (∃ x ∈ interval, f x a = -a^2 + 1))
  ∧ (a < 0 → (∀ x ∈ interval, f x a ≤ 5 - 4 * a) ∧ (∃ x ∈ interval, f x a = 1)) := by
  sorry

end max_min_values_l104_104675


namespace integral_sign_negative_l104_104986

open Topology

-- Define the problem
theorem integral_sign_negative {a b : ℝ} (f : ℝ → ℝ) (h_cont : ContinuousOn f (Set.Icc a b)) (h_lt : ∀ x ∈ Set.Icc a b, f x < 0) (h_ab : a < b) :
  ∫ x in a..b, f x < 0 := 
sorry

end integral_sign_negative_l104_104986


namespace apples_chosen_l104_104426

def total_fruits : ℕ := 12
def bananas : ℕ := 4
def oranges : ℕ := 5
def total_other_fruits := bananas + oranges

theorem apples_chosen : total_fruits - total_other_fruits = 3 :=
by sorry

end apples_chosen_l104_104426


namespace average_speed_l104_104777

theorem average_speed (D : ℝ) (hD : D > 0) :
  let t1 := (D / 3) / 80
  let t2 := (D / 3) / 15
  let t3 := (D / 3) / 48
  let total_time := t1 + t2 + t3
  let avg_speed := D / total_time
  avg_speed = 30 :=
by
  sorry

end average_speed_l104_104777


namespace total_money_l104_104278

-- Conditions
def mark_amount : ℚ := 5 / 6
def carolyn_amount : ℚ := 2 / 5

-- Combine both amounts and state the theorem to be proved
theorem total_money : mark_amount + carolyn_amount = 1.233 := by
  -- placeholder for the actual proof
  sorry

end total_money_l104_104278


namespace min_value_of_expression_l104_104387

theorem min_value_of_expression (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y)
    (h3 : 4 * x^2 + 4 * x * y + y^2 + 2 * x + y - 6 = 0) : 
    ∃ (c : ℝ), c = x * (1 - y) ∧ c = - (1 / 8) :=
by
  sorry

end min_value_of_expression_l104_104387


namespace probability_drawing_balls_l104_104783

theorem probability_drawing_balls :
  let total_balls := 15
  let red_balls := 10
  let blue_balls := 5
  let drawn_balls := 4
  let num_ways_to_draw_4_balls := Nat.choose total_balls drawn_balls
  let num_ways_to_draw_3_red_1_blue := (Nat.choose red_balls 3) * (Nat.choose blue_balls 1)
  let num_ways_to_draw_1_red_3_blue := (Nat.choose red_balls 1) * (Nat.choose blue_balls 3)
  let total_favorable_outcomes := num_ways_to_draw_3_red_1_blue + num_ways_to_draw_1_red_3_blue
  let probability := total_favorable_outcomes / num_ways_to_draw_4_balls
  probability = (140 : ℚ) / 273 :=
sorry

end probability_drawing_balls_l104_104783


namespace extinction_prob_one_l104_104793

-- Define the probabilities
def p : ℝ := 0.6
def q : ℝ := 0.4

-- Define the extinction probability function
def extinction_prob (v : ℕ → ℝ) (k : ℕ) : ℝ :=
  if k = 0 then 1
  else p * v (k + 1) + q * v (k - 1)

-- State the theorem
theorem extinction_prob_one (v : ℕ → ℝ) :
  extinction_prob v 1 = 2 / 3 :=
sorry

end extinction_prob_one_l104_104793


namespace bus_driver_hours_l104_104651

theorem bus_driver_hours (h : ℕ) (regular_rate : ℕ) (extra_rate1 : ℕ) (extra_rate2 : ℕ) (total_earnings : ℕ)
  (h1 : regular_rate = 14)
  (h2 : extra_rate1 = (14 + (14 * 35 / 100)))
  (h3: extra_rate2 = (14 + (14 * 75 / 100)))
  (h4: total_earnings = 1230)
  (h5: total_earnings = 40 * regular_rate + 10 * extra_rate1 + (h - 50) * extra_rate2)
  (condition : 50 < h) :
  h = 69 :=
by
  sorry

end bus_driver_hours_l104_104651


namespace fully_loaded_truck_weight_l104_104969

def empty_truck : ℕ := 12000
def weight_soda_crate : ℕ := 50
def num_soda_crates : ℕ := 20
def weight_dryer : ℕ := 3000
def num_dryers : ℕ := 3
def weight_fresh_produce : ℕ := 2 * (weight_soda_crate * num_soda_crates)

def total_loaded_truck_weight : ℕ :=
  empty_truck + (weight_soda_crate * num_soda_crates) + weight_fresh_produce + (weight_dryer * num_dryers)

theorem fully_loaded_truck_weight : total_loaded_truck_weight = 24000 := by
  sorry

end fully_loaded_truck_weight_l104_104969


namespace num_pos_integers_congruent_to_4_mod_7_l104_104858

theorem num_pos_integers_congruent_to_4_mod_7 (n : ℕ) (h1 : n < 500) (h2 : ∃ k : ℕ, n = 7 * k + 4) : 
  ∃ total : ℕ, total = 71 :=
sorry

end num_pos_integers_congruent_to_4_mod_7_l104_104858


namespace daily_profit_functional_relationship_daily_profit_maximizes_at_120_selling_price_for_2400_profit_l104_104032

-- Given conditions
def cost_price : ℝ := 80
def daily_sales_quantity (x : ℝ) : ℝ := -2 * x + 320
def daily_profit (x : ℝ) : ℝ := (x - cost_price) * daily_sales_quantity x

-- Part 1: Functional relationship
theorem daily_profit_functional_relationship (x : ℝ) (hx : 80 ≤ x ∧ x ≤ 160) : daily_profit x = -2 * x^2 + 480 * x - 25600 :=
by sorry

-- Part 2: Maximizing daily profit
theorem daily_profit_maximizes_at_120 (hx : 80 ≤ 120 ∧ 120 ≤ 160) : daily_profit 120 = 3200 :=
by sorry

-- Part 3: Selling price for a daily profit of $2400
theorem selling_price_for_2400_profit (hx : 80 ≤ 100 ∧ 100 ≤ 160) : daily_profit 100 = 2400 :=
by sorry

end daily_profit_functional_relationship_daily_profit_maximizes_at_120_selling_price_for_2400_profit_l104_104032


namespace tile_covering_possible_l104_104558

theorem tile_covering_possible (m n : ℕ) (hm : m ≥ 2) (hn : n ≥ 2) :
  ((m % 6 = 0) ∨ (n % 6 = 0)) := 
sorry

end tile_covering_possible_l104_104558


namespace smallest_integer_CC4_DD6_rep_l104_104146

-- Lean 4 Statement
theorem smallest_integer_CC4_DD6_rep (C D : ℕ) (hC : C < 4) (hD : D < 6) :
  (5 * C = 7 * D) → (5 * C = 35 ∧ 7 * D = 35) :=
by
  sorry

end smallest_integer_CC4_DD6_rep_l104_104146


namespace expression_is_integer_l104_104112

theorem expression_is_integer (n : ℕ) : 
    ∃ k : ℤ, (n^5 : ℤ) / 5 + (n^3 : ℤ) / 3 + (7 * n : ℤ) / 15 = k :=
by
  sorry

end expression_is_integer_l104_104112


namespace time_after_hours_l104_104142

-- Definitions based on conditions
def current_time : ℕ := 3
def hours_later : ℕ := 2517
def clock_cycle : ℕ := 12

-- Statement to prove
theorem time_after_hours :
  (current_time + hours_later) % clock_cycle = 12 := 
sorry

end time_after_hours_l104_104142


namespace product_divisible_by_5_l104_104430

theorem product_divisible_by_5 (a b : ℕ) (ha : a > 0) (hb : b > 0)
  (h : ∃ k, a * b = 5 * k) : a % 5 = 0 ∨ b % 5 = 0 :=
by
  sorry

end product_divisible_by_5_l104_104430


namespace customers_left_is_31_l104_104663

-- Define the initial number of customers
def initial_customers : ℕ := 33

-- Define the number of additional customers
def additional_customers : ℕ := 26

-- Define the final number of customers after some left and new ones came
def final_customers : ℕ := 28

-- Define the number of customers who left 
def customers_left (x : ℕ) : Prop :=
  (initial_customers - x) + additional_customers = final_customers

-- The proof statement that we aim to prove
theorem customers_left_is_31 : ∃ x : ℕ, customers_left x ∧ x = 31 :=
by
  use 31
  unfold customers_left
  sorry

end customers_left_is_31_l104_104663


namespace files_per_folder_l104_104737

-- Define the conditions
def initial_files : ℕ := 43
def deleted_files : ℕ := 31
def num_folders : ℕ := 2

-- Define the final problem statement
theorem files_per_folder :
  (initial_files - deleted_files) / num_folders = 6 :=
by
  -- proof would go here
  sorry

end files_per_folder_l104_104737


namespace eggs_per_basket_l104_104752

theorem eggs_per_basket
  (kids : ℕ)
  (friends : ℕ)
  (adults : ℕ)
  (baskets : ℕ)
  (eggs_per_person : ℕ)
  (htotal : kids + friends + adults + 1 = 20)
  (eggs_total : (kids + friends + adults + 1) * eggs_per_person = 180)
  (baskets_count : baskets = 15)
  : (180 / 15) = 12 :=
by
  sorry

end eggs_per_basket_l104_104752


namespace birds_left_after_a_week_l104_104493

def initial_chickens := 300
def initial_turkeys := 200
def initial_guinea_fowls := 80
def daily_chicken_loss := 20
def daily_turkey_loss := 8
def daily_guinea_fowl_loss := 5
def days_in_a_week := 7

def remaining_chickens := initial_chickens - daily_chicken_loss * days_in_a_week
def remaining_turkeys := initial_turkeys - daily_turkey_loss * days_in_a_week
def remaining_guinea_fowls := initial_guinea_fowls - daily_guinea_fowl_loss * days_in_a_week

def total_remaining_birds := remaining_chickens + remaining_turkeys + remaining_guinea_fowls

theorem birds_left_after_a_week : total_remaining_birds = 349 := by
  sorry

end birds_left_after_a_week_l104_104493


namespace value_of_x_2011_l104_104987

-- Conditions

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f(x)

def is_increasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f(x) < f(y)

def is_arithmetic_sequence (x : ℕ → ℝ) (d : ℝ) : Prop :=
  ∃ a : ℝ, ∀ n : ℕ, x n = a + d * (n - 1)

def sum_to_zero (f : ℝ → ℝ) (x : ℕ → ℝ) : Prop :=
  f (x 8) + f (x 9) + f (x 10) + f (x 11) = 0

-- Problem

theorem value_of_x_2011 {f : ℝ → ℝ} {x : ℕ → ℝ} (h1 : is_odd_function f) (h2 : is_increasing_function f)
  (h3 : is_arithmetic_sequence x 2) (h4 : sum_to_zero f x) : x 2011 = 4003 :=
by
  sorry

end value_of_x_2011_l104_104987


namespace no_solutions_for_a_ne_4_solutions_for_a_eq_4_infinite_l104_104699

-- Part (i)
theorem no_solutions_for_a_ne_4 (a : ℕ) (h : a ≠ 4) :
  ¬∃ (u v : ℕ), (u > 0 ∧ v > 0 ∧ u^2 + v^2 - a * u * v + 2 = 0) :=
by sorry

-- Part (ii)
theorem solutions_for_a_eq_4_infinite :
  ∃ (a_seq : ℕ → ℕ),
    (a_seq 0 = 1 ∧ a_seq 1 = 3 ∧
     ∀ n, a_seq (n + 2) = 4 * a_seq (n + 1) - a_seq n ∧
    ∀ n, (a_seq n) > 0 ∧ (a_seq (n + 1)) > 0 ∧ (a_seq n)^2 + (a_seq (n + 1))^2 - 4 * (a_seq n) * (a_seq (n + 1)) + 2 = 0) :=
by sorry

end no_solutions_for_a_ne_4_solutions_for_a_eq_4_infinite_l104_104699


namespace derivative_of_f_tangent_line_at_pi_l104_104992

noncomputable def f (x : ℝ) : ℝ := (Real.sin x) / x

theorem derivative_of_f (x : ℝ) (h : x ≠ 0) : deriv f x = (x * Real.cos x - Real.sin x) / (x ^ 2) :=
  sorry

theorem tangent_line_at_pi : 
  let M := (Real.pi, 0)
  let slope := -1 / Real.pi
  let tangent_line (x : ℝ) : ℝ := -x / Real.pi + 1
  ∀ (x y : ℝ), (x, y) = M → y = tangent_line x :=
  sorry

end derivative_of_f_tangent_line_at_pi_l104_104992


namespace circle_equation_translation_l104_104117

theorem circle_equation_translation (x y : ℝ) :
  x^2 + y^2 - 4 * x + 6 * y - 68 = 0 → (x - 2)^2 + (y + 3)^2 = 81 :=
by
  intro h
  sorry

end circle_equation_translation_l104_104117


namespace reciprocal_of_neg_two_l104_104454

-- Define the condition of reciprocal
def is_reciprocal (y x : ℝ) : Prop := y * x = 1

-- Define the number in question
def y : ℝ := -2

-- Define the expected reciprocal
def x : ℝ := -1 / 2

-- State the theorem
theorem reciprocal_of_neg_two : is_reciprocal y x :=
by
  -- Proof can be provided here
  sorry

end reciprocal_of_neg_two_l104_104454


namespace problem1_part1_problem1_part2_problem2_part1_problem2_part2_l104_104484

-- Probability Distribution
theorem problem1_part1 :
  let X := binom 10 0.5 in
  let p := [0.5, 0.5] in
  (P(X = 0) = 1 / 12) ∧ (P(X = 1) = 5 / 12) ∧ (P(X = 2) = 5 / 12) ∧ (P(X = 3) = 1 / 12) := 
by sorry

-- Expected Value
theorem problem1_part2 :
  let X := binom 10 0.5 in
  E X = 3 / 2 :=
by sorry

-- Cutoff Point for Grade C
theorem problem2_part1 :
  let Y := normal 75.8 6, N(75.8, 36) in
  let eta := (Y - 75.8) / 6 in
  (η ≤ 1.04) ≈ 0.85 →
  let cutoff := 75.8 - 1.04 * 6 in
  round(cutoff) = 70 :=
by sorry

-- Value of k to Maximize P(ξ = k)
theorem problem2_part2 :
  let ξ := binom 800 0.788 in
  (P((ξ = k)) is maximized) → k = 631 →
  k :=
by sorry

end problem1_part1_problem1_part2_problem2_part1_problem2_part2_l104_104484


namespace line_slope_translation_l104_104077

theorem line_slope_translation (k : ℝ) (b : ℝ) :
  (∀ x y : ℝ, y = k * x + b → y = k * (x - 3) + (b + 2)) → k = 2 / 3 :=
by
  intro h
  sorry

end line_slope_translation_l104_104077


namespace line_intersection_l104_104677

/-- Prove the intersection of the lines given by the equations
    8x - 5y = 10 and 3x + 2y = 1 is (25/31, -22/31) -/
theorem line_intersection :
  ∃ (x y : ℚ), 8 * x - 5 * y = 10 ∧ 3 * x + 2 * y = 1 ∧ x = 25 / 31 ∧ y = -22 / 31 :=
by
  sorry

end line_intersection_l104_104677


namespace extinction_probability_l104_104800

-- Definitions from conditions
def prob_divide : ℝ := 0.6
def prob_die : ℝ := 0.4

-- Statement of the theorem
theorem extinction_probability :
  ∃ (v : ℝ), v = 2 / 3 :=
by
  sorry

end extinction_probability_l104_104800


namespace problem_1_problem_2_l104_104461

def A (x : ℝ) : Prop := x^2 - 3*x - 10 ≤ 0
def B (m x : ℝ) : Prop := m + 1 ≤ x ∧ x ≤ 2*m - 1

theorem problem_1 (m : ℝ) : (∀ x, B m x → A x)  →  m ≤ 3 := 
sorry

theorem problem_2 (m : ℝ) : (¬ ∃ x, A x ∧ B m x) ↔ (m < 2 ∨ 4 < m) := 
sorry

end problem_1_problem_2_l104_104461


namespace ellipse_foci_distance_l104_104520

theorem ellipse_foci_distance 
  (h : ∀ x y : ℝ, 9 * x^2 + y^2 = 144) : 
  ∃ c : ℝ, c = 16 * Real.sqrt 2 :=
  sorry

end ellipse_foci_distance_l104_104520


namespace dave_initial_apps_l104_104964

theorem dave_initial_apps (x : ℕ) (h1 : x - 18 = 5) : x = 23 :=
by {
  -- This is where the proof would go 
  sorry -- The proof is omitted as per instructions
}

end dave_initial_apps_l104_104964


namespace additional_payment_is_65_l104_104945

def installments (n : ℕ) : ℤ := 65
def first_payment : ℕ := 20
def first_amount : ℤ := 410
def remaining_payment (x : ℤ) : ℕ := 45
def remaining_amount (x : ℤ) : ℤ := 410 + x
def average_amount : ℤ := 455

-- Define the total amount paid using both methods
def total_amount (x : ℤ) : ℤ := (20 * 410) + (45 * (410 + x))
def total_average : ℤ := 65 * 455

theorem additional_payment_is_65 :
  total_amount 65 = total_average :=
sorry

end additional_payment_is_65_l104_104945


namespace ratio_female_to_male_l104_104956

-- Definitions for the conditions
def average_age_female (f : ℕ) : ℕ := 40 * f
def average_age_male (m : ℕ) : ℕ := 25 * m
def average_age_total (f m : ℕ) : ℕ := (30 * (f + m))

-- Statement to prove
theorem ratio_female_to_male (f m : ℕ) 
  (h_avg_f: average_age_female f = 40 * f)
  (h_avg_m: average_age_male m = 25 * m)
  (h_avg_total: average_age_total f m = 30 * (f + m)) : 
  f / m = 1 / 2 :=
by
  sorry

end ratio_female_to_male_l104_104956


namespace factor_polynomial_l104_104196

theorem factor_polynomial (x : ℝ) : 
  54 * x ^ 5 - 135 * x ^ 9 = 27 * x ^ 5 * (2 - 5 * x ^ 4) :=
by 
  sorry

end factor_polynomial_l104_104196


namespace carl_insurance_payment_percentage_l104_104026

variable (property_damage : ℝ) (medical_bills : ℝ) 
          (total_cost : ℝ) (carl_payment : ℝ) (insurance_payment_percentage : ℝ)

theorem carl_insurance_payment_percentage :
  property_damage = 40000 ∧
  medical_bills = 70000 ∧
  total_cost = property_damage + medical_bills ∧
  carl_payment = 22000 ∧
  carl_payment = 0.20 * total_cost →
  insurance_payment_percentage = 100 - 20 :=
by
  sorry

end carl_insurance_payment_percentage_l104_104026


namespace find_PQ_length_l104_104414

-- Defining the problem parameters
variables {X Y Z P Q R : Type}
variables (dXY dXZ dPQ dPR : ℝ)
variable (angle_common : ℝ)

-- Conditions:
def angle_XYZ_PQR_common : Prop :=
  angle_common = 150 ∧ 
  dXY = 10 ∧
  dXZ = 20 ∧
  dPQ = 5 ∧
  dPR = 12

-- Question: Prove PQ = 2.5 given the conditions
theorem find_PQ_length
  (h : angle_XYZ_PQR_common dXY dXZ dPQ dPR angle_common) :
  dPQ = 2.5 :=
sorry

end find_PQ_length_l104_104414


namespace inequality_solution_set_l104_104901

theorem inequality_solution_set :
  { x : ℝ | (10 * x^2 + 20 * x - 68) / ((2 * x - 3) * (x + 4) * (x - 2)) < 3 } =
  { x : ℝ | (-4 < x ∧ x < -2) ∨ (-1 / 3 < x ∧ x < 3 / 2) } :=
by
  sorry

end inequality_solution_set_l104_104901


namespace remainder_when_divided_by_6_l104_104161

theorem remainder_when_divided_by_6 :
  ∃ (n : ℕ), (∃ k : ℕ, n = 3 * k + 2 ∧ ∃ m : ℕ, k = 4 * m + 3) → n % 6 = 5 :=
by
  sorry

end remainder_when_divided_by_6_l104_104161


namespace sequence_terms_distinct_l104_104753

theorem sequence_terms_distinct (n m : ℕ) (hnm : n ≠ m) : 
  (n / (n + 1) : ℚ) ≠ (m / (m + 1) : ℚ) :=
sorry

end sequence_terms_distinct_l104_104753


namespace sequence_odd_l104_104592

theorem sequence_odd (a : ℕ → ℕ)
  (ha1 : a 1 = 2)
  (ha2 : a 2 = 7)
  (hr : ∀ n ≥ 2, -1 < (a (n + 1) : ℤ) - (a n)^2 / a (n - 1) ∧ (a (n + 1) : ℤ) - (a n)^2 / a (n - 1) ≤ 1) :
  ∀ n > 1, Odd (a n) := 
  sorry

end sequence_odd_l104_104592


namespace reciprocal_of_neg_two_l104_104459

theorem reciprocal_of_neg_two : ∃ y : ℝ, (-2) * y = 1 ∧ y = -1/2 :=
by
  use -1/2
  split
  · -- Prove that (-2) * (-1/2) = 1
    sorry
  · -- Prove that y = -1/2
    rfl

end reciprocal_of_neg_two_l104_104459


namespace reciprocal_of_neg_two_l104_104449

variable a : ℤ

theorem reciprocal_of_neg_two (h : a = -2) : ∃ b : ℚ, a * b = 1 ∧ b = -1 / 2 := 
by
  exists (-1 / 2)
  rw [h]
  split
  · norm_num
  · refl

end reciprocal_of_neg_two_l104_104449


namespace countColorings_l104_104528

-- Defining the function that counts the number of valid colorings
def validColorings (n : ℕ) : ℕ :=
  if n = 0 then 1
  else 3 * 2^n - 2

-- Theorem specifying the number of colorings of the grid of length n
theorem countColorings (n : ℕ) : validColorings n = 3 * 2^n - 2 :=
by
  sorry

end countColorings_l104_104528


namespace correct_expression_for_representatives_l104_104164

/-- Definition for the number of representatives y given the class size x
    and the conditions that follow. -/
def elect_representatives (x : ℕ) : ℕ :=
  if 6 < x % 10 then (x + 3) / 10 else x / 10

theorem correct_expression_for_representatives (x : ℕ) :
  elect_representatives x = (x + 3) / 10 :=
by
  sorry

end correct_expression_for_representatives_l104_104164


namespace neg_prop_p_l104_104703

theorem neg_prop_p :
  (¬ (∃ x : ℝ, x^3 - x^2 + 1 ≤ 0)) ↔ (∀ x : ℝ, x^3 - x^2 + 1 > 0) :=
by
  sorry

end neg_prop_p_l104_104703


namespace find_a5_l104_104879

noncomputable def arithmetic_sequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
a₁ + (n - 1) * d

theorem find_a5 (a₁ d : ℚ) (h₁ : arithmetic_sequence a₁ d 1 + arithmetic_sequence a₁ d 5 - arithmetic_sequence a₁ d 8 = 1)
(h₂ : arithmetic_sequence a₁ d 9 - arithmetic_sequence a₁ d 2 = 5) :
arithmetic_sequence a₁ d 5 = 6 :=
sorry

end find_a5_l104_104879


namespace max_profit_at_boundary_l104_104330

noncomputable def profit (x : ℝ) : ℝ :=
  -50 * (x - 55) ^ 2 + 11250

def within_bounds (x : ℝ) : Prop :=
  40 ≤ x ∧ x ≤ 52

theorem max_profit_at_boundary :
  within_bounds 52 ∧ 
  (∀ x : ℝ, within_bounds x → profit x ≤ profit 52) :=
by
  sorry

end max_profit_at_boundary_l104_104330


namespace solve_inequality_l104_104044

theorem solve_inequality (x : ℝ) :
  (3 * x - 2 < (x + 2)^2) ∧ ((x + 2)^2 < 9 * x - 6) ↔ (2 < x) ∧ (x < 3) := by
  sorry

end solve_inequality_l104_104044


namespace metres_sold_is_200_l104_104341

-- Define the conditions
def loss_per_metre : ℕ := 6
def cost_price_per_metre : ℕ := 66
def total_selling_price : ℕ := 12000

-- Define the selling price per metre based on the conditions
def selling_price_per_metre := cost_price_per_metre - loss_per_metre

-- Define the number of metres sold
def metres_sold : ℕ := total_selling_price / selling_price_per_metre

-- Proof statement: Check if the number of metres sold equals 200
theorem metres_sold_is_200 : metres_sold = 200 :=
  by
  sorry

end metres_sold_is_200_l104_104341


namespace movie_marathon_first_movie_length_l104_104465

theorem movie_marathon_first_movie_length 
  (x : ℝ)
  (h2 : 1.5 * x = second_movie)
  (h3 : second_movie + x - 1 = last_movie)
  (h4 : (x + second_movie + last_movie) = 9)
  (h5 : last_movie = 2.5 * x - 1) :
  x = 2 :=
by
  sorry

end movie_marathon_first_movie_length_l104_104465


namespace subtracted_value_from_numbers_l104_104904

theorem subtracted_value_from_numbers (A B C D E X : ℝ) 
  (h1 : (A + B + C + D + E) / 5 = 5)
  (h2 : ((A - X) + (B - X) + (C - X) + (D - X) + E) / 5 = 3.4) :
  X = 2 :=
by
  sorry

end subtracted_value_from_numbers_l104_104904


namespace reciprocal_of_neg_two_l104_104458

theorem reciprocal_of_neg_two : ∃ y : ℝ, (-2) * y = 1 ∧ y = -1/2 :=
by
  use -1/2
  split
  · -- Prove that (-2) * (-1/2) = 1
    sorry
  · -- Prove that y = -1/2
    rfl

end reciprocal_of_neg_two_l104_104458


namespace min_value_expression_l104_104421

theorem min_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  ∃ (v : ℝ), (∀ x y z : ℝ, 0 < x → 0 < y → 0 < z →
    8 * x^4 + 12 * y^4 + 18 * z^4 + 25 / (x * y * z) ≥ v) ∧ v = 30 :=
by
  sorry

end min_value_expression_l104_104421


namespace race_ordering_l104_104717

theorem race_ordering
  (Lotar Manfred Jan Victor Eddy : ℕ) 
  (h1 : Lotar < Manfred) 
  (h2 : Manfred < Jan) 
  (h3 : Jan < Victor) 
  (h4 : Eddy < Victor) : 
  ∀ x, x = Victor ↔ ∀ y, (y = Lotar ∨ y = Manfred ∨ y = Jan ∨ y = Eddy) → y < x :=
by
  sorry

end race_ordering_l104_104717


namespace find_angle_A_l104_104720

def triangle_ABC_angle_A (a b : ℝ) (B A : ℝ) (acute : Prop)
  (ha : a = 2 * Real.sqrt 3)
  (hb : b = 2 * Real.sqrt 2)
  (hB: B = Real.pi / 4)
  (hacute: acute) : Prop :=
  A = Real.pi / 3

theorem find_angle_A 
  (a b A B : ℝ) (acute : Prop)
  (ha : a = 2 * Real.sqrt 3)
  (hb : b = 2 * Real.sqrt 2)
  (hB: B = Real.pi / 4)
  (hacute: acute)
  (h_conditions : triangle_ABC_angle_A a b B A acute ha hb hB hacute) : 
  A = Real.pi / 3 := 
sorry

end find_angle_A_l104_104720


namespace count_integers_modulo_l104_104860

theorem count_integers_modulo (n : ℕ) (h₁ : n < 500) (h₂ : n % 7 = 4) : (setOf (λ n, n > 0 ∧ n < 500 ∧ n % 7 = 4)).card = 71 :=
sorry

end count_integers_modulo_l104_104860


namespace one_plus_i_pow_four_eq_neg_four_l104_104812

theorem one_plus_i_pow_four_eq_neg_four : (1 + complex.I)^4 = -4 :=
by
  sorry

end one_plus_i_pow_four_eq_neg_four_l104_104812


namespace find_m_l104_104395

theorem find_m (m : ℕ) (h1 : (3 * m - 7) % 2 = 0) (h2 : 3 * m - 7 < 0) : m = 1 := 
by
  sorry

end find_m_l104_104395


namespace average_speed_l104_104934

-- Define the problem conditions and provide the proof statement
theorem average_speed (D : ℝ) (hD0 : D > 0) : 
  let speed_1 := 80
  let speed_2 := 24
  let speed_3 := 60
  let time_1 := (D / 3) / speed_1
  let time_2 := (D / 3) / speed_2
  let time_3 := (D / 3) / speed_3
  let total_time := time_1 + time_2 + time_3
  let average_speed := D / total_time
  average_speed = 720 / 17 := 
by
  sorry

end average_speed_l104_104934


namespace man_rate_in_still_water_l104_104933

theorem man_rate_in_still_water (Vm Vs : ℝ) :
  Vm + Vs = 20 ∧ Vm - Vs = 8 → Vm = 14 :=
by
  sorry

end man_rate_in_still_water_l104_104933


namespace find_a_b_c_l104_104681

variable (a b c : ℚ)

def parabola (x : ℚ) : ℚ := a * x^2 + b * x + c

def vertex_condition := ∀ x, parabola a b c x = a * (x - 3)^2 - 2
def contains_point := parabola a b c 0 = 5

theorem find_a_b_c : vertex_condition a b c ∧ contains_point a b c → a + b + c = 10 / 9 :=
by
sorry

end find_a_b_c_l104_104681


namespace unique_nat_number_sum_preceding_eq_self_l104_104225

theorem unique_nat_number_sum_preceding_eq_self :
  ∃! (n : ℕ), (n * (n - 1)) / 2 = n :=
sorry

end unique_nat_number_sum_preceding_eq_self_l104_104225


namespace probability_problem_l104_104169

def ang_blocks : List String := ["red", "blue", "yellow", "white", "green", "orange"]
def ben_blocks : List String := ["red", "blue", "yellow", "white", "green", "orange"]
def jasmin_blocks : List String := ["red", "blue", "yellow", "white", "green", "orange"]

def boxes : Fin 6 := sorry  -- represents 6 empty boxes
def white_restriction (box : Fin 6) : Prop := box ≠ 0  -- white block can't be in the first box

def probability_at_least_one_box_three_same_color : ℚ := 1 / 72  -- The given probability

theorem probability_problem (p q : ℕ) 
  (hpq_coprime : Nat.gcd p q = 1) 
  (hprob_eq : probability_at_least_one_box_three_same_color = p / q) :
  p + q = 73 :=
sorry

end probability_problem_l104_104169


namespace cartesian_coordinates_problem_l104_104413

theorem cartesian_coordinates_problem
  (x1 y1 x2 y2 : ℕ)
  (h1 : x1 < y1)
  (h2 : x2 > y2)
  (h3 : x2 * y2 = x1 * y1 + 67)
  (h4 : 0 < x1 ∧ 0 < y1 ∧ 0 < x2 ∧ 0 < y2)
  : Nat.digits 10 (x1 * 1000 + y1 * 100 + x2 * 10 + y2) = [1, 9, 8, 5] :=
by
  sorry

end cartesian_coordinates_problem_l104_104413


namespace range_of_sine_l104_104046

theorem range_of_sine {x : ℝ} (h1 : 0 ≤ x) (h2 : x ≤ 2 * Real.pi) (h3 : Real.sin x ≥ Real.sqrt 2 / 2) :
  Real.pi / 4 ≤ x ∧ x ≤ 3 * Real.pi / 4 :=
by
  sorry

end range_of_sine_l104_104046


namespace oliver_total_money_l104_104738

-- Define the initial conditions
def initial_amount : ℕ := 9
def allowance_saved : ℕ := 5
def cost_frisbee : ℕ := 4
def cost_puzzle : ℕ := 3
def birthday_money : ℕ := 8

-- Define the problem statement in Lean
theorem oliver_total_money : 
  (initial_amount + allowance_saved - (cost_frisbee + cost_puzzle) + birthday_money) = 15 := 
by 
  sorry

end oliver_total_money_l104_104738


namespace cost_per_serving_l104_104436

def pasta_cost : ℝ := 1.0
def sauce_cost : ℝ := 2.0
def meatballs_cost : ℝ := 5.0
def total_servings : ℝ := 8.0

theorem cost_per_serving : (pasta_cost + sauce_cost + meatballs_cost) / total_servings = 1.0 :=
by sorry

end cost_per_serving_l104_104436


namespace radius_of_inner_circle_l104_104874

theorem radius_of_inner_circle (R a x : ℝ) (hR : 0 < R) (ha : 0 ≤ a) (haR : a < R) :
  (a ≠ R ∧ a ≠ 0) → x = (R^2 - a^2) / (2 * R) :=
by
  sorry

end radius_of_inner_circle_l104_104874


namespace triangle_with_angle_ratios_l104_104255

theorem triangle_with_angle_ratios {α β γ : ℝ} (h : α + β + γ = 180 ∧ (α / 2 = β / 3) ∧ (α / 2 = γ / 5)) : (α = 90 ∨ β = 90 ∨ γ = 90) :=
by
  sorry

end triangle_with_angle_ratios_l104_104255


namespace additional_rocks_needed_l104_104894

-- Define the dimensions of the garden
def length (garden : Type) : ℕ := 15
def width (garden : Type) : ℕ := 10
def rock_cover (rock : Type) : ℕ := 1

-- Define the number of rocks Mrs. Hilt has
def rocks_possessed (mrs_hilt : Type) : ℕ := 64

-- Define the perimeter of the garden
def perimeter (garden : Type) : ℕ :=
  2 * (length garden + width garden)

-- Define the number of rocks required for the first layer
def rocks_first_layer (garden : Type) : ℕ :=
  perimeter garden

-- Define the number of rocks required for the second layer (only longer sides)
def rocks_second_layer (garden : Type) : ℕ :=
  2 * length garden

-- Define the total number of rocks needed
def total_rocks_needed (garden : Type) : ℕ :=
  rocks_first_layer garden + rocks_second_layer garden

-- Prove the number of additional rocks Mrs. Hilt needs
theorem additional_rocks_needed (garden : Type) (mrs_hilt : Type):
  total_rocks_needed garden - rocks_possessed mrs_hilt = 16 := by
  sorry

end additional_rocks_needed_l104_104894


namespace derek_dogs_l104_104818

theorem derek_dogs (d c : ℕ) (h1 : d = 90) 
  (h2 : c = d / 3) 
  (h3 : c + 210 = 2 * (d + 120 - d)) : 
  d + 120 - d = 120 :=
by
  sorry

end derek_dogs_l104_104818


namespace johns_shirt_percentage_increase_l104_104265

variable (P S : ℕ)

theorem johns_shirt_percentage_increase :
  P = 50 →
  S + P = 130 →
  ((S - P) * 100 / P) = 60 := by
  sorry

end johns_shirt_percentage_increase_l104_104265


namespace expand_polynomial_l104_104040

theorem expand_polynomial :
  (x^2 - 3 * x + 3) * (x^2 + 3 * x + 1) = x^4 - 5 * x^2 + 6 * x + 3 :=
by
  sorry

end expand_polynomial_l104_104040


namespace minValue_expression_l104_104711

theorem minValue_expression (x y : ℝ) (h : x + 2 * y = 4) : ∃ (v : ℝ), v = 2^x + 4^y ∧ ∀ (a b : ℝ), a + 2 * b = 4 → 2^a + 4^b ≥ v :=
by 
  sorry

end minValue_expression_l104_104711


namespace room_width_correct_l104_104133

noncomputable def length_of_room : ℝ := 5
noncomputable def total_cost_of_paving : ℝ := 21375
noncomputable def cost_per_square_meter : ℝ := 900
noncomputable def width_of_room : ℝ := 4.75

theorem room_width_correct :
  (total_cost_of_paving / cost_per_square_meter) = (length_of_room * width_of_room) :=
by
  sorry

end room_width_correct_l104_104133


namespace smallest_whole_number_divisible_by_8_leaves_remainder_1_l104_104615

theorem smallest_whole_number_divisible_by_8_leaves_remainder_1 :
  ∃ (n : ℕ), n ≡ 1 [MOD 2] ∧ n ≡ 1 [MOD 3] ∧ n ≡ 1 [MOD 4] ∧ n ≡ 1 [MOD 5] ∧ n ≡ 1 [MOD 7] ∧ n % 8 = 0 ∧ n = 7141 :=
by
  sorry

end smallest_whole_number_divisible_by_8_leaves_remainder_1_l104_104615


namespace no_solution_lines_parallel_l104_104676

theorem no_solution_lines_parallel (m : ℝ) :
  (∀ t s : ℝ, (1 + 5 * t = 4 - 2 * s) ∧ (-3 + 2 * t = 1 + m * s) → false) ↔ m = -4 / 5 :=
by
  sorry

end no_solution_lines_parallel_l104_104676


namespace simplify_expression_l104_104970

theorem simplify_expression : 
  (2 * Real.sqrt 7) / (Real.sqrt 2 + Real.sqrt 3 - Real.sqrt 6) = 
  (2 * Real.sqrt 14 + 8 * Real.sqrt 21 + 2 * Real.sqrt 42 + 8 * Real.sqrt 63) / 23 :=
by
  sorry

end simplify_expression_l104_104970


namespace evaluate_expression_l104_104370

theorem evaluate_expression :
  (827 * 827) - ((827 - 1) * (827 + 1)) = 1 :=
sorry

end evaluate_expression_l104_104370


namespace joe_paint_fraction_l104_104096

theorem joe_paint_fraction :
  let total_paint := 360
  let fraction_first_week := 1 / 9
  let used_first_week := (fraction_first_week * total_paint)
  let remaining_after_first_week := total_paint - used_first_week
  let total_used := 104
  let used_second_week := total_used - used_first_week
  let fraction_second_week := used_second_week / remaining_after_first_week
  fraction_second_week = 1 / 5 :=
by
  sorry

end joe_paint_fraction_l104_104096


namespace hyperbola_problem_l104_104788

theorem hyperbola_problem (s : ℝ) :
    (∃ b > 0, ∀ (x y : ℝ), (x, y) = (-4, 5) → (x^2 / 9) - (y^2 / b^2) = 1)
    ∧ (∀ (x y : ℝ), (x, y) = (-3, 0) → (x^2 / 9) - (y^2 / b^2) = 1)
    ∧ (∀ b > 0, (x, y) = (s, 3) → (x^2 / 9) - (7 * y^2 / 225) = 1)
    → s^2 = (288 / 25) :=
by
  sorry

end hyperbola_problem_l104_104788


namespace range_of_m_if_solution_set_empty_solve_inequality_y_geq_m_l104_104539

noncomputable def quadratic_function (m x : ℝ) : ℝ :=
  (m + 1) * x^2 - m * x + m - 1

-- Part 1
theorem range_of_m_if_solution_set_empty (m : ℝ) :
  (∀ x : ℝ, quadratic_function m x < 0 → false) ↔ m ≥ 2 * Real.sqrt 3 / 3 := sorry

-- Part 2
theorem solve_inequality_y_geq_m (m x : ℝ) (h : m > -2) :
  (quadratic_function m x ≥ m) ↔ 
  (m = -1 → x ≥ 1) ∧
  (m > -1 → x ≤ -1/(m+1) ∨ x ≥ 1) ∧
  (m > -2 ∧ m < -1 → 1 ≤ x ∧ x ≤ -1/(m+1)) := sorry

end range_of_m_if_solution_set_empty_solve_inequality_y_geq_m_l104_104539


namespace players_count_l104_104168

theorem players_count (total_socks : ℕ) (socks_per_player : ℕ) (h1 : total_socks = 16) (h2 : socks_per_player = 2) :
  total_socks / socks_per_player = 8 :=
by
  sorry

end players_count_l104_104168


namespace lcm_36_105_l104_104221

noncomputable def factorize_36 : List (ℕ × ℕ) := [(2, 2), (3, 2)]
noncomputable def factorize_105 : List (ℕ × ℕ) := [(3, 1), (5, 1), (7, 1)]

theorem lcm_36_105 : Nat.lcm 36 105 = 1260 :=
by
  have h_36 : 36 = 2^2 * 3^2 := by norm_num
  have h_105 : 105 = 3^1 * 5^1 * 7^1 := by norm_num
  sorry

end lcm_36_105_l104_104221


namespace point_on_parabola_distance_to_directrix_is_4_l104_104869

noncomputable def distance_from_point_to_directrix (x y : ℝ) (directrix : ℝ) : ℝ :=
  abs (x - directrix)

def parabola (t : ℝ) : ℝ × ℝ :=
  (4 * t^2, 4 * t)

theorem point_on_parabola_distance_to_directrix_is_4 (m : ℝ) (t : ℝ) :
  parabola t = (3, m) → distance_from_point_to_directrix 3 m (-1) = 4 :=
by
  sorry

end point_on_parabola_distance_to_directrix_is_4_l104_104869


namespace increase_factor_l104_104319

noncomputable def old_plates : ℕ := 26 * 10^3
noncomputable def new_plates : ℕ := 26^4 * 10^4
theorem increase_factor : (new_plates / old_plates) = 175760 := by
  sorry

end increase_factor_l104_104319


namespace shark_sightings_l104_104177

theorem shark_sightings (x : ℕ) 
  (h1 : 26 = 5 + 3 * x) : x = 7 :=
by
  sorry

end shark_sightings_l104_104177


namespace round_trip_time_l104_104594

def boat_speed_still_water : ℝ := 16
def stream_speed : ℝ := 2
def distance_to_place : ℝ := 7560

theorem round_trip_time : (distance_to_place / (boat_speed_still_water + stream_speed) + distance_to_place / (boat_speed_still_water - stream_speed)) = 960 := by
  sorry

end round_trip_time_l104_104594


namespace larger_number_is_588_l104_104301

theorem larger_number_is_588
  (A B hcf : ℕ)
  (lcm_factors : ℕ × ℕ)
  (hcf_condition : hcf = 42)
  (lcm_factors_condition : lcm_factors = (12, 14))
  (hcf_prop : Nat.gcd A B = hcf)
  (lcm_prop : Nat.lcm A B = hcf * lcm_factors.1 * lcm_factors.2) :
  max (A) (B) = 588 :=
by
  sorry

end larger_number_is_588_l104_104301


namespace polygon_interior_angles_l104_104141

theorem polygon_interior_angles {n : ℕ} (h : (n - 2) * 180 = 900) : n = 7 :=
by
  sorry

end polygon_interior_angles_l104_104141


namespace height_of_table_l104_104466

variable (h l w h3 : ℝ)

-- Conditions from the problem
def condition1 : Prop := h3 = 4
def configurationA : Prop := l + h - w = 50
def configurationB : Prop := w + h + h3 - l = 44

-- Statement to prove
theorem height_of_table (h l w h3 : ℝ) 
  (cond1 : condition1 h3)
  (confA : configurationA h l w)
  (confB : configurationB h l w h3) : 
  h = 45 := 
by 
  sorry

end height_of_table_l104_104466


namespace susan_walked_9_miles_l104_104038

theorem susan_walked_9_miles (E S : ℕ) (h1 : E + S = 15) (h2 : E = S - 3) : S = 9 :=
by
  sorry

end susan_walked_9_miles_l104_104038


namespace price_of_each_shirt_l104_104266

-- Defining the conditions
def total_pants_cost (pants_price : ℕ) (num_pants : ℕ) := num_pants * pants_price
def total_amount_spent (amount_given : ℕ) (change_received : ℕ) := amount_given - change_received
def total_shirts_cost (amount_spent : ℕ) (pants_cost : ℕ) := amount_spent - pants_cost
def price_per_shirt (shirts_total_cost : ℕ) (num_shirts : ℕ) := shirts_total_cost / num_shirts

-- The main statement
theorem price_of_each_shirt (pants_price num_pants amount_given change_received num_shirts : ℕ) :
  num_pants = 2 →
  pants_price = 54 →
  amount_given = 250 →
  change_received = 10 →
  num_shirts = 4 →
  price_per_shirt (total_shirts_cost (total_amount_spent amount_given change_received) 
                   (total_pants_cost pants_price num_pants)) num_shirts = 33
:= by
  sorry

end price_of_each_shirt_l104_104266


namespace sin_of_7pi_over_6_l104_104206

theorem sin_of_7pi_over_6 : Real.sin (7 * Real.pi / 6) = -1 / 2 :=
by
  sorry

end sin_of_7pi_over_6_l104_104206


namespace find_x_l104_104420

variable {a b x r : ℝ}
variable (h₀ : 0 < a)
variable (h₁ : 0 < b)
variable (h₂ : r = (4 * a)^(2 * b))
variable (h₃ : r = (a^b * x^b)^2)
variable (h₄ : 0 < x)

theorem find_x : x = 4 := by
  sorry

end find_x_l104_104420


namespace smallest_nat_divisible_by_48_squared_l104_104831

theorem smallest_nat_divisible_by_48_squared :
  ∃ n : ℕ, (n % (48^2) = 0) ∧ 
           (∀ (d : ℕ), d ∈ (Nat.digits n 10) → d = 0 ∨ d = 1) ∧ 
           (n = 11111111100000000) := sorry

end smallest_nat_divisible_by_48_squared_l104_104831


namespace monotonicity_f_geq_f_neg_l104_104990

noncomputable def f (a x : ℝ) : ℝ := Real.exp x - a * x - 1

theorem monotonicity (a : ℝ) :
  (a ≤ 0 → ∀ x1 x2 : ℝ, x1 ≤ x2 → f a x1 ≤ f a x2) ∧
  (a > 0 →
    (∀ x1 x2 : ℝ, x1 > Real.log a → x2 > Real.log a → x1 ≤ x2 → f a x1 ≤ f a x2) ∧
    (∀ x1 x2 : ℝ, x1 < Real.log a → x2 < Real.log a → x1 ≤ x2 → f a x1 ≤ f a x2)) :=
by sorry

theorem f_geq_f_neg (x : ℝ) (hx : x ≥ 0) : f 1 x ≥ f 1 (-x) :=
by sorry

end monotonicity_f_geq_f_neg_l104_104990


namespace odd_periodic_function_value_l104_104581

theorem odd_periodic_function_value
  (f : ℝ → ℝ)
  (odd_f : ∀ x, f (-x) = - f x)
  (periodic_f : ∀ x, f (x + 3) = f x)
  (bounded_f : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = 2 * x) :
  f 8.5 = -1 :=
sorry

end odd_periodic_function_value_l104_104581


namespace simplify_expression_l104_104578

theorem simplify_expression :
  (210 / 18) * (6 / 150) * (9 / 4) = 21 / 20 :=
by
  sorry

end simplify_expression_l104_104578


namespace triangle_side_length_b_l104_104406

theorem triangle_side_length_b (a b c : ℝ) (A B C : ℝ)
  (hB : B = 30) 
  (h_area : 1/2 * a * c * Real.sin (B * Real.pi/180) = 3/2) 
  (h_sine : Real.sin (A * Real.pi/180) + Real.sin (C * Real.pi/180) = 2 * Real.sin (B * Real.pi/180)) :
  b = Real.sqrt 3 + 1 :=
by
  sorry

end triangle_side_length_b_l104_104406


namespace trig_identity_on_line_l104_104714

theorem trig_identity_on_line (α : ℝ) (h : Real.tan α = 2) :
  Real.sin α ^ 2 - Real.cos α ^ 2 + Real.sin α * Real.cos α = 1 :=
sorry

end trig_identity_on_line_l104_104714


namespace num_sides_polygon_l104_104599

theorem num_sides_polygon (n : ℕ) (h : (n - 2) * 180 = 1260) : n = 9 := by
  sorry

end num_sides_polygon_l104_104599


namespace inequality_has_solutions_iff_a_ge_4_l104_104872

theorem inequality_has_solutions_iff_a_ge_4 (a x : ℝ) :
  (∃ x : ℝ, |x + 1| + |x - 3| ≤ a) ↔ a ≥ 4 :=
sorry

end inequality_has_solutions_iff_a_ge_4_l104_104872


namespace solve_for_a_l104_104031

noncomputable def special_otimes (a b : ℝ) : ℝ :=
  if a > b then a^2 + b else a + b^2

theorem solve_for_a (a : ℝ) : special_otimes a (-2) = 4 → a = Real.sqrt 6 :=
by
  intro h
  sorry

end solve_for_a_l104_104031


namespace rem_frac_l104_104193

def rem (x y : ℚ) : ℚ := x - y * (⌊x / y⌋ : ℤ)

theorem rem_frac : rem (7 / 12) (-3 / 4) = -1 / 6 :=
by
  sorry

end rem_frac_l104_104193


namespace max_distinct_fans_l104_104645

-- Define the problem conditions and the main statement
theorem max_distinct_fans : 
  let n := 6  -- Number of sectors per fan
  let configurations := 2^n  -- Total configurations without considering symmetry
  let unchanged_flips := 8  -- Number of configurations unchanged by flipping
  let distinct_configurations := (configurations - unchanged_flips) / 2 + unchanged_flips 
  in 
  distinct_configurations = 36 := by sorry  # We state the answer based on the provided steps


end max_distinct_fans_l104_104645


namespace radii_touching_circles_l104_104923

noncomputable def radius_of_circles_touching_unit_circles 
  (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] 
  (centerA centerB centerC : A) 
  (unit_radius : ℝ) (h1 : dist centerA centerB = 2 * unit_radius) 
  (h2 : dist centerB centerC = 2 * unit_radius) (h3 : dist centerC centerA = 2 * unit_radius) 
  : Prop :=
  ∃ r₁ r₂ : ℝ, r₁ = 1/3 ∧ r₂ = 7/3

theorem radii_touching_circles (A B C : Type)
  [MetricSpace A] [MetricSpace B] [MetricSpace C]
  (centerA centerB centerC : A)
  (unit_radius : ℝ) (h1 : dist centerA centerB = 2 * unit_radius)
  (h2 : dist centerB centerC = 2 * unit_radius) (h3 : dist centerC centerA = 2 * unit_radius)
  : radius_of_circles_touching_unit_circles A B C centerA centerB centerC unit_radius h1 h2 h3 :=
sorry

end radii_touching_circles_l104_104923


namespace scarves_per_yarn_correct_l104_104281

def scarves_per_yarn (total_yarns total_scarves : ℕ) : ℕ :=
  total_scarves / total_yarns

theorem scarves_per_yarn_correct :
  scarves_per_yarn (2 + 6 + 4) 36 = 3 :=
by
  sorry

end scarves_per_yarn_correct_l104_104281


namespace max_distinct_fans_l104_104640

-- Definitions related to the problem conditions
def sectors := 6
def initial_configurations := 2 ^ sectors
def symmetrical_configurations := 8
def distinct_configurations := (initial_configurations - symmetrical_configurations) / 2 + symmetrical_configurations

-- The theorem to prove
theorem max_distinct_fans : distinct_configurations = 36 := by
  sorry

end max_distinct_fans_l104_104640


namespace program_output_is_201_l104_104402

theorem program_output_is_201 :
  ∃ x S n, x = 3 + 2 * n ∧ S = n^2 + 4 * n ∧ S ≥ 10000 ∧ x = 201 :=
by
  sorry

end program_output_is_201_l104_104402


namespace find_number_l104_104622

theorem find_number (x : ℝ) (h : x / 5 = 30 + x / 6) : x = 900 :=
  sorry

end find_number_l104_104622


namespace divisible_iff_l104_104846

-- Definitions from the conditions
def a : ℕ → ℕ
  | 0     => 0
  | 1     => 1
  | (n+2) => 2 * a (n + 1) + a n

-- Main theorem statement.
theorem divisible_iff (n k : ℕ) : 2^k ∣ a n ↔ 2^k ∣ n := by
  sorry

end divisible_iff_l104_104846


namespace xy_addition_equals_13_l104_104075

theorem xy_addition_equals_13 (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hx_lt_15 : x < 15) (hy_lt_15 : y < 15) (hxy : x + y + x * y = 49) : x + y = 13 :=
by
  sorry

end xy_addition_equals_13_l104_104075


namespace sqrt_74_between_8_and_9_product_of_consecutive_integers_l104_104605

theorem sqrt_74_between_8_and_9 : 8 < Real.sqrt 74 ∧ Real.sqrt 74 < 9 := sorry

theorem product_of_consecutive_integers (h : 8 < Real.sqrt 74 ∧ Real.sqrt 74 < 9) : 8 * 9 = 72 := by
  have h1 : 8 < Real.sqrt 74 := And.left h
  have h2 : Real.sqrt 74 < 9 := And.right h
  calc
    8 * 9 = 72 := by norm_num

end sqrt_74_between_8_and_9_product_of_consecutive_integers_l104_104605


namespace max_distinct_fans_l104_104641

-- Definitions related to the problem conditions
def sectors := 6
def initial_configurations := 2 ^ sectors
def symmetrical_configurations := 8
def distinct_configurations := (initial_configurations - symmetrical_configurations) / 2 + symmetrical_configurations

-- The theorem to prove
theorem max_distinct_fans : distinct_configurations = 36 := by
  sorry

end max_distinct_fans_l104_104641


namespace smallest_four_digit_palindrome_div_by_3_with_odd_first_digit_l104_104147

theorem smallest_four_digit_palindrome_div_by_3_with_odd_first_digit :
  ∃ (n : ℕ), (∃ A B : ℕ, n = 1001 * A + 110 * B ∧ 1 ≤ A ∧ A < 10 ∧ 0 ≤ B ∧ B < 10 ∧ A % 2 = 1) ∧ 3 ∣ n ∧ n = 1221 :=
by
  sorry

end smallest_four_digit_palindrome_div_by_3_with_odd_first_digit_l104_104147


namespace extinction_prob_one_l104_104795

-- Define the probabilities
def p : ℝ := 0.6
def q : ℝ := 0.4

-- Define the extinction probability function
def extinction_prob (v : ℕ → ℝ) (k : ℕ) : ℝ :=
  if k = 0 then 1
  else p * v (k + 1) + q * v (k - 1)

-- State the theorem
theorem extinction_prob_one (v : ℕ → ℝ) :
  extinction_prob v 1 = 2 / 3 :=
sorry

end extinction_prob_one_l104_104795


namespace parabola_axis_of_symmetry_l104_104404

theorem parabola_axis_of_symmetry (a b : ℝ) (h : a ≠ 0) (hx : (a * -2 + b) = 0) : 
  (y = ax^2 + bx).axis_symmetry = -1 :=
by
  sorry

end parabola_axis_of_symmetry_l104_104404


namespace arithmetic_sequence_problem_l104_104261

theorem arithmetic_sequence_problem
  (a : ℕ → ℝ) (d : ℝ)
  (h1 : ∀ n, a (n + 1) = a n + d)
  (h2 : a 1 + 3 * a 8 + a 15 = 120)
  : 2 * a 9 - a 10 = 24 := by
  sorry

end arithmetic_sequence_problem_l104_104261


namespace sin_of_7pi_over_6_l104_104205

theorem sin_of_7pi_over_6 : Real.sin (7 * Real.pi / 6) = -1 / 2 :=
by
  sorry

end sin_of_7pi_over_6_l104_104205


namespace hurricane_damage_in_euros_l104_104009

-- Define the conditions
def usd_damage : ℝ := 45000000  -- Damage in US dollars
def exchange_rate : ℝ := 0.9    -- Exchange rate from US dollars to Euros

-- Define the target value in Euros
def eur_damage : ℝ := 40500000  -- Expected damage in Euros

-- The theorem to prove
theorem hurricane_damage_in_euros :
  usd_damage * exchange_rate = eur_damage :=
by
  sorry

end hurricane_damage_in_euros_l104_104009


namespace two_circles_tangent_internally_l104_104997

-- Define radii and distance between centers
def R : ℝ := 7
def r : ℝ := 4
def distance_centers : ℝ := 3

-- Statement of the problem
theorem two_circles_tangent_internally :
  distance_centers = R - r → 
  -- Positional relationship: tangent internally
  (distance_centers = abs (R - r)) :=
sorry

end two_circles_tangent_internally_l104_104997


namespace minimum_tan_theta_is_sqrt7_l104_104175

noncomputable def min_tan_theta (z : ℂ) : ℝ := (Complex.abs (Complex.im z) / Complex.abs (Complex.re z))

theorem minimum_tan_theta_is_sqrt7 {z : ℂ} 
  (hz_real : 0 ≤ Complex.re z)
  (hz_imag : 0 ≤ Complex.im z)
  (hz_condition : Complex.abs (z^2 + 2) ≤ Complex.abs z) :
  min_tan_theta z = Real.sqrt 7 := sorry

end minimum_tan_theta_is_sqrt7_l104_104175


namespace cookie_baking_time_l104_104036

theorem cookie_baking_time 
  (total_time : ℕ) 
  (white_icing_time: ℕ)
  (chocolate_icing_time: ℕ) 
  (total_icing_time : white_icing_time + chocolate_icing_time = 60)
  (total_cooking_time : total_time = 120):

  (total_time - (white_icing_time + chocolate_icing_time) = 60) :=
by
  sorry

end cookie_baking_time_l104_104036


namespace inverse_proportionality_example_l104_104582

theorem inverse_proportionality_example (k : ℝ) (x : ℝ) (y : ℝ) (h1 : 5 * 10 = k) (h2 : x * 40 = k) : x = 5 / 4 :=
by
  -- sorry is used to skip the proof.
  sorry

end inverse_proportionality_example_l104_104582


namespace ball_hits_ground_at_t_l104_104299

theorem ball_hits_ground_at_t (t : ℝ) : 
  (∃ t, -8 * t^2 - 12 * t + 64 = 0 ∧ 0 ≤ t) → t = 2 :=
by
  sorry

end ball_hits_ground_at_t_l104_104299


namespace correct_statements_l104_104300

/-- The line (3+m)x+4y-3+3m=0 (m ∈ ℝ) always passes through the fixed point (-3, 3) -/
def statement1 (m : ℝ) : Prop :=
  ∀ x y : ℝ, (3 + m) * x + 4 * y - 3 + 3 * m = 0 → (x = -3 ∧ y = 3)

/-- For segment AB with endpoint B at (3,4) and A moving on the circle x²+y²=4,
    the trajectory equation of the midpoint M of segment AB is (x - 3/2)²+(y - 2)²=1 -/
def statement2 : Prop :=
  ∀ x y x1 y1 : ℝ, ((x1, y1) : ℝ × ℝ) ∈ {p | p.1^2 + p.2^2 = 4} → x = (x1 + 3) / 2 → y = (y1 + 4) / 2 → 
    (x - 3 / 2)^2 + (y - 2)^2 = 1

/-- Given M = {(x, y) | y = √(1 - x²)} and N = {(x, y) | y = x + b},
    if M ∩ N ≠ ∅, then b ∈ [-√2, √2] -/
def statement3 (b : ℝ) : Prop :=
  ∃ x y : ℝ, y = Real.sqrt (1 - x^2) ∧ y = x + b → b ∈ [-Real.sqrt 2, Real.sqrt 2]

/-- Given the circle C: (x - b)² + (y - c)² = a² (a > 0, b > 0, c > 0) intersects the x-axis and is
    separate from the y-axis, then the intersection point of the line ax + by + c = 0 and the line
    x + y + 1 = 0 is in the second quadrant -/
def statement4 (a b c : ℝ) : Prop :=
  a > 0 → b > 0 → c > 0 → b > a → a > c →
  ∃ x y : ℝ, (a * x + b * y + c = 0 ∧ x + y + 1 = 0) ∧ x < 0 ∧ y > 0

/-- Among the statements, the correct ones are 1, 2, and 4 -/
theorem correct_statements : 
  (∀ m : ℝ, statement1 m) ∧ statement2 ∧ (∀ b : ℝ, ¬ statement3 b) ∧ 
  (∀ a b c : ℝ, statement4 a b c) :=
by sorry

end correct_statements_l104_104300


namespace find_ordered_pairs_l104_104972

theorem find_ordered_pairs (a b : ℕ) (h1 : 2 * a + 1 ∣ 3 * b - 1) (h2 : 2 * b + 1 ∣ 3 * a - 1) : 
  (a = 2 ∧ b = 2) ∨ (a = 12 ∧ b = 17) ∨ (a = 17 ∧ b = 12) :=
by {
  sorry -- proof omitted
}

end find_ordered_pairs_l104_104972


namespace probability_of_intersection_in_decagon_is_fraction_l104_104317

open_locale big_operators

noncomputable def probability_intersecting_diagonals_in_decagon : ℚ :=
let num_points := 10 in
let diagonals := (num_points.choose 2) - num_points in
let total_diagonal_pairs := (diagonals.choose 2) in
let valid_intersections := (num_points.choose 4) in
valid_intersections / total_diagonal_pairs

theorem probability_of_intersection_in_decagon_is_fraction :
  probability_intersecting_diagonals_in_decagon = 42 / 119 :=
by {
  unfold probability_intersecting_diagonals_in_decagon,
  sorry
}

end probability_of_intersection_in_decagon_is_fraction_l104_104317


namespace inequality_solution_set_l104_104764

theorem inequality_solution_set (a : ℝ) : (-16 < a ∧ a ≤ 0) ↔ (∀ x : ℝ, a * x^2 + a * x - 4 < 0) :=
by
  sorry

end inequality_solution_set_l104_104764


namespace max_equilateral_triangles_l104_104981

theorem max_equilateral_triangles (length : ℕ) (n : ℕ) (segments : ℕ) : 
  (length = 2) → (segments = 6) → (∀ t, 1 ≤ t ∧ t ≤ 4 → t = 4) :=
by 
  intros length_eq segments_eq h
  sorry

end max_equilateral_triangles_l104_104981


namespace fraction_multiplication_l104_104321

theorem fraction_multiplication :
  (1 / 3 : ℚ) * (1 / 2) * (3 / 4) * (5 / 6) = 5 / 48 := by
  sorry

end fraction_multiplication_l104_104321


namespace determine_n_l104_104610

-- Constants and variables
variables {a : ℕ → ℝ} {n : ℕ}

-- Definition for the condition at each vertex
def vertex_condition (a : ℕ → ℝ) (i : ℕ) : Prop :=
  a i = a (i - 1) * a (i + 1)

-- Mathematical problem statement
theorem determine_n (h : ∀ i, vertex_condition a i) (distinct_a : ∀ i j, a i ≠ a j) : n = 6 :=
sorry

end determine_n_l104_104610


namespace area_ratio_S_T_l104_104270

open Set

def T : Set (ℝ × ℝ × ℝ) := {p | let (x, y, z) := p; x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ x + y + z = 1}

def supports (p q : ℝ × ℝ × ℝ) : Prop :=
  let (x, y, z) := p
  let (a, b, c) := q
  (x ≥ a ∧ y ≥ b) ∨ (x ≥ a ∧ z ≥ c) ∨ (y ≥ b ∧ z ≥ c)

def S : Set (ℝ × ℝ × ℝ) := {p ∈ T | supports p (1/4, 1/4, 1/2)}

theorem area_ratio_S_T : ∃ k : ℝ, k = 3 / 4 ∧
  ∃ (area_T area_S : ℝ), area_T ≠ 0 ∧ (area_S / area_T = k) := sorry

end area_ratio_S_T_l104_104270


namespace exist_positive_m_l104_104418

theorem exist_positive_m {n p q : ℕ} (hn_pos : 0 < n) (hp_prime : Prime p) (hq_prime : Prime q) 
  (h1 : pq ∣ n ^ p + 2) (h2 : n + 2 ∣ n ^ p + q ^ p) : ∃ m : ℕ, q ∣ 4 ^ m * n + 2 := 
sorry

end exist_positive_m_l104_104418


namespace intersection_is_interval_l104_104993

-- Let M be the set of numbers where the domain of the function y = log x is defined.
def M : Set ℝ := {x | 0 < x}

-- Let N be the set of numbers where x^2 - 4 > 0.
def N : Set ℝ := {x | x^2 - 4 > 0}

-- The complement of N in the real numbers ℝ.
def complement_N : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

-- We need to prove that the intersection of M and the complement of N is the interval (0, 2].
theorem intersection_is_interval : (M ∩ complement_N) = {x | 0 < x ∧ x ≤ 2} := 
by 
  sorry

end intersection_is_interval_l104_104993


namespace cupcakes_difference_l104_104348

theorem cupcakes_difference 
    (B_hrly_rate : ℕ) 
    (D_hrly_rate : ℕ) 
    (B_break : ℕ) 
    (total_hours : ℕ) 
    (B_hrly_rate = 10) 
    (D_hrly_rate = 8) 
    (B_break = 2) 
    (total_hours = 5) :
    (D_hrly_rate * total_hours) - (B_hrly_rate * (total_hours - B_break)) = 10 := 
by sorry

end cupcakes_difference_l104_104348


namespace minimum_value_of_fraction_l104_104982

theorem minimum_value_of_fraction {x : ℝ} (hx : x ≥ 3/2) :
  ∃ y : ℝ, y = (2 * (x - 1) + (1 / (x - 1)) + 2) ∧ y = 2 * Real.sqrt 2 + 2 :=
sorry

end minimum_value_of_fraction_l104_104982


namespace cupcake_difference_l104_104349

def betty_rate : ℕ := 10
def dora_rate : ℕ := 8
def total_hours : ℕ := 5
def betty_break_hours : ℕ := 2

theorem cupcake_difference :
  (dora_rate * total_hours) - (betty_rate * (total_hours - betty_break_hours)) = 10 :=
by
  sorry

end cupcake_difference_l104_104349


namespace distance_between_stations_l104_104634

theorem distance_between_stations :
  ∀ (x t : ℕ), 
    (20 * t = x) ∧ 
    (25 * t = x + 70) →
    (2 * x + 70 = 630) :=
by
  sorry

end distance_between_stations_l104_104634


namespace area_of_triangle_ABC_l104_104896

open Real

-- Defining the conditions as per the problem
def triangle_side_equality (AB AC : ℝ) : Prop := AB = AC
def angle_relation (angleBAC angleBTC : ℝ) : Prop := angleBAC = 2 * angleBTC
def side_length_BT (BT : ℝ) : Prop := BT = 70
def side_length_AT (AT : ℝ) : Prop := AT = 37

-- Proving the area of triangle ABC given the conditions
theorem area_of_triangle_ABC
  (AB AC : ℝ)
  (angleBAC angleBTC : ℝ)
  (BT AT : ℝ)
  (h1 : triangle_side_equality AB AC)
  (h2 : angle_relation angleBAC angleBTC)
  (h3 : side_length_BT BT)
  (h4 : side_length_AT AT) 
  : ∃ area : ℝ, area = 420 :=
sorry

end area_of_triangle_ABC_l104_104896


namespace triangle_shortest_side_l104_104087

theorem triangle_shortest_side (a b c : ℝ) (h : 0 < a) (h1 : 0 < b) (h2 : 0 < c) 
    (r : ℝ) (h3 : r = 5) 
    (h4 : a = 4) (h5 : b = 10)
    (circumcircle_tangent_property : 2 * (4 + 10) * r = 30) :
  min a (min b c) = 30 :=
by 
  sorry

end triangle_shortest_side_l104_104087


namespace wings_count_total_l104_104286

def number_of_wings (num_planes : Nat) (wings_per_plane : Nat) : Nat :=
  num_planes * wings_per_plane

theorem wings_count_total :
  number_of_wings 45 2 = 90 :=
  by
    sorry

end wings_count_total_l104_104286


namespace extinction_prob_one_l104_104794

-- Define the probabilities
def p : ℝ := 0.6
def q : ℝ := 0.4

-- Define the extinction probability function
def extinction_prob (v : ℕ → ℝ) (k : ℕ) : ℝ :=
  if k = 0 then 1
  else p * v (k + 1) + q * v (k - 1)

-- State the theorem
theorem extinction_prob_one (v : ℕ → ℝ) :
  extinction_prob v 1 = 2 / 3 :=
sorry

end extinction_prob_one_l104_104794


namespace resulting_total_mass_l104_104791

-- Define initial conditions
def initial_total_mass : ℝ := 12
def initial_white_paint_mass : ℝ := 0.8 * initial_total_mass
def initial_black_paint_mass : ℝ := initial_total_mass - initial_white_paint_mass

-- Required condition for the new mixture
def final_white_paint_percentage : ℝ := 0.9

-- Prove that the resulting total mass of paint is 24 kg
theorem resulting_total_mass (x : ℝ) (h1 : initial_total_mass = 12) 
                            (h2 : initial_white_paint_mass = 0.8 * initial_total_mass)
                            (h3 : initial_black_paint_mass = initial_total_mass - initial_white_paint_mass)
                            (h4 : final_white_paint_percentage = 0.9) 
                            (h5 : (initial_white_paint_mass + x) / (initial_total_mass + x) = final_white_paint_percentage) : 
                            initial_total_mass + x = 24 :=
by 
  -- Temporarily assume the proof without detailing the solution steps
  sorry

end resulting_total_mass_l104_104791


namespace inequality_geq_l104_104732

theorem inequality_geq (t : ℝ) (n : ℕ) (ht : t ≥ 1/2) : 
  t^(2*n) ≥ (t-1)^(2*n) + (2*t-1)^n := 
sorry

end inequality_geq_l104_104732


namespace parabola_intersects_y_axis_l104_104390

theorem parabola_intersects_y_axis (m n : ℝ) :
  (∃ (x y : ℝ), y = x^2 + m * x + n ∧ 
  ((x = -1 ∧ y = -6) ∨ (x = 1 ∧ y = 0))) →
  (0, (-4)) = (0, n) :=
by
  sorry

end parabola_intersects_y_axis_l104_104390


namespace mod_6_computation_l104_104171

theorem mod_6_computation (a b n : ℕ) (h₁ : a ≡ 35 [MOD 6]) (h₂ : b ≡ 16 [MOD 6]) (h₃ : n = 1723) :
  (a ^ n - b ^ n) % 6 = 1 :=
by 
  -- proofs go here
  sorry

end mod_6_computation_l104_104171


namespace time_to_cross_same_direction_l104_104928

-- Defining the conditions
def speed_train1 : ℝ := 60 -- kmph
def speed_train2 : ℝ := 40 -- kmph
def time_opposite_directions : ℝ := 10.000000000000002 -- seconds 
def relative_speed_opposite_directions : ℝ := speed_train1 + speed_train2 -- 100 kmph
def relative_speed_same_direction : ℝ := speed_train1 - speed_train2 -- 20 kmph

-- Defining the proof statement
theorem time_to_cross_same_direction : 
  (time_opposite_directions * (relative_speed_opposite_directions / relative_speed_same_direction)) = 50 :=
by
  sorry

end time_to_cross_same_direction_l104_104928


namespace proof_problem_l104_104388

theorem proof_problem
  (x y z : ℝ)
  (hx : x > 0)
  (hy : y > 0)
  (hz : z > 0) :
  (x + 1 / y ≥ 2) ∨ (y + 1 / z ≥ 2) ∨ (z + 1 / x ≥ 2) :=
sorry

end proof_problem_l104_104388


namespace part1_part2_l104_104835

-- Part (1)
theorem part1 (m n : ℝ) (hm : |m| = 1) (hn : |n| = 4) (opposite : m * n < 0) :
  m + n = -3 ∨ m + n = 3 :=
sorry

-- Part (2)
theorem part2 (m n : ℝ) (hm : |m| = 1) (hn : |n| = 4) :
  (m - n) ≤ 5 :=
sorry

end part1_part2_l104_104835


namespace extinction_prob_l104_104798

-- Define the probabilities
def p : ℝ := 0.6
def q : ℝ := 0.4

-- Define the base case condition
def v_0 : ℝ := 1

-- Define the extinction probability function
def v : ℕ → ℝ
| 0 => v_0
| k => p * v (k + 1) + q * v (k - 1)

-- Define the target value for v_1
def v_1 : ℝ := 2 / 3

-- The main theorem: Prove that v 1 = 2 / 3 given the conditions
theorem extinction_prob : v 1 = v_1 := by
  -- Proof will be provided here
  sorry

end extinction_prob_l104_104798


namespace weight_of_second_piece_of_wood_l104_104029

/--
Given: 
1) The density and thickness of the wood are uniform.
2) The first piece of wood is a square with a side length of 3 inches and a weight of 15 ounces.
3) The second piece of wood is a square with a side length of 6 inches.
Theorem: 
The weight of the second piece of wood is 60 ounces.
-/
theorem weight_of_second_piece_of_wood (s1 s2 w1 w2 : ℕ) (h1 : s1 = 3) (h2 : w1 = 15) (h3 : s2 = 6) :
  w2 = 60 :=
sorry

end weight_of_second_piece_of_wood_l104_104029


namespace time_A_to_complete_work_alone_l104_104483

theorem time_A_to_complete_work_alone :
  ∃ (x : ℝ), (1 / x) + (1 / 20) = (1 / 8.571428571428571) ∧ x = 15 :=
by
  sorry

end time_A_to_complete_work_alone_l104_104483


namespace domain_of_function_l104_104445

theorem domain_of_function :
  {x : ℝ | x ≥ -1 ∧ x ≠ 1 / 2} =
  {x : ℝ | 2 * x - 1 ≠ 0 ∧ x + 1 ≥ 0} :=
by {
  sorry
}

end domain_of_function_l104_104445


namespace find_k_for_line_l104_104382

theorem find_k_for_line (k : ℝ) : (2 * k * (-1/2) + 1 = -7 * 3) → k = 22 :=
by
  intro h
  sorry

end find_k_for_line_l104_104382


namespace derek_dogs_count_l104_104821

theorem derek_dogs_count
  (initial_dogs : ℕ)
  (initial_cars : ℕ)
  (cars_after_10_years : ℕ)
  (dogs_after_10_years : ℕ)
  (h1 : initial_dogs = 90)
  (h2 : initial_dogs = 3 * initial_cars)
  (h3 : cars_after_10_years = initial_cars + 210)
  (h4 : cars_after_10_years = 2 * dogs_after_10_years) :
  dogs_after_10_years = 120 :=
by
  sorry

end derek_dogs_count_l104_104821


namespace vector_perpendicular_sin_cos_l104_104072

open Real

theorem vector_perpendicular_sin_cos (θ : ℝ) (h1 : θ ∈ Ioo (π / 2) π) 
(h2 : sin θ + 2 * cos θ = 0) : sin θ - cos θ = 3 * sqrt 5 / 5 :=
by 
  sorry

end vector_perpendicular_sin_cos_l104_104072


namespace total_amount_after_refunds_and_discounts_l104_104810

-- Definitions
def individual_bookings : ℤ := 12000
def group_bookings_before_discount : ℤ := 16000
def discount_rate : ℕ := 10
def refund_individual_1 : ℤ := 500
def count_refund_individual_1 : ℕ := 3
def refund_individual_2 : ℤ := 300
def count_refund_individual_2 : ℕ := 2
def total_refund_group : ℤ := 800

-- Calculation proofs
theorem total_amount_after_refunds_and_discounts : 
(individual_bookings + (group_bookings_before_discount - (discount_rate * group_bookings_before_discount / 100))) - 
((count_refund_individual_1 * refund_individual_1) + (count_refund_individual_2 * refund_individual_2) + total_refund_group) = 23500 := by
    sorry

end total_amount_after_refunds_and_discounts_l104_104810


namespace problem_solution_l104_104251

theorem problem_solution : 
  (∃ (N : ℕ), (1 + 2 + 3) / 6 = (1988 + 1989 + 1990) / N) → ∃ (N : ℕ), N = 5967 :=
by
  intro h
  sorry

end problem_solution_l104_104251


namespace determine_real_pairs_l104_104510

theorem determine_real_pairs (a b : ℝ) :
  (∀ n : ℕ+, a * ⌊ b * n ⌋ = b * ⌊ a * n ⌋) →
  (∃ c : ℝ, (a = 0 ∧ b = c) ∨ (a = c ∧ b = 0) ∨ (a = c ∧ b = c) ∨ (∃ k l : ℤ, a = k ∧ b = l)) :=
by
  sorry

end determine_real_pairs_l104_104510


namespace emily_total_beads_l104_104037

theorem emily_total_beads (necklaces : ℕ) (beads_per_necklace : ℕ) (total_beads : ℕ) : 
  necklaces = 11 → 
  beads_per_necklace = 28 → 
  total_beads = necklaces * beads_per_necklace → 
  total_beads = 308 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  assumption

end emily_total_beads_l104_104037


namespace diff_in_set_l104_104485

variable (A : Set Int)
variable (ha : ∃ a ∈ A, a > 0)
variable (hb : ∃ b ∈ A, b < 0)
variable (h : ∀ {a b : Int}, a ∈ A → b ∈ A → (2 * a) ∈ A ∧ (a + b) ∈ A)

theorem diff_in_set (x y : Int) (hx : x ∈ A) (hy : y ∈ A) : (x - y) ∈ A :=
  sorry

end diff_in_set_l104_104485


namespace lcm_36_105_l104_104212

theorem lcm_36_105 : Int.lcm 36 105 = 1260 := by
  sorry

end lcm_36_105_l104_104212


namespace missed_bus_time_by_l104_104184

def bus_departure_time : Time := Time.mk 8 0 0
def travel_time_minutes : Int := 30
def departure_time_home : Time := Time.mk 7 50 0
def arrival_time_pickup_point : Time := 
  departure_time_home.addMinutes travel_time_minutes

theorem missed_bus_time_by :
  arrival_time_pickup_point.diff bus_departure_time = 20 * 60 :=
by
  sorry

end missed_bus_time_by_l104_104184


namespace ratio_first_part_l104_104786

theorem ratio_first_part (x : ℕ) (h1 : x / 3 = 2) : x = 6 :=
by
  sorry

end ratio_first_part_l104_104786


namespace probability_four_dice_show_two_l104_104144

theorem probability_four_dice_show_two :
  let roll_two := (1 / 8 : ℝ)
  let roll_not_two := (7 / 8 : ℝ)
  let choose := nat.choose 12 4
  let prob_specific_arrangement := roll_two^4 * roll_not_two^8
  let total_prob := choose * prob_specific_arrangement
  total_prob ≈ 0.091 :=
by {
  let roll_two := (1 / 8 : ℝ)
  let roll_not_two := (7 / 8 : ℝ)
  let choose := nat.choose 12 4
  let prob_specific_arrangement := roll_two^4 * roll_not_two^8
  let total_prob := choose * prob_specific_arrangement
  sorry  
}

end probability_four_dice_show_two_l104_104144


namespace find_ratio_of_hyperbola_l104_104051

noncomputable def hyperbola (x y a b : ℝ) := (x^2 / a^2) - (y^2 / b^2) = 1

theorem find_ratio_of_hyperbola (a b : ℝ) (h : a > b) 
  (h_asymptote_angle : ∀ α : ℝ, (y = ↑(b / a) * x -> α = 45)) :
  a / b = 1 :=
sorry

end find_ratio_of_hyperbola_l104_104051


namespace sides_of_polygon_l104_104602

-- Define the conditions
def polygon_sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

-- State the mathematically equivalent proof problem
theorem sides_of_polygon (n : ℕ) (h : polygon_sum_interior_angles n = 1260) : n = 9 := by
  sorry

end sides_of_polygon_l104_104602


namespace probability_not_integer_ratio_correct_l104_104054

open Finset

noncomputable def four_numbers := {1, 2, 3, 4}

def is_not_integer_ratio (a b : ℕ) : Prop :=
  a ∈ four_numbers ∧ b ∈ (four_numbers.erase a) ∧ ¬ (a % b = 0)

def probability_not_integer_ratio : ℚ :=
  let total_events := 12
  let successful_events := 8
  in successful_events / total_events

theorem probability_not_integer_ratio_correct :
  probability_not_integer_ratio = 2 / 3 :=
by
  sorry

end probability_not_integer_ratio_correct_l104_104054


namespace batsman_new_average_l104_104480

-- Let A be the average score before the 16th inning
def avg_before (A : ℝ) : Prop :=
  ∃ total_runs: ℝ, total_runs = 15 * A

-- Condition 1: The batsman makes 64 runs in the 16th inning
def score_in_16th_inning := 64

-- Condition 2: This increases his average by 3 runs
def avg_increase (A : ℝ) : Prop :=
  A + 3 = (15 * A + score_in_16th_inning) / 16

theorem batsman_new_average (A : ℝ) (h1 : avg_before A) (h2 : avg_increase A) :
  (A + 3) = 19 :=
sorry

end batsman_new_average_l104_104480


namespace avg_of_seven_consecutive_integers_starting_d_plus_1_is_c_plus_7_l104_104751

variable (c d : ℤ)
variable (h₁ : d = (c + (c + 1) + (c + 2) + (c + 3) + (c + 4) + (c + 5) + (c + 6)) / 7)

theorem avg_of_seven_consecutive_integers_starting_d_plus_1_is_c_plus_7 :
  (d + 1 + d + 2 + d + 3 + d + 4 + d + 5 + d + 6 + d + 7) / 7 = c + 7 :=
by
  sorry

end avg_of_seven_consecutive_integers_starting_d_plus_1_is_c_plus_7_l104_104751


namespace fraction_result_l104_104611

theorem fraction_result (x : ℚ) (h₁ : x * (3/4) = (1/6)) : (x - (1/12)) = (5/36) := 
sorry

end fraction_result_l104_104611


namespace total_number_of_vehicles_l104_104792

theorem total_number_of_vehicles 
  (lanes : ℕ) 
  (trucks_per_lane : ℕ) 
  (buses_per_lane : ℕ) 
  (cars_per_lane : ℕ := 2 * lanes * trucks_per_lane) 
  (motorcycles_per_lane : ℕ := 3 * buses_per_lane)
  (total_trucks : ℕ := lanes * trucks_per_lane)
  (total_cars : ℕ := lanes * cars_per_lane)
  (total_buses : ℕ := lanes * buses_per_lane)
  (total_motorcycles : ℕ := lanes * motorcycles_per_lane)
  (total_vehicles : ℕ := total_trucks + total_cars + total_buses + total_motorcycles)
  (hlanes : lanes = 4) 
  (htrucks : trucks_per_lane = 60) 
  (hbuses : buses_per_lane = 40) :
  total_vehicles = 2800 := sorry

end total_number_of_vehicles_l104_104792


namespace number_of_n_divisible_by_prime_lt_20_l104_104557

theorem number_of_n_divisible_by_prime_lt_20 (N : ℕ) : 
  (N = 69) :=
by
  sorry

end number_of_n_divisible_by_prime_lt_20_l104_104557


namespace difference_of_numbers_l104_104306

theorem difference_of_numbers (x y : ℝ) (h₁ : x + y = 25) (h₂ : x * y = 144) : |x - y| = 7 :=
sorry

end difference_of_numbers_l104_104306


namespace total_number_of_meetings_proof_l104_104927

-- Define the conditions in Lean
variable (A B : Type)
variable (starting_time : ℕ)
variable (location_A location_B : A × B)

-- Define speeds
variable (speed_A speed_B : ℕ)

-- Define meeting counts
variable (total_meetings : ℕ)

-- Define A reaches point B 2015 times
variable (A_reaches_B_2015 : Prop)

-- Define that B travels twice as fast as A
axiom speed_ratio : speed_B = 2 * speed_A

-- Define that A reaches point B for the 5th time when B reaches it for the 9th time
axiom meeting_times : A_reaches_B_2015 → (total_meetings = 6044)

-- The Lean statement to prove
theorem total_number_of_meetings_proof : A_reaches_B_2015 → total_meetings = 6044 := by
  sorry

end total_number_of_meetings_proof_l104_104927


namespace correct_calculation_l104_104476

theorem correct_calculation (x : ℤ) (h : 7 * (x + 24) / 5 = 70) :
  (5 * x + 24) / 7 = 22 :=
sorry

end correct_calculation_l104_104476


namespace sheets_in_a_bundle_l104_104024

variable (B : ℕ) -- Denotes the number of sheets in a bundle

-- Conditions
variable (NumBundles NumBunches NumHeaps : ℕ)
variable (SheetsPerBunch SheetsPerHeap TotalSheets : ℕ)

-- Definitions of given conditions
def numBundles := 3
def numBunches := 2
def numHeaps := 5
def sheetsPerBunch := 4
def sheetsPerHeap := 20
def totalSheets := 114

-- Theorem to prove
theorem sheets_in_a_bundle :
  3 * B + 2 * sheetsPerBunch + 5 * sheetsPerHeap = totalSheets → B = 2 := by
  intro h
  sorry

end sheets_in_a_bundle_l104_104024


namespace whipped_cream_needed_l104_104971

def total_days : ℕ := 15
def odd_days_count : ℕ := 8
def even_days_count : ℕ := 7

def pumpkin_pies_on_odd_days : ℕ := 3 * odd_days_count
def apple_pies_on_odd_days : ℕ := 2 * odd_days_count

def pumpkin_pies_on_even_days : ℕ := 2 * even_days_count
def apple_pies_on_even_days : ℕ := 4 * even_days_count

def total_pumpkin_pies_baked : ℕ := pumpkin_pies_on_odd_days + pumpkin_pies_on_even_days
def total_apple_pies_baked : ℕ := apple_pies_on_odd_days + apple_pies_on_even_days

def tiffany_pumpkin_pies_consumed : ℕ := 2
def tiffany_apple_pies_consumed : ℕ := 5

def remaining_pumpkin_pies : ℕ := total_pumpkin_pies_baked - tiffany_pumpkin_pies_consumed
def remaining_apple_pies : ℕ := total_apple_pies_baked - tiffany_apple_pies_consumed

def whipped_cream_for_pumpkin_pies : ℕ := 2 * remaining_pumpkin_pies
def whipped_cream_for_apple_pies : ℕ := remaining_apple_pies

def total_whipped_cream_needed : ℕ := whipped_cream_for_pumpkin_pies + whipped_cream_for_apple_pies

theorem whipped_cream_needed : total_whipped_cream_needed = 111 := by
  -- Proof omitted
  sorry

end whipped_cream_needed_l104_104971


namespace fifth_friend_payment_l104_104978

def contributions (a b c d e : ℕ) : Prop :=
  a + b + c + d + e = 120 ∧
  a = (1 / 3 : ℕ) * (b + c + d + e) ∧
  b = (1 / 4 : ℕ) * (a + c + d + e) ∧
  c = (1 / 5 : ℕ) * (a + b + d + e)

theorem fifth_friend_payment (a b c d e : ℕ) (h : contributions a b c d e) : e = 13 :=
sorry

end fifth_friend_payment_l104_104978


namespace f_even_l104_104866

noncomputable def f : ℝ → ℝ := sorry

axiom f_not_identically_zero : ∃ x : ℝ, f x ≠ 0

axiom f_functional_eqn : ∀ a b : ℝ, 
  f (a + b) + f (a - b) = 2 * f a + 2 * f b

theorem f_even (x : ℝ) : f (-x) = f x :=
  sorry

end f_even_l104_104866


namespace four_digit_solution_l104_104977

-- Definitions for the conditions.
def condition1 (u z x : ℕ) : Prop := u + z - 4 * x = 1
def condition2 (u z y : ℕ) : Prop := u + 10 * z - 2 * y = 14

-- The theorem to prove that the four-digit number xyz is either 1014, 2218, or 1932
theorem four_digit_solution (x y z u : ℕ) (h1 : condition1 u z x) (h2 : condition2 u z y) :
  (x = 1 ∧ y = 0 ∧ z = 1 ∧ u = 4) ∨
  (x = 2 ∧ y = 2 ∧ z = 1 ∧ u = 8) ∨
  (x = 1 ∧ y = 9 ∧ z = 3 ∧ u = 2) := 
sorry

end four_digit_solution_l104_104977


namespace total_pencils_correct_total_erasers_correct_total_rulers_correct_total_sharpeners_correct_total_skittles_correct_l104_104085

-- Given conditions
def kids_A := 7
def kids_B := 9
def kids_C := 5

def pencils_per_child_A := 4
def erasers_per_child_A := 2
def skittles_per_child_A := 13

def pencils_per_child_B := 6
def rulers_per_child_B := 1
def skittles_per_child_B := 8

def pencils_per_child_C := 3
def sharpeners_per_child_C := 1
def skittles_per_child_C := 15

-- Calculated totals
def total_pencils := kids_A * pencils_per_child_A + kids_B * pencils_per_child_B + kids_C * pencils_per_child_C
def total_erasers := kids_A * erasers_per_child_A
def total_rulers := kids_B * rulers_per_child_B
def total_sharpeners := kids_C * sharpeners_per_child_C
def total_skittles := kids_A * skittles_per_child_A + kids_B * skittles_per_child_B + kids_C * skittles_per_child_C

-- Proof obligations
theorem total_pencils_correct : total_pencils = 97 := by
  sorry

theorem total_erasers_correct : total_erasers = 14 := by
  sorry

theorem total_rulers_correct : total_rulers = 9 := by
  sorry

theorem total_sharpeners_correct : total_sharpeners = 5 := by
  sorry

theorem total_skittles_correct : total_skittles = 238 := by
  sorry

end total_pencils_correct_total_erasers_correct_total_rulers_correct_total_sharpeners_correct_total_skittles_correct_l104_104085


namespace determine_range_of_m_l104_104056

noncomputable def range_m (m : ℝ) (x : ℝ) : Prop :=
  ∀ x, (-x^2 + 7 * x + 8 < 0 → (x < 1 - 2 * |m| ∨ x > 1 + 2 * |m|)) ∧
       (∃ x, -x^2 + 7 * x + 8 ≥ 0)

theorem determine_range_of_m (m : ℝ) :
  (-1 ≤ m ∧ m ≤ 1) ↔
  (∀ x, (-x^2 + 7 * x + 8 < 0 → (x < 1 - 2 * |m| ∨ x > 1 + 2 * |m|)) ∧
         (∃ x, -x^2 + 7 * x + 8 ≥ 0)) :=
by
  sorry

end determine_range_of_m_l104_104056


namespace part1_l104_104649

theorem part1 (a b c d : ℤ) (h : a * d - b * c = 1) : Int.gcd (a + b) (c + d) = 1 :=
sorry

end part1_l104_104649


namespace fencing_required_l104_104163

theorem fencing_required (L W : ℝ) (h1 : L = 40) (h2 : L * W = 680) : 2 * W + L = 74 :=
by
  sorry

end fencing_required_l104_104163


namespace volume_of_region_l104_104606

noncomputable def volume_of_sphere (r : ℝ) : ℝ :=
  (4/3) * Real.pi * r^3

theorem volume_of_region (r_small r_large : ℝ) (h_small : r_small = 4) (h_large : r_large = 7) :
  volume_of_sphere r_large - volume_of_sphere r_small = 372 * Real.pi := by
  rw [h_small, h_large]
  sorry

end volume_of_region_l104_104606


namespace rice_field_sacks_l104_104086

theorem rice_field_sacks (x : ℝ)
  (h1 : ∀ x, x + 1.20 * x = 44) : x = 20 :=
sorry

end rice_field_sacks_l104_104086


namespace marble_choice_l104_104926

def numDifferentGroupsOfTwoMarbles (red green blue : ℕ) (yellow : ℕ) (orange : ℕ) : ℕ :=
  if (red = 1 ∧ green = 1 ∧ blue = 1 ∧ yellow = 2 ∧ orange = 2) then 12 else 0

theorem marble_choice:
  let red := 1
  let green := 1
  let blue := 1
  let yellow := 2
  let orange := 2
  numDifferentGroupsOfTwoMarbles red green blue yellow orange = 12 :=
by
  dsimp[numDifferentGroupsOfTwoMarbles]
  split_ifs
  · rfl
  · sorry

-- Ensure the theorem type matches the expected Lean 4 structure.
#print marble_choice

end marble_choice_l104_104926


namespace total_gallons_in_tanks_l104_104105

theorem total_gallons_in_tanks (
  tank1_cap : ℕ := 7000) (tank2_cap : ℕ := 5000) (tank3_cap : ℕ := 3000)
  (fill1_fraction : ℚ := 3/4) (fill2_fraction : ℚ := 4/5) (fill3_fraction : ℚ := 1/2)
  : tank1_cap * fill1_fraction + tank2_cap * fill2_fraction + tank3_cap * fill3_fraction = 10750 := by
  sorry

end total_gallons_in_tanks_l104_104105


namespace degenerate_ellipse_b_value_l104_104173

theorem degenerate_ellipse_b_value :
  ∃ b : ℝ, (∀ x y : ℝ, 3 * x^2 + y^2 + 6 * x - 6 * y + b = 0 → x = -1 ∧ y = 3) ↔ b = 12 :=
by
  sorry

end degenerate_ellipse_b_value_l104_104173


namespace find_math_marks_l104_104176

theorem find_math_marks
  (e p c b : ℕ)
  (n : ℕ)
  (a : ℚ)
  (M : ℕ) :
  e = 96 →
  p = 82 →
  c = 87 →
  b = 92 →
  n = 5 →
  a = 90.4 →
  (a * n = (e + p + c + b + M)) →
  M = 95 :=
by intros
   sorry

end find_math_marks_l104_104176


namespace right_triangle_relation_l104_104258

theorem right_triangle_relation (a b c x : ℝ)
  (h : c^2 = a^2 + b^2)
  (altitude : a * b = c * x) :
  (1 / x^2) = (1 / a^2) + (1 / b^2) :=
sorry

end right_triangle_relation_l104_104258


namespace side_of_larger_square_l104_104494

theorem side_of_larger_square (s S : ℕ) (h₁ : s = 5) (h₂ : S^2 = 4 * s^2) : S = 10 := 
by sorry

end side_of_larger_square_l104_104494


namespace relationship_between_y1_y2_y3_l104_104447

-- Define the parabola equation and points
def parabola (x c : ℝ) : ℝ := 2 * (x - 1)^2 + c

-- Define the points
def point1 := -2
def point2 := 0
def point3 := 5 / 3

-- Define the y values at these points
def y1 (c : ℝ) := parabola point1 c
def y2 (c : ℝ) := parabola point2 c
def y3 (c : ℝ) := parabola point3 c

-- Proof statement
theorem relationship_between_y1_y2_y3 (c : ℝ) : 
  y1 c > y2 c ∧ y2 c > y3 c :=
sorry

end relationship_between_y1_y2_y3_l104_104447


namespace count_integers_modulo_l104_104859

theorem count_integers_modulo (n : ℕ) (h₁ : n < 500) (h₂ : n % 7 = 4) : (setOf (λ n, n > 0 ∧ n < 500 ∧ n % 7 = 4)).card = 71 :=
sorry

end count_integers_modulo_l104_104859


namespace circle_tangent_parabola_height_difference_l104_104948

theorem circle_tangent_parabola_height_difference
  (a b r : ℝ)
  (point_of_tangency_left : a ≠ 0)
  (points_of_tangency_on_parabola : (2 * a^2) = (2 * (-a)^2))
  (center_y_coordinate : ∃ c , c = b)
  (circle_equation_tangent_parabola : ∀ x, (x^2 + (2*x^2 - b)^2 = r^2))
  (quartic_double_root : ∀ x, (x = a ∨ x = -a) → (x^2 + (4 - 2*b)*x^2 + b^2 - r^2 = 0)) :
  b - 2 * a^2 = 2 :=
by
  sorry

end circle_tangent_parabola_height_difference_l104_104948


namespace a_m_power_m_divides_a_n_power_n_a1_does_not_divide_any_an_power_n_l104_104366

theorem a_m_power_m_divides_a_n_power_n:
  ∀ (a : ℕ → ℕ) (m : ℕ), (a 1).gcd (a 2) = 1 ∧ (∀ n, a (n + 2) = a (n + 1) * a n + 1) ∧ m > 1 → ∃ n > m, (a m) ^ m ∣ (a n) ^ n := by 
  sorry

theorem a1_does_not_divide_any_an_power_n:
  ∀ (a : ℕ → ℕ), (a 1).gcd (a 2) = 1 ∧ (∀ n, a (n + 2) = a (n + 1) * a n + 1) → ¬ ∃ n > 1, (a 1) ∣ (a n) ^ n := by
  sorry

end a_m_power_m_divides_a_n_power_n_a1_does_not_divide_any_an_power_n_l104_104366


namespace division_reciprocal_multiplication_l104_104827

theorem division_reciprocal_multiplication : (4 / (8 / 13 : ℚ)) = (13 / 2 : ℚ) := 
by
  sorry

end division_reciprocal_multiplication_l104_104827


namespace loss_percentage_grinder_l104_104416

-- Conditions
def CP_grinder : ℝ := 15000
def CP_mobile : ℝ := 8000
def profit_mobile : ℝ := 0.10
def total_profit : ℝ := 200

-- Theorem to prove the loss percentage on the grinder
theorem loss_percentage_grinder : 
  ( (CP_grinder - (23200 - (CP_mobile * (1 + profit_mobile)))) / CP_grinder ) * 100 = 4 :=
by
  sorry

end loss_percentage_grinder_l104_104416


namespace range_of_a_for_empty_solution_set_l104_104994

theorem range_of_a_for_empty_solution_set :
  ∀ a : ℝ, (∀ x : ℝ, ¬ (|x - 3| + |x - 4| < a)) ↔ a ≤ 1 := 
sorry

end range_of_a_for_empty_solution_set_l104_104994


namespace total_copies_in_half_hour_l104_104005

-- Define the rates of the copy machines
def rate_machine1 : ℕ := 35
def rate_machine2 : ℕ := 65

-- Define the duration of time in minutes
def time_minutes : ℕ := 30

-- Define the total number of copies made by both machines in the given duration
def total_copies_made : ℕ := rate_machine1 * time_minutes + rate_machine2 * time_minutes

-- Prove that the total number of copies made is 3000
theorem total_copies_in_half_hour : total_copies_made = 3000 := by
  -- The proof is skipped with sorry for the demonstration purpose
  sorry

end total_copies_in_half_hour_l104_104005


namespace extinction_probability_l104_104801

-- Definitions from conditions
def prob_divide : ℝ := 0.6
def prob_die : ℝ := 0.4

-- Statement of the theorem
theorem extinction_probability :
  ∃ (v : ℝ), v = 2 / 3 :=
by
  sorry

end extinction_probability_l104_104801


namespace selling_price_conditions_met_l104_104308

-- Definitions based on the problem conditions
def initial_selling_price : ℝ := 50
def purchase_price : ℝ := 40
def initial_volume : ℝ := 500
def decrease_rate : ℝ := 10
def desired_profit : ℝ := 8000
def max_total_cost : ℝ := 10000

-- Definition for the selling price
def selling_price : ℝ := 80

-- Condition: Cost is below $10000 for the valid selling price
def valid_item_count (x : ℝ) : ℝ := initial_volume - decrease_rate * (x - initial_selling_price)

-- Cost calculation function
def total_cost (x : ℝ) : ℝ := purchase_price * valid_item_count x

-- Profit calculation function 
def profit (x : ℝ) : ℝ := (x - purchase_price) * valid_item_count x

-- Main theorem statement
theorem selling_price_conditions_met : 
  profit selling_price = desired_profit ∧ total_cost selling_price < max_total_cost :=
by
  sorry

end selling_price_conditions_met_l104_104308


namespace rationalize_denominator_l104_104750

theorem rationalize_denominator (h : Real.sqrt 200 = 10 * Real.sqrt 2) : 
  (7 / Real.sqrt 200) = (7 * Real.sqrt 2 / 20) :=
by
  sorry

end rationalize_denominator_l104_104750


namespace evaluate_fraction_l104_104039

noncomputable def evaluate_expression : ℚ := 
  1 / (2 - (1 / (2 - (1 / (2 - (1 / 3))))))
  
theorem evaluate_fraction :
  evaluate_expression = 5 / 7 :=
sorry

end evaluate_fraction_l104_104039


namespace geometric_sequence_a3_l104_104695

noncomputable def a_1 (S_4 : ℕ) (q : ℕ) : ℕ :=
  S_4 * (q - 1) / (1 - q^4)

noncomputable def a_3 (a_1 : ℕ) (q : ℕ) : ℕ :=
  a_1 * q^(3 - 1)

theorem geometric_sequence_a3 (a_n : ℕ → ℕ) (S_4 : ℕ) (q : ℕ) :
  (q = 2) →
  (S_4 = 60) →
  a_3 (a_1 S_4 q) q = 16 :=
by
  intro hq hS4
  rw [hq, hS4]
  sorry

end geometric_sequence_a3_l104_104695


namespace largest_c_for_minus3_in_range_of_quadratic_l104_104684

theorem largest_c_for_minus3_in_range_of_quadratic (c : ℝ) :
  (∃ x : ℝ, x^2 + 5*x + c = -3) ↔ c ≤ 13/4 :=
sorry

end largest_c_for_minus3_in_range_of_quadratic_l104_104684


namespace fraction_subtraction_l104_104832

theorem fraction_subtraction :
  (12 / 30) - (1 / 7) = 9 / 35 :=
by sorry

end fraction_subtraction_l104_104832


namespace beam_reflection_equation_l104_104336

theorem beam_reflection_equation:
  ∃ (line : ℝ → ℝ → Prop), 
  (∀ (x y : ℝ), line x y ↔ (5 * x - 2 * y - 10 = 0)) ∧
  (line 4 5) ∧ 
  (line 2 0) :=
by
  sorry

end beam_reflection_equation_l104_104336


namespace downstream_distance_l104_104498

theorem downstream_distance
  (time_downstream : ℝ) (time_upstream : ℝ)
  (distance_upstream : ℝ) (speed_still_water : ℝ)
  (h1 : time_downstream = 3) (h2 : time_upstream = 3)
  (h3 : distance_upstream = 15) (h4 : speed_still_water = 10) :
  ∃ d : ℝ, d = 45 :=
by
  sorry

end downstream_distance_l104_104498


namespace num_green_hats_l104_104635

-- Definitions
def total_hats : ℕ := 85
def blue_hat_cost : ℕ := 6
def green_hat_cost : ℕ := 7
def total_cost : ℕ := 548

-- Prove the number of green hats (g) is 38 given the conditions
theorem num_green_hats (b g : ℕ) 
  (h₁ : b + g = total_hats)
  (h₂ : blue_hat_cost * b + green_hat_cost * g = total_cost) : 
  g = 38 := by
  sorry

end num_green_hats_l104_104635


namespace new_class_mean_l104_104715

theorem new_class_mean 
  (n1 n2 : ℕ) (mean1 mean2 : ℚ) 
  (h1 : n1 = 24) (h2 : n2 = 8) 
  (h3 : mean1 = 85/100) (h4 : mean2 = 90/100) :
  (n1 * mean1 + n2 * mean2) / (n1 + n2) = 345/400 :=
by
  rw [h1, h2, h3, h4]
  sorry

end new_class_mean_l104_104715


namespace algebraic_expression_value_l104_104253

theorem algebraic_expression_value (x y : ℝ) (h1 : x + 2 * y = 4) (h2 : x - 2 * y = -1) : 
  x^2 - 4 * y^2 = -4 :=
by
  sorry

end algebraic_expression_value_l104_104253


namespace tangent_line_at_1_1_l104_104374

noncomputable def f (x : ℝ) : ℝ := x / (2 * x - 1)

theorem tangent_line_at_1_1 :
  let m := -((2 * 1 - 1 - 2 * 1) / (2 * 1 - 1)^2) -- Derivative evaluated at x = 1
  let tangent_line (x y : ℝ) := x + y - 2
  ∀ x y : ℝ, tangent_line x y = 0 → (f x = y ∧ x = 1 → y = 1 → m = -1) :=
by
  sorry

end tangent_line_at_1_1_l104_104374


namespace integer_subset_property_l104_104697

theorem integer_subset_property (M : Set ℤ) (h1 : ∃ a ∈ M, a > 0) (h2 : ∃ b ∈ M, b < 0)
(h3 : ∀ {a b : ℤ}, a ∈ M → b ∈ M → 2 * a ∈ M ∧ a + b ∈ M)
: ∀ a b : ℤ, a ∈ M → b ∈ M → a - b ∈ M :=
by
  sorry

end integer_subset_property_l104_104697


namespace find_b_of_square_binomial_l104_104397

theorem find_b_of_square_binomial (b : ℚ) 
  (h : ∃ c : ℚ, ∀ x : ℚ, (3 * x + c) ^ 2 = 9 * x ^ 2 + 21 * x + b) : 
  b = 49 / 4 := 
sorry

end find_b_of_square_binomial_l104_104397


namespace bricks_in_chimney_900_l104_104351

theorem bricks_in_chimney_900 (h : ℕ) :
  let Brenda_rate := h / 9
  let Brandon_rate := h / 10
  let combined_rate := (Brenda_rate + Brandon_rate) - 10
  5 * combined_rate = h → h = 900 :=
by
  intros Brenda_rate Brandon_rate combined_rate
  sorry

end bricks_in_chimney_900_l104_104351


namespace remainder_poly_l104_104048

noncomputable def remainder_division (f g : ℚ[X]) := 
  let ⟨q, r⟩ := f.div_mod g in r

theorem remainder_poly :
  remainder_division (3 * X ^ 5 - 2 * X ^ 3 + 5 * X - 8) (X ^ 2 - 3 * X + 2) = 84 * X - 84 :=
by
  sorry

end remainder_poly_l104_104048


namespace count_not_perm_sublist_permutations_correct_l104_104547

def is_not_permutation_sublist (a : List ℕ) : Prop :=
  ∀ i, 1 ≤ i ∧ i ≤ 4 → ¬(a.take i ~ List.range' 1 i)

def count_not_perm_sublist_permutations : ℕ :=
  (List.permutes [1, 2, 3, 4, 5]).filter is_not_permutation_sublist).length

theorem count_not_perm_sublist_permutations_correct :
  count_not_perm_sublist_permutations = 70 :=
sorry

end count_not_perm_sublist_permutations_correct_l104_104547


namespace problem_statement_l104_104870

theorem problem_statement (a b c : ℝ) (h1 : a + 2 * b + 3 * c = 12) (h2 : a^2 + b^2 + c^2 = a * b + a * c + b * c) :
  a + b^2 + c^3 = 14 :=
sorry

end problem_statement_l104_104870


namespace fraction_B_A_plus_C_l104_104004

variable (A B C : ℝ)
variable (f : ℝ)
variable (hA : A = 1 / 3 * (B + C))
variable (hB : A = B + 30)
variable (hTotal : A + B + C = 1080)
variable (hf : B = f * (A + C))

theorem fraction_B_A_plus_C :
  f = 2 / 7 :=
sorry

end fraction_B_A_plus_C_l104_104004


namespace binomial_coefficient_example_l104_104359

theorem binomial_coefficient_example :
  2 * (Nat.choose 7 4) = 70 := 
sorry

end binomial_coefficient_example_l104_104359


namespace subset_singleton_zero_l104_104071

def X : Set ℤ := {x | -2 ≤ x ∧ x ≤ 2}

theorem subset_singleton_zero : {0} ⊆ X :=
by
  sorry

end subset_singleton_zero_l104_104071


namespace necessary_but_not_sufficient_l104_104130

theorem necessary_but_not_sufficient (x : ℝ) :
  (x^2 - 5*x + 4 < 0) → (|x - 2| < 1) ∧ ¬( |x - 2| < 1 → x^2 - 5*x + 4 < 0) :=
by 
  sorry

end necessary_but_not_sufficient_l104_104130


namespace remainder_when_divided_by_5_l104_104472

theorem remainder_when_divided_by_5 
  (n : ℕ) 
  (h : n % 10 = 7) : 
  n % 5 = 2 := 
by 
  sorry

end remainder_when_divided_by_5_l104_104472


namespace t_sum_max_min_l104_104850

noncomputable def t_max (a b : ℝ) (h : (a - b / 2) ^ 2 = 1 - 7 / 4 * b ^ 2) : ℝ := sorry
noncomputable def t_min (a b : ℝ) (h : (a - b / 2) ^ 2 = 1 - 7 / 4 * b ^ 2) : ℝ := sorry

theorem t_sum_max_min (a b : ℝ) (h : (a - b / 2) ^ 2 = 1 - 7 / 4 * b ^ 2) :
  t_max a b h + t_min a b h = 16 / 7 := sorry

end t_sum_max_min_l104_104850


namespace solve_trig_eq_l104_104290

theorem solve_trig_eq :
  ∀ x k : ℤ, 
    (x = 2 * π / 3 + 2 * k * π ∨
     x = 7 * π / 6 + 2 * k * π ∨
     x = -π / 6 + 2 * k * π)
    → (|Real.cos x| + Real.cos (3 * x)) / (Real.sin x * Real.cos (2 * x)) = -2 * Real.sqrt 3 := 
by
  intros x k h
  sorry

end solve_trig_eq_l104_104290


namespace number_of_purple_balls_l104_104784

theorem number_of_purple_balls (k : ℕ) (h : k > 0) (E : (24 - k) / (8 + k) = 1) : k = 8 :=
by {
  sorry
}

end number_of_purple_balls_l104_104784


namespace amgm_inequality_abcd_l104_104385

-- Define the variables and their conditions
variables {a b c d : ℝ}
variable (ha : 0 < a)
variable (hb : 0 < b)
variable (hc : 0 < c)
variable (hd : 0 < d)

-- State the theorem
theorem amgm_inequality_abcd :
  a^4 * b + b^4 * c + c^4 * d + d^4 * a ≥ a * b * c * d * (a + b + c + d) :=
by
  sorry

end amgm_inequality_abcd_l104_104385


namespace max_fans_theorem_l104_104638

noncomputable def max_distinct_fans : ℕ :=
  let num_sectors := 6
  let total_configurations := 2 ^ num_sectors
  -- Configurations unchanged by flipping
  let unchanged_configurations := 8
  -- Subtracting unchanged from total and then divide by 2 to account for symmetric duplicates
  -- then add back the unchanged configurations
  (total_configurations - unchanged_configurations) / 2 + unchanged_configurations

theorem max_fans_theorem : max_distinct_fans = 36 := by
  sorry

end max_fans_theorem_l104_104638


namespace eval_expression_l104_104516

theorem eval_expression : 2010^3 - 2009 * 2010^2 - 2009^2 * 2010 + 2009^3 = 4019 :=
by
  sorry

end eval_expression_l104_104516


namespace mathland_transport_l104_104550

theorem mathland_transport (n : ℕ) (h : n ≥ 2) (transport : Fin n -> Fin n -> Prop) :
(∀ i j, transport i j ∨ transport j i) →
(∃ tr : Fin n -> Fin n -> Prop, 
  (∀ i j, transport i j → tr i j) ∨
  (∀ i j, transport j i → tr i j)) :=
by
  sorry

end mathland_transport_l104_104550


namespace acute_triangle_angle_A_is_60_degrees_l104_104921

open Real

variables {A B C : ℝ} -- Assume A, B, C are reals representing the angles of the triangle

theorem acute_triangle_angle_A_is_60_degrees
  (h_acute : A < 90 ∧ B < 90 ∧ C < 90)
  (h_eq_dist : dist A O = dist A H) : A = 60 :=
  sorry

end acute_triangle_angle_A_is_60_degrees_l104_104921


namespace inequality_solution_l104_104294

theorem inequality_solution (x : ℝ) : 
  (7 - 2 * (x + 1) ≥ 1 - 6 * x) ∧ ((1 + 2 * x) / 3 > x - 1) ↔ (-1 ≤ x ∧ x < 4) := 
by
  sorry

end inequality_solution_l104_104294


namespace sin_double_alpha_l104_104532

variable (α β : ℝ)

theorem sin_double_alpha (h1 : Real.pi / 2 < β ∧ β < α ∧ α < 3 * Real.pi / 4)
        (h2 : Real.cos (α - β) = 12 / 13) 
        (h3 : Real.sin (α + β) = -3 / 5) : 
        Real.sin (2 * α) = -56 / 65 := by
  sorry

end sin_double_alpha_l104_104532


namespace functional_relationship_maximum_profit_desired_profit_l104_104034

-- Conditions
def cost_price := 80
def y (x : ℝ) : ℝ := -2 * x + 320
def w (x : ℝ) : ℝ := (x - cost_price) * y x

-- Functional relationship
theorem functional_relationship (x : ℝ) (hx : 80 ≤ x ∧ x ≤ 160) :
  w x = -2 * x^2 + 480 * x - 25600 :=
by sorry

-- Maximizing daily profit
theorem maximum_profit :
  ∃ x, 80 ≤ x ∧ x ≤ 160 ∧ w x = 3200 ∧ (∀ y, 80 ≤ y ∧ y ≤ 160 → w y ≤ 3200) ∧ x = 120 :=
by sorry

-- Desired profit of 2400 dollars
theorem desired_profit (w_desired : ℝ) (hw : w_desired = 2400) :
  ∃ x, 80 ≤ x ∧ x ≤ 160 ∧ w x = w_desired ∧ x = 100 :=
by sorry

end functional_relationship_maximum_profit_desired_profit_l104_104034


namespace proof_y_minus_x_l104_104920

theorem proof_y_minus_x (x y : ℤ) (h1 : x + y = 540) (h2 : x = (4 * y) / 5) : y - x = 60 :=
sorry

end proof_y_minus_x_l104_104920


namespace no_perfect_square_after_swap_l104_104007

def is_consecutive_digits (a b c d : ℕ) : Prop := 
  (b = a + 1) ∧ (c = b + 1) ∧ (d = c + 1)

def swap_hundreds_tens (n : ℕ) : ℕ := 
  let d4 := n / 1000
  let d3 := (n % 1000) / 100
  let d2 := (n % 100) / 10
  let d1 := n % 10
  d4 * 1000 + d2 * 100 + d3 * 10 + d1

theorem no_perfect_square_after_swap : ¬ ∃ (n : ℕ), 
  1000 ≤ n ∧ n < 10000 ∧ 
  (let d4 := n / 1000
   let d3 := (n % 1000) / 100
   let d2 := (n % 100) / 10
   let d1 := n % 10
   is_consecutive_digits d4 d3 d2 d1) ∧ 
  let new_number := swap_hundreds_tens n
  (∃ m : ℕ, m * m = new_number) := 
sorry

end no_perfect_square_after_swap_l104_104007


namespace polynomial_factorization_l104_104098

theorem polynomial_factorization (a b k : ℤ) (h1 : Int.gcd a 3 = 1) (h2 : Int.gcd b 3 = 1) (h3 : a + b = 3 * k):
  ∃ q : ℤ[X], (X^a + X^b + 1) = (X^2 + X + 1) * q :=
by
  sorry

end polynomial_factorization_l104_104098


namespace common_ratio_of_geom_seq_l104_104836

-- Define the conditions: geometric sequence and the given equation
def is_geom_seq (a : ℕ → ℝ) := ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem common_ratio_of_geom_seq
  (a : ℕ → ℝ)
  (h_geom : is_geom_seq a)
  (h_eq : a 1 * a 7 = 3 * a 3 * a 4) :
  ∃ q : ℝ, is_geom_seq a ∧ q = 3 := 
sorry

end common_ratio_of_geom_seq_l104_104836


namespace judy_pencil_cost_l104_104884

theorem judy_pencil_cost :
  (∀ (pencils_per_week days_per_week pencils_per_pack cost_per_pack total_days), 
    (pencils_per_week = 10 → days_per_week = 5 → pencils_per_pack = 30 → cost_per_pack = 4 → total_days = 45 → 
    let pencils_per_day := pencils_per_week / days_per_week in
    let total_pencils_needed := pencils_per_day * total_days in
    let packs_needed := total_pencils_needed / pencils_per_pack in
    let total_cost := packs_needed * cost_per_pack in
    total_cost = 12)) :=
by 
  intros pencils_per_week days_per_week pencils_per_pack cost_per_pack total_days hw hd hp hc ht
  calc 
    let pencils_per_day := pencils_per_week / days_per_week in
    let total_pencils_needed := pencils_per_day * total_days in
    let packs_needed := total_pencils_needed / pencils_per_pack in
    let total_cost := packs_needed * cost_per_pack in
    total_cost = 12 : sorry

end judy_pencil_cost_l104_104884


namespace degree_sequence_a_invalid_degree_sequence_b_invalid_degree_sequence_c_invalid_all_sequences_invalid_l104_104356

-- Definition of the "isValidGraph" function based on degree sequences
-- Placeholder for the actual definition
def isValidGraph (degrees : List ℕ) : Prop :=
  sorry

-- Degree sequences given in the problem
def d_a := [8, 6, 5, 4, 4, 3, 2, 2]
def d_b := [7, 7, 6, 5, 4, 2, 2, 1]
def d_c := [6, 6, 6, 5, 5, 3, 2, 2]

-- Statement that proves none of these sequences can form a valid graph
theorem degree_sequence_a_invalid : ¬ isValidGraph d_a :=
  sorry

theorem degree_sequence_b_invalid : ¬ isValidGraph d_b :=
  sorry

theorem degree_sequence_c_invalid : ¬ isValidGraph d_c :=
  sorry

-- Final statement combining all individual proofs
theorem all_sequences_invalid :
  ¬ isValidGraph d_a ∧ ¬ isValidGraph d_b ∧ ¬ isValidGraph d_c :=
  ⟨degree_sequence_a_invalid, degree_sequence_b_invalid, degree_sequence_c_invalid⟩

end degree_sequence_a_invalid_degree_sequence_b_invalid_degree_sequence_c_invalid_all_sequences_invalid_l104_104356


namespace base_subtraction_l104_104194

def base8_to_base10 (n : Nat) : Nat :=
  -- base 8 number 54321 (in decimal representation)
  5 * 4096 + 4 * 512 + 3 * 64 + 2 * 8 + 1

def base5_to_base10 (n : Nat) : Nat :=
  -- base 5 number 4321 (in decimal representation)
  4 * 125 + 3 * 25 + 2 * 5 + 1

theorem base_subtraction :
  base8_to_base10 54321 - base5_to_base10 4321 = 22151 := by
  sorry

end base_subtraction_l104_104194


namespace max_wooden_pencils_l104_104875

theorem max_wooden_pencils (m w : ℕ) (p : ℕ) (h1 : m + w = 72) (h2 : m = w + p) (hp : Nat.Prime p) : w = 35 :=
by
  sorry

end max_wooden_pencils_l104_104875


namespace find_BC_l104_104058

variable (A B C : Type)
variables (a b : ℝ) -- Angles
variables (AB BC CA : ℝ) -- Sides of the triangle

-- Given conditions:
-- 1: Triangle ABC
-- 2: cos(a - b) + sin(a + b) = 2
-- 3: AB = 4

theorem find_BC (hAB : AB = 4) (hTrig : Real.cos (a - b) + Real.sin (a + b) = 2) :
  BC = 2 * Real.sqrt 2 := 
sorry

end find_BC_l104_104058


namespace mouse_cannot_eat_entire_cheese_l104_104950

-- Defining the conditions of the problem
structure Cheese :=
  (size : ℕ := 3)  -- The cube size is 3x3x3
  (central_cube_removed : Bool := true)  -- The central cube is removed

inductive CubeColor
| black
| white

structure Mouse :=
  (can_eat : CubeColor -> CubeColor -> Bool)
  (adjacency : Nat -> Nat -> Bool)

def cheese_problem (c : Cheese) (m : Mouse) : Bool := sorry

-- The main theorem: It is impossible for the mouse to eat the entire piece of cheese.
theorem mouse_cannot_eat_entire_cheese : ∀ (c : Cheese) (m : Mouse),
  cheese_problem c m = false := sorry

end mouse_cannot_eat_entire_cheese_l104_104950


namespace vegetable_options_l104_104187

open Nat

theorem vegetable_options (V : ℕ) : 
  3 * V + 6 = 57 → V = 5 :=
by
  intro h
  sorry

end vegetable_options_l104_104187


namespace solution_set_ineq_l104_104593

theorem solution_set_ineq (x : ℝ) : x^2 - 2 * abs x - 15 > 0 ↔ x < -5 ∨ x > 5 :=
sorry

end solution_set_ineq_l104_104593


namespace S_2011_l104_104553

variable {α : Type*}

-- Define initial term and sum function for arithmetic sequence
def a1 : ℤ := -2011
noncomputable def S (n : ℕ) : ℤ := n * a1 + (n * (n - 1) / 2) * 2

-- Given conditions
def condition1 : a1 = -2011 := rfl
def condition2 : (S 2010 / 2010) - (S 2008 / 2008) = 2 := by sorry

-- Proof statement
theorem S_2011 : S 2011 = -2011 := by 
  -- Use the given conditions to prove the statement
  sorry

end S_2011_l104_104553


namespace initial_men_garrison_l104_104008

-- Conditions:
-- A garrison has provisions for 31 days.
-- At the end of 16 days, a reinforcement of 300 men arrives.
-- The provisions last only for 5 days more after the reinforcement arrives.

theorem initial_men_garrison (M : ℕ) (P : ℕ) (d1 d2 : ℕ) (r : ℕ) (remaining1 remaining2 : ℕ) :
  P = M * d1 →
  remaining1 = P - M * d2 →
  remaining2 = r * (d1 - d2) →
  remaining1 = remaining2 →
  r = M + 300 →
  d1 = 31 →
  d2 = 16 →
  M = 150 :=
by 
  sorry

end initial_men_garrison_l104_104008


namespace find_x_value_l104_104688

theorem find_x_value (x : ℚ) (h : 5 * (x - 10) = 3 * (3 - 3 * x) + 9) : x = 34 / 7 := by
  sorry

end find_x_value_l104_104688


namespace max_min_lcm_l104_104730

theorem max_min_lcm (n : ℕ) (hn : n > 1) (S : set (finset ℕ)) (hS : S = { A | A.card = n ∧ A ⊆ finset.range (2*n + 1) }):
  (∀ A ∈ S, true) → -- dummy condition to use S in the context
  (max (λ A -> min (λ xy -> let ⟨x, y⟩ := xy in if x ≠ y then nat.lcm x y else ⊤) (A.val.pair_finset A.val)) (S.to_finset) = 
  if n % 2 = 1 then 3*(n+1) else if n = 2 then 12 else if n = 4 then 24 else if n ≥ 6 then 3*(n+2) else 0) :=
begin
  intro h,
  sorry,
end

end max_min_lcm_l104_104730


namespace find_x_l104_104231

def vector := (ℝ × ℝ)

def collinear (u v : vector) : Prop :=
  ∃ k : ℝ, u = (k * v.1, k * v.2)

def a : vector := (1, 2)
def b (x : ℝ) : vector := (x, 1)
def a_minus_b (x : ℝ) : vector := ((1 - x), 1)

theorem find_x (x : ℝ) (h : collinear a (a_minus_b x)) : x = 1/2 :=
by
  sorry

end find_x_l104_104231


namespace solve_eq1_solve_eq2_l104_104118

-- For Equation (1)
theorem solve_eq1 (x : ℝ) : x^2 - 4*x - 6 = 0 → x = 2 + Real.sqrt 10 ∨ x = 2 - Real.sqrt 10 :=
sorry

-- For Equation (2)
theorem solve_eq2 (x : ℝ) : (x / (x - 1) - 1 = 3 / (x^2 - 1)) → x ≠ 1 ∧ x ≠ -1 → x = 2 :=
sorry

end solve_eq1_solve_eq2_l104_104118


namespace fraction_simplification_l104_104774

theorem fraction_simplification (a b c x y : ℝ) (m : ℝ) :
  (∀ (x y : ℝ), (y ≠ 0 → (y^2 / x^2) ≠ (y / x))) ∧
  (∀ (a b c : ℝ), (a + c^2) / (b + c^2) ≠ a / b) ∧
  (∀ (a b m : ℝ), ¬(m ≠ -1 → (a + b) / (m * a + m * b) = 1 / 2)) ∧
  (∃ a b : ℝ, (a - b) / (b - a) = -1) :=
  by
  sorry

end fraction_simplification_l104_104774


namespace evaluate_g_at_2_l104_104844

def g (x : ℝ) : ℝ := x^3 - 2 * x + 1

theorem evaluate_g_at_2 : g 2 = 5 :=
by
  sorry

end evaluate_g_at_2_l104_104844


namespace prod_mod_eq_c_l104_104562

theorem prod_mod_eq_c {n : ℕ} (hn : 0 < n) {a b c : ℤ}
  (ha : is_unit (a % n)) (hb : is_unit (b % n)) (hc : is_unit (c % n))
  (h1 : a ≡ b⁻¹ [ZMOD n]) (h2 : c ≡ a⁻¹ [ZMOD n]) :
  (a * b) * c ≡ c [ZMOD n] :=
sorry

end prod_mod_eq_c_l104_104562


namespace total_marbles_l104_104570

-- Define the number of marbles Mary and Joan have respectively
def mary_marbles := 9
def joan_marbles := 3

-- Prove that the total number of marbles is 12
theorem total_marbles : mary_marbles + joan_marbles = 12 := by
  sorry

end total_marbles_l104_104570


namespace all_terms_perfect_squares_l104_104460

def seq_x : ℕ → ℕ
| 0       => 1
| 1       => 1
| (n + 2) => 14 * seq_x (n + 1) - seq_x n - 4

theorem all_terms_perfect_squares : ∀ n, ∃ k, seq_x n = k^2 :=
by
  sorry

end all_terms_perfect_squares_l104_104460


namespace part_a_part_b_l104_104267

variable (a b c : ℤ)
variable (h : a + b + c = 0)

theorem part_a : (a^4 + b^4 + c^4) % (a^2 + b^2 + c^2) = 0 :=
by
  -- proof goes here
  sorry

theorem part_b : (a^100 + b^100 + c^100) % (a^2 + b^2 + c^2) = 0 :=
by
  -- proof goes here
  sorry

end part_a_part_b_l104_104267


namespace inequality_proof_l104_104066

/-- Given a, b, c as positive real numbers. Prove that 
    √3 * ∛((a + b) * (b + c) * (c + a)) ≥ 2 * √(ab + bc + ca) -/
theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  √3 * ((a + b) * (b + c) * (c + a)).nthRoot 3 ≥ 2 * √(a * b + b * c + c * a) :=
by sorry

end inequality_proof_l104_104066


namespace original_number_increased_by_110_l104_104338

-- Define the conditions and the proof statement without the solution steps
theorem original_number_increased_by_110 {x : ℝ} (h : x + 1.10 * x = 1680) : x = 800 :=
by 
  sorry

end original_number_increased_by_110_l104_104338


namespace news_spread_time_l104_104633

theorem news_spread_time (n : ℕ) (m : ℕ) :
  (2^m < n ∧ n < 2^(m+k+1) ∧ (n % 2 = 1) ∧ n % 2 = 1) →
  ∃ t : ℕ, t = (if n % 2 = 1 then m+2 else m+1) := 
sorry

end news_spread_time_l104_104633


namespace smallest_integer_in_consecutive_set_l104_104876

theorem smallest_integer_in_consecutive_set (n : ℤ) (h : n + 6 < 2 * (n + 3)) : n > 0 := by
  sorry

end smallest_integer_in_consecutive_set_l104_104876


namespace intersection_points_polar_coords_l104_104849

theorem intersection_points_polar_coords :
  (∀ (x y : ℝ), ((x - 4)^2 + (y - 5)^2 = 25 ∧ (x^2 + y^2 - 2*y = 0)) →
  (∃ (ρ θ : ℝ), ρ > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧ 
    ((x, y) = (ρ * Real.cos θ, ρ * Real.sin θ)) ∧
    ((ρ = 2 ∧ θ = Real.pi / 2) ∨ (ρ = Real.sqrt 2 ∧ θ = Real.pi / 4)))) :=
sorry

end intersection_points_polar_coords_l104_104849


namespace mary_total_zoom_time_l104_104279

noncomputable def timeSpentDownloadingMac : ℝ := 10
noncomputable def timeSpentDownloadingWindows : ℝ := 3 * timeSpentDownloadingMac
noncomputable def audioGlitchesCount : ℝ := 2
noncomputable def audioGlitchDuration : ℝ := 4
noncomputable def totalAudioGlitchTime : ℝ := audioGlitchesCount * audioGlitchDuration
noncomputable def videoGlitchDuration : ℝ := 6
noncomputable def totalGlitchTime : ℝ := totalAudioGlitchTime + videoGlitchDuration
noncomputable def glitchFreeTalkingTime : ℝ := 2 * totalGlitchTime

theorem mary_total_zoom_time : 
  timeSpentDownloadingMac + timeSpentDownloadingWindows + totalGlitchTime + glitchFreeTalkingTime = 82 :=
by sorry

end mary_total_zoom_time_l104_104279


namespace discriminant_eq_perfect_square_l104_104710

variables (a b c t : ℝ)

-- Conditions
axiom a_nonzero : a ≠ 0
axiom t_root : a * t^2 + b * t + c = 0

-- Goal
theorem discriminant_eq_perfect_square :
  (b^2 - 4 * a * c) = (2 * a * t + b)^2 :=
by
  -- Conditions and goal are stated, proof to be filled.
  sorry

end discriminant_eq_perfect_square_l104_104710


namespace sum_m_n_is_192_l104_104468

def smallest_prime : ℕ := 2

def largest_four_divisors_under_200 : ℕ :=
  -- we assume this as 190 based on the provided problem's solution
  190

theorem sum_m_n_is_192 :
  smallest_prime = 2 →
  largest_four_divisors_under_200 = 190 →
  smallest_prime + largest_four_divisors_under_200 = 192 :=
by
  intros h1 h2
  sorry

end sum_m_n_is_192_l104_104468


namespace seating_arrangement_l104_104825

theorem seating_arrangement (x y : ℕ) (h : 9 * x + 7 * y = 112) : x = 7 :=
by
  sorry

end seating_arrangement_l104_104825


namespace solve_trig_eq_l104_104293

open Real

theorem solve_trig_eq (k : ℤ) : 
  (∃ x : ℝ, 
    (|cos x| + cos (3 * x)) / (sin x * cos (2 * x)) = -2 * sqrt 3 
    ∧ (x = -π/6 + 2 * k * π ∨ x = 2 * π/3 + 2 * k * π ∨ x = 7 * π/6 + 2 * k * π)) :=
sorry

end solve_trig_eq_l104_104293


namespace basketball_scores_l104_104944

theorem basketball_scores :
  (∃ P : Finset ℕ, P = { P | ∃ x : ℕ, x ∈ (Finset.range 8) ∧ P = x + 14 } ∧ P.card = 8) :=
by
  sorry

end basketball_scores_l104_104944


namespace fraction_left_handed_l104_104022

-- Definitions based on given conditions
def red_ratio : ℝ := 10
def blue_ratio : ℝ := 5
def green_ratio : ℝ := 3
def yellow_ratio : ℝ := 2

def red_left_handed_percent : ℝ := 0.37
def blue_left_handed_percent : ℝ := 0.61
def green_left_handed_percent : ℝ := 0.26
def yellow_left_handed_percent : ℝ := 0.48

-- Statement we want to prove
theorem fraction_left_handed : 
  (red_left_handed_percent * red_ratio + blue_left_handed_percent * blue_ratio +
  green_left_handed_percent * green_ratio + yellow_left_handed_percent * yellow_ratio) /
  (red_ratio + blue_ratio + green_ratio + yellow_ratio) = 8.49 / 20 :=
  sorry

end fraction_left_handed_l104_104022


namespace count_polynomials_with_three_integer_roots_l104_104963

def polynomial_with_roots (n: ℕ) : Nat :=
  have h: n = 8 := by
    sorry
  if n = 8 then
    -- Apply the combinatorial argument as discussed
    52
  else
    -- Case for other n
    0

theorem count_polynomials_with_three_integer_roots:
  polynomial_with_roots 8 = 52 := 
  sorry

end count_polynomials_with_three_integer_roots_l104_104963


namespace sin_of_7pi_over_6_l104_104200

theorem sin_of_7pi_over_6 : Real.sin (7 * Real.pi / 6) = -1 / 2 :=
by
  -- Conditions from the statement in a)
  -- Given conditions: \(\sin (180^\circ + \theta) = -\sin \theta\)
  -- \(\sin 30^\circ = \frac{1}{2}\)
  sorry

end sin_of_7pi_over_6_l104_104200


namespace pinedale_mall_distance_l104_104332

theorem pinedale_mall_distance 
  (speed : ℝ) (time_between_stops : ℝ) (num_stops : ℕ) (distance : ℝ) 
  (h_speed : speed = 60) 
  (h_time_between_stops : time_between_stops = 5 / 60) 
  (h_num_stops : ↑num_stops = 5) :
  distance = 25 :=
by
  sorry

end pinedale_mall_distance_l104_104332


namespace gray_area_l104_104027

def center_C : (ℝ × ℝ) := (6, 5)
def center_D : (ℝ × ℝ) := (14, 5)
def radius_C : ℝ := 3
def radius_D : ℝ := 3

theorem gray_area :
  let area_rectangle := 8 * 5
  let area_sector_C := (1 / 2) * π * radius_C^2
  let area_sector_D := (1 / 2) * π * radius_D^2
  area_rectangle - (area_sector_C + area_sector_D) = 40 - 9 * π :=
by
  sorry

end gray_area_l104_104027


namespace fraction_of_single_men_l104_104021

theorem fraction_of_single_men :
  ∀ (total_faculty : ℕ) (women_percentage : ℝ) (married_percentage : ℝ) (married_men_ratio : ℝ),
    women_percentage = 0.7 → married_percentage = 0.4 → married_men_ratio = 2/3 →
    (total_faculty * (1 - women_percentage)) * (1 - married_men_ratio) / 
    (total_faculty * (1 - women_percentage)) = 1/3 :=
by 
  intros total_faculty women_percentage married_percentage married_men_ratio h_women h_married h_men_marry
  sorry

end fraction_of_single_men_l104_104021


namespace range_of_m_l104_104548

theorem range_of_m 
  (h : ∀ x : ℝ, x^2 + m * x + m^2 - 1 > 0) :
  m ∈ (Set.Ioo (-(2 * Real.sqrt 3) / 3) (-(2 * Real.sqrt 3) / 3)).union (Set.Ioi ((2 * Real.sqrt 3) / 3)) := 
sorry

end range_of_m_l104_104548


namespace find_pairs_l104_104973

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b

theorem find_pairs {a b : ℝ} :
  (0 < b) → (b ≤ 1) → (0 < a) → (a < 1) → (2 * a + b ≤ 2) →
  (∀ x y : ℝ, f a b (x * y) + f a b (x + y) ≥ f a b x * f a b y) :=
by
  intros h_b_gt_zero h_b_le_one h_a_gt_zero h_a_lt_one h_2a_b_le_2
  sorry

end find_pairs_l104_104973


namespace asymptote_hole_sum_l104_104262

noncomputable def number_of_holes (f : ℝ → ℝ) : ℕ := -- Assume a definition to count holes
sorry

noncomputable def number_of_vertical_asymptotes (f : ℝ → ℝ) : ℕ := -- Assume a definition to count vertical asymptotes
sorry

noncomputable def number_of_horizontal_asymptotes (f : ℝ → ℝ) : ℕ := -- Assume a definition to count horizontal asymptotes
sorry

noncomputable def number_of_oblique_asymptotes (f : ℝ → ℝ) : ℕ := -- Assume a definition to count oblique asymptotes
sorry

theorem asymptote_hole_sum :
  let f := λ x => (x^2 + 4*x + 3) / (x^3 - 2*x^2 - x + 2)
  let a := number_of_holes f
  let b := number_of_vertical_asymptotes f
  let c := number_of_horizontal_asymptotes f
  let d := number_of_oblique_asymptotes f
  a + 2 * b + 3 * c + 4 * d = 8 :=
by
  sorry

end asymptote_hole_sum_l104_104262


namespace simplify_expression_l104_104371

theorem simplify_expression (x : ℝ) : 24 * (3 * x - 4) - 6 * x = 66 * x - 96 := 
  sorry

end simplify_expression_l104_104371


namespace fraction_satisfactory_is_two_thirds_l104_104716

-- Total number of students with satisfactory grades
def satisfactory_grades : ℕ := 3 + 7 + 4 + 2

-- Total number of students with unsatisfactory grades
def unsatisfactory_grades : ℕ := 4

-- Total number of students
def total_students : ℕ := satisfactory_grades + unsatisfactory_grades

-- Fraction of satisfactory grades
def fraction_satisfactory : ℚ := satisfactory_grades / total_students

theorem fraction_satisfactory_is_two_thirds :
  fraction_satisfactory = 2 / 3 := by
  sorry

end fraction_satisfactory_is_two_thirds_l104_104716


namespace abs_cube_root_neg_64_l104_104940

-- Definitions required for the problem
def cube_root (x : ℝ) : ℝ := x^(1/3)
def abs_value (x : ℝ) : ℝ := abs x

-- The statement of the problem
theorem abs_cube_root_neg_64 : abs_value (cube_root (-64)) = 4 :=
by sorry

end abs_cube_root_neg_64_l104_104940


namespace minimum_value_of_quadratic_polynomial_l104_104149

-- Define the quadratic polynomial
def quadratic_polynomial (x : ℝ) : ℝ := x^2 + 14 * x + 3

-- Statement to prove
theorem minimum_value_of_quadratic_polynomial : ∃ x : ℝ, quadratic_polynomial x = quadratic_polynomial (-7) :=
sorry

end minimum_value_of_quadratic_polynomial_l104_104149


namespace total_eggs_found_l104_104415

def eggs_club_house := 12
def eggs_park := 5
def eggs_town_hall_garden := 3

theorem total_eggs_found : eggs_club_house + eggs_park + eggs_town_hall_garden = 20 :=
by
  sorry

end total_eggs_found_l104_104415


namespace min_value_expression_l104_104064

variable (a b : ℝ)

theorem min_value_expression :
  0 < a →
  1 < b →
  a + b = 2 →
  (∃ x, x = 3 + 2 * Real.sqrt 2 ∧ (∀ y, y = (2 / a) + (1 / (b - 1)) → y ≥ x)) :=
by
  sorry

end min_value_expression_l104_104064


namespace MrFletcherPaymentPerHour_l104_104103

theorem MrFletcherPaymentPerHour :
  (2 * (10 + 8 + 15)) * x = 660 → x = 10 :=
by
  -- This is where you'd provide the proof, but we skip it as per instructions.
  sorry

end MrFletcherPaymentPerHour_l104_104103


namespace integer_power_sums_l104_104891

theorem integer_power_sums (x : ℝ) (h : x + (1 / x) ∈ ℤ) (n : ℕ) : 
  x^n + (1 / x^n) ∈ ℤ := 
sorry

end integer_power_sums_l104_104891


namespace max_sum_of_abc_l104_104263

theorem max_sum_of_abc (A B C : ℕ) (h₁ : A ≠ B) (h₂ : B ≠ C) (h₃ : A ≠ C) (h₄ : A * B * C = 2310) : 
  A + B + C ≤ 52 :=
sorry

end max_sum_of_abc_l104_104263


namespace total_exercise_time_l104_104725

-- Definitions based on given conditions
def javier_daily : ℕ := 50
def javier_days : ℕ := 7
def sanda_daily : ℕ := 90
def sanda_days : ℕ := 3

-- Proof problem to verify the total exercise time for both Javier and Sanda
theorem total_exercise_time : javier_daily * javier_days + sanda_daily * sanda_days = 620 := by
  sorry

end total_exercise_time_l104_104725


namespace minimum_m_n_squared_l104_104984

theorem minimum_m_n_squared (a b c m n : ℝ) (h1 : c > a) (h2 : c > b) (h3 : c = Real.sqrt (a^2 + b^2)) 
    (h4 : a * m + b * n + c = 0) : m^2 + n^2 ≥ 1 := by
  sorry

end minimum_m_n_squared_l104_104984


namespace valid_password_count_l104_104344

/-- 
The number of valid 4-digit ATM passwords at Fred's Bank, composed of digits from 0 to 9,
that do not start with the sequence "9,1,1" and do not end with the digit "5",
is 8991.
-/
theorem valid_password_count : 
  let total_passwords : ℕ := 10000
  let start_911 : ℕ := 10
  let end_5 : ℕ := 1000
  let start_911_end_5 : ℕ := 1
  total_passwords - (start_911 + end_5 - start_911_end_5) = 8991 :=
by
  let total_passwords : ℕ := 10000
  let start_911 : ℕ := 10
  let end_5 : ℕ := 1000
  let start_911_end_5 : ℕ := 1
  show total_passwords - (start_911 + end_5 - start_911_end_5) = 8991
  sorry

end valid_password_count_l104_104344


namespace lcm_36_105_l104_104214

theorem lcm_36_105 : Nat.lcm 36 105 = 1260 := by
  sorry

end lcm_36_105_l104_104214


namespace equation_of_line_l104_104930

noncomputable def line_equation_parallel (x y : ℝ) : Prop :=
  ∃ (m : ℝ), (3 * x - 6 * y = 9) ∧ (m = 1/2)

theorem equation_of_line (m : ℝ) (b : ℝ) :
  line_equation_parallel 3 9 →
  (m = 1/2) →
  (∀ (x y : ℝ), (y = m * x + b) ↔ (y - (-1) = m * (x - 2))) →
  b = -2 :=
by
  intros h_eq h_m h_line
  sorry

end equation_of_line_l104_104930


namespace sequence_difference_constant_l104_104938

theorem sequence_difference_constant :
  ∀ (x y : ℕ → ℕ), x 1 = 2 → y 1 = 1 →
  (∀ k, k > 1 → x k = 2 * x (k - 1) + 3 * y (k - 1)) →
  (∀ k, k > 1 → y k = x (k - 1) + 2 * y (k - 1)) →
  ∀ k, x k ^ 2 - 3 * y k ^ 2 = 1 :=
by
  -- Insert the proof steps here
  sorry

end sequence_difference_constant_l104_104938


namespace incorrect_statements_for_quadratic_inequality_l104_104988

-- Definitions for given conditions and required expressions
def quadratic_solution_set_empty (a b c : ℝ) : Prop := 
  a > 0 ∧ (b^2 - 4 * a * c) ≤ 0

def minimum_value_expression (a b c : ℝ) (x0 : ℝ) : ℝ :=
  (a + 4 * c) / (b - a)

theorem incorrect_statements_for_quadratic_inequality (a b c x0 : ℝ) :
  (quadratic_solution_set_empty a b c ↔ M = ∅) → 
  (M = { x | x ≠ x0 } ∧ a < b → minimum_value_expression a 4 c = 2 - 2 * real.sqrt 2) :=
sorry

end incorrect_statements_for_quadratic_inequality_l104_104988


namespace actual_distance_between_towns_l104_104895

-- Definitions based on conditions
def scale_inch_to_miles : ℚ := 8
def map_distance_inches : ℚ := 27 / 8

-- Proof statement
theorem actual_distance_between_towns : scale_inch_to_miles * map_distance_inches / (1 / 4) = 108 := by
  sorry

end actual_distance_between_towns_l104_104895


namespace find_integer_k_l104_104153

noncomputable def P : ℤ → ℤ := sorry

theorem find_integer_k :
  P 1 = 2019 ∧ P 2019 = 1 ∧ ∃ k : ℤ, P k = k ∧ k = 1010 :=
by
  sorry

end find_integer_k_l104_104153


namespace lcm_36_105_l104_104209

theorem lcm_36_105 : Int.lcm 36 105 = 1260 := by
  sorry

end lcm_36_105_l104_104209


namespace student_marks_l104_104877

theorem student_marks (x : ℕ) :
  let total_questions := 60
  let correct_answers := 38
  let wrong_answers := total_questions - correct_answers
  let total_marks := 130
  let marks_from_correct := correct_answers * x
  let marks_lost := wrong_answers * 1
  let net_marks := marks_from_correct - marks_lost
  net_marks = total_marks → x = 4 :=
by
  intros
  sorry

end student_marks_l104_104877


namespace number_of_boys_in_class_l104_104126

theorem number_of_boys_in_class 
  (n : ℕ)
  (average_height : ℝ)
  (incorrect_height : ℝ)
  (correct_height : ℝ)
  (actual_average_height : ℝ)
  (initial_average_height : average_height = 185)
  (incorrect_record : incorrect_height = 166)
  (correct_record : correct_height = 106)
  (actual_avg : actual_average_height = 183) 
  (total_height_incorrect : ℝ) 
  (total_height_correct : ℝ) 
  (total_height_eq : total_height_incorrect = 185 * n)
  (correct_total_height_eq : total_height_correct = 185 * n - (incorrect_height - correct_height))
  (actual_total_height_eq : total_height_correct = actual_average_height * n) :
  n = 30 :=
by
  sorry

end number_of_boys_in_class_l104_104126


namespace probability_intersecting_diagonals_l104_104311

def number_of_vertices := 10

def number_of_diagonals : ℕ := Nat.choose number_of_vertices 2 - number_of_vertices

def number_of_ways_choose_two_diagonals := Nat.choose number_of_diagonals 2

def number_of_sets_of_intersecting_diagonals : ℕ := Nat.choose number_of_vertices 4

def intersection_probability : ℚ :=
  (number_of_sets_of_intersecting_diagonals : ℚ) / (number_of_ways_choose_two_diagonals : ℚ)

theorem probability_intersecting_diagonals :
  intersection_probability = 42 / 119 :=
by
  sorry

end probability_intersecting_diagonals_l104_104311


namespace triple_comp_g_of_2_l104_104099

def g (n : ℕ) : ℕ :=
  if n ≤ 3 then n^3 - 2 else 4 * n + 1

theorem triple_comp_g_of_2 : g (g (g 2)) = 101 := by
  sorry

end triple_comp_g_of_2_l104_104099


namespace range_of_k_l104_104706

theorem range_of_k (k x y : ℝ) 
  (h₁ : 2 * x - y = k + 1) 
  (h₂ : x - y = -3) 
  (h₃ : x + y > 2) : k > -4.5 :=
sorry

end range_of_k_l104_104706


namespace first_term_of_geometric_series_l104_104916

theorem first_term_of_geometric_series (a r : ℝ) 
    (h1 : a / (1 - r) = 18) 
    (h2 : a^2 / (1 - r^2) = 72) : 
    a = 72 / 11 := 
  sorry

end first_term_of_geometric_series_l104_104916


namespace james_browsers_l104_104095

def num_tabs_per_window := 10
def num_windows_per_browser := 3
def total_tabs := 60

theorem james_browsers : ∃ B : ℕ, (B * (num_windows_per_browser * num_tabs_per_window) = total_tabs) ∧ (B = 2) := sorry

end james_browsers_l104_104095


namespace sequence_properties_l104_104702

-- Define the sequence formula
def a_n (n : ℤ) : ℤ := n^2 - 5 * n + 4

-- State the theorem about the sequence
theorem sequence_properties :
  -- Part 1: The number of negative terms in the sequence
  (∃ (S : Finset ℤ), ∀ n ∈ S, a_n n < 0 ∧ S.card = 2) ∧
  -- Part 2: The minimum value of the sequence and the value of n at minimum
  (∀ n : ℤ, (a_n n ≥ -9 / 4) ∧ (a_n (5 / 2) = -9 / 4)) :=
by {
  sorry
}

end sequence_properties_l104_104702


namespace complex_pow_eight_l104_104729

theorem complex_pow_eight (z : ℂ) (h : z = (Complex.mk (Real.sqrt 3) 1) / 2) :
  z ^ 8 = -1 := by
  sorry

end complex_pow_eight_l104_104729


namespace g_600_l104_104272

def g : ℕ → ℕ := sorry

axiom g_mul (x y : ℕ) (hx : x > 0) (hy : y > 0) : g (x * y) = g x + g y
axiom g_12 : g 12 = 18
axiom g_48 : g 48 = 26

theorem g_600 : g 600 = 36 :=
by 
  sorry

end g_600_l104_104272


namespace find_value_of_m_l104_104254

theorem find_value_of_m (m : ℤ) (x : ℤ) (h : (x - 3 ≠ 0) ∧ (x = 3)) : 
  ((x - 1) / (x - 3) = m / (x - 3)) → m = 2 :=
by
  sorry

end find_value_of_m_l104_104254


namespace break_even_machines_l104_104365

theorem break_even_machines (cost_parts cost_patent machine_price total_costs : ℝ):
    cost_parts = 3600 ∧ cost_patent = 4500 ∧ machine_price = 180 ∧ total_costs = cost_parts + cost_patent → 
    total_costs / machine_price = 45 :=
by
  intros h
  cases h with h1 h'
  cases h' with h2 h''
  cases h'' with h3 h4
  sorry

end break_even_machines_l104_104365


namespace real_coefficient_polynomials_with_special_roots_l104_104974

noncomputable def P1 : Polynomial ℝ := (Polynomial.X + 1) * (Polynomial.X - 1 / 2) * (Polynomial.X - 2) * (Polynomial.X ^ 2 - Polynomial.X + 1)
noncomputable def P2 : Polynomial ℝ := (Polynomial.X + 1) ^ 3 * (Polynomial.X - 1 / 2) * (Polynomial.X - 2)
noncomputable def P3 : Polynomial ℝ := (Polynomial.X + 1) * (Polynomial.X - 1 / 2) ^ 3 * (Polynomial.X - 2)
noncomputable def P4 : Polynomial ℝ := (Polynomial.X + 1) * (Polynomial.X - 1 / 2) * (Polynomial.X - 2) ^ 3
noncomputable def P5 : Polynomial ℝ := (Polynomial.X + 1) ^ 2 * (Polynomial.X - 1 / 2) ^ 2 * (Polynomial.X - 2)
noncomputable def P6 : Polynomial ℝ := (Polynomial.X + 1) * (Polynomial.X - 1 / 2) ^ 2 * (Polynomial.X - 2) ^ 2
noncomputable def P7 : Polynomial ℝ := (Polynomial.X + 1) ^ 2 * (Polynomial.X - 1 / 2) * (Polynomial.X - 2) ^ 2

theorem real_coefficient_polynomials_with_special_roots (P : Polynomial ℝ) :
  (∀ α, Polynomial.IsRoot P α → Polynomial.IsRoot P (1 - α) ∧ Polynomial.IsRoot P (1 / α)) →
  P = P1 ∨ P = P2 ∨ P = P3 ∨ P = P4 ∨ P = P5 ∨ P = P6 ∨ P = P7 :=
  sorry

end real_coefficient_polynomials_with_special_roots_l104_104974


namespace range_of_a_l104_104398

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (|x + a| < 3 ↔ 2 < x ∧ x < 3)) ↔ -5 ≤ a ∧ a ≤ 0 := by
  sorry

end range_of_a_l104_104398


namespace bandage_overlap_l104_104475

theorem bandage_overlap
  (n : ℕ) (l : ℝ) (total_length : ℝ) (required_length : ℝ)
  (h_n : n = 20) (h_l : l = 15.25) (h_required_length : required_length = 248) :
  (required_length = l * n - (n - 1) * 3) :=
by
  sorry

end bandage_overlap_l104_104475


namespace lcm_36_105_l104_104224

noncomputable def factorize_36 : List (ℕ × ℕ) := [(2, 2), (3, 2)]
noncomputable def factorize_105 : List (ℕ × ℕ) := [(3, 1), (5, 1), (7, 1)]

theorem lcm_36_105 : Nat.lcm 36 105 = 1260 :=
by
  have h_36 : 36 = 2^2 * 3^2 := by norm_num
  have h_105 : 105 = 3^1 * 5^1 * 7^1 := by norm_num
  sorry

end lcm_36_105_l104_104224


namespace total_remaining_staff_l104_104131

-- Definitions of initial counts and doctors and nurses quitting.
def initial_doctors : ℕ := 11
def initial_nurses : ℕ := 18
def doctors_quitting : ℕ := 5
def nurses_quitting : ℕ := 2

-- Definition of remaining doctors and nurses.
def remaining_doctors : ℕ := initial_doctors - doctors_quitting
def remaining_nurses : ℕ := initial_nurses - nurses_quitting

-- Theorem stating the total number of doctors and nurses remaining.
theorem total_remaining_staff : remaining_doctors + remaining_nurses = 22 :=
by
  -- Proof omitted
  sorry

end total_remaining_staff_l104_104131


namespace watermelon_weight_l104_104327

theorem watermelon_weight (B W : ℝ) (n : ℝ) 
  (h1 : B + n * W = 63) 
  (h2 : B + (n / 2) * W = 34) : 
  n * W = 58 :=
sorry

end watermelon_weight_l104_104327


namespace medians_sum_square_l104_104564

-- Define the sides of the triangle
variables {a b c : ℝ}

-- Define diameters
variables {D : ℝ}

-- Define medians of the triangle
variables {m_a m_b m_c : ℝ}

-- Defining the theorem statement
theorem medians_sum_square :
  m_a ^ 2 + m_b ^ 2 + m_c ^ 2 = (3 / 4) * (a ^ 2 + b ^ 2 + c ^ 2) + (3 / 4) * D ^ 2 :=
sorry

end medians_sum_square_l104_104564


namespace product_of_reverse_numbers_l104_104591

def reverse (n : Nat) : Nat :=
  Nat.ofDigits 10 (List.reverse (Nat.digits 10 n))

theorem product_of_reverse_numbers : 
  ∃ (a b : ℕ), a * b = 92565 ∧ b = reverse a ∧ ((a = 165 ∧ b = 561) ∨ (a = 561 ∧ b = 165)) :=
by
  sorry

end product_of_reverse_numbers_l104_104591


namespace sum_of_cubes_divisible_by_9n_l104_104577

theorem sum_of_cubes_divisible_by_9n (n : ℕ) (h : n % 3 ≠ 0) : 
  ((n - 1)^3 + n^3 + (n + 1)^3) % (9 * n) = 0 := by
  sorry

end sum_of_cubes_divisible_by_9n_l104_104577


namespace f_is_zero_l104_104538

noncomputable def f (x : ℝ) : ℝ := sorry

theorem f_is_zero 
  (H1 : ∀ a b : ℝ, f (a * b) = a * f b + b * f a)
  (H2 : ∀ x : ℝ, |f x| ≤ 1) : ∀ x : ℝ, f x = 0 := 
sorry

end f_is_zero_l104_104538


namespace vector_parallel_solution_l104_104083

theorem vector_parallel_solution (x : ℝ) : 
  let a := (2, 3)
  let b := (x, -9)
  (a.snd = 3) → (a.fst = 2) → (b.snd = -9) → (a.fst * b.snd = a.snd * (b.fst)) → x = -6 := 
by
  intros 
  sorry

end vector_parallel_solution_l104_104083


namespace probability_of_intersection_in_decagon_is_fraction_l104_104315

open_locale big_operators

noncomputable def probability_intersecting_diagonals_in_decagon : ℚ :=
let num_points := 10 in
let diagonals := (num_points.choose 2) - num_points in
let total_diagonal_pairs := (diagonals.choose 2) in
let valid_intersections := (num_points.choose 4) in
valid_intersections / total_diagonal_pairs

theorem probability_of_intersection_in_decagon_is_fraction :
  probability_intersecting_diagonals_in_decagon = 42 / 119 :=
by {
  unfold probability_intersecting_diagonals_in_decagon,
  sorry
}

end probability_of_intersection_in_decagon_is_fraction_l104_104315


namespace max_three_digit_divisible_by_4_sequence_l104_104160

theorem max_three_digit_divisible_by_4_sequence (a : ℕ → ℕ) (n : ℕ) (h1 : ∀ k ≤ n - 2, a (k + 2) = 3 * a (k + 1) - 2 * a k - 2)
(h2 : ∀ k1 k2, k1 < k2 → a k1 < a k2) (ha2022 : ∃ k, a k = 2022) (hn : n ≥ 3) :
  ∃ m : ℕ, ∀ k, 100 ≤ a k ∧ a k ≤ 999 → a k % 4 = 0 → m ≤ 225 := by
  sorry

end max_three_digit_divisible_by_4_sequence_l104_104160


namespace virus_infection_l104_104506

theorem virus_infection (x : ℕ) (h : 1 + x + x^2 = 121) : x = 10 := 
sorry

end virus_infection_l104_104506


namespace find_k_l104_104080

theorem find_k : ∃ k : ℕ, (2 * (Real.sqrt (225 + k)) = (Real.sqrt (49 + k) + Real.sqrt (441 + k))) → k = 255 :=
by
  sorry

end find_k_l104_104080


namespace number_of_intersection_points_l104_104120

theorem number_of_intersection_points (f : ℝ → ℝ) (hf : Function.Injective f) :
  ∃ x : Finset ℝ, (∀ y ∈ x, f ((y:ℝ)^2) = f ((y:ℝ)^6)) ∧ x.card = 3 :=
by
  sorry

end number_of_intersection_points_l104_104120


namespace factorize_1_factorize_2_factorize_3_solve_system_l104_104668

-- Proving the factorization identities
theorem factorize_1 (y : ℝ) : 5 * y - 10 * y^2 = 5 * y * (1 - 2 * y) :=
by
  sorry

theorem factorize_2 (m : ℝ) : (3 * m - 1)^2 - 9 = (3 * m + 2) * (3 * m - 4) :=
by
  sorry

theorem factorize_3 (a b : ℝ) : a^2 * b - 4 * a * b + 4 * b = b * (a - 2)^2 :=
by
  sorry

-- Proving the solution to the system of equations
theorem solve_system (x y : ℝ) (h1 : x - y = 3) (h2 : x - 3 * y = -1) : x = 5 ∧ y = 2 :=
by
  sorry

end factorize_1_factorize_2_factorize_3_solve_system_l104_104668


namespace sum_of_possible_values_CDF_l104_104789

theorem sum_of_possible_values_CDF 
  (C D F : ℕ) 
  (hC: 0 ≤ C ∧ C ≤ 9)
  (hD: 0 ≤ D ∧ D ≤ 9)
  (hF: 0 ≤ F ∧ F ≤ 9)
  (hdiv: (C + 4 + 9 + 8 + D + F + 4) % 9 = 0) :
  C + D + F = 2 ∨ C + D + F = 11 → (2 + 11 = 13) :=
by sorry

end sum_of_possible_values_CDF_l104_104789


namespace find_S20_l104_104240

variable {α : Type*} [AddCommGroup α] [Module ℝ α]
variable (a : ℕ → ℝ) (S : ℕ → ℝ)

-- Conditions
axiom sum_first_n_terms (n : ℕ) : S n = n * (a 1 + a n) / 2
axiom points_collinear (A B C O : α) : Collinear ℝ ({A, B, C} : Set α) ∧ O = 0
axiom vector_relationship (A B C O : α) : O = 0 → C = (a 12) • A + (a 9) • B
axiom line_not_through_origin (A B O : α) : ¬Collinear ℝ ({O, A, B} : Set α)

-- Question: To find S 20
theorem find_S20 (A B C O : α) (h_collinear : Collinear ℝ ({A, B, C} : Set α)) 
  (h_vector : O = 0 → C = (a 12) • A + (a 9) • B) 
  (h_origin : O = 0)
  (h_not_through_origin : ¬Collinear ℝ ({O, A, B} : Set α)) : 
  S 20 = 10 := by
  sorry

end find_S20_l104_104240


namespace number_of_men_in_first_group_l104_104003

theorem number_of_men_in_first_group (M : ℕ) : (M * 15 = 25 * 18) → M = 30 :=
by
  sorry

end number_of_men_in_first_group_l104_104003


namespace smoking_lung_disease_confidence_l104_104954

/-- Prove that given the conditions, the correct statement is C:
   If it is concluded from the statistic that there is a 95% confidence 
   that smoking is related to lung disease, then there is a 5% chance of
   making a wrong judgment. -/
theorem smoking_lung_disease_confidence 
  (P Q : Prop) 
  (confidence_level : ℝ) 
  (h_conf : confidence_level = 0.95) 
  (h_PQ : P → (Q → true)) :
  ¬Q → (confidence_level = 1 - 0.05) :=
by
  sorry

end smoking_lung_disease_confidence_l104_104954


namespace geometric_sequence_condition_l104_104297

-- Define the condition ac = b^2
def condition (a b c : ℝ) : Prop := a * c = b ^ 2

-- Define what it means for a, b, c to form a geometric sequence
def geometric_sequence (a b c : ℝ) : Prop := 
  (b ≠ 0 → a / b = b / c) ∧ (a = 0 → b = 0 ∧ c = 0)

-- The goal is to prove the necessary but not sufficient condition
theorem geometric_sequence_condition (a b c : ℝ) :
  condition a b c ↔ (geometric_sequence a b c → condition a b c) ∧ (¬ (geometric_sequence a b c) → condition a b c ∧ ¬ (geometric_sequence (2 : ℝ) (0 : ℝ) (0 : ℝ))) :=
by
  sorry

end geometric_sequence_condition_l104_104297


namespace oliver_money_left_l104_104746

-- Definitions based on the given conditions
def initial_amount : ℕ := 9
def allowance_saved : ℕ := 5
def frisbee_cost : ℕ := 4
def puzzle_cost : ℕ := 3
def birthday_gift : ℕ := 8

-- The proof problem statement
theorem oliver_money_left : 
  initial_amount + allowance_saved - (frisbee_cost + puzzle_cost) + birthday_gift = 15 := 
by
  sorry

end oliver_money_left_l104_104746


namespace ali_peter_fish_ratio_l104_104664

theorem ali_peter_fish_ratio (P J A : ℕ) (h1 : J = P + 1) (h2 : A = 12) (h3 : A + P + J = 25) : A / P = 2 :=
by
  -- Step-by-step simplifications will follow here in the actual proof.
  sorry

end ali_peter_fish_ratio_l104_104664


namespace Rajesh_work_completion_time_l104_104569

-- Definitions based on conditions in a)
def Mahesh_rate := 1 / 60 -- Mahesh's rate of work (work per day)
def Mahesh_work := 20 * Mahesh_rate -- Work completed by Mahesh in 20 days
def Rajesh_time_to_complete_remaining_work := 30 -- Rajesh time to complete remaining work (days)
def Remaining_work := 1 - Mahesh_work -- Remaining work after Mahesh's contribution

-- Statement that needs to be proved
theorem Rajesh_work_completion_time :
  (Rajesh_time_to_complete_remaining_work : ℝ) * (1 / Remaining_work) = 45 :=
sorry

end Rajesh_work_completion_time_l104_104569


namespace inequality_5a2_5b2_5c2_ge_4ab_4ac_4bc_l104_104530

theorem inequality_5a2_5b2_5c2_ge_4ab_4ac_4bc (a b c : ℝ) :
  5 * a^2 + 5 * b^2 + 5 * c^2 ≥ 4 * a * b + 4 * a * c + 4 * b * c ∧
  (5 * a^2 + 5 * b^2 + 5 * c^2 = 4 * a * b + 4 * a * c + 4 * b * c → a = 0 ∧ b = 0 ∧ c = 0) := sorry

end inequality_5a2_5b2_5c2_ge_4ab_4ac_4bc_l104_104530


namespace triangle_side_relation_l104_104429

variable {α β γ : ℝ} -- angles in the triangle
variable {a b c : ℝ} -- sides opposite to the angles

theorem triangle_side_relation
  (h1 : α = 3 * β)
  (h2 : α = 6 * γ)
  (h_sum : α + β + γ = 180)
  : b * c^2 = (a + b) * (a - b)^2 := 
by
  sorry

end triangle_side_relation_l104_104429


namespace find_y_l104_104689

theorem find_y (y : ℕ) : y = (12 ^ 3 * 6 ^ 4) / 432 → y = 5184 :=
by
  intro h
  rw [h]
  sorry

end find_y_l104_104689


namespace no_valid_m_n_l104_104380

theorem no_valid_m_n (m n : ℕ) (hm : m ≠ 0) (hn : n ≠ 0) : ¬ (m * n ∣ 3^m + 1 ∧ m * n ∣ 3^n + 1) :=
by
  sorry

end no_valid_m_n_l104_104380


namespace cubic_sum_identity_l104_104121

theorem cubic_sum_identity
  (x y z : ℝ)
  (h1 : x + y + z = 8)
  (h2 : x * y + x * z + y * z = 17)
  (h3 : x * y * z = -14) :
  x^3 + y^3 + z^3 = 62 :=
sorry

end cubic_sum_identity_l104_104121


namespace largest_of_a_b_c_d_e_l104_104242

theorem largest_of_a_b_c_d_e (a b c d e : ℝ)
  (h1 : a - 2 = b + 3)
  (h2 : a - 2 = c - 4)
  (h3 : a - 2 = d + 5)
  (h4 : a - 2 = e - 6) :
  e > a ∧ e > b ∧ e > c ∧ e > d :=
by
  sorry

end largest_of_a_b_c_d_e_l104_104242


namespace find_inverse_value_l104_104236

noncomputable def f (x : ℝ) : ℝ := sorry -- f(x) function definition goes here

theorem find_inverse_value :
  (∀ x : ℝ, f (x - 1) = f (x + 3)) →
  (∀ x : ℝ, f (-x) = f x) →
  (∀ x : ℝ, 4 ≤ x ∧ x ≤ 6 → f x = 2^x + 1) →
  f⁻¹ 19 = 3 - 2 * (Real.log 3 / Real.log 2) :=
by
  intros h1 h2 h3
  -- Proof goes here
  sorry

end find_inverse_value_l104_104236


namespace remainder_of_division_l104_104049

theorem remainder_of_division :
  ∀ (x : ℝ), (3 * x ^ 5 - 2 * x ^ 3 + 5 * x - 8) % (x ^ 2 - 3 * x + 2) = 74 * x - 76 :=
by
  sorry

end remainder_of_division_l104_104049


namespace mary_total_zoom_time_l104_104280

noncomputable def timeSpentDownloadingMac : ℝ := 10
noncomputable def timeSpentDownloadingWindows : ℝ := 3 * timeSpentDownloadingMac
noncomputable def audioGlitchesCount : ℝ := 2
noncomputable def audioGlitchDuration : ℝ := 4
noncomputable def totalAudioGlitchTime : ℝ := audioGlitchesCount * audioGlitchDuration
noncomputable def videoGlitchDuration : ℝ := 6
noncomputable def totalGlitchTime : ℝ := totalAudioGlitchTime + videoGlitchDuration
noncomputable def glitchFreeTalkingTime : ℝ := 2 * totalGlitchTime

theorem mary_total_zoom_time : 
  timeSpentDownloadingMac + timeSpentDownloadingWindows + totalGlitchTime + glitchFreeTalkingTime = 82 :=
by sorry

end mary_total_zoom_time_l104_104280


namespace points_in_quadrants_l104_104393

theorem points_in_quadrants (x y : ℝ) (h_line : 4 * x + 7 * y = 28)
  (h_equidistant : |x| = |y|) : 
  (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0) :=
sorry

end points_in_quadrants_l104_104393


namespace nat_number_of_the_form_l104_104287

theorem nat_number_of_the_form (a b : ℕ) (h : ∃ (a b : ℕ), a * a * 3 + b * b * 32 = n) :
  ∃ (a' b' : ℕ), a' * a' * 3 + b' * b' * 32 = 97 * n  :=
  sorry

end nat_number_of_the_form_l104_104287


namespace cousin_cards_probability_l104_104264

variable {Isabella_cards : ℕ}
variable {Evan_cards : ℕ}
variable {total_cards : ℕ}

theorem cousin_cards_probability 
  (h1 : Isabella_cards = 8)
  (h2 : Evan_cards = 2)
  (h3 : total_cards = 10) :
  (8 / 10 * 2 / 9) + (2 / 10 * 8 / 9) = 16 / 45 :=
by
  sorry

end cousin_cards_probability_l104_104264


namespace part1_part2_l104_104269

open Nat

variable {a : ℕ → ℝ} -- Defining the arithmetic sequence
variable {S : ℕ → ℝ} -- Defining the sum of the first n terms
variable {m n p q : ℕ} -- Defining the positive integers m, n, p, q
variable {d : ℝ} -- The common difference

-- Conditions
axiom arithmetic_sequence_pos_terms : (∀ k, a k = a 1 + (k - 1) * d) ∧ ∀ k, a k > 0
axiom sum_of_first_n_terms : ∀ n, S n = (n * (2 * a 1 + (n - 1) * d)) / 2
axiom positive_common_difference : d > 0
axiom constraints_on_mnpq : n < p ∧ q < m ∧ m + n = p + q

-- Parts to prove
theorem part1 : a m * a n < a p * a q :=
by sorry

theorem part2 : S m + S n > S p + S q :=
by sorry

end part1_part2_l104_104269


namespace combination_value_l104_104063

theorem combination_value (m : ℕ) (h : (1 / (Nat.choose 5 m) - 1 / (Nat.choose 6 m) = 7 / (10 * Nat.choose 7 m))) : 
    Nat.choose 8 m = 28 := 
sorry

end combination_value_l104_104063


namespace reflection_matrix_over_line_y_eq_x_l104_104523

-- Define the reflection over the line y = x as a linear map
def reflection_over_y_eq_x (v : ℝ × ℝ) : ℝ × ℝ :=
  (v.2, v.1)  -- this swaps the coordinates (x, y) -> (y, x)

-- Define the matrix that corresponds to this reflection
def reflection_matrix := ![
  ![0, 1],
  ![1, 0]
]

theorem reflection_matrix_over_line_y_eq_x :
  ∀ v : ℝ × ℝ, reflection_over_y_eq_x v = matrix.vec_mul reflection_matrix v :=
by
  sorry

end reflection_matrix_over_line_y_eq_x_l104_104523


namespace min_value_PF_PA_l104_104694

open Classical

noncomputable section

def parabola_eq (x y : ℝ) : Prop := y^2 = 16 * x

def point_A : ℝ × ℝ := (1, 2)

def focus_F : ℝ × ℝ := (4, 0)  -- Focus of the given parabola y^2 = 16x

def distance (P1 P2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P1.1 - P2.1)^2 + (P1.2 - P2.2)^2)

def PF_PA (P : ℝ × ℝ) : ℝ :=
  distance P focus_F + distance P point_A

theorem min_value_PF_PA :
  ∃ P : ℝ × ℝ, parabola_eq P.1 P.2 ∧ PF_PA P = 5 :=
sorry

end min_value_PF_PA_l104_104694


namespace lower_right_square_is_one_l104_104307

open Matrix

def grid_initial : Matrix (Fin 5) (Fin 5) (Option ℕ) :=
  ![ ![ some 1, none, none, none, some 2 ],
     ![ none, some 3, none, none, none ],
     ![ some 5, none, some 4, none, none ],
     ![ none, none, some 1, some 3, none ],
     ![ none, none, none, none, none ] ]

def is_valid_grid (grid : Matrix (Fin 5) (Fin 5) (Option ℕ)) : Prop :=
  (∀ i, Finset.univ.map ⟨fun j => grid i j, sorry⟩ = {1, 2, 3, 4, 5}) ∧
  (∀ j, Finset.univ.map ⟨fun i => grid i j, sorry⟩ = {1, 2, 3, 4, 5})

theorem lower_right_square_is_one :
  ∃ grid : Matrix (Fin 5) (Fin 5) (Option ℕ),
  grid_initial ⊆ grid ∧
  is_valid_grid grid ∧
  grid ⟨4, sorry⟩ ⟨4, sorry⟩ = some 1 :=
sorry

end lower_right_square_is_one_l104_104307


namespace find_num_non_officers_l104_104129

-- Define the average salaries and number of officers
def avg_salary_employees : Int := 120
def avg_salary_officers : Int := 470
def avg_salary_non_officers : Int := 110
def num_officers : Int := 15

-- States the problem of finding the number of non-officers
theorem find_num_non_officers : ∃ N : Int,
(15 * 470 + N * 110 = (15 + N) * 120) ∧ N = 525 := 
by {
  sorry
}

end find_num_non_officers_l104_104129


namespace g_two_gt_one_third_g_n_gt_one_third_l104_104696

def seq_a (n : ℕ) : ℕ := 3 * n - 2
noncomputable def f (n : ℕ) : ℝ := (Finset.range n).sum (λ i => 1 / (seq_a (i + 1) : ℝ))
noncomputable def g (n : ℕ) : ℝ := f (n^2) - f (n - 1)

theorem g_two_gt_one_third : g 2 > 1 / 3 :=
sorry

theorem g_n_gt_one_third (n : ℕ) (h : n ≥ 3) : g n > 1 / 3 :=
sorry

end g_two_gt_one_third_g_n_gt_one_third_l104_104696


namespace combined_share_a_c_l104_104659

-- Define the conditions
def total_money : ℕ := 15800
def ratio_a : ℕ := 5
def ratio_b : ℕ := 9
def ratio_c : ℕ := 6
def ratio_d : ℕ := 5

-- The total parts in the ratio
def total_parts : ℕ := ratio_a + ratio_b + ratio_c + ratio_d

-- The value of each part
def value_per_part : ℕ := total_money / total_parts

-- The shares of a and c
def share_a : ℕ := ratio_a * value_per_part
def share_c : ℕ := ratio_c * value_per_part

-- Prove that the combined share of a + c equals 6952
theorem combined_share_a_c : share_a + share_c = 6952 :=
by
  -- This is the proof placeholder
  sorry

end combined_share_a_c_l104_104659


namespace chord_length_l104_104761

theorem chord_length (x y : ℝ) :
  (x^2 + y^2 - 2 * x - 4 * y = 0) →
  (x + 2 * y - 5 + Real.sqrt 5 = 0) →
  ∃ l, l = 4 :=
by
  intros h_circle h_line
  sorry

end chord_length_l104_104761


namespace starting_number_divisible_by_3_count_l104_104437

-- Define a predicate for divisibility by 3
def divisible_by_3 (n : ℕ) : Prop :=
  n % 3 = 0

-- Define the main theorem
theorem starting_number_divisible_by_3_count : 
  ∃ n : ℕ, (∀ m, n ≤ m ∧ m ≤ 50 → divisible_by_3 m → ∃ s, (m = n + 3 * s) ∧ s < 13) ∧
           (∀ k : ℕ, (divisible_by_3 k) → n ≤ k ∧ k ≤ 50 → m = 12) :=
sorry

end starting_number_divisible_by_3_count_l104_104437


namespace power_sum_int_l104_104892

theorem power_sum_int {x : ℝ} (hx : ∃ k : ℤ, x + 1/x = k) : ∀ n : ℕ, ∃ m : ℤ, x^n + 1/x^n = m :=
by
  sorry

end power_sum_int_l104_104892


namespace reflection_matrix_correct_l104_104525

-- Definitions based on the conditions given in the problem
def reflect_over_line_y_eq_x_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![0, 1], ![1, 0]]

-- The main theorem to state the equivalence
theorem reflection_matrix_correct :
  reflect_over_line_y_eq_x_matrix = ![![0, 1], ![1, 0]] :=
by
  sorry

end reflection_matrix_correct_l104_104525


namespace solve_trig_eq_l104_104291

theorem solve_trig_eq :
  ∀ x k : ℤ, 
    (x = 2 * π / 3 + 2 * k * π ∨
     x = 7 * π / 6 + 2 * k * π ∨
     x = -π / 6 + 2 * k * π)
    → (|Real.cos x| + Real.cos (3 * x)) / (Real.sin x * Real.cos (2 * x)) = -2 * Real.sqrt 3 := 
by
  intros x k h
  sorry

end solve_trig_eq_l104_104291


namespace slope_of_regression_line_l104_104551

variable (h : ℝ)
variable (t1 T1 t2 T2 t3 T3 : ℝ)

-- Given conditions.
axiom t2_is_equally_spaced : t2 = t1 + h
axiom t3_is_equally_spaced : t3 = t1 + 2 * h

theorem slope_of_regression_line :
  t2 = t1 + h →
  t3 = t1 + 2 * h →
  (T3 - T1) / (t3 - t1) = (T3 - T1) / ((t1 + 2 * h) - t1) := 
by
  sorry

end slope_of_regression_line_l104_104551


namespace sin_seven_pi_div_six_l104_104202

theorem sin_seven_pi_div_six : Real.sin (7 * Real.pi / 6) = -1 / 2 := 
  sorry

end sin_seven_pi_div_six_l104_104202


namespace dinner_seating_l104_104369

theorem dinner_seating (eight_people : Finset ℕ) (h_card : eight_people.card = 8) :
  ∃ S : Finset (Finset ℕ), S.card = 3360 ∧ ∀ s ∈ S, s.card = 6 := by
sorry

end dinner_seating_l104_104369


namespace brian_video_watching_time_l104_104505

theorem brian_video_watching_time :
  let catVideo : ℕ := 4
  let dogVideo : ℕ := 2 * catVideo
  let combinedCatDog : ℕ := catVideo + dogVideo
  let gorillaVideo : ℕ := 2 * combinedCatDog
  let totalTime : ℕ := catVideo + dogVideo + gorillaVideo
  totalTime = 36 :=
by
  -- Define the variables
  let catVideo := 4
  let dogVideo := 2 * catVideo
  let combinedCatDog := catVideo + dogVideo
  let gorillaVideo := 2 * combinedCatDog
  let totalTime := catVideo + dogVideo + gorillaVideo
  -- Combine all steps and assert the final value
  show totalTime = 36, from
    sorry -- Proof not implemented

end brian_video_watching_time_l104_104505


namespace orange_juice_production_correct_l104_104767

noncomputable def orangeJuiceProduction (total_oranges : Float) (export_percent : Float) (juice_percent : Float) : Float :=
  let remaining_oranges := total_oranges * (1 - export_percent / 100)
  let juice_oranges := remaining_oranges * (juice_percent / 100)
  Float.round (juice_oranges * 10) / 10

theorem orange_juice_production_correct :
  orangeJuiceProduction 8.2 30 40 = 2.3 := by
  sorry

end orange_juice_production_correct_l104_104767


namespace oliver_money_left_l104_104745

-- Definitions based on the given conditions
def initial_amount : ℕ := 9
def allowance_saved : ℕ := 5
def frisbee_cost : ℕ := 4
def puzzle_cost : ℕ := 3
def birthday_gift : ℕ := 8

-- The proof problem statement
theorem oliver_money_left : 
  initial_amount + allowance_saved - (frisbee_cost + puzzle_cost) + birthday_gift = 15 := 
by
  sorry

end oliver_money_left_l104_104745


namespace sin_seven_pi_div_six_l104_104203

theorem sin_seven_pi_div_six : Real.sin (7 * Real.pi / 6) = -1 / 2 := 
  sorry

end sin_seven_pi_div_six_l104_104203


namespace lcm_of_36_and_105_l104_104220

theorem lcm_of_36_and_105 : Nat.lcm 36 105 = 1260 := by
  sorry

end lcm_of_36_and_105_l104_104220


namespace delta_delta_delta_l104_104509

-- Define the function Δ
def Δ (N : ℝ) : ℝ := 0.4 * N + 2

-- Mathematical statement to be proved
theorem delta_delta_delta (x : ℝ) : Δ (Δ (Δ 72)) = 7.728 := by
  sorry

end delta_delta_delta_l104_104509


namespace circumscribed_inscribed_coincide_l104_104719

theorem circumscribed_inscribed_coincide
  {A B C D : Point} (h1 : dist A B = dist C D)
                      (h2 : dist B C = dist A D)
                      (h3 : dist A C = dist B D) :
  ∃ O : Point, is_circumsphere_center O A B C D ∧ is_insphere_center O A B C D :=
sorry

end circumscribed_inscribed_coincide_l104_104719


namespace original_dining_bill_l104_104656

theorem original_dining_bill (B : ℝ) (h1 : B * 1.15 / 5 = 48.53) : B = 211 := 
sorry

end original_dining_bill_l104_104656


namespace proof_P_B_given_A_l104_104438

noncomputable def number_of_activities : ℕ := 5

def event_A (a b : ℕ) (total_activities : ℕ) : Prop :=
  a ≠ b ∧ a < total_activities ∧ b < total_activities

def event_B (a b : ℕ) : Prop :=
  a = 1 ∧ b = 1

def P_B_given_A (n_A : ℕ) (n_AB : ℕ) : ℚ := n_AB / n_A

theorem proof_P_B_given_A :
  let total_activities := number_of_activities in
  let n_A := (total_activities * (total_activities - 1)) in
  let n_AB := 0 in
  P_B_given_A n_A n_AB = 2 / 5 :=
by
  sorry

end proof_P_B_given_A_l104_104438


namespace evaluate_expression_l104_104324

theorem evaluate_expression : 4 * (9 - 3)^2 - 8 = 136 := by
  sorry

end evaluate_expression_l104_104324


namespace total_dots_not_visible_eq_54_l104_104383

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

end total_dots_not_visible_eq_54_l104_104383


namespace handrail_length_nearest_tenth_l104_104803

noncomputable def handrail_length (rise : ℝ) (turn_degree : ℝ) (radius : ℝ) : ℝ :=
  let arc_length := (turn_degree / 360) * (2 * Real.pi * radius)
  Real.sqrt (rise^2 + arc_length^2)

theorem handrail_length_nearest_tenth
  (h_rise : rise = 12)
  (h_turn_degree : turn_degree = 180)
  (h_radius : radius = 3) : handrail_length rise turn_degree radius = 13.1 :=
  by
  sorry

end handrail_length_nearest_tenth_l104_104803


namespace sum_of_isosceles_angles_l104_104323

-- Define the vertex positions
def A : ℝ × ℝ := (Real.cos (45 * Real.pi / 180), Real.sin (45 * Real.pi / 180))
def B : ℝ × ℝ := (Real.cos (90 * Real.pi / 180), Real.sin (90 * Real.pi / 180))
def C (θ : ℝ) : ℝ × ℝ := (Real.cos (θ * Real.pi / 180), Real.sin (θ * Real.pi / 180))

-- Define a function to check if the triangle is isosceles
def is_isosceles (θ : ℝ) : Prop :=
  let AB := (A.1 - B.1)^2 + (A.2 - B.2)^2
  let AC := (A.1 - C θ.1)^2 + (A.2 - C θ.2)^2
  let BC := (B.1 - C θ.1)^2 + (B.2 - C θ.2)^2
  (AB = AC) ∨ (AB = BC) ∨ (AC = BC)

-- Define the main theorem
theorem sum_of_isosceles_angles :
  (∑ θ in {θ | is_isosceles θ ∧ 0 ≤ θ ∧ θ ≤ 360}.to_finset, θ) = 675 := 
begin
  sorry
end

end sum_of_isosceles_angles_l104_104323


namespace distinct_fan_count_l104_104647

def max_distinct_fans : Nat :=
  36

theorem distinct_fan_count (n : Nat) (r b : S) (paint_scheme : Fin n → bool) :
  (∀i, r ≠ b → (paint_scheme i = b ∨ paint_scheme i = r)) ∧ 
  (∀i, paint_scheme i ≠ paint_scheme (i + n / 2 % n)) →
  n = 6 →
  max_distinct_fans = 36 :=
by
  sorry

end distinct_fan_count_l104_104647


namespace num_sides_polygon_l104_104598

theorem num_sides_polygon (n : ℕ) (h : (n - 2) * 180 = 1260) : n = 9 := by
  sorry

end num_sides_polygon_l104_104598


namespace rice_yield_l104_104496

theorem rice_yield (X : ℝ) (h1 : 0 ≤ X ∧ X ≤ 40) :
    0.75 * 400 * X + 0.25 * 800 * X + 500 * (40 - X) = 20000 := by
  sorry

end rice_yield_l104_104496


namespace no_solution_fractional_eq_l104_104580

theorem no_solution_fractional_eq (y : ℝ) (h : y ≠ 3) : 
  ¬ ( (y-2)/(y-3) = 2 - 1/(3-y) ) :=
by
  sorry

end no_solution_fractional_eq_l104_104580


namespace total_gallons_in_tanks_l104_104104

theorem total_gallons_in_tanks (
  tank1_cap : ℕ := 7000) (tank2_cap : ℕ := 5000) (tank3_cap : ℕ := 3000)
  (fill1_fraction : ℚ := 3/4) (fill2_fraction : ℚ := 4/5) (fill3_fraction : ℚ := 1/2)
  : tank1_cap * fill1_fraction + tank2_cap * fill2_fraction + tank3_cap * fill3_fraction = 10750 := by
  sorry

end total_gallons_in_tanks_l104_104104


namespace medicine_supply_duration_l104_104555

noncomputable def pillDuration (numPills : ℕ) (pillFractionPerThreeDays : ℚ) : ℚ :=
  let pillPerDay := pillFractionPerThreeDays / 3
  let daysPerPill := 1 / pillPerDay
  numPills * daysPerPill

theorem medicine_supply_duration (numPills : ℕ) (pillFractionPerThreeDays : ℚ) (daysPerMonth : ℚ) :
  numPills = 90 →
  pillFractionPerThreeDays = 1 / 3 →
  daysPerMonth = 30 →
  pillDuration numPills pillFractionPerThreeDays / daysPerMonth = 27 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  simp [pillDuration]
  sorry

end medicine_supply_duration_l104_104555


namespace sum_fractions_correct_l104_104354

def sum_of_fractions : Prop :=
  (3 / 15 + 5 / 150 + 7 / 1500 + 9 / 15000 = 0.2386)

theorem sum_fractions_correct : sum_of_fractions :=
by
  sorry

end sum_fractions_correct_l104_104354


namespace cricket_matches_total_l104_104756

theorem cricket_matches_total 
  (N : ℕ)
  (avg_total : ℕ → ℕ)
  (avg_first_8 : ℕ)
  (avg_last_4 : ℕ) 
  (h1 : avg_total N = 48)
  (h2 : avg_first_8 = 40)
  (h3 : avg_last_4 = 64) 
  (h_sum : (avg_first_8 * 8 + avg_last_4 * 4 = avg_total N * N)) :
  N = 12 := 
  sorry

end cricket_matches_total_l104_104756


namespace smallest_a_l104_104239

theorem smallest_a (x a : ℝ) (hx : x > 0) (ha : a > 0) (hineq : x + a / x ≥ 4) : a ≥ 4 :=
sorry

end smallest_a_l104_104239


namespace max_fans_theorem_l104_104639

noncomputable def max_distinct_fans : ℕ :=
  let num_sectors := 6
  let total_configurations := 2 ^ num_sectors
  -- Configurations unchanged by flipping
  let unchanged_configurations := 8
  -- Subtracting unchanged from total and then divide by 2 to account for symmetric duplicates
  -- then add back the unchanged configurations
  (total_configurations - unchanged_configurations) / 2 + unchanged_configurations

theorem max_fans_theorem : max_distinct_fans = 36 := by
  sorry

end max_fans_theorem_l104_104639


namespace partnership_total_annual_gain_l104_104804

theorem partnership_total_annual_gain 
  (x : ℝ) 
  (G : ℝ)
  (hA_investment : x * 12 = A_investment)
  (hB_investment : 2 * x * 6 = B_investment)
  (hC_investment : 3 * x * 4 = C_investment)
  (A_share : (A_investment / (A_investment + B_investment + C_investment)) * G = 6000) :
  G = 18000 := 
sorry

end partnership_total_annual_gain_l104_104804


namespace polynomial_operation_correct_l104_104665

theorem polynomial_operation_correct :
    ∀ (s t : ℝ), (s * t + 0.25 * s * t = 0) :=
by
  intros s t
  sorry

end polynomial_operation_correct_l104_104665


namespace exists_indices_l104_104837

theorem exists_indices (a : ℕ → ℕ) 
  (h_seq_perm : ∀ n, ∃ m, a m = n) : 
  ∃ ℓ m, 1 < ℓ ∧ ℓ < m ∧ a 1 + a m = 2 * a ℓ :=
by
  sorry

end exists_indices_l104_104837


namespace sqrt_sqrt_16_eq_pm2_l104_104137

theorem sqrt_sqrt_16_eq_pm2 (h : Real.sqrt 16 = 4) : Real.sqrt 4 = 2 ∨ Real.sqrt 4 = -2 :=
by
  sorry

end sqrt_sqrt_16_eq_pm2_l104_104137


namespace find_common_ratio_l104_104092

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a 1 * q ^ n

def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, S n = a 1 * (1 - q ^ n) / (1 - q)

variables (a : ℕ → ℝ) (S : ℕ → ℝ) (a1 q : ℝ)

noncomputable def a_5_condition : Prop :=
  a 5 = 2 * S 4 + 3

noncomputable def a_6_condition : Prop :=
  a 6 = 2 * S 5 + 3

theorem find_common_ratio (h1 : a_5_condition a S) (h2 : a_6_condition a S)
  (hg : geometric_sequence a q) (hs : sum_of_first_n_terms a S q) :
  q = 3 :=
sorry

end find_common_ratio_l104_104092


namespace sum_of_vertices_l104_104462

theorem sum_of_vertices (num_triangle num_hexagon : ℕ) (vertices_triangle vertices_hexagon : ℕ) :
  num_triangle = 1 → vertices_triangle = 3 →
  num_hexagon = 3 → vertices_hexagon = 6 →
  num_triangle * vertices_triangle + num_hexagon * vertices_hexagon = 21 :=
by
  intros h1 h2 h3 h4
  sorry

end sum_of_vertices_l104_104462


namespace tan_A_mul_tan_B_lt_one_l104_104408

theorem tan_A_mul_tan_B_lt_one (A B C : ℝ) (hC: C > 90) (hABC : A + B + C = 180) :
    Real.tan A * Real.tan B < 1 :=
sorry

end tan_A_mul_tan_B_lt_one_l104_104408


namespace sin_seven_pi_div_six_l104_104201

theorem sin_seven_pi_div_six : Real.sin (7 * Real.pi / 6) = -1 / 2 := 
  sorry

end sin_seven_pi_div_six_l104_104201


namespace reciprocal_of_neg_two_l104_104455

-- Define the condition of reciprocal
def is_reciprocal (y x : ℝ) : Prop := y * x = 1

-- Define the number in question
def y : ℝ := -2

-- Define the expected reciprocal
def x : ℝ := -1 / 2

-- State the theorem
theorem reciprocal_of_neg_two : is_reciprocal y x :=
by
  -- Proof can be provided here
  sorry

end reciprocal_of_neg_two_l104_104455


namespace min_points_in_symmetric_set_l104_104015

theorem min_points_in_symmetric_set (T : Set (ℝ × ℝ)) (h1 : ∀ {a b : ℝ}, (a, b) ∈ T → (a, -b) ∈ T)
                                      (h2 : ∀ {a b : ℝ}, (a, b) ∈ T → (-a, b) ∈ T)
                                      (h3 : ∀ {a b : ℝ}, (a, b) ∈ T → (-b, -a) ∈ T)
                                      (h4 : (1, 4) ∈ T) : 
    ∃ (S : Finset (ℝ × ℝ)), 
          (∀ p ∈ S, p ∈ T) ∧
          (∀ q ∈ T, ∃ p ∈ S, q = (p.1, p.2) ∨ q = (p.1, -p.2) ∨ q = (-p.1, p.2) ∨ q = (-p.1, -p.2) ∨ q = (-p.2, -p.1) ∨ q = (-p.2, p.1) ∨ q = (p.2, p.1) ∨ q = (p.2, -p.1)) ∧
          S.card = 8 := sorry

end min_points_in_symmetric_set_l104_104015


namespace sum_of_divisors_2000_l104_104574

theorem sum_of_divisors_2000 (n : ℕ) (h : n < 2000) :
  ∃ (s : Finset ℕ), (s ⊆ {1, 2, 4, 5, 8, 10, 16, 20, 25, 40, 50, 80, 100, 125, 200, 250, 400, 500, 1000, 2000}) ∧ s.sum id = n :=
by
  -- Proof goes here
  sorry

end sum_of_divisors_2000_l104_104574


namespace profit_margin_in_terms_of_retail_price_l104_104840

theorem profit_margin_in_terms_of_retail_price
  (k c P_R : ℝ) (h1 : ∀ C, P = k * C) (h2 : ∀ C, P_R = c * (P + C)) :
  P = (k / (c * (k + 1))) * P_R :=
by sorry

end profit_margin_in_terms_of_retail_price_l104_104840


namespace dorothy_money_left_l104_104511

def annual_income : ℝ := 60000
def tax_rate : ℝ := 0.18
def tax_amount : ℝ := annual_income * tax_rate
def money_left : ℝ := annual_income - tax_amount

theorem dorothy_money_left : money_left = 49200 := 
by
  sorry

end dorothy_money_left_l104_104511


namespace possible_values_of_a_plus_b_l104_104238

theorem possible_values_of_a_plus_b (a b : ℤ)
  (h1 : ∃ α : ℝ, 0 ≤ α ∧ α < 2 * Real.pi ∧ (∃ (sinα cosα : ℝ), sinα = Real.sin α ∧ cosα = Real.cos α ∧ (sinα + cosα = -a) ∧ (sinα * cosα = 2 * b^2))) :
  a + b = 1 ∨ a + b = -1 := 
sorry

end possible_values_of_a_plus_b_l104_104238


namespace tobias_distance_swum_l104_104924

def swimming_pool_duration : Nat := 3 * 60 -- Tobias at the pool for 3 hours in minutes.

def swimming_duration : Nat := 5  -- Time taken to swim 100 meters in minutes.

def pause_duration : Nat := 5  -- Pause duration after every 25 minutes of swimming.

def swim_interval : Nat := 25  -- Swimming interval before each pause in minutes.

def total_pause_time (total_minutes: Nat) (interval: Nat) (pause: Nat) : Nat :=
  (total_minutes / (interval + pause)) * pause

def total_swimming_time (total_minutes: Nat) (interval: Nat) (pause: Nat) : Nat :=
  total_minutes - total_pause_time(total_minutes, interval, pause)

def distance_swum (swimming_minutes: Nat) (swim_duration: Nat) : Nat :=
  (swimming_minutes / swim_duration) * 100

theorem tobias_distance_swum :
  distance_swum (total_swimming_time swimming_pool_duration swim_interval pause_duration) swimming_duration = 3000 :=
by
  -- perform the calculation steps indicated in the solution
  sorry

end tobias_distance_swum_l104_104924


namespace investor_amount_after_two_years_l104_104469

noncomputable def compound_interest
  (P : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem investor_amount_after_two_years :
  compound_interest 3000 0.10 1 2 = 3630 :=
by
  -- Calculation goes here
  sorry

end investor_amount_after_two_years_l104_104469


namespace sqrt_sqrt_16_eq_pm_2_l104_104139

theorem sqrt_sqrt_16_eq_pm_2 : sqrt (sqrt 16) = 2 ∨ sqrt (sqrt 16) = -2 := 
by
  sorry

end sqrt_sqrt_16_eq_pm_2_l104_104139


namespace sum_of_triangle_ops_l104_104122

def triangle_op (a b c : ℕ) : ℕ := 2 * a + b - c 

theorem sum_of_triangle_ops : 
  triangle_op 1 2 3 + triangle_op 4 6 5 + triangle_op 2 7 1 = 20 :=
by
  sorry

end sum_of_triangle_ops_l104_104122


namespace machine_working_time_l104_104501

theorem machine_working_time (shirts_per_minute : ℕ) (total_shirts : ℕ) (h1 : shirts_per_minute = 3) (h2 : total_shirts = 6) :
  (total_shirts / shirts_per_minute) = 2 :=
by
  -- Begin the proof
  sorry

end machine_working_time_l104_104501


namespace solve_trig_eq_l104_104292

open Real

theorem solve_trig_eq (k : ℤ) : 
  (∃ x : ℝ, 
    (|cos x| + cos (3 * x)) / (sin x * cos (2 * x)) = -2 * sqrt 3 
    ∧ (x = -π/6 + 2 * k * π ∨ x = 2 * π/3 + 2 * k * π ∨ x = 7 * π/6 + 2 * k * π)) :=
sorry

end solve_trig_eq_l104_104292


namespace triangle_base_l104_104295

theorem triangle_base (A h b : ℝ) (hA : A = 15) (hh : h = 6) (hbase : A = 0.5 * b * h) : b = 5 := by
  sorry

end triangle_base_l104_104295


namespace tina_mother_age_l104_104747

variable {x : ℕ}

theorem tina_mother_age (h1 : 10 + x = 2 * x - 20) : 2010 + x = 2040 :=
by 
  sorry

end tina_mother_age_l104_104747


namespace speaking_orders_count_l104_104443

open_locale big_operators

/-- The total number of different speaking orders for 4 speakers selected from 
8 candidates (including A and B) with the constraints that at least one of A 
and B must participate, and if both participate, exactly one person must speak 
between them, is 1080. -/
theorem speaking_orders_count :
  ∃ (A B : Type) (candidate_set : Finset A)
    (condition1 : candidate_set.card = 8) -- 8 candidates
    (condition2 : ∃ (subset : Finset A), subset.card = 4 ∧ subset ⊆ candidate_set), -- 4 speakers selected
    let participating_speakers_count (subset : Finset A) : Prop := 
      (A ∈ subset ∨ B ∈ subset) -- at least one of A or B participates
      ∧ (A ∈ subset ∧ B ∈ subset → (∃ x ∈ subset, (x ≠ A ∧ x ≠ B) ∧ ∀ y, y ∈ subset ∧ y ≠ A → y ≠ x ∧ y ≠ B)), -- exactly one person speaks between A and B
  true := sorry

end speaking_orders_count_l104_104443


namespace fatima_total_donation_l104_104680

theorem fatima_total_donation :
  let cloth1 := 100
  let cloth1_piece1 := 0.40 * cloth1
  let cloth1_piece2 := 0.30 * cloth1
  let cloth1_piece3 := 0.30 * cloth1
  let donation1 := cloth1_piece2 + cloth1_piece3

  let cloth2 := 65
  let cloth2_piece1 := 0.55 * cloth2
  let cloth2_piece2 := 0.45 * cloth2
  let donation2 := cloth2_piece2

  let cloth3 := 48
  let cloth3_piece1 := 0.60 * cloth3
  let cloth3_piece2 := 0.40 * cloth3
  let donation3 := cloth3_piece2

  donation1 + donation2 + donation3 = 108.45 :=
by
  sorry

end fatima_total_donation_l104_104680


namespace least_prime_factor_of_expr_l104_104470

theorem least_prime_factor_of_expr : ∀ n : ℕ, n = 11^5 - 11^2 → (∃ p : ℕ, Nat.Prime p ∧ p ≤ 2 ∧ p ∣ n) :=
by
  intros n h
  -- here will be proof steps, currently skipped
  sorry

end least_prime_factor_of_expr_l104_104470


namespace smallest_multiple_of_6_and_9_l104_104148

theorem smallest_multiple_of_6_and_9 : ∃ n : ℕ, n > 0 ∧ (n % 6 = 0) ∧ (n % 9 = 0) ∧ ∀ m : ℕ, m > 0 ∧ (m % 6 = 0) ∧ (m % 9 = 0) → n ≤ m :=
  by
    sorry

end smallest_multiple_of_6_and_9_l104_104148


namespace distinct_fan_count_l104_104646

def max_distinct_fans : Nat :=
  36

theorem distinct_fan_count (n : Nat) (r b : S) (paint_scheme : Fin n → bool) :
  (∀i, r ≠ b → (paint_scheme i = b ∨ paint_scheme i = r)) ∧ 
  (∀i, paint_scheme i ≠ paint_scheme (i + n / 2 % n)) →
  n = 6 →
  max_distinct_fans = 36 :=
by
  sorry

end distinct_fan_count_l104_104646


namespace range_of_absolute_difference_l104_104669

theorem range_of_absolute_difference : (∃ x : ℝ, y = |x + 4| - |x - 5|) → y ∈ [-9, 9] :=
sorry

end range_of_absolute_difference_l104_104669


namespace arithmetic_sequence_proof_l104_104275

variable (n : ℕ)
variable (a_n S_n : ℕ → ℤ)

noncomputable def a : ℕ → ℤ := 48 - 8 * n
noncomputable def S : ℕ → ℤ := -4 * (n ^ 2) + 44 * n

axiom a_3 : a 3 = 24
axiom S_11 : S 11 = 0

theorem arithmetic_sequence_proof :
  a n = 48 - 8 * n ∧
  S n = -4 * n ^ 2 + 44 * n ∧
  ∃ n, S n = 120 ∧ (n = 5 ∨ n = 6) :=
by
  unfold a S
  sorry

end arithmetic_sequence_proof_l104_104275


namespace find_a2015_l104_104334

def seq (a : ℕ → ℕ) :=
  (a 1 = 1) ∧
  (a 2 = 4) ∧
  (a 3 = 9) ∧
  (∀ n, 4 ≤ n → a n = a (n-1) + a (n-2) - a (n-3))

theorem find_a2015 (a : ℕ → ℕ) (h_seq : seq a) : a 2015 = 8057 :=
sorry

end find_a2015_l104_104334


namespace two_thousand_divisibility_l104_104042

theorem two_thousand_divisibility (n : ℕ) (hn : n > 3) :
  (∃ k : ℕ, k ≤ 2000 ∧ 2^k = (1 + n + nat.choose n 2 + nat.choose n 3)) →
  n = 7 ∨ n = 23 := 
by
  sorry

end two_thousand_divisibility_l104_104042


namespace min_value_expr_l104_104686

theorem min_value_expr (x : ℝ) (hx : x > 0) : 4 * x + 1 / x^2 ≥ 5 :=
by
  sorry

end min_value_expr_l104_104686


namespace number_of_division_games_l104_104943

theorem number_of_division_games (N M : ℕ) (h1 : N > 2 * M) (h2 : M > 5) (h3 : 4 * N + 5 * M = 100) :
  4 * N = 60 :=
by
  sorry

end number_of_division_games_l104_104943


namespace sum_of_first_7_terms_l104_104560

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

theorem sum_of_first_7_terms (h1 : a 2 = 3) (h2 : a 6 = 11)
  (h3 : ∀ n, S n = n * (a 1 + a n) / 2) : S 7 = 49 :=
by 
  sorry

end sum_of_first_7_terms_l104_104560


namespace domain_log_function_l104_104909

theorem domain_log_function :
  { x : ℝ | 12 + x - x^2 > 0 } = { x : ℝ | -3 < x ∧ x < 4 } :=
sorry

end domain_log_function_l104_104909


namespace evaluate_g_at_neg2_l104_104422

-- Definition of the polynomial g
def g (x : ℝ) : ℝ := 3 * x^5 - 20 * x^4 + 40 * x^3 - 25 * x^2 - 75 * x + 90

-- Statement to prove using the condition
theorem evaluate_g_at_neg2 : g (-2) = -596 := 
by 
   sorry

end evaluate_g_at_neg2_l104_104422


namespace max_score_exam_l104_104088

theorem max_score_exam (Gibi_percent Jigi_percent Mike_percent Lizzy_percent : ℝ)
  (avg_score total_score M : ℝ) :
  Gibi_percent = 0.59 →
  Jigi_percent = 0.55 →
  Mike_percent = 0.99 →
  Lizzy_percent = 0.67 →
  avg_score = 490 →
  total_score = avg_score * 4 →
  total_score = (Gibi_percent + Jigi_percent + Mike_percent + Lizzy_percent) * M →
  M = 700 :=
by
  intros hGibi hJigi hMike hLizzy hAvg hTotalScore hEq
  sorry

end max_score_exam_l104_104088


namespace division_equals_fraction_l104_104352

theorem division_equals_fraction:
  180 / (8 + 9 * 3 - 4) = 180 / 31 := 
by
  sorry

end division_equals_fraction_l104_104352


namespace dmitriev_older_by_10_l104_104607

-- Define the ages of each of the elders
variables (A B C D E F : ℕ)

-- The conditions provided in the problem
axiom hAlyosha : A > (A - 1)
axiom hBorya : B > (B - 2)
axiom hVasya : C > (C - 3)
axiom hGrisha : D > (D - 4)

-- Establishing an equation for the age differences leading to the proof
axiom age_sum_relation : A + B + C + D + E = (A - 1) + (B - 2) + (C - 3) + (D - 4) + F

-- We state that Dmitriev is older than Dima by 10 years
theorem dmitriev_older_by_10 : F = E + 10 :=
by
  -- sorry replaces the proof
  sorry

end dmitriev_older_by_10_l104_104607


namespace reflection_matrix_over_y_eq_x_is_correct_l104_104524

theorem reflection_matrix_over_y_eq_x_is_correct :
  let M := matrix.std_basis (fin 2) (fin 2)
  ∃ (R : matrix (fin 2) (fin 2) ℝ), 
    (R ⬝ M 0) = matrix.vec_cons 0 (matrix.vec_cons 1 matrix.vec_empty) ∧
    (R ⬝ M 1) = matrix.vec_cons 1 (matrix.vec_cons 0 matrix.vec_empty) ∧
    R = ![![0, 1], ![1, 0]] :=
sorry

end reflection_matrix_over_y_eq_x_is_correct_l104_104524


namespace job_completion_time_l104_104474

theorem job_completion_time (A_rate D_rate Combined_rate : ℝ) (hA : A_rate = 1 / 3) (hD : D_rate = 1 / 6) (hCombined : Combined_rate = A_rate + D_rate) :
  (1 / Combined_rate) = 2 :=
by sorry

end job_completion_time_l104_104474


namespace necessary_but_not_sufficient_l104_104061

def p (x : ℝ) : Prop := x < 1
def q (x : ℝ) : Prop := x^2 + x - 2 < 0

theorem necessary_but_not_sufficient (x : ℝ):
  (p x → q x) ∧ (q x → p x) → False ∧ (q x → p x) :=
sorry

end necessary_but_not_sufficient_l104_104061


namespace total_gallons_in_tanks_l104_104106

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

end total_gallons_in_tanks_l104_104106


namespace find_fraction_l104_104682

noncomputable def some_fraction_of_number_is (N f : ℝ) : Prop :=
  1 + f * N = 0.75 * N

theorem find_fraction (N : ℝ) (hN : N = 12.0) :
  ∃ f : ℝ, some_fraction_of_number_is N f ∧ f = 2 / 3 :=
by
  sorry

end find_fraction_l104_104682


namespace integer_values_satisfying_sqrt_condition_l104_104765

theorem integer_values_satisfying_sqrt_condition : ∃! n : Nat, 2.5 < Real.sqrt n ∧ Real.sqrt n < 3.5 :=
by {
  sorry -- Proof to be filled in
}

end integer_values_satisfying_sqrt_condition_l104_104765


namespace equal_roots_h_l104_104078

theorem equal_roots_h (h : ℝ) : (∀ x : ℝ, 3 * x^2 - 4 * x + h / 3 = 0) ↔ h = 4 := by
  -- proof goes here
  sorry

end equal_roots_h_l104_104078


namespace chord_square_length_l104_104357

theorem chord_square_length
    (r1 r2 r3 L1 L2 L3 : ℝ)
    (h1 : r1 = 4) 
    (h2 : r2 = 8) 
    (h3 : r3 = 12) 
    (tangent1 : ∀ x, (L1 - x)^2 + (L2 - x)^2 = (r1 + r2)^2)
    (tangent2 : ∀ x, x^2 + (L3 - x)^2 = (r3 - r2)^2) 
    (tangent3 : ∀ x, x^2 + (L3 - x)^2 = (r3 - r1)^2) : L1^2 = 3584 / 9 :=
by
  sorry

end chord_square_length_l104_104357


namespace alex_candles_left_l104_104167

theorem alex_candles_left (candles_start used_candles : ℕ) (h1 : candles_start = 44) (h2 : used_candles = 32) :
  candles_start - used_candles = 12 :=
by
  sorry

end alex_candles_left_l104_104167


namespace extinction_probability_l104_104799

-- Definitions from conditions
def prob_divide : ℝ := 0.6
def prob_die : ℝ := 0.4

-- Statement of the theorem
theorem extinction_probability :
  ∃ (v : ℝ), v = 2 / 3 :=
by
  sorry

end extinction_probability_l104_104799


namespace second_order_det_example_l104_104346

theorem second_order_det_example : matrix.det ![![2, 1], ![-3, 4]] = 11 := by
  sorry

end second_order_det_example_l104_104346


namespace sufficient_but_not_necessary_condition_l104_104232

theorem sufficient_but_not_necessary_condition 
(a b : ℝ) : (b ≥ 0) → ((a + 1)^2 + b ≥ 0) ∧ (¬ (∀ a b, ((a + 1)^2 + b ≥ 0) → b ≥ 0)) :=
by sorry

end sufficient_but_not_necessary_condition_l104_104232


namespace annie_accident_chance_l104_104805

def temperature_effect (temp: ℤ) : ℚ := ((32 - temp) / 3 * 5)

def road_condition_effect (condition: ℚ) : ℚ := condition

def wind_speed_effect (speed: ℤ) : ℚ := if (speed > 20) then ((speed - 20) / 10 * 3) else 0

def skid_chance (temp: ℤ) (condition: ℚ) (speed: ℤ) : ℚ :=
  temperature_effect temp + road_condition_effect condition + wind_speed_effect speed

def accident_chance (skid_chance: ℚ) (tire_effect: ℚ) : ℚ :=
  skid_chance * tire_effect

theorem annie_accident_chance :
  (temperature_effect 8 + road_condition_effect 15 + wind_speed_effect 35) * 0.75 = 43.5 :=
by sorry

end annie_accident_chance_l104_104805


namespace compound_interest_years_l104_104830

-- Definitions for the given conditions
def principal : ℝ := 1200
def rate : ℝ := 0.20
def compound_interest : ℝ := 873.60
def compounded_yearly : ℝ := 1

-- Calculate the future value from principal and compound interest
def future_value : ℝ := principal + compound_interest

-- Statement of the problem: Prove that the number of years t was 3 given the conditions
theorem compound_interest_years :
  ∃ (t : ℝ), future_value = principal * (1 + rate / compounded_yearly)^(compounded_yearly * t) := sorry

end compound_interest_years_l104_104830


namespace polygon_sides_sum_l104_104596

theorem polygon_sides_sum (n : ℕ) (h : (n - 2) * 180 = 1260) : n = 9 :=
by
  sorry

end polygon_sides_sum_l104_104596


namespace cylinder_volume_expansion_l104_104134

theorem cylinder_volume_expansion (r h : ℝ) :
  (π * (2 * r)^2 * h) = 4 * (π * r^2 * h) :=
by
  sorry

end cylinder_volume_expansion_l104_104134


namespace tangent_line_eq_l104_104376

noncomputable def f (x : ℝ) := x / (2 * x - 1)

def tangentLineAtPoint (x : ℝ) : ℝ := -x + 2

theorem tangent_line_eq {x y : ℝ} (hxy : y = f 1) (f_deriv : deriv f 1 = -1) :
  y = 1 → tangentLineAtPoint x = -x + 2 :=
by
  intros
  sorry

end tangent_line_eq_l104_104376


namespace area_of_inscribed_square_l104_104488

theorem area_of_inscribed_square
    (r : ℝ)
    (h : ∀ A : ℝ × ℝ, (A.1 = r - 1 ∨ A.1 = -(r - 1)) ∧ (A.2 = r - 2 ∨ A.2 = -(r - 2)) → A.1^2 + A.2^2 = r^2) :
    4 * r^2 = 100 := by
  -- proof would go here
  sorry

end area_of_inscribed_square_l104_104488

import Mathlib

namespace find_a_minus_b_plus_c_l2021_202114

def a_n (n : ℕ) : ℕ := 4 * n - 3

def S_n (a b c n : ℕ) : ℕ := 2 * a * n ^ 2 + b * n + c

theorem find_a_minus_b_plus_c
  (a b c : ℕ)
  (h : ∀ n : ℕ, n > 0 → S_n a b c n = 2 * n ^ 2 - n)
  : a - b + c = 2 :=
by
  sorry

end find_a_minus_b_plus_c_l2021_202114


namespace cubic_eq_real_roots_roots_product_eq_neg_nine_l2021_202133

theorem cubic_eq_real_roots :
  (∀ x, 0 ≤ x ∧ x ≤ Real.sqrt 3 →
    abs (x^3 + (3 / 2) * (1 - a) * x^2 - 3 * a * x + b) ≤ 1) →
  (∃ x1 x2 x3 : ℝ, x1 < x2 ∧ x2 < x3 ∧ 
    x1^3 + (3 / 2) * (1 - a) * x1^2 - 3 * a * x1 + b = 0 ∧
    x2^3 + (3 / 2) * (1 - a) * x2^2 - 3 * a * x2 + b = 0 ∧
    x3^3 + (3 / 2) * (1 - a) * x3^2 - 3 * a * x3 + b = 0) :=
sorry

theorem roots_product_eq_neg_nine :
  let a := 1
  let b := 1
  (∀ x, 0 ≤ x ∧ x ≤ Real.sqrt 3 →
    abs (x^3 + (3 / 2) * (1 - a) * x^2 - 3 * a * x + b) ≤ 1) →
  (∃ x1 x2 x3 : ℝ, x1 < x2 ∧ x2 < x3 ∧ 
    x1^3 - 3 * x1 + 1 = 0 ∧
    x2^3 - 3 * x2 + 1 = 0 ∧
    x3^3 - 3 * x3 + 1 = 0 ∧
    (x1^2 - 2 - x2) * (x2^2 - 2 - x3) * (x3^2 - 2 - x1) = -9) :=
sorry

end cubic_eq_real_roots_roots_product_eq_neg_nine_l2021_202133


namespace solve_fraction_eq_l2021_202157

theorem solve_fraction_eq (x : ℝ) (h₁ : x ≠ 2) (h₂ : x ≠ 3) 
    (h₃ : 3 / (x - 2) = 6 / (x - 3)) : x = 1 :=
by 
  sorry

end solve_fraction_eq_l2021_202157


namespace smallest_reducible_fraction_l2021_202178

theorem smallest_reducible_fraction :
  ∃ n : ℕ, 0 < n ∧ (∃ d > 1, d ∣ (n - 17) ∧ d ∣ (7 * n + 8)) ∧ n = 144 := by
  sorry

end smallest_reducible_fraction_l2021_202178


namespace oldest_child_age_l2021_202192

theorem oldest_child_age 
  (x : ℕ)
  (h1 : (6 + 8 + 10 + x) / 4 = 9)
  (h2 : 6 + 8 + 10 = 24) :
  x = 12 := 
by 
  sorry

end oldest_child_age_l2021_202192


namespace diagonal_length_l2021_202113

noncomputable def rectangle_diagonal (p : ℝ) (r : ℝ) (d : ℝ) : Prop :=
  ∃ k : ℝ, p = 2 * ((5 * k) + (2 * k)) ∧ r = 5 / 2 ∧ 
           d = Real.sqrt (((5 * k)^2 + (2 * k)^2)) 

theorem diagonal_length 
  (p : ℝ) (r : ℝ) (d : ℝ)
  (h₁ : p = 72) 
  (h₂ : r = 5 / 2)
  : rectangle_diagonal p r d ↔ d = 194 / 7 := 
sorry

end diagonal_length_l2021_202113


namespace necessary_but_not_sufficient_condition_l2021_202121

theorem necessary_but_not_sufficient_condition (x : ℝ) :
  ((0 < x ∧ x < 5) → (|x - 2| < 3)) ∧ ¬ ((|x - 2| < 3) → (0 < x ∧ x < 5)) :=
by
  sorry

end necessary_but_not_sufficient_condition_l2021_202121


namespace total_earnings_l2021_202105

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

end total_earnings_l2021_202105


namespace find_a_l2021_202196

theorem find_a (a : ℝ) (ha : a ≠ 0)
  (h_area : (1/2) * (a/2) * a^2 = 2) :
  a = 2 ∨ a = -2 :=
sorry

end find_a_l2021_202196


namespace mean_of_remaining_number_is_2120_l2021_202117

theorem mean_of_remaining_number_is_2120 (a1 a2 a3 a4 a5 a6 : ℕ) 
    (h1 : a1 = 1451) (h2 : a2 = 1723) (h3 : a3 = 1987) (h4 : a4 = 2056) 
    (h5 : a5 = 2191) (h6 : a6 = 2212) 
    (mean_five : (a1 + a2 + a3 + a4 + a5) = 9500):
-- Prove that the mean of the remaining number a6 is 2120
  (a6 = 2120) :=
by
  -- Placeholder for proof
  sorry

end mean_of_remaining_number_is_2120_l2021_202117


namespace brownie_pieces_count_l2021_202128

theorem brownie_pieces_count :
  let tray_length := 24
  let tray_width := 20
  let brownie_length := 3
  let brownie_width := 2
  let tray_area := tray_length * tray_width
  let brownie_area := brownie_length * brownie_width
  let pieces_count := tray_area / brownie_area
  pieces_count = 80 :=
by
  let tray_length := 24
  let tray_width := 20
  let brownie_length := 3
  let brownie_width := 2
  let tray_area := 24 * 20
  let brownie_area := 3 * 2
  let pieces_count := tray_area / brownie_area
  have h1 : tray_length * tray_width = 480 := by norm_num
  have h2 : brownie_length * brownie_width = 6 := by norm_num
  have h3 : pieces_count = 80 := by norm_num
  exact h3

end brownie_pieces_count_l2021_202128


namespace simplify_expression_l2021_202158

theorem simplify_expression : (245^2 - 225^2) / 20 = 470 := by
  sorry

end simplify_expression_l2021_202158


namespace swimmer_distance_l2021_202136

noncomputable def effective_speed := 4.4 - 2.5
noncomputable def time := 3.684210526315789
noncomputable def distance := effective_speed * time

theorem swimmer_distance :
  distance = 7 := by
  sorry

end swimmer_distance_l2021_202136


namespace non_positive_sequence_l2021_202181

theorem non_positive_sequence
  (N : ℕ)
  (a : ℕ → ℝ)
  (h₀ : a 0 = 0)
  (hN : a N = 0)
  (h_rec : ∀ i, 1 ≤ i ∧ i ≤ N - 1 → a (i + 1) - 2 * a i + a (i - 1) = a i ^ 2) :
  ∀ i, 1 ≤ i ∧ i ≤ N - 1 → a i ≤ 0 := sorry

end non_positive_sequence_l2021_202181


namespace Balint_claim_impossible_l2021_202106

-- Declare the lengths of the ladders and the vertical projection distance
def AC : ℝ := 3
def BD : ℝ := 2
def E_proj : ℝ := 1

-- State the problem conditions and what we need to prove
theorem Balint_claim_impossible (h1 : AC = 3) (h2 : BD = 2) (h3 : E_proj = 1) :
  False :=
  sorry

end Balint_claim_impossible_l2021_202106


namespace expand_polynomial_l2021_202171

theorem expand_polynomial (t : ℝ) : (2 * t^3 - 3 * t + 2) * (-3 * t^2 + 3 * t - 5) = 
  -6 * t^5 + 6 * t^4 - t^3 + 3 * t^2 + 21 * t - 10 :=
by
  sorry

end expand_polynomial_l2021_202171


namespace base_b_of_200_has_5_digits_l2021_202154

theorem base_b_of_200_has_5_digits : ∃ (b : ℕ), (b^4 ≤ 200) ∧ (200 < b^5) ∧ (b = 3) := by
  sorry

end base_b_of_200_has_5_digits_l2021_202154


namespace car_travel_distance_l2021_202104

noncomputable def car_distance_in_30_minutes : ℝ := 
  let train_speed : ℝ := 96
  let car_speed : ℝ := (5 / 8) * train_speed
  let travel_time : ℝ := 0.5  -- 30 minutes is 0.5 hours
  car_speed * travel_time

theorem car_travel_distance : car_distance_in_30_minutes = 30 := by
  sorry

end car_travel_distance_l2021_202104


namespace total_amount_before_brokerage_l2021_202162

variable (A : ℝ)

theorem total_amount_before_brokerage 
  (cash_realized : ℝ) 
  (brokerage_rate : ℝ) 
  (h1 : cash_realized = 106.25) 
  (h2 : brokerage_rate = 1 / 400) :
  A = 42500 / 399 :=
by
  sorry

end total_amount_before_brokerage_l2021_202162


namespace white_seeds_per_slice_l2021_202185

theorem white_seeds_per_slice (W : ℕ) (black_seeds_per_slice : ℕ) (number_of_slices : ℕ) 
(total_seeds : ℕ) (total_black_seeds : ℕ) (total_white_seeds : ℕ) 
(h1 : black_seeds_per_slice = 20)
(h2 : number_of_slices = 40)
(h3 : total_seeds = 1600)
(h4 : total_black_seeds = black_seeds_per_slice * number_of_slices)
(h5 : total_white_seeds = total_seeds - total_black_seeds)
(h6 : W = total_white_seeds / number_of_slices) :
W = 20 :=
by
  sorry

end white_seeds_per_slice_l2021_202185


namespace find_sum_a7_a8_l2021_202125

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a1 q : ℝ), ∀ n : ℕ, a n = a1 * q ^ n

variable (a : ℕ → ℝ)

axiom h_geom : geometric_sequence a
axiom h1 : a 0 + a 1 = 16
axiom h2 : a 2 + a 3 = 32

theorem find_sum_a7_a8 : a 6 + a 7 = 128 :=
sorry

end find_sum_a7_a8_l2021_202125


namespace working_light_bulbs_count_l2021_202141

def lamps := 60
def bulbs_per_lamp := 7

def fraction_with_2_burnt := 1 / 3
def fraction_with_1_burnt := 1 / 4
def fraction_with_3_burnt := 1 / 5

def lamps_with_2_burnt := fraction_with_2_burnt * lamps
def lamps_with_1_burnt := fraction_with_1_burnt * lamps
def lamps_with_3_burnt := fraction_with_3_burnt * lamps
def lamps_with_all_working := lamps - (lamps_with_2_burnt + lamps_with_1_burnt + lamps_with_3_burnt)

def working_bulbs_from_2_burnt := lamps_with_2_burnt * (bulbs_per_lamp - 2)
def working_bulbs_from_1_burnt := lamps_with_1_burnt * (bulbs_per_lamp - 1)
def working_bulbs_from_3_burnt := lamps_with_3_burnt * (bulbs_per_lamp - 3)
def working_bulbs_from_all_working := lamps_with_all_working * bulbs_per_lamp

def total_working_bulbs := working_bulbs_from_2_burnt + working_bulbs_from_1_burnt + working_bulbs_from_3_burnt + working_bulbs_from_all_working

theorem working_light_bulbs_count : total_working_bulbs = 329 := by
  sorry

end working_light_bulbs_count_l2021_202141


namespace find_quotient_l2021_202152

theorem find_quotient (divisor remainder dividend : ℕ) (h_divisor : divisor = 24) (h_remainder : remainder = 5) (h_dividend : dividend = 1565) : 
  (dividend - remainder) / divisor = 65 :=
by
  sorry

end find_quotient_l2021_202152


namespace total_pages_in_book_l2021_202130

-- Given conditions
def pages_first_chapter : ℕ := 13
def pages_second_chapter : ℕ := 68

-- The theorem to prove the total number of pages in the book
theorem total_pages_in_book :
  pages_first_chapter + pages_second_chapter = 81 := by
  sorry

end total_pages_in_book_l2021_202130


namespace cyclic_sum_inequality_l2021_202107

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

end cyclic_sum_inequality_l2021_202107


namespace arithmetic_sequence_sum_l2021_202111

theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h_arith : ∀ n, a (n + 1) - a n = a 1 - a 0) 
  (h_sum : a 3 + a 9 + a 15 + a 21 = 8) :
  a 1 + a 23 = 4 :=
sorry

end arithmetic_sequence_sum_l2021_202111


namespace speed_of_river_l2021_202187

theorem speed_of_river (speed_still_water : ℝ) (total_time : ℝ) (total_distance : ℝ) 
  (h_still_water: speed_still_water = 6) 
  (h_total_time: total_time = 1) 
  (h_total_distance: total_distance = 16/3) : 
  ∃ (speed_river : ℝ), speed_river = 2 :=
by 
  -- sorry is used to skip the proof
  sorry

end speed_of_river_l2021_202187


namespace range_of_ϕ_l2021_202140

noncomputable def f (x : ℝ) (ϕ : ℝ) : ℝ := 2 * Real.sin (2 * x + ϕ) + 1

theorem range_of_ϕ (ϕ : ℝ) (h1 : abs ϕ ≤ Real.pi / 2) 
    (h2 : ∀ (x : ℝ), -Real.pi / 12 < x ∧ x < Real.pi / 3 → f x ϕ > 1) :
  Real.pi / 6 ≤ ϕ ∧ ϕ ≤ Real.pi / 3 :=
sorry

end range_of_ϕ_l2021_202140


namespace more_boys_after_initial_l2021_202186

theorem more_boys_after_initial (X Y Z : ℕ) (hX : X = 22) (hY : Y = 35) : Z = Y - X :=
by
  sorry

end more_boys_after_initial_l2021_202186


namespace solution_set_of_f_inequality_l2021_202108

variable {f : ℝ → ℝ}
variable (h1 : f 1 = 1)
variable (h2 : ∀ x, f' x < 1/2)

theorem solution_set_of_f_inequality :
  {x : ℝ | f (x^2) < x^2 / 2 + 1 / 2} = {x : ℝ | x < -1 ∨ 1 < x} :=
sorry

end solution_set_of_f_inequality_l2021_202108


namespace mike_marbles_l2021_202122

theorem mike_marbles (original : ℕ) (given : ℕ) (final : ℕ) 
  (h1 : original = 8) 
  (h2 : given = 4)
  (h3 : final = original - given) : 
  final = 4 :=
by sorry

end mike_marbles_l2021_202122


namespace problem_solution_l2021_202160

-- Define the structure of the dartboard and scoring
structure Dartboard where
  inner_radius : ℝ
  intermediate_radius : ℝ
  outer_radius : ℝ
  regions : List (List ℤ) -- List of lists representing scores in the regions

-- Define the probability calculation function
noncomputable def probability_odd_score (d : Dartboard) : ℚ := sorry

-- Define the specific dartboard with given conditions
def revised_dartboard : Dartboard :=
  { inner_radius := 4.5,
    intermediate_radius := 6.75,
    outer_radius := 9,
    regions := [[3, 2, 2], [2, 1, 1], [1, 1, 3]] }

-- The theorem to prove the solution to the problem
theorem problem_solution : probability_odd_score revised_dartboard = 265 / 855 :=
  sorry

end problem_solution_l2021_202160


namespace prize_winners_l2021_202194

theorem prize_winners (n : ℕ) (p1 p2 : ℝ) (h1 : n = 100) (h2 : p1 = 0.4) (h3 : p2 = 0.2) :
  ∃ winners : ℕ, winners = (p2 * (p1 * n)) ∧ winners = 8 :=
by
  sorry

end prize_winners_l2021_202194


namespace base7_divisible_by_5_l2021_202112

theorem base7_divisible_by_5 :
  ∃ (d : ℕ), (0 ≤ d ∧ d < 7) ∧ (344 * d + 56) % 5 = 0 ↔ d = 1 :=
by
  sorry

end base7_divisible_by_5_l2021_202112


namespace original_average_weight_l2021_202151

theorem original_average_weight 
  (W : ℝ)
  (h1 : 7 * W + 110 + 60 = 9 * 78) : 
  W = 76 := 
by
  sorry

end original_average_weight_l2021_202151


namespace revenue_correct_l2021_202127

def calculate_revenue : Real :=
  let pumpkin_pie_revenue := 4 * 8 * 5
  let custard_pie_revenue := 5 * 6 * 6
  let apple_pie_revenue := 3 * 10 * 4
  let pecan_pie_revenue := 2 * 12 * 7
  let cookie_revenue := 15 * 2
  let red_velvet_revenue := 6 * 8 * 9
  pumpkin_pie_revenue + custard_pie_revenue + apple_pie_revenue + pecan_pie_revenue + cookie_revenue + red_velvet_revenue

theorem revenue_correct : calculate_revenue = 1090 :=
by
  sorry

end revenue_correct_l2021_202127


namespace remainder_of_k_div_11_l2021_202172

theorem remainder_of_k_div_11 {k : ℕ} (hk1 : k % 5 = 2) (hk2 : k % 6 = 5)
  (hk3 : 0 ≤ k % 7 ∧ k % 7 < 7) (hk4 : k < 38) : (k % 11) = 6 := 
by
  sorry

end remainder_of_k_div_11_l2021_202172


namespace brendas_age_l2021_202190

theorem brendas_age (A B J : ℝ) 
  (h1 : A = 4 * B)
  (h2 : J = B + 7)
  (h3 : A = J) : 
  B = 7 / 3 := 
by 
  sorry

end brendas_age_l2021_202190


namespace tanya_total_sticks_l2021_202103

theorem tanya_total_sticks (n : ℕ) (h : n = 11) : 3 * (n * (n + 1) / 2) = 198 :=
by
  have H : n = 11 := h
  sorry

end tanya_total_sticks_l2021_202103


namespace problem_statement_l2021_202180

theorem problem_statement
  (x y : ℝ)
  (h1 : 4 * x + 2 * y = 12)
  (h2 : 2 * x + 4 * y = 20) :
  20 * x^2 + 24 * x * y + 20 * y^2 = 544 :=
  sorry

end problem_statement_l2021_202180


namespace probability_adjacent_vertices_dodecagon_l2021_202135

noncomputable def prob_adjacent_vertices_dodecagon : ℚ :=
  let total_vertices := 12
  let favorable_outcomes := 2  -- adjacent vertices per chosen vertex
  let total_outcomes := total_vertices - 1  -- choosing any other vertex
  favorable_outcomes / total_outcomes

theorem probability_adjacent_vertices_dodecagon :
  prob_adjacent_vertices_dodecagon = 2 / 11 := by
  sorry

end probability_adjacent_vertices_dodecagon_l2021_202135


namespace relationship_y1_y2_y3_l2021_202177

def on_hyperbola (x y k : ℝ) : Prop := y = k / x

theorem relationship_y1_y2_y3 (y1 y2 y3 k : ℝ) (h1 : on_hyperbola (-5) y1 k) (h2 : on_hyperbola (-1) y2 k) (h3 : on_hyperbola 2 y3 k) (hk : k > 0) :
  y2 < y1 ∧ y1 < y3 :=
sorry

end relationship_y1_y2_y3_l2021_202177


namespace popularity_order_is_correct_l2021_202137

noncomputable def fraction_liking_dodgeball := (13 : ℚ) / 40
noncomputable def fraction_liking_karaoke := (9 : ℚ) / 30
noncomputable def fraction_liking_magicshow := (17 : ℚ) / 60
noncomputable def fraction_liking_quizbowl := (23 : ℚ) / 120

theorem popularity_order_is_correct :
  (fraction_liking_dodgeball ≥ fraction_liking_karaoke) ∧
  (fraction_liking_karaoke ≥ fraction_liking_magicshow) ∧
  (fraction_liking_magicshow ≥ fraction_liking_quizbowl) ∧
  (fraction_liking_dodgeball ≠ fraction_liking_karaoke) ∧
  (fraction_liking_karaoke ≠ fraction_liking_magicshow) ∧
  (fraction_liking_magicshow ≠ fraction_liking_quizbowl) := by
  sorry

end popularity_order_is_correct_l2021_202137


namespace tina_work_time_l2021_202175

theorem tina_work_time (T : ℕ) (h1 : ∀ Ann_hours, Ann_hours = 9)
                       (h2 : ∀ Tina_worked_hours, Tina_worked_hours = 8)
                       (h3 : ∀ Ann_worked_hours, Ann_worked_hours = 3)
                       (h4 : (8 : ℚ) / T + (1 : ℚ) / 3 = 1) : T = 12 :=
by
  sorry

end tina_work_time_l2021_202175


namespace ratio_of_increase_to_current_l2021_202119

-- Define the constants for the problem
def current_deductible : ℝ := 3000
def increase_deductible : ℝ := 2000

-- State the theorem that needs to be proven
theorem ratio_of_increase_to_current : 
  (increase_deductible / current_deductible) = (2 / 3) :=
by sorry

end ratio_of_increase_to_current_l2021_202119


namespace a_plus_b_value_l2021_202176

noncomputable def find_a_plus_b (a b : ℕ) (h_neq : a ≠ b) (h_pos : 0 < a ∧ 0 < b) (h_eq : a^2 - b^2 = 2018 - 2 * a) : ℕ :=
  a + b

theorem a_plus_b_value {a b : ℕ} (h_neq : a ≠ b) (h_pos : 0 < a ∧ 0 < b) (h_eq : a^2 - b^2 = 2018 - 2 * a) : find_a_plus_b a b h_neq h_pos h_eq = 672 :=
  sorry

end a_plus_b_value_l2021_202176


namespace problem1_problem2_l2021_202191

-- Problem 1: Calculation
theorem problem1 :
  (1:Real) - 1^2 + Real.sqrt 12 + Real.sqrt (4 / 3) = -1 + (8 * Real.sqrt 3) / 3 :=
by
  sorry
  
-- Problem 2: Solve the equation 2x^2 - x - 1 = 0
theorem problem2 (x : Real) :
  (2 * x^2 - x - 1 = 0) → (x = -1/2 ∨ x = 1) :=
by
  sorry

end problem1_problem2_l2021_202191


namespace right_triangle_sides_l2021_202199

theorem right_triangle_sides (a b c : ℕ) (h1 : a < b) 
  (h2 : 2 * c / 2 = c) 
  (h3 : exists x y, (x + y = 8 ∧ a < b) ∨ (x + y = 9 ∧ a < b)) 
  (h4 : a^2 + b^2 = c^2) : 
  a = 3 ∧ b = 4 ∧ c = 5 := 
by
  sorry

end right_triangle_sides_l2021_202199


namespace total_cost_l2021_202102

def num_of_rings : ℕ := 2

def cost_per_ring : ℕ := 12

theorem total_cost : num_of_rings * cost_per_ring = 24 :=
by sorry

end total_cost_l2021_202102


namespace chestnuts_distribution_l2021_202170

theorem chestnuts_distribution:
  ∃ (chestnuts_Alya chestnuts_Valya chestnuts_Galya : ℕ),
    chestnuts_Alya + chestnuts_Valya + chestnuts_Galya = 70 ∧
    4 * chestnuts_Valya = 3 * chestnuts_Alya ∧
    6 * chestnuts_Galya = 7 * chestnuts_Alya ∧
    chestnuts_Alya = 24 ∧
    chestnuts_Valya = 18 ∧
    chestnuts_Galya = 28 :=
by {
  sorry
}

end chestnuts_distribution_l2021_202170


namespace packs_of_yellow_bouncy_balls_l2021_202195

-- Define the conditions and the question in Lean
variables (GaveAwayGreen : ℝ) (BoughtGreen : ℝ) (BouncyBallsPerPack : ℝ) (TotalKeptBouncyBalls : ℝ) (Y : ℝ)

-- Assume the given conditions
axiom cond1 : GaveAwayGreen = 4.0
axiom cond2 : BoughtGreen = 4.0
axiom cond3 : BouncyBallsPerPack = 10.0
axiom cond4 : TotalKeptBouncyBalls = 80.0

-- Define the theorem statement
theorem packs_of_yellow_bouncy_balls (h1 : GaveAwayGreen = 4.0) (h2 : BoughtGreen = 4.0) (h3 : BouncyBallsPerPack = 10.0) (h4 : TotalKeptBouncyBalls = 80.0) : Y = 8 :=
sorry

end packs_of_yellow_bouncy_balls_l2021_202195


namespace range_of_m_l2021_202124

theorem range_of_m (m : ℝ) :
  (m + 4 - 4)*(2 + 2 * m - 4) < 0 → 0 < m ∧ m < 1 :=
by
  sorry

end range_of_m_l2021_202124


namespace papers_delivered_to_sunday_only_houses_l2021_202165

-- Define the number of houses in the route and the days
def houses_in_route : ℕ := 100
def days_monday_to_saturday : ℕ := 6

-- Define the number of customers that do not get the paper on Sunday
def non_customers_sunday : ℕ := 10
def total_papers_per_week : ℕ := 720

-- Define the required number of papers delivered on Sunday to houses that only get the paper on Sunday
def papers_only_on_sunday : ℕ :=
  total_papers_per_week - (houses_in_route * days_monday_to_saturday) - (houses_in_route - non_customers_sunday)

theorem papers_delivered_to_sunday_only_houses : papers_only_on_sunday = 30 :=
by
  sorry

end papers_delivered_to_sunday_only_houses_l2021_202165


namespace crayons_total_l2021_202126

theorem crayons_total (blue_crayons : ℕ) (red_crayons : ℕ) 
  (H1 : red_crayons = 4 * blue_crayons) (H2 : blue_crayons = 3) : 
  blue_crayons + red_crayons = 15 := 
by
  sorry

end crayons_total_l2021_202126


namespace angle_B_triangle_perimeter_l2021_202169

variable {A B C a b c : Real}

-- Definitions and conditions for part 1
def sides_relation (a b c : ℝ) (A : ℝ) : Prop :=
  2 * c = a + 2 * b * Real.cos A

-- Definitions and conditions for part 2
def triangle_area (a b c : ℝ) (B : ℝ) : Prop :=
  (1 / 2) * a * c * Real.sin B = Real.sqrt 3

def side_b_value (b : ℝ) : Prop :=
  b = Real.sqrt 13

-- Theorem statement for part 1 
theorem angle_B (a b c A : ℝ) (h1: sides_relation a b c A) : B = Real.pi / 3 :=
sorry

-- Theorem statement for part 2 
theorem triangle_perimeter (a b c B : ℝ) (h1 : triangle_area a b c B) (h2 : side_b_value b) (h3 : B = Real.pi / 3) : a + b + c = 5 + Real.sqrt 13 :=
sorry

end angle_B_triangle_perimeter_l2021_202169


namespace heesu_has_greatest_sum_l2021_202184

theorem heesu_has_greatest_sum :
  let Sora_sum := 4 + 6
  let Heesu_sum := 7 + 5
  let Jiyeon_sum := 3 + 8
  Heesu_sum > Sora_sum ∧ Heesu_sum > Jiyeon_sum :=
by
  let Sora_sum := 4 + 6
  let Heesu_sum := 7 + 5
  let Jiyeon_sum := 3 + 8
  have h1 : Heesu_sum > Sora_sum := by sorry
  have h2 : Heesu_sum > Jiyeon_sum := by sorry
  exact And.intro h1 h2

end heesu_has_greatest_sum_l2021_202184


namespace fraction_sum_l2021_202174

variable {w x y : ℚ}  -- assuming w, x, and y are rational numbers

theorem fraction_sum (h1 : w / x = 1 / 3) (h2 : w / y = 2 / 3) : (x + y) / y = 3 :=
sorry

end fraction_sum_l2021_202174


namespace find_floor_at_same_time_l2021_202159

def timeTaya (n : ℕ) : ℕ := 15 * (n - 22)
def timeJenna (n : ℕ) : ℕ := 120 + 3 * (n - 22)

theorem find_floor_at_same_time (n : ℕ) : n = 32 :=
by
  -- The goal is to show that Taya and Jenna arrive at the same floor at the same time
  have ht : 15 * (n - 22) = timeTaya n := rfl
  have hj : 120 + 3 * (n - 22) = timeJenna n := rfl
  -- equate the times
  have h : timeTaya n = timeJenna n := by sorry
  -- solving the equation for n = 32
  sorry

end find_floor_at_same_time_l2021_202159


namespace sin_alpha_l2021_202167

theorem sin_alpha (α : ℝ) (hα : 0 < α ∧ α < π) (hcos : Real.cos (π + α) = 3 / 5) :
  Real.sin α = 4 / 5 :=
sorry

end sin_alpha_l2021_202167


namespace probability_of_green_apples_l2021_202132

def total_apples : ℕ := 8
def red_apples : ℕ := 5
def green_apples : ℕ := 3
def apples_chosen : ℕ := 3
noncomputable def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem probability_of_green_apples :
  (binomial green_apples apples_chosen : ℚ) / (binomial total_apples apples_chosen : ℚ) = 1 / 56 :=
  sorry

end probability_of_green_apples_l2021_202132


namespace blue_pill_cost_is_25_l2021_202189

variable (blue_pill_cost red_pill_cost : ℕ)

-- Clara takes one blue pill and one red pill each day for 10 days.
-- A blue pill costs $2 more than a red pill.
def pill_cost_condition (blue_pill_cost red_pill_cost : ℕ) : Prop :=
  blue_pill_cost = red_pill_cost + 2 ∧
  10 * blue_pill_cost + 10 * red_pill_cost = 480

-- Prove that the cost of one blue pill is $25.
theorem blue_pill_cost_is_25 (h : pill_cost_condition blue_pill_cost red_pill_cost) : blue_pill_cost = 25 :=
  sorry

end blue_pill_cost_is_25_l2021_202189


namespace at_least_one_nonnegative_l2021_202156

theorem at_least_one_nonnegative (x y z : ℝ) : 
  (x^2 + y + 1/4 ≥ 0) ∨ (y^2 + z + 1/4 ≥ 0) ∨ (z^2 + x + 1/4 ≥ 0) :=
sorry

end at_least_one_nonnegative_l2021_202156


namespace original_number_l2021_202148

theorem original_number (x : ℝ) (h : x - x / 3 = 36) : x = 54 :=
by
  sorry

end original_number_l2021_202148


namespace solve_for_x_l2021_202198

theorem solve_for_x (x y : ℝ) (h : x / (x - 1) = (y^2 + 3 * y - 5) / (y^2 + 3 * y - 7)) :
  x = (y^2 + 3 * y - 5) / 2 :=
by 
  sorry

end solve_for_x_l2021_202198


namespace robbie_weight_l2021_202150

theorem robbie_weight (R P : ℝ) 
  (h1 : P = 4.5 * R - 235)
  (h2 : P = R + 115) :
  R = 100 := 
by 
  sorry

end robbie_weight_l2021_202150


namespace cos_sum_proof_l2021_202123

theorem cos_sum_proof (x : ℝ) (h : Real.cos (x - (Real.pi / 6)) = Real.sqrt 3 / 3) :
  Real.cos x + Real.cos (x - Real.pi / 3) = 1 := 
sorry

end cos_sum_proof_l2021_202123


namespace problem_l2021_202168

noncomputable def f : ℕ → ℕ := sorry
noncomputable def g : ℕ → ℕ := sorry

theorem problem (surj_f : ∀ y, ∃ x, f x = y) 
                (inj_g : ∀ x1 x2, g x1 = g x2 → x1 = x2)
                (f_ge_g : ∀ n, f n ≥ g n) :
  ∀ n, f n = g n := 
by 
  sorry

end problem_l2021_202168


namespace ella_max_book_price_l2021_202173

/--
Given that Ella needs to buy 20 identical books and her total budget, 
after deducting the $5 entry fee, is $195. Each book has the same 
cost in whole dollars, and an 8% sales tax is applied to the price of each book. 
Prove that the highest possible price per book that Ella can afford is $9.
-/
theorem ella_max_book_price : 
  ∀ (n : ℕ) (B T : ℝ), n = 20 → B = 195 → T = 1.08 → 
  ∃ (p : ℕ), (↑p ≤ B / T / n) → (9 ≤ p) := 
by 
  sorry

end ella_max_book_price_l2021_202173


namespace total_opponent_score_l2021_202109

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

end total_opponent_score_l2021_202109


namespace mike_disk_space_l2021_202197

theorem mike_disk_space (F L T : ℕ) (hF : F = 26) (hL : L = 2) : T = 28 :=
by
  have h : T = F + L := by sorry
  rw [hF, hL] at h
  assumption

end mike_disk_space_l2021_202197


namespace sqrt_subtraction_result_l2021_202161

theorem sqrt_subtraction_result : 
  (Real.sqrt (49 + 36) - Real.sqrt (36 - 0)) = 4 :=
by
  sorry

end sqrt_subtraction_result_l2021_202161


namespace no_integer_solutions_l2021_202138

theorem no_integer_solutions (m n : ℤ) : ¬ (m ^ 3 + 6 * m ^ 2 + 5 * m = 27 * n ^ 3 + 9 * n ^ 2 + 9 * n + 1) :=
sorry

end no_integer_solutions_l2021_202138


namespace car_meeting_distance_l2021_202179

theorem car_meeting_distance
  (distance_AB : ℝ)
  (speed_A : ℝ)
  (speed_B : ℝ)
  (midpoint_C : ℝ)
  (meeting_distance_from_C : ℝ) 
  (h1 : distance_AB = 245)
  (h2 : speed_A = 70)
  (h3 : speed_B = 90)
  (h4 : midpoint_C = distance_AB / 2) :
  meeting_distance_from_C = 15.31 := 
sorry

end car_meeting_distance_l2021_202179


namespace mary_blue_marbles_l2021_202143

theorem mary_blue_marbles (dan_blue_marbles mary_blue_marbles : ℕ)
  (h1 : dan_blue_marbles = 5)
  (h2 : mary_blue_marbles = 2 * dan_blue_marbles) : mary_blue_marbles = 10 := 
by
  sorry

end mary_blue_marbles_l2021_202143


namespace no_common_point_in_all_circles_l2021_202155

variable {Point : Type}
variable {Circle : Type}
variable (center : Circle → Point)
variable (contains : Circle → Point → Prop)

-- Given six circles in the plane
variables (C1 C2 C3 C4 C5 C6 : Circle)

-- Condition: None of the circles contain the center of any other circle
axiom condition_1 : ∀ (C D : Circle), C ≠ D → ¬ contains C (center D)

-- Question: Prove that there does not exist a point P that lies in all six circles
theorem no_common_point_in_all_circles : 
  ¬ ∃ (P : Point), (contains C1 P) ∧ (contains C2 P) ∧ (contains C3 P) ∧ (contains C4 P) ∧ (contains C5 P) ∧ (contains C6 P) :=
sorry

end no_common_point_in_all_circles_l2021_202155


namespace greatest_number_same_remainder_l2021_202149

theorem greatest_number_same_remainder (d : ℕ) :
  d ∣ (57 - 25) ∧ d ∣ (105 - 57) ∧ d ∣ (105 - 25) → d ≤ 16 :=
by
  sorry

end greatest_number_same_remainder_l2021_202149


namespace number_of_first_year_students_to_be_sampled_l2021_202139

-- Definitions based on the conditions
def total_students_in_each_grade (x : ℕ) : List ℕ := [4*x, 5*x, 5*x, 6*x]
def total_undergraduate_students (x : ℕ) : ℕ := 4*x + 5*x + 5*x + 6*x
def sample_size : ℕ := 300
def sampling_fraction (x : ℕ) : ℚ := sample_size / total_undergraduate_students x
def first_year_sampling (x : ℕ) : ℕ := (4*x) * sample_size / total_undergraduate_students x

-- Statement to prove
theorem number_of_first_year_students_to_be_sampled {x : ℕ} (hx_pos : x > 0) :
  first_year_sampling x = 60 := 
by
  -- skip the proof
  sorry

end number_of_first_year_students_to_be_sampled_l2021_202139


namespace rectangle_area_l2021_202144

theorem rectangle_area (l w r: ℝ) (h1 : l = 2 * r) (h2 : w = r) : l * w = 2 * r^2 :=
by sorry

end rectangle_area_l2021_202144


namespace point_K_is_intersection_of_diagonals_l2021_202101

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

end point_K_is_intersection_of_diagonals_l2021_202101


namespace acute_angles_in_triangle_l2021_202120

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

end acute_angles_in_triangle_l2021_202120


namespace remainder_of_x50_div_by_x_sub_1_cubed_l2021_202134

theorem remainder_of_x50_div_by_x_sub_1_cubed :
  (x^50 % (x-1)^3) = (1225*x^2 - 2500*x + 1276) :=
sorry

end remainder_of_x50_div_by_x_sub_1_cubed_l2021_202134


namespace consignment_shop_total_items_l2021_202100

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

end consignment_shop_total_items_l2021_202100


namespace sequence_term_10_l2021_202116

theorem sequence_term_10 : ∃ n : ℕ, (1 / (n * (n + 2)) = 1 / 120) ∧ n = 10 := by
  sorry

end sequence_term_10_l2021_202116


namespace minimum_cost_l2021_202110

noncomputable def total_cost (x : ℝ) : ℝ :=
  (1800 / (x + 5)) + 0.5 * x

theorem minimum_cost : 
  (∃ x : ℝ, x = 55 ∧ total_cost x = 57.5) :=
  sorry

end minimum_cost_l2021_202110


namespace stock_percent_change_l2021_202163

theorem stock_percent_change (y : ℝ) : 
  let value_after_day1 := 0.85 * y
  let value_after_day2 := 1.25 * value_after_day1
  (value_after_day2 - y) / y * 100 = 6.25 := by
  sorry

end stock_percent_change_l2021_202163


namespace probability_drawing_3_one_color_1_other_l2021_202145

theorem probability_drawing_3_one_color_1_other (black white : ℕ) (total_balls drawn_balls : ℕ) 
    (total_ways : ℕ) (ways_3_black_1_white : ℕ) (ways_1_black_3_white : ℕ) :
    black = 10 → white = 5 → total_balls = 15 → drawn_balls = 4 →
    total_ways = Nat.choose total_balls drawn_balls →
    ways_3_black_1_white = Nat.choose black 3 * Nat.choose white 1 →
    ways_1_black_3_white = Nat.choose black 1 * Nat.choose white 3 →
    (ways_3_black_1_white + ways_1_black_3_white) / total_ways = 140 / 273 := 
by
  intros h_black h_white h_total_balls h_drawn_balls h_total_ways h_ways_3_black_1_white h_ways_1_black_3_white
  -- The proof would go here, but is not required for this task.
  sorry

end probability_drawing_3_one_color_1_other_l2021_202145


namespace ferry_travel_time_l2021_202131

theorem ferry_travel_time:
  ∀ (v_P v_Q : ℝ) (d_P d_Q : ℝ) (t_P t_Q : ℝ),
    v_P = 8 →
    v_Q = v_P + 1 →
    d_Q = 3 * d_P →
    t_Q = t_P + 5 →
    d_P = v_P * t_P →
    d_Q = v_Q * t_Q →
    t_P = 3 := by
  sorry

end ferry_travel_time_l2021_202131


namespace find_C_l2021_202147

theorem find_C (A B C : ℕ) 
  (h1 : A + B + C = 700) 
  (h2 : A + C = 300) 
  (h3 : B + C = 600) 
  : C = 200 := sorry

end find_C_l2021_202147


namespace ellipse_focus_distance_l2021_202118

theorem ellipse_focus_distance :
  ∀ {x y : ℝ},
    (x^2) / 25 + (y^2) / 16 = 1 →
    (dist (x, y) (3, 0) = 8) →
    dist (x, y) (-3, 0) = 2 :=
by
  intro x y h₁ h₂
  sorry

end ellipse_focus_distance_l2021_202118


namespace system_solution_l2021_202182

theorem system_solution (x y : ℝ) 
  (h1 : (x^2 + x * y + y^2) / (x^2 - x * y + y^2) = 3) 
  (h2 : x^3 + y^3 = 2) : x = 1 ∧ y = 1 :=
  sorry

end system_solution_l2021_202182


namespace book_arrangement_count_l2021_202193

-- Conditions
def num_math_books := 4
def num_history_books := 5

-- The number of arrangements is
def arrangements (n m : Nat) : Nat :=
  let choose_end_books := n * (n - 1)
  let choose_middle_book := (n - 2)
  let remaining_books := (n - 3) + m
  choose_end_books * choose_middle_book * Nat.factorial remaining_books

theorem book_arrangement_count (n m : Nat) (h1 : n = num_math_books) (h2 : m = num_history_books) :
  arrangements n m = 120960 :=
by
  rw [h1, h2, arrangements]
  norm_num
  sorry

end book_arrangement_count_l2021_202193


namespace KrystianaChargesForSecondFloorRooms_Theorem_l2021_202183

noncomputable def KrystianaChargesForSecondFloorRooms (X : ℝ) : Prop :=
  let costFirstFloor := 3 * 15
  let costThirdFloor := 3 * (2 * 15)
  let totalEarnings := costFirstFloor + 3 * X + costThirdFloor
  totalEarnings = 165 → X = 10

-- This is the statement only. The proof is not included.
theorem KrystianaChargesForSecondFloorRooms_Theorem : KrystianaChargesForSecondFloorRooms 10 :=
sorry

end KrystianaChargesForSecondFloorRooms_Theorem_l2021_202183


namespace alex_chairs_l2021_202115

theorem alex_chairs (x y z : ℕ) (h : x + y + z = 74) : z = 74 - x - y :=
by
  sorry

end alex_chairs_l2021_202115


namespace new_equation_incorrect_l2021_202166

-- Definition of a function to change each digit of a number by +1 or -1 randomly.
noncomputable def modify_digit (num : ℕ) : ℕ := sorry

-- Proposition stating the original problem's condition and conclusion.
theorem new_equation_incorrect (a b : ℕ) (c := a + b) (a' b' c' : ℕ)
    (h1 : a' = modify_digit a)
    (h2 : b' = modify_digit b)
    (h3 : c' = modify_digit c) :
    a' + b' ≠ c' :=
sorry

end new_equation_incorrect_l2021_202166


namespace range_of_m_l2021_202146

theorem range_of_m 
  (h : ∀ x : ℝ, x^2 + m * x + m^2 - 1 > 0) :
  m ∈ (Set.Ioo (-(2 * Real.sqrt 3) / 3) (-(2 * Real.sqrt 3) / 3)).union (Set.Ioi ((2 * Real.sqrt 3) / 3)) := 
sorry

end range_of_m_l2021_202146


namespace cube_of_square_of_third_smallest_prime_is_correct_l2021_202188

def cube_of_square_of_third_smallest_prime : Nat := 15625

theorem cube_of_square_of_third_smallest_prime_is_correct :
  let third_smallest_prime := 5
  let square := third_smallest_prime ^ 2
  let cube := square ^ 3
  cube = cube_of_square_of_third_smallest_prime :=
by
  let third_smallest_prime := 5
  let square := third_smallest_prime ^ 2
  let cube := square ^ 3
  show cube = 15625
  sorry

end cube_of_square_of_third_smallest_prime_is_correct_l2021_202188


namespace mul_65_35_l2021_202153

theorem mul_65_35 : (65 * 35) = 2275 := by
  -- define a and b
  let a := 50
  let b := 15
  -- use the equivalence (a + b) and (a - b)
  have h1 : 65 = a + b := by rfl
  have h2 : 35 = a - b := by rfl
  -- use the difference of squares formula
  have h_diff_squares : (a + b) * (a - b) = a^2 - b^2 := by sorry
  -- calculate each square
  have ha_sq : a^2 = 2500 := by sorry
  have hb_sq : b^2 = 225 := by sorry
  -- combine the results
  have h_result : a^2 - b^2 = 2500 - 225 := by sorry
  -- finish the proof
  have final_result : (65 * 35) = 2275 := by sorry
  exact final_result

end mul_65_35_l2021_202153


namespace amelia_remaining_money_l2021_202129

variable {m b n : ℚ}

theorem amelia_remaining_money (h : (1 / 4) * m = (1 / 2) * n * b) : 
  m - n * b = (1 / 2) * m :=
by
  sorry

end amelia_remaining_money_l2021_202129


namespace units_digit_squares_eq_l2021_202142

theorem units_digit_squares_eq (x y : ℕ) (hx : x % 10 + y % 10 = 10) :
  (x * x) % 10 = (y * y) % 10 :=
by
  sorry

end units_digit_squares_eq_l2021_202142


namespace find_coefficients_l2021_202164

theorem find_coefficients (c d : ℝ)
  (h : ∃ u v : ℝ, u ≠ v ∧ (u^3 + c * u^2 + 10 * u + 4 = 0) ∧ (v^3 + c * v^2 + 10 * v + 4 = 0)
     ∧ (u^3 + d * u^2 + 13 * u + 5 = 0) ∧ (v^3 + d * v^2 + 13 * v + 5 = 0)) :
  (c, d) = (7, 8) :=
by
  sorry

end find_coefficients_l2021_202164

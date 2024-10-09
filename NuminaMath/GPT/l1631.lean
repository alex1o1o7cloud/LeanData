import Mathlib

namespace arith_expression_evaluation_l1631_163160

theorem arith_expression_evaluation :
  2 + (1/6:ℚ) + (((4.32:ℚ) - 1.68 - (1 + 8/25:ℚ)) * (5/11:ℚ) - (2/7:ℚ)) / (1 + 9/35:ℚ) = 2 + 101/210 := by
  sorry

end arith_expression_evaluation_l1631_163160


namespace relationship_a_b_l1631_163117

-- Definitions of the two quadratic equations having a single common root
def has_common_root (a b : ℝ) : Prop :=
  ∃ t : ℝ, (t^2 + a * t + b = 0) ∧ (t^2 + b * t + a = 0)

-- Theorem stating the relationship between a and b
theorem relationship_a_b (a b : ℝ) (h : has_common_root a b) : a ≠ b → a + b + 1 = 0 :=
by sorry

end relationship_a_b_l1631_163117


namespace swapped_digit_number_l1631_163168

theorem swapped_digit_number (a b : ℕ) (h1 : 0 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9) :
  10 * b + a = new_number :=
sorry

end swapped_digit_number_l1631_163168


namespace simple_interest_years_l1631_163180

variable (P R T : ℕ)
variable (deltaI : ℕ := 400)
variable (P_value : P = 800)

theorem simple_interest_years 
  (h : (800 * (R + 5) * T / 100) = (800 * R * T / 100) + 400) :
  T = 10 :=
by sorry

end simple_interest_years_l1631_163180


namespace investment_three_years_ago_l1631_163172

noncomputable def initial_investment (final_amount : ℝ) : ℝ :=
  final_amount / (1.08 ^ 3)

theorem investment_three_years_ago :
  abs (initial_investment 439.23 - 348.68) < 0.01 :=
by
  sorry

end investment_three_years_ago_l1631_163172


namespace ticket_cost_l1631_163126

theorem ticket_cost (a : ℝ)
  (h1 : ∀ c : ℝ, c = a / 3)
  (h2 : 3 * a + 5 * (a / 3) = 27.75) :
  6 * a + 9 * (a / 3) = 53.52 := 
sorry

end ticket_cost_l1631_163126


namespace polar_eq_is_circle_l1631_163192

-- Define the polar equation as a condition
def polar_eq (ρ : ℝ) := ρ = 5

-- Define the origin
def origin : ℝ × ℝ := (0, 0)

-- Prove that the curve represented by the polar equation is a circle
theorem polar_eq_is_circle (P : ℝ × ℝ) : (∃ ρ θ, P = (ρ * Real.cos θ, ρ * Real.sin θ) ∧ polar_eq ρ) ↔ dist P origin = 5 := 
by 
  sorry

end polar_eq_is_circle_l1631_163192


namespace incorrect_calculation_d_l1631_163152

theorem incorrect_calculation_d : (1 / 3) / (-1) ≠ 3 * (-1) := 
by {
  -- we'll leave the body of the proof as sorry.
  sorry
}

end incorrect_calculation_d_l1631_163152


namespace value_of_star_l1631_163196

theorem value_of_star : 
  ∀ (star : ℤ), 45 - (28 - (37 - (15 - star))) = 59 → star = -154 :=
by
  intro star
  intro h
  -- Proof to be provided
  sorry

end value_of_star_l1631_163196


namespace cookies_with_five_cups_l1631_163149

-- Define the initial condition: Lee can make 24 cookies with 3 cups of flour
def cookies_per_cup := 24 / 3

-- Theorem stating Lee can make 40 cookies with 5 cups of flour
theorem cookies_with_five_cups : 5 * cookies_per_cup = 40 :=
by
  sorry

end cookies_with_five_cups_l1631_163149


namespace fraction_product_l1631_163138

theorem fraction_product :
  (3 / 7) * (5 / 8) * (9 / 13) * (11 / 17) = 1485 / 12376 := 
by
  sorry

end fraction_product_l1631_163138


namespace sum_of_interior_angles_l1631_163157

def f (n : ℕ) : ℚ := (n - 2) * 180

theorem sum_of_interior_angles (n : ℕ) : f (n + 1) = f n + 180 :=
by
  unfold f
  sorry

end sum_of_interior_angles_l1631_163157


namespace hotel_room_friends_distribution_l1631_163143

theorem hotel_room_friends_distribution 
    (rooms : ℕ)
    (friends : ℕ)
    (min_friends_per_room : ℕ)
    (max_friends_per_room : ℕ)
    (unique_ways : ℕ) :
    rooms = 6 →
    friends = 10 →
    min_friends_per_room = 1 →
    max_friends_per_room = 3 →
    unique_ways = 1058400 :=
by
  intros h_rooms h_friends h_min_friends h_max_friends
  sorry

end hotel_room_friends_distribution_l1631_163143


namespace crayons_lost_l1631_163136

theorem crayons_lost (initial_crayons ending_crayons : ℕ) (h_initial : initial_crayons = 253) (h_ending : ending_crayons = 183) : (initial_crayons - ending_crayons) = 70 :=
by
  sorry

end crayons_lost_l1631_163136


namespace shirt_ratio_l1631_163137

theorem shirt_ratio
  (A B S : ℕ)
  (h1 : A = 6 * B)
  (h2 : B = 3)
  (h3 : S = 72) :
  S / A = 4 :=
by
  sorry

end shirt_ratio_l1631_163137


namespace stick_length_l1631_163113

theorem stick_length (x : ℕ) (h1 : 2 * x + (2 * x - 1) = 14) : x = 3 := sorry

end stick_length_l1631_163113


namespace find_a2_l1631_163194

variables {α : Type*} [LinearOrderedField α]

def geometric_sequence (a : ℕ → α) : Prop :=
  ∀ n m : ℕ, ∃ r : α, a (n + m) = (a n) * (a m) * r

theorem find_a2 (a : ℕ → α) (h_geom : geometric_sequence a) (h1 : a 3 * a 6 = 9) (h2 : a 2 * a 4 * a 5 = 27) :
  a 2 = 3 :=
sorry

end find_a2_l1631_163194


namespace sum_three_times_m_and_half_n_square_diff_minus_square_sum_l1631_163119

-- Problem (1) Statement
theorem sum_three_times_m_and_half_n (m n : ℝ) : 3 * m + 1 / 2 * n = 3 * m + 1 / 2 * n :=
by
  sorry

-- Problem (2) Statement
theorem square_diff_minus_square_sum (a b : ℝ) : (a - b) ^ 2 - (a + b) ^ 2 = (a - b) ^ 2 - (a + b) ^ 2 :=
by
  sorry

end sum_three_times_m_and_half_n_square_diff_minus_square_sum_l1631_163119


namespace minimum_b_value_l1631_163127

noncomputable def f (x a : ℝ) : ℝ := (x - a)^2 + (Real.log (x^2 - 2 * a))^2

theorem minimum_b_value (a : ℝ) : ∃ x_0 > 0, f x_0 a ≤ (4 / 5) :=
sorry

end minimum_b_value_l1631_163127


namespace major_axis_of_ellipse_l1631_163150

structure Ellipse :=
(center : ℝ × ℝ)
(tangent_y_axis : Bool)
(tangent_y_eq_3 : Bool)
(focus_1 : ℝ × ℝ)
(focus_2 : ℝ × ℝ)

noncomputable def major_axis_length (e : Ellipse) : ℝ :=
  2 * (e.focus_1.2 - e.center.2)

theorem major_axis_of_ellipse : 
  ∀ (e : Ellipse), 
    e.center = (3, 0) ∧
    e.tangent_y_axis = true ∧
    e.tangent_y_eq_3 = true ∧
    e.focus_1 = (3, 2 + Real.sqrt 2) ∧
    e.focus_2 = (3, -2 - Real.sqrt 2) →
      major_axis_length e = 4 + 2 * Real.sqrt 2 :=
by
  intro e
  intro h
  sorry

end major_axis_of_ellipse_l1631_163150


namespace bisection_next_interval_l1631_163108

def f (x : ℝ) : ℝ := x^3 - 2 * x - 5

theorem bisection_next_interval (h₀ : f 2.5 > 0) (h₁ : f 2 < 0) :
  ∃ a b, (2 < 2.5) ∧ f 2 < 0 ∧ f 2.5 > 0 ∧ a = 2 ∧ b = 2.5 :=
by
  sorry

end bisection_next_interval_l1631_163108


namespace inequality_with_equality_condition_l1631_163116

variable {a b c d : ℝ}

theorem inequality_with_equality_condition (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) : 
  (a^2 + b^2 + c^2 + d^2)^2 ≥ (a + b) * (b + c) * (c + d) * (d + a) ∧ 
  ((a^2 + b^2 + c^2 + d^2)^2 = (a + b) * (b + c) * (c + d) * (d + a) ↔ a = b ∧ b = c ∧ c = d) := sorry

end inequality_with_equality_condition_l1631_163116


namespace trailingZeros_310_fact_l1631_163186

-- Define the function to compute trailing zeros in factorials
def trailingZeros (n : ℕ) : ℕ := 
  if n = 0 then 0 else n / 5 + trailingZeros (n / 5)

-- Define the specific case for 310!
theorem trailingZeros_310_fact : trailingZeros 310 = 76 := 
by 
  sorry

end trailingZeros_310_fact_l1631_163186


namespace students_more_than_pets_l1631_163163

theorem students_more_than_pets :
  let students_per_classroom := 15
  let rabbits_per_classroom := 1
  let guinea_pigs_per_classroom := 3
  let number_of_classrooms := 6
  let total_students := students_per_classroom * number_of_classrooms
  let total_pets := (rabbits_per_classroom + guinea_pigs_per_classroom) * number_of_classrooms
  total_students - total_pets = 66 :=
by
  sorry

end students_more_than_pets_l1631_163163


namespace total_weekly_pay_proof_l1631_163140

-- Define the weekly pay for employees X and Y
def weekly_pay_employee_y : ℝ := 260
def weekly_pay_employee_x : ℝ := 1.2 * weekly_pay_employee_y

-- Definition of total weekly pay
def total_weekly_pay : ℝ := weekly_pay_employee_x + weekly_pay_employee_y

-- Theorem stating the total weekly pay equals 572
theorem total_weekly_pay_proof : total_weekly_pay = 572 := by
  sorry

end total_weekly_pay_proof_l1631_163140


namespace factorization_identity_l1631_163148

theorem factorization_identity (a : ℝ) : (a + 3) * (a - 7) + 25 = (a - 2) ^ 2 :=
by
  sorry

end factorization_identity_l1631_163148


namespace breaststroke_speed_correct_l1631_163170

-- Defining the given conditions
def total_distance : ℕ := 500
def front_crawl_speed : ℕ := 45
def front_crawl_time : ℕ := 8
def total_time : ℕ := 12

-- Definition of the breaststroke speed given the conditions
def breaststroke_speed : ℕ :=
  let front_crawl_distance := front_crawl_speed * front_crawl_time
  let breaststroke_distance := total_distance - front_crawl_distance
  let breaststroke_time := total_time - front_crawl_time
  breaststroke_distance / breaststroke_time

-- Theorem to prove the breaststroke speed is 35 yards per minute
theorem breaststroke_speed_correct : breaststroke_speed = 35 :=
  sorry

end breaststroke_speed_correct_l1631_163170


namespace minimum_students_for_same_vote_l1631_163181

theorem minimum_students_for_same_vote (n : ℕ) (k : ℕ) (h1 : n = 10) (h2 : k = 2) :
  ∃ m, m = 46 ∧ ∀ (students : Finset (Finset ℕ)), students.card = m → 
    (∃ s1 s2, s1 ≠ s2 ∧ s1.card = k ∧ s2.card = k ∧ s1 ⊆ (Finset.range n) ∧ s2 ⊆ (Finset.range n) ∧ s1 = s2) :=
by 
  sorry

end minimum_students_for_same_vote_l1631_163181


namespace exp_mono_increasing_l1631_163198

theorem exp_mono_increasing (x y : ℝ) (h : x ≤ y) : (2:ℝ)^x ≤ (2:ℝ)^y :=
sorry

end exp_mono_increasing_l1631_163198


namespace total_flowers_in_3_hours_l1631_163173

-- Constants representing the number of each type of flower
def roses : ℕ := 12
def sunflowers : ℕ := 15
def tulips : ℕ := 9
def daisies : ℕ := 18
def orchids : ℕ := 6
def total_flowers : ℕ := 60

-- Number of flowers each bee can pollinate in an hour
def bee_A_rate (roses sunflowers tulips: ℕ) : ℕ := 2 + 3 + 1
def bee_B_rate (daisies orchids: ℕ) : ℕ := 4 + 1
def bee_C_rate (roses sunflowers tulips daisies orchids: ℕ) : ℕ := 1 + 2 + 2 + 3 + 1

-- Total number of flowers pollinated by all bees in an hour
def total_bees_rate (bee_A_rate bee_B_rate bee_C_rate: ℕ) : ℕ := bee_A_rate + bee_B_rate + bee_C_rate

-- Proving the total flowers pollinated in 3 hours
theorem total_flowers_in_3_hours : total_bees_rate 6 5 9 * 3 = total_flowers := 
by {
  sorry
}

end total_flowers_in_3_hours_l1631_163173


namespace band_members_count_l1631_163100

theorem band_members_count :
  ∃ n k m : ℤ, n = 10 * k + 4 ∧ n = 12 * m + 6 ∧ 200 ≤ n ∧ n ≤ 300 ∧ n = 254 :=
by
  -- Declaration of the theorem properties
  sorry

end band_members_count_l1631_163100


namespace geometric_arithmetic_sequence_sum_l1631_163175

theorem geometric_arithmetic_sequence_sum {a b : ℕ → ℝ} (q : ℝ) (n : ℕ) 
(h1 : a 2 = 2)
(h2 : a 2 = 2)
(h3 : 2 * (a 3 + 1) = a 2 + a 4)
(h4 : ∀ (n : ℕ), (a (n + 1)) = a 0 * q ^ (n + 1))
(h5 : b n = n * (n + 1)) :
a 8 + (b 8 - b 7) = 144 :=
by { sorry }

end geometric_arithmetic_sequence_sum_l1631_163175


namespace reduced_price_l1631_163118

theorem reduced_price (P R : ℝ) (h1 : R = 0.8 * P) (h2 : 600 = (600 / P + 4) * R) : R = 30 := 
by
  sorry

end reduced_price_l1631_163118


namespace max_f_l1631_163161

noncomputable def f (θ : ℝ) : ℝ :=
  Real.cos (θ / 2) * (1 + Real.sin θ)

theorem max_f : ∀ (θ : ℝ), 0 < θ ∧ θ < π → f θ ≤ (4 * Real.sqrt 3) / 9 :=
by
  sorry

end max_f_l1631_163161


namespace cost_per_bag_of_potatoes_l1631_163123

variable (x : ℕ)

def chickens_cost : ℕ := 5 * 3
def celery_cost : ℕ := 4 * 2
def total_paid : ℕ := 35
def potatoes_cost (x : ℕ) : ℕ := 2 * x

theorem cost_per_bag_of_potatoes : 
  chickens_cost + celery_cost + potatoes_cost x = total_paid → x = 6 :=
by
  sorry

end cost_per_bag_of_potatoes_l1631_163123


namespace solve_triple_l1631_163132

theorem solve_triple (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c + a * b + c = a^3) : 
  (b = a - 1 ∧ c = a) ∨ (b = 1 ∧ c = a * (a - 1)) :=
by 
  sorry

end solve_triple_l1631_163132


namespace expression_positive_intervals_l1631_163145

theorem expression_positive_intervals :
  {x : ℝ | (x + 2) * (x - 3) > 0} = {x | x < -2} ∪ {x | x > 3} :=
by
  sorry

end expression_positive_intervals_l1631_163145


namespace seq_diff_five_consec_odd_avg_55_l1631_163158

theorem seq_diff_five_consec_odd_avg_55 {a b c d e : ℤ} 
    (h1: a % 2 = 1) (h2: b % 2 = 1) (h3: c % 2 = 1) (h4: d % 2 = 1) (h5: e % 2 = 1)
    (h6: b = a + 2) (h7: c = a + 4) (h8: d = a + 6) (h9: e = a + 8)
    (avg_5_seq : (a + b + c + d + e) / 5 = 55) : 
    e - a = 8 := 
by
    -- proof part can be skipped with sorry
    sorry

end seq_diff_five_consec_odd_avg_55_l1631_163158


namespace smallest_part_2340_division_l1631_163178

theorem smallest_part_2340_division :
  ∃ (A B C : ℕ), (A + B + C = 2340) ∧ 
                 (A / 5 = B / 7) ∧ 
                 (B / 7 = C / 11) ∧ 
                 (A = 510) :=
by 
  sorry

end smallest_part_2340_division_l1631_163178


namespace term_position_in_sequence_l1631_163154

theorem term_position_in_sequence (n : ℕ) (h1 : n > 0) (h2 : 3 * n + 1 = 40) : n = 13 :=
by
  sorry

end term_position_in_sequence_l1631_163154


namespace probability_of_drawing_two_white_balls_l1631_163111

-- Define the total number of balls and their colors
def red_balls : ℕ := 2
def white_balls : ℕ := 2
def total_balls : ℕ := red_balls + white_balls

-- Define the total number of ways to draw 2 balls from 4
def total_draw_ways : ℕ := (total_balls.choose 2)

-- Define the number of ways to draw 2 white balls
def white_draw_ways : ℕ := (white_balls.choose 2)

-- Define the probability of drawing 2 white balls
def probability_white_draw : ℚ := white_draw_ways / total_draw_ways

-- The main theorem statement to prove
theorem probability_of_drawing_two_white_balls :
  probability_white_draw = 1 / 6 := by
  sorry

end probability_of_drawing_two_white_balls_l1631_163111


namespace winner_C_l1631_163166

noncomputable def votes_A : ℕ := 4500
noncomputable def votes_B : ℕ := 7000
noncomputable def votes_C : ℕ := 12000
noncomputable def votes_D : ℕ := 8500
noncomputable def votes_E : ℕ := 3500

noncomputable def total_votes : ℕ := votes_A + votes_B + votes_C + votes_D + votes_E

noncomputable def percentage (votes : ℕ) : ℚ :=
   (votes : ℚ) / (total_votes : ℚ) * 100

noncomputable def percentage_A : ℚ := percentage votes_A
noncomputable def percentage_B : ℚ := percentage votes_B
noncomputable def percentage_C : ℚ := percentage votes_C
noncomputable def percentage_D : ℚ := percentage votes_D
noncomputable def percentage_E : ℚ := percentage votes_E

theorem winner_C : (percentage_C = 33.803) := 
sorry

end winner_C_l1631_163166


namespace max_d_n_is_one_l1631_163120

open Int

/-- The sequence definition -/
def seq (n : ℕ) : ℤ := 100 + n^3

/-- The definition of d_n -/
def d_n (n : ℕ) : ℤ := gcd (seq n) (seq (n + 1))

/-- The theorem stating the maximum value of d_n for positive integers is 1 -/
theorem max_d_n_is_one : ∀ (n : ℕ), 1 ≤ n → d_n n = 1 := by
  sorry

end max_d_n_is_one_l1631_163120


namespace simplify_expression_l1631_163187

variable (a : ℝ)

theorem simplify_expression : 3 * a^2 - a * (2 * a - 1) = a^2 + a :=
by
  sorry

end simplify_expression_l1631_163187


namespace stratified_sampling_grade11_l1631_163164

noncomputable def g10 : ℕ := 500
noncomputable def total_students : ℕ := 1350
noncomputable def g10_sample : ℕ := 120
noncomputable def ratio : ℚ := g10_sample / g10
noncomputable def g11 : ℕ := 450
noncomputable def g12 : ℕ := g11 - 50

theorem stratified_sampling_grade11 :
  g10 + g11 + g12 = total_students →
  (g10_sample / g10) = ratio →
  sample_g11 = g11 * ratio →
  sample_g11 = 108 :=
by
  sorry

end stratified_sampling_grade11_l1631_163164


namespace total_remaining_staff_l1631_163189

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

end total_remaining_staff_l1631_163189


namespace price_reduction_achieves_profit_l1631_163112

theorem price_reduction_achieves_profit :
  ∃ x : ℝ, (40 - x) * (20 + 2 * (x / 4) * 8) = 1200 ∧ x = 20 :=
by
  sorry

end price_reduction_achieves_profit_l1631_163112


namespace impossible_configuration_l1631_163133

theorem impossible_configuration : 
  ¬∃ (f : ℕ → ℕ) (h : ∀n, 1 ≤ f n ∧ f n ≤ 5) (perm : ∀i j, if i < j then f i ≠ f j else true), 
  (f 0 = 3) ∧ (f 1 = 4) ∧ (f 2 = 2) ∧ (f 3 = 1) ∧ (f 4 = 5) :=
sorry

end impossible_configuration_l1631_163133


namespace train_speed_is_5400432_kmh_l1631_163121

noncomputable def train_speed_kmh (time_to_pass_platform : ℝ) (time_to_pass_man : ℝ) (length_platform : ℝ) : ℝ :=
  let speed_m_per_s := length_platform / (time_to_pass_platform - time_to_pass_man)
  speed_m_per_s * 3.6

theorem train_speed_is_5400432_kmh :
  train_speed_kmh 35 20 225.018 = 54.00432 :=
by
  sorry

end train_speed_is_5400432_kmh_l1631_163121


namespace problem_a_problem_c_problem_d_l1631_163122

variables (a b : ℝ)

-- Given condition
def condition : Prop := a + b > 0

-- Proof problems
theorem problem_a (h : condition a b) : a^5 * b^2 + a^4 * b^3 ≥ 0 := sorry

theorem problem_c (h : condition a b) : a^21 + b^21 > 0 := sorry

theorem problem_d (h : condition a b) : (a + 2) * (b + 2) > a * b := sorry

end problem_a_problem_c_problem_d_l1631_163122


namespace tangent_circle_radius_l1631_163144

theorem tangent_circle_radius (O A B C : ℝ) (r1 r2 : ℝ) :
  (O = 5) →
  (abs (A - B) = 8) →
  (C = (2 * A + B) / 3) →
  r1 = 8 / 9 ∨ r2 = 32 / 9 :=
sorry

end tangent_circle_radius_l1631_163144


namespace winning_strategy_for_pawns_l1631_163129

def wiit_or_siti_wins (n : ℕ) : Prop :=
  (∃ k : ℕ, n = 3 * k + 2) ∨ (∃ k : ℕ, n ≠ 3 * k + 2)

theorem winning_strategy_for_pawns (n : ℕ) : wiit_or_siti_wins n :=
sorry

end winning_strategy_for_pawns_l1631_163129


namespace sin_of_right_triangle_l1631_163179

open Real

theorem sin_of_right_triangle (Q : ℝ) (h : 3 * sin Q = 4 * cos Q) : sin Q = 4 / 5 :=
by
  sorry

end sin_of_right_triangle_l1631_163179


namespace initial_cupcakes_l1631_163199

variable (x : ℕ) -- Define x as the number of cupcakes Robin initially made

-- Define the conditions provided in the problem
def cupcakes_sold := 22
def cupcakes_made := 39
def final_cupcakes := 59

-- Formalize the problem statement: Prove that given the conditions, the initial cupcakes equals 42
theorem initial_cupcakes:
  x - cupcakes_sold + cupcakes_made = final_cupcakes → x = 42 := 
by
  -- Placeholder for the proof
  sorry

end initial_cupcakes_l1631_163199


namespace george_boxes_of_eggs_l1631_163188

theorem george_boxes_of_eggs (boxes_eggs : Nat) (h1 : ∀ (eggs_per_box : Nat), eggs_per_box = 3 → boxes_eggs * eggs_per_box = 15) :
  boxes_eggs = 5 :=
by
  sorry

end george_boxes_of_eggs_l1631_163188


namespace ajith_rana_meet_l1631_163177

/--
Ajith and Rana walk around a circular course 115 km in circumference, starting together from the same point.
Ajith walks at 4 km/h, and Rana walks at 5 km/h in the same direction.
Prove that they will meet after 115 hours.
-/
theorem ajith_rana_meet 
  (course_circumference : ℕ)
  (ajith_speed : ℕ)
  (rana_speed : ℕ)
  (relative_speed : ℕ)
  (time : ℕ)
  (start_point : Point)
  (ajith : Person)
  (rana : Person)
  (walk_in_same_direction : Prop)
  (start_time : ℕ)
  (meet_time : ℕ) :
  course_circumference = 115 →
  ajith_speed = 4 →
  rana_speed = 5 →
  relative_speed = rana_speed - ajith_speed →
  time = course_circumference / relative_speed →
  meet_time = start_time + time →
  meet_time = 115 :=
by
  sorry

end ajith_rana_meet_l1631_163177


namespace B_is_not_15_percent_less_than_A_l1631_163135

noncomputable def A (B : ℝ) : ℝ := 1.15 * B

theorem B_is_not_15_percent_less_than_A (B : ℝ) (h : B > 0) : A B ≠ 0.85 * (A B) :=
by
  unfold A
  suffices 1.15 * B ≠ 0.85 * (1.15 * B) by
    intro h1
    exact this h1
  sorry

end B_is_not_15_percent_less_than_A_l1631_163135


namespace xy_value_l1631_163153

theorem xy_value (x y : ℝ) (h1 : x + 2 * y = 8) (h2 : 2 * x + y = -5) : x + y = 1 := 
sorry

end xy_value_l1631_163153


namespace min_cos_C_l1631_163134

theorem min_cos_C (A B C : ℝ) (h : 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧ A + B + C = π)
  (h1 : (1 / Real.sin A) + (2 / Real.sin B) = 3 * ((1 / Real.tan A) + (1 / Real.tan B))) :
  Real.cos C ≥ (2 * Real.sqrt 10 - 2) / 9 := 
sorry

end min_cos_C_l1631_163134


namespace total_weight_of_hay_bales_l1631_163102

theorem total_weight_of_hay_bales
  (initial_bales : Nat) (weight_per_initial_bale : Nat)
  (total_bales_now : Nat) (weight_per_new_bale : Nat) : 
  (initial_bales = 73 ∧ weight_per_initial_bale = 45 ∧ 
   total_bales_now = 96 ∧ weight_per_new_bale = 50) →
  (73 * 45 + (96 - 73) * 50 = 4435) :=
by
  sorry

end total_weight_of_hay_bales_l1631_163102


namespace root_interval_l1631_163101

def f (x : ℝ) : ℝ := 5 * x - 7

theorem root_interval : ∃ x, 1 < x ∧ x < 2 ∧ f x = 0 :=
by
  -- Proof steps should be here
  sorry

end root_interval_l1631_163101


namespace intersection_count_is_one_l1631_163169

theorem intersection_count_is_one :
  (∀ x y : ℝ, y = 2 * x^3 + 6 * x + 1 → y = -3 / x^2) → ∃! p : ℝ × ℝ, p.2 = 2 * p.1^3 + 6 * p.1 + 1 ∧ p.2 = -3 / p.1 :=
sorry

end intersection_count_is_one_l1631_163169


namespace quadratic_has_real_root_for_any_t_l1631_163171

theorem quadratic_has_real_root_for_any_t (s : ℝ) :
  (∀ t : ℝ, ∃ x : ℝ, s * x^2 + t * x + s - 1 = 0) ↔ (0 < s ∧ s ≤ 1) :=
by
  sorry

end quadratic_has_real_root_for_any_t_l1631_163171


namespace profit_percentage_is_25_l1631_163105

variable (CP MP : ℝ) (d : ℝ)

/-- Given an article with a cost price of Rs. 85.5, a marked price of Rs. 112.5, 
    and a 5% discount on the marked price, the profit percentage on the cost 
    price is 25%. -/
theorem profit_percentage_is_25
  (hCP : CP = 85.5)
  (hMP : MP = 112.5)
  (hd : d = 0.05) :
  ((MP - (MP * d) - CP) / CP * 100) = 25 := 
sorry

end profit_percentage_is_25_l1631_163105


namespace select_four_person_committee_l1631_163131

open Nat

theorem select_four_person_committee 
  (n : ℕ)
  (h1 : (n * (n - 1) * (n - 2)) / 6 = 21) 
  : (n = 9) → Nat.choose n 4 = 126 :=
by
  sorry

end select_four_person_committee_l1631_163131


namespace tangent_curve_l1631_163193

theorem tangent_curve (a : ℝ) : 
  (∃ x : ℝ, 3 * x - 2 = x^3 - 2 * a ∧ 3 * x^2 = 3) →
  a = 0 ∨ a = 2 := 
sorry

end tangent_curve_l1631_163193


namespace induction_first_step_l1631_163184

theorem induction_first_step (n : ℕ) (h₁ : n > 1) : 
  1 + 1/2 + 1/3 < 2 := 
sorry

end induction_first_step_l1631_163184


namespace solve_z_plus_inv_y_l1631_163110

theorem solve_z_plus_inv_y (x y z : ℝ) (h1 : x * y * z = 1) (h2 : x + 1 / z = 4) (h3 : y + 1 / x = 30) :
  z + 1 / y = 36 / 119 :=
sorry

end solve_z_plus_inv_y_l1631_163110


namespace isosceles_if_interior_angles_equal_l1631_163139

-- Definition for a triangle
structure Triangle :=
  (A B C : Type)

-- Defining isosceles triangle condition
def is_isosceles (T : Triangle) :=
  ∃ a b c : ℝ, (a = b) ∨ (b = c) ∨ (a = c)

-- Defining the angle equality condition
def interior_angles_equal (T : Triangle) :=
  ∃ a b c : ℝ, (a = b) ∨ (b = c) ∨ (a = c)

-- Main theorem stating the contrapositive
theorem isosceles_if_interior_angles_equal (T : Triangle) : 
  interior_angles_equal T → is_isosceles T :=
by sorry

end isosceles_if_interior_angles_equal_l1631_163139


namespace female_democrats_count_l1631_163162

theorem female_democrats_count 
  (F M : ℕ) 
  (total_participants : F + M = 750)
  (female_democrats : ℕ := F / 2) 
  (male_democrats : ℕ := M / 4)
  (total_democrats : female_democrats + male_democrats = 250) :
  female_democrats = 125 := 
sorry

end female_democrats_count_l1631_163162


namespace hydrogen_atoms_in_compound_l1631_163124

theorem hydrogen_atoms_in_compound : 
  ∀ (Al_weight O_weight H_weight : ℕ) (total_weight : ℕ) (num_Al num_O num_H : ℕ),
  Al_weight = 27 →
  O_weight = 16 →
  H_weight = 1 →
  total_weight = 78 →
  num_Al = 1 →
  num_O = 3 →
  (num_Al * Al_weight + num_O * O_weight + num_H * H_weight = total_weight) →
  num_H = 3 := 
by
  intros
  sorry

end hydrogen_atoms_in_compound_l1631_163124


namespace num_words_with_consonant_l1631_163142

-- Definitions
def letters : List Char := ['A', 'B', 'C', 'D', 'E']
def vowels : List Char := ['A', 'E']
def consonants : List Char := ['B', 'C', 'D']

-- Total number of 4-letter words without restrictions
def total_words : Nat := 5 ^ 4

-- Number of 4-letter words with only vowels
def vowels_only_words : Nat := 2 ^ 4

-- Number of 4-letter words with at least one consonant
def words_with_consonant : Nat := total_words - vowels_only_words

theorem num_words_with_consonant : words_with_consonant = 609 := by
  -- Add proof steps
  sorry

end num_words_with_consonant_l1631_163142


namespace find_value_of_m_l1631_163191

def ellipse_condition (x y : ℝ) (m : ℝ) : Prop :=
  x^2 + m * y^2 = 1

theorem find_value_of_m (m : ℝ) 
  (h1 : ∀ (x y : ℝ), ellipse_condition x y m)
  (h2 : ∀ a b : ℝ, (a^2 = 1/m ∧ b^2 = 1) ∧ (a = 2 * b)) : 
  m = 1/4 :=
by
  sorry

end find_value_of_m_l1631_163191


namespace electrical_appliance_supermarket_l1631_163165

-- Define the known quantities and conditions
def purchase_price_A : ℝ := 140
def purchase_price_B : ℝ := 100
def week1_sales_A : ℕ := 4
def week1_sales_B : ℕ := 3
def week1_revenue : ℝ := 1250
def week2_sales_A : ℕ := 5
def week2_sales_B : ℕ := 5
def week2_revenue : ℝ := 1750
def total_units : ℕ := 50
def budget : ℝ := 6500
def profit_goal : ℝ := 2850

-- Define the unknown selling prices
noncomputable def selling_price_A : ℝ := 200
noncomputable def selling_price_B : ℝ := 150

-- Define the constraints
def cost_constraint (m : ℕ) : Prop := 140 * m + 100 * (50 - m) ≤ 6500
def profit_exceeds_goal (m : ℕ) : Prop := (200 - 140) * m + (150 - 100) * (50 - m) > 2850

-- The main theorem stating the results
theorem electrical_appliance_supermarket :
  (4 * selling_price_A + 3 * selling_price_B = week1_revenue)
  ∧ (5 * selling_price_A + 5 * selling_price_B = week2_revenue)
  ∧ (∃ m : ℕ, m ≤ 37 ∧ cost_constraint m)
  ∧ (∃ m : ℕ, m > 35 ∧ m ≤ 37 ∧ profit_exceeds_goal m) :=
sorry

end electrical_appliance_supermarket_l1631_163165


namespace intersection_M_N_l1631_163167

def M : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def N : Set ℝ := {x | x ≥ 1}

theorem intersection_M_N : M ∩ N = Set.Ico 1 3 := 
by
  sorry

end intersection_M_N_l1631_163167


namespace larger_of_two_numbers_l1631_163109

-- Define necessary conditions
def hcf : ℕ := 23
def factor1 : ℕ := 11
def factor2 : ℕ := 12
def lcm : ℕ := hcf * factor1 * factor2

-- Define the problem statement in Lean
theorem larger_of_two_numbers : ∃ (a b : ℕ), a = hcf * factor1 ∧ b = hcf * factor2 ∧ max a b = 276 := by
  sorry

end larger_of_two_numbers_l1631_163109


namespace sin_double_angle_values_l1631_163125

theorem sin_double_angle_values (α : ℝ) (hα : 0 < α ∧ α < π) (h : 3 * (Real.cos α)^2 = Real.sin ((π / 4) - α)) :
  Real.sin (2 * α) = 1 ∨ Real.sin (2 * α) = -17 / 18 :=
by
  sorry

end sin_double_angle_values_l1631_163125


namespace bluegrass_percentage_l1631_163141

theorem bluegrass_percentage (rx : ℝ) (ry : ℝ) (f : ℝ) (rm : ℝ) (wx : ℝ) (wy : ℝ) (B : ℝ) :
  rx = 0.4 →
  ry = 0.25 →
  f = 0.75 →
  rm = 0.35 →
  wx = 0.6667 →
  wy = 0.3333 →
  (wx * rx + wy * ry = rm) →
  B = 1.0 - rx →
  B = 0.6 :=
by 
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end bluegrass_percentage_l1631_163141


namespace parabola_focus_distance_l1631_163183

-- defining the problem in Lean
theorem parabola_focus_distance
  (A : ℝ × ℝ)
  (hA : A.2^2 = 4 * A.1)
  (h_distance : |A.1| = 3)
  (F : ℝ × ℝ)
  (hF : F = (1, 0)) :
  |(A.1 - F.1)^2 + (A.2 - F.2)^2| = 4 := 
sorry

end parabola_focus_distance_l1631_163183


namespace arnold_protein_intake_l1631_163176

def protein_in_collagen_powder (scoops : ℕ) : ℕ := if scoops = 1 then 9 else 18

def protein_in_protein_powder (scoops : ℕ) : ℕ := 21 * scoops

def protein_in_steak : ℕ := 56

def protein_in_greek_yogurt : ℕ := 15

def protein_in_almonds (cups : ℕ) : ℕ := 6 * cups

theorem arnold_protein_intake :
  protein_in_collagen_powder 1 + 
  protein_in_protein_powder 2 + 
  protein_in_steak + 
  protein_in_greek_yogurt + 
  protein_in_almonds 2 = 134 :=
by
  -- Sorry, the proof is omitted intentionally
  sorry

end arnold_protein_intake_l1631_163176


namespace sum_first_11_terms_eq_99_l1631_163156

variable {a_n : ℕ → ℝ} -- assuming the sequence values are real numbers
variable (S : ℕ → ℝ) -- sum of the first n terms
variable (a₃ a₆ a₉ : ℝ)
variable (h_sequence : ∀ n, a_n n = aₙ 1 + (n - 1) * (a_n 2 - aₙ 1)) -- sequence is arithmetic
variable (h_condition : a₃ + a₉ = 27 - a₆) -- given condition

theorem sum_first_11_terms_eq_99 
  (h_a₃ : a₃ = a_n 3) 
  (h_a₆ : a₆ = a_n 6) 
  (h_a₉ : a₉ = a_n 9) 
  (h_S : S 11 = 11 * a₆) : 
  S 11 = 99 := 
by 
  sorry


end sum_first_11_terms_eq_99_l1631_163156


namespace circle_radius_tangent_lines_l1631_163185

noncomputable def circle_radius (k : ℝ) (r : ℝ) : Prop :=
  k > 8 ∧ r = k / Real.sqrt 2 ∧ r = |k - 8|

theorem circle_radius_tangent_lines :
  ∃ k r : ℝ, k > 8 ∧ r = (k / Real.sqrt 2) ∧ r = |k - 8| ∧ r = 8 * Real.sqrt 2 :=
by
  sorry

end circle_radius_tangent_lines_l1631_163185


namespace wood_rope_equations_l1631_163155

theorem wood_rope_equations (x y : ℝ) (h1 : y - x = 4.5) (h2 : 0.5 * y = x - 1) :
  (y - x = 4.5) ∧ (0.5 * y = x - 1) :=
by
  sorry

end wood_rope_equations_l1631_163155


namespace max_S_value_l1631_163151

noncomputable def max_S (A C : ℝ) [DecidableEq ℝ] : ℝ :=
  if h : 0 < A ∧ A < 2 * Real.pi / 3 ∧ A + C = 2 * Real.pi / 3 then
    (Real.sqrt 3 / 6) * Real.sin (2 * A - Real.pi / 3) + (Real.sqrt 3 / 12)
  else
    0

theorem max_S_value :
  ∃ (A C : ℝ), A + C = 2 * Real.pi / 3 ∧
    (S = (Real.sqrt 3 / 3) * Real.sin A * Real.sin C) ∧
    (max_S A C = Real.sqrt 3 / 4) := 
sorry

end max_S_value_l1631_163151


namespace determine_polynomial_l1631_163106

theorem determine_polynomial (p : ℝ → ℝ) (h : ∀ x : ℝ, 1 + p x = (p (x - 1) + p (x + 1)) / 2) :
  ∃ b c : ℝ, ∀ x : ℝ, p x = x^2 + b * x + c := by
  sorry

end determine_polynomial_l1631_163106


namespace inequality_div_c_squared_l1631_163182

theorem inequality_div_c_squared (a b c : ℝ) (h : a > b) : (a / (c^2 + 1) > b / (c^2 + 1)) :=
by
  sorry

end inequality_div_c_squared_l1631_163182


namespace seeds_per_flowerbed_l1631_163130

theorem seeds_per_flowerbed :
  ∀ (total_seeds flowerbeds seeds_per_bed : ℕ), 
  total_seeds = 32 → 
  flowerbeds = 8 → 
  seeds_per_bed = total_seeds / flowerbeds → 
  seeds_per_bed = 4 :=
  by 
    intros total_seeds flowerbeds seeds_per_bed h_total h_flowerbeds h_calc
    rw [h_total, h_flowerbeds] at h_calc
    exact h_calc

end seeds_per_flowerbed_l1631_163130


namespace smallest_number_of_students_l1631_163190

theorem smallest_number_of_students 
  (A6 A7 A8 : Nat)
  (h1 : A8 * 3 = A6 * 5)
  (h2 : A8 * 5 = A7 * 8) :
  A6 + A7 + A8 = 89 :=
sorry

end smallest_number_of_students_l1631_163190


namespace max_bishops_on_chessboard_l1631_163174

theorem max_bishops_on_chessboard : ∃ n : ℕ, n = 14 ∧ (∃ k : ℕ, n * n = k^2) := 
by {
  sorry
}

end max_bishops_on_chessboard_l1631_163174


namespace find_missing_fraction_l1631_163159

theorem find_missing_fraction :
  ∃ (x : ℚ), (1/2 + -5/6 + 1/5 + 1/4 + -9/20 + -9/20 + x = 9/20) :=
  by
  sorry

end find_missing_fraction_l1631_163159


namespace sum_diff_l1631_163107

-- Define the lengths of the ropes
def shortest_rope_length := 80
def ratio_shortest := 4
def ratio_middle := 5
def ratio_longest := 6

-- Use the given ratio to find the common multiple x.
def x := shortest_rope_length / ratio_shortest

-- Find the lengths of the other ropes
def middle_rope_length := ratio_middle * x
def longest_rope_length := ratio_longest * x

-- Define the sum of the longest and shortest ropes
def sum_of_longest_and_shortest := longest_rope_length + shortest_rope_length

-- Define the difference between the sum of the longest and shortest rope and the middle rope
def difference := sum_of_longest_and_shortest - middle_rope_length

-- Theorem statement
theorem sum_diff : difference = 100 := by
  sorry

end sum_diff_l1631_163107


namespace number_of_words_with_at_least_one_consonant_l1631_163115

def total_5_letter_words : ℕ := 6 ^ 5

def total_5_letter_vowel_words : ℕ := 2 ^ 5

def total_5_letter_words_with_consonant : ℕ := total_5_letter_words - total_5_letter_vowel_words

theorem number_of_words_with_at_least_one_consonant :
  total_5_letter_words_with_consonant = 7744 :=
  by
    -- We assert the calculation follows correctly:
    -- total_5_letter_words == 6^5 = 7776
    -- total_5_letter_vowel_words == 2^5 = 32
    -- 7776 - 32 == 7744
    sorry

end number_of_words_with_at_least_one_consonant_l1631_163115


namespace predicted_sales_volume_l1631_163104

-- Define the linear regression equation
def regression_equation (x : ℝ) : ℝ := 2 * x + 60

-- Use the given condition x = 34
def temperature_value : ℝ := 34

-- State the theorem that the predicted sales volume is 128
theorem predicted_sales_volume : regression_equation temperature_value = 128 :=
by
  sorry

end predicted_sales_volume_l1631_163104


namespace inequality_xyz_l1631_163147

theorem inequality_xyz (x y : ℝ) (hx : x ≥ 1) (hy : y ≥ 1) : 
    x + y + 1 / (x * y) ≤ 1 / x + 1 / y + x * y := 
    sorry

end inequality_xyz_l1631_163147


namespace fraction_simplified_l1631_163103

-- Define the fraction function
def fraction (n : ℕ) := (21 * n + 4, 14 * n + 3)

-- Define the gcd function to check if fractions are simplified.
def is_simplified (a b : ℕ) : Prop := Nat.gcd a b = 1

-- Main theorem
theorem fraction_simplified (n : ℕ) : is_simplified (21 * n + 4) (14 * n + 3) :=
by
  -- Rest of the proof
  sorry

end fraction_simplified_l1631_163103


namespace problem_part_I_problem_part_II_l1631_163146

-- Define the function f(x) given by the problem
def f (x a : ℝ) : ℝ := x^2 - 2 * a * x + 5

-- Define the conditions for part (Ⅰ)
def conditions_part_I (a x : ℝ) : Prop :=
  (1 ≤ x ∧ x ≤ a) ∧ (1 ≤ f x a ∧ f x a ≤ a)

-- Lean statement for part (Ⅰ)
theorem problem_part_I (a : ℝ) (h : a > 1) :
  (∀ x, conditions_part_I a x) → a = 2 := by sorry

-- Define the conditions for part (Ⅱ)
def decreasing_on_interval (a : ℝ) : Prop :=
  ∀ x y, x ≤ y ∧ y ≤ 2 → f x a ≥ f y a

def abs_difference_condition (a : ℝ) : Prop :=
  ∀ x1 x2, 1 ≤ x1 ∧ x1 ≤ a + 1 ∧ 1 ≤ x2 ∧ x2 ≤ a + 1 → |f x1 a - f x2 a| ≤ 4

-- Lean statement for part (Ⅱ)
theorem problem_part_II (a : ℝ) (h : a > 1) :
  (decreasing_on_interval a) ∧ (abs_difference_condition a) → (2 ≤ a ∧ a ≤ 3) := by sorry

end problem_part_I_problem_part_II_l1631_163146


namespace arithmetic_geometric_sequence_l1631_163197

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) (d : ℝ) (h₀ : d ≠ 0)
    (h₁ : a 3 = a 1 + 2 * d) (h₂ : a 9 = a 1 + 8 * d)
    (h₃ : (a 1 + 2 * d)^2 = a 1 * (a 1 + 8 * d)) :
    (a 1 + a 3 + a 9) / (a 2 + a 4 + a 10) = 13 / 16 := 
sorry

end arithmetic_geometric_sequence_l1631_163197


namespace investor_share_purchase_price_l1631_163114

theorem investor_share_purchase_price 
  (dividend_rate : ℝ) 
  (face_value : ℝ) 
  (roi : ℝ) 
  (purchase_price : ℝ)
  (h1 : dividend_rate = 0.125)
  (h2 : face_value = 60)
  (h3 : roi = 0.25)
  (h4 : 0.25 = (0.125 * 60) / purchase_price) 
  : purchase_price = 30 := 
sorry

end investor_share_purchase_price_l1631_163114


namespace problem_statement_l1631_163195

noncomputable def a := 2 * Real.sqrt 2
noncomputable def b := 2
def ellipse_eq (x y : ℝ) := (x^2) / 8 + (y^2) / 4 = 1
def line_eq (x y m : ℝ) := y = x + m
def circle_eq (x y : ℝ) := x^2 + y^2 = 1

theorem problem_statement (x1 y1 x2 y2 x0 y0 m : ℝ) (h1 : ellipse_eq x1 y1) (h2 : ellipse_eq x2 y2) 
  (hm : line_eq x0 y0 m) (h0 : (x1 + x2) / 2 = -2 * m / 3) (h0' : (y1 + y2) / 2 = m / 3) : 
  (ellipse_eq x y ∧ line_eq x y m ∧ circle_eq x0 y0) → m = (3 * Real.sqrt 5) / 5 ∨ m = -(3 * Real.sqrt 5) / 5 := 
by {
  sorry
}

end problem_statement_l1631_163195


namespace sin_alpha_minus_pi_over_6_l1631_163128

open Real

theorem sin_alpha_minus_pi_over_6 (α : ℝ) (h : sin (α + π / 6) + 2 * sin (α / 2) ^ 2 = 1 - sqrt 2 / 2) : 
  sin (α - π / 6) = -sqrt 2 / 2 :=
sorry

end sin_alpha_minus_pi_over_6_l1631_163128

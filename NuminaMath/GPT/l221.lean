import Mathlib

namespace find_number_l221_22126

theorem find_number (x : ℚ) : (x + (-5/12) - (-5/2) = 1/3) → x = -7/4 :=
by
  sorry

end find_number_l221_22126


namespace sum_of_numbers_l221_22129

theorem sum_of_numbers (x y : ℝ) (h1 : x - y = 7) (h2 : x^2 + y^2 = 130) : x + y = -7 :=
by
  sorry

end sum_of_numbers_l221_22129


namespace jen_scored_more_l221_22146

def bryan_score : ℕ := 20
def total_points : ℕ := 35
def sammy_mistakes : ℕ := 7
def sammy_score : ℕ := total_points - sammy_mistakes
def jen_score : ℕ := sammy_score + 2

theorem jen_scored_more :
  jen_score - bryan_score = 10 := by
  -- Proof to be filled in
  sorry

end jen_scored_more_l221_22146


namespace abs_neg_six_l221_22106

theorem abs_neg_six : abs (-6) = 6 := by
  sorry

end abs_neg_six_l221_22106


namespace right_triangle_inequality_l221_22112

theorem right_triangle_inequality (a b c : ℝ) (h1 : a^2 + b^2 = c^2) (h2 : b > a) (h3 : b / a < 2) :
  a^2 / (b^2 + c^2) + b^2 / (a^2 + c^2) > 4 / 9 :=
by
  sorry

end right_triangle_inequality_l221_22112


namespace pencils_left_proof_l221_22102

noncomputable def total_pencils_left (a d : ℕ) : ℕ :=
  let total_initial_pencils : ℕ := 30
  let total_pencils_given_away : ℕ := 15 * a + 105 * d
  total_initial_pencils - total_pencils_given_away

theorem pencils_left_proof (a d : ℕ) :
  total_pencils_left a d = 30 - (15 * a + 105 * d) :=
by
  sorry

end pencils_left_proof_l221_22102


namespace possible_final_state_l221_22124

-- Definitions of initial conditions and operations
def initial_urn : (ℕ × ℕ) := (100, 100)  -- (W, B)

-- Define operations that describe changes in (white, black) marbles
inductive Operation
| operation1 : Operation
| operation2 : Operation
| operation3 : Operation
| operation4 : Operation

def apply_operation (op : Operation) (state : ℕ × ℕ) : ℕ × ℕ :=
  match op with
  | Operation.operation1 => (state.1, state.2 - 2)
  | Operation.operation2 => (state.1, state.2 - 1)
  | Operation.operation3 => (state.1, state.2 - 1)
  | Operation.operation4 => (state.1 - 2, state.2 + 1)

-- The final state in the form of the specific condition to prove.
def final_state (state : ℕ × ℕ) : Prop :=
  state = (2, 0)  -- 2 white marbles are an expected outcome.

-- Statement of the problem in Lean
theorem possible_final_state : ∃ (sequence : List Operation), 
  (sequence.foldl (fun state op => apply_operation op state) initial_urn).1 = 2 :=
sorry

end possible_final_state_l221_22124


namespace roots_of_polynomial_l221_22148

noncomputable def polynomial : Polynomial ℝ := Polynomial.X^3 + Polynomial.X^2 - 6 * Polynomial.X - 6

theorem roots_of_polynomial :
  (Polynomial.rootSet polynomial ℝ) = {-1, 3, -2} := 
sorry

end roots_of_polynomial_l221_22148


namespace factorize_x_squared_plus_2x_l221_22142

theorem factorize_x_squared_plus_2x (x : ℝ) : x^2 + 2 * x = x * (x + 2) :=
by
  sorry

end factorize_x_squared_plus_2x_l221_22142


namespace john_total_cost_l221_22107

-- Define the costs and usage details
def base_cost : ℕ := 25
def cost_per_text_cent : ℕ := 10
def cost_per_extra_minute_cent : ℕ := 15
def included_hours : ℕ := 20
def texts_sent : ℕ := 150
def hours_talked : ℕ := 22

-- Prove that the total cost John had to pay is $58
def total_cost_john : ℕ :=
  let base_cost_dollars := base_cost
  let text_cost_dollars := (texts_sent * cost_per_text_cent) / 100
  let extra_minutes := (hours_talked - included_hours) * 60
  let extra_minutes_cost_dollars := (extra_minutes * cost_per_extra_minute_cent) / 100
  base_cost_dollars + text_cost_dollars + extra_minutes_cost_dollars

theorem john_total_cost (h1 : base_cost = 25)
                        (h2 : cost_per_text_cent = 10)
                        (h3 : cost_per_extra_minute_cent = 15)
                        (h4 : included_hours = 20)
                        (h5 : texts_sent = 150)
                        (h6 : hours_talked = 22) : 
  total_cost_john = 58 := by
  sorry

end john_total_cost_l221_22107


namespace product_of_solutions_l221_22179

theorem product_of_solutions (t : ℝ) (h : t^2 = 64) : t * (-t) = -64 :=
sorry

end product_of_solutions_l221_22179


namespace strawberry_jelly_amount_l221_22173

def totalJelly : ℕ := 6310
def blueberryJelly : ℕ := 4518
def strawberryJelly : ℕ := totalJelly - blueberryJelly

theorem strawberry_jelly_amount : strawberryJelly = 1792 := by
  rfl

end strawberry_jelly_amount_l221_22173


namespace vector_dot_product_calculation_l221_22192

theorem vector_dot_product_calculation : 
  let a := (2, 3, -1)
  let b := (2, 0, 3)
  let c := (0, 2, 2)
  (2 * (2 + 0) + 3 * (0 + 2) + -1 * (3 + 2)) = 5 := 
by
  sorry

end vector_dot_product_calculation_l221_22192


namespace out_of_pocket_expense_l221_22193

theorem out_of_pocket_expense :
  let initial_purchase := 3000
  let tv_return := 700
  let bike_return := 500
  let sold_bike_cost := bike_return + (0.20 * bike_return)
  let sold_bike_sell_price := 0.80 * sold_bike_cost
  let toaster_purchase := 100
  (initial_purchase - tv_return - bike_return - sold_bike_sell_price + toaster_purchase) = 1420 :=
by
  sorry

end out_of_pocket_expense_l221_22193


namespace range_a_l221_22118

theorem range_a (a : ℝ) : 
  (∃ x : ℝ, 0 < x ∧ x < 1 ∧ a * x^2 - x - 1 = 0 ∧ 
  ∀ y : ℝ, (0 < y ∧ y < 1 ∧ a * y^2 - y - 1 = 0 → y = x)) ↔ a > 2 :=
by
  sorry

end range_a_l221_22118


namespace arithmetic_mean_eq_2_l221_22175

theorem arithmetic_mean_eq_2 (a x : ℝ) (hx: x ≠ 0) :
  (1/2) * (((2 * x + a) / x) + ((2 * x - a) / x)) = 2 :=
by
  sorry

end arithmetic_mean_eq_2_l221_22175


namespace study_video_game_inversely_proportional_1_study_video_game_inversely_proportional_2_l221_22161

theorem study_video_game_inversely_proportional_1 (k : ℕ) (s : ℕ) (v : ℕ) 
  (h1 : s * v = k) (h2 : k = 12) (h3 : s = 6) : v = 2 :=
by
  sorry

theorem study_video_game_inversely_proportional_2 (k : ℕ) (s : ℕ) (v : ℕ) 
  (h1 : s * v = k) (h2 : k = 12) (h3 : v = 6) : s = 2 :=
by
  sorry

end study_video_game_inversely_proportional_1_study_video_game_inversely_proportional_2_l221_22161


namespace round_trip_ticket_percentage_l221_22178

variable (P : ℝ) -- Denotes total number of passengers
variable (R : ℝ) -- Denotes number of round-trip ticket holders

-- Condition 1: 15% of passengers held round-trip tickets and took their cars aboard
def condition1 : Prop := 0.15 * P = 0.40 * R

-- Prove that 37.5% of the ship's passengers held round-trip tickets.
theorem round_trip_ticket_percentage (h1 : condition1 P R) : R / P = 0.375 :=
by
  sorry

end round_trip_ticket_percentage_l221_22178


namespace monomial_coeff_degree_product_l221_22122

theorem monomial_coeff_degree_product (m n : ℚ) (h₁ : m = -3/4) (h₂ : n = 4) : m * n = -3 := 
by
  sorry

end monomial_coeff_degree_product_l221_22122


namespace length_is_56_l221_22190

noncomputable def length_of_plot (b : ℝ) : ℝ := b + 12

theorem length_is_56 (b : ℝ) (cost_per_meter : ℝ) (total_cost : ℝ) (h_cost : cost_per_meter = 26.50) (h_total_cost : total_cost = 5300) (h_fencing : 26.50 * (4 * b + 24) = 5300) : length_of_plot b = 56 := 
by 
  sorry

end length_is_56_l221_22190


namespace evaluate_expression_l221_22182

theorem evaluate_expression :
  (5^5 * 5^3) / 3^6 * 2^5 = 12480000 / 729 :=
by sorry

end evaluate_expression_l221_22182


namespace astronaut_total_days_l221_22115

-- Definitions of the regular and leap seasons.
def regular_season_days := 49
def leap_season_days := 51

-- Definition of the number of days in different types of years.
def days_in_regular_year := 2 * regular_season_days + 3 * leap_season_days
def days_in_first_3_years := 2 * regular_season_days + 3 * (leap_season_days + 1)
def days_in_years_7_to_9 := 2 * regular_season_days + 3 * (leap_season_days + 2)

-- Calculation for visits.
def first_visit := regular_season_days
def second_visit := 2 * regular_season_days + 3 * (leap_season_days + 1)
def third_visit := 3 * (2 * regular_season_days + 3 * (leap_season_days + 1))
def fourth_visit := 4 * days_in_regular_year + 3 * days_in_first_3_years + 3 * days_in_years_7_to_9

-- Total days spent.
def total_days := first_visit + second_visit + third_visit + fourth_visit

-- The proof statement.
theorem astronaut_total_days : total_days = 3578 :=
by
  -- We place a sorry here to skip the proof.
  sorry

end astronaut_total_days_l221_22115


namespace subscription_ways_three_households_l221_22147

def num_subscription_ways (n_households : ℕ) (n_newspapers : ℕ) : ℕ :=
  if h : n_households = 3 ∧ n_newspapers = 5 then
    180
  else
    0

theorem subscription_ways_three_households :
  num_subscription_ways 3 5 = 180 :=
by
  unfold num_subscription_ways
  split_ifs
  . rfl
  . contradiction


end subscription_ways_three_households_l221_22147


namespace find_y_l221_22101

variable {a b y : ℝ}
variable (ha : a ≠ 0) (hb : b ≠ 0)

theorem find_y (h1 : (3 * a) ^ (4 * b) = a ^ b * y ^ b) : y = 81 * a ^ 3 := by
  sorry

end find_y_l221_22101


namespace johnsonville_max_members_l221_22139

theorem johnsonville_max_members 
  (n : ℤ) 
  (h1 : 15 * n % 30 = 6) 
  (h2 : 15 * n < 900) 
  : 15 * n ≤ 810 :=
sorry

end johnsonville_max_members_l221_22139


namespace alice_paid_24_percent_l221_22105

theorem alice_paid_24_percent (P : ℝ) (h1 : P > 0) :
  let MP := 0.60 * P
  let price_paid := 0.40 * MP
  (price_paid / P) * 100 = 24 :=
by
  sorry

end alice_paid_24_percent_l221_22105


namespace total_carrots_l221_22198

-- Define constants for the number of carrots grown by each person
def Joan_carrots : ℕ := 29
def Jessica_carrots : ℕ := 11
def Michael_carrots : ℕ := 37
def Taylor_carrots : ℕ := 24

-- The proof problem: Prove that the total number of carrots grown is 101
theorem total_carrots : Joan_carrots + Jessica_carrots + Michael_carrots + Taylor_carrots = 101 :=
by
  sorry

end total_carrots_l221_22198


namespace triangle_properties_l221_22185

-- Define the given sides of the triangle
def a := 6
def b := 8
def c := 10

-- Define necessary parameters and properties
def isRightTriangle (a b c : Nat) : Prop := a^2 + b^2 = c^2
def area (a b : Nat) : Nat := (a * b) / 2
def semiperimeter (a b c : Nat) : Nat := (a + b + c) / 2
def inradius (A s : Nat) : Nat := A / s
def circumradius (c : Nat) : Nat := c / 2

-- The theorem statement
theorem triangle_properties :
  isRightTriangle a b c ∧
  area a b = 24 ∧
  semiperimeter a b c = 12 ∧
  inradius (area a b) (semiperimeter a b c) = 2 ∧
  circumradius c = 5 :=
by
  sorry

end triangle_properties_l221_22185


namespace fractions_equivalence_l221_22110

theorem fractions_equivalence (k : ℝ) (h : k ≠ -5) : (k + 3) / (k + 5) = 3 / 5 ↔ k = 0 := 
by 
  sorry

end fractions_equivalence_l221_22110


namespace sequence_contains_prime_l221_22174

-- Define the conditions for being square-free and relatively prime
def is_square_free (n : ℕ) : Prop :=
  ∀ m : ℕ, m^2 ∣ n → m = 1

def are_relatively_prime (a b : ℕ) : Prop :=
  Nat.gcd a b = 1

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Statement of the problem
theorem sequence_contains_prime :
  ∀ (a : ℕ → ℕ), (∀ i, 1 ≤ i ∧ i ≤ 14 → 2 ≤ a i ∧ a i ≤ 1995 ∧ is_square_free (a i)) →
  (∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ 14 → are_relatively_prime (a i) (a j)) →
  ∃ i, 1 ≤ i ∧ i ≤ 14 ∧ is_prime (a i) :=
sorry

end sequence_contains_prime_l221_22174


namespace problem_l221_22130

theorem problem (a b c : ℝ) (Ha : a > 0) (Hb : b > 0) (Hc : c > 0) : 
  (|a| / a + |b| / b + |c| / c - (abc / |abc|) = 2 ∨ |a| / a + |b| / b + |c| / c - (abc / |abc|) = -2) :=
by
  sorry

end problem_l221_22130


namespace square_area_when_a_eq_b_eq_c_l221_22183

theorem square_area_when_a_eq_b_eq_c {a b c : ℝ} (h : a = b ∧ b = c) :
  ∃ x : ℝ, (x = a * Real.sqrt 2) ∧ (x ^ 2 = 2 * a ^ 2) :=
by
  sorry

end square_area_when_a_eq_b_eq_c_l221_22183


namespace value_of_g_at_neg2_l221_22150

def g (x : ℝ) : ℝ := x^2 - 4*x + 3

theorem value_of_g_at_neg2 : g (-2) = 15 :=
by
  -- This is where the proof steps would go, but we'll skip it
  sorry

end value_of_g_at_neg2_l221_22150


namespace problem1_problem2_l221_22165

-- Problem 1: Prove that (2sin(α) - cos(α)) / (sin(α) + 2cos(α)) = 3/4 given tan(α) = 2
theorem problem1 (α : ℝ) (h : Real.tan α = 2) : (2 * Real.sin α - Real.cos α) / (Real.sin α + 2 * Real.cos α) = 3 / 4 := 
sorry

-- Problem 2: Prove that 2sin^2(x) - sin(x)cos(x) + cos^2(x) = 2 - sin(2x)/2
theorem problem2 (x : ℝ) : 2 * (Real.sin x)^2 - (Real.sin x) * (Real.cos x) + (Real.cos x)^2 = 2 - Real.sin (2 * x) / 2 := 
sorry

end problem1_problem2_l221_22165


namespace tray_contains_correct_number_of_pieces_l221_22189

-- Define the dimensions of the tray
def tray_width : ℕ := 24
def tray_length : ℕ := 20
def tray_area : ℕ := tray_width * tray_length

-- Define the dimensions of each brownie piece
def piece_width : ℕ := 3
def piece_length : ℕ := 4
def piece_area : ℕ := piece_width * piece_length

-- Define the goal: the number of pieces of brownies that the tray contains
def num_pieces : ℕ := tray_area / piece_area

-- The statement to prove
theorem tray_contains_correct_number_of_pieces :
  num_pieces = 40 :=
by
  sorry

end tray_contains_correct_number_of_pieces_l221_22189


namespace fill_half_cistern_time_l221_22137

variable (t_half : ℝ)

-- Define a condition that states the certain amount of time to fill 1/2 of the cistern.
def fill_pipe_half_time (t_half : ℝ) : Prop :=
  t_half > 0

-- The statement to prove that t_half is the time required to fill 1/2 of the cistern.
theorem fill_half_cistern_time : fill_pipe_half_time t_half → t_half = t_half := by
  intros
  rfl

end fill_half_cistern_time_l221_22137


namespace joao_claudia_scores_l221_22128

theorem joao_claudia_scores (joao_score claudia_score total_score : ℕ) 
  (h1 : claudia_score = joao_score + 13)
  (h2 : total_score = joao_score + claudia_score)
  (h3 : 100 ≤ total_score ∧ total_score < 200) :
  joao_score = 68 ∧ claudia_score = 81 := by
  sorry

end joao_claudia_scores_l221_22128


namespace rupert_candles_l221_22117

theorem rupert_candles (peter_candles : ℕ) (rupert_times_older : ℝ) (h1 : peter_candles = 10) (h2 : rupert_times_older = 3.5) :
    ∃ rupert_candles : ℕ, rupert_candles = peter_candles * rupert_times_older := 
by
  sorry

end rupert_candles_l221_22117


namespace yuna_solved_problems_l221_22108

def yuna_problems_per_day : ℕ := 8
def days_per_week : ℕ := 7
def yuna_weekly_problems : ℕ := 56

theorem yuna_solved_problems :
  yuna_problems_per_day * days_per_week = yuna_weekly_problems := by
  sorry

end yuna_solved_problems_l221_22108


namespace basketball_prob_l221_22151

theorem basketball_prob :
  let P_A := 0.7
  let P_B := 0.6
  P_A * P_B = 0.88 := 
by 
  sorry

end basketball_prob_l221_22151


namespace geometric_progression_condition_l221_22162

theorem geometric_progression_condition (a b c d : ℝ) :
  (∃ r : ℝ, (b = a * r ∨ b = a * -r) ∧
             (c = a * r^2 ∨ c = a * (-r)^2) ∧
             (d = a * r^3 ∨ d = a * (-r)^3) ∧
             (a = b / r ∨ a = b / -r) ∧
             (b = c / r ∨ b = c / -r) ∧
             (c = d / r ∨ c = d / -r) ∧
             (d = a / r ∨ d = a / -r)) ↔
  (a = b ∨ a = -b) ∧ (a = c ∨ a = -c) ∧ (a = d ∨ a = -d) := sorry

end geometric_progression_condition_l221_22162


namespace math_problem_l221_22121

theorem math_problem : 1012^2 - 992^2 - 1009^2 + 995^2 = 12024 := sorry

end math_problem_l221_22121


namespace difference_length_breadth_l221_22104

theorem difference_length_breadth (B L A : ℕ) (h1 : B = 11) (h2 : A = 21 * B) (h3 : A = L * B) :
  L - B = 10 :=
by
  sorry

end difference_length_breadth_l221_22104


namespace cost_of_50_snacks_l221_22196

-- Definitions based on conditions
def travel_time_to_work : ℕ := 2 -- hours
def cost_of_snack : ℕ := 10 * (2 * travel_time_to_work) -- Ten times the round trip time

-- The theorem to prove
theorem cost_of_50_snacks : (50 * cost_of_snack) = 2000 := by
  sorry

end cost_of_50_snacks_l221_22196


namespace find_first_spill_l221_22125

def bottle_capacity : ℕ := 20
def refill_count : ℕ := 3
def days : ℕ := 7
def total_water_drunk : ℕ := 407
def second_spill : ℕ := 8

theorem find_first_spill :
  let total_without_spill := bottle_capacity * refill_count * days
  let total_spilled := total_without_spill - total_water_drunk
  let first_spill := total_spilled - second_spill
  first_spill = 5 :=
by
  -- Proof goes here.
  sorry

end find_first_spill_l221_22125


namespace find_green_hats_l221_22172

variable (B G : ℕ)

theorem find_green_hats (h1 : B + G = 85) (h2 : 6 * B + 7 * G = 540) :
  G = 30 :=
by
  sorry

end find_green_hats_l221_22172


namespace Z_3_5_value_l221_22187

def Z (a b : ℕ) : ℕ :=
  b + 12 * a - a ^ 2

theorem Z_3_5_value : Z 3 5 = 32 := by
  sorry

end Z_3_5_value_l221_22187


namespace can_split_3x3x3_into_9_corners_l221_22181

-- Define the conditions
def number_of_cubes_in_3x3x3 : ℕ := 27
def number_of_units_in_corner : ℕ := 3
def number_of_corners : ℕ := 9

-- Prove the proposition
theorem can_split_3x3x3_into_9_corners :
  (number_of_corners * number_of_units_in_corner = number_of_cubes_in_3x3x3) :=
by
  sorry

end can_split_3x3x3_into_9_corners_l221_22181


namespace cosine_angle_between_vectors_l221_22109

noncomputable def vector_cosine (a b : ℝ × ℝ) : ℝ :=
  let dot_product := (a.1 * b.1 + a.2 * b.2)
  let magnitude_a := Real.sqrt (a.1 ^ 2 + a.2 ^ 2)
  let magnitude_b := Real.sqrt (b.1 ^ 2 + b.2 ^ 2)
  dot_product / (magnitude_a * magnitude_b)

theorem cosine_angle_between_vectors : ∀ (k : ℝ), 
  let a := (3, 1)
  let b := (1, 3)
  let c := (k, -2)
  (3 - k) / 3 = 1 →
  vector_cosine a c = Real.sqrt 5 / 5 := by
  intros
  sorry

end cosine_angle_between_vectors_l221_22109


namespace electric_car_charging_cost_l221_22114

/-- The fractional equation for the given problem,
    along with the correct solution for the average charging cost per kilometer. -/
theorem electric_car_charging_cost (
    x : ℝ
) : 
    (200 / x = 4 * (200 / (x + 0.6))) → x = 0.2 :=
by
  intros h_eq
  sorry

end electric_car_charging_cost_l221_22114


namespace sum_of_remainders_11111k_43210_eq_141_l221_22113

theorem sum_of_remainders_11111k_43210_eq_141 :
  (List.sum (List.map (fun k => (11111 * k + 43210) % 31) [0, 1, 2, 3, 4, 5])) = 141 :=
by
  -- Proof is omitted: sorry
  sorry

end sum_of_remainders_11111k_43210_eq_141_l221_22113


namespace distance_between_consecutive_trees_l221_22149

noncomputable def distance_between_trees (yard_length : ℝ) (num_trees : ℕ) (obstacle_pos : ℝ) (obstacle_gap : ℝ) : ℝ :=
  let planting_distance := yard_length - obstacle_gap
  let num_gaps := num_trees - 1
  planting_distance / num_gaps

theorem distance_between_consecutive_trees :
  distance_between_trees 600 36 250 10 = 16.857 := by
  sorry

end distance_between_consecutive_trees_l221_22149


namespace students_on_zoo_trip_l221_22133

theorem students_on_zoo_trip (buses : ℕ) (students_per_bus : ℕ) (students_in_cars : ℕ) 
  (h1 : buses = 7) (h2 : students_per_bus = 56) (h3 : students_in_cars = 4) : 
  buses * students_per_bus + students_in_cars = 396 :=
by
  sorry

end students_on_zoo_trip_l221_22133


namespace fraction_equivalence_l221_22164

theorem fraction_equivalence (a b c : ℝ) (h : (c - a) / (c - b) = 1) : 
  (5 * b - 2 * a) / (c - a) = 3 * a / (c - a) :=
by
  sorry

end fraction_equivalence_l221_22164


namespace picnic_basket_cost_l221_22152

theorem picnic_basket_cost :
  let sandwich_cost := 5
  let fruit_salad_cost := 3
  let soda_cost := 2
  let snack_bag_cost := 4
  let num_people := 4
  let num_sodas_per_person := 2
  let num_snack_bags := 3
  (num_people * sandwich_cost) + (num_people * fruit_salad_cost) + (num_people * num_sodas_per_person * soda_cost) + (num_snack_bags * snack_bag_cost) = 60 :=
by
  sorry

end picnic_basket_cost_l221_22152


namespace simplify_expression_l221_22177

theorem simplify_expression (t : ℝ) : (t ^ 5 * t ^ 3) / t ^ 2 = t ^ 6 :=
by
  sorry

end simplify_expression_l221_22177


namespace unique_solution_to_equation_l221_22144

theorem unique_solution_to_equation (x y z t : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (ht : t > 0) 
  (h : 1 + 5^x = 2^y + 2^z * 5^t) : (x, y, z, t) = (2, 4, 1, 1) := 
sorry

end unique_solution_to_equation_l221_22144


namespace total_pigs_in_barn_l221_22169

-- Define the number of pigs initially in the barn
def initial_pigs : ℝ := 2465.25

-- Define the number of pigs that join
def joining_pigs : ℝ := 5683.75

-- Define the total number of pigs after they join
def total_pigs : ℝ := 8149

-- The theorem that states the total number of pigs is the sum of initial and joining pigs
theorem total_pigs_in_barn : initial_pigs + joining_pigs = total_pigs := 
by
  sorry

end total_pigs_in_barn_l221_22169


namespace minimize_sum_of_digits_l221_22197

-- Define the sum of digits function
def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

-- Define the expression in the problem
def expression (p : ℕ) : ℕ :=
  p^4 - 5 * p^2 + 13

-- Proposition stating the conditions and the expected result
theorem minimize_sum_of_digits (p : ℕ) (h_prime : Nat.Prime p) (h_odd : p % 2 = 1) :
  (∀ q : ℕ, Nat.Prime q → q % 2 = 1 → sum_of_digits (expression q) ≥ sum_of_digits (expression 5)) →
  p = 5 :=
by
  sorry

end minimize_sum_of_digits_l221_22197


namespace ratio_celeste_bianca_l221_22136

-- Definitions based on given conditions
def bianca_hours : ℝ := 12.5
def celest_hours (x : ℝ) : ℝ := 12.5 * x
def mcclain_hours (x : ℝ) : ℝ := 12.5 * x - 8.5

-- The total time worked in hours
def total_hours : ℝ := 54

-- The ratio to prove
def celeste_bianca_ratio : ℝ := 2

-- The proof statement
theorem ratio_celeste_bianca (x : ℝ) (hx :  12.5 + 12.5 * x + (12.5 * x - 8.5) = total_hours) :
  celest_hours 2 / bianca_hours = celeste_bianca_ratio :=
by
  sorry

end ratio_celeste_bianca_l221_22136


namespace average_rate_second_drive_l221_22176

theorem average_rate_second_drive 
 (distance : ℕ) (total_time : ℕ) (d1 d2 d3 : ℕ)
 (t1 t2 t3 : ℕ) (r1 r2 r3 : ℕ)
 (h_distance : d1 = d2 ∧ d2 = d3 ∧ d1 + d2 + d3 = distance)
 (h_total_time : t1 + t2 + t3 = total_time)
 (h_drive_1 : r1 = 4 ∧ t1 = d1 / r1)
 (h_drive_2 : r3 = 6 ∧ t3 = d3 / r3)
 (h_distance_total : distance = 180)
 (h_total_time_val : total_time = 37)
  : r2 = 5 := 
by sorry

end average_rate_second_drive_l221_22176


namespace deposit_amount_l221_22141

theorem deposit_amount (P : ℝ) (deposit remaining : ℝ) (h1 : deposit = 0.1 * P) (h2 : remaining = P - deposit) (h3 : remaining = 1350) : 
  deposit = 150 := 
by
  sorry

end deposit_amount_l221_22141


namespace reflection_points_line_l221_22158

theorem reflection_points_line (m b : ℝ)
  (h1 : (10 : ℝ) = 2 * (6 - m * (6 : ℝ) + b)) -- Reflecting the point (6, (m * 6 + b)) to (10, 7)
  (h2 : (6 : ℝ) * m + b = 5) -- Midpoint condition
  (h3 : (6 : ℝ) = (2 + 10) / 2) -- Calculating midpoint x-coordinate
  (h4 : (5 : ℝ) = (3 + 7) / 2) -- Calculating midpoint y-coordinate
  : m + b = 15 :=
sorry

end reflection_points_line_l221_22158


namespace symmetric_circle_equation_l221_22153

noncomputable def equation_of_symmetric_circle (x y : ℝ) : Prop :=
  (x^2 + y^2 - 2 * x - 6 * y + 9 = 0) ∧ (2 * x + y + 5 = 0)

theorem symmetric_circle_equation :
  ∀ (x y : ℝ), 
    equation_of_symmetric_circle x y → 
    ∃ a b : ℝ, ((x - a)^2 + (y - b)^2 = 1) ∧ (a + 7 = 0) ∧ (b + 1 = 0) :=
sorry

end symmetric_circle_equation_l221_22153


namespace michael_passes_donovan_after_laps_l221_22171

/-- The length of the track in meters -/
def track_length : ℕ := 400

/-- Donovan's lap time in seconds -/
def donovan_lap_time : ℕ := 45

/-- Michael's lap time in seconds -/
def michael_lap_time : ℕ := 36

/-- The number of laps that Michael will have to complete in order to pass Donovan -/
theorem michael_passes_donovan_after_laps : 
  ∃ (laps : ℕ), laps = 5 ∧ (∃ t : ℕ, 400 * t / 36 = 5 ∧ 400 * t / 45 < 5) :=
sorry

end michael_passes_donovan_after_laps_l221_22171


namespace sum_of_integers_l221_22154

theorem sum_of_integers (x y : ℕ) (h1 : x - y = 8) (h2 : x * y = 120) : x + y = 4 * Real.sqrt 34 := by
  sorry

end sum_of_integers_l221_22154


namespace arithmetic_seq_third_sum_l221_22156

-- Define the arithmetic sequence using its first term and common difference
def arithmetic_seq (a₁ d : ℤ) (n : ℕ) : ℤ := a₁ + d * n

theorem arithmetic_seq_third_sum
  (a₁ d : ℤ)
  (h1 : (a₁ + (a₁ + 3 * d) + (a₁ + 6 * d) = 39))
  (h2 : ((a₁ + d) + (a₁ + 4 * d) + (a₁ + 7 * d) = 33)) :
  ((a₁ + 2 * d) + (a₁ + 5 * d) + (a₁ + 8 * d) = 27) :=
by
  sorry

end arithmetic_seq_third_sum_l221_22156


namespace inequality_proof_l221_22135

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
    (x / (y + z)) + (y / (z + x)) + (z / (x + y)) ≥ (3 / 2) :=
sorry

end inequality_proof_l221_22135


namespace sherman_drives_nine_hours_a_week_l221_22166

-- Define the daily commute time in minutes.
def daily_commute_time := 30 + 30

-- Define the number of weekdays Sherman commutes.
def weekdays := 5

-- Define the weekly commute time in minutes.
def weekly_commute_time := weekdays * daily_commute_time

-- Define the conversion from minutes to hours.
def minutes_to_hours (m : ℕ) : ℕ := m / 60

-- Define the weekend driving time in hours.
def weekend_driving_time := 2 * 2

-- Define the total weekly driving time in hours.
def total_weekly_driving_time := minutes_to_hours weekly_commute_time + weekend_driving_time

-- The theorem we need to prove
theorem sherman_drives_nine_hours_a_week :
  total_weekly_driving_time = 9 :=
by
  sorry

end sherman_drives_nine_hours_a_week_l221_22166


namespace planes_parallel_from_plane_l221_22170

-- Define the relationship functions
def parallel (P Q : Plane) : Prop := sorry -- Define parallelism predicate
def perpendicular (l : Line) (P : Plane) : Prop := sorry -- Define perpendicularity predicate

-- Declare the planes α, β, and γ
variable (α β γ : Plane)

-- Main theorem statement
theorem planes_parallel_from_plane (h1 : parallel γ α) (h2 : parallel γ β) : parallel α β := 
sorry

end planes_parallel_from_plane_l221_22170


namespace solution_exists_in_interval_l221_22188

noncomputable def f (x : ℝ) : ℝ := 3^x + x - 3

theorem solution_exists_in_interval : ∃ x, 0 < x ∧ x < 1 ∧ f x = 0 :=
by {
  -- placeholder for the skipped proof
  sorry
}

end solution_exists_in_interval_l221_22188


namespace sequence_term_sum_max_value_sum_equality_l221_22163

noncomputable def a (n : ℕ) : ℝ := -2 * n + 6

def S (n : ℕ) : ℝ := -n^2 + 5 * n

theorem sequence_term (n : ℕ) : ∀ n, a n = 4 + (n - 1) * (-2) :=
by sorry

theorem sum_max_value (n : ℕ) : ∃ n, S n = 6 :=
by sorry

theorem sum_equality : S 2 = 6 ∧ S 3 = 6 :=
by sorry

end sequence_term_sum_max_value_sum_equality_l221_22163


namespace uranus_appearance_minutes_after_6AM_l221_22184

-- Definitions of the given times and intervals
def mars_last_seen : Int := 0 -- 12:10 AM in minutes after midnight
def jupiter_after_mars : Int := 2 * 60 + 41 -- 2 hours and 41 minutes in minutes
def uranus_after_jupiter : Int := 3 * 60 + 16 -- 3 hours and 16 minutes in minutes
def reference_time : Int := 6 * 60 -- 6:00 AM in minutes after midnight

-- Statement of the problem
theorem uranus_appearance_minutes_after_6AM :
  let jupiter_first_appearance := mars_last_seen + jupiter_after_mars
  let uranus_first_appearance := jupiter_first_appearance + uranus_after_jupiter
  (uranus_first_appearance - reference_time) = 7 := by
  sorry

end uranus_appearance_minutes_after_6AM_l221_22184


namespace compare_logarithms_l221_22157

noncomputable def a : ℝ := Real.log 3 / Real.log 4 -- log base 4 of 3
noncomputable def b : ℝ := Real.log 4 / Real.log 3 -- log base 3 of 4
noncomputable def c : ℝ := Real.log 3 / Real.log 5 -- log base 5 of 3

theorem compare_logarithms : b > a ∧ a > c := sorry

end compare_logarithms_l221_22157


namespace alan_tickets_l221_22194

theorem alan_tickets (a m : ℕ) (h1 : a + m = 150) (h2 : m = 5 * a - 6) : a = 26 :=
by
  sorry

end alan_tickets_l221_22194


namespace twice_total_credits_l221_22132

theorem twice_total_credits (Aria Emily Spencer : ℕ) 
(Emily_has_20_credits : Emily = 20) 
(Aria_twice_Emily : Aria = 2 * Emily) 
(Emily_twice_Spencer : Emily = 2 * Spencer) : 
2 * (Aria + Emily + Spencer) = 140 :=
by
  sorry

end twice_total_credits_l221_22132


namespace min_value_x_plus_inv_x_l221_22103

theorem min_value_x_plus_inv_x (x : ℝ) (hx : x > 0) : ∃ y, (y = x + 1/x) ∧ (∀ z, z = x + 1/x → z ≥ 2) :=
by
  sorry

end min_value_x_plus_inv_x_l221_22103


namespace john_february_bill_l221_22195

-- Define the conditions as constants
def base_cost : ℝ := 25
def cost_per_text : ℝ := 0.1 -- 10 cents
def cost_per_over_minute : ℝ := 0.1 -- 10 cents
def texts_sent : ℝ := 200
def hours_talked : ℝ := 51
def included_hours : ℝ := 50
def minutes_per_hour : ℝ := 60

-- Total cost computation
def total_cost : ℝ :=
  base_cost +
  (texts_sent * cost_per_text) +
  ((hours_talked - included_hours) * minutes_per_hour * cost_per_over_minute)

-- Proof statement
theorem john_february_bill : total_cost = 51 := by
  -- Proof omitted
  sorry

end john_february_bill_l221_22195


namespace minimum_sugar_quantity_l221_22138

theorem minimum_sugar_quantity :
  ∃ s f : ℝ, s = 4 ∧ f ≥ 4 + s / 3 ∧ f ≤ 3 * s ∧ 2 * s + 3 * f ≤ 36 :=
sorry

end minimum_sugar_quantity_l221_22138


namespace find_r_l221_22145

noncomputable def parabola_vertex : (ℝ × ℝ) := (0, -1)

noncomputable def intersection_points (r : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  let x1 := (r - Real.sqrt (r^2 + 4)) / 2
  let y1 := r * x1
  let x2 := (r + Real.sqrt (r^2 + 4)) / 2
  let y2 := r * x2
  ((x1, y1), (x2, y2))

noncomputable def triangle_area (r : ℝ) : ℝ :=
  let base := Real.sqrt (r^2 + 4)
  let height := 2
  1/2 * base * height

theorem find_r (r : ℝ) (h : r > 0) : triangle_area r = 32 → r = Real.sqrt 1020 := 
by
  sorry

end find_r_l221_22145


namespace even_function_a_eq_neg_one_l221_22100

-- Definitions for the function f and the condition for it being an even function
def f (x a : ℝ) := (x - 1) * (x - a)

-- The theorem stating that if f is an even function, then a = -1
theorem even_function_a_eq_neg_one (a : ℝ) (h : ∀ x : ℝ, f x a = f (-x) a) : a = -1 :=
by
  sorry

end even_function_a_eq_neg_one_l221_22100


namespace find_range_of_x_l221_22167

def odot (a b : ℝ) : ℝ := a * b + 2 * a + b

theorem find_range_of_x (x : ℝ) : odot x (x - 2) < 0 ↔ -2 < x ∧ x < 1 :=
by sorry

end find_range_of_x_l221_22167


namespace compute_a4_b4_c4_l221_22180

theorem compute_a4_b4_c4 (a b c : ℝ) (h1 : a + b + c = 8) (h2 : ab + ac + bc = 13) (h3 : abc = -22) : a^4 + b^4 + c^4 = 1378 :=
by
  sorry

end compute_a4_b4_c4_l221_22180


namespace calculate_expr1_calculate_expr2_l221_22160

/-- Statement 1: -5 * 3 - 8 / -2 = -11 -/
theorem calculate_expr1 : (-5) * 3 - 8 / -2 = -11 :=
by sorry

/-- Statement 2: (-1)^3 + (5 - (-3)^2) / 6 = -5/3 -/
theorem calculate_expr2 : (-1)^3 + (5 - (-3)^2) / 6 = -(5 / 3) :=
by sorry

end calculate_expr1_calculate_expr2_l221_22160


namespace part1_monotonic_intervals_part2_max_a_l221_22143

noncomputable def f1 (x : ℝ) := Real.log x - 2 * x^2

theorem part1_monotonic_intervals :
  (∀ x, 0 < x ∧ x < 0.5 → f1 x > 0) ∧ (∀ x, x > 0.5 → f1 x < 0) :=
by
  sorry

noncomputable def f2 (x a : ℝ) := Real.log x + a * x^2

theorem part2_max_a (a : ℤ) :
  (∀ x, x > 1 → f2 x a < Real.exp x) → a ≤ 1 :=
by
  sorry

end part1_monotonic_intervals_part2_max_a_l221_22143


namespace eval_fraction_l221_22119

theorem eval_fraction : (3 : ℚ) / (2 - 5 / 4) = 4 := 
by 
  sorry

end eval_fraction_l221_22119


namespace expansion_coeff_sum_l221_22168

theorem expansion_coeff_sum :
  (∃ a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 : ℝ, 
    (2*x - 1)^10 = a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4 + a5*x^5 + a6*x^6 + a7*x^7 + a8*x^8 + a9*x^9 + a10*x^10)
  → (1 - 20 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 = 1 → a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 = 20) :=
by
  sorry

end expansion_coeff_sum_l221_22168


namespace linda_savings_l221_22123

theorem linda_savings (S : ℕ) (h1 : (3 / 4) * S = x) (h2 : (1 / 4) * S = 240) : S = 960 :=
by
  sorry

end linda_savings_l221_22123


namespace no_intersection_points_l221_22199

def intersection_points_eq_zero : Prop :=
∀ x y : ℝ, (y = abs (3 * x + 6)) ∧ (y = -abs (4 * x - 3)) → false

theorem no_intersection_points :
  intersection_points_eq_zero :=
by
  intro x y h
  cases h
  sorry

end no_intersection_points_l221_22199


namespace swimming_lane_length_l221_22155

-- Conditions
def num_round_trips : ℕ := 3
def total_distance : ℕ := 600

-- Hypothesis that 1 round trip is equivalent to 2 lengths of the lane
def lengths_per_round_trip : ℕ := 2

-- Statement to prove
theorem swimming_lane_length :
  (total_distance / (num_round_trips * lengths_per_round_trip) = 100) := by
  sorry

end swimming_lane_length_l221_22155


namespace candy_probability_difference_l221_22186

theorem candy_probability_difference :
  let total := 2004
  let total_ways := Nat.choose total 2
  let different_ways := 2002 * 1002 / 2
  let same_ways := 1002 * 1001 / 2 + 1002 * 1001 / 2
  let q := (different_ways : ℚ) / total_ways
  let p := (same_ways : ℚ) / total_ways
  q - p = 1 / 2003 :=
by sorry

end candy_probability_difference_l221_22186


namespace shadow_boundary_function_correct_l221_22131

noncomputable def sphereShadowFunction : ℝ → ℝ :=
  λ x => (x + 1) / 2

theorem shadow_boundary_function_correct :
  ∀ (x y : ℝ), 
    -- Conditions: 
    -- The sphere with center (0,0,2) and radius 2
    -- A light source at point P = (1, -2, 3)
    -- The shadow must lie on the xy-plane, so z-coordinate is 0
    (sphereShadowFunction x = y) ↔ (- x + 2 * y - 1 = 0) :=
by
  intros x y
  sorry

end shadow_boundary_function_correct_l221_22131


namespace problem_statement_l221_22140

variable {x y : Real}

theorem problem_statement (hx : x * y < 0) (hxy : x > |y|) : x + y > 0 := by
  sorry

end problem_statement_l221_22140


namespace problem_statement_l221_22116

noncomputable def even_increasing (f : ℝ → ℝ) :=
  ∀ x, f x = f (-x) ∧ ∀ x y, x < y → f x < f y

theorem problem_statement {f : ℝ → ℝ} (hf_even_incr : even_increasing f)
  (x1 x2 : ℝ) (hx1_gt_0 : x1 > 0) (hx2_lt_0 : x2 < 0) (hf_lt : f x1 < f x2) : x1 + x2 > 0 :=
sorry

end problem_statement_l221_22116


namespace fifteenth_term_is_44_l221_22134

-- Define the conditions
def first_term : ℕ := 2
def common_difference : ℕ := 3
def term_number : ℕ := 15

-- Define the formula for the nth term of an arithmetic progression
def nth_term (a d n : ℕ) : ℕ := a + (n - 1) * d

-- Prove that the 15th term is 44
theorem fifteenth_term_is_44 : nth_term first_term common_difference term_number = 44 :=
by
  unfold nth_term first_term common_difference term_number
  sorry

end fifteenth_term_is_44_l221_22134


namespace choose_bar_chart_for_comparisons_l221_22120

/-- 
To easily compare the quantities of various items, one should choose a bar chart 
based on the characteristics of statistical charts.
-/
theorem choose_bar_chart_for_comparisons 
  (chart_type: Type) 
  (is_bar_chart: chart_type → Prop)
  (is_ideal_chart_for_comparison: chart_type → Prop)
  (bar_chart_ideal: ∀ c, is_bar_chart c → is_ideal_chart_for_comparison c) 
  (comparison_chart : chart_type) 
  (h: is_bar_chart comparison_chart): 
  is_ideal_chart_for_comparison comparison_chart := 
by
  exact bar_chart_ideal comparison_chart h

end choose_bar_chart_for_comparisons_l221_22120


namespace min_value_of_a2_plus_b2_l221_22159

theorem min_value_of_a2_plus_b2 
  (a b : ℝ) 
  (h : ∃ x : ℝ, x^4 + a * x^3 + b * x^2 + a * x + 1 = 0) : 
  a^2 + b^2 ≥ 4 := 
sorry

end min_value_of_a2_plus_b2_l221_22159


namespace bounces_less_than_50_l221_22191

noncomputable def minBouncesNeeded (initialHeight : ℝ) (bounceFactor : ℝ) (thresholdHeight : ℝ) : ℕ :=
  ⌈(Real.log (thresholdHeight / initialHeight) / Real.log (bounceFactor))⌉₊

theorem bounces_less_than_50 :
  minBouncesNeeded 360 (3/4 : ℝ) 50 = 8 :=
by
  sorry

end bounces_less_than_50_l221_22191


namespace quoted_price_of_shares_l221_22127

theorem quoted_price_of_shares :
  ∀ (investment nominal_value dividend_rate annual_income quoted_price : ℝ),
  investment = 4940 →
  nominal_value = 10 →
  dividend_rate = 14 →
  annual_income = 728 →
  quoted_price = 9.5 :=
by
  intros investment nominal_value dividend_rate annual_income quoted_price
  intros h_investment h_nominal_value h_dividend_rate h_annual_income
  sorry

end quoted_price_of_shares_l221_22127


namespace sum_alternating_series_l221_22111

theorem sum_alternating_series :
  (Finset.sum (Finset.range 2023) (λ k => (-1)^(k + 1))) = -1 := 
by
  sorry

end sum_alternating_series_l221_22111

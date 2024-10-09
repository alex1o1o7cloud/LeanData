import Mathlib

namespace regular_polygon_sides_l1682_168280

theorem regular_polygon_sides (n : ℕ) (h : n ≥ 3) (interior_angle : ℝ) : 
  interior_angle = 144 → n = 10 :=
by
  intro h1
  sorry

end regular_polygon_sides_l1682_168280


namespace complex_div_i_l1682_168237

open Complex

theorem complex_div_i (z : ℂ) (hz : z = -2 - i) : z / i = -1 + 2 * i :=
by
  sorry

end complex_div_i_l1682_168237


namespace max_small_boxes_l1682_168264

-- Define the dimensions of the larger box in meters
def large_box_length : ℝ := 6
def large_box_width : ℝ := 5
def large_box_height : ℝ := 4

-- Define the dimensions of the smaller box in meters
def small_box_length : ℝ := 0.60
def small_box_width : ℝ := 0.50
def small_box_height : ℝ := 0.40

-- Calculate the volume of the larger box
def large_box_volume : ℝ := large_box_length * large_box_width * large_box_height

-- Calculate the volume of the smaller box
def small_box_volume : ℝ := small_box_length * small_box_width * small_box_height

-- State the theorem to prove the maximum number of smaller boxes that can fit in the larger box
theorem max_small_boxes : large_box_volume / small_box_volume = 1000 :=
by
  sorry

end max_small_boxes_l1682_168264


namespace percentage_of_Y_pay_X_is_paid_correct_l1682_168261

noncomputable def percentage_of_Y_pay_X_is_paid
  (total_pay : ℝ) (Y_pay : ℝ) : ℝ :=
  let X_pay := total_pay - Y_pay
  (X_pay / Y_pay) * 100

theorem percentage_of_Y_pay_X_is_paid_correct :
  percentage_of_Y_pay_X_is_paid 700 318.1818181818182 = 120 := 
by
  unfold percentage_of_Y_pay_X_is_paid
  sorry

end percentage_of_Y_pay_X_is_paid_correct_l1682_168261


namespace unique_pair_exists_for_each_n_l1682_168213

theorem unique_pair_exists_for_each_n (n : ℕ) (h : n > 0) : 
  ∃! (a b : ℕ), a > 0 ∧ b > 0 ∧ n = (a + b - 1) * (a + b - 2) / 2 + a :=
sorry

end unique_pair_exists_for_each_n_l1682_168213


namespace maria_age_l1682_168258

variable (M J : Nat)

theorem maria_age (h1 : J = M + 12) (h2 : M + J = 40) : M = 14 := by
  sorry

end maria_age_l1682_168258


namespace polygon_properties_l1682_168224

theorem polygon_properties
  (n : ℕ)
  (h_exterior_angle : 360 / 20 = n)
  (h_n_sides : n = 18) :
  (180 * (n - 2) = 2880) ∧ (n * (n - 3) / 2 = 135) :=
by
  sorry

end polygon_properties_l1682_168224


namespace quadratic_inequality_solution_l1682_168282

theorem quadratic_inequality_solution
  (x : ℝ)
  (h : x^2 - 5 * x + 6 < 0) :
  2 < x ∧ x < 3 :=
by sorry

end quadratic_inequality_solution_l1682_168282


namespace sequence_solution_l1682_168204

theorem sequence_solution (a : ℕ → ℝ) (h1 : a 1 = 2) (h2 : a 2 = 1)
  (h_rec : ∀ n ≥ 2, 2 / a n = 1 / a (n + 1) + 1 / a (n - 1)) :
  ∀ n, a n = 2 / n :=
by
  sorry

end sequence_solution_l1682_168204


namespace sum_of_sequence_l1682_168278

theorem sum_of_sequence (avg : ℕ → ℕ → ℕ) (n : ℕ) (total_sum : ℕ) 
  (condition : avg 16 272 = 17) : 
  total_sum = 272 := 
by 
  sorry

end sum_of_sequence_l1682_168278


namespace rhombus_area_l1682_168210

theorem rhombus_area (side_length : ℝ) (d1_diff_d2 : ℝ) 
  (h_side_length : side_length = Real.sqrt 104) 
  (h_d1_diff_d2 : d1_diff_d2 = 10) : 
  (1 / 2) * (2 * Real.sqrt 104 - d1_diff_d2) * (d1_diff_d2 + 2 * Real.sqrt 104) = 79.17 :=
by
  sorry

end rhombus_area_l1682_168210


namespace arithmetic_sequence_S12_l1682_168232

theorem arithmetic_sequence_S12 (S : ℕ → ℝ) (a : ℕ → ℝ) 
  (h1 : ∀ n : ℕ, S n = n * (a 1 + a n) / 2)
  (hS4 : S 4 = 25) (hS8 : S 8 = 100) : S 12 = 225 :=
by
  sorry

end arithmetic_sequence_S12_l1682_168232


namespace club_planning_committee_l1682_168246

theorem club_planning_committee : Nat.choose 20 3 = 1140 := 
by sorry

end club_planning_committee_l1682_168246


namespace total_plums_picked_l1682_168248

-- Conditions
def Melanie_plums : ℕ := 4
def Dan_plums : ℕ := 9
def Sally_plums : ℕ := 3

-- Proof statement
theorem total_plums_picked : Melanie_plums + Dan_plums + Sally_plums = 16 := by
  sorry

end total_plums_picked_l1682_168248


namespace perfect_squares_as_difference_l1682_168289

theorem perfect_squares_as_difference (N : ℕ) (hN : N = 20000) : 
  (∃ (n : ℕ), n = 71 ∧ 
    ∀ m < N, 
      (∃ a b : ℤ, 
        a^2 = m ∧
        b^2 = m + ((b + 1)^2 - b^2) - 1 ∧ 
        (b + 1)^2 - b^2 = 2 * b + 1)) :=
by 
  sorry

end perfect_squares_as_difference_l1682_168289


namespace arithmetic_geom_seq_a1_over_d_l1682_168291

theorem arithmetic_geom_seq_a1_over_d (a1 a2 a3 a4 d : ℝ) (hne : d ≠ 0)
  (hgeom1 : (a1 + 2*d)^2 = a1 * (a1 + 3*d))
  (hgeom2 : (a1 + d)^2 = a1 * (a1 + 3*d)) :
  (a1 / d = -4) ∨ (a1 / d = 1) :=
by
  sorry

end arithmetic_geom_seq_a1_over_d_l1682_168291


namespace complement_of_M_in_U_l1682_168214

namespace SetComplements

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 3, 5}
def complement_U_M : Set ℕ := U \ M

theorem complement_of_M_in_U :
  complement_U_M = {2, 4, 6} :=
by
  sorry

end SetComplements

end complement_of_M_in_U_l1682_168214


namespace strong_2013_l1682_168255

theorem strong_2013 :
  ∃ x : ℕ, x > 0 ∧ (x ^ (2013 * x) + 1) % (2 ^ 2013) = 0 :=
sorry

end strong_2013_l1682_168255


namespace sum_inequality_l1682_168281

open Real

theorem sum_inequality (a b c : ℝ) (h : a + b + c = 3) :
  (1 / (5 * a^2 - 4 * a + 11) + 1 / (5 * b^2 - 4 * b + 11) + 1 / (5 * c^2 - 4 * c + 11)) ≤ 1 / 4 :=
by
  sorry

end sum_inequality_l1682_168281


namespace chess_piece_max_visitable_squares_l1682_168294

-- Define initial board properties and movement constraints
structure ChessBoard :=
  (rows : ℕ)
  (columns : ℕ)
  (movement : ℕ)
  (board_size : rows * columns = 225)

-- Define condition for unique visitation
def can_visit (movement : ℕ) (board_size : ℕ) : Prop :=
  ∃ (max_squares : ℕ), (max_squares ≤ board_size) ∧ (max_squares = 196)

-- Main theorem statement 
theorem chess_piece_max_visitable_squares (cb : ChessBoard) : 
  can_visit 196 225 :=
by sorry

end chess_piece_max_visitable_squares_l1682_168294


namespace solve_equations_l1682_168207

theorem solve_equations (x : ℝ) (h1 : x^2 - 9 = 0) (h2 : (-x)^3 = (-8)^2) : x = 3 ∨ x = -3 ∨ x = -4 :=
by 
  sorry

end solve_equations_l1682_168207


namespace total_voters_in_districts_l1682_168242

theorem total_voters_in_districts : 
  ∀ (D1 D2 D3 : ℕ),
  (D1 = 322) →
  (D2 = D3 - 19) →
  (D3 = 2 * D1) →
  (D1 + D2 + D3 = 1591) :=
by
  intros D1 D2 D3 h1 h2 h3
  sorry

end total_voters_in_districts_l1682_168242


namespace prob_A_not_losing_prob_A_not_winning_l1682_168245

-- Definitions based on the conditions
def prob_winning : ℝ := 0.41
def prob_tie : ℝ := 0.27

-- The probability of A not losing
def prob_not_losing : ℝ := prob_winning + prob_tie

-- The probability of A not winning
def prob_not_winning : ℝ := 1 - prob_winning

-- Proof problems
theorem prob_A_not_losing : prob_not_losing = 0.68 := by
  sorry

theorem prob_A_not_winning : prob_not_winning = 0.59 := by
  sorry

end prob_A_not_losing_prob_A_not_winning_l1682_168245


namespace ball_hits_ground_time_l1682_168276

theorem ball_hits_ground_time :
  ∃ t : ℝ, -20 * t^2 + 30 * t + 60 = 0 ∧ t = (3 + Real.sqrt 57) / 4 :=
by 
  sorry

end ball_hits_ground_time_l1682_168276


namespace ecuadorian_number_unique_l1682_168287

def is_Ecuadorian (n : ℕ) : Prop :=
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  n < 1000 ∧ c ≠ 0 ∧ n % 36 = 0 ∧ (n - (100 * c + 10 * b + a) > 0) ∧ (n - (100 * c + 10 * b + a)) % 36 = 0

theorem ecuadorian_number_unique (n : ℕ) : 
  is_Ecuadorian n → n = 864 :=
sorry

end ecuadorian_number_unique_l1682_168287


namespace number_of_diagonal_intersections_of_convex_n_gon_l1682_168283

theorem number_of_diagonal_intersections_of_convex_n_gon (n : ℕ) (h : 4 ≤ n) :
  (∀ P : Π m, m = n ↔ m ≥ 4, ∃ i : ℕ, i = n * (n - 1) * (n - 2) * (n - 3) / 24) := 
by
  sorry

end number_of_diagonal_intersections_of_convex_n_gon_l1682_168283


namespace initial_people_in_line_l1682_168296

theorem initial_people_in_line (x : ℕ) (h1 : x + 22 = 83) : x = 61 :=
by sorry

end initial_people_in_line_l1682_168296


namespace steven_sixth_quiz_score_l1682_168272

theorem steven_sixth_quiz_score :
  ∃ x : ℕ, (75 + 80 + 85 + 90 + 100 + x) / 6 = 95 ∧ x = 140 :=
by
  sorry

end steven_sixth_quiz_score_l1682_168272


namespace increasing_on_interval_solution_set_l1682_168273

noncomputable def f (x : ℝ) : ℝ := x / (x ^ 2 + 1)

/- Problem 1 -/
theorem increasing_on_interval : ∀ x1 x2 : ℝ, -1 < x1 → x1 < x2 → x2 < 1 → f x1 < f x2 :=
by
  sorry

/- Problem 2 -/
theorem solution_set : ∀ x : ℝ, f (2 * x - 1) + f x < 0 ↔ 0 < x ∧ x < 1 / 3 :=
by
  sorry

end increasing_on_interval_solution_set_l1682_168273


namespace supermarket_problem_l1682_168201

-- Define that type A costs x yuan and type B costs y yuan
def cost_price_per_item (x y : ℕ) : Prop :=
  (10 * x + 8 * y = 880) ∧ (2 * x + 5 * y = 380)

-- Define purchasing plans with the conditions described
def purchasing_plans (a : ℕ) : Prop :=
  ∀ a : ℕ, 24 ≤ a ∧ a ≤ 26

theorem supermarket_problem : 
  (∃ x y, cost_price_per_item x y ∧ x = 40 ∧ y = 60) ∧ 
  (∃ n, purchasing_plans n ∧ n = 3) :=
by
  sorry

end supermarket_problem_l1682_168201


namespace quadratic_binomial_square_l1682_168298

theorem quadratic_binomial_square (a : ℚ) :
  (∃ r s : ℚ, (ax^2 + 22*x + 9 = (r*x + s)^2) ∧ s = 3 ∧ r = 11 / 3) → a = 121 / 9 := 
by 
  sorry

end quadratic_binomial_square_l1682_168298


namespace semicircle_problem_l1682_168205

open Real

theorem semicircle_problem (r : ℝ) (N : ℕ)
  (h1 : True) -- condition 1: There are N small semicircles each with radius r.
  (h2 : True) -- condition 2: The diameter of the large semicircle is 2Nr.
  (h3 : (N * (π * r^2) / 2) / ((π * (N^2 * r^2) / 2) - (N * (π * r^2) / 2)) = (1 : ℝ) / 12) -- given ratio A / B = 1 / 12 
  : N = 13 :=
sorry

end semicircle_problem_l1682_168205


namespace original_price_of_cupcakes_l1682_168270

theorem original_price_of_cupcakes
  (revenue : ℕ := 32) 
  (cookies_sold : ℕ := 8) 
  (cupcakes_sold : ℕ := 16) 
  (cookie_price: ℕ := 2)
  (half_price_of_cookie: ℕ := 1) :
  (x : ℕ) → (16 * (x / 2)) + (8 * 1) = 32 → x = 3 := 
by
  sorry

end original_price_of_cupcakes_l1682_168270


namespace sum_of_midpoints_l1682_168217

theorem sum_of_midpoints 
  (a b c d e f : ℝ)
  (h1 : a + b + c = 15)
  (h2 : d + e + f = 15) :
  ((a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 15) ∧ 
  ((d + e) / 2 + (d + f) / 2 + (e + f) / 2 = 15) :=
by
  sorry

end sum_of_midpoints_l1682_168217


namespace perpendicular_OP_CD_l1682_168221

variables {Point : Type}

-- Definitions of all the points involved
variables (A B C D P O : Point)
-- Definitions for distances / lengths
variables (dist : Point → Point → ℝ)
-- Definitions for relationships
variables (circumcenter : Point → Point → Point → Point)
variables (perpendicular : Point → Point → Point → Point → Prop)

-- Segment meet condition
variables (meet_at : Point → Point → Point → Prop)

-- Assuming the given conditions
theorem perpendicular_OP_CD 
  (meet : meet_at A C P)
  (meet' : meet_at B D P)
  (h1 : dist P A = dist P D)
  (h2 : dist P B = dist P C)
  (hO : circumcenter P A B = O) :
  perpendicular O P C D :=
sorry

end perpendicular_OP_CD_l1682_168221


namespace car_speed_second_hour_l1682_168216

theorem car_speed_second_hour
  (v1 : ℕ) (avg_speed : ℕ) (time : ℕ) (v2 : ℕ)
  (h1 : v1 = 90)
  (h2 : avg_speed = 70)
  (h3 : time = 2) :
  v2 = 50 :=
by
  sorry

end car_speed_second_hour_l1682_168216


namespace find_number_l1682_168257

theorem find_number (x : ℤ) (n : ℤ) (h1 : x = 88320) (h2 : x + 1315 + n - 1569 = 11901) : n = -75165 :=
by 
  sorry

end find_number_l1682_168257


namespace max_necklaces_with_beads_l1682_168218

noncomputable def necklace_problem : Prop :=
  ∃ (necklaces : ℕ),
    let green_beads := 200
    let white_beads := 100
    let orange_beads := 50
    let beads_per_pattern_green := 3
    let beads_per_pattern_white := 1
    let beads_per_pattern_orange := 1
    necklaces = orange_beads ∧
    green_beads / beads_per_pattern_green >= necklaces ∧
    white_beads / beads_per_pattern_white >= necklaces ∧
    orange_beads / beads_per_pattern_orange >= necklaces

theorem max_necklaces_with_beads : necklace_problem :=
  sorry

end max_necklaces_with_beads_l1682_168218


namespace find_children_and_coins_l1682_168262

def condition_for_child (k m remaining_coins : ℕ) : Prop :=
  ∃ (received_coins : ℕ), (received_coins = k + remaining_coins / 7 ∧ received_coins * 7 = 7 * k + remaining_coins)

def valid_distribution (n m : ℕ) : Prop :=
  ∀ k (hk : 1 ≤ k ∧ k ≤ n),
  ∃ remaining_coins,
    condition_for_child k m remaining_coins

theorem find_children_and_coins :
  ∃ n m, valid_distribution n m ∧ n = 6 ∧ m = 36 :=
sorry

end find_children_and_coins_l1682_168262


namespace handshake_count_l1682_168284

theorem handshake_count : 
  let n := 5  -- number of representatives per company
  let c := 5  -- number of companies
  let total_people := n * c  -- total number of people
  let handshakes_per_person := total_people - n  -- each person shakes hands with 20 others
  (total_people * handshakes_per_person) / 2 = 250 := 
by
  sorry

end handshake_count_l1682_168284


namespace determine_head_start_l1682_168297

def head_start (v : ℝ) (s : ℝ) : Prop :=
  let a_speed := 2 * v
  let distance := 142
  distance / a_speed = (distance - s) / v

theorem determine_head_start (v : ℝ) : head_start v 71 :=
  by
    sorry

end determine_head_start_l1682_168297


namespace interestDifference_l1682_168238

noncomputable def simpleInterest (P R T : ℝ) : ℝ :=
  (P * R * T) / 100

noncomputable def compoundInterest (P R T : ℝ) : ℝ :=
  P * (1 + R / 100)^T - P

theorem interestDifference (P R T : ℝ) (hP : P = 500) (hR : R = 20) (hT : T = 2) :
  compoundInterest P R T - simpleInterest P R T = 120 := by
  sorry

end interestDifference_l1682_168238


namespace money_needed_l1682_168240

def phone_cost : ℕ := 1300
def mike_fraction : ℚ := 0.4

theorem money_needed : mike_fraction * phone_cost + 780 = phone_cost := by
  sorry

end money_needed_l1682_168240


namespace coaches_meet_together_l1682_168234

theorem coaches_meet_together (e s n a : ℕ)
  (h₁ : e = 5) (h₂ : s = 3) (h₃ : n = 9) (h₄ : a = 8) :
  Nat.lcm (Nat.lcm e s) (Nat.lcm n a) = 360 :=
by
  sorry

end coaches_meet_together_l1682_168234


namespace man_is_older_by_24_l1682_168244

-- Define the conditions as per the given problem
def present_age_son : ℕ := 22
def present_age_man (M : ℕ) : Prop := M + 2 = 2 * (present_age_son + 2)

-- State the problem: Prove that the man is 24 years older than his son
theorem man_is_older_by_24 (M : ℕ) (h : present_age_man M) : M - present_age_son = 24 := 
sorry

end man_is_older_by_24_l1682_168244


namespace volume_ratio_l1682_168256

def cube_volume (side_length : ℝ) : ℝ :=
  side_length ^ 3

theorem volume_ratio : 
  let a := (4 : ℝ) / 12   -- 4 inches converted to feet
  let b := (2 : ℝ)       -- 2 feet
  cube_volume a / cube_volume b = 1 / 216 :=
by
  sorry

end volume_ratio_l1682_168256


namespace tangent_line_at_one_m_positive_if_equal_vals_ineq_if_equal_vals_l1682_168266

noncomputable def f (x m : ℝ) : ℝ := (Real.exp (x - 1) - 0.5 * x^2 + x - m * Real.log x)

theorem tangent_line_at_one (m : ℝ) :
  ∃ (y : ℝ → ℝ), (∀ x, y x = (1 - m) * x + m + 0.5) ∧ y 1 = f 1 m ∧ (tangent_slope : ℝ) = 1 - m ∧
    ∀ x, y x = f x m + y 0 :=
sorry

theorem m_positive_if_equal_vals (m x₁ x₂ : ℝ) (h₁ : x₁ ≠ x₂) (h₂ : f x₁ m = f x₂ m) :
  m > 0 :=
sorry

theorem ineq_if_equal_vals (m x₁ x₂ : ℝ) (h₁ : x₁ ≠ x₂) (h₂ : f x₁ m = f x₂ m) :
  2 * m > Real.exp (Real.log x₁ + Real.log x₂) :=
sorry

end tangent_line_at_one_m_positive_if_equal_vals_ineq_if_equal_vals_l1682_168266


namespace expand_and_simplify_l1682_168288

theorem expand_and_simplify (x : ℝ) : (17 * x - 9) * 3 * x = 51 * x^2 - 27 * x := 
by 
  sorry

end expand_and_simplify_l1682_168288


namespace twelve_times_y_plus_three_half_quarter_l1682_168243

theorem twelve_times_y_plus_three_half_quarter (y : ℝ) : 
  (1 / 2) * (1 / 4) * (12 * y + 3) = (3 * y) / 2 + 3 / 8 :=
by sorry

end twelve_times_y_plus_three_half_quarter_l1682_168243


namespace find_X_l1682_168268

theorem find_X (X : ℕ) : 
  (∃ k : ℕ, X = 26 * k + k) ∧ (∃ m : ℕ, X = 29 * m + m) → (X = 270 ∨ X = 540) :=
by
  sorry

end find_X_l1682_168268


namespace evaluate_expression_l1682_168263

theorem evaluate_expression (a b : ℕ) (h1 : a = 3) (h2 : b = 2) : (a^b)^b + (b^a)^a = 593 := by
  sorry

end evaluate_expression_l1682_168263


namespace intersection_proof_complement_proof_range_of_m_condition_l1682_168271

def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 4}
def B : Set ℝ := {x | -3 < x ∧ x < 1}
def C (m : ℝ) : Set ℝ := {x | 2 - m ≤ x ∧ x ≤ 2 + m}

theorem intersection_proof : A ∩ B = {x | -2 ≤ x ∧ x < 1} := sorry

theorem complement_proof : (Set.univ \ B) = {x | x ≤ -3 ∨ x ≥ 1} := sorry

theorem range_of_m_condition (m : ℝ) : (A ∪ C m = A) → (m ≤ 2) := sorry

end intersection_proof_complement_proof_range_of_m_condition_l1682_168271


namespace smallest_positive_integer_l1682_168292
-- Import the required library

-- State the problem in Lean
theorem smallest_positive_integer (x : ℕ) (h : 5 * x ≡ 17 [MOD 31]) : x = 13 :=
sorry

end smallest_positive_integer_l1682_168292


namespace theo_cookies_l1682_168241

theorem theo_cookies (cookies_per_time times_per_day total_cookies total_months : ℕ) (h1 : cookies_per_time = 13) (h2 : times_per_day = 3) (h3 : total_cookies = 2340) (h4 : total_months = 3) : (total_cookies / total_months) / (cookies_per_time * times_per_day) = 20 := 
by
  -- Placeholder for the proof
  sorry

end theo_cookies_l1682_168241


namespace Bill_threw_more_sticks_l1682_168235

-- Definitions based on the given conditions
def Ted_sticks : ℕ := 10
def Ted_rocks : ℕ := 10
def Ted_double_Bill_rocks (R : ℕ) : Prop := Ted_rocks = 2 * R
def Bill_total_objects (S R : ℕ) : Prop := S + R = 21

-- The theorem stating Bill throws 6 more sticks than Ted
theorem Bill_threw_more_sticks (S R : ℕ) (h1 : Ted_double_Bill_rocks R) (h2 : Bill_total_objects S R) : S - Ted_sticks = 6 :=
by
  -- Definitions and conditions are loaded here
  sorry

end Bill_threw_more_sticks_l1682_168235


namespace infinite_values_prime_divisor_l1682_168236

noncomputable def largestPrimeDivisor (n : ℕ) : ℕ :=
  sorry

theorem infinite_values_prime_divisor :
  ∃ᶠ n in at_top, largestPrimeDivisor (n^2 + n + 1) = largestPrimeDivisor ((n+1)^2 + (n+1) + 1) :=
sorry

end infinite_values_prime_divisor_l1682_168236


namespace necessary_condition_for_x_greater_than_2_l1682_168293

-- Define the real number x
variable (x : ℝ)

-- The proof statement
theorem necessary_condition_for_x_greater_than_2 : (x > 2) → (x > 1) :=
by sorry

end necessary_condition_for_x_greater_than_2_l1682_168293


namespace parallel_vectors_x_value_l1682_168269

-- Define the vectors a and b
def a : ℝ × ℝ := (2, 3)
def b (x : ℝ) : ℝ × ℝ := (6, x)

-- Define what it means for vectors to be parallel (they are proportional)
def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

-- Theorem to prove
theorem parallel_vectors_x_value :
  ∀ x : ℝ, parallel a (b x) → x = 9 :=
by
  intros x h
  sorry

end parallel_vectors_x_value_l1682_168269


namespace fraction_exponentiation_example_l1682_168200

theorem fraction_exponentiation_example :
  (5/3)^4 = 625/81 :=
by
  sorry

end fraction_exponentiation_example_l1682_168200


namespace comparison_of_large_exponents_l1682_168211

theorem comparison_of_large_exponents : 2^1997 > 5^850 := sorry

end comparison_of_large_exponents_l1682_168211


namespace mushrooms_gigi_cut_l1682_168226

-- Definitions based on conditions in part a)
def pieces_per_mushroom := 4
def kenny_sprinkled := 38
def karla_sprinkled := 42
def pieces_remaining := 8

-- The total number of pieces is the sum of Kenny's, Karla's, and the remaining pieces.
def total_pieces := kenny_sprinkled + karla_sprinkled + pieces_remaining

-- The number of mushrooms GiGi cut up at the beginning is total_pieces divided by pieces_per_mushroom.
def mushrooms_cut := total_pieces / pieces_per_mushroom

-- The theorem to be proved.
theorem mushrooms_gigi_cut (h1 : pieces_per_mushroom = 4)
                           (h2 : kenny_sprinkled = 38)
                           (h3 : karla_sprinkled = 42)
                           (h4 : pieces_remaining = 8)
                           (h5 : total_pieces = kenny_sprinkled + karla_sprinkled + pieces_remaining)
                           (h6 : mushrooms_cut = total_pieces / pieces_per_mushroom) :
  mushrooms_cut = 22 :=
by
  sorry

end mushrooms_gigi_cut_l1682_168226


namespace other_number_in_product_l1682_168259

theorem other_number_in_product (w : ℕ) (n : ℕ) (hw_pos : 0 < w) (n_factor : Nat.lcm (2^5) (Nat.gcd  864 w) = 2^5 * 3^3) (h_w : w = 144) : n = 6 :=
by
  -- proof would go here
  sorry

end other_number_in_product_l1682_168259


namespace strawberries_jam_profit_l1682_168229

noncomputable def betty_strawberries : ℕ := 25
noncomputable def matthew_strawberries : ℕ := betty_strawberries + 30
noncomputable def natalie_strawberries : ℕ := matthew_strawberries / 3  -- Integer division rounds down
noncomputable def total_strawberries : ℕ := betty_strawberries + matthew_strawberries + natalie_strawberries
noncomputable def strawberries_per_jar : ℕ := 12
noncomputable def jars_of_jam : ℕ := total_strawberries / strawberries_per_jar  -- Integer division rounds down
noncomputable def money_per_jar : ℕ := 6
noncomputable def total_money_made : ℕ := jars_of_jam * money_per_jar

theorem strawberries_jam_profit :
  total_money_made = 48 := by
  sorry

end strawberries_jam_profit_l1682_168229


namespace remainder_3_pow_19_mod_10_l1682_168279

theorem remainder_3_pow_19_mod_10 : (3^19) % 10 = 7 := 
by 
  sorry

end remainder_3_pow_19_mod_10_l1682_168279


namespace simplify_and_evaluate_expression_l1682_168203

noncomputable def given_expression (a : ℝ) : ℝ :=
  (a - 1 - (2 * a - 1) / (a + 1)) / ((a^2 - 4 * a + 4) / (a + 1))

theorem simplify_and_evaluate_expression (a : ℝ) (h : a = 2 + Real.sqrt 3) :
  given_expression a = (2 * Real.sqrt 3 + 3) / 3 :=
by
  sorry

end simplify_and_evaluate_expression_l1682_168203


namespace nearest_multiple_to_457_divisible_by_11_l1682_168220

theorem nearest_multiple_to_457_divisible_by_11 : ∃ n : ℤ, (n % 11 = 0) ∧ (abs (457 - n) = 5) :=
by
  sorry

end nearest_multiple_to_457_divisible_by_11_l1682_168220


namespace probability_of_pink_l1682_168285

theorem probability_of_pink (B P : ℕ) (h1 : (B : ℚ) / (B + P) = 6 / 7) (h2 : (B^2 : ℚ) / (B + P)^2 = 36 / 49) : 
  (P : ℚ) / (B + P) = 1 / 7 :=
by
  sorry

end probability_of_pink_l1682_168285


namespace cost_of_20_pounds_of_bananas_l1682_168231

noncomputable def cost_of_bananas (rate : ℝ) (amount : ℝ) : ℝ :=
rate * amount / 4

theorem cost_of_20_pounds_of_bananas :
  cost_of_bananas 6 20 = 30 :=
by
  sorry

end cost_of_20_pounds_of_bananas_l1682_168231


namespace sandy_hours_per_day_l1682_168223

theorem sandy_hours_per_day (total_hours : ℕ) (days : ℕ) (H : total_hours = 45 ∧ days = 5) : total_hours / days = 9 :=
by
  sorry

end sandy_hours_per_day_l1682_168223


namespace range_of_m_l1682_168265

-- Define the propositions
def p (m : ℝ) : Prop := m ≤ 2
def q (m : ℝ) : Prop := 0 < m ∧ m < 1

-- Problem statement to derive m's range
theorem range_of_m (m : ℝ) (h1: ¬ (p m ∧ q m)) (h2: p m ∨ q m) : m ≤ 0 ∨ (1 ≤ m ∧ m ≤ 2) := 
sorry

end range_of_m_l1682_168265


namespace probability_three_or_more_same_l1682_168299

-- Let us define the total number of outcomes when rolling 5 8-sided dice
def total_outcomes : ℕ := 8 ^ 5

-- Define the number of favorable outcomes where at least three dice show the same number
def favorable_outcomes : ℕ := 4208

-- Define the probability as a fraction
def probability : ℚ := favorable_outcomes / total_outcomes

-- Now we state the theorem that this probability simplifies to 1052/8192
theorem probability_three_or_more_same : probability = 1052 / 8192 :=
sorry

end probability_three_or_more_same_l1682_168299


namespace total_crew_members_l1682_168225

def num_islands : ℕ := 3
def ships_per_island : ℕ := 12
def crew_per_ship : ℕ := 24

theorem total_crew_members : num_islands * ships_per_island * crew_per_ship = 864 := by
  sorry

end total_crew_members_l1682_168225


namespace solve_box_dimensions_l1682_168212

theorem solve_box_dimensions (m n r : ℕ) (h1 : m ≤ n) (h2 : n ≤ r) (h3 : m ≥ 1) (h4 : n ≥ 1) (h5 : r ≥ 1) :
  let k₀ := (m - 2) * (n - 2) * (r - 2)
  let k₁ := 2 * ((m - 2) * (n - 2) + (m - 2) * (r - 2) + (n - 2) * (r - 2))
  let k₂ := 4 * ((m - 2) + (n - 2) + (r - 2))
  (k₀ + k₂ - k₁ = 1985) ↔ ((m = 5 ∧ n = 7 ∧ r = 663) ∨ 
                            (m = 5 ∧ n = 5 ∧ r = 1981) ∨
                            (m = 3 ∧ n = 3 ∧ r = 1981) ∨
                            (m = 1 ∧ n = 7 ∧ r = 399) ∨
                            (m = 1 ∧ n = 3 ∧ r = 1987)) :=
sorry

end solve_box_dimensions_l1682_168212


namespace least_possible_coins_l1682_168274

theorem least_possible_coins : 
  ∃ b : ℕ, b % 7 = 3 ∧ b % 4 = 2 ∧ ∀ n : ℕ, (n % 7 = 3 ∧ n % 4 = 2) → b ≤ n :=
sorry

end least_possible_coins_l1682_168274


namespace roundness_1000000_l1682_168253

-- Definitions based on the conditions in the problem
def prime_factors (n : ℕ) : List (ℕ × ℕ) :=
  if n = 1 then []
  else [(2, 6), (5, 6)] -- Example specifically for 1,000,000

def roundness (n : ℕ) : ℕ :=
  (prime_factors n).map Prod.snd |>.sum

-- The main theorem
theorem roundness_1000000 : roundness 1000000 = 12 := by
  sorry

end roundness_1000000_l1682_168253


namespace complement_of_A_eq_interval_l1682_168275

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x ≥ 1} ∪ {x | x < 0}
def complement_U_A : Set ℝ := {x | 0 ≤ x ∧ x < 1}

theorem complement_of_A_eq_interval : (U \ A) = complement_U_A := by
  sorry

end complement_of_A_eq_interval_l1682_168275


namespace triangle_inequality_l1682_168252

theorem triangle_inequality (a b c : ℝ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) : True :=
  sorry

end triangle_inequality_l1682_168252


namespace ending_number_divisible_by_3_l1682_168215

theorem ending_number_divisible_by_3 : 
∃ n : ℕ, (∀ k : ℕ, (10 + k * 3) ≤ n → (10 + k * 3) % 3 = 0) ∧ 
       (∃ c : ℕ, c = 12 ∧ (n - 10) / 3 + 1 = c) ∧ 
       n = 45 := 
sorry

end ending_number_divisible_by_3_l1682_168215


namespace area_of_enclosed_shape_l1682_168247

noncomputable def enclosed_area : ℝ :=
  ∫ x in (0 : ℝ)..(2 : ℝ), (4 * x - x^3)

theorem area_of_enclosed_shape : enclosed_area = 4 := by
  sorry

end area_of_enclosed_shape_l1682_168247


namespace negation_of_exists_cube_pos_l1682_168250

theorem negation_of_exists_cube_pos :
  (¬ (∃ x : ℝ, x^3 > 0)) ↔ (∀ x : ℝ, x^3 ≤ 0) :=
by
  sorry

end negation_of_exists_cube_pos_l1682_168250


namespace roots_of_polynomial_l1682_168233

theorem roots_of_polynomial :
  {x : ℝ | x^10 - 5*x^8 + 4*x^6 - 64*x^4 + 320*x^2 - 256 = 0} = {x | x = 1 ∨ x = -1 ∨ x = 2 ∨ x = -2} :=
by
  sorry

end roots_of_polynomial_l1682_168233


namespace set_operation_empty_l1682_168254

-- Definition of the universal set I, and sets P and Q with the given properties
variable {I : Set ℕ} -- Universal set
variable {P Q : Set ℕ} -- Non-empty sets with P ⊂ Q ⊂ I
variable (hPQ : P ⊂ Q) (hQI : Q ⊂ I)

-- Prove the set operation expression that results in the empty set
theorem set_operation_empty :
  ∃ (P Q : Set ℕ), P ⊂ Q ∧ Q ⊂ I ∧ P ≠ ∅ ∧ Q ≠ ∅ → 
  P ∩ (I \ Q) = ∅ :=
by
  sorry

end set_operation_empty_l1682_168254


namespace diff_of_squares_l1682_168290

theorem diff_of_squares (a b : ℕ) : 
  (∃ x y : ℤ, a = x^2 - y^2) ∨ (∃ x y : ℤ, b = x^2 - y^2) ∨ (∃ x y : ℤ, a + b = x^2 - y^2) :=
sorry

end diff_of_squares_l1682_168290


namespace power_subtraction_l1682_168286

theorem power_subtraction : 2^4 - 2^3 = 2^3 := by
  sorry

end power_subtraction_l1682_168286


namespace johns_sixth_quiz_score_l1682_168202

theorem johns_sixth_quiz_score
  (score1 score2 score3 score4 score5 : ℕ)
  (h1 : score1 = 85)
  (h2 : score2 = 90)
  (h3 : score3 = 88)
  (h4 : score4 = 92)
  (h5 : score5 = 95)
  : (∃ score6 : ℕ, (score1 + score2 + score3 + score4 + score5 + score6) / 6 = 90) :=
by
  use 90
  sorry

end johns_sixth_quiz_score_l1682_168202


namespace ratio_x_y_l1682_168230

theorem ratio_x_y (x y : ℤ) (h : (8 * x - 5 * y) * 3 = (11 * x - 3 * y) * 2) :
  x / y = 9 / 2 := by
  sorry

end ratio_x_y_l1682_168230


namespace freshman_count_630_l1682_168228

-- Define the variables and conditions
variables (f o j s : ℕ)
variable (total_students : ℕ)

-- Define the ratios given in the problem
def freshmen_to_sophomore : Prop := f = (5 * o) / 4
def sophomore_to_junior : Prop := j = (8 * o) / 7
def junior_to_senior : Prop := s = (7 * j) / 9

-- Total number of students condition
def total_students_condition : Prop := f + o + j + s = total_students

theorem freshman_count_630
  (h1 : freshmen_to_sophomore f o)
  (h2 : sophomore_to_junior o j)
  (h3 : junior_to_senior j s)
  (h4 : total_students_condition f o j s 2158) :
  f = 630 :=
sorry

end freshman_count_630_l1682_168228


namespace quadratic_two_distinct_real_roots_l1682_168227

theorem quadratic_two_distinct_real_roots : 
  ∀ x : ℝ, ∃ a b c : ℝ, (∀ x : ℝ, (x+1)*(x-1) = 2*x + 3 → x^2 - 2*x - 4 = 0) ∧ 
  (a = 1) ∧ (b = -2) ∧ (c = -4) ∧ (b^2 - 4*a*c > 0) :=
by
  sorry

end quadratic_two_distinct_real_roots_l1682_168227


namespace total_birds_in_marsh_l1682_168267

-- Define the number of geese and ducks as constants.
def geese : Nat := 58
def ducks : Nat := 37

-- The theorem that we need to prove.
theorem total_birds_in_marsh : geese + ducks = 95 :=
by
  -- Here, we add the sorry keyword to skip the proof part.
  sorry

end total_birds_in_marsh_l1682_168267


namespace rectangle_area_l1682_168219

theorem rectangle_area :
  ∃ (L B : ℝ), (L - B = 23) ∧ (2 * (L + B) = 206) ∧ (L * B = 2520) :=
sorry

end rectangle_area_l1682_168219


namespace katie_flour_l1682_168239

theorem katie_flour (x : ℕ) (h1 : x + (x + 2) = 8) : x = 3 := 
by
  sorry

end katie_flour_l1682_168239


namespace quadratic_roots_l1682_168208

theorem quadratic_roots (m : ℝ) : 
  (m > 0 → ∃ a b : ℝ, a ≠ b ∧ (a^2 + a - 2 = m) ∧ (b^2 + b - 2 = m)) ∧ 
  ¬(m = 0 ∧ ∃ a : ℝ, (a^2 + a - 2 = m) ∧ (a^2 + a - 2 = m)) ∧ 
  ¬(m < 0 ∧ ¬ ∃ a b : ℝ, (a^2 + a - 2 = m) ∧ (b^2 + b - 2 = m) ) ∧ 
  ¬(∀ m, ∃ a : ℝ, (a^2 + a - 2 = m)) :=
by 
  sorry

end quadratic_roots_l1682_168208


namespace exists_multiple_l1682_168251

theorem exists_multiple (n : ℕ) (a : Fin (n + 1) → ℕ) 
  (h : ∀ i, a i > 0) 
  (h2 : ∀ i, a i ≤ 2 * n) : 
  ∃ i j : Fin (n + 1), i ≠ j ∧ (a i ∣ a j ∨ a j ∣ a i) :=
by
sorry

end exists_multiple_l1682_168251


namespace none_satisfied_l1682_168295

-- Define the conditions
variables {a b c x y z : ℝ}
  
-- Theorem that states that none of the given inequalities are satisfied strictly
theorem none_satisfied (h1 : x ≤ a) (h2 : y ≤ b) (h3 : z ≤ c) :
  ¬(x^2 * y + y^2 * z + z^2 * x < a^2 * b + b^2 * c + c^2 * a) ∧
  ¬(x^3 + y^3 + z^3 < a^3 + b^3 + c^3) :=
  by
    sorry

end none_satisfied_l1682_168295


namespace perpendicular_lines_condition_l1682_168206

variable {A1 B1 C1 A2 B2 C2 : ℝ}

theorem perpendicular_lines_condition :
  (∀ x y : ℝ, A1 * x + B1 * y + C1 = 0) ∧ (∀ x y : ℝ, A2 * x + B2 * y + C2 = 0) → 
  (A1 * A2) / (B1 * B2) = -1 := 
sorry

end perpendicular_lines_condition_l1682_168206


namespace goose_eggs_hatching_l1682_168277

theorem goose_eggs_hatching (x : ℝ) :
  (∃ n_hatched : ℝ, 3 * (2 * n_hatched / 20) = 110 ∧ x = n_hatched / 550) →
  x = 2 / 3 :=
by
  intro h
  sorry

end goose_eggs_hatching_l1682_168277


namespace sector_area_l1682_168209

theorem sector_area (α r : ℝ) (hα : α = π / 3) (hr : r = 2) : 
  1 / 2 * α * r^2 = 2 * π / 3 := 
by 
  rw [hα, hr] 
  simp 
  sorry

end sector_area_l1682_168209


namespace cylinder_radius_and_remaining_space_l1682_168249

theorem cylinder_radius_and_remaining_space 
  (cone_radius : ℝ) (cone_height : ℝ) 
  (cylinder_radius : ℝ) (cylinder_height : ℝ) :
  cone_radius = 8 →
  cone_height = 20 →
  cylinder_height = 2 * cylinder_radius →
  (20 - 2 * cylinder_radius) / cylinder_radius = 20 / 8 →
  (cylinder_radius = 40 / 9 ∧ (cone_height - cylinder_height) = 100 / 9) :=
by
  intros cone_radius_8 cone_height_20 cylinder_height_def similarity_eq
  sorry

end cylinder_radius_and_remaining_space_l1682_168249


namespace probability_of_winning_five_tickets_l1682_168260

def probability_of_winning_one_ticket := 1 / 10000000
def number_of_tickets_bought := 5

theorem probability_of_winning_five_tickets : 
  (number_of_tickets_bought * probability_of_winning_one_ticket) = 5 / 10000000 :=
by
  sorry

end probability_of_winning_five_tickets_l1682_168260


namespace evaluation_l1682_168222
-- Import the entire Mathlib library

-- Define the operations triangle and nabla
def triangle (a b : ℕ) : ℕ := 3 * a + 2 * b
def nabla (a b : ℕ) : ℕ := 2 * a + 3 * b

-- The proof statement
theorem evaluation : triangle 2 (nabla 3 4) = 42 :=
by
  -- Provide a placeholder for the proof
  sorry

end evaluation_l1682_168222

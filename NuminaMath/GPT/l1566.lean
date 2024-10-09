import Mathlib

namespace work_completion_time_l1566_156685

theorem work_completion_time (A B C : ℝ) (hA : A = 1 / 4) (hB : B = 1 / 5) (hC : C = 1 / 20) :
  1 / (A + B + C) = 2 :=
by
  -- Proof goes here
  sorry

end work_completion_time_l1566_156685


namespace pizza_diameter_increase_l1566_156622

theorem pizza_diameter_increase :
  ∀ (d D : ℝ), 
    (D / d)^2 = 1.96 → D = 1.4 * d := by
  sorry

end pizza_diameter_increase_l1566_156622


namespace pictures_per_album_l1566_156693

-- Define the problem conditions
def picturesFromPhone : Nat := 35
def picturesFromCamera : Nat := 5
def totalAlbums : Nat := 5

-- Define the total number of pictures
def totalPictures : Nat := picturesFromPhone + picturesFromCamera

-- Define what we need to prove
theorem pictures_per_album :
  totalPictures / totalAlbums = 8 := by
  sorry

end pictures_per_album_l1566_156693


namespace quotient_unchanged_l1566_156601

-- Define the variables
variables (a b k : ℝ)

-- Condition: k ≠ 0
theorem quotient_unchanged (h : k ≠ 0) : (a * k) / (b * k) = a / b := by
  sorry

end quotient_unchanged_l1566_156601


namespace alfonso_initial_money_l1566_156672

def daily_earnings : ℕ := 6
def days_per_week : ℕ := 5
def total_weeks : ℕ := 10
def cost_of_helmet : ℕ := 340

theorem alfonso_initial_money :
  let weekly_earnings := daily_earnings * days_per_week
  let total_earnings := weekly_earnings * total_weeks
  cost_of_helmet - total_earnings = 40 :=
by
  let weekly_earnings := daily_earnings * days_per_week
  let total_earnings := weekly_earnings * total_weeks
  show cost_of_helmet - total_earnings = 40
  sorry

end alfonso_initial_money_l1566_156672


namespace common_ratio_geometric_sequence_l1566_156638

theorem common_ratio_geometric_sequence 
  (a1 : ℝ) 
  (q : ℝ) 
  (S : ℕ → ℝ) 
  (h1 : ∀ n, S n = a1 * (1 - q^n) / (1 - q)) 
  (h2 : 8 * S 6 = 7 * S 3) 
  (hq : q ≠ 1) : 
  q = -1 / 2 := 
sorry

end common_ratio_geometric_sequence_l1566_156638


namespace max_marks_l1566_156696

theorem max_marks (M : ℝ) (h1 : 0.45 * M = 225) : M = 500 :=
by {
sorry
}

end max_marks_l1566_156696


namespace stocks_higher_price_l1566_156678

theorem stocks_higher_price (total_stocks lower_price higher_price: ℝ)
  (h_total: total_stocks = 8000)
  (h_ratio: higher_price = 1.5 * lower_price)
  (h_sum: lower_price + higher_price = total_stocks) :
  higher_price = 4800 :=
by
  sorry

end stocks_higher_price_l1566_156678


namespace determine_y_value_l1566_156636

theorem determine_y_value {k y : ℕ} (h1 : k > 0) (h2 : y > 0) (hk : k < 10) (hy : y < 10) :
  (8 * 100 + k * 10 + 8) + (k * 100 + 8 * 10 + 8) - (1 * 100 + 6 * 10 + y * 1) = 8 * 100 + k * 10 + 8 → 
  y = 9 :=
by
  sorry

end determine_y_value_l1566_156636


namespace sqrt_expr_eq_two_l1566_156654

noncomputable def expr := Real.sqrt (3 + 2 * Real.sqrt 2) - Real.sqrt (3 - 2 * Real.sqrt 2)

theorem sqrt_expr_eq_two : expr = 2 := 
by
  sorry

end sqrt_expr_eq_two_l1566_156654


namespace largest_of_five_consecutive_ints_15120_l1566_156623

theorem largest_of_five_consecutive_ints_15120 :
  ∃ (a b c d e : ℕ), 
  a < b ∧ b < c ∧ c < d ∧ d < e ∧ 
  a * b * c * d * e = 15120 ∧ 
  e = 10 := 
sorry

end largest_of_five_consecutive_ints_15120_l1566_156623


namespace arithmetic_sequence_sum_9_is_36_l1566_156680

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop := ∃ r, ∀ n, a (n + 1) = r * (a n)
noncomputable def Sn (b : ℕ → ℝ) (n : ℕ) : ℝ := n * (b 1 + b n) / 2

theorem arithmetic_sequence_sum_9_is_36 (a b : ℕ → ℝ) (h_geom : geometric_sequence a) 
    (h_cond : a 4 * a 6 = 2 * a 5) (h_b5 : b 5 = 2 * a 5) : Sn b 9 = 36 :=
by
  sorry

end arithmetic_sequence_sum_9_is_36_l1566_156680


namespace percentage_of_360_is_165_6_l1566_156689

theorem percentage_of_360_is_165_6 :
  (165.6 / 360) * 100 = 46 :=
by
  sorry

end percentage_of_360_is_165_6_l1566_156689


namespace amount_coach_mike_gave_l1566_156607

-- Definitions from conditions
def cost_of_lemonade : ℕ := 58
def change_received : ℕ := 17

-- Theorem stating the proof problem
theorem amount_coach_mike_gave : cost_of_lemonade + change_received = 75 := by
  sorry

end amount_coach_mike_gave_l1566_156607


namespace rearrange_circles_sums13_l1566_156659

def isSum13 (a b c d x y z w : ℕ) : Prop :=
  (a + 4 + b = 13) ∧ (b + 2 + d = 13) ∧ (d + 1 + c = 13) ∧ (c + 3 + a = 13)

theorem rearrange_circles_sums13 : 
  ∃ (a b c d x y z w : ℕ), 
  a = 4 ∧ b = 5 ∧ c = 6 ∧ d = 6 ∧ 
  a + b = 9 ∧ b + z = 11 ∧ z + c = 12 ∧ c + a = 10 ∧ 
  isSum13 a b c d x y z w :=
by {
  sorry
}

end rearrange_circles_sums13_l1566_156659


namespace incorrect_statement_D_l1566_156629

theorem incorrect_statement_D : ∃ a : ℝ, a > 0 ∧ (1 - 1 / (2 * a) < 0) := by
  sorry

end incorrect_statement_D_l1566_156629


namespace sum_of_cubes_l1566_156657

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 :=
by
  sorry

end sum_of_cubes_l1566_156657


namespace red_red_pairs_l1566_156630

theorem red_red_pairs (green_shirts red_shirts total_students total_pairs green_green_pairs : ℕ)
    (hg1 : green_shirts = 64)
    (hr1 : red_shirts = 68)
    (htotal : total_students = 132)
    (htotal_pairs : total_pairs = 66)
    (hgreen_green_pairs : green_green_pairs = 28) :
    (total_students = green_shirts + red_shirts) ∧
    (green_green_pairs ≤ total_pairs) ∧
    (∃ red_red_pairs, red_red_pairs = 30) :=
by
  sorry

end red_red_pairs_l1566_156630


namespace xander_pages_left_to_read_l1566_156631

theorem xander_pages_left_to_read :
  let total_pages := 500
  let read_first_night := 0.2 * 500
  let read_second_night := 0.2 * 500
  let read_third_night := 0.3 * 500
  total_pages - (read_first_night + read_second_night + read_third_night) = 150 :=
by 
  sorry

end xander_pages_left_to_read_l1566_156631


namespace expression_equivalence_l1566_156618

theorem expression_equivalence (a b : ℝ) : 2 * a * b - a^2 - b^2 = -((a - b)^2) :=
by {
  sorry
}

end expression_equivalence_l1566_156618


namespace simplify_expression_l1566_156697

variables {a b : ℝ}

-- Define the conditions
def condition (a b : ℝ) : Prop := (a > 0) ∧ (b > 0) ∧ (a^4 + b^4 = a + b)

-- Define the target goal
def goal (a b : ℝ) : Prop := 
  (a / b + b / a - 1 / (a * b^2)) = (-a - b) / (a * b^2)

-- Statement of the theorem
theorem simplify_expression (h : condition a b) : goal a b :=
by 
  sorry

end simplify_expression_l1566_156697


namespace find_number_of_children_l1566_156658

theorem find_number_of_children (N : ℕ) (B : ℕ) 
    (h1 : B = 2 * N) 
    (h2 : B = 4 * (N - 160)) 
    : N = 320 := 
by
  sorry

end find_number_of_children_l1566_156658


namespace prime_p_sum_of_squares_l1566_156612

theorem prime_p_sum_of_squares (p : ℕ) (hp : p.Prime) 
  (h : ∃ (a : ℕ), 2 * p = a^2 + (a + 1)^2 + (a + 2)^2 + (a + 3)^2) : 
  36 ∣ (p - 7) :=
by 
  sorry

end prime_p_sum_of_squares_l1566_156612


namespace probability_of_same_length_l1566_156652

-- Define the set T and its properties
def T : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
def sides : ℕ := 6
def diagonals : ℕ := 9
def total_elements : ℕ := 15
def probability_same_length : ℚ := 17 / 35

-- Lean 4 statement (theorem)
theorem probability_of_same_length: 
    ( ∃ (sides diagonals : ℕ), sides = 6 ∧ diagonals = 9 ∧ sides + diagonals = 15 →
                                              ∃ probability_same_length : ℚ, probability_same_length = 17 / 35) := 
sorry

end probability_of_same_length_l1566_156652


namespace neither_sufficient_nor_necessary_l1566_156619

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

def is_increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) > a n

theorem neither_sufficient_nor_necessary (a : ℕ → ℝ) (q : ℝ) :
  is_geometric_sequence a q →
  ¬ ((q > 1) ↔ is_increasing_sequence a) :=
sorry

end neither_sufficient_nor_necessary_l1566_156619


namespace pascal_triangle_count_30_rows_l1566_156635

def pascal_row_count (n : Nat) := n + 1

def sum_arithmetic_sequence (a₁ an n : Nat) : Nat :=
  n * (a₁ + an) / 2

theorem pascal_triangle_count_30_rows :
  sum_arithmetic_sequence (pascal_row_count 0) (pascal_row_count 29) 30 = 465 :=
by
  sorry

end pascal_triangle_count_30_rows_l1566_156635


namespace cooper_needs_1043_bricks_l1566_156608

def wall1_length := 15
def wall1_height := 6
def wall1_depth := 3

def wall2_length := 20
def wall2_height := 4
def wall2_depth := 2

def wall3_length := 25
def wall3_height := 5
def wall3_depth := 3

def wall4_length := 17
def wall4_height := 7
def wall4_depth := 2

def bricks_needed_for_wall (length height depth: Nat) : Nat :=
  length * height * depth

def total_bricks_needed : Nat :=
  bricks_needed_for_wall wall1_length wall1_height wall1_depth +
  bricks_needed_for_wall wall2_length wall2_height wall2_depth +
  bricks_needed_for_wall wall3_length wall3_height wall3_depth +
  bricks_needed_for_wall wall4_length wall4_height wall4_depth

theorem cooper_needs_1043_bricks : total_bricks_needed = 1043 := by
  sorry

end cooper_needs_1043_bricks_l1566_156608


namespace compute_fraction_power_l1566_156655

theorem compute_fraction_power (a b : ℕ) (ha : a = 123456) (hb : b = 41152) : (a ^ 5 / b ^ 5) = 243 := by
  sorry

end compute_fraction_power_l1566_156655


namespace min_voters_tall_giraffe_win_l1566_156644

-- Definitions from the problem statement as conditions
def precinct_voters := 3
def precincts_per_district := 9
def districts := 5
def majority_precincts(p : ℕ) := p / 2 + 1  -- Minimum precincts won in a district 
def majority_districts(d : ℕ) := d / 2 + 1  -- Minimum districts won in the final

-- Condition: majority precincts to win a district
def precinct_votes_to_win := majority_precincts precinct_voters

-- Condition: majority districts to win the final
def district_wins_to_win_final := majority_districts districts

-- Minimum precincts the Tall giraffe needs to win overall
def total_precincts_to_win := district_wins_to_win_final * majority_precincts precincts_per_district

-- Proof that the minimum number of voters who could have voted for the Tall giraffe is 30
theorem min_voters_tall_giraffe_win :
  precinct_votes_to_win * total_precincts_to_win = 30 :=
sorry

end min_voters_tall_giraffe_win_l1566_156644


namespace bus_ticket_probability_l1566_156621

theorem bus_ticket_probability :
  let total_tickets := 10 ^ 6
  let choices := Nat.choose 10 6 * 2
  (choices : ℝ) / total_tickets = 0.00042 :=
by
  sorry

end bus_ticket_probability_l1566_156621


namespace value_two_std_dev_less_than_mean_l1566_156606

-- Define the given conditions for the problem.
def mean : ℝ := 15
def std_dev : ℝ := 1.5

-- Define the target value that should be 2 standard deviations less than the mean.
def target_value := mean - 2 * std_dev

-- State the theorem that represents the proof problem.
theorem value_two_std_dev_less_than_mean : target_value = 12 := by
  sorry

end value_two_std_dev_less_than_mean_l1566_156606


namespace percentage_chromium_first_alloy_l1566_156695

theorem percentage_chromium_first_alloy
  (x : ℝ) (h : (x / 100) * 15 + (8 / 100) * 35 = (9.2 / 100) * 50) : x = 12 :=
sorry

end percentage_chromium_first_alloy_l1566_156695


namespace brad_reads_more_pages_l1566_156624

-- Definitions based on conditions
def greg_pages_per_day : ℕ := 18
def brad_pages_per_day : ℕ := 26

-- Statement to prove
theorem brad_reads_more_pages : brad_pages_per_day - greg_pages_per_day = 8 :=
by
  -- sorry is used here to indicate the absence of a proof
  sorry

end brad_reads_more_pages_l1566_156624


namespace isosceles_triangle_of_sine_ratio_obtuse_triangle_of_tan_sum_neg_l1566_156691

open Real

theorem isosceles_triangle_of_sine_ratio (a b c : ℝ) (A B C : ℝ)
  (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) (h_sum : A + B + C = π)
  (h1 : a = b * sin C + c * cos B) :
  C = π / 4 :=
sorry

theorem obtuse_triangle_of_tan_sum_neg (A B C : ℝ) 
  (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) (h_sum : A + B + C = π)
  (h_tan_sum : tan A + tan B + tan C < 0) :
  ∃ (E : ℝ), (A = E ∨ B = E ∨ C = E) ∧ π / 2 < E :=
sorry

end isosceles_triangle_of_sine_ratio_obtuse_triangle_of_tan_sum_neg_l1566_156691


namespace cost_of_country_cd_l1566_156679

theorem cost_of_country_cd
  (cost_rock_cd : ℕ) (cost_pop_cd : ℕ) (cost_dance_cd : ℕ)
  (num_each : ℕ) (julia_has : ℕ) (julia_short : ℕ)
  (total_cost : ℕ) (total_other_cds : ℕ) (cost_country_cd : ℕ) :
  cost_rock_cd = 5 →
  cost_pop_cd = 10 →
  cost_dance_cd = 3 →
  num_each = 4 →
  julia_has = 75 →
  julia_short = 25 →
  total_cost = julia_has + julia_short →
  total_other_cds = num_each * cost_rock_cd + num_each * cost_pop_cd + num_each * cost_dance_cd →
  total_cost = total_other_cds + num_each * cost_country_cd →
  cost_country_cd = 7 :=
by
  intros cost_rock_cost_pop_cost_dance_num julia_diff 
         calc_total_total_other sub_total total_cds
  sorry

end cost_of_country_cd_l1566_156679


namespace cindy_pens_ratio_is_one_l1566_156649

noncomputable def pens_owned_initial : ℕ := 25
noncomputable def pens_given_by_mike : ℕ := 22
noncomputable def pens_given_to_sharon : ℕ := 19
noncomputable def pens_owned_final : ℕ := 75

def pens_before_cindy (initial_pens mike_pens : ℕ) : ℕ := initial_pens + mike_pens
def pens_before_sharon (final_pens sharon_pens : ℕ) : ℕ := final_pens + sharon_pens
def pens_given_by_cindy (pens_before_sharon pens_before_cindy : ℕ) : ℕ := pens_before_sharon - pens_before_cindy
def ratio_pens_given_cindy (cindy_pens pens_before_cindy : ℕ) : ℚ := cindy_pens / pens_before_cindy

theorem cindy_pens_ratio_is_one :
    ratio_pens_given_cindy
        (pens_given_by_cindy (pens_before_sharon pens_owned_final pens_given_to_sharon)
                             (pens_before_cindy pens_owned_initial pens_given_by_mike))
        (pens_before_cindy pens_owned_initial pens_given_by_mike) = 1 := by
    sorry

end cindy_pens_ratio_is_one_l1566_156649


namespace average_mileage_highway_l1566_156600

theorem average_mileage_highway (H : Real) : 
  (∀ d : Real, (d / 7.6) > 23 → false) → 
  (280.6 / 23 = H) → 
  H = 12.2 := by
  sorry

end average_mileage_highway_l1566_156600


namespace lending_period_C_l1566_156643

theorem lending_period_C (R : ℝ) (P_B P_C T_B I : ℝ) (h1 : R = 13.75) (h2 : P_B = 4000) (h3 : P_C = 2000) (h4 : T_B = 2) (h5 : I = 2200) : 
  ∃ T_C : ℝ, T_C = 4 :=
by
  -- Definitions and known facts
  let I_B := (P_B * R * T_B) / 100
  let I_C := I - I_B
  let T_C := I_C / ((P_C * R) / 100)
  -- Prove the target
  use T_C
  sorry

end lending_period_C_l1566_156643


namespace greater_combined_area_l1566_156641

noncomputable def area_of_rectangle (length : ℝ) (width : ℝ) : ℝ :=
  length * width

noncomputable def combined_area (length : ℝ) (width : ℝ) : ℝ :=
  2 * (area_of_rectangle length width)

theorem greater_combined_area 
  (length1 width1 length2 width2 : ℝ)
  (h1 : length1 = 11) (h2 : width1 = 13)
  (h3 : length2 = 6.5) (h4 : width2 = 11) :
  combined_area length1 width1 - combined_area length2 width2 = 143 :=
by
  rw [h1, h2, h3, h4]
  sorry

end greater_combined_area_l1566_156641


namespace smallest_constant_inequality_l1566_156616

theorem smallest_constant_inequality :
  ∀ (x y : ℝ), 1 + (x + y)^2 ≤ (4 / 3) * (1 + x^2) * (1 + y^2) :=
by
  intro x y
  sorry

end smallest_constant_inequality_l1566_156616


namespace arithmetic_sequence_minimum_value_S_l1566_156664

noncomputable def S (n : ℕ) : ℤ := sorry -- The sum of the first n terms of the sequence a_n

def a (n : ℕ) : ℤ := sorry -- Defines a_n

axiom condition1 (n : ℕ) : (2 * S n / n + n = 2 * a n + 1)

theorem arithmetic_sequence (n : ℕ) : ∃ d : ℤ, ∀ k : ℕ, a (k + 1) = a k + d := sorry

axiom geometric_sequence : a 7 ^ 2 = a 4 * a 9

theorem minimum_value_S : ∀ n : ℕ, (a 4 < a 7 ∧ a 7 < a 9) → S n ≥ -78 := sorry

end arithmetic_sequence_minimum_value_S_l1566_156664


namespace min_mn_value_l1566_156609

theorem min_mn_value
  (a : ℝ) (m : ℝ) (n : ℝ)
  (ha_pos : a > 0) (ha_ne_one : a ≠ 1) (hm_pos : m > 0) (hn_pos : n > 0)
  (H : (1 : ℝ) / m + (1 : ℝ) / n = 4) :
  m + n ≥ 1 :=
sorry

end min_mn_value_l1566_156609


namespace tangent_line_to_parabola_l1566_156673

theorem tangent_line_to_parabola (r : ℝ) :
  (∃ x : ℝ, 2 * x^2 - x - r = 0) ∧
  (∀ x1 x2 : ℝ, (2 * x1^2 - x1 - r = 0) ∧ (2 * x2^2 - x2 - r = 0) → x1 = x2) →
  r = -1 / 8 :=
sorry

end tangent_line_to_parabola_l1566_156673


namespace value_of_x2_plus_9y2_l1566_156653

theorem value_of_x2_plus_9y2 (x y : ℝ) 
  (h1 : x + 3 * y = 6)
  (h2 : x * y = -9) :
  x^2 + 9 * y^2 = 90 := 
by {
  sorry
}

end value_of_x2_plus_9y2_l1566_156653


namespace simple_interest_rate_l1566_156688

theorem simple_interest_rate (P : ℝ) (R : ℝ) (SI : ℝ) (T : ℝ) (h1 : T = 4) (h2 : SI = P / 5) (h3 : SI = (P * R * T) / 100) : R = 5 := by
  sorry

end simple_interest_rate_l1566_156688


namespace points_on_fourth_board_l1566_156640

-- Definition of the points scored on each dartboard
def points_board_1 : ℕ := 30
def points_board_2 : ℕ := 38
def points_board_3 : ℕ := 41

-- Statement to prove that points on the fourth board are 34
theorem points_on_fourth_board : (points_board_1 + points_board_2) / 2 = 34 :=
by
  -- Given points on first and second boards
  have h1 : points_board_1 + points_board_2 = 68 := by rfl
  sorry

end points_on_fourth_board_l1566_156640


namespace total_cakes_served_l1566_156663

def Cakes_Monday_Lunch : ℕ := 5
def Cakes_Monday_Dinner : ℕ := 6
def Cakes_Sunday : ℕ := 3
def cakes_served_twice (n : ℕ) : ℕ := 2 * n
def cakes_thrown_away : ℕ := 4

theorem total_cakes_served : 
  Cakes_Sunday + Cakes_Monday_Lunch + Cakes_Monday_Dinner + 
  (cakes_served_twice (Cakes_Monday_Lunch + Cakes_Monday_Dinner) - cakes_thrown_away) = 32 := 
by 
  sorry

end total_cakes_served_l1566_156663


namespace sum_of_values_k_l1566_156665

theorem sum_of_values_k (k : ℕ) : 
  (∀ x y : ℤ, (3 * x * x - k * x + 12 = 0) ∧ (3 * y * y - k * y + 12 = 0) ∧ x ≠ y) → k = 0 :=
by
  sorry

end sum_of_values_k_l1566_156665


namespace sum_of_reciprocal_roots_l1566_156611

theorem sum_of_reciprocal_roots (r s α β : ℝ) (h1 : 7 * r^2 - 8 * r + 6 = 0) (h2 : 7 * s^2 - 8 * s + 6 = 0) (h3 : α = 1 / r) (h4 : β = 1 / s) :
  α + β = 4 / 3 := 
sorry

end sum_of_reciprocal_roots_l1566_156611


namespace zero_in_M_l1566_156602

-- Define the set M
def M : Set ℕ := {0, 1, 2}

-- State the theorem to be proved
theorem zero_in_M : 0 ∈ M := 
  sorry

end zero_in_M_l1566_156602


namespace line_to_slope_intercept_l1566_156668

noncomputable def line_equation (v p q : ℝ × ℝ) : Prop :=
  (v.1 * (p.1 - q.1) + v.2 * (p.2 - q.2)) = 0

theorem line_to_slope_intercept (x y m b : ℝ) :
  line_equation (3, -4) (x, y) (2, 8) → (m, b) = (3 / 4, 6.5) :=
  by
    sorry

end line_to_slope_intercept_l1566_156668


namespace range_of_m_l1566_156666

theorem range_of_m (m : ℝ) :
  (∃ x y, y = x^2 + m * x + 2 ∧ x - y + 1 = 0 ∧ 0 ≤ x ∧ x ≤ 2) → m ≤ -1 :=
by
  sorry

end range_of_m_l1566_156666


namespace soda_cost_is_2_l1566_156615

noncomputable def cost_per_soda (total_bill : ℕ) (num_adults : ℕ) (num_children : ℕ) 
  (adult_meal_cost : ℕ) (child_meal_cost : ℕ) (num_sodas : ℕ) : ℕ :=
  (total_bill - (num_adults * adult_meal_cost + num_children * child_meal_cost)) / num_sodas

theorem soda_cost_is_2 :
  let total_bill := 60
  let num_adults := 6
  let num_children := 2
  let adult_meal_cost := 6
  let child_meal_cost := 4
  let num_sodas := num_adults + num_children
  cost_per_soda total_bill num_adults num_children adult_meal_cost child_meal_cost num_sodas = 2 :=
by
  -- proof goes here
  sorry

end soda_cost_is_2_l1566_156615


namespace milton_sold_total_pies_l1566_156683

-- Definitions for the given conditions.
def apple_pie_slices : ℕ := 8
def peach_pie_slices : ℕ := 6
def cherry_pie_slices : ℕ := 10

def apple_slices_ordered : ℕ := 88
def peach_slices_ordered : ℕ := 78
def cherry_slices_ordered : ℕ := 45

-- Function to compute the number of pies, rounding up as necessary
noncomputable def pies_sold (ordered : ℕ) (slices : ℕ) : ℕ :=
  (ordered + slices - 1) / slices  -- Using integer division to round up

-- The theorem asserting the total number of pies sold 
theorem milton_sold_total_pies : 
  pies_sold apple_slices_ordered apple_pie_slices +
  pies_sold peach_slices_ordered peach_pie_slices +
  pies_sold cherry_slices_ordered cherry_pie_slices = 29 :=
by sorry

end milton_sold_total_pies_l1566_156683


namespace find_ratio_l1566_156662

open Real

theorem find_ratio (x y : ℝ) (h1 : y > x) (h2 : x > 0) (h3 : (x / y) + (y / x) = 8) :
  (x + 2 * y) / (x - 2 * y) = -4 / sqrt 7 :=
by
  sorry

end find_ratio_l1566_156662


namespace average_age_of_two_women_is_30_l1566_156639

-- Given definitions
def avg_age_before_replacement (A : ℝ) := 8 * A
def avg_age_after_increase (A : ℝ) := 8 * (A + 2)
def ages_of_men_replaced := 20 + 24

-- The theorem to prove: the average age of the two women is 30 years
theorem average_age_of_two_women_is_30 (A : ℝ) :
  (avg_age_after_increase A) - (avg_age_before_replacement A) = 16 →
  (ages_of_men_replaced + 16) / 2 = 30 :=
by
  sorry

end average_age_of_two_women_is_30_l1566_156639


namespace probability_MAME_top_l1566_156660

-- Conditions
def paper_parts : ℕ := 8
def desired_top : ℕ := 1

-- Question and Proof Problem (Probability calculation)
theorem probability_MAME_top : (1 : ℚ) / paper_parts = 1 / 8 :=
by
  sorry

end probability_MAME_top_l1566_156660


namespace complex_division_l1566_156632

theorem complex_division (i : ℂ) (hi : i = Complex.I) : (1 + i) / (1 - i) = i :=
by
  sorry

end complex_division_l1566_156632


namespace Mary_work_days_l1566_156671

theorem Mary_work_days :
  ∀ (M : ℝ), (∀ R : ℝ, R = M / 1.30) → (R = 20) → M = 26 :=
by
  intros M h1 h2
  sorry

end Mary_work_days_l1566_156671


namespace range_3x_plus_2y_l1566_156605

theorem range_3x_plus_2y (x y : ℝ) : -1 < x + y ∧ x + y < 4 → 2 < x - y ∧ x - y < 3 → 
  -3/2 < 3*x + 2*y ∧ 3*x + 2*y < 23/2 :=
by
  sorry

end range_3x_plus_2y_l1566_156605


namespace average_first_21_multiples_of_4_l1566_156669

-- Define conditions
def n : ℕ := 21
def a1 : ℕ := 4
def an : ℕ := 4 * n
def sum_series (n a1 an : ℕ) : ℕ := (n * (a1 + an)) / 2

-- The problem statement in Lean 4
theorem average_first_21_multiples_of_4 : 
    (sum_series n a1 an) / n = 44 :=
by
  -- skipping the proof
  sorry

end average_first_21_multiples_of_4_l1566_156669


namespace prove_correct_y_l1566_156628

noncomputable def find_larger_y (x y : ℕ) : Prop :=
  y - x = 1365 ∧ y = 6 * x + 15

noncomputable def correct_y : ℕ := 1635

theorem prove_correct_y (x y : ℕ) (h : find_larger_y x y) : y = correct_y :=
by
  sorry

end prove_correct_y_l1566_156628


namespace gcd_987654_876543_eq_3_l1566_156687

theorem gcd_987654_876543_eq_3 :
  Nat.gcd 987654 876543 = 3 :=
sorry

end gcd_987654_876543_eq_3_l1566_156687


namespace trackball_mice_count_l1566_156682

theorem trackball_mice_count (total_sales wireless_share optical_share : ℕ) 
    (h_total : total_sales = 80)
    (h_wireless : wireless_share = total_sales / 2)
    (h_optical : optical_share = total_sales / 4):
    total_sales - (wireless_share + optical_share) = 20 :=
by
  sorry

end trackball_mice_count_l1566_156682


namespace scientific_notation_correct_l1566_156681

theorem scientific_notation_correct :
  0.000000007 = 7 * 10^(-9) :=
by
  sorry

end scientific_notation_correct_l1566_156681


namespace fill_bathtub_time_l1566_156690

def rate_cold_water : ℚ := 3 / 20
def rate_hot_water : ℚ := 1 / 8
def rate_drain : ℚ := 3 / 40
def net_rate : ℚ := rate_cold_water + rate_hot_water - rate_drain

theorem fill_bathtub_time :
  net_rate = 1/5 → (1 / net_rate) = 5 := by
  sorry

end fill_bathtub_time_l1566_156690


namespace anna_walk_distance_l1566_156661

theorem anna_walk_distance (d: ℚ) 
  (hd: 22 * 1.25 - 4 * 1.25 = d)
  (d2: d = 3.7): d = 3.7 :=
by 
  sorry

end anna_walk_distance_l1566_156661


namespace no_rain_four_days_l1566_156614

-- Define the probability of rain on any given day
def prob_rain : ℚ := 2/3

-- Define the probability that it does not rain on any given day
def prob_no_rain : ℚ := 1 - prob_rain

-- Define the probability that it does not rain at all over four days
def prob_no_rain_four_days : ℚ := prob_no_rain^4

theorem no_rain_four_days : prob_no_rain_four_days = 1/81 := by
  sorry

end no_rain_four_days_l1566_156614


namespace Carol_cleaning_time_l1566_156694

theorem Carol_cleaning_time 
(Alice_time : ℕ) 
(Bob_time : ℕ) 
(Carol_time : ℕ) 
(h1 : Alice_time = 40) 
(h2 : Bob_time = 3 * Alice_time / 4) 
(h3 : Carol_time = 2 * Bob_time) :
  Carol_time = 60 := 
sorry

end Carol_cleaning_time_l1566_156694


namespace james_toys_l1566_156648

-- Define the conditions and the problem statement
theorem james_toys (x : ℕ) (h1 : ∀ x, 2 * x = 60 - x) : x = 20 :=
sorry

end james_toys_l1566_156648


namespace initial_investment_proof_l1566_156676

-- Definitions for the conditions
def initial_investment_A : ℝ := sorry
def contribution_B : ℝ := 15750
def profit_ratio_A : ℝ := 2
def profit_ratio_B : ℝ := 3
def time_A : ℝ := 12
def time_B : ℝ := 4

-- Lean statement to prove
theorem initial_investment_proof : initial_investment_A * time_A * profit_ratio_B = contribution_B * time_B * profit_ratio_A → initial_investment_A = 1750 :=
by
  sorry

end initial_investment_proof_l1566_156676


namespace linear_combination_value_l1566_156677

theorem linear_combination_value (x y : ℝ) (h₁ : 2 * x + y = 8) (h₂ : x + 2 * y = 10) :
  8 * x ^ 2 + 10 * x * y + 8 * y ^ 2 = 164 :=
sorry

end linear_combination_value_l1566_156677


namespace contrapositive_of_implication_l1566_156627

theorem contrapositive_of_implication (a : ℝ) (h : a > 0 → a > 1) : a ≤ 1 → a ≤ 0 :=
by
  sorry

end contrapositive_of_implication_l1566_156627


namespace solve_eq_l1566_156675

theorem solve_eq {x y z : ℕ} :
  2^x + 3^y - 7 = z! ↔ (x = 2 ∧ y = 2 ∧ z = 3) ∨ (x = 2 ∧ y = 3 ∧ z = 4) :=
by
  sorry -- Proof should be provided here

end solve_eq_l1566_156675


namespace union_sets_l1566_156674

-- Define the sets A and B using their respective conditions.
def A : Set ℝ := {x : ℝ | 3 < x ∧ x ≤ 7}
def B : Set ℝ := {x : ℝ | 4 < x ∧ x ≤ 10}

-- The theorem we aim to prove.
theorem union_sets : A ∪ B = {x : ℝ | 3 < x ∧ x ≤ 10} := 
by
  sorry

end union_sets_l1566_156674


namespace part1_part2_l1566_156637

variable (a b c : ℝ)

-- Conditions
axiom pos_a : a > 0
axiom pos_b : b > 0
axiom pos_c : c > 0
axiom eq_sum_square : a^2 + b^2 + 4*c^2 = 3

-- Part 1
theorem part1 : a + b + 2 * c ≤ 3 := 
by
  sorry

-- Additional condition for Part 2
variable (hc : b = 2*c)

-- Part 2
theorem part2 : (1 / a) + (1 / c) ≥ 3 :=
by
  sorry

end part1_part2_l1566_156637


namespace boxes_needed_to_complete_flooring_l1566_156646

-- Definitions of given conditions
def length_of_living_room : ℕ := 16
def width_of_living_room : ℕ := 20
def sq_ft_per_box : ℕ := 10
def sq_ft_already_covered : ℕ := 250

-- Statement to prove
theorem boxes_needed_to_complete_flooring : 
  (length_of_living_room * width_of_living_room - sq_ft_already_covered) / sq_ft_per_box = 7 :=
by
  sorry

end boxes_needed_to_complete_flooring_l1566_156646


namespace arithmetic_progression_product_difference_le_one_l1566_156613

theorem arithmetic_progression_product_difference_le_one 
  (a b : ℝ) :
  ∃ (m n k l : ℤ), |(a + b * m) * (a + b * n) - (a + b * k) * (a + b * l)| ≤ 1 :=
sorry

end arithmetic_progression_product_difference_le_one_l1566_156613


namespace simplify_and_evaluate_l1566_156684

theorem simplify_and_evaluate (a : ℕ) (h : a = 2) : 
  (1 - (1 : ℚ) / (a + 1)) / (a / ((a * a) - 1)) = 1 := by
  sorry

end simplify_and_evaluate_l1566_156684


namespace loss_due_to_simple_interest_l1566_156603

noncomputable def compound_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  P * (1 + r)^t

noncomputable def simple_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  P * r * t

theorem loss_due_to_simple_interest (P : ℝ) (r : ℝ) (t : ℝ)
  (hP : P = 2500) (hr : r = 0.04) (ht : t = 2) :
  let CI := compound_interest P r t
  let SI := simple_interest P r t
  ∃ loss : ℝ, loss = CI - SI ∧ loss = 4 :=
by
  sorry

end loss_due_to_simple_interest_l1566_156603


namespace x_eq_zero_sufficient_not_necessary_l1566_156650

theorem x_eq_zero_sufficient_not_necessary (x : ℝ) : 
  (x = 0 → x^2 - 2 * x = 0) ∧ (x^2 - 2 * x = 0 → x = 0 ∨ x = 2) :=
by
  sorry

end x_eq_zero_sufficient_not_necessary_l1566_156650


namespace find_initial_days_provisions_last_l1566_156699

def initial_days_provisions_last (initial_men reinforcements days_after_reinforcement : ℕ) (x : ℕ) : Prop :=
  initial_men * (x - 15) = (initial_men + reinforcements) * days_after_reinforcement

theorem find_initial_days_provisions_last
  (initial_men reinforcements days_after_reinforcement x : ℕ)
  (h1 : initial_men = 2000)
  (h2 : reinforcements = 1900)
  (h3 : days_after_reinforcement = 20)
  (h4 : initial_days_provisions_last initial_men reinforcements days_after_reinforcement x) :
  x = 54 :=
by
  sorry


end find_initial_days_provisions_last_l1566_156699


namespace fish_weight_l1566_156656

variables (W G T : ℕ)

-- Define the known conditions
axiom tail_weight : W = 1
axiom head_weight : G = W + T / 2
axiom torso_weight : T = G + W

-- Define the proof statement
theorem fish_weight : W + G + T = 8 :=
by
  sorry

end fish_weight_l1566_156656


namespace probability_at_least_one_black_eq_seven_tenth_l1566_156604

noncomputable def probability_drawing_at_least_one_black_ball : ℚ :=
  let total_ways := Nat.choose 5 2
  let ways_no_black := Nat.choose 3 2
  1 - (ways_no_black / total_ways)

theorem probability_at_least_one_black_eq_seven_tenth :
  probability_drawing_at_least_one_black_ball = 7 / 10 :=
by
  sorry

end probability_at_least_one_black_eq_seven_tenth_l1566_156604


namespace problem_statement_l1566_156651

theorem problem_statement (a b : ℝ) (h1 : a - b = 5) (h2 : a * b = 2) : a^2 + b^2 = 29 := 
by
  sorry

end problem_statement_l1566_156651


namespace find_number_of_students_l1566_156610

theorem find_number_of_students
    (S N : ℕ) 
    (h₁ : 4 * S + 3 = N)
    (h₂ : 5 * S = N + 6) : 
  S = 9 :=
by
  sorry

end find_number_of_students_l1566_156610


namespace tingting_solution_correct_l1566_156625

noncomputable def product_of_square_roots : ℝ :=
  (Real.sqrt 8) * (Real.sqrt 18)

theorem tingting_solution_correct : product_of_square_roots = 12 := by
  sorry

end tingting_solution_correct_l1566_156625


namespace valid_passwords_l1566_156667

theorem valid_passwords (total_passwords restricted_passwords : Nat) 
  (h_total : total_passwords = 10^4)
  (h_restricted : restricted_passwords = 8) : 
  total_passwords - restricted_passwords = 9992 := by
  sorry

end valid_passwords_l1566_156667


namespace bank_teller_rolls_of_coins_l1566_156620

theorem bank_teller_rolls_of_coins (tellers : ℕ) (coins_per_roll : ℕ) (total_coins : ℕ) (h_tellers : tellers = 4) (h_coins_per_roll : coins_per_roll = 25) (h_total_coins : total_coins = 1000) : 
  (total_coins / tellers) / coins_per_roll = 10 :=
by 
  sorry

end bank_teller_rolls_of_coins_l1566_156620


namespace adamek_marbles_l1566_156642

theorem adamek_marbles : ∃ n : ℕ, (∀ k : ℕ, n = 4 * k ∧ n = 3 * (k + 8)) → n = 96 :=
by
  sorry

end adamek_marbles_l1566_156642


namespace charge_per_call_proof_l1566_156692

-- Define the conditions as given in the problem
def fixed_rental : ℝ := 350
def free_calls_per_month : ℕ := 200
def charge_per_call_exceed_200 (x : ℝ) (calls : ℕ) : ℝ := 
  if calls > 200 then (calls - 200) * x else 0

def charge_per_call_exceed_400 : ℝ := 1.6
def discount_rate : ℝ := 0.28
def february_calls : ℕ := 150
def march_calls : ℕ := 250
def march_discount (x : ℝ) : ℝ := x * (1 - discount_rate)
def total_march_charge (x : ℝ) : ℝ := 
  fixed_rental + charge_per_call_exceed_200 (march_discount x) march_calls

-- Prove the correct charge per call when calls exceed 200 per month
theorem charge_per_call_proof (x : ℝ) : 
  charge_per_call_exceed_200 x february_calls = 0 ∧ 
  total_march_charge x = fixed_rental + (march_calls - free_calls_per_month) * (march_discount x) → 
  x = x := 
by { 
  sorry 
}

end charge_per_call_proof_l1566_156692


namespace domain_of_f_eq_l1566_156686

def domain_of_fractional_function : Set ℝ := 
  { x : ℝ | x > -1 }

theorem domain_of_f_eq : 
  ∀ x : ℝ, x ∈ domain_of_fractional_function ↔ x > -1 :=
by
  sorry -- Proof this part in Lean 4. The domain of f(x) is (-1, +∞)

end domain_of_f_eq_l1566_156686


namespace union_of_sets_l1566_156634

variable (x : ℝ)

def A : Set ℝ := {x | 0 < x ∧ x < 3}
def B : Set ℝ := {x | -1 < x ∧ x < 2}
def target : Set ℝ := {x | -1 < x ∧ x < 3}

theorem union_of_sets : (A ∪ B) = target :=
by
  sorry

end union_of_sets_l1566_156634


namespace tank_salt_solution_l1566_156633

theorem tank_salt_solution (x : ℝ) (h1 : (0.20 * x + 14) / ((3 / 4) * x + 21) = 1 / 3) : x = 140 :=
sorry

end tank_salt_solution_l1566_156633


namespace total_cost_supplies_l1566_156647

-- Definitions based on conditions
def cost_bow : ℕ := 5
def cost_vinegar : ℕ := 2
def cost_baking_soda : ℕ := 1
def cost_per_student : ℕ := cost_bow + cost_vinegar + cost_baking_soda
def number_of_students : ℕ := 23

-- Statement to be proven
theorem total_cost_supplies : cost_per_student * number_of_students = 184 := by
  sorry

end total_cost_supplies_l1566_156647


namespace arithmetic_mean_is_correct_l1566_156626

open Nat

noncomputable def arithmetic_mean_of_two_digit_multiples_of_9 : ℝ :=
  let smallest := 18
  let largest := 99
  let n := 10
  let sum := (n / 2 : ℝ) * (smallest + largest)
  sum / n

theorem arithmetic_mean_is_correct :
  arithmetic_mean_of_two_digit_multiples_of_9 = 58.5 :=
by
  -- Proof goes here
  sorry

end arithmetic_mean_is_correct_l1566_156626


namespace no_real_solution_to_system_l1566_156645

theorem no_real_solution_to_system :
  ∀ (x y z : ℝ), (x + y - 2 - 4 * x * y = 0) ∧
                 (y + z - 2 - 4 * y * z = 0) ∧
                 (z + x - 2 - 4 * z * x = 0) → false := 
by 
    intros x y z h
    rcases h with ⟨h1, h2, h3⟩
    -- Here would be the proof steps, which are omitted.
    sorry

end no_real_solution_to_system_l1566_156645


namespace Tom_spends_375_dollars_l1566_156617

noncomputable def totalCost (numBricks : ℕ) (halfDiscount : ℚ) (fullPrice : ℚ) : ℚ :=
  let halfBricks := numBricks / 2
  let discountedPrice := fullPrice * halfDiscount
  (halfBricks * discountedPrice) + (halfBricks * fullPrice)

theorem Tom_spends_375_dollars : 
  ∀ (numBricks : ℕ) (halfDiscount fullPrice : ℚ), 
  numBricks = 1000 → halfDiscount = 0.5 → fullPrice = 0.5 → totalCost numBricks halfDiscount fullPrice = 375 := 
by
  intros numBricks halfDiscount fullPrice hnumBricks hhalfDiscount hfullPrice
  rw [hnumBricks, hhalfDiscount, hfullPrice]
  sorry

end Tom_spends_375_dollars_l1566_156617


namespace crayons_in_new_set_l1566_156698

theorem crayons_in_new_set (initial_crayons : ℕ) (half_loss : ℕ) (total_after_purchase : ℕ) (initial_crayons_eq : initial_crayons = 18) (half_loss_eq : half_loss = initial_crayons / 2) (total_eq : total_after_purchase = 29) :
  total_after_purchase - (initial_crayons - half_loss) = 20 :=
by
  sorry

end crayons_in_new_set_l1566_156698


namespace problem1_problem2_l1566_156670

theorem problem1 : 27^((2:ℝ)/(3:ℝ)) - 2^(Real.logb 2 3) * Real.logb 2 (1/8) = 18 := 
by
  sorry -- proof omitted

theorem problem2 : 1/(Real.sqrt 5 - 2) - (Real.sqrt 5 + 2)^0 - Real.sqrt ((2 - Real.sqrt 5)^2) = 2*(Real.sqrt 5 - 1) := 
by
  sorry -- proof omitted

end problem1_problem2_l1566_156670

import Mathlib

namespace forest_area_relationship_l626_62696

variable (a b c x : ℝ)

theorem forest_area_relationship
    (hb : b = a * (1 + x))
    (hc : c = a * (1 + x) ^ 2) :
    a * c = b ^ 2 := by
  sorry

end forest_area_relationship_l626_62696


namespace trapezoid_height_proof_l626_62648

-- Given lengths of the diagonals and the midline of the trapezoid
def diagonal1Length : ℝ := 6
def diagonal2Length : ℝ := 8
def midlineLength : ℝ := 5

-- Target to prove: Height of the trapezoid
def trapezoidHeight : ℝ := 4.8

theorem trapezoid_height_proof :
  ∀ (d1 d2 m : ℝ), d1 = diagonal1Length → d2 = diagonal2Length → m = midlineLength → trapezoidHeight = 4.8 :=
by intros d1 d2 m hd1 hd2 hm; sorry

end trapezoid_height_proof_l626_62648


namespace sum_of_arithmetic_sequence_l626_62623

theorem sum_of_arithmetic_sequence :
  let a := -3
  let d := 6
  let n := 10
  let a_n := a + (n - 1) * d
  let S_n := (n / 2) * (a + a_n)
  S_n = 240 := by {
  let a := -3
  let d := 6
  let n := 10
  let a_n := a + (n - 1) * d
  let S_n := (n / 2) * (a + a_n)
  sorry
}

end sum_of_arithmetic_sequence_l626_62623


namespace cone_lateral_area_l626_62663

theorem cone_lateral_area (r l : ℝ) (h_r : r = 3) (h_l : l = 5) : 
  (1 / 2) * (2 * Real.pi * r) * l = 15 * Real.pi :=
by
  rw [h_r, h_l]
  sorry

end cone_lateral_area_l626_62663


namespace total_revenue_correct_l626_62608

def sections := 5
def seats_per_section_1_4 := 246
def seats_section_5 := 314
def ticket_price_1_4 := 15
def ticket_price_5 := 20

theorem total_revenue_correct :
  4 * seats_per_section_1_4 * ticket_price_1_4 + seats_section_5 * ticket_price_5 = 21040 :=
by
  sorry

end total_revenue_correct_l626_62608


namespace bike_ride_energetic_time_l626_62689

theorem bike_ride_energetic_time :
  ∃ x : ℚ, (22 * x + 15 * (7.5 - x) = 142) ∧ x = (59 / 14) :=
by
  sorry

end bike_ride_energetic_time_l626_62689


namespace morse_code_count_l626_62660

noncomputable def morse_code_sequences : Nat :=
  let case_1 := 2            -- 1 dot or dash
  let case_2 := 2 * 2        -- 2 dots or dashes
  let case_3 := 2 * 2 * 2    -- 3 dots or dashes
  let case_4 := 2 * 2 * 2 * 2-- 4 dots or dashes
  let case_5 := 2 * 2 * 2 * 2 * 2 -- 5 dots or dashes
  case_1 + case_2 + case_3 + case_4 + case_5

theorem morse_code_count : morse_code_sequences = 62 := by
  sorry

end morse_code_count_l626_62660


namespace solve_first_system_solve_second_system_l626_62624

theorem solve_first_system :
  (exists x y : ℝ, 3 * x + 2 * y = 6 ∧ y = x - 2) ->
  (∃ (x y : ℝ), x = 2 ∧ y = 0) := by
  sorry

theorem solve_second_system :
  (exists m n : ℝ, m + 2 * n = 7 ∧ -3 * m + 5 * n = 1) ->
  (∃ (m n : ℝ), m = 3 ∧ n = 2) := by
  sorry

end solve_first_system_solve_second_system_l626_62624


namespace samuel_remaining_distance_l626_62677

noncomputable def remaining_distance
  (total_distance : ℕ)
  (segment1_speed : ℕ) (segment1_time : ℕ)
  (segment2_speed : ℕ) (segment2_time : ℕ)
  (segment3_speed : ℕ) (segment3_time : ℕ)
  (segment4_speed : ℕ) (segment4_time : ℕ) : ℕ :=
  total_distance -
  (segment1_speed * segment1_time +
   segment2_speed * segment2_time +
   segment3_speed * segment3_time +
   segment4_speed * segment4_time)

theorem samuel_remaining_distance :
  remaining_distance 1200 60 2 70 3 50 4 80 5 = 270 :=
by
  sorry

end samuel_remaining_distance_l626_62677


namespace total_sheets_of_paper_l626_62694

theorem total_sheets_of_paper (num_classes : ℕ) (students_per_class : ℕ) (sheets_per_student : ℕ)
  (h1 : num_classes = 4) (h2 : students_per_class = 20) (h3 : sheets_per_student = 5) :
  (num_classes * students_per_class * sheets_per_student) = 400 :=
by
  sorry

end total_sheets_of_paper_l626_62694


namespace symmetric_points_on_parabola_l626_62649

theorem symmetric_points_on_parabola
  (x1 x2 : ℝ)
  (m : ℝ)
  (h1 : 2 * x1 * x1 = 2 * x2 * x2)
  (h2 : 2 * x1 * x1 = 2 * x2 * x2 + m)
  (h3 : x1 * x2 = -1 / 2)
  (h4 : x1 + x2 = -1 / 2)
  : m = 3 / 2 :=
sorry

end symmetric_points_on_parabola_l626_62649


namespace monthly_income_A_l626_62605

theorem monthly_income_A (A B C : ℝ) :
  A + B = 10100 ∧ B + C = 12500 ∧ A + C = 10400 →
  A = 4000 :=
by
  intro h
  have h1 : A + B = 10100 := h.1
  have h2 : B + C = 12500 := h.2.1
  have h3 : A + C = 10400 := h.2.2
  sorry

end monthly_income_A_l626_62605


namespace range_of_a_l626_62661

def proposition_p (a : ℝ) : Prop := a > 1
def proposition_q (a : ℝ) : Prop := 0 < a ∧ a < 4

theorem range_of_a
(a : ℝ)
(h1 : a > 0)
(h2 : ¬ proposition_p a)
(h3 : ¬ proposition_q a)
(h4 : proposition_p a ∨ proposition_q a) :
  (0 < a ∧ a ≤ 1) ∨ (4 ≤ a) :=
by sorry

end range_of_a_l626_62661


namespace product_equals_32_l626_62669

theorem product_equals_32 :
  (1/4) * 8 * (1/16) * 32 * (1/64) * 128 * (1/256) * 512 * (1/1024) * 2048 = 32 :=
by
  sorry

end product_equals_32_l626_62669


namespace side_length_square_field_l626_62621

-- Definitions based on the conditions.
def time_taken := 56 -- in seconds
def speed := 9 * 1000 / 3600 -- in meters per second, converting 9 km/hr to m/s
def distance_covered := speed * time_taken -- calculating the distance covered in meters
def perimeter := 4 * 35 -- defining the perimeter given the side length is 35

-- Problem statement for proof: We need to prove that the calculated distance covered matches the perimeter.
theorem side_length_square_field : distance_covered = perimeter :=
by
  sorry

end side_length_square_field_l626_62621


namespace find_x_l626_62632

variables {x y z : ℝ}

theorem find_x (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x^2 / y = 2) (h2 : y^2 / z = 3) (h3 : z^2 / x = 4) :
  x = 144^(1 / 5) :=
by
  sorry

end find_x_l626_62632


namespace sales_worth_l626_62688

def old_scheme_remuneration (S : ℝ) : ℝ := 0.05 * S
def new_scheme_remuneration (S : ℝ) : ℝ := 1300 + 0.025 * (S - 4000)
def remuneration_difference (S : ℝ) : ℝ := new_scheme_remuneration S - old_scheme_remuneration S

theorem sales_worth (S : ℝ) (h : remuneration_difference S = 600) : S = 24000 :=
by
  sorry

end sales_worth_l626_62688


namespace race_order_l626_62670

theorem race_order (overtakes_G_S_L : (ℕ × ℕ × ℕ))
  (h1 : overtakes_G_S_L.1 = 10)
  (h2 : overtakes_G_S_L.2.1 = 4)
  (h3 : overtakes_G_S_L.2.2 = 6)
  (h4 : ¬(overtakes_G_S_L.2.1 > 0 ∧ overtakes_G_S_L.2.2 > 0))
  (h5 : ∀ i j k : ℕ, i ≠ j → j ≠ k → k ≠ i)
  : overtakes_G_S_L = (10, 4, 6) :=
sorry

end race_order_l626_62670


namespace first_digit_base_4_of_853_l626_62629

theorem first_digit_base_4_of_853 : 
  ∃ (d : ℕ), d = 3 ∧ (d * 256 ≤ 853 ∧ 853 < (d + 1) * 256) :=
by
  sorry

end first_digit_base_4_of_853_l626_62629


namespace ruby_height_l626_62646

/-- Height calculations based on given conditions -/
theorem ruby_height (Janet_height : ℕ) (Charlene_height : ℕ) (Pablo_height : ℕ) (Ruby_height : ℕ) 
  (h₁ : Janet_height = 62) 
  (h₂ : Charlene_height = 2 * Janet_height)
  (h₃ : Pablo_height = Charlene_height + 70)
  (h₄ : Ruby_height = Pablo_height - 2) : Ruby_height = 192 := 
by
  sorry

end ruby_height_l626_62646


namespace travel_time_l626_62664

namespace NatashaSpeedProblem

def distance : ℝ := 60
def speed_limit : ℝ := 50
def speed_over_limit : ℝ := 10
def actual_speed : ℝ := speed_limit + speed_over_limit

theorem travel_time : (distance / actual_speed) = 1 := by
  sorry

end NatashaSpeedProblem

end travel_time_l626_62664


namespace monotonically_increasing_intervals_inequality_solution_set_l626_62643

-- Given conditions for f
def f (a b c d : ℝ) (x : ℝ) : ℝ := a*x^3 + b*x^2 + c*x + d

-- Ⅰ) Prove the intervals of monotonic increase
theorem monotonically_increasing_intervals (a c : ℝ) (x : ℝ) (h_f : ∀ x, f a 0 c 0 x = a*x^3 + c*x)
  (h_a : a = 1) (h_c : c = -3) :
  (∀ x < -1, f a 0 c 0 x < 0) ∧ (∀ x > 1, f a 0 c 0 x > 0) := 
sorry

-- Ⅱ) Prove the solution sets for the inequality given m
theorem inequality_solution_set (m x : ℝ) :
  (m = 0 → x > 0) ∧
  (m > 0 → (x > 4*m ∨ 0 < x ∧ x < m)) ∧
  (m < 0 → (x > 0 ∨ 4*m < x ∧ x < m)) :=
sorry

end monotonically_increasing_intervals_inequality_solution_set_l626_62643


namespace orange_juice_fraction_in_mixture_l626_62698

theorem orange_juice_fraction_in_mixture :
  let capacity1 := 800
  let capacity2 := 700
  let fraction1 := (1 : ℚ) / 4
  let fraction2 := (3 : ℚ) / 7
  let orange_juice1 := capacity1 * fraction1
  let orange_juice2 := capacity2 * fraction2
  let total_orange_juice := orange_juice1 + orange_juice2
  let total_volume := capacity1 + capacity2
  let fraction := total_orange_juice / total_volume
  fraction = (1 : ℚ) / 3 := by
  sorry

end orange_juice_fraction_in_mixture_l626_62698


namespace range_of_a_l626_62635

noncomputable def has_real_roots (a : ℝ) : Prop :=
  ∃ x y : ℝ, x^2 - a*x + 1 = 0 ∧ y^2 - a*y + 1 = 0

def holds_for_all_x (a : ℝ) : Prop :=
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → a^2 - 3*a - x + 1 ≤ 0

theorem range_of_a (a : ℝ) :
  (¬ ((has_real_roots a) ∧ (holds_for_all_x a))) ∧ (¬ (¬ (holds_for_all_x a))) → (1 ≤ a ∧ a < 2) :=
by
  sorry

end range_of_a_l626_62635


namespace meaningful_sqrt_l626_62675

theorem meaningful_sqrt (a : ℝ) (h : a - 4 ≥ 0) : a ≥ 4 :=
sorry

end meaningful_sqrt_l626_62675


namespace meeting_distance_and_time_l626_62620

theorem meeting_distance_and_time 
  (total_distance : ℝ)
  (delta_time : ℝ)
  (x : ℝ)
  (V : ℝ)
  (v : ℝ)
  (t : ℝ) :

  -- Conditions 
  total_distance = 150 ∧
  delta_time = 25 ∧
  (150 - 2 * x) = 25 ∧
  (62.5 / v) = (87.5 / V) ∧
  (150 / v) - (150 / V) = 25 ∧
  t = (62.5 / v)

  -- Show that 
  → x = 62.5 ∧ t = 36 + 28 / 60 := 
by 
  sorry

end meeting_distance_and_time_l626_62620


namespace boys_to_girls_ratio_l626_62640

theorem boys_to_girls_ratio (S G B : ℕ) (h1 : 1 / 2 * G = 1 / 3 * S) (h2 : S = B + G) : B / G = 1 / 2 :=
by
  -- Placeholder for the actual proof
  sorry

end boys_to_girls_ratio_l626_62640


namespace evaluate_expression_at_3_l626_62638

theorem evaluate_expression_at_3 :
  (1 / (3 + 1 / (3 + 1 / (3 - 1 / 3)))) = 0.30337078651685395 :=
  sorry

end evaluate_expression_at_3_l626_62638


namespace time_to_run_100_meters_no_wind_l626_62678

-- Definitions based on the conditions
variables (v w : ℝ)
axiom speed_with_wind : v + w = 9
axiom speed_against_wind : v - w = 7

-- The theorem statement to prove
theorem time_to_run_100_meters_no_wind : (100 / v) = 12.5 :=
by 
  sorry

end time_to_run_100_meters_no_wind_l626_62678


namespace simplify_expr1_simplify_expr2_l626_62612

-- Proof problem for the first expression
theorem simplify_expr1 (x y : ℤ) : (2 - x + 3 * y + 8 * x - 5 * y - 6) = (7 * x - 2 * y -4) := 
by 
   -- Proving steps would go here
   sorry

-- Proof problem for the second expression
theorem simplify_expr2 (a b : ℤ) : (15 * a^2 * b - 12 * a * b^2 + 12 - 4 * a^2 * b - 18 + 8 * a * b^2) = (11 * a^2 * b - 4 * a * b^2 - 6) := 
by 
   -- Proving steps would go here
   sorry

end simplify_expr1_simplify_expr2_l626_62612


namespace complex_div_eq_half_add_half_i_l626_62626

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem to be proven
theorem complex_div_eq_half_add_half_i :
  (i / (1 + i)) = (1 / 2 + (1 / 2) * i) :=
by
  -- The proof will go here
  sorry

end complex_div_eq_half_add_half_i_l626_62626


namespace ones_digit_of_22_to_22_11_11_l626_62686

theorem ones_digit_of_22_to_22_11_11 : (22 ^ (22 * (11 ^ 11))) % 10 = 4 :=
by
  sorry

end ones_digit_of_22_to_22_11_11_l626_62686


namespace spider_travel_distance_l626_62693

theorem spider_travel_distance (r : ℝ) (journey3 : ℝ) (diameter : ℝ) (leg2 : ℝ) :
    r = 75 → journey3 = 110 → diameter = 2 * r → 
    leg2 = Real.sqrt (diameter^2 - journey3^2) → 
    diameter + leg2 + journey3 = 362 :=
by
  sorry

end spider_travel_distance_l626_62693


namespace find_integer_n_l626_62650

theorem find_integer_n (n : ℤ) : (⌊(n^2 / 9 : ℝ)⌋ - ⌊(n / 3 : ℝ)⌋ ^ 2 = 5) → n = 14 :=
by
  -- Proof is omitted
  sorry

end find_integer_n_l626_62650


namespace S8_value_l626_62630

theorem S8_value (a : ℕ → ℝ) (S : ℕ → ℝ) (h1 : S 5 / 5 + S 11 / 11 = 12) (h2 : S 11 = S 8 + 1 / a 9 + 1 / a 10 + 1 / a 11) : S 8 = 48 :=
sorry

end S8_value_l626_62630


namespace complement_of_M_in_U_l626_62607

-- Definition of the universal set U
def U : Set ℝ := { x | 1 ≤ x ∧ x ≤ 5 }

-- Definition of the set M
def M : Set ℝ := { 1 }

-- The statement to prove
theorem complement_of_M_in_U : (U \ M) = {x | 1 < x ∧ x ≤ 5} :=
by
  sorry

end complement_of_M_in_U_l626_62607


namespace identify_base_7_l626_62610

theorem identify_base_7 :
  ∃ b : ℕ, (b > 1) ∧ 
  (2 * b^4 + 3 * b^3 + 4 * b^2 + 5 * b^1 + 1 * b^0) +
  (1 * b^4 + 5 * b^3 + 6 * b^2 + 4 * b^1 + 2 * b^0) =
  (4 * b^4 + 2 * b^3 + 4 * b^2 + 2 * b^1 + 3 * b^0) ∧
  b = 7 :=
by
  sorry

end identify_base_7_l626_62610


namespace set_intersection_l626_62614

def A : Set ℝ := { x | -1 < x ∧ x < 2 }
def B : Set ℝ := { x | -2 < x ∧ x < 1 }

theorem set_intersection :
  A ∩ B = { x | -1 < x ∧ x < 1 } := 
sorry

end set_intersection_l626_62614


namespace min_value_expression_l626_62644

theorem min_value_expression (x y : ℝ) (h : y^2 - 2*x + 4 = 0) : 
  ∃ z : ℝ, z = x^2 + y^2 + 2*x ∧ z = -8 :=
by
  sorry

end min_value_expression_l626_62644


namespace each_child_consumes_3_bottles_per_day_l626_62633

noncomputable def bottles_per_child_per_day : ℕ :=
  let first_group := 14
  let second_group := 16
  let third_group := 12
  let fourth_group := (first_group + second_group + third_group) / 2
  let total_children := first_group + second_group + third_group + fourth_group
  let cases_of_water := 13
  let bottles_per_case := 24
  let initial_bottles := cases_of_water * bottles_per_case
  let additional_bottles := 255
  let total_bottles := initial_bottles + additional_bottles
  let bottles_per_child := total_bottles / total_children
  let days := 3
  bottles_per_child / days

theorem each_child_consumes_3_bottles_per_day :
  bottles_per_child_per_day = 3 :=
by
  sorry

end each_child_consumes_3_bottles_per_day_l626_62633


namespace record_withdrawal_example_l626_62628

-- Definitions based on conditions
def ten_thousand_dollars := 10000
def record_deposit (amount : ℕ) : ℤ := amount / ten_thousand_dollars
def record_withdrawal (amount : ℕ) : ℤ := -(amount / ten_thousand_dollars)

-- Lean 4 statement to prove the problem
theorem record_withdrawal_example :
  (record_deposit 30000 = 3) → (record_withdrawal 20000 = -2) :=
by
  intro h
  sorry

end record_withdrawal_example_l626_62628


namespace hockey_league_games_l626_62673

theorem hockey_league_games (n t : ℕ) (h1 : n = 15) (h2 : t = 1050) :
  ∃ k, ∀ team1 team2 : ℕ, team1 ≠ team2 → k = 10 :=
by
  -- Declare k as the number of times each team faces the other teams
  let k := 10
  -- Verify the total number of teams and games
  have hn : n = 15 := h1
  have ht : t = 1050 := h2
  -- For any two distinct teams, they face each other k times
  use k
  intros team1 team2 hneq
  -- Show that k equals 10 under given conditions
  exact rfl

end hockey_league_games_l626_62673


namespace sum_reciprocal_eq_l626_62653

theorem sum_reciprocal_eq :
  ∃ (a b : ℕ), a + b = 45 ∧ Nat.lcm a b = 120 ∧ Nat.gcd a b = 5 ∧ 
  (1/a + 1/b = (3 : ℚ) / 40) := by
  sorry

end sum_reciprocal_eq_l626_62653


namespace cheezit_bag_weight_l626_62697

-- Definitions based on the conditions of the problem
def cheezit_bags : ℕ := 3
def calories_per_ounce : ℕ := 150
def run_minutes : ℕ := 40
def calories_per_minute : ℕ := 12
def excess_calories : ℕ := 420

-- Main theorem stating the question with the solution
theorem cheezit_bag_weight (x : ℕ) : 
  (calories_per_ounce * cheezit_bags * x) - (run_minutes * calories_per_minute) = excess_calories → 
  x = 2 :=
by
  sorry

end cheezit_bag_weight_l626_62697


namespace tammy_speed_on_second_day_l626_62603

-- Definitions of the conditions
variables (t v : ℝ)
def total_hours := 14
def total_distance := 52

-- Distance equation
def distance_eq := v * t + (v + 0.5) * (t - 2) = total_distance

-- Time equation
def time_eq := t + (t - 2) = total_hours

theorem tammy_speed_on_second_day :
  (time_eq t ∧ distance_eq v t) → v + 0.5 = 4 :=
by sorry

end tammy_speed_on_second_day_l626_62603


namespace max_circles_in_annulus_l626_62616

theorem max_circles_in_annulus (r_inner r_outer : ℝ) (h1 : r_inner = 1) (h2 : r_outer = 9) :
  ∃ n : ℕ, n = 3 ∧ ∀ r : ℝ, r = (r_outer - r_inner) / 2 → r * 3 ≤ 360 :=
sorry

end max_circles_in_annulus_l626_62616


namespace mandy_med_school_ratio_l626_62658

theorem mandy_med_school_ratio 
    (researched_schools : ℕ)
    (applied_ratio : ℚ)
    (accepted_schools : ℕ)
    (h1 : researched_schools = 42)
    (h2 : applied_ratio = 1 / 3)
    (h3 : accepted_schools = 7)
    : (accepted_schools : ℚ) / ((researched_schools : ℚ) * applied_ratio) = 1 / 2 :=
by sorry

end mandy_med_school_ratio_l626_62658


namespace find_x_l626_62667

theorem find_x (x : ℝ) (h : (40 / 100) * x = (25 / 100) * 80) : x = 50 :=
by
  sorry

end find_x_l626_62667


namespace min_m_value_l626_62674

theorem min_m_value :
  ∃ (x y m : ℝ), x - y + 2 ≥ 0 ∧ x + y - 2 ≤ 0 ∧ 2 * y ≥ x + 2 ∧
  (m > 0) ∧ (x^2 / 4 + y^2 = m^2) ∧ m = Real.sqrt 2 / 2 :=
sorry

end min_m_value_l626_62674


namespace wire_lengths_l626_62631

variables (total_length first second third fourth : ℝ)

def wire_conditions : Prop :=
  total_length = 72 ∧
  first = second + 3 ∧
  third = 2 * second - 2 ∧
  fourth = 0.5 * (first + second + third) ∧
  second + first + third + fourth = total_length

theorem wire_lengths 
  (h : wire_conditions total_length first second third fourth) :
  second = 11.75 ∧ first = 14.75 ∧ third = 21.5 ∧ fourth = 24 :=
sorry

end wire_lengths_l626_62631


namespace intersectionAandB_l626_62676

def setA (x : ℝ) : Prop := abs (x + 3) + abs (x - 4) ≤ 9
def setB (x : ℝ) : Prop := ∃ t : ℝ, 0 < t ∧ x = 4 * t + 1 / t - 6

theorem intersectionAandB : {x : ℝ | setA x} ∩ {x : ℝ | setB x} = {x : ℝ | -2 ≤ x ∧ x ≤ 5} := 
by 
  sorry

end intersectionAandB_l626_62676


namespace john_new_earnings_l626_62642

theorem john_new_earnings (original_earnings raise_percentage: ℝ)
  (h1 : original_earnings = 60)
  (h2 : raise_percentage = 40) :
  original_earnings * (1 + raise_percentage / 100) = 84 := 
by
  sorry

end john_new_earnings_l626_62642


namespace image_of_2_in_set_B_l626_62699

theorem image_of_2_in_set_B (f : ℤ → ℤ) (h : ∀ x, f x = 2 * x + 1) : f 2 = 5 :=
by
  apply h

end image_of_2_in_set_B_l626_62699


namespace find_a_and_b_find_set_A_l626_62665

noncomputable def f (x a b : ℝ) := 4 ^ x - a * 2 ^ x + b

theorem find_a_and_b (a b : ℝ)
  (h₁ : f 1 a b = -1)
  (h₂ : ∀ x, ∃ t > 0, f x a b = t ^ 2 - a * t + b) :
  a = 4 ∧ b = 3 :=
sorry

theorem find_set_A (a b : ℝ)
  (ha : a = 4) (hb : b = 3) :
  {x : ℝ | f x a b ≤ 35} = {x : ℝ | x ≤ 3} :=
sorry

end find_a_and_b_find_set_A_l626_62665


namespace total_pencils_correct_l626_62609

def reeta_pencils : Nat := 20
def anika_pencils : Nat := 2 * reeta_pencils + 4
def total_pencils : Nat := reeta_pencils + anika_pencils

theorem total_pencils_correct : total_pencils = 64 :=
by
  sorry

end total_pencils_correct_l626_62609


namespace prism_diagonals_not_valid_l626_62627

theorem prism_diagonals_not_valid
  (a b c : ℕ)
  (h3 : a^2 + b^2 = 3^2 ∨ b^2 + c^2 = 3^2 ∨ a^2 + c^2 = 3^2)
  (h4 : a^2 + b^2 = 4^2 ∨ b^2 + c^2 = 4^2 ∨ a^2 + c^2 = 4^2)
  (h6 : a^2 + b^2 = 6^2 ∨ b^2 + c^2 = 6^2 ∨ a^2 + c^2 = 6^2) :
  False := 
sorry

end prism_diagonals_not_valid_l626_62627


namespace prop_for_real_l626_62690

theorem prop_for_real (a : ℝ) : (∀ x : ℝ, x^2 + 2 * x - a > 0) → a < -1 :=
by
  sorry

end prop_for_real_l626_62690


namespace reciprocal_opposite_of_neg_neg_3_is_neg_one_third_l626_62604

theorem reciprocal_opposite_of_neg_neg_3_is_neg_one_third : 
  (1 / (-(-3))) = -1 / 3 :=
by
  sorry

end reciprocal_opposite_of_neg_neg_3_is_neg_one_third_l626_62604


namespace calculate_weight_difference_l626_62602

noncomputable def joe_weight := 43 -- Joe's weight in kg
noncomputable def original_avg_weight := 30 -- Original average weight in kg
noncomputable def new_avg_weight := 31 -- New average weight in kg after Joe joins
noncomputable def final_avg_weight := 30 -- Final average weight after two students leave

theorem calculate_weight_difference :
  ∃ (n : ℕ) (x : ℝ), 
  (original_avg_weight * n + joe_weight) / (n + 1) = new_avg_weight ∧
  (new_avg_weight * (n + 1) - 2 * x) / (n - 1) = final_avg_weight →
  x - joe_weight = -6.5 :=
by
  sorry

end calculate_weight_difference_l626_62602


namespace cubical_tank_fraction_filled_l626_62613

theorem cubical_tank_fraction_filled (a : ℝ) (h1 : ∀ a:ℝ, (a * a * 1 = 16) )
  : (1 / 4) = (16 / (a^3)) :=
by
  sorry

end cubical_tank_fraction_filled_l626_62613


namespace William_won_10_rounds_l626_62655

theorem William_won_10_rounds (H : ℕ) (total_rounds : H + (H + 5) = 15) : H + 5 = 10 := by
  sorry

end William_won_10_rounds_l626_62655


namespace remainder_of_division_l626_62657

theorem remainder_of_division (L S R : ℕ) (h1 : L - S = 1365) (h2 : L = 1637) (h3 : L = 6 * S + R) : R = 5 :=
by
  sorry

end remainder_of_division_l626_62657


namespace sum_of_two_numbers_is_10_l626_62672

variable (a b : ℝ)

theorem sum_of_two_numbers_is_10
  (h1 : a + b = 10)
  (h2 : a - b = 8)
  (h3 : a^2 - b^2 = 80) :
  a + b = 10 :=
by
  sorry

end sum_of_two_numbers_is_10_l626_62672


namespace factor_theorem_l626_62639

theorem factor_theorem (m : ℝ) : (∀ x : ℝ, x + 5 = 0 → x ^ 2 - m * x - 40 = 0) → m = 3 :=
by
  sorry

end factor_theorem_l626_62639


namespace proof_problem_l626_62679

variable {α : Type*} [LinearOrderedField α]

theorem proof_problem 
  (a b x y : α) 
  (h0 : 0 < a ∧ 0 < b ∧ 0 < x ∧ 0 < y)
  (h1 : a + b + x + y < 2)
  (h2 : a + b^2 = x + y^2)
  (h3 : a^2 + b = x^2 + y) :
  a = x ∧ b = y := 
by
  sorry

end proof_problem_l626_62679


namespace cemc_basketball_team_l626_62668

theorem cemc_basketball_team (t g : ℕ) (h_t : t = 6)
  (h1 : 40 * t + 20 * g = 28 * (g + 4)) :
  g = 16 := by
  -- Start your proof here

  sorry

end cemc_basketball_team_l626_62668


namespace max_value_of_expr_l626_62615

theorem max_value_of_expr (A M C : ℕ) (h : A + M + C = 12) : 
  A * M * C + A * M + M * C + C * A ≤ 112 :=
sorry

end max_value_of_expr_l626_62615


namespace find_P_l626_62618

-- Define the variables A, B, C and their type
variables (A B C P : ℤ)

-- The main theorem statement according to the given conditions and question
theorem find_P (h1 : A = C + 1) (h2 : A + B = C + P) : P = 1 + B :=
by
  sorry

end find_P_l626_62618


namespace train_length_l626_62681

/-- 
Given that a train can cross an electric pole in 200 seconds and its speed is 18 km/h,
prove that the length of the train is 1000 meters.
-/
theorem train_length
  (time_to_cross : ℕ)
  (speed_kmph : ℕ)
  (h_time : time_to_cross = 200)
  (h_speed : speed_kmph = 18)
  : (speed_kmph * 1000 / 3600 * time_to_cross = 1000) :=
by
  sorry

end train_length_l626_62681


namespace angle_between_east_and_south_is_90_degrees_l626_62666

-- Define the main theorem statement
theorem angle_between_east_and_south_is_90_degrees :
  ∀ (circle : Type) (num_rays : ℕ) (direction : ℕ → ℕ) (north east south : ℕ),
  num_rays = 12 →
  (∀ i, i < num_rays → direction i = (i * 360 / num_rays) % 360) →
  direction north = 0 →
  direction east = 90 →
  direction south = 180 →
  (min ((direction south - direction east) % 360) (360 - (direction south - direction east) % 360)) = 90 :=
by
  intros
  -- Skipped the proof
  sorry

end angle_between_east_and_south_is_90_degrees_l626_62666


namespace arccos_one_eq_zero_l626_62685

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := 
by sorry

end arccos_one_eq_zero_l626_62685


namespace no_divide_five_to_n_minus_three_to_n_l626_62634

theorem no_divide_five_to_n_minus_three_to_n (n : ℕ) (h : n ≥ 1) : ¬ (2 ^ n + 65 ∣ 5 ^ n - 3 ^ n) :=
by
  sorry

end no_divide_five_to_n_minus_three_to_n_l626_62634


namespace probability_of_drawing_white_ball_probability_with_additional_white_balls_l626_62651

noncomputable def total_balls := 6 + 9 + 3
noncomputable def initial_white_balls := 3

theorem probability_of_drawing_white_ball :
  (initial_white_balls : ℚ) / (total_balls : ℚ) = 1 / 6 :=
sorry

noncomputable def additional_white_balls_needed := 2

theorem probability_with_additional_white_balls :
  (initial_white_balls + additional_white_balls_needed : ℚ) / (total_balls + additional_white_balls_needed : ℚ) = 1 / 4 :=
sorry

end probability_of_drawing_white_ball_probability_with_additional_white_balls_l626_62651


namespace describe_graph_of_equation_l626_62625

theorem describe_graph_of_equation :
  (∀ x y : ℝ, (x + y)^3 = x^3 + y^3 → (x = 0 ∨ y = 0 ∨ y = -x)) :=
by
  intros x y h
  sorry

end describe_graph_of_equation_l626_62625


namespace distinct_permutations_of_12233_l626_62662

def numFiveDigitIntegers : ℕ :=
  Nat.factorial 5 / (Nat.factorial 2 * Nat.factorial 2)

theorem distinct_permutations_of_12233 : numFiveDigitIntegers = 30 := by
  sorry

end distinct_permutations_of_12233_l626_62662


namespace find_k_l626_62601

def a : ℕ := 786
def b : ℕ := 74
def c : ℝ := 1938.8

theorem find_k (k : ℝ) : (a * b) / k = c → k = 30 :=
by
  intro h
  sorry

end find_k_l626_62601


namespace point_on_inverse_proportion_l626_62671

theorem point_on_inverse_proportion (k : ℝ) (hk : k ≠ 0) :
  (2 * 3 = k) → (1 * 6 = k) :=
by
  intro h
  sorry

end point_on_inverse_proportion_l626_62671


namespace polynomial_not_separable_l626_62652

theorem polynomial_not_separable (f g : Polynomial ℂ) :
  (∀ x y : ℂ, f.eval x * g.eval y = x^200 * y^200 + 1) → False :=
sorry

end polynomial_not_separable_l626_62652


namespace quadratic_real_roots_l626_62617

theorem quadratic_real_roots (m : ℝ) : 
  let a := m - 3
  let b := -2
  let c := 1
  let discriminant := b^2 - 4 * a * c
  (discriminant ≥ 0) ↔ (m ≤ 4 ∧ m ≠ 3) :=
by
  let a := m - 3
  let b := -2
  let c := 1
  let discriminant := b^2 - 4 * a * c
  sorry

end quadratic_real_roots_l626_62617


namespace average_first_19_natural_numbers_l626_62637

theorem average_first_19_natural_numbers : 
  (1 + 19) / 2 = 10 := 
by 
  sorry

end average_first_19_natural_numbers_l626_62637


namespace curve_crosses_itself_at_point_l626_62647

theorem curve_crosses_itself_at_point :
  ∃ t₁ t₂ : ℝ, t₁ ≠ t₂ ∧ t₁^2 - 4 = t₂^2 - 4 ∧ t₁^3 - 6 * t₁ + 4 = t₂^3 - 6 * t₂ + 4 ∧ t₁^2 - 4 = 2 ∧ t₁^3 - 6 * t₁ + 4 = 4 :=
by 
  sorry

end curve_crosses_itself_at_point_l626_62647


namespace find_k_value_l626_62691

theorem find_k_value (k : ℝ) (h : 64 / k = 4) : k = 16 :=
by
  sorry

end find_k_value_l626_62691


namespace hannah_strawberries_l626_62611

theorem hannah_strawberries (days give_away stolen remaining_strawberries x : ℕ) 
  (h1 : days = 30) 
  (h2 : give_away = 20) 
  (h3 : stolen = 30) 
  (h4 : remaining_strawberries = 100) 
  (hx : x = (remaining_strawberries + give_away + stolen) / days) : 
  x = 5 := 
by 
  -- The proof will go here
  sorry

end hannah_strawberries_l626_62611


namespace highest_value_of_a_divisible_by_8_l626_62684

theorem highest_value_of_a_divisible_by_8 :
  ∃ (a : ℕ), (0 ≤ a ∧ a ≤ 9) ∧ (8 ∣ (100 * a + 16)) ∧ 
  (∀ (b : ℕ), (0 ≤ b ∧ b ≤ 9) → 8 ∣ (100 * b + 16) → b ≤ a) :=
sorry

end highest_value_of_a_divisible_by_8_l626_62684


namespace jonathan_typing_time_l626_62692

theorem jonathan_typing_time 
(J : ℕ) 
(h_combined_rate : (1 / (J : ℝ)) + (1 / 30) + (1 / 24) = 1 / 10) : 
  J = 40 :=
by {
  sorry
}

end jonathan_typing_time_l626_62692


namespace calculate_product_l626_62654

theorem calculate_product :
  6^5 * 3^5 = 1889568 := by
  sorry

end calculate_product_l626_62654


namespace find_other_endpoint_l626_62636

theorem find_other_endpoint (x₁ y₁ x y x_mid y_mid : ℝ) 
  (h1 : x₁ = 5) (h2 : y₁ = 2) (h3 : x_mid = 3) (h4 : y_mid = 10) 
  (hx : (x₁ + x) / 2 = x_mid) (hy : (y₁ + y) / 2 = y_mid) : 
  x = 1 ∧ y = 18 := by
  sorry

end find_other_endpoint_l626_62636


namespace parabola_focus_directrix_eq_l626_62687

open Real

def distance (p : ℝ × ℝ) (l : ℝ) : ℝ := abs (p.fst - l)

def parabola_eq (focus_x focus_y l : ℝ) : Prop :=
  ∀ x y, (distance (x, y) focus_x = distance (x, y) l) ↔ y^2 = 2 * x - 1

theorem parabola_focus_directrix_eq :
  parabola_eq 1 0 0 :=
by
  sorry

end parabola_focus_directrix_eq_l626_62687


namespace gratuities_charged_l626_62682

-- Define the conditions in the problem
def total_bill : ℝ := 140
def sales_tax_rate : ℝ := 0.10
def ny_striploin_cost : ℝ := 80
def wine_cost : ℝ := 10

-- Calculate the total cost before tax and gratuities
def subtotal : ℝ := ny_striploin_cost + wine_cost

-- Calculate the taxes paid
def tax : ℝ := subtotal * sales_tax_rate

-- Calculate the total bill before gratuities
def total_before_gratuities : ℝ := subtotal + tax

-- Goal: Prove that gratuities charged is 41
theorem gratuities_charged : (total_bill - total_before_gratuities) = 41 := by sorry

end gratuities_charged_l626_62682


namespace pencil_count_l626_62641

/-- 
If there are initially 115 pencils in the drawer, and Sara adds 100 more pencils, 
then the total number of pencils in the drawer is 215.
-/
theorem pencil_count (initial_pencils added_pencils : ℕ) (h1 : initial_pencils = 115) (h2 : added_pencils = 100) : 
  initial_pencils + added_pencils = 215 := by
  sorry

end pencil_count_l626_62641


namespace repeating_decimal_to_fraction_l626_62606

theorem repeating_decimal_to_fraction : (0.5656565656 : ℚ) = 56 / 99 :=
  by
    sorry

end repeating_decimal_to_fraction_l626_62606


namespace area_units_ordered_correctly_l626_62645

def area_units :=
  ["square kilometers", "hectares", "square meters", "square decimeters", "square centimeters"]

theorem area_units_ordered_correctly :
  area_units = ["square kilometers", "hectares", "square meters", "square decimeters", "square centimeters"] :=
by
  sorry

end area_units_ordered_correctly_l626_62645


namespace max_m_value_l626_62659

theorem max_m_value (a b : ℝ) (m : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h : ∀ a b, 0 < a → 0 < b → (m / (3 * a + b) - 3 / a - 1 / b ≤ 0)) :
  m ≤ 16 :=
sorry

end max_m_value_l626_62659


namespace parabola_vertex_point_sum_l626_62683

theorem parabola_vertex_point_sum (a b c : ℚ) 
  (h1 : ∃ (a b c : ℚ), ∀ x : ℚ, (y = a * x ^ 2 + b * x + c) = (y = - (1 / 3) * (x - 5) ^ 2 + 3)) 
  (h2 : ∀ x : ℚ, ((x = 2) ∧ (y = 0)) → (0 = a * 2 ^ 2 + b * 2 + c)) :
  a + b + c = -7 / 3 := 
sorry

end parabola_vertex_point_sum_l626_62683


namespace student_age_is_24_l626_62680

-- Defining the conditions
variables (S M : ℕ)
axiom h1 : M = S + 26
axiom h2 : M + 2 = 2 * (S + 2)

-- The proof statement
theorem student_age_is_24 : S = 24 :=
by
  sorry

end student_age_is_24_l626_62680


namespace coplanar_lines_k_values_l626_62619

theorem coplanar_lines_k_values (k : ℝ) :
  (∃ t u : ℝ, 
    (1 + t = 2 + u) ∧ 
    (2 + 2 * t = 5 + k * u) ∧ 
    (3 - k * t = 6 + u)) ↔ 
  (k = -2 + Real.sqrt 6 ∨ k = -2 - Real.sqrt 6) :=
sorry

end coplanar_lines_k_values_l626_62619


namespace log_stack_total_l626_62695

theorem log_stack_total :
  let a := 5
  let l := 15
  let n := l - a + 1
  let S := n * (a + l) / 2
  S = 110 :=
sorry

end log_stack_total_l626_62695


namespace missy_tv_watching_time_l626_62622

def reality_show_count : Nat := 5
def reality_show_duration : Nat := 28
def cartoon_duration : Nat := 10

theorem missy_tv_watching_time :
  reality_show_count * reality_show_duration + cartoon_duration = 150 := by
  sorry

end missy_tv_watching_time_l626_62622


namespace total_food_correct_l626_62600

def max_food_per_guest : ℕ := 2
def min_guests : ℕ := 162
def total_food_cons : ℕ := min_guests * max_food_per_guest

theorem total_food_correct : total_food_cons = 324 := by
  sorry

end total_food_correct_l626_62600


namespace fencing_required_l626_62656

theorem fencing_required (L W : ℕ) (A : ℕ) (hL : L = 20) (hA : A = 680) (hArea : A = L * W) : 2 * W + L = 88 :=
by
  sorry

end fencing_required_l626_62656

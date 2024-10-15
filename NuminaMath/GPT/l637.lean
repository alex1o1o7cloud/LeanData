import Mathlib

namespace NUMINAMATH_GPT_distance_between_city_centers_l637_63767

theorem distance_between_city_centers :
  let distance_on_map_cm := 55
  let scale_cm_to_km := 30
  let km_to_m := 1000
  (distance_on_map_cm * scale_cm_to_km * km_to_m) = 1650000 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_city_centers_l637_63767


namespace NUMINAMATH_GPT_combined_percentage_of_students_preferring_tennis_is_39_l637_63727

def total_students_north : ℕ := 1800
def percentage_tennis_north : ℚ := 25 / 100
def total_students_south : ℕ := 3000
def percentage_tennis_south : ℚ := 50 / 100
def total_students_valley : ℕ := 800
def percentage_tennis_valley : ℚ := 30 / 100

def students_prefer_tennis_north : ℚ := total_students_north * percentage_tennis_north
def students_prefer_tennis_south : ℚ := total_students_south * percentage_tennis_south
def students_prefer_tennis_valley : ℚ := total_students_valley * percentage_tennis_valley

def total_students : ℕ := total_students_north + total_students_south + total_students_valley
def total_students_prefer_tennis : ℚ := students_prefer_tennis_north + students_prefer_tennis_south + students_prefer_tennis_valley

def percentage_students_prefer_tennis : ℚ := (total_students_prefer_tennis / total_students) * 100

theorem combined_percentage_of_students_preferring_tennis_is_39 :
  percentage_students_prefer_tennis = 39 := by
  sorry

end NUMINAMATH_GPT_combined_percentage_of_students_preferring_tennis_is_39_l637_63727


namespace NUMINAMATH_GPT_domain_of_c_is_all_real_l637_63786

theorem domain_of_c_is_all_real (a : ℝ) :
  (∀ x : ℝ, -3 * x^2 - 3 * x + a ≠ 0) ↔ a < -3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_domain_of_c_is_all_real_l637_63786


namespace NUMINAMATH_GPT_burger_meal_cost_l637_63726

-- Define the conditions
variables (B S : ℝ)
axiom cost_of_soda : S = (1 / 3) * B
axiom total_cost : B + S + 2 * (B + S) = 24

-- Prove that the cost of the burger meal is $6
theorem burger_meal_cost : B = 6 :=
by {
  -- We'll use both the axioms provided to show B equals 6
  sorry
}

end NUMINAMATH_GPT_burger_meal_cost_l637_63726


namespace NUMINAMATH_GPT_num_ordered_triples_l637_63765

-- Given constants
def b : ℕ := 2024
def constant_value : ℕ := 4096576

-- Number of ordered triples (a, b, c) meeting the conditions
theorem num_ordered_triples (h : b = 2024 ∧ constant_value = 2024 * 2024) :
  ∃ (n : ℕ), n = 10 ∧ ∀ (a c : ℕ), a * c = constant_value → a ≤ c → n = 10 :=
by
  -- Translation of the mathematical conditions into the theorem
  sorry

end NUMINAMATH_GPT_num_ordered_triples_l637_63765


namespace NUMINAMATH_GPT_jack_needs_more_money_l637_63720

variable (cost_per_pair_of_socks : ℝ := 9.50)
variable (number_of_pairs_of_socks : ℕ := 2)
variable (cost_of_soccer_shoes : ℝ := 92.00)
variable (jack_money : ℝ := 40.00)

theorem jack_needs_more_money :
  let total_cost := number_of_pairs_of_socks * cost_per_pair_of_socks + cost_of_soccer_shoes
  let money_needed := total_cost - jack_money
  money_needed = 71.00 := by
  sorry

end NUMINAMATH_GPT_jack_needs_more_money_l637_63720


namespace NUMINAMATH_GPT_cars_in_section_H_l637_63735

theorem cars_in_section_H
  (rows_G : ℕ) (cars_per_row_G : ℕ) (rows_H : ℕ)
  (cars_per_minute : ℕ) (minutes_spent : ℕ)  
  (total_cars_walked_past : ℕ) :
  rows_G = 15 →
  cars_per_row_G = 10 →
  rows_H = 20 →
  cars_per_minute = 11 →
  minutes_spent = 30 →
  total_cars_walked_past = (rows_G * cars_per_row_G) + ((cars_per_minute * minutes_spent) - (rows_G * cars_per_row_G)) →
  (total_cars_walked_past - (rows_G * cars_per_row_G)) / rows_H = 9 :=
by
  intro h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_cars_in_section_H_l637_63735


namespace NUMINAMATH_GPT_balance_balls_l637_63702

-- Define the weights of the balls as variables
variables (B R O S : ℝ)

-- Given conditions
axiom h1 : R = 2 * B
axiom h2 : O = (7 / 3) * B
axiom h3 : S = (5 / 3) * B

-- Statement to prove
theorem balance_balls :
  (5 * R + 3 * O + 4 * S) = (71 / 3) * B :=
by {
  -- The proof is omitted
  sorry
}

end NUMINAMATH_GPT_balance_balls_l637_63702


namespace NUMINAMATH_GPT_pigeonhole_divisible_l637_63754

theorem pigeonhole_divisible (n : ℕ) (a : Fin (n + 1) → ℕ) (h : ∀ i, 1 ≤ a i ∧ a i ≤ 2 * n) :
  ∃ i j, i ≠ j ∧ a i ∣ a j :=
by
  sorry

end NUMINAMATH_GPT_pigeonhole_divisible_l637_63754


namespace NUMINAMATH_GPT_intersecting_lines_sum_constant_l637_63711

theorem intersecting_lines_sum_constant
  (c d : ℝ)
  (h1 : 3 = (1 / 3) * 3 + c)
  (h2 : 3 = (1 / 3) * 3 + d) :
  c + d = 4 :=
by
  sorry

end NUMINAMATH_GPT_intersecting_lines_sum_constant_l637_63711


namespace NUMINAMATH_GPT_quartic_root_sum_l637_63737

theorem quartic_root_sum (a n l : ℝ) (h : ∃ (r1 r2 r3 r4 : ℝ), 
  r1 ≠ r2 ∧ r1 ≠ r3 ∧ r1 ≠ r4 ∧ r2 ≠ r3 ∧ r2 ≠ r4 ∧ r3 ≠ r4 ∧ 
  r1 > 0 ∧ r2 > 0 ∧ r3 > 0 ∧ r4 > 0 ∧ 
  r1 + r2 + r3 + r4 = 10 ∧
  r1 * r2 + r1 * r3 + r1 * r4 + r2 * r3 + r2 * r4 + r3 * r4 = a ∧
  r1 * r2 * r3 + r1 * r2 * r4 + r1 * r3 * r4 + r2 * r3 * r4 = n ∧
  r1 * r2 * r3 * r4 = l) : 
  a + n + l = 109 :=
sorry

end NUMINAMATH_GPT_quartic_root_sum_l637_63737


namespace NUMINAMATH_GPT_candidate_lost_by_2460_votes_l637_63759

noncomputable def total_votes : ℝ := 8199.999999999998
noncomputable def candidate_percentage : ℝ := 0.35
noncomputable def rival_percentage : ℝ := 1 - candidate_percentage
noncomputable def candidate_votes := candidate_percentage * total_votes
noncomputable def rival_votes := rival_percentage * total_votes
noncomputable def votes_lost_by := rival_votes - candidate_votes

theorem candidate_lost_by_2460_votes : votes_lost_by = 2460 := by
  sorry

end NUMINAMATH_GPT_candidate_lost_by_2460_votes_l637_63759


namespace NUMINAMATH_GPT_cosine_of_A_l637_63760

theorem cosine_of_A (a b : ℝ) (A B : ℝ) (h1 : b = (5 / 8) * a) (h2 : A = 2 * B) :
  Real.cos A = 7 / 25 :=
by
  sorry

end NUMINAMATH_GPT_cosine_of_A_l637_63760


namespace NUMINAMATH_GPT_repaired_shoes_lifespan_l637_63715

-- Definitions of given conditions
def cost_repair : Float := 11.50
def cost_new : Float := 28.00
def lifespan_new : Float := 2.0
def percentage_increase : Float := 21.73913043478261 / 100

-- Cost per year of new shoes
def cost_per_year_new : Float := cost_new / lifespan_new

-- Cost per year of repaired shoes
def cost_per_year_repair (T : Float) : Float := cost_repair / T

-- Theorem statement (goal)
theorem repaired_shoes_lifespan (T : Float) (h : cost_per_year_new = cost_per_year_repair T * (1 + percentage_increase)) : T = 0.6745 :=
by
  sorry

end NUMINAMATH_GPT_repaired_shoes_lifespan_l637_63715


namespace NUMINAMATH_GPT_largest_and_smallest_A_l637_63751

noncomputable def is_coprime_with_12 (n : ℕ) : Prop := 
  Nat.gcd n 12 = 1

def problem_statement (A_max A_min : ℕ) : Prop :=
  ∃ B : ℕ, B > 44444444 ∧ is_coprime_with_12 B ∧
  (A_max = 9 * 10^7 + (B - 9) / 10) ∧
  (A_min = 1 * 10^7 + (B - 1) / 10)

theorem largest_and_smallest_A :
  problem_statement 99999998 14444446 := sorry

end NUMINAMATH_GPT_largest_and_smallest_A_l637_63751


namespace NUMINAMATH_GPT_complement_of_A_in_U_l637_63783

-- Definitions
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {1, 2, 4, 5}

-- Proof statement
theorem complement_of_A_in_U : (U \ A) = {3, 6, 7} := by
  sorry

end NUMINAMATH_GPT_complement_of_A_in_U_l637_63783


namespace NUMINAMATH_GPT_birds_not_hawks_warbler_kingfisher_l637_63717

variables (B : ℝ)
variables (hawks paddyfield_warblers kingfishers : ℝ)

-- Conditions
def condition1 := hawks = 0.30 * B
def condition2 := paddyfield_warblers = 0.40 * (B - hawks)
def condition3 := kingfishers = 0.25 * paddyfield_warblers

-- Question: Prove the percentage of birds that are not hawks, paddyfield-warblers, or kingfishers is 35%
theorem birds_not_hawks_warbler_kingfisher (B hawks paddyfield_warblers kingfishers : ℝ) 
 (h1 : hawks = 0.30 * B) 
 (h2 : paddyfield_warblers = 0.40 * (B - hawks)) 
 (h3 : kingfishers = 0.25 * paddyfield_warblers) : 
 (1 - (hawks + paddyfield_warblers + kingfishers) / B) * 100 = 35 :=
by
  sorry

end NUMINAMATH_GPT_birds_not_hawks_warbler_kingfisher_l637_63717


namespace NUMINAMATH_GPT_find_genuine_coin_in_three_weighings_l637_63721

theorem find_genuine_coin_in_three_weighings (coins : Fin 15 → ℝ)
  (even_number_of_counterfeit : ∃ n : ℕ, 2 * n < 15 ∧ (∀ i, coins i = 1) ∨ (∃ j, coins j = 0.5)) : 
  ∃ i, coins i = 1 :=
by sorry

end NUMINAMATH_GPT_find_genuine_coin_in_three_weighings_l637_63721


namespace NUMINAMATH_GPT_james_pays_per_episode_l637_63777

-- Conditions
def minor_characters : ℕ := 4
def major_characters : ℕ := 5
def pay_per_minor_character : ℕ := 15000
def multiplier_major_payment : ℕ := 3

-- Theorems and Definitions needed
def pay_per_major_character : ℕ := pay_per_minor_character * multiplier_major_payment
def total_pay_minor : ℕ := minor_characters * pay_per_minor_character
def total_pay_major : ℕ := major_characters * pay_per_major_character
def total_pay_per_episode : ℕ := total_pay_minor + total_pay_major

-- Main statement to prove
theorem james_pays_per_episode : total_pay_per_episode = 285000 := by
  sorry

end NUMINAMATH_GPT_james_pays_per_episode_l637_63777


namespace NUMINAMATH_GPT_noemi_lost_on_roulette_l637_63709

theorem noemi_lost_on_roulette (initial_purse := 1700) (final_purse := 800) (loss_on_blackjack := 500) :
  (initial_purse - final_purse) - loss_on_blackjack = 400 := by
  sorry

end NUMINAMATH_GPT_noemi_lost_on_roulette_l637_63709


namespace NUMINAMATH_GPT_findC_coordinates_l637_63790

-- Points in the Cartesian coordinate system
structure Point where
  x : ℝ
  y : ℝ

-- Defining points A, B, and stating that point C lies on the positive x-axis
def A : Point := {x := -4, y := -2}
def B : Point := {x := 0, y := -2}
def C (cx : ℝ) : Point := {x := cx, y := 0}

-- The condition that the triangle OBC is similar to triangle ABO
def isSimilar (A B O : Point) (C : Point) : Prop :=
  let AB := (A.x - B.x)^2 + (A.y - B.y)^2
  let OB := (B.x - O.x)^2 + (B.y - O.y)^2
  let OC := (C.x - O.x)^2 + (C.y - O.y)^2
  AB / OB = OB / OC

theorem findC_coordinates :
  ∃ (cx : ℝ), (C cx = {x := 1, y := 0} ∨ C cx = {x := 4, y := 0}) ∧
  isSimilar A B {x := 0, y := 0} (C cx) :=
by
  sorry

end NUMINAMATH_GPT_findC_coordinates_l637_63790


namespace NUMINAMATH_GPT_number_of_roots_eq_seven_l637_63725

noncomputable def problem_function (x : ℝ) : ℝ :=
  (21 * x - 11 + (Real.sin x) / 100) * Real.sin (6 * Real.arcsin x) * Real.sqrt ((Real.pi - 6 * x) * (Real.pi + x))

theorem number_of_roots_eq_seven :
  (∃ xs : List ℝ, (∀ x ∈ xs, problem_function x = 0) ∧ (∀ x ∈ xs, -1 ≤ x ∧ x ≤ 1) ∧ xs.length = 7) :=
sorry

end NUMINAMATH_GPT_number_of_roots_eq_seven_l637_63725


namespace NUMINAMATH_GPT_equation_solution_unique_or_not_l637_63724

theorem equation_solution_unique_or_not (a b : ℝ) :
  (∃ x : ℝ, x ≠ 2 ∧ x ≠ 3 ∧ (x - a) / (x - 2) + (x - b) / (x - 3) = 2) ↔ 
  (a = 2 ∧ b = 3) ∨ (a + b ≠ 5 ∧ a ≠ 2 ∧ b ≠ 3) :=
by
  sorry

end NUMINAMATH_GPT_equation_solution_unique_or_not_l637_63724


namespace NUMINAMATH_GPT_ball_color_arrangement_l637_63722

-- Definitions for the conditions
variable (balls_in_red_box balls_in_white_box balls_in_yellow_box : Nat)
variable (red_balls white_balls yellow_balls : Nat)

-- Conditions as assumptions
axiom more_balls_in_yellow_box_than_yellow_balls : balls_in_yellow_box > yellow_balls
axiom different_balls_in_red_box_than_white_balls : balls_in_red_box ≠ white_balls
axiom fewer_white_balls_than_balls_in_white_box : white_balls < balls_in_white_box

-- The main theorem to prove
theorem ball_color_arrangement
  (more_balls_in_yellow_box_than_yellow_balls : balls_in_yellow_box > yellow_balls)
  (different_balls_in_red_box_than_white_balls : balls_in_red_box ≠ white_balls)
  (fewer_white_balls_than_balls_in_white_box : white_balls < balls_in_white_box) :
  (balls_in_red_box, balls_in_white_box, balls_in_yellow_box) = (yellow_balls, red_balls, white_balls) :=
sorry

end NUMINAMATH_GPT_ball_color_arrangement_l637_63722


namespace NUMINAMATH_GPT_hyperbola_equation_l637_63779

noncomputable def distance_between_vertices : ℝ := 8
noncomputable def eccentricity : ℝ := 5 / 4

theorem hyperbola_equation :
  ∃ a b c : ℝ, 2 * a = distance_between_vertices ∧ 
               c = a * eccentricity ∧ 
               b^2 = c^2 - a^2 ∧ 
               (a = 4 ∧ c = 5 ∧ b^2 = 9) ∧ 
               ∀ x y : ℝ, (x^2 / (a:ℝ)^2) - (y^2 / (b:ℝ)^2) = 1 :=
by 
  sorry

end NUMINAMATH_GPT_hyperbola_equation_l637_63779


namespace NUMINAMATH_GPT_parabola_vertex_l637_63761

theorem parabola_vertex (x y : ℝ) : 
  y^2 + 10 * y + 3 * x + 9 = 0 → 
  (∃ v_x v_y, v_x = 16/3 ∧ v_y = -5 ∧ ∀ (y' : ℝ), (x, y) = (v_x, v_y) ↔ (x, y) = (-1 / 3 * ((y' + 5)^2 - 16), y')) :=
by
  sorry

end NUMINAMATH_GPT_parabola_vertex_l637_63761


namespace NUMINAMATH_GPT_regular_polygon_perimeter_l637_63743

theorem regular_polygon_perimeter (side_length : ℝ) (exterior_angle : ℝ) (n : ℕ)
  (h1 : side_length = 7)  (h2 : exterior_angle = 90) 
  (h3 : exterior_angle = 360 / n) : 
  (side_length * n = 28) := by
  sorry

end NUMINAMATH_GPT_regular_polygon_perimeter_l637_63743


namespace NUMINAMATH_GPT_b_investment_less_c_l637_63704

theorem b_investment_less_c (A B C : ℕ) (y : ℕ) (total_investment : ℕ) (profit : ℕ) (A_share : ℕ)
    (h1 : A + B + C = total_investment)
    (h2 : A = B + 6000)
    (h3 : C = B + y)
    (h4 : profit = 8640)
    (h5 : A_share = 3168) :
    y = 3000 :=
by
  sorry

end NUMINAMATH_GPT_b_investment_less_c_l637_63704


namespace NUMINAMATH_GPT_petr_receives_1000000_l637_63799

def initial_investment_vp := 200000
def initial_investment_pg := 350000
def third_share_value := 1100000
def total_company_value := 3 * third_share_value

theorem petr_receives_1000000 :
  initial_investment_vp = 200000 →
  initial_investment_pg = 350000 →
  third_share_value = 1100000 →
  total_company_value = 3300000 →
  ∃ (share_pg : ℕ), share_pg = 1000000 :=
by
  intros h_vp h_pg h_as h_total
  let x := initial_investment_vp * 1650000
  let y := initial_investment_pg * 1650000
  -- Skipping calculations
  sorry

end NUMINAMATH_GPT_petr_receives_1000000_l637_63799


namespace NUMINAMATH_GPT_smallest_solution_l637_63775

noncomputable def equation (x : ℝ) := x^4 - 40 * x^2 + 400

theorem smallest_solution : ∃ x : ℝ, equation x = 0 ∧ ∀ y : ℝ, equation y = 0 → -2 * Real.sqrt 5 ≤ y :=
by
  sorry

end NUMINAMATH_GPT_smallest_solution_l637_63775


namespace NUMINAMATH_GPT_calculate_expression_l637_63723

theorem calculate_expression : (3^3 * 4^3)^2 = 2985984 := by
  sorry

end NUMINAMATH_GPT_calculate_expression_l637_63723


namespace NUMINAMATH_GPT_johns_average_speed_l637_63707

theorem johns_average_speed :
  let distance1 := 20
  let speed1 := 10
  let distance2 := 30
  let speed2 := 20
  let total_distance := distance1 + distance2
  let time1 := distance1 / speed1
  let time2 := distance2 / speed2
  let total_time := time1 + time2
  let average_speed := total_distance / total_time
  average_speed = 14.29 :=
by
  sorry

end NUMINAMATH_GPT_johns_average_speed_l637_63707


namespace NUMINAMATH_GPT_bread_slices_remaining_l637_63798

-- Conditions
def total_slices : ℕ := 12
def fraction_eaten_for_breakfast : ℕ := total_slices / 3
def slices_used_for_lunch : ℕ := 2

-- Mathematically Equivalent Proof Problem
theorem bread_slices_remaining : total_slices - fraction_eaten_for_breakfast - slices_used_for_lunch = 6 :=
by
  sorry

end NUMINAMATH_GPT_bread_slices_remaining_l637_63798


namespace NUMINAMATH_GPT_least_common_duration_l637_63782

theorem least_common_duration 
    (P Q R : ℝ) 
    (x : ℝ)
    (T : ℝ)
    (h1 : P / Q = 7 / 5)
    (h2 : Q / R = 5 / 3)
    (h3 : 8 * P / (6 * Q) = 7 / 10)
    (h4 : (6 * 10) * R / (30 * T) = 1)
    : T = 6 :=
by
  sorry

end NUMINAMATH_GPT_least_common_duration_l637_63782


namespace NUMINAMATH_GPT_slower_speed_l637_63795

theorem slower_speed (x : ℝ) :
  (50 / x = 70 / 14) → x = 10 := by
  sorry

end NUMINAMATH_GPT_slower_speed_l637_63795


namespace NUMINAMATH_GPT_tangent_line_ellipse_l637_63794

theorem tangent_line_ellipse (x y : ℝ) (h : 2^2 / 8 + 1^2 / 2 = 1) :
    x / 4 + y / 2 = 1 := 
  sorry

end NUMINAMATH_GPT_tangent_line_ellipse_l637_63794


namespace NUMINAMATH_GPT_bargain_range_l637_63784

theorem bargain_range (cost_price lowest_cp highest_cp : ℝ)
  (h_lowest : lowest_cp = 50)
  (h_highest : highest_cp = 200 / 3)
  (h_marked_at : cost_price = 100)
  (h_lowest_markup : lowest_cp * 2 = cost_price)
  (h_highest_markup : highest_cp * 1.5 = cost_price)
  (profit_margin : ∀ (cp : ℝ), (cp * 1.2 ≥ cp)) : 
  (60 ≤ cost_price * 1.2 ∧ cost_price * 1.2 ≤ 80) :=
by
  sorry

end NUMINAMATH_GPT_bargain_range_l637_63784


namespace NUMINAMATH_GPT_prime_factors_of_n_l637_63789

def n : ℕ := 400000001

def is_prime (p: ℕ) : Prop := Nat.Prime p

theorem prime_factors_of_n (p q : ℕ) (hp : is_prime p) (hq : is_prime q) (h : n = p * q) : 
  (p = 19801 ∧ q = 20201) ∨ (p = 20201 ∧ q = 19801) :=
by
  sorry

end NUMINAMATH_GPT_prime_factors_of_n_l637_63789


namespace NUMINAMATH_GPT_tenth_student_solved_six_l637_63741

theorem tenth_student_solved_six : 
  ∀ (n : ℕ), 
    (∀ (i : ℕ) (j : ℕ), 1 ≤ i ∧ i ≤ 10 → 1 ≤ j ∧ j ≤ n → (∀ k : ℕ, k ≤ n → ∃ s : ℕ, s = 7)) → 
    (∀ (i : ℕ), 1 ≤ i ∧ i ≤ 9 → ∃ p : ℕ, p = 4) → ∃ m : ℕ, m = 6 := 
by
  sorry

end NUMINAMATH_GPT_tenth_student_solved_six_l637_63741


namespace NUMINAMATH_GPT_milo_running_distance_l637_63791

theorem milo_running_distance : 
  ∀ (cory_speed milo_skate_speed milo_run_speed time miles_run : ℕ),
  cory_speed = 12 →
  milo_skate_speed = cory_speed / 2 →
  milo_run_speed = milo_skate_speed / 2 →
  time = 2 →
  miles_run = milo_run_speed * time →
  miles_run = 6 :=
by 
  intros cory_speed milo_skate_speed milo_run_speed time miles_run hcory hmilo_skate hmilo_run htime hrun 
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_milo_running_distance_l637_63791


namespace NUMINAMATH_GPT_circle_centered_at_8_neg3_passing_through_5_1_circle_passing_through_ABC_l637_63748

-- Circle 1 with center (8, -3) and passing through point (5, 1)
theorem circle_centered_at_8_neg3_passing_through_5_1 :
  ∃ r : ℝ, (r = 5) ∧ ((x - 8: ℝ)^2 + (y + 3)^2 = r^2) := by
  sorry

-- Circle passing through points A(-1, 5), B(5, 5), and C(6, -2)
theorem circle_passing_through_ABC :
  ∃ D E F : ℝ, (D = -4) ∧ (E = -2) ∧ (F = -20) ∧
    ( ∀ (x : ℝ) (y : ℝ), (x = -1 ∧ y = 5) 
      ∨ (x = 5 ∧ y = 5) 
      ∨ (x = 6 ∧ y = -2) 
      → (x^2 + y^2 + D*x + E*y + F = 0)) := by
  sorry

end NUMINAMATH_GPT_circle_centered_at_8_neg3_passing_through_5_1_circle_passing_through_ABC_l637_63748


namespace NUMINAMATH_GPT_message_channels_encryption_l637_63739

theorem message_channels_encryption :
  ∃ (assign_key : Fin 105 → Fin 105 → Fin 100),
  ∀ (u v w x : Fin 105), 
  u ≠ v → u ≠ w → u ≠ x → v ≠ w → v ≠ x → w ≠ x →
  (assign_key u v = assign_key u w ∧ assign_key u v = assign_key u x ∧ 
   assign_key u v = assign_key v w ∧ assign_key u v = assign_key v x ∧ 
   assign_key u v = assign_key w x) → False :=
by
  sorry

end NUMINAMATH_GPT_message_channels_encryption_l637_63739


namespace NUMINAMATH_GPT_eval_nabla_l637_63736

namespace MathProblem

-- Definition of the operation
def nabla (a b : ℕ) : ℕ :=
  3 + b ^ a

-- Theorem statement
theorem eval_nabla : nabla (nabla 2 3) 4 = 16777219 :=
by
  sorry

end MathProblem

end NUMINAMATH_GPT_eval_nabla_l637_63736


namespace NUMINAMATH_GPT_inequality_positive_numbers_l637_63729

theorem inequality_positive_numbers (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) : 
  (a + b + c) * (1/a + 1/b + 1/c) ≥ 9 := 
sorry

end NUMINAMATH_GPT_inequality_positive_numbers_l637_63729


namespace NUMINAMATH_GPT_two_digit_integer_one_less_than_lcm_of_3_4_7_l637_63732

theorem two_digit_integer_one_less_than_lcm_of_3_4_7 :
  ∃ n : ℕ, 10 ≤ n ∧ n < 100 ∧ (n + 1) % (Nat.lcm (Nat.lcm 3 4) 7) = 0 ∧ n = 83 := by
  sorry

end NUMINAMATH_GPT_two_digit_integer_one_less_than_lcm_of_3_4_7_l637_63732


namespace NUMINAMATH_GPT_slopes_product_of_tangents_l637_63740

theorem slopes_product_of_tangents 
  (x₀ y₀ : ℝ) 
  (h_hyperbola : (2 * x₀^2) / 3 - y₀^2 / 6 = 1) 
  (h_outside_circle : x₀^2 + y₀^2 > 2) :
  ∃ (k₁ k₂ : ℝ), 
    k₁ * k₂ = 4 ∧ 
    (y₀ - k₁ * x₀)^2 + k₁^2 = 2 ∧ 
    (y₀ - k₂ * x₀)^2 + k₂^2 = 2 :=
by {
  -- this proof will use the properties of tangents to a circle and the constraints given
  -- we don't need to implement it now, but we aim to show the correct relationship
  sorry
}

end NUMINAMATH_GPT_slopes_product_of_tangents_l637_63740


namespace NUMINAMATH_GPT_dubblefud_red_balls_zero_l637_63787

theorem dubblefud_red_balls_zero
  (R B G : ℕ)
  (H1 : 2^R * 4^B * 5^G = 16000)
  (H2 : B = G) : R = 0 :=
sorry

end NUMINAMATH_GPT_dubblefud_red_balls_zero_l637_63787


namespace NUMINAMATH_GPT_unsolved_problems_exist_l637_63710

noncomputable def main_theorem: Prop :=
  ∃ (P : Prop), ¬(P = true) ∧ ¬(P = false)

theorem unsolved_problems_exist : main_theorem :=
sorry

end NUMINAMATH_GPT_unsolved_problems_exist_l637_63710


namespace NUMINAMATH_GPT_maria_total_distance_l637_63797

-- Definitions
def total_distance (D : ℝ) : Prop :=
  let d1 := D/2   -- Distance traveled before first stop
  let r1 := D - d1 -- Distance remaining after first stop
  let d2 := r1/4  -- Distance traveled before second stop
  let r2 := r1 - d2 -- Distance remaining after second stop
  let d3 := r2/3  -- Distance traveled before third stop
  let r3 := r2 - d3 -- Distance remaining after third stop
  r3 = 270 -- Remaining distance after third stop equals 270 miles

-- Theorem statement
theorem maria_total_distance : ∃ D : ℝ, total_distance D ∧ D = 1080 :=
sorry

end NUMINAMATH_GPT_maria_total_distance_l637_63797


namespace NUMINAMATH_GPT_Will_worked_on_Tuesday_l637_63757

variable (HourlyWage MondayHours TotalEarnings : ℝ)

-- Given conditions
def Wage : ℝ := 8
def Monday_worked_hours : ℝ := 8
def Total_two_days_earnings : ℝ := 80

theorem Will_worked_on_Tuesday (HourlyWage_eq : HourlyWage = Wage)
  (MondayHours_eq : MondayHours = Monday_worked_hours)
  (TotalEarnings_eq : TotalEarnings = Total_two_days_earnings) :
  let MondayEarnings := MondayHours * HourlyWage
  let TuesdayEarnings := TotalEarnings - MondayEarnings
  let TuesdayHours := TuesdayEarnings / HourlyWage
  TuesdayHours = 2 :=
by
  sorry

end NUMINAMATH_GPT_Will_worked_on_Tuesday_l637_63757


namespace NUMINAMATH_GPT_cricket_team_members_l637_63719

theorem cricket_team_members (n : ℕ) (captain_age wicket_keeper_age average_whole_age average_remaining_age : ℕ) :
  captain_age = 24 →
  wicket_keeper_age = 31 →
  average_whole_age = 23 →
  average_remaining_age = 22 →
  n * average_whole_age - captain_age - wicket_keeper_age = (n - 2) * average_remaining_age →
  n = 11 :=
by
  intros h_cap_age h_wk_age h_avg_whole h_avg_remain h_eq
  sorry

end NUMINAMATH_GPT_cricket_team_members_l637_63719


namespace NUMINAMATH_GPT_domino_trick_l637_63766

theorem domino_trick (x y : ℕ) (h1 : x ≤ 6) (h2 : y ≤ 6)
  (h3 : 10 * x + y + 30 = 62) : x = 3 ∧ y = 2 :=
by
  sorry

end NUMINAMATH_GPT_domino_trick_l637_63766


namespace NUMINAMATH_GPT_scalene_triangle_minimum_altitude_l637_63770

theorem scalene_triangle_minimum_altitude (a b c : ℕ) (h : ℕ) 
  (h₁ : a ≠ b ∧ b ≠ c ∧ c ≠ a) -- scalene condition
  (h₂ : ∃ k : ℕ, ∃ m : ℕ, k * m = a ∧ m = 6) -- first altitude condition
  (h₃ : ∃ k : ℕ, ∃ n : ℕ, k * n = b ∧ n = 8) -- second altitude condition
  (h₄ : c = (7 : ℕ) * b / (3 : ℕ)) -- third side condition given inequalities and area relations
  : h = 2 := 
sorry

end NUMINAMATH_GPT_scalene_triangle_minimum_altitude_l637_63770


namespace NUMINAMATH_GPT_least_positive_integer_mod_cond_l637_63703

theorem least_positive_integer_mod_cond (N : ℕ) :
  (N % 6 = 5) ∧ 
  (N % 7 = 6) ∧ 
  (N % 8 = 7) ∧ 
  (N % 9 = 8) ∧ 
  (N % 10 = 9) ∧ 
  (N % 11 = 10) →
  N = 27719 :=
by
  sorry

end NUMINAMATH_GPT_least_positive_integer_mod_cond_l637_63703


namespace NUMINAMATH_GPT_first_method_of_exhaustion_l637_63793

-- Define the names
inductive Names where
  | ZuChongzhi
  | LiuHui
  | ZhangHeng
  | YangHui
  deriving DecidableEq

-- Statement of the problem
def method_of_exhaustion_author : Names :=
  Names.LiuHui

-- Main theorem to state the result
theorem first_method_of_exhaustion : method_of_exhaustion_author = Names.LiuHui :=
by 
  sorry

end NUMINAMATH_GPT_first_method_of_exhaustion_l637_63793


namespace NUMINAMATH_GPT_probability_first_genuine_on_third_test_l637_63778

noncomputable def probability_of_genuine : ℚ := 3 / 4
noncomputable def probability_of_defective : ℚ := 1 / 4
noncomputable def probability_X_eq_3 := probability_of_defective * probability_of_defective * probability_of_genuine

theorem probability_first_genuine_on_third_test :
  probability_X_eq_3 = 3 / 64 :=
by
  sorry

end NUMINAMATH_GPT_probability_first_genuine_on_third_test_l637_63778


namespace NUMINAMATH_GPT_max_profit_at_300_l637_63728

-- Define the cost and revenue functions and total profit function

noncomputable def cost (x : ℝ) : ℝ := 20000 + 100 * x

noncomputable def revenue (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 390 then -x^3 / 900 + 400 * x else 90090

noncomputable def profit (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 390 then -x^3 / 900 + 300 * x - 20000 else -100 * x + 70090

-- The Lean statement for proving maximum profit occurs at x = 300
theorem max_profit_at_300 : ∀ x : ℝ, profit x ≤ profit 300 :=
sorry

end NUMINAMATH_GPT_max_profit_at_300_l637_63728


namespace NUMINAMATH_GPT_simplify_expression_l637_63769

theorem simplify_expression :
  ( (2^2 - 1) * (3^2 - 1) * (4^2 - 1) * (5^2 - 1) ) / ( (2 * 3) * (3 * 4) * (4 * 5) * (5 * 6) ) = 1 / 5 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l637_63769


namespace NUMINAMATH_GPT_height_of_boxes_l637_63733

-- Conditions
def total_volume : ℝ := 1.08 * 10^6
def cost_per_box : ℝ := 0.2
def total_monthly_cost : ℝ := 120

-- Target height of the boxes
def target_height : ℝ := 12.2

-- Problem: Prove that the height of each box is 12.2 inches
theorem height_of_boxes : 
  (total_monthly_cost / cost_per_box) * ((total_volume / (total_monthly_cost / cost_per_box))^(1/3)) = target_height := 
sorry

end NUMINAMATH_GPT_height_of_boxes_l637_63733


namespace NUMINAMATH_GPT_price_reduction_l637_63771

theorem price_reduction (x : ℝ) :
  (20 + 2 * x) * (40 - x) = 1200 → x = 20 :=
by
  sorry

end NUMINAMATH_GPT_price_reduction_l637_63771


namespace NUMINAMATH_GPT_tangent_line_on_x_axis_l637_63758

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 + a * x + 1/4

theorem tangent_line_on_x_axis (x0 a : ℝ) (h1: f x0 a = 0) (h2: (3 * x0^2 + a) = 0) : a = -3/4 :=
by sorry

end NUMINAMATH_GPT_tangent_line_on_x_axis_l637_63758


namespace NUMINAMATH_GPT_time_to_cross_signal_post_l637_63753

-- Definition of the conditions
def length_of_train : ℝ := 600  -- in meters
def time_to_cross_bridge : ℝ := 8  -- in minutes
def length_of_bridge : ℝ := 7200  -- in meters

-- Equivalent statement
theorem time_to_cross_signal_post (constant_speed : ℝ) (t : ℝ) 
  (h1 : constant_speed * t = length_of_train) 
  (h2 : constant_speed * time_to_cross_bridge = length_of_train + length_of_bridge) : 
  t * 60 = 36.9 := 
sorry

end NUMINAMATH_GPT_time_to_cross_signal_post_l637_63753


namespace NUMINAMATH_GPT_side_length_of_inscribed_square_l637_63749

theorem side_length_of_inscribed_square
  (S1 S2 S3 : ℝ)
  (hS1 : S1 = 1) (hS2 : S2 = 3) (hS3 : S3 = 1) :
  ∃ (x : ℝ), S1 = 1 ∧ S2 = 3 ∧ S3 = 1 ∧ x = 2 := 
by
  sorry

end NUMINAMATH_GPT_side_length_of_inscribed_square_l637_63749


namespace NUMINAMATH_GPT_problem1_div_expr_problem2_div_expr_l637_63772

-- Problem 1
theorem problem1_div_expr : (1 / 30) / ((2 / 3) - (1 / 10) + (1 / 6) - (2 / 5)) = 1 / 10 :=
by 
  -- sorry is added to mark the spot for the proof
  sorry

-- Problem 2
theorem problem2_div_expr : (-1 / 20) / (-(1 / 4) - (2 / 5) + (9 / 10) - (3 / 2)) = 1 / 25 :=
by 
  -- sorry is added to mark the spot for the proof
  sorry

end NUMINAMATH_GPT_problem1_div_expr_problem2_div_expr_l637_63772


namespace NUMINAMATH_GPT_problem_statement_l637_63755

-- Conditions
def p (a : ℝ) : Prop := a > 0 ∧ a ≠ 1 ∧ ∀ x : ℝ, a ^ x > 0
def q (x : ℝ) : Prop := x > 0 ∧ x ≠ 1 ∧ (Real.log 2 / Real.log x + Real.log x / Real.log 2 ≥ 2)

-- Theorem statement
theorem problem_statement (a x : ℝ) : ¬p a ∨ ¬q x :=
by sorry

end NUMINAMATH_GPT_problem_statement_l637_63755


namespace NUMINAMATH_GPT_amount_spent_per_sibling_l637_63788

-- Definitions and conditions
def total_spent := 150
def amount_per_parent := 30
def num_parents := 2
def num_siblings := 3

-- Claim
theorem amount_spent_per_sibling :
  (total_spent - (amount_per_parent * num_parents)) / num_siblings = 30 :=
by
  sorry

end NUMINAMATH_GPT_amount_spent_per_sibling_l637_63788


namespace NUMINAMATH_GPT_multiples_of_5_with_units_digit_0_l637_63764

theorem multiples_of_5_with_units_digit_0 (h1 : ∀ n : ℕ, n % 5 = 0 → (n % 10 = 0 ∨ n % 10 = 5))
  (h2 : ∀ m : ℕ, m < 200 → m % 5 = 0) :
  ∃ k : ℕ, k = 19 ∧ (∀ x : ℕ, (x < 200) ∧ (x % 5 = 0) → (x % 10 = 0) → k = (k - 1) + 1) := sorry

end NUMINAMATH_GPT_multiples_of_5_with_units_digit_0_l637_63764


namespace NUMINAMATH_GPT_expr_min_value_expr_min_at_15_l637_63705

theorem expr_min_value (a x : ℝ) (h1 : 0 ≤ a) (h2 : a ≤ 15) (h3 : a ≤ x) (h4 : x ≤ 15) :
  (|x - a| + |x - 15| + |x - (a + 15)|) = 30 - x := 
sorry

theorem expr_min_at_15 (a : ℝ) (h : 0 ≤ a ∧ a ≤ 15) : 
  (|15 - a| + |15 - 15| + |15 - (a + 15)|) = 15 := 
sorry

end NUMINAMATH_GPT_expr_min_value_expr_min_at_15_l637_63705


namespace NUMINAMATH_GPT_square_free_even_less_than_200_count_l637_63774

def is_square_free (n : ℕ) : Prop :=
  ∀ m : ℕ, m^2 ∣ n → m = 1

def is_multiple_of_2 (n : ℕ) : Prop := n % 2 = 0

theorem square_free_even_less_than_200_count : ∃ (count : ℕ), count = 38 ∧ (∀ n : ℕ, n < 200 ∧ is_multiple_of_2 n ∧ is_square_free n → count = 38) :=
by
  sorry

end NUMINAMATH_GPT_square_free_even_less_than_200_count_l637_63774


namespace NUMINAMATH_GPT_frogs_seen_in_pond_l637_63750

-- Definitions from the problem conditions
def initial_frogs_on_lily_pads : ℕ := 5
def frogs_on_logs : ℕ := 3
def baby_frogs_on_rock : ℕ := 2 * 12  -- Two dozen

-- The statement of the proof
theorem frogs_seen_in_pond : initial_frogs_on_lily_pads + frogs_on_logs + baby_frogs_on_rock = 32 :=
by sorry

end NUMINAMATH_GPT_frogs_seen_in_pond_l637_63750


namespace NUMINAMATH_GPT_chip_exits_from_A2_l637_63718

noncomputable def chip_exit_cell (grid_size : ℕ) (initial_cell : ℕ × ℕ) (move_direction : ℕ × ℕ → ℕ × ℕ) : ℕ × ℕ :=
(1, 2) -- A2; we assume the implementation of function movement follows the solution as described

theorem chip_exits_from_A2 :
  chip_exit_cell 4 (3, 2) move_direction = (1, 2) :=
sorry  -- Proof omitted

end NUMINAMATH_GPT_chip_exits_from_A2_l637_63718


namespace NUMINAMATH_GPT_correct_sampling_methods_l637_63731

-- Definitions for different sampling methods
inductive SamplingMethod
  | Systematic
  | Stratified
  | SimpleRandom

-- Conditions from the problem
def situation1 (students_selected_per_class : Nat) : Prop :=
  students_selected_per_class = 2

def situation2 (students_above_110 : Nat) (students_between_90_and_100 : Nat) (students_below_90 : Nat) : Prop :=
  students_above_110 = 10 ∧ students_between_90_and_100 = 40 ∧ students_below_90 = 12

def situation3 (tracks_arranged_for_students : Nat) : Prop :=
  tracks_arranged_for_students = 6

-- Theorem
theorem correct_sampling_methods :
  ∀ (students_selected_per_class students_above_110 students_between_90_and_100 students_below_90 tracks_arranged_for_students: Nat),
  situation1 students_selected_per_class →
  situation2 students_above_110 students_between_90_and_100 students_below_90 →
  situation3 tracks_arranged_for_students →
  (SamplingMethod.Systematic, SamplingMethod.Stratified, SamplingMethod.SimpleRandom) = (SamplingMethod.Systematic, SamplingMethod.Stratified, SamplingMethod.SimpleRandom) :=
by
  intros
  rfl

end NUMINAMATH_GPT_correct_sampling_methods_l637_63731


namespace NUMINAMATH_GPT_lucy_cleans_aquariums_l637_63785

theorem lucy_cleans_aquariums :
  (∃ rate : ℕ, rate = 2 / 3) →
  (∃ hours : ℕ, hours = 24) →
  (∃ increments : ℕ, increments = 24 / 3) →
  (∃ aquariums : ℕ, aquariums = (2 * (24 / 3))) →
  aquariums = 16 :=
by
  sorry

end NUMINAMATH_GPT_lucy_cleans_aquariums_l637_63785


namespace NUMINAMATH_GPT_second_flower_shop_groups_l637_63738

theorem second_flower_shop_groups (n : ℕ) (h1 : n ≠ 0) (h2 : n ≠ 9) (h3 : Nat.lcm 9 n = 171) : n = 19 := 
by
  sorry

end NUMINAMATH_GPT_second_flower_shop_groups_l637_63738


namespace NUMINAMATH_GPT_election_majority_l637_63763

theorem election_majority
  (total_votes : ℕ)
  (winning_percent : ℝ)
  (other_percent : ℝ)
  (votes_cast : total_votes = 700)
  (winning_share : winning_percent = 0.84)
  (other_share : other_percent = 0.16) :
  ∃ majority : ℕ, majority = 476 := by
  sorry

end NUMINAMATH_GPT_election_majority_l637_63763


namespace NUMINAMATH_GPT_probability_between_C_and_D_l637_63756

theorem probability_between_C_and_D :
  ∀ (A B C D : ℝ) (AB AD BC : ℝ),
    AB = 3 * AD ∧ AB = 6 * BC ∧ D - A = AD ∧ C - A = AD + BC ∧ B - A = AB →
    (C < D) →
    ∃ p : ℝ, p = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_probability_between_C_and_D_l637_63756


namespace NUMINAMATH_GPT_acid_solution_l637_63747

theorem acid_solution (n y : ℝ) (h : n > 30) (h1 : y = 15 * n / (n - 15)) :
  (n / 100) * n = ((n - 15) / 100) * (n + y) :=
by
  sorry

end NUMINAMATH_GPT_acid_solution_l637_63747


namespace NUMINAMATH_GPT_original_price_l637_63730

theorem original_price (P : ℝ) (profit : ℝ) (profit_percentage : ℝ)
  (h1 : profit = 675) (h2 : profit_percentage = 0.35) :
  P = 1928.57 :=
by
  -- The proof is skipped using sorry
  sorry

end NUMINAMATH_GPT_original_price_l637_63730


namespace NUMINAMATH_GPT_find_x_minus_y_l637_63708

theorem find_x_minus_y (x y : ℝ) (hx : |x| = 4) (hy : |y| = 2) (hxy : x * y < 0) : x - y = 6 ∨ x - y = -6 :=
by sorry

end NUMINAMATH_GPT_find_x_minus_y_l637_63708


namespace NUMINAMATH_GPT_sample_size_correct_l637_63706

-- Define the conditions as lean variables
def total_employees := 120
def male_employees := 90
def female_sample := 9

-- Define the proof problem statement
theorem sample_size_correct : ∃ n : ℕ, (total_employees - male_employees) / total_employees = female_sample / n ∧ n = 36 := by 
  sorry

end NUMINAMATH_GPT_sample_size_correct_l637_63706


namespace NUMINAMATH_GPT_man_work_m_alone_in_15_days_l637_63714

theorem man_work_m_alone_in_15_days (M : ℕ) (h1 : 1/M + 1/10 = 1/6) : M = 15 := sorry

end NUMINAMATH_GPT_man_work_m_alone_in_15_days_l637_63714


namespace NUMINAMATH_GPT_num_digits_difference_l637_63734

-- Define the two base-10 integers
def n1 : ℕ := 150
def n2 : ℕ := 950

-- Find the number of digits in the base-2 representation of these numbers.
def num_digits_base2 (n : ℕ) : ℕ :=
  Nat.log2 n + 1

-- State the theorem
theorem num_digits_difference :
  num_digits_base2 n2 - num_digits_base2 n1 = 2 :=
by
  sorry

end NUMINAMATH_GPT_num_digits_difference_l637_63734


namespace NUMINAMATH_GPT_correct_choice_is_B_l637_63780

def draw_ray := "Draw ray OP=3cm"
def connect_points := "Connect points A and B"
def draw_midpoint := "Draw the midpoint of points A and B"
def draw_distance := "Draw the distance between points A and B"

-- Mathematical function to identify the correct statement about drawing
def correct_drawing_statement (s : String) : Prop :=
  s = connect_points

theorem correct_choice_is_B :
  correct_drawing_statement connect_points :=
by
  sorry

end NUMINAMATH_GPT_correct_choice_is_B_l637_63780


namespace NUMINAMATH_GPT_smallest_composite_no_prime_factors_less_than_20_l637_63773

def isComposite (n : ℕ) : Prop := ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = n
def hasNoPrimeFactorsLessThan (n minPrime : ℕ) : Prop :=
  ∀ p : ℕ, Prime p → p < minPrime → ¬(p ∣ n)

theorem smallest_composite_no_prime_factors_less_than_20 : ∃ n : ℕ, isComposite n ∧ hasNoPrimeFactorsLessThan n 20 ∧ ∀ m : ℕ, isComposite m ∧ hasNoPrimeFactorsLessThan m 20 → 529 ≤ m :=
by
  sorry

end NUMINAMATH_GPT_smallest_composite_no_prime_factors_less_than_20_l637_63773


namespace NUMINAMATH_GPT_person_B_processes_components_l637_63752

theorem person_B_processes_components (x : ℕ) (h1 : ∀ x, x > 0 → x + 2 > 0) 
(h2 : ∀ x, x > 0 → (25 / (x + 2)) = (20 / x)) :
  x = 8 := sorry

end NUMINAMATH_GPT_person_B_processes_components_l637_63752


namespace NUMINAMATH_GPT_square_perimeter_ratio_l637_63713

theorem square_perimeter_ratio (a b : ℝ) (h : a^2 / b^2 = 16 / 25) :
  (4 * a) / (4 * b) = 4 / 5 :=
by
  sorry

end NUMINAMATH_GPT_square_perimeter_ratio_l637_63713


namespace NUMINAMATH_GPT_triangle_BC_value_l637_63742

theorem triangle_BC_value (B C A : ℝ) (AB AC BC : ℝ) 
  (hB : B = 45) 
  (hAB : AB = 100)
  (hAC : AC = 100)
  (h_deg : A ≠ 0) :
  BC = 100 * Real.sqrt 2 := 
by 
  sorry

end NUMINAMATH_GPT_triangle_BC_value_l637_63742


namespace NUMINAMATH_GPT_circle_equation_correct_l637_63781

-- Define the given elements: center and radius
def center : (ℝ × ℝ) := (1, -1)
def radius : ℝ := 2

-- Define the equation of the circle with the given center and radius
def circle_eqn (x y : ℝ) : Prop := (x - 1)^2 + (y + 1)^2 = radius^2

-- Prove that the equation of the circle holds with the given center and radius
theorem circle_equation_correct : 
  ∀ x y : ℝ, circle_eqn x y ↔ (x - 1)^2 + (y + 1)^2 = 4 := 
by
  sorry

end NUMINAMATH_GPT_circle_equation_correct_l637_63781


namespace NUMINAMATH_GPT_unique_solution_m_l637_63762

theorem unique_solution_m (m : ℝ) :
  (∃! x : ℝ, m * x^2 + 2 * x - 1 = 0) ↔ (m = 0 ∨ m = -1) :=
by
  sorry

end NUMINAMATH_GPT_unique_solution_m_l637_63762


namespace NUMINAMATH_GPT_total_worth_of_stock_l637_63768

theorem total_worth_of_stock (x y : ℕ) (cheap_cost expensive_cost : ℝ) 
  (h1 : y = 21) (h2 : x + y = 22)
  (h3 : expensive_cost = 10) (h4 : cheap_cost = 2.5) :
  (x * expensive_cost + y * cheap_cost) = 62.5 :=
by
  sorry

end NUMINAMATH_GPT_total_worth_of_stock_l637_63768


namespace NUMINAMATH_GPT_friends_meeting_distance_l637_63796

theorem friends_meeting_distance (R_q : ℝ) (t : ℝ) (D_p D_q trail_length : ℝ) :
  trail_length = 36 ∧ D_p = 1.25 * R_q * t ∧ D_q = R_q * t ∧ D_p + D_q = trail_length → D_p = 20 := by
  sorry

end NUMINAMATH_GPT_friends_meeting_distance_l637_63796


namespace NUMINAMATH_GPT_Ray_wrote_35_l637_63776

theorem Ray_wrote_35 :
  ∃ (x y : ℕ), (10 * x + y = 35) ∧ (10 * x + y = 4 * (x + y) + 3) ∧ (10 * x + y + 18 = 10 * y + x) :=
by
  sorry

end NUMINAMATH_GPT_Ray_wrote_35_l637_63776


namespace NUMINAMATH_GPT_number_in_scientific_notation_l637_63792

/-- Condition: A constant corresponding to the number we are converting. -/
def number : ℕ := 9000000000

/-- Condition: The correct answer we want to prove. -/
def correct_answer : ℕ := 9 * 10^9

/-- Proof Problem: Prove that the number equals the correct_answer when expressed in scientific notation. -/
theorem number_in_scientific_notation : number = correct_answer := by
  sorry

end NUMINAMATH_GPT_number_in_scientific_notation_l637_63792


namespace NUMINAMATH_GPT_sqrt_fraction_sum_as_common_fraction_l637_63712

theorem sqrt_fraction_sum_as_common_fraction (a b c d : ℚ) (ha : a = 25) (hb : b = 36) (hc : c = 16) (hd : d = 9) :
  Real.sqrt ((a / b) + (c / d)) = Real.sqrt 89 / 6 := by
  sorry

end NUMINAMATH_GPT_sqrt_fraction_sum_as_common_fraction_l637_63712


namespace NUMINAMATH_GPT_bonus_percentage_is_correct_l637_63745

theorem bonus_percentage_is_correct (kills total_points enemies_points bonus_threshold bonus_percentage : ℕ) 
  (h1 : enemies_points = 10) 
  (h2 : kills = 150) 
  (h3 : total_points = 2250) 
  (h4 : bonus_threshold = 100) 
  (h5 : kills >= bonus_threshold) 
  (h6 : bonus_percentage = (total_points - kills * enemies_points) * 100 / (kills * enemies_points)) : 
  bonus_percentage = 50 := 
by
  sorry

end NUMINAMATH_GPT_bonus_percentage_is_correct_l637_63745


namespace NUMINAMATH_GPT_max_a_value_l637_63700

theorem max_a_value (a : ℝ) : 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → -2022 ≤ (a - 1) * x^2 - (a - 1) * x + 2022 ∧ 
                                (a - 1) * x^2 - (a - 1) * x + 2022 ≤ 2022) →
  a = 16177 :=
sorry

end NUMINAMATH_GPT_max_a_value_l637_63700


namespace NUMINAMATH_GPT_new_ratio_cooks_waiters_l637_63746

theorem new_ratio_cooks_waiters
  (initial_ratio : ℕ → ℕ → Prop)
  (cooks waiters : ℕ) :
  initial_ratio 9 24 → 
  12 + waiters = 36 →
  initial_ratio 3 8 →
  9 * 4 = 36 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_new_ratio_cooks_waiters_l637_63746


namespace NUMINAMATH_GPT_maximize_profit_l637_63701

noncomputable def profit (m : ℝ) : ℝ := 
  29 - (16 / (m + 1) + (m + 1))

theorem maximize_profit : 
  ∃ m : ℝ, m = 3 ∧ m ≥ 0 ∧ profit m = 21 :=
by
  use 3
  repeat { sorry }

end NUMINAMATH_GPT_maximize_profit_l637_63701


namespace NUMINAMATH_GPT_log_diff_condition_l637_63744

theorem log_diff_condition (a : ℕ → ℝ) (d e : ℝ) (H1 : ∀ n : ℕ, n > 1 → a n = Real.log n / Real.log 3003)
  (H2 : d = a 2 + a 3 + a 4 + a 5 + a 6) (H3 : e = a 15 + a 16 + a 17 + a 18 + a 19) :
  d - e = -Real.log 1938 / Real.log 3003 := by
  sorry

end NUMINAMATH_GPT_log_diff_condition_l637_63744


namespace NUMINAMATH_GPT_rate_percent_calculation_l637_63716

theorem rate_percent_calculation 
  (SI : ℝ) (P : ℝ) (T : ℝ) (R : ℝ) 
  (h1 : SI = 3125) 
  (h2 : P = 12500) 
  (h3 : T = 7) 
  (h4 : SI = P * R * T / 100) :
  R = 3.57 :=
by
  sorry

end NUMINAMATH_GPT_rate_percent_calculation_l637_63716

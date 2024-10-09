import Mathlib

namespace sqrt_trig_identity_l739_73986

theorem sqrt_trig_identity
  (α : ℝ)
  (P : ℝ × ℝ)
  (hP: P = (Real.sin 2, Real.cos 2))
  (h_terminal: ∃ (θ : ℝ), P = (Real.cos θ, Real.sin θ)) :
  Real.sqrt (2 * (1 - Real.sin α)) = 2 * Real.sin 1 := 
sorry

end sqrt_trig_identity_l739_73986


namespace solve_inequality_l739_73951

theorem solve_inequality (x : ℝ) (h : 0 < x ∧ x < 2) : abs (2 * x - 1) < abs x + 1 :=
by
  sorry

end solve_inequality_l739_73951


namespace point_C_values_l739_73932

variable (B C : ℝ)
variable (distance_BC : ℝ)
variable (hB : B = 3)
variable (hDistance : distance_BC = 2)

theorem point_C_values (hBC : abs (C - B) = distance_BC) : (C = 1 ∨ C = 5) := 
by
  sorry

end point_C_values_l739_73932


namespace min_value_a_plus_2b_l739_73905

theorem min_value_a_plus_2b (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a * b = 20) : a + 2 * b = 4 * Real.sqrt 10 :=
by
  sorry

end min_value_a_plus_2b_l739_73905


namespace circuit_boards_fail_inspection_l739_73962

theorem circuit_boards_fail_inspection (P F : ℝ) (h1 : P + F = 3200)
    (h2 : (1 / 8) * P + F = 456) : F = 64 :=
by
  sorry

end circuit_boards_fail_inspection_l739_73962


namespace surface_area_of_sphere_with_diameter_4_l739_73920

theorem surface_area_of_sphere_with_diameter_4 :
    let diameter := 4
    let radius := diameter / 2
    let surface_area := 4 * Real.pi * radius^2
    surface_area = 16 * Real.pi :=
by
  -- Sorry is used in place of the actual proof.
  sorry

end surface_area_of_sphere_with_diameter_4_l739_73920


namespace initially_calculated_average_weight_l739_73944

theorem initially_calculated_average_weight (n : ℕ) (misread_diff correct_avg_weight : ℝ)
  (hn : n = 20) (hmisread_diff : misread_diff = 10) (hcorrect_avg_weight : correct_avg_weight = 58.9) :
  ((correct_avg_weight * n - misread_diff) / n) = 58.4 :=
by
  rw [hn, hmisread_diff, hcorrect_avg_weight]
  sorry

end initially_calculated_average_weight_l739_73944


namespace train_cross_time_l739_73980

noncomputable def speed_conversion (speed_kmh : ℝ) : ℝ :=
  speed_kmh * (1000 / 3600)

noncomputable def time_to_cross_pole (length_m speed_kmh : ℝ) : ℝ :=
  length_m / speed_conversion speed_kmh

theorem train_cross_time (length_m : ℝ) (speed_kmh : ℝ) :
  length_m = 225 → speed_kmh = 250 → time_to_cross_pole length_m speed_kmh = 3.24 := by
  intros hlen hspeed
  simp [time_to_cross_pole, speed_conversion, hlen, hspeed]
  sorry

end train_cross_time_l739_73980


namespace batsman_average_after_17th_inning_l739_73954

theorem batsman_average_after_17th_inning
  (A : ℕ)
  (h1 : (16 * A + 88) / 17 = A + 3) :
  37 + 3 = 40 :=
by sorry

end batsman_average_after_17th_inning_l739_73954


namespace parallel_lines_m_values_l739_73958

theorem parallel_lines_m_values (m : ℝ) :
  (∀ x y : ℝ, (3 + m) * x + 4 * y = 5 → 2 * x + (5 + m) * y = 8) →
  (m = -1 ∨ m = -7) :=
by
  sorry

end parallel_lines_m_values_l739_73958


namespace average_speed_joey_round_trip_l739_73953

noncomputable def average_speed_round_trip
  (d : ℝ) (t₁ : ℝ) (r : ℝ) (s₂ : ℝ) : ℝ :=
  2 * d / (t₁ + d / s₂)

-- Lean statement for the proof problem
theorem average_speed_joey_round_trip :
  average_speed_round_trip 6 1 6 12 = 8 := sorry

end average_speed_joey_round_trip_l739_73953


namespace expression_value_l739_73996

theorem expression_value :
  3 + Real.sqrt 3 + 1 / (3 + Real.sqrt 3) + 1 / (Real.sqrt 3 - 3) = 3 + 2 * Real.sqrt 3 / 3 :=
by
  sorry

end expression_value_l739_73996


namespace equation_of_line_AB_l739_73969

def is_midpoint (P A B : ℝ × ℝ) : Prop :=
  P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

def on_circle (C : ℝ × ℝ) (r : ℝ) (P : ℝ × ℝ) : Prop :=
  (P.1 - C.1) ^ 2 + P.2 ^ 2 = r ^ 2

theorem equation_of_line_AB : 
  ∃ A B : ℝ × ℝ, 
    is_midpoint (2, -1) A B ∧ 
    on_circle (1, 0) 5 A ∧ 
    on_circle (1, 0) 5 B ∧ 
    ∀ x y : ℝ, (x - y - 3 = 0) ∧ 
    ∃ t : ℝ, ∃ u : ℝ, (t - u - 3 = 0) := 
sorry

end equation_of_line_AB_l739_73969


namespace Sam_age_l739_73927

theorem Sam_age (S D : ℕ) (h1 : S + D = 54) (h2 : S = D / 2) : S = 18 :=
by
  -- Proof omitted
  sorry

end Sam_age_l739_73927


namespace real_solutions_l739_73992

theorem real_solutions:
  ∀ x: ℝ, 
    (x ≠ 2) ∧ (x ≠ 4) ∧ 
    ((x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 3) * (x - 2) * (x - 1)) / 
    ((x - 2) * (x - 4) * (x - 2)) = 1 
    → (x = 2 + Real.sqrt 2) ∨ (x = 2 - Real.sqrt 2) :=
by
  sorry

end real_solutions_l739_73992


namespace find_a_l739_73902

-- Definitions of the conditions
variables {a b c : ℤ} 

-- Theorem statement
theorem find_a (h1: a + b = c) (h2: b + c = 7) (h3: c = 4) : a = 1 :=
by
  -- Using sorry to skip the proof
  sorry

end find_a_l739_73902


namespace coin_flip_probability_l739_73904

theorem coin_flip_probability (p : ℝ) 
  (h : p^2 + (1 - p)^2 = 4 * p * (1 - p)) : 
  p = (3 + Real.sqrt 3) / 6 :=
sorry

end coin_flip_probability_l739_73904


namespace minimum_cubes_required_l739_73987

def cube_snaps_visible (n : Nat) : Prop := 
  ∀ (cubes : Fin n → Fin 6 → Bool),
    (∀ i, (cubes i 0 ∧ cubes i 1) ∨ ¬(cubes i 0 ∨ cubes i 1)) → 
    ∃ i j, (i ≠ j) ∧ 
            (cubes i 0 ↔ ¬ cubes j 0) ∧ 
            (cubes i 1 ↔ ¬ cubes j 1)

theorem minimum_cubes_required : 
  ∃ n, cube_snaps_visible n ∧ n = 4 := 
  by sorry

end minimum_cubes_required_l739_73987


namespace snowboard_final_price_l739_73910

noncomputable def original_price : ℝ := 200
noncomputable def discount_friday : ℝ := 0.40
noncomputable def discount_monday : ℝ := 0.25

noncomputable def price_after_friday_discount (orig : ℝ) (discount : ℝ) : ℝ :=
  (1 - discount) * orig

noncomputable def final_price (price_friday : ℝ) (discount : ℝ) : ℝ :=
  (1 - discount) * price_friday

theorem snowboard_final_price :
  final_price (price_after_friday_discount original_price discount_friday) discount_monday = 90 := 
sorry

end snowboard_final_price_l739_73910


namespace annual_rent_per_square_foot_l739_73995

-- Given conditions
def dimensions_length : ℕ := 10
def dimensions_width : ℕ := 10
def monthly_rent : ℕ := 1300

-- Derived conditions
def area : ℕ := dimensions_length * dimensions_width
def annual_rent : ℕ := monthly_rent * 12

-- The problem statement as a theorem in Lean 4
theorem annual_rent_per_square_foot :
  annual_rent / area = 156 := by
  sorry

end annual_rent_per_square_foot_l739_73995


namespace negation_of_p_is_neg_p_l739_73993

def p (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2 : ℝ, (f x2 - f x1) * (x2 - x1) ≥ 0

def neg_p (f : ℝ → ℝ) : Prop :=
  ∃ x1 x2 : ℝ, (f x2 - f x1) * (x2 - x1) < 0

theorem negation_of_p_is_neg_p (f : ℝ → ℝ) : ¬ p f ↔ neg_p f :=
by
  sorry -- Proof of this theorem

end negation_of_p_is_neg_p_l739_73993


namespace intersecting_rectangles_shaded_area_l739_73972

theorem intersecting_rectangles_shaded_area 
  (a_w : ℕ) (a_l : ℕ) (b_w : ℕ) (b_l : ℕ) (c_w : ℕ) (c_l : ℕ)
  (overlap_ab_w : ℕ) (overlap_ab_h : ℕ)
  (overlap_ac_w : ℕ) (overlap_ac_h : ℕ)
  (overlap_bc_w : ℕ) (overlap_bc_h : ℕ)
  (triple_overlap_w : ℕ) (triple_overlap_h : ℕ) :
  a_w = 4 → a_l = 12 →
  b_w = 5 → b_l = 10 →
  c_w = 3 → c_l = 6 →
  overlap_ab_w = 4 → overlap_ab_h = 5 →
  overlap_ac_w = 3 → overlap_ac_h = 4 →
  overlap_bc_w = 3 → overlap_bc_h = 3 →
  triple_overlap_w = 3 → triple_overlap_h = 3 →
  ((a_w * a_l) + (b_w * b_l) + (c_w * c_l)) - 
  ((overlap_ab_w * overlap_ab_h) + (overlap_ac_w * overlap_ac_h) + (overlap_bc_w * overlap_bc_h)) + 
  (triple_overlap_w * triple_overlap_h) = 84 :=
by 
  sorry

end intersecting_rectangles_shaded_area_l739_73972


namespace initial_amount_l739_73961

theorem initial_amount (A : ℝ) (h : (9 / 8) * (9 / 8) * A = 40500) : 
  A = 32000 :=
sorry

end initial_amount_l739_73961


namespace leastCookies_l739_73976

theorem leastCookies (b : ℕ) :
  (b % 6 = 5) ∧ (b % 8 = 3) ∧ (b % 9 = 7) →
  b = 179 :=
by
  sorry

end leastCookies_l739_73976


namespace vertex_angle_of_isosceles_with_angle_30_l739_73956

def isosceles_triangle (a b c : ℝ) : Prop :=
  (a = b ∨ b = c ∨ c = a) ∧ a + b + c = 180

theorem vertex_angle_of_isosceles_with_angle_30 (a b c : ℝ) 
  (ha : isosceles_triangle a b c) 
  (h1 : a = 30 ∨ b = 30 ∨ c = 30) :
  (a = 30 ∨ b = 30 ∨ c = 30) ∨ (a = 120 ∨ b = 120 ∨ c = 120) := 
sorry

end vertex_angle_of_isosceles_with_angle_30_l739_73956


namespace smallest_n_terminating_decimal_l739_73926

theorem smallest_n_terminating_decimal :
  ∃ n : ℕ, (∀ m : ℕ, (∃ k : ℕ, m + 150 = 2^k * (5^0))
  → (m ≥ n)) ∧ (n > 0) ∧ (∃ k : ℕ, n + 150 = 2^k * (5^0)) := 
sorry

end smallest_n_terminating_decimal_l739_73926


namespace arithmetic_mean_is_one_l739_73916

theorem arithmetic_mean_is_one (x a : ℝ) (hx : x ≠ 0) (hx2a : x^2 ≠ a) :
  (1 / 2 * ((x^2 + a) / x^2 + (x^2 - a) / x^2) = 1) :=
by
  sorry

end arithmetic_mean_is_one_l739_73916


namespace min_score_to_achieve_average_l739_73975

theorem min_score_to_achieve_average (a b c : ℕ) (h₁ : a = 76) (h₂ : b = 94) (h₃ : c = 87) :
  ∃ d e : ℕ, d + e = 148 ∧ d ≤ 100 ∧ e ≤ 100 ∧ min d e = 48 :=
by sorry

end min_score_to_achieve_average_l739_73975


namespace sequence_sum_l739_73935

theorem sequence_sum (x y : ℕ) 
  (r : ℚ) 
  (h1 : 4 * r = 1) 
  (h2 : x = 256 * r)
  (h3 : y = x * r): 
  x + y = 80 := 
by 
  sorry

end sequence_sum_l739_73935


namespace tangent_line_at_one_extreme_points_and_inequality_l739_73960

noncomputable def f (x a : ℝ) := x^2 - 2*x + a * Real.log x

-- Question 1: Tangent Line
theorem tangent_line_at_one (x a : ℝ) (h_a : a = 2) (hx_pos : x > 0) :
    2*x - Real.log x - (2*x - Real.log 1 - 1) = 0 := by
  sorry

-- Question 2: Extreme Points and Inequality
theorem extreme_points_and_inequality (a x1 x2 : ℝ) (h1 : 2*x1^2 - 2*x1 + a = 0)
    (h2 : 2*x2^2 - 2*x2 + a = 0) (hx12 : x1 < x2) (hx1_pos : x1 > 0) (hx2_pos : x2 > 0) :
    0 < a ∧ a < 1/2 ∧ (f x1 a) / x2 > -3/2 - Real.log 2 := by
  sorry

end tangent_line_at_one_extreme_points_and_inequality_l739_73960


namespace pets_remaining_l739_73979

-- Definitions based on conditions
def initial_puppies : ℕ := 7
def initial_kittens : ℕ := 6
def sold_puppies : ℕ := 2
def sold_kittens : ℕ := 3

-- Theorem statement
theorem pets_remaining : initial_puppies + initial_kittens - (sold_puppies + sold_kittens) = 8 :=
by
  sorry

end pets_remaining_l739_73979


namespace sin_zero_range_valid_m_l739_73930

noncomputable def sin_zero_range (m : ℝ) : Prop :=
  ∀ f : ℝ → ℝ, 
    (∀ x : ℝ, f x = Real.sin (2 * x - Real.pi / 6) - m) →
    (∃ x1 x2 : ℝ, (0 ≤ x1 ∧ x1 ≤ Real.pi / 2) ∧ (0 ≤ x2 ∧ x2 ≤ Real.pi / 2) ∧ x1 ≠ x2 ∧ f x1 = 0 ∧ f x2 = 0)

theorem sin_zero_range_valid_m : 
  ∀ m : ℝ, sin_zero_range m ↔ (1 / 2 ≤ m ∧ m < 1) :=
sorry

end sin_zero_range_valid_m_l739_73930


namespace find_AE_l739_73931

-- Define the given conditions as hypotheses
variables (AB CD AC AE EC : ℝ)
variables (E : Type _)
variables (triangle_AED triangle_BEC : E)

-- Assume the given conditions
axiom AB_eq_9 : AB = 9
axiom CD_eq_12 : CD = 12
axiom AC_eq_14 : AC = 14
axiom areas_equal : ∀ h : ℝ, 1/2 * AE * h = 1/2 * EC * h

-- Declare the theorem statement to prove AE
theorem find_AE (h : ℝ) (h' : EC = AC - AE) (h'' : 4 * AE = 3 * EC) : AE = 6 :=
by {
  -- proof steps as intermediate steps
  sorry
}

end find_AE_l739_73931


namespace ratio_of_constants_l739_73948

theorem ratio_of_constants (a b c: ℝ) (h1 : 8 = 0.02 * a) (h2 : 2 = 0.08 * b) (h3 : c = b / a) : c = 1 / 16 :=
by sorry

end ratio_of_constants_l739_73948


namespace total_cost_football_games_l739_73988

-- Define the initial conditions
def games_this_year := 14
def games_last_year := 29
def price_this_year := 45
def price_lowest := 40
def price_highest := 65
def one_third_games_last_year := games_last_year / 3
def one_fourth_games_last_year := games_last_year / 4

-- Define the assertions derived from the conditions
def games_lowest_price := 9  -- rounded down from games_last_year / 3
def games_highest_price := 7  -- rounded down from games_last_year / 4
def remaining_games := games_last_year - (games_lowest_price + games_highest_price)

-- Define the costs calculation
def cost_this_year := games_this_year * price_this_year
def cost_lowest_price_games := games_lowest_price * price_lowest
def cost_highest_price_games := games_highest_price * price_highest
def total_cost := cost_this_year + cost_lowest_price_games + cost_highest_price_games

-- The theorem statement
theorem total_cost_football_games (h1 : games_lowest_price = 9) (h2 : games_highest_price = 7) 
  (h3 : cost_this_year = 630) (h4 : cost_lowest_price_games = 360) (h5 : cost_highest_price_games = 455) :
  total_cost = 1445 :=
by
  -- Since this is just the statement, we can simply put 'sorry' here.
  sorry

end total_cost_football_games_l739_73988


namespace triangle_area_l739_73915

theorem triangle_area (a b : ℝ) (h1 : b = (24 / a)) (h2 : 3 * 4 + a * (12 / a) = 12) : b = 3 / 2 :=
by
  sorry

end triangle_area_l739_73915


namespace n_not_2_7_l739_73981

open Set

variable (M N : Set ℕ)

-- Define the given set M
def M_def : Prop := M = {1, 4, 7}

-- Define the condition M ∪ N = M
def union_condition : Prop := M ∪ N = M

-- The main statement to be proved
theorem n_not_2_7 (M_def : M = {1, 4, 7}) (union_condition : M ∪ N = M) : N ≠ {2, 7} :=
  sorry

end n_not_2_7_l739_73981


namespace find_divisor_l739_73963

theorem find_divisor (dividend quotient remainder : ℕ) (h₁ : dividend = 176) (h₂ : quotient = 9) (h₃ : remainder = 5) : 
  ∃ divisor, dividend = (divisor * quotient) + remainder ∧ divisor = 19 := by
sorry

end find_divisor_l739_73963


namespace parabola_vertex_l739_73936

-- Define the condition: the equation of the parabola
def parabola_equation (x y : ℝ) : Prop :=
  y^2 + 4 * y + 3 * x + 1 = 0

-- Define the statement: prove that the vertex of the parabola is (1, -2)
theorem parabola_vertex :
  parabola_equation 1 (-2) :=
by
  sorry

end parabola_vertex_l739_73936


namespace alien_saturday_sequence_l739_73946

def a_1 : String := "A"
def a_2 : String := "AY"
def a_3 : String := "AYYA"
def a_4 : String := "AYYAYAAY"

noncomputable def a_5 : String := a_4 ++ "YAAYAYYA"
noncomputable def a_6 : String := a_5 ++ "YAAYAYYAAAYAYAAY"

theorem alien_saturday_sequence : 
  a_6 = "AYYAYAAYYAAYAYYAYAAYAYYAAAYAYAAY" :=
sorry

end alien_saturday_sequence_l739_73946


namespace kelseys_sister_age_in_2021_l739_73989

-- Definitions based on given conditions
def kelsey_birth_year : ℕ := 1999 - 25
def sister_birth_year : ℕ := kelsey_birth_year - 3

-- Prove that Kelsey's older sister is 50 years old in 2021
theorem kelseys_sister_age_in_2021 : (2021 - sister_birth_year) = 50 :=
by
  -- Add proof here
  sorry

end kelseys_sister_age_in_2021_l739_73989


namespace no_such_function_l739_73923

theorem no_such_function (f : ℕ → ℕ) : ¬ (∀ n : ℕ, f (f n) = n + 2019) :=
sorry

end no_such_function_l739_73923


namespace right_triangle_property_l739_73984

theorem right_triangle_property
  (a b c x : ℝ)
  (h1 : c^2 = a^2 + b^2)
  (h2 : 1/2 * a * b = 1/2 * c * x)
  : 1/x^2 = 1/a^2 + 1/b^2 :=
sorry

end right_triangle_property_l739_73984


namespace decimal_to_fraction_l739_73966

theorem decimal_to_fraction (x : ℝ) (hx : x = 2.35) : x = 47 / 20 := by
  sorry

end decimal_to_fraction_l739_73966


namespace probability_point_between_lines_l739_73913

theorem probability_point_between_lines {x y : ℝ} :
  (∀ x, y = -2 * x + 8) →
  (∀ x, y = -3 * x + 8) →
  0.33 = 0.33 :=
by
  intro hl hm
  sorry

end probability_point_between_lines_l739_73913


namespace friend_pays_correct_percentage_l739_73985

theorem friend_pays_correct_percentage (adoption_fee : ℝ) (james_payment : ℝ) (friend_payment : ℝ) 
  (h1 : adoption_fee = 200) 
  (h2 : james_payment = 150)
  (h3 : friend_payment = adoption_fee - james_payment) : 
  (friend_payment / adoption_fee) * 100 = 25 :=
by
  sorry

end friend_pays_correct_percentage_l739_73985


namespace sum_A_B_l739_73967

noncomputable def num_four_digit_odd_numbers_divisible_by_3 : ℕ := 1500
noncomputable def num_four_digit_multiples_of_7 : ℕ := 1286

theorem sum_A_B (A B : ℕ) :
  A = num_four_digit_odd_numbers_divisible_by_3 →
  B = num_four_digit_multiples_of_7 →
  A + B = 2786 :=
by
  intros hA hB
  rw [hA, hB]
  exact rfl

end sum_A_B_l739_73967


namespace problem_xyz_l739_73964

theorem problem_xyz (x y : ℝ) (h1 : (x + y)^2 = 16) (h2 : x * y = -8) :
  x^2 + y^2 = 32 :=
by
  sorry

end problem_xyz_l739_73964


namespace cube_volume_l739_73909

-- Define the surface area constant
def surface_area : ℝ := 725.9999999999998

-- Define the formula for surface area of a cube and solve for volume given the conditions
theorem cube_volume (SA : ℝ) (h : SA = surface_area) : 11^3 = 1331 :=
by sorry

end cube_volume_l739_73909


namespace min_le_mult_l739_73977

theorem min_le_mult {x y z m : ℝ} (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z)
    (hm : m = min (min (min 1 (x^9)) (y^9)) (z^7)) : m ≤ x * y^2 * z^3 :=
by
  sorry

end min_le_mult_l739_73977


namespace ones_digit_8_power_32_l739_73919

theorem ones_digit_8_power_32 : (8^32) % 10 = 6 :=
by sorry

end ones_digit_8_power_32_l739_73919


namespace Megan_finish_all_problems_in_8_hours_l739_73991

theorem Megan_finish_all_problems_in_8_hours :
  ∀ (math_problems spelling_problems problems_per_hour : ℕ),
    math_problems = 36 →
    spelling_problems = 28 →
    problems_per_hour = 8 →
    (math_problems + spelling_problems) / problems_per_hour = 8 :=
by
  intros
  sorry

end Megan_finish_all_problems_in_8_hours_l739_73991


namespace zoo_animal_difference_l739_73903

theorem zoo_animal_difference :
  let parrots := 8
  let snakes := 3 * parrots
  let monkeys := 2 * snakes
  let elephants := (parrots + snakes) / 2
  let zebras := elephants - 3
  monkeys - zebras = 35 := by
  let parrots := 8
  let snakes := 3 * parrots
  let monkeys := 2 * snakes
  let elephants := (parrots + snakes) / 2
  let zebras := elephants - 3
  show monkeys - zebras = 35
  sorry

end zoo_animal_difference_l739_73903


namespace system_of_two_linear_equations_l739_73990

theorem system_of_two_linear_equations :
  ((∃ x y z, x + z = 5 ∧ x - 2 * y = 6) → False) ∧
  ((∃ x y, x * y = 5 ∧ x - 4 * y = 2) → False) ∧
  ((∃ x y, x + y = 5 ∧ 3 * x - 4 * y = 12) → True) ∧
  ((∃ x y, x^2 + y = 2 ∧ x - y = 9) → False) :=
by {
  sorry
}

end system_of_two_linear_equations_l739_73990


namespace average_billboards_per_hour_l739_73939

-- Define the number of billboards seen in each hour
def billboards_first_hour := 17
def billboards_second_hour := 20
def billboards_third_hour := 23

-- Define the number of hours
def total_hours := 3

-- Prove that the average number of billboards per hour is 20
theorem average_billboards_per_hour : 
  (billboards_first_hour + billboards_second_hour + billboards_third_hour) / total_hours = 20 :=
by
  sorry

end average_billboards_per_hour_l739_73939


namespace probability_of_drawing_diamond_or_ace_l739_73928

-- Define the number of diamonds
def numDiamonds : ℕ := 13

-- Define the number of other Aces
def numOtherAces : ℕ := 3

-- Define the total number of cards in the deck
def totalCards : ℕ := 52

-- Define the number of desirable outcomes (either diamonds or Aces)
def numDesirableOutcomes : ℕ := numDiamonds + numOtherAces

-- Define the probability of drawing a diamond or an Ace
def desiredProbability : ℚ := numDesirableOutcomes / totalCards

theorem probability_of_drawing_diamond_or_ace :
  desiredProbability = 4 / 13 :=
by
  sorry

end probability_of_drawing_diamond_or_ace_l739_73928


namespace intersection_A_B_l739_73952

-- Define sets A and B
def A : Set ℤ := {-2, 0, 2}
def B : Set ℤ := {x | ∃ y ∈ A, |y| = x}

-- Prove that the intersection of A and B is {0, 2}
theorem intersection_A_B :
  A ∩ B = {0, 2} :=
by
  sorry

end intersection_A_B_l739_73952


namespace abscissa_midpoint_range_l739_73938

-- Definitions based on the given conditions.
def circle_eq (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 6
def on_circle (x y : ℝ) : Prop := circle_eq x y
def chord_length (A B : ℝ × ℝ) : Prop := (A.1 - B.1)^2 + (A.2 - B.2)^2 = (2 * Real.sqrt 2)^2
def line_eq (x y : ℝ) : Prop := x - y - 2 = 0
def on_line (x y : ℝ) : Prop := line_eq x y
def segment_length (P Q : ℝ × ℝ) : Prop := (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = 4
def acute_angle (P Q G : ℝ × ℝ) : Prop := -- definition of acute angle condition
  sorry -- placeholder for the actual definition

-- The proof statement.
theorem abscissa_midpoint_range {A B P Q G M : ℝ × ℝ}
  (h_A_on_circle : on_circle A.1 A.2)
  (h_B_on_circle : on_circle B.1 B.2)
  (h_AB_length : chord_length A B)
  (h_P_on_line : on_line P.1 P.2)
  (h_Q_on_line : on_line Q.1 Q.2)
  (h_PQ_length : segment_length P Q)
  (h_angle_acute : acute_angle P Q G)
  (h_G_mid : G = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))
  (h_M_mid : M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)) :
  (M.1 < 0) ∨ (M.1 > 3) :=
sorry

end abscissa_midpoint_range_l739_73938


namespace color_of_182nd_marble_l739_73908

-- conditions
def pattern_length : ℕ := 15
def blue_length : ℕ := 6
def red_length : ℕ := 5
def green_length : ℕ := 4

def marble_color (n : ℕ) : String :=
  let cycle_pos := n % pattern_length
  if cycle_pos < blue_length then
    "blue"
  else if cycle_pos < blue_length + red_length then
    "red"
  else
    "green"

theorem color_of_182nd_marble : marble_color 182 = "blue" :=
by
  sorry

end color_of_182nd_marble_l739_73908


namespace find_incorrect_value_of_observation_l739_73906

noncomputable def incorrect_observation_value (mean1 : ℝ) (mean2 : ℝ) (n : ℕ) : ℝ :=
  let old_sum := mean1 * n
  let new_sum := mean2 * n
  let correct_value := 45
  let incorrect_value := (old_sum - new_sum + correct_value)
  (incorrect_value / -1)

theorem find_incorrect_value_of_observation :
  incorrect_observation_value 36 36.5 50 = 20 :=
by
  -- By the problem setup, incorrect_observation_value 36 36.5 50 is as defined in the proof steps.
  -- As per the proof steps and calculation, incorrect_observation_value 36 36.5 50 should evaluate to 20.
  sorry

end find_incorrect_value_of_observation_l739_73906


namespace estate_value_l739_73943

theorem estate_value (E : ℝ) (x : ℝ) (y: ℝ) (z: ℝ) 
  (h1 : 9 * x = 3 / 4 * E) 
  (h2 : z = 8 * x) 
  (h3 : y = 600) 
  (h4 : E = z + 9 * x + y):
  E = 1440 := 
sorry

end estate_value_l739_73943


namespace total_canoes_built_l739_73914

def geometric_sum (a r n : ℕ) : ℕ :=
  a * ((r^n - 1) / (r - 1))

theorem total_canoes_built : geometric_sum 10 3 7 = 10930 := 
  by
    -- The proof will go here.
    sorry

end total_canoes_built_l739_73914


namespace complete_square_solution_l739_73974

theorem complete_square_solution (x : ℝ) :
  x^2 - 8 * x + 6 = 0 → (x - 4)^2 = 10 :=
by
  intro h
  -- Proof would go here
  sorry

end complete_square_solution_l739_73974


namespace find_s_for_g_neg1_zero_l739_73941

def g (x s : ℝ) : ℝ := 3 * x^4 + 2 * x^3 - x^2 - 4 * x + s

theorem find_s_for_g_neg1_zero (s : ℝ) : g (-1) s = 0 ↔ s = -4 :=
by
  sorry

end find_s_for_g_neg1_zero_l739_73941


namespace find_inradius_l739_73973

-- Define variables and constants
variables (P A : ℝ)
variables (s r : ℝ)

-- Given conditions as definitions
def perimeter_triangle : Prop := P = 36
def area_triangle : Prop := A = 45

-- Semi-perimeter definition
def semi_perimeter : Prop := s = P / 2

-- Inradius and area relationship
def inradius_area_relation : Prop := A = r * s

-- Theorem statement
theorem find_inradius (hP : perimeter_triangle P) (hA : area_triangle A) (hs : semi_perimeter P s) (har : inradius_area_relation A r s) :
  r = 2.5 :=
by
  sorry

end find_inradius_l739_73973


namespace production_bottles_l739_73970

-- Definitions from the problem conditions
def machines_production_rate (machines : ℕ) (rate : ℕ) : ℕ := rate / machines
def total_production (machines rate minutes : ℕ) : ℕ := machines * rate * minutes

-- Theorem to prove the solution
theorem production_bottles :
  machines_production_rate 6 300 = 50 →
  total_production 10 50 4 = 2000 :=
by
  intro h
  have : 10 * 50 * 4 = 2000 := by norm_num
  exact this

end production_bottles_l739_73970


namespace expand_product_l739_73912

theorem expand_product (x : ℝ) :
  (x + 4) * (x - 5) = x^2 - x - 20 :=
by
  -- The proof will use algebraic identities and simplifications.
  sorry

end expand_product_l739_73912


namespace train_speed_proof_l739_73998

noncomputable def train_length : ℝ := 620
noncomputable def crossing_time : ℝ := 30.99752019838413
noncomputable def man_speed_kmh : ℝ := 8

noncomputable def man_speed_ms : ℝ := man_speed_kmh * (1000 / 3600)
noncomputable def relative_speed : ℝ := train_length / crossing_time
noncomputable def train_speed_ms : ℝ := relative_speed + man_speed_ms
noncomputable def train_speed_kmh : ℝ := train_speed_ms * (3600 / 1000)

theorem train_speed_proof : abs (train_speed_kmh - 80) < 0.0001 := by
  sorry

end train_speed_proof_l739_73998


namespace yang_hui_problem_solution_l739_73949

theorem yang_hui_problem_solution (x : ℕ) (h : x * (x - 1) = 650) : x * (x - 1) = 650 :=
by
  exact h

end yang_hui_problem_solution_l739_73949


namespace fraction_of_fraction_of_fraction_l739_73917

theorem fraction_of_fraction_of_fraction (a b c d : ℝ) (h₁ : a = 1/5) (h₂ : b = 1/3) (h₃ : c = 1/6) (h₄ : d = 90) :
  (a * b * c * d) = 1 :=
by
  rw [h₁, h₂, h₃, h₄]
  simp
  sorry -- To indicate that the proof is missing

end fraction_of_fraction_of_fraction_l739_73917


namespace shaded_area_fraction_l739_73918

theorem shaded_area_fraction (ABCD_area : ℝ) (shaded_square1_area : ℝ) (shaded_rectangle_area : ℝ) (shaded_square2_area : ℝ) (total_shaded_area : ℝ)
  (h_ABCD : ABCD_area = 36) 
  (h_shaded_square1 : shaded_square1_area = 4)
  (h_shaded_rectangle : shaded_rectangle_area = 12)
  (h_shaded_square2 : shaded_square2_area = 36)
  (h_total_shaded : total_shaded_area = 16) :
  (total_shaded_area / ABCD_area) = 4 / 9 :=
by 
  simp [h_ABCD, h_total_shaded]
  sorry

end shaded_area_fraction_l739_73918


namespace cyclist_speed_l739_73925

noncomputable def required_speed (d t : ℝ) : ℝ := d / t

theorem cyclist_speed :
  ∀ (d t : ℝ), 
  (d / 10 = t + 1) → 
  (d / 15 = t - 1) →
  required_speed d t = 12 := 
by
  intros d t h1 h2
  sorry

end cyclist_speed_l739_73925


namespace number_of_sections_l739_73999

noncomputable def initial_rope : ℕ := 50
noncomputable def rope_for_art := initial_rope / 5
noncomputable def remaining_rope_after_art := initial_rope - rope_for_art
noncomputable def rope_given_to_friend := remaining_rope_after_art / 2
noncomputable def remaining_rope := remaining_rope_after_art - rope_given_to_friend
noncomputable def section_size : ℕ := 2
noncomputable def sections := remaining_rope / section_size

theorem number_of_sections : sections = 10 :=
by
  sorry

end number_of_sections_l739_73999


namespace strategy_for_antonio_l739_73924

-- We define the concept of 'winning' and 'losing' positions
def winning_position (m n : ℕ) : Prop :=
  ¬ (m % 2 = 0 ∧ n % 2 = 0)

-- Now create the main theorem
theorem strategy_for_antonio (m n : ℕ) : winning_position m n ↔ 
  (¬(m % 2 = 0 ∧ n % 2 = 0)) :=
by
  unfold winning_position
  sorry

end strategy_for_antonio_l739_73924


namespace runway_show_total_time_l739_73907

-- Define the conditions
def time_per_trip : Nat := 2
def num_models : Nat := 6
def trips_bathing_suits_per_model : Nat := 2
def trips_evening_wear_per_model : Nat := 3
def trips_per_model : Nat := trips_bathing_suits_per_model + trips_evening_wear_per_model
def total_trips : Nat := trips_per_model * num_models

-- State the theorem
theorem runway_show_total_time : total_trips * time_per_trip = 60 := by
  -- fill in the proof here
  sorry

end runway_show_total_time_l739_73907


namespace combined_tax_rate_33_33_l739_73959

-- Define the necessary conditions
def mork_tax_rate : ℝ := 0.40
def mindy_tax_rate : ℝ := 0.30
def mindy_income_ratio : ℝ := 2.0

-- Main theorem statement
theorem combined_tax_rate_33_33 :
  ∀ (X : ℝ), ((mork_tax_rate * X + mindy_income_ratio * mindy_tax_rate * X) / (X + mindy_income_ratio * X) * 100) = 100 / 3 :=
by
  intro X
  sorry

end combined_tax_rate_33_33_l739_73959


namespace proof_problem_l739_73965

theorem proof_problem (x y : ℝ) (h1 : 3 * x ^ 2 - 5 * x + 4 * y + 6 = 0) 
                      (h2 : 3 * x - 2 * y + 1 = 0) : 
                      4 * y ^ 2 - 2 * y + 24 = 0 := 
by 
  sorry

end proof_problem_l739_73965


namespace jose_profit_share_l739_73937

def investment_share (toms_investment : ℕ) (jose_investment : ℕ) 
  (toms_duration : ℕ) (jose_duration : ℕ) (total_profit : ℕ) : ℕ :=
  let toms_capital_months := toms_investment * toms_duration
  let jose_capital_months := jose_investment * jose_duration
  let total_capital_months := toms_capital_months + jose_capital_months
  let jose_share_ratio := jose_capital_months / total_capital_months
  jose_share_ratio * total_profit

theorem jose_profit_share 
  (toms_investment : ℕ := 3000)
  (jose_investment : ℕ := 4500)
  (toms_duration : ℕ := 12)
  (jose_duration : ℕ := 10)
  (total_profit : ℕ := 6300) :
  investment_share toms_investment jose_investment toms_duration jose_duration total_profit = 3500 := 
sorry

end jose_profit_share_l739_73937


namespace complex_expression_simplified_l739_73940

theorem complex_expression_simplified :
  let z1 := (1 + 3 * Complex.I) / (1 - 3 * Complex.I)
  let z2 := (1 - 3 * Complex.I) / (1 + 3 * Complex.I)
  let z3 := 1 / (8 * Complex.I^3)
  z1 + z2 + z3 = -1.6 + 0.125 * Complex.I := 
by
  sorry

end complex_expression_simplified_l739_73940


namespace totalTrianglesInFigure_l739_73929

-- Definition of the problem involving a rectangle with subdivisions creating triangles
def numberOfTrianglesInRectangle : Nat :=
  let smallestTriangles := 24   -- Number of smallest triangles
  let nextSizeTriangles1 := 8   -- Triangles formed by combining smallest triangles
  let nextSizeTriangles2 := 12
  let nextSizeTriangles3 := 16
  let largestTriangles := 4
  smallestTriangles + nextSizeTriangles1 + nextSizeTriangles2 + nextSizeTriangles3 + largestTriangles

-- The Lean 4 theorem statement, stating that the total number of triangles equals 64
theorem totalTrianglesInFigure : numberOfTrianglesInRectangle = 64 := 
by
  sorry

end totalTrianglesInFigure_l739_73929


namespace inequality_proof_l739_73997

variable {x y z : ℝ}

theorem inequality_proof (h : True) :
  ( (x^2 + 2*y^2 + 2*z^2) / (x^2 + y*z) +
    (y^2 + 2*z^2 + 2*x^2) / (y^2 + z*x) +
    (z^2 + 2*x^2 + 2*y^2) / (z^2 + x*y) ) ≥ 6 := by
  sorry

end inequality_proof_l739_73997


namespace crickets_needed_to_reach_11_l739_73911

theorem crickets_needed_to_reach_11 (collected_crickets : ℕ) (wanted_crickets : ℕ) 
                                     (h : collected_crickets = 7) (h2 : wanted_crickets = 11) :
  wanted_crickets - collected_crickets = 4 :=
sorry

end crickets_needed_to_reach_11_l739_73911


namespace final_price_of_coat_is_correct_l739_73968

-- Define the conditions as constants
def original_price : ℝ := 120
def discount_rate : ℝ := 0.30
def tax_rate : ℝ := 0.15

-- Define the discounted amount calculation
def discount_amount : ℝ := original_price * discount_rate

-- Define the sale price after the discount
def sale_price : ℝ := original_price - discount_amount

-- Define the tax amount calculation on the sale price
def tax_amount : ℝ := sale_price * tax_rate

-- Define the total selling price
def total_selling_price : ℝ := sale_price + tax_amount

-- The theorem that needs to be proven
theorem final_price_of_coat_is_correct : total_selling_price = 96.6 :=
by
  sorry

end final_price_of_coat_is_correct_l739_73968


namespace solve_for_q_l739_73921

theorem solve_for_q (p q : ℚ) (h1 : 5 * p + 6 * q = 20) (h2 : 6 * p + 5 * q = 29) : q = -25 / 11 :=
by {
  sorry
}

end solve_for_q_l739_73921


namespace base8_246_is_166_in_base10_l739_73945

def convert_base8_to_base10 (n : ℕ) : ℕ :=
  let d0 := n % 10;
  let d1 := (n / 10) % 10;
  let d2 := (n / 100) % 10;
  d2 * 8^2 + d1 * 8^1 + d0 * 8^0

theorem base8_246_is_166_in_base10 : convert_base8_to_base10 246 = 166 :=
  sorry

end base8_246_is_166_in_base10_l739_73945


namespace cost_of_pencil_and_pen_l739_73947

variable (p q : ℝ)

axiom condition1 : 4 * p + 3 * q = 4.20
axiom condition2 : 3 * p + 4 * q = 4.55

theorem cost_of_pencil_and_pen : p + q = 1.25 :=
by
  sorry

end cost_of_pencil_and_pen_l739_73947


namespace product_identity_l739_73950

theorem product_identity : 
  (7^3 - 1) / (7^3 + 1) * 
  (8^3 - 1) / (8^3 + 1) * 
  (9^3 - 1) / (9^3 + 1) * 
  (10^3 - 1) / (10^3 + 1) * 
  (11^3 - 1) / (11^3 + 1) = 
  133 / 946 := 
by
  sorry

end product_identity_l739_73950


namespace smallest_integer_in_correct_range_l739_73900

theorem smallest_integer_in_correct_range :
  ∃ (n : ℤ), n > 1 ∧ n % 3 = 1 ∧ n % 5 = 1 ∧ n % 8 = 1 ∧ n % 7 = 2 ∧ 161 ≤ n ∧ n ≤ 200 :=
by
  sorry

end smallest_integer_in_correct_range_l739_73900


namespace part_a_part_b_l739_73957

-- Part (a)
theorem part_a (m n : ℕ) (hm : 0 < m) (hn : 0 < n) (h : m > n) : 
  (1 + 1 / (m:ℝ))^m > (1 + 1 / (n:ℝ))^n :=
by sorry

-- Part (b)
theorem part_b (m n : ℕ) (hm : 0 < m) (hn : 1 < n) (h : m > n) : 
  (1 + 1 / (m:ℝ))^(m + 1) < (1 + 1 / (n:ℝ))^(n + 1) :=
by sorry

end part_a_part_b_l739_73957


namespace inequality_proof_l739_73983

noncomputable def given_condition_1 (a b c u : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ (∃ x, (a * x^2 - b * x + c = 0)) ∧
  a * u^2 - b * u + c ≤ 0

noncomputable def given_condition_2 (A B C v : ℝ) : Prop :=
  A > 0 ∧ B > 0 ∧ C > 0 ∧ (∃ x, (A * x^2 - B * x + C = 0)) ∧
  A * v^2 - B * v + C ≤ 0

theorem inequality_proof (a b c A B C u v : ℝ) (h1 : given_condition_1 a b c u) (h2 : given_condition_2 A B C v) :
  (a * u + A * v) * (c / u + C / v) ≤ (b + B) ^ 2 / 4 :=
by
    sorry

end inequality_proof_l739_73983


namespace part1_part2_l739_73901
open Real

-- Part 1
theorem part1 (x : ℝ) (h : 0 ≤ x ∧ x ≤ 1) :
  0 < (sqrt (1 + x) + sqrt (1 - x) + 2) * (sqrt (1 - x^2) + 1) ∧
  (sqrt (1 + x) + sqrt (1 - x) + 2) * (sqrt (1 - x^2) + 1) ≤ 8 := 
sorry

-- Part 2
theorem part2 (x : ℝ) (h : 0 ≤ x ∧ x ≤ 1) :
  ∃ β > 0, β = 4 ∧ sqrt (1 + x) + sqrt (1 - x) ≤ 2 - x^2 / β :=
sorry

end part1_part2_l739_73901


namespace number_of_male_rabbits_l739_73978

-- Definitions based on the conditions
def white_rabbits : ℕ := 12
def black_rabbits : ℕ := 9
def female_rabbits : ℕ := 8

-- The question and proof goal
theorem number_of_male_rabbits : 
  (white_rabbits + black_rabbits - female_rabbits) = 13 :=
by
  sorry

end number_of_male_rabbits_l739_73978


namespace inversely_proportional_ratio_l739_73933

theorem inversely_proportional_ratio (x y x1 x2 y1 y2 : ℝ) 
  (h_inv_prop : x * y = x1 * y2) 
  (h_ratio : x1 / x2 = 3 / 5) 
  (x1_nonzero : x1 ≠ 0) 
  (x2_nonzero : x2 ≠ 0) 
  (y1_nonzero : y1 ≠ 0) 
  (y2_nonzero : y2 ≠ 0) : 
  y1 / y2 = 5 / 3 := 
sorry

end inversely_proportional_ratio_l739_73933


namespace math_problem_l739_73922

theorem math_problem (a : ℝ) (h : a^2 - 4 * a + 3 = 0) (h_ne : a ≠ 2 ∧ a ≠ 3 ∧ a ≠ -3) :
  (9 - 3 * a) / (2 * a - 4) / (a + 2 - 5 / (a - 2)) = -3 / 8 :=
sorry

end math_problem_l739_73922


namespace candy_bars_given_to_sister_first_time_l739_73982

theorem candy_bars_given_to_sister_first_time (x : ℕ) :
  (7 - x) + 30 - 4 * x = 22 → x = 3 :=
by
  sorry

end candy_bars_given_to_sister_first_time_l739_73982


namespace lowest_price_correct_l739_73994

noncomputable def lowest_price (cost_per_component shipping_cost_per_unit fixed_costs number_of_components : ℕ) : ℕ :=
(cost_per_component + shipping_cost_per_unit) * number_of_components + fixed_costs

theorem lowest_price_correct :
  lowest_price 80 5 16500 150 / 150 = 195 :=
by
  sorry

end lowest_price_correct_l739_73994


namespace original_strip_length_l739_73955

theorem original_strip_length (x : ℝ) 
  (h1 : 3 + x + 3 + x + 3 + x + 3 + x + 3 = 27) : 
  4 * 9 + 4 * 3 = 57 := 
  sorry

end original_strip_length_l739_73955


namespace number_of_appointments_l739_73971

-- Define the conditions
variables {hours_in_workday : ℕ} {appointments_duration : ℕ} {permit_rate : ℕ} {total_permits : ℕ}
variables (H1 : hours_in_workday = 8) (H2 : appointments_duration = 3) (H3 : permit_rate = 50) (H4: total_permits = 100)

-- Define the question as a theorem with the correct answer
theorem number_of_appointments : 
  (hours_in_workday - (total_permits / permit_rate)) / appointments_duration = 2 :=
by
  -- Proof is not required
  sorry

end number_of_appointments_l739_73971


namespace not_all_zero_iff_at_least_one_non_zero_l739_73934

theorem not_all_zero_iff_at_least_one_non_zero (a b c : ℝ) : ¬ (a = 0 ∧ b = 0 ∧ c = 0) ↔ (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) :=
by
  sorry

end not_all_zero_iff_at_least_one_non_zero_l739_73934


namespace f_eq_four_or_seven_l739_73942

noncomputable def f (a b : ℕ) : ℚ := (a^2 + a * b + b^2) / (a * b - 1)

theorem f_eq_four_or_seven (a b : ℕ) (h : a > 0) (h1 : b > 0) (h2 : a * b ≠ 1) : 
  f a b = 4 ∨ f a b = 7 := 
sorry

end f_eq_four_or_seven_l739_73942

import Mathlib

namespace parallelogram_area_72_l409_40942

def parallelogram_area (base height : ℕ) : ℕ :=
  base * height

theorem parallelogram_area_72 :
  parallelogram_area 12 6 = 72 :=
by
  sorry

end parallelogram_area_72_l409_40942


namespace geometric_segment_l409_40951

theorem geometric_segment (AB A'B' : ℝ) (P D A B P' D' A' B' : ℝ) (x y a : ℝ) :
  AB = 3 ∧ A'B' = 6 ∧ (∀ P, dist P D = x) ∧ (∀ P', dist P' D' = 2 * x) ∧ x = a → x + y = 3 * a :=
by
  sorry

end geometric_segment_l409_40951


namespace solve_equation_l409_40965

-- Given conditions and auxiliary definitions
def is_solution (x y z : ℕ) : Prop := 2 ^ x + 3 ^ y - 7 = Nat.factorial z

-- Primary theorem: the equivalent proof problem
theorem solve_equation (x y z : ℕ) :
  (is_solution x y 3 → (x = 2 ∧ y = 2)) ∧
  (∀ z, (z ≤ 3 → z ≠ 3) → ¬is_solution x y z) ∧
  (z ≥ 4 → ¬is_solution x y z) :=
  sorry

end solve_equation_l409_40965


namespace price_of_small_bags_l409_40993

theorem price_of_small_bags (price_medium_bag : ℤ) (price_large_bag : ℤ) 
  (money_mark_has : ℤ) (balloons_in_small_bag : ℤ) 
  (balloons_in_medium_bag : ℤ) (balloons_in_large_bag : ℤ) 
  (total_balloons : ℤ) : 
  price_medium_bag = 6 → 
  price_large_bag = 12 → 
  money_mark_has = 24 → 
  balloons_in_small_bag = 50 → 
  balloons_in_medium_bag = 75 → 
  balloons_in_large_bag = 200 → 
  total_balloons = 400 → 
  (money_mark_has / (total_balloons / balloons_in_small_bag)) = 3 :=
by 
  sorry

end price_of_small_bags_l409_40993


namespace student_walks_fifth_to_first_l409_40968

theorem student_walks_fifth_to_first :
  let floors := 4
  let staircases := 2
  (staircases ^ floors) = 16 := by
  sorry

end student_walks_fifth_to_first_l409_40968


namespace decagon_diagonal_intersection_probability_l409_40935

def probability_intersect_within_decagon : ℚ :=
  let total_vertices := 10
  let total_pairs_points := Nat.choose total_vertices 2
  let total_diagonals := total_pairs_points - total_vertices
  let ways_to_pick_2_diagonals := Nat.choose total_diagonals 2
  let combinations_4_vertices := Nat.choose total_vertices 4
  (combinations_4_vertices : ℚ) / (ways_to_pick_2_diagonals : ℚ)

theorem decagon_diagonal_intersection_probability :
  probability_intersect_within_decagon = 42 / 119 :=
sorry

end decagon_diagonal_intersection_probability_l409_40935


namespace problem_1_problem_2_l409_40939

noncomputable def f (ω x : ℝ) : ℝ :=
  Real.sin (ω * x + (Real.pi / 4))

theorem problem_1 (ω : ℝ) (hω : ω > 0) : f ω 0 = Real.sqrt 2 / 2 :=
by
  unfold f
  simp [Real.sin_pi_div_four]

theorem problem_2 : 
  ∃ x : ℝ, 
  0 ≤ x ∧ x ≤ Real.pi / 2 ∧ 
  (∀ y : ℝ, 0 ≤ y ∧ y ≤ Real.pi / 2 → f 2 y ≤ f 2 x) ∧ 
  f 2 x = 1 :=
by
  sorry

end problem_1_problem_2_l409_40939


namespace applicants_less_4_years_no_degree_l409_40927

theorem applicants_less_4_years_no_degree
    (total_applicants : ℕ)
    (A : ℕ) 
    (B : ℕ)
    (C : ℕ)
    (D : ℕ)
    (h_total : total_applicants = 30)
    (h_A : A = 10)
    (h_B : B = 18)
    (h_C : C = 9)
    (h_D : total_applicants - (A - C + B - C + C) = D) :
  D = 11 :=
by
  sorry

end applicants_less_4_years_no_degree_l409_40927


namespace essay_count_problem_l409_40945

noncomputable def eighth_essays : ℕ := sorry
noncomputable def seventh_essays : ℕ := sorry

theorem essay_count_problem (x : ℕ) (h1 : eighth_essays = x) (h2 : seventh_essays = (1/2 : ℚ) * x - 2) (h3 : eighth_essays + seventh_essays = 118) : 
  seventh_essays = 38 :=
sorry

end essay_count_problem_l409_40945


namespace longer_string_length_l409_40915

theorem longer_string_length 
  (total_length : ℕ) 
  (length_diff : ℕ)
  (h_total_length : total_length = 348)
  (h_length_diff : length_diff = 72) :
  ∃ (L S : ℕ), 
  L - S = length_diff ∧
  L + S = total_length ∧ 
  L = 210 :=
by
  sorry

end longer_string_length_l409_40915


namespace tangent_line_of_circle_l409_40996
-- Import the required libraries

-- Define the given condition of the circle in polar coordinates
def polar_circle (rho theta : ℝ) : Prop :=
  rho = 4 * Real.cos theta

-- Define the property of the tangent line in polar coordinates
def tangent_line (rho theta : ℝ) : Prop :=
  rho * Real.cos theta = 4

-- State the theorem to be proven
theorem tangent_line_of_circle (rho theta : ℝ) (h : polar_circle rho theta) :
  tangent_line rho theta :=
sorry

end tangent_line_of_circle_l409_40996


namespace probability_of_earning_1900_equals_6_over_125_l409_40917

-- Representation of a slot on the spinner.
inductive Slot
| Bankrupt 
| Dollar1000
| Dollar500
| Dollar4000
| Dollar400 
deriving DecidableEq

-- Condition: There are 5 slots and each has the same probability.
noncomputable def slots := [Slot.Bankrupt, Slot.Dollar1000, Slot.Dollar500, Slot.Dollar4000, Slot.Dollar400]

-- Probability of earning exactly $1900 in three spins.
def probability_of_1900 : ℚ :=
  let target_combination := [Slot.Dollar500, Slot.Dollar400, Slot.Dollar1000]
  let total_ways := 125
  let successful_ways := 6
  (successful_ways : ℚ) / (total_ways : ℚ)

theorem probability_of_earning_1900_equals_6_over_125 :
  probability_of_1900 = 6 / 125 :=
sorry

end probability_of_earning_1900_equals_6_over_125_l409_40917


namespace math_problem_l409_40909

theorem math_problem (x y : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y + x + y = 83) (h4 : x^2 * y + x * y^2 = 1056) :
  x^2 + y^2 = 458 := by 
  sorry

end math_problem_l409_40909


namespace range_of_T_l409_40907

open Real

theorem range_of_T (x y z : ℝ) (h : x^2 + 2 * y^2 + 3 * z^2 = 4) : 
    - (2 * sqrt 6) / 3 ≤ x * y + y * z ∧ x * y + y * z ≤ (2 * sqrt 6) / 3 := 
by 
    sorry

end range_of_T_l409_40907


namespace calculate_expression_l409_40954

theorem calculate_expression : (-1:ℝ)^2 + (1/3:ℝ)^0 = 2 := by
  sorry

end calculate_expression_l409_40954


namespace average_of_distinct_u_l409_40962

theorem average_of_distinct_u :
  let u_values := { u : ℕ | ∃ (r_1 r_2 : ℕ), r_1 + r_2 = 6 ∧ r_1 * r_2 = u }
  u_values = {5, 8, 9} ∧ (5 + 8 + 9) / 3 = 22 / 3 :=
by
  sorry

end average_of_distinct_u_l409_40962


namespace white_washing_cost_l409_40956

theorem white_washing_cost
    (length width height : ℝ)
    (door_width door_height window_width window_height : ℝ)
    (num_doors num_windows : ℝ)
    (paint_cost : ℝ)
    (extra_paint_fraction : ℝ)
    (perimeter := 2 * (length + width))
    (door_area := num_doors * (door_width * door_height))
    (window_area := num_windows * (window_width * window_height))
    (wall_area := perimeter * height)
    (paint_area := wall_area - door_area - window_area)
    (total_area := paint_area * (1 + extra_paint_fraction))
    : total_area * paint_cost = 6652.8 :=
by sorry

end white_washing_cost_l409_40956


namespace recurring_decimal_sum_l409_40931

theorem recurring_decimal_sum :
  let x := (4 / 33)
  let y := (34 / 99)
  x + y = (46 / 99) := by
  sorry

end recurring_decimal_sum_l409_40931


namespace common_ratio_of_geometric_sequence_l409_40903

variable {a_n : ℕ → ℝ}
variable {b_n : ℕ → ℝ}
variable {d : ℝ}

-- Conditions
def is_arithmetic_sequence (a_n : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a_n (n + 1) = a_n n + d

def is_geometric_sequence (b_n : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, b_n (n + 1) = b_n n * r

def arithmetic_to_geometric (a_n : ℕ → ℝ) (b_n : ℕ → ℝ) : Prop :=
  b_n 0 = a_n 2 ∧ b_n 1 = a_n 3 ∧ b_n 2 = a_n 7

-- Mathematical Proof Problem
theorem common_ratio_of_geometric_sequence :
  ∀ (a_n : ℕ → ℝ) (d : ℝ), d ≠ 0 →
  is_arithmetic_sequence a_n d →
  (∃ (b_n : ℕ → ℝ) (r : ℝ), arithmetic_to_geometric a_n b_n ∧ is_geometric_sequence b_n r) →
  ∃ r, r = 4 :=
sorry

end common_ratio_of_geometric_sequence_l409_40903


namespace smallest_prime_with_digit_sum_23_l409_40901

def digits_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem smallest_prime_with_digit_sum_23 : 
  ∃ p, Nat.Prime p ∧ digits_sum p = 23 ∧ ∀ q, Nat.Prime q ∧ digits_sum q = 23 → p ≤ q :=
sorry

end smallest_prime_with_digit_sum_23_l409_40901


namespace arrange_logs_in_order_l409_40989

noncomputable def a : ℝ := Real.log 0.3 / Real.log 0.2
noncomputable def b : ℝ := Real.log 0.8 / Real.log 1.2
noncomputable def c : ℝ := Real.sqrt 1.5

theorem arrange_logs_in_order : b < a ∧ a < c := by
  sorry

end arrange_logs_in_order_l409_40989


namespace prob_t_prob_vowel_l409_40980

def word := "mathematics"
def total_letters : ℕ := 11
def t_count : ℕ := 2
def vowel_count : ℕ := 4

-- Definition of being a letter "t"
def is_t (c : Char) : Prop := c = 't'

-- Definition of being a vowel
def is_vowel (c : Char) : Prop := c = 'a' ∨ c = 'e' ∨ c = 'i'

theorem prob_t : (t_count : ℚ) / total_letters = 2 / 11 :=
by
  sorry

theorem prob_vowel : (vowel_count : ℚ) / total_letters = 4 / 11 :=
by
  sorry

end prob_t_prob_vowel_l409_40980


namespace cricket_bat_selling_price_l409_40943

theorem cricket_bat_selling_price
    (profit : ℝ)
    (profit_percentage : ℝ)
    (CP : ℝ)
    (SP : ℝ)
    (h_profit : profit = 255)
    (h_profit_percentage : profit_percentage = 42.857142857142854)
    (h_CP : CP = 255 * 100 / 42.857142857142854)
    (h_SP : SP = CP + profit) :
    SP = 850 :=
by
  skip -- This is where the proof would go
  sorry -- Placeholder for the required proof

end cricket_bat_selling_price_l409_40943


namespace cubic_polynomial_greater_than_zero_l409_40994

theorem cubic_polynomial_greater_than_zero (x : ℝ) : x^3 + 3*x^2 + x - 5 > 0 → x > 1 :=
sorry

end cubic_polynomial_greater_than_zero_l409_40994


namespace find_divisor_l409_40999

theorem find_divisor (d : ℕ) (h1 : 127 = d * 5 + 2) : d = 25 :=
sorry

end find_divisor_l409_40999


namespace percentage_reduction_in_price_l409_40948

variable (R P : ℝ) (R_eq : R = 30) (H : 600 / R - 600 / P = 4)

theorem percentage_reduction_in_price (R_eq : R = 30) (H : 600 / R - 600 / P = 4) :
  ((P - R) / P) * 100 = 20 := sorry

end percentage_reduction_in_price_l409_40948


namespace compare_neg_fractions_l409_40946

theorem compare_neg_fractions : - (2 / 3 : ℝ) > - (3 / 4 : ℝ) :=
sorry

end compare_neg_fractions_l409_40946


namespace no_a_where_A_eq_B_singleton_l409_40998

def f (a x : ℝ) := x^2 + 4 * x - 2 * a
def g (a x : ℝ) := x^2 - a * x + a + 3

theorem no_a_where_A_eq_B_singleton :
  ∀ a : ℝ,
    (∃ x₁ : ℝ, (f a x₁ ≤ 0 ∧ ∀ x₂, f a x₂ ≤ 0 → x₂ = x₁)) ∧
    (∃ y₁ : ℝ, (g a y₁ ≤ 0 ∧ ∀ y₂, g a y₂ ≤ 0 → y₂ = y₁)) →
    (¬ ∃ z : ℝ, (f a z ≤ 0) ∧ (g a z ≤ 0)) := 
by
  sorry

end no_a_where_A_eq_B_singleton_l409_40998


namespace sufficient_not_necessary_condition_l409_40988

theorem sufficient_not_necessary_condition (x : ℝ) (a : ℝ) (h₁ : -1 ≤ x ∧ x ≤ 2) : a > 4 → x^2 - a ≤ 0 :=
by
  sorry

end sufficient_not_necessary_condition_l409_40988


namespace team_B_score_third_game_l409_40933

theorem team_B_score_third_game (avg_points : ℝ) (additional_needed : ℝ) (total_target : ℝ) (P : ℝ) :
  avg_points = 61.5 → additional_needed = 330 → total_target = 500 →
  2 * avg_points + P + additional_needed = total_target → P = 47 :=
by
  intros avg_points_eq additional_needed_eq total_target_eq total_eq
  rw [avg_points_eq, additional_needed_eq, total_target_eq] at total_eq
  sorry

end team_B_score_third_game_l409_40933


namespace solutions_eq1_solutions_eq2_l409_40916

noncomputable def equation_sol1 : Set ℝ :=
{ x | x^2 - 8 * x + 1 = 0 }

noncomputable def equation_sol2 : Set ℝ :=
{ x | x * (x - 2) - x + 2 = 0 }

theorem solutions_eq1 : ∀ x ∈ equation_sol1, x = 4 + Real.sqrt 15 ∨ x = 4 - Real.sqrt 15 :=
by
  intro x hx
  sorry

theorem solutions_eq2 : ∀ x ∈ equation_sol2, x = 2 ∨ x = 1 :=
by
  intro x hx
  sorry

end solutions_eq1_solutions_eq2_l409_40916


namespace inequality_sqrt_ab_l409_40958

theorem inequality_sqrt_ab {a b : ℝ} (ha : a > 0) (hb : b > 0) (hab : a > b) :
  (a - b)^2 / (8 * a) < (a + b) / 2 - Real.sqrt (a * b) ∧ (a + b) / 2 - Real.sqrt (a * b) < (a - b)^2 / (8 * b) := 
sorry

end inequality_sqrt_ab_l409_40958


namespace time_taken_by_A_l409_40984

theorem time_taken_by_A (t : ℚ) (h1 : 3 * (t + 1 / 2) = 4 * t) : t = 3 / 2 ∧ (t + 1 / 2) = 2 := 
  by
  intros
  sorry

end time_taken_by_A_l409_40984


namespace car_speed_constant_l409_40949

theorem car_speed_constant (v : ℝ) (hv : v ≠ 0)
  (condition_1 : (1 / 36) * 3600 = 100) 
  (condition_2 : (1 / v) * 3600 = 120) :
  v = 30 := by
  sorry

end car_speed_constant_l409_40949


namespace smallest_prime_less_than_square_l409_40991

theorem smallest_prime_less_than_square : 
  ∃ (p : ℕ) (n : ℕ), p = 13 ∧ Prime p ∧ p = n^2 - 12 ∧ 0 < p ∧ ∀ q, (Prime q ∧ ∃ m, q = m^2 - 12 ∧ 0 < q  → q ≥ p) := by
  sorry

end smallest_prime_less_than_square_l409_40991


namespace flower_bed_area_l409_40936

noncomputable def area_of_triangle (a b c : ℝ) (h : a^2 + b^2 = c^2) : ℝ :=
  (1/2) * a * b

theorem flower_bed_area : 
  area_of_triangle 6 8 10 (by norm_num) = 24 := 
sorry

end flower_bed_area_l409_40936


namespace smallest_perfect_square_divisible_by_2_and_5_l409_40981

theorem smallest_perfect_square_divisible_by_2_and_5 :
  ∃ n : ℕ, 0 < n ∧ (∃ k : ℕ, n = k ^ 2) ∧ (n % 2 = 0) ∧ (n % 5 = 0) ∧ 
  (∀ m : ℕ, 0 < m ∧ (∃ k : ℕ, m = k ^ 2) ∧ (m % 2 = 0) ∧ (m % 5 = 0) → n ≤ m) :=
sorry

end smallest_perfect_square_divisible_by_2_and_5_l409_40981


namespace isosceles_triangle_area_l409_40911

theorem isosceles_triangle_area (s b : ℝ) (h₁ : s + b = 20) (h₂ : b^2 + 10^2 = s^2) : 
  1/2 * 2 * b * 10 = 75 :=
by sorry

end isosceles_triangle_area_l409_40911


namespace solve_equation_l409_40976

theorem solve_equation (x : ℝ) (h : (x - 7) / 2 - (1 + x) / 3 = 1) : x = 29 :=
sorry

end solve_equation_l409_40976


namespace ab5_a2_c5_a2_inequality_l409_40955

theorem ab5_a2_c5_a2_inequality 
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a ^ 5 - a ^ 2 + 3) * (b ^ 5 - b ^ 2 + 3) * (c ^ 5 - c ^ 2 + 3) ≥ (a + b + c) ^ 3 := 
by
  sorry

end ab5_a2_c5_a2_inequality_l409_40955


namespace largest_number_among_selected_students_l409_40963

def total_students := 80

def smallest_numbers (x y : ℕ) : Prop :=
  x = 6 ∧ y = 14

noncomputable def selected_students (n : ℕ) : ℕ :=
  6 + (n - 1) * 8

theorem largest_number_among_selected_students :
  ∀ (x y : ℕ), smallest_numbers x y → (selected_students 10 = 78) :=
by
  intros x y h
  rw [smallest_numbers] at h
  have h1 : x = 6 := h.1
  have h2 : y = 14 := h.2
  exact rfl

#check largest_number_among_selected_students

end largest_number_among_selected_students_l409_40963


namespace sum_of_numbers_l409_40997

theorem sum_of_numbers (x y z : ℝ) (h1 : x ≤ y) (h2 : y ≤ z) 
  (h3 : y = 5) (h4 : (x + y + z) / 3 = x + 10) (h5 : (x + y + z) / 3 = z - 15) : 
  x + y + z = 30 := 
by 
  sorry

end sum_of_numbers_l409_40997


namespace available_floor_space_equals_110_sqft_l409_40977

-- Definitions for the conditions
def tile_side_in_feet : ℝ := 0.5
def width_main_section_tiles : ℕ := 15
def length_main_section_tiles : ℕ := 25
def width_alcove_tiles : ℕ := 10
def depth_alcove_tiles : ℕ := 8
def width_pillar_tiles : ℕ := 3
def length_pillar_tiles : ℕ := 5

-- Conversion of tiles to feet
def width_main_section_feet : ℝ := width_main_section_tiles * tile_side_in_feet
def length_main_section_feet : ℝ := length_main_section_tiles * tile_side_in_feet
def width_alcove_feet : ℝ := width_alcove_tiles * tile_side_in_feet
def depth_alcove_feet : ℝ := depth_alcove_tiles * tile_side_in_feet
def width_pillar_feet : ℝ := width_pillar_tiles * tile_side_in_feet
def length_pillar_feet : ℝ := length_pillar_tiles * tile_side_in_feet

-- Area calculations
def area_main_section : ℝ := width_main_section_feet * length_main_section_feet
def area_alcove : ℝ := width_alcove_feet * depth_alcove_feet
def total_area : ℝ := area_main_section + area_alcove
def area_pillar : ℝ := width_pillar_feet * length_pillar_feet
def available_floor_space : ℝ := total_area - area_pillar

-- Proof statement
theorem available_floor_space_equals_110_sqft 
  (h1 : width_main_section_feet = width_main_section_tiles * tile_side_in_feet)
  (h2 : length_main_section_feet = length_main_section_tiles * tile_side_in_feet)
  (h3 : width_alcove_feet = width_alcove_tiles * tile_side_in_feet)
  (h4 : depth_alcove_feet = depth_alcove_tiles * tile_side_in_feet)
  (h5 : width_pillar_feet = width_pillar_tiles * tile_side_in_feet)
  (h6 : length_pillar_feet = length_pillar_tiles * tile_side_in_feet) 
  (h7 : area_main_section = width_main_section_feet * length_main_section_feet)
  (h8 : area_alcove = width_alcove_feet * depth_alcove_feet)
  (h9 : total_area = area_main_section + area_alcove)
  (h10 : area_pillar = width_pillar_feet * length_pillar_feet)
  (h11 : available_floor_space = total_area - area_pillar) : 
  available_floor_space = 110 := 
by 
  sorry

end available_floor_space_equals_110_sqft_l409_40977


namespace fred_red_marbles_l409_40983

theorem fred_red_marbles (total_marbles : ℕ) (dark_blue_marbles : ℕ) (green_marbles : ℕ) (red_marbles : ℕ) 
  (h1 : total_marbles = 63) 
  (h2 : dark_blue_marbles ≥ total_marbles / 3)
  (h3 : green_marbles = 4)
  (h4 : red_marbles = total_marbles - dark_blue_marbles - green_marbles) : 
  red_marbles = 38 := 
sorry

end fred_red_marbles_l409_40983


namespace total_flight_time_l409_40982

theorem total_flight_time
  (distance : ℕ)
  (speed_out : ℕ)
  (speed_return : ℕ)
  (time_out : ℕ)
  (time_return : ℕ)
  (total_time : ℕ)
  (h1 : distance = 1500)
  (h2 : speed_out = 300)
  (h3 : speed_return = 500)
  (h4 : time_out = distance / speed_out)
  (h5 : time_return = distance / speed_return)
  (h6 : total_time = time_out + time_return) :
  total_time = 8 := 
  by {
    sorry
  }

end total_flight_time_l409_40982


namespace Juanita_spends_more_l409_40913

def Grant_annual_spend : ℝ := 200
def weekday_spend : ℝ := 0.50
def sunday_spend : ℝ := 2.00
def weeks_in_year : ℝ := 52

def Juanita_weekly_spend : ℝ := (6 * weekday_spend) + sunday_spend
def Juanita_annual_spend : ℝ := weeks_in_year * Juanita_weekly_spend
def spending_difference : ℝ := Juanita_annual_spend - Grant_annual_spend

theorem Juanita_spends_more : spending_difference = 60 := by
  sorry

end Juanita_spends_more_l409_40913


namespace trick_deck_cost_l409_40929

theorem trick_deck_cost (x : ℝ) (h1 : 6 * x + 2 * x = 64) : x = 8 :=
  sorry

end trick_deck_cost_l409_40929


namespace sum_reciprocals_transformed_roots_l409_40908

theorem sum_reciprocals_transformed_roots (a b c : ℝ) (h : ∀ x, (x^3 - 2 * x - 5 = 0) → (x = a) ∨ (x = b) ∨ (x = c)) : 
  (1 / (a - 2)) + (1 / (b - 2)) + (1 / (c - 2)) = 10 := 
by sorry

end sum_reciprocals_transformed_roots_l409_40908


namespace angle_between_vectors_l409_40973

open Real

noncomputable def vector_norm (v : ℝ × ℝ) : ℝ := sqrt (v.1 * v.1 + v.2 * v.2)

noncomputable def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

theorem angle_between_vectors
  (a b : ℝ × ℝ)
  (h₁ : vector_norm a ≠ 0)
  (h₂ : vector_norm b ≠ 0)
  (h₃ : vector_norm a = vector_norm b)
  (h₄ : vector_norm a = vector_norm (a.1 + 2 * b.1, a.2 + 2 * b.2)) :
  ∃ θ : ℝ, θ = 180 ∧ cos θ = -1 := 
sorry

end angle_between_vectors_l409_40973


namespace hyperbola_equation_l409_40924

-- Define the hyperbola with vertices and other conditions
def Hyperbola (a b : ℝ) (h : a > 0 ∧ b > 0) : Prop :=
  ∀ (x y : ℝ), (x^2 / a^2 - y^2 / b^2 = 1)

-- Given conditions and the proof goal
theorem hyperbola_equation
  (a b : ℝ) (h : a > 0 ∧ b > 0)
  (P : ℝ × ℝ) (M N : ℝ × ℝ)
  (k_PA k_PB : ℝ)
  (PA_PB_condition : k_PA * k_PB = 3)
  (MN_min_value : |(M.1 - N.1) + (M.2 - N.2)| = 4) :
  Hyperbola a b h →
  (a = 2 ∧ b = 2 * Real.sqrt 3 ∧ (∀ (x y : ℝ), (x^2 / 4 - y^2 / 12 = 1)) ∨ 
   a = 2 / 3 ∧ b = 2 * Real.sqrt 3 / 3 ∧ (∀ (x y : ℝ), (9 * x^2 / 4 - 3 * y^2 / 4 = 1)))
:=
sorry

end hyperbola_equation_l409_40924


namespace probability_of_four_odd_slips_l409_40971

-- Define the conditions
def number_of_slips : ℕ := 10
def odd_slips : ℕ := 5
def even_slips : ℕ := 5
def slips_drawn : ℕ := 4

-- Define the required probability calculation
def probability_four_odd_slips : ℚ := (5 / 10) * (4 / 9) * (3 / 8) * (2 / 7)

-- State the theorem we want to prove
theorem probability_of_four_odd_slips :
  probability_four_odd_slips = 1 / 42 :=
by
  sorry

end probability_of_four_odd_slips_l409_40971


namespace baba_yaga_departure_and_speed_l409_40904

variables (T : ℕ) (d : ℕ)

theorem baba_yaga_departure_and_speed :
  (50 * (T + 2) = 150 * (T - 2)) →
  (12 - T = 8) ∧ (d = 50 * (T + 2)) →
  (d = 300) ∧ ((d / T) = 75) :=
by
  intros h1 h2
  sorry

end baba_yaga_departure_and_speed_l409_40904


namespace translate_graph_upwards_l409_40953

theorem translate_graph_upwards (x : ℝ) :
  (∀ x, (3*x - 1) + 3 = 3*x + 2) :=
by
  intro x
  sorry

end translate_graph_upwards_l409_40953


namespace multiplication_correct_l409_40910

theorem multiplication_correct : 121 * 54 = 6534 := by
  sorry

end multiplication_correct_l409_40910


namespace geom_inequality_l409_40995

variables {Point : Type} [MetricSpace Point] {O A B C K L H M : Point}

/-- Conditions -/
def circumcenter_of_triangle (O A B C : Point) : Prop := 
 -- Definition that O is the circumcenter of triangle ABC
 sorry 

def midpoint_of_arc (K B C A : Point) : Prop := 
 -- Definition that K is the midpoint of the arc BC not containing A
 sorry

def lies_on_line (K L A : Point) : Prop := 
 -- Definition that K lies on line AL
 sorry

def similar_triangles (A H L K M : Point) : Prop := 
 -- Definition that triangles AHL and KML are similar
 sorry 

def segment_inequality (AL KL : ℝ) : Prop := 
 -- Definition that AL < KL
 sorry 

/-- Proof Problem -/
theorem geom_inequality (h1 : circumcenter_of_triangle O A B C) 
                       (h2: midpoint_of_arc K B C A)
                       (h3: lies_on_line K L A)
                       (h4: similar_triangles A H L K M)
                       (h5: segment_inequality (dist A L) (dist K L)) : 
  dist A K < dist B C := 
sorry

end geom_inequality_l409_40995


namespace student_marks_l409_40941

variable (max_marks : ℕ) (pass_percent : ℕ) (fail_by : ℕ)

theorem student_marks
  (h_max : max_marks = 400)
  (h_pass : pass_percent = 35)
  (h_fail : fail_by = 40)
  : max_marks * pass_percent / 100 - fail_by = 100 :=
by
  sorry

end student_marks_l409_40941


namespace proof_intersection_l409_40957

def setA : Set ℤ := {x | abs x ≤ 2}

def setB : Set ℝ := {x | x^2 - 2 * x - 8 ≥ 0}

def complementB : Set ℝ := {x | x^2 - 2 * x - 8 < 0}

def intersectionAComplementB : Set ℤ := {x | x ∈ setA ∧ (x : ℝ) ∈ complementB}

theorem proof_intersection : intersectionAComplementB = {-1, 0, 1, 2} := by
  sorry

end proof_intersection_l409_40957


namespace age_transition_l409_40960

theorem age_transition (initial_ages : List ℕ) : 
  initial_ages = [19, 34, 37, 42, 48] →
  (∃ x, 0 < x ∧ x < 10 ∧ 
  new_ages = List.map (fun age => age + x) initial_ages ∧ 
  new_ages = [25, 40, 43, 48, 54]) →
  x = 6 :=
by
  intros h_initial_ages h_exist_x
  sorry

end age_transition_l409_40960


namespace lunch_cost_calc_l409_40985

-- Define the given conditions
def gasoline_cost : ℝ := 8
def gift_cost : ℝ := 5
def grandma_gift : ℝ := 10
def initial_money : ℝ := 50
def return_trip_money : ℝ := 36.35

-- Calculate the total expenses and determine the money spent on lunch
def total_gifts_cost : ℝ := 2 * gift_cost
def total_money_received : ℝ := initial_money + 2 * grandma_gift
def total_gas_gift_cost : ℝ := gasoline_cost + total_gifts_cost
def expected_remaining_money : ℝ := total_money_received - total_gas_gift_cost
def lunch_cost : ℝ := expected_remaining_money - return_trip_money

-- State theorem
theorem lunch_cost_calc : lunch_cost = 15.65 := by
  sorry

end lunch_cost_calc_l409_40985


namespace find_Minchos_chocolate_l409_40932

variable (M : ℕ)  -- Define M as a natural number

-- Define the conditions as Lean hypotheses
def TaeminChocolate := 5 * M
def KibumChocolate := 3 * M
def TotalChocolate := TaeminChocolate M + KibumChocolate M

theorem find_Minchos_chocolate (h : TotalChocolate M = 160) : M = 20 :=
by
  sorry

end find_Minchos_chocolate_l409_40932


namespace yards_mowed_by_christian_l409_40970

-- Definitions based on the provided conditions
def initial_savings := 5 + 7
def sue_earnings := 6 * 2
def total_savings := initial_savings + sue_earnings
def additional_needed := 50 - total_savings
def short_amount := 6
def christian_earnings := additional_needed - short_amount
def charge_per_yard := 5

theorem yards_mowed_by_christian : 
  (christian_earnings / charge_per_yard) = 4 :=
by
  sorry

end yards_mowed_by_christian_l409_40970


namespace number_of_girls_l409_40920

theorem number_of_girls (d c : ℕ) (h1 : c = 2 * (d - 15)) (h2 : d - 15 = 5 * (c - 45)) : d = 40 := 
by
  sorry

end number_of_girls_l409_40920


namespace fraction_books_sold_l409_40926

theorem fraction_books_sold (B : ℕ) (F : ℚ) (h1 : 36 = B - F * B) (h2 : 252 = 3.50 * F * B) : F = 2 / 3 := by
  -- Proof omitted
  sorry

end fraction_books_sold_l409_40926


namespace range_of_m_l409_40938

noncomputable def function_even_and_monotonic (f : ℝ → ℝ) := 
  (∀ x : ℝ, f x = f (-x)) ∧ (∀ x y : ℝ, 0 ≤ x → 0 ≤ y → x < y → f x > f y)

variable (f : ℝ → ℝ)
variable (m : ℝ)

theorem range_of_m (h₁ : function_even_and_monotonic f) 
  (h₂ : f m > f (1 - m)) : m < 1 / 2 :=
sorry

end range_of_m_l409_40938


namespace intersection_is_correct_l409_40944

def setA : Set ℕ := {0, 1, 2}
def setB : Set ℕ := {1, 2, 3}

theorem intersection_is_correct : setA ∩ setB = {1, 2} := by
  sorry

end intersection_is_correct_l409_40944


namespace divisor_of_z_in_form_4n_minus_1_l409_40952

theorem divisor_of_z_in_form_4n_minus_1
  (x y : ℕ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (z : ℕ) 
  (hz : z = 4 * x * y / (x + y)) 
  (hz_odd : z % 2 = 1) :
  ∃ n : ℕ, n > 0 ∧ ∃ d : ℕ, d ∣ z ∧ d = 4 * n - 1 :=
sorry

end divisor_of_z_in_form_4n_minus_1_l409_40952


namespace a_in_A_l409_40906

def A := {x : ℝ | x ≥ 2 * Real.sqrt 2}
def a : ℝ := 3

theorem a_in_A : a ∈ A :=
by 
  sorry

end a_in_A_l409_40906


namespace part_a_l409_40922

def f_X (X : Set (ℝ × ℝ)) (n : ℕ) : ℝ :=
  sorry  -- Placeholder for the largest possible area function

theorem part_a (X : Set (ℝ × ℝ)) (m n : ℕ) (h1 : m ≥ n) (h2 : n > 2) :
  f_X X m + f_X X n ≥ f_X X (m + 1) + f_X X (n - 1) :=
sorry

end part_a_l409_40922


namespace max_value_of_reciprocal_sums_of_zeros_l409_40905

noncomputable def quadratic_part (k : ℝ) (x : ℝ) : ℝ :=
  k * x^2 + 2 * x - 1

noncomputable def linear_part (k : ℝ) (x : ℝ) : ℝ :=
  k * x + 1

theorem max_value_of_reciprocal_sums_of_zeros (k : ℝ) (x1 x2 : ℝ)
  (h0 : -1 < k ∧ k < 0)
  (hx1 : x1 ∈ Set.Ioc 0 1 → quadratic_part k x1 = 0)
  (hx2 : x2 ∈ Set.Ioi 1 → linear_part k x2 = 0)
  (hx_distinct : x1 ≠ x2) :
  (1 / x1) + (1 / x2) = 9 / 4 :=
sorry

end max_value_of_reciprocal_sums_of_zeros_l409_40905


namespace plane_equation_l409_40940

theorem plane_equation (x y z : ℝ) (A B C D : ℤ) (h1 : A = 9) (h2 : B = -6) (h3 : C = 4) (h4 : D = -133) (A_pos : A > 0) (gcd_condition : Int.gcd (Int.natAbs A) (Int.gcd (Int.natAbs B) (Int.gcd (Int.natAbs C) (Int.natAbs D))) = 1) : 
  A * x + B * y + C * z + D = 0 :=
sorry

end plane_equation_l409_40940


namespace intersection_of_sets_l409_40972

theorem intersection_of_sets (M : Set ℤ) (N : Set ℤ) (H_M : M = {0, 1, 2, 3, 4}) (H_N : N = {-2, 0, 2}) :
  M ∩ N = {0, 2} :=
by
  rw [H_M, H_N]
  ext
  simp
  sorry  -- Proof to be filled in

end intersection_of_sets_l409_40972


namespace exists_sum_of_squares_form_l409_40992

theorem exists_sum_of_squares_form (n : ℕ) (h : n % 25 = 9) :
  ∃ (a b c : ℕ), n = (a * (a + 1)) / 2 + (b * (b + 1)) / 2 + (c * (c + 1)) / 2 := 
by 
  sorry

end exists_sum_of_squares_form_l409_40992


namespace discount_of_bag_l409_40975

def discounted_price (marked_price discount_rate : ℕ) : ℕ :=
  marked_price - ((discount_rate * marked_price) / 100)

theorem discount_of_bag : discounted_price 200 40 = 120 :=
by
  unfold discounted_price
  norm_num

end discount_of_bag_l409_40975


namespace prove_0_in_A_prove_13_in_A_prove_74_in_A_prove_A_is_Z_l409_40950

variable {A : Set Int}

-- Assuming set A is closed under subtraction
axiom A_closed_under_subtraction : ∀ x y, x ∈ A → y ∈ A → x - y ∈ A
axiom A_contains_4 : 4 ∈ A
axiom A_contains_9 : 9 ∈ A

theorem prove_0_in_A : 0 ∈ A :=
sorry

theorem prove_13_in_A : 13 ∈ A :=
sorry

theorem prove_74_in_A : 74 ∈ A :=
sorry

theorem prove_A_is_Z : A = Set.univ :=
sorry

end prove_0_in_A_prove_13_in_A_prove_74_in_A_prove_A_is_Z_l409_40950


namespace quadratic_relationship_l409_40947

variable (y_1 y_2 y_3 : ℝ)

-- Conditions
def vertex := (-2, 1)
def opens_downwards := true
def intersects_x_axis_at_two_points := true
def passes_through_points := [(1, y_1), (-1, y_2), (-4, y_3)]

-- Proof statement
theorem quadratic_relationship : y_1 < y_3 ∧ y_3 < y_2 :=
  sorry

end quadratic_relationship_l409_40947


namespace probability_none_solve_l409_40918

theorem probability_none_solve (a b c : ℕ) (ha : 0 < a ∧ a < 10)
                               (hb : 0 < b ∧ b < 10)
                               (hc : 0 < c ∧ c < 10)
                               (P_A : ℚ := 1 / a)
                               (P_B : ℚ := 1 / b)
                               (P_C : ℚ := 1 / c)
                               (H : (1 - (1 / a)) * (1 - (1 / b)) * (1 - (1 / c)) = 8 / 15) :
                               -- Conclusion: The probability that none of them solve the problem is 8/15
                               (1 - (1 / a)) * (1 - (1 / b)) * (1 - (1 / c)) = 8 / 15 :=
sorry

end probability_none_solve_l409_40918


namespace smallest_four_digit_divisible_by_55_l409_40986

theorem smallest_four_digit_divisible_by_55 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 55 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 55 = 0 → n ≤ m := by
  sorry

end smallest_four_digit_divisible_by_55_l409_40986


namespace waxberry_problem_l409_40930

noncomputable def batch_cannot_be_sold : ℚ := 1 - (8 / 9 * 9 / 10)

def probability_distribution (X : ℚ) : ℚ := 
  if X = -3200 then (1 / 5)^4 else
  if X = -2000 then 4 * (1 / 5)^3 * (4 / 5) else
  if X = -800 then 6 * (1 / 5)^2 * (4 / 5)^2 else
  if X = 400 then 4 * (1 / 5) * (4 / 5)^3 else
  if X = 1600 then (4 / 5)^4 else 0

noncomputable def expected_profit : ℚ :=
  -3200 * probability_distribution (-3200) +
  -2000 * probability_distribution (-2000) +
  -800 * probability_distribution (-800) +
  400 * probability_distribution (400) +
  1600 * probability_distribution (1600)

theorem waxberry_problem : 
  batch_cannot_be_sold = 1 / 5 ∧ 
  (probability_distribution (-3200) = 1 / 625 ∧ 
   probability_distribution (-2000) = 16 / 625 ∧ 
   probability_distribution (-800) = 96 / 625 ∧ 
   probability_distribution (400) = 256 / 625 ∧ 
   probability_distribution (1600) = 256 / 625) ∧ 
  expected_profit = 640 :=
by 
  sorry

end waxberry_problem_l409_40930


namespace point_symmetric_y_axis_l409_40923

theorem point_symmetric_y_axis (a b : ℤ) (h₁ : a = -(-2)) (h₂ : b = 3) : a + b = 5 := by
  sorry

end point_symmetric_y_axis_l409_40923


namespace point_reflection_xOy_l409_40934

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def reflection_over_xOy (P : Point3D) : Point3D := 
  {x := P.x, y := P.y, z := -P.z}

theorem point_reflection_xOy :
  reflection_over_xOy {x := 1, y := 2, z := 3} = {x := 1, y := 2, z := -3} := by
  sorry

end point_reflection_xOy_l409_40934


namespace selling_price_is_correct_l409_40900

noncomputable def cost_price : ℝ := 192
def profit_percentage : ℝ := 0.25
def profit (cp : ℝ) (pp : ℝ) : ℝ := pp * cp
def selling_price (cp : ℝ) (pft : ℝ) : ℝ := cp + pft

theorem selling_price_is_correct : selling_price cost_price (profit cost_price profit_percentage) = 240 :=
sorry

end selling_price_is_correct_l409_40900


namespace g_of_minus_3_l409_40912

noncomputable def f (x : ℝ) : ℝ := 4 * x - 7
noncomputable def g (y : ℝ) : ℝ := 3 * ((y + 7) / 4) ^ 2 + 4 * ((y + 7) / 4) + 1

theorem g_of_minus_3 : g (-3) = 8 :=
by
  sorry

end g_of_minus_3_l409_40912


namespace fish_tank_problem_l409_40964

def number_of_fish_in_first_tank
  (F : ℕ)          -- Let F represent the number of fish in the first tank
  (twoF : ℕ)       -- Let twoF represent twice the number of fish in the first tank
  (total : ℕ) :    -- Let total represent the total number of fish
  Prop :=
  (2 * F = twoF)  -- The other two tanks each have twice as many fish as the first
  ∧ (F + twoF + twoF = total)  -- The sum of the fish in all three tanks equals the total number of fish

theorem fish_tank_problem
  (F : ℕ)
  (H : number_of_fish_in_first_tank F (2 * F) 100) : F = 20 :=
by
  sorry

end fish_tank_problem_l409_40964


namespace sum_of_numbers_l409_40921

theorem sum_of_numbers (a b c : ℝ) 
  (h₁ : a^2 + b^2 + c^2 = 62) 
  (h₂ : ab + bc + ca = 131) : 
  a + b + c = 18 :=
sorry

end sum_of_numbers_l409_40921


namespace least_clock_equivalent_l409_40990

theorem least_clock_equivalent (t : ℕ) (h : t > 5) : 
  (t^2 - t) % 24 = 0 → t = 9 :=
by
  sorry

end least_clock_equivalent_l409_40990


namespace ff1_is_1_l409_40919

noncomputable def f (x : ℝ) := Real.log x - 2 * x + 3

theorem ff1_is_1 : f (f 1) = 1 := by
  sorry

end ff1_is_1_l409_40919


namespace equation_of_hyperbola_l409_40902

-- Definitions for conditions

def center_at_origin (center : ℝ × ℝ) : Prop :=
  center = (0, 0)

def focus_point (focus : ℝ × ℝ) : Prop :=
  focus = (Real.sqrt 2, 0)

def distance_to_asymptote (focus : ℝ × ℝ) (distance : ℝ) : Prop :=
  -- Placeholder for the actual distance calculation
  distance = 1 -- The given distance condition in the problem

-- The mathematical proof problem statement

theorem equation_of_hyperbola :
  center_at_origin (0,0) ∧
  focus_point (Real.sqrt 2, 0) ∧
  distance_to_asymptote (Real.sqrt 2, 0) 1 → 
    ∃ a b : ℝ, (a > 0) ∧ (b > 0) ∧
    (a^2 + b^2 = 2) ∧ (a^2 = 1) ∧ (b^2 = 1) ∧ 
    (∀ x y : ℝ, b^2*y^2 = x^2 - a^2*y^2 → (y = 0 ∧ x^2 = 1)) :=
sorry

end equation_of_hyperbola_l409_40902


namespace cos2alpha_minus_sin2alpha_l409_40937

theorem cos2alpha_minus_sin2alpha (α : ℝ) (h1 : α ∈ Set.Icc (-π/2) 0) 
  (h2 : (Real.sin (3 * α)) / (Real.sin α) = 13 / 5) :
  Real.cos (2 * α) - Real.sin (2 * α) = (3 + Real.sqrt 91) / 10 :=
sorry

end cos2alpha_minus_sin2alpha_l409_40937


namespace relay_race_time_reduction_l409_40978

theorem relay_race_time_reduction
    (T T1 T2 T3 T4 T5 : ℝ)
    (h1 : T1 = 0.1 * T)
    (h2 : T2 = 0.2 * T)
    (h3 : T3 = 0.24 * T)
    (h4 : T4 = 0.3 * T)
    (h5 : T5 = 0.16 * T) :
    ((T1 + T2 + T3 + T4 + T5) - (T1 + T2 + T3 + T4 + T5 / 2)) / (T1 + T2 + T3 + T4 + T5) = 0.08 :=
by
  sorry

end relay_race_time_reduction_l409_40978


namespace simplify_expression_l409_40959

variable {a b : ℝ}

theorem simplify_expression : (4 * a^3 * b - 2 * a * b) / (2 * a * b) = 2 * a^2 - 1 := by
  sorry

end simplify_expression_l409_40959


namespace total_quarters_l409_40961

def Sara_initial_quarters : Nat := 21
def quarters_given_by_dad : Nat := 49

theorem total_quarters : Sara_initial_quarters + quarters_given_by_dad = 70 := 
by
  sorry

end total_quarters_l409_40961


namespace symmetry_condition_l409_40974

theorem symmetry_condition (p q r s t u : ℤ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) (ht : t ≠ 0) (hu : u ≠ 0)
  (yx_eq : ∀ x y, y = (p * x ^ 2 + q * x + r) / (s * x ^ 2 + t * x + u) ↔ x = (p * y ^ 2 + q * y + r) / (s * y ^ 2 + t * y + u)) :
  p = s ∧ q = t ∧ r = u :=
sorry

end symmetry_condition_l409_40974


namespace election_winning_candidate_votes_l409_40914

theorem election_winning_candidate_votes (V : ℕ) 
  (h1 : V = (4 / 7) * V + 2000 + 4000) : 
  (4 / 7) * V = 8000 :=
by
  sorry

end election_winning_candidate_votes_l409_40914


namespace ones_digit_of_34_34_times_17_17_is_6_l409_40969

def cyclical_pattern_4 (n : ℕ) : ℕ :=
if n % 2 = 0 then 6 else 4

theorem ones_digit_of_34_34_times_17_17_is_6
  (h1 : 34 % 10 = 4)
  (h2 : ∀ n : ℕ, cyclical_pattern_4 n = if n % 2 = 0 then 6 else 4)
  (h3 : 17 % 2 = 1)
  (h4 : (34 * 17^17) % 2 = 0)
  (h5 : ∀ n : ℕ, cyclical_pattern_4 n = if n % 2 = 0 then 6 else 4) :
  (34^(34 * 17^17)) % 10 = 6 := 
by  
  sorry

end ones_digit_of_34_34_times_17_17_is_6_l409_40969


namespace cost_of_siding_l409_40987

def area_of_wall (length : ℕ) (width : ℕ) : ℕ :=
  length * width

def area_of_roof (length : ℕ) (width : ℕ) : ℕ :=
  2 * (length * width)

def area_of_sheet (length : ℕ) (width : ℕ) : ℕ :=
  length * width

def sheets_needed (total_area : ℕ) (sheet_area : ℕ) : ℕ :=
  (total_area + sheet_area - 1) / sheet_area  -- Cooling the ceiling with integer arithmetic

def total_cost (sheets : ℕ) (price_per_sheet : ℕ) : ℕ :=
  sheets * price_per_sheet

theorem cost_of_siding : 
  ∀ (length_wall width_wall length_roof width_roof length_sheet width_sheet price_per_sheet : ℕ),
  length_wall = 10 → width_wall = 7 →
  length_roof = 10 → width_roof = 6 →
  length_sheet = 10 → width_sheet = 14 →
  price_per_sheet = 50 →
  total_cost (sheets_needed (area_of_wall length_wall width_wall + area_of_roof length_roof width_roof) (area_of_sheet length_sheet width_sheet)) price_per_sheet = 100 :=
by
  intros
  simp [area_of_wall, area_of_roof, area_of_sheet, sheets_needed, total_cost]
  sorry

end cost_of_siding_l409_40987


namespace jen_triple_flips_l409_40979

-- Definitions based on conditions
def tyler_double_flips : ℕ := 12
def flips_per_double_flip : ℕ := 2
def flips_by_tyler : ℕ := tyler_double_flips * flips_per_double_flip
def flips_ratio : ℕ := 2
def flips_per_triple_flip : ℕ := 3
def flips_by_jen : ℕ := flips_by_tyler * flips_ratio

-- Lean 4 statement
theorem jen_triple_flips : flips_by_jen / flips_per_triple_flip = 16 :=
by 
    -- Proof contents should go here. We only need the statement as per the instruction.
    sorry

end jen_triple_flips_l409_40979


namespace vendor_second_day_sale_l409_40928

theorem vendor_second_day_sale (n : ℕ) :
  let sold_first_day := (50 * n) / 100
  let remaining_after_first_sale := n - sold_first_day
  let thrown_away_first_day := (20 * remaining_after_first_sale) / 100
  let remaining_after_first_day := remaining_after_first_sale - thrown_away_first_day
  let total_thrown_away := (30 * n) / 100
  let thrown_away_second_day := total_thrown_away - thrown_away_first_day
  let sold_second_day := remaining_after_first_day - thrown_away_second_day
  let percent_sold_second_day := (sold_second_day * 100) / remaining_after_first_day
  percent_sold_second_day = 50 :=
sorry

end vendor_second_day_sale_l409_40928


namespace trapezoid_shorter_base_l409_40925

theorem trapezoid_shorter_base (L : ℝ) (S : ℝ) (m : ℝ)
  (hL : L = 100)
  (hm : m = 4)
  (h : m = (L - S) / 2) :
  S = 92 :=
by {
  sorry -- Proof is not required
}

end trapezoid_shorter_base_l409_40925


namespace unique_parallelogram_l409_40967

theorem unique_parallelogram :
  ∃! (A B D C : ℤ × ℤ), 
  A = (0, 0) ∧ 
  (B.2 = B.1) ∧ 
  (D.2 = 2 * D.1) ∧ 
  (C.2 = 3 * C.1) ∧ 
  (A.1 = 0 ∧ A.2 = 0) ∧ 
  (B.1 > 0 ∧ B.2 > 0) ∧ 
  (D.1 > 0 ∧ D.2 > 0) ∧ 
  (C.1 > 0 ∧ C.2 > 0) ∧ 
  (B.1 - A.1, B.2 - A.2) + (D.1 - A.1, D.2 - A.2) = (C.1 - A.1, C.2 - A.2) ∧
  (abs ((B.1 * C.2 + C.1 * D.2 + D.1 * A.2 + A.1 * B.2) - (A.1 * C.2 + B.1 * D.2 + C.1 * B.2 + D.1 * A.2)) / 2) = 2000000 
  := by sorry

end unique_parallelogram_l409_40967


namespace find_y_l409_40966

theorem find_y (x y : ℕ) (hx : x > 0) (hy : y > 0) (hr : x % y = 9) (hxy : (x : ℝ) / y = 96.45) : y = 20 :=
by
  sorry

end find_y_l409_40966

import Mathlib

namespace find_a_l440_44009

noncomputable def f (a x : ℝ) := a * x + 1 / Real.sqrt 2

theorem find_a (a : ℝ) (h_pos : 0 < a) (h : f a (f a (1 / Real.sqrt 2)) = f a 0) : a = 0 :=
by
  sorry

end find_a_l440_44009


namespace proof_problem1_proof_problem2_l440_44075

noncomputable def problem1_lhs : ℝ := 
  1 / (Real.sqrt 3 + 1) - Real.sin (Real.pi / 3) + Real.sqrt 32 * Real.sqrt (1 / 8)

noncomputable def problem1_rhs : ℝ := 3 / 2

theorem proof_problem1 : problem1_lhs = problem1_rhs :=
by 
  sorry

noncomputable def problem2_lhs : ℝ := 
  2^(-2 : ℤ) - Real.sqrt ((-2)^2) + 6 * Real.sin (Real.pi / 4) - Real.sqrt 18

noncomputable def problem2_rhs : ℝ := -7 / 4

theorem proof_problem2 : problem2_lhs = problem2_rhs :=
by 
  sorry

end proof_problem1_proof_problem2_l440_44075


namespace correct_subtraction_result_l440_44070

-- Definition of numbers:
def tens_digit := 2
def ones_digit := 4
def correct_number := 10 * tens_digit + ones_digit
def incorrect_number := 59
def incorrect_result := 14
def Z := incorrect_result + incorrect_number

-- Statement of the theorem
theorem correct_subtraction_result : Z - correct_number = 49 :=
by
  sorry

end correct_subtraction_result_l440_44070


namespace factor_expression_l440_44005

theorem factor_expression (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 2) * (x + 3) :=
by
  sorry

end factor_expression_l440_44005


namespace probability_of_2_gold_no_danger_l440_44066

variable (caves : Finset Nat) (n : Nat)

-- Probability definitions
def P_gold_no_danger : ℚ := 1 / 5
def P_danger_no_gold : ℚ := 1 / 10
def P_neither : ℚ := 4 / 5

-- Probability calculation
def P_exactly_2_gold_none_danger : ℚ :=
  10 * (P_gold_no_danger) ^ 2 * (P_neither) ^ 3

theorem probability_of_2_gold_no_danger :
  (P_exactly_2_gold_none_danger) = 128 / 625 :=
sorry

end probability_of_2_gold_no_danger_l440_44066


namespace value_of_C_l440_44097

theorem value_of_C (k : ℝ) (C : ℝ) (h : k = 0.4444444444444444) :
  (2 * k * 0 ^ 2 + 6 * k * 0 + C = 0) ↔ C = 2 :=
by {
  sorry
}

end value_of_C_l440_44097


namespace sequence_formula_l440_44095

theorem sequence_formula (a : ℕ → ℕ)
  (h1 : a 1 = 1)
  (h2 : a 2 = 3)
  (h3 : a 3 = 6)
  (h4 : a 4 = 10)
  (h5 : ∀ n > 0, a (n + 1) - a n = n + 1) :
  ∀ n, a n = n * (n + 1) / 2 :=
by 
  sorry

end sequence_formula_l440_44095


namespace area_above_line_of_circle_l440_44027

-- Define the circle equation
def circle_eq (x y : ℝ) := (x - 10)^2 + (y - 5)^2 = 50

-- Define the line equation
def line_eq (x y : ℝ) := y = x - 6

-- The area to determine
def area_above_line (R : ℝ) := 25 * R

-- Proof statement
theorem area_above_line_of_circle : area_above_line Real.pi = 25 * Real.pi :=
by
  -- mark the proof as sorry to skip the proof
  sorry

end area_above_line_of_circle_l440_44027


namespace volume_of_cone_l440_44006

noncomputable def lateral_surface_area : ℝ := 8 * Real.pi

theorem volume_of_cone (l r h : ℝ)
  (h_lateral_surface : l * Real.pi = 2 * lateral_surface_area)
  (h_radius : l = 2 * r)
  (h_height : h = Real.sqrt (l^2 - r^2)) :
  (1/3) * Real.pi * r^2 * h = (8 * Real.sqrt 3 * Real.pi) / 3 :=
by
  sorry

end volume_of_cone_l440_44006


namespace num_valid_configurations_l440_44032

-- Definitions used in the problem
def grid := (Fin 8) × (Fin 8)
def knights_tell_truth := true
def knaves_lie := true
def statement (i j : Fin 8) (r c : grid → ℕ) := (c ⟨0,j⟩ > r ⟨i,0⟩)

-- The theorem statement to prove
theorem num_valid_configurations : ∃ n : ℕ, n = 255 :=
sorry

end num_valid_configurations_l440_44032


namespace ferry_max_weight_capacity_l440_44033

def automobile_max_weight : ℝ := 3200
def automobile_count : ℝ := 62.5
def pounds_to_tons : ℝ := 2000

theorem ferry_max_weight_capacity : 
  (automobile_max_weight * automobile_count) / pounds_to_tons = 100 := 
by 
  sorry

end ferry_max_weight_capacity_l440_44033


namespace combined_stickers_count_l440_44043

theorem combined_stickers_count :
  let initial_june_stickers := 76
  let initial_bonnie_stickers := 63
  let stickers_given := 25
  let june_total := initial_june_stickers + stickers_given
  let bonnie_total := initial_bonnie_stickers + stickers_given
  june_total + bonnie_total = 189 :=
by
  -- Definitions
  let initial_june_stickers := 76
  let initial_bonnie_stickers := 63
  let stickers_given := 25
  
  -- Calculations
  let june_total := initial_june_stickers + stickers_given
  let bonnie_total := initial_bonnie_stickers + stickers_given

  -- Proof is omitted
  sorry

end combined_stickers_count_l440_44043


namespace y_work_time_l440_44019

theorem y_work_time (x_days : ℕ) (x_work_time : ℕ) (y_work_time : ℕ) :
  x_days = 40 ∧ x_work_time = 8 ∧ y_work_time = 20 →
  let x_rate := 1 / 40
  let work_done_by_x := 8 * x_rate
  let remaining_work := 1 - work_done_by_x
  let y_rate := remaining_work / 20
  y_rate * 25 = 1 :=
by {
  sorry
}

end y_work_time_l440_44019


namespace tangent_lines_to_circle_l440_44017

-- Conditions
def regions_not_enclosed := 68
def num_lines := 30 - 4

-- Theorem statement
theorem tangent_lines_to_circle (h: regions_not_enclosed = 68) : num_lines = 26 :=
by {
  sorry
}

end tangent_lines_to_circle_l440_44017


namespace find_f_19_l440_44088

noncomputable def f : ℝ → ℝ := sorry -- placeholder for the given function

-- Define the conditions
axiom even_function : ∀ x : ℝ, f x = f (-x) 
axiom periodicity : ∀ x : ℝ, f (x + 2) = -f x

-- The statement we need to prove
theorem find_f_19 : f 19 = 0 := 
by
  sorry -- placeholder for the proof

end find_f_19_l440_44088


namespace find_grape_juice_l440_44046

variables (milk water: ℝ) (limit total_before_test grapejuice: ℝ)

-- Conditions
def milk_amt: ℝ := 8
def water_amt: ℝ := 8
def limit_amt: ℝ := 32

-- The total liquid consumed before the test can be computed
def total_before_test_amt (milk water: ℝ) : ℝ := limit_amt - water_amt

-- The given total liquid consumed must be (milk + grape juice)
def total_consumed (milk grapejuice: ℝ) : ℝ := milk + grapejuice

theorem find_grape_juice :
    total_before_test_amt milk_amt water_amt = total_consumed milk_amt grapejuice →
    grapejuice = 16 :=
by
    unfold total_before_test_amt total_consumed
    sorry

end find_grape_juice_l440_44046


namespace x_eq_y_sufficient_not_necessary_abs_l440_44076

theorem x_eq_y_sufficient_not_necessary_abs (x y : ℝ) : (x = y → |x| = |y|) ∧ (|x| = |y| → x = y ∨ x = -y) :=
by {
  sorry
}

end x_eq_y_sufficient_not_necessary_abs_l440_44076


namespace find_certain_number_l440_44016

theorem find_certain_number (a : ℤ) (certain_number : ℤ) (h₁ : a = 105) (h₂ : a^3 = 21 * 25 * 45 * certain_number) : certain_number = 49 := 
sorry

end find_certain_number_l440_44016


namespace otimes_h_h_h_eq_h_l440_44049

variable (h : ℝ)

def otimes (x y : ℝ) : ℝ := x^3 - y

theorem otimes_h_h_h_eq_h : otimes h (otimes h h) = h := by
  -- Proof goes here, but is omitted
  sorry

end otimes_h_h_h_eq_h_l440_44049


namespace number_of_yellow_marbles_l440_44063

/-- 
 In a jar with blue, red, and yellow marbles:
  - there are 7 blue marbles
  - there are 11 red marbles
  - the probability of picking a yellow marble is 1/4
 Show that the number of yellow marbles is 6.
-/
theorem number_of_yellow_marbles 
  (blue red y : ℕ) 
  (h_blue : blue = 7) 
  (h_red : red = 11) 
  (h_prob : y / (18 + y) = 1 / 4) : 
  y = 6 := 
sorry

end number_of_yellow_marbles_l440_44063


namespace inequality_my_problem_l440_44081

theorem inequality_my_problem (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_sum : a * b + b * c + c * a = 1) :
  (Real.sqrt ((1 / a) + 6 * b)) + (Real.sqrt ((1 / b) + 6 * c)) + (Real.sqrt ((1 / c) + 6 * a)) ≤ (1 / (a * b * c)) :=
  sorry

end inequality_my_problem_l440_44081


namespace first_term_geometric_sequence_l440_44084

variable {a : ℕ → ℝ} -- Define the geometric sequence a_n
variable (q : ℝ) -- Define the common ratio q which is a real number

-- Conditions given in the problem
def geom_seq_first_term (a : ℕ → ℝ) (q : ℝ) :=
  a 3 = 2 ∧ a 4 = 4 ∧ (∀ n : ℕ, a (n+1) = a n * q)

-- Assert that if these conditions hold, then the first term is 1/2
theorem first_term_geometric_sequence (hq : geom_seq_first_term a q) : a 1 = 1/2 :=
by
  sorry

end first_term_geometric_sequence_l440_44084


namespace heather_bicycling_time_l440_44023

theorem heather_bicycling_time (distance speed : ℝ) (h_distance : distance = 40) (h_speed : speed = 8) : (distance / speed) = 5 := 
by
  rw [h_distance, h_speed]
  norm_num

end heather_bicycling_time_l440_44023


namespace water_tank_capacity_l440_44059

-- Define the variables and conditions
variables (T : ℝ) (h : 0.35 * T = 36)

-- State the theorem
theorem water_tank_capacity : T = 103 :=
by
  -- Placeholder for proof
  sorry

end water_tank_capacity_l440_44059


namespace correct_factorization_l440_44044

theorem correct_factorization (a : ℝ) : 
  (a ^ 2 + 4 * a ≠ a ^ 2 * (a + 4)) ∧ 
  (a ^ 2 - 9 ≠ (a + 9) * (a - 9)) ∧ 
  (a ^ 2 + 4 * a + 2 ≠ (a + 2) ^ 2) → 
  (a ^ 2 - 2 * a + 1 = (a - 1) ^ 2) :=
by sorry

end correct_factorization_l440_44044


namespace solveForX_l440_44072

theorem solveForX : ∃ (x : ℚ), x + 5/8 = 7/24 + 1/4 ∧ x = -1/12 := by
  sorry

end solveForX_l440_44072


namespace y_intercept_of_parallel_line_l440_44010

theorem y_intercept_of_parallel_line (m : ℝ) (c1 c2 : ℝ) (x1 y1 : ℝ) (H_parallel : m = -3) (H_passing : (x1, y1) = (1, -4)) : 
    c2 = -1 :=
  sorry

end y_intercept_of_parallel_line_l440_44010


namespace min_route_length_5x5_l440_44040

-- Definition of the grid and its properties
def grid : Type := Fin 5 × Fin 5

-- Define a function to calculate the minimum route length
noncomputable def min_route_length (grid_size : ℕ) : ℕ :=
  if h : grid_size = 5 then 68 else 0

-- The proof problem statement
theorem min_route_length_5x5 : min_route_length 5 = 68 :=
by
  -- Skipping the actual proof
  sorry

end min_route_length_5x5_l440_44040


namespace time_to_cross_platform_l440_44021

-- Definitions from conditions
def train_speed_kmph : ℕ := 72
def speed_conversion_factor : ℕ := 1000 / 3600
def train_speed_mps : ℤ := train_speed_kmph * speed_conversion_factor
def time_cross_man_sec : ℕ := 16
def platform_length_meters : ℕ := 280

-- Proving the total time to cross platform
theorem time_to_cross_platform : ∃ t : ℕ, t = (platform_length_meters + (train_speed_mps * time_cross_man_sec)) / train_speed_mps ∧ t = 30 := 
by
  -- Since the proof isn't required, we add "sorry" to act as a placeholder.
  sorry

end time_to_cross_platform_l440_44021


namespace minimum_jumps_l440_44004

theorem minimum_jumps (dist_cm : ℕ) (jump_mm : ℕ) (dist_mm : ℕ) (cm_to_mm_conversion : dist_mm = dist_cm * 10) (leap_condition : ∃ n : ℕ, jump_mm * n ≥ dist_mm) : ∃ n : ℕ, 19 * n = 18120 → n = 954 :=
by
  sorry

end minimum_jumps_l440_44004


namespace expression_equality_l440_44067

theorem expression_equality : (2 + Real.sqrt 2 + 1 / (2 + Real.sqrt 2) + 1 / (Real.sqrt 2 - 2) = 2) :=
sorry

end expression_equality_l440_44067


namespace func_positive_range_l440_44038

theorem func_positive_range (a : ℝ) : 
  (∀ x : ℝ, (5 - a) * x^2 - 6 * x + a + 5 > 0) → (-4 < a ∧ a < 4) := 
by 
  sorry

end func_positive_range_l440_44038


namespace repeated_digit_squares_l440_44045

theorem repeated_digit_squares :
  {n : ℕ | ∃ d : Fin 10, n = d ^ 2 ∧ (∀ m < n, m % 10 = d % 10)} ⊆ {0, 1, 4, 9} := by
  sorry

end repeated_digit_squares_l440_44045


namespace james_total_matches_l440_44056

-- Define the conditions
def dozen : Nat := 12
def boxes_per_dozen : Nat := 5
def matches_per_box : Nat := 20

-- Calculate the expected number of matches
def expected_matches : Nat := boxes_per_dozen * dozen * matches_per_box

-- State the theorem to be proved
theorem james_total_matches : expected_matches = 1200 := by
  sorry

end james_total_matches_l440_44056


namespace price_of_tray_l440_44090

noncomputable def price_per_egg : ℕ := 50
noncomputable def tray_eggs : ℕ := 30
noncomputable def discount_per_egg : ℕ := 10

theorem price_of_tray : (price_per_egg - discount_per_egg) * tray_eggs / 100 = 12 :=
by
  sorry

end price_of_tray_l440_44090


namespace max_eq_zero_max_two_solutions_l440_44082

theorem max_eq_zero_max_two_solutions {a b : Fin 10 → ℝ}
  (h : ∀ i, a i ≠ 0) : 
  ∃ (solution_count : ℕ), solution_count <= 2 ∧
  ∃ (solutions : Fin solution_count → ℝ), 
    ∀ (x : ℝ), (∀ i, max (a i * x + b i) = 0) ↔ ∃ j, x = solutions j := sorry

end max_eq_zero_max_two_solutions_l440_44082


namespace pages_share_units_digit_l440_44060

def units_digit (n : Nat) : Nat :=
  n % 10

theorem pages_share_units_digit :
  (∃ (x_set : Finset ℕ), (∀ (x : ℕ), x ∈ x_set ↔ (1 ≤ x ∧ x ≤ 63 ∧ units_digit x = units_digit (64 - x))) ∧ x_set.card = 13) :=
by
  sorry

end pages_share_units_digit_l440_44060


namespace min_omega_value_l440_44077

theorem min_omega_value (ω : ℝ) (hω : ω > 0)
  (f : ℝ → ℝ)
  (hf_def : ∀ x, f x = Real.cos (ω * x - (Real.pi / 6))) :
  (∀ x, f x ≤ f (Real.pi / 4)) → ω = 2 / 3 :=
by
  sorry

end min_omega_value_l440_44077


namespace tom_seashells_now_l440_44022

def original_seashells : ℕ := 5
def given_seashells : ℕ := 2

theorem tom_seashells_now : original_seashells - given_seashells = 3 :=
by
  sorry

end tom_seashells_now_l440_44022


namespace min_expression_l440_44080

theorem min_expression (a b c d e f : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f)
  (h_sum : a + b + c + d + e + f = 10) : 
  (1 / a + 9 / b + 16 / c + 25 / d + 36 / e + 49 / f) ≥ 67.6 :=
sorry

end min_expression_l440_44080


namespace arithmetic_sequence_smallest_value_l440_44041

theorem arithmetic_sequence_smallest_value:
  ∃ a : ℕ, (7 * a + 63) % 11 = 0 ∧ (a - 9) % 11 = 4 := sorry

end arithmetic_sequence_smallest_value_l440_44041


namespace difference_of_sum_l440_44037

theorem difference_of_sum (a b c : ℤ) (h1 : a = 11) (h2 : b = 13) (h3 : c = 15) :
  (b + c) - a = 17 := by
  sorry

end difference_of_sum_l440_44037


namespace francie_remaining_money_l440_44053

noncomputable def total_savings_before_investment : ℝ :=
  (5 * 8) + (6 * 6) + 20

noncomputable def investment_return : ℝ :=
  0.05 * 10

noncomputable def total_savings_after_investment : ℝ :=
  total_savings_before_investment + investment_return

noncomputable def spent_on_clothes : ℝ :=
  total_savings_after_investment / 2

noncomputable def remaining_after_clothes : ℝ :=
  total_savings_after_investment - spent_on_clothes

noncomputable def amount_remaining : ℝ :=
  remaining_after_clothes - 35

theorem francie_remaining_money : amount_remaining = 13.25 := 
  sorry

end francie_remaining_money_l440_44053


namespace packages_per_box_l440_44086

theorem packages_per_box (P : ℕ) (h1 : 192 > 0) (h2 : 2 > 0) (total_soaps : 2304 > 0) (h : 2 * P * 192 = 2304) : P = 6 :=
by
  sorry

end packages_per_box_l440_44086


namespace eleven_y_minus_x_eq_one_l440_44062

theorem eleven_y_minus_x_eq_one 
  (x y : ℤ) 
  (hx_pos : x > 0)
  (h1 : x = 7 * y + 3)
  (h2 : 2 * x = 6 * (3 * y) + 2) : 
  11 * y - x = 1 := 
by 
  sorry

end eleven_y_minus_x_eq_one_l440_44062


namespace path_length_cube_dot_l440_44087

-- Define the edge length of the cube
def edge_length : ℝ := 2

-- Define the distance of the dot from the center of the top face
def dot_distance_from_center : ℝ := 0.5

-- Define the number of complete rolls
def complete_rolls : ℕ := 2

-- Calculate the constant c such that the path length of the dot is c * π
theorem path_length_cube_dot : ∃ c : ℝ, dot_distance_from_center = 2.236 :=
by
  sorry

end path_length_cube_dot_l440_44087


namespace carton_height_is_60_l440_44007

-- Definitions
def carton_length : ℕ := 30
def carton_width : ℕ := 42
def soap_length : ℕ := 7
def soap_width : ℕ := 6
def soap_height : ℕ := 5
def max_soap_boxes : ℕ := 360

-- Theorem Statement
theorem carton_height_is_60 (h : ℕ) (H : ∀ (layers : ℕ), layers = max_soap_boxes / ((carton_length / soap_length) * (carton_width / soap_width)) → h = layers * soap_height) : h = 60 :=
  sorry

end carton_height_is_60_l440_44007


namespace tetrahedron_equal_reciprocal_squares_l440_44061

noncomputable def tet_condition_heights (h_1 h_2 h_3 h_4 : ℝ) : Prop :=
True

noncomputable def tet_condition_distances (d_1 d_2 d_3 : ℝ) : Prop :=
True

theorem tetrahedron_equal_reciprocal_squares
  (h_1 h_2 h_3 h_4 d_1 d_2 d_3 : ℝ)
  (hc_hts : tet_condition_heights h_1 h_2 h_3 h_4)
  (hc_dsts : tet_condition_distances d_1 d_2 d_3) :
  1 / (h_1 ^ 2) + 1 / (h_2 ^ 2) + 1 / (h_3 ^ 2) + 1 / (h_4 ^ 2) =
  1 / (d_1 ^ 2) + 1 / (d_2 ^ 2) + 1 / (d_3 ^ 2) :=
sorry

end tetrahedron_equal_reciprocal_squares_l440_44061


namespace dot_product_eq_eight_l440_44071

def vec_a : ℝ × ℝ := (0, 4)
def vec_b : ℝ × ℝ := (2, 2)

theorem dot_product_eq_eight : (vec_a.1 * vec_b.1 + vec_a.2 * vec_b.2) = 8 := by
  sorry

end dot_product_eq_eight_l440_44071


namespace geometric_sequence_sum_l440_44079

noncomputable def seq (a : ℕ → ℝ) : Prop :=
∀ n ≥ 2, a n ^ 2 = a (n - 1) * a (n + 1)

theorem geometric_sequence_sum (a : ℕ → ℝ) 
  (h_seq : seq a)
  (h_a2 : a 2 = 3)
  (h_sum : a 2 + a 4 + a 6 = 21) :
  (a 4 + a 6 + a 8) = 42 :=
sorry

end geometric_sequence_sum_l440_44079


namespace ratio_of_areas_l440_44073

theorem ratio_of_areas (s : ℝ) (h_s_pos : 0 < s) :
  let small_triangle_area := (s^2 * Real.sqrt 3) / 4
  let total_small_triangles_area := 6 * small_triangle_area
  let large_triangle_side := 6 * s
  let large_triangle_area := (large_triangle_side^2 * Real.sqrt 3) / 4
  total_small_triangles_area / large_triangle_area = 1 / 6 :=
by
  let small_triangle_area := (s^2 * Real.sqrt 3) / 4
  let total_small_triangles_area := 6 * small_triangle_area
  let large_triangle_side := 6 * s
  let large_triangle_area := (large_triangle_side^2 * Real.sqrt 3) / 4
  sorry
 
end ratio_of_areas_l440_44073


namespace largest_initial_number_l440_44068

theorem largest_initial_number (n : ℕ) (h : (∃ a b c d e : ℕ, n ≠ 0 ∧ n + a + b + c + d + e = 200 
                                              ∧ n % a ≠ 0 ∧ n % b ≠ 0 ∧ n % c ≠ 0 ∧ n % d ≠ 0 ∧ n % e ≠ 0)) 
: n ≤ 189 :=
sorry

end largest_initial_number_l440_44068


namespace number_of_jerseys_bought_l440_44035

-- Define the given constants
def initial_money : ℕ := 50
def cost_per_jersey : ℕ := 2
def cost_basketball : ℕ := 18
def cost_shorts : ℕ := 8
def money_left : ℕ := 14

-- Define the theorem to prove the number of jerseys Jeremy bought.
theorem number_of_jerseys_bought :
  (initial_money - money_left) = (cost_basketball + cost_shorts + 5 * cost_per_jersey) :=
by
  sorry

end number_of_jerseys_bought_l440_44035


namespace tan_2016_l440_44051

-- Define the given condition
def sin_36 (a : ℝ) : Prop := Real.sin (36 * Real.pi / 180) = a

-- Prove the required statement given the condition
theorem tan_2016 (a : ℝ) (h : sin_36 a) : Real.tan (2016 * Real.pi / 180) = a / Real.sqrt (1 - a^2) :=
sorry

end tan_2016_l440_44051


namespace quadratic_root_eq_l440_44011

theorem quadratic_root_eq {b : ℝ} (h : (2 : ℝ)^2 + b * 2 - 6 = 0) : b = 1 :=
by
  sorry

end quadratic_root_eq_l440_44011


namespace balls_in_boxes_l440_44048

/-- Prove that the number of ways to put 6 distinguishable balls in 3 distinguishable boxes is 729 (which is 3^6). -/
theorem balls_in_boxes : (3 ^ 6) = 729 :=
by 
  sorry

end balls_in_boxes_l440_44048


namespace remaining_employees_earn_rate_l440_44003

theorem remaining_employees_earn_rate
  (total_employees : ℕ)
  (employees_12_per_hour : ℕ)
  (employees_14_per_hour : ℕ)
  (total_cost : ℝ)
  (hourly_rate_12 : ℝ)
  (hourly_rate_14 : ℝ)
  (shift_hours : ℝ)
  (remaining_employees : ℕ)
  (remaining_hourly_rate : ℝ) :
  total_employees = 300 →
  employees_12_per_hour = 200 →
  employees_14_per_hour = 40 →
  total_cost = 31840 →
  hourly_rate_12 = 12 →
  hourly_rate_14 = 14 →
  shift_hours = 8 →
  remaining_employees = 60 →
  remaining_hourly_rate = 
    (total_cost - (employees_12_per_hour * hourly_rate_12 * shift_hours) - 
    (employees_14_per_hour * hourly_rate_14 * shift_hours)) / 
    (remaining_employees * shift_hours) →
  remaining_hourly_rate = 17 :=
by
  sorry

end remaining_employees_earn_rate_l440_44003


namespace range_of_m_l440_44039

theorem range_of_m (m : ℝ) : (∀ x : ℝ, |x + 1| + |x - 3| ≥ |m - 1|) → -3 ≤ m ∧ m ≤ 5 :=
by
  sorry

end range_of_m_l440_44039


namespace circle_standard_form1_circle_standard_form2_l440_44030

theorem circle_standard_form1 (x y : ℝ) :
  x^2 + y^2 - 4 * x + 6 * y - 3 = 0 ↔ (x - 2)^2 + (y + 3)^2 = 16 :=
by
  sorry

theorem circle_standard_form2 (x y : ℝ) :
  4 * x^2 + 4 * y^2 - 8 * x + 4 * y - 11 = 0 ↔ (x - 1)^2 + (y + 1 / 2)^2 = 4 :=
by
  sorry

end circle_standard_form1_circle_standard_form2_l440_44030


namespace necessarily_positive_y_plus_z_l440_44052

theorem necessarily_positive_y_plus_z
  (x y z : ℝ)
  (hx : 0 < x ∧ x < 0.5) 
  (hy : -0.5 < y ∧ y < 0) 
  (hz : 0.5 < z ∧ z < 1) :
  y + z > 0 := 
by
  sorry

end necessarily_positive_y_plus_z_l440_44052


namespace ellipse_eccentricity_l440_44028

theorem ellipse_eccentricity (m : ℝ) (e : ℝ) : 
  (∀ x y : ℝ, (x^2 / m) + (y^2 / 4) = 1) ∧ foci_y_axis ∧ e = 1 / 2 → m = 3 :=
by
  sorry

end ellipse_eccentricity_l440_44028


namespace kids_left_playing_l440_44083

-- Define the conditions
def initial_kids : ℝ := 22.0
def kids_went_home : ℝ := 14.0

-- Theorem statement: Prove that the number of kids left playing is 8.0
theorem kids_left_playing : initial_kids - kids_went_home = 8.0 :=
by
  sorry -- Proof is left as an exercise

end kids_left_playing_l440_44083


namespace radius_of_spherical_circle_correct_l440_44094

noncomputable def radius_of_spherical_circle (rho theta phi : ℝ) : ℝ :=
  if rho = 1 ∧ phi = Real.pi / 4 then Real.sqrt 2 / 2 else 0

theorem radius_of_spherical_circle_correct :
  ∀ (theta : ℝ), radius_of_spherical_circle 1 theta (Real.pi / 4) = Real.sqrt 2 / 2 :=
by
  sorry

end radius_of_spherical_circle_correct_l440_44094


namespace girls_attending_picnic_l440_44058

theorem girls_attending_picnic (g b : ℕ) (h1 : g + b = 1200) (h2 : (2 * g) / 3 + b / 2 = 730) : (2 * g) / 3 = 520 :=
by
  -- The proof steps would go here.
  sorry

end girls_attending_picnic_l440_44058


namespace tangent_line_eq_l440_44057

theorem tangent_line_eq : 
  ∀ (x y: ℝ), y = x^3 - x + 3 → (x = 1 ∧ y = 3) → (2 * x - y - 1 = 0) :=
by
  intros x y h_curve h_point
  sorry

end tangent_line_eq_l440_44057


namespace only_n_1_has_integer_solution_l440_44000

theorem only_n_1_has_integer_solution :
  ∀ n : ℕ, (∃ x : ℤ, x^n + (2 + x)^n + (2 - x)^n = 0) ↔ n = 1 := 
by 
  sorry

end only_n_1_has_integer_solution_l440_44000


namespace simplify_and_evaluate_l440_44085

noncomputable 
def expr (a b : ℚ) := 2*(a^2*b - 2*a*b) - 3*(a^2*b - 3*a*b) + a^2*b

theorem simplify_and_evaluate :
  let a := (-2 : ℚ) 
  let b := (1/3 : ℚ)
  expr a b = -10/3 :=
by
  sorry

end simplify_and_evaluate_l440_44085


namespace incorrect_option_c_l440_44012

theorem incorrect_option_c (a b c d : ℝ)
  (h1 : a + b + c ≥ d)
  (h2 : a + b + d ≥ c)
  (h3 : a + c + d ≥ b)
  (h4 : b + c + d ≥ a) :
  ¬(a < 0 ∧ b < 0 ∧ c < 0 ∧ d < 0) :=
by sorry

end incorrect_option_c_l440_44012


namespace power_of_two_divides_factorial_iff_l440_44013

theorem power_of_two_divides_factorial_iff (n : ℕ) (k : ℕ) : 2^(n - 1) ∣ n! ↔ n = 2^k := sorry

end power_of_two_divides_factorial_iff_l440_44013


namespace plane_equation_l440_44036

theorem plane_equation :
  ∃ (A B C D : ℤ), (A > 0) ∧ (Int.gcd (Int.gcd A B) (Int.gcd C D) = 1) ∧
  (∀ x y z : ℤ, 
    (A * x + B * y + C * z + D = 0) ↔
      (x = 1 ∧ y = 6 ∧ z = -8 ∨ (∃ t : ℤ, 
        x = 2 + 4 * t ∧ y = 4 - t ∧ z = -3 + 5 * t))) ∧
  (A = 5 ∧ B = 15 ∧ C = -7 ∧ D = -151) :=
sorry

end plane_equation_l440_44036


namespace emily_patches_difference_l440_44002

theorem emily_patches_difference (h p : ℕ) (h_eq : p = 3 * h) :
  (p * h) - ((p + 5) * (h - 3)) = (4 * h + 15) :=
by
  sorry

end emily_patches_difference_l440_44002


namespace even_function_increasing_l440_44018

variable (a b : ℝ)
def f (x : ℝ) : ℝ := a * x^2 - 2 * b * x + 1

theorem even_function_increasing (h_even : ∀ x : ℝ, f a b x = f a b (-x))
  (h_increasing : ∀ x y : ℝ, x ≤ 0 → y ≤ 0 → x < y → f a b x < f a b y) :
  f a b (a-2) < f a b (b+1) :=
sorry

end even_function_increasing_l440_44018


namespace average_salary_techs_l440_44024

noncomputable def total_salary := 20000
noncomputable def average_salary_all := 750
noncomputable def num_technicians := 5
noncomputable def average_salary_non_tech := 700
noncomputable def total_workers := 20

theorem average_salary_techs :
  (20000 - (num_technicians + average_salary_non_tech * (total_workers - num_technicians))) / num_technicians = 900 := by
  sorry

end average_salary_techs_l440_44024


namespace factorization_from_left_to_right_l440_44025

theorem factorization_from_left_to_right (a x y b : ℝ) :
  (a * (a + 1) = a^2 + a ∨
   a^2 + 3 * a - 1 = a * (a + 3) + 1 ∨
   x^2 - 4 * y^2 = (x + 2 * y) * (x - 2 * y) ∨
   (a - b)^3 = -(b - a)^3) →
  (x^2 - 4 * y^2 = (x + 2 * y) * (x - 2 * y)) := sorry

end factorization_from_left_to_right_l440_44025


namespace final_song_count_l440_44093

theorem final_song_count {init_songs added_songs removed_songs doubled_songs final_songs : ℕ} 
    (h1 : init_songs = 500)
    (h2 : added_songs = 500)
    (h3 : doubled_songs = (init_songs + added_songs) * 2)
    (h4 : removed_songs = 50)
    (h_final : final_songs = doubled_songs - removed_songs) : 
    final_songs = 2950 :=
by
  sorry

end final_song_count_l440_44093


namespace total_number_of_people_l440_44008

theorem total_number_of_people (num_cannoneers num_women num_men total_people : ℕ)
  (h1 : num_women = 2 * num_cannoneers)
  (h2 : num_cannoneers = 63)
  (h3 : num_men = 2 * num_women)
  (h4 : total_people = num_women + num_men) : 
  total_people = 378 := by
  sorry

end total_number_of_people_l440_44008


namespace product_of_intersection_coordinates_l440_44096

noncomputable def circle1 (x y : ℝ) : Prop := (x - 1)^2 + (y - 5)^2 = 1
noncomputable def circle2 (x y : ℝ) : Prop := (x - 5)^2 + (y - 5)^2 = 4

theorem product_of_intersection_coordinates :
  ∃ (x y : ℝ), circle1 x y ∧ circle2 x y ∧ x * y = 15 :=
by
  sorry

end product_of_intersection_coordinates_l440_44096


namespace cylinder_volume_l440_44074

-- Definitions based on conditions
def lateral_surface_to_rectangle (generatrix_a generatrix_b : ℝ) (volume : ℝ) :=
  -- Condition: Rectangle with sides 8π and 4π
  (generatrix_a = 8 * Real.pi ∧ volume = 32 * Real.pi^2) ∨
  (generatrix_a = 4 * Real.pi ∧ volume = 64 * Real.pi^2)

-- Statement
theorem cylinder_volume (generatrix_a generatrix_b : ℝ)
  (h : (generatrix_a = 8 * Real.pi ∨ generatrix_b = 4 * Real.pi) ∧ (generatrix_b = 4 * Real.pi ∨ generatrix_b = 8 * Real.pi)) :
  ∃ (volume : ℝ), lateral_surface_to_rectangle generatrix_a generatrix_b volume :=
sorry

end cylinder_volume_l440_44074


namespace modulus_of_z_l440_44042

noncomputable def i : ℂ := Complex.I

noncomputable def z : ℂ := sorry

theorem modulus_of_z 
  (hz : i * z = (1 - 2 * i)^2) : 
  Complex.abs z = 5 := by
  sorry

end modulus_of_z_l440_44042


namespace factor_expression_l440_44001

theorem factor_expression (x : ℝ) :
  (16 * x ^ 7 + 36 * x ^ 4 - 9) - (4 * x ^ 7 - 6 * x ^ 4 - 9) = 6 * x ^ 4 * (2 * x ^ 3 + 7) :=
by
  sorry

end factor_expression_l440_44001


namespace factorization_correct_l440_44091

theorem factorization_correct {c d : ℤ} (h1 : c + 4 * d = 4) (h2 : c * d = -32) :
  c - d = 12 :=
by
  sorry

end factorization_correct_l440_44091


namespace new_person_weight_l440_44092

theorem new_person_weight
    (avg_weight_20 : ℕ → ℕ)
    (total_weight_20 : ℕ)
    (avg_weight_21 : ℕ)
    (count_20 : ℕ)
    (count_21 : ℕ) :
    avg_weight_20 count_20 = 58 →
    total_weight_20 = count_20 * avg_weight_20 count_20 →
    avg_weight_21 = 53 →
    count_21 = count_20 + 1 →
    ∃ (W : ℕ), total_weight_20 + W = count_21 * avg_weight_21 ∧ W = 47 := 
by 
  sorry

end new_person_weight_l440_44092


namespace circle_range_of_a_l440_44054

theorem circle_range_of_a (a : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 + 2 * a * x - 4 * y + (a^2 + a) = 0 → (x - h)^2 + (y - k)^2 = r^2) ↔ (a < 4) :=
sorry

end circle_range_of_a_l440_44054


namespace remainder_of_product_l440_44015

theorem remainder_of_product (a b c : ℕ) (h1 : a % 7 = 2) (h2 : b % 7 = 3) (h3 : c % 7 = 4) : 
  (a * b * c) % 7 = 3 := 
by
  sorry

end remainder_of_product_l440_44015


namespace round_robin_pairing_possible_l440_44050

def players : Set String := {"A", "B", "C", "D", "E", "F"}

def is_pairing (pairs : List (String × String)) : Prop :=
  ∀ (p : String × String), p ∈ pairs → p.1 ≠ p.2 ∧ p.1 ∈ players ∧ p.2 ∈ players

def unique_pairs (rounds : List (List (String × String))) : Prop :=
  ∀ r, r ∈ rounds → is_pairing r ∧ (∀ p1 p2, p1 ∈ r → p2 ∈ r → p1 ≠ p2 → p1.1 ≠ p2.1 ∧ p1.2 ≠ p2.2)

def all_players_paired (rounds : List (List (String × String))) : Prop :=
  ∀ p, p ∈ players →
  (∀ q, q ∈ players → p ≠ q → 
    (∃ r, r ∈ rounds ∧ (p,q) ∈ r ∨ (q,p) ∈ r))

theorem round_robin_pairing_possible : 
  ∃ rounds, List.length rounds = 5 ∧ unique_pairs rounds ∧ all_players_paired rounds :=
  sorry

end round_robin_pairing_possible_l440_44050


namespace Brian_age_in_eight_years_l440_44031

-- Definitions based on conditions
variable {Christian Brian : ℕ}
variable (h1 : Christian = 2 * Brian)
variable (h2 : Christian + 8 = 72)

-- Target statement to prove Brian's age in eight years
theorem Brian_age_in_eight_years : (Brian + 8) = 40 :=
by 
  sorry

end Brian_age_in_eight_years_l440_44031


namespace prob_both_standard_prob_only_one_standard_l440_44047

-- Given conditions
axiom prob_A1 : ℝ
axiom prob_A2 : ℝ
axiom prob_A1_std : prob_A1 = 0.95
axiom prob_A2_std : prob_A2 = 0.95
axiom prob_not_A1 : ℝ
axiom prob_not_A2 : ℝ
axiom prob_not_A1_std : prob_not_A1 = 0.05
axiom prob_not_A2_std : prob_not_A2 = 0.05
axiom independent_A1_A2 : prob_A1 * prob_A2 = prob_A1 * prob_A2

-- Definitions of events
def event_A1 := true -- Event that the first product is standard
def event_A2 := true -- Event that the second product is standard
def event_not_A1 := not event_A1
def event_not_A2 := not event_A2

-- Proof problems
theorem prob_both_standard :
  prob_A1 * prob_A2 = 0.9025 := by sorry

theorem prob_only_one_standard :
  (prob_A1 * prob_not_A2) + (prob_not_A1 * prob_A2) = 0.095 := by sorry

end prob_both_standard_prob_only_one_standard_l440_44047


namespace train_cross_time_platform_l440_44069

def speed := 36 -- in kmph
def time_for_pole := 12 -- in seconds
def time_for_platform := 44.99736021118311 -- in seconds

theorem train_cross_time_platform :
  time_for_platform = 44.99736021118311 :=
by
  sorry

end train_cross_time_platform_l440_44069


namespace Mike_given_total_cookies_l440_44099

-- All given conditions
variables (total Tim fridge Mike Anna : Nat)
axiom h1 : total = 256
axiom h2 : Tim = 15
axiom h3 : fridge = 188
axiom h4 : Anna = 2 * Tim
axiom h5 : total = Tim + Anna + fridge + Mike

-- The goal of the proof
theorem Mike_given_total_cookies : Mike = 23 :=
by
  sorry

end Mike_given_total_cookies_l440_44099


namespace length_first_train_correct_l440_44064

noncomputable def length_first_train 
    (speed_train1_kmph : ℕ := 120)
    (speed_train2_kmph : ℕ := 80)
    (length_train2_m : ℝ := 290.04)
    (time_sec : ℕ := 9) 
    (conversion_factor : ℝ := (5 / 18)) : ℝ :=
  let relative_speed_kmph := speed_train1_kmph + speed_train2_kmph
  let relative_speed_mps := relative_speed_kmph * conversion_factor
  let total_distance_m := relative_speed_mps * time_sec
  let length_train1_m := total_distance_m - length_train2_m
  length_train1_m

theorem length_first_train_correct 
    (L1_approx : ℝ := 210) :
    length_first_train = L1_approx :=
  by
  sorry

end length_first_train_correct_l440_44064


namespace stationary_train_length_l440_44014

noncomputable def speed_train_kmh : ℝ := 144
noncomputable def speed_train_ms : ℝ := (speed_train_kmh * 1000) / 3600
noncomputable def time_to_pass_pole : ℝ := 8
noncomputable def time_to_pass_stationary : ℝ := 18
noncomputable def length_moving_train : ℝ := speed_train_ms * time_to_pass_pole
noncomputable def total_distance : ℝ := speed_train_ms * time_to_pass_stationary
noncomputable def length_stationary_train : ℝ := total_distance - length_moving_train

theorem stationary_train_length :
  length_stationary_train = 400 := by
  sorry

end stationary_train_length_l440_44014


namespace find_min_y_l440_44034

theorem find_min_y (x y : ℕ) (hx : x = y + 8) 
    (h : Nat.gcd ((x^3 + y^3) / (x + y)) (x * y) = 16) : 
    y = 4 :=
sorry

end find_min_y_l440_44034


namespace pointC_on_same_side_as_point1_l440_44065

-- Definitions of points and the line equation
def is_on_same_side (p1 p2 : ℝ × ℝ) (line : ℝ × ℝ → ℝ) : Prop :=
  (line p1 > 0) ↔ (line p2 > 0)

def line_eq (p : ℝ × ℝ) : ℝ := p.1 + p.2 - 1

def point1 : ℝ × ℝ := (1, 2)
def pointC : ℝ × ℝ := (-1, 3)

-- Theorem to prove the equivalence
theorem pointC_on_same_side_as_point1 :
  is_on_same_side point1 pointC line_eq :=
sorry

end pointC_on_same_side_as_point1_l440_44065


namespace value_of_7th_term_l440_44089

noncomputable def arithmetic_sequence_a1_d_n (a1 d n a7 : ℝ) : Prop := 
  ((5 * a1 + 10 * d = 68) ∧ 
   (5 * (a1 + (n - 1) * d) - 10 * d = 292) ∧
   (n / 2 * (2 * a1 + (n - 1) * d) = 234) ∧ 
   (a1 + 6 * d = a7))

theorem value_of_7th_term (a1 d n a7 : ℝ) : 
  arithmetic_sequence_a1_d_n a1 d n 18 := 
by
  simp [arithmetic_sequence_a1_d_n]
  sorry

end value_of_7th_term_l440_44089


namespace determine_M_l440_44020

theorem determine_M : ∃ M : ℕ, 36^2 * 75^2 = 30^2 * M^2 ∧ M = 90 := 
by
  sorry

end determine_M_l440_44020


namespace number_of_pink_cookies_l440_44098

def total_cookies : ℕ := 86
def red_cookies : ℕ := 36

def pink_cookies (total red : ℕ) : ℕ := total - red

theorem number_of_pink_cookies : pink_cookies total_cookies red_cookies = 50 :=
by
  sorry

end number_of_pink_cookies_l440_44098


namespace solution_set_f_inequality_l440_44055

variable (f : ℝ → ℝ)

axiom domain_of_f : ∀ x : ℝ, true
axiom avg_rate_of_f : ∀ (x1 x2 : ℝ), x1 ≠ x2 → (f x1 - f x2) / (x1 - x2) > 3
axiom f_at_5 : f 5 = 18

theorem solution_set_f_inequality : {x : ℝ | f (3 * x - 1) > 9 * x} = {x : ℝ | x > 2} :=
by
  sorry

end solution_set_f_inequality_l440_44055


namespace range_of_a_l440_44029

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - 2 * a * x + 3 > 0) ↔ (0 ≤ a ∧ a < 3) :=
sorry

end range_of_a_l440_44029


namespace two_pow_ge_n_cubed_l440_44026

theorem two_pow_ge_n_cubed (n : ℕ) : 2^n ≥ n^3 ↔ n ≥ 10 := 
by sorry

end two_pow_ge_n_cubed_l440_44026


namespace find_sum_of_a_b_c_l440_44078

def a := 8
def b := 2
def c := 2

theorem find_sum_of_a_b_c : a + b + c = 12 :=
by
  have ha : a = 8 := rfl
  have hb : b = 2 := rfl
  have hc : c = 2 := rfl
  sorry

end find_sum_of_a_b_c_l440_44078

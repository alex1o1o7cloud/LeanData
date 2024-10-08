import Mathlib

namespace candy_bar_sales_l159_159053

def max_sales : ℕ := 24
def seth_sales (max_sales : ℕ) : ℕ := 3 * max_sales + 6
def emma_sales (seth_sales : ℕ) : ℕ := seth_sales / 2 + 5
def total_sales (seth_sales emma_sales : ℕ) : ℕ := seth_sales + emma_sales

theorem candy_bar_sales : total_sales (seth_sales max_sales) (emma_sales (seth_sales max_sales)) = 122 := by
  sorry

end candy_bar_sales_l159_159053


namespace closest_perfect_square_to_350_l159_159999

theorem closest_perfect_square_to_350 : 
  ∃ n : ℤ, n^2 < 350 ∧ 350 < (n + 1)^2 ∧ (350 - n^2 ≤ (n + 1)^2 - 350) ∨ (350 - n^2 ≥ (n + 1)^2 - 350) ∧ 
  (if (350 - n^2 < (n + 1)^2 - 350) then n+1 else n) = 19 := 
by
  sorry

end closest_perfect_square_to_350_l159_159999


namespace chosen_number_is_120_l159_159347

theorem chosen_number_is_120 (x : ℤ) (h : 2 * x - 138 = 102) : x = 120 :=
sorry

end chosen_number_is_120_l159_159347


namespace problem1_problem2_l159_159609

-- Problem 1
theorem problem1 (α : ℝ) (h : 2 * Real.sin α - Real.cos α = 0) :
  (Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) + (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = -10 / 3 :=
sorry

-- Problem 2
theorem problem2 (x : ℝ) (h : Real.cos (π / 4 + x) = 3 / 5) :
  (Real.sin x ^ 3 + Real.sin x * Real.cos x ^ 2) / (1 - Real.tan x) = 7 * Real.sqrt 2 / 60 :=
sorry

end problem1_problem2_l159_159609


namespace joy_pencils_count_l159_159411

theorem joy_pencils_count :
  ∃ J, J = 30 ∧ (∃ (pencils_cost_J pencils_cost_C : ℕ), 
  pencils_cost_C = 50 * 4 ∧ pencils_cost_J = pencils_cost_C - 80 ∧ J = pencils_cost_J / 4) := sorry

end joy_pencils_count_l159_159411


namespace no_perfect_square_for_nnplus1_l159_159357

theorem no_perfect_square_for_nnplus1 :
  ¬ ∃ (n : ℕ), 0 < n ∧ ∃ (k : ℕ), n * (n + 1) = k * k :=
sorry

end no_perfect_square_for_nnplus1_l159_159357


namespace Doris_needs_3_weeks_l159_159282

-- Definitions based on conditions
def hourly_wage : ℕ := 20
def monthly_expenses : ℕ := 1200
def weekday_hours_per_day : ℕ := 3
def saturdays_hours : ℕ := 5
def weekdays_per_week : ℕ := 5

-- Total hours per week
def total_hours_per_week := (weekday_hours_per_day * weekdays_per_week) + saturdays_hours

-- Weekly earnings
def weekly_earnings := hourly_wage * total_hours_per_week

-- Number of weeks needed for monthly expenses
def weeks_needed := monthly_expenses / weekly_earnings

-- Proposition to prove
theorem Doris_needs_3_weeks :
  weeks_needed = 3 := 
by
  sorry

end Doris_needs_3_weeks_l159_159282


namespace find_second_number_l159_159287

theorem find_second_number 
  (x y z : ℕ)
  (h1 : x + y + z = 120)
  (h2 : x = (3 * y) / 4)
  (h3 : z = (9 * y) / 7) : 
  y = 40 :=
sorry

end find_second_number_l159_159287


namespace ratio_Pat_Mark_l159_159621

-- Definitions inferred from the conditions
def total_hours : ℕ := 135
def Kate_hours (K : ℕ) : ℕ := K
def Pat_hours (K : ℕ) : ℕ := 2 * K
def Mark_hours (K : ℕ) : ℕ := K + 75

-- The main statement
theorem ratio_Pat_Mark (K : ℕ) (h : Kate_hours K + Pat_hours K + Mark_hours K = total_hours) :
  (Pat_hours K) / (Mark_hours K) = 1 / 3 := by
  sorry

end ratio_Pat_Mark_l159_159621


namespace problem_part1_problem_part2_l159_159159

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 2 * (Real.cos x) ^ 2 + 2 * Real.sqrt 3 * (Real.sin x) * (Real.cos x) + a

noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin (4 * x - Real.pi / 6) + 3

theorem problem_part1 (h : ∀ x ∈ Set.Icc 0 (Real.pi / 2), f x 2 ≥ 2) :
    ∃ a : ℝ, a = 2 ∧ 
    ∀ k : ℤ, ∀ x ∈ Set.Icc (k * Real.pi - Real.pi / 3) (k * Real.pi + Real.pi / 6), 
    ∃ m : ℤ, x = (m * Real.pi / 2 + Real.pi / 12) ∨ x = (m * Real.pi / 2 + Real.pi / 4) := sorry

theorem problem_part2 :
    ∀ x ∈ Set.Icc 0 (Real.pi / 2), g x = 4 → 
    ∃ s : ℝ, s = Real.pi / 3 := sorry

end problem_part1_problem_part2_l159_159159


namespace remainder_when_divided_by_18_l159_159131

theorem remainder_when_divided_by_18 (n : ℕ) (r3 r6 r9 : ℕ)
  (hr3 : r3 = n % 3)
  (hr6 : r6 = n % 6)
  (hr9 : r9 = n % 9)
  (h_sum : r3 + r6 + r9 = 15) :
  n % 18 = 17 := sorry

end remainder_when_divided_by_18_l159_159131


namespace inequality_proof_l159_159528

theorem inequality_proof (a b c : ℝ) (hp : 0 < a ∧ 0 < b ∧ 0 < c) (hd : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
    (bc / a + ac / b + ab / c > a + b + c) :=
by
  sorry

end inequality_proof_l159_159528


namespace bound_diff_sqrt_two_l159_159592

theorem bound_diff_sqrt_two (a b k m : ℝ) (h : ∀ x ∈ Set.Icc a b, abs (x^2 - k * x - m) ≤ 1) : b - a ≤ 2 * Real.sqrt 2 := sorry

end bound_diff_sqrt_two_l159_159592


namespace minimum_value_l159_159676

noncomputable def problem_statement (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a * b * c = 27) : ℝ :=
  a^2 + 6 * a * b + 9 * b^2 + 4 * c^2

theorem minimum_value : ∃ (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a * b * c = 27), 
  problem_statement a b c h = 180 :=
sorry

end minimum_value_l159_159676


namespace angle_inclusion_l159_159543

-- Defining the sets based on the given conditions
def M : Set ℝ := { x | 0 < x ∧ x ≤ 90 }
def N : Set ℝ := { x | 0 < x ∧ x < 90 }
def P : Set ℝ := { x | 0 ≤ x ∧ x ≤ 90 }

-- The proof statement
theorem angle_inclusion : N ⊆ M ∧ M ⊆ P :=
by
  sorry

end angle_inclusion_l159_159543


namespace parallelogram_height_l159_159702

variable (base height area : ℝ)
variable (h_eq_diag : base = 30)
variable (h_eq_area : area = 600)

theorem parallelogram_height :
  (height = 20) ↔ (base * height = area) :=
by
  sorry

end parallelogram_height_l159_159702


namespace express_x_n_prove_inequality_l159_159211

variable (a b n : Real)
variable (x : ℕ → Real)

def trapezoid_conditions : Prop :=
  ∀ n, x 1 = a * b / (a + b) ∧ (x (n + 1) / x n = x (n + 1) / a)

theorem express_x_n (h : trapezoid_conditions a b x) : 
  ∀ n, x n = a * b / (a + n * b) := 
by
  sorry

theorem prove_inequality (h : trapezoid_conditions a b x) : 
  ∀ n, x n ≤ (a + n * b) / (4 * n) := 
by
  sorry

end express_x_n_prove_inequality_l159_159211


namespace andrea_sod_rectangles_l159_159914

def section_1_length : ℕ := 35
def section_1_width : ℕ := 42
def section_2_length : ℕ := 55
def section_2_width : ℕ := 86
def section_3_length : ℕ := 20
def section_3_width : ℕ := 50
def section_4_length : ℕ := 48
def section_4_width : ℕ := 66

def sod_length : ℕ := 3
def sod_width : ℕ := 4

def area (length width : ℕ) : ℕ := length * width
def sod_area : ℕ := area sod_length sod_width

def rectangles_needed (section_length section_width sod_area : ℕ) : ℕ :=
  (area section_length section_width + sod_area - 1) / sod_area

def total_rectangles_needed : ℕ :=
  rectangles_needed section_1_length section_1_width sod_area +
  rectangles_needed section_2_length section_2_width sod_area +
  rectangles_needed section_3_length section_3_width sod_area +
  rectangles_needed section_4_length section_4_width sod_area

theorem andrea_sod_rectangles : total_rectangles_needed = 866 := by
  sorry

end andrea_sod_rectangles_l159_159914


namespace deposit_amount_correct_l159_159953

noncomputable def deposit_amount (initial_amount : ℝ) : ℝ :=
  let first_step := 0.30 * initial_amount
  let second_step := 0.25 * first_step
  0.20 * second_step

theorem deposit_amount_correct :
  deposit_amount 50000 = 750 :=
by
  sorry

end deposit_amount_correct_l159_159953


namespace more_crayons_than_erasers_l159_159587

theorem more_crayons_than_erasers
  (E : ℕ) (C : ℕ) (C_left : ℕ) (E_left : ℕ)
  (hE : E = 457) (hC : C = 617) (hC_left : C_left = 523) (hE_left : E_left = E) :
  C_left - E_left = 66 := 
by
  sorry

end more_crayons_than_erasers_l159_159587


namespace find_second_number_l159_159315

variable (n : ℕ)

theorem find_second_number (h : 8000 * n = 480 * 10^5) : n = 6000 :=
by
  sorry

end find_second_number_l159_159315


namespace range_of_a_l159_159250

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x > 0 → x / (x^2 + 3 * x + 1) ≤ a) → a ≥ 1/5 :=
by
  intro h
  sorry

end range_of_a_l159_159250


namespace max_distance_from_ellipse_to_line_l159_159274

theorem max_distance_from_ellipse_to_line :
  let ellipse (x y : ℝ) := (x^2 / 16) + (y^2 / 4) = 1
  let line (x y : ℝ) := x + 2 * y - Real.sqrt 2 = 0
  ∃ (d : ℝ), (∀ (x y : ℝ), ellipse x y → line x y → d = Real.sqrt 10) :=
sorry

end max_distance_from_ellipse_to_line_l159_159274


namespace simple_interest_l159_159645

theorem simple_interest (TD : ℝ) (Sum : ℝ) (SI : ℝ) 
  (h1 : TD = 78) 
  (h2 : Sum = 947.1428571428571) 
  (h3 : SI = Sum - (Sum - TD)) : 
  SI = 78 := 
by 
  sorry

end simple_interest_l159_159645


namespace total_teams_l159_159005

theorem total_teams (m n : ℕ) (hmn : m > n) : 
  (m - n) + 1 = m - n + 1 := 
by sorry

end total_teams_l159_159005


namespace smallest_period_sin_cos_l159_159866

theorem smallest_period_sin_cos (f : ℝ → ℝ) (h : ∀ x, f x = Real.sin x + Real.cos x) :
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T' ≥ T) ∧ T = 2 * Real.pi :=
sorry

end smallest_period_sin_cos_l159_159866


namespace ping_pong_shaved_head_ping_pong_upset_l159_159728

noncomputable def probability_shaved_head (pA pB : ℚ) : ℚ :=
  pA^3 + pB^3

noncomputable def probability_upset (pB pA : ℚ) : ℚ :=
  (pB^3) + (3 * (pB^2) * pA) + (6 * (pA^2) * (pB^2))

theorem ping_pong_shaved_head :
  probability_shaved_head (2/3) (1/3) = 1/3 := 
by
  sorry

theorem ping_pong_upset :
  probability_upset (1/3) (2/3) = 11/27 := 
by
  sorry

end ping_pong_shaved_head_ping_pong_upset_l159_159728


namespace fourth_vertex_of_tetrahedron_exists_l159_159456

theorem fourth_vertex_of_tetrahedron_exists (x y z : ℤ) :
  (∃ (x y z : ℤ), 
     ((x - 1) ^ 2 + y ^ 2 + (z - 3) ^ 2 = 26) ∧ 
     ((x - 5) ^ 2 + (y - 3) ^ 2 + (z - 2) ^ 2 = 26) ∧ 
     ((x - 4) ^ 2 + y ^ 2 + (z - 6) ^ 2 = 26)) :=
sorry

end fourth_vertex_of_tetrahedron_exists_l159_159456


namespace present_age_ratio_l159_159271

theorem present_age_ratio (D J : ℕ) (h1 : Dan = 24) (h2 : James = 20) : Dan / James = 6 / 5 := by
  sorry

end present_age_ratio_l159_159271


namespace spherical_coordinates_cone_l159_159927

open Real

-- Define spherical coordinates and the equation φ = c
def spherical_coordinates (ρ θ φ : ℝ) : Prop := 
  ∃ (c : ℝ), φ = c

-- Prove that φ = c describes a cone
theorem spherical_coordinates_cone (ρ θ : ℝ) (c : ℝ) :
  spherical_coordinates ρ θ c → ∃ ρ' θ', spherical_coordinates ρ' θ' c :=
by
  sorry

end spherical_coordinates_cone_l159_159927


namespace animal_legs_l159_159312

theorem animal_legs (dogs chickens spiders octopus : Nat) (legs_dog legs_chicken legs_spider legs_octopus : Nat)
  (h1 : dogs = 3)
  (h2 : chickens = 4)
  (h3 : spiders = 2)
  (h4 : octopus = 1)
  (h5 : legs_dog = 4)
  (h6 : legs_chicken = 2)
  (h7 : legs_spider = 8)
  (h8 : legs_octopus = 8) :
  dogs * legs_dog + chickens * legs_chicken + spiders * legs_spider + octopus * legs_octopus = 44 := by
    sorry

end animal_legs_l159_159312


namespace expand_product_l159_159240

theorem expand_product (x : ℝ) : (x + 5) * (x - 16) = x^2 - 11 * x - 80 :=
by sorry

end expand_product_l159_159240


namespace division_of_fractions_l159_159819

theorem division_of_fractions :
  (5 / 6) / (9 / 10) = 25 / 27 :=
by sorry

end division_of_fractions_l159_159819


namespace same_color_probability_l159_159174

/-- There are 7 red plates and 5 blue plates. We want to prove that the probability of
    selecting 3 plates, where all are of the same color, is 9/44. -/
theorem same_color_probability :
  let total_plates := 12
  let total_ways_to_choose := Nat.choose total_plates 3
  let red_plates := 7
  let blue_plates := 5
  let ways_to_choose_red := Nat.choose red_plates 3
  let ways_to_choose_blue := Nat.choose blue_plates 3
  let favorable_ways_to_choose := ways_to_choose_red + ways_to_choose_blue
  ∃ (prob : ℚ), prob = (favorable_ways_to_choose : ℚ) / (total_ways_to_choose : ℚ) ∧
                 prob = 9 / 44 :=
by
  sorry

end same_color_probability_l159_159174


namespace perpendicular_lines_solve_for_a_l159_159984

theorem perpendicular_lines_solve_for_a :
  ∀ (a : ℝ), 
  ((3 * a + 2) * (5 * a - 2) + (1 - 4 * a) * (a + 4) = 0) → 
  (a = 0 ∨ a = 1) :=
by
  intro a h
  sorry

end perpendicular_lines_solve_for_a_l159_159984


namespace jill_more_than_jake_l159_159551

-- Definitions from conditions
def jill_peaches := 12
def steven_peaches := jill_peaches + 15
def jake_peaches := steven_peaches - 16

-- Theorem to prove the question == answer given conditions
theorem jill_more_than_jake : jill_peaches - jake_peaches = 1 :=
by
  -- Proof steps would be here, but for the statement requirement we put sorry
  sorry

end jill_more_than_jake_l159_159551


namespace min_value_f_when_a_is_zero_inequality_holds_for_f_l159_159837

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x - a * x^2 - 2 * x

-- Problem (1): Prove the minimum value of f(x) when a = 0 is 2 - 2 * ln 2.
theorem min_value_f_when_a_is_zero : 
  (∃ x : ℝ, f x 0 = 2 - 2 * Real.log 2) :=
sorry

-- Problem (2): Prove that for a < (exp(1) / 2) - 1, f(x) > (exp(1) / 2) - 1 for all x in (0, +∞).
theorem inequality_holds_for_f :
  ∀ a : ℝ, a < (Real.exp 1) / 2 - 1 → 
  ∀ x : ℝ, 0 < x → f x a > (Real.exp 1) / 2 - 1 :=
sorry

end min_value_f_when_a_is_zero_inequality_holds_for_f_l159_159837


namespace number_of_children_l159_159453

theorem number_of_children (total_oranges : ℕ) (oranges_per_child : ℕ) (h1 : oranges_per_child = 3) (h2 : total_oranges = 12) : total_oranges / oranges_per_child = 4 :=
by
  sorry

end number_of_children_l159_159453


namespace calculate_avg_l159_159388

def avg2 (a b : ℚ) : ℚ := (a + b) / 2
def avg3 (a b c : ℚ) : ℚ := (a + b + c) / 3

theorem calculate_avg :
  avg3 (avg3 1 2 0) (avg2 0 2) 0 = 2 / 3 :=
by
  sorry

end calculate_avg_l159_159388


namespace S6_geometric_sum_l159_159381

noncomputable def geometric_sequence_sum (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

theorem S6_geometric_sum (a r : ℝ)
    (sum_n : ℕ → ℝ)
    (geo_seq : ∀ n, sum_n n = geometric_sequence_sum a r n)
    (S2 : sum_n 2 = 6)
    (S4 : sum_n 4 = 30) :
    sum_n 6 = 126 := 
by
  sorry

end S6_geometric_sum_l159_159381


namespace complex_arithmetic_problem_l159_159705
open Complex

theorem complex_arithmetic_problem : (2 - 3 * Complex.I) * (2 + 3 * Complex.I) + (4 - 5 * Complex.I)^2 = 4 - 40 * Complex.I := by
  sorry

end complex_arithmetic_problem_l159_159705


namespace integers_satisfying_condition_l159_159698

-- Define the condition
def condition (x : ℤ) : Prop := x * x < 3 * x

-- Define the theorem stating the proof problem
theorem integers_satisfying_condition :
  {x : ℤ | condition x} = {1, 2} :=
by
  sorry

end integers_satisfying_condition_l159_159698


namespace Leila_donated_2_bags_l159_159151

theorem Leila_donated_2_bags (L : ℕ) (h1 : 25 * L + 7 = 57) : L = 2 :=
by
  sorry

end Leila_donated_2_bags_l159_159151


namespace expanded_figure_perimeter_l159_159319

def side_length : ℕ := 2
def bottom_row_squares : ℕ := 3
def total_squares : ℕ := 4

def perimeter (side_length : ℕ) (bottom_row_squares : ℕ) (total_squares: ℕ) : ℕ :=
  2 * side_length * (bottom_row_squares + 1)

theorem expanded_figure_perimeter : perimeter side_length bottom_row_squares total_squares = 20 :=
by
  sorry

end expanded_figure_perimeter_l159_159319


namespace willie_initial_bananas_l159_159326

/-- Given that Willie will have 13 bananas, we need to prove that the initial number of bananas Willie had was some specific number X. --/
theorem willie_initial_bananas (initial_bananas : ℕ) (final_bananas : ℕ) 
    (h : final_bananas = 13) : initial_bananas = initial_bananas :=
by
  sorry

end willie_initial_bananas_l159_159326


namespace remainder_777_777_mod_13_l159_159365

theorem remainder_777_777_mod_13 : (777 ^ 777) % 13 = 12 := 
by 
  -- Proof steps would go here
  sorry

end remainder_777_777_mod_13_l159_159365


namespace snail_stops_at_25_26_l159_159747

def grid_width : ℕ := 300
def grid_height : ℕ := 50

def initial_position : ℕ × ℕ := (1, 1)

def snail_moves_in_spiral (w h : ℕ) (initial : ℕ × ℕ) : ℕ × ℕ := (25, 26)

theorem snail_stops_at_25_26 :
  snail_moves_in_spiral grid_width grid_height initial_position = (25, 26) :=
sorry

end snail_stops_at_25_26_l159_159747


namespace juice_water_ratio_l159_159648

theorem juice_water_ratio (V : ℝ) :
  let glass_juice_ratio := (2, 1)
  let mug_volume := 2 * V
  let mug_juice_ratio := (4, 1)
  let glass_juice_vol := (2 / 3) * V
  let glass_water_vol := (1 / 3) * V
  let mug_juice_vol := (8 / 5) * V
  let mug_water_vol := (2 / 5) * V
  let total_juice := glass_juice_vol + mug_juice_vol
  let total_water := glass_water_vol + mug_water_vol
  let ratio := total_juice / total_water
  ratio = 34 / 11 :=
by
  sorry

end juice_water_ratio_l159_159648


namespace multiplication_value_l159_159733

theorem multiplication_value : 725143 * 999999 = 725142274857 :=
by
  sorry

end multiplication_value_l159_159733


namespace maximum_area_of_rectangle_with_given_perimeter_l159_159013

theorem maximum_area_of_rectangle_with_given_perimeter {x y : ℕ} (h₁ : 2 * x + 2 * y = 160) : 
  (∃ x y : ℕ, 2 * x + 2 * y = 160 ∧ x * y = 1600) := 
sorry

end maximum_area_of_rectangle_with_given_perimeter_l159_159013


namespace set_intersection_l159_159506

def A := {x : ℝ | x^2 - 3*x ≥ 0}
def B := {x : ℝ | x < 1}
def intersection := {x : ℝ | x ≤ 0}

theorem set_intersection : A ∩ B = intersection :=
  sorry

end set_intersection_l159_159506


namespace product_divisible_by_12_l159_159166

theorem product_divisible_by_12 (a b c d : ℤ) :
  12 ∣ (b - a) * (c - a) * (d - a) * (d - c) * (d - b) * (c - b) := 
by {
  sorry
}

end product_divisible_by_12_l159_159166


namespace fifth_number_l159_159197

def sequence_sum (a b : ℕ) : ℕ :=
  a + b + (a + b) + (a + 2 * b) + (2 * a + 3 * b) + (3 * a + 5 * b)

theorem fifth_number (a b : ℕ) (h : sequence_sum a b = 2008) : 2 * a + 3 * b = 502 := by
  sorry

end fifth_number_l159_159197


namespace components_le_20_components_le_n_squared_div_4_l159_159010

-- Question part b: 8x8 grid, can the number of components be more than 20
theorem components_le_20 {c : ℕ} (h1 : c = 64 / 4) : c ≤ 20 := by
  sorry

-- Question part c: n x n grid, can the number of components be more than n^2 / 4
theorem components_le_n_squared_div_4 (n : ℕ) (h2 : n > 8) {c : ℕ} (h3 : c = n^2 / 4) : 
  c ≤ n^2 / 4 := by
  sorry

end components_le_20_components_le_n_squared_div_4_l159_159010


namespace find_c_solution_l159_159956

theorem find_c_solution {c : ℚ} 
  (h₁ : ∃ x : ℤ, 2 * (x : ℚ)^2 + 17 * x - 55 = 0 ∧ x = ⌊c⌋)
  (h₂ : ∃ x : ℚ, 6 * x^2 - 23 * x + 7 = 0 ∧ 0 ≤ x ∧ x < 1 ∧ x = c - ⌊c⌋) :
  c = -32 / 3 :=
by
  sorry

end find_c_solution_l159_159956


namespace product_of_coefficients_l159_159799

theorem product_of_coefficients (b c : ℤ)
  (H1 : ∀ r, r^2 - 2 * r - 1 = 0 → r^5 - b * r - c = 0):
  b * c = 348 :=
by
  -- Solution steps would go here
  sorry

end product_of_coefficients_l159_159799


namespace identical_sets_l159_159792

def A : Set ℝ := {x : ℝ | ∃ y : ℝ, y = x^2 + 1}
def B : Set ℝ := {y : ℝ | ∃ x : ℝ, y = x^2 + 1}
def C : Set (ℝ × ℝ) := {(x, y) : ℝ × ℝ | y = x^2 + 1}
def D : Set ℝ := {y : ℝ | 1 ≤ y}

theorem identical_sets : B = D :=
by
  sorry

end identical_sets_l159_159792


namespace simplify_correct_l159_159268

open Polynomial

noncomputable def simplify_expression (y : ℚ) : Polynomial ℚ :=
  (3 * (Polynomial.C y) + 2) * (2 * (Polynomial.C y)^12 + 3 * (Polynomial.C y)^11 - (Polynomial.C y)^9 - (Polynomial.C y)^8)

theorem simplify_correct (y : ℚ) : 
  simplify_expression y = 6 * (Polynomial.C y)^13 + 13 * (Polynomial.C y)^12 + 6 * (Polynomial.C y)^11 - 3 * (Polynomial.C y)^10 - 5 * (Polynomial.C y)^9 - 2 * (Polynomial.C y)^8 := 
by 
  simp [simplify_expression]
  sorry

end simplify_correct_l159_159268


namespace algebraic_expression_value_l159_159756

noncomputable def a := Real.sqrt 2 + 1
noncomputable def b := Real.sqrt 2 - 1

theorem algebraic_expression_value : (a^2 - 2 * a * b + b^2) / (a^2 - b^2) = Real.sqrt 2 / 2 := by
  sorry

end algebraic_expression_value_l159_159756


namespace min_sum_of_angles_l159_159027

theorem min_sum_of_angles (A B C : ℝ) (hABC : A + B + C = 180) (h_sin : Real.sin A + Real.sin B + Real.sin C ≤ 1) : 
  min (A + B) (min (B + C) (C + A)) < 30 := 
sorry

end min_sum_of_angles_l159_159027


namespace playground_area_l159_159371

noncomputable def length (w : ℝ) := 2 * w + 30
noncomputable def perimeter (l w : ℝ) := 2 * (l + w)
noncomputable def area (l w : ℝ) := l * w

theorem playground_area :
  ∃ (w l : ℝ), length w = l ∧ perimeter l w = 700 ∧ area l w = 25955.56 :=
by {
  sorry
}

end playground_area_l159_159371


namespace problem_statement_l159_159150

def nabla (a b : ℕ) : ℕ := 3 + b ^ a

theorem problem_statement : nabla (nabla 2 3) 4 = 16777219 := by
  sorry

end problem_statement_l159_159150


namespace find_B_from_period_l159_159469

theorem find_B_from_period (A B C D : ℝ) (h : B ≠ 0) (period_condition : 2 * |2 * π / B| = 4 * π) : B = 1 := sorry

end find_B_from_period_l159_159469


namespace largest_among_four_l159_159136

theorem largest_among_four (a b : ℝ) (ha : a > 0) (hb : b < 0) :
  max (max a (max (a + b) (a - b))) (ab) = a - b :=
by {
  sorry
}

end largest_among_four_l159_159136


namespace necessary_condition_l159_159412

theorem necessary_condition :
  (∀ x : ℝ, (1 / x < 3) → (x > 1 / 3)) → (∀ x : ℝ, (1 / x < 3) ↔ (x > 1 / 3)) → False :=
by
  sorry

end necessary_condition_l159_159412


namespace value_of_expression_l159_159048

variable {a : Nat → Int}

def arithmetic_sequence (a : Nat → Int) : Prop :=
  ∀ n m : Nat, a (n + 1) - a n = a (m + 1) - a m

theorem value_of_expression
  (h_arith_seq : arithmetic_sequence a)
  (h_sum : a 4 + a 6 + a 8 + a 10 + a 12 = 120) :
  2 * a 10 - a 12 = 24 :=
  sorry

end value_of_expression_l159_159048


namespace completing_square_to_simplify_eq_l159_159936

theorem completing_square_to_simplify_eq : 
  ∃ (c : ℝ), (∀ x : ℝ, x^2 - 6 * x + 4 = 0 ↔ (x - 3)^2 = c) :=
by
  use 5
  intro x
  constructor
  { intro h
    -- proof conversion process (skipped)
    sorry }
  { intro h
    -- reverse proof process (skipped)
    sorry }

end completing_square_to_simplify_eq_l159_159936


namespace t_shirt_cost_calculation_l159_159077

variables (initial_amount ticket_cost food_cost money_left t_shirt_cost : ℕ)

axiom h1 : initial_amount = 75
axiom h2 : ticket_cost = 30
axiom h3 : food_cost = 13
axiom h4 : money_left = 9

theorem t_shirt_cost_calculation : 
  t_shirt_cost = initial_amount - (ticket_cost + food_cost) - money_left :=
sorry

end t_shirt_cost_calculation_l159_159077


namespace first_installment_amount_l159_159428

-- Define the conditions stated in the problem
def original_price : ℝ := 480
def discount_rate : ℝ := 0.05
def monthly_installment : ℝ := 102
def number_of_installments : ℕ := 3

-- The final price after discount
def final_price : ℝ := original_price * (1 - discount_rate)

-- The total amount of the 3 monthly installments
def total_of_3_installments : ℝ := monthly_installment * number_of_installments

-- The first installment paid
def first_installment : ℝ := final_price - total_of_3_installments

-- The main theorem to prove the first installment amount
theorem first_installment_amount : first_installment = 150 := by
  unfold first_installment
  unfold final_price
  unfold total_of_3_installments
  unfold original_price
  unfold discount_rate
  unfold monthly_installment
  unfold number_of_installments
  sorry

end first_installment_amount_l159_159428


namespace sum_of_powers_l159_159637

theorem sum_of_powers (m n : ℤ)
  (h1 : m + n = 1)
  (h2 : m^2 + n^2 = 3)
  (h3 : m^3 + n^3 = 4)
  (h4 : m^4 + n^4 = 7)
  (h5 : m^5 + n^5 = 11) :
  m^9 + n^9 = 76 :=
sorry

end sum_of_powers_l159_159637


namespace man_speed_l159_159795

theorem man_speed {m l: ℝ} (TrainLength : ℝ := 385) (TrainSpeedKmH : ℝ := 60)
  (PassTimeSeconds : ℝ := 21) (RelativeSpeed : ℝ) (ManSpeedKmH : ℝ) 
  (ConversionFactor : ℝ := 3.6) (expected_speed : ℝ := 5.99) : 
  RelativeSpeed = TrainSpeedKmH/ConversionFactor + m/ConversionFactor ∧ 
  TrainLength = RelativeSpeed * PassTimeSeconds →
  abs (m*ConversionFactor - expected_speed) < 0.01 :=
by
  sorry

end man_speed_l159_159795


namespace zeros_in_decimal_representation_l159_159946

def term_decimal_zeros (x : ℚ) : ℕ := sorry  -- Function to calculate the number of zeros in the terminating decimal representation.

theorem zeros_in_decimal_representation :
  term_decimal_zeros (1 / (2^7 * 5^9)) = 8 :=
sorry

end zeros_in_decimal_representation_l159_159946


namespace trader_loss_percentage_l159_159594

theorem trader_loss_percentage :
  let SP := 325475
  let gain := 14 / 100
  let loss := 14 / 100
  let CP1 := SP / (1 + gain)
  let CP2 := SP / (1 - loss)
  let TCP := CP1 + CP2
  let TSP := SP + SP
  let profit_or_loss := TSP - TCP
  let profit_or_loss_percentage := (profit_or_loss / TCP) * 100
  profit_or_loss_percentage = -1.958 :=
by
  sorry

end trader_loss_percentage_l159_159594


namespace square_area_l159_159444

theorem square_area (x : ℝ) (h1 : BG = GH) (h2 : GH = HD) (h3 : BG = 20 * Real.sqrt 2) : x = 40 * Real.sqrt 2 → x^2 = 3200 :=
by
  sorry

end square_area_l159_159444


namespace remainder_of_2365487_div_3_l159_159639

theorem remainder_of_2365487_div_3 : (2365487 % 3) = 2 := by
  sorry

end remainder_of_2365487_div_3_l159_159639


namespace volume_of_A_is_2800_l159_159118

-- Define the dimensions of the fishbowl and water heights
def fishbowl_side_length : ℝ := 20
def height_with_A : ℝ := 16
def height_without_A : ℝ := 9

-- Compute the volume of water with and without object (A)
def volume_with_A : ℝ := fishbowl_side_length ^ 2 * height_with_A
def volume_without_A : ℝ := fishbowl_side_length ^ 2 * height_without_A

-- The volume of object (A)
def volume_A : ℝ := volume_with_A - volume_without_A

-- Prove that this volume is 2800 cubic centimeters
theorem volume_of_A_is_2800 :
  volume_A = 2800 := by
  sorry

end volume_of_A_is_2800_l159_159118


namespace harbor_distance_l159_159852

-- Definitions from conditions
variable (d : ℝ)

-- Define the assumptions
def condition_dave := d < 10
def condition_elena := d > 9

-- The proof statement that the interval for d is (9, 10)
theorem harbor_distance (hd : condition_dave d) (he : condition_elena d) : d ∈ Set.Ioo 9 10 :=
sorry

end harbor_distance_l159_159852


namespace rebecca_tent_stakes_l159_159770

variables (T D W : ℕ)

-- Conditions
def drink_mix_eq : Prop := D = 3 * T
def water_eq : Prop := W = T + 2
def total_items_eq : Prop := T + D + W = 22

-- Problem statement
theorem rebecca_tent_stakes 
  (h1 : drink_mix_eq T D)
  (h2 : water_eq T W)
  (h3 : total_items_eq T D W) : 
  T = 4 := 
sorry

end rebecca_tent_stakes_l159_159770


namespace equivalent_expression_l159_159508

theorem equivalent_expression :
  (2 + 3) * (2^2 + 3^2) * (2^4 + 3^4) * (2^8 + 3^8) * (2^16 + 3^16) * (2^32 + 3^32) * (2^64 + 3^64) = 3^128 - 2^128 :=
sorry

end equivalent_expression_l159_159508


namespace probability_of_picking_letter_from_mathematics_l159_159363

-- Definition of the problem conditions
def extended_alphabet_size := 30
def distinct_letters_in_mathematics := 8

-- Theorem statement
theorem probability_of_picking_letter_from_mathematics :
  (distinct_letters_in_mathematics / extended_alphabet_size : ℚ) = 4 / 15 := 
by 
  sorry

end probability_of_picking_letter_from_mathematics_l159_159363


namespace tan_expression_val_l159_159417

theorem tan_expression_val (A B : ℝ) (hA : A = 30) (hB : B = 15) :
  (1 + Real.tan (A * Real.pi / 180)) * (1 + Real.tan (B * Real.pi / 180)) = 2 :=
by
  sorry

end tan_expression_val_l159_159417


namespace doug_initial_marbles_l159_159432

theorem doug_initial_marbles (E D : ℕ) (H1 : E = D + 5) (H2 : E = 27) : D = 22 :=
by
  -- proof provided here would infer the correct answer from the given conditions
  sorry

end doug_initial_marbles_l159_159432


namespace area_of_section_ABD_l159_159539
-- Import everything from the Mathlib library

-- Define the conditions
def is_equilateral_triangle (a b c : ℝ) (ABC_angle : ℝ) : Prop := 
  a = b ∧ b = c ∧ ABC_angle = 60

def plane_angle (angle : ℝ) : Prop := 
  angle = 35 + 18/60

def volume_of_truncated_pyramid (volume : ℝ) : Prop := 
  volume = 15

-- The main theorem based on the above conditions
theorem area_of_section_ABD
  (a b c ABC_angle : ℝ)
  (S : ℝ)
  (V : ℝ)
  (h1 : is_equilateral_triangle a b c ABC_angle)
  (h2 : plane_angle S)
  (h3 : volume_of_truncated_pyramid V) :
  ∃ (area : ℝ), area = 16.25 :=
by
  -- skipping the proof
  sorry

end area_of_section_ABD_l159_159539


namespace trapezoid_angles_l159_159546

-- Definition of the problem statement in Lean 4
theorem trapezoid_angles (A B C D : ℝ) (h1 : A = 60) (h2 : B = 130)
  (h3 : A + D = 180) (h4 : B + C = 180) (h_sum : A + B + C + D = 360) :
  C = 50 ∧ D = 120 :=
by
  sorry

end trapezoid_angles_l159_159546


namespace angle_B_in_triangle_l159_159875

theorem angle_B_in_triangle (A B C : ℝ) (hA : A = 60) (hB : B = 2 * C) (hSum : A + B + C = 180) : B = 80 :=
by sorry

end angle_B_in_triangle_l159_159875


namespace solve_fraction_equation_l159_159930

theorem solve_fraction_equation (x : ℝ) (hx1 : 0 < x) (hx2 : (x - 6) / 12 = 6 / (x - 12)) : x = 18 := 
sorry

end solve_fraction_equation_l159_159930


namespace range_of_x_l159_159076

theorem range_of_x (x : ℝ) : (x ≠ -3) ∧ (x ≤ 4) ↔ (x ≤ 4) ∧ (x ≠ -3) :=
by { sorry }

end range_of_x_l159_159076


namespace number_of_children_l159_159672

def pencils_per_child : ℕ := 2
def total_pencils : ℕ := 16

theorem number_of_children : total_pencils / pencils_per_child = 8 :=
by
  sorry

end number_of_children_l159_159672


namespace seed_total_after_trading_l159_159743

theorem seed_total_after_trading :
  ∀ (Bom Gwi Yeon Eun : ℕ),
  Yeon = 3 * Gwi →
  Gwi = Bom + 40 →
  Eun = 2 * Gwi →
  Bom = 300 →
  Yeon_gives = 20 * Yeon / 100 →
  Bom_gives = 50 →
  let Yeon_after := Yeon - Yeon_gives
  let Gwi_after := Gwi + Yeon_gives
  let Bom_after := Bom - Bom_gives
  let Eun_after := Eun + Bom_gives
  Bom_after + Gwi_after + Yeon_after + Eun_after = 2340 :=
by
  intros Bom Gwi Yeon Eun hYeon hGwi hEun hBom hYeonGives hBomGives Yeon_after Gwi_after Bom_after Eun_after
  sorry

end seed_total_after_trading_l159_159743


namespace kids_on_soccer_field_l159_159359

theorem kids_on_soccer_field (n f : ℕ) (h1 : n = 14) (h2 : f = 3) :
  n + n * f = 56 :=
by
  sorry

end kids_on_soccer_field_l159_159359


namespace sum_consecutive_not_power_of_two_l159_159333

theorem sum_consecutive_not_power_of_two :
  ∀ n k : ℕ, ∀ x : ℕ, n > 0 → k > 0 → (n * (n + 2 * k - 1)) / 2 ≠ 2 ^ x := by
  sorry

end sum_consecutive_not_power_of_two_l159_159333


namespace sum_of_first_six_terms_l159_159940

theorem sum_of_first_six_terms (S : ℕ → ℝ) (a : ℕ → ℝ) (hS : ∀ n, S (n + 1) = S n + a (n + 1)) :
  S 2 = 2 → S 4 = 10 → S 6 = 24 := 
by
  intros h1 h2
  sorry

end sum_of_first_six_terms_l159_159940


namespace trevor_quarters_counted_l159_159466

-- Define the conditions from the problem
variable (Q D : ℕ) 
variable (total_coins : ℕ := 77)
variable (excess : ℕ := 48)

-- Use the conditions to assert the existence of quarters and dimes such that the totals align with the given constraints
theorem trevor_quarters_counted : (Q + D = total_coins) ∧ (D = Q + excess) → Q = 29 :=
by
  -- Add sorry to skip the actual proof, as we are only writing the statement
  sorry

end trevor_quarters_counted_l159_159466


namespace problem_solution_l159_159778

theorem problem_solution (a b c d : ℕ) (h : 342 * (a * b * c * d + a * b + a * d + c * d + 1) = 379 * (b * c * d + b + d)) :
  (a * 10^3 + b * 10^2 + c * 10 + d) = 1949 :=
by
  sorry

end problem_solution_l159_159778


namespace basketball_game_l159_159668

theorem basketball_game (E H : ℕ) (h1 : E = H + 18) (h2 : E + H = 50) : H = 16 :=
by
  sorry

end basketball_game_l159_159668


namespace afternoon_registration_l159_159047

variable (m a t morning_absent : ℕ)

theorem afternoon_registration (m a t morning_absent afternoon : ℕ) (h1 : m = 25) (h2 : a = 4) (h3 : t = 42) (h4 : morning_absent = 3) : 
  afternoon = t - (m - morning_absent + morning_absent + a) :=
by sorry

end afternoon_registration_l159_159047


namespace valerie_needs_72_stamps_l159_159031

noncomputable def total_stamps_needed : ℕ :=
  let thank_you_cards := 5
  let stamps_per_thank_you := 2
  let water_bill_stamps := 3
  let electric_bill_stamps := 2
  let internet_bill_stamps := 5
  let rebates_more_than_bills := 3
  let rebate_stamps := 2
  let job_applications_factor := 2
  let job_application_stamps := 1

  let total_thank_you_stamps := thank_you_cards * stamps_per_thank_you
  let total_bill_stamps := water_bill_stamps + electric_bill_stamps + internet_bill_stamps
  let total_rebates := total_bill_stamps + rebates_more_than_bills
  let total_rebate_stamps := total_rebates * rebate_stamps
  let total_job_applications := total_rebates * job_applications_factor
  let total_job_application_stamps := total_job_applications * job_application_stamps

  total_thank_you_stamps + total_bill_stamps + total_rebate_stamps + total_job_application_stamps

theorem valerie_needs_72_stamps : total_stamps_needed = 72 :=
  by
    sorry

end valerie_needs_72_stamps_l159_159031


namespace total_games_in_conference_l159_159284

-- Definitions based on the conditions
def numTeams := 16
def divisionTeams := 8
def gamesWithinDivisionPerTeam := 21
def gamesAcrossDivisionPerTeam := 16
def totalGamesPerTeam := 37
def totalGameCount := 592
def actualGameCount := 296

-- Proof statement
theorem total_games_in_conference : actualGameCount = (totalGameCount / 2) :=
  by sorry

end total_games_in_conference_l159_159284


namespace expected_scurried_home_mn_sum_l159_159807

theorem expected_scurried_home_mn_sum : 
  let expected_fraction : ℚ := (1/2 + 2/3 + 3/4 + 4/5 + 5/6 + 6/7 + 7/8)
  let m : ℕ := 37
  let n : ℕ := 7
  m + n = 44 := by
  sorry

end expected_scurried_home_mn_sum_l159_159807


namespace num_of_chairs_per_row_l159_159243

theorem num_of_chairs_per_row (total_chairs : ℕ) (num_rows : ℕ) (chairs_per_row : ℕ)
  (h1 : total_chairs = 432)
  (h2 : num_rows = 27) :
  total_chairs = num_rows * chairs_per_row ↔ chairs_per_row = 16 :=
by
  sorry

end num_of_chairs_per_row_l159_159243


namespace no_rational_roots_of_polynomial_l159_159175

theorem no_rational_roots_of_polynomial :
  ¬ ∃ (x : ℚ), (3 * x^4 - 7 * x^3 - 4 * x^2 + 8 * x + 3 = 0) :=
by
  sorry

end no_rational_roots_of_polynomial_l159_159175


namespace peaches_sold_to_friends_l159_159847

theorem peaches_sold_to_friends (x : ℕ) (total_peaches : ℕ) (peaches_to_relatives : ℕ) (peach_price_friend : ℕ) (peach_price_relative : ℝ) (total_earnings : ℝ) (peaches_left : ℕ) (total_peaches_sold : ℕ) 
  (h1 : total_peaches = 15) 
  (h2 : peaches_to_relatives = 4) 
  (h3 : peach_price_relative = 1.25) 
  (h4 : total_earnings = 25) 
  (h5 : peaches_left = 1)
  (h6 : total_peaches_sold = 14)
  (h7 : total_earnings = peach_price_friend * x + peach_price_relative * peaches_to_relatives)
  (h8 : total_peaches_sold = total_peaches - peaches_left) :
  x = 10 := 
sorry

end peaches_sold_to_friends_l159_159847


namespace num_solutions_eq_3_l159_159732

theorem num_solutions_eq_3 : 
  ∃ (x1 x2 x3 : ℝ), (∀ x : ℝ, 2^x - 2 * (⌊x⌋:ℝ) - 1 = 0 → x = x1 ∨ x = x2 ∨ x = x3) 
  ∧ ¬ ∃ x4, (2^x4 - 2 * (⌊x4⌋:ℝ) - 1 = 0 ∧ x4 ≠ x1 ∧ x4 ≠ x2 ∧ x4 ≠ x3) :=
sorry

end num_solutions_eq_3_l159_159732


namespace sandwich_price_l159_159072

-- Definitions based on conditions
def price_of_soda : ℝ := 0.87
def total_cost : ℝ := 6.46
def num_soda : ℝ := 4
def num_sandwich : ℝ := 2

-- The key equation based on conditions
def total_cost_equation (S : ℝ) : Prop := 
  num_sandwich * S + num_soda * price_of_soda = total_cost

theorem sandwich_price :
  ∃ S : ℝ, total_cost_equation S ∧ S = 1.49 :=
by
  sorry

end sandwich_price_l159_159072


namespace odd_function_condition_l159_159313

noncomputable def f (x a b : ℝ) : ℝ :=
  x * abs (x + a) + b

theorem odd_function_condition (a b : ℝ) :
  (∀ x : ℝ, f x a b = f (-x) a b) ↔ (a = 0 ∧ b = 0) := 
by
  sorry

end odd_function_condition_l159_159313


namespace distance_to_second_museum_l159_159476

theorem distance_to_second_museum (d x : ℕ) (h1 : d = 5) (h2 : 2 * d + 2 * x = 40) : x = 15 :=
by
  sorry

end distance_to_second_museum_l159_159476


namespace minimize_cost_l159_159538

theorem minimize_cost (x : ℝ) (h1 : 0 < x) (h2 : 400 / x * 40 ≤ 4 * x) : x = 20 :=
by
  sorry

end minimize_cost_l159_159538


namespace inequality_proof_l159_159597

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^3 + 1 / b^3 - 1) * (b^3 + 1 / c^3 - 1) * (c^3 + 1 / a^3 - 1) ≤ (a * b * c + 1 / (a * b * c) - 1)^3 :=
by
  sorry

end inequality_proof_l159_159597


namespace room_tiling_problem_correct_l159_159634

noncomputable def room_tiling_problem : Prop :=
  let room_length := 6.72
  let room_width := 4.32
  let tile_size := 0.3
  let room_area := room_length * room_width
  let tile_area := tile_size * tile_size
  let num_tiles := (room_area / tile_area).ceil
  num_tiles = 323

theorem room_tiling_problem_correct : room_tiling_problem := 
  sorry

end room_tiling_problem_correct_l159_159634


namespace exists_c_with_same_nonzero_decimal_digits_l159_159995

theorem exists_c_with_same_nonzero_decimal_digits (m n : ℕ) (hm : 0 < m) (hn : 0 < n) : 
  ∃ (c : ℕ), 0 < c ∧ (∃ (k : ℕ), (c * m) % 10^k = (c * n) % 10^k) := 
sorry

end exists_c_with_same_nonzero_decimal_digits_l159_159995


namespace find_minimal_N_l159_159870

theorem find_minimal_N (N : ℕ) (l m n : ℕ) (h1 : (l - 1) * (m - 1) * (n - 1) = 252)
  (h2 : l ≥ 5 ∨ m ≥ 5 ∨ n ≥ 5) : N = l * m * n → N = 280 :=
by
  sorry

end find_minimal_N_l159_159870


namespace fixed_point_of_inverse_l159_159120

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a ^ (x - 1) + 4

theorem fixed_point_of_inverse (a : ℝ) (ha_pos : 0 < a) (ha_ne_one : a ≠ 1) :
  f a (5) = 1 :=
by
  unfold f
  sorry

end fixed_point_of_inverse_l159_159120


namespace equidistant_trajectory_l159_159187

theorem equidistant_trajectory {x y z : ℝ} :
  (x + 1)^2 + (y - 1)^2 + z^2 = (x - 2)^2 + (y + 1)^2 + (z + 1)^2 →
  3 * x - 2 * y - z = 2 :=
sorry

end equidistant_trajectory_l159_159187


namespace triangle_area_ratio_l159_159060

noncomputable def vector_sum_property (OA OB OC : ℝ × ℝ × ℝ) : Prop :=
  OA + (2 : ℝ) • OB + (3 : ℝ) • OC = (0 : ℝ × ℝ × ℝ)

noncomputable def area_ratio (S_ABC S_AOC : ℝ) : Prop :=
  S_ABC / S_AOC = 3

theorem triangle_area_ratio
    (OA OB OC : ℝ × ℝ × ℝ)
    (S_ABC S_AOC : ℝ)
    (h1 : vector_sum_property OA OB OC)
    (h2 : S_ABC = 3 * S_AOC) :
  area_ratio S_ABC S_AOC :=
by
  sorry

end triangle_area_ratio_l159_159060


namespace functional_equation_solution_l159_159720

noncomputable def quadratic_polynomial (P : ℝ → ℝ) :=
  ∃ a b c : ℝ, ∀ x : ℝ, P x = a * x^2 + b * x + c

theorem functional_equation_solution (P : ℝ → ℝ) (f : ℝ → ℝ)
  (h_poly : quadratic_polynomial P)
  (h_additive : ∀ x y : ℝ, f (x + y) = f x + f y)
  (h_preserves_poly : ∀ x : ℝ, f (P x) = f x) :
  ∀ x : ℝ, f x = 0 :=
sorry

end functional_equation_solution_l159_159720


namespace not_divisible_by_n_plus_4_l159_159377

theorem not_divisible_by_n_plus_4 (n : ℕ) (h : n > 0) : ¬ ∃ k : ℕ, n^2 + 8 * n + 15 = k * (n + 4) := by
  sorry

end not_divisible_by_n_plus_4_l159_159377


namespace sum_of_digits_div_by_11_in_consecutive_39_l159_159641

-- Define the sum of digits function for natural numbers.
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- The main theorem statement.
theorem sum_of_digits_div_by_11_in_consecutive_39 :
  ∀ (N : ℕ), ∃ k : ℕ, k < 39 ∧ (sum_of_digits (N + k)) % 11 = 0 :=
by sorry

end sum_of_digits_div_by_11_in_consecutive_39_l159_159641


namespace minimum_value_xyz_l159_159767

theorem minimum_value_xyz (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z = 1) : 
  ∃ m : ℝ, m = 16 ∧ ∀ w, w = (x + y) / (x * y * z) → w ≥ m :=
by
  sorry

end minimum_value_xyz_l159_159767


namespace polynomial_identity_l159_159246

theorem polynomial_identity (x : ℝ) : (x + 2) ^ 2 + 2 * (x + 2) * (4 - x) + (4 - x) ^ 2 = 36 := by
  sorry

end polynomial_identity_l159_159246


namespace find_n_l159_159286

theorem find_n :
  let a := (6 + 12 + 18 + 24 + 30 + 36 + 42) / 7
  let b := (2 * n : ℕ)
  (a*a - b*b = 0) -> (n = 12) := 
by 
  let a := 24
  let b := 2*n
  sorry

end find_n_l159_159286


namespace smallest_divisor_of_2880_that_results_in_perfect_square_l159_159213

theorem smallest_divisor_of_2880_that_results_in_perfect_square : 
  ∃ (n : ℕ), (n ∣ 2880) ∧ (∃ m : ℕ, 2880 / n = m * m) ∧ (∀ k : ℕ, (k ∣ 2880) ∧ (∃ m' : ℕ, 2880 / k = m' * m') → n ≤ k) ∧ n = 10 :=
sorry

end smallest_divisor_of_2880_that_results_in_perfect_square_l159_159213


namespace smallest_x_for_gx_eq_1024_l159_159969

noncomputable def g : ℝ → ℝ
  | x => if 2 ≤ x ∧ x ≤ 6 then 2 - |x - 3| else 0

axiom g_property1 : ∀ x : ℝ, 0 < x → g (4 * x) = 4 * g x
axiom g_property2 : ∀ x : ℝ, 2 ≤ x ∧ x ≤ 6 → g x = 2 - |x - 3|
axiom g_2004 : g 2004 = 1024

theorem smallest_x_for_gx_eq_1024 : ∃ x : ℝ, g x = 1024 ∧ ∀ y : ℝ, g y = 1024 → x ≤ y := sorry

end smallest_x_for_gx_eq_1024_l159_159969


namespace math_proof_problem_l159_159001

/-- Given three real numbers a, b, and c such that a ≥ b ≥ 1 ≥ c ≥ 0 and a + b + c = 3.

Part (a): Prove that 2 ≤ ab + bc + ca ≤ 3.
Part (b): Prove that (24 / (a^3 + b^3 + c^3)) + (25 / (ab + bc + ca)) ≥ 14.
--/
theorem math_proof_problem (a b c : ℝ) (h1 : a ≥ b) (h2 : b ≥ 1) (h3 : 1 ≥ c)
  (h4 : c ≥ 0) (h5 : a + b + c = 3) :
  (2 ≤ a * b + b * c + c * a ∧ a * b + b * c + c * a ≤ 3) ∧ 
  (24 / (a^3 + b^3 + c^3) + 25 / (a * b + b * c + c * a) ≥ 14) 
  :=
by
  sorry

end math_proof_problem_l159_159001


namespace smallest_positive_m_l159_159842

theorem smallest_positive_m {m p q : ℤ} (h_eq : 12 * p^2 - m * p - 360 = 0) (h_pq : p * q = -30) :
  (m = 12 * (p + q)) → 0 < m → m = 12 :=
by
  sorry

end smallest_positive_m_l159_159842


namespace workers_task_solution_l159_159501

-- Defining the variables for the number of days worked by A and B
variables (x y : ℕ)

-- Defining the total earnings for A and B
def total_earnings_A := 30
def total_earnings_B := 14

-- Condition: B worked 3 days less than A
def condition1 := y = x - 3

-- Daily wages of A and B
def daily_wage_A := total_earnings_A / x
def daily_wage_B := total_earnings_B / y

-- New scenario conditions
def new_days_A := x - 2
def new_days_B := y + 5

-- New total earnings in the scenario where they work changed days
def new_earnings_A := new_days_A * daily_wage_A
def new_earnings_B := new_days_B * daily_wage_B

-- Final proof to show the number of days worked and daily wages satisfying the conditions
theorem workers_task_solution 
  (h1 : y = x - 3)
  (h2 : new_earnings_A = new_earnings_B) 
  (hx : x = 10)
  (hy : y = 7) 
  (wageA : daily_wage_A = 3) 
  (wageB : daily_wage_B = 2) : 
  x = 10 ∧ y = 7 ∧ daily_wage_A = 3 ∧ daily_wage_B = 2 :=
by {
  sorry  -- Proof is skipped as instructed
}

end workers_task_solution_l159_159501


namespace part1_part2_l159_159061

open Real

noncomputable def f (x a : ℝ) : ℝ := x * exp (a * x) + x * cos x + 1

theorem part1 (x : ℝ) (hx : 0 ≤ x) : cos x ≥ 1 - (1 / 2) * x^2 := 
sorry

theorem part2 (a x : ℝ) (ha : 1 ≤ a) (hx : 0 ≤ x) : f x a ≥ (1 + sin x)^2 := 
sorry

end part1_part2_l159_159061


namespace smallest_integer_is_10_l159_159741

noncomputable def smallest_integer (a b c : ℕ) : ℕ :=
  if h : (a + b + c = 90) ∧ (2 * b = 3 * a) ∧ (5 * a = 2 * c)
  then a
  else 0

theorem smallest_integer_is_10 (a b c : ℕ) (h₁ : a + b + c = 90) (h₂ : 2 * b = 3 * a) (h₃ : 5 * a = 2 * c) : 
  smallest_integer a b c = 10 :=
sorry

end smallest_integer_is_10_l159_159741


namespace factor_between_l159_159488

theorem factor_between (n a b : ℕ) (h1 : 10 < n) 
(h2 : n = a * a + b) 
(h3 : a ∣ n) 
(h4 : b ∣ n) 
(h5 : a ≠ b) 
(h6 : 1 < a) 
(h7 : 1 < b) : 
    ∃ m : ℕ, b = m * a ∧ 1 < m ∧ a < a + m ∧ a + m < b  :=
by
  -- proof to be filled in
  sorry

end factor_between_l159_159488


namespace range_of_a_l159_159068

theorem range_of_a (p q : Set ℝ) (a : ℝ) (h1 : ∀ x, 2 * x^2 - 3 * x + 1 ≤ 0 → x ∈ p) 
                             (h2 : ∀ x, x^2 - (2 * a + 1) * x + a^2 + a ≤ 0 → x ∈ q)
                             (h3 : ∀ x, p x → q x ∧ ∃ x, ¬p x ∧ q x) : 
  0 ≤ a ∧ a ≤ 1 / 2 :=
by
  sorry

end range_of_a_l159_159068


namespace amoeba_population_at_11am_l159_159216

/-- Sarah observes an amoeba colony where initially there are 50 amoebas at 10:00 a.m. The population triples every 10 minutes and there are no deaths among the amoebas. Prove that the number of amoebas at 11:00 a.m. is 36450. -/
theorem amoeba_population_at_11am : 
  let initial_population := 50
  let growth_rate := 3
  let increments := 6  -- since 60 minutes / 10 minutes per increment = 6
  initial_population * (growth_rate ^ increments) = 36450 :=
by
  sorry

end amoeba_population_at_11am_l159_159216


namespace fred_speed_l159_159951

variable {F : ℝ} -- Fred's speed
variable {T : ℝ} -- Time in hours

-- Conditions
def initial_distance : ℝ := 35
def sam_speed : ℝ := 5
def sam_distance : ℝ := 25
def fred_distance := initial_distance - sam_distance

-- Theorem to prove
theorem fred_speed (h1 : T = sam_distance / sam_speed) (h2 : fred_distance = F * T) :
  F = 2 :=
by
  sorry

end fred_speed_l159_159951


namespace triangle_area_proof_l159_159988

noncomputable def cos_fun1 (x : ℝ) : ℝ := 2 * Real.cos (3 * x) + 1
noncomputable def cos_fun2 (x : ℝ) : ℝ := - Real.cos (2 * x)

theorem triangle_area_proof :
  let P := (5 * Real.pi, cos_fun1 (5 * Real.pi))
  let Q := (9 * Real.pi / 2, cos_fun2 (9 * Real.pi / 2))
  let m := (Q.snd - P.snd) / (Q.fst - P.fst)
  let y_intercept := P.snd - m * P.fst
  let y_intercept_point := (0, y_intercept)
  let x_intercept := -y_intercept / m
  let x_intercept_point := (x_intercept, 0)
  let base := x_intercept
  let height := y_intercept
  17 * Real.pi / 4 ≤ P.fst ∧ P.fst ≤ 21 * Real.pi / 4 ∧
  17 * Real.pi / 4 ≤ Q.fst ∧ Q.fst ≤ 21 * Real.pi / 4 ∧
  (P.fst = 5 * Real.pi ∧ Q.fst = 9 * Real.pi / 2) →
  1/2 * base * height = 361 * Real.pi / 8 :=
by
  sorry

end triangle_area_proof_l159_159988


namespace geometric_series_expr_l159_159056

theorem geometric_series_expr :
  4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4 * 
  (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4 * 
  (1 + 4 * (1 + 4)))))))))) + 100 = 5592504 := 
sorry

end geometric_series_expr_l159_159056


namespace problem1_problem2_l159_159152

-- Problem 1: Solution set for x(7 - x) >= 12
theorem problem1 (x : ℝ) : x * (7 - x) ≥ 12 ↔ (3 ≤ x ∧ x ≤ 4) :=
by
  sorry

-- Problem 2: Solution set for x^2 > 2(x - 1)
theorem problem2 (x : ℝ) : x^2 > 2 * (x - 1) ↔ true :=
by
  sorry

end problem1_problem2_l159_159152


namespace trajectory_midpoints_l159_159252

variables (a b c x y : ℝ)

def arithmetic_sequence (a b c : ℝ) : Prop := c = 2 * b - a

def line_eq (b a c x y : ℝ) : Prop := b * x + a * y + c = 0

def parabola_eq (x y : ℝ) : Prop := y^2 = -0.5 * x

theorem trajectory_midpoints
  (hac : arithmetic_sequence a b c)
  (line_cond : line_eq b a c x y)
  (parabola_cond : parabola_eq x y) :
  (x + 1 = -(2 * y - 1)^2) ∧ (y ≠ 1) :=
sorry

end trajectory_midpoints_l159_159252


namespace find_point_B_l159_159906

structure Point where
  x : Int
  y : Int

def translation (p : Point) (dx dy : Int) : Point :=
  { x := p.x + dx, y := p.y + dy }

theorem find_point_B :
  let A := Point.mk (-2) 3
  let A' := Point.mk 3 2
  let B' := Point.mk 4 0
  let dx := 5
  let dy := -1
  (translation A dx dy = A') →
  ∃ B : Point, translation B dx dy = B' ∧ B = Point.mk (-1) (-1) :=
by
  intros
  use Point.mk (-1) (-1)
  constructor
  sorry
  rfl

end find_point_B_l159_159906


namespace find_a1_l159_159358

theorem find_a1 (f : ℝ → ℝ) (a : ℕ → ℝ) (h₀ : ∀ x, f x = (x - 1)^3 + x + 2)
(h₁ : ∀ n, a (n + 1) = a n + 1/2)
(h₂ : f (a 1) + f (a 2) + f (a 3) + f (a 4) + f (a 5) + f (a 6) = 18) :
a 1 = -1 / 4 :=
by
  sorry

end find_a1_l159_159358


namespace crumble_topping_correct_amount_l159_159860

noncomputable def crumble_topping_total_mass (flour butter sugar : ℕ) (factor : ℚ) : ℚ :=
  factor * (flour + butter + sugar) / 1000  -- convert grams to kilograms

theorem crumble_topping_correct_amount {flour butter sugar : ℕ} (factor : ℚ) (h_flour : flour = 100) (h_butter : butter = 50) (h_sugar : sugar = 50) (h_factor : factor = 2.5) :
  crumble_topping_total_mass flour butter sugar factor = 0.5 :=
by
  sorry

end crumble_topping_correct_amount_l159_159860


namespace value_of_M_l159_159921

-- Define M as given in the conditions
def M : ℤ :=
  (150^2 + 2) + (149^2 - 2) - (148^2 + 2) - (147^2 - 2) + (146^2 + 2) +
  (145^2 - 2) - (144^2 + 2) - (143^2 - 2) + (142^2 + 2) + (141^2 - 2) -
  (140^2 + 2) - (139^2 - 2) + (138^2 + 2) + (137^2 - 2) - (136^2 + 2) -
  (135^2 - 2) + (134^2 + 2) + (133^2 - 2) - (132^2 + 2) - (131^2 - 2) +
  (130^2 + 2) + (129^2 - 2) - (128^2 + 2) - (127^2 - 2) + (126^2 + 2) +
  (125^2 - 2) - (124^2 + 2) - (123^2 - 2) + (122^2 + 2) + (121^2 - 2) -
  (120^2 + 2) - (119^2 - 2) + (118^2 + 2) + (117^2 - 2) - (116^2 + 2) -
  (115^2 - 2) + (114^2 + 2) + (113^2 - 2) - (112^2 + 2) - (111^2 - 2) +
  (110^2 + 2) + (109^2 - 2) - (108^2 + 2) - (107^2 - 2) + (106^2 + 2) +
  (105^2 - 2) - (104^2 + 2) - (103^2 - 2) + (102^2 + 2) + (101^2 - 2) -
  (100^2 + 2) - (99^2 - 2) + (98^2 + 2) + (97^2 - 2) - (96^2 + 2) -
  (95^2 - 2) + (94^2 + 2) + (93^2 - 2) - (92^2 + 2) - (91^2 - 2) +
  (90^2 + 2) + (89^2 - 2) - (88^2 + 2) - (87^2 - 2) + (86^2 + 2) +
  (85^2 - 2) - (84^2 + 2) - (83^2 - 2) + (82^2 + 2) + (81^2 - 2) -
  (80^2 + 2) - (79^2 - 2) + (78^2 + 2) + (77^2 - 2) - (76^2 + 2) -
  (75^2 - 2) + (74^2 + 2) + (73^2 - 2) - (72^2 + 2) - (71^2 - 2) +
  (70^2 + 2) + (69^2 - 2) - (68^2 + 2) - (67^2 - 2) + (66^2 + 2) +
  (65^2 - 2) - (64^2 + 2) - (63^2 - 2) + (62^2 + 2) + (61^2 - 2) -
  (60^2 + 2) - (59^2 - 2) + (58^2 + 2) + (57^2 - 2) - (56^2 + 2) -
  (55^2 - 2) + (54^2 + 2) + (53^2 - 2) - (52^2 + 2) - (51^2 - 2) +
  (50^2 + 2) + (49^2 - 2) - (48^2 + 2) - (47^2 - 2) + (46^2 + 2) +
  (45^2 - 2) - (44^2 + 2) - (43^2 - 2) + (42^2 + 2) + (41^2 - 2) -
  (40^2 + 2) - (39^2 - 2) + (38^2 + 2) + (37^2 - 2) - (36^2 + 2) -
  (35^2 - 2) + (34^2 + 2) + (33^2 - 2) - (32^2 + 2) - (31^2 - 2) +
  (30^2 + 2) + (29^2 - 2) - (28^2 + 2) - (27^2 - 2) + (26^2 + 2) +
  (25^2 - 2) - (24^2 + 2) - (23^2 - 2) + (22^2 + 2) + (21^2 - 2) -
  (20^2 + 2) - (19^2 - 2) + (18^2 + 2) + (17^2 - 2) - (16^2 + 2) -
  (15^2 - 2) + (14^2 + 2) + (13^2 - 2) - (12^2 + 2) - (11^2 - 2) +
  (10^2 + 2) + (9^2 - 2) - (8^2 + 2) - (7^2 - 2) + (6^2 + 2) +
  (5^2 - 2) - (4^2 + 2) - (3^2 - 2) + (2^2 + 2) + (1^2 - 2)

-- Statement to prove that the value of M is 22700
theorem value_of_M : M = 22700 :=
  by sorry

end value_of_M_l159_159921


namespace angle_in_third_quadrant_l159_159006

theorem angle_in_third_quadrant (α : ℝ) (h1 : Real.sin α < 0) (h2 : Real.cos α < 0) : 
  ∃ k : ℤ, α = (2 * k + 1) * Real.pi + β ∧ β ∈ Set.Ioo (0 : ℝ) Real.pi :=
by
  sorry

end angle_in_third_quadrant_l159_159006


namespace ratio_yx_l159_159020

variable (c x y : ℝ)

theorem ratio_yx (h1: x = 0.80 * c) (h2: y = 1.25 * c) : y / x = 25 / 16 := by
  -- Proof to be written here
  sorry

end ratio_yx_l159_159020


namespace thirty_k_divisor_of_929260_l159_159496

theorem thirty_k_divisor_of_929260 (k : ℕ) (h1: 30^k ∣ 929260):
(3^k - k^3 = 2) :=
sorry

end thirty_k_divisor_of_929260_l159_159496


namespace part_a_part_b_l159_159845

-- Part (a)

theorem part_a : ∃ (a b : ℕ), 2015^2 + 2017^2 = 2 * (a^2 + b^2) :=
by
  -- The proof will go here
  sorry

-- Part (b)

theorem part_b (k n : ℕ) : ∃ (a b : ℕ), (2 * k + 1)^2 + (2 * n + 1)^2 = 2 * (a^2 + b^2) :=
by
  -- The proof will go here
  sorry

end part_a_part_b_l159_159845


namespace value_of_x_plus_y_l159_159259

theorem value_of_x_plus_y (x y : ℝ) (h1 : 1/x + 1/y = 4) (h2 : 1/x - 1/y = 2) : x + y = 4/3 :=
sorry

end value_of_x_plus_y_l159_159259


namespace consecutive_odd_integers_l159_159665

theorem consecutive_odd_integers (x : ℤ) (h : x + 4 = 15) : 3 * x - 2 * (x + 4) = 3 :=
by
  sorry

end consecutive_odd_integers_l159_159665


namespace find_A_l159_159308

variable {A B C : ℚ}

theorem find_A (h1 : A = 1/2 * B) (h2 : B = 3/4 * C) (h3 : A + C = 55) : A = 15 :=
by
  sorry

end find_A_l159_159308


namespace negation_of_forall_inequality_l159_159147

theorem negation_of_forall_inequality:
  ¬(∀ x : ℝ, x > 0 → x^2 - x ≤ 0) ↔ ∃ x : ℝ, x > 0 ∧ x^2 - x > 0 :=
by
  sorry

end negation_of_forall_inequality_l159_159147


namespace correct_statements_l159_159864

noncomputable def f : ℝ → ℝ := sorry

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def is_monotonically_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≤ f y

def satisfies_condition (f : ℝ → ℝ) : Prop := 
  ∀ x, f (1 - x) + f (1 + x) = 0

def is_symmetric_about_line (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (a - x) = f (a + x)

theorem correct_statements (f : ℝ → ℝ) :
  is_even f →
  is_monotonically_increasing f (-1) 0 →
  satisfies_condition f →
  (f (-3) = 0 ∧
   is_monotonically_increasing f 1 2 ∧
   is_symmetric_about_line f 1) :=
by
  intros h_even h_mono h_cond
  sorry

end correct_statements_l159_159864


namespace laura_park_time_percentage_l159_159141

theorem laura_park_time_percentage (num_trips: ℕ) (time_in_park: ℝ) (walking_time: ℝ) 
    (total_percentage_in_park: ℝ) 
    (h1: num_trips = 6) 
    (h2: time_in_park = 2) 
    (h3: walking_time = 0.5) 
    (h4: total_percentage_in_park = 80) : 
    (time_in_park * num_trips) / ((time_in_park + walking_time) * num_trips) * 100 = total_percentage_in_park :=
by
  sorry

end laura_park_time_percentage_l159_159141


namespace arithmetic_sequence_max_sum_proof_l159_159075

noncomputable def arithmetic_sequence_max_sum (a_1 d : ℝ) (n : ℕ) : ℝ :=
  n * a_1 + (n * (n - 1)) / 2 * d

theorem arithmetic_sequence_max_sum_proof (a_1 d : ℝ) 
  (h1 : 3 * a_1 + 6 * d = 9)
  (h2 : a_1 + 5 * d = -9) :
  ∃ n : ℕ, n = 3 ∧ arithmetic_sequence_max_sum a_1 d n = 21 :=
by
  sorry

end arithmetic_sequence_max_sum_proof_l159_159075


namespace contestant_wins_quiz_l159_159581

noncomputable def winProbability : ℚ :=
  let p_correct := (1 : ℚ) / 3
  let p_wrong := (2 : ℚ) / 3
  let binom := Nat.choose  -- binomial coefficient function
  ((binom 4 2 * (p_correct ^ 2) * (p_wrong ^ 2)) +
   (binom 4 3 * (p_correct ^ 3) * (p_wrong ^ 1)) +
   (binom 4 4 * (p_correct ^ 4) * (p_wrong ^ 0)))

theorem contestant_wins_quiz :
  winProbability = 11 / 27 :=
by
  simp [winProbability, Nat.choose]
  norm_num
  done

end contestant_wins_quiz_l159_159581


namespace sqrt_expression_l159_159523

theorem sqrt_expression : Real.sqrt ((4^2) * (5^6)) = 500 := by
  sorry

end sqrt_expression_l159_159523


namespace smallest_altitude_leq_three_l159_159553

theorem smallest_altitude_leq_three (a b c : ℝ) (r : ℝ) 
  (ha : a = max a (max b c)) 
  (r_eq : r = 1) 
  (area_eq : ∀ (S : ℝ), S = (a + b + c) / 2 ∧ S = a * h / 2) :
  ∃ h : ℝ, h ≤ 3 :=
by
  sorry

end smallest_altitude_leq_three_l159_159553


namespace complement_union_l159_159920

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define the set M
def M : Set ℕ := {1, 2}

-- Define the set N
def N : Set ℕ := {3, 4}

-- State the theorem to prove the complement of the union of M and N in U is {5}
theorem complement_union {U M N : Set ℕ} (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2}) (hN : N = {3, 4}) :
  U \ (M ∪ N) = {5} :=
by
  sorry

end complement_union_l159_159920


namespace vectors_opposite_directions_l159_159477

variable {V : Type*} [AddCommGroup V]

theorem vectors_opposite_directions (a b : V) (h : a + 4 • b = 0) (ha : a ≠ 0) (hb : b ≠ 0) : a = -4 • b :=
by sorry

end vectors_opposite_directions_l159_159477


namespace cauchy_problem_solution_l159_159219

noncomputable def solution (y : ℝ → ℝ) (x : ℝ) : Prop :=
  y x = (x^2) / 2 + (x^3) / 6 + (x^4) / 12 + (x^5) / 20 + x + 1

theorem cauchy_problem_solution (y : ℝ → ℝ) (x : ℝ) 
  (h1: ∀ x, (deriv^[2] y) x = 1 + x + x^2 + x^3)
  (h2: y 0 = 1)
  (h3: deriv y 0 = 1) : 
  solution y x := 
by
  -- Proof Steps
  sorry

end cauchy_problem_solution_l159_159219


namespace john_thrice_tom_years_ago_l159_159178

-- Define the ages of Tom and John
def T : ℕ := 16
def J : ℕ := 36

-- Condition that John will be 2 times Tom's age in 4 years
def john_twice_tom_in_4_years (J T : ℕ) : Prop := J + 4 = 2 * (T + 4)

-- The number of years ago John was thrice as old as Tom
def years_ago (J T x : ℕ) : Prop := J - x = 3 * (T - x)

-- Prove that the number of years ago John was thrice as old as Tom is 6
theorem john_thrice_tom_years_ago (h1 : john_twice_tom_in_4_years 36 16) : years_ago 36 16 6 :=
by
  -- Import initial values into the context
  unfold john_twice_tom_in_4_years at h1
  unfold years_ago
  -- Solve the steps, more details in the actual solution
  sorry

end john_thrice_tom_years_ago_l159_159178


namespace brenda_has_8_dollars_l159_159532

-- Define the amounts of money each friend has
def emma_money : ℕ := 8
def daya_money : ℕ := emma_money + (25 * emma_money / 100) -- 25% more than Emma's money
def jeff_money : ℕ := 2 * daya_money / 5 -- Jeff has 2/5 of Daya's money
def brenda_money : ℕ := jeff_money + 4 -- Brenda has 4 more dollars than Jeff

-- The theorem stating the final question
theorem brenda_has_8_dollars : brenda_money = 8 :=
by
  sorry

end brenda_has_8_dollars_l159_159532


namespace rectangles_in_grid_at_least_three_cells_l159_159503

theorem rectangles_in_grid_at_least_three_cells :
  let number_of_rectangles (n : ℕ) := (n + 1).choose 2 * (n + 1).choose 2
  let single_cell_rectangles (n : ℕ) := n * n
  let one_by_two_or_two_by_one_rectangles (n : ℕ) := n * (n - 1) * 2
  let total_rectangles (n : ℕ) := number_of_rectangles n - (single_cell_rectangles n + one_by_two_or_two_by_one_rectangles n)
  total_rectangles 6 = 345 :=
by
  sorry

end rectangles_in_grid_at_least_three_cells_l159_159503


namespace power_division_identity_l159_159217

theorem power_division_identity : 
  ∀ (a b c : ℕ), a = 3 → b = 12 → c = 2 → (3 ^ 12 / (3 ^ 2) ^ 2 = 6561) :=
by
  intros a b c h1 h2 h3
  sorry

end power_division_identity_l159_159217


namespace y_in_interval_l159_159633

theorem y_in_interval :
  ∃ (y : ℝ), y = 5 + (1/y) * -y ∧ 2 < y ∧ y ≤ 4 :=
by
  sorry

end y_in_interval_l159_159633


namespace unique_identity_element_l159_159898

variable {G : Type*} [Group G]

theorem unique_identity_element (e e' : G) (h1 : ∀ g : G, e * g = g ∧ g * e = g) (h2 : ∀ g : G, e' * g = g ∧ g * e' = g) : e = e' :=
by 
sorry

end unique_identity_element_l159_159898


namespace average_age_of_9_l159_159563

theorem average_age_of_9 : 
  ∀ (avg_20 avg_5 age_15 : ℝ),
  avg_20 = 15 →
  avg_5 = 14 →
  age_15 = 86 →
  (9 * (69/9)) = 7.67 :=
by
  intros avg_20 avg_5 age_15 avg_20_val avg_5_val age_15_val
  -- The proof is skipped
  sorry

end average_age_of_9_l159_159563


namespace scientific_notation_of_0_00065_l159_159749

/-- 
Prove that the decimal representation of a number 0.00065 can be expressed in scientific notation 
as 6.5 * 10^(-4)
-/
theorem scientific_notation_of_0_00065 : 0.00065 = 6.5 * 10^(-4) := 
by 
  sorry

end scientific_notation_of_0_00065_l159_159749


namespace cubic_expression_l159_159651

theorem cubic_expression (a b c : ℝ) (h1 : a + b + c = 13) (h2 : ab + ac + bc = 30) : a^3 + b^3 + c^3 - 3 * abc = 1027 :=
sorry

end cubic_expression_l159_159651


namespace sequence_general_term_l159_159918

open Nat

def sequence_a (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧
  a 2 = 3 ∧
  (∀ n : ℕ, 0 < n → a (n + 2) ≤ a n + 3 * 2^n) ∧
  (∀ n : ℕ, 0 < n → a (n + 1) ≥ 2 * a n + 1)

theorem sequence_general_term (a : ℕ → ℕ) (h : sequence_a a) :
  ∀ n : ℕ, 0 < n → a n = 2^n - 1 :=
by
  sorry

end sequence_general_term_l159_159918


namespace student_correct_answers_l159_159419

theorem student_correct_answers 
  (c w : ℕ) 
  (h1 : c + w = 60) 
  (h2 : 4 * c - w = 130) : 
  c = 38 :=
by
  sorry

end student_correct_answers_l159_159419


namespace problem_l159_159251

variable {m n r t : ℚ}

theorem problem (h1 : m / n = 5 / 4) (h2 : r / t = 8 / 15) : (3 * m * r - n * t) / (4 * n * t - 7 * m * r) = -3 / 2 :=
by
  sorry

end problem_l159_159251


namespace probability_of_X_eq_2_l159_159900

-- Define the random variable distribution condition
def random_variable_distribution (a : ℝ) (P : ℝ → ℝ) : Prop :=
  P 1 = 1 / (2 * a) ∧ P 2 = 2 / (2 * a) ∧ P 3 = 3 / (2 * a) ∧
  (1 / (2 * a) + 2 / (2 * a) + 3 / (2 * a) = 1)

-- State the theorem given the conditions and the result
theorem probability_of_X_eq_2 (a : ℝ) (P : ℝ → ℝ) (h : random_variable_distribution a P) : 
  P 2 = 1 / 3 :=
sorry

end probability_of_X_eq_2_l159_159900


namespace unique_solution_p_l159_159324

theorem unique_solution_p (p : ℚ) :
  (∀ x : ℝ, (2 * x + 3) / (p * x - 2) = x) ↔ p = -4 / 3 := sorry

end unique_solution_p_l159_159324


namespace incorrect_inequality_l159_159149

theorem incorrect_inequality (a b : ℝ) (h1 : a < 0) (h2 : 0 < b) : ¬ (a^2 < a * b) :=
by
  sorry

end incorrect_inequality_l159_159149


namespace fractions_product_simplified_l159_159222

theorem fractions_product_simplified : (2/3 : ℚ) * (4/7) * (9/11) = 24/77 := by
  sorry

end fractions_product_simplified_l159_159222


namespace problem_arithmetic_sequence_l159_159037

-- Definitions based on given conditions
def a1 : ℕ := 2
def d := (13 - 2 * a1) / 3

-- Definition of the nth term in the arithmetic sequence
def a (n : ℕ) : ℕ := a1 + (n - 1) * d

-- The required proof problem statement
theorem problem_arithmetic_sequence : a 4 + a 5 + a 6 = 42 := 
by
  -- placeholders for the actual proof
  sorry

end problem_arithmetic_sequence_l159_159037


namespace max_min_ab_bc_ca_l159_159664

theorem max_min_ab_bc_ca (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h_sum : a + b + c = 12) (h_ab_bc_ca : a * b + b * c + c * a = 30) :
  max (min (a * b) (min (b * c) (c * a))) = 9 :=
sorry

end max_min_ab_bc_ca_l159_159664


namespace NaOH_combined_l159_159509

theorem NaOH_combined (n : ℕ) (h : n = 54) : 
  (2 * n) / 2 = 54 :=
by
  sorry

end NaOH_combined_l159_159509


namespace problem_statement_l159_159245

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := λ x => f (x - 1)

theorem problem_statement :
  (∀ x : ℝ, f (-x) = f x) →  -- Condition: f is an even function.
  (∀ x : ℝ, g (-x) = -g x) → -- Condition: g is an odd function.
  (g 1 = 3) →                -- Condition: g passes through (1,3).
  (f 2012 + g 2013 = 6) :=   -- Statement to prove.
by
  sorry

end problem_statement_l159_159245


namespace correct_propositions_l159_159671

-- Definitions for the propositions
def prop1 (a M b : Prop) : Prop := (a ∧ M) ∧ (b ∧ M) → a ∧ b
def prop2 (a M b : Prop) : Prop := (a ∧ M) ∧ (b ∧ ¬M) → a ∧ ¬b
def prop3 (a b M : Prop) : Prop := (a ∧ b) ∧ (b ∧ M) → a ∧ M
def prop4 (a M N : Prop) : Prop := (a ∧ ¬M) ∧ (a ∧ N) → ¬M ∧ N

-- Proof problem statement
theorem correct_propositions : 
  ∀ (a b M N : Prop), 
    (prop1 a M b = true) ∨ (prop1 a M b = false) ∧ 
    (prop2 a M b = true) ∨ (prop2 a M b = false) ∧ 
    (prop3 a b M = true) ∨ (prop3 a b M = false) ∧ 
    (prop4 a M N = true) ∨ (prop4 a M N = false) → 
    3 = 3 :=
by
  sorry

end correct_propositions_l159_159671


namespace number_of_bananas_in_bowl_l159_159267

theorem number_of_bananas_in_bowl (A P B : Nat) (h1 : P = A + 2) (h2 : B = P + 3) (h3 : A + P + B = 19) : B = 9 :=
sorry

end number_of_bananas_in_bowl_l159_159267


namespace art_group_students_count_l159_159942

theorem art_group_students_count (x : ℕ) (h1 : x * (1 / 60) + 2 * (x + 15) * (1 / 60) = 1) : x = 10 :=
by {
  sorry
}

end art_group_students_count_l159_159942


namespace range_of_a_analytical_expression_l159_159902

variables {f : ℝ → ℝ}

-- Problem 1
theorem range_of_a (h_odd : ∀ x, f (-x) = -f x)
  (h_mono : ∀ x y, x < y → f x ≥ f y)
  {a : ℝ} (h_ineq : f (1 - a) + f (1 - 2 * a) < 0) :
  0 < a ∧ a ≤ 2 / 3 :=
sorry

-- Problem 2
theorem analytical_expression 
  (h_odd : ∀ x, f (-x) = -f x)
  (h_def : ∀ x, 0 < x ∧ x < 1 → f x = x^2 + x + 1)
  (h_zero : f 0 = 0) :
  ∀ x : ℝ, -1 < x ∧ x < 1 → f x = 
    if x > 0 then x^2 + x + 1
    else if x = 0 then 0
    else -x^2 + x - 1 :=
sorry

end range_of_a_analytical_expression_l159_159902


namespace solve_for_x_l159_159967

theorem solve_for_x (x : ℝ) (hx : x ≠ 0) : x^3 - 2 * x^2 = 0 ↔ x = 2 :=
by sorry

end solve_for_x_l159_159967


namespace smallest_positive_value_l159_159964

theorem smallest_positive_value (x : ℝ) (hx : x > 0) (h : x / 7 + 2 / (7 * x) = 1) : 
  x = (7 - Real.sqrt 41) / 2 :=
sorry

end smallest_positive_value_l159_159964


namespace find_cylinder_radius_l159_159498

-- Define the problem conditions
def cone_diameter := 10
def cone_altitude := 12
def cylinder_height_eq_diameter (r: ℚ) := 2 * r

-- Define the cone and cylinder inscribed properties
noncomputable def inscribed_cylinder_radius (r : ℚ) : Prop :=
  (cylinder_height_eq_diameter r) ≤ cone_altitude ∧
  2 * r ≤ cone_diameter ∧
  cone_altitude - cylinder_height_eq_diameter r = (cone_altitude * r) / (cone_diameter / 2)

-- The proof goal
theorem find_cylinder_radius : ∃ r : ℚ, inscribed_cylinder_radius r ∧ r = 30/11 :=
by
  sorry

end find_cylinder_radius_l159_159498


namespace sin_five_pi_over_six_l159_159200

theorem sin_five_pi_over_six : Real.sin (5 * Real.pi / 6) = 1 / 2 := 
  sorry

end sin_five_pi_over_six_l159_159200


namespace arctan_sum_pi_div_two_l159_159125

noncomputable def arctan_sum : Real :=
  Real.arctan (3 / 4) + Real.arctan (4 / 3)

theorem arctan_sum_pi_div_two : arctan_sum = Real.pi / 2 := 
by sorry

end arctan_sum_pi_div_two_l159_159125


namespace solve_a_solve_inequality_solution_set_l159_159929

theorem solve_a (a : ℝ) :
  (∀ x : ℝ, (1 / 2 < x ∧ x < 2) ↔ ax^2 + 5 * x - 2 > 0) →
  a = -2 :=
by
  sorry

theorem solve_inequality_solution_set (x : ℝ) :
  (a = -2) →
  (2 * x^2 + 5 * x - 3 < 0) ↔
  (-3 < x ∧ x < 1 / 2) :=
by
  sorry

end solve_a_solve_inequality_solution_set_l159_159929


namespace ducks_at_Lake_Michigan_l159_159277

variable (D : ℕ)

def ducks_condition := 2 * D + 6 = 206

theorem ducks_at_Lake_Michigan (h : ducks_condition D) : D = 100 :=
by
  sorry

end ducks_at_Lake_Michigan_l159_159277


namespace find_number_of_folders_l159_159137

theorem find_number_of_folders :
  let price_pen := 1
  let price_notebook := 3
  let price_folder := 5
  let pens_bought := 3
  let notebooks_bought := 4
  let bill := 50
  let change := 25
  let total_cost_pens_notebooks := pens_bought * price_pen + notebooks_bought * price_notebook
  let amount_spent := bill - change
  let amount_spent_on_folders := amount_spent - total_cost_pens_notebooks
  let number_of_folders := amount_spent_on_folders / price_folder
  number_of_folders = 2 :=
by
  sorry

end find_number_of_folders_l159_159137


namespace intersection_length_l159_159169

theorem intersection_length 
  (A B : ℝ × ℝ) 
  (hA : A.1^2 + A.2^2 = 1) 
  (hB : B.1^2 + B.2^2 = 1) 
  (hA_on_line : A.1 = A.2) 
  (hB_on_line : B.1 = B.2) 
  (hAB : A ≠ B) :
  dist A B = 2 :=
by sorry

end intersection_length_l159_159169


namespace father_son_speed_ratio_l159_159699

theorem father_son_speed_ratio
  (F S t : ℝ)
  (distance_hallway : ℝ)
  (distance_meet_from_father : ℝ)
  (H1 : distance_hallway = 16)
  (H2 : distance_meet_from_father = 12)
  (H3 : 12 = F * t)
  (H4 : 4 = S * t)
  : F / S = 3 := by
  sorry

end father_son_speed_ratio_l159_159699


namespace multiple_of_every_positive_integer_is_zero_l159_159024

theorem multiple_of_every_positive_integer_is_zero :
  ∀ (n : ℤ), (∀ (m : ℕ), ∃ (k : ℤ), n = k * (m : ℤ)) → n = 0 := 
by
  sorry

end multiple_of_every_positive_integer_is_zero_l159_159024


namespace parabola_equation_l159_159225

theorem parabola_equation (p : ℝ) (hp : 0 < p)
  (F : ℝ × ℝ) (hF : F = (p / 2, 0))
  (A B : ℝ × ℝ)
  (hA : A = (x1, y1)) (hB : B = (x2, y2))
  (h_intersect : y1^2 = 2*p*x1 ∧ y2^2 = 2*p*x2)
  (M : ℝ × ℝ) (hM : M = ((x1 + x2) / 2, (y1 + y2) / 2))
  (hM_coords : M = (3, 2)) :
  p = 2 ∨ p = 4 :=
sorry

end parabola_equation_l159_159225


namespace triangle_obtuse_of_cos_relation_l159_159351

theorem triangle_obtuse_of_cos_relation
  (a b c : ℝ)
  (A B C : ℝ)
  (hTriangle : A + B + C = Real.pi)
  (hSides : a^2 = b^2 + c^2 - 2*b*c*Real.cos A)
  (hSides' : b^2 = a^2 + c^2 - 2*a*c*Real.cos B)
  (hSides'' : c^2 = a^2 + b^2 - 2*a*b*Real.cos C)
  (hRelation : a * Real.cos C = b + 2/3 * c) :
 ∃ (A' : ℝ), A' = A ∧ A > (Real.pi / 2) := 
sorry

end triangle_obtuse_of_cos_relation_l159_159351


namespace line_not_in_first_quadrant_l159_159615

theorem line_not_in_first_quadrant (t : ℝ) : 
  (∀ x y : ℝ, ¬ ((0 < x ∧ 0 < y) ∧ (2 * t - 3) * x + y + 6 = 0)) ↔ t ≥ 3 / 2 :=
by
  sorry

end line_not_in_first_quadrant_l159_159615


namespace find_c_l159_159342

-- Definitions based on the conditions in the problem
def is_vertex (h k : ℝ) := (5, 1) = (h, k)
def passes_through (x y : ℝ) := (2, 3) = (x, y)

-- Lean theorem statement
theorem find_c (a b c : ℝ) (h k x y : ℝ) (hv : is_vertex h k) (hp : passes_through x y)
  (heq : ∀ y, x = a * y^2 + b * y + c) : c = 17 / 4 :=
by
  sorry

end find_c_l159_159342


namespace max_weight_American_l159_159707

noncomputable def max_weight_of_American_swallow (A E : ℕ) : Prop :=
A = 5 ∧ 2 * E + E = 90 ∧ 60 * A + 60 * 2 * A = 600

theorem max_weight_American (A E : ℕ) : max_weight_of_American_swallow A E :=
by
  sorry

end max_weight_American_l159_159707


namespace F_8_not_true_F_6_might_be_true_l159_159521

variable {n : ℕ}

-- Declare the proposition F
variable (F : ℕ → Prop)

-- Placeholder conditions
axiom condition1 : ¬ F 7
axiom condition2 : ∀ k : ℕ, k > 0 → (F k → F (k + 1))

-- Proof statements
theorem F_8_not_true : ¬ F 8 :=
by {
  sorry
}

theorem F_6_might_be_true : ¬ ¬ F 6 :=
by {
  sorry
}

end F_8_not_true_F_6_might_be_true_l159_159521


namespace smallest_five_digit_divisible_by_15_32_54_l159_159450

theorem smallest_five_digit_divisible_by_15_32_54 : 
  ∃ n : ℤ, n >= 10000 ∧ n < 100000 ∧ (15 ∣ n) ∧ (32 ∣ n) ∧ (54 ∣ n) ∧ n = 17280 :=
  sorry

end smallest_five_digit_divisible_by_15_32_54_l159_159450


namespace new_light_wattage_l159_159389

theorem new_light_wattage (w_old : ℕ) (p : ℕ) (w_new : ℕ) (h1 : w_old = 110) (h2 : p = 30) (h3 : w_new = w_old + (p * w_old / 100)) : w_new = 143 :=
by
  -- Using the conditions provided
  rw [h1, h2] at h3
  norm_num at h3
  exact h3

end new_light_wattage_l159_159389


namespace total_outlets_needed_l159_159195

-- Definitions based on conditions:
def outlets_per_room : ℕ := 6
def number_of_rooms : ℕ := 7

-- Theorem to prove the total number of outlets is 42
theorem total_outlets_needed : outlets_per_room * number_of_rooms = 42 := by
  -- Simple proof with mathematics:
  sorry

end total_outlets_needed_l159_159195


namespace square_side_increase_l159_159101

theorem square_side_increase (s : ℝ) :
  let new_side := 1.5 * s
  let new_area := new_side^2
  let original_area := s^2
  let new_perimeter := 4 * new_side
  let original_perimeter := 4 * s
  let new_diagonal := new_side * Real.sqrt 2
  let original_diagonal := s * Real.sqrt 2
  (new_area - original_area) / original_area * 100 = 125 ∧
  (new_perimeter - original_perimeter) / original_perimeter * 100 = 50 ∧
  (new_diagonal - original_diagonal) / original_diagonal * 100 = 50 :=
by
  sorry

end square_side_increase_l159_159101


namespace sqrt_abc_sum_eq_54_sqrt_5_l159_159861

theorem sqrt_abc_sum_eq_54_sqrt_5 
  (a b c : ℝ) 
  (h1 : b + c = 17) 
  (h2 : c + a = 18) 
  (h3 : a + b = 19) : 
  Real.sqrt (a * b * c * (a + b + c)) = 54 * Real.sqrt 5 := 
by 
  sorry

end sqrt_abc_sum_eq_54_sqrt_5_l159_159861


namespace triangle_perimeter_l159_159552

-- Define the ratios
def ratio1 : ℚ := 1 / 2
def ratio2 : ℚ := 1 / 3
def ratio3 : ℚ := 1 / 4

-- Define the longest side
def longest_side : ℚ := 48

-- Compute the perimeter given the conditions
theorem triangle_perimeter (ratio1 ratio2 ratio3 : ℚ) (longest_side : ℚ) 
  (h_ratio1 : ratio1 = 1 / 2) (h_ratio2 : ratio2 = 1 / 3) (h_ratio3 : ratio3 = 1 / 4)
  (h_longest_side : longest_side = 48) : 
  (longest_side * 6/ (ratio1 * 12 + ratio2 * 12 + ratio3 * 12)) = 104 := by
  sorry

end triangle_perimeter_l159_159552


namespace geometric_progression_l159_159331

theorem geometric_progression (p : ℝ) 
  (a b c : ℝ)
  (h1 : a = p - 2)
  (h2 : b = 2 * Real.sqrt p)
  (h3 : c = -3 - p)
  (h4 : b ^ 2 = a * c) : 
  p = 1 := 
by 
  sorry

end geometric_progression_l159_159331


namespace unique_solution_l159_159247

-- Definitions of the problem
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ (m : ℕ), m ∣ n → m = 1 ∨ m = n

def satisfies_conditions (p q r : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ is_prime r ∧ is_prime (4 * q - 1) ∧ (p + q) * (r - p) = p + r

theorem unique_solution (p q r : ℕ) (h : satisfies_conditions p q r) : (p, q, r) = (2, 3, 3) :=
  sorry

end unique_solution_l159_159247


namespace compare_polynomials_l159_159177

theorem compare_polynomials (x : ℝ) (h : x ≥ 0) : 
  (x > 2 → 5*x^2 - 1 > 3*x^2 + 3*x + 1) ∧ 
  (x = 2 → 5*x^2 - 1 = 3*x^2 + 3*x + 1) ∧ 
  (0 ≤ x → x < 2 → 5*x^2 - 1 < 3*x^2 + 3*x + 1) :=
sorry

end compare_polynomials_l159_159177


namespace mark_egg_supply_in_a_week_l159_159550

def dozen := 12
def eggs_per_day_store1 := 5 * dozen
def eggs_per_day_store2 := 30
def daily_eggs_supplied := eggs_per_day_store1 + eggs_per_day_store2
def days_per_week := 7

theorem mark_egg_supply_in_a_week : daily_eggs_supplied * days_per_week = 630 := by
  sorry

end mark_egg_supply_in_a_week_l159_159550


namespace train_cross_bridge_time_l159_159915

open Nat

-- Defining conditions as per the problem
def train_length : ℕ := 200
def bridge_length : ℕ := 150
def speed_kmph : ℕ := 36
def speed_mps : ℕ := speed_kmph * 5 / 18
def total_distance : ℕ := train_length + bridge_length
def time_to_cross : ℕ := total_distance / speed_mps

-- Stating the theorem
theorem train_cross_bridge_time : time_to_cross = 35 := by
  sorry

end train_cross_bridge_time_l159_159915


namespace parabola_y_intercepts_zero_l159_159426

-- Define the quadratic equation
def quadratic (a b c y: ℝ) : ℝ := a * y^2 + b * y + c

-- Define the discriminant
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Condition: equation of the parabola and discriminant calculation
def parabola_equation : Prop := 
  let a := 3
  let b := -4
  let c := 5
  discriminant a b c < 0

-- Statement to prove
theorem parabola_y_intercepts_zero : 
  (parabola_equation) → (∀ y : ℝ, quadratic 3 (-4) 5 y ≠ 0) :=
by
  intro h
  sorry

end parabola_y_intercepts_zero_l159_159426


namespace time_to_cross_pole_correct_l159_159790

noncomputable def speed_kmph : ℝ := 160 -- Speed of the train in kmph
noncomputable def length_meters : ℝ := 800.064 -- Length of the train in meters

noncomputable def conversion_factor : ℝ := 1000 / 3600 -- Conversion factor from kmph to m/s
noncomputable def speed_mps : ℝ := speed_kmph * conversion_factor -- Speed of the train in m/s

noncomputable def time_to_cross_pole : ℝ := length_meters / speed_mps -- Time to cross the pole

theorem time_to_cross_pole_correct :
  time_to_cross_pole = 800.064 / (160 * (1000 / 3600)) :=
sorry

end time_to_cross_pole_correct_l159_159790


namespace image_of_neg2_3_preimages_2_neg3_l159_159385

variables {A B : Type}
def f (x y : ℤ) : ℤ × ℤ := (x + y, x * y)

-- Prove that the image of (-2, 3) under f is (1, -6)
theorem image_of_neg2_3 : f (-2) 3 = (1, -6) := sorry

-- Find the preimages of (2, -3) under f
def preimages_of_2_neg3 (p : ℤ × ℤ) : Prop := f p.1 p.2 = (2, -3)

theorem preimages_2_neg3 : preimages_of_2_neg3 (-1, 3) ∧ preimages_of_2_neg3 (3, -1) := sorry

end image_of_neg2_3_preimages_2_neg3_l159_159385


namespace not_possible_total_47_l159_159306

open Nat

theorem not_possible_total_47 (h c : ℕ) : ¬ (13 * h + 5 * c = 47) :=
  sorry

end not_possible_total_47_l159_159306


namespace product_of_roots_cubic_l159_159049

theorem product_of_roots_cubic:
  (∀ x : ℝ, x^3 - 15 * x^2 + 60 * x - 45 = 0 → x = r_1 ∨ x = r_2 ∨ x = r_3) →
  r_1 * r_2 * r_3 = 45 :=
by
  intro h
  -- the proof should be filled in here
  sorry

end product_of_roots_cubic_l159_159049


namespace find_metal_molecular_weight_l159_159957

noncomputable def molecular_weight_of_metal (compound_mw: ℝ) (oh_mw: ℝ) : ℝ :=
  compound_mw - oh_mw

theorem find_metal_molecular_weight :
  let compound_mw := 171.00
  let oxygen_mw := 16.00
  let hydrogen_mw := 1.01
  let oh_ions := 2
  let oh_mw := oh_ions * (oxygen_mw + hydrogen_mw)
  molecular_weight_of_metal compound_mw oh_mw = 136.98 :=
by
  sorry

end find_metal_molecular_weight_l159_159957


namespace yellow_surface_area_min_fraction_l159_159374

/-- 
  Given a larger cube with 4-inch edges, constructed from 64 smaller cubes (each with 1-inch edge),
  where 50 cubes are colored blue, and 14 cubes are colored yellow. 
  If the large cube is crafted to display the minimum possible yellow surface area externally,
  then the fraction of the surface area of the large cube that is yellow is 7/48.
-/
theorem yellow_surface_area_min_fraction (n_smaller_cubes blue_cubes yellow_cubes : ℕ) 
  (edge_small edge_large : ℕ) (surface_area_larger_cube yellow_surface_min : ℕ) :
  edge_small = 1 → edge_large = 4 → n_smaller_cubes = 64 → 
  blue_cubes = 50 → yellow_cubes = 14 →
  surface_area_larger_cube = 96 → yellow_surface_min = 14 → 
  (yellow_surface_min : ℚ) / (surface_area_larger_cube : ℚ) = 7 / 48 := 
by 
  intros h_edge_small h_edge_large h_n h_blue h_yellow h_surface_area h_yellow_surface
  sorry

end yellow_surface_area_min_fraction_l159_159374


namespace problem_divisible_by_64_l159_159849

theorem problem_divisible_by_64 (n : ℕ) (hn : n > 0) : (3 ^ (2 * n + 2) - 8 * n - 9) % 64 = 0 := 
by
  sorry

end problem_divisible_by_64_l159_159849


namespace day_of_week_100_days_from_wednesday_l159_159479

theorem day_of_week_100_days_from_wednesday (today_is_wed : ∃ i : ℕ, i % 7 = 3) : 
  (100 % 7 + 3) % 7 = 5 := 
by
  sorry

end day_of_week_100_days_from_wednesday_l159_159479


namespace trajectory_of_circle_center_l159_159410

open Real

noncomputable def circle_trajectory_equation (x y : ℝ) : Prop :=
  (y ^ 2 = 8 * x - 16)

theorem trajectory_of_circle_center (x y : ℝ) :
  (∃ C : ℝ × ℝ, (C.1 = 4 ∧ C.2 = 0) ∧
    (∃ MN : ℝ × ℝ, (MN.1 = 0 ∧ MN.2 ^ 2 = 64) ∧
    (x = C.1 ∧ y = C.2)) ∧
    circle_trajectory_equation x y) :=
sorry

end trajectory_of_circle_center_l159_159410


namespace value_of_f_neg2_l159_159092

def f (x : ℤ) : ℤ := x^2 - 3 * x + 1

theorem value_of_f_neg2 : f (-2) = 11 := by
  sorry

end value_of_f_neg2_l159_159092


namespace points_lie_on_parabola_l159_159080

noncomputable def lies_on_parabola (t : ℝ) : Prop :=
  let x := Real.cos t ^ 2
  let y := Real.sin t * Real.cos t
  y ^ 2 = x * (1 - x)

-- Statement to prove
theorem points_lie_on_parabola : ∀ t : ℝ, lies_on_parabola t :=
by
  intro t
  sorry

end points_lie_on_parabola_l159_159080


namespace find_number_of_members_l159_159317

variable (n : ℕ)

-- We translate the conditions into Lean 4 definitions
def total_collection := 9216
def per_member_contribution := n

-- The goal is to prove that n = 96 given the total collection
theorem find_number_of_members (h : n * n = total_collection) : n = 96 := 
sorry

end find_number_of_members_l159_159317


namespace smallest_AAB_l159_159667

theorem smallest_AAB : ∃ (A B : ℕ), (1 <= A ∧ A <= 9) ∧ (1 <= B ∧ B <= 9) ∧ (AB = 10 * A + B) ∧ (AAB = 100 * A + 10 * A + B) ∧ (110 * A + B = 8 * (10 * A + B)) ∧ (AAB = 221) :=
by
  sorry

end smallest_AAB_l159_159667


namespace S_range_l159_159206

theorem S_range (x : ℝ) (y : ℝ) (S : ℝ) 
  (h1 : y = 2 * x - 1) 
  (h2 : 0 ≤ x) 
  (h3 : x ≤ 1 / 2) 
  (h4 : S = x * y) : 
  -1 / 8 ≤ S ∧ S ≤ 0 := 
sorry

end S_range_l159_159206


namespace five_coins_no_105_cents_l159_159233

theorem five_coins_no_105_cents :
  ¬ ∃ (a b c d e : ℕ), a + b + c + d + e = 5 ∧
    (a * 1 + b * 5 + c * 10 + d * 25 + e * 50 = 105) :=
sorry

end five_coins_no_105_cents_l159_159233


namespace probability_of_queen_after_first_queen_l159_159566

-- Define the standard deck
def standard_deck : Finset (Fin 54) := Finset.univ

-- Define the event of drawing the first queen
def first_queen (deck : Finset (Fin 54)) : Prop := -- placeholder defining first queen draw
  sorry

-- Define the event of drawing a queen immediately after the first queen
def queen_after_first_queen (deck : Finset (Fin 54)) : Prop :=
  sorry

-- Define the probability of an event given a condition
noncomputable def probability (event : Prop) (condition : Prop) : ℚ :=
  sorry

-- Main theorem statement
theorem probability_of_queen_after_first_queen : probability 
  (queen_after_first_queen standard_deck) (first_queen standard_deck) = 2/27 :=
sorry

end probability_of_queen_after_first_queen_l159_159566


namespace find_g_inv_f_8_l159_159704

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f_inv : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry
noncomputable def g_inv : ℝ → ℝ := sorry

axiom f_inv_g : ∀ x : ℝ, f_inv (g x) = x^2 - x
axiom g_bijective : Function.Bijective g

theorem find_g_inv_f_8 : g_inv (f 8) = (1 + Real.sqrt 33) / 2 :=
by
  -- proof is omitted
  sorry

end find_g_inv_f_8_l159_159704


namespace sin_double_angle_plus_pi_over_4_l159_159982

theorem sin_double_angle_plus_pi_over_4 (α : ℝ) 
  (h : Real.tan α = 3) : 
  Real.sin (2 * α + Real.pi / 4) = -Real.sqrt 2 / 10 := 
by 
  sorry

end sin_double_angle_plus_pi_over_4_l159_159982


namespace sum_of_cube_edges_l159_159904

theorem sum_of_cube_edges (edge_len : ℝ) (num_edges : ℕ) (lengths : ℝ) (h1 : edge_len = 15) (h2 : num_edges = 12) : lengths = num_edges * edge_len :=
by
  sorry

end sum_of_cube_edges_l159_159904


namespace linear_regression_change_l159_159431

theorem linear_regression_change : ∀ (x : ℝ), ∀ (y : ℝ), 
  y = 2 - 3.5 * x → (y - (2 - 3.5 * (x + 1))) = 3.5 :=
by
  intros x y h
  sorry

end linear_regression_change_l159_159431


namespace percentage_with_diploma_l159_159386

-- Define the percentages as variables for clarity
def low_income_perc := 0.25
def lower_middle_income_perc := 0.35
def upper_middle_income_perc := 0.25
def high_income_perc := 0.15

def low_income_diploma := 0.05
def lower_middle_income_diploma := 0.35
def upper_middle_income_diploma := 0.60
def high_income_diploma := 0.80

theorem percentage_with_diploma :
  (low_income_perc * low_income_diploma +
   lower_middle_income_perc * lower_middle_income_diploma +
   upper_middle_income_perc * upper_middle_income_diploma +
   high_income_perc * high_income_diploma) = 0.405 :=
by sorry

end percentage_with_diploma_l159_159386


namespace sum_of_dihedral_angles_leq_90_l159_159535
noncomputable section

-- Let θ1 and θ2 be angles formed by a line with two perpendicular planes
variable (θ1 θ2 : ℝ)

-- Define the condition stating the planes are perpendicular, and the line forms dihedral angles
def dihedral_angle_condition (θ1 θ2 : ℝ) : Prop := 
  θ1 ≥ 0 ∧ θ1 ≤ 90 ∧ θ2 ≥ 0 ∧ θ2 ≤ 90

-- The theorem statement capturing the problem
theorem sum_of_dihedral_angles_leq_90 
  (θ1 θ2 : ℝ) 
  (h : dihedral_angle_condition θ1 θ2) : 
  θ1 + θ2 ≤ 90 :=
sorry

end sum_of_dihedral_angles_leq_90_l159_159535


namespace fraction_of_students_with_mentor_l159_159132

theorem fraction_of_students_with_mentor (s n : ℕ) (h : n / 2 = s / 3) :
  (n / 2 + s / 3 : ℚ) / (n + s : ℚ) = 2 / 5 := by
  sorry

end fraction_of_students_with_mentor_l159_159132


namespace kenneth_money_left_l159_159400

noncomputable def baguettes : ℝ := 2 * 2
noncomputable def water : ℝ := 2 * 1

noncomputable def chocolate_bars_cost_before_discount : ℝ := 2 * 1.5
noncomputable def chocolate_bars_cost_after_discount : ℝ := chocolate_bars_cost_before_discount * (1 - 0.20)
noncomputable def chocolate_bars_final_cost : ℝ := chocolate_bars_cost_after_discount * 1.08

noncomputable def milk_cost_after_discount : ℝ := 3.5 * (1 - 0.10)

noncomputable def chips_cost_before_tax : ℝ := 2.5 + (2.5 * 0.50)
noncomputable def chips_final_cost : ℝ := chips_cost_before_tax * 1.08

noncomputable def total_cost : ℝ :=
  baguettes + water + chocolate_bars_final_cost + milk_cost_after_discount + chips_final_cost

noncomputable def initial_amount : ℝ := 50
noncomputable def amount_left : ℝ := initial_amount - total_cost

theorem kenneth_money_left : amount_left = 50 - 15.792 := by
  sorry

end kenneth_money_left_l159_159400


namespace cube_identity_l159_159032

theorem cube_identity (a : ℝ) (h : (a + 1/a) ^ 2 = 3) : a^3 + 1/a^3 = 0 := 
by
  sorry

end cube_identity_l159_159032


namespace no_b_satisfies_condition_l159_159173

noncomputable def f (b x : ℝ) : ℝ :=
  x^2 + 3 * b * x + 5 * b

theorem no_b_satisfies_condition :
  ∀ b : ℝ, ¬ (∃ x : ℝ, ∀ y : ℝ, |f b y| ≤ 5 → y = x) :=
by
  sorry

end no_b_satisfies_condition_l159_159173


namespace train_length_l159_159582

theorem train_length (speed_kmph : ℝ) (cross_time_sec : ℝ) (train_length : ℝ) :
  speed_kmph = 60 → cross_time_sec = 12 → train_length = 200.04 :=
by
  sorry

end train_length_l159_159582


namespace x_intercept_is_neg_three_halves_l159_159654

-- Definition of the points
def pointA : ℝ × ℝ := (-1, 1)
def pointB : ℝ × ℝ := (3, 9)

-- Statement of the theorem: The x-intercept of the line passing through the points is -3/2.
theorem x_intercept_is_neg_three_halves (A B : ℝ × ℝ)
    (hA : A = pointA)
    (hB : B = pointB) :
    ∃ x_intercept : ℝ, x_intercept = -3 / 2 := 
by
    sorry

end x_intercept_is_neg_three_halves_l159_159654


namespace range_of_m_for_decreasing_interval_l159_159484

def function_monotonically_decreasing_in_interval (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, a < x → x < y → y < b → f y ≤ f x

def f (x : ℝ) : ℝ := x ^ 3 - 12 * x

theorem range_of_m_for_decreasing_interval :
  ∀ m : ℝ, function_monotonically_decreasing_in_interval f (2 * m) (m + 1) → -1 ≤ m ∧ m < 1 :=
by
  sorry

end range_of_m_for_decreasing_interval_l159_159484


namespace value_2_stddevs_less_than_mean_l159_159686

-- Definitions based on the conditions
def mean : ℝ := 10.5
def stddev : ℝ := 1
def value := mean - 2 * stddev

-- Theorem we aim to prove
theorem value_2_stddevs_less_than_mean : value = 8.5 := by
  -- proof will go here
  sorry

end value_2_stddevs_less_than_mean_l159_159686


namespace point_on_line_l159_159655

theorem point_on_line (x : ℝ) : 
    (∃ k : ℝ, (-4) = k * (-4) + 8) → 
    (-4 = 2 * x + 8) → 
    x = -6 := 
sorry

end point_on_line_l159_159655


namespace union_of_A_and_B_intersection_of_A_and_B_l159_159724

noncomputable def A : Set ℝ := { x | -4 < x ∧ x < 4 }
noncomputable def B : Set ℝ := { x | x > 3 ∨ x < 1 }

theorem union_of_A_and_B : A ∪ B = Set.univ :=
by
  sorry

theorem intersection_of_A_and_B : A ∩ B = { x | (-4 < x ∧ x < 1) ∨ (3 < x ∧ x < 4) } :=
by
  sorry

end union_of_A_and_B_intersection_of_A_and_B_l159_159724


namespace find_m_l159_159631

theorem find_m (m : ℝ) : 
  (∃ α β : ℝ, (α + β = 2 * (m + 1)) ∧ (α * β = m + 4) ∧ ((1 / α) + (1 / β) = 1)) → m = 2 :=
by
  sorry

end find_m_l159_159631


namespace sector_area_l159_159813

theorem sector_area (θ : ℝ) (r : ℝ) (hθ : θ = (2 * Real.pi) / 3) (hr : r = Real.sqrt 3) : 
    (1/2 * r^2 * θ) = Real.pi :=
by
  sorry

end sector_area_l159_159813


namespace younger_person_age_l159_159461

theorem younger_person_age 
  (y e : ℕ)
  (h1 : e = y + 20)
  (h2 : e - 4 = 5 * (y - 4)) : 
  y = 9 := 
sorry

end younger_person_age_l159_159461


namespace white_cannot_lose_l159_159674

-- Define a type to represent the game state
structure Game :=
  (state : Type)
  (white_move : state → state)
  (black_move : state → state)
  (initial : state)

-- Define a type to represent the double chess game conditions
structure DoubleChess extends Game :=
  (double_white_move : state → state)
  (double_black_move : state → state)

-- Define the hypothesis based on the conditions
noncomputable def white_has_no_losing_strategy (g : DoubleChess) : Prop :=
  ∃ s, g.double_white_move (g.double_white_move s) = g.initial

theorem white_cannot_lose (g : DoubleChess) :
  white_has_no_losing_strategy g :=
sorry

end white_cannot_lose_l159_159674


namespace problem_statement_l159_159344

noncomputable def G (x : ℝ) : ℝ := ((x + 1) ^ 2) / 2 - 4

theorem problem_statement : G (G (G 0)) = -3.9921875 :=
by
  sorry

end problem_statement_l159_159344


namespace number_of_boys_is_320_l159_159256

-- Definition of the problem's conditions
variable (B G : ℕ)
axiom condition1 : B + G = 400
axiom condition2 : G = (B / 400) * 100

-- Stating the theorem to prove number of boys is 320
theorem number_of_boys_is_320 : B = 320 :=
by
  sorry

end number_of_boys_is_320_l159_159256


namespace correct_value_l159_159759

theorem correct_value (x : ℕ) (h : 14 * x = 42) : 12 * x = 36 := by
  sorry

end correct_value_l159_159759


namespace weight_difference_l159_159270

-- Defining the weights of the individuals
variables (a b c d e : ℝ)

-- Given conditions as hypotheses
def conditions :=
  (a = 75) ∧
  ((a + b + c) / 3 = 84) ∧
  ((a + b + c + d) / 4 = 80) ∧
  ((b + c + d + e) / 4 = 79)

-- Theorem statement to prove the desired result
theorem weight_difference (h : conditions a b c d e) : e - d = 3 :=
by
  sorry

end weight_difference_l159_159270


namespace smallest_positive_root_l159_159805

noncomputable def alpha : ℝ := Real.arctan (2 / 9)
noncomputable def beta : ℝ := Real.arctan (6 / 7)

theorem smallest_positive_root :
  ∃ x > 0, (2 * Real.sin (6 * x) + 9 * Real.cos (6 * x) = 6 * Real.sin (2 * x) + 7 * Real.cos (2 * x))
    ∧ x = (alpha + beta) / 8 := sorry

end smallest_positive_root_l159_159805


namespace evaluate_expression_l159_159568

def a : ℕ := 3
def b : ℕ := 2

theorem evaluate_expression : (a^2 * a^5) / (b^2 / b^3) = 4374 := by
  sorry

end evaluate_expression_l159_159568


namespace tan_15pi_over_4_is_neg1_l159_159434

noncomputable def tan_15pi_over_4 : ℝ :=
  Real.tan (15 * Real.pi / 4)

theorem tan_15pi_over_4_is_neg1 :
  tan_15pi_over_4 = -1 :=
sorry

end tan_15pi_over_4_is_neg1_l159_159434


namespace find_x_for_parallel_vectors_l159_159023

def vector := (ℝ × ℝ)

def a (x : ℝ) : vector := (1, x)
def b (x : ℝ) : vector := (2, 2 - x)

def are_parallel (v w : vector) : Prop :=
  ∃ k : ℝ, v = (k * w.1, k * w.2)

theorem find_x_for_parallel_vectors :
  ∀ x : ℝ, are_parallel (a x) (b x) → x = 2/3 :=
by
  sorry

end find_x_for_parallel_vectors_l159_159023


namespace find_c_l159_159459

theorem find_c (c : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Iio (-2) ∪ Set.Ioi 3 → x^2 - c * x + 6 > 0) → c = 1 :=
by
  sorry

end find_c_l159_159459


namespace average_salary_rest_l159_159768

noncomputable def average_salary_of_the_rest : ℕ := 6000

theorem average_salary_rest 
  (N : ℕ) 
  (A : ℕ)
  (T : ℕ)
  (A_T : ℕ)
  (Nr : ℕ)
  (Ar : ℕ)
  (H1 : N = 42)
  (H2 : A = 8000)
  (H3 : T = 7)
  (H4 : A_T = 18000)
  (H5 : Nr = N - T)
  (H6 : Nr = 42 - 7)
  (H7 : Ar = 6000)
  (H8 : 42 * 8000 = (Nr * Ar) + (T * 18000))
  : Ar = average_salary_of_the_rest :=
by
  sorry

end average_salary_rest_l159_159768


namespace opposite_of_neg_six_l159_159652

theorem opposite_of_neg_six : -(-6) = 6 := 
by
  sorry

end opposite_of_neg_six_l159_159652


namespace calculation_correct_l159_159249

theorem calculation_correct : (5 * 7 + 9 * 4 - 36 / 3 : ℤ) = 59 := by
  sorry

end calculation_correct_l159_159249


namespace triangular_prism_distance_sum_l159_159425

theorem triangular_prism_distance_sum (V K H1 H2 H3 H4 S1 S2 S3 S4 : ℝ)
  (h1 : S1 = K)
  (h2 : S2 = 2 * K)
  (h3 : S3 = 3 * K)
  (h4 : S4 = 4 * K)
  (hV : (S1 * H1 + S2 * H2 + S3 * H3 + S4 * H4) / 3 = V) :
  H1 + 2 * H2 + 3 * H3 + 4 * H4 = 3 * V / K :=
by sorry

end triangular_prism_distance_sum_l159_159425


namespace domain_sqrt_frac_l159_159890

theorem domain_sqrt_frac (x : ℝ) :
  (x^2 + 4*x + 3 ≠ 0) ∧ (x + 3 ≥ 0) ↔ ((x ∈ Set.Ioc (-3) (-1)) ∨ (x ∈ Set.Ioi (-1))) :=
by
  sorry

end domain_sqrt_frac_l159_159890


namespace soccer_team_wins_l159_159935

theorem soccer_team_wins :
  ∃ W D : ℕ, 
    (W + 2 + D = 20) ∧  -- total games
    (3 * W + D = 46) ∧  -- total points
    (W = 14) :=         -- correct answer
by
  sorry

end soccer_team_wins_l159_159935


namespace min_value_proof_l159_159354

noncomputable def min_value_of_expression (a b c d e f g h : ℝ) : ℝ :=
  (a * e)^2 + (b * f)^2 + (c * g)^2 + (d * h)^2

theorem min_value_proof (a b c d e f g h : ℝ)
  (h1 : a * b * c * d = 8)
  (h2 : e * f * g * h = 16) :
  ∃ (x : ℝ), x = 32 ∧ min_value_of_expression a b c d e f g h = x :=
by
  use 32
  sorry

end min_value_proof_l159_159354


namespace lowest_common_denominator_l159_159540

theorem lowest_common_denominator (a b c : ℕ) (h1 : a = 9) (h2 : b = 4) (h3 : c = 18) : Nat.lcm (Nat.lcm a b) c = 36 :=
by
  -- Introducing the given conditions
  rw [h1, h2, h3]
  -- Compute the LCM of the provided values
  sorry

end lowest_common_denominator_l159_159540


namespace total_bears_l159_159858

-- Definitions based on given conditions
def brown_bears : ℕ := 15
def white_bears : ℕ := 24
def black_bears : ℕ := 27

-- Theorem to prove the total number of bears
theorem total_bears : brown_bears + white_bears + black_bears = 66 := by
  sorry

end total_bears_l159_159858


namespace math_problem_l159_159738

theorem math_problem (x y : ℕ) (h1 : (x + y * I)^3 = 2 + 11 * I) (h2 : 0 < x) (h3 : 0 < y) : 
  x + y * I = 2 + I :=
sorry

end math_problem_l159_159738


namespace power_of_three_l159_159398

theorem power_of_three (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
by
  sorry

end power_of_three_l159_159398


namespace quadratic_real_roots_k_eq_one_l159_159107

theorem quadratic_real_roots_k_eq_one 
  (k : ℕ) 
  (h_nonneg : k ≥ 0) 
  (h_real_roots : ∃ x : ℝ, k * x^2 - 2 * x + 1 = 0) : 
  k = 1 := 
sorry

end quadratic_real_roots_k_eq_one_l159_159107


namespace average_value_of_powers_l159_159167

theorem average_value_of_powers (z : ℝ) : 
  (z^2 + 3*z^2 + 6*z^2 + 12*z^2 + 24*z^2) / 5 = 46*z^2 / 5 :=
by
  sorry

end average_value_of_powers_l159_159167


namespace lana_needs_to_sell_more_muffins_l159_159932

/--
Lana aims to sell 20 muffins at the bake sale.
She sells 12 muffins in the morning.
She sells another 4 in the afternoon.
How many more muffins does Lana need to sell to hit her goal?
-/
theorem lana_needs_to_sell_more_muffins (goal morningSales afternoonSales : ℕ)
  (h_goal : goal = 20) (h_morning : morningSales = 12) (h_afternoon : afternoonSales = 4) :
  goal - (morningSales + afternoonSales) = 4 :=
by
  sorry

end lana_needs_to_sell_more_muffins_l159_159932


namespace divisibility_of_f_by_cubic_factor_l159_159470

noncomputable def f (x : ℂ) (m n : ℕ) : ℂ := x^(3 * m + 2) + (-x^2 - 1)^(3 * n + 1) + 1

theorem divisibility_of_f_by_cubic_factor (m n : ℕ) : ∀ x : ℂ, x^2 + x + 1 = 0 → f x m n = 0 :=
by
  sorry

end divisibility_of_f_by_cubic_factor_l159_159470


namespace ratio_problem_l159_159608

variable (a b c d : ℚ)

theorem ratio_problem
  (h1 : a / b = 5)
  (h2 : b / c = 1 / 4)
  (h3 : c / d = 7) :
  d / a = 4 / 35 :=
by
  sorry

end ratio_problem_l159_159608


namespace find_number_l159_159186

theorem find_number (x : ℝ) (n : ℝ) (h1 : x = 12) (h2 : (27 / n) * x - 18 = 3 * x + 27) : n = 4 :=
sorry

end find_number_l159_159186


namespace average_rate_of_interest_l159_159760

/-- Given:
    1. A woman has a total of $7500 invested,
    2. Part of the investment is at 5% interest,
    3. The remainder of the investment is at 7% interest,
    4. The annual returns from both investments are equal,
    Prove:
    The average rate of interest realized on her total investment is 5.8%.
-/
theorem average_rate_of_interest
  (total_investment : ℝ) (interest_5_percent : ℝ) (interest_7_percent : ℝ)
  (annual_return_equal : 0.05 * (total_investment - interest_7_percent) = 0.07 * interest_7_percent)
  (total_investment_eq : total_investment = 7500) : 
  (interest_5_percent / total_investment) = 0.058 :=
by
  -- conditions given
  have h1 : total_investment = 7500 := total_investment_eq
  have h2 : 0.05 * (total_investment - interest_7_percent) = 0.07 * interest_7_percent := annual_return_equal

  -- final step, sorry is used to skip the proof
  sorry

end average_rate_of_interest_l159_159760


namespace sum_of_three_squares_l159_159889

theorem sum_of_three_squares (a b c : ℤ) (h1 : 2 * a + 2 * b + c = 27) (h2 : a + 3 * b + c = 25) : 3 * c = 33 :=
  sorry

end sum_of_three_squares_l159_159889


namespace triangle_angle_A_l159_159074

theorem triangle_angle_A (A B C : ℝ) (h1 : C = 3 * B) (h2 : B = 30) (h3 : A + B + C = 180) : A = 60 := by
  sorry

end triangle_angle_A_l159_159074


namespace circle_equation_tangent_l159_159019

theorem circle_equation_tangent (h : ∀ x y : ℝ, (4 * x + 3 * y - 35 ≠ 0) → ((x - 1) ^ 2 + (y - 2) ^ 2 = 25)) :
    ∃ c : ℝ × ℝ, c = (1, 2) ∧ ∃ r : ℝ, r = 5 ∧ ∀ x y : ℝ, (4 * x + 3 * y - 35 ≠ 0) → ((x - 1) ^ 2 + (y - 2) ^ 2 = r ^ 2) := 
by
    sorry

end circle_equation_tangent_l159_159019


namespace sum_of_sins_is_zero_l159_159923

variable {x y z : ℝ}

theorem sum_of_sins_is_zero
  (h1 : Real.sin x = Real.tan y)
  (h2 : Real.sin y = Real.tan z)
  (h3 : Real.sin z = Real.tan x) :
  Real.sin x + Real.sin y + Real.sin z = 0 :=
sorry

end sum_of_sins_is_zero_l159_159923


namespace cos_three_theta_l159_159380

theorem cos_three_theta (θ : ℝ) (h : Real.cos θ = 1 / 4) : Real.cos (3 * θ) = -11 / 16 := by
  sorry

end cos_three_theta_l159_159380


namespace odd_power_divisible_by_sum_l159_159017

theorem odd_power_divisible_by_sum (x y : ℝ) (k : ℕ) (h : k > 0) :
  (x^((2*k - 1)) + y^((2*k - 1))) ∣ (x^(2*k + 1) + y^(2*k + 1)) :=
sorry

end odd_power_divisible_by_sum_l159_159017


namespace remainder_of_expression_l159_159356

theorem remainder_of_expression (n : ℤ) (h : n % 8 = 3) : (4 * n - 10) % 8 = 2 :=
sorry

end remainder_of_expression_l159_159356


namespace banana_ratio_proof_l159_159193

-- Definitions based on conditions
def initial_bananas := 310
def bananas_left_on_tree := 100
def bananas_eaten := 70

-- Auxiliary calculations for clarity
def bananas_cut := initial_bananas - bananas_left_on_tree
def bananas_remaining := bananas_cut - bananas_eaten

-- Theorem we need to prove
theorem banana_ratio_proof :
  bananas_remaining / bananas_eaten = 2 :=
by
  sorry

end banana_ratio_proof_l159_159193


namespace least_value_difference_l159_159742

noncomputable def least_difference (x : ℝ) : ℝ := 6 - 13/5

theorem least_value_difference (x n m : ℝ) (h1 : 2*x + 5 + 4*x - 3 > x + 15)
                               (h2 : 2*x + 5 + x + 15 > 4*x - 3)
                               (h3 : 4*x - 3 + x + 15 > 2*x + 5)
                               (h4 : x + 15 > 2*x + 5)
                               (h5 : x + 15 > 4*x - 3)
                               (h_m : m = 13/5) (h_n : n = 6)
                               (hx : m < x ∧ x < n) :
  n - m = 17 / 5 :=
  by sorry

end least_value_difference_l159_159742


namespace Eddy_travel_time_l159_159684

theorem Eddy_travel_time :
  ∀ (T_F D_F D_E : ℕ) (S_ratio : ℝ),
    T_F = 4 →
    D_F = 360 →
    D_E = 600 →
    S_ratio = 2.2222222222222223 →
    ((D_F / T_F : ℝ) * S_ratio ≠ 0) →
    D_E / ((D_F / T_F) * S_ratio) = 3 :=
by
  intros T_F D_F D_E S_ratio ht hf hd hs hratio
  sorry  -- Proof to be provided

end Eddy_travel_time_l159_159684


namespace abs_f_x_minus_f_a_lt_l159_159176

variable {R : Type*} [LinearOrderedField R]

def f (x : R) (c : R) := x ^ 2 - x + c

theorem abs_f_x_minus_f_a_lt (x a c : R) (h : abs (x - a) < 1) : 
  abs (f x c - f a c) < 2 * (abs a + 1) :=
by
  sorry

end abs_f_x_minus_f_a_lt_l159_159176


namespace ball_hits_ground_time_l159_159613

theorem ball_hits_ground_time (t : ℝ) : 
  (∃ t : ℝ, -10 * t^2 + 40 * t + 50 = 0 ∧ t ≥ 0) → t = 5 := 
by
  -- placeholder for proof
  sorry

end ball_hits_ground_time_l159_159613


namespace find_ck_l159_159619

theorem find_ck (d r k : ℕ) (a_n b_n c_n : ℕ → ℕ) 
  (h_an : ∀ n, a_n n = 1 + (n - 1) * d)
  (h_bn : ∀ n, b_n n = r ^ (n - 1))
  (h_cn : ∀ n, c_n n = a_n n + b_n n)
  (h_ckm1 : c_n (k - 1) = 30)
  (h_ckp1 : c_n (k + 1) = 300) :
  c_n k = 83 := 
sorry

end find_ck_l159_159619


namespace total_marbles_l159_159772

theorem total_marbles (bowl2_capacity : ℕ) (h₁ : bowl2_capacity = 600)
    (h₂ : 3 / 4 * bowl2_capacity = 450) : 600 + (3 / 4 * 600) = 1050 := by
  sorry

end total_marbles_l159_159772


namespace mikes_earnings_l159_159361

-- Definitions based on the conditions:
def blade_cost : ℕ := 47
def game_count : ℕ := 9
def game_cost : ℕ := 6

-- The total money Mike made:
def total_money (M : ℕ) : Prop :=
  M - (blade_cost + game_count * game_cost) = 0

theorem mikes_earnings (M : ℕ) : total_money M → M = 101 :=
by
  sorry

end mikes_earnings_l159_159361


namespace max_product_three_distinct_nats_sum_48_l159_159524

open Nat

theorem max_product_three_distinct_nats_sum_48
  (a b c : ℕ) (h_distinct: a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_sum: a + b + c = 48) :
  a * b * c ≤ 4080 :=
sorry

end max_product_three_distinct_nats_sum_48_l159_159524


namespace face_opposite_A_is_F_l159_159519

structure Cube where
  adjacency : String → String → Prop
  exists_face : ∃ a b c d e f : String, True

variable 
  (C : Cube)
  (adjA_B : C.adjacency "A" "B")
  (adjA_C : C.adjacency "A" "C")
  (adjB_D : C.adjacency "B" "D")

theorem face_opposite_A_is_F : 
  ∃ f : String, f = "F" ∧ ∀ g : String, (C.adjacency "A" g → g ≠ "F") :=
by 
  sorry

end face_opposite_A_is_F_l159_159519


namespace sum_of_ages_26_l159_159604

-- Define an age predicate to manage the three ages
def is_sum_of_ages (kiana twin : ℕ) : Prop :=
  kiana < twin ∧ twin * twin * kiana = 180 ∧ (kiana + twin + twin = 26)

theorem sum_of_ages_26 : 
  ∃ (kiana twin : ℕ), is_sum_of_ages kiana twin :=
by 
  sorry

end sum_of_ages_26_l159_159604


namespace consecutive_numbers_N_l159_159012

theorem consecutive_numbers_N (N : ℕ) (h : ∀ k, 0 < k → k < 15 → N + k < 81) : N = 66 :=
sorry

end consecutive_numbers_N_l159_159012


namespace common_tangent_theorem_l159_159429

-- Define the first circle with given equation (x+2)^2 + (y-2)^2 = 1
def circle1 (x y : ℝ) : Prop := (x + 2)^2 + (y - 2)^2 = 1

-- Define the second circle with given equation (x-2)^2 + (y-5)^2 = 16
def circle2 (x y : ℝ) : Prop := (x - 2)^2 + (y - 5)^2 = 16

-- Define a predicate that expresses the concept of common tangents between two circles
def common_tangents_count (circle1 circle2 : ℝ → ℝ → Prop) : ℕ := sorry

-- The statement to prove that the number of common tangents is 3
theorem common_tangent_theorem : common_tangents_count circle1 circle2 = 3 :=
by
  -- We would proceed with the proof if required, but we end with sorry as requested.
  sorry

end common_tangent_theorem_l159_159429


namespace division_expression_evaluation_l159_159413

theorem division_expression_evaluation : 120 / (6 / 2) = 40 := by
  sorry

end division_expression_evaluation_l159_159413


namespace max_valid_n_eq_3210_l159_159709

-- Define the digit sum function S
def digit_sum (n : ℕ) : ℕ :=
  (Nat.digits 10 n).sum

-- The condition S(3n) = 3S(n) and all digits of n are distinct
def valid_n (n : ℕ) : Prop :=
  digit_sum (3 * n) = 3 * digit_sum n ∧ (Nat.digits 10 n).Nodup

-- Prove that the maximum value of such n is 3210
theorem max_valid_n_eq_3210 : ∃ n : ℕ, valid_n n ∧ n = 3210 :=
by
  existsi 3210
  sorry

end max_valid_n_eq_3210_l159_159709


namespace train_speed_l159_159572

theorem train_speed (train_length bridge_length cross_time : ℝ)
  (h1 : train_length = 250)
  (h2 : bridge_length = 150)
  (h3 : cross_time = 25) :
  (train_length + bridge_length) / cross_time = 16 :=
by
  sorry

end train_speed_l159_159572


namespace trajectory_midpoint_l159_159635

-- Defining the point A(-2, 0)
def A : ℝ × ℝ := (-2, 0)

-- Defining the curve equation
def curve (x y : ℝ) : Prop := 2 * y^2 = x

-- Coordinates of P based on the midpoint formula
def P (x y : ℝ) : ℝ × ℝ := (2 * x + 2, 2 * y)

-- The target trajectory equation
def trajectory_eqn (x y : ℝ) : Prop := x = 4 * y^2 - 1

-- The theorem to be proved
theorem trajectory_midpoint (x y : ℝ) :
  curve (2 * y) (2 * x + 2) → 
  trajectory_eqn x y :=
sorry

end trajectory_midpoint_l159_159635


namespace cubic_has_one_real_root_l159_159657

theorem cubic_has_one_real_root :
  (∃ x : ℝ, x^3 - 6*x^2 + 9*x - 10 = 0) ∧ ∀ x y : ℝ, (x^3 - 6*x^2 + 9*x - 10 = 0) ∧ (y^3 - 6*y^2 + 9*y - 10 = 0) → x = y :=
by
  sorry

end cubic_has_one_real_root_l159_159657


namespace part1_part2_l159_159846

-- Condition: x = -1 is a solution to 2a + 4x = x + 5a
def is_solution_x (a x : ℤ) : Prop := 2 * a + 4 * x = x + 5 * a

-- Part 1: Prove a = -1 given x = -1
theorem part1 (x : ℤ) (h1 : x = -1) (h2 : is_solution_x a x) : a = -1 :=
by sorry

-- Condition: a = -1
def a_value (a : ℤ) : Prop := a = -1

-- Condition: ay + 6 = 6a + 2y
def equation_in_y (a y : ℤ) : Prop := a * y + 6 = 6 * a + 2 * y

-- Part 2: Prove y = 4 given a = -1
theorem part2 (a y : ℤ) (h1 : a_value a) (h2 : equation_in_y a y) : y = 4 :=
by sorry

end part1_part2_l159_159846


namespace total_number_of_animals_l159_159710

-- Definitions for the animal types
def heads_per_hen := 2
def legs_per_hen := 8
def heads_per_peacock := 3
def legs_per_peacock := 9
def heads_per_zombie_hen := 6
def legs_per_zombie_hen := 12

-- Given total heads and legs
def total_heads := 800
def total_legs := 2018

-- Proof that the total number of animals is 203
theorem total_number_of_animals : 
  ∀ (H P Z : ℕ), 
    heads_per_hen * H + heads_per_peacock * P + heads_per_zombie_hen * Z = total_heads
    ∧ legs_per_hen * H + legs_per_peacock * P + legs_per_zombie_hen * Z = total_legs 
    → H + P + Z = 203 :=
by
  sorry

end total_number_of_animals_l159_159710


namespace g_neg6_eq_neg1_l159_159987

def f : ℝ → ℝ := fun x => 4 * x - 6
def g : ℝ → ℝ := fun x => 2 * x^2 + 7 * x - 1

theorem g_neg6_eq_neg1 : g (-6) = -1 := by
  sorry

end g_neg6_eq_neg1_l159_159987


namespace pure_imaginary_product_imaginary_part_fraction_l159_159700

-- Part 1
theorem pure_imaginary_product (m : ℝ) (i : ℂ) (h1 : i * i = -1) (h2 : z1 = m + i) (h3 : z2 = 2 + m * i) :
  (z1 * z2).re = 0 ↔ m = 0 := 
sorry

-- Part 2
theorem imaginary_part_fraction (m : ℝ) (i : ℂ) (h1 : i * i = -1) (h2 : z1 = m + i) (h3 : z2 = 2 + m * i)
  (h4 : z1^2 - 2 * z1 + 2 = 0) :
  (z2 / z1).im = -1 / 2 :=
sorry

end pure_imaginary_product_imaginary_part_fraction_l159_159700


namespace unique_solution_a_exists_l159_159835

open Real

noncomputable def equation (a x : ℝ) :=
  4 * a^2 + 3 * x * log x + 3 * (log x)^2 = 13 * a * log x + a * x

theorem unique_solution_a_exists : 
  ∃! a : ℝ, ∃ x : ℝ, 0 < x ∧ equation a x :=
sorry

end unique_solution_a_exists_l159_159835


namespace sum_of_squares_l159_159663

theorem sum_of_squares (n m : ℕ) (h : 2 * m = n^2 + 1) : ∃ k : ℕ, m = k^2 + (k - 1)^2 :=
sorry

end sum_of_squares_l159_159663


namespace shaded_area_l159_159856

theorem shaded_area 
  (R r : ℝ) 
  (h_area_larger_circle : π * R ^ 2 = 100 * π) 
  (h_shaded_larger_fraction : 2 / 3 = (area_shaded_larger / (π * R ^ 2))) 
  (h_relationship_radius : r = R / 2) 
  (h_area_smaller_circle : π * r ^ 2 = 25 * π)
  (h_shaded_smaller_fraction : 1 / 3 = (area_shaded_smaller / (π * r ^ 2))) : 
  (area_shaded_larger + area_shaded_smaller = 75 * π) := 
sorry

end shaded_area_l159_159856


namespace find_positive_A_l159_159449

theorem find_positive_A (A : ℕ) : (A^2 + 7^2 = 130) → A = 9 :=
by
  intro h
  sorry

end find_positive_A_l159_159449


namespace number_of_non_symmetric_letters_is_3_l159_159241

def letters_in_JUNIOR : List Char := ['J', 'U', 'N', 'I', 'O', 'R']

def axis_of_symmetry (c : Char) : Bool :=
  match c with
  | 'J' => false
  | 'U' => true
  | 'N' => false
  | 'I' => true
  | 'O' => true
  | 'R' => false
  | _   => false

def letters_with_no_symmetry : List Char :=
  letters_in_JUNIOR.filter (λ c => ¬axis_of_symmetry c)

theorem number_of_non_symmetric_letters_is_3 :
  letters_with_no_symmetry.length = 3 :=
by
  sorry

end number_of_non_symmetric_letters_is_3_l159_159241


namespace total_social_media_hours_in_a_week_l159_159888

variable (daily_social_media_hours : ℕ) (days_in_week : ℕ)

theorem total_social_media_hours_in_a_week
(h1 : daily_social_media_hours = 3)
(h2 : days_in_week = 7) :
daily_social_media_hours * days_in_week = 21 := by
  sorry

end total_social_media_hours_in_a_week_l159_159888


namespace projection_of_a_onto_b_is_three_l159_159030

def vec_a : ℝ × ℝ := (3, 4)
def vec_b : ℝ × ℝ := (1, 0)

noncomputable def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)
noncomputable def projection (u v : ℝ × ℝ) : ℝ := dot_product u v / magnitude v

theorem projection_of_a_onto_b_is_three : projection vec_a vec_b = 3 := by
  sorry

end projection_of_a_onto_b_is_three_l159_159030


namespace value_of_phi_l159_159089

theorem value_of_phi { φ : ℝ } (hφ1 : 0 < φ) (hφ2 : φ < π)
  (symm_condition : ∃ k : ℤ, -π / 8 + φ = k * π + π / 2) : φ = 3 * π / 4 := 
by 
  sorry

end value_of_phi_l159_159089


namespace problem1_problem2_l159_159590

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.log x - 1

noncomputable def g (x : ℝ) : ℝ := x / Real.exp x

theorem problem1 (a : ℝ) (h1 : 2 / Real.exp 2 < a) (h2 : a < 1 / Real.exp 1) :
  ∃ (x1 x2 : ℝ), (0 < x1 ∧ x1 < 2) ∧ (0 < x2 ∧ x2 < 2) ∧ x1 ≠ x2 ∧ g x1 = a ∧ g x2 = a :=
sorry

theorem problem2 : ∀ x > 0, f x + 2 / (Real.exp 1 * g x) > 0 :=
sorry

end problem1_problem2_l159_159590


namespace exists_positive_m_dividing_f_100_l159_159751

noncomputable def f (x : ℤ) : ℤ := 3 * x + 2

theorem exists_positive_m_dividing_f_100:
  ∃ (m : ℤ), m > 0 ∧ 19881 ∣ (3^100 * (m + 1) - 1) :=
by
  sorry

end exists_positive_m_dividing_f_100_l159_159751


namespace xyz_inequality_l159_159015

theorem xyz_inequality (a b c d : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d) 
  (h_sum : a + b + c + d = 1) :
  (b * c * d) / (1 - a)^2 + (a * c * d) / (1 - b)^2 + 
  (a * b * d) / (1 - c)^2 + (a * b * c) / (1 - d)^2 ≤ 1 / 9 :=
sorry

end xyz_inequality_l159_159015


namespace range_f_neg2_l159_159480

noncomputable def f (a b x : ℝ): ℝ := a * x^2 + b * x

theorem range_f_neg2 (a b : ℝ) (h1 : 1 ≤ f a b (-1)) (h2 : f a b (-1) ≤ 2)
  (h3 : 3 ≤ f a b 1) (h4 : f a b 1 ≤ 4) : 6 ≤ f a b (-2) ∧ f a b (-2) ≤ 10 :=
by
  sorry

end range_f_neg2_l159_159480


namespace solve_for_b_l159_159021

theorem solve_for_b (a b : ℚ) 
  (h1 : 8 * a + 3 * b = -1) 
  (h2 : a = b - 3 ) : 
  5 * b = 115 / 11 := 
by 
  sorry

end solve_for_b_l159_159021


namespace range_of_a_l159_159122

variable (a : ℝ)

def p (a : ℝ) : Prop := 3/2 < a ∧ a < 5/2
def q (a : ℝ) : Prop := 2 ≤ a ∧ a ≤ 4

theorem range_of_a (h₁ : ¬(p a ∧ q a)) (h₂ : p a ∨ q a) : (3/2 < a ∧ a < 2) ∨ (5/2 ≤ a ∧ a ≤ 4) :=
sorry

end range_of_a_l159_159122


namespace pages_per_side_is_4_l159_159424

-- Define the conditions
def num_books := 2
def pages_per_book := 600
def sheets_used := 150
def sides_per_sheet := 2

-- Define the total number of pages and sides
def total_pages := num_books * pages_per_book
def total_sides := sheets_used * sides_per_sheet

-- Prove the number of pages per side is 4
theorem pages_per_side_is_4 : total_pages / total_sides = 4 := by
  sorry

end pages_per_side_is_4_l159_159424


namespace total_chairs_in_canteen_l159_159441

theorem total_chairs_in_canteen (numRoundTables : ℕ) (numRectangularTables : ℕ) 
                                (chairsPerRoundTable : ℕ) (chairsPerRectangularTable : ℕ)
                                (h1 : numRoundTables = 2)
                                (h2 : numRectangularTables = 2)
                                (h3 : chairsPerRoundTable = 6)
                                (h4 : chairsPerRectangularTable = 7) : 
                                (numRoundTables * chairsPerRoundTable + numRectangularTables * chairsPerRectangularTable = 26) :=
by
  sorry

end total_chairs_in_canteen_l159_159441


namespace minimum_dot_product_l159_159045

noncomputable def min_AE_dot_AF : ℝ :=
  let AB : ℝ := 2
  let BC : ℝ := 1
  let AD : ℝ := 1
  let CD : ℝ := 1
  let angle_ABC : ℝ := 60 -- this is 60 degrees, which should be converted to radians if we need to use it
  sorry

theorem minimum_dot_product :
  let AB : ℝ := 2
  let BC : ℝ := 1
  let AD : ℝ := 1
  let CD : ℝ := 1
  let angle_ABC : ℝ := 60
  ∃ (E F : ℝ), (min_AE_dot_AF = 29 / 18) :=
    sorry

end minimum_dot_product_l159_159045


namespace multiply_powers_l159_159789

theorem multiply_powers (x : ℝ) : x^3 * x^3 = x^6 :=
by sorry

end multiply_powers_l159_159789


namespace avg_height_eq_61_l159_159204

-- Define the constants and conditions
def Brixton : ℕ := 64
def Zara : ℕ := 64
def Zora := Brixton - 8
def Itzayana := Zora + 4

-- Define the total height of the four people
def total_height := Brixton + Zara + Zora + Itzayana

-- Define the average height
def average_height := total_height / 4

-- Theorem stating that the average height is 61 inches
theorem avg_height_eq_61 : average_height = 61 := by
  sorry

end avg_height_eq_61_l159_159204


namespace part1_part2_l159_159612

-- Definition of the conditions given
def february_parcels : ℕ := 200000
def april_parcels : ℕ := 338000
def monthly_growth_rate : ℝ := 0.3

-- Problem 1: Proving the monthly growth rate is 0.3
theorem part1 (x : ℝ) (h : february_parcels * (1 + x)^2 = april_parcels) : x = monthly_growth_rate :=
  sorry

-- Problem 2: Proving the number of parcels in May is less than 450,000 with the given growth rate
theorem part2 (h : monthly_growth_rate = 0.3 ) : february_parcels * (1 + monthly_growth_rate)^3 < 450000 :=
  sorry

end part1_part2_l159_159612


namespace alex_score_l159_159966

theorem alex_score (n : ℕ) (avg19 avg20 alex : ℚ)
  (h1 : n = 20)
  (h2 : avg19 = 72)
  (h3 : avg20 = 74)
  (h_totalscore19 : 19 * avg19 = 1368)
  (h_totalscore20 : 20 * avg20 = 1480)
  (h_alexscore : alex = 112) :
  alex = (1480 - 1368 : ℚ) := 
sorry

end alex_score_l159_159966


namespace find_num_students_l159_159526

variables (N T : ℕ)
variables (h1 : T = N * 80)
variables (h2 : 5 * 20 = 100)
variables (h3 : (T - 100) / (N - 5) = 90)

theorem find_num_students (h1 : T = N * 80) (h3 : (T - 100) / (N - 5) = 90) : N = 35 :=
sorry

end find_num_students_l159_159526


namespace average_rst_l159_159142

variable (r s t : ℝ)

theorem average_rst
  (h : (4 / 3) * (r + s + t) = 12) :
  (r + s + t) / 3 = 3 :=
sorry

end average_rst_l159_159142


namespace perimeter_of_wheel_K_l159_159972

theorem perimeter_of_wheel_K
  (L_turns_K : 4 / 5 = 1 / (length_of_K / length_of_L))
  (L_turns_M : 6 / 7 = 1 / (length_of_L / length_of_M))
  (M_perimeter : length_of_M = 30) :
  length_of_K = 28 := 
sorry

end perimeter_of_wheel_K_l159_159972


namespace jason_borrowed_amount_l159_159514

theorem jason_borrowed_amount (hours cycles value_per_cycle remaining_hrs remaining_value total_value: ℕ) : 
  hours = 39 → cycles = (hours / 7) → value_per_cycle = 28 → remaining_hrs = (hours % 7) →
  remaining_value = (1 + 2 + 3 + 4) →
  total_value = (cycles * value_per_cycle + remaining_value) →
  total_value = 150 := 
by {
  sorry
}

end jason_borrowed_amount_l159_159514


namespace determine_ordered_triple_l159_159970

open Real

theorem determine_ordered_triple (a b c : ℝ) (h₁ : 5 < a) (h₂ : 5 < b) (h₃ : 5 < c) 
  (h₄ : (a + 3)^2 / (b + c - 3) + (b + 6)^2 / (c + a - 6) + (c + 9)^2 / (a + b - 9) = 81) : 
  a = 15 ∧ b = 12 ∧ c = 9 := 
sorry

end determine_ordered_triple_l159_159970


namespace regular_polygon_sides_l159_159440

theorem regular_polygon_sides (n : ℕ) (h : ∀ (x : ℕ), x = 180 * (n - 2) / n → x = 144) :
  n = 10 :=
sorry

end regular_polygon_sides_l159_159440


namespace part_a_part_b_part_c_l159_159486

def f (x : ℝ) := x^2
def g (x : ℝ) := 3 * x - 8
def h (r : ℝ) (x : ℝ) := 3 * x - r

theorem part_a :
  f 2 = 4 ∧ g (f 2) = 4 :=
by {
  sorry
}

theorem part_b :
  ∀ x : ℝ, f (g x) = g (f x) → (x = 2 ∨ x = 6) :=
by {
  sorry
}

theorem part_c :
  ∀ r : ℝ, f (h r 2) = h r (f 2) → (r = 3 ∨ r = 8) :=
by {
  sorry
}

end part_a_part_b_part_c_l159_159486


namespace no_valid_pairs_of_real_numbers_l159_159680

theorem no_valid_pairs_of_real_numbers :
  ∀ (a b : ℝ), ¬ (∃ (x y : ℤ), 3 * a * x + 7 * b * y = 3 ∧ x^2 + y^2 = 85 ∧ (x % 5 = 0 ∨ y % 5 = 0)) :=
by
  sorry

end no_valid_pairs_of_real_numbers_l159_159680


namespace no_real_solutions_l159_159809

theorem no_real_solutions (x : ℝ) : (x - 3 * x + 7)^2 + 1 ≠ -|x| :=
by
  -- The statement of the theorem is sufficient; the proof is not needed as per indicated instructions.
  sorry

end no_real_solutions_l159_159809


namespace opposite_of_neg_quarter_l159_159974

theorem opposite_of_neg_quarter : -(- (1/4 : ℝ)) = (1/4 : ℝ) :=
by
  sorry

end opposite_of_neg_quarter_l159_159974


namespace parametric_to_standard_form_l159_159783

theorem parametric_to_standard_form (t : ℝ) (x y : ℝ)
    (param_eq1 : x = 1 + t)
    (param_eq2 : y = -1 + t) :
    x - y - 2 = 0 :=
sorry

end parametric_to_standard_form_l159_159783


namespace hours_per_day_l159_159305

-- Conditions
def days_worked : ℝ := 3
def total_hours_worked : ℝ := 7.5

-- Theorem to prove the number of hours worked each day
theorem hours_per_day : total_hours_worked / days_worked = 2.5 :=
by
  sorry

end hours_per_day_l159_159305


namespace number_of_students_in_class_l159_159697

theorem number_of_students_in_class :
  ∃ a : ℤ, 100 ≤ a ∧ a ≤ 200 ∧ a % 4 = 1 ∧ a % 3 = 2 ∧ a % 7 = 3 ∧ a = 101 := 
sorry

end number_of_students_in_class_l159_159697


namespace polynomial_multiplication_l159_159214

theorem polynomial_multiplication (x y : ℝ) : 
  (2 * x - 3 * y + 1) * (2 * x + 3 * y - 1) = 4 * x^2 - 9 * y^2 + 6 * y - 1 := by
  sorry

end polynomial_multiplication_l159_159214


namespace followers_after_one_year_l159_159711

theorem followers_after_one_year :
  let initial_followers := 100000
  let daily_new_followers := 1000
  let unfollowers_per_year := 20000
  let days_per_year := 365
  initial_followers + (daily_new_followers * days_per_year - unfollowers_per_year) = 445000 :=
by
  sorry

end followers_after_one_year_l159_159711


namespace volume_ratio_of_cubes_l159_159971

theorem volume_ratio_of_cubes 
  (P_A P_B : ℕ) 
  (h_A : P_A = 40) 
  (h_B : P_B = 64) : 
  (∃ s_A s_B V_A V_B, 
    s_A = P_A / 4 ∧ 
    s_B = P_B / 4 ∧ 
    V_A = s_A^3 ∧ 
    V_B = s_B^3 ∧ 
    (V_A : ℚ) / V_B = 125 / 512) := 
by
  sorry

end volume_ratio_of_cubes_l159_159971


namespace max_y_difference_l159_159298

noncomputable def f (x : ℝ) : ℝ := 4 - x^2 + x^3
noncomputable def g (x : ℝ) : ℝ := 2 + x^2 + x^3

theorem max_y_difference : 
  ∃ x1 x2 : ℝ, 
    f x1 = g x1 ∧ f x2 = g x2 ∧ 
    (∀ x : ℝ, f x = g x → x = x1 ∨ x = x2) ∧ 
    abs ((f x1) - (f x2)) = 2 := 
by
  sorry

end max_y_difference_l159_159298


namespace length_ratio_is_correct_width_ratio_is_correct_l159_159301

-- Definitions based on the conditions
def room_length : ℕ := 25
def room_width : ℕ := 15

-- Calculated perimeter
def room_perimeter : ℕ := 2 * (room_length + room_width)

-- Ratios to be proven
def length_to_perimeter_ratio : ℚ := room_length / room_perimeter
def width_to_perimeter_ratio : ℚ := room_width / room_perimeter

-- Stating the theorems to be proved
theorem length_ratio_is_correct : length_to_perimeter_ratio = 5 / 16 :=
by sorry

theorem width_ratio_is_correct : width_to_perimeter_ratio = 3 / 16 :=
by sorry

end length_ratio_is_correct_width_ratio_is_correct_l159_159301


namespace rotation_locus_l159_159161

-- Definitions for points and structure of the cube
structure Point3D :=
(x : ℝ) (y : ℝ) (z : ℝ)

structure Cube :=
(A : Point3D) (B : Point3D) (C : Point3D) (D : Point3D)
(E : Point3D) (F : Point3D) (G : Point3D) (H : Point3D)

-- Function to perform the required rotations and return the locus geometrical representation
noncomputable def locus_points_on_surface (c : Cube) : Set Point3D :=
sorry

-- Mathematical problem rephrased in Lean 4 statement
theorem rotation_locus (c : Cube) :
  locus_points_on_surface c = {c.D, c.A} ∪ {c.A, c.C} ∪ {c.C, c.D} :=
sorry

end rotation_locus_l159_159161


namespace total_students_l159_159143

theorem total_students (ratio_boys_girls : ℕ) (girls : ℕ) (boys : ℕ) (total_students : ℕ)
  (h1 : ratio_boys_girls = 2)     -- The simplified ratio of boys to girls
  (h2 : girls = 200)              -- There are 200 girls
  (h3 : boys = ratio_boys_girls * girls) -- Number of boys is ratio * number of girls
  (h4 : total_students = boys + girls)   -- Total number of students is the sum of boys and girls
  : total_students = 600 :=             -- Prove that the total number of students is 600
sorry

end total_students_l159_159143


namespace Dabbie_spends_99_dollars_l159_159109

noncomputable def total_cost_turkeys (w1 w2 w3 w4 : ℝ) (cost_per_kg : ℝ) : ℝ :=
  (w1 + w2 + w3 + w4) * cost_per_kg

theorem Dabbie_spends_99_dollars :
  let w1 := 6
  let w2 := 9
  let w3 := 2 * w2
  let w4 := (w1 + w2 + w3) / 2
  let cost_per_kg := 2
  total_cost_turkeys w1 w2 w3 w4 cost_per_kg = 99 := 
by
  sorry

end Dabbie_spends_99_dollars_l159_159109


namespace isosceles_triangle_angles_l159_159510

theorem isosceles_triangle_angles (A B C : ℝ)
    (h_iso : A = B ∨ B = C ∨ C = A)
    (h_one_angle : A = 36 ∨ B = 36 ∨ C = 36)
    (h_sum_angles : A + B + C = 180) :
  (A = 36 ∧ B = 36 ∧ C = 108) ∨
  (A = 72 ∧ B = 72 ∧ C = 36) :=
by 
  sorry

end isosceles_triangle_angles_l159_159510


namespace instantaneous_velocity_at_1_l159_159973

noncomputable def h (t : ℝ) : ℝ := -4.9 * t^2 + 6.5 * t + 10

theorem instantaneous_velocity_at_1 :
  (deriv h 1) = -3.3 :=
by
  sorry

end instantaneous_velocity_at_1_l159_159973


namespace estimated_value_at_28_l159_159341

-- Definitions based on the conditions
def regression_equation (x : ℝ) : ℝ := 4.75 * x + 257

-- Problem statement
theorem estimated_value_at_28 : regression_equation 28 = 390 :=
by
  -- Sorry is used to skip the proof
  sorry

end estimated_value_at_28_l159_159341


namespace find_ratio_EG_ES_l159_159647

variables (EF GH EH EG ES QR : ℝ) -- lengths of the segments
variables (x y : ℝ) -- unknowns for parts of the segments
variables (Q R S : Point) -- points

-- Define conditions based on the problem
def parallelogram_EFGH (EF GH EH EG : ℝ) : Prop :=
  ∀ (x y : ℝ), EF = 8 * x ∧ EH = 9 * y

def point_on_segment_Q (Q : Point) (EF EQ : ℝ) : Prop :=
  ∃ x : ℝ, EQ = (1 / 8) * EF

def point_on_segment_R (R : Point) (EH ER : ℝ) : Prop :=
  ∃ y : ℝ, ER = (1 / 9) * EH

def intersection_at_S (EG QR ES : ℝ) : Prop :=
  ∃ x y : ℝ, ES = (1 / 8) * EG + (1 / 9) * EG

theorem find_ratio_EG_ES :
  parallelogram_EFGH EF GH EH EG →
  point_on_segment_Q Q EF (1/8 * EF) →
  point_on_segment_R R EH (1/9 * EH) →
  intersection_at_S EG QR ES →
  EG / ES = 72 / 17 :=
by
  intros h_parallelogram h_pointQ h_pointR h_intersection
  sorry

end find_ratio_EG_ES_l159_159647


namespace roses_cut_l159_159059

def r_before := 13
def r_after := 14

theorem roses_cut : r_after - r_before = 1 := by
  sorry

end roses_cut_l159_159059


namespace fraction_of_board_shaded_is_one_fourth_l159_159560

def totalArea : ℕ := 16
def shadedTopLeft : ℕ := 4
def shadedBottomRight : ℕ := 4
def fractionShaded (totalArea shadedTopLeft shadedBottomRight : ℕ) : ℚ :=
  (shadedTopLeft + shadedBottomRight) / totalArea

theorem fraction_of_board_shaded_is_one_fourth :
  fractionShaded totalArea shadedTopLeft shadedBottomRight = 1 / 4 := by
  sorry

end fraction_of_board_shaded_is_one_fourth_l159_159560


namespace client_dropped_off_phones_l159_159569

def initial_phones : ℕ := 15
def repaired_phones : ℕ := 3
def coworker_phones : ℕ := 9

theorem client_dropped_off_phones (x : ℕ) : 
  initial_phones - repaired_phones + x = 2 * coworker_phones → x = 6 :=
by
  sorry

end client_dropped_off_phones_l159_159569


namespace molly_age_l159_159567

theorem molly_age
  (S M : ℕ)
  (h_ratio : S / M = 4 / 3)
  (h_sandy_future : S + 6 = 42)
  : M = 27 :=
sorry

end molly_age_l159_159567


namespace square_number_increased_decreased_by_five_remains_square_l159_159975

theorem square_number_increased_decreased_by_five_remains_square :
  ∃ x : ℤ, ∃ u v : ℤ, x^2 + 5 = u^2 ∧ x^2 - 5 = v^2 := by
  sorry

end square_number_increased_decreased_by_five_remains_square_l159_159975


namespace find_value_of_expression_l159_159162

theorem find_value_of_expression (a : ℝ) (h : a^2 - a - 1 = 0) : a^3 - a^2 - a + 2023 = 2023 :=
by
  sorry

end find_value_of_expression_l159_159162


namespace ratio_fraction_4A3B_5C2A_l159_159487

def ratio (a b c : ℝ) := a / b = 3 / 2 ∧ b / c = 2 / 6 ∧ a / c = 3 / 6

theorem ratio_fraction_4A3B_5C2A (A B C : ℝ) (h : ratio A B C) : (4 * A + 3 * B) / (5 * C - 2 * A) = 5 / 8 := 
  sorry

end ratio_fraction_4A3B_5C2A_l159_159487


namespace find_third_month_sale_l159_159138

def sales_first_month : ℕ := 3435
def sales_second_month : ℕ := 3927
def sales_fourth_month : ℕ := 4230
def sales_fifth_month : ℕ := 3562
def sales_sixth_month : ℕ := 1991
def required_average_sale : ℕ := 3500

theorem find_third_month_sale (S3 : ℕ) :
  (sales_first_month + sales_second_month + S3 + sales_fourth_month + sales_fifth_month + sales_sixth_month) / 6 = required_average_sale →
  S3 = 3855 := by
  sorry

end find_third_month_sale_l159_159138


namespace triangle_side_ineq_l159_159260

theorem triangle_side_ineq (a b c : ℝ) 
  (h1 : a + b > c) 
  (h2 : b + c > a) 
  (h3 : c + a > b) :
  (a - b) / (a + b) + (b - c) / (b + c) + (c - a) / (a + c) < 1 / 16 :=
  sorry

end triangle_side_ineq_l159_159260


namespace factorize_difference_of_squares_l159_159119

theorem factorize_difference_of_squares (y : ℝ) : y^2 - 4 = (y + 2) * (y - 2) := 
by
  sorry

end factorize_difference_of_squares_l159_159119


namespace find_a_l159_159810

def A := { x : ℝ | x^2 + 4 * x = 0 }
def B (a : ℝ) := { x : ℝ | x^2 + 2 * (a + 1) * x + (a^2 - 1) = 0 }

theorem find_a (a : ℝ) :
  (∀ x : ℝ, x ∈ (A ∩ B a) ↔ x ∈ B a) → (a = 1 ∨ a ≤ -1) :=
by 
  sorry

end find_a_l159_159810


namespace difference_between_numbers_l159_159723

theorem difference_between_numbers (x y d : ℝ) (h1 : x + y = 10) (h2 : x - y = d) (h3 : x^2 - y^2 = 80) : d = 8 :=
by {
  sorry
}

end difference_between_numbers_l159_159723


namespace lilith_caps_collection_l159_159954

noncomputable def monthlyCollectionYear1 := 3
noncomputable def monthlyCollectionAfterYear1 := 5
noncomputable def christmasCaps := 40
noncomputable def yearlyCapsLost := 15
noncomputable def totalYears := 5

noncomputable def totalCapsCollectedByLilith :=
  let firstYearCaps := monthlyCollectionYear1 * 12
  let remainingYearsCaps := monthlyCollectionAfterYear1 * 12 * (totalYears - 1)
  let christmasCapsTotal := christmasCaps * totalYears
  let totalCapsBeforeLosses := firstYearCaps + remainingYearsCaps + christmasCapsTotal
  let lostCapsTotal := yearlyCapsLost * totalYears
  let totalCapsAfterLosses := totalCapsBeforeLosses - lostCapsTotal
  totalCapsAfterLosses

theorem lilith_caps_collection : totalCapsCollectedByLilith = 401 := by
  sorry

end lilith_caps_collection_l159_159954


namespace exists_coprime_integers_divisible_l159_159606

theorem exists_coprime_integers_divisible {a b p : ℤ} : ∃ k l : ℤ, gcd k l = 1 ∧ p ∣ (a * k + b * l) :=
by
  sorry

end exists_coprime_integers_divisible_l159_159606


namespace nat_games_volunteer_allocation_l159_159910

theorem nat_games_volunteer_allocation 
  (volunteers : Fin 6 → Type) 
  (venues : Fin 3 → Type)
  (A B : volunteers 0)
  (remaining : Fin 4 → Type) 
  (assigned_pairings : Π (v : Fin 3), Fin 2 → volunteers 0) :
  (∀ v, assigned_pairings v 0 = A ∨ assigned_pairings v 1 = B) →
  (3 * 6 = 18) := 
by
  sorry

end nat_games_volunteer_allocation_l159_159910


namespace expected_value_linear_combination_l159_159575

variable (ξ η : ℝ)
variable (E : ℝ → ℝ)
axiom E_lin (a b : ℝ) (X Y : ℝ) : E (a * X + b * Y) = a * E X + b * E Y

axiom E_ξ : E ξ = 10
axiom E_η : E η = 3

theorem expected_value_linear_combination : E (3 * ξ + 5 * η) = 45 := by
  sorry

end expected_value_linear_combination_l159_159575


namespace isosceles_right_triangle_area_l159_159840

theorem isosceles_right_triangle_area
  (a b c : ℝ) 
  (h1 : a = b) 
  (h2 : c = a * Real.sqrt 2) 
  (area : ℝ) 
  (h_area : area = 50)
  (h3 : (1/2) * a * b = area) :
  (a + b + c) / area = 0.4 + 0.2 * Real.sqrt 2 :=
by
  sorry

end isosceles_right_triangle_area_l159_159840


namespace total_race_time_l159_159595

theorem total_race_time 
  (num_runners : ℕ) 
  (first_five_time : ℕ) 
  (additional_time : ℕ) 
  (total_runners : ℕ) 
  (num_first_five : ℕ)
  (num_last_three : ℕ) 
  (total_expected_time : ℕ) 
  (h1 : num_runners = 8) 
  (h2 : first_five_time = 8) 
  (h3 : additional_time = 2) 
  (h4 : num_first_five = 5)
  (h5 : num_last_three = num_runners - num_first_five)
  (h6 : total_runners = num_first_five + num_last_three)
  (h7 : 5 * first_five_time + 3 * (first_five_time + additional_time) = total_expected_time)
  : total_expected_time = 70 := 
by
  sorry

end total_race_time_l159_159595


namespace distinct_solutions_eq_four_l159_159328

theorem distinct_solutions_eq_four : ∃! (x : ℝ), abs (x - abs (3 * x + 2)) = 4 :=
by sorry

end distinct_solutions_eq_four_l159_159328


namespace part1_part2_l159_159275

-- Definitions of sets A, B, and C
def setA : Set ℝ := { x | -2 ≤ x ∧ x ≤ 4 }
def setB : Set ℝ := { x | 1 < x ∧ x < 5 }
def setC (a : ℝ) : Set ℝ := { x | a - 1 < x ∧ x < 2 * a + 3 }

-- part (1)
theorem part1 (x : ℝ) : (x ∈ setA ∨ x ∈ setB) ↔ (-2 ≤ x ∧ x < 5) :=
sorry

-- part (2)
theorem part2 (a : ℝ) : ((setA ∩ setC a) = setC a) ↔ (a ≤ -4 ∨ (-1 ≤ a ∧ a ≤ 1/2)) :=
sorry

end part1_part2_l159_159275


namespace number_of_draw_matches_eq_points_difference_l159_159071

-- Definitions based on the conditions provided
def teams : ℕ := 16
def matches_per_round : ℕ := 8
def rounds : ℕ := 16
def total_points : ℕ := 222
def total_matches : ℕ := matches_per_round * rounds
def hypothetical_points : ℕ := total_matches * 2
def points_difference : ℕ := hypothetical_points - total_points

-- Theorem stating the equivalence to be proved
theorem number_of_draw_matches_eq_points_difference : 
  points_difference = 34 := 
by
  sorry

end number_of_draw_matches_eq_points_difference_l159_159071


namespace smallest_integer_satisfies_inequality_l159_159830

theorem smallest_integer_satisfies_inequality :
  ∃ (x : ℤ), (x^2 < 2 * x + 3) ∧ ∀ (y : ℤ), (y^2 < 2 * y + 3) → x ≤ y ∧ x = 0 :=
sorry

end smallest_integer_satisfies_inequality_l159_159830


namespace number_of_diagonals_25_sides_l159_159629

theorem number_of_diagonals_25_sides (n : ℕ) (h : n = 25) : 
    (n * (n - 3)) / 2 = 275 := by
  sorry

end number_of_diagonals_25_sides_l159_159629


namespace wendy_total_sales_l159_159353

noncomputable def apple_price : ℝ := 1.50
noncomputable def orange_price : ℝ := 1.00
noncomputable def morning_apples : ℕ := 40
noncomputable def morning_oranges : ℕ := 30
noncomputable def afternoon_apples : ℕ := 50
noncomputable def afternoon_oranges : ℕ := 40

theorem wendy_total_sales :
  (morning_apples * apple_price + morning_oranges * orange_price) +
  (afternoon_apples * apple_price + afternoon_oranges * orange_price) = 205 := by
  sorry

end wendy_total_sales_l159_159353


namespace line_segment_value_of_x_l159_159376

theorem line_segment_value_of_x (x : ℝ) (h1 : (1 - 4)^2 + (3 - x)^2 = 25) (h2 : x > 0) : x = 7 :=
sorry

end line_segment_value_of_x_l159_159376


namespace circle_intersection_value_l159_159591

theorem circle_intersection_value {x1 y1 x2 y2 : ℝ} 
  (h_circle : x1^2 + y1^2 = 4)
  (h_non_negative : x1 ≥ 0 ∧ y1 ≥ 0 ∧ x2 ≥ 0 ∧ y2 ≥ 0)
  (h_symmetric : x1 = y2 ∧ x2 = y1) :
  x1^2 + x2^2 = 4 := 
by
  sorry

end circle_intersection_value_l159_159591


namespace count_multiples_less_than_300_l159_159404

theorem count_multiples_less_than_300 : ∀ n : ℕ, n < 300 → (2 * 3 * 5 * 7 ∣ n) ↔ n = 210 :=
by
  sorry

end count_multiples_less_than_300_l159_159404


namespace sum_real_imaginary_part_l159_159288

noncomputable def imaginary_unit : ℂ := Complex.I

theorem sum_real_imaginary_part {z : ℂ} (h : z * imaginary_unit = 1 + imaginary_unit) :
  z.re + z.im = 2 := 
sorry

end sum_real_imaginary_part_l159_159288


namespace eleven_power_five_mod_nine_l159_159291

theorem eleven_power_five_mod_nine : ∃ n : ℕ, (11^5 ≡ n [MOD 9]) ∧ (0 ≤ n ∧ n < 9) ∧ (n = 5) := 
  by 
    sorry

end eleven_power_five_mod_nine_l159_159291


namespace intimate_interval_proof_l159_159399

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 - 3 * x + 4
def g (x : ℝ) : ℝ := 2 * x - 3

-- Define the concept of intimate functions over an interval
def are_intimate_functions (a b : ℝ) : Prop :=
  ∀ x, a ≤ x ∧ x ≤ b → |f x - g x| ≤ 1

-- Prove that the interval [2, 3] is a subset of [a, b]
theorem intimate_interval_proof (a b : ℝ) (h : are_intimate_functions a b) :
  2 ≤ b ∧ a ≤ 3 :=
sorry

end intimate_interval_proof_l159_159399


namespace base_b_equivalence_l159_159695

theorem base_b_equivalence (b : ℕ) (h : (2 * b + 4) ^ 2 = 5 * b ^ 2 + 5 * b + 4) : b = 12 :=
sorry

end base_b_equivalence_l159_159695


namespace nylon_cord_length_l159_159402

-- Let the length of cord be w
-- Dog runs 30 feet forming a semicircle, that is pi * w = 30
-- Prove that w is approximately 9.55

theorem nylon_cord_length (pi_approx : Real := 3.14) : Real :=
  let w := 30 / pi_approx
  w

end nylon_cord_length_l159_159402


namespace winnie_servings_l159_159455

theorem winnie_servings:
  ∀ (x : ℝ), 
  (2 / 5) * x + (21 / 25) * x = 82 →
  x = 30 :=
by
  sorry

end winnie_servings_l159_159455


namespace series_sum_eq_l159_159779

theorem series_sum_eq : 
  (∑' n, (4 * n + 3) / ((4 * n - 2) ^ 2 * (4 * n + 2) ^ 2)) = 1 / 128 := by
sorry

end series_sum_eq_l159_159779


namespace geo_series_sum_l159_159642

theorem geo_series_sum (a r : ℚ) (n: ℕ) (ha : a = 1/3) (hr : r = 1/2) (hn : n = 8) : 
    (a * (1 - r^n) / (1 - r)) = 85 / 128 := 
by
  sorry

end geo_series_sum_l159_159642


namespace quadratic_roots_proof_l159_159392

noncomputable def quadratic_roots_statement : Prop :=
  ∃ (x1 x2 : ℝ), 
    (x1 ≠ x2 ∨ x1 = x2) ∧ 
    (x1 = -20 ∧ x2 = -20) ∧ 
    (x1^2 + 40 * x1 + 300 = -100) ∧ 
    (x1 - x2 = 0 ∧ x1 * x2 = 400)  

theorem quadratic_roots_proof : quadratic_roots_statement :=
sorry

end quadratic_roots_proof_l159_159392


namespace ratio_of_steps_l159_159701

-- Defining the conditions of the problem
def andrew_steps : ℕ := 150
def jeffrey_steps : ℕ := 200

-- Stating the theorem that we need to prove
theorem ratio_of_steps : andrew_steps / Nat.gcd andrew_steps jeffrey_steps = 3 ∧ jeffrey_steps / Nat.gcd andrew_steps jeffrey_steps = 4 :=
by
  -- Placeholder for the proof
  sorry

end ratio_of_steps_l159_159701


namespace profit_distribution_l159_159662

noncomputable def profit_sharing (investment_a investment_d profit: ℝ) : ℝ × ℝ :=
  let total_investment := investment_a + investment_d
  let share_a := investment_a / total_investment
  let share_d := investment_d / total_investment
  (share_a * profit, share_d * profit)

theorem profit_distribution :
  let investment_a := 22500
  let investment_d := 35000
  let first_period_profit := 9600
  let second_period_profit := 12800
  let third_period_profit := 18000
  profit_sharing investment_a investment_d first_period_profit = (3600, 6000) ∧
  profit_sharing investment_a investment_d second_period_profit = (5040, 7760) ∧
  profit_sharing investment_a investment_d third_period_profit = (7040, 10960) :=
sorry

end profit_distribution_l159_159662


namespace proof_f_derivative_neg1_l159_159116

noncomputable def f (x : ℝ) (a b c : ℝ) : ℝ :=
  a * x ^ 4 + b * x ^ 2 + c

noncomputable def f_derivative (x : ℝ) (a b : ℝ) : ℝ :=
  4 * a * x ^ 3 + 2 * b * x

theorem proof_f_derivative_neg1
  (a b c : ℝ) (h : f_derivative 1 a b = 2) :
  f_derivative (-1) a b = -2 :=
by
  sorry

end proof_f_derivative_neg1_l159_159116


namespace tank_capacity_l159_159057

theorem tank_capacity (T : ℝ) (h1 : 0.6 * T = 0.7 * T - 45) : T = 450 :=
by
  sorry

end tank_capacity_l159_159057


namespace tangent_line_circle_l159_159114

theorem tangent_line_circle (r : ℝ) (h : 0 < r) :
  (∀ x y : ℝ, x + y = r → x * x + y * y ≠ 4 * r) →
  r = 8 :=
by
  sorry

end tangent_line_circle_l159_159114


namespace crates_sold_on_monday_l159_159495

variable (M : ℕ)
variable (h : M + 2 * M + (2 * M - 2) + M = 28)

theorem crates_sold_on_monday : M = 5 :=
by
  sorry

end crates_sold_on_monday_l159_159495


namespace symmetric_point_min_value_l159_159774

theorem symmetric_point_min_value (a b : ℝ) 
  (h1 : a > 0 ∧ b > 0) 
  (h2 : ∃ (x₀ y₀ : ℝ), x₀ + y₀ - 2 = 0 ∧ 2 * x₀ + y₀ + 3 = 0 ∧ 
        a + b = x₀ + y₀ ∧ ∃ k, k = (y₀ - b) / (x₀ - a) ∧ y₀ = k * x₀ + 2 - k * (a + k * b))
   : ∃ α β, a = β / α ∧  b = 2 * β / α ∧ (1 / a + 8 / b) = 25 / 9 :=
sorry

end symmetric_point_min_value_l159_159774


namespace cosine_60_degrees_l159_159589

theorem cosine_60_degrees : Real.cos (60 * Real.pi / 180) = 1 / 2 :=
by sorry

end cosine_60_degrees_l159_159589


namespace line_parallel_l159_159160

theorem line_parallel (x y : ℝ) :
  ∃ m b : ℝ, 
    y = m * (x - 2) + (-4) ∧ 
    m = 2 ∧ 
    (∀ (x y : ℝ), y = 2 * x - 8 → 2 * x - y - 8 = 0) :=
sorry

end line_parallel_l159_159160


namespace sufficient_but_not_necessary_l159_159650

theorem sufficient_but_not_necessary (p q : Prop) :
  (¬ (p ∨ q) → ¬ (p ∧ q)) ∧ (¬ (p ∧ q) → p ∨ q → False) :=
by
  sorry

end sufficient_but_not_necessary_l159_159650


namespace sqrt_sum_of_four_terms_of_4_pow_4_l159_159764

-- Proof Statement
theorem sqrt_sum_of_four_terms_of_4_pow_4 : 
  Real.sqrt (4 ^ 4 + 4 ^ 4 + 4 ^ 4 + 4 ^ 4) = 32 := 
by 
  sorry

end sqrt_sum_of_four_terms_of_4_pow_4_l159_159764


namespace q_is_necessary_but_not_sufficient_for_p_l159_159961

theorem q_is_necessary_but_not_sufficient_for_p (a : ℝ) : 
  (∀ x : ℝ, (x^2 + 2*x + 1 - a^2 < 0) → (-1 + a < x ∧ x < -1 - a)) → (a < 1) ∧ (¬ (a < 1 → (∀ x : ℝ, (x^2 + 2*x + 1 - a^2 < 0) → (-1 + a < x ∧ x < -1 - a)))) :=
by
  sorry

end q_is_necessary_but_not_sufficient_for_p_l159_159961


namespace deepak_profit_share_l159_159682

theorem deepak_profit_share (anand_investment : ℕ) (deepak_investment : ℕ) (total_profit : ℕ) 
  (h₁ : anand_investment = 22500) 
  (h₂ : deepak_investment = 35000) 
  (h₃ : total_profit = 13800) : 
  (14 * total_profit / (9 + 14)) = 8400 := 
by
  sorry

end deepak_profit_share_l159_159682


namespace age_problem_l159_159863

theorem age_problem (my_age mother_age : ℕ) 
  (h1 : mother_age = 3 * my_age) 
  (h2 : my_age + mother_age = 40)
  : my_age = 10 :=
by 
  sorry

end age_problem_l159_159863


namespace triangular_array_sum_digits_l159_159766

theorem triangular_array_sum_digits (N : ℕ) (h : N * (N + 1) / 2 = 2145) : (N / 10 + N % 10) = 11 := 
sorry

end triangular_array_sum_digits_l159_159766


namespace line_tangent_to_circle_perpendicular_l159_159952

theorem line_tangent_to_circle_perpendicular 
  (l₁ l₂ : String)
  (C : String)
  (h1 : l₂ = "4 * x - 3 * y + 1 = 0")
  (h2 : C = "x^2 + y^2 + 2 * y - 3 = 0") :
  (l₁ = "3 * x + 4 * y + 14 = 0" ∨ l₁ = "3 * x + 4 * y - 6 = 0") :=
by
  sorry

end line_tangent_to_circle_perpendicular_l159_159952


namespace jesse_mia_total_miles_per_week_l159_159395

noncomputable def jesse_miles_per_day_first_three := 2 / 3
noncomputable def jesse_miles_day_four := 10
noncomputable def mia_miles_per_day_first_four := 3
noncomputable def average_final_three_days := 6

theorem jesse_mia_total_miles_per_week :
  let jesse_total_first_four_days := 3 * jesse_miles_per_day_first_three + jesse_miles_day_four
  let mia_total_first_four_days := 4 * mia_miles_per_day_first_four
  let total_miles_needed_final_three_days := 3 * average_final_three_days * 2
  jesse_total_first_four_days + total_miles_needed_final_three_days = 48 ∧
  mia_total_first_four_days + total_miles_needed_final_three_days = 48 :=
by
  sorry

end jesse_mia_total_miles_per_week_l159_159395


namespace simplify_expression_l159_159848
open Real

theorem simplify_expression (x y : ℝ) : -x + y - 2 * x - 3 * y = -3 * x - 2 * y :=
by
  sorry

end simplify_expression_l159_159848


namespace ferris_wheel_cost_per_child_l159_159492

namespace AmusementPark

def num_children := 5
def daring_children := 3
def merry_go_round_cost_per_child := 3
def ice_cream_cones_per_child := 2
def ice_cream_cost_per_cone := 8
def total_spent := 110

theorem ferris_wheel_cost_per_child (F : ℝ) :
  (daring_children * F + num_children * merry_go_round_cost_per_child +
   num_children * ice_cream_cones_per_child * ice_cream_cost_per_cone = total_spent) →
  F = 5 :=
by
  -- Here we would proceed with the proof steps, but adding sorry to skip it.
  sorry

end AmusementPark

end ferris_wheel_cost_per_child_l159_159492


namespace necessary_but_not_sufficient_condition_for_purely_imaginary_l159_159997

theorem necessary_but_not_sufficient_condition_for_purely_imaginary (m : ℂ) :
  (1 - m^2 + (1 + m) * Complex.I = 0 → m = 1) ∧ 
  ((1 - m^2 + (1 + m) * Complex.I = 0 ↔ m = 1) = false) := by
  sorry

end necessary_but_not_sufficient_condition_for_purely_imaginary_l159_159997


namespace exists_permutation_ab_minus_cd_ge_two_l159_159184

theorem exists_permutation_ab_minus_cd_ge_two (p q r s : ℝ) 
  (h1 : p + q + r + s = 9) 
  (h2 : p^2 + q^2 + r^2 + s^2 = 21) :
  ∃ (a b c d : ℝ), (a, b, c, d) = (p, q, r, s) ∨ (a, b, c, d) = (p, q, s, r) ∨ 
  (a, b, c, d) = (p, r, q, s) ∨ (a, b, c, d) = (p, r, s, q) ∨ 
  (a, b, c, d) = (p, s, q, r) ∨ (a, b, c, d) = (p, s, r, q) ∨ 
  (a, b, c, d) = (q, p, r, s) ∨ (a, b, c, d) = (q, p, s, r) ∨ 
  (a, b, c, d) = (q, r, p, s) ∨ (a, b, c, d) = (q, r, s, p) ∨ 
  (a, b, c, d) = (q, s, p, r) ∨ (a, b, c, d) = (q, s, r, p) ∨ 
  (a, b, c, d) = (r, p, q, s) ∨ (a, b, c, d) = (r, p, s, q) ∨ 
  (a, b, c, d) = (r, q, p, s) ∨ (a, b, c, d) = (r, q, s, p) ∨ 
  (a, b, c, d) = (r, s, p, q) ∨ (a, b, c, d) = (r, s, q, p) ∨ 
  (a, b, c, d) = (s, p, q, r) ∨ (a, b, c, d) = (s, p, r, q) ∨ 
  (a, b, c, d) = (s, q, p, r) ∨ (a, b, c, d) = (s, q, r, p) ∨ 
  (a, b, c, d) = (s, r, p, q) ∨ (a, b, c, d) = (s, r, q, p) ∧ ab - cd ≥ 2 :=
sorry

end exists_permutation_ab_minus_cd_ge_two_l159_159184


namespace square_area_inside_ellipse_l159_159666

theorem square_area_inside_ellipse :
  (∃ s : ℝ, 
    ∀ (x y : ℝ), 
      (x = s ∧ y = s) → 
      (x^2 / 4 + y^2 / 8 = 1) ∧ 
      (4 * (s^2 / 3) = 1) ∧ 
      (area = 4 * (8 / 3))) →
    ∃ area : ℝ, 
      area = 32 / 3 :=
by
  sorry

end square_area_inside_ellipse_l159_159666


namespace cylinder_lateral_surface_area_l159_159754

theorem cylinder_lateral_surface_area :
  let side := 20
  let radius := side / 2
  let height := side
  2 * Real.pi * radius * height = 400 * Real.pi :=
by
  let side := 20
  let radius := side / 2
  let height := side
  sorry

end cylinder_lateral_surface_area_l159_159754


namespace power_of_two_l159_159294

theorem power_of_two (b m n : ℕ) (hb : b > 1) (hmn : m ≠ n) 
  (hprime_divisors : ∀ p : ℕ, p.Prime → (p ∣ b ^ m - 1 ↔ p ∣ b ^ n - 1)) : 
  ∃ k : ℕ, b + 1 = 2 ^ k :=
by
  sorry

end power_of_two_l159_159294


namespace evaluate_expression_l159_159007

-- Define the expression and the expected result
def expression := -(14 / 2 * 9 - 60 + 3 * 9)
def expectedResult := -30

-- The theorem that states the equivalence
theorem evaluate_expression : expression = expectedResult := by
  sorry

end evaluate_expression_l159_159007


namespace pyramid_sphere_proof_l159_159670

theorem pyramid_sphere_proof
  (h R_1 R_2 : ℝ) 
  (O_1 O_2 T_1 T_2 : ℝ) 
  (inscription: h > 0 ∧ R_1 > 0 ∧ R_2 > 0) :
  R_1 * R_2 * h^2 = (R_1^2 - O_1 * T_1^2) * (R_2^2 - O_2 * T_2^2) :=
by
  sorry

end pyramid_sphere_proof_l159_159670


namespace solve_P_Q_l159_159054

theorem solve_P_Q :
  ∃ P Q : ℝ, (∀ x : ℝ, x ≠ -6 ∧ x ≠ 0 ∧ x ≠ 5 →
    (P / (x + 6) + Q / (x * (x - 5)) = (x^2 - 3*x + 15) / (x * (x + 6) * (x - 5)))) ∧
    P = 1 ∧ Q = 5/2 :=
by
  sorry

end solve_P_Q_l159_159054


namespace base_eight_to_base_ten_642_l159_159223

theorem base_eight_to_base_ten_642 :
  let d0 := 2
  let d1 := 4
  let d2 := 6
  let base := 8
  d0 * base^0 + d1 * base^1 + d2 * base^2 = 418 := 
by
  sorry

end base_eight_to_base_ten_642_l159_159223


namespace correct_calculation_l159_159139

theorem correct_calculation : (Real.sqrt 3) ^ 2 = 3 := by
  sorry

end correct_calculation_l159_159139


namespace opposite_of_neg2016_l159_159078

theorem opposite_of_neg2016 : -(-2016) = 2016 := 
by 
  sorry

end opposite_of_neg2016_l159_159078


namespace Walter_age_in_2010_l159_159098

-- Define Walter's age in 2005 as y
def Walter_age_2005 (y : ℕ) : Prop :=
  (2005 - y) + (2005 - 3 * y) = 3858

-- Define Walter's age in 2010
theorem Walter_age_in_2010 (y : ℕ) (hy : Walter_age_2005 y) : y + 5 = 43 :=
by
  sorry

end Walter_age_in_2010_l159_159098


namespace joes_current_weight_l159_159239

theorem joes_current_weight (W : ℕ) (R : ℕ) : 
  (W = 222 - 4 * R) →
  (W - 3 * R = 180) →
  W = 198 :=
by
  intros h1 h2
  -- Skip the proof for now
  sorry

end joes_current_weight_l159_159239


namespace find_ab_l159_159112

def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 - 2 * a * x + 2 + b

theorem find_ab (a b : ℝ) (h₀ : a ≠ 0) (h₁ : f a b 2 = 2) (h₂ : f a b 3 = 5) :
    (a = 1 ∧ b = 0) ∨ (a = -1 ∧ b = 3) :=
by 
  sorry

end find_ab_l159_159112


namespace intersection_A_B_l159_159882

-- Defining the sets A and B
def A : Set ℝ := {x | x^2 - 3*x + 2 < 0}
def B : Set ℝ := {x | 3 - x > 0}

-- Stating the theorem that A ∩ B equals (1, 2)
theorem intersection_A_B : A ∩ B = {x | 1 < x ∧ x < 2} :=
by
  sorry

end intersection_A_B_l159_159882


namespace jason_nickels_is_52_l159_159782

theorem jason_nickels_is_52 (n q : ℕ) (h1 : 5 * n + 10 * q = 680) (h2 : q = n - 10) : n = 52 :=
sorry

end jason_nickels_is_52_l159_159782


namespace average_age_of_school_l159_159165

theorem average_age_of_school 
  (total_students : ℕ)
  (average_age_boys : ℕ)
  (average_age_girls : ℕ)
  (number_of_girls : ℕ)
  (number_of_boys : ℕ := total_students - number_of_girls)
  (total_age_boys : ℕ := average_age_boys * number_of_boys)
  (total_age_girls : ℕ := average_age_girls * number_of_girls)
  (total_age_students : ℕ := total_age_boys + total_age_girls) :
  total_students = 640 →
  average_age_boys = 12 →
  average_age_girls = 11 →
  number_of_girls = 160 →
  (total_age_students : ℝ) / (total_students : ℝ) = 11.75 :=
by
  intros h1 h2 h3 h4
  sorry

end average_age_of_school_l159_159165


namespace chess_tournament_num_players_l159_159474

theorem chess_tournament_num_players (n : ℕ) :
  (∀ k, k ≠ n → exists m, m ≠ n ∧ (k = m)) ∧ 
  ((1 / 2 * (n - 1)) + (1 / 4 * (n - 1))) = (1 / 13 * ((1 / 2 * n * (n - 1)) - ((1 / 2 * (n - 1)) + (1 / 4 * (n - 1))))) →
  n = 21 :=
by
  sorry

end chess_tournament_num_players_l159_159474


namespace violet_has_27_nails_l159_159124

def nails_tickletoe : ℕ := 12  -- T
def nails_violet : ℕ := 2 * nails_tickletoe + 3

theorem violet_has_27_nails (h : nails_tickletoe + nails_violet = 39) : nails_violet = 27 :=
by
  sorry

end violet_has_27_nails_l159_159124


namespace min_y_value_l159_159052

noncomputable def min_value_y : ℝ :=
  18 - 2 * Real.sqrt 106

theorem min_y_value (x y : ℝ) (h : x^2 + y^2 = 20 * x + 36 * y) : 
  y >= 18 - 2 * Real.sqrt 106 :=
sorry

end min_y_value_l159_159052


namespace upstream_distance_l159_159833

variable (Vb Vs Vdown Vup Dup : ℕ)

def boatInStillWater := Vb = 36
def speedStream := Vs = 12
def downstreamSpeed := Vdown = Vb + Vs
def upstreamSpeed := Vup = Vb - Vs
def timeEquality := 80 / Vdown = Dup / Vup

theorem upstream_distance (Vb Vs Vdown Vup Dup : ℕ) 
  (h1 : boatInStillWater Vb)
  (h2 : speedStream Vs)
  (h3 : downstreamSpeed Vb Vs Vdown)
  (h4 : upstreamSpeed Vb Vs Vup)
  (h5 : timeEquality Vdown Vup Dup) : Dup = 40 := 
sorry

end upstream_distance_l159_159833


namespace greatest_x_integer_l159_159593

theorem greatest_x_integer (x : ℤ) (h : ∃ n : ℤ, x^2 + 2 * x + 7 = (x - 4) * n) : x ≤ 35 :=
sorry

end greatest_x_integer_l159_159593


namespace sum_of_dimensions_eq_18_sqrt_1_5_l159_159692

theorem sum_of_dimensions_eq_18_sqrt_1_5 (P Q R : ℝ) (h1 : P * Q = 30) (h2 : P * R = 50) (h3 : Q * R = 90) :
  P + Q + R = 18 * Real.sqrt 1.5 :=
sorry

end sum_of_dimensions_eq_18_sqrt_1_5_l159_159692


namespace distinguishable_octahedrons_l159_159369

noncomputable def number_of_distinguishable_octahedrons (total_colors : ℕ) (used_colors : ℕ) : ℕ :=
  let num_ways_choose_colors := Nat.choose total_colors (used_colors - 1)
  let num_permutations := (used_colors - 1).factorial
  let num_rotations := 3
  (num_ways_choose_colors * num_permutations) / num_rotations

theorem distinguishable_octahedrons (h : number_of_distinguishable_octahedrons 9 8 = 13440) : true := sorry

end distinguishable_octahedrons_l159_159369


namespace smallest_value_of_3a_plus_2_l159_159436

theorem smallest_value_of_3a_plus_2 (a : ℝ) (h : 8 * a^2 + 10 * a + 6 = 2) : 
  ∃ (x : ℝ), x = 3 * a + 2 ∧ x = -1 :=
by
  sorry

end smallest_value_of_3a_plus_2_l159_159436


namespace triangular_number_30_l159_159885

theorem triangular_number_30 : (30 * (30 + 1)) / 2 = 465 :=
by
  sorry

end triangular_number_30_l159_159885


namespace laborers_employed_l159_159190

theorem laborers_employed 
    (H L : ℕ) 
    (h1 : H + L = 35) 
    (h2 : 140 * H + 90 * L = 3950) : 
    L = 19 :=
by
  sorry

end laborers_employed_l159_159190


namespace class_percentage_of_girls_l159_159730

/-
Given:
- Initial number of boys in the class: 11
- Number of girls in the class: 13
- 1 boy is added to the class, resulting in the new total number of boys being 12

Prove:
- The percentage of the class that are girls is 52%.
-/
theorem class_percentage_of_girls (initial_boys : ℕ) (girls : ℕ) (added_boy : ℕ)
  (new_boy_total : ℕ) (total_students : ℕ) (percent_girls : ℕ) (h1 : initial_boys = 11) 
  (h2 : girls = 13) (h3 : added_boy = 1) (h4 : new_boy_total = initial_boys + added_boy) 
  (h5 : total_students = new_boy_total + girls) 
  (h6 : percent_girls = (girls * 100) / total_students) : percent_girls = 52 :=
sorry

end class_percentage_of_girls_l159_159730


namespace mangoes_harvested_l159_159318

theorem mangoes_harvested (neighbors : ℕ) (mangoes_per_neighbor : ℕ) (total_mangoes_distributed : ℕ) (total_mangoes : ℕ) :
  neighbors = 8 ∧ mangoes_per_neighbor = 35 ∧ total_mangoes_distributed = neighbors * mangoes_per_neighbor ∧ total_mangoes = 2 * total_mangoes_distributed →
  total_mangoes = 560 :=
by {
  sorry
}

end mangoes_harvested_l159_159318


namespace steve_take_home_pay_l159_159196

def annual_salary : ℝ := 40000
def tax_rate : ℝ := 0.20
def healthcare_rate : ℝ := 0.10
def union_dues : ℝ := 800

theorem steve_take_home_pay : 
  (annual_salary - (annual_salary * tax_rate + annual_salary * healthcare_rate + union_dues)) = 27200 := 
by 
  sorry

end steve_take_home_pay_l159_159196


namespace files_per_folder_l159_159822

-- Define the conditions
def initial_files : ℕ := 43
def deleted_files : ℕ := 31
def num_folders : ℕ := 2

-- Define the final problem statement
theorem files_per_folder :
  (initial_files - deleted_files) / num_folders = 6 :=
by
  -- proof would go here
  sorry

end files_per_folder_l159_159822


namespace x_is_perfect_square_l159_159121

theorem x_is_perfect_square {x y : ℕ} (hx : x > 0) (hy : y > 0) (h : (x^2 + y^2 - x) % (2 * x * y) = 0) : ∃ z : ℕ, x = z^2 :=
by
  -- The proof will proceed here
  sorry

end x_is_perfect_square_l159_159121


namespace union_of_A_and_B_l159_159475

-- Define the sets A and B as given in the problem
def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {2, 4}

-- State the theorem to prove that A ∪ B = {0, 1, 2, 4}
theorem union_of_A_and_B : A ∪ B = {0, 1, 2, 4} := by
  sorry

end union_of_A_and_B_l159_159475


namespace quadrilateral_is_square_l159_159303

-- Define a structure for a quadrilateral with side lengths and diagonal lengths
structure Quadrilateral :=
  (side_a side_b side_c side_d diag_e diag_f : ℝ)

-- Define what it means for a quadrilateral to be a square
def is_square (quad : Quadrilateral) : Prop :=
  quad.side_a = quad.side_b ∧ 
  quad.side_b = quad.side_c ∧ 
  quad.side_c = quad.side_d ∧  
  quad.diag_e = quad.diag_f

-- Define the problem to prove that the given quadrilateral is a square given the conditions
theorem quadrilateral_is_square (quad : Quadrilateral) 
  (h_sides : quad.side_a = quad.side_b ∧ 
             quad.side_b = quad.side_c ∧ 
             quad.side_c = quad.side_d)
  (h_diagonals : quad.diag_e = quad.diag_f) :
  is_square quad := 
  by
  -- This is where the proof would go
  sorry

end quadrilateral_is_square_l159_159303


namespace scientific_notation_of_15510000_l159_159044

/--
Express 15,510,000 in scientific notation.

Theorem: 
Given that the scientific notation for large numbers is of the form \(a \times 10^n\) where \(1 \leq |a| < 10\),
prove that expressing 15,510,000 in scientific notation results in 1.551 × 10^7.
-/
theorem scientific_notation_of_15510000 : ∃ (a : ℝ) (n : ℤ), 1 ≤ |a| ∧ |a| < 10 ∧ 15510000 = a * 10 ^ n ∧ a = 1.551 ∧ n = 7 :=
by
  sorry

end scientific_notation_of_15510000_l159_159044


namespace points_on_same_sphere_l159_159598

-- Define the necessary structures and assumptions
variables {P : Type*} [MetricSpace P]

-- Definitions of spheres and points
structure Sphere (P : Type*) [MetricSpace P] :=
(center : P)
(radius : ℝ)
(positive_radius : 0 < radius)

def symmetric_point (S A1 : P) : P := sorry -- definition to get the symmetric point A2

-- Given conditions
variables (S A B C A1 B1 C1 A2 B2 C2 : P)
variable (omega : Sphere P)
variable (Omega : Sphere P)
variable (M_S_A : P) -- midpoint of SA
variable (M_S_B : P) -- midpoint of SB
variable (M_S_C : P) -- midpoint of SC

-- Assertions of conditions
axiom sphere_through_vertex : omega.center = S
axiom first_intersections : omega.radius = dist S A1 ∧ omega.radius = dist S B1 ∧ omega.radius = dist S C1
axiom omega_Omega_intersection : ∃ (circle_center : P) (plane_parallel_to_ABC : P), true-- some conditions indicating intersection
axiom symmetric_points_A1_A2 : A2 = symmetric_point S A1
axiom symmetric_points_B1_B2 : B2 = symmetric_point S B1
axiom symmetric_points_C1_C2 : C2 = symmetric_point S C1

-- The theorem to prove
theorem points_on_same_sphere : ∃ (sphere : Sphere P), 
  (dist sphere.center A) = sphere.radius ∧ 
  (dist sphere.center B) = sphere.radius ∧ 
  (dist sphere.center C) = sphere.radius ∧ 
  (dist sphere.center A2) = sphere.radius ∧ 
  (dist sphere.center B2) = sphere.radius ∧ 
  (dist sphere.center C2) = sphere.radius := 
sorry

end points_on_same_sphere_l159_159598


namespace cricket_team_average_age_difference_l159_159638

theorem cricket_team_average_age_difference :
  let team_size := 11
  let captain_age := 26
  let keeper_age := captain_age + 3
  let avg_whole_team := 23
  let total_team_age := avg_whole_team * team_size
  let combined_age := captain_age + keeper_age
  let remaining_players := team_size - 2
  let total_remaining_age := total_team_age - combined_age
  let avg_remaining_players := total_remaining_age / remaining_players
  avg_whole_team - avg_remaining_players = 1 :=
by
  -- Proof omitted
  sorry

end cricket_team_average_age_difference_l159_159638


namespace john_horizontal_distance_l159_159895

theorem john_horizontal_distance
  (vertical_distance_ratio horizontal_distance_ratio : ℕ)
  (initial_elevation final_elevation : ℕ)
  (h_ratio : vertical_distance_ratio = 1)
  (h_dist_ratio : horizontal_distance_ratio = 3)
  (h_initial : initial_elevation = 500)
  (h_final : final_elevation = 3450) :
  (final_elevation - initial_elevation) * horizontal_distance_ratio = 8850 := 
by {
  sorry
}

end john_horizontal_distance_l159_159895


namespace equation_of_ellipse_equation_of_line_AB_l159_159489

-- Step 1: Given conditions for the ellipse and related hyperbola.
def condition_eccentricity (a b c : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ c / a = Real.sqrt 2 / 2

def condition_distance_focus_asymptote (c : ℝ) : Prop :=
  abs c / Real.sqrt (1 + 2) = Real.sqrt 3 / 3

-- Step 2: Given conditions for the line AB.
def condition_line_A_B (k m : ℝ) : Prop :=
  k < 0 ∧ m^2 = 4 / 5 * (1 + k^2) ∧
  ∃ (x1 x2 y1 y2 : ℝ), 
  (1 + 2 * k^2) * x1^2 + 4 * k * m * x1 + 2 * m^2 - 2 = 0 ∧ 
  (1 + 2 * k^2) * x2^2 + 4 * k * m * x2 + 2 * m^2 - 2 = 0 ∧
  x1 + x2 = -4 * k * m / (1 + 2*k^2) ∧ 
  x1 * x2 = (2 * m^2 - 2) / (1 + 2*k^2)

def condition_circle_passes_F2 (x1 x2 k m : ℝ) : Prop :=
  (1 + k^2) * x1 * x2 + (k * m - 1) * (x1 + x2) + m^2 + 1 = 0

noncomputable def problem_data : Prop :=
  ∃ (a b c k m x1 x2 : ℝ),
    condition_eccentricity a b c ∧
    condition_distance_focus_asymptote c ∧
    condition_line_A_B k m ∧
    condition_circle_passes_F2 x1 x2 k m

-- Step 3: Statements to be proven.
theorem equation_of_ellipse : problem_data → 
  ∃ (a b : ℝ), a = Real.sqrt 2 ∧ b = 1 ∧ ∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1) ↔ (x^2 / 2 + y^2 = 1) :=
by sorry

theorem equation_of_line_AB : problem_data → 
  ∃ (k m : ℝ), m = 1 ∧ k = -1/2 ∧ ∀ x y : ℝ, (y = k * x + m) ↔ (y = -0.5 * x + 1) :=
by sorry

end equation_of_ellipse_equation_of_line_AB_l159_159489


namespace cyclic_sum_inequality_l159_159669

open Real

theorem cyclic_sum_inequality (a b c : ℝ) (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c)
  (ha_pos : 0 < a) (hb_pos : 0 < b) (hc_pos : 0 < c) (h_product : a * b * c = 1) :
  (a^6 / ((a - b) * (a - c)) + b^6 / ((b - c) * (b - a)) + c^6 / ((c - a) * (c - b)) > 15) := 
by sorry

end cyclic_sum_inequality_l159_159669


namespace range_of_function_l159_159422

noncomputable def f (x : ℝ) : ℝ := (x^2 + 5*x + 6) / (x + 2)

theorem range_of_function : 
  (∀ x : ℝ, x ≠ -2 → f x ≠ 1) ∧
  (∀ y : ℝ, y ≠ 1 → ∃ x : ℝ, f x = y) :=
sorry

end range_of_function_l159_159422


namespace total_value_is_84_l159_159299

-- Definitions based on conditions
def number_of_stamps : ℕ := 21
def value_of_7_stamps : ℕ := 28
def stamps_per_7 : ℕ := 7
def stamp_value : ℤ := value_of_7_stamps / stamps_per_7
def total_value_of_collection : ℤ := number_of_stamps * stamp_value

-- Statement to prove the total value of the stamp collection
theorem total_value_is_84 : total_value_of_collection = 84 := by
  sorry

end total_value_is_84_l159_159299


namespace final_value_after_three_years_l159_159992

theorem final_value_after_three_years (X : ℝ) :
  (X - 0.40 * X) * (1 - 0.10) * (1 - 0.20) = 0.432 * X := by
  sorry

end final_value_after_three_years_l159_159992


namespace mutually_exclusive_event_3_l159_159832

-- Definitions based on the conditions.
def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_even (n : ℕ) : Prop := n % 2 = 0

-- Events based on problem conditions
def event_1 (a b : ℕ) : Prop := is_even a ∧ is_odd b ∨ is_odd a ∧ is_even b
def event_2 (a b : ℕ) : Prop := (is_odd a ∨ is_odd b) ∧ is_odd a ∧ is_odd b
def event_3 (a b : ℕ) : Prop := (is_odd a ∨ is_odd b) ∧ is_even a ∧ is_even b
def event_4 (a b : ℕ) : Prop := (is_odd a ∨ is_odd b) ∧ (is_even a ∨ is_even b)

-- Problem: Proving that event_3 is mutually exclusive with other events.
theorem mutually_exclusive_event_3 :
  ∀ (a b : ℕ), (event_3 a b) → ¬ (event_1 a b ∨ event_2 a b ∨ event_4 a b) :=
by
  sorry

end mutually_exclusive_event_3_l159_159832


namespace evaluate_expression_l159_159803

theorem evaluate_expression :
  (3 ^ (1 ^ (0 ^ 8)) + ( (3 ^ 1) ^ 0 ) ^ 8) = 4 :=
by
  sorry

end evaluate_expression_l159_159803


namespace frog_problem_l159_159350

theorem frog_problem 
  (N : ℕ) 
  (h1 : N < 50) 
  (h2 : N % 2 = 1) 
  (h3 : N % 3 = 1) 
  (h4 : N % 4 = 1) 
  (h5 : N % 5 = 0) : 
  N = 25 := 
  sorry

end frog_problem_l159_159350


namespace symmetric_point_with_respect_to_y_eq_x_l159_159129

theorem symmetric_point_with_respect_to_y_eq_x :
  ∃ x₀ y₀ : ℝ, (∃ (M : ℝ × ℝ), M = (3, 1) ∧
  ((x₀ + 3) / 2 = (y₀ + 1) / 2) ∧
  ((y₀ - 1) / (x₀ - 3) = -1)) ∧
  (x₀ = 1 ∧ y₀ = 3) :=
by
  sorry

end symmetric_point_with_respect_to_y_eq_x_l159_159129


namespace remainder_division_Q_l159_159154

noncomputable def Q_rest : Polynomial ℝ := -(Polynomial.X : Polynomial ℝ) + 125

theorem remainder_division_Q (Q : Polynomial ℝ) :
  Q.eval 20 = 105 ∧ Q.eval 105 = 20 →
  ∃ R : Polynomial ℝ, Q = (Polynomial.X - 20) * (Polynomial.X - 105) * R + Q_rest :=
by sorry

end remainder_division_Q_l159_159154


namespace shaded_area_l159_159237

-- Defining the conditions
def small_square_side := 4
def large_square_side := 12
def half_large_square_side := large_square_side / 2

-- DG is calculated as (12 / 16) * small_square_side = 3
def DG := (large_square_side / (half_large_square_side + small_square_side)) * small_square_side

-- Calculating area of triangle DGF
def area_triangle_DGF := (DG * small_square_side) / 2

-- Area of the smaller square
def area_small_square := small_square_side * small_square_side

-- Area of the shaded region
def area_shaded_region := area_small_square - area_triangle_DGF

-- The theorem stating the question
theorem shaded_area : area_shaded_region = 10 := by
  sorry

end shaded_area_l159_159237


namespace revenue_highest_visitors_is_48_thousand_l159_159584

-- Define the frequencies for each day
def freq_Oct_1 : ℝ := 0.05
def freq_Oct_2 : ℝ := 0.08
def freq_Oct_3 : ℝ := 0.09
def freq_Oct_4 : ℝ := 0.13
def freq_Oct_5 : ℝ := 0.30
def freq_Oct_6 : ℝ := 0.15
def freq_Oct_7 : ℝ := 0.20

-- Define the revenue on October 1st
def revenue_Oct_1 : ℝ := 80000

-- Define the revenue is directly proportional to the frequency of visitors
def avg_daily_visitor_spending_is_constant := true

-- The goal is to prove that the revenue on the day with the highest frequency is 48 thousand yuan
theorem revenue_highest_visitors_is_48_thousand :
  avg_daily_visitor_spending_is_constant →
  revenue_Oct_1 / freq_Oct_1 = x / freq_Oct_5 →
  x = 48000 :=
by
  sorry

end revenue_highest_visitors_is_48_thousand_l159_159584


namespace line_bisects_circle_and_perpendicular_l159_159201

   def line_bisects_circle_and_is_perpendicular (x y : ℝ) : Prop :=
     (∃ (b : ℝ), ((2 * x - y + b = 0) ∧ (x^2 + y^2 - 2 * x - 4 * y = 0))) ∧
     ∀ b, (2 * 1 - 2 + b = 0) → b = 0 → (2 * x - y = 0)

   theorem line_bisects_circle_and_perpendicular :
     line_bisects_circle_and_is_perpendicular 1 2 :=
   by
     sorry
   
end line_bisects_circle_and_perpendicular_l159_159201


namespace relationship_abc_l159_159880

open Real

variable {x : ℝ}
variable (a b c : ℝ)
variable (h1 : 0 < x ∧ x ≤ 1)
variable (h2 : a = (sin x / x) ^ 2)
variable (h3 : b = sin x / x)
variable (h4 : c = sin (x^2) / x^2)

theorem relationship_abc (h1 : 0 < x ∧ x ≤ 1) (h2 : a = (sin x / x) ^ 2) (h3 : b = sin x / x) (h4 : c = sin (x^2) / x^2) :
  a < b ∧ b ≤ c :=
sorry

end relationship_abc_l159_159880


namespace female_employees_count_l159_159673

theorem female_employees_count (E Male_E Female_E M : ℕ)
  (h1: M = (2 / 5) * E)
  (h2: 200 = (E - Male_E) * (2 / 5))
  (h3: M = (2 / 5) * Male_E + 200) :
  Female_E = 500 := by
{
  sorry
}

end female_employees_count_l159_159673


namespace coin_problem_l159_159737

variable (x y S k : ℕ)

theorem coin_problem
  (h1 : x + y = 14)
  (h2 : 2 * x + 5 * y = S)
  (h3 : S = k + 2 * k)
  (h4 : k * 4 = S) :
  y = 4 ∨ y = 8 ∨ y = 12 :=
by
  sorry

end coin_problem_l159_159737


namespace compute_fg_l159_159403

def f (x : ℤ) : ℤ := x * x
def g (x : ℤ) : ℤ := 3 * x + 4

theorem compute_fg : f (g (-3)) = 25 := by
  sorry

end compute_fg_l159_159403


namespace extremum_condition_l159_159084

noncomputable def quadratic_polynomial (a x : ℝ) : ℝ := a * x^2 + x + 1

theorem extremum_condition (a : ℝ) :
  (∃ x : ℝ, ∃ f' : ℝ → ℝ, 
     (f' = (fun x => 2 * a * x + 1)) ∧ 
     (f' x = 0) ∧ 
     (∃ (f'' : ℝ → ℝ), (f'' = (fun x => 2 * a)) ∧ (f'' x ≠ 0))) ↔ a < 0 := 
sorry

end extremum_condition_l159_159084


namespace max_squares_covered_by_card_l159_159865

theorem max_squares_covered_by_card (side_len : ℕ) (card_side : ℕ) : 
  side_len = 1 → card_side = 2 → n ≤ 12 :=
by
  sorry

end max_squares_covered_by_card_l159_159865


namespace sum_zero_of_absolute_inequalities_l159_159683

theorem sum_zero_of_absolute_inequalities 
  (a b c : ℝ) 
  (h1 : |a| ≥ |b + c|) 
  (h2 : |b| ≥ |c + a|) 
  (h3 : |c| ≥ |a + b|) :
  a + b + c = 0 := 
  by
    sorry

end sum_zero_of_absolute_inequalities_l159_159683


namespace globe_surface_area_l159_159896

theorem globe_surface_area (d : ℚ) (h : d = 9) : 
  4 * Real.pi * (d / 2) ^ 2 = 81 * Real.pi := 
by 
  sorry

end globe_surface_area_l159_159896


namespace mean_median_sum_is_11_l159_159564

theorem mean_median_sum_is_11 (m n : ℕ) (h1 : m + 5 < n)
  (h2 : (m + (m + 3) + (m + 5) + n + (n + 1) + (2 * n - 1)) / 6 = n)
  (h3 : (m + 5 + n) / 2 = n) : m + n = 11 := by
  sorry

end mean_median_sum_is_11_l159_159564


namespace ensure_A_win_product_l159_159806

theorem ensure_A_win_product {s : Finset ℕ} (h1 : s = {1, 2, 3, 4, 5, 6, 7, 8, 9}) (h2 : 8 ∈ s) (h3 : 5 ∈ s) :
  (4 ∈ s ∧ 6 ∈ s ∧ 7 ∈ s) →
  4 * 6 * 7 = 168 := 
by 
  intro _ 
  exact Nat.mul_assoc 4 6 7

end ensure_A_win_product_l159_159806


namespace hiker_miles_l159_159983

-- Defining the conditions as a def
def total_steps (flips : ℕ) (additional_steps : ℕ) : ℕ := flips * 100000 + additional_steps

def steps_per_mile : ℕ := 1500

-- The target theorem to prove the number of miles walked
theorem hiker_miles (flips : ℕ) (additional_steps : ℕ) (s_per_mile : ℕ) 
  (h_flips : flips = 72) (h_additional_steps : additional_steps = 25370) 
  (h_s_per_mile : s_per_mile = 1500) : 
  (total_steps flips additional_steps) / s_per_mile = 4817 :=
by
  -- sorry is used to skip the actual proof
  sorry

end hiker_miles_l159_159983


namespace scientific_notation_of_425000_l159_159467

def scientific_notation (x : ℝ) : ℝ × ℤ := sorry

theorem scientific_notation_of_425000 :
  scientific_notation 425000 = (4.25, 5) := sorry

end scientific_notation_of_425000_l159_159467


namespace problem_statement_l159_159855

variable (a : ℝ)

theorem problem_statement (h : a^2 + 3*a - 4 = 0) : 2*a^2 + 6*a - 3 = 5 := 
by sorry

end problem_statement_l159_159855


namespace phase_shift_right_by_pi_div_3_l159_159963

noncomputable def graph_shift_right_by_pi_div_3 
  (A : ℝ := 1) 
  (ω : ℝ := 1) 
  (φ : ℝ := - (Real.pi / 3)) 
  (y : ℝ → ℝ := fun x => Real.sin (x - Real.pi / 3)) : 
  Prop :=
  y = fun x => Real.sin (x - (Real.pi / 3))

theorem phase_shift_right_by_pi_div_3 (A : ℝ := 1) (ω : ℝ := 1) (φ : ℝ := - (Real.pi / 3)) :
  graph_shift_right_by_pi_div_3 A ω φ (fun x => Real.sin (x - Real.pi / 3)) :=
sorry

end phase_shift_right_by_pi_div_3_l159_159963


namespace find_ab_l159_159985

theorem find_ab (A B : Set ℝ) (a b : ℝ) :
  (A = {x | x^2 - 2*x - 3 > 0}) →
  (B = {x | x^2 + a*x + b ≤ 0}) →
  (A ∪ B = Set.univ) → 
  (A ∩ B = {x | 3 < x ∧ x ≤ 4}) →
  a + b = -7 :=
by
  intros
  sorry

end find_ab_l159_159985


namespace trigonometric_identity_l159_159541

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 2) : 
  (2 * Real.cos α + 3 * Real.sin α) / (3 * Real.cos α - Real.sin α) = 8 :=
by 
  sorry

end trigonometric_identity_l159_159541


namespace teacups_count_l159_159283

theorem teacups_count (total_people teacup_capacity : ℕ) (H1 : total_people = 63) (H2 : teacup_capacity = 9) : total_people / teacup_capacity = 7 :=
by
  sorry

end teacups_count_l159_159283


namespace simplify_expression_l159_159758

variable (x y : ℝ)

theorem simplify_expression : 3 * y + 5 * y + 6 * y + 2 * x + 4 * x = 14 * y + 6 * x :=
by
  sorry

end simplify_expression_l159_159758


namespace larry_substitution_l159_159034

theorem larry_substitution (a b c d e : ℤ)
  (ha : a = 1)
  (hb : b = 2)
  (hc : c = 3)
  (hd : d = 4)
  (h_ignored : a - b - c - d + e = a - (b - (c - (d + e)))) :
  e = 3 :=
by
  sorry

end larry_substitution_l159_159034


namespace rotated_clockwise_120_correct_l159_159405

-- Problem setup definitions
structure ShapePosition :=
  (triangle : Point)
  (smaller_circle : Point)
  (square : Point)

-- Conditions for the initial positions of the shapes
variable (initial : ShapePosition)

def rotated_positions (initial: ShapePosition) : ShapePosition :=
  { 
    triangle := initial.smaller_circle,
    smaller_circle := initial.square,
    square := initial.triangle 
  }

-- Problem statement: show that after a 120° clockwise rotation, 
-- the shapes move to the specified new positions.
theorem rotated_clockwise_120_correct (initial : ShapePosition) 
  (after_rotation : ShapePosition) :
  after_rotation = rotated_positions initial := 
sorry

end rotated_clockwise_120_correct_l159_159405


namespace problem_cos_tan_half_l159_159599

open Real

theorem problem_cos_tan_half
  (α : ℝ)
  (hcos : cos α = -4/5)
  (hquad : π < α ∧ α < 3 * π / 2) :
  (1 + tan (α / 2)) / (1 - tan (α / 2)) = -1 / 2 :=
  sorry

end problem_cos_tan_half_l159_159599


namespace hiring_probability_l159_159362

noncomputable def combinatorics (n k : ℕ) : ℕ := Nat.choose n k

theorem hiring_probability (n : ℕ) (h1 : combinatorics 2 2 = 1)
                          (h2 : combinatorics (n - 2) 1 = n - 2)
                          (h3 : combinatorics n 3 = n * (n - 1) * (n - 2) / 6)
                          (h4 : (6 : ℕ) / (n * (n - 1) : ℚ) = 1 / 15) :
  n = 10 :=
by
  sorry

end hiring_probability_l159_159362


namespace smarties_division_l159_159218

theorem smarties_division (m : ℕ) (h : m % 7 = 5) : (4 * m) % 7 = 6 := by
  sorry

end smarties_division_l159_159218


namespace ball_height_less_than_10_after_16_bounces_l159_159203

noncomputable def bounce_height (initial : ℝ) (ratio : ℝ) (bounces : ℕ) : ℝ :=
  initial * ratio^bounces

theorem ball_height_less_than_10_after_16_bounces :
  let initial_height := 800
  let bounce_ratio := 3 / 4
  ∃ k : ℕ, k = 16 ∧ bounce_height initial_height bounce_ratio k < 10 := by
  let initial_height := 800
  let bounce_ratio := 3 / 4
  use 16
  sorry

end ball_height_less_than_10_after_16_bounces_l159_159203


namespace ratio_of_divisor_to_quotient_l159_159073

noncomputable def r : ℕ := 5
noncomputable def n : ℕ := 113

-- Assuming existence of k and quotient Q
axiom h1 : ∃ (k Q : ℕ), (3 * r + 3 = k * Q) ∧ (n = (3 * r + 3) * Q + r)

theorem ratio_of_divisor_to_quotient : ∃ (D Q : ℕ), (D = 3 * r + 3) ∧ (n = D * Q + r) ∧ (D / Q = 3) :=
  by sorry

end ratio_of_divisor_to_quotient_l159_159073


namespace sum_of_angles_eq_62_l159_159307

noncomputable def Φ (x : ℝ) : ℝ := Real.sin x
noncomputable def Ψ (x : ℝ) : ℝ := Real.cos x
def θ : List ℝ := [31, 30, 1, 0]

theorem sum_of_angles_eq_62 :
  θ.sum = 62 := by
  sorry

end sum_of_angles_eq_62_l159_159307


namespace find_tony_age_l159_159745

variable (y : ℕ)
variable (d : ℕ)

def Tony_day_hours : ℕ := 3
def Tony_hourly_rate (age : ℕ) : ℚ := 0.75 * age
def Tony_days_worked : ℕ := 60
def Tony_total_earnings : ℚ := 945

noncomputable def earnings_before_birthday (age : ℕ) (days : ℕ) : ℚ :=
  Tony_hourly_rate age * Tony_day_hours * days

noncomputable def earnings_after_birthday (age : ℕ) (days : ℕ) : ℚ :=
  Tony_hourly_rate (age + 1) * Tony_day_hours * days

noncomputable def total_earnings (age : ℕ) (days_before : ℕ) : ℚ :=
  (earnings_before_birthday age days_before) +
  (earnings_after_birthday age (Tony_days_worked - days_before))

theorem find_tony_age: ∃ y d : ℕ, total_earnings y d = Tony_total_earnings ∧ y = 6 := by
  sorry

end find_tony_age_l159_159745


namespace NaCl_moles_formed_l159_159390

-- Definitions for the conditions
def NaOH_moles : ℕ := 2
def Cl2_moles : ℕ := 1

-- Chemical reaction of NaOH and Cl2 resulting in NaCl and H2O
def reaction (n_NaOH n_Cl2 : ℕ) : ℕ :=
  if n_NaOH = 2 ∧ n_Cl2 = 1 then 2 else 0

-- Statement to be proved
theorem NaCl_moles_formed : reaction NaOH_moles Cl2_moles = 2 :=
by
  sorry

end NaCl_moles_formed_l159_159390


namespace production_days_l159_159578

theorem production_days (n : ℕ) (h₁ : (50 * n + 95) / (n + 1) = 55) : 
    n = 8 := 
    sorry

end production_days_l159_159578


namespace ellipse_equation_is_correct_line_equation_is_correct_l159_159994

-- Given conditions
variable (a b e x y : ℝ)
variable (a_pos : 0 < a)
variable (b_pos : 0 < b)
variable (ab_order : b < a)
variable (minor_axis_half_major_axis : 2 * a * (1 / 2) = 2 * b)
variable (right_focus_shortest_distance : a - e = 2 - Real.sqrt 3)
variable (ellipse_equation : a^2 = b^2 + e^2)
variable (m : ℝ)
variable (area_triangle_AOB_is_1 : 1 = 1)

-- Part (I) Prove the equation of ellipse C
theorem ellipse_equation_is_correct :
  (∀ x y : ℝ, (x^2 / 4 + y^2 = 1) ↔ (x^2 / a^2 + y^2 / b^2 = 1)) :=
sorry

-- Part (II) Prove the equation of line l
theorem line_equation_is_correct :
  (∀ x y : ℝ, (y = x + m) ↔ ((y = x + (Real.sqrt 10 / 2)) ∨ (y = x - (Real.sqrt 10 / 2)))) :=
sorry

end ellipse_equation_is_correct_line_equation_is_correct_l159_159994


namespace find_least_x_divisible_by_17_l159_159979

theorem find_least_x_divisible_by_17 (x k : ℕ) (h : x + 2 = 17 * k) : x = 15 :=
sorry

end find_least_x_divisible_by_17_l159_159979


namespace maryann_free_time_l159_159610

theorem maryann_free_time
    (x : ℕ)
    (expensive_time : ℕ := 8)
    (friends : ℕ := 3)
    (total_time : ℕ := 42)
    (lockpicking_time : 3 * (x + expensive_time) = total_time) : 
    x = 6 :=
by
  sorry

end maryann_free_time_l159_159610


namespace moles_of_NaOH_combined_l159_159831

-- Given conditions
def moles_AgNO3 := 3
def moles_AgOH := 3
def balanced_ratio_AgNO3_NaOH := 1 -- 1:1 ratio as per the equation

-- Problem statement
theorem moles_of_NaOH_combined : 
  moles_AgOH = moles_AgNO3 → balanced_ratio_AgNO3_NaOH = 1 → 
  (∃ moles_NaOH, moles_NaOH = 3) := by
  sorry

end moles_of_NaOH_combined_l159_159831


namespace total_distance_of_bus_rides_l159_159257

theorem total_distance_of_bus_rides :
  let vince_distance   := 5 / 8
  let zachary_distance := 1 / 2
  let alice_distance   := 17 / 20
  let rebecca_distance := 2 / 5
  let total_distance   := vince_distance + zachary_distance + alice_distance + rebecca_distance
  total_distance = 19/8 := by
  sorry

end total_distance_of_bus_rides_l159_159257


namespace correct_M_l159_159978

-- Definition of the function M for calculating the position number
def M (k : ℕ) : ℕ :=
  if k % 2 = 1 then
    4 * k^2 - 4 * k + 2
  else
    4 * k^2 - 2 * k + 2

-- Theorem stating the correctness of the function M
theorem correct_M (k : ℕ) : M k = if k % 2 = 1 then 4 * k^2 - 4 * k + 2 else 4 * k^2 - 2 * k + 2 := 
by
  -- The proof is to be done later.
  -- sorry is used to indicate a placeholder.
  sorry

end correct_M_l159_159978


namespace polygon_edges_l159_159329

theorem polygon_edges :
  ∃ a b : ℕ, a + b = 2014 ∧
              (a * (a - 3) / 2 + b * (b - 3) / 2 = 1014053) ∧
              a ≤ b ∧
              a = 952 :=
by
  sorry

end polygon_edges_l159_159329


namespace solve_diophantine_equation_l159_159276

theorem solve_diophantine_equation :
  ∃ (x y : ℤ), x^2 - x * y - 6 * y^2 + 2 * x + 19 * y = 18 ∧ (x = 2 ∧ y = 2 ∨ x = -2 ∧ y = 2) :=
  sorry

end solve_diophantine_equation_l159_159276


namespace usual_time_is_20_l159_159194

-- Define the problem
variables (T T': ℕ)

-- Conditions
axiom condition1 : T' = T + 5
axiom condition2 : T' = 5 * T / 4

-- Proof statement
theorem usual_time_is_20 : T = 20 :=
  sorry

end usual_time_is_20_l159_159194


namespace johns_minutes_billed_l159_159490

theorem johns_minutes_billed 
  (monthly_fee : ℝ) (cost_per_minute : ℝ) (total_bill : ℝ) 
  (h1 : monthly_fee = 5) (h2 : cost_per_minute = 0.25) (h3 : total_bill = 12.02) :
  ⌊(total_bill - monthly_fee) / cost_per_minute⌋ = 28 :=
by
  sorry

end johns_minutes_billed_l159_159490


namespace hyperbola_sum_l159_159557

noncomputable def h : ℝ := 3
noncomputable def k : ℝ := -4
noncomputable def a : ℝ := 4
noncomputable def c : ℝ := Real.sqrt 53
noncomputable def b : ℝ := Real.sqrt (c^2 - a^2)

theorem hyperbola_sum : h + k + a + b = 3 + Real.sqrt 37 :=
by
  -- sorry is used to skip the proof as per the instruction
  sorry
  -- exact calc
  --   h + k + a + b = 3 + (-4) + 4 + Real.sqrt 37 : by simp
  --             ... = 3 + Real.sqrt 37 : by simp

end hyperbola_sum_l159_159557


namespace part1_part2_l159_159938

-- Define the linear function
def linear_function (m x : ℝ) : ℝ := (m - 2) * x + 6

-- Prove part 1: If y increases as x increases, then m > 2
theorem part1 (m : ℝ) : (∀ x1 x2 : ℝ, x1 < x2 → linear_function m x1 < linear_function m x2) → m > 2 :=
sorry

-- Prove part 2: When -2 ≤ x ≤ 4, and y ≤ 10, the range of m is (2, 3] or [0, 2)
theorem part2 (m : ℝ) : 
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 4 → linear_function m x ≤ 10) →
  (2 < m ∧ m ≤ 3) ∨ (0 ≤ m ∧ m < 2) :=
sorry

end part1_part2_l159_159938


namespace rates_of_interest_l159_159040

theorem rates_of_interest (P_B P_C T_B T_C SI_B SI_C : ℝ) (R_B R_C : ℝ)
  (hB1 : P_B = 5000) (hB2: T_B = 5) (hB3: SI_B = 2200)
  (hC1 : P_C = 3000) (hC2 : T_C = 7) (hC3 : SI_C = 2730)
  (simple_interest : ∀ {P R T SI : ℝ}, SI = (P * R * T) / 100)
  : R_B = 8.8 ∧ R_C = 13 := by
  sorry

end rates_of_interest_l159_159040


namespace inequality_proof_l159_159352

theorem inequality_proof (x y : ℝ) (h1 : y ≥ 0) (h2 : y * (y + 1) ≤ (x + 1)^2) : y * (y - 1) ≤ x^2 :=
sorry

end inequality_proof_l159_159352


namespace mouse_grasshopper_diff_l159_159547

def grasshopper_jump: ℕ := 19
def frog_jump: ℕ := grasshopper_jump + 10
def mouse_jump: ℕ := frog_jump + 20

theorem mouse_grasshopper_diff:
  (mouse_jump - grasshopper_jump) = 30 :=
by
  sorry

end mouse_grasshopper_diff_l159_159547


namespace probability_wheel_l159_159321

theorem probability_wheel (P : ℕ → ℚ) 
  (hA : P 0 = 1/4) 
  (hB : P 1 = 1/3) 
  (hC : P 2 = 1/6) 
  (hSum : P 0 + P 1 + P 2 + P 3 = 1) : 
  P 3 = 1/4 := 
by 
  -- Proof here
  sorry

end probability_wheel_l159_159321


namespace least_multiple_17_gt_500_l159_159715

theorem least_multiple_17_gt_500 (n : ℕ) (h : (n = 17)) : ∃ m : ℤ, (m * n > 500 ∧ m * n = 510) :=
  sorry

end least_multiple_17_gt_500_l159_159715


namespace quadratic_roots_form_l159_159876

theorem quadratic_roots_form {a b c : ℤ} (h : a = 3 ∧ b = -7 ∧ c = 1) :
  ∃ (m n p : ℤ), (∀ x, 3*x^2 - 7*x + 1 = 0 ↔ x = (m + Real.sqrt n)/p ∨ x = (m - Real.sqrt n)/p)
  ∧ Int.gcd m (Int.gcd n p) = 1 ∧ n = 37 :=
by
  sorry

end quadratic_roots_form_l159_159876


namespace product_abc_l159_159685

theorem product_abc (a b c : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_eqn : a * b * c = a * b^3) (h_c_eq_1 : c = 1) :
  a * b * c = a :=
by
  sorry

end product_abc_l159_159685


namespace employed_females_percentage_l159_159771

theorem employed_females_percentage (total_population : ℝ) (total_employed_percentage : ℝ) (employed_males_percentage : ℝ) :
  total_employed_percentage = 0.7 →
  employed_males_percentage = 0.21 →
  total_population > 0 →
  (total_employed_percentage - employed_males_percentage) / total_employed_percentage * 100 = 70 :=
by
  intros h1 h2 h3
  -- Proof is omitted.
  sorry

end employed_females_percentage_l159_159771


namespace calculate_f_at_8_l159_159679

def f (x : ℝ) : ℝ := 2 * x^4 - 17 * x^3 + 27 * x^2 - 24 * x - 72

theorem calculate_f_at_8 : f 8 = 952 :=
by sorry

end calculate_f_at_8_l159_159679


namespace find_x_l159_159254

-- Defining the number x and the condition
variable (x : ℝ) 

-- The condition given in the problem
def condition := x / 3 = x - 3

-- The theorem to be proved
theorem find_x (h : condition x) : x = 4.5 := 
by 
  sorry

end find_x_l159_159254


namespace ball_arrangements_l159_159829

-- Define the structure of the boxes and balls
structure BallDistributions where
  white_balls_box1 : ℕ
  black_balls_box1 : ℕ
  white_balls_box2 : ℕ
  black_balls_box2 : ℕ
  white_balls_box3 : ℕ
  black_balls_box3 : ℕ

-- Problem conditions
def valid_distribution (d : BallDistributions) : Prop :=
  d.white_balls_box1 + d.black_balls_box1 ≥ 2 ∧
  d.white_balls_box2 + d.black_balls_box2 ≥ 2 ∧
  d.white_balls_box3 + d.black_balls_box3 ≥ 2 ∧
  d.white_balls_box1 ≥ 1 ∧
  d.black_balls_box1 ≥ 1 ∧
  d.white_balls_box2 ≥ 1 ∧
  d.black_balls_box2 ≥ 1 ∧
  d.white_balls_box3 ≥ 1 ∧
  d.black_balls_box3 ≥ 1

def total_white_balls (d : BallDistributions) : ℕ :=
  d.white_balls_box1 + d.white_balls_box2 + d.white_balls_box3

def total_black_balls (d : BallDistributions) : ℕ :=
  d.black_balls_box1 + d.black_balls_box2 + d.black_balls_box3

def correct_distribution (d : BallDistributions) : Prop :=
  total_white_balls d = 4 ∧ total_black_balls d = 5

-- Main theorem to prove
theorem ball_arrangements : ∃ (d : BallDistributions), valid_distribution d ∧ correct_distribution d ∧ (number_of_distributions = 18) :=
  sorry

end ball_arrangements_l159_159829


namespace overall_average_marks_l159_159280

theorem overall_average_marks 
  (num_candidates : ℕ) 
  (num_passed : ℕ) 
  (avg_passed : ℕ) 
  (avg_failed : ℕ)
  (h1 : num_candidates = 120) 
  (h2 : num_passed = 100)
  (h3 : avg_passed = 39)
  (h4 : avg_failed = 15) :
  (num_passed * avg_passed + (num_candidates - num_passed) * avg_failed) / num_candidates = 35 := 
by
  sorry

end overall_average_marks_l159_159280


namespace original_number_increased_l159_159349

theorem original_number_increased (x : ℝ) (h : (1.10 * x) * 1.15 = 632.5) : x = 500 :=
sorry

end original_number_increased_l159_159349


namespace functional_equation_solution_exists_l159_159542

theorem functional_equation_solution_exists (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (f x + y) = 2 * x + f (f y - x)) →
  ∃ c : ℝ, ∀ x : ℝ, f x = x + c :=
by
  intro h
  sorry

end functional_equation_solution_exists_l159_159542


namespace table_price_l159_159242

theorem table_price :
  ∃ C T : ℝ, (2 * C + T = 0.6 * (C + 2 * T)) ∧ (C + T = 72) ∧ (T = 63) :=
by
  sorry

end table_price_l159_159242


namespace domain_correct_l159_159853

def domain_of_function (x : ℝ) : Prop :=
  (x > 2) ∧ (x ≠ 5)

theorem domain_correct : {x : ℝ | domain_of_function x} = {x : ℝ | x > 2 ∧ x ≠ 5} :=
by
  sorry

end domain_correct_l159_159853


namespace math_proof_problem_l159_159485

noncomputable def problem_statement : Prop :=
  ∃ (x : ℝ), (x > 12) ∧ ((x - 5) / 12 = 5 / (x - 12)) ∧ (x = 17)

theorem math_proof_problem : problem_statement :=
by
  sorry

end math_proof_problem_l159_159485


namespace desired_percentage_total_annual_income_l159_159046

variable (investment1 : ℝ)
variable (investment2 : ℝ)
variable (rate1 : ℝ)
variable (rate2 : ℝ)

theorem desired_percentage_total_annual_income (h1 : investment1 = 2000)
  (h2 : rate1 = 0.05)
  (h3 : investment2 = 1000-1e-13)
  (h4 : rate2 = 0.08):
  ((investment1 * rate1 + investment2 * rate2) / (investment1 + investment2) * 100) = 6 := by
  sorry

end desired_percentage_total_annual_income_l159_159046


namespace cos_product_equals_one_eighth_l159_159788

noncomputable def cos_pi_over_9 := Real.cos (Real.pi / 9)
noncomputable def cos_2pi_over_9 := Real.cos (2 * Real.pi / 9)
noncomputable def cos_4pi_over_9 := Real.cos (4 * Real.pi / 9)

theorem cos_product_equals_one_eighth :
  cos_pi_over_9 * cos_2pi_over_9 * cos_4pi_over_9 = 1 / 8 := 
sorry

end cos_product_equals_one_eighth_l159_159788


namespace fg_difference_l159_159841

def f (x : ℝ) : ℝ := 2 * x + 5
def g (x : ℝ) : ℝ := 4 * x - 1

theorem fg_difference : f (g 3) - g (f 3) = -16 := by
  sorry

end fg_difference_l159_159841


namespace negation_of_exists_inequality_l159_159343

theorem negation_of_exists_inequality :
  ¬ (∃ x : ℝ, x * x + 4 * x + 5 ≤ 0) ↔ ∀ x : ℝ, x * x + 4 * x + 5 > 0 :=
by
  sorry

end negation_of_exists_inequality_l159_159343


namespace part1_part2_l159_159530

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := Real.exp x * (Real.log x + k)
noncomputable def f_prime (x : ℝ) (k : ℝ) : ℝ := Real.exp x * (Real.log x + k) + Real.exp x / x
noncomputable def g (x : ℝ) (k : ℝ) : ℝ := f_prime x k - 2 * (f x k + Real.exp x)
noncomputable def phi (x : ℝ) : ℝ := Real.exp x / x

theorem part1 (h : f_prime 1 k = 0) : k = -1 := sorry

theorem part2 (t : ℝ) (h_g_le_phi : ∀ x > 0, g x (-1) ≤ t * phi x) : t ≥ 1 + 1 / Real.exp 2 := sorry

end part1_part2_l159_159530


namespace Derrick_yard_length_l159_159281

variables (Alex_yard Derrick_yard Brianne_yard Carla_yard Derek_yard : ℝ)

-- Given conditions as hypotheses
theorem Derrick_yard_length :
  (Alex_yard = Derrick_yard / 2) →
  (Brianne_yard = 6 * Alex_yard) →
  (Carla_yard = 3 * Brianne_yard + 5) →
  (Derek_yard = Carla_yard / 2 - 10) →
  (Brianne_yard = 30) →
  Derrick_yard = 10 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end Derrick_yard_length_l159_159281


namespace product_a2_a3_a4_l159_159879

open Classical

noncomputable def geometric_sequence (a : ℕ → ℚ) (a1 : ℚ) (q : ℚ) : Prop :=
∀ n : ℕ, a n = a1 * q^(n - 1)

theorem product_a2_a3_a4 (a : ℕ → ℚ) (q : ℚ) 
  (h_seq : geometric_sequence a 1 q)
  (h_a1 : a 1 = 1)
  (h_a5 : a 5 = 1 / 9) :
  a 2 * a 3 * a 4 = 1 / 27 :=
sorry

end product_a2_a3_a4_l159_159879


namespace youseff_blocks_from_office_l159_159814

def blocks_to_office (x : ℕ) : Prop :=
  let walk_time := x  -- it takes x minutes to walk
  let bike_time := (20 * x) / 60  -- it takes (20 / 60) * x = (1 / 3) * x minutes to ride a bike
  walk_time = bike_time + 4  -- walking takes 4 more minutes than biking

theorem youseff_blocks_from_office (x : ℕ) (h : blocks_to_office x) : x = 6 :=
  sorry

end youseff_blocks_from_office_l159_159814


namespace juice_oranges_l159_159850

theorem juice_oranges (oranges_per_glass : ℕ) (glasses : ℕ) (total_oranges : ℕ)
  (h1 : oranges_per_glass = 3)
  (h2 : glasses = 10)
  (h3 : total_oranges = oranges_per_glass * glasses) :
  total_oranges = 30 :=
by
  rw [h1, h2] at h3
  exact h3

end juice_oranges_l159_159850


namespace min_length_M_inter_N_l159_159265

def setM (m : ℝ) : Set ℝ := {x | m ≤ x ∧ x ≤ m + 3 / 4}
def setN (n : ℝ) : Set ℝ := {x | n - 1 / 3 ≤ x ∧ x ≤ n}
def setP : Set ℝ := {x | 0 ≤ x ∧ x ≤ 1}

theorem min_length_M_inter_N (m n : ℝ) 
  (hm : 0 ≤ m ∧ m + 3 / 4 ≤ 1) 
  (hn : 1 / 3 ≤ n ∧ n ≤ 1) : 
  let I := (setM m ∩ setN n)
  ∃ Iinf Isup : ℝ, I = {x | Iinf ≤ x ∧ x ≤ Isup} ∧ Isup - Iinf = 1 / 12 :=
  sorry

end min_length_M_inter_N_l159_159265


namespace find_multiplying_number_l159_159463

variable (a b : ℤ)

theorem find_multiplying_number (h : a^2 * b = 3 * (4 * a + 2)) (ha : a = 1) :
  b = 18 := by
  sorry

end find_multiplying_number_l159_159463


namespace borya_number_l159_159891

theorem borya_number (n : ℕ) (h1 : 10 ≤ n) (h2 : n < 100) 
  (h3 : (n * 2 + 5) * 5 = 715) : n = 69 :=
sorry

end borya_number_l159_159891


namespace count_valid_x_satisfying_heartsuit_condition_l159_159314

def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem count_valid_x_satisfying_heartsuit_condition :
  (∃ n, ∀ x, 1 ≤ x ∧ x < 1000 → digit_sum (digit_sum x) = 4 → n = 36) :=
by
  sorry

end count_valid_x_satisfying_heartsuit_condition_l159_159314


namespace initial_number_is_31_l159_159337

theorem initial_number_is_31 (N : ℕ) (h : ∃ k : ℕ, N - 10 = 21 * k) : N = 31 :=
sorry

end initial_number_is_31_l159_159337


namespace expression_divisible_by_7_l159_159327

theorem expression_divisible_by_7 (n : ℕ) (hn : n > 0) :
  7 ∣ (3^(3*n+1) + 5^(3*n+2) + 7^(3*n+3)) :=
sorry

end expression_divisible_by_7_l159_159327


namespace part1_eq_part2_if_empty_intersection_then_a_geq_3_l159_159179

open Set

variable {U : Type} {a : ℝ}

def universal_set : Set ℝ := univ
def A : Set ℝ := {x : ℝ | 1 < x ∧ x < 3}
def B1 (a : ℝ) : Set ℝ := {x : ℝ | x > a}
def complement_B1 (a : ℝ) : Set ℝ := {x : ℝ | x ≤ a}
def intersection_with_complement (a : ℝ) : Set ℝ := A ∩ complement_B1 a

-- Statement for part (1)
theorem part1_eq {a : ℝ} (h : a = 2) : intersection_with_complement a = {x : ℝ | 1 < x ∧ x ≤ 2} :=
by sorry

-- Statement for part (2)
theorem part2_if_empty_intersection_then_a_geq_3 
(h : A ∩ B1 a = ∅) : a ≥ 3 :=
by sorry

end part1_eq_part2_if_empty_intersection_then_a_geq_3_l159_159179


namespace sum_abc_eq_8_l159_159011

theorem sum_abc_eq_8 (a b c : ℝ) 
  (h : (a - 5) ^ 2 + (b - 6) ^ 2 + (c - 7) ^ 2 - 2 * (a - 5) * (b - 6) = 0) : 
  a + b + c = 8 := 
sorry

end sum_abc_eq_8_l159_159011


namespace line_equation_l159_159370

theorem line_equation {m : ℤ} :
  (∀ x y : ℤ, 2 * x + y + m = 0) →
  (∀ x y : ℤ, 2 * x + y - 10 = 0) →
  (2 * 1 + 0 + m = 0) →
  m = -2 :=
by
  sorry

end line_equation_l159_159370


namespace speed_of_first_car_l159_159289

theorem speed_of_first_car 
  (distance_highway : ℕ)
  (time_to_meet : ℕ)
  (speed_second_car : ℕ)
  (total_distance_covered : distance_highway = time_to_meet * 40 + time_to_meet * speed_second_car): 
  5 * 40 + 5 * 60 = distance_highway := 
by
  /-
    Given:
      - distance_highway : ℕ (The length of the highway, which is 500 miles)
      - time_to_meet : ℕ (The time after which the two cars meet, which is 5 hours)
      - speed_second_car : ℕ (The speed of the second car, which is 60 mph)
      - total_distance_covered : distance_highway = time_to_meet * speed_of_first_car + time_to_meet * speed_second_car

    We need to prove:
      - 5 * 40 + 5 * 60 = distance_highway
  -/

  sorry

end speed_of_first_car_l159_159289


namespace fat_caterpillars_left_l159_159867

-- Define the initial and the newly hatched caterpillars
def initial_caterpillars : ℕ := 14
def hatched_caterpillars : ℕ := 4

-- Define the caterpillars left on the tree now
def current_caterpillars : ℕ := 10

-- Define the total caterpillars before any left
def total_caterpillars : ℕ := initial_caterpillars + hatched_caterpillars
-- Define the caterpillars leaving the tree
def caterpillars_left : ℕ := total_caterpillars - current_caterpillars

-- The theorem to be proven
theorem fat_caterpillars_left : caterpillars_left = 8 :=
by
  sorry

end fat_caterpillars_left_l159_159867


namespace finite_pos_int_set_condition_l159_159309

theorem finite_pos_int_set_condition (X : Finset ℕ) 
  (hX : ∀ a ∈ X, 0 < a) 
  (h2 : 2 ≤ X.card) 
  (hcond : ∀ {a b : ℕ}, a ∈ X → b ∈ X → a > b → b^2 / (a - b) ∈ X) :
  ∃ a : ℕ, X = {a, 2 * a} :=
by
  sorry

end finite_pos_int_set_condition_l159_159309


namespace domain_of_function_l159_159039

noncomputable def function_defined (x : ℝ) : Prop :=
  (x > 1) ∧ (x ≠ 2)

theorem domain_of_function :
  ∀ x : ℝ, (∃ y : ℝ, y = (1 / (Real.sqrt (x - 1))) + (1 / (x - 2))) ↔ function_defined x :=
by sorry

end domain_of_function_l159_159039


namespace difference_of_squares_example_l159_159445

theorem difference_of_squares_example :
  (262^2 - 258^2 = 2080) :=
by {
  sorry -- placeholder for the actual proof
}

end difference_of_squares_example_l159_159445


namespace matchsticks_20th_stage_l159_159656

theorem matchsticks_20th_stage :
  let a1 := 3
  let d := 3
  let a20 := a1 + 19 * d
  a20 = 60 := by
  sorry

end matchsticks_20th_stage_l159_159656


namespace average_of_numbers_l159_159266

theorem average_of_numbers : 
  (12 + 13 + 14 + 510 + 520 + 530 + 1115 + 1120 + 1 + 1252140 + 2345) / 11 = 114391 :=
by
  sorry

end average_of_numbers_l159_159266


namespace maximize_probability_l159_159110

def numbers_list : List Int := [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

def pairs_summing_to_12 (l : List Int) : List (Int × Int) :=
  List.filter (fun (p : Int × Int) => p.1 + p.2 = 12) (List.product l l)

def distinct_pairs (pairs : List (Int × Int)) : List (Int × Int) :=
  List.filter (fun (p : Int × Int) => p.1 ≠ p.2) pairs

def valid_pairs (l : List Int) : List (Int × Int) :=
  distinct_pairs (pairs_summing_to_12 l)

def count_valid_pairs (l : List Int) : Nat :=
  List.length (valid_pairs l)

def remove_and_check (x : Int) : List Int :=
  List.erase numbers_list x

theorem maximize_probability :
  ∀ x : Int, count_valid_pairs (remove_and_check 6) ≥ count_valid_pairs (remove_and_check x) :=
sorry

end maximize_probability_l159_159110


namespace sum_of_first_n_natural_numbers_single_digit_l159_159004

theorem sum_of_first_n_natural_numbers_single_digit (n : ℕ) :
  (∃ a : ℕ, a ≤ 9 ∧ (a ≠ 0) ∧ 37 * (3 * a) = n * (n + 1) / 2) ↔ (n = 36) :=
by
  sorry

end sum_of_first_n_natural_numbers_single_digit_l159_159004


namespace negation_of_universal_quadratic_l159_159208

theorem negation_of_universal_quadratic (P : ∀ a b c : ℝ, a ≠ 0 → ∃ x : ℝ, a * x^2 + b * x + c = 0) :
  ¬(∀ a b c : ℝ, a ≠ 0 → ∃ x : ℝ, a * x^2 + b * x + c = 0) ↔ ∃ a b c : ℝ, a ≠ 0 ∧ ¬(∃ x : ℝ, a * x^2 + b * x + c = 0) :=
by
  sorry

end negation_of_universal_quadratic_l159_159208


namespace expand_polynomials_l159_159339

def p (z : ℝ) : ℝ := 3 * z ^ 2 + 4 * z - 7
def q (z : ℝ) : ℝ := 4 * z ^ 3 - 3 * z + 2

theorem expand_polynomials :
  (p z) * (q z) = 12 * z ^ 5 + 16 * z ^ 4 - 37 * z ^ 3 - 6 * z ^ 2 + 29 * z - 14 := by
  sorry

end expand_polynomials_l159_159339


namespace part_a_part_b_l159_159607

variable (p : ℕ → ℕ)
axiom primes_sequence : ∀ n, (∀ m < p n, m ∣ p n → m = 1 ∨ m = p n) ∧ p 1 = 2 ∧ p 2 = 3 ∧ p 3 = 5 ∧ p 4 = 7 ∧ p 5 = 11

theorem part_a (n : ℕ) (h : n ≥ 5) : p n > 2 * n := 
  by sorry

theorem part_b (n : ℕ) : p n > 3 * n ↔ n ≥ 12 := 
  by sorry

end part_a_part_b_l159_159607


namespace find_m_l159_159886

-- Define the function with given conditions
def f (m : ℕ) (n : ℕ) : ℕ := 
if n > m^2 then n - m + 14 else sorry

-- Define the main problem
theorem find_m (m : ℕ) (hyp : m ≥ 14) : f m 1995 = 1995 ↔ m = 14 ∨ m = 45 :=
by
  sorry

end find_m_l159_159886


namespace line_passes_through_2nd_and_4th_quadrants_l159_159123

theorem line_passes_through_2nd_and_4th_quadrants (b : ℝ) :
  (∀ x : ℝ, x > 0 → -2 * x + b < 0) ∧ (∀ x : ℝ, x < 0 → -2 * x + b > 0) :=
by
  sorry

end line_passes_through_2nd_and_4th_quadrants_l159_159123


namespace infinite_nested_radical_l159_159415

theorem infinite_nested_radical : ∀ (x : ℝ), (x > 0) → (x = Real.sqrt (12 + x)) → x = 4 :=
by
  intro x
  intro hx_pos
  intro hx_eq
  sorry

end infinite_nested_radical_l159_159415


namespace square_perimeter_l159_159316

theorem square_perimeter (area : ℝ) (h : area = 144) : ∃ perimeter : ℝ, perimeter = 48 :=
by
  sorry

end square_perimeter_l159_159316


namespace maximum_m_l159_159036

theorem maximum_m (a b c : ℝ)
  (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c)
  (h₄ : a + b + c = 10)
  (h₅ : a * b + b * c + c * a = 25) :
  ∃ m, (m = min (a * b) (min (b * c) (c * a)) ∧ m = 25 / 9) :=
sorry

end maximum_m_l159_159036


namespace bakery_regular_price_l159_159427

theorem bakery_regular_price (y : ℝ) (h₁ : y / 4 * 0.4 = 2) : y = 20 :=
by {
  sorry
}

end bakery_regular_price_l159_159427


namespace preimage_of_mapping_l159_159157

def f (a b : ℝ) : ℝ × ℝ := (a + 2 * b, 2 * a - b)

theorem preimage_of_mapping : ∃ (a b : ℝ), f a b = (3, 1) ∧ (a, b) = (1, 1) :=
by
  sorry

end preimage_of_mapping_l159_159157


namespace girls_in_class_l159_159712

theorem girls_in_class :
  ∀ (x : ℕ), (12 * 84 + 92 * x = 86 * (12 + x)) → x = 4 :=
by
  sorry

end girls_in_class_l159_159712


namespace total_flour_amount_l159_159093

-- Define the initial amount of flour in the bowl
def initial_flour : ℝ := 2.75

-- Define the amount of flour added by the baker
def added_flour : ℝ := 0.45

-- Prove that the total amount of flour is 3.20 kilograms
theorem total_flour_amount : initial_flour + added_flour = 3.20 :=
by
  sorry

end total_flour_amount_l159_159093


namespace tank_capacity_correct_l159_159719

-- Define rates and times for each pipe
def rate_a : ℕ := 200 -- in liters per minute
def rate_b : ℕ := 50 -- in liters per minute
def rate_c : ℕ := 25 -- in liters per minute

def time_a : ℕ := 1 -- pipe A open time in minutes
def time_b : ℕ := 2 -- pipe B open time in minutes
def time_c : ℕ := 2 -- pipe C open time in minutes

def cycle_time : ℕ := time_a + time_b + time_c -- total time for one cycle in minutes
def total_time : ℕ := 40 -- total time to fill the tank in minutes

-- Net water added in one cycle
def net_water_in_cycle : ℕ :=
  (rate_a * time_a) + (rate_b * time_b) - (rate_c * time_c)

-- Number of cycles needed to fill the tank
def number_of_cycles : ℕ :=
  total_time / cycle_time

-- Total capacity of the tank
def tank_capacity : ℕ :=
  number_of_cycles * net_water_in_cycle

-- The hypothesis to prove
theorem tank_capacity_correct :
  tank_capacity = 2000 :=
  by
    sorry

end tank_capacity_correct_l159_159719


namespace one_fourth_of_six_point_eight_eq_seventeen_tenths_l159_159126

theorem one_fourth_of_six_point_eight_eq_seventeen_tenths :  (6.8 / 4) = (17 / 10) :=
by
  -- Skipping proof
  sorry

end one_fourth_of_six_point_eight_eq_seventeen_tenths_l159_159126


namespace most_appropriate_method_to_solve_4x2_minus_9_eq_0_l159_159986

theorem most_appropriate_method_to_solve_4x2_minus_9_eq_0 :
  (∀ x : ℤ, 4 * x^2 - 9 = 0 ↔ x = 3 / 2 ∨ x = -3 / 2) → true :=
by
  sorry

end most_appropriate_method_to_solve_4x2_minus_9_eq_0_l159_159986


namespace length_less_than_twice_width_l159_159086

def length : ℝ := 24
def width : ℝ := 13.5

theorem length_less_than_twice_width : 2 * width - length = 3 := by
  sorry

end length_less_than_twice_width_l159_159086


namespace simple_interest_rate_l159_159229

theorem simple_interest_rate (P : ℝ) (T : ℝ) (R : ℝ) (SI : ℝ) (hT : T = 8) 
  (hSI : SI = P / 5) : SI = (P * R * T) / 100 → R = 2.5 :=
by
  intro
  sorry

end simple_interest_rate_l159_159229


namespace find_f_7_l159_159529

noncomputable def f (a b c x : ℝ) : ℝ := a * x^7 + b * x^3 + c * x - 5

theorem find_f_7 (a b c : ℝ) (h : f a b c (-7) = 7) : f a b c 7 = -17 :=
by
  dsimp [f] at *
  sorry

end find_f_7_l159_159529


namespace second_smallest_is_3_probability_l159_159947

noncomputable def probability_of_second_smallest_is_3 : ℚ := 
  let total_ways := Nat.choose 10 6
  let favorable_ways := 2 * Nat.choose 7 4
  favorable_ways / total_ways

theorem second_smallest_is_3_probability : probability_of_second_smallest_is_3 = 1 / 3 := sorry

end second_smallest_is_3_probability_l159_159947


namespace measure_of_angle_A_range_of_b2_add_c2_div_a2_l159_159548

variable {A B C a b c : ℝ}
variable {S : ℝ}

theorem measure_of_angle_A
  (h1 : S = 1 / 2 * b * c * Real.sin A)
  (h2 : 4 * Real.sqrt 3 * S = a ^ 2 - (b - c) ^ 2)
  (h3 : a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * Real.cos A) : 
  A = 2 * Real.pi / 3 :=
by
  sorry

theorem range_of_b2_add_c2_div_a2
  (h1 : S = 1 / 2 * b * c * Real.sin A)
  (h2 : 4 * Real.sqrt 3 * S = a ^ 2 - (b - c) ^ 2)
  (h3 : A = 2 * Real.pi / 3) : 
  2 / 3 ≤ (b ^ 2 + c ^ 2) / a ^ 2 ∧ (b ^ 2 + c ^ 2) / a ^ 2 < 1 :=
by
  sorry

end measure_of_angle_A_range_of_b2_add_c2_div_a2_l159_159548


namespace radius_of_inscribed_circle_l159_159443

variable (p q r : ℝ)

theorem radius_of_inscribed_circle (hp : p > 0) (hq : q > 0) (area_eq : q^2 = r * p) : r = q^2 / p :=
by
  sorry

end radius_of_inscribed_circle_l159_159443


namespace factor_expression_l159_159478

theorem factor_expression (x a b c : ℝ) :
  (x - a) ^ 2 * (b - c) + (x - b) ^ 2 * (c - a) + (x - c) ^ 2 * (a - b) = -(a - b) * (b - c) * (c - a) :=
by
  sorry

end factor_expression_l159_159478


namespace collective_earnings_l159_159228

theorem collective_earnings:
  let lloyd_hours := 10.5
  let mary_hours := 12.0
  let tom_hours := 7.0
  let lloyd_normal_hours := 7.5
  let mary_normal_hours := 8.0
  let tom_normal_hours := 9.0
  let lloyd_rate := 4.5
  let mary_rate := 5.0
  let tom_rate := 6.0
  let lloyd_overtime_rate := 2.5 * lloyd_rate
  let mary_overtime_rate := 3.0 * mary_rate
  let tom_overtime_rate := 2.0 * tom_rate
  let lloyd_earnings := (lloyd_normal_hours * lloyd_rate) + ((lloyd_hours - lloyd_normal_hours) * lloyd_overtime_rate)
  let mary_earnings := (mary_normal_hours * mary_rate) + ((mary_hours - mary_normal_hours) * mary_overtime_rate)
  let tom_earnings := (tom_hours * tom_rate)
  let total_earnings := lloyd_earnings + mary_earnings + tom_earnings
  total_earnings = 209.50 := by
  sorry

end collective_earnings_l159_159228


namespace sum_of_three_numbers_l159_159661

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a + b = 36) 
  (h2 : b + c = 55) 
  (h3 : c + a = 60) : 
  a + b + c = 75.5 := 
by 
  sorry

end sum_of_three_numbers_l159_159661


namespace blocks_left_l159_159533

/-- Problem: Randy has 78 blocks. He uses 19 blocks to build a tower. Prove that he has 59 blocks left. -/
theorem blocks_left (initial_blocks : ℕ) (used_blocks : ℕ) (remaining_blocks : ℕ) : initial_blocks = 78 → used_blocks = 19 → remaining_blocks = initial_blocks - used_blocks → remaining_blocks = 59 :=
by
  sorry

end blocks_left_l159_159533


namespace option_b_is_factorization_l159_159740

theorem option_b_is_factorization (m : ℝ) :
  m^2 - 1 = (m + 1) * (m - 1) :=
sorry

end option_b_is_factorization_l159_159740


namespace base_b_representation_l159_159482

theorem base_b_representation (b : ℕ) : (2 * b + 9)^2 = 7 * b^2 + 3 * b + 4 → b = 14 := 
sorry

end base_b_representation_l159_159482


namespace simplified_expression_value_l159_159518

theorem simplified_expression_value (x : ℝ) (h : x = -2) :
  (x - 2)^2 - 4 * x * (x - 1) + (2 * x + 1) * (2 * x - 1) = 7 := 
  by
    -- We are given x = -2
    simp [h]
    -- sorry added to skip the actual solution in Lean
    sorry

end simplified_expression_value_l159_159518


namespace remainder_13_pow_51_mod_5_l159_159420

theorem remainder_13_pow_51_mod_5 : 13^51 % 5 = 2 := by
  sorry

end remainder_13_pow_51_mod_5_l159_159420


namespace standard_eq_of_largest_circle_l159_159221

theorem standard_eq_of_largest_circle 
  (m : ℝ)
  (hm : 0 < m) :
  ∃ r : ℝ, 
  (∀ x y : ℝ, (x^2 + (y - 1)^2 = 8) ↔ 
      (x^2 + (y - 1)^2 = r)) :=
sorry

end standard_eq_of_largest_circle_l159_159221


namespace find_a_b_l159_159018

theorem find_a_b (a b : ℤ) (h : ∀ x : ℤ, (x - 2) * (x + 3) = x^2 + a * x + b) : a = 1 ∧ b = -6 :=
by
  sorry

end find_a_b_l159_159018


namespace euler_criterion_l159_159857

theorem euler_criterion (p : ℕ) (a : ℕ) (hp : Nat.Prime p) (hp_gt_two : p > 2) (ha : 1 ≤ a ∧ a ≤ p - 1) : 
  (∃ b : ℕ, b^2 % p = a % p) ↔ a^((p - 1) / 2) % p = 1 :=
sorry

end euler_criterion_l159_159857


namespace line_equation_l159_159775

theorem line_equation (A : (ℝ × ℝ)) (hA_x : A.1 = 2) (hA_y : A.2 = 0)
  (h_intercept : ∀ B : (ℝ × ℝ), B.1 = 0 → 2 * B.1 + B.2 + 2 = 0 → B = (0, -2)) :
  ∃ (l : ℝ × ℝ → Prop), (l A ∧ l (0, -2)) ∧ 
    (∀ x y : ℝ, l (x, y) ↔ x - y - 2 = 0) :=
by
  sorry

end line_equation_l159_159775


namespace remaining_family_member_age_l159_159335

variable (total_age father_age sister_age : ℕ) (remaining_member_age : ℕ)

def mother_age := father_age - 2
def brother_age := father_age / 2
def known_total_age := father_age + mother_age + brother_age + sister_age

theorem remaining_family_member_age : 
  total_age = 200 ∧ 
  father_age = 60 ∧ 
  sister_age = 40 ∧ 
  known_total_age = total_age - remaining_member_age → 
  remaining_member_age = 12 := by
  sorry

end remaining_family_member_age_l159_159335


namespace all_inequalities_hold_l159_159602

variables (a b c x y z : ℝ)

-- Conditions
def condition1 : Prop := x^2 < a^2
def condition2 : Prop := y^2 < b^2
def condition3 : Prop := z^2 < c^2

-- Inequalities to prove
def inequality1 : Prop := x^2 * y^2 + y^2 * z^2 + z^2 * x^2 < a^2 * b^2 + b^2 * c^2 + c^2 * a^2
def inequality2 : Prop := x^4 + y^4 + z^4 < a^4 + b^4 + c^4
def inequality3 : Prop := x^2 * y^2 * z^2 < a^2 * b^2 * c^2

theorem all_inequalities_hold (h1 : condition1 a x) (h2 : condition2 b y) (h3 : condition3 c z) :
  inequality1 a b c x y z ∧ inequality2 a b c x y z ∧ inequality3 a b c x y z := by
  sorry

end all_inequalities_hold_l159_159602


namespace students_not_in_either_l159_159468

theorem students_not_in_either (total_students chemistry_students biology_students both_subjects neither_subjects : ℕ) 
  (h1 : total_students = 120) 
  (h2 : chemistry_students = 75) 
  (h3 : biology_students = 50) 
  (h4 : both_subjects = 15) 
  (h5 : neither_subjects = total_students - (chemistry_students - both_subjects + biology_students - both_subjects + both_subjects)) : 
  neither_subjects = 10 := 
by 
  sorry

end students_not_in_either_l159_159468


namespace min_value_f_range_m_l159_159378

-- Part I: Prove that the minimum value of f(a) = a^2 + 2/a for a > 0 is 3
theorem min_value_f (a : ℝ) (h : a > 0) : a^2 + 2 / a ≥ 3 :=
sorry

-- Part II: Prove the range of m given the inequality for any positive real number a
theorem range_m (m : ℝ) : (∀ (a : ℝ), a > 0 → a^3 + 2 ≥ 3 * a * (|m - 1| - |2 * m + 3|)) → (m ≤ -3 ∨ m ≥ -1) :=
sorry

end min_value_f_range_m_l159_159378


namespace part_a_part_b_l159_159202

-- Part (a)
theorem part_a (a b : ℕ) (h : (3 * a + b) % 10 = (3 * b + a) % 10) : ¬(a % 10 = b % 10) :=
by sorry

-- Part (b)
theorem part_b (a b c : ℕ)
  (h1 : (2 * a + b) % 10 = (2 * b + c) % 10)
  (h2 : (2 * b + c) % 10 = (2 * c + a) % 10)
  (h3 : (2 * c + a) % 10 = (2 * a + b) % 10) :
  (a % 10 = b % 10) ∧ (b % 10 = c % 10) ∧ (c % 10 = a % 10) :=
by sorry

end part_a_part_b_l159_159202


namespace mixed_fractions_product_l159_159517

theorem mixed_fractions_product :
  ∃ X Y : ℤ, (5 * X + 1) / X * (2 * Y + 1) / 2 = 43 ∧ X = 17 ∧ Y = 8 :=
by
  use 17, 8
  simp
  sorry

end mixed_fractions_product_l159_159517


namespace person_reaches_before_bus_l159_159596

theorem person_reaches_before_bus (dist : ℝ) (speed1 speed2 : ℝ) (miss_time_minutes : ℝ) :
  dist = 2.2 → speed1 = 3 → speed2 = 6 → miss_time_minutes = 12 →
  ((60 : ℝ) * (dist/speed1) - miss_time_minutes) - ((60 : ℝ) * (dist/speed2)) = 10 :=
by
  intro h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end person_reaches_before_bus_l159_159596


namespace fraction_of_students_with_buddy_l159_159516

variable (s n : ℕ)

theorem fraction_of_students_with_buddy (h : n = 4 * s / 3) : 
  (n / 4 + s / 3) / (n + s) = 2 / 7 :=
by
  sorry

end fraction_of_students_with_buddy_l159_159516


namespace sum_of_coefficients_l159_159943

def u (n : ℕ) : ℕ := 
  match n with
  | 0 => 6 -- Assume the sequence starts at u_0 for easier indexing
  | n + 1 => u n + 5 + 2 * n

theorem sum_of_coefficients (u : ℕ → ℕ) : 
  (∀ n, u (n + 1) = u n + 5 + 2 * n) ∧ u 1 = 6 → 
  (∃ a b c : ℕ, (∀ n, u n = a * n^2 + b * n + c) ∧ a + b + c = 6) := 
by
  sorry

end sum_of_coefficients_l159_159943


namespace part_a_part_b_l159_159899

-- Step d: Lean statements for the proof problems
theorem part_a (p : ℕ) (hp : Nat.Prime p) : ∃ a b : ℕ, 0 < a ∧ 0 < b ∧ (a^2 + b^2 + 2018) % p = 0 :=
by {
  sorry
}

theorem part_b (p : ℕ) (hp : Nat.Prime p) : (∃ a b : ℕ, 0 < a ∧ 0 < b ∧ (a^2 + b^2 + 2018) % p = 0 ∧ a % p ≠ 0 ∧ b % p ≠ 0) ↔ p ≠ 3 :=
by {
  sorry
}

end part_a_part_b_l159_159899


namespace last_card_in_box_l159_159545

-- Define the zigzag pattern
def card_position (n : Nat) : Nat :=
  let cycle_pos := n % 12
  if cycle_pos = 0 then
    12
  else
    cycle_pos

def box_for_card (pos : Nat) : Nat :=
  if pos ≤ 7 then
    pos
  else
    14 - pos

theorem last_card_in_box : box_for_card (card_position 2015) = 3 := by
  sorry

end last_card_in_box_l159_159545


namespace arithmetic_sequence_sum_l159_159777

theorem arithmetic_sequence_sum (a : ℕ → ℕ) (d : ℕ) 
(h : a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 420) 
(h_a : ∀ n, a n = a 1 + (n - 1) * d) : a 2 + a 10 = 120 :=
by
  sorry

end arithmetic_sequence_sum_l159_159777


namespace probability_adjacent_points_l159_159624

open Finset

-- Define the hexagon points and adjacency relationship
def hexagon_points : Finset ℕ := {0, 1, 2, 3, 4, 5}

def adjacent (a b : ℕ) : Prop :=
  (a = b + 1 ∨ a = b - 1 ∨ (a = 0 ∧ b = 5) ∨ (a = 5 ∧ b = 0))

-- Total number of ways to choose 2 points from 6 points
def total_pairs := (hexagon_points.card.choose 2)

-- Number of pairs that are adjacent
def favorable_pairs := (6 : ℕ) -- Each point has exactly 2 adjacent points, counted twice

-- The probability of selecting two adjacent points
theorem probability_adjacent_points : (favorable_pairs : ℚ) / total_pairs = 2 / 5 :=
by {
  sorry
}

end probability_adjacent_points_l159_159624


namespace stratified_sample_selection_l159_159623

def TotalStudents : ℕ := 900
def FirstYearStudents : ℕ := 300
def SecondYearStudents : ℕ := 200
def ThirdYearStudents : ℕ := 400
def SampleSize : ℕ := 45
def SamplingRatio : ℚ := 1 / 20

theorem stratified_sample_selection :
  (FirstYearStudents * SamplingRatio = 15) ∧
  (SecondYearStudents * SamplingRatio = 10) ∧
  (ThirdYearStudents * SamplingRatio = 20) :=
by
  sorry

end stratified_sample_selection_l159_159623


namespace ellipse_hyperbola_tangent_l159_159002

theorem ellipse_hyperbola_tangent (m : ℝ) :
  (∃ (x y : ℝ), x^2 + 9 * y^2 = 9 ∧ x^2 - m * (y - 2)^2 = 4) →
  m = 45 / 31 :=
by sorry

end ellipse_hyperbola_tangent_l159_159002


namespace inverse_function_value_l159_159690

def g (x : ℝ) : ℝ := 4 * x ^ 3 - 5

theorem inverse_function_value (x : ℝ) : g x = -1 ↔ x = 1 :=
by
  sorry

end inverse_function_value_l159_159690


namespace max_green_socks_l159_159134

theorem max_green_socks (g y : ℕ) (h1 : g + y ≤ 2025)
  (h2 : (g * (g - 1))/(g + y) * (g + y - 1) = 1/3) : 
  g ≤ 990 := 
sorry

end max_green_socks_l159_159134


namespace ravi_money_l159_159457

theorem ravi_money (n q d : ℕ) (h1 : q = n + 2) (h2 : d = q + 4) (h3 : n = 6) :
  (n * 5 + q * 25 + d * 10) = 350 := by
  sorry

end ravi_money_l159_159457


namespace min_value_l159_159364

noncomputable def minimum_value (a b c : ℝ) : ℝ :=
3 * a + 6 * b + 12 * c

theorem min_value (a b c : ℝ) (h : 9 * a ^ 2 + 4 * b ^ 2 + 36 * c ^ 2 = 4) :
  minimum_value a b c = -2 * Real.sqrt 14 := sorry

end min_value_l159_159364


namespace dragons_total_games_l159_159111

theorem dragons_total_games (y x : ℕ) (h1 : x = 60 * y / 100) (h2 : (x + 8) = 55 * (y + 11) / 100) : y + 11 = 50 :=
by
  sorry

end dragons_total_games_l159_159111


namespace man_speed_is_correct_l159_159877

noncomputable def train_length : ℝ := 275
noncomputable def train_speed_kmh : ℝ := 60
noncomputable def time_seconds : ℝ := 14.998800095992323

noncomputable def train_speed_ms : ℝ := train_speed_kmh * (1000 / 3600)
noncomputable def relative_speed_ms : ℝ := train_length / time_seconds
noncomputable def man_speed_ms : ℝ := relative_speed_ms - train_speed_ms
noncomputable def man_speed_kmh : ℝ := man_speed_ms * (3600 / 1000)
noncomputable def expected_man_speed_kmh : ℝ := 6.006

theorem man_speed_is_correct : abs (man_speed_kmh - expected_man_speed_kmh) < 0.001 :=
by
  -- proof goes here
  sorry

end man_speed_is_correct_l159_159877


namespace probability_more_sons_or_daughters_correct_l159_159950

noncomputable def probability_more_sons_or_daughters : ℚ :=
  let total_combinations := (2 : ℕ) ^ 8
  let equal_sons_daughters := Nat.choose 8 4
  let more_sons_or_daughters := total_combinations - equal_sons_daughters
  more_sons_or_daughters / total_combinations

theorem probability_more_sons_or_daughters_correct :
  probability_more_sons_or_daughters = 93 / 128 := by
  sorry 

end probability_more_sons_or_daughters_correct_l159_159950


namespace num_children_in_family_l159_159409

def regular_ticket_cost := 15
def elderly_ticket_cost := 10
def adult_ticket_cost := 12
def child_ticket_cost := adult_ticket_cost - 5
def total_money_handled := 3 * 50
def change_received := 3
def num_adults := 4
def num_elderly := 2
def total_cost_for_adults := num_adults * adult_ticket_cost
def total_cost_for_elderly := num_elderly * elderly_ticket_cost
def total_cost_of_tickets := total_money_handled - change_received

theorem num_children_in_family : ∃ (num_children : ℕ), 
  total_cost_of_tickets = total_cost_for_adults + total_cost_for_elderly + num_children * child_ticket_cost ∧ 
  num_children = 11 := 
by
  sorry

end num_children_in_family_l159_159409


namespace matrix_vector_subtraction_l159_159473

open Matrix

variable {α : Type*} [AddCommGroup α] [Module ℝ α]

def matrix_mul_vector (M : Matrix (Fin 2) (Fin 2) ℝ) (v : Fin 2 → ℝ) : Fin 2 → ℝ :=
  M.mulVec v

theorem matrix_vector_subtraction (M : Matrix (Fin 2) (Fin 2) ℝ) (v w : Fin 2 → ℝ)
  (hv : matrix_mul_vector M v = ![4, 6])
  (hw : matrix_mul_vector M w = ![5, -4]) :
  matrix_mul_vector M (v - (2 : ℝ) • w) = ![-6, 14] :=
sorry

end matrix_vector_subtraction_l159_159473


namespace mod_product_l159_159346

theorem mod_product : (198 * 955) % 50 = 40 :=
by sorry

end mod_product_l159_159346


namespace cyclist_trip_time_l159_159168

variable (a v : ℝ)
variable (h1 : a / v = 5)

theorem cyclist_trip_time
  (increase_factor : ℝ := 1.25) :
  (a / (2 * v) + a / (2 * (increase_factor * v)) = 4.5) :=
sorry

end cyclist_trip_time_l159_159168


namespace least_multiple_greater_than_500_l159_159322

theorem least_multiple_greater_than_500 : ∃ n : ℕ, n > 500 ∧ n % 32 = 0 := by
  let n := 512
  have h1 : n > 500 := by 
    -- proof omitted, as we're not solving the problem here
    sorry
  have h2 : n % 32 = 0 := by 
    -- proof omitted
    sorry
  exact ⟨n, h1, h2⟩

end least_multiple_greater_than_500_l159_159322


namespace arthur_bought_hamburgers_on_first_day_l159_159643

-- Define the constants and parameters
def D : ℕ := 1
def H : ℕ := 2
def total_cost_day1 : ℕ := 10
def total_cost_day2 : ℕ := 7

-- Define the equation representing the transactions
def equation_day1 (h : ℕ) := H * h + 4 * D = total_cost_day1
def equation_day2 := 2 * H + 3 * D = total_cost_day2

-- The theorem we need to prove: the number of hamburgers h bought on the first day is 3
theorem arthur_bought_hamburgers_on_first_day (h : ℕ) (hd1 : equation_day1 h) (hd2 : equation_day2) : h = 3 := 
by 
  sorry

end arthur_bought_hamburgers_on_first_day_l159_159643


namespace workout_goal_l159_159493

def monday_situps : ℕ := 12
def tuesday_situps : ℕ := 19
def wednesday_situps_needed : ℕ := 59

theorem workout_goal : monday_situps + tuesday_situps + wednesday_situps_needed = 90 := by
  sorry

end workout_goal_l159_159493


namespace simple_interest_years_l159_159170

theorem simple_interest_years (P : ℝ) (hP : P > 0) (R : ℝ := 2.5) (SI : ℝ := P / 5) : 
  ∃ T : ℝ, P * R * T / 100 = SI ∧ T = 8 :=
by
  sorry

end simple_interest_years_l159_159170


namespace populations_equal_in_years_l159_159640

-- Definitions
def populationX (n : ℕ) : ℤ := 68000 - 1200 * n
def populationY (n : ℕ) : ℤ := 42000 + 800 * n

-- Statement to prove
theorem populations_equal_in_years : ∃ n : ℕ, populationX n = populationY n ∧ n = 13 :=
sorry

end populations_equal_in_years_l159_159640


namespace chord_length_l159_159827

theorem chord_length (r : ℝ) (h : r = 15) :
  ∃ (cd : ℝ), cd = 13 * Real.sqrt 3 :=
by
  sorry

end chord_length_l159_159827


namespace find_BM_length_l159_159145

variables (A M C : Type) (MA CA BC BM : ℝ) (x d h : ℝ)

-- Conditions
def condition1 : Prop := MA + (BC - BM) = 2 * CA
def condition2 : Prop := MA = x
def condition3 : Prop := CA = d
def condition4 : Prop := BC = h

-- The proof problem statement
theorem find_BM_length (A M C : Type) (MA CA BC BM : ℝ) (x d h : ℝ)
  (h1 : condition1 MA CA BC BM)
  (h2 : condition2 MA x)
  (h3 : condition3 CA d)
  (h4 : condition4 BC h) :
  BM = 2 * d :=
sorry

end find_BM_length_l159_159145


namespace sequence_sum_n_eq_21_l159_159537

theorem sequence_sum_n_eq_21 (a : ℕ → ℕ) (S : ℕ → ℕ) (n : ℕ) 
  (h1 : a 1 = 1)
  (h2 : ∀ k, a (k + 1) = a k + 1)
  (h3 : ∀ n, S n = (n * (n + 1)) / 2)
  (h4 : S n = 21) :
  n = 6 :=
sorry

end sequence_sum_n_eq_21_l159_159537


namespace total_selling_price_l159_159750

def selling_price_A (purchase_price_A : ℝ) : ℝ :=
  purchase_price_A - (0.15 * purchase_price_A)

def selling_price_B (purchase_price_B : ℝ) : ℝ :=
  purchase_price_B + (0.10 * purchase_price_B)

def selling_price_C (purchase_price_C : ℝ) : ℝ :=
  purchase_price_C - (0.05 * purchase_price_C)

theorem total_selling_price 
  (purchase_price_A : ℝ)
  (purchase_price_B : ℝ)
  (purchase_price_C : ℝ)
  (loss_A : ℝ := 0.15)
  (gain_B : ℝ := 0.10)
  (loss_C : ℝ := 0.05)
  (total_price := selling_price_A purchase_price_A + selling_price_B purchase_price_B + selling_price_C purchase_price_C) :
  purchase_price_A = 1400 → purchase_price_B = 2500 → purchase_price_C = 3200 →
  total_price = 6980 :=
by sorry

end total_selling_price_l159_159750


namespace find_number_l159_159414

def number_condition (N : ℝ) : Prop := 
  0.20 * 0.15 * 0.40 * 0.30 * 0.50 * N = 180

theorem find_number (N : ℝ) (h : number_condition N) : N = 1000000 :=
sorry

end find_number_l159_159414


namespace A_greater_than_B_l159_159384

theorem A_greater_than_B (A B : ℝ) (h₁ : A * 4 = B * 5) (h₂ : A ≠ 0) (h₃ : B ≠ 0) : A > B :=
by
  sorry

end A_greater_than_B_l159_159384


namespace least_n_froods_score_l159_159618

theorem least_n_froods_score (n : ℕ) : (n * (n + 1) / 2 > 12 * n) ↔ (n > 23) := 
by 
  sorry

end least_n_froods_score_l159_159618


namespace ball_hits_ground_at_time_l159_159375

theorem ball_hits_ground_at_time :
  ∃ t : ℚ, -9.8 * t^2 + 5.6 * t + 10 = 0 ∧ t = 131 / 98 :=
by
  sorry

end ball_hits_ground_at_time_l159_159375


namespace solve_floor_equation_l159_159944

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

theorem solve_floor_equation :
  (∃ x : ℝ, (floor ((x - 1) / 2))^2 + 2 * x + 2 = 0) → x = -3 :=
by
  sorry

end solve_floor_equation_l159_159944


namespace initial_average_mark_l159_159009

theorem initial_average_mark (A : ℝ) (n : ℕ) (excluded_avg remaining_avg : ℝ) :
  n = 25 →
  excluded_avg = 40 →
  remaining_avg = 90 →
  (A * n = (n - 5) * remaining_avg + 5 * excluded_avg) →
  A = 80 :=
by
  intros hn_hexcluded_avg hremaining_avg htotal_correct
  sorry

end initial_average_mark_l159_159009


namespace h_at_8_l159_159821

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x + 2

noncomputable def h (x : ℝ) : ℝ :=
  let a := 1
  let b := 1
  let c := 2
  (1/2) * (x - a^3) * (x - b^3) * (x - c^3)

theorem h_at_8 : h 8 = 147 := 
by 
  sorry

end h_at_8_l159_159821


namespace original_wage_l159_159081

theorem original_wage (W : ℝ) 
  (h1: 1.40 * W = 28) : 
  W = 20 :=
sorry

end original_wage_l159_159081


namespace transformed_curve_l159_159148

def curve_C (x y : ℝ) := (x - y)^2 + y^2 = 1

theorem transformed_curve (x y : ℝ) (A : Matrix (Fin 2) (Fin 2) ℝ) :
    A = ![![2, -2], ![0, 1]] →
    (∃ (x0 y0 : ℝ), curve_C x0 y0 ∧ x = 2 * x0 - 2 * y0 ∧ y = y0) →
    (∃ (x y : ℝ), (x^2 / 4 + y^2 = 1)) :=
by
  -- Proof to be completed
  sorry

end transformed_curve_l159_159148


namespace sheep_transaction_gain_l159_159871

noncomputable def percent_gain (cost_per_sheep total_sheep sold_sheep remaining_sheep : ℕ) : ℚ :=
let total_cost := (cost_per_sheep : ℚ) * total_sheep
let initial_revenue := total_cost
let price_per_sheep := initial_revenue / sold_sheep
let remaining_revenue := remaining_sheep * price_per_sheep
let total_revenue := initial_revenue + remaining_revenue
let profit := total_revenue - total_cost
(profit / total_cost) * 100

theorem sheep_transaction_gain :
  percent_gain 1 1000 950 50 = -47.37 := sorry

end sheep_transaction_gain_l159_159871


namespace perpendicular_line_through_point_l159_159734

theorem perpendicular_line_through_point (a b c : ℝ) (hx : a = 2) (hy : b = -1) (hd : c = 3) :
  ∃ k d : ℝ, (k, d) = (-a / b, (a * 1 + b * (1 - c))) ∧ (b * -1, a * -1 + d, -a) = (1, 2, 3) :=
by
  sorry

end perpendicular_line_through_point_l159_159734


namespace reciprocal_neg_sqrt_2_l159_159693

theorem reciprocal_neg_sqrt_2 : 1 / (-Real.sqrt 2) = -Real.sqrt 2 / 2 :=
by
  sorry

end reciprocal_neg_sqrt_2_l159_159693


namespace hash_value_is_minus_15_l159_159014

def hash (a b c : ℝ) : ℝ := b^2 - 3 * a * c

theorem hash_value_is_minus_15 : hash 2 3 4 = -15 :=
by
  sorry

end hash_value_is_minus_15_l159_159014


namespace isosceles_triangle_condition_l159_159580

-- Theorem statement
theorem isosceles_triangle_condition (N : ℕ) (h : N > 2) : 
  (∃ N1 : ℕ, N = N1 ∧ N1 = 10) ∨ (∃ N2 : ℕ, N = N2 ∧ N2 = 11) :=
by sorry

end isosceles_triangle_condition_l159_159580


namespace complex_poly_root_exists_l159_159762

noncomputable def polynomial_has_complex_root (P : Polynomial ℂ) : Prop :=
  ∃ z : ℂ, P.eval z = 0

theorem complex_poly_root_exists (P : Polynomial ℂ) : polynomial_has_complex_root P :=
sorry

end complex_poly_root_exists_l159_159762


namespace mark_more_hours_than_kate_l159_159029

theorem mark_more_hours_than_kate {K : ℕ} (h1 : K + 2 * K + 6 * K = 117) :
  6 * K - K = 65 :=
by
  sorry

end mark_more_hours_than_kate_l159_159029


namespace bread_cost_equality_l159_159993

variable (B : ℝ)
variable (C1 : B + 3 + 2 * B = 9)  -- $3 for butter, 2B for juice, total spent is 9 dollars

theorem bread_cost_equality : B = 2 :=
by
  sorry

end bread_cost_equality_l159_159993


namespace subtract_square_l159_159905

theorem subtract_square (n : ℝ) (h : n = 68.70953354520753) : (n^2 - 20^2) = 4321.000000000001 := by
  sorry

end subtract_square_l159_159905


namespace hyperbola_equation_l159_159903

theorem hyperbola_equation :
  ∃ (b : ℝ), (∀ (x y : ℝ), ((x = 2) ∧ (y = 2)) →
    ((x^2 / 5) - (y^2 / b^2) = 1)) ∧
    (∀ x y, (y = (2 / Real.sqrt 5) * x) ∨ (y = -(2 / Real.sqrt 5) * x) → 
    (∀ (a b : ℝ), (a = 2) → (b = 2) →
      (b^2 = 4) → ((5 * y^2 / 4) - x^2 = 1))) :=
sorry

end hyperbola_equation_l159_159903


namespace find_four_digit_squares_l159_159577

theorem find_four_digit_squares (N : ℕ) (a b : ℕ) 
    (h1 : 100 ≤ N ∧ N < 10000)
    (h2 : 10 ≤ a ∧ a < 100)
    (h3 : 0 ≤ b ∧ b < 100)
    (h4 : N = 100 * a + b)
    (h5 : N = (a + b) ^ 2) : 
    N = 9801 ∨ N = 3025 ∨ N = 2025 :=
    sorry

end find_four_digit_squares_l159_159577


namespace arithmetic_sequence_inequality_l159_159087

variable {α : Type*} [OrderedRing α]

theorem arithmetic_sequence_inequality
  (a : ℕ → α) (d : α)
  (h_arith_seq : ∀ n, a (n + 1) = a n + d)
  (h_d_pos : d > 0)
  (n : ℕ)
  (h_n_gt_1 : n > 1) :
  a 1 * a (n + 1) < a 2 * a n := 
sorry

end arithmetic_sequence_inequality_l159_159087


namespace least_three_digit_multiple_13_l159_159336

theorem least_three_digit_multiple_13 : 
  ∃ n : ℕ, (n ≥ 100) ∧ (∃ k : ℕ, n = 13 * k) ∧ ∀ m, m < n → (m < 100 ∨ ¬∃ k : ℕ, m = 13 * k) :=
by
  sorry

end least_three_digit_multiple_13_l159_159336


namespace geometric_sequence_common_ratio_l159_159504

theorem geometric_sequence_common_ratio
  (a_n : ℕ → ℝ)
  (q : ℝ)
  (h1 : a_n 3 = 7)
  (h2 : a_n 1 + a_n 2 + a_n 3 = 21) :
  q = 1 ∨ q = -1 / 2 :=
sorry

end geometric_sequence_common_ratio_l159_159504


namespace man_double_son_age_in_2_years_l159_159729

def present_age_son : ℕ := 25
def age_difference : ℕ := 27
def years_to_double_age : ℕ := 2

theorem man_double_son_age_in_2_years 
  (S : ℕ := present_age_son)
  (M : ℕ := S + age_difference)
  (Y : ℕ := years_to_double_age) : 
  M + Y = 2 * (S + Y) :=
by sorry

end man_double_son_age_in_2_years_l159_159729


namespace store_A_has_highest_capacity_l159_159937

noncomputable def total_capacity_A : ℕ := 5 * 6 * 9
noncomputable def total_capacity_B : ℕ := 8 * 4 * 7
noncomputable def total_capacity_C : ℕ := 10 * 3 * 8

theorem store_A_has_highest_capacity : total_capacity_A = 270 ∧ total_capacity_A > total_capacity_B ∧ total_capacity_A > total_capacity_C := 
by 
  -- Proof skipped with a placeholder
  sorry

end store_A_has_highest_capacity_l159_159937


namespace problem1_l159_159894

   theorem problem1 : (Real.sqrt (9 / 4) + |2 - Real.sqrt 3| - (64 : ℝ) ^ (1 / 3) + 2⁻¹) = -Real.sqrt 3 :=
   by
     sorry
   
end problem1_l159_159894


namespace sum_of_numbers_Carolyn_removes_l159_159816

noncomputable def game_carolyn_paul_sum : ℕ :=
  let initial_list := [1, 2, 3, 4, 5]
  let removed_by_paul := [3, 4]
  let removed_by_carolyn := [1, 2, 5]
  removed_by_carolyn.sum

theorem sum_of_numbers_Carolyn_removes :
  game_carolyn_paul_sum = 8 :=
by
  sorry

end sum_of_numbers_Carolyn_removes_l159_159816


namespace smallest_multiple_of_seven_l159_159215

/-- The definition of the six-digit number formed by digits a, b, and c followed by "321". -/
def form_number (a b c : ℕ) : ℕ := 100000 * a + 10000 * b + 1000 * c + 321

/-- The condition that a, b, and c are distinct and greater than 3. -/
def valid_digits (a b c : ℕ) : Prop := a > 3 ∧ b > 3 ∧ c > 3 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c

theorem smallest_multiple_of_seven (a b c : ℕ)
  (h_valid : valid_digits a b c)
  (h_mult_seven : form_number a b c % 7 = 0) :
  form_number a b c = 468321 :=
sorry

end smallest_multiple_of_seven_l159_159215


namespace number_of_trumpet_players_l159_159755

def number_of_people_in_orchestra := 21
def number_of_people_known := 1 -- Sebastian
                             + 4 -- Trombone players
                             + 1 -- French horn player
                             + 3 -- Violinists
                             + 1 -- Cellist
                             + 1 -- Contrabassist
                             + 3 -- Clarinet players
                             + 4 -- Flute players
                             + 1 -- Maestro

theorem number_of_trumpet_players : 
  number_of_people_in_orchestra = number_of_people_known + 2 :=
by
  sorry

end number_of_trumpet_players_l159_159755


namespace candy_game_solution_l159_159421

open Nat

theorem candy_game_solution 
  (total_candies : ℕ) 
  (nick_candies : ℕ) 
  (tim_candies : ℕ)
  (tim_wins : ℕ)
  (m n : ℕ)
  (htotal : total_candies = 55) 
  (hnick : nick_candies = 30) 
  (htim : tim_candies = 25)
  (htim_wins : tim_wins = 2)
  (hrounds_total : total_candies = nick_candies + tim_candies)
  (hwinner_condition1 : m > n) 
  (hwinner_condition2 : n > 0) 
  (hwinner_candies_total : total_candies = tim_wins * m + (total_candies / (m + n) - tim_wins) * n)
: m = 8 := 
sorry

end candy_game_solution_l159_159421


namespace schoolchildren_chocolate_l159_159793

theorem schoolchildren_chocolate (m d : ℕ) 
  (h1 : 7 * d + 2 * m > 36)
  (h2 : 8 * d + 4 * m < 48) :
  m = 1 ∧ d = 5 :=
by
  sorry

end schoolchildren_chocolate_l159_159793


namespace city_map_scale_l159_159172

theorem city_map_scale 
  (map_length : ℝ) (actual_length_km : ℝ) (actual_length_cm : ℝ) (conversion_factor : ℝ)
  (h1 : map_length = 240) 
  (h2 : actual_length_km = 18)
  (h3 : actual_length_cm = actual_length_km * conversion_factor)
  (h4 : conversion_factor = 100000) :
  map_length / actual_length_cm = 1 / 7500 :=
by
  sorry

end city_map_scale_l159_159172


namespace modulus_sum_l159_159041

def z1 : ℂ := 3 - 5 * Complex.I
def z2 : ℂ := 3 + 5 * Complex.I

theorem modulus_sum : Complex.abs z1 + Complex.abs z2 = 2 * Real.sqrt 34 := 
by 
  sorry

end modulus_sum_l159_159041


namespace solve_quadratic_and_compute_l159_159981

theorem solve_quadratic_and_compute (y : ℝ) (h : 4 * y^2 + 7 = 6 * y + 12) : (8 * y - 2)^2 = 248 := 
sorry

end solve_quadratic_and_compute_l159_159981


namespace distance_to_line_l159_159096

theorem distance_to_line (a : ℝ) (d : ℝ)
  (h1 : d = 6)
  (h2 : |3 * a + 6| / 5 = d) :
  a = 8 ∨ a = -12 :=
by
  sorry

end distance_to_line_l159_159096


namespace power_division_l159_159272

theorem power_division : (19^11 / 19^6 = 247609) := sorry

end power_division_l159_159272


namespace geometric_progression_first_term_and_ratio_l159_159727

theorem geometric_progression_first_term_and_ratio (
  b_1 q : ℝ
) :
  b_1 * (1 + q + q^2) = 21 →
  b_1^2 * (1 + q^2 + q^4) = 189 →
  (b_1 = 12 ∧ q = 1/2) ∨ (b_1 = 3 ∧ q = 2) :=
by
  intros hsum hsumsq
  sorry

end geometric_progression_first_term_and_ratio_l159_159727


namespace length_of_bridge_l159_159726

noncomputable def convert_speed (km_per_hour : ℝ) : ℝ := km_per_hour * (1000 / 3600)

theorem length_of_bridge 
  (train_length : ℝ)
  (train_speed_kmh : ℝ)
  (passing_time : ℝ)
  (total_distance_covered : ℝ)
  (bridge_length : ℝ) :
  train_length = 120 →
  train_speed_kmh = 40 →
  passing_time = 25.2 →
  total_distance_covered = convert_speed train_speed_kmh * passing_time →
  bridge_length = total_distance_covered - train_length →
  bridge_length = 160 := 
by
  intros h1 h2 h3 h4 h5
  sorry

end length_of_bridge_l159_159726


namespace triangle_altitudes_perfect_square_l159_159340

theorem triangle_altitudes_perfect_square
  (a b c : ℤ)
  (h : (2 * (↑a * ↑b * ↑c )) = (2 * (↑a * ↑c ) + 2 * (↑a * ↑b))) :
  ∃ k : ℤ, a^2 + b^2 + c^2 = k^2 :=
by
  sorry

end triangle_altitudes_perfect_square_l159_159340


namespace typing_speed_equation_l159_159000

theorem typing_speed_equation (x : ℕ) (h_pos : x > 0) :
  120 / x = 180 / (x + 6) :=
sorry

end typing_speed_equation_l159_159000


namespace vanessa_deleted_30_files_l159_159703

-- Define the initial conditions
def original_files : Nat := 16 + 48
def files_left : Nat := 34

-- Define the number of files deleted
def files_deleted : Nat := original_files - files_left

-- The theorem to prove the number of files deleted
theorem vanessa_deleted_30_files : files_deleted = 30 := by
  sorry

end vanessa_deleted_30_files_l159_159703


namespace find_gear_p_rpm_l159_159062

def gear_p_rpm (r : ℕ) (gear_p_revs : ℕ) (gear_q_rpm : ℕ) (time_seconds : ℕ) (extra_revs_q_over_p : ℕ) : Prop :=
  r = gear_p_revs * 2

theorem find_gear_p_rpm (r : ℕ) (gear_q_rpm : ℕ) (time_seconds : ℕ) (extra_revs_q_over_p : ℕ) :
  gear_q_rpm = 40 ∧ time_seconds = 30 ∧ extra_revs_q_over_p = 15 ∧ gear_p_revs = 10 / 2 →
  r = 10 :=
by
  sorry

end find_gear_p_rpm_l159_159062


namespace initial_volume_of_solution_l159_159959

theorem initial_volume_of_solution (V : ℝ) :
  (∀ (init_vol : ℝ), 0.84 * init_vol / (init_vol + 26.9) = 0.58) →
  V = 60 :=
by
  intro h
  sorry

end initial_volume_of_solution_l159_159959


namespace sum_multiple_of_three_l159_159786

theorem sum_multiple_of_three (a b : ℤ) (h₁ : ∃ m, a = 6 * m) (h₂ : ∃ n, b = 9 * n) : ∃ k, (a + b) = 3 * k :=
by
  sorry

end sum_multiple_of_three_l159_159786


namespace joan_change_received_l159_159872

/-- Definition of the cat toy cost -/
def cat_toy_cost : ℝ := 8.77

/-- Definition of the cage cost -/
def cage_cost : ℝ := 10.97

/-- Definition of the total cost -/
def total_cost : ℝ := cat_toy_cost + cage_cost

/-- Definition of the payment amount -/
def payment : ℝ := 20.00

/-- Definition of the change received -/
def change_received : ℝ := payment - total_cost

/-- Statement proving that Joan received $0.26 in change -/
theorem joan_change_received : change_received = 0.26 := by
  sorry

end joan_change_received_l159_159872


namespace trig_identity_solution_l159_159494

open Real

theorem trig_identity_solution :
  sin (15 * (π / 180)) * cos (45 * (π / 180)) + sin (105 * (π / 180)) * sin (135 * (π / 180)) = sqrt 3 / 2 :=
by
  -- Placeholder for the proof
  sorry

end trig_identity_solution_l159_159494


namespace rachel_age_when_emily_half_age_l159_159815

theorem rachel_age_when_emily_half_age 
  (E_0 : ℕ) (R_0 : ℕ) (h1 : E_0 = 20) (h2 : R_0 = 24) 
  (age_diff : R_0 - E_0 = 4) : 
  ∃ R : ℕ, ∃ E : ℕ, E = R / 2 ∧ R = E + 4 ∧ R = 8 :=
by
  sorry

end rachel_age_when_emily_half_age_l159_159815


namespace carol_blocks_l159_159714

theorem carol_blocks (initial_blocks : ℕ) (blocks_lost : ℕ) (final_blocks : ℕ) : 
  initial_blocks = 42 → blocks_lost = 25 → final_blocks = initial_blocks - blocks_lost → final_blocks = 17 :=
by 
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end carol_blocks_l159_159714


namespace eighty_five_squared_l159_159227

theorem eighty_five_squared : 85^2 = 7225 := by
  sorry

end eighty_five_squared_l159_159227


namespace original_average_age_l159_159158

-- Definitions based on conditions
def original_strength : ℕ := 12
def new_student_count : ℕ := 12
def new_student_average_age : ℕ := 32
def age_decrease : ℕ := 4
def total_student_count : ℕ := original_strength + new_student_count
def combined_total_age (A : ℕ) : ℕ := original_strength * A + new_student_count * new_student_average_age
def new_average_age (A : ℕ) : ℕ := A - age_decrease

-- Statement of the problem
theorem original_average_age (A : ℕ) (h : combined_total_age A / total_student_count = new_average_age A) : A = 40 := 
by 
  sorry

end original_average_age_l159_159158


namespace exists_colored_right_triangle_l159_159391

theorem exists_colored_right_triangle (color : ℝ × ℝ → ℕ) 
  (h_nonempty_blue  : ∃ p, color p = 0)
  (h_nonempty_green : ∃ p, color p = 1)
  (h_nonempty_red   : ∃ p, color p = 2) :
  ∃ p1 p2 p3 : ℝ × ℝ, 
    (p1 ≠ p2) ∧ (p2 ≠ p3) ∧ (p1 ≠ p3) ∧ 
    ((color p1 = 0) ∧ (color p2 = 1) ∧ (color p3 = 2) ∨ 
     (color p1 = 0) ∧ (color p2 = 2) ∧ (color p3 = 1) ∨ 
     (color p1 = 1) ∧ (color p2 = 0) ∧ (color p3 = 2) ∨ 
     (color p1 = 1) ∧ (color p2 = 2) ∧ (color p3 = 0) ∨ 
     (color p1 = 2) ∧ (color p2 = 0) ∧ (color p3 = 1) ∨ 
     (color p1 = 2) ∧ (color p2 = 1) ∧ (color p3 = 0))
  ∧ ((p1.1 = p2.1 ∧ p2.2 = p3.2) ∨ (p1.2 = p2.2 ∧ p2.1 = p3.1)) :=
sorry

end exists_colored_right_triangle_l159_159391


namespace difference_red_white_l159_159189

/-
Allie picked 100 wildflowers. The categories of flowers are given as below:
- 13 of the flowers were yellow and white
- 17 of the flowers were red and yellow
- 14 of the flowers were red and white
- 16 of the flowers were blue and yellow
- 9 of the flowers were blue and white
- 8 of the flowers were red, blue, and yellow
- 6 of the flowers were red, white, and blue

The goal is to define the number of flowers containing red and white, and
prove that the difference between the number of flowers containing red and 
those containing white is 3.
-/

def total_flowers : ℕ := 100
def yellow_and_white : ℕ := 13
def red_and_yellow : ℕ := 17
def red_and_white : ℕ := 14
def blue_and_yellow : ℕ := 16
def blue_and_white : ℕ := 9
def red_blue_and_yellow : ℕ := 8
def red_white_and_blue : ℕ := 6

def flowers_with_red : ℕ := red_and_yellow + red_and_white + red_blue_and_yellow + red_white_and_blue
def flowers_with_white : ℕ := yellow_and_white + red_and_white + blue_and_white + red_white_and_blue

theorem difference_red_white : flowers_with_red - flowers_with_white = 3 := by
  rw [flowers_with_red, flowers_with_white]
  sorry

end difference_red_white_l159_159189


namespace sharon_distance_l159_159660

noncomputable def usual_speed (x : ℝ) := x / 180
noncomputable def reduced_speed (x : ℝ) := (x / 180) - 0.5

theorem sharon_distance (x : ℝ) (h : 300 = (x / 2) / usual_speed x + (x / 2) / reduced_speed x) : x = 157.5 :=
by sorry

end sharon_distance_l159_159660


namespace probability_red_ball_first_occurrence_l159_159677

theorem probability_red_ball_first_occurrence 
  (P : ℕ → ℝ) : 
  ∃ (P1 P2 P3 P4 : ℝ),
    P 1 = 0.4 ∧ P 2 = 0.3 ∧ P 3 = 0.2 ∧ P 4 = 0.1 :=
  sorry

end probability_red_ball_first_occurrence_l159_159677


namespace tan_theta_sqrt3_l159_159636

theorem tan_theta_sqrt3 (θ : ℝ) 
  (h : Real.cos (40 * (π / 180) - θ) 
     + Real.cos (40 * (π / 180) + θ) 
     + Real.cos (80 * (π / 180) - θ) = 0) 
  : Real.tan θ = -Real.sqrt 3 := 
by
  sorry

end tan_theta_sqrt3_l159_159636


namespace man_son_ratio_in_two_years_l159_159570

noncomputable def man_and_son_age_ratio (M S : ℕ) (h1 : M = S + 25) (h2 : S = 23) : ℕ × ℕ :=
  let S_in_2_years := S + 2
  let M_in_2_years := M + 2
  (M_in_2_years / S_in_2_years, S_in_2_years / S_in_2_years)

theorem man_son_ratio_in_two_years : man_and_son_age_ratio 48 23 (by norm_num) (by norm_num) = (2, 1) :=
  sorry

end man_son_ratio_in_two_years_l159_159570


namespace opposite_of_2023_l159_159050

def opposite (x : Int) : Int :=
  -x

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l159_159050


namespace machines_needed_l159_159511

theorem machines_needed (x Y : ℝ) (R : ℝ) :
  (4 * R * 6 = x) → (M * R * 6 = Y) → M = 4 * Y / x :=
by
  intros h1 h2
  sorry

end machines_needed_l159_159511


namespace inequalities_indeterminate_l159_159808

variable (s x y z : ℝ)

theorem inequalities_indeterminate (h_s : s > 0) (h_ineq : s * x > z * y) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (¬ (x > z)) ∨ (¬ (-x > -z)) ∨ (¬ (s > z / x)) ∨ (¬ (s < y / x)) :=
by sorry

end inequalities_indeterminate_l159_159808


namespace solve_for_x_l159_159825

theorem solve_for_x (x y : ℚ) :
  (x + 1) / (x - 2) = (y^2 + 4*y + 1) / (y^2 + 4*y - 3) →
  x = -(3*y^2 + 12*y - 1) / 2 :=
by
  intro h
  sorry

end solve_for_x_l159_159825


namespace min_value_of_expression_l159_159290

theorem min_value_of_expression (x y : ℝ) (hx : x > y) (hy : y > 0) (hxy : x + y ≤ 2) :
  ∃ m : ℝ, m = (2 / (x + 3 * y) + 1 / (x - y)) ∧ m = (3 + 2 * Real.sqrt 2) / 4 :=
by
  sorry

end min_value_of_expression_l159_159290


namespace algebraic_expression_evaluation_l159_159393

theorem algebraic_expression_evaluation (a b : ℤ) (h : a - 3 * b = -3) : 5 - a + 3 * b = 8 :=
by 
  sorry

end algebraic_expression_evaluation_l159_159393


namespace find_integers_l159_159659

theorem find_integers (x y : ℕ) (h : 2 * x * y = 21 + 2 * x + y) : (x = 1 ∧ y = 23) ∨ (x = 6 ∧ y = 3) :=
by
  sorry

end find_integers_l159_159659


namespace parabola_hyperbola_coincide_directrix_l159_159491

noncomputable def parabola_directrix (p : ℝ) : ℝ := -p / 2
noncomputable def hyperbola_directrix : ℝ := -3 / 2

theorem parabola_hyperbola_coincide_directrix (p : ℝ) (hp : 0 < p) 
  (h_eq : parabola_directrix p = hyperbola_directrix) : p = 3 :=
by
  have hp_directrix : parabola_directrix p = -p / 2 := rfl
  have h_directrix : hyperbola_directrix = -3 / 2 := rfl
  rw [hp_directrix, h_directrix] at h_eq
  sorry

end parabola_hyperbola_coincide_directrix_l159_159491


namespace total_boat_licenses_l159_159694

/-- A state modifies its boat license requirements to include any one of the letters A, M, or S
followed by any six digits. How many different boat licenses can now be issued? -/
theorem total_boat_licenses : 
  let letters := 3
  let digits := 10
  letters * digits^6 = 3000000 := by
  sorry

end total_boat_licenses_l159_159694


namespace selling_price_of_cycle_l159_159531

theorem selling_price_of_cycle
  (cost_price : ℕ)
  (gain_percent_decimal : ℚ)
  (h_cp : cost_price = 850)
  (h_gpd : gain_percent_decimal = 27.058823529411764 / 100) :
  ∃ selling_price : ℚ, selling_price = cost_price * (1 + gain_percent_decimal) ∧ selling_price = 1080 := 
by
  use (cost_price * (1 + gain_percent_decimal))
  sorry

end selling_price_of_cycle_l159_159531


namespace interior_surface_area_is_correct_l159_159472

-- Define the original dimensions of the rectangular sheet
def original_length : ℕ := 40
def original_width : ℕ := 50

-- Define the side length of the square corners
def corner_side : ℕ := 10

-- Define the area of the original sheet
def area_original : ℕ := original_length * original_width

-- Define the area of one square corner
def area_corner : ℕ := corner_side * corner_side

-- Define the total area removed by all four corners
def area_removed : ℕ := 4 * area_corner

-- Define the remaining area after the corners are removed
def area_remaining : ℕ := area_original - area_removed

-- The theorem to be proved
theorem interior_surface_area_is_correct : area_remaining = 1600 := by
  sorry

end interior_surface_area_is_correct_l159_159472


namespace ratio_of_weights_l159_159447

def initial_weight : ℝ := 2
def weight_after_brownies (w : ℝ) : ℝ := w * 3
def weight_after_more_jelly_beans (w : ℝ) : ℝ := w + 2
def final_weight : ℝ := 16
def weight_before_adding_gummy_worms : ℝ := weight_after_more_jelly_beans (weight_after_brownies initial_weight)

theorem ratio_of_weights :
  final_weight / weight_before_adding_gummy_worms = 2 := 
by
  sorry

end ratio_of_weights_l159_159447


namespace samantha_last_name_length_l159_159844

/-
Given:
1. Jamie’s last name "Grey" has 4 letters.
2. If Bobbie took 2 letters off her last name, her last name would have twice the length of Jamie’s last name.
3. Samantha’s last name has 3 fewer letters than Bobbie’s last name.

Prove:
- Samantha's last name contains 7 letters.
-/

theorem samantha_last_name_length : 
  ∀ (Jamie Bobbie Samantha : ℕ),
    Jamie = 4 →
    Bobbie - 2 = 2 * Jamie →
    Samantha = Bobbie - 3 →
    Samantha = 7 :=
by
  intros Jamie Bobbie Samantha hJamie hBobbie hSamantha
  sorry

end samantha_last_name_length_l159_159844


namespace square_land_perimeter_l159_159687

theorem square_land_perimeter (a p : ℝ) (h1 : a = p^2 / 16) (h2 : 5*a = 10*p + 45) : p = 36 :=
by sorry

end square_land_perimeter_l159_159687


namespace slower_bike_longer_time_by_1_hour_l159_159555

/-- Speed of the slower bike in kmph -/
def speed_slow : ℕ := 60

/-- Speed of the faster bike in kmph -/
def speed_fast : ℕ := 64

/-- Distance both bikes travel in km -/
def distance : ℕ := 960

/-- Time taken to travel the distance by a bike going at a certain speed -/
def time (speed : ℕ) : ℕ :=
  distance / speed

/-- Proof that the slower bike takes 1 hour longer to cover the distance compared to the faster bike -/
theorem slower_bike_longer_time_by_1_hour : 
  (time speed_slow) = (time speed_fast) + 1 := by
sorry

end slower_bike_longer_time_by_1_hour_l159_159555


namespace find_a_l159_159033

theorem find_a (a b : ℝ) (h₀ : b = 4) (h₁ : (4, b) ∈ {p | p.snd = 0.75 * p.fst + 1}) 
  (h₂ : (a, 5) ∈ {p | p.snd = 0.75 * p.fst + 1}) (h₃ : (a, b+1) ∈ {p | p.snd = 0.75 * p.fst + 1}) : 
  a = 5.33 :=
by 
  sorry

end find_a_l159_159033


namespace probability_of_point_in_spheres_l159_159295

noncomputable def radius_of_inscribed_sphere (R : ℝ) : ℝ := 2 * R / 3
noncomputable def radius_of_tangent_spheres (R : ℝ) : ℝ := 2 * R / 3

theorem probability_of_point_in_spheres
  (R : ℝ)  -- Radius of the circumscribed sphere
  (r : ℝ := radius_of_inscribed_sphere R)  -- Radius of the inscribed sphere
  (r_t : ℝ := radius_of_tangent_spheres R)  -- Radius of each tangent sphere
  (volume : ℝ := 4/3 * Real.pi * r^3)  -- Volume of each smaller sphere
  (total_small_volume : ℝ := 5 * volume)  -- Total volume of smaller spheres
  (circumsphere_volume : ℝ := 4/3 * Real.pi * (2 * R)^3)  -- Volume of the circumscribed sphere
  : 
  total_small_volume / circumsphere_volume = 5 / 27 :=
by
  sorry

end probability_of_point_in_spheres_l159_159295


namespace sector_angle_measure_l159_159881

theorem sector_angle_measure (r α : ℝ) 
  (h1 : 2 * r + α * r = 6)
  (h2 : (1 / 2) * α * r^2 = 2) :
  α = 1 ∨ α = 4 := 
sorry

end sector_angle_measure_l159_159881


namespace trigonometric_identity_l159_159748

theorem trigonometric_identity :
  4 * Real.cos (10 * (Real.pi / 180)) - Real.tan (80 * (Real.pi / 180)) = -Real.sqrt 3 := 
by 
  sorry

end trigonometric_identity_l159_159748


namespace modulus_of_z_l159_159708

-- Define the given condition
def condition (z : ℂ) : Prop := (z - 3) * (1 - 3 * Complex.I) = 10

-- State the main theorem
theorem modulus_of_z (z : ℂ) (h : condition z) : Complex.abs z = 5 :=
sorry

end modulus_of_z_l159_159708


namespace find_real_number_l159_159483

theorem find_real_number :
    (∃ y : ℝ, y = 3 + (5 / (2 + 5 / (3 + 5 / (2 + 5 / (3 + 5 / (2 + 5 / (3 + 5 / (2 + 5 / (3 + sorry)))))))))) ∧ 
    y = (3 + Real.sqrt 29) / 2 :=
by
  sorry

end find_real_number_l159_159483


namespace verify_final_weights_l159_159446

-- Define the initial weights
def initial_bench_press : ℝ := 500
def initial_squat : ℝ := 400
def initial_deadlift : ℝ := 600

-- Define the weight adjustment transformations for each exercise
def transform_bench_press (w : ℝ) : ℝ :=
  let w1 := w * 0.20
  let w2 := w1 * 1.60
  let w3 := w2 * 0.80
  let w4 := w3 * 3
  w4

def transform_squat (w : ℝ) : ℝ :=
  let w1 := w * 0.50
  let w2 := w1 * 1.40
  let w3 := w2 * 2
  w3

def transform_deadlift (w : ℝ) : ℝ :=
  let w1 := w * 0.70
  let w2 := w1 * 1.80
  let w3 := w2 * 0.60
  let w4 := w3 * 1.50
  w4

-- The final calculated weights for verification
def final_bench_press : ℝ := 384
def final_squat : ℝ := 560
def final_deadlift : ℝ := 680.4

-- Statement of the problem: prove that the transformed weights are as calculated
theorem verify_final_weights : 
  transform_bench_press initial_bench_press = final_bench_press ∧ 
  transform_squat initial_squat = final_squat ∧ 
  transform_deadlift initial_deadlift = final_deadlift := 
by 
  sorry

end verify_final_weights_l159_159446


namespace angle_expr_correct_l159_159958

noncomputable def angle_expr : Real :=
  Real.cos (40 * Real.pi / 180) * Real.cos (160 * Real.pi / 180) +
  Real.sin (40 * Real.pi / 180) * Real.sin (20 * Real.pi / 180)

theorem angle_expr_correct : angle_expr = -1 / 2 := 
by 
   sorry

end angle_expr_correct_l159_159958


namespace distance_from_highest_point_of_sphere_to_bottom_of_glass_l159_159373

theorem distance_from_highest_point_of_sphere_to_bottom_of_glass :
  ∀ (x y : ℝ),
  x^2 = 2 * y →
  0 ≤ y ∧ y < 15 →
  ∃ b : ℝ, (x^2 + (y - b)^2 = 9) ∧ b = 5 ∧ (b + 3 = 8) :=
by
  sorry

end distance_from_highest_point_of_sphere_to_bottom_of_glass_l159_159373


namespace games_required_for_champion_l159_159396

-- Define the number of players in the tournament
def players : ℕ := 512

-- Define the tournament conditions
def single_elimination_tournament (n : ℕ) : Prop :=
  ∀ (g : ℕ), g = n - 1

-- State the theorem that needs to be proven
theorem games_required_for_champion : single_elimination_tournament players :=
by
  sorry

end games_required_for_champion_l159_159396


namespace equilateral_is_cute_specific_triangle_is_cute_find_AB_length_l159_159079

-- Definition of a cute triangle
def is_cute_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = 2 * c^2 ∨ a^2 + c^2 = 2 * b^2 ∨ b^2 + c^2 = 2 * a^2

-- 1. Prove an equilateral triangle is a cute triangle
theorem equilateral_is_cute (a : ℝ) : is_cute_triangle a a a :=
by
  sorry

-- 2. Prove the triangle with sides 4, 2√6, and 2√5 is a cute triangle
theorem specific_triangle_is_cute : is_cute_triangle 4 (2*Real.sqrt 6) (2*Real.sqrt 5) :=
by
  sorry

-- 3. Prove the length of AB for the given right triangle is 2√6 or 2√3
theorem find_AB_length (AB BC : ℝ) (AC : ℝ := 2*Real.sqrt 2) (h_cute : is_cute_triangle AB BC AC) : AB = 2*Real.sqrt 6 ∨ AB = 2*Real.sqrt 3 :=
by
  sorry

end equilateral_is_cute_specific_triangle_is_cute_find_AB_length_l159_159079


namespace f_10_l159_159127

namespace MathProof

variable (f : ℤ → ℤ)

-- Condition 1: f(1) + 1 > 0
axiom cond1 : f 1 + 1 > 0

-- Condition 2: f(x + y) - x * f(y) - y * f(x) = f(x) * f(y) - x - y + x * y for any x, y ∈ ℤ
axiom cond2 : ∀ x y : ℤ, f (x + y) - x * f y - y * f x = f x * f y - x - y + x * y

-- Condition 3: 2 * f(x) = f(x + 1) - x + 1 for any x ∈ ℤ
axiom cond3 : ∀ x : ℤ, 2 * f x = f (x + 1) - x + 1

-- We need to prove f(10) = 1014
theorem f_10 : f 10 = 1014 :=
by
  sorry

end MathProof

end f_10_l159_159127


namespace measure_of_angle_Z_l159_159573

theorem measure_of_angle_Z (X Y Z : ℝ) (h_sum : X + Y + Z = 180) (h_XY : X + Y = 80) : Z = 100 := 
by
  -- The proof is not required.
  sorry

end measure_of_angle_Z_l159_159573


namespace braden_total_money_after_winning_bet_l159_159887

def initial_amount : ℕ := 400
def factor : ℕ := 2
def winnings (initial_amount : ℕ) (factor : ℕ) : ℕ := factor * initial_amount

theorem braden_total_money_after_winning_bet : 
  winnings initial_amount factor + initial_amount = 1200 := 
by
  sorry

end braden_total_money_after_winning_bet_l159_159887


namespace increasing_on_positive_reals_l159_159913

noncomputable def f (x : ℝ) : ℝ := Real.exp x + Real.exp (-x)

theorem increasing_on_positive_reals : ∀ x y : ℝ, 0 < x → 0 < y → x < y → f x < f y := by
  sorry

end increasing_on_positive_reals_l159_159913


namespace zhang_qiu_jian_problem_l159_159181

-- Define the arithmetic sequence
def arithmeticSequence (a1 d : ℚ) (n : ℕ) : ℚ :=
  a1 + (n - 1) * d

-- Sum of first n terms of an arithmetic sequence
def sumArithmeticSequence (a1 d : ℚ) (n : ℕ) : ℚ :=
  n * a1 + (n * (n - 1) / 2) * d

theorem zhang_qiu_jian_problem :
  sumArithmeticSequence 5 (16 / 29) 30 = 390 := 
by 
  sorry

end zhang_qiu_jian_problem_l159_159181


namespace rose_flyers_l159_159153

theorem rose_flyers (total_flyers made: ℕ) (flyers_jack: ℕ) (flyers_left: ℕ) 
(h1 : total_flyers = 1236)
(h2 : flyers_jack = 120)
(h3 : flyers_left = 796)
: total_flyers - flyers_jack - flyers_left = 320 :=
by
  sorry

end rose_flyers_l159_159153


namespace sales_volume_maximum_profit_l159_159401

noncomputable def profit (x : ℝ) : ℝ := (x - 34) * (-2 * x + 296)

theorem sales_volume (x : ℝ) : 200 - 2 * (x - 48) = -2 * x + 296 := by
  sorry

theorem maximum_profit :
  (∀ x : ℝ, profit x ≤ profit 91) ∧ profit 91 = 6498 := by
  sorry

end sales_volume_maximum_profit_l159_159401


namespace tan_sum_l159_159236

theorem tan_sum (x y : ℝ) (h1 : Real.sin x + Real.sin y = 85 / 65) (h2 : Real.cos x + Real.cos y = 60 / 65) :
  Real.tan x + Real.tan y = 17 / 12 :=
sorry

end tan_sum_l159_159236


namespace josh_initial_wallet_l159_159262

noncomputable def initial_wallet_amount (investment final_wallet: ℕ) (stock_increase_percentage: ℕ): ℕ :=
  let investment_value_after_rise := investment + (investment * stock_increase_percentage / 100)
  final_wallet - investment_value_after_rise

theorem josh_initial_wallet : initial_wallet_amount 2000 2900 30 = 300 :=
by
  sorry

end josh_initial_wallet_l159_159262


namespace customers_tipped_count_l159_159965

variable (initial_customers : ℕ)
variable (added_customers : ℕ)
variable (customers_no_tip : ℕ)

def total_customers (initial_customers added_customers : ℕ) : ℕ :=
  initial_customers + added_customers

theorem customers_tipped_count 
  (h_init : initial_customers = 29)
  (h_added : added_customers = 20)
  (h_no_tip : customers_no_tip = 34) :
  (total_customers initial_customers added_customers - customers_no_tip) = 15 :=
by
  sorry

end customers_tipped_count_l159_159965


namespace initial_pineapple_sweets_l159_159826

-- Define constants for initial number of flavored sweets and actions taken
def initial_cherry_sweets : ℕ := 30
def initial_strawberry_sweets : ℕ := 40
def total_remaining_sweets : ℕ := 55

-- Define Aaron's actions
def aaron_eats_half_sweets (n : ℕ) : ℕ := n / 2
def aaron_gives_to_friend : ℕ := 5

-- Calculate remaining sweets after Aaron's actions
def remaining_cherry_sweets : ℕ := initial_cherry_sweets - (aaron_eats_half_sweets initial_cherry_sweets) - aaron_gives_to_friend
def remaining_strawberry_sweets : ℕ := initial_strawberry_sweets - (aaron_eats_half_sweets initial_strawberry_sweets)

-- Define the problem to prove
theorem initial_pineapple_sweets :
  (total_remaining_sweets - (remaining_cherry_sweets + remaining_strawberry_sweets)) * 2 = 50 :=
by sorry -- Placeholder for the actual proof

end initial_pineapple_sweets_l159_159826


namespace range_of_m_l159_159968

noncomputable def f : ℝ → ℝ := sorry

theorem range_of_m (h1 : ∀ x : ℝ, f (-x) = f x)
                   (h2 : ∀ a b : ℝ, a ≠ b → a ≤ 0 → b ≤ 0 → (f a - f b) / (a - b) < 0)
                   (h3 : f (m + 1) < f 2) : 
  ∃ m : ℝ, -3 < m ∧ m < 1 :=
sorry

end range_of_m_l159_159968


namespace balance_scale_comparison_l159_159605

theorem balance_scale_comparison :
  (4 / 3) * Real.pi * (8 : ℝ)^3 > (4 / 3) * Real.pi * (3 : ℝ)^3 + (4 / 3) * Real.pi * (5 : ℝ)^3 :=
by
  sorry

end balance_scale_comparison_l159_159605


namespace sum_of_bases_l159_159128

theorem sum_of_bases (F1 F2 : ℚ) (R1 R2 : ℕ) (hF1_R1 : F1 = (3 * R1 + 7) / (R1^2 - 1) ∧ F2 = (7 * R1 + 3) / (R1^2 - 1))
    (hF1_R2 : F1 = (2 * R2 + 5) / (R2^2 - 1) ∧ F2 = (5 * R2 + 2) / (R2^2 - 1)) : 
    R1 + R2 = 19 := 
sorry

end sum_of_bases_l159_159128


namespace problem_statement_l159_159232

theorem problem_statement (a b c : ℝ) (ha: 0 ≤ a) (hb: 0 ≤ b) (hc: 0 ≤ c) : 
  a * (a - b) * (a - 2 * b) + b * (b - c) * (b - 2 * c) + c * (c - a) * (c - 2 * a) ≥ 0 :=
by
  sorry

end problem_statement_l159_159232


namespace range_of_a_l159_159099

variable (f : ℝ → ℝ)
variable (a : ℝ)

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def is_increasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y

def holds_on_interval (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, (1/2) ≤ x ∧ x ≤ 1 → f (a*x + 1) ≤ f (x - 2)

theorem range_of_a (h1 : is_even f)
                   (h2 : is_increasing_on_nonneg f)
                   (h3 : holds_on_interval f a) :
  -2 ≤ a ∧ a ≤ 0 := 
sorry

end range_of_a_l159_159099


namespace expression_calculation_l159_159603

theorem expression_calculation : 
  (3^1005 + 7^1006)^2 - (3^1005 - 7^1006)^2 = 28 * 21^1005 :=
by
  sorry

end expression_calculation_l159_159603


namespace find_number_l159_159097

theorem find_number 
  (x : ℚ) 
  (h : (3 / 4) * x - (8 / 5) * x + 63 = 12) : 
  x = 60 := 
by
  sorry

end find_number_l159_159097


namespace yellow_paint_quarts_l159_159571

theorem yellow_paint_quarts (ratio_r : ℕ) (ratio_y : ℕ) (ratio_w : ℕ) (qw : ℕ) : 
  ratio_r = 5 → ratio_y = 3 → ratio_w = 7 → qw = 21 → (qw * ratio_y) / ratio_w = 9 :=
by
  -- No proof required, inserting sorry to indicate missing proof
  sorry

end yellow_paint_quarts_l159_159571


namespace original_cookie_price_l159_159912

theorem original_cookie_price (C : ℝ) (h1 : 1.5 * 16 + (C / 2) * 8 = 32) : C = 2 :=
by
  -- Proof omitted
  sorry

end original_cookie_price_l159_159912


namespace intersection_A_B_l159_159407

def setA (x : ℝ) : Prop := 3 * x + 2 > 0
def setB (x : ℝ) : Prop := (x + 1) * (x - 3) > 0
def A : Set ℝ := { x | setA x }
def B : Set ℝ := { x | setB x }

theorem intersection_A_B : A ∩ B = { x | 3 < x } := by
  sorry

end intersection_A_B_l159_159407


namespace range_of_a_l159_159224

theorem range_of_a (a : ℝ) (h : ¬ ∃ x : ℝ, x^2 + (a + 1) * x + 1 ≤ 0) : -3 < a ∧ a < 1 :=
sorry

end range_of_a_l159_159224


namespace rounding_addition_to_tenth_l159_159332

def number1 : Float := 96.23
def number2 : Float := 47.849

theorem rounding_addition_to_tenth (sum : Float) : 
    sum = number1 + number2 →
    Float.round (sum * 10) / 10 = 144.1 :=
by
  intro h
  rw [h]
  norm_num
  sorry -- Skipping the actual rounding proof

end rounding_addition_to_tenth_l159_159332


namespace gcd_polynomial_l159_159471

theorem gcd_polynomial {b : ℤ} (h1 : ∃ k : ℤ, b = 2 * 7786 * k) : 
  Int.gcd (8 * b^2 + 85 * b + 200) (2 * b + 10) = 10 :=
by
  sorry

end gcd_polynomial_l159_159471


namespace inequality_proof_l159_159616

theorem inequality_proof (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x + y + z = 1) : 
  (x + y)^3 / z + (y + z)^3 / x + (z + x)^3 / y + 9 * x * y * z ≥ 9 * (x * y + y * z + z * x) :=
by 
  sorry

end inequality_proof_l159_159616


namespace number_of_small_gardens_l159_159675

def totalSeeds : ℕ := 85
def tomatoSeeds : ℕ := 42
def capsicumSeeds : ℕ := 26
def cucumberSeeds : ℕ := 17

def plantedTomatoSeeds : ℕ := 24
def plantedCucumberSeeds : ℕ := 17

def remainingTomatoSeeds : ℕ := tomatoSeeds - plantedTomatoSeeds
def remainingCapsicumSeeds : ℕ := capsicumSeeds
def remainingCucumberSeeds : ℕ := cucumberSeeds - plantedCucumberSeeds

def seedsInSmallGardenTomato : ℕ := 2
def seedsInSmallGardenCapsicum : ℕ := 1
def seedsInSmallGardenCucumber : ℕ := 1

theorem number_of_small_gardens : (remainingTomatoSeeds / seedsInSmallGardenTomato = 9) :=
by 
  sorry

end number_of_small_gardens_l159_159675


namespace fraction_simplification_l159_159325

theorem fraction_simplification (x y z : ℝ) (h : x + y + z ≠ 0) :
  (x^2 + y^2 - z^2 + 2 * x * y) / (x^2 + z^2 - y^2 + 2 * x * z) = (x + y - z) / (x + z - y) :=
by
  sorry

end fraction_simplification_l159_159325


namespace david_and_maria_ages_l159_159916

theorem david_and_maria_ages 
  (D Y M : ℕ)
  (h1 : Y = D + 7)
  (h2 : Y = 2 * D)
  (h3 : M = D + 4)
  (h4 : M = Y / 2)
  : D = 7 ∧ M = 11 := by
  sorry

end david_and_maria_ages_l159_159916


namespace irrational_roots_of_odd_coeff_quad_l159_159330

theorem irrational_roots_of_odd_coeff_quad (a b c : ℤ) (ha : a % 2 = 1) (hb : b % 2 = 1) (hc : c % 2 = 1) :
  ¬ ∃ r : ℚ, a * r^2 + b * r + c = 0 := 
sorry

end irrational_roots_of_odd_coeff_quad_l159_159330


namespace isosceles_trapezoid_base_ratio_correct_l159_159323

def isosceles_trapezoid_ratio (x y a b : ℝ) : Prop :=
  b = 2 * x ∧ a = 2 * y ∧ a + b = 10 ∧ (y * (Real.sqrt 2 + 1) = 5) →

  (a / b = (2 * (Real.sqrt 2) - 1) / 2)

theorem isosceles_trapezoid_base_ratio_correct: ∃ (x y a b : ℝ), 
  isosceles_trapezoid_ratio x y a b := sorry

end isosceles_trapezoid_base_ratio_correct_l159_159323


namespace min_value_of_u_l159_159722

variable (a b : ℝ)
variable (h1 : 0 < a)
variable (h2 : 0 < b)
variable (h3 : a^2 - b + 4 ≤ 0)

theorem min_value_of_u : (∃ (u : ℝ), u = (2*a + 3*b) / (a + b) ∧ u ≥ 14/5) :=
sorry

end min_value_of_u_l159_159722


namespace sum_of_first_nine_terms_l159_159757

noncomputable def arithmetic_sequence_sum (a₁ d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

def a_n (a₁ d n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

theorem sum_of_first_nine_terms (a₁ d : ℕ) (h : a_n a₁ d 2 + a_n a₁ d 6 + a_n a₁ d 7 = 18) :
  arithmetic_sequence_sum a₁ d 9 = 54 :=
sorry

end sum_of_first_nine_terms_l159_159757


namespace total_volume_is_10_l159_159063

noncomputable def total_volume_of_final_mixture (V : ℝ) : ℝ :=
  2.5 + V

theorem total_volume_is_10 :
  ∃ (V : ℝ), 
  (0.30 * 2.5 + 0.50 * V = 0.45 * (2.5 + V)) ∧ 
  total_volume_of_final_mixture V = 10 :=
by
  sorry

end total_volume_is_10_l159_159063


namespace problem_solution_l159_159611

theorem problem_solution (x : ℝ) :
          ((3 * x - 4) * (x + 5) ≠ 0) → 
          (10 * x^3 + 20 * x^2 - 75 * x - 105) / ((3 * x - 4) * (x + 5)) < 5 ↔ 
          (x ∈ Set.Ioo (-5 : ℝ) (-1) ∪ Set.Ioi (4 / 3)) :=
sorry

end problem_solution_l159_159611


namespace largest_even_integer_product_l159_159382

theorem largest_even_integer_product (n : ℕ) (h : 2 * n * (2 * n + 2) * (2 * n + 4) * (2 * n + 6) = 5040) :
  2 * n + 6 = 20 :=
by
  sorry

end largest_even_integer_product_l159_159382


namespace number_of_marked_points_l159_159941

theorem number_of_marked_points
  (a1 a2 b1 b2 : ℕ)
  (hA : a1 * a2 = 50)
  (hB : b1 * b2 = 56)
  (h_sum : a1 + a2 = b1 + b2) :
  a1 + a2 + 1 = 16 :=
sorry

end number_of_marked_points_l159_159941


namespace overall_average_output_l159_159632

theorem overall_average_output 
  (initial_cogs : ℕ := 60) 
  (rate_1 : ℕ := 36) 
  (rate_2 : ℕ := 60) 
  (second_batch_cogs : ℕ := 60) :
  (initial_cogs + second_batch_cogs) / ((initial_cogs / rate_1) + (second_batch_cogs / rate_2)) = 45 := 
  sorry

end overall_average_output_l159_159632


namespace Panthers_total_games_l159_159423

/-
Given:
1) The Panthers had won 60% of their basketball games before the district play.
2) During district play, they won four more games and lost four.
3) They finished the season having won half of their total games.
Prove that the total number of games they played in all is 48.
-/

theorem Panthers_total_games
  (y : ℕ) -- total games before district play
  (x : ℕ) -- games won before district play
  (h1 : x = 60 * y / 100) -- they won 60% of the games before district play
  (h2 : (x + 4) = 50 * (y + 8) / 100) -- they won half of the total games including district play
  : (y + 8) = 48 := -- total games they played in all
sorry

end Panthers_total_games_l159_159423


namespace complex_z_eq_neg_i_l159_159614

theorem complex_z_eq_neg_i (z : ℂ) (i : ℂ) (h1 : i * z = 1) (hi : i^2 = -1) : z = -i :=
sorry

end complex_z_eq_neg_i_l159_159614


namespace tan_ineq_solution_l159_159934

theorem tan_ineq_solution (a : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : ∀ x, x = a * Real.pi → ¬ (Real.tan x = a * Real.pi)) :
    {x : ℝ | ∃ k : ℤ, k * Real.pi + Real.pi / 4 ≤ x ∧ x < k * Real.pi + Real.pi / 2}
    = {x : ℝ | ∃ k : ℤ, k * Real.pi + Real.pi / 4 ≤ x ∧ x < k * Real.pi + Real.pi / 2} := sorry

end tan_ineq_solution_l159_159934


namespace find_y_l159_159820

theorem find_y (t : ℝ) (x : ℝ := 3 - 2 * t) (y : ℝ := 5 * t + 6) (h : x = 1) : y = 11 :=
by
  sorry

end find_y_l159_159820


namespace find_number_l159_159263

theorem find_number (n : ℝ) (h : 3 / 5 * ((2 / 3 + 3 / 8) / n) - 1 / 16 = 0.24999999999999994) : n = 48 :=
  sorry

end find_number_l159_159263


namespace eval_polynomial_at_neg2_l159_159721

-- Define the polynomial function
def polynomial (x : ℤ) : ℤ := x^4 + x^3 + x^2 + x + 1

-- Statement of the problem, proving that the polynomial equals 11 when x = -2
theorem eval_polynomial_at_neg2 : polynomial (-2) = 11 := by
  sorry

end eval_polynomial_at_neg2_l159_159721


namespace Buratino_can_solve_l159_159717

theorem Buratino_can_solve :
  ∃ (MA TE TI KA : ℕ), MA ≠ TE ∧ MA ≠ TI ∧ MA ≠ KA ∧ TE ≠ TI ∧ TE ≠ KA ∧ TI ≠ KA ∧
  MA * TE * MA * TI * KA = 2016000 :=
by
  -- skip the proof using sorry
  sorry

end Buratino_can_solve_l159_159717


namespace original_length_equals_13_l159_159933

-- Definitions based on conditions
def original_width := 18
def increased_length (x : ℕ) := x + 2
def increased_width := 20

-- Total area condition
def total_area (x : ℕ) := 
  4 * ((increased_length x) * increased_width) + 2 * ((increased_length x) * increased_width)

theorem original_length_equals_13 (x : ℕ) (h : total_area x = 1800) : x = 13 := 
by
  sorry

end original_length_equals_13_l159_159933


namespace maximum_area_of_triangle_ABQ_l159_159500

open Real

structure Point3D where
  x : ℝ
  y : ℝ

def circle_C (Q : Point3D) : Prop := (Q.x - 3)^2 + (Q.y - 4)^2 = 4

def A := Point3D.mk 1 0
def B := Point3D.mk (-1) 0

noncomputable def area_triangle (P Q R : Point3D) : ℝ :=
  (1 / 2) * abs ((P.x * (Q.y - R.y)) + (Q.x * (R.y - P.y)) + (R.x * (P.y - Q.y)))

theorem maximum_area_of_triangle_ABQ : ∀ (Q : Point3D), circle_C Q → area_triangle A B Q ≤ 6 := by
  sorry

end maximum_area_of_triangle_ABQ_l159_159500


namespace baskets_picked_l159_159064

theorem baskets_picked
  (B : ℕ) -- How many baskets did her brother pick?
  (S : ℕ := 15) -- Each basket contains 15 strawberries
  (H1 : (8 * B * S) + (B * S) + ((8 * B * S) - 93) = 4 * 168) -- Total number of strawberries when divided equally
  (H2 : S = 15) -- Number of strawberries in each basket
: B = 3 :=
sorry

end baskets_picked_l159_159064


namespace part_I_part_II_l159_159192

noncomputable def A : Set ℝ := {x | 2*x^2 - 5*x - 3 <= 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | (x - (2*a + 1)) * (x - (a - 1)) < 0}

theorem part_I :
  (A ∪ B 0 = {x : ℝ | -1 < x ∧ x ≤ 3}) :=
by sorry

theorem part_II (a : ℝ) :
  (A ∩ B a = ∅) →
  (a ≤ -3/4 ∨ a ≥ 4) ∧ a ≠ -2 :=
by sorry


end part_I_part_II_l159_159192


namespace lost_card_number_l159_159834

theorem lost_card_number (p : ℕ) (c : ℕ) (h : 0 ≤ c ∧ c ≤ 9)
  (sum_remaining_cards : 10 * p + 45 - (p + c) = 2012) : p + c = 223 := by
  sorry

end lost_card_number_l159_159834


namespace at_least_one_has_two_distinct_roots_l159_159130

theorem at_least_one_has_two_distinct_roots
  (p q1 q2 : ℝ)
  (h : p = q1 + q2 + 1) :
  (1 - 4 * q1 > 0) ∨ ((q1 + q2 + 1) ^ 2 - 4 * q2 > 0) :=
by sorry

end at_least_one_has_two_distinct_roots_l159_159130


namespace geometric_sequence_l159_159003

variable {α : Type*} [LinearOrderedField α]

-- Define the geometric sequence
def geom_seq (a₁ r : α) (n : ℕ) : α := a₁ * r^(n-1)

theorem geometric_sequence :
  ∀ (a₁ : α), a₁ > 0 → geom_seq a₁ 2 3 * geom_seq a₁ 2 11 = 16 → geom_seq a₁ 2 5 = 1 :=
by
  intros a₁ h_pos h_eq
  sorry

end geometric_sequence_l159_159003


namespace average_of_four_given_conditions_l159_159869

noncomputable def average_of_four_integers : ℕ × ℕ × ℕ × ℕ → ℚ :=
  λ ⟨a, b, c, d⟩ => (a + b + c + d : ℚ) / 4

theorem average_of_four_given_conditions :
  ∀ (A B C D : ℕ), 
    (A + B) / 2 = 35 → 
    C = 130 → 
    D = 1 → 
    average_of_four_integers (A, B, C, D) = 50.25 := 
by
  intros A B C D hAB hC hD
  unfold average_of_four_integers
  sorry

end average_of_four_given_conditions_l159_159869


namespace negation_proposition_l159_159522

theorem negation_proposition (x y : ℝ) :
  (¬ ∃ (x y : ℝ), 2 * x + 3 * y + 3 < 0) ↔ (∀ (x y : ℝ), 2 * x + 3 * y + 3 ≥ 0) :=
by {
  sorry
}

end negation_proposition_l159_159522


namespace total_area_of_squares_l159_159812

theorem total_area_of_squares (x : ℝ) (hx : 4 * x^2 = 240) : 
  let small_square_area := x^2
  let large_square_area := (2 * x)^2
  2 * small_square_area + large_square_area = 360 :=
by
  let small_square_area := x^2
  let large_square_area := (2 * x)^2
  sorry

end total_area_of_squares_l159_159812


namespace average_speed_last_segment_l159_159878

theorem average_speed_last_segment (D : ℝ) (T_mins : ℝ) (S1 S2 : ℝ) (t : ℝ) (S_last : ℝ) :
  D = 150 ∧ T_mins = 135 ∧ S1 = 50 ∧ S2 = 60 ∧ t = 45 →
  S_last = 90 :=
by
    sorry

end average_speed_last_segment_l159_159878


namespace sum_of_coords_of_four_points_l159_159418

noncomputable def four_points_sum_coords : ℤ :=
  let y1 := 13 + 5
  let y2 := 13 - 5
  let x1 := 7 + 12
  let x2 := 7 - 12
  ((x2 + y2) + (x2 + y1) + (x1 + y2) + (x1 + y1))

theorem sum_of_coords_of_four_points : four_points_sum_coords = 80 :=
  by
    sorry

end sum_of_coords_of_four_points_l159_159418


namespace toms_age_is_16_l159_159794

variable (J T : ℕ) -- John's current age is J and Tom's current age is T

-- Condition 1: John was thrice as old as Tom 6 years ago
axiom h1 : J - 6 = 3 * (T - 6)

-- Condition 2: John will be 2 times as old as Tom in 4 years
axiom h2 : J + 4 = 2 * (T + 4)

-- Proving Tom's current age is 16
theorem toms_age_is_16 : T = 16 := by
  sorry

end toms_age_is_16_l159_159794


namespace geometrical_shapes_OABC_l159_159253

/-- Given distinct points A(x₁, y₁), B(x₂, y₂), and C(2x₁ - x₂, 2y₁ - y₂) on a coordinate plane
    and the origin O(0,0), determine the possible geometrical shapes that the figure OABC can form
    among these three possibilities: (1) parallelogram (2) straight line (3) rhombus.
    
    Prove that the figure OABC can form either a parallelogram or a straight line,
    but not a rhombus.
-/
theorem geometrical_shapes_OABC (x₁ y₁ x₂ y₂ : ℝ) (h_distinct : (x₁, y₁) ≠ (x₂, y₂) ∧ (x₁, y₁) ≠ (2 * x₁ - x₂, 2 * y₁ - y₂) ∧ (x₂, y₂) ≠ (2 * x₁ - x₂, 2 * y₁ - y₂)) :
  (∃ t : ℝ, t ≠ 0 ∧ t ≠ 1 ∧ x₂ = t * x₁ ∧ y₂ = t * y₁) ∨
  (2 * x₁ = x₁ + x₂ ∧ 2 * y₁ = y₁ + y₂) :=
sorry

end geometrical_shapes_OABC_l159_159253


namespace flour_cups_l159_159188

theorem flour_cups (f : ℚ) (h : f = 4 + 3/4) : (1/3) * f = 1 + 7/12 := by
  sorry

end flour_cups_l159_159188


namespace area_of_rectangle_is_correct_l159_159210

-- Given Conditions
def radius : ℝ := 7
def diameter : ℝ := 2 * radius
def width : ℝ := diameter
def length : ℝ := 3 * width

-- Question: Find the area of the rectangle
def area := length * width

-- The theorem to prove
theorem area_of_rectangle_is_correct : area = 588 :=
by
  -- Proof steps can go here.
  sorry

end area_of_rectangle_is_correct_l159_159210


namespace real_coefficient_polynomials_with_special_roots_l159_159804

noncomputable def P1 : Polynomial ℝ := (Polynomial.X + 1) * (Polynomial.X - 1 / 2) * (Polynomial.X - 2) * (Polynomial.X ^ 2 - Polynomial.X + 1)
noncomputable def P2 : Polynomial ℝ := (Polynomial.X + 1) ^ 3 * (Polynomial.X - 1 / 2) * (Polynomial.X - 2)
noncomputable def P3 : Polynomial ℝ := (Polynomial.X + 1) * (Polynomial.X - 1 / 2) ^ 3 * (Polynomial.X - 2)
noncomputable def P4 : Polynomial ℝ := (Polynomial.X + 1) * (Polynomial.X - 1 / 2) * (Polynomial.X - 2) ^ 3
noncomputable def P5 : Polynomial ℝ := (Polynomial.X + 1) ^ 2 * (Polynomial.X - 1 / 2) ^ 2 * (Polynomial.X - 2)
noncomputable def P6 : Polynomial ℝ := (Polynomial.X + 1) * (Polynomial.X - 1 / 2) ^ 2 * (Polynomial.X - 2) ^ 2
noncomputable def P7 : Polynomial ℝ := (Polynomial.X + 1) ^ 2 * (Polynomial.X - 1 / 2) * (Polynomial.X - 2) ^ 2

theorem real_coefficient_polynomials_with_special_roots (P : Polynomial ℝ) :
  (∀ α, Polynomial.IsRoot P α → Polynomial.IsRoot P (1 - α) ∧ Polynomial.IsRoot P (1 / α)) →
  P = P1 ∨ P = P2 ∨ P = P3 ∨ P = P4 ∨ P = P5 ∨ P = P6 ∨ P = P7 :=
  sorry

end real_coefficient_polynomials_with_special_roots_l159_159804


namespace asymptotes_of_hyperbola_l159_159070

theorem asymptotes_of_hyperbola (b : ℝ) (h_focus : 2 * Real.sqrt 2 ≠ 0) :
  2 * Real.sqrt 2 = Real.sqrt ((2 * 2) + b^2) → 
  (∀ (x y : ℝ), ((x^2 / 4) - (y^2 / b^2) = 1 → x^2 - y^2 = 4)) → 
  (∀ (x y : ℝ), ((x^2 - y^2 = 4) → y = x ∨ y = -x)) := 
  sorry

end asymptotes_of_hyperbola_l159_159070


namespace find_A_plus_B_l159_159258

/-- Let A, B, C, and D be distinct digits such that 0 ≤ A, B, C, D ≤ 9.
    C and D are non-zero, and A ≠ B ≠ C ≠ D.
    If (A+B)/(C+D) is an integer and C+D is minimized,
    then prove that A + B = 15. -/
theorem find_A_plus_B
  (A B C D : ℕ)
  (h_digits : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (h_range : 0 ≤ A ∧ A ≤ 9 ∧ 0 ≤ B ∧ B ≤ 9 ∧ 0 ≤ C ∧ C ≤ 9 ∧ 0 ≤ D ∧ D ≤ 9)
  (h_nonzero_CD : C ≠ 0 ∧ D ≠ 0)
  (h_integer : (A + B) % (C + D) = 0)
  (h_min_CD : ∀ C' D', (C' ≠ C ∨ D' ≠ D) → (C' ≠ 0 ∧ D' ≠ 0 → (C + D ≤ C' + D'))) :
  A + B = 15 := 
sorry

end find_A_plus_B_l159_159258


namespace pond_water_amount_l159_159926

theorem pond_water_amount : 
  let initial_water := 500 
  let evaporation_rate := 4
  let rain_amount := 2
  let days := 40
  initial_water - days * (evaporation_rate - rain_amount) = 420 :=
by
  sorry

end pond_water_amount_l159_159926


namespace repeating_decimal_product_l159_159617

def repeating_decimal_12 := 12 / 99
def repeating_decimal_34 := 34 / 99

theorem repeating_decimal_product : (repeating_decimal_12 * repeating_decimal_34) = 136 / 3267 := by
  sorry

end repeating_decimal_product_l159_159617


namespace john_income_increase_l159_159561

theorem john_income_increase :
  let initial_job_income := 60
  let initial_freelance_income := 40
  let initial_online_sales_income := 20

  let new_job_income := 120
  let new_freelance_income := 60
  let new_online_sales_income := 35

  let weeks_per_month := 4

  let initial_monthly_income := (initial_job_income + initial_freelance_income + initial_online_sales_income) * weeks_per_month
  let new_monthly_income := (new_job_income + new_freelance_income + new_online_sales_income) * weeks_per_month
  
  let percentage_increase := 100 * (new_monthly_income - initial_monthly_income) / initial_monthly_income

  percentage_increase = 79.17 := by
  sorry

end john_income_increase_l159_159561


namespace complete_the_square_l159_159928

theorem complete_the_square (x : ℝ) : x^2 - 8 * x + 1 = 0 → (x - 4)^2 = 15 :=
by
  intro h
  sorry

end complete_the_square_l159_159928


namespace find_a₉_l159_159620

variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)

axiom S_6_eq : S 6 = 3
axiom S_11_eq : S 11 = 18

noncomputable def a₉ : ℝ := sorry -- Define a₉ here, proof skipped by "sorry"

theorem find_a₉ (a : ℕ → ℝ) (S : ℕ → ℝ) :
  S 6 = 3 →
  S 11 = 18 →
  a₉ = 3 :=
by
  intros S_6_eq S_11_eq
  sorry -- Proof goes here

end find_a₉_l159_159620


namespace fair_coin_three_flips_probability_l159_159939

theorem fair_coin_three_flips_probability :
  ∀ (prob : ℕ → ℚ) (independent : ∀ n, prob n = 1 / 2),
    prob 0 * prob 1 * prob 2 = 1 / 8 := 
by
  intros prob independent
  sorry

end fair_coin_three_flips_probability_l159_159939


namespace ethanol_in_fuel_A_l159_159209

def fuel_tank_volume : ℝ := 208
def fuel_A_volume : ℝ := 82
def fuel_B_volume : ℝ := fuel_tank_volume - fuel_A_volume
def ethanol_in_fuel_B : ℝ := 0.16
def total_ethanol : ℝ := 30

theorem ethanol_in_fuel_A 
  (x : ℝ) 
  (H_fuel_tank_capacity : fuel_tank_volume = 208) 
  (H_fuel_A_volume : fuel_A_volume = 82) 
  (H_fuel_B_volume : fuel_B_volume = 126) 
  (H_ethanol_in_fuel_B : ethanol_in_fuel_B = 0.16) 
  (H_total_ethanol : total_ethanol = 30) 
  : 82 * x + 0.16 * 126 = 30 → x = 0.12 := by
  sorry

end ethanol_in_fuel_A_l159_159209


namespace initial_quantity_of_milk_in_container_A_l159_159884

variables {CA MB MC : ℝ}

theorem initial_quantity_of_milk_in_container_A (h1 : MB = 0.375 * CA)
    (h2 : MC = 0.625 * CA)
    (h_eq : MB + 156 = MC - 156) :
    CA = 1248 :=
by
  sorry

end initial_quantity_of_milk_in_container_A_l159_159884


namespace transformed_passes_through_l159_159628

def original_parabola (x : ℝ) : ℝ :=
  -x^2 - 2*x + 3

def transformed_parabola (x : ℝ) : ℝ :=
  -(x - 1)^2 + 2

theorem transformed_passes_through : transformed_parabola (-1) = 1 :=
  by sorry

end transformed_passes_through_l159_159628


namespace initial_amount_calc_l159_159901

theorem initial_amount_calc 
  (M : ℝ)
  (H1 : M * 0.3675 = 350) :
  M = 952.38 :=
by
  sorry

end initial_amount_calc_l159_159901


namespace initial_number_of_men_l159_159171

theorem initial_number_of_men (M : ℝ) (P : ℝ) (h1 : P = M * 20) (h2 : P = (M + 200) * 16.67) : M = 1000 :=
by
  sorry

end initial_number_of_men_l159_159171


namespace parabola_equation_l159_159644

-- Definitions of the conditions
def parabola_passes_through (x y : ℝ) : Prop :=
  y^2 = -2 * (3 * x)

def focus_on_line (x y : ℝ) : Prop :=
  3 * x - 2 * y - 6 = 0

theorem parabola_equation (x y : ℝ) (hM : x = -6 ∧ y = 6) (hF : ∃ (x y : ℝ), focus_on_line x y) :
  parabola_passes_through x y = (y^2 = -6 * x) :=
by 
  sorry

end parabola_equation_l159_159644


namespace blocks_for_fort_l159_159008

theorem blocks_for_fort :
  let length := 15 
  let width := 12 
  let height := 6
  let thickness := 1
  let V_original := length * width * height
  let interior_length := length - 2 * thickness
  let interior_width := width - 2 * thickness
  let interior_height := height - thickness
  let V_interior := interior_length * interior_width * interior_height
  let V_blocks := V_original - V_interior
  V_blocks = 430 :=
by
  sorry

end blocks_for_fort_l159_159008


namespace inequality_proof_l159_159746

variable (x y z : ℝ)

theorem inequality_proof (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (hxyz : x + y + z = 1) :
  x * (1 - 2 * x) * (1 - 3 * x) + y * (1 - 2 * y) * (1 - 3 * y) + z * (1 - 2 * z) * (1 - 3 * z) ≥ 0 := 
sorry

end inequality_proof_l159_159746


namespace top_leftmost_rectangle_is_B_l159_159565

structure Rectangle :=
  (w x y z : ℕ)

def RectangleA := Rectangle.mk 5 1 9 2
def RectangleB := Rectangle.mk 2 0 6 3
def RectangleC := Rectangle.mk 6 7 4 1
def RectangleD := Rectangle.mk 8 4 3 5
def RectangleE := Rectangle.mk 7 3 8 0

-- Problem Statement: Given these rectangles, prove that the top leftmost rectangle is B.
theorem top_leftmost_rectangle_is_B 
  (A : Rectangle := RectangleA)
  (B : Rectangle := RectangleB)
  (C : Rectangle := RectangleC)
  (D : Rectangle := RectangleD)
  (E : Rectangle := RectangleE) : 
  B = Rectangle.mk 2 0 6 3 := 
sorry

end top_leftmost_rectangle_is_B_l159_159565


namespace min_z_value_l159_159085

theorem min_z_value (x y z : ℝ) (h1 : 2 * x + y = 1) (h2 : z = 4 ^ x + 2 ^ y) : z ≥ 2 * Real.sqrt 2 :=
by
  sorry

end min_z_value_l159_159085


namespace find_p_l159_159088

noncomputable def parabola_focus (p : ℝ) : ℝ × ℝ :=
  (p / 2, 0)

def hyperbola_focus : ℝ × ℝ :=
  (2, 0)

theorem find_p (p : ℝ) (h : p > 0) (hp : parabola_focus p = hyperbola_focus) : p = 4 :=
by
  sorry

end find_p_l159_159088


namespace compare_abc_l159_159273

noncomputable def a := Real.sin (15 * Real.pi / 180) * Real.cos (15 * Real.pi / 180)
noncomputable def b := Real.cos (Real.pi / 6) ^ 2 - Real.sin (Real.pi / 6) ^ 2
noncomputable def c := Real.tan (30 * Real.pi / 180) / (1 - Real.tan (30 * Real.pi / 180) ^ 2)

theorem compare_abc : a < b ∧ b < c :=
by
  sorry

end compare_abc_l159_159273


namespace fourth_number_is_57_l159_159207

noncomputable def digit_sum (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

def sum_list (l : List ℕ) : ℕ :=
  l.foldr (.+.) 0

theorem fourth_number_is_57 : 
  ∃ (N : ℕ), N < 100 ∧ 177 + N = 4 * (33 + digit_sum N) ∧ N = 57 :=
by {
  sorry
}

end fourth_number_is_57_l159_159207


namespace ratio_of_m1_m2_l159_159908

open Real

theorem ratio_of_m1_m2 :
  ∀ (m : ℝ) (p q : ℝ), p ≠ 0 ∧ q ≠ 0 ∧ m ≠ 0 ∧
    (p + q = -((3 - 2 * m) / m)) ∧ 
    (p * q = 4 / m) ∧ 
    (p / q + q / p = 2) → 
   ∃ (m1 m2 : ℝ), 
    (4 * m1^2 - 28 * m1 + 9 = 0) ∧
    (4 * m2^2 - 28 * m2 + 9 = 0) ∧ 
    (m1 ≠ m2) ∧ 
    (m1 + m2 = 7) ∧ 
    (m1 * m2 = 9 / 4) ∧ 
    (m1 / m2 + m2 / m1 = 178 / 9) :=
by sorry

end ratio_of_m1_m2_l159_159908


namespace hexagon_area_l159_159310

-- Define the area of a triangle
def triangle_area (base height: ℝ) : ℝ := 0.5 * base * height

-- Given dimensions for each triangle
def base_unit := 1
def original_height := 3
def new_height := 4

-- Calculate areas of each triangle in the new configuration
def single_triangle_area := triangle_area base_unit new_height
def total_triangle_area := 4 * single_triangle_area

-- The area of the rectangular region formed by the hexagon and triangles
def rectangular_region_area := (base_unit + original_height + original_height) * new_height

-- Prove the area of the hexagon
theorem hexagon_area : rectangular_region_area - total_triangle_area = 32 :=
by
  -- We will provide the proof here
  sorry

end hexagon_area_l159_159310


namespace find_x_given_y_l159_159292

-- Given that x and y are always positive and x^2 and y vary inversely.
-- i.e., we have a relationship x^2 * y = k for a constant k,
-- and given that y = 8 when x = 3, find the value of x when y = 648.

theorem find_x_given_y
  (x y : ℝ)
  (h_pos_x : 0 < x)
  (h_pos_y : 0 < y)
  (h_inv : ∀ x y, x^2 * y = 72)
  (h_y : y = 648) : x = 1 / 3 :=
by
  sorry

end find_x_given_y_l159_159292


namespace Michael_points_l159_159180

theorem Michael_points (total_points : ℕ) (num_other_players : ℕ) (avg_points : ℕ) (Michael_points : ℕ) 
  (h1 : total_points = 75)
  (h2 : num_other_players = 5)
  (h3 : avg_points = 6)
  (h4 : Michael_points = total_points - num_other_players * avg_points) :
  Michael_points = 45 := by
  sorry

end Michael_points_l159_159180


namespace intersection_M_N_l159_159601

def M := {y : ℝ | y <= 4}
def N := {x : ℝ | x > 0}

theorem intersection_M_N : {x : ℝ | x > 0} ∩ {y : ℝ | y <= 4} = {z : ℝ | 0 < z ∧ z <= 4} :=
by
  sorry

end intersection_M_N_l159_159601


namespace fred_green_balloons_l159_159416

theorem fred_green_balloons (initial : ℕ) (given : ℕ) (final : ℕ) (h1 : initial = 709) (h2 : given = 221) (h3 : final = initial - given) : final = 488 :=
by
  sorry

end fred_green_balloons_l159_159416


namespace max_profit_l159_159182

noncomputable def fixed_cost : ℝ := 2.5

noncomputable def cost (x : ℝ) : ℝ :=
  if x < 80 then (1/3) * x^2 + 10 * x
  else 51 * x + 10000 / x - 1450

noncomputable def revenue (x : ℝ) : ℝ := 0.05 * 1000 * x

noncomputable def profit (x : ℝ) : ℝ :=
  revenue x - cost x - fixed_cost * 10

theorem max_profit : ∃ x_opt : ℝ, ∀ x : ℝ, 0 < x → 
  profit x ≤ profit 100 ∧ x_opt = 100 :=
by
  sorry

end max_profit_l159_159182


namespace prob_neither_snow_nor_windy_l159_159626

-- Define the probabilities.
def prob_snow : ℚ := 1 / 4
def prob_windy : ℚ := 1 / 3

-- Define the complementary probabilities.
def prob_not_snow : ℚ := 1 - prob_snow
def prob_not_windy : ℚ := 1 - prob_windy

-- State that the events are independent and calculate the combined probability.
theorem prob_neither_snow_nor_windy :
  prob_not_snow * prob_not_windy = 1 / 2 := by
  sorry

end prob_neither_snow_nor_windy_l159_159626


namespace jose_profit_share_correct_l159_159991

-- Definitions for the conditions
def tom_investment : ℕ := 30000
def tom_months : ℕ := 12
def jose_investment : ℕ := 45000
def jose_months : ℕ := 10
def total_profit : ℕ := 36000

-- Capital months calculations
def tom_capital_months : ℕ := tom_investment * tom_months
def jose_capital_months : ℕ := jose_investment * jose_months
def total_capital_months : ℕ := tom_capital_months + jose_capital_months

-- Jose's share of the profit calculation
def jose_share_of_profit : ℕ := (jose_capital_months * total_profit) / total_capital_months

-- The theorem to prove
theorem jose_profit_share_correct : jose_share_of_profit = 20000 := by
  -- This is where the proof steps would go
  sorry

end jose_profit_share_correct_l159_159991


namespace pebbles_game_invariant_l159_159688

/-- 
The game of pebbles is played on an infinite board of lattice points (i, j).
Initially, there is a pebble at (0, 0).
A move consists of removing a pebble from point (i, j) and placing a pebble at each of the points (i+1, j) and (i, j+1) provided both are vacant.
Show that at any stage of the game there is a pebble at some lattice point (a, b) with 0 ≤ a + b ≤ 3. 
-/
theorem pebbles_game_invariant :
  ∀ (board : ℕ × ℕ → Prop) (initial_state : board (0, 0)) (move : (ℕ × ℕ) → Prop → Prop → Prop),
  (∀ (i j : ℕ), board (i, j) → ¬ board (i+1, j) ∧ ¬ board (i, j+1) → board (i+1, j) ∧ board (i, j+1)) →
  ∃ (a b : ℕ), (0 ≤ a + b ∧ a + b ≤ 3) ∧ board (a, b) :=
by
  intros board initial_state move move_rule
  sorry 

end pebbles_game_invariant_l159_159688


namespace ratio_tuesday_monday_l159_159796

-- Define the conditions
variables (M T W : ℕ) (hM : M = 450) (hW : W = 300) (h_rel : W = T + 75)

-- Define the theorem
theorem ratio_tuesday_monday : (T : ℚ) / M = 1 / 2 :=
by
  -- Sorry means the proof has been omitted in Lean.
  sorry

end ratio_tuesday_monday_l159_159796


namespace shortest_chord_through_M_is_x_plus_y_minus_1_eq_0_l159_159836

noncomputable def circle_C : Set (ℝ × ℝ) := { p | (p.1^2 + p.2^2 - 4*p.1 - 2*p.2) = 0 }

def point_M_in_circle : Prop :=
  (1, 0) ∈ circle_C

theorem shortest_chord_through_M_is_x_plus_y_minus_1_eq_0 :
  point_M_in_circle →
  ∃ (a b c : ℝ), a * 1 + b * 0 + c = 0 ∧
  ∀ (x y : ℝ), (a * x + b * y + c = 0) → (x + y - 1 = 0) :=
by
  sorry

end shortest_chord_through_M_is_x_plus_y_minus_1_eq_0_l159_159836


namespace paityn_red_hats_l159_159917

theorem paityn_red_hats (R : ℕ) : 
  (R + 24 + (4 / 5) * ↑R + 48 = 108) → R = 20 :=
by
  intro h
  sorry


end paityn_red_hats_l159_159917


namespace xy_equation_result_l159_159731

theorem xy_equation_result (x y : ℝ) (h1 : x + y = 6) (h2 : x * y = -5) :
  x + (x^4 / y^3) + (y^4 / x^3) + y = -10.528 :=
by
  sorry

end xy_equation_result_l159_159731


namespace find_f2_l159_159140

noncomputable def f (x a b : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

theorem find_f2 (a b : ℝ) (h : (∃ x : ℝ, f x a b = 10 ∧ x = 1)):
  f 2 a b = 18 ∨ f 2 a b = 11 :=
sorry

end find_f2_l159_159140


namespace domain_of_f_l159_159220

def domain_condition1 (x : ℝ) : Prop := 1 - |x - 1| > 0
def domain_condition2 (x : ℝ) : Prop := x - 1 ≠ 0

theorem domain_of_f :
  (∀ x : ℝ, domain_condition1 x ∧ domain_condition2 x → 0 < x ∧ x < 2 ∧ x ≠ 1) ↔
  (∀ x : ℝ, x ∈ (Set.Ioo 0 1 ∪ Set.Ioo 1 2)) :=
by
  sorry

end domain_of_f_l159_159220


namespace number_of_sides_of_polygon_l159_159780

theorem number_of_sides_of_polygon (exterior_angle : ℝ) (sum_exterior_angles : ℝ) (h1 : exterior_angle = 30) (h2 : sum_exterior_angles = 360) :
  sum_exterior_angles / exterior_angle = 12 := 
by
  sorry

end number_of_sides_of_polygon_l159_159780


namespace avg_weight_ab_l159_159752

theorem avg_weight_ab (A B C : ℝ) 
  (h1 : (A + B + C) / 3 = 30) 
  (h2 : (B + C) / 2 = 28) 
  (h3 : B = 16) : 
  (A + B) / 2 = 25 := 
by 
  sorry

end avg_weight_ab_l159_159752


namespace problem1_problem2_l159_159658

-- Problem 1
theorem problem1 (a b : ℝ) : 4 * a^4 * b^3 / (-2 * a * b)^2 = a^2 * b :=
by
  sorry

-- Problem 2
theorem problem2 (x y : ℝ) : (3 * x - y)^2 - (3 * x + 2 * y) * (3 * x - 2 * y) = 5 * y^2 - 6 * x * y :=
by
  sorry

end problem1_problem2_l159_159658


namespace bigger_part_of_dividing_56_l159_159893

theorem bigger_part_of_dividing_56 (x y : ℕ) (h₁ : x + y = 56) (h₂ : 10 * x + 22 * y = 780) : max x y = 38 :=
by
  sorry

end bigger_part_of_dividing_56_l159_159893


namespace exist_prime_not_dividing_l159_159205

theorem exist_prime_not_dividing (p : ℕ) (hp : Prime p) : 
  ∃ q : ℕ, Prime q ∧ ∀ n : ℕ, 0 < n → ¬ (q ∣ n^p - p) := 
sorry

end exist_prime_not_dividing_l159_159205


namespace find_length_of_AC_in_triangle_ABC_l159_159583

noncomputable def length_AC_in_triangle_ABC
  (AB BC : ℝ) (angle_B : ℝ) (h_AB : AB = 1) (h_BC : BC = 2) (h_angle_B : angle_B = Real.pi / 3) :
  ℝ :=
  let cos_B := Real.cos (Real.pi / 3)
  let AC_squared := AB^2 + BC^2 - 2 * AB * BC * cos_B
  Real.sqrt AC_squared

theorem find_length_of_AC_in_triangle_ABC :
  ∃ AC : ℝ, ∀ (AB BC : ℝ) (angle_B : ℝ) (h_AB : AB = 1) (h_BC : BC = 2) (h_angle_B : angle_B = Real.pi / 3),
    length_AC_in_triangle_ABC AB BC angle_B h_AB h_BC h_angle_B = Real.sqrt 3 :=
by sorry

end find_length_of_AC_in_triangle_ABC_l159_159583


namespace probability_heart_spade_queen_l159_159234

theorem probability_heart_spade_queen (h_cards : ℕ) (s_cards : ℕ) (q_cards : ℕ) (total_cards : ℕ) 
    (h_not_q : ℕ) (remaining_cards_after_2 : ℕ) (remaining_spades : ℕ) 
    (queen_remaining_after_2 : ℕ) (remaining_cards_after_1 : ℕ) :
    h_cards = 13 ∧ s_cards = 13 ∧ q_cards = 4 ∧ total_cards = 52 ∧ h_not_q = 12 ∧ remaining_cards_after_2 = 50 ∧
    remaining_spades = 13 ∧ queen_remaining_after_2 = 3 ∧ remaining_cards_after_1 = 51 →
    (h_cards / total_cards) * (remaining_spades / remaining_cards_after_1) * (q_cards / remaining_cards_after_2) + 
    (q_cards / total_cards) * (remaining_spades / remaining_cards_after_1) * (queen_remaining_after_2 / remaining_cards_after_2) = 
    221 / 44200 := by 
  sorry

end probability_heart_spade_queen_l159_159234


namespace smallest_four_digit_solution_l159_159338

theorem smallest_four_digit_solution :
  ∃ x : ℕ, 1000 ≤ x ∧ x < 10000 ∧
  (3 * x ≡ 6 [MOD 12]) ∧
  (5 * x + 20 ≡ 25 [MOD 15]) ∧
  (3 * x - 2 ≡ 2 * x [MOD 35]) ∧
  x = 1274 :=
by
  sorry

end smallest_four_digit_solution_l159_159338


namespace integer_solutions_l159_159909

theorem integer_solutions (x : ℤ) : 
  (⌊(x : ℚ) / 2⌋ * ⌊(x : ℚ) / 3⌋ * ⌊(x : ℚ) / 4⌋ = x^2) ↔ (x = 0 ∨ x = 24) := 
sorry

end integer_solutions_l159_159909


namespace usual_time_is_180_l159_159102

variable (D S1 T : ℝ)

-- Conditions
def usual_time : Prop := T = D / S1
def reduced_speed : Prop := ∃ S2 : ℝ, S2 = 5 / 6 * S1
def total_delay : Prop := 6 + 12 + 18 = 36
def total_time_reduced_speed_stops : Prop := ∃ T' : ℝ, T' + 36 = 6 / 5 * T
def time_equation : Prop := T + 36 = 6 / 5 * T

-- Proof problem statement
theorem usual_time_is_180 (h1 : usual_time D S1 T)
                          (h2 : reduced_speed S1)
                          (h3 : total_delay)
                          (h4 : total_time_reduced_speed_stops T)
                          (h5 : time_equation T) :
                          T = 180 := by
  sorry

end usual_time_is_180_l159_159102


namespace f_of_13_eq_223_l159_159042

def f (n : ℕ) : ℕ := n^2 + n + 41

theorem f_of_13_eq_223 : f 13 = 223 := 
by sorry

end f_of_13_eq_223_l159_159042


namespace percentage_increase_correct_l159_159448

-- Define the highest and lowest scores as given conditions.
def highest_score : ℕ := 92
def lowest_score : ℕ := 65

-- State that the percentage increase calculation will result in 41.54%
theorem percentage_increase_correct :
  ((highest_score - lowest_score) * 100) / lowest_score = 4154 / 100 :=
by sorry

end percentage_increase_correct_l159_159448


namespace shortest_tree_height_proof_l159_159090

def tallest_tree_height : ℕ := 150
def middle_tree_height : ℕ := (2 * tallest_tree_height) / 3
def shortest_tree_height : ℕ := middle_tree_height / 2

theorem shortest_tree_height_proof : shortest_tree_height = 50 := by
  sorry

end shortest_tree_height_proof_l159_159090


namespace cardinals_count_l159_159451

theorem cardinals_count (C R B S : ℕ) 
  (hR : R = 4 * C)
  (hB : B = 2 * C)
  (hS : S = 3 * C + 1)
  (h_total : C + R + B + S = 31) :
  C = 3 :=
by
  sorry

end cardinals_count_l159_159451


namespace complement_A_in_U_l159_159962

def U : Set ℕ := {x | x ≥ 2}
def A : Set ℕ := {x | x^2 ≥ 5}

theorem complement_A_in_U : (U \ A) = {2} := by
  sorry

end complement_A_in_U_l159_159962


namespace geometric_mean_eq_6_l159_159035

theorem geometric_mean_eq_6 (b c : ℝ) (hb : b = 3) (hc : c = 12) :
  (b * c) ^ (1/2 : ℝ) = 6 := 
by
  sorry

end geometric_mean_eq_6_l159_159035


namespace range_of_m_l159_159091

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := 2 * m * x ^ 2 - 2 * (4 - m) * x + 1
def g (m : ℝ) (x : ℝ) : ℝ := m * x

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, f m x > 0 ∨ g m x > 0) ↔ (0 < m ∧ m < 8) :=
sorry

end range_of_m_l159_159091


namespace f_x_neg_l159_159998

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then x^2 + 1 else -x^2 - 1

theorem f_x_neg (x : ℝ) (h : x < 0) : f x = -x^2 - 1 :=
by
  sorry

end f_x_neg_l159_159998


namespace train_length_l159_159839

theorem train_length
  (time : ℝ) (man_speed train_speed : ℝ) (same_direction : Prop)
  (h_time : time = 62.99496040316775)
  (h_man_speed : man_speed = 6)
  (h_train_speed : train_speed = 30)
  (h_same_direction : same_direction) :
  (train_speed - man_speed) * (1000 / 3600) * time = 1259.899208063355 := 
sorry

end train_length_l159_159839


namespace fraction_negative_iff_x_lt_2_l159_159320

theorem fraction_negative_iff_x_lt_2 (x : ℝ) :
  (-5) / (2 - x) < 0 ↔ x < 2 := by
  sorry

end fraction_negative_iff_x_lt_2_l159_159320


namespace probability_even_sum_l159_159368

def p_even_first_wheel : ℚ := 1 / 3
def p_odd_first_wheel : ℚ := 2 / 3
def p_even_second_wheel : ℚ := 3 / 5
def p_odd_second_wheel : ℚ := 2 / 5

theorem probability_even_sum : 
  (p_even_first_wheel * p_even_second_wheel) + (p_odd_first_wheel * p_odd_second_wheel) = 7 / 15 :=
by
  sorry

end probability_even_sum_l159_159368


namespace prob_exceeds_175_l159_159360

-- Definitions from the conditions
def prob_less_than_160 (p : ℝ) : Prop := p = 0.2
def prob_160_to_175 (p : ℝ) : Prop := p = 0.5

-- The mathematical equivalence proof we need
theorem prob_exceeds_175 (p₁ p₂ p₃ : ℝ) 
  (h₁ : prob_less_than_160 p₁) 
  (h₂ : prob_160_to_175 p₂) 
  (H : p₃ = 1 - (p₁ + p₂)) :
  p₃ = 0.3 := 
by
  -- Placeholder for proof
  sorry

end prob_exceeds_175_l159_159360


namespace det_B_l159_159028

open Matrix

-- Define matrix B
def B (x y : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![x, 2], ![-3, y]]

-- Define the condition B + 2 * B⁻¹ = 0
def condition (x y : ℝ) : Prop :=
  let Binv := (1 / (x * y + 6)) • ![![y, -2], ![3, x]]
  B x y + 2 • Binv = 0

-- Prove that if the condition holds, then det B = 2
theorem det_B (x y : ℝ) (h : condition x y) : det (B x y) = 2 :=
  sorry

end det_B_l159_159028


namespace exists_composite_arith_sequence_pairwise_coprime_l159_159022

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem exists_composite_arith_sequence_pairwise_coprime (n : ℕ) : 
  ∃ seq : Fin n → ℕ, (∀ i, ∃ k, seq i = factorial n + k) ∧ 
  (∀ i j, i ≠ j → gcd (seq i) (seq j) = 1) :=
by
  sorry

end exists_composite_arith_sequence_pairwise_coprime_l159_159022


namespace least_n_exceeds_product_l159_159345

def product_exceeds (n : ℕ) : Prop :=
  10^(n * (n + 1) / 18) > 10^6

theorem least_n_exceeds_product (n : ℕ) (h : n = 12) : product_exceeds n :=
by
  rw [h]
  sorry

end least_n_exceeds_product_l159_159345


namespace min_days_is_9_l159_159437

theorem min_days_is_9 (n : ℕ) (rain_morning rain_afternoon sunny_morning sunny_afternoon : ℕ)
  (h1 : rain_morning + rain_afternoon = 7)
  (h2 : rain_afternoon ≤ sunny_morning)
  (h3 : sunny_afternoon = 5)
  (h4 : sunny_morning = 6) :
  n ≥ 9 :=
sorry

end min_days_is_9_l159_159437


namespace typing_cost_equation_l159_159515

def typing_cost (x : ℝ) : ℝ :=
  200 * x + 80 * 3 + 20 * 6

theorem typing_cost_equation (x : ℝ) (h : typing_cost x = 1360) : x = 5 :=
by
  sorry

end typing_cost_equation_l159_159515


namespace trig_identity_l159_159739

theorem trig_identity (α : ℝ) (h : Real.sin (α - π / 12) = 1 / 3) : Real.cos (α + 5 * π / 12) = -1 / 3 :=
by
  sorry

end trig_identity_l159_159739


namespace johns_speed_l159_159026

def time1 : ℕ := 2
def time2 : ℕ := 3
def total_distance : ℕ := 225

def total_time : ℕ := time1 + time2

theorem johns_speed :
  (total_distance : ℝ) / (total_time : ℝ) = 45 :=
sorry

end johns_speed_l159_159026


namespace parallel_lines_sufficient_condition_l159_159408

theorem parallel_lines_sufficient_condition :
  ∀ a : ℝ, (a^2 - a) = 2 → (a = 2 ∨ a = -1) :=
by
  intro a h
  sorry

end parallel_lines_sufficient_condition_l159_159408


namespace red_balloon_probability_l159_159279

-- Define the conditions
def initial_red_balloons := 2
def initial_blue_balloons := 4
def inflated_red_balloons := 2
def inflated_blue_balloons := 2

-- Define the total number of balloons after inflation
def total_red_balloons := initial_red_balloons + inflated_red_balloons
def total_blue_balloons := initial_blue_balloons + inflated_blue_balloons
def total_balloons := total_red_balloons + total_blue_balloons

-- Define the probability calculation
def red_probability := (total_red_balloons : ℚ) / total_balloons * 100

-- The theorem to prove
theorem red_balloon_probability : red_probability = 40 := by
  sorry -- Skipping the proof itself

end red_balloon_probability_l159_159279


namespace scientific_notation_000073_l159_159454

theorem scientific_notation_000073 : 0.000073 = 7.3 * 10^(-5) := by
  sorry

end scientific_notation_000073_l159_159454


namespace smallest_a_l159_159536

theorem smallest_a (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 96 * a^2 = b^3) : a = 12 :=
by
  sorry

end smallest_a_l159_159536


namespace four_digit_greater_than_three_digit_l159_159689

theorem four_digit_greater_than_three_digit (n m : ℕ) (h₁ : 1000 ≤ n ∧ n ≤ 9999) (h₂ : 100 ≤ m ∧ m ≤ 999) : n > m :=
sorry

end four_digit_greater_than_three_digit_l159_159689


namespace find_common_difference_l159_159977

theorem find_common_difference (a a_n S_n : ℝ) (h1 : a = 3) (h2 : a_n = 50) (h3 : S_n = 318) : 
  ∃ d n, (a + (n - 1) * d = a_n) ∧ (n / 2 * (a + a_n) = S_n) ∧ (d = 47 / 11) := 
by
  sorry

end find_common_difference_l159_159977


namespace complex_z_calculation_l159_159586

theorem complex_z_calculation (z : ℂ) (hz : z^2 + z + 1 = 0) :
  z^99 + z^100 + z^101 + z^102 + z^103 = 1 + z :=
sorry

end complex_z_calculation_l159_159586


namespace tan_theta_solution_l159_159784

theorem tan_theta_solution (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < 15) 
  (h_tan_eq : Real.tan θ + Real.tan (2 * θ) + Real.tan (4 * θ) = 0) :
  Real.tan θ = 1 / Real.sqrt 2 :=
sorry

end tan_theta_solution_l159_159784


namespace sum_of_first_seven_terms_l159_159144

variable (a : ℕ → ℝ)
variable (d : ℝ)

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- Given condition
axiom a3_a4_a5_sum : a 3 + a 4 + a 5 = 12

-- Statement to prove
theorem sum_of_first_seven_terms (h : arithmetic_sequence a d) : 
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 28 :=
sorry

end sum_of_first_seven_terms_l159_159144


namespace find_other_number_l159_159355

theorem find_other_number (LCM HCF num1 num2 : ℕ) 
  (h1 : LCM = 2310) 
  (h2 : HCF = 30) 
  (h3 : num1 = 330) 
  (h4 : LCM * HCF = num1 * num2) : 
  num2 = 210 := by 
  sorry

end find_other_number_l159_159355


namespace function_machine_output_l159_159069

-- Define the initial input
def input : ℕ := 12

-- Define the function machine steps
def functionMachine (x : ℕ) : ℕ :=
  if x * 3 <= 20 then (x * 3) / 2
  else (x * 3) - 2

-- State the property we want to prove
theorem function_machine_output : functionMachine 12 = 34 :=
by
  -- Skip the proof
  sorry

end function_machine_output_l159_159069


namespace find_a_of_inequality_solution_l159_159512

theorem find_a_of_inequality_solution (a : ℝ) :
  (∀ x : ℝ, 0 < x ∧ x < 1 ↔ x^2 - a * x < 0) → a = 1 := 
by 
  sorry

end find_a_of_inequality_solution_l159_159512


namespace interest_groups_ranges_l159_159106

variable (A B C : Finset ℕ)

-- Given conditions
axiom card_A : A.card = 5
axiom card_B : B.card = 4
axiom card_C : C.card = 7
axiom card_A_inter_B : (A ∩ B).card = 3
axiom card_A_inter_B_inter_C : (A ∩ B ∩ C).card = 2

-- Mathematical statement to be proved
theorem interest_groups_ranges :
  2 ≤ ((A ∪ B) ∩ C).card ∧ ((A ∪ B) ∩ C).card ≤ 5 ∧
  8 ≤ (A ∪ B ∪ C).card ∧ (A ∪ B ∪ C).card ≤ 11 := by
  sorry

end interest_groups_ranges_l159_159106


namespace find_a_b_find_A_l159_159622

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := 2 * (Real.log x / Real.log 2) ^ 2 + 2 * a * (Real.log (1 / x) / Real.log 2) + b

theorem find_a_b : (∀ x : ℝ, 0 < x → f x a b = 2 * (Real.log x / Real.log 2)^2 + 2 * a * (Real.log (1 / x) / Real.log 2) + b) 
                     → f (1/2) a b = -8 
                     ∧ ∀ x : ℝ, 0 < x → x ≠ 1/2 → f x a b ≥ f (1 / 2) a b
                     → a = -2 ∧ b = -6 := 
sorry

theorem find_A (a b : ℝ) (h₁ : a = -2) (h₂ : b = -6) : 
  { x : ℝ | 0 < x ∧ f x a b > 0 } = {x | 0 < x ∧ (x < 1/8 ∨ x > 2)} :=
sorry

end find_a_b_find_A_l159_159622


namespace devin_biked_more_l159_159235

def cyra_distance := 77
def cyra_time := 7
def cyra_speed := cyra_distance / cyra_time
def devin_speed := cyra_speed + 3
def marathon_time := 7
def devin_distance := devin_speed * marathon_time
def distance_difference := devin_distance - cyra_distance

theorem devin_biked_more : distance_difference = 21 := 
  by
    sorry

end devin_biked_more_l159_159235


namespace trig_identity_l159_159798

theorem trig_identity (f : ℝ → ℝ) (x : ℝ) (h : f (Real.sin x) = 3 - Real.cos (2 * x)) : f (Real.cos x) = 3 + Real.cos (2 * x) :=
sorry

end trig_identity_l159_159798


namespace identify_mathematicians_l159_159960

def famous_people := List (Nat × String)

def is_mathematician : Nat → Bool
| 1 => false  -- Bill Gates
| 2 => true   -- Gauss
| 3 => false  -- Yuan Longping
| 4 => false  -- Nobel
| 5 => true   -- Chen Jingrun
| 6 => true   -- Hua Luogeng
| 7 => false  -- Gorky
| 8 => false  -- Einstein
| _ => false  -- default case

theorem identify_mathematicians (people : famous_people) : 
  (people.filter (fun (n, _) => is_mathematician n)) = [(2, "Gauss"), (5, "Chen Jingrun"), (6, "Hua Luogeng")] :=
by sorry

end identify_mathematicians_l159_159960


namespace determine_a_b_l159_159823

-- Define the function f
def f (x a b : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

-- Define the first derivative of the function f
def f' (x a b : ℝ) : ℝ := 3*x^2 + 2*a*x + b

-- Define the conditions given in the problem
def conditions (a b : ℝ) : Prop :=
  (f' 1 a b = 0) ∧ (f 1 a b = 10)

-- Provide the main theorem stating the required proof
theorem determine_a_b (a b : ℝ) (h : conditions a b) : a = 4 ∧ b = -11 :=
by {
  sorry
}

end determine_a_b_l159_159823


namespace common_ratio_common_difference_l159_159430

noncomputable def common_ratio_q {a b : ℕ → ℝ} (d : ℝ) (q : ℝ) :=
  (∀ n, b (n+1) = q * b n) ∧ (a 2 = -1) ∧ (a 1 < a 2) ∧ 
  (b 1 = (a 1)^2) ∧ (b 2 = (a 2)^2) ∧ (b 3 = (a 3)^2) ∧ 
  (∀ n, a (n+1) = a n + d)

theorem common_ratio
  {a b : ℕ → ℝ} {d : ℝ}
  (h_arith : ∀ n, a (n + 1) = a n + d) (h_nonzero : d ≠ 0)
  (h_geom : ∀ n, b (n + 1) = (b 1^(1/2)) ^ (2 ^ n))
  (h_b1 : b 1 = (a 1) ^ 2) (h_b2 : b 2 = (a 2) ^ 2)
  (h_b3 : b 3 = (a 3) ^ 2) (h_a2 : a 2 = -1) (h_a1a2 : a 1 < a 2) :
  q = 3 - 2 * (2:ℝ).sqrt :=
sorry

theorem common_difference
  {a b : ℕ → ℝ} {d : ℝ}
  (h_arith : ∀ n, a (n + 1) = a n + d) (h_nonzero : d ≠ 0)
  (h_geom : ∀ n, b (n + 1) = (b 1^(1/2)) ^ (2 ^ n))
  (h_b1 : b 1 = (a 1) ^ 2) (h_b2 : b 2 = (a 2) ^ 2)
  (h_b3 : b 3 = (a 3) ^ 2) (h_a2 : a 2 = -1) (h_a1a2 : a 1 < a 2) :
  d = (2 : ℝ).sqrt :=
sorry

end common_ratio_common_difference_l159_159430


namespace subtract_fractions_l159_159588

theorem subtract_fractions : (18 / 42 - 3 / 8) = 3 / 56 :=
by
  sorry

end subtract_fractions_l159_159588


namespace percentage_of_copper_first_alloy_l159_159713

theorem percentage_of_copper_first_alloy :
  ∃ x : ℝ, 
  (66 * x / 100) + (55 * 21 / 100) = 121 * 15 / 100 ∧
  x = 10 := 
sorry

end percentage_of_copper_first_alloy_l159_159713


namespace game_show_prizes_l159_159442

theorem game_show_prizes :
  let digits := [1, 1, 2, 2, 3, 3, 3, 3]
  let permutations := Nat.factorial 8 / (Nat.factorial 4 * Nat.factorial 2 * Nat.factorial 2)
  let partitions := Nat.choose 7 3
  permutations * partitions = 14700 :=
by
  let digits := [1, 1, 2, 2, 3, 3, 3, 3]
  let permutations := Nat.factorial 8 / (Nat.factorial 4 * Nat.factorial 2 * Nat.factorial 2)
  let partitions := Nat.choose 7 3
  exact sorry

end game_show_prizes_l159_159442


namespace savings_after_increase_l159_159460

-- Conditions
def salary : ℕ := 5000
def initial_savings_ratio : ℚ := 0.20
def expense_increase_ratio : ℚ := 1.20

-- Derived initial values
def initial_savings : ℚ := initial_savings_ratio * salary
def initial_expenses : ℚ := ((1 : ℚ) - initial_savings_ratio) * salary

-- New expenses after increase
def new_expenses : ℚ := expense_increase_ratio * initial_expenses

-- Savings after expense increase
def final_savings : ℚ := salary - new_expenses

theorem savings_after_increase : final_savings = 200 := by
  sorry

end savings_after_increase_l159_159460


namespace total_marks_eq_300_second_candidate_percentage_l159_159527

-- Defining the conditions
def percentage_marks (total_marks : ℕ) : ℕ := 40
def fail_by (fail_marks : ℕ) : ℕ := 40
def passing_marks : ℕ := 160

-- The number of total marks in the exam computed from conditions
theorem total_marks_eq_300 : ∃ T, 0.40 * T = 120 :=
by
  use 300
  sorry

-- The percentage of marks the second candidate gets
theorem second_candidate_percentage : ∃ percent, percent = (180 / 300) * 100 :=
by
  use 60
  sorry

end total_marks_eq_300_second_candidate_percentage_l159_159527


namespace remainder_of_A_div_by_9_l159_159811

theorem remainder_of_A_div_by_9 (A B : ℕ) (h : A = B * 9 + 13) : A % 9 = 4 := by
  sorry

end remainder_of_A_div_by_9_l159_159811


namespace keith_spent_on_cards_l159_159507

theorem keith_spent_on_cards :
  let digimon_card_cost := 4.45
  let num_digimon_packs := 4
  let baseball_card_cost := 6.06
  let total_spent := num_digimon_packs * digimon_card_cost + baseball_card_cost
  total_spent = 23.86 :=
by
  sorry

end keith_spent_on_cards_l159_159507


namespace productivity_increase_l159_159278

/-- 
The original workday is 8 hours. 
During the first 6 hours, productivity is at the planned level (1 unit/hour). 
For the next 2 hours, productivity falls by 25% (0.75 units/hour). 
The workday is extended by 1 hour (now 9 hours). 
During the first 6 hours of the extended shift, productivity remains at the planned level (1 unit/hour). 
For the remaining 3 hours of the extended shift, productivity falls by 30% (0.7 units/hour). 
Prove that the overall productivity for the shift increased by 8% as a result of extending the workday.
-/
theorem productivity_increase
  (planned_productivity : ℝ)
  (initial_work_hours : ℝ)
  (initial_productivity_drop : ℝ)
  (extended_work_hours : ℝ)
  (extended_productivity_drop : ℝ)
  (initial_total_work : ℝ)
  (extended_total_work : ℝ)
  (percentage_increase : ℝ) :
  planned_productivity = 1 →
  initial_work_hours = 8 →
  initial_productivity_drop = 0.25 →
  extended_work_hours = 9 →
  extended_productivity_drop = 0.30 →
  initial_total_work = 7.5 →
  extended_total_work = 8.1 →
  percentage_increase = 8 →
  ((extended_total_work - initial_total_work) / initial_total_work * 100) = percentage_increase :=
sorry

end productivity_increase_l159_159278


namespace greatest_divisor_of_620_and_180_l159_159922

/-- This theorem asserts that the greatest divisor of 620 that 
    is smaller than 100 and also a factor of 180 is 20. -/
theorem greatest_divisor_of_620_and_180 (d : ℕ) (h1 : d ∣ 620) (h2 : d ∣ 180) (h3 : d < 100) : d ≤ 20 :=
by
  sorry

end greatest_divisor_of_620_and_180_l159_159922


namespace empty_square_exists_in_4x4_l159_159464

theorem empty_square_exists_in_4x4  :
  ∀ (points: Finset (Fin 4 × Fin 4)), points.card = 15 → 
  ∃ (i j : Fin 4), (i, j) ∉ points :=
by
  sorry

end empty_square_exists_in_4x4_l159_159464


namespace min_time_to_cook_cakes_l159_159800

theorem min_time_to_cook_cakes (cakes : ℕ) (pot_capacity : ℕ) (time_per_side : ℕ) 
  (h1 : cakes = 3) (h2 : pot_capacity = 2) (h3 : time_per_side = 5) : 
  ∃ t, t = 15 := by
  sorry

end min_time_to_cook_cakes_l159_159800


namespace subset_implication_l159_159925

noncomputable def M (x : ℝ) : Prop := -2 * x + 1 ≥ 0
noncomputable def N (a x : ℝ) : Prop := x < a

theorem subset_implication (a : ℝ) :
  (∀ x, M x → N a x) → a > 1 / 2 :=
by
  sorry

end subset_implication_l159_159925


namespace sophomores_in_program_l159_159348

theorem sophomores_in_program (total_students : ℕ) (not_sophomores_nor_juniors : ℕ) 
    (percentage_sophomores_debate : ℚ) (percentage_juniors_debate : ℚ) 
    (eq_debate_team : ℚ) (total_students := 40) 
    (not_sophomores_nor_juniors := 5) 
    (percentage_sophomores_debate := 0.20) 
    (percentage_juniors_debate := 0.25) 
    (eq_debate_team := (percentage_sophomores_debate * S = percentage_juniors_debate * J)) :
    ∀ (S J : ℚ), S + J = total_students - not_sophomores_nor_juniors → 
    (S = 5 * J / 4) → S = 175 / 9 := 
by 
  sorry

end sophomores_in_program_l159_159348


namespace problem_1_problem_2_l159_159828

noncomputable def f (a b x : ℝ) := a * (x - 1)^2 + b * Real.log x

theorem problem_1 (a : ℝ) (h_deriv : ∀ x ≥ 2, (2 * a * x^2 - 2 * a * x + 1) / x ≤ 0) : 
  a ≤ -1 / 4 :=
sorry

theorem problem_2 (a : ℝ) (h_ineq : ∀ x ≥ 1, a * (x - 1)^2 + Real.log x ≤ x - 1) : 
  a ≤ 0 :=
sorry

end problem_1_problem_2_l159_159828


namespace find_number_l159_159212

theorem find_number (x : ℝ) (h : x - (3/5) * x = 56) : x = 140 :=
sorry

end find_number_l159_159212


namespace lines_through_point_l159_159269

theorem lines_through_point {a b c : ℝ} :
  (3 = a + b) ∧ (3 = b + c) ∧ (3 = c + a) → (a = 1.5 ∧ b = 1.5 ∧ c = 1.5) :=
by
  intros h
  sorry

end lines_through_point_l159_159269


namespace polygon_sides_l159_159387

theorem polygon_sides (n : ℕ) (h₁ : ∀ (m : ℕ), m = n → n > 2) (h₂ : 180 * (n - 2) = 156 * n) : n = 15 :=
by
  sorry

end polygon_sides_l159_159387


namespace x_is_perfect_square_l159_159989

theorem x_is_perfect_square (x y : ℕ) (hxy : x > y) (hdiv : xy ∣ x ^ 2022 + x + y ^ 2) : ∃ n : ℕ, x = n^2 := 
sorry

end x_is_perfect_square_l159_159989


namespace interval_solution_length_l159_159817

theorem interval_solution_length (a b : ℝ) (h : (b - a) / 3 = 8) : b - a = 24 := by
  sorry

end interval_solution_length_l159_159817


namespace harrys_age_l159_159625

-- Definitions of the ages
variable (Kiarra Bea Job Figaro Harry : ℕ)

-- Given conditions
variable (h1 : Kiarra = 2 * Bea)
variable (h2 : Job = 3 * Bea)
variable (h3 : Figaro = Job + 7)
variable (h4 : Harry = Figaro / 2)
variable (h5 : Kiarra = 30)

-- The statement to prove
theorem harrys_age : Harry = 26 := sorry

end harrys_age_l159_159625


namespace max_sum_a_b_c_l159_159016

noncomputable def f (a b c x : ℝ) : ℝ := a * Real.cos x + b * Real.cos (2 * x) + c * Real.cos (3 * x)

theorem max_sum_a_b_c (a b c : ℝ) (h : ∀ x : ℝ, f a b c x ≥ -1) : a + b + c ≤ 3 :=
sorry

end max_sum_a_b_c_l159_159016


namespace correct_option_l159_159576

variable (a : ℤ)

theorem correct_option :
  (-2 * a^2)^3 = -8 * a^6 :=
by
  sorry

end correct_option_l159_159576


namespace total_number_of_workers_l159_159883

theorem total_number_of_workers (W : ℕ) (R : ℕ) 
  (h1 : (7 + R) * 8000 = 7 * 18000 + R * 6000) 
  (h2 : W = 7 + R) : W = 42 :=
by
  -- Proof steps will go here
  sorry

end total_number_of_workers_l159_159883


namespace sum_of_vertices_l159_159773

theorem sum_of_vertices (num_triangle num_hexagon : ℕ) (vertices_triangle vertices_hexagon : ℕ) :
  num_triangle = 1 → vertices_triangle = 3 →
  num_hexagon = 3 → vertices_hexagon = 6 →
  num_triangle * vertices_triangle + num_hexagon * vertices_hexagon = 21 :=
by
  intros h1 h2 h3 h4
  sorry

end sum_of_vertices_l159_159773


namespace frequency_calculation_l159_159911

-- Define the given conditions
def sample_capacity : ℕ := 20
def group_frequency : ℚ := 0.25

-- The main theorem statement
theorem frequency_calculation :
  sample_capacity * group_frequency = 5 :=
by sorry

end frequency_calculation_l159_159911


namespace total_weight_of_lifts_l159_159859

theorem total_weight_of_lifts
  (F S : ℕ)
  (h1 : F = 600)
  (h2 : 2 * F = S + 300) :
  F + S = 1500 := by
  sorry

end total_weight_of_lifts_l159_159859


namespace Lara_age_10_years_from_now_l159_159562

theorem Lara_age_10_years_from_now (current_year_age : ℕ) (age_7_years_ago : ℕ)
  (h1 : age_7_years_ago = 9) (h2 : current_year_age = age_7_years_ago + 7) :
  current_year_age + 10 = 26 :=
by
  sorry

end Lara_age_10_years_from_now_l159_159562


namespace external_tangent_b_value_l159_159394

theorem external_tangent_b_value:
  ∀ {C1 C2 : ℝ × ℝ} (r1 r2 : ℝ) (m b : ℝ),
  C1 = (3, -2) ∧ r1 = 3 ∧ 
  C2 = (15, 8) ∧ r2 = 8 ∧
  m = (60 / 11) →
  (∃ b, y = m * x + b ∧ b = 720 / 11) :=
by 
  sorry

end external_tangent_b_value_l159_159394


namespace common_difference_is_minus_3_l159_159065

variable (a_n : ℕ → ℤ) (a1 d : ℤ)

-- Definitions expressing the conditions of the problem
def arithmetic_prog : Prop := ∀ (n : ℕ), a_n n = a1 + (n - 1) * d

def condition1 : Prop := a1 + (a1 + 6 * d) = -8

def condition2 : Prop := a1 + d = 2

-- The statement we need to prove
theorem common_difference_is_minus_3 :
  arithmetic_prog a_n a1 d ∧ condition1 a1 d ∧ condition2 a1 d → d = -3 :=
by {
  -- The proof would go here
  sorry
}

end common_difference_is_minus_3_l159_159065


namespace quadratic_discriminant_l159_159462

theorem quadratic_discriminant (k : ℝ) :
  (∃ x : ℝ, k*x^2 + 2*x - 1 = 0) ∧ (∀ a b, (a*x + b) ^ 2 = a^2 * x^2 + 2 * a * b * x + b^2) ∧
  (a = k) ∧ (b = 2) ∧ (c = -1) ∧ ((b^2 - 4 * a * c = 0) → (4 + 4 * k = 0)) → k = -1 :=
sorry

end quadratic_discriminant_l159_159462


namespace find_m_of_transformed_point_eq_l159_159067

theorem find_m_of_transformed_point_eq (m : ℝ) (h : m + 1 = 5) : m = 4 :=
by
  sorry

end find_m_of_transformed_point_eq_l159_159067


namespace pencil_eraser_cost_l159_159300

theorem pencil_eraser_cost (p e : ℕ) (h_eq : 10 * p + 4 * e = 120) (h_gt : p > e) : p + e = 15 :=
by sorry

end pencil_eraser_cost_l159_159300


namespace perpendicular_bisector_of_circles_l159_159907

theorem perpendicular_bisector_of_circles
  (circle1 : ∀ x y : ℝ, x^2 + y^2 - 4 * x + 6 * y = 0)
  (circle2 : ∀ x y : ℝ, x^2 + y^2 - 6 * x = 0) :
  ∃ x y : ℝ, (3 * x - y - 9 = 0) :=
by
  sorry

end perpendicular_bisector_of_circles_l159_159907


namespace sin_70_given_sin_10_l159_159255

theorem sin_70_given_sin_10 (k : ℝ) (h : Real.sin 10 = k) : Real.sin 70 = 1 - 2 * k^2 := 
by 
  sorry

end sin_70_given_sin_10_l159_159255


namespace quadratic_rewrite_l159_159706

theorem quadratic_rewrite (a b c x : ℤ) :
  (16 * x^2 - 40 * x - 72 = a^2 * x^2 + 2 * a * b * x + b^2 + c) →
  (a = 4 ∨ a = -4) →
  (2 * a * b = -40) →
  ab = -20 := by
sorry

end quadratic_rewrite_l159_159706


namespace net_income_after_tax_l159_159435

theorem net_income_after_tax (gross_income : ℝ) (tax_rate : ℝ) : 
  (gross_income = 45000) → (tax_rate = 0.13) → 
  (gross_income - gross_income * tax_rate = 39150) :=
by
  intro h1 h2
  rw [h1, h2]
  sorry

end net_income_after_tax_l159_159435


namespace system_solution_l159_159198

theorem system_solution (x y : ℝ) :
  (x + y = 4) ∧ (2 * x - y = 2) → x = 2 ∧ y = 2 := by 
sorry

end system_solution_l159_159198


namespace pine_taller_than_maple_l159_159627

def height_maple : ℚ := 13 + 1 / 4
def height_pine : ℚ := 19 + 3 / 8

theorem pine_taller_than_maple :
  (height_pine - height_maple = 6 + 1 / 8) :=
sorry

end pine_taller_than_maple_l159_159627


namespace roots_squared_sum_l159_159481

theorem roots_squared_sum (x1 x2 : ℝ) (h₁ : x1^2 - 5 * x1 + 3 = 0) (h₂ : x2^2 - 5 * x2 + 3 = 0) :
  x1^2 + x2^2 = 19 :=
by
  sorry

end roots_squared_sum_l159_159481


namespace max_value_sin_cos_combination_l159_159558

theorem max_value_sin_cos_combination :
  ∀ x : ℝ, (5 * Real.sin x + 12 * Real.cos x) ≤ 13 :=
by
  intro x
  sorry

end max_value_sin_cos_combination_l159_159558


namespace melanie_plums_l159_159108

variable (initialPlums : ℕ) (givenPlums : ℕ)

theorem melanie_plums :
  initialPlums = 7 → givenPlums = 3 → initialPlums - givenPlums = 4 :=
by
  intro h1 h2
  -- proof omitted
  exact sorry

end melanie_plums_l159_159108


namespace third_twenty_third_wise_superior_number_l159_159549

def wise_superior_number (x : ℕ) : Prop :=
  ∃ m n : ℕ, m > n ∧ m - n > 1 ∧ x = m^2 - n^2

theorem third_twenty_third_wise_superior_number :
  ∃ T_3 T_23 : ℕ, wise_superior_number T_3 ∧ wise_superior_number T_23 ∧ T_3 = 15 ∧ T_23 = 57 :=
by
  sorry

end third_twenty_third_wise_superior_number_l159_159549


namespace sum_eq_zero_l159_159379

variable {R : Type} [Field R]

-- Define the conditions
def cond1 (a b c : R) : Prop := (a + b) / c = (b + c) / a
def cond2 (a b c : R) : Prop := (b + c) / a = (a + c) / b
def neq (b c : R) : Prop := b ≠ c

-- State the theorem
theorem sum_eq_zero (a b c : R) (h1 : cond1 a b c) (h2 : cond2 a b c) (h3 : neq b c) : a + b + c = 0 := 
by sorry

end sum_eq_zero_l159_159379


namespace factorization_of_2210_l159_159367

theorem factorization_of_2210 : 
  ∃! (a b : ℕ), (10 ≤ a ∧ a < 100) ∧ (10 ≤ b ∧ b < 100) ∧ (a * b = 2210) :=
sorry

end factorization_of_2210_l159_159367


namespace general_term_l159_159304

open Nat

noncomputable def S (n : ℕ) : ℕ := 2 * n^2 - 3 * n

theorem general_term (n : ℕ) (hn : n > 0) : (S n - S (n - 1)) = 4 * n - 5 := by
  sorry

end general_term_l159_159304


namespace negation_of_P_is_exists_Q_l159_159155

def P (x : ℝ) : Prop := x^2 - x + 3 > 0

theorem negation_of_P_is_exists_Q :
  (¬ (∀ x : ℝ, P x)) ↔ (∃ x : ℝ, ¬ P x) :=
sorry

end negation_of_P_is_exists_Q_l159_159155


namespace students_with_dogs_l159_159066

theorem students_with_dogs (total_students : ℕ) (half_students : total_students / 2 = 50)
  (percent_girls_with_dogs : ℕ → ℚ) (percent_boys_with_dogs : ℕ → ℚ)
  (girls_with_dogs : ∀ (total_girls: ℕ), percent_girls_with_dogs total_girls = 0.2)
  (boys_with_dogs : ∀ (total_boys: ℕ), percent_boys_with_dogs total_boys = 0.1) :
  ∀ (total_girls total_boys students_with_dogs: ℕ),
  total_students = 100 →
  total_girls = total_students / 2 →
  total_boys = total_students / 2 →
  total_girls = 50 →
  total_boys = 50 →
  students_with_dogs = (percent_girls_with_dogs (total_students / 2) * (total_students / 2) + 
                        percent_boys_with_dogs (total_students / 2) * (total_students / 2)) →
  students_with_dogs = 15 :=
by
  intros total_girls total_boys students_with_dogs h1 h2 h3 h4 h5 h6
  sorry

end students_with_dogs_l159_159066


namespace area_of_rectangle_l159_159285

noncomputable def length := 44.4
noncomputable def width := 29.6

theorem area_of_rectangle (h1 : width = 2 / 3 * length) (h2 : 2 * (length + width) = 148) : 
  (length * width) = 1314.24 := 
by 
  sorry

end area_of_rectangle_l159_159285


namespace parts_of_milk_in_drink_A_l159_159801

theorem parts_of_milk_in_drink_A (x : ℝ) (h : 63 * (4 * x) / (7 * (x + 3)) = 63 * 3 / (x + 3) + 21) : x = 16.8 :=
by
  sorry

end parts_of_milk_in_drink_A_l159_159801


namespace find_c_l159_159038

-- Define the function
def f (c x : ℝ) : ℝ := x^4 - 8 * x^2 + c

-- Condition: The function has a minimum value of -14 on the interval [-1, 3]
def condition (c : ℝ) : Prop :=
  ∃ x ∈ Set.Icc (-1 : ℝ) 3, ∀ y ∈ Set.Icc (-1 : ℝ) 3, f c x ≤ f c y ∧ f c x = -14

-- The theorem to be proved
theorem find_c : ∃ c : ℝ, condition c ∧ c = 2 :=
sorry

end find_c_l159_159038


namespace zeros_of_f_l159_159649

noncomputable def f (x : ℝ) : ℝ := (x - 1) * (x ^ 2 - 2 * x - 3)

theorem zeros_of_f :
  { x : ℝ | f x = 0 } = {1, -1, 3} :=
sorry

end zeros_of_f_l159_159649


namespace series_sum_equals_one_sixth_l159_159055

noncomputable def series_sum : ℝ :=
  ∑' n, 2^n / (7^(2^n) + 1)

theorem series_sum_equals_one_sixth : series_sum = 1 / 6 :=
by
  sorry

end series_sum_equals_one_sixth_l159_159055


namespace no_real_y_for_two_equations_l159_159438

theorem no_real_y_for_two_equations:
  ¬ ∃ (x y : ℝ), x^2 + y^2 = 16 ∧ x^2 + 3 * y + 30 = 0 :=
by
  sorry

end no_real_y_for_two_equations_l159_159438


namespace fourth_person_height_l159_159465

theorem fourth_person_height 
  (height1 height2 height3 height4 : ℝ)
  (diff12 : height2 = height1 + 2)
  (diff23 : height3 = height2 + 2)
  (diff34 : height4 = height3 + 6)
  (avg_height : (height1 + height2 + height3 + height4) / 4 = 76) :
  height4 = 82 :=
by
  sorry

end fourth_person_height_l159_159465


namespace second_job_pay_rate_l159_159043

-- Definitions of the conditions
def h1 : ℕ := 3 -- hours for the first job
def r1 : ℕ := 7 -- rate for the first job
def h2 : ℕ := 2 -- hours for the second job
def h3 : ℕ := 4 -- hours for the third job
def r3 : ℕ := 12 -- rate for the third job
def d : ℕ := 5   -- number of days
def T : ℕ := 445 -- total earnings

-- The proof statement
theorem second_job_pay_rate (x : ℕ) : 
  d * (h1 * r1 + 2 * x + h3 * r3) = T ↔ x = 10 := 
by 
  -- Implement the necessary proof steps here
  sorry

end second_job_pay_rate_l159_159043


namespace rectangle_within_l159_159897

theorem rectangle_within (a b c d : ℝ) (h1 : a < c) (h2 : c ≤ d) (h3 : d < b) (h4 : a * b < c * d) :
  (b^2 - a^2)^2 ≤ (b * d - a * c)^2 + (b * c - a * d)^2 :=
by
  sorry

end rectangle_within_l159_159897


namespace builder_needs_boards_l159_159776

theorem builder_needs_boards (packages : ℕ) (boards_per_package : ℕ) (total_boards : ℕ)
  (h1 : packages = 52)
  (h2 : boards_per_package = 3)
  (h3 : total_boards = packages * boards_per_package) : 
  total_boards = 156 :=
by
  rw [h1, h2] at h3
  exact h3

end builder_needs_boards_l159_159776


namespace initial_money_l159_159185

theorem initial_money (B S G M : ℕ) 
  (hB : B = 8) 
  (hS : S = 2 * B) 
  (hG : G = 3 * S) 
  (change : ℕ) 
  (h_change : change = 28)
  (h_total : B + S + G + change = M) : 
  M = 100 := 
by 
  sorry

end initial_money_l159_159185


namespace number_of_bags_proof_l159_159226

def total_flight_time_hours : ℕ := 2
def minutes_per_hour : ℕ := 60
def total_minutes := total_flight_time_hours * minutes_per_hour

def peanuts_per_minute : ℕ := 1
def total_peanuts_eaten := total_minutes * peanuts_per_minute

def peanuts_per_bag : ℕ := 30
def number_of_bags : ℕ := total_peanuts_eaten / peanuts_per_bag

theorem number_of_bags_proof : number_of_bags = 4 := by
  -- proof goes here
  sorry

end number_of_bags_proof_l159_159226


namespace complement_union_l159_159133

def R := Set ℝ

def A : Set ℝ := {x | x ≥ 1}

def B : Set ℝ := {y | ∃ x, x ≥ 1 ∧ y = Real.exp x}

theorem complement_union (R : Set ℝ) (A : Set ℝ) (B : Set ℝ) :
  (A ∪ B)ᶜ = {x | x < 1} := by
  sorry

end complement_union_l159_159133


namespace sum_of_abcd_l159_159164

variable (a b c d : ℚ)

def condition (x : ℚ) : Prop :=
  x = a + 3 ∧
  x = b + 7 ∧
  x = c + 5 ∧
  x = d + 9 ∧
  x = a + b + c + d + 13

theorem sum_of_abcd (x : ℚ) (h : condition a b c d x) : a + b + c + d = -28 / 3 := 
by sorry

end sum_of_abcd_l159_159164


namespace like_terms_exponent_equality_l159_159191

theorem like_terms_exponent_equality (m n : ℕ) (a b : ℝ) 
    (H : 3 * a^m * b^2 = 2/3 * a * b^n) : m = 1 ∧ n = 2 :=
by
  sorry

end like_terms_exponent_equality_l159_159191


namespace proof_equivalence_l159_159083

variables {A B C D : Type} [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup D]
variables {α β γ δ : ℝ} -- angles are real numbers

-- Definition of cyclic quadrilateral
def cyclic_quadrilateral (α β γ δ : ℝ) : Prop :=
α + γ = 180 ∧ β + δ = 180

-- Definition of the problem statements
def statement1 (α γ : ℝ) : Prop :=
α = γ → α = 90

def statement3 (α γ : ℝ) : Prop :=
180 - α + 180 - γ = 180

def statement2 (α β : ℝ) (ψ χ : ℝ) : Prop := 
α = β → cyclic_quadrilateral α β ψ χ → ψ = χ ∨ (α = β ∧ α = ψ ∧ α = χ)

def statement4 (α β γ δ : ℝ) : Prop :=
1*α + 2*β + 3*γ + 4*δ = 360

-- Theorem statement
theorem proof_equivalence (α β γ δ : ℝ) :
  cyclic_quadrilateral α β γ δ →
  (statement1 α γ) ∧ (statement3 α γ) ∧ ¬(statement2 α β γ δ) ∧ ¬(statement4 α β γ δ) :=
by
  sorry

end proof_equivalence_l159_159083


namespace find_scalars_l159_159838

open Matrix

def B : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![2, -1],
    ![4, 3]]

def I : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![1, 0],
    ![0, 1]]

theorem find_scalars (r s : ℤ) (h : B^6 = r • B + s • I) :
  (r = 1125) ∧ (s = -1875) :=
sorry

end find_scalars_l159_159838


namespace problem_statement_l159_159892

variables (x y : ℝ)

theorem problem_statement
  (h1 : abs x = 4)
  (h2 : abs y = 2)
  (h3 : abs (x + y) = x + y) : 
  x - y = 2 ∨ x - y = 6 :=
sorry

end problem_statement_l159_159892


namespace problem_statement_l159_159244

noncomputable def f (x : ℝ) (a b α β : ℝ) : ℝ := a * Real.sin (Real.pi * x + α) + b * Real.cos (Real.pi * x + β) + 4

theorem problem_statement (a b α β : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : α ≠ 0) (h₃ : β ≠ 0) (h₄ : f 2013 a b α β = 5) :
  f 2014 a b α β = 3 :=
by
  sorry

end problem_statement_l159_159244


namespace response_rate_is_60_percent_l159_159104

-- Definitions based on conditions
def responses_needed : ℕ := 900
def questionnaires_mailed : ℕ := 1500

-- Derived definition
def response_rate_percentage : ℚ := (responses_needed : ℚ) / (questionnaires_mailed : ℚ) * 100

-- The theorem stating the problem
theorem response_rate_is_60_percent :
  response_rate_percentage = 60 := 
sorry

end response_rate_is_60_percent_l159_159104


namespace population_proof_l159_159802

def population (tosses : ℕ) (values : ℕ) : Prop :=
  (tosses = 7768) ∧ (values = 6)

theorem population_proof : 
  population 7768 6 :=
by
  unfold population
  exact And.intro rfl rfl

end population_proof_l159_159802


namespace garden_area_is_correct_l159_159183

def width_of_property : ℕ := 1000
def length_of_property : ℕ := 2250

def width_of_garden : ℕ := width_of_property / 8
def length_of_garden : ℕ := length_of_property / 10

def area_of_garden : ℕ := width_of_garden * length_of_garden

theorem garden_area_is_correct : area_of_garden = 28125 := by
  -- Skipping proof for the purpose of this example
  sorry

end garden_area_is_correct_l159_159183


namespace tournament_committee_count_l159_159600

-- Given conditions
def num_teams : ℕ := 5
def members_per_team : ℕ := 8
def committee_size : ℕ := 11
def nonhost_member_selection (n : ℕ) : ℕ := (n.choose 2) -- Selection of 2 members from non-host teams
def host_member_selection (n : ℕ) : ℕ := (n.choose 2)   -- Selection of 2 members from the remaining members of the host team; captain not considered in this choose as it's already selected

-- The total number of ways to form the required tournament committee
def total_committee_selections : ℕ :=
  num_teams * host_member_selection 7 * (nonhost_member_selection 8)^4

-- Proof stating the solution to the problem
theorem tournament_committee_count :
  total_committee_selections = 64534080 := by
  sorry

end tournament_committee_count_l159_159600


namespace quadratic_function_equal_values_l159_159439

theorem quadratic_function_equal_values (a m n : ℝ) (h : a ≠ 0) (hmn : a * m^2 - 4 * a * m - 3 = a * n^2 - 4 * a * n - 3) : m + n = 4 :=
by
  sorry

end quadratic_function_equal_values_l159_159439


namespace interest_rate_l159_159520

-- Define the given conditions
def principal : ℝ := 4000
def total_interest : ℝ := 630.50
def future_value : ℝ := principal + total_interest
def time : ℝ := 1.5  -- 1 1/2 years
def times_compounded : ℝ := 2  -- Compounded half yearly

-- Statement to prove the annual interest rate
theorem interest_rate (P A t n : ℝ) (hP : P = principal) (hA : A = future_value) 
    (ht : t = time) (hn : n = times_compounded) :
    ∃ r : ℝ, A = P * (1 + r / n) ^ (n * t) ∧ r = 0.1 := 
by 
  sorry

end interest_rate_l159_159520


namespace sufficient_but_not_necessary_l159_159297

theorem sufficient_but_not_necessary (x : ℝ) : (x = -1 → x^2 - 5 * x - 6 = 0) ∧ (∃ y : ℝ, y ≠ -1 ∧ y^2 - 5 * y - 6 = 0) :=
by
  sorry

end sufficient_but_not_necessary_l159_159297


namespace division_by_repeating_decimal_l159_159366

-- Define the repeating decimal as a fraction
def repeating_decimal := 4 / 9

-- Prove the main theorem
theorem division_by_repeating_decimal : 8 / repeating_decimal = 18 :=
by
  -- lean implementation steps
  sorry

end division_by_repeating_decimal_l159_159366


namespace sequence_terms_l159_159785

theorem sequence_terms (S : ℕ → ℤ) (a : ℕ → ℤ) 
  (hS : ∀ n, S n = 3 ^ n - 2) :
  (a 1 = 1) ∧ (∀ n, n ≥ 2 → a n = 2 * 3 ^ (n - 1)) := by
  sorry

end sequence_terms_l159_159785


namespace product_modulo_10_l159_159231

-- Define the numbers involved
def a := 2457
def b := 7623
def c := 91309

-- Define the modulo operation we're interested in
def modulo_10 (n : Nat) : Nat := n % 10

-- State the theorem we want to prove
theorem product_modulo_10 :
  modulo_10 (a * b * c) = 9 :=
sorry

end product_modulo_10_l159_159231


namespace arithmetic_sequence_fourth_term_l159_159534

-- Define the arithmetic sequence and conditions
variable {a : ℕ → ℕ}
variable {S : ℕ → ℕ}

-- Given conditions
def a₂ := 606
def S₄ := 3834

-- Problem statement
theorem arithmetic_sequence_fourth_term :
  (a 1 + a 2 + a 3 = 1818) →
  (a 4 = 2016) :=
sorry

end arithmetic_sequence_fourth_term_l159_159534


namespace area_of_smallest_square_containing_circle_l159_159691

theorem area_of_smallest_square_containing_circle (r : ℝ) (h : r = 5) : 
  ∃ (a : ℝ), a = 100 :=
by
  sorry

end area_of_smallest_square_containing_circle_l159_159691


namespace solve_for_x_l159_159544

theorem solve_for_x (x : ℚ) (h : (2 * x + 18) / (x - 6) = (2 * x - 4) / (x + 10)) : x = -26 / 9 :=
sorry

end solve_for_x_l159_159544


namespace linear_relationship_increase_in_y_l159_159862

theorem linear_relationship_increase_in_y (x y : ℝ) (hx : x = 12) (hy : y = 10 / 4 * x) : y = 30 := by
  sorry

end linear_relationship_increase_in_y_l159_159862


namespace flour_needed_for_dozen_cookies_l159_159497

/--
Matt uses 4 bags of flour, each weighing 5 pounds, to make a total of 120 cookies.
Prove that 2 pounds of flour are needed to make a dozen cookies.
-/
theorem flour_needed_for_dozen_cookies :
  ∀ (bags_of_flour : ℕ) (weight_per_bag : ℕ) (total_cookies : ℕ),
  bags_of_flour = 4 →
  weight_per_bag = 5 →
  total_cookies = 120 →
  (12 * (bags_of_flour * weight_per_bag)) / total_cookies = 2 :=
by
  sorry

end flour_needed_for_dozen_cookies_l159_159497


namespace simple_interest_rate_l159_159025

theorem simple_interest_rate (P T A R : ℝ) (hT : T = 15) (hA : A = 4 * P)
  (hA_simple_interest : A = P + (P * R * T / 100)) : R = 20 :=
by
  sorry

end simple_interest_rate_l159_159025


namespace find_t_l159_159725

theorem find_t (s t : ℤ) (h1 : 12 * s + 7 * t = 173) (h2 : s = t - 3) : t = 11 :=
by
  sorry

end find_t_l159_159725


namespace solution_in_Quadrant_III_l159_159238

theorem solution_in_Quadrant_III {c x y : ℝ} 
    (h1 : x - y = 4) 
    (h2 : c * x + y = 5) 
    (hx : x < 0) 
    (hy : y < 0) : 
    c < -1 := 
sorry

end solution_in_Quadrant_III_l159_159238


namespace solve_for_y_l159_159383

theorem solve_for_y (y : ℕ) (h : 9^y = 3^12) : y = 6 :=
by {
  sorry
}

end solve_for_y_l159_159383


namespace evaluate_expression_at_x_eq_3_l159_159513

theorem evaluate_expression_at_x_eq_3 :
  (3 ^ 3) ^ (3 ^ 3) = 7625597484987 := by
  sorry

end evaluate_expression_at_x_eq_3_l159_159513


namespace num_of_poly_sci_majors_l159_159868

-- Define the total number of applicants
def total_applicants : ℕ := 40

-- Define the number of applicants with GPA > 3.0
def gpa_higher_than_3_point_0 : ℕ := 20

-- Define the number of applicants who did not major in political science and had GPA ≤ 3.0
def non_poly_sci_and_low_gpa : ℕ := 10

-- Define the number of political science majors with GPA > 3.0
def poly_sci_with_high_gpa : ℕ := 5

-- Prove the number of political science majors
theorem num_of_poly_sci_majors : ∀ (P : ℕ),
  P = poly_sci_with_high_gpa + 
      (total_applicants - non_poly_sci_and_low_gpa - 
       (gpa_higher_than_3_point_0 - poly_sci_with_high_gpa)) → 
  P = 20 :=
by
  intros P h
  sorry

end num_of_poly_sci_majors_l159_159868


namespace math_problem_solution_l159_159051

theorem math_problem_solution (a b n : ℕ) (p : ℕ) (h_prime : Nat.Prime p) (h_eq : a ^ 2013 + b ^ 2013 = p ^ n) :
  ∃ k : ℕ, a = 2 ^ k ∧ b = 2 ^ k ∧ n = 2013 * k + 1 ∧ p = 2 :=
sorry

end math_problem_solution_l159_159051


namespace linear_correlation_l159_159248

variable (r : ℝ) (r_critical : ℝ)

theorem linear_correlation (h1 : r = -0.9362) (h2 : r_critical = 0.8013) :
  |r| > r_critical :=
by
  sorry

end linear_correlation_l159_159248


namespace population_growth_l159_159499

theorem population_growth (P_present P_future : ℝ) (r : ℝ) (n : ℕ)
  (h1 : P_present = 7800)
  (h2 : P_future = 10860.72)
  (h3 : n = 2) :
  P_future = P_present * (1 + r / 100)^n → r = 18.03 :=
by sorry

end population_growth_l159_159499


namespace area_of_trapezium_l159_159556

def length_parallel_side1 : ℝ := 20
def length_parallel_side2 : ℝ := 18
def distance_between_sides : ℝ := 15
def expected_area : ℝ := 285

theorem area_of_trapezium :
  (1 / 2) * (length_parallel_side1 + length_parallel_side2) * distance_between_sides = expected_area :=
by sorry

end area_of_trapezium_l159_159556


namespace g_product_of_roots_l159_159948

def f (x : ℂ) : ℂ := x^6 + x^3 + 1
def g (x : ℂ) : ℂ := x^2 + 1

theorem g_product_of_roots (x_1 x_2 x_3 x_4 x_5 x_6 : ℂ) 
    (h1 : ∀ x, (x - x_1) * (x - x_2) * (x - x_3) * (x - x_4) * (x - x_5) * (x - x_6) = f x) :
    g x_1 * g x_2 * g x_3 * g x_4 * g x_5 * g x_6 = 1 :=
by 
    sorry

end g_product_of_roots_l159_159948


namespace unique_b_positive_solution_l159_159458

theorem unique_b_positive_solution (c : ℝ) (h : c ≠ 0) : 
  (∃ b : ℝ, b > 0 ∧ ∀ b : ℝ, b ≠ 0 → 
    ∀ x : ℝ, x^2 + (b + 1 / b) * x + c = 0 → x = - (b + 1 / b) / 2) 
  ↔ c = (5 + Real.sqrt 21) / 2 ∨ c = (5 - Real.sqrt 21) / 2 := 
by {
  sorry
}

end unique_b_positive_solution_l159_159458


namespace installation_time_l159_159735

-- Definitions (based on conditions)
def total_windows := 14
def installed_windows := 8
def hours_per_window := 8

-- Define what we need to prove
def remaining_windows := total_windows - installed_windows
def total_install_hours := remaining_windows * hours_per_window

theorem installation_time : total_install_hours = 48 := by
  sorry

end installation_time_l159_159735


namespace B_alone_finishes_in_21_days_l159_159585

theorem B_alone_finishes_in_21_days (W_A W_B : ℝ) (h1 : W_A = 0.5 * W_B) (h2 : W_A + W_B = 1 / 14) : W_B = 1 / 21 :=
by sorry

end B_alone_finishes_in_21_days_l159_159585


namespace convert_yahs_to_bahs_l159_159103

theorem convert_yahs_to_bahs :
  (∀ (bahs rahs yahs : ℝ), (10 * bahs = 18 * rahs) 
    ∧ (6 * rahs = 10 * yahs) 
    → (1500 * yahs / (10 / 6) / (18 / 10) = 500 * bahs)) :=
by
  intros bahs rahs yahs h
  sorry

end convert_yahs_to_bahs_l159_159103


namespace tom_received_20_percent_bonus_l159_159791

-- Define the initial conditions
def tom_spent : ℤ := 250
def gems_per_dollar : ℤ := 100
def total_gems_received : ℤ := 30000

-- Calculate the number of gems received without the bonus
def gems_without_bonus : ℤ := tom_spent * gems_per_dollar
def bonus_gems : ℤ := total_gems_received - gems_without_bonus

-- Calculate the percentage of the bonus
def bonus_percentage : ℚ := (bonus_gems : ℚ) / gems_without_bonus * 100

-- State the theorem
theorem tom_received_20_percent_bonus : bonus_percentage = 20 := by
  sorry

end tom_received_20_percent_bonus_l159_159791


namespace maple_trees_planted_plant_maple_trees_today_l159_159113

-- Define the initial number of maple trees
def initial_maple_trees : ℕ := 2

-- Define the number of maple trees the park will have after planting
def final_maple_trees : ℕ := 11

-- Define the number of popular trees, though it is irrelevant for the proof
def initial_popular_trees : ℕ := 5

-- The main statement to prove: number of maple trees planted today
theorem maple_trees_planted : ℕ :=
  final_maple_trees - initial_maple_trees

-- Prove that the number of maple trees planted today is 9
theorem plant_maple_trees_today :
  maple_trees_planted = 9 :=
by
  sorry

end maple_trees_planted_plant_maple_trees_today_l159_159113


namespace rachel_total_clothing_l159_159851

def box_1_scarves : ℕ := 2
def box_1_mittens : ℕ := 3
def box_1_hats : ℕ := 1
def box_2_scarves : ℕ := 4
def box_2_mittens : ℕ := 2
def box_2_hats : ℕ := 2
def box_3_scarves : ℕ := 1
def box_3_mittens : ℕ := 5
def box_3_hats : ℕ := 3
def box_4_scarves : ℕ := 3
def box_4_mittens : ℕ := 4
def box_4_hats : ℕ := 1
def box_5_scarves : ℕ := 5
def box_5_mittens : ℕ := 3
def box_5_hats : ℕ := 2
def box_6_scarves : ℕ := 2
def box_6_mittens : ℕ := 6
def box_6_hats : ℕ := 0
def box_7_scarves : ℕ := 4
def box_7_mittens : ℕ := 1
def box_7_hats : ℕ := 3
def box_8_scarves : ℕ := 3
def box_8_mittens : ℕ := 2
def box_8_hats : ℕ := 4
def box_9_scarves : ℕ := 1
def box_9_mittens : ℕ := 4
def box_9_hats : ℕ := 5

def total_clothing : ℕ := 
  box_1_scarves + box_1_mittens + box_1_hats +
  box_2_scarves + box_2_mittens + box_2_hats +
  box_3_scarves + box_3_mittens + box_3_hats +
  box_4_scarves + box_4_mittens + box_4_hats +
  box_5_scarves + box_5_mittens + box_5_hats +
  box_6_scarves + box_6_mittens + box_6_hats +
  box_7_scarves + box_7_mittens + box_7_hats +
  box_8_scarves + box_8_mittens + box_8_hats +
  box_9_scarves + box_9_mittens + box_9_hats

theorem rachel_total_clothing : total_clothing = 76 :=
by
  sorry

end rachel_total_clothing_l159_159851


namespace xyz_plus_54_l159_159678

theorem xyz_plus_54 (x y z : ℕ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z)
  (h1 : x * y + z = 53) (h2 : y * z + x = 53) (h3 : z * x + y = 53) : 
  x + y + z = 54 := by
  sorry

end xyz_plus_54_l159_159678


namespace geometric_seq_a6_value_l159_159574

theorem geometric_seq_a6_value 
    (a : ℕ → ℝ) 
    (q : ℝ) 
    (h_q_pos : q > 0)
    (h_a_pos : ∀ n, a n > 0)
    (h_a2 : a 2 = 1)
    (h_a8_eq : a 8 = a 6 + 2 * a 4) : 
    a 6 = 4 := 
by 
  sorry

end geometric_seq_a6_value_l159_159574


namespace remainder_of_modified_division_l159_159334

theorem remainder_of_modified_division (x y u v : ℕ) (hx : 0 ≤ v ∧ v < y) (hxy : x = u * y + v) :
  ((x + 3 * u * y) % y) = v := by
  sorry

end remainder_of_modified_division_l159_159334


namespace spa_polish_total_digits_l159_159146

theorem spa_polish_total_digits (girls : ℕ) (digits_per_girl : ℕ) (total_digits : ℕ)
  (h1 : girls = 5) (h2 : digits_per_girl = 20) : total_digits = 100 :=
by
  sorry

end spa_polish_total_digits_l159_159146


namespace proposition_4_l159_159311

theorem proposition_4 (x y ε : ℝ) (h1 : |x - 2| < ε) (h2 : |y - 2| < ε) : |x - y| < 2 * ε :=
by
  sorry

end proposition_4_l159_159311


namespace complex_seventh_root_of_unity_l159_159082

theorem complex_seventh_root_of_unity (r : ℂ) (h1 : r^7 = 1) (h2: r ≠ 1) : 
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) * (r^6 - 1) = 7 :=
by
  sorry

end complex_seventh_root_of_unity_l159_159082


namespace imaginary_part_of_z_l159_159406

theorem imaginary_part_of_z (z : ℂ) (h : z * (1 + I) = 2 - 2 * I) : z.im = -2 :=
sorry

end imaginary_part_of_z_l159_159406


namespace correct_weights_swapped_l159_159261

theorem correct_weights_swapped 
  (W X Y Z : ℝ) 
  (h1 : Z > Y) 
  (h2 : X > W) 
  (h3 : Y + Z > W + X) :
  (W, Z) = (Z, W) :=
sorry

end correct_weights_swapped_l159_159261


namespace Kamal_biology_marks_l159_159135

theorem Kamal_biology_marks 
  (E : ℕ) (M : ℕ) (P : ℕ) (C : ℕ) (A : ℕ) (N : ℕ) (B : ℕ) 
  (hE : E = 66)
  (hM : M = 65)
  (hP : P = 77)
  (hC : C = 62)
  (hA : A = 69)
  (hN : N = 5)
  (h_total : N * A = E + M + P + C + B) 
  : B = 75 :=
by
  sorry

end Kamal_biology_marks_l159_159135


namespace red_ball_prob_gt_black_ball_prob_l159_159525

theorem red_ball_prob_gt_black_ball_prob (m : ℕ) (h : 8 > m) : m ≠ 10 :=
by
  sorry

end red_ball_prob_gt_black_ball_prob_l159_159525


namespace measure_angle_B_triangle_area_correct_l159_159058

noncomputable def triangle_angle_B (a b c : ℝ) (A B C : ℝ) : Prop :=
  let m : ℝ × ℝ := (Real.sin C - Real.sin A, Real.sin C - Real.sin B)
  let n : ℝ × ℝ := (b + c, a)
  a * (Real.sin C - Real.sin A) = (b + c) * (Real.sin C - Real.sin B) → B = Real.pi / 3

noncomputable def triangle_area (a b c : ℝ) (A B C : ℝ) : Prop :=
  let area1 := (3 + Real.sqrt 3)
  let area2 := Real.sqrt 3
  let m : ℝ × ℝ := (Real.sin C - Real.sin A, Real.sin C - Real.sin B)
  let n : ℝ × ℝ := (b + c, a)
  a * (Real.sin C - Real.sin A) = (b + c) * (Real.sin C - Real.sin B) →
  b = 2 * Real.sqrt 3 →
  c = Real.sqrt 6 + Real.sqrt 2 →
  let sinA1 := (Real.sqrt 2 / 2)
  let sinA2 := (Real.sqrt 6 - Real.sqrt 2) / 4
  let S1 := (1 / 2) * b * c * sinA1
  let S2 := (1 / 2) * b * c * sinA2
  S1 = area1 ∨ S2 = area2

theorem measure_angle_B :
  ∀ (a b c A B C : ℝ),
    triangle_angle_B a b c A B C := sorry

theorem triangle_area_correct :
  ∀ (a b c A B C : ℝ),
    triangle_area a b c A B C := sorry

end measure_angle_B_triangle_area_correct_l159_159058


namespace value_of_x_squared_plus_9y_squared_l159_159100

theorem value_of_x_squared_plus_9y_squared (x y : ℝ)
  (h1 : x + 3 * y = 5)
  (h2 : x * y = -8) : x^2 + 9 * y^2 = 73 :=
by
  sorry

end value_of_x_squared_plus_9y_squared_l159_159100


namespace multiple_of_Roseville_population_l159_159296

noncomputable def Willowdale_population : ℕ := 2000

noncomputable def Roseville_population : ℕ :=
  (3 * Willowdale_population) - 500

noncomputable def SunCity_population : ℕ := 12000

theorem multiple_of_Roseville_population :
  ∃ m : ℕ, SunCity_population = (m * Roseville_population) + 1000 ∧ m = 2 :=
by
  sorry

end multiple_of_Roseville_population_l159_159296


namespace sum_of_three_numbers_l159_159502

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 138) 
  (h2 : ab + bc + ca = 131) : 
  a + b + c = 20 := 
by sorry

end sum_of_three_numbers_l159_159502


namespace arithmetic_seq_problem_l159_159452

open Nat

def arithmetic_sequence (a : ℕ → ℚ) (a1 d : ℚ) : Prop :=
  ∀ n : ℕ, a n = a1 + n * d

theorem arithmetic_seq_problem :
  ∃ (a : ℕ → ℚ) (a1 d : ℚ),
    (arithmetic_sequence a a1 d) ∧
    (a 2 + a 3 + a 4 = 3) ∧
    (a 7 = 8) ∧
    (a 11 = 15) :=
  sorry

end arithmetic_seq_problem_l159_159452


namespace min_colors_for_distance_six_l159_159653

/-
Definitions and conditions:
- The board is an infinite checkered paper with a cell side of one unit.
- The distance between two cells is the length of the shortest path of a rook from one cell to another.

Statement:
- Prove that the minimum number of colors needed to color the board such that two cells that are a distance of 6 apart are always painted different colors is 4.
-/

def cell := (ℤ × ℤ)

def rook_distance (c1 c2 : cell) : ℤ :=
  |c1.1 - c2.1| + |c1.2 - c2.2|

theorem min_colors_for_distance_six : ∃ (n : ℕ), (∀ (f : cell → ℕ), (∀ c1 c2, rook_distance c1 c2 = 6 → f c1 ≠ f c2) → n ≤ 4) :=
by
  sorry

end min_colors_for_distance_six_l159_159653


namespace root_of_quadratic_eq_l159_159919

open Complex

theorem root_of_quadratic_eq :
  ∃ z1 z2 : ℂ, (z1 = 3.5 - I) ∧ (z2 = -2.5 + I) ∧ (∀ z : ℂ, z^2 - z = 6 - 6 * I → (z = z1 ∨ z = z2)) := 
sorry

end root_of_quadratic_eq_l159_159919


namespace rectangle_length_l159_159696

theorem rectangle_length (side_square length_rectangle width_rectangle wire_length : ℝ) 
    (h1 : side_square = 12) 
    (h2 : width_rectangle = 6) 
    (h3 : wire_length = 4 * side_square) 
    (h4 : wire_length = 2 * width_rectangle + 2 * length_rectangle) : 
    length_rectangle = 18 := 
by 
  sorry

end rectangle_length_l159_159696


namespace book_pages_read_l159_159630

theorem book_pages_read (pages_per_day : ℕ) (days_per_week : ℕ) (weeks : ℕ) (total_pages : ℕ) :
  (pages_per_day = 100) →
  (days_per_week = 3) →
  (weeks = 7) →
  total_pages = pages_per_day * days_per_week * weeks →
  total_pages = 2100 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end book_pages_read_l159_159630


namespace fruit_problem_l159_159765

def number_of_pears (A : ℤ) : ℤ := (3 * A) / 5
def number_of_apples (B : ℤ) : ℤ := (3 * B) / 7

theorem fruit_problem
  (A B : ℤ)
  (h1 : A + B = 82)
  (h2 : abs (A - B) < 10)
  (x : ℤ := (2 * A) / 5)
  (y : ℤ := (4 * B) / 7) :
  number_of_pears A = 24 ∧ number_of_apples B = 18 :=
by
  sorry

end fruit_problem_l159_159765


namespace solution_amount_of_solution_A_l159_159797

-- Define the conditions
variables (x y : ℝ)
variables (h1 : x + y = 140)
variables (h2 : 0.40 * x + 0.90 * y = 0.80 * 140)

-- State the theorem
theorem solution_amount_of_solution_A : x = 28 :=
by
  -- Here, the proof would be provided, but we replace it with sorry
  sorry

end solution_amount_of_solution_A_l159_159797


namespace fraction_flower_beds_l159_159199

theorem fraction_flower_beds (length1 length2 height triangle_area yard_area : ℝ) (h1 : length1 = 18) (h2 : length2 = 30) (h3 : height = 10) (h4 : triangle_area = 2 * (1 / 2 * (6 ^ 2))) (h5 : yard_area = ((length1 + length2) / 2) * height) : 
  (triangle_area / yard_area) = 3 / 20 :=
by 
  sorry

end fraction_flower_beds_l159_159199


namespace two_positive_numbers_inequality_three_positive_numbers_inequality_l159_159843

theorem two_positive_numbers_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (a + b) * (1 / a + 1 / b) ≥ 4 :=
by sorry

theorem three_positive_numbers_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b + c) * (1 / a + 1 / b + 1 / c) ≥ 9 :=
by sorry

end two_positive_numbers_inequality_three_positive_numbers_inequality_l159_159843


namespace find_1993_star_1935_l159_159293

axiom star (x y : ℕ) : ℕ

axiom star_self {x : ℕ} : star x x = 0
axiom star_assoc {x y z : ℕ} : star x (star y z) = star x y + z

theorem find_1993_star_1935 : star 1993 1935 = 58 :=
by
  sorry

end find_1993_star_1935_l159_159293


namespace rearrange_letters_no_adjacent_repeats_l159_159264

-- Factorial function
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Problem conditions
def distinct_permutations (word : String) (freq_I : ℕ) (freq_L : ℕ) : ℕ :=
  factorial (String.length word) / (factorial freq_I * factorial freq_L)

-- No-adjacent-repeated permutations
def no_adjacent_repeats (word : String) (freq_I : ℕ) (freq_L : ℕ) : ℕ :=
  let total_permutations := distinct_permutations word freq_I freq_L
  let i_superletter_permutations := distinct_permutations (String.dropRight word 1) (freq_I - 1) freq_L
  let l_superletter_permutations := distinct_permutations (String.dropRight word 1) freq_I (freq_L - 1)
  let both_superletter_permutations := factorial (String.length word - 2)
  total_permutations - (i_superletter_permutations + l_superletter_permutations - both_superletter_permutations)

-- Given problem definition
def word := "BRILLIANT"
def freq_I := 2
def freq_L := 2

-- Proof problem statement
theorem rearrange_letters_no_adjacent_repeats :
  no_adjacent_repeats word freq_I freq_L = 55440 := by
  sorry

end rearrange_letters_no_adjacent_repeats_l159_159264


namespace team_X_finishes_with_more_points_than_Y_l159_159579

-- Define the number of teams and games played
def numberOfTeams : ℕ := 8
def gamesPerTeam : ℕ := numberOfTeams - 1

-- Define the probability of winning (since each team has a 50% chance to win any game)
def probOfWin : ℝ := 0.5

-- Define the event that team X finishes with more points than team Y
noncomputable def probXFinishesMorePointsThanY : ℝ := 1 / 2

-- Statement to be proved: 
theorem team_X_finishes_with_more_points_than_Y :
  (∃ p : ℝ, p = probXFinishesMorePointsThanY) :=
sorry

end team_X_finishes_with_more_points_than_Y_l159_159579


namespace smallest_M_l159_159945

def Q (M : ℕ) := (2 * M / 3 + 1) / (M + 1)

theorem smallest_M (M : ℕ) (h : M % 6 = 0) (h_pos : 0 < M) : 
  (∃ k, M = 6 * k ∧ Q M < 3 / 4) ↔ M = 6 := 
by 
  sorry

end smallest_M_l159_159945


namespace Sam_scored_points_l159_159156

theorem Sam_scored_points (total_points friend_points S: ℕ) (h1: friend_points = 12) (h2: total_points = 87) (h3: total_points = S + friend_points) : S = 75 :=
by
  sorry

end Sam_scored_points_l159_159156


namespace number_of_zeros_l159_159736

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then |x| - 2 else 2 * x - 6 + Real.log x

theorem number_of_zeros :
  (∃ x : ℝ, f x = 0) ∧ (∃ y : ℝ, f y = 0) ∧ (∀ z : ℝ, f z = 0 → z = x ∨ z = y) :=
by
  sorry

end number_of_zeros_l159_159736


namespace chord_length_eq_l159_159753

def line_eq (x y : ℝ) : Prop := 3 * x + 4 * y - 5 = 0
def circle_eq (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 4

theorem chord_length_eq : 
  ∀ (x y : ℝ), 
  (line_eq x y) ∧ (circle_eq x y) → 
  ∃ l, l = 2 * Real.sqrt 3 :=
sorry

end chord_length_eq_l159_159753


namespace john_bought_more_than_ray_l159_159094

variable (R_c R_d M_c M_d J_c J_d : ℕ)

-- Define the conditions
def conditions : Prop :=
  (R_c = 10) ∧
  (R_d = 3) ∧
  (M_c = R_c + 6) ∧
  (M_d = R_d + 1) ∧
  (J_c = M_c + 5) ∧
  (J_d = M_d + 2)

-- Define the question
def john_more_chickens_and_ducks (J_c R_c J_d R_d : ℕ) : ℕ :=
  (J_c - R_c) + (J_d - R_d)

-- The proof problem statement
theorem john_bought_more_than_ray :
  conditions R_c R_d M_c M_d J_c J_d → john_more_chickens_and_ducks J_c R_c J_d R_d = 14 :=
by
  intro h
  sorry

end john_bought_more_than_ray_l159_159094


namespace roots_separation_condition_l159_159818

theorem roots_separation_condition (m n p q : ℝ)
  (h_1 : ∃ (x1 x2 : ℝ), x1 + x2 = -m ∧ x1 * x2 = n ∧ x1 ≠ x2)
  (h_2 : ∃ (x3 x4 : ℝ), x3 + x4 = -p ∧ x3 * x4 = q ∧ x3 ≠ x4)
  (h_3 : (∀ x1 x2 x3 x4 : ℝ, x1 + x2 = -m ∧ x1 * x2 = n ∧ x3 + x4 = -p ∧ x3 * x4 = q → 
         (x3 - x1) * (x3 - x2) * (x4 - x1) * (x4 - x2) < 0)) : 
  (n - q)^2 + (m - p) * (m * q - n * p) < 0 :=
sorry

end roots_separation_condition_l159_159818


namespace finding_breadth_and_length_of_floor_l159_159949

noncomputable def length_of_floor (b : ℝ) := 3 * b
noncomputable def area_of_floor (b : ℝ) := (length_of_floor b) * b

theorem finding_breadth_and_length_of_floor
  (breadth : ℝ)
  (length : ℝ := length_of_floor breadth)
  (area : ℝ := area_of_floor breadth)
  (painting_cost : ℝ)
  (cost_per_sqm : ℝ)
  (h1 : painting_cost = 100)
  (h2 : cost_per_sqm = 2)
  (h3 : area = painting_cost / cost_per_sqm) :
  length = Real.sqrt 150 :=
by
  sorry

end finding_breadth_and_length_of_floor_l159_159949


namespace minimal_sum_of_squares_of_roots_l159_159761

open Real

theorem minimal_sum_of_squares_of_roots :
  ∀ a : ℝ,
  (let x1 := 3*a + 1;
   let x2 := 2*a^2 - 3*a - 2;
   (a^2 + 18*a + 9) ≥ 0 →
   (x1^2 - 2*x2) = (5*a^2 + 12*a + 5) →
   a = -9 + 6*sqrt 2) :=
by
  sorry

end minimal_sum_of_squares_of_roots_l159_159761


namespace probability_sum_of_five_l159_159681

def total_outcomes : ℕ := 36
def favorable_outcomes : ℕ := 4

theorem probability_sum_of_five :
  favorable_outcomes / total_outcomes = 1 / 9 := 
by
  sorry

end probability_sum_of_five_l159_159681


namespace poly_roots_equivalence_l159_159769

noncomputable def poly (a b c d : ℝ) (x : ℝ) : ℝ := x^4 + a * x^3 + b * x^2 + c * x + d

theorem poly_roots_equivalence (a b c d : ℝ) 
    (h1 : poly a b c d 4 = 102) 
    (h2 : poly a b c d 3 = 102) 
    (h3 : poly a b c d (-3) = 102) 
    (h4 : poly a b c d (-4) = 102) : 
    {x : ℝ | poly a b c d x = 246} = {0, 5, -5} := 
by 
    sorry

end poly_roots_equivalence_l159_159769


namespace line_graph_displays_trend_l159_159433

-- Define the types of statistical graphs
inductive StatisticalGraph : Type
| barGraph : StatisticalGraph
| lineGraph : StatisticalGraph
| pieChart : StatisticalGraph
| histogram : StatisticalGraph

-- Define the property of displaying trends over time
def displaysTrend (g : StatisticalGraph) : Prop := 
  g = StatisticalGraph.lineGraph

-- Theorem to prove that the type of statistical graph that displays the trend of data is the line graph
theorem line_graph_displays_trend : displaysTrend StatisticalGraph.lineGraph :=
sorry

end line_graph_displays_trend_l159_159433


namespace a_neg_half_not_bounded_a_bounded_range_l159_159117

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  1 + a * (1/3)^x + (1/9)^x

theorem a_neg_half_not_bounded (a : ℝ) :
  a = -1/2 → ¬(∃ M > 0, ∀ x < 0, |f x a| ≤ M) :=
by
  sorry

theorem a_bounded_range (a : ℝ) : 
  (∀ x ≥ 0, |f x a| ≤ 4) → -6 ≤ a ∧ a ≤ 2 :=
by
  sorry

end a_neg_half_not_bounded_a_bounded_range_l159_159117


namespace cost_price_of_article_l159_159924

theorem cost_price_of_article
  (C SP1 SP2 : ℝ)
  (h1 : SP1 = 0.8 * C)
  (h2 : SP2 = 1.05 * C)
  (h3 : SP2 = SP1 + 100) : 
  C = 400 := 
sorry

end cost_price_of_article_l159_159924


namespace geometric_sequence_a5_l159_159505

theorem geometric_sequence_a5 (a : ℕ → ℝ) (q : ℝ) 
  (h1 : a 3 = 6)
  (h2 : a 3 + a 5 + a 7 = 78)
  (h_geom : ∀ n, a (n + 1) = a n * q) : 
  a 5 = 18 :=
by sorry

end geometric_sequence_a5_l159_159505


namespace rectangle_perimeter_eq_l159_159990

noncomputable def rectangle_perimeter (x y : ℝ) := 2 * (x + y)

theorem rectangle_perimeter_eq (x y a b : ℝ)
  (h_area_rect : x * y = 2450)
  (h_area_ellipse : a * b = 2450)
  (h_foci_distance : x + y = 2 * a)
  (h_diag : x^2 + y^2 = 4 * (a^2 - b^2))
  (h_b : b = Real.sqrt (a^2 - 1225))
  : rectangle_perimeter x y = 120 * Real.sqrt 17 := by
  sorry

end rectangle_perimeter_eq_l159_159990


namespace similar_triangles_area_ratio_l159_159854

theorem similar_triangles_area_ratio (r : ℚ) (h : r = 1/3) : (r^2) = 1/9 :=
by
  sorry

end similar_triangles_area_ratio_l159_159854


namespace gcd_8Tn_nplus1_eq_4_l159_159874

noncomputable def T_n (n : ℕ) : ℕ :=
(n * (n + 1)) / 2

theorem gcd_8Tn_nplus1_eq_4 (n : ℕ) (hn: 0 < n) : gcd (8 * T_n n) (n + 1) = 4 :=
sorry

end gcd_8Tn_nplus1_eq_4_l159_159874


namespace cheese_cookies_price_is_correct_l159_159554

-- Define the problem conditions and constants
def total_boxes_per_carton : ℕ := 15
def total_packs_per_box : ℕ := 12
def discount_15_percent : ℝ := 0.15
def total_number_of_cartons : ℕ := 13
def total_cost_paid : ℝ := 2058

-- Calculate the expected price per pack
noncomputable def price_per_pack : ℝ :=
  let total_packs := total_boxes_per_carton * total_packs_per_box * total_number_of_cartons
  let total_cost_without_discount := total_cost_paid / (1 - discount_15_percent)
  total_cost_without_discount / total_packs

theorem cheese_cookies_price_is_correct : 
  abs (price_per_pack - 1.0347) < 0.0001 :=
by sorry

end cheese_cookies_price_is_correct_l159_159554


namespace solve_for_x_l159_159980

theorem solve_for_x : ∃ x : ℚ, 7 * (4 * x + 3) - 3 = -3 * (2 - 5 * x) + 5 * x / 2 ∧ x = -16 / 7 := by
  sorry

end solve_for_x_l159_159980


namespace car_distance_l159_159744

-- Define the conditions
def speed := 162  -- speed of the car in km/h
def time := 5     -- time taken in hours

-- Define the distance calculation
def distance (s : ℕ) (t : ℕ) : ℕ := s * t

-- State the theorem
theorem car_distance : distance speed time = 810 := by
  -- Proof goes here
  sorry

end car_distance_l159_159744


namespace natural_solution_unique_l159_159095

theorem natural_solution_unique (n : ℕ) (h : (2 * n - 1) / n^5 = 3 - 2 / n) : n = 1 := by
  sorry

end natural_solution_unique_l159_159095


namespace Shekar_science_marks_l159_159931

-- Define Shekar's known marks
def math_marks : ℕ := 76
def social_studies_marks : ℕ := 82
def english_marks : ℕ := 47
def biology_marks : ℕ := 85

-- Define the average mark and the number of subjects
def average_mark : ℕ := 71
def number_of_subjects : ℕ := 5

-- Define Shekar's unknown mark in Science
def science_marks : ℕ := sorry  -- We expect to prove science_marks = 65

-- State the theorem to be proved
theorem Shekar_science_marks :
  average_mark * number_of_subjects = math_marks + science_marks + social_studies_marks + english_marks + biology_marks →
  science_marks = 65 :=
by sorry

end Shekar_science_marks_l159_159931


namespace card_length_l159_159955

noncomputable def width_card : ℕ := 2
noncomputable def side_poster_board : ℕ := 12
noncomputable def total_cards : ℕ := 24

theorem card_length :
  ∃ (card_length : ℕ),
    (side_poster_board / width_card) * (side_poster_board / card_length) = total_cards ∧ 
    card_length = 3 := by
  sorry

end card_length_l159_159955


namespace train_length_l159_159716

theorem train_length (L : ℝ) (h1 : ∀ t1 : ℝ, t1 = 15 → ∀ p1 : ℝ, p1 = 180 → (L + p1) / t1 = v)
(h2 : ∀ t2 : ℝ, t2 = 20 → ∀ p2 : ℝ, p2 = 250 → (L + p2) / t2 = v) : 
L = 30 :=
by
  have h1 := h1 15 rfl 180 rfl
  have h2 := h2 20 rfl 250 rfl
  sorry

end train_length_l159_159716


namespace smallest_four_digit_divisible_by_9_l159_159115

theorem smallest_four_digit_divisible_by_9 
    (n : ℕ) 
    (h1 : 1000 ≤ n ∧ n < 10000) 
    (h2 : n % 9 = 0)
    (h3 : n % 10 % 2 = 1)
    (h4 : (n / 1000) % 2 = 1)
    (h5 : (n / 10) % 10 % 2 = 0)
    (h6 : (n / 100) % 10 % 2 = 0) :
  n = 3609 :=
sorry

end smallest_four_digit_divisible_by_9_l159_159115


namespace triangle_geometry_l159_159824

theorem triangle_geometry 
  (A : ℝ × ℝ) 
  (hA : A = (5,1))
  (median_CM : ∀ x y : ℝ, 2 * x - y - 5 = 0)
  (altitude_BH : ∀ x y : ℝ, x - 2 * y - 5 = 0):
  (∀ x y : ℝ, 2 * x + y - 11 = 0) ∧
  (4, 3) ∈ {(x, y) | 2 * x + y = 11 ∧ 2 * x - y = 5} :=
by
  sorry

end triangle_geometry_l159_159824


namespace back_seat_capacity_l159_159976

def left_seats : Nat := 15
def right_seats : Nat := left_seats - 3
def seats_per_person : Nat := 3
def total_capacity : Nat := 92
def regular_seats_people : Nat := (left_seats + right_seats) * seats_per_person

theorem back_seat_capacity :
  total_capacity - regular_seats_people = 11 :=
by
  sorry

end back_seat_capacity_l159_159976


namespace interest_rate_calculation_l159_159105

theorem interest_rate_calculation (P : ℝ) (r : ℝ) (h1 : P * (1 + r / 100)^3 = 800) (h2 : P * (1 + r / 100)^4 = 820) :
  r = 2.5 := 
  sorry

end interest_rate_calculation_l159_159105


namespace degree_to_radian_radian_to_degree_l159_159302

theorem degree_to_radian (d : ℝ) (rad : ℝ) (deg : ℝ) :
  (180 * rad = π) → (d = 210) → rad = (π / 180) → d * rad = 7 * π / 6 :=
by sorry 

theorem radian_to_degree (r : ℝ) (rad : ℝ) (deg : ℝ) :
  (180 * rad = π) → (r = -5 * π / 2) → deg = (180 / π) → r * deg = -450 :=
by sorry

end degree_to_radian_radian_to_degree_l159_159302


namespace sum_of_fractions_l159_159787

def S_1 : List ℚ := List.range' 1 10 |>.map (λ n => n / 10)
def S_2 : List ℚ := List.replicate 4 (20 / 10)

def total_sum : ℚ := S_1.sum + S_2.sum

theorem sum_of_fractions : total_sum = 12.5 := by
  sorry

end sum_of_fractions_l159_159787


namespace percentage_of_dogs_l159_159397

theorem percentage_of_dogs (total_pets : ℕ) (percent_cats : ℕ) (bunnies : ℕ) 
  (h1 : total_pets = 36) (h2 : percent_cats = 50) (h3 : bunnies = 9) : 
  ((total_pets - ((percent_cats * total_pets) / 100) - bunnies) / total_pets * 100) = 25 := by
  sorry

end percentage_of_dogs_l159_159397


namespace base_eight_to_base_ten_l159_159646

theorem base_eight_to_base_ten : (4 * 8^1 + 5 * 8^0 = 37) := by
  sorry

end base_eight_to_base_ten_l159_159646


namespace sqrt_calc_l159_159718

theorem sqrt_calc : Real.sqrt (Real.sqrt (0.00032 ^ (1 / 5))) = 0.669 := by
  sorry

end sqrt_calc_l159_159718


namespace monthly_salary_l159_159781

theorem monthly_salary (S : ℝ) (h1 : 0.20 * S + 1.20 * 0.80 * S = S) (h2 : S - 1.20 * 0.80 * S = 260) : S = 6500 :=
by
  sorry

end monthly_salary_l159_159781


namespace largest_exponent_l159_159372

theorem largest_exponent : 
  ∀ (a b c d e : ℕ), a = 2^5000 → b = 3^4000 → c = 4^3000 → d = 5^2000 → e = 6^1000 → b > a ∧ b > c ∧ b > d ∧ b > e :=
by
  sorry

end largest_exponent_l159_159372


namespace vertical_angles_congruent_l159_159873

-- Define what it means for two angles to be vertical angles
def areVerticalAngles (a b : ℝ) : Prop := -- placeholder definition
  sorry

-- Define what it means for two angles to be congruent
def areCongruentAngles (a b : ℝ) : Prop := a = b

-- State the problem in the form of a theorem
theorem vertical_angles_congruent (a b : ℝ) :
  areVerticalAngles a b → areCongruentAngles a b := by
  sorry

end vertical_angles_congruent_l159_159873


namespace frac_square_between_half_and_one_l159_159163

theorem frac_square_between_half_and_one :
  let fraction := (11 : ℝ) / 12
  let expr := fraction^2
  (1 / 2) < expr ∧ expr < 1 :=
by
  let fraction := (11 : ℝ) / 12
  let expr := fraction^2
  have h1 : (1 / 2) < expr := sorry
  have h2 : expr < 1 := sorry
  exact ⟨h1, h2⟩

end frac_square_between_half_and_one_l159_159163


namespace units_digit_of_7_pow_2500_l159_159230

theorem units_digit_of_7_pow_2500 : (7^2500) % 10 = 1 :=
by
  -- Variables and constants can be used to formalize steps if necessary, 
  -- but focus is on the statement itself.
  -- Sorry is used to skip the proof part.
  sorry

end units_digit_of_7_pow_2500_l159_159230


namespace length_of_second_offset_l159_159559

theorem length_of_second_offset (d₁ d₂ h₁ A : ℝ) (h_d₁ : d₁ = 30) (h_h₁ : h₁ = 9) (h_A : A = 225):
  ∃ h₂, (A = (1/2) * d₁ * h₁ + (1/2) * d₁ * h₂) → h₂ = 6 := by
  sorry

end length_of_second_offset_l159_159559


namespace prime_cube_plus_five_implies_prime_l159_159763

theorem prime_cube_plus_five_implies_prime (p : ℕ) 
  (hp : Nat.Prime p) 
  (hq : Nat.Prime (p^3 + 5)) : p^5 - 7 = 25 := 
by
  sorry

end prime_cube_plus_five_implies_prime_l159_159763


namespace average_distance_to_sides_l159_159996

open Real

noncomputable def side_length : ℝ := 15
noncomputable def diagonal_distance : ℝ := 9.3
noncomputable def right_turn_distance : ℝ := 3

theorem average_distance_to_sides :
  let d1 := 9.58
  let d2 := 6.58
  let d3 := 5.42
  let d4 := 8.42
  (d1 + d2 + d3 + d4) / 4 = 7.5 :=
by
  let d1 := 9.58
  let d2 := 6.58
  let d3 := 5.42
  let d4 := 8.42
  have h : (d1 + d2 + d3 + d4) / 4 = 7.5
  { sorry }
  exact h

end average_distance_to_sides_l159_159996

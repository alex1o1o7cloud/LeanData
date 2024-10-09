import Mathlib

namespace find_some_number_l1039_103973

theorem find_some_number :
  ∃ some_number : ℝ, (3.242 * 10 / some_number) = 0.032420000000000004 ∧ some_number = 1000 :=
by
  sorry

end find_some_number_l1039_103973


namespace adapted_bowling_ball_volume_l1039_103940

noncomputable def volume_adapted_bowling_ball : ℝ :=
  let volume_sphere := (4/3) * Real.pi * (20 ^ 3)
  let volume_hole1 := Real.pi * (1 ^ 2) * 10
  let volume_hole2 := Real.pi * (1.5 ^ 2) * 10
  let volume_hole3 := Real.pi * (2 ^ 2) * 10
  volume_sphere - (volume_hole1 + volume_hole2 + volume_hole3)

theorem adapted_bowling_ball_volume :
  volume_adapted_bowling_ball = 10594.17 * Real.pi :=
sorry

end adapted_bowling_ball_volume_l1039_103940


namespace card_sorting_moves_upper_bound_l1039_103998

theorem card_sorting_moves_upper_bound (n : ℕ) (cells : Fin (n+1) → Fin (n+1)) (cards : Fin (n+1) → Fin (n+1)) : 
  (∃ (moves : (Fin (n+1) × Fin (n+1)) → ℕ),
    (∀ (i : Fin (n+1)), moves (i, cards i) ≤ 2 * n - 1) ∧ 
    (cards 0 = 0 → moves (0, 0) = 2 * n - 1) ∧ 
    (∃! start_pos : Fin (n+1) → Fin (n+1), 
      moves (start_pos (n), start_pos (0)) = 2 * n - 1)) := sorry

end card_sorting_moves_upper_bound_l1039_103998


namespace monochromatic_rectangle_l1039_103934

theorem monochromatic_rectangle (n : ℕ) (coloring : ℕ × ℕ → Fin n) :
  ∃ (a b c d : ℕ × ℕ), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧
  (coloring a = coloring b) ∧ (coloring b = coloring c) ∧ (coloring c = coloring d) :=
sorry

end monochromatic_rectangle_l1039_103934


namespace cyclists_meeting_l1039_103926

-- Define the velocities of the cyclists and the time variable
variables (v₁ v₂ t : ℝ)

-- Define the conditions for the problem
def condition1 : Prop := v₁ * t = v₂ * (2/3)
def condition2 : Prop := v₂ * t = v₁ * 1.5

-- Define the main theorem to be proven
theorem cyclists_meeting (h1 : condition1 v₁ v₂ t) (h2 : condition2 v₁ v₂ t) :
  t = 1 ∧ (v₁ / v₂ = 3 / 2) :=
by sorry

end cyclists_meeting_l1039_103926


namespace candy_bar_cost_l1039_103909

variable (C : ℕ)

theorem candy_bar_cost
  (soft_drink_cost : ℕ)
  (num_candy_bars : ℕ)
  (total_spent : ℕ)
  (h1 : soft_drink_cost = 2)
  (h2 : num_candy_bars = 5)
  (h3 : total_spent = 27) :
  num_candy_bars * C + soft_drink_cost = total_spent → C = 5 := by
  sorry

end candy_bar_cost_l1039_103909


namespace verify_parabola_D_l1039_103979

def vertex_form (a h k : ℝ) (x : ℝ) : ℝ := a * (x - h)^2 + k

def parabola_vertex (y : ℝ → ℝ) (h k : ℝ) : Prop :=
  ∀ x, y x = vertex_form (-1) h k x

-- Given conditions
def h : ℝ := 2
def k : ℝ := 3

-- Possible expressions
def parabola_A (x : ℝ) : ℝ := -((x + 2)^2) - 3
def parabola_B (x : ℝ) : ℝ := -((x - 2)^2) - 3
def parabola_C (x : ℝ) : ℝ := -((x + 2)^2) + 3
def parabola_D (x : ℝ) : ℝ := -((x - 2)^2) + 3

theorem verify_parabola_D : parabola_vertex parabola_D 2 3 :=
by
  -- Placeholder for the proof
  sorry

end verify_parabola_D_l1039_103979


namespace igor_min_score_needed_l1039_103948

theorem igor_min_score_needed
  (scores : List ℕ)
  (goal : ℚ)
  (next_test_score : ℕ)
  (h_scores : scores = [88, 92, 75, 83, 90])
  (h_goal : goal = 87)
  (h_solution : next_test_score = 94)
  : 
  let current_sum := scores.sum
  let current_tests := scores.length
  let required_total := (goal * (current_tests + 1))
  let next_test_needed := required_total - current_sum
  next_test_needed ≤ next_test_score := 
by 
  sorry

end igor_min_score_needed_l1039_103948


namespace largest_angle_of_triangle_l1039_103931

theorem largest_angle_of_triangle (x : ℝ) (h : x + 3 * x + 5 * x = 180) : 5 * x = 100 :=
sorry

end largest_angle_of_triangle_l1039_103931


namespace range_of_g_l1039_103922

noncomputable def g (x : ℝ) : ℝ := 
  (Real.arccos (x / 3))^2 + (Real.pi / 4) * (Real.arcsin (x / 3)) 
    - (Real.arcsin (x / 3))^2 + (Real.pi^2 / 16) * (x^2 + 2 * x + 3)

theorem range_of_g : 
  ∀ x : ℝ, -3 ≤ x ∧ x ≤ 3 → 
  ∃ y, y = g x ∧ y ∈ (Set.Icc (Real.pi^2 / 4) (15 * Real.pi^2 / 16 + Real.pi / 4 * Real.arcsin 1)) :=
by
  sorry

end range_of_g_l1039_103922


namespace people_stools_chairs_l1039_103996

def total_legs (x y z : ℕ) : ℕ := 2 * x + 3 * y + 4 * z 

theorem people_stools_chairs (x y z : ℕ) : 
  (x > y) → (x > z) → (x < y + z) → (total_legs x y z = 32) → 
  (x = 5 ∧ y = 2 ∧ z = 4) :=
by
  intro h1 h2 h3 h4
  sorry

end people_stools_chairs_l1039_103996


namespace necessarily_negative_sum_l1039_103932

theorem necessarily_negative_sum 
  (u v w : ℝ)
  (hu : -1 < u ∧ u < 0)
  (hv : 0 < v ∧ v < 1)
  (hw : -2 < w ∧ w < -1) :
  v + w < 0 :=
sorry

end necessarily_negative_sum_l1039_103932


namespace minimum_value_f_l1039_103967

noncomputable def f (x : ℝ) : ℝ := (x^2 / 8) + x * (Real.cos x) + (Real.cos (2 * x))

theorem minimum_value_f : ∃ x : ℝ, f x = -1 :=
by {
  sorry
}

end minimum_value_f_l1039_103967


namespace abc_sum_l1039_103913

theorem abc_sum
  (a b c : ℤ)
  (h1 : ∀ x : ℤ, x^2 + 17 * x + 70 = (x + a) * (x + b))
  (h2 : ∀ x : ℤ, x^2 - 19 * x + 84 = (x - b) * (x - c)) :
  a + b + c = 29 := by
  sorry

end abc_sum_l1039_103913


namespace multimedia_sets_max_profit_l1039_103904

-- Definitions of conditions:
def cost_A : ℝ := 3
def cost_B : ℝ := 2.4
def price_A : ℝ := 3.3
def price_B : ℝ := 2.8
def total_sets : ℕ := 50
def total_cost : ℝ := 132
def min_m : ℕ := 11

-- Problem 1: Prove the number of sets based on equations
theorem multimedia_sets (x y : ℕ) (h1 : x + y = total_sets) (h2 : cost_A * x + cost_B * y = total_cost) :
  x = 20 ∧ y = 30 :=
by sorry

-- Problem 2: Prove the maximum profit within a given range
theorem max_profit (m : ℕ) (h_m : 10 < m ∧ m < 20) :
  (-(0.1 : ℝ) * m + 20 = 18.9) ↔ m = min_m :=
by sorry

end multimedia_sets_max_profit_l1039_103904


namespace time_to_decorate_l1039_103976

variable (mia_rate billy_rate total_eggs : ℕ)

theorem time_to_decorate (h_mia : mia_rate = 24) (h_billy : billy_rate = 10) (h_total : total_eggs = 170) :
  total_eggs / (mia_rate + billy_rate) = 5 :=
by
  sorry

end time_to_decorate_l1039_103976


namespace scarlett_initial_oil_amount_l1039_103999

theorem scarlett_initial_oil_amount (x : ℝ) (h : x + 0.67 = 0.84) : x = 0.17 :=
by sorry

end scarlett_initial_oil_amount_l1039_103999


namespace estimated_germination_probability_l1039_103982

-- This definition represents the conditions of the problem in Lean.
def germination_data : List (ℕ × ℕ × Real) :=
  [(2, 2, 1.000), (5, 4, 0.800), (10, 9, 0.900), (50, 44, 0.880), (100, 92, 0.920),
   (500, 463, 0.926), (1000, 928, 0.928), (1500, 1396, 0.931), (2000, 1866, 0.933), (3000, 2794, 0.931)]

-- The theorem states that the germination probability is approximately 0.93.
theorem estimated_germination_probability (data : List (ℕ × ℕ × Real)) (h : data = germination_data) :
  ∃ p : Real, p = 0.93 ∧ ∀ n m r, (n, m, r) ∈ data → |r - p| < 0.01 :=
by
  -- Placeholder for proof
  sorry

end estimated_germination_probability_l1039_103982


namespace value_of_expression_l1039_103950

theorem value_of_expression (a : ℝ) (h : a^2 - 2 * a = 1) : 3 * a^2 - 6 * a - 4 = -1 :=
by
  sorry

end value_of_expression_l1039_103950


namespace loss_percentage_on_book_sold_at_loss_l1039_103957

theorem loss_percentage_on_book_sold_at_loss :
  ∀ (total_cost cost1 : ℝ) (gain_percent : ℝ),
    total_cost = 420 → cost1 = 245 → gain_percent = 0.19 →
    (∀ (cost2 SP : ℝ), cost2 = total_cost - cost1 →
                       SP = cost2 * (1 + gain_percent) →
                       SP = 208.25 →
                       ((cost1 - SP) / cost1 * 100) = 15) :=
by
  intros total_cost cost1 gain_percent h_total_cost h_cost1 h_gain_percent cost2 SP h_cost2 h_SP h_SP_value
  sorry

end loss_percentage_on_book_sold_at_loss_l1039_103957


namespace number_of_triangles_l1039_103953

theorem number_of_triangles (points_AB points_BC points_AC : ℕ)
                            (hAB : points_AB = 12)
                            (hBC : points_BC = 9)
                            (hAC : points_AC = 10) :
    let total_points := points_AB + points_BC + points_AC
    let total_combinations := Nat.choose total_points 3
    let degenerate_AB := Nat.choose points_AB 3
    let degenerate_BC := Nat.choose points_BC 3
    let degenerate_AC := Nat.choose points_AC 3
    let valid_triangles := total_combinations - (degenerate_AB + degenerate_BC + degenerate_AC)
    valid_triangles = 4071 :=
by
  sorry

end number_of_triangles_l1039_103953


namespace middle_term_arithmetic_sequence_l1039_103933

theorem middle_term_arithmetic_sequence (m : ℝ) (h : 2 * m = 1 + 5) : m = 3 :=
by
  sorry

end middle_term_arithmetic_sequence_l1039_103933


namespace average_value_correct_l1039_103916

noncomputable def average_value (k z : ℝ) : ℝ :=
  (k + 2 * k * z + 4 * k * z + 8 * k * z + 16 * k * z) / 5

theorem average_value_correct (k z : ℝ) :
  average_value k z = (k * (1 + 30 * z)) / 5 := by
  sorry

end average_value_correct_l1039_103916


namespace geometric_sequence_sum_l1039_103944

noncomputable def geometric_sequence (a₁ q : ℝ) (n : ℕ) : ℝ := a₁ * q^(n-1)

noncomputable def sum_geometric_sequence (a₁ q : ℝ) (n : ℕ) := a₁ * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum 
  (a₁ : ℝ) (q : ℝ) 
  (h_q : q = 1 / 2) 
  (h_a₂ : geometric_sequence a₁ q 2 = 2) : 
  sum_geometric_sequence a₁ q 6 = 63 / 8 :=
by
  -- The proof is skipped here
  sorry

end geometric_sequence_sum_l1039_103944


namespace x_square_minus_5x_is_necessary_not_sufficient_l1039_103924

theorem x_square_minus_5x_is_necessary_not_sufficient (x : ℝ) :
  (x^2 - 5 * x < 0) → (|x - 1| < 1) → (x^2 - 5 * x < 0 ∧ ∃ y : ℝ, (0 < y ∧ y < 2) → x = y) :=
by
  sorry

end x_square_minus_5x_is_necessary_not_sufficient_l1039_103924


namespace tan_double_angle_l1039_103980

theorem tan_double_angle (α : ℝ) (h1 : α > 0) (h2 : α < Real.pi)
  (h3 : Real.cos α + Real.sin α = -1 / 5) : Real.tan (2 * α) = -24 / 7 :=
by
  sorry

end tan_double_angle_l1039_103980


namespace sector_area_l1039_103991

theorem sector_area (r : ℝ) (θ : ℝ) (h_r : r = 12) (h_θ : θ = 40) : (θ / 360) * π * r^2 = 16 * π :=
by
  rw [h_r, h_θ]
  sorry

end sector_area_l1039_103991


namespace minimum_function_value_l1039_103902

theorem minimum_function_value :
  ∃ (x y : ℕ), 0 ≤ x ∧ x ≤ 2 ∧ 0 ≤ y ∧ y ≤ 3 ∧
  (∀ x' y', 0 ≤ x' ∧ x' ≤ 2 → 0 ≤ y' ∧ y' ≤ 3 →
  (x^2 * y^2 : ℝ) / ((x^2 + y^2)^2 : ℝ) ≤ (x'^2 * y'^2 : ℝ) / ((x'^2 + y'^2)^2 : ℝ)) ∧
  (x = 0 ∨ y = 0) ∧ ((x^2 * y^2 : ℝ) / ((x^2 + y^2)^2 : ℝ) = 0) :=
by
  --; Implementation of the theorem would follow
  sorry

end minimum_function_value_l1039_103902


namespace incorrect_observation_l1039_103939

theorem incorrect_observation (n : ℕ) (mean_original mean_corrected correct_obs incorrect_obs : ℝ)
  (h1 : n = 40) 
  (h2 : mean_original = 36) 
  (h3 : mean_corrected = 36.45) 
  (h4 : correct_obs = 34) 
  (h5 : n * mean_original = 1440) 
  (h6 : n * mean_corrected = 1458) 
  (h_diff : 1458 - 1440 = 18) :
  incorrect_obs = 52 :=
by
  sorry

end incorrect_observation_l1039_103939


namespace bear_meat_needs_l1039_103910

theorem bear_meat_needs (B_total : ℕ) (cubs : ℕ) (w_cub : ℚ) 
  (h1 : B_total = 210)
  (h2 : cubs = 4)
  (h3 : w_cub = B_total / cubs) : 
  w_cub = 52.5 :=
by 
  sorry

end bear_meat_needs_l1039_103910


namespace y1_greater_than_y2_l1039_103984

-- Definitions of the conditions.
def point1_lies_on_line (y₁ b : ℝ) : Prop := y₁ = -3 * (-2 : ℝ) + b
def point2_lies_on_line (y₂ b : ℝ) : Prop := y₂ = -3 * (-1 : ℝ) + b

-- The theorem to prove: y₁ > y₂ given the conditions.
theorem y1_greater_than_y2 (y₁ y₂ b : ℝ) (h1 : point1_lies_on_line y₁ b) (h2 : point2_lies_on_line y₂ b) : y₁ > y₂ :=
by {
  sorry
}

end y1_greater_than_y2_l1039_103984


namespace constant_in_denominator_l1039_103970

theorem constant_in_denominator (x y z : ℝ) (some_constant : ℝ)
  (h : ((x - y)^3 + (y - z)^3 + (z - x)^3) / (some_constant * (x - y) * (y - z) * (z - x)) = 0.2) :
  some_constant = 15 := 
sorry

end constant_in_denominator_l1039_103970


namespace problem1_problem2_l1039_103943

-- Problem (1)
theorem problem1 (α : ℝ) (h1 : Real.tan α = 2) : 
  (3 * Real.sin α + 2 * Real.cos α) / (Real.sin α - Real.cos α) = 8 := 
by 
  sorry

-- Problem (2)
theorem problem2 (α : ℝ) (h1 : 0 < α) (h2 : α < Real.pi) (h3 : Real.sin α + Real.cos α = 1 / 5) : 
  Real.tan α = -4 / 3 := 
by
  sorry

end problem1_problem2_l1039_103943


namespace squares_to_nine_l1039_103956

theorem squares_to_nine (x : ℤ) : x^2 = 9 ↔ x = 3 ∨ x = -3 :=
sorry

end squares_to_nine_l1039_103956


namespace simple_interest_rate_l1039_103918

theorem simple_interest_rate (P R: ℝ) (T : ℝ) (hT : T = 8) (h : 2 * P = P + (P * R * T) / 100) : R = 12.5 :=
by
  -- Placeholder for proof steps
  sorry

end simple_interest_rate_l1039_103918


namespace part1_part2_l1039_103983

theorem part1 (a : ℝ) (h1 : ∀ x y, y = a * x + 1 → 3 * x^2 - y^2 = 1) (h2 : ∃ x1 y1 x2 y2 : ℝ, y1 = a * x1 + 1 ∧ y2 = a * x2 + 1 ∧ 3 * x1 * x1 - y1 * y1 = 1 ∧ 3 * x2 * x2 - y2 * y2 = 1 ∧ x1 * x2 + (a * x1 + 1) * (a * x2 + 1) = 0) : a = 1 ∨ a = -1 := sorry

theorem part2 (h : ∀ x y, y = a * x + 1 → 3 * x^2 - y^2 = 1) (a : ℝ) (h2 : ∃ x1 y1 x2 y2 : ℝ, y1 = a * x1 + 1 ∧ y2 = a * x2 + 1 ∧ 3 * x1 * x1 - y1 * y1 = 1 ∧ 3 * x2 * x2 - y2 * y2 = 1 ∧ (y1 + y2) / 2 = (1 / 2) * (x1 + x2) / 2 ∧ (y1 - y2) / (x1 - x2) = -2) : false := sorry

end part1_part2_l1039_103983


namespace sin_45_eq_1_div_sqrt_2_l1039_103966

theorem sin_45_eq_1_div_sqrt_2 : Real.sin (π / 4) = 1 / Real.sqrt 2 := 
by
  sorry

end sin_45_eq_1_div_sqrt_2_l1039_103966


namespace minimum_distance_PQ_l1039_103929

open Real

noncomputable def minimum_distance (t : ℝ) : ℝ := 
  (|t - 1|) / (sqrt (1 + t ^ 2))

theorem minimum_distance_PQ :
  let t := sqrt 2 / 2
  let x_P := 2
  let y_P := 0
  let x_Q := -1 + t
  let y_Q := 2 + t
  let d := minimum_distance (x_Q - y_Q + 3)
  (d - 2) = (5 * sqrt 2) / 2 - 2 :=
sorry

end minimum_distance_PQ_l1039_103929


namespace anna_lemonade_difference_l1039_103958

variables (x y p s : ℝ)

theorem anna_lemonade_difference (h : x * p = 1.5 * (y * s)) : (x * p) - (y * s) = 0.5 * (y * s) :=
by
  -- Insert proof here
  sorry

end anna_lemonade_difference_l1039_103958


namespace larger_number_is_26_l1039_103987

theorem larger_number_is_26 {x y : ℤ} 
  (h1 : x + y = 45) 
  (h2 : x - y = 7) : 
  max x y = 26 :=
by
  sorry

end larger_number_is_26_l1039_103987


namespace tangerine_boxes_l1039_103941

theorem tangerine_boxes
  (num_boxes_apples : ℕ)
  (apples_per_box : ℕ)
  (num_boxes_tangerines : ℕ)
  (tangerines_per_box : ℕ)
  (total_fruits : ℕ)
  (h1 : num_boxes_apples = 19)
  (h2 : apples_per_box = 46)
  (h3 : tangerines_per_box = 170)
  (h4 : total_fruits = 1894)
  : num_boxes_tangerines = 6 := 
  sorry

end tangerine_boxes_l1039_103941


namespace ratio_sheep_horses_l1039_103986

theorem ratio_sheep_horses (amount_food_per_horse : ℕ) (total_food_per_day : ℕ) (num_sheep : ℕ) (num_horses : ℕ) :
  amount_food_per_horse = 230 ∧ total_food_per_day = 12880 ∧ num_sheep = 24 ∧ num_horses = total_food_per_day / amount_food_per_horse →
  num_sheep / num_horses = 3 / 7 :=
by
  sorry

end ratio_sheep_horses_l1039_103986


namespace sum_of_inverses_A_B_C_eq_300_l1039_103949

theorem sum_of_inverses_A_B_C_eq_300 
  (p q r : ℝ)
  (hroots : ∀ x, (x^3 - 30*x^2 + 105*x - 114 = 0) → (x = p ∨ x = q ∨ x = r))
  (A B C : ℝ)
  (hdecomp : ∀ s, s ≠ p ∧ s ≠ q ∧ s ≠ r →
    (1 / (s^3 - 30*s^2 + 105*s - 114) = A/(s - p) + B/(s - q) + C/(s - r))) :
  (1 / A) + (1 / B) + (1 / C) = 300 :=
sorry

end sum_of_inverses_A_B_C_eq_300_l1039_103949


namespace trapezoid_midsegment_l1039_103903

-- Define the problem conditions and question
theorem trapezoid_midsegment (b h x : ℝ) (h_nonzero : h ≠ 0) (hx : x = b + 75)
  (equal_areas : (1 / 2) * (h / 2) * (b + (b + 75)) = (1 / 2) * (h / 2) * ((b + 75) + (b + 150))) :
  ∃ n : ℤ, n = ⌊x^2 / 120⌋ ∧ n = 3000 := 
by 
  sorry

end trapezoid_midsegment_l1039_103903


namespace bob_salary_is_14400_l1039_103912

variables (mario_salary_current : ℝ) (mario_salary_last_year : ℝ) (bob_salary_last_year : ℝ) (bob_salary_current : ℝ)

-- Given Conditions
axiom mario_salary_increase : mario_salary_current = 4000
axiom mario_salary_equation : 1.40 * mario_salary_last_year = mario_salary_current
axiom bob_salary_last_year_equation : bob_salary_last_year = 3 * mario_salary_current
axiom bob_salary_increase : bob_salary_current = bob_salary_last_year + 0.20 * bob_salary_last_year

-- Theorem to prove
theorem bob_salary_is_14400 
    (mario_salary_last_year_eq : mario_salary_last_year = 4000 / 1.40)
    (bob_salary_last_year_eq : bob_salary_last_year = 3 * 4000)
    (bob_salary_current_eq : bob_salary_current = 12000 + 0.20 * 12000) :
    bob_salary_current = 14400 := 
by
  sorry

end bob_salary_is_14400_l1039_103912


namespace angle_of_inclination_of_line_l1039_103901

theorem angle_of_inclination_of_line (θ : ℝ) (m : ℝ) (h : |m| = 1) :
  θ = 45 ∨ θ = 135 :=
sorry

end angle_of_inclination_of_line_l1039_103901


namespace reflection_y_axis_correct_l1039_103964

-- Define the coordinates and reflection across the y-axis
def reflect_y_axis (p : (ℝ × ℝ)) : (ℝ × ℝ) :=
  (-p.1, p.2)

-- Define the original point M
def M : (ℝ × ℝ) := (3, 2)

-- State the theorem we want to prove
theorem reflection_y_axis_correct : reflect_y_axis M = (-3, 2) :=
by
  -- The proof would go here, but it is omitted as per the instructions
  sorry

end reflection_y_axis_correct_l1039_103964


namespace series_sum_l1039_103947

variable {c d : ℝ}

theorem series_sum (h : ∑' n : ℕ, c / d ^ ((3 : ℝ) ^ n) = 9) :
  ∑' n : ℕ, c / (c + 2 * d) ^ (n + 1) = 9 / 11 :=
by
  -- The code that follows will include the steps and proof to reach the conclusion
  sorry

end series_sum_l1039_103947


namespace expression_is_odd_l1039_103960

-- Define positive integers
def is_positive (n : ℕ) := n > 0

-- Define odd integer
def is_odd (n : ℕ) := n % 2 = 1

-- Define multiple of 3
def is_multiple_of_3 (n : ℕ) := ∃ k : ℕ, n = 3 * k

-- The Lean 4 statement to prove the problem
theorem expression_is_odd (a b c : ℕ)
  (ha : is_positive a) (hb : is_positive b) (hc : is_positive c)
  (h_odd_a : is_odd a) (h_odd_b : is_odd b) (h_mult_3_c : is_multiple_of_3 c) :
  is_odd (5^a + (b-1)^2 * c) :=
by
  sorry

end expression_is_odd_l1039_103960


namespace largest_real_root_range_l1039_103977

theorem largest_real_root_range (b0 b1 b2 b3 : ℝ) (h0 : |b0| ≤ 1) (h1 : |b1| ≤ 1) (h2 : |b2| ≤ 1) (h3 : |b3| ≤ 1) :
  ∀ r : ℝ, (Polynomial.eval r (Polynomial.C (1:ℝ) + Polynomial.C b3 * Polynomial.X^3 + Polynomial.C b2 * Polynomial.X^2 + Polynomial.C b1 * Polynomial.X + Polynomial.C b0) = 0) → (5 / 2) < r ∧ r < 3 :=
by
  sorry

end largest_real_root_range_l1039_103977


namespace striped_to_total_ratio_l1039_103989

theorem striped_to_total_ratio (total_students shorts_checkered_diff striped_shorts_diff : ℕ)
    (h_total : total_students = 81)
    (h_shorts_checkered : ∃ checkered, shorts_checkered_diff = checkered + 19)
    (h_striped_shorts : ∃ shorts, striped_shorts_diff = shorts + 8) :
    (striped_shorts_diff : ℚ) / total_students = 2 / 3 :=
by sorry

end striped_to_total_ratio_l1039_103989


namespace divide_triangle_in_half_l1039_103975

def triangle_vertices : Prop :=
  let A := (0, 2)
  let B := (0, 0)
  let C := (10, 0)
  let base := 10
  let height := 2
  let total_area := (1 / 2) * base * height

  ∀ (a : ℝ),
  (1 / 2) * a * height = total_area / 2 → a = 5

theorem divide_triangle_in_half : triangle_vertices := 
  sorry

end divide_triangle_in_half_l1039_103975


namespace remainder_of_p_div_10_is_6_l1039_103914

-- Define the problem
def a : ℕ := sorry -- a is a positive integer and a multiple of 2

-- Define p based on a
def p : ℕ := 4^a

-- The main goal is to prove the remainder when p is divided by 10 is 6
theorem remainder_of_p_div_10_is_6 (ha : a > 0 ∧ a % 2 = 0) : p % 10 = 6 := by
  sorry

end remainder_of_p_div_10_is_6_l1039_103914


namespace units_digit_35_pow_35_mul_17_pow_17_l1039_103928

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_35_pow_35_mul_17_pow_17:
  units_digit (35 ^ (35 * 17 ^ 17)) = 5 := 
by {
  -- Here we're skipping the proof.
  sorry
}

end units_digit_35_pow_35_mul_17_pow_17_l1039_103928


namespace solve_system_of_equations_in_nat_numbers_l1039_103981

theorem solve_system_of_equations_in_nat_numbers :
  ∃ a b c d : ℕ, a * b = c + d ∧ c * d = a + b ∧ a = 2 ∧ b = 2 ∧ c = 2 ∧ d = 2 :=
by
  sorry

end solve_system_of_equations_in_nat_numbers_l1039_103981


namespace sum_of_two_numbers_l1039_103907

theorem sum_of_two_numbers :
  (∃ x y : ℕ, y = 2 * x - 43 ∧ y = 31 ∧ x + y = 68) :=
sorry

end sum_of_two_numbers_l1039_103907


namespace exists_triangle_perimeter_lt_1cm_circumradius_gt_1km_l1039_103990

noncomputable def perimeter (a b c : ℝ) : ℝ := a + b + c

noncomputable def circumradius (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  (a * b * c) / (4 * Real.sqrt (s * (s - a) * (s - b) * (s - c)))

theorem exists_triangle_perimeter_lt_1cm_circumradius_gt_1km :
  ∃ (A B C : ℝ) (a b c : ℝ), a + b + c < 0.01 ∧ circumradius a b c > 1000 :=
by
  sorry

end exists_triangle_perimeter_lt_1cm_circumradius_gt_1km_l1039_103990


namespace required_C6H6_for_C6H5CH3_and_H2_l1039_103905

-- Define the necessary molecular structures and stoichiometry
def C6H6 : Type := ℕ -- Benzene
def CH4 : Type := ℕ -- Methane
def C6H5CH3 : Type := ℕ -- Toluene
def H2 : Type := ℕ -- Hydrogen

-- Balanced equation condition
def balanced_reaction (x : C6H6) (y : CH4) (z : C6H5CH3) (w : H2) : Prop :=
  x = y ∧ x = z ∧ x = w

-- Given conditions
def condition (m : ℕ) : Prop :=
  balanced_reaction m m m m

theorem required_C6H6_for_C6H5CH3_and_H2 :
  ∀ (n : ℕ), condition n → n = 3 → n = 3 :=
by
  intros n h hn
  exact hn

end required_C6H6_for_C6H5CH3_and_H2_l1039_103905


namespace year_2022_form_l1039_103915

theorem year_2022_form :
  ∃ (a b c d e f g h i j : ℕ),
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧ a ≠ j ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧ b ≠ j ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧ c ≠ j ∧
    d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧ d ≠ j ∧
    e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧ e ≠ j ∧
    f ≠ g ∧ f ≠ h ∧ f ≠ i ∧ f ≠ j ∧
    g ≠ h ∧ g ≠ i ∧ g ≠ j ∧
    h ≠ i ∧ h ≠ j ∧
    i ≠ j ∧
    a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧ f < 10 ∧ g < 10 ∧ h < 10 ∧ i < 10 ∧ j < 10 ∧
    2001 ≤ (a + b * c * d * e) / (f + g * h * i * j) ∧ (a + b * c * d * e) / (f + g * h * i * j) ≤ 2100 ∧
    (a + b * c * d * e) / (f + g * h * i * j) = 2022 :=
sorry

end year_2022_form_l1039_103915


namespace real_solutions_iff_a_geq_3_4_l1039_103908

theorem real_solutions_iff_a_geq_3_4:
  (∃ (x y : ℝ), x + y^2 = a ∧ y + x^2 = a) ↔ a ≥ 3 / 4 := sorry

end real_solutions_iff_a_geq_3_4_l1039_103908


namespace product_of_N1_N2_l1039_103925

theorem product_of_N1_N2 :
  (∃ (N1 N2 : ℤ),
    (∀ (x : ℚ),
      (47 * x - 35) * (x - 1) * (x - 2) = N1 * (x - 2) * (x - 1) + N2 * (x - 1) * (x - 2)) ∧
    N1 * N2 = -708) :=
sorry

end product_of_N1_N2_l1039_103925


namespace carpet_dimensions_l1039_103955

theorem carpet_dimensions
  (x y q : ℕ)
  (h_dim : y = 2 * x)
  (h_room1 : ((q^2 + 50^2) = (q * 2 - 50)^2 + (50 * 2 - q)^2))
  (h_room2 : ((q^2 + 38^2) = (q * 2 - 38)^2 + (38 * 2 - q)^2)) :
  x = 25 ∧ y = 50 :=
sorry

end carpet_dimensions_l1039_103955


namespace sum_integer_solutions_correct_l1039_103959

noncomputable def sum_of_integer_solutions (m : ℝ) : ℝ :=
  if (3 ≤ m ∧ m < 6) ∨ (-6 ≤ m ∧ m < -3) then -9 else 0

theorem sum_integer_solutions_correct (m : ℝ) :
  (∀ x : ℝ, (3 * x + m < 0 ∧ x > -5) → (∃ s : ℝ, s = sum_of_integer_solutions m ∧ s = -9)) :=
by
  sorry

end sum_integer_solutions_correct_l1039_103959


namespace cube_vertices_count_l1039_103968

-- Defining the conditions of the problem
def num_faces : ℕ := 6
def num_edges : ℕ := 12
def euler_formula (V E F : ℕ) : Prop := V - E + F = 2

-- Stating the proof problem
theorem cube_vertices_count : ∃ V : ℕ, euler_formula V num_edges num_faces ∧ V = 8 :=
by
  sorry

end cube_vertices_count_l1039_103968


namespace total_frogs_seen_by_hunter_l1039_103969

/-- Hunter saw 5 frogs sitting on lily pads in the pond. -/
def initial_frogs : ℕ := 5

/-- Three more frogs climbed out of the water onto logs floating in the pond. -/
def frogs_on_logs : ℕ := 3

/-- Two dozen baby frogs (24 frogs) hopped onto a big rock jutting out from the pond. -/
def baby_frogs : ℕ := 24

/--
The total number of frogs Hunter saw in the pond.
-/
theorem total_frogs_seen_by_hunter : initial_frogs + frogs_on_logs + baby_frogs = 32 := by
sorry

end total_frogs_seen_by_hunter_l1039_103969


namespace gretchen_total_earnings_l1039_103993

-- Define the conditions
def price_per_drawing : ℝ := 20.0
def caricatures_sold_saturday : ℕ := 24
def caricatures_sold_sunday : ℕ := 16

-- The total caricatures sold
def total_caricatures_sold : ℕ := caricatures_sold_saturday + caricatures_sold_sunday

-- The total amount of money made
def total_money_made : ℝ := total_caricatures_sold * price_per_drawing

-- The theorem to be proven
theorem gretchen_total_earnings : total_money_made = 800.0 := by
  sorry

end gretchen_total_earnings_l1039_103993


namespace johnsonville_max_band_members_l1039_103900

def max_band_members :=
  ∃ m : ℤ, 30 * m % 34 = 2 ∧ 30 * m < 1500 ∧
  ∀ n : ℤ, (30 * n % 34 = 2 ∧ 30 * n < 1500) → 30 * n ≤ 30 * m

theorem johnsonville_max_band_members : ∃ m : ℤ, 30 * m % 34 = 2 ∧ 30 * m < 1500 ∧
                                           30 * m = 1260 :=
by 
  sorry

end johnsonville_max_band_members_l1039_103900


namespace science_book_multiple_l1039_103946

theorem science_book_multiple (history_pages novel_pages science_pages : ℕ)
  (H1 : history_pages = 300)
  (H2 : novel_pages = history_pages / 2)
  (H3 : science_pages = 600) :
  science_pages / novel_pages = 4 := 
by
  -- Proof will be filled out here
  sorry

end science_book_multiple_l1039_103946


namespace area_of_rectangle_l1039_103978

theorem area_of_rectangle (P : ℝ) (w : ℝ) (h : ℝ) (A : ℝ) 
  (hP : P = 28) 
  (hw : w = 6) 
  (hP_formula : P = 2 * (h + w)) 
  (hA_formula : A = h * w) : 
  A = 48 :=
by
  sorry

end area_of_rectangle_l1039_103978


namespace min_value_expr_l1039_103962

open Real

theorem min_value_expr (θ : ℝ) (h1 : 0 < θ) (h2 : θ < π / 2) :
  3 * cos θ + 2 / sin θ + 2 * sqrt 2 * tan θ ≥ (3 : ℝ) * (12 * sqrt 2)^((1 : ℝ) / (3 : ℝ)) := sorry

end min_value_expr_l1039_103962


namespace weekly_allowance_l1039_103995

theorem weekly_allowance (A : ℝ) (H1 : A - (3/5) * A = (2/5) * A)
(H2 : (2/5) * A - (1/3) * ((2/5) * A) = (4/15) * A)
(H3 : (4/15) * A = 0.96) : A = 3.6 := 
sorry

end weekly_allowance_l1039_103995


namespace set_intersection_l1039_103942

noncomputable def U : Set ℕ := {1, 2, 3, 4, 5, 6}
noncomputable def A : Set ℕ := {1, 2, 5}
noncomputable def B : Set ℕ := {x ∈ U | (3 / (2 - x) + 1 ≤ 0)}
noncomputable def C_U_B : Set ℕ := U \ B

theorem set_intersection : A ∩ C_U_B = {1, 2} :=
by {
  sorry
}

end set_intersection_l1039_103942


namespace catch_up_time_l1039_103930

noncomputable def speed_ratios (v : ℝ) : Prop :=
  let a_speed := (4 / 5) * v
  let b_speed := (2 / 5) * v
  a_speed = 2 * b_speed

theorem catch_up_time (v t : ℝ) (a_speed b_speed : ℝ)
  (h1 : a_speed = (4 / 5) * v)
  (h2 : b_speed = (2 / 5) * v)
  (h3 : a_speed = 2 * b_speed) :
  (t = 11) := by
  sorry

end catch_up_time_l1039_103930


namespace find_C_coordinates_l1039_103972

-- Define the points A, B, and the vector relationship
def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (-1, 5)
def C : ℝ × ℝ := (-3, 9)

-- The condition stating vector AC is twice vector AB
def vector_condition (A B C : ℝ × ℝ) : Prop :=
  (C.1 - A.1, C.2 - A.2) = (2 * (B.1 - A.1), 2 * (B.2 - A.2))

-- The theorem we need to prove
theorem find_C_coordinates (A B C : ℝ × ℝ) (hA : A = (1, 1)) (hB : B = (-1, 5))
  (hCondition : vector_condition A B C) : C = (-3, 9) :=
by
  rw [hA, hB] at hCondition
  -- sorry here skips the proof
  sorry

end find_C_coordinates_l1039_103972


namespace sin_30_eq_one_half_cos_11pi_over_4_eq_neg_sqrt2_over_2_l1039_103971

theorem sin_30_eq_one_half : Real.sin (30 * Real.pi / 180) = 1 / 2 :=
by 
  -- This is the statement only, the proof will be here
  sorry

theorem cos_11pi_over_4_eq_neg_sqrt2_over_2 : Real.cos (11 * Real.pi / 4) = - Real.sqrt 2 / 2 :=
by 
  -- This is the statement only, the proof will be here
  sorry

end sin_30_eq_one_half_cos_11pi_over_4_eq_neg_sqrt2_over_2_l1039_103971


namespace proof_2_in_M_l1039_103992

def U : Set ℕ := {1, 2, 3, 4, 5}

def M : Set ℕ := { x | x ∈ U ∧ x ≠ 1 ∧ x ≠ 3 }

theorem proof_2_in_M : 2 ∈ M :=
by
  sorry

end proof_2_in_M_l1039_103992


namespace tan_2beta_l1039_103961

theorem tan_2beta {α β : ℝ} 
  (h₁ : Real.tan (α + β) = 2) 
  (h₂ : Real.tan (α - β) = 3) : 
  Real.tan (2 * β) = -1 / 7 :=
by 
  sorry

end tan_2beta_l1039_103961


namespace exist_coprime_integers_l1039_103920

theorem exist_coprime_integers:
  ∀ (a b p : ℤ), ∃ (k l : ℤ), Int.gcd k l = 1 ∧ p ∣ (a * k + b * l) :=
by
  sorry

end exist_coprime_integers_l1039_103920


namespace greatest_number_is_2040_l1039_103963

theorem greatest_number_is_2040 (certain_number : ℕ) : 
  (∀ d : ℕ, d ∣ certain_number ∧ d ∣ 2037 → d ≤ 1) ∧ 
  (certain_number % 1 = 10) ∧ 
  (2037 % 1 = 7) → 
  certain_number = 2040 :=
by
  sorry

end greatest_number_is_2040_l1039_103963


namespace domain_of_function_l1039_103906

theorem domain_of_function : 
  {x : ℝ | x + 1 ≥ 0 ∧ x ≠ 1} = {x : ℝ | -1 ≤ x ∧ x < 1} ∪ {x : ℝ | 1 < x} :=
by 
  sorry

end domain_of_function_l1039_103906


namespace failed_in_english_l1039_103952

/- Lean definitions and statement -/

def total_percentage := 100
def failed_H := 32
def failed_H_and_E := 12
def passed_H_or_E := 24

theorem failed_in_english (total_percentage failed_H failed_H_and_E passed_H_or_E : ℕ) (h1 : total_percentage = 100) (h2 : failed_H = 32) (h3 : failed_H_and_E = 12) (h4 : passed_H_or_E = 24) :
  total_percentage - (failed_H + (total_percentage - passed_H_or_E - failed_H_and_E)) = 56 :=
by sorry

end failed_in_english_l1039_103952


namespace greatest_solution_of_equation_l1039_103985

theorem greatest_solution_of_equation : ∀ x : ℝ, x ≠ 9 ∧ (x^2 - x - 90) / (x - 9) = 4 / (x + 6) → x ≤ -7 :=
by
  intros x hx
  sorry

end greatest_solution_of_equation_l1039_103985


namespace intersection_distance_l1039_103936

open Real

-- Definition of the curve C in standard coordinates
def curve_C (x y : ℝ) : Prop :=
  y^2 = 4 * x

-- Definition of the line l in parametric form
def line_l (x y t : ℝ) : Prop :=
  x = 1 + t ∧ y = -1 + t

-- The length of the intersection points A and B of curve C and line l
theorem intersection_distance : ∃ t1 t2 : ℝ, (curve_C (1 + t1) (-1 + t1) ∧ curve_C (1 + t2) (-1 + t2)) ∧ (abs (t1 - t2) = 4 * sqrt 6) :=
sorry

end intersection_distance_l1039_103936


namespace arithmetic_seq_a2_l1039_103923

def arithmetic_seq (a : ℕ → ℤ) (d : ℤ) : Prop := ∀ n m : ℕ, a m = a (n + 1) + d * (m - (n + 1))

theorem arithmetic_seq_a2 
  (a : ℕ → ℤ) (d a1 : ℤ)
  (h_arith: ∀ n : ℕ, a n = a1 + n * d)
  (h_sum: a 3 + a 11 = 50)
  (h_a4: a 4 = 13) :
  a 2 = 5 :=
sorry

end arithmetic_seq_a2_l1039_103923


namespace decreasing_function_range_l1039_103935

theorem decreasing_function_range {f : ℝ → ℝ} (h_decreasing : ∀ x y : ℝ, x < y → f x > f y) :
  {x : ℝ | f (x^2 - 3 * x - 3) < f 1} = {x : ℝ | x < -1 ∨ x > 4} :=
by
  sorry

end decreasing_function_range_l1039_103935


namespace coefficient_of_x_in_expansion_l1039_103945

theorem coefficient_of_x_in_expansion : 
  (Polynomial.coeff (((X ^ 2 + 3 * X + 2) ^ 6) : Polynomial ℤ) 1) = 576 := 
by 
  sorry

end coefficient_of_x_in_expansion_l1039_103945


namespace digit_positions_in_8008_l1039_103921

theorem digit_positions_in_8008 :
  (8008 % 10 = 8) ∧ (8008 / 1000 % 10 = 8) :=
by
  sorry

end digit_positions_in_8008_l1039_103921


namespace total_people_in_boats_l1039_103997

theorem total_people_in_boats (bo_num : ℝ) (avg_people : ℝ) (bo_num_eq : bo_num = 3.0) (avg_people_eq : avg_people = 1.66666666699999) : ∃ total_people : ℕ, total_people = 6 := 
by
  sorry

end total_people_in_boats_l1039_103997


namespace cloth_sales_value_l1039_103917

theorem cloth_sales_value (commission_rate : ℝ) (commission : ℝ) (total_sales : ℝ) 
  (h1: commission_rate = 2.5)
  (h2: commission = 18)
  (h3: total_sales = commission / (commission_rate / 100)):
  total_sales = 720 := by
  sorry

end cloth_sales_value_l1039_103917


namespace tim_campaign_total_l1039_103937

theorem tim_campaign_total (amount_max : ℕ) (num_max : ℕ) (num_half : ℕ) (total_donations : ℕ) (total_raised : ℕ)
  (H1 : amount_max = 1200)
  (H2 : num_max = 500)
  (H3 : num_half = 3 * num_max)
  (H4 : total_donations = num_max * amount_max + num_half * (amount_max / 2))
  (H5 : total_donations = 40 * total_raised / 100) :
  total_raised = 3750000 :=
by
  -- Proof is omitted
  sorry

end tim_campaign_total_l1039_103937


namespace photo_students_count_l1039_103951

theorem photo_students_count (n m : ℕ) 
  (h1 : m - 1 = n + 4) 
  (h2 : m - 2 = n) : 
  n * m = 24 := 
by 
  sorry

end photo_students_count_l1039_103951


namespace total_amount_is_105_l1039_103988

theorem total_amount_is_105 (x_amount y_amount z_amount : ℝ) 
  (h1 : ∀ x, y_amount = x * 0.45) 
  (h2 : ∀ x, z_amount = x * 0.30) 
  (h3 : y_amount = 27) : 
  (x_amount + y_amount + z_amount = 105) := 
sorry

end total_amount_is_105_l1039_103988


namespace amy_balloons_l1039_103994

theorem amy_balloons (james_balloons amy_balloons : ℕ) (h1 : james_balloons = 1222) (h2 : james_balloons = amy_balloons + 709) : amy_balloons = 513 :=
by
  sorry

end amy_balloons_l1039_103994


namespace smallest_integer_x_l1039_103927

theorem smallest_integer_x (x : ℤ) : 
  ( ∀ x : ℤ, ( 2 * (x : ℚ) / 5 + 3 / 4 > 7 / 5 → 2 ≤ x )) :=
by
  intro x
  sorry

end smallest_integer_x_l1039_103927


namespace find_expression_for_a_n_l1039_103965

-- Definitions for conditions in the problem
variable (a : ℕ → ℝ) -- Sequence is of positive real numbers
variable (S : ℕ → ℝ) -- Sum of the first n terms of the sequence

-- Condition that all terms in the sequence a_n are positive and indexed by natural numbers starting from 1
axiom pos_seq : ∀ n : ℕ, 0 < a (n + 1)
-- Condition for the sum of the terms: 4S_n = a_n^2 + 2a_n for n ∈ ℕ*
axiom sum_condition : ∀ n : ℕ, 4 * S (n + 1) = (a (n + 1))^2 + 2 * a (n + 1)

-- Theorem stating that sequence a_n = 2n given the above conditions
theorem find_expression_for_a_n : ∀ n : ℕ, a (n + 1) = 2 * (n + 1) := by
  sorry

end find_expression_for_a_n_l1039_103965


namespace solve_recursive_fn_eq_l1039_103919

-- Define the recursive function
def recursive_fn (x : ℝ) : ℝ :=
  2 * (2 * (2 * (2 * (2 * x - 1) - 1) - 1) - 1) - 1

-- State the theorem we need to prove
theorem solve_recursive_fn_eq (x : ℝ) : recursive_fn x = x → x = 1 :=
by
  sorry

end solve_recursive_fn_eq_l1039_103919


namespace grid_labelings_count_l1039_103974

theorem grid_labelings_count :
  ∃ (labeling_count : ℕ), 
    labeling_count = 2448 ∧ 
    (∀ (grid : Matrix (Fin 3) (Fin 3) ℕ),
      grid 0 0 = 1 ∧ 
      grid 2 2 = 2009 ∧ 
      (∀ (i j : Fin 3), j < 2 → grid i j ∣ grid i (j + 1)) ∧ 
      (∀ (i j : Fin 3), i < 2 → grid i j ∣ grid (i + 1) j)) :=
sorry

end grid_labelings_count_l1039_103974


namespace completion_time_workshop_3_l1039_103911

-- Define the times for workshops
def time_in_workshop_3 : ℝ := 8
def time_in_workshop_1 : ℝ := time_in_workshop_3 + 10
def time_in_workshop_2 : ℝ := (time_in_workshop_3 + 10) - 3.6

-- Define the combined work equation
def combined_work_eq := (1 / time_in_workshop_1) + (1 / time_in_workshop_2) = (1 / time_in_workshop_3)

-- Final theorem statement
theorem completion_time_workshop_3 (h : combined_work_eq) : time_in_workshop_3 - 7 = 1 :=
by
  sorry

end completion_time_workshop_3_l1039_103911


namespace coin_problem_l1039_103938

theorem coin_problem :
  ∃ (p n d q : ℕ), p + n + d + q = 11 ∧ 
                   1 * p + 5 * n + 10 * d + 25 * q = 132 ∧
                   p ≥ 1 ∧ n ≥ 1 ∧ d ≥ 1 ∧ q ≥ 1 ∧ 
                   q = 3 :=
by
  sorry

end coin_problem_l1039_103938


namespace AM_GM_inequality_example_l1039_103954

open Real

theorem AM_GM_inequality_example 
  (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  ((a^2 * b + b^2 * c + c^2 * a) * (a * b^2 + b * c^2 + c * a^2)) ≥ 9 * (a^2 * b^2 * c^2) :=
sorry

end AM_GM_inequality_example_l1039_103954

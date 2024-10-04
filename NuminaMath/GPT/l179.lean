import Mathlib

namespace lizzie_garbage_l179_179332

/-- Let G be the amount of garbage Lizzie's group collected. 
We are given that the second group collected G - 39 pounds of garbage,
and the total amount collected by both groups is 735 pounds.
We need to prove that G is 387 pounds. -/
theorem lizzie_garbage (G : ℕ) (h1 : G + (G - 39) = 735) : G = 387 :=
sorry

end lizzie_garbage_l179_179332


namespace value_of_expression_l179_179003

theorem value_of_expression {p q : ℝ} (hp : 3 * p^2 + 9 * p - 21 = 0) (hq : 3 * q^2 + 9 * q - 21 = 0) : 
  (3 * p - 4) * (6 * q - 8) = 122 :=
by
  sorry

end value_of_expression_l179_179003


namespace volume_proportionality_l179_179186

variable (W V : ℕ)
variable (k : ℚ)

-- Given conditions
theorem volume_proportionality (h1 : V = k * W) (h2 : W = 112) (h3 : k = 3 / 7) :
  V = 48 := by
  sorry

end volume_proportionality_l179_179186


namespace office_distance_l179_179742

theorem office_distance (d t : ℝ) 
    (h1 : d = 40 * (t + 1.5)) 
    (h2 : d - 40 = 60 * (t - 2)) : 
    d = 340 :=
by
  -- The detailed proof omitted
  sorry

end office_distance_l179_179742


namespace interpretation_of_neg_two_pow_six_l179_179176

theorem interpretation_of_neg_two_pow_six :
  - (2^6) = -(6 * 2) :=
by
  sorry

end interpretation_of_neg_two_pow_six_l179_179176


namespace ellipse_equation_and_slope_range_l179_179573

theorem ellipse_equation_and_slope_range (a b : ℝ) (e : ℝ) (k : ℝ) :
  a > b ∧ b > 0 ∧ e = (Real.sqrt 3) / 3 ∧
  ∃! ℓ : ℝ × ℝ, (ℓ.2 = 1 ∧ ℓ.1 = -2) ∧
  ∀ x y : ℝ, x^2 + y^2 = b^2 → y = x + 2 →
  ((x - 0)^2 + (y - 0)^2 = b^2) ∧
  (
    (a^2 = (3 * b^2)) ∧ (b = Real.sqrt 2) ∧
    a > 0 ∧
    (∀ x y : ℝ, x^2 / 3 + y^2 / 2 = 1) ∧
    (-((Real.sqrt 2) / 2) < k ∧ k < 0) ∨ (0 < k ∧ k < ((Real.sqrt 2) / 2))
  ) :=
by
  sorry

end ellipse_equation_and_slope_range_l179_179573


namespace sum_is_constant_l179_179479

variable (a b c d : ℚ) -- declare variables states as rational numbers

theorem sum_is_constant :
  (a + 2 = b + 3 ∧ b + 3 = c + 4 ∧ c + 4 = d + 5 ∧ d + 5 = a + b + c + d + 7) →
  a + b + c + d = -(14 / 3) :=
by
  intros h
  sorry

end sum_is_constant_l179_179479


namespace min_value_f_l179_179651

noncomputable def f (x : ℝ) : ℝ := (x^2 - x + 3) / (x - 1)

theorem min_value_f : ∀ (x : ℝ), x ≥ 3 → ∃ m : ℝ, m = 9/2 ∧ ∀ y : ℝ, f y ≥ m :=
by
  sorry

end min_value_f_l179_179651


namespace tiffany_first_level_treasures_l179_179045

-- Conditions
def treasure_points : ℕ := 6
def treasures_second_level : ℕ := 5
def total_points : ℕ := 48

-- Definition for the number of treasures on the first level
def points_from_second_level : ℕ := treasures_second_level * treasure_points
def points_from_first_level : ℕ := total_points - points_from_second_level
def treasures_first_level : ℕ := points_from_first_level / treasure_points

-- The theorem to prove
theorem tiffany_first_level_treasures : treasures_first_level = 3 :=
by
  sorry

end tiffany_first_level_treasures_l179_179045


namespace original_plan_trees_per_day_l179_179831

theorem original_plan_trees_per_day (x : ℕ) :
  (∃ x, (960 / x - 960 / (2 * x) = 4)) → x = 120 := 
sorry

end original_plan_trees_per_day_l179_179831


namespace calculate_expression_l179_179376

theorem calculate_expression : 287 * 287 + 269 * 269 - (2 * 287 * 269) = 324 :=
by
  sorry

end calculate_expression_l179_179376


namespace correct_option_B_l179_179223

theorem correct_option_B (a b : ℝ) : (-a^2 * b^3)^2 = a^4 * b^6 := 
  sorry

end correct_option_B_l179_179223


namespace greatest_distance_between_vertices_l179_179248

theorem greatest_distance_between_vertices 
    (inner_perimeter outer_perimeter : ℝ) 
    (inner_square_perimeter_eq : inner_perimeter = 16)
    (outer_square_perimeter_eq : outer_perimeter = 40)
    : ∃ max_distance, max_distance = 2 * Real.sqrt 34 :=
by
  sorry

end greatest_distance_between_vertices_l179_179248


namespace technicians_in_workshop_l179_179347

theorem technicians_in_workshop (T R : ℕ) 
    (h1 : 700 * 15 = 800 * T + 650 * R)
    (h2 : T + R = 15) : T = 5 := 
by
  sorry

end technicians_in_workshop_l179_179347


namespace constant_sequence_l179_179404

theorem constant_sequence (a : ℕ → ℕ) (h : ∀ i j, 1 ≤ i ∧ i ≤ 2016 ∧ 1 ≤ j ∧ j ≤ 2016 → (i + j) ∣ (i * a i + j * a j)) :
  ∀ i j, 1 ≤ i ∧ i ≤ 2016 ∧ 1 ≤ j ∧ j ≤ 2016 → a i = a j :=
by
  sorry

end constant_sequence_l179_179404


namespace friends_raise_funds_l179_179239

theorem friends_raise_funds (total_amount friends_count min_amount amount_per_person: ℕ)
  (h1 : total_amount = 3000)
  (h2 : friends_count = 10)
  (h3 : min_amount = 300)
  (h4 : amount_per_person = total_amount / friends_count) :
  amount_per_person = min_amount :=
by
  sorry

end friends_raise_funds_l179_179239


namespace primes_solution_l179_179414

theorem primes_solution (p : ℕ) (n : ℕ) (h_prime : Prime p) (h_nat : 0 < n) : 
  (p^2 + n^2 = 3 * p * n + 1) ↔ (p = 3 ∧ n = 1) ∨ (p = 3 ∧ n = 8) := sorry

end primes_solution_l179_179414


namespace probability_red_or_blue_l179_179228

theorem probability_red_or_blue :
  ∀ (total_marbles white_marbles green_marbles red_blue_marbles : ℕ),
    total_marbles = 90 →
    (white_marbles : ℝ) / total_marbles = 1 / 6 →
    (green_marbles : ℝ) / total_marbles = 1 / 5 →
    white_marbles = 15 →
    green_marbles = 18 →
    red_blue_marbles = total_marbles - (white_marbles + green_marbles) →
    (red_blue_marbles : ℝ) / total_marbles = 19 / 30 :=
by
  intros total_marbles white_marbles green_marbles red_blue_marbles
  intros h_total_marbles h_white_prob h_green_prob h_white_count h_green_count h_red_blue_count
  sorry

end probability_red_or_blue_l179_179228


namespace find_x2_plus_y2_l179_179110

-- Given conditions as definitions in Lean
variable {x y : ℝ}
variable (h1 : x > 0)
variable (h2 : y > 0)
variable (h3 : x * y + x + y = 71)
variable (h4 : x^2 * y + x * y^2 = 880)

-- The statement to be proved
theorem find_x2_plus_y2 : x^2 + y^2 = 146 :=
by
  sorry

end find_x2_plus_y2_l179_179110


namespace number_of_solutions_l179_179744

theorem number_of_solutions (θ : ℝ) (h : 0 < θ ∧ θ ≤ 2 * Real.pi) :
  2 - 4 * Real.sin (2 * θ) + 3 * Real.cos (4 * θ) = 0 → 
  ∃ s : Fin 9, s.val = 8 :=
by
  sorry

end number_of_solutions_l179_179744


namespace vitamin_C_relationship_l179_179149

variables (A O G : ℝ)

-- Conditions given in the problem
def condition1 : Prop := A + O + G = 275
def condition2 : Prop := 2 * A + 3 * O + 4 * G = 683

-- Rewrite the math proof problem statement
theorem vitamin_C_relationship (h1 : condition1 A O G) (h2 : condition2 A O G) : O + 2 * G = 133 :=
by {
  sorry
}

end vitamin_C_relationship_l179_179149


namespace ruth_weekly_class_hours_l179_179945

def hours_in_a_day : ℕ := 8
def days_in_a_week : ℕ := 5
def weekly_school_hours := hours_in_a_day * days_in_a_week

def math_class_percentage : ℚ := 0.25
def language_class_percentage : ℚ := 0.30
def science_class_percentage : ℚ := 0.20
def history_class_percentage : ℚ := 0.10

def math_hours := math_class_percentage * weekly_school_hours
def language_hours := language_class_percentage * weekly_school_hours
def science_hours := science_class_percentage * weekly_school_hours
def history_hours := history_class_percentage * weekly_school_hours

def total_class_hours := math_hours + language_hours + science_hours + history_hours

theorem ruth_weekly_class_hours : total_class_hours = 34 := by
  -- Calculation proof logic will go here
  sorry

end ruth_weekly_class_hours_l179_179945


namespace original_board_length_before_final_cut_l179_179371

-- Given conditions
def initial_length : ℕ := 143
def first_cut_length : ℕ := 25
def final_cut_length : ℕ := 7

def board_length_after_first_cut : ℕ := initial_length - first_cut_length
def board_length_after_final_cut : ℕ := board_length_after_first_cut - final_cut_length

-- The theorem to be proved
theorem original_board_length_before_final_cut : board_length_after_first_cut + final_cut_length = 125 :=
by
  sorry

end original_board_length_before_final_cut_l179_179371


namespace differential_savings_l179_179305

theorem differential_savings (income : ℕ) (tax_rate_before : ℝ) (tax_rate_after : ℝ) : 
  income = 36000 → tax_rate_before = 0.46 → tax_rate_after = 0.32 →
  ((income * tax_rate_before) - (income * tax_rate_after)) = 5040 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end differential_savings_l179_179305


namespace impossibility_of_transition_l179_179805

theorem impossibility_of_transition 
  {a b c : ℤ}
  (h1 : a = 2)
  (h2 : b = 2)
  (h3 : c = 2) :
  ¬(∃ x y z : ℤ, x = 19 ∧ y = 1997 ∧ z = 1999 ∧
    (∃ n : ℕ, ∀ i < n, ∃ a' b' c' : ℤ, 
      if i = 0 then a' = 2 ∧ b' = 2 ∧ c' = 2 
      else (a', b', c') = 
        if i % 3 = 0 then (b + c - 1, b, c)
        else if i % 3 = 1 then (a, a + c - 1, c)
        else (a, b, a + b - 1) 
  )) :=
sorry

end impossibility_of_transition_l179_179805


namespace compute_large_expression_l179_179258

theorem compute_large_expression :
  ( (11^4 + 484) * (23^4 + 484) * (35^4 + 484) * (47^4 + 484) * (59^4 + 484) ) / 
  ( (5^4 + 484) * (17^4 + 484) * (29^4 + 484) * (41^4 + 484) * (53^4 + 484) ) = 552.42857 := 
sorry

end compute_large_expression_l179_179258


namespace counter_example_not_power_of_4_for_25_l179_179739

theorem counter_example_not_power_of_4_for_25 : ∃ n ≥ 2, n = 25 ∧ ¬ ∃ k : ℕ, 2 ^ (2 ^ n) % (2 ^ n - 1) = 4 ^ k :=
by {
  sorry
}

end counter_example_not_power_of_4_for_25_l179_179739


namespace simplify_sqrt_expression_correct_l179_179166

noncomputable def simplify_sqrt_expression (x : ℝ) : ℝ :=
  (sqrt (50 * x)) * (sqrt (18 * x)) * (sqrt (32 * x))

theorem simplify_sqrt_expression_correct (x : ℝ) : 
  simplify_sqrt_expression x = 120 * x * sqrt (x) := 
by 
  sorry

end simplify_sqrt_expression_correct_l179_179166


namespace distinct_nonzero_reals_equation_l179_179789

theorem distinct_nonzero_reals_equation {a b c d : ℝ} 
  (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0) (h₃ : d ≠ 0) 
  (h₄ : a ≠ b) (h₅ : b ≠ c) (h₆ : c ≠ d) (h₇ : d ≠ a) (h₈ : a ≠ c) (h₉ : b ≠ d)
  (h₁₀ : a * c = b * d) 
  (h₁₁ : a / b + b / c + c / d + d / a = 4) :
  (a / c + c / a + b / d + d / b = 4) :=
by
  sorry

end distinct_nonzero_reals_equation_l179_179789


namespace a_lt_c_lt_b_l179_179143

noncomputable def a : ℝ := Real.sin (14 * Real.pi / 180) + Real.cos (14 * Real.pi / 180)
noncomputable def b : ℝ := 2 * Real.sqrt 2 * Real.sin (30.5 * Real.pi / 180) * Real.cos (30.5 * Real.pi / 180)
noncomputable def c : ℝ := Real.sqrt 6 / 2

theorem a_lt_c_lt_b : a < c ∧ c < b := by
  sorry

end a_lt_c_lt_b_l179_179143


namespace gcd_105_490_l179_179268

theorem gcd_105_490 : Nat.gcd 105 490 = 35 := by
sorry

end gcd_105_490_l179_179268


namespace find_m_l179_179291

-- Define vectors as tuples
def a : ℝ × ℝ := (1, 2)
def b (m : ℝ) : ℝ × ℝ := (m, -1)
def c (m : ℝ) : ℝ × ℝ := (4, m)

-- Define vector subtraction
def sub_vect (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 - v.1, u.2 - v.2)

-- Define dot product
def dot_prod (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Prove the condition that (a - b) ⊥ c implies m = 4
theorem find_m (m : ℝ) (h : dot_prod (sub_vect a (b m)) (c m) = 0) : m = 4 :=
by
  sorry

end find_m_l179_179291


namespace largest_cube_side_length_largest_rect_parallelepiped_dimensions_l179_179474

variables {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

-- Part (a): Side length of the largest cube
theorem largest_cube_side_length :
  let a₀ := a * b * c / (a * b + b * c + a * c) in
  ∃ a₀, a₀ = a * b * c / (a * b + b * c + a * c) :=
begin
  sorry
end

-- Part (b): Dimensions of the largest rectangular parallelepiped
theorem largest_rect_parallelepiped_dimensions :
  let x := a / 3, y := b / 3, z := c / 3 in
  ∃ x y z, x = a / 3 ∧ y = b / 3 ∧ z = c / 3 :=
begin
  sorry
end

end largest_cube_side_length_largest_rect_parallelepiped_dimensions_l179_179474


namespace sasha_prediction_l179_179698

theorem sasha_prediction (n : ℕ) 
  (white_rook_students : ℕ)
  (black_elephant_students : ℕ)
  (total_games : ℕ) :
  white_rook_students = 15 → 
  black_elephant_students = 20 → 
  total_games = 300 → 
  n = 280 → 
  ∃ s : ℕ, s ≤ white_rook_students ∧ s ≤ black_elephant_students ∧ s * black_elephant_students ≥ total_games - n :=
by
  sorry

end sasha_prediction_l179_179698


namespace min_sum_ab_l179_179000

theorem min_sum_ab (a b : ℤ) (hab : a * b = 72) : a + b ≥ -17 := by
  sorry

end min_sum_ab_l179_179000


namespace breadth_of_rectangle_l179_179503

noncomputable def length (radius : ℝ) : ℝ := (1/4) * radius
noncomputable def side (sq_area : ℝ) : ℝ := Real.sqrt sq_area
noncomputable def radius (side : ℝ) : ℝ := side
noncomputable def breadth (rect_area length : ℝ) : ℝ := rect_area / length

theorem breadth_of_rectangle :
  breadth 200 (length (radius (side 1225))) = 200 / (1/4 * Real.sqrt 1225) :=
by
  sorry

end breadth_of_rectangle_l179_179503


namespace melt_brown_fabric_scientific_notation_l179_179147

theorem melt_brown_fabric_scientific_notation :
  0.000156 = 1.56 * 10^(-4) :=
sorry

end melt_brown_fabric_scientific_notation_l179_179147


namespace pears_sold_l179_179078

theorem pears_sold (m a total : ℕ) (h1 : a = 2 * m) (h2 : m = 120) (h3 : a = 240) : total = 360 :=
by
  sorry

end pears_sold_l179_179078


namespace find_a_l179_179817

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a * x^2 + x
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := 3 * x^2 + 2 * a * x + 1

theorem find_a (a : ℝ) : f' a 1 = 6 → a = 1 :=
by
  intro h
  have h_f_prime : 3 * (1 : ℝ) ^ 2 + 2 * a * (1 : ℝ) + 1 = 6 := h
  sorry

end find_a_l179_179817


namespace solve_equations_l179_179811

theorem solve_equations :
  (∀ x : ℝ, x^2 + 2 * x = 0 ↔ x = 0 ∨ x = -2) ∧ 
  (∀ x : ℝ, 4 * x^2 - 4 * x + 1 = 0 ↔ x = 1/2) :=
by sorry

end solve_equations_l179_179811


namespace orange_profit_loss_l179_179996

variable (C : ℝ) -- Cost price of one orange in rupees

-- Conditions as hypotheses
theorem orange_profit_loss :
  (1 / 16 - C) / C * 100 = 4 :=
by
  have h1 : 1.28 * C = 1 / 12 := sorry
  have h2 : C = 1 / (12 * 1.28) := sorry
  have h3 : C = 1 / 15.36 := sorry
  have h4 : (1/16 - C) = 1 / 384 := sorry
  -- Proof of main statement here
  sorry

end orange_profit_loss_l179_179996


namespace robot_handling_capacity_l179_179705

variables (x : ℝ) (A B : ℝ)

def robot_speed_condition1 : Prop :=
  A = B + 30

def robot_speed_condition2 : Prop :=
  1000 / A = 800 / B

theorem robot_handling_capacity
  (h1 : robot_speed_condition1 A B)
  (h2 : robot_speed_condition2 A B) :
  B = 120 ∧ A = 150 :=
by
  sorry

end robot_handling_capacity_l179_179705


namespace part_a_part_b_l179_179781

open Set

variable (S : Set (ℝ × ℝ))
variable (square : Set (ℝ × ℝ))
variable (side_length : ℝ)
variable (side_eq_one : side_length = 1)

-- Representation of the conditions
def is_in_square (x : ℝ × ℝ) : Prop :=
  (0 ≤ x.1) ∧ (x.1 ≤ side_length) ∧ (0 ≤ x.2) ∧ (x.2 ≤ side_length)

def distance_between_points (x y : ℝ × ℝ) : ℝ :=
  real.sqrt ((x.1 - y.1)^2 + (x.2 - y.2)^2)

def figure_property (fig : Set (ℝ × ℝ)) : Prop :=
  ∀ x y ∈ fig, x ≠ y → distance_between_points x y ≠ 0.001

-- Conditions
variable (fig_property : figure_property S)
variable (in_square : ∀ x ∈ S, is_in_square x)

-- Part (a)
theorem part_a : side_eq_one → fig_property S → in_square S → area S ≤ 0.34 :=
by
  sorry

-- Part (b)
theorem part_b : side_eq_one → fig_property S → in_square S → area S ≤ 0.287 :=
by
  sorry

end part_a_part_b_l179_179781


namespace largest_whole_number_m_satisfies_inequality_l179_179670

theorem largest_whole_number_m_satisfies_inequality :
  ∃ m : ℕ, (1 / 4 + m / 6 : ℚ) < 3 / 2 ∧ ∀ n : ℕ, (1 / 4 + n / 6 : ℚ) < 3 / 2 → n ≤ 7 :=
by
  sorry

end largest_whole_number_m_satisfies_inequality_l179_179670


namespace correlation_coefficient_is_one_l179_179312

noncomputable def correlation_coefficient (x_vals y_vals : List ℝ) : ℝ := sorry

theorem correlation_coefficient_is_one 
  (n : ℕ) 
  (x y : Fin n → ℝ) 
  (h1 : n ≥ 2) 
  (h2 : ∃ i j, i ≠ j ∧ x i ≠ x j) 
  (h3 : ∀ i, y i = 3 * x i + 1) : 
  correlation_coefficient (List.ofFn x) (List.ofFn y) = 1 := 
sorry

end correlation_coefficient_is_one_l179_179312


namespace portion_of_profit_divided_equally_l179_179485

-- Definitions for the given conditions
def total_investment_mary : ℝ := 600
def total_investment_mike : ℝ := 400
def total_profit : ℝ := 7500
def profit_diff : ℝ := 1000

-- Main statement
theorem portion_of_profit_divided_equally (E P : ℝ) 
  (h1 : total_profit = E + P)
  (h2 : E + (3/5) * P = E + (2/5) * P + profit_diff) :
  E = 2500 :=
by
  sorry

end portion_of_profit_divided_equally_l179_179485


namespace cylinder_volume_l179_179959

theorem cylinder_volume (r : ℝ) (h : ℝ) (A : ℝ) (V : ℝ) 
  (sphere_surface_area : A = 256 * Real.pi)
  (cylinder_height : h = 2 * r) 
  (sphere_surface_formula : A = 4 * Real.pi * r^2) 
  (cylinder_volume_formula : V = Real.pi * r^2 * h) : V = 1024 * Real.pi := 
by
  -- Definitions provided as conditions
  sorry

end cylinder_volume_l179_179959


namespace pentagon_area_sol_l179_179792

theorem pentagon_area_sol (a b : ℤ) (h1 : 0 < b) (h2 : b < a) (h3 : a * (3 * b + a) = 792) : a + b = 45 :=
sorry

end pentagon_area_sol_l179_179792


namespace songs_in_first_two_albums_l179_179254

/-
Beyonce releases 5 different singles on iTunes.
She releases 2 albums that each has some songs.
She releases 1 album that has 20 songs.
Beyonce has released 55 songs in total.
Prove that the total number of songs in the first two albums is 30.
-/

theorem songs_in_first_two_albums {A B : ℕ} 
  (h1 : 5 + A + B + 20 = 55) : 
  A + B = 30 :=
by
  sorry

end songs_in_first_two_albums_l179_179254


namespace math_problem_l179_179896

theorem math_problem
  (a b c d m : ℝ)
  (h1 : a = -b)
  (h2 : c = (1 / d) ∨ d = (1 / c))
  (h3 : |m| = 4) :
  (a + b = 0) ∧ (c * d = 1) ∧ (m = 4 ∨ m = -4) ∧
  ((a + b) / 3 + m^2 - 5 * (c * d) = 11) := by
  sorry

end math_problem_l179_179896


namespace bike_ride_distance_l179_179392

theorem bike_ride_distance (D : ℝ) (h : D / 10 = D / 15 + 0.5) : D = 15 :=
  sorry

end bike_ride_distance_l179_179392


namespace fraction_of_students_getting_A_l179_179310

theorem fraction_of_students_getting_A
    (frac_B : ℚ := 1/2)
    (frac_C : ℚ := 1/8)
    (frac_D : ℚ := 1/12)
    (frac_F : ℚ := 1/24)
    (passing_grade_frac: ℚ := 0.875) :
    (1 - (frac_B + frac_C + frac_D + frac_F) = 1/8) :=
by
  sorry

end fraction_of_students_getting_A_l179_179310


namespace simplify_eval_expression_l179_179625

theorem simplify_eval_expression : 
  ∀ (a b : ℤ), a = -1 → b = 4 → ((a - b)^2 - 2 * a * (a + b) + (a + 2 * b) * (a - 2 * b)) = -32 := 
by
  intros a b ha hb
  rw [ha, hb]
  sorry

end simplify_eval_expression_l179_179625


namespace relationship_y1_y2_l179_179595

theorem relationship_y1_y2 (k y1 y2 : ℝ) 
  (h1 : y1 = (k^2 + 1) * (-3) - 5) 
  (h2 : y2 = (k^2 + 1) * 4 - 5) : 
  y1 < y2 :=
sorry

end relationship_y1_y2_l179_179595


namespace chess_tournament_l179_179695

theorem chess_tournament (n : ℕ) (white_rook black_elephant : ℕ) (total_games : ℕ) :
  white_rook = 15 → black_elephant = 20 → total_games = white_rook * black_elephant → 
  (n ≥ 280 → (∃ p, p ∈ {1..white_rook} ∧ (n < 300))) :=
by
  intros hwr hbe htg hn
  sorry

end chess_tournament_l179_179695


namespace Gwen_money_left_l179_179877

theorem Gwen_money_left (received spent : ℕ) (h_received : received = 14) (h_spent : spent = 8) : 
  received - spent = 6 := 
by 
  sorry

end Gwen_money_left_l179_179877


namespace find_c_l179_179925

variable {r s b c : ℚ}

-- Conditions based on roots of the original quadratic equation
def roots_of_original_quadratic (r s : ℚ) := 
  (5 * r ^ 2 - 8 * r + 2 = 0) ∧ (5 * s ^ 2 - 8 * s + 2 = 0)

-- New quadratic equation with roots shifted by 3
def new_quadratic_roots (r s b c : ℚ) :=
  (r - 3) + (s - 3) = -b ∧ (r - 3) * (s - 3) = c 

theorem find_c (r s : ℚ) (hb : b = 22/5) : 
  (roots_of_original_quadratic r s) → 
  (new_quadratic_roots r s b c) → 
  c = 23/5 := 
by
  intros h1 h2
  sorry

end find_c_l179_179925


namespace fraction_zero_iff_numerator_zero_l179_179500

variable (x : ℝ)

def numerator (x : ℝ) : ℝ := x - 5
def denominator (x : ℝ) : ℝ := 6 * x + 12

theorem fraction_zero_iff_numerator_zero (h_denominator_nonzero : denominator 5 ≠ 0) : 
  numerator x / denominator x = 0 ↔ x = 5 :=
by sorry

end fraction_zero_iff_numerator_zero_l179_179500


namespace total_sweaters_calculated_l179_179139

def monday_sweaters := 8
def tuesday_sweaters := monday_sweaters + 2
def wednesday_sweaters := tuesday_sweaters - 4
def thursday_sweaters := tuesday_sweaters - 4
def friday_sweaters := monday_sweaters / 2

def total_sweaters := monday_sweaters + tuesday_sweaters + wednesday_sweaters + thursday_sweaters + friday_sweaters

theorem total_sweaters_calculated : total_sweaters = 34 := 
by sorry

end total_sweaters_calculated_l179_179139


namespace solve_x_plus_y_l179_179589

variable {x y : ℚ} -- Declare x and y as rational numbers

theorem solve_x_plus_y
  (h1: (1 / x) + (1 / y) = 1)
  (h2: (1 / x) - (1 / y) = 5) :
  x + y = -1 / 6 :=
sorry

end solve_x_plus_y_l179_179589


namespace eval_M_plus_N_l179_179795

open Finset

noncomputable def permutations_of (l : List ℕ) : List (List ℕ) :=
  List.permutations l

def sum_to_maximize (l : List ℕ) : ℕ :=
  match l with
  | [x1, x2, x3, x4, x5] => x1 * x2 + x2 * x3 + x3 * x4 + x4 * x5 + x5 * x1
  | _ => 0

noncomputable def max_value_and_count : ℕ × ℕ :=
  let perms := permutations_of [1, 2, 3, 4, 6]
  let max_sum := perms.map sum_to_maximize |>.max' ((1, 2, 3, 4, 6).map sum_to_maximize |>.min)
  let count := perms.filter (fun p => sum_to_maximize p = max_sum) |>.length
  (max_sum, count)

theorem eval_M_plus_N : max_value_and_count = (50, 10) → 60 := by
  intros h
  cases h with
  | intro m_val_count =>
    have m := m_val_count.fst
    have n := m_val_count.snd
    have mn := m + n
    have mn_eq : mn = 60 := by rfl
    exact mn_eq

#eval eval_M_plus_N (50, 10)

end eval_M_plus_N_l179_179795


namespace largest_base_5_three_digit_in_base_10_l179_179211

theorem largest_base_5_three_digit_in_base_10 :
  let n := 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  n = 124 :=
by
  let n := 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  show n = 124
  sorry

end largest_base_5_three_digit_in_base_10_l179_179211


namespace xiaolong_average_speed_l179_179225

noncomputable def averageSpeed (dist_home_store : ℕ) (time_home_store : ℕ) 
                               (speed_store_playground : ℕ) (time_store_playground : ℕ) 
                               (dist_playground_school : ℕ) (speed_playground_school : ℕ) 
                               (total_time : ℕ) : ℕ :=
  let dist_store_playground := speed_store_playground * time_store_playground
  let time_playground_school := dist_playground_school / speed_playground_school
  let total_distance := dist_home_store + dist_store_playground + dist_playground_school
  total_distance / total_time

theorem xiaolong_average_speed :
  averageSpeed 500 7 80 8 300 60 20 = 72 := by
  sorry

end xiaolong_average_speed_l179_179225


namespace jensen_miles_city_l179_179150

theorem jensen_miles_city (total_gallons : ℕ) (highway_miles : ℕ) (highway_mpg : ℕ)
  (city_mpg : ℕ) (highway_gallons : ℕ) (city_gallons : ℕ) (city_miles : ℕ) :
  total_gallons = 9 ∧ highway_miles = 210 ∧ highway_mpg = 35 ∧ city_mpg = 18 ∧
  highway_gallons = highway_miles / highway_mpg ∧
  city_gallons = total_gallons - highway_gallons ∧
  city_miles = city_gallons * city_mpg → city_miles = 54 :=
by
  sorry

end jensen_miles_city_l179_179150


namespace distance_to_campground_l179_179812

-- definitions for speeds and times
def speed1 : ℤ := 50
def time1 : ℤ := 3
def speed2 : ℤ := 60
def time2 : ℤ := 2
def speed3 : ℤ := 55
def time3 : ℤ := 1
def speed4 : ℤ := 65
def time4 : ℤ := 2

-- definitions for calculating the distances
def distance1 : ℤ := speed1 * time1
def distance2 : ℤ := speed2 * time2
def distance3 : ℤ := speed3 * time3
def distance4 : ℤ := speed4 * time4

-- definition for the total distance
def total_distance : ℤ := distance1 + distance2 + distance3 + distance4

-- proof statement
theorem distance_to_campground : total_distance = 455 := by
  sorry -- proof omitted

end distance_to_campground_l179_179812


namespace gasoline_price_april_l179_179129

theorem gasoline_price_april (P₀ : ℝ) (P₁ P₂ P₃ P₄ : ℝ) (x : ℝ)
  (h₁ : P₁ = P₀ * 1.20)  -- Price after January's increase
  (h₂ : P₂ = P₁ * 0.80)  -- Price after February's decrease
  (h₃ : P₃ = P₂ * 1.25)  -- Price after March's increase
  (h₄ : P₄ = P₃ * (1 - x / 100))  -- Price after April's decrease
  (h₅ : P₄ = P₀)  -- Price at the end of April equals the initial price
  : x = 17 := 
by
  sorry

end gasoline_price_april_l179_179129


namespace acute_triangle_sums_to_pi_over_4_l179_179308

theorem acute_triangle_sums_to_pi_over_4 
    (A B : ℝ) 
    (hA : 0 < A ∧ A < π / 2) 
    (hB : 0 < B ∧ B < π / 2) 
    (h_sinA : Real.sin A = (Real.sqrt 5)/5) 
    (h_sinB : Real.sin B = (Real.sqrt 10)/10) : 
    A + B = π / 4 := 
sorry

end acute_triangle_sums_to_pi_over_4_l179_179308


namespace arithmetic_expression_l179_179736

theorem arithmetic_expression :
  10 + 4 * (5 + 3)^3 = 2058 :=
by
  sorry

end arithmetic_expression_l179_179736


namespace larger_page_of_opened_book_l179_179906

theorem larger_page_of_opened_book (x : ℕ) (h : x + (x + 1) = 137) : x + 1 = 69 :=
sorry

end larger_page_of_opened_book_l179_179906


namespace triangle_median_inequality_l179_179422

variable (a b c m_a m_b m_c D : ℝ)

-- Assuming the conditions are required to make the proof valid
axiom median_formula_m_a : 4 * m_a^2 + a^2 = 2 * b^2 + 2 * c^2
axiom median_formula_m_b : 4 * m_b^2 + b^2 = 2 * c^2 + 2 * a^2
axiom median_formula_m_c : 4 * m_c^2 + c^2 = 2 * a^2 + 2 * b^2

theorem triangle_median_inequality : 
  a^2 + b^2 <= m_c * 6 * D ∧ b^2 + c^2 <= m_a * 6 * D ∧ c^2 + a^2 <= m_b * 6 * D → 
  (a^2 + b^2) / m_c + (b^2 + c^2) / m_a + (c^2 + a^2) / m_b <= 6 * D := 
by
  sorry

end triangle_median_inequality_l179_179422


namespace m_divides_product_iff_composite_ne_4_l179_179272

theorem m_divides_product_iff_composite_ne_4 (m : ℕ) : 
  (m ∣ Nat.factorial (m - 1)) ↔ 
  (∃ a b : ℕ, a ≠ b ∧ 1 < a ∧ 1 < b ∧ m = a * b ∧ m ≠ 4) := 
sorry

end m_divides_product_iff_composite_ne_4_l179_179272


namespace max_ratio_two_digit_mean_50_l179_179614

theorem max_ratio_two_digit_mean_50 : 
  ∀ (x y : ℕ), (10 ≤ x ∧ x ≤ 99) ∧ (10 ≤ y ∧ y ≤ 99) ∧ (x + y = 100) → ( x / y ) ≤ 99 := 
by
  intros x y h
  obtain ⟨hx, hy, hsum⟩ := h
  sorry

end max_ratio_two_digit_mean_50_l179_179614


namespace total_number_of_animals_l179_179849

-- Define the given conditions as hypotheses
def num_horses (T : ℕ) : Prop :=
  ∃ (H x z : ℕ), H + x + z = 75

def cows_vs_horses (T : ℕ) : Prop :=
  ∃ (w z : ℕ),  w = z + 10

-- Define the final conclusion we need to prove
def total_animals (T : ℕ) : Prop :=
  T = 170

-- The main theorem which states the conditions imply the conclusion
theorem total_number_of_animals (T : ℕ) (h1 : num_horses T) (h2 : cows_vs_horses T) : total_animals T :=
by
  -- Proof to be filled in later
  sorry

end total_number_of_animals_l179_179849


namespace equation_has_no_solution_l179_179273

theorem equation_has_no_solution (k : ℝ) : ¬ (∃ x : ℝ , (x ≠ 3 ∧ x ≠ 4) ∧ (x - 1) / (x - 3) = (x - k) / (x - 4)) ↔ k = 2 :=
by
  sorry

end equation_has_no_solution_l179_179273


namespace like_terms_sum_three_l179_179442

theorem like_terms_sum_three (m n : ℤ) (h1 : 2 * m = 4 - n) (h2 : m = n - 1) : m + n = 3 :=
sorry

end like_terms_sum_three_l179_179442


namespace shopkeeper_loss_percentages_l179_179991

theorem shopkeeper_loss_percentages 
  (TypeA : Type) (TypeB : Type) (TypeC : Type)
  (theft_percentage_A : ℝ) (theft_percentage_B : ℝ) (theft_percentage_C : ℝ)
  (hA : theft_percentage_A = 0.20)
  (hB : theft_percentage_B = 0.25)
  (hC : theft_percentage_C = 0.30)
  :
  (theft_percentage_A = 0.20 ∧ theft_percentage_B = 0.25 ∧ theft_percentage_C = 0.30) ∧
  ((theft_percentage_A + theft_percentage_B + theft_percentage_C) / 3 = 0.25) :=
by
  sorry

end shopkeeper_loss_percentages_l179_179991


namespace smaller_of_two_numbers_l179_179956

theorem smaller_of_two_numbers (a b : ℕ) (h1 : a * b = 4761) (h2 : 10 ≤ a ∧ a < 100) (h3 : 10 ≤ b ∧ b < 100) : min a b = 53 :=
by {
  sorry -- proof skips as directed
}

end smaller_of_two_numbers_l179_179956


namespace quality_of_algorithm_reflects_number_of_operations_l179_179824

-- Definitions
def speed_of_operation_is_important (c : Type) : Prop :=
  ∀ (c1 : c), true

-- Theorem stating that the number of operations within a unit of time is an important sign of the quality of an algorithm
theorem quality_of_algorithm_reflects_number_of_operations {c : Type} 
    (h_speed_important : speed_of_operation_is_important c) : 
  ∀ (a : Type) (q : a), true := 
sorry

end quality_of_algorithm_reflects_number_of_operations_l179_179824


namespace friends_recycled_pounds_l179_179978

-- Definitions of given conditions
def points_earned : ℕ := 6
def pounds_per_point : ℕ := 8
def zoe_pounds : ℕ := 25

-- Calculation based on given conditions
def total_pounds := points_earned * pounds_per_point
def friends_pounds := total_pounds - zoe_pounds

-- Statement of the proof problem
theorem friends_recycled_pounds : friends_pounds = 23 := by
  sorry

end friends_recycled_pounds_l179_179978


namespace smallest_number_divided_into_18_and_60_groups_l179_179518

theorem smallest_number_divided_into_18_and_60_groups : ∃ n : ℕ, (∀ m : ℕ, (m % 18 = 0 ∧ m % 60 = 0) → n ≤ m) ∧ (n % 18 = 0 ∧ n % 60 = 0) ∧ n = 180 :=
by
  use 180
  sorry

end smallest_number_divided_into_18_and_60_groups_l179_179518


namespace counseling_rooms_l179_179451

theorem counseling_rooms (n : ℕ) (x : ℕ)
  (h1 : n = 20 * x + 32)
  (h2 : n = 24 * (x - 1)) : x = 14 :=
by
  sorry

end counseling_rooms_l179_179451


namespace simplify_radical_expression_l179_179164

theorem simplify_radical_expression (x : ℝ) :
  (sqrt (50 * x)) * (sqrt (18 * x)) * (sqrt (32 * x)) = 120 * x * sqrt (2 * x) := by
sorry

end simplify_radical_expression_l179_179164


namespace angle_OQP_is_90_degrees_l179_179004

theorem angle_OQP_is_90_degrees (A B C D O P Q : Point) 
    (h1 : IsConvexQuadrilateral A B C D)
    (h2 : InscribedInCircle O A B C D)
    (h3 : P = intersection (line_through A C) (line_through B D))
    (h4 : Q ∈ circumcircle (triangle A P D))
    (h5 : Q ∈ circumcircle (triangle B P C))
    (h6 : Q ≠ P) :
    ∠ O Q P = 90 := 
begin
    sorry
end

end angle_OQP_is_90_degrees_l179_179004


namespace mother_age_is_correct_l179_179384

variable (D M : ℕ)

theorem mother_age_is_correct:
  (D + 3 = 26) → (M - 5 = 2 * (D - 5)) → M = 41 := by
  intros h1 h2
  sorry

end mother_age_is_correct_l179_179384


namespace problem_l179_179887

theorem problem (a b c d : ℝ) (h1 : b + c = 7) (h2 : c + d = 5) (h3 : a + d = 2) : a + b = 4 :=
sorry

end problem_l179_179887


namespace number_of_girls_l179_179779

theorem number_of_girls (B G: ℕ) 
  (ratio : 8 * G = 5 * B) 
  (total : B + G = 780) :
  G = 300 := 
sorry

end number_of_girls_l179_179779


namespace magazine_purchase_ways_l179_179068

theorem magazine_purchase_ways :
  let M := 8
  let N := 3
  let C := Nat.choose
  ∑ (C(N, 2) * C(M, 4)) + C(M, 5) = 266
by
  sorry

end magazine_purchase_ways_l179_179068


namespace solve_absolute_value_eq_l179_179630

theorem solve_absolute_value_eq (x : ℝ) : (|x - 3| = 5 - x) → x = 4 :=
by
  sorry

end solve_absolute_value_eq_l179_179630


namespace element_in_set_l179_179582

open Set

noncomputable def A : Set ℝ := { x | x < 2 * Real.sqrt 3 }
def a : ℝ := 2

theorem element_in_set : a ∈ A := by
  sorry

end element_in_set_l179_179582


namespace points_coplanar_if_and_only_if_b_neg1_l179_179748

/-- Points (0, 0, 0), (1, b, 0), (0, 1, b), (b, 0, 1) are coplanar if and only if b = -1. --/
theorem points_coplanar_if_and_only_if_b_neg1 (a b : ℝ) :
  (∃ u v w : ℝ, (u, v, w) = (0, 0, 0) ∨ (u, v, w) = (1, b, 0) ∨ (u, v, w) = (0, 1, b) ∨ (u, v, w) = (b, 0, 1)) →
  (b = -1) :=
sorry

end points_coplanar_if_and_only_if_b_neg1_l179_179748


namespace upstream_swim_distance_l179_179252

-- Definition of the speeds and distances
def downstream_speed (v : ℝ) := 5 + v
def upstream_speed (v : ℝ) := 5 - v
def distance := 54
def time := 6
def woman_speed_in_still_water := 5

-- Given condition: downstream_speed * time = distance
def downstream_condition (v : ℝ) := downstream_speed v * time = distance

-- Given condition: upstream distance is 'd' km
def upstream_distance (v : ℝ) := upstream_speed v * time

-- Prove that given the above conditions and solving the necessary equations, 
-- the distance swam upstream is 6 km.
theorem upstream_swim_distance {d : ℝ} (v : ℝ) (h1 : downstream_condition v) : upstream_distance v = 6 :=
by
  sorry

end upstream_swim_distance_l179_179252


namespace ratio_of_surface_areas_l179_179448

theorem ratio_of_surface_areas (s : ℝ) :
  let cube_surface_area := 6 * s^2
  let tetrahedron_edge := s * Real.sqrt 2
  let tetrahedron_face_area := (Real.sqrt 3 / 4) * (tetrahedron_edge)^2
  let tetrahedron_surface_area := 4 * tetrahedron_face_area
  (cube_surface_area / tetrahedron_surface_area) = Real.sqrt 3 :=
by
  let cube_surface_area := 6 * s^2
  let tetrahedron_edge := s * Real.sqrt 2
  let tetrahedron_face_area := (Real.sqrt 3 / 4) * (tetrahedron_edge)^2
  let tetrahedron_surface_area := 4 * tetrahedron_face_area
  show (cube_surface_area / tetrahedron_surface_area) = Real.sqrt 3
  sorry

end ratio_of_surface_areas_l179_179448


namespace square_perimeter_l179_179854

theorem square_perimeter (s : ℕ) (h : 5 * s / 2 = 40) : 4 * s = 64 := by
  sorry

end square_perimeter_l179_179854


namespace largest_base5_three_digit_in_base10_l179_179220

theorem largest_base5_three_digit_in_base10 :
  let a := 4
      b := 4
      c := 4
      largest_base5 := 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  in largest_base5 = 124 :=
by
  let a := 4
  let b := 4
  let c := 4
  let largest_base5 := 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  show largest_base5 = 124
  sorry

end largest_base5_three_digit_in_base10_l179_179220


namespace pascal_fifth_element_row_20_l179_179054

theorem pascal_fifth_element_row_20 :
  (Nat.choose 20 4) = 4845 := by
  sorry

end pascal_fifth_element_row_20_l179_179054


namespace consecutive_rolls_probability_l179_179341

theorem consecutive_rolls_probability : 
  let total_outcomes := 36
  let consecutive_events := 10
  (consecutive_events / total_outcomes : ℚ) = 5 / 18 :=
by
  sorry

end consecutive_rolls_probability_l179_179341


namespace tour_groups_and_savings_minimum_people_for_savings_l179_179092

theorem tour_groups_and_savings (x y : ℕ) (m : ℕ):
  (x + y = 102) ∧ (45 * x + 50 * y - 40 * 102 = 730) → 
  (x = 58 ∧ y = 44) :=
by
  sorry

theorem minimum_people_for_savings (m : ℕ):
  (∀ m, m < 50 → 50 * m > 45 * 51) → 
  (m ≥ 46) :=
by
  sorry

end tour_groups_and_savings_minimum_people_for_savings_l179_179092


namespace printer_task_total_pages_l179_179685

theorem printer_task_total_pages
  (A B : ℕ)
  (h1 : 1 / A + 1 / B = 1 / 24)
  (h2 : 1 / A = 1 / 60)
  (h3 : B = A + 6) :
  60 * A = 720 := by
  sorry

end printer_task_total_pages_l179_179685


namespace count_divisors_of_100000_l179_179599

theorem count_divisors_of_100000 : 
  ∃ n : ℕ, n = 36 ∧ ∀ k : ℕ, (k ∣ 100000) → ∃ (i j : ℕ), 0 ≤ i ∧ i ≤ 5 ∧ 0 ≤ j ∧ j ≤ 5 ∧ k = 2^i * 5^j := by
  sorry

end count_divisors_of_100000_l179_179599


namespace radius_of_inscribed_circle_l179_179833

noncomputable def radius_inscribed_circle (AB BC AC : ℝ) (s : ℝ) (K : ℝ) : ℝ := K / s

theorem radius_of_inscribed_circle (AB BC AC : ℝ) (h1: AB = 8) (h2: BC = 8) (h3: AC = 10) :
  radius_inscribed_circle AB BC AC 13 (5 * Real.sqrt 39) = (5 * Real.sqrt 39) / 13 :=
  by
  sorry

end radius_of_inscribed_circle_l179_179833


namespace triangle_expression_value_l179_179774

theorem triangle_expression_value :
  ∀ (A B C : ℝ) (a b c : ℝ),
  A = 60 ∧ b = 1 ∧ (1 / 2) * b * c * (Real.sin A) = Real.sqrt 3 →
  (a + 2 * b - 3 * c) / (Real.sin A + 2 * Real.sin B - 3 * Real.sin C) = 2 * (Real.sqrt 39) / 3 :=
by
  intro A B C a b c
  rintro ⟨hA, hb, h_area⟩
  sorry

end triangle_expression_value_l179_179774


namespace sqrt_81_eq_pm_9_l179_179658

theorem sqrt_81_eq_pm_9 (x : ℤ) (hx : x^2 = 81) : x = 9 ∨ x = -9 :=
by
  sorry

end sqrt_81_eq_pm_9_l179_179658


namespace train_passes_jogger_in_approximately_25_8_seconds_l179_179241

noncomputable def jogger_speed_kmh := 7
noncomputable def train_speed_kmh := 60
noncomputable def jogger_head_start_m := 180
noncomputable def train_length_m := 200

noncomputable def kmh_to_ms (speed_kmh : ℕ) : ℕ := speed_kmh * 1000 / 3600

noncomputable def jogger_speed_ms := kmh_to_ms jogger_speed_kmh
noncomputable def train_speed_ms := kmh_to_ms train_speed_kmh

noncomputable def relative_speed_ms := train_speed_ms - jogger_speed_ms
noncomputable def total_distance_to_cover_m := jogger_head_start_m + train_length_m
noncomputable def time_to_pass_sec := total_distance_to_cover_m / (relative_speed_ms : ℝ) 

theorem train_passes_jogger_in_approximately_25_8_seconds :
  abs (time_to_pass_sec - 25.8) < 0.1 := sorry

end train_passes_jogger_in_approximately_25_8_seconds_l179_179241


namespace set_subtraction_M_N_l179_179108

-- Definitions
def A : Set ℝ := { x | ∃ y, y = Real.sqrt (1 - x) }
def B : Set ℝ := { y | ∃ x, y = x^2 ∧ -1 ≤ x ∧ x ≤ 1 }
def M : Set ℝ := { x | ∃ y, y = Real.sqrt (1 - x) }
def N : Set ℝ := { y | ∃ x, y = x^2 ∧ -1 ≤ x ∧ x ≤ 1 }

-- Statement
theorem set_subtraction_M_N : (M \ N) = { x | x < 0 } := by
  sorry

end set_subtraction_M_N_l179_179108


namespace total_distance_maria_l179_179409

theorem total_distance_maria (D : ℝ)
  (half_dist : D/2 + (D/2 - D/8) + 180 = D) :
  3 * D / 8 = 180 → 
  D = 480 :=
by
  sorry

end total_distance_maria_l179_179409


namespace theo_cookies_per_sitting_l179_179044

-- Definitions from conditions
def sittings_per_day : ℕ := 3
def days_per_month : ℕ := 20
def cookies_in_3_months : ℕ := 2340

-- Calculation based on conditions
def sittings_per_month : ℕ := sittings_per_day * days_per_month
def sittings_in_3_months : ℕ := sittings_per_month * 3

-- Target statement
theorem theo_cookies_per_sitting :
  cookies_in_3_months / sittings_in_3_months = 13 :=
sorry

end theo_cookies_per_sitting_l179_179044


namespace maximize_revenue_l179_179841

-- Define the revenue function
def revenue (p : ℝ) : ℝ :=
  p * (150 - 4 * p)

-- Define the price constraints
def price_constraint (p : ℝ) : Prop :=
  0 ≤ p ∧ p ≤ 30

-- The theorem statement to prove that p = 19 maximizes the revenue
theorem maximize_revenue : ∀ p: ℕ, price_constraint p → revenue p ≤ revenue 19 :=
by
  sorry

end maximize_revenue_l179_179841


namespace simplify_and_rationalize_l179_179021

theorem simplify_and_rationalize : 
  (1 / (2 + (1 / (Real.sqrt 5 + 2)))) = (Real.sqrt 5 / 5) :=
by
  sorry

end simplify_and_rationalize_l179_179021


namespace original_length_before_final_cut_l179_179372

-- Defining the initial length of the board
def initial_length : ℕ := 143

-- Defining the length after the first cut
def length_after_first_cut : ℕ := initial_length - 25

-- Defining the length after the final cut
def length_after_final_cut : ℕ := length_after_first_cut - 7

-- Stating the theorem to prove that the original length of the board before cutting the final 7 cm is 125 cm
theorem original_length_before_final_cut : initial_length - 25 + 7 = 125 :=
sorry

end original_length_before_final_cut_l179_179372


namespace sample_mean_correct_probability_calc_overall_defect_rate_calc_defective_first_line_calc_l179_179487

noncomputable def sample_mean : ℝ :=
  (56 * 10 + 67 * 20 + 70 * 48 + 78 * 19 + 86 * 3) / 100

def variance := 36
def std_dev := Real.sqrt variance
def mu := sample_mean

noncomputable def normal_distribution_probability (a b μ σ : ℝ) : ℝ :=
  -- Here we calculate the probability for normal distribution
  sorry-- Implementation of the probability function.

def defect_rate_line1 := 0.015
def defect_rate_line2 := 0.018
def production_eff_line1 := 2 / 3
def production_eff_line2 := 1 / 3

def overall_defect_rate : ℝ :=
  production_eff_line1 * defect_rate_line1 + production_eff_line2 * defect_rate_line2

def defective_from_first_line (pa pb : ℝ) : ℝ :=
  (production_eff_line1 * defect_rate_line1) / pa
  
theorem sample_mean_correct : sample_mean = 70 := by
  sorry

theorem probability_calc : normal_distribution_probability 64 82 mu std_dev ≈ 0.8186 := by
  sorry

theorem overall_defect_rate_calc : overall_defect_rate = 0.016 := by
  sorry

theorem defective_first_line_calc : defective_from_first_line overall_defect_rate defect_rate_line1 = 5 / 8 := by
  sorry

end sample_mean_correct_probability_calc_overall_defect_rate_calc_defective_first_line_calc_l179_179487


namespace simplify_sqrt_expression_correct_l179_179167

noncomputable def simplify_sqrt_expression (x : ℝ) : ℝ :=
  (sqrt (50 * x)) * (sqrt (18 * x)) * (sqrt (32 * x))

theorem simplify_sqrt_expression_correct (x : ℝ) : 
  simplify_sqrt_expression x = 120 * x * sqrt (x) := 
by 
  sorry

end simplify_sqrt_expression_correct_l179_179167


namespace combined_molecular_weight_l179_179196

theorem combined_molecular_weight :
  let CaO_molecular_weight := 56.08
  let CO2_molecular_weight := 44.01
  let HNO3_molecular_weight := 63.01
  let moles_CaO := 5
  let moles_CO2 := 3
  let moles_HNO3 := 2
  moles_CaO * CaO_molecular_weight + moles_CO2 * CO2_molecular_weight + moles_HNO3 * HNO3_molecular_weight = 538.45 :=
by sorry

end combined_molecular_weight_l179_179196


namespace largest_cube_side_length_largest_rect_parallelepiped_dimensions_l179_179476

variables {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

-- Part (a): Side length of the largest cube
theorem largest_cube_side_length :
  let a₀ := a * b * c / (a * b + b * c + a * c) in
  ∃ a₀, a₀ = a * b * c / (a * b + b * c + a * c) :=
begin
  sorry
end

-- Part (b): Dimensions of the largest rectangular parallelepiped
theorem largest_rect_parallelepiped_dimensions :
  let x := a / 3, y := b / 3, z := c / 3 in
  ∃ x y z, x = a / 3 ∧ y = b / 3 ∧ z = c / 3 :=
begin
  sorry
end

end largest_cube_side_length_largest_rect_parallelepiped_dimensions_l179_179476


namespace abs_pi_sub_abs_pi_sub_three_l179_179550

theorem abs_pi_sub_abs_pi_sub_three (h : Real.pi > 3) : 
  abs (Real.pi - abs (Real.pi - 3)) = 2 * Real.pi - 3 := 
by
  sorry

end abs_pi_sub_abs_pi_sub_three_l179_179550


namespace real_part_implies_value_of_a_l179_179115

theorem real_part_implies_value_of_a (a b : ℝ) (h : a = 2 * b) (hb : b = 1) : a = 2 := by
  sorry

end real_part_implies_value_of_a_l179_179115


namespace profit_percentage_mobile_l179_179608

-- Definitions derived from conditions
def cost_price_grinder : ℝ := 15000
def cost_price_mobile : ℝ := 8000
def loss_percentage_grinder : ℝ := 0.05
def total_profit : ℝ := 50
def selling_price_grinder := cost_price_grinder * (1 - loss_percentage_grinder)
def total_cost_price := cost_price_grinder + cost_price_mobile
def total_selling_price := total_cost_price + total_profit
def selling_price_mobile := total_selling_price - selling_price_grinder
def profit_mobile := selling_price_mobile - cost_price_mobile

-- The theorem to prove the profit percentage on the mobile phone is 10%
theorem profit_percentage_mobile : (profit_mobile / cost_price_mobile) * 100 = 10 :=
by
  sorry

end profit_percentage_mobile_l179_179608


namespace initial_deposit_l179_179170

theorem initial_deposit (P R : ℝ) (h1 : 8400 = P + (P * R * 2) / 100) (h2 : 8760 = P + (P * (R + 4) * 2) / 100) : 
  P = 2250 :=
  sorry

end initial_deposit_l179_179170


namespace max_profit_at_300_l179_179706

-- Define the cost and revenue functions and total profit function

noncomputable def cost (x : ℝ) : ℝ := 20000 + 100 * x

noncomputable def revenue (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 390 then -x^3 / 900 + 400 * x else 90090

noncomputable def profit (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 390 then -x^3 / 900 + 300 * x - 20000 else -100 * x + 70090

-- The Lean statement for proving maximum profit occurs at x = 300
theorem max_profit_at_300 : ∀ x : ℝ, profit x ≤ profit 300 :=
sorry

end max_profit_at_300_l179_179706


namespace total_milk_in_a_week_l179_179710

theorem total_milk_in_a_week (cows : ℕ) (milk_per_cow_per_day : ℕ) (days_in_week : ℕ) (total_milk : ℕ) 
(h_cows : cows = 52) (h_milk_per_cow_per_day : milk_per_cow_per_day = 5) 
(h_days_in_week : days_in_week = 7) (h_total_milk : total_milk = 1820) : 
(cows * milk_per_cow_per_day * days_in_week) = total_milk :=
by simp [h_cows, h_milk_per_cow_per_day, h_days_in_week, h_total_milk]; sorry

end total_milk_in_a_week_l179_179710


namespace quadratic_has_one_solution_positive_value_of_n_l179_179101

theorem quadratic_has_one_solution_positive_value_of_n :
  ∃ n : ℝ, (4 * x ^ 2 + n * x + 1 = 0 → n ^ 2 - 16 = 0) ∧ n > 0 ∧ n = 4 :=
sorry

end quadratic_has_one_solution_positive_value_of_n_l179_179101


namespace square_of_sum_l179_179222

theorem square_of_sum (x y : ℝ) (A B C D : ℝ) :
  A = 2 * x^2 + y^2 →
  B = 2 * (x + y)^2 →
  C = 2 * x + y^2 →
  D = (2 * x + y)^2 →
  D = (2 * x + y)^2 :=
by intros; exact ‹D = (2 * x + y)^2›

end square_of_sum_l179_179222


namespace average_weight_of_B_C_D_E_l179_179814

theorem average_weight_of_B_C_D_E 
    (W_A W_B W_C W_D W_E : ℝ)
    (h1 : (W_A + W_B + W_C)/3 = 60)
    (h2 : W_A = 87)
    (h3 : (W_A + W_B + W_C + W_D)/4 = 65)
    (h4 : W_E = W_D + 3) :
    (W_B + W_C + W_D + W_E)/4 = 64 :=
by {
    sorry
}

end average_weight_of_B_C_D_E_l179_179814


namespace adrianna_gum_pieces_l179_179856

-- Definitions based on conditions
def initial_gum_pieces : ℕ := 10
def additional_gum_pieces : ℕ := 3
def friends_count : ℕ := 11

-- Expression to calculate the final pieces of gum
def total_gum_pieces : ℕ := initial_gum_pieces + additional_gum_pieces
def gum_left : ℕ := total_gum_pieces - friends_count

-- Lean statement we want to prove
theorem adrianna_gum_pieces: gum_left = 2 := 
by 
  sorry

end adrianna_gum_pieces_l179_179856


namespace combined_percentage_basketball_l179_179394

theorem combined_percentage_basketball (N_students : ℕ) (S_students : ℕ) 
  (N_percent_basketball : ℚ) (S_percent_basketball : ℚ) :
  N_students = 1800 → S_students = 3000 →
  N_percent_basketball = 0.25 → S_percent_basketball = 0.35 →
  ((N_students * N_percent_basketball) + (S_students * S_percent_basketball)) / (N_students + S_students) * 100 = 31 :=
by
  intros h1 h2 h3 h4
  simp [h1, h2, h3, h4]
  norm_num
  sorry

end combined_percentage_basketball_l179_179394


namespace proof_problem_l179_179279

def necessary_but_not_sufficient (P Q : Prop) : Prop :=
  (Q → P) ∧ ¬(P → Q)

theorem proof_problem (x : ℝ) :
  necessary_but_not_sufficient ((x+3)*(x-1) = 0) (x-1 = 0) :=
by
  sorry

end proof_problem_l179_179279


namespace pablo_days_to_complete_puzzles_l179_179016

-- Define the given conditions 
def puzzle_pieces_300 := 300
def puzzle_pieces_500 := 500
def puzzles_300 := 8
def puzzles_500 := 5
def rate_per_hour := 100
def max_hours_per_day := 7

-- Calculate total number of pieces
def total_pieces_300 := puzzles_300 * puzzle_pieces_300
def total_pieces_500 := puzzles_500 * puzzle_pieces_500
def total_pieces := total_pieces_300 + total_pieces_500

-- Calculate the number of pieces Pablo can put together per day
def pieces_per_day := max_hours_per_day * rate_per_hour

-- Calculate the number of days required for Pablo to complete all puzzles
def days_to_complete := total_pieces / pieces_per_day

-- Proposition to prove
theorem pablo_days_to_complete_puzzles : days_to_complete = 7 := sorry

end pablo_days_to_complete_puzzles_l179_179016


namespace tom_mowing_lawn_l179_179193

theorem tom_mowing_lawn (hours_to_mow : ℕ) (time_worked : ℕ) (fraction_mowed_per_hour : ℚ) : 
  (hours_to_mow = 6) → 
  (time_worked = 3) → 
  (fraction_mowed_per_hour = (1 : ℚ) / hours_to_mow) → 
  (1 - (time_worked * fraction_mowed_per_hour) = (1 : ℚ) / 2) :=
by
  intros h1 h2 h3
  sorry

end tom_mowing_lawn_l179_179193


namespace find_a_l179_179919

open Real

theorem find_a
    (sin_B cos_A cos_C sin_C : ℝ)
    (a b c : ℝ)
    (h1 : sin_B ^ 2 + cos_A ^ 2 - cos_C ^ 2 = sqrt 3 * sin_B * sin_C)
    (h2 : π * (2 : ℝ)^2 = 4 * π)
    (R : ℝ)
    (h3 : 2 = R) :
  a = 2 :=
sorry

end find_a_l179_179919


namespace range_for_a_l179_179366

noncomputable def line_not_in_second_quadrant (a : ℝ) : Prop :=
  ∀ x y : ℝ, (3 * a - 1) * x + (2 - a) * y - 1 = 0 → (x ≥ 0 ∨ y ≥ 0)

theorem range_for_a (a : ℝ) :
  (line_not_in_second_quadrant a) ↔ a ≥ 2 := by
  sorry

end range_for_a_l179_179366


namespace mathematician_daily_questions_l179_179245

/-- Given 518 questions for the first project and 476 for the second project,
if all questions are to be completed in 7 days, prove that the number
of questions completed each day is 142. -/
theorem mathematician_daily_questions (q1 q2 days questions_per_day : ℕ) 
  (h1 : q1 = 518) (h2 : q2 = 476) (h3 : days = 7) 
  (h4 : q1 + q2 = 994) (h5 : questions_per_day = 994 / 7) :
  questions_per_day = 142 :=
sorry

end mathematician_daily_questions_l179_179245


namespace evaluate_expression_l179_179221

theorem evaluate_expression (x : ℤ) (h : x = 3) : x^6 - 6 * x^2 + 7 * x = 696 :=
by
  have hx : x = 3 := h
  sorry

end evaluate_expression_l179_179221


namespace car_stops_at_three_seconds_l179_179172

theorem car_stops_at_three_seconds (t : ℝ) (h : -3 * t^2 + 18 * t = 0) : t = 3 := 
sorry

end car_stops_at_three_seconds_l179_179172


namespace problem_condition_necessary_and_sufficient_l179_179899

theorem problem_condition_necessary_and_sufficient (a b : ℝ) (h : a * b > 0) :
  (a > b) ↔ (1 / a < 1 / b) :=
sorry

end problem_condition_necessary_and_sufficient_l179_179899


namespace unique_z_value_l179_179718

theorem unique_z_value (x y u z : ℕ) (hx : 0 < x)
    (hy : 0 < y) (hu : 0 < u) (hz : 0 < z)
    (h1 : 3 + x + 21 = y + 25 + z)
    (h2 : 3 + x + 21 = 15 + u + 4)
    (h3 : y + 25 + z = 15 + u + 4)
    (h4 : 3 + y + 15 = x + 25 + u)
    (h5 : 3 + y + 15 = 21 + z + 4)
    (h6 : x + 25 + u = 21 + z + 4):
    z = 20 :=
by
    sorry

end unique_z_value_l179_179718


namespace pascal_triangle_row_20_element_5_l179_179048

theorem pascal_triangle_row_20_element_5 : nat.choose 20 4 = 4845 := by
  sorry

end pascal_triangle_row_20_element_5_l179_179048


namespace mask_production_l179_179846

theorem mask_production (M : ℕ) (h : 16 * M = 48000) : M = 3000 :=
by
  sorry

end mask_production_l179_179846


namespace largest_base5_eq_124_l179_179213

-- Define largest base-5 number with three digits
def largest_base5_three_digits : ℕ := 4 * 5^2 + 4 * 5^1 + 4 * 5^0

theorem largest_base5_eq_124 : largest_base5_three_digits = 124 :=
by
  -- Proof steps go here
  sorry

end largest_base5_eq_124_l179_179213


namespace smallest_c_exists_l179_179834

theorem smallest_c_exists (n : ℕ) (a : Fin n → ℝ) : 
  ∃ s : Finset (Fin n), |∑ i in s, a i - Real.floor (∑ i in s, a i) - 1| ≤ 1 / (n + 1) :=
sorry

end smallest_c_exists_l179_179834


namespace false_statements_l179_179427

variable (a b c : ℝ)

theorem false_statements (a b c : ℝ) :
  ¬(a > b → a^2 > b^2) ∧ ¬((a^2 > b^2) → a > b) ∧ ¬(a > b → a * c^2 > b * c^2) ∧ ¬(a > b ↔ |a| > |b|) :=
by
  sorry

end false_statements_l179_179427


namespace part1_part2_l179_179145

noncomputable def f (x a : ℝ) := (x + 1) * Real.log x - a * (x - 1)

theorem part1 : (∀ x a : ℝ, (x + 1) * Real.log x - a * (x - 1) = x - 1 → a = 1) := 
by sorry

theorem part2 (x : ℝ) (h : 1 < x ∧ x < 2) : 
  ( 1 / Real.log x - 1 / Real.log (x - 1) < 1 / ((x - 1) * (2 - x))) :=
by sorry

end part1_part2_l179_179145


namespace max_volume_cube_max_volume_parallelepiped_l179_179459

variables {a b c : ℝ}

-- Problem (a): Cube with the maximum volume entirely contained in the tetrahedron
theorem max_volume_cube (h : a > 0 ∧ b > 0 ∧ c > 0) :
  ∃ s : ℝ, s = (a * b * c) / (a * b + b * c + a * c) := sorry

-- Problem (b): Rectangular parallelepiped with the maximum volume entirely contained in the tetrahedron
theorem max_volume_parallelepiped (h : a > 0 ∧ b > 0 ∧ c > 0) :
  ∃ (x y z : ℝ),
  (x = a / 3 ∧ y = b / 3 ∧ z = c / 3) ∧
  (x * y * z = (a * b * c) / 27) := sorry

end max_volume_cube_max_volume_parallelepiped_l179_179459


namespace sin_triangle_sides_l179_179796

theorem sin_triangle_sides (a b c : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0)
  (h₃ : a + b + c ≤ 2 * Real.pi) (h₄ : a + b > c) (h₅ : b + c > a) (h₆ : c + a > b) :
  ∃ x y z : ℝ, x = Real.sin a ∧ y = Real.sin b ∧ z = Real.sin c ∧ x + y > z ∧ y + z > x ∧ z + x > y := 
by
  sorry

end sin_triangle_sides_l179_179796


namespace total_area_of_pyramid_faces_l179_179059

theorem total_area_of_pyramid_faces (base_edge lateral_edge : ℝ) (h : base_edge = 8) (k : lateral_edge = 5) : 
  4 * (1 / 2 * base_edge * 3) = 48 :=
by
  -- Base edge of the pyramid
  let b := base_edge
  -- Lateral edge of the pyramid
  let l := lateral_edge
  -- Half of the base
  let half_b := 4
  -- Height of the triangular face using Pythagorean theorem
  let h := 3
  -- Total area of four triangular faces
  have triangular_face_area : 1 / 2 * base_edge * h = 12 := sorry
  have total_area_of_faces : 4 * (1 / 2 * base_edge * h) = 48 := sorry
  exact total_area_of_faces

end total_area_of_pyramid_faces_l179_179059


namespace angle_between_hour_and_minute_hand_at_5_oclock_l179_179120

theorem angle_between_hour_and_minute_hand_at_5_oclock : 
  let degrees_in_circle := 360
  let hours_in_clock := 12
  let angle_per_hour := degrees_in_circle / hours_in_clock
  let hour_hand_position := 5
  let minute_hand_position := 0
  let angle := (hour_hand_position - minute_hand_position) * angle_per_hour
  angle = 150 :=
by sorry

end angle_between_hour_and_minute_hand_at_5_oclock_l179_179120


namespace problem1_problem2_problem3_problem4_l179_179868

theorem problem1 : 
  (3 / 5 : ℚ) - ((2 / 15) + (1 / 3)) = (2 / 15) := 
  by 
  sorry

theorem problem2 : 
  (-2 : ℤ) - 12 * ((1 / 3 : ℚ) - (1 / 4 : ℚ) + (1 / 2 : ℚ)) = -8 := 
  by 
  sorry

theorem problem3 : 
  (2 : ℤ) * (-3) ^ 2 - (6 / (-2) : ℚ) * (-1 / 3) = 17 := 
  by 
  sorry

theorem problem4 : 
  (-1 ^ 4 : ℤ) + ((abs (2 ^ 3 - 10)) : ℤ) - ((-3 : ℤ) / (-1) ^ 2019) = -2 := 
  by 
  sorry

end problem1_problem2_problem3_problem4_l179_179868


namespace tomato_seed_cost_l179_179071

theorem tomato_seed_cost (T : ℝ) 
  (h1 : 3 * 2.50 + 4 * T + 5 * 0.90 = 18) : 
  T = 1.50 := 
by
  sorry

end tomato_seed_cost_l179_179071


namespace complement_M_in_U_l179_179435

-- Define the universal set U
def U : Set ℝ := Set.univ

-- Define the set M
def M : Set ℝ := {x | ∃ y : ℝ, y = Real.sqrt (1 - x)}

-- State the theorem to prove that the complement of M in U is (1, +∞)
theorem complement_M_in_U :
  (U \ M) = {x | 1 < x} :=
by
  sorry

end complement_M_in_U_l179_179435


namespace arithmetic_sequence_ratio_l179_179557

def arithmetic_sum (a d n : ℕ) : ℕ := n * (2 * a + (n - 1) * d) / 2

theorem arithmetic_sequence_ratio :
  let a1 := 3
  let d1 := 3
  let l1 := 99
  let a2 := 4
  let d2 := 4
  let l2 := 100
  let n1 := (l1 - a1) / d1 + 1
  let n2 := (l2 - a2) / d2 + 1
  let sum1 := arithmetic_sum a1 d1 n1
  let sum2 := arithmetic_sum a2 d2 n2
  sum1 / sum2 = 1683 / 1300 :=
by {
  let a1 := 3
  let d1 := 3
  let l1 := 99
  let a2 := 4
  let d2 := 4
  let l2 := 100
  let n1 := (l1 - a1) / d1 + 1
  let n2 := (l2 - a2) / d2 + 1
  let sum1 := arithmetic_sum a1 d1 n1
  let sum2 := arithmetic_sum a2 d2 n2
  sorry
}

end arithmetic_sequence_ratio_l179_179557


namespace parabola_vertex_location_l179_179445

theorem parabola_vertex_location (a b c : ℝ) (h1 : ∀ x < 0, a * x^2 + b * x + c ≤ 0) (h2 : a < 0) : 
  -b / (2 * a) ≥ 0 :=
by
  sorry

end parabola_vertex_location_l179_179445


namespace xiaobin_duration_l179_179620

def t1 : ℕ := 9
def t2 : ℕ := 15

theorem xiaobin_duration : t2 - t1 = 6 := by
  sorry

end xiaobin_duration_l179_179620


namespace probability_blue_or_green_l179_179514

def faces : Type := {faces : ℕ // faces = 6}
noncomputable def blue_faces : ℕ := 3
noncomputable def red_faces : ℕ := 2
noncomputable def green_faces : ℕ := 1

theorem probability_blue_or_green :
  (blue_faces + green_faces) / 6 = (2 / 3) := by
  sorry

end probability_blue_or_green_l179_179514


namespace sum_k_log_term_l179_179256

-- Define the main theorem statement
theorem sum_k_log_term : 
  ∑ k in Finset.range 1500, k * (⌈Real.log k / Real.log (Real.sqrt 3)⌉ - ⌊Real.log k / Real.log (Real.sqrt 3)⌋) = 1124657 := 
sorry

end sum_k_log_term_l179_179256


namespace one_eighth_percent_of_160_plus_half_l179_179375

theorem one_eighth_percent_of_160_plus_half :
  ((1 / 8) / 100 * 160) + 0.5 = 0.7 :=
  sorry

end one_eighth_percent_of_160_plus_half_l179_179375


namespace factorable_polynomial_l179_179872

theorem factorable_polynomial (n : ℤ) :
  ∃ (a b c d e f : ℤ), 
    (a = 1) ∧ (d = 1) ∧ 
    (b + e = 2) ∧ 
    (f = b * e) ∧ 
    (c + f + b * e = 2) ∧ 
    (c * f + b * e = -n^2) ↔ 
    (n = 0 ∨ n = 2 ∨ n = -2) :=
by
  sorry

end factorable_polynomial_l179_179872


namespace middle_odd_number_is_26_l179_179185

theorem middle_odd_number_is_26 (x : ℤ) 
  (h : (x - 4) + (x - 2) + x + (x + 2) + (x + 4) = 130) : x = 26 := 
by 
  sorry

end middle_odd_number_is_26_l179_179185


namespace distinct_pos_numbers_implies_not_zero_at_least_one_of_abc_impossible_for_all_neq_l179_179876

noncomputable section

variables (a b c : ℝ) (ha : a ≠ b) (hb : b ≠ c) (hc : c ≠ a) (h1 : 0 < a) 
(h2 : 0 < b) (h3 : 0 < c)

theorem distinct_pos_numbers_implies_not_zero :
  (a - b) ^ 2 + (b - c) ^ 2 + (c - a) ^ 2 ≠ 0 :=
sorry

theorem at_least_one_of_abc :
  a > b ∨ a < b ∨ a = b :=
sorry

theorem impossible_for_all_neq :
  ¬(a ≠ c ∧ b ≠ c ∧ a ≠ b) :=
sorry

end distinct_pos_numbers_implies_not_zero_at_least_one_of_abc_impossible_for_all_neq_l179_179876


namespace part1_solution_part2_solution_l179_179570

open Real

noncomputable def f (x : ℝ) : ℝ := abs (x - 1) + abs (x - 3)

theorem part1_solution : ∀ x, f x ≤ 4 ↔ (0 ≤ x) ∧ (x ≤ 4) :=
by
  intro x
  sorry

theorem part2_solution : ∀ m, (∀ x, f x > m^2 + m) ↔ (-2 < m) ∧ (m < 1) :=
by
  intro m
  sorry

end part1_solution_part2_solution_l179_179570


namespace range_of_function_l179_179751

theorem range_of_function :
  ∀ y : ℝ, ∃ x : ℝ, (x ≤ 1/2) ∧ (y = 2 * x - Real.sqrt (1 - 2 * x)) ↔ y ∈ Set.Iic 1 := 
by
  sorry

end range_of_function_l179_179751


namespace max_cube_side_length_max_rect_parallelepiped_dimensions_l179_179471

-- Part (a)
theorem max_cube_side_length (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ s : ℝ, s = a * b * c / (a * b + b * c + a * c) :=
sorry

-- Part (b)
theorem max_rect_parallelepiped_dimensions (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ x y z : ℝ, x = a / 3 ∧ y = b / 3 ∧ z = c / 3 :=
sorry

end max_cube_side_length_max_rect_parallelepiped_dimensions_l179_179471


namespace number_of_ways_to_split_cities_l179_179309

-- Definitions of the grid and capitals
def city := ℤ × ℤ
def is_gondor_capital (c : city) : Prop := c = (-1, 1)
def is_mordor_capital (c : city) : Prop := c = (1, -1)
def grid := finset.city([(i, j) | i in [-1, 0, 1], j in [-1, 0, 1]])

-- Conditions
def central_city := (0, 0)
def reachable_from (start : city) (target : city) (countries : city → Prop) : Prop :=
  ∃ path : list city, path.head = start ∧ path.last = target ∧
  (∀ c ∈ path, countries c) ∧ (∀ (c₁ c₂ : city), (c₁, c₂) ∈ list.zip path (list.tail path) → ((abs (c₁.fst - c₂.fst) = 1) ∨ (abs (c₁.snd - c₂.snd) = 1)))

-- Total number of ways to split the cities
theorem number_of_ways_to_split_cities : 
  (∃ countries : city → Prop, 
    (countries central_city ∧ 
    reachable_from (-1, 1) central_city countries ∧
    reachable_from (1, -1) central_city (λ c, ¬countries c))) ∨
  (∃ countries : city → Prop, 
    (¬countries central_city ∧ 
    reachable_from (1, -1) central_city countries ∧ 
    reachable_from (-1, 1) central_city (λ c, ¬countries c))) :=
by sorry

end number_of_ways_to_split_cities_l179_179309


namespace fill_time_is_40_minutes_l179_179334

-- Definitions based on the conditions
def pool_volume : ℝ := 60 -- gallons
def filling_rate : ℝ := 1.6 -- gallons per minute
def leaking_rate : ℝ := 0.1 -- gallons per minute

-- Net filling rate
def net_filling_rate : ℝ := filling_rate - leaking_rate

-- Required time to fill the pool
def time_to_fill_pool : ℝ := pool_volume / net_filling_rate

-- Theorem to prove the time is 40 minutes
theorem fill_time_is_40_minutes : time_to_fill_pool = 40 := 
by
  -- This is where the proof would go
  sorry

end fill_time_is_40_minutes_l179_179334


namespace julieta_total_cost_l179_179320

variable (initial_backpack_price : ℕ)
variable (initial_binder_price : ℕ)
variable (backpack_price_increase : ℕ)
variable (binder_price_reduction : ℕ)
variable (discount_rate : ℕ)
variable (num_binders : ℕ)

def calculate_total_cost (initial_backpack_price initial_binder_price backpack_price_increase binder_price_reduction discount_rate num_binders : ℕ) : ℝ :=
  let new_backpack_price := initial_backpack_price + backpack_price_increase
  let new_binder_price := initial_binder_price - binder_price_reduction
  let total_bindable_cost := min num_binders ((num_binders + 1) / 2 * new_binder_price)
  let total_pre_discount := new_backpack_price + total_bindable_cost
  let discount_amount := total_pre_discount * discount_rate / 100
  let total_price := total_pre_discount - discount_amount
  total_price

theorem julieta_total_cost
  (initial_backpack_price : ℕ)
  (initial_binder_price : ℕ)
  (backpack_price_increase : ℕ)
  (binder_price_reduction : ℕ)
  (discount_rate : ℕ)
  (num_binders : ℕ)
  (h_initial_backpack : initial_backpack_price = 50)
  (h_initial_binder : initial_binder_price = 20)
  (h_backpack_inc : backpack_price_increase = 5)
  (h_binder_red : binder_price_reduction = 2)
  (h_discount : discount_rate = 10)
  (h_num_binders : num_binders = 3) :
  calculate_total_cost initial_backpack_price initial_binder_price backpack_price_increase binder_price_reduction discount_rate num_binders = 81.90 :=
by
  sorry

end julieta_total_cost_l179_179320


namespace sarahs_score_is_140_l179_179810

theorem sarahs_score_is_140 (g s : ℕ) (h1 : s = g + 60) 
  (h2 : (s + g) / 2 = 110) (h3 : s + g < 450) : s = 140 :=
by
  sorry

end sarahs_score_is_140_l179_179810


namespace move_point_right_3_units_from_neg_2_l179_179859

noncomputable def move_point_to_right (start : ℤ) (units : ℤ) : ℤ :=
start + units

theorem move_point_right_3_units_from_neg_2 : move_point_to_right (-2) 3 = 1 :=
by
  sorry

end move_point_right_3_units_from_neg_2_l179_179859


namespace flour_more_than_salt_l179_179486

open Function

-- Definitions based on conditions
def flour_needed : ℕ := 12
def flour_added : ℕ := 2
def salt_needed : ℕ := 7
def salt_added : ℕ := 0

-- Given that these definitions hold, prove the following theorem
theorem flour_more_than_salt : (flour_needed - flour_added) - (salt_needed - salt_added) = 3 :=
by
  -- Here you would include the proof, but as instructed, we skip it with "sorry".
  sorry

end flour_more_than_salt_l179_179486


namespace simplify_rationalize_expr_l179_179027

theorem simplify_rationalize_expr :
  (1 / (2 + 1 / (Real.sqrt 5 + 2))) = (Real.sqrt 5 / 5) :=
by
  sorry

end simplify_rationalize_expr_l179_179027


namespace largest_base5_three_digit_to_base10_l179_179198

theorem largest_base5_three_digit_to_base10 : 
  let base5_number := 4 * 5^2 + 4 * 5^1 + 4 * 5^0 in
  base5_number = 124 :=
by
let base5_number := 4 * 5^2 + 4 * 5^1 + 4 * 5^0
have h1 : 4 * 5^2 = 100 := by norm_num
have h2 : 4 * 5^1 = 20 := by norm_num
have h3 : 4 * 5^0 = 4 := by norm_num
have h4 : base5_number = 100 + 20 + 4 := by simp [base5_number, h1, h2, h3]
show base5_number = 124, by simp [h4]
sorry

end largest_base5_three_digit_to_base10_l179_179198


namespace relationship_between_variables_l179_179020

theorem relationship_between_variables
  (a b x y : ℚ)
  (h1 : x + y = a + b)
  (h2 : y - x < a - b)
  (h3 : b > a) :
  y < a ∧ a < b ∧ b < x :=
sorry

end relationship_between_variables_l179_179020


namespace find_c_l179_179555

   noncomputable def c_value (c : ℝ) : Prop :=
     ∃ (x y : ℝ), (x^2 - 8*x + y^2 + 10*y + c = 0) ∧ (x - 4)^2 + (y + 5)^2 = 25

   theorem find_c (c : ℝ) : c_value c → c = 16 := by
     sorry
   
end find_c_l179_179555


namespace sum_gcd_lcm_is_39_l179_179674

theorem sum_gcd_lcm_is_39 : Nat.gcd 30 81 + Nat.lcm 36 12 = 39 := by 
  sorry

end sum_gcd_lcm_is_39_l179_179674


namespace cyclic_inequality_l179_179622

theorem cyclic_inequality (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) :
  (x + y) * Real.sqrt (y + z) * Real.sqrt (z + x) + (y + z) * Real.sqrt (z + x) * Real.sqrt (x + y) + (z + x) * Real.sqrt (x + y) * Real.sqrt (y + z) ≥ 4 * (x * y + y * z + z * x) :=
by
  sorry

end cyclic_inequality_l179_179622


namespace total_earnings_l179_179410

def num_members : ℕ := 20
def candy_bars_per_member : ℕ := 8
def cost_per_candy_bar : ℝ := 0.5

theorem total_earnings :
  (num_members * candy_bars_per_member * cost_per_candy_bar) = 80 :=
by
  sorry

end total_earnings_l179_179410


namespace average_age_calculated_years_ago_l179_179343

theorem average_age_calculated_years_ago
  (n m : ℕ) (a b : ℕ) 
  (total_age_original : ℝ)
  (average_age_original : ℝ)
  (average_age_new : ℝ) :
  n = 6 → 
  a = 19 → 
  m = 7 → 
  b = 1 → 
  total_age_original = n * a → 
  average_age_original = a → 
  average_age_new = a →
  (total_age_original + b) / m = a → 
  1 = 1 := 
by
  intros _ _ _ _ _ _ _ _
  sorry

end average_age_calculated_years_ago_l179_179343


namespace win_sector_area_l179_179235

theorem win_sector_area (r : ℝ) (h1 : r = 8) (h2 : (1 / 4) = 1 / 4) : 
  ∃ (area : ℝ), area = 16 * Real.pi := 
by
  existsi (16 * Real.pi); exact sorry

end win_sector_area_l179_179235


namespace expand_fraction_product_l179_179559

-- Define the variable x and the condition that x ≠ 0
variable (x : ℝ) (h : x ≠ 0)

-- State the theorem
theorem expand_fraction_product (h : x ≠ 0) :
  3 / 7 * (7 / x^2 + 7 * x - 7 / x) = 3 / x^2 + 3 * x - 3 / x :=
sorry

end expand_fraction_product_l179_179559


namespace max_three_digit_is_931_l179_179966

def greatest_product_three_digit : Prop :=
  ∃ (a b c d e : ℕ), {a, b, c, d, e} = {1, 3, 5, 8, 9} ∧
  1 ≤ a ∧ a ≤ 9 ∧
  1 ≤ b ∧ b ≤ 9 ∧
  1 ≤ c ∧ c ≤ 9 ∧
  1 ≤ d ∧ d ≤ 9 ∧
  1 ≤ e ∧ e ≤ 9 ∧
  (100 * a + 10 * b + c) * (10 * d + e) = 81685 ∧
  (100 * a + 10 * b + c) = 931

theorem max_three_digit_is_931 : greatest_product_three_digit :=
begin
  sorry
end

end max_three_digit_is_931_l179_179966


namespace spokes_ratio_l179_179727

theorem spokes_ratio (B : ℕ) (front_spokes : ℕ) (total_spokes : ℕ) 
  (h1 : front_spokes = 20) 
  (h2 : total_spokes = 60) 
  (h3 : front_spokes + B = total_spokes) : 
  B / front_spokes = 2 :=
by 
  sorry

end spokes_ratio_l179_179727


namespace perpendicular_planes_normal_vector_l179_179889

def dot_product (a b : ℝ × ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2.1 * b.2.1 + a.2.2 * b.2.2

theorem perpendicular_planes_normal_vector {m : ℝ} 
  (a : ℝ × ℝ × ℝ) (b : ℝ × ℝ × ℝ) 
  (h₁ : a = (1, 2, -2)) 
  (h₂ : b = (-2, 1, m)) 
  (h₃ : dot_product a b = 0) : 
  m = 0 := 
sorry

end perpendicular_planes_normal_vector_l179_179889


namespace fruit_basket_combinations_l179_179893

namespace FruitBasket

def apples := 3
def oranges := 8
def min_apples := 1
def min_oranges := 1

theorem fruit_basket_combinations : 
  (apples + 1 - min_apples) * (oranges + 1 - min_oranges) = 36 := by
  sorry

end FruitBasket

end fruit_basket_combinations_l179_179893


namespace teddy_bears_ordered_l179_179492

theorem teddy_bears_ordered (days : ℕ) (T : ℕ)
  (h1 : 20 * days + 100 = T)
  (h2 : 23 * days - 20 = T) :
  T = 900 ∧ days = 40 := 
by 
  sorry

end teddy_bears_ordered_l179_179492


namespace min_games_to_predict_l179_179701

theorem min_games_to_predict (W B : ℕ) (total_games : ℕ) (n : ℕ) : 
  W = 15 → B = 20 → total_games = W * B → n = 280 → 
  (∃ x, x ∈ {i | ∃ j, i < W ∧ j < B}) :=
by
  intros hW hB htotal hn
  sorry

end min_games_to_predict_l179_179701


namespace sufficient_but_not_necessary_l179_179107

theorem sufficient_but_not_necessary (a b : ℝ) : (ab >= 2) -> a^2 + b^2 >= 4 ∧ ∃ a b : ℝ, a^2 + b^2 >= 4 ∧ ab < 2 := by
  sorry

end sufficient_but_not_necessary_l179_179107


namespace groupD_can_form_triangle_l179_179954

def groupA := (5, 7, 12)
def groupB := (7, 7, 15)
def groupC := (6, 9, 16)
def groupD := (6, 8, 12)

def canFormTriangle (a b c : Nat) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem groupD_can_form_triangle : canFormTriangle 6 8 12 :=
by
  -- Proof of the above theorem will follow the example from the solution.
  sorry

end groupD_can_form_triangle_l179_179954


namespace number_of_valid_pairs_l179_179483

theorem number_of_valid_pairs : ∃ p : Finset (ℕ × ℕ), 
  (∀ (a b : ℕ), (a, b) ∈ p ↔ a ≤ 10 ∧ b ≤ 10 ∧ 3 * b < a ∧ a < 4 * b) ∧ p.card = 2 :=
by
  sorry

end number_of_valid_pairs_l179_179483


namespace gain_percentage_l179_179837

theorem gain_percentage (x : ℝ) (CP : ℝ := 50 * x) (SP : ℝ := 60 * x) (Profit : ℝ := 10 * x) :
  ((Profit / CP) * 100) = 20 := 
by
  sorry

end gain_percentage_l179_179837


namespace div_by_19_l179_179808

theorem div_by_19 (n : ℕ) (h : n > 0) : (3^(3*n+2) + 5 * 2^(3*n+1)) % 19 = 0 := by
  sorry

end div_by_19_l179_179808


namespace mark_total_flowers_l179_179336

theorem mark_total_flowers (yellow purple green total : ℕ) 
  (hyellow : yellow = 10)
  (hpurple : purple = yellow + (yellow * 80 / 100))
  (hgreen : green = (yellow + purple) * 25 / 100)
  (htotal : total = yellow + purple + green) : 
  total = 35 :=
by
  sorry

end mark_total_flowers_l179_179336


namespace exists_multiple_with_sum_divisible_l179_179927

-- Define the sum of the digits
def sum_of_digits (n : ℕ) : ℕ := -- Implementation of sum_of_digits function is omitted here
sorry

-- Main theorem statement
theorem exists_multiple_with_sum_divisible (n : ℕ) (hn : n > 0) : 
  ∃ k, k % n = 0 ∧ sum_of_digits k ∣ k :=
sorry

end exists_multiple_with_sum_divisible_l179_179927


namespace sum_of_sides_l179_179446

variable (A B C : ℝ)
variable (a b c : ℝ)
variable (cosB cosC : ℝ)
variable (sinB : ℝ)
variable (area : ℝ)

-- Given conditions
axiom h1 : b = 2
axiom h2 : b * cosC + c * cosB = 3 * a * cosB
axiom h3 : area = 3 * Real.sqrt 2 / 2
axiom h4 : sinB = Real.sqrt (1 - cosB ^ 2)

-- Prove the desired result
theorem sum_of_sides (A B C a b c cosB cosC sinB : ℝ) (area : ℝ)
  (h1 : b = 2)
  (h2 : b * cosC + c * cosB = 3 * a * cosB)
  (h3 : area = 3 * Real.sqrt 2 / 2)
  (h4 : sinB = Real.sqrt (1 - cosB ^ 2)) :
  a + c = 4 := 
sorry

end sum_of_sides_l179_179446


namespace sum_of_digits_eq_4_l179_179452

theorem sum_of_digits_eq_4 (A B C D X Y : ℕ) (h1 : A + B + C + D = 22) (h2 : B + D = 9) (h3 : X = 1) (h4 : Y = 3) :
    X + Y = 4 :=
by
  sorry

end sum_of_digits_eq_4_l179_179452


namespace jill_basket_total_weight_l179_179137

def jill_basket_capacity : ℕ := 24
def type_a_weight : ℕ := 150
def type_b_weight : ℕ := 170
def jill_basket_type_a_count : ℕ := 12
def jill_basket_type_b_count : ℕ := 12

theorem jill_basket_total_weight :
  (jill_basket_type_a_count * type_a_weight + jill_basket_type_b_count * type_b_weight) = 3840 :=
by
  -- We provide the calculations for clarification; not essential to the theorem statement
  -- (12 * 150) + (12 * 170) = 1800 + 2040 = 3840
  -- Started proof to provide context; actual proof steps are omitted
  sorry

end jill_basket_total_weight_l179_179137


namespace mixed_doubles_count_l179_179189

theorem mixed_doubles_count : 
  let males := 5
  let females := 4
  ∃ (ways : ℕ), (ways = (Nat.choose males 2) * (Nat.choose females 2) * 2) ∧ ways = 120 := 
by
  sorry

end mixed_doubles_count_l179_179189


namespace max_cube_side_length_max_parallelepiped_dimensions_l179_179463

theorem max_cube_side_length (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) : 
    ∃ (a0 : ℝ), a0 = a * b * c / (a * b + b * c + a * c) := 
sorry

theorem max_parallelepiped_dimensions (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) : 
    ∃ (x y z : ℝ), (x = a / 3) ∧ (y = b / 3) ∧ (z = c / 3) :=
sorry

end max_cube_side_length_max_parallelepiped_dimensions_l179_179463


namespace fraction_multiplication_l179_179971

theorem fraction_multiplication :
  (2 / 3) * (3 / 8) = (1 / 4) :=
sorry

end fraction_multiplication_l179_179971


namespace sequence_next_number_l179_179527

def next_number_in_sequence (seq : List ℕ) : ℕ :=
  if seq = [1, 1, 2, 1, 2, 3, 1, 2, 3, 4, 1, 2] then 3 else sorry

theorem sequence_next_number :
  next_number_in_sequence [1, 1, 2, 1, 2, 3, 1, 2, 3, 4, 1, 2] = 3 :=
by
  -- This proof is to ensure the pattern conditions are met
  sorry

end sequence_next_number_l179_179527


namespace problem_equivalent_l179_179588

theorem problem_equivalent : ∀ m : ℝ, 2 * m^2 + m = -1 → 4 * m^2 + 2 * m + 5 = 3 := 
by
  intros m h
  sorry

end problem_equivalent_l179_179588


namespace f_four_l179_179797

noncomputable def f : ℝ → ℝ := sorry

axiom functional_eq (a b : ℝ) : f (a + b) + f (a - b) = 2 * f a + 2 * f b
axiom f_two : f 2 = 9 
axiom not_identically_zero : ¬ ∀ x : ℝ, f x = 0

theorem f_four : f 4 = 36 :=
by sorry

end f_four_l179_179797


namespace total_money_spent_l179_179609

-- Assume Keanu gave dog 40 fish
def dog_fish := 40

-- Assume Keanu gave cat half as many fish as he gave to his dog
def cat_fish := dog_fish / 2

-- Assume each fish cost $4
def cost_per_fish := 4

-- Prove that total amount of money spent is $240
theorem total_money_spent : (dog_fish + cat_fish) * cost_per_fish = 240 := 
by
  sorry

end total_money_spent_l179_179609


namespace simplify_expression_l179_179491

theorem simplify_expression (y : ℝ) :
  4 * y - 8 * y^2 + 6 - (3 - 6 * y - 9 * y^2 + 2 * y^3) = -2 * y^3 + y^2 + 10 * y + 3 := 
by
  -- Proof goes here, but we just state sorry for now
  sorry

end simplify_expression_l179_179491


namespace length_of_rectangular_plot_l179_179539

variable (L : ℕ)

-- Given conditions
def width := 50
def poles := 14
def distance_between_poles := 20
def intervals := poles - 1
def perimeter := intervals * distance_between_poles

-- The perimeter of the rectangle in terms of length and width
def rectangle_perimeter := 2 * (L + width)

-- The main statement to be proven
theorem length_of_rectangular_plot :
  rectangle_perimeter L = perimeter → L = 80 :=
by
  sorry

end length_of_rectangular_plot_l179_179539


namespace woman_wait_time_to_be_caught_l179_179536

theorem woman_wait_time_to_be_caught 
  (man_speed_mph : ℝ) (woman_speed_mph : ℝ) (wait_time_minutes : ℝ) 
  (conversion_factor : ℝ) (distance_apart_miles : ℝ) :
  man_speed_mph = 6 →
  woman_speed_mph = 12 →
  wait_time_minutes = 10 →
  conversion_factor = 1 / 60 →
  distance_apart_miles = (woman_speed_mph * conversion_factor) * wait_time_minutes →
  ∃ minutes_to_catch_up : ℝ, minutes_to_catch_up = distance_apart_miles / (man_speed_mph * conversion_factor) ∧ minutes_to_catch_up = 20 := sorry

end woman_wait_time_to_be_caught_l179_179536


namespace expand_and_simplify_l179_179558

theorem expand_and_simplify (x : ℝ) : 20 * (3 * x - 4) = 60 * x - 80 := 
by
  sorry

end expand_and_simplify_l179_179558


namespace min_length_intersection_l179_179765

theorem min_length_intersection (m n : ℝ) (h_m1 : 0 ≤ m) (h_m2 : m + 7 / 10 ≤ 1) 
                                (h_n1 : 2 / 5 ≤ n) (h_n2 : n ≤ 1) : 
  ∃ (min_length : ℝ), min_length = 1 / 10 :=
by
  sorry

end min_length_intersection_l179_179765


namespace amusement_park_total_cost_l179_179389

def rides_cost_ferris_wheel : ℕ := 5 * 6
def rides_cost_roller_coaster : ℕ := 7 * 4
def rides_cost_merry_go_round : ℕ := 3 * 10
def rides_cost_bumper_cars : ℕ := 4 * 7
def rides_cost_haunted_house : ℕ := 6 * 5
def rides_cost_log_flume : ℕ := 8 * 3

def snacks_cost_ice_cream : ℕ := 8 * 4
def snacks_cost_hot_dog : ℕ := 6 * 5
def snacks_cost_pizza : ℕ := 4 * 3
def snacks_cost_pretzel : ℕ := 5 * 2
def snacks_cost_cotton_candy : ℕ := 3 * 6
def snacks_cost_soda : ℕ := 2 * 7

def total_rides_cost : ℕ := 
  rides_cost_ferris_wheel + 
  rides_cost_roller_coaster + 
  rides_cost_merry_go_round + 
  rides_cost_bumper_cars + 
  rides_cost_haunted_house + 
  rides_cost_log_flume

def total_snacks_cost : ℕ := 
  snacks_cost_ice_cream + 
  snacks_cost_hot_dog + 
  snacks_cost_pizza + 
  snacks_cost_pretzel + 
  snacks_cost_cotton_candy + 
  snacks_cost_soda

def total_cost : ℕ :=
  total_rides_cost + total_snacks_cost

theorem amusement_park_total_cost :
  total_cost = 286 :=
by
  unfold total_cost total_rides_cost total_snacks_cost
  unfold rides_cost_ferris_wheel 
         rides_cost_roller_coaster 
         rides_cost_merry_go_round 
         rides_cost_bumper_cars 
         rides_cost_haunted_house 
         rides_cost_log_flume
         snacks_cost_ice_cream 
         snacks_cost_hot_dog 
         snacks_cost_pizza 
         snacks_cost_pretzel 
         snacks_cost_cotton_candy 
         snacks_cost_soda
  sorry

end amusement_park_total_cost_l179_179389


namespace max_volume_cube_max_volume_parallelepiped_l179_179460

variables {a b c : ℝ}

-- Problem (a): Cube with the maximum volume entirely contained in the tetrahedron
theorem max_volume_cube (h : a > 0 ∧ b > 0 ∧ c > 0) :
  ∃ s : ℝ, s = (a * b * c) / (a * b + b * c + a * c) := sorry

-- Problem (b): Rectangular parallelepiped with the maximum volume entirely contained in the tetrahedron
theorem max_volume_parallelepiped (h : a > 0 ∧ b > 0 ∧ c > 0) :
  ∃ (x y z : ℝ),
  (x = a / 3 ∧ y = b / 3 ∧ z = c / 3) ∧
  (x * y * z = (a * b * c) / 27) := sorry

end max_volume_cube_max_volume_parallelepiped_l179_179460


namespace nina_ants_count_l179_179804

theorem nina_ants_count 
  (spiders : ℕ) 
  (eyes_per_spider : ℕ) 
  (eyes_per_ant : ℕ) 
  (total_eyes : ℕ) 
  (total_spider_eyes : ℕ) 
  (total_ant_eyes : ℕ) 
  (ants : ℕ) 
  (h1 : spiders = 3) 
  (h2 : eyes_per_spider = 8) 
  (h3 : eyes_per_ant = 2) 
  (h4 : total_eyes = 124) 
  (h5 : total_spider_eyes = spiders * eyes_per_spider) 
  (h6 : total_ant_eyes = total_eyes - total_spider_eyes) 
  (h7 : ants = total_ant_eyes / eyes_per_ant) : 
  ants = 50 := by
  sorry

end nina_ants_count_l179_179804


namespace sqrt_product_simplified_l179_179160

theorem sqrt_product_simplified (x : ℝ) (hx : 0 ≤ x) :
  (Real.sqrt (50 * x) * Real.sqrt (18 * x) * Real.sqrt (32 * x)) = 84 * x * Real.sqrt (2 * x) :=
by 
  sorry

end sqrt_product_simplified_l179_179160


namespace ratio_of_numbers_l179_179821

theorem ratio_of_numbers (a b : ℕ) (hHCF : Nat.gcd a b = 4) (hLCM : Nat.lcm a b = 48) : a / b = 3 / 4 :=
by
  sorry

end ratio_of_numbers_l179_179821


namespace equivalence_a_gt_b_and_inv_a_lt_inv_b_l179_179898

variable {a b : ℝ}

theorem equivalence_a_gt_b_and_inv_a_lt_inv_b (h : a * b > 0) : 
  (a > b) ↔ (1 / a < 1 / b) := 
sorry

end equivalence_a_gt_b_and_inv_a_lt_inv_b_l179_179898


namespace max_area_garden_l179_179388

/-- Given a rectangular garden with a total perimeter of 480 feet and one side twice as long as another,
    prove that the maximum area of the garden is 12800 square feet. -/
theorem max_area_garden (l w : ℝ) (h1 : l = 2 * w) (h2 : 2 * l + 2 * w = 480) : l * w = 12800 := 
sorry

end max_area_garden_l179_179388


namespace range_for_a_l179_179367

noncomputable def line_not_in_second_quadrant (a : ℝ) : Prop :=
  ∀ x y : ℝ, (3 * a - 1) * x + (2 - a) * y - 1 = 0 → (x ≥ 0 ∨ y ≥ 0)

theorem range_for_a (a : ℝ) :
  (line_not_in_second_quadrant a) ↔ a ≥ 2 := by
  sorry

end range_for_a_l179_179367


namespace probability_jack_queen_king_l179_179826

theorem probability_jack_queen_king :
  let deck_size := 52
  let jacks := 4
  let queens := 4
  let kings := 4
  let remaining_after_jack := deck_size - 1
  let remaining_after_queen := deck_size - 2
  (jacks / deck_size) * (queens / remaining_after_jack) * (kings / remaining_after_queen) = 8 / 16575 :=
by
  sorry

end probability_jack_queen_king_l179_179826


namespace average_minutes_run_l179_179450

-- Definitions
def third_graders (fi : ℕ) : ℕ := 6 * fi
def fourth_graders (fi : ℕ) : ℕ := 2 * fi
def fifth_graders (fi : ℕ) : ℕ := fi

-- Number of minutes run by each grade
def third_graders_minutes : ℕ := 10
def fourth_graders_minutes : ℕ := 18
def fifth_graders_minutes : ℕ := 8

-- Main theorem
theorem average_minutes_run 
  (fi : ℕ) 
  (t := third_graders fi) 
  (fr := fourth_graders fi) 
  (f := fifth_graders fi) 
  (minutes_total := 10 * t + 18 * fr + 8 * f) 
  (students_total := t + fr + f) :
  (students_total > 0) →
  (minutes_total : ℚ) / students_total = 104 / 9 :=
by
  sorry

end average_minutes_run_l179_179450


namespace lab_techs_share_l179_179659

theorem lab_techs_share (u c t : ℕ) 
  (h1 : c = 6 * u)
  (h2 : t = u / 2)
  (h3 : u = 12) : 
  (c + u) / t = 14 := 
by 
  sorry

end lab_techs_share_l179_179659


namespace mathematician_daily_questions_l179_179242

theorem mathematician_daily_questions :
  (518 + 476) / 7 = 142 := by
  sorry

end mathematician_daily_questions_l179_179242


namespace set_intersection_complement_l179_179907

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set ℕ := {2, 5, 8}
def B : Set ℕ := {1, 3, 5, 7}

theorem set_intersection_complement :
  ((U \ A) ∩ B) = {1, 3, 7} :=
by
  sorry

end set_intersection_complement_l179_179907


namespace simplify_expression_l179_179628

theorem simplify_expression (x y : ℝ) : 3 * x + 2 * y + 4 * x + 5 * y + 7 = 7 * x + 7 * y + 7 := 
by sorry

end simplify_expression_l179_179628


namespace stream_speed_l179_179679

theorem stream_speed :
  ∀ (v : ℝ),
  (12 - v) / (12 + v) = 1 / 2 →
  v = 4 :=
by
  sorry

end stream_speed_l179_179679


namespace chess_tournament_games_l179_179908

theorem chess_tournament_games (n : ℕ) (h : n = 15) :
  nat.choose n 2 = 105 := by
  rw h
  exact nat.choose_self_sub 15 2

end chess_tournament_games_l179_179908


namespace cost_per_gumball_l179_179938

theorem cost_per_gumball (total_money : ℕ) (num_gumballs : ℕ) (cost_each : ℕ) 
  (h1 : total_money = 32) (h2 : num_gumballs = 4) : cost_each = 8 :=
by
  sorry -- Proof omitted

end cost_per_gumball_l179_179938


namespace neg_existence_of_ge_impl_universal_lt_l179_179807

theorem neg_existence_of_ge_impl_universal_lt : (¬ ∃ x : ℕ, x^2 ≥ x) ↔ ∀ x : ℕ, x^2 < x := 
sorry

end neg_existence_of_ge_impl_universal_lt_l179_179807


namespace solution_10_digit_divisible_by_72_l179_179865

def attach_digits_to_divisible_72 : Prop :=
  ∃ (a d : ℕ), (a < 10) ∧ (d < 10) ∧ a * 10^9 + 20222023 * 10 + d = 3202220232 ∧ (3202220232 % 72 = 0)

theorem solution_10_digit_divisible_by_72 : attach_digits_to_divisible_72 :=
  sorry

end solution_10_digit_divisible_by_72_l179_179865


namespace line_parabola_intersection_l179_179578

theorem line_parabola_intersection (k : ℝ) (M A B : ℝ × ℝ) (h1 : ¬ k = 0) 
  (h2 : M = (2, 0))
  (h3 : ∃ x y, (x = k * y + 2 ∧ (x, y) ∈ {p : ℝ × ℝ | p.2^2 = 4 * p.1} ∧ (p = A ∨ p = B))) 
  : 1 / |dist M A|^2 + 1 / |dist M B|^2 = 1 / 4 := 
by 
  sorry

end line_parabola_intersection_l179_179578


namespace possible_slopes_of_line_intersecting_ellipse_l179_179532

theorem possible_slopes_of_line_intersecting_ellipse (m : ℝ) :
  (∃ x y : ℝ, y = m * x + 3 ∧ 4 * x^2 + 25 * y^2 = 100) ↔ m ∈ (Set.Iic (-2/5) ∪ Set.Ici (2/5)) :=
by
  sorry

end possible_slopes_of_line_intersecting_ellipse_l179_179532


namespace part1_part2_l179_179575

-- Definitions for sets A and B
def A : Set ℝ := {x : ℝ | -3 < x ∧ x < 2}
def B : Set ℝ := {x : ℝ | x < -5 ∨ x > 1}

-- Prove (1): A ∪ B
theorem part1 : A ∪ B = {x : ℝ | x < -5 ∨ x > -3} :=
by
  sorry

-- Prove (2): A ∩ (ℝ \ B)
theorem part2 : A ∩ (Set.compl B) = {x : ℝ | -3 < x ∧ x ≤ 1} :=
by
  sorry

end part1_part2_l179_179575


namespace simplify_and_rationalize_l179_179022

theorem simplify_and_rationalize : 
  (1 / (2 + (1 / (Real.sqrt 5 + 2)))) = (Real.sqrt 5 / 5) :=
by
  sorry

end simplify_and_rationalize_l179_179022


namespace roots_of_equation_l179_179565

theorem roots_of_equation:
  ∀ x : ℝ, (x - 2) * (x - 3) = x - 2 → x = 2 ∨ x = 4 := by
  sorry

end roots_of_equation_l179_179565


namespace largest_base5_three_digits_is_124_l179_179206

noncomputable def largest_base5_three_digits_to_base10 : ℕ :=
  4 * 5^2 + 4 * 5^1 + 4 * 5^0

theorem largest_base5_three_digits_is_124 :
  largest_base5_three_digits_to_base10 = 124 :=
by
  -- calculating 4 * 5^2 + 4 * 5^1 + 4 * 5^0 = 124
  sorry

end largest_base5_three_digits_is_124_l179_179206


namespace lulu_cash_left_l179_179933

theorem lulu_cash_left :
  ∀ (initial money spentIceCream spentTshirt deposited finalCash: ℝ),
    initial = 65 →
    spentIceCream = 5 →
    spentTshirt = 0.5 * (initial - spentIceCream) →
    deposited = (initial - spentIceCream - spentTshirt) / 5 →
    finalCash = initial - spentIceCream - spentTshirt - deposited →
    finalCash = 24 :=
by
  intros
  sorry

end lulu_cash_left_l179_179933


namespace evaluate_expression_l179_179264

theorem evaluate_expression (b : ℕ) (hb : b = 2) : (b^3 * b^4) - b^2 = 124 :=
by
  -- leave the proof empty with a placeholder
  sorry

end evaluate_expression_l179_179264


namespace term_is_18_minimum_value_l179_179133

-- Define the sequence a_n
def a_n (n : ℕ) : ℤ := n^2 - 5 * n + 4

-- Prove that a_n = 18 implies n = 7
theorem term_is_18 (n : ℕ) (h : a_n n = 18) : n = 7 := 
by 
  sorry

-- Prove that the minimum value of a_n is -2 and it occurs at n = 2 or n = 3
theorem minimum_value (n : ℕ) : n = 2 ∨ n = 3 ∧ a_n n = -2 :=
by 
  sorry

end term_is_18_minimum_value_l179_179133


namespace simplify_and_rationalize_correct_l179_179025

noncomputable def simplify_and_rationalize : ℚ :=
  1 / (2 + 1 / (Real.sqrt 5 + 2))

theorem simplify_and_rationalize_correct : simplify_and_rationalize = (Real.sqrt 5) / 5 := by
  sorry

end simplify_and_rationalize_correct_l179_179025


namespace preferred_apples_percentage_l179_179951

theorem preferred_apples_percentage (A B C O G : ℕ) (total freq_apples : ℕ)
  (hA : A = 70) (hB : B = 50) (hC: C = 30) (hO: O = 50) (hG: G = 40)
  (htotal : total = A + B + C + O + G)
  (hfa : freq_apples = A) :
  (freq_apples / total : ℚ) * 100 = 29 :=
by sorry

end preferred_apples_percentage_l179_179951


namespace packs_to_purchase_l179_179152

theorem packs_to_purchase {n m k : ℕ} (h : 8 * n + 15 * m + 30 * k = 135) : n + m + k = 5 :=
sorry

end packs_to_purchase_l179_179152


namespace sum_of_money_l179_179233

theorem sum_of_money (x : ℝ)
  (hC : 0.50 * x = 64)
  (hB : ∀ x, B_shares = 0.75 * x)
  (hD : ∀ x, D_shares = 0.25 * x) :
  let total_sum := x + 0.75 * x + 0.50 * x + 0.25 * x
  total_sum = 320 :=
by
  sorry

end sum_of_money_l179_179233


namespace twenty_four_x_eq_a_cubed_t_l179_179277

-- Define conditions
variables {x : ℝ} {a t : ℝ}
axiom h1 : 2^x = a
axiom h2 : 3^x = t

-- State the theorem
theorem twenty_four_x_eq_a_cubed_t : 24^x = a^3 * t := 
by sorry

end twenty_four_x_eq_a_cubed_t_l179_179277


namespace Mike_height_l179_179937

theorem Mike_height (h_mark: 5 * 12 + 3 = 63) (h_mark_mike:  63 + 10 = 73) (h_foot: 12 = 12)
: 73 / 12 = 6 ∧ 73 % 12 = 1 := 
sorry

end Mike_height_l179_179937


namespace probability_of_selected_member_l179_179910

section Probability

variables {N : ℕ} -- Total number of members in the group

-- Conditions
-- Probabilities of selecting individuals by gender
def P_woman : ℝ := 0.70
def P_man : ℝ := 0.20
def P_non_binary : ℝ := 0.10

-- Conditional probabilities of occupations given gender
def P_engineer_given_woman : ℝ := 0.20
def P_doctor_given_man : ℝ := 0.20
def P_translator_given_non_binary : ℝ := 0.20

-- The main proof statement
theorem probability_of_selected_member :
  (P_woman * P_engineer_given_woman) + (P_man * P_doctor_given_man) + (P_non_binary * P_translator_given_non_binary) = 0.20 :=
by
  sorry

end Probability

end probability_of_selected_member_l179_179910


namespace initial_value_calculation_l179_179747

theorem initial_value_calculation (P : ℝ) (h1 : ∀ n : ℕ, 0 ≤ n →
                                (P:ℝ) * (1 + 1/8) ^ n = 78468.75 → n = 2) :
  P = 61952 :=
sorry

end initial_value_calculation_l179_179747


namespace product_has_no_linear_term_l179_179306

theorem product_has_no_linear_term (m : ℝ) (h : ((x : ℝ) → (x - m) * (x - 3) = x^2 + 3 * m)) : m = -3 := 
by
  sorry

end product_has_no_linear_term_l179_179306


namespace water_evaporation_problem_l179_179848

theorem water_evaporation_problem 
  (W : ℝ) 
  (evaporation_rate : ℝ := 0.01) 
  (evaporation_days : ℝ := 20) 
  (total_evaporation : ℝ := evaporation_rate * evaporation_days) 
  (evaporation_percentage : ℝ := 0.02) 
  (evaporation_amount : ℝ := evaporation_percentage * W) :
  evaporation_amount = total_evaporation → W = 10 :=
by
  sorry

end water_evaporation_problem_l179_179848


namespace hyperbola_asymptote_ratio_l179_179289

theorem hyperbola_asymptote_ratio (a b : ℝ) (h : a > b) 
  (hyp : ∀ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1) → (|x / y| = |a / b + 3 + 2 * real.sqrt 2)) :
  a / b = 3 + 2 * real.sqrt 2 :=
sorry

end hyperbola_asymptote_ratio_l179_179289


namespace correct_choice_l179_179444

theorem correct_choice (x y m : ℝ) (h1 : x > y) (h2 : m > 0) : x - y > 0 := by
  sorry

end correct_choice_l179_179444


namespace find_deeper_depth_l179_179081

noncomputable def swimming_pool_depth_proof 
  (width : ℝ) (length : ℝ) (shallow_depth : ℝ) (volume : ℝ) : Prop :=
  volume = (1 / 2) * (shallow_depth + 4) * width * length

theorem find_deeper_depth
  (h : width = 9)
  (l : length = 12)
  (a : shallow_depth = 1)
  (V : volume = 270) :
  swimming_pool_depth_proof 9 12 1 270 := by
  sorry

end find_deeper_depth_l179_179081


namespace div_by_17_l179_179329

theorem div_by_17 (n : ℕ) (h : ¬ 17 ∣ n) : 17 ∣ (n^8 + 1) ∨ 17 ∣ (n^8 - 1) := 
by sorry

end div_by_17_l179_179329


namespace area_of_WIN_sector_correct_l179_179236

-- Define variables and constants
def radius : ℝ := 8
def probability_of_winning : ℝ := 1 / 4

-- Define the area of the circle
def area_of_circle (r : ℝ) := real.pi * r ^ 2

-- Define the area of the WIN sector given the area of the circle and the probability of winning
def area_of_WIN_sector (area_circle : ℝ) (prob_win : ℝ) := prob_win * area_circle

-- Theorem that the area of the WIN sector is 16π square centimeters
theorem area_of_WIN_sector_correct :
  area_of_WIN_sector (area_of_circle radius) probability_of_winning = 16 * real.pi :=
by 
-- Proof omitted
sorry

end area_of_WIN_sector_correct_l179_179236


namespace number_of_solutions_mod_n_l179_179883

open Nat

theorem number_of_solutions_mod_n (n : ℕ) (h : n ≥ 2) :
  ∃ xs : Finset ℕ, (∀ x ∈ xs, x^2 % n = x % n) ∧ xs.card = nat.divisors n :=
sorry

end number_of_solutions_mod_n_l179_179883


namespace problem1_problem2_l179_179434

-- Definitions of sets A and B
def setA : Set ℝ := { x | x^2 - 8 * x + 15 = 0 }
def setB (a : ℝ) : Set ℝ := { x | a * x - 1 = 0 }

-- Problem 1: If a = 1/5, B is a subset of A.
theorem problem1 : setB (1 / 5) ⊆ setA := sorry

-- Problem 2: If A ∩ B = B, then C = {0, 1/3, 1/5}.
def setC : Set ℝ := { a | a = 0 ∨ a = 1 / 3 ∨ a = 1 / 5 }

theorem problem2 (a : ℝ) : (setA ∩ setB a = setB a) ↔ (a ∈ setC) := sorry

end problem1_problem2_l179_179434


namespace rectangle_width_l179_179355

theorem rectangle_width (w l A : ℕ) 
  (h1 : l = 3 * w)
  (h2 : A = l * w)
  (h3 : A = 108) : 
  w = 6 := 
sorry

end rectangle_width_l179_179355


namespace find_g2_l179_179901

variable (g : ℝ → ℝ)

theorem find_g2 (h : ∀ x : ℝ, g (3 * x - 7) = 5 * x + 11) : g 2 = 26 := by
  sorry

end find_g2_l179_179901


namespace gilbert_herb_plants_count_l179_179276

variable (initial_basil : Nat) (initial_parsley : Nat) (initial_mint : Nat)
variable (dropped_basil_seeds : Nat) (rabbit_ate_all_mint : Bool)

def total_initial_plants (initial_basil initial_parsley initial_mint : Nat) : Nat :=
  initial_basil + initial_parsley + initial_mint

def total_plants_after_dropping_seeds (initial_basil initial_parsley initial_mint dropped_basil_seeds : Nat) : Nat :=
  total_initial_plants initial_basil initial_parsley initial_mint + dropped_basil_seeds

def total_plants_after_rabbit (initial_basil initial_parsley initial_mint dropped_basil_seeds : Nat) (rabbit_ate_all_mint : Bool) : Nat :=
  if rabbit_ate_all_mint then 
    total_plants_after_dropping_seeds initial_basil initial_parsley initial_mint dropped_basil_seeds - initial_mint 
  else 
    total_plants_after_dropping_seeds initial_basil initial_parsley initial_mint dropped_basil_seeds

theorem gilbert_herb_plants_count
  (h1 : initial_basil = 3)
  (h2 : initial_parsley = 1)
  (h3 : initial_mint = 2)
  (h4 : dropped_basil_seeds = 1)
  (h5 : rabbit_ate_all_mint = true) :
  total_plants_after_rabbit initial_basil initial_parsley initial_mint dropped_basil_seeds rabbit_ate_all_mint = 5 := by
  sorry

end gilbert_herb_plants_count_l179_179276


namespace triangle_angle_l179_179283

-- Definitions of the conditions and theorem
variables {a b c : ℝ}
variables {A B C : ℝ}

theorem triangle_angle (h : b^2 + c^2 - a^2 = bc)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hA : 0 < A) (hA_max : A < π) :
  A = π / 3 :=
by
  sorry

end triangle_angle_l179_179283


namespace mark_total_flowers_l179_179337

theorem mark_total_flowers (yellow purple green total : ℕ) 
  (hyellow : yellow = 10)
  (hpurple : purple = yellow + (yellow * 80 / 100))
  (hgreen : green = (yellow + purple) * 25 / 100)
  (htotal : total = yellow + purple + green) : 
  total = 35 :=
by
  sorry

end mark_total_flowers_l179_179337


namespace roots_positive_range_no_negative_roots_opposite_signs_range_l179_179568

theorem roots_positive_range (m : ℝ) : (8 * x^2 - m * x + (m - 6) = 0) → (6 < m ∧ m ≤ 8 ∨ m ≥ 24) :=
sorry

theorem no_negative_roots (m : ℝ) : (8 * x^2 - m * x + (m - 6) = 0) → ¬ (∀ α β, (α < 0 ∧ β < 0)) :=
sorry

theorem opposite_signs_range (m : ℝ) : (8 * x^2 - m * x + (m - 6) = 0) → m < 6 :=
sorry

end roots_positive_range_no_negative_roots_opposite_signs_range_l179_179568


namespace reciprocal_of_neg_one_sixth_is_neg_six_l179_179823

theorem reciprocal_of_neg_one_sixth_is_neg_six : 1 / (- (1 / 6)) = -6 :=
by sorry

end reciprocal_of_neg_one_sixth_is_neg_six_l179_179823


namespace rolls_sold_to_grandmother_l179_179419

theorem rolls_sold_to_grandmother (t u n s g : ℕ) 
  (h1 : t = 45)
  (h2 : u = 10)
  (h3 : n = 6)
  (h4 : s = 28)
  (total_sold : t - s = g + u + n) : 
  g = 1 := 
  sorry

end rolls_sold_to_grandmother_l179_179419


namespace multiple_of_6_is_multiple_of_2_and_3_l179_179720

theorem multiple_of_6_is_multiple_of_2_and_3 (n : ℕ) :
  (∃ k : ℕ, n = 6 * k) → (∃ m1 : ℕ, n = 2 * m1) ∧ (∃ m2 : ℕ, n = 3 * m2) := by
  sorry

end multiple_of_6_is_multiple_of_2_and_3_l179_179720


namespace NineChaptersProblem_l179_179912

-- Conditions: Assign the given conditions to variables
variables (x y : Int)
def condition1 : Prop := y = 8 * x - 3
def condition2 : Prop := y = 7 * x + 4

-- Proof problem: Prove that the system of equations is consistent with the given conditions
theorem NineChaptersProblem : condition1 x y ∧ condition2 x y := sorry

end NineChaptersProblem_l179_179912


namespace negative_x_is_positive_l179_179270

theorem negative_x_is_positive (x : ℝ) (hx : x < 0) : -x > 0 :=
sorry

end negative_x_is_positive_l179_179270


namespace employee_hourly_pay_l179_179989

-- Definitions based on conditions
def initial_employees := 500
def daily_hours := 10
def weekly_days := 5
def monthly_weeks := 4
def additional_employees := 200
def total_payment := 1680000
def total_employees := initial_employees + additional_employees
def monthly_hours_per_employee := daily_hours * weekly_days * monthly_weeks
def total_monthly_hours := total_employees * monthly_hours_per_employee

-- Lean 4 statement proving the hourly pay per employee
theorem employee_hourly_pay : total_payment / total_monthly_hours = 12 := by sorry

end employee_hourly_pay_l179_179989


namespace minimum_loaves_arithmetic_sequence_l179_179493

theorem minimum_loaves_arithmetic_sequence :
  ∃ a d : ℚ, 
    (5 * a = 100) ∧ (3 * a + 3 * d = 7 * (2 * a - 3 * d)) ∧ (a - 2 * d = 5/3) :=
sorry

end minimum_loaves_arithmetic_sequence_l179_179493


namespace mona_unique_players_l179_179007

-- Define the conditions
def groups (mona: String) : ℕ := 9
def players_per_group : ℕ := 4
def repeat_players_group1 : ℕ := 2
def repeat_players_group2 : ℕ := 1

-- Statement of the proof problem
theorem mona_unique_players
  (total_groups : ℕ := groups "Mona")
  (players_each_group : ℕ := players_per_group)
  (repeats_group1 : ℕ := repeat_players_group1)
  (repeats_group2 : ℕ := repeat_players_group2) :
  (total_groups * players_each_group) - (repeats_group1 + repeats_group2) = 33 := by
  sorry

end mona_unique_players_l179_179007


namespace problem_l179_179894

theorem problem (x : ℕ) (h1 : x > 0) (h2 : ∃ k : ℕ, 7 - x = k^2) : x = 3 ∨ x = 6 ∨ x = 7 :=
by
  sorry

end problem_l179_179894


namespace maxCubeSideLength_correct_maxRectParallelepipedDims_correct_l179_179468

noncomputable def maxCubeSideLength (a b c : ℝ) : ℝ :=
  a * b * c / (a * b + b * c + a * c)

noncomputable def maxRectParallelepipedDims (a b c : ℝ) : ℝ × ℝ × ℝ :=
  (a / 3, b / 3, c / 3)

theorem maxCubeSideLength_correct (a b c : ℝ) :
  a > 0 → b > 0 → c > 0 →
  maxCubeSideLength a b c = a * b * c / (a * b + b * c + a * c) :=
sorry

theorem maxRectParallelepipedDims_correct (a b c : ℝ) :
  a > 0 → b > 0 → c > 0 →
  maxRectParallelepipedDims a b c = (a / 3, b / 3, c / 3) :=
sorry

end maxCubeSideLength_correct_maxRectParallelepipedDims_correct_l179_179468


namespace pablo_days_to_complete_all_puzzles_l179_179013

def average_pieces_per_hour : ℕ := 100
def puzzles_300_pieces : ℕ := 8
def puzzles_500_pieces : ℕ := 5
def pieces_per_300_puzzle : ℕ := 300
def pieces_per_500_puzzle : ℕ := 500
def max_hours_per_day : ℕ := 7

theorem pablo_days_to_complete_all_puzzles :
  let total_pieces := (puzzles_300_pieces * pieces_per_300_puzzle) + (puzzles_500_pieces * pieces_per_500_puzzle)
  let pieces_per_day := max_hours_per_day * average_pieces_per_hour
  let days_to_complete := total_pieces / pieces_per_day
  days_to_complete = 7 :=
by
  sorry

end pablo_days_to_complete_all_puzzles_l179_179013


namespace sum_of_coefficients_is_neg40_l179_179752

noncomputable def p (x : ℝ) : ℝ := 3 * (x^8 - x^5 + 2 * x^3 - 6) - 5 * (x^4 + 3 * x^2) + 2 * (x^6 - 5)

theorem sum_of_coefficients_is_neg40 : p 1 = -40 := by
  sorry

end sum_of_coefficients_is_neg40_l179_179752


namespace arithmetic_sequence_proof_l179_179453

noncomputable def a (n : ℕ) (a₁ d : ℝ) : ℝ := a₁ + (n - 1) * d

theorem arithmetic_sequence_proof
  (a₁ d : ℝ)
  (h : a 4 a₁ d + a 6 a₁ d + a 8 a₁ d + a 10 a₁ d + a 12 a₁ d = 120) :
  a 7 a₁ d - (1 / 3) * a 5 a₁ d = 16 :=
by
  sorry

end arithmetic_sequence_proof_l179_179453


namespace lines_perpendicular_l179_179759

-- Define the lines l1 and l2
def line1 (m x y : ℝ) := m * x + y - 1 = 0
def line2 (m x y : ℝ) := x + (m - 1) * y + 2 = 0

-- State the problem: Find the value of m such that the lines l1 and l2 are perpendicular.
theorem lines_perpendicular (m : ℝ) (h₁ : line1 m x y) (h₂ : line2 m x y) : m = 1/2 := 
sorry

end lines_perpendicular_l179_179759


namespace infinite_set_divisor_l179_179323

open Set

noncomputable def exists_divisor (A : Set ℕ) : Prop :=
  ∃ (d : ℕ), d > 1 ∧ ∀ (a : ℕ), a ∈ A → d ∣ a

theorem infinite_set_divisor (A : Set ℕ) (hA1 : ∀ (b : Finset ℕ), (↑b ⊆ A) → ∃ (d : ℕ), d > 1 ∧ ∀ (a : ℕ), a ∈ b → d ∣ a) :
  exists_divisor A :=
sorry

end infinite_set_divisor_l179_179323


namespace triangle_isosceles_l179_179128

theorem triangle_isosceles (a b c : ℝ) (C : ℝ) (h : a = 2 * b * Real.cos C) :
  b = c → IsoscelesTriangle := 
by
  sorry

end triangle_isosceles_l179_179128


namespace gallery_pieces_total_l179_179861

noncomputable def TotalArtGalleryPieces (A : ℕ) : Prop :=
  let D := (1 : ℚ) / 3 * A
  let N := A - D
  let notDisplayedSculptures := (2 : ℚ) / 3 * N
  let totalSculpturesNotDisplayed := 800
  (4 : ℚ) / 9 * A = 800

theorem gallery_pieces_total (A : ℕ) (h : (TotalArtGalleryPieces A)) : A = 1800 :=
by sorry

end gallery_pieces_total_l179_179861


namespace sheila_paintings_l179_179083

theorem sheila_paintings (a b : ℕ) (h1 : a = 9) (h2 : b = 9) : a + b = 18 :=
by
  sorry

end sheila_paintings_l179_179083


namespace find_a_b_l179_179288

noncomputable def f (a b x: ℝ) : ℝ := x / (a * x + b)

theorem find_a_b (a b : ℝ) (h₁ : a ≠ 0) (h₂ : f a b (-4) = 4) (h₃ : ∀ x, f a b x = f b a x) :
  a + b = 3 / 2 :=
sorry

end find_a_b_l179_179288


namespace laborer_monthly_income_l179_179381

theorem laborer_monthly_income
  (I : ℕ)
  (D : ℕ)
  (h1 : 6 * I + D = 510)
  (h2 : 4 * I - D = 270) : I = 78 := by
  sorry

end laborer_monthly_income_l179_179381


namespace arun_brother_weight_upper_limit_l179_179729

theorem arun_brother_weight_upper_limit (w : ℝ) (X : ℝ) 
  (h1 : 61 < w ∧ w < 72)
  (h2 : 60 < w ∧ w < X)
  (h3 : w ≤ 64)
  (h4 : ((62 + 63 + 64) / 3) = 63) :
  X = 64 :=
by
  sorry

end arun_brother_weight_upper_limit_l179_179729


namespace find_a2_plus_a8_l179_179314

variable {a_n : ℕ → ℤ}  -- Assume the sequence is indexed by natural numbers and maps to integers

-- Define the condition in the problem
def seq_property (a_n : ℕ → ℤ) := a_n 3 + a_n 4 + a_n 5 + a_n 6 + a_n 7 = 25

-- Statement to prove
theorem find_a2_plus_a8 (h : seq_property a_n) : a_n 2 + a_n 8 = 10 :=
sorry

end find_a2_plus_a8_l179_179314


namespace max_cube_side_length_max_parallelepiped_dimensions_l179_179461

theorem max_cube_side_length (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) : 
    ∃ (a0 : ℝ), a0 = a * b * c / (a * b + b * c + a * c) := 
sorry

theorem max_parallelepiped_dimensions (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) : 
    ∃ (x y z : ℝ), (x = a / 3) ∧ (y = b / 3) ∧ (z = c / 3) :=
sorry

end max_cube_side_length_max_parallelepiped_dimensions_l179_179461


namespace median_number_of_children_is_three_l179_179038

/-- Define the context of the problem with total number of families. -/
def total_families : Nat := 15

/-- Prove that given the conditions, the median number of children is 3. -/
theorem median_number_of_children_is_three 
  (h : total_families = 15) : 
  ∃ median : Nat, median = 3 :=
by
  sorry

end median_number_of_children_is_three_l179_179038


namespace Peter_can_guarantee_victory_l179_179489

structure Board :=
  (size : ℕ)
  (cells : Fin size × Fin size → Option Color)

inductive Player
  | Peter
  | Victor
deriving DecidableEq

inductive Color
  | Red
  | Green
  | White
deriving DecidableEq

structure Move :=
  (player : Player)
  (rectangle : Fin 2 × Fin 2)
  (position : Fin 7 × Fin 7)

def isValidMove (board : Board) (move : Move) : Prop := sorry

def applyMove (board : Board) (move : Move) : Board := sorry

def allCellsColored (board : Board) : Prop := sorry

theorem Peter_can_guarantee_victory :
  ∀ (initialBoard : Board),
    (∀ (move : Move), move.player = Player.Victor → isValidMove initialBoard move) →
    Player.Peter = Player.Peter →
    (∃ finalBoard : Board,
       allCellsColored finalBoard ∧ 
       ¬ (∃ (move : Move), move.player = Player.Victor ∧ isValidMove finalBoard move)) :=
sorry

end Peter_can_guarantee_victory_l179_179489


namespace lcm_of_23_46_827_l179_179380

theorem lcm_of_23_46_827 : Nat.lcm (Nat.lcm 23 46) 827 = 38042 :=
by
  sorry

end lcm_of_23_46_827_l179_179380


namespace fixed_monthly_charge_for_100_GB_l179_179869

theorem fixed_monthly_charge_for_100_GB
  (fixed_charge M : ℝ)
  (extra_charge_per_GB : ℝ := 0.25)
  (total_bill : ℝ := 65)
  (GB_over : ℝ := 80)
  (extra_charge : ℝ := GB_over * extra_charge_per_GB) :
  total_bill = M + extra_charge → M = 45 :=
by sorry

end fixed_monthly_charge_for_100_GB_l179_179869


namespace haley_deleted_pictures_l179_179061

variable (zoo_pictures : ℕ) (museum_pictures : ℕ) (remaining_pictures : ℕ) (deleted_pictures : ℕ)

theorem haley_deleted_pictures :
  zoo_pictures = 50 → museum_pictures = 8 → remaining_pictures = 20 →
  deleted_pictures = zoo_pictures + museum_pictures - remaining_pictures →
  deleted_pictures = 38 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  simp at h4
  exact h4

end haley_deleted_pictures_l179_179061


namespace NineChaptersProblem_l179_179911

-- Conditions: Assign the given conditions to variables
variables (x y : Int)
def condition1 : Prop := y = 8 * x - 3
def condition2 : Prop := y = 7 * x + 4

-- Proof problem: Prove that the system of equations is consistent with the given conditions
theorem NineChaptersProblem : condition1 x y ∧ condition2 x y := sorry

end NineChaptersProblem_l179_179911


namespace find_angle4_l179_179429

theorem find_angle4 (angle1 angle2 angle3 angle4 : ℝ)
                    (h1 : angle1 + angle2 = 180)
                    (h2 : angle3 = 2 * angle4)
                    (h3 : angle1 = 50)
                    (h4 : angle3 + angle4 = 130) : 
                    angle4 = 130 / 3 := by 
    sorry

end find_angle4_l179_179429


namespace intersection_of_sets_l179_179791

def is_angle_in_first_quadrant (α : ℝ) : Prop :=
  ∃ k : ℤ, k * 360 < α ∧ α < k * 360 + 90

def is_acute_angle (α : ℝ) : Prop :=
  α < 90

theorem intersection_of_sets (α : ℝ) :
  (is_acute_angle α ∧ is_angle_in_first_quadrant α) ↔
  (∃ k : ℤ, k ≤ 0 ∧ k * 360 < α ∧ α < k * 360 + 90) := 
sorry

end intersection_of_sets_l179_179791


namespace lulu_cash_left_l179_179930

-- Define the initial amount
def initial_amount : ℕ := 65

-- Define the amount spent on ice cream
def spent_on_ice_cream : ℕ := 5

-- Define the amount spent on a t-shirt
def spent_on_tshirt (remaining_after_ice_cream : ℕ) : ℕ := remaining_after_ice_cream / 2

-- Define the amount deposited in the bank
def deposited_in_bank (remaining_after_tshirt : ℕ) : ℕ := remaining_after_tshirt / 5

-- Define the remaining cash after all transactions
def remaining_cash (initial : ℕ) (spent_ice_cream : ℕ) (spent_tshirt: ℕ) (deposited: ℕ) :ℕ :=
  initial - spent_ice_cream - spent_tshirt - deposited

-- Theorem statement to prove
theorem lulu_cash_left : remaining_cash initial_amount spent_on_ice_cream (spent_on_tshirt (initial_amount - spent_on_ice_cream)) 
(deposited_in_bank ((initial_amount - spent_on_ice_cream) - (spent_on_tshirt (initial_amount - spent_on_ice_cream)))) = 24 :=
by
  sorry

end lulu_cash_left_l179_179930


namespace max_value_of_z_l179_179770

open Real

theorem max_value_of_z (x y : ℝ) (h₁ : x + y ≥ 1) (h₂ : 2 * x - y ≤ 0) (h₃ : 3 * x - 2 * y + 2 ≥ 0) : 
  ∃ x y, 3 * x - y = 2 :=
sorry

end max_value_of_z_l179_179770


namespace horner_value_at_neg4_l179_179374

noncomputable def f (x : ℝ) : ℝ := 10 + 25 * x - 8 * x^2 + x^4 + 6 * x^5 + 2 * x^6

def horner_rewrite (x : ℝ) : ℝ := (((((2 * x + 6) * x + 1) * x + 0) * x - 8) * x + 25) * x + 10

theorem horner_value_at_neg4 : horner_rewrite (-4) = -36 :=
by sorry

end horner_value_at_neg4_l179_179374


namespace range_of_m_l179_179763

noncomputable def f (x : ℝ) : ℝ := 2 * x - Real.exp (2 * x)
noncomputable def g (m x : ℝ) : ℝ := m * x + 1

def exists_x0 (x1 : ℝ) (m : ℝ) : Prop :=
  ∃ (x0 : ℝ), -1 ≤ x0 ∧ x0 ≤ 1 ∧ g m x0 = f x1

theorem range_of_m (m : ℝ) (cond : ∀ (x1 : ℝ), -1 ≤ x1 → x1 ≤ 1 → exists_x0 x1 m) :
  m ∈ Set.Iic (1 - Real.exp 2) ∨ m ∈ Set.Ici (Real.exp 2 - 1) :=
sorry

end range_of_m_l179_179763


namespace smallest_of_5_consecutive_natural_numbers_sum_100_l179_179043

theorem smallest_of_5_consecutive_natural_numbers_sum_100
  (n : ℕ)
  (h : n + (n + 1) + (n + 2) + (n + 3) + (n + 4) = 100) :
  n = 18 := sorry

end smallest_of_5_consecutive_natural_numbers_sum_100_l179_179043


namespace pow_100_mod_18_l179_179984

theorem pow_100_mod_18 : (5 ^ 100) % 18 = 13 := by
  -- Define the conditions
  have h1 : (5 ^ 1) % 18 = 5 := by norm_num
  have h2 : (5 ^ 2) % 18 = 7 := by norm_num
  have h3 : (5 ^ 3) % 18 = 17 := by norm_num
  have h4 : (5 ^ 4) % 18 = 13 := by norm_num
  have h5 : (5 ^ 5) % 18 = 11 := by norm_num
  have h6 : (5 ^ 6) % 18 = 1 := by norm_num
  
  -- The required theorem is based on the conditions mentioned
  sorry

end pow_100_mod_18_l179_179984


namespace probability_of_bug9_is_zero_l179_179915

-- Definitions based on conditions provided
def vowels : List Char := ['A', 'E', 'I', 'O', 'U']
def non_vowels : List Char := ['B', 'C', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'X', 'Y', 'Z']
def digits_or_vowels : List Char := ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'E', 'I', 'O', 'U']

-- Defining the number of choices for each position
def first_symbol_choices : Nat := 5
def second_symbol_choices : Nat := 21
def third_symbol_choices : Nat := 20
def fourth_symbol_choices : Nat := 15

-- Total number of possible license plates
def total_plates : Nat := first_symbol_choices * second_symbol_choices * third_symbol_choices * fourth_symbol_choices

-- Probability calculation for the specific license plate "BUG9"
def probability_bug9 : Nat := 0

theorem probability_of_bug9_is_zero : probability_bug9 = 0 := by sorry

end probability_of_bug9_is_zero_l179_179915


namespace lisa_interest_l179_179636

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r)^n

theorem lisa_interest (hP : ℝ := 1500) (hr : ℝ := 0.02) (hn : ℕ := 10) :
  (compound_interest hP hr hn - hP) = 328.49 :=
by
  sorry

end lisa_interest_l179_179636


namespace ratio_WX_XY_l179_179040

theorem ratio_WX_XY (p q : ℝ) (h : 3 * p = 4 * q) : (4 * q) / (3 * p) = 12 / 7 := by
  sorry

end ratio_WX_XY_l179_179040


namespace total_amount_correct_l179_179385

namespace ProofExample

def initial_amount : ℝ := 3

def additional_amount : ℝ := 6.8

def total_amount (initial : ℝ) (additional : ℝ) : ℝ := initial + additional

theorem total_amount_correct : total_amount initial_amount additional_amount = 9.8 :=
by
  sorry

end ProofExample

end total_amount_correct_l179_179385


namespace non_adjacent_ball_arrangements_l179_179863

-- Statement only, proof is omitted
theorem non_adjacent_ball_arrangements :
  let n := (3: ℕ) -- Number of identical yellow balls
  let white_red_positions := (4: ℕ) -- Positions around the yellow unit
  let choose_positions := Nat.choose white_red_positions 2
  let arrange_balls := (2: ℕ) -- Ways to arrange the white and red balls in the chosen positions
  let total_arrangements := choose_positions * arrange_balls
  total_arrangements = 12 := 
by
  sorry

end non_adjacent_ball_arrangements_l179_179863


namespace PB_length_l179_179132

/-- In a square ABCD with area 1989 cm², with the center O, and
a point P inside such that ∠OPB = 45° and PA : PB = 5 : 14,
prove that PB = 42 cm. -/
theorem PB_length (s PA PB : ℝ) (h₁ : s^2 = 1989) 
(h₂ : PA / PB = 5 / 14) 
(h₃ : 25 * (PA / PB)^2 + 196 * (PB / PA)^2 = s^2) :
  PB = 42 := 
by sorry

end PB_length_l179_179132


namespace bert_money_problem_l179_179521

-- Define the conditions as hypotheses
theorem bert_money_problem
  (n : ℝ)
  (h1 : n > 0)  -- Since he can't have negative or zero dollars initially
  (h2 : (1/2) * ((3/4) * n - 9) = 15) :
  n = 52 :=
sorry

end bert_money_problem_l179_179521


namespace probability_product_even_gt_one_fourth_l179_179365

def n := 100
def is_even (x : ℕ) : Prop := x % 2 = 0
def is_odd (x : ℕ) : Prop := ¬ is_even x

theorem probability_product_even_gt_one_fourth :
  (∃ (p : ℝ), p > 0 ∧ p = 1 - (50 * 49 * 48 : ℝ) / (100 * 99 * 98) ∧ p > 1 / 4) :=
sorry

end probability_product_even_gt_one_fourth_l179_179365


namespace p_sufficient_but_not_necessary_for_q_l179_179574

def condition_p (x : ℝ) : Prop := abs (x - 1) < 2
def condition_q (x : ℝ) : Prop := x^2 - 5 * x - 6 < 0

theorem p_sufficient_but_not_necessary_for_q : 
  (∀ x, condition_p x → condition_q x) ∧ 
  ¬ (∀ x, condition_q x → condition_p x) := 
by
  sorry

end p_sufficient_but_not_necessary_for_q_l179_179574


namespace school_dance_boys_count_l179_179363

theorem school_dance_boys_count
  (total_attendees : ℕ)
  (percent_faculty_staff : ℝ)
  (fraction_girls : ℝ)
  (h1 : total_attendees = 100)
  (h2 : percent_faculty_staff = 0.1)
  (h3 : fraction_girls = 2/3) :
  let faculty_staff := total_attendees * percent_faculty_staff in
  let students := total_attendees - faculty_staff in
  let girls := students * fraction_girls in
  let boys := students - girls in
  boys = 30 :=
by
  -- Skipping the proof
  sorry

end school_dance_boys_count_l179_179363


namespace max_cube_side_length_max_rect_parallelepiped_dimensions_l179_179472

-- Part (a)
theorem max_cube_side_length (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ s : ℝ, s = a * b * c / (a * b + b * c + a * c) :=
sorry

-- Part (b)
theorem max_rect_parallelepiped_dimensions (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ x y z : ℝ, x = a / 3 ∧ y = b / 3 ∧ z = c / 3 :=
sorry

end max_cube_side_length_max_rect_parallelepiped_dimensions_l179_179472


namespace calculate_expression_l179_179397

theorem calculate_expression : 15 * 35 + 45 * 15 = 1200 :=
by {
  -- hint to the Lean prover to consider associative property
  sorry
}

end calculate_expression_l179_179397


namespace man_monthly_salary_l179_179997

theorem man_monthly_salary (S E : ℝ) (h1 : 0.20 * S = S - 1.20 * E) (h2 : E = 0.80 * S) :
  S = 6000 :=
by
  sorry

end man_monthly_salary_l179_179997


namespace part1_part2_l179_179799

noncomputable def f (x : ℝ) : ℝ := abs (2 * x - 1) - abs (x + 2)

theorem part1 {x : ℝ} : f x > 0 ↔ (x < -1 / 3 ∨ x > 3) := sorry

theorem part2 {m : ℝ} (h : ∃ x₀ : ℝ, f x₀ + 2 * m^2 < 4 * m) : -1 / 2 < m ∧ m < 5 / 2 := sorry

end part1_part2_l179_179799


namespace aprilPriceChange_l179_179130

noncomputable def priceChangeInApril : ℕ :=
  let P0 := 100
  let P1 := P0 + (20 / 100) * P0
  let P2 := P1 - (20 / 100) * P1
  let P3 := P2 + (25 / 100) * P2
  let P4 := P3 - x / 100 * P3
  17

theorem aprilPriceChange (x : ℕ) : x = priceChangeInApril := by
  sorry

end aprilPriceChange_l179_179130


namespace functional_form_of_f_l179_179790

variable (f : ℝ → ℝ)

-- Define the condition as an axiom
axiom cond_f : ∀ (x y : ℝ), |f (x + y) - f (x - y) - y| ≤ y^2

-- State the theorem to be proved
theorem functional_form_of_f :
  ∃ c : ℝ, ∀ x : ℝ, f x = x / 2 + c := by
  sorry

end functional_form_of_f_l179_179790


namespace correct_statement_l179_179224

def synthetic_method_is_direct : Prop := -- define the synthetic method
  True  -- We'll say True to assume it's a direct proof method. This is a simplification.

def analytic_method_is_direct : Prop := -- define the analytic method
  True  -- We'll say True to assume it's a direct proof method. This is a simplification.

theorem correct_statement : synthetic_method_is_direct ∧ analytic_method_is_direct → 
                             "Synthetic method and analytic method are direct proof methods" = "A" :=
by
  intros h
  cases h
  -- This is where you would provide the proof steps. We skip this with sorry.
  sorry

end correct_statement_l179_179224


namespace pascal_fifth_element_row_20_l179_179053

def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem pascal_fifth_element_row_20 : binom 20 4 = 4845 := sorry

end pascal_fifth_element_row_20_l179_179053


namespace slope_of_tangent_line_at_A_l179_179957

noncomputable def f (x : ℝ) := x^2 + 3 * x

def derivative_at (f : ℝ → ℝ) (x : ℝ) : ℝ :=
  (sorry : ℝ)  -- Placeholder for the definition of the derivative

theorem slope_of_tangent_line_at_A : 
  derivative_at f 1 = 5 := 
sorry

end slope_of_tangent_line_at_A_l179_179957


namespace additional_people_needed_l179_179299

-- Define the conditions
def num_people_initial := 9
def work_done_initial := 3 / 5
def days_initial := 14
def days_remaining := 4

-- Calculated values based on conditions
def work_rate_per_person : ℚ :=
  work_done_initial / (num_people_initial * days_initial)

def work_remaining : ℚ := 1 - work_done_initial

def total_people_needed : ℚ :=
  work_remaining / (work_rate_per_person * days_remaining)

-- Formulate the statement to prove
theorem additional_people_needed :
  total_people_needed - num_people_initial = 12 :=
by
  sorry

end additional_people_needed_l179_179299


namespace possible_slopes_of_line_intersecting_ellipse_l179_179531

theorem possible_slopes_of_line_intersecting_ellipse (m : ℝ) :
  (∃ x y : ℝ, y = m * x + 3 ∧ 4 * x^2 + 25 * y^2 = 100) ↔ m ∈ (Set.Iic (-2/5) ∪ Set.Ici (2/5)) :=
by
  sorry

end possible_slopes_of_line_intersecting_ellipse_l179_179531


namespace original_rope_length_l179_179538

variable (S : ℕ) (L : ℕ)

-- Conditions
axiom shorter_piece_length : S = 20
axiom longer_piece_length : L = 2 * S

-- Prove that the original length of the rope is 60 meters
theorem original_rope_length : S + L = 60 :=
by
  -- proof steps will go here
  sorry

end original_rope_length_l179_179538


namespace sum_of_numbers_l179_179882

theorem sum_of_numbers : 
  (87 + 91 + 94 + 88 + 93 + 91 + 89 + 87 + 92 + 86 + 90 + 92 + 88 + 90 + 91 + 86 + 89 + 92 + 95 + 88) = 1799 := 
by 
  sorry

end sum_of_numbers_l179_179882


namespace value_of_a_l179_179012

theorem value_of_a :
  ∀ (a : ℤ) (BO CO : ℤ), 
  BO = 2 → 
  CO = 2 * BO → 
  |a + 3| = CO → 
  a < 0 → 
  a = -7 := by
  intros a BO CO hBO hCO hAbs ha_neg
  sorry

end value_of_a_l179_179012


namespace chess_club_boys_l179_179529

theorem chess_club_boys (G B : ℕ) 
  (h1 : G + B = 30)
  (h2 : (2 / 3) * G + (3 / 4) * B = 18) : B = 24 :=
by
  sorry

end chess_club_boys_l179_179529


namespace simplify_rationalize_expr_l179_179028

theorem simplify_rationalize_expr :
  (1 / (2 + 1 / (Real.sqrt 5 + 2))) = (Real.sqrt 5 / 5) :=
by
  sorry

end simplify_rationalize_expr_l179_179028


namespace simplify_sqrt_product_l179_179163

theorem simplify_sqrt_product (x : ℝ) : 
  (Real.sqrt (50 * x) * Real.sqrt (18 * x) * Real.sqrt (32 * x)) = 120 * x * Real.sqrt (2 * x) := 
by
  sorry

end simplify_sqrt_product_l179_179163


namespace triangle_inequality_l179_179621

variable (a b c : ℝ) -- sides of the triangle
variable (h_a h_b h_c S r R : ℝ) -- heights, area of the triangle, inradius, circumradius

-- Definitions of conditions
axiom h_def : h_a + h_b + h_c = (a + b + c) -- express heights sum in terms of sides sum (for illustrative purposes)
axiom S_def : S = 0.5 * a * h_a  -- area definition (adjust as needed)
axiom r_def : 9 * r ≤ h_a + h_b + h_c -- given in solution
axiom R_def : h_a + h_b + h_c ≤ 9 * R / 2 -- given in solution

theorem triangle_inequality :
  9 * r / (2 * S) ≤ (1 / a) + (1 / b) + (1 / c) ∧ (1 / a) + (1 / b) + (1 / c) ≤ 9 * R / (4 * S) :=
by
  sorry

end triangle_inequality_l179_179621


namespace polynomial_identity_l179_179756

theorem polynomial_identity (x : ℝ) :
  (x - 1)^4 + 4 * (x - 1)^3 + 6 * (x - 1)^2 + 4 * (x - 1) + 1 = x^4 :=
sorry

end polynomial_identity_l179_179756


namespace darryl_books_l179_179484

variable (l m d : ℕ)

theorem darryl_books (h1 : l + m + d = 97) (h2 : l = m - 3) (h3 : m = 2 * d) : d = 20 := 
by
  sorry

end darryl_books_l179_179484


namespace workers_complete_time_l179_179440

theorem workers_complete_time
  (A : ℝ) -- Total work
  (x1 x2 x3 : ℝ) -- Productivities of the workers
  (h1 : x3 = (x1 + x2) / 2)
  (h2 : 10 * x1 = 15 * x2) :
  (A / x1 = 50) ∧ (A / x2 = 75) ∧ (A / x3 = 60) :=
by
  sorry  -- Proof not required

end workers_complete_time_l179_179440


namespace square_perimeter_calculation_l179_179080

noncomputable def perimeter_of_square (radius: ℝ) : ℝ := 
  if radius = 4 then 64 * Real.sqrt 2 else 0

theorem square_perimeter_calculation :
  perimeter_of_square 4 = 64 * Real.sqrt 2 :=
by
  sorry

end square_perimeter_calculation_l179_179080


namespace greatest_mondays_in_45_days_l179_179972

/-- 
Given that a week consists of 7 days, 
and each week has exactly 1 Monday, 
prove that the greatest number of Mondays 
that can occur in the first 45 days of a year is 7.
-/
theorem greatest_mondays_in_45_days : 
  (∃ (n : ℕ), n = 45) → 
  (∀ (week : ℕ), week * 7 ≤ 45 → week.mondays = 1) → 
  ∃ (max_mondays : ℕ), max_mondays = 7 :=
sorry

end greatest_mondays_in_45_days_l179_179972


namespace maxCubeSideLength_correct_maxRectParallelepipedDims_correct_l179_179467

noncomputable def maxCubeSideLength (a b c : ℝ) : ℝ :=
  a * b * c / (a * b + b * c + a * c)

noncomputable def maxRectParallelepipedDims (a b c : ℝ) : ℝ × ℝ × ℝ :=
  (a / 3, b / 3, c / 3)

theorem maxCubeSideLength_correct (a b c : ℝ) :
  a > 0 → b > 0 → c > 0 →
  maxCubeSideLength a b c = a * b * c / (a * b + b * c + a * c) :=
sorry

theorem maxRectParallelepipedDims_correct (a b c : ℝ) :
  a > 0 → b > 0 → c > 0 →
  maxRectParallelepipedDims a b c = (a / 3, b / 3, c / 3) :=
sorry

end maxCubeSideLength_correct_maxRectParallelepipedDims_correct_l179_179467


namespace number_of_people_l179_179240

theorem number_of_people (x : ℕ) (h1 : 175 = 175) (h2: 2 = 2) (h3 : ∀ (p : ℕ), p * x = 175 + p * 10) : x = 7 :=
sorry

end number_of_people_l179_179240


namespace shaded_area_is_110_l179_179317

-- Definitions based on conditions
def equilateral_triangle_area : ℕ := 10
def num_triangles_small : ℕ := 1
def num_triangles_medium : ℕ := 3
def num_triangles_large : ℕ := 7

-- Total area calculation
def total_area : ℕ := (num_triangles_small + num_triangles_medium + num_triangles_large) * equilateral_triangle_area

-- The theorem statement
theorem shaded_area_is_110 : total_area = 110 := 
by 
  sorry

end shaded_area_is_110_l179_179317


namespace total_apples_in_stack_l179_179994

theorem total_apples_in_stack:
  let base_layer := 6 * 9
  let layer_2 := 5 * 8
  let layer_3 := 4 * 7
  let layer_4 := 3 * 6
  let layer_5 := 2 * 5
  let layer_6 := 1 * 4
  let top_layer := 2
  base_layer + layer_2 + layer_3 + layer_4 + layer_5 + layer_6 + top_layer = 156 :=
by sorry

end total_apples_in_stack_l179_179994


namespace max_value_of_exp_l179_179480

theorem max_value_of_exp (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 3) : 
  a^2 * b^3 * c ≤ 27 / 16 := 
  sorry

end max_value_of_exp_l179_179480


namespace polynomial_value_at_neg2_l179_179513

def polynomial (x : ℝ) : ℝ :=
  x^6 - 5 * x^5 + 6 * x^4 + x^2 + 0.3 * x + 2

theorem polynomial_value_at_neg2 :
  polynomial (-2) = 325.4 :=
by
  sorry

end polynomial_value_at_neg2_l179_179513


namespace product_modulo_6_l179_179087

theorem product_modulo_6 :
  (2017 * 2018 * 2019 * 2020) % 6 = 0 :=
by
  -- Conditions provided:
  have h1 : 2017 ≡ 5 [MOD 6] := by sorry
  have h2 : 2018 ≡ 0 [MOD 6] := by sorry
  have h3 : 2019 ≡ 1 [MOD 6] := by sorry
  have h4 : 2020 ≡ 2 [MOD 6] := by sorry
  -- Proof of the theorem:
  sorry

end product_modulo_6_l179_179087


namespace percentage_import_tax_l179_179998

theorem percentage_import_tax (total_value import_paid excess_amount taxable_amount : ℝ) 
  (h1 : total_value = 2570) 
  (h2 : import_paid = 109.90) 
  (h3 : excess_amount = 1000) 
  (h4 : taxable_amount = total_value - excess_amount) : 
  taxable_amount = 1570 →
  (import_paid / taxable_amount) * 100 = 7 := 
by
  intros h_taxable_amount
  simp [h1, h2, h3, h4, h_taxable_amount]
  sorry -- Proof goes here

end percentage_import_tax_l179_179998


namespace min_games_to_predict_l179_179702

theorem min_games_to_predict (W B : ℕ) (total_games : ℕ) (n : ℕ) : 
  W = 15 → B = 20 → total_games = W * B → n = 280 → 
  (∃ x, x ∈ {i | ∃ j, i < W ∧ j < B}) :=
by
  intros hW hB htotal hn
  sorry

end min_games_to_predict_l179_179702


namespace hyperbola_foci_y_axis_l179_179287

theorem hyperbola_foci_y_axis (a b : ℝ) (h : ∀ x y : ℝ, a * x^2 + b * y^2 = 1 → (1/a < 0 ∧ 1/b > 0)) : a < 0 ∧ b > 0 :=
by
  sorry

end hyperbola_foci_y_axis_l179_179287


namespace determine_a_if_derivative_is_even_l179_179496

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 + a * x^2 + x

noncomputable def f' (x : ℝ) (a : ℝ) : ℝ := 3 * x^2 + 2 * a * x + 1

theorem determine_a_if_derivative_is_even (a : ℝ) :
  (∀ x : ℝ, f' x a = f' (-x) a) → a = 0 :=
by
  intros h
  sorry

end determine_a_if_derivative_is_even_l179_179496


namespace find_quotient_l179_179816

-- Variables for larger number L and smaller number S
variables (L S: ℕ)

-- Conditions as definitions
def condition1 := L - S = 1325
def condition2 (quotient: ℕ) := L = S * quotient + 5
def condition3 := L = 1650

-- Statement to prove the quotient is 5
theorem find_quotient : ∃ (quotient: ℕ), condition1 L S ∧ condition2 L S quotient ∧ condition3 L → quotient = 5 := by
  sorry

end find_quotient_l179_179816


namespace model_to_statue_ratio_l179_179542

theorem model_to_statue_ratio 
  (statue_height : ℝ) 
  (model_height_feet : ℝ)
  (model_height_inches : ℝ)
  (conversion_factor : ℝ) :
  statue_height = 45 → model_height_feet = 3 → conversion_factor = 12 → model_height_inches = model_height_feet * conversion_factor →
  (45 / model_height_inches) = 1.25 :=
by
  sorry

end model_to_statue_ratio_l179_179542


namespace walnut_trees_total_l179_179187

variable (current_trees : ℕ) (new_trees : ℕ)

theorem walnut_trees_total (h1 : current_trees = 22) (h2 : new_trees = 55) : current_trees + new_trees = 77 :=
by
  sorry

end walnut_trees_total_l179_179187


namespace best_fit_line_slope_l179_179977

theorem best_fit_line_slope (x1 x2 x3 x4 y1 y2 y3 y4 : ℝ) (d : ℝ) 
  (h1 : x2 - x1 = 2 * d) (h2 : x3 - x2 = 3 * d) (h3 : x4 - x3 = d) : 
  ((y4 - y1) / (x4 - x1)) = (y4 - y1) / (x4 - x1) :=
by
  sorry

end best_fit_line_slope_l179_179977


namespace intersection_eq_l179_179576

-- defining the set A
def A := {x : ℝ | x^2 + 2*x - 3 ≤ 0}

-- defining the set B
def B := {y : ℝ | ∃ x ∈ A, y = x^2 + 4*x + 3}

-- The proof problem statement: prove that A ∩ B = [-1, 1]
theorem intersection_eq : A ∩ B = {y : ℝ | -1 ≤ y ∧ y ≤ 1} :=
by sorry

end intersection_eq_l179_179576


namespace inequality_ab_leq_a_b_l179_179943

theorem inequality_ab_leq_a_b (a b : ℝ) (x : ℝ) (h_a : 0 < a) (h_b : 0 < b) :
  a * b ≤ (a * (Real.sin x) ^ 2 + b * (Real.cos x) ^ 2) * (a * (Real.cos x) ^ 2 + b * (Real.sin x) ^ 2)
  ∧ (a * (Real.sin x) ^ 2 + b * (Real.cos x) ^ 2) * (a * (Real.cos x) ^ 2 + b * (Real.sin x) ^ 2) ≤ (a + b)^2 / 4 := 
sorry

end inequality_ab_leq_a_b_l179_179943


namespace sequence_term_101_l179_179604

theorem sequence_term_101 :
  ∃ a : ℕ → ℚ, a 1 = 2 ∧ (∀ n : ℕ, 2 * a (n+1) - 2 * a n = 1) ∧ a 101 = 52 :=
by
  sorry

end sequence_term_101_l179_179604


namespace largest_base5_eq_124_l179_179216

-- Define largest base-5 number with three digits
def largest_base5_three_digits : ℕ := 4 * 5^2 + 4 * 5^1 + 4 * 5^0

theorem largest_base5_eq_124 : largest_base5_three_digits = 124 :=
by
  -- Proof steps go here
  sorry

end largest_base5_eq_124_l179_179216


namespace matches_in_round1_group1_matches_in_round1_group2_matches_in_round2_l179_179686

def num_teams_group1 : ℕ := 3
def num_teams_group2 : ℕ := 4

def num_matches_round1_group1 (n : ℕ) : ℕ := n * (n - 1) / 2
def num_matches_round1_group2 (n : ℕ) : ℕ := n * (n - 1) / 2

def num_matches_round2 (n1 n2 : ℕ) : ℕ := n1 * n2

theorem matches_in_round1_group1 : num_matches_round1_group1 num_teams_group1 = 3 := 
by
  -- Exact proof steps should be filled in here.
  sorry

theorem matches_in_round1_group2 : num_matches_round1_group2 num_teams_group2 = 6 := 
by
  -- Exact proof steps should be filled in here.
  sorry

theorem matches_in_round2 : num_matches_round2 num_teams_group1 num_teams_group2 = 12 := 
by
  -- Exact proof steps should be filled in here.
  sorry

end matches_in_round1_group1_matches_in_round1_group2_matches_in_round2_l179_179686


namespace right_building_shorter_l179_179156

-- Define the conditions as hypotheses
def middle_building_height : ℕ := 100
def left_building_height : ℕ := (80 * middle_building_height) / 100
def combined_height_left_middle : ℕ := left_building_height + middle_building_height
def total_height : ℕ := 340
def right_building_height : ℕ := total_height - combined_height_left_middle

-- Define the statement we need to prove
theorem right_building_shorter :
  combined_height_left_middle - right_building_height = 20 :=
by sorry

end right_building_shorter_l179_179156


namespace max_naive_number_l179_179417

-- Define the digits and conditions for a naive number
variable (a b c d : ℕ)
variable (M : ℕ)
variable (h1 : b = c + 2)
variable (h2 : a = d + 6)
variable (h3 : M = 1000 * a + 100 * b + 10 * c + d)

-- Define P(M) and Q(M)
def P (a b c d : ℕ) : ℕ := 3 * (a + b) + c + d
def Q (a : ℕ) : ℕ := a - 5

-- Problem statement: Prove the maximum value of M satisfying the divisibility condition
theorem max_naive_number (div_cond : (P a b c d) % (Q a) = 0) (hq : Q a % 10 = 0) : M = 9313 := 
sorry

end max_naive_number_l179_179417


namespace first_year_students_sampled_equals_40_l179_179192

-- Defining the conditions
def num_first_year_students := 800
def num_second_year_students := 600
def num_third_year_students := 500
def num_sampled_third_year_students := 25
def total_students := num_first_year_students + num_second_year_students + num_third_year_students

-- Proving the number of first-year students sampled
theorem first_year_students_sampled_equals_40 :
  (num_first_year_students * num_sampled_third_year_students) / num_third_year_students = 40 := by
  sorry

end first_year_students_sampled_equals_40_l179_179192


namespace sub_of_neg_l179_179255

theorem sub_of_neg : -3 - 2 = -5 :=
by 
  sorry

end sub_of_neg_l179_179255


namespace minimum_value_is_16_l179_179871

noncomputable def minimum_value_expression (x y : ℝ) (hx : x > 1) (hy : y > 1) : ℝ :=
  (x^3 / (y - 1) + y^3 / (x - 1))

theorem minimum_value_is_16 (x y : ℝ) (hx : x > 1) (hy : y > 1) :
  minimum_value_expression x y hx hy ≥ 16 :=
sorry

end minimum_value_is_16_l179_179871


namespace option_D_is_linear_equation_with_two_variables_l179_179676

def is_linear_equation (eq : String) : Prop :=
  match eq with
  | "3x - 6 = x" => false
  | "x = 5 / y - 1" => false
  | "2x - 3y = x^2" => false
  | "3x = 2y" => true
  | _ => false

theorem option_D_is_linear_equation_with_two_variables :
  is_linear_equation "3x = 2y" = true := by
  sorry

end option_D_is_linear_equation_with_two_variables_l179_179676


namespace trigonometric_identity_l179_179554

theorem trigonometric_identity :
  (Real.cos (17 * Real.pi / 180) * Real.sin (43 * Real.pi / 180) + 
   Real.sin (163 * Real.pi / 180) * Real.sin (47 * Real.pi / 180)) = 
  (Real.sqrt 3 / 2) :=
by
  sorry

end trigonometric_identity_l179_179554


namespace arithmetic_sequence_21st_term_and_sum_l179_179352

theorem arithmetic_sequence_21st_term_and_sum 
    (a1 : Int)
    (d : Int)
    : a1 = 2 → d = 5 →
      (arithSeqTerm a1 d 21 = 102 ∧ arithSeqSum a1 d 21 = 1092) := by {
    intros,
    sorry
}

end arithmetic_sequence_21st_term_and_sum_l179_179352


namespace geometric_seq_sum_identity_l179_179114

noncomputable def is_geometric_seq (a : ℕ → ℝ) : Prop :=
  ∃ q ≠ 0, ∀ n, a (n + 1) = q * a n

theorem geometric_seq_sum_identity (a : ℕ → ℝ) (q : ℝ) (hq : q ≠ 0)
  (hgeom : is_geometric_seq a) 
  (h1 : a 2 + a 6 = 3) 
  (h2 : a 6 + a 10 = 12) : 
  a 8 + a 12 = 24 :=
sorry

end geometric_seq_sum_identity_l179_179114


namespace seq_val_a7_l179_179391

theorem seq_val_a7 {a : ℕ} {b : ℕ} 
  (h1 : a < b) 
  (h2 : a_6 = a_1 + 3 * (a_1 + 2 * a_2) + a_5) 
  (h3 : a_n = if a_n = 6 then 74 else a_n) 
  (h4 : ∀ n ≥ 1, a_{n+2} = a_{n+1} + a_n) : 
  a_7 = 119 ∨ a_7 = 120 := 
sorry

end seq_val_a7_l179_179391


namespace sum_of_products_l179_179825

theorem sum_of_products (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 62)
  (h2 : a + b + c = 18) : 
  a * b + b * c + c * a = 131 :=
sorry

end sum_of_products_l179_179825


namespace sausage_left_l179_179847

variables (S x y : ℝ)

-- Conditions
axiom dog_bites : y = x + 300
axiom cat_bites : x = y + 500

-- Theorem Statement
theorem sausage_left {S x y : ℝ}
  (h1 : y = x + 300)
  (h2 : x = y + 500) : S - x - y = 400 :=
by
  sorry

end sausage_left_l179_179847


namespace max_cube_side_length_max_parallelepiped_dimensions_l179_179462

theorem max_cube_side_length (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) : 
    ∃ (a0 : ℝ), a0 = a * b * c / (a * b + b * c + a * c) := 
sorry

theorem max_parallelepiped_dimensions (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) : 
    ∃ (x y z : ℝ), (x = a / 3) ∧ (y = b / 3) ∧ (z = c / 3) :=
sorry

end max_cube_side_length_max_parallelepiped_dimensions_l179_179462


namespace trapezoid_area_l179_179318

theorem trapezoid_area (outer_triangle_area inner_triangle_area : ℝ) (congruent_trapezoids : ℕ) 
  (h1 : outer_triangle_area = 36) (h2 : inner_triangle_area = 4) (h3 : congruent_trapezoids = 3) :
  (outer_triangle_area - inner_triangle_area) / congruent_trapezoids = 32 / 3 :=
by sorry

end trapezoid_area_l179_179318


namespace proof_problem_l179_179023

def problem_expression : ℚ := 1 / (2 + 1 / (Real.sqrt 5 + 2))

theorem proof_problem : problem_expression = Real.sqrt 5 / 5 := by sorry

end proof_problem_l179_179023


namespace a_10_is_100_l179_179758

-- Define the sequence a_n as a function from ℕ+ (the positive naturals) to ℤ
axiom a : ℕ+ → ℤ

-- Given assumptions
axiom seq_relation : ∀ m n : ℕ+, a m + a n = a (m + n) - 2 * m.val * n.val
axiom a1 : a 1 = 1

-- Goal statement
theorem a_10_is_100 : a 10 = 100 :=
by
  -- proof goes here, this is just the statement
  sorry

end a_10_is_100_l179_179758


namespace cone_central_angle_l179_179988

/-- Proof Problem Statement: Given the radius of the base circle of a cone (r) and the slant height of the cone (l),
    prove that the central angle (θ) of the unfolded diagram of the lateral surface of this cone is 120 degrees. -/
theorem cone_central_angle (r l : ℝ) (h_r : r = 10) (h_l : l = 30) : (360 * r) / l = 120 :=
by
  -- The proof steps are omitted
  sorry

end cone_central_angle_l179_179988


namespace value_of_T_l179_179041

variables {A M T E H : ℕ}

theorem value_of_T (H : ℕ) (MATH : ℕ) (MEET : ℕ) (TEAM : ℕ) (H_eq : H = 8) (MATH_eq : MATH = 47) (MEET_eq : MEET = 62) (TEAM_eq : TEAM = 58) :
  T = 9 :=
by
  sorry

end value_of_T_l179_179041


namespace somu_current_age_l179_179063

variable (S F : ℕ)

theorem somu_current_age
  (h1 : S = F / 3)
  (h2 : S - 10 = (F - 10) / 5) :
  S = 20 := by
  sorry

end somu_current_age_l179_179063


namespace solve_equation_l179_179420

noncomputable def maxRational (x y : ℚ) : ℚ :=
  if x > y then x else y

theorem solve_equation (x : ℚ) : maxRational x (-x) = 2 * x + 9 ↔ x = -3 :=
begin
  sorry
end

end solve_equation_l179_179420


namespace data_set_average_l179_179246

theorem data_set_average (a : ℝ) (h : (2 + 3 + 3 + 4 + a) / 5 = 3) : a = 3 := 
sorry

end data_set_average_l179_179246


namespace time_to_fill_pool_l179_179335

variable (pool_volume : ℝ) (fill_rate : ℝ) (leak_rate : ℝ)

theorem time_to_fill_pool (h_pool_volume : pool_volume = 60)
    (h_fill_rate : fill_rate = 1.6)
    (h_leak_rate : leak_rate = 0.1) :
    (pool_volume / (fill_rate - leak_rate)) = 40 :=
by
  -- We skip the proof step, only the theorem statement is required
  sorry

end time_to_fill_pool_l179_179335


namespace remaining_problems_to_grade_l179_179065

-- Define the conditions
def problems_per_worksheet : ℕ := 3
def total_worksheets : ℕ := 15
def graded_worksheets : ℕ := 7

-- The remaining worksheets to grade
def remaining_worksheets : ℕ := total_worksheets - graded_worksheets

-- Theorems stating the amount of problems left to grade
theorem remaining_problems_to_grade : problems_per_worksheet * remaining_worksheets = 24 :=
by
  sorry

end remaining_problems_to_grade_l179_179065


namespace quadratic_solutions_1_quadratic_k_value_and_solutions_l179_179525

-- Problem (Ⅰ):
theorem quadratic_solutions_1 {x : ℝ} :
  x^2 + 6 * x + 5 = 0 ↔ x = -5 ∨ x = -1 :=
sorry

-- Problem (Ⅱ):
theorem quadratic_k_value_and_solutions {x k : ℝ} (x1 x2 : ℝ) :
  x1 + x2 = 3 ∧ x1 * x2 = k ∧ (x1 - 1) * (x2 - 1) = -6 ↔ (k = -4 ∧ (x = 4 ∨ x = -1)) :=
sorry

end quadratic_solutions_1_quadratic_k_value_and_solutions_l179_179525


namespace balls_drawn_ensure_single_color_ge_20_l179_179231

theorem balls_drawn_ensure_single_color_ge_20 (r g y b w bl : ℕ) (h_r : r = 34) (h_g : g = 28) (h_y : y = 23) (h_b : b = 18) (h_w : w = 12) (h_bl : bl = 11) : 
  ∃ (n : ℕ), n ≥ 20 →
    (r + g + y + b + w + bl - n) + 1 > 20 :=
by
  sorry

end balls_drawn_ensure_single_color_ge_20_l179_179231


namespace win_sector_area_l179_179234

-- Define the radius of the circle (spinner)
def radius : ℝ := 8

-- Define the probability of winning on one spin
def probability_winning : ℝ := 1 / 4

-- Define the area of the circle, calculated from the radius
def total_area : ℝ := Real.pi * radius^2

-- The area of the WIN sector to be proven
theorem win_sector_area : (probability_winning * total_area) = 16 * Real.pi := by
  sorry

end win_sector_area_l179_179234


namespace root_of_quadratic_eq_l179_179577

theorem root_of_quadratic_eq (a b : ℝ) (h : a + b - 3 = 0) : a + b = 3 :=
sorry

end root_of_quadratic_eq_l179_179577


namespace divide_participants_into_groups_l179_179524

theorem divide_participants_into_groups (m : ℕ) (h_m : 1 < m) (n : ℕ) (h_n_range : 1 < n ∧ n ≤ m) 
(participants : finset (fin m)) (matches : finset (fin m × fin m)) 
(h_match_condition : ∀ (p1 p2 : fin m), p1 ≠ p2 → (p1, p2) ∈ matches ∨ (p2, p1) ∈ matches → (p1, p2) ∈ matches) :
∃ (groups : finset (finset (fin m))), finset.card groups = n ∧ ∀ p ∈ participants, ∀ g ∈ groups, 
  p ∈ g → (finset.card (finset.filter (λ x, (x, p) ∈ matches ∨ (p, x) ∈ matches) g) ≤ (fintype.card (fin m) / n)) :=
 sorry

end divide_participants_into_groups_l179_179524


namespace city_raised_money_for_charity_l179_179349

-- Definitions based on conditions from part a)
def price_regular_duck : ℝ := 3.0
def price_large_duck : ℝ := 5.0
def number_regular_ducks_sold : ℕ := 221
def number_large_ducks_sold : ℕ := 185

-- Definition to represent the main theorem: Total money raised
noncomputable def total_money_raised : ℝ :=
  price_regular_duck * number_regular_ducks_sold + price_large_duck * number_large_ducks_sold

-- Theorem to prove that the total money raised is $1588.00
theorem city_raised_money_for_charity : total_money_raised = 1588.0 := by
  sorry

end city_raised_money_for_charity_l179_179349


namespace determine_exponent_l179_179586

-- Declare variables
variables {x y : ℝ}
variable {n : ℕ}

-- Use condition that the terms are like terms
theorem determine_exponent (h : - x ^ 2 * y ^ n = 3 * y * x ^ 2) : n = 1 :=
sorry

end determine_exponent_l179_179586


namespace part1_part2_l179_179106

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

end part1_part2_l179_179106


namespace largest_cube_side_length_largest_rect_parallelepiped_dimensions_l179_179475

variables {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

-- Part (a): Side length of the largest cube
theorem largest_cube_side_length :
  let a₀ := a * b * c / (a * b + b * c + a * c) in
  ∃ a₀, a₀ = a * b * c / (a * b + b * c + a * c) :=
begin
  sorry
end

-- Part (b): Dimensions of the largest rectangular parallelepiped
theorem largest_rect_parallelepiped_dimensions :
  let x := a / 3, y := b / 3, z := c / 3 in
  ∃ x y z, x = a / 3 ∧ y = b / 3 ∧ z = c / 3 :=
begin
  sorry
end

end largest_cube_side_length_largest_rect_parallelepiped_dimensions_l179_179475


namespace max_value_expr_l179_179111

def point_on_line (m n : ℝ) : Prop :=
  3 * m + n = -1

def mn_positive (m n : ℝ) : Prop :=
  m * n > 0

theorem max_value_expr (m n : ℝ) (h1 : point_on_line m n) (h2 : mn_positive m n) :
  (3 / m + 1 / n) = -16 :=
sorry

end max_value_expr_l179_179111


namespace problem_condition_necessary_and_sufficient_l179_179900

theorem problem_condition_necessary_and_sufficient (a b : ℝ) (h : a * b > 0) :
  (a > b) ↔ (1 / a < 1 / b) :=
sorry

end problem_condition_necessary_and_sufficient_l179_179900


namespace quadratic_real_roots_a_leq_2_l179_179307

theorem quadratic_real_roots_a_leq_2
    (a : ℝ) :
    (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 - 4*x1 + 2*a = 0) ∧ (x2^2 - 4*x2 + 2*a = 0)) →
    a ≤ 2 :=
by sorry

end quadratic_real_roots_a_leq_2_l179_179307


namespace smallest_number_of_blocks_needed_l179_179985

/--
Given:
  A wall with the following properties:
  1. The wall is 100 feet long and 7 feet high.
  2. Blocks used are 1 foot high and either 1 foot or 2 feet long.
  3. Blocks cannot be cut.
  4. Vertical joins in the blocks must be staggered.
  5. The wall must be even on the ends.
Prove:
  The smallest number of blocks needed to build this wall is 353.
-/
theorem smallest_number_of_blocks_needed :
  let length := 100
  let height := 7
  let block_height := 1
  (∀ b : ℕ, b = 1 ∨ b = 2) →
  ∃ (blocks_needed : ℕ), blocks_needed = 353 :=
by sorry

end smallest_number_of_blocks_needed_l179_179985


namespace two_b_squared_eq_a_squared_plus_c_squared_l179_179840

theorem two_b_squared_eq_a_squared_plus_c_squared (a b c : ℝ) (h : 1 / (a + b) + 1 / (b + c) = 2 / (c + a)) : 
  2 * b^2 = a^2 + c^2 := 
sorry

end two_b_squared_eq_a_squared_plus_c_squared_l179_179840


namespace sergio_more_correct_than_sylvia_l179_179634

theorem sergio_more_correct_than_sylvia
  (num_questions : ℕ)
  (fraction_incorrect_sylvia : ℚ)
  (num_mistakes_sergio : ℕ)
  (sylvia_incorrect : ℕ := (fraction_incorrect_sylvia * num_questions).to_nat)
  (sylvia_correct : ℕ := num_questions - sylvia_incorrect)
  (sergio_correct : ℕ := num_questions - num_mistakes_sergio)
  (correct_answer_diff : ℕ := sergio_correct - sylvia_correct) :
  num_questions = 50 →
  fraction_incorrect_sylvia = 1 / 5 →
  num_mistakes_sergio = 4 →
  correct_answer_diff = 6 :=
begin
  assume (num_questions_eq : num_questions = 50)
  (fraction_incorrect_sylvia_eq : fraction_incorrect_sylvia = 1/5)
  (num_mistakes_sergio_eq : num_mistakes_sergio = 4),
  sorry
end

end sergio_more_correct_than_sylvia_l179_179634


namespace minimum_games_to_predict_participant_l179_179694

theorem minimum_games_to_predict_participant :
  ∃ n, (n ≤ 300) ∧ (∀ m, m < n → (∃ one_participant_remaining, 15 * 20 - m > 20)) ∧ n = 280 := 
sorry

end minimum_games_to_predict_participant_l179_179694


namespace floor_sum_correct_l179_179400

def floor_sum_1_to_24 := 
  let sum := (3 * 1) + (5 * 2) + (7 * 3) + (9 * 4)
  sum

theorem floor_sum_correct : floor_sum_1_to_24 = 70 := by
  sorry

end floor_sum_correct_l179_179400


namespace cube_root_simplification_l179_179519

theorem cube_root_simplification (c d : ℕ) (h1 : c = 3) (h2 : d = 100) : c + d = 103 :=
by
  sorry

end cube_root_simplification_l179_179519


namespace john_new_earnings_l179_179607

theorem john_new_earnings (original_earnings raise_percentage: ℝ)
  (h1 : original_earnings = 60)
  (h2 : raise_percentage = 40) :
  original_earnings * (1 + raise_percentage / 100) = 84 := 
by
  sorry

end john_new_earnings_l179_179607


namespace grasshopper_flea_adjacency_l179_179169

-- Define the types of cells
inductive CellColor
| Red
| White

-- Define the infinite grid as a function from ℤ × ℤ to CellColor
def InfiniteGrid : Type := ℤ × ℤ → CellColor

-- Define the positions of the grasshopper and the flea
variables (g_start f_start : ℤ × ℤ)

-- The conditions for the grid and movement rules
axiom grid_conditions (grid : InfiniteGrid) :
  ∃ g_pos f_pos : ℤ × ℤ, 
  (g_pos = g_start ∧ f_pos = f_start) ∧
  (∀ x y : ℤ × ℤ, grid x = CellColor.Red ∨ grid x = CellColor.White) ∧
  (∀ x y : ℤ × ℤ, grid y = CellColor.Red ∨ grid y = CellColor.White)

-- Define the movement conditions for grasshopper and flea
axiom grasshopper_jumps (grid : InfiniteGrid) (start : ℤ × ℤ) :
  ∃ end_pos : ℤ × ℤ, grid end_pos = CellColor.Red ∧ ((end_pos.1 = start.1 ∨ end_pos.2 = start.2) ∧ abs (end_pos.1 - start.1) ≤ 1 ∧ abs (end_pos.2 - start.2) ≤ 1)

axiom flea_jumps (grid : InfiniteGrid) (start : ℤ × ℤ) :
  ∃ end_pos : ℤ × ℤ, grid end_pos = CellColor.White ∧ ((end_pos.1 = start.1 ∨ end_pos.2 = start.2) ∧ abs (end_pos.1 - start.1) ≤ 1 ∧ abs (end_pos.2 - start.2) ≤ 1)

-- The main theorem statement
theorem grasshopper_flea_adjacency (grid : InfiniteGrid)
    (g_start f_start : ℤ × ℤ) :
    ∃ pos1 pos2 pos3 : ℤ × ℤ,
    (pos1 = g_start ∨ pos1 = f_start) ∧ 
    (pos2 = g_start ∨ pos2 = f_start) ∧ 
    (abs (pos3.1 - g_start.1) + abs (pos3.2 - g_start.2) ≤ 1 ∧ 
    abs (pos3.1 - f_start.1) + abs (pos3.2 - f_start.2) ≤ 1) :=
sorry

end grasshopper_flea_adjacency_l179_179169


namespace quadratic_complete_square_l179_179820

theorem quadratic_complete_square (x d e: ℝ) (h : x^2 - 26 * x + 129 = (x + d)^2 + e) : 
d + e = -53 := sorry

end quadratic_complete_square_l179_179820


namespace pages_revised_twice_l179_179179

theorem pages_revised_twice
  (x : ℕ)
  (h1 : ∀ x, x > 30 → 1000 + 100 + 10 * x ≠ 1400)
  (h2 : ∀ x, x < 30 → 1000 + 100 + 10 * x ≠ 1400)
  (h3 : 1000 + 100 + 10 * 30 = 1400) :
  x = 30 :=
by
  sorry

end pages_revised_twice_l179_179179


namespace sum_f_1_to_10_l179_179104

-- Define the function f with the properties given.

def f (x : ℝ) : ℝ := sorry

-- Specify the conditions of the problem
local notation "R" => ℝ

axiom odd_function : ∀ (x : R), f (-x) = -f (x)
axiom periodicity : ∀ (x : R), f (x + 3) = f (x)
axiom f_neg1 : f (-1) = 1

-- State the theorem to be proved
theorem sum_f_1_to_10 : f 1 + f 2 + f 3 + f 4 + f 5 + f 6 + f 7 + f 8 + f 9 + f 10 = -1 :=
by
  sorry
end sum_f_1_to_10_l179_179104


namespace line_quadrants_l179_179818

theorem line_quadrants (k b : ℝ) (h : ∃ x y : ℝ, y = k * x + b ∧ 
                                          ((x > 0 ∧ y > 0) ∧   -- First quadrant
                                           (x < 0 ∧ y < 0) ∧   -- Third quadrant
                                           (x > 0 ∧ y < 0))) : -- Fourth quadrant
  k > 0 :=
sorry

end line_quadrants_l179_179818


namespace polynomial_expansion_l179_179105

theorem polynomial_expansion :
  let x := 1 
  let y := -1 
  let a_0 := (3 - 2 * x)^5 + (3 - 2 * y)^5 
  let a_1 := (3 - 2 * x)^5 - (3 - 2 * y)^5 
  let a_2 := (3 - 2 * x)^5 + (3 - 2 * y)^5 
  let a_3 := (3 - 2 * x)^5 - (3 - 2 * y)^5 
  let a_4 := (3 - 2 * x)^5 + (3 - 2 * y)^5 
  let a_5 := (3 - 2 * x)^5 - (3 - 2 * y)^5 
  (a_0 + a_2 + a_4)^2 - (a_1 + a_3 + a_5)^2 = 3125 := by
sorry

end polynomial_expansion_l179_179105


namespace box_and_apples_weight_l179_179528

theorem box_and_apples_weight
  (total_weight : ℝ)
  (weight_after_half : ℝ)
  (h1 : total_weight = 62.8)
  (h2 : weight_after_half = 31.8) :
  ∃ (box_weight apple_weight : ℝ), box_weight = 0.8 ∧ apple_weight = 62 :=
by
  sorry

end box_and_apples_weight_l179_179528


namespace correct_system_of_equations_l179_179386

theorem correct_system_of_equations :
  ∃ (x y : ℕ), 
    x + y = 38 
    ∧ 26 * x + 20 * y = 952 := 
by
  sorry

end correct_system_of_equations_l179_179386


namespace fraction_computation_l179_179257

theorem fraction_computation :
  (2 + 4 - 8 + 16 + 32 - 64) / (4 + 8 - 16 + 32 + 64 - 128) = 1 / 2 :=
by
  sorry

end fraction_computation_l179_179257


namespace largest_base5_three_digit_to_base10_l179_179202

theorem largest_base5_three_digit_to_base10 :
  (4 * 5^2 + 4 * 5^1 + 4 * 5^0) = 124 :=
by
  sorry

end largest_base5_three_digit_to_base10_l179_179202


namespace planet_combinations_l179_179123

theorem planet_combinations :
  let a : ℕ := 3 in
  let b : ℕ := 6 in
  let c : ℕ := 4 in
  let d : ℕ := 4 in
  let e : ℕ := 5 in
  let f : ℕ := 2 in
  let combination_a := Nat.choose 5 3 in
  let combination_b := (Nat.choose 5 4) * (Nat.choose 6 4) in
  let combination_c := (Nat.choose 5 5) * (Nat.choose 6 2) in
  let total_combinations := combination_a + combination_b + combination_c in
  total_combinations = 100 :=
by
  sorry

end planet_combinations_l179_179123


namespace car_stops_at_3_seconds_l179_179171

-- Define the distance function S(t)
def S (t : ℝ) : ℝ := -3 * t^2 + 18 * t

-- The proof statement
theorem car_stops_at_3_seconds :
    (∃ t : ℝ, S t = 27 ∧ t = 3) :=
begin
  use 3,
  split,
  { unfold S,
    simp,
    exact eq.refl 27 },
  { exact eq.refl 3 },
end

end car_stops_at_3_seconds_l179_179171


namespace largest_base5_three_digits_is_124_l179_179205

noncomputable def largest_base5_three_digits_to_base10 : ℕ :=
  4 * 5^2 + 4 * 5^1 + 4 * 5^0

theorem largest_base5_three_digits_is_124 :
  largest_base5_three_digits_to_base10 = 124 :=
by
  -- calculating 4 * 5^2 + 4 * 5^1 + 4 * 5^0 = 124
  sorry

end largest_base5_three_digits_is_124_l179_179205


namespace school_dance_boys_count_l179_179364

theorem school_dance_boys_count :
  let total_attendees := 100
  let faculty_and_staff := total_attendees * 10 / 100
  let students := total_attendees - faculty_and_staff
  let girls := 2 * students / 3
  let boys := students - girls
  boys = 30 := by
  sorry

end school_dance_boys_count_l179_179364


namespace shanghai_expo_problem_l179_179494

open Nat

theorem shanghai_expo_problem (year : ℕ) (may_1_weekday : ℕ → ℕ) :
  year = 2010 ∧ (may_1_weekday 2010 = 6) →
  let common_year := year % 4 ≠ 0 in
  common_year ∧
  let days_in_year := 365 in
  days_in_year = 365 ∧
  let may_31_weekday := (may_1_weekday 2010 + 30) % 7 in
  may_31_weekday = 1 ∧
  let total_days_from_may_1_to_oct_31 := 31 + 30 + 31 + 31 + 30  in
  total_days_from_may_1_to_oct_31 = 184 :=
by
  assume h,
  cases h with h_year h_may_1,
  have h_common_year : year % 4 ≠ 0 := by
    rw h_year
    norm_num,
  have h_days_in_year : 365 = 365 := rfl,
  have h_may_31_weekday : (may_1_weekday 2010 + 30) % 7 = 1 := by
    rw [h_may_1]
    norm_num,
  have h_total_days : 31 + 30 + 31 + 31 + 30 = 184 := by
    norm_num,
  exact ⟨h_common_year, ⟨h_days_in_year, ⟨h_may_31_weekday, h_total_days⟩⟩⟩

end shanghai_expo_problem_l179_179494


namespace cryptarithm_C_value_l179_179605

/--
Given digits A, B, and C where A, B, and C are distinct and non-repeating,
and the following conditions hold:
1. ABC - BC = A0A
Prove that C = 9.
-/
theorem cryptarithm_C_value (A B C : ℕ) (h_distinct : A ≠ B ∧ B ≠ C ∧ A ≠ C)
  (h_non_repeating: (0 <= A ∧ A <= 9) ∧ (0 <= B ∧ B <= 9) ∧ (0 <= C ∧ C <= 9))
  (h_subtraction : 100 * A + 10 * B + C - (10 * B + C) = 100 * A + 0 + A) :
  C = 9 := sorry

end cryptarithm_C_value_l179_179605


namespace c_share_l179_179684

theorem c_share (a b c : ℝ) (h1 : a = b / 2) (h2 : b = c / 2) (h3 : a + b + c = 700) : c = 400 :=
by 
  -- Proof goes here
  sorry

end c_share_l179_179684


namespace algae_difference_l179_179148

theorem algae_difference :
  let original_algae := 809
  let current_algae := 3263
  current_algae - original_algae = 2454 :=
by
  sorry

end algae_difference_l179_179148


namespace molecular_weight_C8H10N4O6_eq_258_22_l179_179553

def atomic_weight_C : ℝ := 12.01
def atomic_weight_H : ℝ := 1.01
def atomic_weight_N : ℝ := 14.01
def atomic_weight_O : ℝ := 16.00

def number_C : ℕ := 8
def number_H : ℕ := 10
def number_N : ℕ := 4
def number_O : ℕ := 6

def molecular_weight : ℝ :=
    (number_C * atomic_weight_C) +
    (number_H * atomic_weight_H) +
    (number_N * atomic_weight_N) +
    (number_O * atomic_weight_O)

theorem molecular_weight_C8H10N4O6_eq_258_22 :
  molecular_weight = 258.22 :=
  by
    sorry

end molecular_weight_C8H10N4O6_eq_258_22_l179_179553


namespace jill_peaches_l179_179477

variable (S J : ℕ)

theorem jill_peaches (h1 : S = 19) (h2 : S = J + 13) : J = 6 :=
by
  sorry

end jill_peaches_l179_179477


namespace Mona_unique_players_l179_179009

theorem Mona_unique_players :
  ∀ (g : ℕ) (p : ℕ) (r1 : ℕ) (r2 : ℕ),
  g = 9 → p = 4 → r1 = 2 → r2 = 1 →
  (g * p) - (r1 + r2) = 33 :=
by {
  intros g p r1 r2 hg hp hr1 hr2,
  rw [hg, hp, hr1, hr2],
  norm_num,
  sorry -- skipping proof as per instructions
}

end Mona_unique_players_l179_179009


namespace prime_solutions_l179_179413

theorem prime_solutions (p : ℕ) (n : ℕ) (hp : p.prime) :
  p^2 + n^2 = 3 * p * n + 1 ↔ (p, n) = (3, 1) ∨ (p, n) = (3, 8) :=
by sorry

end prime_solutions_l179_179413


namespace div64_by_expression_l179_179018

theorem div64_by_expression {n : ℕ} (h : n > 0) : ∃ k : ℤ, (3^(2 * n + 2) - 8 * ↑n - 9) = 64 * k :=
by
  sorry

end div64_by_expression_l179_179018


namespace f_f_of_2020_l179_179775

def f (x : ℕ) : ℕ :=
  if x ≤ 1 then 1
  else if 1 < x ∧ x ≤ 1837 then 2
  else if 1837 < x ∧ x < 2019 then 3
  else 2018

theorem f_f_of_2020 : f (f 2020) = 3 := by
  sorry

end f_f_of_2020_l179_179775


namespace right_angled_triangle_l179_179359
  
theorem right_angled_triangle (x : ℝ) (hx : 0 < x) :
  let a := 5 * x
  let b := 12 * x
  let c := 13 * x
  a^2 + b^2 = c^2 :=
by
  let a := 5 * x
  let b := 12 * x
  let c := 13 * x
  sorry

end right_angled_triangle_l179_179359


namespace div_condition_for_lcm_l179_179229

theorem div_condition_for_lcm (x y : ℕ) (hx : x > 1) (hy : y > 1)
  (h : Nat.lcm (x + 2) (y + 2) - Nat.lcm (x + 1) (y + 1) = Nat.lcm (x + 1) (y + 1) - Nat.lcm x y) :
  x ∣ y ∨ y ∣ x :=
sorry

end div_condition_for_lcm_l179_179229


namespace common_chord_length_proof_l179_179504

-- Define the first circle equation
def first_circle (x y : ℝ) : Prop := x^2 + y^2 = 50

-- Define the second circle equation
def second_circle (x y : ℝ) : Prop := x^2 + y^2 - 12*x - 6*y + 40 = 0

-- Define the property that the length of the common chord is equal to 2 * sqrt(5)
noncomputable def common_chord_length : ℝ := 2 * Real.sqrt 5

-- The theorem statement
theorem common_chord_length_proof :
  ∀ x y : ℝ, first_circle x y → second_circle x y → common_chord_length = 2 * Real.sqrt 5 :=
by
  intros x y h1 h2
  sorry

end common_chord_length_proof_l179_179504


namespace amplitude_of_f_phase_shift_of_f_vertical_shift_of_f_l179_179100

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (4 * x - Real.pi / 2) + 1

theorem amplitude_of_f : (∀ x y : ℝ, |f x - f y| ≤ 2 * |x - y|) := sorry

theorem phase_shift_of_f : (∃ φ : ℝ, φ = -Real.pi / 8) := sorry

theorem vertical_shift_of_f : (∃ v : ℝ, v = 1) := sorry

end amplitude_of_f_phase_shift_of_f_vertical_shift_of_f_l179_179100


namespace vasya_can_interfere_with_petya_goal_l179_179153

theorem vasya_can_interfere_with_petya_goal :
  ∃ (evens odds : ℕ), evens + odds = 50 ∧ (evens + odds) % 2 = 1 :=
sorry

end vasya_can_interfere_with_petya_goal_l179_179153


namespace single_colony_reaches_limit_in_24_days_l179_179844

/-- A bacteria colony doubles in size every day. -/
def double (n : ℕ) : ℕ := 2 ^ n

/-- Two bacteria colonies growing simultaneously will take 24 days to reach the habitat's limit. -/
axiom two_colonies_24_days : ∀ k : ℕ, double k + double k = double 24

/-- Prove that it takes 24 days for a single bacteria colony to reach the habitat's limit. -/
theorem single_colony_reaches_limit_in_24_days : ∃ x : ℕ, double x = double 24 :=
sorry

end single_colony_reaches_limit_in_24_days_l179_179844


namespace largest_base5_three_digit_to_base10_l179_179200

theorem largest_base5_three_digit_to_base10 : 
  let base5_number := 4 * 5^2 + 4 * 5^1 + 4 * 5^0 in
  base5_number = 124 :=
by
let base5_number := 4 * 5^2 + 4 * 5^1 + 4 * 5^0
have h1 : 4 * 5^2 = 100 := by norm_num
have h2 : 4 * 5^1 = 20 := by norm_num
have h3 : 4 * 5^0 = 4 := by norm_num
have h4 : base5_number = 100 + 20 + 4 := by simp [base5_number, h1, h2, h3]
show base5_number = 124, by simp [h4]
sorry

end largest_base5_three_digit_to_base10_l179_179200


namespace permute_rows_to_columns_l179_179544

open Function

-- Define the problem
theorem permute_rows_to_columns {α : Type*} [Fintype α] [DecidableEq α] (n : ℕ)
  (table : Fin n → Fin n → α)
  (h_distinct_rows : ∀ i : Fin n, ∀ j₁ j₂ : Fin n, j₁ ≠ j₂ → table i j₁ ≠ table i j₂) :
  ∃ (p : Fin n → Fin n → Fin n), ∀ j : Fin n, ∀ i₁ i₂ : Fin n, i₁ ≠ i₂ →
    table i₁ (p i₁ j) ≠ table i₂ (p i₂ j) := 
sorry

end permute_rows_to_columns_l179_179544


namespace smallest_positive_integer_divisible_by_14_15_18_l179_179102

theorem smallest_positive_integer_divisible_by_14_15_18 : 
  ∃ n : ℕ, n > 0 ∧ (14 ∣ n) ∧ (15 ∣ n) ∧ (18 ∣ n) ∧ n = 630 :=
sorry

end smallest_positive_integer_divisible_by_14_15_18_l179_179102


namespace inequality_proof_l179_179624

theorem inequality_proof (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (2 * x * y) / (x + y) + Real.sqrt ((x ^ 2 + y ^ 2) / 2) ≥ (x + y) / 2 + Real.sqrt (x * y) :=
by
  sorry

end inequality_proof_l179_179624


namespace square_of_sum_possible_l179_179134

theorem square_of_sum_possible (a b c : ℝ) : 
  ∃ d : ℝ, d = (a + b + c)^2 :=
sorry

end square_of_sum_possible_l179_179134


namespace max_product_931_l179_179968

open Nat

theorem max_product_931 : ∀ a b c d e : ℕ,
  -- Digits must be between 1 and 9 and all distinct.
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e ∧ 
  {1, 3, 5, 8, 9} = {a, b, c, d, e} ∧ 
  -- Three-digit and two-digit numbers.
  100 * a + 10 * b + c ≤ 999 ∧ 100 * a + 10 * b + c ≥ 100 ∧
  10 * d + e ≤ 99 ∧ 10 * d + e ≥ 10
  -- Then the resulting product is maximized when abc = 931.
  → 100 * a + 10 * b + c = 931 := 
by
  intros a b c d e h_distinct h_digits h_3digit_range h_2digit_range
  sorry

end max_product_931_l179_179968


namespace value_not_uniquely_determined_l179_179402

variables (v : Fin 9 → ℤ) (s : Fin 9 → ℤ)

-- Given conditions
axiom announced_sums : ∀ i, s i = v ((i - 1) % 9) + v ((i + 1) % 9)
axiom sums_sequence : s 0 = 3 ∧ s 1 = 7 ∧ s 2 = 12 ∧ s 3 = 18 ∧ s 4 = 24 ∧ s 5 = 31 ∧ s 6 = 40 ∧ s 7 = 48 ∧ s 8 = 53

-- Statement asserting the indeterminacy of v_5
theorem value_not_uniquely_determined (h: s 3 = 18) : 
  ∃ v : Fin 9 → ℤ, sorry :=
sorry

end value_not_uniquely_determined_l179_179402


namespace part_a_l179_179099

theorem part_a (a b : ℤ) (x : ℤ) :
  (x % 5 = a) ∧ (x % 6 = b) → x = 6 * a + 25 * b :=
by
  sorry

end part_a_l179_179099


namespace chips_cost_l179_179828

noncomputable def cost_of_each_bag_of_chips (amount_paid_per_friend : ℕ) (number_of_friends : ℕ) (number_of_bags : ℕ) : ℕ :=
  (amount_paid_per_friend * number_of_friends) / number_of_bags

theorem chips_cost
  (amount_paid_per_friend : ℕ := 5)
  (number_of_friends : ℕ := 3)
  (number_of_bags : ℕ := 5) :
  cost_of_each_bag_of_chips amount_paid_per_friend number_of_friends number_of_bags = 3 :=
by
  sorry

end chips_cost_l179_179828


namespace total_bouquets_sold_l179_179716

-- defining the sale conditions
def monday_bouquets := 12
def tuesday_bouquets := 3 * monday_bouquets
def wednesday_bouquets := tuesday_bouquets / 3

-- defining the total sale
def total_bouquets := monday_bouquets + tuesday_bouquets + wednesday_bouquets

-- stating the theorem
theorem total_bouquets_sold : total_bouquets = 60 := by
  -- the proof would go here
  sorry

end total_bouquets_sold_l179_179716


namespace pairs_satisfy_equation_l179_179873

theorem pairs_satisfy_equation :
  ∀ (x n : ℕ), (x > 0 ∧ n > 0) ∧ 3 * 2 ^ x + 4 = n ^ 2 → (x, n) = (2, 4) ∨ (x, n) = (5, 10) ∨ (x, n) = (6, 14) :=
by
  sorry

end pairs_satisfy_equation_l179_179873


namespace interest_rate_eq_ten_l179_179175

theorem interest_rate_eq_ten (R : ℝ) (P : ℝ) (SI CI : ℝ) :
  P = 1400 ∧
  SI = 14 * R ∧
  CI = 1400 * ((1 + R / 200) ^ 2 - 1) ∧
  CI - SI = 3.50 → 
  R = 10 :=
by
  sorry

end interest_rate_eq_ten_l179_179175


namespace transformation_l179_179064

noncomputable def Q (a b c x y : ℝ) : ℝ := a * x^2 + 2 * b * x * y + c * y^2

theorem transformation 
  (a b c d e f x y x₀ y₀ x' y' : ℝ)
  (h : a * c - b^2 ≠ 0)
  (hQ : Q a b c x y + 2 * d * x + 2 * e * y = f)
  (hx : x' = x + x₀)
  (hy : y' = y + y₀) :
  ∃ f' : ℝ, (a * x'^2 + 2 * b * x' * y' + c * y'^2 = f' ∧ 
             f' = f - Q a b c x₀ y₀ + 2 * (d * x₀ + e * y₀)) :=
sorry

end transformation_l179_179064


namespace minimum_games_to_predict_participant_l179_179693

theorem minimum_games_to_predict_participant :
  ∃ n, (n ≤ 300) ∧ (∀ m, m < n → (∃ one_participant_remaining, 15 * 20 - m > 20)) ∧ n = 280 := 
sorry

end minimum_games_to_predict_participant_l179_179693


namespace fifth_element_row_20_pascal_triangle_l179_179057

theorem fifth_element_row_20_pascal_triangle : binom 20 4 = 4845 :=
by 
  sorry

end fifth_element_row_20_pascal_triangle_l179_179057


namespace ratio_of_perimeters_l179_179638

theorem ratio_of_perimeters (s₁ s₂ : ℝ) (h : (s₁^2 / s₂^2) = (16 / 49)) : (4 * s₁) / (4 * s₂) = 4 / 7 :=
by
  -- Proof goes here
  sorry

end ratio_of_perimeters_l179_179638


namespace cost_per_bag_l179_179829

theorem cost_per_bag (total_friends: ℕ) (amount_paid_per_friend: ℕ) (total_bags: ℕ) 
  (h1 : total_friends = 3) (h2 : amount_paid_per_friend = 5) (h3 : total_bags = 5) 
  : total_friends * amount_paid_per_friend / total_bags = 3 := by
  sorry

end cost_per_bag_l179_179829


namespace find_OH_squared_l179_179482

variables {O H : Type} {a b c R : ℝ}

-- Given conditions
def is_circumcenter (O : Type) (ABC : Type) := true -- Placeholder definition
def is_orthocenter (H : Type) (ABC : Type) := true -- Placeholder definition
def circumradius (O : Type) (R : ℝ) := true -- Placeholder definition
def sides_squared_sum (a b c : ℝ) := a^2 + b^2 + c^2

-- The theorem to be proven
theorem find_OH_squared (O H : Type) (a b c : ℝ) (R : ℝ) 
  (circ : is_circumcenter O ABC) 
  (orth: is_orthocenter H ABC) 
  (radius : circumradius O R) 
  (terms_sum : sides_squared_sum a b c = 50)
  (R_val : R = 10) 
  : OH^2 = 850 := 
sorry

end find_OH_squared_l179_179482


namespace total_cost_to_plant_flowers_l179_179646

noncomputable def flower_cost : ℕ := 9
noncomputable def clay_pot_cost : ℕ := flower_cost + 20
noncomputable def soil_bag_cost : ℕ := flower_cost - 2
noncomputable def total_cost : ℕ := flower_cost + clay_pot_cost + soil_bag_cost

theorem total_cost_to_plant_flowers : total_cost = 45 := by
  sorry

end total_cost_to_plant_flowers_l179_179646


namespace village_population_l179_179230

theorem village_population (x : ℝ) (h : 0.96 * x = 23040) : x = 24000 := sorry

end village_population_l179_179230


namespace halfway_fraction_l179_179832

theorem halfway_fraction (a b : ℚ) (h1 : a = 3 / 4) (h2 : b = 6 / 7) :
  (a + b) / 2 = 45 / 56 := 
sorry

end halfway_fraction_l179_179832


namespace width_of_playground_is_250_l179_179962

noncomputable def total_area_km2 : ℝ := 0.6
def num_playgrounds : ℕ := 8
def length_of_playground_m : ℝ := 300

theorem width_of_playground_is_250 :
  let total_area_m2 := total_area_km2 * 1000000
  let area_of_one_playground := total_area_m2 / num_playgrounds
  let width_of_playground := area_of_one_playground / length_of_playground_m
  width_of_playground = 250 := by
  sorry

end width_of_playground_is_250_l179_179962


namespace largest_divisor_even_squares_l179_179793

theorem largest_divisor_even_squares (m n : ℕ) (hm : Even m) (hn : Even n) (h : n < m) :
  ∃ k, k = 4 ∧ ∀ a b : ℕ, Even a → Even b → b < a → k ∣ (a^2 - b^2) :=
by
  sorry

end largest_divisor_even_squares_l179_179793


namespace largest_base5_three_digits_is_124_l179_179207

noncomputable def largest_base5_three_digits_to_base10 : ℕ :=
  4 * 5^2 + 4 * 5^1 + 4 * 5^0

theorem largest_base5_three_digits_is_124 :
  largest_base5_three_digits_to_base10 = 124 :=
by
  -- calculating 4 * 5^2 + 4 * 5^1 + 4 * 5^0 = 124
  sorry

end largest_base5_three_digits_is_124_l179_179207


namespace length_of_track_l179_179785

-- Conditions as definitions
def Janet_runs (m : Nat) := m = 120
def Leah_distance_after_first_meeting (x : Nat) (m : Nat) := m = (x / 2 - 120 + 200)
def Janet_distance_after_first_meeting (x : Nat) (m : Nat) := m = (x - 120 + (x - (x / 2 + 80)))

-- Questions and answers combined in proof statement
theorem length_of_track (x : Nat) (hx : Janet_runs 120) (hy : Leah_distance_after_first_meeting x 280) (hz : Janet_distance_after_first_meeting x (x / 2 - 40)) :
  x = 480 :=
sorry

end length_of_track_l179_179785


namespace equivalence_a_gt_b_and_inv_a_lt_inv_b_l179_179897

variable {a b : ℝ}

theorem equivalence_a_gt_b_and_inv_a_lt_inv_b (h : a * b > 0) : 
  (a > b) ↔ (1 / a < 1 / b) := 
sorry

end equivalence_a_gt_b_and_inv_a_lt_inv_b_l179_179897


namespace Ned_earning_money_l179_179383

def total_games : Nat := 15
def non_working_games : Nat := 6
def price_per_game : Nat := 7
def working_games : Nat := total_games - non_working_games
def total_money : Nat := working_games * price_per_game

theorem Ned_earning_money : total_money = 63 := by
  sorry

end Ned_earning_money_l179_179383


namespace correct_units_l179_179667

def units_time := ["hour", "minute", "second"]
def units_mass := ["gram", "kilogram", "ton"]
def units_length := ["millimeter", "centimeter", "decimeter", "meter", "kilometer"]

theorem correct_units :
  (units_time = ["hour", "minute", "second"]) ∧
  (units_mass = ["gram", "kilogram", "ton"]) ∧
  (units_length = ["millimeter", "centimeter", "decimeter", "meter", "kilometer"]) :=
by
  -- Please provide the proof here
  sorry

end correct_units_l179_179667


namespace find_constants_C_and_A_l179_179266

theorem find_constants_C_and_A :
  ∃ (C A : ℚ), (C * x + 7 - 17)/(x^2 - 9 * x + 20) = A / (x - 4) + 2 / (x - 5) ∧ B = 7 ∧ C = 12/5 ∧ A = 2/5 := sorry

end find_constants_C_and_A_l179_179266


namespace find_a6_l179_179423

variable (a : ℕ → ℝ)

-- condition: a_2 + a_8 = 16
axiom h1 : a 2 + a 8 = 16

-- condition: a_4 = 1
axiom h2 : a 4 = 1

-- question: Prove that a_6 = 15
theorem find_a6 : a 6 = 15 :=
sorry

end find_a6_l179_179423


namespace blueberry_jelly_amount_l179_179159

theorem blueberry_jelly_amount (total_jelly : ℕ) (strawberry_jelly : ℕ) 
  (h_total : total_jelly = 6310) 
  (h_strawberry : strawberry_jelly = 1792) 
  : total_jelly - strawberry_jelly = 4518 := 
by 
  sorry

end blueberry_jelly_amount_l179_179159


namespace yoongi_caught_frogs_l179_179377

theorem yoongi_caught_frogs (initial_frogs caught_later : ℕ) (h1 : initial_frogs = 5) (h2 : caught_later = 2) : (initial_frogs + caught_later = 7) :=
by
  sorry

end yoongi_caught_frogs_l179_179377


namespace ice_cream_not_sold_total_l179_179862

theorem ice_cream_not_sold_total :
  let chocolate_initial := 50
  let mango_initial := 54
  let vanilla_initial := 80
  let strawberry_initial := 40
  let chocolate_sold := (3 / 5 : ℚ) * chocolate_initial
  let mango_sold := (2 / 3 : ℚ) * mango_initial
  let vanilla_sold := (75 / 100 : ℚ) * vanilla_initial
  let strawberry_sold := (5 / 8 : ℚ) * strawberry_initial
  let chocolate_not_sold := chocolate_initial - chocolate_sold
  let mango_not_sold := mango_initial - mango_sold
  let vanilla_not_sold := vanilla_initial - vanilla_sold
  let strawberry_not_sold := strawberry_initial - strawberry_sold
  chocolate_not_sold + mango_not_sold + vanilla_not_sold + strawberry_not_sold = 73 :=
by sorry

end ice_cream_not_sold_total_l179_179862


namespace sum_of_possible_values_of_d_l179_179987

def base_digits (n : ℕ) (b : ℕ) : ℕ := 
  if n = 0 then 1 else Nat.log (n + 1) b

theorem sum_of_possible_values_of_d :
  let min_val_7 := 1 * 7^3
  let max_val_7 := 6 * 7^3 + 6 * 7^2 + 6 * 7^1 + 6 * 7^0
  let min_val_10 := 343
  let max_val_10 := 2400
  let d1 := base_digits min_val_10 3
  let d2 := base_digits max_val_10 3
  d1 + d2 = 13 := sorry

end sum_of_possible_values_of_d_l179_179987


namespace problem_l179_179591

theorem problem (x y z : ℝ) (h : (x - z) ^ 2 - 4 * (x - y) * (y - z) = 0) : z + x - 2 * y = 0 :=
sorry

end problem_l179_179591


namespace ratio_of_perimeters_of_squares_l179_179637

theorem ratio_of_perimeters_of_squares (a₁ a₂ : ℕ) (s₁ s₂ : ℕ) (h : s₁^2 = 16 * a₁ ∧ s₂^2 = 49 * a₂) :
  4 * s₁ = 4 * (4/7) * s₂ :=
by
  have h1: s₁^2 / s₂^2 = 16 / 49 := sorry
  have h2: s₁ / s₂ = 4 / 7 := sorry
  have h3: 4 * s₁ = 4 * (4 / 7) * s₂ :=
    by simp [h2]
  exact h3

end ratio_of_perimeters_of_squares_l179_179637


namespace sasha_prediction_min_n_l179_179703

theorem sasha_prediction_min_n :
  let whiteRook_players : ℕ := 15 in
  let blackElephant_players : ℕ := 20 in
  let total_games : ℕ := whiteRook_players * blackElephant_players in
  ∀ (n : ℕ), n >= 280 → ∃ p ∈ fin (total_games - n), 
    (n < 280 -> ∀ i, ¬one_game_played_at_a_time whiteRook_players blackElephant_players total_games n i) :=
by
  intros whiteRook_players blackElephant_players total_games n hn,
  rw [total_games, mul_comm] at hn,
  sorry

end sasha_prediction_min_n_l179_179703


namespace student_A_more_stable_l179_179964

-- Given conditions
def average_score (n : ℕ) (score : ℕ) := score = 110
def variance_A := 3.6
def variance_B := 4.4

-- Prove that student A has more stable scores than student B
theorem student_A_more_stable : variance_A < variance_B :=
by
  -- Skipping the actual proof
  sorry

end student_A_more_stable_l179_179964


namespace gcd_m_n_l179_179481

def m : ℕ := 333333
def n : ℕ := 7777777

theorem gcd_m_n : Nat.gcd m n = 1 :=
by
  -- Mathematical steps have been omitted as they are not needed
  sorry

end gcd_m_n_l179_179481


namespace inequality_proof_l179_179117

noncomputable def f (x m : ℝ) : ℝ := 2 * m * x - Real.log x

theorem inequality_proof (m x₁ x₂ : ℝ) (hm : m ≥ -1) (hx₁ : x₁ > 0) (hx₂ : x₂ > 0)
  (hineq : (f x₁ m + f x₂ m) / 2 ≤ x₁ ^ 2 + x₂ ^ 2 + (3 / 2) * x₁ * x₂) :
  x₁ + x₂ ≥ (Real.sqrt 3 - 1) / 2 := 
sorry

end inequality_proof_l179_179117


namespace quadratic_value_at_point_a_l179_179547

noncomputable def quadratic (a b c : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + b * x + c

open Real

theorem quadratic_value_at_point_a
  (a b c : ℝ)
  (axis : ℝ)
  (sym : ∀ x, quadratic a b c (2 * axis - x) = quadratic a b c x)
  (at_zero : quadratic a b c 0 = -3) :
  quadratic a b c 20 = -3 := by
  -- proof steps would go here
  sorry

end quadratic_value_at_point_a_l179_179547


namespace largest_base_5_three_digit_in_base_10_l179_179212

theorem largest_base_5_three_digit_in_base_10 :
  let n := 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  n = 124 :=
by
  let n := 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  show n = 124
  sorry

end largest_base_5_three_digit_in_base_10_l179_179212


namespace cost_whitewashing_l179_179498

theorem cost_whitewashing
  (length : ℝ) (breadth : ℝ) (height : ℝ)
  (door_height : ℝ) (door_width : ℝ)
  (window_height : ℝ) (window_width : ℝ)
  (num_windows : ℕ) (cost_per_square_foot : ℝ)
  (room_dimensions : length = 25 ∧ breadth = 15 ∧ height = 12)
  (door_dimensions : door_height = 6 ∧ door_width = 3)
  (window_dimensions : window_height = 4 ∧ window_width = 3)
  (num_windows_condition : num_windows = 3)
  (cost_condition : cost_per_square_foot = 8) :
  (2 * (length + breadth) * height - (door_height * door_width + num_windows * window_height * window_width)) * cost_per_square_foot = 7248 := 
by
  sorry

end cost_whitewashing_l179_179498


namespace length_of_median_in_right_triangle_l179_179601

noncomputable def length_of_median (DE DF : ℝ) : ℝ :=
  let EF := Real.sqrt (DE^2 + DF^2)
  EF / 2

theorem length_of_median_in_right_triangle (DE DF : ℝ) (h1 : DE = 5) (h2 : DF = 12) :
  length_of_median DE DF = 6.5 :=
by
  -- Conditions
  rw [h1, h2]
  -- Proof (to be completed)
  sorry

end length_of_median_in_right_triangle_l179_179601


namespace statement_b_statement_c_l179_179677
-- Import all of Mathlib to include necessary mathematical functions and properties

-- First, the Lean statement for Statement B
theorem statement_b (a b : ℝ) (h : a > |b|) : a^2 > b^2 := 
sorry

-- Second, the Lean statement for Statement C
theorem statement_c (a b : ℝ) (h : a > b) : a^3 > b^3 := 
sorry

end statement_b_statement_c_l179_179677


namespace compare_sqrt_sums_l179_179569

   noncomputable def a : ℝ := Real.sqrt 8 + Real.sqrt 5
   noncomputable def b : ℝ := Real.sqrt 7 + Real.sqrt 6

   theorem compare_sqrt_sums : a < b :=
   by
     sorry
   
end compare_sqrt_sums_l179_179569


namespace ellipse_same_foci_l179_179113

-- Definitions related to the problem
variables {x y p q : ℝ}

-- Condition
def represents_hyperbola (p q : ℝ) : Prop :=
  (p * q > 0) ∧ (∀ x y : ℝ, (x^2 / -p + y^2 / q = 1))

-- Proof Statement
theorem ellipse_same_foci (p q : ℝ) (hpq : p * q > 0)
  (h : ∀ x y : ℝ, x^2 / -p + y^2 / q = 1) :
  (∀ x y : ℝ, x^2 / (2*p + q) + y^2 / p = -1) :=
sorry -- Proof goes here

end ellipse_same_foci_l179_179113


namespace largest_base5_three_digit_to_base10_l179_179204

theorem largest_base5_three_digit_to_base10 :
  (4 * 5^2 + 4 * 5^1 + 4 * 5^0) = 124 :=
by
  sorry

end largest_base5_three_digit_to_base10_l179_179204


namespace enclosed_by_eq_area_l179_179515

noncomputable def enclosed_area : ℝ :=
  let eq := λ (x y : ℝ), x^2 + y^2 = |x| + |y| + 1 in
  π * (√(3 / 2))^2 -- total circle area
  / 2 -- adjustment for quadrant coverage
  + 2 -- the area of triangles fixed by symmetry

theorem enclosed_by_eq_area :
  (enclosed_area = (3 / 2) * π + 2) :=
by
  sorry

end enclosed_by_eq_area_l179_179515


namespace sourav_srinath_same_fruit_days_l179_179340

def sourav_same_fruit_days (m n : ℕ) (hmn_gcd : Nat.gcd m n = 1) : ℕ :=
  (m * n + 1) / 2

theorem sourav_srinath_same_fruit_days (m n : ℕ) (hmn_gcd : Nat.gcd m n = 1) :
  let total_days := m * n
  let same_fruit_days := sourav_same_fruit_days m n hmn_gcd
  same_fruit_days = (total_days + 1) / 2 :=
by
  sorry

end sourav_srinath_same_fruit_days_l179_179340


namespace max_area_of_triangle_l179_179762

-- Defining the side lengths and constraints
def triangle_sides (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Main statement of the area maximization problem
theorem max_area_of_triangle (x : ℝ) (h1 : 2 < x) (h2 : x < 6) :
  triangle_sides 6 x (2 * x) →
  ∃ (S : ℝ), S = 12 :=
by
  sorry

end max_area_of_triangle_l179_179762


namespace factorization_of_a_cubed_minus_a_l179_179560

variable (a : ℝ)

theorem factorization_of_a_cubed_minus_a : a^3 - a = a * (a + 1) * (a - 1) := by
  sorry

end factorization_of_a_cubed_minus_a_l179_179560


namespace propA_neither_sufficient_nor_necessary_l179_179760

def PropA (a b : ℕ) : Prop := a + b ≠ 4
def PropB (a b : ℕ) : Prop := a ≠ 1 ∧ b ≠ 3

theorem propA_neither_sufficient_nor_necessary (a b : ℕ) : 
  ¬((PropA a b → PropB a b) ∧ (PropB a b → PropA a b)) :=
by {
  sorry
}

end propA_neither_sufficient_nor_necessary_l179_179760


namespace find_n_l179_179082

-- Define the values of quarters and dimes in cents
def value_of_quarter : ℕ := 25
def value_of_dime : ℕ := 10

-- Define the number of quarters and dimes
def num_quarters : ℕ := 15
def num_dimes : ℕ := 25

-- Define the total value in cents corresponding to the quarters
def total_value_quarters : ℕ := num_quarters * value_of_quarter

-- Define the condition where total value by quarters equals total value by n dimes
def equivalent_dimes (n : ℕ) : Prop := total_value_quarters = n * value_of_dime

-- The theorem to prove
theorem find_n : ∃ n : ℕ, equivalent_dimes n ∧ n = 38 := 
by {
  use 38,
  sorry
}

end find_n_l179_179082


namespace lab_techs_share_l179_179660

theorem lab_techs_share (u c t : ℕ) 
  (h1 : c = 6 * u)
  (h2 : t = u / 2)
  (h3 : u = 12) : 
  (c + u) / t = 14 := 
by 
  sorry

end lab_techs_share_l179_179660


namespace largest_base5_three_digit_in_base10_l179_179218

theorem largest_base5_three_digit_in_base10 :
  let a := 4
      b := 4
      c := 4
      largest_base5 := 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  in largest_base5 = 124 :=
by
  let a := 4
  let b := 4
  let c := 4
  let largest_base5 := 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  show largest_base5 = 124
  sorry

end largest_base5_three_digit_in_base10_l179_179218


namespace probability_more_sons_or_daughters_l179_179010

theorem probability_more_sons_or_daughters 
  (children : ℕ)
  (genders : Fin children → Bool)
  (probability : ℚ := 0.5) 
  (n : Nat := 8) :
  (∀ i : Fin n, genders i = true ∨ genders i = false) → 
  ∑ b in Finset.univ.filter (λ g, (Finset.filter id g).card ≠ n / 2), probability^(Finset.card b) = (93 / 128) :=
sorry

end probability_more_sons_or_daughters_l179_179010


namespace parabola_hyperbola_focus_l179_179303

/-- 
Proof problem: If the focus of the parabola y^2 = 2px coincides with the right focus of the hyperbola x^2/3 - y^2/1 = 1, then p = 2.
-/
theorem parabola_hyperbola_focus (p : ℝ) :
    ∀ (focus_parabola : ℝ × ℝ) (focus_hyperbola : ℝ × ℝ),
      (focus_parabola = (p, 0)) →
      (focus_hyperbola = (2, 0)) →
      (focus_parabola = focus_hyperbola) →
        p = 2 :=
by
  intros focus_parabola focus_hyperbola h1 h2 h3
  sorry

end parabola_hyperbola_focus_l179_179303


namespace number_of_members_l179_179717

theorem number_of_members (n : ℕ) (h : n * n = 2025) : n = 45 :=
sorry

end number_of_members_l179_179717


namespace license_plate_count_correct_l179_179072

-- Define the number of choices for digits and letters
def num_digit_choices : ℕ := 10^3
def num_letter_block_choices : ℕ := 26^3
def num_position_choices : ℕ := 4

-- Compute the total number of distinct license plates
def total_license_plates : ℕ := num_position_choices * num_digit_choices * num_letter_block_choices

-- The proof statement
theorem license_plate_count_correct : total_license_plates = 70304000 := by
  -- This proof is left as an exercise
  sorry

end license_plate_count_correct_l179_179072


namespace sticky_strips_used_l179_179665

theorem sticky_strips_used 
  (total_decorations : ℕ) 
  (nails_used : ℕ) 
  (decorations_hung_with_nails_fraction : ℚ) 
  (decorations_hung_with_thumbtacks_fraction : ℚ) 
  (nails_used_eq : nails_used = 50)
  (decorations_hung_with_nails_fraction_eq : decorations_hung_with_nails_fraction = 2/3)
  (decorations_hung_with_thumbtacks_fraction_eq : decorations_hung_with_thumbtacks_fraction = 2/5)
  (total_decorations_eq : total_decorations = nails_used / decorations_hung_with_nails_fraction)
  : (total_decorations - nails_used - decorations_hung_with_thumbtacks_fraction * (total_decorations - nails_used)) = 15 := 
by {
  sorry
}

end sticky_strips_used_l179_179665


namespace widget_difference_l179_179939

variable (w t : ℕ)

def monday_widgets (w t : ℕ) : ℕ := w * t
def tuesday_widgets (w t : ℕ) : ℕ := (w + 5) * (t - 3)

theorem widget_difference (h : w = 3 * t) :
  monday_widgets w t - tuesday_widgets w t = 4 * t + 15 :=
by
  sorry

end widget_difference_l179_179939


namespace sasha_prediction_l179_179697

theorem sasha_prediction (n : ℕ) 
  (white_rook_students : ℕ)
  (black_elephant_students : ℕ)
  (total_games : ℕ) :
  white_rook_students = 15 → 
  black_elephant_students = 20 → 
  total_games = 300 → 
  n = 280 → 
  ∃ s : ℕ, s ≤ white_rook_students ∧ s ≤ black_elephant_students ∧ s * black_elephant_students ≥ total_games - n :=
by
  sorry

end sasha_prediction_l179_179697


namespace length_of_bridge_l179_179681

theorem length_of_bridge
  (train_length : ℕ)
  (train_speed_kmh : ℕ)
  (crossing_time_seconds : ℕ)
  (h_train_length : train_length = 125)
  (h_train_speed_kmh : train_speed_kmh = 45)
  (h_crossing_time_seconds : crossing_time_seconds = 30) :
  ∃ (bridge_length : ℕ), bridge_length = 250 :=
by
  sorry

end length_of_bridge_l179_179681


namespace sebastian_total_payment_l179_179158

theorem sebastian_total_payment 
  (cost_per_ticket : ℕ) (number_of_tickets : ℕ) (service_fee : ℕ) (total_paid : ℕ)
  (h1 : cost_per_ticket = 44)
  (h2 : number_of_tickets = 3)
  (h3 : service_fee = 18)
  (h4 : total_paid = (number_of_tickets * cost_per_ticket) + service_fee) :
  total_paid = 150 :=
by
  sorry

end sebastian_total_payment_l179_179158


namespace cost_to_plant_flowers_l179_179647

theorem cost_to_plant_flowers :
  let cost_flowers := 9
  let cost_clay_pot := cost_flowers + 20
  let cost_soil := cost_flowers - 2
  cost_flowers + cost_clay_pot + cost_soil = 45 := 
by
  let cost_flowers := 9
  let cost_clay_pot := cost_flowers + 20
  let cost_soil := cost_flowers - 2
  show cost_flowers + cost_clay_pot + cost_soil = 45
  sorry

end cost_to_plant_flowers_l179_179647


namespace probability_unit_sphere_in_cube_l179_179074

noncomputable def cube_volume : ℝ := (4:ℝ) ^ 3

noncomputable def sphere_volume : ℝ := (4 * Real.pi) / 3

theorem probability_unit_sphere_in_cube:
  let probability := sphere_volume / cube_volume in
  probability = Real.pi / 48 := by
  sorry

end probability_unit_sphere_in_cube_l179_179074


namespace maxCubeSideLength_correct_maxRectParallelepipedDims_correct_l179_179465

noncomputable def maxCubeSideLength (a b c : ℝ) : ℝ :=
  a * b * c / (a * b + b * c + a * c)

noncomputable def maxRectParallelepipedDims (a b c : ℝ) : ℝ × ℝ × ℝ :=
  (a / 3, b / 3, c / 3)

theorem maxCubeSideLength_correct (a b c : ℝ) :
  a > 0 → b > 0 → c > 0 →
  maxCubeSideLength a b c = a * b * c / (a * b + b * c + a * c) :=
sorry

theorem maxRectParallelepipedDims_correct (a b c : ℝ) :
  a > 0 → b > 0 → c > 0 →
  maxRectParallelepipedDims a b c = (a / 3, b / 3, c / 3) :=
sorry

end maxCubeSideLength_correct_maxRectParallelepipedDims_correct_l179_179465


namespace solve_fraction_l179_179769

open Real

theorem solve_fraction (x : ℝ) (hx : 1 - 4 / x + 4 / x^2 = 0) : 2 / x = 1 :=
by
  -- We'll include the necessary steps of the proof here, but for now we leave it as sorry.
  sorry

end solve_fraction_l179_179769


namespace mack_writing_time_tuesday_l179_179940

variable (minutes_per_page_mon : ℕ := 30)
variable (time_mon : ℕ := 60)
variable (pages_wed : ℕ := 5)
variable (total_pages : ℕ := 10)
variable (minutes_per_page_tue : ℕ := 15)

theorem mack_writing_time_tuesday :
  (time_mon / minutes_per_page_mon) + pages_wed + (3 * minutes_per_page_tue / minutes_per_page_tue) = total_pages →
  (3 * minutes_per_page_tue) = 45 := by
  intros h
  sorry

end mack_writing_time_tuesday_l179_179940


namespace largest_base_5_three_digit_in_base_10_l179_179210

theorem largest_base_5_three_digit_in_base_10 :
  let n := 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  n = 124 :=
by
  let n := 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  show n = 124
  sorry

end largest_base_5_three_digit_in_base_10_l179_179210


namespace ratio_y_to_x_l179_179393

-- Definitions based on conditions
variable (c : ℝ) -- Cost price
def x : ℝ := 0.8 * c -- Selling price for a loss of 20%
def y : ℝ := 1.25 * c -- Selling price for a gain of 25%

-- Statement to prove the ratio of y to x
theorem ratio_y_to_x : y / x = 25 / 16 := by
  -- skip the proof
  sorry

end ratio_y_to_x_l179_179393


namespace lulu_cash_left_l179_179929

-- Define the initial amount
def initial_amount : ℕ := 65

-- Define the amount spent on ice cream
def spent_on_ice_cream : ℕ := 5

-- Define the amount spent on a t-shirt
def spent_on_tshirt (remaining_after_ice_cream : ℕ) : ℕ := remaining_after_ice_cream / 2

-- Define the amount deposited in the bank
def deposited_in_bank (remaining_after_tshirt : ℕ) : ℕ := remaining_after_tshirt / 5

-- Define the remaining cash after all transactions
def remaining_cash (initial : ℕ) (spent_ice_cream : ℕ) (spent_tshirt: ℕ) (deposited: ℕ) :ℕ :=
  initial - spent_ice_cream - spent_tshirt - deposited

-- Theorem statement to prove
theorem lulu_cash_left : remaining_cash initial_amount spent_on_ice_cream (spent_on_tshirt (initial_amount - spent_on_ice_cream)) 
(deposited_in_bank ((initial_amount - spent_on_ice_cream) - (spent_on_tshirt (initial_amount - spent_on_ice_cream)))) = 24 :=
by
  sorry

end lulu_cash_left_l179_179929


namespace pref_card_game_arrangements_l179_179603

noncomputable def number_of_arrangements :=
  (Nat.factorial 32) / ((Nat.factorial 10) ^ 3 * Nat.factorial 2 * Nat.factorial 3)

theorem pref_card_game_arrangements :
  number_of_arrangements = (Nat.factorial 32) / ((Nat.factorial 10) ^ 3 * Nat.factorial 2 * Nat.factorial 3) :=
by
  sorry

end pref_card_game_arrangements_l179_179603


namespace acute_angle_at_7_35_l179_179672

def minute_hand_angle (minute : ℕ) : ℝ :=
  minute / 60 * 360

def hour_hand_angle (hour : ℕ) (minute : ℕ) : ℝ :=
  (hour + minute / 60) / 12 * 360

def angle_between_hands (hour : ℕ) (minute : ℕ) : ℝ :=
  abs (hour_hand_angle hour minute - minute_hand_angle minute)

theorem acute_angle_at_7_35 : angle_between_hands 7 35 = 17 :=
by 
  sorry

end acute_angle_at_7_35_l179_179672


namespace number_of_zeros_of_g_l179_179431

noncomputable def f (x a : ℝ) := Real.exp x * (x + a)

noncomputable def g (x a : ℝ) := f (x - a) a - x^2

theorem number_of_zeros_of_g (a : ℝ) :
  (if a < 1 then ∃! x, g x a = 0
   else if a = 1 then ∃! x1 x2, x1 ≠ x2 ∧ g x1 a = 0 ∧ g x2 a = 0
   else ∃! x1 x2 x3, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 ∧ g x1 a = 0 ∧ g x2 a = 0 ∧ g x3 a = 0) := sorry

end number_of_zeros_of_g_l179_179431


namespace trapezoid_area_difference_l179_179549

def trapezoid_area (base1 base2 height : ℝ) : ℝ :=
  0.5 * (base1 + base2) * height

def combined_area (base1 base2 height : ℝ) : ℝ :=
  2 * trapezoid_area base1 base2 height

theorem trapezoid_area_difference :
  let combined_area1 := combined_area 11 19 10
  let combined_area2 := combined_area 9.5 11 8
  combined_area1 - combined_area2 = 136 :=
by
  let combined_area1 := combined_area 11 19 10 
  let combined_area2 := combined_area 9.5 11 8 
  show combined_area1 - combined_area2 = 136
  sorry

end trapezoid_area_difference_l179_179549


namespace total_area_correct_at_stage_5_l179_179590

def initial_side_length := 3

def side_length (n : ℕ) : ℕ := initial_side_length + n

def area (side : ℕ) : ℕ := side * side

noncomputable def total_area_at_stage_5 : ℕ :=
  (area (side_length 0)) + (area (side_length 1)) + (area (side_length 2)) + (area (side_length 3)) + (area (side_length 4))

theorem total_area_correct_at_stage_5 : total_area_at_stage_5 = 135 :=
by
  sorry

end total_area_correct_at_stage_5_l179_179590


namespace solve_absolute_value_eq_l179_179631

theorem solve_absolute_value_eq (x : ℝ) : (|x - 3| = 5 - x) → x = 4 :=
by
  sorry

end solve_absolute_value_eq_l179_179631


namespace apples_vs_cherries_l179_179031

def pies_per_day : Nat := 12
def apple_days_per_week : Nat := 3
def cherry_days_per_week : Nat := 2

theorem apples_vs_cherries :
  (apple_days_per_week * pies_per_day) - (cherry_days_per_week * pies_per_day) = 12 := by
  sorry

end apples_vs_cherries_l179_179031


namespace composite_sum_l179_179953

theorem composite_sum (a b : ℤ) (h : 56 * a = 65 * b) : ∃ m n : ℤ,  m > 1 ∧ n > 1 ∧ a + b = m * n :=
sorry

end composite_sum_l179_179953


namespace remainder_div_1356_l179_179815

theorem remainder_div_1356 :
  ∃ R : ℝ, ∃ L : ℝ, ∃ S : ℝ, S = 268.2 ∧ L - S = 1356 ∧ L = 6 * S + R ∧ R = 15 :=
by
  sorry

end remainder_div_1356_l179_179815


namespace lulu_cash_left_l179_179931

theorem lulu_cash_left :
  ∀ (initial money spentIceCream spentTshirt deposited finalCash: ℝ),
    initial = 65 →
    spentIceCream = 5 →
    spentTshirt = 0.5 * (initial - spentIceCream) →
    deposited = (initial - spentIceCream - spentTshirt) / 5 →
    finalCash = initial - spentIceCream - spentTshirt - deposited →
    finalCash = 24 :=
by
  intros
  sorry

end lulu_cash_left_l179_179931


namespace mathematician_daily_questions_l179_179243

theorem mathematician_daily_questions :
  (518 + 476) / 7 = 142 := by
  sorry

end mathematician_daily_questions_l179_179243


namespace find_general_term_l179_179282

theorem find_general_term (a : ℕ → ℕ) (h₀ : a 1 = 1) (h₁ : ∀ n, a (n + 1) = 2 * a n + n^2) :
  ∀ n, a n = 7 * 2^(n - 1) - n^2 - 2 * n - 3 :=
by
  sorry

end find_general_term_l179_179282


namespace mona_unique_players_l179_179008

theorem mona_unique_players (groups : ℕ) (players_per_group : ℕ) (repeated1 : ℕ) (repeated2 : ℕ) :
  (groups = 9) → (players_per_group = 4) → (repeated1 = 2) → (repeated2 = 1) →
  (groups * players_per_group - (repeated1 + repeated2) = 33) :=
begin
  intros h_groups h_players_per_group h_repeated1 h_repeated2,
  rw [h_groups, h_players_per_group, h_repeated1, h_repeated2],
  norm_num,
end

end mona_unique_players_l179_179008


namespace total_cost_to_plant_flowers_l179_179645

noncomputable def flower_cost : ℕ := 9
noncomputable def clay_pot_cost : ℕ := flower_cost + 20
noncomputable def soil_bag_cost : ℕ := flower_cost - 2
noncomputable def total_cost : ℕ := flower_cost + clay_pot_cost + soil_bag_cost

theorem total_cost_to_plant_flowers : total_cost = 45 := by
  sorry

end total_cost_to_plant_flowers_l179_179645


namespace sequence_periodicity_l179_179088

theorem sequence_periodicity :
  let t : ℕ → ℚ :=
    λ n, if n = 1 then 20 else if n = 2 then 21 else (5 * t (n - 1) + 1) / (25 * t (n - 2))
  in (t 2020 = 101 / 525) ∧ Nat.gcd 101 525 = 1 ∧ (101 + 525 = 626) :=
by
  sorry

end sequence_periodicity_l179_179088


namespace rate_of_interest_l179_179226

theorem rate_of_interest (R : ℝ) (h : 5000 * 2 * R / 100 + 3000 * 4 * R / 100 = 2200) : R = 10 := by
  sorry

end rate_of_interest_l179_179226


namespace total_marbles_l179_179447

variable (b : ℝ)
variable (r : ℝ) (g : ℝ)
variable (h₁ : r = 1.3 * b)
variable (h₂ : g = 1.5 * b)

theorem total_marbles (b : ℝ) (r : ℝ) (g : ℝ) (h₁ : r = 1.3 * b) (h₂ : g = 1.5 * b) : r + b + g = 3.8 * b :=
by
  sorry

end total_marbles_l179_179447


namespace integer_solutions_of_system_l179_179030

theorem integer_solutions_of_system :
  {x : ℤ | - 2 * x + 7 < 10 ∧ (7 * x + 1) / 5 - 1 ≤ x} = {-1, 0, 1, 2} :=
by
  sorry

end integer_solutions_of_system_l179_179030


namespace race_result_l179_179730

-- Defining competitors
inductive Sprinter
| A
| B
| C

open Sprinter

-- Conditions as definitions
def position_changes : Sprinter → Nat
| A => sorry
| B => 5
| C => 6

def finishes_before (s1 s2 : Sprinter) : Prop := sorry

-- Stating the problem as a theorem
theorem race_result :
  position_changes C = 6 →
  position_changes B = 5 →
  finishes_before B A →
  (finishes_before B A ∧ finishes_before A C ∧ finishes_before B C) :=
by
  intros hC hB hBA
  sorry

end race_result_l179_179730


namespace intersection_setA_setB_l179_179616

-- Define set A
def setA : Set ℝ := {x | 2 * x ≤ 4}

-- Define set B as the domain of the function y = log(x - 1)
def setB : Set ℝ := {x | x > 1}

-- Theorem to prove
theorem intersection_setA_setB : setA ∩ setB = {x | 1 < x ∧ x ≤ 2} :=
by
  sorry

end intersection_setA_setB_l179_179616


namespace ages_correct_l179_179449

def ages : List ℕ := [5, 8, 13, 15]
def Tanya : ℕ := 13
def Yura : ℕ := 8
def Sveta : ℕ := 5
def Lena : ℕ := 15

theorem ages_correct (h1 : Tanya ∈ ages) 
                     (h2: Yura ∈ ages)
                     (h3: Sveta ∈ ages)
                     (h4: Lena ∈ ages)
                     (h5: Tanya ≠ Yura)
                     (h6: Tanya ≠ Sveta)
                     (h7: Tanya ≠ Lena)
                     (h8: Yura ≠ Sveta)
                     (h9: Yura ≠ Lena)
                     (h10: Sveta ≠ Lena)
                     (h11: Sveta = 5)
                     (h12: Tanya > Yura)
                     (h13: (Tanya + Sveta) % 3 = 0) :
                     Tanya = 13 ∧ Yura = 8 ∧ Sveta = 5 ∧ Lena = 15 := by
  sorry

end ages_correct_l179_179449


namespace arc_length_of_sector_l179_179495

theorem arc_length_of_sector (r θ : ℝ) (A : ℝ) (h₁ : r = 4)
  (h₂ : A = 7) : (1 / 2) * r^2 * θ = A → r * θ = 3.5 :=
by
  sorry

end arc_length_of_sector_l179_179495


namespace value_of_a_5_l179_179433

-- Define the sequence with the general term formula
def a (n : ℕ) : ℕ := 4 * n - 3

-- Prove that the value of a_5 is 17
theorem value_of_a_5 : a 5 = 17 := by
  sorry

end value_of_a_5_l179_179433


namespace true_product_of_two_digit_number_l179_179249

theorem true_product_of_two_digit_number (a b : ℕ) (h1 : b = 2 * a) (h2 : 136 * (10 * b + a) = 136 * (10 * a + b) + 1224) : 136 * (10 * a + b) = 1632 := 
by sorry

end true_product_of_two_digit_number_l179_179249


namespace shaded_region_area_is_48pi_l179_179070

open Real

noncomputable def small_circle_radius : ℝ := 4
noncomputable def small_circle_area : ℝ := π * small_circle_radius^2
noncomputable def large_circle_radius : ℝ := 2 * small_circle_radius
noncomputable def large_circle_area : ℝ := π * large_circle_radius^2
noncomputable def shaded_region_area : ℝ := large_circle_area - small_circle_area

theorem shaded_region_area_is_48pi :
  shaded_region_area = 48 * π := by
    sorry

end shaded_region_area_is_48pi_l179_179070


namespace correct_sum_is_132_l179_179315

-- Let's define the conditions:
-- The ones digit B is mistakenly taken as 1 (when it should be 7)
-- The tens digit C is mistakenly taken as 6 (when it should be 4)
-- The incorrect sum is 146

def correct_ones_digit (mistaken_ones_digit : Nat) : Nat :=
  -- B was mistaken for 1, so B should be 7
  if mistaken_ones_digit = 1 then 7 else mistaken_ones_digit

def correct_tens_digit (mistaken_tens_digit : Nat) : Nat :=
  -- C was mistaken for 6, so C should be 4
  if mistaken_tens_digit = 6 then 4 else mistaken_tens_digit

def correct_sum (incorrect_sum : Nat) : Nat :=
  -- Correcting the sum based on the mistakes
  incorrect_sum + 6 - 20 -- 6 to correct ones mistake, minus 20 to correct tens mistake

theorem correct_sum_is_132 : correct_sum 146 = 132 :=
  by
    -- The theorem is here to check that the corrected sum equals 132
    sorry

end correct_sum_is_132_l179_179315


namespace total_bouquets_sold_l179_179714

-- Define the conditions as variables
def monday_bouquets : ℕ := 12
def tuesday_bouquets : ℕ := 3 * monday_bouquets
def wednesday_bouquets : ℕ := tuesday_bouquets / 3

-- The statement to prove
theorem total_bouquets_sold : 
  monday_bouquets + tuesday_bouquets + wednesday_bouquets = 60 :=
by
  -- The proof is omitted using sorry
  sorry

end total_bouquets_sold_l179_179714


namespace book_loss_percentage_l179_179902

theorem book_loss_percentage (CP SP_profit SP_loss : ℝ) (L : ℝ) 
  (h1 : CP = 50) 
  (h2 : SP_profit = CP + 0.09 * CP) 
  (h3 : SP_loss = CP - L / 100 * CP) 
  (h4 : SP_profit - SP_loss = 9) : 
  L = 9 :=
by
  sorry

end book_loss_percentage_l179_179902


namespace fraction_left_after_3_days_l179_179379

-- Defining work rates of A and B
def A_rate := 1 / 15
def B_rate := 1 / 20

-- Total work rate of A and B when working together
def combined_rate := A_rate + B_rate

-- Work completed by A and B in 3 days
def work_done := 3 * combined_rate

-- Fraction of work left
def fraction_work_left := 1 - work_done

-- Statement to prove:
theorem fraction_left_after_3_days : fraction_work_left = 13 / 20 :=
by
  have A_rate_def: A_rate = 1 / 15 := rfl
  have B_rate_def: B_rate = 1 / 20 := rfl
  have combined_rate_def: combined_rate = A_rate + B_rate := rfl
  have work_done_def: work_done = 3 * combined_rate := rfl
  have fraction_work_left_def: fraction_work_left = 1 - work_done := rfl
  sorry

end fraction_left_after_3_days_l179_179379


namespace cost_per_bag_l179_179830

theorem cost_per_bag (total_friends: ℕ) (amount_paid_per_friend: ℕ) (total_bags: ℕ) 
  (h1 : total_friends = 3) (h2 : amount_paid_per_friend = 5) (h3 : total_bags = 5) 
  : total_friends * amount_paid_per_friend / total_bags = 3 := by
  sorry

end cost_per_bag_l179_179830


namespace counter_example_exists_l179_179740

theorem counter_example_exists : 
  ∃ n : ℕ, n ≥ 2 ∧ ¬(∃ k : ℕ, (2 ^ 2 ^ n) % (2 ^ n - 1) = 4 ^ k) :=
  sorry

end counter_example_exists_l179_179740


namespace circle_area_l179_179267

theorem circle_area (C : ℝ) (hC : C = 31.4) : 
  ∃ (A : ℝ), A = 246.49 / π := 
by
  sorry -- proof not required

end circle_area_l179_179267


namespace twentieth_triangular_number_l179_179655

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

theorem twentieth_triangular_number : triangular_number 20 = 210 :=
by
  sorry

end twentieth_triangular_number_l179_179655


namespace original_board_length_before_final_cut_l179_179370

-- Given conditions
def initial_length : ℕ := 143
def first_cut_length : ℕ := 25
def final_cut_length : ℕ := 7

def board_length_after_first_cut : ℕ := initial_length - first_cut_length
def board_length_after_final_cut : ℕ := board_length_after_first_cut - final_cut_length

-- The theorem to be proved
theorem original_board_length_before_final_cut : board_length_after_first_cut + final_cut_length = 125 :=
by
  sorry

end original_board_length_before_final_cut_l179_179370


namespace john_buys_spools_l179_179920

theorem john_buys_spools (spool_length necklace_length : ℕ) 
  (necklaces : ℕ) 
  (total_length := necklaces * necklace_length) 
  (spools := total_length / spool_length) :
  spool_length = 20 → 
  necklace_length = 4 → 
  necklaces = 15 → 
  spools = 3 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end john_buys_spools_l179_179920


namespace ancient_chinese_problem_l179_179914

theorem ancient_chinese_problem (x y : ℤ) 
  (h1 : y = 8 * x - 3) 
  (h2 : y = 7 * x + 4) : 
  (y = 8 * x - 3) ∧ (y = 7 * x + 4) :=
by
  exact ⟨h1, h2⟩

end ancient_chinese_problem_l179_179914


namespace work_completion_days_l179_179845

theorem work_completion_days (D : ℕ) 
  (h : 40 * D = 48 * (D - 10)) : D = 60 := 
sorry

end work_completion_days_l179_179845


namespace number_of_action_figures_bought_l179_179411

-- Definitions of conditions
def cost_of_board_game : ℕ := 2
def cost_per_action_figure : ℕ := 7
def total_spent : ℕ := 30

-- The problem to prove
theorem number_of_action_figures_bought : 
  ∃ (n : ℕ), total_spent - cost_of_board_game = n * cost_per_action_figure ∧ n = 4 :=
by
  sorry

end number_of_action_figures_bought_l179_179411


namespace symmetric_point_of_P_l179_179874

-- Let P be a point with coordinates (5, -3)
def P : ℝ × ℝ := (5, -3)

-- Definition of the symmetric point with respect to the x-axis
def symmetric_point (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

-- Theorem stating that the symmetric point to P with respect to the x-axis is (5, 3)
theorem symmetric_point_of_P : symmetric_point P = (5, 3) := 
  sorry

end symmetric_point_of_P_l179_179874


namespace solve_equation_l179_179947

def equation_params (a x : ℝ) : Prop :=
  a * (1 / (Real.cos x) - Real.tan x) = 1

def valid_solutions (a x : ℝ) (k : ℤ) : Prop :=
  (a ≠ 0) ∧ (Real.cos x ≠ 0) ∧ (
    (|a| ≥ 1 ∧ x = Real.arccos (a / Real.sqrt (a * a + 1)) + 2 * Real.pi * k) ∨
    ((-1 < a ∧ a < 0) ∨ (0 < a ∧ a < 1) ∧ x = - Real.arccos (a / Real.sqrt (a * a + 1)) + 2 * Real.pi * k)
  )

theorem solve_equation (a x : ℝ) (k : ℤ) :
  equation_params a x → valid_solutions a x k := by
  sorry

end solve_equation_l179_179947


namespace non_empty_subsets_count_l179_179585

def no_consecutive (S : Set ℕ) : Prop :=
  ∀ (x ∈ S) (y ∈ S), x ≠ y + 1 ∧ x + 1 ≠ y

def valid_subset (S : Set ℕ) (k : ℕ) : Prop :=
  k = S.card ∧ (k > 0) ∧ ∀ x ∈ S, x ≥ k

theorem non_empty_subsets_count :
  ∃ S, no_consecutive S ∧ valid_subset S k → 
  count (S \in Subsets ∧ no_consecutive S ∧ valid_subset S k) = 59 :=
by
  sorry

end non_empty_subsets_count_l179_179585


namespace min_employees_birthday_Wednesday_l179_179085

theorem min_employees_birthday_Wednesday (W D : ℕ) (h_eq : W + 6 * D = 50) (h_gt : W > D) : W = 8 :=
sorry

end min_employees_birthday_Wednesday_l179_179085


namespace standard_equation_of_ellipse_midpoint_of_chord_l179_179424

variables (a b c : ℝ)
variables (x1 y1 x2 y2 : ℝ)
variables (A B : ℝ × ℝ)

axiom conditions :
  a > b ∧ b > 0 ∧
  (c / a = (Real.sqrt 6) / 3) ∧
  a = Real.sqrt 3 ∧
  a^2 = b^2 + c^2 ∧
  (A = (-1, 0)) ∧ (B = (x2, y2)) ∧
  A ≠ B ∧
  (∃ l : ℝ -> ℝ, l (-1) = 0 ∧ ∀ x, l x = x + 1) ∧
  (∃ x1 x2 y1 y2 : ℝ, x1 + x2 = -3 / 2)

theorem standard_equation_of_ellipse :
  ∃ (e : ℝ), e = 1 ∧ (x1 / 3) + y1 = 1 := sorry

theorem midpoint_of_chord :
  ∃ (m : ℝ × ℝ), m = (-(3 / 4), 1 / 4) := sorry

end standard_equation_of_ellipse_midpoint_of_chord_l179_179424


namespace triangular_pyramid_volume_l179_179269

theorem triangular_pyramid_volume
  (b : ℝ) (h : ℝ) (H : ℝ)
  (b_pos : b = 4.5) (h_pos : h = 6) (H_pos : H = 8) :
  let base_area := (b * h) / 2
  let volume := (base_area * H) / 3
  volume = 36 := by
  sorry

end triangular_pyramid_volume_l179_179269


namespace number_of_terms_in_arithmetic_sequence_l179_179090

-- Define the first term, common difference, and the nth term of the sequence
def a : ℤ := -3
def d : ℤ := 4
def a_n : ℤ := 45

-- Define the number of terms in the arithmetic sequence
def num_of_terms : ℤ := 13

-- The theorem states that for the given arithmetic sequence, the number of terms n satisfies the sequence equation
theorem number_of_terms_in_arithmetic_sequence :
  a + (num_of_terms - 1) * d = a_n :=
by
  sorry

end number_of_terms_in_arithmetic_sequence_l179_179090


namespace solve_for_m_l179_179584

theorem solve_for_m (m : ℝ) :
  (1 * m + (3 + m) * 2 = 0) → m = -2 :=
by
  sorry

end solve_for_m_l179_179584


namespace sphere_radius_twice_cone_volume_l179_179390

theorem sphere_radius_twice_cone_volume :
  ∀ (r_cone h_cone : ℝ) (r_sphere : ℝ), 
    r_cone = 2 → h_cone = 8 → 2 * (1 / 3 * Real.pi * r_cone^2 * h_cone) = (4/3 * Real.pi * r_sphere^3) → 
    r_sphere = 2^(4/3) :=
by
  intros r_cone h_cone r_sphere h_r_cone h_h_cone h_volume_equiv
  sorry

end sphere_radius_twice_cone_volume_l179_179390


namespace largest_base5_three_digit_to_base10_l179_179199

theorem largest_base5_three_digit_to_base10 : 
  let base5_number := 4 * 5^2 + 4 * 5^1 + 4 * 5^0 in
  base5_number = 124 :=
by
let base5_number := 4 * 5^2 + 4 * 5^1 + 4 * 5^0
have h1 : 4 * 5^2 = 100 := by norm_num
have h2 : 4 * 5^1 = 20 := by norm_num
have h3 : 4 * 5^0 = 4 := by norm_num
have h4 : base5_number = 100 + 20 + 4 := by simp [base5_number, h1, h2, h3]
show base5_number = 124, by simp [h4]
sorry

end largest_base5_three_digit_to_base10_l179_179199


namespace find_x_l179_179682

-- Given condition that x is 11 percent greater than 90
def eleven_percent_greater (x : ℝ) : Prop := x = 90 + (11 / 100) * 90

-- Theorem statement
theorem find_x (x : ℝ) (h: eleven_percent_greater x) : x = 99.9 :=
  sorry

end find_x_l179_179682


namespace part1_l179_179526

-- Define the vectors a and b
def a : ℝ × ℝ := (-3, 4)
def b : ℝ × ℝ := (2, -1)
-- Define the vectors a - x b and a - b
def vec1 (x : ℝ) : ℝ × ℝ := (a.1 - x * b.1, a.2 - x * b.2)
def vec2 : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)
-- Define the dot product
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2 

-- Main theorem: prove that the vectors being perpendicular implies x = -7/3
theorem part1 (x : ℝ) : dot_product (vec1 x) vec2 = 0 → x = -7 / 3 :=
by
  sorry

end part1_l179_179526


namespace stratified_sampling_admin_staff_l179_179723

theorem stratified_sampling_admin_staff
  (total_employees : ℕ)
  (sales_people : ℕ)
  (admin_staff : ℕ)
  (sample_size : ℕ)
  (proportion : ℚ)
  (admin_sample_size : ℕ)
  (h1 : total_employees = 120)
  (h2 : sales_people = 100)
  (h3 : admin_staff = 20)
  (h4 : sample_size = 12)
  (h5 : proportion = (admin_staff : ℚ) / (total_employees : ℚ))
  (h6 : admin_sample_size = (proportion * sample_size).to_nat) :
  admin_sample_size = 2 := 
sorry

end stratified_sampling_admin_staff_l179_179723


namespace option1_cheaper_when_x_30_more_cost_effective_plan_when_x_30_l179_179687

noncomputable def payment_option1 (x : ℕ) (h : x > 20) : ℝ :=
  200 * (x : ℝ) + 16000

noncomputable def payment_option2 (x : ℕ) (h : x > 20) : ℝ :=
  180 * (x : ℝ) + 18000

theorem option1_cheaper_when_x_30 :
  payment_option1 30 (by norm_num) < payment_option2 30 (by norm_num) :=
by sorry

theorem more_cost_effective_plan_when_x_30 :
  20000 + (0.9 * (10 * 200)) < payment_option1 30 (by norm_num) :=
by sorry

end option1_cheaper_when_x_30_more_cost_effective_plan_when_x_30_l179_179687


namespace compare_abc_l179_179278

noncomputable def a : ℝ := 1 / (1 + Real.exp 2)
noncomputable def b : ℝ := 1 / Real.exp 1
noncomputable def c : ℝ := Real.log ((1 + Real.exp 2) / (Real.exp 2))

theorem compare_abc : b > c ∧ c > a := by
  sorry

end compare_abc_l179_179278


namespace complex_number_real_iff_value_of_x_l179_179593

theorem complex_number_real_iff_value_of_x (x : ℝ) :
  (log 2 (x ^ 2 - 3 * x - 3) + complex.I * log 2 (x - 3)).im = 0 →
  x ^ 2 - 3 * x - 3 > 0 → 
  x = 4 :=
by
  sorry

end complex_number_real_iff_value_of_x_l179_179593


namespace sum_is_two_l179_179806

-- Define the numbers based on conditions
def a : Int := 9
def b : Int := -9 + 2

-- Theorem stating that the sum of the two numbers is 2
theorem sum_is_two : a + b = 2 :=
by
  -- proof goes here
  sorry

end sum_is_two_l179_179806


namespace smallest_positive_debt_resolved_l179_179963

theorem smallest_positive_debt_resolved : ∃ (D : ℕ), D > 0 ∧ (∃ (p g : ℤ), D = 400 * p + 240 * g) ∧ D = 80 := by
  sorry

end smallest_positive_debt_resolved_l179_179963


namespace largest_base5_three_digit_to_base10_l179_179201

theorem largest_base5_three_digit_to_base10 :
  (4 * 5^2 + 4 * 5^1 + 4 * 5^0) = 124 :=
by
  sorry

end largest_base5_three_digit_to_base10_l179_179201


namespace analogy_reasoning_conducts_electricity_l179_179180

theorem analogy_reasoning_conducts_electricity (Gold Silver Copper Iron : Prop) (conducts : Prop)
  (h1 : Gold) (h2 : Silver) (h3 : Copper) (h4 : Iron) :
  (Gold ∧ Silver ∧ Copper ∧ Iron → conducts) → (conducts → !CompleteInductive ∧ !Inductive ∧ !Deductive ∧ Analogical) :=
by
  sorry

end analogy_reasoning_conducts_electricity_l179_179180


namespace original_amount_of_solution_y_l179_179629

theorem original_amount_of_solution_y (Y : ℝ) 
  (h1 : 0 < Y) -- We assume Y > 0 
  (h2 : 0.3 * (Y - 4) + 1.2 = 0.45 * Y) :
  Y = 8 := 
sorry

end original_amount_of_solution_y_l179_179629


namespace equal_cubes_l179_179436

theorem equal_cubes (r s : ℤ) (hr : 0 ≤ r) (hs : 0 ≤ s)
  (h : |r^3 - s^3| = |6 * r^2 - 6 * s^2|) : r = s :=
by
  sorry

end equal_cubes_l179_179436


namespace max_cards_with_digit_three_l179_179190

/-- There are ten cards each of the digits "3", "4", and "5". Choose any 8 cards such that their sum is 27. 
Prove that the maximum number of these cards that can be "3" is 6. -/
theorem max_cards_with_digit_three (c3 c4 c5 : ℕ) (hc3 : c3 + c4 + c5 = 8) (h_sum : 3 * c3 + 4 * c4 + 5 * c5 = 27) :
  c3 ≤ 6 :=
sorry

end max_cards_with_digit_three_l179_179190


namespace pascal_triangle_row_20_element_5_l179_179050

theorem pascal_triangle_row_20_element_5 : binomial 20 4 = 4845 := 
by sorry

end pascal_triangle_row_20_element_5_l179_179050


namespace father_seven_times_as_old_l179_179535

theorem father_seven_times_as_old (x : ℕ) (father_age : ℕ) (son_age : ℕ) :
  father_age = 38 → son_age = 14 → (father_age - x = 7 * (son_age - x) → x = 10) :=
by
  intros h_father_age h_son_age h_equation
  rw [h_father_age, h_son_age] at h_equation
  sorry

end father_seven_times_as_old_l179_179535


namespace sequence_problem_proof_l179_179488

-- Define the sequence terms, using given conditions
def a_1 : ℕ := 1
def a_2 : ℕ := 2
def a_3 : ℕ := a_1 + a_2
def a_4 : ℕ := a_2 + a_3
def x : ℕ := a_3 + a_4

-- Prove that x = 8
theorem sequence_problem_proof : x = 8 := 
by
  sorry

end sequence_problem_proof_l179_179488


namespace properSubsets_31_l179_179891

open Nat

def A : Set ℕ := {x ∈ ℕ | ∃ k ∈ ℕ, 12 = k * (6 - x)}

def properSubsetCount (s : Finset ℕ) : ℕ :=
  2^s.card - 1

theorem properSubsets_31 : properSubsetCount (A.to_finset) = 31 := by
  sorry

end properSubsets_31_l179_179891


namespace find_m_l179_179767

def triangle (x y : ℤ) := x * y + x + y

theorem find_m (m : ℤ) (h : triangle 2 m = -16) : m = -6 :=
by
  sorry

end find_m_l179_179767


namespace Timi_has_five_ears_l179_179191

theorem Timi_has_five_ears (seeing_ears_Imi seeing_ears_Dimi seeing_ears_Timi : ℕ)
  (H1 : seeing_ears_Imi = 8)
  (H2 : seeing_ears_Dimi = 7)
  (H3 : seeing_ears_Timi = 5)
  (total_ears : ℕ := (seeing_ears_Imi + seeing_ears_Dimi + seeing_ears_Timi) / 2) :
  total_ears - seeing_ears_Timi = 5 :=
by
  sorry -- Proof not required.

end Timi_has_five_ears_l179_179191


namespace segment_length_OI_is_3_l179_179516

-- Define the points along the path
def point (n : ℕ) : ℝ × ℝ := (n, n)

-- Use the Pythagorean theorem to calculate the distance from point O to point I
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Define the points O and I
def O : ℝ × ℝ := point 0
def I : ℝ × ℝ := point 3

-- The proposition to prove: 
-- The distance between points O and I is 3
theorem segment_length_OI_is_3 : distance O I = 3 := 
  sorry

end segment_length_OI_is_3_l179_179516


namespace hens_to_roosters_multiplier_l179_179992

def totalChickens : ℕ := 75
def numHens : ℕ := 67

-- Given the total number of chickens and a certain relationship
theorem hens_to_roosters_multiplier
  (numRoosters : ℕ) (multiplier : ℕ)
  (h1 : totalChickens = numHens + numRoosters)
  (h2 : numHens = multiplier * numRoosters - 5) :
  multiplier = 9 :=
by sorry

end hens_to_roosters_multiplier_l179_179992


namespace find_k_and_b_l179_179001

variables (k b : ℝ)

def f (p : ℝ × ℝ) : ℝ × ℝ :=
  (k * p.1, p.2 + b)

theorem find_k_and_b
  (h : f k b (6, 2) = (3, 1)) :
  k = 2 ∧ b = -1 :=
by {
  -- proof steps would go here
  sorry
}

end find_k_and_b_l179_179001


namespace percentage_increase_is_50_l179_179232

-- Definition of the given values
def original_time : ℕ := 30
def new_time : ℕ := 45

-- Assertion stating that the percentage increase is 50%
theorem percentage_increase_is_50 :
  (new_time - original_time) * 100 / original_time = 50 := 
sorry

end percentage_increase_is_50_l179_179232


namespace gcd_positive_ints_l179_179921

theorem gcd_positive_ints (a b c d : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) 
  (hdiv : (a^2 + b^2) ∣ (a * c + b * d)) : 
  Nat.gcd (c^2 + d^2) (a^2 + b^2) > 1 := 
sorry

end gcd_positive_ints_l179_179921


namespace max_m_for_factored_polynomial_l179_179089

theorem max_m_for_factored_polynomial :
  ∃ m, (∀ A B : ℤ, (5 * x ^ 2 + m * x + 45 = (5 * x + A) * (x + B) → AB = 45) → 
    m = 226) :=
sorry

end max_m_for_factored_polynomial_l179_179089


namespace maximum_value_of_f_on_interval_l179_179819

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * x + Real.sin x

theorem maximum_value_of_f_on_interval :
  ∃ M, M = Real.pi ∧ ∀ x ∈ Set.Icc 0 (2 * Real.pi), f x ≤ M :=
by
  sorry

end maximum_value_of_f_on_interval_l179_179819


namespace no_real_roots_of_quadratic_l179_179879

theorem no_real_roots_of_quadratic 
  (a b c : ℝ) 
  (h1 : b - a + c > 0) 
  (h2 : b + a - c > 0) 
  (h3 : b - a - c < 0) 
  (h4 : b + a + c > 0) 
  (x : ℝ) : ¬ ∃ x : ℝ, a^2 * x^2 + (b^2 - a^2 - c^2) * x + c^2 = 0 := 
by
  sorry

end no_real_roots_of_quadratic_l179_179879


namespace find_square_side_length_l179_179174

noncomputable def side_length_PQRS (x : ℝ) : Prop :=
  let PT := 1
  let QU := 2
  let RV := 3
  let SW := 4
  let PQRS_area := x^2
  let TUVW_area := 1 / 2 * x^2
  let triangle_area (base height : ℝ) : ℝ := 1 / 2 * base * height
  PQRS_area = x^2 ∧ TUVW_area = 1 / 2 * x^2 ∧
  triangle_area 1 (x - 4) + (x - 1) + 
  triangle_area 3 (x - 2) + 2 * (x - 3) = 1 / 2 * x^2

theorem find_square_side_length : ∃ x : ℝ, side_length_PQRS x ∧ x = 6 := 
  sorry

end find_square_side_length_l179_179174


namespace find_circle_radius_l179_179499

-- Definitions of given distances and the parallel chord condition
def isChordParallelToDiameter (c d : ℝ × ℝ) (radius distance1 distance2 : ℝ) : Prop :=
  let p1 := distance1
  let p2 := distance2
  p1 = 5 ∧ p2 = 12 ∧ 
  -- Assuming distances from the end of the diameter to the ends of the chord
  true

-- The main theorem which states the radius of the circle given the conditions
theorem find_circle_radius
  (diameter chord : ℝ × ℝ)
  (R p1 p2 : ℝ)
  (h1 : isChordParallelToDiameter diameter chord R p1 p2) :
  R = 6.5 :=
  by
    sorry

end find_circle_radius_l179_179499


namespace kevin_exchanges_l179_179613

variables (x y : ℕ)

def R (x y : ℕ) := 100 - 3 * x + 2 * y
def B (x y : ℕ) := 100 + 2 * x - 4 * y

theorem kevin_exchanges :
  (∃ x y, R x y >= 3 ∧ B x y >= 4 ∧ x + y = 132) :=
sorry

end kevin_exchanges_l179_179613


namespace area_triangle_MNR_l179_179316

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

/-- Given the quadrilateral PQRS with the midpoints M and N of PQ and QR 
and specified lengths, prove the calculated area of triangle MNR. -/
theorem area_triangle_MNR : 
  let P : (ℝ × ℝ) := (0, 5)
  let Q : (ℝ × ℝ) := (10, 5)
  let R : (ℝ × ℝ) := (14, 0)
  let S : (ℝ × ℝ) := (7, 0)
  let M : (ℝ × ℝ) := (5, 5)  -- Midpoint of PQ
  let N : (ℝ × ℝ) := (12, 2.5) -- Midpoint of QR
  distance M.fst M.snd N.fst N.snd = 7.435 →
  ((5 - 0 : ℝ) / 2 = 2.5) →
  (1 / 2 * 7.435 * 2.5) = 9.294375 :=
by
  sorry

end area_triangle_MNR_l179_179316


namespace sum_y_coordinates_of_other_vertices_l179_179296

theorem sum_y_coordinates_of_other_vertices (x1 y1 x2 y2 : ℤ) 
  (h1 : (x1, y1) = (2, 10)) (h2 : (x2, y2) = (-6, -6)) :
  (∃ y3 y4 : ℤ, (4 : ℤ) = y3 + y4) :=
by
  sorry

end sum_y_coordinates_of_other_vertices_l179_179296


namespace simplify_expression_l179_179626

variable (x : ℝ) (h : x ≠ 0)

theorem simplify_expression : (2 * x)⁻¹ + 2 = (1 + 4 * x) / (2 * x) :=
by
  sorry

end simplify_expression_l179_179626


namespace problem_l179_179293

-- Definitions
def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 4}
def N : Set ℕ := {2, 3}
def complement_U (S : Set ℕ) : Set ℕ := U \ S

theorem problem : ((complement_U M) ∩ (complement_U N)) = {5} :=
by
  sorry

end problem_l179_179293


namespace algebra_geometry_probabilities_l179_179131

theorem algebra_geometry_probabilities :
  let total := 5
  let algebra := 2
  let geometry := 3
  let prob_first_algebra := algebra / total
  let prob_second_geometry_after_algebra := geometry / (total - 1)
  let prob_both := prob_first_algebra * prob_second_geometry_after_algebra
  let total_after_first_algebra := total - 1
  let remaining_geometry := geometry
  prob_both = 3 / 10 ∧ remaining_geometry / total_after_first_algebra = 3 / 4 :=
by
  sorry

end algebra_geometry_probabilities_l179_179131


namespace european_stamps_cost_l179_179982

def prices : String → ℕ 
| "Italy"   => 7
| "Japan"   => 7
| "Germany" => 5
| "China"   => 5
| _ => 0

def stamps_1950s : String → ℕ 
| "Italy"   => 5
| "Germany" => 8
| "China"   => 10
| "Japan"   => 6
| _ => 0

def stamps_1960s : String → ℕ 
| "Italy"   => 9
| "Germany" => 12
| "China"   => 5
| "Japan"   => 10
| _ => 0

def total_cost (stamps : String → ℕ) (price : String → ℕ) : ℕ :=
  (stamps "Italy" * price "Italy" +
   stamps "Germany" * price "Germany") 

theorem european_stamps_cost : total_cost stamps_1950s prices + total_cost stamps_1960s prices = 198 :=
by
  sorry

end european_stamps_cost_l179_179982


namespace avg_new_students_l179_179950

-- Definitions for conditions
def orig_strength : ℕ := 17
def orig_avg_age : ℕ := 40
def new_students_count : ℕ := 17
def decreased_avg_age : ℕ := 36 -- given that average decreases by 4 years, i.e., 40 - 4

-- Definition for the original total age
def total_age_orig : ℕ := orig_strength * orig_avg_age

-- Definition for the total number of students after new students join
def total_students : ℕ := orig_strength + new_students_count

-- Definition for the total age after new students join
def total_age_new : ℕ := total_students * decreased_avg_age

-- Definition for the total age of new students
def total_age_new_students : ℕ := total_age_new - total_age_orig

-- Definition for the average age of new students
def avg_age_new_students : ℕ := total_age_new_students / new_students_count

-- Lean theorem stating the proof problem
theorem avg_new_students : 
  avg_age_new_students = 32 := 
by sorry

end avg_new_students_l179_179950


namespace negation_P_l179_179653

-- Define the proposition P
def P (m : ℤ) : Prop := ∃ x : ℤ, 2 * x^2 + x + m ≤ 0

-- Define the negation of the proposition P
theorem negation_P (m : ℤ) : ¬P m ↔ ∀ x : ℤ, 2 * x^2 + x + m > 0 :=
by
  sorry

end negation_P_l179_179653


namespace ratio_Umar_Yusaf_l179_179724

variable (AliAge YusafAge UmarAge : ℕ)

-- Given conditions:
def Ali_is_8_years_old : Prop := AliAge = 8
def Ali_is_3_years_older_than_Yusaf : Prop := AliAge = YusafAge + 3
def Umar_is_10_years_old : Prop := UmarAge = 10

-- Proof statement:
theorem ratio_Umar_Yusaf (h1 : Ali_is_8_years_old AliAge)
                         (h2 : Ali_is_3_years_older_than_Yusaf AliAge YusafAge)
                         (h3 : Umar_is_10_years_old UmarAge) :
  UmarAge / YusafAge = 2 :=
by
  sorry

end ratio_Umar_Yusaf_l179_179724


namespace quadratic_has_non_real_roots_l179_179262

theorem quadratic_has_non_real_roots (c : ℝ) (h : c > 16) :
    ∃ (a b : ℂ), (x^2 - 8 * x + c = 0) = (a * a = -1) ∧ (b * b = -1) :=
sorry

end quadratic_has_non_real_roots_l179_179262


namespace keanu_total_spending_l179_179611

-- Definitions based on conditions
def dog_fish : Nat := 40
def cat_fish : Nat := dog_fish / 2
def total_fish : Nat := dog_fish + cat_fish
def cost_per_fish : Nat := 4
def total_cost : Nat := total_fish * cost_per_fish

-- Theorem statement
theorem keanu_total_spending : total_cost = 240 :=
by 
    sorry

end keanu_total_spending_l179_179611


namespace lucy_found_shells_l179_179801

theorem lucy_found_shells (original current : ℕ) (h1 : original = 68) (h2 : current = 89) : current - original = 21 :=
by {
    sorry
}

end lucy_found_shells_l179_179801


namespace original_length_before_final_cut_l179_179373

-- Defining the initial length of the board
def initial_length : ℕ := 143

-- Defining the length after the first cut
def length_after_first_cut : ℕ := initial_length - 25

-- Defining the length after the final cut
def length_after_final_cut : ℕ := length_after_first_cut - 7

-- Stating the theorem to prove that the original length of the board before cutting the final 7 cm is 125 cm
theorem original_length_before_final_cut : initial_length - 25 + 7 = 125 :=
sorry

end original_length_before_final_cut_l179_179373


namespace lowest_fraction_by_two_people_l179_179522

theorem lowest_fraction_by_two_people 
  (rate_A rate_B rate_C : ℚ)
  (hA : rate_A = 1 / 4) 
  (hB : rate_B = 1 / 6) 
  (hC : rate_C = 1 / 8) : 
  ∃ (r : ℚ), r = 7 / 24 ∧ 
    ∀ (r1 r2 : ℚ), (r1 = rate_A ∧ r2 = rate_B ∨ r1 = rate_A ∧ r2 = rate_C ∨ r1 = rate_B ∧ r2 = rate_C) → 
      r ≤ r1 + r2 := 
sorry

end lowest_fraction_by_two_people_l179_179522


namespace fifth_element_row_20_pascal_triangle_l179_179056

theorem fifth_element_row_20_pascal_triangle : binom 20 4 = 4845 :=
by 
  sorry

end fifth_element_row_20_pascal_triangle_l179_179056


namespace grade12_sample_size_correct_l179_179850

-- Given conditions
def grade10_students : ℕ := 1200
def grade11_students : ℕ := 900
def grade12_students : ℕ := 1500
def total_sample_size : ℕ := 720
def total_students : ℕ := grade10_students + grade11_students + grade12_students

-- Stratified sampling calculation
def fraction_grade12 : ℚ := grade12_students / total_students
def number_grade12_in_sample : ℚ := fraction_grade12 * total_sample_size

-- Main theorem
theorem grade12_sample_size_correct :
  number_grade12_in_sample = 300 := by
  sorry

end grade12_sample_size_correct_l179_179850


namespace trig_identity_proof_l179_179754

theorem trig_identity_proof :
  let sin_95 := Real.cos (Real.pi / 36)
  let sin_65 := Real.cos (5 * Real.pi / 36)
  (Real.sin (Real.pi / 36) * Real.sin (5 * Real.pi / 36) - sin_95 * sin_65) = - (Real.sqrt 3) / 2 :=
by
  let sin_95 := Real.cos (Real.pi / 36)
  let sin_65 := Real.cos (5 * Real.pi / 36)
  sorry

end trig_identity_proof_l179_179754


namespace new_concentration_l179_179250

def vessel1 := (3 : ℝ)  -- 3 litres
def conc1 := (0.25 : ℝ) -- 25% alcohol

def vessel2 := (5 : ℝ)  -- 5 litres
def conc2 := (0.40 : ℝ) -- 40% alcohol

def vessel3 := (7 : ℝ)  -- 7 litres
def conc3 := (0.60 : ℝ) -- 60% alcohol

def vessel4 := (4 : ℝ)  -- 4 litres
def conc4 := (0.15 : ℝ) -- 15% alcohol

def total_volume := (25 : ℝ) -- Total vessel capacity

noncomputable def alcohol_total : ℝ :=
  (vessel1 * conc1) + (vessel2 * conc2) + (vessel3 * conc3) + (vessel4 * conc4)

theorem new_concentration : (alcohol_total / total_volume = 0.302) :=
  sorry

end new_concentration_l179_179250


namespace sequence_an_general_formula_and_sum_bound_l179_179618

theorem sequence_an_general_formula_and_sum_bound (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (b : ℕ → ℝ)
  (T : ℕ → ℝ)
  (h1 : ∀ n, S n = (1 / 4) * (a n + 1) ^ 2)
  (h2 : ∀ n, b n = 1 / (a n * a (n + 1)))
  (h3 : ∀ n, T n = (1 / 2) * (1 - (1 / (2 * n + 1))))
  (h4 : ∀ n, 0 < a n) :
  (∀ n, a n = 2 * n - 1) ∧ (∀ n, T n < 1 / 2) := 
by
  sorry

end sequence_an_general_formula_and_sum_bound_l179_179618


namespace number_of_three_digit_integers_l179_179430

-- Defining the set of available digits
def digits : List ℕ := [3, 5, 8, 9]

-- Defining the property for selecting a digit without repetition
def no_repetition (l : List ℕ) : Prop :=
  ∀ (d : ℕ), d ∈ l → l.filter (fun x => x = d) = [d]

-- The main theorem stating the number of three-digit integers that can be formed
theorem number_of_three_digit_integers (h : no_repetition digits) : 
  ∃ n : ℕ, n = 24 :=
by
  sorry

end number_of_three_digit_integers_l179_179430


namespace roots_of_quadratic_l179_179594

theorem roots_of_quadratic (a b c : ℝ) (h₀ : a ≠ 0) (h₁ : a + b + c = 0) (h₂ : a - b + c = 0) :
  (a * (1 : ℝ)^2 + b * (1 : ℝ) + c = 0) ∧ (a * (-1 : ℝ)^2 + b * (-1 : ℝ) + c = 0) :=
sorry

end roots_of_quadratic_l179_179594


namespace points_satisfy_diamond_eq_l179_179261

noncomputable def diamond (a b : ℝ) : ℝ := a^3 * b - a * b^3

theorem points_satisfy_diamond_eq (x y : ℝ) :
  (diamond x y = diamond y x) ↔ ((x = 0) ∨ (y = 0) ∨ (x = y) ∨ (x = -y)) := 
by
  sorry

end points_satisfy_diamond_eq_l179_179261


namespace negation_of_p_l179_179005

theorem negation_of_p : (¬ ∃ x : ℕ, x^2 > 4^x) ↔ (∀ x : ℕ, x^2 ≤ 4^x) :=
by
  sorry

end negation_of_p_l179_179005


namespace part1_part2_part3_l179_179944

def folklore {a b m n : ℤ} (h1 : a + b * Real.sqrt 3 = (m + n * Real.sqrt 3) ^ 2) : Prop :=
  a = m ^ 2 + 3 * n ^ 2 ∧ b = 2 * m * n

theorem part1 (a b m n : ℤ) (h : a + b * Real.sqrt 3 = (m + n * Real.sqrt 3) ^ 2) : a = m ^ 2 + 3 * n ^ 2 ∧ b = 2 * m * n :=
by sorry

theorem part2 : 13 + 4 * Real.sqrt 3 = (1 + 2 * Real.sqrt 3) ^ 2 :=
by sorry

theorem part3 (a m n : ℤ) (h1 : 4 = 2 * m * n) (h2 : a + 4 * Real.sqrt 3 = (m + n * Real.sqrt 3) ^ 2) : a = 7 ∨ a = 13 :=
by sorry

end part1_part2_part3_l179_179944


namespace compute_R_at_3_l179_179923

def R (x : ℝ) := 3 * x ^ 4 + x ^ 3 + x ^ 2 + x + 1

theorem compute_R_at_3 : R 3 = 283 := by
  sorry

end compute_R_at_3_l179_179923


namespace maxCubeSideLength_correct_maxRectParallelepipedDims_correct_l179_179466

noncomputable def maxCubeSideLength (a b c : ℝ) : ℝ :=
  a * b * c / (a * b + b * c + a * c)

noncomputable def maxRectParallelepipedDims (a b c : ℝ) : ℝ × ℝ × ℝ :=
  (a / 3, b / 3, c / 3)

theorem maxCubeSideLength_correct (a b c : ℝ) :
  a > 0 → b > 0 → c > 0 →
  maxCubeSideLength a b c = a * b * c / (a * b + b * c + a * c) :=
sorry

theorem maxRectParallelepipedDims_correct (a b c : ℝ) :
  a > 0 → b > 0 → c > 0 →
  maxRectParallelepipedDims a b c = (a / 3, b / 3, c / 3) :=
sorry

end maxCubeSideLength_correct_maxRectParallelepipedDims_correct_l179_179466


namespace sequence_a_10_value_l179_179917

theorem sequence_a_10_value : 
  ∀ (a : ℕ → ℤ), a 1 = 3 → (∀ n : ℕ, 0 < n → a (n + 1) - a n = 2) → a 10 = 21 := 
by 
  intros a h1 hdiff
  sorry

end sequence_a_10_value_l179_179917


namespace minimum_games_l179_179690

theorem minimum_games (n : ℕ) : 
  (∃ (w b : ℕ) (W B : ℕ → Prop),
    (∀ i, i < 15 → W i) ∧
    (∀ j, j < 20 → B j) ∧
    w = 15 ∧
    b = 20 ∧
    (∀ i j, W i → B j → ∃ g, g < 300 ∧ g = i * b + j * w) ∧
    (∃ g, g ≤ 300 ∧ (n > 280 → ∃ i, W i ∧ i = g div b))): n = 280 :=
sorry

end minimum_games_l179_179690


namespace vanya_first_place_l179_179084

theorem vanya_first_place {n : ℕ} {E A : Finset ℕ} (e_v : ℕ) (a_v : ℕ)
  (he_v : e_v = n)
  (h_distinct_places : E.card = (E ∪ A).card)
  (h_all_worse : ∀ e_i ∈ E, e_i ≠ e_v → ∃ a_i ∈ A, a_i > e_i)
  : a_v = 1 := 
sorry

end vanya_first_place_l179_179084


namespace motorcycles_count_l179_179909

/-- 
Prove that the number of motorcycles in the parking lot is 28 given the conditions:
1. Each car has 5 wheels (including one spare).
2. Each motorcycle has 2 wheels.
3. Each tricycle has 3 wheels.
4. There are 19 cars in the parking lot.
5. There are 11 tricycles in the parking lot.
6. Altogether all vehicles have 184 wheels.
-/
theorem motorcycles_count 
  (cars := 19) 
  (tricycles := 11) 
  (total_wheels := 184) 
  (wheels_per_car := 5) 
  (wheels_per_tricycle := 3) 
  (wheels_per_motorcycle := 2) :
  (184 - (19 * 5 + 11 * 3)) / 2 = 28 :=
by 
  sorry

end motorcycles_count_l179_179909


namespace roots_of_quadratic_l179_179507

theorem roots_of_quadratic (x : ℝ) : x^2 - 5 * x = 0 ↔ (x = 0 ∨ x = 5) := by 
  sorry

end roots_of_quadratic_l179_179507


namespace sandbox_volume_correct_l179_179722

/- Define the dimensions of the sandbox -/
def Length : ℤ := 312
def Width  : ℤ := 146
def Depth  : ℤ := 56

/- Define the volume of the sandbox -/
def Volume : ℤ := Length * Width * Depth

/- Theorem stating that the volume is equal to 2,555,520 cubic centimeters -/
theorem sandbox_volume_correct : Volume = 2555520 :=
by sorry

end sandbox_volume_correct_l179_179722


namespace can_predict_at_280_l179_179691

-- Definitions based on the problem conditions
def whiteRookStudents : Nat := 15
def blackElephantStudents : Nat := 20
def totalGames : Nat := whiteRookStudents * blackElephantStudents

-- Predicate to determine if Sasha can predict a participant
def canPredictNextParticipant (n : Nat) : Prop :=
  n >= 280

theorem can_predict_at_280 :
  ∀ n, n = 280 → canPredictNextParticipant n :=
begin
  intros,
  unfold canPredictNextParticipant,
  exact Nat.ge_of_eq (Eq.symm a),
end

end can_predict_at_280_l179_179691


namespace mark_garden_total_flowers_l179_179338

theorem mark_garden_total_flowers :
  let yellow := 10
  let purple := yellow + (80 / 100) * yellow
  let total_yellow_purple := yellow + purple
  let green := (25 / 100) * total_yellow_purple
  total_yellow_purple + green = 35 :=
by
  let yellow := 10
  let purple := yellow + (80 / 100) * yellow
  let total_yellow_purple := yellow + purple
  let green := (25 / 100) * total_yellow_purple
  simp [yellow, purple, total_yellow_purple, green]
  sorry

end mark_garden_total_flowers_l179_179338


namespace sum_mod_18_l179_179734

theorem sum_mod_18 :
  (65 + 66 + 67 + 68 + 69 + 70 + 71 + 72) % 18 = 8 :=
by
  sorry

end sum_mod_18_l179_179734


namespace ratio_of_M_to_R_l179_179125

variable (M Q P N R : ℝ)

theorem ratio_of_M_to_R :
      M = 0.40 * Q →
      Q = 0.25 * P →
      N = 0.60 * P →
      R = 0.30 * N →
      M / R = 5 / 9 := by
  sorry

end ratio_of_M_to_R_l179_179125


namespace Sawyer_cleans_in_6_hours_l179_179678

theorem Sawyer_cleans_in_6_hours (N : ℝ) (S : ℝ) (h1 : S = (2/3) * N) 
                                 (h2 : 1/S + 1/N = 1/3.6) : S = 6 :=
by
  sorry

end Sawyer_cleans_in_6_hours_l179_179678


namespace total_milk_in_a_week_l179_179711

theorem total_milk_in_a_week (cows : ℕ) (milk_per_cow_per_day : ℕ) (days_in_week : ℕ) (total_milk : ℕ) 
(h_cows : cows = 52) (h_milk_per_cow_per_day : milk_per_cow_per_day = 5) 
(h_days_in_week : days_in_week = 7) (h_total_milk : total_milk = 1820) : 
(cows * milk_per_cow_per_day * days_in_week) = total_milk :=
by simp [h_cows, h_milk_per_cow_per_day, h_days_in_week, h_total_milk]; sorry

end total_milk_in_a_week_l179_179711


namespace equation1_solution_equation2_solution_l179_179029

-- Equation 1: x^2 + 2x - 8 = 0 has solutions x = -4 and x = 2.
theorem equation1_solution (x : ℝ) : x^2 + 2 * x - 8 = 0 ↔ x = -4 ∨ x = 2 := by
  sorry

-- Equation 2: 2(x+3)^2 = x(x+3) has solutions x = -3 and x = -6.
theorem equation2_solution (x : ℝ) : 2 * (x + 3)^2 = x * (x + 3) ↔ x = -3 ∨ x = -6 := by
  sorry

end equation1_solution_equation2_solution_l179_179029


namespace ice_cream_cones_sold_l179_179941

theorem ice_cream_cones_sold (T W : ℕ) (h1 : W = 2 * T) (h2 : T + W = 36000) : T = 12000 :=
by
  sorry

end ice_cream_cones_sold_l179_179941


namespace pyramid_inscribed_sphere_radius_l179_179194

noncomputable def inscribed_sphere_radius (a : ℝ) : ℝ := 
a * Real.sqrt 2 / (2 * (2 + Real.sqrt 3))

theorem pyramid_inscribed_sphere_radius (a : ℝ) (h1 : a > 0) : 
  inscribed_sphere_radius a = a * Real.sqrt 2 / (2 * (2 + Real.sqrt 3)) :=
by
  sorry

end pyramid_inscribed_sphere_radius_l179_179194


namespace smallest_N_l179_179361

noncomputable def is_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n
  
noncomputable def is_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k * k = n

noncomputable def is_fifth_power (n : ℕ) : Prop :=
  ∃ k : ℕ, k ^ 5 = n

theorem smallest_N :
  ∃ N : ℕ, is_square (N / 2) ∧ is_cube (N / 3) ∧ is_fifth_power (N / 5) ∧
  N = 2^15 * 3^10 * 5^6 :=
by
  exists 2^15 * 3^10 * 5^6
  sorry

end smallest_N_l179_179361


namespace B_share_is_correct_l179_179999

open Real

noncomputable def total_money : ℝ := 10800
noncomputable def ratio_A : ℝ := 0.5
noncomputable def ratio_B : ℝ := 1.5
noncomputable def ratio_C : ℝ := 2.25
noncomputable def ratio_D : ℝ := 3.5
noncomputable def ratio_E : ℝ := 4.25
noncomputable def total_ratio : ℝ := ratio_A + ratio_B + ratio_C + ratio_D + ratio_E
noncomputable def value_per_part : ℝ := total_money / total_ratio
noncomputable def B_share : ℝ := ratio_B * value_per_part

theorem B_share_is_correct : B_share = 1350 := by 
  sorry

end B_share_is_correct_l179_179999


namespace factorization_of_polynomial_solve_quadratic_equation_l179_179066

-- Problem 1: Factorization
theorem factorization_of_polynomial : ∀ y : ℝ, 2 * y^2 - 8 = 2 * (y + 2) * (y - 2) :=
by
  intro y
  sorry

-- Problem 2: Solving the quadratic equation
theorem solve_quadratic_equation : ∀ x : ℝ, x^2 + 4 * x + 3 = 0 ↔ x = -1 ∨ x = -3 :=
by
  intro x
  sorry

end factorization_of_polynomial_solve_quadratic_equation_l179_179066


namespace lab_tech_items_l179_179661

theorem lab_tech_items (num_uniforms : ℕ) (num_coats : ℕ) (num_techs : ℕ) (total_items : ℕ)
  (h_uniforms : num_uniforms = 12)
  (h_coats : num_coats = 6 * num_uniforms)
  (h_techs : num_techs = num_uniforms / 2)
  (h_total : total_items = num_coats + num_uniforms) :
  total_items / num_techs = 14 :=
by
  -- Placeholder for proof, ensuring theorem builds correctly.
  sorry

end lab_tech_items_l179_179661


namespace solution_of_inequality_system_l179_179127

theorem solution_of_inequality_system (a b : ℝ) 
    (h1 : 4 - 2 * a = 0)
    (h2 : (3 + b) / 2 = 1) : a + b = 1 := 
by 
  sorry

end solution_of_inequality_system_l179_179127


namespace no_positive_integer_has_product_as_perfect_square_l179_179093

theorem no_positive_integer_has_product_as_perfect_square:
  ¬ ∃ n : ℕ, n > 0 ∧ ∃ k : ℕ, n * (n + 1) = k * k :=
by
  sorry

end no_positive_integer_has_product_as_perfect_square_l179_179093


namespace loss_percentage_l179_179251

theorem loss_percentage (CP SP SP_new : ℝ) (L : ℝ) 
  (h1 : CP = 1428.57)
  (h2 : SP = CP - (L / 100 * CP))
  (h3 : SP_new = CP + 0.04 * CP)
  (h4 : SP_new = SP + 200) :
  L = 10 := by
    sorry

end loss_percentage_l179_179251


namespace square_number_n_value_l179_179587

theorem square_number_n_value
  (n : ℕ)
  (h : ∃ k : ℕ, 2^6 + 2^9 + 2^n = k^2) :
  n = 10 :=
sorry

end square_number_n_value_l179_179587


namespace george_slices_l179_179247

def num_small_pizzas := 3
def num_large_pizzas := 2
def slices_per_small_pizza := 4
def slices_per_large_pizza := 8
def slices_leftover := 10
def slices_per_person := 3
def total_pizza_slices := (num_small_pizzas * slices_per_small_pizza) + (num_large_pizzas * slices_per_large_pizza)
def slices_eaten := total_pizza_slices - slices_leftover
def G := 6 -- Slices George would like to eat

theorem george_slices :
  G + (G + 1) + ((G + 1) / 2) + (3 * slices_per_person) = slices_eaten :=
by
  sorry

end george_slices_l179_179247


namespace complement_of_B_in_A_l179_179437

def complement (A B : Set Int) := { x ∈ A | x ∉ B }

theorem complement_of_B_in_A (A B : Set Int) (a : Int) (h1 : A = {2, 3, 4}) (h2 : B = {a + 2, a}) (h3 : A ∩ B = B)
: complement A B = {3} :=
  sorry

end complement_of_B_in_A_l179_179437


namespace tom_bought_new_books_l179_179046

def original_books : ℕ := 5
def sold_books : ℕ := 4
def current_books : ℕ := 39

def new_books (original_books sold_books current_books : ℕ) : ℕ :=
  current_books - (original_books - sold_books)

theorem tom_bought_new_books :
  new_books original_books sold_books current_books = 38 :=
by
  sorry

end tom_bought_new_books_l179_179046


namespace pablo_days_to_complete_all_puzzles_l179_179014

def average_pieces_per_hour : ℕ := 100
def puzzles_300_pieces : ℕ := 8
def puzzles_500_pieces : ℕ := 5
def pieces_per_300_puzzle : ℕ := 300
def pieces_per_500_puzzle : ℕ := 500
def max_hours_per_day : ℕ := 7

theorem pablo_days_to_complete_all_puzzles :
  let total_pieces := (puzzles_300_pieces * pieces_per_300_puzzle) + (puzzles_500_pieces * pieces_per_500_puzzle)
  let pieces_per_day := max_hours_per_day * average_pieces_per_hour
  let days_to_complete := total_pieces / pieces_per_day
  days_to_complete = 7 :=
by
  sorry

end pablo_days_to_complete_all_puzzles_l179_179014


namespace solve_for_ab_l179_179895

theorem solve_for_ab (a b : ℤ) 
  (h1 : a + 3 * b = 27) 
  (h2 : 5 * a + 4 * b = 47) : 
  a + b = 11 :=
sorry

end solve_for_ab_l179_179895


namespace number_of_pizzas_ordered_l179_179195

-- Definitions from conditions
def slices_per_pizza : Nat := 2
def total_slices : Nat := 28

-- Proof that the number of pizzas ordered is 14
theorem number_of_pizzas_ordered : total_slices / slices_per_pizza = 14 := by
  sorry

end number_of_pizzas_ordered_l179_179195


namespace largest_base5_three_digit_to_base10_l179_179197

theorem largest_base5_three_digit_to_base10 : 
  let base5_number := 4 * 5^2 + 4 * 5^1 + 4 * 5^0 in
  base5_number = 124 :=
by
let base5_number := 4 * 5^2 + 4 * 5^1 + 4 * 5^0
have h1 : 4 * 5^2 = 100 := by norm_num
have h2 : 4 * 5^1 = 20 := by norm_num
have h3 : 4 * 5^0 = 4 := by norm_num
have h4 : base5_number = 100 + 20 + 4 := by simp [base5_number, h1, h2, h3]
show base5_number = 124, by simp [h4]
sorry

end largest_base5_three_digit_to_base10_l179_179197


namespace cargo_to_cruise_ratio_l179_179842

theorem cargo_to_cruise_ratio
  (C S F : ℕ)
  (hS1 : S = C + 6)
  (hS2 : S = 7 * F)
  (hCruise : 4 = 4)
  (hTotal : 4 + C + S + F = 28) :
  C / 4 = 2 :=
by
  sorry

end cargo_to_cruise_ratio_l179_179842


namespace shaded_percentage_l179_179675

noncomputable def percent_shaded (side_len : ℕ) : ℝ :=
  let total_area := (side_len : ℝ) * side_len
  let shaded_area := (2 * 2) + (2 * 5) + (1 * 7)
  100 * (shaded_area / total_area)

theorem shaded_percentage (PQRS_side : ℕ) (hPQRS : PQRS_side = 7) :
  percent_shaded PQRS_side = 42.857 :=
  by
  rw [hPQRS]
  sorry

end shaded_percentage_l179_179675


namespace counter_example_not_power_of_4_for_25_l179_179738

theorem counter_example_not_power_of_4_for_25 : ∃ n ≥ 2, n = 25 ∧ ¬ ∃ k : ℕ, 2 ^ (2 ^ n) % (2 ^ n - 1) = 4 ^ k :=
by {
  sorry
}

end counter_example_not_power_of_4_for_25_l179_179738


namespace greg_ate_4_halves_l179_179439

def greg_ate_halves (total_cookies : ℕ) (brad_halves : ℕ) (left_halves : ℕ) : ℕ :=
  2 * total_cookies - (brad_halves + left_halves)

theorem greg_ate_4_halves : greg_ate_halves 14 6 18 = 4 := by
  sorry

end greg_ate_4_halves_l179_179439


namespace largest_base5_three_digit_in_base10_l179_179219

theorem largest_base5_three_digit_in_base10 :
  let a := 4
      b := 4
      c := 4
      largest_base5 := 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  in largest_base5 = 124 :=
by
  let a := 4
  let b := 4
  let c := 4
  let largest_base5 := 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  show largest_base5 = 124
  sorry

end largest_base5_three_digit_in_base10_l179_179219


namespace no_feasible_distribution_l179_179311

-- Define the initial conditions
def initial_runs_player_A : ℕ := 320
def initial_runs_player_B : ℕ := 450
def initial_runs_player_C : ℕ := 550

def initial_innings : ℕ := 10

def required_increase_A : ℕ := 4
def required_increase_B : ℕ := 5
def required_increase_C : ℕ := 6

def total_run_limit : ℕ := 250

-- Define the total runs required after 11 innings
def total_required_runs_after_11_innings (initial_runs avg_increase : ℕ) : ℕ :=
  (initial_runs / initial_innings + avg_increase) * 11

-- Calculate the additional runs needed in the next innings
def additional_runs_needed (initial_runs avg_increase : ℕ) : ℕ :=
  total_required_runs_after_11_innings initial_runs avg_increase - initial_runs

-- Calculate the total additional runs needed for all players
def total_additional_runs_needed : ℕ :=
  additional_runs_needed initial_runs_player_A required_increase_A +
  additional_runs_needed initial_runs_player_B required_increase_B +
  additional_runs_needed initial_runs_player_C required_increase_C

-- The statement to verify if the total additional required runs exceed the limit
theorem no_feasible_distribution :
  total_additional_runs_needed > total_run_limit :=
by 
  -- Skipping proofs and just stating the condition is what we aim to show.
  sorry

end no_feasible_distribution_l179_179311


namespace derivative_f_l179_179350

noncomputable def f (x : ℝ) : ℝ := 1 + Real.cos x

theorem derivative_f (x : ℝ) : deriv f x = -Real.sin x := 
by 
  sorry

end derivative_f_l179_179350


namespace dot_product_ABC_l179_179119

-- Defining vectors as pairs of real numbers
def vector := (ℝ × ℝ)

-- Defining the vectors AB and AC
def AB : vector := (1, 0)
def AC : vector := (-2, 3)

-- Definition of vector subtraction
def vector_sub (v1 v2 : vector) : vector := (v1.1 - v2.1, v1.2 - v2.2)

-- Definition of dot product
def dot_product (v1 v2 : vector) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Define vector BC using the given vectors AB and AC
def BC : vector := vector_sub AC AB

-- The theorem stating the desired dot product result
theorem dot_product_ABC : dot_product AB BC = -3 := by
  sorry

end dot_product_ABC_l179_179119


namespace compute_cd_l179_179800

variable (c d : ℝ)

theorem compute_cd (h1 : c + d = 10) (h2 : c^3 + d^3 = 370) : c * d = 21 := by
  -- Proof would go here
  sorry

end compute_cd_l179_179800


namespace number_less_than_one_is_correct_l179_179836

theorem number_less_than_one_is_correct : (1 - 5 = -4) :=
by
  sorry

end number_less_than_one_is_correct_l179_179836


namespace find_triples_prime_l179_179564

theorem find_triples_prime (
  a b c : ℕ
) (ha : Nat.prime (a^2 + 1))
  (hb : Nat.prime (b^2 + 1))
  (hc : (a^2 + 1) * (b^2 + 1) = c^2 + 1) :
  (a = 1 ∧ b = 2 ∧ c = 3) ∨ 
  (a = 2 ∧ b = 1 ∧ c = 3) :=
sorry

end find_triples_prime_l179_179564


namespace root_equivalence_l179_179656

theorem root_equivalence (a_1 a_2 a_3 b : ℝ) :
  (∃ c_1 c_2 c_3 : ℝ, c_1 ≠ c_2 ∧ c_2 ≠ c_3 ∧ c_1 ≠ c_3 ∧
    (∀ x : ℝ, (x - a_1) * (x - a_2) * (x - a_3) = b ↔ (x = c_1 ∨ x = c_2 ∨ x = c_3))) →
  (∀ x : ℝ, (x + c_1) * (x + c_2) * (x + c_3) = b ↔ (x = -a_1 ∨ x = -a_2 ∨ x = -a_3)) :=
by 
  sorry

end root_equivalence_l179_179656


namespace largest_base5_eq_124_l179_179215

-- Define largest base-5 number with three digits
def largest_base5_three_digits : ℕ := 4 * 5^2 + 4 * 5^1 + 4 * 5^0

theorem largest_base5_eq_124 : largest_base5_three_digits = 124 :=
by
  -- Proof steps go here
  sorry

end largest_base5_eq_124_l179_179215


namespace f_periodic_4_l179_179903

noncomputable def f : ℝ → ℝ := sorry -- f is some function ℝ → ℝ

theorem f_periodic_4 (h : ∀ x, f x = -f (x + 2)) : f 100 = f 4 := 
by
  sorry

end f_periodic_4_l179_179903


namespace mark_garden_total_flowers_l179_179339

theorem mark_garden_total_flowers :
  let yellow := 10
  let purple := yellow + (80 / 100) * yellow
  let total_yellow_purple := yellow + purple
  let green := (25 / 100) * total_yellow_purple
  total_yellow_purple + green = 35 :=
by
  let yellow := 10
  let purple := yellow + (80 / 100) * yellow
  let total_yellow_purple := yellow + purple
  let green := (25 / 100) * total_yellow_purple
  simp [yellow, purple, total_yellow_purple, green]
  sorry

end mark_garden_total_flowers_l179_179339


namespace not_prime_4k4_plus_1_not_prime_k4_plus_4_l179_179073

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem not_prime_4k4_plus_1 (k : ℕ) (hk : k > 0) : ¬ is_prime (4 * k^4 + 1) :=
by sorry

theorem not_prime_k4_plus_4 (k : ℕ) (hk : k > 0) : ¬ is_prime (k^4 + 4) :=
by sorry

end not_prime_4k4_plus_1_not_prime_k4_plus_4_l179_179073


namespace possible_slopes_of_line_intersecting_ellipse_l179_179533

theorem possible_slopes_of_line_intersecting_ellipse :
  ∃ m : ℝ, 
    (∀ (x y : ℝ), (y = m * x + 3) → (4 * x^2 + 25 * y^2 = 100)) →
    (m ∈ Set.Iio (-Real.sqrt (16 / 405)) ∪ Set.Ici (Real.sqrt (16 / 405))) :=
sorry

end possible_slopes_of_line_intersecting_ellipse_l179_179533


namespace chess_tournament_l179_179696

theorem chess_tournament (n : ℕ) (white_rook black_elephant : ℕ) (total_games : ℕ) :
  white_rook = 15 → black_elephant = 20 → total_games = white_rook * black_elephant → 
  (n ≥ 280 → (∃ p, p ∈ {1..white_rook} ∧ (n < 300))) :=
by
  intros hwr hbe htg hn
  sorry

end chess_tournament_l179_179696


namespace area_of_base_of_cone_l179_179888

theorem area_of_base_of_cone (semicircle_area : ℝ) (h1 : semicircle_area = 2 * Real.pi) : 
  ∃ (base_area : ℝ), base_area = Real.pi :=
by
  sorry

end area_of_base_of_cone_l179_179888


namespace gum_left_after_sharing_l179_179857

-- Define the initial state of Adrianna's gum and the changes to it
def initial_gum : Nat := 10
def additional_gum : Nat := 3
def given_out_gum : Nat := 11

-- Define the final state of Adrianna's gum
def final_gum : Nat := initial_gum + additional_gum - given_out_gum

-- Prove that Adrianna ends up with 2 pieces of gum under the given conditions
theorem gum_left_after_sharing :
  final_gum = 2 :=
by 
  -- Since this is just the statement and not the proof, we end with sorry.
  sorry

end gum_left_after_sharing_l179_179857


namespace simplify_and_rationalize_correct_l179_179026

noncomputable def simplify_and_rationalize : ℚ :=
  1 / (2 + 1 / (Real.sqrt 5 + 2))

theorem simplify_and_rationalize_correct : simplify_and_rationalize = (Real.sqrt 5) / 5 := by
  sorry

end simplify_and_rationalize_correct_l179_179026


namespace simplify_expression_l179_179946

theorem simplify_expression (x : ℝ) : 5 * x + 6 - x + 12 = 4 * x + 18 :=
by sorry

end simplify_expression_l179_179946


namespace assignment_plans_l179_179188

theorem assignment_plans (students locations : ℕ) (library science_museum nursing_home : ℕ) 
  (students_eq : students = 5) (locations_eq : locations = 3) 
  (lib_gt0 : library > 0) (sci_gt0 : science_museum > 0) (nur_gt0 : nursing_home > 0) 
  (lib_science_nursing : library + science_museum + nursing_home = students) : 
  ∃ (assignments : ℕ), assignments = 150 :=
by
  sorry

end assignment_plans_l179_179188


namespace sampling_interval_is_100_l179_179878

-- Define the total number of numbers (N), the number of samples to be taken (k), and the condition for systematic sampling.
def N : ℕ := 2005
def k : ℕ := 20

-- Define the concept of systematic sampling interval
def sampling_interval (N k : ℕ) : ℕ := N / k

-- The proof that the sampling interval is 100 when 2005 numbers are sampled as per the systematic sampling method.
theorem sampling_interval_is_100 (N k : ℕ) 
  (hN : N = 2005) 
  (hk : k = 20) 
  (h1 : N % k ≠ 0) : 
  sampling_interval (N - (N % k)) k = 100 :=
by
  -- Initialization
  sorry

end sampling_interval_is_100_l179_179878


namespace find_triples_l179_179563

noncomputable def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem find_triples (a b c : ℕ) :
  is_prime (a^2 + 1) ∧
  is_prime (b^2 + 1) ∧
  (a^2 + 1) * (b^2 + 1) = c^2 + 1 →
  (a = 1 ∧ b = 2 ∧ c = 3) ∨ (a = 2 ∧ b = 1 ∧ c = 3) :=
by
  sorry

end find_triples_l179_179563


namespace find_m_symmetry_l179_179284

theorem find_m_symmetry (A B : ℝ × ℝ) (m : ℝ)
  (hA : A = (-3, m)) (hB : B = (3, 4)) (hy : A.2 = B.2) : m = 4 :=
sorry

end find_m_symmetry_l179_179284


namespace unequal_numbers_l179_179292

theorem unequal_numbers {k : ℚ} (h : 3 * (1 : ℚ) + 7 * (1 : ℚ) + 2 * k = 0) (d : (7^2 : ℚ) - 4 * 3 * 2 * k = 0) : 
    (3 : ℚ) ≠ (7 : ℚ) ∧ (3 : ℚ) ≠ k ∧ (7 : ℚ) ≠ k :=
by
  -- adding sorry for skipping proof
  sorry

end unequal_numbers_l179_179292


namespace eventually_periodic_sequence_l179_179321

theorem eventually_periodic_sequence
  (a : ℕ → ℕ) (h_pos : ∀ n, 0 < a n)
  (h_div : ∀ n m, (a (n + 2 * m)) ∣ (a n + a (n + m))) :
  ∃ N d, 0 < N ∧ 0 < d ∧ ∀ n, N < n → a n = a (n + d) :=
by
  sorry

end eventually_periodic_sequence_l179_179321


namespace shopkeeper_percentage_profit_l179_179227

variable {x : ℝ} -- cost price per kg of apples

theorem shopkeeper_percentage_profit 
  (total_weight : ℝ)
  (first_half_sold_at : ℝ)
  (second_half_sold_at : ℝ)
  (first_half_profit : ℝ)
  (second_half_profit : ℝ)
  (total_cost_price : ℝ)
  (total_selling_price : ℝ)
  (total_profit : ℝ)
  (percentage_profit : ℝ) :
  total_weight = 100 →
  first_half_sold_at = 0.5 * total_weight →
  second_half_sold_at = 0.5 * total_weight →
  first_half_profit = 25 →
  second_half_profit = 30 →
  total_cost_price = x * total_weight →
  total_selling_price = (first_half_sold_at * (1 + first_half_profit / 100) * x) + (second_half_sold_at * (1 + second_half_profit / 100) * x) →
  total_profit = total_selling_price - total_cost_price →
  percentage_profit = (total_profit / total_cost_price) * 100 →
  percentage_profit = 27.5 := by
  sorry

end shopkeeper_percentage_profit_l179_179227


namespace domain_sqrt_l179_179643

noncomputable def domain_of_function := {x : ℝ | x ≥ 0 ∧ x - 1 ≥ 0}

theorem domain_sqrt : domain_of_function = {x : ℝ | 1 ≤ x} := by {
  sorry
}

end domain_sqrt_l179_179643


namespace negation_of_zero_product_l179_179942

theorem negation_of_zero_product (x y : ℝ) : (xy ≠ 0) → (x ≠ 0) ∧ (y ≠ 0) :=
sorry

end negation_of_zero_product_l179_179942


namespace probability_even_sum_is_correct_l179_179993

noncomputable def probability_even_sum : ℚ :=
  let p_even_first := (2 : ℚ) / 5
  let p_odd_first := (3 : ℚ) / 5
  let p_even_second := (1 : ℚ) / 4
  let p_odd_second := (3 : ℚ) / 4

  let p_both_even := p_even_first * p_even_second
  let p_both_odd := p_odd_first * p_odd_second

  p_both_even + p_both_odd

theorem probability_even_sum_is_correct : probability_even_sum = 11 / 20 := by
  sorry

end probability_even_sum_is_correct_l179_179993


namespace sum_difference_arithmetic_sequences_l179_179548

open Nat

def arithmetic_sequence_sum (a d n : Nat) : Nat :=
  n * (2 * a + (n - 1) * d) / 2

theorem sum_difference_arithmetic_sequences :
  arithmetic_sequence_sum 2101 1 123 - arithmetic_sequence_sum 401 1 123 = 209100 := by
  sorry

end sum_difference_arithmetic_sequences_l179_179548


namespace evaluate_f_at_minus_2_l179_179772

def f (x : ℚ) : ℚ := (2 * x - 3) / (4 * x + 5)

theorem evaluate_f_at_minus_2 : f (-2) = 7 / 3 := by
  -- Proof is omitted
  sorry

end evaluate_f_at_minus_2_l179_179772


namespace total_deviation_correct_average_weight_per_bag_correct_l179_179983

-- Define standard weight per bag
def standard_weight : ℕ := 150

-- Define the deviations for each bag
def deviations : List ℤ := [-6, -3, -1, 7, 3, 4, -3, -2, -2, 1]

-- Define the number of bags
def num_bags : ℕ := 10

-- Define a function to calculate the total deviation
def total_deviation (devs : List ℤ) : ℤ := devs.sum

-- Define the total weight deviation
def total_weight_deviation := total_deviation deviations

-- Calculate the total weight
def total_weight := (standard_weight * num_bags : ℤ) + total_weight_deviation

-- Calculate the average weight per bag
def average_weight_per_bag := total_weight / (num_bags : ℤ)

-- Verify the total deviation
theorem total_deviation_correct : total_deviation deviations = -2 :=
by
  -- skipping the proof
  sorry

-- Verify the average weight per bag
theorem average_weight_per_bag_correct : average_weight_per_bag = 149.8 :=
by
  -- skipping the proof
  sorry

end total_deviation_correct_average_weight_per_bag_correct_l179_179983


namespace find_a_l179_179426

theorem find_a (x y z a : ℝ) (h1 : 2 * x^2 + 3 * y^2 + 6 * z^2 = a) (h2 : a > 0) (h3 : ∀ x y z : ℝ, 2 * x^2 + 3 * y^2 + 6 * z^2 = a → (x + y + z) ≤ 1) :
  a = 1 := 
sorry

end find_a_l179_179426


namespace cuboid_third_edge_length_l179_179351

theorem cuboid_third_edge_length
  (l w : ℝ)
  (A : ℝ)
  (h : ℝ)
  (hl : l = 4)
  (hw : w = 5)
  (hA : A = 148)
  (surface_area_formula : A = 2 * (l * w + l * h + w * h)) :
  h = 6 :=
by
  sorry

end cuboid_third_edge_length_l179_179351


namespace pascal_triangle_row_20_element_5_l179_179051

theorem pascal_triangle_row_20_element_5 : binomial 20 4 = 4845 := 
by sorry

end pascal_triangle_row_20_element_5_l179_179051


namespace households_subscribing_B_and_C_l179_179076

/-- Each household subscribes to 2 different newspapers.
Residents only subscribe to newspapers A, B, and C.
There are 30 subscriptions for newspaper A.
There are 34 subscriptions for newspaper B.
There are 40 subscriptions for newspaper C.
Thus, the number of households that subscribe to both
newspaper B and newspaper C is 22. -/
theorem households_subscribing_B_and_C (subs_A subs_B subs_C households : ℕ) 
    (hA : subs_A = 30) (hB : subs_B = 34) (hC : subs_C = 40) (h_total : households = (subs_A + subs_B + subs_C) / 2) :
  (households - subs_A) = 22 :=
by
  -- Substitute the values to demonstrate equality based on the given conditions.
  sorry

end households_subscribing_B_and_C_l179_179076


namespace geometric_seq_problem_l179_179428

variable {a : ℕ → ℝ}

def geometric_sequence (a : ℕ → ℝ) :=
  ∃ r : ℝ, ∀ n : ℕ, a (n+1) = a n * r

theorem geometric_seq_problem (h_geom : geometric_sequence a) 
  (h_cond : a 8 * a 9 * a 10 = -a 13 ^ 2 ∧ -a 13 ^ 2 = -1000) :
  a 10 * a 12 = 100 * Real.sqrt 10 :=
by
  sorry

end geometric_seq_problem_l179_179428


namespace range_of_a_l179_179905

theorem range_of_a (a : ℝ) : (∃ x : ℝ, |x - 3| + |x - 4| < a) → a > 1 :=
sorry

end range_of_a_l179_179905


namespace least_k_divisible_by_2160_l179_179302

theorem least_k_divisible_by_2160 (k : ℤ) : k^3 ∣ 2160 → k ≥ 60 := by
  sorry

end least_k_divisible_by_2160_l179_179302


namespace find_8b_l179_179298

variable (a b : ℚ)

theorem find_8b (h1 : 4 * a + 3 * b = 5) (h2 : a = b - 3) : 8 * b = 136 / 7 := by
  sorry

end find_8b_l179_179298


namespace sum_of_roots_l179_179327

theorem sum_of_roots {x1 x2 x3 k m : ℝ} (h1 : x1 ≠ x2) (h2 : x2 ≠ x3) (h3 : x1 ≠ x3)
  (h4 : 2 * x1^3 - k * x1 = m) (h5 : 2 * x2^3 - k * x2 = m) (h6 : 2 * x3^3 - k * x3 = m) :
  x1 + x2 + x3 = 0 :=
sorry

end sum_of_roots_l179_179327


namespace solve_base7_addition_problem_l179_179562

noncomputable def base7_addition_problem : Prop :=
  ∃ (X Y: ℕ), 
    (5 * 7^2 + X * 7 + Y) + (3 * 7^1 + 2) = 6 * 7^2 + 2 * 7 + X ∧
    X + Y = 10 

theorem solve_base7_addition_problem : base7_addition_problem :=
by sorry

end solve_base7_addition_problem_l179_179562


namespace second_closest_location_l179_179035
-- Import all necessary modules from the math library

-- Define the given distances (conditions)
def distance_library : ℝ := 1.912 * 1000  -- distance in meters
def distance_park : ℝ := 876              -- distance in meters
def distance_clothing_store : ℝ := 1.054 * 1000  -- distance in meters

-- State the proof problem
theorem second_closest_location :
  (distance_library = 1912) →
  (distance_park = 876) →
  (distance_clothing_store = 1054) →
  (distance_clothing_store = 1054) :=
by
  intros h1 h2 h3
  -- sorry to skip the proof
  sorry

end second_closest_location_l179_179035


namespace counter_example_exists_l179_179741

theorem counter_example_exists : 
  ∃ n : ℕ, n ≥ 2 ∧ ¬(∃ k : ℕ, (2 ^ 2 ^ n) % (2 ^ n - 1) = 4 ^ k) :=
  sorry

end counter_example_exists_l179_179741


namespace cubic_roots_expression_l179_179926

theorem cubic_roots_expression (p q r : ℝ) 
  (h1 : p + q + r = 4) 
  (h2 : pq + pr + qr = 6) 
  (h3 : pqr = 3) : 
  p / (qr + 2) + q / (pr + 2) + r / (pq + 2) = 4 / 5 := 
by 
  sorry

end cubic_roots_expression_l179_179926


namespace jerome_classmates_count_l179_179787

theorem jerome_classmates_count (C F : ℕ) (h1 : F = C / 2) (h2 : 33 = C + F + 3) : C = 20 :=
by
  sorry

end jerome_classmates_count_l179_179787


namespace minimum_games_l179_179689

theorem minimum_games (n : ℕ) : 
  (∃ (w b : ℕ) (W B : ℕ → Prop),
    (∀ i, i < 15 → W i) ∧
    (∀ j, j < 20 → B j) ∧
    w = 15 ∧
    b = 20 ∧
    (∀ i j, W i → B j → ∃ g, g < 300 ∧ g = i * b + j * w) ∧
    (∃ g, g ≤ 300 ∧ (n > 280 → ∃ i, W i ∧ i = g div b))): n = 280 :=
sorry

end minimum_games_l179_179689


namespace max_volume_cube_max_volume_parallelepiped_l179_179458

variables {a b c : ℝ}

-- Problem (a): Cube with the maximum volume entirely contained in the tetrahedron
theorem max_volume_cube (h : a > 0 ∧ b > 0 ∧ c > 0) :
  ∃ s : ℝ, s = (a * b * c) / (a * b + b * c + a * c) := sorry

-- Problem (b): Rectangular parallelepiped with the maximum volume entirely contained in the tetrahedron
theorem max_volume_parallelepiped (h : a > 0 ∧ b > 0 ∧ c > 0) :
  ∃ (x y z : ℝ),
  (x = a / 3 ∧ y = b / 3 ∧ z = c / 3) ∧
  (x * y * z = (a * b * c) / 27) := sorry

end max_volume_cube_max_volume_parallelepiped_l179_179458


namespace bicyclist_speed_remainder_l179_179011

noncomputable def speed_of_bicyclist (total_distance first_distance remaining_distance time_for_first_distance total_time : ℝ) : ℝ :=
  remaining_distance / (total_time - time_for_first_distance)

theorem bicyclist_speed_remainder 
  (total_distance : ℝ)
  (first_distance : ℝ)
  (remaining_distance : ℝ)
  (first_speed : ℝ)
  (average_speed : ℝ)
  (correct_speed : ℝ) :
  total_distance = 250 → 
  first_distance = 100 →
  remaining_distance = total_distance - first_distance →
  first_speed = 20 →
  average_speed = 16.67 →
  correct_speed = 15 →
  speed_of_bicyclist total_distance first_distance remaining_distance (first_distance / first_speed) (total_distance / average_speed) = correct_speed :=
by
  sorry

end bicyclist_speed_remainder_l179_179011


namespace approximation_hundred_thousandth_place_l179_179728

theorem approximation_hundred_thousandth_place (n : ℕ) (h : n = 537400000) : 
  ∃ p : ℕ, p = 100000 := 
sorry

end approximation_hundred_thousandth_place_l179_179728


namespace mathematician_daily_questions_l179_179244

/-- Given 518 questions for the first project and 476 for the second project,
if all questions are to be completed in 7 days, prove that the number
of questions completed each day is 142. -/
theorem mathematician_daily_questions (q1 q2 days questions_per_day : ℕ) 
  (h1 : q1 = 518) (h2 : q2 = 476) (h3 : days = 7) 
  (h4 : q1 + q2 = 994) (h5 : questions_per_day = 994 / 7) :
  questions_per_day = 142 :=
sorry

end mathematician_daily_questions_l179_179244


namespace choose_three_consecutive_circles_l179_179237

theorem choose_three_consecutive_circles (n : ℕ) (hn : n = 33) : 
  ∃ (ways : ℕ), ways = 57 :=
by
  sorry

end choose_three_consecutive_circles_l179_179237


namespace percentage_of_carnations_is_44_percent_l179_179387

noncomputable def total_flowers : ℕ := sorry
def pink_percentage : ℚ := 2 / 5
def red_percentage : ℚ := 2 / 5
def yellow_percentage : ℚ := 1 / 5
def pink_roses_fraction : ℚ := 2 / 5
def red_carnations_fraction : ℚ := 1 / 2

theorem percentage_of_carnations_is_44_percent
  (F : ℕ)
  (h_pink : pink_percentage * F = 2 / 5 * F)
  (h_red : red_percentage * F = 2 / 5 * F)
  (h_yellow : yellow_percentage * F = 1 / 5 * F)
  (h_pink_roses : pink_roses_fraction * (pink_percentage * F) = 2 / 25 * F)
  (h_red_carnations : red_carnations_fraction * (red_percentage * F) = 1 / 5 * F) :
  ((6 / 25 * F + 5 / 25 * F) / F) * 100 = 44 := sorry

end percentage_of_carnations_is_44_percent_l179_179387


namespace gum_left_after_sharing_l179_179858

-- Define the initial state of Adrianna's gum and the changes to it
def initial_gum : Nat := 10
def additional_gum : Nat := 3
def given_out_gum : Nat := 11

-- Define the final state of Adrianna's gum
def final_gum : Nat := initial_gum + additional_gum - given_out_gum

-- Prove that Adrianna ends up with 2 pieces of gum under the given conditions
theorem gum_left_after_sharing :
  final_gum = 2 :=
by 
  -- Since this is just the statement and not the proof, we end with sorry.
  sorry

end gum_left_after_sharing_l179_179858


namespace point_on_circle_l179_179602

noncomputable def x_value_on_circle : ℝ :=
  let a := (-3 : ℝ)
  let b := (21 : ℝ)
  let Cx := (a + b) / 2
  let Cy := 0
  let radius := (b - a) / 2
  let y := 12
  Cx

theorem point_on_circle (x y : ℝ) (a b : ℝ) (ha : a = -3) (hb : b = 21) (hy : y = 12) :
  let Cx := (a + b) / 2
  let Cy := 0
  let radius := (b - a) / 2
  (x - Cx) ^ 2 + y ^ 2 = radius ^ 2 → x = x_value_on_circle :=
by
  intros
  sorry

end point_on_circle_l179_179602


namespace work_efficiency_ratio_l179_179851
noncomputable section

variable (A_eff B_eff : ℚ)

-- Conditions
def efficient_together (A_eff B_eff : ℚ) : Prop := A_eff + B_eff = 1 / 12
def efficient_alone (A_eff : ℚ) : Prop := A_eff = 1 / 16

-- Theorem to prove
theorem work_efficiency_ratio (A_eff B_eff : ℚ) (h1 : efficient_together A_eff B_eff) (h2 : efficient_alone A_eff) : A_eff / B_eff = 3 := by
  sorry

end work_efficiency_ratio_l179_179851


namespace no_integers_a_b_existence_no_positive_integers_a_b_c_existence_l179_179980

-- Part (a)
theorem no_integers_a_b_existence (a b : ℤ) :
  ¬(a^2 - 3 * (b^2) = 8) :=
sorry

-- Part (b)
theorem no_positive_integers_a_b_c_existence (a b c : ℕ) (ha: a > 0) (hb: b > 0) (hc: c > 0 ) :
  ¬(a^2 + b^2 = 3 * (c^2)) :=
sorry

end no_integers_a_b_existence_no_positive_integers_a_b_c_existence_l179_179980


namespace largest_n_for_divisibility_l179_179669

theorem largest_n_for_divisibility : 
  ∃ n : ℕ, (n + 12 ∣ n^3 + 150) ∧ (∀ m : ℕ, (m + 12 ∣ m^3 + 150) → m ≤ 246) :=
sorry

end largest_n_for_divisibility_l179_179669


namespace calculate_expression_l179_179398

theorem calculate_expression : 15 * 35 + 45 * 15 = 1200 :=
by {
  -- hint to the Lean prover to consider associative property
  sorry
}

end calculate_expression_l179_179398


namespace probability_at_least_25_points_is_half_l179_179777

noncomputable def probability_at_least_25_points : ℚ :=
  let red_points := 10
  let black_points := 5
  let total_draws := 3
  let points_needed := 25
  let draw_red_prob := 1 / 2
  let draw_black_prob := 1 / 2
  let scenarios := [(3, red_points), (2, red_points), (1, black_points)]
  let probability (s : ℕ × ℕ) :=
    match s with
    | (r, p) => if r * p >= points_needed then (draw_red_prob ^ r) * (draw_black_prob ^ (total_draws - r)) else 0
  ∑' s in scenarios, probability s

theorem probability_at_least_25_points_is_half :
  probability_at_least_25_points = 1 / 2 :=
sorry

end probability_at_least_25_points_is_half_l179_179777


namespace sum_of_coefficients_l179_179753

def polynomial := 3 * (λ x : ℝ, x^8 - x^5 + 2 * x^3 - 6) - 5 * (λ x : ℝ, x^4 + 3 * x^2) + 2 * (λ x : ℝ, x^6 - 5)

theorem sum_of_coefficients : polynomial 1 = -40 := 
by
  sorry

end sum_of_coefficients_l179_179753


namespace vector_parallel_eq_l179_179331

theorem vector_parallel_eq (k : ℝ) (a b : ℝ × ℝ) 
  (h_a : a = (k, 2)) (h_b : b = (1, 1)) (h_parallel : (∃ c : ℝ, a = (c * 1, c * 1))) : k = 2 := by
  sorry

end vector_parallel_eq_l179_179331


namespace sergio_has_6_more_correct_answers_l179_179633

-- Define conditions
def total_questions : ℕ := 50
def incorrect_answers_sylvia : ℕ := total_questions / 5
def incorrect_answers_sergio : ℕ := 4

-- Calculate correct answers
def correct_answers_sylvia : ℕ := total_questions - incorrect_answers_sylvia
def correct_answers_sergio : ℕ := total_questions - incorrect_answers_sergio

-- The proof problem
theorem sergio_has_6_more_correct_answers :
  correct_answers_sergio - correct_answers_sylvia = 6 :=
by
  sorry

end sergio_has_6_more_correct_answers_l179_179633


namespace perpendicular_lines_l179_179067

theorem perpendicular_lines (m : ℝ) : 
  (m = -2 → (2-m) * (-(m+3)/(2-m)) + m * (m-3) / (-(m+3)) = 0) → 
  (m = -2 ∨ m = 1) := 
sorry

end perpendicular_lines_l179_179067


namespace distance_A_B_l179_179017

noncomputable def distance_between_points (v_A v_B : ℝ) (t : ℝ) : ℝ := 5 * (6 * t / (2 / 3 * t))

theorem distance_A_B
  (v_A v_B : ℝ)
  (t : ℝ)
  (h1 : v_A = 1.2 * v_B)
  (h2 : ∃ distance_broken, distance_broken = 5)
  (h3 : ∃ delay, delay = (1 / 6) * 6 * t)
  (h4 : ∃ v_B_new, v_B_new = 1.6 * v_B)
  (h5 : distance_between_points v_A v_B t = 45) :
  distance_between_points v_A v_B t = 45 :=
sorry

end distance_A_B_l179_179017


namespace quadrilateral_area_l179_179290

theorem quadrilateral_area 
  (p : ℝ) (hp : p > 0)
  (P : ℝ × ℝ) (hP : P = (1, 1 / 4))
  (focus : ℝ × ℝ) (hfocus : focus = (0, 1))
  (directrix : ℝ → Prop) (hdirectrix : ∀ y, directrix y ↔ y = 1)
  (F : ℝ × ℝ) (hF : F = (0, 1))
  (M : ℝ × ℝ) (hM : M = (0, 1))
  (Q : ℝ × ℝ) 
  (PQ : ℝ)
  (area : ℝ) 
  (harea : area = 13 / 8) :
  ∃ (PQMF : ℝ), PQMF = 13 / 8 :=
sorry

end quadrilateral_area_l179_179290


namespace sale_on_day_five_l179_179530

def sale1 : ℕ := 435
def sale2 : ℕ := 927
def sale3 : ℕ := 855
def sale6 : ℕ := 741
def average_sale : ℕ := 625
def total_days : ℕ := 5

theorem sale_on_day_five : 
  average_sale * total_days - (sale1 + sale2 + sale3 + sale6) = 167 :=
by
  sorry

end sale_on_day_five_l179_179530


namespace smallest_naive_number_max_naive_number_divisible_by_ten_l179_179418

/-- Definition of naive number -/
def is_naive_number (M : ℕ) : Prop :=
  let a := M / 1000
  let b := (M % 1000) / 100
  let c := (M % 100) / 10
  let d := M % 10
  (a = d + 6) ∧ (b = c + 2)

/-- Definition of P function -/
def P (M : ℕ) : ℤ :=
  let a := M / 1000
  let b := (M % 1000) / 100
  let c := (M % 100) / 10
  let d := M % 10
  3 * (a + b) + c + d

/-- Definition of Q function -/
def Q (M : ℕ) : ℤ :=
  let a := M / 1000
  a - 5

namespace NaiveNumber

/-- Smallest naive number is 6200 -/
theorem smallest_naive_number : ∃ M : ℕ, is_naive_number M ∧ M = 6200 :=
  sorry

/-- Maximum naive number such that P(M)/Q(M) is divisible by 10 is 9313 -/
theorem max_naive_number_divisible_by_ten : ∃ M : ℕ, is_naive_number M ∧ (P(M) / Q(M)) % 10 = 0 ∧ M = 9313 :=
  sorry

end NaiveNumber

end smallest_naive_number_max_naive_number_divisible_by_ten_l179_179418


namespace lulu_final_cash_l179_179936

-- Definitions of the problem conditions
def initial_amount : ℕ := 65
def spent_on_ice_cream : ℕ := 5
def spent_on_tshirt (remaining : ℕ) : ℕ := remaining / 2
def deposit_in_bank (remaining : ℕ) : ℕ := remaining / 5

-- The proof problem statement
theorem lulu_final_cash :
  ∃ final_cash : ℕ,
    final_cash = initial_amount - spent_on_ice_cream - spent_on_tshirt (initial_amount - spent_on_ice_cream) - 
                      deposit_in_bank (spent_on_tshirt (initial_amount - spent_on_ice_cream)) ∧
    final_cash = 24 :=
by {
  sorry
}

end lulu_final_cash_l179_179936


namespace number_of_diagonals_in_hexagon_l179_179121

-- Define the number of sides of the hexagon
def sides_of_hexagon : ℕ := 6

-- Define the formula for the number of diagonals in an n-sided polygon
def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- The theorem we want to prove
theorem number_of_diagonals_in_hexagon : number_of_diagonals sides_of_hexagon = 9 :=
by
  sorry

end number_of_diagonals_in_hexagon_l179_179121


namespace tank_capacity_l179_179771

theorem tank_capacity (C : ℝ) (h : (3 / 4) * C + 9 = (7 / 8) * C) : C = 72 :=
sorry

end tank_capacity_l179_179771


namespace billy_picked_36_dandelions_initially_l179_179731

namespace Dandelions

/-- The number of dandelions Billy picked initially. -/
def billy_initial (B : ℕ) : ℕ := B

/-- The number of dandelions George picked initially. -/
def george_initial (B : ℕ) : ℕ := B / 3

/-- The additional dandelions picked by Billy and George respectively. -/
def additional_dandelions : ℕ := 10

/-- The total dandelions picked by Billy and George initially and additionally. -/
def total_dandelions (B : ℕ) : ℕ :=
  billy_initial B + additional_dandelions + george_initial B + additional_dandelions

/-- The average number of dandelions picked by both Billy and George, given as 34. -/
def average_dandelions (total : ℕ) : Prop := total / 2 = 34

/-- The main theorem stating that Billy picked 36 dandelions initially. -/
theorem billy_picked_36_dandelions_initially :
  ∀ B : ℕ, average_dandelions (total_dandelions B) ↔ B = 36 :=
by
  intro B
  sorry

end Dandelions

end billy_picked_36_dandelions_initially_l179_179731


namespace landscape_avoid_repetition_l179_179970

theorem landscape_avoid_repetition :
  let frames : ℕ := 5
  let days_per_month : ℕ := 30
  (Nat.factorial frames) / days_per_month = 4 := by
  sorry

end landscape_avoid_repetition_l179_179970


namespace company_profit_is_correct_l179_179721

structure CompanyInfo where
  num_employees : ℕ
  shirts_per_employee_per_day : ℕ
  hours_per_shift : ℕ
  wage_per_hour : ℕ
  bonus_per_shirt : ℕ
  price_per_shirt : ℕ
  nonemployee_expenses_per_day : ℕ

def daily_profit (info : CompanyInfo) : ℤ :=
  let total_shirts_per_day := info.num_employees * info.shirts_per_employee_per_day
  let total_revenue := total_shirts_per_day * info.price_per_shirt
  let daily_wage_per_employee := info.wage_per_hour * info.hours_per_shift
  let total_daily_wage := daily_wage_per_employee * info.num_employees
  let daily_bonus_per_employee := info.bonus_per_shirt * info.shirts_per_employee_per_day
  let total_daily_bonus := daily_bonus_per_employee * info.num_employees
  let total_labor_cost := total_daily_wage + total_daily_bonus
  let total_expenses := total_labor_cost + info.nonemployee_expenses_per_day
  total_revenue - total_expenses

theorem company_profit_is_correct (info : CompanyInfo) (h : 
  info.num_employees = 20 ∧
  info.shirts_per_employee_per_day = 20 ∧
  info.hours_per_shift = 8 ∧
  info.wage_per_hour = 12 ∧
  info.bonus_per_shirt = 5 ∧
  info.price_per_shirt = 35 ∧
  info.nonemployee_expenses_per_day = 1000
) : daily_profit info = 9080 := 
by
  sorry

end company_profit_is_correct_l179_179721


namespace sum_of_roots_l179_179058

-- States that the sum of the values of x that satisfy the given quadratic equation is 7
theorem sum_of_roots (x : ℝ) :
  (x^2 - 7 * x + 12 = 4) → (∃ a b : ℝ, x^2 - 7 * x + 8 = 0 ∧ a + b = 7) :=
by
  sorry

end sum_of_roots_l179_179058


namespace new_average_score_l179_179033

theorem new_average_score (avg_score : ℝ) (num_students : ℕ) (dropped_score : ℝ) (new_num_students : ℕ) :
  num_students = 16 →
  avg_score = 61.5 →
  dropped_score = 24 →
  new_num_students = num_students - 1 →
  (avg_score * num_students - dropped_score) / new_num_students = 64 :=
by
  sorry

end new_average_score_l179_179033


namespace parallel_line_distance_l179_179455

theorem parallel_line_distance 
    (A_upper : ℝ) (A_middle : ℝ) (A_lower : ℝ)
    (A_total : ℝ) (A_half : ℝ)
    (h_upper : A_upper = 3)
    (h_middle : A_middle = 5)
    (h_lower : A_lower = 2) 
    (h_total : A_total = A_upper + A_middle + A_lower)
    (h_half : A_half = A_total / 2) :
    ∃ d : ℝ, d = 2 + 0.6 ∧ A_middle * 0.6 = 3 := 
sorry

end parallel_line_distance_l179_179455


namespace cost_to_plant_flowers_l179_179648

theorem cost_to_plant_flowers :
  let cost_flowers := 9
  let cost_clay_pot := cost_flowers + 20
  let cost_soil := cost_flowers - 2
  cost_flowers + cost_clay_pot + cost_soil = 45 := 
by
  let cost_flowers := 9
  let cost_clay_pot := cost_flowers + 20
  let cost_soil := cost_flowers - 2
  show cost_flowers + cost_clay_pot + cost_soil = 45
  sorry

end cost_to_plant_flowers_l179_179648


namespace tina_total_time_l179_179511

-- Define constants for the problem conditions
def assignment_time : Nat := 20
def dinner_time : Nat := 17 * 60 + 30 -- 5:30 PM in minutes
def clean_time_per_key : Nat := 7
def total_keys : Nat := 30
def remaining_keys : Nat := total_keys - 1
def dry_time_per_key : Nat := 10
def break_time : Nat := 3
def keys_per_break : Nat := 5

-- Define a function to compute total cleaning time for remaining keys
def total_cleaning_time (keys : Nat) (clean_time : Nat) : Nat :=
  keys * clean_time

-- Define a function to compute total drying time for all keys
def total_drying_time (keys : Nat) (dry_time : Nat) : Nat :=
  keys * dry_time

-- Define a function to compute total break time
def total_break_time (keys : Nat) (keys_per_break : Nat) (break_time : Nat) : Nat :=
  (keys / keys_per_break) * break_time

-- Define a function to compute the total time including cleaning, drying, breaks, and assignment
def total_time (cleaning_time drying_time break_time assignment_time : Nat) : Nat :=
  cleaning_time + drying_time + break_time + assignment_time

-- The theorem to be proven
theorem tina_total_time : 
  total_time (total_cleaning_time remaining_keys clean_time_per_key) 
              (total_drying_time total_keys dry_time_per_key)
              (total_break_time total_keys keys_per_break break_time)
              assignment_time = 541 :=
by sorry

end tina_total_time_l179_179511


namespace product_of_reals_condition_l179_179263

theorem product_of_reals_condition (x : ℝ) (h : x + 1/x = 3 * x) : 
  ∃ x1 x2 : ℝ, x1 + 1/x1 = 3 * x1 ∧ x2 + 1/x2 = 3 * x2 ∧ x1 * x2 = -1/2 := 
sorry

end product_of_reals_condition_l179_179263


namespace part1_part2_l179_179118

noncomputable def f (x : ℝ) : ℝ := (Real.cos x) * (Real.cos (x - Real.pi / 3))

theorem part1 : f (2 * Real.pi / 3) = -1 / 4 :=
by
  sorry

theorem part2 :
  {x : ℝ | f x < 1 / 4} = {x : ℝ | ∃ k : ℤ, x ∈ Set.Ioo (k * Real.pi - 7 * Real.pi / 12) (k * Real.pi - Real.pi / 12)} :=
by
  sorry

end part1_part2_l179_179118


namespace vitamin_d3_total_days_l179_179642

def vitamin_d3_days (capsules_per_bottle : ℕ) (daily_serving_size : ℕ) (bottles_needed : ℕ) : ℕ :=
  (capsules_per_bottle / daily_serving_size) * bottles_needed

theorem vitamin_d3_total_days :
  vitamin_d3_days 60 2 6 = 180 :=
by
  sorry

end vitamin_d3_total_days_l179_179642


namespace divisibility_of_solutions_l179_179322

theorem divisibility_of_solutions (p : ℕ) (k : ℕ) (x₀ y₀ z₀ t₀ : ℕ) 
  (hp_prime : Nat.Prime p)
  (hp_form : p = 4 * k + 3)
  (h_eq : x₀^(2*p) + y₀^(2*p) + z₀^(2*p) = t₀^(2*p)) : 
  p ∣ x₀ ∨ p ∣ y₀ ∨ p ∣ z₀ ∨ p ∣ t₀ :=
sorry

end divisibility_of_solutions_l179_179322


namespace noodles_given_to_William_l179_179870

def initial_noodles : ℝ := 54.0
def noodles_left : ℝ := 42.0
def noodles_given : ℝ := initial_noodles - noodles_left

theorem noodles_given_to_William : noodles_given = 12.0 := 
by
  sorry -- Proof to be filled in

end noodles_given_to_William_l179_179870


namespace simplify_radical_expression_l179_179165

theorem simplify_radical_expression (x : ℝ) :
  (sqrt (50 * x)) * (sqrt (18 * x)) * (sqrt (32 * x)) = 120 * x * sqrt (2 * x) := by
sorry

end simplify_radical_expression_l179_179165


namespace find_circle_radius_l179_179890

theorem find_circle_radius (C D : Circle) (O P Q R L : Point) 
  (h1 : O ∈ C.circumference) 
  (h2 : P ∈ (circle_intersect C D O))
  (h3 : Q ∈ (circle_intersect C D O))
  (h4 : R ∈ (circle_intersect (Circle.mk Q D.radius) D))
  (h5 : L ∈ (line_intersect PR C.circumference)) :
  QL = C.radius :=
  sorry

end find_circle_radius_l179_179890


namespace integer_roots_of_polynomial_l179_179415

theorem integer_roots_of_polynomial : 
  {x : ℤ | x^3 - 6 * x^2 - 4 * x + 24 = 0} = {2, -2} :=
by
  sorry

end integer_roots_of_polynomial_l179_179415


namespace circle_equation_l179_179502

theorem circle_equation (x y : ℝ) : (3 * x - 4 * y + 12 = 0) → (x^2 + 4 * x + y^2 - 3 * y = 0) :=
sorry

end circle_equation_l179_179502


namespace average_next_3_numbers_l179_179034

theorem average_next_3_numbers 
  (a1 a2 b1 b2 b3 c1 c2 c3 : ℝ)
  (h_avg_total : (a1 + a2 + b1 + b2 + b3 + c1 + c2 + c3) / 8 = 25)
  (h_avg_first2: (a1 + a2) / 2 = 20)
  (h_c1_c2 : c1 + 4 = c2)
  (h_c1_c3 : c1 + 6 = c3)
  (h_c3_value : c3 = 30) :
  (b1 + b2 + b3) / 3 = 26 := 
sorry

end average_next_3_numbers_l179_179034


namespace max_cube_side_length_max_parallelepiped_dimensions_l179_179464

theorem max_cube_side_length (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) : 
    ∃ (a0 : ℝ), a0 = a * b * c / (a * b + b * c + a * c) := 
sorry

theorem max_parallelepiped_dimensions (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) : 
    ∃ (x y z : ℝ), (x = a / 3) ∧ (y = b / 3) ∧ (z = c / 3) :=
sorry

end max_cube_side_length_max_parallelepiped_dimensions_l179_179464


namespace Tammy_average_speed_second_day_l179_179635

theorem Tammy_average_speed_second_day : 
  ∀ (t v : ℝ), 
    (t + (t - 2) + (t + 1) = 20) → 
    (7 * v + 5 * (v + 0.5) + 8 * (v + 1.5) = 85) → 
    (v + 0.5 = 4.025) := 
by 
  intros t v ht hv 
  sorry

end Tammy_average_speed_second_day_l179_179635


namespace points_lie_on_circle_l179_179271

theorem points_lie_on_circle (s : ℝ) :
  ( (2 - s^2) / (2 + s^2) )^2 + ( 3 * s / (2 + s^2) )^2 = 1 :=
by sorry

end points_lie_on_circle_l179_179271


namespace A_B_finish_l179_179378

theorem A_B_finish (A B C : ℕ → ℝ) (h1 : A + B + C = 1 / 6) (h2 : C = 1 / 10) :
  1 / (A + B) = 15 :=
by
  sorry

end A_B_finish_l179_179378


namespace ancient_chinese_problem_l179_179913

theorem ancient_chinese_problem (x y : ℤ) 
  (h1 : y = 8 * x - 3) 
  (h2 : y = 7 * x + 4) : 
  (y = 8 * x - 3) ∧ (y = 7 * x + 4) :=
by
  exact ⟨h1, h2⟩

end ancient_chinese_problem_l179_179913


namespace can_predict_at_280_l179_179692

-- Definitions based on the problem conditions
def whiteRookStudents : Nat := 15
def blackElephantStudents : Nat := 20
def totalGames : Nat := whiteRookStudents * blackElephantStudents

-- Predicate to determine if Sasha can predict a participant
def canPredictNextParticipant (n : Nat) : Prop :=
  n >= 280

theorem can_predict_at_280 :
  ∀ n, n = 280 → canPredictNextParticipant n :=
begin
  intros,
  unfold canPredictNextParticipant,
  exact Nat.ge_of_eq (Eq.symm a),
end

end can_predict_at_280_l179_179692


namespace floor_plus_x_eq_205_l179_179749

theorem floor_plus_x_eq_205 (x : ℝ) (h : ⌊x⌋ + x = 20.5) : x = 10.5 :=
sorry

end floor_plus_x_eq_205_l179_179749


namespace find_p_q_sum_l179_179924

noncomputable def roots (r1 r2 r3 : ℝ) := (r1 + r2 + r3 = 11 ∧ r1 ≠ r2 ∧ r2 ≠ r3 ∧ r1 ≠ r3) ∧ 
                                         (∀ x : ℝ, x^3 - 11*x^2 + (r1 * r2 + r2 * r3 + r3 * r1) * x - r1 * r2 * r3 = 0)

theorem find_p_q_sum : ∃ (p q : ℝ), roots 2 4 5 → p + q = 78 :=
by
  sorry

end find_p_q_sum_l179_179924


namespace log_identity_proof_l179_179733

theorem log_identity_proof (lg : ℝ → ℝ) (h1 : lg 50 = lg 2 + lg 25) (h2 : lg 25 = 2 * lg 5) :
  (lg 2)^2 + lg 2 * lg 50 + lg 25 = 2 :=
by sorry

end log_identity_proof_l179_179733


namespace min_value_of_expression_l179_179881

theorem min_value_of_expression (x y : ℝ) (h1 : x > y) (h2 : y > 0) (h3 : 4 * x + 3 * y = 1) :
  1 / (2 * x - y) + 2 / (x + 2 * y) = 9 :=
sorry

end min_value_of_expression_l179_179881


namespace proof_problem_l179_179024

def problem_expression : ℚ := 1 / (2 + 1 / (Real.sqrt 5 + 2))

theorem proof_problem : problem_expression = Real.sqrt 5 / 5 := by sorry

end proof_problem_l179_179024


namespace intersection_of_M_and_N_l179_179761

def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {x | ∃ a ∈ M, x = a^2}

theorem intersection_of_M_and_N :
  (M ∩ N = {0, 1}) :=
by
  sorry

end intersection_of_M_and_N_l179_179761


namespace arc_length_sector_l179_179541

theorem arc_length_sector (r : ℝ) (θ : ℝ) (h_r : r = 3) (h_θ : θ = 150 * Real.pi / 180) :
  θ * r = 5 * Real.pi / 2 :=
by
  rw [h_r, h_θ]
  sorry

end arc_length_sector_l179_179541


namespace average_speed_correct_l179_179843

-- Define the speeds for each hour
def speed_hour1 := 90 -- km/h
def speed_hour2 := 40 -- km/h
def speed_hour3 := 60 -- km/h
def speed_hour4 := 80 -- km/h
def speed_hour5 := 50 -- km/h

-- Define the total time of the journey
def total_time := 5 -- hours

-- Calculate the sum of distances
def total_distance := speed_hour1 + speed_hour2 + speed_hour3 + speed_hour4 + speed_hour5

-- Define the average speed calculation
def average_speed := total_distance / total_time

-- The proof problem: average speed is 64 km/h
theorem average_speed_correct : average_speed = 64 := by
  sorry

end average_speed_correct_l179_179843


namespace rectangle_area_unchanged_l179_179949

theorem rectangle_area_unchanged (x y : ℕ) (h1 : x * y = (x + 5/2) * (y - 2/3)) (h2 : x * y = (x - 5/2) * (y + 4/3)) : x * y = 20 :=
by
  sorry

end rectangle_area_unchanged_l179_179949


namespace infinite_n_divisible_by_p_l179_179155

theorem infinite_n_divisible_by_p (p : ℕ) (hp : Nat.Prime p) : 
  ∃ᶠ n in Filter.atTop, p ∣ (2^n - n) :=
by
  sorry

end infinite_n_divisible_by_p_l179_179155


namespace find_x_l179_179773

theorem find_x (x y : ℝ) :
  (x / (x - 1) = (y^2 + 3*y + 2) / (y^2 + 3*y - 1)) →
  x = (y^2 + 3*y + 2) / 3 :=
by
  intro h
  sorry

end find_x_l179_179773


namespace stickers_remaining_l179_179619

theorem stickers_remaining (total_stickers : ℕ) (front_page_stickers : ℕ) (other_pages_stickers : ℕ) (num_other_pages : ℕ) (remaining_stickers : ℕ)
  (h0 : total_stickers = 89)
  (h1 : front_page_stickers = 3)
  (h2 : other_pages_stickers = 7)
  (h3 : num_other_pages = 6)
  (h4 : remaining_stickers = total_stickers - (front_page_stickers + other_pages_stickers * num_other_pages)) :
  remaining_stickers = 44 :=
by
  sorry

end stickers_remaining_l179_179619


namespace quadratic_inequality_solution_set_l179_179416

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 - 3 * x + 2 ≤ 0} = {x : ℝ | 1 ≤ x ∧ x ≤ 2} :=
sorry

end quadratic_inequality_solution_set_l179_179416


namespace day_of_week_after_10_pow_90_days_l179_179281

theorem day_of_week_after_10_pow_90_days :
  let initial_day := "Friday"
  ∃ day_after_10_pow_90 : String,
  day_after_10_pow_90 = "Saturday" :=
by
  sorry

end day_of_week_after_10_pow_90_days_l179_179281


namespace length_of_second_train_correct_l179_179512

noncomputable def length_of_second_train : ℝ :=
  let speed_first_train := 60 / 3.6
  let speed_second_train := 90 / 3.6
  let relative_speed := speed_first_train + speed_second_train
  let time_to_clear := 6.623470122390208
  let total_distance := relative_speed * time_to_clear
  let length_first_train := 111
  total_distance - length_first_train

theorem length_of_second_train_correct :
  length_of_second_train = 164.978 :=
by
  unfold length_of_second_train
  sorry

end length_of_second_train_correct_l179_179512


namespace ratio_of_doctors_to_engineers_l179_179639

variables (d l e : ℕ) -- number of doctors, lawyers, and engineers

-- Conditions
def avg_age := (40 * d + 55 * l + 50 * e) / (d + l + e) = 45
def doctors_avg := 40 
def lawyers_avg := 55 
def engineers_avg := 50 -- 55 - 5

theorem ratio_of_doctors_to_engineers (h_avg : avg_age d l e) : d = 3 * e :=
sorry

end ratio_of_doctors_to_engineers_l179_179639


namespace sum_of_next_17_consecutive_integers_l179_179958

theorem sum_of_next_17_consecutive_integers (x : ℤ) (h₁ : (List.range 17).sum + 17 * x = 306) :
  (List.range 17).sum + 17 * (x + 17)  = 595 := 
sorry

end sum_of_next_17_consecutive_integers_l179_179958


namespace gcd_difference_5610_210_10_l179_179750

theorem gcd_difference_5610_210_10 : Int.gcd 5610 210 - 10 = 20 := by
  sorry

end gcd_difference_5610_210_10_l179_179750


namespace quadratic_solution_factoring_solution_l179_179632

-- Define the first problem: Solve 2x^2 - 6x - 5 = 0
theorem quadratic_solution (x : ℝ) : 2 * x^2 - 6 * x - 5 = 0 ↔ x = (3 + Real.sqrt 19) / 2 ∨ x = (3 - Real.sqrt 19) / 2 :=
by
  sorry

-- Define the second problem: Solve 3x(4-x) = 2(x-4)
theorem factoring_solution (x : ℝ) : 3 * x * (4 - x) = 2 * (x - 4) ↔ x = 4 ∨ x = -2 / 3 :=
by
  sorry

end quadratic_solution_factoring_solution_l179_179632


namespace find_uv_non_integer_l179_179140

noncomputable def q (x y : ℝ) (b : ℕ → ℝ) := 
  b 0 + b 1 * x + b 2 * y + b 3 * x^2 + b 4 * x * y + b 5 * y^2 + 
  b 6 * x^3 + b 7 * x^2 * y + b 8 * x * y^2 + b 9 * y^3

theorem find_uv_non_integer (b : ℕ → ℝ) 
  (h0 : q 0 0 b = 0) 
  (h1 : q 1 0 b = 0) 
  (h2 : q (-1) 0 b = 0) 
  (h3 : q 0 1 b = 0) 
  (h4 : q 0 (-1) b = 0) 
  (h5 : q 1 1 b = 0) 
  (h6 : q 1 (-1) b = 0) 
  (h7 : q 3 3 b = 0) : 
  ∃ u v : ℝ, q u v b = 0 ∧ u = 17/19 ∧ v = 18/19 := 
  sorry

end find_uv_non_integer_l179_179140


namespace train_cars_count_l179_179976

theorem train_cars_count
  (cars_in_first_15_seconds : ℕ)
  (time_for_first_5_cars : ℕ)
  (total_time_to_pass : ℕ)
  (h_cars_in_first_15_seconds : cars_in_first_15_seconds = 5)
  (h_time_for_first_5_cars : time_for_first_5_cars = 15)
  (h_total_time_to_pass : total_time_to_pass = 210) :
  (total_time_to_pass / time_for_first_5_cars) * cars_in_first_15_seconds = 70 := 
by 
  sorry

end train_cars_count_l179_179976


namespace trajectory_of_M_l179_179285

variable (P : ℝ × ℝ) (A : ℝ × ℝ := (4, 0))
variable (M : ℝ × ℝ)

theorem trajectory_of_M (hP : P.1^2 + 4 * P.2^2 = 4) (hM : M = ((P.1 + 4) / 2, P.2 / 2)) :
  (M.1 - 2)^2 + 4 * M.2^2 = 1 :=
by
  sorry

end trajectory_of_M_l179_179285


namespace ratio_of_boys_to_girls_l179_179864

theorem ratio_of_boys_to_girls (G B : ℕ) (hg : G = 30) (hb : B = G + 18) : B / G = 8 / 5 :=
by
  sorry

end ratio_of_boys_to_girls_l179_179864


namespace lulu_final_cash_l179_179934

-- Definitions of the problem conditions
def initial_amount : ℕ := 65
def spent_on_ice_cream : ℕ := 5
def spent_on_tshirt (remaining : ℕ) : ℕ := remaining / 2
def deposit_in_bank (remaining : ℕ) : ℕ := remaining / 5

-- The proof problem statement
theorem lulu_final_cash :
  ∃ final_cash : ℕ,
    final_cash = initial_amount - spent_on_ice_cream - spent_on_tshirt (initial_amount - spent_on_ice_cream) - 
                      deposit_in_bank (spent_on_tshirt (initial_amount - spent_on_ice_cream)) ∧
    final_cash = 24 :=
by {
  sorry
}

end lulu_final_cash_l179_179934


namespace lulu_cash_left_l179_179932

theorem lulu_cash_left :
  ∀ (initial money spentIceCream spentTshirt deposited finalCash: ℝ),
    initial = 65 →
    spentIceCream = 5 →
    spentTshirt = 0.5 * (initial - spentIceCream) →
    deposited = (initial - spentIceCream - spentTshirt) / 5 →
    finalCash = initial - spentIceCream - spentTshirt - deposited →
    finalCash = 24 :=
by
  intros
  sorry

end lulu_cash_left_l179_179932


namespace sin_cos_value_l179_179579

-- Given function definition
noncomputable def f (α : ℝ) (x : ℝ) : ℝ := x^2 + (Real.sin α - 2 * Real.cos α) * x + 1

-- Definitions and proof problem statement
theorem sin_cos_value (α : ℝ) : 
  (∀ x : ℝ, f α x = f α (-x)) → (Real.sin α * Real.cos α = 2 / 5) :=
by
  intro h_even
  sorry

end sin_cos_value_l179_179579


namespace delta_minus2_3_eq_minus14_l179_179552

def delta (a b : Int) : Int := a * b^2 + b + 1

theorem delta_minus2_3_eq_minus14 : delta (-2) 3 = -14 :=
by
  sorry

end delta_minus2_3_eq_minus14_l179_179552


namespace angle_is_40_l179_179726

theorem angle_is_40 (x : ℝ) 
  : (180 - x = 2 * (90 - x) + 40) → x = 40 :=
by
  sorry

end angle_is_40_l179_179726


namespace perfect_square_proof_l179_179965

noncomputable def isPerfectSquare (f : ℚ[X]) : Prop :=
  ∃ g : ℚ[X], f = g^2

variable {a a' b b' c c' : ℚ}

def condition1 : Prop := isPerfectSquare ((a + b * X)^2 + (a' + b' * X)^2)
def condition2 : Prop := isPerfectSquare ((a + c * X)^2 + (a' + c' * X)^2)

theorem perfect_square_proof (h1 : condition1) (h2 : condition2) :
  isPerfectSquare ((b + c * X)^2 + (b' + c' * X)^2) := 
sorry

end perfect_square_proof_l179_179965


namespace cat_weights_ratio_l179_179509

variable (meg_cat_weight : ℕ) (anne_extra_weight : ℕ) (meg_cat_weight := 20) (anne_extra_weight := 8)

/-- The ratio of the weight of Meg's cat to the weight of Anne's cat -/
theorem cat_weights_ratio : (meg_cat_weight / Nat.gcd meg_cat_weight (meg_cat_weight + anne_extra_weight)) 
                            = 5 ∧ ((meg_cat_weight + anne_extra_weight) / Nat.gcd meg_cat_weight (meg_cat_weight + anne_extra_weight)) 
                            = 7 := by
  sorry

end cat_weights_ratio_l179_179509


namespace total_games_played_l179_179866

def games_attended : ℕ := 14
def games_missed : ℕ := 25

theorem total_games_played : games_attended + games_missed = 39 :=
by
  sorry

end total_games_played_l179_179866


namespace a1_b1_sum_l179_179438

-- Definitions from the conditions:
def strict_inc_seq (s : ℕ → ℕ) : Prop := ∀ n, s n < s (n + 1)

def positive_int_seq (s : ℕ → ℕ) : Prop := ∀ n, s n > 0

def a : ℕ → ℕ := sorry -- Define the sequence 'a' (details skipped).

def b : ℕ → ℕ := sorry -- Define the sequence 'b' (details skipped).

-- Conditions given:
axiom cond_a_inc : strict_inc_seq a

axiom cond_b_inc : strict_inc_seq b

axiom cond_a_pos : positive_int_seq a

axiom cond_b_pos : positive_int_seq b

axiom cond_a10_b10_lt_2017 : a 10 = b 10 ∧ a 10 < 2017

axiom cond_a_rec : ∀ n, a (n + 2) = a (n + 1) + a n

axiom cond_b_rec : ∀ n, b (n + 1) = 2 * b n

-- The theorem to prove:
theorem a1_b1_sum : a 1 + b 1 = 5 :=
sorry

end a1_b1_sum_l179_179438


namespace indoor_tables_count_l179_179348

theorem indoor_tables_count
  (I : ℕ)  -- the number of indoor tables
  (O : ℕ)  -- the number of outdoor tables
  (H1 : O = 12)  -- Condition 1: O = 12
  (H2 : 3 * I + 3 * O = 60)  -- Condition 2: Total number of chairs
  : I = 8 :=
by
  -- Insert the actual proof here
  sorry

end indoor_tables_count_l179_179348


namespace weekly_milk_production_l179_179708

-- Define the conditions
def num_cows : ℕ := 52
def milk_per_cow_per_day : ℕ := 5
def days_per_week : ℕ := 7

-- Define the proof that total weekly milk production is 1820 liters
theorem weekly_milk_production : num_cows * milk_per_cow_per_day * days_per_week = 1820 := by
  sorry

end weekly_milk_production_l179_179708


namespace largest_base_5_three_digit_in_base_10_l179_179209

theorem largest_base_5_three_digit_in_base_10 :
  let n := 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  n = 124 :=
by
  let n := 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  show n = 124
  sorry

end largest_base_5_three_digit_in_base_10_l179_179209


namespace interestDifference_l179_179973

noncomputable def simpleInterest (P R T : ℝ) : ℝ :=
  (P * R * T) / 100

noncomputable def compoundInterest (P R T : ℝ) : ℝ :=
  P * (1 + R / 100)^T - P

theorem interestDifference (P R T : ℝ) (hP : P = 500) (hR : R = 20) (hT : T = 2) :
  compoundInterest P R T - simpleInterest P R T = 120 := by
  sorry

end interestDifference_l179_179973


namespace gilbert_herb_plants_l179_179275

theorem gilbert_herb_plants
  (initial_basil : ℕ)
  (initial_parsley : ℕ)
  (initial_mint : ℕ)
  (extra_basil_mid_spring : ℕ)
  (mint_eaten_by_rabbit : ℕ) :
  initial_basil = 3 →
  initial_parsley = 1 →
  initial_mint = 2 →
  extra_basil_mid_spring = 1 →
  mint_eaten_by_rabbit = 2 →
  initial_basil + initial_parsley + initial_mint + extra_basil_mid_spring - mint_eaten_by_rabbit = 5 :=
by
  intros h_basil h_parsley h_mint h_extra h_eaten
  simp [h_basil, h_parsley, h_mint, h_extra, h_eaten]
  done
  sorry

end gilbert_herb_plants_l179_179275


namespace gcd_incorrect_l179_179974

theorem gcd_incorrect (a b c : ℕ) (h : a * b * c = 3000) : gcd (gcd a b) c ≠ 15 := 
sorry

end gcd_incorrect_l179_179974


namespace polar_equation_is_circle_of_radius_five_l179_179095

theorem polar_equation_is_circle_of_radius_five :
  ∀ θ : ℝ, (3 * Real.sin θ + 4 * Real.cos θ) ^ 2 = 25 :=
by
  sorry

end polar_equation_is_circle_of_radius_five_l179_179095


namespace solve_for_y_l179_179683

theorem solve_for_y (x y : ℝ) (h1 : x * y = 9) (h2 : x / y = 36) (h3 : x > 0) (h4 : y > 0) : y = 1/2 := 
by
  sorry

end solve_for_y_l179_179683


namespace find_dividend_l179_179668

-- Given conditions as definitions
def divisor : ℕ := 16
def quotient : ℕ := 9
def remainder : ℕ := 5

-- Lean 4 statement to be proven
theorem find_dividend : divisor * quotient + remainder = 149 := by
  sorry

end find_dividend_l179_179668


namespace mixture_milk_quantity_l179_179537

variable (M W : ℕ)

theorem mixture_milk_quantity
  (h1 : M = 2 * W)
  (h2 : 6 * (W + 10) = 5 * M) :
  M = 30 := by
  sorry

end mixture_milk_quantity_l179_179537


namespace mark_asphalt_total_cost_l179_179803

noncomputable def total_cost (road_length : ℕ) (road_width : ℕ) (area_per_truckload : ℕ) (cost_per_truckload : ℕ) (sales_tax_rate : ℚ) : ℚ :=
  let total_area := road_length * road_width
  let num_truckloads := total_area / area_per_truckload
  let cost_before_tax := num_truckloads * cost_per_truckload
  let sales_tax := cost_before_tax * sales_tax_rate
  let total_cost := cost_before_tax + sales_tax
  total_cost

theorem mark_asphalt_total_cost :
  total_cost 2000 20 800 75 0.2 = 4500 := 
by sorry

end mark_asphalt_total_cost_l179_179803


namespace case_one_ellipses_foci_xaxis_case_two_ellipses_foci_exact_l179_179875

theorem case_one_ellipses_foci_xaxis :
  ∀ (a : ℝ) (e : ℝ), a = 6 ∧ e = 2 / 3 → (∃ (b : ℝ), (b^2 = (a^2 - (e * a)^2) ∧ (a > 0) → ((∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1)) ∨ (y^2 / a^2 + x^2 / b^2 = 1)))) :=
by
  sorry

theorem case_two_ellipses_foci_exact :
  ∀ (F1 F2 : ℝ × ℝ), F1 = (-4,0) ∧ F2 = (4,0) ∧ ∀ P : ℝ × ℝ, ((dist P F1) + (dist P F2) = 10) →
  ∃ (a : ℝ) (b : ℝ), a = 5 ∧ b^2 = a^2 - 4^2 → ((∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1))) :=
by
  sorry

end case_one_ellipses_foci_xaxis_case_two_ellipses_foci_exact_l179_179875


namespace smallest_integer_proof_l179_179794

noncomputable def smallestInteger (s : ℝ) (h : s < 1 / 2000) : ℤ :=
  Nat.ceil (Real.sqrt (1999 / 3))

theorem smallest_integer_proof (s : ℝ) (h : s < 1 / 2000) (m : ℤ) (hm : m = (smallestInteger s h + s)^3) : smallestInteger s h = 26 :=
by 
  sorry

end smallest_integer_proof_l179_179794


namespace quadratic_two_roots_l179_179042

theorem quadratic_two_roots (b : ℝ) : 
  ∃ (x₁ x₂ : ℝ), (∀ x : ℝ, (x = x₁ ∨ x = x₂) ↔ (x^2 + b*x - 3 = 0)) :=
by
  -- Indicate that a proof is required here
  sorry

end quadratic_two_roots_l179_179042


namespace probability_neither_red_nor_purple_l179_179986

def total_balls : ℕ := 60
def white_balls : ℕ := 22
def green_balls : ℕ := 18
def yellow_balls : ℕ := 8
def red_balls : ℕ := 5
def purple_balls : ℕ := 7

theorem probability_neither_red_nor_purple : 
  (total_balls - (red_balls + purple_balls)) / total_balls = 4 / 5 :=
by sorry

end probability_neither_red_nor_purple_l179_179986


namespace quadratic_binomial_plus_int_l179_179036

theorem quadratic_binomial_plus_int (y : ℝ) : y^2 + 14*y + 60 = (y + 7)^2 + 11 :=
by sorry

end quadratic_binomial_plus_int_l179_179036


namespace min_cost_to_fence_land_l179_179333

theorem min_cost_to_fence_land (w l : ℝ) (h1 : l = 2 * w) (h2 : 2 * w ^ 2 ≥ 500) : 
  5 * (2 * (l + w)) = 150 * Real.sqrt 10 := 
by
  sorry

end min_cost_to_fence_land_l179_179333


namespace hollow_cylinder_surface_area_l179_179707

theorem hollow_cylinder_surface_area (h : ℝ) (r_outer r_inner : ℝ) (h_eq : h = 12) (r_outer_eq : r_outer = 5) (r_inner_eq : r_inner = 2) :
  (2 * π * ((r_outer ^ 2 - r_inner ^ 2)) + 2 * π * r_outer * h + 2 * π * r_inner * h) = 210 * π :=
by
  rw [h_eq, r_outer_eq, r_inner_eq]
  sorry

end hollow_cylinder_surface_area_l179_179707


namespace longest_tape_length_l179_179517

theorem longest_tape_length (a b c : ℕ) (h1 : a = 600) (h2 : b = 500) (h3 : c = 1200) : Nat.gcd (Nat.gcd a b) c = 100 :=
by
  sorry

end longest_tape_length_l179_179517


namespace min_dot_product_l179_179798

noncomputable def vec_a (m : ℝ) : ℝ × ℝ := (1 + 2^m, 1 - 2^m)
noncomputable def vec_b (m : ℝ) : ℝ × ℝ := (4^m - 3, 4^m + 5)

def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem min_dot_product : ∃ m : ℝ, dot_product (vec_a m) (vec_b m) = -6 := by
  sorry

end min_dot_product_l179_179798


namespace solve_system_l179_179168

theorem solve_system 
    (x y z : ℝ) 
    (h1 : x + y - 2 + 4 * x * y = 0) 
    (h2 : y + z - 2 + 4 * y * z = 0) 
    (h3 : z + x - 2 + 4 * z * x = 0) :
    (x = -1 ∧ y = -1 ∧ z = -1) ∨ (x = 1/2 ∧ y = 1/2 ∧ z = 1/2) :=
sorry

end solve_system_l179_179168


namespace solve_inequality_l179_179094

noncomputable def inequality_solution : Set ℝ :=
  { x | x^2 / (x + 2) ≥ 3 / (x - 2) + 7 / 4 }

theorem solve_inequality :
  inequality_solution = { x | -2 < x ∧ x < 2 } ∪ { x | 3 ≤ x } :=
by
  sorry

end solve_inequality_l179_179094


namespace solve_for_x_l179_179297

theorem solve_for_x :
  ∀ (x y : ℚ), (3 * x - 4 * y = 8) → (2 * x + 3 * y = 1) → x = 28 / 17 :=
by
  intros x y h1 h2
  sorry

end solve_for_x_l179_179297


namespace condition_holds_iff_b_eq_10_l179_179615

-- Define xn based on given conditions in the problem
def x_n (b : ℕ) (n : ℕ) : ℕ :=
  if b > 5 then
    b^(2*n) + b^n + 3*b - 5
  else
    0

-- State the main theorem to be proven in Lean
theorem condition_holds_iff_b_eq_10 :
  ∀ (b : ℕ), (b > 5) ↔ ∃ M : ℕ, ∀ n : ℕ, n > M → ∃ k : ℕ, x_n b n = k^2 := sorry

end condition_holds_iff_b_eq_10_l179_179615


namespace sequence_general_term_l179_179918

theorem sequence_general_term (a : ℕ → ℕ) (h₁ : a 1 = 1) (h₂ : ∀ n ≥ 1, a (n + 1) = a n + 2) :
  ∀ n, a n = 2 * n - 1 :=
by
  sorry

end sequence_general_term_l179_179918


namespace minimum_value_of_expression_l179_179144

theorem minimum_value_of_expression (p q r s t u : ℝ) 
  (hpqrsu_pos : 0 < p ∧ 0 < q ∧ 0 < r ∧ 0 < s ∧ 0 < t ∧ 0 < u) 
  (sum_eq : p + q + r + s + t + u = 8) : 
  98 ≤ (2 / p + 4 / q + 9 / r + 16 / s + 25 / t + 36 / u) :=
sorry

end minimum_value_of_expression_l179_179144


namespace sum_of_possible_values_l179_179955

theorem sum_of_possible_values (N : ℝ) (h : N * (N - 4) = -21) :
    ∃ N₁ N₂ : ℝ, (N₁ + N₂ = 4 ∧ N₁ * (N₁ - 4) = -21 ∧ N₂ * (N₂ - 4) = -21) :=
sorry

end sum_of_possible_values_l179_179955


namespace perpendicular_vectors_l179_179666

theorem perpendicular_vectors (a : ℝ) 
  (v1 : ℝ × ℝ := (4, -5))
  (v2 : ℝ × ℝ := (a, 2))
  (perpendicular : v1.fst * v2.fst + v1.snd * v2.snd = 0) :
  a = 5 / 2 :=
sorry

end perpendicular_vectors_l179_179666


namespace least_number_divisible_by_forth_number_l179_179354

theorem least_number_divisible_by_forth_number
  (n : ℕ) (h1 : n = 856)
  (h2 : (n + 8) % 24 = 0)
  (h3 : (n + 8) % 32 = 0)
  (h4 : (n + 8) % 36 = 0) :
  ∃ x, x = 3 ∧ (n + 8) % x = 0 :=
by
  sorry

end least_number_divisible_by_forth_number_l179_179354


namespace minimum_value_inequality_l179_179571

noncomputable def min_value (a b : ℝ) (ha : 0 < a) (hb : 1 < b) (hab : a + b = 2) : ℝ :=
  (4 / a) + (1 / (b - 1))

theorem minimum_value_inequality (a b : ℝ) (ha : 0 < a) (hb : 1 < b) (hab : a + b = 2) : 
  min_value a b ha hb hab ≥ 9 :=
  sorry

end minimum_value_inequality_l179_179571


namespace coordinates_of_B_l179_179886

theorem coordinates_of_B (x y : ℝ) (A : ℝ × ℝ) (a : ℝ × ℝ) :
  A = (2, 4) ∧ a = (3, 4) ∧ (x - 2, y - 4) = (2 * a.1, 2 * a.2) → (x, y) = (8, 12) :=
by
  intros h
  sorry

end coordinates_of_B_l179_179886


namespace predict_participant_after_280_games_l179_179699

-- Definitions according to the problem conditions
def numWhiteRook : Nat := 15
def numBlackElephant : Nat := 20
def totalGames : Nat := numWhiteRook * numBlackElephant

-- The theorem to be proved based on the conditions and desired result
theorem predict_participant_after_280_games (n : Nat) (h1 : n = 280) :
  ∃ participant, participant ∈ (some_game_participants n) :=
sorry

end predict_participant_after_280_games_l179_179699


namespace length_of_AC_l179_179135

variable (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
variables (AB BC AC : ℝ)
variables (right_triangle : AB ^ 2 + BC ^ 2 = AC ^ 2)
variables (tan_A : BC / AB = 4 / 3)
variable (AB_val : AB = 4)

theorem length_of_AC :
  AC = 20 / 3 :=
sorry

end length_of_AC_l179_179135


namespace jane_buys_four_bagels_l179_179606

-- Define Jane's 7-day breakfast choices
def number_of_items (b m : ℕ) := b + m = 7

-- Define the total weekly cost condition
def total_cost_divisible_by_100 (b : ℕ) := (90 * b + 40 * (7 - b)) % 100 = 0

-- The statement to prove
theorem jane_buys_four_bagels (b : ℕ) (m : ℕ) (h1 : number_of_items b m) (h2 : total_cost_divisible_by_100 b) : b = 4 :=
by
  -- proof goes here
  sorry

end jane_buys_four_bagels_l179_179606


namespace difference_sweaters_Monday_Tuesday_l179_179138

-- Define conditions
def sweaters_knit_on_Monday : ℕ := 8
def sweaters_knit_on_Tuesday (T : ℕ) : Prop := T > 8
def sweaters_knit_on_Wednesday (T : ℕ) : ℕ := T - 4
def sweaters_knit_on_Thursday (T : ℕ) : ℕ := T - 4
def sweaters_knit_on_Friday : ℕ := 4

-- Define total sweaters knit in the week
def total_sweaters_knit (T : ℕ) : ℕ :=
  sweaters_knit_on_Monday + T + sweaters_knit_on_Wednesday T + sweaters_knit_on_Thursday T + sweaters_knit_on_Friday

-- Lean Theorem Statement
theorem difference_sweaters_Monday_Tuesday : ∀ T : ℕ, sweaters_knit_on_Tuesday T → total_sweaters_knit T = 34 → T - sweaters_knit_on_Monday = 2 :=
by
  intros T hT_total
  sorry

end difference_sweaters_Monday_Tuesday_l179_179138


namespace range_of_a_l179_179006

noncomputable def proposition_p (a : ℝ) : Prop := 
  0 < a ∧ a < 1

noncomputable def proposition_q (a : ℝ) : Prop := 
  a > 1 / 4

theorem range_of_a (a : ℝ) : 
  (proposition_p a ∨ proposition_q a) ∧ ¬(proposition_p a ∧ proposition_q a) ↔
  (0 < a ∧ a ≤ 1 / 4) ∨ (a ≥ 1) :=
by sorry

end range_of_a_l179_179006


namespace find_constants_and_formula_l179_179783

namespace ArithmeticSequence

variable {a : ℕ → ℤ} -- Sequence a : ℕ → ℤ

-- Given conditions
axiom a_5 : a 5 = 11
axiom a_12 : a 12 = 31

-- Definitions to be proved
def a_1 := -2
def d := 3
def a_formula (n : ℕ) := a_1 + (n - 1) * d

theorem find_constants_and_formula :
  (a 1 = a_1) ∧
  (a 2 - a 1 = d) ∧
  (a 20 = 55) ∧
  (∀ n, a n = a_formula n) := by
  sorry

end ArithmeticSequence

end find_constants_and_formula_l179_179783


namespace total_length_of_rope_l179_179077

theorem total_length_of_rope (x : ℝ) : (∃ r1 r2 : ℝ, r1 / r2 = 2 / 3 ∧ r1 = 16 ∧ x = r1 + r2) → x = 40 :=
by
  intro h
  cases' h with r1 hr
  cases' hr with r2 hs
  sorry

end total_length_of_rope_l179_179077


namespace sum_of_excluded_solutions_l179_179922

noncomputable def P : ℚ := 3
noncomputable def Q : ℚ := 5 / 3
noncomputable def R : ℚ := 25 / 3

theorem sum_of_excluded_solutions :
    (P = 3) ∧
    (Q = 5 / 3) ∧
    (R = 25 / 3) ∧
    (∀ x, (x ≠ -R ∧ x ≠ -10) →
    ((x + Q) * (P * x + 50) / ((x + R) * (x + 10)) = 3)) →
    (-R + -10 = -55 / 3) :=
by
  sorry

end sum_of_excluded_solutions_l179_179922


namespace rectangle_circle_diameter_l179_179641

theorem rectangle_circle_diameter:
  ∀ (m n : ℕ), (∃ (x : ℚ), m + n = 47 ∧ (∀ (r : ℚ), r = (20 / 7)) →
  (2 * r = (40 / 7))) :=
by
  sorry

end rectangle_circle_diameter_l179_179641


namespace points_on_x_axis_circles_intersect_l179_179112

theorem points_on_x_axis_circles_intersect (a b : ℤ)
  (h1 : 3 * a - b = 9)
  (h2 : 2 * a + 3 * b = -5) : (a : ℝ)^b = 1/8 :=
by
  sorry

end points_on_x_axis_circles_intersect_l179_179112


namespace calculate_expression_l179_179822

theorem calculate_expression (x : ℝ) (h : x + 1/x = 3) : x^12 - 7 * x^6 + x^2 = 45363 * x - 17327 :=
by
  sorry

end calculate_expression_l179_179822


namespace circumscribed_sphere_radius_l179_179181

noncomputable def radius_of_circumscribed_sphere (a : ℝ) (α : ℝ) : ℝ :=
  a / (3 * Real.sin α)

theorem circumscribed_sphere_radius (a α : ℝ) :
  radius_of_circumscribed_sphere a α = a / (3 * Real.sin α) :=
by
  sorry

end circumscribed_sphere_radius_l179_179181


namespace roads_going_outside_city_l179_179778

theorem roads_going_outside_city (n : ℕ)
  (h : ∃ (x1 x2 x3 : ℕ), x1 + x2 + x3 = 3 ∧
    (n + x1) % 2 = 0 ∧
    (n + x2) % 2 = 0 ∧
    (n + x3) % 2 = 0) :
  ∃ (x1 x2 x3 : ℕ), (x1 = 1) ∧ (x2 = 1) ∧ (x3 = 1) :=
by 
  sorry

end roads_going_outside_city_l179_179778


namespace pq_conditions_l179_179294

theorem pq_conditions (p q : ℝ) (hp : p > 1) (hq : q > 1) (hq_inverse : 1 / p + 1 / q = 1) (hpq : p * q = 9) :
  (p = (9 + 3 * Real.sqrt 5) / 2 ∧ q = (9 - 3 * Real.sqrt 5) / 2) ∨ (p = (9 - 3 * Real.sqrt 5) / 2 ∧ q = (9 + 3 * Real.sqrt 5) / 2) :=
  sorry

end pq_conditions_l179_179294


namespace triangle_area_given_conditions_l179_179319

theorem triangle_area_given_conditions (a b c : ℝ) (C : ℝ) 
  (h1 : c^2 = (a - b)^2 + 6) (h2 : C = Real.pi / 3) : 
  (1/2) * a * b * Real.sin C = 3 * Real.sqrt 3 / 2 :=
by
  sorry

end triangle_area_given_conditions_l179_179319


namespace possible_slopes_of_line_intersecting_ellipse_l179_179534

theorem possible_slopes_of_line_intersecting_ellipse :
  ∃ m : ℝ, 
    (∀ (x y : ℝ), (y = m * x + 3) → (4 * x^2 + 25 * y^2 = 100)) →
    (m ∈ Set.Iio (-Real.sqrt (16 / 405)) ∪ Set.Ici (Real.sqrt (16 / 405))) :=
sorry

end possible_slopes_of_line_intersecting_ellipse_l179_179534


namespace integral_result_l179_179768

theorem integral_result (b : ℝ) (h : ∫ x in e..b, (2 / x) = 6) : b = Real.exp 4 :=
sorry

end integral_result_l179_179768


namespace initial_marbles_l179_179788

theorem initial_marbles (total_marbles now found: ℕ) (h_found: found = 7) (h_now: now = 28) : 
  total_marbles = now - found → total_marbles = 21 := by
  -- Proof goes here.
  sorry

end initial_marbles_l179_179788


namespace box_dimensions_l179_179382

theorem box_dimensions (x : ℝ) (bow_length_top bow_length_side : ℝ)
  (h1 : bow_length_top = 156 - 6 * x)
  (h2 : bow_length_side = 178 - 7 * x)
  (h_eq : bow_length_top = bow_length_side) :
  x = 22 :=
by sorry

end box_dimensions_l179_179382


namespace novel_pages_total_l179_179735

-- Definitions based on conditions
def pages_first_two_days : ℕ := 2 * 50
def pages_next_four_days : ℕ := 4 * 25
def pages_six_days : ℕ := pages_first_two_days + pages_next_four_days
def pages_seventh_day : ℕ := 30
def total_pages : ℕ := pages_six_days + pages_seventh_day

-- Statement of the problem as a theorem in Lean 4
theorem novel_pages_total : total_pages = 230 := by
  sorry

end novel_pages_total_l179_179735


namespace coefficient_x_5_2_in_expansion_of_2x_minus_sqrt_inv_x_l179_179126

noncomputable def binomialCoefficient (n k : ℕ) : ℚ := (Nat.factorial n : ℚ) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem coefficient_x_5_2_in_expansion_of_2x_minus_sqrt_inv_x (n : ℕ) (h : n = 4) :
  let term := λ r, binomialCoefficient 4 r * (2 ^ (4 - r)) * ((-1) ^ r) * (x ^ (4 - (3 / 2) * r)) in
  term 1 = -32 * (x ^ (5 / 2)) :=
by
  sorry

end coefficient_x_5_2_in_expansion_of_2x_minus_sqrt_inv_x_l179_179126


namespace smallest_b_value_is_6_l179_179325

noncomputable def smallest_b_value (a b c : ℝ) (h_arith : a + c = 2 * b) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_prod : a * b * c = 216) : ℝ :=
b

theorem smallest_b_value_is_6 (a b c : ℝ) (h_arith : a + c = 2 * b) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_prod : a * b * c = 216) : 
  smallest_b_value a b c h_arith h_pos h_prod = 6 :=
sorry

end smallest_b_value_is_6_l179_179325


namespace tangent_slope_at_pi_over_four_l179_179182

theorem tangent_slope_at_pi_over_four :
  deriv (fun x => Real.tan x) (Real.pi / 4) = 2 :=
sorry

end tangent_slope_at_pi_over_four_l179_179182


namespace calculate_f_value_l179_179403

def f (x y : ℚ) : ℚ := x - y * ⌈x / y⌉

theorem calculate_f_value :
  f (1/3) (-3/7) = -2/21 := by
  sorry

end calculate_f_value_l179_179403


namespace negation_of_proposition_l179_179892

theorem negation_of_proposition (x : ℝ) (h : 2 * x + 1 ≤ 0) : ¬ (2 * x + 1 ≤ 0) ↔ 2 * x + 1 > 0 := 
by
  sorry

end negation_of_proposition_l179_179892


namespace tan_of_cos_neg_five_thirteenth_l179_179755

variable {α : Real}

theorem tan_of_cos_neg_five_thirteenth (hcos : Real.cos α = -5/13) (hα : π < α ∧ α < 3 * π / 2) : 
  Real.tan α = 12 / 5 := 
sorry

end tan_of_cos_neg_five_thirteenth_l179_179755


namespace find_m_minus_n_l179_179904

theorem find_m_minus_n (m n : ℝ) (h1 : -5 + 1 = m) (h2 : -5 * 1 = n) : m - n = 1 :=
sorry

end find_m_minus_n_l179_179904


namespace simplify_expression_l179_179627

variable (x : ℝ)

theorem simplify_expression :
  ((3 * x - 2) * (5 * x ^ 12 + 3 * x ^ 11 + 5 * x ^ 10 + 3 * x ^ 9)) =
  (15 * x ^ 13 - x ^ 12 + 9 * x ^ 11 - x ^ 10 - 6 * x ^ 9) :=
by
  sorry

end simplify_expression_l179_179627


namespace largest_base5_three_digit_in_base10_l179_179217

theorem largest_base5_three_digit_in_base10 :
  let a := 4
      b := 4
      c := 4
      largest_base5 := 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  in largest_base5 = 124 :=
by
  let a := 4
  let b := 4
  let c := 4
  let largest_base5 := 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  show largest_base5 = 124
  sorry

end largest_base5_three_digit_in_base10_l179_179217


namespace polynomial_division_l179_179567

open Polynomial

theorem polynomial_division (a b : ℤ) (h : a^2 ≥ 4*b) :
  ∀ n : ℕ, ∃ (k l : ℤ), (x^2 + (C a) * x + (C b)) ∣ (x^2) * (x^2) ^ n + (C a) * x ^ n + (C b) ↔ 
    ((a = -2 ∧ b = 1) ∨ (a = 2 ∧ b = 1) ∨ (a = 0 ∧ b = -1)) :=
sorry

end polynomial_division_l179_179567


namespace line_equation_exists_l179_179757

noncomputable def P : ℝ × ℝ := (-2, 5)
noncomputable def m : ℝ := -3 / 4

theorem line_equation_exists (x y : ℝ) : 
  (y - 5 = -3 / 4 * (x + 2)) ↔ (3 * x + 4 * y - 14 = 0) := 
by 
  sorry

end line_equation_exists_l179_179757


namespace common_difference_of_arithmetic_sequence_l179_179737

theorem common_difference_of_arithmetic_sequence (a_n : ℕ → ℤ) (S_n : ℕ → ℤ) (n d a_2 S_3 a_4 : ℤ) 
  (h1 : a_2 + S_3 = -4) (h2 : a_4 = 3)
  (h3 : ∀ n, S_n = n * (a_n + (a_n + (n - 1) * d)) / 2)
  : d = 2 := by
  sorry

end common_difference_of_arithmetic_sequence_l179_179737


namespace lulu_final_cash_l179_179935

-- Definitions of the problem conditions
def initial_amount : ℕ := 65
def spent_on_ice_cream : ℕ := 5
def spent_on_tshirt (remaining : ℕ) : ℕ := remaining / 2
def deposit_in_bank (remaining : ℕ) : ℕ := remaining / 5

-- The proof problem statement
theorem lulu_final_cash :
  ∃ final_cash : ℕ,
    final_cash = initial_amount - spent_on_ice_cream - spent_on_tshirt (initial_amount - spent_on_ice_cream) - 
                      deposit_in_bank (spent_on_tshirt (initial_amount - spent_on_ice_cream)) ∧
    final_cash = 24 :=
by {
  sorry
}

end lulu_final_cash_l179_179935


namespace present_age_of_son_l179_179979

theorem present_age_of_son
  (S M : ℕ)
  (h1 : M = S + 24)
  (h2 : M + 2 = 2 * (S + 2)) :
  S = 22 :=
by {
  sorry
}

end present_age_of_son_l179_179979


namespace largest_base5_three_digits_is_124_l179_179208

noncomputable def largest_base5_three_digits_to_base10 : ℕ :=
  4 * 5^2 + 4 * 5^1 + 4 * 5^0

theorem largest_base5_three_digits_is_124 :
  largest_base5_three_digits_to_base10 = 124 :=
by
  -- calculating 4 * 5^2 + 4 * 5^1 + 4 * 5^0 = 124
  sorry

end largest_base5_three_digits_is_124_l179_179208


namespace bug_at_vertex_A_after_8_meters_l179_179330

theorem bug_at_vertex_A_after_8_meters (P : ℕ → ℚ) (h₀ : P 0 = 1)
(h : ∀ n, P (n + 1) = 1/3 * (1 - P n)) : 
P 8 = 1823 / 6561 := 
sorry

end bug_at_vertex_A_after_8_meters_l179_179330


namespace range_of_a_l179_179360

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + a * x - 4 < 0 ) ↔ (-16 < a ∧ a ≤ 0) :=
by
  sorry

end range_of_a_l179_179360


namespace max_volume_cube_max_volume_parallelepiped_l179_179457

variables {a b c : ℝ}

-- Problem (a): Cube with the maximum volume entirely contained in the tetrahedron
theorem max_volume_cube (h : a > 0 ∧ b > 0 ∧ c > 0) :
  ∃ s : ℝ, s = (a * b * c) / (a * b + b * c + a * c) := sorry

-- Problem (b): Rectangular parallelepiped with the maximum volume entirely contained in the tetrahedron
theorem max_volume_parallelepiped (h : a > 0 ∧ b > 0 ∧ c > 0) :
  ∃ (x y z : ℝ),
  (x = a / 3 ∧ y = b / 3 ∧ z = c / 3) ∧
  (x * y * z = (a * b * c) / 27) := sorry

end max_volume_cube_max_volume_parallelepiped_l179_179457


namespace smart_charging_piles_growth_l179_179782

-- Define the conditions
variables {x : ℝ}

-- First month charging piles
def first_month_piles : ℝ := 301

-- Third month charging piles
def third_month_piles : ℝ := 500

-- The theorem stating the relationship between the first and third month
theorem smart_charging_piles_growth : 
  first_month_piles * (1 + x) ^ 2 = third_month_piles :=
by
  sorry

end smart_charging_piles_growth_l179_179782


namespace james_final_sticker_count_l179_179784

-- Define the conditions
def initial_stickers := 478
def gift_stickers := 182
def given_away_stickers := 276

-- Define the correct answer
def final_stickers := 384

-- State the theorem
theorem james_final_sticker_count :
  initial_stickers + gift_stickers - given_away_stickers = final_stickers :=
by
  sorry

end james_final_sticker_count_l179_179784


namespace rhombus_diagonal_length_l179_179540

theorem rhombus_diagonal_length 
  (side_length : ℕ) (shorter_diagonal : ℕ) (longer_diagonal : ℕ)
  (h1 : side_length = 34) (h2 : shorter_diagonal = 32) :
  longer_diagonal = 60 :=
sorry

end rhombus_diagonal_length_l179_179540


namespace nephews_count_l179_179543

theorem nephews_count (a_nephews_20_years_ago : ℕ) (third_now_nephews : ℕ) (additional_nephews : ℕ) :
  a_nephews_20_years_ago = 80 →
  third_now_nephews = 3 →
  additional_nephews = 120 →
  ∃ (a_nephews_now : ℕ) (v_nephews_now : ℕ), a_nephews_now = third_now_nephews * a_nephews_20_years_ago ∧ v_nephews_now = a_nephews_now + additional_nephews ∧ (a_nephews_now + v_nephews_now = 600) :=
by
  sorry

end nephews_count_l179_179543


namespace n_divisibility_and_factors_l179_179177

open Nat

theorem n_divisibility_and_factors (n : ℕ) (h1 : 1990 ∣ n) (h2 : ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n):
  n = 4 * 5 * 199 ∨ n = 2 * 25 * 199 ∨ n = 2 * 5 * 39601 := 
sorry

end n_divisibility_and_factors_l179_179177


namespace can_capacity_l179_179598

-- Definition for the capacity of the can
theorem can_capacity 
  (milk_ratio water_ratio : ℕ) 
  (add_milk : ℕ) 
  (final_milk_ratio final_water_ratio : ℕ) 
  (capacity : ℕ) 
  (initial_milk initial_water : ℕ) 
  (h_initial_ratio : milk_ratio = 4 ∧ water_ratio = 3) 
  (h_additional_milk : add_milk = 8) 
  (h_final_ratio : final_milk_ratio = 2 ∧ final_water_ratio = 1) 
  (h_initial_amounts : initial_milk = 4 * (capacity - add_milk) / 7 ∧ initial_water = 3 * (capacity - add_milk) / 7) 
  (h_full_capacity : (initial_milk + add_milk) / initial_water = 2) 
  : capacity = 36 :=
sorry

end can_capacity_l179_179598


namespace great_wall_scientific_notation_l179_179032

theorem great_wall_scientific_notation : 
  (21200000 : ℝ) = 2.12 * 10^7 :=
by
  sorry

end great_wall_scientific_notation_l179_179032


namespace third_class_males_eq_nineteen_l179_179510

def first_class_males : ℕ := 17
def first_class_females : ℕ := 13
def second_class_males : ℕ := 14
def second_class_females : ℕ := 18
def third_class_females : ℕ := 17
def students_unable_to_partner : ℕ := 2
def total_males_from_first_two_classes : ℕ := first_class_males + second_class_males
def total_females_from_first_two_classes : ℕ := first_class_females + second_class_females
def total_females : ℕ := total_females_from_first_two_classes + third_class_females

theorem third_class_males_eq_nineteen (M : ℕ) : 
  total_males_from_first_two_classes + M - (total_females + students_unable_to_partner) = 0 → M = 19 :=
by
  sorry

end third_class_males_eq_nineteen_l179_179510


namespace students_in_same_month_l179_179776

theorem students_in_same_month (students : ℕ) (months : ℕ) 
  (h : students = 50) (h_months : months = 12) : 
  ∃ k ≥ 5, ∃ i, i < months ∧ ∃ f : ℕ → ℕ, (∀ j < students, f j < months) 
  ∧ ∃ n ≥ 5, ∃ j < students, f j = i :=
by 
  sorry

end students_in_same_month_l179_179776


namespace num_four_digit_numbers_l179_179122

/- Definitions established from conditions. -/
def digits : List Nat := [2, 3, 3, 5]
def countPermutations (l : List Nat) : Nat :=
  (l.length.factorial / l.foldr (λ d acc, acc * (l.count d).factorial) 1)

/- The target theorem statement. -/
theorem num_four_digit_numbers : countPermutations digits = 12 := by
  sorry

end num_four_digit_numbers_l179_179122


namespace cost_comparison_compare_cost_when_x_30_more_cost_effective_30_l179_179688

-- Definitions for the conditions in the problem
def cost_of_suit : ℕ := 1000
def cost_of_tie : ℕ := 200

-- Definitions for Option 1 and Option 2 calculations
def option1_total_cost (x : ℕ) (h : x > 20) : ℕ := 200 * x + 16000
def option2_total_cost (x : ℕ) (h : x > 20) : ℕ := 180 * x + 18000

-- Case x=30 for comparison
def x : ℕ := 30
def option1_cost_when_x_30 : ℕ := 200 * x + 16000
def option2_cost_when_x_30 : ℕ := 180 * x + 18000

-- More cost-effective plan when x=30
def more_cost_effective_plan_for_x_30 : ℕ := 21800

theorem cost_comparison (x : ℕ) (h1 : x > 20) :
  option1_total_cost x h1 = 200 * x + 16000 ∧
  option2_total_cost x h1 = 180 * x + 18000 := 
by
  sorry

theorem compare_cost_when_x_30 :
  option1_cost_when_x_30 = 22000 ∧
  option2_cost_when_x_30 = 23400 ∧
  option1_cost_when_x_30 < option2_cost_when_x_30 := 
by
  sorry

theorem more_cost_effective_30 :
  more_cost_effective_plan_for_x_30 = 21800 := 
by
  sorry

end cost_comparison_compare_cost_when_x_30_more_cost_effective_30_l179_179688


namespace inequality_with_integrals_l179_179136

variable {f : ℝ → ℝ} {A B : ℝ}

theorem inequality_with_integrals
  (h_continuous : ContinuousOn f (Set.Icc 0 1))
  (h_bounds : ∀ x ∈ Set.Icc 0 1, 0 < A ∧ A ≤ f x ∧ f x ≤ B) :
  A * B * ∫ x in 0..1, (1 / f x) ≤ A + B - ∫ x in 0..1, f x :=
by
  sorry

end inequality_with_integrals_l179_179136


namespace Mary_chestnuts_l179_179838

noncomputable def MaryPickedTwicePeter (P M : ℕ) := M = 2 * P
noncomputable def LucyPickedMorePeter (P L : ℕ) := L = P + 2
noncomputable def TotalPicked (P M L : ℕ) := P + M + L = 26

theorem Mary_chestnuts (P M L : ℕ) (h1 : MaryPickedTwicePeter P M) (h2 : LucyPickedMorePeter P L) (h3 : TotalPicked P M L) :
  M = 12 :=
sorry

end Mary_chestnuts_l179_179838


namespace pablo_days_to_complete_puzzles_l179_179015

-- Define the given conditions 
def puzzle_pieces_300 := 300
def puzzle_pieces_500 := 500
def puzzles_300 := 8
def puzzles_500 := 5
def rate_per_hour := 100
def max_hours_per_day := 7

-- Calculate total number of pieces
def total_pieces_300 := puzzles_300 * puzzle_pieces_300
def total_pieces_500 := puzzles_500 * puzzle_pieces_500
def total_pieces := total_pieces_300 + total_pieces_500

-- Calculate the number of pieces Pablo can put together per day
def pieces_per_day := max_hours_per_day * rate_per_hour

-- Calculate the number of days required for Pablo to complete all puzzles
def days_to_complete := total_pieces / pieces_per_day

-- Proposition to prove
theorem pablo_days_to_complete_puzzles : days_to_complete = 7 := sorry

end pablo_days_to_complete_puzzles_l179_179015


namespace integral_value_l179_179396

noncomputable def integrate_using_trapezoidal_rule_with_5_parts : ℝ :=
  let f := λ (x : ℝ), 1 / Real.sqrt (x + 4)
  let a : ℝ := 0
  let b : ℝ := 5
  let n : ℝ := 5
  let Δx := (b - a) / n
  let x := λ (i : ℕ), a + i * Δx
  let y := λ i, f (x i)
  (Δx / 2) * (y 0 + 2 * (y 1 + y 2 + y 3 + y 4) + y 5)

theorem integral_value :
  integrate_using_trapezoidal_rule_with_5_parts = 2.0035 :=
sorry

end integral_value_l179_179396


namespace initial_number_2008_l179_179151

theorem initial_number_2008 (x : ℕ) (h : x = 2008 ∨ (∃ y: ℕ, (x = 2*y + 1 ∨ (x = y / (y + 2))))): x = 2008 :=
by
  cases h with
  | inl h2008 => exact h2008
  | inr hexists => cases hexists with
    | intro y hy =>
        cases hy
        case inl h2y => sorry
        case inr hdiv => sorry

end initial_number_2008_l179_179151


namespace quadrilateral_front_view_iff_cylinder_or_prism_l179_179238

inductive Solid
| cone : Solid
| cylinder : Solid
| triangular_pyramid : Solid
| quadrangular_prism : Solid

def has_quadrilateral_front_view (s : Solid) : Prop :=
  s = Solid.cylinder ∨ s = Solid.quadrangular_prism

theorem quadrilateral_front_view_iff_cylinder_or_prism (s : Solid) :
  has_quadrilateral_front_view s ↔ s = Solid.cylinder ∨ s = Solid.quadrangular_prism :=
by
  sorry

end quadrilateral_front_view_iff_cylinder_or_prism_l179_179238


namespace find_n_in_arithmetic_sequence_l179_179142

noncomputable def arithmetic_sequence (a1 d n : ℕ) := a1 + (n - 1) * d

theorem find_n_in_arithmetic_sequence (a1 d an : ℕ) (h1 : a1 = 1) (h2 : d = 5) (h3 : an = 2016) :
  ∃ n : ℕ, an = arithmetic_sequence a1 d n :=
  by
  sorry

end find_n_in_arithmetic_sequence_l179_179142


namespace next_hexagon_dots_l179_179274

theorem next_hexagon_dots (base_dots : ℕ) (increment : ℕ) : base_dots = 2 → increment = 2 → 
  (2 + 6*2) + 6*(2*2) + 6*(3*2) + 6*(4*2) = 122 := 
by
  intros hbd hi
  sorry

end next_hexagon_dots_l179_179274


namespace sin_alpha_neg_point_two_l179_179300

theorem sin_alpha_neg_point_two (a : ℝ) (h : Real.sin (Real.pi + a) = 0.2) : Real.sin a = -0.2 := 
by
  sorry

end sin_alpha_neg_point_two_l179_179300


namespace inner_tetrahedron_volume_l179_179990

def volume_of_inner_tetrahedron(cube_side : ℕ) : ℚ :=
  let base_area := (cube_side * cube_side) / 2
  let height := cube_side
  let original_tetra_volume := (1 / 3) * base_area * height
  let inner_tetra_volume := original_tetra_volume / 8
  inner_tetra_volume

theorem inner_tetrahedron_volume {cube_side : ℕ} (h : cube_side = 2) : 
  volume_of_inner_tetrahedron cube_side = 1 / 6 := 
by
  rw [h]
  unfold volume_of_inner_tetrahedron 
  norm_num
  sorry

end inner_tetrahedron_volume_l179_179990


namespace conference_center_distance_l179_179342

variables (d t: ℝ)

theorem conference_center_distance
  (h1: ∃ t: ℝ, d = 45 * (t + 1.5))
  (h2: ∃ t: ℝ, d - 45 = 55 * (t - 1.25)):
  d = 478.125 :=
by
  sorry

end conference_center_distance_l179_179342


namespace positive_polynomial_l179_179623

theorem positive_polynomial (x : ℝ) : 3 * x ^ 2 - 6 * x + 3.5 > 0 := 
by sorry

end positive_polynomial_l179_179623


namespace expression_simplification_l179_179867

theorem expression_simplification : (4^2 * 7 / (8 * 9^2) * (8 * 9 * 11^2) / (4 * 7 * 11)) = 44 / 9 :=
by
  sorry

end expression_simplification_l179_179867


namespace students_average_comparison_l179_179664

theorem students_average_comparison (t1 t2 t3 : ℝ) (h : t1 < t2) (h' : t2 < t3) :
  (∃ t1 t2 t3 : ℝ, t1 < t2 ∧ t2 < t3 ∧ (t1 + t2 + t3) / 3 = (t1 + t3 + 2 * t2) / 4) ∨
  (∀ t1 t2 t3 : ℝ, t1 < t2 ∧ t2 < t3 → 
     (t1 + t3 + 2 * t2) / 4 > (t1 + t2 + t3) / 3) :=
sorry

end students_average_comparison_l179_179664


namespace major_axis_length_l179_179852

-- Define the radius of the cylinder
def cylinder_radius : ℝ := 2

-- Define the relationship given in the problem
def major_axis_ratio : ℝ := 1.6

-- Define the calculation for minor axis
def minor_axis : ℝ := 2 * cylinder_radius

-- Define the calculation for major axis
def major_axis : ℝ := major_axis_ratio * minor_axis

-- The theorem statement
theorem major_axis_length:
  major_axis = 6.4 :=
by 
  sorry -- Proof to be provided later

end major_axis_length_l179_179852


namespace solve_r_l179_179124

theorem solve_r (k r : ℝ) (h1 : 3 = k * 2^r) (h2 : 15 = k * 4^r) : 
  r = Real.log 5 / Real.log 2 := 
sorry

end solve_r_l179_179124


namespace new_perimeter_of_rectangle_l179_179961

theorem new_perimeter_of_rectangle (w : ℝ) (A : ℝ) (new_area_factor : ℝ) (L : ℝ) (L' : ℝ) (P' : ℝ) 
  (h_w : w = 10) (h_A : A = 150) (h_new_area_factor: new_area_factor = 4 / 3)
  (h_orig_length : L = A / w) (h_new_area: A' = new_area_factor * A) (h_A' : A' = 200)
  (h_new_length : L' = A' / w) (h_perimeter : P' = 2 * (L' + w)) 
  : P' = 60 :=
sorry

end new_perimeter_of_rectangle_l179_179961


namespace intersection_of_sets_l179_179764

/-- Given the definitions of sets A and B, prove that A ∩ B equals {1, 2}. -/
theorem intersection_of_sets :
  let A := {x : ℝ | 0 < x}
  let B := {-2, -1, 1, 2}
  A ∩ B = {1, 2} :=
sorry

end intersection_of_sets_l179_179764


namespace forgotten_code_possibilities_l179_179766

theorem forgotten_code_possibilities:
  let digits_set := {d | ∀ n:ℕ, 0≤n ∧ n≤9 → n≠0 → 
                     (n + 4 + 4 + last_digit ≡ 0 [MOD 3]) ∨ 
                     (n + 7 + 7 + last_digit ≡ 0 [MOD 3]) ∨
                     (n + 4 + 7 + last_digit ≡ 0 [MOD 3]) ∨
                     (n + 7 + 4 + last_digit ≡ 0 [MOD 3])
                    }
  let valid_first_digits := {1, 2, 4, 5, 7, 8}
  let total_combinations := 4 * 3 + 4 * 3 -- middle combinations * valid first digit combinations
  total_combinations = 24 ∧ digits_set = valid_first_digits := by
  sorry

end forgotten_code_possibilities_l179_179766


namespace calories_burned_each_player_l179_179353

theorem calories_burned_each_player :
  ∀ (num_round_trips stairs_per_trip calories_per_stair : ℕ),
  num_round_trips = 40 →
  stairs_per_trip = 32 →
  calories_per_stair = 2 →
  (num_round_trips * (2 * stairs_per_trip) * calories_per_stair) = 5120 :=
by
  intros num_round_trips stairs_per_trip calories_per_stair h_num_round_trips h_stairs_per_trip h_calories_per_stair
  rw [h_num_round_trips, h_stairs_per_trip, h_calories_per_stair]
  simp
  rfl

#eval calories_burned_each_player 40 32 2 rfl rfl rfl

end calories_burned_each_player_l179_179353


namespace sasha_prediction_min_n_l179_179704

theorem sasha_prediction_min_n :
  let whiteRook_players : ℕ := 15 in
  let blackElephant_players : ℕ := 20 in
  let total_games : ℕ := whiteRook_players * blackElephant_players in
  ∀ (n : ℕ), n >= 280 → ∃ p ∈ fin (total_games - n), 
    (n < 280 -> ∀ i, ¬one_game_played_at_a_time whiteRook_players blackElephant_players total_games n i) :=
by
  intros whiteRook_players blackElephant_players total_games n hn,
  rw [total_games, mul_comm] at hn,
  sorry

end sasha_prediction_min_n_l179_179704


namespace row_speed_with_stream_l179_179995

theorem row_speed_with_stream (v : ℝ) (s : ℝ) (h1 : s = 2) (h2 : v - s = 12) : v + s = 16 := by
  -- Placeholder for the proof
  sorry

end row_speed_with_stream_l179_179995


namespace carols_weight_l179_179184

variables (a c : ℝ)

theorem carols_weight (h1 : a + c = 220) (h2 : c - a = c / 3 + 10) : c = 138 :=
by
  sorry

end carols_weight_l179_179184


namespace total_minutes_ironing_over_4_weeks_l179_179546

/-- Define the time spent ironing each day -/
def minutes_ironing_per_day : Nat := 5 + 3

/-- Define the number of days Hayden irons per week -/
def days_ironing_per_week : Nat := 5

/-- Define the number of weeks considered -/
def number_of_weeks : Nat := 4

/-- The main theorem we're proving is that Hayden spends 160 minutes ironing over 4 weeks -/
theorem total_minutes_ironing_over_4_weeks :
  (minutes_ironing_per_day * days_ironing_per_week * number_of_weeks) = 160 := by
  sorry

end total_minutes_ironing_over_4_weeks_l179_179546


namespace arithmetic_sequence_geometric_condition_l179_179259

theorem arithmetic_sequence_geometric_condition 
  (a : ℕ → ℝ) (d : ℝ) (h_nonzero : d ≠ 0) 
  (h_a3 : a 3 = 7)
  (h_geo_seq : (a 2 - 1)^2 = (a 1 - 1) * (a 4 - 1)) : 
  a 10 = 21 :=
sorry

end arithmetic_sequence_geometric_condition_l179_179259


namespace solution_set_of_inequality_system_l179_179183

theorem solution_set_of_inequality_system :
  (6 - 2 * x ≥ 0) ∧ (2 * x + 4 > 0) ↔ (-2 < x ∧ x ≤ 3) := 
sorry

end solution_set_of_inequality_system_l179_179183


namespace percentage_born_in_july_l179_179037

def total_scientists : ℕ := 150
def scientists_born_in_july : ℕ := 15

theorem percentage_born_in_july : (scientists_born_in_july * 100 / total_scientists) = 10 := by
  sorry

end percentage_born_in_july_l179_179037


namespace adrianna_gum_pieces_l179_179855

-- Definitions based on conditions
def initial_gum_pieces : ℕ := 10
def additional_gum_pieces : ℕ := 3
def friends_count : ℕ := 11

-- Expression to calculate the final pieces of gum
def total_gum_pieces : ℕ := initial_gum_pieces + additional_gum_pieces
def gum_left : ℕ := total_gum_pieces - friends_count

-- Lean statement we want to prove
theorem adrianna_gum_pieces: gum_left = 2 := 
by 
  sorry

end adrianna_gum_pieces_l179_179855


namespace Alfred_spent_on_repairs_l179_179860

noncomputable def AlfredRepairCost (purchase_price selling_price gain_percent : ℚ) : ℚ :=
  let R := (selling_price - purchase_price * (1 + gain_percent)) / (1 + gain_percent)
  R

theorem Alfred_spent_on_repairs :
  AlfredRepairCost 4700 5800 0.017543859649122806 = 1000 := by
  sorry

end Alfred_spent_on_repairs_l179_179860


namespace find_line_equation_l179_179091

noncomputable def perpendicular_origin_foot := 
  ∃ l : ℝ → ℝ → Prop,
    (∀ (x y : ℝ), l x y ↔ y = 2 * x + 5) ∧
    l (-2) 1

theorem find_line_equation : 
  ∃ l : ℝ → ℝ → Prop,
    (∀ (x y : ℝ), l x y ↔ 2 * x - y + 5 = 0) ∧
    l (-2) 1 ∧
    ∀ p q : ℝ, p = 0 → q = 0 → ¬ (l p q)
:= sorry

end find_line_equation_l179_179091


namespace computation_l179_179086

theorem computation :
  ( ( (4^3 - 1) / (4^3 + 1) ) * ( (5^3 - 1) / (5^3 + 1) ) * ( (6^3 - 1) / (6^3 + 1) ) * 
    ( (7^3 - 1) / (7^3 + 1) ) * ( (8^3 - 1) / (8^3 + 1) ) 
  ) = (73 / 312) :=
by
  sorry

end computation_l179_179086


namespace sufficient_condition_for_m_l179_179885

variable (x m : ℝ)

def p (x : ℝ) : Prop := abs (x - 4) ≤ 6
def q (x m : ℝ) : Prop := x ≤ 1 + m

theorem sufficient_condition_for_m (h : ∀ x, p x → q x m ∧ ∃ x, ¬p x ∧ q x m) : m ≥ 9 :=
sorry

end sufficient_condition_for_m_l179_179885


namespace notebook_problem_l179_179663

/-- Conditions:
1. If each notebook costs 3 yuan, 6 more notebooks can be bought.
2. If each notebook costs 5 yuan, there is a 30-yuan shortfall.

We need to show:
1. The total number of notebooks \( x \).
2. The number of 3-yuan notebooks \( n_3 \). -/
theorem notebook_problem (x y n3 : ℕ) (h1 : y = 3 * x + 18) (h2 : y = 5 * x - 30) (h3 : 3 * n3 + 5 * (x - n3) = y) :
  x = 24 ∧ n3 = 15 :=
by
  -- proof to be provided
  sorry

end notebook_problem_l179_179663


namespace range_of_a_l179_179368

open Real

noncomputable def doesNotPassThroughSecondQuadrant (a : ℝ) : Prop :=
  ∀ x y : ℝ, (3 * a - 1) * x + (2 - a) * y - 1 ≠ 0

theorem range_of_a : {a : ℝ | doesNotPassThroughSecondQuadrant a} = {a : ℝ | 2 ≤ a } :=
by
  ext
  sorry

end range_of_a_l179_179368


namespace lulu_cash_left_l179_179928

-- Define the initial amount
def initial_amount : ℕ := 65

-- Define the amount spent on ice cream
def spent_on_ice_cream : ℕ := 5

-- Define the amount spent on a t-shirt
def spent_on_tshirt (remaining_after_ice_cream : ℕ) : ℕ := remaining_after_ice_cream / 2

-- Define the amount deposited in the bank
def deposited_in_bank (remaining_after_tshirt : ℕ) : ℕ := remaining_after_tshirt / 5

-- Define the remaining cash after all transactions
def remaining_cash (initial : ℕ) (spent_ice_cream : ℕ) (spent_tshirt: ℕ) (deposited: ℕ) :ℕ :=
  initial - spent_ice_cream - spent_tshirt - deposited

-- Theorem statement to prove
theorem lulu_cash_left : remaining_cash initial_amount spent_on_ice_cream (spent_on_tshirt (initial_amount - spent_on_ice_cream)) 
(deposited_in_bank ((initial_amount - spent_on_ice_cream) - (spent_on_tshirt (initial_amount - spent_on_ice_cream)))) = 24 :=
by
  sorry

end lulu_cash_left_l179_179928


namespace sequence_general_formula_l179_179572

theorem sequence_general_formula (a : ℕ → ℤ) (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, n > 0 → a (n + 1) > a n)
  (h3 : ∀ n : ℕ, n > 0 → (a (n + 1))^2 - 2 * a n * a (n + 1) + (a n)^2 = 1) :
  ∀ n : ℕ, n > 0 → a n = n :=
by 
  sorry

end sequence_general_formula_l179_179572


namespace total_number_of_workers_l179_179173

theorem total_number_of_workers (W : ℕ) (R : ℕ) 
  (h1 : (7 + R) * 8000 = 7 * 18000 + R * 6000) 
  (h2 : W = 7 + R) : W = 42 :=
by
  -- Proof steps will go here
  sorry

end total_number_of_workers_l179_179173


namespace quadratic_equal_roots_l179_179304

theorem quadratic_equal_roots (k : ℝ) : (∃ r : ℝ, (r^2 - 2 * r + k = 0)) → k = 1 := 
by
  sorry

end quadratic_equal_roots_l179_179304


namespace keanu_total_spending_l179_179612

-- Definitions based on conditions
def dog_fish : Nat := 40
def cat_fish : Nat := dog_fish / 2
def total_fish : Nat := dog_fish + cat_fish
def cost_per_fish : Nat := 4
def total_cost : Nat := total_fish * cost_per_fish

-- Theorem statement
theorem keanu_total_spending : total_cost = 240 :=
by 
    sorry

end keanu_total_spending_l179_179612


namespace total_money_spent_l179_179610

-- Assume Keanu gave dog 40 fish
def dog_fish := 40

-- Assume Keanu gave cat half as many fish as he gave to his dog
def cat_fish := dog_fish / 2

-- Assume each fish cost $4
def cost_per_fish := 4

-- Prove that total amount of money spent is $240
theorem total_money_spent : (dog_fish + cat_fish) * cost_per_fish = 240 := 
by
  sorry

end total_money_spent_l179_179610


namespace number_of_boys_l179_179362

-- Define the conditions
def total_attendees : Nat := 100
def faculty_percentage : Rat := 0.1
def faculty_count : Nat := total_attendees * faculty_percentage
def student_count : Nat := total_attendees - faculty_count
def girls_fraction : Rat := 2 / 3
def girls_count : Nat := student_count * girls_fraction

-- Define the question in terms of a Lean theorem
theorem number_of_boys :
  total_attendees = 100 →
  faculty_percentage = 0.1 →
  faculty_count = 10 →
  student_count = 90 →
  girls_fraction = 2 / 3 →
  girls_count = 60 →
  student_count - girls_count = 30 :=
by
  intros
  sorry -- Skip the proof

end number_of_boys_l179_179362


namespace intersection_points_rectangular_coords_l179_179456

theorem intersection_points_rectangular_coords :
  ∃ (x y : ℝ),
    (∃ (ρ θ : ℝ), ρ = 2 * Real.cos θ ∧ ρ^2 * (Real.cos θ)^2 - 4 * ρ^2 * (Real.sin θ)^2 = 4 ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) ∧
    (x = (1 + Real.sqrt 13) / 3 ∧ y = 0) := 
sorry

end intersection_points_rectangular_coords_l179_179456


namespace factorization_of_a_cubed_minus_a_l179_179561

variable (a : ℝ)

theorem factorization_of_a_cubed_minus_a : a^3 - a = a * (a + 1) * (a - 1) := by
  sorry

end factorization_of_a_cubed_minus_a_l179_179561


namespace Jen_distance_from_start_l179_179478

-- Define the rate of Jen's walking (in miles per hour)
def walking_rate : ℝ := 4

-- Define the time Jen walks forward (in hours)
def forward_time : ℝ := 2

-- Define the time Jen walks back (in hours)
def back_time : ℝ := 1

-- Define the distance walked forward
def distance_forward : ℝ := walking_rate * forward_time

-- Define the distance walked back
def distance_back : ℝ := walking_rate * back_time

-- Define the net distance from the starting point
def net_distance : ℝ := distance_forward - distance_back

-- Theorem stating the net distance from the starting point is 4.0 miles
theorem Jen_distance_from_start : net_distance = 4.0 := by
  sorry

end Jen_distance_from_start_l179_179478


namespace sum_of_ages_3_years_ago_l179_179069

noncomputable def siblings_age_3_years_ago (R D S J : ℕ) : Prop :=
  R = D + 6 ∧
  D = S + 8 ∧
  J = R - 5 ∧
  R + 8 = 2 * (S + 8) ∧
  J + 10 = (D + 10) / 2 + 4 ∧
  S + 24 + J = 60 →
  (R - 3) + (D - 3) + (S - 3) + (J - 3) = 43

theorem sum_of_ages_3_years_ago (R D S J : ℕ) :
  siblings_age_3_years_ago R D S J :=
by
  intros
  sorry

end sum_of_ages_3_years_ago_l179_179069


namespace base_conversion_equivalence_l179_179060

theorem base_conversion_equivalence :
  ∃ (n : ℕ), (∃ (C B : ℕ), C < 9 ∧ B < 6 ∧ n = 9 * C + B) ∧
             (∃ (C B : ℕ), C < 9 ∧ B < 6 ∧ n = 6 * B + C) ∧
             n = 0 := 
by 
  sorry

end base_conversion_equivalence_l179_179060


namespace fixed_cost_calculation_l179_179501

theorem fixed_cost_calculation (TC MC n FC : ℕ) (h1 : TC = 16000) (h2 : MC = 200) (h3 : n = 20) (h4 : TC = FC + MC * n) : FC = 12000 :=
by
  sorry

end fixed_cost_calculation_l179_179501


namespace find_length_of_side_c_l179_179596

variables {A B C : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C]

/-- Given that in triangle ABC, sin C = 1 / 2, a = 2 * sqrt 3, b = 2,
we want to prove the length of side c is either 2 or 2 * sqrt 7. -/
theorem find_length_of_side_c (C : Real) (a b c : Real) (h1 : Real.sin C = 1 / 2)
  (h2 : a = 2 * Real.sqrt 3) (h3 : b = 2) :
  c = 2 ∨ c = 2 * Real.sqrt 7 :=
by
  sorry

end find_length_of_side_c_l179_179596


namespace lab_tech_items_l179_179662

theorem lab_tech_items (num_uniforms : ℕ) (num_coats : ℕ) (num_techs : ℕ) (total_items : ℕ)
  (h_uniforms : num_uniforms = 12)
  (h_coats : num_coats = 6 * num_uniforms)
  (h_techs : num_techs = num_uniforms / 2)
  (h_total : total_items = num_coats + num_uniforms) :
  total_items / num_techs = 14 :=
by
  -- Placeholder for proof, ensuring theorem builds correctly.
  sorry

end lab_tech_items_l179_179662


namespace total_bouquets_sold_l179_179713

-- Define the conditions as variables
def monday_bouquets : ℕ := 12
def tuesday_bouquets : ℕ := 3 * monday_bouquets
def wednesday_bouquets : ℕ := tuesday_bouquets / 3

-- The statement to prove
theorem total_bouquets_sold : 
  monday_bouquets + tuesday_bouquets + wednesday_bouquets = 60 :=
by
  -- The proof is omitted using sorry
  sorry

end total_bouquets_sold_l179_179713


namespace part1_solution_l179_179617

def f (x m : ℝ) := |x + m| + |2 * x + 1|

theorem part1_solution (x : ℝ) : f x (-1) ≤ 3 → -1 ≤ x ∧ x ≤ 1 := 
sorry

end part1_solution_l179_179617


namespace general_formulas_max_b_seq_l179_179280

noncomputable def a_seq (n : ℕ) : ℕ := 4 * n - 2
noncomputable def b_seq (n : ℕ) : ℕ := 4 * n - 2 - 2^(n - 1)

-- The general formulas to be proved
theorem general_formulas :
  (∀ n : ℕ, a_seq n = 4 * n - 2) ∧ 
  (∀ n : ℕ, b_seq n = 4 * n - 2 - 2^(n - 1)) :=
by
  sorry

-- The maximum value condition to be proved
theorem max_b_seq :
  ((∀ n : ℕ, b_seq n ≤ b_seq 3) ∨ (∀ n : ℕ, b_seq n ≤ b_seq 4)) :=
by
  sorry

end general_formulas_max_b_seq_l179_179280


namespace celsius_to_fahrenheit_l179_179116

theorem celsius_to_fahrenheit (C F : ℤ) (h1 : C = 50) (h2 : C = 5 / 9 * (F - 32)) : F = 122 :=
by
  sorry

end celsius_to_fahrenheit_l179_179116


namespace clock_angle_7_35_l179_179673

theorem clock_angle_7_35 : 
  let minute_hand_angle := (35 / 60) * 360
  let hour_hand_angle := 7 * 30 + (35 / 60) * 30
  let angle_between := hour_hand_angle - minute_hand_angle
  angle_between = 17.5 := by
sorry

end clock_angle_7_35_l179_179673


namespace solve_quadratic_identity_l179_179506

theorem solve_quadratic_identity (y : ℝ) (h : 7 * y^2 + 2 = 5 * y + 13) :
  (14 * y - 5) ^ 2 = 333 :=
by sorry

end solve_quadratic_identity_l179_179506


namespace survival_probability_l179_179981

theorem survival_probability : 
  let v1 := (486 : ℚ) / 630
  let v2 := (540 : ℚ) / 675 in
  (v1 * v2 = 108 / 175) ∧
  (1 - v1 = 8 / 35) ∧
  (1 - v2 = 1 / 5) ∧
  ((1 - v1) * (1 - v2) = 8 / 175) ∧
  ((1 - v1) * v2 = 32 / 175) ∧
  (v1 * (1 - v2) = 27 / 175) := 
by {
  unfold v1 v2,
  split, { sorry },
  split, { sorry },
  split, { sorry },
  split, { sorry },
  split, { sorry },
  sorry
}

end survival_probability_l179_179981


namespace min_max_sum_eq_one_l179_179421

theorem min_max_sum_eq_one 
  (x : ℕ → ℝ)
  (h_nonneg : ∀ i, 0 ≤ x i)
  (h_sum_eq_one : (x 1 + x 2 + x 3 + x 4 + x 5) = 1) :
  (min (max (x 1 + x 2) (max (x 2 + x 3) (max (x 3 + x 4) (x 4 + x 5)))) = (1 / 3)) :=
by
  sorry

end min_max_sum_eq_one_l179_179421


namespace members_not_playing_any_sport_l179_179780

theorem members_not_playing_any_sport {total_members badminton_players tennis_players both_players : ℕ}
  (h_total : total_members = 28)
  (h_badminton : badminton_players = 17)
  (h_tennis : tennis_players = 19)
  (h_both : both_players = 10) :
  total_members - (badminton_players + tennis_players - both_players) = 2 :=
by
  sorry

end members_not_playing_any_sport_l179_179780


namespace perfect_cubes_count_l179_179405

theorem perfect_cubes_count (x y : ℕ) (h1: x = 2^9 + 1!) (h2: y = 2^{17} + 1!) : 
  ∃ c : ℕ, c = 42 ∧ 
  let lower_bound := (nat.cbrt x).succ in
  let upper_bound := (nat.cbrt y) in
  ∀ n, lower_bound ≤ n ∧ n ≤ upper_bound → n^3 ≥ x ∧ n^3 ≤ y :=
sorry

end perfect_cubes_count_l179_179405


namespace waiters_hired_correct_l179_179395

noncomputable def waiters_hired (W H : ℕ) : Prop :=
  let cooks := 9
  (cooks / W = 3 / 8) ∧ (cooks / (W + H) = 1 / 4) ∧ (H = 12)

theorem waiters_hired_correct (W H : ℕ) : waiters_hired W H :=
  sorry

end waiters_hired_correct_l179_179395


namespace sum_of_two_greatest_values_of_b_sum_of_two_greatest_values_l179_179523

theorem sum_of_two_greatest_values_of_b (b : Real) 
  (h : 4 * b ^ 4 - 41 * b ^ 2 + 100 = 0) :
  b = 2.5 ∨ b = 2 ∨ b = -2.5 ∨ b = -2 :=
sorry

theorem sum_of_two_greatest_values (b1 b2 : Real)
  (hb1 : 4 * b1 ^ 4 - 41 * b1 ^ 2 + 100 = 0)
  (hb2 : 4 * b2 ^ 4 - 41 * b2 ^ 2 + 100 = 0) :
  b1 = 2.5 → b2 = 2 → b1 + b2 = 4.5 :=
sorry

end sum_of_two_greatest_values_of_b_sum_of_two_greatest_values_l179_179523


namespace sum_of_numbers_l179_179062

theorem sum_of_numbers (x y : ℕ) (h1 : x * y = 9375) (h2 : y / x = 15) : x + y = 400 :=
by
  sorry

end sum_of_numbers_l179_179062


namespace max_cube_side_length_max_rect_parallelepiped_dimensions_l179_179469

-- Part (a)
theorem max_cube_side_length (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ s : ℝ, s = a * b * c / (a * b + b * c + a * c) :=
sorry

-- Part (b)
theorem max_rect_parallelepiped_dimensions (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ x y z : ℝ, x = a / 3 ∧ y = b / 3 ∧ z = c / 3 :=
sorry

end max_cube_side_length_max_rect_parallelepiped_dimensions_l179_179469


namespace price_difference_is_correct_l179_179508

noncomputable def total_cost : ℝ := 70.93
noncomputable def cost_of_pants : ℝ := 34.0
noncomputable def cost_of_belt : ℝ := total_cost - cost_of_pants
noncomputable def price_difference : ℝ := cost_of_belt - cost_of_pants

theorem price_difference_is_correct :
  price_difference = 2.93 := by
  sorry

end price_difference_is_correct_l179_179508


namespace problem_sol_l179_179098

open Real Trig

noncomputable def numberOfRealSolutions : ℝ := 95

theorem problem_sol : ∃ x : ℝ, -150 ≤ x ∧ x ≤ 150 ∧ (x / 150 = sin x) ∧ finset.card (finset.filter (λx, x / 150 = sin x) (finset.range 301 - 151)) = 95 :=
by
  sorry

end problem_sol_l179_179098


namespace circle_center_sum_l179_179406

theorem circle_center_sum {x y : ℝ} (h : x^2 + y^2 - 10*x + 4*y + 15 = 0) :
  (x, y) = (5, -2) ∧ x + y = 3 :=
by
  sorry

end circle_center_sum_l179_179406


namespace tangent_line_at_pi_over_4_l179_179432

noncomputable def f (x : ℝ) := Real.cos (2 * x)

theorem tangent_line_at_pi_over_4 :
  let x₀ := Real.pi / 4
  let y₀ := f x₀
  fderiv ℝ f x₀ = (-2 : ℝ) • (1 : ℝ) ∧ y₀ = 0 → 
  ∀ x, f x₀ + (fderiv ℝ f x₀) * (x - x₀) = -2 * (x - x₀) := sorry

end tangent_line_at_pi_over_4_l179_179432


namespace area_of_shaded_region_l179_179916

-- Definitions from conditions
def radius : ℝ := 6

def perpendicular_diameters (x y : circle) : Prop := 
  is_diameter x ∧ is_diameter y ∧ angle x.center y.center = π/2

-- The theorem we want to prove
theorem area_of_shaded_region (radius : ℝ)
    (h : perpendicular_diameters PQ RS) : 
    area_shaded_region (PQ : circle, RS : circle radius) = 36 + 18 * π :=
sorry

end area_of_shaded_region_l179_179916


namespace average_age_add_person_l179_179345

theorem average_age_add_person (n : ℕ) (h1 : (∀ T, T = n * 14 → (T + 34) / (n + 1) = 16)) : n = 9 :=
by
  sorry

end average_age_add_person_l179_179345


namespace average_distance_scientific_notation_l179_179346

theorem average_distance_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ |a| ∧ |a| < 10 ∧ a * 10 ^ n = 384000000 ∧ a = 3.84 ∧ n = 8 :=
sorry

end average_distance_scientific_notation_l179_179346


namespace composite_function_l179_179880

def f (x : ℝ) : ℝ := 2 * x - 1
def g (x : ℝ) : ℝ := x + 1

theorem composite_function : ∀ (x : ℝ), f (g x) = 2 * x + 1 :=
by
  intro x
  sorry

end composite_function_l179_179880


namespace max_cube_side_length_max_rect_parallelepiped_dimensions_l179_179470

-- Part (a)
theorem max_cube_side_length (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ s : ℝ, s = a * b * c / (a * b + b * c + a * c) :=
sorry

-- Part (b)
theorem max_rect_parallelepiped_dimensions (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ x y z : ℝ, x = a / 3 ∧ y = b / 3 ∧ z = c / 3 :=
sorry

end max_cube_side_length_max_rect_parallelepiped_dimensions_l179_179470


namespace remainder_of_m_l179_179566

theorem remainder_of_m (m : ℕ) (h₁ : m ^ 3 % 7 = 6) (h₂ : m ^ 4 % 7 = 4) : m % 7 = 3 := 
sorry

end remainder_of_m_l179_179566


namespace imaginary_part_of_z_l179_179952

-- Define the complex number z
def z : Complex := Complex.mk 3 (-4)

-- State the proof goal
theorem imaginary_part_of_z : z.im = -4 :=
by
  sorry

end imaginary_part_of_z_l179_179952


namespace total_bouquets_sold_l179_179715

-- defining the sale conditions
def monday_bouquets := 12
def tuesday_bouquets := 3 * monday_bouquets
def wednesday_bouquets := tuesday_bouquets / 3

-- defining the total sale
def total_bouquets := monday_bouquets + tuesday_bouquets + wednesday_bouquets

-- stating the theorem
theorem total_bouquets_sold : total_bouquets = 60 := by
  -- the proof would go here
  sorry

end total_bouquets_sold_l179_179715


namespace constant_term_expansion_l179_179644

-- Define the binomial coefficient
noncomputable def binomial_coeff (n k : ℕ) : ℕ := Nat.choose n k

-- Define the general term in the binomial expansion
noncomputable def general_term (r n : ℕ) (x : ℝ) : ℝ := 
  (2:ℝ)^r * binomial_coeff n r * x^((n-5*r)/2)

-- Given problem conditions
def n := 10
def largest_binomial_term_index := 5  -- Represents the sixth term (r = 5)

-- Statement to prove the constant term equals 180
theorem constant_term_expansion {x : ℝ} : 
  general_term 2 n 1 = 180 :=
by {
  sorry
}

end constant_term_expansion_l179_179644


namespace largest_base5_three_digit_to_base10_l179_179203

theorem largest_base5_three_digit_to_base10 :
  (4 * 5^2 + 4 * 5^1 + 4 * 5^0) = 124 :=
by
  sorry

end largest_base5_three_digit_to_base10_l179_179203


namespace simplify_sqrt_product_l179_179162

theorem simplify_sqrt_product (x : ℝ) : 
  (Real.sqrt (50 * x) * Real.sqrt (18 * x) * Real.sqrt (32 * x)) = 120 * x * Real.sqrt (2 * x) := 
by
  sorry

end simplify_sqrt_product_l179_179162


namespace contradiction_method_at_most_one_positive_l179_179969

theorem contradiction_method_at_most_one_positive :
  (∃ a b c : ℝ, (a > 0 → (b ≤ 0 ∧ c ≤ 0)) ∧ (b > 0 → (a ≤ 0 ∧ c ≤ 0)) ∧ (c > 0 → (a ≤ 0 ∧ b ≤ 0))) → 
  (¬(∃ a b c : ℝ, (a > 0 ∧ b > 0) ∨ (b > 0 ∧ c > 0) ∨ (a > 0 ∧ c > 0))) :=
by sorry

end contradiction_method_at_most_one_positive_l179_179969


namespace vertical_asymptote_at_x_4_l179_179301

def P (x : ℝ) : ℝ := x^2 + 2 * x + 8
def Q (x : ℝ) : ℝ := x^2 - 8 * x + 16

theorem vertical_asymptote_at_x_4 : ∃ x : ℝ, Q x = 0 ∧ P x ≠ 0 ∧ x = 4 :=
by
  use 4
  -- Proof skipped
  sorry

end vertical_asymptote_at_x_4_l179_179301


namespace chips_cost_l179_179827

noncomputable def cost_of_each_bag_of_chips (amount_paid_per_friend : ℕ) (number_of_friends : ℕ) (number_of_bags : ℕ) : ℕ :=
  (amount_paid_per_friend * number_of_friends) / number_of_bags

theorem chips_cost
  (amount_paid_per_friend : ℕ := 5)
  (number_of_friends : ℕ := 3)
  (number_of_bags : ℕ := 5) :
  cost_of_each_bag_of_chips amount_paid_per_friend number_of_friends number_of_bags = 3 :=
by
  sorry

end chips_cost_l179_179827


namespace pascal_triangle_row_20_element_5_l179_179049

theorem pascal_triangle_row_20_element_5 : nat.choose 20 4 = 4845 := by
  sorry

end pascal_triangle_row_20_element_5_l179_179049


namespace calculation_l179_179732

theorem calculation :
  ((4.5 - 1.23) * 2.5 = 8.175) := 
by
  sorry

end calculation_l179_179732


namespace find_x_given_y_l179_179960

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

end find_x_given_y_l179_179960


namespace orange_slices_l179_179948

theorem orange_slices (x : ℕ) (hx1 : 5 * x = x + 8) : x + 2 * x + 5 * x = 16 :=
by {
  sorry
}

end orange_slices_l179_179948


namespace arrange_animals_adjacent_l179_179813

theorem arrange_animals_adjacent:
  let chickens := 5
  let dogs := 3
  let cats := 6
  let rabbits := 4
  let total_animals := 18
  let group_orderings := 24 -- 4!
  let chicken_orderings := 120 -- 5!
  let dog_orderings := 6 -- 3!
  let cat_orderings := 720 -- 6!
  let rabbit_orderings := 24 -- 4!
  total_animals = chickens + dogs + cats + rabbits →
  chickens > 0 ∧ dogs > 0 ∧ cats > 0 ∧ rabbits > 0 →
  group_orderings * chicken_orderings * dog_orderings * cat_orderings * rabbit_orderings = 17863680 :=
  by intros; sorry

end arrange_animals_adjacent_l179_179813


namespace sum_of_consecutive_integers_l179_179600

theorem sum_of_consecutive_integers (S : ℕ) (hS : S = 560):
  ∃ (N : ℕ), N = 11 ∧ 
  ∀ n (k : ℕ), 2 ≤ n → (n * (2 * k + n - 1)) = 1120 → N = 11 :=
by
  sorry

end sum_of_consecutive_integers_l179_179600


namespace predict_participant_after_280_games_l179_179700

-- Definitions according to the problem conditions
def numWhiteRook : Nat := 15
def numBlackElephant : Nat := 20
def totalGames : Nat := numWhiteRook * numBlackElephant

-- The theorem to be proved based on the conditions and desired result
theorem predict_participant_after_280_games (n : Nat) (h1 : n = 280) :
  ∃ participant, participant ∈ (some_game_participants n) :=
sorry

end predict_participant_after_280_games_l179_179700


namespace rachel_age_is_19_l179_179019

def rachel_and_leah_ages (R L : ℕ) : Prop :=
  (R = L + 4) ∧ (R + L = 34)

theorem rachel_age_is_19 : ∃ L : ℕ, rachel_and_leah_ages 19 L :=
by {
  sorry
}

end rachel_age_is_19_l179_179019


namespace convert_rect_to_polar_l179_179260

theorem convert_rect_to_polar (y x : ℝ) (h : y = x) : ∃ θ : ℝ, θ = π / 4 :=
by
  sorry

end convert_rect_to_polar_l179_179260


namespace mark_asphalt_total_cost_l179_179802

noncomputable def total_cost (road_length : ℕ) (road_width : ℕ) (area_per_truckload : ℕ) (cost_per_truckload : ℕ) (sales_tax_rate : ℚ) : ℚ :=
  let total_area := road_length * road_width
  let num_truckloads := total_area / area_per_truckload
  let cost_before_tax := num_truckloads * cost_per_truckload
  let sales_tax := cost_before_tax * sales_tax_rate
  let total_cost := cost_before_tax + sales_tax
  total_cost

theorem mark_asphalt_total_cost :
  total_cost 2000 20 800 75 0.2 = 4500 := 
by sorry

end mark_asphalt_total_cost_l179_179802


namespace equilateral_triangle_roots_l179_179328

theorem equilateral_triangle_roots (p q : ℂ) (z1 z2 : ℂ) (h1 : z2 = Complex.exp (2 * Real.pi * Complex.I / 3) * z1)
  (h2 : 0 + p * z1 + q = 0) (h3 : p = -z1 - z2) (h4 : q = z1 * z2) : (p^2 / q) = 1 :=
by
  sorry

end equilateral_triangle_roots_l179_179328


namespace gcd_930_868_l179_179650

theorem gcd_930_868 : Nat.gcd 930 868 = 62 := by
  sorry

end gcd_930_868_l179_179650


namespace incorrect_contrapositive_l179_179725

theorem incorrect_contrapositive (x : ℝ) : (x ≠ 1 → ¬ (x^2 - 1 = 0)) ↔ ¬ (x^2 - 1 = 0 → x^2 = 1) := by
  sorry

end incorrect_contrapositive_l179_179725


namespace fencing_required_l179_179853

theorem fencing_required
  (L : ℝ) (W : ℝ) (A : ℝ) (F : ℝ)
  (hL : L = 25)
  (hA : A = 880)
  (hArea : A = L * W)
  (hF : F = L + 2 * W) :
  F = 95.4 :=
by
  sorry

end fencing_required_l179_179853


namespace remaining_watermelons_l179_179809

def initial_watermelons : ℕ := 4
def eaten_watermelons : ℕ := 3

theorem remaining_watermelons : initial_watermelons - eaten_watermelons = 1 :=
by sorry

end remaining_watermelons_l179_179809


namespace triangle_sides_proportional_l179_179356

theorem triangle_sides_proportional (a b c r d : ℝ)
  (h1 : 2 * r < a) 
  (h2 : a < b) 
  (h3 : b < c) 
  (h4 : a = 2 * r + d)
  (h5 : b = 2 * r + 2 * d)
  (h6 : c = 2 * r + 3 * d)
  (hr_pos : r > 0)
  (hd_pos : d > 0) :
  ∃ k : ℝ, k > 0 ∧ a = 3 * k ∧ b = 4 * k ∧ c = 5 * k :=
sorry

end triangle_sides_proportional_l179_179356


namespace smallest_x_l179_179835

theorem smallest_x (x : ℕ) :
  (x % 6 = 5) ∧ (x % 7 = 6) ∧ (x % 8 = 7) → x = 167 :=
by
  sorry

end smallest_x_l179_179835


namespace b_2_pow_100_value_l179_179326

def seq (b : ℕ → ℕ) : Prop :=
  b 1 = 3 ∧ ∀ n > 0, b (2 * n) = 2 * n * b n

theorem b_2_pow_100_value
  (b : ℕ → ℕ)
  (h_seq : seq b) :
  b (2^100) = 2^5050 * 3 :=
by
  sorry

end b_2_pow_100_value_l179_179326


namespace minimum_value_quadratic_function_l179_179357

noncomputable def quadratic_function (x : ℝ) : ℝ := 3 * x^2 + 2 * x + 1

theorem minimum_value_quadratic_function : ∀ x, x ≥ 0 → quadratic_function x ≥ 1 :=
by
  sorry

end minimum_value_quadratic_function_l179_179357


namespace sequence_term_l179_179103

theorem sequence_term (S : ℕ → ℕ) (h : ∀ (n : ℕ), S n = 5 * n + 2 * n^2) (r : ℕ) : 
  (S r - S (r - 1) = 4 * r + 3) :=
by {
  sorry
}

end sequence_term_l179_179103


namespace unique_pyramid_formation_l179_179712

theorem unique_pyramid_formation:
  ∀ (positions: Finset ℕ)
    (is_position_valid: ℕ → Prop),
    (positions.card = 5) → 
    (∀ n ∈ positions, n < 5) → 
    (∃! n, is_position_valid n) :=
by
  sorry

end unique_pyramid_formation_l179_179712


namespace unique_quadruple_exists_l179_179096

theorem unique_quadruple_exists :
  ∃! (a b c d : ℝ), 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d ∧
  a + b + c + d = 2 ∧
  a^2 + b^2 + c^2 + d^2 = 3 ∧
  (a + b + c + d) * (a^3 + b^3 + c^3 + d^3) = 18 := by
  sorry

end unique_quadruple_exists_l179_179096


namespace rhombus_area_l179_179497

theorem rhombus_area (d1 d2 : ℝ) (h1 : d1 = 15) (h2 : d2 = 12) : (d1 * d2) / 2 = 90 := by
  sorry

end rhombus_area_l179_179497


namespace weekly_milk_production_l179_179709

-- Define the conditions
def num_cows : ℕ := 52
def milk_per_cow_per_day : ℕ := 5
def days_per_week : ℕ := 7

-- Define the proof that total weekly milk production is 1820 liters
theorem weekly_milk_production : num_cows * milk_per_cow_per_day * days_per_week = 1820 := by
  sorry

end weekly_milk_production_l179_179709


namespace minimum_value_of_expression_l179_179324

noncomputable def min_value_expr (a b : ℝ) : ℝ :=
  (a + 1/b) * (a + 1/b - 2023) + (b + 1/a) * (b + 1/a - 2023)

theorem minimum_value_of_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  min_value_expr a b = -2031948.5 :=
  sorry

end minimum_value_of_expression_l179_179324


namespace arithmetic_sequence_inequality_l179_179884

noncomputable def a_n (n : ℕ) : ℕ := 2 * n + 1

noncomputable def b_n (n : ℕ) : ℝ := 1 / (a_n n ^ 2 - 1)

noncomputable def T_n (n : ℕ) : ℝ := ∑ i in finset.range n, b_n (i + 1)

theorem arithmetic_sequence_inequality (n : ℕ) : T_n n < 1 / 4 :=
by {
  -- Proof to be provided
  sorry
}

end arithmetic_sequence_inequality_l179_179884


namespace power_of_three_divides_an_l179_179551

theorem power_of_three_divides_an (a : ℕ → ℕ) (k : ℕ) (h1 : a 1 = 3)
  (h2 : ∀ n, a (n + 1) = ((3 * (a n)^2 + 1) / 2) - a n)
  (h3 : ∃ m, n = 3^m) :
  3^(k + 1) ∣ a (3^k) :=
sorry

end power_of_three_divides_an_l179_179551


namespace reduced_price_per_kg_l179_179075

variable (P R Q : ℝ)

theorem reduced_price_per_kg :
  R = 0.75 * P →
  1200 = (Q + 5) * R →
  Q * P = 1200 →
  R = 60 :=
by
  intro h₁ h₂ h₃
  sorry

end reduced_price_per_kg_l179_179075


namespace find_principal_amount_l179_179680

theorem find_principal_amount (P r : ℝ) (h1 : 720 = P * (1 + 2 * r)) (h2 : 1020 = P * (1 + 7 * r)) : P = 600 :=
by sorry

end find_principal_amount_l179_179680


namespace area_of_region_l179_179745

theorem area_of_region : 
    ∃ (area : ℝ), 
    (∀ (x y : ℝ), (x^2 + y^2 + 6 * x - 10 * y + 5 = 0) → 
    area = 29 * Real.pi) := 
by
  use 29 * Real.pi
  intros x y h
  sorry

end area_of_region_l179_179745


namespace min_value_one_over_x_plus_one_over_y_l179_179443

theorem min_value_one_over_x_plus_one_over_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x * y = 4) : 
  (1 / x + 1 / y) ≥ 1 :=
by
  sorry -- Proof goes here

end min_value_one_over_x_plus_one_over_y_l179_179443


namespace perpendicular_line_x_intercept_l179_179556

theorem perpendicular_line_x_intercept :
  (∃ x : ℝ, ∃ y : ℝ, 4 * x + 5 * y = 10) →
  (∃ y : ℝ, y = (5/4) * x - 3) →
  (∃ x : ℝ, y = 0) →
  x = 12 / 5 :=
by
  sorry

end perpendicular_line_x_intercept_l179_179556


namespace negation_of_exists_lt_zero_l179_179652

theorem negation_of_exists_lt_zero (m : ℝ) :
  ¬ (∃ x : ℝ, x < 0 ∧ x^2 + 2 * x - m > 0) ↔ ∀ x : ℝ, x < 0 → x^2 + 2 * x - m ≤ 0 :=
by sorry

end negation_of_exists_lt_zero_l179_179652


namespace curious_number_is_digit_swap_divisor_l179_179002

theorem curious_number_is_digit_swap_divisor (a b : ℕ) (hab : a ≠ 0 ∧ b ≠ 0) :
  (10 * a + b) ∣ (10 * b + a) → (10 * a + b) = 11 ∨ (10 * a + b) = 22 ∨ (10 * a + b) = 33 ∨ 
  (10 * a + b) = 44 ∨ (10 * a + b) = 55 ∨ (10 * a + b) = 66 ∨ 
  (10 * a + b) = 77 ∨ (10 * a + b) = 88 ∨ (10 * a + b) = 99 :=
by
  sorry

end curious_number_is_digit_swap_divisor_l179_179002


namespace evaluate_powers_of_i_l179_179412

theorem evaluate_powers_of_i :
  (Complex.I ^ 50) + (Complex.I ^ 105) = -1 + Complex.I :=
by 
  sorry

end evaluate_powers_of_i_l179_179412


namespace basketball_points_l179_179597

/-
In a basketball league, each game must have a winner and a loser. 
A team earns 2 points for a win and 1 point for a loss. 
A certain team expects to earn at least 48 points in all 32 games of 
the 2012-2013 season in order to have a chance to enter the playoffs. 
If this team wins x games in the upcoming matches, prove that
the relationship that x should satisfy to reach the goal is:
    2x + (32 - x) ≥ 48.
-/
theorem basketball_points (x : ℕ) (h : 0 ≤ x ∧ x ≤ 32) :
    2 * x + (32 - x) ≥ 48 :=
sorry

end basketball_points_l179_179597


namespace P_Ravi_is_02_l179_179047

def P_Ram : ℚ := 6 / 7
def P_Ram_and_Ravi : ℚ := 0.17142857142857143

theorem P_Ravi_is_02 (P_Ravi : ℚ) : P_Ram_and_Ravi = P_Ram * P_Ravi → P_Ravi = 0.2 :=
by
  intro h
  sorry

end P_Ravi_is_02_l179_179047


namespace M_lt_N_l179_179358

variables (a b c : ℝ)

def f (x : ℝ) : ℝ := a * x^2 + b * x + c

def N : ℝ := |a + b + c| + |2 * a - b|
def M : ℝ := |a - b + c| + |2 * a + b|

axiom h1 : f 1 < 0  -- a + b + c < 0
axiom h2 : f (-1) > 0  -- a - b + c > 0
axiom h3 : a > 0
axiom h4 : -b / (2 * a) > 1

theorem M_lt_N : M a b c < N a b c :=
by
  sorry

end M_lt_N_l179_179358


namespace focaccia_cost_l179_179265

theorem focaccia_cost :
  let almond_croissant := 4.50
  let salami_cheese_croissant := 4.50
  let plain_croissant := 3.00
  let latte := 2.50
  let total_spent := 21.00
  let known_costs := almond_croissant + salami_cheese_croissant + plain_croissant + 2 * latte
  let focaccia_cost := total_spent - known_costs
  focaccia_cost = 4.00 := 
by
  sorry

end focaccia_cost_l179_179265


namespace problem_b_is_proposition_l179_179975

def is_proposition (s : String) : Prop :=
  s = "sin 45° = 1" ∨ s = "x^2 + 2x - 1 > 0"

theorem problem_b_is_proposition : is_proposition "sin 45° = 1" :=
by
  -- insert proof steps to establish that "sin 45° = 1" is a proposition
  sorry

end problem_b_is_proposition_l179_179975


namespace find_constant_a_l179_179286

theorem find_constant_a (a : ℝ) : 
  (∃ x : ℝ, -3 ≤ x ∧ x ≤ 2 ∧ ax^2 + 2 * a * x + 1 = 9) → (a = -8 ∨ a = 1) :=
by
  sorry

end find_constant_a_l179_179286


namespace no_field_with_isomorphic_additive_and_multiplicative_groups_l179_179746

theorem no_field_with_isomorphic_additive_and_multiplicative_groups :
  ∀ (F : Type*) [field F], ¬∃ (f : F → (Fˣ)),
  (∀ x y : F, f (x + y) = f x * f y) := 
by
  -- introduce the field F and assume there exists an isomorphism f as described
  intros F hF
  simp
  sorry

end no_field_with_isomorphic_additive_and_multiplicative_groups_l179_179746


namespace largest_cube_side_length_largest_rect_parallelepiped_dimensions_l179_179473

variables {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

-- Part (a): Side length of the largest cube
theorem largest_cube_side_length :
  let a₀ := a * b * c / (a * b + b * c + a * c) in
  ∃ a₀, a₀ = a * b * c / (a * b + b * c + a * c) :=
begin
  sorry
end

-- Part (b): Dimensions of the largest rectangular parallelepiped
theorem largest_rect_parallelepiped_dimensions :
  let x := a / 3, y := b / 3, z := c / 3 in
  ∃ x y z, x = a / 3 ∧ y = b / 3 ∧ z = c / 3 :=
begin
  sorry
end

end largest_cube_side_length_largest_rect_parallelepiped_dimensions_l179_179473


namespace hyperbola_min_sum_dist_l179_179580

open Real

theorem hyperbola_min_sum_dist (x y : ℝ) (F1 F2 A B : ℝ × ℝ) :
  -- Conditions for the hyperbola and the foci
  (∀ (x y : ℝ), x^2 / 9 - y^2 / 6 = 1) →
  F1 = (-c, 0) →
  F2 = (c, 0) →
  -- Minimum value of |AF2| + |BF2|
  ∃ (l : ℝ × ℝ → Prop), l F1 ∧ (∃ A B, l A ∧ l B ∧ A = (-3, y_A) ∧ B = (-3, y_B) ) →
  |dist A F2| + |dist B F2| = 16 :=
by
  sorry

end hyperbola_min_sum_dist_l179_179580


namespace vector_parallel_m_eq_two_neg_two_l179_179583

theorem vector_parallel_m_eq_two_neg_two (m : ℝ) (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : 2 / x = m / y) : m = 2 ∨ m = -2 :=
by
  sorry

end vector_parallel_m_eq_two_neg_two_l179_179583


namespace smallest_pieces_to_remove_l179_179253

theorem smallest_pieces_to_remove 
  (total_fruit : ℕ)
  (friends : ℕ)
  (h_fruit : total_fruit = 30)
  (h_friends : friends = 4) 
  : ∃ k : ℕ, k = 2 ∧ ((total_fruit - k) % friends = 0) :=
sorry

end smallest_pieces_to_remove_l179_179253


namespace non_shaded_area_l179_179657

theorem non_shaded_area (s : ℝ) (hex_area : ℝ) (tri_area : ℝ) (non_shaded_area : ℝ) :
  s = 12 →
  hex_area = (3 * Real.sqrt 3 / 2) * s^2 →
  tri_area = (Real.sqrt 3 / 4) * (2 * s)^2 →
  non_shaded_area = hex_area - tri_area →
  non_shaded_area = 288 * Real.sqrt 3 :=
by
  intros hs hhex htri hnon
  sorry

end non_shaded_area_l179_179657


namespace pascal_fifth_element_row_20_l179_179055

theorem pascal_fifth_element_row_20 :
  (Nat.choose 20 4) = 4845 := by
  sorry

end pascal_fifth_element_row_20_l179_179055


namespace ratio_of_candies_l179_179786

theorem ratio_of_candies (emily_candies jennifer_candies bob_candies : ℕ)
  (h1 : emily_candies = 6)
  (h2 : bob_candies = 4)
  (h3 : jennifer_candies = 2 * emily_candies) : 
  jennifer_candies / bob_candies = 3 := 
by
  sorry

end ratio_of_candies_l179_179786


namespace solution_l179_179407

def problem_statement : Prop :=
  (3025 - 2880) ^ 2 / 225 = 93

theorem solution : problem_statement :=
by {
  sorry
}

end solution_l179_179407


namespace max_product_yields_831_l179_179967

open Nat

noncomputable def max_product_3digit_2digit : ℕ :=
  let digits : List ℕ := [1, 3, 5, 8, 9]
  let products := digits.permutations.filter (λ l, l.length ≥ 5).map (λ l, 
    let a := l.get? 0 |>.getD 0
    let b := l.get? 1 |>.getD 0
    let c := l.get? 2 |>.getD 0
    let d := l.get? 3 |>.getD 0
    let e := l.get? 4 |>.getD 0
    (100 * a + 10 * b + c) * (10 * d + e)
  )
  products.maximum.getD 0

theorem max_product_yields_831 : 
  ∃ (a b c d e : ℕ), 
    List.permutations [1, 3, 5, 8, 9].any 
      (λ l, l = [a, b, c, d, e] ∧ 
            100 * a + 10 * b + c = 831 ∧
            ∀ (x y z w v : ℕ), 
              List.permutations [1, 3, 5, 8, 9].any 
                (λ l, l = [x, y, z, w, v] ∧ 
                      (100 * x + 10 * y + z) * (10 * w + v) ≤ (100 * a + 10 * b + c) * (10 * d + e)
                )
      )
:= by
  sorry

end max_product_yields_831_l179_179967


namespace exists_polynomial_primes_l179_179490

noncomputable def find_polynomial (n : ℕ) : ℕ → ℤ :=
sorry

theorem exists_polynomial_primes (n : ℕ) (hn : 0 < n) :
  ∃ f : ℕ → ℤ, (∀ k : ℕ, 1 ≤ k ∧ k ≤ n → Nat.Prime (f k)) ∧
               (∀ k₁ k₂ : ℕ, 1 ≤ k₁ ∧ k₁ ≤ n → 1 ≤ k₂ ∧ k₂ ≤ n → k₁ < k₂ → f k₁ < f k₂) :=
sorry

end exists_polynomial_primes_l179_179490


namespace sqrt_product_simplified_l179_179161

theorem sqrt_product_simplified (x : ℝ) (hx : 0 ≤ x) :
  (Real.sqrt (50 * x) * Real.sqrt (18 * x) * Real.sqrt (32 * x)) = 84 * x * Real.sqrt (2 * x) :=
by 
  sorry

end sqrt_product_simplified_l179_179161


namespace solve_for_x_l179_179505

theorem solve_for_x (x y z : ℚ) (h1 : x * y = 2 * (x + y)) (h2 : y * z = 4 * (y + z)) (h3 : x * z = 8 * (x + z)) (hx0 : x ≠ 0) (hy0 : y ≠ 0) (hz0 : z ≠ 0) : x = 16 / 3 := 
sorry

end solve_for_x_l179_179505


namespace tan_neg4095_eq_one_l179_179399

theorem tan_neg4095_eq_one : Real.tan (Real.pi / 180 * -4095) = 1 := by
  sorry

end tan_neg4095_eq_one_l179_179399


namespace total_minutes_ironing_over_4_weeks_l179_179545

/-- Define the time spent ironing each day -/
def minutes_ironing_per_day : Nat := 5 + 3

/-- Define the number of days Hayden irons per week -/
def days_ironing_per_week : Nat := 5

/-- Define the number of weeks considered -/
def number_of_weeks : Nat := 4

/-- The main theorem we're proving is that Hayden spends 160 minutes ironing over 4 weeks -/
theorem total_minutes_ironing_over_4_weeks :
  (minutes_ironing_per_day * days_ironing_per_week * number_of_weeks) = 160 := by
  sorry

end total_minutes_ironing_over_4_weeks_l179_179545


namespace min_vertical_distance_between_graphs_l179_179039

noncomputable def min_distance (x : ℝ) : ℝ :=
  |x| - (-x^2 - 4 * x - 2)

theorem min_vertical_distance_between_graphs :
  ∃ x : ℝ, ∀ y : ℝ, min_distance x ≤ min_distance y := 
    sorry

end min_vertical_distance_between_graphs_l179_179039


namespace g_at_4_l179_179146

noncomputable def g : ℝ → ℝ := sorry

axiom g_condition : ∀ x : ℝ, g x + 3 * g (2 - x) = 4 * x^2

theorem g_at_4 : g 4 = -2 :=
by
  -- Proof would go here
  sorry

end g_at_4_l179_179146


namespace unique_set_of_consecutive_integers_l179_179295

theorem unique_set_of_consecutive_integers (a b c : ℕ) : 
  (a + b + c = 36) ∧ (b = a + 1) ∧ (c = a + 2) → 
  ∃! a : ℕ, (a = 11 ∧ b = 12 ∧ c = 13) := 
sorry

end unique_set_of_consecutive_integers_l179_179295


namespace dorothy_age_relation_l179_179408

theorem dorothy_age_relation (D S : ℕ) (h1: S = 5) (h2: D + 5 = 2 * (S + 5)) : D = 3 * S :=
by
  -- implement the proof here
  sorry

end dorothy_age_relation_l179_179408


namespace find_incorrect_statement_l179_179520

theorem find_incorrect_statement :
  ¬ (∀ a b c : ℝ, c ≠ 0 → (a < b → a * c^2 < b * c^2)) :=
by
  sorry

end find_incorrect_statement_l179_179520


namespace no_valid_prime_l179_179344

open Nat

def base_p_polynomial (p : ℕ) (coeffs : List ℕ) : ℕ → ℕ :=
  fun (n : ℕ) => coeffs.foldl (λ sum coef => sum * p + coef) 0

def num_1013 (p : ℕ) := base_p_polynomial p [1, 0, 1, 3]
def num_207 (p : ℕ) := base_p_polynomial p [2, 0, 7]
def num_214 (p : ℕ) := base_p_polynomial p [2, 1, 4]
def num_100 (p : ℕ) := base_p_polynomial p [1, 0, 0]
def num_10 (p : ℕ) := base_p_polynomial p [1, 0]

def num_321 (p : ℕ) := base_p_polynomial p [3, 2, 1]
def num_403 (p : ℕ) := base_p_polynomial p [4, 0, 3]
def num_210 (p : ℕ) := base_p_polynomial p [2, 1, 0]

theorem no_valid_prime (p : ℕ) [Fact (Nat.Prime p)] :
  num_1013 p + num_207 p + num_214 p + num_100 p + num_10 p ≠
  num_321 p + num_403 p + num_210 p := by
  sorry

end no_valid_prime_l179_179344


namespace arithmetic_sequence_sum_l179_179313

noncomputable def isArithmeticSeq (a : ℕ → ℤ) := ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

noncomputable def sum_first_n (a : ℕ → ℤ) (n : ℕ) := (n + 1) * (a 0 + a n) / 2

theorem arithmetic_sequence_sum (a : ℕ → ℤ) (h_legal_seq : isArithmeticSeq a) (h_sum : sum_first_n a 9 = 120) : 
  a 1 + a 8 = 24 := by
  sorry

end arithmetic_sequence_sum_l179_179313


namespace units_digit_base7_of_multiplied_numbers_l179_179640

-- Define the numbers in base 10
def num1 : ℕ := 325
def num2 : ℕ := 67

-- Define the modulus used for base 7
def base : ℕ := 7

-- Function to determine the units digit of the base-7 representation
def units_digit_base7 (n : ℕ) : ℕ := n % base

-- Prove that units_digit_base7 (num1 * num2) = 5
theorem units_digit_base7_of_multiplied_numbers :
  units_digit_base7 (num1 * num2) = 5 :=
by
  sorry

end units_digit_base7_of_multiplied_numbers_l179_179640


namespace log_inequality_l179_179154

theorem log_inequality (a b c : ℝ) (ha : 1 < a) (hb : 1 < b) (hc : 1 < c) :
  2 * ((Real.log a / Real.log b) / (a + b) + (Real.log b / Real.log c) / (b + c) + (Real.log c / Real.log a) / (c + a)) 
    ≥ 9 / (a + b + c) :=
by
  sorry

end log_inequality_l179_179154


namespace range_of_a_l179_179425

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 - 2 * x - 3 < 0 → x > a) ∧ (∃ x : ℝ, x > a ∧ ¬(x^2 - 2 * x - 3 < 0)) → a ≤ -1 :=
by
  sorry

end range_of_a_l179_179425


namespace least_whole_number_subtracted_from_ratio_l179_179839

theorem least_whole_number_subtracted_from_ratio (x : ℕ) : 
  (6 - x) / (7 - x) < 16 / 21 := by
  sorry

end least_whole_number_subtracted_from_ratio_l179_179839


namespace f_14_52_l179_179649

def f : ℕ × ℕ → ℕ := sorry

axiom f_xx (x : ℕ) : f (x, x) = x
axiom f_symm (x y : ℕ) : f (x, y) = f (y, x)
axiom f_eq (x y : ℕ) : (x + y) * f (x, y) = y * f (x, x + y)

theorem f_14_52 : f (14, 52) = 364 := sorry

end f_14_52_l179_179649


namespace row_sum_odd_probability_l179_179654

open ProbabilityTheory

theorem row_sum_odd_probability :
  let nums := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
  let rows := {r : fin 4 → fin 3 → ℕ // ∀ i j, r i j ∈ nums ∧ ∀ n ∈ nums, ∃ i j, r i j = n}
  let count_odd_sums := rows.filter (λ r, ∀ i, odd (∑ j, r i j))
  (count_odd_sums.card / rows.card : ℚ) = 1 / 22176 :=
sorry

end row_sum_odd_probability_l179_179654


namespace real_complex_number_l179_179592

theorem real_complex_number (x : ℝ) (hx1 : x^2 - 3 * x - 3 > 0) (hx2 : x - 3 = 1) : x = 4 :=
by
  sorry

end real_complex_number_l179_179592


namespace largest_base5_eq_124_l179_179214

-- Define largest base-5 number with three digits
def largest_base5_three_digits : ℕ := 4 * 5^2 + 4 * 5^1 + 4 * 5^0

theorem largest_base5_eq_124 : largest_base5_three_digits = 124 :=
by
  -- Proof steps go here
  sorry

end largest_base5_eq_124_l179_179214


namespace complex_exchange_of_apartments_in_two_days_l179_179454

theorem complex_exchange_of_apartments_in_two_days :
  ∀ (n : ℕ) (p : Fin n → Fin n), ∃ (day1 day2 : Fin n → Fin n),
    (∀ x : Fin n, p (day1 x) = day2 x ∨ day1 (p x) = day2 x) ∧
    (∀ x : Fin n, day1 x ≠ x) ∧
    (∀ x : Fin n, day2 x ≠ x) :=
by
  sorry

end complex_exchange_of_apartments_in_two_days_l179_179454


namespace sum_of_abs_squared_series_correct_l179_179401

noncomputable def sum_of_abs_squared_series (a r : ℝ) (h : |r| < 1) : ℝ :=
  a^2 / (1 - |r|^2)

theorem sum_of_abs_squared_series_correct (a r : ℝ) (h : |r| < 1) :
  sum_of_abs_squared_series a r h = a^2 / (1 - |r|^2) :=
by
  sorry

end sum_of_abs_squared_series_correct_l179_179401


namespace pairs_divisible_by_7_l179_179441

theorem pairs_divisible_by_7 :
  (∃ (pairs : List (ℕ × ℕ)), 
    (∀ p ∈ pairs, (1 ≤ p.fst ∧ p.fst ≤ 1000) ∧ (1 ≤ p.snd ∧ p.snd ≤ 1000) ∧ (p.fst^2 + p.snd^2) % 7 = 0) ∧ 
    pairs.length = 20164) :=
sorry

end pairs_divisible_by_7_l179_179441


namespace num_real_solutions_frac_sine_l179_179097

theorem num_real_solutions_frac_sine :
  (∃ n : ℕ, ∀ x : ℝ, x ∈ Icc (-150) 150 → (x/150 = Real.sin x) ↔ (n = 95)) := 
sorry

end num_real_solutions_frac_sine_l179_179097


namespace points_on_fourth_board_l179_179743

theorem points_on_fourth_board (P_1 P_2 P_3 P_4 : ℕ)
 (h1 : P_1 = 30)
 (h2 : P_2 = 38)
 (h3 : P_3 = 41) :
  P_4 = 34 :=
sorry

end points_on_fourth_board_l179_179743


namespace gcd_polynomial_l179_179109

theorem gcd_polynomial (b : ℤ) (h : 570 ∣ b) :
  Int.gcd (5 * b^4 + 2 * b^3 + 5 * b^2 + 9 * b + 95) b = 95 :=
sorry

end gcd_polynomial_l179_179109


namespace goldfish_added_per_day_is_7_l179_179157

def initial_koi_fish : ℕ := 227 - 2
def initial_goldfish : ℕ := 280 - initial_koi_fish
def added_goldfish : ℕ := 200 - initial_goldfish
def days_in_three_weeks : ℕ := 3 * 7
def goldfish_added_per_day : ℕ := (added_goldfish + days_in_three_weeks - 1) / days_in_three_weeks -- rounding to nearest integer 

theorem goldfish_added_per_day_is_7 : goldfish_added_per_day = 7 :=
by 
-- sorry to skip the proof
sorry

end goldfish_added_per_day_is_7_l179_179157


namespace selling_price_of_article_l179_179719

theorem selling_price_of_article (cost_price : ℕ) (gain_percent : ℕ) (profit : ℕ) (selling_price : ℕ) : 
  cost_price = 100 → gain_percent = 10 → profit = (gain_percent * cost_price) / 100 → selling_price = cost_price + profit → selling_price = 110 :=
by
  intros
  sorry

end selling_price_of_article_l179_179719


namespace range_of_a_l179_179369

open Real

noncomputable def doesNotPassThroughSecondQuadrant (a : ℝ) : Prop :=
  ∀ x y : ℝ, (3 * a - 1) * x + (2 - a) * y - 1 ≠ 0

theorem range_of_a : {a : ℝ | doesNotPassThroughSecondQuadrant a} = {a : ℝ | 2 ≤ a } :=
by
  ext
  sorry

end range_of_a_l179_179369


namespace perimeter_square_l179_179079

-- Definition of the side length
def side_length : ℝ := 9

-- Definition of the perimeter calculation
def perimeter (s : ℝ) : ℝ := 4 * s

-- Theorem stating that the perimeter of a square with side length 9 cm is 36 cm
theorem perimeter_square : perimeter side_length = 36 := 
by sorry

end perimeter_square_l179_179079


namespace max_checkers_on_chessboard_l179_179671

theorem max_checkers_on_chessboard (n : ℕ) : 
  ∃ k : ℕ, k = 2 * n * (n / 2) := sorry

end max_checkers_on_chessboard_l179_179671


namespace negation_of_proposition_p_l179_179581

variable (x : ℝ)

def proposition_p : Prop := ∃ x₀ : ℝ, x₀^2 + 2*x₀ + 2 ≤ 0

theorem negation_of_proposition_p : ¬ proposition_p ↔ ∀ x : ℝ, x^2 + 2*x + 2 > 0 := by
  sorry

end negation_of_proposition_p_l179_179581


namespace sum_sequence_six_l179_179141

def S (n : ℕ) : ℕ := sorry
def a (n : ℕ) : ℤ := sorry

theorem sum_sequence_six :
  (∀ n, S n = 2 * a n + 1) → S 6 = 63 :=
by
  sorry

end sum_sequence_six_l179_179141


namespace sum_of_a_b_l179_179178

theorem sum_of_a_b (a b : ℝ) (h1 : a * b = 1) (h2 : (3 * a + 2 * b) * (3 * b + 2 * a) = 295) : a + b = 7 :=
by
  sorry

end sum_of_a_b_l179_179178


namespace pascal_fifth_element_row_20_l179_179052

def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem pascal_fifth_element_row_20 : binom 20 4 = 4845 := sorry

end pascal_fifth_element_row_20_l179_179052

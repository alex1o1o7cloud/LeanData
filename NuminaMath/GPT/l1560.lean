import Mathlib

namespace hours_learning_english_each_day_l1560_156027

theorem hours_learning_english_each_day (total_hours : ℕ) (days : ℕ) (learning_hours_per_day : ℕ) 
  (h1 : total_hours = 12) 
  (h2 : days = 2) 
  (h3 : total_hours = learning_hours_per_day * days) : 
  learning_hours_per_day = 6 := 
by
  sorry

end hours_learning_english_each_day_l1560_156027


namespace isosceles_right_triangle_area_l1560_156089

theorem isosceles_right_triangle_area (h : ℝ) (l : ℝ) (A : ℝ)
  (h_def : h = 6 * Real.sqrt 2)
  (rel_leg_hypotenuse : h = Real.sqrt 2 * l)
  (area_def : A = 1 / 2 * l * l) :
  A = 18 :=
by
  sorry

end isosceles_right_triangle_area_l1560_156089


namespace chloe_treasures_first_level_l1560_156002

def chloe_treasures_score (T : ℕ) (score_per_treasure : ℕ) (treasures_second_level : ℕ) (total_score : ℕ) :=
  T * score_per_treasure + treasures_second_level * score_per_treasure = total_score

theorem chloe_treasures_first_level :
  chloe_treasures_score T 9 3 81 → T = 6 :=
by
  intro h
  sorry

end chloe_treasures_first_level_l1560_156002


namespace Jimin_weight_l1560_156080

variable (T J : ℝ)

theorem Jimin_weight (h1 : T - J = 4) (h2 : T + J = 88) : J = 42 :=
sorry

end Jimin_weight_l1560_156080


namespace A_investment_l1560_156036

theorem A_investment (B_invest C_invest Total_profit A_share : ℝ) 
  (h1 : B_invest = 4200)
  (h2 : C_invest = 10500)
  (h3 : Total_profit = 12100)
  (h4 : A_share = 3630) 
  (h5 : ∀ {x : ℝ}, A_share / Total_profit = x / (x + B_invest + C_invest)) :
  ∃ A_invest : ℝ, A_invest = 6300 :=
by sorry

end A_investment_l1560_156036


namespace roots_quadratic_sum_of_squares_l1560_156045

theorem roots_quadratic_sum_of_squares :
  ∀ x1 x2 : ℝ, (x1^2 - 2*x1 - 1 = 0 ∧ x2^2 - 2*x2 - 1 = 0) → x1^2 + x2^2 = 6 :=
by
  intros x1 x2 h
  -- proof goes here
  sorry

end roots_quadratic_sum_of_squares_l1560_156045


namespace sum_of_three_squares_l1560_156009

theorem sum_of_three_squares (n : ℕ) (h_pos : 0 < n) (h_square : ∃ m : ℕ, 3 * n + 1 = m^2) : ∃ x y z : ℕ, n + 1 = x^2 + y^2 + z^2 :=
by
  sorry

end sum_of_three_squares_l1560_156009


namespace path_count_1800_l1560_156031

-- Define the coordinates of the points
def A := (0, 8)
def B := (4, 5)
def C := (7, 2)
def D := (9, 0)

-- Function to calculate the number of combinatorial paths
def comb_paths (steps_right steps_down : ℕ) : ℕ :=
  Nat.choose (steps_right + steps_down) steps_right

-- Define the number of steps for each segment
def steps_A_B := (4, 2)  -- 4 right, 2 down
def steps_B_C := (3, 3)  -- 3 right, 3 down
def steps_C_D := (2, 2)  -- 2 right, 2 down

-- Calculate the number of paths for each segment
def paths_A_B := comb_paths steps_A_B.1 steps_A_B.2
def paths_B_C := comb_paths steps_B_C.1 steps_B_C.2
def paths_C_D := comb_paths steps_C_D.1 steps_C_D.2

-- Calculate the total number of paths combining all segments
def total_paths : ℕ :=
  paths_A_B * paths_B_C * paths_C_D

theorem path_count_1800 :
  total_paths = 1800 := by
  sorry

end path_count_1800_l1560_156031


namespace min_value_of_c_l1560_156040

noncomputable def isPerfectSquare (x : ℕ) : Prop :=
  ∃ m : ℕ, x = m^2

noncomputable def isPerfectCube (x : ℕ) : Prop :=
  ∃ n : ℕ, x = n^3

theorem min_value_of_c (c : ℕ) :
  (∃ a b d e : ℕ, a = c-2 ∧ b = c-1 ∧ d = c+1 ∧ e = c+2 ∧ a < b ∧ b < c ∧ c < d ∧ d < e) ∧
  isPerfectSquare (3 * c) ∧
  isPerfectCube (5 * c) →
  c = 675 :=
sorry

end min_value_of_c_l1560_156040


namespace silenos_time_l1560_156051

theorem silenos_time :
  (∃ x : ℝ, ∃ b: ℝ, (x - 2 = x / 2) ∧ (b = x / 3)) → (∃ x : ℝ, x = 3) :=
by sorry

end silenos_time_l1560_156051


namespace factorization_eq1_factorization_eq2_l1560_156063

-- Definitions for the given conditions
variables (a b x y m : ℝ)

-- The problem statement as Lean definitions and the goal theorems
def expr1 : ℝ := -6 * a * b + 3 * a^2 + 3 * b^2
def factored1 : ℝ := 3 * (a - b)^2

def expr2 : ℝ := y^2 * (2 - m) + x^2 * (m - 2)
def factored2 : ℝ := (m - 2) * (x + y) * (x - y)

-- Theorem statements for equivalence
theorem factorization_eq1 : expr1 a b = factored1 a b :=
by
  sorry

theorem factorization_eq2 : expr2 x y m = factored2 x y m :=
by
  sorry

end factorization_eq1_factorization_eq2_l1560_156063


namespace plane_equation_l1560_156072

variable (a b c : ℝ)
variable (ha : a ≠ 0)
variable (hb : b ≠ 0)
variable (hc : c ≠ 0)

theorem plane_equation (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  ∃ x y z : ℝ, (x / a + y / b + z / c = 1) :=
sorry

end plane_equation_l1560_156072


namespace evaluate_polynomial_at_minus_two_l1560_156073

noncomputable def polynomial (x : ℝ) : ℝ := 2 * x^4 + 3 * x^3 - x^2 + 2 * x + 5

theorem evaluate_polynomial_at_minus_two : polynomial (-2) = 5 := by
  sorry

end evaluate_polynomial_at_minus_two_l1560_156073


namespace pie_charts_cannot_show_changes_l1560_156056

def pie_chart_shows_part_whole (P : Type) := true
def bar_chart_shows_amount (B : Type) := true
def line_chart_shows_amount_and_changes (L : Type) := true

theorem pie_charts_cannot_show_changes (P B L : Type) :
  pie_chart_shows_part_whole P ∧ bar_chart_shows_amount B ∧ line_chart_shows_amount_and_changes L →
  ¬ (pie_chart_shows_part_whole P ∧ ¬ line_chart_shows_amount_and_changes P) :=
by sorry

end pie_charts_cannot_show_changes_l1560_156056


namespace negation_of_proposition_l1560_156077

theorem negation_of_proposition :
  (¬∃ x₀ ∈ Set.Ioo 0 (π/2), Real.cos x₀ > Real.sin x₀) ↔ ∀ x ∈ Set.Ioo 0 (π / 2), Real.cos x ≤ Real.sin x :=
by
  sorry

end negation_of_proposition_l1560_156077


namespace salary_increase_l1560_156093

theorem salary_increase (S : ℝ) (P : ℝ) (H0 : P > 0 )  
  (saved_last_year : ℝ := 0.10 * S)
  (salary_this_year : ℝ := S * (1 + P / 100))
  (saved_this_year : ℝ := 0.15 * salary_this_year)
  (H1 : saved_this_year = 1.65 * saved_last_year) :
  P = 10 :=
by
  sorry

end salary_increase_l1560_156093


namespace fraction_of_quarters_from_1860_to_1869_l1560_156044

theorem fraction_of_quarters_from_1860_to_1869
  (total_quarters : ℕ) (quarters_from_1860s : ℕ)
  (h1 : total_quarters = 30) (h2 : quarters_from_1860s = 15) :
  (quarters_from_1860s : ℚ) / (total_quarters : ℚ) = 1 / 2 := by
  sorry

end fraction_of_quarters_from_1860_to_1869_l1560_156044


namespace min_distance_from_origin_to_line_l1560_156057

open Real

theorem min_distance_from_origin_to_line :
    ∀ x y : ℝ, (3 * x + 4 * y - 4 = 0) -> dist (0, 0) (x, y) = 4 / 5 :=
by
  sorry

end min_distance_from_origin_to_line_l1560_156057


namespace blocks_eaten_correct_l1560_156083

def initial_blocks : ℕ := 55
def remaining_blocks : ℕ := 26

-- How many blocks were eaten by the hippopotamus?
def blocks_eaten_by_hippopotamus : ℕ := initial_blocks - remaining_blocks

theorem blocks_eaten_correct :
  blocks_eaten_by_hippopotamus = 29 := by
  sorry

end blocks_eaten_correct_l1560_156083


namespace add_least_number_l1560_156087

theorem add_least_number (n : ℕ) (h1 : n = 1789) (h2 : ∃ k : ℕ, 5 * k = n + 11) (h3 : ∃ j : ℕ, 6 * j = n + 11) (h4 : ∃ m : ℕ, 4 * m = n + 11) (h5 : ∃ l : ℕ, 11 * l = n + 11) : 11 = 11 :=
by
  sorry

end add_least_number_l1560_156087


namespace resultant_force_correct_l1560_156006

-- Define the conditions
def P1 : ℝ := 80
def P2 : ℝ := 130
def distance : ℝ := 12.035
def theta1 : ℝ := 125
def theta2 : ℝ := 135.1939

-- Calculate the correct answer
def result_magnitude : ℝ := 209.299
def result_direction : ℝ := 131.35

-- The goal statement to be proved
theorem resultant_force_correct :
  ∃ (R : ℝ) (theta_R : ℝ), 
    R = result_magnitude ∧ theta_R = result_direction := 
sorry

end resultant_force_correct_l1560_156006


namespace S6_eq_24_l1560_156022

-- Definitions based on the conditions provided
def is_arithmetic_sequence (seq : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, seq (n + 1) - seq n = seq (n + 2) - seq (n + 1)

def S : ℕ → ℝ := sorry  -- Sum of the first n terms of some arithmetic sequence

-- Given conditions
axiom S2_eq_2 : S 2 = 2
axiom S4_eq_10 : S 4 = 10

-- The main theorem to prove
theorem S6_eq_24 : S 6 = 24 :=
by 
  sorry  -- Proof is omitted

end S6_eq_24_l1560_156022


namespace least_number_l1560_156091

theorem least_number (n p q r s : ℕ) : 
  (n + p) % 24 = 0 ∧ 
  (n + q) % 32 = 0 ∧ 
  (n + r) % 36 = 0 ∧
  (n + s) % 54 = 0 →
  n = 863 :=
sorry

end least_number_l1560_156091


namespace range_of_a_l1560_156079

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x ≥ a → |x - 1| < 1) → (∃ x : ℝ, |x - 1| < 1 ∧ x < a) → a ≤ 0 := 
sorry

end range_of_a_l1560_156079


namespace smallest_integer_not_expressible_in_form_l1560_156076

theorem smallest_integer_not_expressible_in_form :
  ∀ (n : ℕ), (0 < n ∧ (∀ a b c d : ℕ, n ≠ (2^a - 2^b) / (2^c - 2^d))) ↔ n = 11 :=
by
  sorry

end smallest_integer_not_expressible_in_form_l1560_156076


namespace find_x_of_floor_eq_72_l1560_156015

theorem find_x_of_floor_eq_72 (x : ℝ) (hx_pos : 0 < x) (hx_eq : x * ⌊x⌋ = 72) : x = 9 :=
by 
  sorry

end find_x_of_floor_eq_72_l1560_156015


namespace intersection_complement_l1560_156035

-- Definitions
def I : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {2, 3, 5}
def B : Set ℕ := {1, 2}

-- Statement to prove
theorem intersection_complement :
  (((I \ B) ∩ A : Set ℕ) = {3, 5}) :=
by
  sorry

end intersection_complement_l1560_156035


namespace balloon_height_per_ounce_l1560_156019

theorem balloon_height_per_ounce
    (total_money : ℕ)
    (sheet_cost : ℕ)
    (rope_cost : ℕ)
    (propane_cost : ℕ)
    (helium_price : ℕ)
    (max_height : ℕ)
    :
    total_money = 200 →
    sheet_cost = 42 →
    rope_cost = 18 →
    propane_cost = 14 →
    helium_price = 150 →
    max_height = 9492 →
    max_height / ((total_money - (sheet_cost + rope_cost + propane_cost)) / helium_price) = 113 :=
by
  intros
  sorry

end balloon_height_per_ounce_l1560_156019


namespace Razorback_total_revenue_l1560_156000

def t_shirt_price : ℕ := 51
def t_shirt_discount : ℕ := 8
def hat_price : ℕ := 28
def hat_discount : ℕ := 5
def t_shirts_sold : ℕ := 130
def hats_sold : ℕ := 85

def discounted_t_shirt_price : ℕ := t_shirt_price - t_shirt_discount
def discounted_hat_price : ℕ := hat_price - hat_discount

def revenue_from_t_shirts : ℕ := t_shirts_sold * discounted_t_shirt_price
def revenue_from_hats : ℕ := hats_sold * discounted_hat_price

def total_revenue : ℕ := revenue_from_t_shirts + revenue_from_hats

theorem Razorback_total_revenue : total_revenue = 7545 := by
  unfold total_revenue
  unfold revenue_from_t_shirts
  unfold revenue_from_hats
  unfold discounted_t_shirt_price
  unfold discounted_hat_price
  unfold t_shirts_sold
  unfold hats_sold
  unfold t_shirt_price
  unfold t_shirt_discount
  unfold hat_price
  unfold hat_discount
  sorry

end Razorback_total_revenue_l1560_156000


namespace equation1_solution_equation2_solutions_l1560_156090

theorem equation1_solution (x : ℝ) : (x - 2) * (x - 3) = x - 2 → (x = 2 ∨ x = 4) :=
by
  intro h
  have h1 : (x - 2) * (x - 3) - (x - 2) = 0 := by sorry
  have h2 : (x - 2) * (x - 4) = 0 := by sorry
  have h3 : x - 2 = 0 ∨ x - 4 = 0 := by sorry
  cases h3 with
  | inl h4 => left; exact eq_of_sub_eq_zero h4
  | inr h5 => right; exact eq_of_sub_eq_zero h5

theorem equation2_solutions (x : ℝ) : 2 * x^2 - 5 * x + 1 = 0 → (x = (5 + Real.sqrt 17) / 4 ∨ x = (5 - Real.sqrt 17) / 4) :=
by
  intro h
  have h1 : (-5)^2 - 4 * 2 * 1 = 17 := by sorry
  have h2 : 2 * x^2 - 5 * x + 1 = 2 * ((x - (5 + Real.sqrt 17) / 4) * (x - (5 - Real.sqrt 17) / 4)) := by sorry
  have h3 : (x = (5 + Real.sqrt 17) / 4 ∨ x = (5 - Real.sqrt 17) / 4) := by sorry
  exact h3

end equation1_solution_equation2_solutions_l1560_156090


namespace generatrix_length_of_cone_l1560_156064

theorem generatrix_length_of_cone (r l : ℝ) (π : ℝ) (sqrt : ℝ → ℝ) 
  (hx : r = sqrt 2)
  (h_baseline_length : 2 * π * r = π * l) : 
  l = 2 * sqrt 2 :=
by
  sorry

end generatrix_length_of_cone_l1560_156064


namespace polygon_properties_l1560_156071

def interior_angle_sum (n : ℕ) : ℝ :=
  (n - 2) * 180

def exterior_angle_sum : ℝ :=
  360

theorem polygon_properties (n : ℕ) (h : interior_angle_sum n = 3 * exterior_angle_sum + 180) :
  n = 9 ∧ interior_angle_sum n / n = 140 :=
by
  sorry

end polygon_properties_l1560_156071


namespace lollipop_problem_l1560_156003

def Henry_lollipops (A : Nat) : Nat := A + 30
def Diane_lollipops (A : Nat) : Nat := 2 * A
def Total_days (H A D : Nat) (daily_rate : Nat) : Nat := (H + A + D) / daily_rate

theorem lollipop_problem
  (A : Nat) (H : Nat) (D : Nat) (daily_rate : Nat)
  (h₁ : A = 60)
  (h₂ : H = Henry_lollipops A)
  (h₃ : D = Diane_lollipops A)
  (h₄ : daily_rate = 45)
  : Total_days H A D daily_rate = 6 := by
  sorry

end lollipop_problem_l1560_156003


namespace quadratic_root_and_coefficient_l1560_156095

theorem quadratic_root_and_coefficient (k : ℝ) :
  (∃ x : ℝ, 5 * x^2 + k * x - 6 = 0 ∧ x = 2) →
  (∃ x₁ : ℝ, (5 * x₁^2 + k * x₁ - 6 = 0 ∧ x₁ ≠ 2) ∧ x₁ = -3/5 ∧ k = -7) :=
by
  sorry

end quadratic_root_and_coefficient_l1560_156095


namespace internal_diagonal_cubes_l1560_156069

-- Define the dimensions of the rectangular solid
def x_dimension : ℕ := 168
def y_dimension : ℕ := 350
def z_dimension : ℕ := 390

-- Define the GCD calculations for the given dimensions
def gcd_xy : ℕ := Nat.gcd x_dimension y_dimension
def gcd_yz : ℕ := Nat.gcd y_dimension z_dimension
def gcd_zx : ℕ := Nat.gcd z_dimension x_dimension
def gcd_xyz : ℕ := Nat.gcd (Nat.gcd x_dimension y_dimension) z_dimension

-- Define a statement that the internal diagonal passes through a certain number of cubes
theorem internal_diagonal_cubes :
  x_dimension + y_dimension + z_dimension - gcd_xy - gcd_yz - gcd_zx + gcd_xyz = 880 :=
by
  -- Configuration of conditions and proof skeleton with sorry
  sorry

end internal_diagonal_cubes_l1560_156069


namespace school_fee_l1560_156098

theorem school_fee (a b c d e f g h i j k l : ℕ) (h1 : a = 2) (h2 : b = 100) (h3 : c = 1) (h4 : d = 50) (h5 : e = 5) (h6 : f = 20) (h7 : g = 3) (h8 : h = 10) (h9 : i = 4) (h10 : j = 5) (h11 : k = 4 ) (h12 : l = 50) :
  a * b + c * d + e * f + g * h + i * j + 3 * b + k * d + 2 * f + l * h + 6 * j = 980 := sorry

end school_fee_l1560_156098


namespace min_value_of_linear_expression_l1560_156010

theorem min_value_of_linear_expression {x y : ℝ} (h1 : 2 * x - y ≥ 0) (h2 : x + y - 3 ≥ 0) (h3 : y - x ≥ 0) :
  ∃ z, z = 2 * x + y ∧ z = 4 := by
  sorry

end min_value_of_linear_expression_l1560_156010


namespace hexagon_side_lengths_l1560_156059

theorem hexagon_side_lengths (a b c d e f : ℕ) (h : a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ e ∧ e ≠ f)
(h1: a = 7 ∧ b = 5 ∧ (a + b + c + d + e + f = 38)) : 
(a + b + c + d + e + f = 38 ∧ a + b + c + d + e + f = 7 + 7 + 7 + 7 + 5 + 5) → 
(a + b + c + d + e + f = (4 * 7) + (2 * 5)) :=
sorry

end hexagon_side_lengths_l1560_156059


namespace min_value_A2_minus_B2_l1560_156050

noncomputable def A (p q r : ℝ) : ℝ := 
  Real.sqrt (p + 3) + Real.sqrt (q + 6) + Real.sqrt (r + 12)

noncomputable def B (p q r : ℝ) : ℝ :=
  Real.sqrt (p + 2) + Real.sqrt (q + 2) + Real.sqrt (r + 2)

theorem min_value_A2_minus_B2
  (h₁ : 0 ≤ p)
  (h₂ : 0 ≤ q)
  (h₃ : 0 ≤ r) :
  ∃ (p q r : ℝ), A p q r ^ 2 - B p q r ^ 2 = 35 + 10 * Real.sqrt 10 := 
sorry

end min_value_A2_minus_B2_l1560_156050


namespace min_integer_solution_l1560_156085

theorem min_integer_solution (x : ℤ) (h1 : 3 - x > 0) (h2 : (4 * x / 3 : ℚ) + 3 / 2 > -(x / 6)) : x = 0 := by
  sorry

end min_integer_solution_l1560_156085


namespace parabola_increasing_implies_a_lt_zero_l1560_156046

theorem parabola_increasing_implies_a_lt_zero (a : ℝ) :
  (∀ x : ℝ, x < 0 → a * (2 * x) > 0) → a < 0 :=
by
  sorry

end parabola_increasing_implies_a_lt_zero_l1560_156046


namespace probability_even_sum_is_half_l1560_156039

-- Definitions for probability calculations
def prob_even_A : ℚ := 2 / 5
def prob_odd_A : ℚ := 3 / 5
def prob_even_B : ℚ := 1 / 2
def prob_odd_B : ℚ := 1 / 2

-- Sum is even if both are even or both are odd
def prob_even_sum := prob_even_A * prob_even_B + prob_odd_A * prob_odd_B

-- Theorem stating the final probability
theorem probability_even_sum_is_half : prob_even_sum = 1 / 2 := by
  sorry

end probability_even_sum_is_half_l1560_156039


namespace sum_powers_of_ab_l1560_156052

theorem sum_powers_of_ab (a b : ℝ) (h1 : a + b = 3) (h2 : a * b = 1)
  (h3 : a^2 + b^2 = 7) (h4 : a^3 + b^3 = 18) (h5 : a^4 + b^4 = 47) :
  a^5 + b^5 = 123 :=
sorry

end sum_powers_of_ab_l1560_156052


namespace simplify_expression_l1560_156020

theorem simplify_expression : 9 * (12 / 7) * ((-35) / 36) = -15 := by
  sorry

end simplify_expression_l1560_156020


namespace fraction_of_cats_l1560_156024

theorem fraction_of_cats (C D : ℕ) 
  (h1 : C + D = 300)
  (h2 : 4 * D = 400) : 
  (C : ℚ) / (C + D) = 2 / 3 :=
by
  sorry

end fraction_of_cats_l1560_156024


namespace remaining_blocks_to_walk_l1560_156047

noncomputable def total_blocks : ℕ := 11 + 6 + 8
noncomputable def walked_blocks : ℕ := 5

theorem remaining_blocks_to_walk : total_blocks - walked_blocks = 20 := by
  sorry

end remaining_blocks_to_walk_l1560_156047


namespace quadratic_inequality_solution_set_l1560_156088

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 - x - 2 < 0} = {x : ℝ | -1 < x ∧ x < 2} :=
by
  sorry

end quadratic_inequality_solution_set_l1560_156088


namespace power_inequality_l1560_156092

theorem power_inequality 
( a b : ℝ )
( h1 : 0 < a )
( h2 : 0 < b )
( h3 : a ^ 1999 + b ^ 2000 ≥ a ^ 2000 + b ^ 2001 ) :
  a ^ 2000 + b ^ 2000 ≤ 2 :=
sorry

end power_inequality_l1560_156092


namespace scientific_notation_example_l1560_156018

theorem scientific_notation_example : 0.00001 = 1 * 10^(-5) :=
sorry

end scientific_notation_example_l1560_156018


namespace remaining_painting_time_l1560_156070

-- Define the conditions
def total_rooms : ℕ := 10
def hours_per_room : ℕ := 8
def rooms_painted : ℕ := 8

-- Define what we want to prove
theorem remaining_painting_time : (total_rooms - rooms_painted) * hours_per_room = 16 :=
by
  -- Here is where you would provide the proof
  sorry

end remaining_painting_time_l1560_156070


namespace range_of_a_l1560_156058

def p (x : ℝ) : Prop := 1 / 2 ≤ x ∧ x ≤ 1

def q (x a : ℝ) : Prop := (x - a) * (x - a - 1) ≤ 0

theorem range_of_a (a x : ℝ) 
  (hp : ∀ x, ¬ (1 / 2 ≤ x ∧ x ≤ 1) → (x < 1 / 2 ∨ x > 1))
  (hq : ∀ x, ¬ ((x - a) * (x - a - 1) ≤ 0) → (x < a ∨ x > a + 1))
  (h : ∀ x, (q x a) → (p x)) :
  0 ≤ a ∧ a ≤ 1 / 2 := 
sorry

end range_of_a_l1560_156058


namespace difference_of_squares_l1560_156037

theorem difference_of_squares (x y : ℝ) 
  (h1 : x + y = 20) 
  (h2 : x - y = 10) : 
  x^2 - y^2 = 200 := 
sorry

end difference_of_squares_l1560_156037


namespace jon_total_cost_l1560_156097
-- Import the complete Mathlib library

-- Define the conditions
def MSRP : ℝ := 30
def insurance_rate : ℝ := 0.20
def tax_rate : ℝ := 0.50

-- Calculate intermediate values based on conditions
noncomputable def insurance_cost : ℝ := insurance_rate * MSRP
noncomputable def subtotal_before_tax : ℝ := MSRP + insurance_cost
noncomputable def state_tax : ℝ := tax_rate * subtotal_before_tax
noncomputable def total_cost : ℝ := subtotal_before_tax + state_tax

-- The theorem we need to prove
theorem jon_total_cost : total_cost = 54 := by
  -- Proof is omitted
  sorry

end jon_total_cost_l1560_156097


namespace complement_U_A_l1560_156034

open Set

def U : Set ℝ := {x | -3 < x ∧ x < 3}
def A : Set ℝ := {x | -2 < x ∧ x ≤ 1}

theorem complement_U_A : 
  (U \ A) = {x | -3 < x ∧ x ≤ -2} ∪ {x | 1 < x ∧ x < 3} :=
by
  sorry

end complement_U_A_l1560_156034


namespace range_of_a_l1560_156099

open Set

def p (a x : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def q (x : ℝ) : Prop := x^2 + 2 * x - 8 > 0

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, p a x → q x) →
  ({ x : ℝ | p a x } ⊆ { x : ℝ | q x }) →
  a ≤ -4 ∨ a ≥ 2 ∨ a = 0 :=
by
  sorry

end range_of_a_l1560_156099


namespace matias_fewer_cards_l1560_156078

theorem matias_fewer_cards (J M C : ℕ) (h1 : J = M) (h2 : C = 20) (h3 : C + M + J = 48) : C - M = 6 :=
by
-- To be proven
  sorry

end matias_fewer_cards_l1560_156078


namespace race_time_l1560_156021

theorem race_time (v_A v_B : ℝ) (t_A t_B : ℝ) (h1 : v_A = 1000 / t_A) (h2 : v_B = 952 / (t_A + 6)) (h3 : v_A = v_B) : t_A = 125 :=
by
  sorry

end race_time_l1560_156021


namespace arithmetic_progression_sum_15_terms_l1560_156096

def arithmetic_progression_sum (a₁ d : ℚ) : ℚ :=
  15 * (2 * a₁ + (15 - 1) * d) / 2

def am_prog3_and_9_sum_and_product (a₁ d : ℚ) : Prop :=
  (a₁ + 2 * d) + (a₁ + 8 * d) = 6 ∧ (a₁ + 2 * d) * (a₁ + 8 * d) = 135 / 16

theorem arithmetic_progression_sum_15_terms (a₁ d : ℚ)
  (h : am_prog3_and_9_sum_and_product a₁ d) :
  arithmetic_progression_sum a₁ d = 37.5 ∨ arithmetic_progression_sum a₁ d = 52.5 :=
sorry

end arithmetic_progression_sum_15_terms_l1560_156096


namespace coordinates_of_vertex_B_equation_of_line_BC_l1560_156054

noncomputable def vertex_A : (ℝ × ℝ) := (5, 1)
def bisector_expr (x y : ℝ) : Prop := x + y - 5 = 0
def median_CM_expr (x y : ℝ) : Prop := 2 * x - y - 5 = 0

theorem coordinates_of_vertex_B (B : ℝ × ℝ) 
  (h1 : ∃ x y, bisector_expr x y ∧ median_CM_expr x y) :
  B = (2, 3) :=
sorry

theorem equation_of_line_BC (coeff_3x coeff_2y const : ℝ) 
  (h1 : ∃ x y, bisector_expr x y ∧ median_CM_expr x y) :
  coeff_3x = 3 ∧ coeff_2y = 2 ∧ const = -12 :=
sorry

end coordinates_of_vertex_B_equation_of_line_BC_l1560_156054


namespace area_increase_l1560_156029

theorem area_increase (a : ℝ) : ((a + 2) ^ 2 - a ^ 2 = 4 * a + 4) := by
  sorry

end area_increase_l1560_156029


namespace no_prime_pairs_sum_53_l1560_156001

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem no_prime_pairs_sum_53 :
  ¬ ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p + q = 53 :=
by
  sorry

end no_prime_pairs_sum_53_l1560_156001


namespace overall_percentage_increase_correct_l1560_156032

def initial_salary : ℕ := 60
def first_raise_salary : ℕ := 90
def second_raise_salary : ℕ := 120
def gym_deduction : ℕ := 10

def final_salary : ℕ := second_raise_salary - gym_deduction
def salary_difference : ℕ := final_salary - initial_salary
def percentage_increase : ℚ := (salary_difference : ℚ) / initial_salary * 100

theorem overall_percentage_increase_correct :
  percentage_increase = 83.33 := by
  sorry

end overall_percentage_increase_correct_l1560_156032


namespace smallest_positive_period_of_f_range_of_a_l1560_156055

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) + Real.cos (2 * x)

theorem smallest_positive_period_of_f : (∃ T > 0, ∀ x, f (x + T) = f x) ∧ (T = π) :=
by
  sorry

theorem range_of_a (a : ℝ) : (∀ x, f x ≤ a) → a ≥ Real.sqrt 2 :=
by
  sorry

end smallest_positive_period_of_f_range_of_a_l1560_156055


namespace basketball_campers_l1560_156014

theorem basketball_campers (total_campers soccer_campers football_campers : ℕ)
  (h_total : total_campers = 88)
  (h_soccer : soccer_campers = 32)
  (h_football : football_campers = 32) :
  total_campers - soccer_campers - football_campers = 24 :=
by
  sorry

end basketball_campers_l1560_156014


namespace cells_at_end_of_9th_day_l1560_156067

def initial_cells : ℕ := 4
def split_ratio : ℕ := 3
def total_days : ℕ := 9
def days_per_split : ℕ := 3

def num_terms : ℕ := total_days / days_per_split

noncomputable def number_of_cells (initial_cells split_ratio num_terms : ℕ) : ℕ :=
  initial_cells * split_ratio ^ (num_terms - 1)

theorem cells_at_end_of_9th_day :
  number_of_cells initial_cells split_ratio num_terms = 36 :=
by
  sorry

end cells_at_end_of_9th_day_l1560_156067


namespace findDivisor_l1560_156065

def addDivisorProblem : Prop :=
  ∃ d : ℕ, ∃ n : ℕ, n = 172835 + 21 ∧ d ∣ n ∧ d = 21

theorem findDivisor : addDivisorProblem :=
by
  sorry

end findDivisor_l1560_156065


namespace campers_went_rowing_and_hiking_in_all_l1560_156038

def C_rm : Nat := 41
def C_hm : Nat := 4
def C_ra : Nat := 26

theorem campers_went_rowing_and_hiking_in_all : (C_rm + C_ra) + C_hm = 71 :=
by
  sorry

end campers_went_rowing_and_hiking_in_all_l1560_156038


namespace geometric_sum_first_six_terms_l1560_156060

variable (a_n : ℕ → ℝ)

axiom geometric_seq (r a1 : ℝ) : ∀ n, a_n n = a1 * r ^ (n - 1)
axiom a2_val : a_n 2 = 2
axiom a5_val : a_n 5 = 16

theorem geometric_sum_first_six_terms (S6 : ℝ) : S6 = 1 * (1 - 2^6) / (1 - 2) := by
  sorry

end geometric_sum_first_six_terms_l1560_156060


namespace distribution_count_l1560_156026

-- Making the function for counting the number of valid distributions
noncomputable def countValidDistributions : ℕ :=
  let cases1 := 4                            -- One box contains all five balls
  let cases2 := 4 * 3                        -- One box has 4 balls, another has 1
  let cases3 := 4 * 3                        -- One box has 3 balls, another has 2
  let cases4 := 6 * 2                        -- Two boxes have 2 balls, and one has 1
  let cases5 := 4 * 3                        -- One box has 3 balls, and two boxes have 1 each
  cases1 + cases2 + cases3 + cases4 + cases5 -- Sum of all cases

-- Theorem statement: the count of valid distributions equals 52
theorem distribution_count : countValidDistributions = 52 := 
  by
    sorry

end distribution_count_l1560_156026


namespace books_in_series_l1560_156008

theorem books_in_series (books_watched : ℕ) (movies_watched : ℕ) (read_more_movies_than_books : books_watched + 3 = movies_watched) (watched_movies : movies_watched = 19) : books_watched = 16 :=
by sorry

end books_in_series_l1560_156008


namespace george_boxes_l1560_156094

-- Define the problem conditions and the question's expected outcome.
def total_blocks : ℕ := 12
def blocks_per_box : ℕ := 6
def expected_num_boxes : ℕ := 2

-- The proof statement that needs to be proved: George has the expected number of boxes.
theorem george_boxes : total_blocks / blocks_per_box = expected_num_boxes := 
  sorry

end george_boxes_l1560_156094


namespace euler_conjecture_counter_example_l1560_156061

theorem euler_conjecture_counter_example :
  ∃ (n : ℕ), 133^5 + 110^5 + 84^5 + 27^5 = n^5 ∧ n = 144 :=
by
  sorry

end euler_conjecture_counter_example_l1560_156061


namespace line_circle_intersect_a_le_0_l1560_156016

theorem line_circle_intersect_a_le_0 :
  (∃ (x y : ℝ), x + a * y + 2 = 0 ∧ x^2 + y^2 + 2 * x - 2 * y + 1 = 0) →
  a ≤ 0 :=
sorry

end line_circle_intersect_a_le_0_l1560_156016


namespace solve_player_coins_l1560_156017

def player_coins (n m k: ℕ) : Prop :=
  ∃ k, 
  (m = k * (n - 1) + 50) ∧ 
  (3 * m = 7 * n * k - 3 * k + 74) ∧ 
  (m = 69)

theorem solve_player_coins (n m k : ℕ) : player_coins n m k :=
by {
  sorry
}

end solve_player_coins_l1560_156017


namespace intersection_P_Q_l1560_156053

open Set

noncomputable def P : Set ℝ := {x | abs (x - 1) < 4}
noncomputable def Q : Set ℝ := {x | ∃ y, y = Real.log (x + 2) }

theorem intersection_P_Q :
  (P ∩ Q) = {x : ℝ | -2 < x ∧ x < 5} :=
by
  sorry

end intersection_P_Q_l1560_156053


namespace square_sum_l1560_156004

theorem square_sum (a b : ℝ) (h1 : a + b = 8) (h2 : a * b = -2) : a^2 + b^2 = 68 := 
by 
  sorry

end square_sum_l1560_156004


namespace decagon_triangle_probability_l1560_156005

theorem decagon_triangle_probability : 
  let total_vertices := 10
  let total_triangles := Nat.choose total_vertices 3
  let favorable_triangles := 10
  (total_triangles > 0) → 
  (favorable_triangles / total_triangles : ℚ) = 1 / 12 :=
by
  sorry

end decagon_triangle_probability_l1560_156005


namespace find_a1_l1560_156081

theorem find_a1 (a b : ℕ → ℝ) (h1 : ∀ n ≥ 1, a (n + 1) + b (n + 1) = (a n + b n) / 2) 
  (h2 : ∀ n ≥ 1, a (n + 1) * b (n + 1) = (a n * b n) ^ (1/2)) 
  (hb2016 : b 2016 = 1) (ha1_pos : a 1 > 0) :
  a 1 = 2^2015 :=
sorry

end find_a1_l1560_156081


namespace find_m_value_l1560_156033

theorem find_m_value :
  ∃ m : ℤ, 3 * 2^2000 - 5 * 2^1999 + 4 * 2^1998 - 2^1997 = m * 2^1997 ∧ m = 11 :=
by
  -- The proof would follow here.
  sorry

end find_m_value_l1560_156033


namespace income_growth_relation_l1560_156068

-- Define all the conditions
def initial_income : ℝ := 1.3
def third_week_income : ℝ := 2
def growth_rate (x : ℝ) : ℝ := (1 + x)^2  -- Compound interest style growth over 2 weeks.

-- Theorem: proving the relationship given the conditions
theorem income_growth_relation (x : ℝ) : initial_income * growth_rate x = third_week_income :=
by
  unfold initial_income third_week_income growth_rate
  sorry  -- Proof not required.

end income_growth_relation_l1560_156068


namespace collinear_points_l1560_156011

axiom collinear (A B C : ℝ × ℝ × ℝ) : Prop

theorem collinear_points (c d : ℝ) (h : collinear (2, c, d) (c, 3, d) (c, d, 4)) : c + d = 6 :=
sorry

end collinear_points_l1560_156011


namespace calc_f_2005_2007_zero_l1560_156048

variable {R : Type} [LinearOrderedField R]

def odd_function (f : R → R) : Prop :=
  ∀ x, f (-x) = -f x

def periodic_function (f : R → R) (p : R) : Prop :=
  ∀ x, f (x + p) = f x

theorem calc_f_2005_2007_zero
  {f : ℝ → ℝ}
  (h_odd : odd_function f)
  (h_period : periodic_function f 4) :
  f 2005 + f 2006 + f 2007 = 0 :=
sorry

end calc_f_2005_2007_zero_l1560_156048


namespace liquid_X_percentage_in_new_solution_l1560_156042

noncomputable def solutionY_initial_kg : ℝ := 10
noncomputable def percentage_liquid_X : ℝ := 0.30
noncomputable def evaporated_water_kg : ℝ := 2
noncomputable def added_solutionY_kg : ℝ := 2

-- Calculate the amount of liquid X in the original solution
noncomputable def initial_liquid_X_kg : ℝ :=
  percentage_liquid_X * solutionY_initial_kg

-- Calculate the remaining weight after evaporation
noncomputable def remaining_weight_kg : ℝ :=
  solutionY_initial_kg - evaporated_water_kg

-- Calculate the amount of liquid X after evaporation
noncomputable def remaining_liquid_X_kg : ℝ := initial_liquid_X_kg

-- Since only water evaporates, remaining water weight
noncomputable def remaining_water_kg : ℝ :=
  remaining_weight_kg - remaining_liquid_X_kg

-- Calculate the amount of liquid X in the added solution
noncomputable def added_liquid_X_kg : ℝ :=
  percentage_liquid_X * added_solutionY_kg

-- Total liquid X in the new solution
noncomputable def new_liquid_X_kg : ℝ :=
  remaining_liquid_X_kg + added_liquid_X_kg

-- Calculate the water in the added solution
noncomputable def percentage_water : ℝ := 0.70
noncomputable def added_water_kg : ℝ :=
  percentage_water * added_solutionY_kg

-- Total water in the new solution
noncomputable def new_water_kg : ℝ :=
  remaining_water_kg + added_water_kg

-- Total weight of the new solution
noncomputable def new_total_weight_kg : ℝ :=
  remaining_weight_kg + added_solutionY_kg

-- Percentage of liquid X in the new solution
noncomputable def percentage_new_liquid_X : ℝ :=
  (new_liquid_X_kg / new_total_weight_kg) * 100

-- The proof statement
theorem liquid_X_percentage_in_new_solution :
  percentage_new_liquid_X = 36 :=
by
  sorry

end liquid_X_percentage_in_new_solution_l1560_156042


namespace problem_l1560_156012

theorem problem (a b c : ℝ) (h1 : a > b) (h2 : (a - b) * (b - c) * (c - a) > 0) : a > c :=
sorry

end problem_l1560_156012


namespace find_constants_for_matrix_condition_l1560_156066

noncomputable section

def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![1, 2, 3], ![0, 1, 2], ![1, 0, 1]]

def I : Matrix (Fin 3) (Fin 3) ℝ :=
  1

theorem find_constants_for_matrix_condition :
  ∃ p q r : ℝ, B^3 + p • B^2 + q • B + r • I = 0 :=
by
  use -5, 3, -6
  sorry

end find_constants_for_matrix_condition_l1560_156066


namespace ellipse_foci_x_axis_l1560_156086

theorem ellipse_foci_x_axis (k : ℝ) : 
  (0 < k ∧ k < 2) ↔ (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ ∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1) ∧ a > b) := 
sorry

end ellipse_foci_x_axis_l1560_156086


namespace coyote_time_lemma_l1560_156062

theorem coyote_time_lemma (coyote_speed darrel_speed : ℝ) (catch_up_time t : ℝ) 
  (h1 : coyote_speed = 15) (h2 : darrel_speed = 30) (h3 : catch_up_time = 1) (h4 : darrel_speed * catch_up_time = coyote_speed * t) :
  t = 2 :=
by
  sorry

end coyote_time_lemma_l1560_156062


namespace initial_percentage_of_jasmine_water_l1560_156030

-- Definitions
def v_initial : ℝ := 80
def v_jasmine_added : ℝ := 8
def v_water_added : ℝ := 12
def percentage_final : ℝ := 16
def v_final : ℝ := v_initial + v_jasmine_added + v_water_added

-- Lean 4 statement that frames the proof problem
theorem initial_percentage_of_jasmine_water (P : ℝ) :
  (P / 100) * v_initial + v_jasmine_added = (percentage_final / 100) * v_final → P = 10 :=
by
  intro h
  sorry

end initial_percentage_of_jasmine_water_l1560_156030


namespace area_first_side_l1560_156023

-- Define dimensions of the box
variables (L W H : ℝ)

-- Define conditions
def area_WH : Prop := W * H = 72
def area_LH : Prop := L * H = 60
def volume_box : Prop := L * W * H = 720

-- Prove the area of the first side
theorem area_first_side (h1 : area_WH W H) (h2 : area_LH L H) (h3 : volume_box L W H) : L * W = 120 :=
by sorry

end area_first_side_l1560_156023


namespace meena_cookies_left_l1560_156075

-- Define the given conditions in terms of Lean definitions
def total_cookies_baked := 5 * 12
def cookies_sold_to_stone := 2 * 12
def cookies_bought_by_brock := 7
def cookies_bought_by_katy := 2 * cookies_bought_by_brock

-- Define the total cookies sold
def total_cookies_sold := cookies_sold_to_stone + cookies_bought_by_brock + cookies_bought_by_katy

-- Define the number of cookies left
def cookies_left := total_cookies_baked - total_cookies_sold

-- Prove that the number of cookies left is 15
theorem meena_cookies_left : cookies_left = 15 := by
  -- The proof is omitted (sorry is used to skip proof)
  sorry

end meena_cookies_left_l1560_156075


namespace correct_choice_l1560_156025

def PropA : Prop := ∀ x : ℝ, x^2 + 3 < 0
def PropB : Prop := ∀ x : ℕ, x^2 ≥ 1
def PropC : Prop := ∃ x : ℤ, x^5 < 1
def PropD : Prop := ∃ x : ℚ, x^2 = 3

theorem correct_choice : ¬PropA ∧ ¬PropB ∧ PropC ∧ ¬PropD := by
  sorry

end correct_choice_l1560_156025


namespace square_tiles_count_l1560_156082

theorem square_tiles_count 
  (h s : ℕ)
  (total_tiles : h + s = 30)
  (total_edges : 6 * h + 4 * s = 128) : 
  s = 26 :=
by
  sorry

end square_tiles_count_l1560_156082


namespace fraction_of_quarters_in_1790s_l1560_156049

theorem fraction_of_quarters_in_1790s (total_coins : ℕ) (coins_in_1790s : ℕ) :
  total_coins = 30 ∧ coins_in_1790s = 7 → 
  (coins_in_1790s : ℚ) / total_coins = 7 / 30 :=
by
  sorry

end fraction_of_quarters_in_1790s_l1560_156049


namespace porche_project_time_l1560_156013

theorem porche_project_time :
  let total_time := 180
  let math_time := 45
  let english_time := 30
  let science_time := 50
  let history_time := 25
  let homework_time := math_time + english_time + science_time + history_time 
  total_time - homework_time = 30 :=
by
  sorry

end porche_project_time_l1560_156013


namespace mean_of_second_set_l1560_156043

def mean (l: List ℕ) : ℚ :=
  (l.sum: ℚ) / l.length

theorem mean_of_second_set (x: ℕ) 
  (h: mean [28, x, 42, 78, 104] = 90): 
  mean [128, 255, 511, 1023, x] = 423 :=
by
  sorry

end mean_of_second_set_l1560_156043


namespace sum_modulo_seven_l1560_156028

theorem sum_modulo_seven :
  let s := 2 + 33 + 444 + 5555 + 66666 + 777777 + 8888888 + 99999999
  s % 7 = 2 :=
by
  sorry

end sum_modulo_seven_l1560_156028


namespace div_by_72_l1560_156084

theorem div_by_72 (x : ℕ) (y : ℕ) (h1 : 0 ≤ x ∧ x ≤ 9) (h2 : x = 4)
    (h3 : 0 ≤ y ∧ y ≤ 9) (h4 : y = 6) : 
    72 ∣ (9834800 + 1000 * x + 10 * y) :=
by 
  sorry

end div_by_72_l1560_156084


namespace distinct_arrangements_balloon_l1560_156074

-- Let's define the basic conditions:
def total_letters : Nat := 7
def repeats_l : Nat := 2
def repeats_o : Nat := 2

-- Now let's state the problem.
theorem distinct_arrangements_balloon : 
  (Nat.factorial total_letters) / ((Nat.factorial repeats_l) * (Nat.factorial repeats_o)) = 1260 := 
by
  sorry

end distinct_arrangements_balloon_l1560_156074


namespace solve_trig_problem_l1560_156007

noncomputable def trig_problem (α : ℝ) : Prop :=
  α ∈ (Set.Ioo 0 (Real.pi / 2)) ∪ Set.Ioo (Real.pi / 2) Real.pi ∧
  ∃ r : ℝ, r ≠ 0 ∧ Real.sin α * r = Real.sin (2 * α) ∧ Real.sin (2 * α) * r = Real.sin (4 * α)

theorem solve_trig_problem (α : ℝ) (h : trig_problem α) : α = 2 * Real.pi / 3 :=
by
  sorry

end solve_trig_problem_l1560_156007


namespace simplify_fraction_l1560_156041

theorem simplify_fraction (h1 : 48 = 2^4 * 3) (h2 : 72 = 2^3 * 3^2) : (48 / 72 : ℚ) = 2 / 3 := 
by
  sorry

end simplify_fraction_l1560_156041

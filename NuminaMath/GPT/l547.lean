import Mathlib

namespace NUMINAMATH_GPT_expected_value_of_unfair_die_l547_54765

noncomputable def seven_sided_die_expected_value : ℝ :=
  let p7 := 1 / 3
  let p_other := (2 / 3) / 6
  ((1 + 2 + 3 + 4 + 5 + 6) * p_other + 7 * p7)

theorem expected_value_of_unfair_die :
  seven_sided_die_expected_value = 14 / 3 :=
by
  sorry

end NUMINAMATH_GPT_expected_value_of_unfair_die_l547_54765


namespace NUMINAMATH_GPT_marbles_problem_l547_54707

theorem marbles_problem (h_total: ℕ) (h_each: ℕ) (h_total_eq: h_total = 35) (h_each_eq: h_each = 7) :
    h_total / h_each = 5 := by
  sorry

end NUMINAMATH_GPT_marbles_problem_l547_54707


namespace NUMINAMATH_GPT_complex_number_solution_l547_54772

theorem complex_number_solution (z : ℂ) (i : ℂ) (h_i : i * i = -1) (h : z + z * i = 1 + 5 * i) : z = 3 + 2 * i :=
sorry

end NUMINAMATH_GPT_complex_number_solution_l547_54772


namespace NUMINAMATH_GPT_sulfuric_acid_percentage_l547_54726

theorem sulfuric_acid_percentage 
  (total_volume : ℝ)
  (first_solution_percentage : ℝ)
  (final_solution_percentage : ℝ)
  (second_solution_volume : ℝ)
  (expected_second_solution_percentage : ℝ) :
  total_volume = 60 ∧
  first_solution_percentage = 0.02 ∧
  final_solution_percentage = 0.05 ∧
  second_solution_volume = 18 →
  expected_second_solution_percentage = 12 :=
by
  sorry

end NUMINAMATH_GPT_sulfuric_acid_percentage_l547_54726


namespace NUMINAMATH_GPT_roses_remain_unchanged_l547_54741

variable (initial_roses : ℕ) (initial_orchids : ℕ) (final_orchids : ℕ)

def unchanged_roses (roses_now : ℕ) : Prop :=
  roses_now = initial_roses

theorem roses_remain_unchanged :
  initial_roses = 13 → 
  initial_orchids = 84 → 
  final_orchids = 91 →
  ∀ (roses_now : ℕ), unchanged_roses initial_roses roses_now :=
by
  intros _ _ _ _
  simp [unchanged_roses]
  sorry

end NUMINAMATH_GPT_roses_remain_unchanged_l547_54741


namespace NUMINAMATH_GPT_solution_of_two_quadratics_l547_54776

theorem solution_of_two_quadratics (x : ℝ) (h1 : 8 * x^2 + 7 * x - 1 = 0) (h2 : 24 * x^2 + 53 * x - 7 = 0) : x = 1 / 8 := 
by 
  sorry

end NUMINAMATH_GPT_solution_of_two_quadratics_l547_54776


namespace NUMINAMATH_GPT_hyperbola_equation_l547_54709

open Real

-- Define the conditions in Lean
def is_hyperbola_form (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

def is_positive (x : ℝ) : Prop := x > 0

def parabola_focus : (ℝ × ℝ) := (1, 0)

def hyperbola_vertex_eq_focus (a : ℝ) : Prop := a = parabola_focus.1

def hyperbola_eccentricity (e a c : ℝ) : Prop := e = c / a

-- Our proof statement
theorem hyperbola_equation :
  ∃ (a b : ℝ), is_positive a ∧ is_positive b ∧
  hyperbola_vertex_eq_focus a ∧
  hyperbola_eccentricity (sqrt 5) a (sqrt 5) ∧
  is_hyperbola_form a b 1 0 :=
by sorry

end NUMINAMATH_GPT_hyperbola_equation_l547_54709


namespace NUMINAMATH_GPT_average_of_three_numbers_is_78_l547_54718

theorem average_of_three_numbers_is_78 (x y z : ℕ) (h1 : z = 2 * y) (h2 : y = 4 * x) (h3 : x = 18) :
  (x + y + z) / 3 = 78 :=
by sorry

end NUMINAMATH_GPT_average_of_three_numbers_is_78_l547_54718


namespace NUMINAMATH_GPT_train_actual_speed_l547_54783
-- Import necessary libraries

-- Define the given conditions and question
def departs_time := 6
def planned_speed := 100
def scheduled_arrival_time := 18
def actual_arrival_time := 16
def distance (t₁ t₂ : ℕ) (s : ℕ) : ℕ := s * (t₂ - t₁)
def actual_speed (d t₁ t₂ : ℕ) : ℕ := d / (t₂ - t₁)

-- Proof problem statement
theorem train_actual_speed:
  actual_speed (distance departs_time scheduled_arrival_time planned_speed) departs_time actual_arrival_time = 120 := by sorry

end NUMINAMATH_GPT_train_actual_speed_l547_54783


namespace NUMINAMATH_GPT_reading_club_coordinator_selection_l547_54782

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem reading_club_coordinator_selection :
  let total_ways := choose 18 4
  let no_former_ways := choose 10 4
  total_ways - no_former_ways = 2850 := by
  sorry

end NUMINAMATH_GPT_reading_club_coordinator_selection_l547_54782


namespace NUMINAMATH_GPT_parabola_x0_range_l547_54722

variables {x₀ y₀ : ℝ}
def parabola (x₀ y₀ : ℝ) : Prop := y₀^2 = 8 * x₀

def focus (x : ℝ) : ℝ := 2

def directrix (x : ℝ) : Prop := x = -2

/-- Prove that for any point (x₀, y₀) on the parabola y² = 8x and 
if a circle centered at the focus intersects the directrix, then x₀ > 2. -/
theorem parabola_x0_range (x₀ y₀ : ℝ) (h1 : parabola x₀ y₀)
  (h2 : ((x₀ - 2)^2 + y₀^2)^(1/2) > (2 : ℝ)) : x₀ > 2 := 
sorry

end NUMINAMATH_GPT_parabola_x0_range_l547_54722


namespace NUMINAMATH_GPT_math_problem_l547_54781

open Set

noncomputable def A : Set ℝ := { x | x < 1 }
noncomputable def B : Set ℝ := { x | x * (x - 1) > 6 }
noncomputable def C (m : ℝ) : Set ℝ := { x | -1 + m < x ∧ x < 2 * m }

theorem math_problem (m : ℝ) (m_range : C m ≠ ∅) :
  (A ∪ B = { x | x > 3 ∨ x < 1 }) ∧
  (A ∩ (compl B) = { x | -2 ≤ x ∧ x < 1 }) ∧
  (-1 < m ∧ m ≤ 0.5) :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l547_54781


namespace NUMINAMATH_GPT_odd_function_property_l547_54798

noncomputable def odd_function := {f : ℝ → ℝ // ∀ x : ℝ, f (-x) = -f x}

theorem odd_function_property (f : odd_function) (h1 : f.1 1 = -2) : f.1 (-1) + f.1 0 = 2 := by
  sorry

end NUMINAMATH_GPT_odd_function_property_l547_54798


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l547_54797

theorem sufficient_but_not_necessary (x : ℝ) (h : 1 / x < 1 / 2) : x > 2 ∨ x < 0 :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l547_54797


namespace NUMINAMATH_GPT_pizza_varieties_l547_54724

-- Definition of the problem conditions
def base_flavors : ℕ := 4
def topping_options : ℕ := 4  -- No toppings, extra cheese, mushrooms, both

-- The math proof problem statement
theorem pizza_varieties : base_flavors * topping_options = 16 := by 
  sorry

end NUMINAMATH_GPT_pizza_varieties_l547_54724


namespace NUMINAMATH_GPT_range_of_a_l547_54736

theorem range_of_a (a b : ℝ) (h1 : 0 ≤ a - b ∧ a - b ≤ 1) (h2 : 1 ≤ a + b ∧ a + b ≤ 4) : 
  1 / 2 ≤ a ∧ a ≤ 5 / 2 := 
sorry

end NUMINAMATH_GPT_range_of_a_l547_54736


namespace NUMINAMATH_GPT_range_of_x_in_tight_sequence_arithmetic_tight_sequence_geometric_tight_sequence_l547_54785

-- Problem (1)
theorem range_of_x_in_tight_sequence (a : ℕ → ℝ) (x : ℝ) (h : ∀ n : ℕ, 1 / 2 ≤ a (n + 1) / a n ∧ a (n + 1) / a n ≤ 2) :
  a 1 = 1 ∧ a 2 = 3 / 2 ∧ a 3 = x ∧ a 4 = 4 → 2 ≤ x ∧ x ≤ 3 :=
sorry

-- Problem (2)
theorem arithmetic_tight_sequence (a : ℕ → ℝ) (a1 d : ℝ) (h : ∀ n : ℕ, 1 / 2 ≤ a (n + 1) / a n ∧ a (n + 1) / a n ≤ 2) :
  ∀ n : ℕ, a n = a1 + ↑n * d → 0 < d ∧ d ≤ a1 →
  ∀ n : ℕ, 1 / 2 ≤ (a (n + 1) / a n) ∧ (a (n + 1) / a n) ≤ 2 :=
sorry

-- Problem (3)
theorem geometric_tight_sequence (a : ℕ → ℝ) (q : ℝ) (a1 : ℝ) (h_seq : ∀ n : ℕ, 1 / 2 ≤ a (n + 1) / a n ∧ a (n + 1) / a n ≤ 2)
(S : ℕ → ℝ) (h_sum_seq : ∀ n : ℕ, 1 / 2 ≤ S (n + 1) / S n ∧ S (n + 1) / S n ≤ 2) :
  (∀ n : ℕ, a n = a1 * q ^ n ∧ S n = a1 * (1 - q ^ (n + 1)) / (1 - q)) → 
  1 / 2 ≤ q ∧ q ≤ 1 :=
sorry

end NUMINAMATH_GPT_range_of_x_in_tight_sequence_arithmetic_tight_sequence_geometric_tight_sequence_l547_54785


namespace NUMINAMATH_GPT_sweets_ratio_l547_54799

theorem sweets_ratio (number_orange_sweets : ℕ) (number_grape_sweets : ℕ) (max_sweets_per_tray : ℕ)
  (h1 : number_orange_sweets = 36) (h2 : number_grape_sweets = 44) (h3 : max_sweets_per_tray = 4) :
  (number_orange_sweets / max_sweets_per_tray) / (number_grape_sweets / max_sweets_per_tray) = 9 / 11 :=
by
  sorry

end NUMINAMATH_GPT_sweets_ratio_l547_54799


namespace NUMINAMATH_GPT_probability_neither_red_nor_purple_l547_54720

theorem probability_neither_red_nor_purple
  (total_balls : ℕ)
  (white_balls : ℕ)
  (green_balls : ℕ)
  (yellow_balls : ℕ)
  (red_balls : ℕ)
  (purple_balls : ℕ)
  (h_total : total_balls = 60)
  (h_white : white_balls = 22)
  (h_green : green_balls = 18)
  (h_yellow : yellow_balls = 17)
  (h_red : red_balls = 3)
  (h_purple : purple_balls = 1) :
  ((total_balls - red_balls - purple_balls) / total_balls : ℚ) = 14 / 15 :=
by
  sorry

end NUMINAMATH_GPT_probability_neither_red_nor_purple_l547_54720


namespace NUMINAMATH_GPT_num_convex_numbers_without_repeats_l547_54796

def is_convex_number (a b c : ℕ) : Prop :=
  a < b ∧ b > c

def is_valid_digit (n : ℕ) : Prop :=
  0 ≤ n ∧ n < 10

def distinct_digits (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c

theorem num_convex_numbers_without_repeats : 
  (∃ (numbers : Finset (ℕ × ℕ × ℕ)), 
    (∀ a b c, (a, b, c) ∈ numbers -> is_convex_number a b c ∧ is_valid_digit a ∧ is_valid_digit b ∧ is_valid_digit c ∧ distinct_digits a b c) ∧
    numbers.card = 204) :=
sorry

end NUMINAMATH_GPT_num_convex_numbers_without_repeats_l547_54796


namespace NUMINAMATH_GPT_no_three_by_three_red_prob_l547_54703

theorem no_three_by_three_red_prob : 
  ∃ (m n : ℕ), 
  Nat.gcd m n = 1 ∧ 
  m / n = 340 / 341 ∧ 
  m + n = 681 :=
by
  sorry

end NUMINAMATH_GPT_no_three_by_three_red_prob_l547_54703


namespace NUMINAMATH_GPT_goldfish_feeding_l547_54775

theorem goldfish_feeding (g : ℕ) (h : g = 8) : 4 * g = 32 :=
by
  sorry

end NUMINAMATH_GPT_goldfish_feeding_l547_54775


namespace NUMINAMATH_GPT_minimize_AB_l547_54757

-- Definition of the circle C
def circleC (x y : ℝ) : Prop := x^2 + y^2 - 2 * y - 3 = 0

-- Definition of the point P
def P : ℝ × ℝ := (-1, 2)

-- Definition of the line l
def line_l (x y : ℝ) : Prop := x - y + 3 = 0

-- The goal is to prove that line_l is the line through P minimizing |AB|
theorem minimize_AB : 
  ∀ l : ℝ → ℝ → Prop, 
  (∀ x y, l x y → (∃ a b, circleC a b ∧ l a b ∧ circleC x y ∧ l x y ∧ (x ≠ a ∨ y ≠ b)) → False) 
  → l = line_l :=
by
  sorry

end NUMINAMATH_GPT_minimize_AB_l547_54757


namespace NUMINAMATH_GPT_find_2a_plus_6b_l547_54715

theorem find_2a_plus_6b (a b : ℕ) (n : ℕ)
  (h1 : 3 * a + 5 * b ≡ 19 [MOD n + 1])
  (h2 : 4 * a + 2 * b ≡ 25 [MOD n + 1])
  (hn : n = 96) :
  2 * a + 6 * b = 96 :=
by
  sorry

end NUMINAMATH_GPT_find_2a_plus_6b_l547_54715


namespace NUMINAMATH_GPT_a_b_work_days_l547_54790

-- Definitions:
def work_days_a_b_together := 40
def work_days_a_alone := 12
def remaining_work_days_with_a := 9

-- Statement to be proven:
theorem a_b_work_days (x : ℕ) 
  (h1 : ∀ W : ℕ, W / work_days_a_b_together + remaining_work_days_with_a * (W / work_days_a_alone) = W) :
  x = 10 :=
sorry

end NUMINAMATH_GPT_a_b_work_days_l547_54790


namespace NUMINAMATH_GPT_basketball_game_first_half_points_l547_54795

noncomputable def total_points_first_half
  (eagles_points : ℕ → ℕ) (lions_points : ℕ → ℕ) (common_ratio : ℕ) (common_difference : ℕ) : ℕ :=
  eagles_points 0 + eagles_points 1 + lions_points 0 + lions_points 1

theorem basketball_game_first_half_points 
  (eagles_points lions_points : ℕ → ℕ)
  (common_ratio : ℕ) (common_difference : ℕ)
  (h1 : eagles_points 0 = lions_points 0)
  (h2 : ∀ n, eagles_points (n + 1) = common_ratio * eagles_points n)
  (h3 : ∀ n, lions_points (n + 1) = lions_points n + common_difference)
  (h4 : eagles_points 0 + eagles_points 1 + eagles_points 2 + eagles_points 3 =
        lions_points 0 + lions_points 1 + lions_points 2 + lions_points 3 + 3)
  (h5 : eagles_points 0 + eagles_points 1 + eagles_points 2 + eagles_points 3 ≤ 120)
  (h6 : lions_points 0 + lions_points 1 + lions_points 2 + lions_points 3 ≤ 120) :
  total_points_first_half eagles_points lions_points common_ratio common_difference = 15 :=
sorry

end NUMINAMATH_GPT_basketball_game_first_half_points_l547_54795


namespace NUMINAMATH_GPT_find_k_unique_solution_l547_54730

theorem find_k_unique_solution :
  ∀ k : ℝ, (∀ x : ℝ, x ≠ 0 → (1/(3*x) = (k - x)/8) → (3*x^2 + (8 - 3*k)*x = 0)) →
    k = 8 / 3 :=
by
  intros k h
  -- Using sorry here to skip the proof
  sorry

end NUMINAMATH_GPT_find_k_unique_solution_l547_54730


namespace NUMINAMATH_GPT_probability_white_ball_second_draw_l547_54760

noncomputable def probability_white_given_red (red_white_yellow_balls : Nat × Nat × Nat) : ℚ :=
  let (r, w, y) := red_white_yellow_balls
  let total_balls := r + w + y
  let p_A := (r : ℚ) / total_balls
  let p_AB := (r : ℚ) / total_balls * (w : ℚ) / (total_balls - 1)
  p_AB / p_A

theorem probability_white_ball_second_draw (r w y : Nat) (h_r : r = 2) (h_w : w = 3) (h_y : y = 1) :
  probability_white_given_red (r, w, y) = 3 / 5 :=
by
  rw [h_r, h_w, h_y]
  unfold probability_white_given_red
  simp
  sorry

end NUMINAMATH_GPT_probability_white_ball_second_draw_l547_54760


namespace NUMINAMATH_GPT_second_number_is_72_l547_54769

-- Define the necessary variables and conditions
variables (x y : ℕ)
variables (h_first_num : x = 48)
variables (h_ratio : 48 / 8 = x / y)
variables (h_LCM : Nat.lcm x y = 432)

-- State the problem as a theorem
theorem second_number_is_72 : y = 72 :=
by
  sorry

end NUMINAMATH_GPT_second_number_is_72_l547_54769


namespace NUMINAMATH_GPT_swimmer_speeds_l547_54732

variable (a s r : ℝ)
variable (x z y : ℝ)

theorem swimmer_speeds (h : s < r) (h' : r < 100 * s / (50 + s)) :
    (100 * s - 50 * r - r * s) / ((3 * s - r) * a) = x ∧ 
    (100 * s - 50 * r - r * s) / ((r - s) * a) = z := by
    sorry

end NUMINAMATH_GPT_swimmer_speeds_l547_54732


namespace NUMINAMATH_GPT_number_is_43_l547_54762

theorem number_is_43 (m : ℕ) : (m > 30 ∧ m < 50) ∧ Nat.Prime m ∧ m % 12 = 7 ↔ m = 43 :=
by
  sorry

end NUMINAMATH_GPT_number_is_43_l547_54762


namespace NUMINAMATH_GPT_ping_pong_balls_sold_l547_54711

theorem ping_pong_balls_sold (total_baseballs initial_baseballs initial_pingpong total_baseballs_sold total_balls_left : ℕ)
  (h1 : total_baseballs = 2754)
  (h2 : initial_pingpong = 1938)
  (h3 : total_baseballs_sold = 1095)
  (h4 : total_balls_left = 3021) :
  initial_pingpong - (total_balls_left - (total_baseballs - total_baseballs_sold)) = 576 :=
by sorry

end NUMINAMATH_GPT_ping_pong_balls_sold_l547_54711


namespace NUMINAMATH_GPT_calculation_error_l547_54710

theorem calculation_error (x y : ℕ) : (25 * x + 5 * y) = 25 * x + 5 * y :=
by
  sorry

end NUMINAMATH_GPT_calculation_error_l547_54710


namespace NUMINAMATH_GPT_simplify_expression_l547_54738

theorem simplify_expression :
  (16 / 54) * (27 / 8) * (64 / 81) = 64 / 9 :=
by sorry

end NUMINAMATH_GPT_simplify_expression_l547_54738


namespace NUMINAMATH_GPT_plan_Y_cheaper_l547_54754

theorem plan_Y_cheaper (y : ℤ) :
  (15 * (y : ℚ) > 2500 + 8 * (y : ℚ)) ↔ y > 358 :=
by
  sorry

end NUMINAMATH_GPT_plan_Y_cheaper_l547_54754


namespace NUMINAMATH_GPT_smallest_y_value_l547_54734

-- Define the original equation
def original_eq (y : ℝ) := 3 * y^2 + 36 * y - 90 = y * (y + 18)

-- Define the problem statement
theorem smallest_y_value : ∃ (y : ℝ), original_eq y ∧ y = -15 :=
by
  sorry

end NUMINAMATH_GPT_smallest_y_value_l547_54734


namespace NUMINAMATH_GPT_max_sum_of_roots_l547_54714

theorem max_sum_of_roots (a b : ℝ) (h_a : a ≠ 0) (m : ℝ) :
  (∀ x : ℝ, (2 * x ^ 2 - 5 * x + m = 0) → 25 - 8 * m ≥ 0) →
  (∃ s, s = -5 / 2) → m = 25 / 8 :=
by
  sorry

end NUMINAMATH_GPT_max_sum_of_roots_l547_54714


namespace NUMINAMATH_GPT_Kenneth_money_left_l547_54759

def initial_amount : ℕ := 50
def number_of_baguettes : ℕ := 2
def cost_per_baguette : ℕ := 2
def number_of_bottles : ℕ := 2
def cost_per_bottle : ℕ := 1

-- This theorem states that Kenneth has $44 left after his purchases.
theorem Kenneth_money_left : initial_amount - (number_of_baguettes * cost_per_baguette + number_of_bottles * cost_per_bottle) = 44 := by
  sorry

end NUMINAMATH_GPT_Kenneth_money_left_l547_54759


namespace NUMINAMATH_GPT_tickets_sold_l547_54746

theorem tickets_sold (T : ℕ) (h1 : 3 * T / 4 > 0)
    (h2 : 5 * (T / 4) / 9 > 0)
    (h3 : 80 > 0)
    (h4 : 20 > 0) :
    (1 / 4 * T - 5 / 36 * T = 100) -> T = 900 :=
by
  sorry

end NUMINAMATH_GPT_tickets_sold_l547_54746


namespace NUMINAMATH_GPT_geometry_problem_l547_54793

noncomputable def vertices_on_hyperbola (A B C : ℝ × ℝ) : Prop :=
  (∃ x1 y1, A = (x1, y1) ∧ 2 * x1^2 - y1^2 = 4) ∧
  (∃ x2 y2, B = (x2, y2) ∧ 2 * x2^2 - y2^2 = 4) ∧
  (∃ x3 y3, C = (x3, y3) ∧ 2 * x3^2 - y3^2 = 4)

noncomputable def midpoints (A B C M N P : ℝ × ℝ) : Prop :=
  (M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) ∧
  (N = ((B.1 + C.1) / 2, (B.2 + C.2) / 2)) ∧
  (P = ((C.1 + A.1) / 2, (C.2 + A.2) / 2))

noncomputable def slopes (A B C M N P : ℝ × ℝ) (k1 k2 k3 : ℝ) : Prop :=
  k1 ≠ 0 ∧ k2 ≠ 0 ∧ k3 ≠ 0 ∧
  k1 = M.2 / M.1 ∧ k2 = N.2 / N.1 ∧ k3 = P.2 / P.1

noncomputable def sum_of_slopes (A B C : ℝ × ℝ) (k1 k2 k3 : ℝ) : Prop :=
  ((A.2 - B.2) / (A.1 - B.1) +
   (B.2 - C.2) / (B.1 - C.1) +
   (C.2 - A.2) / (C.1 - A.1)) = -1

theorem geometry_problem 
  (A B C M N P : ℝ × ℝ) (k1 k2 k3 : ℝ) 
  (h1 : vertices_on_hyperbola A B C)
  (h2 : midpoints A B C M N P) 
  (h3 : slopes A B C M N P k1 k2 k3) 
  (h4 : sum_of_slopes A B C k1 k2 k3) :
  1/k1 + 1/k2 + 1/k3 = -1 / 2 :=
sorry

end NUMINAMATH_GPT_geometry_problem_l547_54793


namespace NUMINAMATH_GPT_sum_interior_numbers_eighth_row_of_pascals_triangle_l547_54725

theorem sum_interior_numbers_eighth_row_of_pascals_triangle :
  let n := 8
  let sum_all_elements := 2 ^ (n - 1)
  let sum_interior_numbers := sum_all_elements - 2
  sum_interior_numbers = 126 :=
by
  let n := 8
  let sum_all_elements := 2 ^ (n - 1)
  let sum_interior_numbers := sum_all_elements - 2
  show sum_interior_numbers = 126
  sorry

end NUMINAMATH_GPT_sum_interior_numbers_eighth_row_of_pascals_triangle_l547_54725


namespace NUMINAMATH_GPT_tan_condition_then_expression_value_l547_54764

theorem tan_condition_then_expression_value (θ : ℝ) (h : Real.tan θ = 2) :
  (2 * Real.sin θ) / (Real.sin θ + 2 * Real.cos θ) = 1 :=
sorry

end NUMINAMATH_GPT_tan_condition_then_expression_value_l547_54764


namespace NUMINAMATH_GPT_eq_of_nonzero_real_x_l547_54749

theorem eq_of_nonzero_real_x (x : ℝ) (hx : x ≠ 0) (a b : ℝ) (ha : a = 9) (hb : b = 18) :
  ((a * x) ^ 10 = (b * x) ^ 5) → x = 2 / 9 :=
by
  sorry

end NUMINAMATH_GPT_eq_of_nonzero_real_x_l547_54749


namespace NUMINAMATH_GPT_calculate_selling_price_l547_54794

theorem calculate_selling_price (cost_price : ℝ) (loss_percentage : ℝ) (selling_price : ℝ) : 
  cost_price = 1500 → 
  loss_percentage = 0.17 →
  selling_price = cost_price - (loss_percentage * cost_price) →
  selling_price = 1245 :=
by 
  intros hc hl hs
  rw [hc, hl] at hs
  norm_num at hs
  exact hs

end NUMINAMATH_GPT_calculate_selling_price_l547_54794


namespace NUMINAMATH_GPT_product_pricing_and_savings_l547_54716

theorem product_pricing_and_savings :
  ∃ (x y : ℝ),
    (6 * x + 3 * y = 600) ∧
    (40 * x + 30 * y = 5200) ∧
    x = 40 ∧
    y = 120 ∧
    (80 * x + 100 * y - (80 * 0.8 * x + 100 * 0.75 * y) = 3640) := 
by
  sorry

end NUMINAMATH_GPT_product_pricing_and_savings_l547_54716


namespace NUMINAMATH_GPT_original_grain_amount_l547_54774

def grain_spilled : ℕ := 49952
def grain_remaining : ℕ := 918

theorem original_grain_amount : grain_spilled + grain_remaining = 50870 :=
by
  sorry

end NUMINAMATH_GPT_original_grain_amount_l547_54774


namespace NUMINAMATH_GPT_competition_results_l547_54786

namespace Competition

-- Define the probabilities for each game
def prob_win_game_A : ℚ := 2 / 3
def prob_win_game_B : ℚ := 1 / 2

-- Define the probability of winning each project (best of five format)
def prob_win_project_A : ℚ := (8 / 27) + (8 / 27) + (16 / 81)
def prob_win_project_B : ℚ := (1 / 8) + (3 / 16) + (3 / 16)

-- Define the distribution of the random variable X (number of projects won by player A)
def P_X_0 : ℚ := (17 / 81) * (1 / 2)
def P_X_2 : ℚ := (64 / 81) * (1 / 2)
def P_X_1 : ℚ := 1 - P_X_0 - P_X_2

-- Define the mathematical expectation of X
def E_X : ℚ := 0 * P_X_0 + 1 * P_X_1 + 2 * P_X_2

-- Theorem stating the results
theorem competition_results :
  prob_win_project_A = 64 / 81 ∧
  prob_win_project_B = 1 / 2 ∧
  P_X_0 = 17 / 162 ∧
  P_X_1 = 81 / 162 ∧
  P_X_2 = 64 / 162 ∧
  E_X = 209 / 162 :=
by sorry

end Competition

end NUMINAMATH_GPT_competition_results_l547_54786


namespace NUMINAMATH_GPT_fraction_pow_zero_l547_54702

theorem fraction_pow_zero (a b : ℤ) (hb_nonzero : b ≠ 0) : (a / (b : ℚ)) ^ 0 = 1 :=
by 
  sorry

end NUMINAMATH_GPT_fraction_pow_zero_l547_54702


namespace NUMINAMATH_GPT_limit_calculation_l547_54770

noncomputable def f (x : ℝ) : ℝ := Real.exp (-x)

theorem limit_calculation :
  (Real.exp (-1) * Real.exp 0 - Real.exp (-1) * Real.exp 0) / 0 = -3 / Real.exp 1 := by
  sorry

end NUMINAMATH_GPT_limit_calculation_l547_54770


namespace NUMINAMATH_GPT_volume_of_pyramid_l547_54735

-- Define the conditions
def pyramid_conditions : Prop :=
  ∃ (s h : ℝ),
  s^2 = 256 ∧
  ∃ (h_A h_C h_B : ℝ),
  ((∃ h_A, 128 = 1/2 * s * h_A) ∧
  (∃ h_C, 112 = 1/2 * s * h_C) ∧
  (∃ h_B, 96 = 1/2 * s * h_B)) ∧
  h^2 + (s/2)^2 = h_A^2 ∧
  h^2 = 256 - (s/2)^2 ∧
  h^2 + (s/4)^2 = h_B^2

-- Define the theorem
theorem volume_of_pyramid :
  pyramid_conditions → 
  ∃ V : ℝ, V = 682.67 * Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_volume_of_pyramid_l547_54735


namespace NUMINAMATH_GPT_fraction_not_simplifiable_l547_54771

theorem fraction_not_simplifiable (n : ℕ) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := 
sorry

end NUMINAMATH_GPT_fraction_not_simplifiable_l547_54771


namespace NUMINAMATH_GPT_OfficerHoppsTotalTickets_l547_54731

theorem OfficerHoppsTotalTickets : 
  (15 * 8 + (31 - 15) * 5 = 200) :=
  by
    sorry

end NUMINAMATH_GPT_OfficerHoppsTotalTickets_l547_54731


namespace NUMINAMATH_GPT_annual_interest_rate_of_second_investment_l547_54713

-- Definitions for the conditions
def total_income : ℝ := 575
def investment1 : ℝ := 3000
def rate1 : ℝ := 0.085
def income1 : ℝ := investment1 * rate1
def investment2 : ℝ := 5000
def target_income : ℝ := total_income - income1

-- Lean 4 statement to prove the annual simple interest rate of the second investment
theorem annual_interest_rate_of_second_investment : ∃ (r : ℝ), target_income = investment2 * (r / 100) ∧ r = 6.4 :=
by sorry

end NUMINAMATH_GPT_annual_interest_rate_of_second_investment_l547_54713


namespace NUMINAMATH_GPT_non_shaded_area_l547_54767

theorem non_shaded_area (s : ℝ) (hex_area : ℝ) (tri_area : ℝ) (non_shaded_area : ℝ) :
  s = 12 →
  hex_area = (3 * Real.sqrt 3 / 2) * s^2 →
  tri_area = (Real.sqrt 3 / 4) * (2 * s)^2 →
  non_shaded_area = hex_area - tri_area →
  non_shaded_area = 288 * Real.sqrt 3 :=
by
  intros hs hhex htri hnon
  sorry

end NUMINAMATH_GPT_non_shaded_area_l547_54767


namespace NUMINAMATH_GPT_friend_reading_time_l547_54791

-- Define the conditions
def my_reading_time : ℝ := 1.5 * 60 -- 1.5 hours converted to minutes
def friend_speed_multiplier : ℝ := 5 -- Friend reads 5 times faster than I do
def distraction_time : ℝ := 15 -- Friend is distracted for 15 minutes

-- Define the time taken for my friend to read the book accounting for distraction
theorem friend_reading_time :
  (my_reading_time / friend_speed_multiplier) + distraction_time = 33 := by
  sorry

end NUMINAMATH_GPT_friend_reading_time_l547_54791


namespace NUMINAMATH_GPT_probability_visible_l547_54755

-- Definitions of the conditions
def lap_time_sarah : ℕ := 120
def lap_time_sam : ℕ := 100
def start_to_photo_min : ℕ := 15
def start_to_photo_max : ℕ := 16
def photo_fraction : ℚ := 1/3
def shadow_start_interval : ℕ := 45
def shadow_duration : ℕ := 15

-- The theorem to prove
theorem probability_visible :
  let total_time := 60
  let valid_overlap_time := 13.33
  valid_overlap_time / total_time = 1333 / 6000 :=
by {
  sorry
}

end NUMINAMATH_GPT_probability_visible_l547_54755


namespace NUMINAMATH_GPT_camilla_blueberry_jelly_beans_l547_54777

theorem camilla_blueberry_jelly_beans (b c : ℕ) 
  (h1 : b = 3 * c)
  (h2 : b - 20 = 2 * (c - 5)) : 
  b = 30 := 
sorry

end NUMINAMATH_GPT_camilla_blueberry_jelly_beans_l547_54777


namespace NUMINAMATH_GPT_min_value_a_p_a_q_l547_54721

theorem min_value_a_p_a_q (a : ℕ → ℕ) (p q : ℕ) (h_arith_geom : ∀ n, a (n + 2) = a (n + 1) + a n * 2)
(h_a9 : a 9 = a 8 + 2 * a 7)
(h_ap_aq : a p * a q = 8 * a 1 ^ 2) :
    (1 / p : ℝ) + (4 / q : ℝ) = 9 / 5 := by
    sorry

end NUMINAMATH_GPT_min_value_a_p_a_q_l547_54721


namespace NUMINAMATH_GPT_compute_t_minus_s_l547_54763

noncomputable def t : ℚ := (40 + 30 + 30 + 20) / 4

noncomputable def s : ℚ := (40 * (40 / 120) + 30 * (30 / 120) + 30 * (30 / 120) + 20 * (20 / 120))

theorem compute_t_minus_s : t - s = -1.67 := by
  sorry

end NUMINAMATH_GPT_compute_t_minus_s_l547_54763


namespace NUMINAMATH_GPT_integer_pairs_satisfying_condition_l547_54733

theorem integer_pairs_satisfying_condition :
  { (m, n) : ℤ × ℤ | ∃ k : ℤ, (n^3 + 1) = k * (m * n - 1) } =
  { (1, 2), (1, 3), (2, 1), (3, 1), (2, 5), (3, 5), (5, 2), (5, 3), (2, 2) } :=
sorry

end NUMINAMATH_GPT_integer_pairs_satisfying_condition_l547_54733


namespace NUMINAMATH_GPT_negation_of_at_most_one_obtuse_is_at_least_two_obtuse_l547_54780

-- Definition of a triangle with a property on the angles.
def triangle (a b c : ℝ) : Prop := a + b + c = 180 ∧ 0 < a ∧ 0 < b ∧ 0 < c

-- Definition of an obtuse angle.
def obtuse (x : ℝ) : Prop := x > 90

-- Proposition: In a triangle, at most one angle is obtuse.
def at_most_one_obtuse (a b c : ℝ) : Prop := 
  triangle a b c ∧ (obtuse a → ¬ obtuse b ∧ ¬ obtuse c) ∧ (obtuse b → ¬ obtuse a ∧ ¬ obtuse c) ∧ (obtuse c → ¬ obtuse a ∧ ¬ obtuse b)

-- Negation: In a triangle, there are at least two obtuse angles.
def at_least_two_obtuse (a b c : ℝ) : Prop := 
  triangle a b c ∧ (obtuse a ∧ obtuse b) ∨ (obtuse a ∧ obtuse c) ∨ (obtuse b ∧ obtuse c)

-- Prove the negation equivalence
theorem negation_of_at_most_one_obtuse_is_at_least_two_obtuse (a b c : ℝ) :
  (¬ at_most_one_obtuse a b c) ↔ at_least_two_obtuse a b c :=
sorry

end NUMINAMATH_GPT_negation_of_at_most_one_obtuse_is_at_least_two_obtuse_l547_54780


namespace NUMINAMATH_GPT_correct_chart_for_percentage_representation_l547_54789

def bar_chart_characteristic := "easily shows the quantity"
def line_chart_characteristic := "shows the quantity and reflects the changes in quantity"
def pie_chart_characteristic := "reflects the relationship between a part and the whole"

def representation_requirement := "represents the percentage of students in each grade level in the fifth grade's physical education test scores out of the total number of students in the grade"

theorem correct_chart_for_percentage_representation : 
  (representation_requirement = pie_chart_characteristic) := 
by 
   -- The proof follows from the prior definition of characteristics.
   sorry

end NUMINAMATH_GPT_correct_chart_for_percentage_representation_l547_54789


namespace NUMINAMATH_GPT_thomas_savings_years_l547_54766

def weekly_allowance : ℕ := 50
def weekly_coffee_shop_earning : ℕ := 9 * 30
def weekly_spending : ℕ := 35
def car_cost : ℕ := 15000
def additional_amount_needed : ℕ := 2000
def weeks_in_a_year : ℕ := 52

def first_year_savings : ℕ := weeks_in_a_year * (weekly_allowance - weekly_spending)
def second_year_savings : ℕ := weeks_in_a_year * (weekly_coffee_shop_earning - weekly_spending)

noncomputable def total_savings_needed : ℕ := car_cost - additional_amount_needed

theorem thomas_savings_years : 
  first_year_savings + second_year_savings = total_savings_needed → 2 = 2 :=
by
  sorry

end NUMINAMATH_GPT_thomas_savings_years_l547_54766


namespace NUMINAMATH_GPT_least_possible_value_of_z_l547_54758

theorem least_possible_value_of_z (x y z : ℤ) 
  (hx : Even x) 
  (hy : Odd y) 
  (hz : Odd z) 
  (h1 : y - x > 5) 
  (h2 : z - x = 9) : 
  z = 11 := 
by
  sorry

end NUMINAMATH_GPT_least_possible_value_of_z_l547_54758


namespace NUMINAMATH_GPT_price_of_orange_is_60_l547_54737

-- Definitions from the conditions
def price_of_apple : ℕ := 40 -- The price of each apple is 40 cents
def total_fruits : ℕ := 10 -- Mary selects a total of 10 apples and oranges
def avg_price_initial : ℕ := 48 -- The average price of the 10 pieces of fruit is 48 cents
def put_back_oranges : ℕ := 2 -- Mary puts back 2 oranges
def avg_price_remaining : ℕ := 45 -- The average price of the remaining fruits is 45 cents

-- Variable definition for the price of an orange which will be solved for
variable (price_of_orange : ℕ)

-- Theorem: proving the price of each orange is 60 cents given the conditions
theorem price_of_orange_is_60 : 
  (∀ a o : ℕ, a + o = total_fruits →
  40 * a + price_of_orange * o = total_fruits * avg_price_initial →
  40 * a + price_of_orange * (o - put_back_oranges) = (total_fruits - put_back_oranges) * avg_price_remaining)
  → price_of_orange = 60 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_price_of_orange_is_60_l547_54737


namespace NUMINAMATH_GPT_students_per_bus_l547_54740

def total_students : ℕ := 360
def number_of_buses : ℕ := 8

theorem students_per_bus : total_students / number_of_buses = 45 :=
by
  sorry

end NUMINAMATH_GPT_students_per_bus_l547_54740


namespace NUMINAMATH_GPT_equal_commissions_implies_list_price_l547_54744

theorem equal_commissions_implies_list_price (x : ℝ) :
  (0.15 * (x - 15) = 0.25 * (x - 25)) → x = 40 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_equal_commissions_implies_list_price_l547_54744


namespace NUMINAMATH_GPT_simultaneous_inequalities_l547_54723

theorem simultaneous_inequalities (x : ℝ) 
    (h1 : x^3 - 11 * x^2 + 10 * x < 0) 
    (h2 : x^3 - 12 * x^2 + 32 * x > 0) : 
    (1 < x ∧ x < 4) ∨ (8 < x ∧ x < 10) :=
sorry

end NUMINAMATH_GPT_simultaneous_inequalities_l547_54723


namespace NUMINAMATH_GPT_remainder_cd_mod_40_l547_54761

theorem remainder_cd_mod_40 (c d : ℤ) (hc : c % 80 = 75) (hd : d % 120 = 117) : (c + d) % 40 = 32 :=
by
  sorry

end NUMINAMATH_GPT_remainder_cd_mod_40_l547_54761


namespace NUMINAMATH_GPT_diagonals_of_hexadecagon_l547_54739

-- Define the function to calculate number of diagonals in a convex polygon
def num_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

-- State the theorem for the number of diagonals in a convex hexadecagon
theorem diagonals_of_hexadecagon : num_diagonals 16 = 104 := by
  -- sorry is used to indicate the proof is skipped
  sorry

end NUMINAMATH_GPT_diagonals_of_hexadecagon_l547_54739


namespace NUMINAMATH_GPT_find_value_of_A_l547_54745

theorem find_value_of_A (x : ℝ) (h₁ : x - 3 * (x - 2) ≥ 2) (h₂ : 4 * x - 2 < 5 * x - 1) (h₃ : x ≠ 1) (h₄ : x ≠ -1) (h₅ : x ≠ 0) (hx : x = 2) :
  let A := (3 * x / (x - 1) - x / (x + 1)) / (x / (x^2 - 1))
  A = 8 :=
by
  -- Proof will be filled in
  sorry

end NUMINAMATH_GPT_find_value_of_A_l547_54745


namespace NUMINAMATH_GPT_de_morgan_implication_l547_54700

variables (p q : Prop)

theorem de_morgan_implication (h : ¬(p ∧ q)) : ¬p ∨ ¬q :=
sorry

end NUMINAMATH_GPT_de_morgan_implication_l547_54700


namespace NUMINAMATH_GPT_inverse_of_h_l547_54706

noncomputable def h (x : ℝ) : ℝ := 3 - 7 * x
noncomputable def k (x : ℝ) : ℝ := (3 - x) / 7

theorem inverse_of_h :
  (∀ x : ℝ, h (k x) = x) ∧ (∀ x : ℝ, k (h x) = x) :=
by
  sorry

end NUMINAMATH_GPT_inverse_of_h_l547_54706


namespace NUMINAMATH_GPT_dot_product_value_l547_54751

-- Define vectors a and b, and the condition of their linear combination
structure Vector2D :=
  (x : ℝ)
  (y : ℝ)

def a : Vector2D := ⟨-1, 2⟩
def b (m : ℝ) : Vector2D := ⟨m, 1⟩

-- Define the condition that vector a + 2b is parallel to 2a - b
def parallel (v w : Vector2D) : Prop := ∃ k : ℝ, v.x = k * w.x ∧ v.y = k * w.y

def vector_add (v w : Vector2D) : Vector2D := ⟨v.x + w.x, v.y + w.y⟩
def scalar_mul (c : ℝ) (v : Vector2D) : Vector2D := ⟨c * v.x, c * v.y⟩

-- Dot product definition
def dot_product (v w : Vector2D) : ℝ := v.x * w.x + v.y * w.y

-- The theorem to prove
theorem dot_product_value (m : ℝ)
  (h : parallel (vector_add a (scalar_mul 2 (b m))) (vector_add (scalar_mul 2 a) (scalar_mul (-1) (b m)))) :
  dot_product a (b m) = 5 / 2 :=
sorry

end NUMINAMATH_GPT_dot_product_value_l547_54751


namespace NUMINAMATH_GPT_complex_in_third_quadrant_l547_54708

theorem complex_in_third_quadrant (x : ℝ) : 
  (x^2 - 6*x + 5 < 0) ∧ (x - 2 < 0) ↔ (1 < x ∧ x < 2) := 
by
  sorry

end NUMINAMATH_GPT_complex_in_third_quadrant_l547_54708


namespace NUMINAMATH_GPT_circles_C1_C2_intersect_C1_C2_l547_54750

noncomputable def center1 : (ℝ × ℝ) := (5, 3)
noncomputable def radius1 : ℝ := 3

noncomputable def center2 : (ℝ × ℝ) := (2, -1)
noncomputable def radius2 : ℝ := Real.sqrt 14

noncomputable def distance : ℝ := Real.sqrt ((5 - 2)^2 + (3 + 1)^2)

def circles_intersect : Prop :=
  radius2 - radius1 < distance ∧ distance < radius2 + radius1

theorem circles_C1_C2_intersect_C1_C2 : circles_intersect :=
by
  -- The proof of this theorem is to be worked out using the given conditions and steps.
  sorry

end NUMINAMATH_GPT_circles_C1_C2_intersect_C1_C2_l547_54750


namespace NUMINAMATH_GPT_whitney_greatest_sets_l547_54784

-- Define the conditions: Whitney has 4 T-shirts and 20 buttons.
def num_tshirts := 4
def num_buttons := 20

-- The problem statement: Prove that the greatest number of sets Whitney can make is 4.
theorem whitney_greatest_sets : Nat.gcd num_tshirts num_buttons = 4 := by
  sorry

end NUMINAMATH_GPT_whitney_greatest_sets_l547_54784


namespace NUMINAMATH_GPT_range_of_a_l547_54756

theorem range_of_a
  (h : ∀ x : ℝ, |x - 1| + |x - 2| > Real.log (a ^ 2) / Real.log 4) :
  a ∈ Set.Ioo (-2 : ℝ) 0 ∪ Set.Ioo 0 2 :=
sorry

end NUMINAMATH_GPT_range_of_a_l547_54756


namespace NUMINAMATH_GPT_x7_value_l547_54747

theorem x7_value
  (x : ℕ → ℕ)
  (h1 : x 6 = 144)
  (h2 : ∀ n, 1 ≤ n ∧ n ≤ 4 → x (n + 3) = x (n + 2) * (x (n + 1) + x n))
  (h3 : ∀ m, m < 1 → 0 < x m) : x 7 = 3456 :=
by
  sorry

end NUMINAMATH_GPT_x7_value_l547_54747


namespace NUMINAMATH_GPT_calibration_measurements_l547_54719

theorem calibration_measurements (holes : Fin 15 → ℝ) (diameter : ℝ)
  (h1 : ∀ i : Fin 15, holes i = 10 + i.val * 0.04)
  (h2 : 10 ≤ diameter ∧ diameter ≤ 10 + 14 * 0.04) :
  ∃ tries : ℕ, (tries ≤ 4) ∧ (∀ (i : Fin 15), if diameter ≤ holes i then True else False) :=
sorry

end NUMINAMATH_GPT_calibration_measurements_l547_54719


namespace NUMINAMATH_GPT_equal_probability_of_selection_l547_54701

-- Define the number of total students
def total_students : ℕ := 86

-- Define the number of students to be eliminated through simple random sampling
def eliminated_students : ℕ := 6

-- Define the number of students selected through systematic sampling
def selected_students : ℕ := 8

-- Define the probability calculation
def probability_not_eliminated : ℚ := 80 / 86
def probability_selected : ℚ := 8 / 80
def combined_probability : ℚ := probability_not_eliminated * probability_selected

theorem equal_probability_of_selection :
  combined_probability = 4 / 43 :=
by
  -- We do not need to complete the proof as per instruction
  sorry

end NUMINAMATH_GPT_equal_probability_of_selection_l547_54701


namespace NUMINAMATH_GPT_satisfy_third_eq_l547_54752

theorem satisfy_third_eq 
  (x y : ℝ) 
  (h1 : x^2 - 3 * x * y + 2 * y^2 + x - y = 0)
  (h2 : x^2 - 2 * x * y + y^2 - 5 * x + 7 * y = 0) 
  : x * y - 12 * x + 15 * y = 0 :=
by
  sorry

end NUMINAMATH_GPT_satisfy_third_eq_l547_54752


namespace NUMINAMATH_GPT_litter_collection_total_weight_l547_54787

/-- Gina collected 8 bags of litter: 5 bags of glass bottles weighing 7 pounds each and 3 bags of plastic waste weighing 4 pounds each. The 25 neighbors together collected 120 times as much glass as Gina and 80 times as much plastic as Gina. Prove that the total weight of all the collected litter is 5207 pounds. -/
theorem litter_collection_total_weight
  (glass_bags_gina : ℕ)
  (glass_weight_per_bag : ℕ)
  (plastic_bags_gina : ℕ)
  (plastic_weight_per_bag : ℕ)
  (neighbors_glass_multiplier : ℕ)
  (neighbors_plastic_multiplier : ℕ)
  (total_weight : ℕ)
  (h1 : glass_bags_gina = 5)
  (h2 : glass_weight_per_bag = 7)
  (h3 : plastic_bags_gina = 3)
  (h4 : plastic_weight_per_bag = 4)
  (h5 : neighbors_glass_multiplier = 120)
  (h6 : neighbors_plastic_multiplier = 80)
  (h_total_weight : total_weight = 5207) : total_weight = 
  glass_bags_gina * glass_weight_per_bag + 
  plastic_bags_gina * plastic_weight_per_bag + 
  neighbors_glass_multiplier * (glass_bags_gina * glass_weight_per_bag) + 
  neighbors_plastic_multiplier * (plastic_bags_gina * plastic_weight_per_bag) := 
by {
  /- Proof omitted -/
  sorry
}

end NUMINAMATH_GPT_litter_collection_total_weight_l547_54787


namespace NUMINAMATH_GPT_people_at_the_beach_l547_54768

-- Conditions
def initial : ℕ := 3  -- Molly and her parents
def joined : ℕ := 100 -- 100 people joined at the beach
def left : ℕ := 40    -- 40 people left at 5:00

-- Proof statement
theorem people_at_the_beach : initial + joined - left = 63 :=
by
  sorry

end NUMINAMATH_GPT_people_at_the_beach_l547_54768


namespace NUMINAMATH_GPT_fraction_sum_l547_54717

variable (x y : ℚ)

theorem fraction_sum (h : x / y = 3 / 4) : (x + y) / y = 7 / 4 := 
by
  sorry

end NUMINAMATH_GPT_fraction_sum_l547_54717


namespace NUMINAMATH_GPT_maximum_value_minimum_value_l547_54753

def is_four_digit_number (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

def check_digits (N M : ℕ) (a b c d e f g h : ℕ) : Prop :=
  N = 1000 * a + 100 * b + 10 * c + d ∧
  M = 1000 * e + 100 * f + 10 * g + h ∧
  a ≠ e ∧
  b ≠ f ∧
  c ≠ g ∧
  d ≠ h ∧
  a ≠ f ∧
  a ≠ g ∧
  a ≠ h ∧
  b ≠ e ∧
  b ≠ g ∧
  b ≠ h ∧
  c ≠ e ∧
  c ≠ f ∧
  c ≠ h ∧
  d ≠ e ∧
  d ≠ f ∧
  d ≠ g

theorem maximum_value (N M a b c d e f g h : ℕ) :
  is_four_digit_number N →
  is_four_digit_number M →
  check_digits N M a b c d e f g h →
  N - M = 1994 →
  N + M = 15000 :=
by
  intros
  sorry

theorem minimum_value (N M a b c d e f g h : ℕ) :
  is_four_digit_number N →
  is_four_digit_number M →
  check_digits N M a b c d e f g h →
  N - M = 1994 →
  N + M = 4998 :=
by
  intros
  sorry

end NUMINAMATH_GPT_maximum_value_minimum_value_l547_54753


namespace NUMINAMATH_GPT_relation_between_x_and_y_l547_54742

variable (t : ℝ)
variable (x : ℝ := t ^ (2 / (t - 1))) (y : ℝ := t ^ ((t + 1) / (t - 1)))

theorem relation_between_x_and_y (h1 : t > 0) (h2 : t ≠ 1) : y ^ (1 / x) = x ^ y :=
by sorry

end NUMINAMATH_GPT_relation_between_x_and_y_l547_54742


namespace NUMINAMATH_GPT_min_value_g_range_of_m_l547_54778

section
variable (x : ℝ)
noncomputable def g (x : ℝ) := Real.exp x - x

theorem min_value_g :
  (∀ x : ℝ, g x ≥ g 0) ∧ g 0 = 1 := 
by 
  sorry

theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ (2 * x - m) / g x > x) → m < Real.log 2 ^ 2 := 
by 
  sorry
end

end NUMINAMATH_GPT_min_value_g_range_of_m_l547_54778


namespace NUMINAMATH_GPT_first_group_men_count_l547_54728

/-- Given that 10 men can complete a piece of work in 90 hours,
prove that the number of men M in the first group who can complete
the same piece of work in 25 hours is 36. -/
theorem first_group_men_count (M : ℕ) (h : (10 * 90 = 25 * M)) : M = 36 :=
by
  sorry

end NUMINAMATH_GPT_first_group_men_count_l547_54728


namespace NUMINAMATH_GPT_percentage_of_first_pay_cut_l547_54743

theorem percentage_of_first_pay_cut
  (x : ℝ)
  (h1 : ∃ y z w : ℝ, y = 1 - x/100 ∧ z = 0.86 ∧ w = 0.82 ∧ y * z * w = 0.648784):
  x = 8.04 := by
-- The proof will be added here, this is just the statement
sorry

end NUMINAMATH_GPT_percentage_of_first_pay_cut_l547_54743


namespace NUMINAMATH_GPT_no_nat_numbers_satisfy_eqn_l547_54704

theorem no_nat_numbers_satisfy_eqn (a b : ℕ) : a^2 - 3 * b^2 ≠ 8 := by
  sorry

end NUMINAMATH_GPT_no_nat_numbers_satisfy_eqn_l547_54704


namespace NUMINAMATH_GPT_gcd_48_30_is_6_l547_54705

/-- Prove that the Greatest Common Divisor (GCD) of 48 and 30 is 6. -/
theorem gcd_48_30_is_6 : Int.gcd 48 30 = 6 := by
  sorry

end NUMINAMATH_GPT_gcd_48_30_is_6_l547_54705


namespace NUMINAMATH_GPT_group_C_both_axis_and_central_l547_54779

def is_axisymmetric (shape : Type) : Prop := sorry
def is_centrally_symmetric (shape : Type) : Prop := sorry

def square : Type := sorry
def rhombus : Type := sorry
def rectangle : Type := sorry
def parallelogram : Type := sorry
def equilateral_triangle : Type := sorry
def isosceles_triangle : Type := sorry

def group_A := [square, rhombus, rectangle, parallelogram]
def group_B := [equilateral_triangle, square, rhombus, rectangle]
def group_C := [square, rectangle, rhombus]
def group_D := [parallelogram, square, isosceles_triangle]

def all_axisymmetric (group : List Type) : Prop :=
  ∀ shape ∈ group, is_axisymmetric shape

def all_centrally_symmetric (group : List Type) : Prop :=
  ∀ shape ∈ group, is_centrally_symmetric shape

theorem group_C_both_axis_and_central :
  (all_axisymmetric group_C ∧ all_centrally_symmetric group_C) ∧
  (∀ (group : List Type), (all_axisymmetric group ∧ all_centrally_symmetric group) →
    group = group_C) :=
by sorry

end NUMINAMATH_GPT_group_C_both_axis_and_central_l547_54779


namespace NUMINAMATH_GPT_f_2008_eq_zero_l547_54788

noncomputable def f : ℝ → ℝ := sorry

-- f is odd function
axiom f_odd : ∀ x : ℝ, f (-x) = -f x

-- f satisfies f(x + 2) = -f(x)
axiom f_periodic : ∀ x : ℝ, f (x + 2) = -f x

theorem f_2008_eq_zero : f 2008 = 0 :=
by
  sorry

end NUMINAMATH_GPT_f_2008_eq_zero_l547_54788


namespace NUMINAMATH_GPT_find_f6_l547_54792

variable {R : Type*} [AddGroup R] [Semiring R]

def functional_equation (f : R → R) :=
∀ x y : R, f (x + y) = f x + f y

theorem find_f6 (f : ℝ → ℝ) (h1 : functional_equation f) (h2 : f 4 = 10) : f 6 = 10 :=
sorry

end NUMINAMATH_GPT_find_f6_l547_54792


namespace NUMINAMATH_GPT_triangle_proof_problem_l547_54729

-- The conditions and question programmed as a Lean theorem statement
theorem triangle_proof_problem
    (A B C : ℝ)
    (h1 : A > B)
    (S T : ℝ)
    (h2 : A = C)
    (K : ℝ)
    (arc_mid_A : K = A): -- K is midpoint of the arc A
    
    RS = K := sorry

end NUMINAMATH_GPT_triangle_proof_problem_l547_54729


namespace NUMINAMATH_GPT_cube_surface_area_sum_of_edges_l547_54748

noncomputable def edge_length (sum_of_edges : ℝ) (num_of_edges : ℝ) : ℝ :=
  sum_of_edges / num_of_edges

noncomputable def surface_area (edge_length : ℝ) : ℝ :=
  6 * edge_length ^ 2

theorem cube_surface_area_sum_of_edges (sum_of_edges : ℝ) (num_of_edges : ℝ) (expected_area : ℝ) :
  num_of_edges = 12 → sum_of_edges = 72 → surface_area (edge_length sum_of_edges num_of_edges) = expected_area :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_cube_surface_area_sum_of_edges_l547_54748


namespace NUMINAMATH_GPT_power_mod_remainder_l547_54712

theorem power_mod_remainder :
  3 ^ 3021 % 13 = 1 :=
by
  sorry

end NUMINAMATH_GPT_power_mod_remainder_l547_54712


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l547_54727

theorem sufficient_but_not_necessary_condition (a : ℝ) : (a > 1 → (1 / a < 1)) ∧ ¬((1 / a < 1) → a > 1) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l547_54727


namespace NUMINAMATH_GPT_minimal_storing_capacity_required_l547_54773

theorem minimal_storing_capacity_required (k : ℕ) (h1 : k > 0)
    (bins : ℕ → ℕ → ℕ → Prop)
    (h_initial : bins 0 0 0)
    (h_laundry_generated : ∀ n, bins (10 * n) (10 * n) (10 * n))
    (h_heaviest_bin_emptied : ∀ n r b g, (r + b + g = 10 * n) → max r (max b g) + 10 * n - max r (max b g) = 10 * n)
    : ∀ (capacity : ℕ), capacity = 25 :=
sorry

end NUMINAMATH_GPT_minimal_storing_capacity_required_l547_54773

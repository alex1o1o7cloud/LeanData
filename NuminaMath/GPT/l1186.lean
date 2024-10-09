import Mathlib

namespace operation_X_value_l1186_118610

def operation_X (a b : ℤ) : ℤ := b + 7 * a - a^3 + 2 * b

theorem operation_X_value : operation_X 4 3 = -27 := by
  sorry

end operation_X_value_l1186_118610


namespace value_of_x_l1186_118678

theorem value_of_x (x y : ℕ) (h1 : x / y = 8 / 3) (h2 : y = 27) : x = 72 :=
by
  sorry

end value_of_x_l1186_118678


namespace circle_radius_l1186_118684

theorem circle_radius (r A C : Real) (h1 : A = π * r^2) (h2 : C = 2 * π * r) (h3 : A + (Real.cos (π / 3)) * C = 56 * π) : r = 7 := 
by 
  sorry

end circle_radius_l1186_118684


namespace quadrilateral_correct_choice_l1186_118625

/-- Define the triangle inequality theorem for four line segments.
    A quadrilateral can be formed if for any:
    - The sum of the lengths of any three segments is greater than the length of the fourth segment.
-/
def is_quadrilateral (a b c d : ℕ) : Prop :=
  (a + b + c > d) ∧ (a + b + d > c) ∧ (a + c + d > b) ∧ (b + c + d > a)

/-- Determine which set of three line segments can form a quadrilateral with a fourth line segment of length 5.
    We prove that the correct choice is the set (3, 3, 3). --/
theorem quadrilateral_correct_choice :
  is_quadrilateral 3 3 3 5 ∧  ¬ is_quadrilateral 1 1 1 5 ∧  ¬ is_quadrilateral 1 1 8 5 ∧  ¬ is_quadrilateral 1 2 2 5 :=
by
  sorry

end quadrilateral_correct_choice_l1186_118625


namespace distance_origin_to_point_l1186_118681

theorem distance_origin_to_point : 
  let origin := (0, 0)
  let point := (8, 15)
  dist origin point = 17 :=
by
  let dist (p1 p2 : ℝ × ℝ) : ℝ :=
    Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  sorry

end distance_origin_to_point_l1186_118681


namespace part1_l1186_118633

-- Define the sets P and Q
def P (a : ℝ) : Set ℝ := {x | 3 * a - 10 ≤ x ∧ x < 2 * a + 1}
def Q : Set ℝ := {x | |2 * x - 3| ≤ 7}

-- Define the complement of Q in ℝ
def Q_complement : Set ℝ := {x | x < -2 ∨ x > 5}

-- Define the specific value of a
def a : ℝ := 2

-- Define the specific set P when a = 2
def P_a2 : Set ℝ := {x | -4 ≤ x ∧ x < 5}

-- Define the intersection
def intersection : Set ℝ := {x | -4 ≤ x ∧ x < -2}

theorem part1 : P a ∩ Q_complement = intersection := sorry

end part1_l1186_118633


namespace temp_product_l1186_118628

theorem temp_product (N : ℤ) (M D : ℤ)
  (h1 : M = D + N)
  (h2 : M - 8 = D + N - 8)
  (h3 : D + 5 = D + 5)
  (h4 : abs ((D + N - 8) - (D + 5)) = 3) :
  (N = 16 ∨ N = 10) →
  16 * 10 = 160 := 
by sorry

end temp_product_l1186_118628


namespace proof_problem_l1186_118617

open Real

-- Define the problem statements as Lean hypotheses
def p : Prop := ∀ a : ℝ, exp a ≥ a + 1
def q : Prop := ∃ α β : ℝ, sin (α + β) = sin α + sin β

theorem proof_problem : p ∧ q :=
by
  sorry

end proof_problem_l1186_118617


namespace prime_base_representation_of_360_l1186_118612

theorem prime_base_representation_of_360 :
  ∃ (exponents : List ℕ), exponents = [3, 2, 1, 0]
  ∧ (2^exponents.head! * 3^(exponents.tail!.head!) * 5^(exponents.tail!.tail!.head!) * 7^(exponents.tail!.tail!.tail!.head!)) = 360 := by
sorry

end prime_base_representation_of_360_l1186_118612


namespace M_intersection_N_l1186_118638

noncomputable def M := {x : ℝ | 0 ≤ x ∧ x < 16}
noncomputable def N := {x : ℝ | x ≥ 1 / 3}

theorem M_intersection_N :
  (M ∩ N) = {x : ℝ | 1 / 3 ≤ x ∧ x < 16} := by
sorry

end M_intersection_N_l1186_118638


namespace sons_ages_l1186_118642

theorem sons_ages (m n : ℕ) (h : m * n + m + n = 34) : 
  (m = 4 ∧ n = 6) ∨ (m = 6 ∧ n = 4) :=
sorry

end sons_ages_l1186_118642


namespace combustion_CH₄_forming_water_l1186_118607

/-
Combustion reaction for Methane: CH₄ + 2 O₂ → CO₂ + 2 H₂O
Given:
  3 moles of Methane
  6 moles of Oxygen
  Balanced equation: CH₄ + 2 O₂ → CO₂ + 2 H₂O
Goal: Prove that 6 moles of Water (H₂O) are formed.
-/

-- Define the necessary definitions for the context
def moles_CH₄ : ℝ := 3
def moles_O₂ : ℝ := 6
def ratio_water_methane : ℝ := 2

theorem combustion_CH₄_forming_water :
  moles_CH₄ * ratio_water_methane = 6 :=
by
  sorry

end combustion_CH₄_forming_water_l1186_118607


namespace value_of_expression_l1186_118698

theorem value_of_expression (x y : ℤ) (h1 : x = -6) (h2 : y = -3) : 4 * (x - y) ^ 2 - x * y = 18 :=
by sorry

end value_of_expression_l1186_118698


namespace fractions_with_smallest_difference_l1186_118601

theorem fractions_with_smallest_difference 
    (x y : ℤ) 
    (f1 : ℚ := (x : ℚ) / 8) 
    (f2 : ℚ := (y : ℚ) / 13) 
    (h : abs (13 * x - 8 * y) = 1): 
    (f1 ≠ f2) ∧ abs ((x : ℚ) / 8 - (y : ℚ) / 13) = 1 / 104 :=
by
  sorry

end fractions_with_smallest_difference_l1186_118601


namespace g_symmetry_value_h_m_interval_l1186_118686

noncomputable def f (x : ℝ) : ℝ :=
  (Real.cos (x + Real.pi / 12)) ^ 2

noncomputable def g (x : ℝ) : ℝ :=
  1 + 1 / 2 * Real.sin (2 * x)

noncomputable def h (x : ℝ) : ℝ :=
  f x + g x

theorem g_symmetry_value (k : ℤ) : 
  g (k * Real.pi / 2 - Real.pi / 12) = (3 + (-1) ^ k) / 4 :=
by
  sorry

theorem h_m_interval (m : ℝ) : 
  (∀ x ∈ Set.Icc (- Real.pi / 12) (5 * Real.pi / 12), |h x - m| ≤ 1) ↔ (1 ≤ m ∧ m ≤ 9 / 4) :=
by
  sorry

end g_symmetry_value_h_m_interval_l1186_118686


namespace workers_cut_down_correct_l1186_118667

def initial_oak_trees : ℕ := 9
def remaining_oak_trees : ℕ := 7
def cut_down_oak_trees : ℕ := initial_oak_trees - remaining_oak_trees

theorem workers_cut_down_correct : cut_down_oak_trees = 2 := by
  sorry

end workers_cut_down_correct_l1186_118667


namespace garden_dimensions_l1186_118614

theorem garden_dimensions (l w : ℕ) (h1 : 2 * l + 2 * w = 60) (h2 : l * w = 221) : 
    (l = 17 ∧ w = 13) ∨ (l = 13 ∧ w = 17) :=
sorry

end garden_dimensions_l1186_118614


namespace alex_basketball_points_l1186_118663

theorem alex_basketball_points (f t s : ℕ) 
  (h : f + t + s = 40) 
  (points_scored : ℝ := 0.8 * f + 0.3 * t + s) :
  points_scored = 28 :=
sorry

end alex_basketball_points_l1186_118663


namespace line_single_point_not_necessarily_tangent_l1186_118637

-- Define a curve
def curve : Type := ℝ → ℝ

-- Define a line
def line (m b : ℝ) : curve := λ x => m * x + b

-- Define a point of intersection
def intersects_at (l : curve) (c : curve) (x : ℝ) : Prop :=
  l x = c x

-- Define the property of having exactly one common point
def has_single_intersection (l : curve) (c : curve) : Prop :=
  ∃ x, ∀ y ≠ x, l y ≠ c y

-- Define the tangent line property
def is_tangent (l : curve) (c : curve) (x : ℝ) : Prop :=
  ∀ ε > 0, ∃ δ > 0, ∀ h, abs h < δ → abs ((c (x + h) - c x) / h - (l (x + h) - l x) / h) < ε

-- The proof statement: There exists a curve c and a line l such that l has exactly one intersection point with c, but l is not necessarily a tangent to c.
theorem line_single_point_not_necessarily_tangent :
  ∃ c : curve, ∃ l : curve, has_single_intersection l c ∧ ∃ x, ¬ is_tangent l c x :=
sorry

end line_single_point_not_necessarily_tangent_l1186_118637


namespace units_digit_of_product_of_first_three_positive_composite_numbers_l1186_118615

theorem units_digit_of_product_of_first_three_positive_composite_numbers :
  (4 * 6 * 8) % 10 = 2 :=
by sorry

end units_digit_of_product_of_first_three_positive_composite_numbers_l1186_118615


namespace sin_ninety_degrees_l1186_118680

theorem sin_ninety_degrees : Real.sin (90 * Real.pi / 180) = 1 := 
by
  sorry

end sin_ninety_degrees_l1186_118680


namespace point_3_units_away_l1186_118693

theorem point_3_units_away (x : ℤ) (h : abs (x + 1) = 3) : x = 2 ∨ x = -4 :=
by
  sorry

end point_3_units_away_l1186_118693


namespace sequence_increasing_l1186_118630

noncomputable def a (n : ℕ) : ℚ := (2 * n) / (2 * n + 1)

theorem sequence_increasing (n : ℕ) (hn : 0 < n) : a n < a (n + 1) :=
by
  -- Proof to be provided
  sorry

end sequence_increasing_l1186_118630


namespace number_of_teams_in_BIG_N_l1186_118651

theorem number_of_teams_in_BIG_N (n : ℕ) (h : n * (n - 1) / 2 = 28) : n = 8 := by
  sorry

end number_of_teams_in_BIG_N_l1186_118651


namespace range_of_3t_plus_s_l1186_118639

noncomputable def f : ℝ → ℝ := sorry

def is_increasing (f : ℝ → ℝ) := ∀ x y, x ≤ y → f x ≤ f y

def symmetric_about (f : ℝ → ℝ) (a b : ℝ) := 
  ∀ x, f (x - a) = b - f (a - x)

def satisfies_inequality (s t : ℝ) (f : ℝ → ℝ) := 
  f (s^2 - 2*s) ≥ -f (2*t - t^2)

def in_interval (s : ℝ) := 1 ≤ s ∧ s ≤ 4

theorem range_of_3t_plus_s (f : ℝ → ℝ) :
  is_increasing f ∧ symmetric_about f 3 0 →
  (∀ s t, satisfies_inequality s t f → in_interval s → -2 ≤ 3 * t + s ∧ 3 * t + s ≤ 16) :=
sorry

end range_of_3t_plus_s_l1186_118639


namespace simplify_sum1_simplify_sum2_l1186_118668

theorem simplify_sum1 : 296 + 297 + 298 + 299 + 1 + 2 + 3 + 4 = 1200 := by
  sorry

theorem simplify_sum2 : 457 + 458 + 459 + 460 + 461 + 462 + 463 = 3220 := by
  sorry

end simplify_sum1_simplify_sum2_l1186_118668


namespace number_of_DVDs_sold_l1186_118694

theorem number_of_DVDs_sold (C D: ℤ) (h₁ : D = 16 * C / 10) (h₂ : D + C = 273) : D = 168 := 
sorry

end number_of_DVDs_sold_l1186_118694


namespace sasha_kolya_distance_l1186_118622

theorem sasha_kolya_distance
  (v_S v_L v_K : ℝ) 
  (h1 : Lesha_dist = 100 - 10) 
  (h2 : Kolya_dist = 100 - 10) 
  (h3 : v_L = (90 / 100) * v_S) 
  (h4 : v_K = (90 / 100) * v_L) 
  : v_S * (100/v_S - 10/v_S) = 19 :=
by
  sorry

end sasha_kolya_distance_l1186_118622


namespace packs_to_purchase_l1186_118636

theorem packs_to_purchase {n m k : ℕ} (h : 8 * n + 15 * m + 30 * k = 135) : n + m + k = 5 :=
sorry

end packs_to_purchase_l1186_118636


namespace complex_number_value_l1186_118669

theorem complex_number_value (i : ℂ) (h : i^2 = -1) : i^13 * (1 + i) = -1 + i :=
by
  sorry

end complex_number_value_l1186_118669


namespace wall_height_correct_l1186_118685

noncomputable def brick_volume : ℝ := 25 * 11.25 * 6

noncomputable def wall_total_volume (num_bricks : ℕ) (brick_vol : ℝ) : ℝ := num_bricks * brick_vol

noncomputable def wall_height (total_volume : ℝ) (length : ℝ) (thickness : ℝ) : ℝ :=
  total_volume / (length * thickness)

theorem wall_height_correct :
  wall_height (wall_total_volume 7200 brick_volume) 900 22.5 = 600 := by
  sorry

end wall_height_correct_l1186_118685


namespace Carrie_has_50_dollars_left_l1186_118629

/-
Conditions:
1. initial_amount = 91
2. sweater_cost = 24
3. tshirt_cost = 6
4. shoes_cost = 11
-/
def initial_amount : ℕ := 91
def sweater_cost : ℕ := 24
def tshirt_cost : ℕ := 6
def shoes_cost : ℕ := 11

/-
Question:
How much money does Carrie have left?
-/
def total_spent : ℕ := sweater_cost + tshirt_cost + shoes_cost
def money_left : ℕ := initial_amount - total_spent

def proof_statement : Prop := money_left = 50

theorem Carrie_has_50_dollars_left : proof_statement :=
by
  sorry

end Carrie_has_50_dollars_left_l1186_118629


namespace ratio_of_larger_to_smaller_l1186_118689

variable {x y : ℝ}

-- Condition for x and y being positive and x > y
axiom x_pos : 0 < x
axiom y_pos : 0 < y
axiom x_gt_y : x > y

-- Condition for sum and difference relationship
axiom sum_diff_relation : x + y = 7 * (x - y)

-- Theorem: Ratio of the larger number to the smaller number is 2
theorem ratio_of_larger_to_smaller : x / y = 2 :=
by
  sorry

end ratio_of_larger_to_smaller_l1186_118689


namespace math_problem_l1186_118650

theorem math_problem
  (p q r : ℕ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r)
  (h1 : p * q + r = 47)
  (h2 : q * r + p = 47)
  (h3 : r * p + q = 47) :
  p + q + r = 48 :=
sorry

end math_problem_l1186_118650


namespace find_smallest_w_l1186_118682

theorem find_smallest_w (w : ℕ) (h : 0 < w) : 
  (∀ k, k = 2^5 ∨ k = 3^3 ∨ k = 12^2 → (k ∣ (936 * w))) ↔ w = 36 := by 
  sorry

end find_smallest_w_l1186_118682


namespace geometric_series_sum_l1186_118620

theorem geometric_series_sum :
  let a := 3
  let r := -2
  let n := 10
  let S := a * ((r^n - 1) / (r - 1))
  S = -1023 :=
by 
  -- Sorry allows us to omit the proof details
  sorry

end geometric_series_sum_l1186_118620


namespace sum_series_eq_l1186_118649

theorem sum_series_eq :
  ∑' n : ℕ, (4 * (n + 1) - 3) / 3^(n + 1) = 3 / 2 :=
sorry

end sum_series_eq_l1186_118649


namespace find_xiao_li_compensation_l1186_118643

-- Define the conditions
variable (total_days : ℕ) (extra_days : ℕ) (extra_compensation : ℕ)
variable (daily_work : ℕ) (daily_reward : ℕ) (xiao_li_days : ℕ)

-- Define the total compensation for Xiao Li
def xiao_li_compensation (xiao_li_days daily_reward : ℕ) : ℕ := xiao_li_days * daily_reward

-- The theorem statement asserting the final answer
theorem find_xiao_li_compensation
  (h1 : total_days = 12)
  (h2 : extra_days = 3)
  (h3 : extra_compensation = 2700)
  (h4 : daily_work = 1)
  (h5 : daily_reward = 225)
  (h6 : xiao_li_days = 2)
  (h7 : (total_days - extra_days) * daily_work = xiao_li_days * daily_work):
  xiao_li_compensation xiao_li_days daily_reward = 450 := 
sorry

end find_xiao_li_compensation_l1186_118643


namespace max_expr_value_l1186_118695

theorem max_expr_value (a b c d : ℝ) (ha : 0 ≤ a) (ha1 : a ≤ 1) (hb : 0 ≤ b) (hb1 : b ≤ 1) (hc : 0 ≤ c) (hc1 : c ≤ 1) (hd : 0 ≤ d) (hd1 : d ≤ 1) : 
  a + b + c + d - a * b - b * c - c * d - d * a ≤ 2 :=
sorry

end max_expr_value_l1186_118695


namespace total_markup_l1186_118688

theorem total_markup (p : ℝ) (o : ℝ) (n : ℝ) (m : ℝ) : 
  p = 48 → o = 0.35 → n = 18 → m = o * p + n → m = 34.8 :=
by
  intro hp ho hn hm
  sorry

end total_markup_l1186_118688


namespace sue_library_inventory_l1186_118646

theorem sue_library_inventory :
  let initial_books := 15
  let initial_movies := 6
  let returned_books := 8
  let returned_movies := initial_movies / 3
  let borrowed_more_books := 9
  let current_books := initial_books - returned_books + borrowed_more_books
  let current_movies := initial_movies - returned_movies
  current_books + current_movies = 20 :=
by
  -- no implementation provided
  sorry

end sue_library_inventory_l1186_118646


namespace find_original_number_l1186_118647

theorem find_original_number (x : ℝ) (h : 0.5 * x = 30) : x = 60 :=
sorry

end find_original_number_l1186_118647


namespace find_e_m_l1186_118659

noncomputable def B (e : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := ![![4, 5], ![7, e]]
noncomputable def B_inv (e : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := (1 / (4 * e - 35)) • ![![e, -5], ![-7, 4]]

theorem find_e_m (e m : ℝ) (B_inv_eq_mB : B_inv e = m • B e) : e = -4 ∧ m = 1 / 51 :=
sorry

end find_e_m_l1186_118659


namespace line_parabola_intersection_l1186_118662

theorem line_parabola_intersection (k : ℝ) :
  (∀ x y, y = k * x + 1 ∧ y^2 = 4 * x → y = 1 ∧ x = 1 / 4) ∨
  (∀ x y, y = k * x + 1 ∧ y^2 = 4 * x → (k^2 * x^2 + (2 * k - 4) * x + 1 = 0) ∧ (4 * k * k - 16 * k + 16 - 4 * k * k = 0) → k = 1) :=
sorry

end line_parabola_intersection_l1186_118662


namespace solution_set_l1186_118640

theorem solution_set (x : ℝ) : (2 : ℝ) ^ (|x-2| + |x-4|) > 2^6 ↔ x < 0 ∨ x > 6 :=
by
  sorry

end solution_set_l1186_118640


namespace rickey_time_l1186_118616

/-- Prejean's speed in a race was three-quarters that of Rickey. 
If they both took a total of 70 minutes to run the race, 
the total number of minutes that Rickey took to finish the race is 40. -/
theorem rickey_time (t : ℝ) (h1 : ∀ p : ℝ, p = (3/4) * t) (h2 : t + (3/4) * t = 70) : t = 40 := 
by
  sorry

end rickey_time_l1186_118616


namespace problem_statement_l1186_118621

-- Define line and plane as types
variable (Line Plane : Type)

-- Define the perpendicularity and parallelism relations
variable (perpendicular : Line → Line → Prop)
variable (perpendicularLPlane : Line → Plane → Prop)
variable (perpendicularPPlane : Plane → Plane → Prop)

-- Distinctness of lines and planes
variable (a b : Line)
variable (α β : Plane)

-- Conditions given in the problem
axiom distinct_lines : a ≠ b
axiom distinct_planes : α ≠ β

-- Statement to be proven
theorem problem_statement :
  perpendicular a b → 
  perpendicularLPlane a α → 
  perpendicularLPlane b β → 
  perpendicularPPlane α β :=
sorry

end problem_statement_l1186_118621


namespace train_length_calculation_l1186_118671

theorem train_length_calculation 
  (bridge_length : ℝ) (crossing_time : ℝ) (train_speed_kmph : ℝ) 
  (h_bridge_length : bridge_length = 150)
  (h_crossing_time : crossing_time = 25) 
  (h_train_speed_kmph : train_speed_kmph = 57.6) : 
  ∃ train_length, train_length = 250 :=
by
  sorry

end train_length_calculation_l1186_118671


namespace find_x_l1186_118683

theorem find_x (x : ℝ) (h : 3.5 * ( (3.6 * 0.48 * 2.50) / (x * 0.09 * 0.5) ) = 2800.0000000000005) : x = 0.3 :=
sorry

end find_x_l1186_118683


namespace smallest_number_meeting_both_conditions_l1186_118632

theorem smallest_number_meeting_both_conditions :
  ∃ n, (n = 2019) ∧
    (∃ a b c d e f : ℕ,
      n = a^4 + b^4 + c^4 + d^4 + e^4 ∧
      a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
      b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
      c ≠ d ∧ c ≠ e ∧
      d ≠ e ∧
      a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0) ∧
    (∃ x y z u v w : ℕ,
      y = x + 1 ∧ z = x + 2 ∧ u = x + 3 ∧ v = x + 4 ∧ w = x + 5 ∧
      n = x + y + z + u + v + w) ∧
    (¬ ∃ m, m < 2019 ∧
      (∃ a b c d e f : ℕ,
        m = a^4 + b^4 + c^4 + d^4 + e^4 ∧
        a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
        b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
        c ≠ d ∧ c ≠ e ∧
        d ≠ e ∧
        a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0) ∧
      (∃ x y z u v w : ℕ,
        y = x + 1 ∧ z = x + 2 ∧ u = x + 3 ∧ v = x + 4 ∧ w = x + 5 ∧
        m = x + y + z + u + v + w)) :=
by
  sorry

end smallest_number_meeting_both_conditions_l1186_118632


namespace boat_distance_downstream_l1186_118613

theorem boat_distance_downstream 
    (boat_speed_still : ℝ) 
    (stream_speed : ℝ) 
    (time_downstream : ℝ) 
    (distance_downstream : ℝ) 
    (h_boat_speed_still : boat_speed_still = 13) 
    (h_stream_speed : stream_speed = 6) 
    (h_time_downstream : time_downstream = 3.6315789473684212) 
    (h_distance_downstream : distance_downstream = 19 * 3.6315789473684212): 
    distance_downstream = 69 := 
by 
  have h_effective_speed : boat_speed_still + stream_speed = 19 := by 
    rw [h_boat_speed_still, h_stream_speed]; norm_num 
  rw [h_distance_downstream]; norm_num 
  sorry

end boat_distance_downstream_l1186_118613


namespace arithmetic_sequence_l1186_118611

variable (a : ℕ → ℕ)
variable (h : a 1 + 3 * a 8 + a 15 = 120)

theorem arithmetic_sequence (h : a 1 + 3 * a 8 + a 15 = 120) : a 2 + a 14 = 48 :=
sorry

end arithmetic_sequence_l1186_118611


namespace quadratic_transform_l1186_118604

theorem quadratic_transform : ∀ (x : ℝ), x^2 = 3 * x + 1 ↔ x^2 - 3 * x - 1 = 0 :=
by
  sorry

end quadratic_transform_l1186_118604


namespace kathryn_gave_56_pencils_l1186_118691

-- Define the initial and total number of pencils
def initial_pencils : ℕ := 9
def total_pencils : ℕ := 65

-- Define the number of pencils Kathryn gave to Anthony
def pencils_given : ℕ := total_pencils - initial_pencils

-- Prove that Kathryn gave Anthony 56 pencils
theorem kathryn_gave_56_pencils : pencils_given = 56 :=
by
  -- Proof is omitted as per the requirement
  sorry

end kathryn_gave_56_pencils_l1186_118691


namespace sin_13pi_over_6_equals_half_l1186_118623

noncomputable def sin_13pi_over_6 : ℝ := Real.sin (13 * Real.pi / 6)

theorem sin_13pi_over_6_equals_half : sin_13pi_over_6 = 1 / 2 := by
  sorry

end sin_13pi_over_6_equals_half_l1186_118623


namespace additional_money_needed_l1186_118605

/-- Mrs. Smith needs to calculate the additional money required after a discount -/
theorem additional_money_needed
  (initial_amount : ℝ) (ratio_more : ℝ) (discount_rate : ℝ) (final_amount_needed : ℝ) (additional_needed : ℝ)
  (h_initial : initial_amount = 500)
  (h_ratio : ratio_more = 2/5)
  (h_discount : discount_rate = 15/100)
  (h_total_needed : final_amount_needed = initial_amount * (1 + ratio_more) * (1 - discount_rate))
  (h_additional : additional_needed = final_amount_needed - initial_amount) :
  additional_needed = 95 :=
by 
  sorry

end additional_money_needed_l1186_118605


namespace smaller_successive_number_l1186_118626

noncomputable def solve_successive_numbers : ℕ :=
  let n := 51
  n

theorem smaller_successive_number (n : ℕ) (h : n * (n + 1) = 2652) : n = solve_successive_numbers :=
  sorry

end smaller_successive_number_l1186_118626


namespace zoo_visitors_per_hour_l1186_118677

theorem zoo_visitors_per_hour 
    (h1 : ∃ V, 0.80 * V = 320)
    (h2 : ∃ H : Nat, H = 8)
    : ∃ N : Nat, N = 50 :=
by
  sorry

end zoo_visitors_per_hour_l1186_118677


namespace garden_length_l1186_118654

theorem garden_length (w l : ℝ) (h1 : l = 2 * w) (h2 : 2 * l + 2 * w = 240) : l = 80 :=
by
  sorry

end garden_length_l1186_118654


namespace initial_average_age_l1186_118644

theorem initial_average_age (A : ℕ) (h1 : ∀ x : ℕ, 10 * A = 10 * A)
  (h2 : 5 * 17 + 10 * A = 15 * (A + 1)) : A = 14 :=
by 
  sorry

end initial_average_age_l1186_118644


namespace dog_years_second_year_l1186_118634

theorem dog_years_second_year (human_years : ℕ) :
  15 + human_years + 5 * 8 = 64 →
  human_years = 9 :=
by
  intro h
  sorry

end dog_years_second_year_l1186_118634


namespace rice_and_grain_separation_l1186_118652

theorem rice_and_grain_separation (total_weight : ℕ) (sample_size : ℕ) (non_rice_sample : ℕ) (non_rice_in_batch : ℕ) :
  total_weight = 1524 →
  sample_size = 254 →
  non_rice_sample = 28 →
  non_rice_in_batch = total_weight * non_rice_sample / sample_size →
  non_rice_in_batch = 168 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end rice_and_grain_separation_l1186_118652


namespace actors_duration_l1186_118673

-- Definition of conditions
def actors_at_a_time := 5
def total_actors := 20
def total_minutes := 60

-- Main statement to prove
theorem actors_duration : total_minutes / (total_actors / actors_at_a_time) = 15 := 
by
  sorry

end actors_duration_l1186_118673


namespace sum_powers_mod_7_l1186_118697

theorem sum_powers_mod_7 :
  (1^6 + 2^6 + 3^6 + 4^6 + 5^6 + 6^6) % 7 = 6 := by
  sorry

end sum_powers_mod_7_l1186_118697


namespace complement_of_union_eq_l1186_118699

-- Define the universal set U
def U : Set ℤ := {-1, 0, 1, 2, 3, 4}

-- Define the subset A
def A : Set ℤ := {-1, 0, 1}

-- Define the subset B
def B : Set ℤ := {0, 1, 2, 3}

-- Define the union of A and B
def A_union_B : Set ℤ := A ∪ B

-- Define the complement of A ∪ B in U
def complement_U_A_union_B : Set ℤ := U \ A_union_B

-- State the theorem to be proved
theorem complement_of_union_eq {U A B : Set ℤ} :
  U = {-1, 0, 1, 2, 3, 4} →
  A = {-1, 0, 1} →
  B = {0, 1, 2, 3} →
  complement_U_A_union_B = {4} :=
by
  intros hU hA hB
  sorry

end complement_of_union_eq_l1186_118699


namespace greg_ate_4_halves_l1186_118657

def greg_ate_halves (total_cookies : ℕ) (brad_halves : ℕ) (left_halves : ℕ) : ℕ :=
  2 * total_cookies - (brad_halves + left_halves)

theorem greg_ate_4_halves : greg_ate_halves 14 6 18 = 4 := by
  sorry

end greg_ate_4_halves_l1186_118657


namespace sufficient_not_necessary_condition_l1186_118655

theorem sufficient_not_necessary_condition (x : ℝ) : (1 < x ∧ x < 2) → (x < 2) ∧ ((x < 2) → ¬(1 < x ∧ x < 2)) :=
by
  sorry

end sufficient_not_necessary_condition_l1186_118655


namespace sum_coordinates_D_l1186_118666

theorem sum_coordinates_D
    (M : (ℝ × ℝ))
    (C : (ℝ × ℝ))
    (D : (ℝ × ℝ))
    (H_M_midpoint : M = (5, 9))
    (H_C_coords : C = (11, 5))
    (H_M_def : M = ((C.1 + D.1) / 2, (C.2 + D.2) / 2)) :
    (D.1 + D.2) = 12 := 
by
  sorry
 
end sum_coordinates_D_l1186_118666


namespace packs_bought_l1186_118618

theorem packs_bought (total_uncommon : ℕ) (cards_per_pack : ℕ) (fraction_uncommon : ℚ) 
  (total_packs : ℕ) (uncommon_per_pack : ℕ)
  (h1 : cards_per_pack = 20)
  (h2 : fraction_uncommon = 1/4)
  (h3 : uncommon_per_pack = fraction_uncommon * cards_per_pack)
  (h4 : total_uncommon = 50)
  (h5 : total_packs = total_uncommon / uncommon_per_pack)
  : total_packs = 10 :=
by 
  sorry

end packs_bought_l1186_118618


namespace function_has_property_T_l1186_118631

noncomputable def property_T (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), a ≠ b ∧ (f a ≠ 0) ∧ (f b ≠ 0) ∧ (f a * f b = -1)

theorem function_has_property_T : property_T (fun x => 1 + x * Real.log x) :=
sorry

end function_has_property_T_l1186_118631


namespace correct_operations_result_l1186_118635

/-
Pat intended to multiply a number by 8 but accidentally divided by 8.
Pat then meant to add 20 to the result but instead subtracted 20.
After these errors, the final outcome was 12.
Prove that if Pat had performed the correct operations, the final outcome would have been 2068.
-/

theorem correct_operations_result (n : ℕ) (h1 : n / 8 - 20 = 12) : 8 * n + 20 = 2068 :=
by
  sorry

end correct_operations_result_l1186_118635


namespace solve_a_minus_b_l1186_118696

theorem solve_a_minus_b (a b : ℝ) (h1 : 2010 * a + 2014 * b = 2018) (h2 : 2012 * a + 2016 * b = 2020) : a - b = -3 :=
sorry

end solve_a_minus_b_l1186_118696


namespace list_of_21_numbers_l1186_118608

theorem list_of_21_numbers (numbers : List ℝ) (n : ℝ) (h_length : numbers.length = 21) 
  (h_mem : n ∈ numbers) 
  (h_n_avg : n = 4 * (numbers.sum - n) / 20) 
  (h_n_sum : n = (numbers.sum) / 6) : numbers.length - 1 = 20 :=
by
  -- We provide the statement with the correct hypotheses
  -- the proof is yet to be filled in
  sorry

end list_of_21_numbers_l1186_118608


namespace products_selling_less_than_1000_l1186_118660

theorem products_selling_less_than_1000 (N: ℕ) 
  (total_products: ℕ := 25) 
  (average_price: ℤ := 1200) 
  (min_price: ℤ := 400) 
  (max_price: ℤ := 12000) 
  (total_revenue := total_products * average_price) 
  (revenue_from_expensive: ℤ := max_price):
  12000 + (24 - N) * 1000 + N * 400 = 30000 ↔ N = 10 :=
by
  sorry

end products_selling_less_than_1000_l1186_118660


namespace coordinates_with_respect_to_origin_l1186_118600

theorem coordinates_with_respect_to_origin (x y : ℤ) (hx : x = 3) (hy : y = -2) : (x, y) = (3, -2) :=
by
  sorry

end coordinates_with_respect_to_origin_l1186_118600


namespace range_of_a_l1186_118664

theorem range_of_a (x a : ℝ) (p : |x - 2| < 3) (q : 0 < x ∧ x < a) :
  (0 < a ∧ a ≤ 5) := 
sorry

end range_of_a_l1186_118664


namespace problem_l1186_118627

theorem problem (a b : ℕ) (h1 : 2^4 + 2^4 = 2^a) (h2 : 3^5 + 3^5 + 3^5 = 3^b) : a + b = 11 :=
by {
  sorry
}

end problem_l1186_118627


namespace smallest_positive_debt_resolvable_l1186_118658

theorem smallest_positive_debt_resolvable :
  ∃ (p g : ℤ), 400 * p + 280 * g = 800 :=
sorry

end smallest_positive_debt_resolvable_l1186_118658


namespace circle_center_radius_sum_l1186_118692

theorem circle_center_radius_sum :
  ∃ (c d s : ℝ), c = -6 ∧ d = -7 ∧ s = Real.sqrt 13 ∧
  (x^2 + 14 * y + 72 = -y^2 - 12 * x → c + d + s = -13 + Real.sqrt 13) :=
sorry

end circle_center_radius_sum_l1186_118692


namespace geometric_sequence_formula_l1186_118624

theorem geometric_sequence_formula (a : ℕ → ℝ)
  (q : ℝ)
  (h1 : a 1 = 3 / 2)
  (h2 : a 1 + a 1 * q + a 1 * q^2 = 9 / 2)
  (geo : ∀ n, a (n + 1) = a n * q) :
  ∀ n, a n = 3 / 2 * (-2)^(n-1) ∨ a n = 3 / 2 :=
by sorry

end geometric_sequence_formula_l1186_118624


namespace beth_score_l1186_118661

-- Conditions
variables (B : ℕ)  -- Beth's points are some number.
def jan_points := 10 -- Jan scored 10 points.
def judy_points := 8 -- Judy scored 8 points.
def angel_points := 11 -- Angel scored 11 points.

-- First team has 3 more points than the second team
def first_team_points := B + jan_points
def second_team_points := judy_points + angel_points
def first_team_more_than_second := first_team_points = second_team_points + 3

-- Statement: Prove that B = 12
theorem beth_score : first_team_more_than_second → B = 12 :=
by
  -- Proof will be provided here
  sorry

end beth_score_l1186_118661


namespace cos_comp_l1186_118674

open Real

theorem cos_comp {a b c : ℝ} (h1 : a = cos (3 / 2)) (h2 : b = -cos (7 / 4)) (h3 : c = sin (1 / 10)) : 
  a < c ∧ c < b := 
by
  -- Assume the hypotheses
  sorry

end cos_comp_l1186_118674


namespace bc_product_l1186_118676

theorem bc_product (b c : ℤ) : (∀ r : ℝ, r^2 - r - 2 = 0 → r^4 - b * r - c = 0) → b * c = 30 :=
by
  sorry

end bc_product_l1186_118676


namespace find_specific_M_in_S_l1186_118609

section MatrixProgression

variable {R : Type*} [CommRing R]

-- Definition of arithmetic progression in a 2x2 matrix.
def is_arithmetic_progression (a b c d : R) : Prop :=
  ∃ r : R, b = a + r ∧ c = a + 2 * r ∧ d = a + 3 * r

-- Definition of set S.
def S : Set (Matrix (Fin 2) (Fin 2) R) :=
  { M | ∃ a b c d : R, M = ![![a, b], ![c, d]] ∧ is_arithmetic_progression a b c d }

-- Main problem statement
theorem find_specific_M_in_S (M : Matrix (Fin 2) (Fin 2) ℝ) (k : ℕ) :
  k > 1 → M ∈ S → ∃ (α : ℝ), (M = α • ![![1, 1], ![1, 1]] ∨ (M = α • ![![ -3, -1], ![1, 3]] ∧ Odd k)) :=
by
  sorry

end MatrixProgression

end find_specific_M_in_S_l1186_118609


namespace expression_equals_500_l1186_118679

theorem expression_equals_500 :
  let A := 5 * 99 + 1
  let B := 100 + 25 * 4
  let C := 88 * 4 + 37 * 4
  let D := 100 * 0 * 5
  C = 500 :=
by
  let A := 5 * 99 + 1
  let B := 100 + 25 * 4
  let C := 88 * 4 + 37 * 4
  let D := 100 * 0 * 5
  sorry

end expression_equals_500_l1186_118679


namespace lindas_savings_l1186_118619

theorem lindas_savings :
  ∃ S : ℝ, (3 / 4 * S) + 150 = S ∧ (S - 150) = 3 / 4 * S := 
sorry

end lindas_savings_l1186_118619


namespace henley_initial_candies_l1186_118653

variables (C : ℝ)
variables (h1 : 0.60 * C = 180)

theorem henley_initial_candies : C = 300 :=
by sorry

end henley_initial_candies_l1186_118653


namespace number_of_combinations_of_planets_is_1141_l1186_118603

def number_of_combinations_of_planets : ℕ :=
  (if 7 ≥ 7 ∧ 8 ≥2 then Nat.choose 7 7 * Nat.choose 8 2 else 0) + 
  (if 7 ≥ 6 ∧ 8 ≥ 4 then Nat.choose 7 6 * Nat.choose 8 4 else 0) + 
  (if 7 ≥ 5 ∧ 8 ≥ 6 then Nat.choose 7 5 * Nat.choose 8 6 else 0) +
  (if 7 ≥ 4 ∧ 8 ≥ 8 then Nat.choose 7 4 * Nat.choose 8 8 else 0)

theorem number_of_combinations_of_planets_is_1141 :
  number_of_combinations_of_planets = 1141 :=
by
  sorry

end number_of_combinations_of_planets_is_1141_l1186_118603


namespace subsetneq_M_N_l1186_118645

def M : Set ℝ := {-1, 1}
def N : Set ℝ := {x | (x < 0) ∨ (x > 1 / 2)}

theorem subsetneq_M_N : M ⊂ N :=
by
  sorry

end subsetneq_M_N_l1186_118645


namespace smallest_rel_prime_210_l1186_118690

theorem smallest_rel_prime_210 : ∃ (y : ℕ), y > 1 ∧ Nat.gcd y 210 = 1 ∧ (∀ z : ℕ, z > 1 ∧ Nat.gcd z 210 = 1 → y ≤ z) ∧ y = 11 :=
by {
  sorry -- proof to be filled in
}

end smallest_rel_prime_210_l1186_118690


namespace percentage_neither_language_l1186_118602

noncomputable def total_diplomats : ℝ := 120
noncomputable def latin_speakers : ℝ := 20
noncomputable def russian_non_speakers : ℝ := 32
noncomputable def both_languages : ℝ := 0.10 * total_diplomats

theorem percentage_neither_language :
  let D := total_diplomats
  let L := latin_speakers
  let R := D - russian_non_speakers
  let LR := both_languages
  ∃ P, P = 100 * (D - (L + R - LR)) / D :=
by
  existsi ((total_diplomats - (latin_speakers + (total_diplomats - russian_non_speakers) - both_languages)) / total_diplomats * 100)
  sorry

end percentage_neither_language_l1186_118602


namespace reducible_fraction_least_n_l1186_118672

theorem reducible_fraction_least_n : ∃ n : ℕ, (0 < n) ∧ (n-15 > 0) ∧ (gcd (n-15) (3*n+4) > 1) ∧
  (∀ m : ℕ, (0 < m) ∧ (m-15 > 0) ∧ (gcd (m-15) (3*m+4) > 1) → n ≤ m) :=
by
  sorry

end reducible_fraction_least_n_l1186_118672


namespace car_distance_in_45_minutes_l1186_118606

theorem car_distance_in_45_minutes
  (train_speed : ℝ)
  (car_speed_ratio : ℝ)
  (time_minutes : ℝ)
  (h_train_speed : train_speed = 90)
  (h_car_speed_ratio : car_speed_ratio = 5 / 6)
  (h_time_minutes : time_minutes = 45) :
  ∃ d : ℝ, d = 56.25 ∧ d = (car_speed_ratio * train_speed) * (time_minutes / 60) :=
by
  sorry

end car_distance_in_45_minutes_l1186_118606


namespace men_required_l1186_118648

variable (m w : ℝ) -- Work done by one man and one woman in one day respectively
variable (x : ℝ) -- Number of men

-- Conditions from the problem
def condition1 (m w : ℝ) (x : ℝ) : Prop :=
  x * m = 12 * w

def condition2 (m w : ℝ) : Prop :=
  (6 * m + 11 * w) * 12 = 1

-- Proving that the number of men required to do the work in 20 days is x
theorem men_required (m w : ℝ) (x : ℝ) (h1 : condition1 m w x) (h2 : condition2 m w) : 
  (∃ x, condition1 m w x ∧ condition2 m w) := 
sorry

end men_required_l1186_118648


namespace min_colored_cells_65x65_l1186_118641

def grid_size : ℕ := 65
def total_cells : ℕ := grid_size * grid_size

-- Define a function that calculates the minimum number of colored cells needed
noncomputable def min_colored_cells_needed (N: ℕ) : ℕ := (N * N) / 3

-- The main theorem stating the proof problem
theorem min_colored_cells_65x65 (H: grid_size = 65) : 
  min_colored_cells_needed grid_size = 1408 :=
by {
  sorry
}

end min_colored_cells_65x65_l1186_118641


namespace part_1_part_2_l1186_118675

def f (x a : ℝ) : ℝ := abs (x - a) + abs (2 * x + 4)

theorem part_1 (a : ℝ) (h : a = 3) :
  { x : ℝ | f x a ≥ 8 } = { x : ℝ | x ≤ -3 } ∪ { x : ℝ | 1 ≤ x ∧ x ≤ 3 } ∪ { x : ℝ | x > 3 } := 
sorry

theorem part_2 (h : ∃ x : ℝ, f x a - abs (x + 2) ≤ 4) :
  -6 ≤ a ∧ a ≤ 2 :=
sorry

end part_1_part_2_l1186_118675


namespace regular_tetrahedron_l1186_118670

-- Define the types for points and tetrahedrons
structure Point :=
(x : ℝ)
(y : ℝ)
(z : ℝ)

structure Tetrahedron :=
(A B C D : Point)
(insphere : Point)

-- Conditions
def sphere_touches_at_angle_bisectors (T : Tetrahedron) : Prop :=
-- Dummy implementation to define the condition (to be filled)
sorry

def sphere_touches_at_altitudes (T : Tetrahedron) : Prop :=
-- Dummy implementation to define the condition (to be filled)
sorry

def sphere_touches_at_medians (T : Tetrahedron) : Prop :=
-- Dummy implementation to define the condition (to be filled)
sorry

-- Main theorem statement
theorem regular_tetrahedron (T : Tetrahedron)
  (h1 : sphere_touches_at_angle_bisectors T)
  (h2 : sphere_touches_at_altitudes T)
  (h3 : sphere_touches_at_medians T) :
  T.A = T.B ∧ T.A = T.C ∧ T.A = T.D := 
sorry

end regular_tetrahedron_l1186_118670


namespace vacation_cost_split_l1186_118656

theorem vacation_cost_split 
  (airbnb_cost : ℕ)
  (car_rental_cost : ℕ)
  (people : ℕ)
  (split_equally : Prop)
  (h1 : airbnb_cost = 3200)
  (h2 : car_rental_cost = 800)
  (h3 : people = 8)
  (h4 : split_equally)
  : (airbnb_cost + car_rental_cost) / people = 500 :=
by
  sorry

end vacation_cost_split_l1186_118656


namespace tangent_line_at_x_2_range_of_m_for_three_roots_l1186_118665

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 + 3

/-
Part 1: Proving the tangent line equation at x = 2
-/
theorem tangent_line_at_x_2 : ∃ k b, (k = 12) ∧ (b = -17) ∧ 
  (∀ x, 12 * x - f 2 - 17 = 0) :=
by
  sorry

/-
Part 2: Proving the range of m for three distinct real roots
-/
theorem range_of_m_for_three_roots (m : ℝ) :
  (∃ x1 x2 x3, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ f x1 + m = 0 ∧ f x2 + m = 0 ∧ f x3 + m = 0) ↔ 
  -3 < m ∧ m < -2 :=
by
  sorry

end tangent_line_at_x_2_range_of_m_for_three_roots_l1186_118665


namespace seohyun_initial_marbles_l1186_118687

variable (M : ℤ)

theorem seohyun_initial_marbles (h1 : (2 / 3) * M = 12) (h2 : (1 / 2) * M + 12 = M) : M = 36 :=
sorry

end seohyun_initial_marbles_l1186_118687

import Mathlib

namespace cover_tiles_count_l105_105192

-- Definitions corresponding to the conditions
def tile_side : ℕ := 6 -- in inches
def tile_area : ℕ := tile_side * tile_side -- area of one tile in square inches

def region_length : ℕ := 3 * 12 -- 3 feet in inches
def region_width : ℕ := 6 * 12 -- 6 feet in inches
def region_area : ℕ := region_length * region_width -- area of the region in square inches

-- The statement of the proof problem
theorem cover_tiles_count : (region_area / tile_area) = 72 :=
by
   -- Proof would be filled in here
   sorry

end cover_tiles_count_l105_105192


namespace ratio_Bipin_Alok_l105_105174

-- Definitions based on conditions
def Alok_age : Nat := 5
def Chandan_age : Nat := 10
def Bipin_age : Nat := 30
def Bipin_age_condition (B C : Nat) : Prop := B + 10 = 2 * (C + 10)

-- Statement to prove
theorem ratio_Bipin_Alok : 
  Bipin_age_condition Bipin_age Chandan_age -> 
  Alok_age = 5 -> 
  Chandan_age = 10 -> 
  Bipin_age / Alok_age = 6 :=
by
  sorry

end ratio_Bipin_Alok_l105_105174


namespace not_divisible_a1a2_l105_105526

theorem not_divisible_a1a2 (a1 a2 b1 b2 : ℕ) (h1 : 1 < b1) (h2 : b1 < a1) (h3 : 1 < b2) (h4 : b2 < a2) (h5 : b1 ∣ a1) (h6 : b2 ∣ a2) :
  ¬ (a1 * a2 ∣ a1 * b1 + a2 * b2 - 1) :=
by
  sorry

end not_divisible_a1a2_l105_105526


namespace power_equality_l105_105348

theorem power_equality (x : ℝ) (n : ℕ) (h : x^(2 * n) = 3) : x^(4 * n) = 9 := 
by 
  sorry

end power_equality_l105_105348


namespace cylinder_radius_in_cone_l105_105830

theorem cylinder_radius_in_cone (d h r : ℝ) (h_d : d = 20) (h_h : h = 24) (h_cylinder : 2 * r = r):
  r = 60 / 11 :=
by
  sorry

end cylinder_radius_in_cone_l105_105830


namespace remainder_poly_l105_105055

theorem remainder_poly (x : ℂ) (h : x^5 + x^4 + x^3 + x^2 + x + 1 = 0) :
  (x^75 + x^60 + x^45 + x^30 + x^15 + 1) % (x^5 + x^4 + x^3 + x^2 + x + 1) = 0 :=
by sorry

end remainder_poly_l105_105055


namespace cost_of_one_dozen_pens_l105_105534

-- Define the initial conditions
def cost_pen : ℕ := 65
def cost_pencil := cost_pen / 5
def total_cost (pencils : ℕ) := 3 * cost_pen + pencils * cost_pencil

-- State the theorem
theorem cost_of_one_dozen_pens (pencils : ℕ) (h : total_cost pencils = 260) :
  12 * cost_pen = 780 :=
by
  -- Preamble to show/conclude that the proofs are given
  sorry

end cost_of_one_dozen_pens_l105_105534


namespace max_plus_min_eq_four_l105_105107

theorem max_plus_min_eq_four {g : ℝ → ℝ} (h_odd_function : ∀ x, g (-x) = -g x)
  (M m : ℝ) (h_f : ∀ x, 2 + g x ≤ M) (h_f' : ∀ x, m ≤ 2 + g x) :
  M + m = 4 :=
by
  sorry

end max_plus_min_eq_four_l105_105107


namespace rakesh_salary_l105_105732

variable (S : ℝ) -- The salary S is a real number
variable (h : 0.595 * S = 2380) -- Condition derived from the problem

theorem rakesh_salary : S = 4000 :=
by
  sorry

end rakesh_salary_l105_105732


namespace exist_n_exactly_3_rainy_days_l105_105768

-- Define the binomial coefficient
def binomial_coeff (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Define the binomial probability
def binomial_prob (n k : ℕ) (p : ℝ) : ℝ :=
  (binomial_coeff n k : ℝ) * p^k * (1 - p)^(n - k)

theorem exist_n_exactly_3_rainy_days (p : ℝ) (k : ℕ) (prob : ℝ) :
  p = 0.5 → k = 3 → prob = 0.25 →
  ∃ n : ℕ, binomial_prob n k p = prob :=
by
  intros h1 h2 h3
  sorry

end exist_n_exactly_3_rainy_days_l105_105768


namespace bet_final_result_l105_105014

theorem bet_final_result :
  let M₀ := 64
  let final_money := (3 / 2) ^ 3 * (1 / 2) ^ 3 * M₀
  final_money = 27 ∧ M₀ - final_money = 37 :=
by
  sorry

end bet_final_result_l105_105014


namespace even_n_divisible_into_equal_triangles_l105_105235

theorem even_n_divisible_into_equal_triangles (n : ℕ) (hn : 3 < n) :
  (∃ (triangles : ℕ), triangles = n) ↔ (∃ (k : ℕ), n = 2 * k) := 
sorry

end even_n_divisible_into_equal_triangles_l105_105235


namespace max_money_received_back_l105_105217

def total_money_before := 3000
def value_chip_20 := 20
def value_chip_100 := 100
def chips_lost_total := 16
def chips_lost_diff_1 (x y : ℕ) := x = y + 2
def chips_lost_diff_2 (x y : ℕ) := x = y - 2

theorem max_money_received_back :
  ∃ (x y : ℕ), 
  (chips_lost_diff_1 x y ∨ chips_lost_diff_2 x y) ∧ 
  (x + y = chips_lost_total) ∧
  total_money_before - (x * value_chip_20 + y * value_chip_100) = 2120 :=
sorry

end max_money_received_back_l105_105217


namespace inequality_AM_GM_HM_l105_105322

theorem inequality_AM_GM_HM (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (hab : a ≠ b) : 
  (a + b) / 2 > Real.sqrt (a * b) ∧ Real.sqrt (a * b) > 2 * (a * b) / (a + b) :=
by
  sorry

end inequality_AM_GM_HM_l105_105322


namespace problem1_l105_105239

theorem problem1 (a b : ℤ) (h1 : abs a = 5) (h2 : abs b = 3) (h3 : abs (a - b) = b - a) : a - b = -8 ∨ a - b = -2 := by 
  sorry

end problem1_l105_105239


namespace find_length_of_EF_l105_105602

-- Definitions based on conditions
noncomputable def AB : ℝ := 300
noncomputable def DC : ℝ := 180
noncomputable def BC : ℝ := 200
noncomputable def E_as_fraction_of_BC : ℝ := (3 / 5)

-- Derived definition based on given conditions
noncomputable def EB : ℝ := E_as_fraction_of_BC * BC
noncomputable def EC : ℝ := BC - EB
noncomputable def EF : ℝ := (EC / BC) * DC

-- The theorem we need to prove
theorem find_length_of_EF : EF = 72 := by
  sorry

end find_length_of_EF_l105_105602


namespace compute_expression_l105_105670

-- Given Conditions
variables (a b c : ℕ)
variable (h : 2^a * 3^b * 5^c = 36000)

-- Proof Statement
theorem compute_expression (h : 2^a * 3^b * 5^c = 36000) : 3 * a + 4 * b + 6 * c = 41 :=
sorry

end compute_expression_l105_105670


namespace number_of_blue_spotted_fish_l105_105367

theorem number_of_blue_spotted_fish : 
  ∀ (fish_total : ℕ) (one_third_blue : ℕ) (half_spotted : ℕ),
    fish_total = 30 →
    one_third_blue = fish_total / 3 →
    half_spotted = one_third_blue / 2 →
    half_spotted = 5 := 
by
  intros fish_total one_third_blue half_spotted ht htb hhs
  sorry

end number_of_blue_spotted_fish_l105_105367


namespace camila_weeks_to_goal_l105_105701

open Nat

noncomputable def camila_hikes : ℕ := 7
noncomputable def amanda_hikes : ℕ := 8 * camila_hikes
noncomputable def steven_hikes : ℕ := amanda_hikes + 15
noncomputable def additional_hikes_needed : ℕ := steven_hikes - camila_hikes
noncomputable def hikes_per_week : ℕ := 4
noncomputable def weeks_to_goal : ℕ := additional_hikes_needed / hikes_per_week

theorem camila_weeks_to_goal : weeks_to_goal = 16 :=
  by sorry

end camila_weeks_to_goal_l105_105701


namespace suzanna_distance_ridden_l105_105150

theorem suzanna_distance_ridden (rate_per_5minutes : ℝ) (time_minutes : ℕ) (total_distance : ℝ) (units_per_interval : ℕ) (interval_distance : ℝ) :
  rate_per_5minutes = 0.75 → time_minutes = 45 → units_per_interval = 5 → interval_distance = 0.75 → total_distance = (time_minutes / units_per_interval) * interval_distance → total_distance = 6.75 := 
by
  intros h1 h2 h3 h4 h5
  sorry

end suzanna_distance_ridden_l105_105150


namespace correct_ignition_time_l105_105942

noncomputable def ignition_time_satisfying_condition (initial_length : ℝ) (l : ℝ) : ℕ :=
  let burn_rate1 := l / 240
  let burn_rate2 := l / 360
  let stub1 t := l - burn_rate1 * t
  let stub2 t := l - burn_rate2 * t
  let stub_length_condition t := stub2 t = 3 * stub1 t
  let time_difference_at_6AM := 360 -- 6 AM is 360 minutes after midnight
  360 - 180 -- time to ignite the candles

theorem correct_ignition_time : ignition_time_satisfying_condition l 6 = 180 := 
by sorry

end correct_ignition_time_l105_105942


namespace smallest_number_is_42_l105_105719

theorem smallest_number_is_42 (x : ℤ) 
  (h1 : x + (x + 1) + (x + 2) + (x + 3) + (x + 4) = 225)
  (h2 : x % 7 = 0) : 
  x = 42 := 
sorry

end smallest_number_is_42_l105_105719


namespace correct_options_l105_105384

variable (f : ℝ → ℝ)

-- Conditions
def is_even_function : Prop := ∀ x : ℝ, f x = f (-x)
def function_definition : Prop := ∀ x : ℝ, (0 < x) → f x = x^2 + x

-- Statements to be proved
def option_A : Prop := f (-1) = 2
def option_B_incorrect : Prop := ¬ (∀ x : ℝ, (f x ≥ f 0) ↔ x ≥ 0) -- Reformulated as not a correct statement
def option_C : Prop := ∀ x : ℝ, x < 0 → f x = x^2 - x
def option_D : Prop := ∀ x : ℝ, (0 < x ∧ x < 2) ↔ f (x - 1) < 2

-- Prove that the correct statements are A, C, and D
theorem correct_options (h_even : is_even_function f) (h_def : function_definition f) :
  option_A f ∧ option_C f ∧ option_D f := by
  sorry

end correct_options_l105_105384


namespace find_extrema_l105_105342

noncomputable def function_extrema (x : ℝ) : ℝ :=
  (2 / 3) * Real.cos (3 * x - Real.pi / 6)

theorem find_extrema :
  (function_extrema (Real.pi / 18) = 2 / 3 ∧
   function_extrema (7 * Real.pi / 18) = -(2 / 3)) ∧
  (0 < Real.pi / 18 ∧ Real.pi / 18 < Real.pi / 2) ∧
  (0 < 7 * Real.pi / 18 ∧ 7 * Real.pi / 18 < Real.pi / 2) :=
by
  sorry

end find_extrema_l105_105342


namespace closed_path_even_length_l105_105705

def is_closed_path (steps : List Char) : Bool :=
  let net_vertical := steps.count 'U' - steps.count 'D'
  let net_horizontal := steps.count 'R' - steps.count 'L'
  net_vertical = 0 ∧ net_horizontal = 0

def move_length (steps : List Char) : Nat :=
  steps.length

theorem closed_path_even_length (steps : List Char) :
  is_closed_path steps = true → move_length steps % 2 = 0 :=
by
  -- Conditions extracted as definitions
  intros h
  -- The proof will handle showing that the length of the closed path is even
  sorry

end closed_path_even_length_l105_105705


namespace minimum_value_sum_l105_105621

theorem minimum_value_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  (a / (b + 3 * c) + b / (8 * c + 4 * a) + 9 * c / (3 * a + 2 * b)) ≥ 47 / 48 :=
by sorry

end minimum_value_sum_l105_105621


namespace Tony_fills_pool_in_90_minutes_l105_105060

def minutes (r : ℚ) : ℚ := 1 / r

theorem Tony_fills_pool_in_90_minutes (J S T : ℚ) 
  (hJ : J = 1 / 30)       -- Jim's rate in pools per minute
  (hS : S = 1 / 45)       -- Sue's rate in pools per minute
  (h_combined : J + S + T = 1 / 15) -- Combined rate of all three

  : minutes T = 90 :=     -- Tony can fill the pool alone in 90 minutes
by sorry

end Tony_fills_pool_in_90_minutes_l105_105060


namespace range_of_m_l105_105873

noncomputable def f (x : ℝ) : ℝ := 3 * x + Real.sin x

theorem range_of_m (m : ℝ) (h : f (2 * m - 1) + f (3 - m) > 0) : m > -2 :=
by
  sorry

end range_of_m_l105_105873


namespace division_decomposition_l105_105835

theorem division_decomposition (a b : ℕ) (h₁ : a = 36) (h₂ : b = 3)
    (h₃ : 30 / b = 10) (h₄ : 6 / b = 2) (h₅ : 10 + 2 = 12) :
    a / b = (30 / b) + (6 / b) := 
sorry

end division_decomposition_l105_105835


namespace find_e_l105_105972

theorem find_e (a b c d e : ℝ) (h1 : a < b) (h2 : b < c) (h3 : c < d) (h4 : d < e)
    (h_lb1 : a + b = 32) (h_lb2 : a + c = 36) (h_lb3 : b + c = 37)
    (h_ub1 : c + e = 48) (h_ub2 : d + e = 51) : e = 27.5 :=
sorry

end find_e_l105_105972


namespace guessing_game_l105_105872

-- Define the conditions
def number : ℕ := 33
def result : ℕ := 2 * 51 - 3

-- Define the factor (to be proven)
def factor (n r : ℕ) : ℕ := r / n

-- The theorem to be proven
theorem guessing_game (n r : ℕ) (h1 : n = 33) (h2 : r = 2 * 51 - 3) : 
  factor n r = 3 := by
  -- Placeholder for the actual proof
  sorry

end guessing_game_l105_105872


namespace find_parabola_l105_105737

variable (P : ℝ × ℝ)
variable (a b : ℝ)

def parabola1 (P : ℝ × ℝ) (a : ℝ) := P.2^2 = 4 * a * P.1
def parabola2 (P : ℝ × ℝ) (b : ℝ) := P.1^2 = 4 * b * P.2

theorem find_parabola (hP : P = (-2, 4)) :
  (∃ a, parabola1 P a ∧ P.2^2 = -8 * P.1) ∨ 
  (∃ b, parabola2 P b ∧ P.1^2 = P.2) := by
  sorry

end find_parabola_l105_105737


namespace min_colors_for_grid_coloring_l105_105017

theorem min_colors_for_grid_coloring : ∃c : ℕ, c = 4 ∧ (∀ (color : ℕ × ℕ → ℕ), 
  (∀ i j : ℕ, i < 5 ∧ j < 5 → 
     ((i < 4 → color (i, j) ≠ color (i+1, j+1)) ∧ 
      (j < 4 → color (i, j) ≠ color (i+1, j-1))) ∧ 
     ((i > 0 → color (i, j) ≠ color (i-1, j-1)) ∧ 
      (j > 0 → color (i, j) ≠ color (i-1, j+1)))) → 
  c = 4) :=
sorry

end min_colors_for_grid_coloring_l105_105017


namespace sample_capacity_l105_105883

theorem sample_capacity (f : ℕ) (r : ℚ) (n : ℕ) (h₁ : f = 40) (h₂ : r = 0.125) (h₃ : r * n = f) : n = 320 :=
sorry

end sample_capacity_l105_105883


namespace dogwood_trees_after_5_years_l105_105843

theorem dogwood_trees_after_5_years :
  let current_trees := 39
  let trees_planted_today := 41
  let growth_rate_today := 2 -- trees per year
  let trees_planted_tomorrow := 20
  let growth_rate_tomorrow := 4 -- trees per year
  let years := 5
  let total_planted_trees := trees_planted_today + trees_planted_tomorrow
  let total_initial_trees := current_trees + total_planted_trees
  let total_growth_today := growth_rate_today * years
  let total_growth_tomorrow := growth_rate_tomorrow * years
  let total_growth := total_growth_today + total_growth_tomorrow
  let final_tree_count := total_initial_trees + total_growth
  final_tree_count = 130 := by
  sorry

end dogwood_trees_after_5_years_l105_105843


namespace smallest_prime_with_digit_sum_25_l105_105288

-- Definitions used in Lean statement:
-- 1. Prime predicate based on primality check.
-- 2. Digit sum function.

def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Lean 4 statement to prove that the smallest prime whose digits sum to 25 is 1699.

theorem smallest_prime_with_digit_sum_25 : ∃ n : ℕ, is_prime n ∧ digit_sum n = 25 ∧ n = 1699 :=
by
  sorry

end smallest_prime_with_digit_sum_25_l105_105288


namespace not_multiple_of_3_l105_105975

noncomputable def exists_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n*(n + 3) = m^2

theorem not_multiple_of_3 
  (n : ℕ) (h1 : 0 < n) (h2 : exists_perfect_square n) : ¬ ∃ k : ℕ, n = 3 * k := 
sorry

end not_multiple_of_3_l105_105975


namespace greatest_integer_less_than_PS_l105_105796

noncomputable def PS := (150 * Real.sqrt 2)

theorem greatest_integer_less_than_PS
  (PQ RS : ℝ)
  (PS : ℝ := PQ * Real.sqrt 2)
  (h₁ : PQ = 150)
  (h_midpoint : PS / 2 = PQ) :
  ∀ n : ℤ, n < PS → n = 212 :=
by
  -- Proof to be completed later
  sorry

end greatest_integer_less_than_PS_l105_105796


namespace clive_change_l105_105111

theorem clive_change (money : ℝ) (olives_needed : ℕ) (olives_per_jar : ℕ) (price_per_jar : ℝ) : 
  (money = 10) → 
  (olives_needed = 80) → 
  (olives_per_jar = 20) →
  (price_per_jar = 1.5) →
  money - (olives_needed / olives_per_jar) * price_per_jar = 4 := by
  sorry

end clive_change_l105_105111


namespace find_percentage_l105_105629

theorem find_percentage (P N : ℕ) (h₁ : N = 125) (h₂ : N = (P * N / 100) + 105) : P = 16 :=
by
  sorry

end find_percentage_l105_105629


namespace segment_length_aa_prime_l105_105330

/-- Given points A, B, and C, and their reflections, show that the length of AA' is 8 -/
theorem segment_length_aa_prime
  (A : ℝ × ℝ) (A_reflected : ℝ × ℝ)
  (x₁ y₁ y₁_neg : ℝ) :
  A = (x₁, y₁) →
  A_reflected = (x₁, y₁_neg) →
  y₁_neg = -y₁ →
  y₁ = 4 →
  x₁ = 2 →
  |y₁ - y₁_neg| = 8 :=
sorry

end segment_length_aa_prime_l105_105330


namespace pythagorean_inequality_l105_105312

variables (a b c : ℝ) (n : ℕ)

theorem pythagorean_inequality (h₀ : a > b) (h₁ : b > c) (h₂ : a^2 = b^2 + c^2) (h₃ : n > 2) : a^n > b^n + c^n :=
sorry

end pythagorean_inequality_l105_105312


namespace ratio_of_hair_lengths_l105_105897

theorem ratio_of_hair_lengths 
  (logan_hair : ℕ)
  (emily_hair : ℕ)
  (kate_hair : ℕ)
  (h1 : logan_hair = 20)
  (h2 : emily_hair = logan_hair + 6)
  (h3 : kate_hair = 7)
  : kate_hair / emily_hair = 7 / 26 :=
by sorry

end ratio_of_hair_lengths_l105_105897


namespace charlie_share_l105_105583

variable (A B C : ℝ)

theorem charlie_share :
  A = (1/3) * B →
  B = (1/2) * C →
  A + B + C = 10000 →
  C = 6000 :=
by
  intros hA hB hSum
  sorry

end charlie_share_l105_105583


namespace functional_eq_solution_l105_105669

theorem functional_eq_solution (g : ℝ → ℝ)
  (h : ∀ x y : ℝ, g x + g (3 * x + y) + 7 * x * y = g (4 * x - y) + 3 * x^2 + 2) :
  g 10 = -48 :=
sorry

end functional_eq_solution_l105_105669


namespace percentage_given_to_close_friends_l105_105366

-- Definitions
def total_boxes : ℕ := 20
def pens_per_box : ℕ := 5
def total_pens : ℕ := total_boxes * pens_per_box
def pens_left_after_classmates : ℕ := 45

-- Proposition
theorem percentage_given_to_close_friends (P : ℝ) :
  total_boxes = 20 → pens_per_box = 5 → pens_left_after_classmates = 45 →
  (3 / 4) * (100 - P) = (pens_left_after_classmates : ℝ) →
  P = 40 :=
by
  intros h_total_boxes h_pens_per_box h_pens_left_after h_eq
  sorry

end percentage_given_to_close_friends_l105_105366


namespace total_figurines_l105_105866

theorem total_figurines:
  let basswood_blocks := 25
  let butternut_blocks := 30
  let aspen_blocks := 35
  let oak_blocks := 40
  let cherry_blocks := 45
  let basswood_figs_per_block := 3
  let butternut_figs_per_block := 4
  let aspen_figs_per_block := 2 * basswood_figs_per_block
  let oak_figs_per_block := 5
  let cherry_figs_per_block := 7
  let basswood_total := basswood_blocks * basswood_figs_per_block
  let butternut_total := butternut_blocks * butternut_figs_per_block
  let aspen_total := aspen_blocks * aspen_figs_per_block
  let oak_total := oak_blocks * oak_figs_per_block
  let cherry_total := cherry_blocks * cherry_figs_per_block
  let total_figs := basswood_total + butternut_total + aspen_total + oak_total + cherry_total
  total_figs = 920 := by sorry

end total_figurines_l105_105866


namespace age_difference_l105_105148

-- defining the conditions
variable (A B : ℕ)
variable (h1 : B = 35)
variable (h2 : A + 10 = 2 * (B - 10))

-- the proof statement
theorem age_difference : A - B = 5 :=
by
  sorry

end age_difference_l105_105148


namespace village_population_l105_105757

theorem village_population (P : ℝ) (h1 : 0.08 * P = 4554) : P = 6325 :=
by
  sorry

end village_population_l105_105757


namespace handshake_problem_7_boys_21_l105_105648

theorem handshake_problem_7_boys_21 :
  let n := 7
  let total_handshakes := n * (n - 1) / 2
  total_handshakes = 21 → (n - 1) = 6 :=
by
  -- Let n be the number of boys (7 in this case)
  let n := 7
  
  -- Define the total number of handshakes equation
  let total_handshakes := n * (n - 1) / 2
  
  -- Assume the total number of handshakes is 21
  intro h
  -- Proof steps would go here
  sorry

end handshake_problem_7_boys_21_l105_105648


namespace scientific_notation_150_billion_l105_105961

theorem scientific_notation_150_billion : 150000000000 = 1.5 * 10^11 :=
sorry

end scientific_notation_150_billion_l105_105961


namespace max_edges_partitioned_square_l105_105842

theorem max_edges_partitioned_square (n v e : ℕ) 
  (h : v - e + n = 1) : e ≤ 3 * n + 1 := 
sorry

end max_edges_partitioned_square_l105_105842


namespace unique_real_x_satisfies_eq_l105_105818

theorem unique_real_x_satisfies_eq (x : ℝ) (h : x ≠ 0) : (7 * x)^5 = (14 * x)^4 ↔ x = 16 / 7 :=
by sorry

end unique_real_x_satisfies_eq_l105_105818


namespace shaded_areas_total_l105_105809

theorem shaded_areas_total (r R : ℝ) (h_divides : ∀ (A : ℝ), ∃ (B : ℝ), B = A / 3)
  (h_center : True) (h_area : π * R^2 = 81 * π) :
  (π * R^2 / 3) + (π * (R / 2)^2 / 3) = 33.75 * π :=
by
  -- The proof here will be added.
  sorry

end shaded_areas_total_l105_105809


namespace exists_positive_n_with_m_zeros_l105_105247

theorem exists_positive_n_with_m_zeros (m : ℕ) (hm : 0 < m) :
  ∃ n : ℕ, 0 < n ∧ ∃ k : ℕ, 7^n = k * 10^m :=
sorry

end exists_positive_n_with_m_zeros_l105_105247


namespace isoland_license_plates_proof_l105_105281

def isoland_license_plates : ℕ :=
  let letters := ['A', 'B', 'D', 'E', 'I', 'L', 'N', 'O', 'R', 'U']
  let valid_letters := letters.erase 'B'
  let first_letter_choices := ['A', 'I']
  let last_letter := 'R'
  let remaining_letters:= valid_letters.erase last_letter
  (first_letter_choices.length * (remaining_letters.length - first_letter_choices.length) * (remaining_letters.length - first_letter_choices.length - 1) * (remaining_letters.length - first_letter_choices.length - 2))

theorem isoland_license_plates_proof :
  isoland_license_plates = 420 := by
  sorry

end isoland_license_plates_proof_l105_105281


namespace positive_roots_implies_nonnegative_m_l105_105022

variables {x1 x2 m : ℝ}

theorem positive_roots_implies_nonnegative_m (h1 : x1 > 0) (h2 : x2 > 0)
  (h3 : x1 * x2 = 1) (h4 : x1 + x2 = m + 2) : m ≥ 0 :=
by
  sorry

end positive_roots_implies_nonnegative_m_l105_105022


namespace range_of_x_l105_105916

noncomputable def f (x : ℝ) : ℝ := Real.exp x + x^3

theorem range_of_x (x : ℝ) (h : f (x^2) < f (3*x - 2)) : 1 < x ∧ x < 2 :=
by
  sorry

end range_of_x_l105_105916


namespace regular_tiles_area_l105_105075

theorem regular_tiles_area (L W : ℝ) (T : ℝ) (h₁ : 1/3 * T * (3 * L * W) + 2/3 * T * (L * W) = 385) : 
  (2/3 * T * (L * W) = 154) :=
by
  sorry

end regular_tiles_area_l105_105075


namespace regular_polygon_sides_l105_105903

theorem regular_polygon_sides (interior_angle exterior_angle : ℕ)
  (h1 : interior_angle = exterior_angle + 60)
  (h2 : interior_angle + exterior_angle = 180) :
  ∃ n : ℕ, n = 6 :=
by
  have ext_angle_eq : exterior_angle = 60 := sorry
  have ext_angles_sum : exterior_angle * 6 = 360 := sorry
  exact ⟨6, by linarith⟩

end regular_polygon_sides_l105_105903


namespace greatest_common_divisor_is_one_l105_105426

-- Define the expressions for a and b
def a : ℕ := 114^2 + 226^2 + 338^2
def b : ℕ := 113^2 + 225^2 + 339^2

-- Now state that the gcd of a and b is 1
theorem greatest_common_divisor_is_one : Nat.gcd a b = 1 := sorry

end greatest_common_divisor_is_one_l105_105426


namespace parabola_point_distance_l105_105625

open Real

noncomputable def parabola_coords (y z: ℝ) : Prop :=
  y^2 = 12 * z

noncomputable def distance (x1 y1 x2 y2: ℝ) : ℝ :=
  sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

theorem parabola_point_distance (x y: ℝ) :
  parabola_coords y x ∧ distance x y 3 0 = 9 ↔ ( x = 6 ∧ (y = 6 * sqrt 2 ∨ y = -6 * sqrt 2)) :=
by
  sorry

end parabola_point_distance_l105_105625


namespace number_of_non_Speedsters_l105_105403

theorem number_of_non_Speedsters (V : ℝ) (h0 : (4 / 15) * V = 12) : (2 / 3) * V = 30 :=
by
  -- The conditions are such that:
  -- V is the total number of vehicles.
  -- (4 / 15) * V = 12 means 4/5 of 1/3 of the total vehicles are convertibles.
  -- We need to prove that 2/3 of the vehicles are not Speedsters.
  sorry

end number_of_non_Speedsters_l105_105403


namespace pow_99_square_pow_neg8_mult_l105_105224

theorem pow_99_square :
  99^2 = 9801 := 
by
  -- Proof omitted
  sorry

theorem pow_neg8_mult :
  (-8) ^ 2009 * (-1/8) ^ 2008 = -8 :=
by
  -- Proof omitted
  sorry

end pow_99_square_pow_neg8_mult_l105_105224


namespace find_flour_amount_l105_105681

variables (F S C : ℕ)

-- Condition 1: Proportions must remain constant
axiom proportion : 11 * S = 7 * F ∧ 7 * C = 5 * S

-- Condition 2: Mary needs 2 more cups of flour than sugar
axiom flour_sugar : F = S + 2

-- Condition 3: Mary needs 1 more cup of sugar than cocoa powder
axiom sugar_cocoa : S = C + 1

-- Question: How many cups of flour did she put in?
theorem find_flour_amount : F = 8 :=
by
  sorry

end find_flour_amount_l105_105681


namespace sphere_surface_area_increase_l105_105440

theorem sphere_surface_area_increase (r : ℝ) (h_r_pos : 0 < r):
  let A := 4 * π * r ^ 2
  let r' := 1.10 * r
  let A' := 4 * π * (r') ^ 2
  let ΔA := A' - A
  (ΔA / A) * 100 = 21 := by
  sorry

end sphere_surface_area_increase_l105_105440


namespace lcm_gcd_eq_product_l105_105917

theorem lcm_gcd_eq_product {a b : ℕ} (h : Nat.lcm a b + Nat.gcd a b = a * b) : a = 2 ∧ b = 2 :=
  sorry

end lcm_gcd_eq_product_l105_105917


namespace spherical_to_rectangular_coordinates_l105_105761

noncomputable
def convert_to_cartesian (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

theorem spherical_to_rectangular_coordinates :
  let ρ1 := 10
  let θ1 := Real.pi / 4
  let φ1 := Real.pi / 6
  let ρ2 := 15
  let θ2 := 5 * Real.pi / 4
  let φ2 := Real.pi / 3
  convert_to_cartesian ρ1 θ1 φ1 = (5 * Real.sqrt 2 / 2, 5 * Real.sqrt 2 / 2, 5 * Real.sqrt 3)
  ∧ convert_to_cartesian ρ2 θ2 φ2 = (-15 * Real.sqrt 6 / 4, -15 * Real.sqrt 6 / 4, 7.5) := 
by
  sorry

end spherical_to_rectangular_coordinates_l105_105761


namespace base_circumference_of_cone_l105_105052

theorem base_circumference_of_cone (r : ℝ) (theta : ℝ) (C : ℝ) 
  (h_radius : r = 6)
  (h_theta : theta = 180)
  (h_C : C = 2 * Real.pi * r) :
  (theta / 360) * C = 6 * Real.pi :=
by
  sorry

end base_circumference_of_cone_l105_105052


namespace business_value_l105_105286

theorem business_value (h₁ : (2/3 : ℝ) * (3/4 : ℝ) * V = 30000) : V = 60000 :=
by
  -- conditions and definitions go here
  sorry

end business_value_l105_105286


namespace probability_is_half_l105_105816

noncomputable def probability_at_least_35_cents : ℚ :=
  let total_outcomes := 32
  let successful_outcomes := 8 + 4 + 4 -- from solution steps (1, 2, 3)
  successful_outcomes / total_outcomes

theorem probability_is_half :
  probability_at_least_35_cents = 1 / 2 := by
  -- proof details are not required as per instructions
  sorry

end probability_is_half_l105_105816


namespace smallest_N_winning_strategy_l105_105225

theorem smallest_N_winning_strategy :
  ∃ (N : ℕ), (N > 0) ∧ (∀ (list : List ℕ), 
    (∀ x, x ∈ list → x > 0 ∧ x ≤ 25) ∧ 
    list.sum ≥ 200 → 
    ∃ (sublist : List ℕ), sublist ⊆ list ∧ 
    200 - N ≤ sublist.sum ∧ sublist.sum ≤ 200 + N) ∧ N = 11 :=
sorry

end smallest_N_winning_strategy_l105_105225


namespace total_sum_of_money_is_71_l105_105383

noncomputable def totalCoins : ℕ := 334
noncomputable def coins20Paise : ℕ := 250
noncomputable def coins25Paise : ℕ := totalCoins - coins20Paise
noncomputable def value20Paise : ℕ := coins20Paise * 20
noncomputable def value25Paise : ℕ := coins25Paise * 25
noncomputable def totalValuePaise : ℕ := value20Paise + value25Paise
noncomputable def totalValueRupees : ℚ := totalValuePaise / 100

theorem total_sum_of_money_is_71 :
  totalValueRupees = 71 := by
  sorry

end total_sum_of_money_is_71_l105_105383


namespace functions_identified_l105_105847

variable (n : ℕ) (hn : n > 1)
variable {f : ℕ → ℝ → ℝ}

-- Define the conditions f1, f2, ..., fn
axiom cond_1 (x y : ℝ) : f 1 x + f 1 y = f 2 x * f 2 y
axiom cond_2 (x y : ℝ) : f 2 (x^2) + f 2 (y^2) = f 3 x * f 3 y
axiom cond_3 (x y : ℝ) : f 3 (x^3) + f 3 (y^3) = f 4 x * f 4 y
-- ... Similarly define conditions up to cond_n
axiom cond_n (x y : ℝ) : f n (x^n) + f n (y^n) = f 1 x * f 1 y

theorem functions_identified (i : ℕ) (hi₁ : 1 ≤ i) (hi₂ : i ≤ n) (x : ℝ) :
  f i x = 0 ∨ f i x = 2 :=
sorry

end functions_identified_l105_105847


namespace art_museum_survey_l105_105277

theorem art_museum_survey (V E : ℕ) 
  (h1 : ∀ (x : ℕ), x = 140 → ¬ (x ≤ E))
  (h2 : E = (3 / 4) * V)
  (h3 : V = E + 140) :
  V = 560 := by
  sorry

end art_museum_survey_l105_105277


namespace no_such_function_exists_l105_105137

theorem no_such_function_exists :
  ¬ ∃ (f : ℝ → ℝ), ∀ x : ℝ, f (f x) = x^2 - 1996 :=
by
  sorry

end no_such_function_exists_l105_105137


namespace minimize_f_a_n_distance_l105_105638

noncomputable def f (x : ℝ) : ℝ :=
  2^x + Real.log x

noncomputable def a (n : ℕ) : ℝ :=
  0.1 * n

theorem minimize_f_a_n_distance :
  ∃ n : ℕ, n = 110 ∧ ∀ m : ℕ, (m > 0) -> |f (a 110) - 2012| ≤ |f (a m) - 2012| :=
by
  sorry

end minimize_f_a_n_distance_l105_105638


namespace solve_for_angle_B_solutions_l105_105564

noncomputable def number_of_solutions_for_angle_B (BC AC : ℝ) (angle_A : ℝ) : ℕ :=
  if (BC = 6 ∧ AC = 8 ∧ angle_A = 40) then 2 else 0

theorem solve_for_angle_B_solutions : number_of_solutions_for_angle_B 6 8 40 = 2 :=
  by sorry

end solve_for_angle_B_solutions_l105_105564


namespace minimum_words_to_learn_for_90_percent_l105_105965

-- Define the conditions
def total_vocabulary_words : ℕ := 800
def minimum_percentage_required : ℚ := 0.90

-- Define the proof goal
theorem minimum_words_to_learn_for_90_percent (x : ℕ) (h1 : (x : ℚ) / total_vocabulary_words ≥ minimum_percentage_required) : x ≥ 720 :=
sorry

end minimum_words_to_learn_for_90_percent_l105_105965


namespace problem1_problem2_problem3_problem4_problem5_problem6_l105_105855

-- Proof for 238 + 45 × 5 = 463
theorem problem1 : 238 + 45 * 5 = 463 := by
  sorry

-- Proof for 65 × 4 - 128 = 132
theorem problem2 : 65 * 4 - 128 = 132 := by
  sorry

-- Proof for 900 - 108 × 4 = 468
theorem problem3 : 900 - 108 * 4 = 468 := by
  sorry

-- Proof for 369 + (512 - 215) = 666
theorem problem4 : 369 + (512 - 215) = 666 := by
  sorry

-- Proof for 758 - 58 × 9 = 236
theorem problem5 : 758 - 58 * 9 = 236 := by
  sorry

-- Proof for 105 × (81 ÷ 9 - 3) = 630
theorem problem6 : 105 * (81 / 9 - 3) = 630 := by
  sorry

end problem1_problem2_problem3_problem4_problem5_problem6_l105_105855


namespace largest_sampled_number_l105_105780

theorem largest_sampled_number (N : ℕ) (a₁ a₂ : ℕ) (k : ℕ) (H_N : N = 1500)
  (H_a₁ : a₁ = 18) (H_a₂ : a₂ = 68) (H_k : k = a₂ - a₁) :
  ∃ m, m ≤ N ∧ (m % k = 18 % k) ∧ ∀ n, (n % k = 18 % k) → n ≤ N → n ≤ m :=
by {
  -- sorry
  sorry
}

end largest_sampled_number_l105_105780


namespace diff_x_y_l105_105462

theorem diff_x_y (x y : ℤ) (h1 : x + y = 14) (h2 : x = 37) : x - y = 60 :=
sorry

end diff_x_y_l105_105462


namespace radius_of_circle_is_4_l105_105377

noncomputable def circle_radius
  (a : ℝ) 
  (radius : ℝ) 
  (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*a*x + 9 = 0 ∧ (-a, 0) = (5, 0) ∧ radius = 4

theorem radius_of_circle_is_4 
  (a x y : ℝ) 
  (radius : ℝ) 
  (h : circle_radius a radius x y) : 
  radius = 4 :=
by 
  sorry

end radius_of_circle_is_4_l105_105377


namespace bob_mother_twice_age_2040_l105_105524

theorem bob_mother_twice_age_2040 :
  ∀ (bob_age_2010 mother_age_2010 : ℕ), 
  bob_age_2010 = 10 ∧ mother_age_2010 = 50 →
  ∃ (x : ℕ), (mother_age_2010 + x = 2 * (bob_age_2010 + x)) ∧ (2010 + x = 2040) :=
by
  sorry

end bob_mother_twice_age_2040_l105_105524


namespace length_of_rectangular_garden_l105_105788

-- Define the perimeter and breadth conditions
def perimeter : ℕ := 950
def breadth : ℕ := 100

-- The formula for the perimeter of a rectangle
def formula (L B : ℕ) : ℕ := 2 * (L + B)

-- State the theorem
theorem length_of_rectangular_garden (L : ℕ) 
  (h1 : perimeter = 2 * (L + breadth)) : 
  L = 375 := 
by
  sorry

end length_of_rectangular_garden_l105_105788


namespace find_x_when_y_is_72_l105_105952

theorem find_x_when_y_is_72 
  (x y : ℝ) (k : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) 
  (h_const : ∀ x y, 0 < x → 0 < y → x^2 * y = k)
  (h_initial : 9 * 8 = k)
  (h_y_72 : y = 72)
  (h_x2_factor : x^2 = 4 * 9) :
  x = 1 :=
sorry

end find_x_when_y_is_72_l105_105952


namespace simplify_complex_expression_l105_105079

theorem simplify_complex_expression (i : ℂ) (h : i^2 = -1) : 
  7 * (4 - 2 * i) + 4 * i * (7 - 3 * i) = 40 + 14 * i :=
by
  sorry

end simplify_complex_expression_l105_105079


namespace intersection_of_A_and_B_l105_105089

-- Define sets A and B
def A := {x : ℝ | x > 0}
def B := {x : ℝ | x < 1}

-- Statement of the proof problem
theorem intersection_of_A_and_B : A ∩ B = {x | 0 < x ∧ x < 1} := by
  sorry -- The proof goes here

end intersection_of_A_and_B_l105_105089


namespace frac_eq_l105_105739

theorem frac_eq (x : ℝ) (h : 3 - 9 / x + 6 / x^2 = 0) : 2 / x = 1 ∨ 2 / x = 2 := 
by 
  sorry

end frac_eq_l105_105739


namespace goose_eggs_laied_l105_105283

theorem goose_eggs_laied (z : ℕ) (hatch_rate : ℚ := 2 / 3) (first_month_survival_rate : ℚ := 3 / 4) 
  (first_year_survival_rate : ℚ := 2 / 5) (geese_survived_first_year : ℕ := 126) :
  (hatch_rate * z) = 420 ∧ (first_month_survival_rate * 315 = 315) ∧ (first_year_survival_rate * 315 = 126) →
  z = 630 :=
by
  sorry

end goose_eggs_laied_l105_105283


namespace hypercube_paths_24_l105_105182

-- Define the 4-dimensional hypercube
structure Hypercube4 :=
(vertices : Fin 16) -- Using Fin 16 to represent the 16 vertices
(edges : Fin 32)    -- Using Fin 32 to represent the 32 edges

def valid_paths (start : Fin 16) : Nat :=
  -- This function should calculate the number of valid paths given the start vertex
  24 -- placeholder, as we are giving the pre-computed total number here

theorem hypercube_paths_24 (start : Fin 16) :
  valid_paths start = 24 :=
by sorry

end hypercube_paths_24_l105_105182


namespace probability_blue_is_approx_50_42_l105_105749

noncomputable def probability_blue_second_pick : ℚ :=
  let yellow := 30
  let green := yellow / 3
  let red := 2 * green
  let total_marbles := 120
  let blue := total_marbles - (yellow + green + red)
  let total_after_first_pick := total_marbles - 1
  let blue_probability := (blue : ℚ) / total_after_first_pick
  blue_probability * 100

theorem probability_blue_is_approx_50_42 :
  abs (probability_blue_second_pick - 50.42) < 0.005 := -- Approximately checking for equality due to possible floating-point precision issues
sorry

end probability_blue_is_approx_50_42_l105_105749


namespace vehicle_value_last_year_l105_105309

variable (v_this_year v_last_year : ℝ)

theorem vehicle_value_last_year:
  v_this_year = 16000 ∧ v_this_year = 0.8 * v_last_year → v_last_year = 20000 :=
by
  -- Proof steps can be added here, but replaced with sorry as per instructions.
  sorry

end vehicle_value_last_year_l105_105309


namespace seeds_germination_percentage_l105_105086

theorem seeds_germination_percentage :
  ∀ (total_seeds first_plot_seeds second_plot_seeds germinated_percentage_total germinated_percentage_second_plot germinated_seeds_total germinated_seeds_second_plot germinated_seeds_first_plot x : ℕ),
    total_seeds = 300 + 200 → 
    germinated_percentage_second_plot = 35 → 
    germinated_percentage_total = 32 → 
    second_plot_seeds = 200 → 
    germinated_seeds_second_plot = (germinated_percentage_second_plot * second_plot_seeds) / 100 → 
    germinated_seeds_total = (germinated_percentage_total * total_seeds) / 100 → 
    germinated_seeds_first_plot = germinated_seeds_total - germinated_seeds_second_plot → 
    x = 30 → 
    x = (germinated_seeds_first_plot * 100) / 300 → 
    x = 30 :=
  by 
    intros total_seeds first_plot_seeds second_plot_seeds germinated_percentage_total germinated_percentage_second_plot germinated_seeds_total germinated_seeds_second_plot germinated_seeds_first_plot x
    sorry

end seeds_germination_percentage_l105_105086


namespace find_quotient_l105_105388

theorem find_quotient
  (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ)
  (h1 : dividend = 131) (h2 : divisor = 14) (h3 : remainder = 5)
  (h4 : dividend = divisor * quotient + remainder) :
  quotient = 9 :=
by
  sorry

end find_quotient_l105_105388


namespace find_f_at_9_over_2_l105_105921

variable (f : ℝ → ℝ)

-- Domain of f is ℝ
axiom domain_f : ∀ x : ℝ, f x = f x

-- f(x+1) is an odd function
axiom odd_f : ∀ x : ℝ, f (x + 1) = -f (-(x - 1))

-- f(x+2) is an even function
axiom even_f : ∀ x : ℝ, f (x + 2) = f (-(x - 2))

-- When x is in [1,2], f(x) = ax^2 + b
variables (a b : ℝ)
axiom on_interval : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → f x = a * x^2 + b

-- f(0) + f(3) = 6
axiom sum_f : f 0 + f 3 = 6 

theorem find_f_at_9_over_2 : f (9/2) = 5/2 := 
by sorry

end find_f_at_9_over_2_l105_105921


namespace coefficient_of_x3_in_expansion_l105_105418

theorem coefficient_of_x3_in_expansion :
  let coeff := 56 * 972 * Real.sqrt 2
  coeff = 54432 * Real.sqrt 2 :=
by
  let coeff := 56 * 972 * Real.sqrt 2
  have h : coeff = 54432 * Real.sqrt 2 := sorry
  exact h

end coefficient_of_x3_in_expansion_l105_105418


namespace y_intercept_of_line_l105_105226

theorem y_intercept_of_line 
  (point : ℝ × ℝ)
  (slope_angle : ℝ)
  (h1 : point = (2, -5))
  (h2 : slope_angle = 135) :
  ∃ b : ℝ, (∀ x y : ℝ, y = -x + b ↔ ((y - (-5)) = (-1) * (x - 2))) ∧ b = -3 := 
sorry

end y_intercept_of_line_l105_105226


namespace sue_travel_time_correct_l105_105940

-- Define the flight and layover times as constants
def NO_to_ATL_flight_hours : ℕ := 2
def ATL_layover_hours : ℕ := 4
def ATL_to_CHI_flight_hours : ℕ := 5
def CHI_time_diff_hours : ℤ := -1
def CHI_layover_hours : ℕ := 3
def CHI_to_NY_flight_hours : ℕ := 3
def NY_time_diff_hours : ℤ := 1
def NY_layover_hours : ℕ := 16
def NY_to_DEN_flight_hours : ℕ := 6
def DEN_time_diff_hours : ℤ := -2
def DEN_layover_hours : ℕ := 5
def DEN_to_SF_flight_hours : ℕ := 4
def SF_time_diff_hours : ℤ := -1

-- Total time calculation including flights, layovers, and time zone changes
def total_travel_time_hours : ℕ :=
  NO_to_ATL_flight_hours +
  ATL_layover_hours +
  (ATL_to_CHI_flight_hours + CHI_time_diff_hours).toNat +  -- Handle time difference (ensure non-negative)
  CHI_layover_hours +
  (CHI_to_NY_flight_hours + NY_time_diff_hours).toNat +
  NY_layover_hours +
  (NY_to_DEN_flight_hours + DEN_time_diff_hours).toNat +
  DEN_layover_hours +
  (DEN_to_SF_flight_hours + SF_time_diff_hours).toNat

-- Statement to prove in Lean:
theorem sue_travel_time_correct : total_travel_time_hours = 45 :=
by {
  -- Skipping proof details since only the statement is required
  sorry
}

end sue_travel_time_correct_l105_105940


namespace fenced_area_l105_105511

theorem fenced_area (w : ℕ) (h : ℕ) (cut_out : ℕ) (rectangle_area : ℕ) (cut_out_area : ℕ) (net_area : ℕ) :
  w = 20 → h = 18 → cut_out = 4 → rectangle_area = w * h → cut_out_area = cut_out * cut_out → net_area = rectangle_area - cut_out_area → net_area = 344 :=
by
  intros
  subst_vars
  sorry

end fenced_area_l105_105511


namespace equal_sum_partition_l105_105588

theorem equal_sum_partition (n : ℕ) (a : Fin n.succ → ℕ)
  (h1 : a 0 = 1)
  (h2 : ∀ i : Fin n, a i ≤ a i.succ ∧ a i.succ ≤ 2 * a i)
  (h3 : (Finset.univ : Finset (Fin n.succ)).sum a % 2 = 0) :
  ∃ (partition : Finset (Fin n.succ)), 
    (partition.sum a = (partitionᶜ : Finset (Fin n.succ)).sum a) :=
by sorry

end equal_sum_partition_l105_105588


namespace car_quotient_div_15_l105_105543

/-- On a straight, one-way, single-lane highway, cars all travel at the same speed
    and obey a modified safety rule: the distance from the back of the car ahead
    to the front of the car behind is exactly two car lengths for each 20 kilometers
    per hour of speed. A sensor by the road counts the number of cars that pass in
    one hour. Each car is 5 meters long. 
    Let N be the maximum whole number of cars that can pass the sensor in one hour.
    Prove that when N is divided by 15, the quotient is 266. -/
theorem car_quotient_div_15 
  (speed : ℕ) 
  (d : ℕ) 
  (sensor_time : ℕ) 
  (car_length : ℕ)
  (N : ℕ)
  (h1 : ∀ m, speed = 20 * m)
  (h2 : d = 2 * car_length)
  (h3 : car_length = 5)
  (h4 : sensor_time = 1)
  (h5 : N = 4000) : 
  N / 15 = 266 := 
sorry

end car_quotient_div_15_l105_105543


namespace total_price_of_basic_computer_and_printer_l105_105675

-- Definitions for the conditions
def basic_computer_price := 2000
def enhanced_computer_price (C : ℕ) := C + 500
def printer_price (C : ℕ) (P : ℕ) := 1/6 * (C + 500 + P)

-- The proof problem statement
theorem total_price_of_basic_computer_and_printer (C P : ℕ) 
  (h1 : C = 2000)
  (h2 : printer_price C P = P) : 
  C + P = 2500 :=
sorry

end total_price_of_basic_computer_and_printer_l105_105675


namespace katie_remaining_juice_l105_105161

-- Define the initial condition: Katie initially has 5 gallons of juice
def initial_gallons : ℚ := 5

-- Define the amount of juice given to Mark
def juice_given : ℚ := 18 / 7

-- Define the expected remaining fraction of juice
def expected_remaining_gallons : ℚ := 17 / 7

-- The theorem statement that Katie should have 17/7 gallons of juice left
theorem katie_remaining_juice : initial_gallons - juice_given = expected_remaining_gallons := 
by
  -- proof would go here
  sorry

end katie_remaining_juice_l105_105161


namespace sin_phi_value_l105_105263

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin x + 4 * Real.cos x
noncomputable def g (x : ℝ) : ℝ := 3 * Real.sin x - 4 * Real.cos x

theorem sin_phi_value (φ : ℝ) (h_shift : ∀ x, g x = f (x - φ)) : Real.sin φ = 24 / 25 :=
by
  sorry

end sin_phi_value_l105_105263


namespace concentration_in_third_flask_l105_105439

-- Definitions for the problem conditions
def first_flask_acid_mass : ℕ := 10
def second_flask_acid_mass : ℕ := 20
def third_flask_acid_mass : ℕ := 30

-- Define the total mass after adding water to achieve given concentrations
def total_mass_first_flask (water_added_first : ℕ) : ℕ := first_flask_acid_mass + water_added_first
def total_mass_second_flask (water_added_second : ℕ) : ℕ := second_flask_acid_mass + water_added_second
def total_mass_third_flask (total_water : ℕ) : ℕ := third_flask_acid_mass + total_water

-- Given concentrations as conditions
def first_flask_concentration (water_added_first : ℕ) : Prop :=
  (first_flask_acid_mass : ℚ) / (total_mass_first_flask water_added_first : ℚ) = 0.05

def second_flask_concentration (water_added_second : ℕ) : Prop :=
  (second_flask_acid_mass : ℚ) / (total_mass_second_flask water_added_second : ℚ) = 70 / 300

-- Define total water added
def total_water (water_added_first water_added_second : ℕ) : ℕ :=
  water_added_first + water_added_second

-- Final concentration in the third flask
def third_flask_concentration (total_water_added : ℕ) : Prop :=
  (third_flask_acid_mass : ℚ) / (total_mass_third_flask total_water_added : ℚ) = 0.105

-- Lean theorem statement
theorem concentration_in_third_flask
  (water_added_first water_added_second : ℕ)
  (h1 : first_flask_concentration water_added_first)
  (h2 : second_flask_concentration water_added_second) :
  third_flask_concentration (total_water water_added_first water_added_second) :=
sorry

end concentration_in_third_flask_l105_105439


namespace find_a_for_parallel_lines_l105_105647

def direction_vector_1 (a : ℝ) : ℝ × ℝ × ℝ :=
  (2 * a, 3, 2)

def direction_vector_2 : ℝ × ℝ × ℝ :=
  (2, 3, 2)

theorem find_a_for_parallel_lines : ∃ a : ℝ, direction_vector_1 a = direction_vector_2 :=
by
  use 1
  unfold direction_vector_1
  sorry  -- proof omitted

end find_a_for_parallel_lines_l105_105647


namespace trains_crossing_time_l105_105293

/-- Define the length of the first train in meters -/
def length_train1 : ℚ := 200

/-- Define the length of the second train in meters -/
def length_train2 : ℚ := 150

/-- Define the speed of the first train in kilometers per hour -/
def speed_train1_kmph : ℚ := 40

/-- Define the speed of the second train in kilometers per hour -/
def speed_train2_kmph : ℚ := 46

/-- Define conversion factor from kilometers per hour to meters per second -/
def kmph_to_mps : ℚ := 1000 / 3600

/-- Calculate the relative speed in meters per second assuming both trains are moving in the same direction -/
def relative_speed_mps : ℚ := (speed_train2_kmph - speed_train1_kmph) * kmph_to_mps

/-- Calculate the combined length of both trains in meters -/
def combined_length : ℚ := length_train1 + length_train2

/-- Prove the time in seconds for the two trains to cross each other when moving in the same direction is 210 seconds -/
theorem trains_crossing_time :
  (combined_length / relative_speed_mps) = 210 := by
  sorry

end trains_crossing_time_l105_105293


namespace coordinate_sum_condition_l105_105799

open Function

theorem coordinate_sum_condition :
  (∃ (g : ℝ → ℝ), g 6 = 5 ∧
    (∃ y : ℝ, 4 * y = g (3 * 2) + 4 ∧ y = 9 / 4 ∧ 2 + y = 17 / 4)) :=
by
  sorry

end coordinate_sum_condition_l105_105799


namespace number_of_classes_l105_105140

theorem number_of_classes
  (s : ℕ)    -- s: number of students in each class
  (bpm : ℕ) -- bpm: books per month per student
  (months : ℕ) -- months: number of months in a year
  (total_books : ℕ) -- total_books: total books read by the entire student body in a year
  (H1 : bpm = 5)
  (H2 : months = 12)
  (H3 : total_books = 60)
  (H4 : total_books = s * bpm * months)
: s = 1 :=
by
  sorry

end number_of_classes_l105_105140


namespace product_mn_l105_105113

-- Λet θ1 be the angle L1 makes with the positive x-axis.
-- Λet θ2 be the angle L2 makes with the positive x-axis.
-- Given that θ1 = 3 * θ2 and m = 6 * n.
-- Using the tangent triple angle formula: tan(3θ) = (3 * tan(θ) - tan^3(θ)) / (1 - 3 * tan^2(θ))
-- We need to prove mn = 9/17.

noncomputable def mn_product_condition (θ1 θ2 : ℝ) (m n : ℝ) : Prop :=
θ1 = 3 * θ2 ∧ m = 6 * n ∧ m = Real.tan θ1 ∧ n = Real.tan θ2

theorem product_mn (θ1 θ2 : ℝ) (m n : ℝ) (h : mn_product_condition θ1 θ2 m n) :
  m * n = 9 / 17 :=
sorry

end product_mn_l105_105113


namespace EllenBreadMakingTime_l105_105360

-- Definitions based on the given problem
def RisingTimeTypeA : ℕ → ℝ := λ n => n * 4
def BakingTimeTypeA : ℕ → ℝ := λ n => n * 2.5
def RisingTimeTypeB : ℕ → ℝ := λ n => n * 3.5
def BakingTimeTypeB : ℕ → ℝ := λ n => n * 3

def TotalTime (nA nB : ℕ) : ℝ :=
  (RisingTimeTypeA nA + BakingTimeTypeA nA) +
  (RisingTimeTypeB nB + BakingTimeTypeB nB)

theorem EllenBreadMakingTime :
  TotalTime 3 2 = 32.5 := by
  sorry

end EllenBreadMakingTime_l105_105360


namespace inequality_one_inequality_two_l105_105331

variable {a b r s : ℝ}

theorem inequality_one (h_a : 0 < a) (h_b : 0 < b) :
  a^2 * b ≤ 4 * ((a + b) / 3)^3 :=
sorry

theorem inequality_two (h_a : 0 < a) (h_b : 0 < b) (h_r : 0 < r) (h_s : 0 < s) 
  (h_eq : 1 / r + 1 / s = 1) : 
  (a^r / r) + (b^s / s) ≥ a * b :=
sorry

end inequality_one_inequality_two_l105_105331


namespace add_alcohol_solve_l105_105516

variable (x : ℝ)

def initial_solution_volume : ℝ := 6
def initial_alcohol_fraction : ℝ := 0.20
def desired_alcohol_fraction : ℝ := 0.50

def initial_alcohol_content : ℝ := initial_alcohol_fraction * initial_solution_volume
def total_solution_volume_after_addition : ℝ := initial_solution_volume + x
def total_alcohol_content_after_addition : ℝ := initial_alcohol_content + x

theorem add_alcohol_solve (x : ℝ) :
  (initial_alcohol_content + x) / (initial_solution_volume + x) = desired_alcohol_fraction →
  x = 3.6 :=
by
  sorry

end add_alcohol_solve_l105_105516


namespace problems_left_to_grade_l105_105890

def worksheets : ℕ := 17
def graded_worksheets : ℕ := 8
def problems_per_worksheet : ℕ := 7

theorem problems_left_to_grade : (worksheets - graded_worksheets) * problems_per_worksheet = 63 := by
  sorry

end problems_left_to_grade_l105_105890


namespace russian_pairing_probability_l105_105077

-- Definitions based on conditions
def total_players : ℕ := 10
def russian_players : ℕ := 4
def non_russian_players : ℕ := total_players - russian_players

-- Probability calculation as a hypothesis
noncomputable def pairing_probability (rs: ℕ) (ns: ℕ) : ℚ :=
  (rs * (rs - 1)) / (total_players * (total_players - 1))

theorem russian_pairing_probability :
  pairing_probability russian_players non_russian_players = 1 / 21 :=
sorry

end russian_pairing_probability_l105_105077


namespace sum_of_numerical_coefficients_binomial_l105_105313

theorem sum_of_numerical_coefficients_binomial (a b : ℕ) (n : ℕ) (h : n = 8) :
  let sum_num_coeff := (a + b)^n
  sum_num_coeff = 256 :=
by 
  sorry

end sum_of_numerical_coefficients_binomial_l105_105313


namespace find_like_term_l105_105074

-- Definition of the problem conditions
def monomials : List (String × String) := 
  [("A", "-2a^2b"), 
   ("B", "a^2b^2"), 
   ("C", "ab^2"), 
   ("D", "3ab")]

-- A function to check if two terms can be combined (like terms)
def like_terms(a b : String) : Prop :=
  a = "a^2b" ∧ b = "-2a^2b"

-- The theorem we need to prove
theorem find_like_term : ∃ x, x ∈ monomials ∧ like_terms "a^2b" (x.2) ∧ x.2 = "-2a^2b" :=
  sorry

end find_like_term_l105_105074


namespace triangle_angle_area_l105_105327

theorem triangle_angle_area
  (A B C : ℝ) (a b c : ℝ)
  (h1 : c * Real.cos B = (2 * a - b) * Real.cos C)
  (h2 : C = Real.pi / 3)
  (h3 : c = 2)
  (h4 : a + b + c = 2 * Real.sqrt 3 + 2) :
  ∃ (area : ℝ), area = (2 * Real.sqrt 3) / 3 :=
by 
  -- Proof is omitted
  sorry

end triangle_angle_area_l105_105327


namespace sum_of_prime_factors_of_143_l105_105914

theorem sum_of_prime_factors_of_143 :
  (Nat.Prime 11) ∧ (Nat.Prime 13) ∧ (143 = 11 * 13) → 11 + 13 = 24 := 
by
  sorry

end sum_of_prime_factors_of_143_l105_105914


namespace wicket_count_l105_105205

theorem wicket_count (initial_avg new_avg : ℚ) (runs_last_match wickets_last_match : ℕ) (delta_avg : ℚ) (W : ℕ) :
  initial_avg = 12.4 →
  new_avg = 12.0 →
  delta_avg = 0.4 →
  runs_last_match = 26 →
  wickets_last_match = 8 →
  initial_avg * W + runs_last_match = new_avg * (W + wickets_last_match) →
  W = 175 := by
  sorry

end wicket_count_l105_105205


namespace system1_solution_system2_solution_l105_105083

-- System (1)
theorem system1_solution (x y : ℚ) (h1 : 3 * y - 4 * x = 0) (h2 : 4 * x + y = 8) : 
  x = 3 / 2 ∧ y = 2 :=
by
  sorry

-- System (2)
theorem system2_solution (x y : ℚ) (h1 : x + y = 3) (h2 : (x - 1) / 4 + y / 2 = 3 / 4) : 
  x = 2 ∧ y = 1 :=
by
  sorry

end system1_solution_system2_solution_l105_105083


namespace fruit_store_problem_l105_105907

-- Define the conditions
def total_weight : Nat := 140
def total_cost : Nat := 1000

def purchase_price_A : Nat := 5
def purchase_price_B : Nat := 9

def selling_price_A : Nat := 8
def selling_price_B : Nat := 13

-- Define the total purchase price equation
def purchase_cost (x : Nat) : Nat := purchase_price_A * x + purchase_price_B * (total_weight - x)

-- Define the profit calculation
def profit (x : Nat) (y : Nat) : Nat := (selling_price_A - purchase_price_A) * x + (selling_price_B - purchase_price_B) * y

-- State the problem
theorem fruit_store_problem :
  ∃ x y : Nat, x + y = total_weight ∧ purchase_cost x = total_cost ∧ profit x y = 495 :=
by
  sorry

end fruit_store_problem_l105_105907


namespace johns_weekly_earnings_percentage_increase_l105_105928

theorem johns_weekly_earnings_percentage_increase (initial final : ℝ) :
  initial = 30 →
  final = 50 →
  ((final - initial) / initial) * 100 = 66.67 :=
by
  intros h_initial h_final
  rw [h_initial, h_final]
  norm_num
  sorry

end johns_weekly_earnings_percentage_increase_l105_105928


namespace unique_solution_l105_105820

theorem unique_solution:
  ∃! (x y z : ℕ), 2^x + 9 * 7^y = z^3 ∧ x = 0 ∧ y = 1 ∧ z = 4 :=
by
  sorry

end unique_solution_l105_105820


namespace gcf_3465_10780_l105_105551

theorem gcf_3465_10780 : Nat.gcd 3465 10780 = 385 := by
  sorry

end gcf_3465_10780_l105_105551


namespace correct_propositions_l105_105840

noncomputable def sequence_condition (a : ℕ → ℝ) : Prop :=
  ∀ n > 1, a n + a (n+1) > 2 * a n

def prop1 (a : ℕ → ℝ) (h : sequence_condition a) : Prop :=
  a 2 > a 1 → ∀ n > 1, a n > a (n-1)

def prop4 (a : ℕ → ℝ) (h : sequence_condition a) : Prop :=
  ∃ d, ∀ n > 1, a n > a 1 + (n-1) * d

theorem correct_propositions {a : ℕ → ℝ}
  (h : sequence_condition a) :
  (prop1 a h) ∧ (prop4 a h) := 
sorry

end correct_propositions_l105_105840


namespace max_sum_of_radii_in_prism_l105_105738

noncomputable def sum_of_radii (AB AD AA1 : ℝ) : ℝ :=
  let r (t : ℝ) := 2 - 2 * t
  let R (t : ℝ) := 3 * t / (1 + t)
  let f (t : ℝ) := R t + r t
  let t_max := 1 / 2
  f t_max

theorem max_sum_of_radii_in_prism :
  let AB := 5
  let AD := 3
  let AA1 := 4
  sum_of_radii AB AD AA1 = 21 / 10 := by
sorry

end max_sum_of_radii_in_prism_l105_105738


namespace maximize_area_center_coordinates_l105_105141

theorem maximize_area_center_coordinates (k : ℝ) :
  (∃ r : ℝ, r^2 = 1 - (3/4) * k^2 ∧ r ≥ 0) →
  ((k = 0) → ∃ a b : ℝ, (a = 0 ∧ b = -1)) :=
by
  sorry

end maximize_area_center_coordinates_l105_105141


namespace euler_quadrilateral_theorem_l105_105395

theorem euler_quadrilateral_theorem (A1 A2 A3 A4 P Q : ℝ) 
  (midpoint_P : P = (A1 + A3) / 2)
  (midpoint_Q : Q = (A2 + A4) / 2) 
  (length_A1A2 length_A2A3 length_A3A4 length_A4A1 length_A1A3 length_A2A4 length_PQ : ℝ)
  (h1 : length_A1A2 = A1A2) (h2 : length_A2A3 = A2A3)
  (h3 : length_A3A4 = A3A4) (h4 : length_A4A1 = A4A1)
  (h5 : length_A1A3 = A1A3) (h6 : length_A2A4 = A2A4)
  (h7 : length_PQ = PQ) :
  length_A1A2^2 + length_A2A3^2 + length_A3A4^2 + length_A4A1^2 = 
  length_A1A3^2 + length_A2A4^2 + 4 * length_PQ^2 := sorry

end euler_quadrilateral_theorem_l105_105395


namespace total_money_is_145_83_l105_105204

noncomputable def jackson_money : ℝ := 125

noncomputable def williams_money : ℝ := jackson_money / 6

noncomputable def total_money : ℝ := jackson_money + williams_money

theorem total_money_is_145_83 :
  total_money = 145.83 := by
sorry

end total_money_is_145_83_l105_105204


namespace scientific_notation_of_5_35_million_l105_105251

theorem scientific_notation_of_5_35_million : 
  (5.35 : ℝ) * 10^6 = 5.35 * 10^6 := 
by
  sorry

end scientific_notation_of_5_35_million_l105_105251


namespace interval_of_decrease_for_f_x_plus_1_l105_105123

def f_prime (x : ℝ) : ℝ := x^2 - 4*x + 3

theorem interval_of_decrease_for_f_x_plus_1 : 
  ∀ x, (f_prime (x + 1) < 0 ↔ 0 < x ∧ x < 2) :=
by 
  intro x
  sorry

end interval_of_decrease_for_f_x_plus_1_l105_105123


namespace simplify_logarithmic_expression_l105_105169

noncomputable def simplify_expression : ℝ :=
  1 / (Real.log 3 / Real.log 12 + 1) +
  1 / (Real.log 2 / Real.log 8 + 1) +
  1 / (Real.log 3 / Real.log 9 + 1)

theorem simplify_logarithmic_expression :
  simplify_expression = 4 / 3 :=
by
  sorry

end simplify_logarithmic_expression_l105_105169


namespace pounds_over_weight_limit_l105_105262

-- Definitions based on conditions

def bookcase_max_weight : ℝ := 80

def weight_hardcover_book : ℝ := 0.5
def number_hardcover_books : ℕ := 70
def total_weight_hardcover_books : ℝ := number_hardcover_books * weight_hardcover_book

def weight_textbook : ℝ := 2
def number_textbooks : ℕ := 30
def total_weight_textbooks : ℝ := number_textbooks * weight_textbook

def weight_knick_knack : ℝ := 6
def number_knick_knacks : ℕ := 3
def total_weight_knick_knacks : ℝ := number_knick_knacks * weight_knick_knack

def total_weight_items : ℝ := total_weight_hardcover_books + total_weight_textbooks + total_weight_knick_knacks

theorem pounds_over_weight_limit : total_weight_items - bookcase_max_weight = 33 := by
  sorry

end pounds_over_weight_limit_l105_105262


namespace probability_of_event_l105_105001

open Set Real

noncomputable def probability_event_interval (x : ℝ) : Prop :=
  1 ≤ 2 * x - 1 ∧ 2 * x - 1 ≤ 3

noncomputable def interval := Icc (0 : ℝ) (3 : ℝ)

noncomputable def event_probability := 1 / 3

theorem probability_of_event :
  ∀ x ∈ interval, probability_event_interval x → (event_probability) = 1 / 3 :=
by
  sorry

end probability_of_event_l105_105001


namespace age_of_b_l105_105352

theorem age_of_b (a b c : ℕ) (h1 : a = b + 2) (h2 : b = 2 * c) (h3 : a + b + c = 42) : b = 16 :=
by
  sorry

end age_of_b_l105_105352


namespace value_of_nested_custom_div_l105_105706

def custom_div (x y z : ℕ) (hz : z ≠ 0) : ℕ :=
  (x + y) / z

theorem value_of_nested_custom_div : custom_div (custom_div 45 15 60 (by decide)) (custom_div 3 3 6 (by decide)) (custom_div 20 10 30 (by decide)) (by decide) = 2 :=
sorry

end value_of_nested_custom_div_l105_105706


namespace percentage_of_loss_l105_105050

theorem percentage_of_loss
    (CP SP : ℝ)
    (h1 : CP = 1200)
    (h2 : SP = 1020)
    (Loss : ℝ)
    (h3 : Loss = CP - SP)
    (Percentage_of_Loss : ℝ)
    (h4 : Percentage_of_Loss = (Loss / CP) * 100) :
  Percentage_of_Loss = 15 := by
  sorry

end percentage_of_loss_l105_105050


namespace how_long_it_lasts_l105_105828

-- Define a structure to hold the conditions
structure MoneySpending where
  mowing_income : ℕ
  weeding_income : ℕ
  weekly_expense : ℕ

-- Example conditions given in the problem
def lukesEarnings : MoneySpending :=
{ mowing_income := 9,
  weeding_income := 18,
  weekly_expense := 3 }

-- Main theorem proving the number of weeks he can sustain his spending
theorem how_long_it_lasts (data : MoneySpending) : 
  (data.mowing_income + data.weeding_income) / data.weekly_expense = 9 := by
  sorry

end how_long_it_lasts_l105_105828


namespace evaluate_g_at_4_l105_105775

def g (x : ℕ) : ℕ := 5 * x - 2

theorem evaluate_g_at_4 : g 4 = 18 := by
  sorry

end evaluate_g_at_4_l105_105775


namespace second_interest_rate_l105_105168

theorem second_interest_rate (P1 P2 : ℝ) (r : ℝ) (total_amount total_income: ℝ) (h1 : total_amount = 2500)
  (h2 : P1 = 1500.0000000000007) (h3 : total_income = 135) :
  P2 = total_amount - P1 →
  P1 * 0.05 = 75 →
  P2 * r = 60 →
  r = 0.06 :=
sorry

end second_interest_rate_l105_105168


namespace xy_relationship_l105_105690

theorem xy_relationship (x y : ℤ) (h1 : 2 * x - y > x + 1) (h2 : x + 2 * y < 2 * y - 3) :
  x < -3 ∧ y < -4 ∧ x > y + 1 :=
sorry

end xy_relationship_l105_105690


namespace greatest_integer_c_not_in_range_l105_105280

theorem greatest_integer_c_not_in_range :
  ∃ c : ℤ, (¬ ∃ x : ℝ, x^2 + (c:ℝ)*x + 18 = -6) ∧ (∀ c' : ℤ, c' > c → (∃ x : ℝ, x^2 + (c':ℝ)*x + 18 = -6)) :=
sorry

end greatest_integer_c_not_in_range_l105_105280


namespace tyler_age_l105_105475

theorem tyler_age (T C : ℕ) (h1 : T = 3 * C + 1) (h2 : T + C = 21) : T = 16 :=
by
  sorry

end tyler_age_l105_105475


namespace sum_of_triangles_l105_105129

def triangle (a b c : ℕ) : ℕ := a * b + c

theorem sum_of_triangles :
  triangle 3 2 5 + triangle 4 1 7 = 22 :=
by
  sorry

end sum_of_triangles_l105_105129


namespace area_of_enclosed_region_is_zero_l105_105449

theorem area_of_enclosed_region_is_zero :
  (∃ (x y : ℝ), x^2 + y^2 = |x| - |y|) → (0 = 0) :=
sorry

end area_of_enclosed_region_is_zero_l105_105449


namespace number_of_integers_congruent_7_mod_9_lessthan_1000_l105_105660

theorem number_of_integers_congruent_7_mod_9_lessthan_1000 : 
  ∃ k : ℕ, ∀ n : ℕ, n ≤ k → 7 + 9 * n < 1000 → k + 1 = 111 :=
by
  sorry

end number_of_integers_congruent_7_mod_9_lessthan_1000_l105_105660


namespace algebraic_simplification_l105_105248

theorem algebraic_simplification (m x : ℝ) (h₀ : 0 < m) (h₁ : m < 10) (h₂ : m ≤ x) (h₃ : x ≤ 10) : 
  |x - m| + |x - 10| + |x - m - 10| = 20 - x :=
by
  sorry

end algebraic_simplification_l105_105248


namespace least_possible_value_l105_105755

theorem least_possible_value (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : 4 * x = 5 * y ∧ 5 * y = 6 * z) : x + y + z = 37 :=
by
  sorry

end least_possible_value_l105_105755


namespace rosa_calls_pages_l105_105700

theorem rosa_calls_pages (pages_last_week : ℝ) (pages_this_week : ℝ) (h_last_week : pages_last_week = 10.2) (h_this_week : pages_this_week = 8.6) : pages_last_week + pages_this_week = 18.8 :=
by sorry

end rosa_calls_pages_l105_105700


namespace total_spider_legs_l105_105289

variable (numSpiders : ℕ)
variable (legsPerSpider : ℕ)
axiom h1 : numSpiders = 5
axiom h2 : legsPerSpider = 8

theorem total_spider_legs : numSpiders * legsPerSpider = 40 :=
by
  -- necessary for build without proof.
  sorry

end total_spider_legs_l105_105289


namespace dice_probability_l105_105220

noncomputable def probability_same_face_in_single_roll : ℝ :=
  (1 / 6)^10

noncomputable def probability_not_all_same_face_in_single_roll : ℝ :=
  1 - probability_same_face_in_single_roll

noncomputable def probability_not_all_same_face_in_five_rolls : ℝ :=
  probability_not_all_same_face_in_single_roll^5

noncomputable def probability_at_least_one_all_same_face : ℝ :=
  1 - probability_not_all_same_face_in_five_rolls

theorem dice_probability :
  probability_at_least_one_all_same_face = 1 - (1 - (1 / 6)^10)^5 :=
sorry

end dice_probability_l105_105220


namespace visibility_time_correct_l105_105684

noncomputable def visibility_time (r : ℝ) (d : ℝ) (v_j : ℝ) (v_k : ℝ) : ℝ :=
  (d / (v_j + v_k)) * (r / (r * (v_j / v_k + 1)))

theorem visibility_time_correct :
  visibility_time 60 240 4 2 = 120 :=
by
  sorry

end visibility_time_correct_l105_105684


namespace photo_album_pages_l105_105180

noncomputable def P1 := 0
noncomputable def P2 := 10
noncomputable def remaining_pages := 20

theorem photo_album_pages (photos total_pages photos_per_page_set1 photos_per_page_set2 photos_per_page_remaining : ℕ) 
  (h1 : photos = 100)
  (h2 : total_pages = 30)
  (h3 : photos_per_page_set1 = 3)
  (h4 : photos_per_page_set2 = 4)
  (h5 : photos_per_page_remaining = 3) : 
  P1 = 0 ∧ P2 = 10 ∧ remaining_pages = 20 :=
by
  sorry

end photo_album_pages_l105_105180


namespace average_of_remaining_numbers_l105_105134

theorem average_of_remaining_numbers 
    (nums : List ℝ) 
    (h_length : nums.length = 12) 
    (h_avg_90 : (nums.sum) / 12 = 90) 
    (h_contains_65_85 : 65 ∈ nums ∧ 85 ∈ nums) 
    (nums' := nums.erase 65)
    (nums'' := nums'.erase 85) : 
   nums''.length = 10 ∧ nums''.sum / 10 = 93 :=
by
  sorry

end average_of_remaining_numbers_l105_105134


namespace ellipse_y_axis_intersection_l105_105955

open Real

/-- Defines an ellipse with given foci and a point on the ellipse,
    and establishes the coordinate of the other y-axis intersection. -/
theorem ellipse_y_axis_intersection :
  ∃ y : ℝ, (dist (0, y) (1, -1) + dist (0, y) (-2, 2) = 3 * sqrt 2) ∧ y = sqrt ((9 * sqrt 2 - 4) / 2) :=
sorry

end ellipse_y_axis_intersection_l105_105955


namespace chess_champion_probability_l105_105035

theorem chess_champion_probability :
  let P_R := 0.6
  let P_S := 0.3
  let P_D := 0.1
  let P := 0.06 + 0.126 + 0.024 + 0.021 + 0.03 + 0.072 + 0.01
  1000 * P = 343 :=
by 
  let P_R := 0.6
  let P_S := 0.3
  let P_D := 0.1
  let P := 0.06 + 0.126 + 0.024 + 0.021 + 0.03 + 0.072 + 0.01
  show 1000 * P = 343
  sorry

end chess_champion_probability_l105_105035


namespace basin_capacity_l105_105754

-- Defining the flow rate of water into the basin
def inflow_rate : ℕ := 24

-- Defining the leak rate of the basin
def leak_rate : ℕ := 4

-- Defining the time taken to fill the basin in seconds
def fill_time : ℕ := 13

-- Net rate of filling the basin
def net_rate : ℕ := inflow_rate - leak_rate

-- Volume of the basin
def basin_volume : ℕ := net_rate * fill_time

-- The goal is to prove that the volume of the basin is 260 gallons
theorem basin_capacity : basin_volume = 260 := by
  sorry

end basin_capacity_l105_105754


namespace sequence_general_term_l105_105657

theorem sequence_general_term :
  ∀ (a : ℕ → ℝ), (a 1 = 1) →
    (∀ n : ℕ, n > 0 → (Real.sqrt (a n) - Real.sqrt (a (n + 1)) = Real.sqrt (a n * a (n + 1)))) →
    (∀ n : ℕ, n > 0 → a n = 1 / (n ^ 2)) :=
by
  intros a ha1 hrec n hn
  sorry

end sequence_general_term_l105_105657


namespace evaluate_expression_correct_l105_105664

noncomputable def evaluate_expression : ℤ :=
  6 - 8 * (9 - 4 ^ 2) * 5 + 2

theorem evaluate_expression_correct : evaluate_expression = 288 := by
  sorry

end evaluate_expression_correct_l105_105664


namespace minimum_value_k_eq_2_l105_105302

noncomputable def quadratic_function_min (a m k : ℝ) (h : 0 < a) : ℝ :=
  a * (-(k / 2)) * (-(k / 2) - k)

theorem minimum_value_k_eq_2 (a m : ℝ) (h : 0 < a) :
  quadratic_function_min a m 2 h = -a := 
by
  unfold quadratic_function_min
  sorry

end minimum_value_k_eq_2_l105_105302


namespace amount_solution_y_correct_l105_105604

-- Define conditions
def solution_x_alcohol_percentage : ℝ := 0.10
def solution_y_alcohol_percentage : ℝ := 0.30
def volume_solution_x : ℝ := 300.0
def target_alcohol_percentage : ℝ := 0.18

-- Define the main question as a theorem
theorem amount_solution_y_correct (y : ℝ) :
  (30 + 0.3 * y = 0.18 * (300 + y)) → y = 200 :=
by
  sorry

end amount_solution_y_correct_l105_105604


namespace greatest_mean_weight_l105_105435

variable (X Y Z : Type) [Group X] [Group Y] [Group Z]

theorem greatest_mean_weight 
  (mean_X : ℝ) (mean_Y : ℝ) (mean_XY : ℝ) (mean_XZ : ℝ)
  (hX : mean_X = 30)
  (hY : mean_Y = 70)
  (hXY : mean_XY = 50)
  (hXZ : mean_XZ = 40) :
  ∃ k : ℝ, k = 70 :=
by {
  sorry
}

end greatest_mean_weight_l105_105435


namespace flag_design_l105_105562

/-- Given three colors and a flag with three horizontal stripes where no adjacent stripes can be the 
same color, there are exactly 12 different possible flags. -/
theorem flag_design {colors : Finset ℕ} (h_colors : colors.card = 3) : 
  ∃ n : ℕ, n = 12 ∧ (∃ f : ℕ → ℕ, (∀ i, f i ∈ colors) ∧ (∀ i < 2, f i ≠ f (i + 1))) :=
sorry

end flag_design_l105_105562


namespace meaningful_expression_range_l105_105824

theorem meaningful_expression_range (x : ℝ) : (∃ y : ℝ, y = (1 / (Real.sqrt (x - 2)))) ↔ (x > 2) := 
sorry

end meaningful_expression_range_l105_105824


namespace triangle_count_relationship_l105_105979

theorem triangle_count_relationship :
  let n_0 : ℕ := 20
  let n_1 : ℕ := 19
  let n_2 : ℕ := 18
  n_0 > n_1 ∧ n_1 > n_2 :=
by
  let n_0 := 20
  let n_1 := 19
  let n_2 := 18
  have h0 : n_0 > n_1 := by sorry
  have h1 : n_1 > n_2 := by sorry
  exact ⟨h0, h1⟩

end triangle_count_relationship_l105_105979


namespace sum_of_solutions_eq_neg4_l105_105430

theorem sum_of_solutions_eq_neg4 :
  ∃ (n : ℕ) (solutions : Fin n → ℝ × ℝ),
    (∀ i, ∃ (x y : ℝ), solutions i = (x, y) ∧ abs (x - 3) = abs (y - 9) ∧ abs (x - 9) = 2 * abs (y - 3)) ∧
    (Finset.univ.sum (fun i => (solutions i).1 + (solutions i).2) = -4) :=
sorry

end sum_of_solutions_eq_neg4_l105_105430


namespace find_prime_pairs_l105_105978

theorem find_prime_pairs (p q : ℕ) (p_prime : Nat.Prime p) (q_prime : Nat.Prime q) 
  (h1 : ∃ a : ℤ, a^2 = p - q)
  (h2 : ∃ b : ℤ, b^2 = p * q - q) : 
  (p, q) = (3, 2) :=
by {
    sorry
}

end find_prime_pairs_l105_105978


namespace symmetric_point_coordinates_l105_105145

def point_symmetric_to_x_axis (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  match p with
  | (x, y, z) => (x, -y, -z)

theorem symmetric_point_coordinates :
  point_symmetric_to_x_axis (-2, 1, 4) = (-2, -1, -4) := by
  sorry

end symmetric_point_coordinates_l105_105145


namespace sophie_total_spend_l105_105157

def total_cost_with_discount_and_tax : ℝ :=
  let cupcakes_price := 5 * 2
  let doughnuts_price := 6 * 1
  let apple_pie_price := 4 * 2
  let cookies_price := 15 * 0.60
  let chocolate_bars_price := 8 * 1.50
  let soda_price := 12 * 1.20
  let gum_price := 3 * 0.80
  let chips_price := 10 * 1.10
  let total_before_discount := cupcakes_price + doughnuts_price + apple_pie_price + cookies_price + chocolate_bars_price + soda_price + gum_price + chips_price
  let discount := 0.10 * total_before_discount
  let subtotal_after_discount := total_before_discount - discount
  let sales_tax := 0.06 * subtotal_after_discount
  let total_cost := subtotal_after_discount + sales_tax
  total_cost

theorem sophie_total_spend :
  total_cost_with_discount_and_tax = 69.45 :=
sorry

end sophie_total_spend_l105_105157


namespace remainder_when_divided_by_44_l105_105985

theorem remainder_when_divided_by_44 (N : ℕ) (Q : ℕ) (R : ℕ)
  (h1 : N = 44 * 432 + R)
  (h2 : N = 31 * Q + 5) :
  R = 2 :=
sorry

end remainder_when_divided_by_44_l105_105985


namespace area_of_room_l105_105589

def length : ℝ := 12
def width : ℝ := 8

theorem area_of_room : length * width = 96 :=
by sorry

end area_of_room_l105_105589


namespace area_of_shaded_part_l105_105267

-- Define the given condition: area of the square
def area_of_square : ℝ := 100

-- Define the proof goal: area of the shaded part
theorem area_of_shaded_part : area_of_square / 2 = 50 := by
  sorry

end area_of_shaded_part_l105_105267


namespace judy_shopping_total_l105_105147

noncomputable def carrot_price := 1
noncomputable def milk_price := 3
noncomputable def pineapple_price := 4 / 2 -- half price
noncomputable def flour_price := 5
noncomputable def ice_cream_price := 7

noncomputable def carrot_quantity := 5
noncomputable def milk_quantity := 3
noncomputable def pineapple_quantity := 2
noncomputable def flour_quantity := 2
noncomputable def ice_cream_quantity := 1

noncomputable def initial_cost : ℝ := 
  carrot_quantity * carrot_price 
  + milk_quantity * milk_price 
  + pineapple_quantity * pineapple_price 
  + flour_quantity * flour_price 
  + ice_cream_quantity * ice_cream_price

noncomputable def final_cost (initial_cost: ℝ) := if initial_cost ≥ 25 then initial_cost - 5 else initial_cost

theorem judy_shopping_total : final_cost initial_cost = 30 := by
  sorry

end judy_shopping_total_l105_105147


namespace linear_function_difference_l105_105025

noncomputable def linear_function (f : ℝ → ℝ) : Prop :=
  ∃ m b : ℝ, ∀ x : ℝ, f x = m * x + b

theorem linear_function_difference (f : ℝ → ℝ) 
  (h_linear : linear_function f)
  (h_cond1 : f 10 - f 5 = 20)
  (h_cond2 : f 0 = 3) :
  f 15 - f 5 = 40 :=
sorry

end linear_function_difference_l105_105025


namespace frank_pie_consumption_l105_105459

theorem frank_pie_consumption :
  let Erik := 0.6666666666666666
  let MoreThanFrank := 0.3333333333333333
  let Frank := Erik - MoreThanFrank
  Frank = 0.3333333333333333 := by
sorry

end frank_pie_consumption_l105_105459


namespace cubic_roots_l105_105153

variable (p q : ℝ)

noncomputable def ω : ℂ := Complex.exp (2 * Real.pi * Complex.I / 3)

theorem cubic_roots (y z : ℂ) (h1 : -3 * y * z = p) (h2 : y^3 + z^3 = q) :
  ∃ (x1 x2 x3 : ℂ),
    (x^3 + p * x + q = 0) ∧
    (x1 = -(y + z)) ∧
    (x2 = -(ω * y + ω^2 * z)) ∧
    (x3 = -(ω^2 * y + ω * z)) :=
by
  sorry

end cubic_roots_l105_105153


namespace soccer_claim_fraction_l105_105772

theorem soccer_claim_fraction 
  (total_students enjoy_soccer do_not_enjoy_soccer claim_do_not_enjoy honesty fraction_3_over_11 : ℕ)
  (h1 : enjoy_soccer = total_students / 2)
  (h2 : do_not_enjoy_soccer = total_students / 2)
  (h3 : claim_do_not_enjoy = enjoy_soccer * 3 / 10)
  (h4 : honesty = do_not_enjoy_soccer * 8 / 10)
  (h5 : fraction_3_over_11 = enjoy_soccer * 3 / (10 * (enjoy_soccer * 3 / 10 + do_not_enjoy_soccer * 2 / 10)))
  : fraction_3_over_11 = 3 / 11 :=
sorry

end soccer_claim_fraction_l105_105772


namespace convert_mps_to_kmph_l105_105463

-- Define the conversion factor
def conversion_factor : ℝ := 3.6

-- Define the initial speed in meters per second
def initial_speed_mps : ℝ := 50

-- Define the target speed in kilometers per hour
def target_speed_kmph : ℝ := 180

-- Problem statement: Prove the conversion is correct
theorem convert_mps_to_kmph : initial_speed_mps * conversion_factor = target_speed_kmph := by
  sorry

end convert_mps_to_kmph_l105_105463


namespace total_pencils_l105_105296

   variables (n p t : ℕ)

   -- Condition 1: number of students
   def students := 12

   -- Condition 2: pencils per student
   def pencils_per_student := 3

   -- Theorem statement: Given the conditions, the total number of pencils given by the teacher is 36
   theorem total_pencils : t = students * pencils_per_student :=
   by
   sorry
   
end total_pencils_l105_105296


namespace Norm_photo_count_l105_105373

variables (L M N : ℕ)

-- Conditions from the problem
def cond1 : Prop := L = N - 60
def cond2 : Prop := N = 2 * L + 10

-- Given the conditions, prove N = 110
theorem Norm_photo_count (h1 : cond1 L N) (h2 : cond2 L N) : N = 110 :=
by
  sorry

end Norm_photo_count_l105_105373


namespace original_salary_l105_105674

-- Given conditions as definitions
def salaryAfterRaise (x : ℝ) : ℝ := 1.10 * x
def salaryAfterReduction (x : ℝ) : ℝ := salaryAfterRaise x * 0.95
def finalSalary : ℝ := 1045

-- Statement to prove
theorem original_salary (x : ℝ) (h : salaryAfterReduction x = finalSalary) : x = 1000 :=
by
  sorry

end original_salary_l105_105674


namespace range_of_y_l105_105402

theorem range_of_y (m n k y : ℝ)
  (h₁ : 0 ≤ m)
  (h₂ : 0 ≤ n)
  (h₃ : 0 ≤ k)
  (h₄ : m - k + 1 = 1)
  (h₅ : 2 * k + n = 1)
  (h₆ : y = 2 * k^2 - 8 * k + 6)
  : 5 / 2 ≤ y ∧ y ≤ 6 :=
by
  sorry

end range_of_y_l105_105402


namespace area_square_field_l105_105062

-- Define the side length of the square
def side_length : ℕ := 12

-- Define the area of the square with the given side length
def area_of_square (side : ℕ) : ℕ := side * side

-- The theorem to state and prove
theorem area_square_field : area_of_square side_length = 144 :=
by
  sorry

end area_square_field_l105_105062


namespace problem_1_problem_2_l105_105121

theorem problem_1 (A B C : ℝ) (h_cond : (abs (B - A)) * (abs (C - A)) * (Real.cos A) = 3 * (abs (A - B)) * (abs (C - B)) * (Real.cos B)) : 
  (Real.tan B = 3 * Real.tan A) := 
sorry

theorem problem_2 (A B C : ℝ) (h_cosC : Real.cos C = Real.sqrt 5 / 5) (h_tanB : Real.tan B = 3 * Real.tan A) : 
  (A = Real.pi / 4) := 
sorry

end problem_1_problem_2_l105_105121


namespace max_digits_in_product_l105_105108

theorem max_digits_in_product :
  let n := (99999 : Nat)
  let m := (999 : Nat)
  let product := n * m
  ∃ d : Nat, product < 10^d ∧ 10^(d-1) ≤ product :=
by
  sorry

end max_digits_in_product_l105_105108


namespace coordinate_of_M_l105_105590

-- Definition and given conditions
def L : ℚ := 1 / 6
def P : ℚ := 1 / 12

def divides_into_three_equal_parts (L P M N : ℚ) : Prop :=
  M = L + (P - L) / 3 ∧ N = L + 2 * (P - L) / 3

theorem coordinate_of_M (M N : ℚ) 
  (h1 : divides_into_three_equal_parts L P M N) : 
  M = 1 / 9 := 
by 
  sorry
  
end coordinate_of_M_l105_105590


namespace find_ratio_of_sides_l105_105479

variable {A B : ℝ}
variable {a b : ℝ}

-- Given condition
axiom given_condition : a * Real.sin A * Real.sin B + b * (Real.cos A)^2 = a * Real.sqrt 3

-- Theorem we need to prove
theorem find_ratio_of_sides (h : a ≠ 0) : b / a = Real.sqrt 3 / 3 :=
by
  sorry

end find_ratio_of_sides_l105_105479


namespace largest_circle_area_l105_105633

theorem largest_circle_area (x y : ℝ) (h1 : x + y = 30) (h2 : x * y = 200) :
  ∃ r : ℝ, (2 * π * r = 60) ∧ (π * r ^ 2 = 900 / π) := 
sorry

end largest_circle_area_l105_105633


namespace aubrey_distance_from_school_l105_105748

-- Define average speed and travel time
def average_speed : ℝ := 22 -- in miles per hour
def travel_time : ℝ := 4 -- in hours

-- Define the distance function
def calc_distance (speed time : ℝ) : ℝ := speed * time

-- State the theorem
theorem aubrey_distance_from_school : calc_distance average_speed travel_time = 88 := 
by
  sorry

end aubrey_distance_from_school_l105_105748


namespace total_students_l105_105555

-- Definition of variables and conditions
def M := 50
def E := 4 * M - 3

-- Statement of the theorem to prove
theorem total_students : E + M = 247 := by
  sorry

end total_students_l105_105555


namespace ranking_of_anna_bella_carol_l105_105905

-- Define three people and their scores
variables (Anna Bella Carol : ℕ)

-- Define conditions based on problem statements
axiom Anna_not_highest : ∃ x : ℕ, x > Anna
axiom Bella_not_lowest : ∃ x : ℕ, x < Bella
axiom Bella_higher_than_Carol : Bella > Carol

-- The theorem to be proven
theorem ranking_of_anna_bella_carol (h : Anna < Bella ∧ Carol < Anna) :
  (Bella > Anna ∧ Anna > Carol) :=
by sorry

end ranking_of_anna_bella_carol_l105_105905


namespace Mart_income_percentage_of_Juan_l105_105198

variable (J T M : ℝ)

-- Conditions
def Tim_income_def : Prop := T = 0.5 * J
def Mart_income_def : Prop := M = 1.6 * T

-- Theorem to prove
theorem Mart_income_percentage_of_Juan
  (h1 : Tim_income_def T J) 
  (h2 : Mart_income_def M T) : 
  (M / J) * 100 = 80 :=
by
  sorry

end Mart_income_percentage_of_Juan_l105_105198


namespace solve_for_x_l105_105561

theorem solve_for_x (x y : ℝ) (h1 : 3 * x - 2 * y = 7) (h2 : x + 3 * y = 6) : x = 3 := 
by 
  sorry

end solve_for_x_l105_105561


namespace consecutive_numbers_difference_l105_105139

theorem consecutive_numbers_difference :
  ∃ (n : ℕ), (n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) = 105) → (n + 5 - n = 5) :=
by {
  sorry
}

end consecutive_numbers_difference_l105_105139


namespace find_x_value_l105_105620

theorem find_x_value (X : ℕ) 
  (top_left : ℕ := 2)
  (top_second : ℕ := 3)
  (top_last : ℕ := 4)
  (bottom_left : ℕ := 3)
  (bottom_middle : ℕ := 5) 
  (top_sum_eq: 2 + 3 + X + 4 = 9 + X)
  (bottom_sum_eq: 3 + 5 + (X + 1) = 9 + X) : 
  X = 1 := by 
  sorry

end find_x_value_l105_105620


namespace infinite_series_converges_l105_105347

theorem infinite_series_converges :
  (∑' n : ℕ, if n > 0 then (3 * n - 2) / (n * (n + 1) * (n + 3)) else 0) = 7 / 6 :=
by
  sorry

end infinite_series_converges_l105_105347


namespace find_smallest_number_l105_105419

variable (x : ℕ)

def second_number := 2 * x
def third_number := 4 * second_number x
def average := (x + second_number x + third_number x) / 3

theorem find_smallest_number (h : average x = 165) : x = 45 := by
  sorry

end find_smallest_number_l105_105419


namespace find_a_l105_105064

-- Define the binomial coefficient function in Lean
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- Define the conditions and the proof problem statement
theorem find_a (a : ℝ) (h: (-a)^7 * binomial 10 7 = -120) : a = 1 :=
sorry

end find_a_l105_105064


namespace batsman_average_l105_105685

/-- The average after 12 innings given that the batsman makes a score of 115 in his 12th innings,
     increases his average by 3 runs, and he had never been 'not out'. -/
theorem batsman_average (A : ℕ) (h1 : 11 * A + 115 = 12 * (A + 3)) : A + 3 = 82 := 
by
  sorry

end batsman_average_l105_105685


namespace distinct_special_sums_l105_105946

def is_special_fraction (a b : ℕ) : Prop := a + b = 18

def is_special_sum (n : ℤ) : Prop :=
  ∃ (a1 b1 a2 b2 : ℕ), is_special_fraction a1 b1 ∧ is_special_fraction a2 b2 ∧ 
  n = (a1 : ℤ) * (b2 : ℤ) * b1 + (a2 : ℤ) * (b1 : ℤ) / a1

theorem distinct_special_sums : 
  (∃ (sums : Finset ℤ), 
    (∀ n, n ∈ sums ↔ is_special_sum n) ∧ 
    sums.card = 7) :=
sorry

end distinct_special_sums_l105_105946


namespace parabola_standard_equation_l105_105563

/-- Given that the directrix of a parabola coincides with the line on which the circles 
    x^2 + y^2 - 4 = 0 and x^2 + y^2 + y - 3 = 0 lie, the standard equation of the parabola 
    is x^2 = 4y.
-/
theorem parabola_standard_equation :
  (∀ x y : ℝ, x^2 + y^2 - 4 = 0 → x^2 + y^2 + y - 3 = 0 → y = -1) →
  ∀ p : ℝ, 4 * (p / 2) = 4 → x^2 = 4 * p * y :=
by
  sorry

end parabola_standard_equation_l105_105563


namespace campers_difference_l105_105412

theorem campers_difference 
       (total : ℕ)
       (campers_two_weeks_ago : ℕ) 
       (campers_last_week : ℕ) 
       (diff: ℕ)
       (h_total : total = 150)
       (h_two_weeks_ago : campers_two_weeks_ago = 40) 
       (h_last_week : campers_last_week = 80) : 
       diff = campers_two_weeks_ago - (total - campers_two_weeks_ago - campers_last_week) :=
by
  sorry

end campers_difference_l105_105412


namespace f_ln2_add_f_ln_half_l105_105236

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (1 + 9 * x ^ 2) - 3 * x) + 1

theorem f_ln2_add_f_ln_half :
  f (Real.log 2) + f (Real.log (1 / 2)) = 2 :=
by
  sorry

end f_ln2_add_f_ln_half_l105_105236


namespace clock_correct_time_fraction_l105_105193

theorem clock_correct_time_fraction :
  let hours := 24
  let incorrect_hours := 6
  let correct_hours_fraction := (hours - incorrect_hours) / hours
  let minutes_per_hour := 60
  let incorrect_minutes_per_hour := 15
  let correct_minutes_fraction := (minutes_per_hour - incorrect_minutes_per_hour) / minutes_per_hour
  correct_hours_fraction * correct_minutes_fraction = (9 / 16) :=
by 
  sorry

end clock_correct_time_fraction_l105_105193


namespace shortest_part_is_15_l105_105376

namespace ProofProblem

def rope_length : ℕ := 60
def ratio_part1 : ℕ := 3
def ratio_part2 : ℕ := 4
def ratio_part3 : ℕ := 5

def total_parts := ratio_part1 + ratio_part2 + ratio_part3
def length_per_part := rope_length / total_parts
def shortest_part_length := ratio_part1 * length_per_part

theorem shortest_part_is_15 :
  shortest_part_length = 15 := by
  sorry

end ProofProblem

end shortest_part_is_15_l105_105376


namespace sin_double_angle_l105_105661

noncomputable def unit_circle_point :=
  (1 / 2, Real.sqrt (1 - (1 / 2) ^ 2))

theorem sin_double_angle 
  (α : Real)
  (h1 : (1 / 2, Real.sqrt (1 - (1 / 2) ^ 2)) ∈ { p : ℝ × ℝ | p.1^2 + p.2^2 = 1 })
  (h2 : α = (Real.arccos (1 / 2)) ∨ α = -(Real.arccos (1 / 2))) :
  Real.sin (π / 2 + 2 * α) = -1 / 2 :=
by
  sorry

end sin_double_angle_l105_105661


namespace triangle_area_is_rational_l105_105028

-- Definition of the area of a triangle given vertices with integer coordinates
def triangle_area (x1 x2 x3 y1 y2 y3 : ℤ) : ℚ :=
0.5 * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

-- The theorem stating that the area of a triangle formed by points with integer coordinates is rational
theorem triangle_area_is_rational (x1 x2 x3 y1 y2 y3 : ℤ) :
  ∃ (area : ℚ), area = triangle_area x1 x2 x3 y1 y2 y3 :=
by
  sorry

end triangle_area_is_rational_l105_105028


namespace ladder_base_distance_l105_105711

theorem ladder_base_distance
  (c : ℕ) (b : ℕ) (hypotenuse : c = 13) (wall_height : b = 12) :
  ∃ x : ℕ, x^2 + b^2 = c^2 ∧ x = 5 := by
  sorry

end ladder_base_distance_l105_105711


namespace chef_sold_12_meals_l105_105699

theorem chef_sold_12_meals
  (initial_meals_lunch : ℕ)
  (additional_meals_dinner : ℕ)
  (meals_left_after_lunch : ℕ)
  (meals_for_dinner : ℕ)
  (H1 : initial_meals_lunch = 17)
  (H2 : additional_meals_dinner = 5)
  (H3 : meals_for_dinner = 10) :
  ∃ (meals_sold_lunch : ℕ), meals_sold_lunch = 12 := by
  sorry

end chef_sold_12_meals_l105_105699


namespace talia_mom_age_to_talia_age_ratio_l105_105785

-- Definitions for the problem
def Talia_current_age : ℕ := 13
def Talia_mom_current_age : ℕ := 39
def Talia_father_current_age : ℕ := 36

-- These definitions match the conditions in the math problem
def condition1 : Prop := Talia_current_age + 7 = 20
def condition2 : Prop := Talia_father_current_age + 3 = Talia_mom_current_age
def condition3 : Prop := Talia_father_current_age = 36

-- The ratio calculation
def ratio := Talia_mom_current_age / Talia_current_age

-- The main theorem to prove
theorem talia_mom_age_to_talia_age_ratio :
  condition1 ∧ condition2 ∧ condition3 → ratio = 3 := by
  sorry

end talia_mom_age_to_talia_age_ratio_l105_105785


namespace no_simultaneous_negative_values_l105_105692

theorem no_simultaneous_negative_values (m n : ℝ) :
  ¬ ((3*m^2 + 4*m*n - 2*n^2 < 0) ∧ (-m^2 - 4*m*n + 3*n^2 < 0)) :=
by
  sorry

end no_simultaneous_negative_values_l105_105692


namespace base_eight_to_base_ten_l105_105031

theorem base_eight_to_base_ten (n : ℕ) (h : n = 4 * 8^1 + 7 * 8^0) : n = 39 := by
  sorry

end base_eight_to_base_ten_l105_105031


namespace calculate_product_l105_105398

theorem calculate_product : 6^6 * 3^6 = 34012224 := by
  sorry

end calculate_product_l105_105398


namespace num_ordered_triples_pos_int_l105_105541

theorem num_ordered_triples_pos_int
  (lcm_ab: lcm a b = 180)
  (lcm_ac: lcm a c = 450)
  (lcm_bc: lcm b c = 1200)
  (gcd_abc: gcd (gcd a b) c = 3) :
  ∃ n: ℕ, n = 4 :=
sorry

end num_ordered_triples_pos_int_l105_105541


namespace find_ab_l105_105054

theorem find_ab (a b : ℤ) :
  (∀ x : ℤ, x^3 + a * x^2 + b * x + 5 % (x - 1) = 7) ∧ (∀ x : ℤ, x^3 + a * x^2 + b * x + 5 % (x + 1) = 9) →
  (a, b) = (3, -2) := 
by
  sorry

end find_ab_l105_105054


namespace phase_shift_sin_l105_105814

theorem phase_shift_sin (x : ℝ) : 
  let B := 4
  let C := - (π / 2)
  let φ := - C / B
  φ = π / 8 := 
by 
  sorry

end phase_shift_sin_l105_105814


namespace arithmetic_geometric_sequence_l105_105877

theorem arithmetic_geometric_sequence (S : ℕ → ℝ) (a : ℕ → ℝ) (q : ℝ) 
(h1 : S 3 = 2) 
(h2 : S 6 = 18) 
(h3 : ∀ n, S n = a 1 * (1 - q^n) / (1 - q)) 
: S 10 / S 5 = 33 := 
sorry

end arithmetic_geometric_sequence_l105_105877


namespace power_of_two_sequence_invariant_l105_105433

theorem power_of_two_sequence_invariant
  (n : ℕ)
  (a b : ℕ → ℕ)
  (h₀ : a 0 = 1)
  (h₁ : b 0 = n)
  (hi : ∀ i : ℕ, a i < b i → a (i + 1) = 2 * a i + 1 ∧ b (i + 1) = b i - a i - 1)
  (hj : ∀ i : ℕ, a i > b i → a (i + 1) = a i - b i - 1 ∧ b (i + 1) = 2 * b i + 1)
  (hk : ∀ i : ℕ, a i = b i → a (i + 1) = a i ∧ b (i + 1) = b i)
  (k : ℕ)
  (h : a k = b k) :
  ∃ m : ℕ, n + 3 = 2 ^ m :=
by
  sorry

end power_of_two_sequence_invariant_l105_105433


namespace discount_difference_l105_105618

noncomputable def single_discount (amount : ℝ) (rate : ℝ) : ℝ :=
  amount * (1 - rate)

noncomputable def successive_discounts (amount : ℝ) (rates : List ℝ) : ℝ :=
  rates.foldl (λ acc rate => acc * (1 - rate)) amount

theorem discount_difference:
  let amount := 12000
  let single_rate := 0.35
  let successive_rates := [0.25, 0.08, 0.02]
  single_discount amount single_rate - successive_discounts amount successive_rates = 314.4 := 
  sorry

end discount_difference_l105_105618


namespace initial_ratio_l105_105981

-- Definitions of the initial state and conditions
variables (M W : ℕ)
def initial_men : ℕ := M
def initial_women : ℕ := W
def men_after_entry : ℕ := M + 2
def women_after_exit_and_doubling : ℕ := (W - 3) * 2
def current_men : ℕ := 14
def current_women : ℕ := 24

-- Theorem to prove the initial ratio
theorem initial_ratio (M W : ℕ) 
    (hm : men_after_entry M = current_men)
    (hw : women_after_exit_and_doubling W = current_women) :
  M / Nat.gcd M W = 4 ∧ W / Nat.gcd M W = 5 :=
by
  sorry

end initial_ratio_l105_105981


namespace zoe_total_cost_correct_l105_105911

theorem zoe_total_cost_correct :
  (6 * 0.5) + (6 * (1 + 2 * 0.75)) + (6 * 2 * 3) = 54 :=
by
  sorry

end zoe_total_cost_correct_l105_105911


namespace inequality_1_system_of_inequalities_l105_105762

-- Statement for inequality (1)
theorem inequality_1 (x : ℝ) : 2 - x ≥ (x - 1) / 3 - 1 → x ≤ 2.5 := 
sorry

-- Statement for system of inequalities (2)
theorem system_of_inequalities (x : ℝ) : 
  (5 * x + 1 < 3 * (x - 1)) ∧ ((x + 8) / 5 < (2 * x - 5) / 3 - 1) → false := 
sorry

end inequality_1_system_of_inequalities_l105_105762


namespace BC_length_47_l105_105299

theorem BC_length_47 (A B C D : ℝ) (h₁ : A ≠ B) (h₂ : B ≠ C) (h₃ : B ≠ D)
  (h₄ : dist A C = 20) (h₅ : dist A D = 45) (h₆ : dist B D = 13)
  (h₇ : C = 0) (h₈ : D = 0) (h₉ : B = A + 43) :
  dist B C = 47 :=
sorry

end BC_length_47_l105_105299


namespace eq_of_plane_contains_points_l105_105683

noncomputable def plane_eq (p q r : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ × ℝ :=
  let ⟨px, py, pz⟩ := p
  let ⟨qx, qy, qz⟩ := q
  let ⟨rx, ry, rz⟩ := r
  -- Vector pq
  let pq := (qx - px, qy - py, qz - pz)
  let ⟨pqx, pqy, pqz⟩ := pq
  -- Vector pr
  let pr := (rx - px, ry - py, rz - pz)
  let ⟨prx, pry, prz⟩ := pr
  -- Normal vector via cross product
  let norm := (pqy * prz - pqz * pry, pqz * prx - pqx * prz, pqx * pry - pqy * prx)
  let ⟨nx, ny, nz⟩ := norm
  -- Use normalized normal vector (1, 2, -2)
  (1, 2, -2, -(1 * px + 2 * py + -2 * pz))

theorem eq_of_plane_contains_points : 
  plane_eq (-2, 5, -3) (2, 5, -1) (4, 3, -2) = (1, 2, -2, -14) :=
by
  sorry

end eq_of_plane_contains_points_l105_105683


namespace find_integer_triplets_l105_105257

theorem find_integer_triplets (x y z : ℤ) : 
  x^3 + y^3 + z^3 - 3 * x * y * z = 2003 ↔ (x = 668 ∧ y = 668 ∧ z = 667) ∨ (x = 668 ∧ y = 667 ∧ z = 668) ∨ (x = 667 ∧ y = 668 ∧ z = 668) :=
by
  sorry

end find_integer_triplets_l105_105257


namespace complement_union_l105_105558

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {1, 3, 5}
def B : Set ℕ := {3, 4, 5}
def complementU (A B : Set ℕ) : Set ℕ := U \ (A ∪ B)

theorem complement_union :
  complementU A B = {2, 6} := by
  sorry

end complement_union_l105_105558


namespace probability_of_exactly_three_positives_l105_105930

noncomputable def choose (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem probability_of_exactly_three_positives :
  let p := 2/5
  let n := 7
  let k := 3
  let positive_prob := p^k
  let negative_prob := (1 - p)^(n - k)
  let binomial_coefficient := choose n k
  binomial_coefficient * positive_prob * negative_prob = 22680/78125 := 
by
  sorry

end probability_of_exactly_three_positives_l105_105930


namespace graph_of_equation_is_two_lines_l105_105372

theorem graph_of_equation_is_two_lines :
  ∀ (x y : ℝ), (2 * x - y)^2 = 4 * x^2 - y^2 ↔ (y = 0 ∨ y = 2 * x) :=
by
  sorry

end graph_of_equation_is_two_lines_l105_105372


namespace arithmetic_sequence_problem_l105_105301

theorem arithmetic_sequence_problem (a : Nat → Int) (d a1 : Int)
  (h1 : ∀ n, a n = a1 + (n - 1) * d) 
  (h2 : a 1 + 3 * a 8 = 1560) :
  2 * a 9 - a 10 = 507 :=
sorry

end arithmetic_sequence_problem_l105_105301


namespace price_of_light_bulb_and_motor_l105_105591

theorem price_of_light_bulb_and_motor
  (x : ℝ) (motor_price : ℝ)
  (h1 : x + motor_price = 12)
  (h2 : 10 / x = 2 * 45 / (12 - x)) :
  x = 3 ∧ motor_price = 9 :=
sorry

end price_of_light_bulb_and_motor_l105_105591


namespace min_a_add_c_l105_105179

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
noncomputable def c : ℝ := sorry
noncomputable def angle_ABC : ℝ := 2 * Real.pi / 3
noncomputable def BD : ℝ := 1

-- The bisector of angle ABC intersects AC at point D
-- Angle bisector theorem and the given information
theorem min_a_add_c : ∃ a c : ℝ, (angle_ABC = 2 * Real.pi / 3) → (BD = 1) → (a * c = a + c) → (a + c ≥ 4) :=
by
  sorry

end min_a_add_c_l105_105179


namespace number_line_point_B_l105_105487

theorem number_line_point_B (A B : ℝ) (AB : ℝ) (h1 : AB = 4 * Real.sqrt 2) (h2 : A = 3 * Real.sqrt 2) :
  B = -Real.sqrt 2 ∨ B = 7 * Real.sqrt 2 :=
sorry

end number_line_point_B_l105_105487


namespace alpha_beta_identity_l105_105576

open Real

theorem alpha_beta_identity 
  (α β : ℝ) 
  (hα : 0 < α ∧ α < π / 2) 
  (hβ : 0 < β ∧ β < π / 2)
  (h : cos β = tan α * (1 + sin β)) : 
  2 * α + β = π / 2 :=
by
  sorry

end alpha_beta_identity_l105_105576


namespace teacher_earnings_l105_105356

noncomputable def cost_per_half_hour : ℝ := 10
noncomputable def lesson_duration_in_hours : ℝ := 1
noncomputable def lessons_per_week : ℝ := 1
noncomputable def weeks : ℝ := 5

theorem teacher_earnings : 
  2 * cost_per_half_hour * lesson_duration_in_hours * lessons_per_week * weeks = 100 :=
by
  sorry

end teacher_earnings_l105_105356


namespace base_conversion_subtraction_l105_105725

def base6_to_base10 (n : Nat) : Nat :=
  n / 100000 * 6^5 +
  (n / 10000 % 10) * 6^4 +
  (n / 1000 % 10) * 6^3 +
  (n / 100 % 10) * 6^2 +
  (n / 10 % 10) * 6^1 +
  (n % 10) * 6^0

def base7_to_base10 (n : Nat) : Nat :=
  n / 10000 * 7^4 +
  (n / 1000 % 10) * 7^3 +
  (n / 100 % 10) * 7^2 +
  (n / 10 % 10) * 7^1 +
  (n % 10) * 7^0

theorem base_conversion_subtraction :
  base6_to_base10 543210 - base7_to_base10 43210 = 34052 := by
  sorry

end base_conversion_subtraction_l105_105725


namespace robin_has_43_packages_of_gum_l105_105146

theorem robin_has_43_packages_of_gum (P : ℕ) (h1 : 23 * P + 8 = 997) : P = 43 :=
by
  sorry

end robin_has_43_packages_of_gum_l105_105146


namespace true_proposition_l105_105791

def proposition_p := ∀ (x : ℤ), x^2 > x
def proposition_q := ∃ (x : ℝ) (hx : x > 0), x + (2 / x) > 4

theorem true_proposition :
  (¬ proposition_p) ∨ proposition_q :=
by
  sorry

end true_proposition_l105_105791


namespace count_multiples_of_4_l105_105125

/-- 
Prove that the number of multiples of 4 between 100 and 300 inclusive is 49.
-/
theorem count_multiples_of_4 : 
  ∃ n : ℕ, (∀ k : ℕ, 100 ≤ 4 * k ∧ 4 * k ≤ 300 ↔ k = 26 + n) ∧ n = 48 :=
by
  sorry

end count_multiples_of_4_l105_105125


namespace shortTreesPlanted_l105_105135

-- Definitions based on conditions
def currentShortTrees : ℕ := 31
def tallTrees : ℕ := 32
def futureShortTrees : ℕ := 95

-- The proposition to be proved
theorem shortTreesPlanted :
  futureShortTrees - currentShortTrees = 64 :=
by
  sorry

end shortTreesPlanted_l105_105135


namespace average_age_of_women_l105_105255

noncomputable def avg_age_two_women (M : ℕ) (new_avg : ℕ) (W : ℕ) :=
  let loss := 20 + 10;
  let gain := 2 * 8;
  W = loss + gain

theorem average_age_of_women (M : ℕ) (new_avg : ℕ) (W : ℕ) (avg_age : ℕ) :
  avg_age_two_women M new_avg W →
  avg_age = 23 :=
sorry

#check average_age_of_women

end average_age_of_women_l105_105255


namespace jason_cousins_l105_105469

theorem jason_cousins :
  let dozen := 12
  let cupcakes_bought := 4 * dozen
  let cupcakes_per_cousin := 3
  let number_of_cousins := cupcakes_bought / cupcakes_per_cousin
  number_of_cousins = 16 :=
by
  sorry

end jason_cousins_l105_105469


namespace find_f_0_abs_l105_105771

noncomputable def f : ℝ → ℝ := sorry -- f is a second-degree polynomial with real coefficients

axiom h1 : ∀ (x : ℝ), x = 1 → |f x| = 9
axiom h2 : ∀ (x : ℝ), x = 2 → |f x| = 9
axiom h3 : ∀ (x : ℝ), x = 3 → |f x| = 9

theorem find_f_0_abs : |f 0| = 9 := sorry

end find_f_0_abs_l105_105771


namespace avg_weight_of_class_l105_105252

def A_students : Nat := 36
def B_students : Nat := 44
def C_students : Nat := 50
def D_students : Nat := 30

def A_avg_weight : ℝ := 40
def B_avg_weight : ℝ := 35
def C_avg_weight : ℝ := 42
def D_avg_weight : ℝ := 38

def A_additional_students : Nat := 5
def A_additional_weight : ℝ := 10

def B_reduced_students : Nat := 7
def B_reduced_weight : ℝ := 8

noncomputable def total_weight_class : ℝ :=
  (A_students * A_avg_weight + A_additional_students * A_additional_weight) +
  (B_students * B_avg_weight - B_reduced_students * B_reduced_weight) +
  (C_students * C_avg_weight) +
  (D_students * D_avg_weight)

noncomputable def total_students_class : Nat :=
  A_students + B_students + C_students + D_students

noncomputable def avg_weight_class : ℝ :=
  total_weight_class / total_students_class

theorem avg_weight_of_class :
  avg_weight_class = 38.84 := by
    sorry

end avg_weight_of_class_l105_105252


namespace jessica_quarters_l105_105645

theorem jessica_quarters (quarters_initial quarters_given : Nat) (h_initial : quarters_initial = 8) (h_given : quarters_given = 3) :
  quarters_initial + quarters_given = 11 := by
  sorry

end jessica_quarters_l105_105645


namespace downstream_distance_15_minutes_l105_105486

theorem downstream_distance_15_minutes
  (speed_boat : ℝ) (speed_current : ℝ) (time_minutes : ℝ)
  (h1 : speed_boat = 24)
  (h2 : speed_current = 3)
  (h3 : time_minutes = 15) :
  let effective_speed := speed_boat + speed_current
  let time_hours := time_minutes / 60
  let distance := effective_speed * time_hours
  distance = 6.75 :=
by {
  sorry
}

end downstream_distance_15_minutes_l105_105486


namespace Benjamin_skating_time_l105_105553

-- Definitions based on the conditions in the problem
def distance : ℕ := 80 -- Distance skated in kilometers
def speed : ℕ := 10 -- Speed in kilometers per hour

-- Theorem to prove that the skating time is 8 hours
theorem Benjamin_skating_time : distance / speed = 8 := by
  -- Proof goes here, we skip it with sorry
  sorry

end Benjamin_skating_time_l105_105553


namespace largest_first_term_geometric_progression_l105_105441

noncomputable def geometric_progression_exists (d : ℝ) : Prop :=
  ∃ (a : ℝ), a = 5 ∧ (a + d + 3) / a = (a + 2 * d + 15) / (a + d + 3)

theorem largest_first_term_geometric_progression : ∀ (d : ℝ), 
  d^2 + 6 * d - 36 = 0 → 
  ∃ (a : ℝ), a = 5 ∧ geometric_progression_exists d ∧ a = 5 ∧ 
  ∀ (a' : ℝ), geometric_progression_exists d → a' ≤ a :=
by intros d h; sorry

end largest_first_term_geometric_progression_l105_105441


namespace minimum_knights_in_tournament_l105_105968

def knights_tournament : Prop :=
  ∃ (N : ℕ), (∀ (x : ℕ), x = N / 4 →
    ∃ (k : ℕ), k = (3 * x - 1) / 7 → N = 20)

theorem minimum_knights_in_tournament : knights_tournament :=
  sorry

end minimum_knights_in_tournament_l105_105968


namespace total_seeds_planted_l105_105817

theorem total_seeds_planted 
    (seeds_per_bed : ℕ) 
    (seeds_grow_per_bed : ℕ) 
    (total_flowers : ℕ) 
    (h1 : seeds_per_bed = 15) 
    (h2 : seeds_grow_per_bed = 60) 
    (h3 : total_flowers = 220) : 
    ∃ (total_seeds : ℕ), total_seeds = 85 := 
by
    sorry

end total_seeds_planted_l105_105817


namespace cover_with_L_shapes_l105_105274

def L_shaped (m n : ℕ) : Prop :=
  m > 1 ∧ n > 1 ∧ ∃ k, m * n = 8 * k -- Conditions and tiling pattern coverage.

-- Problem statement as a theorem
theorem cover_with_L_shapes (m n : ℕ) (h1 : m > 1) (h2 : n > 1) : (∃ k, m * n = 8 * k) ↔ L_shaped m n :=
-- Placeholder for the proof
sorry

end cover_with_L_shapes_l105_105274


namespace gcd_10010_15015_l105_105808

def a := 10010
def b := 15015

theorem gcd_10010_15015 : Nat.gcd a b = 5005 := by
  sorry

end gcd_10010_15015_l105_105808


namespace inequality_holds_for_positive_vars_l105_105891

theorem inequality_holds_for_positive_vars (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
    x^2 + y^2 + 1 ≥ x * y + x + y :=
sorry

end inequality_holds_for_positive_vars_l105_105891


namespace binom_7_4_plus_5_l105_105947

theorem binom_7_4_plus_5 : ((Nat.choose 7 4) + 5) = 40 := by
  sorry

end binom_7_4_plus_5_l105_105947


namespace minimum_value_of_y_l105_105546

theorem minimum_value_of_y (x : ℝ) (h : x > 0) : (∃ y, y = (x^2 + 1) / x ∧ y ≥ 2) ∧ (∃ y, y = (x^2 + 1) / x ∧ y = 2) :=
by
  sorry

end minimum_value_of_y_l105_105546


namespace find_unique_function_l105_105012

theorem find_unique_function (f : ℚ → ℚ)
  (h1 : f 1 = 2)
  (h2 : ∀ x y : ℚ, f (x * y) = f x * f y - f (x + y) + 1) :
  ∀ x : ℚ, f x = x + 1 :=
by
  sorry

end find_unique_function_l105_105012


namespace sequence_exists_and_unique_l105_105237

theorem sequence_exists_and_unique (a : ℕ → ℕ) :
  a 0 = 11 ∧ a 7 = 12 ∧
  (∀ n : ℕ, n < 6 → a n + a (n + 1) + a (n + 2) = 50) →
  (a 0 = 11 ∧ a 1 = 12 ∧ a 2 = 27 ∧ a 3 = 11 ∧ a 4 = 12 ∧
   a 5 = 27 ∧ a 6 = 11 ∧ a 7 = 12) :=
by
  sorry

end sequence_exists_and_unique_l105_105237


namespace regular_polygon_properties_l105_105957

theorem regular_polygon_properties
  (n : ℕ)
  (h1 : (n - 2) * 180 = 3 * 360 + 180)
  (h2 : n > 2) :
  n = 9 ∧ (n - 2) * 180 / n = 140 := by
  sorry

end regular_polygon_properties_l105_105957


namespace meal_cost_is_seven_l105_105545

-- Defining the given conditions
def total_cost : ℕ := 21
def number_of_meals : ℕ := 3

-- The amount each meal costs
def meal_cost : ℕ := total_cost / number_of_meals

-- Prove that each meal costs 7 dollars given the conditions
theorem meal_cost_is_seven : meal_cost = 7 :=
by
  -- The result follows directly from the definition of meal_cost
  unfold meal_cost
  have h : 21 / 3 = 7 := by norm_num
  exact h


end meal_cost_is_seven_l105_105545


namespace cos_sin_225_deg_l105_105181

theorem cos_sin_225_deg : (Real.cos (225 * Real.pi / 180) = -Real.sqrt 2 / 2) ∧ (Real.sin (225 * Real.pi / 180) = -Real.sqrt 2 / 2) :=
by
  -- Lean proof steps would go here
  sorry

end cos_sin_225_deg_l105_105181


namespace farmer_land_l105_105587

theorem farmer_land (initial_land remaining_land : ℚ) (h1 : initial_land - initial_land / 10 = remaining_land) (h2 : remaining_land = 10) : initial_land = 100 / 9 := by
  sorry

end farmer_land_l105_105587


namespace words_per_page_l105_105744

/-- 
  Let p denote the number of words per page.
  Given conditions:
  - A book contains 154 pages.
  - Each page has the same number of words, p, and no page contains more than 120 words.
  - The total number of words in the book (154p) is congruent to 250 modulo 227.
  Prove that the number of words in each page p is congruent to 49 modulo 227.
 -/
theorem words_per_page (p : ℕ) (h1 : p ≤ 120) (h2 : 154 * p ≡ 250 [MOD 227]) : p ≡ 49 [MOD 227] :=
sorry

end words_per_page_l105_105744


namespace candy_mixture_price_l105_105234

theorem candy_mixture_price
  (a : ℝ)
  (h1 : 0 < a) -- Assuming positive amount of money spent, to avoid division by zero
  (p1 p2 : ℝ)
  (h2 : p1 = 2)
  (h3 : p2 = 3)
  (h4 : p2 * (a / p2) = p1 * (a / p1)) -- Condition that the total cost for each type is equal.
  : ( (p1 * (a / p1) + p2 * (a / p2)) / (a / p1 + a / p2) = 2.4 ) :=
  sorry

end candy_mixture_price_l105_105234


namespace add_mul_of_3_l105_105831

theorem add_mul_of_3 (a b : ℤ) (ha : ∃ m : ℤ, a = 6 * m) (hb : ∃ n : ℤ, b = 9 * n) : ∃ k : ℤ, a + b = 3 * k :=
by
  sorry

end add_mul_of_3_l105_105831


namespace tangent_line_slope_at_one_l105_105834

variable {f : ℝ → ℝ}

theorem tangent_line_slope_at_one (h : ∀ x, f x = e * x - e) : deriv f 1 = e :=
by sorry

end tangent_line_slope_at_one_l105_105834


namespace total_age_l105_105066

theorem total_age (a b c : ℕ) (h1 : a = b + 2) (h2 : b = 2 * c) (h3 : b = 8) : a + b + c = 22 :=
by
  sorry

end total_age_l105_105066


namespace smallest_digit_N_divisible_by_6_l105_105793

theorem smallest_digit_N_divisible_by_6 : 
  ∃ N : ℕ, N < 10 ∧ 
          (14530 + N) % 6 = 0 ∧
          ∀ M : ℕ, M < N → (14530 + M) % 6 ≠ 0 := sorry

end smallest_digit_N_divisible_by_6_l105_105793


namespace seats_per_bus_l105_105269

theorem seats_per_bus (students : ℕ) (buses : ℕ) (h1 : students = 111) (h2 : buses = 37) : students / buses = 3 := by
  sorry

end seats_per_bus_l105_105269


namespace equilateral_triangle_side_length_l105_105464

variable (R : ℝ)

theorem equilateral_triangle_side_length (R : ℝ) :
  (∃ (s : ℝ), s = R * Real.sqrt 3) :=
sorry

end equilateral_triangle_side_length_l105_105464


namespace obtuse_angle_half_in_first_quadrant_l105_105040

-- Define α to be an obtuse angle
variable {α : ℝ}

-- The main theorem we want to prove
theorem obtuse_angle_half_in_first_quadrant (h_obtuse : (π / 2) < α ∧ α < π) :
  0 < α / 2 ∧ α / 2 < π / 2 :=
  sorry

end obtuse_angle_half_in_first_quadrant_l105_105040


namespace ball_hits_ground_l105_105678

theorem ball_hits_ground : 
  ∃ t : ℚ, -4.9 * t^2 + 4 * t + 10 = 0 ∧ t = 10 / 7 :=
by sorry

end ball_hits_ground_l105_105678


namespace speed_of_stream_l105_105540

-- Definitions
variable (b s : ℝ)
def downstream_distance : ℝ := 120
def downstream_time : ℝ := 4
def upstream_distance : ℝ := 90
def upstream_time : ℝ := 6

-- Equations
def downstream_eq : Prop := downstream_distance = (b + s) * downstream_time
def upstream_eq : Prop := upstream_distance = (b - s) * upstream_time

-- Main statement
theorem speed_of_stream (h₁ : downstream_eq b s) (h₂ : upstream_eq b s) : s = 7.5 :=
by
  sorry

end speed_of_stream_l105_105540


namespace minimum_money_lost_l105_105019

-- Define the conditions and setup the problem

def check_amount : ℕ := 1270
def T_used (F : ℕ) : Σ' T, (T = F + 1 ∨ T = F - 1) :=
sorry

def money_used (T F : ℕ) : ℕ := 10 * T + 50 * F

def total_bills_used (T F : ℕ) : Prop := T + F = 15

theorem minimum_money_lost : (∃ T F, (T = F + 1 ∨ T = F - 1) ∧ T + F = 15 ∧ (check_amount - (10 * T + 50 * F) = 800)) :=
sorry

end minimum_money_lost_l105_105019


namespace range_of_values_l105_105206

theorem range_of_values (a b : ℝ) : (∀ x : ℝ, x < 1 → ax + b > 2 * (x + 1)) → b > 4 := 
by
  sorry

end range_of_values_l105_105206


namespace find_m_l105_105528

theorem find_m (m : ℕ) : (11 - m + 1 = 5) → m = 7 :=
by
  sorry

end find_m_l105_105528


namespace sum_of_samples_is_six_l105_105067

-- Defining the conditions
def grains_varieties : ℕ := 40
def vegetable_oil_varieties : ℕ := 10
def animal_products_varieties : ℕ := 30
def fruits_and_vegetables_varieties : ℕ := 20
def sample_size : ℕ := 20
def total_varieties : ℕ := grains_varieties + vegetable_oil_varieties + animal_products_varieties + fruits_and_vegetables_varieties

def proportion_sample := (sample_size : ℚ) / total_varieties

-- Definitions for the problem
def vegetable_oil_sampled := (vegetable_oil_varieties : ℚ) * proportion_sample
def fruits_and_vegetables_sampled := (fruits_and_vegetables_varieties : ℚ) * proportion_sample

-- Lean 4 statement for the proof problem
theorem sum_of_samples_is_six :
  vegetable_oil_sampled + fruits_and_vegetables_sampled = 6 := by
  sorry

end sum_of_samples_is_six_l105_105067


namespace retirement_percentage_l105_105484

-- Define the conditions
def gross_pay : ℝ := 1120
def tax_deduction : ℝ := 100
def net_paycheck : ℝ := 740

-- Define the total deduction
def total_deduction : ℝ := gross_pay - net_paycheck
def retirement_deduction : ℝ := total_deduction - tax_deduction

-- Define the theorem to prove
theorem retirement_percentage :
  (retirement_deduction / gross_pay) * 100 = 25 :=
by
  sorry

end retirement_percentage_l105_105484


namespace intersection_of_sets_l105_105695

-- Conditions as Lean definitions
def A : Set Int := {-2, -1}
def B : Set Int := {-1, 2, 3}

-- Stating the proof problem in Lean 4
theorem intersection_of_sets : A ∩ B = {-1} :=
by
  sorry

end intersection_of_sets_l105_105695


namespace angle_C_in_triangle_l105_105552

theorem angle_C_in_triangle (A B C : ℝ)
  (hA : A = 60)
  (hAC : C = 2 * B)
  (hSum : A + B + C = 180) : C = 80 :=
sorry

end angle_C_in_triangle_l105_105552


namespace product_of_last_two_digits_l105_105577

theorem product_of_last_two_digits (n : ℤ) (A B : ℤ) :
  (n % 8 = 0) ∧ (A + B = 15) ∧ (n % 10 = B) ∧ (n / 10 % 10 = A) →
  A * B = 54 :=
by
-- Add proof here
sorry

end product_of_last_two_digits_l105_105577


namespace susan_ate_6_candies_l105_105456

-- Definitions for the conditions
def candies_tuesday : ℕ := 3
def candies_thursday : ℕ := 5
def candies_friday : ℕ := 2
def candies_left : ℕ := 4

-- The total number of candies bought during the week
def total_candies_bought : ℕ := candies_tuesday + candies_thursday + candies_friday

-- The number of candies Susan ate during the week
def candies_eaten : ℕ := total_candies_bought - candies_left

-- Theorem statement
theorem susan_ate_6_candies : candies_eaten = 6 :=
by
  unfold candies_eaten total_candies_bought candies_tuesday candies_thursday candies_friday candies_left
  sorry

end susan_ate_6_candies_l105_105456


namespace yearly_payment_split_evenly_l105_105010

def monthly_cost : ℤ := 14
def split_cost (cost : ℤ) := cost / 2
def total_yearly_cost (monthly_payment : ℤ) := monthly_payment * 12

theorem yearly_payment_split_evenly (h : split_cost monthly_cost = 7) :
  total_yearly_cost (split_cost monthly_cost) = 84 :=
by
  -- Here we use the hypothesis h which simplifies the proof.
  sorry

end yearly_payment_split_evenly_l105_105010


namespace sales_last_year_l105_105197

theorem sales_last_year (x : ℝ) (h1 : 416 = (1 + 0.30) * x) : x = 320 :=
by
  sorry

end sales_last_year_l105_105197


namespace interval_where_f_decreasing_minimum_value_of_a_l105_105805

open Real

noncomputable def f (x : ℝ) : ℝ := log x - x^2 + x
noncomputable def h (a x : ℝ) : ℝ := (a - 1) * x^2 + 2 * a * x - 1

theorem interval_where_f_decreasing :
  {x : ℝ | 1 < x} = {x : ℝ | deriv f x < 0} :=
by sorry

theorem minimum_value_of_a (a : ℤ) (ha : ∀ x : ℝ, 0 < x → (a - 1) * x^2 + 2 * a * x - 1 ≥ log x - x^2 + x) :
  a ≥ 1 :=
by sorry

end interval_where_f_decreasing_minimum_value_of_a_l105_105805


namespace points_divisibility_l105_105611

theorem points_divisibility {k n : ℕ} (hkn : k ≤ n) (hpositive : 0 < n) 
  (hcondition : ∀ x : Fin n, (∃ m : ℕ, (∀ y : Fin n, x.val ≤ y.val → y.val ≤ x.val + 1 → True) ∧ m % k = 0)) :
  k ∣ n :=
sorry

end points_divisibility_l105_105611


namespace ratio_of_spent_to_left_after_video_game_l105_105229

-- Definitions based on conditions
def total_money : ℕ := 100
def spent_on_video_game : ℕ := total_money * 1 / 4
def money_left_after_video_game : ℕ := total_money - spent_on_video_game
def money_left_after_goggles : ℕ := 60
def spent_on_goggles : ℕ := money_left_after_video_game - money_left_after_goggles

-- Statement to prove the ratio
theorem ratio_of_spent_to_left_after_video_game :
  (spent_on_goggles : ℚ) / (money_left_after_video_game : ℚ) = 1 / 5 := 
sorry

end ratio_of_spent_to_left_after_video_game_l105_105229


namespace max_value_l105_105915

noncomputable def max_fraction (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : ℝ :=
  (x + y + z)^2 / (x^2 + y^2 + z^2)

theorem max_value (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  max_fraction x y z hx hy hz ≤ 3 :=
sorry

end max_value_l105_105915


namespace equal_share_payment_l105_105862

theorem equal_share_payment (A B C : ℝ) (h : A < B ∧ B < C) :
  (B + C - 2 * A) / 3 + (A + B - 2 * C) / 3 = ((A + B + C) * 2 / 3) - B :=
by
  sorry

end equal_share_payment_l105_105862


namespace certain_event_at_least_one_good_product_l105_105450

-- Define the number of products and their types
def num_products := 12
def num_good_products := 10
def num_defective_products := 2
def num_selected_products := 3

-- Statement of the problem
theorem certain_event_at_least_one_good_product :
  ∀ (selected : Finset (Fin num_products)),
  selected.card = num_selected_products →
  ∃ p ∈ selected, p.val < num_good_products :=
sorry

end certain_event_at_least_one_good_product_l105_105450


namespace problem1_problem2_l105_105106

theorem problem1 : 1 - 2 + 3 + (-4) = -2 :=
sorry

theorem problem2 : (-6) / 3 - (-10) - abs (-8) = 0 :=
sorry

end problem1_problem2_l105_105106


namespace positive_integer_solution_of_inequality_l105_105520

theorem positive_integer_solution_of_inequality (x : ℕ) (h : 0 < x) : (3 * x - 1) / 2 + 1 ≥ 2 * x → x = 1 :=
by
  intros
  sorry

end positive_integer_solution_of_inequality_l105_105520


namespace original_saved_amount_l105_105183

theorem original_saved_amount (x : ℤ) (h : (3 * x - 42)^2 = 2241) : x = 30 := 
sorry

end original_saved_amount_l105_105183


namespace simplification_l105_105948

-- Define all relevant powers
def pow2_8 : ℤ := 2^8
def pow4_5 : ℤ := 4^5
def pow2_3 : ℤ := 2^3
def pow_neg2_2 : ℤ := (-2)^2

-- Define the expression inside the parentheses
def inner_expr : ℤ := pow2_3 - pow_neg2_2

-- Define the exponentiation of the inner expression
def inner_expr_pow11 : ℤ := inner_expr^11

-- Define the entire expression
def full_expr : ℤ := (pow2_8 + pow4_5) * inner_expr_pow11

-- State the proof goal
theorem simplification : full_expr = 5368709120 := by
  sorry

end simplification_l105_105948


namespace right_triangle_hypotenuse_l105_105548

-- Define the right triangle conditions and hypotenuse calculation
theorem right_triangle_hypotenuse (a b c : ℝ) (h1 : b = a + 3) (h2 : 1 / 2 * a * b = 120) :
  c^2 = 425 :=
by
  sorry

end right_triangle_hypotenuse_l105_105548


namespace james_total_time_l105_105696

def time_to_play_main_game : ℕ := 
  let download_time := 10
  let install_time := download_time / 2
  let update_time := download_time * 2
  let account_time := 5
  let internet_issues_time := 15
  let before_tutorial_time := download_time + install_time + update_time + account_time + internet_issues_time
  let tutorial_time := before_tutorial_time * 3
  before_tutorial_time + tutorial_time

theorem james_total_time : time_to_play_main_game = 220 := by
  sorry

end james_total_time_l105_105696


namespace marc_trip_equation_l105_105045

theorem marc_trip_equation (t : ℝ) 
  (before_stop_speed : ℝ := 90)
  (stop_time : ℝ := 0.5)
  (after_stop_speed : ℝ := 110)
  (total_distance : ℝ := 300)
  (total_trip_time : ℝ := 3.5) :
  before_stop_speed * t + after_stop_speed * (total_trip_time - stop_time - t) = total_distance :=
by 
  sorry

end marc_trip_equation_l105_105045


namespace divya_age_l105_105190

theorem divya_age (D N : ℝ) (h1 : N + 5 = 3 * (D + 5)) (h2 : N + D = 40) : D = 7.5 :=
by sorry

end divya_age_l105_105190


namespace last_three_digits_of_power_l105_105264

theorem last_three_digits_of_power (h : 7^500 ≡ 1 [MOD 1250]) : 7^10000 ≡ 1 [MOD 1250] :=
by
  sorry

end last_three_digits_of_power_l105_105264


namespace yellow_chip_value_l105_105104

theorem yellow_chip_value
  (y b g : ℕ)
  (hb : b = g)
  (hchips : y^4 * (4 * b)^b * (5 * g)^g = 16000)
  (h4yellow : y = 2) :
  y = 2 :=
by {
  sorry
}

end yellow_chip_value_l105_105104


namespace probability_class_4_drawn_first_second_l105_105680

noncomputable def P_1 : ℝ := 1 / 10
noncomputable def P_2 : ℝ := 9 / 100

theorem probability_class_4_drawn_first_second :
  P_1 = 1 / 10 ∧ P_2 = 9 / 100 := by
  sorry

end probability_class_4_drawn_first_second_l105_105680


namespace line_in_slope_intercept_form_l105_105011

-- Given the condition
def line_eq (x y : ℝ) : Prop :=
  (2 * (x - 3)) - (y + 4) = 0

-- Prove that the line equation can be expressed as y = 2x - 10.
theorem line_in_slope_intercept_form (x y : ℝ) :
  line_eq x y ↔ y = 2 * x - 10 := 
sorry

end line_in_slope_intercept_form_l105_105011


namespace arithmetic_sequence_problem_l105_105743

theorem arithmetic_sequence_problem
  (a : ℕ → ℝ)
  (h_arithmetic : ∀ n, a (n + 1) = a n + d)
  (h1 : (a 1 - 3) ^ 3 + 3 * (a 1 - 3) = -3)
  (h12 : (a 12 - 3) ^ 3 + 3 * (a 12 - 3) = 3) :
  a 1 < a 12 ∧ (12 * (a 1 + a 12)) / 2 = 36 :=
by
  sorry

end arithmetic_sequence_problem_l105_105743


namespace boat_distance_downstream_l105_105716

-- Definitions
def boat_speed_in_still_water : ℝ := 24
def stream_speed : ℝ := 4
def time_downstream : ℝ := 3

-- Effective speed downstream
def speed_downstream := boat_speed_in_still_water + stream_speed

-- Distance calculation
def distance_downstream := speed_downstream * time_downstream

-- Proof statement
theorem boat_distance_downstream : distance_downstream = 84 := 
by
  -- This is where the proof would go, but we use sorry for now
  sorry

end boat_distance_downstream_l105_105716


namespace average_of_integers_l105_105320

theorem average_of_integers (A B C D : ℤ) (h1 : A < B) (h2 : B < C) (h3 : C < D) (h4 : D = 90) (h5 : 5 ≤ A) (h6 : A ≠ B ∧ B ≠ C ∧ C ≠ D) :
  (A + B + C + D) / 4 = 27 :=
by
  sorry

end average_of_integers_l105_105320


namespace find_f_lg_lg2_l105_105973

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (1 + x ^ 2) - x) + 4

theorem find_f_lg_lg2 :
  f (Real.logb 10 (2)) = 3 :=
sorry

end find_f_lg_lg2_l105_105973


namespace minimize_PA2_PB2_PC2_l105_105622

def point : Type := ℝ × ℝ

noncomputable def distance_sq (P Q : point) : ℝ := 
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2

noncomputable def PA_sq (P : point) : ℝ := distance_sq P (5, 0)
noncomputable def PB_sq (P : point) : ℝ := distance_sq P (0, 5)
noncomputable def PC_sq (P : point) : ℝ := distance_sq P (-4, -3)

noncomputable def circumcircle (P : point) : Prop := 
  P.1^2 + P.2^2 = 25

noncomputable def objective_function (P : point) : ℝ := 
  PA_sq P + PB_sq P + PC_sq P

theorem minimize_PA2_PB2_PC2 : ∃ P : point, circumcircle P ∧ 
  (∀ Q : point, circumcircle Q → objective_function P ≤ objective_function Q) :=
sorry

end minimize_PA2_PB2_PC2_l105_105622


namespace tea_drinking_proof_l105_105992

theorem tea_drinking_proof :
  ∃ (k : ℝ), 
    (∃ (c_sunday t_sunday c_wednesday t_wednesday : ℝ),
      c_sunday = 8.5 ∧ 
      t_sunday = 4 ∧ 
      c_wednesday = 5 ∧ 
      t_sunday * c_sunday = k ∧ 
      t_wednesday * c_wednesday = k ∧ 
      t_wednesday = 6.8) :=
sorry

end tea_drinking_proof_l105_105992


namespace consecutive_integers_equation_l105_105919

theorem consecutive_integers_equation
  (X Y : ℕ)
  (h_consecutive : Y = X + 1)
  (h_equation : 2 * X^2 + 4 * X + 5 * Y + 3 = (X + Y)^2 + 9 * (X + Y) + 4) :
  X + Y = 15 := by
  sorry

end consecutive_integers_equation_l105_105919


namespace constant_term_in_quadratic_eq_l105_105643

theorem constant_term_in_quadratic_eq : 
  ∀ (x : ℝ), (x^2 - 5 * x = 2) → (∃ a b c : ℝ, a = 1 ∧ a * x^2 + b * x + c = 0 ∧ c = -2) :=
by
  sorry

end constant_term_in_quadratic_eq_l105_105643


namespace sum_of_min_x_y_l105_105428

theorem sum_of_min_x_y : ∃ (x y : ℕ), 
  (∃ a b c : ℕ, 180 = 2^a * 3^b * 5^c) ∧
  (∃ u v w : ℕ, 180 * x = 2^u * 3^v * 5^w ∧ u % 4 = 0 ∧ v % 4 = 0 ∧ w % 4 = 0) ∧
  (∃ p q r : ℕ, 180 * y = 2^p * 3^q * 5^r ∧ p % 6 = 0 ∧ q % 6 = 0 ∧ r % 6 = 0) ∧
  (x + y = 4054500) :=
sorry

end sum_of_min_x_y_l105_105428


namespace general_term_formula_l105_105673

noncomputable def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∃ d, ∀ n, a (n + 1) = a n + d

theorem general_term_formula (a : ℕ → ℝ) (S : ℕ → ℝ) (h_arith : is_arithmetic_sequence a)
  (h1 : a 1 + a 5 = (2 / 7) * (a 3) ^ 2) (h2 : S 7 = 63) :
  ∀ n, a n = 2 * n + 1 := by
  sorry

end general_term_formula_l105_105673


namespace find_value_of_f_f_neg1_l105_105093

noncomputable def f (x : ℝ) : ℝ :=
if x < 0 then -2 / x else 3 + Real.log x / Real.log 2

theorem find_value_of_f_f_neg1 :
  f (f (-1)) = 4 := by
  -- proof omitted
  sorry

end find_value_of_f_f_neg1_l105_105093


namespace negative_x_is_positive_l105_105151

theorem negative_x_is_positive (x : ℝ) (hx : x < 0) : -x > 0 :=
sorry

end negative_x_is_positive_l105_105151


namespace eq_no_sol_l105_105265

open Nat -- Use natural number namespace

theorem eq_no_sol (k : ℤ) (x y z : ℕ) (hk1 : k ≠ 1) (hk3 : k ≠ 3) :
  ¬ (x^2 + y^2 + z^2 = k * x * y * z) := 
sorry

end eq_no_sol_l105_105265


namespace max_strong_boys_l105_105256

theorem max_strong_boys (n : ℕ) (h : n = 100) (a b : Fin n → ℕ) 
  (ha : ∀ i j : Fin n, i < j → a i > a j) 
  (hb : ∀ i j : Fin n, i < j → b i < b j) : 
  ∃ k : ℕ, k = n := 
sorry

end max_strong_boys_l105_105256


namespace number_of_girls_l105_105386

theorem number_of_girls 
  (B G : ℕ) 
  (h1 : B + G = 480) 
  (h2 : 5 * B = 3 * G) :
  G = 300 := 
sorry

end number_of_girls_l105_105386


namespace least_value_of_N_l105_105266

theorem least_value_of_N : ∃ (N : ℕ), (N % 6 = 5) ∧ (N % 5 = 4) ∧ (N % 4 = 3) ∧ (N % 3 = 2) ∧ (N % 2 = 1) ∧ N = 59 :=
by
  sorry

end least_value_of_N_l105_105266


namespace find_speed_of_boat_l105_105047

theorem find_speed_of_boat (r d t : ℝ) (x : ℝ) (h_rate : r = 4) (h_dist : d = 33.733333333333334) (h_time : t = 44 / 60) 
  (h_eq : d = (x + r) * t) : x = 42.09090909090909 :=
  sorry

end find_speed_of_boat_l105_105047


namespace find_a9_a10_l105_105387

noncomputable def geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n * r

theorem find_a9_a10 (a : ℕ → ℝ) (r : ℝ)
  (h_geom : geometric_sequence a r)
  (h1 : a 1 + a 2 = 1)
  (h3 : a 3 + a 4 = 2) :
  a 9 + a 10 = 16 := 
sorry

end find_a9_a10_l105_105387


namespace divisibility_of_product_l105_105977

theorem divisibility_of_product (a b : ℕ) (ha : a > 0) (hb : b > 0) (h : (a * b) % 5 = 0) :
  a % 5 = 0 ∨ b % 5 = 0 :=
sorry

end divisibility_of_product_l105_105977


namespace sum_a4_a5_a6_l105_105509

variable {a : ℕ → ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

variables (h1 : is_arithmetic_sequence a)
          (h2 : a 1 + a 2 + a 3 = 6)
          (h3 : a 7 + a 8 + a 9 = 24)

theorem sum_a4_a5_a6 : a 4 + a 5 + a 6 = 15 :=
by
  sorry

end sum_a4_a5_a6_l105_105509


namespace three_point_seven_five_minus_one_point_four_six_l105_105598

theorem three_point_seven_five_minus_one_point_four_six : 3.75 - 1.46 = 2.29 :=
by sorry

end three_point_seven_five_minus_one_point_four_six_l105_105598


namespace no_empty_boxes_prob_l105_105718

def P (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

theorem no_empty_boxes_prob :
  let num_balls := 3
  let num_boxes := 3
  let total_outcomes := num_boxes ^ num_balls
  let favorable_outcomes := P num_balls num_boxes
  let probability := favorable_outcomes / total_outcomes
  probability = 2 / 9 :=
by
  sorry

end no_empty_boxes_prob_l105_105718


namespace roots_can_be_integers_if_q_positive_roots_cannot_both_be_integers_if_q_negative_l105_105221

-- Part (a)
theorem roots_can_be_integers_if_q_positive (p q : ℤ) (hq : q > 0) :
  (∃ x y : ℤ, x * y = q ∧ x + y = p) ∧ (∃ x y : ℤ, x * y = q ∧ x + y = p + 1) :=
sorry

-- Part (b)
theorem roots_cannot_both_be_integers_if_q_negative (p q : ℤ) (hq : q < 0) :
  ¬(∃ x y z w : ℤ, x * y = q ∧ x + y = p ∧ z * w = q ∧ z + w = p + 1) :=
sorry

end roots_can_be_integers_if_q_positive_roots_cannot_both_be_integers_if_q_negative_l105_105221


namespace circle_equation_through_origin_l105_105058

theorem circle_equation_through_origin (focus : ℝ × ℝ) (radius : ℝ) (x y : ℝ) 
  (h1 : focus = (1, 0)) 
  (h2 : (x - 1)^2 + y^2 = radius^2) : 
  x^2 + y^2 - 2*x = 0 :=
by
  sorry

end circle_equation_through_origin_l105_105058


namespace maximize_x5y3_l105_105742

theorem maximize_x5y3 (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 30) :
  x = 18.75 ∧ y = 11.25 → (x^5 * y^3) = (18.75^5 * 11.25^3) :=
sorry

end maximize_x5y3_l105_105742


namespace arithmetic_sequence_1000th_term_l105_105345

theorem arithmetic_sequence_1000th_term (a_1 : ℤ) (d : ℤ) (n : ℤ) (h1 : a_1 = 1) (h2 : d = 3) (h3 : n = 1000) : 
  a_1 + (n - 1) * d = 2998 := 
by
  sorry

end arithmetic_sequence_1000th_term_l105_105345


namespace least_value_of_g_l105_105448

noncomputable def g (x : ℝ) : ℝ := 3 * x ^ 2 + 6 * x + 1

theorem least_value_of_g : ∃ x : ℝ, ∀ y : ℝ, g y ≥ g x ∧ g x = -2 := by
  sorry

end least_value_of_g_l105_105448


namespace sum_nk_l105_105653

theorem sum_nk (n k : ℕ) (h₁ : 3 * n - 4 * k = 4) (h₂ : 4 * n - 5 * k = 13) : n + k = 55 := by
  sorry

end sum_nk_l105_105653


namespace members_in_both_sets_l105_105521

def U : Nat := 193
def B : Nat := 41
def not_A_or_B : Nat := 59
def A : Nat := 116

theorem members_in_both_sets
  (h1 : 193 = U)
  (h2 : 41 = B)
  (h3 : 59 = not_A_or_B)
  (h4 : 116 = A) :
  (U - not_A_or_B) = A + B - 23 :=
by
  sorry

end members_in_both_sets_l105_105521


namespace range_of_independent_variable_l105_105522

theorem range_of_independent_variable (x : ℝ) : x + 3 ≥ 0 ↔ x ≥ -3 := by
  sorry

end range_of_independent_variable_l105_105522


namespace prime_divisor_property_l105_105908

-- Given conditions
variable (p k : ℕ)
variable (prime_p : Nat.Prime p)
variable (divisor_p : p ∣ (2 ^ (2 ^ k)) + 1)

-- The theorem we need to prove
theorem prime_divisor_property (p k : ℕ) (prime_p : Nat.Prime p) (divisor_p : p ∣ (2 ^ (2 ^ k)) + 1) : (2 ^ (k + 1)) ∣ (p - 1) := 
by 
  sorry

end prime_divisor_property_l105_105908


namespace value_of_m_l105_105990

theorem value_of_m (m : ℕ) (h : 3 * 6^4 + m * 6^3 + 5 * 6^2 + 0 * 6^1 + 2 * 6^0 = 4934) : m = 4 :=
by
  sorry

end value_of_m_l105_105990


namespace polynomial_at_1_gcd_of_72_120_168_l105_105177

-- Define the polynomial function
def polynomial (x : ℤ) : ℤ := 5 * x^5 + 4 * x^4 + 3 * x^3 + 2 * x^2 + x - 6

-- Assertion that the polynomial evaluated at x = 1 gives 9
theorem polynomial_at_1 : polynomial 1 = 9 := by
  -- Usually, this is where the detailed Horner's method proof would go
  sorry

-- Define the gcd function for three numbers
def gcd3 (a b c : ℤ) : ℤ := Int.gcd (Int.gcd a b) c

-- Assertion that the GCD of 72, 120, and 168 is 24
theorem gcd_of_72_120_168 : gcd3 72 120 168 = 24 := by
  -- Usually, this is where the detailed Euclidean algorithm proof would go
  sorry

end polynomial_at_1_gcd_of_72_120_168_l105_105177


namespace range_of_a_l105_105619

theorem range_of_a (a : ℝ) : (∀ (x : ℝ), x > 0 → x / (x ^ 2 + 3 * x + 1) ≤ a) → a ≥ 1 / 5 :=
by
  sorry

end range_of_a_l105_105619


namespace square_vectors_l105_105570

theorem square_vectors (AB CD AD : ℝ × ℝ)
  (side_length: ℝ)
  (M N : ℝ × ℝ)
  (x y: ℝ)
  (MN : ℝ × ℝ):
  side_length = 2 →
  M = ((AB.1 + CD.1) / 2, (AB.2 + CD.2) / 2) →
  N = ((CD.1 + AD.1) / 2, (CD.2 + AD.2) / 2) →
  MN = (x * AB.1 + y * AD.1, x * AB.2 + y * AD.2) →
  (x = -1/2) ∧ (y = 1/2) →
  (x * y = -1/4) ∧ ((N.1 - M.1) * AD.1 + (N.2 - M.2) * AD.2 - (N.1 - M.1) * AB.1 - (N.2 - M.2) * AB.2 = -1) :=
by
  intros side_length_cond M_cond N_cond MN_cond xy_cond
  sorry

end square_vectors_l105_105570


namespace student_correct_answers_l105_105485

theorem student_correct_answers (C W : ℕ) 
  (h1 : 4 * C - W = 130) 
  (h2 : C + W = 80) : 
  C = 42 := by
  sorry

end student_correct_answers_l105_105485


namespace least_number_to_subtract_from_724946_l105_105753

def divisible_by_10 (n : ℕ) : Prop :=
  n % 10 = 0

theorem least_number_to_subtract_from_724946 :
  ∃ k : ℕ, k = 6 ∧ divisible_by_10 (724946 - k) :=
by
  sorry

end least_number_to_subtract_from_724946_l105_105753


namespace original_mixture_litres_l105_105801

theorem original_mixture_litres 
  (x : ℝ)
  (h1 : 0.20 * x = 0.15 * (x + 5)) :
  x = 15 :=
sorry

end original_mixture_litres_l105_105801


namespace gcd_largest_value_l105_105020

/-- Given two positive integers x and y such that x + y = 780,
    this definition states that the largest possible value of gcd(x, y) is 390. -/
theorem gcd_largest_value (x y : ℕ) (hx : x > 0) (hy : y > 0) (h : x + y = 780) : ∃ d, d = Nat.gcd x y ∧ d = 390 :=
sorry

end gcd_largest_value_l105_105020


namespace different_suits_card_combinations_l105_105751

theorem different_suits_card_combinations :
  let num_suits := 4
  let suit_cards := 13
  let choose_suits := Nat.choose 4 4
  let ways_per_suit := suit_cards ^ num_suits
  choose_suits * ways_per_suit = 28561 :=
  sorry

end different_suits_card_combinations_l105_105751


namespace min_value_of_quadratic_function_l105_105258

-- Given the quadratic function y = x^2 + 4x - 5
def quadratic_function (x : ℝ) : ℝ :=
  x^2 + 4*x - 5

-- Statement of the proof in Lean 4
theorem min_value_of_quadratic_function :
  ∃ (x_min y_min : ℝ), y_min = quadratic_function x_min ∧
  ∀ x : ℝ, quadratic_function x ≥ y_min ∧ x_min = -2 ∧ y_min = -9 :=
by
  sorry

end min_value_of_quadratic_function_l105_105258


namespace polynomial_root_interval_l105_105922

open Real

theorem polynomial_root_interval (b : ℝ) (x : ℝ) :
  (x^4 + b*x^3 + x^2 + b*x - 1 = 0) → (b ≤ -2 * sqrt 3 ∨ b ≥ 0) :=
sorry

end polynomial_root_interval_l105_105922


namespace weight_in_kilograms_l105_105413

-- Definitions based on conditions
def weight_of_one_bag : ℕ := 250
def number_of_bags : ℕ := 8

-- Converting grams to kilograms (1000 grams = 1 kilogram)
def grams_to_kilograms (grams : ℕ) : ℕ := grams / 1000

-- Total weight in grams
def total_weight_in_grams : ℕ := weight_of_one_bag * number_of_bags

-- Proof that the total weight in kilograms is 2
theorem weight_in_kilograms : grams_to_kilograms total_weight_in_grams = 2 :=
by
  sorry

end weight_in_kilograms_l105_105413


namespace sum_of_next_five_even_integers_l105_105304

theorem sum_of_next_five_even_integers (a : ℕ) (x : ℕ) 
  (h : a = x + (x + 2) + (x + 4) + (x + 6) + (x + 8)) : 
  (x + 10) + (x + 12) + (x + 14) + (x + 16) + (x + 18) = a + 50 := by
  sorry

end sum_of_next_five_even_integers_l105_105304


namespace find_slope_angle_l105_105741

theorem find_slope_angle (α : ℝ) :
    (∃ x y : ℝ, x * Real.sin (2 * Real.pi / 5) + y * Real.cos (2 * Real.pi / 5) = 0) →
    α = 3 * Real.pi / 5 :=
by
  intro h
  sorry

end find_slope_angle_l105_105741


namespace sum_of_first_15_odd_positive_integers_l105_105858

theorem sum_of_first_15_odd_positive_integers :
  let a := 1
  let d := 2
  let n := 15
  let l := a + (n - 1) * d
  let S_n := (n / 2) * (a + l)
  S_n = 225 :=
by
  let a := 1
  let d := 2
  let n := 15
  let l := a + (n - 1) * d
  let S_n := (n / 2) * (a + l)
  have : S_n = 225 := sorry
  exact this

end sum_of_first_15_odd_positive_integers_l105_105858


namespace vector_parallel_m_eq_two_neg_two_l105_105420

theorem vector_parallel_m_eq_two_neg_two (m : ℝ) (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : 2 / x = m / y) : m = 2 ∨ m = -2 :=
by
  sorry

end vector_parallel_m_eq_two_neg_two_l105_105420


namespace mean_score_classes_is_82_l105_105166

theorem mean_score_classes_is_82
  (F S : ℕ)
  (f s : ℕ)
  (hF : F = 90)
  (hS : S = 75)
  (hf_ratio : f * 6 = s * 5)
  (hf_total : f + s = 66) :
  ((F * f + S * s) / (f + s) : ℚ) = 82 :=
by
  sorry

end mean_score_classes_is_82_l105_105166


namespace new_students_count_l105_105727

-- Define the conditions as given in the problem statement.
def original_average_age := 40
def original_number_students := 17
def new_students_average_age := 32
def decreased_age := 36  -- Since the average decreases by 4 years from 40 to 36

-- Let x be the number of new students, the proof problem is to find x.
def find_new_students (x : ℕ) : Prop :=
  original_average_age * original_number_students + new_students_average_age * x = decreased_age * (original_number_students + x)

-- Prove that find_new_students(x) holds for x = 17
theorem new_students_count : find_new_students 17 :=
by
  sorry -- the proof goes here

end new_students_count_l105_105727


namespace rockham_soccer_league_l105_105357

theorem rockham_soccer_league (cost_socks : ℕ) (cost_tshirt : ℕ) (custom_fee : ℕ) (total_cost : ℕ) :
  cost_socks = 6 →
  cost_tshirt = cost_socks + 7 →
  custom_fee = 200 →
  total_cost = 2892 →
  ∃ members : ℕ, total_cost - custom_fee = members * (2 * (cost_socks + cost_tshirt)) ∧ members = 70 :=
by
  intros
  sorry

end rockham_soccer_league_l105_105357


namespace inequality_abc_l105_105122

variable {a b c : ℝ}

theorem inequality_abc
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a * b * c = 1) :
  (a - 1 + 1 / b) * (b - 1 + 1 / c) * (c - 1 + 1 / a) ≤ 1 := 
by
  sorry

end inequality_abc_l105_105122


namespace minimum_value_l105_105606

theorem minimum_value (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x * y * z = 128) : 
  ∃ (m : ℝ), (∀ (a b c : ℝ), 0 < a → 0 < b → 0 < c → a * b * c = 128 → (a^2 + 8 * a * b + 4 * b^2 + 8 * c^2) ≥ m) 
  ∧ m = 384 :=
sorry


end minimum_value_l105_105606


namespace absolute_value_example_l105_105446

theorem absolute_value_example (x : ℝ) (h : x = 4) : |x - 5| = 1 :=
by
  sorry

end absolute_value_example_l105_105446


namespace solve_x_for_equation_l105_105971

theorem solve_x_for_equation (x : ℝ) (h : 2 / (x + 3) + 3 * x / (x + 3) - 4 / (x + 3) = 4) : x = -14 :=
by 
  sorry

end solve_x_for_equation_l105_105971


namespace remaining_episodes_l105_105303

theorem remaining_episodes (total_seasons : ℕ) (episodes_per_season : ℕ) (fraction_watched : ℚ) 
  (H1 : total_seasons = 12) (H2 : episodes_per_season = 20) (H3 : fraction_watched = 1/3) : 
  (total_seasons * episodes_per_season) - (fraction_watched * (total_seasons * episodes_per_season)) = 160 :=
by
  sorry

end remaining_episodes_l105_105303


namespace problem_solution_l105_105865

def tens_digit_is_odd (n : ℕ) : Bool :=
  let m := (n * n + n) / 10 % 10
  m % 2 = 1

def count_tens_digit_odd : ℕ :=
  List.range 50 |>.filter tens_digit_is_odd |>.length

theorem problem_solution : count_tens_digit_odd = 25 :=
  sorry

end problem_solution_l105_105865


namespace problem1_problem2_l105_105214

noncomputable def sqrt (x : ℝ) := Real.sqrt x

theorem problem1 : sqrt 12 + sqrt 8 * sqrt 6 = 6 * sqrt 3 := by
  sorry

theorem problem2 : sqrt 12 + 1 / (sqrt 3 - sqrt 2) - sqrt 6 * sqrt 3 = 3 * sqrt 3 - 2 * sqrt 2 := by
  sorry

end problem1_problem2_l105_105214


namespace positive_integers_satisfying_inequality_l105_105854

theorem positive_integers_satisfying_inequality (x : ℕ) (hx : x > 0) : 4 - x > 1 ↔ x = 1 ∨ x = 2 :=
by
  sorry

end positive_integers_satisfying_inequality_l105_105854


namespace travel_time_l105_105937

noncomputable def convert_kmh_to_mps (speed_kmh : ℝ) : ℝ :=
  speed_kmh * 1000 / 3600

theorem travel_time
  (speed_kmh : ℝ)
  (distance_m : ℝ) :
  speed_kmh = 63 →
  distance_m = 437.535 →
  (distance_m / convert_kmh_to_mps speed_kmh) = 25 :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end travel_time_l105_105937


namespace positive_partial_sum_existence_l105_105501

variable {n : ℕ}
variable {a : Fin n → ℝ}

theorem positive_partial_sum_existence (h : (Finset.univ.sum a) > 0) :
  ∃ i : Fin n, ∀ j : Fin n, i ≤ j → (Finset.Icc i j).sum a > 0 := by
  sorry

end positive_partial_sum_existence_l105_105501


namespace slower_train_pass_time_l105_105349

noncomputable def relative_speed_km_per_hr (v1 v2 : ℕ) : ℕ :=
v1 + v2

noncomputable def relative_speed_m_per_s (v_km_per_hr : ℕ) : ℝ :=
(v_km_per_hr * 5) / 18

noncomputable def time_to_pass (distance_m : ℕ) (speed_m_per_s : ℝ) : ℝ :=
distance_m / speed_m_per_s

theorem slower_train_pass_time
  (length_train1 length_train2 : ℕ)
  (speed_train1_km_per_hr speed_train2_km_per_hr : ℕ)
  (distance_to_cover : ℕ)
  (h1 : length_train1 = 800)
  (h2 : length_train2 = 600)
  (h3 : speed_train1_km_per_hr = 85)
  (h4 : speed_train2_km_per_hr = 65)
  (h5 : distance_to_cover = length_train2) :
  time_to_pass distance_to_cover (relative_speed_m_per_s (relative_speed_km_per_hr speed_train1_km_per_hr speed_train2_km_per_hr)) = 14.4 := 
sorry

end slower_train_pass_time_l105_105349


namespace min_value_of_reciprocal_sum_l105_105358

open Real

theorem min_value_of_reciprocal_sum (x y : ℝ) (h1 : x > 0) (h2 : y > 0)
  (h3 : x + y = 12) (h4 : x * y = 20) : (1 / x + 1 / y) = 3 / 5 :=
sorry

end min_value_of_reciprocal_sum_l105_105358


namespace calculate_expression_l105_105468

theorem calculate_expression :
  (0.125: ℝ) ^ 3 * (-8) ^ 3 = -1 := 
by
  sorry

end calculate_expression_l105_105468


namespace number_of_band_students_l105_105285

noncomputable def total_students := 320
noncomputable def sports_students := 200
noncomputable def both_activities_students := 60
noncomputable def either_activity_students := 225

theorem number_of_band_students : 
  ∃ B : ℕ, either_activity_students = B + sports_students - both_activities_students ∧ B = 85 :=
by
  sorry

end number_of_band_students_l105_105285


namespace problem_solution_l105_105162

theorem problem_solution : (90 + 5) * (12 / (180 / (3^2))) = 57 :=
by
  sorry

end problem_solution_l105_105162


namespace obtuse_triangle_existence_l105_105970

theorem obtuse_triangle_existence :
  ∃ (a b c : ℝ), (a = 2 ∧ b = 6 ∧ c = 7 ∧ 
  (a^2 + b^2 < c^2 ∨ b^2 + c^2 < a^2 ∨ c^2 + a^2 < b^2)) ∧
  ¬(6^2 + 7^2 < 8^2 ∨ 7^2 + 8^2 < 6^2 ∨ 8^2 + 6^2 < 7^2) ∧
  ¬(7^2 + 8^2 < 10^2 ∨ 8^2 + 10^2 < 7^2 ∨ 10^2 + 7^2 < 8^2) ∧
  ¬(5^2 + 12^2 < 13^2 ∨ 12^2 + 13^2 < 5^2 ∨ 13^2 + 5^2 < 12^2) :=
sorry

end obtuse_triangle_existence_l105_105970


namespace distance_covered_by_center_of_circle_l105_105659

-- Definition of the sides of the triangle
def side1 : ℕ := 5
def side2 : ℕ := 12
def hypotenuse : ℕ := 13

-- Definition of the circle's radius
def radius : ℕ := 2

-- Define a function that calculates the perimeter of the smaller triangle
noncomputable def smallerTrianglePerimeter (s1 s2 hyp r : ℕ) : ℕ :=
  (s1 - 2 * r) + (s2 - 2 * r) + (hyp - 2 * r)

-- Main theorem statement
theorem distance_covered_by_center_of_circle :
  smallerTrianglePerimeter side1 side2 hypotenuse radius = 18 :=
by
  sorry

end distance_covered_by_center_of_circle_l105_105659


namespace points_per_win_is_5_l105_105722

-- Definitions based on conditions
def rounds_played : ℕ := 30
def vlad_points : ℕ := 64
def taro_points (T : ℕ) : ℕ := (3 * T) / 5 - 4
def total_points (T : ℕ) : ℕ := taro_points T + vlad_points

-- Theorem statement to prove the number of points per win
theorem points_per_win_is_5 (T : ℕ) (H : total_points T = T) : T / rounds_played = 5 := sorry

end points_per_win_is_5_l105_105722


namespace sum_of_diagonals_l105_105888

noncomputable def length_AB : ℝ := 31
noncomputable def length_sides : ℝ := 81

def hexagon_inscribed_in_circle (A B C D E F : Type) : Prop :=
-- Assuming A, B, C, D, E, F are suitable points on a circle
-- Definitions to be added as per detailed proof needs
sorry

theorem sum_of_diagonals (A B C D E F : Type) :
    hexagon_inscribed_in_circle A B C D E F →
    (length_AB + length_sides + length_sides + length_sides + length_sides + length_sides = 384) := 
by
  sorry

end sum_of_diagonals_l105_105888


namespace abs_neg_2023_l105_105400

theorem abs_neg_2023 : |(-2023)| = 2023 :=
by
  sorry

end abs_neg_2023_l105_105400


namespace cory_fruit_eating_orders_l105_105954

open Nat

theorem cory_fruit_eating_orders : 
    let apples := 4
    let oranges := 3
    let bananas := 2
    let grape := 1
    let total_fruits := apples + oranges + bananas + grape
    apples + oranges + bananas + grape = 10 →
    total_fruits = 10 →
    apples ≥ 1 →
    factorial 9 / (factorial 3 * factorial 3 * factorial 2 * factorial 1) = 5040 :=
by
  intros apples oranges bananas grape total_fruits h_total h_sum h_apples
  sorry

end cory_fruit_eating_orders_l105_105954


namespace candies_on_second_day_l105_105335

noncomputable def total_candies := 45
noncomputable def days := 5
noncomputable def difference := 3

def arithmetic_sum (n : ℕ) (a₁ d : ℕ) :=
  (n * (2 * a₁ + (n - 1) * d)) / 2

theorem candies_on_second_day (a : ℕ) (h : arithmetic_sum days a difference = total_candies) :
  a + difference = 6 := by
  sorry

end candies_on_second_day_l105_105335


namespace alcohol_quantity_in_mixture_l105_105697

theorem alcohol_quantity_in_mixture 
  (A W : ℝ)
  (h1 : A / W = 4 / 3)
  (h2 : A / (W + 4) = 4 / 5)
  : A = 8 :=
sorry

end alcohol_quantity_in_mixture_l105_105697


namespace latest_start_time_for_liz_l105_105993

def latest_start_time (weight : ℕ) (roast_time_per_pound : ℕ) (num_turkeys : ℕ) (dinner_time : ℕ) : ℕ :=
  dinner_time - (num_turkeys * weight * roast_time_per_pound) / 60

theorem latest_start_time_for_liz : 
  latest_start_time 16 15 2 18 = 10 := by
  sorry

end latest_start_time_for_liz_l105_105993


namespace product_of_three_equal_numbers_l105_105984

theorem product_of_three_equal_numbers
    (a b : ℕ) (x : ℕ)
    (h1 : a = 12)
    (h2 : b = 22)
    (h_mean : (a + b + 3 * x) / 5 = 20) :
    x * x * x = 10648 := by
  sorry

end product_of_three_equal_numbers_l105_105984


namespace joan_total_cost_is_correct_l105_105016

def year1_home_games := 6
def year1_away_games := 3
def year1_home_playoff_games := 1
def year1_away_playoff_games := 1

def year2_home_games := 2
def year2_away_games := 2
def year2_home_playoff_games := 1
def year2_away_playoff_games := 0

def home_game_ticket := 60
def away_game_ticket := 75
def home_playoff_ticket := 120
def away_playoff_ticket := 100

def friend_home_game_ticket := 45
def friend_away_game_ticket := 75

def home_game_transportation := 25
def away_game_transportation := 50

noncomputable def year1_total_cost : ℕ :=
  (year1_home_games * (home_game_ticket + friend_home_game_ticket + home_game_transportation)) +
  (year1_away_games * (away_game_ticket + friend_away_game_ticket + away_game_transportation)) +
  (year1_home_playoff_games * (home_playoff_ticket + friend_home_game_ticket + home_game_transportation)) +
  (year1_away_playoff_games * (away_playoff_ticket + friend_away_game_ticket + away_game_transportation))

noncomputable def year2_total_cost : ℕ :=
  (year2_home_games * (home_game_ticket + friend_home_game_ticket + home_game_transportation)) +
  (year2_away_games * (away_game_ticket + friend_away_game_ticket + away_game_transportation)) +
  (year2_home_playoff_games * (home_playoff_ticket + friend_home_game_ticket + home_game_transportation)) +
  (year2_away_playoff_games * (away_playoff_ticket + friend_away_game_ticket + away_game_transportation))

noncomputable def total_cost : ℕ := year1_total_cost + year2_total_cost

theorem joan_total_cost_is_correct : total_cost = 2645 := by
  sorry

end joan_total_cost_is_correct_l105_105016


namespace find_larger_number_l105_105339

theorem find_larger_number (a b : ℕ) (h1 : a + b = 96) (h2 : a = b + 12) : a = 54 :=
sorry

end find_larger_number_l105_105339


namespace smallest_b_for_factorization_l105_105292

theorem smallest_b_for_factorization :
  ∃ (b : ℕ), (∀ r s : ℕ, (r * s = 3258) → (b = r + s)) ∧ (∀ c : ℕ, (∀ r' s' : ℕ, (r' * s' = 3258) → (c = r' + s')) → b ≤ c) :=
sorry

end smallest_b_for_factorization_l105_105292


namespace child_tickets_sold_l105_105889

noncomputable def price_adult_ticket : ℝ := 7
noncomputable def price_child_ticket : ℝ := 4
noncomputable def total_tickets_sold : ℝ := 900
noncomputable def total_revenue : ℝ := 5100

theorem child_tickets_sold : ∃ (C : ℝ), price_child_ticket * C + price_adult_ticket * (total_tickets_sold - C) = total_revenue ∧ C = 400 :=
by
  sorry

end child_tickets_sold_l105_105889


namespace scientific_notation_of_1653_billion_l105_105997

theorem scientific_notation_of_1653_billion :
  (1653 * (10 ^ 9) = 1.6553 * (10 ^ 12)) :=
sorry

end scientific_notation_of_1653_billion_l105_105997


namespace volume_of_sphere_in_cone_l105_105867

theorem volume_of_sphere_in_cone :
  let diameter_of_base := 16 * Real.sqrt 2
  let radius_of_base := diameter_of_base / 2
  let side_length := radius_of_base * 2 / Real.sqrt 2
  let inradius := side_length / 2
  let r := inradius
  let V := (4 / 3) * Real.pi * r^3
  V = (2048 / 3) * Real.pi := by
  sorry

end volume_of_sphere_in_cone_l105_105867


namespace summer_sales_is_2_million_l105_105786

def spring_sales : ℝ := 4.8
def autumn_sales : ℝ := 7
def winter_sales : ℝ := 2.2
def spring_percentage : ℝ := 0.3

theorem summer_sales_is_2_million :
  ∃ (total_sales : ℝ), total_sales = (spring_sales / spring_percentage) ∧
  ∃ summer_sales : ℝ, total_sales = spring_sales + summer_sales + autumn_sales + winter_sales ∧
  summer_sales = 2 :=
by
  sorry

end summer_sales_is_2_million_l105_105786


namespace trapezoid_dot_product_ad_bc_l105_105515

-- Define the trapezoid and its properties
variables (A B C D O : Type) (AB CD AO BO : ℝ)
variables (AD BC : ℝ)

-- Conditions from the problem
axiom AB_length : AB = 41
axiom CD_length : CD = 24
axiom diagonals_perpendicular : ∀ (v₁ v₂ : ℝ), (v₁ * v₂ = 0)

-- Using these conditions, prove that the dot product of the vectors AD and BC is 984
theorem trapezoid_dot_product_ad_bc : AD * BC = 984 :=
  sorry

end trapezoid_dot_product_ad_bc_l105_105515


namespace mary_income_is_128_percent_of_juan_income_l105_105273

def juan_income : ℝ := sorry
def tim_income : ℝ := 0.80 * juan_income
def mary_income : ℝ := 1.60 * tim_income

theorem mary_income_is_128_percent_of_juan_income
  (J : ℝ) : mary_income = 1.28 * J :=
by
  sorry

end mary_income_is_128_percent_of_juan_income_l105_105273


namespace geometric_sequence_a3_l105_105802

noncomputable def a_1 (S_4 : ℕ) (q : ℕ) : ℕ :=
  S_4 * (q - 1) / (1 - q^4)

noncomputable def a_3 (a_1 : ℕ) (q : ℕ) : ℕ :=
  a_1 * q^(3 - 1)

theorem geometric_sequence_a3 (a_n : ℕ → ℕ) (S_4 : ℕ) (q : ℕ) :
  (q = 2) →
  (S_4 = 60) →
  a_3 (a_1 S_4 q) q = 16 :=
by
  intro hq hS4
  rw [hq, hS4]
  sorry

end geometric_sequence_a3_l105_105802


namespace mrs_sheridan_cats_l105_105893

theorem mrs_sheridan_cats (initial_cats : ℝ) (given_away_cats : ℝ) (remaining_cats : ℝ) :
  initial_cats = 17.0 → given_away_cats = 14.0 → remaining_cats = (initial_cats - given_away_cats) → remaining_cats = 3.0 :=
by
  intros
  sorry

end mrs_sheridan_cats_l105_105893


namespace minimize_total_price_l105_105156

noncomputable def total_price (a : ℝ) (m x : ℝ) : ℝ :=
  a * ((m / 2 + x)^2 + (m / 2 - x)^2)

theorem minimize_total_price (a m : ℝ) : 
  ∃ y : ℝ, (∀ x, total_price a m x ≥ y) ∧ y = total_price a m 0 :=
by
  sorry

end minimize_total_price_l105_105156


namespace monogram_count_l105_105848

theorem monogram_count :
  ∃ (n : ℕ), n = 156 ∧
    (∃ (beforeM : Fin 13) (afterM : Fin 14),
      ∀ (a : Fin 13) (b : Fin 14),
        a < b → (beforeM = a ∧ afterM = b) → n = 12 * 13
    ) :=
by {
  sorry
}

end monogram_count_l105_105848


namespace solve_arctan_eq_pi_over_3_l105_105329

open Real

theorem solve_arctan_eq_pi_over_3 (x : ℝ) :
  arctan (1 / x) + arctan (1 / x^2) = π / 3 ↔ 
  x = (1 + sqrt (13 + 4 * sqrt 3)) / (2 * sqrt 3) ∨
  x = (1 - sqrt (13 + 4 * sqrt 3)) / (2 * sqrt 3) :=
by
  sorry

end solve_arctan_eq_pi_over_3_l105_105329


namespace moles_of_C2H6_are_1_l105_105708

def moles_of_C2H6_reacted (n_C2H6: ℕ) (n_Cl2: ℕ) (n_C2Cl6: ℕ): Prop :=
  n_Cl2 = 6 ∧ n_C2Cl6 = 1 ∧ (n_C2H6 + 6 * (n_Cl2 - 1) = n_C2Cl6 + 6 * (n_Cl2 - 1))

theorem moles_of_C2H6_are_1:
  ∀ (n_C2H6 n_Cl2 n_C2Cl6: ℕ), moles_of_C2H6_reacted n_C2H6 n_Cl2 n_C2Cl6 → n_C2H6 = 1 :=
by
  intros n_C2H6 n_Cl2 n_C2Cl6 h
  sorry

end moles_of_C2H6_are_1_l105_105708


namespace pearJuicePercentageCorrect_l105_105068

-- Define the conditions
def dozen : ℕ := 12
def pears := dozen
def oranges := dozen
def pearJuiceFrom3Pears : ℚ := 8
def orangeJuiceFrom2Oranges : ℚ := 10
def juiceBlendPears : ℕ := 4
def juiceBlendOranges : ℕ := 4
def pearJuicePerPear : ℚ := pearJuiceFrom3Pears / 3
def orangeJuicePerOrange : ℚ := orangeJuiceFrom2Oranges / 2
def totalPearJuice : ℚ := juiceBlendPears * pearJuicePerPear
def totalOrangeJuice : ℚ := juiceBlendOranges * orangeJuicePerOrange
def totalJuice : ℚ := totalPearJuice + totalOrangeJuice

-- Prove that the percentage of pear juice in the blend is 34.78%
theorem pearJuicePercentageCorrect : 
  (totalPearJuice / totalJuice) * 100 = 34.78 := by
  sorry

end pearJuicePercentageCorrect_l105_105068


namespace problem_statement_l105_105118

def assoc_number (x : ℚ) : ℚ :=
  if x >= 0 then 2 * x - 1 else -2 * x + 1

theorem problem_statement (a b : ℚ) (ha : a > 0) (hb : b < 0) (hab : assoc_number a = assoc_number b) :
  (a + b)^2 - 2 * a - 2 * b = -1 :=
sorry

end problem_statement_l105_105118


namespace pie_eating_contest_l105_105393

def pies_eaten (Adam Bill Sierra Taylor: ℕ) : ℕ :=
  Adam + Bill + Sierra + Taylor

theorem pie_eating_contest (Bill : ℕ) 
  (Adam_eq_Bill_plus_3 : ∀ B: ℕ, Adam = B + 3)
  (Sierra_eq_2times_Bill : ∀ B: ℕ, Sierra = 2 * B)
  (Sierra_eq_12 : Sierra = 12)
  (Taylor_eq_avg : ∀ A B S: ℕ, Taylor = (A + B + S) / 3)
  : pies_eaten Adam Bill Sierra Taylor = 36 := sorry

end pie_eating_contest_l105_105393


namespace number_of_boys_in_class_l105_105005

theorem number_of_boys_in_class (B : ℕ) (G : ℕ) (hG : G = 10) (h_combinations : (G * B * (B - 1)) / 2 = 1050) :
    B = 15 :=
by
  sorry

end number_of_boys_in_class_l105_105005


namespace third_candidate_more_votes_than_john_l105_105359

-- Define the given conditions
def total_votes : ℕ := 1150
def john_votes : ℕ := 150
def remaining_votes : ℕ := total_votes - john_votes
def james_votes : ℕ := (7 * remaining_votes) / 10
def john_and_james_votes : ℕ := john_votes + james_votes
def third_candidate_votes : ℕ := total_votes - john_and_james_votes

-- Stating the problem to prove
theorem third_candidate_more_votes_than_john : third_candidate_votes - john_votes = 150 := 
by
  sorry

end third_candidate_more_votes_than_john_l105_105359


namespace find_m_from_parallel_l105_105836

theorem find_m_from_parallel (m : ℝ) : 
  (∃ (A B : ℝ×ℝ), A = (-2, m) ∧ B = (m, 4) ∧
  (∃ (a b c : ℝ), a = 2 ∧ b = 1 ∧ c = -1 ∧
  (a * (B.1 - A.1) + b * (B.2 - A.2) = 0)) ) 
  → m = -8 :=
by
  sorry

end find_m_from_parallel_l105_105836


namespace carol_total_peanuts_l105_105967

-- Conditions as definitions
def carol_initial_peanuts : Nat := 2
def carol_father_peanuts : Nat := 5

-- Theorem stating that the total number of peanuts Carol has is 7
theorem carol_total_peanuts : carol_initial_peanuts + carol_father_peanuts = 7 := by
  -- Proof would go here, but we use sorry to skip
  sorry

end carol_total_peanuts_l105_105967


namespace problem1_problem2_l105_105308

theorem problem1 (x y : ℝ) (h1 : x - y = 4) (h2 : x > 3) (h3 : y < 1) : 
  2 < x + y ∧ x + y < 6 :=
sorry

theorem problem2 (x y m : ℝ) (h1 : y > 1) (h2 : x < -1) (h3 : x - y = m) : 
  m + 2 < x + y ∧ x + y < -m - 2 :=
sorry

end problem1_problem2_l105_105308


namespace parabola_tangents_intersection_y_coord_l105_105693

theorem parabola_tangents_intersection_y_coord
  (a b : ℝ)
  (ha : A = (a, a^2 + 1))
  (hb : B = (b, b^2 + 1))
  (tangent_perpendicular : ∀ t1 t2 : ℝ, t1 * t2 = -1):
  ∃ y : ℝ, y = 3 / 4 :=
by
  sorry

end parabola_tangents_intersection_y_coord_l105_105693


namespace box_volume_l105_105913

theorem box_volume (x y : ℝ) (hx : 0 < x ∧ x < 6) (hy : 0 < y ∧ y < 8) :
  (16 - 2 * x) * (12 - 2 * y) * y = 192 * y - 32 * y^2 - 24 * x * y + 4 * x * y^2 :=
by
  sorry

end box_volume_l105_105913


namespace monotonic_implies_m_l105_105298

noncomputable def cubic_function (x m : ℝ) : ℝ := x^3 + x^2 + m * x + 1

theorem monotonic_implies_m (m : ℝ) :
  (∀ x : ℝ, (3 * x^2 + 2 * x + m) ≥ 0) → m ≥ 1 / 3 :=
  sorry

end monotonic_implies_m_l105_105298


namespace calculate_tough_week_sales_l105_105142

-- Define the conditions
variables (G T : ℝ)
def condition1 := T = G / 2
def condition2 := 5 * G + 3 * T = 10400

-- By substituting and proving
theorem calculate_tough_week_sales (G T : ℝ) (h1 : condition1 G T) (h2 : condition2 G T) : T = 800 := 
by {
  sorry 
}

end calculate_tough_week_sales_l105_105142


namespace solution_l105_105770

-- Define the problem.
def problem (CD : ℝ) (hexagon_side : ℝ) (CY : ℝ) (BY : ℝ) : Prop :=
  CD = 2 ∧ hexagon_side = 2 ∧ CY = 4 * CD ∧ BY = 9 * Real.sqrt 2 → BY = 9 * Real.sqrt 2

theorem solution : problem 2 2 8 (9 * Real.sqrt 2) :=
by
  -- Contextualize the given conditions and directly link to the desired proof.
  intro h
  sorry

end solution_l105_105770


namespace inscribed_circle_circumference_l105_105325

theorem inscribed_circle_circumference (side_length : ℝ) (h : side_length = 10) : 
  ∃ C : ℝ, C = 2 * Real.pi * (side_length / 2) ∧ C = 10 * Real.pi := 
by 
  sorry

end inscribed_circle_circumference_l105_105325


namespace cost_difference_l105_105726

def TMobile_cost (num_lines : ℕ) : ℝ :=
  let base_cost := 50
  let additional_line_cost := 16
  let discount := 0.1
  let data_charge := 3
  let monthly_cost_before_discount := base_cost + (additional_line_cost * (num_lines - 2))
  let total_monthly_cost := monthly_cost_before_discount + (data_charge * num_lines)
  (total_monthly_cost * (1 - discount)) * 12

def MMobile_cost (num_lines : ℕ) : ℝ :=
  let base_cost := 45
  let additional_line_cost := 14
  let activation_fee := 20
  let monthly_cost := base_cost + (additional_line_cost * (num_lines - 2))
  (monthly_cost * 12) + (activation_fee * num_lines)

theorem cost_difference (num_lines : ℕ) (h : num_lines = 5) :
  TMobile_cost num_lines - MMobile_cost num_lines = 76.40 :=
  sorry

end cost_difference_l105_105726


namespace identity_n1_n2_product_l105_105396

theorem identity_n1_n2_product :
  (∃ (N1 N2 : ℤ),
    (∀ x : ℚ, (35 * x - 29) / (x^2 - 3 * x + 2) = N1 / (x - 1) + N2 / (x - 2)) ∧
    N1 * N2 = -246) :=
sorry

end identity_n1_n2_product_l105_105396


namespace half_of_number_l105_105616

theorem half_of_number (N : ℕ) (h : (4 / 15 * 5 / 7 * N) - (4 / 9 * 2 / 5 * N) = 24) : N / 2 = 945 :=
by
  sorry

end half_of_number_l105_105616


namespace symmetric_point_l105_105171

-- Define the given point M
def point_M : ℝ × ℝ × ℝ := (1, 0, -1)

-- Define the line in parametric form
def line (t : ℝ) : ℝ × ℝ × ℝ :=
  (3.5 + 2 * t, 1.5 + 2 * t, 0)

-- Define the symmetric point M'
def point_M' : ℝ × ℝ × ℝ := (2, -1, 1)

-- Statement: Prove that M' is the symmetric point to M with respect to the given line
theorem symmetric_point (M M' : ℝ × ℝ × ℝ) (line : ℝ → ℝ × ℝ × ℝ) :
  M = (1, 0, -1) →
  line (t) = (3.5 + 2 * t, 1.5 + 2 * t, 0) →
  M' = (2, -1, 1) :=
sorry

end symmetric_point_l105_105171


namespace arccos_cos_eq_x_div_3_solutions_l105_105310

theorem arccos_cos_eq_x_div_3_solutions (x : ℝ) :
  (Real.arccos (Real.cos x) = x / 3) ∧ (-3 * Real.pi / 2 ≤ x ∧ x ≤ 3 * Real.pi / 2) 
  ↔ x = -3 * Real.pi / 2 ∨ x = 0 ∨ x = 3 * Real.pi / 2 :=
by
  sorry

end arccos_cos_eq_x_div_3_solutions_l105_105310


namespace sum_of_three_integers_l105_105006

theorem sum_of_three_integers (a b c : ℕ) (h_pos: a > 0 ∧ b > 0 ∧ c > 0) 
  (h_diff: a ≠ b ∧ b ≠ c ∧ c ≠ a) (h_prod: a * b * c = 5^4) : a + b + c = 131 :=
sorry

end sum_of_three_integers_l105_105006


namespace poem_lines_added_l105_105783

theorem poem_lines_added (x : ℕ) 
  (initial_lines : ℕ)
  (months : ℕ)
  (final_lines : ℕ)
  (h_init : initial_lines = 24)
  (h_months : months = 22)
  (h_final : final_lines = 90)
  (h_equation : initial_lines + months * x = final_lines) :
  x = 3 :=
by {
  -- Placeholder for the proof
  sorry
}

end poem_lines_added_l105_105783


namespace ratio_of_areas_l105_105750

theorem ratio_of_areas (C1 C2 : ℝ) (h1 : (60 : ℝ) / 360 * C1 = (48 : ℝ) / 360 * C2) : 
  (C1 / C2) ^ 2 = 16 / 25 := 
by
  sorry

end ratio_of_areas_l105_105750


namespace abs_neg_product_eq_product_l105_105841

variable (a b : ℝ)

theorem abs_neg_product_eq_product (h1 : a < 0) (h2 : 0 < b) : |-a * b| = a * b := by
  sorry

end abs_neg_product_eq_product_l105_105841


namespace two_circles_tangent_internally_l105_105380

-- Define radii and distance between centers
def R : ℝ := 7
def r : ℝ := 4
def distance_centers : ℝ := 3

-- Statement of the problem
theorem two_circles_tangent_internally :
  distance_centers = R - r → 
  -- Positional relationship: tangent internally
  (distance_centers = abs (R - r)) :=
sorry

end two_circles_tangent_internally_l105_105380


namespace product_of_g_xi_l105_105936

noncomputable def x1 : ℂ := sorry
noncomputable def x2 : ℂ := sorry
noncomputable def x3 : ℂ := sorry
noncomputable def x4 : ℂ := sorry
noncomputable def x5 : ℂ := sorry

def f (x : ℂ) : ℂ := x^5 + x^2 + 1
def g (x : ℂ) : ℂ := x^3 - 2

axiom roots_of_f (x : ℂ) : f x = 0 ↔ x = x1 ∨ x = x2 ∨ x = x3 ∨ x = x4 ∨ x = x5

theorem product_of_g_xi : (g x1) * (g x2) * (g x3) * (g x4) * (g x5) = -243 := sorry

end product_of_g_xi_l105_105936


namespace f_4_1981_eq_l105_105778

noncomputable def f : ℕ → ℕ → ℕ
| 0, y => y + 1
| (x+1), 0 => f x 1
| (x+1), (y+1) => f x (f (x+1) y)

theorem f_4_1981_eq : f 4 1981 = 2^1984 - 3 := 
by
  sorry

end f_4_1981_eq_l105_105778


namespace stuffed_animal_total_l105_105846

/-- McKenna has 34 stuffed animals. -/
def mckenna_stuffed_animals : ℕ := 34

/-- Kenley has twice as many stuffed animals as McKenna. -/
def kenley_stuffed_animals : ℕ := 2 * mckenna_stuffed_animals

/-- Tenly has 5 more stuffed animals than Kenley. -/
def tenly_stuffed_animals : ℕ := kenley_stuffed_animals + 5

/-- The total number of stuffed animals the three girls have. -/
def total_stuffed_animals : ℕ := mckenna_stuffed_animals + kenley_stuffed_animals + tenly_stuffed_animals

/-- Prove that the total number of stuffed animals is 175. -/
theorem stuffed_animal_total : total_stuffed_animals = 175 := by
  sorry

end stuffed_animal_total_l105_105846


namespace correct_divisor_l105_105295

theorem correct_divisor (X D : ℕ) (h1 : X / 72 = 24) (h2 : X / D = 48) : D = 36 :=
sorry

end correct_divisor_l105_105295


namespace m_minus_n_is_square_l105_105072

theorem m_minus_n_is_square (m n : ℕ) (h1 : 0 < m) (h2 : 0 < n) (h3 : 2001 * m ^ 2 + m = 2002 * n ^ 2 + n) : ∃ k : ℕ, m - n = k ^ 2 :=
sorry

end m_minus_n_is_square_l105_105072


namespace A_share_of_profit_l105_105586

section InvestmentProfit

variables (capitalA capitalB : ℕ) -- initial capitals
variables (withdrawA advanceB : ℕ) -- changes after 8 months
variables (profit : ℕ) -- total profit

def investment_months (initial : ℕ) (final : ℕ) (first_period : ℕ) (second_period : ℕ) : ℕ :=
  initial * first_period + final * second_period

def ratio (a b : ℕ) : ℚ := (a : ℚ) / (b : ℚ)

def A_share (total_profit : ℕ) (ratioA ratioB : ℚ) : ℚ :=
  (ratioA / (ratioA + ratioB)) * total_profit

theorem A_share_of_profit :
  let capitalA := 3000
  let capitalB := 4000
  let withdrawA := 1000
  let advanceB := 1000
  let profit := 756
  let A_investment_months := investment_months capitalA (capitalA - withdrawA) 8 4
  let B_investment_months := investment_months capitalB (capitalB + advanceB) 8 4
  let ratioA := ratio A_investment_months B_investment_months
  let ratioB := ratio B_investment_months A_investment_months
  A_share profit ratioA ratioB = 288 := sorry

end InvestmentProfit

end A_share_of_profit_l105_105586


namespace profit_sharing_l105_105323

theorem profit_sharing
  (A_investment B_investment C_investment total_profit : ℕ)
  (A_share : ℕ)
  (ratio_A ratio_B ratio_C : ℕ)
  (hA : A_investment = 6300)
  (hB : B_investment = 4200)
  (hC : C_investment = 10500)
  (hShare : A_share = 3810)
  (hRatio : ratio_A = 3 ∧ ratio_B = 2 ∧ ratio_C = 5)
  (hTotRatio : ratio_A + ratio_B + ratio_C = 10)
  (hShareCalc : A_share = (3/10) * total_profit) :
  total_profit = 12700 :=
sorry

end profit_sharing_l105_105323


namespace Ashis_height_more_than_Babji_height_l105_105811

-- Definitions based on conditions
variables {A B : ℝ}
-- Condition expressing the relationship between Ashis's and Babji's height
def Babji_height (A : ℝ) : ℝ := 0.80 * A

-- The proof problem to show the percentage increase
theorem Ashis_height_more_than_Babji_height :
  B = Babji_height A → (A - B) / B * 100 = 25 :=
sorry

end Ashis_height_more_than_Babji_height_l105_105811


namespace find_G16_l105_105369

variable (G : ℝ → ℝ)

def condition1 : Prop := G 8 = 28

def condition2 : Prop := ∀ x : ℝ, 
  (x^2 + 8*x + 16) ≠ 0 → 
  (G (4*x) / G (x + 4) = 16 - (64*x + 80) / (x^2 + 8*x + 16))

theorem find_G16 (h1 : condition1 G) (h2 : condition2 G) : G 16 = 120 :=
sorry

end find_G16_l105_105369


namespace min_distance_ellipse_line_l105_105758

theorem min_distance_ellipse_line :
  let ellipse (x y : ℝ) := (x ^ 2) / 16 + (y ^ 2) / 12 = 1
  let line (x y : ℝ) := x - 2 * y - 12 = 0
  ∃ (d : ℝ), d = 4 * Real.sqrt 5 / 5 ∧
             (∀ (x y : ℝ), ellipse x y → ∃ (d' : ℝ), line x y → d' ≥ d) :=
  sorry

end min_distance_ellipse_line_l105_105758


namespace part_one_solution_set_part_two_range_of_m_l105_105857

-- Part I
theorem part_one_solution_set (x : ℝ) : (|x + 1| + |x - 2| - 5 > 0) ↔ (x > 3 ∨ x < -2) :=
sorry

-- Part II
theorem part_two_range_of_m (m : ℝ) : (∀ x : ℝ, |x + 1| + |x - 2| - m ≥ 2) ↔ (m ≤ 1) :=
sorry

end part_one_solution_set_part_two_range_of_m_l105_105857


namespace downstream_speed_l105_105710

-- Define the given conditions as constants
def V_u : ℝ := 25 -- upstream speed in kmph
def V_m : ℝ := 40 -- speed of the man in still water in kmph

-- Define the speed of the stream
def V_s := V_m - V_u

-- Define the downstream speed
def V_d := V_m + V_s

-- Assertion we need to prove
theorem downstream_speed : V_d = 55 := by
  sorry

end downstream_speed_l105_105710


namespace complex_norm_example_l105_105228

theorem complex_norm_example : 
  abs (-3 - (9 / 4 : ℝ) * I) = 15 / 4 := 
by
  sorry

end complex_norm_example_l105_105228


namespace enclosed_area_of_curve_l105_105560

noncomputable def radius_of_arcs := 1

noncomputable def arc_length := (1 / 2) * Real.pi

noncomputable def side_length_of_octagon := 3

noncomputable def area_of_octagon (s : ℝ) := 
  2 * (1 + Real.sqrt 2) * s ^ 2

noncomputable def area_of_sectors (n : ℕ) (arc_radius : ℝ) (arc_theta : ℝ) := 
  n * (1 / 4) * Real.pi

theorem enclosed_area_of_curve : 
  area_of_octagon side_length_of_octagon + area_of_sectors 12 radius_of_arcs arc_length 
  = 54 + 54 * Real.sqrt 2 + 3 * Real.pi := 
by
  sorry

end enclosed_area_of_curve_l105_105560


namespace sum_of_possible_values_l105_105628

theorem sum_of_possible_values (M : ℝ) (h : M * (M + 4) = 12) : M + (if M = -6 then 2 else -6) = -4 :=
by
  sorry

end sum_of_possible_values_l105_105628


namespace total_drink_volume_l105_105451

variable (T : ℝ)

theorem total_drink_volume :
  (0.15 * T + 0.60 * T + 0.25 * T = 35) → T = 140 :=
by
  intros h
  have h1 : (0.25 * T) = 35 := by sorry
  have h2 : T = 140 := by sorry
  exact h2

end total_drink_volume_l105_105451


namespace original_water_amount_l105_105530

theorem original_water_amount (W : ℝ) 
    (evap_rate : ℝ := 0.03) 
    (days : ℕ := 22) 
    (evap_percent : ℝ := 0.055) 
    (total_evap : ℝ := evap_rate * days) 
    (evap_condition : evap_percent * W = total_evap) : W = 12 :=
by sorry

end original_water_amount_l105_105530


namespace find_integers_l105_105987

def isPerfectSquare (n : ℤ) : Prop := ∃ m : ℤ, m * m = n

theorem find_integers (x : ℤ) (h : isPerfectSquare (x^2 + 19 * x + 95)) : x = -14 ∨ x = -5 := by
  sorry

end find_integers_l105_105987


namespace proof_A_minus_2B_eq_11_l105_105482

theorem proof_A_minus_2B_eq_11 
  (a b : ℤ)
  (hA : ∀ a b, A = 3*b^2 - 2*a^2)
  (hB : ∀ a b, B = ab - 2*b^2 - a^2) 
  (ha : a = 2) 
  (hb : b = -1) : 
  (A - 2*B = 11) :=
by
  sorry

end proof_A_minus_2B_eq_11_l105_105482


namespace bean_seedlings_l105_105760

theorem bean_seedlings
  (beans_per_row : ℕ)
  (pumpkins : ℕ) (pumpkins_per_row : ℕ)
  (radishes : ℕ) (radishes_per_row : ℕ)
  (rows_per_bed : ℕ) (beds : ℕ)
  (H_beans_per_row : beans_per_row = 8)
  (H_pumpkins : pumpkins = 84)
  (H_pumpkins_per_row : pumpkins_per_row = 7)
  (H_radishes : radishes = 48)
  (H_radishes_per_row : radishes_per_row = 6)
  (H_rows_per_bed : rows_per_bed = 2)
  (H_beds : beds = 14) :
  (beans_per_row * ((beds * rows_per_bed) - (pumpkins / pumpkins_per_row) - (radishes / radishes_per_row)) = 64) :=
by
  sorry

end bean_seedlings_l105_105760


namespace total_pages_correct_l105_105764

def history_pages : ℕ := 160
def geography_pages : ℕ := history_pages + 70
def sum_history_geography_pages : ℕ := history_pages + geography_pages
def math_pages : ℕ := sum_history_geography_pages / 2
def science_pages : ℕ := 2 * history_pages
def total_pages : ℕ := history_pages + geography_pages + math_pages + science_pages

theorem total_pages_correct : total_pages = 905 := by
  -- The proof goes here.
  sorry

end total_pages_correct_l105_105764


namespace length_of_fence_l105_105034

theorem length_of_fence (side_length : ℕ) (h : side_length = 28) : 4 * side_length = 112 :=
by
  sorry

end length_of_fence_l105_105034


namespace minimum_value_of_expression_l105_105605

noncomputable def min_value_expr (x y z : ℝ) : ℝ := (x + 3 * y) * (y + 3 * z) * (2 * x * z + 1)

theorem minimum_value_of_expression (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x * y * z = 1) :
  min_value_expr x y z = 24 * Real.sqrt 2 :=
sorry

end minimum_value_of_expression_l105_105605


namespace math_problem_l105_105656

/-- Given a function definition f(x) = 2 * x * f''(1) + x^2,
    Prove that the second derivative f''(0) is equal to -4. -/
theorem math_problem (f : ℝ → ℝ) (h1 : ∀ x, f x = 2 * x * (deriv^[2] (f) 1) + x^2) :
  (deriv^[2] f) 0 = -4 :=
  sorry

end math_problem_l105_105656


namespace number_of_possible_a_values_l105_105007

-- Define the function f(x)
def f (a x : ℝ) := abs (x + 1) + abs (a * x + 1)

-- Define the condition for the minimum value
def minimum_value_of_f (a : ℝ) := ∃ x : ℝ, f a x = (3 / 2)

-- The proof problem statement
theorem number_of_possible_a_values : 
  (∃ (a1 a2 a3 a4 : ℝ),
    minimum_value_of_f a1 ∧
    minimum_value_of_f a2 ∧
    minimum_value_of_f a3 ∧
    minimum_value_of_f a4 ∧
    a1 ≠ a2 ∧ a1 ≠ a3 ∧ a1 ≠ a4 ∧ a2 ≠ a3 ∧ a2 ≠ a4 ∧ a3 ≠ a4) :=
sorry

end number_of_possible_a_values_l105_105007


namespace raw_materials_amount_true_l105_105797

def machinery_cost : ℝ := 2000
def total_amount : ℝ := 5555.56
def cash (T : ℝ) : ℝ := 0.10 * T
def raw_materials_cost (T : ℝ) : ℝ := T - machinery_cost - cash T

theorem raw_materials_amount_true :
  raw_materials_cost total_amount = 3000 := 
  by
  sorry

end raw_materials_amount_true_l105_105797


namespace min_value_of_quadratic_l105_105720

theorem min_value_of_quadratic :
  ∃ x : ℝ, (∀ y : ℝ, 3 * x^2 - 18 * x + 2000 ≤ 3 * y^2 - 18 * y + 2000) ∧ (3 * x^2 - 18 * x + 2000 = 1973) :=
by
  sorry

end min_value_of_quadratic_l105_105720


namespace find_function_f_l105_105721

-- Define the problem in Lean 4
theorem find_function_f (f : ℝ → ℝ) : 
  (f 0 = 1) → 
  ((∀ x y : ℝ, f (x * y + 1) = f x * f y - f y - x + 2)) → 
  (∀ x : ℝ, f x = x + 1) :=
  by
    intros h₁ h₂
    sorry

end find_function_f_l105_105721


namespace range_of_a_l105_105929

theorem range_of_a (a : ℝ) : (¬ ∃ x : ℝ, x^2 + (a - 1) * x + 1 ≤ 0) → -1 < a ∧ a < 3 :=
sorry

end range_of_a_l105_105929


namespace azalea_paid_shearer_l105_105724

noncomputable def amount_paid_to_shearer (number_of_sheep wool_per_sheep price_per_pound profit : ℕ) : ℕ :=
  let total_wool := number_of_sheep * wool_per_sheep
  let total_revenue := total_wool * price_per_pound
  total_revenue - profit

theorem azalea_paid_shearer :
  let number_of_sheep := 200
  let wool_per_sheep := 10
  let price_per_pound := 20
  let profit := 38000
  amount_paid_to_shearer number_of_sheep wool_per_sheep price_per_pound profit = 2000 := 
by
  sorry

end azalea_paid_shearer_l105_105724


namespace barbed_wire_cost_l105_105536

theorem barbed_wire_cost
  (A : ℕ)          -- Area of the square field (sq m)
  (cost_per_meter : ℕ)  -- Cost per meter for the barbed wire (Rs)
  (gate_width : ℕ)      -- Width of each gate (m)
  (num_gates : ℕ)       -- Number of gates
  (side_length : ℕ)     -- Side length of the square field (m)
  (perimeter : ℕ)       -- Perimeter of the square field (m)
  (total_length : ℕ)    -- Total length of the barbed wire needed (m)
  (total_cost : ℕ)      -- Total cost of drawing the barbed wire (Rs)
  (h1 : A = 3136)       -- Given: Area = 3136 sq m
  (h2 : cost_per_meter = 1)  -- Given: Cost per meter = 1 Rs/m
  (h3 : gate_width = 1)      -- Given: Width of each gate = 1 m
  (h4 : num_gates = 2)       -- Given: Number of gates = 2
  (h5 : side_length * side_length = A)  -- Side length calculated from the area
  (h6 : perimeter = 4 * side_length)    -- Perimeter of the square field
  (h7 : total_length = perimeter - (num_gates * gate_width))  -- Actual barbed wire length after gates
  (h8 : total_cost = total_length * cost_per_meter)           -- Total cost calculation
  : total_cost = 222 :=      -- The result we need to prove
sorry

end barbed_wire_cost_l105_105536


namespace number_of_sides_l105_105476

-- Define the conditions
def interior_angle (n : ℕ) : ℝ := 156

-- The main theorem to prove the number of sides
theorem number_of_sides (n : ℕ) (h : interior_angle n = 156) : n = 15 :=
by
  sorry

end number_of_sides_l105_105476


namespace percentage_failed_in_Hindi_l105_105517

-- Define the percentage of students failed in English
def percentage_failed_in_English : ℝ := 56

-- Define the percentage of students failed in both Hindi and English
def percentage_failed_in_both : ℝ := 12

-- Define the percentage of students passed in both subjects
def percentage_passed_in_both : ℝ := 24

-- Define the total percentage of students
def percentage_total : ℝ := 100

-- Define what we need to prove
theorem percentage_failed_in_Hindi:
  ∃ (H : ℝ), H + percentage_failed_in_English - percentage_failed_in_both + percentage_passed_in_both = percentage_total ∧ H = 32 :=
  by 
    sorry

end percentage_failed_in_Hindi_l105_105517


namespace eccentricity_range_l105_105371

def hyperbola (a b x y : ℝ) : Prop := (a > 0) ∧ (b > 0) ∧ (x^2 / a^2 - y^2 / b^2 = 1)
def right_branch_hyperbola_P (a b c x y : ℝ) : Prop := hyperbola a b x y ∧ (c = a) ∧ (2 * c = a)

theorem eccentricity_range {a b c : ℝ} (h: hyperbola a b c c) (h1 : 2 * a = 2 * c) (h2 : c = a) :
  1 < (c / a) ∧ (c / a) ≤ (Real.sqrt 10 / 2 : ℝ) := by
  sorry

end eccentricity_range_l105_105371


namespace total_tape_area_l105_105566

theorem total_tape_area 
  (long_side_1 short_side_1 : ℕ) (boxes_1 : ℕ)
  (long_side_2 short_side_2 : ℕ) (boxes_2 : ℕ)
  (long_side_3 short_side_3 : ℕ) (boxes_3 : ℕ)
  (overlap : ℕ) (tape_width : ℕ) :
  long_side_1 = 30 → short_side_1 = 15 → boxes_1 = 5 →
  long_side_2 = 40 → short_side_2 = 40 → boxes_2 = 2 →
  long_side_3 = 50 → short_side_3 = 20 → boxes_3 = 3 →
  overlap = 2 → tape_width = 2 →
  let total_length_1 := boxes_1 * (long_side_1 + overlap + 2 * (short_side_1 + overlap))
  let total_length_2 := boxes_2 * 3 * (long_side_2 + overlap)
  let total_length_3 := boxes_3 * (long_side_3 + overlap + 2 * (short_side_3 + overlap))
  let total_length := total_length_1 + total_length_2 + total_length_3
  let total_area := total_length * tape_width
  total_area = 1740 :=
  by
  -- Add the proof steps here
  -- sorry can be used to skip the proof
  sorry

end total_tape_area_l105_105566


namespace sin_120_eq_sqrt3_div_2_l105_105568

theorem sin_120_eq_sqrt3_div_2 : Real.sin (120 * Real.pi / 180) = Real.sqrt 3 / 2 :=
by
  sorry

end sin_120_eq_sqrt3_div_2_l105_105568


namespace initial_concentration_alcohol_l105_105833

theorem initial_concentration_alcohol (x : ℝ) 
    (h1 : 0 ≤ x ∧ x ≤ 100)
    (h2 : 0.44 * 10 = (x / 100) * 2 + 3.6) :
    x = 40 :=
sorry

end initial_concentration_alcohol_l105_105833


namespace oliver_workout_hours_l105_105344

variable (x : ℕ)

theorem oliver_workout_hours :
  (x + (x - 2) + 2 * x + 2 * (x - 2) = 18) → x = 4 :=
by
  sorry

end oliver_workout_hours_l105_105344


namespace cindy_gives_3_envelopes_per_friend_l105_105519

theorem cindy_gives_3_envelopes_per_friend
  (initial_envelopes : ℕ) 
  (remaining_envelopes : ℕ)
  (friends : ℕ)
  (envelopes_per_friend : ℕ) 
  (h1 : initial_envelopes = 37) 
  (h2 : remaining_envelopes = 22)
  (h3 : friends = 5) 
  (h4 : initial_envelopes - remaining_envelopes = envelopes_per_friend * friends) :
  envelopes_per_friend = 3 :=
by
  sorry

end cindy_gives_3_envelopes_per_friend_l105_105519


namespace arcsin_one_half_l105_105714

theorem arcsin_one_half : Real.arcsin (1 / 2) = Real.pi / 6 :=
by
  sorry

end arcsin_one_half_l105_105714


namespace expand_product_l105_105596

theorem expand_product (x : ℝ) :
  (3 * x + 4) * (2 * x - 5) = 6 * x^2 - 7 * x - 20 :=
sorry

end expand_product_l105_105596


namespace rate_of_mixed_oil_l105_105245

theorem rate_of_mixed_oil (V1 V2 : ℝ) (P1 P2 : ℝ) : 
  (V1 = 10) → 
  (P1 = 50) → 
  (V2 = 5) → 
  (P2 = 67) → 
  ((V1 * P1 + V2 * P2) / (V1 + V2) = 55.67) :=
by
  intros V1_eq P1_eq V2_eq P2_eq
  rw [V1_eq, P1_eq, V2_eq, P2_eq]
  norm_num
  sorry

end rate_of_mixed_oil_l105_105245


namespace triangle_dimensions_l105_105467

-- Define the problem in Lean 4
theorem triangle_dimensions (a m : ℕ) (h₁ : a = m + 4)
  (h₂ : (a + 12) * (m + 12) = 10 * a * m) : 
  a = 12 ∧ m = 8 := 
by
  sorry

end triangle_dimensions_l105_105467


namespace volunteer_hours_per_year_l105_105130

def volunteers_per_month : ℕ := 2
def hours_per_session : ℕ := 3
def months_per_year : ℕ := 12

theorem volunteer_hours_per_year :
  volunteers_per_month * months_per_year * hours_per_session = 72 :=
by
  -- Proof is omitted
  sorry

end volunteer_hours_per_year_l105_105130


namespace minibus_seat_count_l105_105964

theorem minibus_seat_count 
  (total_children : ℕ) 
  (seats_with_3_children : ℕ) 
  (children_per_3_child_seat : ℕ) 
  (remaining_children : ℕ) 
  (children_per_2_child_seat : ℕ) 
  (total_seats : ℕ) :
  total_children = 19 →
  seats_with_3_children = 5 →
  children_per_3_child_seat = 3 →
  remaining_children = total_children - (seats_with_3_children * children_per_3_child_seat) →
  children_per_2_child_seat = 2 →
  total_seats = seats_with_3_children + (remaining_children / children_per_2_child_seat) →
  total_seats = 7 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end minibus_seat_count_l105_105964


namespace multiple_of_shirt_cost_l105_105117

theorem multiple_of_shirt_cost (S C M : ℕ) (h1 : S = 97) (h2 : C = 300 - S)
  (h3 : C = M * S + 9) : M = 2 :=
by
  -- The proof will be filled in here
  sorry

end multiple_of_shirt_cost_l105_105117


namespace part1_part2_l105_105523

noncomputable def f (x : ℝ) : ℝ := |2 * x - 1| + |2 * x - 3|

theorem part1 (x : ℝ) (hx : f x ≤ 5) : x ∈ Set.Icc (-1/4 : ℝ) (9/4 : ℝ) := sorry

noncomputable def h (x a : ℝ) : ℝ := Real.log (f x + a)

theorem part2 (ha : ∀ x : ℝ, f x + a > 0) : a ∈ Set.Ioi (-2 : ℝ) := sorry

end part1_part2_l105_105523


namespace greatest_roses_for_680_l105_105593

/--
Greatest number of roses that can be purchased for $680
given the following costs:
- $4.50 per individual rose
- $36 per dozen roses
- $50 per two dozen roses
--/
theorem greatest_roses_for_680 (cost_individual : ℝ) 
  (cost_dozen : ℝ) 
  (cost_two_dozen : ℝ) 
  (budget : ℝ) 
  (dozen : ℕ) 
  (two_dozen : ℕ) 
  (total_budget : ℝ) 
  (individual_cost : ℝ) 
  (dozen_cost : ℝ) 
  (two_dozen_cost : ℝ) 
  (roses_dozen : ℕ) 
  (roses_two_dozen : ℕ):
  individual_cost = 4.50 → dozen_cost = 36 → two_dozen_cost = 50 →
  budget = 680 → dozen = 12 → two_dozen = 24 →
  (∀ n : ℕ, n * two_dozen_cost ≤ budget → n * two_dozen + (budget - n * two_dozen_cost) / individual_cost ≤ total_budget) →
  total_budget = 318 := 
by
  sorry

end greatest_roses_for_680_l105_105593


namespace relationship_points_l105_105585

noncomputable def is_on_inverse_proportion (m x y : ℝ) : Prop :=
  y = (-m^2 - 2) / x

theorem relationship_points (a b c m : ℝ) :
  is_on_inverse_proportion m a (-1) ∧
  is_on_inverse_proportion m b 2 ∧
  is_on_inverse_proportion m c 3 →
  a > c ∧ c > b :=
by
  sorry

end relationship_points_l105_105585


namespace cyclist_club_member_count_l105_105227

-- Define the set of valid digits.
def valid_digits : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 9}

-- Define the problem statement
theorem cyclist_club_member_count : valid_digits.card ^ 3 = 512 :=
by
  -- Placeholder for the proof
  sorry

end cyclist_club_member_count_l105_105227


namespace simplified_expr_l105_105876

theorem simplified_expr : 
  (Real.sqrt 3 * Real.sqrt 12 - 2 * Real.sqrt 6 / Real.sqrt 3 + Real.sqrt 32 + (Real.sqrt 2) ^ 2) = (8 + 2 * Real.sqrt 2) := 
by 
  sorry

end simplified_expr_l105_105876


namespace range_of_a_l105_105115

theorem range_of_a (a : ℝ) (h1 : a ≤ 1)
(h2 : ∃ n₁ n₂ n₃ : ℤ, a ≤ n₁ ∧ n₁ < n₂ ∧ n₂ < n₃ ∧ n₃ ≤ 2 - a
  ∧ (∀ x : ℤ, a ≤ x ∧ x ≤ 2 - a → x = n₁ ∨ x = n₂ ∨ x = n₃)) :
  -1 < a ∧ a ≤ 0 :=
by
  sorry

end range_of_a_l105_105115


namespace calculate_area_of_region_l105_105944

theorem calculate_area_of_region :
  let region := {p : ℝ × ℝ | p.1^2 + p.2^2 + 2 * p.1 - 4 * p.2 = 12}
  ∃ area, area = 17 * Real.pi
:= by
  sorry

end calculate_area_of_region_l105_105944


namespace packs_sold_by_Robyn_l105_105436

theorem packs_sold_by_Robyn (total_packs : ℕ) (lucy_packs : ℕ) (robyn_packs : ℕ) 
  (h1 : total_packs = 98) (h2 : lucy_packs = 43) (h3 : robyn_packs = total_packs - lucy_packs) :
  robyn_packs = 55 :=
by
  rw [h1, h2] at h3
  exact h3

end packs_sold_by_Robyn_l105_105436


namespace minimum_value_of_expression_l105_105702

theorem minimum_value_of_expression (a b : ℝ) (h : 1 / a + 2 / b = 1) : 4 * a^2 + b^2 ≥ 32 :=
by sorry

end minimum_value_of_expression_l105_105702


namespace numberOfChromiumAtoms_l105_105478

noncomputable def molecularWeightOfCompound : ℕ := 296
noncomputable def atomicWeightOfPotassium : ℝ := 39.1
noncomputable def atomicWeightOfOxygen : ℝ := 16.0
noncomputable def atomicWeightOfChromium : ℝ := 52.0

def numberOfPotassiumAtoms : ℕ := 2
def numberOfOxygenAtoms : ℕ := 7

theorem numberOfChromiumAtoms
    (mw : ℕ := molecularWeightOfCompound)
    (awK : ℝ := atomicWeightOfPotassium)
    (awO : ℝ := atomicWeightOfOxygen)
    (awCr : ℝ := atomicWeightOfChromium)
    (numK : ℕ := numberOfPotassiumAtoms)
    (numO : ℕ := numberOfOxygenAtoms) :
  numK * awK + numO * awO + (mw - (numK * awK + numO * awO)) / awCr = 2 := 
by
  sorry

end numberOfChromiumAtoms_l105_105478


namespace pure_imaginary_complex_solution_l105_105666

theorem pure_imaginary_complex_solution (a : Real) :
  (a ^ 2 - 1 = 0) ∧ ((a - 1) ≠ 0) → a = -1 := by
  sorry

end pure_imaginary_complex_solution_l105_105666


namespace ratio_of_b_l105_105321

theorem ratio_of_b (a b k a1 a2 b1 b2 : ℝ) (h_nonzero_a2 : a2 ≠ 0) (h_nonzero_b12: b1 ≠ 0 ∧ b2 ≠ 0) :
  (a * b = k) →
  (a1 * b1 = a2 * b2) →
  (a1 / a2 = 3 / 5) →
  (b1 / b2 = 5 / 3) := 
sorry

end ratio_of_b_l105_105321


namespace calculate_A_plus_B_l105_105026

theorem calculate_A_plus_B (A B : ℝ) (h1 : A ≠ B) 
  (h2 : ∀ x : ℝ, (A * (B * x^2 + A * x + 1)^2 + B * (B * x^2 + A * x + 1) + 1) 
                - (B * (A * x^2 + B * x + 1)^2 + A * (A * x^2 + B * x + 1) + 1) 
                = x^4 + 5 * x^3 + x^2 - 4 * x) : A + B = 0 :=
by
  sorry

end calculate_A_plus_B_l105_105026


namespace inequality_proof_l105_105306

theorem inequality_proof (a : ℝ) : 
  2 * a^4 + 2 * a^2 - 1 ≥ (3 / 2) * (a^2 + a - 1) :=
by
  sorry

end inequality_proof_l105_105306


namespace prob_A_is_15_16_prob_B_is_3_4_prob_C_is_5_9_prob_exactly_two_good_ratings_is_77_576_l105_105105

-- Define the probability of success for student A, B, and C on a single jump
def p_A1 := 3 / 4
def p_B1 := 1 / 2
def p_C1 := 1 / 3

-- Calculate the total probability of excellence for A, B, and C
def P_A := p_A1 + (1 - p_A1) * p_A1
def P_B := p_B1 + (1 - p_B1) * p_B1
def P_C := p_C1 + (1 - p_C1) * p_C1

-- Statement to prove probabilities
theorem prob_A_is_15_16 : P_A = 15 / 16 := sorry
theorem prob_B_is_3_4 : P_B = 3 / 4 := sorry
theorem prob_C_is_5_9 : P_C = 5 / 9 := sorry

-- Definition for P(Good_Ratings) - exactly two students get a good rating
def P_Good_Ratings := 
  P_A * (1 - P_B) * (1 - P_C) + 
  (1 - P_A) * P_B * (1 - P_C) + 
  (1 - P_A) * (1 - P_B) * P_C

-- Statement to prove the given condition about good ratings
theorem prob_exactly_two_good_ratings_is_77_576 : P_Good_Ratings = 77 / 576 := sorry

end prob_A_is_15_16_prob_B_is_3_4_prob_C_is_5_9_prob_exactly_two_good_ratings_is_77_576_l105_105105


namespace wine_cost_is_3_60_l105_105812

noncomputable def appetizer_cost : ℕ := 8
noncomputable def steak_cost : ℕ := 20
noncomputable def dessert_cost : ℕ := 6
noncomputable def total_spent : ℝ := 38
noncomputable def tip_percentage : ℝ := 0.20
noncomputable def number_of_wines : ℕ := 2

noncomputable def discounted_steak_cost : ℝ := steak_cost / 2
noncomputable def full_meal_cost : ℝ := appetizer_cost + steak_cost + dessert_cost
noncomputable def meal_cost_after_discount : ℝ := appetizer_cost + discounted_steak_cost + dessert_cost
noncomputable def full_meal_tip := tip_percentage * full_meal_cost
noncomputable def meal_cost_with_tip := meal_cost_after_discount + full_meal_tip
noncomputable def total_wine_cost := total_spent - meal_cost_with_tip
noncomputable def cost_per_wine := total_wine_cost / number_of_wines

theorem wine_cost_is_3_60 : cost_per_wine = 3.60 := by
  sorry

end wine_cost_is_3_60_l105_105812


namespace jed_gives_2_cards_every_two_weeks_l105_105270

theorem jed_gives_2_cards_every_two_weeks
  (starting_cards : ℕ)
  (cards_per_week : ℕ)
  (cards_after_4_weeks : ℕ)
  (number_of_two_week_intervals : ℕ)
  (cards_given_away_each_two_weeks : ℕ):
  starting_cards = 20 →
  cards_per_week = 6 →
  cards_after_4_weeks = 40 →
  number_of_two_week_intervals = 2 →
  (starting_cards + 4 * cards_per_week - number_of_two_week_intervals * cards_given_away_each_two_weeks = cards_after_4_weeks) →
  cards_given_away_each_two_weeks = 2 := 
by
  intros h_start h_week h_4weeks h_intervals h_eq
  sorry

end jed_gives_2_cards_every_two_weeks_l105_105270


namespace proof_sum_of_ab_l105_105688

theorem proof_sum_of_ab :
  ∃ (a b : ℕ), a ≤ b ∧ 0 < a ∧ 0 < b ∧ a ^ 2 + b ^ 2 + 8 * a * b = 2010 ∧ a + b = 42 :=
sorry

end proof_sum_of_ab_l105_105688


namespace find_missing_number_l105_105662

theorem find_missing_number
  (a b c d e : ℝ) (mean : ℝ) (f : ℝ)
  (h1 : a = 13) 
  (h2 : b = 8)
  (h3 : c = 13)
  (h4 : d = 7)
  (h5 : e = 23)
  (hmean : mean = 14.2) :
  (a + b + c + d + e + f) / 6 = mean → f = 21.2 :=
by
  sorry

end find_missing_number_l105_105662


namespace sum_of_bases_l105_105608

theorem sum_of_bases (R1 R2 : ℕ)
  (h1 : ∀ F1 : ℚ, F1 = (4 * R1 + 8) / (R1 ^ 2 - 1) → F1 = (5 * R2 + 9) / (R2 ^ 2 - 1))
  (h2 : ∀ F2 : ℚ, F2 = (8 * R1 + 4) / (R1 ^ 2 - 1) → F2 = (9 * R2 + 5) / (R2 ^ 2 - 1)) :
  R1 + R2 = 24 :=
sorry

end sum_of_bases_l105_105608


namespace gallery_pieces_total_l105_105365

noncomputable def TotalArtGalleryPieces (A : ℕ) : Prop :=
  let D := (1 : ℚ) / 3 * A
  let N := A - D
  let notDisplayedSculptures := (2 : ℚ) / 3 * N
  let totalSculpturesNotDisplayed := 800
  (4 : ℚ) / 9 * A = 800

theorem gallery_pieces_total (A : ℕ) (h : (TotalArtGalleryPieces A)) : A = 1800 :=
by sorry

end gallery_pieces_total_l105_105365


namespace mary_spent_total_amount_l105_105632

def cost_of_berries := 11.08
def cost_of_apples := 14.33
def cost_of_peaches := 9.31
def total_cost := 34.72

theorem mary_spent_total_amount :
  cost_of_berries + cost_of_apples + cost_of_peaches = total_cost :=
by
  sorry

end mary_spent_total_amount_l105_105632


namespace bert_initial_amount_l105_105254

theorem bert_initial_amount (n : ℝ) (h : (1 / 2) * (3 / 4 * n - 9) = 12) : n = 44 :=
sorry

end bert_initial_amount_l105_105254


namespace inequality_preserves_neg_half_l105_105102

variable (a b : ℝ)

theorem inequality_preserves_neg_half (h : a ≤ b) : -a / 2 ≥ -b / 2 := by
  sorry

end inequality_preserves_neg_half_l105_105102


namespace max_winners_at_least_three_matches_l105_105119

theorem max_winners_at_least_three_matches (n : ℕ) (h : n = 200) :
  (∃ k : ℕ, k ≤ n ∧ ∀ m : ℕ, ((m ≥ 3) → ∃ x : ℕ, x = k → k = 66)) := 
sorry

end max_winners_at_least_three_matches_l105_105119


namespace parallel_lines_l105_105932

-- Definitions for the equations of the lines
def l1 (a : ℝ) (x y : ℝ) := (a - 1) * x + 2 * y + 10 = 0
def l2 (a : ℝ) (x y : ℝ) := x + a * y + 3 = 0

-- Theorem stating the conditions under which the lines l1 and l2 are parallel
theorem parallel_lines (a : ℝ) : 
  (∀ x y : ℝ, l1 a x y) = (∀ x y : ℝ, l2 a x y) → a = -1 ∨ a = 2 :=
by sorry

end parallel_lines_l105_105932


namespace arina_should_accept_anton_offer_l105_105894

noncomputable def total_shares : ℕ := 300000
noncomputable def arina_shares : ℕ := 90001
noncomputable def need_to_be_largest : ℕ := 104999 
noncomputable def shares_needed : ℕ := 14999
noncomputable def largest_shareholder_total : ℕ := 105000

noncomputable def maxim_shares : ℕ := 104999
noncomputable def inga_shares : ℕ := 30000
noncomputable def yuri_shares : ℕ := 30000
noncomputable def yulia_shares : ℕ := 30000
noncomputable def anton_shares : ℕ := 15000

noncomputable def maxim_price_per_share : ℕ := 11
noncomputable def inga_price_per_share : ℕ := 1250 / 100
noncomputable def yuri_price_per_share : ℕ := 1150 / 100
noncomputable def yulia_price_per_share : ℕ := 1300 / 100
noncomputable def anton_price_per_share : ℕ := 14

noncomputable def anton_total_cost : ℕ := anton_shares * anton_price_per_share
noncomputable def yuri_total_cost : ℕ := yuri_shares * yuri_price_per_share
noncomputable def inga_total_cost : ℕ := inga_shares * inga_price_per_share
noncomputable def yulia_total_cost : ℕ := yulia_shares * yulia_price_per_share

theorem arina_should_accept_anton_offer :
  anton_total_cost = 210000 := by
  sorry

end arina_should_accept_anton_offer_l105_105894


namespace train_crossing_time_l105_105640

-- Defining a structure for our problem context
structure TrainCrossing where
  length : Real -- length of the train in meters
  speed_kmh : Real -- speed of the train in km/h
  conversion_factor : Real -- conversion factor from km/h to m/s

-- Given the conditions in the problem
def trainData : TrainCrossing :=
  ⟨ 280, 50.4, 0.27778 ⟩

-- The main theorem statement:
theorem train_crossing_time (data : TrainCrossing) : 
  data.length / (data.speed_kmh * data.conversion_factor) = 20 := 
by
  sorry

end train_crossing_time_l105_105640


namespace smallest_number_of_small_bottles_l105_105489

def minimum_bottles_needed (large_bottle_capacity : ℕ) (small_bottle1 : ℕ) (small_bottle2 : ℕ) : ℕ :=
  if large_bottle_capacity = 720 ∧ small_bottle1 = 40 ∧ small_bottle2 = 45 then 16 else 0

theorem smallest_number_of_small_bottles :
  minimum_bottles_needed 720 40 45 = 16 := by
  sorry

end smallest_number_of_small_bottles_l105_105489


namespace find_fayes_age_l105_105471

variable {C D E F : ℕ}

theorem find_fayes_age
  (h1 : D = E - 2)
  (h2 : E = C + 5)
  (h3 : F = C + 4)
  (h4 : D = 15) :
  F = 16 := by
  sorry

end find_fayes_age_l105_105471


namespace cube_side_ratio_l105_105658

theorem cube_side_ratio (a b : ℝ) (h : (6 * a^2) / (6 * b^2) = 36) : a / b = 6 :=
by
  sorry

end cube_side_ratio_l105_105658


namespace intersection_points_count_l105_105885

-- Define the absolute value functions
def f1 (x : ℝ) : ℝ := |3 * x + 6|
def f2 (x : ℝ) : ℝ := -|4 * x - 4|

-- Prove the number of intersection points is 2
theorem intersection_points_count : 
  (∃ x1 y1, (f1 x1 = y1) ∧ (f2 x1 = y1)) ∧ 
  (∃ x2 y2, (f1 x2 = y2) ∧ (f2 x2 = y2) ∧ x1 ≠ x2) :=
sorry

end intersection_points_count_l105_105885


namespace number_of_ways_to_select_team_l105_105513

def calc_binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem number_of_ways_to_select_team : calc_binomial_coefficient 17 4 = 2380 := by
  sorry

end number_of_ways_to_select_team_l105_105513


namespace range_of_a_minus_b_l105_105949

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x^2 + a * x + b

theorem range_of_a_minus_b (a b : ℝ) (h1 : ∃ α β : ℝ, α ≠ β ∧ f α a b = 0 ∧ f β a b = 0)
  (h2 : ∃ x1 x2 x3 x4 : ℝ, x1 < x2 ∧ x2 < x3 ∧ x3 < x4 ∧
                         (x2 - x1 = x3 - x2) ∧ (x3 - x2 = x4 - x3) ∧
                         f (x1^2 + 2 * x1 - 1) a b = 0 ∧
                         f (x2^2 + 2 * x2 - 1) a b = 0 ∧
                         f (x3^2 + 2 * x3 - 1) a b = 0 ∧
                         f (x4^2 + 2 * x4 - 1) a b = 0) :
  a - b ≤ 25 / 9 :=
sorry

end range_of_a_minus_b_l105_105949


namespace minimum_value_of_n_l105_105532

open Int

theorem minimum_value_of_n (n d : ℕ) (h1 : n > 0) (h2 : d > 0) (h3 : d % n = 0)
    (h4 : 10 * n - 20 = 90) : n = 11 :=
by
  sorry

end minimum_value_of_n_l105_105532


namespace total_cans_collected_l105_105186

variable (bags_saturday : ℕ) (bags_sunday : ℕ) (cans_per_bag : ℕ)

def total_bags : ℕ := bags_saturday + bags_sunday

theorem total_cans_collected 
  (h_sat : bags_saturday = 5)
  (h_sun : bags_sunday = 3)
  (h_cans : cans_per_bag = 5) : 
  total_bags bags_saturday bags_sunday * cans_per_bag = 40 :=
by
  sorry

end total_cans_collected_l105_105186


namespace gcd_108_45_l105_105043

theorem gcd_108_45 :
  ∃ g, g = Nat.gcd 108 45 ∧ g = 9 :=
by
  sorry

end gcd_108_45_l105_105043


namespace total_weight_correct_weight_difference_correct_l105_105405

variables (baskets_of_apples baskets_of_pears : ℕ) (kg_per_basket_of_apples kg_per_basket_of_pears : ℕ)

def total_weight_apples_ppears (baskets_of_apples baskets_of_pears kg_per_basket_of_apples kg_per_basket_of_pears : ℕ) : ℕ :=
  (baskets_of_apples * kg_per_basket_of_apples) + (baskets_of_pears * kg_per_basket_of_pears)

def weight_difference_pears_apples (baskets_of_apples baskets_of_pears kg_per_basket_of_apples kg_per_basket_of_pears : ℕ) : ℕ :=
  (baskets_of_pears * kg_per_basket_of_pears) - (baskets_of_apples * kg_per_basket_of_apples)

theorem total_weight_correct (h_apples: baskets_of_apples = 120) (h_pears: baskets_of_pears = 130) (h_kg_apples: kg_per_basket_of_apples = 40) (h_kg_pears: kg_per_basket_of_pears = 50) : 
  total_weight_apples_ppears baskets_of_apples baskets_of_pears kg_per_basket_of_apples kg_per_basket_of_pears = 11300 :=
by
  rw [h_apples, h_pears, h_kg_apples, h_kg_pears]
  sorry

theorem weight_difference_correct (h_apples: baskets_of_apples = 120) (h_pears: baskets_of_pears = 130) (h_kg_apples: kg_per_basket_of_apples = 40) (h_kg_pears: kg_per_basket_of_pears = 50) : 
  weight_difference_pears_apples baskets_of_apples baskets_of_pears kg_per_basket_of_apples kg_per_basket_of_pears = 1700 :=
by
  rw [h_apples, h_pears, h_kg_apples, h_kg_pears]
  sorry

end total_weight_correct_weight_difference_correct_l105_105405


namespace buratino_solved_16_problems_l105_105223

-- Defining the conditions given in the problem
def total_kopeks_received : ℕ := 655 * 100 + 35

def geometric_sum (n : ℕ) : ℕ := 2^n - 1

-- The goal is to prove that Buratino solved 16 problems
theorem buratino_solved_16_problems (n : ℕ) (h : geometric_sum n = total_kopeks_received) : n = 16 := by
  sorry

end buratino_solved_16_problems_l105_105223


namespace evaluate_operation_l105_105777

def operation (x : ℝ) : ℝ := 9 - x

theorem evaluate_operation : operation (operation 15) = 15 :=
by
  -- Proof would go here
  sorry

end evaluate_operation_l105_105777


namespace distinct_four_digit_odd_numbers_l105_105033

-- Define the conditions as Lean definitions
def is_odd_digit (d : ℕ) : Prop :=
  d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

def valid_first_digit (d : ℕ) : Prop :=
  d = 1 ∨ d = 3 ∨ d = 7 ∨ d = 9

-- The proposition we want to prove
theorem distinct_four_digit_odd_numbers (n : ℕ) :
  (∀ d, d ∈ [n / 1000 % 10, n / 100 % 10, n / 10 % 10, n % 10] → is_odd_digit d) →
  valid_first_digit (n / 1000 % 10) →
  1000 ≤ n ∧ n < 10000 →
  n = 500 :=
sorry

end distinct_four_digit_odd_numbers_l105_105033


namespace smallest_possible_value_l105_105506

theorem smallest_possible_value (n : ℕ) (h1 : ∀ m, (Nat.lcm 60 m / Nat.gcd 60 m = 24) → m = n) (h2 : ∀ m, (m % 5 = 0) → m = n) : n = 160 :=
sorry

end smallest_possible_value_l105_105506


namespace inequality_true_for_all_real_l105_105535

theorem inequality_true_for_all_real (a : ℝ) : 
  3 * (1 + a^2 + a^4) ≥ (1 + a + a^2)^2 :=
sorry

end inequality_true_for_all_real_l105_105535


namespace contrapositive_equivalence_l105_105554
-- Importing the necessary libraries

-- Declaring the variables P and Q as propositions
variables (P Q : Prop)

-- The statement that we need to prove
theorem contrapositive_equivalence :
  (P → ¬ Q) ↔ (Q → ¬ P) :=
sorry

end contrapositive_equivalence_l105_105554


namespace inverse_110_mod_667_l105_105759

theorem inverse_110_mod_667 :
  (∃ (a b c : ℕ), a = 65 ∧ b = 156 ∧ c = 169 ∧ c^2 = a^2 + b^2) →
  (∃ n : ℕ, 110 * n % 667 = 1 ∧ 0 ≤ n ∧ n < 667 ∧ n = 608) :=
by
  sorry

end inverse_110_mod_667_l105_105759


namespace smallest_n_divisible_by_31997_l105_105097

noncomputable def smallest_n_divisible_by_prime : Nat :=
  let p := 31997
  let k := p
  2 * k

theorem smallest_n_divisible_by_31997 :
  smallest_n_divisible_by_prime = 63994 :=
by
  unfold smallest_n_divisible_by_prime
  rfl

end smallest_n_divisible_by_31997_l105_105097


namespace planes_perpendicular_l105_105800

variables {m n : Type} -- lines
variables {α β : Type} -- planes

axiom lines_different : m ≠ n
axiom planes_different : α ≠ β
axiom parallel_lines : ∀ (m n : Type), Prop -- m ∥ n
axiom parallel_plane_line : ∀ (m α : Type), Prop -- m ∥ α
axiom perp_plane_line : ∀ (n β : Type), Prop -- n ⊥ β
axiom perp_planes : ∀ (α β : Type), Prop -- α ⊥ β

theorem planes_perpendicular 
  (h1 : parallel_lines m n) 
  (h2 : parallel_plane_line m α) 
  (h3 : perp_plane_line n β) 
: perp_planes α β := 
sorry

end planes_perpendicular_l105_105800


namespace least_positive_integer_special_property_l105_105575

theorem least_positive_integer_special_property : ∃ (N : ℕ) (a b c : ℕ), 
  N = 100 * a + 10 * b + c ∧ a ≠ 0 ∧ 10 * b + c = N / 29 ∧ N = 725 :=
by
  sorry

end least_positive_integer_special_property_l105_105575


namespace graph_of_equation_is_two_lines_l105_105316

-- define the condition
def equation_condition (x y : ℝ) : Prop :=
  (x - y) ^ 2 = x ^ 2 + y ^ 2

-- state the theorem
theorem graph_of_equation_is_two_lines :
  ∀ x y : ℝ, equation_condition x y → (x = 0) ∨ (y = 0) :=
by
  intros x y h
  -- proof here
  sorry

end graph_of_equation_is_two_lines_l105_105316


namespace find_salary_June_l105_105933

variable (J F M A May_s June_s : ℝ)
variable (h1 : J + F + M + A = 4 * 8000)
variable (h2 : F + M + A + May_s = 4 * 8450)
variable (h3 : May_s = 6500)
variable (h4 : M + A + May_s + June_s = 4 * 9000)
variable (h5 : June_s = 1.2 * May_s)

theorem find_salary_June : June_s = 7800 := by
  sorry

end find_salary_June_l105_105933


namespace prime_factorization_count_l105_105071

theorem prime_factorization_count :
  (∃ (S : Finset ℕ), S = {97, 101, 2, 13, 107, 109} ∧ S.card = 6) :=
by
  sorry

end prime_factorization_count_l105_105071


namespace simplify_and_evaluate_l105_105733

def my_expression (x : ℝ) := (x + 2) * (x - 2) + 3 * (1 - x)

theorem simplify_and_evaluate : 
  my_expression (Real.sqrt 2) = 1 - 3 * Real.sqrt 2 := by
    sorry

end simplify_and_evaluate_l105_105733


namespace complementary_angles_difference_l105_105406

def complementary_angles (θ1 θ2 : ℝ) : Prop :=
  θ1 + θ2 = 90

theorem complementary_angles_difference:
  ∀ (θ1 θ2 : ℝ), 
  (θ1 / θ2 = 4 / 5) → 
  complementary_angles θ1 θ2 → 
  abs (θ2 - θ1) = 10 :=
by
  sorry

end complementary_angles_difference_l105_105406


namespace tickets_problem_l105_105863

theorem tickets_problem (A C : ℝ) 
  (h1 : A + C = 200) 
  (h2 : 3 * A + 1.5 * C = 510) : C = 60 :=
by
  sorry

end tickets_problem_l105_105863


namespace largest_corner_sum_l105_105326

-- Definitions based on the given problem
def faces_labeled : List ℕ := [2, 3, 4, 5, 6, 7]
def opposite_faces : List (ℕ × ℕ) := [(2, 7), (3, 6), (4, 5)]

-- Condition that face 2 cannot be adjacent to face 4
def non_adjacent_faces : List (ℕ × ℕ) := [(2, 4)]

-- Function to check adjacency constraints
def adjacent_allowed (f1 f2 : ℕ) : Bool := 
  ¬ (f1, f2) ∈ non_adjacent_faces ∧ ¬ (f2, f1) ∈ non_adjacent_faces

-- Determine the largest sum of three numbers whose faces meet at a corner
theorem largest_corner_sum : ∃ (a b c : ℕ), a ∈ faces_labeled ∧ b ∈ faces_labeled ∧ c ∈ faces_labeled ∧ 
  (adjacent_allowed a b) ∧ (adjacent_allowed b c) ∧ (adjacent_allowed c a) ∧ 
  a + b + c = 18 := 
sorry

end largest_corner_sum_l105_105326


namespace greatest_integer_difference_l105_105275

theorem greatest_integer_difference (x y : ℤ) (h1 : 5 < x ∧ x < 8) (h2 : 8 < y ∧ y < 13)
  (h3 : x % 3 = 0) (h4 : y % 3 = 0) : y - x = 6 :=
sorry

end greatest_integer_difference_l105_105275


namespace find_time_l105_105832

theorem find_time (s z t : ℝ) (h : ∀ s, 0 ≤ s ∧ s ≤ 7 → z = s^2 + 2 * s) : 
  z = 35 ∧ z = t^2 + 2 * t + 20 → 0 ≤ t ∧ t ≤ 7 → t = 3 :=
by
  sorry

end find_time_l105_105832


namespace thirteen_members_divisible_by_13_l105_105307

theorem thirteen_members_divisible_by_13 (B : ℕ) (hB : B < 10) : 
  (∃ B, (2000 + B * 100 + 34) % 13 = 0) ↔ B = 6 :=
by
  sorry

end thirteen_members_divisible_by_13_l105_105307


namespace smallest_four_digit_integer_l105_105827

theorem smallest_four_digit_integer (n : ℕ) (h1 : n ≥ 1000 ∧ n < 10000) 
  (h2 : ∀ d ∈ [1, 5, 6], n % d = 0)
  (h3 : ∀ d1 d2, d1 ≠ d2 → d1 ∈ [1, 5, 6] → d2 ∈ [1, 5, 6] → d1 ≠ d2) :
  n = 1560 :=
by
  sorry

end smallest_four_digit_integer_l105_105827


namespace cookout_kids_2004_l105_105592

variable (kids2005 kids2004 kids2006 : ℕ)

theorem cookout_kids_2004 :
  (kids2006 = 20) →
  (2 * kids2005 = 3 * kids2006) →
  (2 * kids2004 = kids2005) →
  kids2004 = 60 :=
by
  intros h1 h2 h3
  sorry

end cookout_kids_2004_l105_105592


namespace sugar_percentage_is_7_5_l105_105032

theorem sugar_percentage_is_7_5 
  (V1 : ℕ := 340)
  (p_water : ℝ := 88/100)
  (p_kola : ℝ := 5/100)
  (p_sugar : ℝ := 7/100)
  (V_sugar_add : ℝ := 3.2)
  (V_water_add : ℝ := 10)
  (V_kola_add : ℝ := 6.8) : 
  (
    (23.8 + 3.2) / (340 + 3.2 + 10 + 6.8) * 100 = 7.5
  ) :=
  by
  sorry

end sugar_percentage_is_7_5_l105_105032


namespace valid_points_region_equivalence_l105_105091

def valid_point (x y : ℝ) : Prop :=
  |x - 1| + |x + 1| + |2 * y| ≤ 4

def region1 (x y : ℝ) : Prop :=
  x ≤ -1 ∧ y ≤ x + 2 ∧ y ≥ -x - 2

def region2 (x y : ℝ) : Prop :=
  -1 < x ∧ x ≤ 1 ∧ -1 ≤ y ∧ y ≤ 1

def region3 (x y : ℝ) : Prop :=
  1 < x ∧ y ≤ 2 - x ∧ y ≥ x - 2

def solution_region (x y : ℝ) : Prop :=
  region1 x y ∨ region2 x y ∨ region3 x y

theorem valid_points_region_equivalence : 
  ∀ x y : ℝ, valid_point x y ↔ solution_region x y :=
sorry

end valid_points_region_equivalence_l105_105091


namespace original_earnings_l105_105709

variable (x : ℝ) -- John's original weekly earnings

theorem original_earnings:
  (1.20 * x = 72) → 
  (x = 60) :=
by
  intro h
  sorry

end original_earnings_l105_105709


namespace minimize_expression_l105_105912

theorem minimize_expression (a b c d : ℝ) (h1 : 1 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ d) (h5 : d ≤ 5) :
  (a - 1)^2 + (b / a - 1)^2 + (c / b - 1)^2 + (d / c - 1)^2 + (5 / d - 1)^2 ≥ 5 * (5^(1/5) - 1)^2 :=
by
  sorry

end minimize_expression_l105_105912


namespace part_a_l105_105713

theorem part_a (x α : ℝ) (hα : 0 < α ∧ α < 1) (hx : x ≥ 0) : x^α - α * x ≤ 1 - α :=
sorry

end part_a_l105_105713


namespace bubble_gum_cost_l105_105397

theorem bubble_gum_cost (n_pieces : ℕ) (total_cost : ℕ) (cost_per_piece : ℕ) 
  (h1 : n_pieces = 136) (h2 : total_cost = 2448) : cost_per_piece = 18 :=
by
  sorry

end bubble_gum_cost_l105_105397


namespace estimated_prob_is_0_9_l105_105500

section GerminationProbability

-- Defining the experiment data
structure ExperimentData :=
  (totalSeeds : ℕ)
  (germinatedSeeds : ℕ)
  (germinationRate : ℝ)

def experiments : List ExperimentData := [
  ⟨100, 91, 0.91⟩, 
  ⟨400, 358, 0.895⟩, 
  ⟨800, 724, 0.905⟩,
  ⟨1400, 1264, 0.903⟩,
  ⟨3500, 3160, 0.903⟩,
  ⟨7000, 6400, 0.914⟩
]

-- Hypothesis based on the given problem's observation
def estimated_germination_probability (experiments : List ExperimentData) : ℝ :=
  /- Fictively calculating the stable germination rate here; however, logically we should use 
     some weighted average or similar statistical stability method. -/
  0.9  -- Rounded and concluded estimated value based on observation

theorem estimated_prob_is_0_9 :
  estimated_germination_probability experiments = 0.9 :=
  sorry

end GerminationProbability

end estimated_prob_is_0_9_l105_105500


namespace find_f_2023_l105_105717

def is_odd_function (g : ℝ → ℝ) := ∀ x, g x = -g (-x)

def condition1 (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, (f (x + 1)) - 2 = -((f (1 - x)) - 2)

def condition2 (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (1 - x) = f (3 + x)

theorem find_f_2023 (f : ℝ → ℝ) (h1 : ∀ x : ℝ, (f (x + 1)) - 2 = -((f (1 - x)) - 2)) 
  (h2 : ∀ x : ℝ, f (1 - x) = f (3 + x)) : 
  f 2023 = 2 :=
sorry

end find_f_2023_l105_105717


namespace floor_alpha_six_eq_three_l105_105170

noncomputable def floor_of_alpha_six (α : ℝ) (h : α^5 - α^3 + α - 2 = 0) : ℤ :=
  Int.floor (α^6)

theorem floor_alpha_six_eq_three (α : ℝ) (h : α^5 - α^3 + α - 2 = 0) : floor_of_alpha_six α h = 3 :=
sorry

end floor_alpha_six_eq_three_l105_105170


namespace min_value_xy_k_l105_105443

theorem min_value_xy_k (x y k : ℝ) : ∃ x y : ℝ, (xy - k)^2 + (x + y - 1)^2 = 1 := by
  sorry

end min_value_xy_k_l105_105443


namespace Alan_collected_48_shells_l105_105003

def Laurie_shells : ℕ := 36
def Ben_shells : ℕ := Laurie_shells / 3
def Alan_shells : ℕ := 4 * Ben_shells

theorem Alan_collected_48_shells :
  Alan_shells = 48 :=
by
  sorry

end Alan_collected_48_shells_l105_105003


namespace crossing_time_approx_11_16_seconds_l105_105391

noncomputable def length_train_1 : ℝ := 140 -- length of the first train in meters
noncomputable def length_train_2 : ℝ := 170 -- length of the second train in meters
noncomputable def speed_train_1_km_hr : ℝ := 60 -- speed of the first train in km/hr
noncomputable def speed_train_2_km_hr : ℝ := 40 -- speed of the second train in km/hr

noncomputable def speed_conversion_factor : ℝ := 5 / 18 -- conversion factor from km/hr to m/s

-- convert speeds from km/hr to m/s
noncomputable def speed_train_1_m_s : ℝ := speed_train_1_km_hr * speed_conversion_factor
noncomputable def speed_train_2_m_s : ℝ := speed_train_2_km_hr * speed_conversion_factor

-- calculate relative speed in m/s (since they are moving in opposite directions)
noncomputable def relative_speed_m_s : ℝ := speed_train_1_m_s + speed_train_2_m_s

-- total distance to be covered
noncomputable def total_distance : ℝ := length_train_1 + length_train_2

-- calculate the time to cross each other
noncomputable def crossing_time : ℝ := total_distance / relative_speed_m_s

theorem crossing_time_approx_11_16_seconds : abs (crossing_time - 11.16) < 0.01 := by
    sorry

end crossing_time_approx_11_16_seconds_l105_105391


namespace valid_values_of_X_Y_l105_105631

-- Stating the conditions
def odd_combinations := 125
def even_combinations := 64
def revenue_diff (X Y : ℕ) := odd_combinations * X - even_combinations * Y = 5
def valid_limit (n : ℕ) := 0 < n ∧ n < 250

-- The theorem we want to prove
theorem valid_values_of_X_Y (X Y : ℕ) :
  revenue_diff X Y ∧ valid_limit X ∧ valid_limit Y ↔ (X = 41 ∧ Y = 80) ∨ (X = 105 ∧ Y = 205) :=
  sorry

end valid_values_of_X_Y_l105_105631


namespace greatest_prime_factor_5pow8_plus_10pow7_l105_105747

def greatest_prime_factor (n : ℕ) : ℕ := sorry

theorem greatest_prime_factor_5pow8_plus_10pow7 : greatest_prime_factor (5^8 + 10^7) = 19 := by
  sorry

end greatest_prime_factor_5pow8_plus_10pow7_l105_105747


namespace find_c_l105_105073

def f (x : ℤ) : ℤ := x - 2

def F (x y : ℤ) : ℤ := y^2 + x

theorem find_c : ∃ c, c = F 3 (f 16) ∧ c = 199 :=
by
  use F 3 (f 16)
  sorry

end find_c_l105_105073


namespace three_digit_integer_one_more_than_LCM_l105_105542

theorem three_digit_integer_one_more_than_LCM:
  ∃ (n : ℕ), (n > 99 ∧ n < 1000) ∧ (∃ (k : ℕ), n = k + 1 ∧ (∃ m, k = 3 * 4 * 5 * 7 * 2^m)) :=
  sorry

end three_digit_integer_one_more_than_LCM_l105_105542


namespace boat_speed_in_still_water_l105_105362

theorem boat_speed_in_still_water (b s : ℕ) (h1 : b + s = 21) (h2 : b - s = 9) : b = 15 := by
  sorry

end boat_speed_in_still_water_l105_105362


namespace find_real_numbers_l105_105410

theorem find_real_numbers (x : ℝ) :
  (x^3 - x^2 = (x^2 - x)^2) ↔ (x = 0 ∨ x = 1 ∨ x = 2) :=
by
  sorry

end find_real_numbers_l105_105410


namespace time_to_fill_partial_bucket_l105_105923

-- Definitions for the conditions
def time_to_fill_full_bucket : ℝ := 135
def r := 2 / 3

-- The time to fill 2/3 of the bucket should be proven as 90
theorem time_to_fill_partial_bucket : time_to_fill_full_bucket * r = 90 := 
by 
  -- Prove that 90 is the correct time to fill two-thirds of the bucket
  sorry

end time_to_fill_partial_bucket_l105_105923


namespace income_ratio_l105_105423

-- Define the conditions
variables (I_A I_B E_A E_B : ℝ)
variables (Savings_A Savings_B : ℝ)

-- Given conditions
def expenditure_ratio : E_A / E_B = 3 / 2 := sorry
def savings_A : Savings_A = 1600 := sorry
def savings_B : Savings_B = 1600 := sorry
def income_A : I_A = 4000 := sorry
def expenditure_A : E_A = I_A - Savings_A := sorry
def expenditure_B : E_B = I_B - Savings_B := sorry

-- Prove it's implied that the ratio of incomes is 5:4
theorem income_ratio : I_A / I_B = 5 / 4 :=
by
  sorry

end income_ratio_l105_105423


namespace part1_part2_1_part2_2_l105_105188

noncomputable def f (m : ℝ) (a x : ℝ) : ℝ :=
  m / x + Real.log (x / a)

-- Part (1)
theorem part1 (m a : ℝ) (h : m > 0) (ha : a > 0) (hmin : ∀ x, f m a x ≥ 2) : 
  m / a = Real.exp 1 :=
sorry

-- Part (2.1)
theorem part2_1 (a x₀ : ℝ) (ha : a > Real.exp 1) (hx₀ : x₀ > 1) (hzero : f 1 a x₀ = 0) : 
  1 / (2 * x₀) + x₀ < a - 1 :=
sorry

-- Part (2.2)
theorem part2_2 (a x₀ : ℝ) (ha : a > Real.exp 1) (hx₀ : x₀ > 1) (hzero : f 1 a x₀ = 0) : 
  x₀ + 1 / x₀ > 2 * Real.log a - Real.log (Real.log a) :=
sorry

end part1_part2_1_part2_2_l105_105188


namespace min_equilateral_triangles_l105_105136

theorem min_equilateral_triangles (s : ℝ) (S : ℝ) :
  s = 1 → S = 15 → 
  225 = (S / s) ^ 2 :=
by
  intros hs hS
  rw [hs, hS]
  simp
  sorry

end min_equilateral_triangles_l105_105136


namespace smallest_n_l105_105851

theorem smallest_n (n : ℕ) :
  (∀ m : ℤ, 0 < m ∧ m < 2001 →
    ∃ k : ℤ, (m : ℚ) / 2001 < (k : ℚ) / n ∧ (k : ℚ) / n < (m + 1 : ℚ) / 2002) ↔ n = 4003 :=
by
  sorry

end smallest_n_l105_105851


namespace maximum_xy_l105_105458

theorem maximum_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_parallel : 2 * x + y = 2) : 
  xy ≤ 1/2 := 
  sorry

end maximum_xy_l105_105458


namespace polynomial_divisibility_p_q_l105_105927

theorem polynomial_divisibility_p_q (p' q' : ℝ) :
  (∀ x, x^5 - x^4 + x^3 - p' * x^2 + q' * x - 6 = 0 → (x = -1 ∨ x = 2)) →
  p' = 0 ∧ q' = -9 :=
by sorry

end polynomial_divisibility_p_q_l105_105927


namespace current_speed_l105_105163

theorem current_speed (c : ℝ) :
  (∀ d1 t1 u v, d1 = 20 ∧ t1 = 2 ∧ u = 6 ∧ v = c → d1 = t1 * (u + v))
  ∧ (∀ d2 t2 u w, d2 = 4 ∧ t2 = 2 ∧ u = 6 ∧ w = c → d2 = t2 * (u - w)) 
  → c = 4 :=
by 
  intros
  sorry

end current_speed_l105_105163


namespace last_digit_of_power_of_two_l105_105375

theorem last_digit_of_power_of_two (n : ℕ) (h : n ≥ 2) : (2 ^ (2 ^ n) + 1) % 10 = 7 :=
sorry

end last_digit_of_power_of_two_l105_105375


namespace least_positive_multiple_of_primes_l105_105099

theorem least_positive_multiple_of_primes :
  11 * 13 * 17 * 19 = 46189 :=
by
  sorry

end least_positive_multiple_of_primes_l105_105099


namespace solve_cubic_inequality_l105_105837

theorem solve_cubic_inequality :
  { x : ℝ | x^3 + x^2 - 7 * x + 6 < 0 } = { x : ℝ | -3 < x ∧ x < 1 ∨ 1 < x ∧ x < 2 } :=
by
  sorry

end solve_cubic_inequality_l105_105837


namespace red_black_ball_ratio_l105_105723

theorem red_black_ball_ratio (R B x : ℕ) (h1 : 3 * R = B + x) (h2 : 2 * R + x = B) :
  R / B = 2 / 5 := by
  sorry

end red_black_ball_ratio_l105_105723


namespace repeating_decimal_fraction_l105_105689

theorem repeating_decimal_fraction : (0.363636363636 : ℚ) = 4 / 11 := 
sorry

end repeating_decimal_fraction_l105_105689


namespace find_a5_l105_105232

variable {a : ℕ → ℕ}
variable {S : ℕ → ℕ}

-- Condition: The sum of the first n terms of the sequence {a_n} is represented by S_n = 2a_n - 1 (n ∈ ℕ)
axiom sum_of_terms (n : ℕ) : S n = 2 * (a n) - 1

-- Prove that a_5 = 16
theorem find_a5 : a 5 = 16 :=
  sorry

end find_a5_l105_105232


namespace original_price_of_sarees_l105_105098

theorem original_price_of_sarees (P : ℝ) (h : 0.95 * 0.80 * P = 456) : P = 600 :=
by
  sorry

end original_price_of_sarees_l105_105098


namespace second_concert_attendance_correct_l105_105300

def first_concert_attendance : ℕ := 65899
def additional_people : ℕ := 119
def second_concert_attendance : ℕ := 66018

theorem second_concert_attendance_correct :
  first_concert_attendance + additional_people = second_concert_attendance :=
by sorry

end second_concert_attendance_correct_l105_105300


namespace selection_of_representatives_l105_105152

theorem selection_of_representatives 
  (females : ℕ) (males : ℕ)
  (h_females : females = 3) (h_males : males = 4) :
  (females ≥ 1 ∧ males ≥ 1) →
  (females * (males * (males - 1) / 2) + (females * (females - 1) / 2 * males) = 30) := 
by
  sorry

end selection_of_representatives_l105_105152


namespace a5_a6_value_l105_105311

def S (n : ℕ) : ℕ := n^3

theorem a5_a6_value : S 6 - S 4 = 152 :=
by
  sorry

end a5_a6_value_l105_105311


namespace distance_B_amusement_park_l105_105042

variable (d_A d_B v_A v_B t_A t_B : ℝ)

axiom h1 : v_A = 3
axiom h2 : v_B = 4
axiom h3 : d_B = d_A + 2
axiom h4 : t_A + t_B = 4
axiom h5 : t_A = d_A / v_A
axiom h6 : t_B = d_B / v_B

theorem distance_B_amusement_park:
  d_A / 3 + (d_A + 2) / 4 = 4 → d_B = 8 :=
by
  sorry

end distance_B_amusement_park_l105_105042


namespace least_multiple_72_112_199_is_310_l105_105149

theorem least_multiple_72_112_199_is_310 :
  ∃ k : ℕ, (112 ∣ k * 72) ∧ (199 ∣ k * 72) ∧ k = 310 := 
by
  sorry

end least_multiple_72_112_199_is_310_l105_105149


namespace polygon_triangle_existence_l105_105537

theorem polygon_triangle_existence (n : ℕ) (h₁ : n > 1)
  (h₂ : ∀ (k₁ k₂ : ℕ), k₁ ≠ k₂ → (4 ≤ k₁) → (4 ≤ k₂) → k₁ ≠ k₂) :
  ∃ k, k = 3 :=
by
  sorry

end polygon_triangle_existence_l105_105537


namespace symmetric_points_existence_l105_105242

-- Define the ellipse equation
def is_ellipse (x y : ℝ) : Prop :=
  (x^2 / 4) + (y^2 / 3) = 1

-- Define the line equation parameterized by m
def line_eq (x y m : ℝ) : Prop :=
  y = 4 * x + m

-- Define the range for m such that symmetric points exist
def m_in_range (m : ℝ) : Prop :=
  - (2 * Real.sqrt 13) / 13 < m ∧ m < (2 * Real.sqrt 13) / 13

-- Prove the existence of symmetric points criteria for m
theorem symmetric_points_existence (m : ℝ) :
  (∀ (x y : ℝ), is_ellipse x y → line_eq x y m → 
    (∃ (x1 y1 x2 y2 : ℝ), is_ellipse x1 y1 ∧ is_ellipse x2 y2 ∧ line_eq x1 y1 m ∧ line_eq x2 y2 m ∧ 
      (x1 = x2) ∧ (y1 = -y2))) ↔ m_in_range m :=
sorry

end symmetric_points_existence_l105_105242


namespace estimated_germination_probability_stable_l105_105798

structure ExperimentData where
  n : ℕ  -- number of grains per batch
  m : ℕ  -- number of germinations

def experimentalData : List ExperimentData := [
  ⟨50, 47⟩,
  ⟨100, 89⟩,
  ⟨200, 188⟩,
  ⟨500, 461⟩,
  ⟨1000, 892⟩,
  ⟨2000, 1826⟩,
  ⟨3000, 2733⟩
]

def germinationFrequency (data : ExperimentData) : ℚ :=
  data.m / data.n

def closeTo (x y : ℚ) (ε : ℚ) : Prop :=
  |x - y| < ε

theorem estimated_germination_probability_stable :
  ∃ ε > 0, ∀ data ∈ experimentalData, closeTo (germinationFrequency data) 0.91 ε :=
by
  sorry

end estimated_germination_probability_stable_l105_105798


namespace rocky_training_miles_l105_105465

variable (x : ℕ)

theorem rocky_training_miles (h1 : x + 2 * x + 6 * x = 36) : x = 4 :=
by
  -- proof
  sorry

end rocky_training_miles_l105_105465


namespace find_percentage_l105_105131

theorem find_percentage (P : ℝ) (h: (20 / 100) * 580 = (P / 100) * 120 + 80) : P = 30 := 
by
  sorry

end find_percentage_l105_105131


namespace athena_spent_correct_amount_l105_105184

-- Define the conditions
def num_sandwiches : ℕ := 3
def price_per_sandwich : ℝ := 3
def num_drinks : ℕ := 2
def price_per_drink : ℝ := 2.5

-- Define the total cost as per the given conditions
def total_cost : ℝ :=
  (num_sandwiches * price_per_sandwich) + (num_drinks * price_per_drink)

-- The theorem that states the problem and asserts the correct answer
theorem athena_spent_correct_amount : total_cost = 14 := 
  by
    sorry

end athena_spent_correct_amount_l105_105184


namespace men_in_second_group_l105_105284

theorem men_in_second_group (W : ℝ)
  (h1 : W = 18 * 20)
  (h2 : W = M * 30) :
  M = 12 :=
by
  sorry

end men_in_second_group_l105_105284


namespace calculate_speed_of_boat_in_still_water_l105_105088

noncomputable def speed_of_boat_in_still_water (V : ℝ) : Prop :=
    let downstream_speed := 16
    let upstream_speed := 9
    let first_half_current := 3 
    let second_half_current := 5
    let wind_speed := 2
    let effective_current_1 := first_half_current - wind_speed
    let effective_current_2 := second_half_current - wind_speed
    let V1 := downstream_speed - effective_current_1
    let V2 := upstream_speed + effective_current_2
    V = (V1 + V2) / 2

theorem calculate_speed_of_boat_in_still_water : 
    ∃ V : ℝ, speed_of_boat_in_still_water V ∧ V = 13.5 := 
sorry

end calculate_speed_of_boat_in_still_water_l105_105088


namespace tunnel_length_l105_105133

noncomputable def train_length : Real := 2 -- miles
noncomputable def time_to_exit_tunnel : Real := 4 -- minutes
noncomputable def train_speed : Real := 120 -- miles per hour

theorem tunnel_length : ∃ tunnel_length : Real, tunnel_length = 6 :=
  by
  -- We use the conditions given:
  let speed_in_miles_per_minute := train_speed / 60 -- converting speed from miles per hour to miles per minute
  let distance_travelled_by_front_in_4_min := speed_in_miles_per_minute * time_to_exit_tunnel
  let tunnel_length := distance_travelled_by_front_in_4_min - train_length
  have h : tunnel_length = 6 := by sorry
  exact ⟨tunnel_length, h⟩

end tunnel_length_l105_105133


namespace product_third_side_approximation_l105_105290

def triangle_third_side (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

noncomputable def product_of_third_side_lengths : ℝ :=
  Real.sqrt 41 * 3

theorem product_third_side_approximation (a b : ℝ) (h₁ : a = 4) (h₂ : b = 5) :
  ∃ (c₁ c₂ : ℝ), triangle_third_side a b c₁ ∧ triangle_third_side a b c₂ ∧
  abs ((c₁ * c₂) - 19.2) < 0.1 :=
sorry

end product_third_side_approximation_l105_105290


namespace mike_toys_l105_105920

theorem mike_toys (M A T : ℕ) 
  (h1 : A = 4 * M) 
  (h2 : T = A + 2)
  (h3 : M + A + T = 56) 
  : M = 6 := 
by 
  sorry

end mike_toys_l105_105920


namespace ab_cd_not_prime_l105_105154

theorem ab_cd_not_prime (a b c d : ℕ) (ha : a > b) (hb : b > c) (hc : c > d) (hd : d > 0)
  (h : a * c + b * d = (b + d + a - c) * (b + d - a + c)) : ¬ Nat.Prime (a * b + c * d) := 
sorry

end ab_cd_not_prime_l105_105154


namespace circle_eq_l105_105276

theorem circle_eq (x y : ℝ) (h k r : ℝ) (hc : h = 3) (kc : k = 1) (rc : r = 5) :
  (x - h)^2 + (y - k)^2 = r^2 ↔ (x - 3)^2 + (y - 1)^2 = 25 :=
by
  sorry

end circle_eq_l105_105276


namespace least_bulbs_needed_l105_105353

/-- Tulip bulbs come in packs of 15, and daffodil bulbs come in packs of 16.
  Rita wants to buy the same number of tulip and daffodil bulbs. 
  The goal is to prove that the least number of bulbs she needs to buy is 240, i.e.,
  the least common multiple of 15 and 16 is 240. -/
theorem least_bulbs_needed : Nat.lcm 15 16 = 240 := 
by
  sorry

end least_bulbs_needed_l105_105353


namespace batsman_average_after_17th_inning_l105_105610

theorem batsman_average_after_17th_inning :
  ∀ (A : ℕ), (16 * A + 50) / 17 = A + 2 → A = 16 → A + 2 = 18 := by
  intros A h1 h2
  rw [h2] at h1
  linarith

end batsman_average_after_17th_inning_l105_105610


namespace integer_solutions_l105_105404

theorem integer_solutions (x : ℝ) (n : ℤ)
  (h1 : ⌊x⌋ = n) :
  3 * x - 2 * n + 4 = 0 ↔
  x = -4 ∨ x = (-14:ℚ)/3 ∨ x = (-16:ℚ)/3 :=
by sorry

end integer_solutions_l105_105404


namespace find_s_l105_105437

noncomputable def s_value (m : ℝ) : ℝ := m + 16.25

theorem find_s (a b m s : ℝ)
  (h1 : a + b = m) (h2 : a * b = 4) :
  s = s_value m :=
by
  sorry

end find_s_l105_105437


namespace geometric_sequence_a3a5_l105_105272

theorem geometric_sequence_a3a5 :
  ∀ (a : ℕ → ℝ) (r : ℝ), (a 4 = 4) → (a 3 = a 0 * r ^ 3) → (a 5 = a 0 * r ^ 5) →
  a 3 * a 5 = 16 :=
by
  intros a r h1 h2 h3
  sorry

end geometric_sequence_a3a5_l105_105272


namespace walk_time_is_correct_l105_105806

noncomputable def time_to_walk_one_block := 
  let blocks := 18
  let bike_time_per_block := 20 -- seconds
  let additional_walk_time := 12 * 60 -- 12 minutes in seconds
  let walk_time := blocks * bike_time_per_block + additional_walk_time
  walk_time / blocks

theorem walk_time_is_correct : 
  let W := time_to_walk_one_block
  W = 60 := by
    sorry -- proof goes here

end walk_time_is_correct_l105_105806


namespace additional_money_spent_on_dvds_correct_l105_105087

def initial_money : ℕ := 320
def spent_on_books : ℕ := initial_money / 4 + 10
def remaining_after_books : ℕ := initial_money - spent_on_books
def spent_on_dvds_portion : ℕ := 2 * remaining_after_books / 5
def remaining_after_dvds : ℕ := 130
def total_spent_on_dvds : ℕ := remaining_after_books - remaining_after_dvds
def additional_spent_on_dvds : ℕ := total_spent_on_dvds - spent_on_dvds_portion

theorem additional_money_spent_on_dvds_correct : additional_spent_on_dvds = 8 :=
by
  sorry

end additional_money_spent_on_dvds_correct_l105_105087


namespace calculate_x_n_minus_inverse_x_n_l105_105881

theorem calculate_x_n_minus_inverse_x_n
  (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < π) (x : ℝ) (h : x - 1/x = 2 * Real.sin θ) (n : ℕ) (hn : 0 < n) :
  x^n - 1/x^n = 2 * Real.sinh (n * θ) :=
by sorry

end calculate_x_n_minus_inverse_x_n_l105_105881


namespace molecular_weight_CO_l105_105734

theorem molecular_weight_CO :
  let atomic_weight_C := 12.01
  let atomic_weight_O := 16.00
  let molecular_weight := atomic_weight_C + atomic_weight_O
  molecular_weight = 28.01 := 
by
  sorry

end molecular_weight_CO_l105_105734


namespace min_value_expression_l105_105167

theorem min_value_expression (x y : ℝ) (hx : |x| < 1) (hy : |y| < 2) (hxy : x * y = 1) : 
  ∃ k, k = 4 ∧ (∀ z, z = (1 / (1 - x^2) + 4 / (4 - y^2)) → z ≥ k) :=
sorry

end min_value_expression_l105_105167


namespace quadratic_solution_sum_l105_105539

theorem quadratic_solution_sum (m n p : ℕ) (h : m.gcd (n.gcd p) = 1)
  (h₀ : ∀ x, x * (5 * x - 11) = -6 ↔ x = (m + Real.sqrt n) / p ∨ x = (m - Real.sqrt n) / p) :
  m + n + p = 70 :=
sorry

end quadratic_solution_sum_l105_105539


namespace second_candidate_percentage_l105_105416

theorem second_candidate_percentage (V : ℝ) (h1 : 0.15 * V ≠ 0) (h2 : 0.38 * V ≠ 300) :
  (0.38 * V - 300) / (0.85 * V - 250) * 100 = 44.71 :=
by 
  -- Let the math proof be synthesized by a more detailed breakdown of conditions and theorems
  sorry

end second_candidate_percentage_l105_105416


namespace min_value_S_max_value_S_l105_105024

theorem min_value_S 
  (a b c d e : ℝ)
  (h₀ : a ≥ -1)
  (h₁ : b ≥ -1)
  (h₂ : c ≥ -1)
  (h₃ : d ≥ -1)
  (h₄ : e ≥ -1)
  (h_sum : a + b + c + d + e = 5) : 
  (a + b) * (b + c) * (c + d) * (d + e) * (e + a) ≥ -512 := 
sorry

theorem max_value_S 
  (a b c d e : ℝ)
  (h₀ : a ≥ -1)
  (h₁ : b ≥ -1)
  (h₂ : c ≥ -1)
  (h₃ : d ≥ -1)
  (h₄ : e ≥ -1)
  (h_sum : a + b + c + d + e = 5) : 
  (a + b) * (b + c) * (c + d) * (d + e) * (e + a) ≤ 288 := 
sorry

end min_value_S_max_value_S_l105_105024


namespace top_angle_isosceles_triangle_l105_105580

open Real

theorem top_angle_isosceles_triangle (A B C : ℝ) (abc_is_isosceles : (A = B ∨ B = C ∨ A = C))
  (angle_A : A = 40) : (B = 40 ∨ B = 100) :=
sorry

end top_angle_isosceles_triangle_l105_105580


namespace least_common_multiple_l105_105926

open Int

theorem least_common_multiple {a b c : ℕ} 
  (h1 : Nat.lcm a b = 18) 
  (h2 : Nat.lcm b c = 20) : Nat.lcm a c = 90 := 
sorry

end least_common_multiple_l105_105926


namespace negation_of_universal_prop_correct_l105_105039

def negation_of_universal_prop : Prop :=
  ¬ (∀ x : ℝ, x = |x|) ↔ ∃ x : ℝ, x ≠ |x|

theorem negation_of_universal_prop_correct : negation_of_universal_prop := 
by
  sorry

end negation_of_universal_prop_correct_l105_105039


namespace maximize_profits_l105_105769

variable (m : ℝ) (x : ℝ)

def w1 (m x : ℝ) := (8 - m) * x - 30
def w2 (x : ℝ) := -0.01 * x^2 + 8 * x - 80

theorem maximize_profits : 
  (4 ≤ m ∧ m < 5.1 → ∀ x, 0 ≤ x ∧ x ≤ 500 → w1 m x ≥ w2 x) ∧
  (m = 5.1 → ∀ x ≤ 300, w1 m 500 = w2 300) ∧
  (m > 5.1 ∧ m ≤ 6 → ∀ x, 0 ≤ x ∧ x ≤ 300 → w2 x ≥ w1 m x) :=
  sorry

end maximize_profits_l105_105769


namespace total_planks_needed_l105_105502

theorem total_planks_needed (large_planks small_planks : ℕ) (h1 : large_planks = 37) (h2 : small_planks = 42) : large_planks + small_planks = 79 :=
by
  sorry

end total_planks_needed_l105_105502


namespace no_int_solutions_x2_minus_3y2_eq_17_l105_105065

theorem no_int_solutions_x2_minus_3y2_eq_17 : 
  ∀ (x y : ℤ), (x^2 - 3 * y^2 ≠ 17) := 
by
  intros x y
  sorry

end no_int_solutions_x2_minus_3y2_eq_17_l105_105065


namespace malachi_additional_photos_l105_105508

-- Definition of the conditions
def total_photos : ℕ := 2430
def ratio_last_year : ℕ := 10
def ratio_this_year : ℕ := 17
def total_ratio_units : ℕ := ratio_last_year + ratio_this_year
def diff_ratio_units : ℕ := ratio_this_year - ratio_last_year
def photos_per_unit : ℕ := total_photos / total_ratio_units
def additional_photos : ℕ := diff_ratio_units * photos_per_unit

-- The theorem proving how many more photos Malachi took this year than last year
theorem malachi_additional_photos : additional_photos = 630 := by
  sorry

end malachi_additional_photos_l105_105508


namespace cats_eat_fish_l105_105607

theorem cats_eat_fish (c d: ℕ) (h1 : 1 < c) (h2 : c < 10) (h3 : c * d = 91) : c + d = 20 := by
  sorry

end cats_eat_fish_l105_105607


namespace number_of_non_congruent_triangles_with_perimeter_20_l105_105084

theorem number_of_non_congruent_triangles_with_perimeter_20 :
  ∃ T : Finset (Finset ℕ), 
    (∀ t ∈ T, ∃ a b c : ℕ, t = {a, b, c} ∧ a + b + c = 20 ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a) ∧
    T.card = 14 :=
by
  sorry

end number_of_non_congruent_triangles_with_perimeter_20_l105_105084


namespace simplify_and_evaluate_l105_105381

noncomputable section

def x := Real.sqrt 3 + 1

theorem simplify_and_evaluate :
  (x / (x^2 - 1) / (1 - (1 / (x + 1)))) = Real.sqrt 3 / 3 := by
  sorry

end simplify_and_evaluate_l105_105381


namespace emily_journey_length_l105_105579

theorem emily_journey_length
  (y : ℝ)
  (h1 : y / 5 + 30 + y / 3 + y / 6 = y) :
  y = 100 :=
by
  sorry

end emily_journey_length_l105_105579


namespace residue_11_pow_2016_mod_19_l105_105202

theorem residue_11_pow_2016_mod_19 : (11^2016) % 19 = 17 := 
sorry

end residue_11_pow_2016_mod_19_l105_105202


namespace probability_angie_carlos_two_seats_apart_l105_105165

theorem probability_angie_carlos_two_seats_apart :
  let people := ["Angie", "Bridget", "Carlos", "Diego", "Edwin"]
  let table_size := people.length
  let total_arrangements := (Nat.factorial (table_size - 1))
  let favorable_arrangements := 2 * (Nat.factorial (table_size - 2))
  total_arrangements > 0 ∧
  (favorable_arrangements / total_arrangements : ℚ) = 1 / 2 :=
by {
  sorry
}

end probability_angie_carlos_two_seats_apart_l105_105165


namespace P_iff_q_l105_105061

variables (a b c: ℝ)

def P : Prop := a * c < 0
def q : Prop := ∃ α β : ℝ, α * β < 0 ∧ a * α^2 + b * α + c = 0 ∧ a * β^2 + b * β + c = 0

theorem P_iff_q : P a c ↔ q a b c := 
sorry

end P_iff_q_l105_105061


namespace find_m_l105_105175

noncomputable def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem find_m (y b : ℝ) (m : ℕ) 
  (h5 : binomial m 4 * y^(m-4) * b^4 = 210) 
  (h6 : binomial m 5 * y^(m-5) * b^5 = 462) 
  (h7 : binomial m 6 * y^(m-6) * b^6 = 792) : 
  m = 7 := 
sorry

end find_m_l105_105175


namespace lattice_points_in_bounded_region_l105_105707

def isLatticePoint (p : ℤ × ℤ) : Prop :=
  true  -- All (n, m) ∈ ℤ × ℤ are lattice points

def boundedRegion (x y : ℤ) : Prop :=
  y = x ^ 2 ∨ y = 8 - x ^ 2
  
theorem lattice_points_in_bounded_region :
  ∃ S : Finset (ℤ × ℤ), 
    (∀ p ∈ S, isLatticePoint p ∧ boundedRegion p.1 p.2) ∧ S.card = 17 :=
by
  sorry

end lattice_points_in_bounded_region_l105_105707


namespace service_cost_is_correct_l105_105337

def service_cost_per_vehicle(cost_per_liter: ℝ)
                            (num_minivans: ℕ) 
                            (num_trucks: ℕ)
                            (total_cost: ℝ) 
                            (minivan_tank_liters: ℝ)
                            (truck_size_increase_pct: ℝ) 
                            (total_fuel: ℝ) 
                            (total_fuel_cost: ℝ) 
                            (total_service_cost: ℝ)
                            (num_vehicles: ℕ) 
                            (service_cost_per_vehicle: ℝ) : Prop :=
  cost_per_liter = 0.70 ∧
  num_minivans = 4 ∧
  num_trucks = 2 ∧
  total_cost = 395.4 ∧
  minivan_tank_liters = 65 ∧
  truck_size_increase_pct = 1.2 ∧
  total_fuel = (4 * minivan_tank_liters) + (2 * (minivan_tank_liters * (1 + truck_size_increase_pct))) ∧
  total_fuel_cost = total_fuel * cost_per_liter ∧
  total_service_cost = total_cost - total_fuel_cost ∧
  num_vehicles = num_minivans + num_trucks ∧
  service_cost_per_vehicle = total_service_cost / num_vehicles

-- Now, we state the theorem we want to prove.
theorem service_cost_is_correct :
  service_cost_per_vehicle 0.70 4 2 395.4 65 1.2 546 382.2 13.2 6 2.2 :=
by {
    sorry
}

end service_cost_is_correct_l105_105337


namespace minimum_packages_shipped_l105_105218

-- Definitions based on the conditions given in the problem
def Sarah_truck_capacity : ℕ := 18
def Ryan_truck_capacity : ℕ := 11

-- Minimum number of packages shipped
theorem minimum_packages_shipped :
  ∃ (n : ℕ), n = Sarah_truck_capacity * Ryan_truck_capacity :=
by sorry

end minimum_packages_shipped_l105_105218


namespace square_feet_per_acre_l105_105498

theorem square_feet_per_acre 
  (pay_per_acre_per_month : ℕ) 
  (total_pay_per_month : ℕ) 
  (length : ℕ) 
  (width : ℕ) 
  (total_acres : ℕ) 
  (H1 : pay_per_acre_per_month = 30) 
  (H2 : total_pay_per_month = 300) 
  (H3 : length = 360) 
  (H4 : width = 1210) 
  (H5 : total_acres = 10) : 
  (length * width) / total_acres = 43560 :=
by 
  sorry

end square_feet_per_acre_l105_105498


namespace max_value_t_min_value_y_l105_105529

-- 1. Prove that the maximum value of t given |2x+5| + |2x-1| - t ≥ 0 is s = 6.
theorem max_value_t (t : ℝ) (x : ℝ) :
  (abs (2*x + 5) + abs (2*x - 1) - t ≥ 0) → (t ≤ 6) :=
by sorry

-- 2. Given s = 6 and 4a + 5b = s, prove that the minimum value of y = 1/(a+2b) + 4/(3a+3b) is y = 3/2.
theorem min_value_y (a b : ℝ) (s : ℝ) :
  s = 6 → (4*a + 5*b = s) → (a > 0) → (b > 0) → 
  (1/(a + 2*b) + 4/(3*a + 3*b) ≥ 3/2) :=
by sorry

end max_value_t_min_value_y_l105_105529


namespace N_subseteq_M_l105_105686

/--
Let M = { x | ∃ n ∈ ℤ, x = n / 2 + 1 } and
N = { y | ∃ m ∈ ℤ, y = m + 0.5 }.
Prove that N is a subset of M.
-/
theorem N_subseteq_M : 
  let M := { x : ℝ | ∃ n : ℤ, x = n / 2 + 1 }
  let N := { y : ℝ | ∃ m : ℤ, y = m + 0.5 }
  N ⊆ M := sorry

end N_subseteq_M_l105_105686


namespace part_a_correct_part_b_correct_l105_105497

-- Define the alphabet and mapping
inductive Letter
| C | H | M | O
deriving DecidableEq, Inhabited

open Letter

def letter_to_base4 (ch : Letter) : ℕ :=
  match ch with
  | C => 0
  | H => 1
  | M => 2
  | O => 3

def word_to_base4 (word : List Letter) : ℕ :=
  word.foldl (fun acc ch => acc * 4 + letter_to_base4 ch) 0

def base4_to_letter (n : ℕ) : Letter :=
  match n with
  | 0 => C
  | 1 => H
  | 2 => M
  | 3 => O
  | _ => C -- This should not occur if input is in valid base-4 range

def base4_to_word (n : ℕ) (size : ℕ) : List Letter :=
  if size = 0 then []
  else
    let quotient := n / 4
    let remainder := n % 4
    base4_to_letter remainder :: base4_to_word quotient (size - 1)

-- The size of the words is fixed at 8
def word_size : ℕ := 8

noncomputable def part_a : List Letter :=
  base4_to_word 2017 word_size

theorem part_a_correct :
  part_a = [H, O, O, H, M, C] := by
  sorry

def given_word : List Letter :=
  [H, O, M, C, H, O, M, C]

noncomputable def part_b : ℕ :=
  word_to_base4 given_word + 1 -- Adjust for zero-based indexing

theorem part_b_correct :
  part_b = 29299 := by
  sorry

end part_a_correct_part_b_correct_l105_105497


namespace student_marks_l105_105480

theorem student_marks :
  let max_marks := 300
  let passing_percentage := 0.60
  let failed_by := 20
  let passing_marks := max_marks * passing_percentage
  let marks_obtained := passing_marks - failed_by
  marks_obtained = 160 := by
sorry

end student_marks_l105_105480


namespace all_round_trips_miss_capital_same_cost_l105_105253

open Set

variable {City : Type} [Inhabited City]
variable {f : City → City → ℝ}
variable (capital : City)
variable (round_trip_cost : List City → ℝ)

-- The conditions
axiom flight_cost_symmetric (A B : City) : f A B = f B A
axiom equal_round_trip_cost (R1 R2 : List City) :
  (∀ (city : City), city ∈ R1 ↔ city ∈ R2) → 
  round_trip_cost R1 = round_trip_cost R2

noncomputable def constant_trip_cost := 
  ∀ (cities1 cities2 : List City),
     (∀ (city : City), city ∈ cities1 ↔ city ∈ cities2) →
     ¬(capital ∈ cities1 ∨ capital ∈ cities2) →
     round_trip_cost cities1 = round_trip_cost cities2

-- Goal to prove
theorem all_round_trips_miss_capital_same_cost : constant_trip_cost capital round_trip_cost := 
  sorry

end all_round_trips_miss_capital_same_cost_l105_105253


namespace number_of_boxwoods_l105_105615

variables (x : ℕ)
def charge_per_trim := 5
def charge_per_shape := 15
def number_of_shaped_boxwoods := 4
def total_charge := 210
def total_shaping_charge := number_of_shaped_boxwoods * charge_per_shape

theorem number_of_boxwoods (h : charge_per_trim * x + total_shaping_charge = total_charge) : x = 30 :=
by
  sorry

end number_of_boxwoods_l105_105615


namespace find_parabola_eq_l105_105765

noncomputable def parabola_equation (a b c : ℝ) : Prop :=
  ∀ x y : ℝ, y = -3 * x ^ 2 + 18 * x - 22 ↔ (x - 3) ^ 2 + 5 = y

theorem find_parabola_eq :
  ∃ a b c : ℝ, (vertex = (3, 5) ∧ axis_of_symmetry ∧ point_on_parabola = (2, 2)) →
  parabola_equation a b c :=
sorry

end find_parabola_eq_l105_105765


namespace initially_calculated_average_is_correct_l105_105503

theorem initially_calculated_average_is_correct :
  let S := 220
  let incorrect_sum := S - 36 + 26
  let initially_avg := incorrect_sum / 10
  initially_avg = 22 :=
by
  let S := 220
  let incorrect_sum := S - 36 + 26
  let initially_avg := incorrect_sum / 10
  show initially_avg = 22
  sorry

end initially_calculated_average_is_correct_l105_105503


namespace initial_range_calculation_l105_105573

variable (initial_range telescope_range : ℝ)
variable (increased_by : ℝ)
variable (h_telescope : telescope_range = increased_by * initial_range)

theorem initial_range_calculation 
  (h_telescope_range : telescope_range = 150)
  (h_increased_by : increased_by = 3)
  (h_telescope : telescope_range = increased_by * initial_range) :
  initial_range = 50 :=
  sorry

end initial_range_calculation_l105_105573


namespace crates_of_oranges_l105_105960

theorem crates_of_oranges (C : ℕ) (h1 : ∀ crate, crate = 150) (h2 : ∀ box, box = 30) (num_boxes : ℕ) (total_fruits : ℕ) : 
  num_boxes = 16 → total_fruits = 2280 → 150 * C + 16 * 30 = 2280 → C = 12 :=
by
  intros num_boxes_eq total_fruits_eq fruit_eq
  sorry

end crates_of_oranges_l105_105960


namespace rectangle_area_l105_105870

theorem rectangle_area (p : ℝ) (l : ℝ) (h1 : 2 * (l + 2 * l) = p) :
  l * 2 * l = p^2 / 18 :=
by
  sorry

end rectangle_area_l105_105870


namespace problem1_problem2_l105_105871

noncomputable def A : Set ℝ := {x | x^2 - 3 * x + 2 = 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | x^2 - a * x + (a - 1) = 0}
noncomputable def C (m : ℝ) : Set ℝ := {x | x^2 - m * x + 2 = 0}

theorem problem1 (a : ℝ) : (∀ x, x ∈ B a → x ∈ A) → a = 2 ∨ a = 3 := sorry

theorem problem2 (m : ℝ) : (∀ x, x ∈ A → x ∈ C m) → m = 3 ∨ (-2 * Real.sqrt 2 < m ∧ m < 2 * Real.sqrt 2) := sorry

end problem1_problem2_l105_105871


namespace polynomial_coeff_sum_eq_four_l105_105422

theorem polynomial_coeff_sum_eq_four (a a1 a2 a3 a4 a5 a6 a7 a8 : ℤ) :
  (∀ x : ℤ, (2 * x - 1)^6 * (x + 1)^2 = a * x ^ 8 + a1 * x ^ 7 + a2 * x ^ 6 + a3 * x ^ 5 + 
                      a4 * x ^ 4 + a5 * x ^ 3 + a6 * x ^ 2 + a7 * x + a8) →
  a + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 = 4 := by
  sorry

end polynomial_coeff_sum_eq_four_l105_105422


namespace paco_salty_cookies_left_l105_105494

theorem paco_salty_cookies_left (initial_salty : ℕ) (eaten_salty : ℕ) : initial_salty = 26 ∧ eaten_salty = 9 → initial_salty - eaten_salty = 17 :=
by
  intro h
  cases h
  sorry


end paco_salty_cookies_left_l105_105494


namespace problem_1_problem_2_l105_105998

noncomputable def f (a x : ℝ) : ℝ := |x + a| + |x + 1/a|

theorem problem_1 (x : ℝ) : f 2 x > 3 ↔ x < -(11 / 4) ∨ x > 1 / 4 := sorry

theorem problem_2 (a m : ℝ) (ha : a > 0) : f a m + f a (-1 / m) ≥ 4 := sorry

end problem_1_problem_2_l105_105998


namespace loan_difference_eq_1896_l105_105813

/-- 
  Samantha borrows $12,000 with two repayment schemes:
  1. A twelve-year loan with an annual interest rate of 8% compounded semi-annually. 
     At the end of 6 years, she must make a payment equal to half of what she owes, 
     and the remaining balance accrues interest until the end of 12 years.
  2. A twelve-year loan with a simple annual interest rate of 10%, paid as a lump-sum at the end.

  Prove that the positive difference between the total amounts to be paid back 
  under the two schemes is $1,896, rounded to the nearest dollar.
-/
theorem loan_difference_eq_1896 :
  let P := 12000
  let r1 := 0.08
  let r2 := 0.10
  let n := 2
  let t := 12
  let t1 := 6
  let A1 := P * (1 + r1 / n) ^ (n * t1)
  let payment_after_6_years := A1 / 2
  let remaining_balance := A1 / 2
  let compounded_remaining := remaining_balance * (1 + r1 / n) ^ (n * t1)
  let total_compound := payment_after_6_years + compounded_remaining
  let total_simple := P * (1 + r2 * t)
  (total_simple - total_compound).round = 1896 := 
by
  sorry

end loan_difference_eq_1896_l105_105813


namespace whale_ninth_hour_consumption_l105_105392

-- Define the arithmetic sequence conditions
def first_hour_consumption : ℕ := 10
def common_difference : ℕ := 5

-- Define the total consumption over 12 hours
def total_consumption := 12 * (first_hour_consumption + (first_hour_consumption + 11 * common_difference)) / 2

-- Prove the ninth hour (which is the 8th term) consumption
theorem whale_ninth_hour_consumption :
  total_consumption = 450 →
  first_hour_consumption + 8 * common_difference = 50 := 
by
  intros h
  sorry
  

end whale_ninth_hour_consumption_l105_105392


namespace eggs_town_hall_l105_105874

-- Definitions of given conditions
def eggs_club_house : ℕ := 40
def eggs_park : ℕ := 25
def total_eggs_found : ℕ := 80

-- Problem statement
theorem eggs_town_hall : total_eggs_found - (eggs_club_house + eggs_park) = 15 := by
  sorry

end eggs_town_hall_l105_105874


namespace scientific_notation_320000_l105_105779

theorem scientific_notation_320000 : 320000 = 3.2 * 10^5 :=
  by sorry

end scientific_notation_320000_l105_105779


namespace worker_usual_time_l105_105654

theorem worker_usual_time (S T : ℝ) (D : ℝ) (h1 : D = S * T)
    (h2 : D = (3/4) * S * (T + 8)) : T = 24 :=
by
  sorry

end worker_usual_time_l105_105654


namespace sixty_percent_of_number_l105_105159

theorem sixty_percent_of_number (N : ℚ) (h : ((1 / 6) * (2 / 3) * (3 / 4) * (5 / 7) * N = 25)) :
  0.60 * N = 252 := sorry

end sixty_percent_of_number_l105_105159


namespace nearest_int_to_expr_l105_105453

theorem nearest_int_to_expr : 
  |(3 + Real.sqrt 2)^6 - 3707| < 1 :=
by 
  sorry

end nearest_int_to_expr_l105_105453


namespace gcd_72_168_l105_105201

theorem gcd_72_168 : Nat.gcd 72 168 = 24 :=
by
  sorry

end gcd_72_168_l105_105201


namespace solution_set_of_inequality_l105_105096

theorem solution_set_of_inequality: 
  {x : ℝ | (2 * x - 1) / x < 1} = { x : ℝ | 0 < x ∧ x < 1 } :=
by
  sorry

end solution_set_of_inequality_l105_105096


namespace man_fraction_ownership_l105_105556

theorem man_fraction_ownership :
  ∀ (F : ℚ), (3 / 5 * F = 15000) → (75000 = 75000) → (F / 75000 = 1 / 3) :=
by
  intros F h1 h2
  sorry

end man_fraction_ownership_l105_105556


namespace company_a_percentage_l105_105409

theorem company_a_percentage (total_profits: ℝ) (p_b: ℝ) (profit_b: ℝ) (profit_a: ℝ) :
  p_b = 0.40 →
  profit_b = 60000 →
  profit_a = 90000 →
  total_profits = profit_b / p_b →
  (profit_a / total_profits) * 100 = 60 :=
by
  intros h_pb h_profit_b h_profit_a h_total_profits
  sorry

end company_a_percentage_l105_105409


namespace correct_option_is_B_l105_105291

noncomputable def smallest_absolute_value := 0

theorem correct_option_is_B :
  (∀ x : ℝ, |x| ≥ 0) ∧ |(0 : ℝ)| = 0 :=
by
  sorry

end correct_option_is_B_l105_105291


namespace length_of_base_of_isosceles_triangle_l105_105623

noncomputable def length_congruent_sides : ℝ := 8
noncomputable def perimeter_triangle : ℝ := 26

theorem length_of_base_of_isosceles_triangle : 
  ∀ (b : ℝ), 
  2 * length_congruent_sides + b = perimeter_triangle → 
  b = 10 :=
by
  intros b h
  -- The proof is omitted.
  sorry

end length_of_base_of_isosceles_triangle_l105_105623


namespace sum_x_midpoints_of_triangle_l105_105194

theorem sum_x_midpoints_of_triangle (a b c : ℝ) (h : a + b + c = 15) :
  (a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 15 :=
by
  -- Proof omitted, replace with actual proof
  sorry

end sum_x_midpoints_of_triangle_l105_105194


namespace focus_of_parabola_l105_105438

def parabola_focus (a k : ℕ) : ℚ :=
  1 / (4 * a) + k

theorem focus_of_parabola :
  parabola_focus 9 6 = 217 / 36 :=
by
  sorry

end focus_of_parabola_l105_105438


namespace quadratic_inequality_solution_l105_105668

theorem quadratic_inequality_solution (m : ℝ) :
    (∃ x : ℝ, x^2 - m * x + 1 ≤ 0) ↔ m ≥ 2 ∨ m ≤ -2 := by
  sorry

end quadratic_inequality_solution_l105_105668


namespace find_a_of_binomial_square_l105_105527

theorem find_a_of_binomial_square (a : ℚ) :
  (∃ b : ℚ, (3 * (x : ℚ) + b)^2 = 9 * x^2 + 21 * x + a) ↔ a = 49 / 4 :=
by
  sorry

end find_a_of_binomial_square_l105_105527


namespace correct_average_marks_l105_105642

theorem correct_average_marks 
  (n : ℕ) (average initial_wrong new_correct : ℕ) 
  (h_num_students : n = 30)
  (h_average_marks : average = 100)
  (h_initial_wrong : initial_wrong = 70)
  (h_new_correct : new_correct = 10) :
  (average * n - (initial_wrong - new_correct)) / n = 98 := 
by
  sorry

end correct_average_marks_l105_105642


namespace find_total_cost_price_l105_105249

noncomputable def cost_prices (C1 C2 C3 : ℝ) : Prop :=
  0.85 * C1 + 72.50 = 1.125 * C1 ∧
  1.20 * C2 - 45.30 = 0.95 * C2 ∧
  0.92 * C3 + 33.60 = 1.10 * C3

theorem find_total_cost_price :
  ∃ (C1 C2 C3 : ℝ), cost_prices C1 C2 C3 ∧ C1 + C2 + C3 = 631.51 := 
by
  sorry

end find_total_cost_price_l105_105249


namespace oreo_shop_ways_l105_105601

theorem oreo_shop_ways (α β : ℕ) (products total_ways : ℕ) :
  let oreo_flavors := 6
  let milk_flavors := 4
  let total_flavors := oreo_flavors + milk_flavors
  (α + β = products) ∧ (products = 4) ∧ (total_ways = 2143) ∧ 
  (α ≤ 2 * total_flavors) ∧ (β ≤ 4 * oreo_flavors) →
  total_ways = 2143 :=
by sorry


end oreo_shop_ways_l105_105601


namespace red_blue_tile_difference_is_15_l105_105941

def num_blue_tiles : ℕ := 17
def num_red_tiles_initial : ℕ := 8
def additional_red_tiles : ℕ := 24
def num_red_tiles_new : ℕ := num_red_tiles_initial + additional_red_tiles
def tile_difference : ℕ := num_red_tiles_new - num_blue_tiles

theorem red_blue_tile_difference_is_15 : tile_difference = 15 :=
by
  sorry

end red_blue_tile_difference_is_15_l105_105941


namespace voldemort_spending_l105_105155

theorem voldemort_spending :
  let book_price_paid := 8
  let original_book_price := 64
  let journal_price := 2 * book_price_paid
  let total_spent := book_price_paid + journal_price
  (book_price_paid = (original_book_price / 8)) ∧ (total_spent = 24) :=
by
  let book_price_paid := 8
  let original_book_price := 64
  let journal_price := 2 * book_price_paid
  let total_spent := book_price_paid + journal_price
  have h1 : book_price_paid = (original_book_price / 8) := by
    sorry
  have h2 : total_spent = 24 := by
    sorry
  exact ⟨h1, h2⟩

end voldemort_spending_l105_105155


namespace discount_is_25_percent_l105_105613

noncomputable def discount_percentage (M : ℝ) (C : ℝ) (SP : ℝ) : ℝ :=
  ((M - SP) / M) * 100

theorem discount_is_25_percent (M : ℝ) (C : ℝ) (SP : ℝ) 
  (h1 : C = 0.64 * M) 
  (h2 : SP = C * 1.171875) : 
  discount_percentage M C SP = 25 := 
by 
  sorry

end discount_is_25_percent_l105_105613


namespace Ivan_returns_alive_Ivan_takes_princesses_l105_105980

theorem Ivan_returns_alive (Tsarevnas Koscheis: Finset ℕ) (five_girls: Finset ℕ) 
  (cond1: Tsarevnas.card = 3) (cond2: Koscheis.card = 2) (cond3: five_girls.card = 5)
  (cond4: Tsarevnas ∪ Koscheis = five_girls)
  (cond5: ∀ g ∈ five_girls, ∃ t ∈ Tsarevnas, t ≠ g ∧ ∃ k ∈ Koscheis, k ≠ g)
  (cond6: ∀ girl : ℕ, girl ∈ five_girls → 
          ∃ truth_count : ℕ, 
          (truth_count = (if girl ∈ Tsarevnas then 2 else 3))): 
  ∃ princesses : Finset ℕ, princesses.card = 3 ∧ princesses ⊆ Tsarevnas ∧ ∀ k ∈ Koscheis, k ∉ princesses :=
sorry

theorem Ivan_takes_princesses (Tsarevnas Koscheis: Finset ℕ) (five_girls: Finset ℕ) 
  (cond1: Tsarevnas.card = 3) (cond2: Koscheis.card = 2) (cond3: five_girls.card = 5)
  (cond4: Tsarevnas ∪ Koscheis = five_girls)
  (cond5: ∀ g ∈ five_girls, ∃ t ∈ Tsarevnas, t ≠ g ∧ ∃ k ∈ Koscheis, k ≠ g)
  (cond6 and cond7: ∀ girl1 girl2 girl3 : ℕ, girl1 ≠ girl2 → girl2 ≠ girl3 → girl1 ∈ Tsarevnas → girl2 ∈ Tsarevnas → girl3 ∈ Tsarevnas → 
          ∃ (eldest middle youngest : ℕ), 
              (eldest ∈ Tsarevnas ∧ middle ∈ Tsarevnas ∧ youngest ∈ Tsarevnas) 
          ∧
              (eldest ≠ middle ∧ eldest ≠ youngest ∧ middle ≠ youngest)
          ∧
              (∀ k ∈ Koscheis, k ≠ eldest ∧ k ≠ middle ∧ k ≠ youngest)
  ):
  ∃ princesses : Finset ℕ, 
          princesses.card = 3 ∧ princesses ⊆ Tsarevnas ∧ 
          (∃ eldest ,∃ middle,∃ youngest : ℕ, eldest ∈ princesses ∧ middle ∈ princesses ∧ youngest ∈ princesses ∧ 
                 eldest ≠ middle ∧ eldest ≠ youngest ∧ middle ≠ youngest)
:=
sorry

end Ivan_returns_alive_Ivan_takes_princesses_l105_105980


namespace adult_ticket_price_l105_105240

/-- 
The community center sells 85 tickets and collects $275 in total.
35 of those tickets are adult tickets. Each child's ticket costs $2.
We want to find the price of an adult ticket.
-/
theorem adult_ticket_price 
  (total_tickets : ℕ) 
  (total_revenue : ℚ) 
  (adult_tickets_sold : ℕ) 
  (child_ticket_price : ℚ)
  (h1 : total_tickets = 85)
  (h2 : total_revenue = 275) 
  (h3 : adult_tickets_sold = 35) 
  (h4 : child_ticket_price = 2) 
  : ∃ A : ℚ, (35 * A + 50 * 2 = 275) ∧ (A = 5) :=
by
  sorry

end adult_ticket_price_l105_105240


namespace area_one_magnet_is_150_l105_105594

noncomputable def area_one_magnet : ℕ :=
  let length := 15
  let total_circumference := 70
  let combined_width := (total_circumference / 2 - length) / 2
  let width := combined_width
  length * width

theorem area_one_magnet_is_150 :
  area_one_magnet = 150 :=
by
  -- This will skip the actual proof for now
  sorry

end area_one_magnet_is_150_l105_105594


namespace cards_left_l105_105246
noncomputable section

def initial_cards : ℕ := 676
def bought_cards : ℕ := 224

theorem cards_left : initial_cards - bought_cards = 452 := 
by
  sorry

end cards_left_l105_105246


namespace enlarged_poster_height_l105_105334

def original_poster_width : ℝ := 3
def original_poster_height : ℝ := 2
def new_poster_width : ℝ := 12

theorem enlarged_poster_height :
  new_poster_width / original_poster_width * original_poster_height = 8 := 
by
  sorry

end enlarged_poster_height_l105_105334


namespace orange_weight_l105_105665

variable (A O : ℕ)

theorem orange_weight (h1 : O = 5 * A) (h2 : A + O = 12) : O = 10 :=
  sorry

end orange_weight_l105_105665


namespace chris_score_l105_105112

variable (s g c : ℕ)

theorem chris_score  (h1 : s = g + 60) (h2 : (s + g) / 2 = 110) (h3 : c = 110 * 120 / 100) :
  c = 132 := by
  sorry

end chris_score_l105_105112


namespace correct_number_of_students_answered_both_l105_105030

def students_enrolled := 25
def answered_q1_correctly := 22
def answered_q2_correctly := 20
def not_taken_test := 3

def students_answered_both_questions_correctly : Nat :=
  let students_took_test := students_enrolled - not_taken_test
  let b := answered_q2_correctly
  b

theorem correct_number_of_students_answered_both :
  students_answered_both_questions_correctly = answered_q2_correctly :=
by {
  -- this space is for the proof, we are currently not required to provide it
  sorry
}

end correct_number_of_students_answered_both_l105_105030


namespace empty_set_subset_zero_set_l105_105853

-- Define the sets
def zero_set : Set ℕ := {0}
def empty_set : Set ℕ := ∅

-- State the problem
theorem empty_set_subset_zero_set : empty_set ⊂ zero_set :=
sorry

end empty_set_subset_zero_set_l105_105853


namespace complement_union_l105_105407

def U : Set ℤ := {x | -3 < x ∧ x ≤ 4}
def A : Set ℤ := {-2, -1, 3}
def B : Set ℤ := {1, 2, 3}

def C (U : Set ℤ) (S : Set ℤ) : Set ℤ := {x | x ∈ U ∧ x ∉ S}

theorem complement_union (A B : Set ℤ) (U : Set ℤ) :
  C U (A ∪ B) = {0, 4} :=
by
  sorry

end complement_union_l105_105407


namespace abc_value_l105_105090

theorem abc_value (a b c : ℂ) 
  (h1 : a * b + 5 * b = -20)
  (h2 : b * c + 5 * c = -20)
  (h3 : c * a + 5 * a = -20) : 
  a * b * c = -100 := 
by {
  sorry
}

end abc_value_l105_105090


namespace min_trucks_for_crates_l105_105910

noncomputable def min_trucks (total_weight : ℕ) (max_weight_per_crate : ℕ) (truck_capacity : ℕ) : ℕ :=
  if total_weight % truck_capacity = 0 then total_weight / truck_capacity
  else total_weight / truck_capacity + 1

theorem min_trucks_for_crates :
  ∀ (total_weight max_weight_per_crate truck_capacity : ℕ),
    total_weight = 10 →
    max_weight_per_crate = 1 →
    truck_capacity = 3 →
    min_trucks total_weight max_weight_per_crate truck_capacity = 5 :=
by
  intros total_weight max_weight_per_crate truck_capacity h_total h_max h_truck
  rw [h_total, h_max, h_truck]
  sorry

end min_trucks_for_crates_l105_105910


namespace find_k_l105_105445

theorem find_k (k : ℤ) : 
  (∀ (x1 y1 x2 y2 x3 y3 : ℤ),
    (x1, y1) = (2, 9) ∧ (x2, y2) = (5, 18) ∧ (x3, y3) = (8, 27) ∧ 
    ∃ m b : ℤ, y1 = m * x1 + b ∧ y2 = m * x2 + b ∧ y3 = m * x3 + b) 
  ∧ ∃ m b : ℤ, k = m * 42 + b
  → k = 129 :=
sorry

end find_k_l105_105445


namespace Anita_should_buy_more_cartons_l105_105004

def Anita_needs (total_needed : ℕ) : Prop :=
total_needed = 26

def Anita_has (strawberries blueberries : ℕ) : Prop :=
strawberries = 10 ∧ blueberries = 9

def additional_cartons (total_needed strawberries blueberries : ℕ) : ℕ :=
total_needed - (strawberries + blueberries)

theorem Anita_should_buy_more_cartons :
  ∀ (total_needed strawberries blueberries : ℕ),
    Anita_needs total_needed →
    Anita_has strawberries blueberries →
    additional_cartons total_needed strawberries blueberries = 7 :=
by
  intros total_needed strawberries blueberries Hneeds Hhas
  sorry

end Anita_should_buy_more_cartons_l105_105004


namespace percentage_cut_in_magazine_budget_l105_105109

noncomputable def magazine_budget_cut (original_budget : ℕ) (cut_amount : ℕ) : ℕ :=
  (cut_amount * 100) / original_budget

theorem percentage_cut_in_magazine_budget : 
  magazine_budget_cut 940 282 = 30 :=
by
  sorry

end percentage_cut_in_magazine_budget_l105_105109


namespace arithmetic_mean_location_l105_105597

theorem arithmetic_mean_location (a b : ℝ) : 
    abs ((a + b) / 2 - a) = abs (b - (a + b) / 2) := 
by 
    sorry

end arithmetic_mean_location_l105_105597


namespace max_friendly_groups_19_max_friendly_groups_20_l105_105728

def friendly_group {Team : Type} (beat : Team → Team → Prop) (A B C : Team) : Prop :=
  beat A B ∧ beat B C ∧ beat C A

def max_friendly_groups_19_teams : ℕ := 285
def max_friendly_groups_20_teams : ℕ := 330

theorem max_friendly_groups_19 {Team : Type} (n : ℕ) (h : n = 19) (beat : Team → Team → Prop) :
  ∃ (G : ℕ), G = max_friendly_groups_19_teams := sorry

theorem max_friendly_groups_20 {Team : Type} (n : ℕ) (h : n = 20) (beat : Team → Team → Prop) :
  ∃ (G : ℕ), G = max_friendly_groups_20_teams := sorry

end max_friendly_groups_19_max_friendly_groups_20_l105_105728


namespace trees_in_one_row_l105_105297

theorem trees_in_one_row (total_revenue : ℕ) (price_per_apple : ℕ) (apples_per_tree : ℕ) (trees_per_row : ℕ)
  (revenue_condition : total_revenue = 30)
  (price_condition : price_per_apple = 1 / 2)
  (apples_condition : apples_per_tree = 5)
  (trees_condition : trees_per_row = 4) :
  trees_per_row = 4 := by
  sorry

end trees_in_one_row_l105_105297


namespace sticks_form_triangle_l105_105049

theorem sticks_form_triangle:
  (2 + 3 > 4) ∧ (2 + 4 > 3) ∧ (3 + 4 > 2) := by
  sorry

end sticks_form_triangle_l105_105049


namespace gerry_bananas_eaten_l105_105773

theorem gerry_bananas_eaten (b : ℝ) : 
  (b + (b + 8) + (b + 16) + 0 + (b + 24) + (b + 32) + (b + 40) + (b + 48) = 220) →
  b + 48 = 56.67 :=
by
  sorry

end gerry_bananas_eaten_l105_105773


namespace arithmetic_sequence_general_formula_l105_105351

theorem arithmetic_sequence_general_formula :
  (∀ n:ℕ, ∃ (a_n : ℕ), ∀ k:ℕ, a_n = 2 * k → k = n)
  ∧ ( 2 * n + 2 * (n + 2) = 8 → 2 * n + 2 * (n + 3) = 12 → a_n = 2 * n )
  ∧ (S_n = (n * (n + 1)) / 2 → S_n = 420 → n = 20) :=
by { sorry }

end arithmetic_sequence_general_formula_l105_105351


namespace work_in_one_day_l105_105499

theorem work_in_one_day (A_days B_days : ℕ) (hA : A_days = 18) (hB : B_days = A_days / 2) :
  (1 / A_days + 1 / B_days) = 1 / 6 := 
by
  sorry

end work_in_one_day_l105_105499


namespace jordan_total_points_l105_105491

-- Definitions based on conditions in the problem
def jordan_attempts (x y : ℕ) : Prop :=
  x + y = 40

def points_from_three_point_shots (x : ℕ) : ℝ :=
  0.75 * x

def points_from_two_point_shots (y : ℕ) : ℝ :=
  0.8 * y

-- Main theorem to prove the total points scored by Jordan
theorem jordan_total_points (x y : ℕ) 
  (h_attempts : jordan_attempts x y) : 
  points_from_three_point_shots x + points_from_two_point_shots y = 30 := 
by
  sorry

end jordan_total_points_l105_105491


namespace sqrt_two_irrational_l105_105617

theorem sqrt_two_irrational : ¬ ∃ (p q : ℕ), (q ≠ 0) ∧ (Nat.gcd p q = 1) ∧ (p ^ 2 = 2 * q ^ 2) := by
  sorry

end sqrt_two_irrational_l105_105617


namespace sum_of_two_digit_numbers_l105_105649

/-- Given two conditions regarding multiplication mistakes, we prove the sum of the numbers. -/
theorem sum_of_two_digit_numbers
  (A B C D : ℕ)
  (h1 : (10 * A + B) * (60 + D) = 2496)
  (h2 : (10 * A + B) * (20 + D) = 936) :
  (10 * A + B) + (10 * C + D) = 63 :=
by
  -- Conditions and necessary steps for solving the problem would go here.
  -- We're focusing on stating the problem, not the solution.
  sorry

end sum_of_two_digit_numbers_l105_105649


namespace rose_clothing_tax_l105_105766

theorem rose_clothing_tax {total_spent total_tax tax_other tax_clothing amount_clothing amount_food amount_other clothing_tax_rate : ℝ} 
  (h_total_spent : total_spent = 100)
  (h_amount_clothing : amount_clothing = 0.5 * total_spent)
  (h_amount_food : amount_food = 0.2 * total_spent)
  (h_amount_other : amount_other = 0.3 * total_spent)
  (h_no_tax_food : True)
  (h_tax_other_rate : tax_other = 0.08 * amount_other)
  (h_total_tax_rate : total_tax = 0.044 * total_spent)
  (h_calculate_tax_clothing : tax_clothing = total_tax - tax_other) :
  clothing_tax_rate = (tax_clothing / amount_clothing) * 100 → 
  clothing_tax_rate = 4 := 
by
  sorry

end rose_clothing_tax_l105_105766


namespace new_average_of_subtracted_elements_l105_105132

theorem new_average_of_subtracted_elements (a b c d e : ℝ) 
  (h_average : (a + b + c + d + e) / 5 = 5) 
  (new_a : ℝ := a - 2) 
  (new_b : ℝ := b - 2) 
  (new_c : ℝ := c - 2) 
  (new_d : ℝ := d - 2) :
  (new_a + new_b + new_c + new_d + e) / 5 = 3.4 := 
by 
  sorry

end new_average_of_subtracted_elements_l105_105132


namespace prob_and_relation_proof_l105_105991

-- Defining conditions
def total_buses : ℕ := 500

def A_on_time : ℕ := 240
def A_not_on_time : ℕ := 20
def B_on_time : ℕ := 210
def B_not_on_time : ℕ := 30

def A_total : ℕ := A_on_time + A_not_on_time
def B_total : ℕ := B_on_time + B_not_on_time

def prob_A_on_time : ℚ := A_on_time / A_total
def prob_B_on_time : ℚ := B_on_time / B_total

-- Defining K^2 calculation
def n : ℕ := total_buses
def a : ℕ := A_on_time
def b : ℕ := A_not_on_time
def c : ℕ := B_on_time
def d : ℕ := B_not_on_time

def K_squared : ℚ :=
  n * (a * d - b * c)^2 / ((a + b) * (c + d) * (a + c) * (b + d))

def threshold_90_percent : ℚ := 2.706

-- Lean theorem statement
theorem prob_and_relation_proof :
  prob_A_on_time = 12 / 13 ∧
  prob_B_on_time = 7 / 8 ∧
  K_squared > threshold_90_percent :=
by {
   sorry
}

end prob_and_relation_proof_l105_105991


namespace inverse_proportion_inequality_l105_105101

variable {x1 x2 y1 y2 : ℝ}

theorem inverse_proportion_inequality
  (h1 : y1 = 6 / x1)
  (h2 : y2 = 6 / x2)
  (hx : x1 < 0 ∧ 0 < x2) :
  y1 < y2 :=
by
  sorry

end inverse_proportion_inequality_l105_105101


namespace final_net_earnings_l105_105538

-- Declare constants representing the problem conditions
def connor_hourly_rate : ℝ := 7.20
def connor_hours_worked : ℝ := 8.0
def emily_hourly_rate : ℝ := 2 * connor_hourly_rate
def sarah_hourly_rate : ℝ := 5 * connor_hourly_rate
def emily_hours_worked : ℝ := 10.0
def connor_deduction_rate : ℝ := 0.05
def emily_deduction_rate : ℝ := 0.08
def sarah_deduction_rate : ℝ := 0.10

-- Combined final net earnings for the day
def combined_final_net_earnings (connor_hourly_rate emily_hourly_rate sarah_hourly_rate
                                  connor_hours_worked emily_hours_worked
                                  connor_deduction_rate emily_deduction_rate sarah_deduction_rate : ℝ) : ℝ :=
  let connor_gross := connor_hourly_rate * connor_hours_worked
  let emily_gross := emily_hourly_rate * emily_hours_worked
  let sarah_gross := sarah_hourly_rate * connor_hours_worked
  let connor_net := connor_gross * (1 - connor_deduction_rate)
  let emily_net := emily_gross * (1 - emily_deduction_rate)
  let sarah_net := sarah_gross * (1 - sarah_deduction_rate)
  connor_net + emily_net + sarah_net

-- The theorem statement proving their combined final net earnings
theorem final_net_earnings : 
  combined_final_net_earnings 7.20 14.40 36.00 8.0 10.0 0.05 0.08 0.10 = 498.24 :=
by sorry

end final_net_earnings_l105_105538


namespace negate_exists_real_l105_105000

theorem negate_exists_real (h : ¬ ∃ x : ℝ, x^2 - 2 ≤ 0) : ∀ x : ℝ, x^2 - 2 > 0 :=
by
  sorry

end negate_exists_real_l105_105000


namespace man_speed_against_current_l105_105370

theorem man_speed_against_current:
  ∀ (V_current : ℝ) (V_still : ℝ) (current_speed : ℝ),
    V_current = V_still + current_speed →
    V_current = 16 →
    current_speed = 3.2 →
    V_still - current_speed = 9.6 :=
by
  intros V_current V_still current_speed h1 h2 h3
  sorry

end man_speed_against_current_l105_105370


namespace average_weight_l105_105704

variable (A B C : ℝ) 

theorem average_weight (h1 : (A + B) / 2 = 48) (h2 : (B + C) / 2 = 42) (h3 : B = 51) :
  (A + B + C) / 3 = 43 := by
  sorry

end average_weight_l105_105704


namespace perpendicular_lines_l105_105185

theorem perpendicular_lines (a : ℝ)
  (line1 : (a^2 + a - 6) * x + 12 * y - 3 = 0)
  (line2 : (a - 1) * x - (a - 2) * y + 4 - a = 0) :
  (a - 2) * (a - 3) * (a + 5) = 0 := sorry

end perpendicular_lines_l105_105185


namespace max_colors_l105_105819

theorem max_colors (n : ℕ) (color : ℕ → ℕ → ℕ)
  (h_color_property : ∀ i j : ℕ, i < 2^n → j < 2^n → color i j = color j ((i + j) % 2^n)) :
  ∃ (c : ℕ), c ≤ 2^n ∧ (∀ i j : ℕ, i < 2^n → j < 2^n → color i j < c) :=
sorry

end max_colors_l105_105819


namespace relation_among_a_b_c_l105_105671

theorem relation_among_a_b_c
  (a : ℝ) (b : ℝ) (c : ℝ)
  (h1 : a = (3 / 5)^4)
  (h2 : b = (3 / 5)^3)
  (h3 : c = Real.log (3 / 5) / Real.log 3) :
  c < a ∧ a < b :=
by
  sorry

end relation_among_a_b_c_l105_105671


namespace cost_of_adult_ticket_is_10_l105_105635

-- Definitions based on the problem's conditions
def num_adults : ℕ := 5
def num_children : ℕ := 2
def cost_concessions : ℝ := 12
def total_cost : ℝ := 76
def cost_child_ticket : ℝ := 7

-- Statement to prove the cost of an adult ticket being $10
theorem cost_of_adult_ticket_is_10 :
  ∃ A : ℝ, (num_adults * A + num_children * cost_child_ticket + cost_concessions = total_cost) ∧ A = 10 :=
by
  sorry

end cost_of_adult_ticket_is_10_l105_105635


namespace maddie_total_payment_l105_105492

def price_palettes : ℝ := 15
def num_palettes : ℕ := 3
def discount_palettes : ℝ := 0.20
def price_lipsticks : ℝ := 2.50
def num_lipsticks_bought : ℕ := 4
def num_lipsticks_pay : ℕ := 3
def price_hair_color : ℝ := 4
def num_hair_color : ℕ := 3
def discount_hair_color : ℝ := 0.10
def sales_tax_rate : ℝ := 0.08

def total_cost_palettes : ℝ := num_palettes * price_palettes
def total_cost_palettes_after_discount : ℝ := total_cost_palettes * (1 - discount_palettes)

def total_cost_lipsticks : ℝ := num_lipsticks_pay * price_lipsticks

def total_cost_hair_color : ℝ := num_hair_color * price_hair_color
def total_cost_hair_color_after_discount : ℝ := total_cost_hair_color * (1 - discount_hair_color)

def total_pre_tax : ℝ := total_cost_palettes_after_discount + total_cost_lipsticks + total_cost_hair_color_after_discount
def total_sales_tax : ℝ := total_pre_tax * sales_tax_rate
def total_cost : ℝ := total_pre_tax + total_sales_tax

theorem maddie_total_payment : total_cost = 58.64 := by
  sorry

end maddie_total_payment_l105_105492


namespace horizontal_distance_P_Q_l105_105231

-- Definitions for the given conditions
def curve (x : ℝ) : ℝ := x^2 + 2 * x - 3

-- Define the points P and Q on the curve
def P_x : Set ℝ := {x | curve x = 8}
def Q_x : Set ℝ := {x | curve x = -1}

-- State the theorem to prove horizontal distance is 3sqrt3
theorem horizontal_distance_P_Q : 
  ∃ (x₁ x₂ : ℝ), x₁ ∈ P_x ∧ x₂ ∈ Q_x ∧ |x₁ - x₂| = 3 * Real.sqrt 3 :=
sorry

end horizontal_distance_P_Q_l105_105231


namespace number_of_ways_to_prepare_all_elixirs_l105_105082

def fairy_methods : ℕ := 2
def elf_methods : ℕ := 2
def fairy_elixirs : ℕ := 3
def elf_elixirs : ℕ := 4

theorem number_of_ways_to_prepare_all_elixirs : 
  (fairy_methods * fairy_elixirs) + (elf_methods * elf_elixirs) = 14 :=
by
  sorry

end number_of_ways_to_prepare_all_elixirs_l105_105082


namespace minimum_value_of_quadratic_l105_105687

theorem minimum_value_of_quadratic : ∀ x : ℝ, (∃ y : ℝ, y = (x-2)^2 - 3) → ∃ m : ℝ, (∀ x : ℝ, (x-2)^2 - 3 ≥ m) ∧ m = -3 :=
by
  sorry

end minimum_value_of_quadratic_l105_105687


namespace scientific_notation_of_1_656_million_l105_105790

theorem scientific_notation_of_1_656_million :
  (1.656 * 10^6 = 1656000) := by
sorry

end scientific_notation_of_1_656_million_l105_105790


namespace ratio_of_areas_l105_105956

theorem ratio_of_areas 
  (A B C D E F : Type)
  (AB AC AD : ℝ)
  (h1 : AB = 130)
  (h2 : AC = 130)
  (h3 : AD = 26)
  (CF : ℝ)
  (h4 : CF = 91)
  (BD : ℝ)
  (h5 : BD = 104)
  (AF : ℝ)
  (h6 : AF = 221)
  (EF DE BE CE : ℝ)
  (h7 : EF / DE = 91 / 104)
  (h8 : CE / BE = 3.5) :
  EF * CE = 318.5 * DE * BE :=
sorry

end ratio_of_areas_l105_105956


namespace geometric_sequence_tan_sum_l105_105199

theorem geometric_sequence_tan_sum
  (A B C : ℝ)
  (a b c : ℝ)
  (h1 : b^2 = a * c)
  (h2 : Real.tan B = 3/4):
  1 / Real.tan A + 1 / Real.tan C = 5 / 3 := 
by
  sorry

end geometric_sequence_tan_sum_l105_105199


namespace expression_value_l105_105574

theorem expression_value (x y z : ℤ) (hx : x = -2) (hy : y = 1) (hz : z = 1) : 
  x^2 * y * z - x * y * z^2 = 6 :=
by
  rw [hx, hy, hz]
  rfl

end expression_value_l105_105574


namespace perp_case_parallel_distance_l105_105549

open Real

-- Define the line equations
def l1 (x y : ℝ) := 2 * x + y + 4 = 0
def l2 (a x y : ℝ) := a * x + 4 * y + 1 = 0

-- Perpendicular condition between l1 and l2
def perpendicular (a : ℝ) := (∃ x y : ℝ, l1 x y ∧ l2 a x y ∧ (2 * -a) / 4 = -1)

-- Parallel condition between l1 and l2
def parallel (a : ℝ) := (∃ x y : ℝ, l1 x y ∧ l2 a x y ∧ a = 8)

noncomputable def intersection_point : (ℝ × ℝ) := (-3/2, -1)

noncomputable def distance_between_lines : ℝ := (3 * sqrt 5) / 4

-- Statement for the intersection point when perpendicular
theorem perp_case (a : ℝ) : perpendicular a → ∃ x y, l1 x y ∧ l2 (-2) x y := 
by
  sorry

-- Statement for the distance when parallel
theorem parallel_distance {a : ℝ} : parallel a → distance_between_lines = (3 * sqrt 5) / 4 :=
by
  sorry

end perp_case_parallel_distance_l105_105549


namespace raise_percentage_to_original_l105_105850

-- Let original_salary be a variable representing the original salary.
-- Since the salary was reduced by 50%, the reduced_salary is half of the original_salary.
-- We need to prove that to get the reduced_salary back to the original_salary, 
-- it must be increased by 100%.

noncomputable def original_salary : ℝ := sorry
noncomputable def reduced_salary : ℝ := original_salary * 0.5

theorem raise_percentage_to_original :
  (original_salary - reduced_salary) / reduced_salary * 100 = 100 :=
sorry

end raise_percentage_to_original_l105_105850


namespace elisa_math_books_l105_105544

theorem elisa_math_books (N M L : ℕ) (h₀ : 24 + M + L + 1 = N + 1) (h₁ : (N + 1) % 9 = 0) (h₂ : (N + 1) % 4 = 0) (h₃ : N < 100) : M = 7 :=
by
  sorry

end elisa_math_books_l105_105544


namespace smallest_positive_period_of_f_extreme_values_of_f_on_interval_l105_105531

noncomputable def f (x : ℝ) : ℝ :=
  let a : ℝ × ℝ := (2 * Real.cos x, Real.sqrt 3 * Real.cos x)
  let b : ℝ × ℝ := (Real.cos x, 2 * Real.sin x)
  a.1 * b.1 + a.2 * b.2

theorem smallest_positive_period_of_f :
  ∃ p > 0, ∀ x : ℝ, f (x + p) = f x ∧ p = Real.pi := sorry

theorem extreme_values_of_f_on_interval :
  ∃ max_val min_val, (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≤ max_val) ∧
                     (∀ x ∈ Set.Icc 0 (Real.pi / 2), min_val ≤ f x) ∧
                     max_val = 3 ∧ min_val = 0 := sorry

end smallest_positive_period_of_f_extreme_values_of_f_on_interval_l105_105531


namespace inequality_holds_for_gt_sqrt2_l105_105525

theorem inequality_holds_for_gt_sqrt2 (x y : ℝ) (hx : x > Real.sqrt 2) (hy : y > Real.sqrt 2) :
  x^4 - x^3 * y + x^2 * y^2 - x * y^3 + y^4 > x^2 + y^2 :=
by {
  sorry
}

end inequality_holds_for_gt_sqrt2_l105_105525


namespace sally_weekend_reading_l105_105963

theorem sally_weekend_reading (pages_on_weekdays : ℕ) (total_pages : ℕ) (weeks : ℕ) (weekdays_per_week : ℕ) (total_days : ℕ) 
  (finishing_time : ℕ) (weekend_days : ℕ) (pages_weekdays_total : ℕ) :
  pages_on_weekdays = 10 →
  total_pages = 180 →
  weeks = 2 →
  weekdays_per_week = 5 →
  weekend_days = (total_days - weekdays_per_week * weeks) →
  total_days = 7 * weeks →
  finishing_time = weeks →
  pages_weekdays_total = pages_on_weekdays * weekdays_per_week * weeks →
  (total_pages - pages_weekdays_total) / weekend_days = 20 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end sally_weekend_reading_l105_105963


namespace simplify_and_evaluate_expression_l105_105767

theorem simplify_and_evaluate_expression : 
  ∀ x : ℝ, x = 1 → ( (x^2 - 5) / (x - 3) - 4 / (x - 3) ) = 4 :=
by
  intros x hx
  simp [hx]
  have eq : (1 * 1 - 5) = -4 := by norm_num -- Verify that the expression simplifies correctly
  sorry -- Skip the actual complex proof steps

end simplify_and_evaluate_expression_l105_105767


namespace gumballs_remaining_l105_105823

theorem gumballs_remaining (Alicia_gumballs : ℕ) (Pedro_gumballs : ℕ) (Total_gumballs : ℕ) (Gumballs_taken_out : ℕ)
  (h1 : Alicia_gumballs = 20)
  (h2 : Pedro_gumballs = Alicia_gumballs + 3 * Alicia_gumballs)
  (h3 : Total_gumballs = Alicia_gumballs + Pedro_gumballs)
  (h4 : Gumballs_taken_out = 40 * Total_gumballs / 100) :
  Total_gumballs - Gumballs_taken_out = 60 := by
  sorry

end gumballs_remaining_l105_105823


namespace smallest_x_with_18_factors_and_factors_18_24_l105_105355

theorem smallest_x_with_18_factors_and_factors_18_24 :
  ∃ (x : ℕ), (∃ (a b : ℕ), x = 2^a * 3^b ∧ 18 ∣ x ∧ 24 ∣ x ∧ (a + 1) * (b + 1) = 18) ∧
    (∀ y, (∃ (c d : ℕ), y = 2^c * 3^d ∧ 18 ∣ y ∧ 24 ∣ y ∧ (c + 1) * (d + 1) = 18) → x ≤ y) :=
by
  sorry

end smallest_x_with_18_factors_and_factors_18_24_l105_105355


namespace xyz_squared_l105_105368

theorem xyz_squared (x y z p q r : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0)
  (hxy : x + y = p) (hyz : y + z = q) (hzx : z + x = r) :
  x^2 + y^2 + z^2 = (p^2 + q^2 + r^2 - p * q - q * r - r * p) / 2 :=
by
  sorry

end xyz_squared_l105_105368


namespace solve_integer_pairs_l105_105712

-- Definition of the predicate that (m, n) satisfies the given equation
def satisfies_equation (m n : ℤ) : Prop :=
  m * n^2 = 2009 * (n + 1)

-- Theorem stating that the only solutions are (4018, 1) and (0, -1)
theorem solve_integer_pairs :
  ∀ (m n : ℤ), satisfies_equation m n ↔ (m = 4018 ∧ n = 1) ∨ (m = 0 ∧ n = -1) :=
by
  sorry

end solve_integer_pairs_l105_105712


namespace single_solution_inequality_l105_105763

theorem single_solution_inequality (a : ℝ) :
  (∃! (x : ℝ), abs (x^2 + 2 * a * x + 3 * a) ≤ 2) ↔ a = 1 ∨ a = 2 := 
sorry

end single_solution_inequality_l105_105763


namespace initial_worth_is_30_l105_105584

-- Definitions based on conditions
def numberOfCoinsLeft := 2
def amountLeft := 12

-- Definition of the value of each gold coin based on amount left and number of coins left
def valuePerCoin : ℕ := amountLeft / numberOfCoinsLeft

-- Define the total worth of sold coins
def soldCoinsWorth (coinsSold : ℕ) : ℕ := coinsSold * valuePerCoin

-- The total initial worth of Roman's gold coins
def totalInitialWorth : ℕ := amountLeft + soldCoinsWorth 3

-- The proof goal
theorem initial_worth_is_30 : totalInitialWorth = 30 :=
by
  sorry

end initial_worth_is_30_l105_105584


namespace find_x_l105_105794

theorem find_x (y : ℝ) (x : ℝ) : 
  (5 + 2*x) / (7 + 3*x + y) = (3 + 4*x) / (4 + 2*x + y) ↔ 
  x = (-19 + Real.sqrt 329) / 16 ∨ x = (-19 - Real.sqrt 329) / 16 :=
by
  sorry

end find_x_l105_105794


namespace area_of_side_face_l105_105879

theorem area_of_side_face (l w h : ℝ)
  (h_front_top : w * h = 0.5 * (l * h))
  (h_top_side : l * h = 1.5 * (w * h))
  (h_volume : l * w * h = 3000) :
  w * h = 200 := 
sorry

end area_of_side_face_l105_105879


namespace angle_C_45_l105_105672

theorem angle_C_45 (A B C : ℝ) 
(h : (Real.cos A + Real.sin A) * (Real.cos B + Real.sin B) = 2) 
(HA : 0 ≤ A) (HB : 0 ≤ B) (HC : 0 ≤ C):
A + B + C = π → 
A = B →
C = π / 2 - B →
C = π / 4 := 
by
  intros;
  sorry

end angle_C_45_l105_105672


namespace julia_tuesday_l105_105892

variable (M : ℕ) -- The number of kids Julia played with on Monday
variable (T : ℕ) -- The number of kids Julia played with on Tuesday

-- Conditions
def condition1 : Prop := M = T + 8
def condition2 : Prop := M = 22

-- Theorem to prove
theorem julia_tuesday : condition1 M T → condition2 M → T = 14 := by
  sorry

end julia_tuesday_l105_105892


namespace probability_blue_or_green_is_two_thirds_l105_105969

-- Definitions for the given conditions
def blue_faces := 3
def red_faces := 2
def green_faces := 1
def total_faces := blue_faces + red_faces + green_faces
def successful_outcomes := blue_faces + green_faces

-- Probability definition
def probability_blue_or_green := (successful_outcomes : ℚ) / total_faces

-- The theorem we want to prove
theorem probability_blue_or_green_is_two_thirds :
  probability_blue_or_green = (2 / 3 : ℚ) :=
by
  -- here would be the proof steps, but we replace them with sorry as per the instructions
  sorry

end probability_blue_or_green_is_two_thirds_l105_105969


namespace factorization_correct_l105_105803

noncomputable def polynomial_expr := 
  λ x : ℝ => (x^2 + 4 * x + 3) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8)

noncomputable def factored_expr := 
  λ x : ℝ => (x^2 + 6 * x + 9) * (x^2 + 6 * x + 1)

theorem factorization_correct : 
  ∀ x : ℝ, polynomial_expr x = factored_expr x :=
by
  intro x
  sorry

end factorization_correct_l105_105803


namespace find_a_values_l105_105909

theorem find_a_values (a n : ℕ) (h1 : 7 * a * n - 3 * n = 2020) :
    a = 68 ∨ a = 289 := sorry

end find_a_values_l105_105909


namespace find_a2_plus_b2_l105_105966

theorem find_a2_plus_b2 (a b : ℝ) (h1 : a * b = 16) (h2 : a + b = 10) : a^2 + b^2 = 68 :=
sorry

end find_a2_plus_b2_l105_105966


namespace length_AC_eq_9_74_l105_105514

-- Define the cyclic quadrilateral and given constraints
noncomputable def quad (A B C D : Type) : Prop := sorry
def angle_BAC := 50
def angle_ADB := 60
def AD := 3
def BC := 9

-- Prove that length of AC is 9.74 given the above conditions
theorem length_AC_eq_9_74 
  (A B C D : Type)
  (h_quad : quad A B C D)
  (h_angle_BAC : angle_BAC = 50)
  (h_angle_ADB : angle_ADB = 60)
  (h_AD : AD = 3)
  (h_BC : BC = 9) :
  ∃ AC, AC = 9.74 :=
sorry

end length_AC_eq_9_74_l105_105514


namespace lele_has_enough_money_and_remaining_19_yuan_l105_105470

def price_A : ℝ := 46.5
def price_B : ℝ := 54.5
def total_money : ℝ := 120

theorem lele_has_enough_money_and_remaining_19_yuan : 
  (price_A + price_B ≤ total_money) ∧ (total_money - (price_A + price_B) = 19) :=
by
  sorry

end lele_has_enough_money_and_remaining_19_yuan_l105_105470


namespace train_speed_l105_105346

theorem train_speed (v : ℝ) : (∃ t : ℝ, 2 * v + t * v = 285 ∧ t = 285 / 38) → v = 30 :=
by
  sorry

end train_speed_l105_105346


namespace least_x_for_inequality_l105_105691

theorem least_x_for_inequality : 
  ∃ (x : ℝ), (-x^2 + 9 * x - 20 ≤ 0) ∧ ∀ y, (-y^2 + 9 * y - 20 ≤ 0) → x ≤ y ∧ x = 4 := 
by
  sorry

end least_x_for_inequality_l105_105691


namespace contractor_total_received_l105_105792

-- Define the conditions
def days_engaged : ℕ := 30
def daily_earnings : ℝ := 25
def fine_per_absence_day : ℝ := 7.50
def days_absent : ℕ := 4

-- Define the days worked based on conditions
def days_worked : ℕ := days_engaged - days_absent

-- Define the total earnings and total fines
def total_earnings : ℝ := days_worked * daily_earnings
def total_fines : ℝ := days_absent * fine_per_absence_day

-- Define the total amount received
def total_amount_received : ℝ := total_earnings - total_fines

-- State the theorem
theorem contractor_total_received :
  total_amount_received = 620 := 
by
  sorry

end contractor_total_received_l105_105792


namespace total_turtles_rabbits_l105_105882

-- Number of turtles and rabbits on Happy Island
def turtles_happy : ℕ := 120
def rabbits_happy : ℕ := 80

-- Number of turtles and rabbits on Lonely Island
def turtles_lonely : ℕ := turtles_happy / 3
def rabbits_lonely : ℕ := turtles_lonely

-- Number of turtles and rabbits on Serene Island
def rabbits_serene : ℕ := 2 * rabbits_lonely
def turtles_serene : ℕ := (3 * rabbits_lonely) / 4

-- Number of turtles and rabbits on Tranquil Island
def turtles_tranquil : ℕ := (turtles_happy - turtles_serene) + 5
def rabbits_tranquil : ℕ := turtles_tranquil

-- Proving the total numbers
theorem total_turtles_rabbits :
    turtles_happy = 120 ∧ rabbits_happy = 80 ∧
    turtles_lonely = 40 ∧ rabbits_lonely = 40 ∧
    turtles_serene = 30 ∧ rabbits_serene = 80 ∧
    turtles_tranquil = 95 ∧ rabbits_tranquil = 95 ∧
    (turtles_happy + turtles_lonely + turtles_serene + turtles_tranquil = 285) ∧
    (rabbits_happy + rabbits_lonely + rabbits_serene + rabbits_tranquil = 295) := 
    by
        -- Here we prove each part step by step using the definitions and conditions provided above
        sorry

end total_turtles_rabbits_l105_105882


namespace number_of_integer_values_l105_105411

theorem number_of_integer_values (x : ℤ) (h : ⌊Real.sqrt x⌋ = 8) : ∃ n : ℕ, n = 17 :=
by
  sorry

end number_of_integer_values_l105_105411


namespace part1_solution_set_a_eq_2_part2_range_of_a_l105_105782

noncomputable def f (a x : ℝ) : ℝ := abs (x - a) + abs (2 * x - 2)

theorem part1_solution_set_a_eq_2 :
  { x : ℝ | f 2 x > 2 } = { x | x < (2 / 3) } ∪ { x | x > 2 } :=
by
  sorry

theorem part2_range_of_a :
  { a : ℝ | ∀ x : ℝ, f a x ≥ 2 } = { a | a ≤ -1 } ∪ { a | a ≥ 3 } :=
by
  sorry

end part1_solution_set_a_eq_2_part2_range_of_a_l105_105782


namespace value_of_a5_l105_105114

theorem value_of_a5 {a_1 a_3 a_5 : ℤ} (n : ℕ) (hn : n = 8) (h1 : (1 - x)^n = 1 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6 + a_7 * x^7 + a_8 * x^8) (h_ratio : a_1 / a_3 = 1 / 7) :
  a_5 = -56 := 
sorry

end value_of_a5_l105_105114


namespace value_is_20_l105_105612

-- Define the conditions
def number : ℕ := 5
def value := number + 3 * number

-- State the theorem
theorem value_is_20 : value = 20 := by
  -- Proof goes here
  sorry

end value_is_20_l105_105612


namespace grid_square_count_l105_105379

theorem grid_square_count :
  let width := 6
  let height := 6
  let num_1x1 := (width - 1) * (height - 1)
  let num_2x2 := (width - 2) * (height - 2)
  let num_3x3 := (width - 3) * (height - 3)
  let num_4x4 := (width - 4) * (height - 4)
  num_1x1 + num_2x2 + num_3x3 + num_4x4 = 54 :=
by
  sorry

end grid_square_count_l105_105379


namespace fraction_c_d_l105_105282

theorem fraction_c_d (x y c d : ℚ) (hx : x ≠ 0) (hy : y ≠ 0) (hd : d ≠ 0) 
  (h1 : 8 * x - 6 * y = c) (h2 : 10 * y - 15 * x = d) :
  c / d = -8 / 15 :=
sorry

end fraction_c_d_l105_105282


namespace floor_length_l105_105844

theorem floor_length (b l : ℝ)
  (h1 : l = 3 * b)
  (h2 : 3 * b^2 = 484 / 3) :
  l = 22 := 
sorry

end floor_length_l105_105844


namespace isabel_uploaded_pictures_l105_105900

theorem isabel_uploaded_pictures :
  let album1 := 10
  let total_other_pictures := 5 * 3
  let total_pictures := album1 + total_other_pictures
  total_pictures = 25 :=
by
  let album1 := 10
  let total_other_pictures := 5 * 3
  let total_pictures := album1 + total_other_pictures
  show total_pictures = 25
  sorry

end isabel_uploaded_pictures_l105_105900


namespace downstream_speed_l105_105158

-- Definitions based on the conditions
def V_m : ℝ := 50 -- speed of the man in still water
def V_upstream : ℝ := 45 -- speed of the man when rowing upstream

-- The statement to prove
theorem downstream_speed : ∃ (V_s V_downstream : ℝ), V_upstream = V_m - V_s ∧ V_downstream = V_m + V_s ∧ V_downstream = 55 := 
by
  sorry

end downstream_speed_l105_105158


namespace janet_dresses_total_pockets_l105_105715

theorem janet_dresses_total_pockets :
  let dresses := 24
  let with_pockets := dresses / 2
  let with_two_pockets := with_pockets / 3
  let with_three_pockets := with_pockets - with_two_pockets
  let total_two_pockets := with_two_pockets * 2
  let total_three_pockets := with_three_pockets * 3
  total_two_pockets + total_three_pockets = 32 := by
  sorry

end janet_dresses_total_pockets_l105_105715


namespace distance_is_one_l105_105187

noncomputable def distance_between_bisectors_and_centroid : ℝ :=
  let AB := 9
  let AC := 12
  let BC := Real.sqrt (AB^2 + AC^2)
  let CD := BC / 2
  let CE := (2/3) * CD
  let r := (AB * AC) / (2 * (AB + AC + BC) / 2)
  let K := CE - r
  K

theorem distance_is_one : distance_between_bisectors_and_centroid = 1 :=
  sorry

end distance_is_one_l105_105187


namespace train_speed_160m_6sec_l105_105124

noncomputable def train_speed (distance time : ℕ) : ℚ :=
(distance : ℚ) / (time : ℚ)

theorem train_speed_160m_6sec : train_speed 160 6 = 26.67 :=
by
  simp [train_speed]
  norm_num
  sorry

end train_speed_160m_6sec_l105_105124


namespace greatest_possible_value_l105_105630

theorem greatest_possible_value (x : ℝ) (hx : x^3 + (1 / x^3) = 9) : x + (1 / x) = 3 := by
  sorry

end greatest_possible_value_l105_105630


namespace area_of_X_part_l105_105363

theorem area_of_X_part :
    (∃ s : ℝ, s^2 = 2520 ∧ 
     (∃ E F G H : ℝ, E = F ∧ F = G ∧ G = H ∧ 
         E = s / 4 ∧ F = s / 4 ∧ G = s / 4 ∧ H = s / 4) ∧ 
     2520 * 11 / 24 = 1155) :=
by
  sorry

end area_of_X_part_l105_105363


namespace geometric_sequence_divisible_by_ten_million_l105_105037

theorem geometric_sequence_divisible_by_ten_million 
  (a1 a2 : ℝ)
  (h1 : a1 = 1 / 2)
  (h2 : a2 = 50) :
  ∀ n : ℕ, (n ≥ 5) → (∃ k : ℕ, (a1 * (a2 / a1)^(n - 1)) = k * 10^7) :=
by
  sorry

end geometric_sequence_divisible_by_ten_million_l105_105037


namespace algebraic_expression_value_l105_105667

theorem algebraic_expression_value (x y : ℝ) (h : x + 2 * y = 2) : 2 * x + 4 * y - 1 = 3 :=
sorry

end algebraic_expression_value_l105_105667


namespace infinite_squares_in_ap_l105_105731

theorem infinite_squares_in_ap
    (a d : ℤ)
    (h : ∃ n : ℤ, a^2 = a + n * d) :
    ∀ N : ℕ, ∃ m : ℤ, ∃ k : ℕ, k > N ∧ m^2 = a + k * d :=
by
  sorry

end infinite_squares_in_ap_l105_105731


namespace probability_green_ball_eq_l105_105787

noncomputable def prob_green_ball : ℚ := 
  1 / 3 * (5 / 18) + 1 / 3 * (1 / 2) + 1 / 3 * (1 / 2)

theorem probability_green_ball_eq : 
  prob_green_ball = 23 / 54 := 
  by
  sorry

end probability_green_ball_eq_l105_105787


namespace robin_total_distance_l105_105505

theorem robin_total_distance
  (d : ℕ)
  (d1 : ℕ)
  (h1 : d = 500)
  (h2 : d1 = 200)
  : 2 * d1 + d = 900 :=
by
  rewrite [h1, h2]
  rfl

end robin_total_distance_l105_105505


namespace no_real_solution_l105_105976

-- Define the given equation as a function
def equation (x y : ℝ) : ℝ := 3 * x^2 + 5 * y^2 - 9 * x - 20 * y + 30 + 4 * x * y

-- State that the equation equals zero has no real solution.
theorem no_real_solution : ∀ x y : ℝ, equation x y ≠ 0 :=
by sorry

end no_real_solution_l105_105976


namespace original_salary_l105_105382

theorem original_salary (S : ℝ) (h : 1.10 * S * 0.95 = 3135) : S = 3000 := 
by 
  sorry

end original_salary_l105_105382


namespace chocolate_bars_in_large_box_l105_105100

theorem chocolate_bars_in_large_box
  (number_of_small_boxes : ℕ)
  (chocolate_bars_per_box : ℕ)
  (h1 : number_of_small_boxes = 21)
  (h2 : chocolate_bars_per_box = 25) :
  number_of_small_boxes * chocolate_bars_per_box = 525 :=
by {
  sorry
}

end chocolate_bars_in_large_box_l105_105100


namespace spider_crawl_distance_l105_105473

theorem spider_crawl_distance :
  let a := -3
  let b := -8
  let c := 4
  let d := 7
  abs (b - a) + abs (c - b) + abs (d - c) = 20 :=
by
  let a := -3
  let b := -8
  let c := 4
  let d := 7
  sorry

end spider_crawl_distance_l105_105473


namespace blackBurgerCost_l105_105595

def ArevaloFamilyBill (smokySalmonCost blackBurgerCost chickenKatsuCost totalBill : ℝ) : Prop :=
  smokySalmonCost = 40 ∧ chickenKatsuCost = 25 ∧ 
  totalBill = smokySalmonCost + blackBurgerCost + chickenKatsuCost + 
    0.15 * (smokySalmonCost + blackBurgerCost + chickenKatsuCost)

theorem blackBurgerCost (smokySalmonCost chickenKatsuCost change : ℝ) (B : ℝ) 
  (h1 : smokySalmonCost = 40) 
  (h2 : chickenKatsuCost = 25)
  (h3 : 100 - change = 92) 
  (h4 : ArevaloFamilyBill smokySalmonCost B chickenKatsuCost 92) : 
  B = 15 :=
sorry

end blackBurgerCost_l105_105595


namespace lisa_hotdog_record_l105_105477

theorem lisa_hotdog_record
  (hotdogs_eaten : ℕ)
  (eaten_in_first_half : ℕ)
  (rate_per_minute : ℕ)
  (time_in_minutes : ℕ)
  (first_half_duration : ℕ)
  (remaining_time : ℕ) :
  eaten_in_first_half = 20 →
  rate_per_minute = 11 →
  first_half_duration = 5 →
  remaining_time = 5 →
  time_in_minutes = first_half_duration + remaining_time →
  hotdogs_eaten = eaten_in_first_half + rate_per_minute * remaining_time →
  hotdogs_eaten = 75 := by
  intros
  sorry

end lisa_hotdog_record_l105_105477


namespace fg_of_neg5_eq_484_l105_105512

def f (x : Int) : Int := x * x
def g (x : Int) : Int := 6 * x + 8

theorem fg_of_neg5_eq_484 : f (g (-5)) = 484 := 
  sorry

end fg_of_neg5_eq_484_l105_105512


namespace initial_crayons_per_box_l105_105884

-- Define the initial total number of crayons in terms of x
def total_initial_crayons (x : ℕ) : ℕ := 4 * x

-- Define the crayons given to Mae
def crayons_to_Mae : ℕ := 5

-- Define the crayons given to Lea
def crayons_to_Lea : ℕ := 12

-- Define the remaining crayons
def remaining_crayons : ℕ := 15

-- Prove that the initial number of crayons per box is 8 given the conditions
theorem initial_crayons_per_box (x : ℕ) : total_initial_crayons x - crayons_to_Mae - crayons_to_Lea = remaining_crayons → x = 8 :=
by
  intros h
  sorry

end initial_crayons_per_box_l105_105884


namespace Mike_changed_2_sets_of_tires_l105_105338

theorem Mike_changed_2_sets_of_tires
  (wash_time_per_car : ℕ := 10)
  (oil_change_time_per_car : ℕ := 15)
  (tire_change_time_per_set : ℕ := 30)
  (num_washed_cars : ℕ := 9)
  (num_oil_changes : ℕ := 6)
  (total_work_time_minutes : ℕ := 4 * 60) :
  ((total_work_time_minutes - (num_washed_cars * wash_time_per_car + num_oil_changes * oil_change_time_per_car)) / tire_change_time_per_set) = 2 :=
by
  sorry

end Mike_changed_2_sets_of_tires_l105_105338


namespace LTE_divisibility_l105_105078

theorem LTE_divisibility (m : ℕ) (h_pos : 0 < m) :
  (∀ k : ℕ, k % 2 = 1 ∧ k ≥ 3 → 2^m ∣ k^m - 1) ↔ m = 1 ∨ m = 2 ∨ m = 4 :=
by
  sorry

end LTE_divisibility_l105_105078


namespace find_p_over_q_at_neg1_l105_105886

noncomputable def p (x : ℝ) : ℝ := (-27 / 8) * x
noncomputable def q (x : ℝ) : ℝ := (x + 5) * (x - 1)

theorem find_p_over_q_at_neg1 : p (-1) / q (-1) = 27 / 64 := by
  -- Skipping the proof
  sorry

end find_p_over_q_at_neg1_l105_105886


namespace find_second_number_l105_105211

theorem find_second_number (x : ℕ) : 
  ((20 + 40 + 60) / 3 = 4 + ((x + 10 + 28) / 3)) → x = 70 :=
by {
  -- let lhs = (20 + 40 + 60) / 3
  -- let rhs = 4 + ((x + 10 + 28) / 3)
  -- rw rhs at lhs,
  -- value the lhs and rhs,
  -- prove x = 70
  sorry
}

end find_second_number_l105_105211


namespace find_a1_l105_105036

noncomputable def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
∀ n : ℕ, a (n+1) + a n = 4*n

theorem find_a1 (a : ℕ → ℕ) (h : is_arithmetic_sequence a) : a 1 = 1 := by
  sorry

end find_a1_l105_105036


namespace polynomial_has_one_real_root_l105_105641

theorem polynomial_has_one_real_root (a : ℝ) :
  (∃! x : ℝ, x^3 - 2 * a * x^2 + 3 * a * x + a^2 - 2 = 0) :=
sorry

end polynomial_has_one_real_root_l105_105641


namespace castor_chess_players_l105_105896

theorem castor_chess_players (total_players : ℕ) (never_lost_to_ai : ℕ)
  (h1 : total_players = 40) (h2 : never_lost_to_ai = total_players / 4) :
  (total_players - never_lost_to_ai) = 30 :=
by
  sorry

end castor_chess_players_l105_105896


namespace martha_cakes_required_l105_105958

-- Conditions
def number_of_children : ℝ := 3.0
def cakes_per_child : ℝ := 18.0

-- The main statement to prove
theorem martha_cakes_required:
  (number_of_children * cakes_per_child) = 54.0 := 
by
  sorry

end martha_cakes_required_l105_105958


namespace edit_post_time_zero_l105_105504

-- Define the conditions
def total_videos : ℕ := 4
def setup_time : ℕ := 1
def painting_time_per_video : ℕ := 1
def cleanup_time : ℕ := 1
def total_production_time_per_video : ℕ := 3

-- Define the total time spent on setup, painting, and cleanup for one video
def spc_time : ℕ := setup_time + painting_time_per_video + cleanup_time

-- State the theorem to be proven
theorem edit_post_time_zero : (total_production_time_per_video - spc_time) = 0 := by
  sorry

end edit_post_time_zero_l105_105504


namespace simplify_expression_l105_105474

theorem simplify_expression (y : ℝ) : 
  4 * y + 9 * y ^ 2 + 8 - (3 - 4 * y - 9 * y ^ 2) = 18 * y ^ 2 + 8 * y + 5 :=
by
  sorry

end simplify_expression_l105_105474


namespace matching_polygons_pairs_l105_105390

noncomputable def are_matching_pairs (n m : ℕ) : Prop :=
  2 * ((n - 2) * 180 / n) = 3 * (360 / m)

theorem matching_polygons_pairs (n m : ℕ) :
  are_matching_pairs n m → (n, m) = (3, 9) ∨ (n, m) = (4, 6) ∨ (n, m) = (5, 5) ∨ (n, m) = (8, 4) :=
sorry

end matching_polygons_pairs_l105_105390


namespace car_pedestrian_speed_ratio_l105_105041

theorem car_pedestrian_speed_ratio
  (L : ℝ) -- Length of the bridge
  (v_p v_c : ℝ) -- Speed of pedestrian and car
  (h1 : (4 / 9) * L / v_p = (5 / 9) * L / v_p + (5 / 9) * L / v_c) -- Initial meet at bridge start
  (h2 : (4 / 9) * L / v_p = (8 / 9) * L / v_c) -- If pedestrian continues to walk
  : v_c / v_p = 9 :=
sorry

end car_pedestrian_speed_ratio_l105_105041


namespace width_of_margin_l105_105059

-- Given conditions as definitions
def total_area : ℝ := 20 * 30
def percentage_used : ℝ := 0.64
def used_area : ℝ := percentage_used * total_area

-- Definition of the width of the typing area
def width_after_margin (x : ℝ) : ℝ := 20 - 2 * x

-- Definition of the length after top and bottom margins
def length_after_margin : ℝ := 30 - 6

-- Calculate the area used considering the margins
def typing_area (x : ℝ) : ℝ := (width_after_margin x) * length_after_margin

-- Statement to prove
theorem width_of_margin : ∃ x : ℝ, typing_area x = used_area ∧ x = 2 := by
  -- We give the prompt to eventually prove the theorem with the correct value
  sorry

end width_of_margin_l105_105059


namespace small_gate_width_l105_105092

-- Bob's garden dimensions
def garden_length : ℝ := 225
def garden_width : ℝ := 125

-- Total fencing needed, including the gates
def total_fencing : ℝ := 687

-- Width of the large gate
def large_gate_width : ℝ := 10

-- Perimeter of the garden without gates
def garden_perimeter : ℝ := 2 * (garden_length + garden_width)

-- Width of the small gate
theorem small_gate_width :
  2 * (garden_length + garden_width) + small_gate + large_gate_width = total_fencing → small_gate = 3 :=
by
  sorry

end small_gate_width_l105_105092


namespace haley_tickets_l105_105452

-- Conditions
def cost_per_ticket : ℕ := 4
def extra_tickets : ℕ := 5
def total_spent : ℕ := 32
def cost_extra_tickets : ℕ := extra_tickets * cost_per_ticket

-- Main proof problem
theorem haley_tickets (T : ℕ) (h : 4 * T + cost_extra_tickets = total_spent) :
  T = 3 := sorry

end haley_tickets_l105_105452


namespace a_75_eq_24_l105_105776

variable {a : ℕ → ℤ}

-- Conditions for the problem
def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def a_15_eq_8 : a 15 = 8 := sorry

def a_60_eq_20 : a 60 = 20 := sorry

-- The theorem we want to prove
theorem a_75_eq_24 (d : ℤ) (h_seq : is_arithmetic_sequence a d) (h15 : a 15 = 8) (h60 : a 60 = 20) : a 75 = 24 :=
  by
    sorry

end a_75_eq_24_l105_105776


namespace range_of_a_for_increasing_l105_105080

noncomputable def f (a x : ℝ) : ℝ := x * abs (2 * a - x) + 2 * x

theorem range_of_a_for_increasing (a : ℝ) :
  -1 ≤ a ∧ a ≤ 1 ↔ ∀ x y : ℝ, x < y → f a x ≤ f a y :=
sorry

end range_of_a_for_increasing_l105_105080


namespace least_integer_remainder_condition_l105_105315

def is_least_integer_with_remainder_condition (n : ℕ) : Prop :=
  n > 1 ∧ (∀ k ∈ [3, 4, 5, 6, 7, 10, 11], n % k = 1)

theorem least_integer_remainder_condition : ∃ (n : ℕ), is_least_integer_with_remainder_condition n ∧ n = 4621 :=
by
  -- The proof will go here.
  sorry

end least_integer_remainder_condition_l105_105315


namespace complement_union_l105_105048

theorem complement_union (U A B complement_U_A : Set Int) (hU : U = {-1, 0, 1, 2}) 
  (hA : A = {-1, 2}) (hB : B = {0, 2}) (hC : complement_U_A = {0, 1}) :
  complement_U_A ∪ B = {0, 1, 2} := by
  sorry

end complement_union_l105_105048


namespace tank_plastering_cost_l105_105200

noncomputable def plastering_cost (L W D : ℕ) (cost_per_sq_meter : ℚ) : ℚ :=
  let A_bottom := L * W
  let A_long_walls := 2 * (L * D)
  let A_short_walls := 2 * (W * D)
  let A_total := A_bottom + A_long_walls + A_short_walls
  A_total * cost_per_sq_meter

theorem tank_plastering_cost :
  plastering_cost 25 12 6 0.25 = 186 := by
  sorry

end tank_plastering_cost_l105_105200


namespace cuboid_volume_l105_105189

variable (length width height : ℕ)

-- Conditions given in the problem
def cuboid_edges := (length = 2) ∧ (width = 5) ∧ (height = 8)

-- Mathematically equivalent statement to be proved
theorem cuboid_volume : cuboid_edges length width height → length * width * height = 80 := by
  sorry

end cuboid_volume_l105_105189


namespace chris_babysitting_hours_l105_105982

theorem chris_babysitting_hours (h : ℕ) (video_game_cost candy_cost earn_per_hour leftover total_cost : ℕ) :
  video_game_cost = 60 ∧
  candy_cost = 5 ∧
  earn_per_hour = 8 ∧
  leftover = 7 ∧
  total_cost = video_game_cost + candy_cost ∧
  earn_per_hour * h = total_cost + leftover
  → h = 9 := by
  intros
  sorry

end chris_babysitting_hours_l105_105982


namespace dan_money_left_l105_105988

theorem dan_money_left (initial_amount spent_amount remaining_amount : ℤ) (h1 : initial_amount = 300) (h2 : spent_amount = 100) : remaining_amount = 200 :=
by 
  sorry

end dan_money_left_l105_105988


namespace unique_positive_integers_l105_105481

theorem unique_positive_integers (x y : ℕ) (h1 : x^2 + 84 * x + 2008 = y^2) : x + y = 80 :=
  sorry

end unique_positive_integers_l105_105481


namespace arithmetic_sequence_common_difference_l105_105572

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (h_arith : ∀ n m, a (n + 1) - a n = a (m + 1) - a m)
  (h_a2 : a 2 = 3)
  (h_a7 : a 7 = 13) : 
  ∃ d, ∀ n, a n = a 1 + (n - 1) * d ∧ d = 2 := 
by
  sorry

end arithmetic_sequence_common_difference_l105_105572


namespace intersection_A_B_l105_105849

def setA : Set ℝ := {x : ℝ | x > -1}
def setB : Set ℝ := {x : ℝ | x < 3}
def setIntersection : Set ℝ := {x : ℝ | x > -1 ∧ x < 3}

theorem intersection_A_B :
  setA ∩ setB = setIntersection :=
by sorry

end intersection_A_B_l105_105849


namespace alchemerion_age_problem_l105_105868

theorem alchemerion_age_problem
  (A S F : ℕ)  -- Declare the ages as natural numbers
  (h1 : A = 3 * S)  -- Condition 1: Alchemerion is 3 times his son's age
  (h2 : F = 2 * A + 40)  -- Condition 2: His father’s age is 40 years more than twice his age
  (h3 : A + S + F = 1240)  -- Condition 3: Together they are 1240 years old
  (h4 : A = 360)  -- Condition 4: Alchemerion is 360 years old
  : 40 = F - 2 * A :=  -- Conclusion: The number of years more than twice Alchemerion’s age is 40
by
  sorry  -- Proof can be filled in here

end alchemerion_age_problem_l105_105868


namespace negation_of_exists_x_quad_eq_zero_l105_105044

theorem negation_of_exists_x_quad_eq_zero :
  ¬ ∃ x : ℝ, x^2 + 2 * x + 5 = 0 ↔ ∀ x : ℝ, x^2 + 2 * x + 5 ≠ 0 :=
by sorry

end negation_of_exists_x_quad_eq_zero_l105_105044


namespace parabola_equation_l105_105729

theorem parabola_equation (h_axis : ∃ p > 0, x = p / 2) :
  ∃ p > 0, y^2 = -2 * p * x :=
by 
  -- proof steps will be added here
  sorry

end parabola_equation_l105_105729


namespace solve_for_a_l105_105389

theorem solve_for_a (x y a : ℤ) (h1 : x = 1) (h2 : y = 2) (h3 : x - a * y = 3) : a = -1 :=
by
  -- Proof is skipped
  sorry

end solve_for_a_l105_105389


namespace maria_carrots_l105_105069

theorem maria_carrots :
  ∀ (picked initially thrownOut moreCarrots totalLeft : ℕ),
    initially = 48 →
    thrownOut = 11 →
    totalLeft = 52 →
    moreCarrots = totalLeft - (initially - thrownOut) →
    moreCarrots = 15 :=
by
  intros
  sorry

end maria_carrots_l105_105069


namespace crayons_per_unit_l105_105652

theorem crayons_per_unit :
  ∀ (units : ℕ) (cost_per_crayon : ℕ) (total_cost : ℕ),
    units = 4 →
    cost_per_crayon = 2 →
    total_cost = 48 →
    (total_cost / cost_per_crayon) / units = 6 :=
by
  intros units cost_per_crayon total_cost h_units h_cost_per_crayon h_total_cost
  sorry

end crayons_per_unit_l105_105652


namespace eval_dagger_l105_105655

noncomputable def dagger (m n p q : ℕ) : ℚ := 
  (m * p) * (q / n)

theorem eval_dagger : dagger 5 16 12 5 = 75 / 4 := 
by 
  sorry

end eval_dagger_l105_105655


namespace serena_mother_age_l105_105752

theorem serena_mother_age {x : ℕ} (h : 39 + x = 3 * (9 + x)) : x = 6 := 
by
  sorry

end serena_mother_age_l105_105752


namespace rectangle_dimensions_l105_105600

theorem rectangle_dimensions (w l : ℕ) 
  (h1 : l = 2 * w) 
  (h2 : 2 * (w + l) = 6 * w ^ 2) : 
  w = 1 ∧ l = 2 :=
by sorry

end rectangle_dimensions_l105_105600


namespace eval_expression_l105_105825

theorem eval_expression : (2: ℤ)^2 - 3 * (2: ℤ) + 2 = 0 := by
  sorry

end eval_expression_l105_105825


namespace notebooks_per_child_if_half_l105_105676

theorem notebooks_per_child_if_half (C N : ℕ) 
    (h1 : N = C / 8) 
    (h2 : C * N = 512) : 
    512 / (C / 2) = 16 :=
by
    sorry

end notebooks_per_child_if_half_l105_105676


namespace vertex_angle_measure_l105_105051

-- Definitions for Lean Proof
def isosceles_triangle (α β γ : ℝ) : Prop := (α = β) ∨ (α = γ) ∨ (β = γ)
def exterior_angle (interior exterior : ℝ) : Prop := interior + exterior = 180

-- Conditions from the problem
variables (α β γ : ℝ)
variable (ext_angle : ℝ := 110)

-- Lean 4 statement: The measure of the vertex angle is 70° or 40°
theorem vertex_angle_measure :
  isosceles_triangle α β γ ∧
  (exterior_angle γ ext_angle ∨ exterior_angle α ext_angle ∨ exterior_angle β ext_angle) →
  (γ = 70 ∨ γ = 40) :=
by
  sorry

end vertex_angle_measure_l105_105051


namespace mrs_hilt_total_payment_l105_105434

-- Define the conditions
def number_of_hot_dogs : ℕ := 6
def cost_per_hot_dog : ℝ := 0.50

-- Define the total cost
def total_cost : ℝ := number_of_hot_dogs * cost_per_hot_dog

-- State the theorem to prove the total cost
theorem mrs_hilt_total_payment : total_cost = 3.00 := 
by
  sorry

end mrs_hilt_total_payment_l105_105434


namespace garden_length_l105_105745

def PerimeterLength (P : ℕ) (length : ℕ) (breadth : ℕ) : Prop :=
  P = 2 * (length + breadth)

theorem garden_length
  (P : ℕ)
  (breadth : ℕ)
  (h1 : P = 480)
  (h2 : breadth = 100):
  ∃ length : ℕ, PerimeterLength P length breadth ∧ length = 140 :=
by
  use 140
  sorry

end garden_length_l105_105745


namespace compute_expression_l105_105815

variable (a b : ℝ)

theorem compute_expression : 
  (8 * a^3 * b) * (4 * a * b^2) * (1 / (2 * a * b)^3) = 4 * a := 
by sorry

end compute_expression_l105_105815


namespace min_value_of_expression_l105_105415

open Real

noncomputable def minValue (x y z : ℝ) : ℝ :=
  x + 3 * y + 5 * z

theorem min_value_of_expression : 
  ∀ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧ x * y * z = 8 → minValue x y z = 14.796 :=
by
  intros x y z h
  sorry

end min_value_of_expression_l105_105415


namespace quotient_is_36_l105_105002

-- Conditions
def divisor := 85
def remainder := 26
def dividend := 3086

-- The Question and Answer (proof required)
theorem quotient_is_36 (quotient : ℕ) (h : dividend = (divisor * quotient) + remainder) : quotient = 36 := by 
  sorry

end quotient_is_36_l105_105002


namespace determine_k_l105_105421

theorem determine_k (k : ℝ) (h : (-1)^2 - k * (-1) + 1 = 0) : k = -2 :=
by
  sorry

end determine_k_l105_105421


namespace total_highlighters_correct_l105_105216

variable (y p b : ℕ)
variable (total_highlighters : ℕ)

def num_yellow_highlighters := 7
def num_pink_highlighters := num_yellow_highlighters + 7
def num_blue_highlighters := num_pink_highlighters + 5
def total_highlighters_in_drawer := num_yellow_highlighters + num_pink_highlighters + num_blue_highlighters

theorem total_highlighters_correct : 
  total_highlighters_in_drawer = 40 :=
sorry

end total_highlighters_correct_l105_105216


namespace zero_point_in_interval_l105_105023

noncomputable def f (x a : ℝ) := 2^x - 2/x - a

theorem zero_point_in_interval (a : ℝ) : 
  (∃ x, 1 < x ∧ x < 2 ∧ f x a = 0) → 0 < a ∧ a < 3 :=
by
  sorry

end zero_point_in_interval_l105_105023


namespace number_of_women_is_24_l105_105401

-- Define the variables and conditions
variables (x : ℕ) (men_initial : ℕ) (women_initial : ℕ) (men_current : ℕ) (women_current : ℕ)

-- representing the initial ratio and the changes
def initial_conditions : Prop :=
  men_initial = 4 * x ∧ women_initial = 5 * x ∧
  men_current = men_initial + 2 ∧ women_current = 2 * (women_initial - 3)

-- representing the current number of men
def current_men_condition : Prop := men_current = 14

-- The proof we need to generate
theorem number_of_women_is_24 (x : ℕ) (men_initial women_initial men_current women_current : ℕ)
  (h1 : initial_conditions x men_initial women_initial men_current women_current)
  (h2 : current_men_condition men_current) : women_current = 24 :=
by
  -- proof steps here
  sorry

end number_of_women_is_24_l105_105401


namespace mean_of_remaining_two_numbers_l105_105703

theorem mean_of_remaining_two_numbers :
  let n1 := 1871
  let n2 := 1997
  let n3 := 2023
  let n4 := 2029
  let n5 := 2113
  let n6 := 2125
  let n7 := 2137
  let total_sum := n1 + n2 + n3 + n4 + n5 + n6 + n7
  let known_mean := 2100
  let mean_of_other_two := 1397.5
  total_sum = 13295 →
  5 * known_mean = 10500 →
  total_sum - 10500 = 2795 →
  2795 / 2 = mean_of_other_two :=
by
  intros
  sorry

end mean_of_remaining_two_numbers_l105_105703


namespace range_of_m_l105_105804

theorem range_of_m (m : ℝ) :
  (¬(∀ x : ℝ, x^2 - m * x + 1 > 0 → -2 < m ∧ m < 2)) ∧
  (∃ x : ℝ, x^2 < 9 - m^2) ∧
  (-3 < m ∧ m < 3) →
  ((-3 < m ∧ m ≤ -2) ∨ (2 ≤ m ∧ m < 3)) :=
by sorry

end range_of_m_l105_105804


namespace count_three_digit_with_f_l105_105838

open Nat

def f : ℕ → ℕ := sorry 

axiom f_add_add (a b : ℕ) : f (a + b) = f (f a + b)
axiom f_add_small (a b : ℕ) (h : a + b < 10) : f (a + b) = f a + f b
axiom f_10 : f 10 = 1

theorem count_three_digit_with_f (hN : ∀ n : ℕ, f 2^(3^(4^5)) = f n):
  ∃ k, k = 100 ∧ ∀ n, 100 ≤ n ∧ n < 1000 → (f n = f 2^(3^(4^5))) :=
sorry

end count_three_digit_with_f_l105_105838


namespace train_meetings_between_stations_l105_105578

theorem train_meetings_between_stations
  (travel_time : ℕ := 3 * 60 + 30) -- Travel time in minutes
  (first_departure : ℕ := 6 * 60) -- First departure time in minutes from 0 (midnight)
  (departure_interval : ℕ := 60) -- Departure interval in minutes
  (A_departure_time : ℕ := 9 * 60) -- Departure time from Station A at 9:00 AM in minutes
  :
  ∃ n : ℕ, n = 7 :=
by
  sorry

end train_meetings_between_stations_l105_105578


namespace problem1_problem2_problem3_problem4_problem5_problem6_l105_105209

theorem problem1 : 78 * 4 + 488 = 800 := by sorry
theorem problem2 : 1903 - 475 * 4 = 3 := by sorry
theorem problem3 : 350 * (12 + 342 / 9) = 17500 := by sorry
theorem problem4 : 480 / (125 - 117) = 60 := by sorry
theorem problem5 : (3600 - 18 * 200) / 253 = 0 := by sorry
theorem problem6 : (243 - 162) / 27 * 380 = 1140 := by sorry

end problem1_problem2_problem3_problem4_problem5_problem6_l105_105209


namespace train_platform_length_l105_105603

theorem train_platform_length (train_length : ℕ) (platform_crossing_time : ℕ) (pole_crossing_time : ℕ) (length_of_platform : ℕ) :
  train_length = 300 →
  platform_crossing_time = 27 →
  pole_crossing_time = 18 →
  ((train_length * platform_crossing_time / pole_crossing_time) = train_length + length_of_platform) →
  length_of_platform = 150 :=
by
  intros h_train_length h_platform_time h_pole_time h_eq
  -- Proof omitted
  sorry

end train_platform_length_l105_105603


namespace triangle_inequality_cosine_rule_l105_105646

theorem triangle_inequality_cosine_rule (a b c : ℝ) (A B C : ℝ)
  (hA : Real.cos A = (b^2 + c^2 - a^2) / (2 * b * c))
  (hB : Real.cos B = (a^2 + c^2 - b^2) / (2 * a * c))
  (hC : Real.cos C = (a^2 + b^2 - c^2) / (2 * a * b)) :
  a^3 * Real.cos A + b^3 * Real.cos B + c^3 * Real.cos C ≤ (3 / 2) * a * b * c := 
sorry

end triangle_inequality_cosine_rule_l105_105646


namespace cube_volume_given_face_perimeter_l105_105081

-- Define the perimeter condition
def is_face_perimeter (perimeter : ℝ) (side_length : ℝ) : Prop :=
  4 * side_length = perimeter

-- Define volume computation
def cube_volume (side_length : ℝ) : ℝ :=
  side_length^3

-- Theorem stating the relationship between face perimeter and cube volume
theorem cube_volume_given_face_perimeter : 
  ∀ (side_length perimeter : ℝ), is_face_perimeter 40 side_length → cube_volume side_length = 1000 :=
by
  intros side_length perimeter h
  sorry

end cube_volume_given_face_perimeter_l105_105081


namespace bicycle_cost_correct_l105_105038

def pay_rate : ℕ := 5
def hours_p_week : ℕ := 2 + 1 + 3
def weeks : ℕ := 6
def bicycle_cost : ℕ := 180

theorem bicycle_cost_correct :
  pay_rate * hours_p_week * weeks = bicycle_cost :=
by
  sorry

end bicycle_cost_correct_l105_105038


namespace part_a_l105_105427

theorem part_a (a : ℤ) : (a^2 < 4) ↔ (a = -1 ∨ a = 0 ∨ a = 1) := 
sorry

end part_a_l105_105427


namespace total_assembly_time_l105_105677

def chairs := 2
def tables := 2
def bookshelf := 1
def tv_stand := 1

def time_per_chair := 8
def time_per_table := 12
def time_per_bookshelf := 25
def time_per_tv_stand := 35

theorem total_assembly_time : (chairs * time_per_chair) + (tables * time_per_table) + (bookshelf * time_per_bookshelf) + (tv_stand * time_per_tv_stand) = 100 := by
  sorry

end total_assembly_time_l105_105677


namespace every_nat_as_diff_of_same_prime_divisors_l105_105063

-- Conditions
def prime_divisors (n : ℕ) : ℕ :=
  -- function to count the number of distinct prime divisors of n
  sorry

-- Tuple translation
theorem every_nat_as_diff_of_same_prime_divisors :
  ∀ n : ℕ, ∃ a b : ℕ, n = a - b ∧ prime_divisors a = prime_divisors b := 
by
  sorry

end every_nat_as_diff_of_same_prime_divisors_l105_105063


namespace parabola_vertex_l105_105559

theorem parabola_vertex (x y : ℝ) : y^2 + 6*y + 2*x + 5 = 0 → (x, y) = (2, -3) :=
sorry

end parabola_vertex_l105_105559


namespace cost_per_adult_is_3_l105_105518

-- Define the number of people in the group
def total_people : ℕ := 12

-- Define the number of kids in the group
def kids : ℕ := 7

-- Define the total cost for the group
def total_cost : ℕ := 15

-- Define the number of adults, which is the total number of people minus the number of kids
def adults : ℕ := total_people - kids

-- Define the cost per adult meal, which is the total cost divided by the number of adults
noncomputable def cost_per_adult : ℕ := total_cost / adults

-- The theorem stating the cost per adult meal is $3
theorem cost_per_adult_is_3 : cost_per_adult = 3 :=
by
  -- The proof is skipped
  sorry

end cost_per_adult_is_3_l105_105518


namespace arithmetic_sequence_n_value_l105_105195

theorem arithmetic_sequence_n_value
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (n : ℕ)
  (hS9 : S 9 = 18)
  (ha_n_minus_4 : a (n-4) = 30)
  (hSn : S n = 336)
  (harithmetic_sequence : ∀ k, a (k + 1) - a k = a 2 - a 1) :
  n = 21 :=
sorry

end arithmetic_sequence_n_value_l105_105195


namespace elvie_age_l105_105898

variable (E : ℕ) (A : ℕ)

theorem elvie_age (hA : A = 11) (h : E + A + (E * A) = 131) : E = 10 :=
by
  sorry

end elvie_age_l105_105898


namespace quadratic_solution_l105_105212

theorem quadratic_solution (x : ℝ) (h1 : x^2 - 6 * x + 8 = 0) (h2 : x ≠ 0) :
  x = 2 ∨ x = 4 :=
sorry

end quadratic_solution_l105_105212


namespace prob_less_than_9_is_correct_l105_105895

-- Define the probabilities
def prob_ring_10 := 0.24
def prob_ring_9 := 0.28
def prob_ring_8 := 0.19

-- Define the condition for scoring less than 9, which does not include hitting the 10 or 9 ring.
def prob_less_than_9 := 1 - prob_ring_10 - prob_ring_9

-- Now we state the theorem we want to prove.
theorem prob_less_than_9_is_correct : prob_less_than_9 = 0.48 :=
by {
  -- Proof would go here
  sorry
}

end prob_less_than_9_is_correct_l105_105895


namespace school_club_profit_l105_105399

theorem school_club_profit : 
  let purchase_price_per_bar := 3 / 4
  let selling_price_per_bar := 2 / 3
  let total_bars := 1200
  let bars_with_discount := total_bars - 1000
  let discount_per_bar := 0.10
  let total_cost := total_bars * purchase_price_per_bar
  let total_revenue_without_discount := total_bars * selling_price_per_bar
  let total_discount := bars_with_discount * discount_per_bar
  let adjusted_revenue := total_revenue_without_discount - total_discount
  let profit := adjusted_revenue - total_cost
  profit = -116 :=
by sorry

end school_club_profit_l105_105399


namespace charging_time_l105_105222

theorem charging_time (S T L : ℕ → ℕ) 
  (HS : ∀ t, S t = 15 * t) 
  (HT : ∀ t, T t = 8 * t) 
  (HL : ∀ t, L t = 5 * t)
  (smartphone_capacity tablet_capacity laptop_capacity : ℕ)
  (smartphone_percentage tablet_percentage laptop_percentage : ℕ)
  (h_smartphone : smartphone_capacity = 4500)
  (h_tablet : tablet_capacity = 10000)
  (h_laptop : laptop_capacity = 20000)
  (p_smartphone : smartphone_percentage = 75)
  (p_tablet : tablet_percentage = 25)
  (p_laptop : laptop_percentage = 50)
  (required_charge_s required_charge_t required_charge_l : ℕ)
  (h_rq_s : required_charge_s = smartphone_capacity * smartphone_percentage / 100)
  (h_rq_t : required_charge_t = tablet_capacity * tablet_percentage / 100)
  (h_rq_l : required_charge_l = laptop_capacity * laptop_percentage / 100)
  (time_s time_t time_l : ℕ)
  (h_time_s : time_s = required_charge_s / 15)
  (h_time_t : time_t = required_charge_t / 8)
  (h_time_l : time_l = required_charge_l / 5) : 
  max time_s (max time_t time_l) = 2000 := 
by 
  -- This theorem states that the maximum time taken for charging is 2000 minutes
  sorry

end charging_time_l105_105222


namespace xiaoying_final_score_l105_105472

def speech_competition_score (score_content score_expression score_demeanor : ℕ) 
                             (weight_content weight_expression weight_demeanor : ℝ) : ℝ :=
  score_content * weight_content + score_expression * weight_expression + score_demeanor * weight_demeanor

theorem xiaoying_final_score :
  speech_competition_score 86 90 80 0.5 0.4 0.1 = 87 :=
by 
  sorry

end xiaoying_final_score_l105_105472


namespace smallest_date_for_first_Saturday_after_second_Monday_following_second_Thursday_l105_105781

theorem smallest_date_for_first_Saturday_after_second_Monday_following_second_Thursday :
  ∃ d : ℕ, d = 17 :=
by
  -- Assuming the starting condition that the month starts such that the second Thursday is on the 8th
  let second_thursday := 8

  -- Calculate second Monday after the second Thursday
  let second_monday := second_thursday + 4
  
  -- Calculate first Saturday after the second Monday
  let first_saturday := second_monday + 5

  have smallest_date : first_saturday = 17 := rfl
  
  exact ⟨first_saturday, smallest_date⟩

end smallest_date_for_first_Saturday_after_second_Monday_following_second_Thursday_l105_105781


namespace ratio_of_water_level_increase_l105_105634

noncomputable def volume_narrow_cone (h₁ : ℝ) : ℝ := (16 / 3) * Real.pi * h₁
noncomputable def volume_wide_cone (h₂ : ℝ) : ℝ := (64 / 3) * Real.pi * h₂
noncomputable def volume_marble_narrow : ℝ := (32 / 3) * Real.pi
noncomputable def volume_marble_wide : ℝ := (4 / 3) * Real.pi

theorem ratio_of_water_level_increase :
  ∀ (h₁ h₂ h₁' h₂' : ℝ),
  h₁ = 4 * h₂ →
  h₁' = h₁ + 2 →
  h₂' = h₂ + (1 / 16) →
  volume_narrow_cone h₁ = volume_wide_cone h₂ →
  volume_narrow_cone h₁ + volume_marble_narrow = volume_narrow_cone h₁' →
  volume_wide_cone h₂ + volume_marble_wide = volume_wide_cone h₂' →
  (h₁' - h₁) / (h₂' - h₂) = 32 :=
by
  intros h₁ h₂ h₁' h₂' h₁_eq_4h₂ h₁'_eq_h₁_add_2 h₂'_eq_h₂_add_1_div_16 vol_h₁_eq_vol_h₂ vol_nar_eq vol_wid_eq
  sorry

end ratio_of_water_level_increase_l105_105634


namespace lawn_chair_original_price_l105_105244

theorem lawn_chair_original_price (sale_price : ℝ) (discount_percentage : ℝ) (original_price : ℝ) :
  sale_price = 59.95 →
  discount_percentage = 23.09 →
  original_price = sale_price / (1 - discount_percentage / 100) →
  original_price = 77.95 :=
by sorry

end lawn_chair_original_price_l105_105244


namespace probability_of_winning_correct_l105_105432

noncomputable def probability_of_winning (P_L : ℚ) (P_T : ℚ) : ℚ :=
  1 - (P_L + P_T)

theorem probability_of_winning_correct :
  probability_of_winning (3/7) (2/21) = 10/21 :=
by
  sorry

end probability_of_winning_correct_l105_105432


namespace find_x_l105_105268

theorem find_x (x : ℝ) (h : (x + 8 + 5 * x + 4 + 2 * x + 7) / 3 = 3 * x - 10) : x = 49 :=
sorry

end find_x_l105_105268


namespace english_book_pages_l105_105278

def numPagesInOneEnglishBook (x y : ℕ) : Prop :=
  x = y + 12 ∧ 3 * x + 4 * y = 1275 → x = 189

-- The statement with sorry as no proof is required:
theorem english_book_pages (x y : ℕ) (h1 : x = y + 12) (h2 : 3 * x + 4 * y = 1275) : x = 189 :=
  sorry

end english_book_pages_l105_105278


namespace geometric_sequence_term_l105_105650

theorem geometric_sequence_term (a_n : ℕ → ℝ) (S_n : ℕ → ℝ) (n : ℕ) :
  (∀ n, S_n n = 3^n - 1) →
  (a_n n = S_n n - S_n (n - 1)) →
  (a_n n = 2 * 3^(n - 1)) :=
by
  intros h1 h2
  sorry

end geometric_sequence_term_l105_105650


namespace functional_equation_solution_l105_105178

theorem functional_equation_solution (f : ℚ → ℚ) :
  (∀ x y : ℚ, f (x + f y) = f x * f y) →
  (∀ x : ℚ, f x = 0 ∨ f x = 1) :=
by
  sorry

end functional_equation_solution_l105_105178


namespace simplify_expression_l105_105550

theorem simplify_expression :
  2 + 3 / (4 + 5 / (6 + 7 / 8)) = 137 / 52 :=
by
  sorry

end simplify_expression_l105_105550


namespace income_before_taxes_l105_105735

/-- Define given conditions -/
def net_income (x : ℝ) : ℝ := x - 0.10 * (x - 3000)

/-- Prove that the income before taxes must have been 13000 given the conditions. -/
theorem income_before_taxes (x : ℝ) (hx : net_income x = 12000) : x = 13000 :=
by sorry

end income_before_taxes_l105_105735


namespace new_concentration_of_solution_l105_105496

theorem new_concentration_of_solution 
  (Q : ℚ) 
  (initial_concentration : ℚ := 0.4) 
  (new_concentration : ℚ := 0.25) 
  (replacement_fraction : ℚ := 1/3) 
  (new_solution_concentration : ℚ := 0.35) :
  (initial_concentration * (1 - replacement_fraction) + new_concentration * replacement_fraction)
  = new_solution_concentration := 
by 
  sorry

end new_concentration_of_solution_l105_105496


namespace find_x_in_terms_of_abc_l105_105637

variable {x y z a b c : ℝ}

theorem find_x_in_terms_of_abc
  (h1 : xy / (x + y + 1) = a)
  (h2 : xz / (x + z + 1) = b)
  (h3 : yz / (y + z + 1) = c) :
  x = 2 * a * b * c / (a * b + a * c - b * c) := 
sorry

end find_x_in_terms_of_abc_l105_105637


namespace range_of_x_satisfying_inequality_l105_105013

def f (x : ℝ) : ℝ := -- Define the function f (we will leave this definition open for now)
sorry
@[continuity] axiom f_increasing (x y : ℝ) (h : x < y) : f x < f y
axiom f_2_eq_1 : f 2 = 1
axiom f_xy_eq_f_x_add_f_y (x y : ℝ) : f (x * y) = f x + f y

noncomputable def f_4_eq_2 : f 4 = 2 := sorry

theorem range_of_x_satisfying_inequality (x : ℝ) :
  3 < x ∧ x ≤ 4 ↔ f x + f (x - 3) ≤ 2 :=
sorry

end range_of_x_satisfying_inequality_l105_105013


namespace person_age_l105_105138

theorem person_age (x : ℕ) (h : 4 * (x + 3) - 4 * (x - 3) = x) : x = 24 :=
by {
  sorry
}

end person_age_l105_105138


namespace domain_of_f_l105_105164

noncomputable def f (x : ℝ) : ℝ := 1 / (Real.log (x + 1)) + Real.sqrt (4 - x^2)

theorem domain_of_f :
  {x : ℝ | x + 1 > 0 ∧ Real.log (x + 1) ≠ 0 ∧ 4 - x^2 ≥ 0} =
  {x : ℝ | -1 < x ∧ x < 0} ∪ {x : ℝ | 0 < x ∧ x ≤ 2} :=
by
  sorry

end domain_of_f_l105_105164


namespace problem1_l105_105533

theorem problem1 (x y : ℤ) (h : |x + 2| + |y - 3| = 0) : x - y + 1 = -4 :=
sorry

end problem1_l105_105533


namespace cost_of_shirts_l105_105196

theorem cost_of_shirts : 
  let shirt1 := 15
  let shirt2 := 15
  let shirt3 := 15
  let shirt4 := 20
  let shirt5 := 20
  shirt1 + shirt2 + shirt3 + shirt4 + shirt5 = 85 := 
by
  sorry

end cost_of_shirts_l105_105196


namespace percent_counties_l105_105939

def p1 : ℕ := 21
def p2 : ℕ := 44
def p3 : ℕ := 18

theorem percent_counties (h1 : p1 = 21) (h2 : p2 = 44) (h3 : p3 = 18) : p1 + p2 + p3 = 83 :=
by sorry

end percent_counties_l105_105939


namespace min_value_x2_y2_l105_105457

theorem min_value_x2_y2 (x y : ℝ) (h : x + y = 2) : ∃ m, m = x^2 + y^2 ∧ (∀ (x y : ℝ), x + y = 2 → x^2 + y^2 ≥ m) ∧ m = 2 := 
sorry

end min_value_x2_y2_l105_105457


namespace part1_part2_l105_105986

def op (a b : ℤ) := 2 * a - 3 * b

theorem part1 : op (-2) 3 = -13 := 
by
  -- Proof omitted
  sorry

theorem part2 (x : ℤ) : 
  let A := op (3 * x - 2) (x + 1)
  let B := op (-3 / 2 * x + 1) (-1 - 2 * x)
  B > A :=
by
  -- Proof omitted
  sorry

end part1_part2_l105_105986


namespace value_of_a_l105_105243

theorem value_of_a
  (x y a : ℝ)
  (h1 : x + 2 * y = 2 * a - 1)
  (h2 : x - y = 6)
  (h3 : x = -y)
  : a = -1 :=
by
  sorry

end value_of_a_l105_105243


namespace inequality_proof_l105_105318

variable {a b c : ℝ}

theorem inequality_proof (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_eq : a^2 + b^2 + c^2 = 1) : 
  a + b + Real.sqrt 2 * c ≤ 2 := 
by 
  sorry

end inequality_proof_l105_105318


namespace solve_2019_gon_l105_105143

noncomputable def problem_2019_gon (x : ℕ → ℕ) : Prop :=
  (∀ i : ℕ, (x i + x (i+1) + x (i+2) + x (i+3) + x (i+4) + x (i+5) + x (i+6) + x (i+7) + x (i+8) = 300))
  ∧ (x 18 = 19)
  ∧ (x 19 = 20)

theorem solve_2019_gon :
  ∀ x : ℕ → ℕ,
  problem_2019_gon x →
  x 2018 = 61 :=
by sorry

end solve_2019_gon_l105_105143


namespace range_of_alpha_minus_beta_l105_105756

open Real

theorem range_of_alpha_minus_beta (
    α β : ℝ) 
    (h1 : -π / 2 < α) 
    (h2 : α < 0)
    (h3 : 0 < β)
    (h4 : β < π / 3)
  : -5 * π / 6 < α - β ∧ α - β < 0 :=
by
  sorry

end range_of_alpha_minus_beta_l105_105756


namespace arithmetic_mean_of_normal_distribution_l105_105571

theorem arithmetic_mean_of_normal_distribution
  (σ : ℝ) (hσ : σ = 1.5)
  (value : ℝ) (hvalue : value = 11.5)
  (hsd : value = μ - 2 * σ) :
  μ = 14.5 :=
by
  sorry

end arithmetic_mean_of_normal_distribution_l105_105571


namespace distance_points_3_12_and_10_0_l105_105483

theorem distance_points_3_12_and_10_0 : 
  Real.sqrt ((10 - 3)^2 + (0 - 12)^2) = Real.sqrt 193 := 
by
  sorry

end distance_points_3_12_and_10_0_l105_105483


namespace intersection_a_four_range_of_a_l105_105822

variable {x a : ℝ}

-- Problem 1: Intersection of A and B for a = 4
def A (a : ℝ) : Set ℝ := {x | (x - 2) * (x - 2*a - 5) < 0}
def B (a : ℝ) : Set ℝ := {x | 2*a < x ∧ x < a^2 + 2}

theorem intersection_a_four : A 4 ∩ B 4 = {x | 8 < x ∧ x < 13} := 
by  sorry

-- Problem 2: Range of a given condition
theorem range_of_a (a : ℝ) (h1 : a > -3/2) (h2 : ∀ x ∈ A a, x ∈ B a) : 1 ≤ a ∧ a ≤ 3 := 
by  sorry

end intersection_a_four_range_of_a_l105_105822


namespace arithmetic_progression_sum_at_least_66_l105_105144

-- Define the sum of the first n terms of an arithmetic progression
def sum_first_n_terms (a d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a + (n - 1) * d) / 2

-- Define the conditions for the arithmetic progression
def arithmetic_prog_conditions (a1 d : ℤ) (n : ℕ) :=
  sum_first_n_terms a1 d n ≥ 66

-- The main theorem to prove
theorem arithmetic_progression_sum_at_least_66 (n : ℕ) :
  (n >= 3 ∧ n <= 14) → arithmetic_prog_conditions 25 (-3) n :=
by
  sorry

end arithmetic_progression_sum_at_least_66_l105_105144


namespace correct_statement_l105_105887

def angle_terminal_side (a b : ℝ) : Prop :=
∃ k : ℤ, a = b + k * 360

def obtuse_angle (θ : ℝ) : Prop :=
90 < θ ∧ θ < 180

def third_quadrant_angle (θ : ℝ) : Prop :=
180 < θ ∧ θ < 270

def first_quadrant_angle (θ : ℝ) : Prop :=
0 < θ ∧ θ < 90

def acute_angle (θ : ℝ) : Prop :=
0 < θ ∧ θ < 90

theorem correct_statement :
  ¬∀ a b, angle_terminal_side a b → a = b ∧
  ¬∀ θ, obtuse_angle θ → θ < θ - 360 ∧
  ¬∀ θ, first_quadrant_angle θ → acute_angle θ ∧
  ∀ θ, acute_angle θ → first_quadrant_angle θ :=
by
  sorry

end correct_statement_l105_105887


namespace solution_set_of_inequality_l105_105999

theorem solution_set_of_inequality (x : ℝ) : x^2 > x ↔ x < 0 ∨ 1 < x := 
by
  sorry

end solution_set_of_inequality_l105_105999


namespace suitcase_weight_on_return_l105_105856

def initial_weight : ℝ := 5
def perfume_count : ℝ := 5
def perfume_weight_oz : ℝ := 1.2
def chocolate_weight_lb : ℝ := 4
def soap_count : ℝ := 2
def soap_weight_oz : ℝ := 5
def jam_count : ℝ := 2
def jam_weight_oz : ℝ := 8
def oz_per_lb : ℝ := 16

theorem suitcase_weight_on_return :
  initial_weight + (perfume_count * perfume_weight_oz / oz_per_lb) + chocolate_weight_lb +
  (soap_count * soap_weight_oz / oz_per_lb) + (jam_count * jam_weight_oz / oz_per_lb) = 11 := 
  by
  sorry

end suitcase_weight_on_return_l105_105856


namespace compute_expression_l105_105931

theorem compute_expression : (88 * 707 - 38 * 707) / 1414 = 25 :=
by
  sorry

end compute_expression_l105_105931


namespace find_a_l105_105460

theorem find_a (a : ℕ) : 
  (a >= 100 ∧ a <= 999) ∧ 7 ∣ (504000 + a) ∧ 9 ∣ (504000 + a) ∧ 11 ∣ (504000 + a) ↔ a = 711 :=
by {
  sorry
}

end find_a_l105_105460


namespace abc_le_one_ninth_sum_ratios_le_one_over_two_sqrt_abc_l105_105094

open Real

variables (a b c : ℝ)

-- Condition: a, b, c are positive numbers
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
-- Condition: a^(3/2) + b^(3/2) + c^(3/2) = 1
variable  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1)

-- Question 1: Prove abc ≤ 1/9
theorem abc_le_one_ninth : a * b * c ≤ 1 / 9 :=
  sorry

-- Question 2: Prove (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt (a * b * c))
theorem sum_ratios_le_one_over_two_sqrt_abc : 
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt (a * b * c)) :=
  sorry

end abc_le_one_ninth_sum_ratios_le_one_over_two_sqrt_abc_l105_105094


namespace solve_system_l105_105488

theorem solve_system (a b c : ℝ) (h₁ : a^2 + 3 * a + 1 = (b + c) / 2)
                                (h₂ : b^2 + 3 * b + 1 = (a + c) / 2)
                                (h₃ : c^2 + 3 * c + 1 = (a + b) / 2) : 
  a = -1 ∧ b = -1 ∧ c = -1 :=
by
  sorry

end solve_system_l105_105488


namespace program_output_l105_105569

-- Define the initial conditions
def initial_a := 1
def initial_b := 3

-- Define the program transformations
def a_step1 (a b : ℕ) := a + b
def b_step2 (a b : ℕ) := a - b

-- Define the final values after program execution
def final_a := a_step1 initial_a initial_b
def final_b := b_step2 final_a initial_b

-- Statement to prove
theorem program_output :
  final_a = 4 ∧ final_b = 1 :=
by
  -- Placeholder for the actual proof
  sorry

end program_output_l105_105569


namespace combinedHeightOfBuildingsIsCorrect_l105_105057

-- Define the heights to the top floor of the buildings (in feet)
def empireStateBuildingHeightFeet : Float := 1250
def willisTowerHeightFeet : Float := 1450
def oneWorldTradeCenterHeightFeet : Float := 1368

-- Define the antenna heights of the buildings (in feet)
def empireStateBuildingAntennaFeet : Float := 204
def willisTowerAntennaFeet : Float := 280
def oneWorldTradeCenterAntennaFeet : Float := 408

-- Define the conversion factor from feet to meters
def feetToMeters : Float := 0.3048

-- Calculate the total heights of the buildings in meters
def empireStateBuildingTotalHeightMeters : Float := (empireStateBuildingHeightFeet + empireStateBuildingAntennaFeet) * feetToMeters
def willisTowerTotalHeightMeters : Float := (willisTowerHeightFeet + willisTowerAntennaFeet) * feetToMeters
def oneWorldTradeCenterTotalHeightMeters : Float := (oneWorldTradeCenterHeightFeet + oneWorldTradeCenterAntennaFeet) * feetToMeters

-- Calculate the combined total height of the three buildings in meters
def combinedTotalHeightMeters : Float :=
  empireStateBuildingTotalHeightMeters + willisTowerTotalHeightMeters + oneWorldTradeCenterTotalHeightMeters

-- The statement to prove
theorem combinedHeightOfBuildingsIsCorrect : combinedTotalHeightMeters = 1511.8164 := by
  sorry

end combinedHeightOfBuildingsIsCorrect_l105_105057


namespace target_water_percentage_is_two_percent_l105_105974

variable (initial_milk_volume pure_milk_volume : ℕ)
variable (initial_water_percentage target_water_percentage : ℚ)

-- Conditions: Initial milk contains 5% water and we add 15 liters of pure milk
axiom initial_milk_condition : initial_milk_volume = 10
axiom pure_milk_condition : pure_milk_volume = 15
axiom initial_water_condition : initial_water_percentage = 5 / 100

-- Prove that target percentage of water in the milk is 2%
theorem target_water_percentage_is_two_percent :
  target_water_percentage = 2 / 100 := by
  sorry

end target_water_percentage_is_two_percent_l105_105974


namespace min_oranges_in_new_box_l105_105581

theorem min_oranges_in_new_box (m n : ℕ) (x : ℕ) (h1 : m + n ≤ 60) 
    (h2 : 59 * m = 60 * n + x) : x = 30 :=
sorry

end min_oranges_in_new_box_l105_105581


namespace bacteria_seventh_generation_l105_105429

/-- Represents the effective multiplication factor per generation --/
def effective_mult_factor : ℕ := 4

/-- The number of bacteria in the first generation --/
def first_generation : ℕ := 1

/-- A helper function to compute the number of bacteria in the nth generation --/
def bacteria_count (n : ℕ) : ℕ :=
  first_generation * effective_mult_factor ^ n

/-- The number of bacteria in the seventh generation --/
theorem bacteria_seventh_generation : bacteria_count 7 = 4096 := by
  sorry

end bacteria_seventh_generation_l105_105429


namespace ravish_maximum_marks_l105_105305

theorem ravish_maximum_marks (M : ℝ) (h_pass : 0.40 * M = 80) : M = 200 :=
sorry

end ravish_maximum_marks_l105_105305


namespace floor_S_value_l105_105845

noncomputable def floor_S (a b c d : ℝ) : ℝ :=
  a + b + c + d

theorem floor_S_value (a b c d : ℝ) 
  (ha_pos : 0 < a) (hb_pos : 0 < b) (hc_pos : 0 < c) (hd_pos : 0 < d)
  (h_sum_sq : a^2 + b^2 = 2016 ∧ c^2 + d^2 = 2016)
  (h_product : a * c = 1008 ∧ b * d = 1008) :
  ⌊floor_S a b c d⌋ = 117 :=
by
  sorry

end floor_S_value_l105_105845


namespace value_of_y_l105_105341

variable {x y : ℝ}

theorem value_of_y (h1 : x > 2) (h2 : y > 2) (h3 : 1/x + 1/y = 3/4) (h4 : x * y = 8) : y = 4 :=
sorry

end value_of_y_l105_105341


namespace trigonometric_signs_l105_105639

noncomputable def terminal_side (θ α : ℝ) : Prop :=
  ∃ k : ℤ, θ = α + 2 * k * Real.pi

theorem trigonometric_signs :
  ∀ (α θ : ℝ), 
    (∃ k : ℤ, α = 2 * k * Real.pi - Real.pi / 5) ∧ terminal_side θ α →
    (Real.sin θ < 0) ∧ (Real.cos θ > 0) ∧ (Real.tan θ < 0) →
    (Real.sin θ / abs (Real.sin θ) + Real.cos θ / abs (Real.cos θ) + Real.tan θ / abs (Real.tan θ) = -1) :=
by intros
   sorry

end trigonometric_signs_l105_105639


namespace solution_set_ineq_l105_105925

open Set

theorem solution_set_ineq (a x : ℝ) (h : 0 < a ∧ a < 1) : 
 (a < x ∧ x < 1/a) ↔ ((x - a) * (x - 1/a) > 0) :=
by
  sorry

end solution_set_ineq_l105_105925


namespace virginia_taught_fewer_years_l105_105994

-- Definitions based on conditions
variable (V A D : ℕ)

-- Dennis has taught for 34 years
axiom h1 : D = 34

-- Virginia has taught for 9 more years than Adrienne
axiom h2 : V = A + 9

-- Combined total of years taught is 75
axiom h3 : V + A + D = 75

-- Proof statement: Virginia has taught for 9 fewer years than Dennis
theorem virginia_taught_fewer_years : D - V = 9 :=
  sorry

end virginia_taught_fewer_years_l105_105994


namespace total_contribution_is_1040_l105_105076

-- Definitions of contributions based on conditions.
def Niraj_contribution : ℕ := 80
def Brittany_contribution : ℕ := 3 * Niraj_contribution
def Angela_contribution : ℕ := 3 * Brittany_contribution

-- Statement to prove that total contribution is $1040.
theorem total_contribution_is_1040 : Niraj_contribution + Brittany_contribution + Angela_contribution = 1040 := by
  sorry

end total_contribution_is_1040_l105_105076


namespace trip_to_office_duration_l105_105644

noncomputable def distance (D : ℝ) : Prop :=
  let T1 := D / 58
  let T2 := D / 62
  T1 + T2 = 3

theorem trip_to_office_duration (D : ℝ) (h : distance D) : D / 58 = 1.55 :=
by sorry

end trip_to_office_duration_l105_105644


namespace annieka_free_throws_l105_105172

theorem annieka_free_throws (deshawn_throws : ℕ) (kayla_factor : ℝ) (annieka_diff : ℕ) (ht1 : deshawn_throws = 12) (ht2 : kayla_factor = 1.5) (ht3 : annieka_diff = 4) :
  ∃ (annieka_throws : ℕ), annieka_throws = (⌊deshawn_throws * kayla_factor⌋.toNat - annieka_diff) :=
by
  sorry

end annieka_free_throws_l105_105172


namespace smallest_tournament_with_ordered_group_l105_105736

-- Define the concept of a tennis tournament with n players
def tennis_tournament (n : ℕ) := 
  ∀ (i j : ℕ), (i < n) → (j < n) → (i ≠ j) → (i < j) ∨ (j < i)

-- Define what it means for a group of four players to be "ordered"
def ordered_group (p1 p2 p3 p4 : ℕ) : Prop := 
  ∃ (winner : ℕ), ∃ (loser : ℕ), 
    (winner ≠ loser) ∧ (winner = p1 ∨ winner = p2 ∨ winner = p3 ∨ winner = p4) ∧ 
    (loser = p1 ∨ loser = p2 ∨ loser = p3 ∨ loser = p4)

-- Prove that any tennis tournament with 8 players has an ordered group
theorem smallest_tournament_with_ordered_group : 
  ∀ (n : ℕ), ∀ (tournament : tennis_tournament n), 
    (n ≥ 8) → 
    (∃ (p1 p2 p3 p4 : ℕ), ordered_group p1 p2 p3 p4) :=
  by
  -- proof omitted
  sorry

end smallest_tournament_with_ordered_group_l105_105736


namespace part_a_part_b_l105_105821

-- Definition for combination
def combination (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Proof problems as Lean statements
theorem part_a : combination 30 2 = 435 := by
  sorry

theorem part_b : combination 30 3 = 4060 := by
  sorry

end part_a_part_b_l105_105821


namespace spending_on_other_items_is_30_percent_l105_105902

-- Define the total amount Jill spent excluding taxes
variable (T : ℝ)

-- Define the amounts spent on clothing, food, and other items as percentages of T
def clothing_spending : ℝ := 0.50 * T
def food_spending : ℝ := 0.20 * T
def other_items_spending (x : ℝ) : ℝ := x * T

-- Define the tax rates
def clothing_tax_rate : ℝ := 0.04
def food_tax_rate : ℝ := 0.0
def other_items_tax_rate : ℝ := 0.10

-- Define the taxes paid on each category
def clothing_tax : ℝ := clothing_tax_rate * clothing_spending T
def food_tax : ℝ := food_tax_rate * food_spending T
def other_items_tax (x : ℝ) : ℝ := other_items_tax_rate * other_items_spending T x

-- Define the total tax paid as a percentage of the total amount spent excluding taxes
def total_tax_paid : ℝ := 0.05 * T

-- The main theorem stating that the percentage of the amount spent on other items is 30%
theorem spending_on_other_items_is_30_percent (x : ℝ) (h : total_tax_paid T = clothing_tax T + other_items_tax T x) :
  x = 0.30 :=
sorry

end spending_on_other_items_is_30_percent_l105_105902


namespace smallest_five_digit_equiv_11_mod_13_l105_105461

open Nat

theorem smallest_five_digit_equiv_11_mod_13 :
  ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ n % 13 = 11 ∧ n = 10009 :=
by
  sorry

end smallest_five_digit_equiv_11_mod_13_l105_105461


namespace proof_problem_l105_105160

def operation1 (x : ℝ) := 9 - x
def operation2 (x : ℝ) := x - 9

theorem proof_problem : operation2 (operation1 15) = -15 := 
by
  sorry

end proof_problem_l105_105160


namespace equation_solution_l105_105490

theorem equation_solution (x : ℚ) (h₁ : (5 * x^2 + 4 * x + 2) / (x + 2) = 5 * x - 3) : x = 8 / 3 :=
by
  sorry

end equation_solution_l105_105490


namespace three_digit_multiples_of_24_l105_105996

theorem three_digit_multiples_of_24 : 
  let lower_bound := 100
  let upper_bound := 999
  let div_by := 24
  let first := lower_bound + (div_by - lower_bound % div_by) % div_by
  let last := upper_bound - (upper_bound % div_by)
  ∃ n : ℕ, (n + 1) = (last - first) / div_by + 1 := 
sorry

end three_digit_multiples_of_24_l105_105996


namespace cos_double_beta_alpha_plus_double_beta_l105_105324

variable (α β : ℝ)
variable (hα : 0 < α ∧ α < π / 2)
variable (hβ : 0 < β ∧ β < π / 2)
variable (h1 : Real.sin α = Real.sqrt 2 / 10)
variable (h2 : Real.sin β = Real.sqrt 10 / 10)

theorem cos_double_beta :
  Real.cos (2 * β) = 4 / 5 := by 
  sorry

theorem alpha_plus_double_beta :
  α + 2 * β = π / 4 := by 
  sorry

end cos_double_beta_alpha_plus_double_beta_l105_105324


namespace evaluate_expression_l105_105328

theorem evaluate_expression :
  (3^1003 + 7^1004)^2 - (3^1003 - 7^1004)^2 = 5.292 * 10^1003 :=
by sorry

end evaluate_expression_l105_105328


namespace macey_weeks_to_save_l105_105207

theorem macey_weeks_to_save :
  ∀ (total_cost amount_saved weekly_savings : ℝ),
    total_cost = 22.45 →
    amount_saved = 7.75 →
    weekly_savings = 1.35 →
    ⌈(total_cost - amount_saved) / weekly_savings⌉ = 11 :=
by
  intros total_cost amount_saved weekly_savings h_total_cost h_amount_saved h_weekly_savings
  sorry

end macey_weeks_to_save_l105_105207


namespace f_5times_8_eq_l105_105085

def f (x : ℚ) : ℚ := 1 / x ^ 2

theorem f_5times_8_eq :
  f (f (f (f (f (8 : ℚ))))) = 1 / 79228162514264337593543950336 := 
  by
    sorry

end f_5times_8_eq_l105_105085


namespace installation_rates_l105_105029

variables (units_total : ℕ) (teamA_units : ℕ) (teamB_units : ℕ) (team_units_gap : ℕ)
variables (rate_teamA : ℕ) (rate_teamB : ℕ)

-- Conditions
def conditions : Prop :=
  units_total = 140 ∧
  teamA_units = 80 ∧
  teamB_units = units_total - teamA_units ∧
  team_units_gap = 5 ∧
  rate_teamA = rate_teamB + team_units_gap

-- Question to prove
def solution : Prop :=
  rate_teamB = 15 ∧ rate_teamA = 20

-- Statement of the proof
theorem installation_rates (units_total : ℕ) (teamA_units : ℕ) (teamB_units : ℕ) (team_units_gap : ℕ) (rate_teamA : ℕ) (rate_teamB : ℕ) :
  conditions units_total teamA_units teamB_units team_units_gap rate_teamA rate_teamB →
  solution rate_teamA rate_teamB :=
sorry

end installation_rates_l105_105029


namespace necessary_but_not_sufficient_condition_l105_105878

theorem necessary_but_not_sufficient_condition (x : ℝ) (h : x < 5) : (x < 2 → x < 5) ∧ ¬(x < 5 → x < 2) :=
by
  sorry

end necessary_but_not_sufficient_condition_l105_105878


namespace line_through_center_of_circle_l105_105215

theorem line_through_center_of_circle 
    (x y : ℝ) 
    (h : x^2 + y^2 - 4*x + 6*y = 0) : 
    3*x + 2*y = 0 :=
sorry

end line_through_center_of_circle_l105_105215


namespace blue_pens_count_l105_105354

-- Definitions based on the conditions
def total_pens (B R : ℕ) : Prop := B + R = 82
def more_blue_pens (B R : ℕ) : Prop := B = R + 6

-- The theorem to prove
theorem blue_pens_count (B R : ℕ) (h1 : total_pens B R) (h2 : more_blue_pens B R) : B = 44 :=
by {
  -- This is where the proof steps would normally go.
  sorry
}

end blue_pens_count_l105_105354


namespace max_plates_l105_105875

def cost_pan : ℕ := 3
def cost_pot : ℕ := 5
def cost_plate : ℕ := 11
def total_cost : ℕ := 100
def min_pans : ℕ := 2
def min_pots : ℕ := 2

theorem max_plates (p q r : ℕ) :
  p >= min_pans → q >= min_pots → (cost_pan * p + cost_pot * q + cost_plate * r = total_cost) → r = 7 :=
by
  intros h_p h_q h_cost
  sorry

end max_plates_l105_105875


namespace calculate_price_per_pound_of_meat_l105_105350

noncomputable def price_per_pound_of_meat : ℝ :=
  let total_hours := 50
  let w := 8
  let m_pounds := 20
  let fv_pounds := 15
  let fv_pp := 4
  let b_pounds := 60
  let b_pp := 1.5
  let j_wage := 10
  let j_hours := 10
  let j_rate := 1.5

  -- known costs
  let fv_cost := fv_pounds * fv_pp
  let b_cost := b_pounds * b_pp
  let j_cost := j_hours * j_wage * j_rate

  -- total costs
  let total_cost := total_hours * w
  let known_costs := fv_cost + b_cost + j_cost

  (total_cost - known_costs) / m_pounds

theorem calculate_price_per_pound_of_meat : price_per_pound_of_meat = 5 := by
  sorry

end calculate_price_per_pound_of_meat_l105_105350


namespace determine_female_athletes_count_l105_105962

theorem determine_female_athletes_count (m : ℕ) (n : ℕ) (x y : ℕ) (probability : ℚ)
  (h_team : 56 + m = 56 + m) -- redundant, but setting up context
  (h_sample_size : n = 28)
  (h_probability : probability = 1 / 28)
  (h_sample_diff : x - y = 4)
  (h_sample_sum : x + y = n)
  (h_ratio : 56 * y = m * x) : m = 42 :=
by
  sorry

end determine_female_athletes_count_l105_105962


namespace isosceles_triangle_perimeter_l105_105565

theorem isosceles_triangle_perimeter (a b : ℝ) (h_iso : a = 4 ∨ b = 4) (h_iso2 : a = 8 ∨ b = 8) : 
  (a = 4 ∧ b = 8 ∧ 4 + a + b = 16 ∨ 
  a = 4 ∧ b = 8 ∧ b + a + a = 20 ∨ 
  a = 8 ∧ b = 4 ∧ a + a + b = 20) :=
by sorry

end isosceles_triangle_perimeter_l105_105565


namespace minimize_quadratic_l105_105261

theorem minimize_quadratic (x : ℝ) : (x = -9 / 2) → ∀ y : ℝ, y^2 + 9 * y + 7 ≥ (-9 / 2)^2 + 9 * -9 / 2 + 7 :=
by sorry

end minimize_quadratic_l105_105261


namespace wrongly_copied_value_l105_105784

theorem wrongly_copied_value (mean_initial mean_correct : ℕ) (n : ℕ) 
  (wrong_copied_value : ℕ) (total_sum_initial total_sum_correct : ℕ) : 
  (mean_initial = 150) ∧ (mean_correct = 151) ∧ (n = 30) ∧ 
  (wrong_copied_value = 135) ∧ (total_sum_initial = n * mean_initial) ∧ 
  (total_sum_correct = n * mean_correct) → 
  (total_sum_correct - (total_sum_initial - wrong_copied_value) + wrong_copied_value = 300) :=
by
  intros h
  have h1 : mean_initial = 150 := by sorry
  have h2 : mean_correct = 151 := by sorry
  have h3 : n = 30 := by sorry
  have h4 : wrong_copied_value = 135 := by sorry
  have h5 : total_sum_initial = n * mean_initial := by sorry
  have h6 : total_sum_correct = n * mean_correct := by sorry
  sorry -- This is where the proof would go, but is not required per instructions.

end wrongly_copied_value_l105_105784


namespace find_larger_number_l105_105173

theorem find_larger_number (L S : ℕ) (h1 : L - S = 1311) (h2 : L = 11 * S + 11) : L = 1441 :=
sorry

end find_larger_number_l105_105173


namespace sum_of_reciprocals_l105_105260

theorem sum_of_reciprocals {a b : ℕ} (h_sum: a + b = 55) (h_hcf: Nat.gcd a b = 5) (h_lcm: Nat.lcm a b = 120) :
  1 / (a : ℚ) + 1 / (b : ℚ) = 11 / 120 :=
by
  sorry

end sum_of_reciprocals_l105_105260


namespace triangle_inequality_l105_105995

variable (a b c R : ℝ)

-- Assuming a, b, c as the sides of a triangle
-- and R as the circumradius.

theorem triangle_inequality:
  (1 / (a * b)) + (1 / (b * c)) + (1 / (c * a)) ≥ (1 / (R * R)) :=
by
  sorry

end triangle_inequality_l105_105995


namespace rabbit_jump_lengths_order_l105_105317

theorem rabbit_jump_lengths_order :
  ∃ (R : ℕ) (G : ℕ) (P : ℕ) (F : ℕ),
    R = 2730 ∧
    R = P + 1100 ∧
    P = F + 150 ∧
    F = G - 200 ∧
    R > G ∧ G > P ∧ P > F :=
  by
  -- calculations
  sorry

end rabbit_jump_lengths_order_l105_105317


namespace largest_band_members_l105_105053

theorem largest_band_members
  (p q m : ℕ)
  (h1 : p * q + 3 = m)
  (h2 : (q + 1) * (p + 2) = m)
  (h3 : m < 120) :
  m = 119 :=
sorry

end largest_band_members_l105_105053


namespace expression_is_five_l105_105599

-- Define the expression
def given_expression : ℤ := abs (abs (-abs (-2 + 1) - 2) + 2)

-- Prove that the expression equals 5
theorem expression_is_five : given_expression = 5 :=
by
  -- We skip the proof for now
  sorry

end expression_is_five_l105_105599


namespace richard_remaining_distance_l105_105547

noncomputable def remaining_distance : ℝ :=
  let d1 := 45
  let d2 := d1 / 2 - 8
  let d3 := 2 * d2 - 4
  let d4 := (d1 + d2 + d3) / 3 + 3
  let d5 := 0.7 * d4
  let total_walked := d1 + d2 + d3 + d4 + d5
  635 - total_walked

theorem richard_remaining_distance : abs (remaining_distance - 497.5166) < 0.0001 :=
by
  sorry

end richard_remaining_distance_l105_105547


namespace range_of_a_l105_105694

def p (x : ℝ) : Prop := abs (2 * x - 1) ≤ 3

def q (x a : ℝ) : Prop := x^2 - (2*a + 1) * x + a*(a + 1) ≤ 0

theorem range_of_a : 
  (∀ x a, (¬ q x a) → (¬ p x))
  ∧ (∃ x a, (¬ q x a) ∧ (¬ p x))
  → (-1 : ℝ) ≤ a ∧ a ≤ (1 : ℝ) :=
sorry

end range_of_a_l105_105694


namespace largest_n_divisible_l105_105861

theorem largest_n_divisible (n : ℕ) : (n^3 + 150) % (n + 15) = 0 → n ≤ 2385 := by
  sorry

end largest_n_divisible_l105_105861


namespace rotten_pineapples_l105_105582

theorem rotten_pineapples (initial sold fresh remaining rotten: ℕ) 
  (h1: initial = 86) 
  (h2: sold = 48) 
  (h3: fresh = 29) 
  (h4: remaining = initial - sold) 
  (h5: rotten = remaining - fresh) : 
  rotten = 9 := by 
  sorry

end rotten_pineapples_l105_105582


namespace fill_cistern_time_l105_105810

-- Definitions based on conditions
def rate_A : ℚ := 1 / 8
def rate_B : ℚ := 1 / 16
def rate_C : ℚ := -1 / 12

-- Combined rate
def combined_rate : ℚ := rate_A + rate_B + rate_C

-- Time to fill the cistern
def time_to_fill := 1 / combined_rate

-- Lean statement of the proof
theorem fill_cistern_time : time_to_fill = 9.6 := by
  sorry

end fill_cistern_time_l105_105810


namespace parallel_lines_slope_condition_l105_105507

theorem parallel_lines_slope_condition (m : ℝ) :
  (∀ x y : ℝ, 2 * x + (m + 1) * y + 4 = 0 → mx + 3 * y - 2 = 0) →
  (m = 2 ∨ m = -3) :=
by
  sorry

end parallel_lines_slope_condition_l105_105507


namespace largest_multiple_of_45_l105_105103

theorem largest_multiple_of_45 (m : ℕ) 
  (h₁ : m % 45 = 0) 
  (h₂ : ∀ d : ℕ, d ∈ m.digits 10 → d = 8 ∨ d = 0) : 
  m / 45 = 197530 := 
sorry

end largest_multiple_of_45_l105_105103


namespace problem_1_problem_2_l105_105807

open Set

noncomputable def U : Set ℝ := univ
def A : Set ℝ := { x | -4 ≤ x ∧ x < 2 }
def B : Set ℝ := { x | -1 < x ∧ x ≤ 3 }
def P : Set ℝ := { x | x ≤ 0 ∨ x ≥ 5 / 2 }

theorem problem_1 : A ∩ B = { x | -1 < x ∧ x < 2 } :=
sorry

theorem problem_2 : (U \ B) ∪ P = { x | x ≤ 0 ∨ x ≥ 5 / 2 } :=
sorry

end problem_1_problem_2_l105_105807


namespace veranda_width_l105_105070

theorem veranda_width (l w : ℝ) (room_area veranda_area : ℝ) (h1 : l = 20) (h2 : w = 12) (h3 : veranda_area = 144) : 
  ∃ w_v : ℝ, (l + 2 * w_v) * (w + 2 * w_v) - l * w = veranda_area ∧ w_v = 2 := 
by
  sorry

end veranda_width_l105_105070


namespace arithmetic_mean_of_18_24_42_l105_105493

-- Define the numbers a, b, c
def a : ℕ := 18
def b : ℕ := 24
def c : ℕ := 42

-- Define the arithmetic mean
def mean (x y z : ℕ) : ℕ := (x + y + z) / 3

-- State the theorem to be proved
theorem arithmetic_mean_of_18_24_42 : mean a b c = 28 :=
by
  sorry

end arithmetic_mean_of_18_24_42_l105_105493


namespace batsman_average_after_12th_inning_l105_105126

variable (A : ℕ) (total_balls_faced : ℕ)

theorem batsman_average_after_12th_inning 
  (h1 : ∃ A, ∀ total_runs, total_runs = 11 * A)
  (h2 : ∃ A, ∀ total_runs_new, total_runs_new = 12 * (A + 4) ∧ total_runs_new - 60 = 11 * A)
  (h3 : 8 * 4 ≤ 60)
  (h4 : 6000 / total_balls_faced ≥ 130) 
  : (A + 4 = 16) :=
by
  sorry

end batsman_average_after_12th_inning_l105_105126


namespace find_q_l105_105557

open Real

noncomputable def q := (9 + 3 * Real.sqrt 5) / 2

theorem find_q (p q : ℝ) (hp : p > 1) (hq : q > 1) 
  (h1 : 1 / p + 1 / q = 1) (h2 : p * q = 9) : q = (9 + 3 * Real.sqrt 5) / 2 :=
by
  sorry

end find_q_l105_105557


namespace max_n_arithmetic_sequences_l105_105110

theorem max_n_arithmetic_sequences (a b : ℕ → ℤ) 
  (ha : ∀ n, a n = 1 + (n - 1) * 1)  -- Assuming x = 1 for simplicity, as per solution x = y = 1
  (hb : ∀ n, b n = 1 + (n - 1) * 1)  -- Assuming y = 1
  (a1 : a 1 = 1)
  (b1 : b 1 = 1)
  (a2_leq_b2 : a 2 ≤ b 2)
  (hn : ∃ n, a n * b n = 1764) :
  ∃ n, n = 44 ∧ a n * b n = 1764 :=
by
  sorry

end max_n_arithmetic_sequences_l105_105110


namespace common_ratio_is_0_88_second_term_is_475_2_l105_105869

-- Define the first term and the sum of the infinite geometric series
def first_term : Real := 540
def sum_infinite_series : Real := 4500

-- Required properties of the common ratio
def common_ratio (r : Real) : Prop :=
  abs r < 1 ∧ sum_infinite_series = first_term / (1 - r)

-- Prove the common ratio is 0.88 given the conditions
theorem common_ratio_is_0_88 : ∃ r : Real, common_ratio r ∧ r = 0.88 :=
by 
  sorry

-- Calculate the second term of the series
def second_term (r : Real) : Real := first_term * r

-- Prove the second term is 475.2 given the common ratio is 0.88
theorem second_term_is_475_2 : second_term 0.88 = 475.2 :=
by 
  sorry

end common_ratio_is_0_88_second_term_is_475_2_l105_105869


namespace students_with_one_talent_l105_105287

-- Define the given conditions
def total_students := 120
def cannot_sing := 30
def cannot_dance := 50
def both_skills := 10

-- Define the problem statement
theorem students_with_one_talent :
  (total_students - cannot_sing - both_skills) + (total_students - cannot_dance - both_skills) = 130 :=
by
  sorry

end students_with_one_talent_l105_105287


namespace find_a_l105_105938

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
noncomputable def c : ℝ := sorry

axiom cond1 : a^2 / b = 5
axiom cond2 : b^2 / c = 3
axiom cond3 : c^2 / a = 7

theorem find_a : a = 15 := sorry

end find_a_l105_105938


namespace divides_prime_factors_l105_105839

theorem divides_prime_factors (a b : ℕ) (p : ℕ → ℕ → Prop) (k l : ℕ → ℕ) (n : ℕ) : 
  (a ∣ b) ↔ (∀ i : ℕ, i < n → k i ≤ l i) :=
by
  sorry

end divides_prime_factors_l105_105839


namespace intersection_points_count_l105_105364

def line1 (x y : ℝ) : Prop := 4 * y - 3 * x = 2
def line2 (x y : ℝ) : Prop := 2 * x + 3 * y = 9
def line3 (x y : ℝ) : Prop := x - y = 1

theorem intersection_points_count :
  ∃ p1 p2 p3 : ℝ × ℝ,
  (line1 p1.1 p1.2 ∧ line2 p1.1 p1.2) ∧
  (line2 p2.1 p2.2 ∧ line3 p2.1 p2.2) ∧
  (line1 p3.1 p3.2 ∧ line3 p3.1 p3.2) ∧
  (p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3) :=
  sorry

end intersection_points_count_l105_105364


namespace solve_quadratic_eq_l105_105127

theorem solve_quadratic_eq (x : ℝ) : (x^2 + x - 1 = 0) ↔ (x = (-1 + Real.sqrt 5) / 2 ∨ x = (-1 - Real.sqrt 5) / 2) := by
  sorry

end solve_quadratic_eq_l105_105127


namespace f_bounds_l105_105682

noncomputable def f (x1 x2 x3 x4 : ℝ) := 1 - (x1^3 + x2^3 + x3^3 + x4^3) - 6 * (x1 * x2 * x3 + x1 * x2 * x4 + x1 * x3 * x4 + x2 * x3 * x4)

theorem f_bounds (x1 x2 x3 x4 : ℝ) (h : x1 + x2 + x3 + x4 = 1) :
  0 < f x1 x2 x3 x4 ∧ f x1 x2 x3 x4 ≤ 3 / 4 :=
by
  -- Proof steps go here
  sorry

end f_bounds_l105_105682


namespace expression_value_l105_105336

variables (p q r : ℝ)

theorem expression_value (h1 : p + q + r = 5) (h2 : (1 / (p + q)) + (1 / (q + r)) + (1 / (p + r)) = 9) :
  (r / (p + q)) + (p / (q + r)) + (q / (p + r)) = 42 := 
sorry

end expression_value_l105_105336


namespace problem_part1_problem_part2_l105_105495

-- Define the set A and the property it satisfies
variable (A : Set ℝ)
variable (H : ∀ a ∈ A, (1 + a) / (1 - a) ∈ A)

-- Suppose 2 is in A
theorem problem_part1 (h : 2 ∈ A) : A = {2, -3, -1 / 2, 1 / 3} :=
sorry

-- Prove the conjecture based on the elements of A found in part 1
theorem problem_part2 (h : 2 ∈ A) (hA : A = {2, -3, -1 / 2, 1 / 3}) :
  ¬ (0 ∈ A ∨ 1 ∈ A ∨ -1 ∈ A) ∧
  (2 * (-1 / 2) = -1 ∧ -3 * (1 / 3) = -1) :=
sorry

end problem_part1_problem_part2_l105_105495


namespace cylinder_height_relationship_l105_105614

theorem cylinder_height_relationship
  (r1 h1 r2 h2 : ℝ)
  (vol_eq : π * r1^2 * h1 = π * r2^2 * h2)
  (radius_rel : r2 = (6 / 5) * r1) : h1 = (36 / 25) * h2 := 
sorry

end cylinder_height_relationship_l105_105614


namespace range_of_f_l105_105679

noncomputable def f (x : ℝ) : ℝ := x + |x - 2|

theorem range_of_f : Set.range f = Set.Ici 2 :=
sorry

end range_of_f_l105_105679


namespace problem_solution_l105_105027

noncomputable def f (x : ℝ) : ℝ := x / (Real.cos x)

variables (x1 x2 x3 : ℝ)

axiom a1 : |x1| < (Real.pi / 2)
axiom a2 : |x2| < (Real.pi / 2)
axiom a3 : |x3| < (Real.pi / 2)

axiom h1 : f x1 + f x2 ≥ 0
axiom h2 : f x2 + f x3 ≥ 0
axiom h3 : f x3 + f x1 ≥ 0

theorem problem_solution : f (x1 + x2 + x3) ≥ 0 := sorry

end problem_solution_l105_105027


namespace sum_of_two_equal_sides_is_4_l105_105176

noncomputable def isosceles_right_triangle (a c : ℝ) : Prop :=
  c = 2.8284271247461903 ∧ c ^ 2 = 2 * (a ^ 2)

theorem sum_of_two_equal_sides_is_4 :
  ∃ a : ℝ, isosceles_right_triangle a 2.8284271247461903 ∧ 2 * a = 4 :=
by
  sorry

end sum_of_two_equal_sides_is_4_l105_105176


namespace smallest_integer_to_perfect_cube_l105_105294

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, k^3 = n

theorem smallest_integer_to_perfect_cube :
  ∃ n : ℕ, 
    n > 0 ∧ 
    is_perfect_cube (45216 * n) ∧ 
    (∀ m : ℕ, m > 0 ∧ is_perfect_cube (45216 * m) → n ≤ m) ∧ 
    n = 7 := sorry

end smallest_integer_to_perfect_cube_l105_105294


namespace least_number_to_subtract_l105_105567

theorem least_number_to_subtract (x : ℕ) (h : x = 1234567890) : ∃ n, x - n = 5 := 
  sorry

end least_number_to_subtract_l105_105567


namespace tall_students_proof_l105_105333

variables (T : ℕ) (Short Average Tall : ℕ)

-- Given in the problem:
def total_students := T = 400
def short_students := Short = 2 * T / 5
def average_height_students := Average = 150

-- Prove:
theorem tall_students_proof (hT : total_students T) (hShort : short_students T Short) (hAverage : average_height_students Average) :
  Tall = T - (Short + Average) :=
by
  sorry

end tall_students_proof_l105_105333


namespace value_of_expression_l105_105860

theorem value_of_expression (a b c d : ℝ) (h : a + b + c + d = 4) : 12 * a - 6 * b + 3 * c - 2 * d = 40 :=
by sorry

end value_of_expression_l105_105860


namespace min_correct_answers_l105_105698

/-- 
Given:
1. There are 25 questions in the preliminary round.
2. Scoring rules: 
   - 4 points for each correct answer,
   - -1 point for each incorrect or unanswered question.
3. A score of at least 60 points is required to advance to the next round.

Prove that the minimum number of correct answers needed to advance is 17.
-/
theorem min_correct_answers (x : ℕ) (h1 : 0 ≤ x) (h2 : x ≤ 25) (h3 : 4 * x - (25 - x) ≥ 60) : x ≥ 17 :=
sorry

end min_correct_answers_l105_105698


namespace cube_sum_l105_105789

theorem cube_sum (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 13) : x^3 + y^3 = 35 :=
by
  sorry

end cube_sum_l105_105789


namespace trig_expr_correct_l105_105361

noncomputable def trig_expr : ℝ := Real.sin (20 * Real.pi / 180) * Real.cos (10 * Real.pi / 180) - 
                                   Real.cos (160 * Real.pi / 180) * Real.sin (170 * Real.pi / 180)

theorem trig_expr_correct : trig_expr = 1 / 2 := 
  sorry

end trig_expr_correct_l105_105361


namespace seed_mixture_Y_is_25_percent_ryegrass_l105_105116

variables (X Y : ℝ) (R : ℝ)

def proportion_X_is_40_percent_ryegrass : Prop :=
  X = 40 / 100

def proportion_Y_contains_percent_ryegrass (R : ℝ) : Prop :=
  100 - R = 75 / 100 * 100

def mixture_contains_30_percent_ryegrass (X Y R : ℝ) : Prop :=
  (1/3) * (40 / 100) * 100 + (2/3) * (R / 100) * 100 = 30

def weight_of_mixture_is_33_percent_X (X Y : ℝ) : Prop :=
  X / (X + Y) = 1 / 3

theorem seed_mixture_Y_is_25_percent_ryegrass
  (X Y : ℝ) (R : ℝ) 
  (h1 : proportion_X_is_40_percent_ryegrass X)
  (h2 : proportion_Y_contains_percent_ryegrass R)
  (h3 : weight_of_mixture_is_33_percent_X X Y)
  (h4 : mixture_contains_30_percent_ryegrass X Y R) :
  R = 25 :=
sorry

end seed_mixture_Y_is_25_percent_ryegrass_l105_105116


namespace stickers_after_exchange_l105_105935

-- Given conditions
def Ryan_stickers : ℕ := 30
def Steven_stickers : ℕ := 3 * Ryan_stickers
def Terry_stickers : ℕ := Steven_stickers + 20
def Emily_stickers : ℕ := Steven_stickers / 2
def Jasmine_stickers : ℕ := Terry_stickers + Terry_stickers / 10

def total_stickers_before : ℕ := 
  Ryan_stickers + Steven_stickers + Terry_stickers + Emily_stickers + Jasmine_stickers

noncomputable def total_stickers_after : ℕ := 
  total_stickers_before - 2 * 5

-- The goal is to prove that the total stickers after the exchange event is 386
theorem stickers_after_exchange : total_stickers_after = 386 := 
  by sorry

end stickers_after_exchange_l105_105935


namespace minimum_value_of_f_l105_105210

noncomputable def f (x : ℝ) : ℝ := |2*x - 1| + |3*x - 2| + |4*x - 3| + |5*x - 4|

theorem minimum_value_of_f : ∃ x : ℝ, (∀ y : ℝ, f y ≥ 1) ∧ f x = 1 :=
by
  sorry

end minimum_value_of_f_l105_105210


namespace remainder_of_x_plus_2_pow_2022_l105_105455

theorem remainder_of_x_plus_2_pow_2022 (x : ℂ) :
  ∃ r : ℂ, ∃ q : ℂ, (x + 2)^2022 = q * (x^2 - x + 1) + r ∧ (r = x) :=
by
  sorry

end remainder_of_x_plus_2_pow_2022_l105_105455


namespace ratio_wheelbarrow_to_earnings_l105_105340

theorem ratio_wheelbarrow_to_earnings :
  let duck_price := 10
  let chicken_price := 8
  let chickens_sold := 5
  let ducks_sold := 2
  let resale_earn := 60
  let total_earnings := chickens_sold * chicken_price + ducks_sold * duck_price
  let wheelbarrow_cost := resale_earn / 2
  (wheelbarrow_cost / total_earnings = 1 / 2) :=
by
  sorry

end ratio_wheelbarrow_to_earnings_l105_105340


namespace fraction_sum_equals_l105_105636

theorem fraction_sum_equals :
  (1 / 20 : ℝ) + (2 / 10 : ℝ) + (4 / 40 : ℝ) = 0.35 :=
by
  sorry

end fraction_sum_equals_l105_105636


namespace increase_a1_intervals_of_increase_l105_105095

noncomputable def f (x a : ℝ) : ℝ := x - (a + 1) * Real.log x - a / x

-- Prove that when a = 1, f(x) has no extreme points (i.e., it is monotonically increasing in (0, +∞))
theorem increase_a1 : ∀ x : ℝ, 0 < x → f x 1 = x - 2 * Real.log x - 1 / x :=
sorry

-- Find the intervals of increase for f(x) = x - (a+1) ln x - a/x
theorem intervals_of_increase (a : ℝ) : 
  (a ≤ 0 → ∀ x : ℝ, 1 < x → 0 ≤ (f x a - f 1 a)) ∧ 
  (0 < a ∧ a < 1 → (∀ x : ℝ, 0 < x ∧ x < a → 0 ≤ f x a) ∧ ∀ x : ℝ, 1 < x → 0 ≤ f x a ) ∧ 
  (a = 1 → ∀ x : ℝ, 0 < x → 0 ≤ f x a) ∧ 
  (a > 1 → (∀ x : ℝ, 0 < x ∧ x < 1 → 0 ≤ f x a) ∧ ∀ x : ℝ, a < x → 0 ≤ f x a ) :=
sorry

end increase_a1_intervals_of_increase_l105_105095


namespace least_value_expr_l105_105120

   variable {x y : ℝ}

   theorem least_value_expr : ∃ x y : ℝ, (x^3 * y - 1)^2 + (x + y)^2 = 1 :=
   by
     sorry
   
end least_value_expr_l105_105120


namespace calculate_speed_l105_105314

-- Define the distance and time conditions
def distance : ℝ := 390
def time : ℝ := 4

-- Define the expected answer for speed
def expected_speed : ℝ := 97.5

-- Prove that speed equals expected_speed given the conditions
theorem calculate_speed : (distance / time) = expected_speed :=
by
  -- skipped proof steps
  sorry

end calculate_speed_l105_105314


namespace arithmetic_sequence_common_difference_l105_105009

theorem arithmetic_sequence_common_difference
  (a₁ d : ℝ)
  (h_a4 : a₁ + 3 * d = -2)
  (h_sum : 10 * a₁ + 45 * d = 65) :
  d = 17 / 3 :=
sorry

end arithmetic_sequence_common_difference_l105_105009


namespace min_sum_of_factors_l105_105008

theorem min_sum_of_factors (a b c : ℕ) (h1 : a * b * c = 2310) : a + b + c ≥ 42 :=
sorry

end min_sum_of_factors_l105_105008


namespace citric_acid_molecular_weight_l105_105238

def molecular_weight_citric_acid := 192.12 -- in g/mol

theorem citric_acid_molecular_weight :
  molecular_weight_citric_acid = 192.12 :=
by sorry

end citric_acid_molecular_weight_l105_105238


namespace ab_leq_one_l105_105924

theorem ab_leq_one (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : a + b = 2) : ab ≤ 1 := by
  sorry

end ab_leq_one_l105_105924


namespace solve_x_l105_105859

theorem solve_x (x : ℝ) (hx : (1/x + 1/(2*x) + 1/(3*x) = 1/12)) : x = 22 :=
  sorry

end solve_x_l105_105859


namespace abs_fraction_inequality_l105_105128

theorem abs_fraction_inequality (x : ℝ) :
  (abs ((3 * x - 4) / (x - 2)) > 3) ↔
  (x ∈ Set.Iio (5 / 3) ∪ Set.Ioo (5 / 3) 2 ∪ Set.Ioi 2) :=
by 
  sorry

end abs_fraction_inequality_l105_105128


namespace calculate_expression_l105_105374

theorem calculate_expression : 16^4 * 8^2 / 4^12 = (1 : ℚ) / 4 := by
  sorry

end calculate_expression_l105_105374


namespace red_paint_amount_l105_105852

theorem red_paint_amount (r w : ℕ) (hrw : r / w = 5 / 7) (hwhite : w = 21) : r = 15 :=
by {
  sorry
}

end red_paint_amount_l105_105852


namespace lulu_final_cash_l105_105378

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

end lulu_final_cash_l105_105378


namespace evaluate_expression_l105_105904

theorem evaluate_expression : (24^36 / 72^18) = 8^18 := by
  sorry

end evaluate_expression_l105_105904


namespace area_of_square_l105_105424

noncomputable def square_area (u v : ℝ) (h_u : 0 < u) (h_v : 0 < v) : ℝ :=
  (v * v) / 4

theorem area_of_square (u v : ℝ) (h_u : 0 < u) (h_v : 0 < v) (h_cond : ∀ (A B C : ℝ × ℝ), 
  A = (0, 0) → B = (u, 0) → C = (u, v) → 
  (u - 0) * (u - 0) + (v - 0) * (v - 0) = (u - 0) * (u - 0)) :
  square_area u v h_u h_v = v * v / 4 := 
by 
  sorry

end area_of_square_l105_105424


namespace total_quantities_l105_105627

theorem total_quantities (N : ℕ) (S S₃ S₂ : ℕ)
  (h1 : S = 12 * N)
  (h2 : S₃ = 12)
  (h3 : S₂ = 48)
  (h4 : S = S₃ + S₂) :
  N = 5 :=
by
  sorry

end total_quantities_l105_105627


namespace value_of_3_over_x_l105_105018

theorem value_of_3_over_x (x : ℝ) (hx : 1 - 6 / x + 9 / x^2 - 4 / x^3 = 0) : 
  (3 / x = 3 ∨ 3 / x = 3 / 4) :=
  sorry

end value_of_3_over_x_l105_105018


namespace swim_club_member_count_l105_105385

theorem swim_club_member_count :
  let total_members := 60
  let passed_percentage := 0.30
  let passed_members := total_members * passed_percentage
  let not_passed_members := total_members - passed_members
  let preparatory_course_members := 12
  not_passed_members - preparatory_course_members = 30 :=
by
  sorry

end swim_club_member_count_l105_105385


namespace train_B_speed_l105_105663

-- Given conditions
def speed_train_A := 70 -- km/h
def time_after_meet_A := 9 -- hours
def time_after_meet_B := 4 -- hours

-- Proof statement
theorem train_B_speed : 
  ∃ (V_b : ℕ),
    V_b * time_after_meet_B + V_b * s = speed_train_A * time_after_meet_A + speed_train_A * s ∧
    V_b = speed_train_A := 
sorry

end train_B_speed_l105_105663


namespace length_of_ae_l105_105219

-- Definitions for lengths of segments
variable {ab bc cd de ac ae : ℝ}

-- Given conditions as assumptions
axiom h1 : bc = 3 * cd
axiom h2 : de = 8
axiom h3 : ab = 5
axiom h4 : ac = 11

-- The main theorem to prove
theorem length_of_ae : ae = ab + bc + cd + de → bc = ac - ab → bc = 6 → cd = bc / 3 → ae = 21 :=
by sorry

end length_of_ae_l105_105219


namespace domain_of_f_l105_105829

noncomputable def f (x : ℝ) : ℝ := (3 * x^2) / Real.sqrt (1 - 2 * x) + Real.log (1 + 2 * x)

theorem domain_of_f : {x : ℝ | 1 - 2 * x > 0 ∧ 1 + 2 * x > 0} = {x : ℝ | -1 / 2 < x ∧ x < 1 / 2} :=
by
    sorry

end domain_of_f_l105_105829


namespace find_angle_4_l105_105241

def angle_sum_180 (α β : ℝ) : Prop := α + β = 180
def angle_equality (γ δ : ℝ) : Prop := γ = δ
def triangle_angle_values (A B : ℝ) : Prop := A = 80 ∧ B = 50

theorem find_angle_4
  (A B : ℝ) (angle1 angle2 angle3 angle4 : ℝ)
  (h1 : angle_sum_180 angle1 angle2)
  (h2 : angle_equality angle3 angle4)
  (h3 : triangle_angle_values A B)
  (h4 : angle_sum_180 (angle1 + A + B) 180)
  (h5 : angle_sum_180 (angle2 + angle3 + angle4) 180) :
  angle4 = 25 :=
by sorry

end find_angle_4_l105_105241


namespace dummies_remainder_l105_105901

/-
  Prove that if the number of Dummies in one bag is such that when divided among 10 kids, 3 pieces are left over,
  then the number of Dummies in four bags when divided among 10 kids leaves 2 pieces.
-/
theorem dummies_remainder (n : ℕ) (h : n % 10 = 3) : (4 * n) % 10 = 2 := 
by {
  sorry
}

end dummies_remainder_l105_105901


namespace triangle_area_base_6_height_8_l105_105233

noncomputable def triangle_area (base height : ℕ) : ℕ :=
  (base * height) / 2

theorem triangle_area_base_6_height_8 : triangle_area 6 8 = 24 := by
  sorry

end triangle_area_base_6_height_8_l105_105233


namespace two_star_neg_five_eq_neg_one_neg_two_star_two_star_neg_three_eq_one_l105_105213

def star (a b : ℤ) : ℤ := a ^ 2 - b + a * b

theorem two_star_neg_five_eq_neg_one : star 2 (-5) = -1 := by
  sorry

theorem neg_two_star_two_star_neg_three_eq_one : star (-2) (star 2 (-3)) = 1 := by
  sorry

end two_star_neg_five_eq_neg_one_neg_two_star_two_star_neg_three_eq_one_l105_105213


namespace log_equivalence_l105_105951

theorem log_equivalence :
  (Real.log 9 / Real.log 2) * (Real.log 4 / Real.log 3) = 4 :=
by
  sorry

end log_equivalence_l105_105951


namespace binary_101110_to_octal_l105_105906

-- Definition: binary number 101110 represents some decimal number
def binary_101110 : ℕ := 0 * 2^0 + 1 * 2^1 + 1 * 2^2 + 1 * 2^3 + 0 * 2^4 + 1 * 2^5

-- Definition: decimal number 46 represents some octal number
def decimal_46 := 46

-- A utility function to convert decimal to octal (returns the digits as a list)
def decimal_to_octal (n : ℕ) : List ℕ :=
  if n < 8 then [n]
  else decimal_to_octal (n / 8) ++ [n % 8]

-- Hypothesis: the binary 101110 equals the decimal 46
lemma binary_101110_eq_46 : binary_101110 = decimal_46 := by sorry

-- Hypothesis: the decimal 46 converts to the octal number 56 (in list form)
def octal_56 := [5, 6]

-- Theorem: binary 101110 converts to the octal number 56
theorem binary_101110_to_octal :
  decimal_to_octal binary_101110 = octal_56 := by
  rw [binary_101110_eq_46]
  sorry

end binary_101110_to_octal_l105_105906


namespace new_student_info_l105_105624

-- Definitions of the information pieces provided by each classmate.
structure StudentInfo where
  last_name : String
  gender : String
  total_score : Nat
  specialty : String

def student_A : StudentInfo := {
  last_name := "Ji",
  gender := "Male",
  total_score := 260,
  specialty := "Singing"
}

def student_B : StudentInfo := {
  last_name := "Zhang",
  gender := "Female",
  total_score := 220,
  specialty := "Dancing"
}

def student_C : StudentInfo := {
  last_name := "Chen",
  gender := "Male",
  total_score := 260,
  specialty := "Singing"
}

def student_D : StudentInfo := {
  last_name := "Huang",
  gender := "Female",
  total_score := 220,
  specialty := "Drawing"
}

def student_E : StudentInfo := {
  last_name := "Zhang",
  gender := "Female",
  total_score := 240,
  specialty := "Singing"
}

-- The theorem we need to prove based on the given conditions.
theorem new_student_info :
  ∃ info : StudentInfo,
    info.last_name = "Huang" ∧
    info.gender = "Male" ∧
    info.total_score = 240 ∧
    info.specialty = "Dancing" :=
  sorry

end new_student_info_l105_105624


namespace find_solution_pairs_l105_105651

theorem find_solution_pairs (m n : ℕ) (t : ℕ) (ht : t > 0) (hcond : 2 ≤ m ∧ 2 ≤ n ∧ n ∣ (1 + m^(3^n) + m^(2 * 3^n))) : 
  ∃ t : ℕ, t > 0 ∧ m = 3 * t - 2 ∧ n = 3 :=
by sorry

end find_solution_pairs_l105_105651


namespace cone_new_height_eq_sqrt_85_l105_105950

/-- A cone has a uniform circular base of radius 6 feet and a slant height of 13 feet.
    After the side breaks, the slant height reduces by 2 feet, making the new slant height 11 feet.
    We need to determine the new height from the base to the tip of the cone, and prove it is sqrt(85). -/
theorem cone_new_height_eq_sqrt_85 :
  let r : ℝ := 6
  let l : ℝ := 13
  let l' : ℝ := 11
  let h : ℝ := Real.sqrt (13^2 - 6^2)
  let H : ℝ := Real.sqrt (11^2 - 6^2)
  H = Real.sqrt 85 :=
by
  sorry


end cone_new_height_eq_sqrt_85_l105_105950


namespace product_bc_l105_105250

theorem product_bc (b c : ℤ)
    (h1 : ∀ s : ℤ, s^2 = 2 * s + 1 → s^6 - b * s - c = 0) :
    b * c = 2030 :=
sorry

end product_bc_l105_105250


namespace range_of_fraction_l105_105431

theorem range_of_fraction (x1 y1 : ℝ) (h1 : y1 = -2 * x1 + 8) (h2 : 2 ≤ x1 ∧ x1 ≤ 5) :
  -1/6 ≤ (y1 + 1) / (x1 + 1) ∧ (y1 + 1) / (x1 + 1) ≤ 5/3 :=
sorry

end range_of_fraction_l105_105431


namespace small_z_value_l105_105943

noncomputable def w (n : ℕ) := n
noncomputable def x (n : ℕ) := n + 1
noncomputable def y (n : ℕ) := n + 2
noncomputable def z (n : ℕ) := n + 4

theorem small_z_value (n : ℕ) 
  (h : w n ^ 3 + x n ^ 3 + y n ^ 3 = z n ^ 3)
  : z n = 9 :=
sorry

end small_z_value_l105_105943


namespace correct_statements_l105_105279

theorem correct_statements :
  (20 / 100 * 40 = 8) ∧
  (2^3 = 8) ∧
  (7 - 3 * 2 ≠ 8) ∧
  (3^2 - 1^2 = 8) ∧
  (2 * (6 - 4)^2 = 8) :=
by
  sorry

end correct_statements_l105_105279


namespace monday_has_greatest_temp_range_l105_105454

-- Define the temperatures
def high_temp (day : String) : Int :=
  if day = "Monday" then 6 else
  if day = "Tuesday" then 3 else
  if day = "Wednesday" then 4 else
  if day = "Thursday" then 4 else
  if day = "Friday" then 8 else 0

def low_temp (day : String) : Int :=
  if day = "Monday" then -4 else
  if day = "Tuesday" then -6 else
  if day = "Wednesday" then -2 else
  if day = "Thursday" then -5 else
  if day = "Friday" then 0 else 0

-- Define the temperature range for a given day
def temp_range (day : String) : Int :=
  high_temp day - low_temp day

-- Statement to prove: Monday has the greatest temperature range
theorem monday_has_greatest_temp_range : 
  temp_range "Monday" > temp_range "Tuesday" ∧
  temp_range "Monday" > temp_range "Wednesday" ∧
  temp_range "Monday" > temp_range "Thursday" ∧
  temp_range "Monday" > temp_range "Friday" := 
sorry

end monday_has_greatest_temp_range_l105_105454


namespace arithmetic_sequence_sum_l105_105417

theorem arithmetic_sequence_sum
  (a : ℕ → ℕ)
  (S : ℕ → ℕ)
  (h_arith : ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0)
  (h_S_def : ∀ n : ℕ, S n = n * (a 0 + a (n - 1)) / 2)
  (h_S2 : S 2 = 4)
  (h_S4 : S 4 = 16) :
  a 5 + a 6 = 20 :=
sorry

end arithmetic_sequence_sum_l105_105417


namespace a3_is_neg_10_a1_a3_a5_sum_is_neg_16_l105_105015

variable (a0 a1 a2 a3 a4 a5 : ℝ)

noncomputable def polynomial (x : ℝ) : ℝ :=
  a0 + a1 * (1 - x) + a2 * (1 - x)^2 + a3 * (1 - x)^3 + a4 * (1 - x)^4 + a5 * (1 - x)^5

theorem a3_is_neg_10 (h : ∀ x, x^5 = polynomial a0 a1 a2 a3 a4 a5 x) : a3 = -10 :=
sorry

theorem a1_a3_a5_sum_is_neg_16 (h : ∀ x, x^5 = polynomial a0 a1 a2 a3 a4 a5 x) : a1 + a3 + a5 = -16 :=
sorry

end a3_is_neg_10_a1_a3_a5_sum_is_neg_16_l105_105015


namespace max_area_equilateral_triangle_in_rectangle_l105_105918

-- Define the problem parameters
def rect_width : ℝ := 12
def rect_height : ℝ := 15

-- State the theorem to be proved
theorem max_area_equilateral_triangle_in_rectangle 
  (width height : ℝ) (h_width : width = rect_width) (h_height : height = rect_height) :
  ∃ area : ℝ, area = 369 * Real.sqrt 3 - 540 := 
sorry

end max_area_equilateral_triangle_in_rectangle_l105_105918


namespace batches_engine_count_l105_105056

theorem batches_engine_count (x : ℕ) 
  (h1 : ∀ e, 1/4 * e = 0) -- every batch has engines, no proof needed for this question
  (h2 : 5 * (3/4 : ℚ) * x = 300) : 
  x = 80 := 
sorry

end batches_engine_count_l105_105056


namespace page_shoes_l105_105343

/-- Page's initial collection of shoes -/
def initial_collection : ℕ := 80

/-- Page donates 30% of her collection -/
def donation (n : ℕ) : ℕ := n * 30 / 100

/-- Page buys additional shoes -/
def additional_shoes : ℕ := 6

/-- Page's final collection after donation and purchase -/
def final_collection (n : ℕ) : ℕ := (n - donation n) + additional_shoes

/-- Proof that the final collection of shoes is 62 given the initial collection of 80 pairs -/
theorem page_shoes : (final_collection initial_collection) = 62 := 
by sorry

end page_shoes_l105_105343


namespace cost_price_A_l105_105444

theorem cost_price_A (CP_A : ℝ) (CP_B : ℝ) (SP_C : ℝ) 
(h1 : CP_B = 1.20 * CP_A)
(h2 : SP_C = 1.25 * CP_B)
(h3 : SP_C = 225) : 
CP_A = 150 := 
by 
  sorry

end cost_price_A_l105_105444


namespace simplify_expression_l105_105203

-- Define the problem context
variables {x y : ℝ} {i : ℂ}

-- The mathematical simplification problem
theorem simplify_expression :
  (x ^ 2 + i * y) ^ 3 * (x ^ 2 - i * y) ^ 3 = x ^ 12 - 9 * x ^ 8 * y ^ 2 - 9 * x ^ 4 * y ^ 4 - y ^ 6 :=
by {
  -- Proof steps would go here
  sorry
}

end simplify_expression_l105_105203


namespace whitney_money_left_over_l105_105774

def total_cost (posters_cost : ℝ) (notebooks_cost : ℝ) (bookmarks_cost : ℝ) (pencils_cost : ℝ) (tax_rate : ℝ) :=
  let pre_tax := (3 * posters_cost) + (4 * notebooks_cost) + (5 * bookmarks_cost) + (2 * pencils_cost)
  let tax := pre_tax * tax_rate
  pre_tax + tax

def money_left_over (initial_money : ℝ) (total_cost : ℝ) :=
  initial_money - total_cost

theorem whitney_money_left_over :
  let initial_money := 40
  let posters_cost := 7.50
  let notebooks_cost := 5.25
  let bookmarks_cost := 3.10
  let pencils_cost := 1.15
  let tax_rate := 0.08
  money_left_over initial_money (total_cost posters_cost notebooks_cost bookmarks_cost pencils_cost tax_rate) = -26.20 :=
by
  sorry

end whitney_money_left_over_l105_105774


namespace raj_snow_removal_volume_l105_105510

theorem raj_snow_removal_volume :
  let length := 30
  let width := 4
  let depth_layer1 := 0.5
  let depth_layer2 := 0.3
  let volume_layer1 := length * width * depth_layer1
  let volume_layer2 := length * width * depth_layer2
  let total_volume := volume_layer1 + volume_layer2
  total_volume = 96 := by
sorry

end raj_snow_removal_volume_l105_105510


namespace Faye_apps_left_l105_105899

theorem Faye_apps_left (total_apps gaming_apps utility_apps deleted_gaming_apps deleted_utility_apps remaining_apps : ℕ)
  (h1 : total_apps = 12) 
  (h2 : gaming_apps = 5) 
  (h3 : utility_apps = total_apps - gaming_apps) 
  (h4 : remaining_apps = total_apps - (deleted_gaming_apps + deleted_utility_apps))
  (h5 : deleted_gaming_apps = gaming_apps) 
  (h6 : deleted_utility_apps = 3) : 
  remaining_apps = 4 :=
by
  sorry

end Faye_apps_left_l105_105899


namespace good_pair_bound_all_good_pairs_l105_105983

namespace good_pairs

-- Definition of a "good" pair
def is_good_pair (r s : ℕ) : Prop :=
  ∃ (P : ℤ → ℤ) (a : Fin r → ℤ) (b : Fin s → ℤ),
  (∀ i j : Fin r, i ≠ j → a i ≠ a j) ∧
  (∀ i j : Fin s, i ≠ j → b i ≠ b j) ∧
  (∀ i : Fin r, P (a i) = 2) ∧
  (∀ j : Fin s, P (b j) = 5)

-- (a) Show that for every good pair (r, s), r, s ≤ 3
theorem good_pair_bound (r s : ℕ) (h : is_good_pair r s) : r ≤ 3 ∧ s ≤ 3 :=
sorry

-- (b) Determine all good pairs
theorem all_good_pairs (r s : ℕ) : is_good_pair r s ↔ (r ≤ 3 ∧ s ≤ 3 ∧ (
  (r = 1 ∧ s = 1) ∨ (r = 1 ∧ s = 2) ∨ (r = 1 ∧ s = 3) ∨
  (r = 2 ∧ s = 1) ∨ (r = 2 ∧ s = 2) ∨ (r = 2 ∧ s = 3) ∨
  (r = 3 ∧ s = 1) ∨ (r = 3 ∧ s = 2))) :=
sorry

end good_pairs

end good_pair_bound_all_good_pairs_l105_105983


namespace chess_tournament_l105_105408

theorem chess_tournament (n k : ℕ) (S : ℕ) (m : ℕ) 
  (h1 : S ≤ k * n) 
  (h2 : S ≥ m * n) 
  : m ≤ k := 
by 
  sorry

end chess_tournament_l105_105408


namespace relationship_M_N_l105_105880

theorem relationship_M_N (a b : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : 0 < b) (h4 : b < 1) 
  (M : ℝ) (hM : M = a * b) (N : ℝ) (hN : N = a + b - 1) : M > N :=
by
  sorry

end relationship_M_N_l105_105880


namespace third_roll_six_probability_l105_105730

noncomputable def Die_A_six_prob : ℚ := 1 / 6
noncomputable def Die_B_six_prob : ℚ := 1 / 2
noncomputable def Die_C_one_prob : ℚ := 3 / 5
noncomputable def Die_B_not_six_prob : ℚ := 1 / 10
noncomputable def Die_C_not_one_prob : ℚ := 1 / 15

noncomputable def prob_two_sixes_die_A : ℚ := Die_A_six_prob ^ 2
noncomputable def prob_two_sixes_die_B : ℚ := Die_B_six_prob ^ 2
noncomputable def prob_two_sixes_die_C : ℚ := Die_C_not_one_prob ^ 2

noncomputable def total_prob_two_sixes : ℚ := 
  (1 / 3) * (prob_two_sixes_die_A + prob_two_sixes_die_B + prob_two_sixes_die_C)

noncomputable def cond_prob_die_A_given_two_sixes : ℚ := prob_two_sixes_die_A / total_prob_two_sixes
noncomputable def cond_prob_die_B_given_two_sixes : ℚ := prob_two_sixes_die_B / total_prob_two_sixes
noncomputable def cond_prob_die_C_given_two_sixes : ℚ := prob_two_sixes_die_C / total_prob_two_sixes

noncomputable def prob_third_six : ℚ := 
  cond_prob_die_A_given_two_sixes * Die_A_six_prob + 
  cond_prob_die_B_given_two_sixes * Die_B_six_prob + 
  cond_prob_die_C_given_two_sixes * Die_C_not_one_prob

theorem third_roll_six_probability : 
  prob_third_six = sorry := 
  sorry

end third_roll_six_probability_l105_105730


namespace inequality_problem_l105_105230

variable (a b c d : ℝ)

open Real

theorem inequality_problem 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (hprod : a * b * c * d = 1) :
  1 / (a * (1 + b)) + 1 / (b * (1 + c)) + 1 / (c * (1 + d)) + 1 / (d * (1 + a)) ≥ 2 := 
by 
  sorry

end inequality_problem_l105_105230


namespace toothpicks_needed_for_cube_grid_l105_105989

-- Defining the conditions: a cube-shaped grid with dimensions 5x5x5.
def grid_length : ℕ := 5
def grid_width : ℕ := 5
def grid_height : ℕ := 5

-- The theorem to prove the number of toothpicks needed is 2340.
theorem toothpicks_needed_for_cube_grid (L W H : ℕ) (h1 : L = grid_length) (h2 : W = grid_width) (h3 : H = grid_height) :
  (L + 1) * (W + 1) * H + 2 * (L + 1) * W * (H + 1) = 2340 :=
  by
    -- Proof goes here
    sorry

end toothpicks_needed_for_cube_grid_l105_105989


namespace new_number_formed_l105_105332

variable (a b : ℕ)

theorem new_number_formed (ha : a < 10) (hb : b < 10) : 
  ((10 * a + b) * 10 + 2) = 100 * a + 10 * b + 2 := 
by
  sorry

end new_number_formed_l105_105332


namespace new_plants_description_l105_105826

-- Condition: Anther culture of diploid corn treated with colchicine.
def diploid_corn := Type
def colchicine_treatment (plant : diploid_corn) : Prop := -- assume we have some method to define it
sorry

def anther_culture (plant : diploid_corn) (treated : colchicine_treatment plant) : Type := -- assume we have some method to define it
sorry

-- Describe the properties of new plants
def is_haploid (plant : diploid_corn) : Prop := sorry
def has_no_homologous_chromosomes (plant : diploid_corn) : Prop := sorry
def cannot_form_fertile_gametes (plant : diploid_corn) : Prop := sorry
def has_homologous_chromosomes_in_somatic_cells (plant : diploid_corn) : Prop := sorry
def can_form_fertile_gametes (plant : diploid_corn) : Prop := sorry
def is_homozygous_or_heterozygous (plant : diploid_corn) : Prop := sorry
def is_definitely_homozygous (plant : diploid_corn) : Prop := sorry
def is_diploid (plant : diploid_corn) : Prop := sorry

-- Equivalent math proof problem
theorem new_plants_description (plant : diploid_corn) (treated : colchicine_treatment plant) : 
  is_haploid (anther_culture plant treated) ∧ 
  has_homologous_chromosomes_in_somatic_cells (anther_culture plant treated) ∧ 
  can_form_fertile_gametes (anther_culture plant treated) ∧ 
  is_homozygous_or_heterozygous (anther_culture plant treated) := sorry

end new_plants_description_l105_105826


namespace conclusion_friendly_not_large_l105_105447

variable {Snake : Type}
variable (isLarge isFriendly canClimb canSwim : Snake → Prop)
variable (marysSnakes : Finset Snake)
variable (h1 : marysSnakes.card = 16)
variable (h2 : (marysSnakes.filter isLarge).card = 6)
variable (h3 : (marysSnakes.filter isFriendly).card = 7)
variable (h4 : ∀ s, isFriendly s → canClimb s)
variable (h5 : ∀ s, isLarge s → ¬ canSwim s)
variable (h6 : ∀ s, ¬ canSwim s → ¬ canClimb s)

theorem conclusion_friendly_not_large :
  ∀ s, isFriendly s → ¬ isLarge s :=
by
  sorry

end conclusion_friendly_not_large_l105_105447


namespace root_equation_satisfies_expr_l105_105442

theorem root_equation_satisfies_expr (a : ℝ) (h : 2 * a ^ 2 - 7 * a - 1 = 0) :
  a * (2 * a - 7) + 5 = 6 :=
by
  sorry

end root_equation_satisfies_expr_l105_105442


namespace calculate_expression_l105_105959

theorem calculate_expression 
  (a1 : 84 + 4 / 19 = 1600 / 19) 
  (a2 : 105 + 5 / 19 = 2000 / 19) 
  (a3 : 1.375 = 11 / 8) 
  (a4 : 0.8 = 4 / 5) :
  84 * (4 / 19) * (11 / 8) + 105 * (5 / 19) * (4 / 5) = 200 := 
sorry

end calculate_expression_l105_105959


namespace saddle_value_l105_105945

theorem saddle_value (S : ℝ) (H : ℝ) (h1 : S + H = 100) (h2 : H = 7 * S) : S = 12.50 :=
by
  sorry

end saddle_value_l105_105945


namespace smallest_a_l105_105425

theorem smallest_a (a b : ℝ) (h₁ : 0 ≤ a) (h₂ : 0 ≤ b) (h₃ : ∀ x : ℤ, Real.sin (a * x + b) = Real.sin (17 * x)) :
  a = 17 :=
by
  sorry

end smallest_a_l105_105425


namespace mod_remainder_of_sum_of_primes_l105_105934

def odd_primes_less_than_32 : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

def sum_of_odd_primes : ℕ := List.sum odd_primes_less_than_32

theorem mod_remainder_of_sum_of_primes : sum_of_odd_primes % 32 = 30 := by
  sorry

end mod_remainder_of_sum_of_primes_l105_105934


namespace suitable_survey_l105_105414

def survey_suitable_for_census (A B C D : Prop) : Prop :=
  A ∧ ¬B ∧ ¬C ∧ ¬D

theorem suitable_survey {A B C D : Prop} (h_A : A) (h_B : ¬B) (h_C : ¬C) (h_D : ¬D) : survey_suitable_for_census A B C D :=
by
  unfold survey_suitable_for_census
  exact ⟨h_A, h_B, h_C, h_D⟩

end suitable_survey_l105_105414


namespace second_meeting_time_l105_105191

-- Given conditions and constants.
def pool_length : ℕ := 120
def initial_george_distance : ℕ := 80
def initial_henry_distance : ℕ := 40
def george_speed (t : ℕ) : ℕ := initial_george_distance / t
def henry_speed (t : ℕ) : ℕ := initial_henry_distance / t

-- Main statement to prove the question and answer.
theorem second_meeting_time (t : ℕ) (h_t_pos : t > 0) : 
  5 * t = 15 / 2 :=
sorry

end second_meeting_time_l105_105191


namespace correct_calculation_of_mistake_l105_105740

theorem correct_calculation_of_mistake (x : ℝ) (h : x - 48 = 52) : x + 48 = 148 :=
by
  sorry

end correct_calculation_of_mistake_l105_105740


namespace addition_correct_l105_105271

-- Define the integers involved
def num1 : ℤ := 22
def num2 : ℤ := 62
def result : ℤ := 84

-- Theorem stating the relationship between the given numbers
theorem addition_correct : num1 + num2 = result :=
by {
  -- proof goes here
  sorry
}

end addition_correct_l105_105271


namespace positive_integer_triplets_l105_105394

theorem positive_integer_triplets (a b c : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
    (h_lcm : a + b + c = Nat.lcm a (Nat.lcm b c)) :
  (∃ k, k ≥ 1 ∧ a = k ∧ b = 2 * k ∧ c = 3 * k) :=
sorry

end positive_integer_triplets_l105_105394


namespace smallest_m_for_triangle_sides_l105_105208

noncomputable def is_triangle_sides (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem smallest_m_for_triangle_sides (a b c : ℝ) (h : is_triangle_sides a b c) :
  (a^2 + c^2) / (b + c)^2 < 1 / 2 := sorry

end smallest_m_for_triangle_sides_l105_105208


namespace find_m_l105_105953

theorem find_m 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ)
  (h_arith_seq : ∀ n, a (n - 1) + a (n + 1) = 2 * a n)
  (h_cond1 : a (m - 1) + a (m + 1) - a m ^ 2 = 0)
  (h_cond2 : S (2 * m - 1) = 38) 
  : m = 10 :=
sorry

end find_m_l105_105953


namespace candy_last_days_l105_105046

def pieces_from_neighbors : ℝ := 11.0
def pieces_from_sister : ℝ := 5.0
def pieces_per_day : ℝ := 8.0
def total_pieces : ℝ := pieces_from_neighbors + pieces_from_sister

theorem candy_last_days : total_pieces / pieces_per_day = 2 := by
    sorry

end candy_last_days_l105_105046


namespace length_cd_l105_105795

noncomputable def isosceles_triangle (A B E : Type*) (area_abe : ℝ) (trapezoid_area : ℝ) (altitude_abe : ℝ) :
  ℝ := sorry

theorem length_cd (A B E C D : Type*) (area_abe : ℝ) (trapezoid_area : ℝ) (altitude_abe : ℝ)
  (h1 : area_abe = 144) (h2 : trapezoid_area = 108) (h3 : altitude_abe = 24) :
  isosceles_triangle A B E area_abe trapezoid_area altitude_abe = 6 := by
  sorry

end length_cd_l105_105795


namespace complement_intersection_l105_105626

noncomputable def U : Set ℤ := {-1, 0, 2}
noncomputable def A : Set ℤ := {-1, 2}
noncomputable def B : Set ℤ := {0, 2}
noncomputable def C_U_A : Set ℤ := U \ A

theorem complement_intersection :
  (C_U_A ∩ B) = {0} :=
by {
  -- sorry to skip the proof part as per instruction
  sorry
}

end complement_intersection_l105_105626


namespace log_function_domain_l105_105746

theorem log_function_domain {x : ℝ} (h : 1 / x - 1 > 0) : 0 < x ∧ x < 1 :=
sorry

end log_function_domain_l105_105746


namespace point_in_third_quadrant_l105_105021

theorem point_in_third_quadrant (x y : ℝ) (h1 : x + y < 0) (h2 : x * y > 0) : x < 0 ∧ y < 0 := 
sorry

end point_in_third_quadrant_l105_105021


namespace min_value_proof_l105_105609

noncomputable def min_value : ℝ := 3 + 2 * Real.sqrt 2

theorem min_value_proof (m n : ℝ) (hm : 0 < m) (hn : 0 < n) (h : m + n = 1) :
  (1 / m + 2 / n) = min_value :=
sorry

end min_value_proof_l105_105609


namespace cuboid_length_l105_105319

theorem cuboid_length (b h : ℝ) (A : ℝ) (l : ℝ) : b = 6 → h = 5 → A = 120 → 2 * (l * b + b * h + h * l) = A → l = 30 / 11 :=
by
  intros hb hh hA hSurfaceArea
  rw [hb, hh] at hSurfaceArea
  sorry

end cuboid_length_l105_105319


namespace max_gcd_11n_3_6n_1_l105_105259

theorem max_gcd_11n_3_6n_1 : ∃ n : ℕ+, ∀ k : ℕ+,  11 * n + 3 = 7 * k + 1 ∧ 6 * n + 1 = 7 * k + 2 → ∃ d : ℕ, d = Nat.gcd (11 * n + 3) (6 * n + 1) ∧ d = 7 :=
by
  sorry

end max_gcd_11n_3_6n_1_l105_105259


namespace fifteen_horses_fifteen_bags_l105_105864

-- Definitions based on the problem
def days_for_one_horse_one_bag : ℝ := 1  -- It takes 1 day for 1 horse to eat 1 bag of grain

-- Theorem statement
theorem fifteen_horses_fifteen_bags {d : ℝ} (h : d = days_for_one_horse_one_bag) :
  d = 1 :=
by
  sorry

end fifteen_horses_fifteen_bags_l105_105864


namespace common_ratio_of_geometric_progression_l105_105466

-- Define the problem conditions
variables {a b c q : ℝ}

-- The sequence a, b, c is a geometric progression
def geometric_progression (a b c : ℝ) (q : ℝ) : Prop :=
  b = a * q ∧ c = a * q^2

-- The sequence 577a, (2020b/7), (c/7) is an arithmetic progression
def arithmetic_progression (x y z : ℝ) : Prop :=
  2 * y = x + z

-- Main theorem statement to prove
theorem common_ratio_of_geometric_progression (h1 : geometric_progression a b c q) 
  (h2 : arithmetic_progression (577 * a) (2020 * b / 7) (c / 7)) 
  (h3 : b < a ∧ c < b) : q = 4039 :=
sorry

end common_ratio_of_geometric_progression_l105_105466

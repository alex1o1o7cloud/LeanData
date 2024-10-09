import Mathlib

namespace plants_per_row_l518_51895

-- Define the conditions from the problem
def rows : ℕ := 7
def extra_plants : ℕ := 15
def total_plants : ℕ := 141

-- Define the problem statement to prove
theorem plants_per_row :
  ∃ x : ℕ, rows * x + extra_plants = total_plants ∧ x = 18 :=
by
  sorry

end plants_per_row_l518_51895


namespace circles_do_not_intersect_first_scenario_circles_do_not_intersect_second_scenario_l518_51834

-- Define radii of the circles
def r1 : ℝ := 3
def r2 : ℝ := 5

-- Statement for first scenario (distance = 9)
theorem circles_do_not_intersect_first_scenario (d : ℝ) (h : d = 9) : ¬ (|r1 - r2| ≤ d ∧ d ≤ r1 + r2) :=
by sorry

-- Statement for second scenario (distance = 1)
theorem circles_do_not_intersect_second_scenario (d : ℝ) (h : d = 1) : d < |r1 - r2| ∨ ¬ (|r1 - r2| ≤ d ∧ d ≤ r1 + r2) :=
by sorry

end circles_do_not_intersect_first_scenario_circles_do_not_intersect_second_scenario_l518_51834


namespace lines_parallel_if_perpendicular_to_plane_l518_51844

variables {α β γ : Plane} {m n : Line}

-- Define the properties of perpendicular lines to planes and parallel lines
def perpendicular_to (l : Line) (p : Plane) : Prop := 
sorry -- definition skipped

def parallel_to (l1 l2 : Line) : Prop := 
sorry -- definition skipped

-- Theorem Statement (equivalent translation of the given question and its correct answer)
theorem lines_parallel_if_perpendicular_to_plane 
  (h1 : perpendicular_to m α) 
  (h2 : perpendicular_to n α) : parallel_to m n :=
sorry

end lines_parallel_if_perpendicular_to_plane_l518_51844


namespace number_of_chickens_l518_51822

-- Definitions based on conditions
def totalAnimals := 100
def legDifference := 26

-- The problem statement to be proved
theorem number_of_chickens (x : Nat) (r : Nat) (legs_chickens : Nat) (legs_rabbits : Nat) (total : Nat := totalAnimals) (diff : Nat := legDifference) :
  x + r = total ∧ 2 * x + 4 * r - 4 * r = 2 * x + diff → x = 71 :=
by
  intro h
  sorry

end number_of_chickens_l518_51822


namespace maximum_xy_l518_51845

theorem maximum_xy (x y : ℕ) (h1 : 7 * x + 2 * y = 110) : ∃ x y, (7 * x + 2 * y = 110) ∧ (x > 0) ∧ (y > 0) ∧ (x * y = 216) :=
by
  sorry

end maximum_xy_l518_51845


namespace truncated_cone_volume_correct_larger_cone_volume_correct_l518_51875

def larger_base_radius : ℝ := 10 -- R
def smaller_base_radius : ℝ := 5  -- r
def height_truncated_cone : ℝ := 8 -- h
def height_small_cone : ℝ := 8 -- x

noncomputable def volume_truncated_cone : ℝ :=
  (1/3) * Real.pi * height_truncated_cone * 
  (larger_base_radius^2 + larger_base_radius * smaller_base_radius + smaller_base_radius^2)

theorem truncated_cone_volume_correct :
  volume_truncated_cone = 466 + 2/3 * Real.pi := sorry

noncomputable def total_height_larger_cone : ℝ :=
  height_small_cone + height_truncated_cone

noncomputable def volume_larger_cone : ℝ :=
  (1/3) * Real.pi * (larger_base_radius^2) * total_height_larger_cone

theorem larger_cone_volume_correct :
  volume_larger_cone = 533 + 1/3 * Real.pi := sorry

end truncated_cone_volume_correct_larger_cone_volume_correct_l518_51875


namespace find_middle_integer_l518_51871

theorem find_middle_integer (a b c : ℕ) (h1 : a^2 = 97344) (h2 : c^2 = 98596) (h3 : c = a + 2) : b = a + 1 ∧ b = 313 :=
by
  sorry

end find_middle_integer_l518_51871


namespace find_initial_population_l518_51810

noncomputable def population_first_year (P : ℝ) : ℝ :=
  let P1 := 0.90 * P    -- population after 1st year
  let P2 := 0.99 * P    -- population after 2nd year
  let P3 := 0.891 * P   -- population after 3rd year
  P3

theorem find_initial_population (h : population_first_year P = 4455) : P = 4455 / 0.891 :=
by
  sorry

end find_initial_population_l518_51810


namespace ellipse_standard_equation_l518_51896

theorem ellipse_standard_equation
  (a b c : ℝ)
  (h1 : (3 * a) / (-a) + 16 / b = 1)
  (h2 : (3 * a) / c + 16 / (-b) = 1)
  (h3 : a > 0)
  (h4 : b > 0)
  (h5 : a > b)
  (h6 : a^2 = b^2 + c^2) : 
  (a = 5 ∧ b = 4 ∧ c = 3) ∧ (∀ x y, x^2 / 25 + y^2 / 16 = 1 ↔ (a = 5 ∧ b = 4)) := 
sorry

end ellipse_standard_equation_l518_51896


namespace question_1_question_2_question_3_l518_51873

def deck_size : Nat := 32

theorem question_1 :
  let hands_when_order_matters := deck_size * (deck_size - 1)
  hands_when_order_matters = 992 :=
by
  let hands_when_order_matters := deck_size * (deck_size - 1)
  sorry

theorem question_2 :
  let hands_when_order_does_not_matter := (deck_size * (deck_size - 1)) / 2
  hands_when_order_does_not_matter = 496 :=
by
  let hands_when_order_does_not_matter := (deck_size * (deck_size - 1)) / 2
  sorry

theorem question_3 :
  let hands_3_cards_order_does_not_matter := (deck_size * (deck_size - 1) * (deck_size - 2)) / 6
  hands_3_cards_order_does_not_matter = 4960 :=
by
  let hands_3_cards_order_does_not_matter := (deck_size * (deck_size - 1) * (deck_size - 2)) / 6
  sorry

end question_1_question_2_question_3_l518_51873


namespace sanity_indeterminable_transylvanian_is_upyr_l518_51808

noncomputable def transylvanianClaim := "I have lost my mind."

/-- Proving whether the sanity of the Transylvanian can be determined from the statement -/
theorem sanity_indeterminable (claim : String) : 
  claim = transylvanianClaim → 
  ¬ (∀ (sane : Prop), sane ∨ ¬ sane) := 
by 
  intro h
  rw [transylvanianClaim] at h
  sorry

/-- Proving the nature of whether the Transylvanian is an upyr or human from the statement -/
theorem transylvanian_is_upyr (claim : String) : 
  claim = transylvanianClaim → 
  ∀ (human upyr : Prop), ¬ human ∧ upyr := 
by 
  intro h
  rw [transylvanianClaim] at h
  sorry

end sanity_indeterminable_transylvanian_is_upyr_l518_51808


namespace annual_rate_of_decrease_l518_51886

variable (r : ℝ) (initial_population population_after_2_years : ℝ)

-- Conditions
def initial_population_eq : initial_population = 30000 := sorry
def population_after_2_years_eq : population_after_2_years = 19200 := sorry
def population_formula : population_after_2_years = initial_population * (1 - r)^2 := sorry

-- Goal: Prove that the annual rate of decrease r is 0.2
theorem annual_rate_of_decrease :
  r = 0.2 := sorry

end annual_rate_of_decrease_l518_51886


namespace assignment_statement_correct_l518_51811

-- Definitions for the conditions:
def cond_A : Prop := ∀ M : ℕ, (M = M + 3)
def cond_B : Prop := ∀ M : ℕ, (M = M + (3 - M))
def cond_C : Prop := ∀ M : ℕ, (M = M + 3)
def cond_D : Prop := true ∧ cond_A ∧ cond_B ∧ cond_C

-- Theorem statement proving the correct interpretation of the assignment is condition B
theorem assignment_statement_correct : cond_B :=
by
  sorry

end assignment_statement_correct_l518_51811


namespace mushrooms_collected_l518_51823

variable (P V : ℕ)

theorem mushrooms_collected (h1 : P = (V * 100) / (P + V)) (h2 : V % 2 = 1) :
  P + V = 25 ∨ P + V = 300 ∨ P + V = 525 ∨ P + V = 1900 ∨ P + V = 9900 := by
  sorry

end mushrooms_collected_l518_51823


namespace situps_combined_l518_51878

theorem situps_combined (peter_situps : ℝ) (greg_per_set : ℝ) (susan_per_set : ℝ) 
                        (peter_per_set : ℝ) (sets : ℝ) 
                        (peter_situps_performed : peter_situps = sets * peter_per_set) 
                        (greg_situps_performed : sets * greg_per_set = 4.5 * 6)
                        (susan_situps_performed : sets * susan_per_set = 3.75 * 6) :
    peter_situps = 37.5 ∧ greg_per_set = 4.5 ∧ susan_per_set = 3.75 ∧ peter_per_set = 6.25 → 
    4.5 * 6 + 3.75 * 6 = 49.5 :=
by
  sorry

end situps_combined_l518_51878


namespace smallest_d_for_divisibility_by_9_l518_51806

theorem smallest_d_for_divisibility_by_9 : ∃ d : ℕ, 0 ≤ d ∧ d < 10 ∧ (437003 + d * 100) % 9 = 0 ∧ ∀ d', 0 ≤ d' ∧ d' < d → ((437003 + d' * 100) % 9 ≠ 0) :=
by
  sorry

end smallest_d_for_divisibility_by_9_l518_51806


namespace investment_ratio_l518_51884

-- Define the investments
def A_investment (x : ℝ) : ℝ := 3 * x
def B_investment (x : ℝ) : ℝ := x
def C_investment (y : ℝ) : ℝ := y

-- Define the total profit and B's share of the profit
def total_profit : ℝ := 4400
def B_share : ℝ := 800

-- Define the ratio condition B's share based on investments
def B_share_cond (x y : ℝ) : Prop := (B_investment x / (A_investment x + B_investment x + C_investment y)) * total_profit = B_share

-- Define what we need to prove
theorem investment_ratio (x y : ℝ) (h : B_share_cond x y) : x / y = 2 / 3 :=
by 
  sorry

end investment_ratio_l518_51884


namespace slope_of_line_l518_51856

theorem slope_of_line {x1 x2 y1 y2 : ℝ} 
  (h1 : (1 / x1 + 2 / y1 = 0)) 
  (h2 : (1 / x2 + 2 / y2 = 0)) 
  (h_neq : x1 ≠ x2) : 
  (y2 - y1) / (x2 - x1) = -2 := 
sorry

end slope_of_line_l518_51856


namespace g_at_pi_over_4_l518_51880

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 3)
noncomputable def g (x : ℝ) : ℝ := f x + 1

theorem g_at_pi_over_4 : g (Real.pi / 4) = 3 / 2 :=
by 
  sorry

end g_at_pi_over_4_l518_51880


namespace rowing_speed_in_still_water_l518_51867

theorem rowing_speed_in_still_water (v c : ℝ) (t : ℝ) (h1 : c = 1.1) (h2 : (v + c) * t = (v - c) * 2 * t) : v = 3.3 :=
sorry

end rowing_speed_in_still_water_l518_51867


namespace largest_prime_m_satisfying_quadratic_inequality_l518_51863

theorem largest_prime_m_satisfying_quadratic_inequality :
  ∃ (m : ℕ), m = 5 ∧ m^2 - 11 * m + 28 < 0 ∧ Prime m :=
by sorry

end largest_prime_m_satisfying_quadratic_inequality_l518_51863


namespace wilson_theorem_application_l518_51800

theorem wilson_theorem_application (h_prime : Nat.Prime 101) : 
  Nat.factorial 100 % 101 = 100 :=
by
  -- By Wilson's theorem, (p - 1)! ≡ -1 (mod p) for a prime p.
  -- Here p = 101, so (101 - 1)! ≡ -1 (mod 101).
  -- Therefore, 100! ≡ -1 (mod 101).
  -- Knowing that -1 ≡ 100 (mod 101), we can conclude that
  -- 100! ≡ 100 (mod 101).
  sorry

end wilson_theorem_application_l518_51800


namespace circumference_to_diameter_ratio_l518_51887

-- Definitions from the conditions
def r : ℝ := 15
def C : ℝ := 90
def D : ℝ := 2 * r

-- The proof goal
theorem circumference_to_diameter_ratio : C / D = 3 := 
by sorry

end circumference_to_diameter_ratio_l518_51887


namespace incorrect_option_C_l518_51843

def line (α : Type*) := α → Prop
def plane (α : Type*) := α → Prop

variables {α : Type*} (m n : line α) (a b : plane α)

def parallel (m n : line α) : Prop := ∀ x, m x → n x
def perpendicular (m n : line α) : Prop := ∃ x, m x ∧ n x

def lies_in (m : line α) (a : plane α) : Prop := ∀ x, m x → a x

theorem incorrect_option_C (h : lies_in m a) : ¬ (parallel m n ∧ lies_in m a → parallel n a) :=
sorry

end incorrect_option_C_l518_51843


namespace find_x_l518_51849

def f (x : ℝ) : ℝ := 3 * x - 5

theorem find_x (x : ℝ) (h : 2 * f x - 19 = f (x - 4)) : x = 4 := 
by 
  sorry

end find_x_l518_51849


namespace sum_of_six_consecutive_integers_l518_51862

theorem sum_of_six_consecutive_integers (n : ℤ) : 
  (n + (n+1) + (n+2) + (n+3) + (n+4) + (n+5)) = 6 * n + 15 :=
by
  sorry

end sum_of_six_consecutive_integers_l518_51862


namespace find_y_l518_51889

def star (a b : ℝ) : ℝ := a * b + 3 * b - a

theorem find_y (y : ℝ) (h : star 7 y = 47) : y = 5.4 := 
by 
  sorry

end find_y_l518_51889


namespace rowing_speed_l518_51855

theorem rowing_speed :
  ∀ (initial_width final_width increase_per_10m : ℝ) (time_seconds : ℝ)
  (yards_to_meters : ℝ → ℝ) (width_increase_in_yards : ℝ) (distance_10m_segments : ℝ) 
  (total_distance : ℝ),
  initial_width = 50 →
  final_width = 80 →
  increase_per_10m = 2 →
  time_seconds = 30 →
  yards_to_meters 1 = 0.9144 →
  width_increase_in_yards = (final_width - initial_width) →
  width_increase_in_yards * (yards_to_meters 1) = 27.432 →
  distance_10m_segments = (width_increase_in_yards * (yards_to_meters 1)) / 10 →
  total_distance = distance_10m_segments * 10 →
  (total_distance / time_seconds) = 0.9144 :=
by
  intros initial_width final_width increase_per_10m time_seconds yards_to_meters 
        width_increase_in_yards distance_10m_segments total_distance
  sorry

end rowing_speed_l518_51855


namespace ensure_two_of_each_kind_l518_51816

def tablets_A := 10
def tablets_B := 14
def least_number_of_tablets_to_ensure_two_of_each := 12

theorem ensure_two_of_each_kind 
  (total_A : ℕ) 
  (total_B : ℕ) 
  (extracted : ℕ) 
  (hA : total_A = tablets_A) 
  (hB : total_B = tablets_B)
  (hExtract : extracted = least_number_of_tablets_to_ensure_two_of_each) : 
  ∃ (extracted : ℕ), extracted = least_number_of_tablets_to_ensure_two_of_each ∧ extracted ≥ tablets_A + 2 := 
sorry

end ensure_two_of_each_kind_l518_51816


namespace sum_le_30_l518_51853

variable (a b x y : ℝ)
variable (ha_pos : 0 < a) (hb_pos : 0 < b) (hx_pos : 0 < x) (hy_pos : 0 < y)
variable (h1 : a * x ≤ 5) (h2 : a * y ≤ 10) (h3 : b * x ≤ 10) (h4 : b * y ≤ 10)

theorem sum_le_30 : a * x + a * y + b * x + b * y ≤ 30 := sorry

end sum_le_30_l518_51853


namespace factorization_of_polynomial_l518_51836

theorem factorization_of_polynomial (x : ℝ) :
  x^6 - x^4 - x^2 + 1 = (x - 1) * (x + 1) * (x^2 + 1) := 
sorry

end factorization_of_polynomial_l518_51836


namespace always_positive_expression_l518_51874

variable (x a b : ℝ)

theorem always_positive_expression (h : ∀ x, (x - a)^2 + b > 0) : b > 0 :=
sorry

end always_positive_expression_l518_51874


namespace weight_of_smallest_box_l518_51890

variables (M S L : ℕ)

theorem weight_of_smallest_box
  (h1 : M + S = 83)
  (h2 : L + S = 85)
  (h3 : L + M = 86) :
  S = 41 :=
sorry

end weight_of_smallest_box_l518_51890


namespace mn_value_l518_51848

theorem mn_value (m n : ℤ) (h1 : m + n = 1) (h2 : m - n + 2 = 1) : m * n = 0 := 
by 
  sorry

end mn_value_l518_51848


namespace houses_with_both_l518_51829

theorem houses_with_both (G P N Total B : ℕ) 
  (hG : G = 50) 
  (hP : P = 40) 
  (hN : N = 10) 
  (hTotal : Total = 65)
  (hEquation : G + P - B = Total - N) 
  : B = 35 := 
by 
  sorry

end houses_with_both_l518_51829


namespace total_weight_of_pumpkins_l518_51801

def first_pumpkin_weight : ℝ := 12.6
def second_pumpkin_weight : ℝ := 23.4
def total_weight : ℝ := 36

theorem total_weight_of_pumpkins :
  first_pumpkin_weight + second_pumpkin_weight = total_weight :=
by
  sorry

end total_weight_of_pumpkins_l518_51801


namespace row_column_crossout_l518_51812

theorem row_column_crossout (M : Matrix (Fin 1000) (Fin 1000) Bool) :
  (∃ rows : Finset (Fin 1000), rows.card = 990 ∧ ∀ j : Fin 1000, ∃ i ∈ rowsᶜ, M i j = 1) ∨
  (∃ cols : Finset (Fin 1000), cols.card = 990 ∧ ∀ i : Fin 1000, ∃ j ∈ colsᶜ, M i j = 0) :=
by {
  sorry
}

end row_column_crossout_l518_51812


namespace find_P_nplus1_l518_51892

-- Conditions
def P (n : ℕ) (k : ℕ) : ℚ :=
  1 / Nat.choose n k

-- Lean 4 statement for the proof
theorem find_P_nplus1 (n : ℕ) : (if Even n then P n (n+1) = 1 else P n (n+1) = 0) := by
  sorry

end find_P_nplus1_l518_51892


namespace count_valid_c_l518_51879

theorem count_valid_c : ∃ (count : ℕ), count = 670 ∧ 
  ∀ (c : ℤ), (-2007 ≤ c ∧ c ≤ 2007) → 
    (∃ (x : ℤ), (x^2 + c) % (2^2007) = 0) ↔ count = 670 :=
sorry

end count_valid_c_l518_51879


namespace max_lines_between_points_l518_51893

noncomputable def maxLines (points : Nat) := 
  let deg := [1, 2, 3, 4, 5]
  (1 * (points - 1) + 2 * (points - 2) + 3 * (points - 3) + 4 * (points - 4) + 5 * (points - 5)) / 2

theorem max_lines_between_points :
  ∀ (n : Nat), n = 15 → maxLines n = 85 :=
by
  intros n hn
  sorry

end max_lines_between_points_l518_51893


namespace vertex_below_x_axis_l518_51821

theorem vertex_below_x_axis (a : ℝ) : 
  (∃ x : ℝ, x^2 + 2 * x + a < 0) → a < 1 :=
by 
  sorry

end vertex_below_x_axis_l518_51821


namespace square_side_length_l518_51846

variable (x : ℝ) (π : ℝ) (hπ: π = Real.pi)

theorem square_side_length (h1: 4 * x = 10 * π) : 
  x = (5 * π) / 2 := 
by
  sorry

end square_side_length_l518_51846


namespace exponent_rule_l518_51857

variable (a : ℝ) (m n : ℕ)

theorem exponent_rule (h1 : a^m = 3) (h2 : a^n = 2) : a^(m + n) = 6 :=
by
  sorry

end exponent_rule_l518_51857


namespace second_term_geometric_series_l518_51854

theorem second_term_geometric_series (a r S : ℝ) (h1 : r = 1 / 4) (h2 : S = 48) (h3 : S = a / (1 - r)) :
  a * r = 9 :=
by
  -- Sorry is used to finalize the theorem without providing a proof here
  sorry

end second_term_geometric_series_l518_51854


namespace find_nat_nums_satisfying_eq_l518_51827

theorem find_nat_nums_satisfying_eq (m n : ℕ) (h_m : m = 3) (h_n : n = 3) : 2 ^ n + 1 = m ^ 2 :=
by
  rw [h_m, h_n]
  sorry

end find_nat_nums_satisfying_eq_l518_51827


namespace find_duplicated_page_number_l518_51860

noncomputable def duplicated_page_number (n : ℕ) (incorrect_sum : ℕ) : ℕ :=
  incorrect_sum - n * (n + 1) / 2

theorem find_duplicated_page_number :
  ∃ n k, (1 <= k ∧ k <= n) ∧ ( ∃ n, (1 <= n) ∧ ( n * (n + 1) / 2 + k = 2550) )
  ∧ duplicated_page_number 70 2550 = 65 :=
by
  sorry

end find_duplicated_page_number_l518_51860


namespace sum_of_net_gains_is_correct_l518_51825

namespace DepartmentRevenue

def revenueIncreaseA : ℝ := 0.1326
def revenueIncreaseB : ℝ := 0.0943
def revenueIncreaseC : ℝ := 0.7731
def taxRate : ℝ := 0.235
def initialRevenue : ℝ := 4.7 -- in millions

def netGain (revenueIncrease : ℝ) (taxRate : ℝ) (initialRevenue : ℝ) : ℝ :=
  (initialRevenue * (1 + revenueIncrease)) * (1 - taxRate)

def netGainA : ℝ := netGain revenueIncreaseA taxRate initialRevenue
def netGainB : ℝ := netGain revenueIncreaseB taxRate initialRevenue
def netGainC : ℝ := netGain revenueIncreaseC taxRate initialRevenue

def netGainSum : ℝ := netGainA + netGainB + netGainC

theorem sum_of_net_gains_is_correct :
  netGainSum = 14.38214 := by
    sorry

end DepartmentRevenue

end sum_of_net_gains_is_correct_l518_51825


namespace cost_per_pancake_correct_l518_51815

-- Define the daily rent expense
def daily_rent := 30

-- Define the daily supplies expense
def daily_supplies := 12

-- Define the number of pancakes needed to cover expenses
def number_of_pancakes := 21

-- Define the total daily expenses
def total_daily_expenses := daily_rent + daily_supplies

-- Define the cost per pancake calculation
def cost_per_pancake := total_daily_expenses / number_of_pancakes

-- The theorem to prove the cost per pancake
theorem cost_per_pancake_correct :
  cost_per_pancake = 2 := 
by
  sorry

end cost_per_pancake_correct_l518_51815


namespace express_as_sum_of_cubes_l518_51870

variables {a b : ℝ}

theorem express_as_sum_of_cubes (a b : ℝ) : 
  2 * a * (a^2 + 3 * b^2) = (a + b)^3 + (a - b)^3 :=
by sorry

end express_as_sum_of_cubes_l518_51870


namespace repeating_decimals_subtraction_l518_51881

/--
Calculate the value of 0.\overline{234} - 0.\overline{567} - 0.\overline{891}.
Express your answer as a fraction in its simplest form.

Shown that:
Let x = 0.\overline{234}, y = 0.\overline{567}, z = 0.\overline{891},
Then 0.\overline{234} - 0.\overline{567} - 0.\overline{891} = -1224/999
-/
theorem repeating_decimals_subtraction : 
  let x : ℚ := 234 / 999
  let y : ℚ := 567 / 999
  let z : ℚ := 891 / 999
  x - y - z = -1224 / 999 := 
by
  sorry

end repeating_decimals_subtraction_l518_51881


namespace sym_sum_ineq_l518_51897

theorem sym_sum_ineq (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (h : x + y + z = 1 / x + 1 / y + 1 / z) : x * y + y * z + z * x ≥ 3 :=
by
  sorry

end sym_sum_ineq_l518_51897


namespace hike_duration_l518_51859

def initial_water := 11
def final_water := 2
def leak_rate := 1
def water_drunk := 6

theorem hike_duration (time_hours : ℕ) :
  initial_water - final_water = water_drunk + time_hours * leak_rate →
  time_hours = 3 :=
by intro h; sorry

end hike_duration_l518_51859


namespace equivalent_octal_to_decimal_l518_51831

def octal_to_decimal (n : Nat) : Nat :=
  match n with
  | 0 => 0
  | n+1 => (n % 10) + 8 * octal_to_decimal (n / 10)

theorem equivalent_octal_to_decimal : octal_to_decimal 753 = 491 :=
by
  sorry

end equivalent_octal_to_decimal_l518_51831


namespace rational_function_sum_l518_51813

-- Define the problem conditions and the target equality
theorem rational_function_sum (p q : ℝ → ℝ) :
  (∀ x, (p x) / (q x) = (x - 1) / ((x + 1) * (x - 1))) ∧
  (∀ x ≠ -1, q x ≠ 0) ∧
  (q 2 = 3) ∧
  (p 2 = 1) →
  (p x + q x = x^2 + x - 2) := by
  sorry

end rational_function_sum_l518_51813


namespace johns_website_visits_l518_51833

theorem johns_website_visits (c: ℝ) (d: ℝ) (days: ℕ) (h1: c = 0.01) (h2: d = 10) (h3: days = 30) :
  d / c * days = 30000 :=
by
  sorry

end johns_website_visits_l518_51833


namespace distance_of_point_P_to_origin_l518_51891

noncomputable def dist_to_origin (P : ℝ × ℝ) : ℝ :=
  Real.sqrt (P.1 ^ 2 + P.2 ^ 2)

theorem distance_of_point_P_to_origin :
  let F1 := (-Real.sqrt 2, 0)
  let F2 := (Real.sqrt 2, 0)
  let y_P := 1 / 2
  ∃ x_P : ℝ, (x_P, y_P) = P ∧
    (dist_to_origin P = Real.sqrt 6 / 2) :=
by
  sorry

end distance_of_point_P_to_origin_l518_51891


namespace possible_third_side_of_triangle_l518_51840

theorem possible_third_side_of_triangle (a b : ℝ) (ha : a = 3) (hb : b = 6) (x : ℝ) :
  3 < x ∧ x < 9 → x = 6 :=
by
  intros h
  have h1 : 3 < x := h.left
  have h2 : x < 9 := h.right
  have h3 : a + b > x := by linarith
  have h4 : b - a < x := by linarith
  sorry

end possible_third_side_of_triangle_l518_51840


namespace problem1_problem2_l518_51826

open Real -- Open the Real namespace to use real number trigonometric functions

-- Problem 1
theorem problem1 (α : ℝ) (hα : tan α = 3) : 
  (4 * sin α - 2 * cos α) / (5 * cos α + 3 * sin α) = 5/7 :=
sorry

-- Problem 2
theorem problem2 (θ : ℝ) (hθ : tan θ = -3/4) : 
  2 + sin θ * cos θ - cos θ ^ 2 = 22 / 25 :=
sorry

end problem1_problem2_l518_51826


namespace initial_potatoes_count_l518_51861

theorem initial_potatoes_count (initial_tomatoes picked_tomatoes total_remaining : ℕ) 
    (h_initial_tomatoes : initial_tomatoes = 177)
    (h_picked_tomatoes : picked_tomatoes = 53)
    (h_total_remaining : total_remaining = 136) :
  (initial_tomatoes - picked_tomatoes + x = total_remaining) → 
  x = 12 :=
by 
  sorry

end initial_potatoes_count_l518_51861


namespace trig_expression_l518_51817

theorem trig_expression (α : ℝ) (h : Real.tan α = 2) : 
    (2 * Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = 1 := 
by 
  sorry

end trig_expression_l518_51817


namespace plane_distance_last_10_seconds_l518_51820

theorem plane_distance_last_10_seconds (s : ℝ → ℝ) (h : ∀ t, s t = 60 * t - 1.5 * t^2) : 
  s 20 - s 10 = 150 := 
by 
  sorry

end plane_distance_last_10_seconds_l518_51820


namespace cos_7theta_l518_51842

theorem cos_7theta (θ : ℝ) (h : Real.cos θ = 1/4) : Real.cos (7 * θ) = -45682/8192 :=
by
  sorry

end cos_7theta_l518_51842


namespace left_handed_women_percentage_l518_51899

noncomputable section

variables (x y : ℕ) (percentage : ℝ)

-- Conditions
def right_handed_ratio := 3
def left_handed_ratio := 1
def men_ratio := 3
def women_ratio := 2

def total_population_by_hand := right_handed_ratio * x + left_handed_ratio * x -- i.e., 4x
def total_population_by_gender := men_ratio * y + women_ratio * y -- i.e., 5y

-- Main Statement
theorem left_handed_women_percentage (h1 : total_population_by_hand = total_population_by_gender) :
    percentage = 25 :=
by
  sorry

end left_handed_women_percentage_l518_51899


namespace number_divided_by_three_l518_51828

theorem number_divided_by_three (x : ℝ) (h : x / 3 = x - 3) : x = 4.5 :=
sorry

end number_divided_by_three_l518_51828


namespace cement_percentage_first_concrete_correct_l518_51876

open Real

noncomputable def cement_percentage_of_first_concrete := 
  let total_weight := 4500 
  let cement_percentage := 10.8 / 100
  let weight_each_type := 1125
  let total_cement_weight := cement_percentage * total_weight
  let x := 2.0 / 100
  let y := 21.6 / 100 - x
  (weight_each_type * x + weight_each_type * y = total_cement_weight) →
  (x = 2.0 / 100)

theorem cement_percentage_first_concrete_correct :
  cement_percentage_of_first_concrete := sorry

end cement_percentage_first_concrete_correct_l518_51876


namespace roots_greater_than_one_l518_51804

def quadratic_roots_greater_than_one (a : ℝ) : Prop :=
  ∀ x : ℝ, (1 + a) * x^2 - 3 * a * x + 4 * a = 0 → x > 1

theorem roots_greater_than_one (a : ℝ) :
  -16/7 < a ∧ a < -1 → quadratic_roots_greater_than_one a :=
sorry

end roots_greater_than_one_l518_51804


namespace compute_expression_l518_51809

noncomputable def c : ℝ := Real.log 8
noncomputable def d : ℝ := Real.log 25

theorem compute_expression : 5^(c / d) + 2^(d / c) = 2 * Real.sqrt 2 + 5^(2 / 3) :=
by
  sorry

end compute_expression_l518_51809


namespace find_d_l518_51894

theorem find_d (c d : ℝ) (f g : ℝ → ℝ)
  (hf : ∀ x, f x = 5 * x + c)
  (hg : ∀ x, g x = c * x + 3)
  (hfg : ∀ x, f (g x) = 15 * x + d) :
  d = 18 :=
sorry

end find_d_l518_51894


namespace equation_of_line_l518_51851

theorem equation_of_line (a b : ℝ) (h1 : a = -2) (h2 : b = 2) :
  (∀ x y : ℝ, (x / a + y / b = 1) → x - y + 2 = 0) :=
by
  sorry

end equation_of_line_l518_51851


namespace find_a2_l518_51877

variable {a_n : ℕ → ℚ}

def arithmetic_seq (a : ℕ → ℚ) : Prop :=
  ∃ a1 d, ∀ n, a n = a1 + (n-1) * d

theorem find_a2 (h_seq : arithmetic_seq a_n) (h3_5 : a_n 3 + a_n 5 = 15) (h6 : a_n 6 = 7) :
  a_n 2 = 8 := 
sorry

end find_a2_l518_51877


namespace number_of_points_in_star_polygon_l518_51852

theorem number_of_points_in_star_polygon :
  ∀ (n : ℕ) (D C : ℕ),
    (∀ i : ℕ, i < n → C = D - 15) →
    n * (D - (D - 15)) = 360 → n = 24 :=
by
  intros n D C h1 h2
  sorry

end number_of_points_in_star_polygon_l518_51852


namespace sum_two_numbers_l518_51805

theorem sum_two_numbers :
  let X := (2 * 10) + 6
  let Y := (4 * 10) + 1
  X + Y = 67 :=
by
  sorry

end sum_two_numbers_l518_51805


namespace candy_distribution_l518_51832

theorem candy_distribution (candies : ℕ) (family_members : ℕ) (required_candies : ℤ) :
  (candies = 45) ∧ (family_members = 5) →
  required_candies = 0 :=
by sorry

end candy_distribution_l518_51832


namespace geometric_sequence_S5_equals_l518_51838

theorem geometric_sequence_S5_equals :
  ∀ (a : ℕ → ℤ) (q : ℤ), 
    a 1 = 1 → 
    (a 3 + a 4) / (a 1 + a 2) = 4 → 
    ((S5 : ℤ) = 31 ∨ (S5 : ℤ) = 11) :=
by
  sorry

end geometric_sequence_S5_equals_l518_51838


namespace f_2011_l518_51866

noncomputable def f : ℝ → ℝ := sorry

axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom f_periodic : ∀ x : ℝ, f (x + 2) = -f x
axiom f_defined_segment : ∀ x, 0 < x ∧ x < 2 → f x = 2 * x^2

theorem f_2011 : f 2011 = -2 := by
  sorry

end f_2011_l518_51866


namespace sum_of_possible_coefficient_values_l518_51824

theorem sum_of_possible_coefficient_values :
  let pairs := [(1, 48), (2, 24), (3, 16), (4, 12), (6, 8)]
  let values := pairs.map (fun (r, s) => r + s)
  values.sum = 124 :=
by
  sorry

end sum_of_possible_coefficient_values_l518_51824


namespace fraction_of_jumbo_tiles_l518_51802

-- Definitions for conditions
variables (L W : ℝ) -- Length and width of regular tiles
variables (n : ℕ) -- Number of regular tiles
variables (m : ℕ) -- Number of jumbo tiles

-- Conditions
def condition1 : Prop := (n : ℝ) * (L * W) = 40 -- Regular tiles cover 40 square feet
def condition2 : Prop := (n : ℝ) * (L * W) + (m : ℝ) * (3 * L * W) = 220 -- Entire wall is 220 square feet
def condition3 : Prop := ∃ (k : ℝ), (m : ℝ) = k * (n : ℝ) ∧ k = 1.5 -- Relationship ratio between jumbo and regular tiles

-- Theorem to be proved
theorem fraction_of_jumbo_tiles (L W : ℝ) (n m : ℕ)
  (h1 : condition1 L W n)
  (h2 : condition2 L W n m)
  (h3 : condition3 n m) :
  (m : ℝ) / ((n : ℝ) + (m : ℝ)) = 3 / 5 :=
sorry

end fraction_of_jumbo_tiles_l518_51802


namespace jaewoong_ran_the_most_l518_51835

def distance_jaewoong : ℕ := 20000 -- Jaewoong's distance in meters
def distance_seongmin : ℕ := 2600  -- Seongmin's distance in meters
def distance_eunseong : ℕ := 5000  -- Eunseong's distance in meters

theorem jaewoong_ran_the_most : distance_jaewoong > distance_seongmin ∧ distance_jaewoong > distance_eunseong := by
  sorry

end jaewoong_ran_the_most_l518_51835


namespace real_numbers_correspond_to_number_line_l518_51898

noncomputable def number_line := ℝ

def real_numbers := ℝ

theorem real_numbers_correspond_to_number_line :
  ∀ (p : ℝ), ∃ (r : real_numbers), r = p ∧ ∀ (r : real_numbers), ∃ (p : ℝ), p = r :=
by
  sorry

end real_numbers_correspond_to_number_line_l518_51898


namespace solve_equation_l518_51803

/-- 
  Given the equation:
    ∀ x, (x = 2 ∨ (3 < x ∧ x < 4)) ↔ (⌊(1/x) * ⌊x⌋^2⌋ = 2),
  where ⌊u⌋ represents the greatest integer less than or equal to u.
-/
theorem solve_equation (x : ℝ) : (x = 2 ∨ (3 < x ∧ x < 4)) ↔ ⌊(1/x) * ⌊x⌋^2⌋ = 2 := 
sorry

end solve_equation_l518_51803


namespace jan_drives_more_miles_than_ian_l518_51882

-- Definitions of conditions
variables (s t d m: ℝ)

-- Ian's travel equation
def ian_distance := d = s * t

-- Han's travel equation
def han_distance := (d + 115) = (s + 8) * (t + 2)

-- Jan's travel equation
def jan_distance := m = (s + 12) * (t + 3)

-- The proof statement we want to prove
theorem jan_drives_more_miles_than_ian :
    (∀ (s t d m : ℝ),
    d = s * t →
    (d + 115) = (s + 8) * (t + 2) →
    m = (s + 12) * (t + 3) →
    (m - d) = 184.5) :=
    sorry

end jan_drives_more_miles_than_ian_l518_51882


namespace problem1_problem2_l518_51888

-- Problem 1
theorem problem1 : ((- (1/2) - (1/3) + (3/4)) * -60) = 5 :=
by
  -- The proof steps would go here
  sorry

-- Problem 2
theorem problem2 : ((-1)^4 - (1/6) * (3 - (-3)^2)) = 2 :=
by
  -- The proof steps would go here
  sorry

end problem1_problem2_l518_51888


namespace original_cost_of_car_l518_51868

theorem original_cost_of_car (C : ℝ) 
  (repair_cost : ℝ := 15000)
  (selling_price : ℝ := 64900)
  (profit_percent : ℝ := 13.859649122807017) :
  C = 43837.21 :=
by
  have h1 : C + repair_cost = selling_price - (selling_price - (C + repair_cost)) := by sorry
  have h2 : profit_percent / 100 = (selling_price - (C + repair_cost)) / C := by sorry
  have h3 : C = 43837.21 := by sorry
  exact h3

end original_cost_of_car_l518_51868


namespace proof_mn_eq_9_l518_51847

theorem proof_mn_eq_9 (m n : ℕ) (h1 : 2 * m + n = 8) (h2 : m - n = 1) : m^n = 9 :=
by {
  sorry 
}

end proof_mn_eq_9_l518_51847


namespace proof_custom_operations_l518_51865

def customOp1 (a b : ℕ) : ℕ := a * b / (a + b)
def customOp2 (a b : ℕ) : ℕ := a * a + b * b

theorem proof_custom_operations :
  customOp2 (customOp1 7 14) 2 = 200 := 
by 
  sorry

end proof_custom_operations_l518_51865


namespace bob_repayment_days_l518_51872

theorem bob_repayment_days :
  ∃ (x : ℕ), (15 + 3 * x ≥ 45) ∧ (∀ y : ℕ, (15 + 3 * y ≥ 45) → x ≤ y) ∧ x = 10 := 
by
  sorry

end bob_repayment_days_l518_51872


namespace squirrel_population_difference_l518_51864

theorem squirrel_population_difference :
  ∀ (total_population scotland_population rest_uk_population : ℕ), 
  scotland_population = 120000 →
  120000 = 75 * total_population / 100 →
  rest_uk_population = total_population - scotland_population →
  scotland_population - rest_uk_population = 80000 :=
by
  intros total_population scotland_population rest_uk_population h1 h2 h3
  sorry

end squirrel_population_difference_l518_51864


namespace direct_proportion_conditions_l518_51818

theorem direct_proportion_conditions (k b : ℝ) : 
  (y = (k - 4) * x + b → (k ≠ 4 ∧ b = 0)) ∧ ¬ (b ≠ 0 ∨ k ≠ 4) :=
sorry

end direct_proportion_conditions_l518_51818


namespace jordan_length_eq_six_l518_51830

def carol_length := 12
def carol_width := 15
def jordan_width := 30

theorem jordan_length_eq_six
  (h1 : carol_length * carol_width = jordan_width * jordan_length) : 
  jordan_length = 6 := by
  sorry

end jordan_length_eq_six_l518_51830


namespace susan_min_packages_l518_51850

theorem susan_min_packages (n : ℕ) (cost_per_package : ℕ := 5) (earnings_per_package : ℕ := 15) (initial_cost : ℕ := 1200) :
  15 * n - 5 * n ≥ 1200 → n ≥ 120 :=
by {
  sorry -- Proof goes here
}

end susan_min_packages_l518_51850


namespace compute_binom_value_l518_51841

noncomputable def binom (x : ℝ) (k : ℕ) : ℝ :=
  if k = 0 then 1 else x * binom (x - 1) (k - 1) / k

theorem compute_binom_value : 
  (binom (1/2) 2014 * 4^2014 / binom 4028 2014) = -1/4027 :=
by 
  sorry

end compute_binom_value_l518_51841


namespace find_difference_l518_51885

variable (f : ℝ → ℝ)

-- Conditions
axiom linear_f : ∀ x y a b, f (a * x + b * y) = a * f x + b * f y
axiom f_difference : f 6 - f 2 = 12

theorem find_difference : f 12 - f 2 = 30 :=
by
  sorry

end find_difference_l518_51885


namespace max_det_A_l518_51807

open Real

-- Define the matrix and the determinant expression
noncomputable def A (θ : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![
    ![1, 1, 1],
    ![1, 1 + cos θ, 1],
    ![1 + sin θ, 1, 1]
  ]

-- Lean statement to prove the maximum value of the determinant of matrix A
theorem max_det_A : ∃ θ : ℝ, (Matrix.det (A θ)) ≤ 1/2 := by
  sorry

end max_det_A_l518_51807


namespace minimum_combined_horses_ponies_l518_51883

noncomputable def ranch_min_total (P H : ℕ) : ℕ :=
  P + H

theorem minimum_combined_horses_ponies (P H : ℕ) 
  (h1 : ∃ k : ℕ, P = 16 * k ∧ k ≥ 1)
  (h2 : H = P + 3) 
  (h3 : P = 80) 
  (h4 : H = 83) :
  ranch_min_total P H = 163 :=
by
  sorry

end minimum_combined_horses_ponies_l518_51883


namespace oscar_leap_vs_elmer_stride_l518_51858

/--
Given:
1. The 51st telephone pole is exactly 6600 feet from the first pole.
2. Elmer the emu takes 50 equal strides to walk between consecutive telephone poles.
3. Oscar the ostrich can cover the same distance in 15 equal leaps.
4. There are 50 gaps between the 51 poles.

Prove:
Oscar's leap is 6 feet longer than Elmer's stride.
-/
theorem oscar_leap_vs_elmer_stride : 
  let total_distance := 6600 
  let elmer_strides_per_gap := 50
  let oscar_leaps_per_gap := 15
  let num_gaps := 50
  let elmer_total_strides := elmer_strides_per_gap * num_gaps
  let oscar_total_leaps := oscar_leaps_per_gap * num_gaps
  let elmer_stride_length := total_distance / elmer_total_strides
  let oscar_leap_length := total_distance / oscar_total_leaps
  oscar_leap_length - elmer_stride_length = 6 := 
by {
  -- The proof would go here.
  sorry
}

end oscar_leap_vs_elmer_stride_l518_51858


namespace no_prime_satisfies_condition_l518_51837

theorem no_prime_satisfies_condition (p : ℕ) (hp : Nat.Prime p) : 
  ¬ ∃ n : ℕ, 0 < n ∧ ∃ k : ℕ, (Real.sqrt (p + n) + Real.sqrt n) = k :=
by
  sorry

end no_prime_satisfies_condition_l518_51837


namespace no_intersection_points_l518_51869

theorem no_intersection_points : ¬ ∃ x y : ℝ, y = x ∧ y = x - 2 := by
  sorry

end no_intersection_points_l518_51869


namespace convex_polygon_interior_angle_l518_51839

theorem convex_polygon_interior_angle (n : ℕ) (h1 : 3 ≤ n)
  (h2 : (n - 2) * 180 = 2570 + x) : x = 130 :=
sorry

end convex_polygon_interior_angle_l518_51839


namespace mushroom_pickers_l518_51819

theorem mushroom_pickers (n : ℕ) (hn : n = 18) (total_mushrooms : ℕ) (h_total : total_mushrooms = 162) (h_each : ∀ i : ℕ, i < n → 0 < 1) : 
  ∃ i j : ℕ, i < n ∧ j < n ∧ i ≠ j ∧ (total_mushrooms / n = (total_mushrooms / n)) :=
sorry

end mushroom_pickers_l518_51819


namespace find_number_of_shorts_l518_51814

def price_of_shorts : ℕ := 7
def price_of_shoes : ℕ := 20
def total_spent : ℕ := 75

-- We represent the price of 4 tops as a variable
variable (T : ℕ)

theorem find_number_of_shorts (S : ℕ) (h : 7 * S + 4 * T + 20 = 75) : S = 7 :=
by
  sorry

end find_number_of_shorts_l518_51814

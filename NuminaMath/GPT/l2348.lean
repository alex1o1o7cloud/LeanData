import Mathlib

namespace least_integer_value_l2348_234855

-- Define the condition and then prove the statement
theorem least_integer_value (x : ℤ) (h : 3 * |x| - 2 > 13) : x = -6 :=
by
  sorry

end least_integer_value_l2348_234855


namespace train_cross_time_approx_l2348_234888

noncomputable def length_of_train : ℝ := 100
noncomputable def speed_of_train_km_hr : ℝ := 80
noncomputable def length_of_bridge : ℝ := 142
noncomputable def total_distance : ℝ := length_of_train + length_of_bridge
noncomputable def speed_of_train_m_s : ℝ := speed_of_train_km_hr * 1000 / 3600
noncomputable def time_to_cross_bridge : ℝ := total_distance / speed_of_train_m_s

theorem train_cross_time_approx :
  abs (time_to_cross_bridge - 10.89) < 0.01 :=
by
  sorry

end train_cross_time_approx_l2348_234888


namespace general_term_of_sequence_l2348_234806

theorem general_term_of_sequence (S : ℕ → ℤ) (a : ℕ → ℤ) :
  (∀ n, S (n + 1) = 3 * (n + 1) ^ 2 - 2 * (n + 1)) →
  a 1 = 1 →
  (∀ n, a (n + 1) = S (n + 1) - S n) →
  (∀ n, a n = 6 * n - 5) := 
by
  intros hS ha1 ha
  sorry

end general_term_of_sequence_l2348_234806


namespace part_a_part_b_l2348_234849

-- Define the conditions for part (a)
def psychic_can_guess_at_least_19_cards : Prop :=
  ∃ (deck : Fin 36 → Fin 4) (psychic_guess : Fin 36 → Fin 4)
    (assistant_arrangement : Fin 36 → Bool),
    -- assistant and psychic agree on a method ensuring at least 19 correct guesses
    (∃ n : ℕ, n ≥ 19 ∧
      ∃ correct_guesses_set : Finset (Fin 36),
        correct_guesses_set.card = n ∧
        ∀ i ∈ correct_guesses_set, psychic_guess i = deck i)

-- Prove that the above condition is satisfied
theorem part_a : psychic_can_guess_at_least_19_cards :=
by
  sorry

-- Define the conditions for part (b)
def psychic_can_guess_at_least_23_cards : Prop :=
  ∃ (deck : Fin 36 → Fin 4) (psychic_guess : Fin 36 → Fin 4)
    (assistant_arrangement : Fin 36 → Bool),
    -- assistant and psychic agree on a method ensuring at least 23 correct guesses
    (∃ n : ℕ, n ≥ 23 ∧
      ∃ correct_guesses_set : Finset (Fin 36),
        correct_guesses_set.card = n ∧
        ∀ i ∈ correct_guesses_set, psychic_guess i = deck i)

-- Prove that the above condition is satisfied
theorem part_b : psychic_can_guess_at_least_23_cards :=
by
  sorry

end part_a_part_b_l2348_234849


namespace determine_a_l2348_234895

theorem determine_a (a : ℝ) (h : 2 * (-1) + a = 3) : a = 5 := sorry

end determine_a_l2348_234895


namespace sum_first_13_terms_l2348_234836

theorem sum_first_13_terms
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ)
  (h₀ : ∀ n : ℕ, S n = n * (a 1 + a n) / 2)
  (h₁ : a 4 + a 10 - (a 7)^2 + 15 = 0)
  (h₂ : ∀ n : ℕ, a n > 0) :
  S 13 = 65 :=
sorry

end sum_first_13_terms_l2348_234836


namespace problem_correct_l2348_234864

theorem problem_correct (x : ℝ) : 
  14 * ((150 / 3) + (35 / 7) + (16 / 32) + x) = 777 + 14 * x := 
by
  sorry

end problem_correct_l2348_234864


namespace haylee_has_36_guppies_l2348_234850

variables (H J C N : ℝ)
variables (total_guppies : ℝ := 84)

def jose_has_half_of_haylee := J = H / 2
def charliz_has_third_of_jose := C = J / 3
def nicolai_has_four_times_charliz := N = 4 * C
def total_guppies_eq_84 := H + J + C + N = total_guppies

theorem haylee_has_36_guppies 
  (hJ : jose_has_half_of_haylee H J)
  (hC : charliz_has_third_of_jose J C)
  (hN : nicolai_has_four_times_charliz C N)
  (htotal : total_guppies_eq_84 H J C N) :
  H = 36 := 
  sorry

end haylee_has_36_guppies_l2348_234850


namespace heart_op_ratio_l2348_234841

def heart_op (n m : ℕ) : ℕ := n^3 * m^2

theorem heart_op_ratio : heart_op 3 5 / heart_op 5 3 = 5 / 9 := 
by 
  sorry

end heart_op_ratio_l2348_234841


namespace vector_expression_simplification_l2348_234800

variable (a b : Type)
variable (α : Type) [Field α]
variable [AddCommGroup a] [Module α a]

theorem vector_expression_simplification
  (vector_a vector_b : a) :
  (1/3 : α) • (vector_a - (2 : α) • vector_b) + vector_b = (1/3 : α) • vector_a + (1/3 : α) • vector_b :=
by
  sorry

end vector_expression_simplification_l2348_234800


namespace power_function_half_l2348_234827

theorem power_function_half (a : ℝ) (ha : (4 : ℝ)^a / (2 : ℝ)^a = 3) : (1 / 2 : ℝ) ^ a = 1 / 3 := 
by
  sorry

end power_function_half_l2348_234827


namespace find_min_value_l2348_234899

noncomputable def min_value (a b : ℝ) (h_a : a > 0) (h_b : b > 0) (h_slope : 2 * a + b = 1) :=
  (8 * a + b) / (a * b)

theorem find_min_value (a b : ℝ) (h_a : a > 0) (h_b : b > 0) (h_slope : 2 * a + b = 1) :
  min_value a b h_a h_b h_slope = 18 :=
sorry

end find_min_value_l2348_234899


namespace johnnyMoneyLeft_l2348_234851

noncomputable def johnnySavingsSeptember : ℝ := 30
noncomputable def johnnySavingsOctober : ℝ := 49
noncomputable def johnnySavingsNovember : ℝ := 46
noncomputable def johnnySavingsDecember : ℝ := 55

noncomputable def johnnySavingsJanuary : ℝ := johnnySavingsDecember * 1.15

noncomputable def totalSavings : ℝ := johnnySavingsSeptember + johnnySavingsOctober + johnnySavingsNovember + johnnySavingsDecember + johnnySavingsJanuary

noncomputable def videoGameCost : ℝ := 58
noncomputable def bookCost : ℝ := 25
noncomputable def birthdayPresentCost : ℝ := 40

noncomputable def totalSpent : ℝ := videoGameCost + bookCost + birthdayPresentCost

noncomputable def moneyLeft : ℝ := totalSavings - totalSpent

theorem johnnyMoneyLeft : moneyLeft = 120.25 := by
  sorry

end johnnyMoneyLeft_l2348_234851


namespace monotonicity_m_eq_zero_range_of_m_l2348_234872

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := Real.exp x - m * x^2 - 2 * x

theorem monotonicity_m_eq_zero :
  ∀ x : ℝ, (x < Real.log 2 → f x 0 < f (x + 1) 0) ∧ (x > Real.log 2 → f x 0 > f (x - 1) 0) := 
sorry

theorem range_of_m :
  ∀ x : ℝ, x ∈ Set.Ici 0 → f x m > (Real.exp 1 / 2 - 1) → m < (Real.exp 1 / 2 - 1) := 
sorry

end monotonicity_m_eq_zero_range_of_m_l2348_234872


namespace marathon_distance_l2348_234894

theorem marathon_distance (d_1 : ℕ) (n : ℕ) (h1 : d_1 = 3) (h2 : n = 5): 
  (2 ^ (n - 1)) * d_1 = 48 :=
by
  sorry

end marathon_distance_l2348_234894


namespace find_percentage_decrease_in_fourth_month_l2348_234811

theorem find_percentage_decrease_in_fourth_month
  (P0 : ℝ) (P1 : ℝ) (P2 : ℝ) (P3 : ℝ) (x : ℝ) :
  (P0 = 100) →
  (P1 = P0 + 0.30 * P0) →
  (P2 = P1 - 0.15 * P1) →
  (P3 = P2 + 0.10 * P2) →
  (P0 = P3 - x / 100 * P3) →
  x = 18 :=
by
  sorry

end find_percentage_decrease_in_fourth_month_l2348_234811


namespace symmetric_y_axis_l2348_234852

theorem symmetric_y_axis (a b : ℝ) (h₁ : a = -4) (h₂ : b = 3) : a - b = -7 :=
by
  rw [h₁, h₂]
  norm_num

end symmetric_y_axis_l2348_234852


namespace students_left_early_l2348_234896

theorem students_left_early :
  let initial_groups := 3
  let students_per_group := 8
  let students_remaining := 22
  let total_students := initial_groups * students_per_group
  total_students - students_remaining = 2 :=
by
  -- Define the initial conditions
  let initial_groups := 3
  let students_per_group := 8
  let students_remaining := 22
  let total_students := initial_groups * students_per_group
  -- Proof (to be completed)
  sorry

end students_left_early_l2348_234896


namespace probability_distribution_m_l2348_234825

theorem probability_distribution_m (m : ℚ) : 
  (m + m / 2 + m / 3 + m / 4 = 1) → m = 12 / 25 :=
by sorry

end probability_distribution_m_l2348_234825


namespace simplify_expr_l2348_234802

theorem simplify_expr : (1 / (1 + Real.sqrt 3)) * (1 / (1 - Real.sqrt 3)) = -1 / 2 :=
by
  sorry

end simplify_expr_l2348_234802


namespace simplify_and_evaluate_expression_l2348_234830

theorem simplify_and_evaluate_expression (x y : ℝ) (h₁ : x = 2) (h₂ : y = -1) : 
  2 * x * y - (1 / 2) * (4 * x * y - 8 * x^2 * y^2) + 2 * (3 * x * y - 5 * x^2 * y^2) = -36 := by
  sorry

end simplify_and_evaluate_expression_l2348_234830


namespace percent_of_value_l2348_234885

theorem percent_of_value : (2 / 5) * (1 / 100) * 450 = 1.8 :=
by sorry

end percent_of_value_l2348_234885


namespace triangle_inequality_holds_l2348_234812

theorem triangle_inequality_holds (a b c : ℝ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) :
  a^3 + b^3 + c^3 + 4 * a * b * c ≤ (9 / 32) * (a + b + c)^3 :=
by {
  sorry
}

end triangle_inequality_holds_l2348_234812


namespace dan_blue_marbles_l2348_234840

variable (m d : ℕ)
variable (h1 : m = 2 * d)
variable (h2 : m = 10)

theorem dan_blue_marbles : d = 5 :=
by
  sorry

end dan_blue_marbles_l2348_234840


namespace net_gain_loss_l2348_234861

-- Definitions of the initial conditions
structure InitialState :=
  (cash_x : ℕ) (painting_value : ℕ) (cash_y : ℕ)

-- Definitions of transactions
structure Transaction :=
  (sell_price : ℕ) (commission_rate : ℕ)

def apply_transaction (initial_cash : ℕ) (tr : Transaction) : ℕ :=
  initial_cash + (tr.sell_price - (tr.sell_price * tr.commission_rate / 100))

def revert_transaction (initial_cash : ℕ) (tr : Transaction) : ℕ :=
  initial_cash - tr.sell_price + (tr.sell_price * tr.commission_rate / 100)

def compute_final_cash (initial_states : InitialState) (trans1 : Transaction) (trans2 : Transaction) : ℕ :=
  let cash_x_after_first := apply_transaction initial_states.cash_x trans1
  let cash_y_after_first := initial_states.cash_y - trans1.sell_price
  let cash_x_after_second := revert_transaction cash_x_after_first trans2
  let cash_y_after_second := cash_y_after_first + (trans2.sell_price - (trans2.sell_price * trans2.commission_rate / 100))
  cash_x_after_second - initial_states.cash_x + (cash_y_after_second - initial_states.cash_y)

-- Statement of the theorem
theorem net_gain_loss (initial_states : InitialState) (trans1 : Transaction) (trans2 : Transaction)
  (h1 : initial_states.cash_x = 15000)
  (h2 : initial_states.painting_value = 15000)
  (h3 : initial_states.cash_y = 18000)
  (h4 : trans1.sell_price = 20000)
  (h5 : trans1.commission_rate = 5)
  (h6 : trans2.sell_price = 14000)
  (h7 : trans2.commission_rate = 5) : 
  compute_final_cash initial_states trans1 trans2 = 5000 - 6700 :=
sorry

end net_gain_loss_l2348_234861


namespace factoring_difference_of_squares_l2348_234890

theorem factoring_difference_of_squares (a : ℝ) : a^2 - 9 = (a + 3) * (a - 3) := 
sorry

end factoring_difference_of_squares_l2348_234890


namespace largest_N_l2348_234891

-- Definition of the problem conditions
def problem_conditions (n : ℕ) (N : ℕ) (a : Fin (N + 1) → ℝ) : Prop :=
  (n ≥ 2) ∧
  (a 0 + a 1 = -(1 : ℝ) / n) ∧  
  (∀ k : ℕ, 1 ≤ k → k ≤ N - 1 → (a k + a (k - 1)) * (a k + a (k + 1)) = a (k - 1) - a (k + 1))

-- The theorem stating that the largest integer N is n
theorem largest_N (n : ℕ) (N : ℕ) (a : Fin (N + 1) → ℝ) :
  problem_conditions n N a → N = n :=
sorry

end largest_N_l2348_234891


namespace rectangle_perimeter_of_right_triangle_l2348_234881

noncomputable def right_triangle_area (a b: ℕ) : ℝ := (1/2 : ℝ) * a * b

noncomputable def rectangle_length (area width: ℝ) : ℝ := area / width

noncomputable def rectangle_perimeter (length width: ℝ) : ℝ := 2 * (length + width)

theorem rectangle_perimeter_of_right_triangle :
  rectangle_perimeter (rectangle_length (right_triangle_area 7 24) 5) 5 = 43.6 :=
by
  sorry

end rectangle_perimeter_of_right_triangle_l2348_234881


namespace eliminate_denominators_l2348_234846

theorem eliminate_denominators (x : ℝ) :
  (6 : ℝ) * ((x - 1) / 3) = (6 : ℝ) * (4 - (2 * x + 1) / 2) ↔ 2 * (x - 1) = 24 - 3 * (2 * x + 1) :=
by
  intros
  sorry

end eliminate_denominators_l2348_234846


namespace hyperbola_circle_intersection_l2348_234823

open Real

theorem hyperbola_circle_intersection (a r : ℝ) (P Q R S : ℝ × ℝ) 
  (hP : P.1^2 - P.2^2 = a^2) (hQ : Q.1^2 - Q.2^2 = a^2) (hR : R.1^2 - R.2^2 = a^2) (hS : S.1^2 - S.2^2 = a^2)
  (hO : r ≥ 0)
  (hPQRS : (P.1 - 0)^2 + (P.2 - 0)^2 = r^2 ∧
            (Q.1 - 0)^2 + (Q.2 - 0)^2 = r^2 ∧
            (R.1 - 0)^2 + (R.2 - 0)^2 = r^2 ∧
            (S.1 - 0)^2 + (S.2 - 0)^2 = r^2) : 
  (P.1^2 + P.2^2) + (Q.1^2 + Q.2^2) + (R.1^2 + R.2^2) + (S.1^2 + S.2^2) = 4 * r^2 :=
by
  sorry

end hyperbola_circle_intersection_l2348_234823


namespace analysis_hours_l2348_234887

-- Define the conditions: number of bones and minutes per bone
def number_of_bones : Nat := 206
def minutes_per_bone : Nat := 45

-- Define the conversion factor: minutes per hour
def minutes_per_hour : Nat := 60

-- Define the total minutes spent analyzing all bones
def total_minutes (number_of_bones minutes_per_bone : Nat) : Nat :=
  number_of_bones * minutes_per_bone

-- Define the total hours required for analysis
def total_hours (total_minutes minutes_per_hour : Nat) : Float :=
  total_minutes.toFloat / minutes_per_hour.toFloat

-- Prove that total_hours equals 154.5 hours
theorem analysis_hours : total_hours (total_minutes number_of_bones minutes_per_bone) minutes_per_hour = 154.5 := by
  sorry

end analysis_hours_l2348_234887


namespace gcd_problem_l2348_234842

theorem gcd_problem :
  ∃ n : ℕ, (80 ≤ n) ∧ (n ≤ 100) ∧ (n % 9 = 0) ∧ (Nat.gcd n 27 = 9) ∧ (n = 90) :=
by sorry

end gcd_problem_l2348_234842


namespace false_statement_l2348_234868

theorem false_statement :
  ¬ (∀ x : ℝ, x^2 + 1 > 3 * x) = (∃ x : ℝ, x^2 + 1 ≤ 3 * x) := sorry

end false_statement_l2348_234868


namespace soldiers_first_side_l2348_234875

theorem soldiers_first_side (x : ℤ) (h1 : ∀ s1 : ℤ, s1 = 10)
                           (h2 : ∀ s2 : ℤ, s2 = 8)
                           (h3 : ∀ y : ℤ, y = x - 500)
                           (h4 : (10 * x + 8 * (x - 500)) = 68000) : x = 4000 :=
by
  -- Left blank for Lean to fill in the required proof steps
  sorry

end soldiers_first_side_l2348_234875


namespace Anya_walks_to_school_l2348_234843

theorem Anya_walks_to_school
  (t_f t_b : ℝ)
  (h1 : t_f + t_b = 1.5)
  (h2 : 2 * t_b = 0.5) :
  2 * t_f = 2.5 :=
by
  -- The proof details will go here eventually.
  sorry

end Anya_walks_to_school_l2348_234843


namespace relationship_among_abc_l2348_234818

noncomputable def a : ℝ := 20.3
noncomputable def b : ℝ := 0.32
noncomputable def c : ℝ := Real.log 25 / Real.log 10

theorem relationship_among_abc : b < a ∧ a < c :=
by
  -- Proof needs to be filled in here
  sorry

end relationship_among_abc_l2348_234818


namespace percent_decrease_l2348_234831

-- Definitions based on conditions
def originalPrice : ℝ := 100
def salePrice : ℝ := 10

-- The percentage decrease is the main statement to prove
theorem percent_decrease : ((originalPrice - salePrice) / originalPrice) * 100 = 90 := 
by
  -- Placeholder for proof
  sorry

end percent_decrease_l2348_234831


namespace factor_x4_plus_64_monic_real_l2348_234838

theorem factor_x4_plus_64_monic_real :
  ∀ x : ℝ, x^4 + 64 = (x^2 + 4 * x + 8) * (x^2 - 4 * x + 8) := 
by
  intros
  sorry

end factor_x4_plus_64_monic_real_l2348_234838


namespace combined_mpg_is_30_l2348_234889

-- Define the constants
def ray_efficiency : ℕ := 50 -- miles per gallon
def tom_efficiency : ℕ := 25 -- miles per gallon
def ray_distance : ℕ := 100 -- miles
def tom_distance : ℕ := 200 -- miles

-- Define the combined miles per gallon calculation and the proof statement.
theorem combined_mpg_is_30 :
  (ray_distance + tom_distance) /
  ((ray_distance / ray_efficiency) + (tom_distance / tom_efficiency)) = 30 :=
by
  -- All proof steps are skipped using sorry
  sorry

end combined_mpg_is_30_l2348_234889


namespace boat_speed_in_still_water_l2348_234805

/-- Prove the speed of the boat in still water given the conditions -/
theorem boat_speed_in_still_water (V_s : ℝ) (T : ℝ) (D : ℝ) (V_b : ℝ) :
  V_s = 4 ∧ T = 4 ∧ D = 112 ∧ (D / T = V_b + V_s) → V_b = 24 := sorry

end boat_speed_in_still_water_l2348_234805


namespace choir_members_unique_l2348_234893

theorem choir_members_unique (n : ℕ) :
  (n % 10 = 6) ∧ 
  (n % 11 = 6) ∧ 
  (150 ≤ n) ∧ 
  (n ≤ 300) → 
  n = 226 := 
by
  sorry

end choir_members_unique_l2348_234893


namespace abs_inequality_range_l2348_234810

theorem abs_inequality_range (x : ℝ) (b : ℝ) (h : 0 < b) : (b > 2) ↔ ∃ x : ℝ, |x - 5| + |x - 7| < b :=
sorry

end abs_inequality_range_l2348_234810


namespace illumination_ways_l2348_234876

def ways_to_illuminate_traffic_lights (n : ℕ) : ℕ :=
  3^n

theorem illumination_ways (n : ℕ) : ways_to_illuminate_traffic_lights n = 3 ^ n :=
by
  sorry

end illumination_ways_l2348_234876


namespace ed_lost_seven_marbles_l2348_234826

theorem ed_lost_seven_marbles (D L : ℕ) (h1 : ∃ (Ed_init Tim_init : ℕ), Ed_init = D + 19 ∧ Tim_init = D - 10)
(h2 : ∃ (Ed_final Tim_final : ℕ), Ed_final = D + 19 - L - 4 ∧ Tim_final = D - 10 + 4 + 3)
(h3 : ∀ (Ed_final : ℕ), Ed_final = D + 8)
(h4 : ∀ (Tim_final : ℕ), Tim_final = D):
  L = 7 :=
by
  sorry

end ed_lost_seven_marbles_l2348_234826


namespace minimum_abs_a_l2348_234880

-- Given conditions as definitions
def has_integer_coeffs (a b c : ℤ) : Prop := true
def has_roots_in_range (a b c : ℤ) (x1 x2 : ℚ) : Prop :=
  x1 ≠ x2 ∧ 0 < x1 ∧ x1 < 1 ∧ 0 < x2 ∧ x2 < 1 ∧
  (a : ℚ) * x1^2 + (b : ℚ) * x1 + (c : ℚ) = 0 ∧
  (a : ℚ) * x2^2 + (b : ℚ) * x2 + (c : ℚ) = 0

-- Main statement (abstractly mentioning existence of x1, x2 such that they fulfill the polynomial conditions)
theorem minimum_abs_a (a b c : ℤ) (x1 x2 : ℚ) :
  has_integer_coeffs a b c →
  has_roots_in_range a b c x1 x2 →
  |a| ≥ 5 :=
by
  intros _ _
  sorry

end minimum_abs_a_l2348_234880


namespace mark_total_young_fish_l2348_234801

-- Define the conditions
def num_tanks : ℕ := 5
def fish_per_tank : ℕ := 6
def young_per_fish : ℕ := 25

-- Define the total number of young fish
def total_young_fish := num_tanks * fish_per_tank * young_per_fish

-- The theorem statement
theorem mark_total_young_fish : total_young_fish = 750 :=
  by
    sorry

end mark_total_young_fish_l2348_234801


namespace decimal_equivalent_one_half_pow_five_l2348_234815

theorem decimal_equivalent_one_half_pow_five :
  (1 / 2) ^ 5 = 0.03125 :=
by sorry

end decimal_equivalent_one_half_pow_five_l2348_234815


namespace smallest_angle_of_triangle_l2348_234807

theorem smallest_angle_of_triangle (y : ℝ) (h : 40 + 70 + y = 180) : 
  ∃ smallest_angle : ℝ, smallest_angle = 40 ∧ smallest_angle = min 40 (min 70 y) := 
by
  use 40
  sorry

end smallest_angle_of_triangle_l2348_234807


namespace JackOfHeartsIsSane_l2348_234860

inductive Card
  | Ace
  | Two
  | Three
  | Four
  | Five
  | Six
  | Seven
  | JackOfHearts

open Card

def Sane (c : Card) : Prop := sorry

axiom Condition1 : Sane Three → ¬ Sane Ace
axiom Condition2 : Sane Four → (¬ Sane Three ∨ ¬ Sane Two)
axiom Condition3 : Sane Five → (Sane Ace ↔ Sane Four)
axiom Condition4 : Sane Six → (Sane Ace ∧ Sane Two)
axiom Condition5 : Sane Seven → ¬ Sane Five
axiom Condition6 : Sane JackOfHearts → (¬ Sane Six ∨ ¬ Sane Seven)

theorem JackOfHeartsIsSane : Sane JackOfHearts := by
  sorry

end JackOfHeartsIsSane_l2348_234860


namespace determine_CD_l2348_234870

theorem determine_CD (AB : ℝ) (BD : ℝ) (BC : ℝ) (CD : ℝ) (Angle_ADB : ℝ)
  (sin_A : ℝ) (sin_C : ℝ)
  (h1 : AB = 30)
  (h2 : Angle_ADB = 90)
  (h3 : sin_A = 4/5)
  (h4 : sin_C = 1/5)
  (h5 : BD = sin_A * AB)
  (h6 : BC = BD / sin_C) :
  CD = 24 * Real.sqrt 23 := by
  sorry

end determine_CD_l2348_234870


namespace trackball_mice_count_l2348_234859

theorem trackball_mice_count
  (total_mice : ℕ)
  (wireless_fraction : ℕ)
  (optical_fraction : ℕ)
  (h_total : total_mice = 80)
  (h_wireless : wireless_fraction = total_mice / 2)
  (h_optical : optical_fraction = total_mice / 4) :
  total_mice - (wireless_fraction + optical_fraction) = 20 :=
sorry

end trackball_mice_count_l2348_234859


namespace pass_rate_correct_l2348_234863

variable {a b : ℝ}

-- Assumptions: defect rates are between 0 and 1
axiom h_a : 0 ≤ a ∧ a ≤ 1
axiom h_b : 0 ≤ b ∧ b ≤ 1

-- Definition: Pass rate is 1 minus the defect rate
def pass_rate (a b : ℝ) : ℝ := (1 - a) * (1 - b)

-- Theorem: Proving the pass rate is (1 - a) * (1 - b)
theorem pass_rate_correct : pass_rate a b = (1 - a) * (1 - b) := 
by
  sorry

end pass_rate_correct_l2348_234863


namespace length_of_boat_l2348_234865

-- Definitions based on the conditions
def breadth : ℝ := 3
def sink_depth : ℝ := 0.01
def man_mass : ℝ := 120
def g : ℝ := 9.8 -- acceleration due to gravity

-- Derived from the conditions
def weight_man : ℝ := man_mass * g
def density_water : ℝ := 1000

-- Statement to be proved
theorem length_of_boat : ∃ L : ℝ, (breadth * sink_depth * L * density_water * g = weight_man) → L = 4 :=
by
  sorry

end length_of_boat_l2348_234865


namespace sum_of_nonnegative_reals_l2348_234877

theorem sum_of_nonnegative_reals (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z)
  (h1 : x^2 + y^2 + z^2 = 52) (h2 : x * y + y * z + z * x = 27) :
  x + y + z = Real.sqrt 106 :=
sorry

end sum_of_nonnegative_reals_l2348_234877


namespace hyperbola_focus_l2348_234874

theorem hyperbola_focus :
    ∃ (f : ℝ × ℝ), f = (-2 - Real.sqrt 6, -2) ∧
    ∀ (x y : ℝ), 2 * x^2 - y^2 + 8 * x - 4 * y - 8 = 0 → 
    ∃ a b h k : ℝ, 
        (a = Real.sqrt 2) ∧ (b = 2) ∧ (h = -2) ∧ (k = -2) ∧
        ((2 * (x + h)^2 - (y + k)^2 = 4) ∧ 
         (x, y) = f) :=
sorry

end hyperbola_focus_l2348_234874


namespace arithmetic_progression_25th_term_l2348_234813

def arithmetic_progression_nth_term (a₁ d n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

theorem arithmetic_progression_25th_term : arithmetic_progression_nth_term 5 7 25 = 173 := by
  sorry

end arithmetic_progression_25th_term_l2348_234813


namespace solve_system_l2348_234834

def system_of_equations (x y : ℝ) : Prop :=
  (4 * (x - y) = 8 - 3 * y) ∧ (x / 2 + y / 3 = 1)

theorem solve_system : ∃ x y : ℝ, system_of_equations x y ∧ x = 2 ∧ y = 0 := 
  by
  sorry

end solve_system_l2348_234834


namespace min_value_sum_pos_int_l2348_234829

theorem min_value_sum_pos_int 
  (a b c : ℕ)
  (h_pos: a > 0 ∧ b > 0 ∧ c > 0)
  (h_roots: ∃ (A B : ℝ), A < 0 ∧ A > -1 ∧ B > 0 ∧ B < 1 ∧ (∀ x : ℝ, x^2*x*a + x*b + c = 0 → x = A ∨ x = B))
  : a + b + c = 11 :=
sorry

end min_value_sum_pos_int_l2348_234829


namespace proof_problem_l2348_234866

def A : Set ℝ := {x | x < 4}
def B : Set ℝ := {x | x^2 - 4 * x + 3 > 0}

theorem proof_problem : {x | x ∈ A ∧ x ∉ (A ∩ B)} = {x | 1 ≤ x ∧ x ≤ 3} :=
by {
  sorry
}

end proof_problem_l2348_234866


namespace arithmetic_sequence_1005th_term_l2348_234856

theorem arithmetic_sequence_1005th_term (p r : ℤ) 
  (h1 : 11 = p + 2 * r)
  (h2 : 11 + 2 * r = 4 * p - r) :
  (5 + 1004 * 6) = 6029 :=
by
  sorry

end arithmetic_sequence_1005th_term_l2348_234856


namespace alpha_range_l2348_234879

theorem alpha_range (α : ℝ) (hα : 0 < α ∧ α < 2 * Real.pi) : 
  (Real.sin α < Real.sqrt 3 / 2 ∧ Real.cos α > 1 / 2) ↔ 
  (0 < α ∧ α < Real.pi / 3 ∨ 5 * Real.pi / 3 < α ∧ α < 2 * Real.pi) := 
sorry

end alpha_range_l2348_234879


namespace abs_algebraic_expression_l2348_234832

theorem abs_algebraic_expression (x : ℝ) (h : |2 * x - 3| - 3 + 2 * x = 0) : |2 * x - 5| = 5 - 2 * x := 
by sorry

end abs_algebraic_expression_l2348_234832


namespace min_vans_proof_l2348_234854

-- Define the capacity and availability of each type of van
def capacity_A : Nat := 7
def capacity_B : Nat := 9
def capacity_C : Nat := 12

def available_A : Nat := 3
def available_B : Nat := 4
def available_C : Nat := 2

-- Define the number of people going on the trip
def students : Nat := 40
def adults : Nat := 14

-- Define the total number of people
def total_people : Nat := students + adults

-- Define the minimum number of vans needed
def min_vans_needed : Nat := 6

-- Define the number of each type of van used
def vans_A_used : Nat := 0
def vans_B_used : Nat := 4
def vans_C_used : Nat := 2

-- Prove the minimum number of vans needed to accommodate everyone is 6
theorem min_vans_proof : min_vans_needed = 6 ∧ 
  (vans_A_used * capacity_A + vans_B_used * capacity_B + vans_C_used * capacity_C = total_people) ∧
  vans_A_used <= available_A ∧ vans_B_used <= available_B ∧ vans_C_used <= available_C :=
by 
  sorry

end min_vans_proof_l2348_234854


namespace repeating_decimal_fraction_value_l2348_234839

def repeating_decimal_to_fraction (d : ℚ) : ℚ :=
  d

theorem repeating_decimal_fraction_value :
  repeating_decimal_to_fraction (73 / 100 + 246 / 999000) = 731514 / 999900 :=
by
  sorry

end repeating_decimal_fraction_value_l2348_234839


namespace num_values_between_l2348_234892

theorem num_values_between (x y : ℕ) (h1 : x + y ≥ 200) (h2 : x + y ≤ 1000) 
  (h3 : (x * (x - 1) + y * (y - 1)) * 2 = (x + y) * (x + y - 1)) : 
  ∃ n : ℕ, n - 1 = 17 := by
  sorry

end num_values_between_l2348_234892


namespace determine_x_l2348_234873

theorem determine_x (x : ℕ) (hx : 27^3 + 27^3 + 27^3 = 3^x) : x = 10 :=
sorry

end determine_x_l2348_234873


namespace Lily_balls_is_3_l2348_234883

-- Definitions from conditions
variable (L : ℕ)

def Frodo_balls := L + 8
def Brian_balls := 2 * (L + 8)

axiom Brian_has_22 : Brian_balls L = 22

-- The goal is to prove that Lily has 3 tennis balls
theorem Lily_balls_is_3 : L = 3 :=
by
  sorry

end Lily_balls_is_3_l2348_234883


namespace completion_time_B_l2348_234837

-- Definitions based on conditions
def work_rate_A : ℚ := 1 / 10 -- A's rate of completing work per day

def efficiency_B : ℚ := 1.75 -- B is 75% more efficient than A

def work_rate_B : ℚ := efficiency_B * work_rate_A -- B's work rate per day

-- The main theorem that we need to prove
theorem completion_time_B : (1 : ℚ) / work_rate_B = 40 / 7 :=
by 
  sorry

end completion_time_B_l2348_234837


namespace complex_expression_l2348_234844

-- The condition: n is a positive integer
variable (n : ℕ) (hn : 0 < n)

-- Definition of the problem to be proved
theorem complex_expression (n : ℕ) (hn : 0 < n) : 
  (Complex.I ^ (4 * n) + Complex.I ^ (4 * n + 1) + Complex.I ^ (4 * n + 2) + Complex.I ^ (4 * n + 3)) = 0 :=
sorry

end complex_expression_l2348_234844


namespace max_crates_first_trip_l2348_234835

theorem max_crates_first_trip (x : ℕ) : (∀ w, w ≥ 120) ∧ (600 ≥ x * 120) → x = 5 := 
by
  -- Condition: The weight of any crate is no less than 120 kg
  intro h
  have h1 : ∀ w, w ≥ 120 := h.left
  
  -- Condition: The maximum weight for the first trip
  have h2 : 600 ≥ x * 120 := h.right 
  
  -- Derivation of maximum crates
  have h3 : x ≤ 600 / 120 := by sorry  -- This inequality follows from h2 by straightforward division
  
  have h4 : x ≤ 5 := by sorry  -- This follows from evaluating 600 / 120 = 5
  
  -- Knowing x is an integer and the maximum possible value is 5
  exact by sorry

end max_crates_first_trip_l2348_234835


namespace simple_interest_years_l2348_234858

variables (T R : ℝ)

def principal : ℝ := 1000
def additional_interest : ℝ := 90

theorem simple_interest_years
  (H: principal * (R + 3) * T / 100 - principal * R * T / 100 = additional_interest) :
  T = 3 :=
by sorry

end simple_interest_years_l2348_234858


namespace haley_fuel_consumption_ratio_l2348_234884

theorem haley_fuel_consumption_ratio (gallons: ℕ) (miles: ℕ) (h_gallons: gallons = 44) (h_miles: miles = 77) :
  (gallons / Nat.gcd gallons miles) = 4 ∧ (miles / Nat.gcd gallons miles) = 7 :=
by
  sorry

end haley_fuel_consumption_ratio_l2348_234884


namespace ellipse_standard_equation_l2348_234819

theorem ellipse_standard_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : (-4)^2 / a^2 + 3^2 / b^2 = 1) 
    (h4 : a^2 = b^2 + 5^2) : 
    ∃ (a b : ℝ), a^2 = 40 ∧ b^2 = 15 ∧ 
    (∀ x y : ℝ, x^2 / 40 + y^2 / 15 = 1 → (∃ f1 f2 : ℝ, f1 = 5 ∧ f2 = -5)) :=
by {
    sorry
}

end ellipse_standard_equation_l2348_234819


namespace general_term_of_sequence_l2348_234803

noncomputable def a (n : ℕ) : ℝ :=
  if n = 1 then 1 else
  if n = 2 then 2 else
  sorry -- the recurrence relation will go here, but we'll skip its implementation

theorem general_term_of_sequence :
  ∀ n : ℕ, n ≥ 1 → a n = 3 - (2 / n) :=
by sorry

end general_term_of_sequence_l2348_234803


namespace quadratic_roots_l2348_234862

theorem quadratic_roots (k : ℝ) : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 + k*x1 + (k - 1) = 0) ∧ (x2^2 + k*x2 + (k - 1) = 0) :=
by
  sorry

end quadratic_roots_l2348_234862


namespace batsman_average_proof_l2348_234814

noncomputable def batsman_average_after_17th_inning (A : ℝ) : ℝ :=
  (A * 16 + 87) / 17

theorem batsman_average_proof (A : ℝ) (h1 : 16 * A + 87 = 17 * (A + 2)) : batsman_average_after_17th_inning 53 = 55 :=
by
  sorry

end batsman_average_proof_l2348_234814


namespace total_money_l2348_234882

theorem total_money (total_coins nickels dimes : ℕ) (val_nickel val_dime : ℕ)
  (h1 : total_coins = 8)
  (h2 : nickels = 2)
  (h3 : total_coins = nickels + dimes)
  (h4 : val_nickel = 5)
  (h5 : val_dime = 10) :
  (nickels * val_nickel + dimes * val_dime) = 70 :=
by
  sorry

end total_money_l2348_234882


namespace probability_of_snow_at_least_once_first_week_l2348_234869

theorem probability_of_snow_at_least_once_first_week :
  let p_first4 := 1 / 4
  let p_next3 := 1 / 3
  let p_no_snow_first4 := (1 - p_first4) ^ 4
  let p_no_snow_next3 := (1 - p_next3) ^ 3
  let p_no_snow_week := p_no_snow_first4 * p_no_snow_next3
  1 - p_no_snow_week = 29 / 32 :=
by
  sorry

end probability_of_snow_at_least_once_first_week_l2348_234869


namespace jog_time_each_morning_is_1_5_hours_l2348_234809

-- Define the total time Mr. John spent jogging
def total_time_spent_jogging : ℝ := 21

-- Define the number of days Mr. John jogged
def number_of_days_jogged : ℕ := 14

-- Define the time Mr. John jogs each morning
noncomputable def time_jogged_each_morning : ℝ := total_time_spent_jogging / number_of_days_jogged

-- State the theorem that the time jogged each morning is 1.5 hours
theorem jog_time_each_morning_is_1_5_hours : time_jogged_each_morning = 1.5 := by
  sorry

end jog_time_each_morning_is_1_5_hours_l2348_234809


namespace triangle_tangent_half_angle_l2348_234848

theorem triangle_tangent_half_angle (a b c : ℝ) (A : ℝ) (C : ℝ)
  (h : a + c = 2 * b) :
  Real.tan (A / 2) * Real.tan (C / 2) = 1 / 3 := 
sorry

end triangle_tangent_half_angle_l2348_234848


namespace value_of_angle_C_perimeter_range_l2348_234817

-- Part (1): Prove angle C value
theorem value_of_angle_C
  {a b c : ℝ} {A B C : ℝ}
  (acute_ABC : 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2)
  (m : ℝ × ℝ := (Real.sin C, Real.cos C))
  (n : ℝ × ℝ := (2 * Real.sin A - Real.cos B, -Real.sin B))
  (orthogonal_mn : m.1 * n.1 + m.2 * n.2 = 0) 
  : C = π / 6 := sorry

-- Part (2): Prove perimeter range
theorem perimeter_range
  {a b c : ℝ} {A B C : ℝ}
  (A_range : π / 3 < A ∧ A < π / 2)
  (C_value : C = π / 6)
  (a_value : a = 2)
  (acute_ABC : 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2)
  : 3 + 2 * Real.sqrt 3 < a + b + c ∧ a + b + c < 2 + 3 * Real.sqrt 3 := sorry

end value_of_angle_C_perimeter_range_l2348_234817


namespace prob_sunny_l2348_234867

variables (A B C : Prop) 
variables (P : Prop → ℝ)

-- Conditions
axiom prob_A : P A = 0.45
axiom prob_B : P B = 0.2
axiom mutually_exclusive : P A + P B + P C = 1

-- Proof problem
theorem prob_sunny : P C = 0.35 :=
by sorry

end prob_sunny_l2348_234867


namespace ratio_AD_DC_in_ABC_l2348_234897

theorem ratio_AD_DC_in_ABC 
  (A B C D : Type)
  [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
  (AB BC AC : Real) 
  (hAB : AB = 6) (hBC : BC = 8) (hAC : AC = 10) 
  (BD : Real) 
  (hBD : BD = 8) 
  (AD DC : Real)
  (hAD : AD = 2 * Real.sqrt 7)
  (hDC : DC = 10 - 2 * Real.sqrt 7) :
  AD / DC = (10 * Real.sqrt 7 + 14) / 36 :=
sorry

end ratio_AD_DC_in_ABC_l2348_234897


namespace tournament_players_l2348_234833

theorem tournament_players (n : ℕ) :
  (∃ k : ℕ, k = n + 12 ∧
    -- Exactly one-third of the points earned by each player were earned against the twelve players with the least number of points.
    (2 * (1 / 3 * (n * (n - 1) / 2)) + 2 / 3 * 66 + 66 = (k * (k - 1)) / 2) ∧
    --- Solving the quadratic equation derived
    (n = 4)) → 
    k = 16 :=
by
  sorry

end tournament_players_l2348_234833


namespace kan_krao_park_walkways_l2348_234847

-- Definitions for the given conditions
structure Park (α : Type*) := 
  (entrances : Finset α)
  (walkways : α → α → Prop)
  (brick_paved : α → α → Prop)
  (asphalt_paved : α → α → Prop)
  (no_three_intersections : ∀ (x y z w : α), x ≠ y → y ≠ z → z ≠ w → w ≠ x → (walkways x y ∧ walkways z w) → ¬ (walkways x z ∧ walkways y w))

-- Conditions based on the given problem
variables {α : Type*} [Finite α] [DecidableRel (@walkways α)]
variable (p : Park α)
variables [Fintype α]

-- Translate conditions to definitions
def has_lotuses (p : α → α → Prop) (q : α → α → Prop) (x y : α) : Prop := p x y ∧ p x y
def has_waterlilies (p : α → α → Prop) (q : α → α → Prop) (x y : α) : Prop := (p x y ∧ q x y) ∨ (q x y ∧ p x y)
def is_lit (p : α → α → Prop) (q : α → α → Prop) : Prop := ∃ (x y : α), x ≠ y ∧ (has_lotuses p q x y ∧ has_lotuses p q x y ∧ ∃ sz, sz ≥ 45)

-- Mathematically equivalent proof problem
theorem kan_krao_park_walkways (p : Park α) :
  (∃ walkways_same_material : α → α → Prop, ∃ (lit_walkways : Finset (α × α)), lit_walkways.card ≥ 11) :=
sorry

end kan_krao_park_walkways_l2348_234847


namespace negation_correct_l2348_234845

def original_statement (a : ℝ) : Prop :=
  a > 0 → a^2 > 0

def negated_statement (a : ℝ) : Prop :=
  a ≤ 0 → a^2 ≤ 0

theorem negation_correct (a : ℝ) : ¬ (original_statement a) ↔ negated_statement a :=
by
  sorry

end negation_correct_l2348_234845


namespace remainder_of_98_times_102_divided_by_9_l2348_234898

theorem remainder_of_98_times_102_divided_by_9 : (98 * 102) % 9 = 6 :=
by
  sorry

end remainder_of_98_times_102_divided_by_9_l2348_234898


namespace find_x_value_l2348_234857

theorem find_x_value :
  let a := (2021 : ℝ)
  let b := (2022 : ℝ)
  ∀ x : ℝ, (a / b - b / a + x = 0) → (x = b / a - a / b) :=
  by
    intros a b x h
    sorry

end find_x_value_l2348_234857


namespace least_integer_value_y_l2348_234853

theorem least_integer_value_y (y : ℤ) (h : abs (3 * y - 4) ≤ 25) : y = -7 :=
sorry

end least_integer_value_y_l2348_234853


namespace two_baskets_of_peaches_l2348_234816

theorem two_baskets_of_peaches (R G : ℕ) (h1 : G = R + 2) (h2 : 2 * R + 2 * G = 12) : R = 2 :=
by
  sorry

end two_baskets_of_peaches_l2348_234816


namespace max_fraction_sum_l2348_234824

theorem max_fraction_sum (a b c : ℝ) 
  (h_nonneg: a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0)
  (h_sum: a + b + c = 2) :
  (ab / (a + b)) + (ac / (a + c)) + (bc / (b + c)) ≤ 1 :=
sorry

end max_fraction_sum_l2348_234824


namespace sum_of_coefficients_eq_39_l2348_234804

theorem sum_of_coefficients_eq_39 :
  5 * (2 * 1^8 - 3 * 1^3 + 4) - 6 * (1^6 + 4 * 1^3 - 9) = 39 :=
by
  sorry

end sum_of_coefficients_eq_39_l2348_234804


namespace largest_pos_integer_binary_op_l2348_234878

def binary_op (n : ℤ) : ℤ := n - n * 5

theorem largest_pos_integer_binary_op :
  ∃ n : ℕ, binary_op n < 14 ∧ ∀ m : ℕ, binary_op m < 14 → m ≤ 1 :=
sorry

end largest_pos_integer_binary_op_l2348_234878


namespace find_value_of_x_squared_plus_inverse_squared_l2348_234871

theorem find_value_of_x_squared_plus_inverse_squared (x : ℝ) (hx : x + (1/x) = 2) : x^2 + (1/x^2) = 2 :=
sorry

end find_value_of_x_squared_plus_inverse_squared_l2348_234871


namespace remainder_identity_l2348_234828

variable {n : ℕ}

theorem remainder_identity
  (a b a_1 b_1 a_2 b_2 : ℕ)
  (ha : a = a_1 + a_2 * n)
  (hb : b = b_1 + b_2 * n) :
  (((a + b) % n = (a_1 + b_1) % n) ∧ ((a - b) % n = (a_1 - b_1) % n)) ∧ ((a * b) % n = (a_1 * b_1) % n) := by
  sorry

end remainder_identity_l2348_234828


namespace function_value_range_l2348_234886

noncomputable def f (x : ℝ) : ℝ := 9^x - 3^(x+1) + 2

theorem function_value_range :
  ∀ x, -1 ≤ x ∧ x ≤ 1 → -1/4 ≤ f x ∧ f x ≤ 2 :=
by
  sorry

end function_value_range_l2348_234886


namespace initial_peanuts_l2348_234822

theorem initial_peanuts (x : ℕ) (h : x + 4 = 8) : x = 4 :=
sorry

end initial_peanuts_l2348_234822


namespace sequences_of_length_15_l2348_234808

def odd_runs_of_A_even_runs_of_B (n : ℕ) : ℕ :=
  (if n = 1 then 1 else 0) + (if n = 2 then 1 else 0)

theorem sequences_of_length_15 : 
  odd_runs_of_A_even_runs_of_B 15 = 47260 :=
  sorry

end sequences_of_length_15_l2348_234808


namespace last_digit_of_sum_is_four_l2348_234821

theorem last_digit_of_sum_is_four (x y z : ℕ)
  (hx : 1 ≤ x ∧ x ≤ 9)
  (hy : 0 ≤ y ∧ y ≤ 9)
  (hz : 0 ≤ z ∧ z ≤ 9)
  (h : 1950 ≤ 200 * x + 11 * y + 11 * z ∧ 200 * x + 11 * y + 11 * z < 2000) :
  (200 * x + 11 * y + 11 * z) % 10 = 4 :=
sorry

end last_digit_of_sum_is_four_l2348_234821


namespace number_of_roses_l2348_234820

def total_flowers : ℕ := 10
def carnations : ℕ := 5
def roses : ℕ := total_flowers - carnations

theorem number_of_roses : roses = 5 := by
  sorry

end number_of_roses_l2348_234820

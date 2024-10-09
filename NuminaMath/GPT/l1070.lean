import Mathlib

namespace cubic_sum_expression_l1070_107092

theorem cubic_sum_expression (x y z p q r : ℝ) (h1 : x * y = p) (h2 : x * z = q) (h3 : y * z = r) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  x^3 + y^3 + z^3 = (p^2 * q^2 + p^2 * r^2 + q^2 * r^2) / (p * q * r) :=
by
  sorry

end cubic_sum_expression_l1070_107092


namespace largest_divisor_of_expression_l1070_107074

theorem largest_divisor_of_expression :
  ∃ x : ℕ, (∀ y : ℕ, x ∣ (7^y + 12*y - 1)) ∧ (∀ z : ℕ, (∀ y : ℕ, z ∣ (7^y + 12*y - 1)) → z ≤ x) :=
sorry

end largest_divisor_of_expression_l1070_107074


namespace quadratic_inequality_solution_set_l1070_107037

variables {a b c : ℝ}

theorem quadratic_inequality_solution_set (h : ∀ x, x > -1 ∧ x < 2 → ax^2 - bx + c > 0) :
  a + b + c = 0 :=
sorry

end quadratic_inequality_solution_set_l1070_107037


namespace negation_of_universal_sin_l1070_107066

theorem negation_of_universal_sin (h : ∀ x : ℝ, Real.sin x > 0) : ∃ x : ℝ, Real.sin x ≤ 0 :=
sorry

end negation_of_universal_sin_l1070_107066


namespace average_of_integers_is_ten_l1070_107047

theorem average_of_integers_is_ten (k m r s t : ℕ) 
  (h1 : k < m) (h2 : m < r) (h3 : r < s) (h4 : s < t)
  (h5 : k > 0) (h6 : m > 0)
  (h7 : t = 20) (h8 : r = 13)
  (h9 : k = 1) (h10 : m = 2) (h11 : s = 14) :
  (k + m + r + s + t) / 5 = 10 := by
  sorry

end average_of_integers_is_ten_l1070_107047


namespace value_of_a_plus_b_2023_l1070_107036

theorem value_of_a_plus_b_2023 
    (x y a b : ℤ)
    (h1 : 4*x + 3*y = 11)
    (h2 : 2*x - y = 3)
    (h3 : a*x + b*y = -2)
    (h4 : b*x - a*y = 6)
    (hx : x = 2)
    (hy : y = 1) :
    (a + b) ^ 2023 = 0 := 
sorry

end value_of_a_plus_b_2023_l1070_107036


namespace initial_concentration_of_hydrochloric_acid_l1070_107030

theorem initial_concentration_of_hydrochloric_acid
  (initial_mass : ℕ)
  (drained_mass : ℕ)
  (added_concentration : ℕ)
  (final_concentration : ℕ)
  (total_mass : ℕ)
  (initial_concentration : ℕ) :
  initial_mass = 300 ∧ drained_mass = 25 ∧ added_concentration = 80 ∧ final_concentration = 25 ∧ total_mass = 300 →
  (275 * initial_concentration / 100 + 20 = 75) →
  initial_concentration = 20 :=
by
  intros h_eq h_new_solution
  -- Rewriting the data given in h_eq and solving h_new_solution
  rcases h_eq with ⟨h_initial_mass, h_drained_mass, h_added_concentration, h_final_concentration, h_total_mass⟩
  sorry

end initial_concentration_of_hydrochloric_acid_l1070_107030


namespace digits_with_five_or_seven_is_5416_l1070_107019

/-- The total number of four-digit positive integers. -/
def total_four_digit_integers : ℕ := 9000

/-- The number of four-digit integers without the digits 5 or 7. -/
def no_five_seven_integers : ℕ := 3584

/-- The number of four-digit positive integers that have at least one digit that is a 5 or a 7. -/
def digits_with_five_or_seven : ℕ :=
  total_four_digit_integers - no_five_seven_integers

/-- Proof that the number of four-digit integers with at least one digit that is a 5 or 7 is 5416. -/
theorem digits_with_five_or_seven_is_5416 :
  digits_with_five_or_seven = 5416 :=
by
  sorry

end digits_with_five_or_seven_is_5416_l1070_107019


namespace ratio_of_x_to_y_l1070_107009

theorem ratio_of_x_to_y (x y : ℝ) (h : y = 0.20 * x) : x / y = 5 :=
by
  sorry

end ratio_of_x_to_y_l1070_107009


namespace sum_of_cube_faces_l1070_107098

theorem sum_of_cube_faces (a b c d e f : ℕ) (h_pos: 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f) 
    (h_eq_sum: (a * b * c) + (a * e * c) + (a * b * f) + (a * e * f) + (d * b * c) + (d * e * c) + (d * b * f) + (d * e * f) = 1089) :
    a + b + c + d + e + f = 31 := 
by
  sorry

end sum_of_cube_faces_l1070_107098


namespace approximate_value_correct_l1070_107045

noncomputable def P1 : ℝ := (47 / 100) * 1442
noncomputable def P2 : ℝ := (36 / 100) * 1412
noncomputable def result : ℝ := (P1 - P2) + 63

theorem approximate_value_correct : abs (result - 232.42) < 0.01 := 
by
  -- Proof to be completed
  sorry

end approximate_value_correct_l1070_107045


namespace Eric_test_score_l1070_107026

theorem Eric_test_score (n : ℕ) (old_avg new_avg : ℚ) (Eric_score : ℚ) :
  n = 22 →
  old_avg = 84 →
  new_avg = 85 →
  Eric_score = (n * new_avg) - ((n - 1) * old_avg) →
  Eric_score = 106 :=
by
  intros h1 h2 h3 h4
  sorry

end Eric_test_score_l1070_107026


namespace find_n_values_l1070_107031

theorem find_n_values (n : ℕ) (h : ∃ k : ℕ, n^2 - 19 * n + 91 = k^2) : n = 9 ∨ n = 10 :=
sorry

end find_n_values_l1070_107031


namespace chromium_percentage_new_alloy_l1070_107004

variable (w1 w2 : ℝ) (cr1 cr2 : ℝ)

theorem chromium_percentage_new_alloy (h_w1 : w1 = 15) (h_w2 : w2 = 30) (h_cr1 : cr1 = 0.12) (h_cr2 : cr2 = 0.08) :
  (cr1 * w1 + cr2 * w2) / (w1 + w2) * 100 = 9.33 := by
  sorry

end chromium_percentage_new_alloy_l1070_107004


namespace largest_inscribed_triangle_area_l1070_107033

-- Definition of the conditions
def radius : ℝ := 10
def diameter : ℝ := 2 * radius

-- The theorem to be proven
theorem largest_inscribed_triangle_area (r : ℝ) (D : ℝ) (h : D = 2 * r) : 
  ∃ (A : ℝ), A = 100 := by
  have base := D
  have height := r
  have area := (1 / 2) * base * height
  use area
  sorry

end largest_inscribed_triangle_area_l1070_107033


namespace team_C_games_played_l1070_107054

variable (x : ℕ)
variable (winC : ℕ := 5 * x / 7)
variable (loseC : ℕ := 2 * x / 7)
variable (winD : ℕ := 2 * x / 3)
variable (loseD : ℕ := x / 3)

theorem team_C_games_played :
  winD = winC - 5 →
  loseD = loseC - 5 →
  x = 105 := by
  sorry

end team_C_games_played_l1070_107054


namespace hotdogs_sold_correct_l1070_107006

def initial_hotdogs : ℕ := 99
def remaining_hotdogs : ℕ := 97
def sold_hotdogs : ℕ := initial_hotdogs - remaining_hotdogs

theorem hotdogs_sold_correct : sold_hotdogs = 2 := by
  sorry

end hotdogs_sold_correct_l1070_107006


namespace upgraded_fraction_l1070_107042

theorem upgraded_fraction (N U : ℕ) (h1 : ∀ (k : ℕ), k = 24)
  (h2 : ∀ (n : ℕ), N = n) (h3 : ∀ (u : ℕ), U = u)
  (h4 : N = U / 8) : U / (24 * N + U) = 1 / 4 := by
  sorry

end upgraded_fraction_l1070_107042


namespace lucy_times_three_ago_l1070_107010

  -- Defining the necessary variables and conditions
  def lucy_age_now : ℕ := 50
  def lovely_age (x : ℕ) : ℕ := 20  -- The age of Lovely when x years has passed
  
  -- Statement of the problem
  theorem lucy_times_three_ago {x : ℕ} : 
    (lucy_age_now - x = 3 * (lovely_age x - x)) → (lucy_age_now + 10 = 2 * (lovely_age x + 10)) → x = 5 := 
  by
  -- Proof is omitted
  sorry
  
end lucy_times_three_ago_l1070_107010


namespace ab_value_l1070_107043

theorem ab_value (a b : ℝ) (h1 : |a| = 3) (h2 : |b - 2| = 9) (h3 : a + b > 0) :
  ab = 33 ∨ ab = -33 :=
by
  sorry

end ab_value_l1070_107043


namespace solution_set_l1070_107062

noncomputable def f : ℝ → ℝ := sorry
def dom := {x : ℝ | x < 0 ∨ x > 0 } -- Definition of the function domain

-- Assumptions and conditions as definitions in Lean
axiom f_odd : ∀ x ∈ dom, f (-x) = -f x
axiom f_at_1 : f 1 = 1
axiom symmetric_f : ∀ x ∈ dom, (f (x + 1)) = -f (-x + 1)
axiom inequality_condition : ∀ (x1 x2 : ℝ), x1 ∈ dom → x2 ∈ dom → x1 ≠ x2 → (x1^3 * f x1 - x2^3 * f x2) / (x1 - x2) > 0

-- The main statement to be proved
theorem solution_set :
  {x ∈ dom | f x ≤ 1 / x^3} = {x ∈ dom | x ≤ -1} ∪ {x ∈ dom | 0 < x ∧ x ≤ 1} :=
sorry

end solution_set_l1070_107062


namespace fractions_problem_l1070_107061

theorem fractions_problem (x y : ℚ) (hx : x = 2 / 3) (hy : y = 3 / 2) :
  (1 / 3) * x^5 * y^6 = 3 / 2 := by
  sorry

end fractions_problem_l1070_107061


namespace combined_weight_difference_l1070_107005

def chemistry_weight : ℝ := 7.125
def geometry_weight : ℝ := 0.625
def calculus_weight : ℝ := -5.25
def biology_weight : ℝ := 3.755

theorem combined_weight_difference :
  (chemistry_weight - calculus_weight) - (geometry_weight + biology_weight) = 7.995 :=
by
  sorry

end combined_weight_difference_l1070_107005


namespace mabel_age_l1070_107059

theorem mabel_age (n : ℕ) (h : n * (n + 1) / 2 = 28) : n = 7 :=
sorry

end mabel_age_l1070_107059


namespace option1_payment_correct_option2_payment_correct_most_cost_effective_for_30_l1070_107099

variables (x : ℕ) (hx : x > 10)

def suit_price : ℕ := 1000
def tie_price : ℕ := 200
def num_suits : ℕ := 10

-- Option 1: Buy one suit, get one tie for free
def option1_payment : ℕ := 200 * x + 8000

-- Option 2: All items sold at a 10% discount
def option2_payment : ℕ := (10 * 1000 + x * 200) * 9 / 10

-- For x = 30, which option is more cost-effective
def x_value := 30
def option1_payment_30 : ℕ := 200 * x_value + 8000
def option2_payment_30 : ℕ := (10 * 1000 + x_value * 200) * 9 / 10
def more_cost_effective_option_30 : ℕ := if option1_payment_30 < option2_payment_30 then option1_payment_30 else option2_payment_30

-- Most cost-effective option for x = 30 with new combination plan
def combination_payment_30 : ℕ := 10000 + 20 * 200 * 9 / 10

-- Statements to be proved
theorem option1_payment_correct : option1_payment x = 200 * x + 8000 := sorry

theorem option2_payment_correct : option2_payment x = (10 * 1000 + x * 200) * 9 / 10 := sorry

theorem most_cost_effective_for_30 :
  option1_payment_30 = 14000 ∧ 
  option2_payment_30 = 14400 ∧ 
  more_cost_effective_option_30 = 14000 ∧
  combination_payment_30 = 13600 := sorry

end option1_payment_correct_option2_payment_correct_most_cost_effective_for_30_l1070_107099


namespace train_speed_l1070_107027

noncomputable def trainLength : ℕ := 400
noncomputable def timeToCrossPole : ℕ := 20

theorem train_speed : (trainLength / timeToCrossPole) = 20 := by
  sorry

end train_speed_l1070_107027


namespace inequality_solution_l1070_107020

theorem inequality_solution {x : ℝ} (h : -2 < (x^2 - 18*x + 24) / (x^2 - 4*x + 8) ∧ (x^2 - 18*x + 24) / (x^2 - 4*x + 8) < 2) : 
  x ∈ Set.Ioo (-2 : ℝ) (10 / 3) :=
by
  sorry

end inequality_solution_l1070_107020


namespace problem1_problem2_l1070_107016

-- Definitions and conditions
def A (m : ℝ) : Set ℝ := { x | m ≤ x ∧ x ≤ m + 1 }
def B : Set ℝ := { x | x < -6 ∨ x > 1 }

-- (Ⅰ) Problem statement: Prove that if A ∩ B = ∅, then -6 ≤ m ≤ 0.
theorem problem1 (m : ℝ) : A m ∩ B = ∅ ↔ -6 ≤ m ∧ m ≤ 0 := 
by
  sorry

-- (Ⅱ) Problem statement: Prove that if A ⊆ B, then m < -7 or m > 1.
theorem problem2 (m : ℝ) : A m ⊆ B ↔ m < -7 ∨ m > 1 := 
by
  sorry

end problem1_problem2_l1070_107016


namespace rower_rate_in_still_water_l1070_107076

theorem rower_rate_in_still_water (V_m V_s : ℝ) (h1 : V_m + V_s = 16) (h2 : V_m - V_s = 12) : V_m = 14 := 
sorry

end rower_rate_in_still_water_l1070_107076


namespace olafs_dad_points_l1070_107041

-- Let D be the number of points Olaf's dad scored.
def dad_points : ℕ := sorry

-- Olaf scored three times more points than his dad.
def olaf_points (dad_points : ℕ) : ℕ := 3 * dad_points

-- Total points scored is 28.
def total_points (dad_points olaf_points : ℕ) : Prop := dad_points + olaf_points = 28

theorem olafs_dad_points (D : ℕ) :
  (D + olaf_points D = 28) → (D = 7) :=
by
  sorry

end olafs_dad_points_l1070_107041


namespace no_solutions_cryptarithm_l1070_107083

theorem no_solutions_cryptarithm : 
  ∀ (K O P H A B U y C : ℕ), 
  K ≠ O ∧ K ≠ P ∧ K ≠ H ∧ K ≠ A ∧ K ≠ B ∧ K ≠ U ∧ K ≠ y ∧ K ≠ C ∧ 
  O ≠ P ∧ O ≠ H ∧ O ≠ A ∧ O ≠ B ∧ O ≠ U ∧ O ≠ y ∧ O ≠ C ∧ 
  P ≠ H ∧ P ≠ A ∧ P ≠ B ∧ P ≠ U ∧ P ≠ y ∧ P ≠ C ∧ 
  H ≠ A ∧ H ≠ B ∧ H ≠ U ∧ H ≠ y ∧ H ≠ C ∧ 
  A ≠ B ∧ A ≠ U ∧ A ≠ y ∧ A ≠ C ∧ 
  B ≠ U ∧ B ≠ y ∧ B ≠ C ∧ 
  U ≠ y ∧ U ≠ C ∧ 
  y ≠ C ∧
  K < O ∧ O < P ∧ P > O ∧ O > H ∧ H > A ∧ A > B ∧ B > U ∧ U > P ∧ P > y ∧ y > C → 
  false :=
sorry

end no_solutions_cryptarithm_l1070_107083


namespace largest_integer_less_than_100_with_remainder_5_l1070_107063

theorem largest_integer_less_than_100_with_remainder_5 (n : ℤ) (h₁ : n < 100) (h₂ : n % 8 = 5) : n ≤ 99 :=
sorry

end largest_integer_less_than_100_with_remainder_5_l1070_107063


namespace range_of_m_l1070_107097

noncomputable def quadratic_polynomial (m : ℝ) (x : ℝ) : ℝ :=
  x^2 + (m - 1) * x + m^2 - 2

theorem range_of_m (m : ℝ) (h1 : ∃ x1 x2 : ℝ, x1 < -1 ∧ x2 > 1 ∧ quadratic_polynomial m x1 = 0 ∧ quadratic_polynomial m x2 = 0) :
  0 < m ∧ m < 1 :=
sorry

end range_of_m_l1070_107097


namespace intersection_point_of_curve_and_line_l1070_107034

theorem intersection_point_of_curve_and_line : 
  ∃ (e : ℝ), (0 < e) ∧ (e = Real.exp 1) ∧ ((e, e) ∈ { p : ℝ × ℝ | ∃ (x y : ℝ), x ^ y = y ^ x ∧ 0 ≤ x ∧ 0 ≤ y}) :=
by {
  sorry
}

end intersection_point_of_curve_and_line_l1070_107034


namespace select_subset_divisible_by_n_l1070_107000

theorem select_subset_divisible_by_n (n : ℕ) (h : n > 0) (l : List ℤ) (hl : l.length = 2 * n - 1) :
  ∃ s : Finset ℤ, s.card = n ∧ (s.sum id) % n = 0 := 
sorry

end select_subset_divisible_by_n_l1070_107000


namespace solve_system_of_equations_l1070_107040

-- Conditions from the problem
variables (x y : ℚ)

-- Definitions (the original equations)
def equation1 := x + 2 * y = 3
def equation2 := 9 * x - 8 * y = 5

-- Correct answer
def solution_x := 17 / 13
def solution_y := 11 / 13

-- The final proof statement
theorem solve_system_of_equations (h1 : equation1 solution_x solution_y) (h2 : equation2 solution_x solution_y) :
  x = solution_x ∧ y = solution_y := sorry

end solve_system_of_equations_l1070_107040


namespace max_value_of_vector_dot_product_l1070_107002

theorem max_value_of_vector_dot_product :
  ∀ (x y : ℝ), (-2 ≤ x ∧ x ≤ 2 ∧ -2 ≤ y ∧ y ≤ 2) → (2 * x - y ≤ 4) :=
by
  intros x y h
  sorry

end max_value_of_vector_dot_product_l1070_107002


namespace largest_trifecta_sum_l1070_107079

def trifecta (a b c : ℕ) : Prop :=
  a < b ∧ b < c ∧ a ∣ b ∧ b ∣ c ∧ c ∣ (a * b) ∧ (100 ≤ a) ∧ (a < 1000) ∧ (100 ≤ b) ∧ (b < 1000) ∧ (100 ≤ c) ∧ (c < 1000)

theorem largest_trifecta_sum : ∃ (a b c : ℕ), trifecta a b c ∧ a + b + c = 700 :=
sorry

end largest_trifecta_sum_l1070_107079


namespace cows_eat_grass_l1070_107029

theorem cows_eat_grass (ha_per_cow_per_week : ℝ) (ha_grow_per_week : ℝ) :
  (∀ (weeks_cows_weeks_ha : ℕ × ℕ × ℕ × ℕ), weeks_cows_weeks_ha = (2, 3, 2, 2) →
    (2 : ℝ) = 3 * 2 * ha_per_cow_per_week - 2 * ha_grow_per_week) → 
  (∀ (weeks_cows_weeks_ha : ℕ × ℕ × ℕ × ℕ), weeks_cows_weeks_ha = (4, 2, 4, 2) →
    (2 : ℝ) = 2 * 4 * ha_per_cow_per_week - 4 * ha_grow_per_week) → 
  ∃ (cows : ℕ), (6 : ℝ) = cows * 6 * ha_per_cow_per_week - 6 * ha_grow_per_week ∧ cows = 3 :=
sorry

end cows_eat_grass_l1070_107029


namespace u_less_than_v_l1070_107023

noncomputable def f (u : ℝ) := (u + u^2 + u^3 + u^4 + u^5 + u^6 + u^7 + u^8) + 10 * u^9
noncomputable def g (v : ℝ) := (v + v^2 + v^3 + v^4 + v^5 + v^6 + v^7 + v^8 + v^9 + v^10) + 10 * v^11

theorem u_less_than_v
  (u v : ℝ)
  (hu : f u = 8)
  (hv : g v = 8) :
  u < v := 
sorry

end u_less_than_v_l1070_107023


namespace problem1_problem2_l1070_107028

def vector_dot (v1 v2 : ℝ × ℝ) : ℝ := 
  v1.1 * v2.1 + v1.2 * v2.2

def perpendicular (v1 v2 : ℝ × ℝ) : Prop := 
  vector_dot v1 v2 = 0

def parallel (v1 v2 : ℝ × ℝ) : Prop := 
  v1.1 * v2.2 = v1.2 * v2.1

-- Given vectors in the problem
def a : ℝ × ℝ := (-3, 1)
def b : ℝ × ℝ := (1, -2)
def c : ℝ × ℝ := (1, -1)
def n (k : ℝ) : ℝ × ℝ := (a.1 + k * b.1, a.2 + k * b.2)
def v : ℝ × ℝ := (2 * a.1 - b.1, 2 * a.2 - b.2)

-- Problem 1: Find k when n is perpendicular to v
theorem problem1 (k : ℝ) : perpendicular (n k) v → k = 5 / 3 := 
by sorry

-- Problem 2: Find k when n is parallel to c + k * b
theorem problem2 (k : ℝ) : parallel (n k) (c.1 + k * b.1, c.2 + k * b.2) → k = -1 / 3 := 
by sorry

end problem1_problem2_l1070_107028


namespace cost_of_3600_pens_l1070_107046

-- Define the conditions
def cost_per_200_pens : ℕ := 50
def pens_bought : ℕ := 3600

-- Define a theorem to encapsulate our question and provide the necessary definitions
theorem cost_of_3600_pens : cost_per_200_pens / 200 * pens_bought = 900 := by sorry

end cost_of_3600_pens_l1070_107046


namespace sum_first10PrimesGT50_eq_732_l1070_107022

def first10PrimesGT50 := [53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

theorem sum_first10PrimesGT50_eq_732 :
  first10PrimesGT50.sum = 732 := by
  sorry

end sum_first10PrimesGT50_eq_732_l1070_107022


namespace conversion_base_10_to_5_l1070_107078

theorem conversion_base_10_to_5 : 
  (425 : ℕ) = 3 * 5^3 + 2 * 5^2 + 0 * 5^1 + 0 * 5^0 :=
by sorry

end conversion_base_10_to_5_l1070_107078


namespace only_other_list_with_same_product_l1070_107094

-- Assigning values to letters
def letter_value (ch : Char) : ℕ :=
  match ch with
  | 'A' => 1 | 'B' => 2 | 'C' => 3 | 'D' => 4 | 'E' => 5 | 'F' => 6 | 'G' => 7 | 'H' => 8
  | 'I' => 9 | 'J' => 10| 'K' => 11| 'L' => 12| 'M' => 13| 'N' => 14| 'O' => 15| 'P' => 16
  | 'Q' => 17| 'R' => 18| 'S' => 19| 'T' => 20| 'U' => 21| 'V' => 22| 'W' => 23| 'X' => 24
  | 'Y' => 25| 'Z' => 26| _ => 0

-- Define the product function for a list of 4 letters
def product_of_list (lst : List Char) : ℕ :=
  lst.map letter_value |> List.prod

-- Define the specific lists
def BDFH : List Char := ['B', 'D', 'F', 'H']
def BCDH : List Char := ['B', 'C', 'D', 'H']

-- The main statement to prove
theorem only_other_list_with_same_product : 
  product_of_list BCDH = product_of_list BDFH :=
by
  -- Sorry is a placeholder for the proof
  sorry

end only_other_list_with_same_product_l1070_107094


namespace percentage_increase_equal_price_l1070_107014

/-
A merchant has selected two items to be placed on sale, one of which currently sells for 20 percent less than the other.
He wishes to raise the price of the cheaper item so that the two items are equally priced.
By what percentage must he raise the price of the less expensive item?
-/
theorem percentage_increase_equal_price (P: ℝ) : (P > 0) → 
  (∀ cheap_item, cheap_item = 0.80 * P → ((P - cheap_item) / cheap_item) * 100 = 25) :=
by
  intro P_pos
  intro cheap_item
  intro h
  sorry

end percentage_increase_equal_price_l1070_107014


namespace find_certain_number_l1070_107090

theorem find_certain_number (x : ℝ) 
    (h : 7 * x - 6 - 12 = 4 * x) : x = 6 := 
by
  sorry

end find_certain_number_l1070_107090


namespace tensor_12_9_l1070_107001

def tensor (a b : ℚ) : ℚ := a + (4 * a) / (3 * b)

theorem tensor_12_9 : tensor 12 9 = 13 + 7 / 9 :=
by
  sorry

end tensor_12_9_l1070_107001


namespace crayons_birthday_l1070_107048

theorem crayons_birthday (C E : ℕ) (hC : C = 523) (hE : E = 457) (hDiff : C = E + 66) : C = 523 := 
by {
  -- proof would go here
  sorry
}

end crayons_birthday_l1070_107048


namespace f_inequality_l1070_107087

noncomputable def f (a x : ℝ) : ℝ := a * (Real.exp x + a) - x

theorem f_inequality (a : ℝ) (h : a > 0) (x : ℝ) : 
  f a x > 2 * Real.log a + 3 / 2 := 
sorry 

end f_inequality_l1070_107087


namespace find_theta_l1070_107093

theorem find_theta (θ : Real) : 
  (0 ≤ θ ∧ θ ≤ 2 * Real.pi) →
  (∀ x : Real, 0 ≤ x ∧ x ≤ 1 → 
    x ^ 3 * Real.sin θ + x ^ 2 * Real.cos θ - x * (1 - x) + (1 - x) ^ 2 * Real.sin θ > 0) → 
  Real.sin θ > 0 → 
  Real.cos θ + Real.sin θ > 0 → 
  (Real.pi / 12 < θ ∧ θ < 5 * Real.pi / 12) :=
by
  intro θ_range all_x_condition sin_pos cos_sin_pos
  sorry

end find_theta_l1070_107093


namespace determine_constants_l1070_107044

theorem determine_constants (k a b : ℝ) :
  (3*x^2 - 4*x + 5)*(5*x^2 + k*x + 8) = 15*x^4 - 47*x^3 + a*x^2 - b*x + 40 →
  k = -9 ∧ a = 15 ∧ b = 72 :=
by
  sorry

end determine_constants_l1070_107044


namespace probability_at_least_one_head_and_die_3_l1070_107086

-- Define the probability of an event happening
noncomputable def probability_of_event (total_outcomes : ℕ) (successful_outcomes : ℕ) : ℚ :=
  successful_outcomes / total_outcomes

-- Define the problem specific values
def total_coin_outcomes : ℕ := 4
def successful_coin_outcomes : ℕ := 3
def total_die_outcomes : ℕ := 8
def successful_die_outcome : ℕ := 1
def total_outcomes : ℕ := total_coin_outcomes * total_die_outcomes
def successful_outcomes : ℕ := successful_coin_outcomes * successful_die_outcome

-- Prove that the probability of at least one head in two coin flips and die showing a 3 is 3/32
theorem probability_at_least_one_head_and_die_3 : 
  probability_of_event total_outcomes successful_outcomes = 3 / 32 := by
  sorry

end probability_at_least_one_head_and_die_3_l1070_107086


namespace range_of_a_l1070_107003

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x| ≥ a * x) → -1 ≤ a ∧ a ≤ 1 :=
by
  sorry

end range_of_a_l1070_107003


namespace final_price_relative_l1070_107057

-- Definitions of the conditions
variable (x : ℝ)
#check x * 1.30  -- original price increased by 30%
#check x * 1.30 * 0.85  -- after 15% discount on increased price
#check x * 1.30 * 0.85 * 1.05  -- after applying 5% tax on discounted price

-- Theorem to prove the final price relative to the original price
theorem final_price_relative (x : ℝ) : 
  (x * 1.30 * 0.85 * 1.05) = (1.16025 * x) :=
by
  sorry

end final_price_relative_l1070_107057


namespace range_of_f_l1070_107055

theorem range_of_f (x : ℝ) (h1 : 1 ≤ x) (h2 : x ≤ 2) : -3 ≤ (3^x - 6/x) ∧ (3^x - 6/x) ≤ 6 :=
by
  sorry

end range_of_f_l1070_107055


namespace problem_l1070_107025

def f (x : ℤ) : ℤ := 7 * x - 3

theorem problem : f (f (f 3)) = 858 := by
  sorry

end problem_l1070_107025


namespace range_of_expression_l1070_107067

theorem range_of_expression (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 ≤ β ∧ β ≤ π / 2) :
  -π / 6 < 2 * α - β / 3 ∧ 2 * α - β / 3 < π := by
  sorry

end range_of_expression_l1070_107067


namespace compare_fractions_l1070_107069

def a : ℚ := -3 / 4
def b : ℚ := -4 / 5

theorem compare_fractions : a > b :=
by {
  sorry
}

end compare_fractions_l1070_107069


namespace average_earnings_per_minute_l1070_107082

theorem average_earnings_per_minute 
  (laps : ℕ) (meters_per_lap : ℕ) (dollars_per_100_meters : ℝ) (total_minutes : ℕ) (total_laps : ℕ)
  (h_laps : total_laps = 24)
  (h_meters_per_lap : meters_per_lap = 100)
  (h_dollars_per_100_meters : dollars_per_100_meters = 3.5)
  (h_total_minutes : total_minutes = 12)
  : (total_laps * meters_per_lap / 100 * dollars_per_100_meters / total_minutes) = 7 := 
by
  sorry

end average_earnings_per_minute_l1070_107082


namespace sum_of_digits_of_N_is_19_l1070_107056

-- Given facts about N
variables (N : ℕ) (h1 : 100 ≤ N ∧ N < 1000) 
           (h2 : N % 10 = 7) 
           (h3 : N % 11 = 7) 
           (h4 : N % 12 = 7)

-- Main theorem statement
theorem sum_of_digits_of_N_is_19 : 
  ((N / 100) + ((N % 100) / 10) + (N % 10) = 19) := sorry

end sum_of_digits_of_N_is_19_l1070_107056


namespace other_root_of_quadratic_l1070_107085

theorem other_root_of_quadratic (a c : ℝ) (h : a ≠ 0) (h_root : 4 * a * 0^2 - 2 * a * 0 + c = 0) :
  ∃ t : ℝ, (4 * a * t^2 - 2 * a * t + c = 0) ∧ t = 1 / 2 :=
by
  sorry

end other_root_of_quadratic_l1070_107085


namespace heart_beats_during_marathon_l1070_107052

theorem heart_beats_during_marathon :
  (∃ h_per_min t1 t2 total_time,
    h_per_min = 140 ∧
    t1 = 15 * 6 ∧
    t2 = 15 * 5 ∧
    total_time = t1 + t2 ∧
    23100 = h_per_min * total_time) :=
  sorry

end heart_beats_during_marathon_l1070_107052


namespace largest_divisor_of_m_l1070_107053

theorem largest_divisor_of_m (m : ℤ) (hm_pos : 0 < m) (h : 33 ∣ m^2) : 33 ∣ m :=
sorry

end largest_divisor_of_m_l1070_107053


namespace AngeliCandies_l1070_107073

def CandyProblem : Prop :=
  ∃ (C B G : ℕ), 
    (1/3 : ℝ) * C = 3 * (B : ℝ) ∧
    (2/3 : ℝ) * C = 2 * (G : ℝ) ∧
    (B + G = 40) ∧ 
    C = 144

theorem AngeliCandies :
  CandyProblem :=
sorry

end AngeliCandies_l1070_107073


namespace Total_marbles_equal_231_l1070_107071

def Connie_marbles : Nat := 39
def Juan_marbles : Nat := Connie_marbles + 25
def Maria_marbles : Nat := 2 * Juan_marbles
def Total_marbles : Nat := Connie_marbles + Juan_marbles + Maria_marbles

theorem Total_marbles_equal_231 : Total_marbles = 231 := sorry

end Total_marbles_equal_231_l1070_107071


namespace unique_solution_l1070_107038

noncomputable def uniquely_solvable (a : ℝ) : Prop :=
  ∀ x : ℝ, a > 0 ∧ a ≠ 1 → ∃! x, a^x = (Real.log x / Real.log (1/4))

theorem unique_solution (a : ℝ) : a > 0 ∧ a ≠ 1 → uniquely_solvable a :=
by sorry

end unique_solution_l1070_107038


namespace points_per_right_answer_l1070_107018

variable (p : ℕ)
variable (total_problems : ℕ := 25)
variable (wrong_problems : ℕ := 3)
variable (score : ℤ := 85)

theorem points_per_right_answer :
  (total_problems - wrong_problems) * p - wrong_problems = score -> p = 4 :=
  sorry

end points_per_right_answer_l1070_107018


namespace tangent_circle_exists_l1070_107017
open Set

-- Definitions of given point, line, and circle
variables {Point : Type*} {Line : Type*} {Circle : Type*} 
variables (M : Point) (l : Line) (S : Circle)
variables (center_S : Point) (radius_S : ℝ)

-- Conditions of the problem
variables (touches_line : Circle → Line → Prop) (touches_circle : Circle → Circle → Prop)
variables (passes_through : Circle → Point → Prop) (center_of : Circle → Point)
variables (radius_of : Circle → ℝ)

-- Existence theorem to prove
theorem tangent_circle_exists 
  (given_tangent_to_line : Circle → Line → Bool)
  (given_tangent_to_circle : Circle → Circle → Bool)
  (given_passes_through : Circle → Point → Bool):
  ∃ (Ω : Circle), 
    given_tangent_to_line Ω l ∧
    given_tangent_to_circle Ω S ∧
    given_passes_through Ω M :=
sorry

end tangent_circle_exists_l1070_107017


namespace tan_identity_l1070_107096

theorem tan_identity
  (α : ℝ)
  (h : Real.tan (π / 3 - α) = 1 / 3) :
  Real.tan (2 * π / 3 + α) = -1 / 3 := 
sorry

end tan_identity_l1070_107096


namespace strokes_over_par_l1070_107080

theorem strokes_over_par (n s p : ℕ) (t : ℕ) (par : ℕ )
  (h1 : n = 9)
  (h2 : s = 4)
  (h3 : p = 3)
  (h4: t = n * s)
  (h5: par = n * p) :
  t - par = 9 :=
by 
  sorry

end strokes_over_par_l1070_107080


namespace find_f2_f5_sum_l1070_107011

theorem find_f2_f5_sum
  (f : ℤ → ℤ)
  (a b : ℤ)
  (h1 : f 1 = 4)
  (h2 : ∀ z : ℤ, f z = 3 * z + 6)
  (h3 : ∀ x y : ℤ, f (x + y) = f x + f y + a * x * y + b) :
  f 2 + f 5 = 33 :=
sorry

end find_f2_f5_sum_l1070_107011


namespace area_of_triangle_is_3_l1070_107058

noncomputable def area_of_triangle_ABC (A B C : ℝ × ℝ) : ℝ :=
1 / 2 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem area_of_triangle_is_3 : 
  ∀ (A B C : ℝ × ℝ), 
  A = (-5, -2) → 
  B = (0, 0) → 
  C = (7, -4) →
  area_of_triangle_ABC A B C = 3 :=
by
  intros A B C hA hB hC
  rw [hA, hB, hC]
  sorry

end area_of_triangle_is_3_l1070_107058


namespace number_of_blue_eyed_students_in_k_class_l1070_107015

-- Definitions based on the given conditions
def total_students := 40
def blond_hair_to_blue_eyes_ratio := 2.5
def students_with_both := 8
def students_with_neither := 5

-- We need to prove that the number of blue-eyed students is 10
theorem number_of_blue_eyed_students_in_k_class 
  (x : ℕ)  -- number of blue-eyed students
  (H1 : total_students = 40)
  (H2 : ∀ x, blond_hair_to_blue_eyes_ratio * x = number_of_blond_students)
  (H3 : students_with_both = 8)
  (H4 : students_with_neither = 5)
  : x = 10 :=
sorry

end number_of_blue_eyed_students_in_k_class_l1070_107015


namespace sequence_of_arrows_l1070_107050

theorem sequence_of_arrows (n : ℕ) (h : n % 5 = 0) : 
  (n < 570 ∧ n % 5 = 0) → 
  (n + 1 < 573 ∧ (n + 1) % 5 = 1) → 
  (n + 2 < 573 ∧ (n + 2) % 5 = 2) → 
  (n + 3 < 573 ∧ (n + 3) % 5 = 3) →
    true :=
by
  sorry

end sequence_of_arrows_l1070_107050


namespace segment_measure_l1070_107060

theorem segment_measure (a b : ℝ) (m : ℝ) (h : a = m * b) : (1 / m) * a = b :=
by sorry

end segment_measure_l1070_107060


namespace right_triangle_inequality_l1070_107032

theorem right_triangle_inequality {a b c : ℝ} (h₁ : a^2 + b^2 = c^2) : a + b ≤ c * Real.sqrt 2 := by
  sorry

end right_triangle_inequality_l1070_107032


namespace years_of_school_eq_13_l1070_107024

/-- Conditions definitions -/
def cost_per_semester : ℕ := 20000
def semesters_per_year : ℕ := 2
def total_cost : ℕ := 520000

/-- Derived definitions from conditions -/
def cost_per_year := cost_per_semester * semesters_per_year
def number_of_years := total_cost / cost_per_year

/-- Proof that number of years equals 13 given the conditions -/
theorem years_of_school_eq_13 : number_of_years = 13 :=
by sorry

end years_of_school_eq_13_l1070_107024


namespace prob_two_white_balls_l1070_107072

open Nat

def total_balls : ℕ := 8 + 10

def prob_first_white : ℚ := 8 / total_balls

def prob_second_white (total_balls_minus_one : ℕ) : ℚ := 7 / total_balls_minus_one

theorem prob_two_white_balls : 
  ∃ (total_balls_minus_one : ℕ) (p_first p_second : ℚ), 
    total_balls_minus_one = total_balls - 1 ∧
    p_first = prob_first_white ∧
    p_second = prob_second_white total_balls_minus_one ∧
    p_first * p_second = 28 / 153 := 
by
  sorry

end prob_two_white_balls_l1070_107072


namespace area_of_common_region_l1070_107039

theorem area_of_common_region (β : ℝ) (h1 : 0 < β ∧ β < π / 2) (h2 : Real.cos β = 3 / 5) :
  ∃ (area : ℝ), area = 4 / 9 := 
by 
  sorry

end area_of_common_region_l1070_107039


namespace volume_ratio_of_trapezoidal_pyramids_l1070_107089

theorem volume_ratio_of_trapezoidal_pyramids 
  (V U : ℝ) (m n m₁ n₁ : ℝ)
  (hV : V > 0) (hU : U > 0) (hm : m > 0) (hn : n > 0) (hm₁ : m₁ > 0) (hn₁ : n₁ > 0)
  (h_ratio : U / V = (m₁ + n₁)^2 / (m + n)^2) :
  U / V = (m₁ + n₁)^2 / (m + n)^2 :=
sorry

end volume_ratio_of_trapezoidal_pyramids_l1070_107089


namespace ninth_term_arithmetic_sequence_l1070_107070

theorem ninth_term_arithmetic_sequence :
  ∃ (a d : ℤ), (a + 2 * d = 5 ∧ a + 5 * d = 17) ∧ (a + 8 * d = 29) := 
by
  sorry

end ninth_term_arithmetic_sequence_l1070_107070


namespace circles_intersect_l1070_107095

-- Define the parameters and conditions given in the problem.
def r1 : ℝ := 5  -- Radius of circle O1
def r2 : ℝ := 8  -- Radius of circle O2
def d : ℝ := 8   -- Distance between the centers of O1 and O2

-- The main theorem that needs to be proven.
theorem circles_intersect (r1 r2 d : ℝ) (h_r1 : r1 = 5) (h_r2 : r2 = 8) (h_d : d = 8) :
  r2 - r1 < d ∧ d < r1 + r2 :=
by
  sorry

end circles_intersect_l1070_107095


namespace max_profit_l1070_107035

-- Define the given conditions
def cost_price : ℝ := 80
def sales_relationship (x : ℝ) : ℝ := -0.5 * x + 160
def selling_price_range (x : ℝ) : Prop := 120 ≤ x ∧ x ≤ 180

-- Define the profit function
def profit (x : ℝ) : ℝ := (x - cost_price) * sales_relationship x

-- The goal: prove the maximum profit and the selling price that achieves it
theorem max_profit : ∃ (x : ℝ), selling_price_range x ∧ profit x = 7000 := 
  sorry

end max_profit_l1070_107035


namespace consecutive_even_numbers_average_35_greatest_39_l1070_107008

-- Defining the conditions of the problem
def average_of_even_numbers (n : ℕ) (S : ℕ) : ℕ := (n * S + (2 * n * (n - 1)) / 2) / n

-- Main statement to be proven
theorem consecutive_even_numbers_average_35_greatest_39 : 
  ∃ (n : ℕ), average_of_even_numbers n (38 - (n - 1) * 2) = 35 ∧ (38 - (n - 1) * 2) + (n - 1) * 2 = 38 :=
by
  sorry

end consecutive_even_numbers_average_35_greatest_39_l1070_107008


namespace boat_speed_in_still_water_l1070_107088

theorem boat_speed_in_still_water (B S : ℝ) (h1 : B + S = 13) (h2 : B - S = 9) : B = 11 :=
by
  sorry

end boat_speed_in_still_water_l1070_107088


namespace trajectory_equation_l1070_107007

def fixed_point : ℝ × ℝ := (1, 2)

def moving_point (x y : ℝ) : ℝ × ℝ := (x, y)

def dot_product (p1 p2 : ℝ × ℝ) : ℝ :=
p1.1 * p2.1 + p1.2 * p2.2

theorem trajectory_equation (x y : ℝ) (h : dot_product (moving_point x y) fixed_point = 4) :
  x + 2 * y - 4 = 0 :=
sorry

end trajectory_equation_l1070_107007


namespace art_collection_area_l1070_107064

theorem art_collection_area :
  let square_paintings := 3 * (6 * 6)
  let small_paintings := 4 * (2 * 3)
  let large_painting := 1 * (10 * 15)
  square_paintings + small_paintings + large_painting = 282 := by
  sorry

end art_collection_area_l1070_107064


namespace square_area_eq_36_l1070_107091

theorem square_area_eq_36 :
  let triangle_side1 := 5.5
  let triangle_side2 := 7.5
  let triangle_side3 := 11
  let triangle_perimeter := triangle_side1 + triangle_side2 + triangle_side3
  let square_perimeter := triangle_perimeter
  let square_side_length := square_perimeter / 4
  let square_area := square_side_length * square_side_length
  square_area = 36 := by
  sorry

end square_area_eq_36_l1070_107091


namespace incorrect_positional_relationship_l1070_107013

-- Definitions for the geometric relationships
def line := Type
def plane := Type

def parallel (l : line) (α : plane) : Prop := sorry
def perpendicular (l : line) (α : plane) : Prop := sorry
def subset (l : line) (α : plane) : Prop := sorry
def distinct (l m : line) : Prop := l ≠ m

-- Given conditions
variables (l m : line) (α : plane)

-- Theorem statement: prove that D is incorrect given the conditions
theorem incorrect_positional_relationship
  (h_distinct : distinct l m)
  (h_parallel_l_α : parallel l α)
  (h_parallel_m_α : parallel m α) :
  ¬ (parallel l m) :=
sorry

end incorrect_positional_relationship_l1070_107013


namespace spencer_total_distance_l1070_107081

def distances : ℝ := 0.3 + 0.1 + 0.4

theorem spencer_total_distance :
  distances = 0.8 :=
sorry

end spencer_total_distance_l1070_107081


namespace last_digit_2_pow_1000_last_digit_3_pow_1000_last_digit_7_pow_1000_l1070_107012

-- Define the cycle period used in the problem
def cycle_period_2 := [2, 4, 8, 6]
def cycle_period_3 := [3, 9, 7, 1]
def cycle_period_7 := [7, 9, 3, 1]

-- Define a function to get the last digit from the cycle for given n
def last_digit_from_cycle (cycle : List ℕ) (n : ℕ) : ℕ :=
  let cycle_length := cycle.length
  cycle.get! ((n % cycle_length) - 1)

-- Problem statements
theorem last_digit_2_pow_1000 : last_digit_from_cycle cycle_period_2 1000 = 6 := sorry
theorem last_digit_3_pow_1000 : last_digit_from_cycle cycle_period_3 1000 = 1 := sorry
theorem last_digit_7_pow_1000 : last_digit_from_cycle cycle_period_7 1000 = 1 := sorry

end last_digit_2_pow_1000_last_digit_3_pow_1000_last_digit_7_pow_1000_l1070_107012


namespace count_solutions_l1070_107075

theorem count_solutions :
  ∃ (n : ℕ), (∀ (x y z : ℕ), x * y * z + x * y + y * z + z * x + x + y + z = 2012 ↔ n = 27) :=
sorry

end count_solutions_l1070_107075


namespace debate_students_handshake_l1070_107051

theorem debate_students_handshake 
    (S1 S2 S3 : ℕ)
    (h1 : S1 = 2 * S2)
    (h2 : S2 = S3 + 40)
    (h3 : S3 = 200) :
    S1 + S2 + S3 = 920 :=
by
  sorry

end debate_students_handshake_l1070_107051


namespace nth_term_correct_l1070_107084

noncomputable def nth_term (a b : ℝ) (n : ℕ) : ℝ :=
  (-1 : ℝ)^n * (2 * n - 1) * b / a^n

theorem nth_term_correct (a b : ℝ) (n : ℕ) (h : 0 < a) : 
  nth_term a b n = (-1 : ℝ)^↑n * (2 * n - 1) * b / a^n :=
by sorry

end nth_term_correct_l1070_107084


namespace rectangle_properties_l1070_107049

theorem rectangle_properties (w l : ℝ) (h₁ : l = 4 * w) (h₂ : 2 * l + 2 * w = 200) :
  ∃ A d, A = 1600 ∧ d = 82.46 := 
by {
  sorry
}

end rectangle_properties_l1070_107049


namespace least_n_for_distance_l1070_107068

theorem least_n_for_distance (n : ℕ) : n = 17 ↔ (100 ≤ n * (n + 1) / 3) := sorry

end least_n_for_distance_l1070_107068


namespace no_integer_solution_l1070_107077

theorem no_integer_solution (a b : ℤ) : ¬ (4 ∣ a^2 + b^2 + 1) :=
by
  -- Prevent use of the solution steps and add proof obligations
  sorry

end no_integer_solution_l1070_107077


namespace prob_both_primes_l1070_107065

-- Define the set of integers from 1 through 30
def int_set : Set ℕ := {n | 1 ≤ n ∧ n ≤ 30}

-- Define the set of prime numbers between 1 and 30
def primes_between_1_and_30 : Set ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

-- Calculate the number of ways to choose two distinct elements from a set
noncomputable def combination (n k : ℕ) : ℕ := if k > n then 0 else n.choose k

-- Define the probabilities
noncomputable def prob_primes : ℚ :=
  (combination 10 2) / (combination 30 2)

-- State the theorem to prove
theorem prob_both_primes : prob_primes = 10 / 87 := by
  sorry

end prob_both_primes_l1070_107065


namespace total_cost_is_130_l1070_107021

-- Defining the number of each type of pet
def n_puppies : ℕ := 2
def n_kittens : ℕ := 2
def n_parakeets : ℕ := 3

-- Defining the cost of one parakeet
def c_parakeet : ℕ := 10

-- Defining the cost of one puppy and one kitten based on the conditions
def c_puppy : ℕ := 3 * c_parakeet
def c_kitten : ℕ := 2 * c_parakeet

-- Defining the total cost of all pets
def total_cost : ℕ :=
  (n_puppies * c_puppy) + (n_kittens * c_kitten) + (n_parakeets * c_parakeet)

-- Lean theorem stating that the total cost is 130 dollars
theorem total_cost_is_130 : total_cost = 130 := by
  -- The proof will be filled in here.
  sorry

end total_cost_is_130_l1070_107021

import Mathlib

namespace box_depth_is_10_l2245_224577

variable (depth : ℕ)

theorem box_depth_is_10 
  (length width : ℕ)
  (cubes : ℕ)
  (h1 : length = 35)
  (h2 : width = 20)
  (h3 : cubes = 56)
  (h4 : ∃ (cube_size : ℕ), ∀ (c : ℕ), c = cube_size → (length % cube_size = 0 ∧ width % cube_size = 0 ∧ 56 * cube_size^3 = length * width * depth)) :
  depth = 10 :=
by
  sorry

end box_depth_is_10_l2245_224577


namespace edge_ratio_of_cubes_l2245_224568

theorem edge_ratio_of_cubes (a b : ℝ) (h : a^3 / b^3 = 27 / 8) : a / b = 3 / 2 :=
by
  sorry

end edge_ratio_of_cubes_l2245_224568


namespace p_or_q_iff_not_p_and_not_q_false_l2245_224515

variables (p q : Prop)

theorem p_or_q_iff_not_p_and_not_q_false : (p ∨ q) ↔ ¬(¬p ∧ ¬q) :=
by sorry

end p_or_q_iff_not_p_and_not_q_false_l2245_224515


namespace find_a_add_b_l2245_224510

theorem find_a_add_b (a b : ℝ) 
  (h1 : ∀ (x : ℝ), y = a + b / (x^2 + 1))
  (h2 : (y = 3) → (x = 1)) 
  (h3 : (y = 2) → (x = 0)) : a + b = 2 :=
by
  sorry

end find_a_add_b_l2245_224510


namespace repeating_decimal_addition_l2245_224561

def repeating_decimal_45 := (45 / 99 : ℚ)
def repeating_decimal_36 := (36 / 99 : ℚ)

theorem repeating_decimal_addition :
  repeating_decimal_45 + repeating_decimal_36 = 9 / 11 :=
by
  sorry

end repeating_decimal_addition_l2245_224561


namespace solve_for_a_l2245_224573

theorem solve_for_a (x a : ℝ) (h : x = 3) (eqn : 2 * (x - 1) - a = 0) : a = 4 := 
by 
  sorry

end solve_for_a_l2245_224573


namespace minimum_value_of_a_plus_4b_l2245_224584

variable (a b : ℝ)
variable (ha : a > 0)
variable (hb : b > 0)
variable (hgeo : Real.sqrt (a * b) = 2)

theorem minimum_value_of_a_plus_4b : a + 4 * b = 8 := by
  sorry

end minimum_value_of_a_plus_4b_l2245_224584


namespace dimes_given_l2245_224555

theorem dimes_given (initial_dimes final_dimes dimes_dad_gave : ℕ)
  (h1 : initial_dimes = 9)
  (h2 : final_dimes = 16)
  (h3 : final_dimes = initial_dimes + dimes_dad_gave) :
  dimes_dad_gave = 7 :=
by
  rw [h1, h2] at h3
  linarith

end dimes_given_l2245_224555


namespace integer_values_of_a_l2245_224585

theorem integer_values_of_a (x : ℤ) (a : ℤ)
  (h : x^3 + 3*x^2 + a*x + 11 = 0) :
  a = -155 ∨ a = -15 ∨ a = 13 ∨ a = 87 :=
sorry

end integer_values_of_a_l2245_224585


namespace lines_intersect_l2245_224538

theorem lines_intersect :
  ∃ x y : ℚ, 
  8 * x - 5 * y = 40 ∧ 
  6 * x - y = -5 ∧ 
  x = 15 / 38 ∧ 
  y = 140 / 19 :=
by { sorry }

end lines_intersect_l2245_224538


namespace smallest_number_l2245_224572

theorem smallest_number (a b c d e : ℕ) (h₁ : a = 12) (h₂ : b = 16) (h₃ : c = 18) (h₄ : d = 21) (h₅ : e = 28) : 
    ∃ n : ℕ, (n - 4) % Nat.lcm a (Nat.lcm b (Nat.lcm c (Nat.lcm d e))) = 0 ∧ n = 1012 :=
by
    sorry

end smallest_number_l2245_224572


namespace approx_val_l2245_224503

variable (x : ℝ) (y : ℝ)

-- Definitions based on rounding condition
def approx_0_000315 : ℝ := 0.0003
def approx_7928564 : ℝ := 8000000

-- Main theorem statement
theorem approx_val (h1: x = approx_0_000315) (h2: y = approx_7928564) :
  x * y = 2400 := by
  sorry

end approx_val_l2245_224503


namespace isosceles_triangle_base_angle_l2245_224529

theorem isosceles_triangle_base_angle
    (X : ℝ)
    (h1 : 0 < X)
    (h2 : 2 * X + X + X = 180)
    (h3 : X + X + 2 * X = 180) :
    X = 45 ∨ X = 72 :=
by sorry

end isosceles_triangle_base_angle_l2245_224529


namespace article_cost_price_l2245_224591

theorem article_cost_price :
  ∃ C : ℝ, 
  (1.05 * C) - 2 = (1.045 * C) ∧ 
  ∃ C_new : ℝ, C_new = (0.95 * C) ∧ ((1.045 * C) = (C_new + 0.1 * C_new)) ∧ C = 400 := 
sorry

end article_cost_price_l2245_224591


namespace cone_prism_volume_ratio_correct_l2245_224580

noncomputable def cone_prism_volume_ratio (π : ℝ) : ℝ :=
  let r := 1.5
  let h := 5
  let V_cone := (1 / 3) * π * r^2 * h
  let V_prism := 3 * 4 * h
  V_cone / V_prism

theorem cone_prism_volume_ratio_correct (π : ℝ) : 
  cone_prism_volume_ratio π = π / 4.8 :=
sorry

end cone_prism_volume_ratio_correct_l2245_224580


namespace cylindrical_container_volume_increase_l2245_224508

theorem cylindrical_container_volume_increase (R H : ℝ)
  (initial_volume : ℝ)
  (x : ℝ) : 
  R = 10 ∧ H = 5 ∧ initial_volume = π * R^2 * H →
  π * (R + 2 * x)^2 * H = π * R^2 * (H + 3 * x) →
  x = 5 :=
by
  -- Given conditions
  intro conditions volume_equation
  obtain ⟨hR, hH, hV⟩ := conditions
  -- Simplifying and solving the resulting equation
  sorry

end cylindrical_container_volume_increase_l2245_224508


namespace find_a7_l2245_224501

def seq (a : ℕ → ℚ) : Prop :=
  a 1 = -4/3 ∧ (∀ n, a (n + 2) = 1 / (a n + 1))

theorem find_a7 (a : ℕ → ℚ) (h : seq a) : a 7 = 2 :=
by
  sorry

end find_a7_l2245_224501


namespace determine_n_l2245_224504

theorem determine_n (k : ℕ) (n : ℕ) (h1 : 21^k ∣ n) (h2 : 7^k - k^7 = 1) : n = 1 :=
sorry

end determine_n_l2245_224504


namespace avg_score_calculation_l2245_224554

-- Definitions based on the conditions
def directly_proportional (a b : ℝ) : Prop := ∃ k, a = k * b

variables (score_math : ℝ) (score_science : ℝ)
variables (hours_math : ℝ := 4) (hours_science : ℝ := 5)
variables (next_hours_math_science : ℝ := 5)
variables (expected_avg_score : ℝ := 97.5)

axiom h1 : directly_proportional 80 4
axiom h2 : directly_proportional 95 5

-- Define the goal: Expected average score given the study hours next time
theorem avg_score_calculation :
  (score_math / hours_math = score_science / hours_science) →
  (score_math = 100 ∧ score_science = 95) →
  ((next_hours_math_science * score_math / hours_math + next_hours_math_science * score_science / hours_science) / 2 = expected_avg_score) :=
by sorry

end avg_score_calculation_l2245_224554


namespace value_of_2a_plus_b_l2245_224598

theorem value_of_2a_plus_b : ∀ (a b : ℝ), (∀ x : ℝ, x^2 - 4*x + 7 = 19 → (x = a ∨ x = b)) → a ≥ b → 2 * a + b = 10 :=
by
  intros a b h_sol h_order
  sorry

end value_of_2a_plus_b_l2245_224598


namespace team_not_losing_probability_l2245_224559

theorem team_not_losing_probability
  (p_center_forward : ℝ) (p_winger : ℝ) (p_attacking_midfielder : ℝ)
  (rate_center_forward : ℝ) (rate_winger : ℝ) (rate_attacking_midfielder : ℝ)
  (h_center_forward : p_center_forward = 0.2) (h_winger : p_winger = 0.5) (h_attacking_midfielder : p_attacking_midfielder = 0.3)
  (h_rate_center_forward : rate_center_forward = 0.4) (h_rate_winger : rate_winger = 0.2) (h_rate_attacking_midfielder : rate_attacking_midfielder = 0.2) :
  (p_center_forward * (1 - rate_center_forward) + p_winger * (1 - rate_winger) + p_attacking_midfielder * (1 - rate_attacking_midfielder)) = 0.76 :=
by
  sorry

end team_not_losing_probability_l2245_224559


namespace gcd_930_868_l2245_224506

theorem gcd_930_868 : Nat.gcd 930 868 = 62 := by
  sorry

end gcd_930_868_l2245_224506


namespace total_spent_l2245_224533

def price_almond_croissant : ℝ := 4.50
def price_salami_cheese_croissant : ℝ := 4.50
def price_plain_croissant : ℝ := 3.00
def price_focaccia : ℝ := 4.00
def price_latte : ℝ := 2.50
def num_lattes : ℕ := 2

theorem total_spent :
  price_almond_croissant + price_salami_cheese_croissant + price_plain_croissant +
  price_focaccia + (num_lattes * price_latte) = 21.00 := by
  sorry

end total_spent_l2245_224533


namespace triangle_inequality_l2245_224526

theorem triangle_inequality (a b c : ℝ) (h_nonneg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0)
  (h_triangle : (a^2 + b^2 > c^2) ∧ (b^2 + c^2 > a^2) ∧ (c^2 + a^2 > b^2)) :
  (a + b + c) * (a^2 + b^2 + c^2) * (a^3 + b^3 + c^3) ≥ 4 * (a^6 + b^6 + c^6) :=
sorry

end triangle_inequality_l2245_224526


namespace sufficient_condition_for_sets_l2245_224567

theorem sufficient_condition_for_sets (A B : Set ℝ) (m : ℝ) :
    (∀ x, x ∈ A → x ∈ B) → (m ≥ 3 / 4 ∨ m ≤ -3 / 4) :=
by
    have A_def : A = {y | ∃ x, y = x^2 - (3 / 2) * x + 1 ∧ (1 / 4) ≤ x ∧ x ≤ 2} := sorry
    have B_def : B = {x | x ≥ 1 - m^2} := sorry
    sorry

end sufficient_condition_for_sets_l2245_224567


namespace country_math_l2245_224569

theorem country_math (h : (1 / 3 : ℝ) * 4 = 6) : 
  ∃ x : ℝ, (1 / 6 : ℝ) * x = 15 ∧ x = 405 :=
by
  sorry

end country_math_l2245_224569


namespace charlie_max_success_ratio_l2245_224507

-- Given:
-- Alpha scored 180 points out of 360 attempted on day one.
-- Alpha scored 120 points out of 240 attempted on day two.
-- Charlie did not attempt 360 points on the first day.
-- Charlie's success ratio on each day was less than Alpha’s.
-- Total points attempted by Charlie on both days are 600.
-- Alpha's two-day success ratio is 300/600 = 1/2.
-- Find the largest possible two-day success ratio that Charlie could have achieved.

theorem charlie_max_success_ratio:
  ∀ (x y z w : ℕ),
  0 < x ∧ 0 < z ∧ 0 < y ∧ 0 < w ∧
  y + w = 600 ∧
  (2 * x < y) ∧ (2 * z < w) ∧
  (x + z < 300) -> (299 / 600 = 299 / 600) :=
by
  sorry

end charlie_max_success_ratio_l2245_224507


namespace amy_bike_total_l2245_224595

-- Define the miles Amy biked yesterday
def y : ℕ := 12

-- Define the miles Amy biked today
def t : ℕ := 2 * y - 3

-- Define the total miles Amy biked in two days
def total : ℕ := y + t

-- The theorem stating the total distance biked equals 33 miles
theorem amy_bike_total : total = 33 := by
  sorry

end amy_bike_total_l2245_224595


namespace part_a_l2245_224517

theorem part_a (x y : ℝ) 
  (h1 : x > 0) 
  (h2 : y > 0) 
  (h3 : x + y = 2) : 
  (1 / x + 1 / y) ≤ (1 / x^2 + 1 / y^2) := 
sorry

end part_a_l2245_224517


namespace village_population_l2245_224597

theorem village_population (P : ℕ) (h : 80 * P = 32000 * 100) : P = 40000 :=
sorry

end village_population_l2245_224597


namespace jenni_age_l2245_224596

theorem jenni_age (B J : ℕ) (h1 : B + J = 70) (h2 : B - J = 32) : J = 19 :=
by
  sorry

end jenni_age_l2245_224596


namespace mark_hours_per_week_l2245_224563

theorem mark_hours_per_week (w_historical : ℕ) (w_spring : ℕ) (h_spring : ℕ) (e_spring : ℕ) (e_goal : ℕ) (w_goal : ℕ) (h_goal : ℚ) :
  (e_spring : ℚ) / (w_historical * w_spring) = h_spring / w_spring →
  e_goal = 21000 →
  w_goal = 50 →
  h_spring = 35 →
  w_spring = 15 →
  e_spring = 4200 →
  (h_goal : ℚ) = 2625 / w_goal →
  h_goal = 52.5 :=
sorry

end mark_hours_per_week_l2245_224563


namespace circle_through_and_tangent_l2245_224556

noncomputable def circle_eq (a b r : ℝ) (x y : ℝ) : ℝ :=
  (x - a) ^ 2 + (y - b) ^ 2 - r ^ 2

theorem circle_through_and_tangent
(h1 : circle_eq 1 2 2 1 0 = 0)
(h2 : ∀ x y, circle_eq 1 2 2 x y = 0 → (x = 1 → y = 2 ∨ y = -2))
: ∀ x y, circle_eq 1 2 2 x y = 0 → (x - 1) ^ 2 + (y - 2) ^ 2 = 4 :=
by
  sorry

end circle_through_and_tangent_l2245_224556


namespace problem_l2245_224548

theorem problem {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a^2 + b^2 + c^2 + a*b*c = 4) : 
  a + b + c ≤ 3 := 
sorry

end problem_l2245_224548


namespace exist_students_with_comparable_scores_l2245_224543

theorem exist_students_with_comparable_scores :
  ∃ (A B : ℕ) (a1 a2 a3 b1 b2 b3 : ℕ), 
    A ≠ B ∧ A < 49 ∧ B < 49 ∧
    (0 ≤ a1 ∧ a1 ≤ 7) ∧ (0 ≤ a2 ∧ a2 ≤ 7) ∧ (0 ≤ a3 ∧ a3 ≤ 7) ∧ 
    (0 ≤ b1 ∧ b1 ≤ 7) ∧ (0 ≤ b2 ∧ b2 ≤ 7) ∧ (0 ≤ b3 ∧ b3 ≤ 7) ∧ 
    (a1 ≥ b1) ∧ (a2 ≥ b2) ∧ (a3 ≥ b3) := 
sorry

end exist_students_with_comparable_scores_l2245_224543


namespace vector_calculation_l2245_224594

namespace VectorProof

variables (a b : ℝ × ℝ) (m : ℝ)

def parallel (v1 v2 : ℝ × ℝ) : Prop := ∃ k : ℝ, v1 = (k • v2)

theorem vector_calculation
  (h₁ : a = (1, -2))
  (h₂ : b = (m, 4))
  (h₃ : parallel a b) :
  2 • a - b = (4, -8) :=
sorry

end VectorProof

end vector_calculation_l2245_224594


namespace roots_polynomial_identity_l2245_224520

theorem roots_polynomial_identity (a b x₁ x₂ : ℝ) 
  (h₁ : x₁^2 + b*x₁ + b^2 + a = 0) 
  (h₂ : x₂^2 + b*x₂ + b^2 + a = 0) : x₁^2 + x₁*x₂ + x₂^2 + a = 0 :=
by 
  sorry

end roots_polynomial_identity_l2245_224520


namespace claire_flour_cost_l2245_224537

def num_cakes : ℕ := 2
def flour_per_cake : ℕ := 2
def cost_per_flour : ℕ := 3
def total_cost (num_cakes flour_per_cake cost_per_flour : ℕ) : ℕ := 
  num_cakes * flour_per_cake * cost_per_flour

theorem claire_flour_cost : total_cost num_cakes flour_per_cake cost_per_flour = 12 := by
  sorry

end claire_flour_cost_l2245_224537


namespace negation_proposition_l2245_224516

theorem negation_proposition :
  (¬ ∃ x : ℝ, (x > -1 ∧ x < 3) ∧ (x^2 - 1 ≤ 2 * x)) ↔ 
  (∀ x : ℝ, (x > -1 ∧ x < 3) → (x^2 - 1 > 2 * x)) :=
by {
  sorry
}

end negation_proposition_l2245_224516


namespace sum_F_G_H_l2245_224582

theorem sum_F_G_H : 
  ∀ (F G H : ℕ), 
    (F < 10 ∧ G < 10 ∧ H < 10) ∧ 
    ∃ k : ℤ, 
      (F - 8 + 6 - 1 + G - 2 - H - 11 * k = 0) → 
        F + G + H = 23 :=
by sorry

end sum_F_G_H_l2245_224582


namespace triangle_enlargement_invariant_l2245_224575

theorem triangle_enlargement_invariant (α β γ : ℝ) (h_sum : α + β + γ = 180) (f : ℝ) :
  (α * f ≠ α) ∧ (β * f ≠ β) ∧ (γ * f ≠ γ) → (α * f + β * f + γ * f = 180 * f) → α + β + γ = 180 :=
by
  sorry

end triangle_enlargement_invariant_l2245_224575


namespace initial_caps_correct_l2245_224587

variable (bought : ℕ)
variable (total : ℕ)

def initial_bottle_caps (bought : ℕ) (total : ℕ) : ℕ :=
  total - bought

-- Given conditions
def bought_caps : ℕ := 7
def total_caps : ℕ := 47

theorem initial_caps_correct : initial_bottle_caps bought_caps total_caps = 40 :=
by
  -- proof here
  sorry

end initial_caps_correct_l2245_224587


namespace days_to_fulfill_order_l2245_224544

theorem days_to_fulfill_order (bags_per_batch : ℕ) (total_order : ℕ) (initial_bags : ℕ) (required_days : ℕ) :
  bags_per_batch = 10 →
  total_order = 60 →
  initial_bags = 20 →
  required_days = (total_order - initial_bags) / bags_per_batch →
  required_days = 4 :=
by
  intros
  sorry

end days_to_fulfill_order_l2245_224544


namespace math_problem_l2245_224534

variable (a b c : ℝ)

theorem math_problem (h1 : -10 ≤ a ∧ a < 0) (h2 : 0 < a ∧ a < b ∧ b < c) : 
  (a * c < b * c) ∧ (a + c < b + c) ∧ (c / a > 1) :=
by
  sorry

end math_problem_l2245_224534


namespace stripes_distance_l2245_224571

theorem stripes_distance (d : ℝ) (L : ℝ) (c : ℝ) (y : ℝ) 
  (hd : d = 40) (hL : L = 50) (hc : c = 15)
  (h_ratio : y / d = c / L) : y = 12 :=
by
  rw [hd, hL, hc] at h_ratio
  sorry

end stripes_distance_l2245_224571


namespace find_line_through_midpoint_of_hyperbola_l2245_224578

theorem find_line_through_midpoint_of_hyperbola
  (x1 y1 x2 y2 : ℝ)
  (P : ℝ × ℝ := (4, 1))
  (A : ℝ × ℝ := (x1, y1))
  (B : ℝ × ℝ := (x2, y2))
  (H_midpoint : P = ((x1 + x2) / 2, (y1 + y2) / 2))
  (H_hyperbola_A : (x1^2 / 4 - y1^2 = 1))
  (H_hyperbola_B : (x2^2 / 4 - y2^2 = 1)) :
  ∃ m b : ℝ, (m = 1) ∧ (b = 3) ∧ (∀ x y : ℝ, y = m * x + b → x - y - 3 = 0) := by
  sorry

end find_line_through_midpoint_of_hyperbola_l2245_224578


namespace find_number_of_eggs_l2245_224551

namespace HalloweenCleanup

def eggs (E : ℕ) (seconds_per_egg : ℕ) (minutes_per_roll : ℕ) (total_time : ℕ) (num_rolls : ℕ) : Prop :=
  seconds_per_egg = 15 ∧
  minutes_per_roll = 30 ∧
  total_time = 225 ∧
  num_rolls = 7 ∧
  E * (seconds_per_egg / 60) + num_rolls * minutes_per_roll = total_time

theorem find_number_of_eggs : ∃ E : ℕ, eggs E 15 30 225 7 :=
  by
    use 60
    unfold eggs
    simp
    exact sorry

end HalloweenCleanup

end find_number_of_eggs_l2245_224551


namespace largest_value_satisfies_abs_equation_l2245_224530

theorem largest_value_satisfies_abs_equation (x : ℝ) : |5 - x| = 15 + x → x = -5 := by
  intros h
  sorry

end largest_value_satisfies_abs_equation_l2245_224530


namespace students_without_an_A_l2245_224574

theorem students_without_an_A :
  ∀ (total_students : ℕ) (history_A : ℕ) (math_A : ℕ) (computing_A : ℕ)
    (math_and_history_A : ℕ) (history_and_computing_A : ℕ)
    (math_and_computing_A : ℕ) (all_three_A : ℕ),
  total_students = 40 →
  history_A = 10 →
  math_A = 18 →
  computing_A = 9 →
  math_and_history_A = 5 →
  history_and_computing_A = 3 →
  math_and_computing_A = 4 →
  all_three_A = 2 →
  total_students - (history_A + math_A + computing_A - math_and_history_A - history_and_computing_A - math_and_computing_A + all_three_A) = 13 :=
by
  intros total_students history_A math_A computing_A math_and_history_A history_and_computing_A math_and_computing_A all_three_A 
         ht_total_students ht_history_A ht_math_A ht_computing_A ht_math_and_history_A ht_history_and_computing_A ht_math_and_computing_A ht_all_three_A
  sorry

end students_without_an_A_l2245_224574


namespace exist_triangle_l2245_224592

-- Definitions of points and properties required in the conditions
structure Point :=
(x : ℝ) (y : ℝ)

def orthocenter (M : Point) := M 
def centroid (S : Point) := S 
def vertex (C : Point) := C 

-- The problem statement that needs to be proven
theorem exist_triangle (M S C : Point) 
    (h_orthocenter : orthocenter M = M)
    (h_centroid : centroid S = S)
    (h_vertex : vertex C = C) : 
    ∃ (A B : Point), 
        -- A, B, and C form a triangle ABC
        -- S is the centroid of this triangle
        -- M is the orthocenter of this triangle
        -- C is one of the vertices
        true := 
sorry

end exist_triangle_l2245_224592


namespace cross_fills_space_without_gaps_l2245_224541

structure Cube :=
(x : ℤ)
(y : ℤ)
(z : ℤ)

structure Cross :=
(center : Cube)
(adjacent : List Cube)

def is_adjacent (c1 c2 : Cube) : Prop :=
  (c1.x = c2.x ∧ c1.y = c2.y ∧ abs (c1.z - c2.z) = 1) ∨
  (c1.x = c2.x ∧ abs (c1.y - c2.y) = 1 ∧ c1.z = c2.z) ∨
  (abs (c1.x - c2.x) = 1 ∧ c1.y = c2.y ∧ c1.z = c2.z)

def valid_cross (c : Cross) : Prop :=
  ∀ (adj : Cube), adj ∈ c.adjacent → is_adjacent c.center adj

def fills_space (crosses : List Cross) : Prop :=
  ∀ (pos : Cube), ∃ (c : Cross), c ∈ crosses ∧ 
    (pos = c.center ∨ pos ∈ c.adjacent)

theorem cross_fills_space_without_gaps 
  (crosses : List Cross) 
  (Hcross : ∀ c ∈ crosses, valid_cross c) : 
  fills_space crosses :=
sorry

end cross_fills_space_without_gaps_l2245_224541


namespace sum_and_divide_repeating_decimals_l2245_224560

noncomputable def repeating_decimal_83 : ℚ := 83 / 99
noncomputable def repeating_decimal_18 : ℚ := 18 / 99

theorem sum_and_divide_repeating_decimals :
  (repeating_decimal_83 + repeating_decimal_18) / (1 / 5) = 505 / 99 :=
by
  sorry

end sum_and_divide_repeating_decimals_l2245_224560


namespace sarah_interviewed_students_l2245_224532

theorem sarah_interviewed_students :
  let oranges := 70
  let pears := 120
  let apples := 147
  let strawberries := 113
  oranges + pears + apples + strawberries = 450 := by
sorry

end sarah_interviewed_students_l2245_224532


namespace ratio_of_times_l2245_224589

-- Given conditions as definitions
def distance : ℕ := 630 -- distance in km
def previous_time : ℕ := 6 -- time in hours
def new_speed : ℕ := 70 -- speed in km/h

-- Calculation of times
def previous_speed : ℕ := distance / previous_time

def new_time : ℕ := distance / new_speed

-- Main theorem statement
theorem ratio_of_times :
  (new_time : ℚ) / (previous_time : ℚ) = 3 / 2 :=
  sorry

end ratio_of_times_l2245_224589


namespace tablets_taken_l2245_224535

theorem tablets_taken (total_time interval_time : ℕ) (h1 : total_time = 60) (h2 : interval_time = 15) : total_time / interval_time = 4 :=
by
  sorry

end tablets_taken_l2245_224535


namespace smallest_of_consecutive_even_numbers_l2245_224509

theorem smallest_of_consecutive_even_numbers (n : ℤ) (h : ∃ a b c : ℤ, a % 2 = 0 ∧ b % 2 = 0 ∧ c % 2 = 0 ∧ b = a + 2 ∧ c = a + 4 ∧ c = 2 * n + 1) :
  ∃ a b c : ℤ, a % 2 = 0 ∧ b % 2 = 0 ∧ c % 2 = 0 ∧ b = a + 2 ∧ c = a + 4 ∧ a = 2 * n - 3 :=
by
  sorry

end smallest_of_consecutive_even_numbers_l2245_224509


namespace max_difference_two_digit_numbers_l2245_224550

theorem max_difference_two_digit_numbers (A B : ℤ) (hA : 10 ≤ A ∧ A ≤ 99) (hB : 10 ≤ B ∧ B ≤ 99) (h : 2 * A * 3 = 2 * B * 7) : 
  56 ≤ A - B :=
sorry

end max_difference_two_digit_numbers_l2245_224550


namespace proof_prob_at_least_one_die_3_or_5_l2245_224565

def probability_at_least_one_die_3_or_5 (total_outcomes : ℕ) (favorable_outcomes : ℕ) : ℚ :=
  favorable_outcomes / total_outcomes

theorem proof_prob_at_least_one_die_3_or_5 :
  let total_outcomes := 36
  let favorable_outcomes := 20
  probability_at_least_one_die_3_or_5 total_outcomes favorable_outcomes = 5 / 9 := 
by 
  sorry

end proof_prob_at_least_one_die_3_or_5_l2245_224565


namespace sides_of_triangle_inequality_l2245_224518

theorem sides_of_triangle_inequality {a b c : ℝ} (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) :
  a^4 + b^4 + c^4 - 2*a^2*b^2 - 2*b^2*c^2 - 2*c^2*a^2 < 0 :=
by
  sorry

end sides_of_triangle_inequality_l2245_224518


namespace unique_real_root_of_quadratic_l2245_224558

theorem unique_real_root_of_quadratic (k : ℝ) :
  (∃ a : ℝ, ∀ b : ℝ, ((k^2 - 9) * b^2 - 2 * (k + 1) * b + 1 = 0 → b = a)) ↔ (k = 3 ∨ k = -3 ∨ k = -5) :=
by
  sorry

end unique_real_root_of_quadratic_l2245_224558


namespace smallest_n_l2245_224593

theorem smallest_n (n : ℕ) (h1 : n % 6 = 4) (h2 : n % 7 = 3) (h3 : n % 8 = 5) (h4 : n > 20) : n = 136 := by
  sorry

end smallest_n_l2245_224593


namespace determine_a_l2245_224513

theorem determine_a (a : ℝ) : (∃! x : ℝ, a * x^2 + 2 * x + 1 = 0) → (a = 0 ∨ a = 1) := 
sorry

end determine_a_l2245_224513


namespace expression_meaningful_l2245_224502

theorem expression_meaningful (x : ℝ) : 
  (x - 1 ≠ 0 ∧ true) ↔ x ≠ 1 := 
sorry

end expression_meaningful_l2245_224502


namespace storks_more_than_birds_l2245_224547

theorem storks_more_than_birds :
  let initial_birds := 3
  let additional_birds := 2
  let storks := 6
  storks - (initial_birds + additional_birds) = 1 :=
by
  sorry

end storks_more_than_birds_l2245_224547


namespace sum_of_four_consecutive_integers_is_even_l2245_224528

theorem sum_of_four_consecutive_integers_is_even (n : ℤ) : 2 ∣ ((n - 1) + n + (n + 1) + (n + 2)) :=
by sorry

end sum_of_four_consecutive_integers_is_even_l2245_224528


namespace length_of_second_edge_l2245_224566

-- Define the edge lengths and volume
def edge1 : ℕ := 6
def edge3 : ℕ := 6
def volume : ℕ := 180

-- The theorem to state the length of the second edge
theorem length_of_second_edge (edge2 : ℕ) (h : edge1 * edge2 * edge3 = volume) :
  edge2 = 5 :=
by
  -- Skipping the proof
  sorry

end length_of_second_edge_l2245_224566


namespace correct_statement_of_abs_l2245_224581

theorem correct_statement_of_abs (r : ℚ) :
  ¬ (∀ r : ℚ, abs r > 0) ∧
  ¬ (∀ a b : ℚ, a ≠ b → abs a ≠ abs b) ∧
  (∀ r : ℚ, abs r ≥ 0) ∧
  ¬ (∀ r : ℚ, r < 0 → abs r = -r ∧ abs r < 0 → abs r ≠ -r) :=
by
  sorry

end correct_statement_of_abs_l2245_224581


namespace find_h_neg_one_l2245_224552

theorem find_h_neg_one (h : ℝ → ℝ) (H : ∀ x, (x^7 - 1) * h x = (x + 1) * (x^2 + 1) * (x^4 + 1) + 1) : 
  h (-1) = 1 := 
by 
  sorry

end find_h_neg_one_l2245_224552


namespace find_d_l2245_224505

namespace NineDigitNumber

variables {A B C D E F G : ℕ}

theorem find_d 
  (h1 : 6 + A + B = 13) 
  (h2 : A + B + C = 13)
  (h3 : B + C + D = 13)
  (h4 : C + D + E = 13)
  (h5 : D + E + F = 13)
  (h6 : E + F + G = 13)
  (h7 : F + G + 3 = 13) :
  D = 4 :=
sorry

end NineDigitNumber

end find_d_l2245_224505


namespace retrievers_count_l2245_224542

-- Definitions of given conditions
def huskies := 5
def pitbulls := 2
def retrievers := Nat
def husky_pups := 3
def pitbull_pups := 3
def retriever_extra_pups := 2
def total_pups_excess := 30

-- Equation derived from the problem conditions
def total_pups (G : Nat) := huskies * husky_pups + pitbulls * pitbull_pups + G * (husky_pups + retriever_extra_pups)
def total_adults (G : Nat) := huskies + pitbulls + G

theorem retrievers_count : ∃ G : Nat, G = 4 ∧ total_pups G = total_adults G + total_pups_excess :=
by
  sorry

end retrievers_count_l2245_224542


namespace gcd_lcm_product_360_l2245_224588

theorem gcd_lcm_product_360 (a b : ℕ) (h : Nat.gcd a b * Nat.lcm a b = 360) :
    {d : ℕ | d = Nat.gcd a b } =
    {1, 2, 4, 8, 3, 6, 12, 24} := 
by
  sorry

end gcd_lcm_product_360_l2245_224588


namespace cylinder_volume_ratio_l2245_224527

theorem cylinder_volume_ratio (h_C r_D : ℝ) (V_C V_D : ℝ) :
  h_C = 3 * r_D →
  r_D = h_C →
  V_C = 3 * V_D →
  V_C = (1 / 9) * π * h_C^3 :=
by
  sorry

end cylinder_volume_ratio_l2245_224527


namespace accounting_vs_calling_clients_l2245_224545

/--
Given:
1. Total time Maryann worked today is 560 minutes.
2. Maryann spent 70 minutes calling clients.

Prove:
Maryann spends 7 times longer doing accounting than calling clients.
-/
theorem accounting_vs_calling_clients 
  (total_time : ℕ) 
  (calling_time : ℕ) 
  (h_total : total_time = 560) 
  (h_calling : calling_time = 70) : 
  (total_time - calling_time) / calling_time = 7 :=
  sorry

end accounting_vs_calling_clients_l2245_224545


namespace number_of_piles_l2245_224524

-- Defining the number of walnuts in total
def total_walnuts : Nat := 55

-- Defining the number of walnuts in the first pile
def first_pile_walnuts : Nat := 7

-- Defining the number of walnuts in each of the rest of the piles
def other_pile_walnuts : Nat := 12

-- The proposition we want to prove
theorem number_of_piles (n : Nat) :
  (n > 1) →
  (other_pile_walnuts * (n - 1) + first_pile_walnuts = total_walnuts) → n = 5 :=
sorry

end number_of_piles_l2245_224524


namespace ceil_square_of_neg_five_thirds_l2245_224514

theorem ceil_square_of_neg_five_thirds : Int.ceil ((-5 / 3:ℚ)^2) = 3 := by
  sorry

end ceil_square_of_neg_five_thirds_l2245_224514


namespace points_per_game_l2245_224500

theorem points_per_game (total_points : ℝ) (num_games : ℝ) (h1 : total_points = 120.0) (h2 : num_games = 10.0) : (total_points / num_games) = 12.0 :=
by 
  rw [h1, h2]
  norm_num
  -- sorry


end points_per_game_l2245_224500


namespace rectangle_area_l2245_224525

theorem rectangle_area (AB AD AE : ℝ) (S_trapezoid S_triangle : ℝ) (perim_triangle perim_trapezoid : ℝ)
  (h1 : AD - AB = 9)
  (h2 : S_trapezoid = 5 * S_triangle)
  (h3 : perim_triangle + 68 = perim_trapezoid)
  (h4 : S_trapezoid + S_triangle = S_triangle * 6)
  (h5 : perim_triangle = AB + AE + (AE - AB))
  (h6 : perim_trapezoid = AB + AD + AE + (2 * (AD - AE))) :
  AD * AB = 3060 := by
  sorry

end rectangle_area_l2245_224525


namespace amount_subtracted_correct_l2245_224523

noncomputable def find_subtracted_amount (N : ℝ) (A : ℝ) : Prop :=
  0.40 * N - A = 23

theorem amount_subtracted_correct :
  find_subtracted_amount 85 11 :=
by
  sorry

end amount_subtracted_correct_l2245_224523


namespace sum_of_slopes_correct_l2245_224590

noncomputable def sum_of_slopes : ℚ :=
  let Γ1 := {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 - 1)^2 = 1}
  let Γ2 := {p : ℝ × ℝ | (p.1 - 10)^2 + (p.2 - 11)^2 = 1}
  let l := {k : ℝ | ∃ p1 ∈ Γ1, ∃ p2 ∈ Γ1, ∃ p3 ∈ Γ2, ∃ p4 ∈ Γ2, p1 ≠ p2 ∧ p3 ≠ p4 ∧ p1.2 = k * p1.1 ∧ p3.2 = k * p3.1}
  let valid_slopes := {k | k ∈ l ∧ (k = 11/10 ∨ k = 1 ∨ k = 5/4)}
  (11 / 10) + 1 + (5 / 4)

theorem sum_of_slopes_correct : sum_of_slopes = 67 / 20 := 
  by sorry

end sum_of_slopes_correct_l2245_224590


namespace geometric_series_sum_condition_l2245_224536

def geometric_series_sum (a q n : ℕ) : ℕ := a * (1 - q^n) / (1 - q)

theorem geometric_series_sum_condition (S : ℕ → ℕ) (a : ℕ) (q : ℕ) (h1 : a = 1) 
  (h2 : ∀ n, S n = geometric_series_sum a q n)
  (h3 : S 7 - 4 * S 6 + 3 * S 5 = 0) : 
  S 4 = 40 := 
by 
  sorry

end geometric_series_sum_condition_l2245_224536


namespace range_of_a_l2245_224512

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) :
  (∀ x, f (2^x) = x^2 - 2 * a * x + a^2 - 1) →
  (∀ x, 2^(a-1) ≤ x ∧ x ≤ 2^(a^2 - 2*a + 2) → -1 ≤ f x ∧ f x ≤ 0) →
  ((3 - Real.sqrt 5) / 2 ≤ a ∧ a ≤ 1) ∨ (2 ≤ a ∧ a ≤ (3 + Real.sqrt 5) / 2) :=
by
  sorry

end range_of_a_l2245_224512


namespace value_of_ratios_l2245_224562

variable (x y z : ℝ)

-- Conditions
def geometric_sequence : Prop :=
  4 * y / (3 * x) = 5 * z / (4 * y)

def arithmetic_sequence : Prop :=
  2 / y = 1 / x + 1 / z

-- Theorem/Proof Statement
theorem value_of_ratios (h1 : geometric_sequence x y z) (h2 : arithmetic_sequence x y z) :
  (x / z) + (z / x) = 34 / 15 :=
by
  sorry

end value_of_ratios_l2245_224562


namespace stuffed_animal_cost_is_6_l2245_224549

-- Definitions for the costs of items
def sticker_cost (s : ℕ) := s
def magnet_cost (m : ℕ) := m
def stuffed_animal_cost (a : ℕ) := a

-- Conditions given in the problem
def conditions (m s a : ℕ) :=
  (m = 3) ∧
  (m = 3 * s) ∧
  (m = (2 * a) / 4)

-- The theorem stating the cost of a single stuffed animal
theorem stuffed_animal_cost_is_6 (s m a : ℕ) (h : conditions m s a) : a = 6 :=
by
  sorry

end stuffed_animal_cost_is_6_l2245_224549


namespace fraction_boxes_loaded_by_day_crew_l2245_224576

variables {D W_d : ℝ}

theorem fraction_boxes_loaded_by_day_crew
  (h1 : ∀ (D W_d: ℝ), D > 0 → W_d > 0 → ∃ (D' W_n : ℝ), (D' = 0.5 * D) ∧ (W_n = 0.8 * W_d))
  (h2 : ∃ (D W_d : ℝ), ∀ (D' W_n : ℝ), (D' = 0.5 * D) → (W_n = 0.8 * W_d) → 
        (D * W_d / (D * W_d + D' * W_n)) = (5 / 7)) :
  (∃ (D W_d : ℝ), D > 0 → W_d > 0 → (D * W_d)/(D * W_d + 0.5 * D * 0.8 * W_d) = (5/7)) := 
  sorry 

end fraction_boxes_loaded_by_day_crew_l2245_224576


namespace no_real_solutions_for_equation_l2245_224531

theorem no_real_solutions_for_equation (x : ℝ) : ¬(∃ x : ℝ, (8 * x^2 + 150 * x - 5) / (3 * x + 50) = 4 * x + 7) :=
sorry

end no_real_solutions_for_equation_l2245_224531


namespace B_investment_amount_l2245_224570

-- Define given conditions in Lean 4

def A_investment := 400
def total_months := 12
def B_investment_months := 6
def total_profit := 100
def A_share := 80
def B_share := total_profit - A_share

-- The problem statement in Lean 4 that needs to be proven:
theorem B_investment_amount (A_investment B_investment_months total_profit A_share B_share: ℕ)
  (hA_investment : A_investment = 400)
  (htotal_months : total_months = 12)
  (hB_investment_months : B_investment_months = 6)
  (htotal_profit : total_profit = 100)
  (hA_share : A_share = 80)
  (hB_share : B_share = total_profit - A_share) 
  : (∃ (B: ℕ), 
       (5 * (A_investment * total_months) = 4 * (400 * total_months + B * B_investment_months)) 
       ∧ B = 200) :=
sorry

end B_investment_amount_l2245_224570


namespace total_amount_spent_l2245_224522

def cost_of_tshirt : ℕ := 100
def cost_of_pants : ℕ := 250
def num_of_tshirts : ℕ := 5
def num_of_pants : ℕ := 4

theorem total_amount_spent : (num_of_tshirts * cost_of_tshirt) + (num_of_pants * cost_of_pants) = 1500 := by
  sorry

end total_amount_spent_l2245_224522


namespace simplify_and_evaluate_expression_l2245_224539

theorem simplify_and_evaluate_expression (x : ℝ) (h : x = Real.sqrt 2 + 3) :
  ( (x^2 - 1) / (x^2 - 6 * x + 9) * (1 - x / (x - 1)) / ((x + 1) / (x - 3)) ) = - (Real.sqrt 2 / 2) :=
  sorry

end simplify_and_evaluate_expression_l2245_224539


namespace travel_ways_A_to_C_l2245_224583

-- We define the number of ways to travel from A to B
def ways_A_to_B : ℕ := 3

-- We define the number of ways to travel from B to C
def ways_B_to_C : ℕ := 2

-- We state the problem as a theorem
theorem travel_ways_A_to_C : ways_A_to_B * ways_B_to_C = 6 :=
by
  sorry

end travel_ways_A_to_C_l2245_224583


namespace evaluate_expression_l2245_224519

theorem evaluate_expression (a b : ℕ) (ha : a = 3) (hb : b = 2) :
  (a^4 + b^4) / (a^2 - a * b + b^2) = 97 / 7 := by
  sorry

example : (3^4 + 2^4) / (3^2 - 3 * 2 + 2^2) = 97 / 7 := evaluate_expression 3 2 rfl rfl

end evaluate_expression_l2245_224519


namespace find_number_l2245_224579

theorem find_number (f : ℝ → ℝ) (x : ℝ)
  (h : f (x * 0.004) / 0.03 = 9.237333333333334)
  (h_linear : ∀ a, f a = a) :
  x = 69.3 :=
by
  -- Proof goes here
  sorry

end find_number_l2245_224579


namespace mass_percentage_C_in_CaCO3_is_correct_l2245_224546

structure Element where
  name : String
  molar_mass : ℚ

def Ca : Element := ⟨"Ca", 40.08⟩
def C : Element := ⟨"C", 12.01⟩
def O : Element := ⟨"O", 16.00⟩

def molar_mass_CaCO3 : ℚ :=
  Ca.molar_mass + C.molar_mass + 3 * O.molar_mass

def mass_percentage_C_in_CaCO3 : ℚ :=
  (C.molar_mass / molar_mass_CaCO3) * 100

theorem mass_percentage_C_in_CaCO3_is_correct :
  mass_percentage_C_in_CaCO3 = 12.01 :=
by
  sorry

end mass_percentage_C_in_CaCO3_is_correct_l2245_224546


namespace quadratic_inequality_solution_l2245_224511

theorem quadratic_inequality_solution :
  ∀ x : ℝ, (x^2 - 4*x + 3) < 0 ↔ 1 < x ∧ x < 3 :=
by
  sorry

end quadratic_inequality_solution_l2245_224511


namespace sandy_spent_on_shirt_l2245_224540

-- Define the conditions
def cost_of_shorts : ℝ := 13.99
def cost_of_jacket : ℝ := 7.43
def total_spent_on_clothes : ℝ := 33.56

-- Define the amount spent on the shirt
noncomputable def cost_of_shirt : ℝ :=
  total_spent_on_clothes - (cost_of_shorts + cost_of_jacket)

-- Prove that Sandy spent $12.14 on the shirt
theorem sandy_spent_on_shirt : cost_of_shirt = 12.14 :=
by
  sorry

end sandy_spent_on_shirt_l2245_224540


namespace cafeteria_green_apples_l2245_224521

def number_of_green_apples (G : ℕ) : Prop :=
  42 + G - 9 = 40 → G = 7

theorem cafeteria_green_apples
  (red_apples : ℕ)
  (students_wanting_fruit : ℕ)
  (extra_fruit : ℕ)
  (G : ℕ)
  (h1 : red_apples = 42)
  (h2 : students_wanting_fruit = 9)
  (h3 : extra_fruit = 40)
  : number_of_green_apples G :=
by
  -- Place for proof omitted intentionally
  sorry

end cafeteria_green_apples_l2245_224521


namespace range_of_f_l2245_224553

open Set

noncomputable def f (x : ℝ) : ℝ := 3^x + 5

theorem range_of_f :
  range f = Ioi 5 :=
sorry

end range_of_f_l2245_224553


namespace percentage_increase_in_freelance_l2245_224557

open Real

def initial_part_time_earnings := 65
def new_part_time_earnings := 72
def initial_freelance_earnings := 45
def new_freelance_earnings := 72

theorem percentage_increase_in_freelance :
  (new_freelance_earnings - initial_freelance_earnings) / initial_freelance_earnings * 100 = 60 :=
by
  -- Proof will go here
  sorry

end percentage_increase_in_freelance_l2245_224557


namespace compare_powers_l2245_224586

theorem compare_powers (a b c d : ℝ) (h1 : a + b = 0) (h2 : c + d = 0) : a^5 + d^6 = c^6 - b^5 :=
by
  sorry

end compare_powers_l2245_224586


namespace sin_cos_value_l2245_224564

theorem sin_cos_value (x : ℝ) (h : Real.cos x - 3 * Real.sin x = 2) :
  (3 * Real.sin x + Real.cos x = 0) ∨ (3 * Real.sin x + Real.cos x = -4) :=
sorry

end sin_cos_value_l2245_224564


namespace pure_alcohol_addition_l2245_224599

variable (x : ℝ)

def initial_volume : ℝ := 6
def initial_concentration : ℝ := 0.25
def final_concentration : ℝ := 0.50

theorem pure_alcohol_addition :
  (1.5 + x) / (initial_volume + x) = final_concentration → x = 3 :=
by
  sorry

end pure_alcohol_addition_l2245_224599

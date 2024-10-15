import Mathlib

namespace NUMINAMATH_GPT_triangle_is_isosceles_l1348_134879

variable (A B C : ℝ)
variable (a b c : ℝ)
variable (triangle : Type)

noncomputable def is_isosceles_triangle (A B C : ℝ) (a b c : ℝ) (triangle : Type) : Prop :=
  c = 2 * a * Real.cos B → A = B ∨ B = C ∨ C = A

theorem triangle_is_isosceles (A B C : ℝ) (a b c : ℝ) (triangle : Type) (h : c = 2 * a * Real.cos B) :
  is_isosceles_triangle A B C a b c triangle :=
sorry

end NUMINAMATH_GPT_triangle_is_isosceles_l1348_134879


namespace NUMINAMATH_GPT_total_houses_is_160_l1348_134804

namespace MariamNeighborhood

-- Define the given conditions as variables in Lean.
def houses_on_one_side : ℕ := 40
def multiplier : ℕ := 3

-- Define the number of houses on the other side of the road.
def houses_on_other_side : ℕ := multiplier * houses_on_one_side

-- Define the total number of houses in Mariam's neighborhood.
def total_houses : ℕ := houses_on_one_side + houses_on_other_side

-- Prove that the total number of houses is 160.
theorem total_houses_is_160 : total_houses = 160 :=
by
  -- Placeholder for proof
  sorry

end MariamNeighborhood

end NUMINAMATH_GPT_total_houses_is_160_l1348_134804


namespace NUMINAMATH_GPT_Kim_nail_polishes_l1348_134891

-- Define the conditions
variable (K : ℕ)
def Heidi_nail_polishes (K : ℕ) : ℕ := K + 5
def Karen_nail_polishes (K : ℕ) : ℕ := K - 4

-- The main statement to prove
theorem Kim_nail_polishes (K : ℕ) (H : Heidi_nail_polishes K + Karen_nail_polishes K = 25) : K = 12 := by
  sorry

end NUMINAMATH_GPT_Kim_nail_polishes_l1348_134891


namespace NUMINAMATH_GPT_beth_sold_l1348_134826

theorem beth_sold {initial_coins additional_coins total_coins sold_coins : ℕ} 
  (h_init : initial_coins = 125)
  (h_add : additional_coins = 35)
  (h_total : total_coins = initial_coins + additional_coins)
  (h_sold : sold_coins = total_coins / 2) :
  sold_coins = 80 := 
sorry

end NUMINAMATH_GPT_beth_sold_l1348_134826


namespace NUMINAMATH_GPT_arun_age_l1348_134866

theorem arun_age (A G M : ℕ) (h1 : (A - 6) / 18 = G) (h2 : G = M - 2) (h3 : M = 5) : A = 60 :=
by
  sorry

end NUMINAMATH_GPT_arun_age_l1348_134866


namespace NUMINAMATH_GPT_equation_no_solution_at_5_l1348_134817

theorem equation_no_solution_at_5 :
  ∀ (some_expr : ℝ), ¬(1 / (5 + 5) + some_expr = 1 / (5 - 5)) :=
by
  intro some_expr
  sorry

end NUMINAMATH_GPT_equation_no_solution_at_5_l1348_134817


namespace NUMINAMATH_GPT_year_2024_AD_representation_l1348_134869

def year_representation (y: Int) : Int :=
  if y > 0 then y else -y

theorem year_2024_AD_representation : year_representation 2024 = 2024 :=
by sorry

end NUMINAMATH_GPT_year_2024_AD_representation_l1348_134869


namespace NUMINAMATH_GPT_total_trees_after_planting_l1348_134835

def initial_trees : ℕ := 39
def trees_planted_today : ℕ := 41
def trees_planted_tomorrow : ℕ := 20

theorem total_trees_after_planting :
  initial_trees + trees_planted_today + trees_planted_tomorrow = 100 := 
by sorry

end NUMINAMATH_GPT_total_trees_after_planting_l1348_134835


namespace NUMINAMATH_GPT_consistent_scale_l1348_134805

-- Conditions definitions

def dist_gardensquare_newtonsville : ℕ := 3  -- in inches
def dist_newtonsville_madison : ℕ := 4  -- in inches
def speed_gardensquare_newtonsville : ℕ := 50  -- mph
def time_gardensquare_newtonsville : ℕ := 2  -- hours
def speed_newtonsville_madison : ℕ := 60  -- mph
def time_newtonsville_madison : ℕ := 3  -- hours

-- Actual distances calculated
def actual_distance_gardensquare_newtonsville : ℕ := speed_gardensquare_newtonsville * time_gardensquare_newtonsville
def actual_distance_newtonsville_madison : ℕ := speed_newtonsville_madison * time_newtonsville_madison

-- Prove the scale is consistent across the map
theorem consistent_scale :
  actual_distance_gardensquare_newtonsville / dist_gardensquare_newtonsville =
  actual_distance_newtonsville_madison / dist_newtonsville_madison :=
by
  sorry

end NUMINAMATH_GPT_consistent_scale_l1348_134805


namespace NUMINAMATH_GPT_children_group_size_l1348_134893

theorem children_group_size (x : ℕ) (h1 : 255 % 17 = 0) (h2: ∃ n : ℕ, n * 17 = 255) 
                            (h3 : ∀ a c, a = c → a = 255 → c = 255 → x = 17) : 
                            (255 / x = 15) → x = 17 :=
by
  sorry

end NUMINAMATH_GPT_children_group_size_l1348_134893


namespace NUMINAMATH_GPT_min_value_fraction_l1348_134888

noncomputable section

open Real

theorem min_value_fraction (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 2 * y = 4) : 
  ∃ t : ℝ, (∀ x' y' : ℝ, (x' > 0 ∧ y' > 0 ∧ x' + 2 * y' = 4) → (2 / x' + 1 / y') ≥ t) ∧ t = 2 :=
by
  sorry

end NUMINAMATH_GPT_min_value_fraction_l1348_134888


namespace NUMINAMATH_GPT_binom_18_6_eq_13260_l1348_134812

/-- The binomial coefficient formula. -/
def binom (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- The specific proof problem: compute binom(18, 6) and show that it equals 13260. -/
theorem binom_18_6_eq_13260 : binom 18 6 = 13260 :=
by
  sorry

end NUMINAMATH_GPT_binom_18_6_eq_13260_l1348_134812


namespace NUMINAMATH_GPT_lucy_picked_more_l1348_134813

variable (Mary Peter Lucy : ℕ)
variable (Mary_amt Peter_amt Lucy_amt : ℕ)

-- Conditions
def mary_amount : Mary_amt = 12 := sorry
def twice_as_peter : Mary_amt = 2 * Peter_amt := sorry
def total_picked : Mary_amt + Peter_amt + Lucy_amt = 26 := sorry

-- Statement to Prove
theorem lucy_picked_more (h1: Mary_amt = 12) (h2: Mary_amt = 2 * Peter_amt) (h3: Mary_amt + Peter_amt + Lucy_amt = 26) :
  Lucy_amt - Peter_amt = 2 := 
sorry

end NUMINAMATH_GPT_lucy_picked_more_l1348_134813


namespace NUMINAMATH_GPT_alpha_parallel_to_beta_l1348_134849

variables (a b : ℝ → ℝ → ℝ) (α β : ℝ → ℝ)

-- Definitions based on conditions
def are_distinct_lines : a ≠ b := sorry
def are_distinct_planes : α ≠ β := sorry

def line_parallel_to_plane (l : ℝ → ℝ → ℝ) (p : ℝ → ℝ) : Prop := sorry -- Define parallel relation
def line_perpendicular_to_plane (l : ℝ → ℝ → ℝ) (p : ℝ → ℝ) : Prop := sorry -- Define perpendicular relation
def planes_parallel (p1 p2 : ℝ → ℝ) : Prop := sorry -- Define planes being parallel

-- Given as conditions
axiom a_perpendicular_to_alpha : line_perpendicular_to_plane a α
axiom b_perpendicular_to_beta : line_perpendicular_to_plane b β
axiom a_parallel_to_b : a = b

-- The proposition to prove
theorem alpha_parallel_to_beta : planes_parallel α β :=
by {
  -- Placeholder for the logic provided through the previous solution steps.
  sorry
}

end NUMINAMATH_GPT_alpha_parallel_to_beta_l1348_134849


namespace NUMINAMATH_GPT_alfred_saving_goal_l1348_134842

theorem alfred_saving_goal (leftover : ℝ) (monthly_saving : ℝ) (months : ℕ) :
  leftover = 100 → monthly_saving = 75 → months = 12 → leftover + monthly_saving * months = 1000 :=
by
  sorry

end NUMINAMATH_GPT_alfred_saving_goal_l1348_134842


namespace NUMINAMATH_GPT_cost_formula_l1348_134896

def cost (P : ℕ) : ℕ :=
  if P ≤ 5 then 5 * P + 10 else 5 * P + 5

theorem cost_formula (P : ℕ) : 
  cost P = (if P ≤ 5 then 5 * P + 10 else 5 * P + 5) :=
by 
  sorry

end NUMINAMATH_GPT_cost_formula_l1348_134896


namespace NUMINAMATH_GPT_burpees_percentage_contribution_l1348_134845

theorem burpees_percentage_contribution :
  let total_time : ℝ := 20
  let jumping_jacks : ℝ := 30
  let pushups : ℝ := 22
  let situps : ℝ := 45
  let burpees : ℝ := 15
  let lunges : ℝ := 25

  let jumping_jacks_rate := jumping_jacks / total_time
  let pushups_rate := pushups / total_time
  let situps_rate := situps / total_time
  let burpees_rate := burpees / total_time
  let lunges_rate := lunges / total_time

  let total_rate := jumping_jacks_rate + pushups_rate + situps_rate + burpees_rate + lunges_rate

  (burpees_rate / total_rate) * 100 = 10.95 :=
by
  sorry

end NUMINAMATH_GPT_burpees_percentage_contribution_l1348_134845


namespace NUMINAMATH_GPT_negation_of_proposition_l1348_134863

theorem negation_of_proposition : (¬ ∃ x : ℝ, x > 0 ∧ x^2 - 5 * x + 6 > 0) ↔ ∀ x : ℝ, x > 0 → x^2 - 5 * x + 6 ≤ 0 := by
  sorry

end NUMINAMATH_GPT_negation_of_proposition_l1348_134863


namespace NUMINAMATH_GPT_unique_k_largest_n_l1348_134858

theorem unique_k_largest_n :
  ∃! k : ℤ, ∃ n : ℕ, (n > 0) ∧ (5 / 18 < n / (n + k) ∧ n / (n + k) < 9 / 17) ∧ (n = 1) :=
by
  sorry

end NUMINAMATH_GPT_unique_k_largest_n_l1348_134858


namespace NUMINAMATH_GPT_evaluate_f_at_points_l1348_134820

def f (x : ℝ) : ℝ :=
  3 * x ^ 2 - 6 * x + 10

theorem evaluate_f_at_points : 3 * f 2 + 2 * f (-2) = 98 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_f_at_points_l1348_134820


namespace NUMINAMATH_GPT_shpuntik_can_form_triangle_l1348_134885

theorem shpuntik_can_form_triangle 
  (x1 x2 x3 y1 y2 y3 : ℝ)
  (hx : x1 + x2 + x3 = 1)
  (hy : y1 + y2 + y3 = 1)
  (infeasibility_vintik : x1 ≥ x2 + x3) :
  ∃ (a b c : ℝ), a + b + c = 1 ∧ a < b + c ∧ b < a + c ∧ c < a + b :=
sorry

end NUMINAMATH_GPT_shpuntik_can_form_triangle_l1348_134885


namespace NUMINAMATH_GPT_option_b_not_valid_l1348_134811

theorem option_b_not_valid (a b c d : ℝ) (h_arith_seq : b - a = d ∧ c - b = d ∧ d ≠ 0) : 
  a^3 * b + b^3 * c + c^3 * a < a^4 + b^4 + c^4 :=
by sorry

end NUMINAMATH_GPT_option_b_not_valid_l1348_134811


namespace NUMINAMATH_GPT_find_m_if_divisible_by_11_l1348_134822

theorem find_m_if_divisible_by_11 : ∃ m : ℕ, m < 10 ∧ (734000000 + m*100000 + 8527) % 11 = 0 ↔ m = 6 :=
by {
    sorry
}

end NUMINAMATH_GPT_find_m_if_divisible_by_11_l1348_134822


namespace NUMINAMATH_GPT_tuition_fee_l1348_134801

theorem tuition_fee (R T : ℝ) (h1 : T + R = 2584) (h2 : T = R + 704) : T = 1644 := by sorry

end NUMINAMATH_GPT_tuition_fee_l1348_134801


namespace NUMINAMATH_GPT_central_angle_correct_l1348_134839

-- Define arc length, radius, and central angle
variables (l r α : ℝ)

-- Given conditions
def arc_length := 3
def radius := 2

-- Theorem to prove
theorem central_angle_correct : (l = arc_length) → (r = radius) → (l = r * α) → α = 3 / 2 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_central_angle_correct_l1348_134839


namespace NUMINAMATH_GPT_Eddie_number_divisibility_l1348_134838

theorem Eddie_number_divisibility (n: ℕ) (h₁: n = 40) (h₂: n % 5 = 0): n % 2 = 0 := 
by
  sorry

end NUMINAMATH_GPT_Eddie_number_divisibility_l1348_134838


namespace NUMINAMATH_GPT_full_tank_capacity_l1348_134841

theorem full_tank_capacity (speed : ℝ) (gas_usage_per_mile : ℝ) (time : ℝ) (gas_used_fraction : ℝ) (distance_per_tank : ℝ) (gallons_used : ℝ)
  (h1 : speed = 50)
  (h2 : gas_usage_per_mile = 1 / 30)
  (h3 : time = 5)
  (h4 : gas_used_fraction = 0.8333333333333334)
  (h5 : distance_per_tank = speed * time)
  (h6 : gallons_used = distance_per_tank * gas_usage_per_mile)
  (h7 : gallons_used = 0.8333333333333334 * 10) :
  distance_per_tank / 30 / 0.8333333333333334 = 10 :=
by sorry

end NUMINAMATH_GPT_full_tank_capacity_l1348_134841


namespace NUMINAMATH_GPT_greatest_possible_selling_price_l1348_134843

variable (products : ℕ)
variable (average_price : ℝ)
variable (min_price : ℝ)
variable (less_than_1000_products : ℕ)

theorem greatest_possible_selling_price
  (h1 : products = 20)
  (h2 : average_price = 1200)
  (h3 : min_price = 400)
  (h4 : less_than_1000_products = 10) :
  ∃ max_price, max_price = 11000 := 
by
  sorry

end NUMINAMATH_GPT_greatest_possible_selling_price_l1348_134843


namespace NUMINAMATH_GPT_part1_part2_l1348_134860

theorem part1 (m : ℝ) :
  ∀ x : ℝ, x^2 + ( (2 * m - 1) : ℝ) * x + m^2 = 0 → m ≤ 1 / 4 :=
sorry

theorem part2 (m : ℝ) 
  (h : ∀ x1 x2 : ℝ, (x1^2 + (2*m -1)*x1 + m^2 = 0) ∧ (x2^2 + (2*m -1)*x2 + m^2 = 0) ∧ (x1*x2 + x1 + x2 = 4)) :
    m = -1 :=
sorry

end NUMINAMATH_GPT_part1_part2_l1348_134860


namespace NUMINAMATH_GPT_prob_teamB_wins_first_game_l1348_134815
-- Import the necessary library

-- Define the conditions and the question in a Lean theorem statement
theorem prob_teamB_wins_first_game :
  (∀ (win_A win_B : ℕ), win_A < 4 ∧ win_B = 4) →
  (∀ (team_wins_game : ℕ → Prop), (team_wins_game 2 = false) ∧ (team_wins_game 3 = true)) →
  (∀ (team_wins_series : Prop), team_wins_series = (win_B ≥ 4 ∧ win_A < 4)) →
  (∀ (game_outcome_distribution : ℕ → ℕ → ℕ → ℕ → ℚ), game_outcome_distribution 4 4 2 2 = 1 / 2) →
  (∀ (first_game_outcome : Prop), first_game_outcome = true) →
  true :=
sorry

end NUMINAMATH_GPT_prob_teamB_wins_first_game_l1348_134815


namespace NUMINAMATH_GPT_problem_solution_l1348_134834

noncomputable def a (n : ℕ) : ℕ := 2 * n - 3

noncomputable def b (n : ℕ) : ℕ := 2 ^ n

noncomputable def c (n : ℕ) : ℕ := a n * b n

noncomputable def sum_c (n : ℕ) : ℕ :=
  (2 * n - 5) * 2 ^ (n + 1) + 10

theorem problem_solution :
  ∀ n : ℕ, n > 0 →
  (S_n = 2 * (b n - 1)) ∧
  (a 2 = b 1 - 1) ∧
  (a 5 = b 3 - 1)
  →
  (∀ n, a n = 2 * n - 3) ∧
  (∀ n, b n = 2 ^ n) ∧
  (sum_c n = (2 * n - 5) * 2 ^ (n + 1) + 10) :=
by
  intros n hn h
  sorry


end NUMINAMATH_GPT_problem_solution_l1348_134834


namespace NUMINAMATH_GPT_hexagon_perimeter_eq_4_sqrt_3_over_3_l1348_134824

noncomputable def hexagon_perimeter (s : ℝ) : ℝ :=
  6 * s

theorem hexagon_perimeter_eq_4_sqrt_3_over_3 :
  ∀ (s : ℝ), (∃ s, (3 * Real.sqrt 3 / 2) * s^2 = s) → hexagon_perimeter s = 4 * Real.sqrt 3 / 3 :=
by
  simp
  sorry

end NUMINAMATH_GPT_hexagon_perimeter_eq_4_sqrt_3_over_3_l1348_134824


namespace NUMINAMATH_GPT_committee_vote_change_l1348_134833

-- Let x be the number of votes for the resolution initially.
-- Let y be the number of votes against the resolution initially.
-- The total number of voters is 500: x + y = 500.
-- The initial margin by which the resolution was defeated: y - x = m.
-- In the re-vote, the resolution passed with a margin three times the initial margin: x' - y' = 3m.
-- The number of votes for the re-vote was 13/12 of the votes against initially: x' = 13/12 * y.
-- The total number of voters remains 500 in the re-vote: x' + y' = 500.

theorem committee_vote_change (x y x' y' m : ℕ)
  (h1 : x + y = 500)
  (h2 : y - x = m)
  (h3 : x' - y' = 3 * m)
  (h4 : x' = 13 * y / 12)
  (h5 : x' + y' = 500) : x' - x = 40 := 
  by
  sorry

end NUMINAMATH_GPT_committee_vote_change_l1348_134833


namespace NUMINAMATH_GPT_same_different_color_ways_equal_l1348_134828

-- Definitions based on conditions in the problem
def num_black : ℕ := 15
def num_white : ℕ := 10

def same_color_ways : ℕ :=
  Nat.choose num_black 2 + Nat.choose num_white 2

def different_color_ways : ℕ :=
  num_black * num_white

-- The proof statement
theorem same_different_color_ways_equal : same_color_ways = different_color_ways :=
by
  sorry

end NUMINAMATH_GPT_same_different_color_ways_equal_l1348_134828


namespace NUMINAMATH_GPT_friends_meeting_probability_l1348_134856

noncomputable def n_value (d e f : ℝ) (h1 : d = 60) (h2 : e = 30) (h3 : f = 2) : ℝ :=
  d - e * Real.sqrt f

theorem friends_meeting_probability (n : ℝ) (d e f : ℝ) (h1 : d = 60) (h2 : e = 30) (h3 : f = 2)
  (H : n = n_value d e f h1 h2 h3) : d + e + f = 92 :=
  by
  sorry

end NUMINAMATH_GPT_friends_meeting_probability_l1348_134856


namespace NUMINAMATH_GPT_product_of_integers_whose_cubes_sum_to_189_l1348_134806

theorem product_of_integers_whose_cubes_sum_to_189 :
  ∃ (a b : ℤ), a^3 + b^3 = 189 ∧ a * b = 20 :=
by
  sorry

end NUMINAMATH_GPT_product_of_integers_whose_cubes_sum_to_189_l1348_134806


namespace NUMINAMATH_GPT_calculate_group5_students_l1348_134895

variable (total_students : ℕ) (freq_group1 : ℕ) (sum_freq_group2_3 : ℝ) (freq_group4 : ℝ)

theorem calculate_group5_students
  (h1 : total_students = 50)
  (h2 : freq_group1 = 7)
  (h3 : sum_freq_group2_3 = 0.46)
  (h4 : freq_group4 = 0.2) :
  (total_students * (1 - (freq_group1 / total_students + sum_freq_group2_3 + freq_group4)) = 10) :=
by
  sorry

end NUMINAMATH_GPT_calculate_group5_students_l1348_134895


namespace NUMINAMATH_GPT_sum_of_midpoint_coordinates_l1348_134894

theorem sum_of_midpoint_coordinates 
  (x1 y1 z1 x2 y2 z2 : ℝ) 
  (h1 : (x1, y1, z1) = (2, 3, 4)) 
  (h2 : (x2, y2, z2) = (8, 15, 12)) : 
  (x1 + x2) / 2 + (y1 + y2) / 2 + (z1 + z2) / 2 = 22 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_midpoint_coordinates_l1348_134894


namespace NUMINAMATH_GPT_problem_l1348_134873

theorem problem (p q : ℝ) (h : 5 * p^2 - 20 * p + 15 = 0 ∧ 5 * q^2 - 20 * q + 15 = 0) : (p * q - 3)^2 = 0 := 
sorry

end NUMINAMATH_GPT_problem_l1348_134873


namespace NUMINAMATH_GPT_problem1_problem2_l1348_134865

def A (x : ℝ) : Prop := -1 ≤ x ∧ x ≤ 7
def S (x : ℝ) (k : ℝ) : Prop := k + 1 ≤ x ∧ x ≤ 2 * k - 1

theorem problem1 (k : ℝ) : (∀ x, S x k → A x) → k ≤ 4 :=
by
  sorry

theorem problem2 (k : ℝ) : (∀ x, ¬(A x ∧ S x k)) → k < 2 ∨ k > 6 :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1348_134865


namespace NUMINAMATH_GPT_simplify_expression_solve_inequality_system_l1348_134877

-- Problem 1
theorem simplify_expression (m n : ℝ) (h1 : 3 * m - 2 * n ≠ 0) (h2 : 3 * m + 2 * n ≠ 0) (h3 : 9 * m ^ 2 - 4 * n ^ 2 ≠ 0) :
  ((1 / (3 * m - 2 * n) - 1 / (3 * m + 2 * n)) / (m * n / (9 * m ^ 2 - 4 * n ^ 2))) = (4 / m) :=
sorry

-- Problem 2
theorem solve_inequality_system (x : ℝ) (h1 : 3 * x + 10 > 5 * x - 2 * (5 - x)) (h2 : (x + 3) / 5 > 1 - x) :
  1 / 3 < x ∧ x < 5 :=
sorry

end NUMINAMATH_GPT_simplify_expression_solve_inequality_system_l1348_134877


namespace NUMINAMATH_GPT_students_in_classes_saved_money_strategy_class7_1_l1348_134831

-- Part (1): Prove the number of students in each class
theorem students_in_classes (x : ℕ) (h1 : 40 < x) (h2 : x < 50) 
  (h3 : 105 - x > 50) (h4 : 15 * x + 12 * (105 - x) = 1401) : x = 47 ∧ (105 - x) = 58 := by
  sorry

-- Part (2): Prove the amount saved by purchasing tickets together
theorem saved_money(amt_per_ticket : ℕ → ℕ) 
  (h1 : amt_per_ticket 105 = 1401) 
  (h2 : ∀n, n > 100 → amt_per_ticket n = 1050) : amt_per_ticket 105 - 1050 = 351 := by
  sorry

-- Part (3): Strategy to save money for class 7 (1)
theorem strategy_class7_1 (students_1 : ℕ) (h1 : students_1 = 47) 
  (cost_15 : students_1 * 15 = 705) 
  (cost_51 : 51 * 12 = 612) : 705 - 612 = 93 := by
  sorry

end NUMINAMATH_GPT_students_in_classes_saved_money_strategy_class7_1_l1348_134831


namespace NUMINAMATH_GPT_solve_quadratic_inequality_l1348_134851

theorem solve_quadratic_inequality (a : ℝ) (x : ℝ) :
  (x^2 - a * x + a - 1 ≤ 0) ↔
  (a < 2 ∧ a - 1 ≤ x ∧ x ≤ 1) ∨
  (a = 2 ∧ x = 1) ∨
  (a > 2 ∧ 1 ≤ x ∧ x ≤ a - 1) := 
by
  sorry

end NUMINAMATH_GPT_solve_quadratic_inequality_l1348_134851


namespace NUMINAMATH_GPT_trig_identity_l1348_134800

theorem trig_identity (α : ℝ) (h : Real.tan α = 2) :
    Real.cos (2 * α) - Real.sin α * Real.cos α = -1 := 
by 
  sorry

end NUMINAMATH_GPT_trig_identity_l1348_134800


namespace NUMINAMATH_GPT_principal_amount_borrowed_l1348_134889

theorem principal_amount_borrowed (P : ℝ) (R : ℝ) (T : ℝ) (SI : ℝ) 
  (R_eq : R = 12) (T_eq : T = 3) (SI_eq : SI = 7200) :
  (SI = (P * R * T) / 100) → P = 20000 :=
by sorry

end NUMINAMATH_GPT_principal_amount_borrowed_l1348_134889


namespace NUMINAMATH_GPT_cost_per_mile_first_plan_l1348_134880

theorem cost_per_mile_first_plan 
  (initial_fee : ℝ) (cost_per_mile_first : ℝ) (cost_per_mile_second : ℝ) (miles : ℝ)
  (h_first : initial_fee = 65)
  (h_cost_second : cost_per_mile_second = 0.60)
  (h_miles : miles = 325)
  (h_equal_cost : initial_fee + miles * cost_per_mile_first = miles * cost_per_mile_second) :
  cost_per_mile_first = 0.40 :=
by
  sorry

end NUMINAMATH_GPT_cost_per_mile_first_plan_l1348_134880


namespace NUMINAMATH_GPT_problem1_problem2_problem3_l1348_134899

-- Definition of sets A, B, and U
def A : Set ℤ := {1, 2, 3, 4, 5}
def B : Set ℤ := {-1, 1, 2, 3}
def U : Set ℤ := {x | -1 ≤ x ∧ x < 6}

-- The complement of B in U
def C_U (B : Set ℤ) : Set ℤ := {x ∈ U | x ∉ B}

-- Problem statements
theorem problem1 : A ∩ B = {1, 2, 3} := by sorry
theorem problem2 : A ∪ B = {-1, 1, 2, 3, 4, 5} := by sorry
theorem problem3 : (C_U B) ∩ A = {4, 5} := by sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_l1348_134899


namespace NUMINAMATH_GPT_polar_to_cartesian_l1348_134878

theorem polar_to_cartesian (r θ : ℝ) (h_r : r = 2) (h_θ : θ = π / 6) :
  (r * Real.cos θ, r * Real.sin θ) = (Real.sqrt 3, 1) :=
by
  rw [h_r, h_θ]
  have h_cos : Real.cos (π / 6) = Real.sqrt 3 / 2 := sorry -- This identity can be used from trigonometric property.
  have h_sin : Real.sin (π / 6) = 1 / 2 := sorry -- This identity can be used from trigonometric property.
  rw [h_cos, h_sin]
  -- some algebraic steps to simplifiy left sides to (Real.sqrt 3, 1) should follow here. using multiplication and commmutaivity properties mainly.
  sorry

end NUMINAMATH_GPT_polar_to_cartesian_l1348_134878


namespace NUMINAMATH_GPT_dog_total_bones_l1348_134847

-- Define the number of original bones and dug up bones as constants
def original_bones : ℕ := 493
def dug_up_bones : ℕ := 367

-- Define the total bones the dog has now
def total_bones : ℕ := original_bones + dug_up_bones

-- State and prove the theorem
theorem dog_total_bones : total_bones = 860 := by
  -- placeholder for the proof
  sorry

end NUMINAMATH_GPT_dog_total_bones_l1348_134847


namespace NUMINAMATH_GPT_initial_overs_played_l1348_134814

-- Define the conditions
def initial_run_rate : ℝ := 6.2
def remaining_overs : ℝ := 40
def remaining_run_rate : ℝ := 5.5
def target_runs : ℝ := 282

-- Define what we seek to prove
theorem initial_overs_played :
  ∃ x : ℝ, (6.2 * x) + (5.5 * 40) = 282 ∧ x = 10 :=
by
  sorry

end NUMINAMATH_GPT_initial_overs_played_l1348_134814


namespace NUMINAMATH_GPT_circle_center_radius_sum_correct_l1348_134853

noncomputable def circle_center_radius_sum (eq : String) : ℝ :=
  if h : eq = "x^2 + 8x - 2y^2 - 6y = -6" then
    let c : ℝ := -4
    let d : ℝ := -3 / 2
    let s : ℝ := Real.sqrt (47 / 4)
    c + d + s
  else 0

theorem circle_center_radius_sum_correct :
  circle_center_radius_sum "x^2 + 8x - 2y^2 - 6y = -6" = (-11 + Real.sqrt 47) / 2 :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_circle_center_radius_sum_correct_l1348_134853


namespace NUMINAMATH_GPT_fraction_equality_l1348_134872

theorem fraction_equality : (18 / (5 * 107 + 3) = 18 / 538) := 
by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_fraction_equality_l1348_134872


namespace NUMINAMATH_GPT_find_f_l1348_134870

theorem find_f (f : ℝ → ℝ) (h : ∀ x : ℝ, 2 * f x - f (-x) = 3 * x + 1) : ∀ x : ℝ, f x = x + 1 :=
by
  sorry

end NUMINAMATH_GPT_find_f_l1348_134870


namespace NUMINAMATH_GPT_min_value_expression_l1348_134890

theorem min_value_expression (x : ℝ) (hx : x > 0) : 9 * x + 1 / x^3 ≥ 10 :=
sorry

end NUMINAMATH_GPT_min_value_expression_l1348_134890


namespace NUMINAMATH_GPT_bd_squared_l1348_134844

theorem bd_squared (a b c d : ℤ) (h1 : a - b - c + d = 13) (h2 : a + b - c - d = 9) : 
  (b - d) ^ 2 = 4 := 
sorry

end NUMINAMATH_GPT_bd_squared_l1348_134844


namespace NUMINAMATH_GPT_adult_tickets_l1348_134875

theorem adult_tickets (A C : ℕ) (h1 : A + C = 130) (h2 : 12 * A + 4 * C = 840) : A = 40 :=
by {
  -- Proof omitted
  sorry
}

end NUMINAMATH_GPT_adult_tickets_l1348_134875


namespace NUMINAMATH_GPT_part_a_part_b_l1348_134848

theorem part_a (x y : ℕ) (h : x^3 + 5 * y = y^3 + 5 * x) : x = y :=
sorry

theorem part_b : ∃ (x y : ℝ), x ≠ y ∧ x > 0 ∧ y > 0 ∧ (x^3 + 5 * y = y^3 + 5 * x) :=
sorry

end NUMINAMATH_GPT_part_a_part_b_l1348_134848


namespace NUMINAMATH_GPT_final_position_total_distance_l1348_134819

-- Define the movements as a list
def movements : List Int := [-8, 7, -3, 9, -6, -4, 10]

-- Prove that the final position of the turtle is 5 meters north of the starting point
theorem final_position (movements : List Int) (h : movements = [-8, 7, -3, 9, -6, -4, 10]) : List.sum movements = 5 :=
by
  rw [h]
  sorry

-- Prove that the total distance crawled by the turtle is 47 meters
theorem total_distance (movements : List Int) (h : movements = [-8, 7, -3, 9, -6, -4, 10]) : List.sum (List.map Int.natAbs movements) = 47 :=
by
  rw [h]
  sorry

end NUMINAMATH_GPT_final_position_total_distance_l1348_134819


namespace NUMINAMATH_GPT_cars_sold_on_second_day_l1348_134864

theorem cars_sold_on_second_day (x : ℕ) 
  (h1 : 14 + x + 27 = 57) : x = 16 :=
by 
  sorry

end NUMINAMATH_GPT_cars_sold_on_second_day_l1348_134864


namespace NUMINAMATH_GPT_min_f_l1348_134827

noncomputable def f (x y z : ℝ) : ℝ :=
  (3 * x^2 - x) / (1 + x^2) + 
  (3 * y^2 - y) / (1 + y^2) + 
  (3 * z^2 - z) / (1 + z^2)

theorem min_f (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 1) :
  f x y z = 0 ↔ x = 1/3 ∧ y = 1/3 ∧ z = 1/3 :=
sorry

end NUMINAMATH_GPT_min_f_l1348_134827


namespace NUMINAMATH_GPT_xiao_pang_xiao_ya_books_l1348_134868

theorem xiao_pang_xiao_ya_books : 
  ∀ (x y : ℕ), 
    (x + 2 * x = 66) → 
    (y + y / 3 = 92) → 
    (2 * x = 2 * x) → 
    (y = 3 * (y / 3)) → 
    ((22 + 69) - (2 * 22 + 69 / 3) = 24) :=
by
  intros x y h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_xiao_pang_xiao_ya_books_l1348_134868


namespace NUMINAMATH_GPT_no_natural_number_n_exists_l1348_134802

theorem no_natural_number_n_exists :
  ∀ (n : ℕ), ¬ ∃ (a b : ℕ), 3 * n + 1 = a * b := by
  sorry

end NUMINAMATH_GPT_no_natural_number_n_exists_l1348_134802


namespace NUMINAMATH_GPT_vector_combination_l1348_134898

-- Definitions of the given vectors and condition of parallelism
def vec_a : (ℝ × ℝ) := (1, -2)
def vec_b (m : ℝ) : (ℝ × ℝ) := (2, m)
def are_parallel (a b : ℝ × ℝ) : Prop := a.1 * b.2 - a.2 * b.1 = 0

-- Goal to prove
theorem vector_combination :
  ∀ m : ℝ, are_parallel vec_a (vec_b m) → 3 * vec_a.1 + 2 * (vec_b m).1 = 7 ∧ 3 * vec_a.2 + 2 * (vec_b m).2 = -14 :=
by
  intros m h_par
  sorry

end NUMINAMATH_GPT_vector_combination_l1348_134898


namespace NUMINAMATH_GPT_pencils_ratio_l1348_134859

theorem pencils_ratio 
  (cindi_pencils : ℕ := 60)
  (marcia_mul_cindi : ℕ := 2)
  (total_pencils : ℕ := 480)
  (marcia_pencils : ℕ := marcia_mul_cindi * cindi_pencils) 
  (donna_pencils : ℕ := total_pencils - marcia_pencils) :
  donna_pencils / marcia_pencils = 3 := by
  sorry

end NUMINAMATH_GPT_pencils_ratio_l1348_134859


namespace NUMINAMATH_GPT_line_circle_no_intersect_l1348_134809

theorem line_circle_no_intersect :
  ∀ (x y : ℝ), (3 * x + 4 * y = 12) → (x^2 + y^2 = 4) → false :=
by
  -- use the given equations to demonstrate the lack of intersection points
  sorry

end NUMINAMATH_GPT_line_circle_no_intersect_l1348_134809


namespace NUMINAMATH_GPT_ratio_pentagon_rectangle_l1348_134857

theorem ratio_pentagon_rectangle (s_p w : ℝ) (H_pentagon : 5 * s_p = 60) (H_rectangle : 6 * w = 80) : s_p / w = 9 / 10 :=
by
  sorry

end NUMINAMATH_GPT_ratio_pentagon_rectangle_l1348_134857


namespace NUMINAMATH_GPT_solve_for_x_l1348_134861

theorem solve_for_x (x y : ℤ) (h1 : x + 3 * y = 33) (h2 : y = 10) : x = 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1348_134861


namespace NUMINAMATH_GPT_rat_to_chihuahua_ratio_is_six_to_one_l1348_134803

noncomputable def chihuahuas_thought_to_be : ℕ := 70
noncomputable def actual_rats : ℕ := 60

theorem rat_to_chihuahua_ratio_is_six_to_one
    (h : chihuahuas_thought_to_be - actual_rats = 10) :
    actual_rats / (chihuahuas_thought_to_be - actual_rats) = 6 :=
by
  sorry

end NUMINAMATH_GPT_rat_to_chihuahua_ratio_is_six_to_one_l1348_134803


namespace NUMINAMATH_GPT_no_such_two_digit_number_exists_l1348_134854

theorem no_such_two_digit_number_exists :
  ¬ ∃ (x y : ℕ), 1 ≤ x ∧ x ≤ 9 ∧ 0 ≤ y ∧ y ≤ 9 ∧
                 (10 * x + y = 2 * (x^2 + y^2) + 6) ∧
                 (10 * x + y = 4 * (x * y) + 6) := by
  -- We need to prove that no two-digit number satisfies
  -- both conditions.
  sorry

end NUMINAMATH_GPT_no_such_two_digit_number_exists_l1348_134854


namespace NUMINAMATH_GPT_determine_cards_per_friend_l1348_134825

theorem determine_cards_per_friend (n_cards : ℕ) (n_friends : ℕ) (h : n_cards = 12) : n_friends > 0 → (n_cards / n_friends) = (12 / n_friends) :=
by
  sorry

end NUMINAMATH_GPT_determine_cards_per_friend_l1348_134825


namespace NUMINAMATH_GPT_find_ck_l1348_134829

theorem find_ck 
  (d r : ℕ)                -- d : common difference, r : common ratio
  (k : ℕ)                  -- k : integer such that certain conditions hold
  (hn2 : (k-2) > 0)        -- ensure (k-2) > 0
  (hk1 : (k+1) > 0)        -- ensure (k+1) > 0
  (h1 : 1 + (k-3) * d + r^(k-3) = 120) -- c_{k-1} = 120
  (h2 : 1 + k * d + r^k = 1200) -- c_{k+1} = 1200
  : (1 + (k-1) * d + r^(k-1)) = 263 := -- c_k = 263
sorry

end NUMINAMATH_GPT_find_ck_l1348_134829


namespace NUMINAMATH_GPT_gumball_draw_probability_l1348_134846

def prob_blue := 2 / 3
def prob_two_blue := (16 / 36)
def prob_pink := 1 - prob_blue

theorem gumball_draw_probability
    (h1 : prob_two_blue = prob_blue * prob_blue)
    (h2 : prob_blue + prob_pink = 1) :
    prob_pink = 1 / 3 := 
by
  sorry

end NUMINAMATH_GPT_gumball_draw_probability_l1348_134846


namespace NUMINAMATH_GPT_point_B_coordinates_l1348_134874

-- Defining the vector a
def vec_a : ℝ × ℝ := (1, 0)

-- Defining the point A
def A : ℝ × ℝ := (4, 4)

-- Definition of the line y = 2x
def on_line (P : ℝ × ℝ) : Prop := P.2 = 2 * P.1

-- Defining a vector as being parallel to another vector
def parallel (v w : ℝ × ℝ) : Prop := v.1 * w.2 = v.2 * w.1

-- Lean statement for the proof
theorem point_B_coordinates (B : ℝ × ℝ) (h1 : on_line B) (h2 : parallel (B.1 - 4, B.2 - 4) vec_a) :
  B = (2, 4) :=
sorry

end NUMINAMATH_GPT_point_B_coordinates_l1348_134874


namespace NUMINAMATH_GPT_random_event_is_option_D_l1348_134852

-- Definitions based on conditions
def rains_without_clouds : Prop := false
def like_charges_repel : Prop := true
def seeds_germinate_without_moisture : Prop := false
def draw_card_get_1 : Prop := true

-- Proof statement
theorem random_event_is_option_D : 
  (¬ rains_without_clouds ∧ like_charges_repel ∧ ¬ seeds_germinate_without_moisture ∧ draw_card_get_1) →
  (draw_card_get_1 = true) :=
by sorry

end NUMINAMATH_GPT_random_event_is_option_D_l1348_134852


namespace NUMINAMATH_GPT_expand_product_l1348_134837

theorem expand_product (x : ℝ) (hx : x ≠ 0) : (3 / 7) * (7 / x - 5 * x ^ 3) = 3 / x - (15 / 7) * x ^ 3 :=
by
  sorry

end NUMINAMATH_GPT_expand_product_l1348_134837


namespace NUMINAMATH_GPT_find_m_for_parallel_lines_l1348_134881

theorem find_m_for_parallel_lines (m : ℝ) :
  (∀ x y, 2 * x + (m + 1) * y + 4 = 0 → mx + 3 * y - 2 = 0 → 
  -((2 : ℝ) / (m + 1)) = -(m / 3)) → (m = 2 ∨ m = -3) :=
by
  sorry

end NUMINAMATH_GPT_find_m_for_parallel_lines_l1348_134881


namespace NUMINAMATH_GPT_total_value_of_item_l1348_134807

variable (V : ℝ) -- Total value of the item

def import_tax (V : ℝ) := 0.07 * (V - 1000) -- Definition of import tax

theorem total_value_of_item
  (htax_paid : import_tax V = 112.70) :
  V = 2610 := 
by
  sorry

end NUMINAMATH_GPT_total_value_of_item_l1348_134807


namespace NUMINAMATH_GPT_length_of_rectangle_l1348_134823

theorem length_of_rectangle (l : ℝ) (s : ℝ) 
  (perimeter_square : 4 * s = 160) 
  (area_relation : s^2 = 5 * (l * 10)) : 
  l = 32 :=
by
  sorry

end NUMINAMATH_GPT_length_of_rectangle_l1348_134823


namespace NUMINAMATH_GPT_min_a_plus_b_l1348_134832

open Real

theorem min_a_plus_b (a b : ℕ) (h_a_pos : a > 1) (h_ab : ∃ a b, (a^2 * b - 1) / (a * b^2) = 1 / 2024) :
  a + b = 228 :=
sorry

end NUMINAMATH_GPT_min_a_plus_b_l1348_134832


namespace NUMINAMATH_GPT_largest_adjacent_to_1_number_of_good_cells_l1348_134886

def table_width := 51
def table_height := 3
def total_cells := 153

-- Conditions
def condition_1_present (n : ℕ) : Prop := n ∈ Finset.range (total_cells + 1)
def condition_2_bottom_left : Prop := (1 = 1)
def condition_3_adjacent (a b : ℕ) : Prop := 
  (a = b + 1) ∨ 
  (a + 1 = b) ∧ 
  (condition_1_present a) ∧ 
  (condition_1_present b)

-- Part (a): Largest number adjacent to cell containing 1 is 152.
theorem largest_adjacent_to_1 : ∃ b, b = 152 ∧ condition_3_adjacent 1 b :=
by sorry

-- Part (b): Number of good cells that can contain the number 153 is 76.
theorem number_of_good_cells : ∃ count, count = 76 ∧ 
  ∀ (i : ℕ) (j: ℕ), (i, j) ∈ (Finset.range table_height).product (Finset.range table_width) →
  condition_1_present 153 ∧
  (i = table_height - 1 ∨ j = 0 ∨ j = table_width - 1 ∨ j ∈ (Finset.range (table_width - 2)).erase 1) →
  (condition_3_adjacent (i*table_width + j) 153) :=
by sorry

end NUMINAMATH_GPT_largest_adjacent_to_1_number_of_good_cells_l1348_134886


namespace NUMINAMATH_GPT_profit_percentage_is_25_percent_l1348_134850

noncomputable def costPrice : ℝ := 47.50
noncomputable def markedPrice : ℝ := 64.54
noncomputable def discountRate : ℝ := 0.08

noncomputable def discountAmount : ℝ := discountRate * markedPrice
noncomputable def sellingPrice : ℝ := markedPrice - discountAmount
noncomputable def profit : ℝ := sellingPrice - costPrice
noncomputable def profitPercentage : ℝ := (profit / costPrice) * 100

theorem profit_percentage_is_25_percent :
  profitPercentage = 25 := by
  sorry

end NUMINAMATH_GPT_profit_percentage_is_25_percent_l1348_134850


namespace NUMINAMATH_GPT_polynomial_sum_l1348_134855

noncomputable def g (a b c d : ℝ) (x : ℂ) : ℂ := x^4 + a * x^3 + b * x^2 + c * x + d

theorem polynomial_sum : ∃ a b c d : ℝ, 
  (g a b c d (-3 * Complex.I) = 0) ∧
  (g a b c d (1 + Complex.I) = 0) ∧
  (g a b c d (3 * Complex.I) = 0) ∧
  (g a b c d (1 - Complex.I) = 0) ∧ 
  (a + b + c + d = 9) := by
  sorry

end NUMINAMATH_GPT_polynomial_sum_l1348_134855


namespace NUMINAMATH_GPT_eggs_collection_l1348_134816

theorem eggs_collection (b : ℕ) (c : ℕ) (t : ℕ) 
  (h₁ : b = 6) 
  (h₂ : c = 3 * b) 
  (h₃ : t = b - 4) : 
  b + c + t = 26 :=
by
  sorry

end NUMINAMATH_GPT_eggs_collection_l1348_134816


namespace NUMINAMATH_GPT_find_three_digit_perfect_square_l1348_134876

noncomputable def digit_product (n : ℕ) : ℕ :=
  (n / 100) * ((n % 100) / 10) * (n % 10)

theorem find_three_digit_perfect_square :
  ∃ (n H : ℕ), (100 ≤ n ∧ n ≤ 999) ∧ (n = H * H) ∧ (digit_product n = H - 1) :=
by {
  sorry
}

end NUMINAMATH_GPT_find_three_digit_perfect_square_l1348_134876


namespace NUMINAMATH_GPT_sum_of_fractions_removal_l1348_134883

theorem sum_of_fractions_removal :
  (1 / 3 + 1 / 6 + 1 / 9 + 1 / 12 + 1 / 15 + 1 / 18 + 1 / 21) 
  - (1 / 12 + 1 / 21) = 3 / 4 := 
 by sorry

end NUMINAMATH_GPT_sum_of_fractions_removal_l1348_134883


namespace NUMINAMATH_GPT_factorial_divisibility_l1348_134897

theorem factorial_divisibility (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h : (a.factorial + (a + b).factorial) ∣ (a.factorial * (a + b).factorial)) : a ≥ 2 * b + 1 :=
sorry

end NUMINAMATH_GPT_factorial_divisibility_l1348_134897


namespace NUMINAMATH_GPT_correct_operation_l1348_134840

theorem correct_operation (a : ℝ) : 
  (a^3 * a^2 ≠ a^6) ∧ 
  ((-4 * a^3)^2 = 16 * a^6) ∧ 
  (a^6 / a^6 ≠ 0) ∧ 
  ((a - 1)^2 ≠ a^2 - 1) := by
  sorry

end NUMINAMATH_GPT_correct_operation_l1348_134840


namespace NUMINAMATH_GPT_probability_of_25_cents_heads_l1348_134821

/-- 
Considering the flipping of five specific coins: a penny, a nickel, a dime,
a quarter, and a half dollar, prove that the probability of getting at least
25 cents worth of heads is 3 / 4.
-/
theorem probability_of_25_cents_heads :
  let total_outcomes := 2^5
  let successful_outcomes_1 := 2^4
  let successful_outcomes_2 := 2^3
  let successful_outcomes := successful_outcomes_1 + successful_outcomes_2
  (successful_outcomes / total_outcomes : ℚ) = 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_25_cents_heads_l1348_134821


namespace NUMINAMATH_GPT_problem_l1348_134818

def is_acute_angle (θ: ℝ) : Prop := θ > 0 ∧ θ < 90
def in_first_quadrant (θ: ℝ) : Prop := θ > 0 ∧ θ < 90
def in_second_quadrant (θ: ℝ) : Prop := θ > 90 ∧ θ < 180

def cond1 (θ: ℝ) : Prop := θ < 90 → is_acute_angle θ
def cond2 (θ: ℝ) : Prop := in_first_quadrant θ → θ ≥ 0
def cond3 (θ: ℝ) : Prop := is_acute_angle θ → in_first_quadrant θ
def cond4 (θ θ': ℝ) : Prop := in_second_quadrant θ → in_first_quadrant θ' → θ > θ'

theorem problem :
  (¬ ∃ θ, cond1 θ) ∧ (¬ ∃ θ, cond2 θ) ∧ (∃ θ, cond3 θ) ∧ (¬ ∃ θ θ', cond4 θ θ') →
  (number_of_correct_propositions = 1) :=
  by
    sorry

end NUMINAMATH_GPT_problem_l1348_134818


namespace NUMINAMATH_GPT_solution_to_system_l1348_134830

def system_of_equations (x y : ℝ) : Prop := (x^2 - 9 * y^2 = 36) ∧ (3 * x + y = 6)

theorem solution_to_system : 
  {p : ℝ × ℝ | system_of_equations p.1 p.2} = { (12 / 5, -6 / 5), (3, -3) } := 
by sorry

end NUMINAMATH_GPT_solution_to_system_l1348_134830


namespace NUMINAMATH_GPT_quadratic_equation_in_x_l1348_134836

theorem quadratic_equation_in_x (m : ℤ) (h1 : abs m = 2) (h2 : m + 2 ≠ 0) : m = 2 :=
sorry

end NUMINAMATH_GPT_quadratic_equation_in_x_l1348_134836


namespace NUMINAMATH_GPT_orthogonal_trajectories_angle_at_origin_l1348_134867

theorem orthogonal_trajectories_angle_at_origin (x y : ℝ) (a : ℝ) :
  ((x + 2 * y) ^ 2 = a * (x + y)) →
  (∃ φ : ℝ, φ = π / 4) :=
by
  sorry

end NUMINAMATH_GPT_orthogonal_trajectories_angle_at_origin_l1348_134867


namespace NUMINAMATH_GPT_greatest_divisor_l1348_134892

theorem greatest_divisor (n : ℕ) (h1 : 1428 % n = 9) (h2 : 2206 % n = 13) : n = 129 :=
sorry

end NUMINAMATH_GPT_greatest_divisor_l1348_134892


namespace NUMINAMATH_GPT_min_sum_of_factors_of_2310_l1348_134887

theorem min_sum_of_factors_of_2310 : ∃ a b c : ℕ, a * b * c = 2310 ∧ a + b + c = 52 :=
by
  sorry

end NUMINAMATH_GPT_min_sum_of_factors_of_2310_l1348_134887


namespace NUMINAMATH_GPT_percentage_water_fresh_fruit_l1348_134862

-- Definitions of the conditions
def weight_dried_fruit : ℝ := 12
def water_content_dried_fruit : ℝ := 0.15
def weight_fresh_fruit : ℝ := 101.99999999999999

-- Derived definitions based on the conditions
def weight_non_water_dried_fruit : ℝ := weight_dried_fruit - (water_content_dried_fruit * weight_dried_fruit)
def weight_non_water_fresh_fruit : ℝ := weight_non_water_dried_fruit
def weight_water_fresh_fruit : ℝ := weight_fresh_fruit - weight_non_water_fresh_fruit

-- Proof statement
theorem percentage_water_fresh_fruit :
  (weight_water_fresh_fruit / weight_fresh_fruit) * 100 = 90 :=
sorry

end NUMINAMATH_GPT_percentage_water_fresh_fruit_l1348_134862


namespace NUMINAMATH_GPT_no_solution_l1348_134884

theorem no_solution (x : ℝ) (h₁ : x ≠ -1/3) (h₂ : x ≠ -4/5) :
  (2 * x - 4) / (3 * x + 1) ≠ (2 * x - 10) / (5 * x + 4) := 
sorry

end NUMINAMATH_GPT_no_solution_l1348_134884


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l1348_134871

theorem quadratic_inequality_solution :
  {x : ℝ | x^2 - 3 * x - 10 > 0} = {x : ℝ | x < -2} ∪ {x : ℝ | x > 5} :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l1348_134871


namespace NUMINAMATH_GPT_equal_divide_remaining_amount_all_girls_l1348_134810

theorem equal_divide_remaining_amount_all_girls 
    (debt : ℕ) (savings_lulu : ℕ) (savings_nora : ℕ) (savings_tamara : ℕ)
    (total_savings : ℕ) (remaining_amount : ℕ)
    (each_girl_gets : ℕ)
    (Lulu_saved : savings_lulu = 6)
    (Nora_saved_multiple_of_Lulu : savings_nora = 5 * savings_lulu)
    (Nora_saved_multiple_of_Tamara : savings_nora = 3 * savings_tamara)
    (total_saved_calculated : total_savings = savings_nora + savings_tamara + savings_lulu)
    (debt_value : debt = 40)
    (remaining_calculated : remaining_amount = total_savings - debt)
    (division_among_girls : each_girl_gets = remaining_amount / 3) :
  each_girl_gets = 2 := 
sorry

end NUMINAMATH_GPT_equal_divide_remaining_amount_all_girls_l1348_134810


namespace NUMINAMATH_GPT_values_of_x_l1348_134882

theorem values_of_x (x : ℤ) :
  (∃ t : ℤ, x = 105 * t + 22) ∨ (∃ t : ℤ, x = 105 * t + 37) ↔ 
  (5 * x^3 - x + 17) % 15 = 0 ∧ (2 * x^2 + x - 3) % 7 = 0 :=
by {
  sorry
}

end NUMINAMATH_GPT_values_of_x_l1348_134882


namespace NUMINAMATH_GPT_manufacturer_l1348_134808

-- Let x be the manufacturer's suggested retail price
variable (x : ℝ)

-- Regular discount range from 10% to 30%
def regular_discount (d : ℝ) : Prop := d >= 0.10 ∧ d <= 0.30

-- Additional discount during sale 
def additional_discount : ℝ := 0.20

-- The final discounted price is $16.80
def final_price (x : ℝ) : Prop := ∃ d, regular_discount d ∧ 0.80 * ((1 - d) * x) = 16.80

theorem manufacturer's_suggested_retail_price :
  final_price x → x = 30 := by
  sorry

end NUMINAMATH_GPT_manufacturer_l1348_134808

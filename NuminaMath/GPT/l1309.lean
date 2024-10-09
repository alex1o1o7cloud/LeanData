import Mathlib

namespace monkey_climbing_time_l1309_130905

theorem monkey_climbing_time (tree_height : ℕ) (hop_distance : ℕ) (slip_distance : ℕ) (final_hop : ℕ) (net_gain : ℕ) :
  tree_height = 19 →
  hop_distance = 3 →
  slip_distance = 2 →
  net_gain = hop_distance - slip_distance →
  final_hop = hop_distance →
  (tree_height - final_hop) % net_gain = 0 →
  18 / net_gain + 1 = (tree_height - final_hop) / net_gain + 1 := 
by {
  sorry
}

end monkey_climbing_time_l1309_130905


namespace percentage_increase_second_year_l1309_130953

theorem percentage_increase_second_year :
  let initial_deposit : ℤ := 1000
  let balance_first_year : ℤ := 1100
  let total_balance_two_years : ℤ := 1320
  let percent_increase_first_year : ℚ := ((balance_first_year - initial_deposit) / initial_deposit) * 100
  let percent_increase_total : ℚ := ((total_balance_two_years - initial_deposit) / initial_deposit) * 100
  let increase_second_year : ℤ := total_balance_two_years - balance_first_year
  let percent_increase_second_year : ℚ := (increase_second_year / balance_first_year) * 100
  percent_increase_first_year = 10 ∧
  percent_increase_total = 32 ∧
  increase_second_year = 220 → 
  percent_increase_second_year = 20 := by
  intros initial_deposit balance_first_year total_balance_two_years percent_increase_first_year
         percent_increase_total increase_second_year percent_increase_second_year
  sorry

end percentage_increase_second_year_l1309_130953


namespace geometric_sequence_a5_l1309_130902

variable {a : Nat → ℝ} {q : ℝ}

-- Conditions
def is_geometric_sequence (a : Nat → ℝ) (q : ℝ) :=
  a 1 = 1 ∧ ∀ n, a (n + 1) = q * a n

def condition_eq (a : Nat → ℝ) :=
  a 5 + a 4 = 3 * (a 3 + a 2)

-- Proof statement
theorem geometric_sequence_a5 (hq : q ≠ -1)
  (hg : is_geometric_sequence a q)
  (hc : condition_eq a) : a 5 = 9 :=
  sorry

end geometric_sequence_a5_l1309_130902


namespace find_number_l1309_130983

theorem find_number (N x : ℝ) (h : x = 9) (h1 : N - (5 / x) = 4 + (4 / x)) : N = 5 :=
by
  sorry

end find_number_l1309_130983


namespace find_a10_l1309_130951

noncomputable def ladder_geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ (n : ℕ), (a (n + 3))^2 = a n * a (n + 6)

theorem find_a10 {a : ℕ → ℝ} (h1 : ladder_geometric_sequence a) 
(h2 : a 1 = 1) 
(h3 : a 4 = 2) : a 10 = 8 :=
sorry

end find_a10_l1309_130951


namespace problem_B_height_l1309_130916

noncomputable def point_B_height (cos : ℝ → ℝ) : ℝ :=
  let θ := 30 * (Real.pi / 180)
  let cos30 := cos θ
  let original_vertical_height := 1 / 2
  let additional_height := cos30 * (1 / 2)
  original_vertical_height + additional_height

theorem problem_B_height : 
  point_B_height Real.cos = (2 + Real.sqrt 3) / 4 := 
by 
  sorry

end problem_B_height_l1309_130916


namespace find_A_l1309_130966

def spadesuit (A B : ℝ) : ℝ := 4 * A + 3 * B + 6

theorem find_A (A : ℝ) (h : spadesuit A 5 = 59) : A = 9.5 :=
by sorry

end find_A_l1309_130966


namespace problem_solution_l1309_130990

noncomputable def greatest_integer_not_exceeding (z : ℝ) : ℤ := Int.floor z

theorem problem_solution (x : ℝ) (y : ℝ) 
  (h1 : y = 4 * greatest_integer_not_exceeding x + 4)
  (h2 : y = 5 * greatest_integer_not_exceeding (x - 3) + 7)
  (h3 : x > 3 ∧ ¬ ∃ (n : ℤ), x = ↑n) :
  64 < x + y ∧ x + y < 65 :=
by
  sorry

end problem_solution_l1309_130990


namespace complement_A_eq_interval_l1309_130935

open Set

-- Define the universal set U as the set of all real numbers
def U : Set ℝ := {x : ℝ | True}

-- Define the set A according to the given conditions
def A : Set ℝ := {x : ℝ | x ≥ 1} ∪ {x : ℝ | x ≤ 0}

-- State the theorem that the complement of A with respect to U is (0, 1)
theorem complement_A_eq_interval : ∀ x : ℝ, x ∈ U \ A ↔ x ∈ Ioo 0 1 := by
  intros x
  -- Proof skipped
  sorry

end complement_A_eq_interval_l1309_130935


namespace sally_initial_cards_l1309_130988

theorem sally_initial_cards (X : ℕ) (h1 : X + 41 + 20 = 88) : X = 27 :=
by
  -- Proof goes here
  sorry

end sally_initial_cards_l1309_130988


namespace rancher_unique_solution_l1309_130931

-- Defining the main problem statement
theorem rancher_unique_solution : ∃! (b h : ℕ), 30 * b + 32 * h = 1200 ∧ b > h := by
  sorry

end rancher_unique_solution_l1309_130931


namespace solve_fraction_inequality_l1309_130939

theorem solve_fraction_inequality :
  { x : ℝ | x / (x + 5) ≥ 0 } = { x : ℝ | x < -5 } ∪ { x : ℝ | x ≥ 0 } := by
  sorry

end solve_fraction_inequality_l1309_130939


namespace percentage_problem_l1309_130938

theorem percentage_problem (x : ℝ)
  (h : 0.70 * 600 = 0.40 * x) : x = 1050 :=
sorry

end percentage_problem_l1309_130938


namespace sufficient_condition_for_ellipse_with_foci_y_axis_l1309_130945

theorem sufficient_condition_for_ellipse_with_foci_y_axis (m n : ℝ) (h : m > n ∧ n > 0) :
  (∃ a b : ℝ, (a^2 = m / n) ∧ (b^2 = 1 / n) ∧ (a > b)) ∧ ¬(∀ u v : ℝ, (u^2 = m / v) → (v^2 = 1 / v) → (u > v) → (v = n ∧ u = m)) :=
by
  sorry

end sufficient_condition_for_ellipse_with_foci_y_axis_l1309_130945


namespace smallest_number_is_27_l1309_130978

theorem smallest_number_is_27 (a b c : ℕ) (h_mean : (a + b + c) / 3 = 30) (h_median : b = 28) (h_largest : c = b + 7) : a = 27 :=
by {
  sorry
}

end smallest_number_is_27_l1309_130978


namespace total_dots_l1309_130954

def ladybugs_monday : ℕ := 8
def ladybugs_tuesday : ℕ := 5
def dots_per_ladybug : ℕ := 6

theorem total_dots :
  (ladybugs_monday + ladybugs_tuesday) * dots_per_ladybug = 78 :=
by
  sorry

end total_dots_l1309_130954


namespace no_triangle_satisfies_condition_l1309_130995

theorem no_triangle_satisfies_condition (x y z : ℝ) (h_tri : x + y > z ∧ x + z > y ∧ y + z > x) :
  x^3 + y^3 + z^3 ≠ (x + y) * (y + z) * (z + x) :=
by
  sorry

end no_triangle_satisfies_condition_l1309_130995


namespace pencil_count_l1309_130962

theorem pencil_count (a : ℕ) :
  200 ≤ a ∧ a ≤ 300 ∧ a % 10 = 7 ∧ a % 12 = 9 → (a = 237 ∨ a = 297) :=
by sorry

end pencil_count_l1309_130962


namespace jason_has_21_toys_l1309_130973

-- Definitions based on the conditions
def rachel_toys : ℕ := 1
def john_toys : ℕ := rachel_toys + 6
def jason_toys : ℕ := 3 * john_toys

-- The theorem to prove
theorem jason_has_21_toys : jason_toys = 21 := by
  -- Proof not needed, hence sorry
  sorry

end jason_has_21_toys_l1309_130973


namespace perimeter_of_first_square_l1309_130908

theorem perimeter_of_first_square
  (s1 s2 s3 : ℝ)
  (P1 P2 P3 : ℝ)
  (A1 A2 A3 : ℝ)
  (hs2 : s2 = 8)
  (hs3 : s3 = 10)
  (hP2 : P2 = 4 * s2)
  (hP3 : P3 = 4 * s3)
  (hP2_val : P2 = 32)
  (hP3_val : P3 = 40)
  (hA2 : A2 = s2^2)
  (hA3 : A3 = s3^2)
  (hA1_A2_A3 : A3 = A1 + A2)
  (hA3_val : A3 = 100)
  (hA2_val : A2 = 64) :
  P1 = 24 := by
  sorry

end perimeter_of_first_square_l1309_130908


namespace perpendicular_line_l1309_130989

theorem perpendicular_line (x y : ℝ) (h : 2 * x + y - 10 = 0) : 
    (∃ k : ℝ, (x = 1 ∧ y = 2) → (k * (-2) = -1)) → 
    (∃ m b : ℝ, b = 3 ∧ m = 1/2) → 
    (x - 2 * y + 3 = 0) := 
sorry

end perpendicular_line_l1309_130989


namespace trees_left_after_typhoon_and_growth_l1309_130961

-- Conditions
def initial_trees : ℕ := 9
def trees_died_in_typhoon : ℕ := 4
def new_trees : ℕ := 5

-- Question (Proof Problem)
theorem trees_left_after_typhoon_and_growth : 
  initial_trees - trees_died_in_typhoon + new_trees = 10 := 
by
  sorry

end trees_left_after_typhoon_and_growth_l1309_130961


namespace cosine_identity_l1309_130948

theorem cosine_identity
  (α β : ℝ)
  (h1 : Real.sin (α - β) = 1 / 3)
  (h2 : Real.cos α * Real.sin β = 1 / 6) :
  Real.cos (2 * α + 2 * β) = 1 / 9 :=
  sorry

end cosine_identity_l1309_130948


namespace estimated_value_of_n_l1309_130965

-- Definitions from the conditions of the problem
def total_balls (n : ℕ) : ℕ := n + 18 + 9
def probability_of_yellow (n : ℕ) : ℚ := 18 / total_balls n

-- The theorem stating what we need to prove
theorem estimated_value_of_n : ∃ n : ℕ, probability_of_yellow n = 0.30 ∧ n = 42 :=
by {
  sorry
}

end estimated_value_of_n_l1309_130965


namespace perpendicular_lines_a_l1309_130917

theorem perpendicular_lines_a {a : ℝ} :
  ((∀ x y : ℝ, (2 * a - 1) * x + a * y + a = 0) → (∀ x y : ℝ, a * x - y + 2 * a = 0) → a = 0 ∨ a = 1) :=
by
  intro h₁ h₂
  sorry

end perpendicular_lines_a_l1309_130917


namespace johns_watermelon_weight_l1309_130926

theorem johns_watermelon_weight (michael_weight clay_weight john_weight : ℕ)
  (h1 : michael_weight = 8)
  (h2 : clay_weight = 3 * michael_weight)
  (h3 : john_weight = clay_weight / 2) :
  john_weight = 12 :=
by
  sorry

end johns_watermelon_weight_l1309_130926


namespace cos_double_angle_l1309_130930

theorem cos_double_angle (θ : ℝ) (h : Real.sin (Real.pi / 2 + θ) = 1 / 3) :
  Real.cos (2 * θ) = -7 / 9 :=
sorry

end cos_double_angle_l1309_130930


namespace base_b_representation_l1309_130991

theorem base_b_representation (b : ℕ) (h₁ : 1 * b + 5 = n) (h₂ : n^2 = 4 * b^2 + 3 * b + 3) : b = 7 :=
by {
  sorry
}

end base_b_representation_l1309_130991


namespace robot_transport_max_robots_l1309_130997

section
variable {A B : ℕ}   -- Define the variables A and B
variable {m : ℕ}     -- Define the variable m

-- Part 1
theorem robot_transport (h1 : A = B + 30) (h2 : 1500 * B = 1000 * (B + 30)) : A = 90 ∧ B = 60 :=
by
  sorry

-- Part 2
theorem max_robots (h3 : 50000 * m + 30000 * (12 - m) ≤ 450000) : m ≤ 4 :=
by
  sorry
end

end robot_transport_max_robots_l1309_130997


namespace right_triangle_area_l1309_130932

theorem right_triangle_area (a b c : ℝ)
    (h1 : a = 16)
    (h2 : ∃ r, r = 6)
    (h3 : c = Real.sqrt (a^2 + b^2)) 
    (h4 : a^2 + b^2 = c^2) :
    1/2 * a * b = 240 := 
by
  -- given:
  -- a = 16
  -- ∃ r, r = 6
  -- c = Real.sqrt (a^2 + b^2)
  -- a^2 + b^2 = c^2
  -- Prove: 1/2 * a * b = 240
  sorry

end right_triangle_area_l1309_130932


namespace compute_value_l1309_130911

-- Definitions based on problem conditions
def x : ℤ := (150 - 100 + 1) * (100 + 150) / 2  -- Sum of integers from 100 to 150

def y : ℤ := (150 - 100) / 2 + 1  -- Number of even integers from 100 to 150

def z : ℤ := 0  -- Product of odd integers from 100 to 150 (including even numbers makes the product 0)

-- The theorem to prove
theorem compute_value : x + y - z = 6401 :=
by
  sorry

end compute_value_l1309_130911


namespace ratio_of_costs_l1309_130901

theorem ratio_of_costs (R N : ℝ) (hR : 3 * R = 0.25 * (3 * R + 3 * N)) : N / R = 3 := 
sorry

end ratio_of_costs_l1309_130901


namespace solve_fraction_eq_l1309_130910

theorem solve_fraction_eq :
  ∀ x : ℝ, (x - 3 ≠ 0) → ((x + 6) / (x - 3) = 4) → x = 6 :=
by
  intros x h_nonzero h_eq
  sorry

end solve_fraction_eq_l1309_130910


namespace find_fourth_root_l1309_130977

theorem find_fourth_root (b c α : ℝ)
  (h₁ : b * (-3)^4 + (b + 3 * c) * (-3)^3 + (c - 4 * b) * (-3)^2 + (19 - b) * (-3) - 2 = 0)
  (h₂ : b * 4^4 + (b + 3 * c) * 4^3 + (c - 4 * b) * 4^2 + (19 - b) * 4 - 2 = 0)
  (h₃ : b * 2^4 + (b + 3 * c) * 2^3 + (c - 4 * b) * 2^2 + (19 - b) * 2 - 2 = 0)
  (h₄ : (-3) + 4 + 2 + α = 2)
  : α = 1 :=
sorry

end find_fourth_root_l1309_130977


namespace largest_among_a_b_c_d_l1309_130950

noncomputable def a : ℝ := Real.log 2022 / Real.log 2021
noncomputable def b : ℝ := Real.log 2023 / Real.log 2022
noncomputable def c : ℝ := 2022 / 2021
noncomputable def d : ℝ := 2023 / 2022

theorem largest_among_a_b_c_d : max a (max b (max c d)) = c := 
sorry

end largest_among_a_b_c_d_l1309_130950


namespace union_of_A_and_B_l1309_130981

def setA : Set ℝ := {x : ℝ | x > 1 / 2}
def setB : Set ℝ := {x : ℝ | -1 < x ∧ x < 3}

theorem union_of_A_and_B : setA ∪ setB = {x : ℝ | -1 < x} :=
by
  sorry

end union_of_A_and_B_l1309_130981


namespace distilled_water_required_l1309_130940

theorem distilled_water_required :
  ∀ (nutrient_concentrate distilled_water : ℝ) (total_solution prep_solution : ℝ), 
    nutrient_concentrate = 0.05 →
    distilled_water = 0.025 →
    total_solution = 0.075 → 
    prep_solution = 0.6 →
    (prep_solution * (distilled_water / total_solution)) = 0.2 :=
by
  intros nutrient_concentrate distilled_water total_solution prep_solution
  sorry

end distilled_water_required_l1309_130940


namespace original_number_of_men_l1309_130928

/-- 
Given:
1. A group of men decided to do a work in 20 days,
2. When 2 men became absent, the remaining men did the work in 22 days,

Prove:
The original number of men in the group was 22.
-/
theorem original_number_of_men (x : ℕ) (h : 20 * x = 22 * (x - 2)) : x = 22 :=
by
  sorry

end original_number_of_men_l1309_130928


namespace smallest_number_of_seats_required_l1309_130976

theorem smallest_number_of_seats_required (total_chairs : ℕ) (condition : ∀ (N : ℕ), ∀ (seating : Finset ℕ),
  seating.card = N → (∀ x ∈ seating, (x + 1) % total_chairs ∈ seating ∨ (x + total_chairs - 1) % total_chairs ∈ seating)) :
  total_chairs = 100 → ∃ N : ℕ, N = 20 :=
by
  intros
  sorry

end smallest_number_of_seats_required_l1309_130976


namespace jack_second_half_time_l1309_130963

def time_jack_first_half := 19
def time_between_jill_and_jack := 7
def time_jill := 32

def time_jack (time_jill time_between_jill_and_jack : ℕ) : ℕ :=
  time_jill - time_between_jill_and_jack

def time_jack_second_half (time_jack time_jack_first_half : ℕ) : ℕ :=
  time_jack - time_jack_first_half

theorem jack_second_half_time :
  time_jack_second_half (time_jack time_jill time_between_jill_and_jack) time_jack_first_half = 6 :=
by
  sorry

end jack_second_half_time_l1309_130963


namespace number_of_winning_scores_l1309_130941

theorem number_of_winning_scores : 
  ∃ (scores: ℕ), scores = 19 := by
  sorry

end number_of_winning_scores_l1309_130941


namespace find_prime_p_l1309_130906

open Int

theorem find_prime_p (p k m n : ℕ) (hp : Nat.Prime p) 
  (hk : 0 < k) (hm : 0 < m)
  (h_eq : (mk^2 + 2 : ℤ) * p - (m^2 + 2 * k^2 : ℤ) = n^2 * (mp + 2 : ℤ)) :
  p = 3 ∨ p = 1 := sorry

end find_prime_p_l1309_130906


namespace train_overtakes_motorbike_in_80_seconds_l1309_130968

-- Definitions of the given conditions
def speed_train_kmph : ℝ := 100
def speed_motorbike_kmph : ℝ := 64
def length_train_m : ℝ := 800.064

-- Definition to convert kmph to m/s
noncomputable def kmph_to_mps (speed_kmph : ℝ) : ℝ :=
  speed_kmph * 1000 / 3600

-- Relative speed in m/s
noncomputable def relative_speed_mps : ℝ :=
  kmph_to_mps (speed_train_kmph - speed_motorbike_kmph)

-- Time taken for the train to overtake the motorbike
noncomputable def time_to_overtake (distance_m : ℝ) (speed_mps : ℝ) : ℝ :=
  distance_m / speed_mps

-- The statement to be proved
theorem train_overtakes_motorbike_in_80_seconds :
  time_to_overtake length_train_m relative_speed_mps = 80.0064 :=
by
  sorry

end train_overtakes_motorbike_in_80_seconds_l1309_130968


namespace sequence_general_term_formula_l1309_130912

-- Definitions based on conditions
def alternating_sign (n : ℕ) : ℤ := (-1) ^ n
def arithmetic_sequence (n : ℕ) : ℤ := 4 * n - 3

-- Definition for the general term formula
def general_term (n : ℕ) : ℤ := alternating_sign n * arithmetic_sequence n

-- Theorem stating that the given sequence's general term formula is a_n = (-1)^n * (4n - 3)
theorem sequence_general_term_formula (n : ℕ) : general_term n = (-1) ^ n * (4 * n - 3) :=
by
  -- Proof logic will go here
  sorry

end sequence_general_term_formula_l1309_130912


namespace train_length_l1309_130946

theorem train_length (time_crossing : ℝ) (speed_train : ℝ) (speed_man : ℝ) (rel_speed : ℝ) (length_train : ℝ) 
    (h1 : time_crossing = 39.99680025597952)
    (h2 : speed_train = 56)
    (h3 : speed_man = 2)
    (h4 : rel_speed = (speed_train - speed_man) * (1000 / 3600))
    (h5 : length_train = rel_speed * time_crossing):
 length_train = 599.9520038396928 :=
by 
  sorry

end train_length_l1309_130946


namespace alpha_eq_pi_over_3_l1309_130955

theorem alpha_eq_pi_over_3 (α β γ : ℝ) (h1 : 0 < α ∧ α < π) (h2 : α + β + γ = π) 
    (h3 : 2 * Real.sin α + Real.tan β + Real.tan γ = 2 * Real.sin α * Real.tan β * Real.tan γ) :
    α = π / 3 :=
by
  sorry

end alpha_eq_pi_over_3_l1309_130955


namespace circle_condition_tangent_lines_right_angle_triangle_l1309_130957

-- Part (1): Range of m for the equation to represent a circle
theorem circle_condition {m : ℝ} : 
  (∀ x y : ℝ, x^2 + y^2 - 2*x + 2*m*y + m^2 - 2*m - 2 = 0 →
  (m > -3 / 2)) :=
sorry

-- Part (2): Equation of tangent line to circle C
theorem tangent_lines {m : ℝ} (h : m = -1) : 
  ∀ x y : ℝ,
  ((x - 1)^2 + (y - 1)^2 = 1 →
  ((x = 2) ∨ (4*x - 3*y + 4 = 0))) :=
sorry

-- Part (3): Value of t for the line intersecting circle at a right angle
theorem right_angle_triangle {t : ℝ} :
  (∀ x y : ℝ, 
  (x + y + t = 0) →
  (t = -3 ∨ t = -1)) :=
sorry

end circle_condition_tangent_lines_right_angle_triangle_l1309_130957


namespace average_rainfall_l1309_130993

theorem average_rainfall (rainfall_Tuesday : ℝ) (rainfall_others : ℝ) (days_in_week : ℝ)
  (h1 : rainfall_Tuesday = 10.5) 
  (h2 : rainfall_Tuesday = rainfall_others)
  (h3 : days_in_week = 7) : 
  (rainfall_Tuesday + rainfall_others) / days_in_week = 3 :=
by
  sorry

end average_rainfall_l1309_130993


namespace divisor_of_a_l1309_130921

theorem divisor_of_a (a b : ℕ) (hx : a % x = 3) (hb : b % 6 = 5) (hab : (a * b) % 48 = 15) : x = 48 :=
by sorry

end divisor_of_a_l1309_130921


namespace fractionOf_Product_Of_Fractions_l1309_130969

noncomputable def fractionOfProductOfFractions := 
  let a := (2 : ℚ) / 9 * (5 : ℚ) / 6 -- Define the product of the fractions
  let b := (3 : ℚ) / 4 -- Define another fraction
  a / b = 20 / 81 -- Statement to be proven

theorem fractionOf_Product_Of_Fractions: fractionOfProductOfFractions :=
by sorry

end fractionOf_Product_Of_Fractions_l1309_130969


namespace percentage_in_quarters_l1309_130970

theorem percentage_in_quarters (dimes quarters nickels : ℕ) (value_dime value_quarter value_nickel : ℕ)
  (h_dimes : dimes = 40)
  (h_quarters : quarters = 30)
  (h_nickels : nickels = 10)
  (h_value_dime : value_dime = 10)
  (h_value_quarter : value_quarter = 25)
  (h_value_nickel : value_nickel = 5) :
  (quarters * value_quarter : ℚ) / ((dimes * value_dime + quarters * value_quarter + nickels * value_nickel) : ℚ) * 100 = 62.5 := 
  sorry

end percentage_in_quarters_l1309_130970


namespace kendra_words_learned_l1309_130934

theorem kendra_words_learned (Goal : ℕ) (WordsNeeded : ℕ) (WordsAlreadyLearned : ℕ) 
  (h1 : Goal = 60) (h2 : WordsNeeded = 24) :
  WordsAlreadyLearned = Goal - WordsNeeded :=
sorry

end kendra_words_learned_l1309_130934


namespace problem_l1309_130952

theorem problem (a b c d : ℝ) (h1 : a = 4 * b) (h2 : b = 3 * c) (h3 : c = 5 * d) :
  (a * c) / (b * d) = 20 := 
by sorry

end problem_l1309_130952


namespace total_painting_cost_l1309_130956

variable (house_area : ℕ) (price_per_sqft : ℕ)

theorem total_painting_cost (h1 : house_area = 484) (h2 : price_per_sqft = 20) :
  house_area * price_per_sqft = 9680 :=
by
  sorry

end total_painting_cost_l1309_130956


namespace particles_probability_computation_l1309_130998

theorem particles_probability_computation : 
  let L0 := 32
  let R0 := 68
  let N := 100
  let a := 1
  let b := 2
  let P_all_on_left := (a:ℚ) / b
  100 * a + b = 102 := by
  sorry

end particles_probability_computation_l1309_130998


namespace infinitesolutions_k_l1309_130984

-- Define the system of equations as given in the problem
def system_of_equations (x y k : ℝ) : Prop :=
  (3 * x - 4 * y = 5) ∧ (9 * x - 12 * y = k)

-- State the theorem that describes the condition for infinitely many solutions
theorem infinitesolutions_k (k : ℝ) :
  (∀ (x y : ℝ), system_of_equations x y k) ↔ k = 15 :=
by
  sorry

end infinitesolutions_k_l1309_130984


namespace intersection_of_complement_l1309_130994

open Set

theorem intersection_of_complement (U : Set ℕ) (A : Set ℕ) (B : Set ℕ) (hU : U = {1, 2, 3, 4, 5, 6})
  (hA : A = {1, 3, 4}) (hB : B = {2, 3, 4, 5}) : A ∩ (U \ B) = {1} :=
by
  rw [hU, hA, hB]
  -- Proof steps go here
  sorry

end intersection_of_complement_l1309_130994


namespace smallest_floor_sum_l1309_130920

theorem smallest_floor_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
    ⌊(a + b) / c⌋ + ⌊(b + c) / a⌋ + ⌊(c + a) / b⌋ ≥ 4 :=
sorry

end smallest_floor_sum_l1309_130920


namespace length_of_train_l1309_130985

theorem length_of_train (speed_km_hr : ℝ) (platform_length_m : ℝ) (time_sec : ℝ) 
  (h1 : speed_km_hr = 72) (h2 : platform_length_m = 250) (h3 : time_sec = 30) : 
  ∃ (train_length : ℝ), train_length = 350 := 
by 
  -- Definitions of the given conditions
  let speed_m_per_s := speed_km_hr * (5 / 18)
  let total_distance := speed_m_per_s * time_sec
  let train_length := total_distance - platform_length_m
  -- Verifying the length of the train
  use train_length
  sorry

end length_of_train_l1309_130985


namespace find_positive_m_l1309_130971

theorem find_positive_m (m : ℝ) : (∃ x : ℝ, 16 * x^2 + m * x + 4 = 0 ∧ ∀ y : ℝ, 16 * y^2 + m * y + 4 = 0 → x = y) ↔ m = 16 :=
by
  sorry

end find_positive_m_l1309_130971


namespace increase_in_value_l1309_130937

-- Define the conditions
def starting_weight : ℝ := 400
def weight_multiplier : ℝ := 1.5
def price_per_pound : ℝ := 3

-- Define new weight and values
def new_weight : ℝ := starting_weight * weight_multiplier
def value_at_starting_weight : ℝ := starting_weight * price_per_pound
def value_at_new_weight : ℝ := new_weight * price_per_pound

-- Theorem to prove
theorem increase_in_value : value_at_new_weight - value_at_starting_weight = 600 := by
  sorry

end increase_in_value_l1309_130937


namespace year_proof_l1309_130907

variable (n : ℕ)

def packaging_waste_exceeds_threshold (y0 : ℝ) (rate : ℝ) (threshold : ℝ) : Prop :=
  let y := y0 * (rate^n)
  y > threshold

noncomputable def year_when_waste_exceeds := 
  let initial_year := 2015
  let y0 := 4 * 10^6 -- in tons
  let rate := (3.0 / 2.0) -- growth rate per year
  let threshold := 40 * 10^6 -- threshold in tons
  ∃ n, packaging_waste_exceeds_threshold n y0 rate threshold ∧ (initial_year + n = 2021)

theorem year_proof : year_when_waste_exceeds :=
  sorry

end year_proof_l1309_130907


namespace sum_of_extreme_values_of_x_l1309_130960

open Real

theorem sum_of_extreme_values_of_x 
  (x y z : ℝ)
  (h1 : x + y + z = 6)
  (h2 : x^2 + y^2 + z^2 = 14) : 
  (min x + max x) = (10 / 3) :=
sorry

end sum_of_extreme_values_of_x_l1309_130960


namespace lowest_point_graph_l1309_130987

theorem lowest_point_graph (x : ℝ) (h : x > -1) : ∃ y, y = (x^2 + 2*x + 2) / (x + 1) ∧ y ≥ 2 ∧ (x = 0 → y = 2) :=
  sorry

end lowest_point_graph_l1309_130987


namespace dice_sum_prime_probability_l1309_130914

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

noncomputable def roll_dice_prob_prime : ℚ :=
  let total_outcomes := 6^7
  let prime_sums := [7, 11, 13, 17, 19, 23, 29, 31, 37, 41]
  let P := 80425 -- Assume pre-computed sum counts based on primes
  (P : ℚ) / total_outcomes

theorem dice_sum_prime_probability :
  roll_dice_prob_prime = 26875 / 93312 :=
by
  sorry

end dice_sum_prime_probability_l1309_130914


namespace chen_steps_recorded_correct_l1309_130982

-- Define the standard for steps per day
def standard : ℕ := 5000

-- Define the steps walked by Xia
def xia_steps : ℕ := 6200

-- Define the recorded steps for Xia
def xia_recorded : ℤ := xia_steps - standard

-- Assert that Xia's recorded steps are +1200
lemma xia_steps_recorded_correct : xia_recorded = 1200 := by
  sorry

-- Define the steps walked by Chen
def chen_steps : ℕ := 4800

-- Define the recorded steps for Chen
def chen_recorded : ℤ := standard - chen_steps

-- State and prove that Chen's recorded steps are -200
theorem chen_steps_recorded_correct : chen_recorded = -200 :=
  sorry

end chen_steps_recorded_correct_l1309_130982


namespace negation_of_proposition_l1309_130949

theorem negation_of_proposition :
  ¬ (∀ x : ℝ, x^2 + 1 > 0) ↔ ∃ x : ℝ, x^2 + 1 ≤ 0 :=
by
  sorry

end negation_of_proposition_l1309_130949


namespace friends_picked_strawberries_with_Lilibeth_l1309_130980

-- Define the conditions
def Lilibeth_baskets : ℕ := 6
def strawberries_per_basket : ℕ := 50
def total_strawberries : ℕ := 1200

-- Define the calculation of strawberries picked by Lilibeth
def Lilibeth_strawberries : ℕ := Lilibeth_baskets * strawberries_per_basket

-- Define the calculation of strawberries picked by friends
def friends_strawberries : ℕ := total_strawberries - Lilibeth_strawberries

-- Define the number of friends who picked strawberries
def friends_picked_with_Lilibeth : ℕ := friends_strawberries / Lilibeth_strawberries

-- The theorem we need to prove
theorem friends_picked_strawberries_with_Lilibeth : friends_picked_with_Lilibeth = 3 :=
by
  -- Proof goes here
  sorry

end friends_picked_strawberries_with_Lilibeth_l1309_130980


namespace initial_cost_of_article_correct_l1309_130967

noncomputable def initial_cost_of_article (final_cost : ℝ) : ℝ :=
  final_cost / (0.75 * 0.85 * 1.10 * 1.05)

theorem initial_cost_of_article_correct (final_cost : ℝ) (h : final_cost = 1226.25) :
  initial_cost_of_article final_cost = 1843.75 :=
by
  rw [h]
  norm_num
  rw [initial_cost_of_article]
  simp [initial_cost_of_article]
  norm_num
  sorry

end initial_cost_of_article_correct_l1309_130967


namespace peter_age_l1309_130900

theorem peter_age (P Q : ℕ) (h1 : Q - P = P / 2) (h2 : P + Q = 35) : Q = 21 :=
  sorry

end peter_age_l1309_130900


namespace last_digit_fib_mod_12_l1309_130959

noncomputable def F : ℕ → ℕ
| 0       => 1
| 1       => 1
| (n + 2) => (F n + F (n + 1)) % 12

theorem last_digit_fib_mod_12 : ∃ N, ∀ n < N, (∃ k, F k % 12 = n) ∧ ∀ m > N, F m % 12 ≠ 11 :=
sorry

end last_digit_fib_mod_12_l1309_130959


namespace average_chemistry_mathematics_l1309_130996

variable {P C M : ℝ}

theorem average_chemistry_mathematics (h : P + C + M = P + 150) : (C + M) / 2 = 75 :=
by
  sorry

end average_chemistry_mathematics_l1309_130996


namespace shaded_region_area_l1309_130922

def isosceles_triangle (AB AC BC : ℝ) (BAC : ℝ) : Prop :=
  AB = AC ∧ BAC = 120 ∧ BC = 32

def circle_with_diameter (diameter : ℝ) (radius : ℝ) : Prop :=
  radius = diameter / 2

theorem shaded_region_area :
  ∀ (AB AC BC : ℝ) (BAC : ℝ) (O : Type) (a b c : ℕ),
    isosceles_triangle AB AC BC BAC →
    circle_with_diameter BC 8 →
    (a = 43) ∧ (b = 128) ∧ (c = 3) →
    a + b + c = 174 :=
by
  sorry

end shaded_region_area_l1309_130922


namespace boxes_same_number_oranges_l1309_130903

theorem boxes_same_number_oranges 
  (total_boxes : ℕ) (min_oranges : ℕ) (max_oranges : ℕ) 
  (boxes : ℕ) (range_oranges : ℕ) :
  total_boxes = 150 →
  min_oranges = 130 →
  max_oranges = 160 →
  range_oranges = max_oranges - min_oranges + 1 →
  boxes = total_boxes / range_oranges →
  31 = range_oranges →
  4 ≤ boxes :=
by sorry

end boxes_same_number_oranges_l1309_130903


namespace problem1_problem2_l1309_130943

-- Problem 1
theorem problem1 : 2023^2 - 2024 * 2022 = 1 :=
sorry

-- Problem 2
variables (a b c : ℝ)
theorem problem2 : 5 * a^2 * b^3 * (-1/10 * a * b^3 * c) / (1/2 * a * b^2)^3 = -4 * c :=
sorry

end problem1_problem2_l1309_130943


namespace smallest_digit_divisible_by_9_l1309_130972

theorem smallest_digit_divisible_by_9 :
  ∃ (d : ℕ), (25 + d) % 9 = 0 ∧ (∀ e : ℕ, (25 + e) % 9 = 0 → e ≥ d) :=
by
  sorry

end smallest_digit_divisible_by_9_l1309_130972


namespace smallest_denominator_fraction_interval_exists_l1309_130979

def interval (a b c d : ℕ) : Prop :=
a = 14 ∧ b = 73 ∧ c = 5 ∧ d = 26

theorem smallest_denominator_fraction_interval_exists :
  ∃ (a b c d : ℕ), 
    a / b < 19 / 99 ∧ b < 99 ∧
    19 / 99 < c / d ∧ d < 99 ∧
    interval a b c d :=
by
  sorry

end smallest_denominator_fraction_interval_exists_l1309_130979


namespace remainder_7459_div_9_l1309_130975

theorem remainder_7459_div_9 : 7459 % 9 = 7 := 
by
  sorry

end remainder_7459_div_9_l1309_130975


namespace part1_part2_part3_l1309_130913

theorem part1 {k b : ℝ} (h₀ : k ≠ 0) (h₁ : b = 1) (h₂ : k + b = 2) : 
  ∀ x : ℝ, y = k * x + b → y = x + 1 :=
by sorry

theorem part2 : ∃ (C : ℝ × ℝ), 
  C.1 = 3 ∧ C.2 = 4 ∧ y = x + 1 :=
by sorry

theorem part3 {n k : ℝ} (h₀ : k ≠ 0) 
  (h₁ : ∀ x : ℝ, x < 3 → (2 / 3) * x + n > x + 1 ∧ (2 / 3) * x + n < 4) 
  (h₂ : ∀ x : ℝ, y = (2 / 3) * x + n → y = 4 ∧ x = 3) :
  n = 2 :=
by sorry

end part1_part2_part3_l1309_130913


namespace scalene_triangle_height_ratio_l1309_130974

theorem scalene_triangle_height_ratio {a b c : ℝ} (h1 : a > b ∧ b > c ∧ a > c)
  (h2 : a + c = 2 * b) : 
  1 / 3 < c / a ∧ c / a < 1 :=
by sorry

end scalene_triangle_height_ratio_l1309_130974


namespace range_of_a_l1309_130992

def set1 : Set ℝ := {x | x ≤ 2}
def set2 (a : ℝ) : Set ℝ := {x | x > a}
variable (a : ℝ)

theorem range_of_a (h : set1 ∪ set2 a = Set.univ) : a ≤ 2 :=
by sorry

end range_of_a_l1309_130992


namespace teta_beta_gamma_l1309_130904

theorem teta_beta_gamma : 
  ∃ T E T' A B E' T'' A' G A'' M M' A''' A'''' : ℕ, 
  TETA = T * 1000 + E * 100 + T' * 10 + A ∧ 
  BETA = B * 1000 + E' * 100 + T'' * 10 + A' ∧ 
  GAMMA = G * 10000 + A'' * 1000 + M * 100 + M' * 10 + A''' ∧
  TETA + BETA = GAMMA ∧ 
  A = A'''' ∧ E = E' ∧ T = T' ∧ T' = T'' ∧ A = A' ∧ A = A'' ∧ A = A''' ∧ M = M' ∧ 
  T ≠ E ∧ T ≠ A ∧ T ≠ B ∧ T ≠ G ∧ T ≠ M ∧
  E ≠ A ∧ E ≠ B ∧ E ≠ G ∧ E ≠ M ∧
  A ≠ B ∧ A ≠ G ∧ A ≠ M ∧
  B ≠ G ∧ B ≠ M ∧
  G ≠ M ∧
  TETA = 4940 ∧ BETA = 5940 ∧ GAMMA = 10880
  :=
sorry

end teta_beta_gamma_l1309_130904


namespace number_of_fences_painted_l1309_130944

-- Definitions based on the problem conditions
def meter_fee : ℝ := 0.2
def fence_length : ℝ := 500
def total_earnings : ℝ := 5000

-- Target statement
theorem number_of_fences_painted : (total_earnings / (fence_length * meter_fee)) = 50 := by
sorry

end number_of_fences_painted_l1309_130944


namespace female_guests_from_jays_family_l1309_130909

theorem female_guests_from_jays_family (total_guests : ℕ) (percent_females : ℝ) (percent_from_jays_family : ℝ)
    (h1 : total_guests = 240) (h2 : percent_females = 0.60) (h3 : percent_from_jays_family = 0.50) :
    total_guests * percent_females * percent_from_jays_family = 72 := by
  sorry

end female_guests_from_jays_family_l1309_130909


namespace table_cost_l1309_130986

variable (T : ℝ) -- Cost of the table
variable (C : ℝ) -- Cost of a chair

-- Conditions
axiom h1 : C = T / 7
axiom h2 : T + 4 * C = 220

theorem table_cost : T = 140 :=
by
  sorry

end table_cost_l1309_130986


namespace part1_solution_part2_solution_l1309_130933

def f (x : ℝ) (a : ℝ) := |x + 1| - |a * x - 1|

-- Statement for part 1
theorem part1_solution (x : ℝ) : (f x 1 > 1) ↔ (x > 1 / 2) := sorry

-- Statement for part 2
theorem part2_solution (a : ℝ) (x : ℝ) (h : 0 < x ∧ x < 1) : 
  (f x a > x) ↔ (0 < a ∧ a ≤ 2) := sorry

end part1_solution_part2_solution_l1309_130933


namespace second_person_days_l1309_130942

theorem second_person_days (x : ℕ) (h1 : ∀ y : ℝ, y = 24 → 1 / y = 1 / 24)
  (h2 : ∀ z : ℝ, z = 15 → 1 / z = 1 / 15) :
  (1 / 24 + 1 / x = 1 / 15) → x = 40 :=
by
  intro h
  have h3 : 15 * (x + 24) = 24 * x := sorry
  have h4 : 15 * x + 360 = 24 * x := sorry
  have h5 : 360 = 24 * x - 15 * x := sorry
  have h6 : 360 = 9 * x := sorry
  have h7 : x = 360 / 9 := sorry
  have h8 : x = 40 := sorry
  exact h8

end second_person_days_l1309_130942


namespace tenth_term_of_arithmetic_sequence_l1309_130923

theorem tenth_term_of_arithmetic_sequence :
  ∃ a : ℕ → ℤ, (∀ n : ℕ, a n + 1 - a n = 2) ∧ a 1 = 1 ∧ a 10 = 19 :=
sorry

end tenth_term_of_arithmetic_sequence_l1309_130923


namespace sum_of_integers_is_24_l1309_130999

theorem sum_of_integers_is_24 (x y : ℕ) (hx : x > y) (h1 : x - y = 4) (h2 : x * y = 132) : x + y = 24 :=
by
  sorry

end sum_of_integers_is_24_l1309_130999


namespace volume_region_inequality_l1309_130925

theorem volume_region_inequality : 
  ∃ (V : ℝ), V = (20 / 3) ∧ 
    ∀ (x y z : ℝ), |x + y + z| + |x + y - z| + |x - y + z| + |-x + y + z| ≤ 4 
    → x^2 + y^2 + z^2 ≤ V :=
sorry

end volume_region_inequality_l1309_130925


namespace B_finishes_work_in_54_days_l1309_130924

-- The problem statement rewritten in Lean 4.
theorem B_finishes_work_in_54_days
  (A_eff : ℕ) -- amount of work A can do in one day
  (B_eff : ℕ) -- amount of work B can do in one day
  (work_days_together : ℕ) -- number of days A and B work together to finish the work
  (h1 : A_eff = 2 * B_eff)
  (h2 : A_eff + B_eff = 3)
  (h3 : work_days_together = 18) :
  work_days_together * (A_eff + B_eff) / B_eff = 54 :=
by
  sorry

end B_finishes_work_in_54_days_l1309_130924


namespace min_correct_all_four_l1309_130915

def total_questions : ℕ := 15
def correct_xiaoxi : ℕ := 11
def correct_xiaofei : ℕ := 12
def correct_xiaomei : ℕ := 13
def correct_xiaoyang : ℕ := 14

theorem min_correct_all_four : 
(∀ total_questions correct_xiaoxi correct_xiaofei correct_xiaomei correct_xiaoyang, 
  total_questions = 15 → correct_xiaoxi = 11 → 
  correct_xiaofei = 12 → correct_xiaomei = 13 → 
  correct_xiaoyang = 14 → 
  ∃ k : ℕ, k = 5 ∧ 
    k = total_questions - ((total_questions - correct_xiaoxi) + 
    (total_questions - correct_xiaofei) + 
    (total_questions - correct_xiaomei) + 
    (total_questions - correct_xiaoyang)) / 4) := 
sorry

end min_correct_all_four_l1309_130915


namespace max_H2O_produced_l1309_130958

theorem max_H2O_produced :
  ∀ (NaOH H2SO4 H2O : ℝ)
  (n_NaOH : NaOH = 1.5)
  (n_H2SO4 : H2SO4 = 1)
  (balanced_reaction : 2 * NaOH + H2SO4 = 2 * H2O + 1 * (NaOH + H2SO4)),
  H2O = 1.5 :=
by
  intros NaOH H2SO4 H2O n_NaOH n_H2SO4 balanced_reaction
  sorry

end max_H2O_produced_l1309_130958


namespace value_of_b_l1309_130936

theorem value_of_b (b : ℚ) (h : b + b / 4 = 3) : b = 12 / 5 := by
  sorry

end value_of_b_l1309_130936


namespace area_of_walkways_is_214_l1309_130929

-- Definitions for conditions
def width_of_flower_beds : ℕ := 2 * 7  -- two beds each 7 feet wide
def walkways_between_beds_width : ℕ := 3 * 2  -- three walkways each 2 feet wide (one on each side and one in between)
def total_width : ℕ := width_of_flower_beds + walkways_between_beds_width  -- Total width

def height_of_flower_beds : ℕ := 3 * 3  -- three rows of beds each 3 feet high
def walkways_between_beds_height : ℕ := 4 * 2  -- four walkways each 2 feet wide (one on each end and one between each row)
def total_height : ℕ := height_of_flower_beds + walkways_between_beds_height  -- Total height

def total_area_of_garden : ℕ := total_width * total_height  -- Total area of the garden including walkways

def area_of_one_flower_bed : ℕ := 7 * 3  -- Area of one flower bed
def total_area_of_flower_beds : ℕ := 6 * area_of_one_flower_bed  -- Total area of six flower beds

def total_area_walkways : ℕ := total_area_of_garden - total_area_of_flower_beds  -- Total area of the walkways

-- Theorem to prove the area of the walkways
theorem area_of_walkways_is_214 : total_area_walkways = 214 := sorry

end area_of_walkways_is_214_l1309_130929


namespace number_of_possible_lists_l1309_130927

theorem number_of_possible_lists : 
  let balls := 15
  let draws := 4
  (balls ^ draws) = 50625 := by
  sorry

end number_of_possible_lists_l1309_130927


namespace find_missing_exponent_l1309_130919

theorem find_missing_exponent (b e₁ e₂ e₃ e₄ : ℝ) (h1 : e₁ = 5.6) (h2 : e₂ = 10.3) (h3 : e₃ = 13.33744) (h4 : e₄ = 2.56256) :
  (b ^ e₁ * b ^ e₂) / b ^ e₄ = b ^ e₃ :=
by
  have h5 : e₁ + e₂ = 15.9 := sorry
  have h6 : 15.9 - e₄ = 13.33744 := sorry
  exact sorry

end find_missing_exponent_l1309_130919


namespace unique_solution_l1309_130964

theorem unique_solution (a b : ℤ) (h : a > b ∧ b > 0) (hab : a * b - a - b = 1) : a = 3 ∧ b = 2 := by
  sorry

end unique_solution_l1309_130964


namespace a2_plus_a3_eq_40_l1309_130918

theorem a2_plus_a3_eq_40 : 
  ∀ (a a1 a2 a3 a4 a5 : ℤ), 
  (2 * x - 1)^5 = a * x^5 + a1 * x^4 + a2 * x^3 + a3 * x^2 + a4 * x + a5 → 
  a2 + a3 = 40 :=
by
  sorry

end a2_plus_a3_eq_40_l1309_130918


namespace length_of_platform_proof_l1309_130947

def convert_speed_to_mps (kmph : Float) : Float := kmph * (5/18)

def distance_covered (speed : Float) (time : Float) : Float := speed * time

def length_of_platform (total_distance : Float) (train_length : Float) : Float := total_distance - train_length

theorem length_of_platform_proof :
  let speed_kmph := 72.0
  let speed_mps := convert_speed_to_mps speed_kmph
  let time_seconds := 36.0
  let train_length := 470.06
  let total_distance := distance_covered speed_mps time_seconds
  length_of_platform total_distance train_length = 249.94 :=
by
  sorry

end length_of_platform_proof_l1309_130947

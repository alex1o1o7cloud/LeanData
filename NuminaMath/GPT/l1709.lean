import Mathlib

namespace relationship_depends_on_b_l1709_170943

theorem relationship_depends_on_b (a b : ℝ) : 
  (a + b > a - b ∨ a + b < a - b ∨ a + b = a - b) ↔ (b > 0 ∨ b < 0 ∨ b = 0) :=
by
  sorry

end relationship_depends_on_b_l1709_170943


namespace fraction_relationships_l1709_170958

variable (p r s u : ℚ)

theorem fraction_relationships (h1 : p / r = 8) (h2 : s / r = 5) (h3 : s / u = 1 / 3) :
  u / p = 15 / 8 :=
sorry

end fraction_relationships_l1709_170958


namespace wire_division_l1709_170901

theorem wire_division (initial_length : ℝ) (num_parts : ℕ) (final_length : ℝ) :
  initial_length = 69.76 ∧ num_parts = 8 ∧
  final_length = (initial_length / num_parts) / num_parts →
  final_length = 1.09 :=
by
  sorry

end wire_division_l1709_170901


namespace ratio_of_points_l1709_170960

theorem ratio_of_points (B J S : ℕ) 
  (h1 : B = J + 20) 
  (h2 : B + J + S = 160) 
  (h3 : B = 45) : 
  B / S = 1 / 2 :=
  sorry

end ratio_of_points_l1709_170960


namespace solve_inequality_l1709_170965

-- We will define the conditions and corresponding solution sets
def solution_set (a x : ℝ) : Prop :=
  (a < -1 ∧ (x > -a ∨ x < 1)) ∨
  (a = -1 ∧ x ≠ 1) ∨
  (a > -1 ∧ (x < -a ∨ x > 1))

theorem solve_inequality (a x : ℝ) :
  (x - 1) * (x + a) > 0 ↔ solution_set a x :=
by
  sorry

end solve_inequality_l1709_170965


namespace problem_one_problem_two_l1709_170926

noncomputable def f (x m : ℝ) : ℝ := x^2 - (m-1) * x + 2 * m

theorem problem_one (m : ℝ) : (∀ x : ℝ, 0 < x → f x m > 0) ↔ (-2 * Real.sqrt 6 + 5 ≤ m ∧ m ≤ 2 * Real.sqrt 6 + 5) :=
by
  sorry

theorem problem_two (m : ℝ) : (∃ x : ℝ, 0 < x ∧ x < 1 ∧ f x m = 0) ↔ (m ∈ Set.Ioo (-2 : ℝ) 0) :=
by
  sorry

end problem_one_problem_two_l1709_170926


namespace problem_part_1_problem_part_2_l1709_170935

def f (x m : ℝ) := 2 * x^2 + (2 - m) * x - m
def g (x m : ℝ) := x^2 - x + 2 * m

theorem problem_part_1 (x : ℝ) : f x 1 > 0 ↔ (x > 1/2 ∨ x < -1) :=
by sorry

theorem problem_part_2 {m x : ℝ} (hm : 0 < m) : f x m ≤ g x m ↔ (-3 ≤ x ∧ x ≤ m) :=
by sorry

end problem_part_1_problem_part_2_l1709_170935


namespace necessary_and_sufficient_condition_l1709_170914

theorem necessary_and_sufficient_condition (a b : ℝ) (h : a * b ≠ 0) : 
  a - b = 1 ↔ a^3 - b^3 - a * b - a^2 - b^2 = 0 := by
  sorry

end necessary_and_sufficient_condition_l1709_170914


namespace compute_XY_l1709_170973

theorem compute_XY (BC AC AB : ℝ) (hBC : BC = 30) (hAC : AC = 50) (hAB : AB = 60) :
  let XA := (BC * AB) / AC 
  let AY := (BC * AC) / AB
  let XY := XA + AY
  XY = 61 :=
by
  sorry

end compute_XY_l1709_170973


namespace sam_distance_when_meeting_l1709_170918

theorem sam_distance_when_meeting :
  ∃ t : ℝ, (35 = 2 * t + 5 * t) ∧ (5 * t = 25) :=
by
  sorry

end sam_distance_when_meeting_l1709_170918


namespace smallest_n_for_sqrt_18n_integer_l1709_170900

theorem smallest_n_for_sqrt_18n_integer :
  ∃ n : ℕ, (n > 0) ∧ (∀ m : ℕ, m > 0 → (∃ k : ℕ, k^2 = 18 * m) → n <= m) ∧ (∃ k : ℕ, k^2 = 18 * n) :=
sorry

end smallest_n_for_sqrt_18n_integer_l1709_170900


namespace perpendicular_graphs_solve_a_l1709_170972

theorem perpendicular_graphs_solve_a (a : ℝ) : 
  (∀ x y : ℝ, 2 * y + x + 3 = 0 → 3 * y + a * x + 2 = 0 → 
  ∀ m1 m2 : ℝ, (y = m1 * x + b1 → m1 = -1 / 2) →
  (y = m2 * x + b2 → m2 = -a / 3) →
  m1 * m2 = -1) → a = -6 :=
by
  sorry

end perpendicular_graphs_solve_a_l1709_170972


namespace line_intersects_circle_l1709_170938

theorem line_intersects_circle (k : ℝ) :
  ∃ x y : ℝ, (x^2 + y^2 - 2*y = 0) ∧ (y - 1 = k * (x - 1)) :=
sorry

end line_intersects_circle_l1709_170938


namespace point_in_third_quadrant_l1709_170917

theorem point_in_third_quadrant (m n : ℝ) (h1 : m > 0) (h2 : n > 0) : (-m < 0) ∧ (-n < 0) :=
by
  sorry

end point_in_third_quadrant_l1709_170917


namespace betsy_to_cindy_ratio_l1709_170930

-- Definitions based on the conditions
def cindy_time : ℕ := 12
def tina_time : ℕ := cindy_time + 6
def betsy_time : ℕ := tina_time / 3

-- Theorem statement to prove
theorem betsy_to_cindy_ratio :
  (betsy_time : ℚ) / cindy_time = 1 / 2 :=
by sorry

end betsy_to_cindy_ratio_l1709_170930


namespace circumference_of_minor_arc_l1709_170949

-- Given:
-- 1. Three points (D, E, F) are on a circle with radius 25
-- 2. The angle ∠EFD = 120°

-- We need to prove that the length of the minor arc DE is 50π / 3
theorem circumference_of_minor_arc 
  (D E F : Point) 
  (r : ℝ) (h : r = 25) 
  (angleEFD : ℝ) 
  (hAngle : angleEFD = 120) 
  (circumference : ℝ) 
  (hCircumference : circumference = 2 * Real.pi * r) :
  arc_length_DE = 50 * Real.pi / 3 :=
by
  sorry

end circumference_of_minor_arc_l1709_170949


namespace root_situation_l1709_170928

theorem root_situation (a b : ℝ) : 
  ∃ (m n : ℝ), 
    (x - a) * (x - (a + b)) = 1 → 
    (m < a ∧ a < n) ∨ (n < a ∧ a < m) :=
sorry

end root_situation_l1709_170928


namespace problem_l1709_170992

-- Condition that defines s and t
def s : ℤ := 4
def t : ℤ := 3

theorem problem (s t : ℤ) (h_s : s = 4) (h_t : t = 3) : s - 2 * t = -2 := by
  sorry

end problem_l1709_170992


namespace asymptote_problem_l1709_170912

-- Definitions for the problem
def r (x : ℝ) : ℝ := -3 * (x + 2) * (x - 1)
def s (x : ℝ) : ℝ := (x + 2) * (x - 4)

-- Assertion to prove
theorem asymptote_problem : r (-1) / s (-1) = 6 / 5 :=
by {
  -- This is where the proof would be carried out
  sorry
}

end asymptote_problem_l1709_170912


namespace number_of_boys_is_50_l1709_170923

-- Definitions for conditions:
def total_students : Nat := 100
def boys (x : Nat) : Nat := x
def girls (x : Nat) : Nat := x

-- Theorem statement:
theorem number_of_boys_is_50 (x : Nat) (g : Nat) (h1 : x + g = total_students) (h2 : g = boys x) : boys x = 50 :=
by
  sorry

end number_of_boys_is_50_l1709_170923


namespace average_fruits_per_basket_is_correct_l1709_170955

noncomputable def average_fruits_per_basket : ℕ :=
  let basket_A := 15
  let basket_B := 30
  let basket_C := 20
  let basket_D := 25
  let basket_E := 35
  let total_fruits := basket_A + basket_B + basket_C + basket_D + basket_E
  let number_of_baskets := 5
  total_fruits / number_of_baskets

theorem average_fruits_per_basket_is_correct : average_fruits_per_basket = 25 := by
  unfold average_fruits_per_basket
  rfl

end average_fruits_per_basket_is_correct_l1709_170955


namespace no_real_values_of_p_for_equal_roots_l1709_170947

theorem no_real_values_of_p_for_equal_roots (p : ℝ) : ¬ ∃ (p : ℝ), (p^2 - 2*p + 5 = 0) :=
by sorry

end no_real_values_of_p_for_equal_roots_l1709_170947


namespace total_books_l1709_170962

theorem total_books (x : ℕ) (h1 : 3 * x + 2 * x + (3 / 2) * x > 3000) : 
  ∃ (T : ℕ), T = 3 * x + 2 * x + (3 / 2) * x ∧ T > 3000 ∧ T = 3003 := 
by 
  -- Our theorem states there exists an integer T such that the total number of books is 3003.
  sorry

end total_books_l1709_170962


namespace sqrt_sixteen_l1709_170954

theorem sqrt_sixteen : ∃ x : ℝ, x^2 = 16 ∧ x = 4 :=
by
  sorry

end sqrt_sixteen_l1709_170954


namespace intersection_is_2_l1709_170977

noncomputable def M : Set ℝ := {x | x^2 - 3 * x + 2 = 0}
noncomputable def N : Set ℝ := {x | x^2 ≥ 2 * x}
noncomputable def intersection : Set ℝ := M ∩ N

theorem intersection_is_2 : intersection = {2} := by
  sorry

end intersection_is_2_l1709_170977


namespace infinite_triangular_pairs_l1709_170974

theorem infinite_triangular_pairs : ∃ (a_i b_i : ℕ → ℕ), (∀ m : ℕ, ∃ n : ℕ, m = n * (n + 1) / 2 ↔ ∃ k : ℕ, a_i k * m + b_i k = k * (k + 1) / 2) ∧ ∀ j : ℕ, ∃ k : ℕ, k > j :=
by {
  sorry
}

end infinite_triangular_pairs_l1709_170974


namespace min_value_expression_l1709_170978

theorem min_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 1) : 
  (x^2 / (1 + x) + y^2 / (1 + y) + z^2 / (1 + z) >= 1 / 4) ∧ (x = 1/3 ∧ y = 1/3 ∧ z = 1/3 → x^2 / (1 + x) + y^2 / (1 + y) + z^2 / (1 + z) = 1 / 4) :=
sorry

end min_value_expression_l1709_170978


namespace ferris_wheel_seats_l1709_170946

def number_of_people_per_seat := 6
def total_number_of_people := 84

def number_of_seats := total_number_of_people / number_of_people_per_seat

theorem ferris_wheel_seats : number_of_seats = 14 := by
  sorry

end ferris_wheel_seats_l1709_170946


namespace sequence_sum_l1709_170924

def arithmetic_seq (a₀ : ℕ) (d : ℕ) : ℕ → ℕ
  | n => a₀ + n * d

def geometric_seq (b₀ : ℕ) (r : ℕ) : ℕ → ℕ
  | n => b₀ * r^(n)

theorem sequence_sum :
  let a : ℕ → ℕ := arithmetic_seq 3 1
  let b : ℕ → ℕ := geometric_seq 1 2
  b (a 0) + b (a 1) + b (a 2) + b (a 3) = 60 :=
  by
    let a : ℕ → ℕ := arithmetic_seq 3 1
    let b : ℕ → ℕ := geometric_seq 1 2
    have h₀ : a 0 = 3 := by rfl
    have h₁ : a 1 = 4 := by rfl
    have h₂ : a 2 = 5 := by rfl
    have h₃ : a 3 = 6 := by rfl
    have hsum : b 3 + b 4 + b 5 + b 6 = 60 := by sorry
    exact hsum

end sequence_sum_l1709_170924


namespace prime_factors_of_69_l1709_170925

theorem prime_factors_of_69 
  (prime : ℕ → Prop)
  (is_prime : ∀ n, prime n ↔ n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7 ∨ n = 11 ∨ 
                        n = 13 ∨ n = 17 ∨ n = 19 ∨ n = 23)
  (x y : ℕ)
  (h1 : 15 < 69)
  (h2 : 69 < 70)
  (h3 : prime y)
  (h4 : 13 < y)
  (h5 : y < 25)
  (h6 : 69 = x * y)
  : prime x ∧ x = 3 := 
sorry

end prime_factors_of_69_l1709_170925


namespace good_set_exists_l1709_170975

def is_good_set (A : List ℕ) : Prop :=
  ∀ i ∈ A, i > 0 ∧ ∀ j ∈ A, i ≠ j → i ^ 2015 % (List.prod (A.erase i)) = 0

theorem good_set_exists (n : ℕ) (h : 3 ≤ n ∧ n ≤ 2015) : 
  ∃ A : List ℕ, A.length = n ∧ ∀ (a : ℕ), a ∈ A → a > 0 ∧ is_good_set A :=
sorry

end good_set_exists_l1709_170975


namespace price_of_5_pound_bag_l1709_170908

-- Definitions based on conditions
def price_10_pound_bag : ℝ := 20.42
def price_25_pound_bag : ℝ := 32.25
def min_pounds : ℝ := 65
def max_pounds : ℝ := 80
def total_min_cost : ℝ := 98.77

-- Define the sought price of the 5-pound bag in the hypothesis
variable {price_5_pound_bag : ℝ}

-- The theorem to prove based on the given conditions
theorem price_of_5_pound_bag
  (h₁ : price_10_pound_bag = 20.42)
  (h₂ : price_25_pound_bag = 32.25)
  (h₃ : min_pounds = 65)
  (h₄ : max_pounds = 80)
  (h₅ : total_min_cost = 98.77) :
  price_5_pound_bag = 2.02 :=
sorry

end price_of_5_pound_bag_l1709_170908


namespace min_small_containers_needed_l1709_170941

def medium_container_capacity : ℕ := 450
def small_container_capacity : ℕ := 28

theorem min_small_containers_needed : ⌈(medium_container_capacity : ℝ) / small_container_capacity⌉ = 17 :=
by
  sorry

end min_small_containers_needed_l1709_170941


namespace measure_of_angle_is_135_l1709_170971

noncomputable def degree_measure_of_angle (x : ℝ) : Prop :=
  (x = 3 * (180 - x)) ∧ (2 * x + (180 - x) = 180) -- Combining all conditions

theorem measure_of_angle_is_135 (x : ℝ) (h : degree_measure_of_angle x) : x = 135 :=
by sorry

end measure_of_angle_is_135_l1709_170971


namespace find_other_number_l1709_170922

def HCF (a b : ℕ) : ℕ := sorry
def LCM (a b : ℕ) : ℕ := sorry

theorem find_other_number (B : ℕ) 
 (h1 : HCF 24 B = 15) 
 (h2 : LCM 24 B = 312) 
 : B = 195 := 
by
  sorry

end find_other_number_l1709_170922


namespace no_distinct_ordered_pairs_l1709_170968

theorem no_distinct_ordered_pairs (x y : ℕ) (h₁ : 0 < x) (h₂ : 0 < y) :
  (x^2 * y^2)^2 - 14 * x^2 * y^2 + 49 ≠ 0 :=
by
  sorry

end no_distinct_ordered_pairs_l1709_170968


namespace warehouse_rental_comparison_purchase_vs_rent_comparison_l1709_170999

-- Define the necessary constants and conditions
def monthly_cost_first : ℕ := 50000
def monthly_cost_second : ℕ := 10000
def moving_cost : ℕ := 70000
def months_in_year : ℕ := 12
def purchase_cost : ℕ := 2000000
def duration_installments : ℕ := 3 * 12 -- 3 years in months
def worst_case_prob : ℕ := 50

-- Question (a)
theorem warehouse_rental_comparison
  (annual_cost_first : ℕ := monthly_cost_first * months_in_year)
  (cost_second_4months : ℕ := monthly_cost_second * 4)
  (cost_switching : ℕ := moving_cost)
  (cost_first_8months : ℕ := monthly_cost_first * 8)
  (worst_case_cost_second : ℕ := cost_second_4months + cost_first_8months + cost_switching) :
  annual_cost_first > worst_case_cost_second :=
by
  sorry

-- Question (b)
theorem purchase_vs_rent_comparison
  (total_rent_cost_4years : ℕ := 4 * annual_cost_first + worst_case_cost_second)
  (total_purchase_cost : ℕ := purchase_cost) :
  total_rent_cost_4years > total_purchase_cost :=
by
  sorry

end warehouse_rental_comparison_purchase_vs_rent_comparison_l1709_170999


namespace percentage_of_ll_watchers_l1709_170982

theorem percentage_of_ll_watchers 
  (T : ℕ) 
  (IS : ℕ) 
  (ME : ℕ) 
  (E2 : ℕ) 
  (A3 : ℕ) 
  (total_residents : T = 600)
  (is_watchers : IS = 210)
  (me_watchers : ME = 300)
  (e2_watchers : E2 = 108)
  (a3_watchers : A3 = 21)
  (at_least_one_show : IS + (by sorry) + ME - E2 + A3 = T) :
  ∃ x : ℕ, (x * 100 / T) = 115 :=
by sorry

end percentage_of_ll_watchers_l1709_170982


namespace solve_problem_l1709_170915

def problem_statement : Prop :=
  ⌊ (2011^3 : ℝ) / (2009 * 2010) - (2009^3 : ℝ) / (2010 * 2011) ⌋ = 8

theorem solve_problem : problem_statement := 
  by sorry

end solve_problem_l1709_170915


namespace max_acute_angles_l1709_170948

theorem max_acute_angles (n : ℕ) : 
  ∃ k : ℕ, k ≤ (2 * n / 3) + 1 :=
sorry

end max_acute_angles_l1709_170948


namespace find_probabilities_l1709_170990

theorem find_probabilities (p_1 p_3 : ℝ)
  (h1 : p_1 + 0.15 + p_3 + 0.25 + 0.35 = 1)
  (h2 : p_3 = 4 * p_1) :
  p_1 = 0.05 ∧ p_3 = 0.20 :=
by
  sorry

end find_probabilities_l1709_170990


namespace intersection_points_l1709_170969

theorem intersection_points (g : ℝ → ℝ) (hg_inv : Function.Injective g) : 
  ∃ n, n = 3 ∧ ∀ x, g (x^3) = g (x^5) ↔ x = 0 ∨ x = 1 ∨ x = -1 :=
by {
  sorry
}

end intersection_points_l1709_170969


namespace point_on_same_side_as_l1709_170945

def f (x y : ℝ) : ℝ := 2 * x - y + 1

theorem point_on_same_side_as (x1 y1 : ℝ) (h : f 1 2 > 0) : f 1 0 > 0 := sorry

end point_on_same_side_as_l1709_170945


namespace part1_part2_part3_l1709_170944

-- Part 1
theorem part1 (a b : ℝ) : 3*(a-b)^2 - 6*(a-b)^2 + 2*(a-b)^2 = -(a-b)^2 :=
sorry

-- Part 2
theorem part2 (x y : ℝ) (h : x^2 - 2*y = 4) : 3*x^2 - 6*y - 21 = -9 :=
sorry

-- Part 3
theorem part3 (a b c d : ℝ) (h1 : a - 5*b = 3) (h2 : 5*b - 3*c = -5) (h3 : 3*c - d = 10) : 
  (a - 3*c) + (5*b - d) - (5*b - 3*c) = 8 :=
sorry

end part1_part2_part3_l1709_170944


namespace quadratic_root_value_l1709_170952

theorem quadratic_root_value (m : ℝ) :
  ∃ m, (∀ x, x^2 - m * x - 3 = 0 → x = -2) → m = -1/2 :=
by
  sorry

end quadratic_root_value_l1709_170952


namespace total_time_spent_l1709_170980

def chess_game_duration_hours : ℕ := 20
def chess_game_duration_minutes : ℕ := 15
def additional_analysis_time : ℕ := 22
def total_expected_time : ℕ := 1237

theorem total_time_spent : 
  (chess_game_duration_hours * 60 + chess_game_duration_minutes + additional_analysis_time) = total_expected_time :=
  by
    sorry

end total_time_spent_l1709_170980


namespace geometric_series_common_ratio_l1709_170986

theorem geometric_series_common_ratio (a : ℕ → ℚ) (q : ℚ) (h1 : a 1 + a 3 = 10) 
(h2 : a 4 + a 6 = 5 / 4) 
(h_geom : ∀ n : ℕ, a (n + 1) = a n * q) : q = 1 / 2 :=
sorry

end geometric_series_common_ratio_l1709_170986


namespace rate_per_kg_mangoes_l1709_170957

theorem rate_per_kg_mangoes 
  (weight_grapes : ℕ) 
  (rate_grapes : ℕ) 
  (weight_mangoes : ℕ) 
  (total_paid : ℕ)
  (total_grapes_cost : ℕ)
  (total_mangoes_cost : ℕ)
  (rate_mangoes : ℕ) 
  (h1 : weight_grapes = 14) 
  (h2 : rate_grapes = 54)
  (h3 : weight_mangoes = 10) 
  (h4 : total_paid = 1376) 
  (h5 : total_grapes_cost = weight_grapes * rate_grapes)
  (h6 : total_mangoes_cost = total_paid - total_grapes_cost) 
  (h7 : rate_mangoes = total_mangoes_cost / weight_mangoes):
  rate_mangoes = 62 :=
by
  sorry

end rate_per_kg_mangoes_l1709_170957


namespace range_of_p_l1709_170963

noncomputable def f (x p : ℝ) : ℝ := x - p/x + p/2

theorem range_of_p (p : ℝ) :
  (∀ x : ℝ, 1 < x → (1 + p / x^2) > 0) → p ≥ -1 :=
by
  intro h
  sorry

end range_of_p_l1709_170963


namespace olivia_cookies_total_l1709_170961

def cookies_total (baggie_cookie_count : ℝ) (chocolate_chip_cookies : ℝ) 
                  (baggies_oatmeal_cookies : ℝ) (total_cookies : ℝ) : Prop :=
  let oatmeal_cookies := baggies_oatmeal_cookies * baggie_cookie_count
  oatmeal_cookies + chocolate_chip_cookies = total_cookies

theorem olivia_cookies_total :
  cookies_total 9.0 13.0 3.111111111 41.0 :=
by
  -- Proof goes here
  sorry

end olivia_cookies_total_l1709_170961


namespace largest_cyclic_decimal_l1709_170995

def digits_on_circle := [1, 3, 9, 5, 7, 9, 1, 3, 9, 5, 7, 1]

def max_cyclic_decimal : ℕ := sorry

theorem largest_cyclic_decimal :
  max_cyclic_decimal = 957913 :=
sorry

end largest_cyclic_decimal_l1709_170995


namespace problem_statement_l1709_170921

noncomputable def a : ℝ := 6 * Real.sqrt 2
noncomputable def b : ℝ := 18 * Real.sqrt 2
noncomputable def c : ℝ := 6 * Real.sqrt 21
noncomputable def d : ℝ := 24 * Real.sqrt 2
noncomputable def e : ℝ := 48 * Real.sqrt 2
noncomputable def N : ℝ := 756 * Real.sqrt 10

axiom condition_a : a^2 + b^2 + c^2 + d^2 + e^2 = 504
axiom positive_numbers : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0

theorem problem_statement : N + a + b + c + d + e = 96 * Real.sqrt 2 + 6 * Real.sqrt 21 + 756 * Real.sqrt 10 :=
by
  -- We'll insert the proof here later
  sorry

end problem_statement_l1709_170921


namespace joe_has_more_shirts_l1709_170983

theorem joe_has_more_shirts (alex_shirts : ℕ) (ben_shirts : ℕ) (ben_joe_diff : ℕ)
  (h_a : alex_shirts = 4)
  (h_b : ben_shirts = 15)
  (h_bj : ben_shirts = joe_shirts + ben_joe_diff)
  (h_bj_diff : ben_joe_diff = 8) :
  joe_shirts - alex_shirts = 3 :=
by {
  sorry
}

end joe_has_more_shirts_l1709_170983


namespace remainder_when_divided_by_15_l1709_170988

def N (k : ℤ) : ℤ := 35 * k + 25

theorem remainder_when_divided_by_15 (k : ℤ) : (N k) % 15 = 10 := 
by 
  -- proof would go here
  sorry

end remainder_when_divided_by_15_l1709_170988


namespace concert_duration_is_805_l1709_170993

def hours_to_minutes (hours : ℕ) : ℕ :=
  hours * 60

def total_duration (hours : ℕ) (extra_minutes : ℕ) : ℕ :=
  hours_to_minutes hours + extra_minutes

theorem concert_duration_is_805 : total_duration 13 25 = 805 :=
by
  -- Proof skipped
  sorry

end concert_duration_is_805_l1709_170993


namespace calculate_expression_l1709_170933

theorem calculate_expression (a b c d : ℤ) (h1 : 3^0 = 1) (h2 : (-1 / 2 : ℚ)^(-2 : ℤ) = 4) : 
  (202 : ℤ) * 3^0 + (-1 / 2 : ℚ)^(-2 : ℤ) = 206 :=
by
  sorry

end calculate_expression_l1709_170933


namespace certain_number_is_five_hundred_l1709_170934

theorem certain_number_is_five_hundred (x : ℝ) (h : 0.60 * x = 0.50 * 600) : x = 500 := 
by sorry

end certain_number_is_five_hundred_l1709_170934


namespace abs_inequality_solution_l1709_170929

theorem abs_inequality_solution (x : ℝ) : |2 * x + 1| - 2 * |x - 1| > 0 ↔ x > 1 / 4 :=
sorry

end abs_inequality_solution_l1709_170929


namespace goldfish_in_each_pond_l1709_170904

variable (x : ℕ)
variable (l1 h1 l2 h2 : ℕ)

-- Conditions
def cond1 : Prop := l1 + h1 = x ∧ l2 + h2 = x
def cond2 : Prop := 4 * l1 = 3 * h1
def cond3 : Prop := 3 * l2 = 5 * h2
def cond4 : Prop := l2 = l1 + 33

theorem goldfish_in_each_pond : cond1 x l1 h1 l2 h2 ∧ cond2 l1 h1 ∧ cond3 l2 h2 ∧ cond4 l1 l2 → 
  x = 168 := 
by 
  sorry

end goldfish_in_each_pond_l1709_170904


namespace tigers_in_zoo_l1709_170903

-- Given definitions
def ratio_lions_tigers := 3 / 4
def number_of_lions := 21
def number_of_tigers := 28

-- Problem statement
theorem tigers_in_zoo : (number_of_lions : ℚ) / 3 * 4 = number_of_tigers := by
  sorry

end tigers_in_zoo_l1709_170903


namespace evaluate_expression_at_x_eq_2_l1709_170997

theorem evaluate_expression_at_x_eq_2 : (3 * 2 + 4)^2 - 10 * 2 = 80 := by
  sorry

end evaluate_expression_at_x_eq_2_l1709_170997


namespace savings_after_20_days_l1709_170910

-- Definitions based on conditions
def daily_earnings : ℕ := 80
def days_worked : ℕ := 20
def total_spent : ℕ := 1360

-- Prove the savings after 20 days
theorem savings_after_20_days : daily_earnings * days_worked - total_spent = 240 :=
by
  sorry

end savings_after_20_days_l1709_170910


namespace max_value_A_l1709_170966

noncomputable def f (x : ℝ) : ℝ :=
  2 * Real.cos x * Real.sin (x + Real.pi / 6)

theorem max_value_A (A : ℝ) (hA : A = Real.pi / 6) : 
  ∀ x : ℝ, f x ≤ f A :=
sorry

end max_value_A_l1709_170966


namespace map_distance_representation_l1709_170939

theorem map_distance_representation
  (cm_to_km_ratio : 15 = 90)
  (km_to_m_ratio : 1000 = 1000) :
  20 * (90 / 15) * 1000 = 120000 := by
  sorry

end map_distance_representation_l1709_170939


namespace D_time_to_complete_job_l1709_170959

-- Let A_rate be the rate at which A works (jobs per hour)
-- Let D_rate be the rate at which D works (jobs per hour)
def A_rate : ℚ := 1 / 3
def combined_rate : ℚ := 1 / 2

-- We need to prove that D_rate, the rate at which D works alone, is 1/6 jobs per hour
def D_rate := 1 / 6

-- And thus, that D can complete the job in 6 hours
theorem D_time_to_complete_job :
  (A_rate + D_rate = combined_rate) → (1 / D_rate) = 6 :=
by
  sorry

end D_time_to_complete_job_l1709_170959


namespace trees_still_left_l1709_170991

theorem trees_still_left 
  (initial_trees : ℕ) 
  (trees_died : ℕ) 
  (trees_cut : ℕ) 
  (initial_trees_eq : initial_trees = 86) 
  (trees_died_eq : trees_died = 15) 
  (trees_cut_eq : trees_cut = 23) 
  : initial_trees - (trees_died + trees_cut) = 48 :=
by
  sorry

end trees_still_left_l1709_170991


namespace salary_of_N_l1709_170911

theorem salary_of_N (total_salary : ℝ) (percent_M_from_N : ℝ) (N_salary : ℝ) : 
  (percent_M_from_N * N_salary + N_salary = total_salary) → (N_salary = 280) :=
by
  sorry

end salary_of_N_l1709_170911


namespace power_mod_eq_remainder_l1709_170940

theorem power_mod_eq_remainder (b m e : ℕ) (hb : b = 17) (hm : m = 23) (he : e = 2090) : 
  b^e % m = 12 := 
  by sorry

end power_mod_eq_remainder_l1709_170940


namespace interval_proof_l1709_170970

noncomputable def valid_interval (x : ℝ) : Prop :=
  ∀ y : ℝ, y > 0 → (5 * (x * y^2 + x^2 * y + 4 * y^2 + 4 * x * y)) / (x + y) > 3 * x^2 * y

theorem interval_proof : ∀ x : ℝ, valid_interval x ↔ (0 ≤ x ∧ x < 4) :=
by
  sorry

end interval_proof_l1709_170970


namespace vertex_on_x_axis_segment_cut_on_x_axis_l1709_170996

-- Define the quadratic function
def quadratic_func (k x : ℝ) : ℝ :=
  (k + 2) * x^2 - 2 * k * x + 3 * k

-- The conditions to prove
theorem vertex_on_x_axis (k : ℝ) :
  (4 * k^2 - 4 * 3 * k * (k + 2) = 0) ↔ (k = 0 ∨ k = -3) :=
sorry

theorem segment_cut_on_x_axis (k : ℝ) :
  ((2 * k / (k + 2))^2 - 12 * k / (k + 2) = 16) ↔ (k = -8/3 ∨ k = -1) :=
sorry

end vertex_on_x_axis_segment_cut_on_x_axis_l1709_170996


namespace part1_part2_a_eq_1_part2_a_gt_1_part2_a_lt_1_l1709_170942

section part1
variable (x : ℝ)

theorem part1 (a : ℝ) (h : a = 2) : (x + a) * (x - 2 * a + 1) < 0 ↔ -2 < x ∧ x < 3 :=
by
  sorry
end part1

section part2
variable (x a : ℝ)

-- Case: a = 1
theorem part2_a_eq_1 (h : a = 1) : (x - 1) * (x - 2 * a + 1) < 0 ↔ False :=
by
  sorry

-- Case: a > 1
theorem part2_a_gt_1 (h : a > 1) : (x - 1) * (x - 2 * a + 1) < 0 ↔ 1 < x ∧ x < 2 * a - 1 :=
by
  sorry

-- Case: a < 1
theorem part2_a_lt_1 (h : a < 1) : (x - 1) * (x - 2 * a + 1) < 0 ↔ 2 * a - 1 < x ∧ x < 1 :=
by
  sorry
end part2

end part1_part2_a_eq_1_part2_a_gt_1_part2_a_lt_1_l1709_170942


namespace total_invested_amount_l1709_170979

theorem total_invested_amount :
  ∃ (A B : ℝ), (A = 3000 ∧ B = 5000 ∧ 
  0.085 * A + 0.064 * B = 575 ∧ A + B = 8000)
  ∨ 
  (A = 5000 ∧ B = 3000 ∧ 
  0.085 * A + 0.064 * B = 575 ∧ A + B = 8000) :=
sorry

end total_invested_amount_l1709_170979


namespace abs_sum_of_factors_of_quadratic_l1709_170984

variable (h b c d : ℤ)

theorem abs_sum_of_factors_of_quadratic :
  (∀ x : ℤ, 6 * x * x + x - 12 = (h * x + b) * (c * x + d)) →
  (|h| + |b| + |c| + |d| = 12) :=
by
  sorry

end abs_sum_of_factors_of_quadratic_l1709_170984


namespace value_of_x_l1709_170907

theorem value_of_x (x y : ℝ) (h1 : x - y = 10) (h2 : x + y = 8) : x = 9 :=
by
  sorry

end value_of_x_l1709_170907


namespace total_profit_calculation_l1709_170913

variables {I_B T_B : ℝ}

-- Conditions as definitions
def investment_A (I_B : ℝ) : ℝ := 3 * I_B
def period_A (T_B : ℝ) : ℝ := 2 * T_B
def profit_B (I_B T_B : ℝ) : ℝ := I_B * T_B
def total_profit (I_B T_B : ℝ) : ℝ := 7 * I_B * T_B

-- To prove
theorem total_profit_calculation
  (h1 : investment_A I_B = 3 * I_B)
  (h2 : period_A T_B = 2 * T_B)
  (h3 : profit_B I_B T_B = 4000)
  : total_profit I_B T_B = 28000 := by
  sorry

end total_profit_calculation_l1709_170913


namespace problem_l1709_170953

theorem problem (x y : ℝ) 
  (h1 : |x + y - 9| = -(2 * x - y + 3) ^ 2) :
  x = 2 ∧ y = 7 :=
sorry

end problem_l1709_170953


namespace complex_magnitude_l1709_170932

open Complex

noncomputable def complexZ : ℂ := sorry -- Definition of complex number z

theorem complex_magnitude (z : ℂ) (h : (1 + 2 * Complex.I) * z = -3 + 4 * Complex.I) : abs z = Real.sqrt 5 :=
sorry

end complex_magnitude_l1709_170932


namespace zip_code_relationship_l1709_170981

theorem zip_code_relationship (A B C D E : ℕ) 
(h1 : A + B + C + D + E = 10) 
(h2 : C = 0) 
(h3 : D = 2 * A) 
(h4 : D + E = 8) : 
A + B = 2 :=
sorry

end zip_code_relationship_l1709_170981


namespace largest_power_of_2_divides_n_l1709_170909

def n : ℤ := 17^4 - 13^4

theorem largest_power_of_2_divides_n : ∃ (k : ℕ), 2^4 = k ∧ 2^k ∣ n ∧ ¬ (2^(k + 1) ∣ n) := by
  sorry

end largest_power_of_2_divides_n_l1709_170909


namespace rose_needs_more_money_l1709_170964

theorem rose_needs_more_money 
    (paintbrush_cost : ℝ)
    (paints_cost : ℝ)
    (easel_cost : ℝ)
    (money_rose_has : ℝ) :
    paintbrush_cost = 2.40 →
    paints_cost = 9.20 →
    easel_cost = 6.50 →
    money_rose_has = 7.10 →
    (paintbrush_cost + paints_cost + easel_cost - money_rose_has) = 11 :=
by
  intros
  sorry

end rose_needs_more_money_l1709_170964


namespace cos_omega_x_3_zeros_interval_l1709_170976

theorem cos_omega_x_3_zeros_interval (ω : ℝ) (hω : ω > 0)
  (h3_zeros : ∃ a b c : ℝ, (0 ≤ a ∧ a ≤ 2 * Real.pi) ∧
    (0 ≤ b ∧ b ≤ 2 * Real.pi ∧ b ≠ a) ∧
    (0 ≤ c ∧ c ≤ 2 * Real.pi ∧ c ≠ a ∧ c ≠ b) ∧
    (∀ x : ℝ, (0 ≤ x ∧ x ≤ 2 * Real.pi) →
      (Real.cos (ω * x) - 1 = 0 ↔ x = a ∨ x = b ∨ x = c))) :
  2 ≤ ω ∧ ω < 3 :=
sorry

end cos_omega_x_3_zeros_interval_l1709_170976


namespace jacob_age_proof_l1709_170931

-- Definitions based on given conditions
def rehana_current_age : ℕ := 25
def rehana_age_in_five_years : ℕ := rehana_current_age + 5
def phoebe_age_in_five_years : ℕ := rehana_age_in_five_years / 3
def phoebe_current_age : ℕ := phoebe_age_in_five_years - 5
def jacob_current_age : ℕ := 3 * phoebe_current_age / 5

-- Statement of the problem
theorem jacob_age_proof :
  jacob_current_age = 3 :=
by 
  -- Skipping the proof for now
  sorry

end jacob_age_proof_l1709_170931


namespace football_game_initial_population_l1709_170998

theorem football_game_initial_population (B G : ℕ) (h1 : G = 240)
  (h2 : (3 / 4 : ℚ) * B + (7 / 8 : ℚ) * G = 480) : B + G = 600 :=
sorry

end football_game_initial_population_l1709_170998


namespace calc_expression_l1709_170902

theorem calc_expression :
  15 * (216 / 3 + 36 / 9 + 16 / 25 + 2^2) = 30240 / 25 :=
by
  sorry

end calc_expression_l1709_170902


namespace original_number_is_28_l1709_170905

theorem original_number_is_28 (N : ℤ) :
  (∃ k : ℤ, N - 11 = 17 * k) → N = 28 :=
by
  intro h
  obtain ⟨k, h₁⟩ := h
  have h₂: N = 17 * k + 11 := by linarith
  have h₃: k = 1 := sorry
  linarith [h₃]
 
end original_number_is_28_l1709_170905


namespace election_problem_l1709_170985

theorem election_problem :
  ∃ (n : ℕ), n = (10 * 9) * Nat.choose 8 3 :=
  by
  use 5040
  sorry

end election_problem_l1709_170985


namespace thief_speed_l1709_170927

theorem thief_speed
  (distance_initial : ℝ := 100 / 1000) -- distance (100 meters converted to kilometers)
  (policeman_speed : ℝ := 10) -- speed of the policeman in km/hr
  (thief_distance : ℝ := 400 / 1000) -- distance thief runs in kilometers (400 meters converted)
  : ∃ V_t : ℝ, V_t = 8 :=
by
  sorry

end thief_speed_l1709_170927


namespace f_analytical_expression_l1709_170967

noncomputable def f (x : ℝ) : ℝ := (2^(x + 1) - 2^(-x)) / 3

theorem f_analytical_expression :
  ∀ x : ℝ, f (-x) + 2 * f x = 2^x :=
by
  sorry

end f_analytical_expression_l1709_170967


namespace geometric_sequence_problem_l1709_170950

noncomputable def geometric_sequence_solution (a_1 a_2 a_3 a_4 a_5 q : ℝ) : Prop :=
  (a_5 - a_1 = 15) ∧
  (a_4 - a_2 = 6) ∧
  (a_3 = 4 ∧ q = 2 ∨ a_3 = -4 ∧ q = 1/2)

theorem geometric_sequence_problem :
  ∃ a_1 a_2 a_3 a_4 a_5 q : ℝ, geometric_sequence_solution a_1 a_2 a_3 a_4 a_5 q :=
by
  sorry

end geometric_sequence_problem_l1709_170950


namespace min_value_xy_l1709_170987

theorem min_value_xy (x y : ℝ) (h : 1 / x + 2 / y = Real.sqrt (x * y)) : x * y ≥ 2 * Real.sqrt 2 :=
by
  sorry

end min_value_xy_l1709_170987


namespace original_price_of_sarees_l1709_170919

theorem original_price_of_sarees (P : ℝ):
  (0.80 * P) * 0.95 = 152 → P = 200 :=
by
  intro h1
  -- You can omit the proof here because the task requires only the statement.
  sorry

end original_price_of_sarees_l1709_170919


namespace condition_iff_odd_function_l1709_170937

theorem condition_iff_odd_function (f : ℝ → ℝ) :
  (∀ x, f x + f (-x) = 0) ↔ (∀ x, f (-x) = -f x) :=
by
  sorry

end condition_iff_odd_function_l1709_170937


namespace inequality_proof_l1709_170956

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
    (b^2 / a + a^2 / b) ≥ (a + b) := 
    sorry

end inequality_proof_l1709_170956


namespace find_huabei_number_l1709_170920

theorem find_huabei_number :
  ∃ (hua bei sai : ℕ), 
    (hua ≠ 4 ∧ hua ≠ 8 ∧ bei ≠ 4 ∧ bei ≠ 8 ∧ sai ≠ 4 ∧ sai ≠ 8) ∧
    (hua ≠ bei ∧ hua ≠ sai ∧ bei ≠ sai) ∧
    (1 ≤ hua ∧ hua ≤ 9 ∧ 1 ≤ bei ∧ bei ≤ 9 ∧ 1 ≤ sai ∧ sai ≤ 9) ∧
    ((100 * hua + 10 * bei + sai) = 7632) :=
sorry

end find_huabei_number_l1709_170920


namespace determine_x_l1709_170916

theorem determine_x (x : ℚ) (h : ∀ (y : ℚ), 10 * x * y - 18 * y + x - 2 = 0) : x = 9 / 5 :=
sorry

end determine_x_l1709_170916


namespace John_lost_socks_l1709_170951

theorem John_lost_socks (initial_socks remaining_socks : ℕ) (H1 : initial_socks = 20) (H2 : remaining_socks = 14) : initial_socks - remaining_socks = 6 :=
by
-- Proof steps can be skipped
sorry

end John_lost_socks_l1709_170951


namespace trigonometric_identity_l1709_170989

theorem trigonometric_identity (α x : ℝ) (h₁ : 5 * Real.cos α = x) (h₂ : x ^ 2 + 16 = 25) (h₃ : α > Real.pi / 2 ∧ α < Real.pi):
  x = -3 ∧ Real.tan α = -4 / 3 :=
by
  sorry

end trigonometric_identity_l1709_170989


namespace ellipse_equation_l1709_170906

theorem ellipse_equation (a b c c1 : ℝ)
  (h_hyperbola_eq : ∀ x y, (y^2 / 4 - x^2 / 12 = 1))
  (h_sum_eccentricities : (c / a) + (c1 / 2) = 13 / 5)
  (h_foci_x_axis : c1 = 4) :
  (a = 5 ∧ b = 4 ∧ c = 3) → 
  ∀ x y, (x^2 / 25 + y^2 / 16 = 1) :=
by
  sorry

end ellipse_equation_l1709_170906


namespace parts_from_blanks_9_parts_from_blanks_14_blanks_needed_for_40_parts_l1709_170994

theorem parts_from_blanks_9 : ∀ (produced_parts : ℕ), produced_parts = 13 :=
by
  sorry

theorem parts_from_blanks_14 : ∀ (produced_parts : ℕ), produced_parts = 20 :=
by
  sorry

theorem blanks_needed_for_40_parts : ∀ (required_blanks : ℕ), required_blanks = 27 :=
by
  sorry

end parts_from_blanks_9_parts_from_blanks_14_blanks_needed_for_40_parts_l1709_170994


namespace train_length_approx_l1709_170936

noncomputable def speed_kmh_to_ms (v: ℝ) : ℝ :=
  v * (1000 / 3600)

noncomputable def length_of_train (v_kmh: ℝ) (time_s: ℝ) : ℝ :=
  (speed_kmh_to_ms v_kmh) * time_s

theorem train_length_approx (v_kmh: ℝ) (time_s: ℝ) (L: ℝ) 
  (h1: v_kmh = 58) 
  (h2: time_s = 9) 
  (h3: L = length_of_train v_kmh time_s) : 
  |L - 145| < 1 :=
  by sorry

end train_length_approx_l1709_170936

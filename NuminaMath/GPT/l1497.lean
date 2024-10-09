import Mathlib

namespace resulting_surface_area_l1497_149785

-- Defining the initial condition for the cube structure
def cube_surface_area (side_length : ℕ) : ℕ :=
  6 * side_length^2

-- Defining the structure and the modifications
def initial_structure : ℕ :=
  64 * (cube_surface_area 2)

def removed_cubes_exposure : ℕ :=
  4 * (cube_surface_area 2)

-- The final lean statement to prove the surface area after removing central cubes
theorem resulting_surface_area : initial_structure + removed_cubes_exposure = 1632 := by
  sorry

end resulting_surface_area_l1497_149785


namespace triangular_array_sum_digits_l1497_149710

theorem triangular_array_sum_digits (N : ℕ) (h : N * (N + 1) / 2 = 2145) : (N / 10 + N % 10) = 11 := 
sorry

end triangular_array_sum_digits_l1497_149710


namespace tangent_line_at_P_no_zero_points_sum_of_zero_points_l1497_149798

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log x - a * x

/-- Given that f(x) = ln(x) - 2x, prove that the tangent line at point P(1, -2) has the equation x + y + 1 = 0. -/
theorem tangent_line_at_P (a : ℝ) (h : a = 2) : ∀ x y : ℝ, x + y + 1 = 0 :=
sorry

/-- Show that for f(x) = ln(x) - ax, the function f(x) has no zero points if a > 1/e. -/
theorem no_zero_points (a : ℝ) (h : a > 1 / Real.exp 1) : ¬∃ x : ℝ, f x a = 0 :=
sorry

/-- For f(x) = ln(x) - ax and x1 ≠ x2 such that f(x1) = f(x2) = 0, prove that x1 + x2 > 2 / a. -/
theorem sum_of_zero_points (a x₁ x₂ : ℝ) (h₁ : x₁ ≠ x₂) (h₂ : f x₁ a = 0) (h₃ : f x₂ a = 0) : x₁ + x₂ > 2 / a :=
sorry

end tangent_line_at_P_no_zero_points_sum_of_zero_points_l1497_149798


namespace arithmetic_sequence_sum_l1497_149713

theorem arithmetic_sequence_sum (a : ℕ → ℕ) (d : ℕ) 
(h : a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 420) 
(h_a : ∀ n, a n = a 1 + (n - 1) * d) : a 2 + a 10 = 120 :=
by
  sorry

end arithmetic_sequence_sum_l1497_149713


namespace frank_has_4_five_dollar_bills_l1497_149776

theorem frank_has_4_five_dollar_bills
    (one_dollar_bills : ℕ := 7)
    (ten_dollar_bills : ℕ := 2)
    (twenty_dollar_bills : ℕ := 1)
    (change : ℕ := 4)
    (peanut_cost_per_pound : ℕ := 3)
    (days_in_week : ℕ := 7)
    (peanuts_per_day : ℕ := 3) :
    let initial_amount := (one_dollar_bills * 1) + (ten_dollar_bills * 10) + (twenty_dollar_bills * 20)
    let total_peanuts_cost := (peanuts_per_day * days_in_week) * peanut_cost_per_pound
    let F := (total_peanuts_cost + change - initial_amount) / 5 
    F = 4 :=
by
  repeat { admit }


end frank_has_4_five_dollar_bills_l1497_149776


namespace evaluate_expression_l1497_149766

def my_star (A B : ℕ) : ℕ := (A + B) / 2
def my_hash (A B : ℕ) : ℕ := A * B + 1

theorem evaluate_expression : my_hash (my_star 4 6) 5 = 26 := 
by
  sorry

end evaluate_expression_l1497_149766


namespace gather_half_of_nuts_l1497_149762

open Nat

theorem gather_half_of_nuts (a b c : ℕ) (h₀ : (a + b + c) % 2 = 0) : ∃ k, k = (a + b + c) / 2 :=
  sorry

end gather_half_of_nuts_l1497_149762


namespace union_of_A_and_B_l1497_149789

variable (a b : ℕ)

def A : Set ℕ := {3, 2^a}
def B : Set ℕ := {a, b}
def intersection_condition : A a ∩ B a b = {2} := by sorry

theorem union_of_A_and_B (h : A a ∩ B a b = {2}) : 
  A a ∪ B a b = {1, 2, 3} := by sorry

end union_of_A_and_B_l1497_149789


namespace prime_cube_plus_five_implies_prime_l1497_149744

theorem prime_cube_plus_five_implies_prime (p : ℕ) 
  (hp : Nat.Prime p) 
  (hq : Nat.Prime (p^3 + 5)) : p^5 - 7 = 25 := 
by
  sorry

end prime_cube_plus_five_implies_prime_l1497_149744


namespace toes_on_bus_is_164_l1497_149786

def num_toes_hoopit : Nat := 3 * 4
def num_toes_neglart : Nat := 2 * 5

def num_hoopits : Nat := 7
def num_neglarts : Nat := 8

def total_toes_on_bus : Nat :=
  num_hoopits * num_toes_hoopit + num_neglarts * num_toes_neglart

theorem toes_on_bus_is_164 : total_toes_on_bus = 164 := by
  sorry

end toes_on_bus_is_164_l1497_149786


namespace radio_selling_price_l1497_149751

theorem radio_selling_price (CP LP Loss SP : ℝ) (h1 : CP = 1500) (h2 : LP = 11)
  (h3 : Loss = (LP / 100) * CP) (h4 : SP = CP - Loss) : SP = 1335 := 
  by
  -- hint: Apply the given conditions.
  sorry

end radio_selling_price_l1497_149751


namespace binom_10_0_eq_1_l1497_149757

theorem binom_10_0_eq_1 :
  (Nat.choose 10 0) = 1 :=
by
  sorry

end binom_10_0_eq_1_l1497_149757


namespace find_diagonal_length_l1497_149779

noncomputable def parallelepiped_diagonal_length 
  (s : ℝ) -- Side length of square face
  (h : ℝ) -- Length of vertical edge
  (θ : ℝ) -- Angle between vertical edge and square face edges
  (hsq : s = 5) -- Length of side of the square face ABCD
  (hedge : h = 5) -- Length of vertical edge AA1
  (θdeg : θ = 60) -- Angle in degrees
  : ℝ :=
5 * Real.sqrt 3

-- The main theorem to be proved
theorem find_diagonal_length
  (s : ℝ)
  (h : ℝ)
  (θ : ℝ)
  (hsq : s = 5)
  (hedge : h = 5)
  (θdeg : θ = 60)
  : parallelepiped_diagonal_length s h θ hsq hedge θdeg = 5 * Real.sqrt 3 := 
sorry

end find_diagonal_length_l1497_149779


namespace range_of_m_l1497_149752

theorem range_of_m (m : ℝ) :
  (¬ ∃ x_0 : ℝ, x_0^2 + 2 * m * x_0 + m + 2 < 0) ↔ (-1 : ℝ) ≤ m ∧ m ≤ 2 :=
sorry

end range_of_m_l1497_149752


namespace number_of_trumpet_players_l1497_149732

def number_of_people_in_orchestra := 21
def number_of_people_known := 1 -- Sebastian
                             + 4 -- Trombone players
                             + 1 -- French horn player
                             + 3 -- Violinists
                             + 1 -- Cellist
                             + 1 -- Contrabassist
                             + 3 -- Clarinet players
                             + 4 -- Flute players
                             + 1 -- Maestro

theorem number_of_trumpet_players : 
  number_of_people_in_orchestra = number_of_people_known + 2 :=
by
  sorry

end number_of_trumpet_players_l1497_149732


namespace graph_of_g_contains_1_0_and_sum_l1497_149797

noncomputable def f : ℝ → ℝ := sorry

def g (x y : ℝ) : Prop := 3 * y = 2 * f (3 * x) + 4

theorem graph_of_g_contains_1_0_and_sum :
  f 3 = -2 → g 1 0 ∧ (1 + 0 = 1) :=
by
  intro h
  sorry

end graph_of_g_contains_1_0_and_sum_l1497_149797


namespace cost_of_shoes_is_150_l1497_149755

def cost_sunglasses : ℕ := 50
def pairs_sunglasses : ℕ := 2
def cost_jeans : ℕ := 100

def cost_basketball_cards : ℕ := 25
def decks_basketball_cards : ℕ := 2

-- Define the total amount spent by Mary and Rose
def total_mary : ℕ := cost_sunglasses * pairs_sunglasses + cost_jeans
def cost_shoes (total_rose : ℕ) (cost_cards : ℕ) : ℕ := total_rose - cost_cards

theorem cost_of_shoes_is_150 (total_spent : ℕ) :
  total_spent = total_mary →
  cost_shoes total_spent (cost_basketball_cards * decks_basketball_cards) = 150 :=
by
  intro h
  sorry

end cost_of_shoes_is_150_l1497_149755


namespace minimum_value_xyz_l1497_149711

theorem minimum_value_xyz (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z = 1) : 
  ∃ m : ℝ, m = 16 ∧ ∀ w, w = (x + y) / (x * y * z) → w ≥ m :=
by
  sorry

end minimum_value_xyz_l1497_149711


namespace series_sum_eq_l1497_149706

theorem series_sum_eq : 
  (∑' n, (4 * n + 3) / ((4 * n - 2) ^ 2 * (4 * n + 2) ^ 2)) = 1 / 128 := by
sorry

end series_sum_eq_l1497_149706


namespace distinct_x_sum_l1497_149756

theorem distinct_x_sum (x y z : ℂ) 
(h1 : x + y * z = 9) 
(h2 : y + x * z = 12) 
(h3 : z + x * y = 12) : 
(x = 1 ∨ x = 3) ∧ (¬(x = 1 ∧ x = 3) → x ≠ 1 ∧ x ≠ 3) ∧ (1 + 3 = 4) :=
by
  sorry

end distinct_x_sum_l1497_149756


namespace parametric_to_standard_form_l1497_149708

theorem parametric_to_standard_form (t : ℝ) (x y : ℝ)
    (param_eq1 : x = 1 + t)
    (param_eq2 : y = -1 + t) :
    x - y - 2 = 0 :=
sorry

end parametric_to_standard_form_l1497_149708


namespace find_tony_age_l1497_149718

variable (y : ℕ)
variable (d : ℕ)

def Tony_day_hours : ℕ := 3
def Tony_hourly_rate (age : ℕ) : ℚ := 0.75 * age
def Tony_days_worked : ℕ := 60
def Tony_total_earnings : ℚ := 945

noncomputable def earnings_before_birthday (age : ℕ) (days : ℕ) : ℚ :=
  Tony_hourly_rate age * Tony_day_hours * days

noncomputable def earnings_after_birthday (age : ℕ) (days : ℕ) : ℚ :=
  Tony_hourly_rate (age + 1) * Tony_day_hours * days

noncomputable def total_earnings (age : ℕ) (days_before : ℕ) : ℚ :=
  (earnings_before_birthday age days_before) +
  (earnings_after_birthday age (Tony_days_worked - days_before))

theorem find_tony_age: ∃ y d : ℕ, total_earnings y d = Tony_total_earnings ∧ y = 6 := by
  sorry

end find_tony_age_l1497_149718


namespace maximum_possible_value_of_k_l1497_149763

theorem maximum_possible_value_of_k :
  ∀ (k : ℕ), 
    (∃ (x : ℕ → ℝ), 
      (∀ i j : ℕ, 1 ≤ i ∧ i ≤ k ∧ 1 ≤ j ∧ j ≤ k → x i > 1 ∧ x i ≠ x j ∧ x i ^ ⌊x j⌋ = x j ^ ⌊x i⌋)) 
      → k ≤ 4 :=
by
  sorry

end maximum_possible_value_of_k_l1497_149763


namespace pages_read_on_Monday_l1497_149746

variable (P : Nat) (W : Nat)
def TotalPages : Nat := P + 12 + W

theorem pages_read_on_Monday :
  (TotalPages P W = 51) → (P = 39) :=
by
  sorry

end pages_read_on_Monday_l1497_149746


namespace largest_number_of_stores_visited_l1497_149799

theorem largest_number_of_stores_visited
  (stores : ℕ) (total_visits : ℕ) (total_peopled_shopping : ℕ)
  (people_visiting_2_stores : ℕ) (people_visiting_3_stores : ℕ)
  (people_visiting_4_stores : ℕ) (people_visiting_1_store : ℕ)
  (everyone_visited_at_least_one_store : ∀ p : ℕ, 0 < people_visiting_1_store + people_visiting_2_stores + people_visiting_3_stores + people_visiting_4_stores)
  (h1 : stores = 15) (h2 : total_visits = 60) (h3 : total_peopled_shopping = 30)
  (h4 : people_visiting_2_stores = 12) (h5 : people_visiting_3_stores = 6)
  (h6 : people_visiting_4_stores = 4) (h7 : people_visiting_1_store = total_peopled_shopping - (people_visiting_2_stores + people_visiting_3_stores + people_visiting_4_stores + 2)) :
  ∃ p : ℕ, ∀ person, person ≤ p ∧ p = 4 := sorry

end largest_number_of_stores_visited_l1497_149799


namespace identical_sets_l1497_149728

def A : Set ℝ := {x : ℝ | ∃ y : ℝ, y = x^2 + 1}
def B : Set ℝ := {y : ℝ | ∃ x : ℝ, y = x^2 + 1}
def C : Set (ℝ × ℝ) := {(x, y) : ℝ × ℝ | y = x^2 + 1}
def D : Set ℝ := {y : ℝ | 1 ≤ y}

theorem identical_sets : B = D :=
by
  sorry

end identical_sets_l1497_149728


namespace percentage_calculation_l1497_149768

-- Define total and part amounts
def total_amount : ℕ := 800
def part_amount : ℕ := 200

-- Define the percentage calculation
def percentage (part : ℕ) (whole : ℕ) : ℕ := (part * 100) / whole

-- Theorem to show the percentage is 25%
theorem percentage_calculation :
  percentage part_amount total_amount = 25 :=
sorry

end percentage_calculation_l1497_149768


namespace avg_weight_ab_l1497_149702

theorem avg_weight_ab (A B C : ℝ) 
  (h1 : (A + B + C) / 3 = 30) 
  (h2 : (B + C) / 2 = 28) 
  (h3 : B = 16) : 
  (A + B) / 2 = 25 := 
by 
  sorry

end avg_weight_ab_l1497_149702


namespace balls_into_boxes_l1497_149769

theorem balls_into_boxes : (4 ^ 5 = 1024) :=
by
  -- The proof is omitted; the statement is required
  sorry

end balls_into_boxes_l1497_149769


namespace larger_integer_is_neg4_l1497_149777

-- Definitions of the integers used in the problem
variables (x y : ℤ)

-- Conditions given in the problem
def condition1 : x + y = -9 := sorry
def condition2 : x - y = 1 := sorry

-- The theorem to prove
theorem larger_integer_is_neg4 (h1 : x + y = -9) (h2 : x - y = 1) : x = -4 := 
sorry

end larger_integer_is_neg4_l1497_149777


namespace correct_operations_result_greater_than_1000_l1497_149794

theorem correct_operations_result_greater_than_1000
    (finalResultIncorrectOps : ℕ)
    (originalNumber : ℕ)
    (finalResultCorrectOps : ℕ)
    (H1 : finalResultIncorrectOps = 40)
    (H2 : originalNumber = (finalResultIncorrectOps + 12) * 8)
    (H3 : finalResultCorrectOps = (originalNumber * 8) + (2 * originalNumber) + 12) :
  finalResultCorrectOps > 1000 := 
sorry

end correct_operations_result_greater_than_1000_l1497_149794


namespace number_of_sides_of_polygon_l1497_149701

theorem number_of_sides_of_polygon (exterior_angle : ℝ) (sum_exterior_angles : ℝ) (h1 : exterior_angle = 30) (h2 : sum_exterior_angles = 360) :
  sum_exterior_angles / exterior_angle = 12 := 
by
  sorry

end number_of_sides_of_polygon_l1497_149701


namespace steve_halfway_time_longer_than_danny_l1497_149774

theorem steve_halfway_time_longer_than_danny 
  (T_D : ℝ) (T_S : ℝ)
  (h1 : T_D = 33) 
  (h2 : T_S = 2 * T_D):
  (T_S / 2) - (T_D / 2) = 16.5 :=
by sorry

end steve_halfway_time_longer_than_danny_l1497_149774


namespace average_salary_rest_l1497_149735

noncomputable def average_salary_of_the_rest : ℕ := 6000

theorem average_salary_rest 
  (N : ℕ) 
  (A : ℕ)
  (T : ℕ)
  (A_T : ℕ)
  (Nr : ℕ)
  (Ar : ℕ)
  (H1 : N = 42)
  (H2 : A = 8000)
  (H3 : T = 7)
  (H4 : A_T = 18000)
  (H5 : Nr = N - T)
  (H6 : Nr = 42 - 7)
  (H7 : Ar = 6000)
  (H8 : 42 * 8000 = (Nr * Ar) + (T * 18000))
  : Ar = average_salary_of_the_rest :=
by
  sorry

end average_salary_rest_l1497_149735


namespace complete_square_l1497_149750

theorem complete_square (a b c : ℕ) (h : 49 * x ^ 2 + 70 * x - 121 = 0) :
  a = 7 ∧ b = 5 ∧ c = 146 ∧ a + b + c = 158 :=
by sorry

end complete_square_l1497_149750


namespace largest_five_digit_negative_int_congruent_mod_23_l1497_149793

theorem largest_five_digit_negative_int_congruent_mod_23 :
  ∃ n : ℤ, 23 * n + 1 < -9999 ∧ 23 * n + 1 = -9994 := 
sorry

end largest_five_digit_negative_int_congruent_mod_23_l1497_149793


namespace num_candidates_above_630_l1497_149761

noncomputable def normal_distribution_candidates : Prop :=
  let μ := 530
  let σ := 50
  let total_candidates := 1000
  let probability_above_630 := (1 - 0.954) / 2  -- Probability of scoring above 630
  let expected_candidates_above_630 := total_candidates * probability_above_630
  expected_candidates_above_630 = 23

theorem num_candidates_above_630 : normal_distribution_candidates := by
  sorry

end num_candidates_above_630_l1497_149761


namespace jacob_calories_l1497_149773

theorem jacob_calories (goal : ℕ) (breakfast : ℕ) (lunch : ℕ) (dinner : ℕ) 
  (h_goal : goal = 1800) 
  (h_breakfast : breakfast = 400) 
  (h_lunch : lunch = 900) 
  (h_dinner : dinner = 1100) : 
  (breakfast + lunch + dinner) - goal = 600 :=
by 
  sorry

end jacob_calories_l1497_149773


namespace sum_of_solutions_l1497_149753

theorem sum_of_solutions (x : ℝ) (h : ∀ x, (x ≠ 1) ∧ (x ≠ -1) → ( -15 * x / (x^2 - 1) = 3 * x / (x + 1) - 9 / (x - 1) )) : 
  (∀ x, (x ≠ 1) ∧ (x ≠ -1) → -15 * x / (x^2 - 1) = 3 * x / (x+1) - 9 / (x-1)) → (x = ( -1 + Real.sqrt 13 ) / 2 ∨ x = ( -1 - Real.sqrt 13 ) / 2) → (x + ( -x ) = -1) :=
by
  sorry

end sum_of_solutions_l1497_149753


namespace car_distance_l1497_149717

-- Define the conditions
def speed := 162  -- speed of the car in km/h
def time := 5     -- time taken in hours

-- Define the distance calculation
def distance (s : ℕ) (t : ℕ) : ℕ := s * t

-- State the theorem
theorem car_distance : distance speed time = 810 := by
  -- Proof goes here
  sorry

end car_distance_l1497_149717


namespace total_shirts_l1497_149784

def initial_shirts : ℕ := 9
def new_shirts : ℕ := 8

theorem total_shirts : initial_shirts + new_shirts = 17 := by
  sorry

end total_shirts_l1497_149784


namespace symmetric_point_min_value_l1497_149704

theorem symmetric_point_min_value (a b : ℝ) 
  (h1 : a > 0 ∧ b > 0) 
  (h2 : ∃ (x₀ y₀ : ℝ), x₀ + y₀ - 2 = 0 ∧ 2 * x₀ + y₀ + 3 = 0 ∧ 
        a + b = x₀ + y₀ ∧ ∃ k, k = (y₀ - b) / (x₀ - a) ∧ y₀ = k * x₀ + 2 - k * (a + k * b))
   : ∃ α β, a = β / α ∧  b = 2 * β / α ∧ (1 / a + 8 / b) = 25 / 9 :=
sorry

end symmetric_point_min_value_l1497_149704


namespace sqrt_sum_of_four_terms_of_4_pow_4_l1497_149724

-- Proof Statement
theorem sqrt_sum_of_four_terms_of_4_pow_4 : 
  Real.sqrt (4 ^ 4 + 4 ^ 4 + 4 ^ 4 + 4 ^ 4) = 32 := 
by 
  sorry

end sqrt_sum_of_four_terms_of_4_pow_4_l1497_149724


namespace induction_two_pow_n_plus_two_times_three_pow_n_plus_five_n_minus_four_divisible_by_25_l1497_149772

open Nat

theorem induction_two_pow_n_plus_two_times_three_pow_n_plus_five_n_minus_four_divisible_by_25 :
  ∀ n : ℕ, n > 0 → 25 ∣ (2^(n+2) * 3^n + 5*n - 4) :=
by
  intro n hn
  sorry

end induction_two_pow_n_plus_two_times_three_pow_n_plus_five_n_minus_four_divisible_by_25_l1497_149772


namespace cos_product_equals_one_eighth_l1497_149730

noncomputable def cos_pi_over_9 := Real.cos (Real.pi / 9)
noncomputable def cos_2pi_over_9 := Real.cos (2 * Real.pi / 9)
noncomputable def cos_4pi_over_9 := Real.cos (4 * Real.pi / 9)

theorem cos_product_equals_one_eighth :
  cos_pi_over_9 * cos_2pi_over_9 * cos_4pi_over_9 = 1 / 8 := 
sorry

end cos_product_equals_one_eighth_l1497_149730


namespace range_of_c_monotonicity_of_g_l1497_149788

noncomputable def f (x: ℝ) : ℝ := 2 * Real.log x + 1

theorem range_of_c (c: ℝ) : (∀ x > 0, f x ≤ 2 * x + c) → c ≥ -1 := by
  sorry

noncomputable def g (x a: ℝ) : ℝ := (f x - f a) / (x - a)

theorem monotonicity_of_g (a: ℝ) (ha: a > 0) : 
  (∀ x > 0, x ≠ a → ((x < a → g x a < g a a) ∧ (x > a → g x a < g a a))) := by
  sorry

end range_of_c_monotonicity_of_g_l1497_149788


namespace area_ratio_of_squares_l1497_149745

theorem area_ratio_of_squares (a b : ℝ) (h : 4 * a = 16 * b) : a ^ 2 = 16 * b ^ 2 := by
  sorry

end area_ratio_of_squares_l1497_149745


namespace range_of_a_l1497_149778

variable (a x : ℝ)

def P : Prop := a < x ∧ x < a + 1
def q : Prop := x^2 - 7 * x + 10 ≤ 0

theorem range_of_a (h₁ : P a x → q x) (h₂ : ∃ x, q x ∧ ¬P a x) : 2 ≤ a ∧ a ≤ 4 := 
sorry

end range_of_a_l1497_149778


namespace find_a_l1497_149765

theorem find_a (a : ℤ) (A B : Set ℤ) (hA : A = {1, 3, a}) (hB : B = {1, a^2 - a + 1}) (h_subset : B ⊆ A) :
  a = -1 ∨ a = 2 := 
by
  sorry

end find_a_l1497_149765


namespace fifth_term_of_sequence_is_31_l1497_149760

namespace SequenceProof

def sequence (a : ℕ → ℕ) :=
  a 1 = 1 ∧ ∀ n ≥ 2, a n = 2 * a (n - 1) + 1

theorem fifth_term_of_sequence_is_31 :
  ∃ a : ℕ → ℕ, sequence a ∧ a 5 = 31 :=
by
  sorry

end SequenceProof

end fifth_term_of_sequence_is_31_l1497_149760


namespace race_distance_l1497_149764

theorem race_distance (d x y z : ℝ) 
  (h1 : d / x = (d - 25) / y)
  (h2 : d / y = (d - 15) / z)
  (h3 : d / x = (d - 35) / z) : 
  d = 75 := 
sorry

end race_distance_l1497_149764


namespace largest_variable_l1497_149783

theorem largest_variable {x y z w : ℤ} 
  (h1 : x + 3 = y - 4)
  (h2 : x + 3 = z + 2)
  (h3 : x + 3 = w - 1) :
  y > x ∧ y > z ∧ y > w :=
by sorry

end largest_variable_l1497_149783


namespace algebraic_expression_value_l1497_149733

noncomputable def a := Real.sqrt 2 + 1
noncomputable def b := Real.sqrt 2 - 1

theorem algebraic_expression_value : (a^2 - 2 * a * b + b^2) / (a^2 - b^2) = Real.sqrt 2 / 2 := by
  sorry

end algebraic_expression_value_l1497_149733


namespace pear_counts_after_events_l1497_149759

theorem pear_counts_after_events (Alyssa_picked Nancy_picked Carlos_picked : ℕ) (give_away : ℕ)
  (eat_fraction : ℚ) (share_fraction : ℚ) :
  Alyssa_picked = 42 →
  Nancy_picked = 17 →
  Carlos_picked = 25 →
  give_away = 5 →
  eat_fraction = 0.20 →
  share_fraction = 0.5 →
  ∃ (Alyssa_picked_final Nancy_picked_final Carlos_picked_final : ℕ),
    Alyssa_picked_final = 30 ∧
    Nancy_picked_final = 14 ∧
    Carlos_picked_final = 18 :=
by
  sorry

end pear_counts_after_events_l1497_149759


namespace mul_neg_x_squared_cubed_l1497_149775

theorem mul_neg_x_squared_cubed (x : ℝ) : (-x^2) * x^3 = -x^5 :=
sorry

end mul_neg_x_squared_cubed_l1497_149775


namespace man_speed_l1497_149723

theorem man_speed {m l: ℝ} (TrainLength : ℝ := 385) (TrainSpeedKmH : ℝ := 60)
  (PassTimeSeconds : ℝ := 21) (RelativeSpeed : ℝ) (ManSpeedKmH : ℝ) 
  (ConversionFactor : ℝ := 3.6) (expected_speed : ℝ := 5.99) : 
  RelativeSpeed = TrainSpeedKmH/ConversionFactor + m/ConversionFactor ∧ 
  TrainLength = RelativeSpeed * PassTimeSeconds →
  abs (m*ConversionFactor - expected_speed) < 0.01 :=
by
  sorry

end man_speed_l1497_149723


namespace problem_solution_l1497_149714

theorem problem_solution (a b c d : ℕ) (h : 342 * (a * b * c * d + a * b + a * d + c * d + 1) = 379 * (b * c * d + b + d)) :
  (a * 10^3 + b * 10^2 + c * 10 + d) = 1949 :=
by
  sorry

end problem_solution_l1497_149714


namespace chord_length_eq_l1497_149705

def line_eq (x y : ℝ) : Prop := 3 * x + 4 * y - 5 = 0
def circle_eq (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 4

theorem chord_length_eq : 
  ∀ (x y : ℝ), 
  (line_eq x y) ∧ (circle_eq x y) → 
  ∃ l, l = 2 * Real.sqrt 3 :=
sorry

end chord_length_eq_l1497_149705


namespace log_expression_equality_l1497_149747

theorem log_expression_equality : 
  (Real.log 3 / Real.log 2) * (Real.log 4 / Real.log 3) + 
  (Real.log 8 / Real.log 4) + 
  2 = 11 / 2 :=
by 
  sorry

end log_expression_equality_l1497_149747


namespace range_of_a_l1497_149754

noncomputable def function_with_extreme_at_zero_only (a b : ℝ) : Prop :=
∀ x : ℝ, x ≠ 0 → 4 * x^2 + 3 * a * x + 4 > 0

theorem range_of_a (a b : ℝ) (h : function_with_extreme_at_zero_only a b) : 
  -8 / 3 ≤ a ∧ a ≤ 8 / 3 :=
sorry

end range_of_a_l1497_149754


namespace total_selling_price_l1497_149727

def selling_price_A (purchase_price_A : ℝ) : ℝ :=
  purchase_price_A - (0.15 * purchase_price_A)

def selling_price_B (purchase_price_B : ℝ) : ℝ :=
  purchase_price_B + (0.10 * purchase_price_B)

def selling_price_C (purchase_price_C : ℝ) : ℝ :=
  purchase_price_C - (0.05 * purchase_price_C)

theorem total_selling_price 
  (purchase_price_A : ℝ)
  (purchase_price_B : ℝ)
  (purchase_price_C : ℝ)
  (loss_A : ℝ := 0.15)
  (gain_B : ℝ := 0.10)
  (loss_C : ℝ := 0.05)
  (total_price := selling_price_A purchase_price_A + selling_price_B purchase_price_B + selling_price_C purchase_price_C) :
  purchase_price_A = 1400 → purchase_price_B = 2500 → purchase_price_C = 3200 →
  total_price = 6980 :=
by sorry

end total_selling_price_l1497_149727


namespace batteries_C_equivalent_l1497_149795

variables (x y z W : ℝ)

-- Conditions
def cond1 := 4 * x + 18 * y + 16 * z = W * z
def cond2 := 2 * x + 15 * y + 24 * z = W * z
def cond3 := 6 * x + 12 * y + 20 * z = W * z

-- Equivalent statement to prove
theorem batteries_C_equivalent (h1 : cond1 x y z W) (h2 : cond2 x y z W) (h3 : cond3 x y z W) : W = 48 :=
sorry

end batteries_C_equivalent_l1497_149795


namespace builder_needs_boards_l1497_149715

theorem builder_needs_boards (packages : ℕ) (boards_per_package : ℕ) (total_boards : ℕ)
  (h1 : packages = 52)
  (h2 : boards_per_package = 3)
  (h3 : total_boards = packages * boards_per_package) : 
  total_boards = 156 :=
by
  rw [h1, h2] at h3
  exact h3

end builder_needs_boards_l1497_149715


namespace tan_theta_solution_l1497_149716

theorem tan_theta_solution (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < 15) 
  (h_tan_eq : Real.tan θ + Real.tan (2 * θ) + Real.tan (4 * θ) = 0) :
  Real.tan θ = 1 / Real.sqrt 2 :=
sorry

end tan_theta_solution_l1497_149716


namespace inequality_proof_l1497_149719

variable (x y z : ℝ)

theorem inequality_proof (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (hxyz : x + y + z = 1) :
  x * (1 - 2 * x) * (1 - 3 * x) + y * (1 - 2 * y) * (1 - 3 * y) + z * (1 - 2 * z) * (1 - 3 * z) ≥ 0 := 
sorry

end inequality_proof_l1497_149719


namespace correct_value_l1497_149740

theorem correct_value (x : ℕ) (h : 14 * x = 42) : 12 * x = 36 := by
  sorry

end correct_value_l1497_149740


namespace sum_of_first_nine_terms_l1497_149734

noncomputable def arithmetic_sequence_sum (a₁ d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

def a_n (a₁ d n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

theorem sum_of_first_nine_terms (a₁ d : ℕ) (h : a_n a₁ d 2 + a_n a₁ d 6 + a_n a₁ d 7 = 18) :
  arithmetic_sequence_sum a₁ d 9 = 54 :=
sorry

end sum_of_first_nine_terms_l1497_149734


namespace polygon_sides_eq_nine_l1497_149787

theorem polygon_sides_eq_nine (n : ℕ) (h : n - 1 = 8) : n = 9 := by
  sorry

end polygon_sides_eq_nine_l1497_149787


namespace line_equation_l1497_149703

theorem line_equation (A : (ℝ × ℝ)) (hA_x : A.1 = 2) (hA_y : A.2 = 0)
  (h_intercept : ∀ B : (ℝ × ℝ), B.1 = 0 → 2 * B.1 + B.2 + 2 = 0 → B = (0, -2)) :
  ∃ (l : ℝ × ℝ → Prop), (l A ∧ l (0, -2)) ∧ 
    (∀ x y : ℝ, l (x, y) ↔ x - y - 2 = 0) :=
by
  sorry

end line_equation_l1497_149703


namespace lcm_gcd_product_difference_l1497_149781
open Nat

theorem lcm_gcd_product_difference :
  (Nat.lcm 12 9) * (Nat.gcd 12 9) - (Nat.gcd 15 9) = 105 :=
by
  sorry

end lcm_gcd_product_difference_l1497_149781


namespace determine_coefficients_l1497_149771

theorem determine_coefficients (A B C : ℝ) 
  (h1 : 3 * A - 1 = 0)
  (h2 : 3 * A^2 + 3 * B = 0)
  (h3 : A^3 + 6 * A * B + 3 * C = 0) :
  A = 1 / 3 ∧ B = -1 / 9 ∧ C = 5 / 81 :=
by 
  sorry

end determine_coefficients_l1497_149771


namespace scientific_notation_of_0_00065_l1497_149721

/-- 
Prove that the decimal representation of a number 0.00065 can be expressed in scientific notation 
as 6.5 * 10^(-4)
-/
theorem scientific_notation_of_0_00065 : 0.00065 = 6.5 * 10^(-4) := 
by 
  sorry

end scientific_notation_of_0_00065_l1497_149721


namespace monthly_salary_l1497_149726

theorem monthly_salary (S : ℝ) (h1 : 0.20 * S + 1.20 * 0.80 * S = S) (h2 : S - 1.20 * 0.80 * S = 260) : S = 6500 :=
by
  sorry

end monthly_salary_l1497_149726


namespace principal_amount_borrowed_l1497_149770

theorem principal_amount_borrowed 
  (SI : ℝ) (R : ℝ) (T : ℝ) (P : ℝ) 
  (h1 : SI = 9000) 
  (h2 : R = 0.12) 
  (h3 : T = 3) 
  (h4 : SI = P * R * T) : 
  P = 25000 :=
sorry

end principal_amount_borrowed_l1497_149770


namespace value_of_expression_l1497_149748

theorem value_of_expression (x y z : ℝ) (h : (x * y * z) / (|x * y * z|) = 1) :
  (|x| / x + y / |y| + |z| / z) = 3 ∨ (|x| / x + y / |y| + |z| / z) = -1 :=
sorry

end value_of_expression_l1497_149748


namespace tom_received_20_percent_bonus_l1497_149707

-- Define the initial conditions
def tom_spent : ℤ := 250
def gems_per_dollar : ℤ := 100
def total_gems_received : ℤ := 30000

-- Calculate the number of gems received without the bonus
def gems_without_bonus : ℤ := tom_spent * gems_per_dollar
def bonus_gems : ℤ := total_gems_received - gems_without_bonus

-- Calculate the percentage of the bonus
def bonus_percentage : ℚ := (bonus_gems : ℚ) / gems_without_bonus * 100

-- State the theorem
theorem tom_received_20_percent_bonus : bonus_percentage = 20 := by
  sorry

end tom_received_20_percent_bonus_l1497_149707


namespace minimal_sum_of_squares_of_roots_l1497_149712

open Real

theorem minimal_sum_of_squares_of_roots :
  ∀ a : ℝ,
  (let x1 := 3*a + 1;
   let x2 := 2*a^2 - 3*a - 2;
   (a^2 + 18*a + 9) ≥ 0 →
   (x1^2 - 2*x2) = (5*a^2 + 12*a + 5) →
   a = -9 + 6*sqrt 2) :=
by
  sorry

end minimal_sum_of_squares_of_roots_l1497_149712


namespace cylinder_lateral_surface_area_l1497_149737

theorem cylinder_lateral_surface_area :
  let side := 20
  let radius := side / 2
  let height := side
  2 * Real.pi * radius * height = 400 * Real.pi :=
by
  let side := 20
  let radius := side / 2
  let height := side
  sorry

end cylinder_lateral_surface_area_l1497_149737


namespace toms_age_is_16_l1497_149722

variable (J T : ℕ) -- John's current age is J and Tom's current age is T

-- Condition 1: John was thrice as old as Tom 6 years ago
axiom h1 : J - 6 = 3 * (T - 6)

-- Condition 2: John will be 2 times as old as Tom in 4 years
axiom h2 : J + 4 = 2 * (T + 4)

-- Proving Tom's current age is 16
theorem toms_age_is_16 : T = 16 := by
  sorry

end toms_age_is_16_l1497_149722


namespace trigonometric_identity_l1497_149720

theorem trigonometric_identity :
  4 * Real.cos (10 * (Real.pi / 180)) - Real.tan (80 * (Real.pi / 180)) = -Real.sqrt 3 := 
by 
  sorry

end trigonometric_identity_l1497_149720


namespace contrapositive_l1497_149791

theorem contrapositive (x : ℝ) (h : x^2 ≥ 1) : x ≥ 0 ∨ x ≤ -1 :=
sorry

end contrapositive_l1497_149791


namespace max_k_possible_l1497_149767

-- Given the sequence formed by writing all three-digit numbers from 100 to 999 consecutively
def digits_sequence : List Nat := List.join (List.map (fun n => [n / 100, (n / 10) % 10, n % 10]) (List.range' 100 (999 - 100 + 1)))

-- Function to get a k-digit number from the sequence
def get_k_digit_number (seq : List Nat) (start k : Nat) : List Nat := seq.drop start |>.take k

-- Statement to prove the maximum k
theorem max_k_possible : ∃ k : Nat, (∀ start1 start2, start1 ≠ start2 → get_k_digit_number digits_sequence start1 5 = get_k_digit_number digits_sequence start2 5) ∧ (¬ ∃ k' > 5, (∀ start1 start2, start1 ≠ start2 → get_k_digit_number digits_sequence start1 k' = get_k_digit_number digits_sequence start2 k')) :=
sorry

end max_k_possible_l1497_149767


namespace bakery_item_count_l1497_149782

theorem bakery_item_count : ∃ (s c : ℕ), 5 * s + 25 * c = 500 ∧ s + c = 12 := by
  sorry

end bakery_item_count_l1497_149782


namespace simplify_expression_l1497_149739

variable (x y : ℝ)

theorem simplify_expression : 3 * y + 5 * y + 6 * y + 2 * x + 4 * x = 14 * y + 6 * x :=
by
  sorry

end simplify_expression_l1497_149739


namespace poly_roots_equivalence_l1497_149736

noncomputable def poly (a b c d : ℝ) (x : ℝ) : ℝ := x^4 + a * x^3 + b * x^2 + c * x + d

theorem poly_roots_equivalence (a b c d : ℝ) 
    (h1 : poly a b c d 4 = 102) 
    (h2 : poly a b c d 3 = 102) 
    (h3 : poly a b c d (-3) = 102) 
    (h4 : poly a b c d (-4) = 102) : 
    {x : ℝ | poly a b c d x = 246} = {0, 5, -5} := 
by 
    sorry

end poly_roots_equivalence_l1497_149736


namespace max_lights_correct_l1497_149749

def max_lights_on (n : ℕ) : ℕ :=
  if n % 2 = 0 then n^2 / 2 else (n^2 - 1) / 2

theorem max_lights_correct (n : ℕ) :
  max_lights_on n = if n % 2 = 0 then n^2 / 2 else (n^2 - 1) / 2 :=
by sorry

end max_lights_correct_l1497_149749


namespace paint_cans_needed_l1497_149790

-- Conditions as definitions
def bedrooms : ℕ := 3
def other_rooms : ℕ := 2 * bedrooms
def paint_per_room : ℕ := 2
def color_can_capacity : ℕ := 1
def white_can_capacity : ℕ := 3

-- Total gallons needed
def total_color_gallons_needed : ℕ := paint_per_room * bedrooms
def total_white_gallons_needed : ℕ := paint_per_room * other_rooms

-- Total cans needed
def total_color_cans_needed : ℕ := total_color_gallons_needed / color_can_capacity
def total_white_cans_needed : ℕ := total_white_gallons_needed / white_can_capacity
def total_cans_needed : ℕ := total_color_cans_needed + total_white_cans_needed

theorem paint_cans_needed : total_cans_needed = 10 := by
  -- Proof steps (skipped) to show total_cans_needed = 10
  sorry

end paint_cans_needed_l1497_149790


namespace probability_both_tell_truth_l1497_149758

theorem probability_both_tell_truth (pA pB : ℝ) (hA : pA = 0.80) (hB : pB = 0.60) : pA * pB = 0.48 :=
by
  subst hA
  subst hB
  sorry

end probability_both_tell_truth_l1497_149758


namespace sum_of_fractions_l1497_149709

def S_1 : List ℚ := List.range' 1 10 |>.map (λ n => n / 10)
def S_2 : List ℚ := List.replicate 4 (20 / 10)

def total_sum : ℚ := S_1.sum + S_2.sum

theorem sum_of_fractions : total_sum = 12.5 := by
  sorry

end sum_of_fractions_l1497_149709


namespace find_N_l1497_149792

theorem find_N (a b c N : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : N = a * b * c) (h2 : N = 8 * (a + b + c)) (h3 : c = a + b) : N = 272 :=
sorry

end find_N_l1497_149792


namespace employed_females_percentage_l1497_149731

theorem employed_females_percentage (total_population : ℝ) (total_employed_percentage : ℝ) (employed_males_percentage : ℝ) :
  total_employed_percentage = 0.7 →
  employed_males_percentage = 0.21 →
  total_population > 0 →
  (total_employed_percentage - employed_males_percentage) / total_employed_percentage * 100 = 70 :=
by
  intros h1 h2 h3
  -- Proof is omitted.
  sorry

end employed_females_percentage_l1497_149731


namespace schoolchildren_chocolate_l1497_149729

theorem schoolchildren_chocolate (m d : ℕ) 
  (h1 : 7 * d + 2 * m > 36)
  (h2 : 8 * d + 4 * m < 48) :
  m = 1 ∧ d = 5 :=
by
  sorry

end schoolchildren_chocolate_l1497_149729


namespace sum_of_vertices_l1497_149742

theorem sum_of_vertices (num_triangle num_hexagon : ℕ) (vertices_triangle vertices_hexagon : ℕ) :
  num_triangle = 1 → vertices_triangle = 3 →
  num_hexagon = 3 → vertices_hexagon = 6 →
  num_triangle * vertices_triangle + num_hexagon * vertices_hexagon = 21 :=
by
  intros h1 h2 h3 h4
  sorry

end sum_of_vertices_l1497_149742


namespace jason_nickels_is_52_l1497_149700

theorem jason_nickels_is_52 (n q : ℕ) (h1 : 5 * n + 10 * q = 680) (h2 : q = n - 10) : n = 52 :=
sorry

end jason_nickels_is_52_l1497_149700


namespace tagged_fish_in_second_catch_l1497_149780

-- Definitions and conditions
def total_fish_in_pond : ℕ := 1750
def tagged_fish_initial : ℕ := 70
def fish_caught_second_time : ℕ := 50
def ratio_tagged_fish : ℚ := tagged_fish_initial / total_fish_in_pond

-- Theorem statement
theorem tagged_fish_in_second_catch (T : ℕ) : (T : ℚ) / fish_caught_second_time = ratio_tagged_fish → T = 2 :=
by
  sorry

end tagged_fish_in_second_catch_l1497_149780


namespace line_does_not_pass_second_quadrant_l1497_149796

theorem line_does_not_pass_second_quadrant (a : ℝ) (ha : a ≠ 0) :
  ∀ (x y : ℝ), (x - y - a^2 = 0) → ¬(x < 0 ∧ y > 0) :=
sorry

end line_does_not_pass_second_quadrant_l1497_149796


namespace rebecca_tent_stakes_l1497_149741

variables (T D W : ℕ)

-- Conditions
def drink_mix_eq : Prop := D = 3 * T
def water_eq : Prop := W = T + 2
def total_items_eq : Prop := T + D + W = 22

-- Problem statement
theorem rebecca_tent_stakes 
  (h1 : drink_mix_eq T D)
  (h2 : water_eq T W)
  (h3 : total_items_eq T D W) : 
  T = 4 := 
sorry

end rebecca_tent_stakes_l1497_149741


namespace fruit_problem_l1497_149725

def number_of_pears (A : ℤ) : ℤ := (3 * A) / 5
def number_of_apples (B : ℤ) : ℤ := (3 * B) / 7

theorem fruit_problem
  (A B : ℤ)
  (h1 : A + B = 82)
  (h2 : abs (A - B) < 10)
  (x : ℤ := (2 * A) / 5)
  (y : ℤ := (4 * B) / 7) :
  number_of_pears A = 24 ∧ number_of_apples B = 18 :=
by
  sorry

end fruit_problem_l1497_149725


namespace time_to_cross_pole_correct_l1497_149738

noncomputable def speed_kmph : ℝ := 160 -- Speed of the train in kmph
noncomputable def length_meters : ℝ := 800.064 -- Length of the train in meters

noncomputable def conversion_factor : ℝ := 1000 / 3600 -- Conversion factor from kmph to m/s
noncomputable def speed_mps : ℝ := speed_kmph * conversion_factor -- Speed of the train in m/s

noncomputable def time_to_cross_pole : ℝ := length_meters / speed_mps -- Time to cross the pole

theorem time_to_cross_pole_correct :
  time_to_cross_pole = 800.064 / (160 * (1000 / 3600)) :=
sorry

end time_to_cross_pole_correct_l1497_149738


namespace complex_poly_root_exists_l1497_149743

noncomputable def polynomial_has_complex_root (P : Polynomial ℂ) : Prop :=
  ∃ z : ℂ, P.eval z = 0

theorem complex_poly_root_exists (P : Polynomial ℂ) : polynomial_has_complex_root P :=
sorry

end complex_poly_root_exists_l1497_149743

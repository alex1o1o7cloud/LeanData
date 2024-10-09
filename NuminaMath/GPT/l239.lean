import Mathlib

namespace integer_values_not_satisfying_inequality_l239_23925

theorem integer_values_not_satisfying_inequality :
  (∃ x : ℤ, ¬(3 * x^2 + 17 * x + 28 > 25)) ∧ (∃ x1 x2 : ℤ, x1 = -2 ∧ x2 = -1) ∧
  ∀ x : ℤ, (x = -2 ∨ x = -1) -> ¬(3 * x^2 + 17 * x + 28 > 25) :=
by
  sorry

end integer_values_not_satisfying_inequality_l239_23925


namespace find_k_l239_23901

-- Auxiliary function to calculate the product of the digits of a number
def productOfDigits (n : ℕ) : ℕ :=
  (n.digits 10).foldl (λ acc d => acc * d) 1

theorem find_k (k : ℕ) (h1 : 0 < k) (h2 : productOfDigits k = (25 * k) / 8 - 211) : 
  k = 72 ∨ k = 88 :=
by
  sorry

end find_k_l239_23901


namespace find_digit_B_l239_23981

theorem find_digit_B (A B : ℕ) (h1 : 100 * A + 78 - (210 + B) = 364) : B = 4 :=
by sorry

end find_digit_B_l239_23981


namespace number_of_cow_herds_l239_23934

theorem number_of_cow_herds 
    (total_cows : ℕ) 
    (cows_per_herd : ℕ) 
    (h1 : total_cows = 320)
    (h2 : cows_per_herd = 40) : 
    total_cows / cows_per_herd = 8 :=
by
  sorry

end number_of_cow_herds_l239_23934


namespace propositions_are_3_and_4_l239_23927

-- Conditions
def stmt_1 := "Is it fun to study math?"
def stmt_2 := "Do your homework well and strive to pass the math test next time;"
def stmt_3 := "2 is not a prime number"
def stmt_4 := "0 is a natural number"

-- Representation of a propositional statement
def isPropositional (stmt : String) : Bool :=
  stmt ≠ stmt_1 ∧ stmt ≠ stmt_2

-- The theorem proving the question given the conditions
theorem propositions_are_3_and_4 :
  isPropositional stmt_3 ∧ isPropositional stmt_4 :=
by
  -- Proof to be filled in later
  sorry

end propositions_are_3_and_4_l239_23927


namespace problem1_problem2_l239_23972

-- Problem 1
theorem problem1 (x : ℝ) (hx1 : x ≠ 1) (hx2 : x ≠ 0) :
  (x^2 + x) / (x^2 - 2 * x + 1) / (2 / (x - 1) - 1 / x) = x^2 / (x - 1) := by
  sorry

-- Problem 2
theorem problem2 (x : ℤ) (hx1 : x > 0) :
  (2 * x + 1) / 3 - (5 * x - 1) / 2 < 1 ∧ 
  (5 * x - 1 < 3 * (x + 2)) →
  x = 1 ∨ x = 2 ∨ x = 3 := by
  sorry

end problem1_problem2_l239_23972


namespace painting_area_l239_23917

theorem painting_area
  (wall_height : ℝ) (wall_length : ℝ)
  (window_height : ℝ) (window_length : ℝ)
  (door_height : ℝ) (door_length : ℝ)
  (cond1 : wall_height = 10) (cond2 : wall_length = 15)
  (cond3 : window_height = 3) (cond4 : window_length = 5)
  (cond5 : door_height = 2) (cond6 : door_length = 7) :
  wall_height * wall_length - window_height * window_length - door_height * door_length = 121 := 
by
  simp [cond1, cond2, cond3, cond4, cond5, cond6]
  sorry

end painting_area_l239_23917


namespace tangent_line_at_A_tangent_line_through_B_l239_23908

open Real

noncomputable def f (x : ℝ) : ℝ := 4 / x
noncomputable def f' (x : ℝ) : ℝ := -4 / (x^2)

theorem tangent_line_at_A : 
  ∃ m b, m = -1 ∧ b = 4 ∧ (∀ x, 1 ≤ x → (x + b = 4)) :=
sorry

theorem tangent_line_through_B :
  ∃ m b, m = 4 ∧ b = -8 ∧ (∀ x, 1 ≤ x → (4*x + b = 8)) :=
sorry

end tangent_line_at_A_tangent_line_through_B_l239_23908


namespace hyperbola_equation_Q_on_fixed_circle_l239_23982

-- Define the hyperbola and necessary conditions
def hyperbola (a : ℝ) (x y : ℝ) : Prop := x^2 / a^2 - y^2 / (3 * a^2) = 1

-- Given conditions
variables (a : ℝ) (h_pos : a > 0)
variables (F1 F2 : ℝ × ℝ)
variables (dist_F2_asymptote : ℝ) (h_dist : dist_F2_asymptote = sqrt 3)
variables (left_vertex : ℝ × ℝ) (right_branch_intersect : ℝ × ℝ)
variables (line_x_half : ℝ × ℝ)
variables (line_PF2 : ℝ × ℝ)
variables (point_Q : ℝ × ℝ)

-- Prove that the equation of the hyperbola is correct
theorem hyperbola_equation :
  hyperbola a x y ↔ x^2 - y^2 / 3 = 1 :=
sorry

-- Prove that point Q lies on a fixed circle
theorem Q_on_fixed_circle :
  dist point_Q F2 = 4 :=
sorry

end hyperbola_equation_Q_on_fixed_circle_l239_23982


namespace remainder_check_l239_23905

theorem remainder_check (q : ℕ) (n : ℕ) (h1 : q = 3^19) (h2 : n = 1162261460) : q % n = 7 := by
  rw [h1, h2]
  -- Proof skipped
  sorry

end remainder_check_l239_23905


namespace simplify_and_evaluate_l239_23969

theorem simplify_and_evaluate (a : ℝ) (h : a = 3) : ((2 * a / (a + 1) - 1) / ((a - 1)^2 / (a + 1))) = 1 / 2 := by
  sorry

end simplify_and_evaluate_l239_23969


namespace smallest_four_digit_congruent_one_mod_17_l239_23996

theorem smallest_four_digit_congruent_one_mod_17 :
  ∃ (n : ℕ), 1000 ≤ n ∧ n % 17 = 1 ∧ n = 1003 :=
by
sorry

end smallest_four_digit_congruent_one_mod_17_l239_23996


namespace tan_of_alpha_l239_23991

theorem tan_of_alpha
  (α : ℝ)
  (h1 : Real.sin (α + Real.pi / 2) = 1 / 3)
  (h2 : 0 < α ∧ α < Real.pi / 2) : 
  Real.tan α = 2 * Real.sqrt 2 := 
sorry

end tan_of_alpha_l239_23991


namespace total_dots_not_visible_l239_23959

-- Define the conditions and variables
def total_dots_one_die : Nat := 1 + 2 + 3 + 4 + 5 + 6
def number_of_dice : Nat := 4
def total_dots_all_dice : Nat := number_of_dice * total_dots_one_die
def visible_numbers : List Nat := [6, 6, 4, 4, 3, 2, 1]

-- The question can be formalized as proving that the total number of dots not visible is 58
theorem total_dots_not_visible :
  total_dots_all_dice - visible_numbers.sum = 58 :=
by
  -- Statement only, proof skipped
  sorry

end total_dots_not_visible_l239_23959


namespace vector_addition_example_l239_23955

theorem vector_addition_example : 
  let v1 := (⟨-5, 3⟩ : ℝ × ℝ)
  let v2 := (⟨7, -6⟩ : ℝ × ℝ)
  v1 + v2 = (⟨2, -3⟩ : ℝ × ℝ) := 
by {
  sorry
}

end vector_addition_example_l239_23955


namespace cost_of_lamp_and_flashlight_max_desk_lamps_l239_23954

-- Part 1: Cost of purchasing one desk lamp and one flashlight
theorem cost_of_lamp_and_flashlight (x : ℕ) (desk_lamp_cost flashlight_cost : ℕ) 
        (hx : desk_lamp_cost = x + 20)
        (hdesk : 400 = x / 2 * desk_lamp_cost)
        (hflash : 160 = x * flashlight_cost)
        (hnum : desk_lamp_cost = 2 * flashlight_cost) : 
        desk_lamp_cost = 25 ∧ flashlight_cost = 5 :=
sorry

-- Part 2: Maximum number of desk lamps Rongqing Company can purchase
theorem max_desk_lamps (a : ℕ) (desk_lamp_cost flashlight_cost : ℕ)
        (hc1 : desk_lamp_cost = 25)
        (hc2 : flashlight_cost = 5)
        (free_flashlight : ℕ := a) (required_flashlight : ℕ := 2 * a + 8) 
        (total_cost : ℕ := desk_lamp_cost * a + flashlight_cost * required_flashlight)
        (hcost : total_cost ≤ 670) :
        a ≤ 21 :=
sorry

end cost_of_lamp_and_flashlight_max_desk_lamps_l239_23954


namespace table_height_l239_23976

variable (l h w : ℝ)

-- Given conditions:
def conditionA := l + h - w = 36
def conditionB := w + h - l = 30

-- Proof that height of the table h is 33 inches
theorem table_height {l h w : ℝ} 
  (h1 : l + h - w = 36) 
  (h2 : w + h - l = 30) : 
  h = 33 := 
by
  sorry

end table_height_l239_23976


namespace inequality_for_abcd_one_l239_23993

theorem inequality_for_abcd_one (a b c d : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d) (h_prod : a * b * c * d = 1) :
  (1 / (1 + a)) + (1 / (1 + b)) + (1 / (1 + c)) + (1 / (1 + d)) > 1 := 
by
  sorry

end inequality_for_abcd_one_l239_23993


namespace tangent_lines_parallel_to_4x_minus_1_l239_23919

noncomputable def f (x : ℝ) : ℝ := x^3 + x - 2

theorem tangent_lines_parallel_to_4x_minus_1 :
  ∃ (a b : ℝ), (f a = b ∧ 3 * a^2 + 1 = 4) → (b = 4 * a - 4 ∨ b = 4 * a) :=
by
  sorry

end tangent_lines_parallel_to_4x_minus_1_l239_23919


namespace unique_prime_triple_l239_23978

/-- A prime is an integer greater than 1 whose only positive integer divisors are itself and 1. -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

/-- Prove that the only triple of primes (p, q, r), such that p = q + 2 and q = r + 2 is (7, 5, 3). -/
theorem unique_prime_triple (p q r : ℕ) (hp : is_prime p) (hq : is_prime q) (hr : is_prime r) :
  (p = q + 2) ∧ (q = r + 2) → (p = 7 ∧ q = 5 ∧ r = 3) := by
  sorry

end unique_prime_triple_l239_23978


namespace greatest_possible_positive_integer_difference_l239_23946

theorem greatest_possible_positive_integer_difference (x y : ℤ) (hx : 4 < x) (hx' : x < 6) (hy : 6 < y) (hy' : y < 10) :
  y - x = 4 :=
sorry

end greatest_possible_positive_integer_difference_l239_23946


namespace mrs_bil_earnings_percentage_in_may_l239_23950

theorem mrs_bil_earnings_percentage_in_may
  (M F : ℝ)
  (h₁ : 1.10 * M / (1.10 * M + F) = 0.7196) :
  M / (M + F) = 0.70 :=
sorry

end mrs_bil_earnings_percentage_in_may_l239_23950


namespace cubes_not_touching_tin_foil_volume_l239_23975

-- Definitions for the conditions given
variables (l w h : ℕ)
-- Condition 1: Width is twice the length
def width_twice_length := w = 2 * l
-- Condition 2: Width is twice the height
def width_twice_height := w = 2 * h
-- Condition 3: The adjusted width for the inner structure in inches
def adjusted_width := w = 8

-- The theorem statement to prove the final answer
theorem cubes_not_touching_tin_foil_volume : 
  width_twice_length l w → 
  width_twice_height w h →
  adjusted_width w →
  l * w * h = 128 :=
by
  intros h1 h2 h3
  sorry

end cubes_not_touching_tin_foil_volume_l239_23975


namespace baker_initial_cakes_l239_23926

theorem baker_initial_cakes (sold : ℕ) (left : ℕ) (initial : ℕ) 
  (h_sold : sold = 41) (h_left : left = 13) : 
  sold + left = initial → initial = 54 :=
by
  intros
  exact sorry

end baker_initial_cakes_l239_23926


namespace min_value_expr_l239_23964

theorem min_value_expr : ∀ (x : ℝ), 0 < x ∧ x < 4 → ∃ y : ℝ, y = (1 / (4 - x) + 2 / x) ∧ y = (3 + 2 * Real.sqrt 2) / 4 :=
by
  sorry

end min_value_expr_l239_23964


namespace work_completion_days_l239_23907

theorem work_completion_days (D_a : ℝ) (R_a R_b : ℝ)
  (h1 : R_a = 1 / D_a)
  (h2 : R_b = 1 / (1.5 * D_a))
  (h3 : R_a = 1.5 * R_b)
  (h4 : 1 / 18 = R_a + R_b) : D_a = 30 := 
by
  sorry

end work_completion_days_l239_23907


namespace single_elimination_games_needed_l239_23906

theorem single_elimination_games_needed (teams : ℕ) (h : teams = 19) : 
∃ games, games = 18 ∧ (∀ (teams_left : ℕ), teams_left = teams - 1 → games = teams - 1) :=
by
  -- define the necessary parameters and properties here 
  sorry

end single_elimination_games_needed_l239_23906


namespace problem_statement_l239_23916

theorem problem_statement (a b c : ℝ) (h1 : a + 2 * b + 3 * c = 12) (h2 : a^2 + b^2 + c^2 = a * b + a * c + b * c) :
  a + b^2 + c^3 = 14 :=
sorry

end problem_statement_l239_23916


namespace sqrt_simplify_l239_23995

theorem sqrt_simplify :
  Real.sqrt 450 = 15 * Real.sqrt 2 :=
by
  sorry

end sqrt_simplify_l239_23995


namespace stewart_farm_horse_food_l239_23939

theorem stewart_farm_horse_food 
  (ratio : ℚ) (food_per_horse : ℤ) (num_sheep : ℤ) (num_horses : ℤ)
  (h1 : ratio = 5 / 7)
  (h2 : food_per_horse = 230)
  (h3 : num_sheep = 40)
  (h4 : ratio * num_horses = num_sheep) : 
  (num_horses * food_per_horse = 12880) := 
sorry

end stewart_farm_horse_food_l239_23939


namespace smallest_arith_prog_l239_23966

theorem smallest_arith_prog (a d : ℝ) 
  (h1 : (a - 2 * d) < (a - d) ∧ (a - d) < a ∧ a < (a + d) ∧ (a + d) < (a + 2 * d))
  (h2 : (a - 2 * d)^2 + (a - d)^2 + a^2 + (a + d)^2 + (a + 2 * d)^2 = 70)
  (h3 : (a - 2 * d)^3 + (a - d)^3 + a^3 + (a + d)^3 + (a + 2 * d)^3 = 0)
  : (a - 2 * d) = -2 * Real.sqrt 7 :=
sorry

end smallest_arith_prog_l239_23966


namespace increasing_function_implies_a_nonpositive_l239_23977

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x - 1

theorem increasing_function_implies_a_nonpositive (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x ≤ f a y) → a ≤ 0 :=
by
  sorry

end increasing_function_implies_a_nonpositive_l239_23977


namespace correct_option_is_D_l239_23992

noncomputable def data : List ℕ := [7, 5, 3, 5, 10]

theorem correct_option_is_D :
  let mean := (7 + 5 + 3 + 5 + 10) / 5
  let variance := (1 / 5 : ℚ) * ((7 - mean) ^ 2 + (5 - mean) ^ 2 + (5 - mean) ^ 2 + (3 - mean) ^ 2 + (10 - mean) ^ 2)
  let mode := 5
  let median := 5
  mean = 6 ∧ variance ≠ 3.6 ∧ mode ≠ 10 ∧ median ≠ 3 :=
by
  sorry

end correct_option_is_D_l239_23992


namespace max_value_of_function_l239_23928

theorem max_value_of_function (x : ℝ) (h : x < 5 / 4) :
    (∀ y, y = 4 * x - 2 + 1 / (4 * x - 5) → y ≤ 1):=
sorry

end max_value_of_function_l239_23928


namespace sqrt_meaningful_range_l239_23965

theorem sqrt_meaningful_range (x : ℝ) (h : 0 ≤ x - 2) : x ≥ 2 :=
sorry

end sqrt_meaningful_range_l239_23965


namespace chuck_distance_l239_23947

theorem chuck_distance
  (total_time : ℝ) (out_speed : ℝ) (return_speed : ℝ) (D : ℝ)
  (h1 : total_time = 3)
  (h2 : out_speed = 16)
  (h3 : return_speed = 24)
  (h4 : D / out_speed + D / return_speed = total_time) :
  D = 28.80 :=
by
  sorry

end chuck_distance_l239_23947


namespace shaded_trapezoids_perimeter_l239_23985

theorem shaded_trapezoids_perimeter :
  let l := 8
  let w := 6
  let half_diagonal_1 := (l^2 + w^2) / 2
  let perimeter := 2 * (w + (half_diagonal_1 / l))
  let total_perimeter := perimeter + perimeter + half_diagonal_1
  total_perimeter = 48 :=
by 
  sorry

end shaded_trapezoids_perimeter_l239_23985


namespace find_max_term_of_sequence_l239_23987

theorem find_max_term_of_sequence :
  ∃ m : ℕ, (m = 8) ∧ ∀ n : ℕ, (0 < n → n ≠ m → a_n = (n - 7) / (n - 5 * Real.sqrt 2)) :=
by
  sorry

end find_max_term_of_sequence_l239_23987


namespace highest_total_zits_l239_23973

def zits_per_student_Swanson := 5
def students_Swanson := 25
def total_zits_Swanson := zits_per_student_Swanson * students_Swanson -- should be 125

def zits_per_student_Jones := 6
def students_Jones := 32
def total_zits_Jones := zits_per_student_Jones * students_Jones -- should be 192

def zits_per_student_Smith := 7
def students_Smith := 20
def total_zits_Smith := zits_per_student_Smith * students_Smith -- should be 140

def zits_per_student_Brown := 8
def students_Brown := 16
def total_zits_Brown := zits_per_student_Brown * students_Brown -- should be 128

def zits_per_student_Perez := 4
def students_Perez := 30
def total_zits_Perez := zits_per_student_Perez * students_Perez -- should be 120

theorem highest_total_zits : 
  total_zits_Jones = max total_zits_Swanson (max total_zits_Smith (max total_zits_Brown (max total_zits_Perez total_zits_Jones))) :=
by
  sorry

end highest_total_zits_l239_23973


namespace regression_line_intercept_l239_23936

theorem regression_line_intercept
  (x : ℕ → ℝ)
  (y : ℕ → ℝ)
  (h_x_sum : x 1 + x 2 + x 3 + x 4 + x 5 + x 6 = 10)
  (h_y_sum : y 1 + y 2 + y 3 + y 4 + y 5 + y 6 = 4) :
  ∃ a : ℝ, (∀ i, y i = (1 / 4) * x i + a) → a = 1 / 4 :=
by
  sorry

end regression_line_intercept_l239_23936


namespace trigonometric_expression_l239_23967

theorem trigonometric_expression (θ : ℝ) (h : Real.tan θ = -3) :
    2 / (3 * (Real.sin θ) ^ 2 - (Real.cos θ) ^ 2) = 10 / 13 :=
by
  -- sorry to skip the proof
  sorry

end trigonometric_expression_l239_23967


namespace cost_of_cookies_equal_3_l239_23949

def selling_price : ℝ := 1.5
def cost_price : ℝ := 1
def number_of_bracelets : ℕ := 12
def amount_left : ℝ := 3

theorem cost_of_cookies_equal_3 : 
  (selling_price - cost_price) * number_of_bracelets - amount_left = 3 := by
  sorry

end cost_of_cookies_equal_3_l239_23949


namespace value_of_k_range_of_k_l239_23937

noncomputable def quadratic_eq_has_real_roots (k : ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), x₁ ^ 2 + (2 - 2 * k) * x₁ + k ^ 2 = 0 ∧
    x₂ ^ 2 + (2 - 2 * k) * x₂ + k ^ 2 = 0

def roots_condition (x₁ x₂ : ℝ) : Prop :=
  |(x₁ + x₂)| + 1 = x₁ * x₂

theorem value_of_k (k : ℝ) :
  quadratic_eq_has_real_roots k →
  (∀ (x₁ x₂ : ℝ), roots_condition x₁ x₂ → x₁ ^ 2 + (2 - 2 * k) * x₁ + k ^ 2 = 0 →
                    x₂ ^ 2 + (2 - 2 * k) * x₂ + k ^ 2 = 0 → k = -3) :=
by sorry

theorem range_of_k :
  ∃ (k : ℝ), quadratic_eq_has_real_roots k → k ≤ 1 :=
by sorry

end value_of_k_range_of_k_l239_23937


namespace sum_binomial_coefficients_l239_23958

theorem sum_binomial_coefficients :
  let a := 1
  let b := 1
  let binomial := (2 * a + 2 * b)
  (binomial)^7 = 16384 := by
  -- Proof omitted
  sorry

end sum_binomial_coefficients_l239_23958


namespace max_sum_ge_zero_l239_23974

-- Definition for max and min functions for real numbers
noncomputable def max_real (x y : ℝ) := if x ≥ y then x else y
noncomputable def min_real (x y : ℝ) := if x ≤ y then x else y

-- Condition: a + b + c + d = 0
def sum_zero (a b c d : ℝ) := a + b + c + d = 0

-- Lean statement for Problem (a)
theorem max_sum_ge_zero (a b c d : ℝ) (h : sum_zero a b c d) : 
  max_real a b + max_real a c + max_real a d + max_real b c + max_real b d + max_real c d ≥ 0 :=
sorry

-- Lean statement for Problem (b)
def find_max_k : ℕ :=
2

end max_sum_ge_zero_l239_23974


namespace quadratic_difference_sum_l239_23956

theorem quadratic_difference_sum :
  let a := 2
  let b := -10
  let c := 3
  let Δ := b * b - 4 * a * c
  let root1 := (10 + Real.sqrt Δ) / (2 * a)
  let root2 := (10 - Real.sqrt Δ) / (2 * a)
  let diff := root1 - root2
  let m := 19  -- from the difference calculation
  let n := 1   -- from the simplified form
  m + n = 20 :=
by
  -- Placeholders for calculation and proof steps.
  sorry

end quadratic_difference_sum_l239_23956


namespace sum_of_roots_l239_23909

variables {a b c : ℝ}

-- Conditions
-- The polynomial with roots a, b, c
def poly (x : ℝ) : ℝ := 24 * x^3 - 36 * x^2 + 14 * x - 1

-- The roots are in (0, 1)
def in_interval (x : ℝ) : Prop := 0 < x ∧ x < 1

-- All roots are distinct
def distinct (a b c : ℝ) : Prop := a ≠ b ∧ b ≠ c ∧ c ≠ a

-- Main Theorem
theorem sum_of_roots :
  (∀ x, poly x = 0 → x = a ∨ x = b ∨ x = c) →
  in_interval a →
  in_interval b →
  in_interval c →
  distinct a b c →
  (1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c) = 3 / 2) :=
by
  intros
  sorry

end sum_of_roots_l239_23909


namespace larry_wins_probability_l239_23948

noncomputable def probability (n : ℕ) : ℝ :=
  if n % 2 = 1 then (1/2)^(n) else 0

noncomputable def inf_geometric_sum (a : ℝ) (r : ℝ) : ℝ :=
  a / (1 - r)

theorem larry_wins_probability :
  inf_geometric_sum (1/2) (1/4) = 2/3 :=
by
  sorry

end larry_wins_probability_l239_23948


namespace interior_angle_of_regular_nonagon_l239_23932

theorem interior_angle_of_regular_nonagon : 
  let n := 9
  let sum_of_interior_angles := 180 * (n - 2)
  (sum_of_interior_angles / n) = 140 := 
by
  let n := 9
  let sum_of_interior_angles := 180 * (n - 2)
  show sum_of_interior_angles / n = 140
  sorry

end interior_angle_of_regular_nonagon_l239_23932


namespace combined_length_of_all_CDs_l239_23912

-- Define the lengths of each CD based on the conditions
def length_cd1 := 1.5
def length_cd2 := 1.5
def length_cd3 := 2 * length_cd1
def length_cd4 := length_cd2 / 2
def length_cd5 := length_cd1 + length_cd2

-- Define the combined length of all CDs
def combined_length := length_cd1 + length_cd2 + length_cd3 + length_cd4 + length_cd5

-- State the theorem
theorem combined_length_of_all_CDs : combined_length = 9.75 := by
  sorry

end combined_length_of_all_CDs_l239_23912


namespace alice_bob_coffee_shop_spending_l239_23902

theorem alice_bob_coffee_shop_spending (A B : ℝ) (h1 : B = 0.5 * A) (h2 : A = B + 15) : A + B = 45 :=
by
  sorry

end alice_bob_coffee_shop_spending_l239_23902


namespace survey_students_l239_23963

theorem survey_students (S F : ℕ) (h1 : F = 20 + 60) (h2 : F = 40 * S / 100) : S = 200 :=
by
  sorry

end survey_students_l239_23963


namespace compensation_problem_l239_23941

namespace CompensationProof

variables (a b c : ℝ)

def geometric_seq_with_ratio_1_by_2 (a b c : ℝ) : Prop :=
  c = (1/2) * b ∧ b = (1/2) * a

def total_compensation_eq (a b c : ℝ) : Prop :=
  4 * c + 2 * b + a = 50

theorem compensation_problem :
  total_compensation_eq a b c ∧ geometric_seq_with_ratio_1_by_2 a b c → c = 50 / 7 :=
sorry

end CompensationProof

end compensation_problem_l239_23941


namespace toys_produced_each_day_l239_23921

theorem toys_produced_each_day (weekly_production : ℕ) (days_worked : ℕ) 
  (h1 : weekly_production = 5500) (h2 : days_worked = 4) : 
  (weekly_production / days_worked = 1375) :=
sorry

end toys_produced_each_day_l239_23921


namespace volume_of_mixture_l239_23971

section
variable (Va Vb Vtotal : ℝ)

theorem volume_of_mixture :
  (Va / Vb = 3 / 2) →
  (800 * Va + 850 * Vb = 2460) →
  (Vtotal = Va + Vb) →
  Vtotal = 2.998 :=
by
  intros h1 h2 h3
  sorry
end

end volume_of_mixture_l239_23971


namespace inequality_ab_gt_ac_l239_23943

theorem inequality_ab_gt_ac {a b c : ℝ} (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) : a * b > a * c :=
sorry

end inequality_ab_gt_ac_l239_23943


namespace find_monthly_salary_l239_23929

variables (x h_1 h_2 h_3 : ℕ)

theorem find_monthly_salary 
    (half_salary_bank : h_1 = x / 2)
    (half_remaining_mortgage : h_2 = (h_1 - 300) / 2)
    (half_remaining_expenses : h_3 = (h_2 + 300) / 2)
    (remaining_salary : h_3 = 800) :
  x = 7600 :=
sorry

end find_monthly_salary_l239_23929


namespace ratio_of_width_perimeter_is_3_16_l239_23918

-- We define the conditions
def length_of_room : ℕ := 25
def width_of_room : ℕ := 15

-- We define the calculation and verification of the ratio
theorem ratio_of_width_perimeter_is_3_16 :
  let P := 2 * (length_of_room + width_of_room)
  let ratio := width_of_room / P
  let a := 15 / Nat.gcd 15 80
  let b := 80 / Nat.gcd 15 80
  (a, b) = (3, 16) :=
by 
  -- The proof is skipped with sorry
  sorry

end ratio_of_width_perimeter_is_3_16_l239_23918


namespace system_has_three_solutions_l239_23980

theorem system_has_three_solutions (a : ℝ) :
  (a = 4 ∨ a = 64 ∨ a = 51 + 10 * Real.sqrt 2) ↔
  ∃ (x y : ℝ), 
    (x = abs (y - Real.sqrt a) + Real.sqrt a - 4 
    ∧ (abs x - 6)^2 + (abs y - 8)^2 = 100) 
        ∧ (∃! x1 y1 : ℝ, (x1 = abs (y1 - Real.sqrt a) + Real.sqrt a - 4 
        ∧ (abs x1 - 6)^2 + (abs y1 - 8)^2 = 100)) :=
by
  sorry

end system_has_three_solutions_l239_23980


namespace sqrt_expression_l239_23962

theorem sqrt_expression : Real.sqrt (3^2 * 4^4) = 48 := by
  sorry

end sqrt_expression_l239_23962


namespace evaluate_expression_l239_23988

theorem evaluate_expression : 
  Int.ceil (7 / 3 : ℚ) + Int.floor (-7 / 3 : ℚ) + Int.ceil (4 / 5 : ℚ) + Int.floor (-4 / 5 : ℚ) = 0 :=
by
  sorry

end evaluate_expression_l239_23988


namespace reciprocal_eq_self_is_one_or_neg_one_l239_23998

/-- If a rational number equals its own reciprocal, then the number is either 1 or -1. -/
theorem reciprocal_eq_self_is_one_or_neg_one (x : ℚ) (h : x = 1 / x) : x = 1 ∨ x = -1 := 
by
  sorry

end reciprocal_eq_self_is_one_or_neg_one_l239_23998


namespace painting_time_l239_23968

theorem painting_time (n₁ t₁ n₂ t₂ : ℕ) (h1 : n₁ = 8) (h2 : t₁ = 12) (h3 : n₂ = 6) (h4 : n₁ * t₁ = n₂ * t₂) : t₂ = 16 :=
by
  sorry

end painting_time_l239_23968


namespace probability_different_colors_l239_23983

-- Define the number of chips of each color
def num_blue := 6
def num_red := 5
def num_yellow := 4
def num_green := 3

-- Total number of chips
def total_chips := num_blue + num_red + num_yellow + num_green

-- Probability of drawing a chip of different color
theorem probability_different_colors : 
  (num_blue / total_chips) * ((total_chips - num_blue) / total_chips) +
  (num_red / total_chips) * ((total_chips - num_red) / total_chips) +
  (num_yellow / total_chips) * ((total_chips - num_yellow) / total_chips) +
  (num_green / total_chips) * ((total_chips - num_green) / total_chips) =
  119 / 162 := 
sorry

end probability_different_colors_l239_23983


namespace harrys_fish_count_l239_23935

theorem harrys_fish_count : 
  let sam_fish := 7
  let joe_fish := 8 * sam_fish
  let harry_fish := 4 * joe_fish
  harry_fish = 224 :=
by
  sorry

end harrys_fish_count_l239_23935


namespace teacher_selection_l239_23970

/-- A school has 150 teachers, including 15 senior teachers, 45 intermediate teachers, 
and 90 junior teachers. By stratified sampling, 30 teachers are selected to 
participate in the teachers' representative conference. 
--/

def total_teachers : ℕ := 150
def senior_teachers : ℕ := 15
def intermediate_teachers : ℕ := 45
def junior_teachers : ℕ := 90

def total_selected_teachers : ℕ := 30
def selected_senior_teachers : ℕ := 3
def selected_intermediate_teachers : ℕ := 9
def selected_junior_teachers : ℕ := 18

def ratio (a b : ℕ) : ℕ × ℕ := (a / (gcd a b), b / (gcd a b))

theorem teacher_selection :
  ratio senior_teachers (gcd senior_teachers total_teachers) = ratio intermediate_teachers (gcd intermediate_teachers total_teachers) ∧
  ratio intermediate_teachers (gcd intermediate_teachers total_teachers) = ratio junior_teachers (gcd junior_teachers total_teachers) →
  selected_senior_teachers / selected_intermediate_teachers / selected_junior_teachers = 1 / 3 / 6 → 
  selected_senior_teachers + selected_intermediate_teachers + selected_junior_teachers = 30 :=
sorry

end teacher_selection_l239_23970


namespace find_a_l239_23933

theorem find_a (f : ℤ → ℤ) (h1 : ∀ (x : ℤ), f (2 * x + 1) = 3 * x + 2) (h2 : f a = 2) : a = 1 := by
sorry

end find_a_l239_23933


namespace students_received_B_l239_23984

/-!
# Problem Statement

Given:
1. In Mr. Johnson's class, 18 out of 30 students received a B.
2. Ms. Smith has 45 students in total, and the ratio of students receiving a B is the same as in Mr. Johnson's class.
Prove:
27 students in Ms. Smith's class received a B.
-/

theorem students_received_B (s1 s2 b1 : ℕ) (r1 : ℚ) (r2 : ℕ) (h₁ : s1 = 30) (h₂ : b1 = 18) (h₃ : s2 = 45) (h₄ : r1 = 3/5) 
(H : (b1 : ℚ) / s1 = r1) : r2 = 27 :=
by
  -- Conditions provided
  -- h₁ : s1 = 30
  -- h₂ : b1 = 18
  -- h₃ : s2 = 45
  -- h₄ : r1 = 3/5
  -- H : (b1 : ℚ) / s1 = r1
  sorry

end students_received_B_l239_23984


namespace watched_movies_count_l239_23904

theorem watched_movies_count {M : ℕ} (total_books total_movies read_books : ℕ) 
  (h1 : total_books = 15) (h2 : total_movies = 14) (h3 : read_books = 11) 
  (h4 : read_books = M + 1) : M = 10 :=
by
  sorry

end watched_movies_count_l239_23904


namespace triangle_side_length_l239_23986

theorem triangle_side_length (A : ℝ) (AC BC AB : ℝ) 
  (hA : A = 60)
  (hAC : AC = 4)
  (hBC : BC = 2 * Real.sqrt 3) :
  AB = 2 :=
sorry

end triangle_side_length_l239_23986


namespace solve_system_of_equations_l239_23931

variable (a b c : Real)

def K : Real := a * b * c + a^2 * c + c^2 * b + b^2 * a

theorem solve_system_of_equations 
    (h₁ : (a + b) * (a - b) * (b + c) * (b - c) * (c + a) * (c - a) ≠ 0)
    (h₂ : K a b c ≠ 0) :
    ∃ (x y z : Real), 
    x = b^2 - c^2 ∧
    y = c^2 - a^2 ∧
    z = a^2 - b^2 ∧
    (x / (b + c) + y / (c - a) = a + b) ∧
    (y / (c + a) + z / (a - b) = b + c) ∧
    (z / (a + b) + x / (b - c) = c + a) :=
by
  sorry

end solve_system_of_equations_l239_23931


namespace range_of_k_l239_23960

-- Definitions to use in statement
variable (k : ℝ)

-- Statement: Proving the range of k
theorem range_of_k (h : ∀ x : ℝ, k * x^2 - k * x - 1 < 0) : -4 < k ∧ k ≤ 0 :=
  sorry

end range_of_k_l239_23960


namespace cannot_determine_total_inhabitants_without_additional_info_l239_23944

variable (T : ℝ) (M F : ℝ)

axiom inhabitants_are_males_females : M + F = 1
axiom twenty_percent_of_males_are_literate : M * 0.20 * T = 0.20 * M * T
axiom twenty_five_percent_of_all_literates : 0.25 = 0.25 * T / T
axiom thirty_two_five_percent_of_females_are_literate : F = 1 - M ∧ F * 0.325 * T = 0.325 * (1 - M) * T

theorem cannot_determine_total_inhabitants_without_additional_info :
  ∃ (T : ℝ), True ↔ False := by
  sorry

end cannot_determine_total_inhabitants_without_additional_info_l239_23944


namespace num_divisors_of_36_l239_23989

theorem num_divisors_of_36 : (∃ (S : Finset ℤ), (∀ x, x ∈ S ↔ x ∣ 36) ∧ S.card = 18) :=
sorry

end num_divisors_of_36_l239_23989


namespace find_value_of_a_l239_23957

theorem find_value_of_a (b : ℤ) (q : ℚ) (a : ℤ) (h₁ : b = 2120) (h₂ : q = 0.5) (h₃ : (a : ℚ) / b = q) : a = 1060 :=
sorry

end find_value_of_a_l239_23957


namespace complex_division_identity_l239_23951

noncomputable def left_hand_side : ℂ := (-2 : ℂ) + (5 : ℂ) * Complex.I / (6 : ℂ) - (3 : ℂ) * Complex.I
noncomputable def right_hand_side : ℂ := - (9 : ℂ) / 15 + (8 : ℂ) / 15 * Complex.I

theorem complex_division_identity : left_hand_side = right_hand_side := 
by
  sorry

end complex_division_identity_l239_23951


namespace work_rate_c_l239_23922

variables (rate_a rate_b rate_c : ℚ)

-- Given conditions
axiom h1 : rate_a + rate_b = 1 / 15
axiom h2 : rate_a + rate_b + rate_c = 1 / 6

theorem work_rate_c : rate_c = 1 / 10 :=
by sorry

end work_rate_c_l239_23922


namespace great_grandson_age_l239_23997

theorem great_grandson_age (n : ℕ) : 
  ∃ n, (n * (n + 1)) / 2 = 666 :=
by
  -- Solution steps would go here
  sorry

end great_grandson_age_l239_23997


namespace n_consecutive_even_sum_l239_23938

theorem n_consecutive_even_sum (n k : ℕ) (hn : n > 2) (hk : k > 2) : 
  ∃ (a : ℕ), (n * (n - 1)^(k - 1)) = (2 * a + (2 * a + 2 * (n - 1))) / 2 * n :=
by
  sorry

end n_consecutive_even_sum_l239_23938


namespace cylindrical_surface_area_increase_l239_23900

theorem cylindrical_surface_area_increase (x : ℝ) :
  (2 * Real.pi * (10 + x)^2 + 2 * Real.pi * (10 + x) * (5 + x) = 
   2 * Real.pi * 10^2 + 2 * Real.pi * 10 * (5 + x)) →
   (x = -10 + 5 * Real.sqrt 6 ∨ x = -10 - 5 * Real.sqrt 6) :=
by
  intro h
  sorry

end cylindrical_surface_area_increase_l239_23900


namespace range_of_k_l239_23914

theorem range_of_k (k : ℝ) :
  (∃ x y : ℝ, k^2 * x^2 + y^2 - 4 * k * x + 2 * k * y + k^2 - 1 = 0 ∧ (x, y) = (0, 0)) →
  0 < |k| ∧ |k| < 1 :=
by
  intros
  sorry

end range_of_k_l239_23914


namespace min_odd_integers_is_zero_l239_23910

noncomputable def minOddIntegers (a b c d e f : ℤ) : ℕ :=
  if h₁ : a + b = 22 ∧ a + b + c + d = 36 ∧ a + b + c + d + e + f = 50 then
    0
  else
    6 -- default, just to match type expectations

theorem min_odd_integers_is_zero (a b c d e f : ℤ)
  (h₁ : a + b = 22)
  (h₂ : a + b + c + d = 36)
  (h₃ : a + b + c + d + e + f = 50) :
  minOddIntegers a b c d e f = 0 :=
  sorry

end min_odd_integers_is_zero_l239_23910


namespace dave_tickets_l239_23924

-- Definitions based on given conditions
def initial_tickets : ℕ := 25
def spent_tickets : ℕ := 22
def additional_tickets : ℕ := 15

-- Proof statement to demonstrate Dave would have 18 tickets
theorem dave_tickets : initial_tickets - spent_tickets + additional_tickets = 18 := by
  sorry

end dave_tickets_l239_23924


namespace roommates_condition_l239_23923

def f (x : ℝ) := 3 * x ^ 2 + 5 * x - 1
def g (x : ℝ) := 2 * x ^ 2 - 3 * x + 5

theorem roommates_condition : f 3 = 2 * g 3 + 5 := 
by {
  sorry
}

end roommates_condition_l239_23923


namespace s_at_1_l239_23952

def t (x : ℚ) := 5 * x - 12
def s (y : ℚ) := (y + 12) / 5 ^ 2 + 5 * ((y + 12) / 5) - 4

theorem s_at_1 : s 1 = 394 / 25 := by
  sorry

end s_at_1_l239_23952


namespace find_C_monthly_income_l239_23994

theorem find_C_monthly_income (A_m B_m C_m : ℝ) (h1 : A_m / B_m = 5 / 2) (h2 : B_m = 1.12 * C_m) (h3 : 12 * A_m = 504000) : C_m = 15000 :=
sorry

end find_C_monthly_income_l239_23994


namespace common_number_l239_23945

theorem common_number (a b c d e u v w : ℝ) (h1 : (a + b + c + d + e) / 5 = 7) 
                                            (h2 : (u + v + w) / 3 = 10) 
                                            (h3 : (a + b + c + d + e + u + v + w) / 8 = 8) 
                                            (h4 : a + b + c + d + e = 35) 
                                            (h5 : u + v + w = 30) 
                                            (h6 : a + b + c + d + e + u + v + w = 64) 
                                            (h7 : 35 + 30 = 65):
  d = u := 
by
  sorry

end common_number_l239_23945


namespace total_viewing_time_l239_23940

theorem total_viewing_time :
  let original_times := [4, 6, 7, 5, 9]
  let new_species_times := [3, 7, 8, 10]
  let total_breaks := 8
  let break_time_per_animal := 2
  let total_time := (original_times.sum + new_species_times.sum) + (total_breaks * break_time_per_animal)
  total_time = 75 :=
by
  sorry

end total_viewing_time_l239_23940


namespace ratio_a2_a3_l239_23903

namespace SequenceProof

def a (n : ℕ) : ℤ := 3 - 2^n

theorem ratio_a2_a3 : a 2 / a 3 = 1 / 5 := by
  sorry

end SequenceProof

end ratio_a2_a3_l239_23903


namespace cube_root_110592_l239_23915

theorem cube_root_110592 :
  (∃ x : ℕ, x^3 = 110592) ∧ 
  10^3 = 1000 ∧ 11^3 = 1331 ∧ 12^3 = 1728 ∧ 13^3 = 2197 ∧ 14^3 = 2744 ∧ 
  15^3 = 3375 ∧ 20^3 = 8000 ∧ 21^3 = 9261 ∧ 22^3 = 10648 ∧ 23^3 = 12167 ∧ 
  24^3 = 13824 ∧ 25^3 = 15625 → 48^3 = 110592 :=
by
  sorry

end cube_root_110592_l239_23915


namespace find_c_deg3_l239_23911

-- Define the polynomials f and g.
def f (x : ℚ) : ℚ := 2 - 10 * x + 4 * x^2 - 5 * x^3 + 7 * x^4
def g (x : ℚ) : ℚ := 5 - 3 * x - 8 * x^3 + 11 * x^4

-- The statement that needs proof.
theorem find_c_deg3 (c : ℚ) : (∀ x : ℚ, f x + c * g x ≠ 0 → f x + c * g x = 2 - 10 * x + 4 * x^2 - 5 * x^3 - c * 8 * x^3) ↔ c = -7 / 11 :=
sorry

end find_c_deg3_l239_23911


namespace coefficient_x2_expansion_l239_23930

theorem coefficient_x2_expansion : 
  let binomial_coeff (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k
  let expansion_coeff (a b : ℤ) (n k : ℕ) : ℤ := (b ^ k) * (binomial_coeff n k) * (a ^ (n - k))
  (expansion_coeff 1 (-2) 4 2) = 24 :=
by
  let binomial_coeff (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k
  let expansion_coeff (a b : ℤ) (n k : ℕ) : ℤ := (b ^ k) * (binomial_coeff n k) * (a ^ (n - k))
  have coeff : ℤ := expansion_coeff 1 (-2) 4 2
  sorry -- Proof goes here

end coefficient_x2_expansion_l239_23930


namespace base_conversion_problem_l239_23999

theorem base_conversion_problem (n d : ℕ) (hn : 0 < n) (hd : d < 10) 
  (h1 : 3 * n^2 + 2 * n + d = 263) (h2 : 3 * n^2 + 2 * n + 4 = 253 + 6 * d) : 
  n + d = 11 :=
by
  sorry

end base_conversion_problem_l239_23999


namespace swim_club_percentage_l239_23990

theorem swim_club_percentage (P : ℕ) (total_members : ℕ) (not_passed_taken_course : ℕ) (not_passed_not_taken_course : ℕ) :
  total_members = 50 →
  not_passed_taken_course = 5 →
  not_passed_not_taken_course = 30 →
  (total_members - (total_members * P / 100) = not_passed_taken_course + not_passed_not_taken_course) →
  P = 30 :=
by
  sorry

end swim_club_percentage_l239_23990


namespace climbing_difference_l239_23979

theorem climbing_difference (rate_matt rate_jason time : ℕ) (h_rate_matt : rate_matt = 6) (h_rate_jason : rate_jason = 12) (h_time : time = 7) : 
  rate_jason * time - rate_matt * time = 42 :=
by
  sorry

end climbing_difference_l239_23979


namespace expensive_feed_cost_l239_23961

/-- Tim and Judy mix two kinds of feed for pedigreed dogs. They made 35 pounds of feed worth 0.36 dollars per pound by mixing one kind worth 0.18 dollars per pound with another kind. They used 17 pounds of the cheaper kind in the mix. What is the cost per pound of the more expensive kind of feed? --/
theorem expensive_feed_cost 
  (total_feed : ℝ := 35) 
  (avg_cost : ℝ := 0.36) 
  (cheaper_feed : ℝ := 17) 
  (cheaper_cost : ℝ := 0.18) 
  (total_cost : ℝ := total_feed * avg_cost) 
  (cheaper_total_cost : ℝ := cheaper_feed * cheaper_cost) 
  (expensive_feed : ℝ := total_feed - cheaper_feed) : 
  (total_cost - cheaper_total_cost) / expensive_feed = 0.53 :=
by
  sorry

end expensive_feed_cost_l239_23961


namespace gina_initial_money_l239_23942

variable (M : ℝ)
variable (kept : ℝ := 170)

theorem gina_initial_money (h1 : M * 1 / 4 + M * 1 / 8 + M * 1 / 5 + kept = M) : 
  M = 400 :=
by
  sorry

end gina_initial_money_l239_23942


namespace diver_descend_rate_l239_23953

theorem diver_descend_rate (depth : ℕ) (time : ℕ) (rate : ℕ) 
  (h1 : depth = 6400) (h2 : time = 200) : rate = 32 :=
by
  sorry

end diver_descend_rate_l239_23953


namespace train_length_l239_23920

theorem train_length (v_kmph : ℝ) (t_s : ℝ) (L_p : ℝ) (L_t : ℝ) : 
  (v_kmph = 72) ∧ (t_s = 15) ∧ (L_p = 250) →
  L_t = 50 :=
by
  intro h
  sorry

end train_length_l239_23920


namespace correct_blanks_l239_23913

def fill_in_blanks (category : String) (plural_noun : String) : String :=
  "For many, winning remains " ++ category ++ " dream, but they continue trying their luck as there're always " ++ plural_noun ++ " chances that they might succeed."

theorem correct_blanks :
  fill_in_blanks "a" "" = "For many, winning remains a dream, but they continue trying their luck as there're always chances that they might succeed." :=
sorry

end correct_blanks_l239_23913

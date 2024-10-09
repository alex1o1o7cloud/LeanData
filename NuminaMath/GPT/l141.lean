import Mathlib

namespace neg_p_is_exists_x_l141_14143

variable (x : ℝ)

def p : Prop := ∀ x, x^2 + x + 1 ≠ 0

theorem neg_p_is_exists_x : ¬ p ↔ ∃ x, x^2 + x + 1 = 0 := by
  sorry

end neg_p_is_exists_x_l141_14143


namespace geometric_sequence_a1_value_l141_14138

variable {a_1 q : ℝ}

theorem geometric_sequence_a1_value
  (h1 : a_1 * q^2 = 1)
  (h2 : a_1 * q^4 + (3 / 2) * a_1 * q^3 = 1) :
  a_1 = 4 := by
  sorry

end geometric_sequence_a1_value_l141_14138


namespace find_number_l141_14160

theorem find_number :
  ∃ n : ℤ,
    (n % 12 = 11) ∧ 
    (n % 11 = 10) ∧ 
    (n % 10 = 9) ∧ 
    (n % 9 = 8) ∧ 
    (n % 8 = 7) ∧ 
    (n % 7 = 6) ∧ 
    (n % 6 = 5) ∧ 
    (n % 5 = 4) ∧ 
    (n % 4 = 3) ∧ 
    (n % 3 = 2) ∧ 
    (n % 2 = 1) ∧
    n = 27719 :=
sorry

end find_number_l141_14160


namespace range_of_a_l141_14171

noncomputable def a_n (a : ℝ) (n : ℕ) : ℝ :=
  if n <= 7 then (3 - a) * n - 3 else a ^ (n - 6)

def increasing_seq (a : ℝ) (n : ℕ) : Prop :=
  a_n a n < a_n a (n + 1)

theorem range_of_a (a : ℝ) :
  (∀ n, increasing_seq a n) ↔ (9 / 4 < a ∧ a < 3) :=
sorry

end range_of_a_l141_14171


namespace fixed_point_f_l141_14145

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (Real.log (2 * x + 1) / Real.log a) + 2

theorem fixed_point_f (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) : f a 0 = 2 :=
by
  sorry

end fixed_point_f_l141_14145


namespace platform_length_is_correct_l141_14129

-- Given Definitions
def length_of_train : ℝ := 300
def time_to_cross_platform : ℝ := 42
def time_to_cross_pole : ℝ := 18

-- Definition to prove
theorem platform_length_is_correct :
  ∃ L : ℝ, L = 400 ∧ (length_of_train + L) / time_to_cross_platform = length_of_train / time_to_cross_pole :=
by
  sorry

end platform_length_is_correct_l141_14129


namespace matt_peanut_revenue_l141_14104

theorem matt_peanut_revenue
    (plantation_length : ℕ)
    (plantation_width : ℕ)
    (peanut_production : ℕ)
    (peanut_to_peanut_butter_rate_peanuts : ℕ)
    (peanut_to_peanut_butter_rate_butter : ℕ)
    (peanut_butter_price_per_kg : ℕ)
    (expected_revenue : ℕ) :
    plantation_length = 500 →
    plantation_width = 500 →
    peanut_production = 50 →
    peanut_to_peanut_butter_rate_peanuts = 20 →
    peanut_to_peanut_butter_rate_butter = 5 →
    peanut_butter_price_per_kg = 10 →
    expected_revenue = 31250 :=
by
  sorry

end matt_peanut_revenue_l141_14104


namespace gear_B_turns_l141_14188

theorem gear_B_turns (teeth_A teeth_B turns_A: ℕ) (h₁: teeth_A = 6) (h₂: teeth_B = 8) (h₃: turns_A = 12) :
(turn_A * teeth_A) / teeth_B = 9 :=
by  sorry

end gear_B_turns_l141_14188


namespace sum_of_two_numbers_l141_14105

theorem sum_of_two_numbers (L S : ℕ) (hL : L = 22) (hExceeds : L = S + 10) : L + S = 34 := by
  sorry

end sum_of_two_numbers_l141_14105


namespace marble_count_l141_14153

theorem marble_count (r g b : ℝ) (h1 : g + b = 9) (h2 : r + b = 7) (h3 : r + g = 5) :
  r + g + b = 10.5 :=
by sorry

end marble_count_l141_14153


namespace maisie_flyers_count_l141_14137

theorem maisie_flyers_count (M : ℕ) (h1 : 71 = 2 * M + 5) : M = 33 :=
by
  sorry

end maisie_flyers_count_l141_14137


namespace work_rates_l141_14178

theorem work_rates (A B : ℝ) (combined_days : ℝ) (b_rate: B = 35) 
(combined_rate: combined_days = 20 / 11):
    A = 700 / 365 :=
by
  have h1 : B = 35 := by sorry
  have h2 : combined_days = 20 / 11 := by sorry
  have : 1/A + 1/B = 11/20 := by sorry
  have : 1/A = 11/20 - 1/B := by sorry
  have : 1/A =  365 / 700:= by sorry
  have : A = 700 / 365 := by sorry
  assumption

end work_rates_l141_14178


namespace revenue_from_full_price_tickets_l141_14157

theorem revenue_from_full_price_tickets (f h p : ℕ) (H1 : f + h = 150) (H2 : f * p + h * (p / 2) = 2450) : 
  f * p = 1150 :=
by 
  sorry

end revenue_from_full_price_tickets_l141_14157


namespace points_on_opposite_sides_l141_14197

-- Definitions and the conditions written to Lean
def satisfies_A (a x y : ℝ) : Prop :=
  5 * a^2 - 6 * a * x - 2 * a * y + 2 * x^2 + 2 * x * y + y^2 = 0

def satisfies_B (a x y : ℝ) : Prop :=
  a^2 * x^2 + a^2 * y^2 - 8 * a^2 * x - 2 * a^3 * y + 12 * a * y + a^4 + 36 = 0

def opposite_sides_of_line (y_A y_B : ℝ) : Prop :=
  (y_A - 1) * (y_B - 1) < 0

theorem points_on_opposite_sides (a : ℝ) (x_A y_A x_B y_B : ℝ) :
  satisfies_A a x_A y_A →
  satisfies_B a x_B y_B →
  -2 > a ∨ (-1 < a ∧ a < 0) ∨ 3 < a →
  opposite_sides_of_line y_A y_B → 
  x_A = 2 * a ∧ y_A = -a ∧ x_B = 4 ∧ y_B = a - 6/a :=
sorry

end points_on_opposite_sides_l141_14197


namespace find_t_l141_14192

theorem find_t (p q r s t : ℤ)
  (h₁ : p - q - r + s - t = -t)
  (h₂ : p - (q - (r - (s - t))) = -4 + t) :
  t = 2 := 
sorry

end find_t_l141_14192


namespace total_pennies_after_addition_l141_14187

def initial_pennies_per_compartment : ℕ := 10
def compartments : ℕ := 20
def added_pennies_per_compartment : ℕ := 15

theorem total_pennies_after_addition :
  (initial_pennies_per_compartment + added_pennies_per_compartment) * compartments = 500 :=
by 
  sorry

end total_pennies_after_addition_l141_14187


namespace maxwells_walking_speed_l141_14150

theorem maxwells_walking_speed 
    (brad_speed : ℕ) 
    (distance_between_homes : ℕ) 
    (maxwell_distance : ℕ)
    (meeting : maxwell_distance = 12)
    (brad_speed_condition : brad_speed = 6)
    (distance_between_homes_condition: distance_between_homes = 36) : 
    (maxwell_distance / (distance_between_homes - maxwell_distance) * brad_speed ) = 3 := by
  sorry

end maxwells_walking_speed_l141_14150


namespace no_solution_l141_14180

theorem no_solution (n : ℕ) (x y k : ℕ) (h1 : n ≥ 1) (h2 : x > 0) (h3 : y > 0) (h4 : k > 1) (h5 : Nat.gcd x y = 1) (h6 : 3^n = x^k + y^k) : False :=
by
  sorry

end no_solution_l141_14180


namespace cid_earnings_l141_14181

theorem cid_earnings :
  let model_a_oil_change_cost := 20
  let model_a_repair_cost := 30
  let model_a_wash_cost := 5
  let model_b_oil_change_cost := 25
  let model_b_repair_cost := 40
  let model_b_wash_cost := 8
  let model_c_oil_change_cost := 30
  let model_c_repair_cost := 50
  let model_c_wash_cost := 10

  let model_a_oil_changes := 5
  let model_a_repairs := 10
  let model_a_washes := 15
  let model_b_oil_changes := 3
  let model_b_repairs := 4
  let model_b_washes := 10
  let model_c_oil_changes := 2
  let model_c_repairs := 6
  let model_c_washes := 5

  let total_earnings := 
      (model_a_oil_change_cost * model_a_oil_changes) +
      (model_a_repair_cost * model_a_repairs) +
      (model_a_wash_cost * model_a_washes) +
      (model_b_oil_change_cost * model_b_oil_changes) +
      (model_b_repair_cost * model_b_repairs) +
      (model_b_wash_cost * model_b_washes) +
      (model_c_oil_change_cost * model_c_oil_changes) +
      (model_c_repair_cost * model_c_repairs) +
      (model_c_wash_cost * model_c_washes)

  total_earnings = 1200 := by
  sorry

end cid_earnings_l141_14181


namespace solve_system_l141_14100

theorem solve_system :
  ∀ (x y : ℝ) (triangle : ℝ), 
  (2 * x - 3 * y = 5) ∧ (x + y = triangle) ∧ (x = 4) →
  (y = 1) ∧ (triangle = 5) :=
by
  -- Skipping the proof steps
  sorry

end solve_system_l141_14100


namespace hyperbola_find_a_b_l141_14124

def hyperbola_conditions (a b : ℝ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ (∀ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1)) ∧
  (∃ e : ℝ, e = 2) ∧ (∃ c : ℝ, c = 4)

theorem hyperbola_find_a_b (a b : ℝ) : hyperbola_conditions a b → a = 2 ∧ b = 2 * Real.sqrt 3 := 
sorry

end hyperbola_find_a_b_l141_14124


namespace determine_k_for_quadratic_eq_l141_14158

theorem determine_k_for_quadratic_eq {k : ℝ} :
  (∀ r s : ℝ, 3 * r^2 + 5 * r + k = 0 ∧ 3 * s^2 + 5 * s + k = 0 →
    (|r + s| = r^2 + s^2)) ↔ k = -10/3 := by
sorry

end determine_k_for_quadratic_eq_l141_14158


namespace arithmetic_seq_sum_l141_14140

theorem arithmetic_seq_sum (a : ℕ → ℝ) (d : ℝ) 
  (h₁ : ∀ n : ℕ, a (n + 1) = a n + d)
  (h₂ : a 3 + a 7 = 37) :
  a 2 + a 4 + a 6 + a 8 = 74 := 
sorry

end arithmetic_seq_sum_l141_14140


namespace mandy_used_nutmeg_l141_14106

theorem mandy_used_nutmeg (x : ℝ) (h1 : 0.67 = x + 0.17) : x = 0.50 :=
  by
  sorry

end mandy_used_nutmeg_l141_14106


namespace number_of_maple_trees_planted_today_l141_14111

-- Define the initial conditions
def initial_maple_trees : ℕ := 2
def poplar_trees : ℕ := 5
def final_maple_trees : ℕ := 11

-- State the main proposition
theorem number_of_maple_trees_planted_today : 
  (final_maple_trees - initial_maple_trees) = 9 := by
  sorry

end number_of_maple_trees_planted_today_l141_14111


namespace find_ordered_pair_l141_14123

-- Definitions based on the conditions
variable (a c : ℝ)
def has_exactly_one_solution :=
  (-6)^2 - 4 * a * c = 0

def sum_is_twelve :=
  a + c = 12

def a_less_than_c :=
  a < c

-- The proof statement
theorem find_ordered_pair
  (h₁ : has_exactly_one_solution a c)
  (h₂ : sum_is_twelve a c)
  (h₃ : a_less_than_c a c) :
  a = 3 ∧ c = 9 := 
sorry

end find_ordered_pair_l141_14123


namespace proof_problem_l141_14175

-- Definition of the condition
def condition (y : ℝ) : Prop := 6 * y^2 + 5 = 2 * y + 10

-- Stating the theorem
theorem proof_problem : ∀ y : ℝ, condition y → (12 * y - 5)^2 = 133 :=
by
  intro y
  intro h
  sorry

end proof_problem_l141_14175


namespace initial_mixture_l141_14147

theorem initial_mixture (M : ℝ) (h1 : 0.20 * M + 20 = 0.36 * (M + 20)) : 
  M = 80 :=
by
  sorry

end initial_mixture_l141_14147


namespace num_solutions_in_interval_l141_14183

theorem num_solutions_in_interval : 
  ∃ n : ℕ, n = 2 ∧ ∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ 2 * Real.pi → 
  2 ^ Real.cos θ = Real.sin θ → n = 2 := 
sorry

end num_solutions_in_interval_l141_14183


namespace smallest_a_divisible_by_65_l141_14121

theorem smallest_a_divisible_by_65 (a : ℤ) 
  (h : ∀ (n : ℤ), (5 * n ^ 13 + 13 * n ^ 5 + 9 * a * n) % 65 = 0) : 
  a = 63 := 
by {
  sorry
}

end smallest_a_divisible_by_65_l141_14121


namespace constant_term_expansion_l141_14159

theorem constant_term_expansion : 
  ∃ r : ℕ, (9 - 3 * r / 2 = 0) ∧ 
  ∀ (x : ℝ) (hx : x ≠ 0), (2 * x - 1 / Real.sqrt x) ^ 9 = 672 := 
by sorry

end constant_term_expansion_l141_14159


namespace ef_length_l141_14113

theorem ef_length (FR RG : ℝ) (cos_ERH : ℝ) (h1 : FR = 12) (h2 : RG = 6) (h3 : cos_ERH = 1 / 5) : EF = 30 :=
by
  sorry

end ef_length_l141_14113


namespace expected_value_m_plus_n_l141_14119

-- Define the main structures and conditions
def spinner_sectors : List ℚ := [-1.25, -1, 0, 1, 1.25]
def initial_value : ℚ := 1

-- Define a function that returns the largest expected value on the paper
noncomputable def expected_largest_written_value (sectors : List ℚ) (initial : ℚ) : ℚ :=
  -- The expected value calculation based on the problem and solution analysis
  11/6  -- This is derived from the correct solution steps not shown here

-- Define the final claim
theorem expected_value_m_plus_n :
  let m := 11
  let n := 6
  expected_largest_written_value spinner_sectors initial_value = 11/6 → m + n = 17 :=
by sorry

end expected_value_m_plus_n_l141_14119


namespace at_least_one_is_one_l141_14120

theorem at_least_one_is_one (a b c : ℝ) 
  (h1 : a + b + c = (1 / a) + (1 / b) + (1 / c)) 
  (h2 : a * b * c = 1) : a = 1 ∨ b = 1 ∨ c = 1 := 
by 
  sorry

end at_least_one_is_one_l141_14120


namespace factorize_x_squared_minus_one_l141_14184

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_x_squared_minus_one_l141_14184


namespace maoming_population_scientific_notation_l141_14108

-- Definitions for conditions
def is_scientific_notation (a : ℝ) (n : ℤ) (x : ℝ) : Prop :=
  x = a * 10^n ∧ 1 ≤ |a| ∧ |a| < 10

-- The main theorem to prove
theorem maoming_population_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), is_scientific_notation a n 6800000 ∧ a = 6.8 ∧ n = 6 :=
sorry

end maoming_population_scientific_notation_l141_14108


namespace cost_per_text_message_for_first_plan_l141_14103

theorem cost_per_text_message_for_first_plan (x : ℝ) : 
  (9 + 60 * x = 60 * 0.40) → (x = 0.25) :=
by
  intro h
  sorry

end cost_per_text_message_for_first_plan_l141_14103


namespace max_non_attacking_rooks_l141_14107

theorem max_non_attacking_rooks (a b : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ 299) (h3 : 1 ≤ b) (h4 : b ≤ 299) :
  ∃ max_rooks : ℕ, max_rooks = 400 :=
  sorry

end max_non_attacking_rooks_l141_14107


namespace sum_of_eggs_is_3712_l141_14176

-- Definitions based on the conditions
def eggs_yesterday : ℕ := 1925
def eggs_fewer_today : ℕ := 138
def eggs_today : ℕ := eggs_yesterday - eggs_fewer_today

-- Theorem stating the equivalence of the sum of eggs
theorem sum_of_eggs_is_3712 : eggs_yesterday + eggs_today = 3712 :=
by
  sorry

end sum_of_eggs_is_3712_l141_14176


namespace product_of_solutions_l141_14190

-- Definitions based on given conditions
def equation (x : ℝ) : Prop := |x| = 3 * (|x| - 2)

-- Statement of the proof problem
theorem product_of_solutions : ∃ x1 x2 : ℝ, equation x1 ∧ equation x2 ∧ x1 * x2 = -9 := by
  sorry

end product_of_solutions_l141_14190


namespace increase_in_average_l141_14131

theorem increase_in_average (s1 s2 s3 s4 s5: ℝ)
  (h1: s1 = 92) (h2: s2 = 86) (h3: s3 = 89) (h4: s4 = 94) (h5: s5 = 91):
  ( ((s1 + s2 + s3 + s4 + s5) / 5) - ((s1 + s2 + s3) / 3) ) = 1.4 :=
by
  sorry

end increase_in_average_l141_14131


namespace area_intersection_A_B_l141_14144

noncomputable def A : Set (Real × Real) := {
  p | ∃ α β : ℝ, p.1 = 2 * Real.sin α + 2 * Real.sin β ∧ p.2 = 2 * Real.cos α + 2 * Real.cos β
}

noncomputable def B : Set (Real × Real) := {
  p | Real.sin (p.1 + p.2) * Real.cos (p.1 + p.2) ≥ 0
}

theorem area_intersection_A_B :
  let intersection := Set.inter A B
  let area : ℝ := 8 * Real.pi
  ∀ (x y : ℝ), (x, y) ∈ intersection → True := sorry

end area_intersection_A_B_l141_14144


namespace best_fit_line_slope_l141_14193

theorem best_fit_line_slope (x1 x2 x3 x4 y1 y2 y3 y4 : ℝ) (d : ℝ) 
  (h1 : x2 - x1 = 2 * d) (h2 : x3 - x2 = 3 * d) (h3 : x4 - x3 = d) : 
  ((y4 - y1) / (x4 - x1)) = (y4 - y1) / (x4 - x1) :=
by
  sorry

end best_fit_line_slope_l141_14193


namespace intersection_empty_set_l141_14117

def M : Set ℝ := { y | ∃ x, x > 0 ∧ y = 2^x }
def N : Set ℝ := { y | ∃ x, y = Real.sqrt (2*x - x^2) }

theorem intersection_empty_set :
  M ∩ N = ∅ :=
by
  sorry

end intersection_empty_set_l141_14117


namespace alpha_beta_roots_l141_14174

theorem alpha_beta_roots (α β : ℝ) (hαβ1 : α^2 + α - 1 = 0) (hαβ2 : β^2 + β - 1 = 0) (h_sum : α + β = -1) :
  α^4 - 3 * β = 5 :=
by
  sorry

end alpha_beta_roots_l141_14174


namespace no_equilateral_triangle_on_integer_lattice_l141_14169

theorem no_equilateral_triangle_on_integer_lattice :
  ∀ (A B C : ℤ × ℤ), 
  A ≠ B → B ≠ C → C ≠ A →
  (dist A B = dist B C ∧ dist B C = dist C A) → 
  false :=
by sorry

end no_equilateral_triangle_on_integer_lattice_l141_14169


namespace initial_quantity_of_A_l141_14136

noncomputable def initial_quantity_of_A_in_can (initial_total_mixture : ℤ) (x : ℤ) := 7 * x

theorem initial_quantity_of_A
  (initial_ratio_A : ℤ) (initial_ratio_B : ℤ) (initial_ratio_C : ℤ)
  (initial_total_mixture : ℤ) (drawn_off_mixture : ℤ) (new_quantity_of_B : ℤ)
  (new_ratio_A : ℤ) (new_ratio_B : ℤ) (new_ratio_C : ℤ)
  (h1 : initial_ratio_A = 7) (h2 : initial_ratio_B = 5) (h3 : initial_ratio_C = 3)
  (h4 : initial_total_mixture = 15 * x)
  (h5 : new_ratio_A = 7) (h6 : new_ratio_B = 9) (h7 : new_ratio_C = 3)
  (h8 : drawn_off_mixture = 18)
  (h9 : new_quantity_of_B = 5 * x - (5 / 15) * 18 + 18)
  (h10 : (7 * x - (7 / 15) * 18) / new_quantity_of_B = 7 / 9) :
  initial_quantity_of_A_in_can initial_total_mixture x = 54 :=
by
  sorry

end initial_quantity_of_A_l141_14136


namespace sum_of_reciprocals_l141_14101

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 8 * x * y) : 
  (1 / x) + (1 / y) = 8 := 
by 
  sorry

end sum_of_reciprocals_l141_14101


namespace area_of_smaller_circle_l141_14173

noncomputable def radius_smaller_circle : ℝ := sorry
noncomputable def radius_larger_circle : ℝ := 3 * radius_smaller_circle

-- Given: PA = AB = 5
def PA : ℝ := 5
def AB : ℝ := 5

-- Final goal: The area of the smaller circle is 5/3 * π
theorem area_of_smaller_circle (r_s : ℝ) (rsq : r_s^2 = 5 / 3) : (π * r_s^2 = 5/3 * π) :=
by
  exact sorry

end area_of_smaller_circle_l141_14173


namespace part1_part2_case1_part2_case2_part2_case3_part3_l141_14152

variable (m : ℝ)
def f (x : ℝ) := (m + 1) * x^2 - (m - 1) * x + (m - 1)

-- Part (1)
theorem part1 (h : ∀ x : ℝ, f m x < 1) : m < (1 - 2 * Real.sqrt 7) / 3 :=
sorry

-- Part (2)
theorem part2_case1 (h : m = -1) : ∀ x, f m x ≥ (m + 1) * x ↔ x ≥ 1 :=
sorry

theorem part2_case2 (h : m > -1) : ∀ x, f m x ≥ (m + 1) * x ↔ x ≤ (m - 1) / (m + 1) ∨ x ≥ 1 :=
sorry

theorem part2_case3 (h : m < -1) : ∀ x, f m x ≥ (m + 1) * x ↔ 1 ≤ x ∧ x ≤ (m - 1) / (m + 1) :=
sorry

-- Part (3)
theorem part3 (h : ∀ x ∈ Set.Icc (-1/2 : ℝ) (1/2), f m x ≥ 0) : m ≥ 1 :=
sorry

end part1_part2_case1_part2_case2_part2_case3_part3_l141_14152


namespace total_amount_is_152_l141_14102

noncomputable def total_amount (p q r s t : ℝ) : ℝ := p + q + r + s + t

noncomputable def p_share (x : ℝ) : ℝ := 2 * x
noncomputable def q_share (x : ℝ) : ℝ := 1.75 * x
noncomputable def r_share (x : ℝ) : ℝ := 1.5 * x
noncomputable def s_share (x : ℝ) : ℝ := 1.25 * x
noncomputable def t_share (x : ℝ) : ℝ := 1.1 * x

theorem total_amount_is_152 (x : ℝ) (h1 : q_share x = 35) :
  total_amount (p_share x) (q_share x) (r_share x) (s_share x) (t_share x) = 152 := by
  sorry

end total_amount_is_152_l141_14102


namespace real_values_of_x_l141_14134

theorem real_values_of_x :
  {x : ℝ | (∃ y, y = (x^2 + 2 * x^3 - 3 * x^4) / (x + 2 * x^2 - 3 * x^3) ∧ y ≥ -1)} =
  {x | -1 ≤ x ∧ x < -1/3 ∨ -1/3 < x ∧ x < 0 ∨ 0 < x ∧ x < 1 ∨ 1 < x} := 
sorry

end real_values_of_x_l141_14134


namespace tan_double_alpha_plus_pi_over_four_sin_cos_fraction_l141_14156

-- Condition: Given tan(α) = 2
variable (α : ℝ) (h₀ : Real.tan α = 2)

-- Statement (1): Prove tan(2α + π/4) = 9
theorem tan_double_alpha_plus_pi_over_four :
  Real.tan (2 * α + Real.pi / 4) = 9 := by
  sorry

-- Statement (2): Prove (6 sin α + cos α) / (3 sin α - 2 cos α) = 13 / 4
theorem sin_cos_fraction :
  (6 * Real.sin α + Real.cos α) / (3 * Real.sin α - 2 * Real.cos α) = 13 / 4 := by
  sorry

end tan_double_alpha_plus_pi_over_four_sin_cos_fraction_l141_14156


namespace radius_of_inscribed_circle_in_quarter_circle_l141_14154

noncomputable def inscribed_circle_radius (R : ℝ) : ℝ :=
  R * (Real.sqrt 2 - 1)

theorem radius_of_inscribed_circle_in_quarter_circle 
  (R : ℝ) (hR : R = 6) : inscribed_circle_radius R = 6 * Real.sqrt 2 - 6 :=
by
  rw [inscribed_circle_radius, hR]
  -- Apply the necessary simplifications and manipulations from the given solution steps here
  sorry

end radius_of_inscribed_circle_in_quarter_circle_l141_14154


namespace aaron_already_had_lids_l141_14185

-- Definitions for conditions
def number_of_boxes : ℕ := 3
def can_lids_per_box : ℕ := 13
def total_can_lids : ℕ := 53
def lids_from_boxes : ℕ := number_of_boxes * can_lids_per_box

-- The statement to be proven
theorem aaron_already_had_lids : total_can_lids - lids_from_boxes = 14 := 
by
  sorry

end aaron_already_had_lids_l141_14185


namespace interval_sum_l141_14196

theorem interval_sum (a b : ℝ) (h : ∀ x,  |3 * x - 80| ≤ |2 * x - 105| ↔ (a ≤ x ∧ x ≤ b)) :
  a + b = 12 :=
sorry

end interval_sum_l141_14196


namespace functional_relationship_minimum_wage_l141_14118

/-- Problem setup and conditions --/
def total_area : ℝ := 1200
def team_A_rate : ℝ := 100
def team_B_rate : ℝ := 50
def team_A_wage : ℝ := 4000
def team_B_wage : ℝ := 3000
def min_days_A : ℝ := 3

/-- Prove Part 1: y as a function of x --/
def y_of_x (x : ℝ) : ℝ := 24 - 2 * x

theorem functional_relationship (x : ℝ) :
  100 * x + 50 * y_of_x x = total_area := by
  sorry

/-- Prove Part 2: Minimum wage calculation --/
def total_wage (a b : ℝ) : ℝ := team_A_wage * a + team_B_wage * b

theorem minimum_wage :
  ∀ (a b : ℝ), 3 ≤ a → a ≤ b → b = 24 - 2 * a → 
  total_wage a b = 56000 → a = 8 ∧ b = 8 := by
  sorry

end functional_relationship_minimum_wage_l141_14118


namespace find_asterisk_value_l141_14170

theorem find_asterisk_value : 
  (∃ x : ℕ, (x / 21) * (x / 189) = 1) → x = 63 :=
by
  intro h
  sorry

end find_asterisk_value_l141_14170


namespace find_stamps_l141_14191

def stamps_problem (x y : ℕ) : Prop :=
  (x + y = 70) ∧ (y = 4 * x + 5)

theorem find_stamps (x y : ℕ) (h : stamps_problem x y) : 
  x = 13 ∧ y = 57 :=
sorry

end find_stamps_l141_14191


namespace roots_abs_gt_4_or_l141_14146

theorem roots_abs_gt_4_or
    (r1 r2 : ℝ)
    (q : ℝ) 
    (h1 : r1 ≠ r2)
    (h2 : r1 + r2 = -q)
    (h3 : r1 * r2 = -10) :
    |r1| > 4 ∨ |r2| > 4 :=
sorry

end roots_abs_gt_4_or_l141_14146


namespace max_value_x2_y2_l141_14195

noncomputable def max_x2_y2 (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x^2 + y ≥ x^3 + y^2) : ℝ := 2

theorem max_value_x2_y2 (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x^2 + y ≥ x^3 + y^2) : 
  x^2 + y^2 ≤ max_x2_y2 x y hx hy h :=
by
  sorry

end max_value_x2_y2_l141_14195


namespace solution_correct_l141_14141

noncomputable def solve_system (a b c : ℝ) : ℝ × ℝ × ℝ :=
  let x := (3 * c - a - b) / 4
  let y := (3 * b - a - c) / 4
  let z := (3 * a - b - c) / 4
  (x, y, z)

theorem solution_correct (a b c : ℝ) (x y z : ℝ) :
  (x + y + 2 * z = a) →
  (x + 2 * y + z = b) →
  (2 * x + y + z = c) →
  (x, y, z) = solve_system a b c :=
by sorry

end solution_correct_l141_14141


namespace tank_full_time_l141_14166

def tank_capacity : ℕ := 900
def fill_rate_A : ℕ := 40
def fill_rate_B : ℕ := 30
def drain_rate_C : ℕ := 20
def cycle_time : ℕ := 3
def net_fill_per_cycle : ℕ := fill_rate_A + fill_rate_B - drain_rate_C

theorem tank_full_time :
  (tank_capacity / net_fill_per_cycle) * cycle_time = 54 :=
by
  sorry

end tank_full_time_l141_14166


namespace fraction_of_buttons_l141_14162

variable (K S M : ℕ)  -- Kendra's buttons, Sue's buttons, Mari's buttons

theorem fraction_of_buttons (H1 : M = 5 * K + 4) 
                            (H2 : S = 6)
                            (H3 : M = 64) :
  S / K = 1 / 2 := by
  sorry

end fraction_of_buttons_l141_14162


namespace triangle_properties_l141_14115

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) (BD : ℝ) (D : ℝ) : 
  (a + c) * Real.sin A = Real.sin A + Real.sin C →
  c^2 + c = b^2 - 1 →
  D = (a + c) / 2 →
  BD = Real.sqrt 3 / 2 →
  B = 2 * Real.pi / 3 ∧ (1 / 2) * a * c * Real.sin B = Real.sqrt 3 / 2 :=
by
  sorry

end triangle_properties_l141_14115


namespace marbles_remaining_l141_14126

theorem marbles_remaining 
  (initial_remaining : ℕ := 400)
  (num_customers : ℕ := 20)
  (marbles_per_customer : ℕ := 15) :
  initial_remaining - (num_customers * marbles_per_customer) = 100 :=
by
  sorry

end marbles_remaining_l141_14126


namespace intersect_in_third_quadrant_l141_14186

theorem intersect_in_third_quadrant (b : ℝ) : (¬ (∃ x y : ℝ, y = 2*x + 1 ∧ y = 3*x + b ∧ x < 0 ∧ y < 0)) ↔ b > 3 / 2 := sorry

end intersect_in_third_quadrant_l141_14186


namespace paper_area_l141_14132

theorem paper_area (L W : ℝ) 
(h1 : 2 * L + W = 34) 
(h2 : L + 2 * W = 38) : 
L * W = 140 := by
  sorry

end paper_area_l141_14132


namespace man_l141_14167

variable (v : ℝ) (speed_with_current : ℝ) (speed_of_current : ℝ)

theorem man's_speed_against_current :
  speed_with_current = 12 ∧ speed_of_current = 2 → v - speed_of_current = 8 :=
by
  sorry

end man_l141_14167


namespace area_of_triangle_PQR_l141_14155

-- Define the problem conditions
def PQ : ℝ := 4
def PR : ℝ := 4
def angle_P : ℝ := 45 -- degrees

-- Define the main problem
theorem area_of_triangle_PQR : 
  (PQ = PR) ∧ (angle_P = 45) ∧ (PR = 4) → 
  ∃ A, A = 8 := 
by
  sorry

end area_of_triangle_PQR_l141_14155


namespace area_of_field_l141_14122

theorem area_of_field (b l : ℝ) (h1 : l = b + 30) (h2 : 2 * (l + b) = 540) : l * b = 18000 := 
by
  sorry

end area_of_field_l141_14122


namespace rainfall_comparison_l141_14133

-- Define the conditions
def rainfall_mondays (n_mondays : ℕ) (rain_monday : ℝ) : ℝ :=
  n_mondays * rain_monday

def rainfall_tuesdays (n_tuesdays : ℕ) (rain_tuesday : ℝ) : ℝ :=
  n_tuesdays * rain_tuesday

def rainfall_difference (total_monday : ℝ) (total_tuesday : ℝ) : ℝ :=
  total_tuesday - total_monday

-- The proof statement
theorem rainfall_comparison :
  rainfall_difference (rainfall_mondays 13 1.75) (rainfall_tuesdays 16 2.65) = 19.65 := by
  sorry

end rainfall_comparison_l141_14133


namespace sheep_count_l141_14165

-- Define the conditions
def TotalAnimals : ℕ := 200
def NumberCows : ℕ := 40
def NumberGoats : ℕ := 104

-- Define the question and its corresponding answer
def NumberSheep : ℕ := TotalAnimals - (NumberCows + NumberGoats)

-- State the theorem
theorem sheep_count : NumberSheep = 56 := by
  -- Skipping the proof
  sorry

end sheep_count_l141_14165


namespace count_perfect_squares_lt_10_pow_9_multiple_36_l141_14127

theorem count_perfect_squares_lt_10_pow_9_multiple_36 : 
  ∃ N : ℕ, ∀ n < 31622, (n % 6 = 0 → n^2 < 10^9 ∧ 36 ∣ n^2 → n ≤ 31620 → N = 5270) :=
by
  sorry

end count_perfect_squares_lt_10_pow_9_multiple_36_l141_14127


namespace range_of_m_tangent_not_parallel_l141_14179

noncomputable def f (x : ℝ) : ℝ := Real.log x
noncomputable def g (x : ℝ) (k : ℝ) : ℝ := (1 / 2) * x^2 - k * x
noncomputable def h (x : ℝ) (m : ℝ) : ℝ := f x + g x (m + (1 / m))
noncomputable def M (x : ℝ) (m : ℝ) : ℝ := f x - g x (m + (1 / m))

theorem range_of_m (m : ℝ) (h_extreme : ∃ x ∈ Set.Ioo 0 2, ∀ y ∈ Set.Ioo 0 2, h y m ≤ h x m) : 
  (0 < m ∧ m ≤ 1 / 2) ∨ (m ≥ 2) :=
  sorry

theorem tangent_not_parallel (x1 x2 x0 : ℝ) (m : ℝ) (h_zeros : M x1 m = 0 ∧ M x2 m = 0 ∧ x1 > x2 ∧ 2 * x0 = x1 + x2) :
  ¬ (∃ l : ℝ, ∀ x : ℝ, M x m = l * (x - x0) + M x0 m ∧ l = 0) :=
  sorry

end range_of_m_tangent_not_parallel_l141_14179


namespace kyle_lift_weight_l141_14130

theorem kyle_lift_weight (this_year_weight last_year_weight : ℕ) 
  (h1 : this_year_weight = 80) 
  (h2 : this_year_weight = 3 * last_year_weight) : 
  (this_year_weight - last_year_weight) = 53 := by
  sorry

end kyle_lift_weight_l141_14130


namespace total_students_correct_l141_14109

def num_first_graders : ℕ := 358
def num_second_graders : ℕ := num_first_graders - 64
def total_students : ℕ := num_first_graders + num_second_graders

theorem total_students_correct : total_students = 652 :=
by
  sorry

end total_students_correct_l141_14109


namespace remainder_when_divided_by_20_l141_14149

theorem remainder_when_divided_by_20
  (a b : ℤ) 
  (h1 : a % 60 = 49)
  (h2 : b % 40 = 29) :
  (a + b) % 20 = 18 :=
by
  sorry

end remainder_when_divided_by_20_l141_14149


namespace vasya_fraction_is_0_4_l141_14168

-- Defining the variables and conditions
variables (a b c d s : ℝ)
axiom cond1 : a = b / 2
axiom cond2 : c = a + d
axiom cond3 : d = s / 10
axiom cond4 : a + b + c + d = s

-- Stating the theorem
theorem vasya_fraction_is_0_4 (a b c d s : ℝ) (h1 : a = b / 2) (h2 : c = a + d) (h3 : d = s / 10) (h4 : a + b + c + d = s) : (b / s) = 0.4 := 
by
  sorry

end vasya_fraction_is_0_4_l141_14168


namespace problem_solution_l141_14110

theorem problem_solution
  (m : ℝ) (n : ℝ)
  (h1 : m = 1 / (Real.sqrt 3 + Real.sqrt 2))
  (h2 : n = 1 / (Real.sqrt 3 - Real.sqrt 2)) :
  (m - 1) * (n - 1) = -2 * Real.sqrt 3 :=
by sorry

end problem_solution_l141_14110


namespace goose_price_remains_affordable_l141_14172

theorem goose_price_remains_affordable :
  ∀ (h v : ℝ),
  h + v = 1 →
  h + (v / 2) = 1 →
  h * 1.2 ≤ 1 :=
by
  intros h v h_eq v_eq
  /- Proof will go here -/
  sorry

end goose_price_remains_affordable_l141_14172


namespace paco_cookie_problem_l141_14128

theorem paco_cookie_problem (x : ℕ) (hx : x + 9 = 18) : x = 9 :=
by sorry

end paco_cookie_problem_l141_14128


namespace original_price_of_book_l141_14148

theorem original_price_of_book (final_price : ℝ) (increase_percentage : ℝ) (original_price : ℝ) 
  (h1 : final_price = 360) (h2 : increase_percentage = 0.20) 
  (h3 : final_price = (1 + increase_percentage) * original_price) : original_price = 300 := 
by
  sorry

end original_price_of_book_l141_14148


namespace minimum_m_l141_14199

/-
  Given that for all 2 ≤ x ≤ 3, 3 ≤ y ≤ 6, the inequality mx^2 - xy + y^2 ≥ 0 always holds,
  prove that the minimum value of the real number m is 0.
-/
theorem minimum_m (m : ℝ) :
  (∀ x y : ℝ, 2 ≤ x ∧ x ≤ 3 → 3 ≤ y ∧ y ≤ 6 → m * x^2 - x * y + y^2 ≥ 0) → m = 0 :=
sorry -- proof to be provided

end minimum_m_l141_14199


namespace line_passes_through_fixed_point_l141_14182

-- Statement to prove that the line always passes through the point (2, 2)
theorem line_passes_through_fixed_point :
  ∀ k : ℝ, ∃ x y : ℝ, 
  (1 + 4 * k) * x - (2 - 3 * k) * y + (2 - 14 * k) = 0 ∧ x = 2 ∧ y = 2 :=
sorry

end line_passes_through_fixed_point_l141_14182


namespace andy_loss_more_likely_than_win_l141_14114

def prob_win_first := 0.30
def prob_lose_first := 0.70

def prob_win_second := 0.50
def prob_lose_second := 0.50

def prob_win_both := prob_win_first * prob_win_second
def prob_lose_both := prob_lose_first * prob_lose_second
def diff_probability := prob_lose_both - prob_win_both
def percentage_more_likely := (diff_probability / prob_win_both) * 100

theorem andy_loss_more_likely_than_win :
  percentage_more_likely = 133.33 := sorry

end andy_loss_more_likely_than_win_l141_14114


namespace melanie_attended_games_l141_14125

theorem melanie_attended_games 
(missed_games total_games attended_games : ℕ) 
(h1 : total_games = 64) 
(h2 : missed_games = 32)
(h3 : attended_games = total_games - missed_games) 
: attended_games = 32 :=
by sorry

end melanie_attended_games_l141_14125


namespace common_ratio_of_gp_l141_14164

variable (r : ℝ)(n : ℕ)

theorem common_ratio_of_gp (h1 : 9 * r ^ (n - 1) = 1/3) 
                           (h2 : 9 * (1 - r ^ n) / (1 - r) = 40 / 3) : 
                           r = 1/3 := 
sorry

end common_ratio_of_gp_l141_14164


namespace Yoongi_score_is_53_l141_14161

-- Define the scores of the three students
variables (score_Yoongi score_Eunji score_Yuna : ℕ)

-- Define the conditions given in the problem
axiom Yoongi_Eunji : score_Eunji = score_Yoongi - 25
axiom Eunji_Yuna  : score_Yuna = score_Eunji - 20
axiom Yuna_score  : score_Yuna = 8

theorem Yoongi_score_is_53 : score_Yoongi = 53 := by
  sorry

end Yoongi_score_is_53_l141_14161


namespace license_plate_difference_l141_14112

theorem license_plate_difference :
  (26^4 * 10^3 - 26^5 * 10^2 = -731161600) :=
sorry

end license_plate_difference_l141_14112


namespace necessary_not_sufficient_for_circle_l141_14189

theorem necessary_not_sufficient_for_circle (a : ℝ) :
  (a ≤ 2 → (x^2 + y^2 - 2*x + 2*y + a = 0 → ∃ r : ℝ, r > 0)) ∧
  (a ≤ 2 ∧ ∃ b, b < 2 → a = b) := sorry

end necessary_not_sufficient_for_circle_l141_14189


namespace count_squares_with_dot_l141_14135

theorem count_squares_with_dot (n : ℕ) (dot_center : (n = 5)) :
  n = 5 → ∃ k, k = 19 :=
by sorry

end count_squares_with_dot_l141_14135


namespace interest_equality_l141_14177

theorem interest_equality (total_sum : ℝ) (part1 : ℝ) (part2 : ℝ) (rate1 : ℝ) (time1 : ℝ) (rate2 : ℝ) (n : ℝ) :
  total_sum = 2730 ∧ part1 = 1050 ∧ part2 = 1680 ∧
  rate1 = 3 ∧ time1 = 8 ∧ rate2 = 5 ∧ part1 * rate1 * time1 = part2 * rate2 * n →
  n = 3 :=
by
  sorry

end interest_equality_l141_14177


namespace water_fraction_final_l141_14194

noncomputable def initial_water_volume : ℚ := 25
noncomputable def first_removal_water : ℚ := 5
noncomputable def first_add_antifreeze : ℚ := 5
noncomputable def first_water_fraction : ℚ := (initial_water_volume - first_removal_water) / initial_water_volume

noncomputable def second_removal_fraction : ℚ := 5 / initial_water_volume
noncomputable def second_water_fraction : ℚ := (initial_water_volume - first_removal_water - second_removal_fraction * (initial_water_volume - first_removal_water)) / initial_water_volume

noncomputable def third_removal_fraction : ℚ := 5 / initial_water_volume
noncomputable def third_water_fraction := (second_water_fraction * (initial_water_volume - 5) + 2) / initial_water_volume

theorem water_fraction_final :
  third_water_fraction = 14.8 / 25 := sorry

end water_fraction_final_l141_14194


namespace orchids_cut_l141_14139

-- Define initial and final number of orchids in the vase
def initialOrchids : ℕ := 2
def finalOrchids : ℕ := 21

-- Formulate the claim to prove the number of orchids Jessica cut
theorem orchids_cut : finalOrchids - initialOrchids = 19 := by
  sorry

end orchids_cut_l141_14139


namespace geometric_sequence_fourth_term_l141_14142

/-- In a geometric sequence with common ratio 2, where the sequence is denoted as {a_n},
and it is given that a_1 * a_3 = 6 * a_2, prove that a_4 = 24. -/
theorem geometric_sequence_fourth_term (a : ℕ → ℝ) (h : ∀ n, a (n + 1) = 2 * a n)
  (h1 : a 1 * a 3 = 6 * a 2) : a 4 = 24 :=
sorry

end geometric_sequence_fourth_term_l141_14142


namespace trisha_walked_distance_l141_14163

theorem trisha_walked_distance :
  ∃ x : ℝ, (x + x + 0.67 = 0.89) ∧ (x = 0.11) :=
by sorry

end trisha_walked_distance_l141_14163


namespace symmetric_point_origin_l141_14116

def symmetric_point (p: ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, -p.2)

theorem symmetric_point_origin : 
  (symmetric_point (3, -2)) = (-3, 2) :=
by
  sorry

end symmetric_point_origin_l141_14116


namespace marble_total_weight_l141_14151

theorem marble_total_weight :
  0.3333333333333333 + 0.3333333333333333 + 0.08333333333333333 + 0.21666666666666667 + 0.4583333333333333 + 0.12777777777777778 = 1.5527777777777777 :=
by
  sorry

end marble_total_weight_l141_14151


namespace volume_of_cube_l141_14198

theorem volume_of_cube (a : ℕ) (h : ((a - 2) * a * (a + 2)) = a^3 - 16) : a^3 = 64 :=
sorry

end volume_of_cube_l141_14198

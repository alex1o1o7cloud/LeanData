import Mathlib

namespace max_value_of_f_l21_2193

noncomputable def f (x : ℝ) : ℝ := Real.sin (Real.cos x) + Real.cos (Real.sin x)

theorem max_value_of_f : ∀ x : ℝ, f x ≤ Real.sin 1 + 1 ∧ (f 0 = Real.sin 1 + 1) :=
by
  intro x
  sorry

end max_value_of_f_l21_2193


namespace solve_for_A_l21_2118

def hash (A B : ℝ) : ℝ := A^2 + B^2

theorem solve_for_A (A : ℝ) (h : hash A 7 = 200) : A = Real.sqrt 151 :=
by
  sorry

end solve_for_A_l21_2118


namespace find_f2_l21_2123

def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ :=
  x^5 + a * x^3 + b * x + 1

theorem find_f2 (a b : ℝ) (h : f (-2) a b = 10) : f 2 a b = -8 :=
by
  sorry

end find_f2_l21_2123


namespace A_can_give_C_start_l21_2170

def canGiveStart (total_distance start_A_B start_B_C start_A_C : ℝ) :=
  (total_distance - start_A_B) / total_distance * (total_distance - start_B_C) / total_distance = 
  (total_distance - start_A_C) / total_distance

theorem A_can_give_C_start :
  canGiveStart 1000 70 139.7849462365591 200 :=
by
  sorry

end A_can_give_C_start_l21_2170


namespace totalKidsInLawrenceCounty_l21_2191

-- Constants representing the number of kids in each category
def kidsGoToCamp : ℕ := 629424
def kidsStayHome : ℕ := 268627

-- Statement of the total number of kids in Lawrence county
theorem totalKidsInLawrenceCounty : kidsGoToCamp + kidsStayHome = 898051 := by
  sorry

end totalKidsInLawrenceCounty_l21_2191


namespace symmetric_scanning_codes_count_l21_2164

theorem symmetric_scanning_codes_count :
  let grid_size := 5
  let total_squares := grid_size * grid_size
  let symmetry_classes := 5 -- Derived from classification in the solution
  let possible_combinations := 2 ^ symmetry_classes
  let invalid_combinations := 2 -- All black or all white grid
  total_squares = 25 
  ∧ (possible_combinations - invalid_combinations) = 30 :=
by sorry

end symmetric_scanning_codes_count_l21_2164


namespace exp_value_l21_2100

theorem exp_value (a : ℝ) (m n : ℕ) (h1 : a^m = 2) (h2 : a^n = 3) : a^(m + 2 * n) = 18 := 
by
  sorry

end exp_value_l21_2100


namespace power_mod_result_l21_2131

theorem power_mod_result :
  9^1002 % 50 = 1 := by
  sorry

end power_mod_result_l21_2131


namespace find_f2_l21_2126

theorem find_f2 (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x + 1) = x^2) : f 2 = 1 := 
by
  sorry

end find_f2_l21_2126


namespace cooper_savings_l21_2194

theorem cooper_savings :
  let daily_savings := 34
  let days_in_year := 365
  daily_savings * days_in_year = 12410 :=
by
  sorry

end cooper_savings_l21_2194


namespace convert_polar_to_rectangular_l21_2136

noncomputable def polarToRectangular (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

theorem convert_polar_to_rectangular :
  polarToRectangular 8 (7 * Real.pi / 6) = (-4 * Real.sqrt 3, -4) :=
by
  sorry

end convert_polar_to_rectangular_l21_2136


namespace solve_trig_problem_l21_2195

open Real

theorem solve_trig_problem (x : ℝ) (h1 : 0 ≤ x) (h2 : x < 2 * π) (h3 : sin x + cos x = 1) :
  x = 0 ∨ x = π / 2 := sorry

end solve_trig_problem_l21_2195


namespace correct_calculation_incorrect_calculation_A_incorrect_calculation_B_incorrect_calculation_D_l21_2177

variable {a b : ℝ}

theorem correct_calculation : a ^ 3 * a = a ^ 4 := 
by
  sorry

theorem incorrect_calculation_A : a ^ 3 + a ^ 3 ≠ 2 * a ^ 6 := 
by
  sorry

theorem incorrect_calculation_B : (a ^ 3) ^ 3 ≠ a ^ 6 :=
by
  sorry

theorem incorrect_calculation_D : (a - b) ^ 2 ≠ a ^ 2 - b ^ 2 :=
by
  sorry

end correct_calculation_incorrect_calculation_A_incorrect_calculation_B_incorrect_calculation_D_l21_2177


namespace original_selling_price_l21_2119

theorem original_selling_price (P : ℝ) (S : ℝ) (h1 : S = 1.10 * P) (h2 : 1.17 * P = 1.10 * P + 35) : S = 550 := 
by
  sorry

end original_selling_price_l21_2119


namespace rational_linear_independent_sqrt_prime_l21_2133

theorem rational_linear_independent_sqrt_prime (p : ℕ) (hp : Nat.Prime p) (m n m1 n1 : ℚ) :
  m + n * Real.sqrt p = m1 + n1 * Real.sqrt p → m = m1 ∧ n = n1 :=
sorry

end rational_linear_independent_sqrt_prime_l21_2133


namespace complex_square_eq_l21_2159

variables {a b : ℝ} {i : ℂ}

theorem complex_square_eq :
  a + i = 2 - b * i → (a + b * i) ^ 2 = 3 - 4 * i :=
by sorry

end complex_square_eq_l21_2159


namespace ratio_of_amount_spent_on_movies_to_weekly_allowance_l21_2147

-- Define weekly allowance
def weekly_allowance : ℕ := 10

-- Define final amount after all transactions
def final_amount : ℕ := 11

-- Define earnings from washing the car
def earnings : ℕ := 6

-- Define amount left before washing the car
def amount_left_before_wash : ℕ := final_amount - earnings

-- Define amount spent on movies
def amount_spent_on_movies : ℕ := weekly_allowance - amount_left_before_wash

-- Define the ratio function
def ratio (a b : ℕ) : ℚ := a / b

-- Prove the required ratio
theorem ratio_of_amount_spent_on_movies_to_weekly_allowance :
  ratio amount_spent_on_movies weekly_allowance = 1 / 2 :=
by
  sorry

end ratio_of_amount_spent_on_movies_to_weekly_allowance_l21_2147


namespace rolls_to_neighbor_l21_2128

theorem rolls_to_neighbor (total_needed rolls_to_grandmother rolls_to_uncle rolls_needed : ℕ) (h1 : total_needed = 45) (h2 : rolls_to_grandmother = 1) (h3 : rolls_to_uncle = 10) (h4 : rolls_needed = 28) :
  total_needed - rolls_needed - (rolls_to_grandmother + rolls_to_uncle) = 6 := by
  sorry

end rolls_to_neighbor_l21_2128


namespace intersection_points_l21_2151

theorem intersection_points :
  {p : ℝ × ℝ |
    (∃ x : ℝ, p = (x, 3*x^2 - 4*x + 2) ∧ p = (x, x^3 - 2*x^2 + 5*x - 1))} =
  {(1, 1), (3, 17)} :=
  sorry

end intersection_points_l21_2151


namespace articles_profit_l21_2130

variable {C S : ℝ}

theorem articles_profit (h1 : 20 * C = x * S) (h2 : S = 1.25 * C) : x = 16 :=
by
  sorry

end articles_profit_l21_2130


namespace max_flow_increase_proof_l21_2117

noncomputable def max_flow_increase : ℕ :=
  sorry

theorem max_flow_increase_proof
  (initial_pipes_AB: ℕ) (initial_pipes_BC: ℕ) (flow_increase_per_pipes_swap: ℕ)
  (swap_increase: initial_pipes_AB = 10)
  (swap_increase_2: initial_pipes_BC = 10)
  (flow_increment: flow_increase_per_pipes_swap = 30) : 
  max_flow_increase = 150 :=
  sorry

end max_flow_increase_proof_l21_2117


namespace infinite_series_sum_l21_2180

theorem infinite_series_sum (a r : ℝ) (h₀ : -1 < r) (h₁ : r < 1) :
    (∑' n, if (n % 2 = 0) then a * r^(n/2) else a^2 * r^((n+1)/2)) = (a * (1 + a * r))/(1 - r^2) :=
by
  sorry

end infinite_series_sum_l21_2180


namespace no_rearrangement_of_power_of_two_l21_2158

theorem no_rearrangement_of_power_of_two (k n : ℕ) (hk : k > 3) (hn : n > k) : 
  ∀ m : ℕ, 
    (m.toDigits = (2^k).toDigits → m ≠ 2^n) :=
by
  sorry

end no_rearrangement_of_power_of_two_l21_2158


namespace volleyball_team_l21_2134

theorem volleyball_team :
  let total_combinations := (Nat.choose 15 6)
  let without_triplets := (Nat.choose 12 6)
  total_combinations - without_triplets = 4081 :=
by
  -- Definitions based on the problem conditions
  let team_size := 15
  let starters := 6
  let triplets := 3
  let total_combinations := Nat.choose team_size starters
  let without_triplets := Nat.choose (team_size - triplets) starters
  -- Identify the proof goal
  have h : total_combinations - without_triplets = 4081 := sorry
  exact h

end volleyball_team_l21_2134


namespace infinite_series_sum_l21_2140

theorem infinite_series_sum :
  (∑' n : ℕ, if n = 0 then 0 else (3^n / (1 + 3^n + 3^(n+1) + 3^(2*n+1)))) = 1 / 4 :=
by
  sorry

end infinite_series_sum_l21_2140


namespace find_number_l21_2145

-- Definitions used in the given problem conditions
def condition (x : ℝ) : Prop := (3.242 * x) / 100 = 0.04863

-- Statement of the problem
theorem find_number (x : ℝ) (h : condition x) : x = 1.5 :=
by
  sorry
 
end find_number_l21_2145


namespace divisible_digit_B_l21_2182

-- Define the digit type as natural numbers within the range 0 to 9.
def digit := {n : ℕ // n <= 9}

-- Define what it means for a number to be even.
def even (n : ℕ) : Prop := ∃ k, n = 2 * k

-- Define what it means for a number to be divisible by 3.
def divisible_by_3 (n : ℕ) : Prop := ∃ k, n = 3 * k

-- Define our problem in Lean as properties of the digit B.
theorem divisible_digit_B (B : digit) (h_even : even B.1) (h_div_by_3 : divisible_by_3 (14 + B.1)) : B.1 = 4 :=
sorry

end divisible_digit_B_l21_2182


namespace problem_statement_l21_2167

theorem problem_statement : 1103^2 - 1097^2 - 1101^2 + 1099^2 = 8800 := by
  sorry

end problem_statement_l21_2167


namespace num_even_three_digit_numbers_with_sum_of_tens_and_units_10_l21_2181

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def is_even (n : ℕ) : Prop :=
  n % 2 = 0

def sum_of_tens_and_units_is_ten (n : ℕ) : Prop :=
  (n / 10 % 10) + (n % 10) = 10

theorem num_even_three_digit_numbers_with_sum_of_tens_and_units_10 : 
  ∃! (N : ℕ), (N = 36) ∧ 
               (∀ n : ℕ, is_three_digit n → is_even n → sum_of_tens_and_units_is_ten n →
                         n = 36) := 
sorry

end num_even_three_digit_numbers_with_sum_of_tens_and_units_10_l21_2181


namespace total_players_l21_2190

def num_teams : Nat := 35
def players_per_team : Nat := 23

theorem total_players :
  num_teams * players_per_team = 805 :=
by
  sorry

end total_players_l21_2190


namespace max_value_xyz_l21_2162

theorem max_value_xyz 
  (x y z : ℝ) 
  (h_nonneg : x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0) 
  (h_sum : x + y + z = 3) : 
  ∃ M, M = 243 ∧ (x + y^4 + z^5) ≤ M := 
  by sorry

end max_value_xyz_l21_2162


namespace smallest_y_value_l21_2132

theorem smallest_y_value (y : ℝ) : 3 * y ^ 2 + 33 * y - 105 = y * (y + 16) → y = -21 / 2 ∨ y = 5 := sorry

end smallest_y_value_l21_2132


namespace true_propositions_count_l21_2107

theorem true_propositions_count {a b c : ℝ} (h : a ≤ b) : 
  (if (c^2 ≥ 0 ∧ a * c^2 ≤ b * c^2) then 1 else 0) +
  (if (c^2 ≥ 0 ∧ a * c^2 > b * c^2) then 1 else 0) +
  (if (c^2 ≥ 0 ∧ ¬(a * c^2 ≤ b * c^2) → ¬(a ≤ b)) then 1 else 0) +
  (if (c^2 ≥ 0 ∧ ¬(a ≤ b) → ¬(a * c^2 ≤ b * c^2)) then 1 else 0) = 2 :=
sorry

end true_propositions_count_l21_2107


namespace exam_max_incorrect_answers_l21_2198

theorem exam_max_incorrect_answers :
  ∀ (c w b : ℕ),
  (c + w + b = 30) →
  (4 * c - w ≥ 85) → 
  (c ≥ 22) →
  (w ≤ 3) :=
by
  intros c w b h1 h2 h3
  sorry

end exam_max_incorrect_answers_l21_2198


namespace average_daily_visitors_l21_2150

theorem average_daily_visitors
    (avg_sun : ℕ)
    (avg_other : ℕ)
    (days : ℕ)
    (starts_sun : Bool)
    (H1 : avg_sun = 630)
    (H2 : avg_other = 240)
    (H3 : days = 30)
    (H4 : starts_sun = true) :
    (5 * avg_sun + 25 * avg_other) / days = 305 :=
by
  sorry

end average_daily_visitors_l21_2150


namespace algebraic_expression_value_l21_2127

theorem algebraic_expression_value (p q : ℝ)
  (h : p * 3^3 + q * 3 + 3 = 2005) :
  p * (-3)^3 + q * (-3) + 3 = -1999 :=
by
   sorry

end algebraic_expression_value_l21_2127


namespace spherical_triangle_area_correct_l21_2178

noncomputable def spherical_triangle_area (R α β γ : ℝ) : ℝ :=
  R^2 * (α + β + γ - Real.pi)

theorem spherical_triangle_area_correct (R α β γ : ℝ) :
  spherical_triangle_area R α β γ = R^2 * (α + β + γ - Real.pi) := by
  sorry

end spherical_triangle_area_correct_l21_2178


namespace find_n_l21_2192

variable (a b c n : ℤ)
variable (h1 : a + b + c = 100)
variable (h2 : a + b / 2 = 40)

theorem find_n : n = a - c := by
  sorry

end find_n_l21_2192


namespace rectangle_area_ratio_is_three_l21_2199

variables {a b : ℝ}

-- Rectangle ABCD with midpoint F on CD, BC = 3 * BE
def rectangle_midpoint_condition (CD_length : ℝ) (BC_length : ℝ) (BE_length : ℝ) (F_midpoint : Prop) :=
  F_midpoint ∧ BC_length = 3 * BE_length

-- Areas and the ratio
def area_rectangle (CD_length BC_length : ℝ) : ℝ :=
  CD_length * BC_length

def area_shaded (a b : ℝ) : ℝ :=
  2 * a * b

theorem rectangle_area_ratio_is_three (h : rectangle_midpoint_condition (2 * a) (3 * b) b (F_midpoint := True)) :
  area_rectangle (2 * a) (3 * b) = 3 * area_shaded a b :=
by
  unfold rectangle_midpoint_condition at h
  unfold area_rectangle area_shaded
  rw [←mul_assoc, ←mul_assoc]
  sorry

end rectangle_area_ratio_is_three_l21_2199


namespace anagrams_without_three_consecutive_identical_l21_2121

theorem anagrams_without_three_consecutive_identical :
  let total_anagrams := 100800
  let anagrams_with_three_A := 6720
  let anagrams_with_three_B := 6720
  let anagrams_with_three_A_and_B := 720
  let valid_anagrams := total_anagrams - anagrams_with_three_A - anagrams_with_three_B + anagrams_with_three_A_and_B
  valid_anagrams = 88080 := by
  sorry

end anagrams_without_three_consecutive_identical_l21_2121


namespace salary_increase_after_five_years_l21_2183

theorem salary_increase_after_five_years (S : ℝ) : 
  let final_salary := S * (1.12)^5
  let increase := final_salary - S
  let percent_increase := (increase / S) * 100
  percent_increase = 76.23 :=
by
  let final_salary := S * (1.12)^5
  let increase := final_salary - S
  let percent_increase := (increase / S) * 100
  sorry

end salary_increase_after_five_years_l21_2183


namespace average_age_of_both_teams_l21_2103

theorem average_age_of_both_teams (n_men : ℕ) (age_men : ℕ) (n_women : ℕ) (age_women : ℕ) :
  n_men = 8 → age_men = 35 → n_women = 6 → age_women = 30 → 
  (8 * 35 + 6 * 30) / (8 + 6) = 32.857 := 
by
  intros h1 h2 h3 h4
  -- Proof is omitted
  sorry

end average_age_of_both_teams_l21_2103


namespace probability_same_color_opposite_feet_l21_2129

/-- Define the initial conditions: number of pairs of each color. -/
def num_black_pairs : ℕ := 8
def num_brown_pairs : ℕ := 4
def num_gray_pairs : ℕ := 3
def num_red_pairs : ℕ := 1

/-- The total number of shoes. -/
def total_shoes : ℕ := 2 * (num_black_pairs + num_brown_pairs + num_gray_pairs + num_red_pairs)

theorem probability_same_color_opposite_feet :
  ((num_black_pairs * (num_black_pairs - 1)) + 
   (num_brown_pairs * (num_brown_pairs - 1)) + 
   (num_gray_pairs * (num_gray_pairs - 1)) + 
   (num_red_pairs * (num_red_pairs - 1))) * 2 / (total_shoes * (total_shoes - 1)) = 45 / 248 :=
by sorry

end probability_same_color_opposite_feet_l21_2129


namespace smallest_integer_solution_l21_2179

open Int

theorem smallest_integer_solution :
  ∃ x : ℤ, (⌊ (x : ℚ) / 8 ⌋ - ⌊ (x : ℚ) / 40 ⌋ + ⌊ (x : ℚ) / 240 ⌋ = 210) ∧ x = 2016 :=
by
  sorry

end smallest_integer_solution_l21_2179


namespace non_negative_combined_quadratic_l21_2153

theorem non_negative_combined_quadratic (a b c A B C : ℝ) (h1 : a ≥ 0) (h2 : b^2 ≤ a * c) (h3 : A ≥ 0) (h4 : B^2 ≤ A * C) :
  ∀ x : ℝ, a * A * x^2 + 2 * b * B * x + c * C ≥ 0 :=
by
  sorry

end non_negative_combined_quadratic_l21_2153


namespace area_of_306090_triangle_l21_2141

-- Conditions
def is_306090_triangle (a b c : ℝ) : Prop :=
  a / b = 1 / Real.sqrt 3 ∧ a / c = 1 / 2

-- Given values
def hypotenuse : ℝ := 6

-- To prove
theorem area_of_306090_triangle :
  ∃ (a b c : ℝ), is_306090_triangle a b c ∧ c = hypotenuse ∧ (1 / 2) * a * b = (9 * Real.sqrt 3) / 2 :=
by
  sorry

end area_of_306090_triangle_l21_2141


namespace bottles_from_B_l21_2157

-- Definitions for the bottles from each shop and the total number of bottles Don can buy
def bottles_from_A : Nat := 150
def bottles_from_C : Nat := 220
def total_bottles : Nat := 550

-- Lean statement to prove that the number of bottles Don buys from Shop B is 180
theorem bottles_from_B :
  total_bottles - (bottles_from_A + bottles_from_C) = 180 := 
by
  sorry

end bottles_from_B_l21_2157


namespace range_of_a_l21_2125

noncomputable def f (x a : ℝ) : ℝ := x^3 - x^2 - 2 * a

theorem range_of_a (a : ℝ) :
  (∃ (x₀ : ℝ), x₀ ≤ a ∧ f x₀ a ≥ 0) ↔ (a ∈ Set.Icc (-1 : ℝ) 0 ∪ Set.Ici 2) := by
  sorry

end range_of_a_l21_2125


namespace relay_team_average_time_l21_2110

theorem relay_team_average_time :
  let d1 := 200
  let t1 := 38
  let d2 := 300
  let t2 := 56
  let d3 := 250
  let t3 := 47
  let d4 := 400
  let t4 := 80
  let total_distance := d1 + d2 + d3 + d4
  let total_time := t1 + t2 + t3 + t4
  let average_time_per_meter := total_time / total_distance
  average_time_per_meter = 0.1922 := by
  sorry

end relay_team_average_time_l21_2110


namespace smallest_value_a_b_l21_2172

theorem smallest_value_a_b (a b : ℕ) (h : 2^6 * 3^9 = a^b) : a > 0 ∧ b > 0 ∧ (a + b = 111) :=
by
  sorry

end smallest_value_a_b_l21_2172


namespace mike_chocolate_squares_l21_2105

theorem mike_chocolate_squares (M : ℕ) (h1 : 65 = 3 * M + 5) : M = 20 :=
by {
  -- proof of the theorem (not included as per instructions)
  sorry
}

end mike_chocolate_squares_l21_2105


namespace necessary_and_sufficient_condition_for_extreme_value_l21_2189

-- Defining the function f(x) = ax^3 + x + 1
def f (a x : ℝ) : ℝ := a * x^3 + x + 1

-- Defining the condition for f to have an extreme value
def has_extreme_value (a : ℝ) : Prop := ∃ x : ℝ, deriv (f a) x = 0

-- Stating the problem
theorem necessary_and_sufficient_condition_for_extreme_value (a : ℝ) :
  has_extreme_value a ↔ a < 0 := by
  sorry

end necessary_and_sufficient_condition_for_extreme_value_l21_2189


namespace divisors_form_60k_l21_2171

-- Define the conditions in Lean
def is_positive_divisor (n d : ℕ) : Prop := d > 0 ∧ n % d = 0

def satisfies_conditions (n a b c : ℕ) : Prop :=
  is_positive_divisor n a ∧
  is_positive_divisor n b ∧
  is_positive_divisor n c ∧
  a > b ∧ b > c ∧
  is_positive_divisor n (a^2 - b^2) ∧
  is_positive_divisor n (b^2 - c^2) ∧
  is_positive_divisor n (a^2 - c^2)

-- State the theorem to be proven in Lean
theorem divisors_form_60k (n : ℕ) (a b c : ℕ) (h1 : satisfies_conditions n a b c) : 
  ∃ k : ℕ, n = 60 * k :=
sorry

end divisors_form_60k_l21_2171


namespace original_number_l21_2160

theorem original_number (x y : ℝ) (h1 : 10 * x + 22 * y = 780) (h2 : y = 37.66666666666667) : 
  x + y = 32.7 := 
sorry

end original_number_l21_2160


namespace book_cost_l21_2113

theorem book_cost (b : ℝ) : (11 * b < 15) ∧ (12 * b > 16.20) → b = 1.36 :=
by
  intros h
  sorry

end book_cost_l21_2113


namespace find_third_term_l21_2111

theorem find_third_term :
  ∃ (a : ℕ → ℝ), a 0 = 5 ∧ a 4 = 2025 ∧ (∀ n, a (n + 1) = a n * r) ∧ a 2 = 225 :=
by
  sorry

end find_third_term_l21_2111


namespace ratio_of_sums_l21_2108

variable {α : Type*} [LinearOrderedField α] 

variable (a : ℕ → α) (S : ℕ → α)
variable (a1 d : α)

def isArithmeticSequence (a : ℕ → α) : Prop :=
  ∃ a1 d, ∀ n, a n = a1 + n * d

def sumArithmeticSequence (a : α) (d : α) (n : ℕ) : α :=
  n / 2 * (2 * a + (n - 1) * d)

theorem ratio_of_sums (h_arith : isArithmeticSequence a) (h_S : ∀ n, S n = sumArithmeticSequence a1 d n)
  (h_a5_5a3 : a 5 = 5 * a 3) : S 9 / S 5 = 9 := by sorry

end ratio_of_sums_l21_2108


namespace polynomial_coef_sum_l21_2112

theorem polynomial_coef_sum :
  ∃ (a b c d : ℝ), (∀ x : ℝ, (4 * x^2 - 6 * x + 3) * (8 - 3 * x) = a * x^3 + b * x^2 + c * x + d) ∧ (8 * a + 4 * b + 2 * c + d = 14) :=
by
  sorry

end polynomial_coef_sum_l21_2112


namespace pages_per_day_l21_2174

-- Define the given conditions
def total_pages : ℕ := 957
def total_days : ℕ := 47

-- State the theorem based on the conditions and the required proof
theorem pages_per_day (p : ℕ) (d : ℕ) (h1 : p = total_pages) (h2 : d = total_days) :
  p / d = 20 := by
  sorry

end pages_per_day_l21_2174


namespace scientific_notation_of_393000_l21_2166

theorem scientific_notation_of_393000 : 
  ∃ (a : ℝ) (n : ℤ), a = 3.93 ∧ n = 5 ∧ (393000 = a * 10^n) := 
by
  use 3.93
  use 5
  sorry

end scientific_notation_of_393000_l21_2166


namespace find_x3_y3_l21_2138

theorem find_x3_y3 (x y : ℝ) (h1 : x + y = 6) (h2 : x^2 + y^2 = 18) : x^3 + y^3 = 54 := 
by 
  sorry

end find_x3_y3_l21_2138


namespace find_the_number_l21_2187

theorem find_the_number (x : ℝ) (h : 8 * x + 64 = 336) : x = 34 :=
by
  sorry

end find_the_number_l21_2187


namespace distance_between_consecutive_trees_l21_2185

-- Define the conditions as separate definitions
def num_trees : ℕ := 57
def yard_length : ℝ := 720
def spaces_between_trees := num_trees - 1

-- Define the target statement to prove
theorem distance_between_consecutive_trees :
  yard_length / spaces_between_trees = 12.857142857 := sorry

end distance_between_consecutive_trees_l21_2185


namespace highest_wave_height_l21_2101

-- Definitions of surfboard length and shortest wave conditions
def surfboard_length : ℕ := 7
def shortest_wave_height (H : ℕ) : ℕ := H + 4

-- Lean statement to be proved
theorem highest_wave_height (H : ℕ) (condition1 : H + 4 = surfboard_length + 3) : 
  4 * H + 2 = 26 :=
sorry

end highest_wave_height_l21_2101


namespace prove_solution_l21_2156

noncomputable def problem_statement : Prop := ∀ x : ℝ, (16 : ℝ)^(2 * x - 3) = (4 : ℝ)^(3 - x) → x = 9 / 5

theorem prove_solution : problem_statement :=
by
  intro x h
  -- The proof would go here
  sorry

end prove_solution_l21_2156


namespace population_approx_10000_2090_l21_2143

def population (initial_population : ℕ) (years : ℕ) : ℕ :=
  initial_population * 2 ^ (years / 20)

theorem population_approx_10000_2090 :
  ∃ y, y = 2090 ∧ population 500 (2090 - 2010) = 500 * 2 ^ (80 / 20) :=
by
  sorry

end population_approx_10000_2090_l21_2143


namespace total_slices_sold_l21_2135

theorem total_slices_sold (sold_yesterday served_today : ℕ) (h1 : sold_yesterday = 5) (h2 : served_today = 2) :
  sold_yesterday + served_today = 7 :=
by
  -- Proof skipped
  exact sorry

end total_slices_sold_l21_2135


namespace vectors_parallel_opposite_directions_l21_2196

theorem vectors_parallel_opposite_directions
  (a b : ℝ × ℝ)
  (h₁ : a = (-1, 2))
  (h₂ : b = (2, -4)) :
  b = (-2 : ℝ) • a ∧ b = -2 • a :=
by
  sorry

end vectors_parallel_opposite_directions_l21_2196


namespace ratio_horizontal_to_checkered_l21_2124

/--
In a cafeteria, 7 people are wearing checkered shirts, while the rest are wearing vertical stripes
and horizontal stripes. There are 40 people in total, and 5 of them are wearing vertical stripes.
What is the ratio of the number of people wearing horizontal stripes to the number of people wearing
checkered shirts?
-/
theorem ratio_horizontal_to_checkered
  (total_people : ℕ)
  (checkered_people : ℕ)
  (vertical_people : ℕ)
  (horizontal_people : ℕ)
  (ratio : ℕ)
  (h_total : total_people = 40)
  (h_checkered : checkered_people = 7)
  (h_vertical : vertical_people = 5)
  (h_horizontal : horizontal_people = total_people - checkered_people - vertical_people)
  (h_ratio : ratio = horizontal_people / checkered_people) :
  ratio = 4 :=
by
  sorry

end ratio_horizontal_to_checkered_l21_2124


namespace range_of_a_l21_2176

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 2 * |x - 1| + |x - a| ≥ 2) ↔ (a ≤ -1 ∨ a ≥ 3) :=
sorry

end range_of_a_l21_2176


namespace probability_same_group_l21_2139

noncomputable def calcProbability : ℚ := 
  let totalOutcomes := 18 * 17
  let favorableCase1 := 6 * 5
  let favorableCase2 := 4 * 3
  let totalFavorableOutcomes := favorableCase1 + favorableCase2
  totalFavorableOutcomes / totalOutcomes

theorem probability_same_group (cards : Finset ℕ) (draws : Finset ℕ) (number1 number2 : ℕ) (condition_cardinality : cards.card = 20) 
  (condition_draws : draws.card = 4) (condition_numbers : number1 = 5 ∧ number2 = 14 ∧ number1 ∈ cards ∧ number2 ∈ cards) 
  : calcProbability = 7 / 51 :=
sorry

end probability_same_group_l21_2139


namespace sarah_class_choices_l21_2188

-- Conditions 
def total_classes : ℕ := 10
def choose_classes : ℕ := 4
def specific_classes : ℕ := 2

-- Statement
theorem sarah_class_choices : 
  ∃ (n : ℕ), n = Nat.choose (total_classes - specific_classes) 3 ∧ n = 56 :=
by 
  sorry

end sarah_class_choices_l21_2188


namespace longest_playing_time_l21_2152

theorem longest_playing_time (total_playtime : ℕ) (n : ℕ) (k : ℕ) (standard_time : ℚ) (long_time : ℚ) :
  total_playtime = 120 ∧ n = 6 ∧ k = 2 ∧ long_time = k * standard_time →
  5 * standard_time + long_time = 240 →
  long_time = 68 :=
by
  sorry

end longest_playing_time_l21_2152


namespace election_winner_won_by_votes_l21_2186

theorem election_winner_won_by_votes (V : ℝ) (winner_votes : ℝ) (loser_votes : ℝ)
    (h1 : winner_votes = 0.62 * V)
    (h2 : winner_votes = 930)
    (h3 : loser_votes = 0.38 * V)
    : winner_votes - loser_votes = 360 := 
  sorry

end election_winner_won_by_votes_l21_2186


namespace find_fg_of_3_l21_2184

def f (x : ℤ) : ℤ := 2 * x - 1
def g (x : ℤ) : ℤ := x^2 + 4 * x - 5

theorem find_fg_of_3 : f (g 3) = 31 := by
  sorry

end find_fg_of_3_l21_2184


namespace jade_living_expenses_l21_2149

-- Definitions from the conditions
variable (income : ℝ) (insurance_fraction : ℝ) (savings : ℝ) (P : ℝ)

-- Constants from the given problem
noncomputable def jadeIncome : income = 1600 := by sorry
noncomputable def jadeInsuranceFraction : insurance_fraction = 1 / 5 := by sorry
noncomputable def jadeSavings : savings = 80 := by sorry

-- The proof problem statement
theorem jade_living_expenses :
    (P * 1600 + (1 / 5) * 1600 + 80 = 1600) → P = 3 / 4 := by
    intros h
    sorry

end jade_living_expenses_l21_2149


namespace perimeter_with_new_tiles_l21_2168

theorem perimeter_with_new_tiles (p_original : ℕ) (num_original_tiles : ℕ) (num_new_tiles : ℕ)
  (h1 : p_original = 16)
  (h2 : num_original_tiles = 9)
  (h3 : num_new_tiles = 3) :
  ∃ p_new : ℕ, p_new = 17 :=
by
  sorry

end perimeter_with_new_tiles_l21_2168


namespace ratio_preference_l21_2165

-- Definitions based on conditions
def total_respondents : ℕ := 180
def preferred_brand_x : ℕ := 150
def preferred_brand_y : ℕ := total_respondents - preferred_brand_x

-- Theorem statement to prove the ratio of preferences
theorem ratio_preference : preferred_brand_x / preferred_brand_y = 5 := by
  sorry

end ratio_preference_l21_2165


namespace room_area_in_square_meters_l21_2197

theorem room_area_in_square_meters :
  ∀ (length_ft width_ft : ℝ), 
  (length_ft = 15) → 
  (width_ft = 8) → 
  (1 / 9 * 0.836127 = 0.092903) → 
  (length_ft * width_ft * 0.092903 = 11.14836) :=
by
  intros length_ft width_ft h_length h_width h_conversion
  -- sorry to skip the proof steps.
  sorry

end room_area_in_square_meters_l21_2197


namespace gcd_polynomial_l21_2175

theorem gcd_polynomial (b : ℤ) (h : 1820 ∣ b) : Int.gcd (b^2 + 11 * b + 28) (b + 6) = 2 := 
sorry

end gcd_polynomial_l21_2175


namespace no_intersection_of_asymptotes_l21_2161

noncomputable def given_function (x : ℝ) : ℝ :=
  (x^2 - 9 * x + 20) / (x^2 - 9 * x + 18)

theorem no_intersection_of_asymptotes : 
  (∀ x, x = 3 → ¬ ∃ y, y = given_function x) ∧ 
  (∀ x, x = 6 → ¬ ∃ y, y = given_function x) ∧ 
  ¬ ∃ x, (x = 3 ∨ x = 6) ∧ given_function x = 1 := 
by
  sorry

end no_intersection_of_asymptotes_l21_2161


namespace caleb_apples_less_than_kayla_l21_2122

theorem caleb_apples_less_than_kayla :
  ∀ (Kayla Suraya Caleb : ℕ),
  (Kayla = 20) →
  (Suraya = Kayla + 7) →
  (Suraya = Caleb + 12) →
  (Suraya = 27) →
  (Kayla - Caleb = 5) :=
by
  intros Kayla Suraya Caleb hKayla hSuraya1 hSuraya2 hSuraya3
  sorry

end caleb_apples_less_than_kayla_l21_2122


namespace satisfying_integers_l21_2102

theorem satisfying_integers (a b : ℤ) :
  a^4 + (a + b)^4 + b^4 = x^2 → a = 0 ∧ b = 0 :=
by
  -- Proof is required to be filled in here.
  sorry

end satisfying_integers_l21_2102


namespace correct_statement_l21_2148

variable (P Q : Prop)
variable (hP : P)
variable (hQ : Q)

theorem correct_statement :
  (P ∧ Q) :=
by
  exact ⟨hP, hQ⟩

end correct_statement_l21_2148


namespace total_profit_is_28000_l21_2109

noncomputable def investment_A (investment_B : ℝ) : ℝ := 3 * investment_B
noncomputable def period_A (period_B : ℝ) : ℝ := 2 * period_B
noncomputable def profit_B : ℝ := 4000
noncomputable def total_profit (investment_B period_B : ℝ) : ℝ :=
  let x := investment_B * period_B
  let a_share := 6 * x
  profit_B + a_share

theorem total_profit_is_28000 (investment_B period_B : ℝ) : 
  total_profit investment_B period_B = 28000 :=
by
  have h1 : profit_B = 4000 := rfl
  have h2 : investment_A investment_B = 3 * investment_B := rfl
  have h3 : period_A period_B = 2 * period_B := rfl
  simp [total_profit, h1, h2, h3]
  have x_def : investment_B * period_B = 4000 := by sorry
  simp [x_def]
  sorry

end total_profit_is_28000_l21_2109


namespace largest_number_is_310_l21_2115

def largest_number_formed (a b c : ℕ) : ℕ :=
  max (a * 100 + b * 10 + c) (max (a * 100 + c * 10 + b) (max (b * 100 + a * 10 + c) 
  (max (b * 100 + c * 10 + a) (max (c * 100 + a * 10 + b) (c * 100 + b * 10 + a)))))

theorem largest_number_is_310 : largest_number_formed 3 1 0 = 310 :=
by simp [largest_number_formed]; sorry

end largest_number_is_310_l21_2115


namespace combined_work_time_l21_2106

theorem combined_work_time (A B C D : ℕ) (hA : A = 10) (hB : B = 15) (hC : C = 20) (hD : D = 30) :
  1 / (1 / A + 1 / B + 1 / C + 1 / D) = 4 := by
  -- Replace the following "sorry" with your proof.
  sorry

end combined_work_time_l21_2106


namespace intersection_A_B_l21_2144

def A : Set ℕ := {1, 3, 5, 7, 9}
def B : Set ℕ := { x | 2 ≤ x ∧ x ≤ 5 }

theorem intersection_A_B : A ∩ B = {3, 5} :=
  sorry

end intersection_A_B_l21_2144


namespace a_six_between_three_and_four_l21_2114

theorem a_six_between_three_and_four (a : ℝ) (h : a^5 - a^3 + a = 2) : 3 < a^6 ∧ a^6 < 4 := 
sorry

end a_six_between_three_and_four_l21_2114


namespace find_a_if_perpendicular_l21_2146

theorem find_a_if_perpendicular (a : ℝ) :
  (∀ x y : ℝ, x + a * y + 2 = 0 → 2 * x + 3 * y + 1 = 0 → False) →
  a = -2 / 3 :=
by
  sorry

end find_a_if_perpendicular_l21_2146


namespace count_negative_numbers_l21_2137

def evaluate (e : String) : Int :=
  match e with
  | "-3^2" => -9
  | "(-3)^2" => 9
  | "-(-3)" => 3
  | "-|-3|" => -3
  | _ => 0

def isNegative (n : Int) : Bool := n < 0

def countNegatives (es : List String) : Int :=
  es.map evaluate |>.filter isNegative |>.length

theorem count_negative_numbers :
  countNegatives ["-3^2", "(-3)^2", "-(-3)", "-|-3|"] = 2 :=
by
  sorry

end count_negative_numbers_l21_2137


namespace number_of_cows_l21_2154

variable (D C : Nat)

theorem number_of_cows (h : 2 * D + 4 * C = 2 * (D + C) + 30) : C = 15 :=
by
  sorry

end number_of_cows_l21_2154


namespace value_of_g_at_3_l21_2116

def g (x : ℝ) := x^2 + 1

theorem value_of_g_at_3 : g 3 = 10 := by
  sorry

end value_of_g_at_3_l21_2116


namespace iterate_F_l21_2173

def F (x : ℝ) : ℝ := x^3 + 3*x^2 + 3*x

theorem iterate_F (x : ℝ) : (Nat.iterate F 2017 x) = (x + 1)^(3^2017) - 1 :=
by
  sorry

end iterate_F_l21_2173


namespace min_sum_m_n_l21_2155

noncomputable def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem min_sum_m_n (m n : ℕ) (h : (binomial m 2) * 2 = binomial (m + n) 2) : m + n = 4 := by
  sorry

end min_sum_m_n_l21_2155


namespace solve_equation_1_solve_equation_2_l21_2163

theorem solve_equation_1 (x : ℝ) : 2 * x^2 - x = 0 ↔ x = 0 ∨ x = 1 / 2 := 
by sorry

theorem solve_equation_2 (x : ℝ) : (2 * x + 1)^2 - 9 = 0 ↔ x = 1 ∨ x = -2 := 
by sorry

end solve_equation_1_solve_equation_2_l21_2163


namespace rectangle_perimeter_l21_2142

variable (L W : ℝ) 

theorem rectangle_perimeter (h1 : L > 4) (h2 : W > 4) (h3 : (L * W) - ((L - 4) * (W - 4)) = 168) : 
  2 * (L + W) = 92 := 
  sorry

end rectangle_perimeter_l21_2142


namespace kelly_raisins_l21_2120

theorem kelly_raisins (weight_peanuts : ℝ) (total_weight_snacks : ℝ) (h1 : weight_peanuts = 0.1) (h2 : total_weight_snacks = 0.5) : total_weight_snacks - weight_peanuts = 0.4 := by
  sorry

end kelly_raisins_l21_2120


namespace decrease_in_B_share_l21_2169

theorem decrease_in_B_share (a b c : ℝ) (x : ℝ) 
  (h1 : c = 495)
  (h2 : a + b + c = 1010)
  (h3 : (a - 25) / 3 = (b - x) / 2)
  (h4 : (a - 25) / 3 = (c - 15) / 5) :
  x = 10 :=
by
  sorry

end decrease_in_B_share_l21_2169


namespace angle_movement_condition_l21_2104

noncomputable def angle_can_reach_bottom_right (m n : ℕ) (h1 : 2 ≤ m) (h2 : 2 ≤ n) : Prop :=
  (m % 2 = 1) ∧ (n % 2 = 1)

theorem angle_movement_condition (m n : ℕ) (h1 : 2 ≤ m) (h2 : 2 ≤ n) :
  angle_can_reach_bottom_right m n h1 h2 ↔ (m % 2 = 1 ∧ n % 2 = 1) :=
sorry

end angle_movement_condition_l21_2104

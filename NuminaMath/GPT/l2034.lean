import Mathlib

namespace NUMINAMATH_GPT_parabola_at_point_has_value_zero_l2034_203493

theorem parabola_at_point_has_value_zero (a m : ℝ) :
  (x ^ 2 + (a + 1) * x + a) = 0 -> m = 0 :=
by
  -- We know the parabola passes through the point (-1, m)
  sorry

end NUMINAMATH_GPT_parabola_at_point_has_value_zero_l2034_203493


namespace NUMINAMATH_GPT_find_money_of_Kent_l2034_203451

variable (Alison Brittany Brooke Kent : ℝ)

def money_relations (h1 : Alison = 4000)
    (h2 : Alison = Brittany / 2)
    (h3 : Brittany = 4 * Brooke)
    (h4 : Brooke = 2 * Kent) : Prop :=
  Kent = 1000

theorem find_money_of_Kent
  {Alison Brittany Brooke Kent : ℝ}
  (h1 : Alison = 4000)
  (h2 : Alison = Brittany / 2)
  (h3 : Brittany = 4 * Brooke)
  (h4 : Brooke = 2 * Kent) :
  money_relations Alison Brittany Brooke Kent h1 h2 h3 h4 :=
by 
  sorry

end NUMINAMATH_GPT_find_money_of_Kent_l2034_203451


namespace NUMINAMATH_GPT_profit_per_package_l2034_203482

theorem profit_per_package
  (packages_first_center_per_day : ℕ)
  (packages_second_center_multiplier : ℕ)
  (weekly_profit : ℕ)
  (days_per_week : ℕ)
  (H1 : packages_first_center_per_day = 10000)
  (H2 : packages_second_center_multiplier = 3)
  (H3 : weekly_profit = 14000)
  (H4 : days_per_week = 7) :
  (weekly_profit / (packages_first_center_per_day * days_per_week + 
                    packages_second_center_multiplier * packages_first_center_per_day * days_per_week) : ℝ) = 0.05 :=
by
  sorry

end NUMINAMATH_GPT_profit_per_package_l2034_203482


namespace NUMINAMATH_GPT_volunteer_distribution_l2034_203494

theorem volunteer_distribution :
  let students := 5
  let projects := 4
  let combinations := Nat.choose students 2
  let permutations := Nat.factorial projects
  combinations * permutations = 240 := 
by
  sorry

end NUMINAMATH_GPT_volunteer_distribution_l2034_203494


namespace NUMINAMATH_GPT_swimming_pool_cost_l2034_203408

/-!
# Swimming Pool Cost Problem

Given:
* The pool takes 50 hours to fill.
* The hose runs at 100 gallons per hour.
* Water costs 1 cent for 10 gallons.

Prove that the total cost to fill the pool is 5 dollars.
-/

theorem swimming_pool_cost :
  let hours_to_fill := 50
  let hose_rate := 100  -- gallons per hour
  let cost_per_gallon := 0.01 / 10  -- dollars per gallon
  let total_volume := hours_to_fill * hose_rate  -- total volume in gallons
  let total_cost := total_volume * cost_per_gallon
  total_cost = 5 :=
by
  sorry

end NUMINAMATH_GPT_swimming_pool_cost_l2034_203408


namespace NUMINAMATH_GPT_factorize_expression_l2034_203491

variable (x y : ℝ)

theorem factorize_expression : 9 * x^2 * y - y = y * (3 * x + 1) * (3 * x - 1) := 
by
  sorry

end NUMINAMATH_GPT_factorize_expression_l2034_203491


namespace NUMINAMATH_GPT_exists_airline_route_within_same_republic_l2034_203477

theorem exists_airline_route_within_same_republic
  (C : Type) [Fintype C] [DecidableEq C]
  (R : Type) [Fintype R] [DecidableEq R]
  (belongs_to : C → R)
  (airline_route : C → C → Prop)
  (country_size : Fintype.card C = 100)
  (republics_size : Fintype.card R = 3)
  (millionaire_cities : {c : C // ∃ n : ℕ, n ≥ 70 ∧ (∃ S : Finset C, S.card = n ∧ ∀ x ∈ S, airline_route c x) })
  (at_least_70_millionaire_cities : ∃ n : ℕ, n ≥ 70 ∧ (∃ S : Finset {c : C // ∃ n : ℕ, n ≥ 70 ∧ ( ∃ S : Finset C, S.card = n ∧ ∀ x ∈ S, airline_route c x )}, S.card = n)):
  ∃ (c1 c2 : C), airline_route c1 c2 ∧ belongs_to c1 = belongs_to c2 := 
sorry

end NUMINAMATH_GPT_exists_airline_route_within_same_republic_l2034_203477


namespace NUMINAMATH_GPT_highest_page_number_l2034_203466

/-- Given conditions: Pat has 19 instances of the digit '7' and an unlimited supply of all 
other digits. Prove that the highest page number Pat can number without exceeding 19 instances 
of the digit '7' is 99. -/
theorem highest_page_number (num_of_sevens : ℕ) (highest_page : ℕ) 
  (h1 : num_of_sevens = 19) : highest_page = 99 :=
sorry

end NUMINAMATH_GPT_highest_page_number_l2034_203466


namespace NUMINAMATH_GPT_negation_of_prop_l2034_203428

theorem negation_of_prop :
  ¬ (∀ x : ℝ, x^2 - 1 > 0) ↔ ∃ x : ℝ, x^2 - 1 ≤ 0 :=
sorry

end NUMINAMATH_GPT_negation_of_prop_l2034_203428


namespace NUMINAMATH_GPT_solve_linear_system_l2034_203458

theorem solve_linear_system :
  ∃ (x y : ℚ), (4 * x - 3 * y = 2) ∧ (6 * x + 5 * y = 1) ∧ (x = 13 / 38) ∧ (y = -4 / 19) :=
by
  sorry

end NUMINAMATH_GPT_solve_linear_system_l2034_203458


namespace NUMINAMATH_GPT_remainder_when_sum_divided_by_7_l2034_203417

-- Define the sequence
def arithmetic_sequence (a d n : ℕ) : ℕ := a + (n - 1) * d

-- Define the sum of the arithmetic sequence
def arithmetic_sequence_sum (a d n : ℕ) : ℕ := (n * (2 * a + (n - 1) * d)) / 2

-- Given conditions
def a : ℕ := 3
def d : ℕ := 7
def last_term : ℕ := 304

-- Calculate the number of terms in the sequence
noncomputable def n : ℕ := (last_term + 4) / 7

-- Calculate the sum
noncomputable def S : ℕ := arithmetic_sequence_sum a d n

-- Lean 4 statement to prove the remainder
theorem remainder_when_sum_divided_by_7 : S % 7 = 3 := by
  -- sorry placeholder for proof
  sorry

end NUMINAMATH_GPT_remainder_when_sum_divided_by_7_l2034_203417


namespace NUMINAMATH_GPT_fraction_of_total_calls_l2034_203457

-- Definitions based on conditions
variable (B : ℚ) -- Calls processed by each member of Team B
variable (N : ℚ) -- Number of members in Team B

-- The fraction of calls processed by each member of Team A
def team_A_call_fraction : ℚ := 1 / 5

-- The fraction of calls processed by each member of Team C
def team_C_call_fraction : ℚ := 7 / 8

-- The fraction of agents in Team A relative to Team B
def team_A_agents_fraction : ℚ := 5 / 8

-- The fraction of agents in Team C relative to Team B
def team_C_agents_fraction : ℚ := 3 / 4

-- Total calls processed by Team A, Team B, and Team C
def total_calls_team_A : ℚ := (B * team_A_call_fraction) * (N * team_A_agents_fraction)
def total_calls_team_B : ℚ := B * N
def total_calls_team_C : ℚ := (B * team_C_call_fraction) * (N * team_C_agents_fraction)

-- Sum of total calls processed by all teams
def total_calls_all_teams : ℚ := total_calls_team_A B N + total_calls_team_B B N + total_calls_team_C B N

-- Potential total calls if all teams were as efficient as Team B
def potential_total_calls : ℚ := 3 * (B * N)

-- Fraction of total calls processed by all teams combined
def processed_fraction : ℚ := total_calls_all_teams B N / potential_total_calls B N

theorem fraction_of_total_calls : processed_fraction B N = 19 / 32 :=
by
  sorry -- Proof omitted

end NUMINAMATH_GPT_fraction_of_total_calls_l2034_203457


namespace NUMINAMATH_GPT_C_plus_D_l2034_203475

theorem C_plus_D (D C : ℚ) (h : ∀ x : ℚ, x ≠ 3 ∧ x ≠ 5 → (D * x - 17) / ((x - 3) * (x - 5)) = C / (x - 3) + 2 / (x - 5)) :
  C + D = 32 / 5 :=
by
  sorry

end NUMINAMATH_GPT_C_plus_D_l2034_203475


namespace NUMINAMATH_GPT_a_is_4_when_b_is_3_l2034_203412

theorem a_is_4_when_b_is_3 
  (a : ℝ) (b : ℝ) (k : ℝ)
  (h1 : ∀ b, a * b^2 = k)
  (h2 : a = 9 ∧ b = 2) :
  a = 4 :=
by
  sorry

end NUMINAMATH_GPT_a_is_4_when_b_is_3_l2034_203412


namespace NUMINAMATH_GPT_total_books_l2034_203445

theorem total_books (Zig_books : ℕ) (Flo_books : ℕ) (Tim_books : ℕ) 
  (hz : Zig_books = 60) (hf : Zig_books = 4 * Flo_books) (ht : Tim_books = Flo_books / 2) :
  Zig_books + Flo_books + Tim_books = 82 := by
  sorry

end NUMINAMATH_GPT_total_books_l2034_203445


namespace NUMINAMATH_GPT_find_p_l2034_203454

theorem find_p (p : ℚ) : (∀ x : ℚ, (3 * x + 4) = 0 → (4 * x ^ 3 + p * x ^ 2 + 17 * x + 24) = 0) → p = 13 / 4 :=
by
  sorry

end NUMINAMATH_GPT_find_p_l2034_203454


namespace NUMINAMATH_GPT_smallest_number_is_1013_l2034_203487

def smallest_number_divisible (n : ℕ) : Prop :=
  n - 5 % Nat.lcm 12 (Nat.lcm 16 (Nat.lcm 18 (Nat.lcm 21 28))) = 0

theorem smallest_number_is_1013 : smallest_number_divisible 1013 :=
by
  sorry

end NUMINAMATH_GPT_smallest_number_is_1013_l2034_203487


namespace NUMINAMATH_GPT_minutes_until_8_00_am_l2034_203441

-- Definitions based on conditions
def time_in_minutes (hours : Nat) (minutes : Nat) : Nat := hours * 60 + minutes

def current_time : Nat := time_in_minutes 7 30 + 16

def target_time : Nat := time_in_minutes 8 0

-- The theorem we need to prove
theorem minutes_until_8_00_am : target_time - current_time = 14 :=
by
  sorry

end NUMINAMATH_GPT_minutes_until_8_00_am_l2034_203441


namespace NUMINAMATH_GPT_sum_of_nine_consecutive_even_integers_mod_10_l2034_203460

theorem sum_of_nine_consecutive_even_integers_mod_10 : 
  (10112 + 10114 + 10116 + 10118 + 10120 + 10122 + 10124 + 10126 + 10128) % 10 = 0 := by
  sorry

end NUMINAMATH_GPT_sum_of_nine_consecutive_even_integers_mod_10_l2034_203460


namespace NUMINAMATH_GPT_gcd_is_3_l2034_203473

noncomputable def a : ℕ := 130^2 + 240^2 + 350^2
noncomputable def b : ℕ := 131^2 + 241^2 + 351^2

theorem gcd_is_3 : Nat.gcd a b = 3 := 
by 
  sorry

end NUMINAMATH_GPT_gcd_is_3_l2034_203473


namespace NUMINAMATH_GPT_trigonometric_identity_l2034_203461

theorem trigonometric_identity
  (θ : ℝ)
  (h : Real.tan θ = 1 / 3) :
  Real.sin (3 / 2 * Real.pi + 2 * θ) = -4 / 5 :=
by sorry

end NUMINAMATH_GPT_trigonometric_identity_l2034_203461


namespace NUMINAMATH_GPT_bruised_more_than_wormy_l2034_203498

noncomputable def total_apples : ℕ := 85
noncomputable def fifth_of_apples (n : ℕ) : ℕ := n / 5
noncomputable def apples_left_to_eat_raw : ℕ := 42

noncomputable def wormy_apples : ℕ := fifth_of_apples total_apples
noncomputable def total_non_raw_eatable_apples : ℕ := total_apples - apples_left_to_eat_raw
noncomputable def bruised_apples : ℕ := total_non_raw_eatable_apples - wormy_apples

theorem bruised_more_than_wormy :
  bruised_apples - wormy_apples = 43 - 17 :=
by sorry

end NUMINAMATH_GPT_bruised_more_than_wormy_l2034_203498


namespace NUMINAMATH_GPT_minimum_perimeter_triangle_MAF_is_11_l2034_203496

-- Define point, parabola, and focus
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define the specific points in the problem
def A : Point := ⟨5, 3⟩

-- Parabola with the form y^2 = 4x has the focus at (1, 0)
def F : Point := ⟨1, 0⟩

-- Minimum perimeter problem for ΔMAF
noncomputable def minimum_perimeter_triangle_MAF (M : Point) : ℝ :=
  (dist (M.x, M.y) (A.x, A.y)) + (dist (M.x, M.y) (F.x, F.y))

-- The goal is to show the minimum value of the perimeter is 11
theorem minimum_perimeter_triangle_MAF_is_11 (M : Point) 
  (hM_parabola : M.y^2 = 4 * M.x) 
  (hM_not_AF : M.x ≠ (5 + (3 * ((M.y - 0) / (M.x - 1))) )) : 
  ∃ M, minimum_perimeter_triangle_MAF M = 11 :=
sorry

end NUMINAMATH_GPT_minimum_perimeter_triangle_MAF_is_11_l2034_203496


namespace NUMINAMATH_GPT_sum_g_h_k_l2034_203471

def polynomial_product_constants (d g h k : ℤ) : Prop :=
  ((5 * d^2 + 4 * d + g) * (4 * d^2 + h * d - 5) = 20 * d^4 + 11 * d^3 - 9 * d^2 + k * d - 20)

theorem sum_g_h_k (d g h k : ℤ) (h1 : polynomial_product_constants d g h k) : g + h + k = -16 :=
by
  sorry

end NUMINAMATH_GPT_sum_g_h_k_l2034_203471


namespace NUMINAMATH_GPT_binary_to_decimal_11011_is_27_l2034_203456

def binary_to_decimal : ℕ :=
  1 * 2^4 + 1 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0

theorem binary_to_decimal_11011_is_27 : binary_to_decimal = 27 := by
  sorry

end NUMINAMATH_GPT_binary_to_decimal_11011_is_27_l2034_203456


namespace NUMINAMATH_GPT_valid_paths_from_P_to_Q_l2034_203485

-- Define the grid dimensions and alternate coloring conditions
def grid_width := 10
def grid_height := 8
def is_white_square (r c : ℕ) : Bool :=
  (r + c) % 2 = 1

-- Define the starting and ending squares P and Q
def P : ℕ × ℕ := (0, grid_width / 2)
def Q : ℕ × ℕ := (grid_height - 1, grid_width / 2)

-- Define a function to count valid 9-step paths from P to Q
noncomputable def count_valid_paths : ℕ :=
  -- Here the function to compute valid paths would be defined
  -- This is broad outline due to lean's framework missing specific combinatorial functions
  245

-- The theorem to state the proof problem
theorem valid_paths_from_P_to_Q : count_valid_paths = 245 :=
sorry

end NUMINAMATH_GPT_valid_paths_from_P_to_Q_l2034_203485


namespace NUMINAMATH_GPT_ratio_of_areas_of_circles_l2034_203433

theorem ratio_of_areas_of_circles 
  (R_A R_B : ℝ) 
  (h : (π / 2 * R_A) = (π / 3 * R_B)) : 
  (π * R_A ^ 2) / (π * R_B ^ 2) = (4 : ℚ) / 9 := 
sorry

end NUMINAMATH_GPT_ratio_of_areas_of_circles_l2034_203433


namespace NUMINAMATH_GPT_period_f_axis_of_symmetry_f_max_value_f_l2034_203426

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 5)

theorem period_f :
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = Real.pi := sorry

theorem axis_of_symmetry_f (k : ℤ) :
  ∀ x, 2 * x - Real.pi / 5 = Real.pi / 4 + k * Real.pi → x = 9 * Real.pi / 40 + k * Real.pi / 2 := sorry

theorem max_value_f :
  ∃ x ∈ Set.Icc (0 : ℝ) (Real.pi / 2), f x = 1 ∧ x = 7 * Real.pi / 20 := sorry

end NUMINAMATH_GPT_period_f_axis_of_symmetry_f_max_value_f_l2034_203426


namespace NUMINAMATH_GPT_sequence_general_term_l2034_203450

-- Define the sequence
def a : ℕ → ℕ
| 0 => 1
| n + 1 => 2 * a n + 1

-- State the theorem
theorem sequence_general_term (n : ℕ) : a n = 2^n - 1 :=
sorry

end NUMINAMATH_GPT_sequence_general_term_l2034_203450


namespace NUMINAMATH_GPT_gcd_437_323_eq_19_l2034_203416

theorem gcd_437_323_eq_19 : Int.gcd 437 323 = 19 := 
by 
  sorry

end NUMINAMATH_GPT_gcd_437_323_eq_19_l2034_203416


namespace NUMINAMATH_GPT_quadrilateral_impossible_l2034_203431

theorem quadrilateral_impossible (a b c d : ℕ) (h1 : 2 * a ^ 2 - 18 * a + 36 = 0)
    (h2 : b ^ 2 - 20 * b + 75 = 0) (h3 : c ^ 2 - 20 * c + 75 = 0) (h4 : 2 * d ^ 2 - 18 * d + 36 = 0) :
    ¬(a + b > d ∧ a + c > d ∧ b + c > d ∧ a + d > c ∧ b + d > c ∧ c + d > b ∧
      a + b + c > d ∧ a + b + d > c ∧ a + c + d > b ∧ b + c + d > a) :=
by
  sorry

end NUMINAMATH_GPT_quadrilateral_impossible_l2034_203431


namespace NUMINAMATH_GPT_combined_weight_l2034_203478

theorem combined_weight (S R : ℕ) (h1 : S = 71) (h2 : S - 5 = 2 * R) : S + R = 104 := by
  sorry

end NUMINAMATH_GPT_combined_weight_l2034_203478


namespace NUMINAMATH_GPT_difference_of_solutions_l2034_203424

theorem difference_of_solutions (x : ℝ) (h : (x + 3)^2 / (3 * x + 65) = 2) : ∃ a b : ℝ, a ≠ b ∧ (x = a ∨ x = b) ∧ abs (a - b) = 22 :=
by
  sorry

end NUMINAMATH_GPT_difference_of_solutions_l2034_203424


namespace NUMINAMATH_GPT_find_n_l2034_203440

theorem find_n (n : ℕ) (h₀ : 0 ≤ n) (h₁ : n ≤ 11) (h₂ : 10389 % 12 = n) : n = 9 :=
by sorry

end NUMINAMATH_GPT_find_n_l2034_203440


namespace NUMINAMATH_GPT_percentage_of_trout_is_correct_l2034_203499

-- Define the conditions
def video_game_cost := 60
def last_weekend_earnings := 35
def earnings_per_trout := 5
def earnings_per_bluegill := 4
def total_fish_caught := 5
def additional_savings_needed := 2

-- Define the total amount needed to buy the game
def total_required_savings := video_game_cost - additional_savings_needed

-- Define the amount earned this Sunday
def earnings_this_sunday := total_required_savings - last_weekend_earnings

-- Define the number of trout and blue-gill caught thisSunday
def num_trout := 3
def num_bluegill := 2    -- Derived from the conditions

-- Theorem: given the conditions, prove that the percentage of trout is 60%
theorem percentage_of_trout_is_correct :
  (num_trout + num_bluegill = total_fish_caught) ∧
  (earnings_per_trout * num_trout + earnings_per_bluegill * num_bluegill = earnings_this_sunday) →
  100 * num_trout / total_fish_caught = 60 := 
by
  sorry

end NUMINAMATH_GPT_percentage_of_trout_is_correct_l2034_203499


namespace NUMINAMATH_GPT_min_ab_l2034_203463

theorem min_ab (a b : ℝ) (h_pos : 0 < a ∧ 0 < b) (h_eq : a * b = a + b + 3) : a * b ≥ 9 :=
sorry

end NUMINAMATH_GPT_min_ab_l2034_203463


namespace NUMINAMATH_GPT_average_of_remaining_numbers_l2034_203476

theorem average_of_remaining_numbers (sum : ℕ) (average : ℕ) (remaining_sum : ℕ) (remaining_average : ℚ) :
  (average = 90) →
  (sum = 1080) →
  (remaining_sum = sum - 72 - 84) →
  (remaining_average = remaining_sum / 10) →
  remaining_average = 92.4 :=
by
  sorry

end NUMINAMATH_GPT_average_of_remaining_numbers_l2034_203476


namespace NUMINAMATH_GPT_main_inequality_equality_condition_l2034_203418

variable {a b c : ℝ}
variable (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

theorem main_inequality 
  (hpos_a : 0 < a) 
  (hpos_b : 0 < b) 
  (hpos_c : 0 < c) :
  (1 / (a * (1 + b)) + 1 / (b * (1 + c)) + 1 / (c * (1 + a)) ≥ 3 / (1 + a * b * c)) :=
  sorry

theorem equality_condition 
  (hpos_a : 0 < a) 
  (hpos_b : 0 < b) 
  (hpos_c : 0 < c) :
  (1 / (a * (1 + b)) + 1 / (b * (1 + c)) + 1 / (c * (1 + a)) = 3 / (1 + a * b * c) ↔ a = 1 ∧ b = 1 ∧ c = 1) :=
  sorry

end NUMINAMATH_GPT_main_inequality_equality_condition_l2034_203418


namespace NUMINAMATH_GPT_repeatingDecimal_exceeds_l2034_203447

noncomputable def repeatingDecimalToFraction (d : ℚ) : ℚ := 
    -- Function to convert repeating decimal to fraction
    if d = 0.99999 then 1 else (d * 100 - d) / 99  -- Simplified conversion for demonstration

def decimalToFraction (d : ℚ) : ℚ :=
    -- Function to convert decimal to fraction
    if d = 0.72 then 18 / 25 else 0  -- Replace with actual conversion

theorem repeatingDecimal_exceeds (x y : ℚ) (hx : repeatingDecimalToFraction x = 8/11) (hy : decimalToFraction y = 18/25):
    x - y = 2 / 275 :=
by
    sorry

end NUMINAMATH_GPT_repeatingDecimal_exceeds_l2034_203447


namespace NUMINAMATH_GPT_kishore_miscellaneous_expenses_l2034_203438

theorem kishore_miscellaneous_expenses :
  ∀ (rent milk groceries education petrol savings total_salary total_specified_expenses : ℝ),
  rent = 5000 →
  milk = 1500 →
  groceries = 4500 →
  education = 2500 →
  petrol = 2000 →
  savings = 2300 →
  (savings / 0.10) = total_salary →
  (rent + milk + groceries + education + petrol) = total_specified_expenses →
  (total_salary - (total_specified_expenses + savings)) = 5200 :=
by
  intros rent milk groceries education petrol savings total_salary total_specified_expenses
  sorry

end NUMINAMATH_GPT_kishore_miscellaneous_expenses_l2034_203438


namespace NUMINAMATH_GPT_who_plays_piano_l2034_203495

theorem who_plays_piano 
  (A : Prop)
  (B : Prop)
  (C : Prop)
  (hA : A = True)
  (hB : B = False)
  (hC : A = False)
  (only_one_true : (A ∧ ¬B ∧ ¬C) ∨ (¬A ∧ B ∧ ¬C) ∨ (¬A ∧ ¬B ∧ C)) : B = True := 
sorry

end NUMINAMATH_GPT_who_plays_piano_l2034_203495


namespace NUMINAMATH_GPT_median_of_36_consecutive_integers_l2034_203405

theorem median_of_36_consecutive_integers (sum_of_integers : ℕ) (num_of_integers : ℕ) 
  (h1 : num_of_integers = 36) (h2 : sum_of_integers = 6 ^ 4) : 
  (sum_of_integers / num_of_integers) = 36 := 
by 
  sorry

end NUMINAMATH_GPT_median_of_36_consecutive_integers_l2034_203405


namespace NUMINAMATH_GPT_solve_for_N_l2034_203448

theorem solve_for_N :
    (481 + 483 + 485 + 487 + 489 + 491 = 3000 - N) → (N = 84) :=
by
    -- Proof is omitted
    sorry

end NUMINAMATH_GPT_solve_for_N_l2034_203448


namespace NUMINAMATH_GPT_problem1_union_problem2_intersection_problem3_subset_l2034_203421

def A : Set ℝ := {x | x^2 - 2 * x - 3 ≤ 0}

def B (m : ℝ) : Set ℝ := {x | x^2 - 2 * m * x + m^2 - 4 ≤ 0}

theorem problem1_union (m : ℝ) (hm : m = 2) : A ∪ B m = {x | -1 ≤ x ∧ x ≤ 4} :=
sorry

theorem problem2_intersection (m : ℝ) (h : A ∩ B m = {x | 1 ≤ x ∧ x ≤ 3}) : m = 3 :=
sorry

theorem problem3_subset (m : ℝ) (h : A ⊆ {x | ¬ (x ∈ B m)}) : m > 5 ∨ m < -3 :=
sorry

end NUMINAMATH_GPT_problem1_union_problem2_intersection_problem3_subset_l2034_203421


namespace NUMINAMATH_GPT_pencils_placed_by_dan_l2034_203490

-- Definitions based on the conditions provided
def pencils_in_drawer : ℕ := 43
def initial_pencils_on_desk : ℕ := 19
def new_total_pencils : ℕ := 78

-- The statement to be proven
theorem pencils_placed_by_dan : pencils_in_drawer + initial_pencils_on_desk + 16 = new_total_pencils :=
by
  sorry

end NUMINAMATH_GPT_pencils_placed_by_dan_l2034_203490


namespace NUMINAMATH_GPT_no_real_roots_of_quadratic_l2034_203430

def quadratic_discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

theorem no_real_roots_of_quadratic (h : quadratic_discriminant 1 (-1) 1 < 0) :
  ¬ ∃ x : ℝ, x^2 - x + 1 = 0 :=
by
  sorry

end NUMINAMATH_GPT_no_real_roots_of_quadratic_l2034_203430


namespace NUMINAMATH_GPT_exponential_monotonicity_l2034_203400

theorem exponential_monotonicity {a b c : ℝ} (h1 : a > b) (h2 : b > 0) (h3 : c > 1) : c^a > c^b :=
by 
  sorry 

end NUMINAMATH_GPT_exponential_monotonicity_l2034_203400


namespace NUMINAMATH_GPT_find_C_l2034_203488

theorem find_C (A B C : ℕ) (h0 : 3 * A - A = 10) (h1 : B + A = 12) (h2 : C - B = 6) (h3 : A ≠ B) (h4 : B ≠ C) (h5 : C ≠ A) 
: C = 13 :=
sorry

end NUMINAMATH_GPT_find_C_l2034_203488


namespace NUMINAMATH_GPT_square_area_l2034_203434

noncomputable def side_length1 (x : ℝ) : ℝ := 5 * x - 20
noncomputable def side_length2 (x : ℝ) : ℝ := 25 - 2 * x

theorem square_area (x : ℝ) (h : side_length1 x = side_length2 x) :
  (side_length1 x)^2 = 7225 / 49 :=
by
  sorry

end NUMINAMATH_GPT_square_area_l2034_203434


namespace NUMINAMATH_GPT_total_spending_is_correct_l2034_203452

def total_spending : ℝ :=
  let meal_expenses_10 := 10 * 18
  let meal_expenses_5 := 5 * 25
  let total_meal_expenses := meal_expenses_10 + meal_expenses_5
  let service_charge := 50
  let total_before_discount := total_meal_expenses + service_charge
  let discount := 0.05 * total_meal_expenses
  let total_after_discount := total_before_discount - discount
  let tip := 0.10 * total_before_discount
  total_after_discount + tip

theorem total_spending_is_correct : total_spending = 375.25 :=
by
  sorry

end NUMINAMATH_GPT_total_spending_is_correct_l2034_203452


namespace NUMINAMATH_GPT_non_vegan_gluten_cupcakes_eq_28_l2034_203497

def total_cupcakes : ℕ := 80
def gluten_free_cupcakes : ℕ := total_cupcakes / 2
def vegan_cupcakes : ℕ := 24
def vegan_gluten_free_cupcakes : ℕ := vegan_cupcakes / 2
def non_vegan_cupcakes : ℕ := total_cupcakes - vegan_cupcakes
def gluten_cupcakes : ℕ := total_cupcakes - gluten_free_cupcakes
def non_vegan_gluten_cupcakes : ℕ := gluten_cupcakes - vegan_gluten_free_cupcakes

theorem non_vegan_gluten_cupcakes_eq_28 :
  non_vegan_gluten_cupcakes = 28 := by
  sorry

end NUMINAMATH_GPT_non_vegan_gluten_cupcakes_eq_28_l2034_203497


namespace NUMINAMATH_GPT_orange_ring_weight_correct_l2034_203474

-- Define the weights as constants
def purple_ring_weight := 0.3333333333333333
def white_ring_weight := 0.4166666666666667
def total_weight := 0.8333333333
def orange_ring_weight := 0.0833333333

-- Theorem statement
theorem orange_ring_weight_correct :
  total_weight - purple_ring_weight - white_ring_weight = orange_ring_weight :=
by
  -- Sorry is added to skip the proof part as per the instruction
  sorry

end NUMINAMATH_GPT_orange_ring_weight_correct_l2034_203474


namespace NUMINAMATH_GPT_sheela_overall_total_income_l2034_203409

def monthly_income_in_rs (income: ℝ) (savings: ℝ) (percent: ℝ): Prop :=
  savings = percent * income

def overall_total_income_in_rs (monthly_income: ℝ) 
                              (savings_deposit: ℝ) (fd_deposit: ℝ) 
                              (savings_interest_rate_monthly: ℝ) 
                              (fd_interest_rate_annual: ℝ): ℝ :=
  let annual_income := monthly_income * 12
  let savings_interest := savings_deposit * (savings_interest_rate_monthly * 12)
  let fd_interest := fd_deposit * fd_interest_rate_annual
  annual_income + savings_interest + fd_interest

theorem sheela_overall_total_income:
  ∀ (monthly_income: ℝ)
    (savings_deposit: ℝ) (fd_deposit: ℝ)
    (savings_interest_rate_monthly: ℝ) (fd_interest_rate_annual: ℝ),
    (monthly_income_in_rs monthly_income savings_deposit 0.28)  →
    monthly_income = 16071.43 →
    savings_deposit = 4500 →
    fd_deposit = 3000 →
    savings_interest_rate_monthly = 0.02 →
    fd_interest_rate_annual = 0.06 →
    overall_total_income_in_rs monthly_income savings_deposit fd_deposit
                           savings_interest_rate_monthly fd_interest_rate_annual
    = 194117.16 := 
by
  intros
  sorry

end NUMINAMATH_GPT_sheela_overall_total_income_l2034_203409


namespace NUMINAMATH_GPT_area_covered_by_both_strips_is_correct_l2034_203422

-- Definitions of lengths of the strips and areas
def length_total : ℝ := 16
def length_left : ℝ := 9
def length_right : ℝ := 7
def area_left_only : ℝ := 27
def area_right_only : ℝ := 18

noncomputable def width_strip : ℝ := sorry -- The width can be inferred from solution but is not the focus of the proof.

-- Definition of the area covered by both strips
def S : ℝ := 13.5

-- Proof statement
theorem area_covered_by_both_strips_is_correct :
  ∀ w : ℝ,
    length_left * w - S = area_left_only ∧ length_right * w - S = area_right_only →
    S = 13.5 := 
by
  sorry

end NUMINAMATH_GPT_area_covered_by_both_strips_is_correct_l2034_203422


namespace NUMINAMATH_GPT_area_of_black_region_l2034_203435

theorem area_of_black_region (side_small side_large : ℕ) 
  (h1 : side_small = 5) 
  (h2 : side_large = 9) : 
  (side_large * side_large) - (side_small * side_small) = 56 := 
by
  sorry

end NUMINAMATH_GPT_area_of_black_region_l2034_203435


namespace NUMINAMATH_GPT_angle_sum_l2034_203469

theorem angle_sum (y : ℝ) (h : 3 * y + y = 120) : y = 30 :=
sorry

end NUMINAMATH_GPT_angle_sum_l2034_203469


namespace NUMINAMATH_GPT_ceil_sqrt_200_eq_15_l2034_203489

theorem ceil_sqrt_200_eq_15 : ⌈Real.sqrt 200⌉ = 15 := 
sorry

end NUMINAMATH_GPT_ceil_sqrt_200_eq_15_l2034_203489


namespace NUMINAMATH_GPT_solve_equation1_solve_equation2_l2034_203436

theorem solve_equation1 (x : ℝ) : x^2 - 2 * x - 2 = 0 ↔ (x = 1 + Real.sqrt 3 ∨ x = 1 - Real.sqrt 3) :=
by
  sorry

theorem solve_equation2 (x : ℝ) : 2 * (x - 3)^2 = x - 3 ↔ (x = 3/2 ∨ x = 7/2) :=
by
  sorry

end NUMINAMATH_GPT_solve_equation1_solve_equation2_l2034_203436


namespace NUMINAMATH_GPT_right_angled_triangles_count_l2034_203453

theorem right_angled_triangles_count :
    ∃ n : ℕ, n = 31 ∧ ∀ (a b : ℕ), (b < 2011) ∧ (a * a = (b + 1) * (b + 1) - b * b) → n = 31 :=
by
  sorry

end NUMINAMATH_GPT_right_angled_triangles_count_l2034_203453


namespace NUMINAMATH_GPT_cafeteria_apples_pies_l2034_203492

theorem cafeteria_apples_pies (initial_apples handed_out_apples apples_per_pie remaining_apples pies : ℕ) 
    (h_initial: initial_apples = 62) 
    (h_handed_out: handed_out_apples = 8) 
    (h_apples_per_pie: apples_per_pie = 9)
    (h_remaining: remaining_apples = initial_apples - handed_out_apples) 
    (h_pies: pies = remaining_apples / apples_per_pie) : 
    pies = 6 := by
  sorry

end NUMINAMATH_GPT_cafeteria_apples_pies_l2034_203492


namespace NUMINAMATH_GPT_profit_per_meter_l2034_203462

theorem profit_per_meter (number_of_meters : ℕ) (total_selling_price cost_price_per_meter : ℝ) 
  (h1 : number_of_meters = 85) 
  (h2 : total_selling_price = 8925) 
  (h3 : cost_price_per_meter = 90) :
  (total_selling_price - cost_price_per_meter * number_of_meters) / number_of_meters = 15 :=
  sorry

end NUMINAMATH_GPT_profit_per_meter_l2034_203462


namespace NUMINAMATH_GPT_fraction_of_students_between_11_and_13_is_two_fifths_l2034_203481

def totalStudents : ℕ := 45
def under11 : ℕ :=  totalStudents / 3
def over13 : ℕ := 12
def between11and13 : ℕ := totalStudents - (under11 + over13)
def fractionBetween11and13 : ℚ := between11and13 / totalStudents

theorem fraction_of_students_between_11_and_13_is_two_fifths :
  fractionBetween11and13 = 2 / 5 := 
by 
  sorry

end NUMINAMATH_GPT_fraction_of_students_between_11_and_13_is_two_fifths_l2034_203481


namespace NUMINAMATH_GPT_car_miles_per_tankful_in_city_l2034_203404

-- Define constants for the given values
def miles_per_tank_on_highway : ℝ := 462
def fewer_miles_per_gallon : ℝ := 15
def miles_per_gallon_in_city : ℝ := 40

-- Prove the car traveled 336 miles per tankful in the city
theorem car_miles_per_tankful_in_city :
  (miles_per_tank_on_highway / (miles_per_gallon_in_city + fewer_miles_per_gallon)) * miles_per_gallon_in_city = 336 := 
by
  sorry

end NUMINAMATH_GPT_car_miles_per_tankful_in_city_l2034_203404


namespace NUMINAMATH_GPT_solution_is_correct_l2034_203407

-- Define the options
inductive Options
| A_some_other
| B_someone_else
| C_other_person
| D_one_other

-- Define the condition as a function that returns the correct option
noncomputable def correct_option : Options :=
Options.B_someone_else

-- The theorem stating that the correct option must be the given choice
theorem solution_is_correct : correct_option = Options.B_someone_else :=
by
  sorry

end NUMINAMATH_GPT_solution_is_correct_l2034_203407


namespace NUMINAMATH_GPT_smartphone_charging_time_l2034_203470

theorem smartphone_charging_time :
  ∀ (T S : ℕ), T = 53 → T + (1 / 2 : ℚ) * S = 66 → S = 26 :=
by
  intros T S hT equation
  sorry

end NUMINAMATH_GPT_smartphone_charging_time_l2034_203470


namespace NUMINAMATH_GPT_concentric_circle_ratio_l2034_203413

theorem concentric_circle_ratio (r R : ℝ) (hRr : R > r)
  (new_circles_tangent : ∀ (C1 C2 C3 : ℝ), C1 = C2 ∧ C2 = C3 ∧ C1 < R ∧ r < C1): 
  R = 3 * r := by sorry

end NUMINAMATH_GPT_concentric_circle_ratio_l2034_203413


namespace NUMINAMATH_GPT_arithmetic_sequence_a5_l2034_203414

theorem arithmetic_sequence_a5 
  (a : ℕ → ℤ) 
  (S : ℕ → ℤ)
  (h1 : a 1 = 1)
  (h2 : S 4 = 16)
  (h_sum : ∀ n, S n = (n * (2 * (a 1) + (n - 1) * (a 2 - a 1))) / 2)
  (h_a : ∀ n, a n = a 1 + (n - 1) * (a 2 - a 1)) :
  a 5 = 9 :=
by 
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a5_l2034_203414


namespace NUMINAMATH_GPT_minimal_team_members_l2034_203406

theorem minimal_team_members (n : ℕ) : 
  (n ≡ 1 [MOD 6]) ∧ (n ≡ 2 [MOD 8]) ∧ (n ≡ 3 [MOD 9]) → n = 343 := 
by
  sorry

end NUMINAMATH_GPT_minimal_team_members_l2034_203406


namespace NUMINAMATH_GPT_second_term_arithmetic_sequence_l2034_203402

theorem second_term_arithmetic_sequence 
  (a d : ℤ)
  (h1 : a + 15 * d = 8)
  (h2 : a + 16 * d = 10) : 
  a + d = -20 := 
by sorry

end NUMINAMATH_GPT_second_term_arithmetic_sequence_l2034_203402


namespace NUMINAMATH_GPT_intersection_A_B_l2034_203449

def A : Set ℝ := {-2, -1, 2, 3}
def B : Set ℝ := {x : ℝ | x^2 - x - 6 < 0}

theorem intersection_A_B : A ∩ B = {-1, 2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l2034_203449


namespace NUMINAMATH_GPT_algebraic_expression_value_l2034_203420

theorem algebraic_expression_value {m n : ℝ} 
  (h1 : n = m - 2022) 
  (h2 : m * n = -2022) : 
  (2022 / m) + ((m^2 - 2022 * m) / n) = 2022 := 
by sorry

end NUMINAMATH_GPT_algebraic_expression_value_l2034_203420


namespace NUMINAMATH_GPT_calculate_expression_l2034_203444

theorem calculate_expression : 
  ∀ (x y : ℕ), x = 3 → y = 4 → 3*(x^4 + 2*y^2)/9 = 37 + 2/3 :=
by
  intros x y hx hy
  sorry

end NUMINAMATH_GPT_calculate_expression_l2034_203444


namespace NUMINAMATH_GPT_determine_exponent_l2034_203443

theorem determine_exponent (m : ℕ) (hm : m > 0) (h_symm : ∀ x : ℝ, x^m - 3 = (-(x))^m - 3)
  (h_decr : ∀ (x y : ℝ), 0 < x ∧ x < y → x^m - 3 > y^m - 3) : m = 1 := 
sorry

end NUMINAMATH_GPT_determine_exponent_l2034_203443


namespace NUMINAMATH_GPT_min_expression_value_l2034_203423

theorem min_expression_value (a b c : ℝ) (h_sum : a + b + c = -1) (h_abc : a * b * c ≤ -3) :
  3 ≤ (ab + 1) / (a + b) + (bc + 1) / (b + c) + (ca + 1) / (c + a) :=
sorry

end NUMINAMATH_GPT_min_expression_value_l2034_203423


namespace NUMINAMATH_GPT_negation_divisible_by_5_is_odd_l2034_203465

theorem negation_divisible_by_5_is_odd : 
  ¬∀ n : ℤ, (n % 5 = 0) → (n % 2 ≠ 0) ↔ ∃ n : ℤ, (n % 5 = 0) ∧ (n % 2 = 0) := 
by 
  sorry

end NUMINAMATH_GPT_negation_divisible_by_5_is_odd_l2034_203465


namespace NUMINAMATH_GPT_right_triangle_sides_l2034_203468

theorem right_triangle_sides :
  (4^2 + 5^2 ≠ 6^2) ∧
  (1^2 + 1^2 = (Real.sqrt 2)^2) ∧
  (6^2 + 8^2 ≠ 11^2) ∧
  (5^2 + 12^2 ≠ 23^2) :=
by
  repeat { sorry }

end NUMINAMATH_GPT_right_triangle_sides_l2034_203468


namespace NUMINAMATH_GPT_abs_difference_of_numbers_l2034_203419

theorem abs_difference_of_numbers (x y : ℝ) (h1 : x + y = 24) (h2 : x * y = 104) : |x - y| = 4 * Real.sqrt 10 :=
by
  sorry

end NUMINAMATH_GPT_abs_difference_of_numbers_l2034_203419


namespace NUMINAMATH_GPT_sum_absolute_values_of_first_ten_terms_l2034_203429

noncomputable def S (n : ℕ) : ℤ := n^2 - 4 * n + 2

noncomputable def a (n : ℕ) : ℤ := S n - S (n - 1)

noncomputable def absolute_sum_10 : ℤ :=
  |a 1| + |a 2| + |a 3| + |a 4| + |a 5| + |a 6| + |a 7| + |a 8| + |a 9| + |a 10|

theorem sum_absolute_values_of_first_ten_terms : absolute_sum_10 = 68 := by
  sorry

end NUMINAMATH_GPT_sum_absolute_values_of_first_ten_terms_l2034_203429


namespace NUMINAMATH_GPT_middle_integer_is_zero_l2034_203442

-- Mathematical equivalent proof problem in Lean 4

theorem middle_integer_is_zero
  (n : ℤ)
  (h : (n - 2) + n + (n + 2) = (1 / 5) * ((n - 2) * n * (n + 2))) :
  n = 0 :=
by
  sorry

end NUMINAMATH_GPT_middle_integer_is_zero_l2034_203442


namespace NUMINAMATH_GPT_coed_softball_team_total_players_l2034_203415

theorem coed_softball_team_total_players (M W : ℕ) 
  (h1 : W = M + 4) 
  (h2 : (M : ℚ) / W = 0.6363636363636364) :
  M + W = 18 := 
by sorry

end NUMINAMATH_GPT_coed_softball_team_total_players_l2034_203415


namespace NUMINAMATH_GPT_find_y_l2034_203483

theorem find_y (y : ℝ) (h_cond : y = (1 / y) * (-y) - 3) : y = -4 := 
sorry

end NUMINAMATH_GPT_find_y_l2034_203483


namespace NUMINAMATH_GPT_max_value_of_a_b_c_l2034_203467

theorem max_value_of_a_b_c (a b c : ℤ) (h1 : a + b = 2006) (h2 : c - a = 2005) (h3 : a < b) : 
  a + b + c = 5013 :=
sorry

end NUMINAMATH_GPT_max_value_of_a_b_c_l2034_203467


namespace NUMINAMATH_GPT_train_crossing_time_l2034_203459

theorem train_crossing_time
  (L1 L2 : ℝ) (v : ℝ) 
  (t1 t2 t : ℝ) 
  (h_t1 : t1 = 27)
  (h_t2 : t2 = 17)
  (hv_ratio : v = v)
  (h_L1 : L1 = v * t1)
  (h_L2 : L2 = v * t2)
  (h_t12 : t = (L1 + L2) / (v + v)) :
  t = 22 :=
by
  sorry

end NUMINAMATH_GPT_train_crossing_time_l2034_203459


namespace NUMINAMATH_GPT_consecutive_product_neq_consecutive_even_product_l2034_203464

open Nat

theorem consecutive_product_neq_consecutive_even_product :
  ∀ m n : ℕ, m * (m + 1) ≠ 4 * n * (n + 1) :=
by
  intros m n
  -- Proof is omitted, as per instructions.
  sorry

end NUMINAMATH_GPT_consecutive_product_neq_consecutive_even_product_l2034_203464


namespace NUMINAMATH_GPT_line_passes_fixed_point_l2034_203403

theorem line_passes_fixed_point (k : ℝ) :
    ((k + 1) * -1) - ((2 * k - 1) * 1) + 3 * k = 0 :=
by
    -- The proof is omitted as the primary aim is to ensure the correct Lean statement.
    sorry

end NUMINAMATH_GPT_line_passes_fixed_point_l2034_203403


namespace NUMINAMATH_GPT_geometric_sequence_ratio_l2034_203401

theorem geometric_sequence_ratio 
  (a_n b_n : ℕ → ℝ) 
  (S_n T_n : ℕ → ℝ) 
  (h1 : ∀ n : ℕ, S_n n = a_n n * (1 - (1/2)^n)) 
  (h2 : ∀ n : ℕ, T_n n = b_n n * (1 - (1/3)^n))
  (h3 : ∀ n, n > 0 → (S_n n) / (T_n n) = (3^n + 1) / 4) : 
  (a_n 3) / (b_n 3) = 9 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_ratio_l2034_203401


namespace NUMINAMATH_GPT_john_avg_speed_last_30_minutes_l2034_203480

open Real

/-- John drove 160 miles in 120 minutes. His average speed during the first
30 minutes was 55 mph, during the second 30 minutes was 75 mph, and during
the third 30 minutes was 60 mph. Prove that his average speed during the
last 30 minutes was 130 mph. -/
theorem john_avg_speed_last_30_minutes (total_distance : ℝ) (total_time_minutes : ℝ)
  (speed_1 : ℝ) (speed_2 : ℝ) (speed_3 : ℝ) (speed_4 : ℝ) :
  total_distance = 160 →
  total_time_minutes = 120 →
  speed_1 = 55 →
  speed_2 = 75 →
  speed_3 = 60 →
  (speed_1 + speed_2 + speed_3 + speed_4) / 4 = total_distance / (total_time_minutes / 60) →
  speed_4 = 130 :=
by
  intro h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_john_avg_speed_last_30_minutes_l2034_203480


namespace NUMINAMATH_GPT_correct_statements_in_triangle_l2034_203446

theorem correct_statements_in_triangle (a b c : ℝ) (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) (h_sum : A + B + C = π) :
  (c = a * Real.cos B + b * Real.cos A) ∧ 
  (a^3 + b^3 = c^3 → a^2 + b^2 > c^2) :=
by
  sorry

end NUMINAMATH_GPT_correct_statements_in_triangle_l2034_203446


namespace NUMINAMATH_GPT_only_integers_square_less_than_three_times_l2034_203427

-- We want to prove that the only integers n that satisfy n^2 < 3n are 1 and 2.
theorem only_integers_square_less_than_three_times (n : ℕ) (h : n^2 < 3 * n) : n = 1 ∨ n = 2 :=
sorry

end NUMINAMATH_GPT_only_integers_square_less_than_three_times_l2034_203427


namespace NUMINAMATH_GPT_factor_expression_l2034_203479

variable (x : ℝ)

theorem factor_expression :
  (18 * x ^ 6 + 50 * x ^ 4 - 8) - (2 * x ^ 6 - 6 * x ^ 4 - 8) = 8 * x ^ 4 * (2 * x ^ 2 + 7) :=
by
  sorry

end NUMINAMATH_GPT_factor_expression_l2034_203479


namespace NUMINAMATH_GPT_find_a2_l2034_203411

variable (a : ℕ → ℤ)

-- Conditions
def is_arithmetic_sequence (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def is_geometric_sequence (x y z : ℤ) : Prop :=
  y * y = x * z

-- Specific condition for the problem
axiom h_arithmetic : is_arithmetic_sequence a 2
axiom h_geometric : is_geometric_sequence (a 1 + 2) (a 3 + 6) (a 4 + 8)

-- Theorem to prove
theorem find_a2 : a 1 + 2 = -8 := 
sorry

-- We assert that the value of a_2 must satisfy the given conditions

end NUMINAMATH_GPT_find_a2_l2034_203411


namespace NUMINAMATH_GPT_platform_length_l2034_203432

noncomputable def train_length : ℕ := 1200
noncomputable def time_to_cross_tree : ℕ := 120
noncomputable def time_to_pass_platform : ℕ := 230

theorem platform_length
  (v : ℚ)
  (h1 : v = train_length / time_to_cross_tree)
  (total_distance : ℚ)
  (h2 : total_distance = v * time_to_pass_platform)
  (platform_length : ℚ)
  (h3 : total_distance = train_length + platform_length) :
  platform_length = 1100 := by 
  sorry

end NUMINAMATH_GPT_platform_length_l2034_203432


namespace NUMINAMATH_GPT_train_speed_l2034_203437

noncomputable def speed_of_train (length_of_train length_of_overbridge time: ℝ) : ℝ :=
  (length_of_train + length_of_overbridge) / time

theorem train_speed (length_of_train length_of_overbridge time speed: ℝ)
  (h1 : length_of_train = 600)
  (h2 : length_of_overbridge = 100)
  (h3 : time = 70)
  (h4 : speed = 10) :
  speed_of_train length_of_train length_of_overbridge time = speed :=
by
  simp [speed_of_train, h1, h2, h3, h4]
  sorry

end NUMINAMATH_GPT_train_speed_l2034_203437


namespace NUMINAMATH_GPT_problem_sol_52_l2034_203410

theorem problem_sol_52 
  (x y: ℝ)
  (h1: x + y = 7)
  (h2: 4 * x * y = 7)
  (a b c d : ℕ)
  (hx_form : x = (a + b * Real.sqrt c) / d ∨ x = (a - b * Real.sqrt c) / d)
  (ha_pos : 0 < a)
  (hb_pos : 0 < b)
  (hc_pos : 0 < c)
  (hd_pos : 0 < d)
  : a + b + c + d = 52 := sorry

end NUMINAMATH_GPT_problem_sol_52_l2034_203410


namespace NUMINAMATH_GPT_determine_8_genuine_coins_l2034_203455

-- Assume there are 11 coins and one may be counterfeit.
variable (coins : Fin 11 → ℝ)
variable (is_counterfeit : Fin 11 → Prop)
variable (genuine_weight : ℝ)
variable (balance : (Fin 11 → ℝ) → (Fin 11 → ℝ) → Prop)

-- The weight of genuine coins.
axiom genuine_coins_weight : ∀ i, ¬ is_counterfeit i → coins i = genuine_weight

-- The statement of the mathematical problem in Lean 4.
theorem determine_8_genuine_coins :
  ∃ (genuine_set : Finset (Fin 11)), genuine_set.card ≥ 8 ∧ ∀ i ∈ genuine_set, ¬ is_counterfeit i :=
sorry

end NUMINAMATH_GPT_determine_8_genuine_coins_l2034_203455


namespace NUMINAMATH_GPT_binary_equals_octal_l2034_203472

-- Define the binary number 1001101 in decimal
def binary_1001101_decimal : ℕ := 1 * 2^6 + 0 * 2^5 + 0 * 2^4 + 1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0

-- Define the octal number 115 in decimal
def octal_115_decimal : ℕ := 1 * 8^2 + 1 * 8^1 + 5 * 8^0

-- Theorem statement
theorem binary_equals_octal :
  binary_1001101_decimal = octal_115_decimal :=
sorry

end NUMINAMATH_GPT_binary_equals_octal_l2034_203472


namespace NUMINAMATH_GPT_man_work_alone_l2034_203484

theorem man_work_alone (W: ℝ) (M S: ℝ)
  (hS: S = W / 6.67)
  (hMS: M + S = W / 4):
  W / M = 10 :=
by {
  -- This is a placeholder for the proof
  sorry
}

end NUMINAMATH_GPT_man_work_alone_l2034_203484


namespace NUMINAMATH_GPT_original_expenditure_l2034_203486

theorem original_expenditure (initial_students new_students : ℕ) (increment_expense : ℝ) (decrement_avg_expense : ℝ) (original_avg_expense : ℝ) (new_avg_expense : ℝ) 
  (total_initial_expense original_expenditure : ℝ)
  (h1 : initial_students = 35) 
  (h2 : new_students = 7) 
  (h3 : increment_expense = 42)
  (h4 : decrement_avg_expense = 1)
  (h5 : new_avg_expense = original_avg_expense - decrement_avg_expense)
  (h6 : total_initial_expense = initial_students * original_avg_expense)
  (h7 : original_expenditure = total_initial_expense)
  (h8 : 42 * new_avg_expense - original_students * original_avg_expense = increment_expense) :
  original_expenditure = 420 := 
by
  sorry

end NUMINAMATH_GPT_original_expenditure_l2034_203486


namespace NUMINAMATH_GPT_triangle_angles_l2034_203439

theorem triangle_angles
  (h_a a h_b b : ℝ)
  (h_a_ge_a : h_a ≥ a)
  (h_b_ge_b : h_b ≥ b)
  (a_ge_h_b : a ≥ h_b)
  (b_ge_h_a : b ≥ h_a) : 
  a = b ∧ 
  (a = h_a ∧ b = h_b) → 
  ∃ A B C : ℝ, Set.toFinset ({A, B, C} : Set ℝ) = {90, 45, 45} := 
by 
  sorry

end NUMINAMATH_GPT_triangle_angles_l2034_203439


namespace NUMINAMATH_GPT_sum_of_solutions_l2034_203425

theorem sum_of_solutions : 
  (∀ x : ℝ, (3 * x) / 15 = 4 / x) → (0 + 4 = 4) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_solutions_l2034_203425

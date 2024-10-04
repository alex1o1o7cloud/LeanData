import Mathlib
import Mathlib.Algebra!
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.Factorial
import Mathlib.Algebra.Field
import Mathlib.Algebra.GCDMonoid.Basic
import Mathlib.Algebra.GeomSum
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.GroupPower.Basic
import Mathlib.Algebra.Module.Equiv
import Mathlib.Algebra.Polynomials
import Mathlib.Algebra.Trigonometry
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.Trigonometry.Tangent
import Mathlib.Combinatorics
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Int.ModEq
import Mathlib.Data.List.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Gcd
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.SinCos
import Mathlib.NumberTheory.Pell
import Mathlib.Probability.Basic
import Mathlib.Probability.Continuous.Normal
import Mathlib.Probability.Independence
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.LibrarySearch
import Mathlib.Tactic.Linarith
import Mathlib.Topology.InfiniteSum
import data.rat.basic
import data.real.basic

namespace find_cost_per_sq_foot_l150_150286

noncomputable def monthly_rent := 2800 / 2
noncomputable def old_annual_rent (C : ℝ) := 750 * C * 12
noncomputable def new_annual_rent := monthly_rent * 12
noncomputable def annual_savings := old_annual_rent - new_annual_rent

theorem find_cost_per_sq_foot (C : ℝ):
    (750 * C * 12 - 2800 / 2 * 12 = 1200) ↔ (C = 2) :=
sorry

end find_cost_per_sq_foot_l150_150286


namespace exponent_property_l150_150107

theorem exponent_property : 3000 * 3000^2500 = 3000^2501 := 
by sorry

end exponent_property_l150_150107


namespace find_f_of_1_over_3_l150_150667

theorem find_f_of_1_over_3
  (g : ℝ → ℝ)
  (f : ℝ → ℝ)
  (h1 : ∀ x, g x = 1 - x^2)
  (h2 : ∀ x, x ≠ 0 → f (g x) = (1 - x^2) / x^2) :
  f (1 / 3) = 1 / 2 := by
  sorry -- Proof goes here

end find_f_of_1_over_3_l150_150667


namespace product_of_fractions_l150_150108

theorem product_of_fractions :
  (∏ n in finset.range 751, (4 * n + 4) / (4 * n + 8)) = (1 / 752) := by
  sorry

end product_of_fractions_l150_150108


namespace count_valid_n_decomposition_l150_150117

theorem count_valid_n_decomposition : 
  ∃ (count : ℕ), count = 108 ∧ 
  ∀ (a b c n : ℕ), 
    8 * a + 88 * b + 888 * c = 8000 → 
    0 ≤ b ∧ b ≤ 90 → 
    0 ≤ c ∧ c ≤ 9 → 
    n = a + 2 * b + 3 * c → 
    n < 1000 :=
sorry

end count_valid_n_decomposition_l150_150117


namespace sum_of_greatest_elements_l150_150088

theorem sum_of_greatest_elements (a b : ℤ)
  (h1 : 10 ≤ a ∧ a + 3 ≤ 99)
  (h2 : a ≡ 0 [MOD 6] ∨ a+1 ≡ 0 [MOD 6] ∨ a+2 ≡ 0 [MOD 6] ∨ a+3 ≡ 0 [MOD 6])
  (h3 : a ≡ 0 [MOD 7] ∨ a+1 ≡ 0 [MOD 7] ∨ a+2 ≡ 0 [MOD 7] ∨ a+3 ≡ 0 [MOD 7])
  (h4 : ¬ (a ≡ 5 [MOD 7] ∨ a ≡ 6 [MOD 7])) :
  (∃ S : set ℤ, (∀ x ∈ S, ∃ d, x = a + d ∧ 0 ≤ d ∧ d < 4) ∧
                (∑ x in S, x) = 204) :=
by
  sorry

end sum_of_greatest_elements_l150_150088


namespace distance_between_parallel_lines_l150_150550

theorem distance_between_parallel_lines
  (line1 : ∀ (x y : ℝ), 3*x - 2*y - 1 = 0)
  (line2 : ∀ (x y : ℝ), 3*x - 2*y + 1 = 0) :
  ∃ d : ℝ, d = (2 * Real.sqrt 13) / 13 :=
by
  sorry

end distance_between_parallel_lines_l150_150550


namespace fly_distance_l150_150910

theorem fly_distance (d v_a v_b : ℝ) (fly_speed_a fly_speed_b : ℝ)
  (initial_distance : d = 50) (relative_speed : v_a + v_b = 20) 
  (speed_from_ann_to_anne : fly_speed_a = 20) 
  (speed_from_anne_to_ann : fly_speed_b = 30) 
  :
  real.to_nnreal (d * (fly_speed_a * fly_speed_b) / (fly_speed_a + fly_speed_b)) = 55 := 
begin
  sorry,
end

end fly_distance_l150_150910


namespace digit_B_divisible_by_9_l150_150386

theorem digit_B_divisible_by_9 (B : ℕ) (k : ℤ) (h1 : 0 ≤ B) (h2 : B ≤ 9) (h3 : 2 * B + 6 = 9 * k) : B = 6 := 
by
  sorry

end digit_B_divisible_by_9_l150_150386


namespace time_to_cross_platform_l150_150479

def speed_km_hr : ℝ := 72
def length_train : ℝ := 260
def length_platform : ℝ := 260
def speed_m_s : ℝ := 20  -- 72 km/hr converted to m/s
def total_distance : ℝ := length_train + length_platform

theorem time_to_cross_platform : 
  let time := total_distance / speed_m_s in
  time = 26 :=
by
  sorry

end time_to_cross_platform_l150_150479


namespace length_of_chord_l150_150066

theorem length_of_chord {P : ℝ × ℝ} (hP : P = (1, Real.sqrt 3)) :
  let O := (0, 0)
  let r := 1
  let circle_eq := (λ P : ℝ × ℝ, P.1^2 + P.2^2 = r^2)
  let tangent_point (P : ℝ × ℝ) (O : ℝ × ℝ) (Q : ℝ × ℝ) :
    P.1 * Q.1 + P.2 * Q.2 = 0
  in circle_eq O →
     ∃ A B : ℝ × ℝ, tangent_point P O A ∧ tangent_point P O B ∧ 
     dist A B = Real.sqrt 3 :=
begin
  intros O r circle_eq tangent_point hO,
  sorry
end

end length_of_chord_l150_150066


namespace chris_least_money_l150_150903

variables (A B C D E : ℝ)

theorem chris_least_money
  (h1 : C < B)
  (h2 : D < B)
  (h3 : A > C)
  (h4 : E > C)
  (h5 : D = E)
  (h6 : D < A)
  (h7 : B > E) : 
  C < D ∧ C < E ∧ C < A ∧ C < B :=
by
  -- provided conditions and implications derived
  have h8 : C < D, from lt_of_lt_of_eq h4 (eq.symm h5),
  have h9 : C < E, from h4,
  have h10 : C < A, from h3,
  have h11 : C < B, from h1,
  exact ⟨h8, h9, h10, h11⟩

end chris_least_money_l150_150903


namespace sin_B_value_cos_A_value_l150_150682

theorem sin_B_value (A B C S : Real)
  (h1: ∃ (a b c : Real), 
    (a * c * Real.cos (π - B) = (3/2) * (1/2) * a * c * Real.sin B) ∧ 
    (S = (1/2) * a * c * Real.sin B)) : 
  Real.sin B = 4/5 :=
sorry

theorem cos_A_value (A B C : Real)
  (h1: ∃ (a b c : Real), 
    (a * c * Real.cos (π - B) = (3/2) * (1/2) * a * c * Real.sin B) ∧ 
    (S = (1/2) * a * c * Real.sin B)) 
  (h2: A - C = π/4)
  (h3: Real.sin B = 4/5) 
  (h4: Real.cos B = -3/5): 
  Real.cos A = Real.sqrt (50 + 5 * Real.sqrt 2) / 10 :=
sorry

end sin_B_value_cos_A_value_l150_150682


namespace balls_in_boxes_l150_150659

/-- Prove that the number of ways to put 6 distinguishable balls in 3 distinguishable boxes is 729 (which is 3^6). -/
theorem balls_in_boxes : (3 ^ 6) = 729 :=
by 
  sorry

end balls_in_boxes_l150_150659


namespace hypotenuse_length_l150_150034

theorem hypotenuse_length
  (a b : ℝ)
  (V1 : ℝ := (1/3) * Real.pi * a * b^2)
  (V2 : ℝ := (1/3) * Real.pi * b * a^2)
  (hV1 : V1 = 800 * Real.pi)
  (hV2 : V2 = 1920 * Real.pi) :
  Real.sqrt (a^2 + b^2) = 26 :=
by
  sorry

end hypotenuse_length_l150_150034


namespace smallest_x_y_sum_l150_150596

theorem smallest_x_y_sum (x y : ℕ) (h1 : x ≠ y) (h2 : x > 0) (h3 : y > 0) (h4 : 1 / (x : ℚ) + 1 / (y : ℚ) = 1 / 10) : x + y = 45 :=
by
  sorry

end smallest_x_y_sum_l150_150596


namespace sum_ratio_l150_150706

-- Assume an arithmetic sequence a_n
def arithmetic_seq (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a n = a 0 + n * d

-- Sum of the first n terms of the sequence
def sum_of_n (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  n * ( a 0 + (a (n - 1)) ) / 2

-- Define the given conditions
variables (a : ℕ → ℤ) (h_arith : arithmetic_seq a) (h_ratio : a 4 / a 2 = 5 / 9)

-- Define S9 and S5
def S_n (n : ℕ) := sum_of_n a n
def S_9 := S_n 9
def S_5 := S_n 5

-- The proof problem
theorem sum_ratio (a : ℕ → ℤ) (h_arith : arithmetic_seq a) (h_ratio : a 4 / a 2 = 5 / 9) :
  S_9 a h_arith h_ratio / S_5 a h_arith h_ratio = 1 := by
  sorry

end sum_ratio_l150_150706


namespace valid_k_values_l150_150230

theorem valid_k_values
  (k : ℝ)
  (h : k = -7 ∨ k = -5 ∨ k = 1 ∨ k = 4) :
  (∀ x, -4 < x ∧ x < 1 → (x < k ∨ x > k + 2)) → (k = -7 ∨ k = 1 ∨ k = 4) :=
by sorry

end valid_k_values_l150_150230


namespace find_a_l150_150961

variable (x y : ℕ → ℝ) (n : ℕ)
variable (h1 : n = 8)
variable (h2 : ∑ i in Finset.range n, x i = 6)
variable (h3 : ∑ i in Finset.range n, y i = 9)
variable (h4 : ∀ i, y i = (1/6) * x i + a)

theorem find_a (a : ℝ) (h1 : n = 8) (h2 : ∑ i in Finset.range n, x i = 6) (h3 : ∑ i in Finset.range n, y i = 9)
  (h4 : ∀ i, y i = (1/6) * x i + a) : a = 1 := 
by
  sorry

end find_a_l150_150961


namespace people_per_van_l150_150754

theorem people_per_van (num_students num_adults num_vans people_per_van : ℕ)
    (h_students : num_students = 25)
    (h_adults : num_adults = 5)
    (h_vans : num_vans = 6)
    (h_people_per_van : people_per_van = 5) :
    (num_students + num_adults) / num_vans = people_per_van :=
by
    rw [h_students, h_adults, h_vans, h_people_per_van]
    norm_num
    sorry

end people_per_van_l150_150754


namespace pool_capacity_l150_150042

-- Conditions
variables (C : ℝ) -- total capacity of the pool in gallons
variables (h1 : 300 = 0.75 * C - 0.45 * C) -- the pool requires an additional 300 gallons to be filled to 75%
variables (h2 : 300 = 0.30 * C) -- pumping in these additional 300 gallons will increase the amount of water by 30%

-- Goal
theorem pool_capacity : C = 1000 :=
by sorry

end pool_capacity_l150_150042


namespace sally_children_sum_of_ages_l150_150339

def ages : List ℕ := [1, 3, 5, 7, 9, 11, 13]

def sum_of_ages (ages : List ℕ) : ℕ := ages.foldr (· + ·) 0

theorem sally_children_sum_of_ages (x : ℕ) (N P : ℕ) :
  sum_of_ages ages = 49 → 
  (∃ x, sum_of_ages ages + 7 * x = P ∧ P = k^2 ∧ x = 21 ∧ N = 1 + x) → 
  N + P = 218 :=
by
  intros h₀ h₁ 
  cases h₁ with x_ h₂
  cases h₂ with h₃ h₄
  cases h₄ with h₅ h₆
  unsafe_assume h₃ as user_input 
  have h₄ as N + P = 218 from sorry
  exact h₄

end sally_children_sum_of_ages_l150_150339


namespace snow_probability_at_least_once_l150_150810

theorem snow_probability_at_least_once :
  let p := 3 / 4
  let prob_no_snow_single_day := 1 - p
  let prob_no_snow_all_days := prob_no_snow_single_day ^ 5
  let prob_snow_at_least_once := 1 - prob_no_snow_all_days
  prob_snow_at_least_once = 1023 / 1024 :=
by
  sorry

end snow_probability_at_least_once_l150_150810


namespace number_of_subsets_A_l150_150057

def A : Finset ℕ := {1, 2}

theorem number_of_subsets_A : A.powerset.card = 4 := by
  sorry

end number_of_subsets_A_l150_150057


namespace area_ratio_of_triangle_l150_150728

theorem area_ratio_of_triangle
  (A B C P : ℝ × ℝ)
  (h : (P.1 - A.1, P.2 - A.2) + (3 *(P.1 - B.1), 3 *(P.2 - B.2)) + (4 *(P.1 - C.1), 4 *(P.2 - C.2)) = (0, 0)) :
  let area := λ x y z: ℝ×ℝ, 0.5 * abs ((x.1 * (y.2 - z.2) + y.1 * (z.2 - x.2) + z.1 * (x.2 - y.2)) : ℝ) in
  (area A B C) / (area A P B) = 2.5 :=
  by
  sorry

end area_ratio_of_triangle_l150_150728


namespace distribute_6_balls_in_3_boxes_l150_150657

def number_of_ways_to_distribute_balls (balls boxes : Nat) : Nat :=
  boxes ^ balls

theorem distribute_6_balls_in_3_boxes : number_of_ways_to_distribute_balls 6 3 = 729 := by
  sorry

end distribute_6_balls_in_3_boxes_l150_150657


namespace range_of_m_l150_150632

theorem range_of_m (m : ℝ) (x1 x2 : ℝ) : 
  (∃ x1 x2 ∈ Ico 0 (2 * π), ∣x1 - x2∣ ≥ π ∧ sin (π - x1) + sin (π / 2 + x1) = m ∧ sin (π - x2) + sin (π / 2 + x2) = m) → 
  0 ≤ m ∧ m < 1 :=
sorry

end range_of_m_l150_150632


namespace find_f4_l150_150232

variable (a b : ℝ)
variable (f : ℝ → ℝ)
variable (h1 : f 1 = 5)
variable (h2 : f 2 = 8)
variable (h3 : f 3 = 11)
variable (h4 : ∀ x, f x = a * x + b)

theorem find_f4 : f 4 = 14 := by
  sorry

end find_f4_l150_150232


namespace tangent_segments_ratio_l150_150065

variables {r s : ℝ}

noncomputable def triangle_side_lengths := (10 : ℝ, 15 : ℝ, 19 : ℝ)
noncomputable def tangency_segments (r s : ℝ) := r + s = 10

theorem tangent_segments_ratio (h : 3 + 7 = 10) : r / s = 3 / 7 := by
  sorry

end tangent_segments_ratio_l150_150065


namespace fraction_four_or_older_l150_150471

theorem fraction_four_or_older (total_students : ℕ) (under_three : ℕ) (not_between_three_and_four : ℕ)
  (h_total : total_students = 300) (h_under_three : under_three = 20) (h_not_between_three_and_four : not_between_three_and_four = 50) :
  (not_between_three_and_four - under_three) / total_students = 1 / 10 :=
by
  sorry

end fraction_four_or_older_l150_150471


namespace total_money_in_dollars_l150_150038

/-- You have some amount in nickels and quarters.
    You have 40 nickels and the same number of quarters.
    Prove that the total amount of money in dollars is 12. -/
theorem total_money_in_dollars (n_nickels n_quarters : ℕ) (value_nickel value_quarter : ℕ) 
  (h1: n_nickels = 40) (h2: n_quarters = 40) (h3: value_nickel = 5) (h4: value_quarter = 25) : 
  (n_nickels * value_nickel + n_quarters * value_quarter) / 100 = 12 :=
  sorry

end total_money_in_dollars_l150_150038


namespace jack_pays_back_l150_150280

-- conditions in the problem 
def principal : ℝ := 1200
def interest_rate : ℝ := 0.1

-- the theorem statement equivalent to the question and correct answer
theorem jack_pays_back (principal_interest: principal * interest_rate) (total_amount: principal + principal_interest) : total_amount = 1320 :=
by
  sorry

end jack_pays_back_l150_150280


namespace probability_of_inequality_l150_150673

noncomputable def probability_condition {x : ℝ} (h : x ∈ Icc (-1 : ℝ) 4) : Prop :=
  2 * x - 2 * x^2 ≥ -4

theorem probability_of_inequality : 
  (∃ P : ℚ, P = 3 / 5 ∧ ∀ x : ℝ, (x ∈ Icc (-1 : ℝ) 4) → probability_condition (set.mem_Icc.mpr ⟨by linarith, by linarith⟩)) :=
  sorry

end probability_of_inequality_l150_150673


namespace find_second_number_l150_150374

theorem find_second_number (x y : ℤ) (h1 : x = -63) (h2 : (2 + y + x) / 3 = 5) : y = 76 :=
sorry

end find_second_number_l150_150374


namespace proj_eq_line_eqn_l150_150406

theorem proj_eq_line_eqn (x y : ℝ)
  (h : (6 * x + 3 * y) * 6 / 45 = -3 ∧ (6 * x + 3 * y) * 3 / 45 = -3 / 2) :
  y = -2 * x - 15 / 2 :=
by
  sorry

end proj_eq_line_eqn_l150_150406


namespace compute_factorial_ratio_l150_150173

theorem compute_factorial_ratio (n : ℕ) (K : ℕ) (hK: P(ξ = K) = 1 / 2^K) : factorial n / (factorial 3 * factorial (n - 3)) = 35 := by 
  -- define P and ξ appropriately here

  sorry

end compute_factorial_ratio_l150_150173


namespace five_digit_even_unit_probability_l150_150478

noncomputable def even_unit_digit_probability : ℚ :=
  let even_digits := {0, 2, 4, 6, 8}.to_finset
  let all_digits := (finset.range 10).to_finset
  (even_digits.card : ℚ) / all_digits.card

theorem five_digit_even_unit_probability : even_unit_digit_probability = 1 / 2 :=
  by
    sorry

end five_digit_even_unit_probability_l150_150478


namespace daily_sales_volume_80_sales_volume_function_price_for_profit_l150_150366

-- Define all relevant conditions
def cost_price : ℝ := 70
def max_price : ℝ := 99
def initial_price : ℝ := 95
def initial_sales : ℕ := 50
def price_reduction_effect : ℕ := 2

-- Part 1: Proving daily sales volume at 80 yuan
theorem daily_sales_volume_80 : 
  (initial_price - 80) * price_reduction_effect + initial_sales = 80 := 
by sorry

-- Part 2: Proving functional relationship
theorem sales_volume_function (x : ℝ) (h₁ : 70 ≤ x) (h₂ : x ≤ 99) : 
  (initial_sales + price_reduction_effect * (initial_price - x) = -2 * x + 240) :=
by sorry

-- Part 3: Proving price for 1200 yuan daily profit
theorem price_for_profit (profit_target : ℝ) (h : profit_target = 1200) :
  ∃ x, (x - cost_price) * (initial_sales + price_reduction_effect * (initial_price - x)) = profit_target ∧ x ≤ max_price :=
by sorry

end daily_sales_volume_80_sales_volume_function_price_for_profit_l150_150366


namespace area_of_paper_l150_150846

theorem area_of_paper (L W : ℕ) (h1 : 2 * L + W = 34) (h2 : L + 2 * W = 38) : 
  L * W = 140 := 
by sorry

end area_of_paper_l150_150846


namespace find_a_l150_150184

theorem find_a (a : ℝ) :
  ∃ x : ℝ, y = x + 2 ∧ y = log (x + a) ∧ (∀ x_ : ℝ, deriv (λ x, log (x + a)) x_ = 1) → a = 3 :=
sorry

end find_a_l150_150184


namespace smallest_sum_of_xy_l150_150610

theorem smallest_sum_of_xy (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hxy : x ≠ y)
  (hcond : (1 : ℚ) / x + (1 : ℚ) / y = 1 / 10) : x + y = 45 :=
sorry

end smallest_sum_of_xy_l150_150610


namespace connect_four_no_win_probability_l150_150697

-- Definitions based on the conditions
def connect_four := {grid : array (7*6) (option (sum unit unit)) // 
  ∀ (row column : ℕ), (∀ direction : ℤ × ℤ, direction ≠ (0, 0) →  
  (0 ≤ row + 3 * direction.1 ∧ row + 3 * direction.1 < 6) ∧ 
  (0 ≤ column + 3 * direction.2 ∧ column + 3 * direction.2 < 7) → 
  (1 ≤ row ∧ row < 6) ∧ (1 ≤ column ∧ column < 7) → 
  ¬ (grid.get! (row + 3 * direction.1) (column + 3 * direction.2) = 
    some (sum.inl ())))}

def random_play (players_turn : ℕ) (grid: array (7*6) (option (sum unit unit))) :
    array (7*6) (option (sum unit unit)) :=
  sorry -- definition of a random play will be complex and is not provided here
  
def probability_no_win : ℝ :=
  sorry -- simulated or empirical estimation of the probability

theorem connect_four_no_win_probability :
  probability_no_win ≈ 0.0025632817 :=
sorry

end connect_four_no_win_probability_l150_150697


namespace each_persons_contribution_l150_150361

def total_cost : ℝ := 67
def coupon : ℝ := 4
def num_people : ℝ := 3

theorem each_persons_contribution :
  (total_cost - coupon) / num_people = 21 := by
  sorry

end each_persons_contribution_l150_150361


namespace mutually_exclusive_pairs_l150_150930

/-
We establish the definitions for mutually exclusive events and verify which pairs 
from the given conditions satisfy this property.
-/

def mutually_exclusive (A B : Prop) : Prop := A → ¬ B

def event1 := (hit_7_ring : Prop) ∧ (hit_8_ring : Prop)
def event2 := (∃ a b, a ∨ b) ∧ (a ∧ ¬ b)
def event3 := (∃ at_least_one_black : Prop) ∧ (both_red: Prop)
def event4 := (no_black : Prop) ∧ (exactly_one_red : Prop)

theorem mutually_exclusive_pairs :
  mutually_exclusive event1 event3 ∧ 
  mutually_exclusive event1 event4 ∧ 
  mutually_exclusive event3 event4 := 
sorry

end mutually_exclusive_pairs_l150_150930


namespace smallest_m_divisible_by_15_l150_150741

noncomputable def largest_prime_with_2023_digits : ℕ := sorry

theorem smallest_m_divisible_by_15 :
  ∃ m : ℕ, m > 0 ∧ (largest_prime_with_2023_digits ^ 2 - m) % 15 = 0 ∧ m = 1 :=
  sorry

end smallest_m_divisible_by_15_l150_150741


namespace uncle_dave_ice_cream_sandwiches_l150_150827

theorem uncle_dave_ice_cream_sandwiches (n : ℕ) (s : ℕ) (total : ℕ) 
  (h1 : n = 11) (h2 : s = 13) (h3 : total = n * s) : total = 143 := by
  sorry

end uncle_dave_ice_cream_sandwiches_l150_150827


namespace coeff_of_x3_in_expansion_l150_150127

theorem coeff_of_x3_in_expansion : 
  coeff (expand (λ x => (1/2 * x^2 - 1 / x)^6) 3) = -5 / 2 :=
sorry

end coeff_of_x3_in_expansion_l150_150127


namespace find_WZ_length_l150_150702

noncomputable def WZ_length (XY YZ XZ WX : ℝ) (theta : ℝ) : ℝ :=
  Real.sqrt ((WX^2 + XZ^2 - 2 * WX * XZ * (-1 / 2)))

-- Define the problem within the context of the provided lengths and condition
theorem find_WZ_length :
  WZ_length 3 5 7 8.5 (-1 / 2) = Real.sqrt 180.75 :=
by 
  -- This "by sorry" is used to indicate the proof is omitted
  sorry

end find_WZ_length_l150_150702


namespace arc_radius_l150_150078

theorem arc_radius (α S R : ℝ) (h_eq : R = (Real.sqrt (S * Real.sqrt 3)) / (2 * (Real.sin (α/4))^2)) : 
  ∃ R, R = (Real.sqrt (S * Real.sqrt 3)) / (2 * (Real.sin (α / 4))^2) :=
by
  use R
  exact h_eq

end arc_radius_l150_150078


namespace parking_savings_l150_150490

theorem parking_savings (weekly_cost : ℕ) (monthly_cost : ℕ) (weeks_in_year : ℕ) (months_in_year : ℕ)
  (h_weekly_cost : weekly_cost = 10)
  (h_monthly_cost : monthly_cost = 42)
  (h_weeks_in_year : weeks_in_year = 52)
  (h_months_in_year : months_in_year = 12) :
  weekly_cost * weeks_in_year - monthly_cost * months_in_year = 16 := 
by
  sorry

end parking_savings_l150_150490


namespace inequality_sqrt2_a3_l150_150160

theorem inequality_sqrt2_a3 (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  sqrt 2 * a^3 + 3 / (a * b - b^2) ≥ 10 :=
sorry

end inequality_sqrt2_a3_l150_150160


namespace find_coordinates_of_P_l150_150625

noncomputable theory

open Real

def PointLiesOnLine (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  2 * x + y - 3 = 0

def PointInFirstQuadrant (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  x > 0 ∧ y > 0

def DistanceToLine (P : ℝ × ℝ) : ℝ :=
  let (x, y) := P
  abs (x - 2 * y - 4) / sqrt (1^2 + (-2)^2)

theorem find_coordinates_of_P (P : ℝ × ℝ) (h1 : PointLiesOnLine P) (h2 : PointInFirstQuadrant P) (h3 : DistanceToLine P = sqrt 5) : P = (1, 1) :=
sorry

end find_coordinates_of_P_l150_150625


namespace johnPaysPerYear_l150_150719

-- Define the conditions
def epiPenCost : ℝ := 500
def insuranceCoverage : ℝ := 0.75
def epiPenFrequencyPerYear : ℝ := 2 -- Twice a year since 12 months / 6 months per EpiPen

-- Calculate the cost after insurance
def costAfterInsurance (cost : ℝ) (coverage : ℝ) : ℝ :=
  cost * (1 - coverage)

-- Calculate the annual cost
def annualCost (freq : ℝ) (cost : ℝ) : ℝ :=
  freq * cost

-- The statement asserting the proof
theorem johnPaysPerYear (cost : ℝ) (coverage : ℝ) (freq : ℝ) : 
  epiPenCost = cost → 
  insuranceCoverage = coverage → 
  epiPenFrequencyPerYear = freq → 
  annualCost freq (costAfterInsurance cost coverage) = 250 := 
by 
  intros _ _ _ 
  sorry

end johnPaysPerYear_l150_150719


namespace polar_eq_C2_dist_AB_C1C2_l150_150265

-- Definitions based on conditions
def C1 (t : ℝ) : ℝ × ℝ :=
  (1 + 2 * t, 2 - 2 * t)

def C2 (θ : ℝ) (h : 0 ≤ θ ∧ θ ≤ 2 * Real.pi) : ℝ × ℝ :=
  (2 * Real.cos θ + 2, 2 * Real.sin θ)

-- Statement for polar equation of C2
theorem polar_eq_C2 (θ : ℝ) (h : 0 ≤ θ ∧ θ ≤ 2 * Real.pi) :
  ∃ ρ, C2 θ h = (ρ * Real.cos θ, ρ * Real.sin θ) ∧ ρ = 4 * Real.cos θ := 
sorry

-- Statement for distance |AB| between intersection points of C1 and C2
theorem dist_AB_C1C2 :
  ∀ (t1 t2 : ℝ), 
  C1 t1 = (x1, y1) →
  C1 t2 = (x2, y2) →
  (∃ θ1 θ2, C2 θ1 h1 = (x1, y1) ∧ C2 θ2 h2 = (x2, y2)) →
  ∃ d, d = Real.sqrt 14 ∧ 
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) = d :=
sorry

end polar_eq_C2_dist_AB_C1C2_l150_150265


namespace coords_of_point_P_l150_150237

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x + 2)

theorem coords_of_point_P :
  ∀ (a : ℝ), 0 < a ∧ a ≠ 1 → ∃ P : ℝ × ℝ, (P = (1, -2) ∧ ∀ y, f (f a (-2)) y = y) :=
by
  sorry

end coords_of_point_P_l150_150237


namespace minimum_cost_l150_150015

theorem minimum_cost (
    x y m w : ℝ) 
    (h1 : 4 * x + 2 * y = 400)
    (h2 : 2 * x + 4 * y = 320)
    (h3 : m ≥ 16)
    (h4 : m + (80 - m) = 80)
    (h5 : w = 80 * m + 40 * (80 - m)) :
    x = 80 ∧ y = 40 ∧ w = 3840 :=
by 
  sorry

end minimum_cost_l150_150015


namespace find_n_for_positive_root_l150_150241

theorem find_n_for_positive_root :
  ∃ x : ℝ, x > 0 ∧ (∃ n : ℝ, (n / (x - 1) + 2 / (1 - x) = 1)) ↔ n = 2 :=
by
  sorry

end find_n_for_positive_root_l150_150241


namespace rex_remaining_cards_l150_150335

-- Definitions based on the conditions provided:
def nicole_cards : ℕ := 400
def cindy_cards (nicole_cards : ℕ) : ℕ := 2 * nicole_cards
def combined_total (nicole_cards cindy_cards : ℕ) : ℕ := nicole_cards + cindy_cards nicole_cards
def rex_cards (combined_total : ℕ) : ℕ := combined_total / 2
def rex_divided_cards (rex_cards siblings : ℕ) : ℕ := rex_cards / (1 + siblings)

-- The theorem to be proved based on the question and correct answer:
theorem rex_remaining_cards : rex_divided_cards (rex_cards (combined_total nicole_cards (cindy_cards nicole_cards))) 3 = 150 :=
by sorry

end rex_remaining_cards_l150_150335


namespace all_values_achievable_all_values_achievable_1_all_values_achievable_2_all_values_achievable_3_all_values_achievable_4_l150_150785

def coin_values : Set ℤ := {1, 5, 10, 25}

theorem all_values_achievable (a b c d: ℕ) (h: a + b + c + d = 6) (h_a: a * 1 + b * 5 + c * 10 + d * 25 = 30) 
  (coins: Set ℤ := coin_values) : 
  ∃ (x y z w: ℕ), x + y + z + w = 6 ∧ x * 1 + y * 5 + z * 10 + w * 25 = a * 1 + b * 5 + c * 10 + d * 25 :=
by sorry

theorem all_values_achievable_1 (a b c d: ℕ) (h: a + b + c + d = 6) (h_a: a * 1 + b * 5 + c * 10 + d * 25 = 40) 
  (coins: Set ℤ := coin_values) : 
  ∃ (x y z w: ℕ), x + y + z + w = 6 ∧ x * 1 + y * 5 + z * 10 + w * 25 = a * 1 + b * 5 + c * 10 + d * 25 :=
by sorry

theorem all_values_achievable_2 (a b c d: ℕ) (h: a + b + c + d = 6) (h_a: a * 1 + b * 5 + c * 10 + d * 25 = 50) 
  (coins: Set ℤ := coin_values) : 
  ∃ (x y z w: ℕ), x + y + z + w = 6 ∧ x * 1 + y * 5 + z * 10 + w * 25 = a * 1 + b * 5 + c * 10 + d * 25 :=
by sorry

theorem all_values_achievable_3 (a b c d: ℕ) (h: a + b + c + d = 6) (h_a: a * 1 + b * 5 + c * 10 + d * 25 = 60) 
  (coins: Set ℤ := coin_values) : 
  ∃ (x y z w: ℕ), x + y + z + w = 6 ∧ x * 1 + y * 5 + z * 10 + w * 25 = a * 1 + b * 5 + c * 10 + d * 25 :=
by sorry

theorem all_values_achievable_4 (a b c d: ℕ) (h: a + b + c + d = 6) (h_a: a * 1 + b * 5 + c * 10 + d * 25 = 70) 
  (coins: Set ℤ := coin_values) : 
  ∃ (x y z w: ℕ), x + y + z + w = 6 ∧ x * 1 + y * 5 + z * 10 + w * 25 = a * 1 + b * 5 + c * 10 + d * 25 :=
by sorry

end all_values_achievable_all_values_achievable_1_all_values_achievable_2_all_values_achievable_3_all_values_achievable_4_l150_150785


namespace son_l150_150487

theorem son's_age (S M : ℕ) 
  (h1 : M = S + 35)
  (h2 : M + 2 = 2 * (S + 2)) : 
  S = 33 := 
by
  sorry

end son_l150_150487


namespace component_unqualified_l150_150470

/-- 
    The specified diameter range for a component is within [19.98, 20.02].
    The measured diameter of the component is 19.9.
    Prove that the component is unqualified.
-/
def is_unqualified (diameter_measured : ℝ) : Prop :=
    diameter_measured < 19.98 ∨ diameter_measured > 20.02

theorem component_unqualified : is_unqualified 19.9 :=
by
  -- Proof goes here
  sorry

end component_unqualified_l150_150470


namespace cost_of_baseball_is_correct_l150_150325

-- Define the costs and total amount spent
def cost_of_marbles : ℝ := 9.05
def cost_of_football : ℝ := 4.95
def total_amount_spent : ℝ := 20.52

-- Define the cost of the baseball
def cost_of_baseball : ℝ := total_amount_spent - (cost_of_marbles + cost_of_football)

-- The theorem we want to prove
theorem cost_of_baseball_is_correct :
  cost_of_baseball = 6.52 := by
  sorry

end cost_of_baseball_is_correct_l150_150325


namespace find_integer_b_l150_150542

-- Define the polynomial
def polynomial (x b : ℤ) : ℤ := x^3 + 4 * x^2 + b * x + 12

-- Prove that having an integer root implies b is in the expected set
theorem find_integer_b (b : ℤ) : (∃ x : ℤ, polynomial x b = 0) → 
  b ∈ {-177, -62, -35, -25, -18, -17, 9, 16, 27, 48, 144, 1296} := by
  sorry

end find_integer_b_l150_150542


namespace largest_binomial_coefficient_term_largest_coefficient_term_l150_150998

theorem largest_binomial_coefficient_term (n : ℕ) (hn : 4^n - 2^n = 992) :
  let expansion := (x : ℝ) := x^((2:ℝ)/3) + 3*x^(2:ℝ)
  let power_term := expansion^n
  ∃ r s : ℕ, (choose n r) * 3^r * x^((10 + 4 * r) / 3) = 90 * x^6 ∨ (choose n s) * 3^s * x^((10 + 4 * s) / 3) = 270 * x^((22:ℝ)/3) :=
by 
  use 2, 3
  sorry

theorem largest_coefficient_term (n : ℕ) (hn : 4^n - 2^n = 992) :
  let expansion := (x : ℝ) := x^((2:ℝ)/3) + 3*x^(2:ℝ)
  let power_term := expansion^n
  ∃ t : ℕ, (choose n t) * 3^t * x^((10 + 4 * t) / 3) = 405 * x^((26:ℝ)/3) :=
by
  use 4
  sorry

end largest_binomial_coefficient_term_largest_coefficient_term_l150_150998


namespace zero_count_in_circular_sequence_l150_150769

/--
The number of zeroes occurring in a circular sequence without repetitions is \(0, 1, 2,\) or \(4\).
-/
theorem zero_count_in_circular_sequence (n : ℕ) (f : ℕ → ℤ) (h_circular : ∀ m, f (m + n) = f m)
  (h_no_repetition : ∀ i j, (i < n ∧ j < n ∧ f i = f j) → i = j): 
  ∃ k ∈ {0, 1, 2, 4}, ∀ i < n, f i = 0 → (∃ j < 5, i = j) :=
sorry

end zero_count_in_circular_sequence_l150_150769


namespace part1_a_range_part2_x_range_l150_150973
open Real

-- Definitions based on given conditions
def quad_func (a b x : ℝ) : ℝ :=
  a * x^2 + b * x + 2

def y_at_x1 (a b : ℝ) : Prop :=
  quad_func a b 1 = 1

def pos_on_interval (a b l r : ℝ) (x : ℝ) : Prop :=
  l < x ∧ x < r → 0 < quad_func a b x

-- Part 1 proof statement in Lean 4
theorem part1_a_range (a b : ℝ) (h1 : y_at_x1 a b) (h2 : ∀ x : ℝ, pos_on_interval a b 2 5 x) :
  a > 3 - 2 * sqrt 2 :=
sorry

-- Part 2 proof statement in Lean 4
theorem part2_x_range (a b : ℝ) (h1 : y_at_x1 a b) (h2 : ∀ a' : ℝ, -2 ≤ a' ∧ a' ≤ -1 → 0 < quad_func a' b x) :
  (1 - sqrt 17) / 4 < x ∧ x < (1 + sqrt 17) / 4 :=
sorry

end part1_a_range_part2_x_range_l150_150973


namespace smallest_sum_l150_150601

theorem smallest_sum (x y : ℕ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x ≠ y) (eq : 1/x + 1/y = 1/10) : x + y = 45 :=
by
  sorry

end smallest_sum_l150_150601


namespace maximum_height_l150_150059

-- Define the initial conditions and the physics problem setup.
def v0 : ℝ := 20
def h0 : ℝ := 5
def g : ℝ := 10

-- Define the height function h(t) as a function of time t in seconds.
def h (t : ℝ) : ℝ := - (1 / 2) * g * t^2 + v0 * t + h0

theorem maximum_height : ∃ t, h(t) = 25 :=
by
  -- We skip the proof for now.
  sorry

end maximum_height_l150_150059


namespace region_area_l150_150948

noncomputable def area_of_region := 
  let a := 0
  let b := Real.sqrt 2 / 2
  ∫ x in a..b, (Real.arccos x) - (Real.arcsin x)

theorem region_area : area_of_region = 2 - Real.sqrt 2 :=
by
  sorry

end region_area_l150_150948


namespace probability_of_snowing_at_least_once_l150_150813

theorem probability_of_snowing_at_least_once (p : ℚ) (h : p = 3 / 4) :
  let q := 1 - p in
  let not_snowing_five_days := q ^ 5 in
  let at_least_once := 1 - not_snowing_five_days in
  at_least_once = 1023 / 1024 :=
by
  have q_def : q = 1 - p := rfl,
  have not_snowing_five_days_def : not_snowing_five_days = q ^ 5 := rfl,
  have at_least_once_def : at_least_once = 1 - not_snowing_five_days := rfl,
  sorry

end probability_of_snowing_at_least_once_l150_150813


namespace product_of_three_greater_than_product_of_two_or_four_l150_150171

theorem product_of_three_greater_than_product_of_two_or_four
  (nums : Fin 10 → ℝ)
  (h_positive : ∀ i, 0 < nums i)
  (h_distinct : Function.Injective nums) :
  ∃ (a b c : Fin 10),
    (∃ (d e : Fin 10), (a ≠ b) ∧ (a ≠ c) ∧ (b ≠ c) ∧ (a ≠ d) ∧ (a ≠ e) ∧ (b ≠ d) ∧ (b ≠ e) ∧ (c ≠ d) ∧ (c ≠ e) ∧ nums a * nums b * nums c > nums d * nums e) ∨
    (∃ (d e f g : Fin 10), (a ≠ b) ∧ (a ≠ c) ∧ (b ≠ c) ∧ (a ≠ d) ∧ (a ≠ e) ∧ (a ≠ f) ∧ (a ≠ g) ∧ (b ≠ d) ∧ (b ≠ e) ∧ (b ≠ f) ∧ (b ≠ g) ∧ (c ≠ d) ∧ (c ≠ e) ∧ (c ≠ f) ∧ (c ≠ g) ∧ nums a * nums b * nums c > nums d * nums e * nums f * nums g) :=
sorry

end product_of_three_greater_than_product_of_two_or_four_l150_150171


namespace remaining_bollards_to_be_installed_l150_150900

-- Definitions based on the problem's conditions
def total_bollards_per_side := 4000
def sides := 2
def total_bollards := total_bollards_per_side * sides
def installed_fraction := 3 / 4
def installed_bollards := installed_fraction * total_bollards
def remaining_bollards := total_bollards - installed_bollards

-- The statement to be proved
theorem remaining_bollards_to_be_installed : remaining_bollards = 2000 :=
  by sorry

end remaining_bollards_to_be_installed_l150_150900


namespace max_students_distribution_l150_150852

theorem max_students_distribution (pens toys : ℕ) (h_pens : pens = 451) (h_toys : toys = 410) :
  Nat.gcd pens toys = 41 :=
by
  sorry

end max_students_distribution_l150_150852


namespace perfect_square_n_l150_150311

theorem perfect_square_n (n : ℤ) (h1 : n > 0) (h2 : ∃ k : ℤ, n^2 + 19 * n + 48 = k^2) : n = 33 :=
sorry

end perfect_square_n_l150_150311


namespace ferris_wheel_time_l150_150774

noncomputable def radius : ℝ := 30
noncomputable def revolution_time : ℝ := 90
noncomputable def desired_height : ℝ := 15

theorem ferris_wheel_time :
  ∃ t : ℝ, 0 <= t ∧ t <= revolution_time / 2 ∧ 30 * real.cos ((real.pi / 45) * t) + 30 = 15 ∧ t = 30 :=
by
  sorry

end ferris_wheel_time_l150_150774


namespace population_growth_proof_l150_150250

noncomputable def population_growth_percent (p : ℕ) : ℝ :=
  let final_population := (p + q - p) ^ 2
  let initial_population := p ^ 2
  ((final_population - initial_population) / initial_population) * 100

theorem population_growth_proof (p q : ℕ) (h1 : p^2 = initial_population) (h2 : q^2 = 12 + p^2 + 180) :
    p = 11 → q = 17 → population_growth_percent p = 265 :=
sorry

end population_growth_proof_l150_150250


namespace volume_ratio_l150_150346

-- Definitions of points and edges
variables {P A B C M N K : Point}
variables {PA PB PC : Line}
variables {PM : Segment PA} {PA : Segment PA}
variables {PN : Segment PB} {PB : Segment PB}
variables {PK : Segment PC} {PC : Segment PC}

-- The theorem to prove the volume ratio

theorem volume_ratio 
(h1: PM ∈ PA)
(h2: PN ∈ PB)
(h3: PK ∈ PC) :
  (volume (P, M, N, K) / volume (P, A, B, C)) = 
  (length PM / length PA) * (length PN / length PB) * (length PK / length PC) := sorry

end volume_ratio_l150_150346


namespace number_of_incorrect_statements_l150_150999

def statement1 (I : Set ℝ) (f : ℝ → ℝ) : Prop :=
  ∃ (x1 x2 : ℝ), x1 ∈ I ∧ x2 ∈ I ∧ x1 < x2 ∧ f x1 < f x2 → ∀ (x1 x2 : ℝ), x1 ∈ I ∧ x2 ∈ I ∧ x1 < x2 → f x1 ≤ f x2

def statement2 : Prop := 
  ∀ (x : ℝ), x ≠ 0 → deriv (λ x, 1/x) x < 0

def statement3 : Prop := 
  ∀ (x : ℝ), x > 0 → deriv (λ x, (x - 1)^2) x ≥ 0

def statement4 : Prop := 
  ∀ (x : ℝ), x < 0 → deriv (λ x, -1/x) x ≥ 0

theorem number_of_incorrect_statements :
  ∃ I f, ¬statement1 I f ∧ ¬statement2 ∧ ¬statement3 ∧ statement4 → 
  3 = ({s1 | ¬statement1 I f} ∪ {s2 | ¬statement2} ∪ {s3 | ¬statement3} ∪ {s4 | statement4}).card :=
by
  sorry

end number_of_incorrect_statements_l150_150999


namespace initial_population_l150_150060

theorem initial_population (P : ℝ) (h : 3553 = 0.85 * (0.95 * P)) : P ≈ 4400 :=
by
  sorry

end initial_population_l150_150060


namespace find_grade_2_l150_150514

-- Definitions for the problem
def grade_1 := 78
def weight_1 := 20
def weight_2 := 30
def grade_3 := 90
def weight_3 := 10
def grade_4 := 85
def weight_4 := 40
def overall_average := 83

noncomputable def calc_weighted_average (G : ℕ) : ℝ :=
  (grade_1 * weight_1 + G * weight_2 + grade_3 * weight_3 + grade_4 * weight_4) / (weight_1 + weight_2 + weight_3 + weight_4)

theorem find_grade_2 (G : ℕ) : calc_weighted_average G = overall_average → G = 81 := sorry

end find_grade_2_l150_150514


namespace john_pays_per_year_l150_150716

-- Define the costs and insurance parameters.
def cost_per_epipen : ℝ := 500
def insurance_coverage : ℝ := 0.75

-- Number of months in a year.
def months_in_year : ℕ := 12

-- Number of months each EpiPen lasts.
def months_per_epipen : ℕ := 6

-- Amount covered by insurance for each EpiPen.
def insurance_amount (cost : ℝ) (coverage : ℝ) : ℝ :=
  cost * coverage

-- Amount John pays after insurance for each EpiPen.
def amount_john_pays_per_epipen (cost : ℝ) (covered: ℝ) : ℝ :=
  cost - covered

-- Number of EpiPens John needs per year.
def epipens_per_year (months_in_year : ℕ) (months_per_epipen : ℕ) : ℕ :=
  months_in_year / months_per_epipen

-- Total amount John pays per year.
def total_amount_john_pays_per_year (amount_per_epipen : ℝ) (epipens_per_year : ℕ) : ℝ :=
  amount_per_epipen * epipens_per_year

-- Theorem to prove the correct answer.
theorem john_pays_per_year :
  total_amount_john_pays_per_year (amount_john_pays_per_epipen cost_per_epipen (insurance_amount cost_per_epipen insurance_coverage)) (epipens_per_year months_in_year months_per_epipen) = 250 := 
by
  sorry

end john_pays_per_year_l150_150716


namespace convergence_iff_expected_condition_l150_150299

open MeasureTheory

variable {X : ℕ → ℝ}
variable μ : Measure ℝ

def uniformly_integrable_family (X : ℕ → ℝ → ℝ) : Prop :=
  ∀ (ε > 0), ∃ δ > 0, ∀ n, ∫⁻ x, ennreal.of_real (|X n x|) ∂μ ≤ ε

def convergence_in_distribution (X : ℕ → ℝ) (Z : ℝ → ℝ) : Prop :=
  ∀ (f : ℝ → ℝ) (hf : Continuous f), tendsto (λ n, ∫⁻ x, ennreal.of_real (f (X n x)) ∂μ) at_top (𝓝 (∫⁻ x, ennreal.of_real (f (Z x)) ∂μ))

noncomputable def expected_condition (X : ℕ → ℝ) (t : ℝ) : Prop :=
  tendsto (λ n, ∫⁻ x, ennreal.of_real (X n x * cexp (complex.I * t * X n x) - complex.I * t * cexp (complex.I * t * X n x))) at_top (𝓝 0)

theorem convergence_iff_expected_condition (X : ℕ → ℝ) :
  (convergence_in_distribution X (λ x, real.of_en (pdf_normal 0 1 x))) ↔
  (∀ t : ℝ, expected_condition X t) := sorry

end convergence_iff_expected_condition_l150_150299


namespace diameter_of_large_circle_is_approximately_28_9442_l150_150943

-- Define the small circle's radius
def radius_small : ℝ := 4.0

-- The number of smaller circles
def num_small_circles : ℕ := 8

-- Distance function using the distance between centers of two tangent circles
def side_length_of_octagon (r : ℝ) := 2 * r

-- Relationship between side length s and radius R of a regular octagon 
def radius_of_octagon (s : ℝ) := s / (2 * Real.sin (Real.pi / 8))

-- Calculating the radius of the larger circle
def radius_large_circle (r : ℝ) : ℝ := 
  let s := side_length_of_octagon r
  let R := radius_of_octagon s
  R + r

-- Calculating the diameter of the larger circle
def diameter_large_circle (r : ℝ) : ℝ :=
  2 * radius_large_circle r

theorem diameter_of_large_circle_is_approximately_28_9442 :
  (diameter_large_circle 4 ≈ 28.9442) :=
by
  sorry

end diameter_of_large_circle_is_approximately_28_9442_l150_150943


namespace two_circles_externally_tangent_l150_150247

noncomputable def externally_tangent_circles_specify_m (m : ℝ) : Prop :=
  let circle1 := λ (x y : ℝ), x^2 + y^2 - 2 * m * x = 0
  let circle2 := λ (x y : ℝ), x^2 + (y - 2)^2 = 1
  let center1 := (m, 0)
  let radius1 := |m|
  let center2 := (0, 2)
  let radius2 := 1
  dist center1 center2 = radius1 + radius2

theorem two_circles_externally_tangent :
  externally_tangent_circles_specify_m (frac 3 2) ∨ 
  externally_tangent_circles_specify_m (-frac 3 2) :=
sorry

end two_circles_externally_tangent_l150_150247


namespace number_of_foreign_stamps_l150_150253

theorem number_of_foreign_stamps 
    (total_stamps : ℕ) (old_stamps : ℕ)
    (foreign_and_old_stamps : ℕ) (neither_stamps : ℕ)
    (h1 : total_stamps = 200) (h2 : old_stamps = 50)
    (h3 : foreign_and_old_stamps = 20) (h4 : neither_stamps = 90) :
    (∀ (F : ℕ), total_stamps = F + old_stamps - foreign_and_old_stamps + neither_stamps → F = 80) :=
by {
    intros F h,
    have eq1 : F + 120 = 200,
    from eq.trans h (by rw [h1, h2, h3, h4]),
    exact nat.sub_eq_of_eq_add (eq.symm eq1)
}

end number_of_foreign_stamps_l150_150253


namespace bisect_angle_GHD_l150_150731

variable {A B C D E F G H M : Type*} [MetricSpace A]
variable (sides : AB AC : LineSegment AB AC)
variable (triangleABF_sim_triangleACE : Similar (Triangle A B F) (Triangle A C E))
variable (right_angle : Angle A B F = 90)
variable (intersection_M : Intersection (Line B E) (Line C F) = M)
variable (intersection_D : Intersection (Line B E) (Line A C) = D)
variable (intersection_G : Intersection (Line C F) (Line A B) = G)
variable (perpendicular_H : Perpendicular (Line M H) (Line B C) at H)

theorem bisect_angle_GHD : Bisects (Line M H) (Angle G H D) :=
sorry

end bisect_angle_GHD_l150_150731


namespace fraction_of_bananas_is_3_div_5_l150_150063

-- Definitions of the initial conditions
def initial_apples : Nat := 12
def initial_bananas : Nat := 15
def additional_bananas : Nat := 3

-- Calculation of the expected number of bananas and total fruits
def total_bananas : Nat := initial_bananas + additional_bananas
def total_fruit : Nat := initial_apples + total_bananas
def fraction_bananas : Rat := ⟨total_bananas, total_fruit⟩ -- The fraction of bananas

-- The theorem representing the problem statement
theorem fraction_of_bananas_is_3_div_5 :
  fraction_bananas = 3 / 5 :=
sorry

end fraction_of_bananas_is_3_div_5_l150_150063


namespace taxi_fare_distance_l150_150409

theorem taxi_fare_distance (x : ℕ) (h₁ : 8 + 2 * (x - 3) = 20) : x = 9 :=
by {
  sorry
}

end taxi_fare_distance_l150_150409


namespace diagonal_length_of_quadrilateral_l150_150545

theorem diagonal_length_of_quadrilateral 
  (area : ℝ) (m n : ℝ) (d : ℝ) 
  (h_area : area = 210) 
  (h_m : m = 9) 
  (h_n : n = 6) 
  (h_formula : area = 0.5 * d * (m + n)) : 
  d = 28 :=
by 
  sorry

end diagonal_length_of_quadrilateral_l150_150545


namespace time_to_sell_all_cars_l150_150105

/-- Conditions: -/
def total_cars : ℕ := 500
def number_of_sales_professionals : ℕ := 10
def cars_per_salesperson_per_month : ℕ := 10

/-- Proof Statement: -/
theorem time_to_sell_all_cars 
  (total_cars : ℕ) 
  (number_of_sales_professionals : ℕ) 
  (cars_per_salesperson_per_month : ℕ) : 
  ((number_of_sales_professionals * cars_per_salesperson_per_month) > 0) →
  (total_cars / (number_of_sales_professionals * cars_per_salesperson_per_month)) = 5 :=
by
  sorry

end time_to_sell_all_cars_l150_150105


namespace Intersection_A_B_l150_150747

open Set

theorem Intersection_A_B :
  let A := {x : ℝ | 2 * x + 1 < 3}
  let B := {x : ℝ | -3 < x ∧ x < 2}
  A ∩ B = {x : ℝ | -3 < x ∧ x < 1} := by
  let A := {x : ℝ | 2 * x + 1 < 3}
  let B := {x : ℝ | -3 < x ∧ x < 2}
  show A ∩ B = {x : ℝ | -3 < x ∧ x < 1}
  sorry

end Intersection_A_B_l150_150747


namespace num_divisible_by_2_3_5_7_lt_500_l150_150215

theorem num_divisible_by_2_3_5_7_lt_500 : 
  (finset.count (λ n : ℕ, n < 500 ∧ (n % 2 = 0 ∧ n % 3 = 0 ∧ n % 5 = 0 ∧ n % 7 = 0)) (finset.range 500)) = 2 :=
sorry

end num_divisible_by_2_3_5_7_lt_500_l150_150215


namespace infinite_series_value_l150_150926

noncomputable def series_term (n : ℕ) : ℝ :=
  (n^5 + 2*n^3 + 5*n^2 + 15*n + 15) / (2^n * (n^5 + 5))

noncomputable def sum_series : ℝ :=
  ∑' (n : ℕ) in finset.Icc 3 ∞, series_term n

theorem infinite_series_value :
  sum_series = 1 / 4 :=
by sorry

end infinite_series_value_l150_150926


namespace infinite_solutions_l150_150362

theorem infinite_solutions (x0 y0 D : ℤ) (h : x0^2 - D * y0^2 = -1) (hD : ∀ z : ℤ, D ≠ z^2) :
  ∃ (f : ℕ → ℤ × ℤ), (∀ k : ℕ, k % 2 = 1 → let (x, y) := f k in x^2 - D * y^2 = -1) ∧ 
  ∀ n m, n ≠ m → f n ≠ f m :=
sorry

end infinite_solutions_l150_150362


namespace bus_full_problem_l150_150871

theorem bus_full_problem
      (cap : ℕ := 80)
      (first_pickup_ratio : ℚ := 3/5)
      (second_pickup_exit : ℕ := 15)
      (waiting_people : ℕ := 50) :
      waiting_people - (cap - (first_pickup_ratio * cap - second_pickup_exit)) = 3 := by
  sorry

end bus_full_problem_l150_150871


namespace girls_count_l150_150006

theorem girls_count (G B : ℕ) (hB : B = 4) (h_alt : ∀ (G B : ℕ), 
  (∃ (f : ℕ → ℕ), ∀ g, f g ∈ {1, 2, 3, 4!} →  ∃ f (G = 5)), ∀ b, f b ∈ {4!}) (h_total : 4! * G! := 2880 ) : 
  G = 5 :=
  
  sorry
  
end girls_count_l150_150006


namespace magnitude_z_is_sqrt_2_l150_150586

open Complex

noncomputable def z (x y : ℝ) : ℂ := x + y * I

theorem magnitude_z_is_sqrt_2 (x y : ℝ) (h1 : (2 * x) / (1 - I) = 1 + y * I) : abs (z x y) = Real.sqrt 2 :=
by
  -- You would fill in the proof steps here based on the problem's solution.
  sorry

end magnitude_z_is_sqrt_2_l150_150586


namespace unique_line_through_point_parallel_to_line_l150_150624

open Set

variables {α : Type*} [AffineSpace ℝ α]
variables (a : Line ℝ α) (α : AffineSubspace ℝ α) (P : α.carrier)

-- Line a is parallel to plane α
def line_parallel_plane : Prop := a.parallel α.carrier

-- Point P is on plane α
def point_on_plane : Prop := P ∈ α.carrier

theorem unique_line_through_point_parallel_to_line
  (hp : point_on_plane α P)
  (hl : line_parallel_plane a α) :
  ∃! l : Line ℝ α, l.parallel a ∧ P ∈ l.carrier ∧ l ⊆ α.carrier := sorry

end unique_line_through_point_parallel_to_line_l150_150624


namespace snow_probability_at_least_once_l150_150809

theorem snow_probability_at_least_once :
  let p := 3 / 4
  let prob_no_snow_single_day := 1 - p
  let prob_no_snow_all_days := prob_no_snow_single_day ^ 5
  let prob_snow_at_least_once := 1 - prob_no_snow_all_days
  prob_snow_at_least_once = 1023 / 1024 :=
by
  sorry

end snow_probability_at_least_once_l150_150809


namespace odd_function_property_l150_150183

theorem odd_function_property (f : ℝ → ℝ) 
  (h_odd : ∀ x : ℝ, f (-x) = -f x) 
  (h_pos : ∀ x : ℝ, 0 < x → f x = x - 1) :
  ∀ x : ℝ, x < 0 → f x * f (-x) ≤ 0 :=
by
  intro x hx
  have h1 : f (-x) = -(x - 1), from h_pos (-x) (neg_pos.mpr hx)
  have h2 : f x = -f (-x), from h_odd x
  rw [h1, h2]
  simp
  sorry

end odd_function_property_l150_150183


namespace quadratic_root_value_l150_150533

theorem quadratic_root_value :
  ∃ p : ℝ, (∀ a b c : ℝ, a = 5 ∧ b = 7 ∧ p = c →
  (∀ x : ℂ, x = ((-b + complex.I * real.sqrt 231) / 10) ∨ x = ((-b - complex.I * real.sqrt 231) / 10))) → 
  p = 14 :=
begin
  sorry
end

end quadratic_root_value_l150_150533


namespace probability_1_lt_X_lt_4_l150_150996

noncomputable def probability_X_i (i a : ℕ) := i / (3 * a : ℚ)

theorem probability_1_lt_X_lt_4 (a : ℕ) (h_dist : ∀ i, i ∈ {1, 2, 3, 4, 5} → probability_X_i i a = i / (3 * a : ℚ)) :
  probability_X_i 2 a + probability_X_i 3 a = 1 / 3 :=
by
  sorry

end probability_1_lt_X_lt_4_l150_150996


namespace face_value_of_shares_l150_150884

theorem face_value_of_shares :
  ∃ F : ℝ, 
    (let investment : ℝ := 4940
         quotation_price : ℝ := 9.50
         dividend_rate : ℝ := 0.14
         annual_income : ℝ := 728 in
     annual_income = dividend_rate * F * (investment / quotation_price)) → F = 10 :=
by
   sorry

end face_value_of_shares_l150_150884


namespace count_valid_a_values_l150_150955

def is_perfect_square (n : ℕ) : Prop :=
∃ k : ℕ, k * k = n

def valid_a_values (a : ℕ) : Prop :=
1 ≤ a ∧ a ≤ 100 ∧ is_perfect_square (16 * a + 9)

theorem count_valid_a_values :
  ∃ N : ℕ, N = Nat.card {a : ℕ | valid_a_values a} := sorry

end count_valid_a_values_l150_150955


namespace max_k_sum_odd_numbers_l150_150568

theorem max_k_sum_odd_numbers (k : ℕ) (h : k ≤ 51) :
  ∑ i in (finset.range k).map (λ i, 2 * i + 1) = 1949 → k = 44 :=
begin
  sorry
end

end max_k_sum_odd_numbers_l150_150568


namespace john_pays_per_year_l150_150715

-- Define the costs and insurance parameters.
def cost_per_epipen : ℝ := 500
def insurance_coverage : ℝ := 0.75

-- Number of months in a year.
def months_in_year : ℕ := 12

-- Number of months each EpiPen lasts.
def months_per_epipen : ℕ := 6

-- Amount covered by insurance for each EpiPen.
def insurance_amount (cost : ℝ) (coverage : ℝ) : ℝ :=
  cost * coverage

-- Amount John pays after insurance for each EpiPen.
def amount_john_pays_per_epipen (cost : ℝ) (covered: ℝ) : ℝ :=
  cost - covered

-- Number of EpiPens John needs per year.
def epipens_per_year (months_in_year : ℕ) (months_per_epipen : ℕ) : ℕ :=
  months_in_year / months_per_epipen

-- Total amount John pays per year.
def total_amount_john_pays_per_year (amount_per_epipen : ℝ) (epipens_per_year : ℕ) : ℝ :=
  amount_per_epipen * epipens_per_year

-- Theorem to prove the correct answer.
theorem john_pays_per_year :
  total_amount_john_pays_per_year (amount_john_pays_per_epipen cost_per_epipen (insurance_amount cost_per_epipen insurance_coverage)) (epipens_per_year months_in_year months_per_epipen) = 250 := 
by
  sorry

end john_pays_per_year_l150_150715


namespace complex_sum_diff_squares_eq_find_abs_diff_given_magnitudes_l150_150313

noncomputable theory

open Complex

variables (z1 z2 : ℂ)

theorem complex_sum_diff_squares_eq :
  |z1 + z2|^2 + |z1 - z2|^2 = 2 * |z1|^2 + 2 * |z2|^2 :=
by sorry

theorem find_abs_diff_given_magnitudes (hz1 : |z1| = 3) (hz2 : |z2| = 5) (h_sum : |z1 + z2| = 6) :
  |z1 - z2| = 4 * Real.sqrt 2 :=
by sorry

end complex_sum_diff_squares_eq_find_abs_diff_given_magnitudes_l150_150313


namespace sum_f_a_eq_2017_l150_150617

noncomputable def a (n : ℕ) : ℝ := sorry -- a geometric sequence
noncomputable def f (x : ℝ) : ℝ := 2 / (1 + x^2)

theorem sum_f_a_eq_2017 :
  (a 1) * (a 2017) = 1 →
  ∑ i in Finset.range 2017, f (a (i + 1)) = 2017 :=
begin
  intro h1,
  sorry
end

end sum_f_a_eq_2017_l150_150617


namespace arithmetic_sequence_sol_l150_150588

def sequence_arithmetic (a : ℕ → ℕ) (A B C : ℕ → ℕ) :=
  ∀ n, B n - A n = C n - B n

def sequence_conditions_arithmetic (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ a 2 = 5 ∧ sequence_arithmetic a (λ n => ∑ i in List.range (n + 1), a i) 
  (λ n => ∑ i in List.range' 1 (n + 1), a (i+1)) 
  (λ n => ∑ i in List.range' 2 (n + 1), a (i+2))

theorem arithmetic_sequence_sol (a : ℕ → ℕ) :
  sequence_conditions_arithmetic a → ∀ n : ℕ, a n = 4 * n - 3 :=
by
  sorry

end arithmetic_sequence_sol_l150_150588


namespace sector_area_ratio_l150_150762

theorem sector_area_ratio (C D E: Point) (O: Circle) (A B : Point) 
  (h1: ∠AOC = 40) (h2: ∠DOB = 60) (h3: ∠BOE = 25) (h4 : A, B on diameter O) : 
  ratio (sector COD ∪ sector DOE) (area O) = 7 / 24 := by 
  sorry

end sector_area_ratio_l150_150762


namespace john_annual_payment_l150_150714

open Real

-- Definitions extracted from the problem:
def epipen_cost : ℝ := 500
def insurance_coverage : ℝ := 0.75
def epipen_frequency_per_year : ℕ := 2
def john_payment_per_epipen : ℝ := epipen_cost * (1 - insurance_coverage)

-- The statement to be proved:
theorem john_annual_payment :
  john_payment_per_epipen * epipen_frequency_per_year = 250 :=
by
  sorry

end john_annual_payment_l150_150714


namespace range_of_m_correct_l150_150240

noncomputable def range_of_m (m : ℝ) : Prop :=
∃ x : ℝ, 2^(2 * x) + (m^2 - 2 * m - 5) * 2^x + 1 = 0

theorem range_of_m_correct : {m : ℝ | range_of_m m} = set.Icc (-1 : ℝ) 3 :=
by
  sorry

end range_of_m_correct_l150_150240


namespace systematic_sampling_removal_count_l150_150016

theorem systematic_sampling_removal_count :
  ∀ (N n : ℕ), N = 3204 ∧ n = 80 → N % n = 4 := 
by
  sorry

end systematic_sampling_removal_count_l150_150016


namespace infinite_zeros_sin_log_l150_150660

theorem infinite_zeros_sin_log : ∀ x : ℝ, 0 < x ∧ x < 1 → ∃ (seq : ℕ → ℝ), (strict_mono seq) ∧ (∀ n, g(seq n) = 0) :=
by
  let g : ℝ → ℝ := λ x, Real.sin (Real.log x)
  sorry

end infinite_zeros_sin_log_l150_150660


namespace eval_sum_l150_150134

theorem eval_sum : 
  (4 / 3 + 8 / 9 + 16 / 27 + 32 / 81 + 64 / 243 + 128 / 729 - 8 : ℚ) = -1 / 729 :=
by
  sorry

end eval_sum_l150_150134


namespace circle_radius_is_six_l150_150814

open Real

theorem circle_radius_is_six
  (r : ℝ)
  (h : 2 * 3 * 2 * π * r = 2 * π * r^2) :
  r = 6 := sorry

end circle_radius_is_six_l150_150814


namespace circle_area_percentage_decrease_l150_150450

theorem circle_area_percentage_decrease (r : ℝ) (A : ℝ := Real.pi * r^2) 
  (r' : ℝ := 0.5 * r) (A' : ℝ := Real.pi * (r')^2) :
  (A - A') / A * 100 = 75 := 
by
  sorry

end circle_area_percentage_decrease_l150_150450


namespace second_experimental_point_is_correct_l150_150058

-- Define the temperature range
def lower_bound : ℝ := 1400
def upper_bound : ℝ := 1600

-- Define the golden ratio constant
def golden_ratio : ℝ := 0.618

-- Calculate the first experimental point using 0.618 method
def first_point : ℝ := lower_bound + golden_ratio * (upper_bound - lower_bound)

-- Calculate the second experimental point
def second_point : ℝ := upper_bound - (first_point - lower_bound)

-- Theorem stating the calculated second experimental point equals 1476.4
theorem second_experimental_point_is_correct :
  second_point = 1476.4 := by
  sorry

end second_experimental_point_is_correct_l150_150058


namespace suitable_altitude_range_l150_150064

theorem suitable_altitude_range :
  ∀ (temperature_at_base : ℝ) (temp_decrease_per_100m : ℝ) (suitable_temp_low : ℝ) (suitable_temp_high : ℝ) (altitude_at_base : ℝ),
  (22 = temperature_at_base) →
  (0.5 = temp_decrease_per_100m) →
  (18 = suitable_temp_low) →
  (20 = suitable_temp_high) →
  (0 = altitude_at_base) →
  400 ≤ ((temperature_at_base - suitable_temp_high) / temp_decrease_per_100m * 100) ∧ ((temperature_at_base - suitable_temp_low) / temp_decrease_per_100m * 100) ≤ 800 :=
by
  intros temperature_at_base temp_decrease_per_100m suitable_temp_low suitable_temp_high altitude_at_base
  intro h1 h2 h3 h4 h5
  sorry

end suitable_altitude_range_l150_150064


namespace distance_preserving_functions_count_l150_150287

noncomputable def number_of_distance_preserving_functions : ℕ :=
  12 * (Nat.factorial 1000)^2

theorem distance_preserving_functions_count :
  ∀ (X : set (fin 2000 → ℕ)),
    (∀ (a : fin 1000 → ℕ), (∀ i, a i ∈ {0, 1, 2}) ∧
      (∀ (b : fin 1000 → ℕ), (∀ i, b i ∈ {0, 1}) ∧
        ∀ (a b : fin 2000 → ℕ), (∀ i, a i ≠ b i → a i ≠ b i) → 
          (∑ i, if a i ≠ b i then 1 else 0) =
          (∑ i, if f(a) i ≠ f(b) i then 1 else 0))) →
    number_of_distance_preserving_functions = 12 * (Nat.factorial 1000)^2 := sorry

end distance_preserving_functions_count_l150_150287


namespace lumber_cut_length_l150_150962

-- Define lengths of the pieces
def length_W : ℝ := 5
def length_X : ℝ := 3
def length_Y : ℝ := 5
def length_Z : ℝ := 4

-- Define distances from line M to the left end of the pieces
def distance_X : ℝ := 3
def distance_Y : ℝ := 2
def distance_Z : ℝ := 1.5

-- Define the total length of the pieces
def total_length : ℝ := 17

-- Define the length per side when cut by L
def length_per_side : ℝ := 8.5

theorem lumber_cut_length :
    (∃ (d : ℝ), 4 * d - 6.5 = 8.5 ∧ d = 3.75) :=
by
  sorry

end lumber_cut_length_l150_150962


namespace necessary_condition_for_monotonic_decrease_l150_150636

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 2 * x - 2 / x - a * log x

theorem necessary_condition_for_monotonic_decrease (a : ℝ) :
  (∀ (x : ℝ), 1 < x ∧ x < 2 → deriv (λ x, f x a) x < 0) → a > 5 :=
sorry

end necessary_condition_for_monotonic_decrease_l150_150636


namespace correct_match_results_l150_150854

-- Define the teams in the league
inductive Team
| Scotland : Team
| England  : Team
| Wales    : Team
| Ireland  : Team

-- Define a match result for a pair of teams
structure MatchResult where
  team1 : Team
  team2 : Team
  goals1 : ℕ
  goals2 : ℕ

def scotland_vs_england : MatchResult := {
  team1 := Team.Scotland,
  team2 := Team.England,
  goals1 := 3,
  goals2 := 0
}

-- All possible match results
def england_vs_ireland : MatchResult := {
  team1 := Team.England,
  team2 := Team.Ireland,
  goals1 := 1,
  goals2 := 0
}

def wales_vs_england : MatchResult := {
  team1 := Team.Wales,
  team2 := Team.England,
  goals1 := 1,
  goals2 := 1
}

def wales_vs_ireland : MatchResult := {
  team1 := Team.Wales,
  team2 := Team.Ireland,
  goals1 := 2,
  goals2 := 1
}

def scotland_vs_ireland : MatchResult := {
  team1 := Team.Scotland,
  team2 := Team.Ireland,
  goals1 := 2,
  goals2 := 0
}

theorem correct_match_results : 
  (england_vs_ireland.goals1 = 1 ∧ england_vs_ireland.goals2 = 0) ∧
  (wales_vs_england.goals1 = 1 ∧ wales_vs_england.goals2 = 1) ∧
  (scotland_vs_england.goals1 = 3 ∧ scotland_vs_england.goals2 = 0) ∧
  (wales_vs_ireland.goals1 = 2 ∧ wales_vs_ireland.goals2 = 1) ∧
  (scotland_vs_ireland.goals1 = 2 ∧ scotland_vs_ireland.goals2 = 0) :=
by 
  sorry

end correct_match_results_l150_150854


namespace csc_square_value_l150_150152

noncomputable def f (x : ℝ) : ℝ :=
if x = 0 ∨ x = 1 then 0 -- provision for the illegal inputs as defined in the question
else 1/(x / (x - 1))

theorem csc_square_value (t : ℝ) (ht : 0 ≤ t ∧ t ≤ π / 2) :
  f (1 / (Real.sin t)^2) = (Real.cos t)^2 :=
by
  sorry

end csc_square_value_l150_150152


namespace pieces_per_package_l150_150776

-- Define Robin's packages
def numGumPackages := 28
def numCandyPackages := 14

-- Define total number of pieces
def totalPieces := 7

-- Define the total number of packages
def totalPackages := numGumPackages + numCandyPackages

-- Define the expected number of pieces per package as the theorem to prove
theorem pieces_per_package : (totalPieces / totalPackages) = 1/6 := by
  sorry

end pieces_per_package_l150_150776


namespace smallest_sum_l150_150603

theorem smallest_sum (x y : ℕ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x ≠ y) (eq : 1/x + 1/y = 1/10) : x + y = 45 :=
by
  sorry

end smallest_sum_l150_150603


namespace find_least_d_l150_150048

theorem find_least_d :
  ∃ d : ℕ, (d % 7 = 1) ∧ (d % 5 = 2) ∧ (d % 3 = 2) ∧ d = 92 :=
by 
  sorry

end find_least_d_l150_150048


namespace percentage_of_women_lawyers_l150_150061

theorem percentage_of_women_lawyers
  (T : ℝ) 
  (h1 : 0.70 * T = W) 
  (h2 : 0.28 * T = WL) : 
  ((WL / W) * 100 = 40) :=
by
  sorry

end percentage_of_women_lawyers_l150_150061


namespace digit_B_value_l150_150389

theorem digit_B_value (B : ℕ) (h : 4 * 1000 + B * 100 + B * 10 + 2 % 9 = 0):
  B = 6 :=
begin
  sorry
end

end digit_B_value_l150_150389


namespace profit_maximization_problem_l150_150872

-- Step 1: Define the data points and linear function
def data_points : List (ℝ × ℝ) := [(65, 70), (70, 60), (75, 50), (80, 40)]

-- Step 2: Define the linear function between y and x
def linear_function (k b x : ℝ) : ℝ := k * x + b

-- Step 3: Define cost and profit function
def cost_per_kg : ℝ := 60
def profit_function (y x : ℝ) : ℝ := y * (x - cost_per_kg)

-- Step 4: The main problem statement
theorem profit_maximization_problem :
  ∃ (k b : ℝ), 
  (∀ (x₁ x₂ : ℝ), (x₁, y₁) ∈ data_points ∧ (x₂, y₂) ∈ data_points → linear_function k b x₁ = y₁ ∧ linear_function k b x₂ = y₂) ∧
  ∃ (x : ℝ), profit_function (linear_function k b x) x = 600 ∧
  ∀ x : ℝ, -2 * x^2 + 320 * x - 12000 ≤ -2 * 80^2 + 320 * 80 - 12000
  :=
sorry

end profit_maximization_problem_l150_150872


namespace value_of_x_l150_150202

theorem value_of_x (x : ℝ) (M : set ℝ) (hM : M = {-2, 3 * x^2 + 3 * x - 4}) (h : 2 ∈ M) :
  x = 1 ∨ x = -2 :=
by sorry

end value_of_x_l150_150202


namespace smallest_x_plus_y_l150_150606

theorem smallest_x_plus_y 
  (x y : ℕ) 
  (hx_pos : 0 < x) 
  (hy_pos : 0 < y) 
  (hxy_neq : x ≠ y)
  (h_eq : (1/x + 1/y = 1/10)) : x + y = 45 :=
sorry

end smallest_x_plus_y_l150_150606


namespace incorrect_connection_probability_l150_150418

noncomputable def probability_of_incorrect_connection (p : ℝ) : ℝ :=
  let r2 := 1 / 9
  let r3 := (8 / 9) * (1 / 9)
  (3 * p^2 * (1 - p) * r2) + (1 * p^3 * r3)

theorem incorrect_connection_probability : probability_of_incorrect_connection 0.02 = 0.000131 :=
by
  sorry

end incorrect_connection_probability_l150_150418


namespace log_e_2_irrational_l150_150363

theorem log_e_2_irrational (e_transcendental : Transcendental e) :
  irrational (log e 2) :=
begin
  by_contra h,
  obtain ⟨p, q, hq, rfl⟩ := h,
  have h2 : (2 : ℝ) = exp (p / q),
  { rw [← log_exp (p / q), log_eq_self h], },
  have h3 := congr_arg (λ x, x ^ q) h2,
  norm_num at h3,
  have h4 : (2 : ℝ) ^ q = e ^ p,
  { rwa [exp_nat_mul', mul_comm] at h3, },
  let f : polynomial ℝ := polynomial.C (2^q) - polynomial.X ^ p,
  have he : f.eval e = 0,
  { rw [polynomial.eval_sub, polynomial.eval_C, polynomial.eval_X_pow],
    rw [h4, sub_self], },
  exact e_transcendental f he,
end

end log_e_2_irrational_l150_150363


namespace incorrect_connection_probability_l150_150410

theorem incorrect_connection_probability :
  let p := 0.02
  let r2 := 1 / 9
  let r3 := 8 / 81
  let probability_wrong_two_errors := 3 * p^2 * (1 - p) * r2
  let probability_wrong_three_errors := 1 * p^3 * r3
  let total_probability_correct_despite_errors := probability_wrong_two_errors + probability_wrong_three_errors
  let total_probability_incorrect := 1 - total_probability_correct_despite_errors
  ((total_probability_correct_despite_errors ≈ 0.000131) → 
  (total_probability_incorrect ≈ 1 - 0.000131)) :=
by
  sorry

end incorrect_connection_probability_l150_150410


namespace days_in_month_l150_150804

-- The number of days in the month
variable (D : ℕ)

-- The conditions provided in the problem
def mean_daily_profit (D : ℕ) := 350
def mean_first_fifteen_days := 225
def mean_last_fifteen_days := 475
def total_profit := mean_first_fifteen_days * 15 + mean_last_fifteen_days * 15

-- The Lean statement to prove the number of days in the month
theorem days_in_month : D = 30 :=
by
  -- mean_daily_profit(D) * D should be equal to total_profit
  have h : mean_daily_profit D * D = total_profit := sorry
  -- solve for D
  sorry

end days_in_month_l150_150804


namespace triangle_ratio_l150_150270

noncomputable def vector_ratio (p q r : ℝ^3) : ℝ :=
  let g := (1 / 5) • p + (4 / 5) • q 
  let h := (1 / 5) • q + (4 / 5) • r
  let j := 5 • h - 4 • g
  ( (h - g).norm / (j - h).norm )

theorem triangle_ratio (p q r : ℝ^3) :
  ∃ G H J : ℝ^3,
    G = (1 / 5) • p + (4 / 5) • q ∧
    H = (1 / 5) • q + (4 / 5) • r ∧
    J = 5 • H - 4 • G ∧
    vector_ratio p q r = 4 := 
by sorry

end triangle_ratio_l150_150270


namespace smallest_five_digit_in_pascal_l150_150435

-- Define the conditions
def pascal_triangle_increases (n k : ℕ) : Prop := 
  ∀ (r ≥ n) (c ≥ k), c ≤ r → ∃ (x : ℕ), x >= Nat.choose r c

def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

-- State the proof problem and the expected answer
theorem smallest_five_digit_in_pascal :
  (∃ (n k : ℕ), binomial_coefficient n k = 10000) ∧ (∀ (m l : ℕ), binomial_coefficient m l = 10000 → n ≤ m) := sorry

end smallest_five_digit_in_pascal_l150_150435


namespace single_digit_y_for_divisibility_by_6_l150_150532

theorem single_digit_y_for_divisibility_by_6 :
  ∃ y : ℕ, y < 10 ∧ (divisible_by_6 (62160 + 100 * y)) ∧ y = 3 :=
by
  sorry

def divisible_by_6 (n : ℕ) : Prop :=
  (n % 2 = 0) ∧ (n % 3 = 0)

end single_digit_y_for_divisibility_by_6_l150_150532


namespace count_integers_divisible_by_2_3_5_7_l150_150226

theorem count_integers_divisible_by_2_3_5_7 :
  ∃ n : ℕ, (∀ k : ℕ, k < 500 → (k % 2 = 0 ∧ k % 3 = 0 ∧ k % 5 = 0 ∧ k % 7 = 0) → k ≠ n → k < 500 ∧ k > 0) ∧
  (n = 2) :=
by
  sorry

end count_integers_divisible_by_2_3_5_7_l150_150226


namespace value_of_expression_l150_150025

open Real

theorem value_of_expression :
  let a := 1296
  let b := 4096
  let ans := 800
  (a = 6^4) → (b = 2^12) →
  ((a ^ (log b / log 6)) ^ (1/6)) ≈ ans :=
by
  intros h1 h2
  rw [h1, h2]
  -- Continue the proof of equivalence of (6^4 ^ (log (2^12) / log 6)) ^ (1/6) to 800
  sorry

end value_of_expression_l150_150025


namespace amit_work_days_l150_150904

variable (x : ℕ)

theorem amit_work_days
  (ananthu_rate : ℚ := 1/30) -- Ananthu's work rate is 1/30
  (amit_days : ℕ := 3) -- Amit worked for 3 days
  (ananthu_days : ℕ := 24) -- Ananthu worked for remaining 24 days
  (total_days : ℕ := 27) -- Total work completed in 27 days
  (amit_work: ℚ := amit_days * 1/x) -- Amit's work rate
  (ananthu_work: ℚ := ananthu_days * ananthu_rate) -- Ananthu's work rate
  (total_work : ℚ := 1) -- Total work completed  
  : 3 * (1/x) + 24 * (1/30) = 1 ↔ x = 15 := 
by
  sorry

end amit_work_days_l150_150904


namespace binomial_theorem_example_l150_150231

theorem binomial_theorem_example :
  let a : ℕ → ℕ → ℤ := λ n k, (-2)^k * nat.choose n k in
  (a 5 3) / (a 5 2) = -2 :=
by
  sorry

end binomial_theorem_example_l150_150231


namespace find_quantities_max_pendants_l150_150369

noncomputable def num_items (x y : ℕ) : Prop :=
  x + y = 180 ∧ 80 * x + 50 * y = 11400

theorem find_quantities : ∃ (x y : ℕ), num_items x y ∧ x = 80 ∧ y = 100 :=
by
  have h1 : 80 + 100 = 180, by norm_num
  have h2 : 80 * 80 + 50 * 100 = 11400, by norm_num
  exact ⟨80, 100, ⟨h1, h2⟩, rfl, rfl⟩

noncomputable def profit_formula (m : ℕ) : Prop :=
  (180 - m) * 20 + m * 10 ≥ 2900

theorem max_pendants (m : ℕ) : ∃ m, profit_formula m ∧ m = 70 :=
by
  have h : (180 - 70) * 20 + 70 * 10 = 2900, by norm_num
  exact ⟨70, h⟩

end find_quantities_max_pendants_l150_150369


namespace tangent_line_equation_at_point_l150_150949

-- Definitions from the problem in a)
def curve (x : ℝ) : ℝ := x^3 - 2 * x + 1
def point : ℝ × ℝ := (1, 0)

-- Lean statement for the proof problem
theorem tangent_line_equation_at_point :
  ∃ m b : ℝ, (∀ x : ℝ, curve x = m * x + b) ∧ point.1 - point.2 - 1 = 0 :=
begin
  sorry
end

end tangent_line_equation_at_point_l150_150949


namespace moving_circle_passes_through_focus_l150_150457

-- Given conditions
def is_on_parabola (x y : ℝ) : Prop :=
  y^2 = 8 * x

def is_tangent_to_line (circle_center_x : ℝ) : Prop :=
  circle_center_x + 2 = 0

-- Prove that the point (2,0) lies on the moving circle
theorem moving_circle_passes_through_focus (circle_center_x circle_center_y : ℝ) :
  is_on_parabola circle_center_x circle_center_y →
  is_tangent_to_line circle_center_x →
  (circle_center_x - 2)^2 + circle_center_y^2 = (circle_center_x + 2)^2 :=
by
  -- Proof skipped with sorry.
  sorry

end moving_circle_passes_through_focus_l150_150457


namespace systematic_sampling_number_l150_150494

theorem systematic_sampling_number (a d n: ℕ) (h_a : a = 3) (h_d : d = 10) (h_n : n = 2) : 
  let a_n := a + (n - 1) * d in 11 ≤ a_n ∧ a_n ≤ 20 ∧ a_n = 13 :=
by
  sorry

end systematic_sampling_number_l150_150494


namespace proof_d_minus_e_l150_150954

def a_n (n : ℕ) (h : 1 < n) := 1 / (Real.log 1001 / Real.log n)

def d := a_n 2 (by norm_num) + a_n 3 (by norm_num) + a_n 4 (by norm_num) + a_n 5 (by norm_num)

def e := a_n 10 (by norm_num) + a_n 11 (by norm_num) + a_n 12 (by norm_num) + a_n 13 (by norm_num) + a_n 14 (by norm_num)

theorem proof_d_minus_e : d - e = -1 - Real.log 3 / Real.log 1001 := by
  sorry

end proof_d_minus_e_l150_150954


namespace smallest_solution_of_eq_l150_150560

theorem smallest_solution_of_eq (x : ℝ) (h : x^4 - 64*x^2 + 576 = 0) : x = -2*real.sqrt(6) ∨ x = 2*real.sqrt(6) ∨ x = -2*real.sqrt(10) ∨ x = 2*real.sqrt(10) :=
by sorry

end smallest_solution_of_eq_l150_150560


namespace find_k_l150_150009

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

noncomputable def k (a b c d : V) :=
  if a + b + c + d = 0 then 0 else sorry

theorem find_k (a b c d : V) (h : a + b + c + d = 0) :
  k a b c d = 0 :=
by
  dsimp [k]
  rw if_pos h
  refl

end find_k_l150_150009


namespace distinct_values_f2014_f2_f2016_l150_150289

noncomputable def f (n : ℕ) : ℕ := sorry

axiom f_divides_pow {n : ℕ} : f(n) ∣ n^2016
axiom f_pythagorean_a2b2c2 {a b c : ℕ} (h : a^2 + b^2 = c^2) : f(a) * f(b) = f(c)

theorem distinct_values_f2014_f2_f2016 : ∃ k > 0, k = (2^2017 - 1) :=
  sorry

end distinct_values_f2014_f2_f2016_l150_150289


namespace integral_sin_cos_eq_l150_150855

theorem integral_sin_cos_eq (x : ℝ) :
  ∫ (∫ \frac{\sin x - \cos x}{(\cos x + \sin x) ^ 5} dx) = -\frac{1}{4 * (cos x + sin x) ^ 4} + C :=
by sorry

end integral_sin_cos_eq_l150_150855


namespace area_PQR_ge_area_ABC_l150_150725

-- Given
variable {A B C M P Q R : Type}
variable [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C]
variable (AMC AMB BMC : ℝ)
variable [Decidable (AMC = 90)] [Decidable (AMB = 150)] [Decidable (BMC = 120)]

-- Conditions
def angle_AMC := AMC = 90
def angle_AMB := AMB = 150
def angle_BMC := BMC = 120

-- Circumcenters
def circumcenter_AMC := P
def circumcenter_AMB := Q
def circumcenter_BMC := R

-- Proof statement
theorem area_PQR_ge_area_ABC
  (h_AMC: angle_AMC)
  (h_AMB: angle_AMB)
  (h_BMC: angle_BMC)
  (h_P: circumcenter_AMC)
  (h_Q: circumcenter_AMB)
  (h_R: circumcenter_BMC):
  area ΔPQR ≥ area ΔABC :=
by sorry

end area_PQR_ge_area_ABC_l150_150725


namespace sin_A_eq_a_eq_3_l150_150684

variable (A B C a b c : ℝ)
variable (S : ℝ) (h_angle_sum : A + B + C = π)
variable (h1 : 3 * a * Real.sin C = c * Real.cos A)
variable (h2 : B = π / 4)
variable (h3 : (1 / 2) * a * c * Real.sin B = 9)

theorem sin_A_eq ((h1 : 3 * a * Real.sin C = c * Real.cos A) : Real.sin A = sqrt 10 / 10 := sorry

theorem a_eq_3 ((h3 : (1 / 2) * a * c * Real.sin (π / 4) = 9) : a = 3 := sorry

end sin_A_eq_a_eq_3_l150_150684


namespace power_of_power_calc_3_squared_4_l150_150515

theorem power_of_power (a : ℕ) (m n : ℕ) : (a ^ m) ^ n = a ^ (m * n) := by
  sorry

theorem calc_3_squared_4 : (3^2)^4 = 6561 := by
  calc
    (3^2)^4 = 3^(2 * 4) : by rw power_of_power
            ... = 3^8   : by rw [← mul_assoc, nat.mul_comm]
            ... = 6561  : by norm_num

end power_of_power_calc_3_squared_4_l150_150515


namespace cross_section_prism_in_sphere_l150_150890

noncomputable def cross_section_area 
  (a R : ℝ) 
  (h1 : a > 0) 
  (h2 : R > 0) 
  (h3 : a < 2 * R) : ℝ :=
  a * Real.sqrt (4 * R^2 - a^2)

theorem cross_section_prism_in_sphere 
  (a R : ℝ) 
  (h1 : a > 0) 
  (h2 : R > 0) 
  (h3 : a < 2 * R) :
  cross_section_area a R h1 h2 h3 = a * Real.sqrt (4 * R^2 - a^2) := 
  by
    sorry

end cross_section_prism_in_sphere_l150_150890


namespace range_of_k_not_monotonic_l150_150678

noncomputable def f (x : ℝ) : ℝ := x^2 - (1 / 2) * Real.log x + 1

theorem range_of_k_not_monotonic : 
  ∀ k : ℝ, 1 ≤ k ∧ k < (3 / 2) ↔ ∃ x : ℝ, f' x = 0 ∧ x ∈ (k - 1, k + 1)
  := by
    sorry

end range_of_k_not_monotonic_l150_150678


namespace end_behavior_of_g_l150_150118

noncomputable def g (x : ℝ) : ℝ := -3 * x^4 + 5 * x^3 - 4

theorem end_behavior_of_g :
  (filter.tendsto g filter.at_top filter.at_bot) ∧ 
  (filter.tendsto g filter.at_bot filter.at_bot) :=
by
  sorry

end end_behavior_of_g_l150_150118


namespace statement_correct_l150_150958

noncomputable def curve (x y : ℝ) : Prop :=
  x >= 0 ∧ y >= 0 ∧ sqrt x + sqrt y = 1

theorem statement_correct :
  (¬ ∃ x y : ℝ, curve x y ∧ sqrt (x^2 + y^2) = sqrt 2 / 2) ∧
  (∃ area : ℝ, area ≤ 1/2) := by sorry

end statement_correct_l150_150958


namespace find_n_values_l150_150462

noncomputable def board_exists (n : ℕ) : Prop :=
  n ≥ 3 ∧ ∃ (f : Fin (n*n) → Fin (n+1)), 
  ∀ i j : Fin (n-1), 
  ∀ k l : Fin (n-1), 
  (i, j) ≠ (k, l) → 
  f (i*n + j) + f ((i+1)*n + j) + f (i*n + (j+1)) + f ((i+1)*n + (j+1)) ≠ 
  f (k*n + l) + f ((k+1)*n + l) + f (k*n + (l+1)) + f ((k+1)*n + (l+1))

theorem find_n_values : { n : ℕ | board_exists n } = {3, 4, 5, 6} := 
sorry

end find_n_values_l150_150462


namespace radius_of_circle_l150_150186

theorem radius_of_circle : 
  ∀ (r : ℝ),
  (∀ (x y : ℝ), (x - 3)^2 + (y + 5)^2 = r^2 → (∃ x y : ℝ, 4 * x - 3 * y - 2 = 0 ∧ ((x - 3)^2 + (y + 5)^2 = r^2 ∧ 1 = (abs (4 * 3 - 3 * (-5) - 2) / sqrt ((4^2 + (-3)^2)))) )) →
  r = 4 :=
by
  intros r h
  sorry

end radius_of_circle_l150_150186


namespace minimum_BC_length_l150_150860

-- Define the lengths of the sides
def AB := 7 : ℝ
def AC := 15 : ℝ
def EC := 10 : ℝ
def BE := 25 : ℝ

-- Prove that the minimum possible length for BC is 15 cm
theorem minimum_BC_length (BC : ℝ) : AB = 7 → AC = 15 → EC = 10 → BE = 25 → BC ≥ 15 :=
by
  intros hAB hAC hEC hBE
  have ABC_ineq := calc
    BC > AC - AB : by linarith [hAC, hAB]
    ... = 8 : by norm_num
  have EBC_ineq := calc
    BC > BE - EC : by linarith [hBE, hEC]
    ... = 15 : by norm_num
  exact max_le_iff.mpr (and.intro ABC_ineq EBC_ineq)

end minimum_BC_length_l150_150860


namespace increase_in_average_l150_150460

variable (average_before : ℝ)

-- Conditions from the problem
def total_runs_after_10_innings : ℝ := 10 * average_before
def condition_average_after_11th_inning : Prop := (total_runs_after_10_innings + 80) / 11 = 30

-- Statement to prove
theorem increase_in_average (h : condition_average_after_11th_inning average_before) : 
  let average_before_11 := average_before in
  let total_runs_after_11 := total_runs_after_10_innings + 80 in
  let average_after := total_runs_after_11 / 11 in
  average_after - average_before_11 = 5 := 
by sorry

end increase_in_average_l150_150460


namespace jack_pays_back_l150_150278

-- conditions in the problem 
def principal : ℝ := 1200
def interest_rate : ℝ := 0.1

-- the theorem statement equivalent to the question and correct answer
theorem jack_pays_back (principal_interest: principal * interest_rate) (total_amount: principal + principal_interest) : total_amount = 1320 :=
by
  sorry

end jack_pays_back_l150_150278


namespace number_of_dolls_at_discounted_price_l150_150124

def original_price : ℝ := 4
def initial_dolls : ℕ := 15
def discount_rate : ℝ := 0.2
def savings : ℝ := original_price * initial_dolls
def discounted_price : ℝ := original_price - (discount_rate * original_price)
def number_dolls (savings discounted_price : ℝ) : ℕ := int.floor (savings / discounted_price).to_real

theorem number_of_dolls_at_discounted_price : number_dolls savings discounted_price = 18 :=
by
  sorry

end number_of_dolls_at_discounted_price_l150_150124


namespace angle_ACB_is_90_l150_150018

-- Define the relevant elements and conditions of the problem in Lean 4

-- Given an arbitrary triangle ABC
variables {A B C D E F : Type} [has_angle A B C]

-- Conditions
def triangle_ABC : Prop := is_right_angle A C B ∧ dist A B = 3 * dist A C
def point_D_on_AB : Prop := is_on D (segment A B)
def point_E_on_BC : Prop := is_on E (segment B C)
def angle_BAE_eq_ACD : Prop := angle A B E = angle A C D
def intersection_F_AE_CD : Prop := F = intersection (line A E) (line C D)
def triangle_CFE_equilateral : Prop := is_equilateral C F E

-- Theorem to be proved
theorem angle_ACB_is_90 :
  triangle_ABC → point_D_on_AB → point_E_on_BC → angle_BAE_eq_ACD → intersection_F_AE_CD → triangle_CFE_equilateral → angle A C B = 90 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end angle_ACB_is_90_l150_150018


namespace matrix_sum_100_l150_150441

def matrix_sum (n : ℕ) : ℕ :=
  ∑ i in Finset.range n, ∑ j in Finset.range n, (i + j + 1)

theorem matrix_sum_100 :
  matrix_sum 100 = 1000000 := by
  sorry

end matrix_sum_100_l150_150441


namespace binomial_theorem_expansion_l150_150737

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
nat.choose n k

theorem binomial_theorem_expansion (n : ℕ) :
  ∑ k in finset.range (n + 1), (-1) ^ k * binomial_coefficient n k * 2 ^ (n - k) = 1 :=
by
  sorry

end binomial_theorem_expansion_l150_150737


namespace contribution_per_person_l150_150358

-- Define constants for the given conditions
def total_price : ℕ := 67
def coupon : ℕ := 4
def number_of_people : ℕ := 3

-- The theorem to prove
theorem contribution_per_person : (total_price - coupon) / number_of_people = 21 :=
by 
  -- The proof is omitted for this exercise
  sorry

end contribution_per_person_l150_150358


namespace ratio_a_b_eq_neg_one_fifth_l150_150645

theorem ratio_a_b_eq_neg_one_fifth (x y a b : ℝ) (hb_ne_zero : b ≠ 0) 
    (h1 : 4 * x - 2 * y = a) (h2 : 5 * y - 10 * x = b) : a / b = -1 / 5 :=
by {
  sorry
}

end ratio_a_b_eq_neg_one_fifth_l150_150645


namespace money_conditions_l150_150155

theorem money_conditions (a b : ℝ) (h1 : 4 * a - b > 32) (h2 : 2 * a + b = 26) : 
  a > 9.67 ∧ b < 6.66 := 
sorry

end money_conditions_l150_150155


namespace net_progress_l150_150071

-- Define the conditions as properties
def lost_yards : ℕ := 5
def gained_yards : ℕ := 10

-- Prove that the team's net progress is 5 yards
theorem net_progress : (gained_yards - lost_yards) = 5 :=
by
  sorry

end net_progress_l150_150071


namespace kim_gets_change_of_5_l150_150723

noncomputable def meal_cost : ℝ := 10
noncomputable def drink_cost : ℝ := 2.5
noncomputable def tip_rate : ℝ := 0.20
noncomputable def payment : ℝ := 20
noncomputable def total_cost_before_tip := meal_cost + drink_cost
noncomputable def tip := tip_rate * total_cost_before_tip
noncomputable def total_cost_with_tip := total_cost_before_tip + tip
noncomputable def change := payment - total_cost_with_tip

theorem kim_gets_change_of_5 : change = 5 := by
  sorry

end kim_gets_change_of_5_l150_150723


namespace probability_one_each_from_drawer_l150_150661

theorem probability_one_each_from_drawer :
  let total_shirts := 6
  let total_shorts := 7
  let total_socks := 8
  let total_hats := 3
  let total_clothing := total_shirts + total_shorts + total_socks + total_hats
  let total_ways_to_choose_4 := Nat.choose total_clothing 4
  let specific_ways := total_shirts * total_shorts * total_socks * total_hats
  (specific_ways : ℚ) / total_ways_to_choose_4 = 144 / 1815 := by
  sorry

end probability_one_each_from_drawer_l150_150661


namespace problem_intervals_monotonicity_problem_inequality_m_l150_150649

noncomputable def a (x : ℝ) : ℝ × ℝ := (sqrt 3 * sin x, cos x + sin x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (2 * cos x, sin x - cos x)
noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

theorem problem_intervals_monotonicity :
    (∀ x ∈ (Icc (-π / 6 + k * π) (π / 3 + k * π) for some k ∈ ℤ), f(x) is monotonically increasing) ∧
    (∀ x ∈ (Icc (π / 3 + k * π) (5 * π / 6 + k * π) for some k ∈ ℤ), f(x) is monotonically decreasing) :=
sorry

theorem problem_inequality_m (x ∈ Icc (5 * π / 24) (5 * π / 12)) :
    0 ≤ m ∧ m ≤ 4 → (∀ t : ℝ, mt^2 + mt + 3 ≥ f(x)) :=
sorry

end problem_intervals_monotonicity_problem_inequality_m_l150_150649


namespace z_pow_n_add_inv_pow_n_eq_two_cos_n_alpha_l150_150579

-- Given conditions and definitions
variables {α : ℝ} {z : ℂ} 
  (hz : z + 1/z = 2 * Real.cos α)

-- The target statement
theorem z_pow_n_add_inv_pow_n_eq_two_cos_n_alpha (n : ℕ) (hz : z + 1/z = 2 * Real.cos α) : 
  z ^ n + 1 / (z ^ n) = 2 * Real.cos (n * α) := 
  sorry

end z_pow_n_add_inv_pow_n_eq_two_cos_n_alpha_l150_150579


namespace problem1_l150_150293

theorem problem1 (n : ℕ) (A : Set ℂ) (h1 : 2 ≤ n)
  (h2 : A.card = n)
  (h3 : ∀ i ∈ A, ∃ B : Set ℂ, B = {z : ℂ | ∃ (j : ℂ) (hj : j ∈ A), z = i * j} ∧ B = A) :
  (∀ z ∈ A, abs z = 1) ∧ ∀ z ∈ A, conj z ∈ A :=
by.
  sorry

end problem1_l150_150293


namespace bounded_area_l150_150549

theorem bounded_area :
  let R : ℝ := (15 / (2 * π))^(1/2),
      is_inside_circle (x y : ℝ) : Prop := x^2 + y^2 ≤ 15 / (2 * π),
      satisfies_second_ineq (x y : ℝ) : Prop := (x^3 - y) * (x + y^3) ≤ 0 in
  (∃ (R : ℝ), (R = (15 / (2 * π))^(1/2)) ∧
  ∀ (x y : ℝ), is_inside_circle x y ∧ satisfies_second_ineq x y → 
  (area_of_region (interior_of_circle R) ∩ (region_defined_by_second_ineq) = 3.75)
  :=
begin
  sorry
end

end bounded_area_l150_150549


namespace tissues_per_box_l150_150395

theorem tissues_per_box :
  (9 + 10 + 11) * (λ box : ℕ, box) = 1200 → 1200 / (9 + 10 + 11) = 40 :=
begin
  sorry
end

end tissues_per_box_l150_150395


namespace count_of_divisibles_l150_150212

theorem count_of_divisibles (n : ℕ) (h : n < 500) : 
  (∃ k, n = 2 * 3 * 5 * 7 * k ∧ k > 0) →
  (∃! k, k = 2) :=
by {
  have LCM := 210,
  -- sorry is used to skip the proof
  sorry
}

end count_of_divisibles_l150_150212


namespace count_integers_divisible_by_2_3_5_7_l150_150224

theorem count_integers_divisible_by_2_3_5_7 :
  ∃ n : ℕ, (∀ k : ℕ, k < 500 → (k % 2 = 0 ∧ k % 3 = 0 ∧ k % 5 = 0 ∧ k % 7 = 0) → k ≠ n → k < 500 ∧ k > 0) ∧
  (n = 2) :=
by
  sorry

end count_integers_divisible_by_2_3_5_7_l150_150224


namespace travel_distance_l150_150772

noncomputable def distance_traveled (AB BC : ℝ) : ℝ :=
  let BD := Real.sqrt (AB^2 + BC^2)
  let arc1 := (2 * Real.pi * BD) / 4
  let arc2 := (2 * Real.pi * AB) / 4
  arc1 + arc2

theorem travel_distance (hAB : AB = 3) (hBC : BC = 4) : 
  distance_traveled AB BC = 4 * Real.pi := by
    sorry

end travel_distance_l150_150772


namespace number_of_true_conclusions_l150_150732

section
variables (a b c : ℝ)

def star (a b : ℝ) : ℝ := (a + b)^2

theorem number_of_true_conclusions :
  let number_of_true_conclusions := 
    ((star a b = 0 → a = 0 ∧ b = 0) + 
     (star a b = star b a) + 
     (star a (b + c) = star a b + star a c) + 
     (star a b = star (-a) (-b)))
   in number_of_true_conclusions = 2 :=
by 
  sorry
end

end number_of_true_conclusions_l150_150732


namespace max_S_value_l150_150704

noncomputable def max_S (A C : ℝ) [DecidableEq ℝ] : ℝ :=
  if h : 0 < A ∧ A < 2 * Real.pi / 3 ∧ A + C = 2 * Real.pi / 3 then
    (Real.sqrt 3 / 6) * Real.sin (2 * A - Real.pi / 3) + (Real.sqrt 3 / 12)
  else
    0

theorem max_S_value :
  ∃ (A C : ℝ), A + C = 2 * Real.pi / 3 ∧
    (S = (Real.sqrt 3 / 3) * Real.sin A * Real.sin C) ∧
    (max_S A C = Real.sqrt 3 / 4) := 
sorry

end max_S_value_l150_150704


namespace cuboid_surface_area_two_cubes_l150_150427

noncomputable def cuboid_surface_area (b : ℝ) : ℝ :=
  let l := 2 * b
  let w := b
  let h := b
  2 * (l * w + l * h + w * h)

theorem cuboid_surface_area_two_cubes (b : ℝ) : cuboid_surface_area b = 10 * b^2 := by
  sorry

end cuboid_surface_area_two_cubes_l150_150427


namespace hyperbola_focal_length_l150_150380

theorem hyperbola_focal_length : 
  (∃ (f : ℝ) (x y : ℝ), (3 * x^2 - y^2 = 3) ∧ (f = 4)) :=
by {
  sorry
}

end hyperbola_focal_length_l150_150380


namespace symmetry_y_axis_l150_150694

theorem symmetry_y_axis (A : ℝ × ℝ) (hA : A = (3, 1)) :
    ∃ B : ℝ × ℝ, B = (-3, 1) ∧ B.2 = A.2 ∧ B.1 = -A.1 :=
by {
    use (-3, 1),
    simp [hA],
}

end symmetry_y_axis_l150_150694


namespace rex_remaining_cards_l150_150336

-- Definitions based on the conditions provided:
def nicole_cards : ℕ := 400
def cindy_cards (nicole_cards : ℕ) : ℕ := 2 * nicole_cards
def combined_total (nicole_cards cindy_cards : ℕ) : ℕ := nicole_cards + cindy_cards nicole_cards
def rex_cards (combined_total : ℕ) : ℕ := combined_total / 2
def rex_divided_cards (rex_cards siblings : ℕ) : ℕ := rex_cards / (1 + siblings)

-- The theorem to be proved based on the question and correct answer:
theorem rex_remaining_cards : rex_divided_cards (rex_cards (combined_total nicole_cards (cindy_cards nicole_cards))) 3 = 150 :=
by sorry

end rex_remaining_cards_l150_150336


namespace find_unique_n_l150_150857

noncomputable def S_n (n : ℕ) (a : Fin n → ℝ) : ℝ :=
  ∑ k in Finset.range n, Real.sqrt ((2 * (k + 1) - 1) ^ 2 + (a k) ^ 2)

theorem find_unique_n : ∃! n : ℕ, 
  (∀ a : Fin n → ℝ, (∑ k in Finset.range n, a k = 17) → 
  (∃ k : ℕ, S_n n a = k)) ∧ n = 12 :=
begin
  sorry
end

end find_unique_n_l150_150857


namespace snow_at_Brecknock_l150_150511

theorem snow_at_Brecknock (hilt_snow brecknock_snow : ℕ) (h1 : hilt_snow = 29) (h2 : hilt_snow = brecknock_snow + 12) : brecknock_snow = 17 :=
by
  sorry

end snow_at_Brecknock_l150_150511


namespace second_integer_value_l150_150820

-- Definitions of conditions directly from a)
def consecutive_integers (a b c : ℤ) : Prop :=
  b = a + 1 ∧ c = b + 1

def sum_of_first_and_third (a c : ℤ) (sum : ℤ) : Prop :=
  a + c = sum

-- Translated proof problem
theorem second_integer_value (n: ℤ) (h1: consecutive_integers (n - 1) n (n + 1))
  (h2: sum_of_first_and_third (n - 1) (n + 1) 118) : 
  n = 59 :=
by
  sorry

end second_integer_value_l150_150820


namespace smallest_x_plus_y_l150_150604

theorem smallest_x_plus_y 
  (x y : ℕ) 
  (hx_pos : 0 < x) 
  (hy_pos : 0 < y) 
  (hxy_neq : x ≠ y)
  (h_eq : (1/x + 1/y = 1/10)) : x + y = 45 :=
sorry

end smallest_x_plus_y_l150_150604


namespace binom_25_5_l150_150615

theorem binom_25_5 :
  (Nat.choose 23 3 = 1771) ∧
  (Nat.choose 23 4 = 8855) ∧
  (Nat.choose 23 5 = 33649) → 
  Nat.choose 25 5 = 53130 := by
sorry

end binom_25_5_l150_150615


namespace trig_identity_1_l150_150055

theorem trig_identity_1 : 
  cos 89 * cos 1 + sin 91 * sin 181 = cos 88 + sin 1^2 := 
sorry

end trig_identity_1_l150_150055


namespace total_water_output_l150_150430

theorem total_water_output (flow_rate: ℚ) (time_duration: ℕ) (total_water: ℚ) :
  flow_rate = 2 + 2 / 3 → time_duration = 9 → total_water = 24 →
  flow_rate * time_duration = total_water :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end total_water_output_l150_150430


namespace infinite_series_sum_zero_l150_150113

theorem infinite_series_sum_zero : ∑' n : ℕ, (3 * n + 4) / ((n + 1) * (n + 2) * (n + 3)) = 0 :=
by
  sorry

end infinite_series_sum_zero_l150_150113


namespace negative_rational_number_l150_150907

theorem negative_rational_number :
  ∀ (a b c d : ℚ), a = -(-2010) ∧ b = -abs(-2010) ∧ c = (-2011)^2010 ∧ d = (-2010)/(-2011) → b < 0 :=
by
  intros a b c d h,
  obtain ⟨h₁, h₂, h₃, h₄⟩ := h,
  rw h₂,
  norm_num,
  sorry

end negative_rational_number_l150_150907


namespace sum_of_solutions_l150_150569

noncomputable def f (a b x : ℝ) : ℝ :=
if x ≤ 0 then x^2 + (a + b) * x + 2 else 2

theorem sum_of_solutions :
  (∃ a : ℝ, a + log a = 4) → (∃ b : ℝ, b + 10^b = 4) → 
  (a b : ℝ) → (a + log a = 4) → (b + 10^b = 4) → 
  (let solutions := {x : ℝ | f a b x = x } in 
    ∑ x in solutions, x = -1) :=
begin
  intros exists_a exists_b a a_eq b b_eq,
  have f_eq : ∀ x, f a b x = if x ≤ 0 then x^2 + 4 * x + 2 else 2 := sorry,
  have solutions_set : {x | f a b x = x} = {-2, -1, 2} := sorry,
  have sum_eq : ∑ x in {-2, -1, 2}, x = -1 := sorry,
  exact sum_eq,
end

end sum_of_solutions_l150_150569


namespace area_inside_circle_Z_outside_X_Y_l150_150924

-- Definitions for the circles and their properties
structure Circle :=
  (radius : ℝ)
  (center : ℝ × ℝ)

-- A function to define the area of a circle given its radius
def area (c : Circle) : ℝ :=
  π * c.radius^2

-- Hypotheses 
def circleX : Circle := {radius := 2, center := (0, 0)}
def circleY : Circle := {radius := 2, center := (4, 0)}
def circleZ : Circle := {radius := 1, center := (2, 3)}

-- Properties
def circles_tangent (c1 c2 : Circle) : Prop :=
  dist c1.center c2.center = c1.radius + c2.radius

-- Main theorem statement
theorem area_inside_circle_Z_outside_X_Y : 
  circles_tangent circleX circleY ∧
  circles_tangent circleZ circleX ∧
  circles_tangent circleZ circleY →
  area circleZ = π :=
by
  sorry

end area_inside_circle_Z_outside_X_Y_l150_150924


namespace union_of_intervals_l150_150642

open Set

theorem union_of_intervals :
  let M := { x : ℝ | 1 < x ∧ x ≤ 3 }
  let N := { x : ℝ | 2 < x ∧ x ≤ 5 }
  M ∪ N = { x : ℝ | 1 < x ∧ x ≤ 5 } :=
by
  let M := { x : ℝ | 1 < x ∧ x ≤ 3 }
  let N := { x : ℝ | 2 < x ∧ x ≤ 5 }
  sorry

end union_of_intervals_l150_150642


namespace smallest_sum_l150_150600

theorem smallest_sum (x y : ℕ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x ≠ y) (eq : 1/x + 1/y = 1/10) : x + y = 45 :=
by
  sorry

end smallest_sum_l150_150600


namespace avg_zits_per_kid_mr_jones_class_l150_150394

-- Define the conditions
def avg_zits_ms_swanson_class := 5
def num_kids_ms_swanson_class := 25
def num_kids_mr_jones_class := 32
def extra_zits_mr_jones_class := 67

-- Define the total number of zits in Ms. Swanson's class
def total_zits_ms_swanson_class := avg_zits_ms_swanson_class * num_kids_ms_swanson_class

-- Define the total number of zits in Mr. Jones' class
def total_zits_mr_jones_class := total_zits_ms_swanson_class + extra_zits_mr_jones_class

-- Define the problem statement to prove: the average number of zits per kid in Mr. Jones' class
theorem avg_zits_per_kid_mr_jones_class : 
  total_zits_mr_jones_class / num_kids_mr_jones_class = 6 := by
  sorry

end avg_zits_per_kid_mr_jones_class_l150_150394


namespace cos_six_times_arccos_half_l150_150918

theorem cos_six_times_arccos_half : 
  cos (6 * arccos (1 / 2)) = 1 := 
by
  sorry

end cos_six_times_arccos_half_l150_150918


namespace a_10_is_1_over_28_l150_150589

noncomputable def sequence : ℕ → ℚ
| 1     := 1
| (n+1) := (λ a_n, a_n - 3 * a_n * a_n) (sequence n)

theorem a_10_is_1_over_28 : sequence 10 = 1 / 28 :=
by sorry

end a_10_is_1_over_28_l150_150589


namespace z_pow_n_add_inv_pow_n_eq_two_cos_n_alpha_l150_150580

-- Given conditions and definitions
variables {α : ℝ} {z : ℂ} 
  (hz : z + 1/z = 2 * Real.cos α)

-- The target statement
theorem z_pow_n_add_inv_pow_n_eq_two_cos_n_alpha (n : ℕ) (hz : z + 1/z = 2 * Real.cos α) : 
  z ^ n + 1 / (z ^ n) = 2 * Real.cos (n * α) := 
  sorry

end z_pow_n_add_inv_pow_n_eq_two_cos_n_alpha_l150_150580


namespace trevor_bills_more_than_coins_l150_150823

-- Declare noncomputable as necessary due to division and real numbers
noncomputable def coin_conversion : Nat :=
  let num_quarters := 45
  let num_dimes := 34
  let num_nickels := 19
  let num_pennies := 153 - num_quarters - num_dimes - num_nickels
  let total_value : Real :=
    num_quarters * 0.25 + num_dimes * 0.10 + num_nickels * 0.05 + num_pennies * 0.01
  let num_five_dollar_bills := total_value / 5
  let remaining_value := total_value % 5
  let num_one_dollar_coins := remaining_value / 1
  num_five_dollar_bills - num_one_dollar_coins

theorem trevor_bills_more_than_coins :
  coin_conversion = 2 :=
by
  sorry

end trevor_bills_more_than_coins_l150_150823


namespace find_q_and_a3_l150_150404

open Nat

variables {a : ℕ → ℝ} {q : ℝ} {a1 : ℝ}

/-- Given conditions on the geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n, a (n + 1) = a 1 * q ^ n 

/-- Sum of the first n terms of the sequence -/
def partial_sum (a : ℕ → ℝ) (n : ℕ) :=
  (finset.range n).sum (λ k, a k)

/-- The arithmetic mean condition on the given problem -/
def arithmetic_mean_condition (a : ℕ → ℝ) (q : ℝ) (a1 : ℝ) :=
  (10 / 3) * a 1 * q^2 = a 1 * (q + q^3)

/-- The sequence sum condition -/
def sum_condition (a : ℕ → ℝ) (q : ℝ) (a1 : ℝ) :=
  (a 1 * (1 - q^5)) / (1 - q) = 484

theorem find_q_and_a3 (a : ℕ → ℝ) (q a1 : ℝ) 
  (h_geom : geometric_sequence a q)
  (h_sum : sum_condition a q a1)
  (h_mean : arithmetic_mean_condition a q a1)
  (h_pos_q : q > 1)
  (h_pos_a : ∀ n, a n > 0)
  : q = 3 ∧ a 3 = 36 :=
by
  sorry

end find_q_and_a3_l150_150404


namespace cos_minus_sin_l150_150993

noncomputable def alpha := sorry -- This will represent the angle in radians.

def pointP : ℝ × ℝ := (Real.sqrt 3, -1)

-- Define r as the magnitude of pointP 
def r := Real.sqrt ((pointP.1)^2 + (pointP.2)^2)

-- Define the coordinates
def x := pointP.1
def y := pointP.2

-- Define cos and sin
def cos_alpha := x / r
def sin_alpha := y / r

theorem cos_minus_sin (h : cos_alpha - sin_alpha = (Real.sqrt 3 + 1) / 2) : 
  cos_alpha - sin_alpha = (Real.sqrt 3 + 1) / 2 :=
by
  rw h
  sorry

end cos_minus_sin_l150_150993


namespace inverse_of_matrix_l150_150141

theorem inverse_of_matrix :
  let A : Matrix (Fin 2) (Fin 2) Rational := ![![5, -3], ![2, 1]]
  let A_inv : Matrix (Fin 2) (Fin 2) Rational := ![
    ![1 / 11, 3 / 11],
    ![-2 / 11, 5 / 11]
  ]
  det A ≠ 0 → A ⁻¹ = A_inv :=
by
  intro h_det
  sorry

end inverse_of_matrix_l150_150141


namespace qin_jiushao_operations_required_l150_150828

def polynomial (x : ℝ) : ℝ := 5 * x^5 + 4 * x^4 + 3 * x^3 + 2 * x^2 + x + 1

theorem qin_jiushao_operations_required : 
  (∃ x : ℝ, polynomial x = (((((5 * x + 4) * x + 3) * x + 2) * x + 1) * x + 1)) →
  (∃ m a : ℕ, m = 5 ∧ a = 5) := by
  sorry

end qin_jiushao_operations_required_l150_150828


namespace A_is_irrational_l150_150751

-- Define a sequence of digits where each digit block comes from consecutive primes
noncomputable def A : ℝ :=
  let digits := List.map showDigit [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, ...] -- and so on
  Real.mkDigits 10 digits

theorem A_is_irrational : irrational A :=
  by
    sorry

end A_is_irrational_l150_150751


namespace express_c_in_terms_of_a_and_b_l150_150618

-- Define vector tuple and alias
abbreviation Vector2 := ℝ × ℝ

-- Define the given vectors
def a : Vector2 := (1, 1)
def b : Vector2 := (1, -1)
def c : Vector2 := (-1, 2)

-- Prove that vector c can be expressed in terms of vector a and b
theorem express_c_in_terms_of_a_and_b : 
  c = (1/2 : ℝ) • a + (-3/2 : ℝ) • b :=
by
  sorry

end express_c_in_terms_of_a_and_b_l150_150618


namespace exists_square_in_interval_l150_150726

def x_k (k : ℕ) : ℕ := k * (k + 1) / 2

noncomputable def sum_x (n : ℕ) : ℕ := (List.range n).map x_k |>.sum

theorem exists_square_in_interval (n : ℕ) (hn : n ≥ 10) :
  ∃ m, (sum_x n - x_k n ≤ m^2 ∧ m^2 ≤ sum_x n) :=
by sorry

end exists_square_in_interval_l150_150726


namespace A_minus_one_not_prime_l150_150957

theorem A_minus_one_not_prime (n : ℕ) (h : 0 < n) (m : ℕ) (h1 : 10^(m-1) < 14^n) (h2 : 14^n < 10^m) :
  ¬ (Nat.Prime (2^n * 10^m + 14^n - 1)) :=
by
  sorry

end A_minus_one_not_prime_l150_150957


namespace clock_angle_at_537pm_l150_150432

noncomputable def smaller_angle_between_clock_hands (hour minute : ℕ) : ℝ :=
  let minute_angle := minute * 6
  let hour_angle := (hour % 12) * 30 + minute * 0.5
  let diff := abs (minute_angle - hour_angle)
  if diff > 180 then 360 - diff else diff

theorem clock_angle_at_537pm :
  smaller_angle_between_clock_hands 5 37 = 53.5 :=
by
  sorry

end clock_angle_at_537pm_l150_150432


namespace inverse_function_of_f_l150_150991

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1 + a^x

theorem inverse_function_of_f (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1)
  (h₂ : (f a 2) = 5) :
  ∀ (x : ℝ), x > 1 → f⁻¹(x) = Real.logBase 2 (x - 1) :=
by
  sorry

end inverse_function_of_f_l150_150991


namespace evaluate_f_pi_div_six_l150_150161

def f (x : ℝ) : ℝ :=
  if x ≥ Real.sin x then Real.sin x else x

theorem evaluate_f_pi_div_six : f (Real.pi / 6) = 1 / 2 :=
by
  sorry

end evaluate_f_pi_div_six_l150_150161


namespace exists_natural_numbers_solving_equation_l150_150036

theorem exists_natural_numbers_solving_equation :
  ∃ (x y z : ℕ), 28 * x + 30 * y + 31 * z = 365 := by
  sorry

end exists_natural_numbers_solving_equation_l150_150036


namespace jack_pays_back_l150_150281

-- Define the principal amount P and interest rate r
def principal_amount : ℝ := 1200
def interest_rate : ℝ := 0.1

-- Define the interest and the total amount Jack has to pay back
def interest : ℝ := interest_rate * principal_amount
def total_amount : ℝ := principal_amount + interest

-- State the theorem to prove that the total amount Jack pays back is 1320
theorem jack_pays_back : total_amount = 1320 := by
  sorry

end jack_pays_back_l150_150281


namespace min_area_monochromatic_triangle_l150_150392

-- Definition of the integer lattice in the plane.
def lattice_points : Set (ℤ × ℤ) := { p | ∃ x y : ℤ, p = (x, y) }

-- The 3-coloring condition
def coloring (c : (ℤ × ℤ) → Fin 3) := ∀ p : (ℤ × ℤ), p ∈ lattice_points → (c p) < 3

-- Definition of the area of a triangle
def triangle_area (A B C : ℤ × ℤ) : ℝ :=
  0.5 * abs (((B.1 - A.1) * (C.2 - A.2)) - ((C.1 - A.1) * (B.2 - A.2)))

-- The statement we need to prove
theorem min_area_monochromatic_triangle :
  ∃ S : ℝ, S = 3 ∧ ∀ (c : (ℤ × ℤ) → Fin 3), coloring c → ∃ (A B C : ℤ × ℤ), A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ (c A = c B ∧ c B = c C) ∧ triangle_area A B C = S :=
sorry

end min_area_monochromatic_triangle_l150_150392


namespace green_apples_count_l150_150403

def red_apples := 33
def students_took := 21
def extra_apples := 35

theorem green_apples_count : ∃ G : ℕ, red_apples + G - students_took = extra_apples ∧ G = 23 :=
by
  use 23
  have h1 : 33 + 23 - 21 = 35 := by norm_num
  exact ⟨h1, rfl⟩

end green_apples_count_l150_150403


namespace constant_term_in_binomial_expansion_l150_150268

theorem constant_term_in_binomial_expansion 
  (n : ℕ) 
  (h : 4^n + 2^n = 72) :
  let exp := (√x + 3/x)^n in
  let T₂ := 
    if n = 3 then 
      (3 * nat.choose n 1) 
    else 0 in
  T₂ = 9 :=
by 
  sorry

end constant_term_in_binomial_expansion_l150_150268


namespace range_of_m_when_p_and_q_true_range_of_m_when_p_and_q_false_and_p_or_q_true_l150_150204

variable (m : ℝ)

def p : Prop :=
  (|2 * m + 1| / Real.sqrt (m^2 + 1) ≤ 2)

def q : Prop :=
  ((m - 1) * (2 - m) < 0)

theorem range_of_m_when_p_and_q_true :
  (p ∧ q) → m ≤ 3 / 4 :=
by
  intro h
  sorry

theorem range_of_m_when_p_and_q_false_and_p_or_q_true :
  (¬(p ∧ q) ∧ (p ∨ q)) → (3 / 4 < m ∧ m < 1) :=
by
  intro h
  sorry

end range_of_m_when_p_and_q_true_range_of_m_when_p_and_q_false_and_p_or_q_true_l150_150204


namespace bus_system_carry_per_day_l150_150794

theorem bus_system_carry_per_day (total_people : ℕ) (weeks : ℕ) (days_in_week : ℕ) (people_per_day : ℕ) :
  total_people = 109200000 →
  weeks = 13 →
  days_in_week = 7 →
  people_per_day = total_people / (weeks * days_in_week) →
  people_per_day = 1200000 :=
by
  intros htotal hweeks hdays hcalc
  sorry

end bus_system_carry_per_day_l150_150794


namespace incorrect_connection_probability_l150_150417

noncomputable def probability_of_incorrect_connection (p : ℝ) : ℝ :=
  let r2 := 1 / 9
  let r3 := (8 / 9) * (1 / 9)
  (3 * p^2 * (1 - p) * r2) + (1 * p^3 * r3)

theorem incorrect_connection_probability : probability_of_incorrect_connection 0.02 = 0.000131 :=
by
  sorry

end incorrect_connection_probability_l150_150417


namespace right_triangles_in_rectangle_l150_150799

-- Define points A, B, C, D, P, Q, R
variables (A B C D P Q R : Type)

-- The condition that ABCD is a rectangle
def is_rectangle (A B C D : Type) : Prop := sorry

-- Conditions of the problem
def condition (A B C D P Q R : Type) : Prop :=
  is_rectangle A B C D ∧ 
  (∃ PQ_divides : (A P Q B ∧ Q R ∧ R C) / (equivalent_to_squares P Q B C))

-- Theorems regarding the right triangles
theorem right_triangles_in_rectangle (A B C D P Q R : Type) (h : condition A B C D P Q R) : 
  count_right_triangles A B C D P Q R = 14 :=
sorry

end right_triangles_in_rectangle_l150_150799


namespace unique_solution_condition_l150_150933

theorem unique_solution_condition (a b : ℝ) : (4 * x - 6 + a = (b + 1) * x + 2) → b ≠ 3 :=
by
  intro h
  -- Given the condition equation
  have eq1 : 4 * x - 6 + a = (b + 1) * x + 2 := h
  -- Simplify to the form (3 - b) * x = 8 - a
  sorry

end unique_solution_condition_l150_150933


namespace player_b_max_abs_sum_l150_150019

theorem player_b_max_abs_sum :
  (∀ (signs : fin 20 → ℤ) 
    (A_strategy B_strategy : fin 20 → ℤ → fin 20 → ℤ),
    (∀ i, signs i ∈ {-1, 1}) →
    (∀ i, A_strategy i (B_strategy i (signs i)) i = signs i) →
    (| (finset.univ.sum (λ i, signs i)) | ≤ 30)) :=
sorry

end player_b_max_abs_sum_l150_150019


namespace period_of_f_side_length_c_l150_150193

noncomputable def f (x : ℝ) : ℝ := cos x * cos (x + π / 3)

theorem period_of_f : ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = π := by
  sorry

variable {A B C : ℝ}
variable {a b c : ℝ}

noncomputable def area_of_triangle : ℝ := 2 * sqrt 3

theorem side_length_c (h_f : f c = -1 / 4) (h_a : a = 2) 
  (h_area : 1 / 2 * a * b * sin C = area_of_triangle) : c = 2 * sqrt 3 := by
  sorry

end period_of_f_side_length_c_l150_150193


namespace pet_food_total_weight_l150_150756

theorem pet_food_total_weight:
  let cat_food_bags := 3
  let weight_per_cat_food_bag := 3 -- pounds
  let dog_food_bags := 4 
  let weight_per_dog_food_bag := 5 -- pounds
  let bird_food_bags := 5
  let weight_per_bird_food_bag := 2 -- pounds
  let total_weight_pounds := (cat_food_bags * weight_per_cat_food_bag) + (dog_food_bags * weight_per_dog_food_bag) + (bird_food_bags * weight_per_bird_food_bag)
  let total_weight_ounces := total_weight_pounds * 16
  total_weight_ounces = 624 :=
by
  let cat_food_bags := 3
  let weight_per_cat_food_bag := 3
  let dog_food_bags := 4
  let weight_per_dog_food_bag := 5
  let bird_food_bags := 5
  let weight_per_bird_food_bag := 2
  let total_weight_pounds := (cat_food_bags * weight_per_cat_food_bag) + (dog_food_bags * weight_per_dog_food_bag) + (bird_food_bags * weight_per_bird_food_bag)
  let total_weight_ounces := total_weight_pounds * 16
  show total_weight_ounces = 624
  sorry

end pet_food_total_weight_l150_150756


namespace b_20_value_l150_150314

noncomputable def b : ℕ → ℕ
| 0 => 1
| 1 => 2
| (n+2) => b (n+1) * b n

theorem b_20_value : b 19 = 2^4181 :=
sorry

end b_20_value_l150_150314


namespace range_of_f_when_k_is_4_range_of_k_monotonically_increasing_on_interval_l150_150748

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (k-2) * x^2 + 2 * k * x - 3

theorem range_of_f_when_k_is_4 : set.range (λ x, f 4 x) ∩ set.Icc (-4:ℝ) 1 = set.Icc (-11:ℝ) 7 :=
  sorry

theorem range_of_k_monotonically_increasing_on_interval : 
  {k : ℝ | ∀ x ∈ set.Icc (1:ℝ) 2, 0 ≤ (f k x - f k 1) / (x - 1)} = set.Ici (4/3) :=
  sorry

end range_of_f_when_k_is_4_range_of_k_monotonically_increasing_on_interval_l150_150748


namespace evaluate_expression_l150_150365

theorem evaluate_expression (a b : ℚ) (h1 : a + b = 4) (h2 : a - b = 2) :
  ( (a^2 - 6 * a * b + 9 * b^2) / (a^2 - 2 * a * b) / ((5 * b^2 / (a - 2 * b)) - (a + 2 * b)) - 1 / a ) = -1 / 3 :=
by
  sorry

end evaluate_expression_l150_150365


namespace rows_with_stars_eq_columns_with_stars_l150_150257

-- Define the types and variables
variables (R C : Type) [Fintype R] [Fintype C] (stars : R → C → Prop)
  (nrStarsInRow : R → ℕ) (ncStarsInCol : C → ℕ)

-- Conditions from the problem
def column_star_count_eq_row_star_count := 
  ∀ r c, stars r c → ncStarsInCol c = nrStarsInRow r

-- The proof theorem
theorem rows_with_stars_eq_columns_with_stars 
  (h : column_star_count_eq_row_star_count stars nrStarsInRow ncStarsInCol) :
  Fintype.card {r // ∃ c, stars r c} = Fintype.card {c // ∃ r, stars r c} :=
sorry

end rows_with_stars_eq_columns_with_stars_l150_150257


namespace line_slope_l150_150434

theorem line_slope : 
  (∀ (x y : ℝ), (x / 4 - y / 3 = -2) → (y = -3/4 * x - 6)) ∧ (∀ (x : ℝ), ∃ y : ℝ, (x / 4 - y / 3 = -2)) :=
by
  sorry

end line_slope_l150_150434


namespace average_yield_correct_l150_150091

-- We define the given conditions
def base : ℝ := 200  -- base of the triangle in meters
def multiplier : ℝ := 1.2  -- height is 1.2 times the base
def total_harvest : ℝ := 2.4  -- total harvest in tons

-- Define height based on conditions
def height : ℝ := multiplier * base

-- Calculate area of the triangle
def area_sq_m : ℝ := 0.5 * base * height

-- Convert area to hectares (1 hectare = 10,000 square meters)
def area_hectare : ℝ := area_sq_m / 10000

-- Calculate the average yield per hectare
def average_yield : ℝ := total_harvest / area_hectare

-- The final statement to prove
theorem average_yield_correct : average_yield = 1 := 
  by 
  sorry

end average_yield_correct_l150_150091


namespace number_of_performance_orders_l150_150068

-- Define the options for the programs
def programs : List String := ["A", "B", "C", "D", "E", "F", "G", "H"]

-- Define a function to count valid performance orders under given conditions
def countPerformanceOrders (progs : List String) : ℕ :=
  sorry  -- This is where the logic to count performance orders goes

-- The theorem to assert the total number of performance orders
theorem number_of_performance_orders : countPerformanceOrders programs = 2860 :=
by
  sorry  -- Proof of the theorem

end number_of_performance_orders_l150_150068


namespace area_of_common_part_geq_3484_l150_150496

theorem area_of_common_part_geq_3484 :
  ∀ (R : ℝ) (S T : ℝ → Prop), 
  (R = 1) →
  (∀ x y, S x ↔ (x * x + y * y = R * R) ∧ T y) →
  ∃ (S_common : ℝ) (T_common : ℝ),
    (S_common + T_common > 3.484) :=
by
  sorry

end area_of_common_part_geq_3484_l150_150496


namespace curve_in_second_quadrant_l150_150189

theorem curve_in_second_quadrant (a : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 + 2*a*x - 4*a*y + 5*a^2 - 4 = 0 → x < 0 ∧ y > 0) ↔ (a > 2) :=
sorry

end curve_in_second_quadrant_l150_150189


namespace principal_amount_l150_150000

theorem principal_amount (SI : ℝ) (R : ℝ) (T : ℕ) (P : ℝ) :
  SI = 3.45 → R = 0.05 → T = 3 → SI = P * R * T → P = 23 :=
by
  -- The proof steps would go here but are omitted as specified.
  sorry

end principal_amount_l150_150000


namespace minimum_total_length_of_removed_segments_l150_150103

-- Definitions based on conditions
def right_angled_triangle_sides : Nat × Nat × Nat := (3, 4, 5)

def large_square_side_length : Nat := 7

-- Statement of the problem to be proved
theorem minimum_total_length_of_removed_segments
  (triangles : Fin 4 → (Nat × Nat × Nat) := fun _ => right_angled_triangle_sides)
  (side_length_of_large_square : Nat := large_square_side_length) :
  ∃ (removed_length : Nat), removed_length = 7 :=
sorry

end minimum_total_length_of_removed_segments_l150_150103


namespace relationship_y_values_l150_150154

theorem relationship_y_values (x1 x2 y1 y2 : ℝ) (h1 : x1 > x2) (h2 : 0 < x2) (h3 : y1 = - (3 / x1)) (h4 : y2 = - (3 / x2)) : y1 > y2 :=
by
  sorry

end relationship_y_values_l150_150154


namespace beetles_sixth_jar_l150_150869

theorem beetles_sixth_jar :
  ∃ (x y : ℕ), 
    (150 = ∑ i in finset.range 10, x + i) ∧
    (∀ i ∈ finset.range 9, x + i < x + (i + 1)) ∧
    (x ≥ (y - x) / 2) ∧
    (y = x + 9) ∧
    let n_beetles := x + 5 in
    n_beetles = 16 := sorry

end beetles_sixth_jar_l150_150869


namespace term_3007_l150_150379

def sum_of_cubes_of_digits (n : ℕ) : ℕ := 
  let digits := n.digits 10
  digits.map (fun d => d^3).sum

def sequence_term (n : ℕ) : ℕ :=
  Nat.iterate sum_of_cubes_of_digits n

theorem term_3007 :
  sequence_term 3007 3007 = 370 :=
by
  sorry

end term_3007_l150_150379


namespace cost_of_tax_free_item_D_l150_150711

theorem cost_of_tax_free_item_D 
  (P_A P_B P_C : ℝ)
  (H1 : 0.945 * P_A + 1.064 * P_B + 1.18 * P_C = 225)
  (H2 : 0.045 * P_A + 0.12 * P_B + 0.18 * P_C = 30) :
  250 - (0.945 * P_A + 1.064 * P_B + 1.18 * P_C) = 25 := 
by
  -- The proof steps would go here.
  sorry

end cost_of_tax_free_item_D_l150_150711


namespace collinear_c1_c2_l150_150101

-- Define the vectors a and b
def a : ℝ × ℝ × ℝ := (3, 7, 0)
def b : ℝ × ℝ × ℝ := (1, -3, 4)

-- Define the vectors c1 and c2 based on a and b
def c1 : ℝ × ℝ × ℝ := (4 * 3, 4 * 7, 4 * 0) - (2 * 1, 2 * -3, 2 * 4)
def c2 : ℝ × ℝ × ℝ := (1, -3, 4) - (2 * 3, 2 * 7, 2 * 0)

-- The theorem to prove that c1 and c2 are collinear
theorem collinear_c1_c2 : c1 = (-2 : ℝ) • c2 := by sorry

end collinear_c1_c2_l150_150101


namespace ball_hits_ground_l150_150796

noncomputable def ball_height (t : ℝ) : ℝ := -9 * t^2 + 15 * t + 72

theorem ball_hits_ground :
  (∃ t : ℝ, t = (5 + Real.sqrt 313) / 6 ∧ ball_height t = 0) :=
sorry

end ball_hits_ground_l150_150796


namespace find_b_l150_150699

theorem find_b (x y : Fin 6 → ℝ) (b : ℝ)
  (h1 : ∑ i, x i = 11)
  (h2 : ∑ i, y i = 13)
  (h3 : ∑ i, (x i)^2 = 21)
  (h4 : ∀ i, y i = b * (x i)^2 - 1/3) :
  b = 5/7 := 
sorry

end find_b_l150_150699


namespace zhao_total_cost_l150_150566

-- Definitions from the problem (conditions)
def cost.of_ticket := 12.5 -- This is inferred based on the given solutions
def discount.minor := 0.4
def discount.senior1 := 0.3
def discount.grandfather := 0.2

-- Prices computed from conditions
def price.minor (cost : Real) := cost * (1 - discount.minor)
def price.senior1 (cost : Real) := cost * (1 - discount.senior1)
def price.grandfather (cost : Real) := cost * (1 - discount.grandfather)

-- Number in generations
def num.youngest := 3
def num.second_younger := 2
def num.second_oldest := 1

-- Prices according to generation
def price.youngest_total (cost : Real) := num.youngest * price.minor cost
def price.second_younger_total (cost : Real) := num.second_younger * cost
def price.second_oldest_total (cost : Real) := num.second_oldest * price.senior1 cost
def price.grandfather_total (cost : Real) := price.grandfather cost

-- The proof problem verifying the total cost
theorem zhao_total_cost : price.youngest_total cost.of_ticket
                          + price.second_younger_total cost.of_ticket
                          + price.second_oldest_total cost.of_ticket
                          + price.grandfather_total cost.of_ticket
                          = 66.25 := by sorry

end zhao_total_cost_l150_150566


namespace time_to_traverse_l150_150466

theorem time_to_traverse (n : ℕ) (h : 2 ≤ n)
    (h₃ :   let v₃ := 1 / 3 in
            v₃ * 3 = 1) :
    let c := 4 / 3 in
    let v_n := c / (n-1)^2 in
    let t_n := 1 / v_n in
    t_n = (3 * (n-1)^2) / 4 :=
by
  sorry

end time_to_traverse_l150_150466


namespace cos_alpha_minus_beta_l150_150571

noncomputable def cos_alpha_beta : ℝ :=
  let α := (2*Real.pi) / 3 -- α in (π/2, π)
  let β := (3*Real.pi) / 4 -- β in the third quadrant
  let tanα : ℝ := -4 / 3
  let cosβ : ℝ := -5 /13
  let sinβ : ℝ := (- (1 - cosβ^2).sqrt)
  -- manually compute cos(α-β) using the values
  let cosα := -3 / 5
  let sinα := 4 / 5
  in cosα * cosβ + sinα * sinβ

theorem cos_alpha_minus_beta : cos_alpha_beta = -33 / 65 := 
  by 
    -- This section will be filled in later
    sorry

end cos_alpha_minus_beta_l150_150571


namespace sin_values_l150_150177

theorem sin_values (x : ℝ) 
    (h : (1 / real.cos x) - (real.sin x / real.cos x) = 5 / 3) :
    real.sin x = 53 / 68 ∨ real.sin x = -41 / 68 :=
by
  sorry

end sin_values_l150_150177


namespace min_value_of_f_is_46852_l150_150556

def f (x : ℝ) : ℝ := ∑ k in Finset.range 52, (x - (2 * k)) ^ 2

theorem min_value_of_f_is_46852 : (∃ x : ℝ, f x = 46852) := 
sorry

end min_value_of_f_is_46852_l150_150556


namespace determine_lambda_l150_150622

variable {ℝ : Type*} [LinearOrderedField ℝ]

noncomputable def f : ℝ → ℝ := sorry
-- f is odd means f(-x) = -f(x)
axiom f_odd : ∀ x : ℝ, f (-x) = -f x

-- f is monotonic
axiom f_monotonic : Monotone f

theorem determine_lambda (unique_zero : ∃! x : ℝ, f (2 * x ^ 2 + 1) + f (λ - x) = 0) : λ = -7 / 8 :=
sorry

end determine_lambda_l150_150622


namespace problem1_coordinates_and_hyperbola_problem2_distance_and_position_l150_150861

-- Definitions for Problem 1
def distance (x1 y1 x2 y2 : ℝ) := (x2 - x1) ^ 2 + (y2 - y1) ^ 2

-- Problem 1: Point P and Hyperbola
theorem problem1_coordinates_and_hyperbola :
  let A := (20, 0)
  let B := (-20, 0)
  let P := (15 * sqrt 2 / 2, 5 * sqrt 6 / 2)
  distance P.1 P.2 A.1 A.2 - distance P.1 P.2 B.1 B.2 = 20 ∧ 
  ∃ (x y : ℝ), x = P.1 ∧ y = P.2 ∧ (x^2 / 100) - (y^2 / 300) = 1 :=
by 
  let A := (20, 0)
  let B := (-20, 0)
  let P := (15 * sqrt 2 / 2, 5 * sqrt 6 / 2)
  have h_dist : distance P.1 P.2 A.1 A.2 - distance P.1 P.2 B.1 B.2 = 20 := sorry,
  have h_hyperbola : ∃ (x y : ℝ), x = P.1 ∧ y = P.2 ∧ (x^2 / 100) - (y^2 / 300) = 1 := sorry,
  exact ⟨h_dist, h_hyperbola⟩

-- Definitions for Problem 2
def approx_equal (a b : ℝ) (ε : ℝ := 1.0e-3) := abs (a - b) < ε 

-- Problem 2: Point Q, Distance and Position
theorem problem2_distance_and_position :
  let A := (20, 0)
  let B := (-20, 0)
  let C := (0, -15)
  let D := (0, 15)
  let Q := (sqrt (14400 / 47), sqrt (2975 / 47))
  ∃ (distance_QO : ℝ), approx_equal distance_QO 19 ∧ 
  approx_equal (atan2 Q.2 Q.1 * 180 / Real.pi) 66 :=
by 
  let A := (20, 0)
  let B := (-20, 0)
  let C := (0, -15)
  let D := (0, 15)
  let Q := (sqrt (14400 / 47), sqrt (2975 / 47))
  have h_dist : approx_equal (sqrt (Q.1^2 + Q.2^2)) 19 := sorry,
  have h_angle : approx_equal (atan2 Q.2 Q.1 * 180 / Real.pi) 66 := sorry,
  exact ⟨sqrt (Q.1^2 + Q.2^2), h_dist, h_angle⟩

end problem1_coordinates_and_hyperbola_problem2_distance_and_position_l150_150861


namespace a_eq_3_condition_for_parallel_l150_150956

noncomputable def l1 (a : ℝ) : ℝ × ℝ × ℝ := (1, a, 2)
noncomputable def l2 (a : ℝ) : ℝ × ℝ × ℝ := ((a - 2), 3, 6 * a)

def parallel (l1 l2 : ℝ × ℝ × ℝ) : Prop :=
l1.1 * l2.2 = l1.2 * l2.1

theorem a_eq_3_condition_for_parallel {a : ℝ} :
  (a = 3) ↔ parallel (l1 a) (l2 a) := 
sorry

end a_eq_3_condition_for_parallel_l150_150956


namespace contribution_per_person_l150_150359

-- Define constants for the given conditions
def total_price : ℕ := 67
def coupon : ℕ := 4
def number_of_people : ℕ := 3

-- The theorem to prove
theorem contribution_per_person : (total_price - coupon) / number_of_people = 21 :=
by 
  -- The proof is omitted for this exercise
  sorry

end contribution_per_person_l150_150359


namespace angle_in_second_quadrant_l150_150984

theorem angle_in_second_quadrant (α : ℝ) (h1 : Real.sin α > 0) (h2 : (2 * Real.tan (α / 2)) / (1 - (Real.tan (α / 2))^2) < 0) : 
  ∃ q, q = 2 ∧ α ∈ {α | 0 < α ∧ α < π} :=
by
  sorry

end angle_in_second_quadrant_l150_150984


namespace arrangements_5_people_l150_150562

theorem arrangements_5_people : 
  let people := {A, B, C, D, E} in
  let positions := {1, 2, 3, 4, 5} in
  let valid_arrangements := {
    arrangement ∈ positions.permutations |
      arrangement.head ≠ A ∧ arrangement.tail.head ≠ B
  } in
  valid_arrangements.card = 78 :=
sorry

end arrangements_5_people_l150_150562


namespace viewers_watching_program_A_l150_150761

theorem viewers_watching_program_A (T : ℕ) (hT : T = 560) (x : ℕ)
  (h_ratio : 1 * x + (2 * x - x) + (3 * x - x) = T) : 2 * x = 280 :=
by
  -- by solving the given equation, we find x = 140
  -- substituting x = 140 in 2 * x gives 2 * x = 280
  sorry

end viewers_watching_program_A_l150_150761


namespace sum_of_integer_coeffs_of_factorization_l150_150135

theorem sum_of_integer_coeffs_of_factorization (x y : ℝ) :
  let expr := 27 * x^6 - 512 * y^6 in
  let factor_1 := 3 * x^2 - 8 * y^2 in
  let factor_2 := 9 * x^4 + 24 * x^2 * y^2 + 64 * y^4 in
  let sum_of_coeffs := 3 + (-8) + 9 + 24 + 64 in
  expr = factor_1 * factor_2 → sum_of_coeffs = 92 :=
by
  sorry

end sum_of_integer_coeffs_of_factorization_l150_150135


namespace actual_cost_of_article_l150_150901

theorem actual_cost_of_article (x : ℝ) (h : 0.80 * x = 620) : x = 775 :=
sorry

end actual_cost_of_article_l150_150901


namespace pentagon_tiling_18gon_l150_150942

-- Definitions based on the conditions and questions

def pentagon_side_length : ℝ := sorry
def pentagon_angle1 := 60
def pentagon_angle2 := 160
def pentagon_angle3 := 80
def pentagon_angle4 := 100
def pentagon_angle5 := 540 - 60 - 160 - 80 - 100-- The calculation for the fifth angle
def n_sided_polygon (n : ℕ) := polygon n sorry -- Regular n-sided polygon with given side length

-- The theorem we want to prove
theorem pentagon_tiling_18gon 
  (pentagon : Type) 
  (side_length : ℝ)
  (a1 a2 a3 a4 a5 : ℕ)
  (hl : side_length = pentagon_side_length)
  (h1 : a1 = pentagon_angle1)
  (h2 : a2 = pentagon_angle2)
  (h3 : a3 = pentagon_angle3)
  (h4 : a4 = pentagon_angle4) 
  (h5 : a5 = pentagon_angle5)
  : 
  tiling (n_sided_polygon 18) (λ _ : fin 18, pentagon) :=
sorry

end pentagon_tiling_18gon_l150_150942


namespace range_of_f_x_plus_1_gt_0_l150_150677

variables {ℝ : Type*} [linearOrderedField ℝ]

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def is_increasing_on (f : ℝ → ℝ) (s : set ℝ) : Prop :=
  ∀ x y : ℝ, x ∈ s → y ∈ s → x < y → f x < f y

theorem range_of_f_x_plus_1_gt_0 (f : ℝ → ℝ)
  (h_even : is_even_function f)
  (h_increasing : is_increasing_on f {x : ℝ | x ≤ 0})
  (h_f_3_zero : f 3 = 0) :
  {x : ℝ | f (x + 1) > 0} = set.Ioo (-4 : ℝ) 2 :=
sorry

end range_of_f_x_plus_1_gt_0_l150_150677


namespace roundness_of_hundred_billion_l150_150934

def roundness (n : ℕ) : ℕ :=
  let pf := n.factorization
  pf 2 + pf 5

theorem roundness_of_hundred_billion : roundness 100000000000 = 22 := by
  sorry

end roundness_of_hundred_billion_l150_150934


namespace paul_wins_remainder_383_l150_150350

theorem paul_wins_remainder_383 :
  ∃ (m n : ℕ), (m/n : ℚ) = /* probability calculation logic */ ∧ m.gcd n = 1 ∧ (m + n) % 1000 = 383 := 
sorry

end paul_wins_remainder_383_l150_150350


namespace training_trip_duration_l150_150367

-- Define the number of supervisors
def num_supervisors : ℕ := 15

-- Define the number of supervisors overseeing the pool each day
def supervisors_per_day : ℕ := 3

-- Define the number of pairs supervised per day
def pairs_per_day : ℕ := (supervisors_per_day * (supervisors_per_day - 1)) / 2

-- Define the total number of pairs from the given number of supervisors
def total_pairs : ℕ := (num_supervisors * (num_supervisors - 1)) / 2

-- Define the number of days required
def num_days : ℕ := total_pairs / pairs_per_day

-- The theorem we need to prove
theorem training_trip_duration : 
  (num_supervisors = 15) ∧
  (supervisors_per_day = 3) ∧
  (∀ (a b : ℕ), a * (a - 1) / 2 = b * (b - 1) / 2 → a = b) ∧ 
  (∀ (N : ℕ), total_pairs = N * pairs_per_day → N = 35) :=
by
  sorry

end training_trip_duration_l150_150367


namespace div_by_9_digit_B_l150_150383

theorem div_by_9_digit_B (B : ℕ) (h : (4 + B + B + 2) % 9 = 0) : B = 6 :=
by sorry

end div_by_9_digit_B_l150_150383


namespace probability_snow_at_least_once_l150_150805

noncomputable def probability_at_least_once_snow : ℚ :=
  1 - (↑((1:ℚ) / 4) ^ 5)

theorem probability_snow_at_least_once (p : ℚ) (h : p = 3 / 4) :
  probability_at_least_once_snow = 1023 / 1024 := by
  sorry

end probability_snow_at_least_once_l150_150805


namespace find_adult_ticket_cost_l150_150093

noncomputable def adult_ticket_cost (A : ℝ) : Prop :=
  let num_adults := 152
  let num_children := num_adults / 2
  let children_ticket_cost := 2.50
  let total_receipts := 1026
  total_receipts = num_adults * A + num_children * children_ticket_cost

theorem find_adult_ticket_cost : adult_ticket_cost 5.50 :=
by
  sorry

end find_adult_ticket_cost_l150_150093


namespace coeff_x10_eq_zero_l150_150140

theorem coeff_x10_eq_zero (x : ℝ) : 
  polynomial.coeff ((2 - 3 * polynomial.X + 2 * polynomial.X ^ 2)^5) 10 = 0 := 
sorry

end coeff_x10_eq_zero_l150_150140


namespace find_quantities_max_pendants_l150_150368

noncomputable def num_items (x y : ℕ) : Prop :=
  x + y = 180 ∧ 80 * x + 50 * y = 11400

theorem find_quantities : ∃ (x y : ℕ), num_items x y ∧ x = 80 ∧ y = 100 :=
by
  have h1 : 80 + 100 = 180, by norm_num
  have h2 : 80 * 80 + 50 * 100 = 11400, by norm_num
  exact ⟨80, 100, ⟨h1, h2⟩, rfl, rfl⟩

noncomputable def profit_formula (m : ℕ) : Prop :=
  (180 - m) * 20 + m * 10 ≥ 2900

theorem max_pendants (m : ℕ) : ∃ m, profit_formula m ∧ m = 70 :=
by
  have h : (180 - 70) * 20 + 70 * 10 = 2900, by norm_num
  exact ⟨70, h⟩

end find_quantities_max_pendants_l150_150368


namespace Gretchen_weekend_profit_l150_150206

theorem Gretchen_weekend_profit :
  let saturday_revenue := 24 * 25
  let sunday_revenue := 16 * 15
  let total_revenue := saturday_revenue + sunday_revenue
  let park_fee := 5 * 6 * 2
  let art_supplies_cost := 8 * 2
  let total_expenses := park_fee + art_supplies_cost
  let profit := total_revenue - total_expenses
  profit = 764 :=
by
  sorry

end Gretchen_weekend_profit_l150_150206


namespace opp_edges_equal_l150_150264

variables {α : Type*} [MetricSpace α] [NormedSpace ℝ α]

structure Tetrahedron (α : Type*) [metric_space α] [normed_space ℝ α] :=
(A B C D : α)

def circumradius {α : Type*} [MetricSpace α] [NormedSpace ℝ α] (t : Tetrahedron α) : α → ℝ
| t.A := sorry
| t.B := sorry
| t.C := sorry
| t.D := sorry

noncomputable def radii_equal {α : Type*} [MetricSpace α] [NormedSpace ℝ α] 
  (t : Tetrahedron α) : Prop :=
circumradius t t.A = circumradius t t.B ∧
circumradius t t.B = circumradius t t.C ∧
circumradius t t.C = circumradius t t.D

theorem opp_edges_equal {α : Type*} [MetricSpace α] [NormedSpace ℝ α] 
  (t : Tetrahedron α) (h : radii_equal t) : 
  dist t.A t.B = dist t.C t.D ∧
  dist t.A t.C = dist t.B t.D ∧
  dist t.A t.D = dist t.B t.C := 
sorry

end opp_edges_equal_l150_150264


namespace exists_congruent_triangle_covering_with_parallel_side_l150_150630

variable {Point : Type}
variable [MetricSpace Point]
variable {Triangle : Type}
variable {Polygon : Type}

-- Definitions of triangle and polygon covering relationships.
def covers (T : Triangle) (P : Polygon) : Prop := sorry 
def congruent (T1 T2 : Triangle) : Prop := sorry
def side_parallel_or_coincident (T : Triangle) (P : Polygon) : Prop := sorry

-- Statement: Given a triangle covering a polygon, there exists a congruent triangle which covers the polygon 
-- and has one side parallel to or coincident with a side of the polygon.
theorem exists_congruent_triangle_covering_with_parallel_side 
  (ABC : Triangle) (M : Polygon) 
  (h_cover : covers ABC M) : 
  ∃ Δ : Triangle, congruent Δ ABC ∧ covers Δ M ∧ side_parallel_or_coincident Δ M := 
sorry

end exists_congruent_triangle_covering_with_parallel_side_l150_150630


namespace smallest_x_plus_y_l150_150605

theorem smallest_x_plus_y 
  (x y : ℕ) 
  (hx_pos : 0 < x) 
  (hy_pos : 0 < y) 
  (hxy_neq : x ≠ y)
  (h_eq : (1/x + 1/y = 1/10)) : x + y = 45 :=
sorry

end smallest_x_plus_y_l150_150605


namespace find_cost_of_baseball_l150_150327

noncomputable def total_amount : ℝ := 20.52
noncomputable def cost_of_marbles : ℝ := 9.05
noncomputable def cost_of_football : ℝ := 4.95
noncomputable def cost_of_baseball : ℝ := total_amount - (cost_of_marbles + cost_of_football)

theorem find_cost_of_baseball : cost_of_baseball = 6.52 := sorry

end find_cost_of_baseball_l150_150327


namespace general_term_a_n_sum_T_n_l150_150995

noncomputable def a_n (n : ℕ) : ℕ := n

def b_n (n : ℕ) : ℚ := 1 / (a_n n * a_n (n + 1))

def T_n (n : ℕ) : ℚ := ∑ i in Finset.range n, b_n i

theorem general_term_a_n :
  (a_2 = 2) ∧ (a_4 = 4) ∧ (∀ n : ℕ, a_n n = n) :=
by {
  sorry
}

theorem sum_T_n (n : ℕ) :
  (∀ n : ℕ, b_n n = 1 / (a_n n * a_n (n + 1))) →
  (T_n n = (n: ℚ) / (n + 1)) :=
by {
  sorry
}

end general_term_a_n_sum_T_n_l150_150995


namespace determine_k_l150_150196

-- Define the function f
def f (k : ℝ) (x : ℝ) : ℝ := k * x^2 + 2 * k * x + 1

-- State the problem
theorem determine_k (k : ℝ) :
  (∀ x ∈ Set.Icc (-3 : ℝ) 2, f k x ≤ 4) ∧ (∃ x ∈ Set.Icc (-3 : ℝ) 2, f k x = 4)
  ↔ (k = 3 / 8 ∨ k = -3) :=
by
  sorry

end determine_k_l150_150196


namespace sin_values_l150_150176

theorem sin_values (x : ℝ) 
    (h : (1 / real.cos x) - (real.sin x / real.cos x) = 5 / 3) :
    real.sin x = 53 / 68 ∨ real.sin x = -41 / 68 :=
by
  sorry

end sin_values_l150_150176


namespace snow_probability_at_least_once_l150_150808

theorem snow_probability_at_least_once :
  let p := 3 / 4
  let prob_no_snow_single_day := 1 - p
  let prob_no_snow_all_days := prob_no_snow_single_day ^ 5
  let prob_snow_at_least_once := 1 - prob_no_snow_all_days
  prob_snow_at_least_once = 1023 / 1024 :=
by
  sorry

end snow_probability_at_least_once_l150_150808


namespace coeff_x3_in_expansion_of_sum_l150_150128

noncomputable def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem coeff_x3_in_expansion_of_sum (x : ℝ) :
  let a := (1 - x)^5
  let b := (1 - x)^6
  let c := (1 - x)^7
  let d := (1 - x)^8
  let expansion := a + b + c + d
  expansion.coeff 3 = -121 :=
by
  -- Sorry is used to skip the thorough proof steps
  sorry

end coeff_x3_in_expansion_of_sum_l150_150128


namespace solution_set_of_inequality_l150_150951

theorem solution_set_of_inequality (x : ℝ) : (x + 3) * (x - 5) < 0 ↔ (-3 < x ∧ x < 5) :=
by
  sorry

end solution_set_of_inequality_l150_150951


namespace max_value_l150_150739

def max_value_expression (n : ℕ) (x : Fin n → ℝ) : ℝ :=
  ∑ k in Finset.range n, (k + 1) * (x ⟨k, by linarith only [Fin.is_lt]⟩) ^ 2 +
  ∑ i in Finset.range n, ∑ j in Finset.range n, if i < j then (i + 1 + j + 1) * x ⟨i, by linarith only [Fin.is_lt]⟩ * x ⟨j, by linarith only [Fin.is_lt]⟩ else 0

theorem max_value (n : ℕ) (h : n > 1) (x : Fin n → ℝ)
  (h_norm : ∑ i in Finset.range n, (x ⟨i, by linarith only [Fin.is_lt]⟩) ^ 2 = 1) :
  max_value_expression n x ≤ (n / 4) * (n + 1 + 2 * Real.sqrt ((n + 1) * (2 * n + 1) / 6)) := 
by sorry

end max_value_l150_150739


namespace total_value_of_coins_l150_150459

theorem total_value_of_coins (n : ℕ) (hn : n = 20) :
  let value_one_rupee := n * 1
  let value_fifty_paise := (n * 50) / 100
  let value_twenty_five_paise := (n * 25) / 100
  in value_one_rupee + value_fifty_paise + value_twenty_five_paise = 35 :=
by
  sorry

end total_value_of_coins_l150_150459


namespace largest_two_digit_prime_factor_l150_150906

theorem largest_two_digit_prime_factor :
  ∀ (product : ℕ), (product = ∏ n in finset.range (149 - 101 + 1), 101 + n) →
  (prime 73) →
  (∀ p, prime p → p < 100 → p ∣ product → p ≤ 73) :=
by
  sorry

end largest_two_digit_prime_factor_l150_150906


namespace smallest_x_plus_y_l150_150607

theorem smallest_x_plus_y 
  (x y : ℕ) 
  (hx_pos : 0 < x) 
  (hy_pos : 0 < y) 
  (hxy_neq : x ≠ y)
  (h_eq : (1/x + 1/y = 1/10)) : x + y = 45 :=
sorry

end smallest_x_plus_y_l150_150607


namespace max_sigma_squared_l150_150302

theorem max_sigma_squared (c d : ℝ) (h_pos_c : 0 < c) (h_pos_d : 0 < d) (h_c_ge_d : c ≥ d)
    (h : ∃ x y : ℝ, 0 ≤ x ∧ x < c ∧ 0 ≤ y ∧ y < d ∧ 
      c^2 + y^2 = d^2 + x^2 ∧ d^2 + x^2 = (c - x) ^ 2 + (d - y) ^ 2) : 
    σ^2 = 4 / 3 := by
  sorry

end max_sigma_squared_l150_150302


namespace ellipse_equation_l150_150959

theorem ellipse_equation (a b c : ℝ) (h0 : a > b) (h1 : b > 0) (h2 : ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1) 
  (h3 : dist (3, y) (5 - 5 / 2, 0) = 6.5) (h4 : dist (3, y) (5 + 5 / 2, 0) = 3.5) : 
  ( ∀ x y, (x^2 / 25) + (y^2 / (75 / 4)) = 1 ) :=
sorry

end ellipse_equation_l150_150959


namespace square_of_distance_is_82_l150_150067

noncomputable def square_distance_from_B_to_center (a b : ℝ) : ℝ := a^2 + b^2

theorem square_of_distance_is_82
  (a b : ℝ)
  (r : ℝ := 11)
  (ha : a^2 + (b + 7)^2 = r^2)
  (hc : (a + 3)^2 + b^2 = r^2) :
  square_distance_from_B_to_center a b = 82 := by
  -- Proof steps omitted
  sorry

end square_of_distance_is_82_l150_150067


namespace find_a12_l150_150631

noncomputable def a (n : ℕ) : ℝ := sorry

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem find_a12 :
  (arithmetic_sequence a) →
  (a 7 + a 9 = 16) →
  (a 4 = 4) →
  a 12 = 12 :=
begin
  intros h_seq h1 h2,
  sorry
end

end find_a12_l150_150631


namespace effect_on_area_l150_150045

variables (L B : ℝ)
def original_area := L * B
def new_length := 1.20 * L
def new_breadth := 0.80 * B
def new_area := new_length * new_breadth

-- Theorem: The new area is 96% of the original area
theorem effect_on_area : new_area L B = 0.96 * original_area L B :=
by sorry

end effect_on_area_l150_150045


namespace number_of_sets_satisfying_conditions_l150_150664

open Finset

theorem number_of_sets_satisfying_conditions :
  let M := {M : Finset (Fin 5) // 
              M ⊆ {0, 1, 2, 3, 4} ∧
              M ∩ {0, 1, 2} = {0, 1}} 
  in M.card = 4 :=
by sorry

end number_of_sets_satisfying_conditions_l150_150664


namespace find_a_l150_150244

theorem find_a (a : ℝ) (h1 : ∃ α : ℝ, tan α = -1/2 ∧ (cos α ≠ 0 ∧ sin α / cos α = 1/a)) : a = -2 :=
sorry

end find_a_l150_150244


namespace both_channels_l150_150501

variable (U : Type) (students : Finset U)
variable (sports arts neither both : Finset U)

-- Conditions
noncomputable def total_students := 100
noncomputable def neither_students := 3
noncomputable def sports_students := 68
noncomputable def arts_students := 55

-- Assuming facts
axiom card_students : students.card = total_students
axiom card_neither : neither.card = neither_students
axiom card_sports : sports.card = sports_students
axiom card_arts : arts.card = arts_students
axiom sports_arts_disj : students = sports ∪ arts ∪ neither

theorem both_channels :
  (sports ∩ arts).card = 26 := 
by
  sorry

end both_channels_l150_150501


namespace zebra_to_fox_ratio_l150_150333

theorem zebra_to_fox_ratio (cows foxes sheep total animals : ℕ) 
  (hcows : cows = 20) (hfoxes : foxes = 15) (hsheep : sheep = 20) (htotal : total = 100) :
  let zebras := total - (cows + foxes + sheep) in
  let ratio := zebras / gcd zebras foxes = 3 / gcd 3 1 :=
  ratio = 3 / 1 := sorry

end zebra_to_fox_ratio_l150_150333


namespace n_times_s_l150_150310

def g (f : ℝ → ℝ) : Prop := ∀ x y, f x * f y - f (x * y) = 2 * x - y

theorem n_times_s (f : ℝ → ℝ) (hf : g f) (n s : ℕ) (h_n : n = 2) (h_s : s = 3) : n * s = 6 := 
by 
  rw [h_n, h_s]
  exact Nat.mul_comm _ _

end n_times_s_l150_150310


namespace distance_focus_to_asymptote_l150_150965

-- definition of the hyperbola condition with m > 0
def hyperbola (m : ℝ) (hm : m > 0) : Prop :=
  ∀ (x y : ℝ), x^2 - m * y^2 = 3 * m

-- defining the distance function from a point (x1, y1) to a line defined as ay + bx + c = 0
def point_to_line_distance (x1 y1 a b c : ℝ) : ℝ :=
  abs (a * x1 + b * y1 + c) / real.sqrt (a^2 + b^2)

-- theorem to prove the distance from the focus of the hyperbola to its asymptote is sqrt(3)
theorem distance_focus_to_asymptote {m : ℝ} (hm : m > 0) :
  point_to_line_distance (real.sqrt (3*m + 3)) 0 (-1 / real.sqrt m) 1 0 = real.sqrt 3 :=
sorry

end distance_focus_to_asymptote_l150_150965


namespace brownies_on_counter_l150_150329

-- Define the initial number of dozen brownies
def initial_dozens : ℕ := 2

-- Define the conversion from dozens to brownies
def dozen_to_brownies (d : ℕ) : ℕ := d * 12

-- Define the initial number of brownies
def initial_brownies : ℕ := dozen_to_brownies initial_dozens

-- Define the number of brownies father ate
def father_ate : ℕ := 8

-- Define the number of brownies Mooney ate
def mooney_ate : ℕ := 4

-- Define the number of dozen brownies made the next morning
def next_morning_dozens : ℕ := 2

-- Define the number of brownies made the next morning
def next_morning_brownies : ℕ := dozen_to_brownies next_morning_dozens

-- Calculate the remaining brownies after father and Mooney ate some
def remaining_brownies : ℕ := initial_brownies - father_ate - mooney_ate

-- Calculate the total number of brownies after adding the new ones the next morning
def total_brownies : ℕ := remaining_brownies + next_morning_brownies

theorem brownies_on_counter : total_brownies = 36 := by
  sorry

end brownies_on_counter_l150_150329


namespace measure_of_smaller_angle_l150_150020

variable (x : ℝ)

-- Defining the two supplementary angles based on the given ratio.
def larger_angle := 5 * x
def smaller_angle := 4 * x

-- Given the condition of supplementary angles.
axiom supplementary_condition : larger_angle x + smaller_angle x = 180

-- Now we state the proof problem: Prove that the measure of the smaller angle is 80.
theorem measure_of_smaller_angle : 4 * (180 / 9) = 80 :=
by
  have hx : x = 180 / 9 := sorry
  rw [←hx]
  calc
    4 * x = 4 * (180 / 9) : by rw [hx]
    ...   = 80 : by norm_num

end measure_of_smaller_angle_l150_150020


namespace ways_to_fill_grid_l150_150945

theorem ways_to_fill_grid : 
  let n := 16
  let k := 14
  let grid := List.range n
  let pre_filled := {4, 13}
  ∃ f : List ℕ → ℕ,
  ∀ (r c : ℕ), 
  (1 ≤ r ∧ r ≤ 4 ∧ 1 ≤ c ∧ c ≤ 4) ∧ 
  (∀ i j, (1 ≤ i ∧ i ≤ 4 ∧ 1 ≤ j ∧ j ≤ 4) → (j > i → grid[i] < grid[j])) → 
  grid.perm f = grid.removeAll pre_filled ∧ 
  (∀ a b, a < b → pre_filled a < pre_filled b) ∧ (pre_filled = {4, 13}) →
  1120 :=
by
  sorry

end ways_to_fill_grid_l150_150945


namespace sequence_product_l150_150971

-- Definitions for the arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a1 d : ℝ), ∀ n, a n = a1 + (n - 1) * d

-- Definitions for the geometric sequence
def is_geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ (b1 r : ℝ), ∀ n, b n = b1 * r ^ (n - 1)

-- Defining the main proposition
theorem sequence_product (a b : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a)
  (h_geom  : is_geometric_sequence b)
  (h_eq    : b 7 = a 7)
  (h_cond  : 2 * a 2 - (a 7) ^ 2 + 2 * a 12 = 0) :
  b 3 * b 11 = 16 :=
sorry

end sequence_product_l150_150971


namespace smallest_sum_of_xy_l150_150609

theorem smallest_sum_of_xy (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hxy : x ≠ y)
  (hcond : (1 : ℚ) / x + (1 : ℚ) / y = 1 / 10) : x + y = 45 :=
sorry

end smallest_sum_of_xy_l150_150609


namespace train_length_approx_200_l150_150090

-- Definition of the given conditions
def time_to_cross := 3.3330666879982935 -- in seconds
def speed_km_hr := 216 -- in km/hr

-- Conversion factor from km/hr to m/s
def speed_ms := speed_km_hr * (1000 / 3600 : ℝ)

-- Calculating the length of the train
def length_of_train := speed_ms * time_to_cross

-- Prove that the length of the train is approximately 200 meters
theorem train_length_approx_200 : length_of_train ≈ 200 :=
by
  -- Proof will be filled in here
  sorry

end train_length_approx_200_l150_150090


namespace number_reciprocal_100_l150_150670

theorem number_reciprocal_100 (x : ℝ) (h : 8 * x = 16) : 200 * (1 / x) = 100 :=
by
  have hx : x = 2 := by
    rw [← mul_div_assoc, mul_comm, div_eq_iff_mul_eq, mul_comm]
    assumption
  rw [hx, one_div, mul_inv_cancel]
  norm_num
  norm_num
  exact two_ne_zero

end number_reciprocal_100_l150_150670


namespace distribute_6_balls_in_3_boxes_l150_150656

def number_of_ways_to_distribute_balls (balls boxes : Nat) : Nat :=
  boxes ^ balls

theorem distribute_6_balls_in_3_boxes : number_of_ways_to_distribute_balls 6 3 = 729 := by
  sorry

end distribute_6_balls_in_3_boxes_l150_150656


namespace find_m_l150_150263

variable (a b m : ℝ)

def right_triangle (a b c : ℝ) : Prop := a^2 + b^2 = c^2

theorem find_m 
  (h₁ : right_triangle a b 5)
  (h₂ : a + b = 2*m - 1)
  (h₃ : a * b = 4 * (m - 1)) : 
  m = 4 := 
sorry

end find_m_l150_150263


namespace find_k_values_l150_150703

theorem find_k_values (k : ℝ) : 
  ((2 * 1 + 3 * k = 0) ∨
   (1 * 2 + (3 - k) * 3 = 0) ∨
   (1 * 1 + (3 - k) * k = 0)) →
   (k = -2/3 ∨ k = 11/3 ∨ k = (3 + Real.sqrt 3)/2 ∨ k = (3 - Real.sqrt 3)/2) := 
by
  sorry

end find_k_values_l150_150703


namespace exists_p_seq_l150_150352

theorem exists_p_seq
  (x : ℝ) (hx : Irrational x) (hx0 : 0 < x) (hx1 : x < 1)
  (n : ℕ) (hn : 0 < n) :
  ∃ (p : ℕ → ℕ), (StrictMono p) ∧ (∀ i, 1 ≤ i ∧ i ≤ n → 0 < x - (∑ i in Finset.range n, 1 / (p i)) ∧ x - (∑ i in Finset.range n, 1 / (p i)) < 1 / ((nat.factorial n) * (nat.factorial n + 1))) :=
sorry

end exists_p_seq_l150_150352


namespace line_through_origin_and_point_l150_150988

theorem line_through_origin_and_point :
  ∃ (k : ℝ), ∀ (x y : ℝ), (x = 0 ∧ y = 0) ∨ (x = -4 ∧ y = 3) → y = k * x :=
begin
  use -3/4,
  intros x y h,
  cases h,
  { rw [h.left, h.right, mul_zero], },
  { rw [h.left, h.right], },
  sorry
end

end line_through_origin_and_point_l150_150988


namespace andrea_living_room_area_l150_150864

/-- Given that 60% of Andrea's living room floor is covered by a carpet 
     which has dimensions 4 feet by 9 feet, prove that the area of 
     Andrea's living room floor is 60 square feet. -/
theorem andrea_living_room_area :
  ∃ A, (0.60 * A = 4 * 9) ∧ A = 60 :=
by
  sorry

end andrea_living_room_area_l150_150864


namespace sum_of_integers_is_23_l150_150815

theorem sum_of_integers_is_23
  (x y : ℕ) (x_pos : 0 < x) (y_pos : 0 < y) (h : x * y + x + y = 155) 
  (rel_prime : Nat.gcd x y = 1) (x_lt_30 : x < 30) (y_lt_30 : y < 30) :
  x + y = 23 :=
by
  sorry

end sum_of_integers_is_23_l150_150815


namespace last_digit_of_2_pow_2004_l150_150759

theorem last_digit_of_2_pow_2004 : (2 ^ 2004) % 10 = 6 := 
by {
  sorry
}

end last_digit_of_2_pow_2004_l150_150759


namespace intersection_line_constant_l150_150472

-- Definitions based on conditions provided:
def circle1_eq (x y : ℝ) : Prop := (x + 6)^2 + (y - 2)^2 = 144
def circle2_eq (x y : ℝ) : Prop := (x - 3)^2 + (y - 9)^2 = 65

-- The theorem statement
theorem intersection_line_constant (c : ℝ) : 
  (∃ x y : ℝ, circle1_eq x y ∧ circle2_eq x y ∧ x + y = c) ↔ c = 6 :=
by
  sorry

end intersection_line_constant_l150_150472


namespace parametric_circle_eqn_l150_150577

variables (t x y : ℝ)

theorem parametric_circle_eqn (h1 : y = t * x) (h2 : x^2 + y^2 - 4 * y = 0) :
  x = 4 * t / (1 + t^2) ∧ y = 4 * t^2 / (1 + t^2) :=
by
  sorry

end parametric_circle_eqn_l150_150577


namespace value_of_y_l150_150931

theorem value_of_y (x y : ℝ) (h1 : y = (x^2 - 9) / (x - 3)) (h2 : y = 3 * x) : y = 9 / 2 :=
sorry

end value_of_y_l150_150931


namespace mary_baseball_cards_count_l150_150322

def mary_initial_cards := 18
def mary_torn_cards := 8
def fred_gift_cards := 26
def mary_bought_cards := 40
def exchange_with_tom := 0
def mary_lost_cards := 5
def trade_with_lisa_gain := 1
def exchange_with_alex_loss := 2

theorem mary_baseball_cards_count : 
  mary_initial_cards - mary_torn_cards
  + fred_gift_cards
  + mary_bought_cards 
  + exchange_with_tom
  - mary_lost_cards
  + trade_with_lisa_gain 
  - exchange_with_alex_loss 
  = 70 := 
by
  sorry

end mary_baseball_cards_count_l150_150322


namespace compute_x_l150_150742

-- Define the problem conditions as hypotheses
variables (x y : ℝ)
hypothesis (h1 : x < y)
hypothesis (h2 : 0 < x)
hypothesis (h3 : 0 < y)
hypothesis (h4 : Real.sqrt x + Real.sqrt y = 4)
hypothesis (h5 : Real.sqrt (x + 2) + Real.sqrt (y + 2) = 5)

-- State the proposition that needs to be proved
theorem compute_x : x = 49 / 36 :=
by
  -- The proof is omitted
  sorry

end compute_x_l150_150742


namespace sum_of_solutions_l150_150561

theorem sum_of_solutions (x : ℝ) (hx : x + 36 / x = 12) : x = 6 ∨ x = -6 := sorry

end sum_of_solutions_l150_150561


namespace wrapping_paper_area_l150_150463

theorem wrapping_paper_area (s : ℝ) :
  let base_side := 2 * s
  let height := 3 * s
  let quadrant_area := base_side * height / 2 + 2 * (base_side / 2 * height / 2) 
  4 * quadrant_area = 24 * s^2 :=
by
  let base_side := 2 * s
  let height := 3 * s
  let quadrant_area := base_side * height / 2 + 2 * (base_side / 2 * height / 2)
  have h : 4 * quadrant_area = 24 * s^2
  exact h

end wrapping_paper_area_l150_150463


namespace complex_exp_cos_l150_150581

theorem complex_exp_cos (z : ℂ) (α : ℂ) (n : ℕ) (h : z + z⁻¹ = 2 * Complex.cos α) : 
  z^n + z⁻¹^n = 2 * Complex.cos (n * α) :=
by
  sorry

end complex_exp_cos_l150_150581


namespace present_age_of_son_l150_150485

-- Define variables for the current ages of the son and the man (father).
variables (S M : ℕ)

-- Define the conditions:
-- The man is 35 years older than his son.
def condition1 : Prop := M = S + 35

-- In two years, the man's age will be twice the age of his son.
def condition2 : Prop := M + 2 = 2 * (S + 2)

-- The theorem that we need to prove:
theorem present_age_of_son : condition1 S M ∧ condition2 S M → S = 33 :=
by
  -- Add sorry to skip the proof.
  sorry

end present_age_of_son_l150_150485


namespace segment_if_length_l150_150051

noncomputable def r := (10 / 2) * Real.sqrt(4 + 2 * Real.sqrt 2)

def A : ℝ × ℝ := (r, 0)
def C : ℝ × ℝ := (0, r)
def D : ℝ × ℝ := (-Real.sqrt(2) / 2 * r, Real.sqrt(2) / 2 * r)
def F : ℝ × ℝ := (-Real.sqrt(2) / 2 * r, -Real.sqrt(2) / 2 * r)

-- Circle with center A and radius AC
def circle_A (x y : ℝ) : Prop := (x - r)^2 + y^2 = r^2
-- Circle with center D and radius CD
def circle_D (x y : ℝ) : Prop := (x + Real.sqrt(2) / 2 * r) ^ 2 + (y - Real.sqrt(2) / 2 * r) ^ 2 = 100

-- Intersection point different from C
axiom I_intersection (I : ℝ × ℝ) (hI : I ≠ C) : 
  circle_A I.1 I.2 ∧ circle_D I.1 I.2

-- Prove the length of segment IF is 10
theorem segment_if_length : ∃ (I : ℝ × ℝ), I_intersection I ∧ Real.sqrt((I.1 - F.1) ^ 2 + (I.2 - F.2) ^ 2) = 10 :=
sorry

end segment_if_length_l150_150051


namespace jack_pays_back_l150_150283

-- Define the principal amount P and interest rate r
def principal_amount : ℝ := 1200
def interest_rate : ℝ := 0.1

-- Define the interest and the total amount Jack has to pay back
def interest : ℝ := interest_rate * principal_amount
def total_amount : ℝ := principal_amount + interest

-- State the theorem to prove that the total amount Jack pays back is 1320
theorem jack_pays_back : total_amount = 1320 := by
  sorry

end jack_pays_back_l150_150283


namespace find_line_eq_p_parallel_tangent_curve_l150_150551

theorem find_line_eq_p_parallel_tangent_curve (P M : ℝ × ℝ) (curve : ℝ → ℝ)
  (h_P : P = (-1, 2))
  (h_M : M = (1, 1))
  (h_curve : curve = λ x, 3 * x^2 - 4 * x + 2) :
  ∃ a b c : ℝ, a * P.1 + b * P.2 + c = 0 ∧ a * M.1 + b * (6 * M.1 - 4) + c = 0 ∧
              a = 2 ∧ b = -1 ∧ c = 4 :=
by
  sorry

end find_line_eq_p_parallel_tangent_curve_l150_150551


namespace comparison_of_logs_l150_150575

def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

variables (a b c : ℝ)

-- Conditions
def h1 : a = log_base 2 3.6 := rfl
def h2 : b = log_base 4 3.2 := rfl
def h3 : c = log_base 4 3.6 := rfl

-- Theorem statement
theorem comparison_of_logs (a b c : ℝ) (h1 : a = log_base 2 3.6) (h2 : b = log_base 4 3.2) (h3 : c = log_base 4 3.6) : a > c ∧ c > b := 
by
  sorry

end comparison_of_logs_l150_150575


namespace general_formula_a_sum_of_bn_l150_150707

-- Problem statement definitions
def S (n : ℕ) : ℝ := sorry  -- Sum of the first n terms of sequence {a_n}
def a (n : ℕ) : ℝ := sorry  -- General formula for the sequence {a_n}
def b (n : ℕ) : ℝ := 1 / (Real.log2 (a n) * Real.log2 (a (n + 1)))  -- Given bn definition
def T (n : ℕ) : ℝ := (Finset.range n).sum (λ k, b k)  -- Sum of first n terms of sequence {b_n}

-- Problem statement conditions
axiom h1 : ∀ n, 3 * S n = 1 - a n

-- Questions to be proved
theorem general_formula_a : ∀ n, a n = (1/4 : ℝ)^n := sorry

theorem sum_of_bn : ∀ n, T n = n / (4 * (n + 1)) := sorry

end general_formula_a_sum_of_bn_l150_150707


namespace reciprocal_of_neg_eight_l150_150401

theorem reciprocal_of_neg_eight : (1 / (-8 : ℝ)) = -1 / 8 := sorry

end reciprocal_of_neg_eight_l150_150401


namespace find_value_l150_150312

variables (x1 x2 y1 y2 : ℝ)

def condition1 := x1 ^ 2 + 5 * x2 ^ 2 = 10
def condition2 := x2 * y1 - x1 * y2 = 5
def condition3 := x1 * y1 + 5 * x2 * y2 = Real.sqrt 105

theorem find_value (h1 : condition1 x1 x2) (h2 : condition2 x1 x2 y1 y2) (h3 : condition3 x1 x2 y1 y2) :
  y1 ^ 2 + 5 * y2 ^ 2 = 23 :=
sorry

end find_value_l150_150312


namespace vector_ratio_l150_150298

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (C D Q : V) {p q : ℝ}

theorem vector_ratio (h : CQ:QD = 3:5) (hC : C ∈ ℝ^3) (hD : D ∈ ℝ^3) (hQ : Q ∈ ℝ^3):
  Q = (3/8 : ℝ) • C + (5/8 : ℝ) • D := sorry

end vector_ratio_l150_150298


namespace slope_angle_of_given_line_l150_150001

theorem slope_angle_of_given_line : 
  ∃ α : ℝ, x - sqrt 3 * y - 1 = 0 -> 0 ≤ α ∧ α ≤ π ∧ tan α = sqrt 3 / 3 :=
sorry

end slope_angle_of_given_line_l150_150001


namespace B_initial_investment_l150_150866

-- Definitions for investments and conditions
def A_init_invest : Real := 3000
def A_later_invest := 2 * A_init_invest

def A_yearly_investment := (A_init_invest * 6) + (A_later_invest * 6)

-- The amount B needs to invest for the yearly investment to be equal in the profit ratio 1:1
def B_investment (x : Real) := x * 12 

-- Definition of the proof problem
theorem B_initial_investment (x : Real) : A_yearly_investment = B_investment x → x = 4500 := 
by 
  sorry

end B_initial_investment_l150_150866


namespace part_a_l150_150022

def is_tricubic (k : ℕ) : Prop :=
  ∃ a b c : ℕ, k = a^3 + b^3 + c^3

theorem part_a : ∃ (n : ℕ), is_tricubic n ∧ ¬ is_tricubic (n + 2) ∧ ¬ is_tricubic (n + 28) :=
by 
  let n := 3 * (3*1+1)^3
  exists n
  sorry

end part_a_l150_150022


namespace sin_omega_max_increasing_l150_150639

theorem sin_omega_max_increasing (ω : ℕ) :
  (∃ I : set ℝ, closed_interval I ∧ length I = 1 ∧ ∃ x1 x2 ∈ I, x1 ≠ x2 ∧ sin(ω * x1) = 1 ∧ sin(ω * x2) = 1)
  ∧ (∀ x1 x2 ∈ I, (I = [-π / 16, π / 15] ∧ x1 < x2) → sin(ω * x1) < sin(ω * x2))
  → ω = 8 :=
by
  sorry

end sin_omega_max_increasing_l150_150639


namespace conditional_probability_complement_event_l150_150172

variables {Ω : Type*} [ProbabilitySpace Ω]
variable {A B : Event Ω}
variable P : ProbabilityMeasure Ω

theorem conditional_probability_complement_event :
  P A = 2/3 → P B = 5/8 → P (A ∩ B) = 1/2 → P (B | Aᶜ) = 3/8 :=
by
  intros hA hB hAB
  sorry

end conditional_probability_complement_event_l150_150172


namespace largest_common_value_less_than_1000_l150_150129

theorem largest_common_value_less_than_1000 :
  ∃ a : ℕ, 
    (∃ n : ℕ, a = 4 + 5 * n) ∧
    (∃ m : ℕ, a = 5 + 10 * m) ∧
    a % 4 = 1 ∧
    a < 1000 ∧
    (∀ b : ℕ, 
      (∃ n : ℕ, b = 4 + 5 * n) ∧
      (∃ m : ℕ, b = 5 + 10 * m) ∧
      b % 4 = 1 ∧
      b < 1000 → 
      b ≤ a) ∧ 
    a = 989 :=
by
  sorry

end largest_common_value_less_than_1000_l150_150129


namespace find_positive_integer_pair_l150_150145

noncomputable def quadratic_has_rational_solutions (d : ℤ) : Prop :=
  ∃ x : ℚ, 7 * x^2 + 13 * x + d = 0

theorem find_positive_integer_pair :
  ∃ (d1 d2 : ℕ), 
  d1 > 0 ∧ d2 > 0 ∧ 
  quadratic_has_rational_solutions d1 ∧ quadratic_has_rational_solutions d2 ∧ 
  d1 * d2 = 2 := 
sorry -- Proof left as an exercise

end find_positive_integer_pair_l150_150145


namespace parallel_vectors_angle_l150_150730

theorem parallel_vectors_angle (x : ℝ) (h : (∀α, α > 0 → α < π / 2 → (sin x, (3:ℝ) / 4) = (α * (1 / 3), α * (1 / 2 * cos x)))) :
  x = π / 4 :=
by
  sorry

end parallel_vectors_angle_l150_150730


namespace remaining_bollards_to_be_installed_l150_150899

-- Definitions based on the problem's conditions
def total_bollards_per_side := 4000
def sides := 2
def total_bollards := total_bollards_per_side * sides
def installed_fraction := 3 / 4
def installed_bollards := installed_fraction * total_bollards
def remaining_bollards := total_bollards - installed_bollards

-- The statement to be proved
theorem remaining_bollards_to_be_installed : remaining_bollards = 2000 :=
  by sorry

end remaining_bollards_to_be_installed_l150_150899


namespace triangle_inequality_sum_2_l150_150504

theorem triangle_inequality_sum_2 (a b c : ℝ) (h_triangle : a + b + c = 2) (h_side_ineq : a + c > b ∧ a + b > c ∧ b + c > a):
  1 ≤ a * b + b * c + c * a - a * b * c ∧ a * b + b * c + c * a - a * b * c ≤ 1 + 1 / 27 :=
by
  sorry

end triangle_inequality_sum_2_l150_150504


namespace find_equation_of_circle_l150_150584

-- Given conditions
variables (a r : ℝ)

-- The circle is symmetrical about the y-axis and passes through the point (1, 0)
def circle_passes_focus : a ≠ 0 ∧ 1 + a^2 = r^2 :=
begin
  sorry
end

-- The circle is divided by y = x into two arcs with a length ratio of 1:2
def arc_length_ratio : abs a / sqrt 2 = abs r / 2 :=
begin
  sorry
end

-- Goal: find the equation of circle c, which is x^2 + (y - a)^2 = r^2
theorem find_equation_of_circle (a r : ℝ)
  (h₁ : a ≠ 0)
  (h₂ : 1 + a^2 = r^2)
  (h₃ : abs a / sqrt 2 = abs r / 2) :
  (∃ a : ℝ, a = 1 ∨ a = -1) ∧ r^2 = 2 :=
sorry

end find_equation_of_circle_l150_150584


namespace ratio_of_administrators_to_teachers_l150_150692

-- Define the conditions
def graduates : ℕ := 50
def parents_per_graduate : ℕ := 2
def teachers : ℕ := 20
def total_chairs : ℕ := 180

-- Calculate intermediate values
def parents : ℕ := graduates * parents_per_graduate
def graduates_and_parents_chairs : ℕ := graduates + parents
def total_graduates_parents_teachers_chairs : ℕ := graduates_and_parents_chairs + teachers
def administrators : ℕ := total_chairs - total_graduates_parents_teachers_chairs

-- Specify the theorem to prove the ratio of administrators to teachers
theorem ratio_of_administrators_to_teachers : administrators / teachers = 1 / 2 :=
by
  -- Proof is omitted; placeholder 'sorry'
  sorry

end ratio_of_administrators_to_teachers_l150_150692


namespace count_of_divisibles_l150_150211

theorem count_of_divisibles (n : ℕ) (h : n < 500) : 
  (∃ k, n = 2 * 3 * 5 * 7 * k ∧ k > 0) →
  (∃! k, k = 2) :=
by {
  have LCM := 210,
  -- sorry is used to skip the proof
  sorry
}

end count_of_divisibles_l150_150211


namespace abs_eq_solution_l150_150442

theorem abs_eq_solution (x : ℝ) (h : abs (x - 3) = abs (x + 2)) : x = 1 / 2 :=
sorry

end abs_eq_solution_l150_150442


namespace find_b_l150_150274

theorem find_b 
    (x1 x2 b c : ℝ)
    (h_distinct : x1 ≠ x2)
    (h_root_x : ∀ x, (x^2 + 5 * b * x + c = 0) → x = x1 ∨ x = x2)
    (h_common_root : ∃ y, (y^2 + 2 * x1 * y + 2 * x2 = 0) ∧ (y^2 + 2 * x2 * y + 2 * x1 = 0)) :
  b = 1 / 10 := 
sorry

end find_b_l150_150274


namespace rearrange_marked_cells_below_diagonal_l150_150259

theorem rearrange_marked_cells_below_diagonal (n : ℕ) (marked_cells : Finset (Fin n × Fin n)) :
  marked_cells.card = n - 1 →
  ∃ row_permutation col_permutation : Equiv (Fin n) (Fin n), ∀ (i j : Fin n),
    (row_permutation i, col_permutation j) ∈ marked_cells → j < i :=
by
  sorry

end rearrange_marked_cells_below_diagonal_l150_150259


namespace stone_123_is_12_l150_150136

/-- Definitions: 
  1. Fifteen stones arranged in a circle counted in a specific pattern: clockwise and counterclockwise.
  2. The sequence of stones enumerated from 1 to 123
  3. The repeating pattern occurs every 28 stones
-/
def stones_counted (n : Nat) : Nat :=
  if n % 28 <= 15 then (n % 28) else (28 - (n % 28) + 1)

theorem stone_123_is_12 : stones_counted 123 = 12 :=
by
  sorry

end stone_123_is_12_l150_150136


namespace max_value_l150_150458

theorem max_value (a b c : ℕ) (h1 : a = 2^35) (h2 : b = 26) (h3 : c = 1) : max a (max b c) = 2^35 :=
by
  -- This is where the proof would go
  sorry

end max_value_l150_150458


namespace quadratic_has_two_distinct_real_roots_l150_150446

/-- The quadratic equation x^2 + 2x - 3 = 0 has two distinct real roots. -/
theorem quadratic_has_two_distinct_real_roots :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x₁ ^ 2 + 2 * x₁ - 3 = 0) ∧ (x₂ ^ 2 + 2 * x₂ - 3 = 0) := by
sorry

end quadratic_has_two_distinct_real_roots_l150_150446


namespace prove_even_and_odd_l150_150736

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

variables (f g : ℝ → ℝ)

hypothesis h_odd_f : is_odd f
hypothesis h_even_g : is_even g

theorem prove_even_and_odd:
  (is_even (λ x, |f x| + g x)) ∧ (is_odd (λ x, f x * abs (g x))) :=
by
  sorry

end prove_even_and_odd_l150_150736


namespace factor_quadratic_l150_150235

theorem factor_quadratic (m p : ℝ) (h : (m - 8) ∣ (m^2 - p * m - 24)) : p = 5 :=
sorry

end factor_quadratic_l150_150235


namespace angle_between_skew_lines_l150_150981

noncomputable def angle_skew_lines {R : Type*} [linear_ordered_field R] (a b : euclidean_space R (fin 3))  := 
  acos ((real_inner a b) / (norm a * norm b))

variables {a b : euclidean_space ℝ (fin 3)} (h₁ : a = ![-1, 1, 0]) (h₂ : b = ![1, 0, -1])

theorem angle_between_skew_lines : 
  angle_skew_lines a b = real.pi / 3 :=
by sorry

end angle_between_skew_lines_l150_150981


namespace bicycle_position_and_journey_time_l150_150424

-- Definitions: walking and cycling speeds
def walk_speed := 4 -- km/h
def cycle_speed := 20 -- km/h

-- The total distance to the stadium
def total_distance := 20 -- km

-- Let x be the distance the second brother walks before finding the bicycle
variable (x : ℝ)

-- Total journey time for the second brother (walking and cycling)
def second_brother_total_time (x : ℝ) : ℝ :=
  (x / walk_speed) + ((total_distance - x) / cycle_speed)

-- Total journey time for the first brother (cycling and walking)
def first_brother_total_time (x : ℝ) : ℝ :=
  (x / cycle_speed) + ((total_distance - x) / walk_speed)

-- Equality of total journey times
def journey_time_equality (x : ℝ) : Prop :=
  second_brother_total_time x = first_brother_total_time x

-- The first brother should leave the bicycle at the midpoint (10 km from the starting point)
-- and the total journey should take 3 hours.
theorem bicycle_position_and_journey_time : journey_time_equality 10 ∧ first_brother_total_time 10 = 3 :=
sorry

end bicycle_position_and_journey_time_l150_150424


namespace digit_B_value_l150_150387

theorem digit_B_value (B : ℕ) (h : 4 * 1000 + B * 100 + B * 10 + 2 % 9 = 0):
  B = 6 :=
begin
  sorry
end

end digit_B_value_l150_150387


namespace maya_additional_cars_l150_150753

theorem maya_additional_cars : 
  ∃ n : ℕ, 29 + n ≥ 35 ∧ (29 + n) % 7 = 0 ∧ n = 6 :=
by
  sorry

end maya_additional_cars_l150_150753


namespace equilateral_triangle_side_length_l150_150425

-- Definitions from the problem's conditions
variables (R r x : ℝ)

-- The theorem stating the required property
theorem equilateral_triangle_side_length 
  (R_pos : 0 < R) (r_pos : 0 < r) (tangent : (R - r) > 0) : 
  x = (real.sqrt 3) * R * r / real.sqrt (R^2 + r^2 - R*r) :=
sorry

end equilateral_triangle_side_length_l150_150425


namespace find_f_at_6_5_l150_150986

noncomputable def f : ℝ → ℝ := sorry

axiom even_function (x : ℝ) : f (-x) = f x
axiom functional_equation (x : ℝ) : f (x + 2) = - (1 / f x)
axiom initial_condition (x : ℝ) (h : 1 ≤ x ∧ x ≤ 2) : f x = x - 2

theorem find_f_at_6_5 : f 6.5 = -0.5 := by
  sorry

end find_f_at_6_5_l150_150986


namespace range_of_n_l150_150997

def hyperbola_equation (m n : ℝ) : Prop :=
  (m^2 + n) * (3 * m^2 - n) > 0

def foci_distance (m n : ℝ) : Prop :=
  (m^2 + n) + (3 * m^2 - n) = 4

theorem range_of_n (m n : ℝ) :
  hyperbola_equation m n ∧ foci_distance m n →
  -1 < n ∧ n < 3 :=
by
  intro h
  have hyperbola_condition := h.1
  have distance_condition := h.2
  sorry

end range_of_n_l150_150997


namespace min_value_f_l150_150167

theorem min_value_f : 
  ∃ (a : Fin 2020 → ℕ), 
    (a 0 = 1 ∧ a 2019 = 99 ∧ 
     (∀ i, 0 ≤ i ∧ i < 2019 → a i ≤ a (i + 1))) ∧ 
     (∑ i in Finset.range 2019, (a i) ^ 2 - 
      ∑ i in Finset.range 1009, (a (2 * i)) * (a (2 * i + 2)) = 43000) :=
begin
  sorry,
end

end min_value_f_l150_150167


namespace sum_of_k_values_l150_150837

-- Conditions
def P (x : ℝ) : ℝ := x^2 - 4 * x + 3
def Q (x k : ℝ) : ℝ := x^2 - 6 * x + k

-- Statement of the mathematical problem
theorem sum_of_k_values (k1 k2 : ℝ) (h1 : P 1 = 0) (h2 : P 3 = 0) 
  (h3 : Q 1 k1 = 0) (h4 : Q 3 k2 = 0) : k1 + k2 = 14 := 
by
  -- Here we would proceed with the proof steps corresponding to the solution
  sorry

end sum_of_k_values_l150_150837


namespace incorrect_connection_probability_is_correct_l150_150414

noncomputable def incorrect_connection_probability : ℝ :=
  let p := 0.02 in
  let C := (n k : ℕ) => Nat.choose n k in
  let r2 := 1/9 in
  let r3 := 8/81 in
  C 3 2 * p^2 * (1 - p) * r2 + C 3 3 * p^3 * r3

theorem incorrect_connection_probability_is_correct :
  incorrect_connection_probability ≈ 0.000131 :=
  sorry

end incorrect_connection_probability_is_correct_l150_150414


namespace multiples_of_15_between_35_and_200_l150_150654

theorem multiples_of_15_between_35_and_200 : 
  ∃ n : ℕ, ∀ k : ℕ, 35 < k * 15 ∧ k * 15 < 200 ↔ k = n :=
begin
  sorry,
end

end multiples_of_15_between_35_and_200_l150_150654


namespace flu_epidemic_infection_rate_l150_150536

theorem flu_epidemic_infection_rate : 
  ∃ x : ℝ, 1 + x + x * (1 + x) = 100 ∧ x = 9 := 
by
  sorry

end flu_epidemic_infection_rate_l150_150536


namespace find_general_formulas_and_sum_l150_150628

variables {ℕ : Type} [nontrivial ℕ]

-- Definitions of arithmetic sequence a_n and sequences b_n and c_n
def a_seq (n : ℕ) : ℕ := 2 * n - 1 

def b_seq (n : ℕ) : ℕ := 2 ^ (n - 1)

def c_seq (n : ℕ) : ℕ := a_seq n * b_seq n

-- Definitions of S_n and T_n
def S_seq (n : ℕ) : ℕ := 2 * b_seq n - 1

def T_seq (n : ℕ) : ℕ := (2 * n - 3) * 2 ^ n + 3

-- Conditions
lemma a_seq_condition : a_seq 3 = 5 :=
by sorry

lemma a_seq_condition2 : a_seq 7 = 13 :=
by sorry

lemma S_seq_condition (n : ℕ) : sum_seq (finset.range n) b_seq = S_seq n :=
by sorry

-- Theorem statement
theorem find_general_formulas_and_sum (n : ℕ) : 
  (∀ n ∈ ℕ, a_seq n = 2 * n - 1) ∧ 
  (∀ n ∈ ℕ, b_seq n = 2 ^ (n - 1)) ∧ 
  (sum_seq (finset.range n) c_seq = T_seq n) :=
by sorry

end find_general_formulas_and_sum_l150_150628


namespace system_no_solution_iff_n_eq_neg_half_l150_150939

theorem system_no_solution_iff_n_eq_neg_half (x y z n : ℝ) :
  (¬ ∃ x y z, 2 * n * x + y = 2 ∧ n * y + 2 * z = 2 ∧ x + 2 * n * z = 2) ↔ n = -1/2 := by
  sorry

end system_no_solution_iff_n_eq_neg_half_l150_150939


namespace correct_chart_for_percentage_representation_l150_150423

def bar_chart_characteristic := "easily shows the quantity"
def line_chart_characteristic := "shows the quantity and reflects the changes in quantity"
def pie_chart_characteristic := "reflects the relationship between a part and the whole"

def representation_requirement := "represents the percentage of students in each grade level in the fifth grade's physical education test scores out of the total number of students in the grade"

theorem correct_chart_for_percentage_representation : 
  (representation_requirement = pie_chart_characteristic) := 
by 
   -- The proof follows from the prior definition of characteristics.
   sorry

end correct_chart_for_percentage_representation_l150_150423


namespace geometric_sequence_sum_l150_150181

/-- Given a geometric sequence with common ratio r = 2, and the sum of the first four terms
    equals 1, the sum of the first eight terms is 17. -/
theorem geometric_sequence_sum (a r : ℝ) (h : r = 2) (h_sum_four : a * (1 + r + r^2 + r^3) = 1) :
  a * (1 + r + r^2 + r^3 + r^4 + r^5 + r^6 + r^7) = 17 :=
by
  sorry

end geometric_sequence_sum_l150_150181


namespace tangent_lines_from_point_condition_l150_150491

theorem tangent_lines_from_point_condition (m : ℝ) :
  ∃ (A : ℝ × ℝ), A = (m, 2) ∧ ((m + 1)^2 > 4 → (m < -3 ∨ m > 1)) :=
by
  let A := (m, 2)
  have h : (m + 1)^2 > 4 ↔ (m < -3 ∨ m > 1) := sorry
  exact ⟨A, rfl, h⟩

end tangent_lines_from_point_condition_l150_150491


namespace eccentricity_of_hyperbola_l150_150508

-- Define the basic setup for the problem
variables (A B C H : Type) [inner_product_space ℝ A]
variables (cosCdiv2 : ℝ) (HperpBC : ℝ) (AB_dot_CA_CB : ℝ)

def conditions :=
cosCdiv2 = (2 * real.sqrt 5) / 5 ∧
HperpBC = 0 ∧
AB_dot_CA_CB = 0

-- The proof that the eccentricity of the hyperbola is 2
theorem eccentricity_of_hyperbola {A B C H : Type} [inner_product_space ℝ A]
  (cosCdiv2 : ℝ) (HperpBC: ℝ) (AB_dot_CA_CB : ℝ) 
  (hc : conditions (A B C H) cosCdiv2 HperpBC AB_dot_CA_CB) :
  eccentricity_of_hyperbola A B C H = 2 :=
sorry

end eccentricity_of_hyperbola_l150_150508


namespace arc_length_of_sector_is_one_l150_150990

-- Defining the problem conditions
def central_angle : ℝ := 1 / 5
def radius : ℝ := 5

-- Statement of the proof problem
theorem arc_length_of_sector_is_one :
  central_angle * radius = 1 :=
by
  -- The proof is omitted as per the instructions
  sorry

end arc_length_of_sector_is_one_l150_150990


namespace fiona_probability_correct_l150_150821

def lilyPads : List ℕ := List.range 16
def predators : Set ℕ := {4, 9}
def food : ℕ := 14
def startPad : ℕ := 0

-- Step probabilities we need to set manually
noncomputable def nextPadProb : ℝ := (1 : ℝ) / 3
noncomputable def jump2PadsProb : ℝ := (1 : ℝ) / 3
noncomputable def jump3PadsProb (n : ℕ) : ℝ := if n % 2 = 0 then (1 : ℝ) / 3 else 0

def validPad (n : ℕ) : Prop := n ≠ 4 ∧ n ≠ 9

theorem fiona_probability_correct :
  ∀ (n : ℕ), n = 14 → (probability to reach 14 from 0 using above rules) = (28 : ℝ) / 6561 :=
sorry

end fiona_probability_correct_l150_150821


namespace smallest_x_y_sum_l150_150599

theorem smallest_x_y_sum (x y : ℕ) (h1 : x ≠ y) (h2 : x > 0) (h3 : y > 0) (h4 : 1 / (x : ℚ) + 1 / (y : ℚ) = 1 / 10) : x + y = 45 :=
by
  sorry

end smallest_x_y_sum_l150_150599


namespace smaller_angle_at_2_30_minutes_l150_150530

theorem smaller_angle_at_2_30_minutes : 
  let degree_full_circle := 360  -- There are 360 degrees in a full circle
  let degrees_per_hour := degree_full_circle / 12  -- Each hour represents 30 degrees
  let minute_hand_angle := 180  -- The minute hand at 2:30 is at the 6 o'clock position which is 180 degrees
  let hour_hand_angle := 2 * degrees_per_hour + degrees_per_hour / 2  -- The hour hand at 2:30 is halfway between 2 and 3, so it's at 75 degrees
  let angle_between_hands := abs (minute_hand_angle - hour_hand_angle)  -- The absolute difference between the minute and hour hand
  let smaller_angle := min angle_between_hands (degree_full_circle - angle_between_hands)  -- The smaller angle is the minimum of the two possible angles
  smaller_angle = 105 := 
by
  sorry

end smaller_angle_at_2_30_minutes_l150_150530


namespace lines_parallel_and_coplanar_result_l150_150307

noncomputable def lines_parallel_and_coplanar (f g h : Line) : Prop :=
∀ (e : Line), (e.intersect f ∧ e.intersect g) → e.intersect h

theorem lines_parallel_and_coplanar_result 
  (f g h : Line)
  (distinct : f ≠ g ∧ g ≠ h ∧ f ≠ h)
  (condition : lines_parallel_and_coplanar f g h) :
  coplanar f g h ∧ parallel f g ∧ parallel g h ∧ parallel f h :=
sorry

end lines_parallel_and_coplanar_result_l150_150307


namespace jack_pays_back_l150_150279

-- conditions in the problem 
def principal : ℝ := 1200
def interest_rate : ℝ := 0.1

-- the theorem statement equivalent to the question and correct answer
theorem jack_pays_back (principal_interest: principal * interest_rate) (total_amount: principal + principal_interest) : total_amount = 1320 :=
by
  sorry

end jack_pays_back_l150_150279


namespace days_to_complete_work_together_l150_150464

theorem days_to_complete_work_together :
  (20 * 35) / (20 + 35) = 140 / 11 :=
by
  sorry

end days_to_complete_work_together_l150_150464


namespace part1_part2_l150_150637

def f (ω x : ℝ) : ℝ :=
  cos (ω * x) * sin (ω * x - π / 3) + sqrt 3 * (cos (ω * x))^2 - sqrt 3 / 4

theorem part1 (h1 : ∀ x : ℝ, x ∈ ℝ → f ω x = cos (ω * x) * sin (ω * x - π / 3) + sqrt 3 * (cos (ω * x))^2 - sqrt 3 / 4)
  (h2 : ∀ x : ℝ, ω > 0 → (f ω (x + π / 4) = f ω x) ↔ (f ω (x - π / 4) = f ω x))
  (symm_dist : ∀ x : ℝ, (f ω x = f ω (x + π / 4) ∨ f ω x = f ω (x - π / 4))) : 
  ω = 1 ∧ ∃ k : ℤ, ∀ x : ℝ, x = (1/2 : ℝ) * k * π + π / 12 :=
sorry

theorem part2 (A B a b c : ℝ) (h1 : ∀ x : ℝ, f 1 x = (1 / 2) * sin (2 * x + π / 3)) 
  (h2 : ∀ x : ℝ, f 1 A = 0)
  (h3 : sin B = 4 / 5)
  (h4 : a = sqrt 3) :
  b = 2 / 5 :=
sorry

end part1_part2_l150_150637


namespace parallel_line_with_y_intercept_l150_150797

theorem parallel_line_with_y_intercept (x y : ℝ) (m : ℝ) : 
  ((x + y + 4 = 0) → (x + y + m = 0)) ∧ (m = 1)
 := by sorry

end parallel_line_with_y_intercept_l150_150797


namespace locus_of_Q_max_oa_dot_an_l150_150594

-- Definitions based on conditions
def circle_eq (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 8
def point_N : ℝ × ℝ := (0, -1)

-- Problem statements
theorem locus_of_Q : 
  (∀ (x y : ℝ), 
    (∃ (px py : ℝ), circle_eq px py ∧ (x, y) lies_on_perpendicular_bisector (px, py) (0, -1)) 
    ↔ (y^2 / 2 + x^2 = 1)) :=
sorry

theorem max_oa_dot_an : 
  (∀ (x y : ℝ), 
    (y^2 / 2 + x^2 = 1 → 
      let A := (x, y) in 
      max (dot_product O A A (point_N.1, point_N.2)) = -1 / 2) :=
sorry

end locus_of_Q_max_oa_dot_an_l150_150594


namespace cannot_arrange_digits_l150_150705

def is_valid_arrangement (grid : ℕ → ℕ → ℕ) : Prop :=
  ∀ i j k, 1 ≤ i ∧ i ≤ 3 ∧ 1 ≤ j ∧ j ≤ 3 ∧ 1 ≤ k ∧ k ≤ 3 ∧ j ≠ k →
    grid i j ≠ grid i k ∧
    grid j i ≠ grid k i ∧
    (i = j ∨ i = 2 - j + 2) →
    grid i i ≠ grid j j ∧ 
    grid i (4 - i) ≠ grid j (4 - j)

theorem cannot_arrange_digits (grid : ℕ → ℕ → ℕ) :
  (∀ i j, 1 ≤ i ∧ i ≤ 3 ∧ 1 ≤ j ∧ j ≤ 3 → grid i j ∈ {0, 1, 2, 9}) →
  ¬ is_valid_arrangement grid :=
sorry

end cannot_arrange_digits_l150_150705


namespace find_x_l150_150883

theorem find_x (x : ℝ) 
  (h1 : (x - 2)^2 + (4 - 2)^2 = 100) 
  (h2 : x > 0) : 
  x = 2 + 4 * real.sqrt(6) := by
  sorry

end find_x_l150_150883


namespace lines_parallel_l150_150701

def point := (ℝ × ℝ × ℝ)

def direction_vector (p1 p2 : point) : point :=
  (p2.1 - p1.1, p2.2 - p1.2, p2.3 - p1.3)

def are_parallel (v1 v2 : point) : Prop :=
  ∃ k : ℝ, v1 = (k * v2.1, k * v2.2, k * v2.3)

theorem lines_parallel :
  let A : point := (1, 2, 3)
  let B : point := (-2, -1, 6)
  let C : point := (3, 2, 1)
  let D : point := (4, 3, 0)
  are_parallel (direction_vector A B) (direction_vector C D) :=
by
  let A : point := (1, 2, 3)
  let B : point := (-2, -1, 6)
  let C : point := (3, 2, 1)
  let D : point := (4, 3, 0)
  let AB := direction_vector A B
  let CD := direction_vector C D
  sorry

end lines_parallel_l150_150701


namespace ellipse_standard_equation_midpoint_trajectory_equation_l150_150975

theorem ellipse_standard_equation :
  ∃ a b : ℝ, a > b ∧ b > 0 ∧ (∀ x y, (x, y) = (2, 0) → x^2 / a^2 + y^2 / b^2 = 1) → (a = 2 ∧ b = 1) :=
sorry

theorem midpoint_trajectory_equation :
  ∀ x y : ℝ,
  (∃ x0 y0 : ℝ, x0 = 2 * x - 1 ∧ y0 = 2 * y - 1 / 2 ∧ (x0^2 / 4 + y0^2 = 1)) →
  (x - 1 / 2)^2 + 4 * (y - 1 / 4)^2 = 1 :=
sorry

end ellipse_standard_equation_midpoint_trajectory_equation_l150_150975


namespace composite_numbers_with_same_main_divisors_are_equal_l150_150320

theorem composite_numbers_with_same_main_divisors_are_equal (a b : ℕ) 
  (h_a_not_prime : ¬ Prime a)
  (h_b_not_prime : ¬ Prime b)
  (h_a_comp : 1 < a ∧ ∃ p, p ∣ a ∧ p ≠ a)
  (h_b_comp : 1 < b ∧ ∃ p, p ∣ b ∧ p ≠ b)
  (main_divisors : {d : ℕ // d ∣ a ∧ d ≠ a} = {d : ℕ // d ∣ b ∧ d ≠ b}) :
  a = b := 
sorry

end composite_numbers_with_same_main_divisors_are_equal_l150_150320


namespace propA_sufficient_not_necessary_l150_150979

variables {E F G H : Type} [affine_space E]
variables (P : affine_subspace ℝ E) (Q : affine_subspace ℝ E)

def non_coplanar : Prop :=
  ∀ (P : affine_subspace ℝ E) (Q : affine_subspace ℝ E), P ≠ Q → ¬ (∀ (x y z w : E), affine_independent ℝ ![x, y, z, w] → ∃ (t : ℝ), ∃ (s : ℝ), P t = Q s)

def not_intersect (l1 l2 : affine_subspace ℝ E) : Prop :=
  ∀ (p : E), p ∉ l1 ∨ p ∉ l2

theorem propA_sufficient_not_necessary {E F G H : Type} [affine_space E]
  (P Q : affine_subspace ℝ E) (hE : E ∈ P) (hF : F ∈ P) (hG : G ∉ P) (hH : H ∉ P) :
  (non_coplanar P) → not_intersect P Q ∧ (¬ not_intersect P Q → ¬ non_coplanar P) :=
by
  sorry

end propA_sufficient_not_necessary_l150_150979


namespace determine_alpha_l150_150964

theorem determine_alpha (α : ℝ) (h1 : α ∈ {-2, -1, -1/2, 2})
  (h2 : ∀ x, (x : ℝ) > 0 → (x^α = (-x)^α))
  (h3 : ∀ x y, (0 < x) → (x < y) → (x^α > y^α)) : α = -2 :=
by
  sorry

end determine_alpha_l150_150964


namespace geometric_series_sum_value_l150_150030

theorem geometric_series_sum_value :
  let a : ℚ := 3 / 4
  let r : ℚ := 3 / 4
  let n : ℕ := 12
  \(\sum_{k\in \mathbb{N}, 1 \leq k \leq n} a * r^(k - 1)\) = \(\frac{48758625}{16777216}\) :=
sorry

end geometric_series_sum_value_l150_150030


namespace perpendicular_bisectors_divide_into_three_equal_parts_l150_150521

noncomputable def equilateral_triangle (A B C : Point) : Prop :=
∀ a b c : ℝ, a = b ∧ b = c

noncomputable def centroid (O A B C : Point) : Prop :=
-- Definition assuming O is the centroid can be provided here.

noncomputable def is_perpendicular_bisector (D E B C O : Point) : Prop :=
-- Definition for perpendicular bisectors intersecting in points D and E.

noncomputable def divides_into_equal_parts (B C D E : Point) : Prop :=
distance B D = distance D E ∧ distance D E = distance E C

theorem perpendicular_bisectors_divide_into_three_equal_parts
  (A B C O D E : Point)
  (h_eq_triangle : equilateral_triangle A B C)
  (h_centroid : centroid O A B C)
  (h_perpendicular_bisector_OBC : is_perpendicular_bisector D E B C O) 
  : divides_into_equal_parts B C D E := 
sorry

end perpendicular_bisectors_divide_into_three_equal_parts_l150_150521


namespace unique_strictly_increasing_sequence_l150_150010

/-- There exists a unique strictly increasing sequence of nonnegative integers
    b_1, b_2, ..., b_m such that
    (2^305 + 1) / (2^17 + 1) = 2^b_1 + 2^b_2 + ... + 2^b_m and m = 153. -/
theorem unique_strictly_increasing_sequence :
  ∃! (b : Finset ℕ), (∀ (x ∈ b) (y ∈ b), x < y) ∧ 2^305 + 1 = (2^17 + 1) * (b.sum (λ x, 2^x)) ∧ b.card = 153 :=
sorry

end unique_strictly_increasing_sequence_l150_150010


namespace find_common_chord_and_distance_l150_150166

noncomputable def circle_C (x y : ℝ) : Prop := 
  x^2 + y^2 - 10*x - 10*y = 0

noncomputable def circle_M (x y : ℝ) : Prop := 
  x^2 + y^2 + 6*x + 2*y - 40 = 0

noncomputable def common_chord (x y : ℝ) : Prop := 
  4*x + 3*y - 10 = 0

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := 
  real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem find_common_chord_and_distance :
  (∀ x y, circle_C x y ∧ circle_M x y ↔ common_chord x y) ∧ 
  distance (-2) 6 4 (-2) = 10 :=
by sorry

end find_common_chord_and_distance_l150_150166


namespace sampling_method_is_systematic_l150_150689

theorem sampling_method_is_systematic :
  ∀ (num_classes num_students_per_class student_number),
  (num_classes = 10) →
  (num_students_per_class = 50) →
  (student_number = 15) →
  (∃ method, method = "Systematic sampling") :=
by
  intros num_classes num_students_per_class student_number h1 h2 h3
  use "Systematic sampling"
  sorry

end sampling_method_is_systematic_l150_150689


namespace incorrect_connection_probability_is_correct_l150_150413

noncomputable def incorrect_connection_probability : ℝ :=
  let p := 0.02 in
  let C := (n k : ℕ) => Nat.choose n k in
  let r2 := 1/9 in
  let r3 := 8/81 in
  C 3 2 * p^2 * (1 - p) * r2 + C 3 3 * p^3 * r3

theorem incorrect_connection_probability_is_correct :
  incorrect_connection_probability ≈ 0.000131 :=
  sorry

end incorrect_connection_probability_is_correct_l150_150413


namespace monotonically_increasing_interval_l150_150800

def f (x : ℝ) (a : ℝ) : ℝ := |2 * x + a| + 3

theorem monotonically_increasing_interval (a : ℝ) :
  (∀ x₁ x₂ : ℝ, 1 < x₁ → x₁ < x₂ → f x₁ a ≤ f x₂ a) → a ≥ -2 :=
by
  sorry

end monotonically_increasing_interval_l150_150800


namespace charge_18m3_charge_26m3_charge_nm3_find_n_l150_150773

-- Define the charge function
def charge (a : ℝ) (n : ℝ) : ℝ :=
  if n ≤ 12 then n * a
  else if n ≤ 20 then 12 * a + (n - 12) * 1.5 * a
  else 12 * a + 8 * 1.5 * a + (n - 20) * 2 * a

-- Theorem for different cases
theorem charge_18m3 (a : ℝ) : charge a 18 = 21 * a := by
  sorry

theorem charge_26m3 (a : ℝ) : charge a 26 = 36 * a := by
  sorry

theorem charge_nm3 (a : ℝ) (n : ℝ) (h : n > 20) : charge a n = (2 * n - 16) * a := by
  sorry

theorem find_n (a : ℝ) (n : ℝ) (h : a = 1.5) (h₁ : charge a n = 60) (h₂ : n > 20) : n = 28 := by
  sorry

end charge_18m3_charge_26m3_charge_nm3_find_n_l150_150773


namespace original_speed_correct_l150_150003

variables (t m s : ℝ)

noncomputable def original_speed (t m s : ℝ) : ℝ :=
  ((t * m + Real.sqrt (t^2 * m^2 + 4 * t * m * s)) / (2 * t))

theorem original_speed_correct (t m s : ℝ) (ht : 0 < t) : 
  original_speed t m s = (Real.sqrt (t * m * (4 * s + t * m)) - t * m) / (2 * t) :=
by
  sorry

end original_speed_correct_l150_150003


namespace brownies_on_counter_l150_150330

-- Define the initial number of dozen brownies
def initial_dozens : ℕ := 2

-- Define the conversion from dozens to brownies
def dozen_to_brownies (d : ℕ) : ℕ := d * 12

-- Define the initial number of brownies
def initial_brownies : ℕ := dozen_to_brownies initial_dozens

-- Define the number of brownies father ate
def father_ate : ℕ := 8

-- Define the number of brownies Mooney ate
def mooney_ate : ℕ := 4

-- Define the number of dozen brownies made the next morning
def next_morning_dozens : ℕ := 2

-- Define the number of brownies made the next morning
def next_morning_brownies : ℕ := dozen_to_brownies next_morning_dozens

-- Calculate the remaining brownies after father and Mooney ate some
def remaining_brownies : ℕ := initial_brownies - father_ate - mooney_ate

-- Calculate the total number of brownies after adding the new ones the next morning
def total_brownies : ℕ := remaining_brownies + next_morning_brownies

theorem brownies_on_counter : total_brownies = 36 := by
  sorry

end brownies_on_counter_l150_150330


namespace house_number_count_l150_150941

def is_two_digit_prime (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ Nat.Prime n

def valid_house_numbers : Set ℕ :=
  { n | 1000 ≤ n ∧ n < 10000 ∧ 
        (let AB := n / 100 in let CD := n % 100 in 
        AB ≠ 0 ∧ CD ≠ 0 ∧ is_two_digit_prime AB ∧ is_two_digit_prime CD ∧ AB ≠ CD) }

theorem house_number_count : 
  ∃ (n : ℕ), n = 110 ∧ (∀ x, x ∈ valid_house_numbers ↔ x = n) :=
by
  sorry

end house_number_count_l150_150941


namespace jack_pays_back_l150_150282

-- Define the principal amount P and interest rate r
def principal_amount : ℝ := 1200
def interest_rate : ℝ := 0.1

-- Define the interest and the total amount Jack has to pay back
def interest : ℝ := interest_rate * principal_amount
def total_amount : ℝ := principal_amount + interest

-- State the theorem to prove that the total amount Jack pays back is 1320
theorem jack_pays_back : total_amount = 1320 := by
  sorry

end jack_pays_back_l150_150282


namespace cyclic_quadrilateral_equality_l150_150734

variable (a b c d : ℝ)
variable (h_a h_b h_c h_d : ℝ)

theorem cyclic_quadrilateral_equality 
  (cyclic_quadrilateral : (a, b, c, d) ∈ {cyclic_quadrilaterals})
  (center_inside : (center_of_circumscribed_circle cyclic_quadrilateral) ∈ (inside cyclic_quadrilateral)) :
  a * h_c + c * h_a = b * h_d + d * h_b :=
sorry

end cyclic_quadrilateral_equality_l150_150734


namespace solve_for_x_l150_150938

theorem solve_for_x : 
  ∀ x : ℚ, x + 5/6 = 7/18 - 2/9 → x = -2/3 :=
by
  intro x
  intro h
  sorry

end solve_for_x_l150_150938


namespace lower_right_is_one_l150_150013

def initial_grid : Matrix (Fin 5) (Fin 5) (Option (Fin 5)) :=
![![some 0, none, some 1, none, none],
  ![some 1, some 3, none, none, none],
  ![none, none, none, some 4, none],
  ![none, some 4, none, none, none],
  ![none, none, none, none, none]]

theorem lower_right_is_one 
  (complete_grid : Matrix (Fin 5) (Fin 5) (Fin 5)) 
  (unique_row_col : ∀ i j k, 
      complete_grid i j = complete_grid i k ↔ j = k ∧ 
      complete_grid i j = complete_grid k j ↔ i = k)
  (matches_partial : ∀ i j, ∃ x, 
      initial_grid i j = some x → complete_grid i j = x) :
  complete_grid 4 4 = 0 := 
sorry

end lower_right_is_one_l150_150013


namespace smallest_non_lucky_multiple_of_8_correct_l150_150439

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_lucky (n : ℕ) : Prop :=
  n % sum_of_digits n = 0

def smallest_non_lucky_multiple_of_8 := 16

theorem smallest_non_lucky_multiple_of_8_correct :
  smallest_non_lucky_multiple_of_8 = 16 ∧
  is_lucky smallest_non_lucky_multiple_of_8 = false :=
by
  sorry

end smallest_non_lucky_multiple_of_8_correct_l150_150439


namespace topless_box_configurations_l150_150788

noncomputable def T_shape_foldable_configurations (squares : Finset ℕ) (additional_square : ℕ) : ℕ :=
  if (additional_square = 0 ∨ additional_square = 1 ∨ additional_square = 3 ∨ additional_square = 4 ∨ additional_square = 5 ∨ additional_square = 7)
  then 1 else 0

theorem topless_box_configurations : 
  (Finset.card (Finset.filter (λ x, x = T_shape_foldable_configurations {0, 1, 2, 3, 4, 5, 6, 7} x) {0, 1, 2, 3, 4, 5, 6, 7})) = 6 :=
by 
  sorry

end topless_box_configurations_l150_150788


namespace train_speed_l150_150893

theorem train_speed (length : ℝ) (time : ℝ) (length_eq : length = 400) (time_eq : time = 16) :
  (length / time) * (3600 / 1000) = 90 :=
by 
  rw [length_eq, time_eq]
  sorry

end train_speed_l150_150893


namespace problem1_problem2_l150_150056

-- Definitions for Problem 1
def cond1 (x t : ℝ) : Prop := |2 * x + t| - t ≤ 8
def sol_set1 (x : ℝ) : Prop := -5 ≤ x ∧ x ≤ 4

theorem problem1 {t : ℝ} : (∀ x, cond1 x t → sol_set1 x) → t = 1 :=
sorry

-- Definitions for Problem 2
def cond2 (x y z : ℝ) : Prop := x^2 + (1 / 4) * y^2 + (1 / 9) * z^2 = 2

theorem problem2 {x y z : ℝ} : cond2 x y z → x + y + z ≤ 2 * Real.sqrt 7 :=
sorry

end problem1_problem2_l150_150056


namespace minimum_modulus_of_z_l150_150968

noncomputable def quadratic_roots_are_real (z : ℂ) : Prop :=
  let Δ := (-8 * z)^2 - 4 * 4 * (4 * complex.I + 3)
  0 ≤ Δ.im ∧ Δ.re ≥ 0

theorem minimum_modulus_of_z (z : ℂ) (h : quadratic_roots_are_real z) : |z| ≥ 1 :=
sorry

end minimum_modulus_of_z_l150_150968


namespace notebook_cost_l150_150251

/-- In Mr. Numbers' class, 42 students are present, and more than half of them purchased notebooks. 
Each student who bought notebooks purchased the same number of notebooks, which was a prime number. 
The cost per notebook in cents was greater than the number of notebooks each student bought, 
and the total cost for all notebooks was 2310 cents.
Prove that the cost of a notebook in cents is 21. -/
theorem notebook_cost 
  (s : ℕ) (c : ℕ) (n : ℕ)
  (h1 : s > 21)
  (h2 : 42 = s)
  (h3 : nat.prime n)
  (h4 : c > n)
  (h5 : s * c * n = 2310) :
  c = 21 :=
sorry

end notebook_cost_l150_150251


namespace arc_length_proof_l150_150552

noncomputable def arc_length (a : ℝ) (h : a > 0) : ℝ :=
2 * ∫ (t : ℝ) in 0..(3 * Real.pi / 2), a * (Real.cos (t / 3))^2

theorem arc_length_proof (a : ℝ) (h : a > 0) :
  arc_length a h = 3 * Real.pi * a / 2 :=
sorry

end arc_length_proof_l150_150552


namespace circumscribed_circle_radius_l150_150081

noncomputable def radius_of_circumscribed_circle (theta : ℝ) : ℝ :=
  if h : theta > Real.pi / 2 ∧ theta < Real.pi then 8 * Real.sec (theta / 2) else 0

theorem circumscribed_circle_radius (theta : ℝ) (h_theta : theta > Real.pi / 2 ∧ theta < Real.pi) :
  radius_of_circumscribed_circle theta = 8 * Real.sec (theta / 2) :=
by sorry

end circumscribed_circle_radius_l150_150081


namespace water_formed_l150_150544

/-- 
The molar mass of water (H₂O) is approximately 18.015 g/mol. 
Given that 1 mole of sodium hydroxide (NaOH) reacts with 
1 mole of hydrochloric acid (HCl) to produce water (H₂O), 
we need to find the amount of water formed.
-/

theorem water_formed :
  ∀ (NaOH HCl : ℕ), (NaOH = 1) → (HCl = 1) → 
  (molar_mass_H2O : ℝ) (h_molar_mass : molar_mass_H2O = 18.015) →
  amount_of_H2O_formed = NaOH * molar_mass_H2O :=
by
  intros NaOH HCl hNaOH hHCl molar_mass_H2O h_molar_mass
  sorry

end water_formed_l150_150544


namespace sum_of_series_l150_150306

theorem sum_of_series :
  let b : ℕ → ℕ := λ n, Nat.recOn n.succ 2 (λ n b, match n.succ with
                                                 | 1 => 3
                                                 | k+2 => b k.succ + b k
                                                 end)
  in
  ∑' n, (b n) / 9^(n+1) = 1 / 29 :=
by
  let b : ℕ → ℕ := λ n, Nat.recOn n.succ 2 (λ n b, match n.succ with
                                                   | 1 => 3
                                                   | k+2 => b k.succ + b k
                                                   end)
  sorry

end sum_of_series_l150_150306


namespace snowfall_rate_in_Hamilton_l150_150512

theorem snowfall_rate_in_Hamilton 
  (initial_depth_Kingston : ℝ := 12.1)
  (rate_Kingston : ℝ := 2.6)
  (initial_depth_Hamilton : ℝ := 18.6)
  (duration : ℕ := 13)
  (final_depth_equal : initial_depth_Kingston + rate_Kingston * duration = initial_depth_Hamilton + duration * x)
  (x : ℝ) :
  x = 2.1 :=
sorry

end snowfall_rate_in_Hamilton_l150_150512


namespace sum_of_distinct_elements_l150_150049

theorem sum_of_distinct_elements (k : ℕ) :
    (∃ (m : ℕ) (S : Set ℕ), ∀ n > m, ∃* t ∈ Finset.powerset S, Finset.sum t id = n) ↔ ∃ (a : ℕ), k = 2^a := sorry

end sum_of_distinct_elements_l150_150049


namespace range_of_a_l150_150294

open Set

variable {α : Type*} [LinearOrder α]

def A : Set α := {x | 2 ≤ x ∧ x ≤ 6}
def B (a : α) : Set α := {x | 2 * a ≤ x ∧ x ≤ a + 3}

theorem range_of_a (a : α) (h : A ∪ B a = A) : 1 ≤ a :=
by
  sorry

end range_of_a_l150_150294


namespace max_sum_of_products_l150_150099

-- Define the set of numbers
def numbers : Set ℕ := {1, 3, 4, 6, 8, 9}

-- Define the labels for the cube faces
variables {a b c d e f : ℕ}

-- Condition that each number is assigned to a face of the cube
axiom faces_condition : a ∈ numbers ∧ b ∈ numbers ∧ c ∈ numbers ∧ d ∈ numbers ∧ e ∈ numbers ∧ f ∈ numbers ∧
                        a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
                        b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
                        c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
                        d ≠ e ∧ d ≠ f ∧
                        e ≠ f

-- Corresponding pairs defining opposite faces
def pairs := [(a, b), (c, d), (e, f)]

-- Summing over the pairs for products
def product_sum (p : List (ℕ × ℕ)) : ℕ :=
  p.foldl (λ acc ⟨x, y⟩, acc + (x + y)) 0

-- Main theorem to prove
theorem max_sum_of_products :
  ∃ a b c d e f ∈ numbers, 
    (∀ x y, (x, y) ∈ [(a, b), (c, d), (e, f)] -> x ≠ y) ∧ 
    product_sum pairs = 1100 :=
sorry

end max_sum_of_products_l150_150099


namespace polynomial_value_l150_150033

theorem polynomial_value 
  (x : ℝ) 
  (h1 : x = (1 + (1994 : ℝ).sqrt) / 2) : 
  (4 * x ^ 3 - 1997 * x - 1994) ^ 20001 = -1 := 
  sorry

end polynomial_value_l150_150033


namespace remaining_number_after_pairs_l150_150119

/-- 
We start with the set of numbers from 1 to 32. This set is repeatedly paired and replaced by the largest prime divisor of the sum of the pair.
We need to prove that the final remaining number is 11 after continuing this process until only one number remains.
-/
theorem remaining_number_after_pairs : 
  let nums := list.range 32 in 
  let num := 1 + 31 + 1 + sorry in -- Dummy calculation placeholder
  true
:= sorry

end remaining_number_after_pairs_l150_150119


namespace circle_radius_is_six_l150_150474

-- Given definitions
def radius_of_circle (r : ℝ) : Prop :=
  let x := π * r^2
  let y := 2 * π * r
  x + y = 72 * π

-- The proof problem
theorem circle_radius_is_six (r : ℝ) (h : radius_of_circle r) : r = 6 :=
by sorry

end circle_radius_is_six_l150_150474


namespace find_price_of_pastry_l150_150868

-- Define the known values and conditions
variable (P : ℕ)  -- Price of a pastry
variable (usual_pastries : ℕ := 20)
variable (usual_bread : ℕ := 10)
variable (bread_price : ℕ := 4)
variable (today_pastries : ℕ := 14)
variable (today_bread : ℕ := 25)
variable (price_difference : ℕ := 48)

-- Define the usual daily total and today's total
def usual_total := usual_pastries * P + usual_bread * bread_price
def today_total := today_pastries * P + today_bread * bread_price

-- Define the problem statement
theorem find_price_of_pastry (h: usual_total - today_total = price_difference) : P = 18 :=
  by sorry

end find_price_of_pastry_l150_150868


namespace boxes_of_apples_with_cherries_l150_150399

-- Define everything in the conditions
variable (A P Sp Sa : ℕ)
variable (box_cherries box_apples : ℕ)

-- Given conditions
axiom price_relation : 2 * P = 3 * A
axiom size_relation  : Sa = 12 * Sp
axiom cherries_per_box : box_cherries = 12

-- The problem statement (to be proved)
theorem boxes_of_apples_with_cherries : box_apples * A = box_cherries * P → box_apples = 18 :=
by
  sorry

end boxes_of_apples_with_cherries_l150_150399


namespace diagonals_length_and_t_value_l150_150695

def A : ℝ × ℝ := (-1, -2)
def B : ℝ × ℝ := (2, 3)
def C : ℝ × ℝ := (-2, -1)

def vec_minus (p q : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 - q.1, p.2 - q.2)

def vec_plus (p q : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + q.1, p.2 + q.2)

def vec_dot (p q : ℝ × ℝ) : ℝ :=
  p.1 * q.1 + p.2 * q.2

def magnitude (p : ℝ × ℝ) : ℝ :=
  Real.sqrt (p.1 ^ 2 + p.2 ^ 2)

theorem diagonals_length_and_t_value :
  let AB := vec_minus B A,
      AC := vec_minus C A,
      diag1 := vec_plus AB AC,
      diag2 := vec_minus AB AC,
      OC := C in
  magnitude diag1 = 2 * Real.sqrt 10 ∧
  magnitude diag2 = 4 * Real.sqrt 2 ∧
  ∃ t : ℝ, (vec_minus AB (t • OC)) ∙ OC = 0 ∧ t = -11/5 :=
sorry

end diagonals_length_and_t_value_l150_150695


namespace parameter_interval_solutions_l150_150454

noncomputable def distinct_solutions_interval (a : ℝ) : Prop :=
  ∃ t1 t2 : ℝ, 
    (0 < t1 ∧ t1 < π / 2 ∧ 0 < t2 ∧ t2 < π / 2 ∧ t1 ≠ t2 ∧ 
     (4 * a * (sin t1)^2 + 4 * a * (1 + 2 * (sqrt 2)) * (cos t1) - 4 * (a - 1) * (sin t1) - 5 * a + 2) / (2 * (sqrt 2) * (cos t1) - (sin t1)) = 4 * a ∧ 
     (4 * a * (sin t2)^2 + 4 * a * (1 + 2 * (sqrt 2)) * (cos t2) - 4 * (a - 1) * (sin t2) - 5 * a + 2) / (2 * (sqrt 2) * (cos t2) - (sin t2)) = 4 * a)

theorem parameter_interval_solutions :
  ∀ a : ℝ, (6 < a ∧ a < 18 + 24 * sqrt 2) ∨ (a > 18 + 24 * sqrt 2)
  → distinct_solutions_interval a :=
begin
  intro a,
  intro h,
  sorry
end

end parameter_interval_solutions_l150_150454


namespace S6_is_48_l150_150592

-- Define the first term and common difference
def a₁ : ℕ := 3
def d : ℕ := 2

-- Define the formula for sum of the first n terms of an arithmetic sequence
def sum_of_arithmetic_sequence (n : ℕ) : ℕ :=
  n / 2 * (2 * a₁ + (n - 1) * d)

-- Prove that the sum of the first 6 terms is 48
theorem S6_is_48 : sum_of_arithmetic_sequence 6 = 48 := by
  sorry

end S6_is_48_l150_150592


namespace first_pipe_fills_in_10_hours_l150_150825

def pipe_equation (x : ℝ) : Prop :=
  1/x + 1/12 - 1/20 = 1/7.5

theorem first_pipe_fills_in_10_hours : pipe_equation 10 :=
by
  -- Statement of the theorem
  sorry

end first_pipe_fills_in_10_hours_l150_150825


namespace inscribed_circle_area_l150_150347

theorem inscribed_circle_area 
  {A B C K : Type} 
  [DecidableEq A] [DecidableEq B] [DecidableEq C] [DecidableEq K]
  (angle_BCD_eq_angle_AKB : ∀ {b c d e k : Type} (A B C K : Type), 
  by trivial)
  (AK : ℝ) (BK : ℝ) (KC : ℝ) 
  (hAK : AK = 4) 
  (hBK : BK = 9) 
  (hKC : KC = 3)
  : (function (S : Type) 
    ((fun (pi : ℝ) => (35/13) * pi) = area_of_the_inscribed_circle_S :=
      sorry

end inscribed_circle_area_l150_150347


namespace pool_people_count_l150_150801

theorem pool_people_count (P : ℕ) (total_money : ℝ) (cost_per_person : ℝ) (leftover_money : ℝ) 
  (h1 : total_money = 30) 
  (h2 : cost_per_person = 2.50) 
  (h3 : leftover_money = 5) 
  (h4 : total_money - leftover_money = cost_per_person * P) : 
  P = 10 :=
sorry

end pool_people_count_l150_150801


namespace radius_of_circumscribed_sphere_l150_150164

theorem radius_of_circumscribed_sphere (r : ℝ) (A B C O : ℝ^3) (AB_dist : ℝ) 
  (dihedral_angle : ℝ) (circum_radius : ℝ) : 
  -- Given conditions
  r = 4 → 
  dist A B = 4 * real.sqrt 2 → 
  dist O A = r → 
  dist O B = r → 
  dist O C = r →
  -- Dihedral angle condition in radians
  dihedral_angle = real.pi / 3 →
  -- Conclusion
  circum_radius = 4 * real.sqrt 6 / 3 :=
by
  sorry

end radius_of_circumscribed_sphere_l150_150164


namespace joao_candle_problem_l150_150874

theorem joao_candle_problem (initial_candles : ℕ) (stubs_per_candle : ℕ) (new_candle_from_stubs : ℕ) : 
  initial_candles = 43 → stubs_per_candle = 1 → new_candle_from_stubs = 4 → 
  let rec count_nights (candles stubs : ℕ) : ℕ :=
    if candles = 0 then 0
    else 
      let new_candles := stubs / new_candle_from_stubs
      let remaining_stubs := stubs % new_candle_from_stubs
      1 + count_nights (candles - 1 + new_candles) (remaining_stubs + 1)
  in count_nights initial_candles 0 = 57 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end joao_candle_problem_l150_150874


namespace initial_men_count_l150_150786

-- Definitions based on problem conditions
def initial_days : ℝ := 18
def extra_men : ℝ := 400
def final_days : ℝ := 12.86

-- Proposition to show the initial number of men based on conditions
theorem initial_men_count (M : ℝ) (h : M * initial_days = (M + extra_men) * final_days) : M = 1000 := by
  sorry

end initial_men_count_l150_150786


namespace max_area_of_triangle_is_4sqrt3_l150_150583

noncomputable def max_area_of_triangle (a b : ℝ) : ℝ :=
  if a + b - 4 = 0 then
    (a + b - 4) * (a + b + 4) = 3 * a * b
    ∧ 4^2 = a^2 + b^2 - a * b := 16
    ∧ S_ΔABC = (1/2) * a * b * (sqrt 3) / 2
  in S_ΔABC

theorem max_area_of_triangle_is_4sqrt3 (a b : ℝ) (h : (a + b - 4) * (a + b + 4) = 3 * a * b) (h1 : 4 = 4) :
  max_area_of_triangle a b = 4 * sqrt 3 :=
by
  sorry

end max_area_of_triangle_is_4sqrt3_l150_150583


namespace find_KM_l150_150323

variables {K L M P Q G : Type}
variables {dist : K → K → ℝ}
variables [metric_space K] [metric_space L] [metric_space M]

-- Definitions for the lengths
def KP := 15
def LQ := 20

-- Conditions translating perpendicular medians and isosceles nature of the triangle
def perpendicular (a b c d : K) : Prop := 
  let m1 := (dist a b) in
  let m2 := (dist c d) in
  m1 * m2 = 0

-- Isosceles condition
def is_isosceles (a b c : K) : Prop := dist a b = dist a c

-- Problem statement
theorem find_KM (h1 : perpendicular K P L Q) (h2 : dist K P = KP) (h3 : dist L Q = LQ)
  (h4 : is_isosceles K L M) : dist K M = (20 * real.sqrt 13) / 3 :=
begin
  sorry
end

end find_KM_l150_150323


namespace isosceles_right_triangle_third_angle_l150_150262

/-- In an isosceles right triangle where one of the angles opposite the equal sides measures 45 degrees, 
    the measure of the third angle is 90 degrees. -/
theorem isosceles_right_triangle_third_angle (θ : ℝ) 
  (h1 : θ = 45)
  (h2 : ∀ (a b c : ℝ), a + b + c = 180) : θ + θ + 90 = 180 :=
by
  sorry

end isosceles_right_triangle_third_angle_l150_150262


namespace ladder_velocity_l150_150094

-- Definitions of variables and conditions
variables (a τ l : ℝ)
def v1 : ℝ := a * τ
def sin_alpha : ℝ := a * τ^2 / (2 * l)
def cos_alpha : ℝ := Real.sqrt (1 - (a * τ^2 / (2 * l))^2)

-- Main statement to prove
theorem ladder_velocity (h : v1 * sin_alpha = v2 * cos_alpha) : 
  v2 = (a^2 * τ^3) / (Real.sqrt (4 * l^2 - a^2 * τ^4)) := 
sorry

end ladder_velocity_l150_150094


namespace find_m_l150_150662

theorem find_m (m : ℝ) : (∫ x in 2..3, 3 * x^2 - 2 * m * x) = 34 → m = -3 :=
by
  sorry

end find_m_l150_150662


namespace mans_rate_correct_l150_150483

-- Defining the conditions
def speed_with_stream : ℝ := 12
def speed_against_stream : ℝ := 4

-- Defining the man's rate in still water (as the problem's question)
def mans_rate_in_still_water : ℝ := (speed_with_stream + speed_against_stream) / 2

-- Proof statement
theorem mans_rate_correct : mans_rate_in_still_water = 8 :=
by
  -- The proof itself is omitted
  sorry

end mans_rate_correct_l150_150483


namespace find_original_revenue_l150_150894

variable (currentRevenue : ℝ) (percentageDecrease : ℝ)
noncomputable def originalRevenue (currentRevenue : ℝ) (percentageDecrease : ℝ) : ℝ :=
  currentRevenue / (1 - percentageDecrease)

theorem find_original_revenue (h1 : currentRevenue = 48.0) (h2 : percentageDecrease = 0.3333333333333333) :
  originalRevenue currentRevenue percentageDecrease = 72.0 := by
  rw [h1, h2]
  unfold originalRevenue
  norm_num
  sorry

end find_original_revenue_l150_150894


namespace min_area_triangle_OAB_l150_150345

open Real

/-- Given points O, A, and B where A and B lie on the parabola y = x^2
and ∠AOB is a right angle, prove the minimum possible area of triangle AOB is 1. -/
theorem min_area_triangle_OAB : ∀ a b : ℝ,
  (a ≠ 0) ∧ (b ≠ 0) ∧ ((a * b) + (a^2 * b^2) = 0) →
  ∃ s : ℝ, (area_OAB a b = s) ∧ (s = 1) :=
by
  sorry

/-- Calculate the area of triangle OAB when the points are given. -/
noncomputable def area_OAB (a b : ℝ) : ℝ :=
  (1 / 2) * |a * (b^2) - b * (a^2)|

end min_area_triangle_OAB_l150_150345


namespace product_remainder_31_l150_150680

theorem product_remainder_31 (m n : ℕ) (h₁ : m % 31 = 7) (h₂ : n % 31 = 12) : (m * n) % 31 = 22 :=
by
  sorry

end product_remainder_31_l150_150680


namespace union_of_P_Q_l150_150317

variable {a b : ℝ}

def P : Set ℝ := {3, Real.log a / Real.log 2}
def Q : Set ℝ := {a, b}

theorem union_of_P_Q (hPQ : P ∩ Q = {0}) : P ∪ Q = {3, 0, 1} := by
  sorry

end union_of_P_Q_l150_150317


namespace problem_proof_l150_150634

noncomputable def f (x : ℝ) : ℝ :=
  if x < 4 then log 4 / log 2 
  else 1 + 2^(x - 1)

theorem problem_proof : f 0 + f (log 32 / log 2) = 19 :=
by
  -- Conditions derived from the problem
  have h1 : log 4 / log 2 = 2 := by sorry  -- Simplified log base 2 of 4
  have h2 : log 32 / log 2 = 5 := by sorry  -- Simplified log base 2 of 32
  rw [h1, h2],
  -- Thus evaluating f(0) and f(5)
  have h3 : f 0 = 2 := by sorry  -- f(0) because 0 < 4
  have h4 : f 5 = 17 := by sorry  -- f(5) because 5 >= 4 and 1 + 2^(5-1) = 17
  rw [h3, h4],
  exact rfl

end problem_proof_l150_150634


namespace log2_log2_16_l150_150109

theorem log2_log2_16 : Real.log 2 (Real.log 2 16) = 2 := by
  sorry

end log2_log2_16_l150_150109


namespace curvature_formula_l150_150351

noncomputable def curvature_squared (x y : ℝ → ℝ) (t : ℝ) :=
  let x' := (deriv x t)
  let y' := (deriv y t)
  let x'' := (deriv (deriv x) t)
  let y'' := (deriv (deriv y) t)
  (x'' * y' - y'' * x')^2 / (x'^2 + y'^2)^3

theorem curvature_formula (x y : ℝ → ℝ) (t : ℝ) :
  let k_sq := curvature_squared x y t
  k_sq = ((deriv (deriv x) t * deriv y t - deriv (deriv y) t * deriv x t)^2 /
         ((deriv x t)^2 + (deriv y t)^2)^3) := 
by 
  sorry

end curvature_formula_l150_150351


namespace f_has_four_distinct_real_roots_l150_150309

noncomputable def f (x d : ℝ) := x ^ 2 + 4 * x + d

theorem f_has_four_distinct_real_roots (d : ℝ) (h : d = 2) :
  ∃ r1 r2 r3 r4 : ℝ, r1 ≠ r2 ∧ r2 ≠ r3 ∧ r3 ≠ r4 ∧ r1 ≠ r3 ∧ r1 ≠ r4 ∧ r2 ≠ r4 ∧ 
  f (f r1 d) = 0 ∧ f (f r2 d) = 0 ∧ f (f r3 d) = 0 ∧ f (f r4 d) = 0 :=
by
  sorry

end f_has_four_distinct_real_roots_l150_150309


namespace emails_difference_l150_150709

def morning_emails : ℕ := 6
def afternoon_emails : ℕ := 8

theorem emails_difference :
  afternoon_emails - morning_emails = 2 := 
by
  sorry

end emails_difference_l150_150709


namespace unique_prime_arith_seq_with_diff_80_l150_150768

theorem unique_prime_arith_seq_with_diff_80 :
  ∃! (p1 p2 p3 : ℕ), 
    nat.prime p1 ∧ nat.prime p2 ∧ nat.prime p3 ∧ 
    p2 = p1 + 80 ∧ p3 = p1 + 160 :=
sorry

end unique_prime_arith_seq_with_diff_80_l150_150768


namespace storybooks_sciencebooks_correct_l150_150889

-- Given conditions
def total_books : ℕ := 144
def ratio_storybooks_sciencebooks := (7, 5)
def fraction_storybooks := 7 / (7 + 5)
def fraction_sciencebooks := 5 / (7 + 5)

-- Prove the number of storybooks and science books
def number_of_storybooks : ℕ := 84
def number_of_sciencebooks : ℕ := 60

theorem storybooks_sciencebooks_correct :
  (fraction_storybooks * total_books = number_of_storybooks) ∧
  (fraction_sciencebooks * total_books = number_of_sciencebooks) :=
by
  sorry

end storybooks_sciencebooks_correct_l150_150889


namespace number_of_solutions_decrease_l150_150818

-- Define the conditions and the main theorem
theorem number_of_solutions_decrease (a : ℝ) :
  (∃ x y : ℝ, x^2 - y^2 = 0 ∧ (x - a)^2 + y^2 = 1) → 
  (∀ x y : ℝ, x^2 - x^2 = 0 ∧ (x - a)^2 + x^2 = 1) →
  a = 1 ∨ a = -1 := 
sorry

end number_of_solutions_decrease_l150_150818


namespace inequality_proof_l150_150783

theorem inequality_proof (x y : ℝ) (hx : x > -1) (hy : y > -1) (hxy : x + y = 1) : 
    (x / (y + 1) + y / (x + 1) ≥ 2 / 3) := 
  sorry

end inequality_proof_l150_150783


namespace HUN_3_l150_150456

theorem HUN_3 (points : Finset (ℝ × ℝ)) (h1 : points.card = 4000) 
    (h2 : ∀ l : Line, 2 < (points.filter (λ p, l.contains p)).card) :
    ∃ quads : Finset (Finset (ℝ × ℝ)), quads.card = 1000 ∧ 
    ∀ quad ∈ quads, quad.card = 4 ∧ QuadIsDisjoint (⋃₀ quads) :=
by sorry

-- Auxiliary definition: Line
structure Line :=
  (slope : ℝ)
  (intercept : ℝ)

-- Auxiliary function: Line contains a point
def Line.contains (l : Line) (p : ℝ × ℝ) : Prop :=
  p.2 = l.slope * p.1 + l.intercept

-- Auxiliary predicate: Check if a collection of quadrilaterals is disjoint
def QuadIsDisjoint (quads : Finset (ℝ × ℝ)) : Prop :=
  ∀ p1 p2 : ℝ × ℝ, p1 ∈ quads → p2 ∈ quads → p1 ≠ p2 → p1 ≠ p2

end HUN_3_l150_150456


namespace intersection_A_B_l150_150203

def A : Set ℝ := { x | 1/2 ≤ 2^x ∧ 2^x < 16 }
def B : Set ℝ := { x | 9 - x^2 > 0 }

theorem intersection_A_B : A ∩ B = { x | -1 ≤ x ∧ x < 3 } :=
  by sorry

end intersection_A_B_l150_150203


namespace range_of_y_l150_150234

theorem range_of_y :
  ∀ (y : ℝ), y < 0 → (⌈y⌉ : ℝ) * (⌊y⌋ : ℝ) = 132 → y ∈ set.Ioo (-12 : ℝ) (-11) :=
by
  intros y hy h_eq
  -- The proof steps would go here
  sorry

end range_of_y_l150_150234


namespace find_y_given_conditions_l150_150233

theorem find_y_given_conditions (x y : ℝ) (hx : x = 102) 
                                (h : x^3 * y - 3 * x^2 * y + 3 * x * y = 106200) : 
  y = 10 / 97 :=
by
  sorry

end find_y_given_conditions_l150_150233


namespace count_integers_l150_150738

def Q (x : ℝ) : ℝ := (x - 1) * (x - 4) * (x - 9) * (x - 16) * (x - 25) * (x - 36) * (x - 49) * (x - 64) * (x - 81)

theorem count_integers (Q_le_0 : ∀ n : ℤ, Q n ≤ 0 → ∃ k : ℕ, k = 53) : ∃ k : ℕ, k = 53 := by
  sorry

end count_integers_l150_150738


namespace remaining_volume_l150_150859

-- Given
variables (a d : ℚ) 
-- Define the volumes of sections as arithmetic sequence terms
def volume (n : ℕ) := a + n*d

-- Define total volume of bottom three sections
def bottomThreeVolume := volume a 0 + volume a d + volume a (2 * d) = 4

-- Define total volume of top four sections
def topFourVolume := volume a (5 * d) + volume a (6 * d) + volume a (7 * d) + volume a (8 * d) = 3

-- Define the volumes of the two middle sections
def middleTwoVolume := volume a (3 * d) + volume a (4 * d) = 2 + 3 / 22

-- Prove that the total volume of the remaining two sections is 2 3/22
theorem remaining_volume : bottomThreeVolume a d ∧ topFourVolume a d → middleTwoVolume a d :=
sorry  -- Placeholder for the actual proof

end remaining_volume_l150_150859


namespace geometric_series_sum_proof_l150_150027

theorem geometric_series_sum_proof :
  ∑ k in Finset.range 12, (4: ℚ) ^ (-k) * 3 ^ k = 48750225 / 16777216 :=
by sorry

end geometric_series_sum_proof_l150_150027


namespace indeterminate_number_of_men_l150_150671

theorem indeterminate_number_of_men (n : ℕ) (h1 : ∀ n, (job_completion_days n 15) = 15) : 
  ∃ m : ℕ, m = n → indeterminate n :=
by
  sorry

end indeterminate_number_of_men_l150_150671


namespace incorrect_connection_probability_l150_150416

noncomputable def probability_of_incorrect_connection (p : ℝ) : ℝ :=
  let r2 := 1 / 9
  let r3 := (8 / 9) * (1 / 9)
  (3 * p^2 * (1 - p) * r2) + (1 * p^3 * r3)

theorem incorrect_connection_probability : probability_of_incorrect_connection 0.02 = 0.000131 :=
by
  sorry

end incorrect_connection_probability_l150_150416


namespace f_evaluated_l150_150192

-- Define the piecewise function f
noncomputable def f : ℝ → ℝ
| x := if x ≥ 3 then (1 / 2) ^ x else f (x + 1)

-- Define the logarithmic value of interest
def x_val := 1 + Real.log 3 / Real.log 2

-- The theorem to prove the equivalence
theorem f_evaluated : f x_val = 1 / 12 :=
sorry

end f_evaluated_l150_150192


namespace area_of_region_l150_150546

theorem area_of_region :
  let circle_radius := real.sqrt (15 / (2 * real.pi)) in
  let circle_eq := (λ x y : ℝ, 2 * real.pi * (x^2 + y^2) ≤ 15) in
  let inequality1 := (λ x y : ℝ, 2 * real.pi * (x^2 + y^2) ≤ 15) in
  let inequality2 := (λ x y : ℝ, x^4 - y^4 ≤ x * y - x^3 * y^3) in
  ∀ (condition1 : ∀ x y : ℝ, inequality1 x y),
  ∀ (condition2 : ∀ x y : ℝ, inequality2 x y),
  let total_area := (real.pi * (circle_radius ^ 2)) / 2 in
  total_area = 3.75 :=
sorry

end area_of_region_l150_150546


namespace parallelepiped_lateral_surface_area_and_volume_l150_150950

-- Define the conditions
variables {h : ℝ} {α β : ℝ}
-- Assume the base is a rhombus and the height is h
-- Define the conditions explicitly
def right_parallelepiped (h : ℝ) (α β : ℝ) :=
  ∃ (height : ℝ) (angle_α angle_beta : ℝ), height = h ∧ angle_α = α ∧ angle_beta = β

-- Define the theorem for lateral surface area and volume
theorem parallelepiped_lateral_surface_area_and_volume :
  ∀ h α β : ℝ,
  let lateral_surface_area := 2 * h^2 * (real.sqrt (real.cot α ^ 2 + real.cot β ^ 2)) in
  let volume := (1 / 2) * h^3 * (real.cot α) * (real.cot β) in
  right_parallelepiped h α β →
  (∃ S V, S = lateral_surface_area ∧ V = volume) :=
by
  unfold right_parallelepiped
  assume h α β lateral_surface_area volume hp
  use [lateral_surface_area, volume]
  split
  all_goals
    sorry

end parallelepiped_lateral_surface_area_and_volume_l150_150950


namespace exists_eps_sum_norm_le_sqrt_three_l150_150724

open_locale big_operators

variables {ι : Type} [fintype ι] {v : ι → ℝ × ℝ} (n : ℕ)

def vector_norm (v : ℝ × ℝ) : ℝ := real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem exists_eps_sum_norm_le_sqrt_three (h : (∀ i, vector_norm (v i) ≤ 1)) :
  ∃ (ε : ι → ℤ), (∀ i, ε i = 1 ∨ ε i = -1) ∧ 
    vector_norm (finset.univ.sum (λ i, (ε i : ℝ) • v i)) ≤ real.sqrt 3 :=
sorry

end exists_eps_sum_norm_le_sqrt_three_l150_150724


namespace number_of_words_with_Z_l150_150802

-- Define the alphabet size
def alphabet_size := 26

-- Define the maximum word length
def max_length := 5

-- Define the number of words that must contain the letter Z at least once
def words_with_Z : ℕ :=
  let all_words (n : ℕ) := alphabet_size^n in
  let words_without_Z (n : ℕ) := (alphabet_size - 1)^n in
  (all_words 1 - words_without_Z 1) +
  (all_words 2 - words_without_Z 2) +
  (all_words 3 - words_without_Z 3) +
  (all_words 4 - words_without_Z 4) +
  (all_words 5 - words_without_Z 5)

theorem number_of_words_with_Z : words_with_Z = 2205115 := by
  -- Placeholder for the proof
  sorry

end number_of_words_with_Z_l150_150802


namespace digit_B_divisible_by_9_l150_150385

theorem digit_B_divisible_by_9 (B : ℕ) (k : ℤ) (h1 : 0 ≤ B) (h2 : B ≤ 9) (h3 : 2 * B + 6 = 9 * k) : B = 6 := 
by
  sorry

end digit_B_divisible_by_9_l150_150385


namespace bounded_area_l150_150548

theorem bounded_area :
  let R : ℝ := (15 / (2 * π))^(1/2),
      is_inside_circle (x y : ℝ) : Prop := x^2 + y^2 ≤ 15 / (2 * π),
      satisfies_second_ineq (x y : ℝ) : Prop := (x^3 - y) * (x + y^3) ≤ 0 in
  (∃ (R : ℝ), (R = (15 / (2 * π))^(1/2)) ∧
  ∀ (x y : ℝ), is_inside_circle x y ∧ satisfies_second_ineq x y → 
  (area_of_region (interior_of_circle R) ∩ (region_defined_by_second_ineq) = 3.75)
  :=
begin
  sorry
end

end bounded_area_l150_150548


namespace square_construction_condition_l150_150974

theorem square_construction_condition (a b : ℝ) (h1 : a > 0) (h2 : b > 0):
  (∃ S : ℝ, S = a ∧ ∃ T : ℝ, T = b ∧ (
    ∀ x1 y1 x2 y2: ℝ, 
    (x1, y1) ≠ (0, 0) -> x1² + y1² = b² -> 
    ∃ x y: ℝ, 
    (x, x1) = (y, y1)
  )) -> (a / 2 * real.sqrt 2 < b ∧ b ≤ a * real.sqrt 2) :=
by
  intros
  sorry

end square_construction_condition_l150_150974


namespace total_people_in_community_l150_150475

theorem total_people_in_community (X : ℝ) : 
  let seniors := 3.5 * X in
  let children := 4 * seniors in
  let teenagers := 2.5 * children in
  let women := 3 * teenagers in
  let men := 1.5 * women in
  let total := men + women + teenagers + children + seniors + X in
  total = 316 * X :=
by
  sorry

end total_people_in_community_l150_150475


namespace sin_x_one_of_sec_sub_tan_l150_150175

theorem sin_x_one_of_sec_sub_tan (x : ℝ) (h : sec x - tan x = 5 / 3) : sin x = 1 :=
sorry

end sin_x_one_of_sec_sub_tan_l150_150175


namespace jack_pays_back_total_l150_150275

noncomputable def principal : ℝ := 1200
noncomputable def rate : ℝ := 0.10
noncomputable def interest : ℝ := principal * rate
noncomputable def total : ℝ := principal + interest

theorem jack_pays_back_total : total = 1320 := by
  sorry

end jack_pays_back_total_l150_150275


namespace angle_AOC_correct_l150_150296

noncomputable def angle_AOC (OA OB OC : Ω → ℝ^3) : ℝ :=
  let α := (OA + 2 * OC) ∙ (OA + 2 * OC)
  let β := (-√3) * (OB ∙ OB)
  let r := √((α - β) / 4)
  if r ≠ 0 then
    let cosAngle := (-1) / 2
    if cosAngle = cos (π / 3) then (2 * π) / 3 else 0
  else
    0

theorem angle_AOC_correct {Ω : Type*}
  (r : ℝ) (h1 : ∃ (OA OB OC : Ω → ℝ^3), OA + sqrt 3 * OB + 2 * OC = 0)
  : angle_AOC = (2 * π) / 3 :=
begin
  sorry
end

end angle_AOC_correct_l150_150296


namespace constant_term_binomial_expansion_l150_150681

theorem constant_term_binomial_expansion (n : ℕ) (h1 : (∑ i in Finset.range (n+1), Nat.choose n i) = 512)
  (h2 : n = 9) : 
  ∃ k, (Nat.choose 9 k) * ((-1) ^ k) = 84 ∧ 2*(9-k) - k = 0 := 
by
  sorry

end constant_term_binomial_expansion_l150_150681


namespace soccer_field_kids_l150_150053

def a := 14
def b := 22
def c := a + b

theorem soccer_field_kids : c = 36 :=
by
    sorry

end soccer_field_kids_l150_150053


namespace sequence_arithmetic_and_geometric_l150_150200

/-- A sequence of numbers is defined such that it could potentially be both arithmetic and geometric. -/
def sequence : ℕ → ℝ
| 0 => 3
| 1 => 9
| n => 729  -- In practice, the rest of the sequence is unknown

/-- To prove that the sequence 3, 9, ..., 729 can be both arithmetic and geometric. -/
theorem sequence_arithmetic_and_geometric :
  (∃ d : ℝ, ∀ n : ℕ, n > 0 → sequence (n + 1) - sequence n = d)
    ∧ (∃ r : ℝ, ∀ n : ℕ, n > 0 → sequence (n + 1) / sequence n = r) :=
sorry

end sequence_arithmetic_and_geometric_l150_150200


namespace slope_intercept_form_of_line_l150_150162

theorem slope_intercept_form_of_line :
  ∀ (x y : ℝ), (∀ (a b : ℝ), (a, b) = (0, 4) ∨ (a, b) = (3, 0) → y = - (4 / 3) * x + 4) := 
by
  sorry

end slope_intercept_form_of_line_l150_150162


namespace work_rate_problem_l150_150040

theorem work_rate_problem 
  (W : ℝ)
  (rate_ab : ℝ)
  (rate_c : ℝ)
  (rate_abc : ℝ)
  (cond1 : rate_c = W / 2)
  (cond2 : rate_abc = W / 1)
  (cond3 : rate_ab = (W / 1) - rate_c) :
  rate_ab = W / 2 :=
by 
  -- We can add the solution steps here, but we skip that part following the guidelines
  sorry

end work_rate_problem_l150_150040


namespace part1_part2_l150_150644

noncomputable def A (a : ℝ) : set ℝ := {x | (x - a) * (x - 3 * a) < 0 ∧ a > 0}
noncomputable def B : set ℝ := {x | ∃ t, x = 2^(t-2) ∧ 2 < t ∧ t < 3}
def a : ℝ := 1

theorem part1 : A a ∩ (set.compl B) = { x | 2 ≤ x ∧ x < 3 } := sorry

theorem part2 (a : ℝ) : (A a = A a) ∧ (B = B) ∧ (a > 0 ∧ ∀ x, x ∈ B → x ∈ A a) → (2 / 3 ≤ a ∧ a ≤ 1) := sorry

end part1_part2_l150_150644


namespace train_pass_time_l150_150043

noncomputable def train_length : ℕ := 360
noncomputable def platform_length : ℕ := 140
noncomputable def train_speed_kmh : ℕ := 45

noncomputable def convert_speed_to_mps (speed_kmh : ℕ) : ℚ := 
  (speed_kmh * 1000) / 3600

noncomputable def total_distance (train_len platform_len : ℕ) : ℕ :=
  train_len + platform_len

noncomputable def time_to_pass (distance : ℕ) (speed_mps : ℚ) : ℚ :=
  distance / speed_mps

theorem train_pass_time 
  (train_len : ℕ) 
  (platform_len : ℕ) 
  (speed_kmh : ℕ) : 
  time_to_pass (total_distance train_len platform_len) (convert_speed_to_mps speed_kmh) = 40 := 
by 
  sorry

end train_pass_time_l150_150043


namespace right_triangle_set_C_l150_150506

theorem right_triangle_set_C : 
  ∀ (a b c : ℕ), a = 3 ∧ b = 4 ∧ c = 5 → a^2 + b^2 = c^2 :=
by
  intros a b c h,
  rcases h with ⟨ha, hb, hc⟩,
  rw [ha, hb, hc],
  calc
    3^2 + 4^2 = 9 + 16 : by norm_num
          ... = 25     : by norm_num
          ... = 5^2    : by norm_num

end right_triangle_set_C_l150_150506


namespace find_a_plus_b_l150_150194

noncomputable def is_even_function (f : ℝ → ℝ) :=
∀ x : ℝ, f x = f (-x)

theorem find_a_plus_b :
  ∃ (a b : ℝ), (∀ x ∈ set.Icc (a^2 - 2) a, f x = ax^2 + (b - 3) * x + 3) ∧
    is_even_function (λ x, ax^2 + (b-3)*x + 3) ∧
    a + b = 4 :=
begin
  sorry
end

end find_a_plus_b_l150_150194


namespace problem1_problem2_l150_150520

theorem problem1 : sqrt 16 + cbrt (-64) - sqrt ((-3 : ℝ)^2) + abs (sqrt 3 - 1) = -4 + sqrt 3 :=
by sorry

theorem problem2 : ∀ x : ℝ, (x + 1)^2 = 16 ↔ x = 3 ∨ x = -5 :=
by sorry

end problem1_problem2_l150_150520


namespace number_of_integer_areas_l150_150564

def floor_sqrt (x : ℝ) : ℤ := int.floor (real.sqrt x)

-- Define the region and area function A(n).
noncomputable def A (n : ℤ) : ℝ :=
  if h : 2 ≤ n then ∑ k in finset.range (n.to_nat + 1), 
    (1/2) * (floor_sqrt k * (2 * k - 1)) 
  else 0

-- The final statement to prove.
theorem number_of_integer_areas : 
  (finset.filter (λ n, A n ∈ int) ({x : ℤ | 2 ≤ x ∧ x ≤ 1000}).to_finset).card = 483 :=
sorry

end number_of_integer_areas_l150_150564


namespace germs_per_dish_l150_150849

theorem germs_per_dish (total_germs number_of_dishes : ℝ) (h1 : total_germs = 0.037 * 10^5) (h2 : number_of_dishes = 148000 * 10^(-3)) : 
  (total_germs / number_of_dishes) = 25 :=
by
  sorry

end germs_per_dish_l150_150849


namespace probability_three_or_more_same_after_rerolling_l150_150567

-- Noncomputable to handle probability calculations which are not purely constructive
noncomputable def probability_at_least_three_identical (pair_value non_pair_value_1 non_pair_value_2 : ℕ) 
  (h1 : 1 ≤ pair_value ∧ pair_value ≤ 6)
  (h2 : 1 ≤ non_pair_value_1 ∧ non_pair_value_1 ≤ 6)
  (h3 : 1 ≤ non_pair_value_2 ∧ non_pair_value_2 ≤ 6)
  (h4 : non_pair_value_1 ≠ non_pair_value_2)
  (h5 : pair_value ≠ non_pair_value_1)
  (h6 : pair_value ≠ non_pair_value_2) : ℚ :=
let outcomes := 36 in
let successful_outcomes :=
  -- Exact one match the pair and one does not
  2 * (1 / 6) * (5 / 6) +
  -- Both match the pair
  (1 / 6) * (1 / 6) +
  -- Both match each other but different from the pair
  5 * (1 / 6) * (1 / 6)
in successful_outcomes / outcomes

theorem probability_three_or_more_same_after_rerolling : 
  ∀ (pair_value non_pair_value_1 non_pair_value_2 : ℕ), 
    (1 ≤ pair_value) ∧ (pair_value ≤ 6) ∧ 
    (1 ≤ non_pair_value_1) ∧ (non_pair_value_1 ≤ 6) ∧ 
    (1 ≤ non_pair_value_2) ∧ (non_pair_value_2 ≤ 6) ∧ 
    (non_pair_value_1 ≠ non_pair_value_2) ∧ 
    (pair_value ≠ non_pair_value_1) ∧ 
    (pair_value ≠ non_pair_value_2) →
    probability_at_least_three_identical pair_value non_pair_value_1 non_pair_value_2 
    sorry = 4 / 9 :=
sorry

end probability_three_or_more_same_after_rerolling_l150_150567


namespace number_above_345_l150_150261

-- Define the function that calculates the number of elements up to row k
def num_elements_up_to_row : ℕ → ℕ
| 0       => 0
| (k + 1) => num_elements_up_to_row k + (3 * (k + 1) - 2)

-- Define the function that finds k such that total elements up to row k is at least n
def find_row (n : ℕ) : ℕ :=
  Nat.find (λ k, num_elements_up_to_row k ≥ n)

-- Define the function to get the starting number of a row
def row_start (k : ℕ) : ℕ :=
  if k = 0 then 1 else num_elements_up_to_row (k - 1) + 1

-- Define the function to get the relative position in a row
def position_in_row (n : ℕ) (k : ℕ) : ℕ :=
  n - row_start k + 1

-- Define the function to find the number directly above n
def number_directly_above (n : ℕ) : ℕ :=
  let k := find_row n
  let pos := position_in_row n k
  let above_pos := pos
  let above_k := k - 1
  row_start above_k + above_pos - 1

theorem number_above_345 : number_directly_above 345 = 308 := by
  sorry

end number_above_345_l150_150261


namespace wire_pieces_for_10x10x10_cube_l150_150858

theorem wire_pieces_for_10x10x10_cube :
  let n := 10 in
  let edges_per_dimension := (n+1)^2 * n in
  let total_edges := 3 * edges_per_dimension in
  total_edges = 3630 := by
  sorry

end wire_pieces_for_10x10x10_cube_l150_150858


namespace wedge_top_half_volume_l150_150086

theorem wedge_top_half_volume (r : ℝ) (C : ℝ) (V : ℝ) : 
  (C = 18 * π) ∧ (C = 2 * π * r) ∧ (V = (4/3) * π * r^3) ∧ 
  (V / 3 / 2) = 162 * π :=
  sorry

end wedge_top_half_volume_l150_150086


namespace lines_concurrent_l150_150295

variables {n : ℕ} (E : Finset (EuclideanSpace ℝ 2))
variables (Γ : Circle (EuclideanSpace ℝ 2))
variables (points_on_circle : ∀ (M : EuclideanSpace ℝ 2), M ∈ E → M ∈ Γ)

noncomputable def centroid (points : Finset (EuclideanSpace ℝ 2)) : EuclideanSpace ℝ 2 :=
(points.1.sum • (↑(Finset.card points)⁻¹))

variables {G : Fin (n+1) → EuclideanSpace ℝ 2}
variables (∆ : Fin (n+1) → EuclideanSpace ℝ 2 → EuclideanSpace ℝ 2)

-- Conditions of the problem
variables (h_Gi : ∀ i, G i = centroid (E.erase (Fin.val i)))
variables (h_Diai : ∀ i, ∆ i = λ x, ⟨⟨ G i ⟩, is_perpendicular_to_tangent (Γ) (E.nth i)⟩)

theorem lines_concurrent : 
  ∃ (O : EuclideanSpace ℝ 2), ∀ i, ∆ i (G i) = O :=
sorry

end lines_concurrent_l150_150295


namespace monotonicity_intervals_number_of_zeros_l150_150749

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (x - 1) * Real.exp x - k / 2 * x^2

theorem monotonicity_intervals (k : ℝ) :
  (k ≤ 0 → (∀ x, x < 0 → f k x < 0) ∧ (∀ x, x ≥ 0 → f k x > 0)) ∧
  (0 < k ∧ k < 1 → 
    (∀ x, x < Real.log k → f k x < 0) ∧ (∀ x, x ≥ Real.log k ∧ x < 0 → f k x > 0) ∧ 
    (∀ x, x > 0 → f k x > 0)) ∧
  (k = 1 → ∀ x, f k x > 0) ∧
  (k > 1 → 
    (∀ x, x < 0 → f k x < 0) ∧ 
    (∀ x, x ≥ 0 ∧ x < Real.log k → f k x > 0) ∧ 
    (∀ x, x > Real.log k → f k x > 0)) :=
sorry

theorem number_of_zeros (k : ℝ) (h_nonpos : k ≤ 0) :
  (k < 0 → (∃ a b : ℝ, a < 0 ∧ b > 0 ∧ f k a = 0 ∧ f k b = 0)) ∧
  (k = 0 → f k 1 = 0 ∧ (∀ x, x ≠ 1 → f k x ≠ 0)) :=
sorry

end monotonicity_intervals_number_of_zeros_l150_150749


namespace count_of_divisibles_l150_150209

theorem count_of_divisibles (n : ℕ) (h : n < 500) : 
  (∃ k, n = 2 * 3 * 5 * 7 * k ∧ k > 0) →
  (∃! k, k = 2) :=
by {
  have LCM := 210,
  -- sorry is used to skip the proof
  sorry
}

end count_of_divisibles_l150_150209


namespace savings_are_equal_and_correct_l150_150816

-- Definitions of the given conditions
variables (I1 I2 E1 E2 : ℝ)
variables (S1 S2 : ℝ)
variables (rI : ℝ := 5/4) -- ratio of incomes
variables (rE : ℝ := 3/2) -- ratio of expenditures
variables (I1_val : ℝ := 3000) -- P1's income

-- Given conditions
def given_conditions : Prop :=
  I1 = I1_val ∧
  I1 / I2 = rI ∧
  E1 / E2 = rE ∧
  S1 = S2

-- Required proof
theorem savings_are_equal_and_correct (I2_val : I2 = (I1_val * 4/5)) (x : ℝ) (h1 : E1 = 3 * x) (h2 : E2 = 2 * x) (h3 : S1 = 1200) :
  S1 = S2 ∧ S1 = 1200 := by
  sorry

end savings_are_equal_and_correct_l150_150816


namespace decagon_diagonal_intersections_l150_150258

theorem decagon_diagonal_intersections:
  let n := 10,
  let diagonals := n * (n - 3) / 2,
  let intersections := Nat.choose n 4 in
  intersections = 210 := by
  sorry

end decagon_diagonal_intersections_l150_150258


namespace find_three_digit_number_l150_150672

theorem find_three_digit_number :
  ∃ x : ℕ, 100 ≤ x ∧ x ≤ 999 ∧
    (x - 6) % 7 = 0 ∧
    (x - 7) % 8 = 0 ∧
    (x - 8) % 9 = 0 ∧
    x = 503 :=
by
  sorry

end find_three_digit_number_l150_150672


namespace num_divisible_by_2_3_5_7_under_500_l150_150222

-- Let the LCM of 2, 3, 5, and 7 be computed
def LCM_2_3_5_7 : ℕ := Nat.lcm (Nat.lcm 2 3) (Nat.lcm 5 7)

def num_multiples_less_than (n k : ℕ) : ℕ :=
  n / k

theorem num_divisible_by_2_3_5_7_under_500 : num_multiples_less_than 500 LCM_2_3_5_7 = 2 := by
  sorry

end num_divisible_by_2_3_5_7_under_500_l150_150222


namespace range_of_set_is_8_l150_150082

theorem range_of_set_is_8 (a b c : ℕ) 
  (h1 : (a + b + c) / 3 = 6) 
  (h2 : b = 6) 
  (h3 : a = 2) 
  : max a (max b c) - min a (min b c) = 8 := 
by sorry

end range_of_set_is_8_l150_150082


namespace kiyiv_first_problem_kiyiv_second_problem_l150_150781

/-- Let x and y be positive real numbers such that xy ≥ 1.
Prove that x^3 + y^3 + 4xy ≥ x^2 + y^2 + x + y + 2. -/
theorem kiyiv_first_problem (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : 1 ≤ x * y) :
  x^3 + y^3 + 4 * x * y ≥ x^2 + y^2 + x + y + 2 :=
sorry

/-- Let x and y be positive real numbers such that xy ≥ 1.
Prove that 2(x^3 + y^3 + xy + x + y) ≥ 5(x^2 + y^2). -/
theorem kiyiv_second_problem (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : 1 ≤ x * y) :
  2 * (x^3 + y^3 + x * y + x + y) ≥ 5 * (x^2 + y^2) :=
sorry

end kiyiv_first_problem_kiyiv_second_problem_l150_150781


namespace problem_1_problem_2_problem_3_l150_150700

noncomputable theory

-- Definitions and conditions for the sequence
def a_seq (n : ℕ) (a : ℝ) (k : ℝ) : ℕ → ℝ
| 0 := 1
| 1 := a
| (n+2) := k * (a_seq n + a_seq (n + 1))

-- Sum of the first n terms of the sequence
def S_n (n : ℕ) (a : ℝ) (k : ℝ) := ∑ i in range (n + 1), a_seq i a k

-- Problem 1: Sum of the first n terms if the sequence is arithmetic
theorem problem_1 (a : ℝ) (n : ℕ) : 
  (∀ n, a_seq n a (1/2) = 1 + n * (a - 1)) → 
  S_n n a (1/2) = ((a - 1) * n^2 - (a - 3) * n) / 2 :=
sorry

-- Problem 2: Special case when a = 1 and k = -1/2
theorem problem_2 (n : ℕ) :
  (∀ n, a_seq n 1 (-1/2) = if even n then 1 else -1) → 
  S_n n 1 (-1/2) = if even n then n else 2 - n :=
sorry

-- Problem 3: Existence of a specific k making the sequence geometric and satisfy an arithmetic arrangement condition
theorem problem_3 (a : ℝ) :
  (∃ k, (∀ n, a_seq n a k = a^n) ∧ ∀ m, ∃ p q r, 
  {a_seq m a k, a_seq (m+1) a k, a_seq (m+2) a k} = {p, q, r} ∧ 
  2*q = p + r) ↔ k = -2/5 :=
sorry

end problem_1_problem_2_problem_3_l150_150700


namespace num_correct_statements_l150_150633

noncomputable def are_mutually_exclusive (A B : Set) : Prop :=
  A ∩ B = ∅

noncomputable def are_complementary (A B : Set) : Prop :=
  A ∪ B = univ ∧ A ∩ B = ∅

theorem num_correct_statements :
  (¬ ∀ A B : Set, are_mutually_exclusive A B → are_complementary A B) ∧
  (∀ A B : Set, are_complementary A B → are_mutually_exclusive A B) ∧
  (∃ A B : Set, are_mutually_exclusive A B ∧ ¬ are_complementary A B) ∧
  ¬ ∀ A B : Set, are_mutually_exclusive A B → P(A) = 1 - P(B) :=
  sorry

end num_correct_statements_l150_150633


namespace part_I_part_II_l150_150745

-- Translate the conditions and questions to Lean definition statements.

-- First part of the problem: proving the value of a
theorem part_I (a : ℝ) (f : ℝ → ℝ) (Hf : ∀ x, f x = |a * x - 1|) 
(Hsol : ∀ x, f x ≤ 2 ↔ -6 ≤ x ∧ x ≤ 2) : a = -1 / 2 :=
sorry

-- Second part of the problem: proving the range of m
theorem part_II (m : ℝ) 
(H : ∃ x : ℝ, |4 * x + 1| - |2 * x - 3| ≤ 7 - 3 * m) : m ≤ 7 / 2 :=
sorry

end part_I_part_II_l150_150745


namespace Lizzie_group_area_l150_150098

theorem Lizzie_group_area (total_area area_other_group area_remaining : ℕ) 
  (h1 : total_area = 900) 
  (h2 : area_other_group = 265) 
  (h3 : area_remaining = 385) : 
  total_area - area_other_group - area_remaining = 250 :=
by
  rw [h1, h2, h3]
  norm_num
  sorry

end Lizzie_group_area_l150_150098


namespace sides_of_polygon_count_l150_150116

theorem sides_of_polygon_count (K : Type) (vertices : Fin 20 → K) :
  (∑ i in Finset.range 20, ∑ j in Finset.Icc (i + 3) (i + 17) ∩ Finset.range 20, 
   ∑ k in Finset.Icc (j + 3) (j + 17) ∩ Finset.range 20, 1) = 520 := 
begin
  sorry
end

end sides_of_polygon_count_l150_150116


namespace triangle_area_l150_150792

-- Definitions based on conditions
def is_right_triangle (A B C : Type) (angleBAC : ℝ) (angleABC : ℝ) (angleACB : ℝ) :=
  angleBAC = 45 ∧ angleABC = 45 ∧ angleACB = 90

def altitude_to_hypotenuse (alt : ℝ) :=
  alt = 5

-- The theorem to prove
theorem triangle_area (A B C : Type) (angleBAC angleABC angleACB : ℝ) (alt : ℝ)
  (h_right_triangle : is_right_triangle A B C angleBAC angleABC angleACB)
  (h_altitude : altitude_to_hypotenuse alt) :
  (area : ℝ) := 25 * Real.sqrt 2 :=
by
  sorry

end triangle_area_l150_150792


namespace instantaneous_velocity_at_t3_l150_150676

noncomputable def displacement (t : ℝ) : ℝ := 2 * t^3

noncomputable def velocity (t : ℝ) : ℝ := derivative displacement t

theorem instantaneous_velocity_at_t3 : velocity 3 = 54 :=
by 
  -- Define s as the displacement
  have s : (t : ℝ) → ℝ := λ t, 2 * t^3
  have v : (t : ℝ) → ℝ := λ t, derivative s t
  change v 3 = 54
  rw [←velocity]
  rw [←displacement]
  sorry  -- This is where the actual proof steps would be filled in.

end instantaneous_velocity_at_t3_l150_150676


namespace percentage_increase_l150_150722

theorem percentage_increase (old_earnings new_earnings : ℝ) (h_old : old_earnings = 50) (h_new : new_earnings = 70) :
  ((new_earnings - old_earnings) / old_earnings) * 100 = 40 :=
by
  rw [h_old, h_new]
  -- Simplification and calculation steps would go here
  sorry

end percentage_increase_l150_150722


namespace special_topics_inequalities_part1_special_topics_inequalities_part2_l150_150538

theorem special_topics_inequalities_part1 (a b : ℝ) 
  (h1 : a ∈ (- (1 : ℝ)/2, 1/2)) 
  (h2 : b ∈ (- (1 : ℝ)/2, 1/2)) : 
  |(1/3 : ℝ) * a + (1/6 : ℝ) * b| < 1/4 :=
  sorry

theorem special_topics_inequalities_part2 (a b : ℝ) 
  (h1 : a ∈ (- (1 : ℝ)/2, 1/2)) 
  (h2 : b ∈ (- (1 : ℝ)/2, 1/2)) : 
  |1 - 4 * a * b| > 2 * |a - b| :=
  sorry

end special_topics_inequalities_part1_special_topics_inequalities_part2_l150_150538


namespace line_equation_l150_150627

-- Define the point A
def A : Point := ⟨0, 4⟩

-- Define the line equation that line l is perpendicular to
def line_perpendicular : Line := ⟨2, 1, -3⟩

-- Define the equation of the line l
def line_l : Line := ⟨1, -2, 8⟩

-- Prove the relationship using the conditions and the desired equation
theorem line_equation :
  (∀ P : Point, P ∈ line_l → (2 * P.1 + P.2 - 3 = 0 → 1 * P.1 - 2 * P.2 + 8 = 0)) ∧
  (∃ P : Point, P = A ∧ P ∈ line_l) :=
sorry

end line_equation_l150_150627


namespace jack_pays_back_total_l150_150277

noncomputable def principal : ℝ := 1200
noncomputable def rate : ℝ := 0.10
noncomputable def interest : ℝ := principal * rate
noncomputable def total : ℝ := principal + interest

theorem jack_pays_back_total : total = 1320 := by
  sorry

end jack_pays_back_total_l150_150277


namespace part1_part2_l150_150300


noncomputable def is_infinite_sum (a : ℕ → ℝ) : Prop :=
  ∀ (M : ℝ), ∃ (N : ℕ), ∀ n > N, (∑ k in finset.range n, a k) > M

theorem part1 :
  is_infinite_sum (λ n, 1 / (2 * (n : ℝ) - 1)) :=
sorry

theorem part2 : 
  ∃ (f : ℕ → ℕ), bijective f ∧ is_infinite_sum (λ n, (-1)^(f(n)-1) / (f(n))) :=
sorry

end part1_part2_l150_150300


namespace saree_sale_price_l150_150452

theorem saree_sale_price (original_price : ℝ) (first_discount : ℝ) (second_discount : ℝ) : 
  original_price = 298 ∧ first_discount = 0.12 ∧ second_discount = 0.15 → 
  let first_discount_amount := original_price * first_discount,
      price_after_first_discount := original_price - first_discount_amount,
      second_discount_amount := price_after_first_discount * second_discount,
      final_price := price_after_first_discount - second_discount_amount in
  final_price ≈ 223 :=
sorry

end saree_sale_price_l150_150452


namespace car_total_distance_in_12_hours_l150_150853

noncomputable def distance_travelled (hour : ℕ) : ℕ :=
  if hour = 0 then 0
  else 35 + (hour - 1) * 2

theorem car_total_distance_in_12_hours :
  (∑ i in Finset.range 12, distance_travelled (i + 1)) = 546 := 
by
  sorry

end car_total_distance_in_12_hours_l150_150853


namespace lucy_l150_150148

theorem lucy's_age 
  (L V: ℕ)
  (h1: L - 5 = 3 * (V - 5))
  (h2: L + 10 = 2 * (V + 10)) :
  L = 50 :=
by
  sorry

end lucy_l150_150148


namespace john_annual_payment_l150_150713

open Real

-- Definitions extracted from the problem:
def epipen_cost : ℝ := 500
def insurance_coverage : ℝ := 0.75
def epipen_frequency_per_year : ℕ := 2
def john_payment_per_epipen : ℝ := epipen_cost * (1 - insurance_coverage)

-- The statement to be proved:
theorem john_annual_payment :
  john_payment_per_epipen * epipen_frequency_per_year = 250 :=
by
  sorry

end john_annual_payment_l150_150713


namespace find_formulas_l150_150319

variable {ℕ : Type} [Nat] (n : ℕ) (a b : ℕ → ℕ) 

def S (n : ℕ) : ℕ := 
  ∑ i in range n, b i

axiom h1 : ∀ n : ℕ, S n = ℕ 1 - b n
axiom h2 : a 2 - 1 = 1 / b 1
axiom h3 : a 5 = 1 / b 3 + 1

theorem find_formulas :
  (∀ n : ℕ, a n = 2 * n - 1) ∧ (∀ n : ℕ, b n = 1 / 2^n) ∧ (∀ n : ℕ, T n = 3 - (2 * n + 3) / 2^n) := sorry

end find_formulas_l150_150319


namespace tank_leak_time_l150_150014

theorem tank_leak_time :
  (rate_inlet := 1 / 3)
  (rate_outlet := - (1 / 4))
  (combined_rate := 1 / 14)
  (leak_rate := rate_inlet + rate_outlet - combined_rate) :
  (time_to_leak := 1 / leak_rate) = 84 :=
by
  sorry

end tank_leak_time_l150_150014


namespace Robert_books_read_in_six_hours_l150_150355

theorem Robert_books_read_in_six_hours (P H T: ℕ)
    (h1: P = 270)
    (h2: H = 90)
    (h3: T = 6):
    T * H / P = 2 :=
by 
    -- sorry placeholder to indicate that this is where the proof goes.
    sorry

end Robert_books_read_in_six_hours_l150_150355


namespace probability_snow_at_least_once_l150_150806

noncomputable def probability_at_least_once_snow : ℚ :=
  1 - (↑((1:ℚ) / 4) ^ 5)

theorem probability_snow_at_least_once (p : ℚ) (h : p = 3 / 4) :
  probability_at_least_once_snow = 1023 / 1024 := by
  sorry

end probability_snow_at_least_once_l150_150806


namespace rex_cards_left_l150_150338

-- Definitions
def nicole_cards : ℕ := 400
def cindy_cards : ℕ := 2 * nicole_cards
def combined_total : ℕ := nicole_cards + cindy_cards
def rex_cards : ℕ := combined_total / 2
def people_count : ℕ := 4
def cards_per_person : ℕ := rex_cards / people_count

-- Proof statement
theorem rex_cards_left : cards_per_person = 150 := by
  sorry

end rex_cards_left_l150_150338


namespace number_of_paths_20_l150_150357

def grid := fin 4

def adj (a b : grid) : Prop := 
  (a = 0 ∧ b = 1) ∨ (a = 0 ∧ b = 3) ∨
  (a = 1 ∧ b = 0) ∨ (a = 1 ∧ b = 2) ∨
  (a = 2 ∧ b = 1) ∨ (a = 2 ∧ b = 3) ∨
  (a = 3 ∧ b = 0) ∨ (a = 3 ∧ b = 2)

def sum_on_path (path: list grid) : ℕ :=
  (path.map (λ x, match x with
                  | 0 => 1
                  | 1 => 2
                  | 2 => 3
                  | 3 => 4
                  end)).sum

def valid_path (path : list grid) : Prop :=
path.head = some 0 ∧ all_adjacent path
where all_adjacent : list grid -> Prop
| [] := true
| [x] := true
| x::y::rest := adj x y ∧ all_adjacent (y::rest)

theorem number_of_paths_20 (starting_square: grid) :
  finset.card ((list.filter (λ path, sum_on_path path = 20) 
                  (list.filter valid_path (list.permutations ([0, 1, 2, 3].bind (λ_, [0, 1, 2, 3])))))) 
                  = 167 :=
begin
  sorry
end

end number_of_paths_20_l150_150357


namespace largest_n_value_l150_150431

theorem largest_n_value : 
  ∃ (n : ℕ), 
    n < 200000 ∧ 
    10 * (n - 3)^5 - 2 * n^2 + 20 * n - 36 % 3 = 0 ∧ 
    ∀ (m : ℕ), m < 200000 → 10 * (m - 3)^5 - 2 * m^2 + 20 * m - 36 % 3 = 0 → m ≤ n :=
begin
  use 199999,
  split,
  { sorry }, -- Proof n < 200000, i.e., 199999 < 200000
  split,
  { sorry }, -- Proof the expression is multiple of 3 for n = 199999
  { sorry }  -- Prove that 199999 is the largest such n
end

end largest_n_value_l150_150431


namespace total_brownies_correct_l150_150332

noncomputable def initial_brownies : ℕ := 2 * 12
noncomputable def brownies_after_father : ℕ := initial_brownies - 8
noncomputable def brownies_after_mooney : ℕ := brownies_after_father - 4
noncomputable def additional_brownies : ℕ := 2 * 12
noncomputable def total_brownies : ℕ := brownies_after_mooney + additional_brownies

theorem total_brownies_correct : total_brownies = 36 := by
  sorry

end total_brownies_correct_l150_150332


namespace num_divisible_by_2_3_5_7_lt_500_l150_150216

theorem num_divisible_by_2_3_5_7_lt_500 : 
  (finset.count (λ n : ℕ, n < 500 ∧ (n % 2 = 0 ∧ n % 3 = 0 ∧ n % 5 = 0 ∧ n % 7 = 0)) (finset.range 500)) = 2 :=
sorry

end num_divisible_by_2_3_5_7_lt_500_l150_150216


namespace range_of_a_l150_150576

noncomputable def p (x : ℝ) : Prop := abs (3 * x - 4) > 2
noncomputable def q (x : ℝ) : Prop := 1 / (x^2 - x - 2) > 0
noncomputable def r (x a : ℝ) : Prop := (x - a) * (x - a - 1) < 0

theorem range_of_a {a : ℝ} :
  (∀ x : ℝ, ¬ r x a → ¬ p x) → (a ≥ 2 ∨ a ≤ -1/3) :=
sorry

end range_of_a_l150_150576


namespace countBeautifulDatesIn2023_l150_150911

def isBeautifulDate (date : Nat) (month : Nat) (year : Nat) : Bool :=
  let yearDigits := [2, 0, 2, 3]
  let dateDigits := (date.digits 10).erase_dup
  let monthDigits := (month.digits 10).erase_dup
  let allDigits := dateDigits ++ monthDigits ++ yearDigits
  allDigits.length == 6

theorem countBeautifulDatesIn2023 : 
  let year := 2023
  let validMonths := [1, 4, 5, 6, 7, 8, 9, 10]
  let validDays := List.range' 14 6 -- From 14 to 19
  6 * validMonths.length = 30 :=
by
  sorry

end countBeautifulDatesIn2023_l150_150911


namespace relationship_among_P_Q_R_l150_150878

variable {f : ℝ → ℝ}

def condition1 (x y : ℝ) (hx : -1 < x) (hx1 : x < 1) (hy : -1 < y) (hy1 : y < 1) : Prop :=
  f(x) - f(y) = f((x - y) / (1 - x * y))

def condition2 (x : ℝ) (hx : -1 < x) (hx1 : x < 0) : Prop :=
  f(x) > 0

def P : ℝ := f(1 / 5) + f(1 / 11)
def Q : ℝ := f(1 / 2)
def R : ℝ := f(0)

theorem relationship_among_P_Q_R
  (h1 : ∀ (x y : ℝ), -1 < x → x < 1 → -1 < y → y < 1 → condition1 x y)
  (h2 : ∀ (x : ℝ), -1 < x → x < 0 → condition2 x)
  (h3 : f(0) = 0) :
  R > P ∧ P > Q :=
sorry

end relationship_among_P_Q_R_l150_150878


namespace digit_B_divisible_by_9_l150_150384

theorem digit_B_divisible_by_9 (B : ℕ) (k : ℤ) (h1 : 0 ≤ B) (h2 : B ≤ 9) (h3 : 2 * B + 6 = 9 * k) : B = 6 := 
by
  sorry

end digit_B_divisible_by_9_l150_150384


namespace johnPaysPerYear_l150_150720

-- Define the conditions
def epiPenCost : ℝ := 500
def insuranceCoverage : ℝ := 0.75
def epiPenFrequencyPerYear : ℝ := 2 -- Twice a year since 12 months / 6 months per EpiPen

-- Calculate the cost after insurance
def costAfterInsurance (cost : ℝ) (coverage : ℝ) : ℝ :=
  cost * (1 - coverage)

-- Calculate the annual cost
def annualCost (freq : ℝ) (cost : ℝ) : ℝ :=
  freq * cost

-- The statement asserting the proof
theorem johnPaysPerYear (cost : ℝ) (coverage : ℝ) (freq : ℝ) : 
  epiPenCost = cost → 
  insuranceCoverage = coverage → 
  epiPenFrequencyPerYear = freq → 
  annualCost freq (costAfterInsurance cost coverage) = 250 := 
by 
  intros _ _ _ 
  sorry

end johnPaysPerYear_l150_150720


namespace infinite_lattice_points_in_swept_region_l150_150120

noncomputable def y_of_l (x : ℝ) : ℝ := (4 + Real.sqrt 15) * x

theorem infinite_lattice_points_in_swept_region 
  : ∃ᶠ P : ℕ × ℕ in Filter.cofinite, 
    (∃ x y : ℤ, 1 + 6 * P.1 * P.2 = x^2 ∧ 1 + 10 * P.1 * P.2 = y^2) ∧
    (P.1 ∣ (P.2^2 - 1) ∧ P.2 ∣ (P.1^2 - 1)) :=
begin
  sorry
end

end infinite_lattice_points_in_swept_region_l150_150120


namespace count_of_divisibles_l150_150210

theorem count_of_divisibles (n : ℕ) (h : n < 500) : 
  (∃ k, n = 2 * 3 * 5 * 7 * k ∧ k > 0) →
  (∃! k, k = 2) :=
by {
  have LCM := 210,
  -- sorry is used to skip the proof
  sorry
}

end count_of_divisibles_l150_150210


namespace bisecting_line_eq_l150_150936

theorem bisecting_line_eq : ∃ (a : ℝ), (∀ x y : ℝ, (y = a * x) ↔ y = -1 / 6 * x) ∧ 
  (∀ p : ℝ × ℝ, (3 * p.1 - 5 * p.2  = 6 → p.2 = a * p.1) ∧ 
                  (4 * p.1 + p.2 + 6 = 0 → p.2 = a * p.1)) :=
by
  use -1 / 6
  sorry

end bisecting_line_eq_l150_150936


namespace cos_alpha_l150_150187

-- Define the conditions
variable (α : Real)
variable (x y r : Real)
-- Given the point (-3, 4)
def point_condition (x : Real) (y : Real) : Prop := x = -3 ∧ y = 4

-- Define r as the distance
def radius_condition (x y r : Real) : Prop := r = Real.sqrt (x ^ 2 + y ^ 2)

-- Prove that cos α and cos 2α are the given values
theorem cos_alpha (α : Real) (x y r : Real) (h1 : point_condition x y) (h2 : radius_condition x y r) :
  Real.cos α = -3 / 5 ∧ Real.cos (2 * α) = -7 / 25 :=
by
  sorry

end cos_alpha_l150_150187


namespace determinant_expression_l150_150735

theorem determinant_expression (a b c p q : ℝ) 
  (h_root : ∀ x : ℝ, (x = a ∨ x = b ∨ x = c) → (Polynomial.eval x (Polynomial.X ^ 3 - 3 * Polynomial.C p * Polynomial.X + 2 * Polynomial.C q) = 0)) :
  Matrix.det ![![2 + a, 1, 1], ![1, 2 + b, 1], ![1, 1, 2 + c]] = -3 * p - 2 * q + 4 :=
by {
  sorry
}

end determinant_expression_l150_150735


namespace smallest_of_five_consecutive_l150_150002

theorem smallest_of_five_consecutive (n : ℤ) (h : (n - 2) + (n - 1) + n + (n + 1) + (n + 2) = 2015) : n - 2 = 401 :=
by sorry

end smallest_of_five_consecutive_l150_150002


namespace product_evaluation_l150_150539

theorem product_evaluation : (3 - 1) * 3 * (3 + 1) * (3 + 2) * (3 + 3) * (3 + 4) = 5040 := by
  -- sorry
  exact rfl  -- This is just a placeholder. The proof would go here.

end product_evaluation_l150_150539


namespace required_bollards_l150_150897

theorem required_bollards 
  (bollards_per_side : ℕ)
  (sides : ℕ)
  (fraction_installed : ℚ)
  : bollards_per_side = 4000 → 
    sides = 2 → 
    fraction_installed = 3/4 → 
    let total_bollards := bollards_per_side * sides in 
    let installed_bollards := fraction_installed * total_bollards in 
    let remaining_bollards := total_bollards - installed_bollards in 
    remaining_bollards = 2000 :=
by
  intros h1 h2 h3
  simp [h1, h2, h3]
  sorry

end required_bollards_l150_150897


namespace partial_derivative_equality_l150_150784

noncomputable def f (x y a : ℝ) : ℝ := x / (x^2 - a * y^2)

theorem partial_derivative_equality (x y a : ℝ) :
  (∂^2 (f x y a) / ∂ x^2) = a * (∂^2 (f x y a) / ∂ y^2) :=
by
  sorry

end partial_derivative_equality_l150_150784


namespace present_age_of_son_l150_150484

-- Define variables for the current ages of the son and the man (father).
variables (S M : ℕ)

-- Define the conditions:
-- The man is 35 years older than his son.
def condition1 : Prop := M = S + 35

-- In two years, the man's age will be twice the age of his son.
def condition2 : Prop := M + 2 = 2 * (S + 2)

-- The theorem that we need to prove:
theorem present_age_of_son : condition1 S M ∧ condition2 S M → S = 33 :=
by
  -- Add sorry to skip the proof.
  sorry

end present_age_of_son_l150_150484


namespace find_usual_time_l150_150046

noncomputable def journey_time (S T : ℝ) : Prop :=
  (6 / 5) = (T + (1 / 5)) / T

theorem find_usual_time (S T : ℝ) (h1 : ∀ S T, S / (5 / 6 * S) = (T + (12 / 60)) / T) : T = 1 :=
by
  -- Let the conditions defined by the user be:
  -- h1 : condition (e.g., the cab speed and time relationship)
  -- Given that the cab is \(\frac{5}{6}\) times its speed and is late by 12 minutes
  let h1 := journey_time S T
  sorry

end find_usual_time_l150_150046


namespace smallest_n_divisible_by_247_l150_150559

theorem smallest_n_divisible_by_247 : ∃ n : ℕ, n > 0 ∧ (∀ m : ℕ, m > 0 ∧ n > m → ( m * (m + 1) * (m + 2) ) % 247 ≠ 0) ∧ (n * (n + 1) * (n + 2)) % 247 = 0 :=
begin
  use 37,
  split,
  { -- n > 0
    exact nat.zero_lt_succ (nat.zero_lt_succ (nat.zero_lt_succ 34)),
  },
  split,
  { -- smallest n such that (n(n+1)(n+2)) % 247 = 0
    intros m hm1 hm2,
    sorry,
  },
  { -- (37 * 38 * 39) % 247 = 0
    unfold has_mod.mod,
    have h1 : 37 * (37 + 1) * (37 + 2) = 37 * 38 * 39 := by ring,
    rw [h1, ←nat.dvd_iff_mod_eq_zero],
    sorry,
  }
end

end smallest_n_divisible_by_247_l150_150559


namespace find_t_l150_150674

-- Define sets M and N
def M (t : ℝ) : Set ℝ := {1, t^2}
def N (t : ℝ) : Set ℝ := {-2, t + 2}

-- Goal: prove that t = 2 given M ∩ N ≠ ∅
theorem find_t (t : ℝ) (h : (M t ∩ N t).Nonempty) : t = 2 :=
sorry

end find_t_l150_150674


namespace jack_pays_back_total_l150_150276

noncomputable def principal : ℝ := 1200
noncomputable def rate : ℝ := 0.10
noncomputable def interest : ℝ := principal * rate
noncomputable def total : ℝ := principal + interest

theorem jack_pays_back_total : total = 1320 := by
  sorry

end jack_pays_back_total_l150_150276


namespace each_persons_contribution_l150_150360

def total_cost : ℝ := 67
def coupon : ℝ := 4
def num_people : ℝ := 3

theorem each_persons_contribution :
  (total_cost - coupon) / num_people = 21 := by
  sorry

end each_persons_contribution_l150_150360


namespace partition_sequences_l150_150507

/-- A sequence is defined as a list of length 2022 consisting of exactly 1011 zeros and 1011 ones. -/
def is_valid_sequence (s : List ℕ) :=
  s.length = 2022 ∧ s.count 0 = 1011 ∧ s.count 1 = 1011

/-- Two sequences are compatible if they match at exactly 4 positions. -/
def are_compatible (s1 s2 : List ℕ) :=
  (s1.zip s2).count (λ (x : ℕ × ℕ), x.fst = x.snd) = 4

/-- Prove that we can partition all valid sequences into 20 groups 
such that no two sequences in the same group are compatible. -/
theorem partition_sequences : 
  ∃ (groups : Finset (Finset (List ℕ))), 
    groups.card = 20 ∧ 
    (∀ g ∈ groups, 
     ∀ s1 s2 ∈ g, s1 ≠ s2 → 
     ¬ are_compatible s1 s2) :=
sorry

end partition_sequences_l150_150507


namespace inequality_sqrt_sum_l150_150987

theorem inequality_sqrt_sum (a b c : ℝ) (h1: a > 0) (h2: b > 0) (h3: c > 0) (h4: ab + bc + ca = 1) :
  sqrt(a^3 + a) + sqrt(b^3 + b) + sqrt(c^3 + c) ≥ 2 * sqrt(a + b + c) := 
sorry

end inequality_sqrt_sum_l150_150987


namespace vector_BC_eq_neg7_neg4_l150_150595

-- Definitions of points and vectors
structure Point where
  x : ℝ
  y : ℝ

def A : Point := ⟨0, 1⟩
def B : Point := ⟨3, 2⟩

-- Definitions of vectors
structure Vector where
  x : ℝ
  y : ℝ

def AC : Vector := ⟨-4, -3⟩

-- Vector subtraction
def vector_sub (v1 v2 : Vector) : Vector :=
  ⟨v1.x - v2.x, v1.y - v2.y⟩

-- Calculate AB
def AB : Vector := ⟨B.x - A.x, B.y - A.y⟩

-- Statement to prove
theorem vector_BC_eq_neg7_neg4 : vector_sub AC AB = ⟨-7, -4⟩ :=
by
  sorry

end vector_BC_eq_neg7_neg4_l150_150595


namespace incorrect_connection_probability_is_correct_l150_150415

noncomputable def incorrect_connection_probability : ℝ :=
  let p := 0.02 in
  let C := (n k : ℕ) => Nat.choose n k in
  let r2 := 1/9 in
  let r3 := 8/81 in
  C 3 2 * p^2 * (1 - p) * r2 + C 3 3 * p^3 * r3

theorem incorrect_connection_probability_is_correct :
  incorrect_connection_probability ≈ 0.000131 :=
  sorry

end incorrect_connection_probability_is_correct_l150_150415


namespace coefficient_x_term_expansion_l150_150935

theorem coefficient_x_term_expansion (X : ℝ) : 
  (coeff (expand (X * (1 + X)^6)) 1) = 15 := 
sorry

end coefficient_x_term_expansion_l150_150935


namespace greatest_possible_value_of_q_minus_r_l150_150451

noncomputable def max_difference (q r : ℕ) : ℕ :=
  if q < r then r - q else q - r

theorem greatest_possible_value_of_q_minus_r (q r : ℕ) (x y : ℕ) (hq : q = 10 * x + y) (hr : r = 10 * y + x) (cond : q ≠ r) (hqr : max_difference q r < 20) : q - r = 18 :=
  sorry

end greatest_possible_value_of_q_minus_r_l150_150451


namespace binomial_510_510_l150_150925

theorem binomial_510_510 : Nat.choose 510 510 = 1 :=
by
  sorry

end binomial_510_510_l150_150925


namespace problem1_problem2_l150_150519

-- Proof problem 1
theorem problem1 : (-3)^2 / 3 + abs (-7) + 3 * (-1/3) = 3 :=
by
  sorry

-- Proof problem 2
theorem problem2 : (-1) ^ 2022 - ( (-1/4) - (-1/3) ) / (-1/12) = 2 :=
by
  sorry

end problem1_problem2_l150_150519


namespace correct_options_l150_150842

noncomputable theory
open_locale classical

variables (a b : ℝ × ℝ × ℝ)
variables (c m : ℝ × ℝ × ℝ)
variables (u v d n : ℝ × ℝ × ℝ)
variables (θ : ℝ)

def is_parallel (v1 v2 : ℝ × ℝ × ℝ) : Prop := ∃ k : ℝ, v1 = k • v2

def is_perpendicular (v1 v2 : ℝ × ℝ × ℝ) : Prop := v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3 = 0

def angle_between_line_and_plane (d n : ℝ × ℝ × ℝ) (θ : ℝ) : Prop :=
  sin θ = abs ((d.1 * n.1 + d.2 * n.2 + d.3 * n.3) / (real.sqrt (d.1 ^ 2 + d.2 ^ 2 + d.3 ^ 2) * real.sqrt (n.1 ^ 2 + n.2 ^ 2 + n.3 ^ 2)))

theorem correct_options :
  let a := (2, 0, -1)
  let b := (-4, 0, 2)
  let c := (1, -1, 2)
  let m := (6, 4, -1)
  let u := (2, 2, -1)
  let v := (-3, 4, 2)
  let d := (0, 1, 1)
  let n := (1, 0, 1)
  ∃ (θ_real : ℝ), (θ_real = real.pi / 3) →
  (is_parallel a b) ∧ (is_perpendicular u v) :=
by
  sorry

end correct_options_l150_150842


namespace average_weight_l150_150687

theorem average_weight {w : ℝ} 
  (h1 : 62 < w) 
  (h2 : w < 72) 
  (h3 : 60 < w) 
  (h4 : w < 70) 
  (h5 : w ≤ 65) : w = 63.5 :=
by
  sorry

end average_weight_l150_150687


namespace range_of_a_l150_150242

-- Define the function f(x)
def f (a x : ℝ) : ℝ := (1/3) * x^3 - (1/2) * a * x^2 + (a - 1) * x + 1

-- Define the derivative of f(x)
def f' (a x : ℝ) : ℝ := x^2 - a * x + (a - 1)

-- The statement that needs to be proven
theorem range_of_a 
  (f_increasing : ∀ x > 1, f' a x ≥ 0) :
  a ≤ 2 :=
sorry

end range_of_a_l150_150242


namespace smallest_four_digit_equiv_mod_five_l150_150833

theorem smallest_four_digit_equiv_mod_five : 
  ∃ (n : ℤ), n >= 1000 ∧ n % 5 = 4 ∧ ∀ m, (m >= 1000 ∧ m % 5 = 4) → n ≤ m :=
by
  use 1004
  split
  sorry

end smallest_four_digit_equiv_mod_five_l150_150833


namespace john_pays_per_year_l150_150717

-- Define the costs and insurance parameters.
def cost_per_epipen : ℝ := 500
def insurance_coverage : ℝ := 0.75

-- Number of months in a year.
def months_in_year : ℕ := 12

-- Number of months each EpiPen lasts.
def months_per_epipen : ℕ := 6

-- Amount covered by insurance for each EpiPen.
def insurance_amount (cost : ℝ) (coverage : ℝ) : ℝ :=
  cost * coverage

-- Amount John pays after insurance for each EpiPen.
def amount_john_pays_per_epipen (cost : ℝ) (covered: ℝ) : ℝ :=
  cost - covered

-- Number of EpiPens John needs per year.
def epipens_per_year (months_in_year : ℕ) (months_per_epipen : ℕ) : ℕ :=
  months_in_year / months_per_epipen

-- Total amount John pays per year.
def total_amount_john_pays_per_year (amount_per_epipen : ℝ) (epipens_per_year : ℕ) : ℝ :=
  amount_per_epipen * epipens_per_year

-- Theorem to prove the correct answer.
theorem john_pays_per_year :
  total_amount_john_pays_per_year (amount_john_pays_per_epipen cost_per_epipen (insurance_amount cost_per_epipen insurance_coverage)) (epipens_per_year months_in_year months_per_epipen) = 250 := 
by
  sorry

end john_pays_per_year_l150_150717


namespace bobs_average_speed_is_correct_l150_150106

noncomputable def bobsAverageSpeed : ℝ :=
  let distance1 := 380 
  let distance2 := 420
  let distance3 := 400
  let time1 := 70 
  let time2 := 85
  let time3 := 80 
  let total_distance := distance1 + distance2 + distance3
  let total_time := time1 + time2 + time3
  total_distance / total_time

theorem bobs_average_speed_is_correct : abs(bobsAverageSpeed - 5.106) < 0.001 :=
by
  rw [bobsAverageSpeed]
  norm_num
  sorry

end bobs_average_speed_is_correct_l150_150106


namespace area_of_union_of_six_triangles_l150_150882

-- Define the conditions
def equilateral_triangle_side_length : ℝ := 3 * real.sqrt 3

def num_triangles : ℕ := 6

def triangle_area (side_length : ℝ) : ℝ :=
  (real.sqrt 3 / 4) * (side_length ^ 2)

def overlap_area (side_length : ℝ) : ℝ :=
  (real.sqrt 3 / 4) * ((side_length / 2) ^ 2)

-- Define the goal
theorem area_of_union_of_six_triangles 
  (s : ℝ)
  (n : ℕ)
  (h_s : s = 3 * real.sqrt 3)
  (h_n : n = 6) :
  let total_area := n * triangle_area s,
      total_overlap := 5 * overlap_area s,
      net_area := total_area - total_overlap 
  in net_area = 513 * real.sqrt 3 / 16 := 
sorry

end area_of_union_of_six_triangles_l150_150882


namespace dan_has_3_potatoes_left_l150_150123

-- Defining the number of potatoes Dan originally had
def original_potatoes : ℕ := 7

-- Defining the number of potatoes the rabbits ate
def potatoes_eaten : ℕ := 4

-- The theorem we want to prove: Dan has 3 potatoes left.
theorem dan_has_3_potatoes_left : original_potatoes - potatoes_eaten = 3 := by
  sorry

end dan_has_3_potatoes_left_l150_150123


namespace darnel_distance_calc_l150_150125

theorem darnel_distance_calc:
  let sprint1 := 0.88 in
  let sprint2 := 1.12 in
  let jog1 := 0.75 in
  let jog2 := 0.45 in
  let walk := 0.32 in
  let total_distance := sprint1 + sprint2 + jog1 + jog2 + walk in
  let total_sprint := sprint1 + sprint2 in
  let total_jog := jog1 + jog2 in
  let additional_sprint := total_sprint - (total_jog + walk) in
  total_distance = 3.52 ∧ additional_sprint = 0.48 :=
by {
  -- sorry placeholder for proof
  sorry
}

end darnel_distance_calc_l150_150125


namespace bug_travel_distance_half_l150_150880

-- Define the conditions
def isHexagonalGrid (side_length : ℝ) : Prop :=
  side_length = 1

def shortest_path_length (path_length : ℝ) : Prop :=
  path_length = 100

-- Define a theorem that encapsulates the problem statement
theorem bug_travel_distance_half (side_length path_length : ℝ)
  (H1 : isHexagonalGrid side_length)
  (H2 : shortest_path_length path_length) :
  ∃ one_direction_distance : ℝ, one_direction_distance = path_length / 2 :=
sorry -- Proof to be provided.

end bug_travel_distance_half_l150_150880


namespace triangle_segments_length_l150_150249

noncomputable def triangle_side_lengths (AB BC AC : ℕ) : Prop :=
  AB = 400 ∧ BC = 480 ∧ AC = 560

noncomputable def segments_equal_length (d : ℚ) : Prop :=
  d = 218 + 2 / 9

theorem triangle_segments_length :
  ∀ (P : Point) (AB BC AC : ℕ) (d : ℚ),
  triangle_side_lengths AB BC AC →
  segments_equal_length d →
  ∃ (ΔABC : Triangle) (segments_through_P : Segment),
  segment_length segments_through_P = d :=
by
  sorry

end triangle_segments_length_l150_150249


namespace total_weight_is_28_87_l150_150777

def blue_ball_weight : ℝ := 6
def brown_ball_weight : ℝ := 3.12
def green_ball_weight : ℝ := 4.25

def red_ball_weight : ℝ := 2 * green_ball_weight
def yellow_ball_weight : ℝ := red_ball_weight - 1.5

def total_weight : ℝ := blue_ball_weight + brown_ball_weight + green_ball_weight + red_ball_weight + yellow_ball_weight

theorem total_weight_is_28_87 : total_weight = 28.87 :=
by
  /- proof goes here -/
  sorry

end total_weight_is_28_87_l150_150777


namespace sqrt_xyz_sum_l150_150305

def real_nums (x y z : ℝ) : Prop :=
  (y + z = 16) ∧ (z + x = 18) ∧ (x + y = 20)

theorem sqrt_xyz_sum (x y z : ℝ) (h : real_nums x y z) : 
  sqrt (x * y * z * (x + y + z)) = 9 * sqrt 77 :=
by 
  sorry

end sqrt_xyz_sum_l150_150305


namespace carrie_strawberries_l150_150110

theorem carrie_strawberries
    (length : ℕ) (width : ℕ) (density : ℕ) (yield_per_plant : ℕ)
    (garden_area : ℕ = length * width)
    (total_plants : ℕ = density * garden_area)
    (total_strawberries : ℕ = yield_per_plant * total_plants) :
    length = 10 → width = 9 → density = 5 → yield_per_plant = 12 → total_strawberries = 5400 := by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  simp
  sorry

end carrie_strawberries_l150_150110


namespace a_2004_bounds_l150_150199

noncomputable def a_seq : ℕ → ℝ
| 0     := 1
| 1     := 2
| (n+2) := a_seq (n+1) + (1 / a_seq (n+1))

theorem a_2004_bounds : 63 < a_seq 2004 ∧ a_seq 2004 < 78 :=
sorry

end a_2004_bounds_l150_150199


namespace find_x_values_l150_150139

theorem find_x_values :
  ∀ x : ℝ, 
  (8 / (sqrt (x - 9) - 10) + 
   2 / (sqrt (x - 9) - 5) + 
   9 / (sqrt (x - 9) + 5) + 
   15 / (sqrt (x - 9) + 10) = 0) ↔ 
  (x ≈ 12.777 ∨ x ≈ 14.556 ∨ x ≈ 26.882) :=
sorry

end find_x_values_l150_150139


namespace find_p_plus_q_l150_150297

-- Define a point Q in the unit square
structure Point where
  x : ℝ
  y : ℝ
  deriving Repr

-- Q is a uniformly random point in the interior of the unit square
def isInUnitSquare (Q : Point) : Prop :=
  0 ≤ Q.x ∧ Q.x ≤ 1 ∧ 0 ≤ Q.y ∧ Q.y ≤ 1

-- The slope condition that the slope of the line through Q and (1/2, 1/4) is ≥ 1
def slopeCondition (Q : Point) : Prop :=
  (Q.y - 1/4) ≥ (Q.x - 1/2)

-- Define the probability that the point Q meets the slope condition
def probabilityOfSlopeCondition : ℚ := 9/32

-- Prove the statement that p + q = 41 where p and q are relatively prime such that probability = p/q
theorem find_p_plus_q (p q : ℕ) (h1 : Nat.gcd p q = 1) (h2 : probabilityOfSlopeCondition = p / q) : p + q = 41 := sorry

end find_p_plus_q_l150_150297


namespace functional_equation_solution_l150_150856

theorem functional_equation_solution :
  ∀ (f : ℝ → ℝ), (∀ x y : ℝ, f (x + f y) + f (y + f x) = 2 * f (x * f y)) → (∀ x : ℝ, f x = 0) :=
by
  intro f h
  sorry

end functional_equation_solution_l150_150856


namespace problem1_problem2_l150_150272

variable (A B C a b c p : ℝ)
variable (hA : A = ∠ A)
variable (hC : C = ∠ C)
variable (ha : a = side opposite A)
variable (hb : b = side opposite B)
variable (hc : c = side opposite C)
variable (hsin : sin A + sin C = p * sin B)
variable (h_fourac : 4 * a * c = b * b)

theorem problem1 :
  p = 5 / 4 → b = 1 →
  (a = 1 ∧ c = 1 / 4) ∨ (a = 1 / 4 ∧ c = 1) :=
by
  intros hp hb1
  sorry

theorem problem2 :
  (cos B > 0) →
  (cos B < 1) →
  (b = sqrt (4 * a * c)) →
  (∃ p, (sqrt(6) / 2 < p ∧ p < sqrt 2)) :=
by
  intros h1 h2 h3
  sorry

end problem1_problem2_l150_150272


namespace beautiful_dates_in_2023_l150_150914

def is_beautiful_date (d1 d2 m1 m2 y1 y2 : ℕ) : Prop :=
  let digits := [d1, d2, m1, m2, y1, y2]
  (digits.nodup) ∧ (d1 < 10) ∧ (d2 < 10) ∧ (m1 < 10) ∧ (m2 < 10) ∧ (y1 < 10) ∧ (y2 < 10)

theorem beautiful_dates_in_2023 : ∃ n, n = 30 ∧
  n = (Σ m1 m2 d1 d2, is_beautiful_date d1 d2 m1 m2 2 0 ∧ is_beautiful_date d1 d2 m1 m2 2 3) :=
sorry

end beautiful_dates_in_2023_l150_150914


namespace velocity_upper_end_l150_150096

variable (a l τ : ℝ)

-- Assuming the given conditions
def x := a * τ^2 / 2
def v1 := a * τ
def sin_alpha := a * τ^2 / (2 * l)
def cos_alpha := sqrt (4 * l^2 - a^2 * τ^4) / (2 * l)

-- Define the mathematical question we need to prove
theorem velocity_upper_end (h1 : v1 * sin_alpha = (a * τ) * (a * τ^2 / (2 * l)))
  : ∃ v2, v2 = a^2 * τ^3 / sqrt (4 * l^2 - a^2 * τ^4) :=
by
  use a^2 * τ^3 / sqrt (4 * l^2 - a^2 * τ^4)
  sorry

end velocity_upper_end_l150_150096


namespace sebastian_age_correct_l150_150149

-- Define the ages involved
def sebastian_age_now := 40
def sister_age_now (S : ℕ) := S - 10
def father_age_now := 85

-- Define the conditions
def age_difference_condition (S : ℕ) := (sister_age_now S) = S - 10
def father_age_condition := father_age_now = 85
def past_age_sum_condition (S : ℕ) := (S - 5) + (sister_age_now S - 5) = 3 / 4 * (father_age_now - 5)

theorem sebastian_age_correct (S : ℕ) 
  (h1 : age_difference_condition S) 
  (h2 : father_age_condition) 
  (h3 : past_age_sum_condition S) : 
  S = sebastian_age_now := 
  by sorry

end sebastian_age_correct_l150_150149


namespace smallest_constant_inequality_l150_150304

theorem smallest_constant_inequality (n : ℤ) (x : ℕ → ℝ) (h : n ≥ 2) (hx : ∀ i, x i ≥ 0) : 
    ∃ c : ℝ, (∀ n ≥ 2, ∀ x : ℕ → ℝ, (∀ i, x i ≥ 0) → 
    (∑ (i : ℕ) in range (n + 1), ∑ (j : ℕ) in range (n + 1), if i < j then x i * x j * (x i ^ 2 + x j ^ 2) else 0) ≤ 
    c * (∑ (i : ℕ) in range (n + 1), x i) ^ 4) ∧ c = 1/8 :=
begin
  -- Proof goes here
  sorry
end

end smallest_constant_inequality_l150_150304


namespace argument_of_sum_is_17pi_over_36_l150_150919

theorem argument_of_sum_is_17pi_over_36 :
  ∀ (z1 z2 z3 z4 z5 : ℂ), 
    z1 = exp(5 * real.pi * complex.I / 36) →
    z2 = exp(11 * real.pi * complex.I / 36) →
    z3 = exp(17 * real.pi * complex.I / 36) →
    z4 = exp(23 * real.pi * complex.I / 36) →
    z5 = exp(29 * real.pi * complex.I / 36) →
    complex.arg (z1 + z2 + z3 + z4 + z5) = 17 * real.pi / 36 :=
by
  intros z1 z2 z3 z4 z5 H1 H2 H3 H4 H5
  sorry

end argument_of_sum_is_17pi_over_36_l150_150919


namespace num_divisible_by_2_3_5_7_lt_500_l150_150214

theorem num_divisible_by_2_3_5_7_lt_500 : 
  (finset.count (λ n : ℕ, n < 500 ∧ (n % 2 = 0 ∧ n % 3 = 0 ∧ n % 5 = 0 ∧ n % 7 = 0)) (finset.range 500)) = 2 :=
sorry

end num_divisible_by_2_3_5_7_lt_500_l150_150214


namespace convex_polygon_of_empty_triangles_l150_150787

def isEmptyTriangle {M : set (ℝ × ℝ)} (p1 p2 p3 : ℝ × ℝ) : Prop :=
∀ x ∈ M, ¬ (x ≠ p1 ∧ x ≠ p2 ∧ x ≠ p3 ∧ is_in_triangle x p1 p2 p3)

def is_in_triangle (p : ℝ × ℝ) (a b c : ℝ × ℝ) : Prop :=
-- Define the condition for a point p to be inside the triangle formed by points a, b, c
-- You could use inequalities involving determinants or a similar method.
sorry

theorem convex_polygon_of_empty_triangles 
  (M : set (ℝ × ℝ)) 
  (hM : ∀ p1 p2 p3 ∈ M, isEmptyTriangle p1 p2 p3) :
  ∃ poly : ℝ → ℝ → Prop, is_convex_polygon poly ∧ ∀ p ∈ M, poly p :=
sorry

end convex_polygon_of_empty_triangles_l150_150787


namespace domino_square_sum_possible_values_l150_150829

theorem domino_square_sum_possible_values :
  (∃ S : ℕ, S ∈ {22, 23, 24, 25, 26} ∧ (∀ (a b c d : ℕ), 
  a ∈ (∅ : finset ℕ) ∧ b ∈ (∅ : finset ℕ) ∧ c ∈ (∅ : finset ℕ) ∧ d ∈ (∅ : finset ℕ) ∧ 
  (a + b + c + d = 8 * S) 
  ∧ (∑ x in finset.range 28, x ≤ 6 * 28 ∧ x ≥ 0) 
  ∧ (∀ x : ℕ, x ∈ finset.range 7 × finset.range 7)) 
  ∧ S ∈ range 7*28 :=
  by sorry

end domino_square_sum_possible_values_l150_150829


namespace num_divisible_by_2_3_5_7_lt_500_l150_150217

theorem num_divisible_by_2_3_5_7_lt_500 : 
  (finset.count (λ n : ℕ, n < 500 ∧ (n % 2 = 0 ∧ n % 3 = 0 ∧ n % 5 = 0 ∧ n % 7 = 0)) (finset.range 500)) = 2 :=
sorry

end num_divisible_by_2_3_5_7_lt_500_l150_150217


namespace quadratic_roots_evaluation_l150_150245

theorem quadratic_roots_evaluation (x1 x2 : ℝ) (h1 : x1 + x2 = 1) (h2 : x1 * x2 = -2) :
  (1 + x1) + x2 * (1 - x1) = 4 :=
by
  sorry

end quadratic_roots_evaluation_l150_150245


namespace max_non_attacking_rooks_l150_150963

-- Define the 12x12 board and the cut-out 4x4 square.
structure Board :=
  (rows : Fin 12)
  (cols : Fin 12)

def is_removed (b : Board) : Prop :=
  4 ≤ b.rows.val ∧ b.rows.val ≤ 7 ∧ 4 ≤ b.cols.val ∧ b.cols.val ≤ 7

-- Define Rook placement as a set of non-attacking rooks on the board.
def non_attacking_rooks (rooks : Finset Board) : Prop :=
  rooks.pairwise (λ b1 b2 => b1.rows ≠ b2.rows ∧ b1.cols ≠ b2.cols)

-- Define the maximum number of non-attacking rooks under the given conditions.
theorem max_non_attacking_rooks (S : Finset Board) (h : ∀ b ∈ S, ¬ is_removed b) : S.card ≤ 15 :=
sorry

end max_non_attacking_rooks_l150_150963


namespace min_sides_perimeter_l150_150767

def perimeter_min_sides (A B C D E F G H : Type*) (adj_perpendicular : (A -> B -> Type) -> (B -> C -> Type) -> (C -> D -> Type) -> (D -> E -> Type) -> (E -> F -> Type) -> (F -> G -> Type) -> (G -> H -> Type) -> (H -> A -> Type) -> Prop) : Prop :=
  (∀ (horiz : A -> H -> Prop) (horiz_GF : G -> F -> Prop) (horiz_ED : E -> D -> Prop), horiz = horiz_GF ∧ horiz_GF = horiz_ED ∧ horiz = (B -> C -> Prop)) ∧
  (∀ (vert_HG : H -> G -> Prop) (vert_DC : D -> C -> Prop), vert_HG = vert_DC ∧ vert_HG = (A -> B -> Prop) ∧ vert_DC = (E -> F -> Prop) → 
  ∃ (min_sides : Nat), min_sides = 3)

theorem min_sides_perimeter (A B C D E F G H : Type*) (adj_perpendicular : (A -> B -> Type) -> (B -> C -> Type) -> (C -> D -> Type) -> (D -> E -> Type) -> (E -> F -> Type) -> (F -> G -> Type) -> (G -> H -> Type) -> (H -> A -> Type) -> Prop) :
  perimeter_min_sides A B C D E F G H adj_perpendicular :=
by
  sorry

end min_sides_perimeter_l150_150767


namespace area_triangle_ABE_l150_150252

def Point : Type := ℝ × ℝ  -- A point in 2D space (x, y)

-- Define points A, B, and E
def A : Point := (0, 0)  -- Intersection of Park Avenue and Sunflower Street
def B : Point := (0, 5)  -- 5 miles north of A
def E : Point := (4, 0)  -- 4 miles east of A along Sunflower Street

-- Function to calculate the area of a triangle given its vertices
def triangle_area (A B E : Point) : ℝ :=
  1 / 2 * abs ((B.1 - A.1) * (E.2 - A.2) - (E.1 - A.1) * (B.2 - A.2))

-- Prove that the area of triangle ABE is 10 square miles
theorem area_triangle_ABE : triangle_area A B E = 10 :=
  by
    -- Detailed proof would go here
    sorry

end area_triangle_ABE_l150_150252


namespace pipe_R_fill_time_l150_150766

theorem pipe_R_fill_time :
  (1 / 6 + 1 / 12 + 1 / x = 2 / 7) → x = 28 :=
by 
  intro h
  replace h := h.symm
  have h1 : (1 / 6 + 1 / 12 = 3 / 12), by norm_num
  have h2 : (3 / 12 + 1 / x = 2 / 7), from h,
  sorry

end pipe_R_fill_time_l150_150766


namespace num_divisible_by_2_3_5_7_under_500_l150_150220

-- Let the LCM of 2, 3, 5, and 7 be computed
def LCM_2_3_5_7 : ℕ := Nat.lcm (Nat.lcm 2 3) (Nat.lcm 5 7)

def num_multiples_less_than (n k : ℕ) : ℕ :=
  n / k

theorem num_divisible_by_2_3_5_7_under_500 : num_multiples_less_than 500 LCM_2_3_5_7 = 2 := by
  sorry

end num_divisible_by_2_3_5_7_under_500_l150_150220


namespace probability_four_or_more_same_value_l150_150147

theorem probability_four_or_more_same_value :
  let n := 5 -- number of dice
  let d := 10 -- number of sides on each die
  let event := "at least four of the five dice show the same value"
  let probability := (23 : ℚ) / 5000 -- given probability
  n = 5 ∧ d = 10 ∧ event = "at least four of the five dice show the same value" →
  (probability = 23 / 5000) := 
by
  intros
  sorry

end probability_four_or_more_same_value_l150_150147


namespace ellipse_tangent_line_l150_150266

section

variables {a b : ℝ} (a_pos : 0 < a) (b_pos : 0 < b) (h : a > b)  
          (focus_condition : ∃ c, c = 1 ∧ (sqrt (a^2 - b^2)) = c) 
          (point_condition : ∃ x y, x = 0 ∧ y = 1 ∧ (x^2 / a^2 + y^2 / b^2 = 1)) 
          (parabola : ∀ x y, y^2 = 4 * x) 

theorem ellipse_tangent_line :
  (a = sqrt 2) → (b = 1) → 
  (1 = sqrt (a^2 - b^2)) → 
  (∃ l : ℝ → ℝ, (∀ x y, (y = l x) ↔ (y = (sqrt 2 / 2) * x + sqrt 2) ∨ (y = -(sqrt 2 / 2) * x - sqrt 2)) ∧ 
     ∀ x y, y^2 = 4 * x → y = l x) :=
by
  sorry

end

end ellipse_tangent_line_l150_150266


namespace div_by_9_digit_B_l150_150381

theorem div_by_9_digit_B (B : ℕ) (h : (4 + B + B + 2) % 9 = 0) : B = 6 :=
by sorry

end div_by_9_digit_B_l150_150381


namespace part_a_part_b_part_c_l150_150803

def perfect_square_sums_possible (n : ℕ) : Prop :=
  ∃ f : fin n → fin n, ∀ k : fin n, ∃ m : ℕ, (k + f k = m * m)

theorem part_a : perfect_square_sums_possible 9 :=
sorry

theorem part_b : ¬ perfect_square_sums_possible 11 :=
sorry

theorem part_c : perfect_square_sums_possible 1996 :=
sorry

end part_a_part_b_part_c_l150_150803


namespace detergent_required_l150_150334

def ounces_of_detergent_per_pound : ℕ := 2
def pounds_of_clothes : ℕ := 9

theorem detergent_required :
  (ounces_of_detergent_per_pound * pounds_of_clothes) = 18 := by
  sorry

end detergent_required_l150_150334


namespace fraction_addition_l150_150137

theorem fraction_addition (a b c d : ℚ) (h1 : a = 3/4) (h2 : b = 5/9) : a + b = 47/36 :=
by
  rw [h1, h2]
  sorry

end fraction_addition_l150_150137


namespace total_students_in_classes_l150_150426

theorem total_students_in_classes (t1 t2 x y: ℕ) (h1 : t1 = 273) (h2 : t2 = 273) (h3 : (x - 1) * 7 = t1) (h4 : (y - 1) * 13 = t2) : x + y = 62 :=
by
  sorry

end total_students_in_classes_l150_150426


namespace division_of_decimals_l150_150115

theorem division_of_decimals : 0.36 / 0.004 = 90 := by
  sorry

end division_of_decimals_l150_150115


namespace ball_total_distance_third_touch_l150_150764

theorem ball_total_distance_third_touch :
  let initial_height := 100
  let rebound_factor := 0.5
  let first_drop := initial_height
  let first_rebound := initial_height * rebound_factor
  let second_drop := first_rebound
  let second_rebound := first_rebound * rebound_factor
  total_distance (first_drop + first_rebound + second_drop) = 200 := by
  sorry

end ball_total_distance_third_touch_l150_150764


namespace max_diagonal_intersections_l150_150534

theorem max_diagonal_intersections (n : ℕ) (h : n ≥ 4) : 
    ∃ k, k = n * (n - 1) * (n - 2) * (n - 3) / 24 :=
by
    sorry

end max_diagonal_intersections_l150_150534


namespace find_b_plus_m_l150_150523

open Matrix

noncomputable def X (b : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![1, 3, b], ![0, 1, 5], ![0, 0, 1]]

noncomputable def Y : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![1, 27, 8085], ![0, 1, 45], ![0, 0, 1]]

theorem find_b_plus_m (b m : ℝ)
    (h1 : X b ^ m = Y) : b + m = 847 := sorry

end find_b_plus_m_l150_150523


namespace range_of_x_l150_150239

theorem range_of_x (x : ℝ) : (1 / real.sqrt (x - 2) : ℝ) ∈ set_of (λ y, y ∈ real) → x > 2 := 
by
sorry

end range_of_x_l150_150239


namespace tan_alpha_solution_l150_150619

variable (α : ℝ)
variable (h₀ : 0 < α ∧ α < π)
variable (h₁ : Real.sin α + Real.cos α = 7 / 13)

theorem tan_alpha_solution : Real.tan α = -12 / 5 := 
by
  sorry

end tan_alpha_solution_l150_150619


namespace order_of_variables_l150_150574

theorem order_of_variables : 
  let a := 2 ^ 0.3
  let b := Real.log2 1.5
  let c := Real.log 0.7
  in a > b ∧ b > c :=
by
  sorry

end order_of_variables_l150_150574


namespace smallest_five_digit_in_pascal_l150_150436

-- Define the conditions
def pascal_triangle_increases (n k : ℕ) : Prop := 
  ∀ (r ≥ n) (c ≥ k), c ≤ r → ∃ (x : ℕ), x >= Nat.choose r c

def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

-- State the proof problem and the expected answer
theorem smallest_five_digit_in_pascal :
  (∃ (n k : ℕ), binomial_coefficient n k = 10000) ∧ (∀ (m l : ℕ), binomial_coefficient m l = 10000 → n ≤ m) := sorry

end smallest_five_digit_in_pascal_l150_150436


namespace correct_statement_D_l150_150843

-- Definitions around planes and lines and their properties
variables {α β : Type*} [plane α] [plane β] [line l]

-- Given conditions for the problem
variable (h : ¬perpendicular α β)

-- The proof that 'if plane α is not perpendicular to plane β,
-- then there is no line in β perpendicular to plane α'
theorem correct_statement_D :
  ¬ ∃ l : β, perpendicular l α :=
by
  sorry

end correct_statement_D_l150_150843


namespace smallest_five_digit_in_pascals_triangle_l150_150438

/-- In Pascal's triangle, the smallest five-digit number is 10000. -/
theorem smallest_five_digit_in_pascals_triangle : 
  ∃ (n k : ℕ), (10000 = Nat.choose n k) ∧ (∀ m l : ℕ, Nat.choose m l < 10000) → (n > m) := 
sorry

end smallest_five_digit_in_pascals_triangle_l150_150438


namespace who_visited_HappinessSquare_l150_150079

inductive Student
| A | B | C | D

open Student

def statement_A (x: Student): Prop := (x ≠ A)
def statement_B (x: Student): Prop := (x = D)
def statement_C (x: Student): Prop := (x = B)
def statement_D (x: Student): Prop := (x ≠ D)

def exactly_one_visited (x: Student): Prop := 
  (x = A) ∨ (x = B) ∨ (x = C) ∨ (x = D) ∧
  (x ≠ A ∨ x ≠ B ∨ x ≠ C ∨ x ≠ D)

def exactly_one_lied (x: Student): Prop := 
  let lies := [statement_A x, statement_B x, statement_C x, statement_D x] in
  (count lies id = 1)

theorem who_visited_HappinessSquare: ∃ x : Student, exactly_one_visited x ∧ exactly_one_lied x ∧ (x = B) :=
sorry

end who_visited_HappinessSquare_l150_150079


namespace factorization_of_polynomial_l150_150944

theorem factorization_of_polynomial (x : ℝ) :
  x^6 - x^4 - x^2 + 1 = (x - 1) * (x + 1) * (x^2 + 1) := 
sorry

end factorization_of_polynomial_l150_150944


namespace event_day_price_l150_150104

theorem event_day_price (original_price : ℝ) (first_discount second_discount : ℝ)
  (h1 : original_price = 250) (h2 : first_discount = 0.4) (h3 : second_discount = 0.25) : 
  ∃ discounted_price : ℝ, 
  discounted_price = (original_price * (1 - first_discount)) * (1 - second_discount) → 
  discounted_price = 112.5 :=
by
  use (250 * (1 - 0.4) * (1 - 0.25))
  sorry

end event_day_price_l150_150104


namespace surface_area_of_cube_in_terms_of_d_l150_150246

-- Definitions based on the conditions
def volume_of_cube (x : ℝ) : ℝ := x^3

def space_diagonal (d : ℝ) : Prop := ∀ (s : ℝ), d = s * real.sqrt 3

def surface_area_in_terms_of_d (d : ℝ) : ℝ := 2 * d^2

-- Statement of the theorem
theorem surface_area_of_cube_in_terms_of_d (x d : ℝ) 
  (volume_eq : volume_of_cube x = d^3 / 3 * real.sqrt 3) 
  (diagonal_eq : space_diagonal d) : 
  surface_area_in_terms_of_d d = 2 * d^2 :=
sorry

end surface_area_of_cube_in_terms_of_d_l150_150246


namespace max_power_of_two_divides_product_l150_150744

-- The Lean definitions and statement for the mathematical problem
theorem max_power_of_two_divides_product : 
  ∀ (a : ℕ → ℕ) (n : ℕ) (distinct : ∀ i j, i < n → j < n → i ≠ j → a i ≠ a j),
    (∑ i in finset.range n, a i = 2021) →
    (∀ m ∈ finset.range n, 0 < a m) →
    -- Assuming the product is maximized
    (∃ M : ℕ, M = (∏ i in finset.range n, a i) ∧
    -- Prove that the largest k such that 2^k divides M is 62
    (∀ k : ℕ, (2 ^ k ∣ M) ↔ k ≤ 62)) :=
by 
  intros a n distinct sum_cond pos_cond
  -- Prove the statement assuming the conditions
  sorry

end max_power_of_two_divides_product_l150_150744


namespace convert_point_to_spherical_l150_150932

def point := (4 : ℝ, 4 * real.sqrt 3, -2 * real.sqrt 6)

def spherical_coords (pt : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := pt
  let rho := real.sqrt (x^2 + y^2 + z^2)
  let phi := real.arccos (z / rho)
  let theta := real.arctan2 y x
  (rho, theta, phi)

theorem convert_point_to_spherical :
  ∃ ρ θ φ, 
    (ρ, θ, φ) = spherical_coords point ∧
    ρ = 2 * real.sqrt 22 ∧
    θ = real.pi / 3 ∧
    φ = real.arccos (-real.sqrt 6 / real.sqrt 22) :=
by
  sorry

end convert_point_to_spherical_l150_150932


namespace train_length_proof_l150_150844

-- Defining the conditions
def speed_kmph : ℕ := 72
def platform_length : ℕ := 250  -- in meters
def time_seconds : ℕ := 26

-- Conversion factor from kmph to m/s
def kmph_to_mps (v : ℕ) : ℕ := (v * 1000) / 3600

-- The main goal: the length of the train
def train_length (speed_kmph : ℕ) (platform_length : ℕ) (time_seconds : ℕ) : ℕ :=
  let speed_mps := kmph_to_mps speed_kmph
  let total_distance := speed_mps * time_seconds
  total_distance - platform_length

theorem train_length_proof : train_length speed_kmph platform_length time_seconds = 270 := 
by 
  unfold train_length kmph_to_mps
  sorry

end train_length_proof_l150_150844


namespace cone_lateral_surface_area_l150_150069

noncomputable def lateralSurfaceAreaConical (r : ℝ) (h : ℝ) : ℝ :=
  (1 / 2) * Real.pi * (sqrt (r^2 + h^2))^2

theorem cone_lateral_surface_area :
  lateralSurfaceAreaConical 3 (sqrt (3^2 * 2)) = 18 * Real.pi := by
  sorry

end cone_lateral_surface_area_l150_150069


namespace closest_to_zero_l150_150445

theorem closest_to_zero :
  let A := 6 + 5 + 4
  let B := 6 + 5 - 4
  let C := 6 + 5 * 4
  let D := 6 - 5 * 4
  let E := 6 * 5 / 4
  min (abs A) (min (abs B) (min (abs C) (min (abs D) (abs E)))) = abs B :=
by
  sorry

end closest_to_zero_l150_150445


namespace amount_paid_to_Y_l150_150824

theorem amount_paid_to_Y (total_payment : ℝ) (percentage_X_wrt_Y : ℝ) 
  (h1 : total_payment = 580) (h2 : percentage_X_wrt_Y = 1.2) : 
  ∃ Y : ℝ, Y = 263.64 ∧ Y + (percentage_X_wrt_Y * Y) = total_payment :=
by 
  -- Define the amount paid to Y as Y
  let Y := total_payment / (1 + percentage_X_wrt_Y)
  
  -- Calculate the value Y
  have h3 : Y = 580 / 2.2, from sorry,
  
  -- Prove the value of Y is 263.64
  have h4 : Y = 263.64, from sorry,
  
  -- Prove the amounts aggregate correctly
  have h5 : Y + 1.2 * Y = 580, from sorry,

  -- Existential statement proving the requirements
  exact ⟨Y, ⟨h4, h5⟩⟩
  sorry

end amount_paid_to_Y_l150_150824


namespace ap_eq_ao_l150_150873

-- Define given conditions
variables {A B C O P : Point} -- Points in the Euclidean plane
variables (circle : Circle) -- Circle inscribed in ∠BAC
variables (centerO : circle.center = O) -- Center of the circle is O
variables (tangentP : TangentToCircle circle P) -- P is a point where the tangent intersects AB

-- Define the geometric properties and prove the required equality
theorem ap_eq_ao (h1 : circle.inscribedIn ∠BAC)
  (h2 : TangentToCircleParallelLine circle P AO AB) : distance A P = distance A O := by
  sorry

end ap_eq_ao_l150_150873


namespace cs_competition_hits_l150_150495

theorem cs_competition_hits :
  (∃ x y z : ℕ, 5 * x + 4 * y + 3 * z = 15 ∧ x + y + z ≥ 1)
  ∧ (∀ x y z : ℕ, 5 * x + 4 * y + 3 * z = 15 ∧ x + y + z ≥ 1 → (x = 1 ∧ y = 1 ∧ z = 2) ∨ (x = 0 ∧ y = 3 ∧ z = 1)) :=
by
  sorry

end cs_competition_hits_l150_150495


namespace find_all_solutions_l150_150138

def is_solution (x : ℝ) : Prop := 
  (2 / (x + 2) + 4 / (x + 8)) <= (5 / 4) ∧ x ≠ -2 ∧ x ≠ -8

theorem find_all_solutions (x : ℝ) : 
  is_solution x ↔ x ∈ set.Iio (-8) ∪ set.Icc (-8 : ℝ) (-2) :=
by sorry

end find_all_solutions_l150_150138


namespace largest_divisor_of_408_also_factor_of_310_l150_150831

def is_factor (a b : ℕ) : Prop := ∃ k : ℕ, b = k * a

def gcd (a b : ℕ) : ℕ := Nat.gcd a b

theorem largest_divisor_of_408_also_factor_of_310 :
  gcd 408 310 = 2 := by
  sorry

end largest_divisor_of_408_also_factor_of_310_l150_150831


namespace probability_closer_to_7_than_4_l150_150888

noncomputable def midpoint (a b : ℝ) : ℝ := (a + b) / 2

noncomputable def interval_closer_to_7_than_4 (x : ℝ) : Prop := x > midpoint 4 7

noncomputable def interval_probability (a b c d : ℝ) : ℝ :=
  let interval_len := d - midpoint a b
  in interval_len / (d - c)

theorem probability_closer_to_7_than_4 : interval_probability 4 7 0 8 = 0.3 := by
  sorry

end probability_closer_to_7_than_4_l150_150888


namespace initial_men_l150_150822

/-- Initial number of men M being catered for. 
Proof that the initial number of men M is equal to 760 given the conditions. -/
theorem initial_men (M : ℕ)
  (H1 : 22 * M = 20 * M)
  (H2 : 2 * (M + 3040) = M) : M = 760 := 
sorry

end initial_men_l150_150822


namespace set_intersection_l150_150750

open Set

variable (U : Set ℝ) (A B : Set ℝ)

def CU (A : Set ℝ) : Set ℝ := {x | x ≤ 0 ∨ x ≥ 2}

theorem set_intersection :
  let U := univ,
      A := { x : ℝ | 0 < x ∧ x < 2 },
      B := { x : ℝ | |x| ≤ 1 }
  in (CU A) ∩ B = { x : ℝ | -1 ≤ x ∧ x ≤ 0 } := by
  sorry

end set_intersection_l150_150750


namespace sample_size_l150_150493

theorem sample_size (num_classes : ℕ) (students_per_class : ℕ) (students_sent_per_class : ℕ)
  (h1 : num_classes = 40) (h2 : students_per_class = 50) (h3 : students_sent_per_class = 3) : 
  num_classes * students_sent_per_class = 120 :=
by
  rw [h1, h3]
  norm_num
  done

end sample_size_l150_150493


namespace component_unqualified_l150_150468

theorem component_unqualified :
  ∀ (φ : ℝ), (19.98 ≤ φ ∧ φ ≤ 20.02) → ¬(φ = 19.9) → True :=
by
  intro φ
  intro h
  intro h'
  -- skip proof
  sorry

end component_unqualified_l150_150468


namespace train_pass_time_l150_150892

/--
  This theorem proves that a train of length 110 m running at a speed of 90 km/h will pass
  a man running in the opposite direction at 9 km/h in 4 seconds.
--/
theorem train_pass_time (length_train : ℝ) (speed_train_kmh : ℝ) (speed_man_kmh : ℝ) :
  length_train = 110 ∧ speed_train_kmh = 90 ∧ speed_man_kmh = 9 →
  (length_train / ((speed_train_kmh * 1000 / 3600) + (speed_man_kmh * 1000 / 3600))) = 4 :=
by
  intro h
  cases h with h1 h2,
  cases h2 with h2 h3,
  rw [h1, h2, h3],
  norm_num,
  sorry

end train_pass_time_l150_150892


namespace rectangle_dimensions_l150_150397

theorem rectangle_dimensions (x y : ℝ) (h1 : x = 2 * y) (h2 : 2 * (x + y) = 2 * x * y) : 
  (x = 3 ∧ y = 1.5) :=
by
  sorry

end rectangle_dimensions_l150_150397


namespace unique_solution_for_2_pow_m_plus_1_eq_n_square_l150_150529

theorem unique_solution_for_2_pow_m_plus_1_eq_n_square (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  2 ^ m + 1 = n ^ 2 → (m = 3 ∧ n = 3) :=
by {
  sorry
}

end unique_solution_for_2_pow_m_plus_1_eq_n_square_l150_150529


namespace only_positive_root_of_equation_l150_150946

-- Define the function f(x) corresponding to the given problem
def f (x : ℝ) : ℝ := x^x + x^(1 - x) - x - 1

-- State the theorem that the only positive root of the equation is x = 1
theorem only_positive_root_of_equation : ∀ (x : ℝ), (x > 0) → (f x = 0 → x = 1) := 
by
sorry

end only_positive_root_of_equation_l150_150946


namespace multiples_of_15_between_35_and_200_l150_150655

theorem multiples_of_15_between_35_and_200 : 
  ∃ n : ℕ, ∀ k : ℕ, 35 < k * 15 ∧ k * 15 < 200 ↔ k = n :=
begin
  sorry,
end

end multiples_of_15_between_35_and_200_l150_150655


namespace find_y_l150_150126

def operation (a b : ℝ) : ℝ := (sqrt (3 * a + 2 * b)) ^ 2

theorem find_y : ∃ y: ℝ, operation 5 y = 16 ∧ y = 1 / 2 :=
by
  sorry

end find_y_l150_150126


namespace incorrect_connection_probability_l150_150411

theorem incorrect_connection_probability :
  let p := 0.02
  let r2 := 1 / 9
  let r3 := 8 / 81
  let probability_wrong_two_errors := 3 * p^2 * (1 - p) * r2
  let probability_wrong_three_errors := 1 * p^3 * r3
  let total_probability_correct_despite_errors := probability_wrong_two_errors + probability_wrong_three_errors
  let total_probability_incorrect := 1 - total_probability_correct_despite_errors
  ((total_probability_correct_despite_errors ≈ 0.000131) → 
  (total_probability_incorrect ≈ 1 - 0.000131)) :=
by
  sorry

end incorrect_connection_probability_l150_150411


namespace smallest_positive_integer_l150_150024

theorem smallest_positive_integer (x : ℕ) : 
  (5 * x ≡ 18 [MOD 33]) ∧ (x ≡ 4 [MOD 7]) → x = 10 := 
by 
  sorry

end smallest_positive_integer_l150_150024


namespace tail_wind_distance_l150_150886

-- Definitions based on conditions
def speed_still_air : ℝ := 262.5
def t1 : ℝ := 3
def t2 : ℝ := 4

def effective_speed_tail_wind (w : ℝ) : ℝ := speed_still_air + w
def effective_speed_against_wind (w : ℝ) : ℝ := speed_still_air - w

theorem tail_wind_distance (w : ℝ) (d : ℝ) :
  effective_speed_tail_wind w * t1 = effective_speed_against_wind w * t2 →
  d = t1 * effective_speed_tail_wind w →
  d = 900 :=
by
  sorry

end tail_wind_distance_l150_150886


namespace least_number_divisible_increased_by_seven_l150_150851

theorem least_number_divisible_increased_by_seven : 
  ∃ n : ℕ, (∀ k ∈ [24, 32, 36, 54], (n + 7) % k = 0) ∧ n = 857 := 
by
  sorry

end least_number_divisible_increased_by_seven_l150_150851


namespace find_f_zero_f_is_odd_solve_inequality_l150_150587

variable {f : ℝ → ℝ}
variable (h_incr : ∀ x y, x ≤ y → f x ≤ f y)
variable (h_add : ∀ x y, f (x + y) = f x + f y)
variable (h_f1 : f 1 = 2)

theorem find_f_zero : f 0 = 0 :=
by {
  have h := h_add 0 0,
  rw [zero_add] at h,
  linarith
}

theorem f_is_odd : ∀ x, f (-x) = -f x :=
by {
  intro x,
  have h := h_add x (-x),
  rw [add_right_neg] at h,
  linarith [find_f_zero h_incr h_add]
}

theorem solve_inequality (x : ℝ) : f x - f (3 - x) < 4 → x < 5 / 2 :=
by {
  intro h,
  have f2_is_4 : f 2 = 4 :=
    by {
      have h := h_add 1 1,
      rw h_f1 at h,
      linarith
    },
  have h1 := h_incr x (5 - x),
  replace h1 := h_incr (3 - x) (2),
  linarith
}

end find_f_zero_f_is_odd_solve_inequality_l150_150587


namespace polarBearDailyFish_l150_150537

-- Define the conditions
def polarBearDailyTrout : ℝ := 0.2
def polarBearDailySalmon : ℝ := 0.4

-- Define the statement to be proven
theorem polarBearDailyFish : polarBearDailyTrout + polarBearDailySalmon = 0.6 :=
by
  sorry

end polarBearDailyFish_l150_150537


namespace verify_statements_l150_150169

open Complex Real

/-- Given the complex number z₀ as a unit complex number and z₁ as a conjugate complex number,
    where z₀ = cos x + i * sin x and z₁ = a + b * i, their product is
    z = z₀ * z₁, and we have:
    f(x) is the real part of z and g(x) is the imaginary part of z.

    The following statements are verified:
    - A: f(x) = a * cos x - b * sin x
    - D: Given a = sqrt(3), b = -1, and g(x) = 6/5, the sine of the acute angle x is (3 * sqrt(3) + 4) / 10. -/
theorem verify_statements (x a b : ℝ) :
  let z₀ := cos x + sin x * I
  let z₁ := a + b * I
  let z := z₀ * z₁
  let f : ℝ → ℝ := λ x, a * cos x - b * sin x
  let g : ℝ → ℝ := λ x, a * sin x + b * cos x in
  (f x = a * cos x - b * sin x) ∧
  (∀ x, a = sqrt 3 ∧ b = -1 ∧ g x = 6 / 5 → sin x = (3 * sqrt 3 + 4) / 10) :=
by {
  sorry
}

end verify_statements_l150_150169


namespace part_a_part_b_part_c_part_d_l150_150159

open BigOperators

def fibonacci : ℕ → ℤ
| 0       := 0
| 1       := 1
| (n + 2) := fibonacci n + fibonacci (n + 1)

theorem part_a (m : ℕ) (h1 : 0 < m) :
  ∃ (i j : ℕ), (i < j ∧ j ≤ m * m ∧ (fibonacci i) % m = (fibonacci j) % m ∧ (fibonacci (i + 1)) % m = (fibonacci (j + 1)) % m) :=
sorry

theorem part_b (m : ℕ) (h1 : 0 < m) :
  ∃ (k : ℕ), (0 < k ∧ ∀ (n : ℕ), (fibonacci (n + k)) % m = (fibonacci n) % m) :=
sorry

def k_m (m : ℕ) : ℕ :=
  if h : ∃ k, 0 < k ∧ ∀ n, fibonacci (n + k) % m = fibonacci n % m then 
    @nat.find {k // 0 < k ∧ ∀ n, fibonacci (n + k) % m = fibonacci n % m} (classical.indefinite_description _ h )
  else 0

theorem part_c (m : ℕ) (h_pos : 0 < m) (k_m_val : k_m m ≠ 0) :
  (fibonacci (k_m m) % m = 0 ∧ fibonacci ((k_m m) + 1) % m = 1) :=
sorry

theorem part_d (m k : ℕ) (h_pos : 0 < m) :
  (∀ n, fibonacci (n + k) % m = fibonacci n % m) ↔ k % (k_m m) = 0 :=
sorry

end part_a_part_b_part_c_part_d_l150_150159


namespace dan_remaining_marbles_l150_150528

-- Define the initial number of marbles Dan has
def initial_marbles : ℕ := 64

-- Define the number of marbles Dan gave to Mary
def marbles_given : ℕ := 14

-- Define the number of remaining marbles
def remaining_marbles : ℕ := initial_marbles - marbles_given

-- State the theorem
theorem dan_remaining_marbles : remaining_marbles = 50 := by
  -- Placeholder for the proof
  sorry

end dan_remaining_marbles_l150_150528


namespace angle_bisector_segment_rel_l150_150407

variable (a b c : ℝ) -- The sides of the triangle
variable (u v : ℝ)   -- The segments into which fa divides side a
variable (fa : ℝ)    -- The length of the angle bisector

-- Statement setting up the given conditions and the proof we need
theorem angle_bisector_segment_rel : 
  (u : ℝ) = a * c / (b + c) → 
  (v : ℝ) = a * b / (b + c) → 
  (fa : ℝ) = 2 * (Real.sqrt (b * s * (s - a) * c)) / (b + c) → 
  fa^2 = b * c - u * v :=
sorry

end angle_bisector_segment_rel_l150_150407


namespace prove_triangles_may_or_may_not_be_equal_and_may_have_equal_areas_l150_150248

noncomputable def are_triangles_may_or_may_not_be_equal_and_may_have_equal_areas
  {α : Type} [MetricSpace α]
  {A B C D E F : α}
  (hAB_DE : dist A B = dist D E)
  (hAC_DF : dist A C = dist D F)
  (hAngle_BAC_EDF : ∠BAC = ∠EDF) : Prop :=
  (∃ (congruent : Prop), congruent ∨ ¬congruent) ∧
  (∃ (equal_areas : Prop), equal_areas)

theorem prove_triangles_may_or_may_not_be_equal_and_may_have_equal_areas
  {α : Type} [MetricSpace α]
  (A B C D E F : α)
  (hAB_DE : dist A B = dist D E)
  (hAC_DF : dist A C = dist D F)
  (hAngle_BAC_EDF : ∠BAC = ∠EDF) : are_triangles_may_or_may_not_be_equal_and_may_have_equal_areas hAB_DE hAC_DF hAngle_BAC_EDF :=
sorry

end prove_triangles_may_or_may_not_be_equal_and_may_have_equal_areas_l150_150248


namespace green_pieces_count_l150_150285

variable (G : ℕ)

theorem green_pieces_count :
  (G = Nat.floor (0.25 * (20 + G + 85))) → G = 35 :=
by sorry

end green_pieces_count_l150_150285


namespace line_through_two_points_l150_150377

theorem line_through_two_points (x y : ℝ) (hA : (x, y) = (3, 0)) (hB : (x, y) = (0, 2)) :
  2 * x + 3 * y - 6 = 0 :=
sorry 

end line_through_two_points_l150_150377


namespace cost_price_of_toy_l150_150072

theorem cost_price_of_toy
  (sold_price : ℕ)
  (toys_sold : ℕ)
  (gain_toys: ℕ)
  (total_price: ℕ):
  (toys_sold + gain_toys) * (total_price / (toys_sold + gain_toys)) = sold_price  →
  (toys_sold = 18) →
  (sold_price = 16800) →
  (gain_toys = 3) →
  (total_price / (toys_sold + gain_toys) = 800) :=
by
  intros h_eq h_sold h_price h_gain
  rw [h_sold, h_price, h_gain] at h_eq
  exact h_eq
  sorry

end cost_price_of_toy_l150_150072


namespace spherical_coords_neg_x_l150_150887

open Real

-- Define the given conditions
def given_spherical_coords : ℝ × ℝ × ℝ :=
  (3, 5 * π / 6, π / 4)

-- Extract the spherical coordinates (ρ, θ, φ)
def (ρ, θ, φ) := given_spherical_coords

-- Define the rectangular coordinates from the given spherical coordinates
def x : ℝ := ρ * sin φ * cos θ
def y : ℝ := ρ * sin φ * sin θ
def z : ℝ := ρ * cos φ

-- Now define the new rectangular coordinates (negating x)
def new_x := -x
def new_y := y
def new_z := z

-- The spherical coordinates we want to prove
def expected_spherical_coords : ℝ × ℝ × ℝ :=
  (3, π / 6, π / 4)

-- The Lean theorem to prove the correctness of the spherical coordinates
theorem spherical_coords_neg_x :
  given_spherical_coords = (ρ, θ, φ) →
  expected_spherical_coords = (ρ, π - θ, φ) :=
by
  intros h
  rw h
  -- skipping the actual proof with sorry
  sorry

end spherical_coords_neg_x_l150_150887


namespace no_valid_x_in_choices_l150_150663

noncomputable def find_x : Option ℝ :=
  let choices : List ℝ := [0.2, 0.3, 0.4, 0.5]
  choices.find? (λ x => 16^(x + 1) = 288 + 16^x)

theorem no_valid_x_in_choices :
  find_x = none := 
by
  sorry

end no_valid_x_in_choices_l150_150663


namespace sum_first_19_terms_l150_150591

variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)
variable (d : ℝ)
variable (a₀ a₃ a₁₇ a₁₀ : ℝ)

noncomputable def is_arithmetic_sequence (a : ℕ → ℝ) :=
  ∃ a₀ d, ∀ n, a n = a₀ + n * d

noncomputable def sum_first_n_terms (S : ℕ → ℝ) (a : ℕ → ℝ) :=
  ∀ n, S n = (n / 2) * (2 * a 1 + (n - 1) * (a 2 - a 1))

theorem sum_first_19_terms (h1 : is_arithmetic_sequence a)
                          (h2 : a 3 + a 17 = 10)
                          (h3 : sum_first_n_terms S a) :
  S 19 = 95 :=
sorry

end sum_first_19_terms_l150_150591


namespace jane_waiting_time_l150_150710

-- Given conditions as constants for readability
def base_coat_drying_time := 2
def first_color_coat_drying_time := 3
def second_color_coat_drying_time := 3
def top_coat_drying_time := 5

-- Total drying time calculation
def total_drying_time := base_coat_drying_time 
                       + first_color_coat_drying_time 
                       + second_color_coat_drying_time 
                       + top_coat_drying_time

-- The theorem to prove
theorem jane_waiting_time : total_drying_time = 13 := 
by
  sorry

end jane_waiting_time_l150_150710


namespace A_B_can_make_C_lose_l150_150008

-- Define the initial conditions
def bowls := ℕ
structure Game (X Y Z : bowls) :=
  (num_pieces : bowls)
  (max_pieces : bowls)

-- Define the players and their moves
inductive Player := | A | B | C

-- Define the rules
def move (p : Player) (g : Game X Y Z) : Game X Y Z :=
  match p with
  | Player.A => { g with num_pieces := g.num_pieces + 1 }
  | Player.B => { g with num_pieces := g.num_pieces + 1 }
  | Player.C => { g with num_pieces := g.num_pieces + 1 }
  | _ => g

theorem A_B_can_make_C_lose
  (X Y Z : bowls)
  (game : Game X Y Z)
  (A_moves : ∀ m : Game X Y Z, move Player.A m = m)
  (B_moves : ∀ m : Game X Y Z, move Player.B m = m)
  (C_moves : ∀ m : Game X Y Z, move Player.C m = m) 
  (max_pieces := 1999) :
  ∃ game, game.num_pieces = max_pieces → Player.C = lose :=
sorry

end A_B_can_make_C_lose_l150_150008


namespace balls_in_boxes_l150_150658

/-- Prove that the number of ways to put 6 distinguishable balls in 3 distinguishable boxes is 729 (which is 3^6). -/
theorem balls_in_boxes : (3 ^ 6) = 729 :=
by 
  sorry

end balls_in_boxes_l150_150658


namespace min_f_l150_150555

def f (x : ℝ) : ℝ :=
  ∑ k in finset.range(52), (x - 2 * k)^2

theorem min_f : ∃ x : ℝ, ∀ y : ℝ, f x ≤ f y ∧ f x = 46852 :=
sorry

end min_f_l150_150555


namespace magic_square_solution_l150_150328

theorem magic_square_solution (d e k f g h x y : ℤ)
  (h1 : x + 4 + f = 87 + d + f)
  (h2 : x + d + h = 87 + e + h)
  (h3 : x + y + 87 = 4 + d + e)
  (h4 : f + g + h = x + y + 87)
  (h5 : d = x - 83)
  (h6 : e = 2 * x - 170)
  (h7 : y = 3 * x - 274)
  (h8 : f = g)
  (h9 : g = h) :
  x = 62 ∧ y = -88 :=
by
  sorry

end magic_square_solution_l150_150328


namespace chris_ate_21_cookies_l150_150923

theorem chris_ate_21_cookies (total_cookies : ℕ) (fraction_given fraction_eaten : ℚ)
  (h1 : total_cookies = 84)
  (h2 : fraction_given = 1 / 3)
  (h3 : fraction_eaten = 3 / 4)
  : (fraction_eaten * (fraction_given * total_cookies : ℚ) = 21) := 
by {
  rw [h1, h2, h3],
  norm_num,
  sorry
}

end chris_ate_21_cookies_l150_150923


namespace lara_bag_total_chips_l150_150420

theorem lara_bag_total_chips (C : ℕ)
  (h1 : ∃ (b : ℕ), b = C / 6)
  (h2 : 34 + 16 + C / 6 = C) :
  C = 60 := by
  sorry

end lara_bag_total_chips_l150_150420


namespace smallest_sum_of_xy_l150_150608

theorem smallest_sum_of_xy (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hxy : x ≠ y)
  (hcond : (1 : ℚ) / x + (1 : ℚ) / y = 1 / 10) : x + y = 45 :=
sorry

end smallest_sum_of_xy_l150_150608


namespace maximum_squares_crossed_l150_150473

noncomputable def circle_max_squares (r : ℝ) (grid_size : ℝ) : ℕ :=
  if r = 100 ∧ grid_size = 1 then 800 else sorry

theorem maximum_squares_crossed (r : ℝ) (grid_size : ℝ) 
  (h_r : r = 100) (h_grid : grid_size = 1) : 
  circle_max_squares r grid_size = 800 := by
  simp [circle_max_squares, h_r, h_grid]
  sorry

end maximum_squares_crossed_l150_150473


namespace polynomial_expansion_evaluation_l150_150243

theorem polynomial_expansion_evaluation :
  let f := (3* X^3 - 5* X^2 + 4* X - 6) * (7 - 2* X),
  let a := -(6 : ℝ),
  let b := (31 : ℝ),
  let c := -(43 : ℝ),
  let d := (40 : ℝ),
  let e := -(42 : ℝ) in
  ∑ i in (finRange 5), [16, 8, 4, 2, 1][i] * (monomial i : ℝ →₀ ℝ)([a, b, c, d, e][i]) = 42 := by
sorry

end polynomial_expansion_evaluation_l150_150243


namespace binom_26_6_equality_l150_150616

theorem binom_26_6_equality : (binom 24 4 = 10626) ∧ (binom 24 5 = 42504) ∧ (binom 24 6 = 53130) → binom 26 6 = 148764 := by
  sorry

end binom_26_6_equality_l150_150616


namespace inequality_proof_l150_150578

-- Define the main theorem with the given conditions and proof goal
theorem inequality_proof (x : ℝ) (n : ℕ) (h1 : 0 < x) (h2 : x < π / 2) (hn : 0 < n) :
  ((1 / (sin x)^(2 * n) - 1) * (1 / (cos x)^(2 * n) - 1) >= (2^n - 1)^2) :=
by
  -- The proof would be inserted here
  sorry

end inequality_proof_l150_150578


namespace alice_investment_ratio_l150_150902

-- Definitions of initial amounts and investments
def initial_investment : ℝ := 2000

def alice_final_amount (x : ℝ) : ℝ := initial_investment * x

def bob_final_amount : ℝ := initial_investment * 6 -- Because Bob makes five times more money

-- Condition: Bob's final amount is $8000 more than Alice's final amount
def bob_has_more (x : ℝ) : Prop := bob_final_amount = alice_final_amount(x) + 8000

-- The ratio of Alice's final amount to her initial investment
def ratio := alice_final_amount(2) / initial_investment

-- Statement to prove
theorem alice_investment_ratio (x : ℝ) (hx : bob_has_more x) : ratio = 2 := by
  sorry

end alice_investment_ratio_l150_150902


namespace locus_of_vertex_C_l150_150830

-- Definitions for vertices and rays
variable (ABC : Type) [unit_triangle : EquilateralTriangle ABC 1] -- regular triangle with unit side length
variable (A B C O X Y : Points) -- Points in the space
variable [ray_OX : Ray O X] [ray_OY : Ray O Y] -- Definitions for the rays
variable [condition1 : A ∈ ray_OX] -- Vertex A lies on OX
variable [condition2 : B ∈ ray_OY] -- Vertex B lies on OY
variable [condition3 : ¬ same_side O (line A B) C] -- Line AB separates C from O

-- Theorem statement
theorem locus_of_vertex_C :
  locus C = segment C_1 C_2 :=  -- Locus of C is the segment C_1 C_2
sorry

end locus_of_vertex_C_l150_150830


namespace find_reading_l150_150041

variable (a_1 a_2 a_3 a_4 : ℝ) (x : ℝ)
variable (h1 : a_1 = 2) (h2 : a_2 = 2.1) (h3 : a_3 = 2) (h4 : a_4 = 2.2)
variable (mean : (a_1 + a_2 + a_3 + a_4 + x) / 5 = 2)

theorem find_reading : x = 1.7 :=
by
  sorry

end find_reading_l150_150041


namespace value_of_MN_l150_150665

theorem value_of_MN (M N : ℝ) (log : ℝ → ℝ → ℝ)
    (h1 : log (M ^ 2) N = log N (M ^ 2))
    (h2 : M ≠ N)
    (h3 : M * N > 0)
    (h4 : M ≠ 1)
    (h5 : N ≠ 1) :
    M * N = N^(1/2) :=
  sorry

end value_of_MN_l150_150665


namespace square_area_l150_150498

theorem square_area (w : ℝ) (A : ℝ) (x y : ℝ) (area_eq : ∀ r : ℝ, r = A) (hw : w = 5) :
  let width := 5 + 2 * (15 / 2) in width ^ 2 = 400 :=
by
  sorry

end square_area_l150_150498


namespace vector_relation_l150_150733

variables {V : Type*} [inner_product_space ℝ V]

/-- If |a + b| = |a| - |b|, then there exists a real number λ such that b = λa. -/
theorem vector_relation (a b : V) (h₁ : ∥a + b∥ = ∥a∥ - ∥b∥) :
  ∃ λ : ℝ, b = λ • a :=
sorry

end vector_relation_l150_150733


namespace bikers_meet_again_l150_150011

theorem bikers_meet_again
    (t1 t2 t3 : ℕ)
    (t1_def : t1 = 12)
    (t2_def : t2 = 18)
    (t3_def : t3 = 24) :
    lcm t1 (lcm t2 t3) = 72 :=
by {
  rw [t1_def, t2_def, t3_def],
  exact sorry
}

end bikers_meet_again_l150_150011


namespace intersection_product_l150_150641

noncomputable def parametric_line_equation (t : ℝ) : ℝ × ℝ :=
  (1 + 2018 * t, real.sqrt 3 + 2018 * t)

noncomputable def polar_curve_equation (θ ρ : ℝ) : Prop :=
  ρ ^ 2 = 4 * ρ * real.cos θ + 2 * real.sqrt 3 * ρ * real.sin θ - 4

noncomputable def cartesian_line_equation (x y : ℝ) : Prop :=
  y = real.sqrt 3 * x

noncomputable def cartesian_curve_equation (x y : ℝ) : Prop :=
  (x - 2) ^ 2 + (y - real.sqrt 3) ^ 2 = 3

theorem intersection_product :
  (∀ t : ℝ, parametric_line_equation t ∈ {p : ℝ × ℝ | cartesian_line_equation p.1 p.2}) →
  (∀ θ : ℝ, ∀ ρ : ℝ, polar_curve_equation θ ρ → cartesian_curve_equation (ρ * real.cos θ) (ρ * real.sin θ)) →
  let ρ₁ ρ₂ := roots of ρ^2 - 5*ρ + 4 in
  |ρ₁ * ρ₂| = 4 :=
by
  sorry

end intersection_product_l150_150641


namespace flight_time_sum_l150_150284

noncomputable def flight_info : Nat × Nat :=
  ((11 + 60) * 60 + 7 - (9 * 60 + 15) * 60 - 45) div 3600, 
  ((11 + 60) * 60 + 7 - (9 * 60 + 15) * 60 - 45) % 3600 / 60

theorem flight_time_sum : (flight_info.1 + flight_info.2) = 12 := by
  sorry

end flight_time_sum_l150_150284


namespace component_unqualified_l150_150467

theorem component_unqualified :
  ∀ (φ : ℝ), (19.98 ≤ φ ∧ φ ≤ 20.02) → ¬(φ = 19.9) → True :=
by
  intro φ
  intro h
  intro h'
  -- skip proof
  sorry

end component_unqualified_l150_150467


namespace find_r_floor_plus_r_l150_150947

theorem find_r_floor_plus_r :
  ∃ r : ℝ, floor r + r = 18.75 ∧ r = 9.75 :=
by 
  sorry

end find_r_floor_plus_r_l150_150947


namespace GPFQ_is_rectangle_max_area_of_GPFQ_l150_150378

theorem GPFQ_is_rectangle (p l : ℝ) (hp : p > 0) (hl : l > 0) :
  ∃ (G P Q : ℝ × ℝ), 
  let F := (p / 2, 0),
      chord : set (ℝ × ℝ) := {z | (z.2)^2 = 2 * p * z.1 ∧ z.1 * l = 1}, 
      FG := (0, directrix_intersection p),
      P := (0, intersection_y_axis_G_from_B G FG),
      Q := (0, intersection_y_axis_G_from_A G FG)
  in
  is_rectangle ((0, directrix_intersection p), F, P, Q) :=
begin
  sorry
end

theorem max_area_of_GPFQ (p l : ℝ) (hp : p > 0) (hl : l > 0) :
  let F := (p / 2, 0),
      max_S := l^2 / 8
  in
  rect_area (F, max_S) :=
begin
  sorry
end

end GPFQ_is_rectangle_max_area_of_GPFQ_l150_150378


namespace max_value_of_abs_z_l150_150669

-- Given conditions
def condition (z : ℂ) : Prop := |z + 3 + 4 * complex.I| ≤ 2

-- Prove that the maximum value of |z| is 7 given the conditions
theorem max_value_of_abs_z (z : ℂ) (h : condition z) : |z| ≤ 7 :=
  sorry

end max_value_of_abs_z_l150_150669


namespace adriatic_equals_tyrrhenian_l150_150522

-- Define properties of Adriatic sequences
def is_adriatic_sequence (s : List ℕ) : Prop :=
  s.head = 1 ∧ ∀ i, i < s.length - 1 → s.get i.succ ≥ 2 * s.get i

-- Define properties of Tyrrhenian sequences
def is_tyrrhenian_sequence (s : List ℕ) (n : ℕ) : Prop :=
  s.last = n ∧ ∀ i, i < s.length - 1 → s.get i.succ > s.get_prefix i.succ.sum + 1

-- Define the set of all Adriatic sequences with elements from {1, 2, ..., n}
def adriatic_sequences (n : ℕ) : Set (List ℕ) :=
  { s | is_adriatic_sequence s ∧ s.all (λ x, 1 ≤ x ∧ x ≤ n) }

-- Define the set of all Tyrrhenian sequences with elements from {1, 2, ..., n}
def tyrrhenian_sequences (n : ℕ) : Set (List ℕ) :=
  { s | is_tyrrhenian_sequence s n ∧ s.all (λ x, 1 ≤ x ∧ x ≤ n) }

-- Prove the number of adriatic_sequences is equal to tyrrhenian_sequences
theorem adriatic_equals_tyrrhenian (n : ℕ) :
  Set.card (adriatic_sequences n) = Set.card (tyrrhenian_sequences n) :=
by sorry

end adriatic_equals_tyrrhenian_l150_150522


namespace rex_cards_left_l150_150337

-- Definitions
def nicole_cards : ℕ := 400
def cindy_cards : ℕ := 2 * nicole_cards
def combined_total : ℕ := nicole_cards + cindy_cards
def rex_cards : ℕ := combined_total / 2
def people_count : ℕ := 4
def cards_per_person : ℕ := rex_cards / people_count

-- Proof statement
theorem rex_cards_left : cards_per_person = 150 := by
  sorry

end rex_cards_left_l150_150337


namespace largest_power_of_7_dividing_product_of_first_50_square_numbers_l150_150288

theorem largest_power_of_7_dividing_product_of_first_50_square_numbers : 
  let P := (Finset.range 50).prod (λ i, ((i + 1) * (i + 1))) in
  ∃ k : ℕ, 7^k ∣ P ∧ ∀ k' : ℕ, 7^k' ∣ P → k' ≤ 16 :=
begin
  let P := (Finset.range 50).prod (λ i, ((i + 1) * (i + 1))),
  use 16,
  sorry
end

end largest_power_of_7_dividing_product_of_first_50_square_numbers_l150_150288


namespace find_difference_l150_150205

-- Define the vectors a and b
def vec_a : ℝ × ℝ := (1, 2)
def vec_b : ℝ × ℝ := (-2, 1)

-- Define the target vector
def target_vec : ℝ × ℝ := (3, -4)

-- Given conditions
variables (m n : ℝ)
hypothesis h : m • vec_a - n • vec_b = target_vec

-- The goal
theorem find_difference : m - n = -3 :=
by {
  sorry
}

end find_difference_l150_150205


namespace median_interval_l150_150122

-- Define the total number of students
def total_students : ℕ := 100

-- Define the frequency of students in each score interval
def frequency_65_to_69 : ℕ := 30
def frequency_60_to_64 : ℕ := 25
def frequency_55_to_59 : ℕ := 20
def frequency_50_to_54 : ℕ := 15
def frequency_45_to_49 : ℕ := 10

-- Sum of frequencies (should equal total_students)
def total_frequency: ℕ :=
  frequency_65_to_69 + frequency_60_to_64 + 
  frequency_55_to_59 + frequency_50_to_54 + frequency_45_to_49

-- Median position
def median_position (n : ℕ) : ℕ := (n + 1) / 2

-- Theorem stating the median interval
theorem median_interval :
  median_position total_students > frequency_65_to_69 ∧ median_position(total_students) <= frequency_65_to_69 + frequency_60_to_64 :=
  sorry

end median_interval_l150_150122


namespace distance_flash_runs_l150_150092

variables {v x y : ℝ}
variables (h₀ : 0.5 < x)

theorem distance_flash_runs (h₁ : v > 0) (h₂ : y ≥ 0) :
  let flash_distance := (4 * x * y + 2 * x * v) / (2 * x - 1)
  in flash_distance = (4 * x * y + 2 * x * v) / (2 * x - 1) :=
begin
  -- skipped proof
  sorry
end

end distance_flash_runs_l150_150092


namespace velocity_upper_end_l150_150097

variable (a l τ : ℝ)

-- Assuming the given conditions
def x := a * τ^2 / 2
def v1 := a * τ
def sin_alpha := a * τ^2 / (2 * l)
def cos_alpha := sqrt (4 * l^2 - a^2 * τ^4) / (2 * l)

-- Define the mathematical question we need to prove
theorem velocity_upper_end (h1 : v1 * sin_alpha = (a * τ) * (a * τ^2 / (2 * l)))
  : ∃ v2, v2 = a^2 * τ^3 / sqrt (4 * l^2 - a^2 * τ^4) :=
by
  use a^2 * τ^3 / sqrt (4 * l^2 - a^2 * τ^4)
  sorry

end velocity_upper_end_l150_150097


namespace not_all_inequalities_hold_l150_150612

theorem not_all_inequalities_hold (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  ¬ ((a + b < c + d) ∧ ((a + b) * (c + d) < a * b + c * d) ∧ ((a + b) * c * d < a * b * (c + d))) :=
by
  sorry

end not_all_inequalities_hold_l150_150612


namespace rhombus_area_in_square_l150_150826

theorem rhombus_area_in_square :
  ∀ (s : ℝ) (d1 d2 : ℝ), 
  s = 2 ∧ d1 = 2 ∧ d2 = 2 →
  (1 / 2) * d1 * d2 = 2 :=
by
  intros s d1 d2 h
  cases h with h_side h_diag
  cases h_diag with h_d1 h_d2
  rw [h_d1, h_d2]
  norm_num
  sorry

end rhombus_area_in_square_l150_150826


namespace moles_of_water_formed_l150_150558

theorem moles_of_water_formed 
  (HCl_moles : ℕ) (CaCO3_moles : ℕ) :
  (HCl_moles = 4) → 
  (CaCO3_moles = 2) →
  (∀ (n : ℕ), (n * 2 = HCl_moles) → (CaCO3_moles = n) → True) →
  (∃ (H2O_moles : ℕ), H2O_moles = 2) :=
by 
  intros h1 h2 h3
  use 2
  sorry

end moles_of_water_formed_l150_150558


namespace rihanna_initial_money_l150_150775

theorem rihanna_initial_money : 
  ∃ (initial_money : ℕ), 
  let mango_cost := 3 in
  let juice_cost := 3 in
  let mangoes_bought := 6 in
  let juice_bought := 6 in
  let total_spent := mangoes_bought * mango_cost + juice_bought * juice_cost in
  let money_left := 14 in
  initial_money = total_spent + money_left :=
sorry

end rihanna_initial_money_l150_150775


namespace range_of_set_l150_150084

theorem range_of_set (a b c : ℕ) (h1 : (a + b + c) / 3 = 6) (h2 : b = 6) (h3 : a = 2) : max a (max b c) - min a (min b c) = 8 :=
by
  sorry

end range_of_set_l150_150084


namespace angle_between_hands_at_3_15_l150_150433

-- Definitions based on conditions
def minuteHandAngleAt_3_15 : ℝ := 90 -- The position of the minute hand at 3:15 is 90 degrees.

def hourHandSpeed : ℝ := 0.5 -- The hour hand moves at 0.5 degrees per minute.

def hourHandAngleAt_3_15 : ℝ := 3 * 30 + 15 * hourHandSpeed
-- The hour hand starts at 3 o'clock (90 degrees) and moves 0.5 degrees per minute.

-- Statement to prove
theorem angle_between_hands_at_3_15 : abs (minuteHandAngleAt_3_15 - hourHandAngleAt_3_15) = 82.5 :=
by
  sorry

end angle_between_hands_at_3_15_l150_150433


namespace sequence_values_count_l150_150198

theorem sequence_values_count :
  let a_n (n : ℕ) := 1 - (1 / (n + 1 : ℝ))
  (values : List ℝ) : List.countp (λ x, ∃ n, a_n n = x) values = 2 :=
by
  let a_n (n : ℕ) := 1 - (1 / (n + 1 : ℝ))
  let values := [0.98, 0.96, 0.94]
  sorry

end sequence_values_count_l150_150198


namespace pairs_intersect_within_circle_l150_150273

theorem pairs_intersect_within_circle :
  ∃ pairs : list (point × point), 
  ∀ (A B C D : point), 
  (A, B) ∈ pairs → (C, D) ∈ pairs → 
  intersection (line_through A B) (line_through C D) ∈ circle → (
  ∀ (i j : ℕ) (hi : i < pairs.length) (hj : j < pairs.length) 
  (pi pj : point × point) 
  (hpi : pi = pairs.nth_le i hi) 
  (hpj : pj = pairs.nth_le j hj),
  pi.1 ≠ pj.1 → pi.2 = pj.2 → 
  intersection (line_through (pi.1) (pi.2)) 
  (line_through  (pj.1) (pj.2))) :=
sorry

end pairs_intersect_within_circle_l150_150273


namespace candy_weight_probability_l150_150875

open Probability
open MeasureTheory 

theorem candy_weight_probability (X : MeasureTheory.Measure ξ) [isNormal_var : IsNormal X 500 σ] (p : ℝ) 
  (condition : P (|X - 500| > 5) = p) : 
  P (495 ≤ X ∧ X ≤ 500) = (1 - p) / 2 := 
sorry

end candy_weight_probability_l150_150875


namespace num_divisible_by_2_3_5_7_under_500_l150_150221

-- Let the LCM of 2, 3, 5, and 7 be computed
def LCM_2_3_5_7 : ℕ := Nat.lcm (Nat.lcm 2 3) (Nat.lcm 5 7)

def num_multiples_less_than (n k : ℕ) : ℕ :=
  n / k

theorem num_divisible_by_2_3_5_7_under_500 : num_multiples_less_than 500 LCM_2_3_5_7 = 2 := by
  sorry

end num_divisible_by_2_3_5_7_under_500_l150_150221


namespace factorize_expression_equilateral_triangle_of_sides_two_p_eq_m_plus_n_l150_150771

-- Problem 1: Factorize x^2 - y^2 + 2x - 2y
theorem factorize_expression (x y : ℝ) : x^2 - y^2 + 2 * x - 2 * y = (x - y) * (x + y + 2) := 
by sorry

-- Problem 2: Determine the shape of a triangle given a^2 + c^2 - 2b(a - b + c) = 0
theorem equilateral_triangle_of_sides (a b c : ℝ) (h : a^2 + c^2 - 2 * b * (a - b + c) = 0) : a = b ∧ b = c :=
by sorry

-- Problem 3: Prove that 2p = m + n given (1/4)(m - n)^2 = (p - n)(m - p)
theorem two_p_eq_m_plus_n (m n p : ℝ) (h : (1/4) * (m - n)^2 = (p - n) * (m - p)) : 2 * p = m + n := 
by sorry

end factorize_expression_equilateral_triangle_of_sides_two_p_eq_m_plus_n_l150_150771


namespace pyramid_coloring_l150_150130

-- Definitions based on the conditions:
def colors : ℕ := 5
def vertices := { P, A, B, C, D }

def connected (v1 v2 : vertices) : Prop :=
  (v1 = P ∧ v2 ∈ {A, B, C, D}) ∨
  (v1 ∈ {A, B, C, D} ∧ v2 ∈ {A, B, C, D} ∧ (v1 ≠ v2))

-- The main theorem we want to prove:
theorem pyramid_coloring :
  ∃ (coloring : vertices → fin colors),
    (∀ v1 v2, connected v1 v2 → coloring v1 ≠ coloring v2) ∧
    (finset.univ.pi (λ _, finset.univ.biUnion (λ v, (finset.univ.filter (λ c, coloring v = c)))).card = 420) :=
sorry

end pyramid_coloring_l150_150130


namespace smallest_five_digit_in_pascals_triangle_l150_150437

/-- In Pascal's triangle, the smallest five-digit number is 10000. -/
theorem smallest_five_digit_in_pascals_triangle : 
  ∃ (n k : ℕ), (10000 = Nat.choose n k) ∧ (∀ m l : ℕ, Nat.choose m l < 10000) → (n > m) := 
sorry

end smallest_five_digit_in_pascals_triangle_l150_150437


namespace odd_ceil_factorial_div_l150_150290

noncomputable def is_prime (p : ℕ) := p.prime

def ceil_div (a b : ℕ) : ℕ := (a + b - 1) / b

def is_odd (n : ℕ) : Prop := n % 2 = 1

theorem odd_ceil_factorial_div (n : ℕ) (hp: 6 < n)
  (h_prime: is_prime (n + 1)) :
  is_odd (ceil_div ((n - 1)!) (n * (n + 1))) :=
sorry

end odd_ceil_factorial_div_l150_150290


namespace basketball_game_first_half_points_l150_150254

noncomputable def total_points_first_half
  (eagles_points : ℕ → ℕ) (lions_points : ℕ → ℕ) (common_ratio : ℕ) (common_difference : ℕ) : ℕ :=
  eagles_points 0 + eagles_points 1 + lions_points 0 + lions_points 1

theorem basketball_game_first_half_points 
  (eagles_points lions_points : ℕ → ℕ)
  (common_ratio : ℕ) (common_difference : ℕ)
  (h1 : eagles_points 0 = lions_points 0)
  (h2 : ∀ n, eagles_points (n + 1) = common_ratio * eagles_points n)
  (h3 : ∀ n, lions_points (n + 1) = lions_points n + common_difference)
  (h4 : eagles_points 0 + eagles_points 1 + eagles_points 2 + eagles_points 3 =
        lions_points 0 + lions_points 1 + lions_points 2 + lions_points 3 + 3)
  (h5 : eagles_points 0 + eagles_points 1 + eagles_points 2 + eagles_points 3 ≤ 120)
  (h6 : lions_points 0 + lions_points 1 + lions_points 2 + lions_points 3 ≤ 120) :
  total_points_first_half eagles_points lions_points common_ratio common_difference = 15 :=
sorry

end basketball_game_first_half_points_l150_150254


namespace smallest_x_y_sum_l150_150598

theorem smallest_x_y_sum (x y : ℕ) (h1 : x ≠ y) (h2 : x > 0) (h3 : y > 0) (h4 : 1 / (x : ℚ) + 1 / (y : ℚ) = 1 / 10) : x + y = 45 :=
by
  sorry

end smallest_x_y_sum_l150_150598


namespace opposite_of_neg_abs_opposite_of_neg_abs_correct_l150_150398

theorem opposite_of_neg_abs (x : ℚ) (hx : |x| = 2 / 5) : -|x| = - (2 / 5) := sorry

theorem opposite_of_neg_abs_correct (x : ℚ) (hx : |x| = 2 / 5) : - -|x| = 2 / 5 := by
  rw [opposite_of_neg_abs x hx]
  simp

end opposite_of_neg_abs_opposite_of_neg_abs_correct_l150_150398


namespace factorization_problem_I_factorization_problem_II_l150_150541

-- Proving the factorization of the first expression
theorem factorization_problem_I (m x : ℝ) : mx^2 - 2m^2 * x + m^3 = m * (x - m)^2 :=
by sorry

-- Proving the factorization of the second expression
theorem factorization_problem_II (m n : ℝ) : 8m^2 * n + 2mn = 2mn * (4m + 1) :=
by sorry

end factorization_problem_I_factorization_problem_II_l150_150541


namespace div_by_9_digit_B_l150_150382

theorem div_by_9_digit_B (B : ℕ) (h : (4 + B + B + 2) % 9 = 0) : B = 6 :=
by sorry

end div_by_9_digit_B_l150_150382


namespace log_255_is_approx_l150_150111

theorem log_255_is_approx:
  log 10 255 = 2.4065 := 
by
  sorry

end log_255_is_approx_l150_150111


namespace eval_expression_l150_150133

theorem eval_expression : (3 + Real.sqrt 3 + (1 / (3 + Real.sqrt 3)) + (1 / (Real.sqrt 3 - 3)) = 3 + (2 * Real.sqrt 3 / 3)) :=
by sorry

end eval_expression_l150_150133


namespace smallest_x_plus_y_l150_150953

theorem smallest_x_plus_y (x y : ℕ) (h1 : x ≥ 1) (h2 : y ≥ 1) (h3 : x^2 - 29 * y^2 = 1) : x + y = 11621 := 
sorry

end smallest_x_plus_y_l150_150953


namespace intersection_a_eq_1_range_of_a_l150_150170

-- Define the sets A and B
def setA (a : ℝ) : set ℝ := {x | a - 1 < x ∧ x < a + 1}
def setB : set ℝ := {x | 1 < x ∧ x < 5}

-- 1st problem: Prove the intersection for a = 1
theorem intersection_a_eq_1 :
  setA 1 ∩ setB = {x | 1 < x ∧ x < 2} :=
sorry

-- 2nd problem: Prove the range of a when A ⊆ A ∩ B
theorem range_of_a (a : ℝ) (h : setA a ⊆ setA a ∩ setB) :
  2 ≤ a ∧ a ≤ 4 :=
sorry

end intersection_a_eq_1_range_of_a_l150_150170


namespace area_of_region_l150_150547

theorem area_of_region :
  let circle_radius := real.sqrt (15 / (2 * real.pi)) in
  let circle_eq := (λ x y : ℝ, 2 * real.pi * (x^2 + y^2) ≤ 15) in
  let inequality1 := (λ x y : ℝ, 2 * real.pi * (x^2 + y^2) ≤ 15) in
  let inequality2 := (λ x y : ℝ, x^4 - y^4 ≤ x * y - x^3 * y^3) in
  ∀ (condition1 : ∀ x y : ℝ, inequality1 x y),
  ∀ (condition2 : ∀ x y : ℝ, inequality2 x y),
  let total_area := (real.pi * (circle_radius ^ 2)) / 2 in
  total_area = 3.75 :=
sorry

end area_of_region_l150_150547


namespace circumscribed_quadrilateral_identity_l150_150970

variables 
  (α β γ θ : ℝ)
  (h_angle_sum : α + β + γ + θ = 180)
  (OA OB OC OD AB BC CD DA : ℝ)
  (h_OA : OA = 1 / Real.sin α)
  (h_OB : OB = 1 / Real.sin β)
  (h_OC : OC = 1 / Real.sin γ)
  (h_OD : OD = 1 / Real.sin θ)
  (h_AB : AB = Real.sin (α + β) / (Real.sin α * Real.sin β))
  (h_BC : BC = Real.sin (β + γ) / (Real.sin β * Real.sin γ))
  (h_CD : CD = Real.sin (γ + θ) / (Real.sin γ * Real.sin θ))
  (h_DA : DA = Real.sin (θ + α) / (Real.sin θ * Real.sin α))

theorem circumscribed_quadrilateral_identity :
  OA * OC + OB * OD = Real.sqrt (AB * BC * CD * DA) := 
sorry

end circumscribed_quadrilateral_identity_l150_150970


namespace median_of_combined_seq_is_1100_l150_150121

/-- Define the sequence A of the first 1000 odd numbers. -/
def seq_A : List ℕ := List.range 1000 |>.map (λ n => 2 * n + 1)

/-- Define the sequence B of the squares of the first 100 integers. -/
def seq_B : List ℕ := List.range 100 |>.map (λ n => (n + 1) ^ 2)

/-- Combined sorted sequence of A and B. -/
def combined_seq : List ℕ := (seq_A ++ seq_B).qsort (≤)

/-- Median of the combined sequence -/
def median_combined : ℕ := (combined_seq.get! 549 + combined_seq.get! 550) / 2

/-- Proof that the median is 1100. -/
theorem median_of_combined_seq_is_1100 : median_combined = 1100 := by
  sorry

end median_of_combined_seq_is_1100_l150_150121


namespace volume_of_parallelepiped_with_sphere_and_extension_l150_150928

-- Define the conditions
def dimensions := (5, 6, 7)
def sphere_radius := 2
def extension := 1

-- Define the volume of the set of points under consideration
def volume_of_interest := (424 + 22 * Real.pi / 3)

-- Prove that the volume is equal to the given result
theorem volume_of_parallelepiped_with_sphere_and_extension :
  volume_of_interest = (1302 + 22 * Real.pi) / 3 :=
sorry

end volume_of_parallelepiped_with_sphere_and_extension_l150_150928


namespace binet_formula_variant_l150_150353

noncomputable def fib (n : ℕ) : ℚ :=
  let sqrt5 := real.sqrt 5
  let α := (1 + sqrt5) / 2
  let β := (1 - sqrt5) / 2
  (α ^ n - β ^ n) / sqrt5

theorem binet_formula_variant (n : ℕ) :
  2^(n-1) * fib(n) = ∑ k in finset.range( ⌊(n - 1) / 2⌋ + 1), (nat.choose n (2*k+1) * 5^k) :=
by
  sorry

end binet_formula_variant_l150_150353


namespace probability_snow_at_least_once_l150_150807

noncomputable def probability_at_least_once_snow : ℚ :=
  1 - (↑((1:ℚ) / 4) ^ 5)

theorem probability_snow_at_least_once (p : ℚ) (h : p = 3 / 4) :
  probability_at_least_once_snow = 1023 / 1024 := by
  sorry

end probability_snow_at_least_once_l150_150807


namespace bears_on_each_shelf_l150_150502

theorem bears_on_each_shelf (initial_bears : ℕ) (additional_bears : ℕ) (shelves : ℕ) (total_bears : ℕ) (bears_per_shelf : ℕ) :
  initial_bears = 5 → additional_bears = 7 → shelves = 2 → total_bears = initial_bears + additional_bears → bears_per_shelf = total_bears / shelves → bears_per_shelf = 6 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end bears_on_each_shelf_l150_150502


namespace unique_positive_integer_appending_digits_eq_sum_l150_150208

-- Define the problem in terms of Lean types and properties
theorem unique_positive_integer_appending_digits_eq_sum :
  ∃! (A : ℕ), (A > 0) ∧ (∃ (B : ℕ), (0 ≤ B ∧ B < 1000) ∧ (1000 * A + B = (A * (A + 1)) / 2)) :=
sorry

end unique_positive_integer_appending_digits_eq_sum_l150_150208


namespace max_area_triangle_centroid_l150_150505

-- Condition that A, B, C are points on the fixed ellipse
def on_ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

-- Points on the ellipse parameterized
def A (a b u : ℝ) := (a * Real.cos u, b * Real.sin u)
def B (a b v : ℝ) := (a * Real.cos v, b * Real.sin v)
def C (a b w : ℝ) := (a * Real.cos w, b * Real.sin w)

-- Centroid calculation
def centroid (Ax Ay Bx By Cx Cy : ℝ) : ℝ × ℝ :=
  ((Ax + Bx + Cx) / 3, (Ay + By + Cy) / 3)

-- Main theorem: maximum area condition
theorem max_area_triangle_centroid (a b : ℝ) (u v w : ℝ) :
  on_ellipse a b (A a b u).fst (A a b u).snd →
  on_ellipse a b (B a b v).fst (B a b v).snd →
  on_ellipse a b (C a b w).fst (C a b w).snd →
  let G := centroid (A a b u).fst (A a b u).snd (B a b v).fst (B a b v).snd (C a b w).fst (C a b w).snd in
  (G = (0, 0)) ↔ (∃ k, (u = k ∨ u = k + 2 * Real.pi / 3 ∨ u = k - 2 * Real.pi / 3) ∧
                   (v = k ∨ v = k + 2 * Real.pi / 3 ∨ v = k - 2 * Real.pi / 3) ∧
                   (w = k ∨ w = k + 2 * Real.pi / 3 ∨ w = k - 2 * Real.pi / 3)) :=
by
  sorry

end max_area_triangle_centroid_l150_150505


namespace interval_of_increase_range_of_a_for_monotonicity_l150_150158

noncomputable def f (x a : ℝ) : ℝ := Real.exp x - a * x - 1

theorem interval_of_increase (a : ℝ) :
  (∀ x, (Real.exp x - a) ≥ 0) ↔ 
  (if a <= 0 then ∀ x, x ∈ (-∞, +∞) else ∀ x, x ∈ (Real.log a, +∞)) :=
sorry

theorem range_of_a_for_monotonicity :
  (∀ x, (Real.exp x - a) ≥ 0) ↔ (a ≤ 0) :=
sorry

end interval_of_increase_range_of_a_for_monotonicity_l150_150158


namespace find_r_s_l150_150940

def quadratic_eq (x r s : ℝ) : ℝ := 3 * x ^ 2 + r * x + s

theorem find_r_s :
  (∃ r s : ℝ, (∀ (x: ℝ), quadratic_eq x r s = 0 ↔ x = 2 + real.sqrt 3 ∨ x = 2 - real.sqrt 3) ∧ r = -12 ∧ s = 3) :=
sorry

end find_r_s_l150_150940


namespace value_of_expression_l150_150178

theorem value_of_expression (a b c d m : ℝ)
  (h1 : a = -b)
  (h2 : c * d = 1)
  (h3 : |m| = 5)
  : 2 * (a + b) - 3 * c * d + m = 2 ∨ 2 * (a + b) - 3 * c * d + m = -8 := by
  sorry

end value_of_expression_l150_150178


namespace johnPaysPerYear_l150_150718

-- Define the conditions
def epiPenCost : ℝ := 500
def insuranceCoverage : ℝ := 0.75
def epiPenFrequencyPerYear : ℝ := 2 -- Twice a year since 12 months / 6 months per EpiPen

-- Calculate the cost after insurance
def costAfterInsurance (cost : ℝ) (coverage : ℝ) : ℝ :=
  cost * (1 - coverage)

-- Calculate the annual cost
def annualCost (freq : ℝ) (cost : ℝ) : ℝ :=
  freq * cost

-- The statement asserting the proof
theorem johnPaysPerYear (cost : ℝ) (coverage : ℝ) (freq : ℝ) : 
  epiPenCost = cost → 
  insuranceCoverage = coverage → 
  epiPenFrequencyPerYear = freq → 
  annualCost freq (costAfterInsurance cost coverage) = 250 := 
by 
  intros _ _ _ 
  sorry

end johnPaysPerYear_l150_150718


namespace remy_water_usage_l150_150356

theorem remy_water_usage :
  ∃ R : ℕ, (Remy = 3 * R + 1) ∧ 
    (Riley = R + (3 * R + 1) - 2) ∧ 
    (R + (3 * R + 1) + (R + (3 * R + 1) - 2) = 48) ∧ 
    (Remy = 19) :=
sorry

end remy_water_usage_l150_150356


namespace quadratic_graph_properties_l150_150524

theorem quadratic_graph_properties (a b : ℝ) (h : a ≠ 0):
  let c := (9 * b^2) / (16 * a)
  let f := λ x : ℝ, a * x^2 + b * x + c
  (a < 0 → ∃ x_max, ∀ x, f x ≤ f x_max) ∧ (a > 0 → ∃ x_min, ∀ x, f x ≥ f x_min) :=
by
  sorry

end quadratic_graph_properties_l150_150524


namespace smallest_n_l150_150620

theorem smallest_n (n : ℕ) (h : nat.has_sqrt (96 * n)) : n = 6 :=
by
  sorry

end smallest_n_l150_150620


namespace area_perspective_drawing_l150_150793

-- Define the variables
variables (a b : ℝ)

-- Given conditions
def area_horizontally_placed_triangle (a b : ℝ) := (1 / 2) * a * b = sqrt 6 / 2

def height_perspective_drawing (b : ℝ) := (sqrt 2 / 4) * b

-- The Lean statement we want to prove
theorem area_perspective_drawing (a b : ℝ) (h₁ : area_horizontally_placed_triangle a b) : 
  (1 / 2) * a * height_perspective_drawing b = sqrt 3 / 4 :=
sorry

end area_perspective_drawing_l150_150793


namespace dice_probability_l150_150758

def first_die_prob : ℚ := 3 / 8
def second_die_prob : ℚ := 3 / 4
def combined_prob : ℚ := first_die_prob * second_die_prob

theorem dice_probability :
  combined_prob = 9 / 32 :=
by
  -- Here we write the proof steps.
  sorry

end dice_probability_l150_150758


namespace range_of_2x_plus_y_range_of_c_l150_150983

open Real

def point_on_circle (x y : ℝ) : Prop := x^2 + y^2 = 2 * y

theorem range_of_2x_plus_y (x y : ℝ) (h : point_on_circle x y) : 
  1 - sqrt 2 ≤ 2 * x + y ∧ 2 * x + y ≤ 1 + sqrt 2 :=
sorry

theorem range_of_c (c : ℝ) : 
  (∀ x y : ℝ, point_on_circle x y → x + y + c > 0) → c ≥ -1 :=
sorry

end range_of_2x_plus_y_range_of_c_l150_150983


namespace sin_inequality_l150_150308

noncomputable theory

def f (a : ℕ → ℝ) (n : ℕ) (x : ℝ) : ℝ :=
  ∑ k in finset.range (n + 1), a k * (Real.sin ((k : ℝ) * x))

theorem sin_inequality
  (a : ℕ → ℝ)
  (n : ℕ)
  (h : ∀ x : ℝ, |f a n x| ≤ |Real.sin x|) :
  |∑ k in finset.range (n + 1), k * a k| ≤ 1 :=
sorry

end sin_inequality_l150_150308


namespace squares_below_16x_144y_1152_l150_150390

noncomputable def count_squares_below_line (a b c : ℝ) (x_max y_max : ℝ) : ℝ :=
  let total_squares := x_max * y_max
  let line_slope := -a/b
  let squares_crossed_by_diagonal := x_max + y_max - 1
  (total_squares - squares_crossed_by_diagonal) / 2

theorem squares_below_16x_144y_1152 : 
  count_squares_below_line 16 144 1152 72 8 = 248.5 := 
by
  sorry

end squares_below_16x_144y_1152_l150_150390


namespace tailor_cut_difference_l150_150500

def dress_silk_cut : ℝ := 0.75
def dress_satin_cut : ℝ := 0.60
def dress_chiffon_cut : ℝ := 0.55
def pants_cotton_cut : ℝ := 0.50
def pants_polyester_cut : ℝ := 0.45

theorem tailor_cut_difference :
  (dress_silk_cut + dress_satin_cut + dress_chiffon_cut) - (pants_cotton_cut + pants_polyester_cut) = 0.95 :=
by
  sorry

end tailor_cut_difference_l150_150500


namespace remainder_zero_l150_150740

theorem remainder_zero {n : ℕ} (h : n > 0) : 
  (2013^n - 1803^n - 1781^n + 1774^n) % 203 = 0 :=
by {
  sorry
}

end remainder_zero_l150_150740


namespace parallelogram_vertex_sum_l150_150012

theorem parallelogram_vertex_sum (P Q R : ℝ × ℝ) (S : ℝ × ℝ) 
  (hP : P = (-3, -2)) (hQ : Q = (1, -5)) (hR : R = (9, 1)) 
  (hP_diagonal : P = (fst R, snd R)) : 
  (fst S + snd S = 9) :=
sorry

end parallelogram_vertex_sum_l150_150012


namespace calculate_summer_sales_l150_150480

theorem calculate_summer_sales (fall_sales spring_sales winter_sales : ℕ) (fall_sales_percent : ℕ) (total_sales : ℕ) :
  fall_sales_percent = 20 → 
  fall_sales = 3 →
  spring_sales = 2 → 
  winter_sales = 3 → 
  total_sales = fall_sales * 5 → 
  let summer_sales := total_sales - (spring_sales + winter_sales + fall_sales) in
  summer_sales = 7 :=
by 
  intros h1 h2 h3 h4 h5;
  simp [h1, h2, h3, h4, h5];
  let summer_sales := 15 - (2 + 3 + 3);
  exact Eq.refl 7


end calculate_summer_sales_l150_150480


namespace element_in_subset_l150_150980

theorem element_in_subset (x : ℕ) (h : {1, x} ⊆ {1, 2, 3}) : x = 2 ∨ x = 3 := 
by 
  have hx : x ∈ {2, 3} := 
  begin
    simp at h,
    cases h with _ hx,
    exact hx 1 ⟨1, or.inl rfl⟩, 
  end,
  rwa [Set.mem_insert_iff, Set.mem_singleton_iff] at hx,
sorrry

end element_in_subset_l150_150980


namespace num_rows_of_gold_bars_l150_150778

-- Definitions from the problem conditions
def num_bars_per_row : ℕ := 20
def total_worth : ℕ := 1600000

-- Statement to prove
theorem num_rows_of_gold_bars :
  (total_worth / (total_worth / num_bars_per_row)) = 1 := 
by sorry

end num_rows_of_gold_bars_l150_150778


namespace big_boats_needed_l150_150863

-- Define the given conditions
variables (students total_boats big_boat_seats small_boat_seats : ℕ)

-- Assume the problem's conditions
axiom students_condition : students = 56
axiom total_boats_condition : total_boats = 8
axiom big_boat_seats_condition : big_boat_seats = 8
axiom small_boat_seats_condition : small_boat_seats = 4

-- Let n be the number of big boats
noncomputable def number_of_big_boats_needed := 
  ∃ (n m : ℕ), n + m = total_boats ∧ n * big_boat_seats + m * small_boat_seats = students ∧ n = 6

theorem big_boats_needed : number_of_big_boats_needed 56 8 8 4 :=
begin
  sorry
end

end big_boats_needed_l150_150863


namespace find_principal_sum_l150_150087

-- Define the conditions
def simple_interest (P R T : ℝ) : ℝ := P * R * T / 100
def given_si : ℝ := 2700
def given_r : ℝ := 6
def given_t : ℝ := 3
def result_p : ℝ := 15000

-- Formulate the statement
theorem find_principal_sum : simple_interest result_p given_r given_t = given_si :=
sorry

end find_principal_sum_l150_150087


namespace cos_graph_symmetry_center_l150_150679

noncomputable def shifted_symmetry_center (x : ℝ) : Prop :=
  ∃ k : ℤ, x = k * π / 2 - π / 24

theorem cos_graph_symmetry_center :
  shifted_symmetry_center (11 * π / 24) :=
begin
  use 1,
  norm_num,
end

end cos_graph_symmetry_center_l150_150679


namespace luncheon_cost_l150_150482

variables (s c p : ℝ)

def eq1 := 5 * s + 8 * c + 2 * p = 5.10
def eq2 := 6 * s + 11 * c + 2 * p = 6.45

theorem luncheon_cost (h₁ : 5 * s + 8 * c + 2 * p = 5.10) (h₂ : 6 * s + 11 * c + 2 * p = 6.45) : 
  s + c + p = 1.35 :=
  sorry

end luncheon_cost_l150_150482


namespace yellow_area_percentage_l150_150497

theorem yellow_area_percentage {s : ℝ} (h1 : s > 0) 
  (h2 : (green_cross_area + yellow_square_area = 0.25 * s^2))
  (h3 : yellow_square_area = 0.01 * s^2) : 
  (yellow_square_area / s^2) * 100 = 1 :=
by 
  calc (yellow_square_area / s^2) * 100 = (0.01 * s^2 / s^2) * 100 : by rw h3
                                  ... = 0.01 * 100 : by rw div_self (ne_of_gt h1)
                                  ... = 1 : by norm_num

end yellow_area_percentage_l150_150497


namespace probability_of_snowing_at_least_once_l150_150812

theorem probability_of_snowing_at_least_once (p : ℚ) (h : p = 3 / 4) :
  let q := 1 - p in
  let not_snowing_five_days := q ^ 5 in
  let at_least_once := 1 - not_snowing_five_days in
  at_least_once = 1023 / 1024 :=
by
  have q_def : q = 1 - p := rfl,
  have not_snowing_five_days_def : not_snowing_five_days = q ^ 5 := rfl,
  have at_least_once_def : at_least_once = 1 - not_snowing_five_days := rfl,
  sorry

end probability_of_snowing_at_least_once_l150_150812


namespace dihedral_angles_equal_polyhedral_angles_equal_l150_150847

-- Define what it means to be a nearly regular polyhedron
def nearly_regular_polyhedron (P : Type) [polyhedron P] : Prop :=
  (∀ f : face P, regular_polygon f) ∧
  (∃ (symmetry : symmetry_group P), vertex_transitive symmetry)

-- Prove that all dihedral angles are equal for nearly regular polyhedron
theorem dihedral_angles_equal {P : Type} [polyhedron P] (h : nearly_regular_polyhedron P) :
  ∀ (e1 e2 : edge P), dihedral_angle e1 = dihedral_angle e2 :=
by sorry

-- Prove that all polyhedral angles are equal for nearly regular polyhedron
theorem polyhedral_angles_equal {P : Type} [polyhedron P] (h : nearly_regular_polyhedron P) :
  ∀ (v1 v2 : vertex P), polyhedral_angle v1 = polyhedral_angle v2 :=
by sorry

end dihedral_angles_equal_polyhedral_angles_equal_l150_150847


namespace min_value_expression_l150_150553

theorem min_value_expression (x : ℝ) (hx : x > 0) : 9 * x + 1 / x^3 ≥ 10 :=
sorry

end min_value_expression_l150_150553


namespace countBeautifulDatesIn2023_l150_150912

def isBeautifulDate (date : Nat) (month : Nat) (year : Nat) : Bool :=
  let yearDigits := [2, 0, 2, 3]
  let dateDigits := (date.digits 10).erase_dup
  let monthDigits := (month.digits 10).erase_dup
  let allDigits := dateDigits ++ monthDigits ++ yearDigits
  allDigits.length == 6

theorem countBeautifulDatesIn2023 : 
  let year := 2023
  let validMonths := [1, 4, 5, 6, 7, 8, 9, 10]
  let validDays := List.range' 14 6 -- From 14 to 19
  6 * validMonths.length = 30 :=
by
  sorry

end countBeautifulDatesIn2023_l150_150912


namespace geometric_series_sum_proof_l150_150028

theorem geometric_series_sum_proof :
  ∑ k in Finset.range 12, (4: ℚ) ^ (-k) * 3 ^ k = 48750225 / 16777216 :=
by sorry

end geometric_series_sum_proof_l150_150028


namespace borrowed_amount_l150_150075

/-- 
A person borrows some money (P) for 2 years at 4% p.a. simple interest,
and lends it at 8% p.a. simple interest for 2 years.
His gain in the transaction per year is Rs. 200.
Prove that the amount of money borrowed is Rs. 5000.
-/
theorem borrowed_amount {P : ℝ} 
  (h_borrow_rate : ∀ P, P * (4 / 100) * 2)
  (h_lend_rate : ∀ P, P * (8 / 100) * 2)
  (h_gain_per_year : ∀ P, 200)
  (h_total_gain : ∀ P, 200 * 2 = 400) :
  P = 5000 := 
sorry

end borrowed_amount_l150_150075


namespace total_age_in_3_years_l150_150779

theorem total_age_in_3_years (Sam Sue Kendra : ℕ)
  (h1 : Kendra = 18)
  (h2 : Kendra = 3 * Sam)
  (h3 : Sam = 2 * Sue) :
  Sam + Sue + Kendra + 3 * 3 = 36 :=
by
  sorry

end total_age_in_3_years_l150_150779


namespace rancher_steers_cows_solution_l150_150492

theorem rancher_steers_cows_solution :
  ∃ (s c : ℕ), s > 0 ∧ c > 0 ∧ (30 * s + 31 * c = 1200) ∧ (s = 9) ∧ (c = 30) :=
by
  sorry

end rancher_steers_cows_solution_l150_150492


namespace trapezoid_ratio_l150_150763

variables {P Q R S K L M N: Type} [field K]
variables (A B C D : P)
variables {f : P → K} [linear_ordered_field K]

def trapezoid (A B C D : P) : Prop :=
∃ (S : set P), S = {A, B, C, D} ∧ ∀ x y ∈ S, (A, B) ∥ (C, D) ∧ (A ≠ B) ∧ (C ≠ D)

variables (AB BC CD DA : K)
variables (x y : K)
variables (S1 S2 S3 S4 : K)

-- Given:
-- 1. ABCD is a trapezoid with parallel bases AD and BC.
-- 2. K is on AB and L is on CD, such that KL is divided into three equal parts by the diagonals.
-- 3. Let AB = x and CD = y.

theorem trapezoid_ratio (h_trapezoid : trapezoid A B C D)
  (h_K_on_AB : K ∈ line A B) (h_L_on_CD : L ∈ line C D)
  (h_division : ∀ S S : K = 3)
  (h_AB_eq_x : f (A, B) = x) (h_CD_eq_y : f (C, D) = y)
  : x / y = 2 := sorry

end trapezoid_ratio_l150_150763


namespace minimum_value_of_z_l150_150179

theorem minimum_value_of_z (x y : ℝ) (h : x^2 + 2*x*y - 3*y^2 = 1) : ∃ min_z, min_z = (1 + Real.sqrt 5) / 4 ∧ ∀ z, z = x^2 + y^2 → min_z ≤ z :=
by
  sorry

end minimum_value_of_z_l150_150179


namespace candies_equal_l150_150755

theorem candies_equal (minyoung_candies : ℕ) (taehyung_candies : ℕ) (x : ℕ) :
  minyoung_candies = 9 → taehyung_candies = 3 → (minyoung_candies - x) = (taehyung_candies + x) → x = 3 :=
by
  intros h1 h2 h3
  -- definitions from conditions
  have total_candies := h1.trans (Nat.add_comm 9 3).symm,
  have equal_candies := total_candies ▸ (Nat.add_comm 6 6),
  sorry

end candies_equal_l150_150755


namespace sufficient_but_not_necessary_condition_l150_150054

theorem sufficient_but_not_necessary_condition (k : ℝ) : 
  (k = 1 → ∃ x y : ℝ, x^2 + y^2 = 1 ∧ x - y + k = 0) ∧ 
  ¬(∀ k : ℝ, ∃ x y : ℝ, x^2 + y^2 = 1 ∧ x - y + k = 0 → k = 1) :=
by
  sorry

end sufficient_but_not_necessary_condition_l150_150054


namespace axis_of_symmetry_value_at_shift_l150_150650

noncomputable def f (x : ℝ) : ℝ := (sin x) + (cos x)

theorem axis_of_symmetry :
  ∃ k : ℤ, ∀ x : ℝ, x = k * real.pi + real.pi / 4 :=
begin
  sorry
end

theorem value_at_shift (θ : ℝ) (hθ : θ ∈ set.Ioo 0 (real.pi / 2)) 
  (h : f (θ + real.pi / 4) = real.sqrt 2 / 3) :
  f (θ - real.pi / 4) = 4 / 3 :=
begin
  sorry
end

end axis_of_symmetry_value_at_shift_l150_150650


namespace monotonic_interval_l150_150191

def f (x : ℝ) : ℝ := if x ≤ 4 then -x^2 + 4*x else Real.log 2 x

theorem monotonic_interval (a : ℝ) : 
   (∀ x : ℝ, a < x ∧ x < a + 1 → f x ≤ f (x + 1)) ↔ a ∈ Set.Iic 1 ∨ a ∈ Set.Ici 4 :=
sorry

end monotonic_interval_l150_150191


namespace find_positive_integer_l150_150845

theorem find_positive_integer (n : ℕ) (hn : 0 < n) :
  let a := (5 + 10 + 15 + 20 + 25 + 30 + 35) / 7
  let b := 2 * n
  a ^ 2 = b ^ 2 → n = 10 :=
begin
  sorry,
end

end find_positive_integer_l150_150845


namespace range_of_sum_of_two_l150_150613

theorem range_of_sum_of_two (a b c : ℝ) (h1 : a + b + c = 1) (h2 : a^2 + b^2 + c^2 = 1) : 
  0 ≤ a + b ∧ a + b ≤ 4 / 3 :=
by
  -- Proof goes here.
  sorry

end range_of_sum_of_two_l150_150613


namespace range_of_k_for_one_solution_l150_150989

-- Definitions
def angle_B : ℝ := 60 -- Angle B in degrees
def side_b : ℝ := 12 -- Length of side b
def side_a (k : ℝ) : ℝ := k -- Length of side a (parameterized by k)

-- Theorem stating the range of k that makes the side_a have exactly one solution
theorem range_of_k_for_one_solution (k : ℝ) : (0 < k ∧ k <= 12) ∨ k = 8 * Real.sqrt 3 := 
sorry

end range_of_k_for_one_solution_l150_150989


namespace percentage_increase_is_5_l150_150349

noncomputable def new_salary := 2100
noncomputable def old_salary := 2000

def increase_in_salary := new_salary - old_salary

def percentage_increase := (increase_in_salary / old_salary.toFloat) * 100

theorem percentage_increase_is_5 : percentage_increase = 5 := by
  sorry

end percentage_increase_is_5_l150_150349


namespace intersection_triangles_similar_l150_150419

variables {α : ℝ} (hα : α < 180) (A B C A1 B1 C1 : Point) {O : Circle}

-- Assume rotation of triangle ABC by angle α around the center of its circumcircle creates triangle A1B1C1
def rotation_triangle (A B C : Point) (α : ℝ) (O : Circle) : Triangle :=
  rotate_triangle ABC α O

-- Points of intersection of sides or extensions
def intersection_points (A B C A1 B1 C1 : Point) : Triangle :=
  {A_B1 := intersect (line_through A B) (line_through A1 B1),
   B_C1 := intersect (line_through B C) (line_through B1 C1),
   C_A1 := intersect (line_through C A) (line_through C1 A1)}

theorem intersection_triangles_similar (hα : α < 180) :
  ∀ (O : Circle) (ABC A1B1C1 : Triangle),
  A1B1C1 = rotation_triangle ABC α O →
  similar_triangles (intersection_points A B C A1 B1 C1) ABC := sorry

end intersection_triangles_similar_l150_150419


namespace probability_of_triangle_formation_l150_150449

open Finset

def valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def stick_lengths : Finset ℕ := {1, 2, 4, 6, 9, 10, 14, 15, 18}

def all_combinations : Finset (ℕ × ℕ × ℕ) :=
  (stick_lengths.product stick_lengths).product stick_lengths
    |>.filter (λ t, ∃ a b c, (t.1.1 = a ∧ t.1.2 = b ∧ t.2 = c ∧ a < b ∧ b < c))

def valid_combinations : Finset (ℕ × ℕ × ℕ) :=
  all_combinations.filter (λ ⟨⟨a, b⟩, c⟩, valid_triangle a b c)

def probability_triangle : ℚ :=
  valid_combinations.card / all_combinations.card

theorem probability_of_triangle_formation : probability_triangle = 4 / 21 :=
by sorry

end probability_of_triangle_formation_l150_150449


namespace sin_identity_l150_150570

theorem sin_identity (α : ℝ) (h : Real.sin (π * α) = 4 / 5) : 
  Real.sin (π / 2 + 2 * α) = -24 / 25 :=
by
  sorry

end sin_identity_l150_150570


namespace routes_from_Bristol_to_Carlisle_l150_150073

-- Given conditions as definitions
def routes_Bristol_to_Birmingham : ℕ := 8
def routes_Birmingham_to_Manchester : ℕ := 5
def routes_Manchester_to_Sheffield : ℕ := 4
def routes_Sheffield_to_Newcastle : ℕ := 3
def routes_Newcastle_to_Carlisle : ℕ := 2

-- Define the total number of routes from Bristol to Carlisle
def total_routes_Bristol_to_Carlisle : ℕ := routes_Bristol_to_Birmingham *
                                            routes_Birmingham_to_Manchester *
                                            routes_Manchester_to_Sheffield *
                                            routes_Sheffield_to_Newcastle *
                                            routes_Newcastle_to_Carlisle

-- The theorem to be proved
theorem routes_from_Bristol_to_Carlisle :
  total_routes_Bristol_to_Carlisle = 960 :=
by
  -- Proof will be provided here
  sorry

end routes_from_Bristol_to_Carlisle_l150_150073


namespace inverse_g_of_5_l150_150666

def g (x : ℝ) : ℝ := 25 / (4 + 5 * x)

noncomputable def g_inv (y : ℝ) : ℝ :=
  (25 - 4 * y) / (5 * y)

theorem inverse_g_of_5 : (g_inv 5) ^ (-2) = 25 := by
  sorry

end inverse_g_of_5_l150_150666


namespace ratio_of_length_to_width_l150_150817

-- Define the conditions: area and difference
variables (L W : ℕ)
axiom area_eq : L * W = 676
axiom diff_eq : L - W = 39

-- State the theorem to prove the ratio L:W = 4:1
theorem ratio_of_length_to_width : L / W = 4 :=
by 
  sorry

end ratio_of_length_to_width_l150_150817


namespace range_of_set_is_8_l150_150083

theorem range_of_set_is_8 (a b c : ℕ) 
  (h1 : (a + b + c) / 3 = 6) 
  (h2 : b = 6) 
  (h3 : a = 2) 
  : max a (max b c) - min a (min b c) = 8 := 
by sorry

end range_of_set_is_8_l150_150083


namespace max_distance_line_l150_150798

noncomputable def equation_of_line (x y : ℝ) : ℝ := x + 2 * y - 5

theorem max_distance_line (x y : ℝ) : 
  equation_of_line 1 2 = 0 ∧ 
  (∀ (a b c : ℝ), c ≠ 0 → (x = 1 ∧ y = 2 → equation_of_line x y = 0)) ∧ 
  (∀ (L : ℝ → ℝ → ℝ), L 1 2 = 0 → (L = equation_of_line)) :=
sorry

end max_distance_line_l150_150798


namespace exists_contiguous_group_l150_150789

-- Define variables and conditions
variable (n : ℕ) (students : Fin n → Bool)

-- Main theorem statement
theorem exists_contiguous_group :
    n = 1000 →
    ∃ k, 100 ≤ k ∧ k ≤ 300 ∧ 
    ∃ start, let group := (List.finRange (2 * k)).map (λ i => students ((start + i) % n)) in 
    group.take k.count id = group.drop k.count id :=
by
  sorry

end exists_contiguous_group_l150_150789


namespace problem_part1_problem_part2_l150_150976

theorem problem_part1 :
  ∀ m : ℝ, (∀ x : ℝ, |x - 3| + |x - m| ≥ 2 * m) → m ≤ 1 :=
by
sorry

theorem problem_part2 :
  ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 → a + b + c = 1 →
    4 * a^2 + 9 * b^2 + c^2 ≥ 36 / 49 ∧
    (4 * a^2 + 9 * b^2 + c^2 = 36 / 49 ↔ a = 9 / 49 ∧ b = 4 / 49 ∧ c = 36 / 49) :=
by
sorry

end problem_part1_problem_part2_l150_150976


namespace unhappy_passengers_most_probable_is_1_expected_unhappy_passengers_is_variance_unhappy_passengers_is_l150_150348

noncomputable def unhappy_passengers_most_probable (n : ℕ) : ℕ :=
1

noncomputable def expected_unhappy_passengers (n : ℕ) : ℝ :=
Real.sqrt (n / Real.pi)

noncomputable def variance_unhappy_passengers (n : ℕ) : ℝ :=
0.182 * n

theorem unhappy_passengers_most_probable_is_1 (n : ℕ) : unhappy_passengers_most_probable n = 1 :=
sorry

theorem expected_unhappy_passengers_is (n : ℕ) : expected_unhappy_passengers n = Real.sqrt (n / Real.pi) :=
sorry

theorem variance_unhappy_passengers_is (n : ℕ) : variance_unhappy_passengers n = 0.182 * n :=
sorry

end unhappy_passengers_most_probable_is_1_expected_unhappy_passengers_is_variance_unhappy_passengers_is_l150_150348


namespace sum_even_integers_50_to_100_l150_150440

theorem sum_even_integers_50_to_100 : 
  let a := 50
  let l := 100
  let d := 2
  let n := (l - a) / d + 1
  S = n / 2 * (a + l) := 
  S = 1950 := 
by
  let a := 50
  let l := 100
  let d := 2
  let n := (l - a) / d + 1
  have h : S = n / 2 * (a + l)
  { sorry }
  have h2 : S = 1950
  { sorry }
  exact h2

end sum_even_integers_50_to_100_l150_150440


namespace smallest_x_y_sum_l150_150597

theorem smallest_x_y_sum (x y : ℕ) (h1 : x ≠ y) (h2 : x > 0) (h3 : y > 0) (h4 : 1 / (x : ℚ) + 1 / (y : ℚ) = 1 / 10) : x + y = 45 :=
by
  sorry

end smallest_x_y_sum_l150_150597


namespace cube_piercing_possible_l150_150070

theorem cube_piercing_possible : 
  ∃ (p₁ p₂ : ℕ → ℕ → ℕ),
  (p₁.face = opposite p₂.face) ∧ 
  (∀ brick ∈ cubes 20 20 20 (2, 2, 1), 
    (¬ (intersect_brick needle path brick))) :=
sorry

end cube_piercing_possible_l150_150070


namespace hyperbola_problem_line_intersects_hyperbola_l150_150640

noncomputable def hyperbola_equation (a b c : ℝ) (h : a > 0) (k : b > 0) 
                                      (h1 : c / a = sqrt 3) (h2 : c^2 + b^2 = 5) (h3 : c^2 = a^2 + b^2) : Prop :=
  (a = 1) ∧ (c = sqrt 3) ∧ (b = sqrt 2) ∧ ( ∀ x y, x^2 - (y^2)/(b^2) = 1 ↔ x^2 - y^2/2 = 1 )

theorem hyperbola_problem (a b c : ℝ) (h0 : a > 0) (h1 : b > 0)
                          (h2 : c / a = sqrt 3) (h3 : c^2 + b^2 = 5) 
                          (h4 : c^2 = a^2 + b^2) : hyperbola_equation a b c h0 h1 h2 h3 h4 := sorry

noncomputable def line_midpoint_property (m : ℝ) (h : 
    (∀ x y : ℝ, x^2 - y^2/2 = 1 → ∃ x1 y1 x2 y2 : ℝ, x1 ≠ x2 ∧ x2 - y2 + m = 0 
     ∧ ∀ x0 y0, x0 = (x1 + x2) / 2 ∧ y0 = x0 + m - 2*m → x0^2 + y0^2 = 5)) : Prop :=
  m = 1 ∨ m = -1

theorem line_intersects_hyperbola (m : ℝ) : 
  (∀ x y, x^2 - y^2 / 2 = 1 → 
    (∃ x1 y1 x2 y2, x1 ≠ x2 ∧ x1 - y1 + m = 0 ∧ 
    ∀ x0 y0, x0 = (x1 + x2) / 2 ∧ y0 = x0 + m - 2 * m → 
    x0^2 + y0^2 = 5)) → line_midpoint_property m := sorry

end hyperbola_problem_line_intersects_hyperbola_l150_150640


namespace evaluate_f_at_3_div_5_l150_150318

def f (x : ℚ) : ℚ := 15 * x^5 + 6 * x^4 + x^3 - x^2 - 2 * x - 1

theorem evaluate_f_at_3_div_5 : f (3 / 5) = -2 / 5 :=
by
sorry

end evaluate_f_at_3_div_5_l150_150318


namespace find_ab_l150_150144

variable (r s a b : ℝ)

-- Conditions of the problem
def cubic1 : polynomial ℝ := polynomial.X^3 + polynomial.C a * polynomial.X^2 + polynomial.C 10 * polynomial.X + polynomial.C 8
def cubic2 : polynomial ℝ := polynomial.X^3 + polynomial.C b * polynomial.X^2 + polynomial.C 17 * polynomial.X + polynomial.C 12

-- Common roots condition
def common_roots :=
  cubic1.eval r = 0 ∧ cubic1.eval s = 0 ∧ cubic2.eval r = 0 ∧ cubic2.eval s = 0 ∧ r ≠ s

-- The required proof problem
theorem find_ab (h : common_roots r s a b) : (a, b) = (12, 11) :=
sorry

end find_ab_l150_150144


namespace minimum_value_of_nS_n_l150_150563

noncomputable def a₁ (d : ℝ) : ℝ := -9/2 * d

noncomputable def S (n : ℕ) (d : ℝ) : ℝ :=
  n / 2 * (2 * a₁ d + (n - 1) * d)

theorem minimum_value_of_nS_n :
  S 10 (2/3) = 0 → S 15 (2/3) = 25 → ∃ (n : ℕ), (n * S n (2/3)) = -48 :=
by 
  intros h10 h15
  sorry

end minimum_value_of_nS_n_l150_150563


namespace num_divisible_by_2_3_5_7_lt_500_l150_150218

theorem num_divisible_by_2_3_5_7_lt_500 : 
  (finset.count (λ n : ℕ, n < 500 ∧ (n % 2 = 0 ∧ n % 3 = 0 ∧ n % 5 = 0 ∧ n % 7 = 0)) (finset.range 500)) = 2 :=
sorry

end num_divisible_by_2_3_5_7_lt_500_l150_150218


namespace num_divisible_by_2_3_5_7_under_500_l150_150219

-- Let the LCM of 2, 3, 5, and 7 be computed
def LCM_2_3_5_7 : ℕ := Nat.lcm (Nat.lcm 2 3) (Nat.lcm 5 7)

def num_multiples_less_than (n k : ℕ) : ℕ :=
  n / k

theorem num_divisible_by_2_3_5_7_under_500 : num_multiples_less_than 500 LCM_2_3_5_7 = 2 := by
  sorry

end num_divisible_by_2_3_5_7_under_500_l150_150219


namespace jellybean_proof_l150_150321

def number_vanilla_jellybeans : ℕ := 120

def number_grape_jellybeans (V : ℕ) : ℕ := 5 * V + 50

def number_strawberry_jellybeans (V : ℕ) : ℕ := (2 * V) / 3

def total_number_jellybeans (V G S : ℕ) : ℕ := V + G + S

def cost_per_vanilla_jellybean : ℚ := 0.05

def cost_per_grape_jellybean : ℚ := 0.08

def cost_per_strawberry_jellybean : ℚ := 0.07

def total_cost_jellybeans (V G S : ℕ) : ℚ := 
  (cost_per_vanilla_jellybean * V) + 
  (cost_per_grape_jellybean * G) + 
  (cost_per_strawberry_jellybean * S)

theorem jellybean_proof :
  ∃ (V G S : ℕ), 
    V = number_vanilla_jellybeans ∧
    G = number_grape_jellybeans V ∧
    S = number_strawberry_jellybeans V ∧
    total_number_jellybeans V G S = 850 ∧
    total_cost_jellybeans V G S = 63.60 :=
by
  sorry

end jellybean_proof_l150_150321


namespace sum_digits_c_plus_d_l150_150929

-- Define the integers c and d
def c : ℕ := (List.repeat 9 1986).foldl (λ acc d, acc * 10 + d) 0
def d : ℕ := (List.repeat 6 1986).foldl (λ acc d, acc * 10 + d) 0

-- Define the sum of the digits function
def sum_of_digits (n : ℕ) : ℕ :=
n.digits 10 |>.sum

-- Define the main statement
theorem sum_digits_c_plus_d : sum_of_digits (c + d) = 9931 := by
  sorry

end sum_digits_c_plus_d_l150_150929


namespace geometric_series_sum_proof_l150_150029

theorem geometric_series_sum_proof :
  ∑ k in Finset.range 12, (4: ℚ) ^ (-k) * 3 ^ k = 48750225 / 16777216 :=
by sorry

end geometric_series_sum_proof_l150_150029


namespace balance_equilibrium_even_sum_l150_150927

theorem balance_equilibrium_even_sum (n : ℕ) (h : n ≥ 2) (m : Fin n → ℕ)
  (h_m : ∀ k, 1 ≤ m k ∧ m k ≤ (k : ℕ) + 1) :
  (∃ p : Set (Fin n), S = (∑ i in p, m i) ∧ S = (∑ i in Fin n \ p, m i)) ↔
  (∑ i, m i) % 2 = 0 :=
by
  sorry

end balance_equilibrium_even_sum_l150_150927


namespace harmonic_mean_closest_to_l150_150391

theorem harmonic_mean_closest_to :
  Int.closestTo (2 * 5 * 2023 / (5 + 2023)) = 10 := by
  sorry

end harmonic_mean_closest_to_l150_150391


namespace smallest_stamps_l150_150839

theorem smallest_stamps : ∃ S, 1 < S ∧ (S % 9 = 1) ∧ (S % 10 = 1) ∧ (S % 11 = 1) ∧ S = 991 :=
by
  sorry

end smallest_stamps_l150_150839


namespace find_m_l150_150643

theorem find_m {A B : Set ℝ} (m : ℝ) :
  (A = {x : ℝ | x^2 + x - 12 = 0}) →
  (B = {x : ℝ | mx + 1 = 0}) →
  (A ∩ B = {3}) →
  m = -1 / 3 := 
by
  intros hA hB h_inter
  sorry

end find_m_l150_150643


namespace coeff_x4_is_neg2_l150_150518

-- Define the expression as a polynomial
def expr : Polynomial ℤ := 5 * (Polynomial.Coeff ℤ 4 (Monomial.mk 1) - Polynomial.Coeff ℤ 3 (Monomial.mk 2))
                      + 3 * (Polynomial.Coeff ℤ 2 (Monomial.mk 2) - Polynomial.Coeff ℤ 4 (Monomial.mk 3) + Polynomial.Coeff ℤ 6 (Monomial.mk 1))
                      - (Polynomial.Coeff ℤ 6 (Monomial.mk 5) - Polynomial.Coeff ℤ 4 (Monomial.mk 2))

-- The property we want to prove
theorem coeff_x4_is_neg2 : Polynomial.coeff expr 4 = -2 := 
by
  sorry

end coeff_x4_is_neg2_l150_150518


namespace count_integers_divisible_by_2_3_5_7_l150_150227

theorem count_integers_divisible_by_2_3_5_7 :
  ∃ n : ℕ, (∀ k : ℕ, k < 500 → (k % 2 = 0 ∧ k % 3 = 0 ∧ k % 5 = 0 ∧ k % 7 = 0) → k ≠ n → k < 500 ∧ k > 0) ∧
  (n = 2) :=
by
  sorry

end count_integers_divisible_by_2_3_5_7_l150_150227


namespace student_prob_l150_150255

-- The problem statement rephrased in Lean 4 to prove the given probability
theorem student_prob (students : Fin 4 → Fin 2) :
  (∃ i j, students i = 0 ∧ students j = 1) ->
  (probability (λ students, ∃ i j, students i = 0 ∧ students j = 1) = 7/8) :=
by
saying
  sorry

end student_prob_l150_150255


namespace min_value_of_f_is_46852_l150_150557

def f (x : ℝ) : ℝ := ∑ k in Finset.range 52, (x - (2 * k)) ^ 2

theorem min_value_of_f_is_46852 : (∃ x : ℝ, f x = 46852) := 
sorry

end min_value_of_f_is_46852_l150_150557


namespace cosine_of_114_is_negative_l150_150908

theorem cosine_of_114_is_negative :
  cos (114 * (Real.pi / 180)) < 0 :=
by
  -- skipping proof
  sorry

end cosine_of_114_is_negative_l150_150908


namespace compute_n_l150_150393

theorem compute_n (avg1 avg2 avg3 avg4 avg5 : ℚ) (h1 : avg1 = 1234 ∨ avg2 = 1234 ∨ avg3 = 1234 ∨ avg4 = 1234 ∨ avg5 = 1234)
  (h2 : avg1 = 345 ∨ avg2 = 345 ∨ avg3 = 345 ∨ avg4 = 345 ∨ avg5 = 345)
  (h3 : avg1 = 128 ∨ avg2 = 128 ∨ avg3 = 128 ∨ avg4 = 128 ∨ avg5 = 128)
  (h4 : avg1 = 19 ∨ avg2 = 19 ∨ avg3 = 19 ∨ avg4 = 19 ∨ avg5 = 19)
  (h5 : avg1 = 9.5 ∨ avg2 = 9.5 ∨ avg3 = 9.5 ∨ avg4 = 9.5 ∨ avg5 = 9.5) :
  ∃ n : ℕ, n = 2014 :=
by
  sorry

end compute_n_l150_150393


namespace number_of_valid_four_digit_numbers_correct_l150_150651

def first_digit_set := {2, 4, 6, 8}
def second_digit_set := {1, 3, 5, 7, 9}
def digit_set := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

noncomputable def number_of_valid_four_digit_numbers : ℕ := 
  let n_first := first_digit_set.to_finset.card in
  let n_second := (second_digit_set.to_finset \ first_digit_set.to_finset).card in
  let n_third := (digit_set.to_finset \ (first_digit_set ∪ second_digit_set).to_finset).card in
  let n_fourth := (n_third - 1) / 2 in
  n_first * n_second * n_third * n_fourth

theorem number_of_valid_four_digit_numbers_correct :
  number_of_valid_four_digit_numbers = 480 := by
  sorry

end number_of_valid_four_digit_numbers_correct_l150_150651


namespace eccentricity_of_ellipse_l150_150004

def is_eccentricity_of_ellipse (a b e c : ℝ) : Prop :=
  0 < b ∧ b < a ∧ e = (c / a) ∧ 2 * c = ∥(F₁ - F₂: ℝ)∥ ∧
  ∥F₁ - F₂∥ = 2 * c ∧ a = (c * (sqrt 3 + 1)) ∧
  e = (sqrt 3 - 1)

theorem eccentricity_of_ellipse (a b c : ℝ) (h: is_eccentricity_of_ellipse a b (sqrt 3 - 1) c) :
  e = sqrt 3 - 1 := 
  sorry

end eccentricity_of_ellipse_l150_150004


namespace false_proposition_l150_150168

-- Definitions of the conditions
def p1 := ∃ x0 : ℝ, x0^2 - 2*x0 + 1 ≤ 0
def p2 := ∀ x : ℝ, (1 ≤ x ∧ x ≤ 2) → x^2 - 1 ≥ 0

-- Statement to prove
theorem false_proposition : ¬ (¬ p1 ∧ ¬ p2) :=
by sorry

end false_proposition_l150_150168


namespace min_f_l150_150554

def f (x : ℝ) : ℝ :=
  ∑ k in finset.range(52), (x - 2 * k)^2

theorem min_f : ∃ x : ℝ, ∀ y : ℝ, f x ≤ f y ∧ f x = 46852 :=
sorry

end min_f_l150_150554


namespace assignment_three_booths_l150_150260

/--
In a sub-venue of the World Chinese Business Conference, there are three booths, A, B, and C, and 
four "bilingual" volunteers, namely 甲, 乙, 丙, and 丁. Each booth must have at least one person. 
Prove that the number of different ways to assign volunteers 甲 and 乙 to the same booth is 6.
-/
theorem assignment_three_booths :
  ∃ (volunteers : set string) (booths : set string),
    volunteers = {"甲", "乙", "丙", "丁"} ∧
    booths = {"A", "B", "C"} ∧
    (∀ booth ∈ booths, ∃ volunteer ∈ volunteers, volunteer_assigned_to_booth volunteer booth) ∧
    (number_of_ways_to_assign_same_booth "甲" "乙" = 6) :=
sorry

end assignment_three_booths_l150_150260


namespace find_b_sq_range_AB_l150_150593

-- Define the conditions of the problem
def ellipse_eq (x y b : ℝ) : Prop := 
  (x^2 / 8) + (y^2 / b^2) = 1

def circle_eq (x y : ℝ) : Prop := 
  x^2 + y^2 = 8 / 3

-- Define the tangent line condition for intersection points A and B such that OA ⊥ OB
def orthogonal_vectors (O A B : ℝ × ℝ) : Prop := 
  let ⟨x1, y1⟩ := A in
  let ⟨x2, y2⟩ := B in
  (x1 * x2 + y1 * y2) = 0

-- Define the problem statements
theorem find_b_sq (b : ℝ) (x1 x2 y1 y2 : ℝ) : 
  (∀ x y, ellipse_eq x y b) → 
  (∀ x y, circle_eq x y) →
  orthogonal_vectors (0, 0) (x1, y1) (x2, y2) →
  b^2 = 4 :=
sorry

theorem range_AB (x1 x2 y1 y2 : ℝ) (A B : ℝ × ℝ) : 
  (∀ x y, circle_eq x y) →
  orthogonal_vectors (0, 0) A B →
  ∃ AB_lower AB_upper, 
    AB_lower = (4 * real.sqrt 6) / 3 ∧ 
    AB_upper = 2 * real.sqrt 3 ∧
    AB_lower ≤ real.dist A B ∧ 
    real.dist A B ≤ AB_upper :=
sorry

end find_b_sq_range_AB_l150_150593


namespace trucks_transportation_l150_150422

theorem trucks_transportation (k : ℕ) (H : ℝ) : 
  (∃ (A B C : ℕ), 
     A + B + C = k ∧ 
     A ≤ k / 2 ∧ B ≤ k / 2 ∧ C ≤ k / 2 ∧ 
     (0 ≤ (k - 2*A)) ∧ (0 ≤ (k - 2*B)) ∧ (0 ≤ (k - 2*C))) 
  →  (k = 7 → (2 : ℕ) = 2) :=
sorry

end trucks_transportation_l150_150422


namespace find_group_of_number_2018_l150_150405

-- We define the sequence and groups
def nth_group_count (n : ℕ) : ℕ := 3 * n - 2
def even_seq (k : ℕ) : ℕ := 2 * k

-- We establish the condition under which a number belongs to a group
def sum_of_group_counts (n : ℕ) : ℕ := (List.range n).map nth_group_count).sum
def group_contains_number (n k : ℕ) : Prop := 
  sum_of_group_counts (n - 1) < k ∧ k ≤ sum_of_group_counts n

-- We are asked to prove that 2018 belongs to the 27th group
theorem find_group_of_number_2018 : ∃ n : ℕ, group_contains_number n 1009 := sorry

end find_group_of_number_2018_l150_150405


namespace digit_B_value_l150_150388

theorem digit_B_value (B : ℕ) (h : 4 * 1000 + B * 100 + B * 10 + 2 % 9 = 0):
  B = 6 :=
begin
  sorry
end

end digit_B_value_l150_150388


namespace probability_satisfied_l150_150909

variable (N : ℕ) (p : ℝ) (negative_reviews : ℕ := 60) (positive_reviews : ℕ := 20)

axiom condition1 : 0.80 * N * (1 - p) = 60
axiom condition2 : 0.15 * N * p = 20

theorem probability_satisfied : p = 0.64 := by
  have condition1_rearranged : 0.80 * (1 - p) * N = 60 := condition1
  have condition2_rearranged : 0.15 * p * N = 20 := condition2
  have ratio_eq : (0.15 * p * N) / (0.80 * (1 - p) * N) = positive_reviews / negative_reviews := by
    sorry
  have ratio_simplified : (0.15 * p) / (0.80 * (1 - p)) = 20 / 60 := by
    sorry
  have cross_multiply : 0.15 * p * 3 = 0.80 * (1 - p) := by
    sorry
  have eliminate_parenthesis : 0.45 * p = 0.80 - 0.80 * p := by
    sorry
  have combine_terms : 1.25 * p = 0.80 := by
    sorry
  have solve_p : p = 0.80 / 1.25 := by
    sorry
  have final_value : p ≈ 0.64 := by
    sorry
  exact final_value

end probability_satisfied_l150_150909


namespace composite_expression_l150_150565

theorem composite_expression (n : ℕ) (h : n > 1) : ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ 3^(2*n+1) - 2^(2*n+1) - 6^n = a * b :=
sorry

end composite_expression_l150_150565


namespace logarithm_inequality_l150_150623

theorem logarithm_inequality (n m a : ℝ) (h : n < m ∧ m < 0) (ha : 0 < a ∧ a < 1) : 
  Real.log a (-m) > Real.log a (-n) :=
sorry

end logarithm_inequality_l150_150623


namespace correct_equations_count_l150_150100

def eq1(x : ℝ) : Prop := (x^4)^4 = x^8
def eq2(y : ℝ) : Prop := ((y^2)^2)^2 = y^8
def eq3(y : ℝ) : Prop := (-y^2)^3 = y^6
def eq4(x : ℝ) : Prop := ((-x)^3)^2 = x^6

theorem correct_equations_count : 
  ∀ (x y : ℝ), 
  [eq1 x, eq2 y, eq3 y, eq4 x].count (λ eq, eq) = 2 := 
by 
  sorry

end correct_equations_count_l150_150100


namespace donation_amount_l150_150917

theorem donation_amount :
  let betty_strawberries := 25
  let matthew_strawberries := betty_strawberries + 30
  let natalie_strawberries := nat.floor (matthew_strawberries / 3)
  let emily_strawberries := natalie_strawberries / 2
  let ethan_strawberries := natalie_strawberries * 2
  let total_strawberries := betty_strawberries + matthew_strawberries + 
                            natalie_strawberries + emily_strawberries + 
                            ethan_strawberries
  let jars := total_strawberries / 12
  let revenue := jars * 6
  let donation := 0.40 * revenue
  donation = 26.40 :=
by
  let betty_strawberries := 25
  let matthew_strawberries := betty_strawberries + 30
  let natalie_strawberries := 18
  let emily_strawberries := 9
  let ethan_strawberries := 36
  let total_strawberries := 143
  let jars := nat.floor (total_strawberries / 12)
  let revenue := jars * 6
  let donation := 0.40 * revenue
  sorry

end donation_amount_l150_150917


namespace line_of_sight_not_blocked_by_circle_l150_150977

theorem line_of_sight_not_blocked_by_circle (a : ℝ) : 
  let circle_eq := (x y : ℝ) → x^2 + y^2 = 1,
      A := (0, -2 : ℝ),
      B := (a, 2 : ℝ)
  in  a < - (4 * real.sqrt 3 / 3) ∨ a > (4 * real.sqrt 3 / 3) :=
sorry

end line_of_sight_not_blocked_by_circle_l150_150977


namespace min_dist_one_l150_150291

def circle_center_radius (x y : ℝ) : Prop := (x - 4)^2 + (y + 3)^2 = 16
def parabola_point (x y : ℝ) : Prop := x^2 = 8 * y

def dist (p1 p2 : ℝ × ℝ) : ℝ := (Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2))

noncomputable def min_AB_dist : ℝ :=
  let C : ℝ × ℝ := (4, -3) in
  let t : ℝ := 0 in
  let B : ℝ × ℝ := (2 * Real.sqrt 2 * t, 2 * t^2) in
  dist C B - 4

theorem min_dist_one : min_AB_dist = 1 := 
by 
  sorry

end min_dist_one_l150_150291


namespace intersection_A_B_l150_150238

def A : Set ℝ := { x | log (1 / 2) (2 * x + 1) > -1 }
def B : Set ℝ := { x | 1 < 3 ^ x ∧ 3 ^ x < 9 }

theorem intersection_A_B : A ∩ B = { x : ℝ | 0 < x ∧ x < 1 / 2 } :=
by
  sorry

end intersection_A_B_l150_150238


namespace age_ratio_l150_150685

theorem age_ratio (B_age : ℕ) (H1 : B_age = 34) (A_age : ℕ) (H2 : A_age = B_age + 4) :
  (A_age + 10) / (B_age - 10) = 2 :=
by
  sorry

end age_ratio_l150_150685


namespace incorrect_connection_probability_l150_150412

theorem incorrect_connection_probability :
  let p := 0.02
  let r2 := 1 / 9
  let r3 := 8 / 81
  let probability_wrong_two_errors := 3 * p^2 * (1 - p) * r2
  let probability_wrong_three_errors := 1 * p^3 * r3
  let total_probability_correct_despite_errors := probability_wrong_two_errors + probability_wrong_three_errors
  let total_probability_incorrect := 1 - total_probability_correct_despite_errors
  ((total_probability_correct_despite_errors ≈ 0.000131) → 
  (total_probability_incorrect ≈ 1 - 0.000131)) :=
by
  sorry

end incorrect_connection_probability_l150_150412


namespace distance_between_A_and_B_l150_150870

noncomputable def downstream_time (D : ℝ) := D / 22
noncomputable def upstream_time (D : ℝ) := (D / 2) / 8

def total_travel_time (D : ℝ) := downstream_time D + upstream_time D

theorem distance_between_A_and_B : ∃ (D : ℝ), total_travel_time D = 19 ∧ D ≈ 111 := 
by {
  -- proof will be inserted here
  sorry
}

end distance_between_A_and_B_l150_150870


namespace sum_logarithms_divisors_l150_150408

theorem sum_logarithms_divisors (n : ℕ) (h : (∑ a in finset.range (n+1), ∑ b in finset.range (n+1), real.log10 (2^a * 3^b)) = 468) : n = 9 := sorry

end sum_logarithms_divisors_l150_150408


namespace solution_set_of_inequality_l150_150819

theorem solution_set_of_inequality (x : ℝ) : (∃ x, (0 ≤ x ∧ x < 1) ↔ (x-2)/(x-1) ≥ 2) :=
sorry

end solution_set_of_inequality_l150_150819


namespace sqrt_cos_product_l150_150937

def cos_squared_product (θ : ℝ) : ℝ :=
  2 - (Real.cos θ) ^ 2

theorem sqrt_cos_product :
  sqrt (cos_squared_product (Real.pi / 9) *
        cos_squared_product ((2 * Real.pi) / 9) *
        cos_squared_product ((4 * Real.pi) / 9)) = sqrt 995 := by
  sorry

end sqrt_cos_product_l150_150937


namespace positive_difference_of_perimeters_l150_150005

-- Conditions
def perimeter_7x2_rect := 18
def perimeter_2x3_rect := 2 * (2 + 3)

-- Assertion to prove
theorem positive_difference_of_perimeters : abs (perimeter_7x2_rect - perimeter_2x3_rect) = 8 := by
  sorry

end positive_difference_of_perimeters_l150_150005


namespace correct_solution_l150_150448

def fractional_equation (x : ℚ) : Prop :=
  (1 - x) / (2 - x) - 1 = (3 * x - 4) / (x - 2)

theorem correct_solution (x : ℚ) (h : fractional_equation x) : 
  x = 5 / 3 :=
sorry

end correct_solution_l150_150448


namespace work_hours_together_l150_150037

theorem work_hours_together (t : ℚ) :
  (1 / 9) * (9 : ℚ) = 1 ∧ (1 / 12) * (12 : ℚ) = 1 ∧
  (7 / 36) * t + (1 / 9) * (15 / 4) = 1 → t = 3 :=
by
  sorry

end work_hours_together_l150_150037


namespace intersection_A_B_l150_150292

def A : Set ℝ := {x | x^2 + x - 12 < 0}
def B : Set ℝ := {x | sqrt (x + 2) < 3}
def C : Set ℝ := {x | -2 ≤ x ∧ x < 3}

theorem intersection_A_B :
  A ∩ B = C := by
  sorry

end intersection_A_B_l150_150292


namespace relationship_among_a_b_c_l150_150157

noncomputable def a : ℝ := (1 / 3) ^ (Real.log 3 / Real.log 2)
noncomputable def b : ℝ := (1 / 3) ^ (Real.log 4 / Real.log 5)
noncomputable def c : ℝ := 3 ^ Real.log 3

theorem relationship_among_a_b_c :
  a < b ∧ b < c :=
by
  have h1 : Real.log 2 < Real.log 3, from by sorry
  have h2 : Real.log 5 < Real.log 4, from by sorry
  have h3 : Real.log 1/3 < 0, from by sorry
  have h4 : (1 / 3) ^ x is strictly decreasing, from by sorry
  have h5 : a = (1 /3) ^( Real.log 3 / Real.log 2), from by sorry
  have h6 : b = (1 /3) ^( Real.log 4 / Real.log 5 ), from by sorry
  have h7 : c = 3 ^ Real.log 3, from by sorry
  exact sorry
 

end relationship_among_a_b_c_l150_150157


namespace truck_left_1_hour_later_l150_150465

theorem truck_left_1_hour_later (v_car v_truck : ℝ) (time_to_pass : ℝ) : 
  v_car = 55 ∧ v_truck = 65 ∧ time_to_pass = 6.5 → 
  1 = time_to_pass - (time_to_pass * (v_car / v_truck)) := 
by
  intros h
  sorry

end truck_left_1_hour_later_l150_150465


namespace max_largestSum_l150_150509

noncomputable def largestSum (a b c d : ℕ) : ℕ :=
  ab + bc + cd + ad

theorem max_largestSum (a b c d : ℕ) (h : {a, b, c, d} = {3, 4, 5, 6}) :
  largestSum a b c d = 77 := by
  sorry

end max_largestSum_l150_150509


namespace center_of_3x3_square_l150_150269

theorem center_of_3x3_square:
  (∃ a b c d e f g h i : ℕ, 
    set.univ = {a, b, c, d, e, f, g, h, i} ∧ 
    a + b + c + d + e + f + g + h + i = 45 ∧
    a + e + i = 6 ∧
    c + e + g = 20) → 
  ∃ e : ℕ, e = 3 :=
by 
  sorry

end center_of_3x3_square_l150_150269


namespace complex_exp_cos_l150_150582

theorem complex_exp_cos (z : ℂ) (α : ℂ) (n : ℕ) (h : z + z⁻¹ = 2 * Complex.cos α) : 
  z^n + z⁻¹^n = 2 * Complex.cos (n * α) :=
by
  sorry

end complex_exp_cos_l150_150582


namespace x_seq_inequality_l150_150400

theorem x_seq_inequality {x : ℕ → ℝ} (h : ∀ i j : ℕ, x (i + j) ≤ x i + x j) (n : ℕ) : 
  (∑ i in range n, x (i + 1) / (i + 1 + 1)) ≥ x n := 
by
  sorry

end x_seq_inequality_l150_150400


namespace expression_not_equal_l150_150840

variable (a b c : ℝ)

theorem expression_not_equal :
  (a - (b - c)) ≠ (a - b - c) :=
by sorry

end expression_not_equal_l150_150840


namespace product_permutation_l150_150969

theorem product_permutation (m : ℕ) (h : 0 < m) : m * (m + 1) * (m + 2) * ... * (m + 15) = (A_{m + 15}^16) :=
by
  sorry

end product_permutation_l150_150969


namespace classical_probability_experiment_example_l150_150035

/-- 
Prove that the experiment of "four students draw lots to select one person 
to attend a meeting" is a classical probability model.
-/
def is_classical_probability_model : Prop :=
  ∃ (outcome_space : Finset ℕ), (∀ outcome ∈ outcome_space, ∃ (prob : ℝ), prob = 1 / (outcome_space.card)) ∧
                                 outcome_space.card = 4

theorem classical_probability_experiment_example : 
  is_classical_probability_model :=
begin
  sorry
end

end classical_probability_experiment_example_l150_150035


namespace smallest_four_digit_equiv_mod_five_l150_150832

theorem smallest_four_digit_equiv_mod_five : 
  ∃ (n : ℤ), n >= 1000 ∧ n % 5 = 4 ∧ ∀ m, (m >= 1000 ∧ m % 5 = 4) → n ≤ m :=
by
  use 1004
  split
  sorry

end smallest_four_digit_equiv_mod_five_l150_150832


namespace items_purchased_max_pendants_l150_150371

theorem items_purchased (x y : ℕ) (h1 : x + y = 180) (h2 : 80 * x + 50 * y = 11400) : x = 80 ∧ y = 100 :=
by {
  have h3 : y = 180 - x,
  { rw [←h1], },
  rw [←h3] at h2,
  linarith,
  sorry
}

theorem max_pendants (m : ℕ) (h1 : 180 - m ≥ 0) (h2 : 10 * m + 20 * (180 - m) ≥ 2900) : m ≤ 70 :=
by {
  linarith,
  sorry
}

end items_purchased_max_pendants_l150_150371


namespace probability_of_sequential_draws_l150_150421

theorem probability_of_sequential_draws :
  let total_cards := 52
  let num_fours := 4
  let remaining_after_first_draw := total_cards - 1
  let remaining_after_second_draw := remaining_after_first_draw - 1
  num_fours / total_cards * 1 / remaining_after_first_draw * 1 / remaining_after_second_draw = 1 / 33150 :=
by sorry

end probability_of_sequential_draws_l150_150421


namespace willow_played_total_hours_l150_150447

variable (minutesFootball : ℕ) (minutesBasketball : ℕ) (totalMinutes : ℕ) (totalHours : ℕ)

-- Defining the conditions
def minutes_played : Prop := 
  minutesFootball = 60 ∧ minutesBasketball = 60 ∧ totalMinutes = minutesFootball + minutesBasketball

-- The total hours calculation based on the given conditions
noncomputable def hours_played (totalMinutes : ℕ) : ℕ := totalMinutes / 60

-- Prove the total playing time in hours
theorem willow_played_total_hours 
  (h : minutes_played)
  : hours_played totalMinutes = 2 := sorry

end willow_played_total_hours_l150_150447


namespace exact_fraction_difference_l150_150920

theorem exact_fraction_difference :
  let x := (8:ℚ) / 11
  let y := (18:ℚ) / 25 
  x - y = (2:ℚ) / 275 :=
by
  -- Definitions from conditions: x = 0.\overline{72} and y = 0.72
  let x := (8:ℚ) / 11
  let y := (18:ℚ) / 25 
  -- Goal is to prove the exact fraction difference
  show x - y = (2:ℚ) / 275
  sorry

end exact_fraction_difference_l150_150920


namespace propositions_correct_l150_150841

variable {R : Type} [LinearOrderedField R] {A B : Set R}

theorem propositions_correct :
  (¬ ∃ x : R, x^2 + x + 1 = 0) ∧
  (¬ (∃ x : R, x + 1 ≤ 2) → ∀ x : R, x + 1 > 2) ∧
  (∀ x : R, x ∈ A ∩ B → x ∈ A) ∧
  (∀ x : R, x > 3 → x^2 > 9 ∧ ∃ y : R, y^2 > 9 ∧ y < 3) :=
by
  sorry

end propositions_correct_l150_150841


namespace quadrilaterals_exist_l150_150967

theorem quadrilaterals_exist
  (points : Fin 4000 → ℝ × ℝ)
  (h_no_three_collinear : ∀ (i j k : Fin 4000), 
    i ≠ j → j ≠ k → i ≠ k → ¬ collinear (points i) (points j) (points k)) :
  ∃ (quadrilaterals : Fin 1000 → Fin 4 → Fin 4000),
  (∀ i j k l : Fin 4000, quadrilaterals i 0 ≠ quadrilaterals i 1 ∧ 
   quadrilaterals i 0 ≠ quadrilaterals i 2 ∧ quadrilaterals i 0 ≠ quadrilaterals i 3 ∧ 
   quadrilaterals i 1 ≠ quadrilaterals i 2 ∧ quadrilaterals i 1 ≠ quadrilaterals i 3 ∧ 
   quadrilaterals i 2 ≠ quadrilaterals i 3) ∧
  (∀ i j : Fin 1000, i ≠ j → ∀ x ∈ quadrilaterals i, ∀ y ∈ quadrilaterals j, quadrilaterals i x ≠ quadrilaterals j y) :=
sorry

end quadrilaterals_exist_l150_150967


namespace find_other_number_l150_150848

-- Given conditions
def lcm (a b : ℕ) : ℕ := (a * b) / (Nat.gcd a b)  -- Definition of LCM
def gcd (a b : ℕ) : ℕ := Nat.gcd a b              -- Definition of GCD
-- Conditions are LCM(4, x) = 36 and GCD(4, x) = 2
def cond_lcm (x : ℕ) : Prop := lcm 4 x = 36
def cond_gcd (x : ℕ) : Prop := gcd 4 x = 2

-- Theorem to be proved
theorem find_other_number (x : ℕ) (condL : cond_lcm x) (condG : cond_gcd x) : x = 18 :=
by
  sorry

end find_other_number_l150_150848


namespace monotonicity_nonneg_f_plus_sqrt_e_l150_150190

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  (a * x^2 + x - 1) * Real.exp (-x)

theorem monotonicity (a : ℝ) (h : a >= 0) :
  (a = 0 → (∀ x < 2, (f a x) < (f a 2)) ∧ ∀ x > 2, (f a x) < (f a 2)) ∧
  (a > 0 → (∀ x > -(1/a) ∧ x < 2, (f a x) < (f a 2)) ∧ 
    ∀ x < -(1/a), (f a x) < (f a (-(1/a))) ∧ ∀ x > 2, (f a x) < (f a 2)) :=
  sorry

theorem nonneg_f_plus_sqrt_e (a : ℝ) (h : a >= 2) :
  ∀ x, f a x + Real.sqrt (Real.exp 1) >= 0 :=
  sorry

end monotonicity_nonneg_f_plus_sqrt_e_l150_150190


namespace Aerith_negative_number_eventually_l150_150896

theorem Aerith_negative_number_eventually :
  ∀ (n : ℕ), (n >= 50) →
  (∀ (initial_numbers : ℕ → ℤ), (∀ k, 0 ≤ initial_numbers k) → 
    ∃ t : ℕ, ∃ x : ℕ, (x < n) /\ initial_numbers (x) < 0) :=
begin
  sorry
end

end Aerith_negative_number_eventually_l150_150896


namespace wrapping_paper_area_l150_150077

theorem wrapping_paper_area (s : ℝ) : 
    let base_side := s;
    let height := 2 * s;
    let side_length := 4 * s in
    side_length ^ 2 = 16 * s ^ 2 :=
by 
    let base_side := s;
    let height := 2 * s;
    let side_length := 4 * s;
    show side_length ^ 2 = 16 * s ^ 2;
    sorry

end wrapping_paper_area_l150_150077


namespace rica_fraction_l150_150354

theorem rica_fraction (F : ℝ) (spent_fraction : ℝ) (remaining_money: ℝ) (group_prize : ℝ) (frac_won : ℝ) :
  spent_fraction = 1 / 5 →
  remaining_money = 300 →
  group_prize = 1000 →
  frac_won = F / group_prize →
  (frac_won = 3 / 8): 
  (4 / 5 * F = remaining_money) := 
begin
  sorry
end

end rica_fraction_l150_150354


namespace num_divisible_by_2_3_5_7_under_500_l150_150223

-- Let the LCM of 2, 3, 5, and 7 be computed
def LCM_2_3_5_7 : ℕ := Nat.lcm (Nat.lcm 2 3) (Nat.lcm 5 7)

def num_multiples_less_than (n k : ℕ) : ℕ :=
  n / k

theorem num_divisible_by_2_3_5_7_under_500 : num_multiples_less_than 500 LCM_2_3_5_7 = 2 := by
  sorry

end num_divisible_by_2_3_5_7_under_500_l150_150223


namespace or_is_true_given_p_true_q_false_l150_150180

theorem or_is_true_given_p_true_q_false (p q : Prop) (hp : p) (hq : ¬q) : p ∨ q :=
by
  sorry

end or_is_true_given_p_true_q_false_l150_150180


namespace cost_difference_correct_l150_150513

-- Define the data for Bottle R
def capsules_R := 250
def cost_R := 6.25

-- Define the data for Bottle T
def capsules_T := 100
def cost_T := 3.0

-- Define the cost per capsule for Bottle R
def cost_per_capsule_R := cost_R / capsules_R

-- Define the cost per capsule for Bottle T
def cost_per_capsule_T := cost_T / capsules_T

-- Define the expected difference
def expected_difference := 0.005

-- Lean statement to prove the difference
theorem cost_difference_correct :
  (cost_per_capsule_T - cost_per_capsule_R = expected_difference) :=
by
  sorry

end cost_difference_correct_l150_150513


namespace rectangle_sides_not_odd_intersections_l150_150344

/-- A type representing a point with coordinates that do not lie on the grid lines. -/
structure NonIntegerPoint (α : Type) :=
  (x y : α)
  (h_x : ¬ isInteger x)
  (h_y : ¬ isInteger y)

/-- A rectangle in the coordinate plane with sides forming 45 degree angles with the grid lines and whose vertices do not lie on the grid lines. -/
structure Rectangle45 (α : Type) :=
  (A B C D : NonIntegerPoint α)
  (h45_1 : ∃ (θ : ℝ), θ = 45 ∧ (B.x - A.x) = (C.x - D.x) ∧ (B.y - A.y) = (C.y - D.y))
  (h45_2 : ∃ (θ : ℝ), θ = 45 ∧ (C.x - B.x) = (D.x - A.x) ∧ (C.y - B.y) = (D.y - A.y))

open Rectangle45

/-- Prove that each side of the given rectangle cannot intersect an odd number of grid lines. -/
theorem rectangle_sides_not_odd_intersections (α : Type) [linear_ordered_field α] (R : Rectangle45 α) : 
  ¬ (∀ (side : R.A × R.B | R.B × R.C | R.C × R.D | R.D × R.A), is_odd (number_of_grid_line_intersections side)) := sorry

end rectangle_sides_not_odd_intersections_l150_150344


namespace men_with_6_boys_work_l150_150862

theorem men_with_6_boys_work (m b : ℚ) (x : ℕ) :
  2 * m + 4 * b = 1 / 4 →
  x * m + 6 * b = 1 / 3 →
  2 * b = 5 * m →
  x = 1 :=
by
  intros h1 h2 h3
  sorry

end men_with_6_boys_work_l150_150862


namespace number_of_multiples_of_15_between_35_and_200_l150_150652

theorem number_of_multiples_of_15_between_35_and_200 : ∃ n : ℕ, n = 11 ∧ ∃ k : ℕ, k ≤ 200 ∧ k ≥ 35 ∧ (∃ m : ℕ, m < n ∧ 45 + m * 15 = k) :=
by
  sorry

end number_of_multiples_of_15_between_35_and_200_l150_150652


namespace fourth_term_of_geometric_sequence_l150_150879

theorem fourth_term_of_geometric_sequence 
  (a r : ℕ) 
  (h₁ : a = 3)
  (h₂ : a * r^2 = 75) :
  a * r^3 = 375 := 
by
  sorry

end fourth_term_of_geometric_sequence_l150_150879


namespace main_problem_l150_150102

variables {m n : ℝ}
variables (h_m_pos : 0 < m) (h_n_pos : 0 < n) (h_m_ne_n : m ≠ n)

def ellipse_eq (x y : ℝ) := m * x^2 + n * y^2 = 1

theorem main_problem
  (A B C D E F : ℝ × ℝ)
  (h_AB_slope : A.2 - B.2 = A.1 - B.1)
  (h_perp_bisector : (E.1, E.2) = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))
  (h_perp_bisector_ellipse1 : ellipse_eq m n C.1 C.2)
  (h_perp_bisector_ellipse2 : ellipse_eq m n D.1 D.2)
  (h_perp_slope : E.1 = F.1)
  (h_F : F = (E.1, E.2)) :
  (C.1 - D.1) ^ 2 + (C.2 - D.2) ^ 2 - ((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2) = 4 * ((E.1 - F.1) ^ 2 + (E.2 - F.2) ^ 2) ∧
  ∃ O : ℝ × ℝ, ∀ P ∈ {A, B, C, D}, (P.1 - O.1)^2 + (P.2 - O.2)^2 = r^2 
  :=
sorry

end main_problem_l150_150102


namespace watches_sync_after_1600_days_l150_150340

-- Define the rate at which Glafira's watch gains time
def glafira_gain_per_day : ℕ := 36  -- in seconds

-- Define the rate at which Gavrila's watch loses time
def gavrila_loss_per_day : ℕ := 18  -- in seconds

-- Define total daily time deviation
def total_daily_time_deviation : ℕ := glafira_gain_per_day + gavrila_loss_per_day

-- Define the total number of seconds in a day
def total_seconds_in_a_day : ℕ := 86400

-- Define the number of days after which both watches display the correct time again
def days_until_correct_time : ℕ := total_seconds_in_a_day / total_daily_time_deviation

theorem watches_sync_after_1600_days : days_until_correct_time = 1600 := by
  simp [days_until_correct_time, total_seconds_in_a_day, total_daily_time_deviation, glafira_gain_per_day, gavrila_loss_per_day]
  rw [Nat.div_eq_of_lt] 
  sorry

end watches_sync_after_1600_days_l150_150340


namespace constant_term_in_binomial_expansion_l150_150795

theorem constant_term_in_binomial_expansion :
  let expr := (λ x : ℝ, (x^(1/2) + x^(-2))^10)
  let general_term := (λ r : ℕ, (Nat.choose 10 r) * (x^(1/2))^(10 - r) * (x^(-2))^r)
  ∃ (r : ℕ), r = 2 → general_term r = 45 := by 
  sorry

end constant_term_in_binomial_expansion_l150_150795


namespace hyperbola_focus_asymptote_distance_l150_150982

theorem hyperbola_focus_asymptote_distance (m : ℝ) (h : m > 0) :
  let F := (real.sqrt (3 + 3 * m), 0)
  let asymptote := (λ x, x / real.sqrt m)
  let d := (λ F a, (abs (F.1 - 0)) / real.sqrt (1 + m))
  d F asymptote = real.sqrt 3 :=
sorry

end hyperbola_focus_asymptote_distance_l150_150982


namespace large_font_pages_l150_150721

theorem large_font_pages (L S : ℕ) (h1 : L + S = 21) (h2 : 3 * L = 2 * S) : L = 8 :=
by {
  sorry -- Proof can be filled in Lean; this ensures the statement aligns with problem conditions.
}

end large_font_pages_l150_150721


namespace complement_A_in_U_l150_150646

def U := set ℝ
def A : set ℝ := {x | -1 ≤ x ∧ x < 2}

theorem complement_A_in_U : (U \ A) = {x | x < -1 ∨ x ≥ 2} :=
by
  sorry

end complement_A_in_U_l150_150646


namespace total_area_of_rectangles_l150_150525

/-- The combined area of two adjacent rectangular regions given their conditions -/
theorem total_area_of_rectangles (u v w z : ℝ) 
  (hu : 0 < u) (hv : 0 < v) (hw : 0 < w) (hz : w < z) : 
  (u + v) * z = (u + v) * w + (u + v) * (z - w) :=
by
  sorry

end total_area_of_rectangles_l150_150525


namespace geometric_series_sum_value_l150_150032

theorem geometric_series_sum_value :
  let a : ℚ := 3 / 4
  let r : ℚ := 3 / 4
  let n : ℕ := 12
  \(\sum_{k\in \mathbb{N}, 1 \leq k \leq n} a * r^(k - 1)\) = \(\frac{48758625}{16777216}\) :=
sorry

end geometric_series_sum_value_l150_150032


namespace smallest_positive_four_digit_integer_equivalent_to_4_mod_5_l150_150834

theorem smallest_positive_four_digit_integer_equivalent_to_4_mod_5 :
  ∃ n : ℕ, n ≡ 4 [MOD 5] ∧ n ≥ 1000 ∧ n = 1004 := 
begin
  use 1004,
  split,
  { norm_num, },
  split,
  { norm_num, },
  { refl, },
end

end smallest_positive_four_digit_integer_equivalent_to_4_mod_5_l150_150834


namespace number_of_possible_digits_to_make_divisible_by_4_l150_150876

def four_digit_number_divisible_by_4 (N : ℕ) : Prop :=
  let number := N * 1000 + 264
  number % 4 = 0

theorem number_of_possible_digits_to_make_divisible_by_4 :
  ∃ (count : ℕ), count = 10 ∧ (∀ (N : ℕ), N < 10 → four_digit_number_divisible_by_4 N) :=
by {
  sorry
}

end number_of_possible_digits_to_make_divisible_by_4_l150_150876


namespace mass_percentage_of_H_in_BaOH2_8H2O_l150_150142

/-- Define the molar masses of elements involved -/
def molar_mass_ba : ℝ := 137.327
def molar_mass_o : ℝ := 15.999
def molar_mass_h : ℝ := 1.008

/-- Define the total molar mass of Ba(OH)2·8H2O -/
def total_molar_mass : ℝ := molar_mass_ba + 10 * molar_mass_o + 18 * molar_mass_h -- note the calculations inline

/-- Define the mass of Hydrogen in the compound -/
def mass_h : ℝ := 18 * molar_mass_h

/-- Define the mass percentage of Hydrogen in the compound -/
def mass_percentage_h : ℝ := (mass_h / total_molar_mass) * 100

/-- The theorem to prove the mass percentage of Hydrogen in Ba(OH)2·8H2O is approximately 5.754% -/
theorem mass_percentage_of_H_in_BaOH2_8H2O : mass_percentage_h ≈ 5.754 := by sorry

end mass_percentage_of_H_in_BaOH2_8H2O_l150_150142


namespace q_value_l150_150743

-- Define the conditions and the problem statement
theorem q_value (a b m p q : ℚ) (h1 : a * b = 3) 
  (h2 : (a + 1 / b) * (b + 1 / a) = q) : 
  q = 16 / 3 :=
by
  sorry

end q_value_l150_150743


namespace parallel_planes_implies_parallel_line_l150_150648

-- Definitions of parallelism for planes and lines
variables (α β : Type) [Plane α] [Plane β]
variable (a : Line)

-- Given conditions
axiom a_in_alpha : a ⊆ α
axiom planes_parallel : α ∥ β → a ∥ β

-- Main theorem: α ∥ β is a sufficient but not necessary condition for a ∥ β
theorem parallel_planes_implies_parallel_line (α β : Type) [Plane α] [Plane β] (a : Line)
  (a_in_alpha : a ⊆ α) (planes_parallel : α ∥ β → a ∥ β) :
  (α ∥ β ↔ a ∥ β) → False :=
by
  -- Proof steps would go here
  sorry

end parallel_planes_implies_parallel_line_l150_150648


namespace cyclist_speed_l150_150428

theorem cyclist_speed (c d : ℕ) (h1 : d = c + 5) (hc : c ≠ 0) (hd : d ≠ 0)
    (H1 : ∀ tC tD : ℕ, 80 = c * tC → 120 = d * tD → tC = tD) : c = 10 := by
  sorry

end cyclist_speed_l150_150428


namespace min_ab_12_min_rec_expression_2_l150_150573

noncomputable def condition1 (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ (1 / a + 3 / b = 1)

theorem min_ab_12 {a b : ℝ} (h : condition1 a b) : 
  a * b = 12 :=
sorry

theorem min_rec_expression_2 {a b : ℝ} (h : condition1 a b) :
  (1 / (a - 1)) + (3 / (b - 3)) = 2 :=
sorry

end min_ab_12_min_rec_expression_2_l150_150573


namespace biased_coin_probability_l150_150461

theorem biased_coin_probability :
  ∃ x : ℝ, x < 1 / 2 ∧ (∃ k : ℝ, (x^3 * (1 - x)^3) = 1 / 400 ∧ x = 0.159) :=
begin
  sorry
end

end biased_coin_probability_l150_150461


namespace find_n_values_l150_150543

theorem find_n_values (n : ℕ) (h1 : 0 < n) : 
  (∃ (a : ℕ), n * 2^n + 1 = a * a) ↔ (n = 2 ∨ n = 3) := 
by
  sorry

end find_n_values_l150_150543


namespace part1_part2a_part2b_l150_150195

noncomputable def f (a x : ℝ) : ℝ := exp x * (1/3 * x^3 - 2 * x^2 + (a + 4) * x - 2 * a - 4)

theorem part1 (a : ℝ) :
  (∀ x : ℝ, x < 2 → f a x < -4/3 * exp x) ↔ (0 ≤ a) :=
sorry

theorem part2a (a : ℝ) :
  (∀ x : ℝ, f' a x = exp x * (1/3 * x^3 - x^2 + a * x - a) → (a ≥ 0) → count_extreme_points (f a) = 1) :=
sorry

theorem part2b (a : ℝ) :
  (∀ x : ℝ, f' a x = exp x * (1/3 * x^3 - x^2 + a * x - a) → (a < 0) → count_extreme_points (f a) = 3) :=
sorry

-- Auxiliary function to count the number of extreme points
noncomputable def count_extreme_points (f : ℝ → ℝ) : ℕ := sorry

end part1_part2a_part2b_l150_150195


namespace items_purchased_max_pendants_l150_150370

theorem items_purchased (x y : ℕ) (h1 : x + y = 180) (h2 : 80 * x + 50 * y = 11400) : x = 80 ∧ y = 100 :=
by {
  have h3 : y = 180 - x,
  { rw [←h1], },
  rw [←h3] at h2,
  linarith,
  sorry
}

theorem max_pendants (m : ℕ) (h1 : 180 - m ≥ 0) (h2 : 10 * m + 20 * (180 - m) ≥ 2900) : m ≤ 70 :=
by {
  linarith,
  sorry
}

end items_purchased_max_pendants_l150_150370


namespace ladder_velocity_l150_150095

-- Definitions of variables and conditions
variables (a τ l : ℝ)
def v1 : ℝ := a * τ
def sin_alpha : ℝ := a * τ^2 / (2 * l)
def cos_alpha : ℝ := Real.sqrt (1 - (a * τ^2 / (2 * l))^2)

-- Main statement to prove
theorem ladder_velocity (h : v1 * sin_alpha = v2 * cos_alpha) : 
  v2 = (a^2 * τ^3) / (Real.sqrt (4 * l^2 - a^2 * τ^4)) := 
sorry

end ladder_velocity_l150_150095


namespace max_triangle_difference_l150_150476

theorem max_triangle_difference (N : ℕ) :
  let T := N - 2 in
  let w := (N - N % 3) / 3 in
  let b := (N + 1 - N % 3) / 3 in
  if N % 3 = 1 then
    T > 0 → (w - b) = (N - 1) / 3 - 1
  else
    T > 0 → (w - b) = (N - N % 3) / 3 :=
by
  sorry

end max_triangle_difference_l150_150476


namespace count_special_numbers_within_100_l150_150489

-- Statement of the problem in Lean 4
theorem count_special_numbers_within_100 : 
  let numbers := {n : ℕ | n < 100 ∧ (n % 2 = 1) ∧ (n % 3 = 2) ∧ (n % 7 = 0)} in
  #numbers = 2 :=
by
  sorry

end count_special_numbers_within_100_l150_150489


namespace f_eq_f_at_neg_one_f_at_neg_500_l150_150790

noncomputable def f : ℝ → ℝ := sorry

theorem f_eq : ∀ x y : ℝ, f (x * y) + x = x * f y + f x := sorry
theorem f_at_neg_one : f (-1) = 1 := sorry

theorem f_at_neg_500 : f (-500) = 999 := sorry

end f_eq_f_at_neg_one_f_at_neg_500_l150_150790


namespace tangent_line_equation_l150_150626

theorem tangent_line_equation (f : ℝ → ℝ)
  (h : ∀ x, f (2 - x) = 2 * x^2 - 7 * x + 6) :
  let y := f 1 in
  3 * 1 - 2 = f' 1 → ∀ x : ℝ, y = 3 * x - 2 :=
sorry

end tangent_line_equation_l150_150626


namespace invalid_conclusion_of_given_identity_l150_150453

theorem invalid_conclusion_of_given_identity (n : ℤ) :
  (n ^ 2 - n * (2 * n + 1) = (n + 1) ^ 2 - (n + 1) * (2 * n + 1))
  → ∀ n : ℤ, ¬ (n = n + 1) :=
begin
  sorry
end

end invalid_conclusion_of_given_identity_l150_150453


namespace distinct_powers_exist_l150_150488

theorem distinct_powers_exist :
  ∃ (a1 a2 b1 b2 c1 c2 d1 d2 : ℕ),
    (∃ n, a1 = n^2) ∧ (∃ m, a2 = m^2) ∧
    (∃ p, b1 = p^3) ∧ (∃ q, b2 = q^3) ∧
    (∃ r, c1 = r^5) ∧ (∃ s, c2 = s^5) ∧
    (∃ t, d1 = t^7) ∧ (∃ u, d2 = u^7) ∧
    a1 - a2 = b1 - b2 ∧ b1 - b2 = c1 - c2 ∧ c1 - c2 = d1 - d2 ∧
    a1 ≠ b1 ∧ a1 ≠ c1 ∧ a1 ≠ d1 ∧ b1 ≠ c1 ∧ b1 ≠ d1 ∧ c1 ≠ d1 := 
sorry

end distinct_powers_exist_l150_150488


namespace complex_number_quadrant_l150_150585

def is_in_third_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im < 0

theorem complex_number_quadrant (z : ℂ) (h : z * complex.I = 2 - complex.I) :
  is_in_third_quadrant z :=
by
  sorry

end complex_number_quadrant_l150_150585


namespace sector_radius_l150_150373

theorem sector_radius (l S r : ℝ) (hl : l = 10 * real.pi) (hS : S = 60 * real.pi) (h_formula : S = (1 / 2) * l * r) : r = 12 :=
by
  -- insert proof here
  sorry

end sector_radius_l150_150373


namespace total_brownies_correct_l150_150331

noncomputable def initial_brownies : ℕ := 2 * 12
noncomputable def brownies_after_father : ℕ := initial_brownies - 8
noncomputable def brownies_after_mooney : ℕ := brownies_after_father - 4
noncomputable def additional_brownies : ℕ := 2 * 12
noncomputable def total_brownies : ℕ := brownies_after_mooney + additional_brownies

theorem total_brownies_correct : total_brownies = 36 := by
  sorry

end total_brownies_correct_l150_150331


namespace smallest_sum_of_xy_l150_150611

theorem smallest_sum_of_xy (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hxy : x ≠ y)
  (hcond : (1 : ℚ) / x + (1 : ℚ) / y = 1 / 10) : x + y = 45 :=
sorry

end smallest_sum_of_xy_l150_150611


namespace tangent_line_slope_4_tangent_line_at_point_2_6_l150_150638

noncomputable def f (x : ℝ) : ℝ := x^3 + x - 16

theorem tangent_line_slope_4 (x0 : ℝ) (y0 : ℝ) : 
  f' x0 = 4 → 
    ((x0 = 1 ∧ y0 = -14) ∧ (4 * x0 - y0 - 18 = 0)) ∨ 
    ((x0 = -1 ∧ y0 = -18) ∧ (4 * x0 - y0 - 14 = 0)) :=
  sorry

theorem tangent_line_at_point_2_6 : 
  (∃ k : ℝ, k = f' 2 ∧ k = 13) ∧ 13 * 2 - (-6) - 32 = 0 :=
  sorry

end tangent_line_slope_4_tangent_line_at_point_2_6_l150_150638


namespace number_of_valid_colorings_l150_150112

-- 4x4 grid
def Grid : Type := List (List Bool)

-- A coloring is valid if it satisfies the given conditions
def valid_coloring (g : Grid) : Prop :=
  ∃(cols : List (List Bool)), 
  -- Each column must start from the bottom and be consecutive
  (∀c ∈ cols, ∃(k : ℕ), ∀i, (i < k) → (nth_le c i sorry = true)) ∧
  -- Sum of colored cells in all columns must be 6
  (cols.map (λ c, c.filter (λ x, x = true).length)).sum = 6 ∧
  -- Number of colored cells in left column ≥ number of colored cells in right column
  ∀(i : ℕ), ((i < cols.length - 1) → 
            (cols[i].filter (λ x, x = true)).length ≥ 
            (cols[i + 1].filter (λ x, x = true)).length)

-- Theorem: Number of valid colorings for the 4x4 grid
theorem number_of_valid_colorings : ∃(n : ℕ), n = 8 ∧ ∀ (g : Grid), (valid_coloring g → g.length = n) := 
sorry

end number_of_valid_colorings_l150_150112


namespace sin_x_one_of_sec_sub_tan_l150_150174

theorem sin_x_one_of_sec_sub_tan (x : ℝ) (h : sec x - tan x = 5 / 3) : sin x = 1 :=
sorry

end sin_x_one_of_sec_sub_tan_l150_150174


namespace perimeter_shaded_region_l150_150690

-- Define the problem conditions
def is_center_of_circle (O : Type) := ∀ (R S : Type), (dist O R = 8) ∧ (dist O S = 8)

-- Define the total perimeter of shaded region given conditions
theorem perimeter_shaded_region (O R S : Type) (h : is_center_of_circle O) : 
  let arc_length_RS := (5/6) * (2 * Real.pi * 8) in
  dist O R + dist O S + arc_length_RS = 16 + (40 / 3) * Real.pi := 
by { sorry }

end perimeter_shaded_region_l150_150690


namespace grocer_rainy_day_theorem_l150_150881

noncomputable theory

def smallest_amount (x y z d h q : ℕ) :=
  x = 8 * d ∧ y = 8 * h ∧ z = 8 * q ∧
  7 ∣ 8 * d ∧ 7 ∣ 8 * h ∧ 7 ∣ 8 * q ∧
  6 ∣ 8 * d ∧ 6 ∣ 8 * h ∧ 6 ∣ 8 * q ∧
  ¬ 5 ∣ 8 * d ∧ ¬ 5 ∣ 8 * h ∧ ¬ 5 ∣ 8 * q ∧
  x + 0.5 * y + 0.25 * z = 294

theorem grocer_rainy_day_theorem (x y z d h q : ℕ) :
  smallest_amount x y z d h q := sorry

end grocer_rainy_day_theorem_l150_150881


namespace track_extension_needed_l150_150074

noncomputable def additional_track_length (r : ℝ) (g1 g2 : ℝ) : ℝ :=
  let l1 := r / g1
  let l2 := r / g2
  l2 - l1

theorem track_extension_needed :
  additional_track_length 800 0.04 0.015 = 33333 :=
by
  sorry

end track_extension_needed_l150_150074


namespace size_of_angle_C_l150_150271

theorem size_of_angle_C (a b c : ℝ) (h₁ : a = 7) (h₂ : b = 8) (h₃ : c = 13) :
  ∠A B C = 2 * π / 3 :=
by
  sorry

end size_of_angle_C_l150_150271


namespace least_possible_students_l150_150757

def TotalNumberOfStudents : ℕ := 35
def NumberOfStudentsWithBrownEyes : ℕ := 15
def NumberOfStudentsWithLunchBoxes : ℕ := 25
def NumberOfStudentsWearingGlasses : ℕ := 10

theorem least_possible_students (TotalNumberOfStudents NumberOfStudentsWithBrownEyes NumberOfStudentsWithLunchBoxes NumberOfStudentsWearingGlasses : ℕ) :
  ∃ n, n = 5 :=
sorry

end least_possible_students_l150_150757


namespace faster_train_speed_l150_150429

theorem faster_train_speed (v : ℝ) (t : ℝ) (d1 d2 : ℝ) (relative_speed_factor : ℝ) (time_to_cross : ℝ) :
  d1 = 100 → d2 = 100 → relative_speed_factor = 3 → time_to_cross = 10 →
  v * 3 = (d1 + d2) / time_to_cross →
  let faster_speed := 2 * v in
  let faster_speed_kmh := faster_speed * 3.6 in
  faster_speed_kmh = 48 :=
by
  intros
  let distance := d1 + d2
  let speed_relation : ℝ := v * relative_speed_factor
  assumption sorry

end faster_train_speed_l150_150429


namespace find_m_values_l150_150146

noncomputable def lines_cannot_form_triangle (m : ℝ) : Prop :=
  (4 * m - 1 = 0) ∨ (6 * m + 1 = 0) ∨ (m^2 + m / 3 - 2 / 3 = 0)

theorem find_m_values :
  { m : ℝ | lines_cannot_form_triangle m } = {4, -1 / 6, -1, 2 / 3} :=
by
  sorry

end find_m_values_l150_150146


namespace process_terminates_l150_150132

-- Definitions based on the given conditions.
def is_on_number_line (n : ℕ) : Prop :=
  n ≤ 2022

def valid_move (a b : ℕ) (positions : finset ℕ) : Prop :=
  (a ≠ b) ∧ (a + 2 ≤ b)

def move (a b : ℕ) (positions : finset ℕ) : finset ℕ :=
  (positions.erase a).erase b ∪ {a + 1, b - 1}

-- Theorem statement to show the process ends and determine the final configuration.
theorem process_terminates (positions : finset ℕ) (n : ℕ)
  (h_initial : ∀ p ∈ positions, is_on_number_line p)
  (h_total : positions.card = 2023)
  (h_avg : positions.sum id = 1011 * 2023) :
  ∃ final_positions : finset ℕ,
    (∀ p ∈ final_positions, is_on_number_line p) ∧
    final_positions.card = 2023 ∧
    final_positions.sum id = 1011 * 2023 ∧
    (∀ a b, valid_move a b final_positions → false) ∧
    final_positions = finset.singleton 1011 :=
sorry

end process_terminates_l150_150132


namespace required_bollards_l150_150898

theorem required_bollards 
  (bollards_per_side : ℕ)
  (sides : ℕ)
  (fraction_installed : ℚ)
  : bollards_per_side = 4000 → 
    sides = 2 → 
    fraction_installed = 3/4 → 
    let total_bollards := bollards_per_side * sides in 
    let installed_bollards := fraction_installed * total_bollards in 
    let remaining_bollards := total_bollards - installed_bollards in 
    remaining_bollards = 2000 :=
by
  intros h1 h2 h3
  simp [h1, h2, h3]
  sorry

end required_bollards_l150_150898


namespace values_a_cannot_take_l150_150201

def set_A (a : ℝ) : Set ℝ := {1, a^2}

theorem values_a_cannot_take (a : ℝ) : a ≠ 1 ∧ a ≠ -1 ↔ a ∈ {-1, 1} :=
by
  sorry

end values_a_cannot_take_l150_150201


namespace factor_expression_l150_150540

theorem factor_expression (z : ℂ) : 
  75 * z^12 + 162 * z^24 + 27 = 3 * (9 + z^12 * (25 + 54 * z^12)) :=
sorry

end factor_expression_l150_150540


namespace incenter_is_midpoint_l150_150572

noncomputable def midpoint_incenter_condition (A B C D E I O : Point) (R r : ℝ) 
  (k k₀ : Circle) : Prop :=
  (k.center = O) ∧ (k.radius = R) ∧
  (k₀.tangent_to_side CA D) ∧ (k₀.tangent_to_side CB E) ∧ 
  (k₀.internally_tangent k) ∧
  (incircle.center = I) ∧ (incircle.radius = r)

theorem incenter_is_midpoint {A B C D E O I : Point} {R r : ℝ}
  (k k₀ : Circle) 
  (h : midpoint_incenter_condition A B C D E I O R r k k₀) :
  midpoint I D E :=
sorry

end incenter_is_midpoint_l150_150572


namespace blueprint_length_conversion_l150_150341

theorem blueprint_length_conversion (scale : ℕ) (blueprint_length_cm : ℕ) (conversion_factor : ℕ) 
  (h_scale : scale = 50)
  (h_blueprint_length : blueprint_length_cm = 10)
  (h_conversion_factor : conversion_factor = 100) :
  (blueprint_length_cm * scale / conversion_factor) = 5 := 
by 
  -- conditions
  simp [h_scale, h_blueprint_length, h_conversion_factor]
  -- middle steps are skipped for simplicity
  sorry

end blueprint_length_conversion_l150_150341


namespace number_of_multiples_of_15_between_35_and_200_l150_150653

theorem number_of_multiples_of_15_between_35_and_200 : ∃ n : ℕ, n = 11 ∧ ∃ k : ℕ, k ≤ 200 ∧ k ≥ 35 ∧ (∃ m : ℕ, m < n ∧ 45 + m * 15 = k) :=
by
  sorry

end number_of_multiples_of_15_between_35_and_200_l150_150653


namespace problem_statement_l150_150316

variable (a : ℕ → ℝ)

-- Defining sequences {b_n} and {c_n}
def b (n : ℕ) := a n - a (n + 2)
def c (n : ℕ) := a n + 2 * a (n + 1) + 3 * a (n + 2)

-- Defining that a sequence is arithmetic
def is_arithmetic (seq : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, seq (n + 1) - seq n = d

-- Problem statement
theorem problem_statement :
  is_arithmetic a ↔ (is_arithmetic (c a) ∧ ∀ n, b a n ≤ b a (n + 1)) :=
sorry

end problem_statement_l150_150316


namespace common_root_implies_equal_coefficients_l150_150375

theorem common_root_implies_equal_coefficients 
  (p1 p2 q1 q2 : ℤ)
  (α : ℝ)
  (h1 : α^2 + p1 * α + q1 = 0)
  (h2 : α^2 + p2 * α + q2 = 0)
  (h3 : α ∉ ℤ) : 
  p1 = p2 ∧ q1 = q2 :=
sorry

end common_root_implies_equal_coefficients_l150_150375


namespace count_integers_divisible_by_2_3_5_7_l150_150228

theorem count_integers_divisible_by_2_3_5_7 :
  ∃ n : ℕ, (∀ k : ℕ, k < 500 → (k % 2 = 0 ∧ k % 3 = 0 ∧ k % 5 = 0 ∧ k % 7 = 0) → k ≠ n → k < 500 ∧ k > 0) ∧
  (n = 2) :=
by
  sorry

end count_integers_divisible_by_2_3_5_7_l150_150228


namespace shaded_area_correct_l150_150693

noncomputable def total_unique_shaded_area 
  (r R d : ℝ) (rectangle_width rectangle_height : ℝ) 
  (radius_small radius_large : ℝ) 
  (radial_distance: ℝ) 
  (rect_width rect_height : ℝ) 
  (A_intersection: ℝ): ℝ :=
  let A_r := real.pi * r^2
  let A_R := real.pi * R^2
  let A_rect := rectangle_width * rectangle_height
  A_rect - (A_r + A_R - A_intersection)

theorem shaded_area_correct 
  (r R d : ℝ) (rectangle_width rectangle_height : ℝ)
  (radius_small : r = 3)
  (radius_large : R = 6)
  (radial_distance : d = 3)
  (rect_width : rectangle_width = 18)
  (rect_height : rectangle_height = 48)
  (A_intersection: ℝ): 
  total_unique_shaded_area r R d rectangle_width rectangle_height 3 6 3 18 48 A_intersection = 864 - 42 * real.pi :=
sorry

end shaded_area_correct_l150_150693


namespace find_p_AF_perp_BF_l150_150696

theorem find_p (C : Type) [parabola C (λy x, y^2 = 2 * p * x) ]
  (line : ℝ → ℝ → Prop)
  (h_line : ∀ x y, line x y ↔ x - 2 * y + 4 = 0)
  (tangent_condition : ∀ x y, line x y → C x y → (by sorry : tangent_to_parabola x y C line))
  :
  p = 2 := by
  sorry

theorem AF_perp_BF (F : Point := ⟨1, 0⟩)
  (line1 line2 : ℝ → ℝ → Prop)
  (h_line1 : ∀ x y, line1 x y ↔ tangent_to_parabola_at_point x y (-1, some_coordinate))
  (h_line2 : ∀ x y, line2 x y ↔ tangent_to_parabola_at_point x y (-1, some_coordinate))
  (A B : Point)
  (h_intersection_A : line_intersection A line1 line)
  (h_intersection_B : line_intersection B line2 line)
  :
  is_perpendicular (F.vector_to A) (F.vector_to B) := by
  sorry

end find_p_AF_perp_BF_l150_150696


namespace f_g_3_l150_150236

def f (x : ℝ) : ℝ := 4 - Real.sqrt x
def g (x : ℝ) : ℝ := 4 * x + 3 * x^2 + x^3

theorem f_g_3 : f (g 3) = 4 - Real.sqrt 66 := by
  sorry

end f_g_3_l150_150236


namespace branches_on_fourth_tree_l150_150527

theorem branches_on_fourth_tree :
  ∀ (height_1 branches_1 height_2 branches_2 height_3 branches_3 height_4 avg_branches_per_foot : ℕ),
    height_1 = 50 →
    branches_1 = 200 →
    height_2 = 40 →
    branches_2 = 180 →
    height_3 = 60 →
    branches_3 = 180 →
    height_4 = 34 →
    avg_branches_per_foot = 4 →
    (height_4 * avg_branches_per_foot = 136) :=
by
  intros height_1 branches_1 height_2 branches_2 height_3 branches_3 height_4 avg_branches_per_foot
  intros h1_eq_50 b1_eq_200 h2_eq_40 b2_eq_180 h3_eq_60 b3_eq_180 h4_eq_34 avg_eq_4
  -- We assume the conditions of the problem are correct, so add them to the context
  have height1 := h1_eq_50
  have branches1 := b1_eq_200
  have height2 := h2_eq_40
  have branches2 := b2_eq_180
  have height3 := h3_eq_60
  have branches3 := b3_eq_180
  have height4 := h4_eq_34
  have avg_branches := avg_eq_4
  -- Now prove the desired result
  sorry

end branches_on_fourth_tree_l150_150527


namespace piggy_bank_total_value_l150_150885

def dimes : ℕ := 35
def total_coins : ℕ := 100
def quarter_value : ℚ := 0.25
def dime_value : ℚ := 0.10

theorem piggy_bank_total_value :
  let quarters := total_coins - dimes in
  (dimes * dime_value + quarters * quarter_value) = 19.75 :=
by
  let quarters := total_coins - dimes
  sorry

end piggy_bank_total_value_l150_150885


namespace jessica_mark_earnings_l150_150343

theorem jessica_mark_earnings (t : ℤ) : 
    (t + 2) * (4t + 1) = (4t - 7) * (t + 3) + 4 ↔ t = 5 := 
by {
  sorry
}

end jessica_mark_earnings_l150_150343


namespace sum_1998_terms_sequence_l150_150590

def sequence_cond (seq : ℕ → ℕ) : Prop :=
  seq 0 = 1 ∧
  (∀ k n, n > 0 → seq (k + n + 2^k - 1) = if n = k + 1 then 2 else seq (k + n))

def sum_first_n_terms (seq : ℕ → ℕ) (n : ℕ) : ℕ :=
  (finset.range n).sum seq

theorem sum_1998_terms_sequence :
  ∀ seq : ℕ → ℕ,
    sequence_cond seq →
    sum_first_n_terms seq 1998 = 3986 :=
by
  sorry

end sum_1998_terms_sequence_l150_150590


namespace eggs_leftover_l150_150895

-- Definitions for the conditions
def abigail_eggs : ℕ := 28
def beatrice_eggs : ℕ := 53
def carson_eggs : ℕ := 19
def carton_size : ℕ := 10

-- Total number of eggs
def total_eggs : ℕ := abigail_eggs + beatrice_eggs + carson_eggs

-- Statement to be proved
theorem eggs_leftover : total_eggs % carton_size = 0 :=
by apply_nat_mod_eq_zero sorry

end eggs_leftover_l150_150895


namespace problem_f_val_l150_150150

noncomputable def f : ℝ → ℝ := sorry

theorem problem_f_val (f : ℝ → ℝ)
  (h1 : ∀ x : ℝ, f (-x) = -f x)
  (h2 : ∀ x : ℝ, f (1 + x) = f (1 - x))
  (h3 : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → f x = x^3) :
  f 2015 = -1 :=
  sorry

end problem_f_val_l150_150150


namespace solve_quadratic_equation_l150_150026

theorem solve_quadratic_equation :
  ∀ x : ℝ, (10 - x) ^ 2 = 2 * x ^ 2 + 4 * x ↔ x = 3.62 ∨ x = -27.62 := by
  sorry

end solve_quadratic_equation_l150_150026


namespace sum_of_a6_and_a7_l150_150621

theorem sum_of_a6_and_a7 (a : ℕ → ℝ) (h_arith : ∃ d, ∀ n, a (n + 1) = a n + d)
  (h_sum : a 2 + a 3 + a 10 + a 11 = 48) : a 6 + a 7 = 24 :=
by
  sorry

end sum_of_a6_and_a7_l150_150621


namespace find_missing_number_l150_150952

theorem find_missing_number (x : ℕ) (h : 10010 - 12 * 3 * x = 9938) : x = 2 :=
by {
  sorry
}

end find_missing_number_l150_150952


namespace apples_given_by_Susan_l150_150780

theorem apples_given_by_Susan (x y final_apples : ℕ) (h1 : y = 9) (h2 : final_apples = 17) (h3: final_apples = y + x) : x = 8 := by
  sorry

end apples_given_by_Susan_l150_150780


namespace cos_C_value_l150_150683

-- Definition of the triangle and given conditions
variables (A B C : Type) [triangle A B C]
variable (BC : ℝ) (AC : ℝ) (cos_A_B : ℝ)
-- Given conditions
axiom BC_val : BC = 5
axiom AC_val : AC = 4
axiom cos_A_B_val : cos_A_B = 7 / 8

-- Auxiliary point D and its properties
variables (D : Type)
-- Assume D is such that DA = DB
axiom DA_eq_DB : distance D A = distance D B

-- Target: prove the value of cos C
theorem cos_C_value 
  (BC : ℝ := BC) (AC : ℝ := AC) (cos_A_B : ℝ := cos_A_B) 
  (BC_val : BC = 5) (AC_val : AC = 4) (cos_A_B_val : cos_A_B = 7 / 8) :
  cos (angle C) = -1 / 4 :=
by
  sorry

end cos_C_value_l150_150683


namespace not_subset_T_to_S_l150_150156

def is_odd (x : ℤ) : Prop := ∃ n : ℤ, x = 2 * n + 1
def is_of_form_4k_plus_1 (y : ℤ) : Prop := ∃ k : ℤ, y = 4 * k + 1

theorem not_subset_T_to_S :
  ¬ (∀ y, is_of_form_4k_plus_1 y → is_odd y) :=
sorry

end not_subset_T_to_S_l150_150156


namespace convert_polar_to_rectangular_minimum_distance_to_line_l150_150185

-- Necessary definitions and conditions
def polar_to_rectangular (ρ θ : ℝ) : ℝ × ℝ := (ρ * cos θ, ρ * sin θ)

variable (P : ℝ × ℝ)
variable (α : ℝ)
variable (p_eqn : P = (4 * cos α, 3 * sin α))
variable (ellipse_eqn : (P.1 ^ 2) / 16 + (P.2 ^ 2) / 9 = 1)
variable (l_eqn_polar : ∀ (ρ θ : ℝ), ρ * sin (θ - π / 4) = 3 * sqrt 2 → polar_to_rectangular ρ θ)
variable (l_eqn_rect : ∀ (x y : ℝ), x - y + 6 = 0)

-- Prove that the rectangular coordinate equation of line l is x - y + 6 = 0
theorem convert_polar_to_rectangular :
  ∀ (ρ θ : ℝ), ρ * sin (θ - π / 4) = 3 * sqrt 2 →
  ∃ (x y : ℝ), polar_to_rectangular ρ θ = (x, y) ∧ x - y + 6 = 0 :=
sorry

-- Prove that the minimum distance from point P to line l is sqrt 2 / 2
theorem minimum_distance_to_line :
  (P.1 ^ 2) / 16 + (P.2 ^ 2) / 9 = 1 →
  P = (4 * cos α, 3 * sin α) →
  ∀ (x y : ℝ), x - y + 6 = 0 →
  ∃ (d : ℝ), d = sqrt 2 / 2 :=
sorry

end convert_polar_to_rectangular_minimum_distance_to_line_l150_150185


namespace minimize_g_value_l150_150301

noncomputable def f (a x : ℝ) : ℝ := |x^2 - a * x|

noncomputable def g (a : ℝ) : ℝ :=
  sup { f a x | x : ℝ, 0 ≤ x ∧ x ≤ 1 }

theorem minimize_g_value : 
  let a := 2 * Real.sqrt 2 - 2 in g a = (3 - 2 * Real.sqrt 2) := 
sorry

end minimize_g_value_l150_150301


namespace find_cost_of_baseball_l150_150326

noncomputable def total_amount : ℝ := 20.52
noncomputable def cost_of_marbles : ℝ := 9.05
noncomputable def cost_of_football : ℝ := 4.95
noncomputable def cost_of_baseball : ℝ := total_amount - (cost_of_marbles + cost_of_football)

theorem find_cost_of_baseball : cost_of_baseball = 6.52 := sorry

end find_cost_of_baseball_l150_150326


namespace exists_a_func_max_on_interval_eq_zero_l150_150535

noncomputable def func (a x : ℝ) : ℝ :=
  cos x ^ 2 + a * sin x + 5 * a / 8 - 5 / 2

theorem exists_a_func_max_on_interval_eq_zero :
  ∃ (a : ℝ), a = 3 / 2 ∧
    ∃ (x ∈ Icc (0:ℝ) (π)), 
      ∀ (t ∈ Icc (0:ℝ) (π)), func a t ≤ func a x ∧ func a x = 0 :=
by
  sorry

end exists_a_func_max_on_interval_eq_zero_l150_150535


namespace min_translation_to_odd_function_l150_150017

noncomputable def f (x : ℝ) : ℝ := 2 * sqrt 3 * (cos x)^2 - 2 * sin x * cos x - sqrt 3

theorem min_translation_to_odd_function (t : ℝ) (h : t > 0) :
  ∃ (k : ℤ), 2 * t + π / 6 = k * π + π / 2 → t = π / 6 := sorry

end min_translation_to_odd_function_l150_150017


namespace smallest_angle_of_cyclic_quadrilateral_l150_150372

theorem smallest_angle_of_cyclic_quadrilateral (angles : ℝ → ℝ) (a d : ℝ) :
  -- Conditions
  (∀ n : ℕ, angles n = a + n * d) ∧ 
  (angles 3 = 140) ∧
  (a + d + (a + 3 * d) = 180) →
  -- Conclusion
  (a = 40) :=
by sorry

end smallest_angle_of_cyclic_quadrilateral_l150_150372


namespace Ben_has_card_5_l150_150691

open Finset

def card_game_deck : Finset ℕ := (range (10)) ∪ (range 5 15) -- Deck numbers from 5 to 14

structure PlayerScores (α : Type) :=
  (Emma Ben Mia Noah Lucas : α)

def player_scores : PlayerScores ℕ :=
  ⟨19, 10, 13, 23, 20⟩

def has_no_consecutive_numbers (a b : ℕ) : Prop :=
  (a < b - 1 ∨ b < a - 1)

noncomputable def player_cards : Type :=
  (Finset ℕ × Finset ℕ)

theorem Ben_has_card_5 :
  ∃ (Ben_cards : player_cards), 
  has_no_consecutive_numbers Ben_cards.1 Ben_cards.2 ∧
  Ben_cards.1 ∈ card_game_deck ∧ Ben_cards.2 ∈ card_game_deck ∧
  Ben_cards.1 + Ben_cards.2 = player_scores.Ben ∧
  (5 ∈ Ben_cards.1 ∨ 5 ∈ Ben_cards.2) :=
sorry

end Ben_has_card_5_l150_150691


namespace each_person_pays_12_10_l150_150791

noncomputable def total_per_person : ℝ :=
  let taco_salad := 10
  let daves_single := 6 * 5
  let french_fries := 5 * 2.5
  let peach_lemonade := 7 * 2
  let apple_pecan_salad := 4 * 6
  let chocolate_frosty := 5 * 3
  let chicken_sandwiches := 3 * 4
  let chili := 2 * 3.5
  let subtotal := taco_salad + daves_single + french_fries + peach_lemonade + apple_pecan_salad + chocolate_frosty + chicken_sandwiches + chili
  let discount := 0.10
  let tax := 0.08
  let subtotal_after_discount := subtotal * (1 - discount)
  let total_after_tax := subtotal_after_discount * (1 + tax)
  total_after_tax / 10

theorem each_person_pays_12_10 :
  total_per_person = 12.10 :=
by
  -- omitted proof
  sorry

end each_person_pays_12_10_l150_150791


namespace speed_computation_l150_150089

def train_length : ℝ := 45
def crossing_time : ℝ := 1.4998800095992322
def speed_of_train : ℝ := 108

theorem speed_computation (distance : ℝ) (time : ℝ) : distance = train_length → time = crossing_time →
  real.to_nnreal (distance / (time / 3600) / 1000) = real.to_nnreal speed_of_train :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end speed_computation_l150_150089


namespace sunny_ahead_in_second_race_l150_150688

theorem sunny_ahead_in_second_race
  (s w : ℝ)
  (h1 : s / w = 8 / 7) :
  let sunny_new_speed := 0.9 * s
  let distance_sunny_runs := 450
  let distance_windy_runs := 400
  let time_sunny := distance_sunny_runs / sunny_new_speed
  let distance_windy_in_time_sunny := w * time_sunny
  450 - distance_windy_in_time_sunny = 12.5 :=
by
  let sunny_new_speed := 0.9 * s
  let distance_sunny_runs := 450
  let distance_windy_runs := 400
  let time_sunny := distance_sunny_runs / sunny_new_speed
  let distance_windy_in_time_sunny := w * time_sunny
  sorry

end sunny_ahead_in_second_race_l150_150688


namespace polynomial_coefficients_are_rational_l150_150052

-- Given conditions
def P : Polynomial ℝ := sorry
axiom P_integer_valued : ∀ n : ℤ, P.eval (n.toReal) ∈ ℤ

-- The theorem to prove
theorem polynomial_coefficients_are_rational :
  ∀ i : ℕ, i ≤ P.degree → (P.coeff i) ∈ ℚ := sorry

end polynomial_coefficients_are_rational_l150_150052


namespace component_unqualified_l150_150469

/-- 
    The specified diameter range for a component is within [19.98, 20.02].
    The measured diameter of the component is 19.9.
    Prove that the component is unqualified.
-/
def is_unqualified (diameter_measured : ℝ) : Prop :=
    diameter_measured < 19.98 ∨ diameter_measured > 20.02

theorem component_unqualified : is_unqualified 19.9 :=
by
  -- Proof goes here
  sorry

end component_unqualified_l150_150469


namespace conjugate_of_z_l150_150376

def z : ℂ := (-3 + complex.I) / (2 + complex.I)

theorem conjugate_of_z :
  conj (z) = -1 - complex.I :=
by
  sorry

end conjugate_of_z_l150_150376


namespace determine_rectangle_R_area_l150_150481

def side_length_large_square (s : ℕ) : Prop :=
  s = 4

def area_rectangle_R (s : ℕ) (area_R : ℕ) : Prop :=
  s * s - (1 * 4 + 1 * 1) = area_R

theorem determine_rectangle_R_area :
  ∃ (s : ℕ) (area_R : ℕ), side_length_large_square s ∧ area_rectangle_R s area_R :=
by {
  sorry
}

end determine_rectangle_R_area_l150_150481


namespace scarves_per_box_l150_150765

theorem scarves_per_box (S M : ℕ) (h1 : S = M) (h2 : 6 * (S + M) = 60) : S = 5 :=
by
  sorry

end scarves_per_box_l150_150765


namespace john_annual_payment_l150_150712

open Real

-- Definitions extracted from the problem:
def epipen_cost : ℝ := 500
def insurance_coverage : ℝ := 0.75
def epipen_frequency_per_year : ℕ := 2
def john_payment_per_epipen : ℝ := epipen_cost * (1 - insurance_coverage)

-- The statement to be proved:
theorem john_annual_payment :
  john_payment_per_epipen * epipen_frequency_per_year = 250 :=
by
  sorry

end john_annual_payment_l150_150712


namespace sum_of_roots_of_quadratic_l150_150838

theorem sum_of_roots_of_quadratic (a b c : ℝ) (h_eq : 2 * (X^2) - 8 * X + 6 = 0) : 
  (-b / a) = 4 :=
sorry

end sum_of_roots_of_quadratic_l150_150838


namespace son_l150_150486

theorem son's_age (S M : ℕ) 
  (h1 : M = S + 35)
  (h2 : M + 2 = 2 * (S + 2)) : 
  S = 33 := 
by
  sorry

end son_l150_150486


namespace exponentiation_multiplication_l150_150921

theorem exponentiation_multiplication (a : ℝ) : a^6 * a^2 = a^8 :=
by sorry

end exponentiation_multiplication_l150_150921


namespace smallest_ndigit_integers_difference_l150_150303

theorem smallest_ndigit_integers_difference 
  (m n : ℕ)
  (hm : m = 110)
  (hn : n = 1010)
  (Hm : m % 13 = 6 ∧ m ≥ 100 ∧ ∀ k ∈ ℕ, (k % 13 = 6 ∧ k ≥ 100 ∧ k < m) → False)
  (Hn : n % 17 = 7 ∧ n ≥ 1000 ∧ ∀ l ∈ ℕ, (l % 17 = 7 ∧ l ≥ 1000 ∧ l < n) → False) :
  n - m = 900 := by
  sorry

end smallest_ndigit_integers_difference_l150_150303


namespace race_result_l150_150256

noncomputable def athlete := fin 6

variables (E D V G B A : athlete)

-- Conditions translation
def condition1 := B < A ∧ ∃ x y, x ≠ y ∧ x ≠ A ∧ y ≠ A ∧ x ≠ B ∧ y ≠ B ∧ B < x ∧ x < y ∧ y < A
def condition2 := D < V ∧ V < G
def condition3 := E < D ∧ D < B

theorem race_result (E D V G B A : athlete)
  (h1 : condition1 B A)
  (h2 : condition2 D V G)
  (h3 : condition3 E D B) :
  (E < D ∧ D < V ∧ V < G ∧ G < B ∧ B < A) :=
sorry

end race_result_l150_150256


namespace binary_to_decimal_l150_150526

theorem binary_to_decimal : (1 * 2^0 + 1 * 2^1 + 0 * 2^2 + 0 * 2^3 + 1 * 2^4 + 1 * 2^5) = 51 :=
by
  sorry

end binary_to_decimal_l150_150526


namespace incorrect_option_A_l150_150729

-- Definitions from the conditions
variables {α β : Type*} [plane α] [plane β]
variable {l : Type*}
variables (h_diff : α ≠ β) (h_l_not_in_beta : l ∉ β)

-- To be proved
theorem incorrect_option_A
  (h1 : parallel l α) (h2 : perpendicular α β) : ¬ perpendicular l β :=
sorry

end incorrect_option_A_l150_150729


namespace minimal_triangle_area_eq_minimal_sum_eq_l150_150978

def point_P : (ℝ × ℝ) := (4, 1)

def line_through (P : ℝ × ℝ) : Prop :=
∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (P.1 / a) + (P.2 / b) = 1

theorem minimal_triangle_area_eq (P : ℝ × ℝ) (hP : P = point_P) :
  ∃ l : ℝ × ℝ → ℝ, line_through P ∧ l = (λ (x : ℝ × ℝ), x.1 + 4 * x.2 - 8) :=
sorry

theorem minimal_sum_eq (P : ℝ × ℝ) (hP : P = point_P) :
  ∃ l : ℝ × ℝ → ℝ, line_through P ∧ l = (λ (x : ℝ × ℝ), x.1 + 2 * x.2 - 6) :=
sorry

end minimal_triangle_area_eq_minimal_sum_eq_l150_150978


namespace F_leq_zero_range_l150_150708

noncomputable def f (x : ℝ) := (1 / 2) * 2^x + (1 / 2)

noncomputable def f_inverse (y : ℝ) := Real.log2 (2 * y - 1)

noncomputable def F (x : ℝ) := f_inverse (2^(x - 1)) - Real.logb (1 / 2) (f x)

theorem F_leq_zero_range (x : ℝ) : 
  0 < x ∧ x < Real.log2 (Real.sqrt 3) → F x ≤ 0 :=
sorry

end F_leq_zero_range_l150_150708


namespace min_S_at_n_10_l150_150165

-- Define the conditions
variables {a₁ d : ℝ} (h_d : d ≠ 0) (h_a₁ : a₁ < 0)
noncomputable def S (n : ℕ) : ℝ := (n / 2) * (2 * a₁ + (n - 1) * d)

-- Lean theorem statement to prove that the minimum of S_n is at n = 10
theorem min_S_at_n_10 (h : S 7 = S 13) : ∀ n, S n ≥ S 10 :=
begin
  sorry 
end

end min_S_at_n_10_l150_150165


namespace right_triangles_area_perimeter_l150_150207

theorem right_triangles_area_perimeter :
  ∃ (n : ℕ), n = 7 ∧
  (∀ a b c : ℕ, a > 0 ∧ b > 0 ∧ c > 0 ∧ a * a + b * b = c * c ∧
                 (a * b / 2 = 3 * (a + b + ↑c)) → 
                  n = 7) :=
begin
  sorry
end

end right_triangles_area_perimeter_l150_150207


namespace smallest_sum_l150_150602

theorem smallest_sum (x y : ℕ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x ≠ y) (eq : 1/x + 1/y = 1/10) : x + y = 45 :=
by
  sorry

end smallest_sum_l150_150602


namespace scientific_notation_correct_l150_150782

-- Define the given condition
def average_daily_users : ℝ := 2590000

-- The proof problem
theorem scientific_notation_correct :
  average_daily_users = 2.59 * 10^6 :=
sorry

end scientific_notation_correct_l150_150782


namespace factorial_power_comparison_l150_150131

theorem factorial_power_comparison :
  (nat.factorial 1000) ^ 2 ≥ 1000 ^ 1000 :=
sorry

end factorial_power_comparison_l150_150131


namespace optimal_permutation_minimizes_K_l150_150021

variables {n : ℕ} (p T : Fin n → ℝ) (perm : List (Fin n))

-- Define expected search time function K
def K (perm : List (Fin n)) : ℝ :=
  perm.foldl (λ acc j => acc + p j * (perm.take (perm.indexOf j + 1)).sum (λ i => T i)) 0

-- Define a function to calculate the ratio list of p / T
def ratio (i : Fin n) : ℝ := p i / T i

-- Define the optimal permutation condition
def is_optimal_perm (perm : List (Fin n)) : Prop :=
  perm.sorted (λ i j => ratio i ≥ ratio j)

-- State the main theorem
theorem optimal_permutation_minimizes_K :
  ∃ perm : List (Fin n), is_optimal_perm p T perm ∧ ∀ perm', K p T perm ≤ K p T perm' :=
sorry

end optimal_permutation_minimizes_K_l150_150021


namespace compute_8_pow_neg_two_thirds_l150_150114

theorem compute_8_pow_neg_two_thirds : 8^(-2/3 : ℝ) = 1/4 :=
by
  sorry

end compute_8_pow_neg_two_thirds_l150_150114


namespace prove_new_ratio_l150_150342

noncomputable def new_ratio_of_horses_to_cows (x : ℕ) (initial_horses : ℕ) (initial_cows : ℕ) (horses_sold : ℕ) (cows_bought : ℕ) (final_horses: ℕ) (final_cows: ℕ) : Prop :=
  initial_horses = 4 * x ∧ initial_cows = x ∧ horses_sold = 15 ∧ cows_bought = 15 ∧ final_horses = initial_horses - horses_sold ∧ final_cows = initial_cows + cows_bought ∧ final_horses = final_cows + 30

theorem prove_new_ratio (x initial_horses initial_cows horses_sold cows_bought final_horses final_cows : ℕ) 
  (h : new_ratio_of_horses_to_cows x initial_horses initial_cows horses_sold cows_bought final_horses final_cows) : final_horses : final_cows = 13 : 7 := by
  sorry

end prove_new_ratio_l150_150342


namespace rectangle_existence_l150_150062

theorem rectangle_existence (grid_size : ℕ) (square_size : ℕ) (removed_squares : finset (fin grid_size × fin grid_size))
  (h_grid_size : grid_size = 2015) (h_square_size : square_size = 10) :
  (∃ rect : (fin grid_size × fin grid_size) × (fin grid_size × fin grid_size), 
    rect.2.1 - rect.1.1 = 1 ∧ rect.2.2 - rect.1.2 = square_size) ∧
  (∃ rects : fin 5 → ((fin grid_size × fin grid_size) × (fin grid_size × fin grid_size)), 
    ∀ i, rects i).2.1 - (rects i).1.1 = 1 ∧ (rects i).2.2 - (rects i).1.2 = square_size := sorry

end rectangle_existence_l150_150062


namespace distance_from_point_to_line_l150_150698

theorem distance_from_point_to_line 
  (ρ θ : ℝ) 
  (h_point : (ρ, θ) = (2, 5 * Real.pi / 6))
  (h_line : ρ * Real.sin (θ - Real.pi / 3) = 4) : 
  distance_point_to_line (ρ, θ) (2, 5 * Real.pi / 6) 4 = 2 := 
sorry

end distance_from_point_to_line_l150_150698


namespace rational_root_of_polynomial_l150_150727

theorem rational_root_of_polynomial (P Q : polynomial ℤ) (hP_nonzero : P ≠ 0) (hQ_nonzero : Q ≠ 0)
  (hdeg : P.degree > Q.degree) 
  (h_inf_rational_roots : ∃ᶠ p in filter.at_top, p.prime ∧ ∃ r : ℚ, is_root ((p : ℤ) * P + Q) r) :
  ∃ r : ℚ, is_root P r := 
sorry

end rational_root_of_polynomial_l150_150727


namespace range_of_set_l150_150085

theorem range_of_set (a b c : ℕ) (h1 : (a + b + c) / 3 = 6) (h2 : b = 6) (h3 : a = 2) : max a (max b c) - min a (min b c) = 8 :=
by
  sorry

end range_of_set_l150_150085


namespace prob_of_factors_less_than_6_l150_150023

theorem prob_of_factors_less_than_6 :
  let n := 36 in
  let factors := {x : ℕ // x > 0 ∧ n % x = 0} in
  let total_factors := (factors.count) in
  let favorable_factors := {x : ℕ // x > 0 ∧ n % x = 0 ∧ x < 6} in
  let favorable_count := (favorable_factors.count) in
  favorable_count / total_factors = 4 / 9 :=
by
  sorry

end prob_of_factors_less_than_6_l150_150023


namespace smallest_element_in_M_l150_150891

def f : ℝ → ℝ := sorry
axiom f1 (x y : ℝ) (h1 : x ≥ 1) (h2 : y = 3 * x) : f y = 3 * f x
axiom f2 (x : ℝ) (h : 1 ≤ x ∧ x ≤ 3) : f x = 1 - abs (x - 2)
axiom f99_value : f 99 = 18

theorem smallest_element_in_M : ∃ x : ℝ, x = 45 ∧ f x = 18 := by
  -- proof will be provided later
  sorry

end smallest_element_in_M_l150_150891


namespace percent_university_diploma_l150_150044

-- Definitions as per the problem conditions
def prob_no_diploma_have_job : ℝ := 0.1
def prob_have_job : ℝ := 0.2
def prob_diploma_given_no_job : ℝ := 0.25

-- The theorem to prove
theorem percent_university_diploma :
  let total_percent_with_diploma := 
    (prob_have_job - prob_no_diploma_have_job) + 
    (prob_diploma_given_no_job * (1 - prob_have_job)) in
  total_percent_with_diploma = 0.3 :=
by
  sorry

end percent_university_diploma_l150_150044


namespace final_concentration_of_milk_l150_150865

variable (x : ℝ) (total_vol : ℝ) (initial_milk : ℝ)
axiom x_value : x = 33.333333333333336
axiom total_volume : total_vol = 100
axiom initial_milk_vol : initial_milk = 36

theorem final_concentration_of_milk :
  let first_removal := x / total_vol * initial_milk
  let remaining_milk_after_first := initial_milk - first_removal
  let second_removal := x / total_vol * remaining_milk_after_first
  let final_milk := remaining_milk_after_first - second_removal
  (final_milk / total_vol) * 100 = 16 :=
by {
  sorry
}

end final_concentration_of_milk_l150_150865


namespace sin_double_angle_in_second_quadrant_l150_150985

theorem sin_double_angle_in_second_quadrant (θ : ℝ) 
  (h1 : sin⁴ θ + cos⁴ θ = 5 / 9)
  (h2 : π / 2 < θ ∧ θ < π) :
  sin (2 * θ) = -2 * sqrt 2 / 3 :=
by
  sorry

end sin_double_angle_in_second_quadrant_l150_150985


namespace cylindricalCupSpillage_correct_l150_150477

noncomputable def cylindricalCupSpillage : ℝ :=
  let r := 4 -- Radius of the base
  let h := 8 * Real.sqrt 3 -- Height of the cylinder
  let V_initial := π * r^2 * (h / 2) -- Initial volume (height halved due to spillage condition)
  let V_final := π * r^2 * (h / 2) -- Final volume (remaining water)
  ((V_initial - V_final) : ℝ)

theorem cylindricalCupSpillage_correct :
  cylindricalCupSpillage = (128 * Real.sqrt 3 * π / 3) := by
  sorry

end cylindricalCupSpillage_correct_l150_150477


namespace angle_BAC_is_pi_over_3_l150_150647

noncomputable def vector_AB : ℝ × ℝ := (-1, real.sqrt 3)
noncomputable def vector_AC : ℝ × ℝ := (1, real.sqrt 3)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

def magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2)

def cos_angle (u v : ℝ × ℝ) : ℝ :=
  dot_product u v / (magnitude u * magnitude v)

def angle_in_radians (cos_value : ℝ) : ℝ :=
  real.acos cos_value

theorem angle_BAC_is_pi_over_3 :
  ∃ θ : ℝ, θ = angle_in_radians (cos_angle vector_AB vector_AC) ∧ θ = real.pi / 3 :=
begin
  sorry,
end

end angle_BAC_is_pi_over_3_l150_150647


namespace probability_of_snowing_at_least_once_l150_150811

theorem probability_of_snowing_at_least_once (p : ℚ) (h : p = 3 / 4) :
  let q := 1 - p in
  let not_snowing_five_days := q ^ 5 in
  let at_least_once := 1 - not_snowing_five_days in
  at_least_once = 1023 / 1024 :=
by
  have q_def : q = 1 - p := rfl,
  have not_snowing_five_days_def : not_snowing_five_days = q ^ 5 := rfl,
  have at_least_once_def : at_least_once = 1 - not_snowing_five_days := rfl,
  sorry

end probability_of_snowing_at_least_once_l150_150811


namespace similar_triangles_area_ratio_l150_150992

theorem similar_triangles_area_ratio (h_ratio : ℝ) (area_ratio : ℝ) 
  (h_ratio_given : h_ratio = 1 / 2) 
  (area_ratio_def : area_ratio = h_ratio ^ 2) : 
  area_ratio = 1 / 4 :=
by
  rw [h_ratio_given, area_ratio_def]
  sorry

end similar_triangles_area_ratio_l150_150992


namespace angle_condition_necessity_l150_150315

variable (A B C : ℝ) -- Angles of the triangle
variable (h_triangle_sum : A + B + C = 180) -- Sum of angles in a triangle
variable (h_angle_condition : A - B = B - C) -- Condition q

noncomputable def one_angle_is_60 := (A = 60) ∨ (B = 60) ∨ (C = 60) -- Condition p

theorem angle_condition_necessity (h : (one_angle_is_60 A B C)) :
  ((B = 60) → (h_angle_condition A B C)) ∧ (¬((B = 60) → ¬(h_angle_condition A B C))) :=
by sorry

end angle_condition_necessity_l150_150315


namespace options_necessarily_positive_l150_150510

variable (x y z : ℝ)

theorem options_necessarily_positive (h₁ : -1 < x) (h₂ : x < 0) (h₃ : 0 < y) (h₄ : y < 1) (h₅ : 2 < z) (h₆ : z < 3) :
  y + x^2 * z > 0 ∧
  y + x^2 > 0 ∧
  y + y^2 > 0 ∧
  y + 2 * z > 0 := 
  sorry

end options_necessarily_positive_l150_150510


namespace ted_age_proof_l150_150915

theorem ted_age_proof (s t : ℝ) (h1 : t = 3 * s - 20) (h2 : t + s = 78) : t = 53.5 :=
by
  sorry  -- Proof steps are not required, hence using sorry.

end ted_age_proof_l150_150915


namespace num_integers_prod_zero_l150_150153

open Complex

-- Definition of main problem conditions
def is_prod_zero (n : ℕ) : Prop :=
  ∃ k : ℕ, k < n ∧ (1 + exp (2 * π * I * k / n))^n + I = 0

-- The main theorem to be proved
theorem num_integers_prod_zero (count : ℕ) :
  count = 503 ↔ finset.filter (λ (n : ℕ), 1 ≤ n ∧ n ≤ 2015 ∧ is_prod_zero n) (finset.range 2016).card := sorry

end num_integers_prod_zero_l150_150153


namespace number_of_students_l150_150080

theorem number_of_students (n : ℕ) (h1 : n < 60) (h2 : n % 6 = 4) (h3 : n % 8 = 5) : n = 46 := by
  sorry

end number_of_students_l150_150080


namespace olya_number_sequence_l150_150760

theorem olya_number_sequence :
  ∃ (a b c d : ℕ), (100 * a + 10 * b + a = 929) ∧
                   (2 * a + b = 2 * 10 + 0 = 20) ∧
                   (b + c = 2) :=
begin
  sorry
end

end olya_number_sequence_l150_150760


namespace original_price_of_dish_l150_150850

theorem original_price_of_dish :
  let P : ℝ := 40
  (0.9 * P + 0.15 * P) - (0.9 * P + 0.15 * 0.9 * P) = 0.60 → P = 40 := by
  intros P h
  sorry

end original_price_of_dish_l150_150850


namespace part1_part2_part3_l150_150614

open Set

variable (R : Type) [LinearOrder R]

def A := {x : R | x ≤ -1 ∨ x ≥ 3}
def B := {x : R | 1 ≤ x ∧ x ≤ 6}
def C (m : R) := {x : R | m + 1 ≤ x ∧ x ≤ 2 * m}

theorem part1 (r : R) : r ∈ A ∧ r ∈ B ↔ (3 ≤ r ∧ r ≤ 6) :=
sorry

theorem part2 (r : R) : r ∉ A ∨ r ∈ B ↔ (-1 < r ∧ r ≤ 6) :=
sorry

theorem part3 (m : R) : B ∪ C m = B → m ∈ Iic 3 :=
by
  simp only [OrdIcc, OrdIio, OrdIic, Set.mem_setOf_eq]
  exact sorry

end part1_part2_part3_l150_150614


namespace prob_with_replacement_prob_without_replacement_l150_150867

-- Define the condition for total number of red, black, and white balls
def totalBalls := 6
def redBalls := 3
def blackBalls := 2
def whiteBalls := 1

-- Define the probability calculation functions with replacement and without replacement
def probReplace (draws : List ℕ) : ℚ :=
(if (draws.head = redBalls) then (redBalls/totalBalls) else (1/2)) *
(if (draws.tail.head = redBalls) then (redBalls/totalBalls) else (1/2))

def probNoReplace (draws : List ℕ) : ℚ :=
(if (draws.head = redBalls) then (redBalls/totalBalls) else ((totalBalls - redBalls)/totalBalls)) *
(if (draws.tail.head = redBalls) then (redBalls/(totalBalls-1)) else ((totalBalls - redBalls - 1)/(totalBalls-1)))

-- Question (1): With replacement
theorem prob_with_replacement :
  probReplace [redBalls, totalBalls - redBalls] + probReplace [totalBalls - redBalls, redBalls] = 1/2 := by
sorry

-- Question (2): Without replacement
theorem prob_without_replacement :
  probNoReplace [redBalls, totalBalls - redBalls - 1] + probNoReplace [totalBalls - redBalls - 1, redBalls] = 3/5 := by
sorry

end prob_with_replacement_prob_without_replacement_l150_150867


namespace necessary_but_not_sufficient_condition_l150_150746

noncomputable def p (x : ℝ) : Prop := (1 - x^2 < 0 ∧ |x| - 2 > 0) ∨ (1 - x^2 > 0 ∧ |x| - 2 < 0)
noncomputable def q (x : ℝ) : Prop := x^2 + x - 6 > 0

theorem necessary_but_not_sufficient_condition (x : ℝ) :
  (q x → p x) ∧ ¬(p x → q x) :=
sorry

end necessary_but_not_sufficient_condition_l150_150746


namespace runs_twice_l150_150686

-- Definitions of the conditions
def game_count : ℕ := 6
def runs_one : ℕ := 1
def runs_five : ℕ := 5
def average_runs : ℕ := 4

-- Assuming the number of runs scored twice is x
variable (x : ℕ)

-- Definition of total runs scored based on the conditions
def total_runs : ℕ := runs_one + 2 * x + 3 * runs_five

-- Statement to prove the number of runs scored twice
theorem runs_twice :
  (total_runs x) / game_count = average_runs → x = 4 :=
by
  sorry

end runs_twice_l150_150686


namespace planA_text_message_cost_l150_150039

def planA_cost (x : ℝ) : ℝ := 60 * x + 9
def planB_cost : ℝ := 60 * 0.40

theorem planA_text_message_cost (x : ℝ) (h : planA_cost x = planB_cost) : x = 0.25 :=
by
  -- h represents the condition that the costs are equal
  -- The proof is skipped with sorry
  sorry

end planA_text_message_cost_l150_150039


namespace geometric_series_sum_value_l150_150031

theorem geometric_series_sum_value :
  let a : ℚ := 3 / 4
  let r : ℚ := 3 / 4
  let n : ℕ := 12
  \(\sum_{k\in \mathbb{N}, 1 \leq k \leq n} a * r^(k - 1)\) = \(\frac{48758625}{16777216}\) :=
sorry

end geometric_series_sum_value_l150_150031


namespace area_of_reflected_triangle_l150_150402

theorem area_of_reflected_triangle (AB BC : ℝ) (hAB : AB = 5) (hBC : BC = 12)
  (h_right_triangle : ∀ A B C, angle A B C = π / 2)
  (A A' B B' C C' : Type) (map_A : A → A') (map_B : B → B') (map_C : C → C')
  (h_map_A : ∀ A, reflection (angle_bisector (angle A B C)) A = A')
  (h_map_B : ∀ B, reflection (midpoint B C) B = B')
  (h_map_C : ∀ C, reflection (perpendicular_bisector (B A)) C = C') :
  area (triangle A' B' C') = 17.5 :=
sorry

end area_of_reflected_triangle_l150_150402


namespace derivative_y_wrt_x_l150_150050

noncomputable def y (t : ℝ) : ℝ := (1/2) * (Real.tan t)^2 + Real.log (Real.cos t)
noncomputable def x (t : ℝ) : ℝ := Real.log (Real.sqrt ((1 - Real.sin t) / (1 + Real.sin t)))

noncomputable def dy_dx (t : ℝ) : ℝ := (Real.sin t * Real.cos t - 1) / Real.cos t

theorem derivative_y_wrt_x (t : ℝ) :
  let y' := (Real.tan t)^3 in
  let x' := -Real.sec t in
  deriv y (y t) = deriv x (x t) → (deriv y (y t)) / (deriv x (x t)) = dy_dx t ∧ dy_dx t = (Real.sin t * Real.cos t - 1) / Real.cos t :=
sorry

end derivative_y_wrt_x_l150_150050


namespace count_of_divisibles_l150_150213

theorem count_of_divisibles (n : ℕ) (h : n < 500) : 
  (∃ k, n = 2 * 3 * 5 * 7 * k ∧ k > 0) →
  (∃! k, k = 2) :=
by {
  have LCM := 210,
  -- sorry is used to skip the proof
  sorry
}

end count_of_divisibles_l150_150213


namespace number_of_trailing_zeros_in_P_l150_150229

-- Define the product P as given in the problem
def P : ℕ := ∏ i in (finset.range 90).filter (λ k, k ≥ 11), i

-- State that the number of trailing zeros in P is 18
theorem number_of_trailing_zeros_in_P : nat.trailing_zeros(P) = 18 := by
  sorry

end number_of_trailing_zeros_in_P_l150_150229


namespace determine_Tn_l150_150151

noncomputable def geometricSequence (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r ^ n) / (1 - r)

noncomputable def sumOfSums (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  ∑ k in Finset.range (n + 1), geometricSequence a r k

theorem determine_Tn (a r : ℝ) (S20 : ℝ) : ∀ Tn : ℝ, Tn = sumOfSums a r 20 :=
by 
  sorry

end determine_Tn_l150_150151


namespace x_cubed_plus_y_cubed_l150_150668

theorem x_cubed_plus_y_cubed (x y : ℝ) (h1 : x + y = 8) (h2 : x * y = 14) : x^3 + y^3 = 176 :=
sorry

end x_cubed_plus_y_cubed_l150_150668


namespace tan_of_alpha_l150_150966

theorem tan_of_alpha 
  (α : ℝ)
  (h1 : Real.sin α = (3 / 5))
  (h2 : α ∈ Set.Ioo (π / 2) π) : Real.tan α = -3 / 4 :=
sorry

end tan_of_alpha_l150_150966


namespace beautiful_dates_in_2023_l150_150913

def is_beautiful_date (d1 d2 m1 m2 y1 y2 : ℕ) : Prop :=
  let digits := [d1, d2, m1, m2, y1, y2]
  (digits.nodup) ∧ (d1 < 10) ∧ (d2 < 10) ∧ (m1 < 10) ∧ (m2 < 10) ∧ (y1 < 10) ∧ (y2 < 10)

theorem beautiful_dates_in_2023 : ∃ n, n = 30 ∧
  n = (Σ m1 m2 d1 d2, is_beautiful_date d1 d2 m1 m2 2 0 ∧ is_beautiful_date d1 d2 m1 m2 2 3) :=
sorry

end beautiful_dates_in_2023_l150_150913


namespace sum_of_square_roots_correct_l150_150517

def numbers_between (a b : ℕ) : List ℕ :=
  List.filter (λ n => a ≤ n ∧ n ≤ b) (List.range (b + 1))

def divisible_by_seven (n : ℕ) : Prop := n % 7 = 0

def divisible_by_other_primes (n : ℕ) : Prop :=
  ∃ p : ℕ, p ∣ n ∧ (p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 11 ∨ p = 13 ∨ p = 17 ∨ p = 19 ∨ p = 23 ∨ p = 29 ∨ p = 31 ∨ p = 37 ∨ p = 41 ∨ p = 43 ∨ p = 47 ∨ p = 53)

def valid_numbers : List ℕ :=
  List.filter (λ n => divisible_by_seven n ∧ ¬divisible_by_other_primes n) (numbers_between 18 57)

def sum_of_square_roots : ℝ :=
  (List.map (λ n => Real.sqrt n.to_real) valid_numbers).sum

theorem sum_of_square_roots_correct : sum_of_square_roots = 7 := by
  sorry

end sum_of_square_roots_correct_l150_150517


namespace carrie_remaining_money_l150_150922

def initial_money : ℝ := 200
def sweater_cost : ℝ := 36
def tshirt_cost : ℝ := 12
def tshirt_discount : ℝ := 0.10
def shoes_cost : ℝ := 45
def jeans_cost : ℝ := 52
def scarf_cost : ℝ := 18
def sales_tax_rate : ℝ := 0.05

-- Calculate tshirt price after discount
def tshirt_final_price : ℝ := tshirt_cost * (1 - tshirt_discount)

-- Sum all the item costs before tax
def total_cost_before_tax : ℝ := sweater_cost + tshirt_final_price + shoes_cost + jeans_cost + scarf_cost

-- Calculate the total sales tax
def sales_tax : ℝ := total_cost_before_tax * sales_tax_rate

-- Calculate total cost after tax
def total_cost_after_tax : ℝ := total_cost_before_tax + sales_tax

-- Calculate the remaining money
def remaining_money (initial : ℝ) (total : ℝ) : ℝ := initial - total

theorem carrie_remaining_money
  (initial_money : ℝ)
  (sweater_cost : ℝ)
  (tshirt_cost : ℝ)
  (tshirt_discount : ℝ)
  (shoes_cost : ℝ)
  (jeans_cost : ℝ)
  (scarf_cost : ℝ)
  (sales_tax_rate : ℝ)
  (h₁ : initial_money = 200)
  (h₂ : sweater_cost = 36)
  (h₃ : tshirt_cost = 12)
  (h₄ : tshirt_discount = 0.10)
  (h₅ : shoes_cost = 45)
  (h₆ : jeans_cost = 52)
  (h₇ : scarf_cost = 18)
  (h₈ : sales_tax_rate = 0.05) :
  remaining_money initial_money (total_cost_after_tax) = 30.11 := 
by 
  simp only [remaining_money, total_cost_after_tax, total_cost_before_tax, tshirt_final_price, sales_tax];
  sorry

end carrie_remaining_money_l150_150922


namespace count_integers_divisible_by_2_3_5_7_l150_150225

theorem count_integers_divisible_by_2_3_5_7 :
  ∃ n : ℕ, (∀ k : ℕ, k < 500 → (k % 2 = 0 ∧ k % 3 = 0 ∧ k % 5 = 0 ∧ k % 7 = 0) → k ≠ n → k < 500 ∧ k > 0) ∧
  (n = 2) :=
by
  sorry

end count_integers_divisible_by_2_3_5_7_l150_150225


namespace trig_identity_l150_150364

theorem trig_identity : (sin 40 + sin 80) / (cos 40 + cos 80) = real.sqrt 3 :=
by
  sorry

end trig_identity_l150_150364


namespace cost_effective_purchase_is_10_cartons_l150_150499

noncomputable def costPerPackAfterDiscounts 
    (numCartons: ℕ) (initialCostPerDozenCartons: ℝ) 
    (quantityDiscount: ℝ) (membershipDiscount: ℝ) 
    (seasonalDiscount: ℝ) (boxesPerCarton: ℕ) (packsPerBox: ℕ): ℝ :=
let costPerCarton := initialCostPerDozenCartons / 12
let totalDiscount := quantityDiscount + membershipDiscount + seasonalDiscount
let discountedCostPerCarton := costPerCarton * (1 - totalDiscount)
let packsPerCarton := boxesPerCarton * packsPerBox
discountedCostPerCarton / packsPerCarton

def minimizedCostEffectivePurchase 
    (initialCostPerDozenCartons: ℝ) (boxesPerCarton: ℕ) (packsPerBox: ℕ) 
    (quantityDiscount5: ℝ) (quantityDiscount10: ℝ) 
    (membershipDiscountGold: ℝ) (seasonalDiscount: ℝ): ℕ × ℝ :=
let cost5Cartons := costPerPackAfterDiscounts 5 initialCostPerDozenCartons quantityDiscount5 membershipDiscountGold seasonalDiscount boxesPerCarton packsPerBox
let cost10Cartons := costPerPackAfterDiscounts 10 initialCostPerDozenCartons quantityDiscount10 membershipDiscountGold seasonalDiscount boxesPerCarton packsPerBox
if cost5Cartons < cost10Cartons then (5, cost5Cartons) else (10, cost10Cartons)

theorem cost_effective_purchase_is_10_cartons:
    minimizedCostEffectivePurchase 3000 15 12 0.10 0.15 0.10 0.03 = (10, 1.00) :=
by 
  sorry

end cost_effective_purchase_is_10_cartons_l150_150499


namespace number_of_valid_pairs_l150_150143

-- Definitions based on problem conditions
def no_zero_digit (n : ℕ) : Prop := ∀ d ∈ n.digits 10, d ≠ 0

def valid_pair (a b : ℕ) : Prop := no_zero_digit a ∧ no_zero_digit b

-- Statement of the proof problem
theorem number_of_valid_pairs :
  { (a, b) : ℕ × ℕ | a + b = 1000 ∧ valid_pair a b }.to_finset.card = 738 :=
by {
  sorry
}

end number_of_valid_pairs_l150_150143


namespace find_y_value_l150_150267

/-- Given angles and conditions, find the value of y in the geometric figure. -/
theorem find_y_value
  (AB_parallel_DC : true) -- AB is parallel to DC
  (ACE_straight_line : true) -- ACE is a straight line
  (angle_ACF : ℝ := 130) -- ∠ACF = 130°
  (angle_CBA : ℝ := 60) -- ∠CBA = 60°
  (angle_ACB : ℝ := 100) -- ∠ACB = 100°
  (angle_ADC : ℝ := 125) -- ∠ADC = 125°
  : 35 = 35 := -- y = 35°
by
  sorry

end find_y_value_l150_150267


namespace smallest_four_digit_number_divisible_by_12_with_two_even_two_odd_l150_150836

theorem smallest_four_digit_number_divisible_by_12_with_two_even_two_odd :
  ∃ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ (∃ (d1 d2 d3 d4 : ℕ), 
  n = d1 * 1000 + d2 * 100 + d3 * 10 + d4 ∧ 
  (d1 + d2 + d3 + d4) % 3 = 0 ∧ 
  (d3 * 10 + d4) % 4 = 0 ∧ 
  even d1 ∧ even d3 ∧ odd d2 ∧ odd d4 ∧
  ∀ m : ℕ, m ≥ 1000 ∧ m < 10000 ∧ (∃ (e1 e2 e3 e4 : ℕ),
  m = e1 * 1000 + e2 * 100 + e3 * 10 + e4 ∧
  (e1 + e2 + e3 + e4) % 3 = 0 ∧
  (e3 * 10 + e4) % 4 = 0 ∧
  even e1 ∧ even e3 ∧ odd e2 ∧ odd e4) → m ≥ 1218) := sorry

end smallest_four_digit_number_divisible_by_12_with_two_even_two_odd_l150_150836


namespace circumscribed_quadrilateral_converse_arithmetic_progression_l150_150076

theorem circumscribed_quadrilateral (a b c d : ℝ) (k : ℝ) (h1 : b = a + k) (h2 : d = a + 2 * k) (h3 : c = a + 3 * k) :
  a + c = b + d :=
by
  sorry

theorem converse_arithmetic_progression (a b c d : ℝ) (h : a + c = b + d) :
  ∃ k : ℝ, b = a + k ∧ d = a + 2 * k ∧ c = a + 3 * k :=
by
  sorry

end circumscribed_quadrilateral_converse_arithmetic_progression_l150_150076


namespace factory_correct_decision_prob_l150_150877

theorem factory_correct_decision_prob:
  let p := 0.8 in
  let q := 1 - p in
  let n := 3 in
  let correct_two_consultants := (Nat.choose n 2) * p^2 * q in
  let correct_three_consultants := (Nat.choose n 3) * p^3 in
  let probability_correct_decision := correct_two_consultants + correct_three_consultants in
  probability_correct_decision = 0.896 :=
by 
  sorry

end factory_correct_decision_prob_l150_150877


namespace point_in_first_quadrant_l150_150675

noncomputable def z : ℂ := (2 + 3 * complex.i) / (1 + complex.i)

theorem point_in_first_quadrant (h : z + z * complex.i = 2 + 3 * complex.i) : 
  (z.re > 0) ∧ (z.im > 0) :=
by sorry

end point_in_first_quadrant_l150_150675


namespace smallest_product_among_l1_l2_l3_l150_150455

theorem smallest_product_among_l1_l2_l3
  (a b c : ℝ)
  (h1 : a > b)
  (h2 : b > c)
  (h3 : c > 0) :
  let l1 := Real.sqrt ((a+c)^2 + b^2),
      l2 := Real.sqrt (a^2 + (b+c)^2),
      l3 := Real.sqrt ((a+b)^2 + c^2) in
  min (min (min (l1 * l2) (l1 * l3)) (min (l2 * l3) (l2^2))) (l3^2) = l2^2 :=
by
  sorry

end smallest_product_among_l1_l2_l3_l150_150455


namespace alpha_beta_sum_eq_l150_150629

theorem alpha_beta_sum_eq (a : ℝ) (h : 1 < a) (α β : ℝ) 
  (hα : α ∈ Set.Ioo (-π / 2) (π / 2))
  (hβ : β ∈ Set.Ioo (-π / 2) (π / 2)) 
  (h_roots : (∀ x, x^2 + 3 * a * x + (3 * a + 1) = 0 → x = Real.tan α ∨ x = Real.tan β)) :
  α + β = -3 * π / 4 := 
sorry

end alpha_beta_sum_eq_l150_150629


namespace number_of_valid_sets_l150_150163

theorem number_of_valid_sets : 
  (∃ (S : Finset ℕ), S ⊆ Finset.mk [1, 2, 3, 4, 5] (by simp) ∧ (∀ a ∈ S, 6 - a ∈ S)) → 
  (Finset.card { S : Finset ℕ | S ⊆ Finset.mk [1, 2, 3, 4, 5] (by simp) ∧ (∀ a ∈ S, 6 - a ∈ S) } = 7) := 
sorry

end number_of_valid_sets_l150_150163


namespace correct_calculation_l150_150444

-- Define the conditions as hypotheses
variables (x : ℝ) -- Here we use ℝ to signify real numbers, but this also applies to any field.

-- Define the theorem to prove option D is the correct calculation
theorem correct_calculation : 
  (x^2 / x^5 = x^(-3)) ∧ (x^(-3) = 1 / x^3) :=
by {
  split,
  { calc x^2 / x^5 = x^(2 - 5) : by rw [div_eq_mul_inv, ←pow_sub x 2 5]
                    ... = x^(-3) : by norm_num },
  { calc x^(-3) = 1 / x^3 : by rw [inv_eq_one_div x^3, pow_neg] }
}.

end correct_calculation_l150_150444


namespace interval_monotonicity_and_extremum_range_of_a_l150_150635

noncomputable theory

-- Conditions
def f (x : ℝ) : ℝ := (Real.log x) / x
def g (x : ℝ) (a : ℝ) : ℝ := x * (Real.log x) - a * (x^2 - 1)

-- Questions
theorem interval_monotonicity_and_extremum :
  (∀ x, (0 < x ∧ x < Real.exp 1 → 0 < (1 - Real.log x) / x^2) ∧
       (Real.exp 1 < x → (1 - Real.log x) / x^2 < 0)) ∧
  (Real.exp 1 = 1 ∧ ∀ x, x ≠ Real.exp 1 → f (Real.exp 1) > f x) :=
sorry

theorem range_of_a (a : ℝ) :
  (∀ x, x ≥ 1 → f x ≤ a * (1 - 1 / x^2)) ↔ a ∈ Set.Ici (1/2) :=
sorry

end interval_monotonicity_and_extremum_range_of_a_l150_150635


namespace average_is_correct_l150_150047

def numbers : List ℕ := [1200, 1300, 1400, 1510, 1520, 1530, 1200]

def sum_of_numbers : ℕ := numbers.sum
def count_of_numbers : ℕ := numbers.length
def average_of_numbers : ℚ := sum_of_numbers / count_of_numbers

theorem average_is_correct : average_of_numbers = 1380 := 
by 
  -- Here, you would normally put the proof steps.
  sorry

end average_is_correct_l150_150047


namespace cost_of_baseball_is_correct_l150_150324

-- Define the costs and total amount spent
def cost_of_marbles : ℝ := 9.05
def cost_of_football : ℝ := 4.95
def total_amount_spent : ℝ := 20.52

-- Define the cost of the baseball
def cost_of_baseball : ℝ := total_amount_spent - (cost_of_marbles + cost_of_football)

-- The theorem we want to prove
theorem cost_of_baseball_is_correct :
  cost_of_baseball = 6.52 := by
  sorry

end cost_of_baseball_is_correct_l150_150324


namespace abs_ineq_range_m_l150_150960

theorem abs_ineq_range_m :
  ∀ m : ℝ, (∀ x : ℝ, |x - 1| + |x + 2| ≥ m) ↔ m ≤ 3 :=
by
  sorry

end abs_ineq_range_m_l150_150960


namespace smallest_positive_four_digit_integer_equivalent_to_4_mod_5_l150_150835

theorem smallest_positive_four_digit_integer_equivalent_to_4_mod_5 :
  ∃ n : ℕ, n ≡ 4 [MOD 5] ∧ n ≥ 1000 ∧ n = 1004 := 
begin
  use 1004,
  split,
  { norm_num, },
  split,
  { norm_num, },
  { refl, },
end

end smallest_positive_four_digit_integer_equivalent_to_4_mod_5_l150_150835


namespace difference_of_roots_of_quadratic_l150_150531

theorem difference_of_roots_of_quadratic (p q : ℝ) (h : p ≠ q) :
  let r1 := (p + q) / 2 + ((p - q)^2 - 4*p*q)^(1/2) / 2,
      r2 := (p + q) / 2 - ((p - q)^2 - 4*p*q)^(1/2) / 2
  in r1 - r2 = |p - q| :=
by
  sorry

end difference_of_roots_of_quadratic_l150_150531


namespace find_equation_length_range_l150_150396

open Real

variables {a b : ℝ}

-- Conditions
def focus1 := (-1 : ℝ, 0)
def focus2 := (1 : ℝ, 0)
def ellipse (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1
def angle_condition (A B : ℝ × ℝ) (l : ℝ) : Prop := ∃ x y, (A = (x, y) ∧ B = (l, y)) ∧ (A.1 = l ∧ B.1 = l) ∧ (3 * l * A.2 = 2 * (l * (B.2)))

-- Questions
theorem find_equation (h₁ : a^2 - b^2 = 1) (h₂ : a > 0) (h₃ : b > 0) (h₄ : ∀ A B, angle_condition A B focus2) :
  ellipse x y = (x^2 / 3 + y^2/2 = 1) :=
sorry

theorem length_range {λ : ℝ} (h_λ : 1 ≤ λ ∧ λ ≤ 2) :
  ∃ M : ℝ × ℝ, λ ∈ (λ_min, λ_max) ↔ (λ_min = sqrt(51)/4 ∧ λ_max = 2) :=
sorry

end find_equation_length_range_l150_150396


namespace max_min_PA_distance_l150_150188

theorem max_min_PA_distance
  (C_param : ∀ θ : ℝ, 4 * (x θ / 3)^2 + (y θ / 4)^2 = 1)
  (l_param : ∀ t : ℝ, (x (3 + t), y (5 - 2 * t)) ) :
  ∃ (maxPA : ℝ) (minPA : ℝ), 
    maxPA = 32 * Real.sqrt 5 / 5 ∧ minPA = 12 * Real.sqrt 5 / 5 :=
by
  sorry

end max_min_PA_distance_l150_150188


namespace tea_in_each_box_initially_l150_150007

theorem tea_in_each_box_initially (x : ℕ) 
  (h₁ : 4 * (x - 9) = x) : 
  x = 12 := 
sorry

end tea_in_each_box_initially_l150_150007


namespace special_integers_count_l150_150516

open Nat

def proper_divisors (n : ℕ) : List ℕ := (List.range (n - 1)).filter (λ d, d + 1 ∣ n)

def g (n : ℕ) : ℕ := (proper_divisors n).prod

def is_prime (n : ℕ) : Prop := 2 ≤ n ∧ ∀ m < n, m ∣ n → m = 1

def prime_squares (n : ℕ) : Prop := ∃ p, is_prime p ∧ p^2 = n

def not_divides_g (n : ℕ) : Prop := ¬ (n ∣ g n)

def count_special_integers : ℕ := 
  (List.range' 2 99).filter (λ n, not_divides_g n).length

theorem special_integers_count : count_special_integers = 29 := sorry

end special_integers_count_l150_150516


namespace correct_calculation_l150_150443

theorem correct_calculation (a b : ℝ) : 
  (¬ (2 * (a - 1) = 2 * a - 1)) ∧ 
  (3 * a^2 - 2 * a^2 = a^2) ∧ 
  (¬ (3 * a^2 - 2 * a^2 = 1)) ∧ 
  (¬ (3 * a + 2 * b = 5 * a * b)) :=
by
  sorry

end correct_calculation_l150_150443


namespace interval_increasing_l150_150182

open Set

variable {f : ℝ → ℝ}

-- Given conditions
def is_even (f : ℝ → ℝ) := ∀ x, f(x) = f(-x)
def is_increasing_on (f : ℝ → ℝ) (s : Set ℝ) := ∀ ⦃x y⦄, x ∈ s → y ∈ s → x < y → f x < f y
def is_decreasing_on (f : ℝ → ℝ) (s : Set ℝ) := ∀ ⦃x y⦄, x ∈ s → y ∈ s → x < y → f y < f x

-- Mathematical proof problem
theorem interval_increasing (heven : is_even f)
                            (hinc : is_increasing_on f (Ioo 2 6)) :
  is_increasing_on (fun x => f (2 - x)) (Ioo 4 8) :=
by
  sorry

end interval_increasing_l150_150182


namespace find_h_l150_150197

theorem find_h (h : ℝ) :
  (∃ x : ℝ, 3 ≤ x ∧ x ≤ 7 ∧ -(x - h)^2 = -1) → (h = 2 ∨ h = 8) :=
by sorry

end find_h_l150_150197


namespace sin_cos_combination_l150_150994

theorem sin_cos_combination (x y r : ℝ) (h1 : x = 4) (h2 : y = -3) (h3 : r = 5) (h4 : x^2 + y^2 = r^2) :
  2 * sin ⟨x, y, h4⟩ + cos ⟨x, y, h4⟩ = -2 / 5 :=
by
  sorry

end sin_cos_combination_l150_150994


namespace train_length_approx_l150_150503

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

end train_length_approx_l150_150503


namespace sum_and_num_of_factors_eq_1767_l150_150972

theorem sum_and_num_of_factors_eq_1767 (n : ℕ) (σ d : ℕ → ℕ) :
  (σ n + d n = 1767) → 
  ∃ m : ℕ, σ m + d m = 1767 :=
by 
  sorry

end sum_and_num_of_factors_eq_1767_l150_150972


namespace count_valid_four_digit_numbers_l150_150905

-- Define a four-digit number
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

-- Define a function to extract the units digit of a number
def units_digit (n : ℕ) : ℕ := n % 10

-- Define a function to extract the thousands digit of a number
def thousands_digit (n : ℕ) : ℕ := (n / 1000) % 10

-- Define the condition that the absolute difference between the units and thousands digit is 3
def abs_diff_is_three (n : ℕ) : Prop := 
  | (units_digit n : ℤ) - (thousands_digit n : ℤ) | = 3

-- Main theorem statement
theorem count_valid_four_digit_numbers : 
  finset.card (finset.filter abs_diff_is_three (finset.Icc 1000 9999)) = 728 := 
sorry

end count_valid_four_digit_numbers_l150_150905


namespace lawn_remains_l150_150752

def mary_rate (h : ℝ) : ℝ := 1 / 5
def tom_rate (h : ℝ) : ℝ := 1 / 6
def tom_work_time (t : ℝ) : ℝ := 3
def tom_work_done (t : ℝ) : ℝ := tom_rate t * tom_work_time t

theorem lawn_remains : 
  tom_work_done 3 = 1 / 2 → 
  1 - tom_work_done 3 = 1 / 2 :=
by
  intro h
  rw h
  norm_num
  sorry

end lawn_remains_l150_150752


namespace unique_monotonic_involution_l150_150770

theorem unique_monotonic_involution :
  ∃! (f : ℝ → ℝ), (∀ x y : ℝ, x ≤ y → f(x) ≤ f(y)) ∧ (∀ x : ℝ, f(f(x)) = x) ∧ (∀ x : ℝ, f(x) = x) :=
by
  sorry

end unique_monotonic_involution_l150_150770


namespace salad_proof_problem_l150_150916

def initial_conditions :=
 let mushrooms := 6
 let cherry_tomatoes := 3 * mushrooms
 let pickles := (1 / 2) * (mushrooms + cherry_tomatoes)
 let bacon_bits := 5 * pickles
 let croutons := (1 / 4) * bacon_bits
 let olives := 2 * pickles
 let cheese_cubes := 8
 let adjusted_cherry_tomatoes := cherry_tomatoes - 3
 let red_bacon_bits := (1 / 5) * bacon_bits
 let green_olives := (3 / 4) * olives
 let green_olives_condition := green_olives = (2 / 3) * adjusted_cherry_tomatoes
 ⟨mushrooms, cherry_tomatoes, pickles, bacon_bits, croutons, olives, cheese_cubes, adjusted_cherry_tomatoes, red_bacon_bits, green_olives, green_olives_condition⟩

theorem salad_proof_problem :
  let ic := initial_conditions in
  ic.red_bacon_bits = 12 ∧ ic.green_olives = 18 ∧ ic.cheese_cubes = 8 := 
sorry

end salad_proof_problem_l150_150916

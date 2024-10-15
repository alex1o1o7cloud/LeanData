import Mathlib

namespace NUMINAMATH_GPT_april_roses_l2032_203211

theorem april_roses (price_per_rose earnings number_of_roses_left : ℕ) 
  (h1 : price_per_rose = 7) 
  (h2 : earnings = 35) 
  (h3 : number_of_roses_left = 4) : 
  (earnings / price_per_rose + number_of_roses_left) = 9 :=
by
  sorry

end NUMINAMATH_GPT_april_roses_l2032_203211


namespace NUMINAMATH_GPT_find_a_from_conditions_l2032_203276

theorem find_a_from_conditions
  (a b : ℤ)
  (h₁ : 2584 * a + 1597 * b = 0)
  (h₂ : 1597 * a + 987 * b = -1) :
  a = 1597 :=
by sorry

end NUMINAMATH_GPT_find_a_from_conditions_l2032_203276


namespace NUMINAMATH_GPT_ratio_of_a_to_b_l2032_203232

-- Given conditions
variables {a b x : ℝ}
-- a and b are positive real numbers distinct from 1
variables (h1 : a > 0) (h2 : a ≠ 1) (h3 : b > 0) (h4 : b ≠ 1)
-- Given equation involving logarithms
variables (h5 : 5 * (Real.log x / Real.log a) ^ 2 + 7 * (Real.log x / Real.log b) ^ 2 = 10 * (Real.log x) ^ 2)

-- Prove that the ratio of a to b is a^(sqrt(7/5))
theorem ratio_of_a_to_b (h1 : a > 0) (h2 : a ≠ 1) (h3 : b > 0) (h4 : b ≠ 1) (h5 : 5 * (Real.log x / Real.log a) ^ 2 + 7 * (Real.log x / Real.log b) ^ 2 = 10 * (Real.log x) ^ 2) :
  b = a ^ Real.sqrt (7 / 5) :=
sorry

end NUMINAMATH_GPT_ratio_of_a_to_b_l2032_203232


namespace NUMINAMATH_GPT_find_starting_number_of_range_l2032_203215

theorem find_starting_number_of_range :
  ∃ n : ℕ, ∀ k : ℕ, k < 7 → (n + k * 9) ∣ 9 ∧ (n + k * 9) ≤ 97 ∧ (∀ m < k, (n + m * 9) < n + (m + 1) * 9) := 
sorry

end NUMINAMATH_GPT_find_starting_number_of_range_l2032_203215


namespace NUMINAMATH_GPT_evaluate_expression_l2032_203281

theorem evaluate_expression :
  -(12 * 2) - (3 * 2) + ((-18 / 3) * -4) = -6 := 
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2032_203281


namespace NUMINAMATH_GPT_minimum_minutes_for_planB_cheaper_l2032_203288

-- Define the costs for Plan A and Plan B as functions of minutes
def planACost (x : Nat) : Nat := 1500 + 12 * x
def planBCost (x : Nat) : Nat := 3000 + 6 * x

-- Statement to prove
theorem minimum_minutes_for_planB_cheaper : 
  ∃ x : Nat, (planBCost x < planACost x) ∧ ∀ y : Nat, y < x → planBCost y ≥ planACost y :=
by
  sorry

end NUMINAMATH_GPT_minimum_minutes_for_planB_cheaper_l2032_203288


namespace NUMINAMATH_GPT_replace_asterisks_l2032_203274

theorem replace_asterisks (x : ℕ) (h : (x / 20) * (x / 180) = 1) : x = 60 := by
  sorry

end NUMINAMATH_GPT_replace_asterisks_l2032_203274


namespace NUMINAMATH_GPT_stacy_days_to_complete_paper_l2032_203252

def total_pages : ℕ := 66
def pages_per_day : ℕ := 11

theorem stacy_days_to_complete_paper :
  total_pages / pages_per_day = 6 := by
  sorry

end NUMINAMATH_GPT_stacy_days_to_complete_paper_l2032_203252


namespace NUMINAMATH_GPT_simplify_expression_l2032_203200

theorem simplify_expression : (625:ℝ)^(1/4) * (256:ℝ)^(1/2) = 80 := 
by 
  sorry

end NUMINAMATH_GPT_simplify_expression_l2032_203200


namespace NUMINAMATH_GPT_john_less_than_david_by_4_l2032_203291

/-
The conditions are:
1. Zachary did 51 push-ups.
2. David did 22 more push-ups than Zachary.
3. John did 69 push-ups.

We need to prove that John did 4 push-ups less than David.
-/

def zachary_pushups : ℕ := 51
def david_pushups : ℕ := zachary_pushups + 22
def john_pushups : ℕ := 69

theorem john_less_than_david_by_4 :
  david_pushups - john_pushups = 4 :=
by
  -- Proof goes here.
  sorry

end NUMINAMATH_GPT_john_less_than_david_by_4_l2032_203291


namespace NUMINAMATH_GPT_geometric_progression_condition_l2032_203221

noncomputable def condition_for_geometric_progression (a q : ℝ) (n p : ℤ) : Prop :=
  ∃ m : ℤ, a = q^m

theorem geometric_progression_condition (a q : ℝ) (n p k : ℤ) :
  condition_for_geometric_progression a q n p ↔ a * q^(n + p) = a * q^k :=
by
  sorry

end NUMINAMATH_GPT_geometric_progression_condition_l2032_203221


namespace NUMINAMATH_GPT_proposition_not_true_3_l2032_203203

theorem proposition_not_true_3 (P : ℕ → Prop) (h1 : ∀ n, P n → P (n + 1)) (h2 : ¬ P 4) : ¬ P 3 :=
by
  sorry

end NUMINAMATH_GPT_proposition_not_true_3_l2032_203203


namespace NUMINAMATH_GPT_pencils_cost_proportion_l2032_203217

/-- 
If a set of 15 pencils costs 9 dollars and the price of the set is directly 
proportional to the number of pencils it contains, then the cost of a set of 
35 pencils is 21 dollars.
--/
theorem pencils_cost_proportion :
  ∀ (p : ℕ), (∀ n : ℕ, n * 9 = p * 15) -> (35 * 9 = 21 * 15) :=
by
  intro p h1
  sorry

end NUMINAMATH_GPT_pencils_cost_proportion_l2032_203217


namespace NUMINAMATH_GPT_triangle_sides_inequality_l2032_203297

theorem triangle_sides_inequality (a b c : ℝ) (h₁ : a + b > c) (h₂ : b + c > a) (h₃ : c + a > b) :
    (a/(b + c - a) + b/(c + a - b) + c/(a + b - c)) ≥ ((b + c - a)/a + (c + a - b)/b + (a + b - c)/c) ∧
    ((b + c - a)/a + (c + a - b)/b + (a + b - c)/c) ≥ 3 :=
by
  sorry

end NUMINAMATH_GPT_triangle_sides_inequality_l2032_203297


namespace NUMINAMATH_GPT_sum_of_solutions_of_fx_eq_0_l2032_203293

noncomputable def f : ℝ → ℝ
| x => if x ≤ 1 then 7 * x + 10 else 3 * x - 15

theorem sum_of_solutions_of_fx_eq_0 :
  let x1 := -10 / 7
  let x2 := 5
  f x1 = 0 ∧ f x2 = 0 ∧ x1 ≤ 1 ∧ x2 > 1 → x1 + x2 = 25 / 7 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_solutions_of_fx_eq_0_l2032_203293


namespace NUMINAMATH_GPT_find_range_of_a_l2032_203266

def have_real_roots (a : ℝ) : Prop := a^2 - 16 ≥ 0

def is_increasing_on_interval (a : ℝ) : Prop := a ≥ -12

theorem find_range_of_a (a : ℝ) : ((have_real_roots a ∨ is_increasing_on_interval a) ∧ ¬(have_real_roots a ∧ is_increasing_on_interval a)) → (a < -12 ∨ (-4 < a ∧ a < 4)) :=
by 
  sorry

end NUMINAMATH_GPT_find_range_of_a_l2032_203266


namespace NUMINAMATH_GPT_find_a_l2032_203257

noncomputable def triangle_side (a b c : ℝ) (A : ℝ) (area : ℝ) : ℝ :=
if b + c = 2 * Real.sqrt 3 ∧ A = Real.pi / 3 ∧ area = Real.sqrt 3 / 2 then
  Real.sqrt (b^2 + c^2 - 2 * b * c * Real.cos A)
else 0

theorem find_a (b c : ℝ) (h1 : b + c = 2 * Real.sqrt 3) (h2 : Real.cos (Real.pi / 3) = 1 / 2) (area : ℝ)
  (h3 : area = Real.sqrt 3 / 2)
  (a := triangle_side (Real.sqrt 6) b c (Real.pi / 3) (Real.sqrt 3 / 2)) :
  a = Real.sqrt 6 :=
sorry

end NUMINAMATH_GPT_find_a_l2032_203257


namespace NUMINAMATH_GPT_find_A_n_find_d1_d2_zero_l2032_203286

-- Defining the arithmetic sequences {a_n} and {b_n} with common differences d1 and d2 respectively
variables (a b : ℕ → ℤ)
variables (d1 d2 : ℤ)

-- Conditions on the sequences
axiom a_n_arith : ∀ n, a (n + 1) = a n + d1
axiom b_n_arith : ∀ n, b (n + 1) = b n + d2

-- Definitions of A_n and B_n
def A_n (n : ℕ) : ℤ := a n + b n
def B_n (n : ℕ) : ℤ := a n * b n

-- Given initial conditions
axiom A_1 : A_n a b 1 = 1
axiom A_2 : A_n a b 2 = 3

-- Prove that A_n = 2n - 1
theorem find_A_n : ∀ n, A_n a b n = 2 * n - 1 :=
by sorry

-- Condition that B_n is an arithmetic sequence
axiom B_n_arith : ∀ n, B_n a b (n + 1) - B_n a b n = B_n a b 1 - B_n a b 0

-- Prove that d1 * d2 = 0
theorem find_d1_d2_zero : d1 * d2 = 0 :=
by sorry

end NUMINAMATH_GPT_find_A_n_find_d1_d2_zero_l2032_203286


namespace NUMINAMATH_GPT_TrishulPercentageLessThanRaghu_l2032_203204

-- Define the variables and conditions
variables (R T V : ℝ)

-- Raghu's investment is Rs. 2200
def RaghuInvestment := (R : ℝ) = 2200

-- Vishal invested 10% more than Trishul
def VishalInvestment := (V : ℝ) = 1.10 * T

-- Total sum of investments is Rs. 6358
def TotalInvestment := R + T + V = 6358

-- Define the proof statement
theorem TrishulPercentageLessThanRaghu (R_is_2200 : RaghuInvestment R) 
    (V_is_10_percent_more : VishalInvestment V T) 
    (total_sum_is_6358 : TotalInvestment R T V) : 
  ((2200 - T) / 2200) * 100 = 10 :=
sorry

end NUMINAMATH_GPT_TrishulPercentageLessThanRaghu_l2032_203204


namespace NUMINAMATH_GPT_binom_divisible_by_4_l2032_203207

theorem binom_divisible_by_4 (n : ℕ) : (n ≠ 0) ∧ (¬ (∃ k : ℕ, n = 2^k)) ↔ 4 ∣ n * (Nat.choose (2 * n) n) :=
by
  sorry

end NUMINAMATH_GPT_binom_divisible_by_4_l2032_203207


namespace NUMINAMATH_GPT_negation_proof_l2032_203259

theorem negation_proof :
  (¬ ∀ x : ℝ, x < 0 → 1 - x > Real.exp x) ↔ (∃ x_0 : ℝ, x_0 < 0 ∧ 1 - x_0 ≤ Real.exp x_0) :=
by
  sorry

end NUMINAMATH_GPT_negation_proof_l2032_203259


namespace NUMINAMATH_GPT_probability_at_most_one_red_light_l2032_203225

def probability_of_no_red_light (p : ℚ) (n : ℕ) : ℚ := (1 - p) ^ n

def probability_of_exactly_one_red_light (p : ℚ) (n : ℕ) : ℚ :=
  (n.choose 1) * p ^ 1 * (1 - p) ^ (n - 1)

theorem probability_at_most_one_red_light (p : ℚ) (n : ℕ) (h : p = 1/3 ∧ n = 4) :
  probability_of_no_red_light p n + probability_of_exactly_one_red_light p n = 16 / 27 :=
by
  rw [h.1, h.2]
  sorry

end NUMINAMATH_GPT_probability_at_most_one_red_light_l2032_203225


namespace NUMINAMATH_GPT_width_of_crate_l2032_203227

theorem width_of_crate
  (r : ℝ) (h : ℝ) (w : ℝ)
  (h_crate : h = 6 ∨ h = 10 ∨ w = 6 ∨ w = 10)
  (r_tank : r = 4)
  (height_longest_crate : h > w)
  (maximize_volume : ∃ d : ℝ, d = 2 * r ∧ w = d) :
  w = 8 := 
sorry

end NUMINAMATH_GPT_width_of_crate_l2032_203227


namespace NUMINAMATH_GPT_strawb_eaten_by_friends_l2032_203290

theorem strawb_eaten_by_friends (initial_strawberries remaining_strawberries eaten_strawberries : ℕ) : 
  initial_strawberries = 35 → 
  remaining_strawberries = 33 → 
  eaten_strawberries = initial_strawberries - remaining_strawberries → 
  eaten_strawberries = 2 := 
by 
  intros h₁ h₂ h₃
  rw [h₁, h₂] at h₃
  exact h₃

end NUMINAMATH_GPT_strawb_eaten_by_friends_l2032_203290


namespace NUMINAMATH_GPT_max_a3_b3_c3_d3_l2032_203296

-- Define that a, b, c, d are real numbers that satisfy the given conditions.
theorem max_a3_b3_c3_d3 (a b c d : ℝ) 
  (h1 : a^2 + b^2 + c^2 + d^2 = 16)
  (h2 : a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧ a ≠ c ∧ b ≠ d) :
  a^3 + b^3 + c^3 + d^3 ≤ 64 :=
sorry

end NUMINAMATH_GPT_max_a3_b3_c3_d3_l2032_203296


namespace NUMINAMATH_GPT_common_ratio_geometric_sequence_l2032_203210

theorem common_ratio_geometric_sequence (a b c d : ℤ) (h1 : a = 10) (h2 : b = -20) (h3 : c = 40) (h4 : d = -80) :
    b / a = -2 ∧ c = b * -2 ∧ d = c * -2 := by
  sorry

end NUMINAMATH_GPT_common_ratio_geometric_sequence_l2032_203210


namespace NUMINAMATH_GPT_find_youngest_age_l2032_203256

noncomputable def youngest_child_age 
  (meal_cost_mother : ℝ) 
  (meal_cost_per_year : ℝ) 
  (total_bill : ℝ) 
  (triplets_count : ℕ) := 
  {y : ℝ // 
    (∃ t : ℝ, 
      meal_cost_mother + meal_cost_per_year * (triplets_count * t + y) = total_bill ∧ y = 2 ∨ y = 5)}

theorem find_youngest_age : 
  youngest_child_age 3.75 0.50 12.25 3 := 
sorry

end NUMINAMATH_GPT_find_youngest_age_l2032_203256


namespace NUMINAMATH_GPT_cube_volume_given_surface_area_l2032_203231

theorem cube_volume_given_surface_area (SA : ℝ) (a V : ℝ) (h : SA = 864) (h1 : 6 * a^2 = SA) (h2 : V = a^3) : 
  V = 1728 := 
by 
  sorry

end NUMINAMATH_GPT_cube_volume_given_surface_area_l2032_203231


namespace NUMINAMATH_GPT_sum_of_fractions_l2032_203250

-- Definitions (Conditions)
def frac1 : ℚ := 5 / 13
def frac2 : ℚ := 9 / 11

-- Theorem (Equivalent Proof Problem)
theorem sum_of_fractions : frac1 + frac2 = 172 / 143 := 
by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_sum_of_fractions_l2032_203250


namespace NUMINAMATH_GPT_cost_of_pet_snake_l2032_203283

theorem cost_of_pet_snake (original_amount : ℕ) (amount_left : ℕ) (cost : ℕ) 
  (h1 : original_amount = 73) (h2 : amount_left = 18) : cost = 55 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_pet_snake_l2032_203283


namespace NUMINAMATH_GPT_xy_range_l2032_203264

theorem xy_range (x y : ℝ) (h1 : y = 3 * (⌊x⌋) + 2) (h2 : y = 4 * (⌊x - 3⌋) + 6) (h3 : (⌊x⌋ : ℝ) ≠ x) :
  34 < x + y ∧ x + y < 35 := 
by 
  sorry

end NUMINAMATH_GPT_xy_range_l2032_203264


namespace NUMINAMATH_GPT_solve_for_k_l2032_203255

variable {x y k : ℝ}

theorem solve_for_k (h1 : 2 * x + y = 1) (h2 : x + 2 * y = k - 2) (h3 : x - y = 2) : k = 1 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_k_l2032_203255


namespace NUMINAMATH_GPT_billy_age_l2032_203214

theorem billy_age (B J : ℕ) (h1 : B = 3 * J) (h2 : B + J = 60) : B = 45 :=
by
  sorry

end NUMINAMATH_GPT_billy_age_l2032_203214


namespace NUMINAMATH_GPT_prob1_prob2_odd_prob2_monotonic_prob3_l2032_203258

variable (a : ℝ) (f : ℝ → ℝ)
variable (hf : ∀ x : ℝ, f (log a x) = a / (a^2 - 1) * (x - 1 / x))
variable (ha : 0 < a ∧ a < 1)

-- Problem 1: Prove the expression for f(x)
theorem prob1 (x : ℝ) : f x = a / (a^2 - 1) * (a^x - a^(-x)) := sorry

-- Problem 2: Prove oddness and monotonicity of f(x)
theorem prob2_odd : ∀ x, f (-x) = -f x := sorry
theorem prob2_monotonic : ∀ x₁ x₂ : ℝ, (x₁ < x₂) → (f x₁ < f x₂) := sorry

-- Problem 3: Determine the range of k
theorem prob3 (k : ℝ) : (∀ t : ℝ, 1 ≤ t ∧ t ≤ 3 → f (3 * t^2 - 1) + f (4 * t - k) > 0) → (k < 6) := sorry

end NUMINAMATH_GPT_prob1_prob2_odd_prob2_monotonic_prob3_l2032_203258


namespace NUMINAMATH_GPT_problem_l2032_203202

def otimes (x y : ℝ) : ℝ := x^3 + y - 2 * x

theorem problem (k : ℝ) : otimes k (otimes k k) = 2 * k^3 - 3 * k :=
by
  sorry

end NUMINAMATH_GPT_problem_l2032_203202


namespace NUMINAMATH_GPT_remaining_standby_time_l2032_203244

variable (fully_charged_standby : ℝ) (fully_charged_gaming : ℝ)
variable (standby_time : ℝ) (gaming_time : ℝ)

theorem remaining_standby_time
  (h1 : fully_charged_standby = 10)
  (h2 : fully_charged_gaming = 2)
  (h3 : standby_time = 4)
  (h4 : gaming_time = 1.5) :
  (10 - ((standby_time * (1 / fully_charged_standby)) + (gaming_time * (1 / fully_charged_gaming)))) * 10 = 1 :=
by
  sorry

end NUMINAMATH_GPT_remaining_standby_time_l2032_203244


namespace NUMINAMATH_GPT_flower_count_l2032_203236

theorem flower_count (roses carnations : ℕ) (h₁ : roses = 5) (h₂ : carnations = 5) : roses + carnations = 10 :=
by
  sorry

end NUMINAMATH_GPT_flower_count_l2032_203236


namespace NUMINAMATH_GPT_sophie_buys_six_doughnuts_l2032_203241

variable (num_doughnuts : ℕ)

theorem sophie_buys_six_doughnuts 
  (h1 : 5 * 2 = 10)
  (h2 : 4 * 2 = 8)
  (h3 : 15 * 0.60 = 9)
  (h4 : 10 + 8 + 9 = 27)
  (h5 : 33 - 27 = 6)
  (h6 : num_doughnuts * 1 = 6) :
  num_doughnuts = 6 := 
  by
    sorry

end NUMINAMATH_GPT_sophie_buys_six_doughnuts_l2032_203241


namespace NUMINAMATH_GPT_problem_solution_l2032_203201

noncomputable def y (x : ℝ) : ℝ := Real.sqrt (x - 1)

theorem problem_solution (x : ℝ) : x ≥ 1 →
  (Real.sqrt (x + 5 - 6 * Real.sqrt (x - 1)) + Real.sqrt (x + 12 - 8 * Real.sqrt (x - 1)) = 2) ↔ (x = 13.25) :=
sorry

end NUMINAMATH_GPT_problem_solution_l2032_203201


namespace NUMINAMATH_GPT_Cassini_l2032_203285

-- Define the Fibonacci sequence
def Fibonacci : ℕ → ℤ
| 0       => 0
| 1       => 1
| (n + 2) => Fibonacci (n + 1) + Fibonacci n

-- State Cassini's Identity theorem
theorem Cassini (n : ℕ) : Fibonacci (n + 1) * Fibonacci (n - 1) - (Fibonacci n) ^ 2 = (-1) ^ n := 
by sorry

end NUMINAMATH_GPT_Cassini_l2032_203285


namespace NUMINAMATH_GPT_proof_problem_l2032_203251

noncomputable def a : ℝ := 0.85 * 250
noncomputable def b : ℝ := 0.75 * 180
noncomputable def c : ℝ := 0.90 * 320

theorem proof_problem :
  (a - b = 77.5) ∧ (77.5 < c) :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l2032_203251


namespace NUMINAMATH_GPT_vertex_of_quadratic1_vertex_of_quadratic2_l2032_203239

theorem vertex_of_quadratic1 :
  ∃ x y : ℝ, 
  (∀ x', 2 * x'^2 - 4 * x' - 1 = 2 * (x' - x)^2 + y) ∧ 
  (x = 1 ∧ y = -3) :=
by sorry

theorem vertex_of_quadratic2 :
  ∃ x y : ℝ, 
  (∀ x', -3 * x'^2 + 6 * x' - 2 = -3 * (x' - x)^2 + y) ∧ 
  (x = 1 ∧ y = 1) :=
by sorry

end NUMINAMATH_GPT_vertex_of_quadratic1_vertex_of_quadratic2_l2032_203239


namespace NUMINAMATH_GPT_arccos_cos_10_l2032_203226

theorem arccos_cos_10 : Real.arccos (Real.cos 10) = 2 := by
  sorry

end NUMINAMATH_GPT_arccos_cos_10_l2032_203226


namespace NUMINAMATH_GPT_lead_atom_ratio_l2032_203216

noncomputable def ratio_of_lead_atoms (average_weight : ℝ) 
  (weight_206 : ℕ) (weight_207 : ℕ) (weight_208 : ℕ) 
  (number_206 : ℕ) (number_207 : ℕ) (number_208 : ℕ) : Prop :=
  average_weight = 207.2 ∧ 
  weight_206 = 206 ∧ 
  weight_207 = 207 ∧ 
  weight_208 = 208 ∧ 
  number_208 = number_206 + number_207 →
  (number_206 : ℚ) / (number_207 : ℚ) = 3 / 2 ∧
  (number_208 : ℚ) / (number_207 : ℚ) = 5 / 2

theorem lead_atom_ratio : ratio_of_lead_atoms 207.2 206 207 208 3 2 5 :=
by sorry

end NUMINAMATH_GPT_lead_atom_ratio_l2032_203216


namespace NUMINAMATH_GPT_reading_days_l2032_203294

theorem reading_days (total_pages pages_per_day_1 pages_per_day_2 : ℕ ) :
  total_pages = 525 →
  pages_per_day_1 = 25 →
  pages_per_day_2 = 21 →
  (total_pages / pages_per_day_1 = 21) ∧ (total_pages / pages_per_day_2 = 25) :=
by
  sorry

end NUMINAMATH_GPT_reading_days_l2032_203294


namespace NUMINAMATH_GPT_find_EQ_l2032_203279

open Real

noncomputable def Trapezoid_EFGH (EF FG GH HE EQ QF : ℝ) : Prop :=
  EF = 110 ∧
  FG = 60 ∧
  GH = 23 ∧
  HE = 75 ∧
  EQ + QF = EF ∧
  EQ = 250 / 3

theorem find_EQ (EF FG GH HE EQ QF : ℝ) (h : Trapezoid_EFGH EF FG GH HE EQ QF) :
  EQ = 250 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_EQ_l2032_203279


namespace NUMINAMATH_GPT_car_dealership_sales_l2032_203228

theorem car_dealership_sales (trucks_ratio suvs_ratio trucks_expected suvs_expected : ℕ)
  (h_ratio : trucks_ratio = 5 ∧ suvs_ratio = 8)
  (h_expected : trucks_expected = 35 ∧ suvs_expected = 56) :
  (trucks_ratio : ℚ) / suvs_ratio = (trucks_expected : ℚ) / suvs_expected :=
by
  sorry

end NUMINAMATH_GPT_car_dealership_sales_l2032_203228


namespace NUMINAMATH_GPT_find_b_l2032_203242

theorem find_b (b : ℕ) (h1 : 40 < b) (h2 : b < 120) 
    (h3 : b % 4 = 3) (h4 : b % 5 = 3) (h5 : b % 6 = 3) : 
    b = 63 := by
  sorry

end NUMINAMATH_GPT_find_b_l2032_203242


namespace NUMINAMATH_GPT_equation_represents_circle_m_condition_l2032_203218

theorem equation_represents_circle_m_condition (m : ℝ) :
  (∃ (x y : ℝ), x^2 + y^2 - x + y + m = 0) → m < 1/2 := 
by
  sorry

end NUMINAMATH_GPT_equation_represents_circle_m_condition_l2032_203218


namespace NUMINAMATH_GPT_total_dinners_sold_203_l2032_203270

def monday_dinners : ℕ := 40
def tuesday_dinners : ℕ := monday_dinners + 40
def wednesday_dinners : ℕ := tuesday_dinners / 2
def thursday_dinners : ℕ := wednesday_dinners + 3

def total_dinners_sold : ℕ := monday_dinners + tuesday_dinners + wednesday_dinners + thursday_dinners

theorem total_dinners_sold_203 : total_dinners_sold = 203 := by
  sorry

end NUMINAMATH_GPT_total_dinners_sold_203_l2032_203270


namespace NUMINAMATH_GPT_find_somus_age_l2032_203209

def somus_current_age (S F : ℕ) := S = F / 3
def somus_age_7_years_ago (S F : ℕ) := (S - 7) = (F - 7) / 5

theorem find_somus_age (S F : ℕ) 
  (h1 : somus_current_age S F) 
  (h2 : somus_age_7_years_ago S F) : S = 14 :=
sorry

end NUMINAMATH_GPT_find_somus_age_l2032_203209


namespace NUMINAMATH_GPT_sara_picked_6_pears_l2032_203229

def total_pears : ℕ := 11
def tim_pears : ℕ := 5
def sara_pears : ℕ := total_pears - tim_pears

theorem sara_picked_6_pears : sara_pears = 6 := by
  sorry

end NUMINAMATH_GPT_sara_picked_6_pears_l2032_203229


namespace NUMINAMATH_GPT_value_of_expression_l2032_203233

theorem value_of_expression (x : ℝ) (h : x^2 + x + 1 = 8) : 4 * x^2 + 4 * x + 9 = 37 :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l2032_203233


namespace NUMINAMATH_GPT_smallest_possible_value_l2032_203213

theorem smallest_possible_value (n : ℕ) (h1 : 100 ≤ n ∧ n < 1000) (h2 : n ≡ 2 [MOD 9]) (h3 : n ≡ 6 [MOD 7]) :
  n = 116 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_smallest_possible_value_l2032_203213


namespace NUMINAMATH_GPT_value_of_y_l2032_203289

theorem value_of_y (y: ℚ) (h: (2 / 5 - 1 / 7) = 14 / y): y = 490 / 9 :=
by
  sorry

end NUMINAMATH_GPT_value_of_y_l2032_203289


namespace NUMINAMATH_GPT_notebooks_problem_l2032_203206

variable (a b c : ℕ)

theorem notebooks_problem (h1 : a + 6 = b + c) (h2 : b + 10 = a + c) : c = 8 :=
  sorry

end NUMINAMATH_GPT_notebooks_problem_l2032_203206


namespace NUMINAMATH_GPT_candles_used_l2032_203275

theorem candles_used (starting_candles used_candles remaining_candles : ℕ) (h1 : starting_candles = 44) (h2 : remaining_candles = 12) : used_candles = 32 :=
by
  sorry

end NUMINAMATH_GPT_candles_used_l2032_203275


namespace NUMINAMATH_GPT_min_value_2x_plus_y_l2032_203243

theorem min_value_2x_plus_y (x y : ℝ) (h1 : |y| ≤ 2 - x) (h2 : x ≥ -1) : 
  ∃ (x y : ℝ), |y| ≤ 2 - x ∧ x ≥ -1 ∧ (∀ y : ℝ, |y| ≤ 2 - x → x ≥ -1 → 2 * x + y ≥ -5) ∧ (2 * x + y = -5) :=
by
  sorry

end NUMINAMATH_GPT_min_value_2x_plus_y_l2032_203243


namespace NUMINAMATH_GPT_polynomial_roots_identity_l2032_203205

theorem polynomial_roots_identity {p q α β γ δ : ℝ} 
  (h1 : α^2 + p*α + 1 = 0)
  (h2 : β^2 + p*β + 1 = 0)
  (h3 : γ^2 + q*γ + 1 = 0)
  (h4 : δ^2 + q*δ + 1 = 0) :
  (α - γ) * (β - γ) * (α + δ) * (β + δ) = q^2 - p^2 :=
sorry

end NUMINAMATH_GPT_polynomial_roots_identity_l2032_203205


namespace NUMINAMATH_GPT_minimum_m_l2032_203246

theorem minimum_m (m n : ℕ) (hm : m > 0) (hn : n > 0) (h : 24 * m = n ^ 4) : m ≥ 54 :=
sorry

end NUMINAMATH_GPT_minimum_m_l2032_203246


namespace NUMINAMATH_GPT_product_of_solutions_eq_zero_l2032_203265

theorem product_of_solutions_eq_zero :
  (∃ x : ℝ, (x + 3) / (3 * x + 3) = (5 * x + 4) / (8 * x + 4)) →
  (∀ x : ℝ, ((x + 3) / (3 * x + 3) = (5 * x + 4) / (8 * x + 4)) → (x = 0 ∨ x = -4/7)) →
  (0 * (-4/7) = 0) :=
by
  sorry

end NUMINAMATH_GPT_product_of_solutions_eq_zero_l2032_203265


namespace NUMINAMATH_GPT_min_value_of_expression_min_value_achieved_l2032_203261

noncomputable def f (x : ℝ) : ℝ :=
  (15 - x) * (8 - x) * (15 + x) * (8 + x)

theorem min_value_of_expression : ∀ x : ℝ, (15 - x) * (8 - x) * (15 + x) * (8 + x) ≥ -6452.25 :=
by sorry

theorem min_value_achieved : ∃ x : ℝ, (15 - x) * (8 - x) * (15 + x) * (8 + x) = -6452.25 :=
by sorry

end NUMINAMATH_GPT_min_value_of_expression_min_value_achieved_l2032_203261


namespace NUMINAMATH_GPT_functional_equation_solution_l2032_203269

theorem functional_equation_solution :
  ∀ (f : ℝ → ℝ), (∀ x y : ℝ, f (x + f y) + f (y + f x) = 2 * f (x * f y)) → (∀ x : ℝ, f x = 0) :=
by
  intro f h
  sorry

end NUMINAMATH_GPT_functional_equation_solution_l2032_203269


namespace NUMINAMATH_GPT_count_boys_correct_l2032_203292

def total_vans : ℕ := 5
def students_per_van : ℕ := 28
def number_of_girls : ℕ := 80

theorem count_boys_correct : 
  (total_vans * students_per_van) - number_of_girls = 60 := 
by
  sorry

end NUMINAMATH_GPT_count_boys_correct_l2032_203292


namespace NUMINAMATH_GPT_sample_size_is_100_l2032_203268

-- Define the number of students selected for the sample.
def num_students_sampled : ℕ := 100

-- The statement that the sample size is equal to the number of students sampled.
theorem sample_size_is_100 : num_students_sampled = 100 := 
by {
  -- Proof goes here
  sorry
}

end NUMINAMATH_GPT_sample_size_is_100_l2032_203268


namespace NUMINAMATH_GPT_students_sampled_from_second_grade_l2032_203277

def arithmetic_sequence (a d : ℕ) : Prop :=
  3 * a - d = 1200

def stratified_sampling (total students second_grade : ℕ) : ℕ :=
  (second_grade * students) / total

theorem students_sampled_from_second_grade 
  (total students : ℕ)
  (h1 : total = 1200)
  (h2 : students = 48)
  (a d : ℕ)
  (h3 : arithmetic_sequence a d)
: stratified_sampling total students a = 16 :=
by
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_students_sampled_from_second_grade_l2032_203277


namespace NUMINAMATH_GPT_num_divisors_not_divisible_by_2_of_360_l2032_203240

def is_divisor (n d : ℕ) : Prop := d ∣ n

def is_prime (p : ℕ) : Prop := Nat.Prime p

noncomputable def prime_factors (n : ℕ) : List ℕ := sorry -- To be implemented if needed

def count_divisors_not_divisible_by_2 (n : ℕ) : ℕ :=
  let factors : List ℕ := prime_factors 360
  let a := 0
  let b_choices := [0, 1, 2]
  let c_choices := [0, 1]
  (b_choices.length) * (c_choices.length)

theorem num_divisors_not_divisible_by_2_of_360 :
  count_divisors_not_divisible_by_2 360 = 6 :=
by sorry

end NUMINAMATH_GPT_num_divisors_not_divisible_by_2_of_360_l2032_203240


namespace NUMINAMATH_GPT_plates_remove_proof_l2032_203219

noncomputable def total_weight_initial (plates: ℤ) (weight_per_plate: ℤ): ℤ :=
  plates * weight_per_plate

noncomputable def weight_limit (pounds: ℤ) (ounces_per_pound: ℤ): ℤ :=
  pounds * ounces_per_pound

noncomputable def plates_to_remove (initial_weight: ℤ) (limit: ℤ) (weight_per_plate: ℤ): ℤ :=
  (initial_weight - limit) / weight_per_plate

theorem plates_remove_proof :
  let pounds := 20
  let ounces_per_pound := 16
  let plates_initial := 38
  let weight_per_plate := 10
  let initial_weight := total_weight_initial plates_initial weight_per_plate
  let limit := weight_limit pounds ounces_per_pound
  plates_to_remove initial_weight limit weight_per_plate = 6 :=
by
  sorry

end NUMINAMATH_GPT_plates_remove_proof_l2032_203219


namespace NUMINAMATH_GPT_find_a_and_b_l2032_203208

noncomputable def f (x: ℝ) (b: ℝ): ℝ := x^2 + 5*x + b
noncomputable def g (x: ℝ) (b: ℝ): ℝ := 2*b*x + 3

theorem find_a_and_b (a b: ℝ):
  (∀ x: ℝ, f (g x b) b = a * x^2 + 30 * x + 24) →
  a = 900 / 121 ∧ b = 15 / 11 :=
by
  intro H
  -- Proof is omitted as requested
  sorry

end NUMINAMATH_GPT_find_a_and_b_l2032_203208


namespace NUMINAMATH_GPT_carpenter_job_duration_l2032_203295

theorem carpenter_job_duration
  (total_estimate : ℤ)
  (carpenter_hourly_rate : ℤ)
  (assistant_hourly_rate : ℤ)
  (material_cost : ℤ)
  (H1 : total_estimate = 1500)
  (H2 : carpenter_hourly_rate = 35)
  (H3 : assistant_hourly_rate = 25)
  (H4 : material_cost = 720) :
  (total_estimate - material_cost) / (carpenter_hourly_rate + assistant_hourly_rate) = 13 :=
by
  sorry

end NUMINAMATH_GPT_carpenter_job_duration_l2032_203295


namespace NUMINAMATH_GPT_extra_chairs_added_l2032_203298

theorem extra_chairs_added (rows cols total_chairs extra_chairs : ℕ) 
  (h1 : rows = 7) 
  (h2 : cols = 12) 
  (h3 : total_chairs = 95) 
  (h4 : extra_chairs = total_chairs - rows * cols) : 
  extra_chairs = 11 := by 
  sorry

end NUMINAMATH_GPT_extra_chairs_added_l2032_203298


namespace NUMINAMATH_GPT_range_of_m_l2032_203249

noncomputable def quadratic_function : Type := ℝ → ℝ

variable (f : quadratic_function)

axiom quadratic : ∃ a b : ℝ, ∀ x : ℝ, f x = a * (x-2)^2 + b
axiom symmetry : ∀ x : ℝ, f (x + 2) = f (-x + 2)
axiom condition1 : f 0 = 3
axiom condition2 : f 2 = 1
axiom max_value : ∀ m : ℝ, ∀ (x : ℝ) (hx : 0 ≤ x ∧ x ≤ m), f x ≤ 3
axiom min_value : ∀ m : ℝ, ∀ (x : ℝ) (hx : 0 ≤ x ∧ x ≤ m), 1 ≤ f x

theorem range_of_m : ∀ m : ℝ, (∀ (x : ℝ) (hx : 0 ≤ x ∧ x ≤ m), 1 ≤ f x ∧ f x ≤ 3) → 2 ≤ m ∧ m ≤ 4 :=
by
  intro m
  intro h
  sorry

end NUMINAMATH_GPT_range_of_m_l2032_203249


namespace NUMINAMATH_GPT_max_value_of_expression_l2032_203230

theorem max_value_of_expression (x₁ y₁ x₂ y₂ : ℝ) 
  (h₁ : x₁^2 / 4 + 9 * y₁^2 / 4 = 1) 
  (h₂ : x₂^2 / 4 + 9 * y₂^2 / 4 = 1) 
  (h₃ : x₁ * x₂ + 9 * y₁ * y₂ = -2) :
  (|2 * x₁ + 3 * y₁ - 3| + |2 * x₂ + 3 * y₂ - 3|) ≤ 6 + 2 * Real.sqrt 5 :=
sorry

end NUMINAMATH_GPT_max_value_of_expression_l2032_203230


namespace NUMINAMATH_GPT_prove_Φ_eq_8_l2032_203272

-- Define the structure of the problem.
def condition (Φ : ℕ) : Prop := 504 / Φ = 40 + 3 * Φ

-- Define the main proof question.
theorem prove_Φ_eq_8 (Φ : ℕ) (h : condition Φ) : Φ = 8 := 
sorry

end NUMINAMATH_GPT_prove_Φ_eq_8_l2032_203272


namespace NUMINAMATH_GPT_speed_first_hour_l2032_203284

theorem speed_first_hour (x : ℝ) :
  (∃ x, (x + 45) / 2 = 65) → x = 85 :=
by
  intro h
  obtain ⟨x, hx⟩ := h
  sorry

end NUMINAMATH_GPT_speed_first_hour_l2032_203284


namespace NUMINAMATH_GPT_largest_lcm_l2032_203260

def lcm_list : List ℕ := [
  Nat.lcm 15 3,
  Nat.lcm 15 5,
  Nat.lcm 15 9,
  Nat.lcm 15 10,
  Nat.lcm 15 12,
  Nat.lcm 15 15
]

theorem largest_lcm : List.maximum lcm_list = 60 := by
  sorry

end NUMINAMATH_GPT_largest_lcm_l2032_203260


namespace NUMINAMATH_GPT_theta_in_third_quadrant_l2032_203271

-- Define the mathematical conditions
variable (θ : ℝ)
axiom cos_theta_neg : Real.cos θ < 0
axiom cos_minus_sin_eq_sqrt : Real.cos θ - Real.sin θ = Real.sqrt (1 - 2 * Real.sin θ * Real.cos θ)

-- Prove that θ is in the third quadrant
theorem theta_in_third_quadrant : 
  (∀ θ : ℝ, Real.cos θ < 0 → Real.cos θ - Real.sin θ = Real.sqrt (1 - 2 * Real.sin θ * Real.cos θ) → 
    Real.sin θ < 0 ∧ Real.cos θ < 0) :=
by sorry

end NUMINAMATH_GPT_theta_in_third_quadrant_l2032_203271


namespace NUMINAMATH_GPT_percentage_students_enrolled_in_bio_l2032_203280

-- Problem statement
theorem percentage_students_enrolled_in_bio (total_students : ℕ) (students_not_in_bio : ℕ) 
    (h1 : total_students = 880) (h2 : students_not_in_bio = 462) : 
    ((total_students - students_not_in_bio : ℚ) / total_students) * 100 = 47.5 := by 
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_percentage_students_enrolled_in_bio_l2032_203280


namespace NUMINAMATH_GPT_total_earning_proof_l2032_203222

noncomputable def total_earning (daily_wage_c : ℝ) (days_a : ℕ) (days_b : ℕ) (days_c : ℕ) (ratio_a : ℕ) (ratio_b : ℕ) (ratio_c : ℕ) : ℝ :=
  let daily_wage_a := (ratio_a : ℝ) / (ratio_c : ℝ) * daily_wage_c
  let daily_wage_b := (ratio_b : ℝ) / (ratio_c : ℝ) * daily_wage_c
  (daily_wage_a * days_a) + (daily_wage_b * days_b) + (daily_wage_c * days_c)

theorem total_earning_proof : 
  total_earning 71.15384615384615 16 9 4 3 4 5 = 1480 := 
by 
  -- calculations here
  sorry

end NUMINAMATH_GPT_total_earning_proof_l2032_203222


namespace NUMINAMATH_GPT_number_division_l2032_203237

theorem number_division (x : ℤ) (h : x - 17 = 55) : x / 9 = 8 :=
by 
  sorry

end NUMINAMATH_GPT_number_division_l2032_203237


namespace NUMINAMATH_GPT_katrina_cookies_left_l2032_203278

/-- Katrina’s initial number of cookies. -/
def initial_cookies : ℕ := 120

/-- Cookies sold in the morning. 
    1 dozen is 12 cookies, so 3 dozen is 36 cookies. -/
def morning_sales : ℕ := 3 * 12

/-- Cookies sold during the lunch rush. -/
def lunch_sales : ℕ := 57

/-- Cookies sold in the afternoon. -/
def afternoon_sales : ℕ := 16

/-- Calculate the number of cookies left after all sales. -/
def cookies_left : ℕ :=
  initial_cookies - morning_sales - lunch_sales - afternoon_sales

/-- Prove that the number of cookies left for Katrina to take home is 11. -/
theorem katrina_cookies_left : cookies_left = 11 := by
  sorry

end NUMINAMATH_GPT_katrina_cookies_left_l2032_203278


namespace NUMINAMATH_GPT_find_varphi_l2032_203287

theorem find_varphi 
  (f g : ℝ → ℝ) 
  (x1 x2 varphi : ℝ) 
  (h_f : ∀ x, f x = 2 * Real.cos (2 * x)) 
  (h_g : ∀ x, g x = 2 * Real.cos (2 * x - 2 * varphi)) 
  (h_varphi_range : 0 < varphi ∧ varphi < π / 2) 
  (h_diff_cos : |f x1 - g x2| = 4) 
  (h_min_dist : |x1 - x2| = π / 6) 
: varphi = π / 3 := 
sorry

end NUMINAMATH_GPT_find_varphi_l2032_203287


namespace NUMINAMATH_GPT_calculate_x_one_minus_f_l2032_203245

noncomputable def x := (2 + Real.sqrt 3) ^ 500
noncomputable def n := Int.floor x
noncomputable def f := x - n

theorem calculate_x_one_minus_f : x * (1 - f) = 1 := by
  sorry

end NUMINAMATH_GPT_calculate_x_one_minus_f_l2032_203245


namespace NUMINAMATH_GPT_simplify_polynomial_l2032_203238

theorem simplify_polynomial (x : ℝ) :
  (5 - 5 * x - 10 * x^2 + 10 + 15 * x - 20 * x^2 - 10 + 20 * x + 30 * x^2) = 5 + 30 * x :=
  by sorry

end NUMINAMATH_GPT_simplify_polynomial_l2032_203238


namespace NUMINAMATH_GPT_complement_intersection_eq_l2032_203267

-- Definitions of sets
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set ℕ := {2, 5, 8}
def B : Set ℕ := {1, 3, 5, 7}

-- Definition of complement of A in U
def complement_U_A : Set ℕ := U \ A

-- The main statement to prove
theorem complement_intersection_eq :
  (complement_U_A ∩ B) = {1, 3, 7} :=
by sorry

end NUMINAMATH_GPT_complement_intersection_eq_l2032_203267


namespace NUMINAMATH_GPT_trig_identity_l2032_203263

open Real

theorem trig_identity (α : ℝ) (h : tan α = -1/2) : 1 - sin (2 * α) = 9/5 := 
  sorry

end NUMINAMATH_GPT_trig_identity_l2032_203263


namespace NUMINAMATH_GPT_min_point_transformed_graph_l2032_203223

noncomputable def original_eq (x : ℝ) : ℝ := 2 * |x| - 4

noncomputable def translated_eq (x : ℝ) : ℝ := 2 * |x - 3| - 8

theorem min_point_transformed_graph : translated_eq 3 = -8 :=
by
  -- Solution steps would go here
  sorry

end NUMINAMATH_GPT_min_point_transformed_graph_l2032_203223


namespace NUMINAMATH_GPT_first_course_cost_l2032_203247

theorem first_course_cost (x : ℝ) (h1 : 60 - (x + (x + 5) + 0.25 * (x + 5)) = 20) : x = 15 :=
by sorry

end NUMINAMATH_GPT_first_course_cost_l2032_203247


namespace NUMINAMATH_GPT_find_ABC_l2032_203254

theorem find_ABC (A B C : ℝ) (h : ∀ n : ℕ, n > 0 → 2 * n^3 + 3 * n^2 = A * (n * (n - 1) * (n - 2)) / 6 + B * (n * (n - 1)) / 2 + C * n) :
  A = 12 ∧ B = 18 ∧ C = 5 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_ABC_l2032_203254


namespace NUMINAMATH_GPT_negation_of_p_l2032_203273

-- Define the original predicate
def p (x₀ : ℝ) : Prop := x₀^2 > 1

-- Define the negation of the predicate
def not_p : Prop := ∀ x : ℝ, x^2 ≤ 1

-- Prove the negation of the proposition
theorem negation_of_p : (∃ x₀ : ℝ, p x₀) ↔ not_p := by
  sorry

end NUMINAMATH_GPT_negation_of_p_l2032_203273


namespace NUMINAMATH_GPT_base_329_digits_even_l2032_203235

noncomputable def base_of_four_digit_even_final : ℕ := 5

theorem base_329_digits_even (b : ℕ) (h1 : b^3 ≤ 329) (h2 : 329 < b^4)
  (h3 : ∀ d, 329 % b = d → d % 2 = 0) : b = base_of_four_digit_even_final :=
by sorry

end NUMINAMATH_GPT_base_329_digits_even_l2032_203235


namespace NUMINAMATH_GPT_best_fit_model_l2032_203262

-- Define the coefficients of determination for each model
noncomputable def R2_Model1 : ℝ := 0.75
noncomputable def R2_Model2 : ℝ := 0.90
noncomputable def R2_Model3 : ℝ := 0.45
noncomputable def R2_Model4 : ℝ := 0.65

-- State the theorem 
theorem best_fit_model : 
  R2_Model2 ≥ R2_Model1 ∧ 
  R2_Model2 ≥ R2_Model3 ∧ 
  R2_Model2 ≥ R2_Model4 :=
by
  sorry

end NUMINAMATH_GPT_best_fit_model_l2032_203262


namespace NUMINAMATH_GPT_evaluate_expression_l2032_203224

variable (g : ℕ → ℕ)
variable (g_inv : ℕ → ℕ)
variable (h1 : g 4 = 7)
variable (h2 : g 6 = 2)
variable (h3 : g 3 = 6)
variable (h4 : ∀ x, g (g_inv x) = x)
variable (h5 : ∀ x, g_inv (g x) = x)

theorem evaluate_expression : g_inv (g_inv 6 + g_inv 7) = 4 :=
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2032_203224


namespace NUMINAMATH_GPT_largest_value_of_a_l2032_203212

theorem largest_value_of_a : 
  ∃ (a : ℚ), (3 * a + 4) * (a - 2) = 9 * a ∧ ∀ b : ℚ, (3 * b + 4) * (b - 2) = 9 * b → b ≤ 4 :=
by
  sorry

end NUMINAMATH_GPT_largest_value_of_a_l2032_203212


namespace NUMINAMATH_GPT_product_of_two_numbers_l2032_203299

theorem product_of_two_numbers (a b : ℕ) (ha : a ≠ 0) (hb : b ≠ 0)
  (h_sum : a + b = 210) (h_lcm : Nat.lcm a b = 1547) : a * b = 10829 :=
by
  sorry

end NUMINAMATH_GPT_product_of_two_numbers_l2032_203299


namespace NUMINAMATH_GPT_angle_at_630_is_15_degrees_l2032_203248

-- Definitions for positions of hour and minute hands at 6:30 p.m.
def angle_per_hour : ℝ := 30
def minute_hand_position_630 : ℝ := 180
def hour_hand_position_630 : ℝ := 195

-- The angle between the hour hand and minute hand at 6:30 p.m.
def angle_between_hands_630 : ℝ := |hour_hand_position_630 - minute_hand_position_630|

-- Statement to prove
theorem angle_at_630_is_15_degrees :
  angle_between_hands_630 = 15 := by
  sorry

end NUMINAMATH_GPT_angle_at_630_is_15_degrees_l2032_203248


namespace NUMINAMATH_GPT_dice_sum_not_possible_l2032_203282

theorem dice_sum_not_possible (a b c d : ℕ) (h₁ : 1 ≤ a ∧ a ≤ 6) (h₂ : 1 ≤ b ∧ b ≤ 6) 
(h₃ : 1 ≤ c ∧ c ≤ 6) (h₄ : 1 ≤ d ∧ d ≤ 6) (h_product : a * b * c * d = 216) : 
(a + b + c + d ≠ 15) ∧ (a + b + c + d ≠ 16) ∧ (a + b + c + d ≠ 18) :=
sorry

end NUMINAMATH_GPT_dice_sum_not_possible_l2032_203282


namespace NUMINAMATH_GPT_arithmetic_sequence_a15_value_l2032_203234

variables {a : ℕ → ℤ}

noncomputable def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a15_value
  (h1 : is_arithmetic_sequence a)
  (h2 : a 3 + a 13 = 20)
  (h3 : a 2 = -2) : a 15 = 24 :=
by sorry

end NUMINAMATH_GPT_arithmetic_sequence_a15_value_l2032_203234


namespace NUMINAMATH_GPT_sum_of_roots_l2032_203220

theorem sum_of_roots {a b : Real} (h1 : a * (a - 4) = 5) (h2 : b * (b - 4) = 5) (h3 : a ≠ b) : a + b = 4 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_roots_l2032_203220


namespace NUMINAMATH_GPT_triangle_side_length_l2032_203253

theorem triangle_side_length 
  (side1 : ℕ) (side2 : ℕ) (side3 : ℕ) (P : ℕ)
  (h_side1 : side1 = 5)
  (h_side3 : side3 = 30)
  (h_P : P = 55) :
  side1 + side2 + side3 = P → side2 = 20 :=
by
  intros h
  sorry 

end NUMINAMATH_GPT_triangle_side_length_l2032_203253

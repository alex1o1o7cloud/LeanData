import Mathlib

namespace NUMINAMATH_GPT_correct_expansion_l505_50550

variables {x y : ℝ}

theorem correct_expansion : 
  (-x + y)^2 = x^2 - 2 * x * y + y^2 := sorry

end NUMINAMATH_GPT_correct_expansion_l505_50550


namespace NUMINAMATH_GPT_least_prime_factor_of_11_pow_5_minus_11_pow_4_l505_50569

theorem least_prime_factor_of_11_pow_5_minus_11_pow_4 : 
  Nat.minFac (11^5 - 11^4) = 2 := 
by sorry

end NUMINAMATH_GPT_least_prime_factor_of_11_pow_5_minus_11_pow_4_l505_50569


namespace NUMINAMATH_GPT_domain_of_function_l505_50529

theorem domain_of_function :
  { x : ℝ | 0 ≤ 2 * x - 10 ∧ 2 * x - 10 ≠ 0 } = { x : ℝ | x > 5 } :=
by
  sorry

end NUMINAMATH_GPT_domain_of_function_l505_50529


namespace NUMINAMATH_GPT_sum_three_positive_numbers_ge_three_l505_50534

theorem sum_three_positive_numbers_ge_three 
  (a b c : ℝ) 
  (h_pos_a : 0 < a) 
  (h_pos_b : 0 < b) 
  (h_pos_c : 0 < c)
  (h_eq : (a + 1) * (b + 1) * (c + 1) = 8) : 
  a + b + c ≥ 3 :=
sorry

end NUMINAMATH_GPT_sum_three_positive_numbers_ge_three_l505_50534


namespace NUMINAMATH_GPT_function_is_odd_and_monotonically_increasing_on_pos_l505_50501

-- Define odd function
def odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f (x)

-- Define monotonically increasing on (0, +∞)
def monotonically_increasing_on_pos (f : ℝ → ℝ) := ∀ x y : ℝ, (0 < x ∧ x < y) → f (x) < f (y)

-- Define the function in question
def f (x : ℝ) := x * |x|

-- Prove the function is odd and monotonically increasing on (0, +∞)
theorem function_is_odd_and_monotonically_increasing_on_pos :
  odd_function f ∧ monotonically_increasing_on_pos f :=
by
  sorry

end NUMINAMATH_GPT_function_is_odd_and_monotonically_increasing_on_pos_l505_50501


namespace NUMINAMATH_GPT_initial_carrots_count_l505_50509

theorem initial_carrots_count (x : ℕ) (h1 : x - 2 + 21 = 31) : x = 12 := by
  sorry

end NUMINAMATH_GPT_initial_carrots_count_l505_50509


namespace NUMINAMATH_GPT_first_player_wins_l505_50528

-- Define the initial conditions
def initial_pieces : ℕ := 1
def final_pieces (m n : ℕ) : ℕ := m * n
def num_moves (pieces : ℕ) : ℕ := pieces - 1

-- Theorem statement: Given the initial dimensions and the game rules,
-- prove that the first player will win.
theorem first_player_wins (m n : ℕ) (h_m : m = 6) (h_n : n = 8) : 
  (num_moves (final_pieces m n)) % 2 = 0 → false :=
by
  -- The solution details and the proof will be here.
  sorry

end NUMINAMATH_GPT_first_player_wins_l505_50528


namespace NUMINAMATH_GPT_altitude_eqn_equidistant_eqn_l505_50577

-- Define the points A, B, and C
def A : (ℝ × ℝ) := (1, 1)
def B : (ℝ × ℝ) := (-1, 3)
def C : (ℝ × ℝ) := (3, 4)

-- Definition of a line in the form Ax + By + C = 0
structure Line :=
  (A B C : ℝ)
  (non_zero : A ≠ 0 ∨ B ≠ 0)

-- Equation of line l1 (altitude to side BC)
def l1 : Line := { A := 4, B := 1, C := -5, non_zero := Or.inl (by norm_num) }

-- Equation of line l2 (passing through C, equidistant from A and B), two possible values
def l2a : Line := { A := 1, B := 1, C := -7, non_zero := Or.inl (by norm_num) }
def l2b : Line := { A := 2, B := -3, C := 6, non_zero := Or.inl (by norm_num) }

-- Prove the equations for l1 and l2 are correct given the points A, B, and C
theorem altitude_eqn (h : A = (1, 1) ∧ B = (-1, 3) ∧ C = (3, 4)) :
  l1 = { A := 4, B := 1, C := -5, non_zero := Or.inl (by norm_num) } := sorry

theorem equidistant_eqn (h : A = (1, 1) ∧ B = (-1, 3) ∧ C = (3, 4)) :
  l2a = { A := 1, B := 1, C := -7, non_zero := Or.inl (by norm_num) } ∨
  l2b = { A := 2, B := -3, C := 6, non_zero := Or.inl (by norm_num) } := sorry

end NUMINAMATH_GPT_altitude_eqn_equidistant_eqn_l505_50577


namespace NUMINAMATH_GPT_find_a_plus_b_l505_50547

noncomputable def f (a b x : ℝ) := a ^ x + b

theorem find_a_plus_b (a b : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) 
  (dom1 : f a b (-2) = -2) (dom2 : f a b 0 = 0) :
  a + b = (Real.sqrt 3) / 3 - 3 :=
by
  unfold f at dom1 dom2
  sorry

end NUMINAMATH_GPT_find_a_plus_b_l505_50547


namespace NUMINAMATH_GPT_paving_cost_l505_50565

theorem paving_cost (l w r : ℝ) (h_l : l = 5.5) (h_w : w = 4) (h_r : r = 700) :
  l * w * r = 15400 :=
by sorry

end NUMINAMATH_GPT_paving_cost_l505_50565


namespace NUMINAMATH_GPT_sum_of_two_numbers_l505_50586

theorem sum_of_two_numbers :
  ∃ x y : ℝ, (x * y = 9375 ∧ y / x = 15) ∧ (x + y = 400) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_two_numbers_l505_50586


namespace NUMINAMATH_GPT_fundraiser_contribution_l505_50523

theorem fundraiser_contribution :
  let sasha_muffins := 30
  let melissa_muffins := 4 * sasha_muffins
  let tiffany_muffins := (sasha_muffins + melissa_muffins) / 2
  let total_muffins := sasha_muffins + melissa_muffins + tiffany_muffins
  let price_per_muffin := 4
  total_muffins * price_per_muffin = 900 :=
by
  let sasha_muffins := 30
  let melissa_muffins := 4 * sasha_muffins
  let tiffany_muffins := (sasha_muffins + melissa_muffins) / 2
  let total_muffins := sasha_muffins + melissa_muffins + tiffany_muffins
  let price_per_muffin := 4
  sorry

end NUMINAMATH_GPT_fundraiser_contribution_l505_50523


namespace NUMINAMATH_GPT_sum_of_coefficients_l505_50592

noncomputable def polynomial_eq (x : ℝ) : ℝ := 1 + x^5
noncomputable def linear_combination (a0 a1 a2 a3 a4 a5 x : ℝ) : ℝ :=
  a0 + a1 * (x - 1) + a2 * (x - 1) ^ 2 + a3 * (x - 1) ^ 3 + a4 * (x - 1) ^ 4 + a5 * (x - 1) ^ 5

theorem sum_of_coefficients (a0 a1 a2 a3 a4 a5 : ℝ) :
  polynomial_eq 1 = linear_combination a0 a1 a2 a3 a4 a5 1 →
  polynomial_eq 2 = linear_combination a0 a1 a2 a3 a4 a5 2 →
  a0 = 2 →
  a1 + a2 + a3 + a4 + a5 = 31 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_sum_of_coefficients_l505_50592


namespace NUMINAMATH_GPT_lcm_45_75_l505_50585

theorem lcm_45_75 : Nat.lcm 45 75 = 225 :=
by
  sorry

end NUMINAMATH_GPT_lcm_45_75_l505_50585


namespace NUMINAMATH_GPT_find_a1_l505_50558

theorem find_a1 (a : ℕ → ℝ) (h_rec : ∀ n, a (n + 1) = a n / (2 * a n + 1)) (h_init : a 3 = 1 / 5) : a 1 = 1 := by
  sorry

end NUMINAMATH_GPT_find_a1_l505_50558


namespace NUMINAMATH_GPT_prove_ab_eq_neg_26_l505_50597

theorem prove_ab_eq_neg_26
  (a b : ℚ)
  (H : ∀ k : ℚ, ∃ x : ℚ, (2 * k * x + a) / 3 = 2 + (x - b * k) / 6) :
  a * b = -26 := sorry

end NUMINAMATH_GPT_prove_ab_eq_neg_26_l505_50597


namespace NUMINAMATH_GPT_preferred_point_condition_l505_50526

theorem preferred_point_condition (x y : ℝ) (h₁ : x^2 + y^2 ≤ 2008)
  (cond : ∀ x' y', (x'^2 + y'^2 ≤ 2008) → (x' ≤ x → y' ≥ y) → (x = x' ∧ y = y')) :
  x^2 + y^2 = 2008 ∧ x ≤ 0 ∧ y ≥ 0 :=
by
  sorry

end NUMINAMATH_GPT_preferred_point_condition_l505_50526


namespace NUMINAMATH_GPT_suff_not_nec_l505_50561

theorem suff_not_nec (x : ℝ) : (x > 0 → x ≠ 0) ∧ (x ≠ 0 → ¬(x ≤ 0)) :=
by
  sorry

end NUMINAMATH_GPT_suff_not_nec_l505_50561


namespace NUMINAMATH_GPT_compound_interest_l505_50573

theorem compound_interest (P R T : ℝ) (SI CI : ℝ)
  (hSI : SI = P * R * T / 100)
  (h_given_SI : SI = 50)
  (h_given_R : R = 5)
  (h_given_T : T = 2)
  (h_compound_interest : CI = P * ((1 + R / 100)^T - 1)) :
  CI = 51.25 :=
by
  -- Since we are only required to state the theorem, we add 'sorry' here.
  sorry

end NUMINAMATH_GPT_compound_interest_l505_50573


namespace NUMINAMATH_GPT_solution_problem_l505_50580

noncomputable def proof_problem (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 1) : Prop :=
  (-1 < (x - y)) ∧ ((x - y) < 1) ∧ (∀ (x y : ℝ), (0 < x) ∧ (0 < y) ∧ (x + y = 1) → (min ((1/x) + (x/y)) = 3))

theorem solution_problem (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 1) :
  proof_problem x y hx hy h := 
sorry

end NUMINAMATH_GPT_solution_problem_l505_50580


namespace NUMINAMATH_GPT_find_g_five_l505_50541

def g (a b c x : ℝ) : ℝ := a * x^7 + b * x^6 + c * x - 3

theorem find_g_five (a b c : ℝ) (h : g a b c (-5) = -3) : g a b c 5 = 31250 * b - 3 := 
sorry

end NUMINAMATH_GPT_find_g_five_l505_50541


namespace NUMINAMATH_GPT_symmetric_point_l505_50504

theorem symmetric_point (x y : ℝ) (a b : ℝ) :
  (x = 3 ∧ y = 9 ∧ a = -1 ∧ b = -3) ∧ (∀ k: ℝ, k ≠ 0 → (y - 9 = k * (x - 3)) ∧ 
  ((x - 3)^2 + (y - 9)^2 = (a - 3)^2 + (b - 9)^2) ∧ 
  (x >= 0 → (a >= 0 ↔ x = 3) ∧ (b >= 0 ↔ y = 9))) :=
by
  sorry

end NUMINAMATH_GPT_symmetric_point_l505_50504


namespace NUMINAMATH_GPT_runner_speed_comparison_l505_50525

theorem runner_speed_comparison
  (t1 t2 : ℕ → ℝ) -- function to map lap-time.
  (s v1 v2 : ℝ)  -- speed of runners v1 and v2 respectively, and the street distance s.
  (h1 : t1 1 < t2 1) -- first runner overtakes the second runner twice implying their lap-time comparison.
  (h2 : ∀ n, t1 (n + 1) = t1 n + t1 1) -- lap time consistency for runner 1
  (h3 : ∀ n, t2 (n + 1) = t2 n + t2 1) -- lap time consistency for runner 2
  (h4 : t1 3 < t2 2) -- first runner completes 3 laps faster than second runner completes 2 laps
   : 2 * v2 ≤ v1 := sorry

end NUMINAMATH_GPT_runner_speed_comparison_l505_50525


namespace NUMINAMATH_GPT_inequality_ineq_l505_50510

theorem inequality_ineq (x y : ℝ) (hx: x > Real.sqrt 2) (hy: y > Real.sqrt 2) : 
  x^4 - x^3 * y + x^2 * y^2 - x * y^3 + y^4 > x^2 + y^2 := 
  sorry

end NUMINAMATH_GPT_inequality_ineq_l505_50510


namespace NUMINAMATH_GPT_transport_tax_to_be_paid_l505_50553

noncomputable def engine_power : ℕ := 150
noncomputable def tax_rate : ℕ := 20
noncomputable def annual_tax : ℕ := engine_power * tax_rate
noncomputable def months_used : ℕ := 8
noncomputable def prorated_tax : ℕ := (months_used * annual_tax) / 12

theorem transport_tax_to_be_paid : prorated_tax = 2000 := 
by 
  -- sorry is used to skip the proof step
  sorry

end NUMINAMATH_GPT_transport_tax_to_be_paid_l505_50553


namespace NUMINAMATH_GPT_marble_ratio_correct_l505_50517

-- Necessary given conditions
variables (x : ℕ) (Ben_initial John_initial : ℕ) (John_post Ben_post : ℕ)
variables (h1 : Ben_initial = 18)
variables (h2 : John_initial = 17)
variables (h3 : Ben_post = Ben_initial - x)
variables (h4 : John_post = John_initial + x)
variables (h5 : John_post = Ben_post + 17)

-- Define the ratio of the number of marbles Ben gave to John to the number of marbles Ben had initially
def marble_ratio := (x : ℕ) / Ben_initial

-- The theorem we want to prove
theorem marble_ratio_correct (h1 : Ben_initial = 18) (h2 : John_initial = 17) (h3 : Ben_post = Ben_initial - x)
(h4 : John_post = John_initial + x) (h5 : John_post = Ben_post + 17) : marble_ratio x Ben_initial = 1/2 := by 
  sorry

end NUMINAMATH_GPT_marble_ratio_correct_l505_50517


namespace NUMINAMATH_GPT_range_of_a_for_monotonic_increasing_f_l505_50536

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - a * x - 2 * Real.log x

theorem range_of_a_for_monotonic_increasing_f (a : ℝ) (h : ∀ x : ℝ, 1 ≤ x → (x - a - 2 / x) ≥ 0) : a ≤ -1 :=
by {
  -- Placeholder for the detailed proof steps
  sorry
}

end NUMINAMATH_GPT_range_of_a_for_monotonic_increasing_f_l505_50536


namespace NUMINAMATH_GPT_proportion_Q_to_R_l505_50563

theorem proportion_Q_to_R (q r : ℕ) (h1 : 3 * q + 5 * r = 1000) (h2 : 4 * r - 2 * q = 250) : q = r :=
by sorry

end NUMINAMATH_GPT_proportion_Q_to_R_l505_50563


namespace NUMINAMATH_GPT_three_digit_numbers_div_by_17_l505_50572

theorem three_digit_numbers_div_by_17 : ∃ n : ℕ, n = 53 ∧ 
  let min_k := Nat.ceil (100 / 17)
  let max_k := Nat.floor (999 / 17)
  min_k = 6 ∧ max_k = 58 ∧ (max_k - min_k + 1) = n :=
by
  sorry

end NUMINAMATH_GPT_three_digit_numbers_div_by_17_l505_50572


namespace NUMINAMATH_GPT_gcd_54_180_l505_50564

theorem gcd_54_180 : Nat.gcd 54 180 = 18 := by
  sorry

end NUMINAMATH_GPT_gcd_54_180_l505_50564


namespace NUMINAMATH_GPT_toys_produced_per_day_l505_50542

theorem toys_produced_per_day :
  (3400 / 5 = 680) :=
by
  sorry

end NUMINAMATH_GPT_toys_produced_per_day_l505_50542


namespace NUMINAMATH_GPT_eggs_collected_l505_50540

def total_eggs_collected (b1 e1 b2 e2 : ℕ) : ℕ :=
  b1 * e1 + b2 * e2

theorem eggs_collected :
  total_eggs_collected 450 36 405 42 = 33210 :=
by
  sorry

end NUMINAMATH_GPT_eggs_collected_l505_50540


namespace NUMINAMATH_GPT_seven_lines_regions_l505_50559

theorem seven_lines_regions (n : ℕ) (hn : n = 7) (h1 : ¬ ∃ l1 l2 : ℝ, l1 = l2) (h2 : ∀ l1 l2 l3 : ℝ, ¬ (l1 ≠ l2 ∧ l2 ≠ l3 ∧ l1 ≠ l3 ∧ (l1 = l2 ∧ l2 = l3))) :
  ∃ R : ℕ, R = 29 :=
by
  sorry

end NUMINAMATH_GPT_seven_lines_regions_l505_50559


namespace NUMINAMATH_GPT_person_before_you_taller_than_you_l505_50549

-- Define the persons involved in the problem.
variable (Person : Type)
variable (Taller : Person → Person → Prop)
variable (P Q You : Person)

-- The conditions given in the problem.
axiom standing_queue : Taller P Q
axiom queue_structure : You = Q

-- The question we need to prove, which is the correct answer to the problem.
theorem person_before_you_taller_than_you : Taller P You :=
by
  sorry

end NUMINAMATH_GPT_person_before_you_taller_than_you_l505_50549


namespace NUMINAMATH_GPT_percent_of_1600_l505_50589

theorem percent_of_1600 (x : ℝ) (h1 : 0.25 * 1600 = 400) (h2 : x / 100 * 400 = 20) : x = 5 :=
sorry

end NUMINAMATH_GPT_percent_of_1600_l505_50589


namespace NUMINAMATH_GPT_inequality_solution_l505_50533

theorem inequality_solution :
  {x : ℝ | ((x - 1) * (x - 4) * (x - 5)) / ((x - 2) * (x - 6) * (x - 7)) > 0} = 
  {x : ℝ | (1 < x ∧ x < 2) ∨ (4 < x ∧ x < 5) ∨ (6 < x ∧ x < 7)} :=
sorry

end NUMINAMATH_GPT_inequality_solution_l505_50533


namespace NUMINAMATH_GPT_tan_alpha_equiv_l505_50539

theorem tan_alpha_equiv (α : ℝ) (h : Real.tan α = 2) : 
    (2 * Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = 1 := 
by 
  sorry

end NUMINAMATH_GPT_tan_alpha_equiv_l505_50539


namespace NUMINAMATH_GPT_ratio_of_blue_to_red_area_l505_50584

theorem ratio_of_blue_to_red_area :
  let r₁ := 1 / 2
  let r₂ := 3 / 2
  let A_red := Real.pi * r₁^2
  let A_large := Real.pi * r₂^2
  let A_blue := A_large - A_red
  A_blue / A_red = 8 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_blue_to_red_area_l505_50584


namespace NUMINAMATH_GPT_elevenRowTriangleTotalPieces_l505_50578

-- Definitions and problem statement
def numRodsInRow (n : ℕ) : ℕ := 3 * n

def sumFirstN (n : ℕ) : ℕ := n * (n + 1) / 2

def totalRods (rows : ℕ) : ℕ := 3 * (sumFirstN rows)

def totalConnectors (rows : ℕ) : ℕ := sumFirstN (rows + 1)

def totalPieces (rows : ℕ) : ℕ := totalRods rows + totalConnectors rows

-- Lean proof problem
theorem elevenRowTriangleTotalPieces : totalPieces 11 = 276 := 
by
  sorry

end NUMINAMATH_GPT_elevenRowTriangleTotalPieces_l505_50578


namespace NUMINAMATH_GPT_tan_angle_addition_l505_50562

theorem tan_angle_addition (x : ℝ) (h : Real.tan x = 2) : Real.tan (x + Real.pi / 3) = (5 * Real.sqrt 3 + 8) / -11 := by
  sorry

end NUMINAMATH_GPT_tan_angle_addition_l505_50562


namespace NUMINAMATH_GPT_part1_part2_l505_50513

variable {U : Type} [TopologicalSpace U]

-- Definitions of the sets A and B
def A : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}
def B (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ 3 - 2 * a}

-- Part (1): 
theorem part1 (U : Set ℝ) (a : ℝ) (h : (Aᶜ ∪ B a) = U) : a ≤ 0 := sorry

-- Part (2):
theorem part2 (a : ℝ) (h : ¬ (A ∩ B a = B a)) : a < 1 / 2 := sorry

end NUMINAMATH_GPT_part1_part2_l505_50513


namespace NUMINAMATH_GPT_find_sin_beta_l505_50521

variable (α β : ℝ)
variable (hα : 0 < α ∧ α < π/2) -- α is acute
variable (hβ : 0 < β ∧ β < π/2) -- β is acute

variable (hcosα : Real.cos α = 4/5)
variable (hcosαβ : Real.cos (α + β) = 5/13)

theorem find_sin_beta (hα : 0 < α ∧ α < π/2) (hβ : 0 < β ∧ β < π/2) 
    (hcosα : Real.cos α = 4/5) (hcosαβ : Real.cos (α + β) = 5/13) : 
    Real.sin β = 33/65 := 
sorry

end NUMINAMATH_GPT_find_sin_beta_l505_50521


namespace NUMINAMATH_GPT_max_alpha_value_l505_50556

variable (a b x y α : ℝ)

theorem max_alpha_value (h1 : a = 2 * b)
    (h2 : a^2 + y^2 = b^2 + x^2)
    (h3 : b^2 + x^2 = (a - x)^2 + (b - y)^2)
    (h4 : 0 ≤ x) (h5 : x < a) (h6 : 0 ≤ y) (h7 : y < b) :
    α = a / b → α^2 = 4 := 
by
  sorry

end NUMINAMATH_GPT_max_alpha_value_l505_50556


namespace NUMINAMATH_GPT_sum_of_factors_l505_50591

-- Define 36
def num : ℕ := 36

-- Prime factorization of 36
def prime_factorization : (ℕ × ℕ) := (2, 2) -- represents 2^2
def prime_factorization_2 : (ℕ × ℕ) := (3, 2) -- represents 3^2

-- Sum of divisors function formula
def sigma (n : ℕ) : ℕ := 
  (1 + 2 + 2^2) * (1 + 3 + 3^2)

-- Theorem: sum of the positive factors of 36 is 91
theorem sum_of_factors : sigma num = 91 := 
by {
  -- Skipping the proof steps
  sorry
}

end NUMINAMATH_GPT_sum_of_factors_l505_50591


namespace NUMINAMATH_GPT_fraction_of_number_l505_50596

theorem fraction_of_number (x y : ℝ) (h : x = 7/8) (z : ℝ) (h' : z = 48) : 
  x * z = 42 := by
  sorry

end NUMINAMATH_GPT_fraction_of_number_l505_50596


namespace NUMINAMATH_GPT_janet_home_time_l505_50571

def blocks_north := 3
def blocks_west := 7 * blocks_north
def blocks_south := blocks_north
def blocks_east := 2 * blocks_south -- Initially mistaken, recalculating needed
def remaining_blocks_west := blocks_west - blocks_east
def total_blocks_home := blocks_south + remaining_blocks_west
def walking_speed := 2 -- blocks per minute

theorem janet_home_time :
  (blocks_south + remaining_blocks_west) / walking_speed = 9 := by
  -- We assume that Lean can handle the arithmetic properly here.
  sorry

end NUMINAMATH_GPT_janet_home_time_l505_50571


namespace NUMINAMATH_GPT_algebraic_expression_value_l505_50554

theorem algebraic_expression_value (x : ℝ) (h : x = Real.sqrt 19 - 1) : x^2 + 2 * x + 2 = 20 := by
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l505_50554


namespace NUMINAMATH_GPT_Carlos_gave_Rachel_21_blocks_l505_50545

def initial_blocks : Nat := 58
def remaining_blocks : Nat := 37
def given_blocks : Nat := initial_blocks - remaining_blocks

theorem Carlos_gave_Rachel_21_blocks : given_blocks = 21 :=
by
  sorry

end NUMINAMATH_GPT_Carlos_gave_Rachel_21_blocks_l505_50545


namespace NUMINAMATH_GPT_rhind_papyrus_prob_l505_50520

theorem rhind_papyrus_prob (a₁ a₂ a₃ a₄ a₅ : ℝ) (q : ℝ) 
  (h_geom_seq : a₂ = a₁ * q ∧ a₃ = a₁ * q^2 ∧ a₄ = a₁ * q^3 ∧ a₅ = a₁ * q^4)
  (h_loaves_sum : a₁ + a₂ + a₃ + a₄ + a₅ = 93)
  (h_condition : a₁ + a₂ = (3/4) * a₃) 
  (q_gt_one : q > 1) :
  a₃ = 12 :=
sorry

end NUMINAMATH_GPT_rhind_papyrus_prob_l505_50520


namespace NUMINAMATH_GPT_centrally_symmetric_equidecomposable_l505_50543

-- Assume we have a type for Polyhedra
variable (Polyhedron : Type)

-- Conditions
variable (sameVolume : Polyhedron → Polyhedron → Prop)
variable (centrallySymmetricFaces : Polyhedron → Prop)
variable (equidecomposable : Polyhedron → Polyhedron → Prop)

-- Theorem statement
theorem centrally_symmetric_equidecomposable 
  (P Q : Polyhedron) 
  (h1 : sameVolume P Q) 
  (h2 : centrallySymmetricFaces P) 
  (h3 : centrallySymmetricFaces Q) :
  equidecomposable P Q := 
sorry

end NUMINAMATH_GPT_centrally_symmetric_equidecomposable_l505_50543


namespace NUMINAMATH_GPT_remaining_sausage_meat_l505_50576

-- Define the conditions
def total_meat_pounds : ℕ := 10
def sausage_links : ℕ := 40
def links_eaten_by_Brandy : ℕ := 12
def pounds_to_ounces : ℕ := 16

-- Calculate the remaining sausage meat and prove the correctness
theorem remaining_sausage_meat :
  (total_meat_pounds * pounds_to_ounces - links_eaten_by_Brandy * (total_meat_pounds * pounds_to_ounces / sausage_links)) = 112 :=
by
  sorry

end NUMINAMATH_GPT_remaining_sausage_meat_l505_50576


namespace NUMINAMATH_GPT_hypotenuse_45_45_90_l505_50579

theorem hypotenuse_45_45_90 (leg : ℝ) (h_leg : leg = 10) (angle : ℝ) (h_angle : angle = 45) :
  ∃ hypotenuse : ℝ, hypotenuse = leg * Real.sqrt 2 :=
by
  use 10 * Real.sqrt 2
  sorry

end NUMINAMATH_GPT_hypotenuse_45_45_90_l505_50579


namespace NUMINAMATH_GPT_map_x_eq_3_and_y_eq_2_under_z_squared_to_uv_l505_50548

theorem map_x_eq_3_and_y_eq_2_under_z_squared_to_uv :
  (∀ (z : ℂ), (z = 3 + I * z.im) → ((z^2).re = 9 - (9*z.im^2) / 36)) ∧
  (∀ (z : ℂ), (z = z.re + I * 2) → ((z^2).re = (4*z.re^2) / 16 - 4)) :=
by 
  sorry

end NUMINAMATH_GPT_map_x_eq_3_and_y_eq_2_under_z_squared_to_uv_l505_50548


namespace NUMINAMATH_GPT_remainder_add_l505_50581

theorem remainder_add (a b : ℤ) (n m : ℤ) 
  (ha : a = 60 * n + 41) 
  (hb : b = 45 * m + 14) : 
  (a + b) % 15 = 10 := by 
  sorry

end NUMINAMATH_GPT_remainder_add_l505_50581


namespace NUMINAMATH_GPT_inscribed_circle_radius_l505_50566

variable (A p s r : ℝ)

-- Condition: Area is twice the perimeter
def twice_perimeter_condition : Prop := A = 2 * p

-- Condition: The formula connecting the area, inradius, and semiperimeter
def area_inradius_semiperimeter_relation : Prop := A = r * s

-- Condition: The perimeter is twice the semiperimeter
def perimeter_semiperimeter_relation : Prop := p = 2 * s

-- Prove the radius of the inscribed circle is 4
theorem inscribed_circle_radius (h1 : twice_perimeter_condition A p)
                                (h2 : area_inradius_semiperimeter_relation A r s)
                                (h3 : perimeter_semiperimeter_relation p s) :
  r = 4 :=
by
  sorry

end NUMINAMATH_GPT_inscribed_circle_radius_l505_50566


namespace NUMINAMATH_GPT_shoe_size_combination_l505_50519

theorem shoe_size_combination (J A : ℕ) (hJ : J = 7) (hA : A = 2 * J) : J + A = 21 := by
  sorry

end NUMINAMATH_GPT_shoe_size_combination_l505_50519


namespace NUMINAMATH_GPT_geometric_probability_l505_50568

noncomputable def probability_point_within_rectangle (l w : ℝ) (A_rectangle A_circle : ℝ) : ℝ :=
  A_rectangle / A_circle

theorem geometric_probability (l w : ℝ) (r : ℝ) (A_rectangle : ℝ) (h_length : l = 4) 
  (h_width : w = 3) (h_radius : r = 2.5) (h_area_rectangle : A_rectangle = 12) :
  A_rectangle / (Real.pi * r^2) = 48 / (25 * Real.pi) :=
by
  sorry

end NUMINAMATH_GPT_geometric_probability_l505_50568


namespace NUMINAMATH_GPT_average_eq_5_times_non_zero_l505_50546

theorem average_eq_5_times_non_zero (x : ℝ) (h1 : x ≠ 0) (h2 : (x + x^2) / 2 = 5 * x) : x = 9 := 
by sorry

end NUMINAMATH_GPT_average_eq_5_times_non_zero_l505_50546


namespace NUMINAMATH_GPT_daily_sales_profit_45_selling_price_for_1200_profit_l505_50544

-- Definitions based on given conditions

def cost_price : ℤ := 30
def base_selling_price : ℤ := 40
def base_sales_volume : ℤ := 80
def price_increase_effect : ℤ := 2
def max_selling_price : ℤ := 55

-- Part (1): Prove that for a selling price of 45 yuan, the daily sales profit is 1050 yuan.
theorem daily_sales_profit_45 :
  let selling_price := 45
  let increase_in_price := selling_price - base_selling_price
  let decrease_in_volume := increase_in_price * price_increase_effect
  let new_sales_volume := base_sales_volume - decrease_in_volume
  let profit_per_item := selling_price - cost_price
  let daily_profit := profit_per_item * new_sales_volume
  daily_profit = 1050 := by sorry

-- Part (2): Prove that to achieve a daily profit of 1200 yuan, the selling price should be 50 yuan.
theorem selling_price_for_1200_profit :
  let target_profit := 1200
  ∃ (selling_price : ℤ), 
  let increase_in_price := selling_price - base_selling_price
  let decrease_in_volume := increase_in_price * price_increase_effect
  let new_sales_volume := base_sales_volume - decrease_in_volume
  let profit_per_item := selling_price - cost_price
  let daily_profit := profit_per_item * new_sales_volume
  daily_profit = target_profit ∧ selling_price ≤ max_selling_price ∧ selling_price = 50 := by sorry

end NUMINAMATH_GPT_daily_sales_profit_45_selling_price_for_1200_profit_l505_50544


namespace NUMINAMATH_GPT_find_exponent_l505_50522

theorem find_exponent (n : ℕ) (some_number : ℕ) (h1 : n = 27) 
  (h2 : 2 ^ (2 * n) + 2 ^ (2 * n) + 2 ^ (2 * n) + 2 ^ (2 * n) = 4 ^ some_number) :
  some_number = 28 :=
by 
  sorry

end NUMINAMATH_GPT_find_exponent_l505_50522


namespace NUMINAMATH_GPT_comparing_exponents_l505_50503

theorem comparing_exponents {a b : ℝ} (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : Real.exp a + 2 * a = Real.exp b + 3 * b) : a > b :=
sorry

end NUMINAMATH_GPT_comparing_exponents_l505_50503


namespace NUMINAMATH_GPT_fisher_needed_score_l505_50514

-- Condition 1: To have an average of at least 85% over all four quarters
def average_score_threshold := 85
def total_score := 4 * average_score_threshold

-- Condition 2: Fisher's scores for the first three quarters
def first_three_scores := [82, 77, 75]
def current_total_score := first_three_scores.sum

-- Define the Lean statement to prove
theorem fisher_needed_score : ∃ x, current_total_score + x = total_score ∧ x = 106 := by
  sorry

end NUMINAMATH_GPT_fisher_needed_score_l505_50514


namespace NUMINAMATH_GPT_phone_not_answered_prob_l505_50537

noncomputable def P_not_answered_within_4_rings : ℝ :=
  let P1 := 1 - 0.1
  let P2 := 1 - 0.3
  let P3 := 1 - 0.4
  let P4 := 1 - 0.1
  P1 * P2 * P3 * P4

theorem phone_not_answered_prob : 
  P_not_answered_within_4_rings = 0.3402 := 
by 
  -- The detailed steps and proof will be implemented here 
  sorry

end NUMINAMATH_GPT_phone_not_answered_prob_l505_50537


namespace NUMINAMATH_GPT_Laura_more_than_200_paperclips_on_Friday_l505_50595

theorem Laura_more_than_200_paperclips_on_Friday:
  ∀ (n : ℕ), (n = 4 ∨ n = 0 ∨ n ≥ 1 ∧ (n - 1 = 0 ∨ n = 1) → 4 * 3 ^ n > 200) :=
by
  sorry

end NUMINAMATH_GPT_Laura_more_than_200_paperclips_on_Friday_l505_50595


namespace NUMINAMATH_GPT_n_salary_eq_260_l505_50594

variables (m n : ℕ)
axiom total_salary : m + n = 572
axiom m_salary : m = 120 * n / 100

theorem n_salary_eq_260 : n = 260 :=
by
  sorry

end NUMINAMATH_GPT_n_salary_eq_260_l505_50594


namespace NUMINAMATH_GPT_solve_system_equations_l505_50583

theorem solve_system_equations (x y z b : ℝ) :
  (3 * x * y * z - x^3 - y^3 - z^3 = b^3) ∧ 
  (x + y + z = 2 * b) ∧ 
  (x^2 + y^2 + z^2 = b^2) → 
  b = 0 ∧ (∃ t, (x = 0 ∧ y = t ∧ z = -t) ∨ 
                (x = t ∧ y = 0 ∧ z = -t) ∨ 
                (x = -t ∧ y = t ∧ z = 0)) :=
by
  sorry -- Proof to be provided

end NUMINAMATH_GPT_solve_system_equations_l505_50583


namespace NUMINAMATH_GPT_probability_is_7_over_26_l505_50515

section VowelProbability

def num_students : Nat := 26

def is_vowel (c : Char) : Bool :=
  c = 'A' || c = 'E' || c = 'I' || c = 'O' || c = 'U' || c = 'Y' || c = 'W'

def num_vowels : Nat := 7

def probability_of_vowel_initials : Rat :=
  (num_vowels : Nat) / (num_students : Nat)

theorem probability_is_7_over_26 :
  probability_of_vowel_initials = 7 / 26 := by
  sorry

end VowelProbability

end NUMINAMATH_GPT_probability_is_7_over_26_l505_50515


namespace NUMINAMATH_GPT_find_x_for_collinear_vectors_l505_50502

noncomputable def collinear_vectors (a : ℝ × ℝ) (b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b = (k * a.1, k * a.2)

theorem find_x_for_collinear_vectors : ∀ (x : ℝ), collinear_vectors (2, -3) (x, 6) → x = -4 := by
  intros x h
  sorry

end NUMINAMATH_GPT_find_x_for_collinear_vectors_l505_50502


namespace NUMINAMATH_GPT_odd_function_f_a_zero_l505_50551

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + (a + 1) * Real.cos x + x

theorem odd_function_f_a_zero (a : ℝ) (h_odd : ∀ x : ℝ, f a (-x) = -f a x) : f a a = 0 := 
sorry

end NUMINAMATH_GPT_odd_function_f_a_zero_l505_50551


namespace NUMINAMATH_GPT_zero_of_f_a_neg_sqrt2_range_of_a_no_zero_l505_50500

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x < 2 then 2^x + a else a - x

theorem zero_of_f_a_neg_sqrt2 : 
  ∀ x, f x (- Real.sqrt 2) = 0 ↔ x = 1/2 :=
by
  sorry

theorem range_of_a_no_zero :
  ∀ a, (¬∃ x, f x a = 0) ↔ a ∈ Set.Iic (-4) ∪ Set.Ico 0 2 :=
by
  sorry

end NUMINAMATH_GPT_zero_of_f_a_neg_sqrt2_range_of_a_no_zero_l505_50500


namespace NUMINAMATH_GPT_mehki_age_l505_50531

theorem mehki_age (Z J M : ℕ) (h1 : Z = 6) (h2 : J = Z - 4) (h3 : M = 2 * (J + Z)) : M = 16 := by
  sorry

end NUMINAMATH_GPT_mehki_age_l505_50531


namespace NUMINAMATH_GPT_glycerin_percentage_l505_50516

theorem glycerin_percentage (x : ℝ) 
  (h1 : 100 * 0.75 = 75)
  (h2 : 75 + 75 = 100)
  (h3 : 75 * 0.30 + (x/100) * 75 = 75) : x = 70 :=
by
  sorry

end NUMINAMATH_GPT_glycerin_percentage_l505_50516


namespace NUMINAMATH_GPT_angle_conversion_l505_50532

theorem angle_conversion : (1 : ℝ) * (π / 180) * (-225) = - (5 * π / 4) :=
by
  sorry

end NUMINAMATH_GPT_angle_conversion_l505_50532


namespace NUMINAMATH_GPT_least_number_to_add_1055_to_div_by_23_l505_50505

theorem least_number_to_add_1055_to_div_by_23 : ∃ k : ℕ, (1055 + k) % 23 = 0 ∧ k = 3 :=
by
  sorry

end NUMINAMATH_GPT_least_number_to_add_1055_to_div_by_23_l505_50505


namespace NUMINAMATH_GPT_prime_factors_count_900_l505_50593

theorem prime_factors_count_900 : 
  ∃ (S : Finset ℕ), (∀ x ∈ S, Nat.Prime x ∧ x ∣ 900) ∧ S.card = 3 :=
by 
  sorry

end NUMINAMATH_GPT_prime_factors_count_900_l505_50593


namespace NUMINAMATH_GPT_no_such_function_exists_l505_50582

theorem no_such_function_exists :
  ¬ ∃ (f : ℝ → ℝ), (∀ x : ℝ, 2 * f (Real.cos x) = f (Real.sin x) + Real.sin x) :=
by
  sorry

end NUMINAMATH_GPT_no_such_function_exists_l505_50582


namespace NUMINAMATH_GPT_negation_of_p_l505_50590

def is_odd (x : ℤ) : Prop := x % 2 = 1
def is_even (x : ℤ) : Prop := x % 2 = 0

def A (x : ℤ) : Prop := is_odd x
def B (x : ℤ) : Prop := is_even x
def p : Prop := ∀ x, A x → B (2 * x)

theorem negation_of_p : ¬ p ↔ ∃ x, A x ∧ ¬ B (2 * x) :=
by
  -- problem statement equivalent in Lean 4
  sorry

end NUMINAMATH_GPT_negation_of_p_l505_50590


namespace NUMINAMATH_GPT_solve_equation1_solve_equation2_l505_50588

theorem solve_equation1 (x : ℝ) : 4 - x = 3 * (2 - x) ↔ x = 1 :=
by sorry

theorem solve_equation2 (x : ℝ) : (2 * x - 1) / 2 - (2 * x + 5) / 3 = (6 * x - 1) / 6 - 1 ↔ x = -3 / 2 :=
by sorry

end NUMINAMATH_GPT_solve_equation1_solve_equation2_l505_50588


namespace NUMINAMATH_GPT_determine_sum_of_digits_l505_50552

theorem determine_sum_of_digits (x y : ℕ) (hx : x < 10) (hy : y < 10)
  (h : ∃ a b c d : ℕ, 
       a = 30 + x ∧ b = 10 * y + 4 ∧
       c = (a * (b % 10)) % 100 ∧ 
       d = (a * (b % 10)) / 100 ∧ 
       10 * d + c = 156) :
  x + y = 13 :=
by
  sorry

end NUMINAMATH_GPT_determine_sum_of_digits_l505_50552


namespace NUMINAMATH_GPT_symm_diff_complement_symm_diff_union_subset_symm_diff_inter_subset_l505_50560

-- Definitions of sequences of events and symmetric difference
variable (A : ℕ → Set α) (B : ℕ → Set α)

-- Definition of symmetric difference
def symm_diff (S T : Set α) : Set α := (S \ T) ∪ (T \ S)

-- Theorems to be proven
theorem symm_diff_complement (A1 B1 : Set α) :
  symm_diff A1 B1 = symm_diff (Set.compl A1) (Set.compl B1) := sorry

theorem symm_diff_union_subset :
  symm_diff (⋃ n, A n) (⋃ n, B n) ⊆ ⋃ n, symm_diff (A n) (B n) := sorry

theorem symm_diff_inter_subset :
  symm_diff (⋂ n, A n) (⋂ n, B n) ⊆ ⋃ n, symm_diff (A n) (B n) := sorry

end NUMINAMATH_GPT_symm_diff_complement_symm_diff_union_subset_symm_diff_inter_subset_l505_50560


namespace NUMINAMATH_GPT_shifted_quadratic_eq_l505_50508

-- Define the original quadratic function
def orig_fn (x : ℝ) : ℝ := -x^2

-- Define the function after shifting 1 unit to the left
def shifted_left_fn (x : ℝ) : ℝ := - (x + 1)^2

-- Define the final function after also shifting 3 units up
def final_fn (x : ℝ) : ℝ := - (x + 1)^2 + 3

-- Prove the final function is the correctly transformed function from the original one
theorem shifted_quadratic_eq : ∀ (x : ℝ), final_fn x = - (x + 1)^2 + 3 :=
by 
  intro x
  sorry

end NUMINAMATH_GPT_shifted_quadratic_eq_l505_50508


namespace NUMINAMATH_GPT_angle_RPS_is_27_l505_50570

theorem angle_RPS_is_27 (PQ BP PR QS QS PSQ QPRS : ℝ) :
  PQ + PSQ + QS = 180 ∧ 
  QS = 48 ∧ 
  PSQ = 38 ∧ 
  QPRS = 67
  → (QS - QPRS = 27) := 
by {
  sorry
}

end NUMINAMATH_GPT_angle_RPS_is_27_l505_50570


namespace NUMINAMATH_GPT_solution_product_l505_50524

theorem solution_product (p q : ℝ) (hpq : p ≠ q) (h1 : (x-3)*(3*x+18) = x^2-15*x+54) (hp : (x - p) * (x - q) = x^2 - 12 * x + 54) :
  (p + 2) * (q + 2) = -80 := sorry

end NUMINAMATH_GPT_solution_product_l505_50524


namespace NUMINAMATH_GPT_octal_addition_correct_l505_50555

def octal_to_decimal (n : ℕ) : ℕ := 
  /- function to convert an octal number to decimal goes here -/
  sorry

def decimal_to_octal (n : ℕ) : ℕ :=
  /- function to convert a decimal number to octal goes here -/
  sorry

theorem octal_addition_correct :
  let a := 236 
  let b := 521
  let c := 74
  let sum_decimal := octal_to_decimal a + octal_to_decimal b + octal_to_decimal c
  decimal_to_octal sum_decimal = 1063 :=
by
  sorry

end NUMINAMATH_GPT_octal_addition_correct_l505_50555


namespace NUMINAMATH_GPT_ellipse_range_x_plus_y_l505_50587

/-- The problem conditions:
Given any point P(x, y) on the ellipse x^2 / 144 + y^2 / 25 = 1,
prove that the range of values for x + y is [-13, 13].
-/
theorem ellipse_range_x_plus_y (x y : ℝ) (h : (x^2 / 144) + (y^2 / 25) = 1) : 
  -13 ≤ x + y ∧ x + y ≤ 13 := sorry

end NUMINAMATH_GPT_ellipse_range_x_plus_y_l505_50587


namespace NUMINAMATH_GPT_symmetric_parabola_equation_l505_50567

theorem symmetric_parabola_equation (x y : ℝ) (h : y^2 = 2 * x) : (y^2 = -2 * (x + 2)) :=
by
  sorry

end NUMINAMATH_GPT_symmetric_parabola_equation_l505_50567


namespace NUMINAMATH_GPT_problem_solution_l505_50518

variable (a b c : ℝ)

theorem problem_solution (h1 : 0 < a) (h2 : a ≤ b) (h3 : b ≤ c) :
  a + b ≤ 3 * c := 
sorry

end NUMINAMATH_GPT_problem_solution_l505_50518


namespace NUMINAMATH_GPT_regular_ducks_sold_l505_50527

theorem regular_ducks_sold (R : ℕ) (h1 : 3 * R + 5 * 185 = 1588) : R = 221 :=
by {
  sorry
}

end NUMINAMATH_GPT_regular_ducks_sold_l505_50527


namespace NUMINAMATH_GPT_problem_inequality_a3_a2_problem_inequality_relaxed_general_inequality_l505_50512

theorem problem_inequality_a3_a2 (a : ℝ) (ha : a > 1) : 
  a^3 + (1 / a^3) > a^2 + (1 / a^2) := 
sorry

theorem problem_inequality_relaxed (a : ℝ) (ha1 : a > 0) (ha2 : a ≠ 1) : 
  a^3 + (1 / a^3) > a^2 + (1 / a^2) := 
sorry

theorem general_inequality (a : ℝ) (m n : ℝ) (ha1 : a > 0) (ha2 : a ≠ 1) (hmn1 : m > n) (hmn2 : n > 0) : 
  a^m + (1 / a^m) > a^n + (1 / a^n) := 
sorry

end NUMINAMATH_GPT_problem_inequality_a3_a2_problem_inequality_relaxed_general_inequality_l505_50512


namespace NUMINAMATH_GPT_required_sampling_methods_l505_50507

-- Defining the given conditions
def total_households : Nat := 2000
def farmer_households : Nat := 1800
def worker_households : Nat := 100
def intellectual_households : Nat := total_households - farmer_households - worker_households
def sample_size : Nat := 40

-- Statement representing the proof problem
theorem required_sampling_methods :
  stratified_sampling_needed ∧ systematic_sampling_needed ∧ simple_random_sampling_needed :=
sorry

end NUMINAMATH_GPT_required_sampling_methods_l505_50507


namespace NUMINAMATH_GPT_range_of_8x_plus_y_l505_50538

theorem range_of_8x_plus_y (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_condition : 1 / x + 2 / y = 2) : 8 * x + y ≥ 9 :=
by
  sorry

end NUMINAMATH_GPT_range_of_8x_plus_y_l505_50538


namespace NUMINAMATH_GPT_pebbles_divisibility_impossibility_l505_50530

def initial_pebbles (K A P D : Nat) := K + A + P + D

theorem pebbles_divisibility_impossibility 
  (K A P D : Nat)
  (hK : K = 70)
  (hA : A = 30)
  (hP : P = 21)
  (hD : D = 45) :
  ¬ (∃ n : Nat, initial_pebbles K A P D = 4 * n) :=
by
  sorry

end NUMINAMATH_GPT_pebbles_divisibility_impossibility_l505_50530


namespace NUMINAMATH_GPT_average_marks_correct_l505_50598

-- Define constants for the marks in each subject
def english_marks : ℕ := 76
def math_marks : ℕ := 65
def physics_marks : ℕ := 82
def chemistry_marks : ℕ := 67
def biology_marks : ℕ := 85

-- Define the total number of subjects
def num_subjects : ℕ := 5

-- Define the total marks as the sum of individual subjects
def total_marks : ℕ := english_marks + math_marks + physics_marks + chemistry_marks + biology_marks

-- Define the average marks
def average_marks : ℕ := total_marks / num_subjects

-- Prove that the average marks is as expected
theorem average_marks_correct : average_marks = 75 :=
by {
  -- skip the proof
  sorry
}

end NUMINAMATH_GPT_average_marks_correct_l505_50598


namespace NUMINAMATH_GPT_kids_stayed_home_l505_50511

open Nat

theorem kids_stayed_home (kids_camp : ℕ) (additional_kids_home : ℕ) (total_kids_home : ℕ) 
  (h1 : kids_camp = 202958) 
  (h2 : additional_kids_home = 574664) 
  (h3 : total_kids_home = kids_camp + additional_kids_home) : 
  total_kids_home = 777622 := 
by 
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_kids_stayed_home_l505_50511


namespace NUMINAMATH_GPT_hat_price_reduction_l505_50506

theorem hat_price_reduction (original_price : ℚ) (r1 r2 : ℚ) (price_after_reductions : ℚ) :
  original_price = 12 → r1 = 0.20 → r2 = 0.25 →
  price_after_reductions = original_price * (1 - r1) * (1 - r2) →
  price_after_reductions = 7.20 :=
by
  intros original_price_eq r1_eq r2_eq price_calc_eq
  sorry

end NUMINAMATH_GPT_hat_price_reduction_l505_50506


namespace NUMINAMATH_GPT_find_A_B_l505_50599

theorem find_A_B (A B : ℝ) (h : ∀ x : ℝ, x ≠ 5 ∧ x ≠ -2 → 
  (A / (x - 5) + B / (x + 2) = (5 * x - 4) / (x^2 - 3 * x - 10))) :
  A = 3 ∧ B = 2 :=
sorry

end NUMINAMATH_GPT_find_A_B_l505_50599


namespace NUMINAMATH_GPT_product_of_functions_l505_50575

noncomputable def f (x : ℝ) : ℝ := 2 * x
noncomputable def g (x : ℝ) : ℝ := -(3 * x - 1) / x

theorem product_of_functions (x : ℝ) (h : x ≠ 0) : f x * g x = -6 * x + 2 := by
  sorry

end NUMINAMATH_GPT_product_of_functions_l505_50575


namespace NUMINAMATH_GPT_directrix_of_parabola_l505_50557

theorem directrix_of_parabola (y x : ℝ) (p : ℝ) (h₁ : y = 8 * x ^ 2) (h₂ : y = 4 * p * x) : 
  p = 2 ∧ (y = -p ↔ y = -2) :=
by
  sorry

end NUMINAMATH_GPT_directrix_of_parabola_l505_50557


namespace NUMINAMATH_GPT_determine_omega_l505_50535

noncomputable def f (ω : ℝ) (ϕ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + ϕ)

-- Conditions
variables (ω : ℝ) (ϕ : ℝ)
axiom omega_pos : ω > 0
axiom phi_bound : abs ϕ < Real.pi / 2
axiom symm_condition1 : ∀ x, f ω ϕ (Real.pi / 4 - x) = -f ω ϕ (Real.pi / 4 + x)
axiom symm_condition2 : ∀ x, f ω ϕ (-Real.pi / 2 - x) = f ω ϕ x
axiom monotonic_condition : ∀ x1 x2, 0 < x1 → x1 < x2 → x2 < Real.pi / 8 → f ω ϕ x1 < f ω ϕ x2

theorem determine_omega : ω = 1 ∨ ω = 5 :=
sorry

end NUMINAMATH_GPT_determine_omega_l505_50535


namespace NUMINAMATH_GPT_probability_m_eq_kn_l505_50574

/- 
Define the conditions and question in Lean 4 -/
def die_faces : Finset ℕ := {1, 2, 3, 4, 5, 6}

def valid_rolls : Finset (ℕ × ℕ) := Finset.product die_faces die_faces

def events_satisfying_condition : Finset (ℕ × ℕ) :=
  {(1, 1), (2, 1), (2, 2), (3, 1), (3, 3), (4, 1), (4, 2), (4, 4), 
   (5, 1), (5, 5), (6, 1), (6, 2), (6, 3), (6, 6)}

theorem probability_m_eq_kn (k : ℕ) (h : k > 0) :
  (events_satisfying_condition.card : ℚ) / (valid_rolls.card : ℚ) = 7/18 := by
  sorry

end NUMINAMATH_GPT_probability_m_eq_kn_l505_50574

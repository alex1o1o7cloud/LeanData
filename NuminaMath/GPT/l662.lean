import Mathlib

namespace value_of_k_h_5_l662_66242

def h (x : ℝ) : ℝ := 4 * x + 6
def k (x : ℝ) : ℝ := 6 * x - 8

theorem value_of_k_h_5 : k (h 5) = 148 :=
by
  have h5 : h 5 = 4 * 5 + 6 := rfl
  simp [h5, h, k]
  sorry

end value_of_k_h_5_l662_66242


namespace lcm_36_105_l662_66244

noncomputable def factorize_36 : List (ℕ × ℕ) := [(2, 2), (3, 2)]
noncomputable def factorize_105 : List (ℕ × ℕ) := [(3, 1), (5, 1), (7, 1)]

theorem lcm_36_105 : Nat.lcm 36 105 = 1260 :=
by
  have h_36 : 36 = 2^2 * 3^2 := by norm_num
  have h_105 : 105 = 3^1 * 5^1 * 7^1 := by norm_num
  sorry

end lcm_36_105_l662_66244


namespace albert_age_l662_66294

theorem albert_age
  (A : ℕ)
  (dad_age : ℕ)
  (h1 : dad_age = 48)
  (h2 : dad_age - 4 = 4 * (A - 4)) :
  A = 15 :=
by
  sorry

end albert_age_l662_66294


namespace boat_speed_still_water_l662_66266

def effective_upstream_speed (b c : ℝ) : ℝ := b - c
def effective_downstream_speed (b c : ℝ) : ℝ := b + c

theorem boat_speed_still_water :
  ∃ b c : ℝ, effective_upstream_speed b c = 9 ∧ effective_downstream_speed b c = 15 ∧ b = 12 :=
by {
  sorry
}

end boat_speed_still_water_l662_66266


namespace sufficient_condition_for_solution_l662_66249

theorem sufficient_condition_for_solution 
  (a : ℝ) (f g h : ℝ → ℝ) (h_a : 1 < a)
  (h_fg_h : ∀ x : ℝ, 0 ≤ f x + g x + h x) 
  (h_common_root : ∃ x : ℝ, f x = 0 ∧ g x = 0 ∧ h x = 0) : 
  ∃ x : ℝ, a^(f x) + a^(g x) + a^(h x) = 3 := 
by
  sorry

end sufficient_condition_for_solution_l662_66249


namespace gcd_180_450_l662_66219

theorem gcd_180_450 : Nat.gcd 180 450 = 90 :=
by
  sorry

end gcd_180_450_l662_66219


namespace train_passes_jogger_in_approx_36_seconds_l662_66283

noncomputable def jogger_speed_kmph : ℝ := 8
noncomputable def train_speed_kmph : ℝ := 55
noncomputable def distance_ahead_m : ℝ := 340
noncomputable def train_length_m : ℝ := 130

noncomputable def kmph_to_mps (speed_kmph : ℝ) : ℝ :=
  (speed_kmph * 1000) / 3600

noncomputable def jogger_speed_mps : ℝ :=
  kmph_to_mps jogger_speed_kmph

noncomputable def train_speed_mps : ℝ :=
  kmph_to_mps train_speed_kmph

noncomputable def relative_speed_mps : ℝ :=
  train_speed_mps - jogger_speed_mps

noncomputable def total_distance_m : ℝ :=
  distance_ahead_m + train_length_m

noncomputable def time_to_pass_jogger_s : ℝ :=
  total_distance_m / relative_speed_mps

theorem train_passes_jogger_in_approx_36_seconds : 
  abs (time_to_pass_jogger_s - 36) < 1 := 
sorry

end train_passes_jogger_in_approx_36_seconds_l662_66283


namespace distinct_students_27_l662_66262

variable (students_euler : ℕ) (students_fibonacci : ℕ) (students_gauss : ℕ) (overlap_euler_fibonacci : ℕ)

-- Conditions
def conditions : Prop := 
  students_euler = 12 ∧ 
  students_fibonacci = 10 ∧ 
  students_gauss = 11 ∧ 
  overlap_euler_fibonacci = 3

-- Question and correct answer
def distinct_students (students_euler students_fibonacci students_gauss overlap_euler_fibonacci : ℕ) : ℕ :=
  (students_euler + students_fibonacci + students_gauss) - overlap_euler_fibonacci

theorem distinct_students_27 : conditions students_euler students_fibonacci students_gauss overlap_euler_fibonacci →
  distinct_students students_euler students_fibonacci students_gauss overlap_euler_fibonacci = 27 :=
by
  sorry

end distinct_students_27_l662_66262


namespace real_nums_inequality_l662_66200

theorem real_nums_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h : a ^ 2000 + b ^ 2000 = a ^ 1998 + b ^ 1998) :
  a ^ 2 + b ^ 2 ≤ 2 :=
sorry

end real_nums_inequality_l662_66200


namespace solve_equation_l662_66223

theorem solve_equation (x : ℚ) (h₁ : x ≠ 1) (h₂ : x ≠ -1) : 
  (2 * x / (x + 1) - 2 = 3 / (x^2 - 1)) → x = -0.5 := 
by
  sorry

end solve_equation_l662_66223


namespace square_of_binomial_l662_66215

theorem square_of_binomial (c : ℝ) (h : c = 3600) :
  ∃ a : ℝ, (x : ℝ) → (x + a)^2 = x^2 + 120 * x + c := by
  sorry

end square_of_binomial_l662_66215


namespace ordered_pair_solution_l662_66293

theorem ordered_pair_solution :
  ∃ (x y : ℚ), 
  (3 * x - 2 * y = (6 - 2 * x) + (6 - 2 * y)) ∧
  (x + 3 * y = (2 * x + 1) - (2 * y + 1)) ∧
  x = 12 / 5 ∧
  y = 12 / 25 :=
by
  sorry

end ordered_pair_solution_l662_66293


namespace total_prep_time_is_8_l662_66251

-- Defining the conditions
def prep_vocab_sentence_eq := 3
def prep_analytical_writing := 2
def prep_quantitative_reasoning := 3

-- Stating the total preparation time
def total_prep_time := prep_vocab_sentence_eq + prep_analytical_writing + prep_quantitative_reasoning

-- The Lean statement of the mathematical proof problem
theorem total_prep_time_is_8 : total_prep_time = 8 := by
  sorry

end total_prep_time_is_8_l662_66251


namespace remove_toothpicks_l662_66222

-- Definitions based on problem conditions
def toothpicks := 40
def triangles := 40
def initial_triangulation := True
def additional_condition := True

-- Statement to be proved
theorem remove_toothpicks :
  initial_triangulation ∧ additional_condition ∧ (triangles > 40) → ∃ (t: ℕ), t = 15 :=
by
  sorry

end remove_toothpicks_l662_66222


namespace value_of_f_of_x_minus_3_l662_66235

theorem value_of_f_of_x_minus_3 (x : ℝ) (f : ℝ → ℝ) (h : ∀ y : ℝ, f y = y^2) : f (x - 3) = x^2 - 6*x + 9 :=
by
  sorry

end value_of_f_of_x_minus_3_l662_66235


namespace count_of_integer_values_not_satisfying_inequality_l662_66274

theorem count_of_integer_values_not_satisfying_inequality :
  ∃ n : ℕ, n = 8 ∧ ∀ x : ℤ, (3 * x^2 + 11 * x + 10 ≤ 17) ↔ (x = -7 ∨ x = -6 ∨ x = -5 ∨ x = -4 ∨ x = -3 ∨ x = -2 ∨ x = -1 ∨ x = 0) :=
by sorry

end count_of_integer_values_not_satisfying_inequality_l662_66274


namespace brianna_more_chocolates_than_alix_l662_66267

def Nick_ClosetA : ℕ := 10
def Nick_ClosetB : ℕ := 6
def Alix_ClosetA : ℕ := 3 * Nick_ClosetA
def Alix_ClosetB : ℕ := 3 * Nick_ClosetA
def Mom_Takes_From_AlixA : ℚ := (1/4:ℚ) * Alix_ClosetA
def Brianna_ClosetA : ℚ := 2 * (Nick_ClosetA + Alix_ClosetA - Mom_Takes_From_AlixA)
def Brianna_ClosetB_after : ℕ := 18
def Brianna_ClosetB : ℚ := Brianna_ClosetB_after / (0.8:ℚ)

def Brianna_Total : ℚ := Brianna_ClosetA + Brianna_ClosetB
def Alix_Total : ℚ := Alix_ClosetA + Alix_ClosetB
def Difference : ℚ := Brianna_Total - Alix_Total

theorem brianna_more_chocolates_than_alix : Difference = 35 := by
  sorry

end brianna_more_chocolates_than_alix_l662_66267


namespace heptagon_labeling_impossible_l662_66221

/-- 
  Let a heptagon be given with vertices labeled by integers a₁, a₂, a₃, a₄, a₅, a₆, a₇.
  The following two conditions are imposed:
  1. For every pair of consecutive vertices (aᵢ, aᵢ₊₁) (with indices mod 7), 
     at least one of aᵢ and aᵢ₊₁ divides the other.
  2. For every pair of non-consecutive vertices (aᵢ, aⱼ) where i ≠ j ± 1 mod 7, 
     neither aᵢ divides aⱼ nor aⱼ divides aᵢ. 

  Prove that such a labeling is impossible.
-/
theorem heptagon_labeling_impossible :
  ¬ ∃ (a : Fin 7 → ℕ),
    (∀ i : Fin 7, a i ∣ a ((i + 1) % 7) ∨ a ((i + 1) % 7) ∣ a i) ∧
    (∀ {i j : Fin 7}, (i ≠ j + 1 % 7) → (i ≠ j + 6 % 7) → ¬ (a i ∣ a j) ∧ ¬ (a j ∣ a i)) :=
sorry

end heptagon_labeling_impossible_l662_66221


namespace no_valid_n_l662_66205

theorem no_valid_n (n : ℕ) (h₁ : 100 ≤ n / 4) (h₂ : n / 4 ≤ 999) (h₃ : 100 ≤ 4 * n) (h₄ : 4 * n ≤ 999) : false := by
  sorry

end no_valid_n_l662_66205


namespace f_zero_eq_one_f_pos_all_f_increasing_l662_66201

noncomputable def f : ℝ → ℝ := sorry

axiom f_nonzero : f 0 ≠ 0
axiom f_pos : ∀ x, 0 < x → 1 < f x
axiom f_mul : ∀ a b : ℝ, f (a + b) = f a * f b

theorem f_zero_eq_one : f 0 = 1 :=
sorry

theorem f_pos_all : ∀ x : ℝ, 0 < f x :=
sorry

theorem f_increasing : ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂ :=
sorry

end f_zero_eq_one_f_pos_all_f_increasing_l662_66201


namespace find_remainder_l662_66278

-- Define the numbers
def a := 98134
def b := 98135
def c := 98136
def d := 98137
def e := 98138
def f := 98139

-- Theorem statement
theorem find_remainder :
  (a + b + c + d + e + f) % 9 = 3 :=
by {
  sorry
}

end find_remainder_l662_66278


namespace tens_digit_seven_last_digit_six_l662_66245

theorem tens_digit_seven_last_digit_six (n : ℕ) (h : ((n * n) / 10) % 10 = 7) :
  (n * n) % 10 = 6 :=
sorry

end tens_digit_seven_last_digit_six_l662_66245


namespace distance_down_correct_l662_66212

-- Conditions
def rate_up : ℕ := 5  -- rate on the way up (miles per day)
def time_up : ℕ := 2  -- time to travel up (days)
def rate_factor : ℕ := 3 / 2  -- factor for the rate on the way down
def time_down := time_up  -- time to travel down is the same

-- Formula for computation
def distance_up : ℕ := rate_up * time_up
def rate_down : ℕ := rate_up * rate_factor
def distance_down : ℕ := rate_down * time_down

-- Theorem to be proved
theorem distance_down_correct : distance_down = 15 := by
  sorry

end distance_down_correct_l662_66212


namespace minimum_tangent_length_l662_66209

theorem minimum_tangent_length
  (a b : ℝ)
  (h_circle : ∀ x y : ℝ, x^2 + y^2 + 2 * x - 4 * y + 3 = 0)
  (h_symmetry : ∀ x y : ℝ, (x + 1)^2 + (y - 2)^2 = 4 → 2 * a * x + b * y + 6 = 0) :
  ∃ t : ℝ, t = 2 :=
by sorry

end minimum_tangent_length_l662_66209


namespace sqrt_meaningful_range_l662_66263

theorem sqrt_meaningful_range (x : ℝ) : x + 2 ≥ 0 → x ≥ -2 :=
by 
  intro h
  linarith [h]

end sqrt_meaningful_range_l662_66263


namespace eval_f_at_10_l662_66208

def f (x : ℚ) : ℚ := (6 * x + 3) / (x - 2)

theorem eval_f_at_10 : f 10 = 63 / 8 :=
by
  sorry

end eval_f_at_10_l662_66208


namespace statistical_hypothesis_independence_l662_66289

def independence_test_statistical_hypothesis (A B: Prop) (independence_test: Prop) : Prop :=
  (independence_test ∧ A ∧ B) → (A = B)

theorem statistical_hypothesis_independence (A B: Prop) (independence_test: Prop) :
  (independence_test ∧ A ∧ B) → (A = B) :=
by
  sorry

end statistical_hypothesis_independence_l662_66289


namespace divisibility_condition_l662_66280

theorem divisibility_condition (n : ℕ) : 
  13 ∣ (4 * 3^(2^n) + 3 * 4^(2^n)) ↔ Even n := 
sorry

end divisibility_condition_l662_66280


namespace ninas_money_l662_66224

theorem ninas_money (C M : ℝ) (h1 : 6 * C = M) (h2 : 8 * (C - 1.15) = M) : M = 27.6 := 
by
  sorry

end ninas_money_l662_66224


namespace hyperbola_asymptotes_equation_l662_66226

noncomputable def hyperbola_asymptotes (x y : ℝ) : Prop :=
  (x^2 / 4 - y^2 / 9 = 1) → (y = (3 / 2) * x) ∨ (y = -(3 / 2) * x)

-- Now we assert the theorem that states this
theorem hyperbola_asymptotes_equation :
  ∀ (x y : ℝ), hyperbola_asymptotes x y :=
by
  intros x y
  unfold hyperbola_asymptotes
  -- proof here
  sorry

end hyperbola_asymptotes_equation_l662_66226


namespace Marty_combinations_l662_66229

theorem Marty_combinations : 
  let colors := 4
  let decorations := 3
  colors * decorations = 12 :=
by
  sorry

end Marty_combinations_l662_66229


namespace find_y_l662_66239

theorem find_y (k p y : ℝ) (hk : k ≠ 0) (hp : p ≠ 0) 
  (h : (y - 2 * k)^2 - (y - 3 * k)^2 = 4 * k^2 - p) : 
  y = -(p + k^2) / (2 * k) :=
sorry

end find_y_l662_66239


namespace nguyen_fabric_yards_l662_66258

open Nat

theorem nguyen_fabric_yards :
  let fabric_per_pair := 8.5
  let pairs_needed := 7
  let fabric_still_needed := 49
  let total_fabric_needed := pairs_needed * fabric_per_pair
  let fabric_already_have := total_fabric_needed - fabric_still_needed
  let yards_of_fabric := fabric_already_have / 3
  yards_of_fabric = 3.5 := by
    sorry

end nguyen_fabric_yards_l662_66258


namespace ratio_ab_l662_66277

variable (x y a b : ℝ)
variable (h1 : 4 * x - 2 * y = a)
variable (h2 : 6 * y - 12 * x = b)
variable (h3 : b ≠ 0)

theorem ratio_ab : 4 * x - 2 * y = a ∧ 6 * y - 12 * x = b ∧ b ≠ 0 → a / b = -1 / 3 := by
  sorry

end ratio_ab_l662_66277


namespace length_to_width_ratio_l662_66233

-- Define the conditions: perimeter and length
variable (P : ℕ) (l : ℕ) (w : ℕ)

-- Given conditions
def conditions : Prop := (P = 100) ∧ (l = 40) ∧ (P = 2 * l + 2 * w)

-- The proposition we want to prove
def ratio : Prop := l / w = 4

-- The main theorem
theorem length_to_width_ratio (h : conditions P l w) : ratio l w :=
by sorry

end length_to_width_ratio_l662_66233


namespace john_total_payment_l662_66271

-- Definitions of the conditions
def yearly_cost_first_8_years : ℕ := 10000
def yearly_cost_9_to_18_years : ℕ := 2 * yearly_cost_first_8_years
def university_tuition : ℕ := 250000
def total_cost := (8 * yearly_cost_first_8_years) + (10 * yearly_cost_9_to_18_years) + university_tuition

-- John pays half of the total cost
def johns_total_cost := total_cost / 2

-- Theorem stating the total cost John pays
theorem john_total_payment : johns_total_cost = 265000 := by
  sorry

end john_total_payment_l662_66271


namespace proof_l662_66281

def statement : Prop :=
  ∀ (a : ℝ),
    (¬ (∃ x : ℝ, x ∈ Set.Icc (-1) 1 ∧ a^2 - 3 * a - x + 1 > 0) ∧
    ¬ (a^2 - 4 ≥ 0 ∧
    (∃ x : ℝ, x ∈ Set.Icc (-1) 1 ∧ a^2 - 3 * a - x + 1 > 0)))
    → (1 ≤ a ∧ a < 2)

theorem proof : statement :=
by
  sorry

end proof_l662_66281


namespace folder_cost_l662_66252

theorem folder_cost (cost_pens : ℕ) (cost_notebooks : ℕ) (total_spent : ℕ) (folders : ℕ) :
  cost_pens = 3 → cost_notebooks = 12 → total_spent = 25 → folders = 2 →
  ∃ (cost_per_folder : ℕ), cost_per_folder = 5 :=
by
  intros
  sorry

end folder_cost_l662_66252


namespace range_of_a_l662_66254

variable {x a : ℝ}

def p (x a : ℝ) : Prop := x > a
def q (x : ℝ) : Prop := x^2 + x - 2 > 0

theorem range_of_a 
  (h_sufficient : ∀ x, p x a → q x)
  (h_not_necessary : ∃ x, q x ∧ ¬ p x a) :
  a ≥ 1 :=
sorry

end range_of_a_l662_66254


namespace part1_part2_l662_66253

-- (1) Prove that if 2 ∈ M and M is the solution set of ax^2 + 5x - 2 > 0, then a > -2.
theorem part1 (a : ℝ) (h : 2 * (a * 4 + 10) - 2 > 0) : a > -2 :=
sorry

-- (2) Given M = {x | 1/2 < x < 2} and M is the solution set of ax^2 + 5x - 2 > 0,
-- prove that the solution set of ax^2 - 5x + a^2 - 1 > 0 is -3 < x < 1/2
theorem part2 (a : ℝ) (h1 : ∀ x : ℝ, (1/2 < x ∧ x < 2) ↔ ax^2 + 5*x - 2 > 0) (h2 : a = -2) :
  ∀ x : ℝ, (-3 < x ∧ x < 1/2) ↔ (-2 * x^2 - 5 * x + 3 > 0) :=
sorry

end part1_part2_l662_66253


namespace sheela_total_income_l662_66296

-- Define the monthly income as I
def monthly_income (I : Real) : Prop :=
  4500 = 0.28 * I

-- Define the annual income computed from monthly income
def annual_income (I : Real) : Real :=
  I * 12

-- Define the interest earned from savings account 
def interest_savings (principal : Real) (monthly_rate : Real) : Real :=
  principal * (monthly_rate * 12)

-- Define the interest earned from fixed deposit
def interest_fixed (principal : Real) (annual_rate : Real) : Real :=
  principal * annual_rate

-- Overall total income after one year calculation
def overall_total_income (annual_income : Real) (interest_savings : Real) (interest_fixed : Real) : Real :=
  annual_income + interest_savings + interest_fixed

-- Given conditions
variable (I : Real)
variable (principal_savings : Real := 4500)
variable (principal_fixed : Real := 3000)
variable (monthly_rate_savings : Real := 0.02)
variable (annual_rate_fixed : Real := 0.06)

-- Theorem statement to be proved
theorem sheela_total_income :
  monthly_income I →
  overall_total_income (annual_income I) 
                      (interest_savings principal_savings monthly_rate_savings)
                      (interest_fixed principal_fixed annual_rate_fixed)
  = 194117.16 :=
by
  sorry

end sheela_total_income_l662_66296


namespace f_six_equals_twenty_two_l662_66246

-- Definitions as per conditions
variable (n : ℕ) (f : ℕ → ℕ)

-- Conditions of the problem
-- n is a natural number greater than or equal to 3
-- f(n) satisfies the properties defined in the given solution
axiom f_base : f 1 = 2
axiom f_recursion {k : ℕ} (hk : k ≥ 1) : f (k + 1) = f k + (k + 1)

-- Goal to prove
theorem f_six_equals_twenty_two : f 6 = 22 := sorry

end f_six_equals_twenty_two_l662_66246


namespace two_digit_numbers_count_l662_66203

theorem two_digit_numbers_count : 
  ∃ (count : ℕ), (
    (∀ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ b = 2 * a → 
      (10 * b + a = 7 / 4 * (10 * a + b))) 
      ∧ count = 4
  ) :=
sorry

end two_digit_numbers_count_l662_66203


namespace solution_set_l662_66234

open Set Real

noncomputable def f : ℝ → ℝ := sorry

axiom f_odd : ∀ x : ℝ, f (-x) = - f x
axiom f_at_two : f 2 = 0
axiom f_cond : ∀ x : ℝ, 0 < x → x * (deriv (deriv f) x) + f x < 0

theorem solution_set :
  {x : ℝ | x * f x > 0} = Ioo (-2 : ℝ) 0 ∪ Ioo 0 2 :=
by
  sorry

end solution_set_l662_66234


namespace total_videos_watched_l662_66269

variable (Ekon Uma Kelsey : ℕ)

theorem total_videos_watched
  (hKelsey : Kelsey = 160)
  (hKelsey_Ekon : Kelsey = Ekon + 43)
  (hEkon_Uma : Ekon = Uma - 17) :
  Kelsey + Ekon + Uma = 411 := by
  sorry

end total_videos_watched_l662_66269


namespace find_W_l662_66260

noncomputable def volumeOutsideCylinder (r_cylinder r_sphere : ℝ) : ℝ :=
  let h := 2 * Real.sqrt (r_sphere^2 - r_cylinder^2)
  let V_sphere := (4 / 3) * Real.pi * r_sphere^3
  let V_cylinder := Real.pi * r_cylinder^2 * h
  V_sphere - V_cylinder

theorem find_W : 
  volumeOutsideCylinder 4 7 = (1372 / 3 - 32 * Real.sqrt 33) * Real.pi :=
by
  sorry

end find_W_l662_66260


namespace find_honeydews_left_l662_66256

theorem find_honeydews_left 
  (cantaloupe_price : ℕ)
  (honeydew_price : ℕ)
  (initial_cantaloupes : ℕ)
  (initial_honeydews : ℕ)
  (dropped_cantaloupes : ℕ)
  (rotten_honeydews : ℕ)
  (end_cantaloupes : ℕ)
  (total_revenue : ℕ)
  (honeydews_left : ℕ) :
  cantaloupe_price = 2 →
  honeydew_price = 3 →
  initial_cantaloupes = 30 →
  initial_honeydews = 27 →
  dropped_cantaloupes = 2 →
  rotten_honeydews = 3 →
  end_cantaloupes = 8 →
  total_revenue = 85 →
  honeydews_left = 9 :=
by
  sorry

end find_honeydews_left_l662_66256


namespace avg_cost_equals_0_22_l662_66236

-- Definitions based on conditions
def num_pencils : ℕ := 150
def cost_pencils : ℝ := 24.75
def shipping_cost : ℝ := 8.50

-- Calculating total cost and average cost
noncomputable def total_cost : ℝ := cost_pencils + shipping_cost
noncomputable def avg_cost_per_pencil : ℝ := total_cost / num_pencils

-- Lean theorem statement
theorem avg_cost_equals_0_22 : avg_cost_per_pencil = 0.22 :=
by
  sorry

end avg_cost_equals_0_22_l662_66236


namespace tan_double_angle_l662_66237

theorem tan_double_angle (theta : ℝ) (h : 2 * Real.sin theta + Real.cos theta = 0) :
  Real.tan (2 * theta) = - 4 / 3 :=
sorry

end tan_double_angle_l662_66237


namespace ratio_of_x_y_l662_66232

theorem ratio_of_x_y (x y : ℚ) (h : (3 * x - 2 * y) / (2 * x + 3 * y + 1) = 4 / 5) : x / y = 22 / 7 :=
sorry

end ratio_of_x_y_l662_66232


namespace quadratic_inequality_solution_l662_66279

theorem quadratic_inequality_solution (x : ℝ) : 3 * x^2 - 5 * x - 8 > 0 ↔ x < -4/3 ∨ x > 2 :=
by
  sorry

end quadratic_inequality_solution_l662_66279


namespace gcd_72_108_150_l662_66265

theorem gcd_72_108_150 : Nat.gcd (Nat.gcd 72 108) 150 = 6 := by
  sorry

end gcd_72_108_150_l662_66265


namespace chef_earns_2_60_less_l662_66292

/--
At Joe's Steakhouse, the hourly wage for a chef is 20% greater than that of a dishwasher,
and the hourly wage of a dishwasher is half as much as the hourly wage of a manager.
If a manager's wage is $6.50 per hour, prove that a chef earns $2.60 less per hour than a manager.
-/
theorem chef_earns_2_60_less {w_manager w_dishwasher w_chef : ℝ} 
  (h1 : w_dishwasher = w_manager / 2)
  (h2 : w_chef = w_dishwasher * 1.20)
  (h3 : w_manager = 6.50) :
  w_manager - w_chef = 2.60 :=
by
  sorry

end chef_earns_2_60_less_l662_66292


namespace parabola_focus_l662_66272

theorem parabola_focus (a : ℝ) (h : a = 4) : 
  ∃ f : ℝ × ℝ, f = (0, 1 / (4 * a)) ∧ y = 4 * x^2 → f = (0, 1 / 16) := 
by {
  sorry
}

end parabola_focus_l662_66272


namespace total_bowling_balls_l662_66216

theorem total_bowling_balls (red_balls : ℕ) (green_balls : ℕ) (h1 : red_balls = 30) (h2 : green_balls = red_balls + 6) : red_balls + green_balls = 66 :=
by
  sorry

end total_bowling_balls_l662_66216


namespace ellipse_equation_parabola_equation_l662_66207

noncomputable def ellipse_standard_equation (a b c : ℝ) : Prop :=
  a = 6 → b = 2 * Real.sqrt 5 → c = 4 → 
  ((∀ x y : ℝ, (y^2 / 36) + (x^2 / 20) = 1))

noncomputable def parabola_standard_equation (focus_x focus_y : ℝ) : Prop :=
  focus_x = 3 → focus_y = 0 → 
  (∀ x y : ℝ, y^2 = 12 * x)

theorem ellipse_equation : ellipse_standard_equation 6 (2 * Real.sqrt 5) 4 := by
  sorry

theorem parabola_equation : parabola_standard_equation 3 0 := by
  sorry

end ellipse_equation_parabola_equation_l662_66207


namespace triangle_area_is_18_l662_66255

noncomputable def area_triangle : ℝ :=
  let vertices : List (ℝ × ℝ) := [(1, 2), (7, 6), (1, 8)]
  let base := (8 - 2) -- Length between (1, 2) and (1, 8)
  let height := (7 - 1) -- Perpendicular distance from (7, 6) to x = 1
  (1 / 2) * base * height

theorem triangle_area_is_18 : area_triangle = 18 := by
  sorry

end triangle_area_is_18_l662_66255


namespace total_coin_tosses_l662_66250

variable (heads : ℕ) (tails : ℕ)

theorem total_coin_tosses (h_head : heads = 9) (h_tail : tails = 5) : heads + tails = 14 := by
  sorry

end total_coin_tosses_l662_66250


namespace math_problem_l662_66288

theorem math_problem (n a b : ℕ) (hn_pos : n > 0) (h1 : 3 * n + 1 = a^2) (h2 : 5 * n - 1 = b^2) :
  (∃ x y: ℕ, 7 * n + 13 = x * y ∧ 1 < x ∧ 1 < y) ∧
  (∃ p q: ℕ, 8 * (17 * n^2 + 3 * n) = p^2 + q^2) :=
  sorry

end math_problem_l662_66288


namespace percentage_of_first_solution_l662_66243

theorem percentage_of_first_solution (P : ℕ) 
  (h1 : 28 * P / 100 + 12 * 80 / 100 = 40 * 45 / 100) : 
  P = 30 :=
sorry

end percentage_of_first_solution_l662_66243


namespace find_k_b_l662_66295

-- Define the sets A and B
def A : Set (ℝ × ℝ) := { p | ∃ x y: ℝ, p = (x, y) }
def B : Set (ℝ × ℝ) := { p | ∃ x y: ℝ, p = (x, y) }

-- Define the mapping f
def f (p : ℝ × ℝ) (k b : ℝ) : ℝ × ℝ := (k * p.1, p.2 + b)

-- Define the conditions
def condition (f : (ℝ × ℝ) → ℝ × ℝ) :=
  f (3,1) = (6,2)

-- Statement: Prove that the values of k and b are 2 and 1 respectively
theorem find_k_b : ∃ (k b : ℝ), f (3, 1) k b = (6, 2) ∧ k = 2 ∧ b = 1 :=
by
  sorry

end find_k_b_l662_66295


namespace zoo_problem_l662_66257

theorem zoo_problem :
  let parrots := 8
  let snakes := 3 * parrots
  let monkeys := 2 * snakes
  let elephants := (parrots + snakes) / 2
  let zebras := monkeys - 35
  elephants - zebras = 3 :=
by
  sorry

end zoo_problem_l662_66257


namespace count_three_digit_concave_numbers_l662_66261

def is_concave_number (a b c : ℕ) : Prop :=
  a > b ∧ c > b

theorem count_three_digit_concave_numbers : 
  (∃! n : ℕ, n = 240) := by
  sorry

end count_three_digit_concave_numbers_l662_66261


namespace star_operation_example_l662_66276

-- Define the operation ☆
def star (a b : ℚ) : ℚ := a - b + 1

-- The theorem to prove
theorem star_operation_example : star (star 2 3) 2 = -1 := by
  sorry

end star_operation_example_l662_66276


namespace transformed_graph_passes_point_l662_66231

theorem transformed_graph_passes_point (f : ℝ → ℝ) 
  (h₁ : f 1 = 3) :
  f (-1) + 1 = 4 :=
by
  sorry

end transformed_graph_passes_point_l662_66231


namespace prob_2_lt_X_le_4_l662_66227

-- Define the PMF of the random variable X
noncomputable def pmf_X (k : ℕ) : ℝ :=
  if h : k ≥ 1 then 1 / (2 ^ k) else 0

-- Define the probability that X lies in the range (2, 4]
noncomputable def P_2_lt_X_le_4 : ℝ :=
  pmf_X 3 + pmf_X 4

-- Theorem stating the probability of x lying in (2, 4) is 3/16.
theorem prob_2_lt_X_le_4 : P_2_lt_X_le_4 = 3 / 16 := 
by
  -- Provide proof here
  sorry

end prob_2_lt_X_le_4_l662_66227


namespace percentage_of_students_wearing_blue_shirts_l662_66241

theorem percentage_of_students_wearing_blue_shirts :
  ∀ (total_students red_percent green_percent students_other_colors : ℕ),
  total_students = 800 →
  red_percent = 23 →
  green_percent = 15 →
  students_other_colors = 136 →
  ((total_students - students_other_colors) - (red_percent + green_percent) = 45) :=
by
  intros total_students red_percent green_percent students_other_colors h_total h_red h_green h_other
  have h_other_percent : (students_other_colors * 100 / total_students) = 17 :=
    sorry
  exact sorry

end percentage_of_students_wearing_blue_shirts_l662_66241


namespace scientific_notation_9600000_l662_66264

theorem scientific_notation_9600000 :
  ∃ (a : ℝ) (n : ℤ), (1 ≤ |a| ∧ |a| < 10) ∧ 9600000 = a * 10 ^ n ∧ a = 9.6 ∧ n = 6 :=
by
  exists 9.6
  exists 6
  simp
  sorry

end scientific_notation_9600000_l662_66264


namespace evaluate_expression_l662_66298

theorem evaluate_expression (x y : ℤ) (h₁ : x = 3) (h₂ : y = 4) : 
  (x^4 + 3 * x^2 - 2 * y + 2 * y^2) / 6 = 22 :=
by
  -- Conditions from the problem
  rw [h₁, h₂]
  -- Sorry is used to skip the proof
  sorry

end evaluate_expression_l662_66298


namespace rolls_in_package_l662_66285

theorem rolls_in_package (n : ℕ) :
  (9 : ℝ) = (n : ℝ) * (1 - 0.25) → n = 12 :=
by
  sorry

end rolls_in_package_l662_66285


namespace fg_of_3_is_94_l662_66259

def g (x : ℕ) : ℕ := 4 * x + 5
def f (x : ℕ) : ℕ := 6 * x - 8

theorem fg_of_3_is_94 : f (g 3) = 94 := by
  sorry

end fg_of_3_is_94_l662_66259


namespace binary_1011_is_11_decimal_124_is_174_l662_66291

-- Define the conversion from binary to decimal
def binaryToDecimal (n : Nat) : Nat :=
  (n % 10) * 2^0 + ((n / 10) % 10) * 2^1 + ((n / 100) % 10) * 2^2 + ((n / 1000) % 10) * 2^3

-- Define the conversion from decimal to octal through division and remainder
noncomputable def decimalToOctal (n : Nat) : String := 
  let rec aux (n : Nat) (acc : List Nat) : List Nat :=
    if n = 0 then acc else aux (n / 8) ((n % 8) :: acc)
  (aux n []).foldr (fun d s => s ++ d.repr) ""

-- Prove that the binary number 1011 (base 2) equals the decimal number 11
theorem binary_1011_is_11 : binaryToDecimal 1011 = 11 := by
  sorry

-- Prove that the decimal number 124 equals the octal number 174 (base 8)
theorem decimal_124_is_174 : decimalToOctal 124 = "174" := by
  sorry

end binary_1011_is_11_decimal_124_is_174_l662_66291


namespace miles_from_second_friend_to_work_l662_66230
variable (distance_to_first_friend := 8)
variable (distance_to_second_friend := distance_to_first_friend / 2)
variable (total_distance_to_second_friend := distance_to_first_friend + distance_to_second_friend)
variable (distance_to_work := 3 * total_distance_to_second_friend)

theorem miles_from_second_friend_to_work :
  distance_to_work = 36 := 
by
  sorry

end miles_from_second_friend_to_work_l662_66230


namespace cans_left_to_be_loaded_l662_66275

def cartons_total : ℕ := 50
def cartons_loaded : ℕ := 40
def cans_per_carton : ℕ := 20

theorem cans_left_to_be_loaded : (cartons_total - cartons_loaded) * cans_per_carton = 200 := by
  sorry

end cans_left_to_be_loaded_l662_66275


namespace largest_angle_in_triangle_l662_66247

theorem largest_angle_in_triangle 
  (A B C : ℝ)
  (h_sum_angles: 2 * A + 20 = 105)
  (h_triangle_sum: A + (A + 20) + C = 180)
  (h_A_ge_0: A ≥ 0)
  (h_B_ge_0: B ≥ 0)
  (h_C_ge_0: C ≥ 0) : 
  max A (max (A + 20) C) = 75 := 
by
  -- Placeholder proof
  sorry

end largest_angle_in_triangle_l662_66247


namespace find_analytical_expression_of_f_l662_66210

variable (f : ℝ → ℝ)

theorem find_analytical_expression_of_f
  (h : ∀ x : ℝ, f (2 * x + 1) = 4 * x^2 + 4 * x) :
  ∀ x : ℝ, f x = x^2 - 1 :=
sorry

end find_analytical_expression_of_f_l662_66210


namespace fraction_simplification_l662_66204

theorem fraction_simplification : (4 * 5) / 10 = 2 := 
by 
  sorry

end fraction_simplification_l662_66204


namespace man_twice_son_age_l662_66299

theorem man_twice_son_age (S M Y : ℕ) (h1 : S = 18) (h2 : M = S + 20) 
  (h3 : M + Y = 2 * (S + Y)) : Y = 2 :=
by
  -- Proof steps can be added here later
  sorry

end man_twice_son_age_l662_66299


namespace remainder_proof_l662_66273

theorem remainder_proof : 1234567 % 12 = 7 := sorry

end remainder_proof_l662_66273


namespace problem1_l662_66238

theorem problem1 (f : ℚ → ℚ) (a : Fin 7 → ℚ) (h₁ : ∀ x, f x = (1 - 3 * x) * (1 + x) ^ 5)
  (h₂ : ∀ x, f x = a 0 + a 1 * x + a 2 * x^2 + a 3 * x^3 + a 4 * x^4 + a 5 * x^5 + a 6 * x^6) :
  a 0 + (1/3) * a 1 + (1/3^2) * a 2 + (1/3^3) * a 3 + (1/3^4) * a 4 + (1/3^5) * a 5 + (1/3^6) * a 6 = 
  (1 - 3 * (1/3)) * (1 + (1/3))^5 :=
by sorry

end problem1_l662_66238


namespace domain_of_f_l662_66282

noncomputable def f (x : ℝ) : ℝ := (x + 3) / (x^2 - 5 * x + 6)

theorem domain_of_f :
  {x : ℝ | x^2 - 5 * x + 6 ≠ 0} = {x : ℝ | x < 2} ∪ {x : ℝ | 2 < x ∧ x < 3} ∪ {x : ℝ | x > 3} :=
by
  sorry

end domain_of_f_l662_66282


namespace car_trip_proof_l662_66297

def initial_oil_quantity (y : ℕ → ℕ) : Prop :=
  y 0 = 50

def consumption_rate (y : ℕ → ℕ) : Prop :=
  ∀ t, y t = y (t - 1) - 5

def relationship_between_y_and_t (y : ℕ → ℕ) : Prop :=
  ∀ t, y t = 50 - 5 * t

def oil_left_after_8_hours (y : ℕ → ℕ) : Prop :=
  y 8 = 10

theorem car_trip_proof (y : ℕ → ℕ) :
  initial_oil_quantity y ∧ consumption_rate y ∧ relationship_between_y_and_t y ∧ oil_left_after_8_hours y :=
by
  -- the proof goes here
  sorry

end car_trip_proof_l662_66297


namespace negation_example_l662_66284

variable {I : Set ℝ}

theorem negation_example (h : ∀ x ∈ I, x^3 - x^2 + 1 ≤ 0) : ¬(∀ x ∈ I, x^3 - x^2 + 1 ≤ 0) ↔ ∃ x ∈ I, x^3 - x^2 + 1 > 0 :=
by
  sorry

end negation_example_l662_66284


namespace possible_shapes_l662_66268

def is_valid_shapes (T S C : ℕ) : Prop :=
  T + S + C = 24 ∧ T = 7 * S

theorem possible_shapes :
  ∃ (T S C : ℕ), is_valid_shapes T S C ∧ 
    (T = 0 ∧ S = 0 ∧ C = 24) ∨
    (T = 7 ∧ S = 1 ∧ C = 16) ∨
    (T = 14 ∧ S = 2 ∧ C = 8) ∨
    (T = 21 ∧ S = 3 ∧ C = 0) :=
by
  sorry

end possible_shapes_l662_66268


namespace pool_drain_rate_l662_66211

-- Define the dimensions and other conditions
def poolLength : ℝ := 150
def poolWidth : ℝ := 40
def poolDepth : ℝ := 10
def poolCapacityPercent : ℝ := 0.80
def drainTime : ℕ := 800

-- Define the problem statement
theorem pool_drain_rate :
  let fullVolume := poolLength * poolWidth * poolDepth
  let volumeAt80Percent := fullVolume * poolCapacityPercent
  let drainRate := volumeAt80Percent / drainTime
  drainRate = 60 :=
by
  sorry

end pool_drain_rate_l662_66211


namespace max_food_per_guest_l662_66270

theorem max_food_per_guest (total_food : ℕ) (min_guests : ℕ)
    (H1 : total_food = 406) (H2 : min_guests = 163) :
    2 ≤ total_food / min_guests ∧ total_food / min_guests < 3 := by
  sorry

end max_food_per_guest_l662_66270


namespace determine_y_l662_66286

theorem determine_y (y : ℕ) : (8^5 + 8^5 + 2 * 8^5 = 2^y) → y = 17 := 
by {
  sorry
}

end determine_y_l662_66286


namespace remaining_payment_l662_66202

theorem remaining_payment (deposit_percent : ℝ) (deposit_amount : ℝ) (total_percent : ℝ) (total_price : ℝ) :
  deposit_percent = 5 ∧ deposit_amount = 50 ∧ total_percent = 100 → total_price - deposit_amount = 950 :=
by {
  sorry
}

end remaining_payment_l662_66202


namespace area_smallest_region_enclosed_l662_66217

theorem area_smallest_region_enclosed {x y : ℝ} (circle_eq : x^2 + y^2 = 9) (abs_line_eq : y = |x|) :
  ∃ area, area = (9 * Real.pi) / 4 :=
by
  sorry

end area_smallest_region_enclosed_l662_66217


namespace find_eq_thirteen_l662_66225

open Real

theorem find_eq_thirteen
  (a x b y c z : ℝ)
  (h1 : x / a + y / b + z / c = 5)
  (h2 : a / x + b / y + c / z = 6) :
  x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 13 := 
sorry

end find_eq_thirteen_l662_66225


namespace no_real_roots_of_f_l662_66287

def f (x : ℝ) : ℝ := (x + 1) * |x + 1| - x * |x| + 1

theorem no_real_roots_of_f :
  ∀ x : ℝ, f x ≠ 0 := by
  sorry

end no_real_roots_of_f_l662_66287


namespace graph_passes_through_fixed_point_l662_66213

theorem graph_passes_through_fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∃ y : ℝ, y = a^0 + 3 ∧ (0, y) = (0, 4)) :=
by
  use 4
  have h : a^0 = 1 := by simp
  rw [h]
  simp
  sorry

end graph_passes_through_fixed_point_l662_66213


namespace x_plus_y_equals_22_l662_66240

theorem x_plus_y_equals_22 (x y : ℕ) (h1 : 2^x = 4^(y + 2)) (h2 : 27^y = 9^(x - 7)) : x + y = 22 := 
sorry

end x_plus_y_equals_22_l662_66240


namespace cricket_bat_profit_percentage_l662_66214

-- Definitions for the problem conditions
def selling_price : ℝ := 850
def profit : ℝ := 255
def cost_price : ℝ := selling_price - profit
def expected_profit_percentage : ℝ := 42.86

-- The theorem to be proven
theorem cricket_bat_profit_percentage : 
  (profit / cost_price) * 100 = expected_profit_percentage :=
by 
  sorry

end cricket_bat_profit_percentage_l662_66214


namespace simplify_product_l662_66218

theorem simplify_product : 
  18 * (8 / 15) * (2 / 27) = 32 / 45 :=
by
  sorry

end simplify_product_l662_66218


namespace square_side_lengths_l662_66228

theorem square_side_lengths (x y : ℝ) (h1 : x + y = 20) (h2 : x^2 - y^2 = 120) :
  (x = 13 ∧ y = 7) ∨ (x = 7 ∧ y = 13) :=
by {
  -- skip proof
  sorry
}

end square_side_lengths_l662_66228


namespace divisible_by_6_and_sum_15_l662_66290

theorem divisible_by_6_and_sum_15 (A B : ℕ) (h1 : A + B = 15) (h2 : (10 * A + B) % 6 = 0) :
  (A * B = 56) ∨ (A * B = 54) :=
by sorry

end divisible_by_6_and_sum_15_l662_66290


namespace amount_of_bill_correct_l662_66220

noncomputable def TD : ℝ := 360
noncomputable def BD : ℝ := 421.7142857142857
noncomputable def computeFV (TD BD : ℝ) := (TD * BD) / (BD - TD)

theorem amount_of_bill_correct :
  computeFV TD BD = 2460 := 
sorry

end amount_of_bill_correct_l662_66220


namespace uncle_welly_roses_l662_66206

theorem uncle_welly_roses :
  let roses_two_days_ago := 50
  let roses_yesterday := roses_two_days_ago + 20
  let roses_today := 2 * roses_two_days_ago
  roses_two_days_ago + roses_yesterday + roses_today = 220 :=
by
  let roses_two_days_ago := 50
  let roses_yesterday := roses_two_days_ago + 20
  let roses_today := 2 * roses_two_days_ago
  show roses_two_days_ago + roses_yesterday + roses_today = 220
  sorry

end uncle_welly_roses_l662_66206


namespace polynomial_characterization_l662_66248

theorem polynomial_characterization (P : ℝ → ℝ) :
  (∀ a b c : ℝ, ab + bc + ca = 0 → P (a - b) + P (b - c) + P (c - a) = 2 * P (a + b + c)) →
  ∃ (α β : ℝ), ∀ x : ℝ, P x = α * x^4 + β * x^2 :=
by
  sorry

end polynomial_characterization_l662_66248

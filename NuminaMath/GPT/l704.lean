import Mathlib

namespace employee_n_salary_l704_70428

theorem employee_n_salary (m n : ℝ) (h1: m + n = 594) (h2: m = 1.2 * n) : n = 270 := by
  sorry

end employee_n_salary_l704_70428


namespace problem1_problem2_l704_70411

variable (α : ℝ) (tan_alpha_eq_three : Real.tan α = 3)

theorem problem1 : (4 * Real.sin α - Real.cos α) / (3 * Real.sin α + 5 * Real.cos α) = 11 / 14 :=
by sorry

theorem problem2 : Real.sin α * Real.cos α = 3 / 10 :=
by sorry

end problem1_problem2_l704_70411


namespace sum_of_series_l704_70430

theorem sum_of_series :
  (3 + 13 + 23 + 33 + 43) + (11 + 21 + 31 + 41 + 51) = 270 := 
by
  sorry

end sum_of_series_l704_70430


namespace quadratic_sum_of_roots_l704_70412

theorem quadratic_sum_of_roots (a b : ℝ)
  (h1: ∀ x: ℝ, x^2 + b * x - a < 0 ↔ 3 < x ∧ x < 4):
  a + b = -19 :=
sorry

end quadratic_sum_of_roots_l704_70412


namespace op_value_l704_70476

def op (x y : ℕ) : ℕ := x^3 - 3*x*y^2 + y^3

theorem op_value :
  op 2 1 = 3 := by sorry

end op_value_l704_70476


namespace initial_pen_count_is_30_l704_70447

def pen_count (initial_pens : ℕ) : ℕ :=
  let after_mike := initial_pens + 20
  let after_cindy := 2 * after_mike
  let after_sharon := after_cindy - 10
  after_sharon

theorem initial_pen_count_is_30 : pen_count 30 = 30 :=
by
  sorry

end initial_pen_count_is_30_l704_70447


namespace neg_p_l704_70478

noncomputable def f (a x : ℝ) : ℝ := a^x - x - a

theorem neg_p :
  ∃ (a : ℝ), a > 0 ∧ a ≠ 1 ∧ ∀ (x : ℝ), f a x ≠ 0 :=
sorry

end neg_p_l704_70478


namespace equal_powers_eq_a_b_l704_70461

theorem equal_powers_eq_a_b 
  (a b : ℝ) 
  (ha_pos : 0 < a) 
  (hb_pos : 0 < b)
  (h_exp_eq : a^b = b^a)
  (h_a_lt_1 : a < 1) : 
  a = b :=
sorry

end equal_powers_eq_a_b_l704_70461


namespace fraction_of_students_who_walk_l704_70410

def fraction_by_bus : ℚ := 2 / 5
def fraction_by_car : ℚ := 1 / 5
def fraction_by_scooter : ℚ := 1 / 8
def total_fraction_not_walk := fraction_by_bus + fraction_by_car + fraction_by_scooter

theorem fraction_of_students_who_walk :
  (1 - total_fraction_not_walk) = 11 / 40 :=
by
  sorry

end fraction_of_students_who_walk_l704_70410


namespace find_value_l704_70439

theorem find_value (x v : ℝ) (h1 : 0.80 * x + v = x) (h2 : x = 100) : v = 20 := by
    sorry

end find_value_l704_70439


namespace exist_three_integers_l704_70413

theorem exist_three_integers :
  ∃ (a b c : ℤ), a * b - c = 2018 ∧ b * c - a = 2018 ∧ c * a - b = 2018 := 
sorry

end exist_three_integers_l704_70413


namespace find_ordered_triple_l704_70456

theorem find_ordered_triple
  (a b c : ℝ)
  (h1 : a > 2)
  (h2 : b > 2)
  (h3 : c > 2)
  (h4 : (a + 3) ^ 2 / (b + c - 3) + (b + 5) ^ 2 / (c + a - 5) + (c + 7) ^ 2 / (a + b - 7) = 48) :
  (a, b, c) = (7, 5, 3) :=
by {
  sorry
}

end find_ordered_triple_l704_70456


namespace sin_300_eq_neg_sqrt_3_div_2_l704_70436

noncomputable def sin_300_degrees : ℝ := Real.sin (300 * Real.pi / 180)

theorem sin_300_eq_neg_sqrt_3_div_2 : sin_300_degrees = -Real.sqrt 3 / 2 :=
by
  sorry

end sin_300_eq_neg_sqrt_3_div_2_l704_70436


namespace puppy_cost_l704_70453

variable (P : ℕ)

theorem puppy_cost (hc : 2 * 50 = 100) (hd : 3 * 100 = 300) (htotal : 2 * 50 + 3 * 100 + 2 * P = 700) : P = 150 :=
by
  sorry

end puppy_cost_l704_70453


namespace shirt_original_price_l704_70497

theorem shirt_original_price (P : ℝ) : 
  (18 = P * 0.75 * 0.75 * 0.90 * 1.15) → 
  P = 18 / (0.75 * 0.75 * 0.90 * 1.15) :=
by
  intro h
  sorry

end shirt_original_price_l704_70497


namespace series_ln2_series_1_ln2_l704_70470

theorem series_ln2 :
  ∑' n : ℕ, (1 / (n + 1) / (n + 2)) = Real.log 2 :=
sorry

theorem series_1_ln2 :
  ∑' k : ℕ, (1 / ((2 * k + 2) * (2 * k + 3))) = 1 - Real.log 2 :=
sorry

end series_ln2_series_1_ln2_l704_70470


namespace bucket_weight_full_l704_70498

variable (p q x y : ℝ)

theorem bucket_weight_full (h1 : x + (3 / 4) * y = p)
                           (h2 : x + (1 / 3) * y = q) :
  x + y = (1 / 5) * (8 * p - 3 * q) :=
by
  sorry

end bucket_weight_full_l704_70498


namespace inequality_order_l704_70405

theorem inequality_order (a b c : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : c ≠ 0)
  (h : (a^2 / (b^2 + c^2)) < (b^2 / (c^2 + a^2)) ∧ (b^2 / (c^2 + a^2)) < (c^2 / (a^2 + b^2))) :
  |a| < |b| ∧ |b| < |c| := 
sorry

end inequality_order_l704_70405


namespace company_total_payment_correct_l704_70462

def totalEmployees : Nat := 450
def firstGroup : Nat := 150
def secondGroup : Nat := 200
def thirdGroup : Nat := 100

def firstBaseSalary : Nat := 2000
def secondBaseSalary : Nat := 2500
def thirdBaseSalary : Nat := 3000

def firstInitialBonus : Nat := 500
def secondInitialBenefit : Nat := 400
def thirdInitialBenefit : Nat := 600

def firstLayoffRound1 : Nat := (20 * firstGroup) / 100
def secondLayoffRound1 : Nat := (25 * secondGroup) / 100
def thirdLayoffRound1 : Nat := (15 * thirdGroup) / 100

def remainingFirstGroupRound1 : Nat := firstGroup - firstLayoffRound1
def remainingSecondGroupRound1 : Nat := secondGroup - secondLayoffRound1
def remainingThirdGroupRound1 : Nat := thirdGroup - thirdLayoffRound1

def firstAdjustedBonusRound1 : Nat := 400
def secondAdjustedBenefitRound1 : Nat := 300

def firstLayoffRound2 : Nat := (10 * remainingFirstGroupRound1) / 100
def secondLayoffRound2 : Nat := (15 * remainingSecondGroupRound1) / 100
def thirdLayoffRound2 : Nat := (5 * remainingThirdGroupRound1) / 100

def remainingFirstGroupRound2 : Nat := remainingFirstGroupRound1 - firstLayoffRound2
def remainingSecondGroupRound2 : Nat := remainingSecondGroupRound1 - secondLayoffRound2
def remainingThirdGroupRound2 : Nat := remainingThirdGroupRound1 - thirdLayoffRound2

def thirdAdjustedBenefitRound2 : Nat := (80 * thirdInitialBenefit) / 100

def totalBaseSalary : Nat :=
  (remainingFirstGroupRound2 * firstBaseSalary)
  + (remainingSecondGroupRound2 * secondBaseSalary)
  + (remainingThirdGroupRound2 * thirdBaseSalary)

def totalBonusesAndBenefits : Nat :=
  (remainingFirstGroupRound2 * firstAdjustedBonusRound1)
  + (remainingSecondGroupRound2 * secondAdjustedBenefitRound1)
  + (remainingThirdGroupRound2 * thirdAdjustedBenefitRound2)

def totalPayment : Nat :=
  totalBaseSalary + totalBonusesAndBenefits

theorem company_total_payment_correct :
  totalPayment = 893200 :=
by
  -- proof steps
  sorry

end company_total_payment_correct_l704_70462


namespace kelly_apples_total_l704_70402

def initial_apples : ℕ := 56
def second_day_pick : ℕ := 105
def third_day_pick : ℕ := 84
def apples_eaten : ℕ := 23

theorem kelly_apples_total :
  initial_apples + second_day_pick + third_day_pick - apples_eaten = 222 := by
  sorry

end kelly_apples_total_l704_70402


namespace probability_reach_edge_within_five_hops_l704_70490

-- Define the probability of reaching an edge within n hops from the center
noncomputable def probability_reach_edge_by_hops (n : ℕ) : ℚ :=
if n = 5 then 121 / 128 else 0 -- This is just a placeholder for the real recursive computation.

-- Main theorem to prove
theorem probability_reach_edge_within_five_hops :
  probability_reach_edge_by_hops 5 = 121 / 128 :=
by
  -- Skipping the actual proof here
  sorry

end probability_reach_edge_within_five_hops_l704_70490


namespace problem_statement_l704_70473

theorem problem_statement (x y z : ℝ) (h₀ : 0 ≤ x) (h₁ : 0 ≤ y) (h₂ : 0 ≤ z) (h₃ : x + y + z = 1) :
  0 ≤ x * y + y * z + z * x - 2 * x * y * z ∧ x * y + y * z + z * x - 2 * x * y * z ≤ 7 / 27 := 
sorry

end problem_statement_l704_70473


namespace gcd_m_n_l704_70495

def m : ℕ := 333333
def n : ℕ := 7777777

theorem gcd_m_n : Nat.gcd m n = 1 :=
by
  -- Mathematical steps have been omitted as they are not needed
  sorry

end gcd_m_n_l704_70495


namespace inequality_abc_l704_70493

open Real

theorem inequality_abc 
  (a b c : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (h : a * b * c = 1) : 
  (1 / (a ^ 3 * (b + c)) + 1 / (b ^ 3 * (c + a)) + 1 / (c ^ 3 * (a + b))) ≥ 3 / 2 :=
sorry

end inequality_abc_l704_70493


namespace quadratic_no_real_roots_l704_70404

theorem quadratic_no_real_roots (k : ℝ) : 
  (∀ x : ℝ, ¬(x^2 - 2*x - k = 0)) ↔ k < -1 :=
by sorry

end quadratic_no_real_roots_l704_70404


namespace unique_solution_qx2_minus_16x_plus_8_eq_0_l704_70426

theorem unique_solution_qx2_minus_16x_plus_8_eq_0 (q : ℝ) (hq : q ≠ 0) :
  (∀ x : ℝ, q * x^2 - 16 * x + 8 = 0 → (256 - 32 * q = 0)) → q = 8 :=
by
  sorry

end unique_solution_qx2_minus_16x_plus_8_eq_0_l704_70426


namespace profits_to_revenues_ratio_l704_70441

theorem profits_to_revenues_ratio (R P: ℝ) 
    (rev_2009: R_2009 = 0.8 * R) 
    (profit_2009_rev_2009: P_2009 = 0.2 * R_2009)
    (profit_2009: P_2009 = 1.6 * P):
    (P / R) * 100 = 10 :=
by
  sorry

end profits_to_revenues_ratio_l704_70441


namespace alex_serge_equiv_distinct_values_l704_70465

-- Defining the context and data structures
variable {n : ℕ} -- Number of boxes
variable {c : ℕ → ℕ} -- Function representing number of cookies in each box, indexed by box number
variable {m : ℕ} -- Number of plates
variable {p : ℕ → ℕ} -- Function representing number of cookies on each plate, indexed by plate number

-- Define the sets representing the unique counts recorded by Alex and Serge
def Alex_record (c : ℕ → ℕ) (n : ℕ) : Set ℕ := 
  { x | ∃ i, i < n ∧ c i = x }

def Serge_record (p : ℕ → ℕ) (m : ℕ) : Set ℕ := 
  { y | ∃ j, j < m ∧ p j = y }

-- The proof goal: Alex's record contains the same number of distinct values as Serge's record
theorem alex_serge_equiv_distinct_values
  (c : ℕ → ℕ) (n : ℕ) (p : ℕ → ℕ) (m : ℕ) :
  Alex_record c n = Serge_record p m :=
sorry

end alex_serge_equiv_distinct_values_l704_70465


namespace value_of_x_l704_70466

theorem value_of_x (x y : ℤ) (h1 : x - y = 8) (h2 : x + y = 14) : x = 11 :=
by
  sorry

end value_of_x_l704_70466


namespace fill_digits_subtraction_correct_l704_70487

theorem fill_digits_subtraction_correct :
  ∀ (A B : ℕ), A236 - (B*100 + 97) = 5439 → A = 6 ∧ B = 7 :=
by
  sorry

end fill_digits_subtraction_correct_l704_70487


namespace rational_numbers_property_l704_70469

theorem rational_numbers_property (n : ℕ) (h : n > 0) :
  ∃ (a b : ℚ), a ≠ b ∧ (∀ k, 1 ≤ k ∧ k ≤ n → ∃ m : ℤ, a^k - b^k = m) ∧ 
  ∀ i, (a : ℝ) ≠ i ∧ (b : ℝ) ≠ i :=
sorry

end rational_numbers_property_l704_70469


namespace teacher_discount_l704_70463

-- Definitions that capture the conditions in Lean
def num_students : ℕ := 30
def num_pens_per_student : ℕ := 5
def num_notebooks_per_student : ℕ := 3
def num_binders_per_student : ℕ := 1
def num_highlighters_per_student : ℕ := 2
def cost_per_pen : ℚ := 0.50
def cost_per_notebook : ℚ := 1.25
def cost_per_binder : ℚ := 4.25
def cost_per_highlighter : ℚ := 0.75
def amount_spent : ℚ := 260

-- Compute the total cost without discount
def total_cost : ℚ :=
  (num_students * num_pens_per_student) * cost_per_pen +
  (num_students * num_notebooks_per_student) * cost_per_notebook +
  (num_students * num_binders_per_student) * cost_per_binder +
  (num_students * num_highlighters_per_student) * cost_per_highlighter

-- The main theorem to prove the applied teacher discount
theorem teacher_discount :
  total_cost - amount_spent = 100 := by
  sorry

end teacher_discount_l704_70463


namespace sum_in_base7_l704_70401

-- An encoder function for base 7 integers
def to_base7 (n : ℕ) : string :=
sorry -- skipping the implementation for brevity

-- Decoding the string representation back to a natural number
def from_base7 (s : string) : ℕ :=
sorry -- skipping the implementation for brevity

-- The provided numbers in base 7
def x : ℕ := from_base7 "666"
def y : ℕ := from_base7 "66"
def z : ℕ := from_base7 "6"

-- The expected sum in base 7
def expected_sum : ℕ := from_base7 "104"

-- The statement to be proved
theorem sum_in_base7 : x + y + z = expected_sum :=
sorry -- The proof is omitted

end sum_in_base7_l704_70401


namespace necessary_but_not_sufficient_l704_70414

theorem necessary_but_not_sufficient (a : ℝ) (h : a > 0) : a > 0 ↔ ((a > 0) ∧ (a < 2) → (a^2 - 2 * a < 0)) :=
by
    sorry

end necessary_but_not_sufficient_l704_70414


namespace first_class_students_count_l704_70488

theorem first_class_students_count 
  (x : ℕ) 
  (avg1 : ℕ) (avg2 : ℕ) (num2 : ℕ) (overall_avg : ℝ)
  (h_avg1 : avg1 = 40)
  (h_avg2 : avg2 = 60)
  (h_num2 : num2 = 50)
  (h_overall_avg : overall_avg = 52.5)
  (h_eq : 40 * x + 60 * 50 = (52.5:ℝ) * (x + 50)) :
  x = 30 :=
by
  sorry

end first_class_students_count_l704_70488


namespace ap_sub_aq_l704_70486

variable {n : ℕ} (hn : n > 0)

def S (n : ℕ) : ℕ := 2 * n^2 - 3 * n

def a (n : ℕ) (hn : n > 0) : ℕ :=
S n - S (n - 1)

theorem ap_sub_aq (p q : ℕ) (hp : p > 0) (hq : q > 0) (h : p - q = 5) :
  a p hp - a q hq = 20 :=
sorry

end ap_sub_aq_l704_70486


namespace travel_time_l704_70420

theorem travel_time (v : ℝ) (d : ℝ) (t : ℝ) (hv : v = 65) (hd : d = 195) : t = 3 :=
by
  sorry

end travel_time_l704_70420


namespace common_solution_exists_l704_70454

theorem common_solution_exists (a b : ℝ) :
  (∃ x y : ℝ, 19 * x^2 + 19 * y^2 + a * x + b * y + 98 = 0 ∧
                         98 * x^2 + 98 * y^2 + a * x + b * y + 19 = 0)
  → a^2 + b^2 ≥ 13689 :=
by
  -- Proof omitted
  sorry

end common_solution_exists_l704_70454


namespace find_monthly_fee_l704_70475

variable (monthly_fee : ℝ) (cost_per_minute : ℝ := 0.12) (minutes_used : ℕ := 178) (total_bill : ℝ := 23.36)

theorem find_monthly_fee
  (h1 : total_bill = monthly_fee + (cost_per_minute * minutes_used)) :
  monthly_fee = 2 :=
by
  sorry

end find_monthly_fee_l704_70475


namespace quadratic_inequality_condition_l704_70435

theorem quadratic_inequality_condition (a b c : ℝ) (h : a < 0) (disc : b^2 - 4 * a * c ≤ 0) : 
  ∀ x : ℝ, a * x^2 + b * x + c ≤ 0 :=
sorry

end quadratic_inequality_condition_l704_70435


namespace train_speed_l704_70477

-- Definitions of the given conditions
def platform_length : ℝ := 250
def train_length : ℝ := 470.06
def time_taken : ℝ := 36

-- Definition of the total distance covered
def total_distance := platform_length + train_length

-- The proof problem: Prove that the calculated speed is approximately 20.0017 m/s
theorem train_speed :
  (total_distance / time_taken) = 20.0017 :=
by
  -- The actual proof goes here, but for now we leave it as sorry
  sorry

end train_speed_l704_70477


namespace find_n_l704_70400

theorem find_n : ∃ n : ℕ, (∃ A B : ℕ, A ≠ B ∧ 10^(n-1) ≤ A ∧ A < 10^n ∧ 10^(n-1) ≤ B ∧ B < 10^n ∧ (10^n * A + B) % (10^n * B + A) = 0) ↔ n % 6 = 3 :=
by
  sorry

end find_n_l704_70400


namespace central_angle_agree_l704_70452

theorem central_angle_agree (ratio_agree : ℕ) (ratio_disagree : ℕ) (ratio_no_preference : ℕ) (total_angle : ℝ) :
  ratio_agree = 7 → ratio_disagree = 2 → ratio_no_preference = 1 → total_angle = 360 →
  (ratio_agree / (ratio_agree + ratio_disagree + ratio_no_preference) * total_angle = 252) :=
by
  -- conditions and assumptions
  intros h_agree h_disagree h_no_preference h_total_angle
  -- simplified steps here
  sorry

end central_angle_agree_l704_70452


namespace value_of_polynomial_l704_70422

theorem value_of_polynomial : 
  99^5 - 5 * 99^4 + 10 * 99^3 - 10 * 99^2 + 5 * 99 - 1 = 98^5 := by
  sorry

end value_of_polynomial_l704_70422


namespace steps_left_to_climb_l704_70448

-- Define the conditions
def total_stairs : ℕ := 96
def climbed_stairs : ℕ := 74

-- The problem: Prove that the number of stairs left to climb is 22
theorem steps_left_to_climb : (total_stairs - climbed_stairs) = 22 :=
by 
  sorry

end steps_left_to_climb_l704_70448


namespace area_of_rhombus_l704_70446

theorem area_of_rhombus (d1 d2 : ℕ) (h1 : d1 = 16) (h2 : d2 = 20) : (d1 * d2) / 2 = 160 := by
  sorry

end area_of_rhombus_l704_70446


namespace find_binomial_params_l704_70482

noncomputable def binomial_params (n p : ℝ) := 2.4 = n * p ∧ 1.44 = n * p * (1 - p)

theorem find_binomial_params (n p : ℝ) (h : binomial_params n p) : n = 6 ∧ p = 0.4 :=
by
  sorry

end find_binomial_params_l704_70482


namespace inequality_for_natural_n_l704_70484

theorem inequality_for_natural_n (n : ℕ) : (2 * n + 1)^n ≥ (2 * n)^n + (2 * n - 1)^n :=
by sorry

end inequality_for_natural_n_l704_70484


namespace cube_prism_surface_area_l704_70457

theorem cube_prism_surface_area (a : ℝ) (h : a > 0) :
  2 * (6 * a^2) > 4 * a^2 + 2 * (2 * a * a) :=
by sorry

end cube_prism_surface_area_l704_70457


namespace fraction_draw_l704_70485

/-
Theorem: Given the win probabilities for Amy, Lily, and Eve, the fraction of the time they end up in a draw is 3/10.
-/

theorem fraction_draw (P_Amy P_Lily P_Eve : ℚ) (h_Amy : P_Amy = 2/5) (h_Lily : P_Lily = 1/5) (h_Eve : P_Eve = 1/10) : 
  1 - (P_Amy + P_Lily + P_Eve) = 3 / 10 := by
  sorry

end fraction_draw_l704_70485


namespace count_measures_of_angle_A_l704_70467

theorem count_measures_of_angle_A :
  ∃ n : ℕ, n = 17 ∧
  ∃ (A B : ℕ), A > 0 ∧ B > 0 ∧ A + B = 180 ∧ (∃ k : ℕ, k ≥ 1 ∧ A = k * B) ∧ (∀ (A' B' : ℕ), A' > 0 ∧ B' > 0 ∧ A' + B' = 180 ∧ (∀ k : ℕ, k ≥ 1 ∧ A' = k * B') → n = 17) :=
sorry

end count_measures_of_angle_A_l704_70467


namespace cylinder_in_cone_l704_70403

noncomputable def cylinder_radius : ℝ :=
  let cone_radius : ℝ := 4
  let cone_height : ℝ := 10
  let r : ℝ := (10 * 2) / 9  -- based on the derived form of r calculation
  r

theorem cylinder_in_cone :
  let cone_radius : ℝ := 4
  let cone_height : ℝ := 10
  let r : ℝ := cylinder_radius
  (r = 20 / 9) :=
by
  sorry -- Proof mechanism is skipped as per instructions.

end cylinder_in_cone_l704_70403


namespace range_of_a_for_inequality_l704_70443

theorem range_of_a_for_inequality (a : ℝ) : 
  (∀ x : ℝ, x^2 - 2 * x + a > 0) → a > 1 :=
by
  sorry

end range_of_a_for_inequality_l704_70443


namespace Michaela_needs_20_oranges_l704_70481

variable (M : ℕ)
variable (C : ℕ)

theorem Michaela_needs_20_oranges 
  (h1 : C = 2 * M)
  (h2 : M + C = 60):
  M = 20 :=
by 
  sorry

end Michaela_needs_20_oranges_l704_70481


namespace work_completion_days_l704_70494

variables (M D X : ℕ) (W : ℝ)

-- Original conditions
def original_men : ℕ := 15
def planned_days : ℕ := 40
def men_absent : ℕ := 5

-- Theorem to prove
theorem work_completion_days :
  M = original_men →
  D = planned_days →
  W > 0 →
  (M - men_absent) * X * W = M * D * W →
  X = 60 :=
by
  intros hM hD hW h_work
  sorry

end work_completion_days_l704_70494


namespace find_alpha_l704_70491

theorem find_alpha (P : Real × Real) (h: P = (Real.sin (50 * Real.pi / 180), 1 + Real.cos (50 * Real.pi / 180))) :
  ∃ α : Real, α = 65 * Real.pi / 180 := by
  sorry

end find_alpha_l704_70491


namespace units_digit_of_7_pow_6_pow_5_l704_70479

theorem units_digit_of_7_pow_6_pow_5 : (7^(6^5)) % 10 = 1 := by
  -- Proof goes here
  sorry

end units_digit_of_7_pow_6_pow_5_l704_70479


namespace greta_hourly_wage_is_12_l704_70415

-- Define constants
def greta_hours : ℕ := 40
def lisa_hourly_wage : ℕ := 15
def lisa_hours : ℕ := 32

-- Define the total earnings of Greta and Lisa
def greta_earnings (G : ℕ) : ℕ := greta_hours * G
def lisa_earnings : ℕ := lisa_hours * lisa_hourly_wage

-- Main theorem statement
theorem greta_hourly_wage_is_12 (G : ℕ) (h : greta_earnings G = lisa_earnings) : G = 12 :=
by
  sorry

end greta_hourly_wage_is_12_l704_70415


namespace correct_forecast_interpretation_l704_70425

/-- The probability of precipitation in the area tomorrow is 80%. -/
def prob_precipitation_tomorrow : ℝ := 0.8

/-- Multiple choice options regarding the interpretation of the probability of precipitation. -/
inductive forecast_interpretation
| A : forecast_interpretation
| B : forecast_interpretation
| C : forecast_interpretation
| D : forecast_interpretation

/-- The correct interpretation is Option C: "There is an 80% chance of rain in the area tomorrow." -/
def correct_interpretation : forecast_interpretation :=
forecast_interpretation.C

theorem correct_forecast_interpretation :
  (prob_precipitation_tomorrow = 0.8) → (correct_interpretation = forecast_interpretation.C) :=
by
  sorry

end correct_forecast_interpretation_l704_70425


namespace inequality_correct_l704_70432

open BigOperators

theorem inequality_correct {a b : ℝ} (ha : 0 < a) (hb : 0 < b) :
  (a^2 + b^2) / 2 ≥ (a + b)^2 / 4 ∧ (a + b)^2 / 4 ≥ a * b :=
by 
  sorry

end inequality_correct_l704_70432


namespace parabola_from_hyperbola_l704_70450

noncomputable def hyperbola_equation (x y : ℝ) : Prop := 16 * x^2 - 9 * y^2 = 144

noncomputable def parabola_equation_1 (x y : ℝ) : Prop := y^2 = -24 * x

noncomputable def parabola_equation_2 (x y : ℝ) : Prop := y^2 = 24 * x

theorem parabola_from_hyperbola :
  (∃ x y : ℝ, hyperbola_equation x y) →
  (∃ x y : ℝ, parabola_equation_1 x y ∨ parabola_equation_2 x y) :=
by
  intro h
  -- proof is omitted
  sorry

end parabola_from_hyperbola_l704_70450


namespace painted_pictures_in_june_l704_70427

theorem painted_pictures_in_june (J : ℕ) (h1 : J + (J + 2) + 9 = 13) : J = 1 :=
by
  -- Given condition translates to J + J + 2 + 9 = 13
  -- Simplification yields 2J + 11 = 13
  -- Solving 2J + 11 = 13 gives J = 1
  sorry

end painted_pictures_in_june_l704_70427


namespace determine_min_guesses_l704_70455

def minimum_guesses (n k : ℕ) (h : n > k) : ℕ :=
  if n = 2 * k then 2 else 1

theorem determine_min_guesses (n k : ℕ) (h : n > k) :
  (if n = 2 * k then 2 else 1) = minimum_guesses n k h := by
  sorry

end determine_min_guesses_l704_70455


namespace percentage_profit_first_bicycle_l704_70429

theorem percentage_profit_first_bicycle :
  ∃ (C1 C2 : ℝ), 
    (C1 + C2 = 1980) ∧ 
    (0.9 * C2 = 990) ∧ 
    (12.5 / 100 * C1 = (990 - C1) / C1 * 100) :=
by
  sorry

end percentage_profit_first_bicycle_l704_70429


namespace seq_eighth_term_l704_70468

-- Define the sequence recursively
def seq : ℕ → ℕ
| 0     => 1  -- Base case, since 1 is the first term of the sequence
| (n+1) => seq n + (n + 1)  -- Recursive case, each term is the previous term plus the index number (which is n + 1) minus 1

-- Define the statement to prove 
theorem seq_eighth_term : seq 7 = 29 :=  -- Note: index 7 corresponds to the 8th term since indexing is 0-based
  by
  sorry

end seq_eighth_term_l704_70468


namespace time_to_walk_l704_70459

variable (v l r w : ℝ)
variable (h1 : l = 15 * (v + r))
variable (h2 : l = 30 * (v + w))
variable (h3 : l = 20 * r)

theorem time_to_walk (h1 : l = 15 * (v + r)) (h2 : l = 30 * (v + w)) (h3 : l = 20 * r) : l / w = 60 := 
by sorry

end time_to_walk_l704_70459


namespace inequality_holds_l704_70492

theorem inequality_holds (x y z : ℝ) : x^2 + y^2 + z^2 ≥ Real.sqrt 2 * (x * y + y * z) := 
by 
  sorry

end inequality_holds_l704_70492


namespace cos_double_angle_trig_identity_l704_70496

theorem cos_double_angle_trig_identity
  (α : ℝ) 
  (h : Real.sin (α - Real.pi / 3) = 4 / 5) : 
  Real.cos (2 * α + Real.pi / 3) = 7 / 25 :=
by
  sorry

end cos_double_angle_trig_identity_l704_70496


namespace find_C_l704_70409

variable (A B C : ℕ)

theorem find_C (h1 : A + B + C = 500) (h2 : A + C = 200) (h3 : B + C = 360) : C = 60 := by
  sorry

end find_C_l704_70409


namespace circle_center_l704_70424

theorem circle_center :
  ∃ c : ℝ × ℝ, c = (-1, 3) ∧ ∀ (x y : ℝ), (4 * x^2 + 8 * x + 4 * y^2 - 24 * y + 96 = 0 ↔ (x + 1)^2 + (y - 3)^2 = 14) :=
by
  sorry

end circle_center_l704_70424


namespace iron_weight_l704_70460

theorem iron_weight 
  (A : ℝ) (hA : A = 0.83) 
  (I : ℝ) (hI : I = A + 10.33) : 
  I = 11.16 := 
by 
  sorry

end iron_weight_l704_70460


namespace grid_covering_impossible_l704_70480

theorem grid_covering_impossible :
  ∀ (x y : ℕ), x + y = 19 → 6 * x + 7 * y = 132 → False :=
by
  intros x y h1 h2
  -- Proof would go here.
  sorry

end grid_covering_impossible_l704_70480


namespace calculation_correct_l704_70499

theorem calculation_correct : 1984 + 180 / 60 - 284 = 1703 := 
by 
  sorry

end calculation_correct_l704_70499


namespace only_n_divides_2_pow_n_minus_1_l704_70419

theorem only_n_divides_2_pow_n_minus_1 : ∀ (n : ℕ), n > 0 ∧ n ∣ (2^n - 1) ↔ n = 1 := by
  sorry

end only_n_divides_2_pow_n_minus_1_l704_70419


namespace least_subtraction_divisibility_l704_70431

theorem least_subtraction_divisibility :
  ∃ k : ℕ, 427398 - k = 14 * n ∧ k = 6 :=
by
  use 6
  sorry

end least_subtraction_divisibility_l704_70431


namespace cabbages_difference_l704_70489

noncomputable def numCabbagesThisYear : ℕ := 4096
noncomputable def numCabbagesLastYear : ℕ := 3969
noncomputable def diffCabbages : ℕ := numCabbagesThisYear - numCabbagesLastYear

theorem cabbages_difference :
  diffCabbages = 127 := by
  sorry

end cabbages_difference_l704_70489


namespace solve_x_l704_70423

theorem solve_x (x : ℝ) (h : x ≠ 0) (h_eq : (5 * x) ^ 10 = (10 * x) ^ 5) : x = 2 / 5 :=
by
  sorry

end solve_x_l704_70423


namespace selected_number_in_14th_group_is_272_l704_70438

-- Definitions based on conditions
def total_students : ℕ := 400
def sample_size : ℕ := 20
def first_selected_number : ℕ := 12
def sampling_interval : ℕ := total_students / sample_size
def target_group : ℕ := 14

-- Correct answer definition
def selected_number_in_14th_group : ℕ := first_selected_number + (target_group - 1) * sampling_interval

-- Theorem stating the correct answer is 272
theorem selected_number_in_14th_group_is_272 :
  selected_number_in_14th_group = 272 :=
sorry

end selected_number_in_14th_group_is_272_l704_70438


namespace determine_c_absolute_value_l704_70472

theorem determine_c_absolute_value
  (a b c : ℤ)
  (h_gcd : Int.gcd (Int.gcd a b) c = 1)
  (h_root : a * (Complex.mk 3 1)^4 + b * (Complex.mk 3 1)^3 + c * (Complex.mk 3 1)^2 + b * (Complex.mk 3 1) + a = 0) :
  |c| = 109 := 
sorry

end determine_c_absolute_value_l704_70472


namespace value_of_m_l704_70440

theorem value_of_m (m : ℝ) (h1 : m - 2 ≠ 0) (h2 : |m| - 1 = 1) : m = -2 := by {
  sorry
}

end value_of_m_l704_70440


namespace semi_circle_radius_l704_70421

theorem semi_circle_radius (P : ℝ) (r : ℝ) (π : ℝ) (h_perimeter : P = 113) (h_pi : π = Real.pi) :
  r = P / (π + 2) :=
sorry

end semi_circle_radius_l704_70421


namespace largest_lambda_inequality_l704_70451

theorem largest_lambda_inequality :
  ∀ (a b c d e : ℝ), 0 ≤ a → 0 ≤ b → 0 ≤ c → 0 ≤ d → 0 ≤ e →
  (a^2 + b^2 + c^2 + d^2 + e^2 ≥ a * b + (5/4) * b * c + c * d + d * e) :=
by
  sorry

end largest_lambda_inequality_l704_70451


namespace fernanda_savings_before_payments_l704_70406

open Real

theorem fernanda_savings_before_payments (aryan_debt kyro_debt aryan_payment kyro_payment total_savings before_savings : ℝ) 
  (h1: aryan_debt = 1200)
  (h2: aryan_debt = 2 * kyro_debt)
  (h3: aryan_payment = 0.6 * aryan_debt)
  (h4: kyro_payment = 0.8 * kyro_debt)
  (h5: total_savings = before_savings + aryan_payment + kyro_payment)
  (h6: total_savings = 1500) :
  before_savings = 300 :=
by
  sorry

end fernanda_savings_before_payments_l704_70406


namespace expand_and_simplify_l704_70445

theorem expand_and_simplify (x : ℝ) : (x^2 + 4) * (x - 5) = x^3 - 5 * x^2 + 4 * x - 20 := 
sorry

end expand_and_simplify_l704_70445


namespace borrowed_dimes_calculation_l704_70458

-- Define Sam's initial dimes and remaining dimes after borrowing
def original_dimes : ℕ := 8
def remaining_dimes : ℕ := 4

-- Statement to prove that the borrowed dimes is 4
theorem borrowed_dimes_calculation : (original_dimes - remaining_dimes) = 4 :=
by
  -- This is the proof section which follows by simple arithmetic computation
  sorry

end borrowed_dimes_calculation_l704_70458


namespace problem_statement_l704_70417

variables {x y P Q : ℝ}

theorem problem_statement (h1 : x^2 + y^2 = (x + y)^2 + P) (h2 : x^2 + y^2 = (x - y)^2 + Q) : P = -2 * x * y ∧ Q = 2 * x * y :=
by
  sorry

end problem_statement_l704_70417


namespace probability_all_white_balls_l704_70442

-- Definitions
def total_balls : ℕ := 15
def white_balls : ℕ := 8
def black_balls : ℕ := 7
def balls_drawn : ℕ := 7

-- Lean theorem statement
theorem probability_all_white_balls :
  (Nat.choose white_balls balls_drawn : ℚ) / (Nat.choose total_balls balls_drawn) = 8 / 6435 :=
sorry

end probability_all_white_balls_l704_70442


namespace triangle_XYZ_median_l704_70483

theorem triangle_XYZ_median (XYZ : Triangle) (YZ : ℝ) (XM : ℝ) (XY2_add_XZ2 : ℝ) 
  (hYZ : YZ = 12) (hXM : XM = 7) : XY2_add_XZ2 = 170 → N - n = 0 := by
  sorry

end triangle_XYZ_median_l704_70483


namespace game_prob_comparison_l704_70464

theorem game_prob_comparison
  (P_H : ℚ) (P_T : ℚ) (h : P_H = 3/4 ∧ P_T = 1/4)
  (independent : ∀ (n : ℕ), (1 - P_H)^n = (1 - P_T)^n) :
  ((P_H^4 + P_T^4) = (P_H^3 * P_T^2 + P_T^3 * P_H^2) + 1/4) :=
by
  sorry

end game_prob_comparison_l704_70464


namespace remove_five_magazines_l704_70418

theorem remove_five_magazines (magazines : Fin 10 → Set α) 
  (coffee_table : Set α) 
  (h_cover : (⋃ i, magazines i) = coffee_table) :
  ∃ ( S : Set α), S ⊆ coffee_table ∧ (∃ (removed : Finset (Fin 10)), removed.card = 5 ∧ 
    coffee_table \ (⋃ i ∈ removed, magazines i) ⊆ S ∧ (S = coffee_table \ (⋃ i ∈ removed, magazines i) ) ∧ 
    (⋃ i ∉ removed, magazines i) ∩ S = ∅) := 
sorry

end remove_five_magazines_l704_70418


namespace remaining_amount_spent_on_watermelons_l704_70471

def pineapple_cost : ℕ := 7
def total_spent : ℕ := 38
def pineapples_purchased : ℕ := 2

theorem remaining_amount_spent_on_watermelons:
  total_spent - (pineapple_cost * pineapples_purchased) = 24 :=
by
  sorry

end remaining_amount_spent_on_watermelons_l704_70471


namespace max_a_squared_b_l704_70407

theorem max_a_squared_b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a * (a + b) = 27) : a^2 * b ≤ 54 :=
sorry

end max_a_squared_b_l704_70407


namespace range_of_a_l704_70416

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 0 < x → 2 * x + 1 / x - a > 0) → a < 2 * Real.sqrt 2 :=
by
  intro h
  sorry

end range_of_a_l704_70416


namespace market_value_of_house_l704_70449

theorem market_value_of_house 
  (M : ℝ) -- Market value of the house
  (S : ℝ) -- Selling price of the house
  (P : ℝ) -- Pre-tax amount each person gets
  (after_tax : ℝ := 135000) -- Each person's amount after taxes
  (tax_rate : ℝ := 0.10) -- Tax rate
  (num_people : ℕ := 4) -- Number of people splitting the revenue
  (over_market_value_rate : ℝ := 0.20): 
  S = M + over_market_value_rate * M → 
  (num_people * P) = S → 
  after_tax = (1 - tax_rate) * P → 
  M = 500000 := 
by
  sorry

end market_value_of_house_l704_70449


namespace P_sufficient_but_not_necessary_for_Q_l704_70437

variable (x : ℝ)

def P := x ≥ 0
def Q := 2 * x + 1 / (2 * x + 1) ≥ 1

theorem P_sufficient_but_not_necessary_for_Q : (P x → Q x) ∧ ¬(Q x → P x) :=
by
  sorry

end P_sufficient_but_not_necessary_for_Q_l704_70437


namespace part_a_part_b_l704_70474

-- Definitions based on the conditions:
def probability_of_hit (p : ℝ) := p
def probability_of_miss (p : ℝ) := 1 - p

-- Condition: exactly three unused rockets after firing at five targets
def exactly_three_unused_rockets (p : ℝ) : ℝ := 10 * (probability_of_hit p) ^ 3 * (probability_of_miss p) ^ 2

-- Condition: expected number of targets hit when there are nine targets
def expected_targets_hit (p : ℝ) : ℝ := 10 * p - p ^ 10

-- Lean 4 statements representing the proof problems:
theorem part_a (p : ℝ) (h_p_nonneg : 0 ≤ p) (h_p_le_one : p ≤ 1) : 
  exactly_three_unused_rockets p = 10 * p ^ 3 * (1 - p) ^ 2 :=
by sorry

theorem part_b (p : ℝ) (h_p_nonneg : 0 ≤ p) (h_p_le_one : p ≤ 1) :
  expected_targets_hit p = 10 * p - p ^ 10 :=
by sorry

end part_a_part_b_l704_70474


namespace number_line_is_line_l704_70444

-- Define the terms
def number_line : Type := ℝ -- Assume number line can be considered real numbers for simplicity
def is_line (l : Type) : Prop := l = ℝ

-- Proving that number line is a line.
theorem number_line_is_line : is_line number_line :=
by {
  -- by definition of the number_line and is_line
  sorry
}

end number_line_is_line_l704_70444


namespace smallest_n_not_divisible_by_10_smallest_n_correct_l704_70433

theorem smallest_n_not_divisible_by_10 :
  ∃ n ≥ 2017, n % 4 = 0 ∧ (1^n + 2^n + 3^n + 4^n) % 10 ≠ 0 :=
by
  -- Existence proof of such n is omitted
  sorry

def smallest_n : Nat :=
  Nat.find $ smallest_n_not_divisible_by_10

theorem smallest_n_correct : smallest_n = 2020 :=
by
  -- Correctness proof of smallest_n is omitted
  sorry

end smallest_n_not_divisible_by_10_smallest_n_correct_l704_70433


namespace at_op_subtraction_l704_70408

-- Define the operation @
def at_op (x y : ℝ) : ℝ := 3 * x * y - 2 * x + y

-- Prove the problem statement
theorem at_op_subtraction :
  at_op 6 4 - at_op 4 6 = -6 :=
by
  sorry

end at_op_subtraction_l704_70408


namespace min_value_of_3a_plus_2_l704_70434

theorem min_value_of_3a_plus_2 
  (a : ℝ) 
  (h : 4 * a^2 + 7 * a + 3 = 2)
  : 3 * a + 2 >= -1 :=
sorry

end min_value_of_3a_plus_2_l704_70434

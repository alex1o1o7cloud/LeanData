import Mathlib

namespace cats_left_l416_41605

theorem cats_left (siamese house sold : ℕ) (h1 : siamese = 12) (h2 : house = 20) (h3 : sold = 20) :  
  (siamese + house) - sold = 12 := 
by
  sorry

end cats_left_l416_41605


namespace inscribed_circle_radius_of_DEF_l416_41630

theorem inscribed_circle_radius_of_DEF (DE DF EF : ℝ) (r : ℝ)
  (hDE : DE = 8) (hDF : DF = 8) (hEF : EF = 10) :
  r = 5 * Real.sqrt 39 / 13 :=
by
  sorry

end inscribed_circle_radius_of_DEF_l416_41630


namespace find_numbers_l416_41631

theorem find_numbers (x : ℚ) (a : ℚ) (b : ℚ) (h₁ : a = 8 * x) (h₂ : b = x^2 - 1) :
  (a * b + a = (2 * x)^3) ∧ (a * b + b = (2 * x - 1)^3) → 
  x = 14 / 13 ∧ a = 112 / 13 ∧ b = 27 / 169 :=
by
  intros h
  sorry

end find_numbers_l416_41631


namespace task_D_cannot_be_sampled_l416_41603

def task_A := "Measuring the range of a batch of shells"
def task_B := "Determining the content of a certain microorganism in ocean waters"
def task_C := "Calculating the difficulty of each question on the math test after the college entrance examination"
def task_D := "Checking the height and weight of all sophomore students in a school"

def sampling_method (description: String) : Prop :=
  description = task_A ∨ description = task_B ∨ description = task_C

theorem task_D_cannot_be_sampled : ¬ sampling_method task_D :=
sorry

end task_D_cannot_be_sampled_l416_41603


namespace smallest_D_for_inequality_l416_41680

theorem smallest_D_for_inequality :
  ∃ D : ℝ, (∀ x y z : ℝ, 2 * x^2 + 3 * y^2 + z^2 + 3 ≥ D * (x + y + z)) ∧ 
           D = -Real.sqrt (72 / 11) :=
by
  sorry

end smallest_D_for_inequality_l416_41680


namespace good_goods_not_cheap_l416_41626

-- Define the propositions "good goods" and "not cheap"
variables (p q : Prop)

-- State that "good goods are not cheap" is expressed by the implication p → q
theorem good_goods_not_cheap : p → q → (p → q) ↔ (p ∧ q → p ∧ q) := by
  sorry

end good_goods_not_cheap_l416_41626


namespace correct_calculation_l416_41618

variable {a : ℝ} (ha : a ≠ 0)

theorem correct_calculation (a : ℝ) (ha : a ≠ 0) : (a^2 * a^3 = a^5) :=
by sorry

end correct_calculation_l416_41618


namespace markup_amount_l416_41688

def purchase_price : ℝ := 48
def overhead_percentage : ℝ := 0.35
def net_profit : ℝ := 18

def overhead : ℝ := purchase_price * overhead_percentage
def total_cost : ℝ := purchase_price + overhead
def selling_price : ℝ := total_cost + net_profit
def markup : ℝ := selling_price - purchase_price

theorem markup_amount : markup = 34.80 := by
  sorry

end markup_amount_l416_41688


namespace sqrt_three_irrational_l416_41673

-- Define what it means for a number to be rational
def is_rational (x : ℝ) : Prop := ∃ (p q : ℤ), q ≠ 0 ∧ x = p / q

-- Define what it means for a number to be irrational
def is_irrational (x : ℝ) : Prop := ¬ is_rational x

-- State that sqrt(3) is irrational
theorem sqrt_three_irrational : is_irrational (Real.sqrt 3) :=
sorry

end sqrt_three_irrational_l416_41673


namespace jane_oldest_babysat_age_l416_41608

-- Given conditions
def jane_babysitting_has_constraints (jane_current_age jane_stop_babysitting_age jane_start_babysitting_age : ℕ) : Prop :=
  jane_current_age - jane_stop_babysitting_age = 10 ∧
  jane_stop_babysitting_age - jane_start_babysitting_age = 2

-- Helper definition for prime number constraint
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m < n, m > 1 → ¬ (n % m = 0)

-- Main goal: the current age of the oldest person Jane could have babysat is 19
theorem jane_oldest_babysat_age
  (jane_current_age jane_stop_babysitting_age jane_start_babysitting_age : ℕ)
  (H_constraints : jane_babysitting_has_constraints jane_current_age jane_stop_babysitting_age jane_start_babysitting_age) :
  ∃ (child_age : ℕ), child_age = 19 ∧ is_prime child_age ∧
  (child_age = (jane_stop_babysitting_age / 2 + 10) ∨ child_age = (jane_stop_babysitting_age / 2 + 9)) :=
sorry  -- Proof to be filled in.

end jane_oldest_babysat_age_l416_41608


namespace find_coordinates_of_P_l416_41692

noncomputable def pointP_minimizes_dot_product : Prop :=
  let OA := (2, 2)
  let OB := (4, 1)
  let AP x := (x - 2, -2)
  let BP x := (x - 4, -1)
  let dot_product x := (AP x).1 * (BP x).1 + (AP x).2 * (BP x).2
  ∃ x, (dot_product x = (x - 3) ^ 2 + 1) ∧ (∀ y, dot_product y ≥ dot_product x) ∧ (x = 3)

theorem find_coordinates_of_P : pointP_minimizes_dot_product :=
  sorry

end find_coordinates_of_P_l416_41692


namespace A_leaves_after_one_day_l416_41642

-- Define and state all the conditions
def A_work_rate := 1 / 21
def B_work_rate := 1 / 28
def C_work_rate := 1 / 35
def total_work := 1
def B_time_after_A_leave := 21
def C_intermittent_working_cycle := 3 / 1 -- C works 1 out of every 3 days

-- The statement that needs to be proved
theorem A_leaves_after_one_day :
  ∃ x : ℕ, x = 1 ∧
  (A_work_rate * x + B_work_rate * x + (C_work_rate * (x / C_intermittent_working_cycle)) + (B_work_rate * B_time_after_A_leave) + (C_work_rate * (B_time_after_A_leave / C_intermittent_working_cycle)) = total_work) :=
sorry

end A_leaves_after_one_day_l416_41642


namespace parallel_lines_have_equal_slopes_l416_41640

theorem parallel_lines_have_equal_slopes (m : ℝ) :
  (∀ x y : ℝ, x + 2 * y - 1 = 0 → mx - y = 0) → m = -1 / 2 :=
by
  sorry

end parallel_lines_have_equal_slopes_l416_41640


namespace div_power_n_minus_one_l416_41675

theorem div_power_n_minus_one (n : ℕ) (hn : n > 0) (h : n ∣ (2^n - 1)) : n = 1 := by
  sorry

end div_power_n_minus_one_l416_41675


namespace f_strictly_increasing_on_l416_41655

-- Define the function
def f (x : ℝ) : ℝ := x^2 * (2 - x)

-- Define the derivative of the function
def f' (x : ℝ) : ℝ := -3 * x^2 + 4 * x

-- Define the property that the function is strictly increasing on an interval
def strictly_increasing_on (a b : ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

-- State the theorem
theorem f_strictly_increasing_on : strictly_increasing_on 0 (4/3) f :=
sorry

end f_strictly_increasing_on_l416_41655


namespace decreasing_exponential_range_l416_41647

theorem decreasing_exponential_range {a : ℝ} (h : ∀ x y : ℝ, x < y → (a + 1)^x > (a + 1)^y) : -1 < a ∧ a < 0 :=
sorry

end decreasing_exponential_range_l416_41647


namespace cube_sum_from_square_l416_41693

noncomputable def a_plus_inv_a_squared_eq_5 (a : ℝ) : Prop :=
  (a + 1/a) ^ 2 = 5

theorem cube_sum_from_square (a : ℝ) (h : a_plus_inv_a_squared_eq_5 a) :
  a^3 + (1/a)^3 = 2 * Real.sqrt 5 ∨ a^3 + (1/a)^3 = -2 * Real.sqrt 5 :=
by
  sorry

end cube_sum_from_square_l416_41693


namespace pills_first_day_l416_41658

theorem pills_first_day (P : ℕ) 
  (h1 : P + (P + 2) + (P + 4) + (P + 6) + (P + 8) + (P + 10) + (P + 12) = 49) : 
  P = 1 :=
by sorry

end pills_first_day_l416_41658


namespace ballsInBoxes_theorem_l416_41613

noncomputable def ballsInBoxes : ℕ :=
  let distributions := [(5,0,0), (4,1,0), (3,2,0), (3,1,1), (2,2,1)]
  distributions.foldl (λ acc (x : ℕ × ℕ × ℕ) =>
    acc + match x with
      | (5,0,0) => 3
      | (4,1,0) => 6
      | (3,2,0) => 6
      | (3,1,1) => 6
      | (2,2,1) => 3
      | _ => 0
  ) 0

theorem ballsInBoxes_theorem : ballsInBoxes = 24 :=
by
  unfold ballsInBoxes
  rfl

end ballsInBoxes_theorem_l416_41613


namespace find_plane_equation_l416_41695

def point := ℝ × ℝ × ℝ

def plane_equation (A B C D : ℝ) (x y z : ℝ) : Prop :=
  A * x + B * y + C * z + D = 0

def points := (0, 3, -1) :: (4, 7, 1) :: (2, 5, 0) :: []

def correct_plane_equation : Prop :=
  ∃ A B C D : ℝ, plane_equation A B C D = fun x y z => A * x + B * y + C * z + D = 0 ∧ 
  (A, B, C, D) = (0, 1, -2, -5) ∧ ∀ x y z, (x, y, z) ∈ points → plane_equation A B C D x y z

theorem find_plane_equation : correct_plane_equation :=
sorry

end find_plane_equation_l416_41695


namespace find_a4_and_s5_l416_41679

def geometric_sequence (a : ℕ → ℚ) (q : ℚ) : Prop :=
  ∀ n, a (n + 1) = a n * q

variable (a : ℕ → ℚ) (q : ℚ)

axiom condition_1 : a 1 + a 3 = 10
axiom condition_2 : a 4 + a 6 = 1 / 4

theorem find_a4_and_s5 (h_geom : geometric_sequence a q) :
  a 4 = 1 ∧ (a 1 * (1 - q^5) / (1 - q)) = 31 / 2 :=
by
  sorry

end find_a4_and_s5_l416_41679


namespace range_of_k_l416_41684

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := x^(-k^2 + k + 2)

theorem range_of_k (k : ℝ) : (∃ k, (f 2 k < f 3 k)) ↔ (-1 < k) ∧ (k < 2) :=
by
  sorry

end range_of_k_l416_41684


namespace initial_roses_l416_41621

theorem initial_roses (R : ℕ) (h : R + 16 = 23) : R = 7 :=
sorry

end initial_roses_l416_41621


namespace total_pure_acid_in_mixture_l416_41629

-- Definitions of the conditions
def solution1_volume : ℝ := 8
def solution1_concentration : ℝ := 0.20
def solution2_volume : ℝ := 5
def solution2_concentration : ℝ := 0.35

-- Proof statement
theorem total_pure_acid_in_mixture :
  solution1_volume * solution1_concentration + solution2_volume * solution2_concentration = 3.35 := by
  sorry

end total_pure_acid_in_mixture_l416_41629


namespace value_of_expression_l416_41616

theorem value_of_expression (x y : ℝ) (h1 : x = Real.sqrt 5 + Real.sqrt 3) (h2 : y = Real.sqrt 5 - Real.sqrt 3) : x^2 + x * y + y^2 = 18 :=
by sorry

end value_of_expression_l416_41616


namespace melanies_plums_l416_41610

variable (pickedPlums : ℕ)
variable (gavePlums : ℕ)

theorem melanies_plums (h1 : pickedPlums = 7) (h2 : gavePlums = 3) : (pickedPlums - gavePlums) = 4 :=
by
  sorry

end melanies_plums_l416_41610


namespace students_on_bleachers_l416_41635

theorem students_on_bleachers (F B : ℕ) (h1 : F + B = 26) (h2 : F / (F + B) = 11 / 13) : B = 4 :=
by sorry

end students_on_bleachers_l416_41635


namespace max_a_l416_41697

-- Define the conditions
def line_equation (m : ℚ) (x : ℤ) : ℚ := m * x + 3

def no_lattice_points (m : ℚ) : Prop :=
  ∀ (x : ℤ), 1 ≤ x ∧ x ≤ 50 → ¬ ∃ (y : ℤ), line_equation m x = y

def m_range (m a : ℚ) : Prop := (2 : ℚ) / 5 < m ∧ m < a

-- Define the problem statement
theorem max_a (a : ℚ) : (a = 22 / 51) ↔ (∃ m, no_lattice_points m ∧ m_range m a) :=
by 
  sorry

end max_a_l416_41697


namespace eval_gg3_l416_41656

def g (x : ℕ) : ℕ := 3 * x^2 + 3 * x - 2

theorem eval_gg3 : g (g 3) = 3568 :=
by 
  sorry

end eval_gg3_l416_41656


namespace middle_marble_radius_l416_41671

theorem middle_marble_radius (r_1 r_5 : ℝ) (h1 : r_1 = 8) (h5 : r_5 = 18) : 
  ∃ r_3 : ℝ, r_3 = 12 :=
by
  let r_3 := Real.sqrt (r_1 * r_5)
  have h : r_3 = 12 := sorry
  exact ⟨r_3, h⟩

end middle_marble_radius_l416_41671


namespace find_g7_l416_41690

namespace ProofProblem

variable (g : ℝ → ℝ)
variable (h1 : ∀ x y : ℝ, g (x + y) = g x + g y)
variable (h2 : g 6 = 8)

theorem find_g7 : g 7 = 28 / 3 := by
  sorry

end ProofProblem

end find_g7_l416_41690


namespace hyperbola_asymptote_b_value_l416_41651

theorem hyperbola_asymptote_b_value (b : ℝ) (hb : 0 < b) : 
  (∀ x y, x^2 - y^2 / b^2 = 1 → y = 3 * x ∨ y = -3 * x) → b = 3 := 
by
  sorry

end hyperbola_asymptote_b_value_l416_41651


namespace percentage_of_profits_to_revenues_l416_41663

theorem percentage_of_profits_to_revenues (R P : ℝ) (h1 : 0.7 * R = R - 0.3 * R) (h2 : 0.105 * R = 0.15 * (0.7 * R)) (h3 : 0.105 * R = 1.0499999999999999 * P) :
  (P / R) * 100 = 10 :=
by
  sorry

end percentage_of_profits_to_revenues_l416_41663


namespace sum_k_over_3_pow_k_eq_three_fourths_l416_41670

noncomputable def sum_k_over_3_pow_k : ℝ :=
  ∑' k : ℕ, (k + 1) / 3 ^ (k + 1)

theorem sum_k_over_3_pow_k_eq_three_fourths :
  sum_k_over_3_pow_k = 3 / 4 := sorry

end sum_k_over_3_pow_k_eq_three_fourths_l416_41670


namespace min_value_of_expression_l416_41601

theorem min_value_of_expression (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_abc : a * b * c = 4) :
  (3 * a + b) * (2 * b + 3 * c) * (a * c + 4) ≥ 384 := 
by sorry

end min_value_of_expression_l416_41601


namespace sin_x1_x2_value_l416_41602

open Real

theorem sin_x1_x2_value (m x1 x2 : ℝ) :
  (2 * sin (2 * x1) + cos (2 * x1) = m) →
  (2 * sin (2 * x2) + cos (2 * x2) = m) →
  (0 ≤ x1 ∧ x1 ≤ π / 2) →
  (0 ≤ x2 ∧ x2 ≤ π / 2) →
  sin (x1 + x2) = 2 * sqrt 5 / 5 := 
by
  sorry

end sin_x1_x2_value_l416_41602


namespace geometric_sequence_ratio_l416_41672

theorem geometric_sequence_ratio (a1 : ℕ) (S : ℕ → ℕ) (r : ℤ) (h1 : r = -2) (h2 : ∀ n, S n = a1 * (1 - r ^ n) / (1 - r)) :
  S 4 / S 2 = 5 :=
by
  -- Placeholder for proof steps
  sorry

end geometric_sequence_ratio_l416_41672


namespace nina_homework_total_l416_41676

-- Definitions based on conditions
def ruby_math_homework : Nat := 6
def ruby_reading_homework : Nat := 2
def nina_math_homework : Nat := 4 * ruby_math_homework
def nina_reading_homework : Nat := 8 * ruby_reading_homework
def nina_total_homework : Nat := nina_math_homework + nina_reading_homework

-- The theorem to prove
theorem nina_homework_total : nina_total_homework = 40 := by
  sorry

end nina_homework_total_l416_41676


namespace opposite_of_3_is_neg3_l416_41683

theorem opposite_of_3_is_neg3 : forall (n : ℤ), n = 3 -> -n = -3 :=
by
  sorry

end opposite_of_3_is_neg3_l416_41683


namespace problem1_problem2_l416_41606

noncomputable def f (x : ℝ) (a : ℝ) := (1 / 2) * x^2 + 2 * a * x
noncomputable def g (x : ℝ) (a : ℝ) (b : ℝ) := 3 * a^2 * Real.log x + b

theorem problem1 (a b x₀ : ℝ) (h : x₀ = a):
  a > 0 →
  (1 / 2) * x₀^2 + 2 * a * x₀ = 3 * a^2 * Real.log x₀ + b →
  x₀ + 2 * a = 3 * a^2 / x₀ →
  b = (5 * a^2 / 2) - 3 * a^2 * Real.log a := sorry

theorem problem2 (a b : ℝ):
  -2 ≤ b ∧ b ≤ 2 →
  ∀ x > 0, x < 4 →
  ∀ x, x - b + 3 * a^2 / x ≥ 0 →
  a ≥ Real.sqrt 3 / 3 ∨ a ≤ -Real.sqrt 3 / 3 := sorry

end problem1_problem2_l416_41606


namespace peter_money_left_l416_41678

variable (soda_cost : ℝ) (money_brought : ℝ) (soda_ounces : ℝ)

theorem peter_money_left (h1 : soda_cost = 0.25) (h2 : money_brought = 2) (h3 : soda_ounces = 6) : 
    money_brought - soda_ounces * soda_cost = 0.50 := 
by 
  sorry

end peter_money_left_l416_41678


namespace min_value_l416_41638

theorem min_value (x : ℝ) (h : x > 1) : x + 4 / (x - 1) ≥ 5 :=
sorry

end min_value_l416_41638


namespace suff_condition_not_necc_condition_l416_41696

variable (x : ℝ)

def A : Prop := 0 < x ∧ x < 5
def B : Prop := |x - 2| < 3

theorem suff_condition : A x → B x := by
  sorry

theorem not_necc_condition : B x → ¬ A x := by
  sorry

end suff_condition_not_necc_condition_l416_41696


namespace jacket_price_is_48_l416_41657

-- Definitions according to the conditions
def jacket_problem (P S D : ℝ) : Prop :=
  S = P + 0.40 * S ∧
  D = 0.80 * S ∧
  16 = D - P

-- Statement of the theorem
theorem jacket_price_is_48 :
  ∃ P S D, jacket_problem P S D ∧ P = 48 :=
by
  sorry

end jacket_price_is_48_l416_41657


namespace steak_amount_per_member_l416_41682

theorem steak_amount_per_member : 
  ∀ (num_members steaks_needed ounces_per_steak total_ounces each_amount : ℕ),
    num_members = 5 →
    steaks_needed = 4 →
    ounces_per_steak = 20 →
    total_ounces = steaks_needed * ounces_per_steak →
    each_amount = total_ounces / num_members →
    each_amount = 16 :=
by
  intros num_members steaks_needed ounces_per_steak total_ounces each_amount
  intro h_members h_steaks h_ounces_per_steak h_total_ounces h_each_amount
  sorry

end steak_amount_per_member_l416_41682


namespace simplify_expression_l416_41691

-- Define the given conditions
def pow_2_5 : ℕ := 32
def pow_4_4 : ℕ := 256
def pow_2_2 : ℕ := 4
def pow_neg_2_3 : ℤ := -8

-- State the theorem to prove
theorem simplify_expression : 
  (pow_2_5 + pow_4_4) * (pow_2_2 - pow_neg_2_3)^8 = 123876479488 := 
by
  sorry

end simplify_expression_l416_41691


namespace range_of_m_l416_41648

theorem range_of_m (m : ℝ) (h : 9 > m^2 ∧ m ≠ 0) : m ∈ Set.Ioo (-3) 0 ∨ m ∈ Set.Ioo 0 3 := 
sorry

end range_of_m_l416_41648


namespace Polly_lunch_time_l416_41685

-- Define the conditions
def breakfast_time_per_day := 20
def total_days_in_week := 7
def dinner_time_4_days := 10
def remaining_days_in_week := 3
def remaining_dinner_time_per_day := 30
def total_cooking_time := 305

-- Define the total time Polly spends cooking breakfast in a week
def total_breakfast_time := breakfast_time_per_day * total_days_in_week

-- Define the total time Polly spends cooking dinner in a week
def total_dinner_time := (dinner_time_4_days * 4) + (remaining_dinner_time_per_day * remaining_days_in_week)

-- Define the time Polly spends cooking lunch in a week
def lunch_time := total_cooking_time - (total_breakfast_time + total_dinner_time)

-- The theorem to prove Polly's lunch time
theorem Polly_lunch_time : lunch_time = 35 :=
by
  sorry

end Polly_lunch_time_l416_41685


namespace maria_bottles_count_l416_41625

-- Definitions from the given conditions
def b_initial : ℕ := 23
def d : ℕ := 12
def g : ℕ := 5
def b : ℕ := 65

-- Definition of the question based on conditions
def b_final : ℕ := b_initial - d - g + b

-- The statement to prove the correctness of the answer
theorem maria_bottles_count : b_final = 71 := by
  -- We skip the proof for this statement
  sorry

end maria_bottles_count_l416_41625


namespace sum_first_9_terms_l416_41624

variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)
variable (d : ℝ)
variable (a_1 a_2 a_3 a_4 a_5 a_6 : ℝ)

-- Conditions
axiom h1 : a 1 + a 5 = 10
axiom h2 : a 2 + a 6 = 14

-- Calculations
axiom h3 : a 3 = 5
axiom h4 : a 4 = 7
axiom h5 : d = 2
axiom h6 : a 5 = 9

-- The sum of the first 9 terms
axiom h7 : S 9 = 9 * a 5

theorem sum_first_9_terms : S 9 = 81 :=
by {
  sorry
}

end sum_first_9_terms_l416_41624


namespace coin_difference_l416_41623

-- Definitions based on problem conditions
def denominations : List ℕ := [5, 10, 25, 50]
def amount_owed : ℕ := 55

-- Proof statement
theorem coin_difference :
  let min_coins := 1 + 1 -- one 50-cent coin and one 5-cent coin
  let max_coins := 11 -- eleven 5-cent coins
  max_coins - min_coins = 9 :=
by
  -- Proof details skipped
  sorry

end coin_difference_l416_41623


namespace parallel_lines_slope_l416_41660

theorem parallel_lines_slope (a : ℝ) (h : ∀ x y : ℝ, (x + a * y + 6 = 0) → ((a - 2) * x + 3 * y + 2 * a = 0)) : a = -1 :=
by
  sorry

end parallel_lines_slope_l416_41660


namespace percentage_of_loss_is_10_l416_41668

-- Definitions based on conditions
def cost_price : ℝ := 1800
def selling_price : ℝ := 1620
def loss : ℝ := cost_price - selling_price

-- The goal: prove the percentage of loss equals 10%
theorem percentage_of_loss_is_10 :
  (loss / cost_price) * 100 = 10 := by
  sorry

end percentage_of_loss_is_10_l416_41668


namespace cafeteria_sales_comparison_l416_41637

theorem cafeteria_sales_comparison
  (S : ℝ) -- initial sales
  (a : ℝ) -- monthly increment for Cafeteria A
  (p : ℝ) -- monthly percentage increment for Cafeteria B
  (h1 : S > 0) -- initial sales are positive
  (h2 : a > 0) -- constant increment for Cafeteria A is positive
  (h3 : p > 0) -- constant percentage increment for Cafeteria B is positive
  (h4 : S + 8 * a = S * (1 + p) ^ 8) -- sales are equal in September 2013
  (h5 : S = S) -- sales are equal in January 2013 (trivially true)
  : S + 4 * a > S * (1 + p) ^ 4 := 
sorry

end cafeteria_sales_comparison_l416_41637


namespace arithmetic_sequence_condition_l416_41609

theorem arithmetic_sequence_condition (a b c d : ℝ) :
  (∃ k : ℝ, b = a + k ∧ c = a + 2*k ∧ d = a + 3*k) ↔ (a + d = b + c) :=
sorry

end arithmetic_sequence_condition_l416_41609


namespace number_of_multiples_of_15_l416_41687

theorem number_of_multiples_of_15 (a b : ℕ) (h₁ : a = 15) (h₂ : b = 305) : 
  ∃ n : ℕ, n = 20 ∧ ∀ k, (1 ≤ k ∧ k ≤ n) → (15 * k) ≥ a ∧ (15 * k) ≤ b := by
  sorry

end number_of_multiples_of_15_l416_41687


namespace compare_ab_1_to_a_b_two_pow_b_eq_one_div_b_sign_of_expression_l416_41614

-- Part 1
theorem compare_ab_1_to_a_b {a b : ℝ} (h1 : a^b * b^a + Real.log b / Real.log a = 0) (ha : a > 0) (hb : b > 0) : ab + 1 < a + b := sorry

-- Part 2
theorem two_pow_b_eq_one_div_b {b : ℝ} (h : 2^b * b^2 + Real.log b / Real.log 2 = 0) : 2^b = 1 / b := sorry

-- Part 3
theorem sign_of_expression {b : ℝ} (h : 2^b * b^2 + Real.log b / Real.log 2 = 0) : (2 * b + 1 - Real.sqrt 5) * (3 * b - 2) < 0 := sorry

end compare_ab_1_to_a_b_two_pow_b_eq_one_div_b_sign_of_expression_l416_41614


namespace sum_of_intersections_l416_41694

theorem sum_of_intersections :
  (∃ x1 y1 x2 y2 x3 y3 x4 y4, 
    y1 = (x1 - 1)^2 ∧ y2 = (x2 - 1)^2 ∧ y3 = (x3 - 1)^2 ∧ y4 = (x4 - 1)^2 ∧
    x1 - 2 = (y1 + 1)^2 ∧ x2 - 2 = (y2 + 1)^2 ∧ x3 - 2 = (y3 + 1)^2 ∧ x4 - 2 = (y4 + 1)^2 ∧
    (x1 + x2 + x3 + x4 + y1 + y2 + y3 + y4) = 2) :=
sorry

end sum_of_intersections_l416_41694


namespace relation_P_Q_l416_41661

def P : Set ℝ := {x | x ≠ 0}
def Q : Set ℝ := {x | x > 0}
def complement_P : Set ℝ := {0}

theorem relation_P_Q : Q ∩ complement_P = ∅ := 
by sorry

end relation_P_Q_l416_41661


namespace roses_cut_l416_41667

def initial_roses : ℕ := 6
def new_roses : ℕ := 16

theorem roses_cut : new_roses - initial_roses = 10 := by
  sorry

end roses_cut_l416_41667


namespace initial_pipes_l416_41639

variables (x : ℕ)

-- Defining the conditions
def one_pipe_time := x -- time for 1 pipe to fill the tank in hours
def eight_pipes_time := 1 / 4 -- 15 minutes = 1/4 hour

-- Proving the number of pipes
theorem initial_pipes (h1 : eight_pipes_time * 8 = one_pipe_time) : x = 2 :=
by
  sorry

end initial_pipes_l416_41639


namespace arccos_cos_11_l416_41633

theorem arccos_cos_11 : Real.arccos (Real.cos 11) = 1.425 :=
by
  sorry

end arccos_cos_11_l416_41633


namespace fraction_without_cable_or_vcr_l416_41620

theorem fraction_without_cable_or_vcr (T : ℕ) (h1 : ℚ) (h2 : ℚ) (h3 : ℚ) 
  (h1 : h1 = 1 / 5 * T) 
  (h2 : h2 = 1 / 10 * T) 
  (h3 : h3 = 1 / 3 * (1 / 5 * T)) 
: (T - (1 / 5 * T + 1 / 10 * T - 1 / 3 * (1 / 5 * T))) / T = 23 / 30 := 
by 
  sorry

end fraction_without_cable_or_vcr_l416_41620


namespace min_valid_n_l416_41644

theorem min_valid_n (n : ℕ) (h_pos : 0 < n) (h_int : ∃ m : ℕ, m * m = 51 + n) : n = 13 :=
  sorry

end min_valid_n_l416_41644


namespace total_seats_l416_41664

theorem total_seats (KA_pos : ℕ) (SL_pos : ℕ) (h1 : KA_pos = 10) (h2 : SL_pos = 29) (h3 : SL_pos = KA_pos + (KA_pos * 2 - 1) / 2):
  let total_positions := 2 * (SL_pos - KA_pos - 1) + 2
  total_positions = 38 :=
by
  sorry

end total_seats_l416_41664


namespace vec_c_is_linear_comb_of_a_b_l416_41653

structure Vec2 :=
  (x : ℝ)
  (y : ℝ)

def a := Vec2.mk 1 2
def b := Vec2.mk (-2) 3
def c := Vec2.mk 4 1

theorem vec_c_is_linear_comb_of_a_b : c = Vec2.mk (2 * a.x - b.x) (2 * a.y - b.y) :=
  by
    sorry

end vec_c_is_linear_comb_of_a_b_l416_41653


namespace supplement_of_angle_l416_41617

theorem supplement_of_angle (complement_of_angle : ℝ) (h1 : complement_of_angle = 30) :
  ∃ (angle supplement_angle : ℝ), angle + complement_of_angle = 90 ∧ angle + supplement_angle = 180 ∧ supplement_angle = 120 :=
by
  sorry

end supplement_of_angle_l416_41617


namespace range_of_omega_l416_41627

theorem range_of_omega (ω : ℝ) (h_pos : ω > 0) (h_three_high_points : (9 * π / 2) ≤ ω + π / 4 ∧ ω + π / 4 < 6 * π + π / 2) : 
           (17 * π / 4) ≤ ω ∧ ω < (25 * π / 4) :=
  sorry

end range_of_omega_l416_41627


namespace find_f_at_8_l416_41615

theorem find_f_at_8 (f : ℝ → ℝ) (h : ∀ x : ℝ, f (3 * x - 1) = x^2 + 2 * x + 4) :
  f 8 = 19 :=
sorry

end find_f_at_8_l416_41615


namespace cousins_arrangement_l416_41654

def number_of_arrangements (cousins rooms : ℕ) (min_empty_rooms : ℕ) : ℕ := sorry

theorem cousins_arrangement : number_of_arrangements 5 4 1 = 56 := 
by sorry

end cousins_arrangement_l416_41654


namespace largest_vs_smallest_circles_l416_41622

variable (M : Type) [MetricSpace M] [MeasurableSpace M]

def non_overlapping_circles (M : Type) [MetricSpace M] [MeasurableSpace M] : ℕ := sorry

def covering_circles (M : Type) [MetricSpace M] [MeasurableSpace M] : ℕ := sorry

theorem largest_vs_smallest_circles (M : Type) [MetricSpace M] [MeasurableSpace M] :
  non_overlapping_circles M ≥ covering_circles M :=
sorry

end largest_vs_smallest_circles_l416_41622


namespace trigonometric_identity_l416_41677

theorem trigonometric_identity :
  (3 / (Real.sin (20 * Real.pi / 180))^2) - 
  (1 / (Real.cos (20 * Real.pi / 180))^2) + 
  64 * (Real.sin (20 * Real.pi / 180))^2 = 32 :=
by sorry

end trigonometric_identity_l416_41677


namespace correct_calculation_l416_41699

theorem correct_calculation (a b c d : ℤ) (h1 : a = -1) (h2 : b = -3) (h3 : c = 3) (h4 : d = -3) :
  a * b = c :=
by 
  rw [h1, h2]
  exact h3.symm

end correct_calculation_l416_41699


namespace totalMarbles_l416_41611

def originalMarbles : ℕ := 22
def marblesGiven : ℕ := 20

theorem totalMarbles : originalMarbles + marblesGiven = 42 := by
  sorry

end totalMarbles_l416_41611


namespace arrange_students_l416_41666

theorem arrange_students 
  (students : Fin 6 → Type) 
  (A B : Type) 
  (h1 : ∃ i j, students i = A ∧ students j = B ∧ (i = j + 1 ∨ j = i + 1)) : 
  (∃ (n : ℕ), n = 240) := 
sorry

end arrange_students_l416_41666


namespace find_m_when_circle_tangent_to_line_l416_41698

theorem find_m_when_circle_tangent_to_line 
    (m : ℝ)
    (circle_eq : (x y : ℝ) → (x - 1)^2 + (y - 1)^2 = 4 * m)
    (line_eq : (x y : ℝ) → x + y = 2 * m) :
    (m = 2 + Real.sqrt 3) ∨ (m = 2 - Real.sqrt 3) :=
sorry

end find_m_when_circle_tangent_to_line_l416_41698


namespace max_triangle_side_l416_41686

-- Definitions of conditions
def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def has_perimeter (a b c : ℕ) (p : ℕ) : Prop :=
  a + b + c = p

def different_integers (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c

-- The main theorem to prove
theorem max_triangle_side (a b c : ℕ) (h_triangle : is_triangle a b c)
                         (h_perimeter : has_perimeter a b c 24)
                         (h_diff : different_integers a b c) :
  c ≤ 11 :=
sorry

end max_triangle_side_l416_41686


namespace infinite_series_k3_over_3k_l416_41604

theorem infinite_series_k3_over_3k :
  ∑' k : ℕ, (k^3 : ℝ) / 3^k = 165 / 16 := 
sorry

end infinite_series_k3_over_3k_l416_41604


namespace Jovana_shells_l416_41652

theorem Jovana_shells (initial_shells : ℕ) (added_shells : ℕ) (total_shells : ℕ) :
  initial_shells = 5 → added_shells = 12 → total_shells = initial_shells + added_shells → total_shells = 17 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end Jovana_shells_l416_41652


namespace inequality_proof_l416_41628

theorem inequality_proof (x y z : ℝ) :
  (x^2 + 2 * y^2 + 2 * z^2) / (x^2 + y * z) +
  (y^2 + 2 * z^2 + 2 * x^2) / (y^2 + z * x) +
  (z^2 + 2 * x^2 + 2 * y^2) / (z^2 + x * y) > 6 :=
by
  sorry

end inequality_proof_l416_41628


namespace FI_squared_l416_41634

-- Definitions for the given conditions
-- Note: Further geometric setup and formalization might be necessary to carry 
-- out the complete proof in Lean, but the setup will follow these basic definitions.

-- Let ABCD be a square
def ABCD_square (A B C D : ℝ × ℝ) : Prop :=
  -- conditions for ABCD being a square (to be properly defined based on coordinates and properties)
  sorry

-- Triangle AEH is an equilateral triangle with side length sqrt(3)
def equilateral_AEH (A E H : ℝ × ℝ) (s : ℝ) : Prop :=
  dist A E = s ∧ dist E H = s ∧ dist H A = s 

-- Points E and H lie on AB and DA respectively
-- Points F and G lie on BC and CD respectively
-- Points I and J lie on EH with FI ⊥ EH and GJ ⊥ EH
-- Areas of triangles and quadrilaterals
def geometric_conditions (A B C D E F G H I J : ℝ × ℝ) : Prop :=
  sorry

-- Final statement to prove
theorem FI_squared (A B C D E F G H I J : ℝ × ℝ) (s : ℝ) 
  (h_square: ABCD_square A B C D) 
  (h_equilateral: equilateral_AEH A E H (Real.sqrt 3))
  (h_geo: geometric_conditions A B C D E F G H I J) :
  dist F I ^ 2 = 4 / 3 :=
sorry

end FI_squared_l416_41634


namespace stream_speed_l416_41650

variable (D : ℝ) -- Distance rowed

theorem stream_speed (v : ℝ) (h : D / (60 - v) = 2 * (D / (60 + v))) : v = 20 :=
by
  sorry

end stream_speed_l416_41650


namespace evaluate_expression_1_evaluate_expression_2_l416_41619

-- Problem 1
def expression_1 (a b : Int) : Int :=
  2 * a + 3 * b - 2 * a * b - a - 4 * b - a * b

theorem evaluate_expression_1 : expression_1 6 (-1) = 25 :=
by
  sorry

-- Problem 2
def expression_2 (m n : Int) : Int :=
  m^2 + 2 * m * n + n^2

theorem evaluate_expression_2 (m n : Int) (hm : |m| = 3) (hn : |n| = 2) (hmn : m < n) : expression_2 m n = 1 :=
by
  sorry

end evaluate_expression_1_evaluate_expression_2_l416_41619


namespace find_symbols_l416_41641

theorem find_symbols (x y otimes oplus : ℝ) 
  (h1 : x + otimes * y = 3) 
  (h2 : 3 * x - otimes * y = 1) 
  (h3 : x = oplus) 
  (h4 : y = 1) : 
  otimes = 2 ∧ oplus = 1 := 
by
  sorry

end find_symbols_l416_41641


namespace cylinder_height_in_sphere_l416_41600

noncomputable def height_of_cylinder (r R : ℝ) : ℝ :=
  2 * Real.sqrt (R ^ 2 - r ^ 2)

theorem cylinder_height_in_sphere :
  height_of_cylinder 3 6 = 6 * Real.sqrt 3 :=
by
  sorry

end cylinder_height_in_sphere_l416_41600


namespace intersection_count_l416_41607

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x
noncomputable def g (x : ℝ) : ℝ := x^2 - 4 * x + 5

theorem intersection_count : ∃! (x1 x2 : ℝ), 
  x1 > 0 ∧ x2 > 0 ∧ f x1 = g x1 ∧ f x2 = g x2 ∧ x1 ≠ x2 :=
sorry

end intersection_count_l416_41607


namespace metallic_sheet_length_l416_41662

theorem metallic_sheet_length (w : ℝ) (s : ℝ) (v : ℝ) (L : ℝ) 
  (h_w : w = 38) 
  (h_s : s = 8) 
  (h_v : v = 5632) 
  (h_volume : (L - 2 * s) * (w - 2 * s) * s = v) : 
  L = 48 :=
by
  -- To complete the proof, follow the mathematical steps:
  -- (L - 2 * s) * (w - 2 * s) * s = v
  -- (L - 2 * 8) * (38 - 2 * 8) * 8 = 5632
  -- Simplify and solve for L
  sorry

end metallic_sheet_length_l416_41662


namespace election_percentage_l416_41674

-- Define the total number of votes (V), winner's votes, and the vote difference
def total_votes (V : ℕ) : Prop := V = 1944 + (1944 - 288)

-- Define the percentage calculation from the problem
def percentage_of_votes (votes_received total_votes : ℕ) : ℕ := (votes_received * 100) / total_votes

-- State the core theorem to prove the winner received 54 percent of the total votes
theorem election_percentage (V : ℕ) (h : total_votes V) : percentage_of_votes 1944 V = 54 := by
  sorry

end election_percentage_l416_41674


namespace jill_spent_30_percent_on_food_l416_41689

variables (T F : ℝ)

theorem jill_spent_30_percent_on_food
  (h1 : 0.04 * T = 0.016 * T + 0.024 * T)
  (h2 : 0.40 + 0.30 + F = 1) :
  F = 0.30 :=
by 
  sorry

end jill_spent_30_percent_on_food_l416_41689


namespace value_of_b_l416_41681

noncomputable def problem (a1 a2 a3 a4 a5 : ℤ) (b : ℤ) :=
  (a1 ≠ a2) ∧ (a1 ≠ a3) ∧ (a1 ≠ a4) ∧ (a1 ≠ a5) ∧
  (a2 ≠ a3) ∧ (a2 ≠ a4) ∧ (a2 ≠ a5) ∧
  (a3 ≠ a4) ∧ (a3 ≠ a5) ∧
  (a4 ≠ a5) ∧
  (a1 + a2 + a3 + a4 + a5 = 9) ∧
  ((b - a1) * (b - a2) * (b - a3) * (b - a4) * (b - a5) = 2009) ∧
  (∃ b : ℤ, b = 10)

theorem value_of_b (a1 a2 a3 a4 a5 : ℤ) (b : ℤ) :
  problem a1 a2 a3 a4 a5 b → b = 10 :=
  sorry

end value_of_b_l416_41681


namespace find_g3_l416_41632

-- Define a function g from ℝ to ℝ
variable (g : ℝ → ℝ)

-- Condition: ∀ x, g(3^x) + 2 * x * g(3^(-x)) = 3
axiom condition : ∀ x : ℝ, g (3^x) + 2 * x * g (3^(-x)) = 3

-- The theorem we need to prove
theorem find_g3 : g 3 = -3 := 
by 
  sorry

end find_g3_l416_41632


namespace solve_chestnut_problem_l416_41645

def chestnut_problem : Prop :=
  ∃ (P M L : ℕ), (M = 2 * P) ∧ (L = P + 2) ∧ (P + M + L = 26) ∧ (M = 12)

theorem solve_chestnut_problem : chestnut_problem :=
by 
  sorry

end solve_chestnut_problem_l416_41645


namespace rooms_needed_l416_41649

/-
  We are given that there are 30 students and each hotel room accommodates 5 students.
  Prove that the number of rooms required to accommodate all students is 6.
-/
theorem rooms_needed (total_students : ℕ) (students_per_room : ℕ) (h1 : total_students = 30) (h2 : students_per_room = 5) : total_students / students_per_room = 6 := by
  -- proof
  sorry

end rooms_needed_l416_41649


namespace min_value_of_a_l416_41669

theorem min_value_of_a 
  {f : ℕ → ℝ} 
  (h : ∀ x : ℕ, 0 < x → f x = (x^2 + a * x + 11) / (x + 1)) 
  (ineq : ∀ x : ℕ, 0 < x → f x ≥ 3) : a ≥ -8 / 3 :=
sorry

end min_value_of_a_l416_41669


namespace find_a_l416_41646

theorem find_a (x y a : ℝ) (h1 : x = 2) (h2 : y = 1) (h3 : x - a * y = 3) : a = -1 :=
by
  -- Solution steps go here
  sorry

end find_a_l416_41646


namespace speed_of_boat_is_15_l416_41612

noncomputable def speed_of_boat_in_still_water (x : ℝ) : Prop :=
  ∃ (t : ℝ), t = 1 / 5 ∧ (x + 3) * t = 3.6 ∧ x = 15

theorem speed_of_boat_is_15 (x : ℝ) (t : ℝ) (rate_of_current : ℝ) (distance_downstream : ℝ) :
  rate_of_current = 3 →
  distance_downstream = 3.6 →
  t = 1 / 5 →
  (x + rate_of_current) * t = distance_downstream →
  x = 15 :=
by
  intros h1 h2 h3 h4
  -- proof goes here
  sorry

end speed_of_boat_is_15_l416_41612


namespace equilateral_triangle_properties_l416_41643

noncomputable def equilateral_triangle_perimeter (a : ℝ) : ℝ :=
3 * a

noncomputable def equilateral_triangle_bisector_length (a : ℝ) : ℝ :=
(a * Real.sqrt 3) / 2

theorem equilateral_triangle_properties (a : ℝ) (h : a = 10) :
  equilateral_triangle_perimeter a = 30 ∧
  equilateral_triangle_bisector_length a = 5 * Real.sqrt 3 :=
by
  sorry

end equilateral_triangle_properties_l416_41643


namespace matrix_inverse_eq_scaling_l416_41665

variable (d k : ℚ)

def B : Matrix (Fin 3) (Fin 3) ℚ := ![
  ![1, 2, 3],
  ![4, 5, d],
  ![6, 7, 8]
]

theorem matrix_inverse_eq_scaling :
  (B d)⁻¹ = k • (B d) →
  d = 13/9 ∧ k = -329/52 :=
by
  sorry

end matrix_inverse_eq_scaling_l416_41665


namespace part1_part2_part3_l416_41659

noncomputable def A : Set ℝ := { x | x ≥ 1 ∨ x ≤ -3 }
noncomputable def B : Set ℝ := { x | -4 < x ∧ x < 0 }
noncomputable def C : Set ℝ := { x | x ≤ -4 ∨ x ≥ 0 }

theorem part1 : A ∩ B = { x | -4 < x ∧ x ≤ -3 } := 
by { sorry }

theorem part2 : A ∪ B = { x | x < 0 ∨ x ≥ 1 } := 
by { sorry }

theorem part3 : A ∪ C = { x | x ≤ -3 ∨ x ≥ 0 } := 
by { sorry }

end part1_part2_part3_l416_41659


namespace inequality_f_c_f_a_f_b_l416_41636

-- Define the function f and the conditions
def f : ℝ → ℝ := sorry

noncomputable def a : ℝ := Real.log (1 / Real.pi)
noncomputable def b : ℝ := (Real.log Real.pi) ^ 2
noncomputable def c : ℝ := Real.log (Real.sqrt Real.pi)

-- Theorem statement
theorem inequality_f_c_f_a_f_b :
  (∀ x : ℝ, f x = f (-x)) →
  (∀ x1 x2 : ℝ, 0 < x1 → 0 < x2 → x1 < x2 → f x1 > f x2) →
  f c > f a ∧ f a > f b :=
by
  -- Proof omitted
  sorry

end inequality_f_c_f_a_f_b_l416_41636

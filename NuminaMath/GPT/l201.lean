import Mathlib

namespace NUMINAMATH_GPT_polar_curve_is_parabola_l201_20185

theorem polar_curve_is_parabola (ρ θ : ℝ) (h : 3 * ρ * Real.sin θ ^ 2 + Real.cos θ = 0) : ∃ (x y : ℝ), x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ ∧ 3 * y ^ 2 + x = 0 :=
by
  sorry

end NUMINAMATH_GPT_polar_curve_is_parabola_l201_20185


namespace NUMINAMATH_GPT_difference_between_numbers_l201_20191

-- Given definitions based on conditions
def sum_of_two_numbers (x y : ℝ) : Prop := x + y = 15
def difference_of_two_numbers (x y : ℝ) : Prop := x - y = 10
def difference_of_squares (x y : ℝ) : Prop := x^2 - y^2 = 150

theorem difference_between_numbers (x y : ℝ) 
  (h1 : sum_of_two_numbers x y) 
  (h2 : difference_of_two_numbers x y) 
  (h3 : difference_of_squares x y) :
  x - y = 10 :=
by
  sorry

end NUMINAMATH_GPT_difference_between_numbers_l201_20191


namespace NUMINAMATH_GPT_prince_cd_total_spent_l201_20139

theorem prince_cd_total_spent (total_cds : ℕ)
    (pct_20 : ℕ) (pct_15 : ℕ) (pct_10 : ℕ)
    (bought_20_pct : ℕ) (bought_15_pct : ℕ)
    (bought_10_pct : ℕ) (bought_6_pct : ℕ)
    (discount_cnt_4 : ℕ) (discount_amount_4 : ℕ)
    (discount_cnt_5 : ℕ) (discount_amount_5 : ℕ)
    (total_cost_no_discount : ℕ) (total_discount : ℕ) (total_spent : ℕ) :
    total_cds = 400 ∧
    pct_20 = 25 ∧ pct_15 = 30 ∧ pct_10 = 20 ∧
    bought_20_pct = 70 ∧ bought_15_pct = 40 ∧
    bought_10_pct = 80 ∧ bought_6_pct = 100 ∧
    discount_cnt_4 = 4 ∧ discount_amount_4 = 5 ∧
    discount_cnt_5 = 5 ∧ discount_amount_5 = 3 ∧
    total_cost_no_discount - total_discount = total_spent ∧
    total_spent = 3119 := by
  sorry

end NUMINAMATH_GPT_prince_cd_total_spent_l201_20139


namespace NUMINAMATH_GPT_systematic_sampling_method_l201_20129

-- Defining the conditions of the problem as lean definitions
def sampling_interval_is_fixed (interval : ℕ) : Prop :=
  interval = 10

def production_line_uniformly_flowing : Prop :=
  true  -- Assumption

-- The main theorem formulation
theorem systematic_sampling_method :
  ∀ (interval : ℕ), sampling_interval_is_fixed interval → production_line_uniformly_flowing →
  (interval = 10 → true) :=
by {
  sorry
}

end NUMINAMATH_GPT_systematic_sampling_method_l201_20129


namespace NUMINAMATH_GPT_units_digit_of_sum_is_4_l201_20193

-- Definitions and conditions based on problem
def base_8_add (a b : List Nat) : List Nat :=
    sorry -- Function to perform addition in base 8, returning result as a list of digits

def units_digit (a : List Nat) : Nat :=
    a.headD 0  -- Function to get the units digit of the result

-- The list representation for the digits of 65 base 8 and 37 base 8
def sixty_five_base8 := [6, 5]
def thirty_seven_base8 := [3, 7]

-- The theorem that asserts the final result
theorem units_digit_of_sum_is_4 : units_digit (base_8_add sixty_five_base8 thirty_seven_base8) = 4 :=
    sorry

end NUMINAMATH_GPT_units_digit_of_sum_is_4_l201_20193


namespace NUMINAMATH_GPT_total_bushels_needed_l201_20176

def cows := 5
def sheep := 4
def chickens := 8
def pigs := 6
def horses := 2

def cow_bushels := 3.5
def sheep_bushels := 1.75
def chicken_bushels := 1.25
def pig_bushels := 4.5
def horse_bushels := 5.75

theorem total_bushels_needed
  (cows : ℕ) (sheep : ℕ) (chickens : ℕ) (pigs : ℕ) (horses : ℕ)
  (cow_bushels: ℝ) (sheep_bushels: ℝ) (chicken_bushels: ℝ) (pig_bushels: ℝ) (horse_bushels: ℝ) :
  cows * cow_bushels + sheep * sheep_bushels + chickens * chicken_bushels + pigs * pig_bushels + horses * horse_bushels = 73 :=
by
  -- Skipping the proof
  sorry

end NUMINAMATH_GPT_total_bushels_needed_l201_20176


namespace NUMINAMATH_GPT_polynomial_coeff_sum_l201_20160

theorem polynomial_coeff_sum :
  let p1 : Polynomial ℝ := Polynomial.C 4 * Polynomial.X ^ 2 - Polynomial.C 6 * Polynomial.X + Polynomial.C 5
  let p2 : Polynomial ℝ := Polynomial.C 8 - Polynomial.C 3 * Polynomial.X
  let product : Polynomial ℝ := p1 * p2
  let a : ℝ := - (product.coeff 3)
  let b : ℝ := (product.coeff 2)
  let c : ℝ := - (product.coeff 1)
  let d : ℝ := (product.coeff 0)
  8 * a + 4 * b + 2 * c + d = 18 := sorry

end NUMINAMATH_GPT_polynomial_coeff_sum_l201_20160


namespace NUMINAMATH_GPT_Kara_books_proof_l201_20131

-- Let's define the conditions and the proof statement in Lean 4

def Candice_books : ℕ := 18
def Amanda_books := Candice_books / 3
def Kara_books := Amanda_books / 2

theorem Kara_books_proof : Kara_books = 3 := by
  -- setting up the conditions based on the given problem.
  have Amanda_books_correct : Amanda_books = 6 := by
    exact Nat.div_eq_of_eq_mul_right (Nat.zero_lt_succ 2) (rfl) -- 18 / 3 = 6

  have Kara_books_correct : Kara_books = 3 := by
    exact Nat.div_eq_of_eq_mul_right (Nat.zero_lt_succ 1) Amanda_books_correct -- 6 / 2 = 3

  exact Kara_books_correct

end NUMINAMATH_GPT_Kara_books_proof_l201_20131


namespace NUMINAMATH_GPT_find_integer_n_cos_l201_20124

theorem find_integer_n_cos : ∃ n : ℤ, 0 ≤ n ∧ n ≤ 180 ∧ (Real.cos (n * Real.pi / 180) = Real.cos (1124 * Real.pi / 180)) ∧ n = 44 := by
  sorry

end NUMINAMATH_GPT_find_integer_n_cos_l201_20124


namespace NUMINAMATH_GPT_limit_for_regular_pay_l201_20109

theorem limit_for_regular_pay 
  (x : ℕ) 
  (regular_pay_rate : ℕ := 3) 
  (overtime_pay_rate : ℕ := 6) 
  (total_pay : ℕ := 186) 
  (overtime_hours : ℕ := 11) 
  (H : 3 * x + (6 * 11) = 186) 
  :
  x = 40 :=
sorry

end NUMINAMATH_GPT_limit_for_regular_pay_l201_20109


namespace NUMINAMATH_GPT_wendy_lost_lives_l201_20144

theorem wendy_lost_lives (L : ℕ) (h1 : 10 - L + 37 = 41) : L = 6 :=
by
  sorry

end NUMINAMATH_GPT_wendy_lost_lives_l201_20144


namespace NUMINAMATH_GPT_minimum_value_of_expression_l201_20150

theorem minimum_value_of_expression {a c : ℝ} (h_pos : a > 0)
  (h_range : ∀ x, a * x ^ 2 - 4 * x + c ≥ 1) :
  ∃ a c, a > 0 ∧ (∀ x, a * x ^ 2 - 4 * x + c ≥ 1) ∧ (∃ a, a > 0 ∧ ∃ c, c - 1 = 4 / a ∧ (a / 4 + 9 / a = 3)) :=
by sorry

end NUMINAMATH_GPT_minimum_value_of_expression_l201_20150


namespace NUMINAMATH_GPT_trig_identity_and_perimeter_l201_20167

theorem trig_identity_and_perimeter
  (a b c : ℝ) (A B C : ℝ)
  (h1 : Real.sin C * Real.sin (A - B) = Real.sin B * Real.sin (C - A))
  (a_val : a = 5)
  (cos_A : Real.cos A = 25 / 31) :
  (2 * a^2 = b^2 + c^2) ∧ (a + b + c = 14) :=
by
  sorry

end NUMINAMATH_GPT_trig_identity_and_perimeter_l201_20167


namespace NUMINAMATH_GPT_inequality_solution_l201_20186

theorem inequality_solution (x : ℝ) : (2 * x - 1) / 3 > (3 * x - 2) / 2 - 1 → x < 2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_inequality_solution_l201_20186


namespace NUMINAMATH_GPT_A_finishes_remaining_work_in_6_days_l201_20162

-- Definitions for conditions
def A_workdays : ℕ := 18
def B_workdays : ℕ := 15
def B_worked_days : ℕ := 10

-- Proof problem statement
theorem A_finishes_remaining_work_in_6_days (A_workdays B_workdays B_worked_days : ℕ) :
  let rate_A := 1 / A_workdays
  let rate_B := 1 / B_workdays
  let work_done_by_B := B_worked_days * rate_B
  let remaining_work := 1 - work_done_by_B
  let days_A_needs := remaining_work / rate_A
  days_A_needs = 6 :=
by
  sorry

end NUMINAMATH_GPT_A_finishes_remaining_work_in_6_days_l201_20162


namespace NUMINAMATH_GPT_brenda_age_correct_l201_20166

open Nat

noncomputable def brenda_age_proof : Prop :=
  ∃ (A B J : ℚ), 
  (A = 4 * B) ∧ 
  (J = B + 8) ∧ 
  (A = J) ∧ 
  (B = 8 / 3)

theorem brenda_age_correct : brenda_age_proof := 
  sorry

end NUMINAMATH_GPT_brenda_age_correct_l201_20166


namespace NUMINAMATH_GPT_hitting_probability_l201_20156

theorem hitting_probability (P_miss : ℝ) (P_6 P_7 P_8 P_9 P_10 : ℝ) :
  P_miss = 0.2 →
  P_6 = 0.1 →
  P_7 = 0.2 →
  P_8 = 0.3 →
  P_9 = 0.15 →
  P_10 = 0.05 →
  1 - P_miss = 0.8 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_hitting_probability_l201_20156


namespace NUMINAMATH_GPT_students_in_sample_l201_20171

theorem students_in_sample (T : ℕ) (S : ℕ) (F : ℕ) (J : ℕ) (se : ℕ)
  (h1 : J = 22 * T / 100)
  (h2 : S = 25 * T / 100)
  (h3 : se = 160)
  (h4 : F = S + 64)
  (h5 : ∀ x, x ∈ ({F, S, J, se} : Finset ℕ) → x ≤ T ∧  x ≥ 0):
  T = 800 :=
by
  have h6 : T = F + S + J + se := sorry
  sorry

end NUMINAMATH_GPT_students_in_sample_l201_20171


namespace NUMINAMATH_GPT_cost_option1_eq_cost_option2_eq_option1_final_cost_eq_option2_final_cost_eq_option1_more_cost_effective_optimal_cost_eq_l201_20125

section price_calculations

variables {x : ℕ} (hx : x > 20)

-- Definitions based on the problem statement.
def suit_price : ℕ := 400
def tie_price : ℕ := 80

def option1_cost (x : ℕ) : ℕ :=
  20 * suit_price + tie_price * (x - 20)

def option2_cost (x : ℕ) : ℕ :=
  (20 * suit_price + tie_price * x) * 9 / 10

def option1_final_cost := option1_cost 30
def option2_final_cost := option2_cost 30

def optimal_cost : ℕ := 20 * suit_price + tie_price * 10 * 9 / 10

-- Proof obligations
theorem cost_option1_eq : option1_cost x = 80 * x + 6400 :=
by sorry

theorem cost_option2_eq : option2_cost x = 72 * x + 7200 :=
by sorry

theorem option1_final_cost_eq : option1_final_cost = 8800 :=
by sorry

theorem option2_final_cost_eq : option2_final_cost = 9360 :=
by sorry

theorem option1_more_cost_effective : option1_final_cost < option2_final_cost :=
by sorry

theorem optimal_cost_eq : optimal_cost = 8720 :=
by sorry

end price_calculations

end NUMINAMATH_GPT_cost_option1_eq_cost_option2_eq_option1_final_cost_eq_option2_final_cost_eq_option1_more_cost_effective_optimal_cost_eq_l201_20125


namespace NUMINAMATH_GPT_negation_of_universal_proposition_l201_20140

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 0 → x^2 - x ≤ 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 - x > 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_universal_proposition_l201_20140


namespace NUMINAMATH_GPT_parallel_condition_l201_20169

theorem parallel_condition (a : ℝ) : (a = -2) ↔ (∀ x y : ℝ, ax + 2 * y = 0 → (-a / 2) = 1) :=
by
  sorry

end NUMINAMATH_GPT_parallel_condition_l201_20169


namespace NUMINAMATH_GPT_min_distinct_values_l201_20159

theorem min_distinct_values (n : ℕ) (mode_freq : ℕ) (total : ℕ)
  (h1 : total = 3000) (h2 : mode_freq = 15) :
  n = 215 :=
by
  sorry

end NUMINAMATH_GPT_min_distinct_values_l201_20159


namespace NUMINAMATH_GPT_range_of_k_l201_20142

theorem range_of_k (k : ℝ) 
  (h1 : ∀ x y : ℝ, (x^2 / (k-3) + y^2 / (2-k) = 1) → (k-3 < 0) ∧ (2-k > 0)) : 
  k < 2 := by
  sorry

end NUMINAMATH_GPT_range_of_k_l201_20142


namespace NUMINAMATH_GPT_pure_imaginary_solution_l201_20119

theorem pure_imaginary_solution (b : ℝ) (z : ℂ) 
  (H : z = (b + Complex.I) / (2 + Complex.I))
  (H_imaginary : z.im = z ∧ z.re = 0) :
  b = -1 / 2 := 
by 
  sorry

end NUMINAMATH_GPT_pure_imaginary_solution_l201_20119


namespace NUMINAMATH_GPT_sara_staircase_l201_20164

theorem sara_staircase (n : ℕ) (h : 2 * n * (n + 1) = 360) : n = 13 :=
sorry

end NUMINAMATH_GPT_sara_staircase_l201_20164


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l201_20108

theorem quadratic_inequality_solution
  (x : ℝ) 
  (h1 : ∀ x, x^2 + 2 * x - 3 > 0 ↔ x < -3 ∨ x > 1) :
  (2 * x^2 - 3 * x - 2 < 0) ↔ (-1 / 2 < x ∧ x < 2) :=
by {
  sorry
}

end NUMINAMATH_GPT_quadratic_inequality_solution_l201_20108


namespace NUMINAMATH_GPT_probability_of_at_least_one_solving_l201_20127

variable (P1 P2 : ℝ)

theorem probability_of_at_least_one_solving : 
  (1 - (1 - P1) * (1 - P2)) = P1 + P2 - P1 * P2 := 
sorry

end NUMINAMATH_GPT_probability_of_at_least_one_solving_l201_20127


namespace NUMINAMATH_GPT_meaningful_sqrt_range_l201_20173

theorem meaningful_sqrt_range (x : ℝ) (h : 1 - x ≥ 0) : x ≤ 1 :=
sorry

end NUMINAMATH_GPT_meaningful_sqrt_range_l201_20173


namespace NUMINAMATH_GPT_rabbit_carrot_count_l201_20113

theorem rabbit_carrot_count
  (r h : ℕ)
  (hr : r = h - 3)
  (eq_carrots : 4 * r = 5 * h) :
  4 * r = 36 :=
by
  sorry

end NUMINAMATH_GPT_rabbit_carrot_count_l201_20113


namespace NUMINAMATH_GPT_percentage_of_employees_in_manufacturing_l201_20172

theorem percentage_of_employees_in_manufacturing (d total_degrees : ℝ) (h1 : d = 144) (h2 : total_degrees = 360) :
    (d / total_degrees) * 100 = 40 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_employees_in_manufacturing_l201_20172


namespace NUMINAMATH_GPT_sine_beta_value_l201_20155

variable (α β : ℝ)
variable (h1 : 0 < α ∧ α < (π / 2))
variable (h2 : 0 < β ∧ β < (π / 2))
variable (h3 : Real.cos α = 4 / 5)
variable (h4 : Real.cos (α + β) = 3 / 5)

theorem sine_beta_value : Real.sin β = 7 / 25 :=
by
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_sine_beta_value_l201_20155


namespace NUMINAMATH_GPT_chocolate_bar_cost_l201_20182

theorem chocolate_bar_cost 
  (x : ℝ)  -- cost of each bar in dollars
  (total_bars : ℕ)  -- total number of bars in the box
  (sold_bars : ℕ)  -- number of bars sold
  (amount_made : ℝ)  -- amount made in dollars
  (h1 : total_bars = 9)  -- condition: total bars in the box is 9
  (h2 : sold_bars = total_bars - 3)  -- condition: Wendy sold all but 3 bars
  (h3 : amount_made = 18)  -- condition: Wendy made $18
  (h4 : amount_made = sold_bars * x)  -- condition: amount made from selling sold bars
  : x = 3 := 
sorry

end NUMINAMATH_GPT_chocolate_bar_cost_l201_20182


namespace NUMINAMATH_GPT_no_solution_range_of_a_l201_20100

noncomputable def range_of_a : Set ℝ := {a | ∀ x : ℝ, ¬(abs (x - 1) + abs (x - 2) ≤ a^2 + a + 1)}

theorem no_solution_range_of_a :
  range_of_a = {a | -1 < a ∧ a < 0} :=
by
  sorry

end NUMINAMATH_GPT_no_solution_range_of_a_l201_20100


namespace NUMINAMATH_GPT_smallest_k_for_inequality_l201_20199

theorem smallest_k_for_inequality : 
  ∃ k : ℕ,  k > 0 ∧ ( (k-10) ^ 5026 ≥ 2013 ^ 2013 ) ∧ 
  (∀ m : ℕ, m > 0 ∧ ((m-10) ^ 5026) ≥ 2013 ^ 2013 → m ≥ 55) :=
sorry

end NUMINAMATH_GPT_smallest_k_for_inequality_l201_20199


namespace NUMINAMATH_GPT_geometric_sequence_a10_a11_l201_20189

noncomputable def a (n : ℕ) : ℝ := sorry  -- define the geometric sequence {a_n}

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n m, a (n + m) = a n * q^m

variables (a : ℕ → ℝ) (q : ℝ)

-- Conditions given in the problem
axiom h1 : a 1 + a 5 = 5
axiom h2 : a 4 + a 5 = 15
axiom geom_seq : is_geometric_sequence a q

theorem geometric_sequence_a10_a11 : a 10 + a 11 = 135 :=
by {
  sorry
}

end NUMINAMATH_GPT_geometric_sequence_a10_a11_l201_20189


namespace NUMINAMATH_GPT_least_five_digit_congruent_to_7_mod_18_l201_20194

theorem least_five_digit_congruent_to_7_mod_18 :
  ∃ (n : ℕ), 10000 ≤ n ∧ n < 100000 ∧ n % 18 = 7 ∧ ∀ m : ℕ, 10000 ≤ m ∧ m < n → m % 18 ≠ 7 :=
by
  sorry

end NUMINAMATH_GPT_least_five_digit_congruent_to_7_mod_18_l201_20194


namespace NUMINAMATH_GPT_at_least_one_irrational_l201_20152

theorem at_least_one_irrational (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) 
  (h₃ : a^2 * b^2 * (a^2 * b^2 + 4) = 2 * (a^6 + b^6)) : 
  ¬ (∀ a b : ℚ, a ≠ 0 ∧ b ≠ 0 → a^2 * b^2 * (a^2 * b^2 + 4) = 2 * (a^6 + b^6)) :=
by sorry

end NUMINAMATH_GPT_at_least_one_irrational_l201_20152


namespace NUMINAMATH_GPT_find_f_5_l201_20104

theorem find_f_5 : 
  ∀ (f : ℝ → ℝ) (y : ℝ), 
  (∀ x, f x = 2 * x ^ 2 + y) ∧ f 2 = 60 -> f 5 = 102 :=
by
  sorry

end NUMINAMATH_GPT_find_f_5_l201_20104


namespace NUMINAMATH_GPT_angle_C_triangle_area_l201_20117

theorem angle_C 
  (a b c : ℝ) (A B C : ℝ)
  (h1 : a * Real.cos B + b * Real.cos A = -2 * c * Real.cos C) :
  C = 2 * Real.pi / 3 :=
sorry

theorem triangle_area 
  (a b c : ℝ) (C : ℝ)
  (h1 : a * Real.cos B + b * Real.cos A = -2 * c * Real.cos C)
  (h2 : c = Real.sqrt 7)
  (h3 : b = 2) :
  1 / 2 * a * b * Real.sin C = Real.sqrt 3 / 2 :=
sorry

end NUMINAMATH_GPT_angle_C_triangle_area_l201_20117


namespace NUMINAMATH_GPT_range_of_f_when_a_neg_2_is_0_to_4_and_bounded_range_of_a_if_f_bounded_by_4_l201_20179

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 + a * Real.sin x - Real.cos x ^ 2

theorem range_of_f_when_a_neg_2_is_0_to_4_and_bounded :
  (∀ x : ℝ, 0 ≤ f (-2) x ∧ f (-2) x ≤ 4) :=
sorry

theorem range_of_a_if_f_bounded_by_4 :
  (∀ x : ℝ, abs (f a x) ≤ 4) → (-2 ≤ a ∧ a ≤ 2) :=
sorry

end NUMINAMATH_GPT_range_of_f_when_a_neg_2_is_0_to_4_and_bounded_range_of_a_if_f_bounded_by_4_l201_20179


namespace NUMINAMATH_GPT_functional_expression_point_M_coordinates_l201_20165

variables (x y : ℝ) (k : ℝ)

-- Given conditions
def proportional_relation : Prop := y + 4 = k * (x - 3)
def initial_condition : Prop := (x = 1 → y = 0)
def point_M : Prop := ∃ m : ℝ, (m + 1, 2 * m) = (1, 0)

-- Proof of the functional expression
theorem functional_expression (h1 : proportional_relation x y k) (h2 : initial_condition x y) :
  ∃ k : ℝ, k = -2 ∧ y = -2 * x + 2 := 
sorry

-- Proof of the coordinates of point M
theorem point_M_coordinates (h : ∀ m : ℝ, (m + 1, 2 * m) = (1, 0)) :
  ∃ m : ℝ, m = 0 ∧ (m + 1, 2 * m) = (1, 0) := 
sorry

end NUMINAMATH_GPT_functional_expression_point_M_coordinates_l201_20165


namespace NUMINAMATH_GPT_each_friend_eats_six_slices_l201_20192

-- Definitions
def slices_per_loaf : ℕ := 15
def loaves_bought : ℕ := 4
def friends : ℕ := 10
def total_slices : ℕ := loaves_bought * slices_per_loaf
def slices_per_friend : ℕ := total_slices / friends

-- Theorem to prove
theorem each_friend_eats_six_slices (h1 : slices_per_loaf = 15) (h2 : loaves_bought = 4) (h3 : friends = 10) : slices_per_friend = 6 :=
by
  sorry

end NUMINAMATH_GPT_each_friend_eats_six_slices_l201_20192


namespace NUMINAMATH_GPT_chi_squared_test_expectation_correct_distribution_table_correct_l201_20126

-- Given data for the contingency table
def male_good := 52
def male_poor := 8
def female_good := 28
def female_poor := 12
def total := 100

-- Define the $\chi^2$ calculation
def chi_squared_value : ℚ :=
  (total * (male_good * female_poor - male_poor * female_good)^2) / 
  ((male_good + male_poor) * (female_good + female_poor) * (male_good + female_good) * (male_poor + female_poor))

-- The $\chi^2$ value to compare against for 99% confidence
def critical_value_99 : ℚ := 6.635

-- Prove that $\chi^2$ value is less than the critical value for 99% confidence
theorem chi_squared_test :
  chi_squared_value < critical_value_99 :=
by
  -- Sorry to skip the proof as instructed
  sorry

-- Probability data and expectations for successful shots
def prob_male_success : ℚ := 2 / 3
def prob_female_success : ℚ := 1 / 2

-- Probabilities of the number of successful shots
def prob_X_0 : ℚ := (1 - prob_male_success) ^ 2 * (1 - prob_female_success)
def prob_X_1 : ℚ := 2 * prob_male_success * (1 - prob_male_success) * (1 - prob_female_success) +
                    (1 - prob_male_success) ^ 2 * prob_female_success
def prob_X_2 : ℚ := prob_male_success ^ 2 * (1 - prob_female_success) +
                    2 * prob_male_success * (1 - prob_male_success) * prob_female_success
def prob_X_3 : ℚ := prob_male_success ^ 2 * prob_female_success

def expectation_X : ℚ :=
  0 * prob_X_0 + 
  1 * prob_X_1 + 
  2 * prob_X_2 + 
  3 * prob_X_3

-- The expected value of X
def expected_value_X : ℚ := 11 / 6

-- Prove the expected value is as calculated
theorem expectation_correct :
  expectation_X = expected_value_X :=
by
  -- Sorry to skip the proof as instructed
  sorry

-- Define the distribution table based on calculated probabilities
def distribution_table : List (ℚ × ℚ) :=
  [(0, prob_X_0), (1, prob_X_1), (2, prob_X_2), (3, prob_X_3)]

-- The correct distribution table
def correct_distribution_table : List (ℚ × ℚ) :=
  [(0, 1 / 18), (1, 5 / 18), (2, 4 / 9), (3, 2 / 9)]

-- Prove the distribution table is as calculated
theorem distribution_table_correct :
  distribution_table = correct_distribution_table :=
by
  -- Sorry to skip the proof as instructed
  sorry

end NUMINAMATH_GPT_chi_squared_test_expectation_correct_distribution_table_correct_l201_20126


namespace NUMINAMATH_GPT_MN_equal_l201_20107

def M : Set ℝ := {x | ∃ (m : ℤ), x = Real.sin ((2 * m - 3) * Real.pi / 6)}
def N : Set ℝ := {y | ∃ (n : ℤ), y = Real.cos (n * Real.pi / 3)}

theorem MN_equal : M = N := by
  sorry

end NUMINAMATH_GPT_MN_equal_l201_20107


namespace NUMINAMATH_GPT_sufficiency_of_inequality_l201_20138

theorem sufficiency_of_inequality (x : ℝ) (h : x > 5) : x^2 > 25 :=
sorry

end NUMINAMATH_GPT_sufficiency_of_inequality_l201_20138


namespace NUMINAMATH_GPT_perimeter_proof_l201_20153

noncomputable def perimeter (x : ℝ) : ℝ :=
  if x ≥ 0 ∧ x ≤ (Real.sqrt 3) / 3 then 3 * Real.sqrt 6 * x
  else if x > (Real.sqrt 3) / 3 ∧ x ≤ (2 * Real.sqrt 3) / 3 then 3 * Real.sqrt 2
  else if x > (2 * Real.sqrt 3) / 3 ∧ x ≤ Real.sqrt 3 then 3 * Real.sqrt 6 * (Real.sqrt 3 - x)
  else 0

theorem perimeter_proof (x : ℝ) (h₀ : 0 ≤ x) (h₁ : x ≤ Real.sqrt 3) :
  perimeter x = 
    if x ≤ (Real.sqrt 3) / 3 then 3 * Real.sqrt 6 * x
    else if x ≤ (2 * Real.sqrt 3) / 3 then 3 * Real.sqrt 2
    else 3 * Real.sqrt 6 * (Real.sqrt 3 - x) :=
by 
  sorry

end NUMINAMATH_GPT_perimeter_proof_l201_20153


namespace NUMINAMATH_GPT_distance_midpoint_parabola_y_axis_l201_20198

theorem distance_midpoint_parabola_y_axis (x1 x2 : ℝ) (y1 y2 : ℝ) 
  (hA : y1 ^ 2 = x1) (hB : y2 ^ 2 = x2) 
  (h_focus : ∀ {p : ℝ × ℝ}, p = (x1, y1) ∨ p = (x2, y2) → |p.1 - 1/4| = |p.1 + 1/4|)
  (h_dist : |x1 - 1/4| + |x2 - 1/4| = 3) :
  abs ((x1 + x2) / 2) = 5 / 4 :=
by sorry

end NUMINAMATH_GPT_distance_midpoint_parabola_y_axis_l201_20198


namespace NUMINAMATH_GPT_proof_problem_l201_20120

open Set

variable {U : Set ℕ} {A : Set ℕ} {B : Set ℕ}

def problem_statement (U A B : Set ℕ) : Prop :=
  ((U \ A) ∪ B) = {2, 3}

theorem proof_problem :
  problem_statement {0, 1, 2, 3} {0, 1, 2} {2, 3} :=
by
  unfold problem_statement
  simp
  sorry

end NUMINAMATH_GPT_proof_problem_l201_20120


namespace NUMINAMATH_GPT_cylinder_volume_calc_l201_20141

def cylinder_volume (r h : ℝ) (π : ℝ) : ℝ := π * r^2 * h

theorem cylinder_volume_calc :
    cylinder_volume 5 (5 + 3) 3.14 = 628 :=
by
  -- We set r = 5, h = 8 (since h = r + 3), and π = 3.14 to calculate the volume
  sorry

end NUMINAMATH_GPT_cylinder_volume_calc_l201_20141


namespace NUMINAMATH_GPT_total_yearly_car_leasing_cost_l201_20132

-- Define mileage per day
def mileage_per_day (day : String) : ℕ :=
  if day = "Monday" ∨ day = "Wednesday" ∨ day = "Friday" ∨ day = "Sunday" then 50
  else if day = "Tuesday" ∨ day = "Thursday" then 80
  else if day = "Saturday" then 120
  else 0

-- Define weekly mileage
def weekly_mileage : ℕ := 4 * 50 + 2 * 80 + 120

-- Define cost parameters
def cost_per_mile : ℕ := 1 / 10
def weekly_fee : ℕ := 100
def monthly_toll_parking_fees : ℕ := 50
def discount_every_5th_week : ℕ := 30
def number_of_weeks_in_year : ℕ := 52

-- Define total yearly cost
def total_cost_yearly : ℕ :=
  let total_weekly_cost := (weekly_mileage * cost_per_mile + weekly_fee)
  let total_yearly_cost := total_weekly_cost * number_of_weeks_in_year
  let total_discounts := (number_of_weeks_in_year / 5) * discount_every_5th_week
  let annual_cost_without_tolls := total_yearly_cost - total_discounts
  let total_toll_fees := monthly_toll_parking_fees * 12
  annual_cost_without_tolls + total_toll_fees

-- Define the main theorem
theorem total_yearly_car_leasing_cost : total_cost_yearly = 7996 := 
  by
    -- Proof omitted
    sorry

end NUMINAMATH_GPT_total_yearly_car_leasing_cost_l201_20132


namespace NUMINAMATH_GPT_behavior_on_neg_interval_l201_20112

variable (f : ℝ → ℝ)

-- condition 1: f is an odd function
def odd_function : Prop :=
  ∀ x, f (-x) = -f x

-- condition 2: f is increasing on [3, 7]
def increasing_3_7 : Prop :=
  ∀ x y, (3 ≤ x ∧ x < y ∧ y ≤ 7) → f x < f y

-- condition 3: minimum value of f on [3, 7] is 5
def minimum_3_7 : Prop :=
  ∃ a, 3 ≤ a ∧ a ≤ 7 ∧ f a = 5

-- Use the above conditions to prove the required property on [-7, -3].
theorem behavior_on_neg_interval 
  (h1 : odd_function f) 
  (h2 : increasing_3_7 f) 
  (h3 : minimum_3_7 f) : 
  (∀ x y, (-7 ≤ x ∧ x < y ∧ y ≤ -3) → f x < f y) 
  ∧ ∀ x, -7 ≤ x ∧ x ≤ -3 → f x ≤ -5 :=
sorry

end NUMINAMATH_GPT_behavior_on_neg_interval_l201_20112


namespace NUMINAMATH_GPT_part_a_part_b_l201_20123

-- Define the natural numbers m and n
variable (m n : Nat)

-- Condition: m * n is divisible by m + n
def divisible_condition : Prop :=
  ∃ (k : Nat), m * n = k * (m + n)

-- Define prime number
def is_prime (p : Nat) : Prop :=
  p > 1 ∧ ∀ d : Nat, d ∣ p → d = 1 ∨ d = p

-- Define n as the product of two distinct primes
def is_product_of_two_distinct_primes (n : Nat) : Prop :=
  ∃ (p₁ p₂ : Nat), is_prime p₁ ∧ is_prime p₂ ∧ p₁ ≠ p₂ ∧ n = p₁ * p₂

-- Problem (a): Prove that m is divisible by n when n is a prime number and m * n is divisible by m + n
theorem part_a (prime_n : is_prime n) (h : divisible_condition m n) : n ∣ m := sorry

-- Problem (b): Prove that m is not necessarily divisible by n when n is a product of two distinct prime numbers
theorem part_b (prod_of_primes_n : is_product_of_two_distinct_primes n) (h : divisible_condition m n) :
  ¬ (n ∣ m) := sorry

end NUMINAMATH_GPT_part_a_part_b_l201_20123


namespace NUMINAMATH_GPT_selection_plans_l201_20146

-- Definitions for the students
inductive Student
| A | B | C | D | E | F

open Student

-- Definitions for the subjects
inductive Subject
| Mathematics | Physics | Chemistry | Biology

open Subject

-- A function to count the number of valid selections such that A and B do not participate in Biology.
def countValidSelections : Nat :=
  let totalWays := Nat.factorial 6 / Nat.factorial 2 / Nat.factorial (6 - 4)
  let forbiddenWays := 2 * (Nat.factorial 5 / Nat.factorial 2 / Nat.factorial (5 - 3))
  totalWays - forbiddenWays

theorem selection_plans :
  countValidSelections = 240 :=
by
  sorry

end NUMINAMATH_GPT_selection_plans_l201_20146


namespace NUMINAMATH_GPT_find_number_l201_20190

def single_digit (n : ℕ) : Prop := n < 10
def greater_than_zero (n : ℕ) : Prop := n > 0
def less_than_two (n : ℕ) : Prop := n < 2

theorem find_number (n : ℕ) : 
  single_digit n ∧ greater_than_zero n ∧ less_than_two n → n = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l201_20190


namespace NUMINAMATH_GPT_line_segment_is_symmetric_l201_20114

def is_axial_symmetric (shape : Type) : Prop := sorry
def is_central_symmetric (shape : Type) : Prop := sorry

def equilateral_triangle : Type := sorry
def isosceles_triangle : Type := sorry
def parallelogram : Type := sorry
def line_segment : Type := sorry

theorem line_segment_is_symmetric : 
  is_axial_symmetric line_segment ∧ is_central_symmetric line_segment := 
by
  sorry

end NUMINAMATH_GPT_line_segment_is_symmetric_l201_20114


namespace NUMINAMATH_GPT_quadratic_distinct_roots_l201_20195

theorem quadratic_distinct_roots (p q : ℚ) :
  (∀ x : ℚ, x^2 + p * x + q = 0 ↔ x = 2 * p ∨ x = p + q) →
  (p = 2 / 3 ∧ q = -8 / 3) :=
by sorry

end NUMINAMATH_GPT_quadratic_distinct_roots_l201_20195


namespace NUMINAMATH_GPT_projectile_reaches_75_feet_l201_20136

def projectile_height (t : ℝ) : ℝ := -16 * t^2 + 80 * t

theorem projectile_reaches_75_feet :
  ∃ t : ℝ, projectile_height t = 75 ∧ t = 1.25 :=
by
  -- Skipping the proof as instructed
  sorry

end NUMINAMATH_GPT_projectile_reaches_75_feet_l201_20136


namespace NUMINAMATH_GPT_problem_statement_l201_20149

variables (x y : ℝ)

def p : Prop := x > 1 ∧ y > 1
def q : Prop := x + y > 2

theorem problem_statement : (p x y → q x y) ∧ ¬(q x y → p x y) := sorry

end NUMINAMATH_GPT_problem_statement_l201_20149


namespace NUMINAMATH_GPT_edward_initial_money_l201_20143

theorem edward_initial_money (cars qty : Nat) (car_cost race_track_cost left_money initial_money : ℝ) 
    (h1 : cars = 4) 
    (h2 : car_cost = 0.95) 
    (h3 : race_track_cost = 6.00)
    (h4 : left_money = 8.00)
    (h5 : initial_money = (cars * car_cost) + race_track_cost + left_money) :
  initial_money = 17.80 := sorry

end NUMINAMATH_GPT_edward_initial_money_l201_20143


namespace NUMINAMATH_GPT_units_digit_2008_pow_2008_l201_20168

theorem units_digit_2008_pow_2008 : (2008 ^ 2008) % 10 = 6 := 
by
  -- The units digits of powers of 8 repeat in a cycle: 8, 4, 2, 6
  -- 2008 mod 4 = 0 which implies it falls on the 4th position in the pattern cycle
  sorry

end NUMINAMATH_GPT_units_digit_2008_pow_2008_l201_20168


namespace NUMINAMATH_GPT_evaluate_T_l201_20158

def T (a b : ℤ) : ℤ := 4 * a - 7 * b

theorem evaluate_T : T 6 3 = 3 := by
  sorry

end NUMINAMATH_GPT_evaluate_T_l201_20158


namespace NUMINAMATH_GPT_tim_prank_combinations_l201_20181

def number_of_combinations : Nat :=
  let monday_choices := 1
  let tuesday_choices := 2
  let wednesday_choices := 6
  let thursday_choices := 5
  let friday_choices := 1
  monday_choices * tuesday_choices * wednesday_choices * thursday_choices * friday_choices

theorem tim_prank_combinations : number_of_combinations = 60 :=
by
  sorry

end NUMINAMATH_GPT_tim_prank_combinations_l201_20181


namespace NUMINAMATH_GPT_roller_coaster_people_l201_20180

def num_cars : ℕ := 7
def seats_per_car : ℕ := 2
def num_runs : ℕ := 6
def total_seats_per_run : ℕ := num_cars * seats_per_car
def total_people : ℕ := total_seats_per_run * num_runs

theorem roller_coaster_people:
  total_people = 84 := 
by
  sorry

end NUMINAMATH_GPT_roller_coaster_people_l201_20180


namespace NUMINAMATH_GPT_find_two_digit_number_l201_20121

theorem find_two_digit_number (N : ℕ) (a b c : ℕ) 
  (h_end_digits : N % 1000 = c + 10 * b + 100 * a)
  (hN2_end_digits : N^2 % 1000 = c + 10 * b + 100 * a)
  (h_nonzero : a ≠ 0) :
  10 * a + b = 24 := 
by
  sorry

end NUMINAMATH_GPT_find_two_digit_number_l201_20121


namespace NUMINAMATH_GPT_range_of_m_l201_20178

theorem range_of_m (m : ℝ) : (-1 : ℝ) ≤ m ∧ m ≤ 3 ∧ ∀ x y : ℝ, x - ((m^2) - 2 * m + 4) * y - 6 > 0 → (x, y) ≠ (-1, -1) := 
by sorry

end NUMINAMATH_GPT_range_of_m_l201_20178


namespace NUMINAMATH_GPT_kit_costs_more_l201_20151

-- Defining the individual prices of the filters and the kit price
def price_filter1 := 16.45
def price_filter2 := 14.05
def price_filter3 := 19.50
def kit_price := 87.50

-- Calculating the total price of the filters if bought individually
def total_individual_price := (2 * price_filter1) + (2 * price_filter2) + price_filter3

-- Calculate the amount saved
def amount_saved := total_individual_price - kit_price

-- The theorem to show the amount saved 
theorem kit_costs_more : amount_saved = -7.00 := by
  sorry

end NUMINAMATH_GPT_kit_costs_more_l201_20151


namespace NUMINAMATH_GPT_money_left_after_expenditures_l201_20137

variable (initial_amount : ℝ) (P : initial_amount = 15000)
variable (gas_percentage food_fraction clothing_fraction entertainment_percentage : ℝ) 
variable (H1 : gas_percentage = 0.35) (H2 : food_fraction = 0.2) (H3 : clothing_fraction = 0.25) (H4 : entertainment_percentage = 0.15)

theorem money_left_after_expenditures
  (money_left : ℝ):
  money_left = initial_amount * (1 - gas_percentage) *
                (1 - food_fraction) * 
                (1 - clothing_fraction) * 
                (1 - entertainment_percentage) → 
  money_left = 4972.50 :=
by
  sorry

end NUMINAMATH_GPT_money_left_after_expenditures_l201_20137


namespace NUMINAMATH_GPT_average_temperature_MTWT_l201_20145

theorem average_temperature_MTWT (T_TWTF : ℝ) (T_M : ℝ) (T_F : ℝ) (T_MTWT : ℝ) :
    T_TWTF = 40 →
    T_M = 42 →
    T_F = 10 →
    T_MTWT = ((4 * T_TWTF - T_F + T_M) / 4) →
    T_MTWT = 48 := 
by
  intros hT_TWTF hT_M hT_F hT_MTWT
  rw [hT_TWTF, hT_M, hT_F] at hT_MTWT
  norm_num at hT_MTWT
  exact hT_MTWT

end NUMINAMATH_GPT_average_temperature_MTWT_l201_20145


namespace NUMINAMATH_GPT_negation_of_p_range_of_m_if_p_false_l201_20111

open Real

noncomputable def neg_p (m : ℝ) : Prop :=
∀ x : ℝ, x^2 - m*x - m > 0

theorem negation_of_p (m : ℝ) :
  (¬ ∃ x : ℝ, x^2 - m*x - m ≤ 0) ↔ neg_p m := 
by sorry

theorem range_of_m_if_p_false : 
  (∀ m : ℝ, neg_p m → (-4 < m ∧ m < 0)) :=
by sorry

end NUMINAMATH_GPT_negation_of_p_range_of_m_if_p_false_l201_20111


namespace NUMINAMATH_GPT_supplementary_angles_difference_l201_20174

theorem supplementary_angles_difference 
  (x : ℝ) 
  (h1 : 5 * x + 3 * x = 180) 
  (h2 : 0 < x) : 
  abs (5 * x - 3 * x) = 45 :=
by sorry

end NUMINAMATH_GPT_supplementary_angles_difference_l201_20174


namespace NUMINAMATH_GPT_distance_calculation_l201_20148

-- Define the given constants
def time_minutes : ℕ := 30
def average_speed : ℕ := 1
def seconds_per_minute : ℕ := 60

-- Define the total time in seconds
def time_seconds : ℕ := time_minutes * seconds_per_minute

-- The proof goal: that the distance covered is 1800 meters
theorem distance_calculation :
  time_seconds * average_speed = 1800 := by
  -- Calculation steps (using axioms and known values)
  sorry

end NUMINAMATH_GPT_distance_calculation_l201_20148


namespace NUMINAMATH_GPT_fraction_sum_l201_20101

theorem fraction_sum (x a b : ℕ) (h1 : x = 36 / 99) (h2 : a = 4) (h3 : b = 11) (h4 : Nat.gcd a b = 1) : a + b = 15 :=
by
  sorry

end NUMINAMATH_GPT_fraction_sum_l201_20101


namespace NUMINAMATH_GPT_sin_double_angle_sin_multiple_angle_l201_20163

-- Prove that |sin(2x)| <= 2|sin(x)| for any value of x
theorem sin_double_angle (x : ℝ) : |Real.sin (2 * x)| ≤ 2 * |Real.sin x| := 
by sorry

-- Prove that |sin(nx)| <= n|sin(x)| for any positive integer n and any value of x
theorem sin_multiple_angle (n : ℕ) (x : ℝ) (h : 0 < n) : |Real.sin (n * x)| ≤ n * |Real.sin x| :=
by sorry

end NUMINAMATH_GPT_sin_double_angle_sin_multiple_angle_l201_20163


namespace NUMINAMATH_GPT_lemons_minus_pears_l201_20102

theorem lemons_minus_pears
  (apples : ℕ)
  (pears : ℕ)
  (tangerines : ℕ)
  (lemons : ℕ)
  (watermelons : ℕ)
  (h1 : apples = 8)
  (h2 : pears = 5)
  (h3 : tangerines = 12)
  (h4 : lemons = 17)
  (h5 : watermelons = 10) :
  lemons - pears = 12 := 
sorry

end NUMINAMATH_GPT_lemons_minus_pears_l201_20102


namespace NUMINAMATH_GPT_maximum_bunnies_drum_l201_20105

-- Define the conditions as provided in the problem
def drumsticks := ℕ -- Natural number type for simplicity
def drum := ℕ -- Natural number type for simplicity

structure Bunny :=
(drum_size : drum)
(stick_length : drumsticks)

def max_drumming_bunnies (bunnies : List Bunny) : ℕ := 
  -- Actual implementation to find the maximum number of drumming bunnies
  sorry

theorem maximum_bunnies_drum (bunnies : List Bunny) (h_size : bunnies.length = 7) : max_drumming_bunnies bunnies = 6 :=
by
  -- Proof of the theorem
  sorry

end NUMINAMATH_GPT_maximum_bunnies_drum_l201_20105


namespace NUMINAMATH_GPT_gp_values_l201_20106

theorem gp_values (p : ℝ) (hp : 0 < p) :
  let a := -p - 12
  let b := 2 * Real.sqrt p
  let c := p - 5
  (b / a = c / b) ↔ p = 4 :=
by
  sorry

end NUMINAMATH_GPT_gp_values_l201_20106


namespace NUMINAMATH_GPT_single_line_points_l201_20118

theorem single_line_points (S : ℝ) (h1 : 6 * S + 4 * (8 * S) = 38000) : S = 1000 :=
by
  sorry

end NUMINAMATH_GPT_single_line_points_l201_20118


namespace NUMINAMATH_GPT_cars_on_river_road_l201_20147

theorem cars_on_river_road (B C : ℕ) (h1 : B = C - 60) (h2 : B * 13 = C) : C = 65 :=
sorry

end NUMINAMATH_GPT_cars_on_river_road_l201_20147


namespace NUMINAMATH_GPT_average_bull_weight_l201_20170

def ratioA : ℚ := 7 / 28  -- Ratio of cows to total cattle in section A
def ratioB : ℚ := 5 / 20  -- Ratio of cows to total cattle in section B
def ratioC : ℚ := 3 / 12  -- Ratio of cows to total cattle in section C

def total_cattle : ℕ := 1220  -- Total cattle on the farm
def total_bull_weight : ℚ := 200000  -- Total weight of bulls in kg

theorem average_bull_weight :
  ratioA = 7 / 28 ∧
  ratioB = 5 / 20 ∧
  ratioC = 3 / 12 ∧
  total_cattle = 1220 ∧
  total_bull_weight = 200000 →
  ∃ avg_weight : ℚ, avg_weight = 218.579 :=
sorry

end NUMINAMATH_GPT_average_bull_weight_l201_20170


namespace NUMINAMATH_GPT_find_k_l201_20184

theorem find_k : 
  (∃ y, -3 * (-15) + y = k ∧ 0.3 * (-15) + y = 10) → k = 59.5 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l201_20184


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l201_20196

theorem sufficient_but_not_necessary (x : ℝ) :
  (x > 1 → x > 0) ∧ ¬ (x > 0 → x > 1) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l201_20196


namespace NUMINAMATH_GPT_b_in_terms_of_a_axis_of_symmetry_and_a_value_range_of_a_l201_20116

-- Definitions based on problem conditions
def parabola (a b x : ℝ) : ℝ := a * x^2 + b * x
def passes_through_A (a b : ℝ) : Prop := parabola a b 3 = 3
def points_on_parabola (a b x1 x2 : ℝ) : Prop := x1 < x2 ∧ x1 + x2 = 2
def equal_y_values (a b x1 x2 : ℝ) : Prop := parabola a b x1 = parabola a b x2
def less_than_y_values (a b x1 x2 : ℝ) : Prop := parabola a b x1 < parabola a b x2

-- 1) Express b in terms of a
theorem b_in_terms_of_a (a : ℝ) (h : passes_through_A a (1 - 3 * a)) : True := sorry

-- 2) Axis of symmetry and the value of a when y1 = y2
theorem axis_of_symmetry_and_a_value (a : ℝ) (x1 x2 : ℝ) 
    (h1 : points_on_parabola a (1 - 3 * a) x1 x2)
    (h2 : equal_y_values a (1 - 3 * a) x1 x2) 
    : a = 1 ∧ -1 / 2 * (1 - 3 * a) / a = 1 := sorry

-- 3) Range of values for a when y1 < y2
theorem range_of_a (a x1 x2 : ℝ) 
    (h1 : points_on_parabola a (1 - 3 * a) x1 x2)
    (h2 : less_than_y_values a (1 - 3 * a) x1 x2) 
    (h3 : a ≠ 0) : 0 < a ∧ a < 1 := sorry

end NUMINAMATH_GPT_b_in_terms_of_a_axis_of_symmetry_and_a_value_range_of_a_l201_20116


namespace NUMINAMATH_GPT_triangle_area_l201_20187

theorem triangle_area (a b c : ℝ) (h1 : a = 9) (h2 : b = 40) (h3 : c = 41) (h4 : c^2 = a^2 + b^2) :
  (1 / 2) * a * b = 180 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_l201_20187


namespace NUMINAMATH_GPT_values_of_cos_0_45_l201_20183

-- Define the interval and the condition for the cos function
def in_interval (x : ℝ) : Prop := 0 ≤ x ∧ x < 360
def cos_condition (x : ℝ) : Prop := Real.cos x = 0.45

-- Final theorem statement
theorem values_of_cos_0_45 : 
  ∃ (n : ℕ), n = 2 ∧ ∀ (x : ℝ), in_interval x ∧ cos_condition x ↔ x = 1 ∨ x = 2 := 
sorry

end NUMINAMATH_GPT_values_of_cos_0_45_l201_20183


namespace NUMINAMATH_GPT_years_ago_twice_age_l201_20134

variables (H J x : ℕ)

def henry_age : ℕ := 20
def jill_age : ℕ := 13

axiom age_sum : H + J = 33
axiom age_difference : H - x = 2 * (J - x)

theorem years_ago_twice_age (H := henry_age) (J := jill_age) : x = 6 :=
by sorry

end NUMINAMATH_GPT_years_ago_twice_age_l201_20134


namespace NUMINAMATH_GPT_m_le_n_l201_20133

theorem m_le_n (k m n : ℕ) (hk_pos : 0 < k) (hm_pos : 0 < m) (hn_pos : 0 < n) (h : m^2 + n = k^2 + k) : m ≤ n := 
sorry

end NUMINAMATH_GPT_m_le_n_l201_20133


namespace NUMINAMATH_GPT_project_presentation_periods_l201_20177

def students : ℕ := 32
def period_length : ℕ := 40
def presentation_time_per_student : ℕ := 5

theorem project_presentation_periods : 
  (students * presentation_time_per_student) / period_length = 4 := by
  sorry

end NUMINAMATH_GPT_project_presentation_periods_l201_20177


namespace NUMINAMATH_GPT_problem_solution_l201_20135

universe u

variable (U : Set Nat) (M : Set Nat)
variable (complement_U_M : Set Nat)

axiom U_def : U = {1, 2, 3, 4, 5}
axiom complement_U_M_def : complement_U_M = {1, 3}
axiom M_def : M = U \ complement_U_M

theorem problem_solution : 2 ∈ M := by
  sorry

end NUMINAMATH_GPT_problem_solution_l201_20135


namespace NUMINAMATH_GPT_min_value_fraction_sum_l201_20161

theorem min_value_fraction_sum (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 1) : 
  ∃ (x y : ℝ), x = 2/5 ∧ y = 3/5 ∧ (∃ (k : ℝ), k = 4/x + 9/y ∧ k = 25) :=
by
  sorry

end NUMINAMATH_GPT_min_value_fraction_sum_l201_20161


namespace NUMINAMATH_GPT_total_tweets_l201_20110

-- Conditions and Definitions
def tweets_happy_per_minute := 18
def tweets_hungry_per_minute := 4
def tweets_reflection_per_minute := 45
def minutes_each_period := 20

-- Proof Problem Statement
theorem total_tweets : 
  (minutes_each_period * tweets_happy_per_minute) + 
  (minutes_each_period * tweets_hungry_per_minute) + 
  (minutes_each_period * tweets_reflection_per_minute) = 1340 :=
by
  sorry

end NUMINAMATH_GPT_total_tweets_l201_20110


namespace NUMINAMATH_GPT_A_alone_completes_one_work_in_32_days_l201_20103

def amount_of_work_per_day_by_B : ℝ := sorry
def amount_of_work_per_day_by_A : ℝ := 3 * amount_of_work_per_day_by_B
def total_work : ℝ := (amount_of_work_per_day_by_A + amount_of_work_per_day_by_B) * 24

theorem A_alone_completes_one_work_in_32_days :
  total_work = amount_of_work_per_day_by_A * 32 :=
by
  sorry

end NUMINAMATH_GPT_A_alone_completes_one_work_in_32_days_l201_20103


namespace NUMINAMATH_GPT_inequalities_no_solution_l201_20128

theorem inequalities_no_solution (x n : ℝ) (h1 : x ≤ 1) (h2 : x ≥ n) : n > 1 :=
sorry

end NUMINAMATH_GPT_inequalities_no_solution_l201_20128


namespace NUMINAMATH_GPT_find_cos_F1PF2_l201_20130

noncomputable def cos_angle_P_F1_F2 : ℝ :=
  let F1 := (-(4:ℝ), 0)
  let F2 := ((4:ℝ), 0)
  let a := (5:ℝ)
  let b := (3:ℝ)
  let P : ℝ × ℝ := sorry -- P is a point on the ellipse
  let area_triangle : ℝ := 3 * Real.sqrt 3
  let cos_angle : ℝ := 1 / 2
  cos_angle

def cos_angle_F1PF2_lemma (F1 F2 : ℝ × ℝ) (ellipse_Area : ℝ) (cos_angle : ℝ) : Prop :=
  cos_angle = 1/2

theorem find_cos_F1PF2 (a b : ℝ) (F1 F2 : ℝ × ℝ) (P : ℝ × ℝ) (Area_PF1F2 : ℝ) :
  (F1 = (-(4:ℝ), 0) ∧ F2 = ((4:ℝ), 0)) ∧ (Area_PF1F2 = 3 * Real.sqrt 3) ∧
  (P.1^2 / (a^2) + P.2^2 / (b^2) = 1) → cos_angle_F1PF2_lemma F1 F2 Area_PF1F2 (cos_angle_P_F1_F2)
:=
  sorry

end NUMINAMATH_GPT_find_cos_F1PF2_l201_20130


namespace NUMINAMATH_GPT_rational_iff_geometric_progression_l201_20188

theorem rational_iff_geometric_progression :
  (∃ x a b c : ℤ, a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ (x + a)*(x + c) = (x + b)^2) ↔
  (∃ x : ℚ, ∃ a b c : ℤ, a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ (x + (a : ℚ))*(x + (c : ℚ)) = (x + (b : ℚ))^2) :=
sorry

end NUMINAMATH_GPT_rational_iff_geometric_progression_l201_20188


namespace NUMINAMATH_GPT_june_walked_miles_l201_20154

theorem june_walked_miles
  (step_counter_reset : ℕ)
  (resets_per_year : ℕ)
  (final_steps : ℕ)
  (steps_per_mile : ℕ)
  (h1 : step_counter_reset = 100000)
  (h2 : resets_per_year = 52)
  (h3 : final_steps = 30000)
  (h4 : steps_per_mile = 2000) :
  (resets_per_year * step_counter_reset + final_steps) / steps_per_mile = 2615 := 
by 
  sorry

end NUMINAMATH_GPT_june_walked_miles_l201_20154


namespace NUMINAMATH_GPT_workers_production_l201_20122

theorem workers_production
    (x y : ℝ)
    (h1 : x + y = 72)
    (h2 : 1.15 * x + 1.25 * y = 86) :
    1.15 * x = 46 ∧ 1.25 * y = 40 :=
by {
  sorry
}

end NUMINAMATH_GPT_workers_production_l201_20122


namespace NUMINAMATH_GPT_prove_value_of_question_l201_20197

theorem prove_value_of_question :
  let a := 9548
  let b := 7314
  let c := 3362
  let value_of_question : ℕ := by 
    sorry -- Proof steps to show the computation.

  (a + b = value_of_question) ∧ (c + 13500 = value_of_question) :=
by {
  let a := 9548
  let b := 7314
  let c := 3362
  let sum_of_a_b := a + b
  let computed_question := sum_of_a_b - c
  sorry -- Proof steps to show sum_of_a_b and the final result.
}

end NUMINAMATH_GPT_prove_value_of_question_l201_20197


namespace NUMINAMATH_GPT_gift_cost_l201_20175

theorem gift_cost (C F : ℕ) (hF : F = 15) (h_eq : C / (F - 4) = C / F + 12) : C = 495 :=
by
  -- Using the conditions given, we need to show that C computes to 495.
  -- Details are skipped using sorry.
  sorry

end NUMINAMATH_GPT_gift_cost_l201_20175


namespace NUMINAMATH_GPT_bombardiers_shots_l201_20157

theorem bombardiers_shots (x y z : ℕ) :
  x + y = z + 26 →
  x + y + 38 = y + z →
  x + z = y + 24 →
  x = 25 ∧ y = 64 ∧ z = 63 := by
  sorry

end NUMINAMATH_GPT_bombardiers_shots_l201_20157


namespace NUMINAMATH_GPT_doris_weeks_to_meet_expenses_l201_20115

def doris_weekly_hours : Nat := 5 * 3 + 5 -- 5 weekdays (3 hours each) + 5 hours on Saturday
def doris_hourly_rate : Nat := 20 -- Doris earns $20 per hour
def doris_weekly_earnings : Nat := doris_weekly_hours * doris_hourly_rate -- The total earnings per week
def doris_monthly_expenses : Nat := 1200 -- Doris's monthly expense

theorem doris_weeks_to_meet_expenses : ∃ w : Nat, doris_weekly_earnings * w ≥ doris_monthly_expenses ∧ w = 3 :=
by
  sorry

end NUMINAMATH_GPT_doris_weeks_to_meet_expenses_l201_20115

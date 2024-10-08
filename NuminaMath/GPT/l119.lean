import Mathlib

namespace yeast_population_at_130pm_l119_119082

noncomputable def yeast_population (initial_population : ℕ) (time_increments : ℕ) (growth_factor : ℕ) : ℕ :=
  initial_population * growth_factor ^ time_increments

theorem yeast_population_at_130pm : yeast_population 30 3 3 = 810 :=
by
  sorry

end yeast_population_at_130pm_l119_119082


namespace derivative_at_one_l119_119946

variable (x : ℝ)

def f (x : ℝ) := x^2 - 2*x + 3

theorem derivative_at_one : deriv f 1 = 0 := 
by 
  sorry

end derivative_at_one_l119_119946


namespace distribution_ways_l119_119333

theorem distribution_ways (n_problems n_friends : ℕ) (h_problems : n_problems = 6) (h_friends : n_friends = 8) : (n_friends ^ n_problems) = 262144 :=
by
  rw [h_problems, h_friends]
  norm_num

end distribution_ways_l119_119333


namespace percentage_increase_proof_l119_119457

def breakfast_calories : ℕ := 500
def shakes_total_calories : ℕ := 3 * 300
def total_daily_calories : ℕ := 3275

noncomputable def percentage_increase_in_calories (P : ℝ) : Prop :=
  let lunch_calories := breakfast_calories * (1 + P / 100)
  let dinner_calories := 2 * lunch_calories
  breakfast_calories + lunch_calories + dinner_calories + shakes_total_calories = total_daily_calories

theorem percentage_increase_proof : percentage_increase_in_calories 125 :=
by
  sorry

end percentage_increase_proof_l119_119457


namespace complex_pure_imaginary_l119_119356

theorem complex_pure_imaginary (a : ℝ) : (↑a + Complex.I) / (1 - Complex.I) = 0 + b * Complex.I → a = 1 :=
by
  intro h
  -- Proof content here
  sorry

end complex_pure_imaginary_l119_119356


namespace max_period_of_function_l119_119453

theorem max_period_of_function (f : ℝ → ℝ) (h1 : ∀ x, f (1 + x) = f (1 - x)) (h2 : ∀ x, f (8 + x) = f (8 - x)) :
  ∃ T > 0, (∀ x, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T' ≥ 14) ∧ T = 14 :=
sorry

end max_period_of_function_l119_119453


namespace man_l119_119553

theorem man's_age_twice_son (S M Y : ℕ) (h1 : M = S + 26) (h2 : S = 24) (h3 : M + Y = 2 * (S + Y)) : Y = 2 :=
by
  sorry

end man_l119_119553


namespace min_value_inequality_equality_condition_l119_119164

theorem min_value_inequality (a b : ℝ) (ha : 1 < a) (hb : 1 < b) :
  (b^2 / (a - 1) + a^2 / (b - 1)) ≥ 8 :=
sorry

theorem equality_condition (a b : ℝ) (ha : 1 < a) (hb : 1 < b) :
  (b^2 / (a - 1) + a^2 / (b - 1) = 8) ↔ ((a = 2) ∧ (b = 2)) :=
sorry

end min_value_inequality_equality_condition_l119_119164


namespace inequality_solution_l119_119879

-- We define the problem
def interval_of_inequality : Set ℝ := { x : ℝ | (x + 1) * (2 - x) > 0 }

-- We define the expected solution set
def expected_solution_set : Set ℝ := { x : ℝ | -1 < x ∧ x < 2 }

-- The theorem to be proved
theorem inequality_solution :
  interval_of_inequality = expected_solution_set := by 
  sorry

end inequality_solution_l119_119879


namespace remainder_of_cake_l119_119885

theorem remainder_of_cake (John Emily : ℝ) (h1 : 0.60 ≤ John) (h2 : Emily = 0.50 * (1 - John)) :
  1 - John - Emily = 0.20 :=
by
  sorry

end remainder_of_cake_l119_119885


namespace fish_to_rice_value_l119_119522

variable (f l r : ℝ)

theorem fish_to_rice_value (h1 : 5 * f = 3 * l) (h2 : 2 * l = 7 * r) : f = 2.1 * r :=
by
  sorry

end fish_to_rice_value_l119_119522


namespace incorrect_gcd_statement_l119_119062

theorem incorrect_gcd_statement :
  ¬(gcd 85 357 = 34) ∧ (gcd 16 12 = 4) ∧ (gcd 78 36 = 6) ∧ (gcd 105 315 = 105) :=
by
  sorry

end incorrect_gcd_statement_l119_119062


namespace problem_ABCD_cos_l119_119895

/-- In convex quadrilateral ABCD, angle A = 2 * angle C, AB = 200, CD = 200, the perimeter of 
ABCD is 720, and AD ≠ BC. Find the floor of 1000 * cos A. -/
theorem problem_ABCD_cos (A C : ℝ) (AB CD AD BC : ℝ) (h1 : AB = 200)
  (h2 : CD = 200) (h3 : AD + BC = 320) (h4 : A = 2 * C)
  (h5 : AD ≠ BC) : ⌊1000 * Real.cos A⌋ = 233 := 
sorry

end problem_ABCD_cos_l119_119895


namespace train_length_l119_119568

theorem train_length (v_train_kmph : ℝ) (v_man_kmph : ℝ) (time_sec : ℝ) 
  (h1 : v_train_kmph = 25) 
  (h2 : v_man_kmph = 2) 
  (h3 : time_sec = 20) : 
  (150 : ℝ) = (v_train_kmph + v_man_kmph) * (1000 / 3600) * time_sec := 
by {
  -- sorry for the steps here
  sorry
}

end train_length_l119_119568


namespace minimal_distance_ln_x_x_minimal_distance_graphs_ex_ln_x_l119_119900

theorem minimal_distance_ln_x_x :
  ∀ (x : ℝ), x > 0 → ∃ (d : ℝ), d = |Real.log x - x| → d ≥ 0 :=
by sorry

theorem minimal_distance_graphs_ex_ln_x :
  ∀ (x : ℝ), x > 0 → ∀ (y : ℝ), ∃ (d : ℝ), y = d → d = 2 :=
by sorry

end minimal_distance_ln_x_x_minimal_distance_graphs_ex_ln_x_l119_119900


namespace not_possible_perimeter_72_l119_119380

variable (a b : ℕ)
variable (P : ℕ)

def valid_perimeter_range (a b : ℕ) : Set ℕ := 
  { P | ∃ x, 15 < x ∧ x < 35 ∧ P = a + b + x }

theorem not_possible_perimeter_72 :
  (a = 10) → (b = 25) → ¬ (72 ∈ valid_perimeter_range 10 25) := 
by
  sorry

end not_possible_perimeter_72_l119_119380


namespace most_stable_performance_l119_119396

structure Shooter :=
(average_score : ℝ)
(variance : ℝ)

def A := Shooter.mk 8.9 0.45
def B := Shooter.mk 8.9 0.42
def C := Shooter.mk 8.9 0.51

theorem most_stable_performance : 
  B.variance < A.variance ∧ B.variance < C.variance :=
by
  sorry

end most_stable_performance_l119_119396


namespace tan_neg_225_is_neg_1_l119_119555

def tan_neg_225_eq_neg_1 : Prop :=
  Real.tan (-225 * Real.pi / 180) = -1

theorem tan_neg_225_is_neg_1 : tan_neg_225_eq_neg_1 :=
  by
    sorry

end tan_neg_225_is_neg_1_l119_119555


namespace karl_total_income_is_53_l119_119832

noncomputable def compute_income (tshirt_price pant_price skirt_price sold_tshirts sold_pants sold_skirts sold_refurbished_tshirts: ℕ) : ℝ :=
  let tshirt_income := 2 * tshirt_price
  let pant_income := sold_pants * pant_price
  let skirt_income := sold_skirts * skirt_price
  let refurbished_tshirt_price := (tshirt_price : ℝ) / 2
  let refurbished_tshirt_income := sold_refurbished_tshirts * refurbished_tshirt_price
  tshirt_income + pant_income + skirt_income + refurbished_tshirt_income

theorem karl_total_income_is_53 : compute_income 5 4 6 2 1 4 6 = 53 := by
  sorry

end karl_total_income_is_53_l119_119832


namespace problem1_problem2_l119_119420

variable {a b : ℝ}

theorem problem1 (h : a > b) : a - 3 > b - 3 :=
by sorry

theorem problem2 (h : a > b) : -4 * a < -4 * b :=
by sorry

end problem1_problem2_l119_119420


namespace intersect_A_B_l119_119393

def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {x | -1 < x ∧ x ≤ 1}

theorem intersect_A_B : A ∩ B = {0, 1} :=
by
  sorry

end intersect_A_B_l119_119393


namespace virginia_more_years_l119_119676

variable {V A D x : ℕ}

theorem virginia_more_years (h1 : V + A + D = 75) (h2 : D = 34) (h3 : V = A + x) (h4 : V = D - x) : x = 9 :=
by
  sorry

end virginia_more_years_l119_119676


namespace parabola_addition_l119_119092

def f (a b c x : ℝ) : ℝ := a * x^2 - b * (x + 3) + c
def g (a b c x : ℝ) : ℝ := a * x^2 + b * (x - 4) + c

theorem parabola_addition (a b c x : ℝ) : 
  (f a b c x + g a b c x) = (2 * a * x^2 + 2 * c - 7 * b) :=
by
  sorry

end parabola_addition_l119_119092


namespace new_profit_percentage_l119_119685

theorem new_profit_percentage (P SP NSP NP : ℝ) 
  (h1 : SP = 1.10 * P) 
  (h2 : SP = 879.9999999999993) 
  (h3 : NP = 0.90 * P) 
  (h4 : NSP = SP + 56) : 
  (NSP - NP) / NP * 100 = 30 := 
by
  sorry

end new_profit_percentage_l119_119685


namespace secretary_longest_time_l119_119110

theorem secretary_longest_time (h_ratio : ∃ x : ℕ, ∃ y : ℕ, ∃ z : ℕ, y = 2 * x ∧ z = 3 * x ∧ (5 * x = 40)) :
  5 * x = 40 := sorry

end secretary_longest_time_l119_119110


namespace solve_a_b_powers_l119_119098

theorem solve_a_b_powers :
  ∃ a b : ℂ, (a + b = 1) ∧ 
             (a^2 + b^2 = 3) ∧ 
             (a^3 + b^3 = 4) ∧ 
             (a^4 + b^4 = 7) ∧ 
             (a^5 + b^5 = 11) ∧ 
             (a^10 + b^10 = 93) :=
sorry

end solve_a_b_powers_l119_119098


namespace ratio_of_terms_l119_119996

noncomputable def geometric_sum (a₁ q : ℝ) (n : ℕ) : ℝ := a₁ * (1 - q^n) / (1 - q)

theorem ratio_of_terms
  (a : ℕ → ℝ) (b : ℕ → ℝ)
  (S T : ℕ → ℝ)
  (h₀ : ∀ n : ℕ, S n = geometric_sum (a 1) (a 2) n)
  (h₁ : ∀ n : ℕ, T n = geometric_sum (b 1) (b 2) n)
  (h₂ : ∀ n : ℕ, n > 0 → S n / T n = (3 ^ n + 1) / 4) :
  a 3 / b 4 = 3 := 
sorry

end ratio_of_terms_l119_119996


namespace largest_natural_number_not_sum_of_two_composites_l119_119680

def is_composite (n : ℕ) : Prop :=
  2 ≤ n ∧ ∃ m : ℕ, 2 ≤ m ∧ m < n ∧ n % m = 0

def is_sum_of_two_composites (n : ℕ) : Prop :=
  ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b

theorem largest_natural_number_not_sum_of_two_composites :
  ∀ n : ℕ, (n < 12) → ¬ (is_sum_of_two_composites n) → n ≤ 11 := 
sorry

end largest_natural_number_not_sum_of_two_composites_l119_119680


namespace max_take_home_pay_at_5000_dollars_l119_119069

noncomputable def income_tax (x : ℕ) : ℕ :=
  if x ≤ 5000 then x * 5 / 100
  else 250 + 10 * ((x - 5000 / 1000) - 5) ^ 2

noncomputable def take_home_pay (y : ℕ) : ℕ :=
  y - income_tax y

theorem max_take_home_pay_at_5000_dollars : ∀ y : ℕ, take_home_pay y ≤ take_home_pay 5000 := by
  sorry

end max_take_home_pay_at_5000_dollars_l119_119069


namespace product_divisible_by_15_l119_119339

theorem product_divisible_by_15 (n : ℕ) (hn1 : n % 2 = 1) (hn2 : n > 0) :
  15 ∣ (n + 2) * (n + 4) * (n + 6) * (n + 8) * (n + 10) :=
sorry

end product_divisible_by_15_l119_119339


namespace find_n_values_l119_119096

theorem find_n_values (n : ℤ) (hn : ∃ x y : ℤ, x ≠ y ∧ x^2 - 6*x - 4*n^2 - 32*n = 0 ∧ y^2 - 6*y - 4*n^2 - 32*n = 0):
  n = 10 ∨ n = 0 ∨ n = -8 ∨ n = -18 := 
sorry

end find_n_values_l119_119096


namespace find_range_a_l119_119929

noncomputable def f (a x : ℝ) : ℝ := x^2 + (a^2 - 1) * x + (a - 2)

theorem find_range_a (a : ℝ) (h : ∃ x y : ℝ, x ≠ y ∧ f a x = 0 ∧ f a y = 0 ∧ x > 1 ∧ y < 1 ) :
  -2 < a ∧ a < 1 := sorry

end find_range_a_l119_119929


namespace total_people_in_class_l119_119203

-- Define the number of people based on their interests
def likes_both: Nat := 5
def only_baseball: Nat := 2
def only_football: Nat := 3
def likes_neither: Nat := 6

-- Define the total number of people in the class
def total_people := likes_both + only_baseball + only_football + likes_neither

-- Theorem statement
theorem total_people_in_class : total_people = 16 :=
by
  -- Proof is skipped
  sorry

end total_people_in_class_l119_119203


namespace volume_box_values_l119_119687

theorem volume_box_values :
  let V := (x + 3) * (x - 3) * (x^2 - 10*x + 25)
  ∃ (x_values : Finset ℕ),
    ∀ x ∈ x_values, V < 1000 ∧ x > 0 ∧ x_values.card = 3 :=
by
  sorry

end volume_box_values_l119_119687


namespace arithmetic_sequence_general_formula_bn_sequence_sum_l119_119814

/-- 
  In an arithmetic sequence {a_n}, a_2 = 5 and a_6 = 21. 
  Prove the general formula for the nth term a_n and the sum of the first n terms S_n. 
-/
theorem arithmetic_sequence_general_formula (a : ℕ → ℤ) (S : ℕ → ℤ) (h1 : a 2 = 5) (h2 : a 6 = 21) : 
  (∀ n, a n = 4 * n - 3) ∧ (∀ n, S n = n * (2 * n - 1)) := 
sorry

/--
  Given b_n = 2 / (S_n + 5 * n), prove the sum of the first n terms T_n for the sequence {b_n}.
-/
theorem bn_sequence_sum (a : ℕ → ℤ) (S : ℕ → ℤ) (b : ℕ → ℚ) (T : ℕ → ℚ) 
  (h1 : a 2 = 5) (h2 : a 6 = 21) 
  (ha : ∀ n, a n = 4 * n - 3) (hS : ∀ n, S n = n * (2 * n - 1)) 
  (hb : ∀ n, b n = 2 / (S n + 5 * n)) : 
  (∀ n, T n = 3 / 4 - 1 / (2 * (n + 1)) - 1 / (2 * (n + 2))) :=
sorry

end arithmetic_sequence_general_formula_bn_sequence_sum_l119_119814


namespace sum_of_reciprocals_l119_119190

noncomputable def reciprocal_sum (x y : ℝ) : ℝ :=
  (1 / x) + (1 / y)

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 40) (h2 : x * y = 375) :
  reciprocal_sum x y = 8 / 75 :=
by
  unfold reciprocal_sum
  -- Intermediate steps would go here, but we'll use sorry to denote the proof is omitted.
  sorry

end sum_of_reciprocals_l119_119190


namespace difference_in_interest_rates_l119_119064

-- Definitions
def Principal : ℝ := 2300
def Time : ℝ := 3
def ExtraInterest : ℝ := 69

-- The difference in rates
theorem difference_in_interest_rates (R dR : ℝ) (h : (Principal * (R + dR) * Time) / 100 =
    (Principal * R * Time) / 100 + ExtraInterest) : dR = 1 :=
  sorry

end difference_in_interest_rates_l119_119064


namespace find_angle_C_l119_119311

variable (a b c : ℝ)
variable (A B C : ℝ)
variable (triangle_ABC : Type)

-- Given conditions
axiom ten_a_cos_B_eq_three_b_cos_A : 10 * a * Real.cos B = 3 * b * Real.cos A
axiom cos_A_value : Real.cos A = 5 * Real.sqrt 26 / 26

-- Required to prove
theorem find_angle_C : C = 3 * Real.pi / 4 := by
  sorry

end find_angle_C_l119_119311


namespace find_possible_values_l119_119919
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def satisfies_conditions (a bc de fg : ℕ) : Prop :=
  (a % 2 = 0) ∧ (is_prime bc) ∧ (de % 5 = 0) ∧ (fg % 3 = 0) ∧
  (fg - de = de - bc) ∧ (de - bc = bc - a)

theorem find_possible_values :
  ∃ (debc1 debc2 : ℕ),
    (satisfies_conditions 6 (debc1 % 100) ((debc1 / 100) % 100) ((debc1 / 10000) % 100)) ∧
    (satisfies_conditions 6 (debc2 % 100) ((debc2 / 100) % 100) ((debc2 / 10000) % 100)) ∧
    (debc1 = 2013 ∨ debc1 = 4023) ∧
    (debc2 = 2013 ∨ debc2 = 4023) :=
  sorry

end find_possible_values_l119_119919


namespace propA_necessary_but_not_sufficient_l119_119980

variable {a : ℝ}

-- Proposition A: ∀ x ∈ ℝ, ax² + 2ax + 1 > 0
def propA (a : ℝ) : Prop :=
  ∀ x : ℝ, a * x^2 + 2 * a * x + 1 > 0

-- Proposition B: 0 < a < 1
def propB (a : ℝ) : Prop :=
  0 < a ∧ a < 1

-- Theorem statement: Proposition A is necessary but not sufficient for Proposition B
theorem propA_necessary_but_not_sufficient (a : ℝ) :
  (propB a → propA a) ∧
  (propA a → propB a → False) :=
by
  sorry

end propA_necessary_but_not_sufficient_l119_119980


namespace calculation_l119_119765

theorem calculation : 120 / 5 / 3 * 2 = 16 := by
  sorry

end calculation_l119_119765


namespace proof_equation_of_line_l119_119350
   
   -- Define the point P
   structure Point where
     x : ℝ
     y : ℝ
     
   -- Define conditions
   def passesThroughP (line : ℝ → ℝ → Prop) : Prop :=
     line 2 (-1)
     
   def interceptRelation (line : ℝ → ℝ → Prop) : Prop :=
     ∃ a : ℝ, a ≠ 0 ∧ (∀ x y, line x y ↔ (x / a + y / (2 * a) = 1))
   
   -- Define the line equation
   def line_equation (line : ℝ → ℝ → Prop) : Prop :=
     passesThroughP line ∧ interceptRelation line
     
   -- The final statement
   theorem proof_equation_of_line (line : ℝ → ℝ → Prop) :
     line_equation line →
     (∀ x y, line x y ↔ (2 * x + y = 3)) ∨ (∀ x y, line x y ↔ (x + 2 * y = 0)) :=
   by
     sorry
   
end proof_equation_of_line_l119_119350


namespace factor_expression_l119_119402

theorem factor_expression (x y : ℤ) : 231 * x^2 * y + 33 * x * y = 33 * x * y * (7 * x + 1) := by
  sorry

end factor_expression_l119_119402


namespace solution_of_fraction_l119_119251

theorem solution_of_fraction (x : ℝ) (h1 : x^2 - 9 = 0) (h2 : x + 3 ≠ 0) : x = 3 :=
by
  sorry

end solution_of_fraction_l119_119251


namespace difference_is_four_l119_119066

def chickens_in_coop := 14
def chickens_in_run := 2 * chickens_in_coop
def chickens_free_ranging := 52
def difference := 2 * chickens_in_run - chickens_free_ranging

theorem difference_is_four : difference = 4 := by
  sorry

end difference_is_four_l119_119066


namespace instantaneous_velocity_at_t4_l119_119651

def position (t : ℝ) : ℝ := t^2 - t + 2

theorem instantaneous_velocity_at_t4 : 
  (deriv position 4) = 7 := 
by
  sorry

end instantaneous_velocity_at_t4_l119_119651


namespace water_fraction_after_replacements_l119_119349

-- Initially given conditions
def radiator_capacity : ℚ := 20
def initial_water_fraction : ℚ := 1
def antifreeze_quarts : ℚ := 5
def replacements : ℕ := 5

-- Derived condition
def water_remain_fraction : ℚ := 3 / 4

-- Statement of the problem
theorem water_fraction_after_replacements :
  (water_remain_fraction ^ replacements) = 243 / 1024 :=
by
  -- Proof goes here
  sorry

end water_fraction_after_replacements_l119_119349


namespace residue_calculation_l119_119896

theorem residue_calculation :
  (196 * 18 - 21 * 9 + 5) % 18 = 14 := 
by 
  sorry

end residue_calculation_l119_119896


namespace slope_of_line_determined_by_any_two_solutions_l119_119617

theorem slope_of_line_determined_by_any_two_solutions 
  (x₁ y₁ x₂ y₂ : ℝ) 
  (h₁ : 4 / x₁ + 5 / y₁ = 0) 
  (h₂ : 4 / x₂ + 5 / y₂ = 0) 
  (h_distinct : x₁ ≠ x₂) : 
  (y₂ - y₁) / (x₂ - x₁) = -5 / 4 := 
sorry

end slope_of_line_determined_by_any_two_solutions_l119_119617


namespace find_pairs_l119_119468

theorem find_pairs (a b : ℤ) (ha : a ≥ 1) (hb : b ≥ 1)
  (h1 : (a^2 + b) % (b^2 - a) = 0) 
  (h2 : (b^2 + a) % (a^2 - b) = 0) :
  (a = 2 ∧ b = 2) ∨ (a = 3 ∧ b = 3) ∨ (a = 1 ∧ b = 2) ∨ 
  (a = 2 ∧ b = 1) ∨ (a = 2 ∧ b = 3) ∨ (a = 3 ∧ b = 2) := 
sorry

end find_pairs_l119_119468


namespace bears_on_each_shelf_l119_119642

theorem bears_on_each_shelf 
    (initial_bears : ℕ) (shipment_bears : ℕ) (shelves : ℕ)
    (h1 : initial_bears = 4) (h2 : shipment_bears = 10) (h3 : shelves = 2) :
    (initial_bears + shipment_bears) / shelves = 7 := by
  sorry

end bears_on_each_shelf_l119_119642


namespace final_score_is_80_l119_119776

def adam_final_score : ℕ :=
  let first_half := 8
  let second_half := 2
  let points_per_question := 8
  (first_half + second_half) * points_per_question

theorem final_score_is_80 : adam_final_score = 80 := by
  sorry

end final_score_is_80_l119_119776


namespace number_of_children_admitted_l119_119254

variable (children adults : ℕ)

def admission_fee_children : ℝ := 1.5
def admission_fee_adults  : ℝ := 4

def total_people : ℕ := 315
def total_fees   : ℝ := 810

theorem number_of_children_admitted :
  ∃ (C A : ℕ), C + A = total_people ∧ admission_fee_children * C + admission_fee_adults * A = total_fees ∧ C = 180 :=
by
  sorry

end number_of_children_admitted_l119_119254


namespace sum_of_cubes_l119_119820

theorem sum_of_cubes (a b t : ℝ) (h : a + b = t^2) : 2 * (a^3 + b^3) = (a * t)^2 + (b * t)^2 + (a * t - b * t)^2 :=
by
  sorry

end sum_of_cubes_l119_119820


namespace perfect_square_fraction_l119_119910

theorem perfect_square_fraction (n : ℤ) : 
  n < 30 ∧ ∃ k : ℤ, (n / (30 - n)) = k^2 → ∃ cnt : ℕ, cnt = 4 :=
  by
  sorry

end perfect_square_fraction_l119_119910


namespace star_computation_l119_119659

-- Define the operation ☆
def star (m n : Int) := m^2 - m * n + n

-- Define the main proof problem
theorem star_computation :
  star 3 4 = 1 ∧ star (-1) (star 2 (-3)) = 15 := 
by
  sorry

end star_computation_l119_119659


namespace largest_integer_n_l119_119149

-- Define the condition for existence of positive integers x, y, z that satisfy the given equation
def condition (n : ℕ) : Prop :=
  ∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ n^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 5*x + 5*y + 5*z - 10

-- State that the largest such integer n is 4
theorem largest_integer_n : ∀ (n : ℕ), condition n → n ≤ 4 :=
by {
  sorry
}

end largest_integer_n_l119_119149


namespace expression_evaluation_l119_119285

theorem expression_evaluation : (6 * 111) - (2 * 111) = 444 :=
by
  sorry

end expression_evaluation_l119_119285


namespace find_symmetric_sequence_l119_119755

noncomputable def symmetric_sequence (b : ℕ → ℤ) (n : ℕ) : Prop :=
  ∀ k, 1 ≤ k ∧ k ≤ n → b k = b (n - k + 1)

noncomputable def arithmetic_sequence (b : ℕ → ℤ) : Prop :=
  ∃ d, b 2 = b 1 + d ∧ b 3 = b 2 + d ∧ b 4 = b 3 + d

theorem find_symmetric_sequence :
  ∃ b : ℕ → ℤ, symmetric_sequence b 7 ∧ arithmetic_sequence b ∧ b 1 = 2 ∧ b 2 + b 4 = 16 ∧
  (b 1 = 2 ∧ b 2 = 5 ∧ b 3 = 8 ∧ b 4 = 11 ∧ b 5 = 8 ∧ b 6 = 5 ∧ b 7 = 2) :=
by {
  sorry
}

end find_symmetric_sequence_l119_119755


namespace polynomial_has_one_positive_real_solution_l119_119078

-- Define the polynomial
def f (x : ℝ) : ℝ := x ^ 10 + 4 * x ^ 9 + 7 * x ^ 8 + 2023 * x ^ 7 - 2024 * x ^ 6

-- The proof problem statement
theorem polynomial_has_one_positive_real_solution :
  ∃! x : ℝ, 0 < x ∧ f x = 0 := by
  sorry

end polynomial_has_one_positive_real_solution_l119_119078


namespace fraction_of_girls_is_one_half_l119_119160

def fraction_of_girls (total_students_jasper : ℕ) (ratio_jasper : ℕ × ℕ) (total_students_brookstone : ℕ) (ratio_brookstone : ℕ × ℕ) : ℚ :=
  let (boys_ratio_jasper, girls_ratio_jasper) := ratio_jasper
  let (boys_ratio_brookstone, girls_ratio_brookstone) := ratio_brookstone
  let girls_jasper := (total_students_jasper * girls_ratio_jasper) / (boys_ratio_jasper + girls_ratio_jasper)
  let girls_brookstone := (total_students_brookstone * girls_ratio_brookstone) / (boys_ratio_brookstone + girls_ratio_brookstone)
  let total_girls := girls_jasper + girls_brookstone
  let total_students := total_students_jasper + total_students_brookstone
  total_girls / total_students

theorem fraction_of_girls_is_one_half :
  fraction_of_girls 360 (7, 5) 240 (3, 5) = 1 / 2 :=
  sorry

end fraction_of_girls_is_one_half_l119_119160


namespace distance_between_trees_l119_119939

-- The conditions given
def trees_on_yard := 26
def yard_length := 500
def trees_at_ends := true

-- Theorem stating the proof
theorem distance_between_trees (h1 : trees_on_yard = 26) 
                               (h2 : yard_length = 500) 
                               (h3 : trees_at_ends = true) : 
  500 / (26 - 1) = 20 :=
by
  sorry

end distance_between_trees_l119_119939


namespace g_f_x_not_quadratic_l119_119721

open Real

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

theorem g_f_x_not_quadratic (h : ∃ x : ℝ, x - f (g x) = 0) :
  ∀ x : ℝ, g (f x) ≠ x^2 + x + 1 / 5 := sorry

end g_f_x_not_quadratic_l119_119721


namespace intersection_single_point_l119_119063

def A (x y : ℝ) := x^2 + y^2 = 4
def B (x y : ℝ) (r : ℝ) := (x - 3)^2 + (y - 4)^2 = r^2

theorem intersection_single_point (r : ℝ) (h : r > 0) :
  (∃! p : ℝ × ℝ, A p.1 p.2 ∧ B p.1 p.2 r) → r = 3 :=
by
  apply sorry -- Proof goes here

end intersection_single_point_l119_119063


namespace intersection_M_N_l119_119193

-- Definitions for sets M and N
def set_M : Set ℝ := {x | abs x < 1}
def set_N : Set ℝ := {x | x^2 <= x}

-- The theorem stating the intersection of M and N
theorem intersection_M_N : {x : ℝ | x ∈ set_M ∧ x ∈ set_N} = {x : ℝ | 0 <= x ∧ x < 1} :=
by
  sorry

end intersection_M_N_l119_119193


namespace inequality_range_l119_119601

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, |2 * x + 1| - |2 * x - 1| < a) → a > 2 :=
by
  sorry

end inequality_range_l119_119601


namespace remainder_of_8673_div_7_l119_119345

theorem remainder_of_8673_div_7 : 8673 % 7 = 3 :=
by
  -- outline structure, proof to be inserted
  sorry

end remainder_of_8673_div_7_l119_119345


namespace fouad_double_ahmed_l119_119537

/-- Proof that in 4 years, Fouad's age will be double of Ahmed's age given their current ages. -/
theorem fouad_double_ahmed (x : ℕ) (ahmed_age fouad_age : ℕ) (h1 : ahmed_age = 11) (h2 : fouad_age = 26) :
  (fouad_age + x = 2 * (ahmed_age + x)) → x = 4 :=
by
  -- This is the statement only, proof is omitted
  sorry

end fouad_double_ahmed_l119_119537


namespace product_odd_primes_mod_32_l119_119419

open Nat

theorem product_odd_primes_mod_32 : 
  let primes := [3, 5, 7, 11, 13] 
  let product := primes.foldl (· * ·) 1 
  product % 32 = 7 := 
by
  sorry

end product_odd_primes_mod_32_l119_119419


namespace union_complement_l119_119922

open Set

variable (U A B : Set ℕ)
variable (u_spec : U = {1, 2, 3, 4, 5})
variable (a_spec : A = {1, 2, 3})
variable (b_spec : B = {2, 4})

theorem union_complement (U A B : Set ℕ)
  (u_spec : U = {1, 2, 3, 4, 5})
  (a_spec : A = {1, 2, 3})
  (b_spec : B = {2, 4}) :
  A ∪ (U \ B) = {1, 2, 3, 5} := by
  sorry

end union_complement_l119_119922


namespace total_weight_proof_l119_119610

-- Definitions of the variables and conditions given in the problem
variable (M D C : ℕ)
variable (h1 : D + C = 60)  -- Daughter and grandchild together weigh 60 kg
variable (h2 : C = 1 / 5 * M)  -- Grandchild's weight is 1/5th of grandmother's weight
variable (h3 : D = 42)  -- Daughter's weight is 42 kg

-- The goal is to prove the total weight is 150 kg
theorem total_weight_proof (M D C : ℕ) (h1 : D + C = 60) (h2 : C = 1 / 5 * M) (h3 : D = 42) :
  M + D + C = 150 :=
by
  sorry

end total_weight_proof_l119_119610


namespace sum_of_numbers_l119_119105

open Function

theorem sum_of_numbers (a b c : ℕ) (h1 : a ≤ b) (h2 : b ≤ c) 
  (h3 : b = 8) 
  (h4 : (a + b + c) / 3 = a + 7) 
  (h5 : (a + b + c) / 3 = c - 20) : 
  a + b + c = 63 := 
by 
  sorry

end sum_of_numbers_l119_119105


namespace rob_final_value_in_euros_l119_119081

noncomputable def initial_value_in_usd : ℝ := 
  (7 * 0.25) + (3 * 0.10) + (5 * 0.05) + (12 * 0.01) + (3 * 0.50) + (2 * 1.00)

noncomputable def value_after_losing_coins : ℝ := 
  (6 * 0.25) + (2 * 0.10) + (4 * 0.05) + (11 * 0.01) + (2 * 0.50) + (1 * 1.00)

noncomputable def value_after_first_exchange : ℝ :=
  (6 * 0.25) + (4 * 0.10) + (1 * 0.05) + (11 * 0.01) + (2 * 0.50) + (1 * 1.00)

noncomputable def value_after_second_exchange : ℝ :=
  (7 * 0.25) + (6 * 0.10) + (1 * 0.05) + (11 * 0.01) + (1 * 0.50) + (1 * 1.00)

noncomputable def value_after_third_exchange : ℝ :=
  (7 * 0.25) + (6 * 0.10) + (1 * 0.05) + (61 * 0.01) + (1 * 0.50)

noncomputable def final_value_in_usd : ℝ := 
  (7 * 0.25) + (6 * 0.10) + (1 * 0.05) + (61 * 0.01) + (1 * 0.50)

noncomputable def exchange_rate_usd_to_eur : ℝ := 0.85

noncomputable def final_value_in_eur : ℝ :=
  final_value_in_usd * exchange_rate_usd_to_eur

theorem rob_final_value_in_euros : final_value_in_eur = 2.9835 := by
  sorry

end rob_final_value_in_euros_l119_119081


namespace sin_cos_sixth_power_l119_119993

theorem sin_cos_sixth_power (θ : ℝ) 
  (h : Real.sin (3 * θ) = 1 / 2) :
  Real.sin θ ^ 6 + Real.cos θ ^ 6 = 11 / 12 :=
  sorry

end sin_cos_sixth_power_l119_119993


namespace count_n_satisfies_conditions_l119_119516

theorem count_n_satisfies_conditions :
  ∃ (count : ℕ), count = 36 ∧ ∀ (n : ℕ), 
    0 < n ∧ n < 150 →
    ∃ (k : ℕ), 
    (n = 2*k + 2) ∧ 
    (k*(k + 2) % 4 = 0) :=
by
  sorry

end count_n_satisfies_conditions_l119_119516


namespace fill_parentheses_l119_119055

variable (a b : ℝ)

theorem fill_parentheses :
  1 - a^2 + 2 * a * b - b^2 = 1 - (a^2 - 2 * a * b + b^2) :=
by
  sorry

end fill_parentheses_l119_119055


namespace hyperbola_equation_l119_119878

-- Definitions based on problem conditions
def asymptotes (x y : ℝ) : Prop :=
  y = (1/3) * x ∨ y = -(1/3) * x

def focus (p : ℝ × ℝ) : Prop :=
  p = (Real.sqrt 10, 0)

-- The main statement to prove
theorem hyperbola_equation :
  (∃ p, focus p) ∧ (∀ (x y : ℝ), asymptotes x y) →
  (∀ x y : ℝ, (x^2 / 9 - y^2 = 1)) :=
sorry

end hyperbola_equation_l119_119878


namespace perimeters_ratio_l119_119428

noncomputable def ratio_perimeters_of_squares (area_ratio : ℚ) (ratio_area : area_ratio = 49 / 64) : ℚ :=
if h : area_ratio = 49 / 64 
then (7 / 8) 
else 0  -- This shouldn't happen since we enforce the condition

theorem perimeters_ratio (area_ratio : ℚ) (h : area_ratio = 49 / 64) : ratio_perimeters_of_squares area_ratio h = 7 / 8 :=
by {
  -- Proof goes here
  sorry
}

end perimeters_ratio_l119_119428


namespace integers_abs_no_greater_than_2_pos_div_by_3_less_than_10_non_neg_int_less_than_5_sum_eq_6_in_nat_expressing_sequence_l119_119335

-- Proof problem 1
theorem integers_abs_no_greater_than_2 :
    {n : ℤ | |n| ≤ 2} = {-2, -1, 0, 1, 2} :=
by {
  sorry
}

-- Proof problem 2
theorem pos_div_by_3_less_than_10 :
    {n : ℕ | n > 0 ∧ n % 3 = 0 ∧ n < 10} = {3, 6, 9} :=
by {
  sorry
}

-- Proof problem 3
theorem non_neg_int_less_than_5 :
    {n : ℤ | n = |n| ∧ n < 5} = {0, 1, 2, 3, 4} :=
by {
  sorry
}

-- Proof problem 4
theorem sum_eq_6_in_nat :
    {p : ℕ × ℕ | p.1 + p.2 = 6 ∧ p.1 > 0 ∧ p.2 > 0} = {(1, 5), (2, 4), (3, 3), (4, 2), (5, 1)} :=
by {
  sorry
}

-- Proof problem 5
theorem expressing_sequence:
    {-3, -1, 1, 3, 5} = {x : ℤ | ∃ k : ℤ, x = 2 * k - 1 ∧ -1 ≤ k ∧ k ≤ 3} :=
by {
  sorry
}

end integers_abs_no_greater_than_2_pos_div_by_3_less_than_10_non_neg_int_less_than_5_sum_eq_6_in_nat_expressing_sequence_l119_119335


namespace increase_productivity_RnD_l119_119501

theorem increase_productivity_RnD :
  let RnD_t := 2640.92
  let ΔAPL_t2 := 0.81
  RnD_t / ΔAPL_t2 = 3260 :=
by
  let RnD_t := 2640.92
  let ΔAPL_t2 := 0.81
  have h : RnD_t / ΔAPL_t2 = 3260 := sorry
  exact h

end increase_productivity_RnD_l119_119501


namespace x_equals_neg_one_l119_119507

theorem x_equals_neg_one
  (a b c : ℝ)
  (h1 : a ≠ 0)
  (h2 : b ≠ 0)
  (h3 : c ≠ 0)
  (h4 : (a + b - c) / c = (a - b + c) / b ∧ (a + b - c) / c = (-a + b + c) / a)
  (x : ℝ)
  (h5 : x = (a + b) * (b + c) * (c + a) / (a * b * c))
  (h6 : x < 0) :
  x = -1 := 
sorry

end x_equals_neg_one_l119_119507


namespace find_a_plus_b_l119_119293

theorem find_a_plus_b (a b : ℝ) (h1 : 2 * a = -6) (h2 : a^2 - b = 4) : a + b = 2 :=
by
  sorry

end find_a_plus_b_l119_119293


namespace find_c_l119_119085

theorem find_c (c d : ℝ) (h : ∀ x : ℝ, 9 * x^2 - 24 * x + c = (3 * x + d)^2) : c = 16 :=
sorry

end find_c_l119_119085


namespace boys_without_notebooks_l119_119247

/-
Given that:
1. There are 16 boys in Ms. Green's history class.
2. 20 students overall brought their notebooks to class.
3. 11 of the students who brought notebooks are girls.

Prove that the number of boys who did not bring their notebooks is 7.
-/

theorem boys_without_notebooks (total_boys : ℕ) (total_notebooks : ℕ) (girls_with_notebooks : ℕ)
  (hb : total_boys = 16) (hn : total_notebooks = 20) (hg : girls_with_notebooks = 11) : 
  (total_boys - (total_notebooks - girls_with_notebooks) = 7) :=
by
  sorry

end boys_without_notebooks_l119_119247


namespace eggs_left_over_l119_119141

def david_eggs : ℕ := 44
def elizabeth_eggs : ℕ := 52
def fatima_eggs : ℕ := 23
def carton_size : ℕ := 12

theorem eggs_left_over : 
  (david_eggs + elizabeth_eggs + fatima_eggs) % carton_size = 11 :=
by sorry

end eggs_left_over_l119_119141


namespace carol_rectangle_width_l119_119602

theorem carol_rectangle_width 
  (area_jordan : ℕ) (length_jordan width_jordan : ℕ) (width_carol length_carol : ℕ)
  (h1 : length_jordan = 12)
  (h2 : width_jordan = 10)
  (h3 : width_carol = 24)
  (h4 : area_jordan = length_jordan * width_jordan)
  (h5 : area_jordan = length_carol * width_carol) :
  length_carol = 5 :=
by
  sorry

end carol_rectangle_width_l119_119602


namespace sphere_radius_eq_three_l119_119195

theorem sphere_radius_eq_three (r : ℝ) (h : 4 / 3 * π * r ^ 3 = 4 * π * r ^ 2) : r = 3 :=
by
  sorry

end sphere_radius_eq_three_l119_119195


namespace gears_can_rotate_l119_119697

theorem gears_can_rotate (n : ℕ) : (∃ f : ℕ → Prop, f 0 ∧ (∀ k, f (k+1) ↔ ¬f k) ∧ f n = f 0) ↔ (n % 2 = 0) :=
by
  sorry

end gears_can_rotate_l119_119697


namespace find_num_carbon_atoms_l119_119118

def num_carbon_atoms (nH nO mH mC mO mol_weight : ℕ) : ℕ :=
  (mol_weight - (nH * mH + nO * mO)) / mC

theorem find_num_carbon_atoms :
  num_carbon_atoms 2 3 1 12 16 62 = 1 :=
by
  -- The proof is skipped
  sorry

end find_num_carbon_atoms_l119_119118


namespace find_a_l119_119655

theorem find_a (a : ℝ) (h : (3 * a + 2) + (a + 14) = 0) : a = -4 :=
sorry

end find_a_l119_119655


namespace value_of_k_l119_119541

theorem value_of_k (k m : ℝ)
    (h1 : m = k / 3)
    (h2 : 2 = k / (3 * m - 1)) :
    k = 2 := by
  sorry

end value_of_k_l119_119541


namespace min_value_y_l119_119838

theorem min_value_y : ∀ (x : ℝ), ∃ y_min : ℝ, y_min = (x^2 + 16 * x + 10) ∧ ∀ (x' : ℝ), (x'^2 + 16 * x' + 10) ≥ y_min := 
by 
  sorry

end min_value_y_l119_119838


namespace sum_of_a_for_unique_solution_l119_119327

theorem sum_of_a_for_unique_solution (a : ℝ) (h : (a + 12)^2 - 384 = 0) : 
  let a1 := -12 + 16 * Real.sqrt 6
  let a2 := -12 - 16 * Real.sqrt 6
  a1 + a2 = -24 := 
by
  sorry

end sum_of_a_for_unique_solution_l119_119327


namespace intersection_of_A_and_B_l119_119831

def A : Set ℝ := {x | -2 < x ∧ x < 1}
def B : Set ℝ := {x | x^2 - 2*x ≤ 0}

theorem intersection_of_A_and_B : A ∩ B = {x | 0 ≤ x ∧ x < 1} := 
by
  sorry

end intersection_of_A_and_B_l119_119831


namespace sequence_sum_is_25_div_3_l119_119317

noncomputable def sum_of_arithmetic_sequence (a n d : ℝ) : ℝ := (n / 2) * (2 * a + (n - 1) * d)

theorem sequence_sum_is_25_div_3 (a d : ℝ)
  (h1 : a + 4 * d = 1)
  (h2 : 3 * a + 15 * d = 2 * a + 8 * d) :
  sum_of_arithmetic_sequence a 10 d = 25 / 3 := by
  sorry

end sequence_sum_is_25_div_3_l119_119317


namespace angle_ACD_measure_l119_119272

theorem angle_ACD_measure {ABD BAE ABC ACD : ℕ} 
  (h1 : ABD = 125) 
  (h2 : BAE = 95) 
  (h3 : ABC = 180 - ABD) 
  (h4 : ABD + ABC = 180 ) : 
  ACD = 180 - (BAE + ABC) :=
by 
  sorry

end angle_ACD_measure_l119_119272


namespace election_winning_percentage_l119_119664

def total_votes (a b c : ℕ) : ℕ := a + b + c

def winning_percentage (votes_winning : ℕ) (total : ℕ) : ℚ :=
(votes_winning * 100 : ℚ) / total

theorem election_winning_percentage (a b c : ℕ) (h_votes : a = 6136 ∧ b = 7636 ∧ c = 11628) :
  winning_percentage c (total_votes a b c) = 45.78 := by
  sorry

end election_winning_percentage_l119_119664


namespace klinker_age_l119_119492

theorem klinker_age (K D : ℕ) (h1 : D = 10) (h2 : K + 15 = 2 * (D + 15)) : K = 35 :=
by
  sorry

end klinker_age_l119_119492


namespace find_angle_C_l119_119411

variables {A B C : ℝ} {a b c : ℝ} 

theorem find_angle_C (h1 : a^2 + b^2 - c^2 + a*b = 0) (C_pos : 0 < C) (C_lt_pi : C < Real.pi) :
  C = (2 * Real.pi) / 3 :=
sorry

end find_angle_C_l119_119411


namespace fewer_servings_per_day_l119_119173

theorem fewer_servings_per_day :
  ∀ (daily_consumption servings_old servings_new: ℕ),
    daily_consumption = 64 →
    servings_old = 8 →
    servings_new = 16 →
    (daily_consumption / servings_old) - (daily_consumption / servings_new) = 4 :=
by
  intros daily_consumption servings_old servings_new h1 h2 h3
  sorry

end fewer_servings_per_day_l119_119173


namespace number_of_packs_l119_119213

-- Given conditions
def cost_per_pack : ℕ := 11
def total_money : ℕ := 110

-- Statement to prove
theorem number_of_packs :
  total_money / cost_per_pack = 10 := by
  sorry

end number_of_packs_l119_119213


namespace base7_digit_divisibility_l119_119005

-- Define base-7 digit integers
notation "digit" => Fin 7

-- Define conversion from base-7 to base-10 for the form 3dd6_7
def base7_to_base10 (d : digit) : ℤ := 3 * (7^3) + (d:ℤ) * (7^2) + (d:ℤ) * 7 + 6

-- Define the property of being divisible by 13
def is_divisible_by_13 (n : ℤ) : Prop := ∃ k : ℤ, n = 13 * k

-- Formalize the theorem
theorem base7_digit_divisibility (d : digit) :
  is_divisible_by_13 (base7_to_base10 d) ↔ d = 4 :=
sorry

end base7_digit_divisibility_l119_119005


namespace part_length_proof_l119_119882

-- Define the scale length in feet and inches
def scale_length_ft : ℕ := 6
def scale_length_inch : ℕ := 8

-- Define the number of equal parts
def num_parts : ℕ := 4

-- Calculate total length in inches
def total_length_inch : ℕ := scale_length_ft * 12 + scale_length_inch

-- Calculate the length of each part in inches
def part_length_inch : ℕ := total_length_inch / num_parts

-- Prove that each part is 1 foot 8 inches long
theorem part_length_proof :
  part_length_inch = 1 * 12 + 8 :=
by
  sorry

end part_length_proof_l119_119882


namespace inverse_square_relationship_l119_119605

theorem inverse_square_relationship (k : ℝ) (y : ℝ) (h1 : ∀ x y, x = k / y^2)
  (h2 : ∃ y, 1 = k / y^2) (h3 : 0.5625 = k / 4^2) :
  ∃ y, 1 = 9 / y^2 ∧ y = 3 :=
by
  sorry

end inverse_square_relationship_l119_119605


namespace gadgets_selling_prices_and_total_amount_l119_119760

def cost_price_mobile : ℕ := 16000
def cost_price_laptop : ℕ := 25000
def cost_price_camera : ℕ := 18000

def loss_percentage_mobile : ℕ := 20
def gain_percentage_laptop : ℕ := 15
def loss_percentage_camera : ℕ := 10

def selling_price_mobile : ℕ := cost_price_mobile - (cost_price_mobile * loss_percentage_mobile / 100)
def selling_price_laptop : ℕ := cost_price_laptop + (cost_price_laptop * gain_percentage_laptop / 100)
def selling_price_camera : ℕ := cost_price_camera - (cost_price_camera * loss_percentage_camera / 100)

def total_amount_received : ℕ := selling_price_mobile + selling_price_laptop + selling_price_camera

theorem gadgets_selling_prices_and_total_amount :
  selling_price_mobile = 12800 ∧
  selling_price_laptop = 28750 ∧
  selling_price_camera = 16200 ∧
  total_amount_received = 57750 := by
  sorry

end gadgets_selling_prices_and_total_amount_l119_119760


namespace rectangular_solid_surface_area_l119_119321

theorem rectangular_solid_surface_area (a b c : ℕ) (h_a_prime : Nat.Prime a) (h_b_prime : Nat.Prime b) (h_c_prime : Nat.Prime c) 
  (volume_eq : a * b * c = 273) :
  2 * (a * b + b * c + c * a) = 302 := 
sorry

end rectangular_solid_surface_area_l119_119321


namespace xy_yz_zx_value_l119_119673

theorem xy_yz_zx_value (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x^2 + x * y + y^2 = 9) (h2 : y^2 + y * z + z^2 = 16) (h3 : z^2 + z * x + x^2 = 25) :
  x * y + y * z + z * x = 8 * Real.sqrt 3 :=
by sorry

end xy_yz_zx_value_l119_119673


namespace parakeets_per_cage_l119_119908

theorem parakeets_per_cage 
  (num_cages : ℕ) 
  (parrots_per_cage : ℕ) 
  (total_birds : ℕ) 
  (hcages : num_cages = 6) 
  (hparrots : parrots_per_cage = 6) 
  (htotal : total_birds = 48) :
  (total_birds - num_cages * parrots_per_cage) / num_cages = 2 := 
  by
  sorry

end parakeets_per_cage_l119_119908


namespace reduced_price_equals_50_l119_119414

noncomputable def reduced_price (P : ℝ) : ℝ := 0.75 * P

theorem reduced_price_equals_50 (P : ℝ) (X : ℝ) 
  (h1 : 1000 = X * P)
  (h2 : 1000 = (X + 5) * 0.75 * P) : reduced_price P = 50 :=
sorry

end reduced_price_equals_50_l119_119414


namespace right_triangle_condition_l119_119381

theorem right_triangle_condition (A B C : ℝ) (a b c : ℝ) :
  (A + B = 90) → (A + B + C = 180) → (C = 90) := 
by
  sorry

end right_triangle_condition_l119_119381


namespace third_year_award_count_l119_119181

-- Define the variables and conditions
variables (x x1 x2 x3 x4 x5 : ℕ)

-- The conditions and definition for the problem
def conditions : Prop :=
  (x1 = x) ∧
  (x5 = 3 * x) ∧
  (x1 < x2) ∧
  (x2 < x3) ∧
  (x3 < x4) ∧
  (x4 < x5) ∧
  (x1 + x2 + x3 + x4 + x5 = 27)

-- The theorem statement
theorem third_year_award_count (h : conditions x x1 x2 x3 x4 x5) : x3 = 5 :=
sorry

end third_year_award_count_l119_119181


namespace matrix_product_is_correct_l119_119666

-- Define the matrices A and B
def A : Matrix (Fin 3) (Fin 3) ℤ := ![
  ![3, 1, 1],
  ![2, 1, 2],
  ![1, 2, 3]
]

def B : Matrix (Fin 3) (Fin 3) ℤ := ![
  ![1, 1, -1],
  ![2, -1, 1],
  ![1, 0, 1]
]

-- Define the expected product matrix C
def C : Matrix (Fin 3) (Fin 3) ℤ := ![
  ![6, 2, -1],
  ![6, 1, 1],
  ![8, -1, 4]
]

-- The statement of the problem
theorem matrix_product_is_correct : (A * B) = C := by
  sorry -- Proof is omitted as per instructions

end matrix_product_is_correct_l119_119666


namespace Sawyer_cleans_in_6_hours_l119_119777

theorem Sawyer_cleans_in_6_hours (N : ℝ) (S : ℝ) (h1 : S = (2/3) * N) 
                                 (h2 : 1/S + 1/N = 1/3.6) : S = 6 :=
by
  sorry

end Sawyer_cleans_in_6_hours_l119_119777


namespace Parabola_vertex_form_l119_119608

theorem Parabola_vertex_form (x : ℝ) (y : ℝ) : 
  (∃ h k : ℝ, (h = -2) ∧ (k = 1) ∧ (y = (x + h)^2 + k) ) ↔ (y = (x + 2)^2 + 1) :=
by
  sorry

end Parabola_vertex_form_l119_119608


namespace min_sum_of_abc_conditions_l119_119783

theorem min_sum_of_abc_conditions
  (a b c d : ℕ)
  (hab : a + b = 2)
  (hac : a + c = 3)
  (had : a + d = 4)
  (hbc : b + c = 5)
  (hbd : b + d = 6)
  (hcd : c + d = 7) :
  a + b + c + d = 9 :=
sorry

end min_sum_of_abc_conditions_l119_119783


namespace sum_of_parts_l119_119540

theorem sum_of_parts (x y : ℝ) (h1 : x + y = 52) (h2 : y = 30.333333333333332) :
  10 * x + 22 * y = 884 :=
sorry

end sum_of_parts_l119_119540


namespace two_A_plus_B_l119_119385

theorem two_A_plus_B (A B : ℕ) (h1 : A = Nat.gcd (Nat.gcd 12 18) 30) (h2 : B = Nat.lcm (Nat.lcm 12 18) 30) : 2 * A + B = 192 :=
by
  sorry

end two_A_plus_B_l119_119385


namespace grill_ran_for_16_hours_l119_119674

def coals_burn_time_A (bags : List ℕ) : ℕ :=
  bags.foldl (λ acc n => acc + (n / 15 * 20)) 0

def coals_burn_time_B (bags : List ℕ) : ℕ :=
  bags.foldl (λ acc n => acc + (n / 10 * 30)) 0

def total_grill_time (bags_A bags_B : List ℕ) : ℕ :=
  coals_burn_time_A bags_A + coals_burn_time_B bags_B

def bags_A : List ℕ := [60, 75, 45]
def bags_B : List ℕ := [50, 70, 40, 80]

theorem grill_ran_for_16_hours :
  total_grill_time bags_A bags_B = 960 / 60 :=
by
  unfold total_grill_time coals_burn_time_A coals_burn_time_B
  unfold bags_A bags_B
  norm_num
  sorry

end grill_ran_for_16_hours_l119_119674


namespace total_floor_area_l119_119903

theorem total_floor_area
    (n : ℕ) (a_cm : ℕ)
    (num_of_slabs : n = 30)
    (length_of_slab_cm : a_cm = 130) :
    (30 * ((130 * 130) / 10000)) = 50.7 :=
by
  sorry

end total_floor_area_l119_119903


namespace megan_works_per_day_hours_l119_119735

theorem megan_works_per_day_hours
  (h : ℝ)
  (earnings_per_hour : ℝ)
  (days_per_month : ℝ)
  (total_earnings_two_months : ℝ) :
  earnings_per_hour = 7.50 →
  days_per_month = 20 →
  total_earnings_two_months = 2400 →
  2 * days_per_month * earnings_per_hour * h = total_earnings_two_months →
  h = 8 :=
by {
  sorry
}

end megan_works_per_day_hours_l119_119735


namespace num_of_tenths_in_1_9_num_of_hundredths_in_0_8_l119_119945

theorem num_of_tenths_in_1_9 : (1.9 / 0.1) = 19 :=
by sorry

theorem num_of_hundredths_in_0_8 : (0.8 / 0.01) = 80 :=
by sorry

end num_of_tenths_in_1_9_num_of_hundredths_in_0_8_l119_119945


namespace find_x_solution_l119_119421

theorem find_x_solution
  (x y z : ℤ)
  (h1 : 4 * x + y + z = 80)
  (h2 : 2 * x - y - z = 40)
  (h3 : 3 * x + y - z = 20) :
  x = 20 :=
by
  -- Proof steps go here...
  sorry

end find_x_solution_l119_119421


namespace total_pages_in_book_l119_119593

theorem total_pages_in_book (pages_monday pages_tuesday total_pages_read total_pages_book : ℝ)
    (h1 : pages_monday = 15.5)
    (h2 : pages_tuesday = 1.5 * pages_monday + 16)
    (h3 : total_pages_read = pages_monday + pages_tuesday)
    (h4 : total_pages_book = 2 * total_pages_read) :
    total_pages_book = 109.5 :=
by
  sorry

end total_pages_in_book_l119_119593


namespace third_generation_tail_length_l119_119034

theorem third_generation_tail_length (tail_length : ℕ → ℕ) (h0 : tail_length 0 = 16)
    (h_next : ∀ n, tail_length (n + 1) = tail_length n + (25 * tail_length n) / 100) :
    tail_length 2 = 25 :=
by
  sorry

end third_generation_tail_length_l119_119034


namespace repeating_decimal_as_fraction_l119_119768

-- Define the repeating decimal x as .overline{37}
def x : ℚ := 37 / 99

-- The theorem we need to prove
theorem repeating_decimal_as_fraction : x = 37 / 99 := by
  sorry

end repeating_decimal_as_fraction_l119_119768


namespace min_angle_for_quadrilateral_l119_119860

theorem min_angle_for_quadrilateral (d : ℝ) (h : ∀ (a b c d : ℝ), 
  0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ a + b + c + d = 360 → (a < d ∨ b < d)) :
  d = 120 :=
by
  sorry

end min_angle_for_quadrilateral_l119_119860


namespace horner_eval_at_minus_point_two_l119_119003

def f (x : ℝ) : ℝ :=
  1 + x + 0.5 * x^2 + 0.16667 * x^3 + 0.04167 * x^4 + 0.00833 * x^5

theorem horner_eval_at_minus_point_two :
  f (-0.2) = 0.81873 :=
by 
  sorry

end horner_eval_at_minus_point_two_l119_119003


namespace evening_sales_l119_119038

theorem evening_sales
  (remy_bottles_morning : ℕ := 55)
  (nick_bottles_fewer : ℕ := 6)
  (price_per_bottle : ℚ := 0.50)
  (evening_sales_more : ℚ := 3) :
  let nick_bottles_morning := remy_bottles_morning - nick_bottles_fewer
  let remy_sales_morning := remy_bottles_morning * price_per_bottle
  let nick_sales_morning := nick_bottles_morning * price_per_bottle
  let total_morning_sales := remy_sales_morning + nick_sales_morning
  let total_evening_sales := total_morning_sales + evening_sales_more
  total_evening_sales = 55 :=
by
  sorry

end evening_sales_l119_119038


namespace fractional_equation_no_solution_l119_119120

theorem fractional_equation_no_solution (a : ℝ) :
  (¬ ∃ x, x ≠ 1 ∧ x ≠ 0 ∧ ((x - a) / (x - 1) - 3 / x = 1)) → (a = 1 ∨ a = -2) :=
by
  sorry

end fractional_equation_no_solution_l119_119120


namespace tickets_won_in_skee_ball_l119_119009

-- Define the conditions as Lean definitions
def tickets_from_whack_a_mole : ℕ := 8
def ticket_cost_per_candy : ℕ := 5
def candies_bought : ℕ := 3

-- We now state the conjecture (mathematical proof problem) 
-- Prove that the number of tickets won in skee ball is 7.
theorem tickets_won_in_skee_ball :
  (candies_bought * ticket_cost_per_candy) - tickets_from_whack_a_mole = 7 :=
by
  sorry

end tickets_won_in_skee_ball_l119_119009


namespace equal_area_intersection_l119_119298

variable (p q r s : ℚ)
noncomputable def intersection_point (x y : ℚ) : Prop :=
  4 * x + 5 * p / q = 12 * p / q ∧ 8 * y = p 

theorem equal_area_intersection :
  intersection_point p q r s /\
  p + q + r + s = 60 := 
by 
  sorry

end equal_area_intersection_l119_119298


namespace volleyballTeam_starters_l119_119219

noncomputable def chooseStarters (totalPlayers : ℕ) (quadruplets : ℕ) (starters : ℕ) : ℕ :=
  let remainingPlayers := totalPlayers - quadruplets
  let chooseQuadruplet := quadruplets
  let chooseRemaining := Nat.choose remainingPlayers (starters - 1)
  chooseQuadruplet * chooseRemaining

theorem volleyballTeam_starters :
  chooseStarters 16 4 6 = 3168 :=
by
  sorry

end volleyballTeam_starters_l119_119219


namespace probability_is_8point64_percent_l119_119315

/-- Define the probabilities based on given conditions -/
def p_excel : ℝ := 0.45
def p_night_shift_given_excel : ℝ := 0.32
def p_no_weekend_given_night_shift : ℝ := 0.60

/-- Calculate the combined probability -/
def combined_probability :=
  p_excel * p_night_shift_given_excel * p_no_weekend_given_night_shift

theorem probability_is_8point64_percent :
  combined_probability = 0.0864 :=
by
  -- We will skip the proof for now
  sorry

end probability_is_8point64_percent_l119_119315


namespace probability_adjacent_A_before_B_l119_119672

theorem probability_adjacent_A_before_B 
  (total_students : ℕ)
  (A B C D : ℚ)
  (hA : total_students = 8)
  (hB : B = 1/3) : 
  (∃ prob : ℚ, prob = 1/3) :=
by
  sorry

end probability_adjacent_A_before_B_l119_119672


namespace geometric_series_sum_l119_119083

theorem geometric_series_sum : 
  ∑' n : ℕ, (1 / 4) * (1 / 2)^n = 1 / 2 := 
by 
  sorry

end geometric_series_sum_l119_119083


namespace add_percentages_10_30_15_50_l119_119772

-- Define the problem conditions:
def ten_percent (x : ℝ) : ℝ := 0.10 * x
def fifteen_percent (y : ℝ) : ℝ := 0.15 * y
def add_percentages (x y : ℝ) : ℝ := ten_percent x + fifteen_percent y

theorem add_percentages_10_30_15_50 :
  add_percentages 30 50 = 10.5 :=
by
  sorry

end add_percentages_10_30_15_50_l119_119772


namespace positional_relationship_perpendicular_l119_119959

theorem positional_relationship_perpendicular 
  (a b c : ℝ) 
  (A B C : ℝ)
  (h : b * Real.sin A - a * Real.sin B = 0) :
  (∀ x y : ℝ, (x * Real.sin A + a * y + c = 0) ↔ (b * x - y * Real.sin B + Real.sin C = 0)) :=
sorry

end positional_relationship_perpendicular_l119_119959


namespace smallest_x_for_max_f_l119_119623

noncomputable def f (x : ℝ) : ℝ := Real.sin (x / 4) + Real.sin (x / 12)

theorem smallest_x_for_max_f : ∃ x > 0, f x = 2 ∧ ∀ y > 0, (f y = 2 → y ≥ x) :=
sorry

end smallest_x_for_max_f_l119_119623


namespace journey_total_time_l119_119514

theorem journey_total_time (speed1 time1 speed2 total_distance : ℕ) 
  (h1 : speed1 = 40) 
  (h2 : time1 = 3) 
  (h3 : speed2 = 60) 
  (h4 : total_distance = 240) : 
  time1 + (total_distance - speed1 * time1) / speed2 = 5 := 
by 
  sorry

end journey_total_time_l119_119514


namespace students_remaining_after_four_stops_l119_119390

theorem students_remaining_after_four_stops :
  let initial_students := 60 
  let fraction_remaining := (2 / 3 : ℚ)
  let stop1_students := initial_students * fraction_remaining
  let stop2_students := stop1_students * fraction_remaining
  let stop3_students := stop2_students * fraction_remaining
  let stop4_students := stop3_students * fraction_remaining
  stop4_students = (320 / 27 : ℚ) :=
by
  sorry

end students_remaining_after_four_stops_l119_119390


namespace polynomial_sequence_symmetric_l119_119898

def P : ℕ → ℝ → ℝ → ℝ → ℝ 
| 0, x, y, z => 1
| (m + 1), x, y, z => (x + z) * (y + z) * P m x y (z + 1) - z^2 * P m x y z

theorem polynomial_sequence_symmetric (m : ℕ) (x y z : ℝ) (σ : ℝ × ℝ × ℝ): 
  P m x y z = P m σ.1 σ.2.1 σ.2.2 :=
sorry

end polynomial_sequence_symmetric_l119_119898


namespace model_distance_comparison_l119_119275

theorem model_distance_comparison (m h c x y z : ℝ) (hm : 0 < m) (hh : 0 < h) (hc : 0 < c) (hz : 0 < z) (hx : 0 < x) (hy : 0 < y)
    (h_eq : (x - c) * z = (y - c) * (z + m) + h) :
    (if h > c * m then (x * z > y * (z + m))
     else if h < c * m then (x * z < y * (z + m))
     else (h = c * m → x * z = y * (z + m))) :=
by
  sorry

end model_distance_comparison_l119_119275


namespace axis_of_symmetry_l119_119957

noncomputable def f (x : ℝ) : ℝ :=
  (Real.cos (x + (Real.pi / 2))) * (Real.cos (x + (Real.pi / 4)))

theorem axis_of_symmetry : 
  ∃ (a : ℝ), a = 5 * Real.pi / 8 ∧ ∀ x : ℝ, f (2 * a - x) = f x := 
by
  sorry

end axis_of_symmetry_l119_119957


namespace subtraction_identity_l119_119594

theorem subtraction_identity : 4444444444444 - 2222222222222 - 444444444444 = 1777777777778 :=
  by norm_num

end subtraction_identity_l119_119594


namespace range_of_a_l119_119467

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (0 < x) → (-3^x ≤ a)) ↔ (a ≥ -1) :=
by
  sorry

end range_of_a_l119_119467


namespace composite_numbers_l119_119562

theorem composite_numbers (n : ℕ) (hn : n > 0) :
  (∃ p q, p > 1 ∧ q > 1 ∧ 2 * 2^(2^n) + 1 = p * q) ∧ 
  (∃ p q, p > 1 ∧ q > 1 ∧ 3 * 2^(2*n) + 1 = p * q) :=
sorry

end composite_numbers_l119_119562


namespace minimum_value_of_quadratic_function_l119_119663

def quadratic_function (x : ℝ) : ℝ :=
  x^2 - 8 * x + 15

theorem minimum_value_of_quadratic_function :
  ∃ x : ℝ, quadratic_function x = -1 ∧ ∀ y : ℝ, quadratic_function y ≥ -1 :=
by
  sorry

end minimum_value_of_quadratic_function_l119_119663


namespace dave_trips_l119_119397

theorem dave_trips :
  let trays_at_a_time := 12
  let trays_table_1 := 26
  let trays_table_2 := 49
  let trays_table_3 := 65
  let trays_table_4 := 38
  let total_trays := trays_table_1 + trays_table_2 + trays_table_3 + trays_table_4
  let trips := (total_trays + trays_at_a_time - 1) / trays_at_a_time
  trips = 15 := by
    repeat { sorry }

end dave_trips_l119_119397


namespace find_y_l119_119445

theorem find_y (x y : ℝ) (h1 : x = 8) (h2 : x^(3 * y) = 64) : y = 2 / 3 :=
by
  -- Proof omitted
  sorry

end find_y_l119_119445


namespace find_x_l119_119627

variable (A B : Set ℕ)
variable (x : ℕ)

theorem find_x (hA : A = {1, 3}) (hB : B = {2, x}) (hUnion : A ∪ B = {1, 2, 3, 4}) : x = 4 := by
  sorry

end find_x_l119_119627


namespace single_digit_pairs_l119_119563

theorem single_digit_pairs:
  ∃ x y: ℕ, x ≠ 1 ∧ x ≠ 9 ∧ y ≠ 1 ∧ y ≠ 9 ∧ x < 10 ∧ y < 10 ∧ 
  (x * y < 100 ∧ ((x * y) % 10 + (x * y) / 10 == x ∨ (x * y) % 10 + (x * y) / 10 == y))
  → (x, y) ∈ [(3, 4), (3, 7), (6, 4), (6, 7)] :=
by
  sorry

end single_digit_pairs_l119_119563


namespace number_whose_square_is_64_l119_119656

theorem number_whose_square_is_64 (x : ℝ) (h : x^2 = 64) : x = 8 ∨ x = -8 :=
sorry

end number_whose_square_is_64_l119_119656


namespace train_speed_l119_119450

theorem train_speed (length_train : ℝ) (time_to_cross : ℝ) (length_bridge : ℝ)
  (h_train : length_train = 100) (h_time : time_to_cross = 12.499)
  (h_bridge : length_bridge = 150) : 
  ((length_train + length_bridge) / time_to_cross * 3.6) = 72 := 
by 
  sorry

end train_speed_l119_119450


namespace compute_r_l119_119796

noncomputable def r (side_length : ℝ) : ℝ :=
  let a := (0.5 * side_length, 0.5 * side_length)
  let b := (1.5 * side_length, 2.5 * side_length)
  let c := (2.5 * side_length, 1.5 * side_length)
  let ab := Real.sqrt ((b.1 - a.1)^2 + (b.2 - a.2)^2)
  let ac := Real.sqrt ((c.1 - a.1)^2 + (c.2 - a.2)^2)
  let bc := Real.sqrt ((c.1 - b.1)^2 + (c.2 - b.2)^2)
  let s := (ab + ac + bc) / 2
  let area_ABC := Real.sqrt (s * (s - ab) * (s - ac) * (s - bc))
  let circumradius := ab * ac * bc / (4 * area_ABC)
  circumradius - (side_length / 2)

theorem compute_r :
  r 1 = (5 * Real.sqrt 2 - 3) / 6 :=
by
  unfold r
  sorry

end compute_r_l119_119796


namespace inequality_proof_l119_119270

open Real

theorem inequality_proof 
  (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_cond : a^2 + b^2 + c^2 = 3) :
  (a / (a + 5) + b / (b + 5) + c / (c + 5) ≤ 1 / 2) :=
by
  sorry

end inequality_proof_l119_119270


namespace john_total_distance_l119_119115

def speed : ℕ := 45
def time1 : ℕ := 2
def time2 : ℕ := 3

theorem john_total_distance:
  speed * (time1 + time2) = 225 := by
  sorry

end john_total_distance_l119_119115


namespace sum_of_solutions_l119_119855

-- Define the quadratic equation as a product of linear factors
def quadratic_eq (x : ℚ) : Prop := (4 * x + 6) * (3 * x - 8) = 0

-- Define the roots of the quadratic equation
def root1 : ℚ := -3 / 2
def root2 : ℚ := 8 / 3

-- Sum of the roots of the quadratic equation
def sum_of_roots : ℚ := root1 + root2

-- Theorem stating that the sum of the roots is 7/6
theorem sum_of_solutions : sum_of_roots = 7 / 6 := by
  sorry

end sum_of_solutions_l119_119855


namespace speed_of_car_A_l119_119386

variable (V_A V_B T : ℕ)
variable (h1 : V_B = 35) (h2 : T = 10) (h3 : 2 * V_B * T = V_A * T)

theorem speed_of_car_A :
  V_A = 70 :=
by
  sorry

end speed_of_car_A_l119_119386


namespace find_certain_number_l119_119243

theorem find_certain_number (x y : ℕ) (h1 : x = 19) (h2 : x + y = 36) :
  8 * x + 3 * y = 203 := by
  sorry

end find_certain_number_l119_119243


namespace rent_for_each_room_l119_119979

theorem rent_for_each_room (x : ℝ) (ha : 4800 / x = 4200 / (x - 30)) (hx : x = 240) :
  x = 240 ∧ (x - 30) = 210 :=
by
  sorry

end rent_for_each_room_l119_119979


namespace correctness_of_option_C_l119_119473

-- Define the conditions as hypotheses
variable (x y : ℝ)

def condA : Prop := ∀ x: ℝ, x^3 * x^5 = x^15
def condB : Prop := ∀ x y: ℝ, 2 * x + 3 * y = 5 * x * y
def condC : Prop := ∀ x y: ℝ, 2 * x^2 * (3 * x^2 - 5 * y) = 6 * x^4 - 10 * x^2 * y
def condD : Prop := ∀ x: ℝ, (x - 2)^2 = x^2 - 4

-- State the proof problem is correct
theorem correctness_of_option_C (x y : ℝ) : 2 * x^2 * (3 * x^2 - 5 * y) = 6 * x^4 - 10 * x^2 * y := by
  sorry

end correctness_of_option_C_l119_119473


namespace necessary_but_not_sufficient_condition_l119_119554

theorem necessary_but_not_sufficient_condition (x : ℝ) (h : x > e) : x > 1 :=
sorry

end necessary_but_not_sufficient_condition_l119_119554


namespace angle_C_is_70_l119_119836

namespace TriangleAngleSum

def angle_sum_in_triangle (A B C : ℝ) : Prop :=
  A + B + C = 180

def sum_of_two_angles (A B : ℝ) : Prop :=
  A + B = 110

theorem angle_C_is_70 {A B C : ℝ} (h1 : angle_sum_in_triangle A B C) (h2 : sum_of_two_angles A B) : C = 70 :=
by
  sorry

end TriangleAngleSum

end angle_C_is_70_l119_119836


namespace mark_height_feet_l119_119612

theorem mark_height_feet
  (mark_height_inches : ℕ)
  (mike_height_feet : ℕ)
  (mike_height_inches : ℕ)
  (mike_taller_than_mark : ℕ)
  (foot_in_inches : ℕ)
  (mark_height_eq : mark_height_inches = 3)
  (mike_height_eq : mike_height_feet * foot_in_inches + mike_height_inches = 73)
  (mike_taller_eq : mike_height_feet * foot_in_inches + mike_height_inches = mark_height_inches + mike_taller_than_mark)
  (foot_in_inches_eq : foot_in_inches = 12) :
  mark_height_inches = 63 ∧ mark_height_inches / foot_in_inches = 5 := by
sorry

end mark_height_feet_l119_119612


namespace julias_change_l119_119665

theorem julias_change :
  let snickers := 2
  let mms := 3
  let cost_snickers := 1.5
  let cost_mms := 2 * cost_snickers
  let money_given := 2 * 10
  let total_cost := snickers * cost_snickers + mms * cost_mms
  let change := money_given - total_cost
  change = 8 :=
by
  sorry

end julias_change_l119_119665


namespace ways_to_divide_week_l119_119469

-- Define the total number of seconds in a week
def total_seconds_in_week : ℕ := 604800

-- Define the math problem statement
theorem ways_to_divide_week (n m : ℕ) (h : n * m = total_seconds_in_week) (hn : 0 < n) (hm : 0 < m) : 
  (∃ (n_pairs : ℕ), n_pairs = 144) :=
sorry

end ways_to_divide_week_l119_119469


namespace arithmetic_sequence_fifth_term_l119_119918

theorem arithmetic_sequence_fifth_term :
  let a1 := 3
  let d := 4
  let a5 := a1 + (5 - 1) * d
  a5 = 19 :=
by
  let a1 := 3
  let d := 4
  let a5 := a1 + (5 - 1) * d
  show a5 = 19
  sorry

end arithmetic_sequence_fifth_term_l119_119918


namespace exact_consecutive_hits_l119_119346

/-
Prove the number of ways to arrange 8 shots with exactly 3 hits such that exactly 2 out of the 3 hits are consecutive is 30.
-/

def count_distinct_sequences (total_shots : ℕ) (hits : ℕ) (consecutive_hits : ℕ) : ℕ :=
  if total_shots = 8 ∧ hits = 3 ∧ consecutive_hits = 2 then 30 else 0

theorem exact_consecutive_hits :
  count_distinct_sequences 8 3 2 = 30 :=
by
  -- The proof is omitted.
  sorry

end exact_consecutive_hits_l119_119346


namespace prob_sum_7_9_11_correct_l119_119729

def die1 : List ℕ := [1, 2, 3, 3, 4, 4]
def die2 : List ℕ := [2, 2, 5, 6, 7, 8]

def prob_sum_7_9_11 : ℚ := 
  (1/6 * 1/6 + 1/6 * 1/6) + 2/6 * 3/6

theorem prob_sum_7_9_11_correct :
  prob_sum_7_9_11 = 4 / 9 := 
by
  sorry

end prob_sum_7_9_11_correct_l119_119729


namespace farmer_harvest_correct_l119_119829

def estimated_harvest : ℕ := 48097
def additional_harvest : ℕ := 684
def total_harvest : ℕ := 48781

theorem farmer_harvest_correct : estimated_harvest + additional_harvest = total_harvest :=
by
  sorry

end farmer_harvest_correct_l119_119829


namespace label_sum_l119_119974

theorem label_sum (n : ℕ) : 
  (∃ S : ℕ → ℕ, S 1 = 2 ∧ (∀ k, k > 1 → (S (k + 1) = 2 * S k)) ∧ S n = 2 * 3 ^ (n - 1)) := 
sorry

end label_sum_l119_119974


namespace tom_rope_stories_l119_119812

/-- Define the conditions given in the problem. --/
def story_length : ℝ := 10
def rope_length : ℝ := 20
def loss_percentage : ℝ := 0.25
def pieces_of_rope : ℕ := 4

/-- Theorem to prove the number of stories Tom can lower the rope down. --/
theorem tom_rope_stories (story_length rope_length loss_percentage : ℝ) (pieces_of_rope : ℕ) : 
    story_length = 10 → 
    rope_length = 20 →
    loss_percentage = 0.25 →
    pieces_of_rope = 4 →
    pieces_of_rope * rope_length * (1 - loss_percentage) / story_length = 6 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end tom_rope_stories_l119_119812


namespace find_first_week_customers_l119_119775

def commission_per_customer := 1
def first_week_customers (C : ℕ) := C
def second_week_customers (C : ℕ) := 2 * C
def third_week_customers (C : ℕ) := 3 * C
def salary := 500
def bonus := 50
def total_earnings := 760

theorem find_first_week_customers (C : ℕ) (H : salary + bonus + commission_per_customer * (first_week_customers C + second_week_customers C + third_week_customers C) = total_earnings) : 
  C = 35 :=
by
  sorry

end find_first_week_customers_l119_119775


namespace teacher_buys_total_21_pens_l119_119517

def num_black_pens : Nat := 7
def num_blue_pens : Nat := 9
def num_red_pens : Nat := 5
def total_pens : Nat := num_black_pens + num_blue_pens + num_red_pens

theorem teacher_buys_total_21_pens : total_pens = 21 := 
by
  unfold total_pens num_black_pens num_blue_pens num_red_pens
  rfl -- reflexivity (21 = 21)

end teacher_buys_total_21_pens_l119_119517


namespace contradiction_with_angles_l119_119437

-- Definitions of conditions
def triangle (α β γ : ℝ) : Prop := α + β + γ = 180 ∧ α > 0 ∧ β > 0 ∧ γ > 0

-- The proposition we want to prove by contradiction
def at_least_one_angle_not_greater_than_60 (α β γ : ℝ) : Prop := α ≤ 60 ∨ β ≤ 60 ∨ γ ≤ 60

-- The assumption for contradiction
def all_angles_greater_than_60 (α β γ : ℝ) : Prop := α > 60 ∧ β > 60 ∧ γ > 60

-- The proof problem
theorem contradiction_with_angles (α β γ : ℝ) (h : triangle α β γ) :
  ¬ all_angles_greater_than_60 α β γ → at_least_one_angle_not_greater_than_60 α β γ :=
sorry

end contradiction_with_angles_l119_119437


namespace find_b_l119_119750

-- Define the problem based on the conditions identified
theorem find_b (b : ℕ) (h₁ : b > 0) (h₂ : (b : ℝ)/(b+15) = 0.75) : b = 45 := 
  sorry

end find_b_l119_119750


namespace bus_minibus_seats_l119_119793

theorem bus_minibus_seats (x y : ℕ) 
    (h1 : x = y + 20) 
    (h2 : 5 * x + 5 * y = 300) : 
    x = 40 ∧ y = 20 := 
by
  sorry

end bus_minibus_seats_l119_119793


namespace hawks_score_l119_119936

theorem hawks_score (E H : ℕ) (h1 : E + H = 82) (h2 : E = H + 22) : H = 30 :=
by
  sorry

end hawks_score_l119_119936


namespace parabola_vertex_f_l119_119230

theorem parabola_vertex_f (d e f : ℝ) (h_vertex : ∀ y, (d * (y - 3)^2 + 5) = (d * y^2 + e * y + f))
  (h_point : d * (6 - 3)^2 + 5 = 2) : f = 2 :=
by
  sorry

end parabola_vertex_f_l119_119230


namespace first_digit_of_base16_representation_l119_119136

-- Firstly we define the base conversion from base 4 to base 10 and from base 10 to base 16.
-- For simplicity, we assume that the required functions exist and skip their implementations.

-- Assume base 4 to base 10 conversion function
def base4_to_base10 (n : String) : Nat :=
  sorry

-- Assume base 10 to base 16 conversion function that gives the first digit
def first_digit_base16 (n : Nat) : Nat :=
  sorry

-- Given the base 4 number as string
def y_base4 : String := "20313320132220312031"

-- Define the final statement
theorem first_digit_of_base16_representation :
  first_digit_base16 (base4_to_base10 y_base4) = 5 :=
by
  sorry

end first_digit_of_base16_representation_l119_119136


namespace maximum_area_rectangle_l119_119103

-- Define the conditions
def length (x : ℝ) := x
def width (x : ℝ) := 2 * x
def perimeter (x : ℝ) := 2 * (length x + width x)

-- The proof statement
theorem maximum_area_rectangle (h : perimeter x = 40) : 2 * (length x) * (width x) = 800 / 9 :=
by
  sorry

end maximum_area_rectangle_l119_119103


namespace CE_length_l119_119991

theorem CE_length (AF ED AE area : ℝ) (hAF : AF = 30) (hED : ED = 50) (hAE : AE = 120) (h_area : area = 7200) : 
  ∃ CE : ℝ, CE = 138 :=
by
  -- omitted proof steps
  sorry

end CE_length_l119_119991


namespace commute_time_l119_119733

theorem commute_time (d s1 s2 : ℝ) (h1 : s1 = 45) (h2 : s2 = 30) (h3 : d = 18) : (d / s1 + d / s2 = 1) :=
by
  -- Definitions and assumptions
  rw [h1, h2, h3]
  -- Total time calculation
  exact sorry

end commute_time_l119_119733


namespace Mr_Tom_invested_in_fund_X_l119_119175

theorem Mr_Tom_invested_in_fund_X (a b : ℝ) (h1 : a + b = 100000) (h2 : 0.17 * b = 0.23 * a + 200) : a = 42000 := 
by
  sorry

end Mr_Tom_invested_in_fund_X_l119_119175


namespace find_k_l119_119260

variables (k : ℝ)
def vector_a : ℝ × ℝ := (1, 2)
def vector_b : ℝ × ℝ := (-3, 2)
def vector_k_a_plus_b (k : ℝ) : ℝ × ℝ := (k*1 + (-3), k*2 + 2)
def vector_a_minus_2b : ℝ × ℝ := (1 - 2*(-3), 2 - 2*2)

theorem find_k (h : (vector_k_a_plus_b k).fst * (vector_a_minus_2b).snd = (vector_k_a_plus_b k).snd * (vector_a_minus_2b).fst) : k = -1/2 :=
sorry

end find_k_l119_119260


namespace increasing_function_range_l119_119443

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x ≤ 1 then
  -x^2 - a*x - 5
else
  a / x

theorem increasing_function_range (a : ℝ) : 
  (∀ x y : ℝ, x ≤ y → f a x ≤ f a y) ↔ (-3 ≤ a ∧ a ≤ -2) :=
by
  sorry

end increasing_function_range_l119_119443


namespace residual_at_sample_point_l119_119649

theorem residual_at_sample_point :
  ∀ (x y : ℝ), (8 * x - 70 = 10) → (x = 10) → (y = 13) → (13 - (8 * x - 70) = 3) :=
by
  intros x y h1 h2 h3
  sorry

end residual_at_sample_point_l119_119649


namespace arithmetic_sequence_ratio_l119_119677

theorem arithmetic_sequence_ratio (a x b : ℝ) 
  (h1 : x - a = b - x)
  (h2 : 2 * x - b = b - x) :
  a / b = 1 / 3 :=
by
  sorry

end arithmetic_sequence_ratio_l119_119677


namespace proportion_of_segments_l119_119017

theorem proportion_of_segments
  (a b c d : ℝ)
  (h1 : b = 3)
  (h2 : c = 4)
  (h3 : d = 6)
  (h4 : a / b = c / d) :
  a = 2 :=
by
  sorry

end proportion_of_segments_l119_119017


namespace age_of_new_teacher_l119_119274

theorem age_of_new_teacher (sum_of_20_teachers : ℕ)
  (avg_age_20_teachers : ℕ)
  (total_teachers_after_new_teacher : ℕ)
  (new_avg_age_after_new_teacher : ℕ)
  (h1 : sum_of_20_teachers = 20 * 49)
  (h2 : avg_age_20_teachers = 49)
  (h3 : total_teachers_after_new_teacher = 21)
  (h4 : new_avg_age_after_new_teacher = 48) :
  ∃ (x : ℕ), x = 28 :=
by
  sorry

end age_of_new_teacher_l119_119274


namespace minimum_xy_l119_119382

theorem minimum_xy (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 1/x + 1/y = 1/2) : xy ≥ 16 :=
sorry

end minimum_xy_l119_119382


namespace strawberries_to_grapes_ratio_l119_119290

-- Define initial conditions
def initial_grapes : ℕ := 100
def fruits_left : ℕ := 96

-- Define the number of strawberries initially
def strawberries_init (S : ℕ) : Prop :=
  (S - (2 * (1/5) * S) = fruits_left - initial_grapes + ((2 * (1/5)) * initial_grapes))

-- Define the ratio problem in Lean
theorem strawberries_to_grapes_ratio (S : ℕ) (h : strawberries_init S) : (S / initial_grapes = 3 / 5) :=
sorry

end strawberries_to_grapes_ratio_l119_119290


namespace number_of_black_boxcars_l119_119675

def red_boxcars : Nat := 3
def blue_boxcars : Nat := 4
def black_boxcar_capacity : Nat := 4000
def boxcar_total_capacity : Nat := 132000

def blue_boxcar_capacity : Nat := 2 * black_boxcar_capacity
def red_boxcar_capacity : Nat := 3 * blue_boxcar_capacity

def red_boxcar_total_capacity : Nat := red_boxcars * red_boxcar_capacity
def blue_boxcar_total_capacity : Nat := blue_boxcars * blue_boxcar_capacity

def other_total_capacity : Nat := red_boxcar_total_capacity + blue_boxcar_total_capacity
def remaining_capacity : Nat := boxcar_total_capacity - other_total_capacity
def expected_black_boxcars : Nat := remaining_capacity / black_boxcar_capacity

theorem number_of_black_boxcars :
  expected_black_boxcars = 7 := by
  sorry

end number_of_black_boxcars_l119_119675


namespace complement_of_A_in_R_l119_119197

open Set

variable (R : Set ℝ) (A : Set ℝ)

def real_numbers : Set ℝ := {x | true}

def set_A : Set ℝ := {y | ∃ x : ℝ, y = x ^ 2}

theorem complement_of_A_in_R : (real_numbers \ set_A) = {y | y < 0} := by
  sorry

end complement_of_A_in_R_l119_119197


namespace total_hair_cut_l119_119566

-- Define the amounts cut on two consecutive days
def first_cut : ℝ := 0.375
def second_cut : ℝ := 0.5

-- Statement: Prove that the total amount cut off is 0.875 inches
theorem total_hair_cut : first_cut + second_cut = 0.875 :=
by {
  -- The exact proof would go here
  sorry
}

end total_hair_cut_l119_119566


namespace trucks_and_goods_l119_119157

variable (x : ℕ) -- Number of trucks
variable (goods : ℕ) -- Total tons of goods

-- Conditions
def condition1 : Prop := goods = 3 * x + 5
def condition2 : Prop := goods = 4 * (x - 5)

theorem trucks_and_goods (h1 : condition1 x goods) (h2 : condition2 x goods) : x = 25 ∧ goods = 80 :=
by
  sorry

end trucks_and_goods_l119_119157


namespace area_of_shaded_region_l119_119592

def parallelogram_exists (EFGH : Type) : Prop :=
  ∃ (E F G H : EFGH) (EJ JH EH : ℝ) (height : ℝ), EJ + JH = EH ∧ EH = 12 ∧ JH = 8 ∧ height = 10

theorem area_of_shaded_region {EFGH : Type} (h : parallelogram_exists EFGH) : 
  ∃ (area_shaded : ℝ), area_shaded = 100 := 
by
  sorry

end area_of_shaded_region_l119_119592


namespace geom_seq_mult_l119_119347

variable {α : Type*} [LinearOrderedField α]

def is_geom_seq (a : ℕ → α) :=
  ∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)

theorem geom_seq_mult (a : ℕ → α) (h : is_geom_seq a) (hpos : ∀ n, 0 < a n) (h4_8 : a 4 * a 8 = 4) :
  a 5 * a 6 * a 7 = 8 := 
sorry

end geom_seq_mult_l119_119347


namespace lucas_change_l119_119377

-- Define the initial amount of money Lucas has
def initial_amount : ℕ := 20

-- Define the cost of one avocado
def cost_per_avocado : ℕ := 2

-- Define the number of avocados Lucas buys
def number_of_avocados : ℕ := 3

-- Calculate the total cost of avocados
def total_cost : ℕ := number_of_avocados * cost_per_avocado

-- Calculate the remaining amount of money (change)
def remaining_amount : ℕ := initial_amount - total_cost

-- The proposition to prove: Lucas brings home $14
theorem lucas_change : remaining_amount = 14 := by
  sorry

end lucas_change_l119_119377


namespace largest_integer_n_exists_l119_119827

theorem largest_integer_n_exists :
  ∃ (x y z n : ℤ), (x^2 + y^2 + z^2 + 2 * x * y + 2 * y * z + 2 * z * x + 5 * x + 5 * y + 5 * z - 10 = n^2) ∧ (n = 6) :=
by
  sorry

end largest_integer_n_exists_l119_119827


namespace average_visitors_per_day_l119_119079

theorem average_visitors_per_day (avg_sunday : ℕ) (avg_other_day : ℕ) (days_in_month : ℕ) (starts_on_sunday : Bool) :
  avg_sunday = 570 →
  avg_other_day = 240 →
  days_in_month = 30 →
  starts_on_sunday = true →
  (5 * avg_sunday + 25 * avg_other_day) / days_in_month = 295 :=
by
  intros
  sorry

end average_visitors_per_day_l119_119079


namespace accessory_factory_growth_l119_119938

theorem accessory_factory_growth (x : ℝ) :
  600 + 600 * (1 + x) + 600 * (1 + x) ^ 2 = 2180 :=
sorry

end accessory_factory_growth_l119_119938


namespace divisors_log_sum_eq_l119_119191

open BigOperators

/-- Given the sum of the base-10 logarithms of the divisors of \( 10^{2n} = 4752 \), prove that \( n = 12 \). -/
theorem divisors_log_sum_eq (n : ℕ) (h : ∑ a in Finset.range (2*n + 1), ∑ b in Finset.range (2*n + 1), 
  (a * Real.log (2) / Real.log (10) + b * Real.log (5) / Real.log (10)) = 4752) : n = 12 :=
by {
  sorry
}

end divisors_log_sum_eq_l119_119191


namespace scientific_notation_of_0_000815_l119_119810

theorem scientific_notation_of_0_000815 :
  (∃ (c : ℝ) (n : ℤ), 0.000815 = c * 10^n ∧ 1 ≤ c ∧ c < 10) ∧ (0.000815 = 8.15 * 10^(-4)) :=
by
  sorry

end scientific_notation_of_0_000815_l119_119810


namespace ensure_mixed_tablets_l119_119487

theorem ensure_mixed_tablets (A B : ℕ) (total : ℕ) (hA : A = 10) (hB : B = 16) (htotal : total = 18) :
  ∃ (a b : ℕ), a + b = total ∧ a ≤ A ∧ b ≤ B ∧ a > 0 ∧ b > 0 :=
by
  sorry

end ensure_mixed_tablets_l119_119487


namespace remainder_of_division_l119_119797

theorem remainder_of_division (d : ℝ) (q : ℝ) (r : ℝ) : 
  d = 187.46067415730337 → q = 89 → 16698 = (d * q) + r → r = 14 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  simp at h3
  sorry

end remainder_of_division_l119_119797


namespace required_butter_l119_119154

-- Define the given conditions
variables (butter sugar : ℕ)
def recipe_butter : ℕ := 25
def recipe_sugar : ℕ := 125
def used_sugar : ℕ := 1000

-- State the theorem
theorem required_butter (h1 : butter = recipe_butter) (h2 : sugar = recipe_sugar) :
  (used_sugar * recipe_butter) / recipe_sugar = 200 := 
by 
  sorry

end required_butter_l119_119154


namespace growth_rate_double_l119_119031

noncomputable def lake_coverage (days : ℕ) : ℝ := if days = 39 then 1 else if days = 38 then 0.5 else 0  -- Simplified condition statement

theorem growth_rate_double (days : ℕ) : 
  (lake_coverage 39 = 1) → (lake_coverage 38 = 0.5) → (∀ n, lake_coverage (n + 1) = 2 * lake_coverage n) := 
  by 
  intros h39 h38 
  apply sorry  -- Proof not required

end growth_rate_double_l119_119031


namespace total_count_pens_pencils_markers_l119_119529

-- Define the conditions
def ratio_pens_pencils (pens pencils : ℕ) : Prop :=
  6 * pens = 5 * pencils

def nine_more_pencils (pens pencils : ℕ) : Prop :=
  pencils = pens + 9

def ratio_markers_pencils (markers pencils : ℕ) : Prop :=
  3 * markers = 4 * pencils

-- Theorem statement to be proved 
theorem total_count_pens_pencils_markers 
  (pens pencils markers : ℕ) 
  (h1 : ratio_pens_pencils pens pencils)
  (h2 : nine_more_pencils pens pencils)
  (h3 : ratio_markers_pencils markers pencils) : 
  pens + pencils + markers = 171 :=
sorry

end total_count_pens_pencils_markers_l119_119529


namespace shift_right_inverse_exp_eq_ln_l119_119324

variable (f : ℝ → ℝ)

theorem shift_right_inverse_exp_eq_ln :
  (∀ x, f (x - 1) = Real.log x) → ∀ x, f x = Real.log (x + 1) :=
by
  sorry

end shift_right_inverse_exp_eq_ln_l119_119324


namespace percentage_difference_height_l119_119095

-- Define the heights of persons B, A, and C
variables (H_B H_A H_C : ℝ)

-- Condition: Person A's height is 30% less than person B's height
def person_A_height : Prop := H_A = 0.70 * H_B

-- Condition: Person C's height is 20% more than person A's height
def person_C_height : Prop := H_C = 1.20 * H_A

-- The proof problem: Prove that the percentage difference between H_B and H_C is 16%
theorem percentage_difference_height (h1 : person_A_height H_B H_A) (h2 : person_C_height H_A H_C) :
  ((H_B - H_C) / H_B) * 100 = 16 :=
by
  sorry

end percentage_difference_height_l119_119095


namespace transformer_coils_flawless_l119_119144

theorem transformer_coils_flawless (x y : ℕ) (hx : x + y = 8200)
  (hdef : (2 * x / 100) + (3 * y / 100) = 216) :
  ((x = 3000 ∧ y = 5200) ∧ ((x * 98 / 100) = 2940) ∧ ((y * 97 / 100) = 5044)) :=
by
  sorry

end transformer_coils_flawless_l119_119144


namespace IntervalForKTriangleLengths_l119_119688

noncomputable def f (x k : ℝ) := (x^4 + k * x^2 + 1) / (x^4 + x^2 + 1)

theorem IntervalForKTriangleLengths (k : ℝ) :
  (∀ (x : ℝ), 1 ≤ f x k ∧
              (k ≥ 1 → f x k ≤ (k + 2) / 3) ∧ 
              (k < 1 → f x k ≥ (k + 2) / 3)) →
  (∀ (a b c : ℝ), (f a k < f b k + f c k) ∧ 
                  (f b k < f a k + f c k) ∧ 
                  (f c k < f a k + f b k)) ↔ (-1/2 < k ∧ k < 4) :=
by sorry

#check f
#check IntervalForKTriangleLengths

end IntervalForKTriangleLengths_l119_119688


namespace molecular_weight_proof_l119_119532

/-- Atomic weights in atomic mass units (amu) --/
def atomic_weight_Al : ℝ := 26.98
def atomic_weight_O : ℝ := 16.00
def atomic_weight_H : ℝ := 1.01
def atomic_weight_N : ℝ := 14.01
def atomic_weight_P : ℝ := 30.97

/-- Number of atoms in the compound --/
def num_Al : ℝ := 2
def num_O : ℝ := 4
def num_H : ℝ := 6
def num_N : ℝ := 3
def num_P : ℝ := 1

/-- calculating the molecular weight --/
def molecular_weight : ℝ := 
  (num_Al * atomic_weight_Al) +
  (num_O * atomic_weight_O) +
  (num_H * atomic_weight_H) +
  (num_N * atomic_weight_N) +
  (num_P * atomic_weight_P)

-- The proof statement
theorem molecular_weight_proof : molecular_weight = 197.02 := 
by
  sorry

end molecular_weight_proof_l119_119532


namespace total_notebooks_distributed_l119_119503

theorem total_notebooks_distributed :
  ∀ (N C : ℕ), 
    (N / C = C / 8) →
    (N = 16 * (C / 2)) →
    N = 512 := 
by
  sorry

end total_notebooks_distributed_l119_119503


namespace total_amount_spent_l119_119323

def price_per_deck (n : ℕ) : ℝ :=
if n <= 3 then 8 else if n <= 6 then 7 else 6

def promotion_price (price : ℝ) : ℝ :=
price * 0.5

def total_cost (decks_victor decks_friend : ℕ) : ℝ :=
let cost_victor :=
  if decks_victor % 2 = 0 then
    let pairs := decks_victor / 2
    price_per_deck decks_victor * pairs + promotion_price (price_per_deck decks_victor) * pairs
  else sorry
let cost_friend :=
  if decks_friend = 2 then
    price_per_deck decks_friend + promotion_price (price_per_deck decks_friend)
  else sorry
cost_victor + cost_friend

theorem total_amount_spent : total_cost 6 2 = 43.5 := sorry

end total_amount_spent_l119_119323


namespace find_remaining_area_l119_119318

theorem find_remaining_area 
    (base_RST : ℕ) 
    (height_RST : ℕ) 
    (base_RSC : ℕ) 
    (height_RSC : ℕ) 
    (area_RST : ℕ := (1 / 2) * base_RST * height_RST) 
    (area_RSC : ℕ := (1 / 2) * base_RSC * height_RSC) 
    (remaining_area : ℕ := area_RST - area_RSC) 
    (h_base_RST : base_RST = 5) 
    (h_height_RST : height_RST = 4) 
    (h_base_RSC : base_RSC = 1) 
    (h_height_RSC : height_RSC = 4) : 
    remaining_area = 8 := 
by 
  sorry

end find_remaining_area_l119_119318


namespace prod_lcm_gcd_eq_216_l119_119595

theorem prod_lcm_gcd_eq_216 (a b : ℕ) (h1 : a = 12) (h2 : b = 18) :
  (Nat.gcd a b) * (Nat.lcm a b) = 216 := by
  sorry

end prod_lcm_gcd_eq_216_l119_119595


namespace ratio_a_c_l119_119786

-- Define variables and conditions
variables (a b c d : ℚ)

-- Conditions
def ratio_a_b : Prop := a / b = 5 / 4
def ratio_c_d : Prop := c / d = 4 / 3
def ratio_d_b : Prop := d / b = 1 / 5

-- Theorem statement
theorem ratio_a_c (h1 : ratio_a_b a b)
                  (h2 : ratio_c_d c d)
                  (h3 : ratio_d_b d b) : 
  (a / c = 75 / 16) :=
sorry

end ratio_a_c_l119_119786


namespace original_number_is_neg2_l119_119171

theorem original_number_is_neg2 (x : ℚ) (h : 2 - 1/x = 5/2) : x = -2 :=
sorry

end original_number_is_neg2_l119_119171


namespace expression_value_l119_119112

theorem expression_value (b : ℝ) (hb : b = 1 / 3) :
    (3 * b⁻¹ - b⁻¹ / 3) / b^2 = 72 :=
sorry

end expression_value_l119_119112


namespace cost_milk_is_5_l119_119725

-- Define the total cost the baker paid
def total_cost : ℕ := 80

-- Define the cost components
def cost_flour : ℕ := 3 * 3
def cost_eggs : ℕ := 3 * 10
def cost_baking_soda : ℕ := 2 * 3

-- Define the number of liters of milk
def liters_milk : ℕ := 7

-- Define the unknown cost per liter of milk
noncomputable def cost_per_liter_milk (c : ℕ) : Prop :=
  c * liters_milk = total_cost - (cost_flour + cost_eggs + cost_baking_soda)

-- State the theorem we want to prove
theorem cost_milk_is_5 : cost_per_liter_milk 5 := 
by
  sorry

end cost_milk_is_5_l119_119725


namespace nursery_school_students_l119_119433

theorem nursery_school_students (S : ℕ)
  (h1 : ∃ x, x = S / 10)
  (h2 : 20 + (S / 10) = 25) : S = 50 :=
by
  sorry

end nursery_school_students_l119_119433


namespace billy_age_is_45_l119_119326

variable (Billy_age Joe_age : ℕ)

-- Given conditions
def condition1 := Billy_age = 3 * Joe_age
def condition2 := Billy_age + Joe_age = 60
def condition3 := Billy_age > 60 / 2

-- Prove Billy's age is 45
theorem billy_age_is_45 (h1 : condition1 Billy_age Joe_age) (h2 : condition2 Billy_age Joe_age) (h3 : condition3 Billy_age) : Billy_age = 45 :=
by
  sorry

end billy_age_is_45_l119_119326


namespace inequality_proof_l119_119331

theorem inequality_proof (x y : ℝ) (h : x^8 + y^8 ≤ 1) : 
  x^12 - y^12 + 2 * x^6 * y^6 ≤ π / 2 :=
sorry

end inequality_proof_l119_119331


namespace rectangle_width_l119_119614

theorem rectangle_width (length_rect : ℝ) (width_rect : ℝ) (side_square : ℝ)
  (h1 : side_square * side_square = 5 * (length_rect * width_rect))
  (h2 : length_rect = 125)
  (h3 : 4 * side_square = 800) : width_rect = 64 :=
by 
  sorry

end rectangle_width_l119_119614


namespace johns_speed_l119_119968

theorem johns_speed (J : ℝ)
  (lewis_speed : ℝ := 60)
  (distance_AB : ℝ := 240)
  (meet_distance_A : ℝ := 160)
  (time_lewis_to_B : ℝ := distance_AB / lewis_speed)
  (time_lewis_back_80 : ℝ := 80 / lewis_speed)
  (total_time_meet : ℝ := time_lewis_to_B + time_lewis_back_80)
  (total_distance_john_meet : ℝ := J * total_time_meet) :
  total_distance_john_meet = meet_distance_A → J = 30 := 
by
  sorry

end johns_speed_l119_119968


namespace solve_for_c_l119_119361

variables (m c b a : ℚ) -- Declaring variables as rationals for added precision

theorem solve_for_c (h : m = (c * b * a) / (a - c)) : 
  c = (m * a) / (m + b * a) := 
by 
  sorry -- Proof not required as per the instructions

end solve_for_c_l119_119361


namespace equations_of_motion_l119_119102

-- Initial conditions and setup
def omega : ℝ := 10
def OA : ℝ := 90
def AB : ℝ := 90
def AM : ℝ := 45

-- Questions:
-- 1. Equations of motion for point M
-- 2. Equation of the trajectory of point M
-- 3. Velocity of point M

theorem equations_of_motion (t : ℝ) :
  let xM := 45 * (1 + Real.cos (omega * t))
  let yM := 45 * Real.sin (omega * t)
  xM = 45 * (1 + Real.cos (omega * t)) ∧
  yM = 45 * Real.sin (omega * t) ∧
  ((yM / 45) ^ 2 + ((xM - 45) / 45) ^ 2 = 1) ∧
  let vMx := -450 * Real.sin (omega * t)
  let vMy := 450 * Real.cos (omega * t)
  (vMx = -450 * Real.sin (omega * t)) ∧
  (vMy = 450 * Real.cos (omega * t)) :=
by
  sorry

end equations_of_motion_l119_119102


namespace cottonwood_fiber_diameter_in_scientific_notation_l119_119853

theorem cottonwood_fiber_diameter_in_scientific_notation:
  (∃ (a : ℝ) (n : ℤ), 0.0000108 = a * 10 ^ n ∧ 1 ≤ a ∧ a < 10) → (0.0000108 = 1.08 * 10 ^ (-5)) :=
by
  sorry

end cottonwood_fiber_diameter_in_scientific_notation_l119_119853


namespace ice_cream_melting_l119_119543

theorem ice_cream_melting :
  ∀ (r1 r2 : ℝ) (h : ℝ),
    r1 = 3 ∧ r2 = 10 →
    4 / 3 * π * r1^3 = π * r2^2 * h →
    h = 9 / 25 :=
by intros r1 r2 h hcond voldist
   sorry

end ice_cream_melting_l119_119543


namespace time_per_harvest_is_three_months_l119_119534

variable (area : ℕ) (trees_per_m2 : ℕ) (coconuts_per_tree : ℕ) 
variable (price_per_coconut : ℚ) (total_earning_6_months : ℚ)

theorem time_per_harvest_is_three_months 
  (h1 : area = 20) 
  (h2 : trees_per_m2 = 2) 
  (h3 : coconuts_per_tree = 6) 
  (h4 : price_per_coconut = 0.50) 
  (h5 : total_earning_6_months = 240) :
    (6 / (total_earning_6_months / (area * trees_per_m2 * coconuts_per_tree * price_per_coconut)) = 3) := 
  by 
    sorry

end time_per_harvest_is_three_months_l119_119534


namespace smallest_debt_exists_l119_119934

theorem smallest_debt_exists :
  ∃ (p g : ℤ), 50 = 200 * p + 150 * g := by
  sorry

end smallest_debt_exists_l119_119934


namespace correct_random_variable_l119_119618

-- Define the given conditions
def total_white_balls := 5
def total_red_balls := 3
def total_balls := total_white_balls + total_red_balls
def balls_drawn := 3

-- Define the random variable
noncomputable def is_random_variable_correct (option : ℕ) :=
  option = 2

-- The theorem to be proved
theorem correct_random_variable: is_random_variable_correct 2 :=
by
  sorry

end correct_random_variable_l119_119618


namespace time_to_cross_approx_l119_119155

-- Define train length, tunnel length, speed in km/hr, conversion factors, and the final equation
def length_of_train : ℕ := 415
def length_of_tunnel : ℕ := 285
def speed_in_kmph : ℕ := 63
def km_to_m : ℕ := 1000
def hr_to_sec : ℕ := 3600

-- Convert speed to m/s
def speed_in_mps : ℚ := (speed_in_kmph * km_to_m) / hr_to_sec

-- Calculate total distance
def total_distance : ℕ := length_of_train + length_of_tunnel

-- Calculate the time to cross the tunnel in seconds
def time_to_cross : ℚ := total_distance / speed_in_mps

theorem time_to_cross_approx : abs (time_to_cross - 40) < 0.1 :=
sorry

end time_to_cross_approx_l119_119155


namespace obtuse_and_acute_angles_in_convex_octagon_l119_119739

theorem obtuse_and_acute_angles_in_convex_octagon (m n : ℕ) (h₀ : n + m = 8) : m > n :=
sorry

end obtuse_and_acute_angles_in_convex_octagon_l119_119739


namespace part1_part2_l119_119076

theorem part1 (m : ℝ) (P : ℝ × ℝ) : (P = (3*m - 6, m + 1)) → (P.1 = 0) → (P = (0, 3)) :=
by
  sorry

theorem part2 (m : ℝ) (A P : ℝ × ℝ) : A = (1, -2) → (P = (3*m - 6, m + 1)) → (P.2 = A.2) → (P = (-15, -2)) :=
by
  sorry

end part1_part2_l119_119076


namespace baseball_card_ratio_l119_119726

-- Define the conditions
variable (T : ℤ) -- Number of baseball cards on Tuesday

-- Given conditions
-- On Monday, Buddy has 30 baseball cards
def monday_cards : ℤ := 30

-- On Wednesday, Buddy has T + 12 baseball cards
def wednesday_cards : ℤ := T + 12

-- On Thursday, Buddy buys a third of what he had on Tuesday
def thursday_additional_cards : ℤ := T / 3

-- Total number of cards on Thursday is 32
def thursday_cards (T : ℤ) : ℤ := T + 12 + T / 3

-- We are given that Buddy has 32 baseball cards on Thursday
axiom thursday_total : thursday_cards T = 32

-- The theorem we want to prove: the ratio of Tuesday's to Monday's cards is 1:2
theorem baseball_card_ratio
  (T : ℤ)
  (htotal : thursday_cards T = 32)
  (hmon : monday_cards = 30) :
  T = 15 ∧ (T : ℚ) / monday_cards = 1 / 2 := by
  -- Proof goes here
  sorry

end baseball_card_ratio_l119_119726


namespace riley_mistakes_l119_119983

theorem riley_mistakes :
  ∃ R O : ℕ, R + O = 17 ∧ O = 35 - ((35 - R) / 2 + 5) ∧ R = 3 := by
  sorry

end riley_mistakes_l119_119983


namespace find_ab_average_l119_119590

variable (a b c k : ℝ)

-- Conditions
def sum_condition : Prop := (4 + 6 + 8 + 12 + a + b + c) / 7 = 20
def abc_condition : Prop := a + b + c = 3 * ((4 + 6 + 8) / 3)

-- Theorem
theorem find_ab_average 
  (sum_cond : sum_condition a b c) 
  (abc_cond : abc_condition a b c) 
  (c_eq_k : c = k) : 
  (a + b) / 2 = (18 - k) / 2 :=
sorry  -- Proof is omitted


end find_ab_average_l119_119590


namespace rectangle_area_constant_l119_119071

noncomputable def k (d : ℝ) : ℝ :=
  let x := d / Real.sqrt 29
  10 / 29

theorem rectangle_area_constant (d : ℝ) : 
  let k := 10 / 29
  let length := 5 * (d / Real.sqrt 29)
  let width := 2 * (d / Real.sqrt 29)
  let diagonal := d
  let area := length * width
  area = k * d^2 :=
by
  sorry

end rectangle_area_constant_l119_119071


namespace chessboard_problem_proof_l119_119500

variable (n : ℕ)

noncomputable def chessboard_problem : Prop :=
  ∀ (colors : Fin (2 * n) → Fin (2 * n) → Fin n),
  ∃ i₁ i₂ j₁ j₂,
    i₁ ≠ i₂ ∧
    j₁ ≠ j₂ ∧
    colors i₁ j₁ = colors i₁ j₂ ∧
    colors i₂ j₁ = colors i₂ j₂

/-- Given a 2n x 2n chessboard colored with n colors, there exist 2 tiles in either the same column 
or row such that if the colors of both tiles are swapped, then there exists a rectangle where all 
its four corner tiles have the same color. -/
theorem chessboard_problem_proof (n : ℕ) : chessboard_problem n :=
sorry

end chessboard_problem_proof_l119_119500


namespace problem_statement_l119_119262

theorem problem_statement (a b : ℝ) (h1 : a^3 - b^3 = 2) (h2 : a^5 - b^5 ≥ 4) : a^2 + b^2 ≥ 2 := 
sorry

end problem_statement_l119_119262


namespace fraction_of_students_received_As_l119_119978

/-- Assume A is the fraction of students who received A's,
and B is the fraction of students who received B's,
and T is the total fraction of students who received either A's or B's. -/
theorem fraction_of_students_received_As
  (A B T : ℝ)
  (hB : B = 0.2)
  (hT : T = 0.9)
  (h : A + B = T) :
  A = 0.7 := 
by
  -- establishing the proof steps
  sorry

end fraction_of_students_received_As_l119_119978


namespace find_divisor_l119_119773

-- Define the problem specifications
def divisor_problem (D Q R d : ℕ) : Prop :=
  D = d * Q + R

-- The specific instance with given values
theorem find_divisor :
  divisor_problem 15968 89 37 179 :=
by
  -- Proof omitted
  sorry

end find_divisor_l119_119773


namespace beatrice_tv_ratio_l119_119021

theorem beatrice_tv_ratio (T1 T2 T Ttotal : ℕ)
  (h1 : T1 = 8)
  (h2 : T2 = 10)
  (h_total : Ttotal = 42)
  (h_T : T = Ttotal - T1 - T2) :
  (T / gcd T T1, T1 / gcd T T1) = (3, 1) :=
by {
  sorry
}

end beatrice_tv_ratio_l119_119021


namespace number_of_blue_pens_minus_red_pens_is_seven_l119_119454

-- Define the problem conditions in Lean
variable (R B K T : ℕ) -- where R is red pens, B is black pens, K is blue pens, T is total pens

-- Define the hypotheses from the problem conditions
def hypotheses :=
  (R = 8) ∧ 
  (B = R + 10) ∧ 
  (T = 41) ∧ 
  (T = R + B + K)

-- Define the theorem we need to prove based on the question and the correct answer
theorem number_of_blue_pens_minus_red_pens_is_seven : 
  hypotheses R B K T → K - R = 7 :=
by 
  intro h
  sorry

end number_of_blue_pens_minus_red_pens_is_seven_l119_119454


namespace rattlesnake_tail_percentage_difference_l119_119518

-- Definitions for the problem
def eastern_segments : Nat := 6
def western_segments : Nat := 8

-- The statement to prove
theorem rattlesnake_tail_percentage_difference :
  100 * (western_segments - eastern_segments) / western_segments = 25 := by
  sorry

end rattlesnake_tail_percentage_difference_l119_119518


namespace area_of_folded_shape_is_two_units_squared_l119_119761

/-- 
A square piece of paper with each side of length 2 units is divided into 
four equal squares along both its length and width. From the top left corner to 
bottom right corner, a line is drawn through the center dividing the square diagonally.
The paper is folded along this line to form a new shape.
We prove that the area of the folded shape is 2 units².
-/
theorem area_of_folded_shape_is_two_units_squared
  (side_len : ℝ)
  (area_original : ℝ)
  (area_folded : ℝ)
  (h1 : side_len = 2)
  (h2 : area_original = side_len * side_len)
  (h3 : area_folded = area_original / 2) :
  area_folded = 2 := by
  -- Place proof here
  sorry

end area_of_folded_shape_is_two_units_squared_l119_119761


namespace quadratic_inequality_solution_range_l119_119953

theorem quadratic_inequality_solution_range (a : ℝ) :
  (∀ x : ℝ, 2 * x^2 + (a - 1) * x + 1 / 2 > 0) ↔ (-1 < a ∧ a < 3) :=
by
  sorry

end quadratic_inequality_solution_range_l119_119953


namespace evaluate_expression_at_one_l119_119367

theorem evaluate_expression_at_one : 
  (4 + (4 + x^2) / x) / ((x + 2) / x) = 3 := by
  sorry

end evaluate_expression_at_one_l119_119367


namespace find_number_l119_119287

theorem find_number (x : ℝ) : (5 / 3) * x = 45 → x = 27 := by
  sorry

end find_number_l119_119287


namespace range_of_f3_l119_119712

noncomputable def f (a b x : ℝ) : ℝ := a * x ^ 2 + b * x + 1

theorem range_of_f3 {a b : ℝ}
  (h1 : -2 ≤ a - b ∧ a - b ≤ 0) 
  (h2 : -3 ≤ 4 * a + 2 * b ∧ 4 * a + 2 * b ≤ 1) :
  -7 ≤ f a b 3 ∧ f a b 3 ≤ 3 :=
sorry

end range_of_f3_l119_119712


namespace tan_sum_l119_119125

theorem tan_sum (x y : ℝ)
  (h1 : Real.sin x + Real.sin y = 72 / 65)
  (h2 : Real.cos x + Real.cos y = 96 / 65) : 
  Real.tan x + Real.tan y = 868 / 112 := 
by sorry

end tan_sum_l119_119125


namespace exist_projections_l119_119724

-- Define types for lines and points
variable {Point : Type} [MetricSpace Point]

-- Define the projection operator
def projection (t_i t_j : Set Point) (p : Point) : Point := 
  sorry -- projection definition will go here

-- Define t1, t2, ..., tk
variables (t : ℕ → Set Point) (k : ℕ)
  (hk : k > 1)  -- condition: k > 1
  (ht_distinct : ∀ i j, i ≠ j → t i ≠ t j)  -- condition: different lines

-- Define the proposition
theorem exist_projections : 
  ∃ (P : ℕ → Point), 
    (∀ i, 1 ≤ i ∧ i < k → P (i + 1) = projection (t i) (t (i + 1)) (P i)) ∧ 
    P 1 = projection (t k) (t 1) (P k) :=
sorry

end exist_projections_l119_119724


namespace count_multiples_4_6_not_5_9_l119_119400

/-- The number of integers between 1 and 500 that are multiples of both 4 and 6 but not of either 5 or 9 is 22. -/
theorem count_multiples_4_6_not_5_9 :
  let lcm_4_6 := (Nat.lcm 4 6)
  let lcm_4_6_5 := (Nat.lcm lcm_4_6 5)
  let lcm_4_6_9 := (Nat.lcm lcm_4_6 9)
  let lcm_4_6_5_9 := (Nat.lcm lcm_4_6_5 9)
  let count_multiples (x : Nat) := (500 / x)
  count_multiples lcm_4_6 - count_multiples lcm_4_6_5 - count_multiples lcm_4_6_9 + count_multiples lcm_4_6_5_9 = 22 :=
by
  let lcm_4_6 := (Nat.lcm 4 6)
  let lcm_4_6_5 := (Nat.lcm lcm_4_6 5)
  let lcm_4_6_9 := (Nat.lcm lcm_4_6 9)
  let lcm_4_6_5_9 := (Nat.lcm lcm_4_6_5 9)
  let count_multiples (x : Nat) := (500 / x)
  show count_multiples lcm_4_6 - count_multiples lcm_4_6_5 - count_multiples lcm_4_6_9 + count_multiples lcm_4_6_5_9 = 22
  sorry

end count_multiples_4_6_not_5_9_l119_119400


namespace infinite_solutions_x2_y2_z2_x3_y3_z3_l119_119961

-- Define the parametric forms
def param_x (k : ℤ) := k * (2 * k^2 + 1)
def param_y (k : ℤ) := 2 * k^2 + 1
def param_z (k : ℤ) := -k * (2 * k^2 + 1)

-- Prove the equation
theorem infinite_solutions_x2_y2_z2_x3_y3_z3 :
  ∀ k : ℤ, param_x k ^ 2 + param_y k ^ 2 + param_z k ^ 2 = param_x k ^ 3 + param_y k ^ 3 + param_z k ^ 3 :=
by
  intros k
  -- Calculation needs to be proved here, we place a placeholder for now
  sorry

end infinite_solutions_x2_y2_z2_x3_y3_z3_l119_119961


namespace sequence_geometric_condition_l119_119824

theorem sequence_geometric_condition
  (a : ℕ → ℤ)
  (p q : ℤ)
  (h1 : a 1 = -1)
  (h2 : ∀ n, a (n + 1) = 2 * (a n - n + 3))
  (h3 : ∀ n, (a (n + 1) - p * (n + 1) + q) = 2 * (a n - p * n + q)) :
  a (Int.natAbs (p + q)) = 40 :=
sorry

end sequence_geometric_condition_l119_119824


namespace carla_zoo_l119_119884

theorem carla_zoo (zebras camels monkeys giraffes : ℕ) 
  (hz : zebras = 12)
  (hc : camels = zebras / 2)
  (hm : monkeys = 4 * camels)
  (hg : giraffes = 2) : 
  monkeys - giraffes = 22 := by sorry

end carla_zoo_l119_119884


namespace prove_inequality_l119_119657

theorem prove_inequality (x : ℝ) (h : x > 2) : x + 1 / (x - 2) ≥ 4 :=
  sorry

end prove_inequality_l119_119657


namespace sphere_surface_area_l119_119567

theorem sphere_surface_area (V : ℝ) (hV : V = 72 * Real.pi) : 
  ∃ S : ℝ, S = 36 * 2^(2/3) * Real.pi :=
by
  sorry

end sphere_surface_area_l119_119567


namespace fraction_calculation_l119_119691

theorem fraction_calculation : (8 / 24) - (5 / 72) + (3 / 8) = 23 / 36 :=
by
  sorry

end fraction_calculation_l119_119691


namespace water_consumption_correct_l119_119300

theorem water_consumption_correct (w n r : ℝ) 
  (hw : w = 21428) 
  (hn : n = 26848.55) 
  (hr : r = 302790.13) :
  w = 21428 ∧ n = 26848.55 ∧ r = 302790.13 :=
by 
  sorry

end water_consumption_correct_l119_119300


namespace frac_x_y_value_l119_119140

theorem frac_x_y_value (x y : ℝ) (h1 : 3 < (2 * x - y) / (x + 2 * y))
(h2 : (2 * x - y) / (x + 2 * y) < 7) (h3 : ∃ (t : ℤ), x = t * y) : x / y = -4 := by
  sorry

end frac_x_y_value_l119_119140


namespace distance_walked_on_third_day_l119_119342

theorem distance_walked_on_third_day:
  ∃ x : ℝ, 
    4 * x + 2 * x + x + (1 / 2) * x + (1 / 4) * x + (1 / 8) * x = 378 ∧
    x = 48 := 
by
  sorry

end distance_walked_on_third_day_l119_119342


namespace max_sum_arithmetic_sequence_l119_119988

theorem max_sum_arithmetic_sequence (n : ℕ) (M : ℝ) (hM : 0 < M) 
  (a : ℕ → ℝ) (h_arith_seq : ∀ k, a (k + 1) - a k = a 1 - a 0) 
  (h_constraint : a 1 ^ 2 + a (n + 1) ^ 2 ≤ M) :
  ∃ S, S = (n + 1) * (Real.sqrt (10 * M)) / 2 :=
sorry

end max_sum_arithmetic_sequence_l119_119988


namespace daily_rate_problem_l119_119011

noncomputable def daily_rate : ℝ := 126.19 -- Correct answer

theorem daily_rate_problem
  (days : ℕ := 14)
  (pet_fee : ℝ := 100)
  (service_fee_rate : ℝ := 0.20)
  (security_deposit : ℝ := 1110)
  (deposit_rate : ℝ := 0.50)
  (x : ℝ) : x = daily_rate :=
by
  have total_cost := days * x + pet_fee + service_fee_rate * (days * x)
  have total_cost_with_fees := days * x * (1 + service_fee_rate) + pet_fee
  have security_deposit_cost := deposit_rate * total_cost_with_fees
  have eq_security : security_deposit_cost = security_deposit := sorry
  sorry

end daily_rate_problem_l119_119011


namespace solve_for_x_l119_119128

def f (x : ℝ) : ℝ := 2 * x - 3

theorem solve_for_x : ∃ (x : ℝ), 2 * (f x) - 11 = f (x - 2) :=
by
  use 5
  have h1 : f 5 = 2 * 5 - 3 := rfl
  have h2 : f (5 - 2) = 2 * (5 - 2) - 3 := rfl
  simp [f] at *
  exact sorry

end solve_for_x_l119_119128


namespace find_quadratic_eq_l119_119020

theorem find_quadratic_eq (x y : ℝ) (hx : x + y = 10) (hy : |x - y| = 12) :
    ∃ a b c : ℝ, a = 1 ∧ b = -10 ∧ c = -11 ∧ (x^2 + b * x + c = 0) ∧ (y^2 + b * y + c = 0) := by
  sorry

end find_quadratic_eq_l119_119020


namespace rainfall_difference_l119_119650

theorem rainfall_difference :
  let day1 := 26
  let day2 := 34
  let day3 := day2 - 12
  let total_rainfall := day1 + day2 + day3
  let average_rainfall := 140
  (average_rainfall - total_rainfall = 58) :=
by
  sorry

end rainfall_difference_l119_119650


namespace intersection_of_A_and_B_l119_119337

open Set

def A : Set Int := {x | x + 2 = 0}

def B : Set Int := {x | x^2 - 4 = 0}

theorem intersection_of_A_and_B : A ∩ B = {-2} :=
by
  sorry

end intersection_of_A_and_B_l119_119337


namespace central_angle_of_sector_l119_119408

theorem central_angle_of_sector {r l : ℝ} 
  (h1 : 2 * r + l = 4) 
  (h2 : (1 / 2) * l * r = 1) : 
  l / r = 2 :=
by 
  sorry

end central_angle_of_sector_l119_119408


namespace negation_example_l119_119713

theorem negation_example : 
  (¬ ∃ x_0 : ℚ, x_0 - 2 = 0) = (∀ x : ℚ, x - 2 ≠ 0) :=
by 
  sorry

end negation_example_l119_119713


namespace minimum_routes_A_C_l119_119407

namespace SettlementRoutes

-- Define three settlements A, B, and C
variable (A B C : Type)

-- Assume there are more than one roads connecting each settlement pair directly
variable (k m n : ℕ) -- k: roads between A and B, m: roads between B and C, n: roads between A and C

-- Conditions: Total paths including intermediate nodes
axiom h1 : k + m * n = 34
axiom h2 : m + k * n = 29

-- Theorem: Minimum number of routes connecting A and C is 26
theorem minimum_routes_A_C : ∃ n k m : ℕ, k + m * n = 34 ∧ m + k * n = 29 ∧ n + k * m = 26 := sorry

end SettlementRoutes

end minimum_routes_A_C_l119_119407


namespace inequality_solution_l119_119569

theorem inequality_solution : {x : ℝ | -2 < (x^2 - 12 * x + 20) / (x^2 - 4 * x + 8) ∧ (x^2 - 12 * x + 20) / (x^2 - 4 * x + 8) < 2} = {x : ℝ | 5 < x} := 
sorry

end inequality_solution_l119_119569


namespace part1_part2_l119_119632

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (Real.exp x / x) - Real.log x + x - a

theorem part1 (a : ℝ) : (∀ x > 0, f x a ≥ 0) ↔ a ≤ Real.exp 1 + 1 :=
  sorry

theorem part2 (a : ℝ) (x1 x2 : ℝ) (h1 : f x1 a = 0) (h2 : f x2 a = 0) : x1 * x2 < 1 :=
  sorry

end part1_part2_l119_119632


namespace contrapositive_l119_119013

-- Definitions based on the conditions
def original_proposition (a b : ℝ) : Prop := a^2 + b^2 = 0 → a = 0 ∧ b = 0

-- The theorem to prove the contrapositive
theorem contrapositive (a b : ℝ) : original_proposition a b ↔ (¬(a = 0 ∧ b = 0) → a^2 + b^2 ≠ 0) :=
sorry

end contrapositive_l119_119013


namespace find_sin_angle_BAD_l119_119439

def isosceles_right_triangle (A B C : ℝ → ℝ → Prop) (AB BC AC : ℝ) : Prop :=
  AB = 2 ∧ BC = 2 ∧ AC = 2 * Real.sqrt 2

def right_triangle_on_hypotenuse (A C D : ℝ → ℝ → Prop) (AC CD DA : ℝ) (DAC : ℝ) : Prop :=
  AC = 2 * Real.sqrt 2 ∧ CD = DA / 2 ∧ DAC = Real.pi / 6

def equal_perimeters (AC CD DA : ℝ) : Prop := 
  AC + CD + DA = 4 + 2 * Real.sqrt 2

theorem find_sin_angle_BAD :
  ∀ (A B C D : ℝ → ℝ → Prop) (AB BC AC CD DA : ℝ),
  isosceles_right_triangle A B C AB BC AC →
  right_triangle_on_hypotenuse A C D AC CD DA (Real.pi / 6) →
  equal_perimeters AC CD DA →
  Real.sin (2 * (Real.pi / 4 + Real.pi / 6)) = 1 / 2 :=
by
  intros
  sorry

end find_sin_angle_BAD_l119_119439


namespace second_odd_integer_is_72_l119_119904

def consecutive_odd_integers (n : ℤ) : ℤ × ℤ × ℤ :=
  (n - 2, n, n + 2)

theorem second_odd_integer_is_72 (n : ℤ) (h : (n - 2) + (n + 2) = 144) : n = 72 :=
by {
  sorry
}

end second_odd_integer_is_72_l119_119904


namespace linear_regression_change_l119_119067

theorem linear_regression_change (x : ℝ) :
  let y1 := 2 - 1.5 * x
  let y2 := 2 - 1.5 * (x + 1)
  y2 - y1 = -1.5 := by
  -- y1 = 2 - 1.5 * x
  -- y2 = 2 - 1.5 * x - 1.5
  -- Δ y = y2 - y1
  sorry

end linear_regression_change_l119_119067


namespace initial_price_of_article_l119_119138

theorem initial_price_of_article (P : ℝ) (h : 0.4025 * P = 620) : P = 620 / 0.4025 :=
by
  sorry

end initial_price_of_article_l119_119138


namespace chess_team_selection_l119_119587

theorem chess_team_selection:
  let boys := 10
  let girls := 12
  let team_size := 8     -- total team size
  let boys_selected := 5 -- number of boys to select
  let girls_selected := 3 -- number of girls to select
  ∃ (w : ℕ), 
  (w = Nat.choose boys boys_selected * Nat.choose girls girls_selected) ∧ 
  w = 55440 :=
by
  sorry

end chess_team_selection_l119_119587


namespace swim_club_members_l119_119358

theorem swim_club_members (X : ℝ) 
  (h1 : 0.30 * X = 0.30 * X)
  (h2 : 0.70 * X = 42) : X = 60 :=
sorry

end swim_club_members_l119_119358


namespace total_matches_in_2006_world_cup_l119_119699

-- Define relevant variables and conditions
def teams := 32
def groups := 8
def top2_from_each_group := 16

-- Calculate the number of matches in Group Stage
def matches_in_group_stage :=
  let matches_per_group := 6
  matches_per_group * groups

-- Calculate the number of matches in Knockout Stage
def matches_in_knockout_stage :=
  let first_round_matches := 8
  let quarter_final_matches := 4
  let semi_final_matches := 2
  let final_and_third_place_matches := 2
  first_round_matches + quarter_final_matches + semi_final_matches + final_and_third_place_matches

-- Total number of matches
theorem total_matches_in_2006_world_cup : matches_in_group_stage + matches_in_knockout_stage = 64 := by
  sorry

end total_matches_in_2006_world_cup_l119_119699


namespace intersection_of_lines_l119_119292

theorem intersection_of_lines :
  ∃ x y : ℚ, 3 * y = -2 * x + 6 ∧ 2 * y = 6 * x - 4 ∧ x = 12 / 11 ∧ y = 14 / 11 := by
  sorry

end intersection_of_lines_l119_119292


namespace equation_of_hyperbola_l119_119019

variable (a b c : ℝ)
variable (x y : ℝ)

theorem equation_of_hyperbola :
  (0 < a) ∧ (0 < b) ∧ (c / a = Real.sqrt 3) ∧ (a^2 / c = 1) ∧ (c = 3) ∧ (b = Real.sqrt 6)
  → (x^2 / 3 - y^2 / 6 = 1) :=
by
  sorry

end equation_of_hyperbola_l119_119019


namespace trig_expression_value_l119_119073

open Real

theorem trig_expression_value (x : ℝ) (h : tan (π - x) = -2) : 
  4 * sin x ^ 2 - 3 * sin x * cos x - 5 * cos x ^ 2 = 1 := 
sorry

end trig_expression_value_l119_119073


namespace carl_max_value_l119_119616

-- Definitions based on problem conditions.
def value_of_six_pound_rock : ℕ := 20
def weight_of_six_pound_rock : ℕ := 6
def value_of_three_pound_rock : ℕ := 9
def weight_of_three_pound_rock : ℕ := 3
def value_of_two_pound_rock : ℕ := 4
def weight_of_two_pound_rock : ℕ := 2
def max_weight_carl_can_carry : ℕ := 24

/-- Proves that Carl can carry rocks worth maximum 80 dollars given the conditions. -/
theorem carl_max_value : ∃ (n m k : ℕ),
    n * weight_of_six_pound_rock + m * weight_of_three_pound_rock + k * weight_of_two_pound_rock ≤ max_weight_carl_can_carry ∧
    n * value_of_six_pound_rock + m * value_of_three_pound_rock + k * value_of_two_pound_rock = 80 :=
by
  sorry

end carl_max_value_l119_119616


namespace correct_option_b_l119_119572

theorem correct_option_b (a : ℝ) : (-2 * a ^ 4) ^ 3 = -8 * a ^ 12 :=
sorry

end correct_option_b_l119_119572


namespace fraction_increase_invariance_l119_119692

theorem fraction_increase_invariance (x y : ℝ) :
  (3 * (2 * y)) / (2 * x + 2 * y) = 3 * y / (x + y) :=
by
  sorry

end fraction_increase_invariance_l119_119692


namespace melanie_total_plums_l119_119545

namespace Melanie

def initial_plums : ℝ := 7.0
def plums_given_by_sam : ℝ := 3.0

theorem melanie_total_plums : initial_plums + plums_given_by_sam = 10.0 :=
by
  sorry

end Melanie

end melanie_total_plums_l119_119545


namespace car_speed_reduction_and_increase_l119_119576

theorem car_speed_reduction_and_increase (V x : ℝ)
  (h1 : V > 0) -- V is positive
  (h2 : V * (1 - x / 100) * (1 + 0.5 * x / 100) = V * (1 - 0.6 * x / 100)) :
  x = 20 :=
sorry

end car_speed_reduction_and_increase_l119_119576


namespace boy_reaches_early_l119_119180

-- Given conditions
def usual_time : ℚ := 42
def rate_multiplier : ℚ := 7 / 6

-- Derived variables
def new_time : ℚ := (6 / 7) * usual_time
def early_time : ℚ := usual_time - new_time

-- The statement to prove
theorem boy_reaches_early : early_time = 6 := by
  sorry

end boy_reaches_early_l119_119180


namespace problem_solution_l119_119625

theorem problem_solution
  (x : ℝ) (a b : ℕ) (hx_pos : 0 < x) (ha_pos : 0 < a) (hb_pos : 0 < b)
  (h_eq : x ^ 2 + 5 * x + 5 / x + 1 / x ^ 2 = 40)
  (h_form : x = a + Real.sqrt b) :
  a + b = 11 :=
sorry

end problem_solution_l119_119625


namespace area_fraction_above_line_l119_119123

-- Define the points of the rectangle
def A := (2,0)
def B := (7,0)
def C := (7,4)
def D := (2,4)

-- Define the points used for the line
def P := (2,1)
def Q := (7,3)

-- The area of the rectangle
def rect_area := (7 - 2) * 4

-- The fraction of the area of the rectangle above the line
theorem area_fraction_above_line : 
  ∀ A B C D P Q, 
    A = (2,0) → B = (7,0) → C = (7,4) → D = (2,4) →
    P = (2,1) → Q = (7,3) →
    (rect_area = 20) → 1 - ((1/2) * 5 * 2 / 20) = 3 / 4 :=
by
  intros A B C D P Q
  intros hA hB hC hD hP hQ h_area
  sorry

end area_fraction_above_line_l119_119123


namespace solve_fraction_eq_l119_119955

theorem solve_fraction_eq (x : ℝ) 
  (h₁ : x ≠ -9) 
  (h₂ : x ≠ -7) 
  (h₃ : x ≠ -10) 
  (h₄ : x ≠ -6) 
  (h₅ : 1 / (x + 9) + 1 / (x + 7) = 1 / (x + 10) + 1 / (x + 6)) : 
  x = -8 := 
sorry

end solve_fraction_eq_l119_119955


namespace sum_of_squares_of_roots_l119_119351

theorem sum_of_squares_of_roots : 
  ∀ r1 r2 : ℝ, (r1 + r2 = 10) → (r1 * r2 = 9) → (r1 > 5 ∨ r2 > 5) → (r1^2 + r2^2 = 82) :=
by
  intros r1 r2 h1 h2 h3
  sorry

end sum_of_squares_of_roots_l119_119351


namespace first_train_speed_l119_119329

theorem first_train_speed:
  ∃ v : ℝ, 
    (∀ t : ℝ, t = 1 → (v * t) + (4 * v) = 200) ∧ 
    (∀ t : ℝ, t = 4 → 50 * t = 200) → 
    v = 40 :=
by {
 sorry
}

end first_train_speed_l119_119329


namespace picture_frame_length_l119_119384

theorem picture_frame_length (h : ℕ) (l : ℕ) (P : ℕ) (h_eq : h = 12) (P_eq : P = 44) (perimeter_eq : P = 2 * (l + h)) : l = 10 :=
by
  -- proof would go here
  sorry

end picture_frame_length_l119_119384


namespace total_number_of_bricks_l119_119771

/-- Given bricks of volume 80 unit cubes and 42 unit cubes,
 and a box of volume 1540 unit cubes,
 prove the total number of bricks that can fill the box exactly is 24. -/
theorem total_number_of_bricks (x y : ℕ) (vol_a vol_b total_vol : ℕ)
  (vol_a_def : vol_a = 80)
  (vol_b_def : vol_b = 42)
  (total_vol_def : total_vol = 1540)
  (volume_filled : x * vol_a + y * vol_b = total_vol) :
  x + y = 24 :=
  sorry

end total_number_of_bricks_l119_119771


namespace multiple_of_distance_l119_119301

namespace WalkProof

variable (H R M : ℕ)

/-- Rajesh walked 10 kilometers less than a certain multiple of the distance that Hiro walked. 
    Together they walked 25 kilometers. Rajesh walked 18 kilometers. 
    Prove that the multiple of the distance Hiro walked that Rajesh walked less than is 4. -/
theorem multiple_of_distance (h1 : R = M * H - 10) 
                             (h2 : H + R = 25)
                             (h3 : R = 18) :
                             M = 4 :=
by
  sorry

end WalkProof

end multiple_of_distance_l119_119301


namespace unique_alphabets_count_l119_119497

theorem unique_alphabets_count
  (total_alphabets : ℕ)
  (each_written_times : ℕ)
  (total_written : total_alphabets * each_written_times = 10) :
  total_alphabets = 5 := by
  -- The proof would be filled in here.
  sorry

end unique_alphabets_count_l119_119497


namespace initial_balls_count_l119_119686

variables (y w : ℕ)

theorem initial_balls_count (h1 : y = 2 * (w - 10)) (h2 : w - 10 = 5 * (y - 9)) :
  y = 10 ∧ w = 15 :=
sorry

end initial_balls_count_l119_119686


namespace sum_of_coefficients_l119_119263

-- Definition of the binomial coefficient
def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem sum_of_coefficients (n : ℕ) (hn1 : 5 < n) (hn2 : n < 7)
  (coeff_cond : binom n 3 > binom n 2 ∧ binom n 3 > binom n 4) :
  (1 + 1)^n = 64 :=
by
  have h : n = 6 :=
    by sorry -- provided conditions force n to be 6
  show 2^n = 64
  rw [h]
  exact rfl

end sum_of_coefficients_l119_119263


namespace page_mistakenly_added_twice_l119_119374

theorem page_mistakenly_added_twice (n k: ℕ) (h₁: n = 77) (h₂: (n * (n + 1)) / 2 + k = 3050) : k = 47 :=
by
  -- sorry here to indicate the proof is not needed
  sorry

end page_mistakenly_added_twice_l119_119374


namespace tree_growth_rate_l119_119217

-- Given conditions
def currentHeight : ℝ := 52
def futureHeightInches : ℝ := 1104
def oneFootInInches : ℝ := 12
def years : ℝ := 8

-- Prove the yearly growth rate in feet
theorem tree_growth_rate:
  (futureHeightInches / oneFootInInches - currentHeight) / years = 5 := 
by
  sorry

end tree_growth_rate_l119_119217


namespace number_of_students_l119_119267

theorem number_of_students (N T : ℕ) (h1 : T = 80 * N)
  (h2 : (T - 100) / (N - 5) = 90) : N = 35 := 
by 
  sorry

end number_of_students_l119_119267


namespace hyperbola_eccentricity_l119_119257

theorem hyperbola_eccentricity : 
  (∃ (a b : ℝ), (a^2 = 1 ∧ b^2 = 2) ∧ ∀ e : ℝ, e = Real.sqrt (1 + b^2 / a^2) → e = Real.sqrt 3) :=
by 
  sorry

end hyperbola_eccentricity_l119_119257


namespace intersection_A_B_intersection_A_complementB_l119_119024

-- Definitions of the sets A and B
def setA : Set ℝ := { x | -5 ≤ x ∧ x ≤ 3 }
def setB : Set ℝ := { x | x < -2 ∨ x > 4 }

-- Proof problem 1: A ∩ B = { x | -5 ≤ x < -2 }
theorem intersection_A_B:
  setA ∩ setB = { x : ℝ | -5 ≤ x ∧ x < -2 } :=
sorry

-- Definition of the complement of B
def complB : Set ℝ := { x | -2 ≤ x ∧ x ≤ 4 }

-- Proof problem 2: A ∩ (complB) = { x | -2 ≤ x ≤ 3 }
theorem intersection_A_complementB:
  setA ∩ complB = { x : ℝ | -2 ≤ x ∧ x ≤ 3 } :=
sorry

end intersection_A_B_intersection_A_complementB_l119_119024


namespace sector_area_correct_l119_119743

-- Define the initial conditions
def arc_length := 4 -- Length of the arc in cm
def central_angle := 2 -- Central angle in radians
def radius := arc_length / central_angle -- Radius of the circle

-- Define the formula for the area of the sector
def sector_area := (1 / 2) * radius * arc_length

-- The statement of our theorem
theorem sector_area_correct : sector_area = 4 := by
  -- Proof goes here
  sorry

end sector_area_correct_l119_119743


namespace smallest_x_for_M_squared_l119_119117

theorem smallest_x_for_M_squared (M x : ℤ) (h1 : 540 = 2^2 * 3^3 * 5) (h2 : 540 * x = M^2) (h3 : x > 0) : x = 15 :=
sorry

end smallest_x_for_M_squared_l119_119117


namespace max_a_plus_b_cubed_plus_c_fourth_l119_119431

theorem max_a_plus_b_cubed_plus_c_fourth (a b c : ℕ) (h : a > 0 ∧ b > 0 ∧ c > 0) (h_sum : a + b + c = 2) :
  a + b^3 + c^4 ≤ 2 := sorry

end max_a_plus_b_cubed_plus_c_fourth_l119_119431


namespace stock_price_is_500_l119_119816

-- Conditions
def income : ℝ := 1000
def dividend_rate : ℝ := 0.50
def investment : ℝ := 10000
def face_value : ℝ := 100

-- Theorem Statement
theorem stock_price_is_500 : 
  (dividend_rate * face_value / (investment / 1000)) = 500 := by
  sorry

end stock_price_is_500_l119_119816


namespace triangle_side_inequality_l119_119596

theorem triangle_side_inequality (a b c : ℝ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : 1 = 1 / 2 * b * c) : b ≥ Real.sqrt 2 :=
sorry

end triangle_side_inequality_l119_119596


namespace given_expression_equality_l119_119709

theorem given_expression_equality (x : ℝ) (A ω φ b : ℝ) (hA : 0 < A)
  (h : 2 * (Real.cos x)^2 + Real.sin (2 * x) = A * Real.sin (ω * x + φ) + b) :
  A = Real.sqrt 2 ∧ b = 1 :=
sorry

end given_expression_equality_l119_119709


namespace how_much_milk_did_joey_drink_l119_119051

theorem how_much_milk_did_joey_drink (initial_milk : ℚ) (rachel_fraction : ℚ) (joey_fraction : ℚ) :
  initial_milk = 3/4 →
  rachel_fraction = 1/3 →
  joey_fraction = 1/2 →
  joey_fraction * (initial_milk - rachel_fraction * initial_milk) = 1/4 :=
by
  intro h1 h2 h3
  rw [h1, h2, h3]
  sorry

end how_much_milk_did_joey_drink_l119_119051


namespace carla_smoothies_serving_l119_119256

theorem carla_smoothies_serving :
  ∀ (watermelon_puree : ℕ) (cream : ℕ) (serving_size : ℕ),
  watermelon_puree = 500 → cream = 100 → serving_size = 150 →
  (watermelon_puree + cream) / serving_size = 4 :=
by
  intros watermelon_puree cream serving_size
  intro h1 -- watermelon_puree = 500
  intro h2 -- cream = 100
  intro h3 -- serving_size = 150
  sorry

end carla_smoothies_serving_l119_119256


namespace ratio_area_of_circle_to_triangle_l119_119059

theorem ratio_area_of_circle_to_triangle
  (h r b : ℝ)
  (h_triangle : ∃ a, a = b + r ∧ a^2 + b^2 = h^2) :
  (∃ A s : ℝ, s = b + (r + h) / 2 ∧ A = r * s ∧ (∃ circle_area triangle_area : ℝ, circle_area = π * r^2 ∧ triangle_area = 2 * A ∧ circle_area / triangle_area = 2 * π * r / (2 * b + r + h))) :=
by
  sorry

end ratio_area_of_circle_to_triangle_l119_119059


namespace yoongi_rank_l119_119930

def namjoon_rank : ℕ := 2
def yoongi_offset : ℕ := 10

theorem yoongi_rank : namjoon_rank + yoongi_offset = 12 := 
by
  sorry

end yoongi_rank_l119_119930


namespace discounted_price_l119_119053

theorem discounted_price (P : ℝ) (original_price : ℝ) (discount_rate : ℝ)
  (h1 : original_price = 975)
  (h2 : discount_rate = 0.20)
  (h3 : P = original_price - discount_rate * original_price) : 
  P = 780 := 
by
  sorry

end discounted_price_l119_119053


namespace bus_people_count_l119_119139

-- Define the initial number of people on the bus
def initial_people_on_bus : ℕ := 34

-- Define the number of people who got off the bus
def people_got_off : ℕ := 11

-- Define the number of people who got on the bus
def people_got_on : ℕ := 24

-- Define the final number of people on the bus
def final_people_on_bus : ℕ := (initial_people_on_bus - people_got_off) + people_got_on

-- Theorem: The final number of people on the bus is 47.
theorem bus_people_count : final_people_on_bus = 47 := by
  sorry

end bus_people_count_l119_119139


namespace preceding_integer_l119_119106

def bin_to_nat (b : List Bool) : Nat :=
  b.foldl (fun acc bit => 2 * acc + if bit then 1 else 0) 0

theorem preceding_integer : bin_to_nat [true, true, false, false, false] - 1 = bin_to_nat [true, false, true, true, true] := by
  sorry

end preceding_integer_l119_119106


namespace largest_possible_pencils_in_each_package_l119_119207

def ming_pencils : ℕ := 48
def catherine_pencils : ℕ := 36
def lucas_pencils : ℕ := 60

theorem largest_possible_pencils_in_each_package (d : ℕ) (h_ming: ming_pencils % d = 0) (h_catherine: catherine_pencils % d = 0) (h_lucas: lucas_pencils % d = 0) : d ≤ ming_pencils ∧ d ≤ catherine_pencils ∧ d ≤ lucas_pencils ∧ (∀ e, (ming_pencils % e = 0 ∧ catherine_pencils % e = 0 ∧ lucas_pencils % e = 0) → e ≤ d) → d = 12 :=
by 
  sorry

end largest_possible_pencils_in_each_package_l119_119207


namespace problem1_proof_problem2_proof_l119_119258

noncomputable def problem1 (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : a^2 + b^2 = 1) : Prop :=
  |a| + |b| ≤ Real.sqrt 2

noncomputable def problem2 (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : a^2 + b^2 = 1) : Prop :=
  |a^3 / b| + |b^3 / a| ≥ 1

theorem problem1_proof (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : a^2 + b^2 = 1) : problem1 a b h₁ h₂ h₃ :=
  sorry

theorem problem2_proof (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : a^2 + b^2 = 1) : problem2 a b h₁ h₂ h₃ :=
  sorry

end problem1_proof_problem2_proof_l119_119258


namespace find_m_l119_119696

-- Define the vectors a, b, and c
def a : ℝ × ℝ := (2, -1)
def b : ℝ × ℝ := (1, 0)
def c : ℝ × ℝ := (1, -2)

-- Define the condition that a is parallel to m * b - c
def is_parallel (a : ℝ × ℝ) (v : ℝ × ℝ) : Prop :=
  a.1 * v.2 = a.2 * v.1

-- The main theorem we want to prove
theorem find_m (m : ℝ) (h : is_parallel a (m * b.1 - c.1, m * b.2 - c.2)) : m = -3 :=
by {
  -- This will be filled in with the appropriate proof
  sorry
}

end find_m_l119_119696


namespace baker_made_cakes_l119_119271

theorem baker_made_cakes (sold_cakes left_cakes total_cakes : ℕ) (h1 : sold_cakes = 108) (h2 : left_cakes = 59) :
  total_cakes = sold_cakes + left_cakes → total_cakes = 167 := by
  intro h
  rw [h1, h2] at h
  exact h

-- The proof part is omitted since only the statement is required

end baker_made_cakes_l119_119271


namespace units_digit_7_pow_3_l119_119800

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_7_pow_3 : units_digit (7^3) = 3 :=
by
  -- Proof of the theorem would go here
  sorry

end units_digit_7_pow_3_l119_119800


namespace max_tiles_on_floor_l119_119471

-- Definitions based on the given conditions
def tile_length1 := 35 -- in cm
def tile_length2 := 30 -- in cm
def floor_length := 1000 -- in cm
def floor_width := 210 -- in cm

-- Lean 4 statement for the proof problem
theorem max_tiles_on_floor : 
  (max ((floor_length / tile_length1) * (floor_width / tile_length2))
       ((floor_length / tile_length2) * (floor_width / tile_length1))) = 198 := by
  sorry

end max_tiles_on_floor_l119_119471


namespace gcd_euclidean_algorithm_l119_119366

theorem gcd_euclidean_algorithm (a b : ℕ) : 
  ∃ d : ℕ, d = gcd a b ∧ ∀ m : ℕ, (m ∣ a ∧ m ∣ b) → m ∣ d :=
by
  sorry

end gcd_euclidean_algorithm_l119_119366


namespace find_m_l119_119658

theorem find_m (m : ℤ) (h₀ : 0 ≤ m) (h₁ : m < 31) (h₂ : 79453 % 31 = m) : m = 0 :=
by
  sorry

end find_m_l119_119658


namespace xiaolong_correct_answers_l119_119291

/-- There are 50 questions in the exam. Correct answers earn 3 points each,
incorrect answers deduct 1 point each, and unanswered questions score 0 points.
Xiaolong scored 120 points. Prove that the maximum number of questions 
Xiaolong answered correctly is 42. -/
theorem xiaolong_correct_answers :
  ∃ (x y : ℕ), 3 * x - y = 120 ∧ x + y = 48 ∧ x ≤ 50 ∧ y ≤ 50 ∧ x = 42 :=
by
  sorry

end xiaolong_correct_answers_l119_119291


namespace inequality_sine_cosine_l119_119238

theorem inequality_sine_cosine (t : ℝ) (ht : t > 0) : 3 * Real.sin t < 2 * t + t * Real.cos t := 
sorry

end inequality_sine_cosine_l119_119238


namespace Kyle_throws_farther_l119_119841

theorem Kyle_throws_farther (Parker_distance : ℕ) (Grant_ratio : ℚ) (Kyle_ratio : ℚ) (Grant_distance : ℚ) (Kyle_distance : ℚ) :
  Parker_distance = 16 → 
  Grant_ratio = 0.25 → 
  Kyle_ratio = 2 → 
  Grant_distance = Parker_distance + Parker_distance * Grant_ratio → 
  Kyle_distance = Kyle_ratio * Grant_distance → 
  Kyle_distance - Parker_distance = 24 :=
by
  intros hp hg hk hg_dist hk_dist
  subst hp
  subst hg
  subst hk
  subst hg_dist
  subst hk_dist
  -- The proof steps are omitted
  sorry

end Kyle_throws_farther_l119_119841


namespace product_of_three_consecutive_integers_surrounding_twin_primes_divisible_by_240_l119_119124

theorem product_of_three_consecutive_integers_surrounding_twin_primes_divisible_by_240
 (p : ℕ) (prime_p : Prime p) (prime_p_plus_2 : Prime (p + 2)) (p_gt_7 : p > 7) :
  240 ∣ ((p - 1) * p * (p + 1)) := by
  sorry

end product_of_three_consecutive_integers_surrounding_twin_primes_divisible_by_240_l119_119124


namespace determine_C_cards_l119_119635

-- Define the card numbers
def card_numbers : List ℕ := [1,2,3,4,5,6,7,8,9,10,11,12]

-- Define the card sum each person should have
def card_sum := 26

-- Define person's cards
def A_cards : List ℕ := [10, 12]
def B_cards : List ℕ := [6, 11]

-- Define sum constraints for A and B
def sum_A := A_cards.sum
def sum_B := B_cards.sum

-- Define C's complete set of numbers based on remaining cards and sum constraints
def remaining_cards := card_numbers.diff (A_cards ++ B_cards)
def sum_remaining := remaining_cards.sum

theorem determine_C_cards :
  (sum_A + (26 - sum_A)) = card_sum ∧
  (sum_B + (26 - sum_B)) = card_sum ∧
  (sum_remaining = card_sum) → 
  (remaining_cards = [8, 9]) :=
by
  sorry

end determine_C_cards_l119_119635


namespace average_score_l119_119383

theorem average_score (avg1 avg2 : ℕ) (matches1 matches2 : ℕ) (h_avg1 : avg1 = 60) (h_matches1 : matches1 = 10) (h_avg2 : avg2 = 70) (h_matches2 : matches2 = 15) : 
  (matches1 * avg1 + matches2 * avg2) / (matches1 + matches2) = 66 :=
by
  sorry

end average_score_l119_119383


namespace find_x_satisfying_conditions_l119_119598

theorem find_x_satisfying_conditions :
  ∃ x : ℕ, (x % 2 = 1) ∧ (x % 3 = 2) ∧ (x % 4 = 3) ∧ (x % 5 = 4) ∧ x = 59 :=
by
  sorry

end find_x_satisfying_conditions_l119_119598


namespace tutoring_minutes_l119_119737

def flat_rate : ℤ := 20
def per_minute_rate : ℤ := 7
def total_paid : ℤ := 146

theorem tutoring_minutes (m : ℤ) : total_paid = flat_rate + (per_minute_rate * m) → m = 18 :=
by
  sorry

end tutoring_minutes_l119_119737


namespace smallest_p_condition_l119_119887

theorem smallest_p_condition (n p : ℕ) (hn1 : n % 2 = 1) (hn2 : n % 7 = 5) (hp : (n + p) % 10 = 0) : p = 1 := by
  sorry

end smallest_p_condition_l119_119887


namespace ellipse_equation_l119_119965

theorem ellipse_equation (c a b : ℝ)
  (foci1 foci2 : ℝ × ℝ) 
  (h_foci1 : foci1 = (-1, 0)) 
  (h_foci2 : foci2 = (1, 0)) 
  (h_c : c = 1) 
  (h_major_axis : 2 * a = 10) 
  (h_b_sq : b^2 = a^2 - c^2) :
  (∀ x y : ℝ, (x^2 / 25 + y^2 / 24 = 1)) :=
by
  sorry

end ellipse_equation_l119_119965


namespace john_pennies_more_than_kate_l119_119538

theorem john_pennies_more_than_kate (kate_pennies : ℕ) (john_pennies : ℕ) (h_kate : kate_pennies = 223) (h_john : john_pennies = 388) : john_pennies - kate_pennies = 165 := by
  sorry

end john_pennies_more_than_kate_l119_119538


namespace distance_between_parallel_lines_l119_119736

theorem distance_between_parallel_lines 
  (d : ℝ) 
  (r : ℝ)
  (h1 : (42 * 21 + (d / 2) * 42 * (d / 2) = 42 * r^2))
  (h2 : (40 * 20 + (3 * d / 2) * 40 * (3 * d / 2) = 40 * r^2)) :
  d = 3 + 3 / 8 :=
  sorry

end distance_between_parallel_lines_l119_119736


namespace dice_probability_divisible_by_three_ge_one_fourth_l119_119889

theorem dice_probability_divisible_by_three_ge_one_fourth
  (p q r : ℝ) 
  (h1 : 0 ≤ p) (h2 : 0 ≤ q) (h3 : 0 ≤ r) 
  (h4 : p + q + r = 1) : 
  p^3 + q^3 + r^3 + 6 * p * q * r ≥ 1 / 4 :=
sorry

end dice_probability_divisible_by_three_ge_one_fourth_l119_119889


namespace difference_is_2395_l119_119865

def S : ℕ := 476
def L : ℕ := 6 * S + 15
def difference : ℕ := L - S

theorem difference_is_2395 : difference = 2395 :=
by
  sorry

end difference_is_2395_l119_119865


namespace youngest_age_is_20_l119_119119

-- Definitions of the ages
def siblings_ages (y : ℕ) : List ℕ := [y, y+2, y+7, y+11]

-- Condition of the problem: average age is 25
def average_age_25 (y : ℕ) : Prop := (siblings_ages y).sum = 100

-- The statement to be proven
theorem youngest_age_is_20 (y : ℕ) (h : average_age_25 y) : y = 20 :=
  sorry

end youngest_age_is_20_l119_119119


namespace maximum_value_condition_l119_119564

open Real

theorem maximum_value_condition {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (h1 : x + y = 16) (h2 : x = 2 * y) :
  (1 / x + 1 / y) = 9 / 32 :=
by
  sorry

end maximum_value_condition_l119_119564


namespace find_f_29_l119_119990

theorem find_f_29 (f : ℤ → ℤ) (h : ∀ x : ℤ, f (2 * x + 3) = (x - 3) * (x + 4)) : f 29 = 170 := 
by
  sorry

end find_f_29_l119_119990


namespace sheila_hourly_wage_l119_119911

def weekly_working_hours : Nat :=
  (8 * 3) + (6 * 2)

def weekly_earnings : Nat :=
  468

def hourly_wage : Nat :=
  weekly_earnings / weekly_working_hours

theorem sheila_hourly_wage : hourly_wage = 13 :=
by
  sorry

end sheila_hourly_wage_l119_119911


namespace monotonic_increasing_odd_function_implies_a_eq_1_find_max_m_l119_119388

noncomputable def f (a x : ℝ) : ℝ := a - 2 / (2^x + 1)

-- 1. Monotonicity of f(x)
theorem monotonic_increasing (a : ℝ) : ∀ x1 x2 : ℝ, x1 < x2 → f a x1 < f a x2 := sorry

-- 2. f(x) is odd implies a = 1
theorem odd_function_implies_a_eq_1 (h : ∀ x : ℝ, f a (-x) = -f a x) : a = 1 := sorry

-- 3. Find max m such that f(x) ≥ m / 2^x for all x ∈ [2, 3]
theorem find_max_m (h : ∀ x : ℝ, 2 ≤ x ∧ x ≤ 3 → f 1 x ≥ m / 2^x) : m ≤ 12/5 := sorry

end monotonic_increasing_odd_function_implies_a_eq_1_find_max_m_l119_119388


namespace range_of_m_l119_119360

noncomputable def f (x : ℝ) : ℝ := x^3 - x^2 - x

theorem range_of_m (m : ℝ) (x : ℝ) (h1 : x ∈ Set.Icc (-1 : ℝ) 2) : 
  (∀ x ∈ Set.Icc (-1 : ℝ) 2, f x < m) ↔ 2 < m := 
by 
  sorry

end range_of_m_l119_119360


namespace rationalized_expression_correct_A_B_C_D_E_sum_correct_l119_119850

noncomputable def A : ℤ := -18
noncomputable def B : ℤ := 2
noncomputable def C : ℤ := 30
noncomputable def D : ℤ := 5
noncomputable def E : ℤ := 428
noncomputable def expression := 3 / (2 * Real.sqrt 18 + 5 * Real.sqrt 20)
noncomputable def rationalized_form := (A * Real.sqrt B + C * Real.sqrt D) / E

theorem rationalized_expression_correct :
  rationalized_form = (18 * Real.sqrt 2 - 30 * Real.sqrt 5) / -428 :=
by
  sorry

theorem A_B_C_D_E_sum_correct :
  A + B + C + D + E = 447 :=
by
  sorry

end rationalized_expression_correct_A_B_C_D_E_sum_correct_l119_119850


namespace complex_square_l119_119925

theorem complex_square (i : ℂ) (hi : i^2 = -1) : (1 + i)^2 = 2 * i :=
by
  sorry

end complex_square_l119_119925


namespace find_x_l119_119621

theorem find_x (x y : ℝ)
  (h1 : 2 * x + (x - 30) = 360)
  (h2 : y = x - 30)
  (h3 : 2 * x = 4 * y) :
  x = 130 := 
sorry

end find_x_l119_119621


namespace negation_of_P_l119_119638

theorem negation_of_P : ¬(∀ x : ℝ, x^2 + 1 ≥ 2 * x) ↔ ∃ x : ℝ, x^2 + 1 < 2 * x :=
by sorry

end negation_of_P_l119_119638


namespace tip_calculation_l119_119489

def pizza_price : ℤ := 10
def number_of_pizzas : ℤ := 4
def total_pizza_cost := pizza_price * number_of_pizzas
def bill_given : ℤ := 50
def change_received : ℤ := 5
def total_spent := bill_given - change_received
def tip_given := total_spent - total_pizza_cost

theorem tip_calculation : tip_given = 5 :=
by
  -- skipping the proof
  sorry

end tip_calculation_l119_119489


namespace semicircle_triangle_l119_119935

variable (a b r : ℝ)

-- Conditions: 
-- (1) Semicircle of radius r inside a right-angled triangle
-- (2) Shorter edges of the triangle (tangents to the semicircle) have lengths a and b
-- (3) Diameter of the semicircle lies on the hypotenuse of the triangle

theorem semicircle_triangle (h1 : a > 0) (h2 : b > 0) (h3 : r > 0)
  (tangent_property : true) -- Assumed relevant tangent properties are true
  (angle_property : true) -- Assumed relevant angle properties are true
  (geom_configuration : true) -- Assumed specific geometric configuration is correct
  : 1 / r = 1 / a + 1 / b := 
  sorry

end semicircle_triangle_l119_119935


namespace point_B_in_first_quadrant_l119_119458

theorem point_B_in_first_quadrant 
  (a b : ℝ) 
  (h1 : a > 0) 
  (h2 : -b > 0) : 
  (a > 0) ∧ (b > 0) := 
by 
  sorry

end point_B_in_first_quadrant_l119_119458


namespace vacation_hours_per_week_l119_119045

open Nat

theorem vacation_hours_per_week :
  let planned_hours_per_week := 25
  let total_weeks := 15
  let total_money_needed := 4500
  let sick_weeks := 3
  let hourly_rate := total_money_needed / (planned_hours_per_week * total_weeks)
  let remaining_weeks := total_weeks - sick_weeks
  let total_hours_needed := total_money_needed / hourly_rate
  let required_hours_per_week := total_hours_needed / remaining_weeks
  required_hours_per_week = 31.25 := by
sorry

end vacation_hours_per_week_l119_119045


namespace melted_mixture_weight_l119_119997

theorem melted_mixture_weight (Z C : ℝ) (h_ratio : Z / C = 9 / 11) (h_zinc : Z = 28.8) : Z + C = 64 :=
by
  sorry

end melted_mixture_weight_l119_119997


namespace geometric_sequence_second_term_l119_119842

theorem geometric_sequence_second_term
  (first_term : ℕ) (fourth_term : ℕ) (r : ℕ)
  (h1 : first_term = 6)
  (h2 : first_term * r^3 = fourth_term)
  (h3 : fourth_term = 768) :
  first_term * r = 24 := by
  sorry

end geometric_sequence_second_term_l119_119842


namespace sequence_sum_eq_ten_implies_n_eq_120_l119_119967

theorem sequence_sum_eq_ten_implies_n_eq_120 :
  (∀ (a : ℕ → ℝ), (∀ n, a n = 1 / (Real.sqrt n + Real.sqrt (n + 1))) →
    (∃ n, (Finset.sum (Finset.range n) a) = 10 → n = 120)) :=
by
  intro a h
  use 120
  intro h_sum
  sorry

end sequence_sum_eq_ten_implies_n_eq_120_l119_119967


namespace evaluate_ceiling_expression_l119_119276

theorem evaluate_ceiling_expression:
  (Int.ceil ((23 : ℚ) / 9 - Int.ceil ((35 : ℚ) / 23)))
  / (Int.ceil ((35 : ℚ) / 9 + Int.ceil ((9 * 23 : ℚ) / 35))) = 1 / 12 := by
  sorry

end evaluate_ceiling_expression_l119_119276


namespace range_f_log_l119_119825

noncomputable def f : ℝ → ℝ := sorry

axiom f_even (x : ℝ) : f x = f (-x)
axiom f_increasing (x y : ℝ) (h₀ : 0 ≤ x) (h₁ : x ≤ y) : f x ≤ f y
axiom f_at_1 : f 1 = 0

theorem range_f_log (x : ℝ) : f (Real.log x / Real.log (1 / 2)) > 0 ↔ (0 < x ∧ x < 1 / 2) ∨ (2 < x) :=
by
  sorry

end range_f_log_l119_119825


namespace smallest_xy_l119_119097

theorem smallest_xy :
  ∃ (x y : ℕ), (0 < x) ∧ (0 < y) ∧ (1 / x + 1 / (3 * y) = 1 / 6) ∧ (∀ (x' y' : ℕ), (0 < x') ∧ (0 < y') ∧ (1 / x' + 1 / (3 * y') = 1 / 6) → x' * y' ≥ x * y) ∧ x * y = 48 :=
sorry

end smallest_xy_l119_119097


namespace calc_expr_l119_119068

theorem calc_expr : (3^5 * 6^3 + 3^3) = 52515 := by
  sorry

end calc_expr_l119_119068


namespace max_value_of_x_plus_3y_l119_119226

theorem max_value_of_x_plus_3y (x y : ℝ) (h : x^2 / 9 + y^2 = 1) : 
    ∃ θ : ℝ, x = 3 * Real.cos θ ∧ y = Real.sin θ ∧ (x + 3 * y) ≤ 3 * Real.sqrt 2 :=
by
  sorry

end max_value_of_x_plus_3y_l119_119226


namespace sum_of_coefficients_condition_l119_119678

theorem sum_of_coefficients_condition 
  (t : ℕ → ℤ) 
  (d e f : ℤ) 
  (h0 : t 0 = 3) 
  (h1 : t 1 = 7) 
  (h2 : t 2 = 17) 
  (h3 : t 3 = 86)
  (rec_relation : ∀ k ≥ 2, t (k + 1) = d * t k + e * t (k - 1) + f * t (k - 2)) : 
  d + e + f = 14 :=
by
  sorry

end sum_of_coefficients_condition_l119_119678


namespace number_of_space_diagonals_l119_119002

theorem number_of_space_diagonals (V E F tF qF : ℕ)
    (hV : V = 30) (hE : E = 72) (hF : F = 44) (htF : tF = 34) (hqF : qF = 10) : 
    V * (V - 1) / 2 - E - qF * 2 = 343 :=
by
  sorry

end number_of_space_diagonals_l119_119002


namespace problem_l119_119332

theorem problem (a b c : ℤ) :
  (∀ x : ℤ, x^2 + 19 * x + 88 = (x + a) * (x + b)) →
  (∀ x : ℤ, x^2 - 23 * x + 132 = (x - b) * (x - c)) →
  a + b + c = 31 :=
by
  intros h1 h2
  sorry

end problem_l119_119332


namespace total_interval_length_l119_119510

noncomputable def interval_length : ℝ :=
  1 / (1 + 2^Real.pi)

theorem total_interval_length :
  ∀ x : ℝ, x < 1 ∧ Real.tan (Real.log x / Real.log 4) > 0 →
  (∃ y, interval_length = y) :=
by
  sorry

end total_interval_length_l119_119510


namespace quadratic_no_roots_c_positive_l119_119422

theorem quadratic_no_roots_c_positive
  (a b c : ℝ)
  (h_no_roots : ∀ x : ℝ, a * x^2 + b * x + c ≠ 0)
  (h_positive : a + b + c > 0) :
  c > 0 :=
sorry

end quadratic_no_roots_c_positive_l119_119422


namespace molecular_weight_calc_l119_119153

theorem molecular_weight_calc (total_weight : ℕ) (num_moles : ℕ) (one_mole_weight : ℕ) :
  total_weight = 1170 → num_moles = 5 → one_mole_weight = total_weight / num_moles → one_mole_weight = 234 :=
by
  intros h1 h2 h3
  sorry

end molecular_weight_calc_l119_119153


namespace gross_profit_value_l119_119417

theorem gross_profit_value
  (sales_price : ℝ)
  (gross_profit_percentage : ℝ)
  (sales_price_eq : sales_price = 91)
  (gross_profit_percentage_eq : gross_profit_percentage = 1.6)
  (C : ℝ)
  (cost_eqn : sales_price = C + gross_profit_percentage * C) :
  gross_profit_percentage * C = 56 :=
by
  sorry

end gross_profit_value_l119_119417


namespace domain_f_l119_119695

noncomputable def f (x : ℝ) : ℝ := (x - 2) ^ (1 / 2) + 1 / (x - 3)

theorem domain_f :
  {x : ℝ | x ≥ 2 ∧ x ≠ 3 } = {x : ℝ | (2 ≤ x ∧ x < 3) ∨ (3 < x)} :=
by
  sorry

end domain_f_l119_119695


namespace larger_cylinder_candies_l119_119198

theorem larger_cylinder_candies (v₁ v₂ : ℝ) (c₁ c₂ : ℕ) (h₁ : v₁ = 72) (h₂ : c₁ = 30) (h₃ : v₂ = 216) (h₄ : (c₁ : ℝ)/v₁ = (c₂ : ℝ)/v₂) : c₂ = 90 := by
  -- v1 h1 h2 v2 c2 h4 are directly appearing in the conditions
  -- ratio h4 states the condition for densities to be the same 
  sorry

end larger_cylinder_candies_l119_119198


namespace number_of_boxes_l119_119495

-- Definitions based on conditions
def pieces_per_box := 500
def total_pieces := 3000

-- Theorem statement, we need to prove that the number of boxes is 6
theorem number_of_boxes : total_pieces / pieces_per_box = 6 :=
by {
  sorry
}

end number_of_boxes_l119_119495


namespace factorize_x4_minus_64_l119_119868

theorem factorize_x4_minus_64 (x : ℝ) : (x^4 - 64) = (x^2 - 8) * (x^2 + 8) :=
by sorry

end factorize_x4_minus_64_l119_119868


namespace union_A_B_l119_119116

open Set Real

def A : Set ℝ := {x | x^2 - x - 2 < 0}
def B : Set ℝ := {y | ∃ x : ℝ, y = sin x}

theorem union_A_B : A ∪ B = Ico (-1 : ℝ) 2 := by
  sorry

end union_A_B_l119_119116


namespace dogs_with_flea_collars_l119_119690

-- Conditions
def T : ℕ := 80
def Tg : ℕ := 45
def B : ℕ := 6
def N : ℕ := 1

-- Goal: prove the number of dogs with flea collars is 40 given the above conditions
theorem dogs_with_flea_collars : ∃ F : ℕ, F = 40 ∧ T = Tg + F - B + N := 
by
  use 40
  sorry

end dogs_with_flea_collars_l119_119690


namespace sum_and_count_evens_20_30_l119_119223

def sum_integers (a b : ℕ) : ℕ :=
  (b - a + 1) * (a + b) / 2

def count_even_integers (a b : ℕ) : ℕ :=
  (b - a) / 2 + 1

theorem sum_and_count_evens_20_30 :
  let x := sum_integers 20 30
  let y := count_even_integers 20 30
  x + y = 281 :=
by
  sorry

end sum_and_count_evens_20_30_l119_119223


namespace jen_profit_l119_119740

-- Definitions based on the conditions
def cost_per_candy := 80 -- in cents
def sell_price_per_candy := 100 -- in cents
def total_candies_bought := 50
def total_candies_sold := 48

-- Total cost and total revenue calculations
def total_cost := cost_per_candy * total_candies_bought
def total_revenue := sell_price_per_candy * total_candies_sold

-- Profit calculation
def profit := total_revenue - total_cost

-- Main theorem to prove
theorem jen_profit : profit = 800 := by
  -- Proof is skipped
  sorry

end jen_profit_l119_119740


namespace number_of_special_three_digit_numbers_l119_119982

theorem number_of_special_three_digit_numbers : ∃ (n : ℕ), n = 3 ∧
  (∀ (A B C : ℕ), 
    (100 * A + 10 * B + C < 1000 ∧ 100 * A + 10 * B + C ≥ 100) ∧
    B = 2 * C ∧
    B = (A + C) / 2 → 
    (A = 3 * C ∧ C ≤ 3 ∧ B = 2 * C ∧ 100 * A + 10 * B + C = 312 ∨ 
     A = 3 * C ∧ C ≤ 3 ∧ B = 2 * C ∧ 100 * A + 10 * B + C = 642 ∨
     A = 3 * C ∧ C ≤ 3 ∧ B = 2 * C ∧ 100 * A + 10 * B + C = 963))
:= 
sorry

end number_of_special_three_digit_numbers_l119_119982


namespace distance_school_house_l119_119435

def speed_to_school : ℝ := 6
def speed_from_school : ℝ := 4
def total_time : ℝ := 10

theorem distance_school_house : 
  ∃ D : ℝ, (D / speed_to_school + D / speed_from_school = total_time) ∧ (D = 24) :=
sorry

end distance_school_house_l119_119435


namespace parallel_lines_slope_l119_119722

theorem parallel_lines_slope (a : ℝ) :
  (∀ x y : ℝ, x + 2 * a * y - 1 = 0 → (a - 1) * x + a * y + 1 = 0) → a = 3 / 2 :=
by
  sorry

end parallel_lines_slope_l119_119722


namespace meat_needed_l119_119184

theorem meat_needed (meat_per_hamburger : ℚ) (h_meat : meat_per_hamburger = (3 : ℚ) / 8) : 
  (24 * meat_per_hamburger) = 9 :=
by
  sorry

end meat_needed_l119_119184


namespace min_shift_value_l119_119216

theorem min_shift_value (φ : ℝ) (hφ : φ > 0) :
  (∃ k : ℤ, φ = -k * π / 3 + π / 6) →
  ∃ φ_min : ℝ, φ_min = π / 6 ∧ (∀ φ', φ' > 0 → ∃ k' : ℤ, φ' = -k' * π / 3 + π / 6 → φ_min ≤ φ') :=
by
  intro h
  use π / 6
  constructor
  . sorry
  . sorry

end min_shift_value_l119_119216


namespace cubes_closed_under_multiplication_l119_119504

-- Define the set of cubes of positive integers
def is_cube (n : ℕ) : Prop := ∃ k : ℕ, n = k^3

-- Define the multiplication operation on the set of cubes
def cube_mult_closed : Prop :=
  ∀ x y : ℕ, is_cube x → is_cube y → is_cube (x * y)

-- The statement we want to prove
theorem cubes_closed_under_multiplication : cube_mult_closed :=
sorry

end cubes_closed_under_multiplication_l119_119504


namespace range_of_m_l119_119794

-- Define the conditions based on the problem statement
def equation (x m : ℝ) : Prop := (2 * x + m) = (x - 1)

-- The goal is to prove that if there exists a positive solution x to the equation, then m < -1
theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, equation x m ∧ x > 0) → m < -1 :=
by
  sorry

end range_of_m_l119_119794


namespace circumscribed_circle_area_l119_119863

/-- 
Statement: The area of the circle circumscribed about an equilateral triangle with side lengths of 9 units is 27π square units.
-/
theorem circumscribed_circle_area (s : ℕ) (h : s = 9) : 
  let radius := (2/3) * (4.5 * Real.sqrt 3)
  let area := Real.pi * (radius ^ 2)
  area = 27 * Real.pi :=
by
  -- Axis and conditions definitions
  have := h

  -- Definition for the area based on the radius
  let radius := (2/3) * (4.5 * Real.sqrt 3)
  let area := Real.pi * (radius ^ 2)

  -- Statement of the equality to be proven
  show area = 27 * Real.pi
  sorry

end circumscribed_circle_area_l119_119863


namespace Maggie_bought_one_fish_book_l119_119932

-- Defining the variables and constants
def books_about_plants := 9
def science_magazines := 10
def price_book := 15
def price_magazine := 2
def total_amount_spent := 170
def cost_books_about_plants := books_about_plants * price_book
def cost_science_magazines := science_magazines * price_magazine
def cost_books_about_fish := total_amount_spent - (cost_books_about_plants + cost_science_magazines)
def books_about_fish := cost_books_about_fish / price_book

-- Theorem statement
theorem Maggie_bought_one_fish_book : books_about_fish = 1 := by
  -- Proof goes here
  sorry

end Maggie_bought_one_fish_book_l119_119932


namespace minimum_routes_l119_119643

theorem minimum_routes (a b c : ℕ) (h1 : a + b ≥ 14) (h2 : b + c ≥ 14) (h3 : c + a ≥ 14) :
  a + b + c ≥ 21 :=
by sorry

end minimum_routes_l119_119643


namespace q_simplification_l119_119811

noncomputable def q (x a b c D : ℝ) : ℝ :=
  (x + a)^2 / ((a - b) * (a - c)) + 
  (x + b)^2 / ((b - a) * (b - c)) + 
  (x + c)^2 / ((c - a) * (c - b))

theorem q_simplification (a b c D x : ℝ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : b ≠ c) :
  q x a b c D = a + b + c + 2 * x + 3 * D / (a + b + c) :=
by
  sorry

end q_simplification_l119_119811


namespace factorization_example_l119_119047

theorem factorization_example :
  (4 : ℤ) * x^2 - 1 = (2 * x + 1) * (2 * x - 1) := 
by
  sorry

end factorization_example_l119_119047


namespace a2016_value_l119_119718

def seq (a : ℕ → ℚ) : Prop :=
  a 1 = -2 ∧ ∀ n, a (n + 1) = 1 - (1 / a n)

theorem a2016_value : ∃ a : ℕ → ℚ, seq a ∧ a 2016 = 1 / 3 :=
by
  sorry

end a2016_value_l119_119718


namespace find_y_l119_119620

theorem find_y (x y : ℝ) (h1 : 3 * x + 2 = 2) (h2 : y - x = 2) : y = 2 :=
by
  sorry

end find_y_l119_119620


namespace eval_expression_l119_119490

theorem eval_expression : (-3)^5 + 2^(2^3 + 5^2 - 8^2) = -242.999999999535 := by
  sorry

end eval_expression_l119_119490


namespace sum_on_simple_interest_is_1400_l119_119629

noncomputable def sum_placed_on_simple_interest : ℝ :=
  let P_c := 4000
  let r := 0.10
  let n := 1
  let t_c := 2
  let t_s := 3
  let A := P_c * (1 + r / n)^(n * t_c)
  let CI := A - P_c
  let SI := CI / 2
  100 * SI / (r * t_s)

theorem sum_on_simple_interest_is_1400 : sum_placed_on_simple_interest = 1400 := by
  sorry

end sum_on_simple_interest_is_1400_l119_119629


namespace find_common_ratio_l119_119667

-- Define the variables and constants involved.
variables (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ)

-- Define the conditions of the problem.
def is_geometric_sequence := ∀ n, a (n + 1) = q * a n
def sum_of_first_n_terms := ∀ n, S n = a 0 * (1 - q^(n + 1)) / (1 - q)
def condition1 := a 5 = 4 * S 4 + 3
def condition2 := a 6 = 4 * S 5 + 3

-- The main statement that needs to be proved.
theorem find_common_ratio
  (h1: is_geometric_sequence a q)
  (h2: sum_of_first_n_terms a S q)
  (h3: condition1 a S)
  (h4: condition2 a S) : 
  q = 5 :=
sorry -- proof to be provided

end find_common_ratio_l119_119667


namespace maximum_b_value_l119_119941

noncomputable def f (a x : ℝ) := (1 / 2) * x ^ 2 + a * x
noncomputable def g (a b x : ℝ) := 2 * a ^ 2 * Real.log x + b

theorem maximum_b_value (a b : ℝ) (h_a : 0 < a) :
  (∃ x : ℝ, f a x = g a b x ∧ (deriv (f a) x = deriv (g a b) x))
  → b ≤ Real.exp (1 / 2) := 
sorry

end maximum_b_value_l119_119941


namespace floor_div_add_floor_div_succ_eq_l119_119070

theorem floor_div_add_floor_div_succ_eq (n : ℤ) : 
  (⌊(n : ℝ)/2⌋ + ⌊(n + 1 : ℝ)/2⌋ : ℤ) = n := 
sorry

end floor_div_add_floor_div_succ_eq_l119_119070


namespace water_removal_l119_119214

theorem water_removal (n : ℕ) : 
  (∀n, (2:ℚ) / (n + 2) = 1 / 8) ↔ (n = 14) := 
by 
  sorry

end water_removal_l119_119214


namespace intersection_A_B_l119_119461

def setA : Set ℝ := { x | x^2 - 2*x < 3 }
def setB : Set ℝ := { x | x ≤ 2 }
def setC : Set ℝ := { x | -1 < x ∧ x ≤ 2 }

theorem intersection_A_B :
  (setA ∩ setB) = setC :=
by
  sorry

end intersection_A_B_l119_119461


namespace sum_real_imag_parts_l119_119491

noncomputable section

open Complex

theorem sum_real_imag_parts (z : ℂ) (h : z / (1 + 2 * i) = 2 + i) : 
  ((z + 5).re + (z + 5).im) = 0 :=
  by
  sorry

end sum_real_imag_parts_l119_119491


namespace medicine_dose_per_part_l119_119394

-- Define the given conditions
def kg_weight : ℕ := 30
def ml_per_kg : ℕ := 5
def parts : ℕ := 3

-- The theorem statement
theorem medicine_dose_per_part : 
  (kg_weight * ml_per_kg) / parts = 50 :=
by
  sorry

end medicine_dose_per_part_l119_119394


namespace carter_reading_pages_l119_119785

theorem carter_reading_pages (c l o : ℕ)
  (h1: c = l / 2)
  (h2: l = o + 20)
  (h3: o = 40) : c = 30 := by
  sorry

end carter_reading_pages_l119_119785


namespace average_marks_l119_119784

-- Define the conditions
variables (M P C : ℝ)
variables (h1 : M + P = 60) (h2 : C = P + 10)

-- Define the theorem statement
theorem average_marks : (M + C) / 2 = 35 :=
by {
  sorry -- Placeholder for the proof.
}

end average_marks_l119_119784


namespace samBill_l119_119719

def textMessageCostPerText := 8 -- cents
def extraMinuteCostPerMinute := 15 -- cents
def planBaseCost := 25 -- dollars
def includedPlanHours := 25
def centToDollar (cents: Nat) : Nat := cents / 100

def totalBill (texts: Nat) (hours: Nat) : Nat :=
  let textCost := centToDollar (texts * textMessageCostPerText)
  let extraHours := if hours > includedPlanHours then hours - includedPlanHours else 0
  let extraMinutes := extraHours * 60
  let extraMinuteCost := centToDollar (extraMinutes * extraMinuteCostPerMinute)
  planBaseCost + textCost + extraMinuteCost

theorem samBill :
  totalBill 150 26 = 46 := 
sorry

end samBill_l119_119719


namespace Jake_weight_is_118_l119_119001

-- Define the current weights of Jake, his sister, and Mark
variable (J S M : ℕ)

-- Define the given conditions
axiom h1 : J - 12 = 2 * (S + 4)
axiom h2 : M = J + S + 50
axiom h3 : J + S + M = 385

theorem Jake_weight_is_118 : J = 118 :=
by
  sorry

end Jake_weight_is_118_l119_119001


namespace union_M_N_inter_complement_M_N_union_complement_M_N_l119_119574

open Set

variable (U : Set ℝ) (M : Set ℝ) (N : Set ℝ)

noncomputable def universal_set := U = univ

def set_M := M = {x : ℝ | x ≤ 3}
def set_N := N = {x : ℝ | x < 1}

theorem union_M_N (hU : universal_set U) (hM : set_M M) (hN : set_N N) :
    M ∪ N = {x : ℝ | x ≤ 3} :=
by sorry

theorem inter_complement_M_N (hU : universal_set U) (hM : set_M M) (hN : set_N N) :
    (U \ M) ∩ N = ∅ :=
by sorry

theorem union_complement_M_N (hU : universal_set U) (hM : set_M M) (hN : set_N N) :
    (U \ M) ∪ (U \ N) = {x : ℝ | x ≥ 1} :=
by sorry

end union_M_N_inter_complement_M_N_union_complement_M_N_l119_119574


namespace urn_contains_three_red_three_blue_after_five_operations_l119_119890

def initial_red_balls : ℕ := 2
def initial_blue_balls : ℕ := 1
def total_operations : ℕ := 5

noncomputable def calculate_probability (initial_red: ℕ) (initial_blue: ℕ) (operations: ℕ) : ℚ :=
  sorry

theorem urn_contains_three_red_three_blue_after_five_operations :
  calculate_probability initial_red_balls initial_blue_balls total_operations = 8 / 105 :=
by sorry

end urn_contains_three_red_three_blue_after_five_operations_l119_119890


namespace find_b_l119_119130

-- Define the conditions as constants
def x := 36 -- angle a in degrees
def y := 44 -- given
def z := 52 -- given
def w := 48 -- angle b we need to find

-- Define the problem as a theorem
theorem find_b : x + w + y + z = 180 :=
by
  -- Substitute the given values and show the sum
  have h : 36 + 48 + 44 + 52 = 180 := by norm_num
  exact h

end find_b_l119_119130


namespace remaining_surface_area_correct_l119_119619

noncomputable def remaining_surface_area (a : ℕ) (c : ℕ) : ℕ :=
  let original_surface_area := 6 * a^2
  let corner_cube_area := 3 * c^2
  let net_change := corner_cube_area - corner_cube_area
  original_surface_area + 8 * net_change 

theorem remaining_surface_area_correct :
  remaining_surface_area 4 1 = 96 := by
  sorry

end remaining_surface_area_correct_l119_119619


namespace unit_digit_3_pow_2012_sub_1_l119_119221

theorem unit_digit_3_pow_2012_sub_1 :
  (3 ^ 2012 - 1) % 10 = 0 :=
sorry

end unit_digit_3_pow_2012_sub_1_l119_119221


namespace range_of_x_in_function_l119_119111

theorem range_of_x_in_function (x : ℝ) (h : x ≠ 8) : true := sorry

end range_of_x_in_function_l119_119111


namespace father_age_is_32_l119_119609

noncomputable def father_age (D F : ℕ) : Prop :=
  F = 4 * D ∧ (F + 5) + (D + 5) = 50

theorem father_age_is_32 (D F : ℕ) (h : father_age D F) : F = 32 :=
by
  sorry

end father_age_is_32_l119_119609


namespace second_box_probability_nth_box_probability_l119_119486

noncomputable def P_A1 : ℚ := 2 / 3
noncomputable def P_A2 : ℚ := 5 / 9
noncomputable def P_An (n : ℕ) : ℚ :=
  1 / 2 * (1 / 3) ^ n + 1 / 2

theorem second_box_probability :
  P_A2 = 5 / 9 := by
  sorry

theorem nth_box_probability (n : ℕ) :
  P_An n = 1 / 2 * (1 / 3) ^ n + 1 / 2 := by
  sorry

end second_box_probability_nth_box_probability_l119_119486


namespace jane_wins_l119_119781

/-- Define the total number of possible outcomes and the number of losing outcomes -/
def total_outcomes := 64
def losing_outcomes := 12

/-- Define the probability that Jane wins -/
def jane_wins_probability := (total_outcomes - losing_outcomes) / total_outcomes

/-- Problem: Jane wins with a probability of 13/16 given the conditions -/
theorem jane_wins :
  jane_wins_probability = 13 / 16 :=
sorry

end jane_wins_l119_119781


namespace relationship_between_A_and_B_l119_119924

def A : Set ℤ := {-2, 0, 2}
def B : Set ℤ := {x | x^2 + 2 * x = 0}

theorem relationship_between_A_and_B : B ⊆ A :=
sorry

end relationship_between_A_and_B_l119_119924


namespace catherine_initial_pens_l119_119307

-- Defining the conditions
def equal_initial_pencils_and_pens (P : ℕ) : Prop := true
def pens_given_away_per_friend : ℕ := 8
def pencils_given_away_per_friend : ℕ := 6
def number_of_friends : ℕ := 7
def remaining_pens_and_pencils : ℕ := 22

-- The total number of items given away
def total_pens_given_away : ℕ := pens_given_away_per_friend * number_of_friends
def total_pencils_given_away : ℕ := pencils_given_away_per_friend * number_of_friends

-- The problem statement in Lean 4
theorem catherine_initial_pens (P : ℕ) 
  (h1 : equal_initial_pencils_and_pens P)
  (h2 : P - total_pens_given_away + P - total_pencils_given_away = remaining_pens_and_pencils) : 
  P = 60 :=
sorry

end catherine_initial_pens_l119_119307


namespace solve_for_x_l119_119236

theorem solve_for_x (x : ℝ) (h : x - 5.90 = 9.28) : x = 15.18 :=
by
  sorry

end solve_for_x_l119_119236


namespace find_a_plus_2b_l119_119921

variable (a b : ℝ)

theorem find_a_plus_2b (h : (a^2 + 4 * a + 6) * (2 * b^2 - 4 * b + 7) ≤ 10) : 
  a + 2 * b = 0 := 
sorry

end find_a_plus_2b_l119_119921


namespace log2_x_value_l119_119405

noncomputable def log_base (a b : ℝ) : ℝ := Real.log b / Real.log a

theorem log2_x_value
  (x : ℝ)
  (h : log_base (5 * x) (2 * x) = log_base (625 * x) (8 * x)) :
  log_base 2 x = Real.log 5 / (2 * Real.log 2 - 3 * Real.log 5) :=
by
  sorry

end log2_x_value_l119_119405


namespace log_base_function_inequalities_l119_119028

/-- 
Given the function y = log_(1/(sqrt(2))) (1/(x + 3)),
prove that:
1. for y > 0, x ∈ (-2, +∞)
2. for y < 0, x ∈ (-3, -2)
-/
theorem log_base_function_inequalities :
  let y (x : ℝ) := Real.logb (1 / Real.sqrt 2) (1 / (x + 3))
  ∀ x : ℝ, (y x > 0 ↔ x > -2) ∧ (y x < 0 ↔ -3 < x ∧ x < -2) :=
by
  intros
  -- Proof steps would go here
  sorry

end log_base_function_inequalities_l119_119028


namespace remainder_when_divided_by_14_l119_119972

theorem remainder_when_divided_by_14 (A : ℕ) (h1 : A % 1981 = 35) (h2 : A % 1982 = 35) : A % 14 = 7 :=
sorry

end remainder_when_divided_by_14_l119_119972


namespace total_corn_cobs_l119_119792

-- Definitions for the conditions
def rows_first_field : ℕ := 13
def rows_second_field : ℕ := 16
def cobs_per_row : ℕ := 4

-- Statement to prove
theorem total_corn_cobs : (rows_first_field * cobs_per_row + rows_second_field * cobs_per_row) = 116 :=
by sorry

end total_corn_cobs_l119_119792


namespace arthur_additional_muffins_l119_119277

/-- Define the number of muffins Arthur has already baked -/
def muffins_baked : ℕ := 80

/-- Define the multiplier for the total output Arthur wants -/
def desired_multiplier : ℝ := 2.5

/-- Define the equation representing the total desired muffins -/
def total_muffins : ℝ := muffins_baked * desired_multiplier

/-- Define the number of additional muffins Arthur needs to bake -/
def additional_muffins : ℝ := total_muffins - muffins_baked

theorem arthur_additional_muffins : additional_muffins = 120 := by
  sorry

end arthur_additional_muffins_l119_119277


namespace probability_drop_l119_119817

open Real

noncomputable def probability_of_oil_drop_falling_in_hole (c : ℝ) : ℝ :=
  (0.25 * c^2) / (π * (c^2 / 4))

theorem probability_drop (c : ℝ) (hc : c > 0) : 
  probability_of_oil_drop_falling_in_hole c = 0.25 / π :=
by
  sorry

end probability_drop_l119_119817


namespace parallel_lines_eq_a2_l119_119297

theorem parallel_lines_eq_a2
  (a : ℝ)
  (h : ∀ x y : ℝ, x + a * y - 1 = 0 → (a - 1) * x + a * y + 1 = 0)
  : a = 2 := 
  sorry

end parallel_lines_eq_a2_l119_119297


namespace even_abs_func_necessary_not_sufficient_l119_119883

-- Definitions
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

def is_symmetrical_about_origin (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Theorem statement
theorem even_abs_func_necessary_not_sufficient (f : ℝ → ℝ) :
  (∀ x : ℝ, f (-x) = -f x) → (∀ x : ℝ, |f (-x)| = |f x|) ∧ (∃ g : ℝ → ℝ, (∀ x : ℝ, |g (-x)| = |g x|) ∧ ¬(∀ x : ℝ, g (-x) = -g x)) :=
by
  -- Proof omitted.
  sorry

end even_abs_func_necessary_not_sufficient_l119_119883


namespace max_area_triangle_after_t_seconds_l119_119758

-- Define the problem conditions and question
def second_hand_rotation_rate : ℝ := 6 -- degrees per second
def minute_hand_rotation_rate : ℝ := 0.1 -- degrees per second
def perpendicular_angle : ℝ := 90 -- degrees

theorem max_area_triangle_after_t_seconds : 
  ∃ (t : ℝ), (second_hand_rotation_rate - minute_hand_rotation_rate) * t = perpendicular_angle ∧ t = 15 + 15 / 59 :=
by
  -- This is a statement of the proof problem; the proof itself is omitted.
  sorry

end max_area_triangle_after_t_seconds_l119_119758


namespace find_x_squared_plus_y_squared_l119_119061

theorem find_x_squared_plus_y_squared (x y : ℝ) (h1 : x * y = 6) (h2 : x^2 - y^2 + x + y = 44) : x^2 + y^2 = 109 :=
sorry

end find_x_squared_plus_y_squared_l119_119061


namespace ticket_cost_correct_l119_119204

theorem ticket_cost_correct : 
  ∀ (a : ℝ), 
  (3 * a + 5 * (a / 2) = 30) → 
  10 * a + 8 * (a / 2) ≥ 10 * a + 8 * (a / 2) * 0.9 →
  10 * a + 8 * (a / 2) * 0.9 = 68.733 :=
by
  intro a
  intro h1 h2
  sorry

end ticket_cost_correct_l119_119204


namespace solve_for_x_l119_119050

variable (x : ℝ)

-- Define the condition: 20% of x = 300
def twenty_percent_eq_300 := (0.20 * x = 300)

-- Define the goal: 120% of x = 1800
def one_twenty_percent_eq_1800 := (1.20 * x = 1800)

theorem solve_for_x (h : twenty_percent_eq_300 x) : one_twenty_percent_eq_1800 x :=
sorry

end solve_for_x_l119_119050


namespace equation_B_no_real_solution_l119_119483

theorem equation_B_no_real_solution : ∀ x : ℝ, |3 * x + 1| + 6 ≠ 0 := 
by 
  sorry

end equation_B_no_real_solution_l119_119483


namespace union_complement_l119_119010

universe u

def U : Set ℕ := {0, 2, 4, 6, 8, 10}
def A : Set ℕ := {2, 4, 6}
def B : Set ℕ := {1}

theorem union_complement (U A B : Set ℕ) (hU : U = {0, 2, 4, 6, 8, 10}) (hA : A = {2, 4, 6}) (hB : B = {1}) :
  (U \ A) ∪ B = {0, 1, 8, 10} :=
by
  -- The proof is omitted.
  sorry

end union_complement_l119_119010


namespace same_terminal_side_l119_119052

theorem same_terminal_side (k : ℤ) : 
  ((2 * k + 1) * 180) % 360 = ((4 * k + 1) * 180) % 360 ∨ ((2 * k + 1) * 180) % 360 = ((4 * k - 1) * 180) % 360 := 
sorry

end same_terminal_side_l119_119052


namespace rational_square_root_l119_119583

theorem rational_square_root {x y : ℚ} 
  (h : (x^2 + y^2 - 2) * (x + y)^2 + (xy + 1)^2 = 0) : 
  ∃ r : ℚ, r * r = 1 + x * y := 
sorry

end rational_square_root_l119_119583


namespace napkin_ratio_l119_119998

theorem napkin_ratio (initial_napkins : ℕ) (napkins_after : ℕ) (olivia_napkins : ℕ) (amelia_napkins : ℕ)
  (h1 : initial_napkins = 15) (h2 : napkins_after = 45) (h3 : olivia_napkins = 10)
  (h4 : initial_napkins + olivia_napkins + amelia_napkins = napkins_after) :
  amelia_napkins / olivia_napkins = 2 := by
  sorry

end napkin_ratio_l119_119998


namespace roof_length_width_diff_l119_119178

variable (w l : ℝ)
variable (h1 : l = 4 * w)
variable (h2 : l * w = 676)

theorem roof_length_width_diff :
  l - w = 39 :=
by
  sorry

end roof_length_width_diff_l119_119178


namespace find_discount_l119_119170

noncomputable def children_ticket_cost : ℝ := 4.25
noncomputable def adult_ticket_cost : ℝ := children_ticket_cost + 3.25
noncomputable def total_cost_without_discount : ℝ := 2 * adult_ticket_cost + 4 * children_ticket_cost
noncomputable def total_spent : ℝ := 30
noncomputable def discount_received : ℝ := total_cost_without_discount - total_spent

theorem find_discount :
  discount_received = 2 := by
  sorry

end find_discount_l119_119170


namespace man_is_older_by_22_l119_119302

/-- 
Given the present age of the son is 20 years and in two years the man's age will be 
twice the age of his son, prove that the man is 22 years older than his son.
-/
theorem man_is_older_by_22 (S M : ℕ) (h1 : S = 20) (h2 : M + 2 = 2 * (S + 2)) : M - S = 22 :=
by
  sorry  -- Proof will be provided here

end man_is_older_by_22_l119_119302


namespace sum_of_consecutive_pages_l119_119549

theorem sum_of_consecutive_pages (n : ℕ) 
  (h : n * (n + 1) = 20412) : n + (n + 1) + (n + 2) = 429 := by
  sorry

end sum_of_consecutive_pages_l119_119549


namespace maximal_area_of_AMNQ_l119_119530

theorem maximal_area_of_AMNQ (s q : ℝ) (Hq1 : 0 ≤ q) (Hq2 : q ≤ s) :
  let Q := (s, q)
  ∃ M N : ℝ × ℝ, 
    (M.1 ∈ [0,s] ∧ M.2 = 0) ∧ 
    (N.1 = s ∧ N.2 ∈ [0,s]) ∧ 
    if q ≤ (2/3) * s 
    then 
      (M.1 * M.2 / 2 = (CQ/2)) 
    else 
      (N = (s, s)) :=
by sorry

end maximal_area_of_AMNQ_l119_119530


namespace pies_per_day_l119_119575

theorem pies_per_day (daily_pies total_pies : ℕ) (h1 : daily_pies = 8) (h2 : total_pies = 56) :
  total_pies / daily_pies = 7 :=
by sorry

end pies_per_day_l119_119575


namespace remainder_of_difference_l119_119734

open Int

theorem remainder_of_difference (a b : ℕ) (ha : a % 6 = 2) (hb : b % 6 = 3) (h : a > b) : (a - b) % 6 = 5 :=
  sorry

end remainder_of_difference_l119_119734


namespace simplify_expression_and_find_ratio_l119_119790

theorem simplify_expression_and_find_ratio:
  ∀ (k : ℤ), (∃ (a b : ℤ), (a = 1 ∧ b = 3) ∧ (6 * k + 18 = 6 * (a * k + b))) →
  (1 : ℤ) / (3 : ℤ) = (1 : ℤ) / (3 : ℤ) :=
by
  intro k
  intro h
  sorry

end simplify_expression_and_find_ratio_l119_119790


namespace kenneth_left_with_amount_l119_119570

theorem kenneth_left_with_amount (total_earnings : ℝ) (percentage_spent : ℝ) (amount_left : ℝ) 
    (h_total_earnings : total_earnings = 450) (h_percentage_spent : percentage_spent = 0.10) 
    (h_spent_amount : total_earnings * percentage_spent = 45) : 
    amount_left = total_earnings - total_earnings * percentage_spent :=
by sorry

end kenneth_left_with_amount_l119_119570


namespace problem_1_problem_2_l119_119915

noncomputable def f (x a : ℝ) : ℝ := |x - a|

theorem problem_1 (x : ℝ) : (f x 2) ≥ (7 - |x - 1|) ↔ (x ≤ -2 ∨ x ≥ 5) := 
by
  sorry

theorem problem_2 (m n : ℝ) (h_pos_m : 0 < m) (h_pos_n : 0 < n) 
  (h : (f (1/m) 1) + (f (1/(2*n)) 1) = 1) : m + 4 * n ≥ 2 * Real.sqrt 2 + 3 := 
by
  sorry

end problem_1_problem_2_l119_119915


namespace graph_represents_two_intersecting_lines_l119_119746

theorem graph_represents_two_intersecting_lines (x y : ℝ) :
  (x - 1) * (x + y + 2) = (y - 1) * (x + y + 2) → 
  (x + y + 2 = 0 ∨ x = y) ∧ 
  (∃ (x y : ℝ), (x = -1 ∧ y = -1 ∧ x = y ∨ x = -y - 2) ∧ (y = x ∨ y = -x - 2)) :=
by
  sorry

end graph_represents_two_intersecting_lines_l119_119746


namespace chocolateBarsPerBox_l119_119444

def numberOfSmallBoxes := 20
def totalChocolateBars := 500

theorem chocolateBarsPerBox : totalChocolateBars / numberOfSmallBoxes = 25 :=
by
  -- Skipping the proof here
  sorry

end chocolateBarsPerBox_l119_119444


namespace coin_flips_probability_equal_heads_l119_119398

def fair_coin (p : ℚ) := p = 1 / 2
def second_coin (p : ℚ) := p = 3 / 5
def third_coin (p : ℚ) := p = 2 / 3

theorem coin_flips_probability_equal_heads :
  ∀ p1 p2 p3, fair_coin p1 → second_coin p2 → third_coin p3 →
  ∃ m n, m + n = 119 ∧ m / n = 29 / 90 :=
by
  sorry

end coin_flips_probability_equal_heads_l119_119398


namespace find_m_n_l119_119552

def is_prime (n : Nat) : Prop := ∀ m, m ∣ n → m = 1 ∨ m = n

theorem find_m_n (p k : ℕ) (hk : 1 < k) (hp : is_prime p) : 
  (∃ (m n : ℕ), m > 0 ∧ n > 0 ∧ (m, n) ≠ (1, 1) ∧ (m^p + n^p) / 2 = (m + n) / 2 ^ k) ↔ k = p :=
sorry

end find_m_n_l119_119552


namespace parallel_lines_condition_l119_119964

theorem parallel_lines_condition (a : ℝ) : 
  (∃ l1 l2 : ℝ → ℝ, 
    (∀ x y : ℝ, l1 x + a * y + 6 = 0) ∧ 
    (∀ x y : ℝ, (a - 2) * x + 3 * y + 2 * a = 0) ∧
    l1 = l2 ↔ a = 3) :=
sorry

end parallel_lines_condition_l119_119964


namespace sock_pairing_l119_119403

def sockPicker : Prop :=
  let white_socks := 5
  let brown_socks := 5
  let blue_socks := 2
  let total_socks := 12
  let choose (n k : ℕ) := Nat.choose n k
  (choose white_socks 2 + choose brown_socks 2 + choose blue_socks 2 = 21) ∧
  (choose (white_socks + brown_socks) 2 = 45) ∧
  (45 = 45)

theorem sock_pairing :
  sockPicker :=
by sorry

end sock_pairing_l119_119403


namespace find_m_from_hyperbola_and_parabola_l119_119864

theorem find_m_from_hyperbola_and_parabola (a m : ℝ) 
  (h_eccentricity : (Real.sqrt (a^2 + 4)) / a = 3 * Real.sqrt 5 / 5) 
  (h_focus_coincide : (m / 4) = -3) : m = -12 := 
  sorry

end find_m_from_hyperbola_and_parabola_l119_119864


namespace batsman_average_after_17th_inning_l119_119986

theorem batsman_average_after_17th_inning 
  (score_17 : ℕ)
  (delta_avg : ℤ)
  (n_before : ℕ)
  (initial_avg : ℤ)
  (h1 : score_17 = 74)
  (h2 : delta_avg = 3)
  (h3 : n_before = 16)
  (h4 : initial_avg = 23) :
  (initial_avg + delta_avg) = 26 := 
by
  sorry

end batsman_average_after_17th_inning_l119_119986


namespace number_of_oranges_l119_119751

def apples : ℕ := 14
def more_oranges : ℕ := 10

theorem number_of_oranges (o : ℕ) (apples_eq : apples = 14) (more_oranges_eq : more_oranges = 10) :
  o = apples + more_oranges :=
by
  sorry

end number_of_oranges_l119_119751


namespace text_message_cost_eq_l119_119018

theorem text_message_cost_eq (x : ℝ) (CA CB : ℝ) : 
  (CA = 0.25 * x + 9) → (CB = 0.40 * x) → CA = CB → x = 60 :=
by
  intros hCA hCB heq
  sorry

end text_message_cost_eq_l119_119018


namespace find_a_tangent_slope_at_point_l119_119401

theorem find_a_tangent_slope_at_point :
  ∃ (a : ℝ), (∃ (y : ℝ), y = (fun (x : ℝ) => x^4 + a * x^2 + 1) (-1) ∧ (∃ (y' : ℝ), y' = (fun (x : ℝ) => 4 * x^3 + 2 * a * x) (-1) ∧ y' = 8)) ∧ a = -6 :=
by
  -- Used to skip the proof
  sorry

end find_a_tangent_slope_at_point_l119_119401


namespace negation_exists_positive_real_square_plus_one_l119_119353

def exists_positive_real_square_plus_one : Prop :=
  ∃ (x : ℝ), x^2 + 1 > 0

def forall_non_positive_real_square_plus_one : Prop :=
  ∀ (x : ℝ), x^2 + 1 ≤ 0

theorem negation_exists_positive_real_square_plus_one :
  ¬ exists_positive_real_square_plus_one ↔ forall_non_positive_real_square_plus_one :=
by
  sorry

end negation_exists_positive_real_square_plus_one_l119_119353


namespace picture_distance_l119_119505

theorem picture_distance (wall_width picture_width x y : ℝ)
  (h_wall : wall_width = 25)
  (h_picture : picture_width = 5)
  (h_relation : x = 2 * y)
  (h_total : x + picture_width + y = wall_width) :
  x = 13.34 :=
by
  sorry

end picture_distance_l119_119505


namespace log2_bounds_l119_119465

noncomputable def log2 (x : ℝ) := Real.log x / Real.log 2

theorem log2_bounds (h1 : 10^3 = 1000) (h2 : 10^4 = 10000) 
  (h3 : 2^10 = 1024) (h4 : 2^11 = 2048) (h5 : 2^12 = 4096) 
  (h6 : 2^13 = 8192) (h7 : 2^14 = 16384) :
  (3 : ℝ) / 10 < log2 10 ∧ log2 10 < (2 : ℝ) / 7 :=
by
  sorry

end log2_bounds_l119_119465


namespace zora_is_shorter_by_eight_l119_119418

noncomputable def zora_height (z : ℕ) (b : ℕ) (i : ℕ) (zara : ℕ) (average_height : ℕ) : Prop :=
  i = z + 4 ∧
  zara = b ∧
  average_height = 61 ∧
  (z + i + zara + b) / 4 = average_height

theorem zora_is_shorter_by_eight (Z B : ℕ)
  (h1 : zora_height Z B (Z + 4) 64 61) : (B - Z) = 8 :=
by
  sorry

end zora_is_shorter_by_eight_l119_119418


namespace length_of_AB_l119_119705
-- Import the necessary libraries

-- Define the quadratic function
def quad (x : ℝ) : ℝ := x^2 - 2 * x - 3

-- Define a predicate to state that x is a root of the quadratic
def is_root (x : ℝ) : Prop := quad x = 0

-- Define the length between the intersection points
theorem length_of_AB :
  (is_root (-1)) ∧ (is_root 3) → |3 - (-1)| = 4 :=
by {
  sorry
}

end length_of_AB_l119_119705


namespace larry_expression_correct_l119_119237

theorem larry_expression_correct (a b c d : ℤ) (e : ℤ) :
  (a = 1) → (b = 2) → (c = 3) → (d = 4) →
  (a - b - c - d + e = -2 - e) → (e = 3) :=
by
  intros ha hb hc hd heq
  rw [ha, hb, hc, hd] at heq
  linarith

end larry_expression_correct_l119_119237


namespace rhombus_side_length_l119_119459

theorem rhombus_side_length (d1 d2 : ℕ) (h1 : d1 = 24) (h2 : d2 = 70) : 
  ∃ (a : ℕ), a^2 = (d1 / 2)^2 + (d2 / 2)^2 ∧ a = 37 :=
by
  sorry

end rhombus_side_length_l119_119459


namespace sum_of_first_twelve_multiples_of_18_l119_119264

-- Given conditions
def sum_of_first_n_positives (n : ℕ) : ℕ := n * (n + 1) / 2

def first_twelve_multiples_sum (k : ℕ) : ℕ := k * (sum_of_first_n_positives 12)

-- The question to prove
theorem sum_of_first_twelve_multiples_of_18 : first_twelve_multiples_sum 18 = 1404 :=
by
  sorry

end sum_of_first_twelve_multiples_of_18_l119_119264


namespace evaluate_expression_at_y_minus3_l119_119714

theorem evaluate_expression_at_y_minus3 :
  let y := -3
  (5 + y * (2 + y) - 4^2) / (y - 4 + y^2 - y) = -8 / 5 :=
by
  let y := -3
  sorry

end evaluate_expression_at_y_minus3_l119_119714


namespace one_sofa_in_room_l119_119806

def num_sofas_in_room : ℕ :=
  let num_4_leg_tables := 4
  let num_4_leg_chairs := 2
  let num_3_leg_tables := 3
  let num_1_leg_table := 1
  let num_2_leg_rocking_chairs := 1
  let total_legs := 40

  let legs_of_4_leg_tables := num_4_leg_tables * 4
  let legs_of_4_leg_chairs := num_4_leg_chairs * 4
  let legs_of_3_leg_tables := num_3_leg_tables * 3
  let legs_of_1_leg_table := num_1_leg_table * 1
  let legs_of_2_leg_rocking_chairs := num_2_leg_rocking_chairs * 2

  let accounted_legs := legs_of_4_leg_tables + legs_of_4_leg_chairs + legs_of_3_leg_tables + legs_of_1_leg_table + legs_of_2_leg_rocking_chairs

  let remaining_legs := total_legs - accounted_legs

  let sofa_legs := 4
  remaining_legs / sofa_legs

theorem one_sofa_in_room : num_sofas_in_room = 1 :=
  by
    unfold num_sofas_in_room
    rfl

end one_sofa_in_room_l119_119806


namespace employees_excluding_manager_l119_119571

theorem employees_excluding_manager (E : ℕ) (avg_salary_employee : ℕ) (manager_salary : ℕ) (new_avg_salary : ℕ) (total_employees_with_manager : ℕ) :
  avg_salary_employee = 1800 →
  manager_salary = 4200 →
  new_avg_salary = avg_salary_employee + 150 →
  total_employees_with_manager = E + 1 →
  (1800 * E + 4200) / total_employees_with_manager = new_avg_salary →
  E = 15 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end employees_excluding_manager_l119_119571


namespace typing_speed_ratio_l119_119525

variable (T M : ℝ)

-- Conditions
def condition1 : Prop := T + M = 12
def condition2 : Prop := T + 1.25 * M = 14

-- Proof statement
theorem typing_speed_ratio (h1 : condition1 T M) (h2 : condition2 T M) : M / T = 2 := by
  sorry

end typing_speed_ratio_l119_119525


namespace simplify_neg_neg_l119_119960

theorem simplify_neg_neg (a b : ℝ) : -(-a - b) = a + b :=
sorry

end simplify_neg_neg_l119_119960


namespace parallel_lines_perpendicular_lines_l119_119854

-- Define the lines
def l₁ (a : ℝ) (x y : ℝ) := (a - 1) * x + 2 * y + 1 = 0
def l₂ (a : ℝ) (x y : ℝ) := x + a * y + 3 = 0

-- The first proof statement: lines l₁ and l₂ are parallel
theorem parallel_lines (a : ℝ) : 
  (∀ x y : ℝ, l₁ a x y → l₂ a x y → (a * (a - 1) - 2 = 0)) → (a = 2 ∨ a = -1) :=
by
  sorry

-- The second proof statement: lines l₁ and l₂ are perpendicular
theorem perpendicular_lines (a : ℝ) :
  (∀ x y : ℝ, l₁ a x y → l₂ a x y → ((a - 1) * 1 + 2 * a = 0)) → (a = -1 / 3) :=
by
  sorry

end parallel_lines_perpendicular_lines_l119_119854


namespace function_decreasing_interval_l119_119239

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 6 * x^2 - 18 * x + 7

def decreasing_interval (a b : ℝ) : Prop :=
  ∀ x : ℝ, a < x ∧ x < b → 0 > (deriv f x)

theorem function_decreasing_interval : decreasing_interval (-1) 3 :=
by 
  sorry

end function_decreasing_interval_l119_119239


namespace part1_part2_l119_119702

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  1 - (4 / (2 * a^x + a))

theorem part1 (h₁ : ∀ x, f a x = -f a (-x)) (h₂ : a > 0) (h₃ : a ≠ 1) : a = 2 :=
  sorry

theorem part2 (h₁ : a = 2) (x : ℝ) (hx : 0 < x ∧ x ≤ 1) (t : ℝ) :
  t * (f a x) ≥ 2^x - 2 ↔ t ≥ 0 :=
  sorry

end part1_part2_l119_119702


namespace intersection_A_B_l119_119805

open Set Real

def A := { x : ℝ | x ^ 2 - 6 * x + 5 ≤ 0 }
def B := { x : ℝ | ∃ y : ℝ, y = log (x - 2) / log 2 }

theorem intersection_A_B : A ∩ B = { x : ℝ | 2 < x ∧ x ≤ 5 } :=
by
  sorry

end intersection_A_B_l119_119805


namespace shopkeeper_loss_percent_l119_119528

theorem shopkeeper_loss_percent (I : ℝ) (h1 : I > 0) : 
  (0.1 * (I - 0.4 * I)) = 0.4 * (1.1 * I) :=
by
  -- proof goes here
  sorry

end shopkeeper_loss_percent_l119_119528


namespace find_u_l119_119752

theorem find_u 
    (a b c p q u : ℝ) 
    (H₁: (∀ x, x^3 + 2*x^2 + 5*x - 8 = 0 → x = a ∨ x = b ∨ x = c))
    (H₂: (∀ x, x^3 + p*x^2 + q*x + u = 0 → x = a+b ∨ x = b+c ∨ x = c+a)) :
    u = 18 :=
by 
    sorry

end find_u_l119_119752


namespace anna_coaching_days_l119_119109

/-- The total number of days from January 1 to September 4 in a non-leap year -/
def total_days_in_non_leap_year_up_to_sept4 : ℕ :=
  let days_in_january := 31
  let days_in_february := 28
  let days_in_march := 31
  let days_in_april := 30
  let days_in_may := 31
  let days_in_june := 30
  let days_in_july := 31
  let days_in_august := 31
  let days_up_to_sept4 := 4
  days_in_january + days_in_february + days_in_march + days_in_april +
  days_in_may + days_in_june + days_in_july + days_in_august + days_up_to_sept4

theorem anna_coaching_days : total_days_in_non_leap_year_up_to_sept4 = 247 :=
by
  -- Proof omitted
  sorry

end anna_coaching_days_l119_119109


namespace quadratic_expression_rewrite_l119_119858

theorem quadratic_expression_rewrite :
  ∃ a b c : ℚ, (∀ k : ℚ, 12 * k^2 + 8 * k - 16 = a * (k + b)^2 + c) ∧ c + 3 * b = -49/3 :=
sorry

end quadratic_expression_rewrite_l119_119858


namespace pen_sales_average_l119_119942

theorem pen_sales_average (d : ℕ) (h1 : 96 + 44 * d > 0) (h2 : (96 + 44 * d) / (d + 1) = 48) : d = 12 :=
by
  sorry

end pen_sales_average_l119_119942


namespace cos_pi_six_plus_alpha_l119_119539

variable (α : ℝ)

theorem cos_pi_six_plus_alpha (h : Real.sin (Real.pi / 3 - α) = 1 / 6) : 
  Real.cos (Real.pi / 6 + α) = 1 / 6 :=
sorry

end cos_pi_six_plus_alpha_l119_119539


namespace shortest_wire_length_l119_119359

theorem shortest_wire_length (d1 d2 : ℝ) (r1 r2 : ℝ) (t : ℝ) :
  d1 = 8 ∧ d2 = 20 ∧ r1 = 4 ∧ r2 = 10 ∧ t = 8 * Real.sqrt 10 + 17.4 * Real.pi → 
  ∃ l : ℝ, l = t :=
by 
  sorry

end shortest_wire_length_l119_119359


namespace ratio_boys_girls_l119_119845

variable (S G : ℕ)

theorem ratio_boys_girls (h : (2 / 3 : ℚ) * G = (1 / 5 : ℚ) * S) :
  (S - G) * 3 = 7 * G := by
  -- Proof goes here
  sorry

end ratio_boys_girls_l119_119845


namespace intersection_A_B_l119_119660

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | 2 * x^2 - x - 1 < 0}
def B : Set ℝ := {x : ℝ | Real.log x / Real.log (1/2) < 3}

-- Define the intersection A ∩ B and state the theorem
theorem intersection_A_B : A ∩ B = {x : ℝ | 1/8 < x ∧ x < 1} := by
   sorry

end intersection_A_B_l119_119660


namespace train_can_speed_up_l119_119707

theorem train_can_speed_up (d t_reduced v_increased v_safe : ℝ) 
  (h1 : d = 1600) (h2 : t_reduced = 4) (h3 : v_increased = 20) (h4 : v_safe = 140) :
  ∃ x : ℝ, (x > 0) ∧ (d / x) = (d / (x + v_increased) + t_reduced) ∧ ((x + v_increased) < v_safe) :=
by 
  sorry

end train_can_speed_up_l119_119707


namespace number_of_male_students_l119_119670

variables (total_students sample_size female_sampled female_students male_students : ℕ)
variables (h_total : total_students = 1600)
variables (h_sample : sample_size = 200)
variables (h_female_sampled : female_sampled = 95)
variables (h_prob : (sample_size : ℚ) / total_students = (female_sampled : ℚ) / female_students)
variables (h_female_students : female_students = 760)

theorem number_of_male_students : male_students = total_students - female_students := by
  sorry

end number_of_male_students_l119_119670


namespace projection_vector_satisfies_conditions_l119_119225

variable (v1 v2 : ℚ)

def line_l (t : ℚ) : ℚ × ℚ :=
(2 + 3 * t, 5 - 2 * t)

def line_m (s : ℚ) : ℚ × ℚ :=
(-2 + 3 * s, 7 - 2 * s)

theorem projection_vector_satisfies_conditions :
  3 * v1 + 2 * v2 = 6 ∧ 
  ∃ k : ℚ, v1 = k * 3 ∧ v2 = k * (-2) → 
  (v1, v2) = (18 / 5, -12 / 5) :=
by
  sorry

end projection_vector_satisfies_conditions_l119_119225


namespace sara_spent_correct_amount_on_movies_l119_119228

def cost_ticket : ℝ := 10.62
def num_tickets : ℕ := 2
def cost_rented_movie : ℝ := 1.59
def cost_purchased_movie : ℝ := 13.95

def total_amount_spent : ℝ :=
  num_tickets * cost_ticket + cost_rented_movie + cost_purchased_movie

theorem sara_spent_correct_amount_on_movies :
  total_amount_spent = 36.78 :=
sorry

end sara_spent_correct_amount_on_movies_l119_119228


namespace triangle_properties_l119_119229

noncomputable def triangle_side_lengths (m1 m2 m3 : ℝ) : Prop :=
  ∃ a b c s,
    m1 = 20 ∧
    m2 = 24 ∧
    m3 = 30 ∧
    a = 36.28 ∧
    b = 30.24 ∧
    c = 24.19 ∧
    s = 362.84

theorem triangle_properties :
  triangle_side_lengths 20 24 30 :=
by
  sorry

end triangle_properties_l119_119229


namespace squat_percentage_loss_l119_119579

variable (original_squat : ℕ)
variable (original_bench : ℕ)
variable (original_deadlift : ℕ)
variable (lost_deadlift : ℕ)
variable (new_total : ℕ)
variable (unchanged_bench : ℕ)

theorem squat_percentage_loss
  (h1 : original_squat = 700)
  (h2 : original_bench = 400)
  (h3 : original_deadlift = 800)
  (h4 : lost_deadlift = 200)
  (h5 : new_total = 1490)
  (h6 : unchanged_bench = 400) :
  (original_squat - (new_total - (unchanged_bench + (original_deadlift - lost_deadlift)))) * 100 / original_squat = 30 :=
by sorry

end squat_percentage_loss_l119_119579


namespace smaller_of_x_and_y_is_15_l119_119389

variable {x y : ℕ}

/-- Given two positive numbers x and y are in the ratio 3:5, 
and the sum of x and y plus 10 equals 50,
prove that the smaller of x and y is 15. -/
theorem smaller_of_x_and_y_is_15 (h1 : x * 5 = y * 3) (h2 : x + y + 10 = 50) (h3 : 0 < x) (h4 : 0 < y) : x = 15 :=
by
  sorry

end smaller_of_x_and_y_is_15_l119_119389


namespace isosceles_triangles_l119_119259

noncomputable def is_isosceles_triangle (a b c : ℝ) : Prop :=
  a = b ∨ b = c ∨ a = c

theorem isosceles_triangles (a b c : ℝ) (h : a ≥ b ∧ b ≥ c ∧ c > 0)
    (H : ∀ n : ℕ, a ^ n + b ^ n > c ^ n ∧ b ^ n + c ^ n > a ^ n ∧ c ^ n + a ^ n > b ^ n) :
    is_isosceles_triangle a b c :=
  sorry

end isosceles_triangles_l119_119259


namespace expression_for_x_l119_119715

variable (A B C x y : ℝ)

-- Conditions
def condition1 := A > C
def condition2 := C > B
def condition3 := B > 0
def condition4 := C = (1 + y / 100) * B
def condition5 := A = (1 + x / 100) * C

-- The theorem
theorem expression_for_x (h1 : condition1 A C) (h2 : condition2 C B) (h3 : condition3 B) (h4 : condition4 B C y) (h5 : condition5 A C x) :
    x = 100 * ((100 * (A - B)) / (100 + y)) :=
sorry

end expression_for_x_l119_119715


namespace solve_for_x_l119_119835

theorem solve_for_x (x : ℚ) (h : (x + 4) / (x - 3) = (x - 2) / (x + 2)) : x = -2 / 11 :=
by
  sorry

end solve_for_x_l119_119835


namespace number_of_true_propositions_l119_119146

-- Let's state the propositions
def original_proposition (P Q : Prop) := P → Q
def converse_proposition (P Q : Prop) := Q → P
def inverse_proposition (P Q : Prop) := ¬P → ¬Q
def contrapositive_proposition (P Q : Prop) := ¬Q → ¬P

-- Main statement we need to prove
theorem number_of_true_propositions (P Q : Prop) (hpq : original_proposition P Q) 
  (hc: contrapositive_proposition P Q) (hev: converse_proposition P Q)  (hbv: inverse_proposition P Q) : 
  (¬(P ↔ Q) ∨ (¬¬P ↔ ¬¬Q) ∨ (¬Q → ¬P) ∨ (P → Q)) := sorry

end number_of_true_propositions_l119_119146


namespace lcm_two_primes_is_10_l119_119872

theorem lcm_two_primes_is_10 (x y : ℕ) (h_prime_x : Nat.Prime x) (h_prime_y : Nat.Prime y) (h_lcm : Nat.lcm x y = 10) (h_gt : x > y) : 2 * x + y = 12 :=
sorry

end lcm_two_primes_is_10_l119_119872


namespace unique_factor_and_multiple_of_13_l119_119430

theorem unique_factor_and_multiple_of_13 (n : ℕ) (h1 : n ∣ 13) (h2 : 13 ∣ n) : n = 13 :=
sorry

end unique_factor_and_multiple_of_13_l119_119430


namespace find_fx_l119_119807

def is_even_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = f x

theorem find_fx (f : ℝ → ℝ)
  (h_even : is_even_function f)
  (h_neg : ∀ x : ℝ, x < 0 → f x = x * (x - 1)) :
  ∀ x : ℝ, x > 0 → f x = x * (x + 1) :=
by
  sorry

end find_fx_l119_119807


namespace max_ABC_ge_4_9_max_alpha_beta_gamma_ge_4_9_l119_119224

variable (p q : ℝ) (x y : ℝ)
variable (A B C α β γ : ℝ)

-- Conditions
axiom hp : 0 ≤ p ∧ p ≤ 1 
axiom hq : 0 ≤ q ∧ q ≤ 1 
axiom h1 : (p * x + (1 - p) * y)^2 = A * x^2 + B * x * y + C * y^2
axiom h2 : (p * x + (1 - p) * y) * (q * x + (1 - q) * y) = α * x^2 + β * x * y + γ * y^2

-- Problem
theorem max_ABC_ge_4_9 : max A (max B C) ≥ 4 / 9 := 
sorry

theorem max_alpha_beta_gamma_ge_4_9 : max α (max β γ) ≥ 4 / 9 := 
sorry

end max_ABC_ge_4_9_max_alpha_beta_gamma_ge_4_9_l119_119224


namespace area_difference_l119_119821

theorem area_difference (A B a b : ℝ) : (A * b) - (a * B) = A * b - a * B :=
by {
  -- proof goes here
  sorry
}

end area_difference_l119_119821


namespace sum_of_squares_l119_119630

theorem sum_of_squares (k₁ k₂ k₃ : ℝ)
  (h_sum : k₁ + k₂ + k₃ = 1) : k₁^2 + k₂^2 + k₃^2 ≥ 1/3 :=
by sorry

end sum_of_squares_l119_119630


namespace quadratic_not_proposition_l119_119769

def is_proposition (P : Prop) : Prop := ∃ (b : Bool), (b = true ∨ b = false)

theorem quadratic_not_proposition : ¬ is_proposition (∃ x : ℝ, x^2 + 2*x - 3 < 0) :=
by 
  sorry

end quadratic_not_proposition_l119_119769


namespace fraction_multiplication_l119_119058

theorem fraction_multiplication : ((1 / 2) * (1 / 3) * (1 / 6) * 72 = 2) :=
by
  sorry

end fraction_multiplication_l119_119058


namespace total_acorns_proof_l119_119788

variable (x y : ℝ)

def total_acorns (x y : ℝ) : ℝ :=
  let shawna := x
  let sheila := 5.3 * x
  let danny := 5.3 * x + y
  let ella := 2 * (4.3 * x + y)
  shawna + sheila + danny + ella

theorem total_acorns_proof (x y : ℝ) :
  total_acorns x y = 20.2 * x + 3 * y :=
by
  unfold total_acorns
  sorry

end total_acorns_proof_l119_119788


namespace maximum_profit_l119_119582

def cost_price_per_unit : ℕ := 40
def initial_selling_price_per_unit : ℕ := 50
def units_sold_per_month : ℕ := 210
def price_increase_effect (x : ℕ) : ℕ := units_sold_per_month - 10 * x
def profit_function (x : ℕ) : ℕ := (price_increase_effect x) * (initial_selling_price_per_unit + x - cost_price_per_unit)

theorem maximum_profit :
  profit_function 5 = 2400 ∧ profit_function 6 = 2400 :=
by
  sorry

end maximum_profit_l119_119582


namespace solve_for_x_l119_119479

theorem solve_for_x (x : ℝ) : (2010 + 2 * x) ^ 2 = x ^ 2 → x = -2010 ∨ x = -670 := by
  sorry

end solve_for_x_l119_119479


namespace sum_positive_implies_at_least_one_positive_l119_119578

variables {a b : ℝ}

theorem sum_positive_implies_at_least_one_positive (h : a + b > 0) : a > 0 ∨ b > 0 :=
sorry

end sum_positive_implies_at_least_one_positive_l119_119578


namespace student_knows_german_l119_119523

-- Definitions for each classmate's statement
def classmate1 (lang: String) : Prop := lang ≠ "French"
def classmate2 (lang: String) : Prop := lang = "Spanish" ∨ lang = "German"
def classmate3 (lang: String) : Prop := lang = "Spanish"

-- Conditions: at least one correct and at least one incorrect
def at_least_one_correct (lang: String) : Prop :=
  classmate1 lang ∨ classmate2 lang ∨ classmate3 lang

def at_least_one_incorrect (lang: String) : Prop :=
  ¬classmate1 lang ∨ ¬classmate2 lang ∨ ¬classmate3 lang

-- The statement to prove
theorem student_knows_german : ∀ lang : String,
  at_least_one_correct lang → at_least_one_incorrect lang → lang = "German" :=
by
  intros lang Hcorrect Hincorrect
  revert Hcorrect Hincorrect
  -- sorry stands in place of direct proof
  sorry

end student_knows_german_l119_119523


namespace compute_vector_expression_l119_119976

theorem compute_vector_expression :
  4 • (⟨3, -5⟩ : ℝ × ℝ) - 3 • (⟨2, -6⟩ : ℝ × ℝ) + 2 • (⟨0, 3⟩ : ℝ × ℝ) = (⟨6, 4⟩ : ℝ × ℝ) := 
sorry

end compute_vector_expression_l119_119976


namespace sum_reciprocal_l119_119551

-- Definition of the problem
theorem sum_reciprocal (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x^2 + y^2 = 4 * x * y) : 
  (1 / x) + (1 / y) = 1 :=
sorry

end sum_reciprocal_l119_119551


namespace scarves_per_box_l119_119931

theorem scarves_per_box (S : ℕ) 
  (boxes : ℕ)
  (mittens_per_box : ℕ)
  (total_clothes : ℕ)
  (h1 : boxes = 4)
  (h2 : mittens_per_box = 6)
  (h3 : total_clothes = 32)
  (total_mittens := boxes * mittens_per_box)
  (total_scarves := total_clothes - total_mittens) :
  total_scarves / boxes = 2 :=
by
  sorry

end scarves_per_box_l119_119931


namespace regular_polygon_sides_l119_119039

theorem regular_polygon_sides (h : ∀ n : ℕ, (120 * n) = 180 * (n - 2)) : 6 = 6 :=
by
  sorry

end regular_polygon_sides_l119_119039


namespace min_value_of_2gx_sq_minus_fx_l119_119641

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x + b
noncomputable def g (a c : ℝ) (x : ℝ) : ℝ := a * x + c

theorem min_value_of_2gx_sq_minus_fx (a b c : ℝ) (h_a_nonzero : a ≠ 0)
  (h_min_fx : ∃ x : ℝ, 2 * (f a b x)^2 - g a c x = 7 / 2) :
  ∃ x : ℝ, 2 * (g a c x)^2 - f a b x = -15 / 4 :=
sorry

end min_value_of_2gx_sq_minus_fx_l119_119641


namespace ship_length_l119_119476

theorem ship_length (E S L : ℕ) (h1 : 150 * E = L + 150 * S) (h2 : 90 * E = L - 90 * S) : 
  L = 24 :=
by
  sorry

end ship_length_l119_119476


namespace evaluate_g_at_3_l119_119809

def g (x : ℝ) : ℝ := 7 * x^3 - 5 * x^2 - 7 * x + 3

theorem evaluate_g_at_3 : g 3 = 126 := 
by 
  sorry

end evaluate_g_at_3_l119_119809


namespace inf_many_non_prime_additions_l119_119392

theorem inf_many_non_prime_additions :
  ∃ᶠ (a : ℕ) in at_top, ∀ n : ℕ, n > 0 → ¬ Prime (n^4 + a) :=
by {
  sorry -- proof to be provided
}

end inf_many_non_prime_additions_l119_119392


namespace problem_solution_l119_119909

variable (a b : ℝ)

theorem problem_solution (h : 2 * a - 3 * b = 5) : 4 * a^2 - 9 * b^2 - 30 * b + 1 = 26 :=
sorry

end problem_solution_l119_119909


namespace find_y_value_l119_119188

theorem find_y_value (k : ℝ) (h1 : ∀ (x : ℝ), y = k * x) 
(h2 : y = 4 ∧ x = 2) : 
(∀ (x : ℝ), x = -2 → y = -4) := 
by 
  sorry

end find_y_value_l119_119188


namespace positions_after_317_moves_l119_119851

-- Define positions for the cat and dog
inductive ArchPosition
| North | East | South | West
deriving DecidableEq

inductive PathPosition
| North | Northeast | East | Southeast | South | Southwest
deriving DecidableEq

-- Define the movement function for cat and dog
def cat_position (n : Nat) : ArchPosition :=
  match n % 4 with
  | 0 => ArchPosition.North
  | 1 => ArchPosition.East
  | 2 => ArchPosition.South
  | _ => ArchPosition.West

def dog_position (n : Nat) : PathPosition :=
  match n % 6 with
  | 0 => PathPosition.North
  | 1 => PathPosition.Northeast
  | 2 => PathPosition.East
  | 3 => PathPosition.Southeast
  | 4 => PathPosition.South
  | _ => PathPosition.Southwest

-- Theorem statement to prove the positions after 317 moves
theorem positions_after_317_moves :
  cat_position 317 = ArchPosition.North ∧
  dog_position 317 = PathPosition.South :=
by
  sorry

end positions_after_317_moves_l119_119851


namespace trigonometric_identity_l119_119694

variable {α : ℝ}

theorem trigonometric_identity (h : Real.tan α = -3) :
  (Real.cos α + 2 * Real.sin α) / (Real.cos α - 3 * Real.sin α) = -1 / 2 :=
  sorry

end trigonometric_identity_l119_119694


namespace reduction_rate_equation_l119_119477

-- Define the given conditions
def original_price : ℝ := 23
def reduced_price : ℝ := 18.63
def monthly_reduction_rate (x : ℝ) : ℝ := (1 - x) ^ 2

-- Prove that the given equation holds
theorem reduction_rate_equation (x : ℝ) : 
  original_price * monthly_reduction_rate x = reduced_price :=
by
  sorry

end reduction_rate_equation_l119_119477


namespace tetrahedron_volume_l119_119147

noncomputable def volume_of_tetrahedron (S1 S2 a α : ℝ) : ℝ :=
  (2 * S1 * S2 * Real.sin α) / (3 * a)

theorem tetrahedron_volume (S1 S2 a α : ℝ) :
  a > 0 → S1 > 0 → S2 > 0 → α ≥ 0 → α ≤ Real.pi → volume_of_tetrahedron S1 S2 a α =
  (2 * S1 * S2 * Real.sin α) / (3 * a) := 
by
  intros
  -- The proof is omitted here.
  sorry

end tetrahedron_volume_l119_119147


namespace tony_initial_money_l119_119320

theorem tony_initial_money (ticket_cost hotdog_cost money_left initial_money : ℕ) 
  (h_ticket : ticket_cost = 8)
  (h_hotdog : hotdog_cost = 3) 
  (h_left : money_left = 9)
  (h_spent : initial_money = ticket_cost + hotdog_cost + money_left) :
  initial_money = 20 := 
by 
  sorry

end tony_initial_money_l119_119320


namespace servings_per_guest_l119_119700

-- Definitions based on conditions
def num_guests : ℕ := 120
def servings_per_bottle : ℕ := 6
def num_bottles : ℕ := 40

-- Theorem statement
theorem servings_per_guest : (num_bottles * servings_per_bottle) / num_guests = 2 := by
  sorry

end servings_per_guest_l119_119700


namespace neg_p_sufficient_not_necessary_q_l119_119041

theorem neg_p_sufficient_not_necessary_q (p q : Prop) 
  (h₁ : p → ¬q) 
  (h₂ : ¬(¬q → p)) : (q → ¬p) ∧ ¬(¬p → q) :=
sorry

end neg_p_sufficient_not_necessary_q_l119_119041


namespace add_2001_1015_l119_119871

theorem add_2001_1015 : 2001 + 1015 = 3016 := 
by
  sorry

end add_2001_1015_l119_119871


namespace cylinder_height_l119_119306

theorem cylinder_height (r h : ℝ) (SA : ℝ) (h₀ : r = 3) (h₁ : SA = 36 * Real.pi) (h₂ : SA = 2 * Real.pi * r^2 + 2 * Real.pi * r * h) : h = 3 :=
by
  -- The proof will be constructed here
  sorry

end cylinder_height_l119_119306


namespace maximum_volume_of_prism_l119_119429

noncomputable def maximum_volume_prism (s : ℝ) (θ : ℝ) (face_area_sum : ℝ) : ℝ := 
  if (s = 6 ∧ θ = Real.pi / 3 ∧ face_area_sum = 36) then 27 
  else 0

theorem maximum_volume_of_prism : 
  ∀ (s θ face_area_sum), s = 6 ∧ θ = Real.pi / 3 ∧ face_area_sum = 36 → maximum_volume_prism s θ face_area_sum = 27 :=
by
  intros
  sorry

end maximum_volume_of_prism_l119_119429


namespace first_tv_cost_is_672_l119_119387

-- width and height of the first TV
def width_first_tv : ℕ := 24
def height_first_tv : ℕ := 16
-- width and height of the new TV
def width_new_tv : ℕ := 48
def height_new_tv : ℕ := 32
-- cost of the new TV
def cost_new_tv : ℕ := 1152
-- extra cost per square inch for the first TV
def extra_cost_per_square_inch : ℕ := 1

noncomputable def cost_first_tv : ℕ :=
  let area_first_tv := width_first_tv * height_first_tv
  let area_new_tv := width_new_tv * height_new_tv
  let cost_per_square_inch_new_tv := cost_new_tv / area_new_tv
  let cost_per_square_inch_first_tv := cost_per_square_inch_new_tv + extra_cost_per_square_inch
  cost_per_square_inch_first_tv * area_first_tv

theorem first_tv_cost_is_672 : cost_first_tv = 672 := by
  sorry

end first_tv_cost_is_672_l119_119387


namespace quadratic_roots_correct_l119_119830

def quadratic (b c : ℝ) (x : ℝ) : ℝ := x^2 + b * x + c

theorem quadratic_roots_correct (b c : ℝ) 
  (h₀ : quadratic b c (-2) = 5)
  (h₁ : quadratic b c (-1) = 0)
  (h₂ : quadratic b c 0 = -3)
  (h₃ : quadratic b c 1 = -4)
  (h₄ : quadratic b c 2 = -3)
  (h₅ : quadratic b c 4 = 5)
  : (quadratic b c (-1) = 0) ∧ (quadratic b c 3 = 0) :=
sorry

end quadratic_roots_correct_l119_119830


namespace total_yards_run_l119_119194

theorem total_yards_run (Malik_yards_per_game : ℕ) (Josiah_yards_per_game : ℕ) (Darnell_yards_per_game : ℕ) (games : ℕ) 
  (hM : Malik_yards_per_game = 18) (hJ : Josiah_yards_per_game = 22) (hD : Darnell_yards_per_game = 11) (hG : games = 4) : 
  Malik_yards_per_game * games + Josiah_yards_per_game * games + Darnell_yards_per_game * games = 204 := by
  sorry

end total_yards_run_l119_119194


namespace evaluate_expression_l119_119046

noncomputable def a := Real.sqrt 5 + Real.sqrt 7 + Real.sqrt 35 + 2
noncomputable def b := -Real.sqrt 5 + Real.sqrt 7 + Real.sqrt 35 + 2
noncomputable def c := Real.sqrt 5 - Real.sqrt 7 + Real.sqrt 35 + 2
noncomputable def d := -Real.sqrt 5 - Real.sqrt 7 + Real.sqrt 35 + 2

theorem evaluate_expression : (1 / a + 1 / b + 1 / c + 1 / d)^2 = 39 / 140 := 
by
  sorry

end evaluate_expression_l119_119046


namespace problem_statement_l119_119599

theorem problem_statement (a b : ℝ) (h : a ≠ b) : (a - b) ^ 2 > 0 := sorry

end problem_statement_l119_119599


namespace center_of_circle_in_second_or_fourth_quadrant_l119_119891

theorem center_of_circle_in_second_or_fourth_quadrant
  (α : ℝ) 
  (hyp1 : ∀ x y : ℝ, x^2 * Real.cos α - y^2 * Real.sin α + 2 = 0 → Real.cos α * Real.sin α > 0)
  (circle_eq : ∀ x y : ℝ, x^2 + y^2 + 2*x*Real.cos α - 2*y*Real.sin α = 0) :
  (-Real.cos α > 0 ∧ Real.sin α > 0) ∨ (-Real.cos α < 0 ∧ Real.sin α < 0) :=
sorry

end center_of_circle_in_second_or_fourth_quadrant_l119_119891


namespace max_sum_at_11_l119_119368

noncomputable def is_arithmetic_seq (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

noncomputable def sum_seq (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (n + 1) * a 0 + (n * (n + 1) / 2) * (a 1 - a 0)

theorem max_sum_at_11 (a : ℕ → ℚ) (d : ℚ) (h_arith : is_arithmetic_seq a) (h_a1_gt_0 : a 0 > 0)
 (h_sum_eq : sum_seq a 13 = sum_seq a 7) : 
  ∃ n : ℕ, sum_seq a n = sum_seq a 10 + (a 10 + a 11) := sorry


end max_sum_at_11_l119_119368


namespace meal_cost_is_25_l119_119008

def total_cost_samosas : ℕ := 3 * 2
def total_cost_pakoras : ℕ := 4 * 3
def cost_mango_lassi : ℕ := 2
def tip_percentage : ℝ := 0.25

def total_food_cost : ℕ := total_cost_samosas + total_cost_pakoras + cost_mango_lassi
def tip_amount : ℝ := total_food_cost * tip_percentage
def total_meal_cost : ℝ := total_food_cost + tip_amount

theorem meal_cost_is_25 : total_meal_cost = 25 := by
    sorry

end meal_cost_is_25_l119_119008


namespace largest_angle_of_triangle_l119_119448

theorem largest_angle_of_triangle
  (a b y : ℝ)
  (h1 : a = 60)
  (h2 : b = 70)
  (h3 : a + b + y = 180) :
  max a (max b y) = b :=
by
  sorry

end largest_angle_of_triangle_l119_119448


namespace production_steps_description_l119_119233

-- Definition of the choices
inductive FlowchartType
| ProgramFlowchart
| ProcessFlowchart
| KnowledgeStructureDiagram
| OrganizationalStructureDiagram

-- Conditions
def describeProductionSteps (flowchart : FlowchartType) : Prop :=
flowchart = FlowchartType.ProcessFlowchart

-- The statement to be proved
theorem production_steps_description:
  describeProductionSteps FlowchartType.ProcessFlowchart := 
sorry -- proof to be provided

end production_steps_description_l119_119233


namespace at_least_one_non_negative_l119_119280

variable (x : ℝ)
def a : ℝ := x^2 - 1
def b : ℝ := 2*x + 2

theorem at_least_one_non_negative (x : ℝ) : ¬ (a x < 0 ∧ b x < 0) :=
by
  sorry

end at_least_one_non_negative_l119_119280


namespace intersection_l119_119847

def setA : Set ℝ := { x | ∃ y, y = Real.sqrt (1 - x) }
def setB : Set ℝ := { x | x^2 - 2 * x ≥ 0 }

theorem intersection: setA ∩ setB = { x : ℝ | x ≤ 0 } := by
  sorry

end intersection_l119_119847


namespace range_of_a_l119_119176

theorem range_of_a (hP : ¬ ∃ x : ℝ, x^2 + 2 * a * x + a ≤ 0) : 0 < a ∧ a < 1 :=
sorry

end range_of_a_l119_119176


namespace PU_squared_fraction_l119_119995

noncomputable def compute_PU_squared : ℚ :=
  sorry -- Proof of the distance computation PU^2.

theorem PU_squared_fraction :
  ∃ (a b : ℕ), (gcd a b = 1) ∧ (compute_PU_squared = a / b) :=
  sorry -- Proof that the resulting fraction a/b is in its simplest form.

end PU_squared_fraction_l119_119995


namespace comic_books_ratio_l119_119037

variable (S : ℕ)

def initial_comics := 22
def remaining_comics := 17
def comics_bought := 6

theorem comic_books_ratio (h1 : initial_comics - S + comics_bought = remaining_comics) :
  (S : ℚ) / initial_comics = 1 / 2 := by
  sorry

end comic_books_ratio_l119_119037


namespace m_range_l119_119432

noncomputable def range_of_m (m : ℝ) : Prop :=
  ∃ x : ℝ, 
    x ≠ 2 ∧ 
    (x + m) / (x - 2) - 3 = (x - 1) / (2 - x) ∧ 
    x ≥ 0

theorem m_range (m : ℝ) : 
  range_of_m m ↔ m ≥ -5 ∧ m ≠ -3 := 
sorry

end m_range_l119_119432


namespace possible_remainders_of_a2_l119_119048

theorem possible_remainders_of_a2 (p : ℕ) (k : ℕ) (hp : Nat.Prime p) (hk : 0 < k) 
  (hresidue : ∀ i : ℕ, i < p → ∃ j : ℕ, j < p ∧ ((j^k+j) % p = i)) :
  ∃ s : Finset ℕ, s = Finset.range p ∧ (2^k + 2) % p ∈ s := 
sorry

end possible_remainders_of_a2_l119_119048


namespace math_problem_l119_119151

noncomputable def exponential_sequence (a : ℕ → ℝ) : Prop :=
∀ n, a n = 2 * 3^(n - 1)

noncomputable def geometric_sum (a : ℕ → ℝ) (n : ℕ) : ℝ :=
(2 * 3^n - 2) / 2

theorem math_problem 
  (a : ℕ → ℝ) (b : ℕ → ℕ) (c : ℕ → ℝ) (S T P : ℕ → ℝ)
  (h1 : exponential_sequence a)
  (h2 : a 1 * a 3 = 36)
  (h3 : a 3 + a 4 = 9 * (a 1 + a 2))
  (h4 : ∀ n, S n + 1 = 3^(b n))
  (h5 : ∀ n, T n = (2 * n - 1) * 3^n / 2 + 1 / 2)
  (h6 : ∀ n, c n = a n / ((a n + 1) * (a (n + 1) + 1)))
  (h7 : ∀ n, P (2 * n) = 1 / 6 - 1 / (4 * 3^(2 * n) + 2)) :
  (∀ n, a n = 2 * 3^(n - 1)) ∧
  ∀ n, b n = n ∧
  ∀ n, a n * b n = 2 * n * 3^(n - 1) ∧
  ∃ n, T n = (2 * n - 1) * 3^n / 2 + 1 / 2 ∧
  P (2 * n) = 1 / 6 - 1 / (4 * 3^(2 * n) + 2) :=
by sorry

end math_problem_l119_119151


namespace inverse_proposition_l119_119901

theorem inverse_proposition (q_1 q_2 : ℚ) :
  (q_1 ^ 2 = q_2 ^ 2 → q_1 = q_2) ↔ (q_1 = q_2 → q_1 ^ 2 = q_2 ^ 2) :=
sorry

end inverse_proposition_l119_119901


namespace value_of_X_l119_119434

def M := 2007 / 3
def N := M / 3
def X := M - N

theorem value_of_X : X = 446 := by
  sorry

end value_of_X_l119_119434


namespace even_function_value_at_2_l119_119222

theorem even_function_value_at_2 {a : ℝ} (h : ∀ x : ℝ, (x + 1) * (x - a) = (-x + 1) * (-x - a)) : 
  ((2 + 1) * (2 - a)) = 3 := by
  sorry

end even_function_value_at_2_l119_119222


namespace ashley_friends_ages_correct_sum_l119_119493

noncomputable def ashley_friends_ages_sum : Prop :=
  ∃ (a b c d : ℕ), (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧
                   (1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ 1 ≤ c ∧ c ≤ 9 ∧ 1 ≤ d ∧ d ≤ 9) ∧
                   (a * b = 36) ∧ (c * d = 30) ∧ (a + b + c + d = 24)

theorem ashley_friends_ages_correct_sum : ashley_friends_ages_sum := sorry

end ashley_friends_ages_correct_sum_l119_119493


namespace arithmetic_sequence_find_m_l119_119546

theorem arithmetic_sequence_find_m (S : ℕ → ℤ) (m : ℕ)
  (h1 : S (m - 1) = -2)
  (h2 : S m = 0)
  (h3 : S (m + 1) = 3) :
  m = 5 :=
sorry

end arithmetic_sequence_find_m_l119_119546


namespace wine_barrels_l119_119341

theorem wine_barrels :
  ∃ x y : ℝ, (6 * x + 4 * y = 48) ∧ (5 * x + 3 * y = 38) :=
by
  -- Proof is left out
  sorry

end wine_barrels_l119_119341


namespace no_odd_integers_satisfy_equation_l119_119090

theorem no_odd_integers_satisfy_equation :
  ¬ ∃ (x y z : ℤ), (x % 2 ≠ 0) ∧ (y % 2 ≠ 0) ∧ (z % 2 ≠ 0) ∧ 
  (x + y)^2 + (x + z)^2 = (y + z)^2 :=
by
  sorry

end no_odd_integers_satisfy_equation_l119_119090


namespace geometric_sequence_sum_l119_119340

noncomputable def geom_seq (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a 1 * q^n

theorem geometric_sequence_sum :
  ∃ (a : ℕ → ℝ) (q : ℝ), 
  (∀ n : ℕ, a (n + 1) = a 1 * q^n) ∧ 
  (a 2 * a 4 = 1) ∧ 
  (a 1 * (q^0 + q^1 + q^2) = 7) ∧ 
  (a 1 / (1 - q) * (1 - q^5) = 31 / 4) := by
  sorry

end geometric_sequence_sum_l119_119340


namespace calculate_value_l119_119631

theorem calculate_value : (2200 - 2090)^2 / (144 + 25) = 64 := 
by
  sorry

end calculate_value_l119_119631


namespace percentage_calculation_l119_119573

theorem percentage_calculation :
  let total_amt := 1600
  let pct_25 := 0.25 * total_amt
  let pct_5 := 0.05 * pct_25
  pct_5 = 20 := by
sorry

end percentage_calculation_l119_119573


namespace pyramid_base_edge_length_l119_119652

theorem pyramid_base_edge_length (height : ℝ) (radius : ℝ) (side_len : ℝ) :
  height = 4 ∧ radius = 3 →
  side_len = (12 * Real.sqrt 14) / 7 :=
by
  intros h
  rcases h with ⟨h1, h2⟩
  sorry

end pyramid_base_edge_length_l119_119652


namespace problem_a_problem_b_l119_119252

-- Define necessary elements for the problem
def is_divisible_by_seven (n : ℕ) : Prop := n % 7 = 0

-- Define the method to check divisibility by seven
noncomputable def check_divisibility_by_seven (n : ℕ) : ℕ :=
  let last_digit := n % 10
  let remaining_digits := n / 10
  remaining_digits - 2 * last_digit

-- Problem a: Prove that 4578 is divisible by 7
theorem problem_a : is_divisible_by_seven 4578 :=
  sorry

-- Problem b: Prove that there are 13 three-digit numbers of the form AB5 divisible by 7
theorem problem_b : ∃ (count : ℕ), count = 13 ∧ (∀ a b : ℕ, a ≠ 0 ∧ 1 ≤ a ∧ a < 10 ∧ 0 ≤ b ∧ b < 10 → is_divisible_by_seven (100 * a + 10 * b + 5) → count = count + 1) :=
  sorry

end problem_a_problem_b_l119_119252


namespace nth_group_sum_correct_l119_119975

-- Define the function that computes the sum of the numbers in the nth group
def nth_group_sum (n : ℕ) : ℕ :=
  n * (n^2 + 1) / 2

-- The theorem statement
theorem nth_group_sum_correct (n : ℕ) : 
  nth_group_sum n = n * (n^2 + 1) / 2 := by
  sorry

end nth_group_sum_correct_l119_119975


namespace ordering_eight_four_three_l119_119732

noncomputable def eight_pow_ten := 8 ^ 10
noncomputable def four_pow_fifteen := 4 ^ 15
noncomputable def three_pow_twenty := 3 ^ 20

theorem ordering_eight_four_three :
  eight_pow_ten < three_pow_twenty ∧ three_pow_twenty < four_pow_fifteen :=
by
  sorry

end ordering_eight_four_three_l119_119732


namespace gcd_306_522_l119_119145

theorem gcd_306_522 : Nat.gcd 306 522 = 18 := 
  by sorry

end gcd_306_522_l119_119145


namespace smallest_positive_period_max_min_values_l119_119220

noncomputable def f (x a : ℝ) : ℝ :=
  (Real.cos x) * (2 * Real.sqrt 3 * Real.sin x - Real.cos x) + a * Real.sin x ^ 2

theorem smallest_positive_period (a : ℝ) (h : f (Real.pi / 12) a = 0) : 
  ∃ T : ℝ, T > 0 ∧ (∀ x, f (x + T) a = f x a) ∧ (∀ ε > 0, ε < T → ∃ y, y < T ∧ f y a ≠ f 0 a) := 
sorry

theorem max_min_values (a : ℝ) (h : f (Real.pi / 12) a = 0) :
  (∀ x ∈ Set.Icc (-Real.pi / 6) (Real.pi / 4), f x a ≤ Real.sqrt 3) ∧ 
  (∀ x ∈ Set.Icc (-Real.pi / 6) (Real.pi / 4), -2 ≤ f x a) := 
sorry

end smallest_positive_period_max_min_values_l119_119220


namespace problem1_problem2_l119_119647

-- Definition of the function f(x)
def f (x a : ℝ) : ℝ := |2 * x - a| + |2 * x - 1|

-- 1st problem: Prove the solution set for f(x) ≤ 2 when a = -1 is { x | x = ± 1/2 }
theorem problem1 : (∀ x : ℝ, f x (-1) ≤ 2 ↔ x = 1/2 ∨ x = -1/2) :=
by sorry

-- 2nd problem: Prove the range of real number a is [0, 3]
theorem problem2 : (∃ a : ℝ, (∀ x ∈ Set.Icc (1/2:ℝ) 1, f x a ≤ |2 * x + 1| ) ↔ 0 ≤ a ∧ a ≤ 3) :=
by sorry

end problem1_problem2_l119_119647


namespace yoque_payment_months_l119_119447

-- Define the conditions
def monthly_payment : ℝ := 15
def amount_borrowed : ℝ := 150
def total_payment : ℝ := amount_borrowed * 1.1

-- Define the proof problem
theorem yoque_payment_months :
  ∃ (n : ℕ), n * monthly_payment = total_payment :=
by 
  have monthly_payment : ℝ := 15
  have amount_borrowed : ℝ := 150
  have total_payment : ℝ := amount_borrowed * 1.1
  use 11
  sorry

end yoque_payment_months_l119_119447


namespace total_fish_catch_l119_119325

noncomputable def Johnny_fishes : ℕ := 8
noncomputable def Sony_fishes : ℕ := 4 * Johnny_fishes
noncomputable def total_fishes : ℕ := Sony_fishes + Johnny_fishes

theorem total_fish_catch : total_fishes = 40 := by
  sorry

end total_fish_catch_l119_119325


namespace range_of_q_l119_119698

variable (a_n : ℕ → ℝ) (q : ℝ) (S_n : ℕ → ℝ)
variable (hg_seq : ∀ n : ℕ, n > 0 → ∃ a_1 : ℝ, S_n n = a_1 * (1 - q ^ n) / (1 - q))
variable (pos_sum : ∀ n : ℕ, n > 0 → S_n n > 0)

theorem range_of_q : q ∈ Set.Ioo (-1 : ℝ) 0 ∪ Set.Ioi (0 : ℝ) := sorry

end range_of_q_l119_119698


namespace marcia_wardrobe_cost_l119_119893

theorem marcia_wardrobe_cost :
  let skirt_price := 20
  let blouse_price := 15
  let pant_price := 30
  let num_skirts := 3
  let num_blouses := 5
  let num_pants := 2
  let pant_offer := buy_1_get_1_half
  let skirt_cost := num_skirts * skirt_price
  let blouse_cost := num_blouses * blouse_price
  let pant_full_price := pant_price
  let pant_half_price := pant_price / 2
  let pant_cost := pant_full_price + pant_half_price
  let total_cost := skirt_cost + blouse_cost + pant_cost
  total_cost = 180 :=
by
  sorry -- proof is omitted

end marcia_wardrobe_cost_l119_119893


namespace length_of_the_train_l119_119227

noncomputable def length_of_train (s1 s2 : ℝ) (t1 t2 : ℕ) : ℝ :=
  (s1 * t1 + s2 * t2) / 2

theorem length_of_the_train :
  ∀ (s1 s2 : ℝ) (t1 t2 : ℕ), s1 = 25 → t1 = 8 → s2 = 100 / 3 → t2 = 6 → length_of_train s1 s2 t1 t2 = 200 :=
by
  intros s1 s2 t1 t2 hs1 ht1 hs2 ht2
  rw [hs1, ht1, hs2, ht2]
  simp [length_of_train]
  norm_num

end length_of_the_train_l119_119227


namespace find_k_eq_neg2_l119_119943

theorem find_k_eq_neg2 (k : ℝ) (h : (-1)^2 - k * (-1) + 1 = 0) : k = -2 :=
by sorry

end find_k_eq_neg2_l119_119943


namespace probability_three_blue_jellybeans_l119_119379

theorem probability_three_blue_jellybeans:
  let total_jellybeans := 20
  let blue_jellybeans := 10
  let red_jellybeans := 10
  let draws := 3
  let q := (1 / 2) * (9 / 19) * (4 / 9)
  q = 2 / 19 :=
sorry

end probability_three_blue_jellybeans_l119_119379


namespace number_of_solutions_l119_119172

theorem number_of_solutions :
  (∃ (a b c : ℕ), 4 * a = 6 * c ∧ 168 * a = 6 * a * b * c) → 
  ∃ (s : Finset ℕ), s.card = 6 :=
by sorry

end number_of_solutions_l119_119172


namespace calculate_expression_l119_119496

theorem calculate_expression : (3 / 4 - 1 / 8) ^ 5 = 3125 / 32768 :=
by
  sorry

end calculate_expression_l119_119496


namespace chord_length_intercepted_by_curve_l119_119823

theorem chord_length_intercepted_by_curve
(param_eqns : ∀ θ : ℝ, (x = 2 * Real.cos θ ∧ y = 1 + 2 * Real.sin θ))
(line_eqn : 3 * x - 4 * y - 1 = 0) :
  ∃ (chord_length : ℝ), chord_length = 2 * Real.sqrt 3 := 
sorry

end chord_length_intercepted_by_curve_l119_119823


namespace factor_expression_l119_119255

variable (x : ℝ)

theorem factor_expression : 75 * x^3 - 250 * x^7 = 25 * x^3 * (3 - 10 * x^4) :=
by
  sorry

end factor_expression_l119_119255


namespace find_a_maximize_profit_l119_119689

-- Definition of parameters
def a := 260
def purchase_price_table := a
def purchase_price_chair := a - 140

-- Condition 1: The number of dining chairs purchased for 600 yuan is the same as the number of dining tables purchased for 1300 yuan.
def condition1 := (600 / (purchase_price_chair : ℚ)) = (1300 / (purchase_price_table : ℚ))

-- Given conditions for profit maximization
def qty_tables := 30
def qty_chairs := 5 * qty_tables + 20
def total_qty := qty_tables + qty_chairs

-- Condition: Total quantity of items does not exceed 200 units.
def condition2 := total_qty ≤ 200

-- Profit calculation
def profit := 280 * qty_tables + 800

-- Theorem statements
theorem find_a : condition1 → a = 260 := sorry

theorem maximize_profit : condition2 ∧ (8 * qty_tables + 800 > 0) → 
  (qty_tables = 30) ∧ (qty_chairs = 170) ∧ (profit = 9200) := sorry

end find_a_maximize_profit_l119_119689


namespace selling_price_correct_l119_119426

/-- Define the total number of units to be sold -/
def total_units : ℕ := 5000

/-- Define the variable cost per unit -/
def variable_cost_per_unit : ℕ := 800

/-- Define the total fixed costs -/
def fixed_costs : ℕ := 1000000

/-- Define the desired profit -/
def desired_profit : ℕ := 1500000

/-- The selling price p must be calculated such that revenues exceed expenses by the desired profit -/
theorem selling_price_correct : 
  ∃ p : ℤ, p = 1300 ∧ (total_units * p) - (fixed_costs + (total_units * variable_cost_per_unit)) = desired_profit :=
by
  sorry

end selling_price_correct_l119_119426


namespace third_group_members_l119_119966

-- Define the total number of members in the choir
def total_members : ℕ := 70

-- Define the number of members in the first group
def first_group_members : ℕ := 25

-- Define the number of members in the second group
def second_group_members : ℕ := 30

-- Prove that the number of members in the third group is 15
theorem third_group_members : total_members - first_group_members - second_group_members = 15 := 
by 
  sorry

end third_group_members_l119_119966


namespace third_month_sale_l119_119622

theorem third_month_sale (s1 s2 s4 s5 s6 avg_sale: ℕ) (h1: s1 = 5420) (h2: s2 = 5660) (h3: s4 = 6350) (h4: s5 = 6500) (h5: s6 = 8270) (h6: avg_sale = 6400) :
  ∃ s3: ℕ, s3 = 6200 :=
by
  sorry

end third_month_sale_l119_119622


namespace vanessa_earnings_l119_119376

def cost : ℕ := 4
def total_bars : ℕ := 11
def bars_unsold : ℕ := 7
def bars_sold : ℕ := total_bars - bars_unsold
def money_made : ℕ := bars_sold * cost

theorem vanessa_earnings : money_made = 16 := by
  sorry

end vanessa_earnings_l119_119376


namespace eggs_total_l119_119738

-- Definitions based on conditions
def isPackageSize (n : Nat) : Prop :=
  n = 6 ∨ n = 11

def numLargePacks : Nat := 5

def largePackSize : Nat := 11

-- Mathematical statement to prove
theorem eggs_total : ∃ totalEggs : Nat, totalEggs = numLargePacks * largePackSize :=
  by sorry

end eggs_total_l119_119738


namespace number_of_ordered_triples_modulo_1000000_l119_119463

def p : ℕ := 2017
def N : ℕ := sorry -- N is the number of ordered triples (a, b, c)

theorem number_of_ordered_triples_modulo_1000000 (N : ℕ) (h : ∀ (a b c : ℕ), 1 ≤ a ∧ a ≤ p * (p - 1) ∧ 1 ≤ b ∧ b ≤ p * (p - 1) ∧ a^b - b^a = p * c → true) : 
  N % 1000000 = 2016 :=
sorry

end number_of_ordered_triples_modulo_1000000_l119_119463


namespace mod_sum_example_l119_119114

theorem mod_sum_example :
  (9^5 + 8^4 + 7^6) % 5 = 4 :=
by sorry

end mod_sum_example_l119_119114


namespace shooter_mean_hits_l119_119607

theorem shooter_mean_hits (p : ℝ) (n : ℕ) (h_prob : p = 0.9) (h_shots : n = 10) : n * p = 9 := by
  sorry

end shooter_mean_hits_l119_119607


namespace distribution_of_balls_l119_119920

theorem distribution_of_balls (n k : ℕ) (h_n : n = 6) (h_k : k = 3) : k^n = 729 := by
  rw [h_n, h_k]
  exact rfl

end distribution_of_balls_l119_119920


namespace fraction_equality_l119_119912

-- Defining the main problem statement
theorem fraction_equality (x y z : ℚ) (k : ℚ) 
  (h1 : x = 3 * k) (h2 : y = 5 * k) (h3 : z = 7 * k) :
  (y + z) / (3 * x - y) = 3 :=
by
  sorry

end fraction_equality_l119_119912


namespace log_inequality_l119_119808

theorem log_inequality (a b c : ℝ) (ha : 1 < a) (hb : 1 < b) (hc : 1 < c) :
  2 * ((Real.log a / Real.log b) / (a + b) + (Real.log b / Real.log c) / (b + c) + (Real.log c / Real.log a) / (c + a)) 
    ≥ 9 / (a + b + c) :=
by
  sorry

end log_inequality_l119_119808


namespace range_of_sin_cos_expression_l119_119199

variable (a b c A B C : ℝ)

theorem range_of_sin_cos_expression
  (h1 : a = b)
  (h2 : c * Real.sin A = -a * Real.cos C) :
  1 < 2 * Real.sin (A + Real.pi / 6) :=
sorry

end range_of_sin_cos_expression_l119_119199


namespace range_of_a_opposite_sides_l119_119185

theorem range_of_a_opposite_sides (a : ℝ) :
  (3 * (-2) - 2 * 1 - a) * (3 * 1 - 2 * 1 - a) < 0 ↔ -8 < a ∧ a < 1 := by
  sorry

end range_of_a_opposite_sides_l119_119185


namespace sum_of_ages_twins_l119_119126

-- Define that Evan has two older twin sisters and their ages are such that the product of all three ages is 162
def twin_sisters_ages (a : ℕ) (b : ℕ) (c : ℕ) : Prop :=
  a * b * c = 162

-- Given the above definition, we need to prove the sum of these ages is 20
theorem sum_of_ages_twins (a b c : ℕ) (h : twin_sisters_ages a b c) (ha : b = c) : a + b + c = 20 :=
by 
  sorry

end sum_of_ages_twins_l119_119126


namespace number_of_primes_between_30_and_50_l119_119480

/-- 
  Prove that there are exactly 5 prime numbers in the range from 30 to 50. 
  These primes are 31, 37, 41, 43, and 47.
-/
theorem number_of_primes_between_30_and_50 : 
  (Finset.filter Nat.Prime (Finset.range 51)).card - 
  (Finset.filter Nat.Prime (Finset.range 31)).card = 5 := 
by 
  sorry

end number_of_primes_between_30_and_50_l119_119480


namespace gasoline_tank_capacity_l119_119375

theorem gasoline_tank_capacity (x : ℕ) (h1 : 5 * x / 6 - 2 * x / 3 = 15) : x = 90 :=
sorry

end gasoline_tank_capacity_l119_119375


namespace power_of_six_evaluation_l119_119521

noncomputable def example_expr : ℝ := (6 : ℝ)^(1/4) / (6 : ℝ)^(1/6)

theorem power_of_six_evaluation : example_expr = (6 : ℝ)^(1/12) := 
by
  sorry

end power_of_six_evaluation_l119_119521


namespace platform_length_l119_119519

theorem platform_length (length_train : ℝ) (speed_train_kmph : ℝ) (time_sec : ℝ) (length_platform : ℝ) :
  length_train = 1020 → speed_train_kmph = 102 → time_sec = 50 →
  length_platform = (speed_train_kmph * 1000 / 3600) * time_sec - length_train :=
by
  intros
  sorry

end platform_length_l119_119519


namespace derivative_of_y_correct_l119_119511

noncomputable def derivative_of_y (x : ℝ) : ℝ :=
  let y := (4^x * (Real.log 4 * Real.sin (4 * x) - 4 * Real.cos (4 * x))) / (16 + (Real.log 4) ^ 2)
  let u := 4^x * (Real.log 4 * Real.sin (4 * x) - 4 * Real.cos (4 * x))
  let v := 16 + (Real.log 4) ^ 2
  let du_dx := (4^x * Real.log 4) * (Real.log 4 * Real.sin (4 * x) - 4 * Real.cos (4 * x)) +
               (4^x) * (4 * Real.log 4 * Real.cos (4 * x) + 16 * Real.sin (4 * x))
  let dv_dx := 0
  (du_dx * v - u * dv_dx) / (v ^ 2)

theorem derivative_of_y_correct (x : ℝ) : derivative_of_y x = 4^x * Real.sin (4 * x) :=
  sorry

end derivative_of_y_correct_l119_119511


namespace cricket_initial_avg_runs_l119_119907

theorem cricket_initial_avg_runs (A : ℝ) (h : 11 * (A + 4) = 10 * A + 86) : A = 42 :=
sorry

end cricket_initial_avg_runs_l119_119907


namespace C_completion_time_l119_119189

noncomputable def racer_time (v_C : ℝ) : ℝ := 100 / v_C

theorem C_completion_time
  (v_A v_B v_C : ℝ)
  (h1 : 100 / v_A = 10)
  (h2 : 85 / v_B = 10)
  (h3 : 90 / v_C = 100 / v_B) :
  racer_time v_C = 13.07 :=
by
  sorry

end C_completion_time_l119_119189


namespace min_socks_for_pairs_l119_119474

-- Definitions for conditions
def pairs_of_socks : ℕ := 4
def sizes : ℕ := 2
def colors : ℕ := 2

-- Theorem statement
theorem min_socks_for_pairs : 
  ∃ n, n = 7 ∧ 
  ∀ (socks : ℕ), socks >= pairs_of_socks → socks ≥ 7 :=
sorry

end min_socks_for_pairs_l119_119474


namespace max_value_quadratic_function_l119_119913

noncomputable def quadratic_function (x : ℝ) : ℝ :=
  -3 * x^2 + 8

theorem max_value_quadratic_function : ∃(x : ℝ), quadratic_function x = 8 :=
by
  sorry

end max_value_quadratic_function_l119_119913


namespace relationship_of_y_values_l119_119242

theorem relationship_of_y_values (m : ℝ) (y1 y2 y3 : ℝ) :
  (∀ x y, (x = -2 ∧ y = y1 ∨ x = -1 ∧ y = y2 ∨ x = 1 ∧ y = y3) → (y = (m^2 + 1) / x)) →
  y2 < y1 ∧ y1 < y3 :=
by
  sorry

end relationship_of_y_values_l119_119242


namespace max_min_condition_monotonic_condition_l119_119963

-- (1) Proving necessary and sufficient condition for f(x) to have both a maximum and minimum value
theorem max_min_condition (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ > 0 ∧ x₂ > 0 ∧ -2*x₁ + a - (1/x₁) = 0 ∧ -2*x₂ + a - (1/x₂) = 0) ↔ a > Real.sqrt 8 :=
sorry

-- (2) Proving the range of values for a such that f(x) is monotonic on [1, 2]
theorem monotonic_condition (a : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → (-2 * x + a - (1 / x)) ≥ 0) ∨
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → (-2 * x + a - (1 / x)) ≤ 0) ↔ a ≤ 3 ∨ a ≥ 4.5 :=
sorry

end max_min_condition_monotonic_condition_l119_119963


namespace sin_cos_inequality_l119_119205

theorem sin_cos_inequality (x : ℝ) (n : ℕ) : 
  (Real.sin (2 * x))^n + (Real.sin x^n - Real.cos x^n)^2 ≤ 1 := 
by
  sorry

end sin_cos_inequality_l119_119205


namespace count_blanks_l119_119399

theorem count_blanks (B : ℝ) (h1 : 10 + B = T) (h2 : 0.7142857142857143 = B / T) : B = 25 :=
by
  -- The conditions are taken into account as definitions or parameters
  -- We skip the proof itself by using 'sorry'
  sorry

end count_blanks_l119_119399


namespace geometric_sequence_sum_l119_119313

theorem geometric_sequence_sum (q a₁ : ℝ) (hq : q > 1) (h₁ : a₁ + a₁ * q^3 = 18) (h₂ : a₁^2 * q^3 = 32) :
  (a₁ * (1 - q^8) / (1 - q) = 510) :=
by
  sorry

end geometric_sequence_sum_l119_119313


namespace person_age_l119_119954

-- Define the conditions
def current_age : ℕ := 18

-- Define the equation based on the person's statement
def age_equation (A Y : ℕ) : Prop := 3 * (A + 3) - 3 * (A - Y) = A

-- Statement to be proven
theorem person_age (Y : ℕ) : 
  age_equation current_age Y → Y = 3 := 
by 
  sorry

end person_age_l119_119954


namespace sum_of_products_l119_119846

def is_positive (x : ℝ) := 0 < x

theorem sum_of_products 
  (x y z : ℝ) 
  (hx : is_positive x)
  (hy : is_positive y)
  (hz : is_positive z)
  (h1 : x^2 + x * y + y^2 = 27)
  (h2 : y^2 + y * z + z^2 = 25)
  (h3 : z^2 + z * x + x^2 = 52) :
  x * y + y * z + z * x = 30 :=
  sorry

end sum_of_products_l119_119846


namespace find_prices_l119_119513

def price_system_of_equations (x y : ℕ) : Prop :=
  3 * x + 2 * y = 474 ∧ x - y = 8

theorem find_prices (x y : ℕ) :
  price_system_of_equations x y :=
by
  sorry

end find_prices_l119_119513


namespace twelfth_term_of_geometric_sequence_l119_119917

theorem twelfth_term_of_geometric_sequence (a : ℕ) (r : ℕ) (h1 : a * r ^ 4 = 8) (h2 : a * r ^ 8 = 128) : 
  a * r ^ 11 = 1024 :=
sorry

end twelfth_term_of_geometric_sequence_l119_119917


namespace number_of_students_l119_119338

theorem number_of_students (T : ℕ) (n : ℕ) (h1 : (T + 20) / n = T / n + 1 / 2) : n = 40 :=
  sorry

end number_of_students_l119_119338


namespace gcd_372_684_is_12_l119_119352

theorem gcd_372_684_is_12 :
  Nat.gcd 372 684 = 12 :=
sorry

end gcd_372_684_is_12_l119_119352


namespace functional_eq_solutions_l119_119485

-- Define the conditions for the problem
def func_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * f y - y * f x) = f (x * y) - x * y

-- Define the two solutions to be proven correct
def f1 (x : ℝ) : ℝ := x
def f2 (x : ℝ) : ℝ := |x|

-- State the main theorem to be proven
theorem functional_eq_solutions (f : ℝ → ℝ) (h : func_equation f) : f = f1 ∨ f = f2 :=
sorry

end functional_eq_solutions_l119_119485


namespace abs_inequality_solution_l119_119460

theorem abs_inequality_solution (x : ℝ) : 
  (|x - 1| < 1) ↔ (0 < x ∧ x < 2) :=
sorry

end abs_inequality_solution_l119_119460


namespace probability_not_buy_l119_119033

-- Define the given probability of Sam buying a new book
def P_buy : ℚ := 5 / 8

-- Theorem statement: The probability that Sam will not buy a new book is 3 / 8
theorem probability_not_buy : 1 - P_buy = 3 / 8 :=
by
  -- Proof omitted
  sorry

end probability_not_buy_l119_119033


namespace find_f_28_l119_119336

theorem find_f_28 (f : ℕ → ℚ) (h1 : ∀ n : ℕ, f (n + 1) = (3 * f n + n) / 3) (h2 : f 1 = 1) :
  f 28 = 127 := by
sorry

end find_f_28_l119_119336


namespace split_payment_l119_119866

noncomputable def Rahul_work_per_day := (1 : ℝ) / 3
noncomputable def Rajesh_work_per_day := (1 : ℝ) / 2
noncomputable def Ritesh_work_per_day := (1 : ℝ) / 4

noncomputable def total_work_per_day := Rahul_work_per_day + Rajesh_work_per_day + Ritesh_work_per_day

noncomputable def Rahul_proportion := Rahul_work_per_day / total_work_per_day
noncomputable def Rajesh_proportion := Rajesh_work_per_day / total_work_per_day
noncomputable def Ritesh_proportion := Ritesh_work_per_day / total_work_per_day

noncomputable def total_payment := 510

noncomputable def Rahul_share := Rahul_proportion * total_payment
noncomputable def Rajesh_share := Rajesh_proportion * total_payment
noncomputable def Ritesh_share := Ritesh_proportion * total_payment

theorem split_payment :
  Rahul_share + Rajesh_share + Ritesh_share = total_payment :=
by
  sorry

end split_payment_l119_119866


namespace greatest_value_product_l119_119624

def is_prime (n : ℕ) : Prop := n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def divisible_by (m n : ℕ) : Prop := ∃ k, m = k * n

theorem greatest_value_product (a b : ℕ) : 
    is_prime a → is_prime b → a < 10 → b < 10 → divisible_by (110 + 10 * a + b) 55 → a * b = 15 :=
by
    sorry

end greatest_value_product_l119_119624


namespace min_m_plus_n_l119_119606

theorem min_m_plus_n (m n : ℕ) (h1 : 0 < m) (h2 : 0 < n) (h3 : 32 * m = n^5) : m + n = 3 :=
  sorry

end min_m_plus_n_l119_119606


namespace john_weekly_earnings_increase_l119_119056

theorem john_weekly_earnings_increase :
  let earnings_before := 60 + 100
  let earnings_after := 78 + 120
  let increase := earnings_after - earnings_before
  (increase / earnings_before : ℚ) * 100 = 23.75 :=
by
  -- Definitions
  let earnings_before := (60 : ℚ) + 100
  let earnings_after := (78 : ℚ) + 120
  let increase := earnings_after - earnings_before

  -- Calculation of percentage increase
  let percentage_increase : ℚ := (increase / earnings_before) * 100

  -- Expected result
  have expected_result : percentage_increase = 23.75 := by sorry
  exact expected_result

end john_weekly_earnings_increase_l119_119056


namespace remainder_of_n_when_divided_by_7_l119_119215

theorem remainder_of_n_when_divided_by_7 (n : ℕ) :
  (n^2 ≡ 2 [MOD 7]) ∧ (n^3 ≡ 6 [MOD 7]) → (n ≡ 3 [MOD 7]) :=
by sorry

end remainder_of_n_when_divided_by_7_l119_119215


namespace ratio_doubled_to_original_l119_119580

theorem ratio_doubled_to_original (x : ℝ) (h : 3 * (2 * x + 9) = 69) : (2 * x) / x = 2 :=
by
  -- We skip the proof here.
  sorry

end ratio_doubled_to_original_l119_119580


namespace double_bed_heavier_l119_119466

-- Define the problem conditions
variable (S D B : ℝ)
variable (h1 : 5 * S = 50)
variable (h2 : 2 * S + 4 * D + 3 * B = 180)
variable (h3 : 3 * B = 60)

-- Define the goal to prove
theorem double_bed_heavier (S D B : ℝ) (h1 : 5 * S = 50) (h2 : 2 * S + 4 * D + 3 * B = 180) (h3 : 3 * B = 60) : D - S = 15 :=
by
  sorry

end double_bed_heavier_l119_119466


namespace number_of_families_l119_119456

theorem number_of_families (x : ℕ) (h1 : x + x / 3 = 100) : x = 75 :=
sorry

end number_of_families_l119_119456


namespace man_speed_is_correct_l119_119179

noncomputable def speed_of_man (train_speed_kmh : ℝ) (train_length_m : ℝ) (time_to_pass_s : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let relative_speed_ms := train_length_m / time_to_pass_s
  let man_speed_ms := relative_speed_ms - train_speed_ms
  man_speed_ms * 3600 / 1000

theorem man_speed_is_correct : 
  speed_of_man 60 110 5.999520038396929 = 6.0024 := 
by
  sorry

end man_speed_is_correct_l119_119179


namespace total_cats_in_training_center_l119_119822

-- Definitions corresponding to the given conditions
def cats_can_jump : ℕ := 60
def cats_can_fetch : ℕ := 35
def cats_can_meow : ℕ := 40
def cats_jump_fetch : ℕ := 20
def cats_fetch_meow : ℕ := 15
def cats_jump_meow : ℕ := 25
def cats_all_three : ℕ := 11
def cats_none : ℕ := 10

-- Theorem statement corresponding to proving question == answer given conditions
theorem total_cats_in_training_center
    (cjump : ℕ := cats_can_jump)
    (cfetch : ℕ := cats_can_fetch)
    (cmeow : ℕ := cats_can_meow)
    (cjf : ℕ := cats_jump_fetch)
    (cfm : ℕ := cats_fetch_meow)
    (cjm : ℕ := cats_jump_meow)
    (cat : ℕ := cats_all_three)
    (cno : ℕ := cats_none) :
    cjump
    + cfetch
    + cmeow
    - cjf
    - cfm
    - cjm
    + cat
    + cno
    = 96 := sorry

end total_cats_in_training_center_l119_119822


namespace pumps_280_gallons_in_30_minutes_l119_119165

def hydraflow_rate_per_hour := 560 -- gallons per hour
def time_fraction_in_hour := 1 / 2

theorem pumps_280_gallons_in_30_minutes : hydraflow_rate_per_hour * time_fraction_in_hour = 280 := by
  sorry

end pumps_280_gallons_in_30_minutes_l119_119165


namespace solution_set_of_inequality_l119_119550

theorem solution_set_of_inequality (x : ℝ) (hx : x ≠ 0) :
  (x + 1) / x ≤ 3 ↔ x ∈ Set.Iio 0 ∪ Set.Ici 0.5 :=
by sorry

end solution_set_of_inequality_l119_119550


namespace min_value_l119_119956

theorem min_value (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a * (a + b + c) + b * c = 4 - 2 * Real.sqrt 3) :
  2 * a + b + c ≥ 2 * Real.sqrt 3 - 2 :=
sorry

end min_value_l119_119956


namespace tel_aviv_rain_probability_l119_119646

def binom (n k : ℕ) : ℕ := Nat.choose n k

def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (binom n k : ℝ) * (p ^ k) * ((1 - p) ^ (n - k))

theorem tel_aviv_rain_probability :
  binomial_probability 6 4 0.5 = 0.234375 :=
by
  sorry

end tel_aviv_rain_probability_l119_119646


namespace geometric_sequence_product_l119_119585

theorem geometric_sequence_product (a : ℕ → ℝ) (r : ℝ) (h_geom : ∀ n, a (n+1) = r * a n) (h_cond : a 7 * a 12 = 5) :
  a 8 * a 9 * a 10 * a 11 = 25 :=
by 
  sorry

end geometric_sequence_product_l119_119585


namespace num_children_eq_3_l119_119406

-- Definitions from the conditions
def regular_ticket_cost : ℕ := 9
def child_ticket_discount : ℕ := 2
def given_amount : ℕ := 20 * 2
def received_change : ℕ := 1
def num_adults : ℕ := 2

-- Derived data
def total_ticket_cost : ℕ := given_amount - received_change
def adult_ticket_cost : ℕ := num_adults * regular_ticket_cost
def children_ticket_cost : ℕ := total_ticket_cost - adult_ticket_cost
def child_ticket_cost : ℕ := regular_ticket_cost - child_ticket_discount

-- Statement to prove
theorem num_children_eq_3 : (children_ticket_cost / child_ticket_cost) = 3 := by
  sorry

end num_children_eq_3_l119_119406


namespace digit_sum_solution_l119_119727

def S (n : ℕ) : ℕ := (n.digits 10).sum

theorem digit_sum_solution : S (S (S (S (2017 ^ 2017)))) = 1 := 
by
  sorry

end digit_sum_solution_l119_119727


namespace rooks_control_chosen_squares_l119_119295

theorem rooks_control_chosen_squares (n : Nat) 
  (chessboard : Fin (2 * n) × Fin (2 * n)) 
  (chosen_squares : Finset (Fin (2 * n) × Fin (2 * n))) 
  (h : chosen_squares.card = 3 * n) :
  ∃ rooks : Finset (Fin (2 * n) × Fin (2 * n)), rooks.card = n ∧
  ∀ (square : Fin (2 * n) × Fin (2 * n)), square ∈ chosen_squares → 
  (square ∈ rooks ∨ ∃ (rook : Fin (2 * n) × Fin (2 * n)) (hr : rook ∈ rooks), 
  rook.1 = square.1 ∨ rook.2 = square.2) :=
sorry

end rooks_control_chosen_squares_l119_119295


namespace find_X_eq_A_l119_119844

variable {α : Type*}
variable (A X : Set α)

theorem find_X_eq_A (h : X ∩ A = X ∪ A) : X = A := by
  sorry

end find_X_eq_A_l119_119844


namespace cone_volume_l119_119782

theorem cone_volume (d : ℝ) (h : ℝ) (π : ℝ) (volume : ℝ) 
  (hd : d = 10) (hh : h = 0.8 * d) (hπ : π = Real.pi) : 
  volume = (200 / 3) * π :=
by
  sorry

end cone_volume_l119_119782


namespace olly_needs_24_shoes_l119_119560

-- Define the number of paws for different types of pets
def dogs : ℕ := 3
def cats : ℕ := 2
def ferret : ℕ := 1

def paws_per_dog : ℕ := 4
def paws_per_cat : ℕ := 4
def paws_per_ferret : ℕ := 4

-- The theorem we want to prove
theorem olly_needs_24_shoes : 
  dogs * paws_per_dog + cats * paws_per_cat + ferret * paws_per_ferret = 24 :=
by
  sorry

end olly_needs_24_shoes_l119_119560


namespace point_not_in_image_of_plane_l119_119427

def satisfies_plane (P : ℝ × ℝ × ℝ) (A B C D : ℝ) : Prop :=
  let (x, y, z) := P
  A * x + B * y + C * z + D = 0

theorem point_not_in_image_of_plane :
  let A := (2, -3, 1)
  let aA := 1
  let aB := 1
  let aC := -2
  let aD := 2
  let k := 5 / 2
  let a'A := aA
  let a'B := aB
  let a'C := aC
  let a'D := k * aD
  ¬ satisfies_plane A a'A a'B a'C a'D :=
by
  -- TODO: Proof needed
  sorry

end point_not_in_image_of_plane_l119_119427


namespace roller_coaster_ticket_cost_l119_119177

def ferrisWheelCost : ℕ := 6
def logRideCost : ℕ := 7
def initialTickets : ℕ := 2
def ticketsToBuy : ℕ := 16

def totalTicketsNeeded : ℕ := initialTickets + ticketsToBuy
def ridesCost : ℕ := ferrisWheelCost + logRideCost
def rollerCoasterCost : ℕ := totalTicketsNeeded - ridesCost

theorem roller_coaster_ticket_cost :
  rollerCoasterCost = 5 :=
by
  sorry

end roller_coaster_ticket_cost_l119_119177


namespace coefficient_of_x_in_first_term_l119_119524

variable {a k n : ℝ} (x : ℝ)

theorem coefficient_of_x_in_first_term (h1 : (3 * x + 2) * (2 * x - 3) = a * x^2 + k * x + n) 
  (h2 : a - n + k = 7) :
  3 = 3 := 
sorry

end coefficient_of_x_in_first_term_l119_119524


namespace second_hand_angle_after_2_minutes_l119_119742

theorem second_hand_angle_after_2_minutes :
  ∀ angle_in_radians, (∀ rotations:ℝ, rotations = 2 → one_full_circle = 2 * Real.pi → angle_in_radians = - (rotations * one_full_circle)) →
  angle_in_radians = -4 * Real.pi :=
by
  intros
  sorry

end second_hand_angle_after_2_minutes_l119_119742


namespace students_average_age_l119_119240

theorem students_average_age (A : ℝ) (students_count teacher_age total_average new_count : ℝ) 
  (h1 : students_count = 30)
  (h2 : teacher_age = 45)
  (h3 : new_count = students_count + 1)
  (h4 : total_average = 15) 
  (h5 : total_average = (A * students_count + teacher_age) / new_count) : 
  A = 14 :=
by
  sorry

end students_average_age_l119_119240


namespace train_passes_bridge_in_52_seconds_l119_119589

def length_of_train : ℕ := 510
def speed_of_train_kmh : ℕ := 45
def length_of_bridge : ℕ := 140
def total_distance := length_of_train + length_of_bridge
def speed_of_train_ms := speed_of_train_kmh * 1000 / 3600
def time_to_pass_bridge := total_distance / speed_of_train_ms

theorem train_passes_bridge_in_52_seconds :
  time_to_pass_bridge = 52 := sorry

end train_passes_bridge_in_52_seconds_l119_119589


namespace find_g_values_l119_119803

open Function

-- Defining the function g and its properties
axiom g : ℝ → ℝ
axiom g_domain : ∀ x, 0 ≤ x → 0 ≤ g x
axiom g_proper : ∀ x, 0 ≤ x → 0 ≤ g (g x)
axiom g_func : ∀ x, 0 ≤ x → g (g x) = 3 * x / (x + 3)
axiom g_interval : ∀ x, 2 ≤ x ∧ x ≤ 3 → g x = (x + 1) / 2

-- Problem statement translating to Lean
theorem find_g_values :
  g 2021 = 2021.5 ∧ g (1 / 2021) = 6 := by {
  sorry 
}

end find_g_values_l119_119803


namespace solve_cyclic_quadrilateral_area_l119_119438

noncomputable def cyclic_quadrilateral_area (AB BC AD CD : ℝ) (cyclic : Bool) : ℝ :=
  if cyclic ∧ AB = 2 ∧ BC = 6 ∧ AD = 4 ∧ CD = 4 then 8 * Real.sqrt 3 else 0

theorem solve_cyclic_quadrilateral_area :
  cyclic_quadrilateral_area 2 6 4 4 true = 8 * Real.sqrt 3 :=
by
  sorry

end solve_cyclic_quadrilateral_area_l119_119438


namespace sum_bases_exponents_max_product_l119_119355

theorem sum_bases_exponents_max_product (A : ℕ) (hA : A = 3 ^ 670 * 2 ^ 2) : 
    (3 + 2 + 670 + 2 = 677) := by
  sorry

end sum_bases_exponents_max_product_l119_119355


namespace find_point_on_parabola_l119_119928

open Real

theorem find_point_on_parabola :
  ∃ (x y : ℝ), 
  (0 ≤ x ∧ 0 ≤ y) ∧
  (x^2 = 8 * y) ∧
  sqrt (x^2 + (y - 2)^2) = 120 ∧
  (x = 2 * sqrt 236 ∧ y = 118) :=
by
  sorry

end find_point_on_parabola_l119_119928


namespace find_a_l119_119626

-- Define the function f given a parameter a
def f (x a : ℝ) : ℝ := x^3 - 3*x^2 + a

-- Condition: f(x+1) is an odd function
theorem find_a (a : ℝ) (h : ∀ x : ℝ, f (-(x+1)) a = -f (x+1) a) : a = 2 := 
sorry

end find_a_l119_119626


namespace arc_length_of_given_curve_l119_119498

open Real

noncomputable def arc_length (f : ℝ → ℝ) (a b : ℝ) :=
  ∫ x in a..b, sqrt (1 + (deriv f x)^2)

noncomputable def given_function (x : ℝ) : ℝ :=
  arccos (sqrt x) - sqrt (x - x^2) + 4

theorem arc_length_of_given_curve :
  arc_length given_function 0 (1/2) = sqrt 2 :=
by
  sorry

end arc_length_of_given_curve_l119_119498


namespace prob_white_first_yellow_second_l119_119753

-- Defining the number of yellow and white balls
def yellow_balls : ℕ := 6
def white_balls : ℕ := 4

-- Defining the total number of balls
def total_balls : ℕ := yellow_balls + white_balls

-- Define the events A and B
def event_A : Prop := true -- event A: drawing a white ball first
def event_B : Prop := true -- event B: drawing a yellow ball second

-- Conditional probability P(B|A)
def prob_B_given_A : ℚ := 6 / (total_balls - 1)

-- Main theorem stating the proof problem
theorem prob_white_first_yellow_second : prob_B_given_A = 2 / 3 :=
by
  sorry

end prob_white_first_yellow_second_l119_119753


namespace koala_fiber_absorption_l119_119415

theorem koala_fiber_absorption (x : ℝ) (h1 : 0 < x) (h2 : x * 0.30 = 15) : x = 50 :=
sorry

end koala_fiber_absorption_l119_119415


namespace appropriate_survey_method_l119_119482

def survey_method_suitability (method : String) (context : String) : Prop :=
  match context, method with
  | "daily floating population of our city", "sampling survey" => true
  | "security checks before passengers board an airplane", "comprehensive survey" => true
  | "killing radius of a batch of shells", "sampling survey" => true
  | "math scores of Class 1 in Grade 7 of a certain school", "census method" => true
  | _, _ => false

theorem appropriate_survey_method :
  survey_method_suitability "census method" "daily floating population of our city" = false ∧
  survey_method_suitability "comprehensive survey" "security checks before passengers board an airplane" = false ∧
  survey_method_suitability "sampling survey" "killing radius of a batch of shells" = false ∧
  survey_method_suitability "census method" "math scores of Class 1 in Grade 7 of a certain school" = true :=
by
  sorry

end appropriate_survey_method_l119_119482


namespace consecutive_sum_ways_l119_119947

theorem consecutive_sum_ways (S : ℕ) (hS : S = 385) :
  ∃! n : ℕ, ∃! k : ℕ, n ≥ 2 ∧ S = n * (2 * k + n - 1) / 2 :=
sorry

end consecutive_sum_ways_l119_119947


namespace shooting_guard_seconds_l119_119862

-- Define the given conditions
def x_pg := 130
def x_sf := 85
def x_pf := 60
def x_c := 180
def avg_time_per_player := 120
def total_players := 5

-- Define the total footage
def total_footage : Nat := total_players * avg_time_per_player

-- Define the footage for four players
def footage_of_four : Nat := x_pg + x_sf + x_pf + x_c

-- Define the footage of the shooting guard, which is a variable we want to compute
def x_sg := total_footage - footage_of_four

-- The statement we want to prove
theorem shooting_guard_seconds :
  x_sg = 145 := by
  sorry

end shooting_guard_seconds_l119_119862


namespace distance_A_focus_l119_119372

-- Definitions from the problem conditions
def parabola_eq (x y : ℝ) : Prop := x^2 = 4 * y
def point_A (x : ℝ) : Prop := parabola_eq x 4
def focus_y_coord : ℝ := 1 -- Derived from the standard form of the parabola x^2 = 4py where p=1

-- State the theorem in Lean 4
theorem distance_A_focus (x : ℝ) (hA : point_A x) : |4 - focus_y_coord| = 3 :=
by
  -- Proof would go here
  sorry

end distance_A_focus_l119_119372


namespace score_comparison_l119_119210

theorem score_comparison :
  let sammy_score := 20
  let gab_score := 2 * sammy_score
  let cher_score := 2 * gab_score
  let alex_score := cher_score + cher_score / 10
  let combined_score := sammy_score + gab_score + cher_score + alex_score
  let opponent_score := 85
  combined_score - opponent_score = 143 :=
by
  let sammy_score := 20
  let gab_score := 2 * sammy_score
  let cher_score := 2 * gab_score
  let alex_score := cher_score + cher_score / 10
  let combined_score := sammy_score + gab_score + cher_score + alex_score
  let opponent_score := 85
  sorry

end score_comparison_l119_119210


namespace part_a_part_b_l119_119322

-- Definition of the function f and the condition it satisfies
variable (f : ℕ → ℕ)
variable (k n : ℕ)

theorem part_a (h1 : ∀ k n : ℕ, (k * f n) ≤ f (k * n) ∧ f (k * n) ≤ (k * f n) + k - 1)
  (a b : ℕ) :
  f a + f b ≤ f (a + b) ∧ f (a + b) ≤ f a + f b + 1 :=
by
  exact sorry  -- Proof to be supplied

theorem part_b (h1 : ∀ k n : ℕ, (k * f n) ≤ f (k * n) ∧ f (k * n) ≤ (k * f n) + k - 1)
  (h2 : ∀ n : ℕ, f (2007 * n) ≤ 2007 * f n + 200) :
  ∃ c : ℕ, f (2007 * c) = 2007 * f c :=
by
  exact sorry  -- Proof to be supplied

end part_a_part_b_l119_119322


namespace boxes_needed_l119_119880

theorem boxes_needed (total_muffins : ℕ) (muffins_per_box : ℕ) (available_boxes : ℕ) (h1 : total_muffins = 95) (h2 : muffins_per_box = 5) (h3 : available_boxes = 10) : 
  total_muffins - (available_boxes * muffins_per_box) / muffins_per_box = 9 :=
by
  sorry

end boxes_needed_l119_119880


namespace original_selling_price_l119_119077

theorem original_selling_price (P : ℝ) (h : 0.7 * P = 560) : P = 800 :=
by
  sorry

end original_selling_price_l119_119077


namespace sin_double_angle_l119_119766

theorem sin_double_angle (α : ℝ) (h : Real.sin (α + Real.pi / 4) = 1 / 2) : Real.sin (2 * α) = -1 / 2 :=
sorry

end sin_double_angle_l119_119766


namespace range_of_a_l119_119004

noncomputable def f (a x : ℝ) := a * x - x^2 - Real.log x

theorem range_of_a (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ 2*x₁*x₁ - a*x₁ + 1 = 0 ∧ 
  2*x₂*x₂ - a*x₂ + 1 = 0 ∧ f a x₁ + f a x₂ ≥ 4 + Real.log 2) ↔ 
  a ∈ Set.Ici (2 * Real.sqrt 3) := 
sorry

end range_of_a_l119_119004


namespace sin_eq_sin_sinx_l119_119166

noncomputable def S (x : ℝ) := Real.sin x - x

theorem sin_eq_sin_sinx (x : ℝ) (h : 0 ≤ x ∧ x ≤ Real.arcsin 742) :
  ∃! x, Real.sin x = Real.sin (Real.sin x) :=
by
  sorry

end sin_eq_sin_sinx_l119_119166


namespace sum_of_roots_l119_119036

theorem sum_of_roots (a b c d : ℝ) (h : ∀ x : ℝ, 
  a * (x ^ 3 - x) ^ 3 + b * (x ^ 3 - x) ^ 2 + c * (x ^ 3 - x) + d 
  ≥ a * (x ^ 2 + x + 1) ^ 3 + b * (x ^ 2 + x + 1) ^ 2 + c * (x ^ 2 + x + 1) + d) :
  b / a = -6 :=
sorry

end sum_of_roots_l119_119036


namespace janet_freelancer_income_difference_l119_119278

theorem janet_freelancer_income_difference :
  let hours_per_week := 40
  let current_job_hourly_rate := 30
  let freelancer_hourly_rate := 40
  let fica_taxes_per_week := 25
  let healthcare_premiums_per_month := 400
  let weeks_per_month := 4
  
  let current_job_weekly_income := hours_per_week * current_job_hourly_rate
  let current_job_monthly_income := current_job_weekly_income * weeks_per_month
  
  let freelancer_weekly_income := hours_per_week * freelancer_hourly_rate
  let freelancer_monthly_income := freelancer_weekly_income * weeks_per_month
  
  let freelancer_monthly_fica_taxes := fica_taxes_per_week * weeks_per_month
  let freelancer_total_additional_costs := freelancer_monthly_fica_taxes + healthcare_premiums_per_month
  
  let freelancer_net_monthly_income := freelancer_monthly_income - freelancer_total_additional_costs
  
  freelancer_net_monthly_income - current_job_monthly_income = 1100 :=
by
  sorry

end janet_freelancer_income_difference_l119_119278


namespace power_function_point_l119_119992

theorem power_function_point (m n: ℝ) (h: (m - 1) * m^n = 8) : n^(-m) = 1/9 := 
  sorry

end power_function_point_l119_119992


namespace mirror_tweet_rate_is_45_l119_119559

-- Defining the conditions given in the problem
def happy_tweet_rate : ℕ := 18
def hungry_tweet_rate : ℕ := 4
def mirror_tweet_rate (x : ℕ) : ℕ := x
def happy_minutes : ℕ := 20
def hungry_minutes : ℕ := 20
def mirror_minutes : ℕ := 20
def total_tweets : ℕ := 1340

-- Proving the rate of tweets when Polly watches herself in the mirror
theorem mirror_tweet_rate_is_45 : mirror_tweet_rate 45 * mirror_minutes = total_tweets - (happy_tweet_rate * happy_minutes + hungry_tweet_rate * hungry_minutes) :=
by 
  sorry

end mirror_tweet_rate_is_45_l119_119559


namespace length_of_train_l119_119022

-- Definitions for the given conditions:
def speed : ℝ := 60   -- in kmph
def time : ℝ := 20    -- in seconds
def platform_length : ℝ := 213.36  -- in meters

-- Conversion factor from km/h to m/s
noncomputable def kmph_to_mps (kmph : ℝ) : ℝ := kmph * (1000 / 3600)

-- Total distance covered by train while crossing the platform
noncomputable def total_distance (speed_in_kmph : ℝ) (time_in_seconds : ℝ) : ℝ := 
  (kmph_to_mps speed_in_kmph) * time_in_seconds

-- Length of the train
noncomputable def train_length (total_distance_covered : ℝ) (platform_len : ℝ) : ℝ :=
  total_distance_covered - platform_len

-- Expected length of the train
def expected_train_length : ℝ := 120.04

-- Theorem to prove the length of the train given the conditions
theorem length_of_train : 
  train_length (total_distance speed time) platform_length = expected_train_length :=
by 
  sorry

end length_of_train_l119_119022


namespace a_plus_b_plus_c_at_2_l119_119897

noncomputable def quadratic (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

def maximum_value (a b c : ℝ) : Prop :=
  ∃ x : ℝ, quadratic a b c x = 75

def passes_through (a b c : ℝ) (p1 p2 : ℝ × ℝ) : Prop :=
  quadratic a b c p1.1 = p1.2 ∧ quadratic a b c p2.1 = p2.2

theorem a_plus_b_plus_c_at_2 
  (a b c : ℝ)
  (hmax : maximum_value a b c)
  (hpoints : passes_through a b c (-3, 0) (3, 0))
  (hvertex : ∀ x : ℝ, quadratic a 0 c x ≤ quadratic a (2 * b) c 0) : 
  quadratic a b c 2 = 125 / 3 :=
sorry

end a_plus_b_plus_c_at_2_l119_119897


namespace abs_neg_2023_l119_119135

theorem abs_neg_2023 : |(-2023 : ℤ)| = 2023 :=
by
  sorry

end abs_neg_2023_l119_119135


namespace subset_P_Q_l119_119723

def P := {x : ℝ | x > 1}
def Q := {x : ℝ | x^2 - x > 0}

theorem subset_P_Q : P ⊆ Q :=
by
  sorry

end subset_P_Q_l119_119723


namespace remainder_of_sum_division_l119_119424

def a1 : ℕ := 2101
def a2 : ℕ := 2103
def a3 : ℕ := 2105
def a4 : ℕ := 2107
def a5 : ℕ := 2109
def n : ℕ := 12

theorem remainder_of_sum_division : ((a1 + a2 + a3 + a4 + a5) % n) = 1 :=
by {
  sorry
}

end remainder_of_sum_division_l119_119424


namespace sqrt_factorial_mul_squared_l119_119137

theorem sqrt_factorial_mul_squared :
  (Nat.sqrt (Nat.factorial 5 * Nat.factorial 4)) ^ 2 = 2880 :=
by
  sorry

end sqrt_factorial_mul_squared_l119_119137


namespace pos_real_x_plus_inv_ge_two_l119_119470

theorem pos_real_x_plus_inv_ge_two (x : ℝ) (hx : x > 0) : x + (1 / x) ≥ 2 :=
by
  sorry

end pos_real_x_plus_inv_ge_two_l119_119470


namespace initial_amount_of_money_l119_119072

-- Define the costs and purchased quantities
def cost_tshirt : ℕ := 8
def cost_keychain_set : ℕ := 2
def cost_bag : ℕ := 10
def tshirts_bought : ℕ := 2
def bags_bought : ℕ := 2
def keychains_bought : ℕ := 21

-- Define derived quantities
def sets_of_keychains_bought : ℕ := keychains_bought / 3

-- Define the total costs
def total_cost_tshirts : ℕ := tshirts_bought * cost_tshirt
def total_cost_bags : ℕ := bags_bought * cost_bag
def total_cost_keychains : ℕ := sets_of_keychains_bought * cost_keychain_set

-- Define the initial amount of money
def total_initial_amount : ℕ := total_cost_tshirts + total_cost_bags + total_cost_keychains

-- The theorem proving the initial amount Timothy had
theorem initial_amount_of_money : total_initial_amount = 50 := by
  -- The proof is not required, so we use sorry to skip it
  sorry

end initial_amount_of_money_l119_119072


namespace people_dislike_both_radio_and_music_l119_119043

theorem people_dislike_both_radio_and_music (N : ℕ) (p_r p_rm : ℝ) (hN : N = 2000) (hp_r : p_r = 0.25) (hp_rm : p_rm = 0.15) : 
  N * p_r * p_rm = 75 :=
by {
  sorry
}

end people_dislike_both_radio_and_music_l119_119043


namespace number_of_flower_sets_l119_119644

theorem number_of_flower_sets (total_flowers : ℕ) (flowers_per_set : ℕ) (sets : ℕ) 
  (h1 : total_flowers = 270) 
  (h2 : flowers_per_set = 90) 
  (h3 : sets = total_flowers / flowers_per_set) : 
  sets = 3 := 
by 
  sorry

end number_of_flower_sets_l119_119644


namespace find_f_2021_l119_119395

def f (x : ℝ) : ℝ := sorry

theorem find_f_2021 (h : ∀ a b : ℝ, f ((a + 2 * b) / 3) = (f a + 2 * f b) / 3)
    (h1 : f 1 = 5) (h4 : f 4 = 2) : f 2021 = -2015 :=
by
  sorry

end find_f_2021_l119_119395


namespace smallest_x_of_quadratic_eqn_l119_119747

theorem smallest_x_of_quadratic_eqn : ∃ x : ℝ, (12*x^2 - 44*x + 40 = 0) ∧ x = 5 / 3 :=
by
  sorry

end smallest_x_of_quadratic_eqn_l119_119747


namespace min_cost_correct_l119_119648

noncomputable def min_cost_to_feed_group : ℕ :=
  let main_courses := 50
  let salads := 30
  let soups := 15
  let price_salad := 200
  let price_soup_main := 350
  let price_salad_main := 350
  let price_all_three := 500
  17000

theorem min_cost_correct : min_cost_to_feed_group = 17000 :=
by
  sorry

end min_cost_correct_l119_119648


namespace sum_lent_is_1000_l119_119565

theorem sum_lent_is_1000
    (P : ℝ)
    (r : ℝ)
    (t : ℝ)
    (I : ℝ)
    (h1 : r = 5)
    (h2 : t = 5)
    (h3 : I = P - 750)
    (h4 : I = P * r * t / 100) :
  P = 1000 :=
by sorry

end sum_lent_is_1000_l119_119565


namespace haley_total_lives_l119_119876

-- Define initial conditions
def initial_lives : ℕ := 14
def lives_lost : ℕ := 4
def lives_gained : ℕ := 36

-- Definition to calculate total lives
def total_lives (initial_lives lives_lost lives_gained : ℕ) : ℕ :=
  initial_lives - lives_lost + lives_gained

-- The theorem statement we want to prove
theorem haley_total_lives : total_lives initial_lives lives_lost lives_gained = 46 :=
by 
  sorry

end haley_total_lives_l119_119876


namespace multiplicative_inverse_sum_is_zero_l119_119933

theorem multiplicative_inverse_sum_is_zero (a b : ℝ) (h : a * b = 1) :
  a^(2015) * b^(2016) + a^(2016) * b^(2017) + a^(2017) * b^(2016) + a^(2016) * b^(2015) = 0 :=
sorry

end multiplicative_inverse_sum_is_zero_l119_119933


namespace xy_value_l119_119557

theorem xy_value (x y : ℝ) (h1 : x + y = 10) (h2 : x^3 + y^3 = 370) : xy = 21 :=
sorry

end xy_value_l119_119557


namespace average_percentage_taller_l119_119615

theorem average_percentage_taller 
  (h1 b1 h2 b2 h3 b3 : ℝ)
  (h1_eq : h1 = 228) (b1_eq : b1 = 200)
  (h2_eq : h2 = 120) (b2_eq : b2 = 100)
  (h3_eq : h3 = 147) (b3_eq : b3 = 140) :
  ((h1 - b1) / b1 * 100 + (h2 - b2) / b2 * 100 + (h3 - b3) / b3 * 100) / 3 = 13 := by
  rw [h1_eq, b1_eq, h2_eq, b2_eq, h3_eq, b3_eq]
  sorry

end average_percentage_taller_l119_119615


namespace max_XG_l119_119303

theorem max_XG :
  ∀ (G X Y Z : ℝ),
    Y - X = 5 ∧ Z - Y = 3 ∧ (1 / G + 1 / (G - 5) + 1 / (G - 8) = 0) →
    G = 20 / 3 :=
by
  sorry

end max_XG_l119_119303


namespace augmented_matrix_solution_l119_119600

theorem augmented_matrix_solution (c₁ c₂ : ℝ) (x y : ℝ) 
  (h1 : 2 * x + 3 * y = c₁) (h2 : 3 * x + 2 * y = c₂)
  (hx : x = 2) (hy : y = 1) : c₁ - c₂ = -1 := 
by
  sorry

end augmented_matrix_solution_l119_119600


namespace max_minute_hands_l119_119099

theorem max_minute_hands (m n : ℕ) (h1 : m * n = 27) : m + n ≤ 28 :=
by sorry

end max_minute_hands_l119_119099


namespace midpoint_of_complex_numbers_l119_119940

theorem midpoint_of_complex_numbers :
  let A := (1 - 1*I) / (1 + 1)
  let B := (1 + 1*I) / (1 + 1)
  (A + B) / 2 = 1 / 2 := by
sorry

end midpoint_of_complex_numbers_l119_119940


namespace total_seeds_l119_119778

theorem total_seeds (seeds_per_watermelon : ℕ) (number_of_watermelons : ℕ) 
(seeds_each : seeds_per_watermelon = 100)
(watermelons_count : number_of_watermelons = 4) :
(seeds_per_watermelon * number_of_watermelons) = 400 := by
  sorry

end total_seeds_l119_119778


namespace amy_race_time_l119_119344

theorem amy_race_time (patrick_time : ℕ) (manu_time : ℕ) (amy_time : ℕ)
  (h1 : patrick_time = 60)
  (h2 : manu_time = patrick_time + 12)
  (h3 : amy_time = manu_time / 2) : 
  amy_time = 36 := 
sorry

end amy_race_time_l119_119344


namespace solve_for_x_l119_119412

theorem solve_for_x (x : ℝ) : 
  x^2 - 2 * x - 8 = -(x + 2) * (x - 6) → (x = 5 ∨ x = -2) :=
by
  intro h
  sorry

end solve_for_x_l119_119412


namespace solution_set_ineq_l119_119633

theorem solution_set_ineq (x : ℝ) :
  x * (2 * x^2 - 3 * x + 1) ≤ 0 ↔ (x ≤ 0 ∨ (1/2 ≤ x ∧ x ≤ 1)) :=
sorry

end solution_set_ineq_l119_119633


namespace daily_evaporation_l119_119741

theorem daily_evaporation :
  ∀ (initial_amount : ℝ) (percentage_evaporated : ℝ) (days : ℕ),
  initial_amount = 10 →
  percentage_evaporated = 6 →
  days = 50 →
  (initial_amount * (percentage_evaporated / 100)) / days = 0.012 :=
by
  intros initial_amount percentage_evaporated days
  intros h_initial h_percentage h_days
  rw [h_initial, h_percentage, h_days]
  sorry

end daily_evaporation_l119_119741


namespace admission_price_for_children_l119_119547

theorem admission_price_for_children 
  (admission_price_adult : ℕ)
  (total_persons : ℕ)
  (total_amount_dollars : ℕ)
  (children_attended : ℕ)
  (admission_price_children : ℕ)
  (h1 : admission_price_adult = 60)
  (h2 : total_persons = 280)
  (h3 : total_amount_dollars = 140)
  (h4 : children_attended = 80)
  (h5 : (total_persons - children_attended) * admission_price_adult + children_attended * admission_price_children = total_amount_dollars * 100)
  : admission_price_children = 25 := 
by 
  sorry

end admission_price_for_children_l119_119547


namespace solution_set_inequality_l119_119875

open Set

theorem solution_set_inequality :
  {x : ℝ | (x+1)/(x-4) ≥ 3} = Iio 4 ∪ Ioo 4 (13/2) ∪ {13/2} :=
by
  sorry

end solution_set_inequality_l119_119875


namespace ratio_of_customers_third_week_l119_119026

def ratio_of_customers (c1 c3 : ℕ) (s k t : ℕ) : Prop := s = 500 ∧ k = 50 ∧ t = 760 ∧ c1 = 35 ∧ c3 = 105 ∧ (t - s - k) - (35 + 70) = c1 ∧ c3 = 105 ∧ (c3 / c1 = 3)

theorem ratio_of_customers_third_week (c1 c3 : ℕ) (s k t : ℕ)
  (h1 : s = 500)
  (h2 : k = 50)
  (h3 : t = 760)
  (h4 : c1 = 35)
  (h5 : c3 = 105)
  (h6 : (t - s - k) - (35 + 70) = c1)
  (h7 : c3 = 105) :
  (c3 / c1) = 3 :=
  sorry

end ratio_of_customers_third_week_l119_119026


namespace intersection_is_1_l119_119542

def M : Set ℤ := {-1, 1, 2}
def N : Set ℤ := {y | ∃ x ∈ M, y = x ^ 2}
theorem intersection_is_1 : M ∩ N = {1} := by
  sorry

end intersection_is_1_l119_119542


namespace initial_scooter_value_l119_119364

theorem initial_scooter_value (V : ℝ) (h : V * (3/4)^2 = 22500) : V = 40000 :=
by
  sorry

end initial_scooter_value_l119_119364


namespace fraction_is_one_third_l119_119012

theorem fraction_is_one_third :
  (3 + 9 - 27 + 81 + 243 - 729) / (9 + 27 - 81 + 243 + 729 - 2187) = 1 / 3 :=
by
  sorry

end fraction_is_one_third_l119_119012


namespace largest_whole_number_l119_119985

theorem largest_whole_number (x : ℤ) : 9 * x < 200 → x ≤ 22 := by
  sorry

end largest_whole_number_l119_119985


namespace number_of_pairs_l119_119478

theorem number_of_pairs (f m : ℕ) (n : ℕ) :
  n = 6 →
  (f + m ≤ n) →
  ∃! pairs : ℕ, pairs = 2 :=
by
  intro h1 h2
  sorry

end number_of_pairs_l119_119478


namespace count_prime_sum_112_l119_119693

noncomputable def primeSum (primes : List ℕ) : ℕ :=
  if H : ∀ p ∈ primes, Nat.Prime p ∧ p > 10 then primes.sum else 0

theorem count_prime_sum_112 :
  ∃ (primes : List ℕ), primeSum primes = 112 ∧ primes.length = 6 := by
  sorry

end count_prime_sum_112_l119_119693


namespace series_converges_to_l119_119373

noncomputable def series_sum := ∑' n : Nat, (4 * n + 3) / ((4 * n + 1) ^ 2 * (4 * n + 5) ^ 2)

theorem series_converges_to : series_sum = 1 / 200 := 
by 
  sorry

end series_converges_to_l119_119373


namespace minnie_penny_time_difference_l119_119981

noncomputable def minnie_time_uphill (distance speed: ℝ) := distance / speed
noncomputable def minnie_time_downhill (distance speed: ℝ) := distance / speed
noncomputable def minnie_time_flat (distance speed: ℝ) := distance / speed
noncomputable def penny_time_flat (distance speed: ℝ) := distance / speed
noncomputable def penny_time_downhill (distance speed: ℝ) := distance / speed
noncomputable def penny_time_uphill (distance speed: ℝ) := distance / speed
noncomputable def break_time (minutes: ℝ) := minutes / 60

noncomputable def minnie_total_time :=
  minnie_time_uphill 12 6 + minnie_time_downhill 18 25 + minnie_time_flat 25 18

noncomputable def penny_total_time :=
  penny_time_flat 25 25 + penny_time_downhill 12 35 + 
  penny_time_uphill 18 12 + break_time 10

noncomputable def time_difference := (minnie_total_time - penny_total_time) * 60

theorem minnie_penny_time_difference :
  time_difference = 66 := by
  sorry

end minnie_penny_time_difference_l119_119981


namespace discount_problem_l119_119916

variable (x : ℝ)

theorem discount_problem :
  (400 * (1 - x)^2 = 225) :=
sorry

end discount_problem_l119_119916


namespace coconut_tree_difference_l119_119167

-- Define the known quantities
def mango_trees : ℕ := 60
def total_trees : ℕ := 85
def half_mango_trees : ℕ := 30 -- half of 60
def coconut_trees : ℕ := 25 -- 85 - 60

-- Define the proof statement
theorem coconut_tree_difference : (half_mango_trees - coconut_trees) = 5 := by
  -- The proof steps are given
  sorry

end coconut_tree_difference_l119_119167


namespace expression_value_l119_119706

theorem expression_value (a b c d : ℝ) (h1 : a * b = 1) (h2 : c + d = 0) :
  -((a * b) ^ (1/3)) + (c + d).sqrt + 1 = 0 :=
by sorry

end expression_value_l119_119706


namespace nickels_left_l119_119662

theorem nickels_left (n b : ℕ) (h₁ : n = 31) (h₂ : b = 20) : n - b = 11 :=
by
  sorry

end nickels_left_l119_119662


namespace algebraic_expression_value_l119_119834

theorem algebraic_expression_value (x y : ℝ) (h : 2 * x - y = 2) : 6 * x - 3 * y + 1 = 7 := 
by
  sorry

end algebraic_expression_value_l119_119834


namespace gcd_of_36_and_60_is_12_l119_119843

theorem gcd_of_36_and_60_is_12 :
  Nat.gcd 36 60 = 12 :=
sorry

end gcd_of_36_and_60_is_12_l119_119843


namespace trigonometric_identity_l119_119950

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 2) :
  7 * (Real.sin α)^2 + 3 * (Real.cos α)^2 = 31 / 5 := by
  sorry

end trigonometric_identity_l119_119950


namespace quadratic_has_one_solution_l119_119679

theorem quadratic_has_one_solution (m : ℝ) : 3 * (49 / 12) - 7 * (49 / 12) + m = 0 → m = 49 / 12 :=
by
  sorry

end quadratic_has_one_solution_l119_119679


namespace proof_problem_l119_119795

def M : Set ℝ := { x | x > -1 }

theorem proof_problem : {0} ⊆ M := by
  sorry

end proof_problem_l119_119795


namespace no_three_digit_number_l119_119187

theorem no_three_digit_number :
  ¬ ∃ (a b c : ℕ), (1 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) ∧ (0 ≤ c ∧ c ≤ 9) ∧ (100 * a + 10 * b + c = 3 * (100 * b + 10 * c + a)) :=
by
  sorry

end no_three_digit_number_l119_119187


namespace intersection_of_A_and_B_l119_119132

def setA : Set ℝ := { x | x - 2 ≥ 0 }
def setB : Set ℝ := { x | 0 < Real.log x / Real.log 2 ∧ Real.log x / Real.log 2 < 2 }

theorem intersection_of_A_and_B :
  setA ∩ setB = { x | 2 ≤ x ∧ x < 4 } :=
sorry

end intersection_of_A_and_B_l119_119132


namespace greatest_int_less_than_200_with_gcd_18_eq_9_l119_119508

theorem greatest_int_less_than_200_with_gcd_18_eq_9 :
  ∃ n, n < 200 ∧ Int.gcd n 18 = 9 ∧ ∀ m, m < 200 ∧ Int.gcd m 18 = 9 → m ≤ n :=
sorry

end greatest_int_less_than_200_with_gcd_18_eq_9_l119_119508


namespace edward_chocolate_l119_119462

theorem edward_chocolate (total_chocolate : ℚ) (num_piles : ℕ) (piles_received_by_Edward : ℕ) :
  total_chocolate = 75 / 7 → num_piles = 5 → piles_received_by_Edward = 2 → 
  (total_chocolate / num_piles) * piles_received_by_Edward = 30 / 7 := 
by
  intros ht hn hp
  sorry

end edward_chocolate_l119_119462


namespace find_y_l119_119757

-- Given conditions
def x : Int := 129
def student_operation (y : Int) : Int := x * y - 148
def result : Int := 110

-- The theorem statement
theorem find_y :
  ∃ y : Int, student_operation y = result ∧ y = 2 := 
sorry

end find_y_l119_119757


namespace min_max_of_quadratic_l119_119987

theorem min_max_of_quadratic 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = 2 * x^2 - 6 * x + 1)
  (h2 : ∀ x, -1 ≤ x ∧ x ≤ 1) : 
  (∃ xmin, ∃ xmax, f xmin = -3 ∧ f xmax = 9 ∧ -1 ≤ xmin ∧ xmin ≤ 1 ∧ -1 ≤ xmax ∧ xmax ≤ 1) :=
sorry

end min_max_of_quadratic_l119_119987


namespace chord_length_intercepted_by_line_on_curve_l119_119926

-- Define the curve and line from the problem
def curve (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 6*y + 1 = 0
def line (x y : ℝ) : Prop := 2*x + y = 0

-- Prove the length of the chord intercepted by the line on the curve is 4
theorem chord_length_intercepted_by_line_on_curve : 
  ∀ (x y : ℝ), curve x y → line x y → False := sorry

end chord_length_intercepted_by_line_on_curve_l119_119926


namespace fraction_of_number_l119_119442

theorem fraction_of_number (N : ℕ) (hN : N = 180) : 
  (6 + (1 / 2) * (1 / 3) * (1 / 5) * N) = (1 / 25) * N := 
by
  sorry

end fraction_of_number_l119_119442


namespace pushups_total_l119_119282

theorem pushups_total (z d e : ℕ)
  (hz : z = 44) 
  (hd : d = z + 58) 
  (he : e = 2 * d) : 
  z + d + e = 350 := by
  sorry

end pushups_total_l119_119282


namespace negation_if_then_l119_119192

theorem negation_if_then (x : ℝ) : ¬ (x > 2 → x > 1) ↔ (x ≤ 2 → x ≤ 1) :=
by 
  sorry

end negation_if_then_l119_119192


namespace quadratic_c_over_b_l119_119299

theorem quadratic_c_over_b :
  ∃ (b c : ℤ), (x^2 + 500 * x + 1000 = (x + b)^2 + c) ∧ (c / b = -246) :=
by sorry

end quadratic_c_over_b_l119_119299


namespace solve_equation_real_l119_119309

theorem solve_equation_real (x : ℝ) (h : (x ^ 2 - x + 1) * (3 * x ^ 2 - 10 * x + 3) = 20 * x ^ 2) :
    x = (5 + Real.sqrt 21) / 2 ∨ x = (5 - Real.sqrt 21) / 2 :=
by
  sorry

end solve_equation_real_l119_119309


namespace total_packs_l119_119577

theorem total_packs (cards_bought : ℕ) (cards_per_pack : ℕ) (num_people : ℕ)
  (h1 : cards_bought = 540) (h2 : cards_per_pack = 20) (h3 : num_people = 4) :
  (cards_bought / cards_per_pack) * num_people = 108 :=
by
  sorry

end total_packs_l119_119577


namespace find_A_in_terms_of_B_and_C_l119_119113

theorem find_A_in_terms_of_B_and_C 
  (A B C : ℝ) (hB : B ≠ 0) 
  (f : ℝ → ℝ) (g : ℝ → ℝ) 
  (hf : ∀ x, f x = A * x - 2 * B^2)
  (hg : ∀ x, g x = B * x + C * x^2)
  (hfg : f (g 1) = 4 * B^2)
  : A = 6 * B * B / (B + C) :=
by
  sorry

end find_A_in_terms_of_B_and_C_l119_119113


namespace polynomial_coeff_sum_l119_119294

/-- 
Given that the product of the polynomials (4x^2 - 6x + 5)(8 - 3x) can be written as
ax^3 + bx^2 + cx + d, prove that 9a + 3b + c + d = 19.
-/
theorem polynomial_coeff_sum :
  ∃ a b c d : ℝ, 
  (∀ x : ℝ, (4 * x^2 - 6 * x + 5) * (8 - 3 * x) = a * x^3 + b * x^2 + c * x + d) ∧
  9 * a + 3 * b + c + d = 19 :=
sorry

end polynomial_coeff_sum_l119_119294


namespace math_problem_l119_119804

noncomputable def a : ℝ := (0.96)^3 
noncomputable def b : ℝ := (0.1)^3 
noncomputable def c : ℝ := (0.96)^2 
noncomputable def d : ℝ := (0.1)^2 

theorem math_problem : a - b / c + 0.096 + d = 0.989651 := 
by 
  -- skip proof 
  sorry

end math_problem_l119_119804


namespace intersection_A_B_l119_119548

open Set

def set_A : Set ℤ := {x | ∃ k : ℤ, x = 2 * k + 1}
def set_B : Set ℤ := {x | 0 < x ∧ x < 5}

theorem intersection_A_B : set_A ∩ set_B = {1, 3} := 
by 
  sorry

end intersection_A_B_l119_119548


namespace tug_of_war_matches_l119_119886

-- Define the number of classes
def num_classes : ℕ := 7

-- Define the number of matches Grade 3 Class 6 competes in
def matches_class6 : ℕ := num_classes - 1

-- Define the total number of matches
def total_matches : ℕ := (num_classes - 1) * num_classes / 2

-- Main theorem stating the problem
theorem tug_of_war_matches :
  matches_class6 = 6 ∧ total_matches = 21 := by
  sorry

end tug_of_war_matches_l119_119886


namespace final_expression_l119_119040

theorem final_expression (y : ℝ) : (3 * (1 / 2 * (12 * y + 3))) = 18 * y + 4.5 :=
by
  sorry

end final_expression_l119_119040


namespace flour_per_new_bread_roll_l119_119312

theorem flour_per_new_bread_roll (p1 f1 p2 f2 c : ℚ)
  (h1 : p1 = 40)
  (h2 : f1 = 1 / 8)
  (h3 : p2 = 25)
  (h4 : c = p1 * f1)
  (h5 : c = p2 * f2) :
  f2 = 1 / 5 :=
by
  sorry

end flour_per_new_bread_roll_l119_119312


namespace necessary_and_sufficient_condition_l119_119304

theorem necessary_and_sufficient_condition (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (|a + b| = |a| + |b|) ↔ (a * b > 0) :=
sorry

end necessary_and_sufficient_condition_l119_119304


namespace residue_n_mod_17_l119_119669

noncomputable def satisfies_conditions (m n k : ℕ) : Prop :=
  m^2 + 1 = 2 * n^2 ∧ 2 * m^2 + 1 = 11 * k^2 

theorem residue_n_mod_17 (m n k : ℕ) (h : satisfies_conditions m n k) : n % 17 = 5 :=
  sorry

end residue_n_mod_17_l119_119669


namespace count_negative_terms_in_sequence_l119_119235

theorem count_negative_terms_in_sequence : 
  ∃ (s : List ℕ), (∀ n ∈ s, n^2 - 8*n + 12 < 0) ∧ s.length = 3 ∧ (∀ n ∈ s, 2 < n ∧ n < 6) :=
by
  sorry

end count_negative_terms_in_sequence_l119_119235


namespace intersection_points_eq_one_l119_119029

-- Definitions for the equations of the circles
def circle1 (x y : ℝ) : ℝ := x^2 + (y - 3)^2
def circle2 (x y : ℝ) : ℝ := x^2 + (y + 2)^2

-- The proof problem statement
theorem intersection_points_eq_one : 
  ∃ p : ℝ × ℝ, (circle1 p.1 p.2 = 9) ∧ (circle2 p.1 p.2 = 4) ∧
  (∀ q : ℝ × ℝ, (circle1 q.1 q.2 = 9) ∧ (circle2 q.1 q.2 = 4) → q = p) :=
sorry

end intersection_points_eq_one_l119_119029


namespace tangent_line_to_circle_l119_119636

theorem tangent_line_to_circle : 
  ∀ (ρ θ : ℝ), (ρ = 4 * Real.sin θ) → (∃ ρ θ : ℝ, ρ * Real.cos θ = 2) :=
by
  sorry

end tangent_line_to_circle_l119_119636


namespace remaining_pencils_l119_119348

/-
Given the initial number of pencils in the drawer and the number of pencils Sally took out,
prove that the number of pencils remaining in the drawer is 5.
-/
def pencils_in_drawer (initial_pencils : ℕ) (pencils_taken : ℕ) : ℕ :=
  initial_pencils - pencils_taken

theorem remaining_pencils : pencils_in_drawer 9 4 = 5 := by
  sorry

end remaining_pencils_l119_119348


namespace smallest_integer_k_distinct_real_roots_l119_119093

theorem smallest_integer_k_distinct_real_roots :
  ∃ k : ℤ, (∀ x : ℝ, x^2 - x + 2 - k = 0 → x ≠ 0) ∧ k = 2 :=
by
  sorry

end smallest_integer_k_distinct_real_roots_l119_119093


namespace smallest_possible_value_of_N_l119_119248

-- Define the dimensions of the block
variables (l m n : ℕ) 

-- Define the condition that the product of dimensions minus one is 143
def hidden_cubes_count (l m n : ℕ) : Prop := (l - 1) * (m - 1) * (n - 1) = 143

-- Define the total number of cubes in the outer block
def total_cubes (l m n : ℕ) : ℕ := l * m * n

-- The final proof statement
theorem smallest_possible_value_of_N : 
  ∃ (l m n : ℕ), hidden_cubes_count l m n → N = total_cubes l m n → N = 336 :=
sorry

end smallest_possible_value_of_N_l119_119248


namespace non_negative_integers_abs_less_than_3_l119_119087

theorem non_negative_integers_abs_less_than_3 :
  { x : ℕ | x < 3 } = {0, 1, 2} :=
by
  sorry

end non_negative_integers_abs_less_than_3_l119_119087


namespace russian_writer_surname_l119_119410

def is_valid_surname (x y z w v u : ℕ) : Prop :=
  x = z ∧
  y = w ∧
  v = x + 9 ∧
  u = y + w - 2 ∧
  3 * x = y - 4 ∧
  x + y + z + w + v + u = 83

def position_to_letter (n : ℕ) : String :=
  if n = 4 then "Г"
  else if n = 16 then "О"
  else if n = 13 then "Л"
  else if n = 30 then "Ь"
  else "?"

theorem russian_writer_surname : ∃ x y z w v u : ℕ, 
  is_valid_surname x y z w v u ∧
  position_to_letter x ++ position_to_letter y ++ position_to_letter z ++ position_to_letter w ++ position_to_letter v ++ position_to_letter u = "Гоголь" :=
by
  sorry

end russian_writer_surname_l119_119410


namespace multiply_digits_correctness_l119_119533

theorem multiply_digits_correctness (a b c : ℕ) :
  (10 * a + b) * (10 * a + c) = 10 * a * (10 * a + c + b) + b * c :=
by sorry

end multiply_digits_correctness_l119_119533


namespace range_of_k_l119_119949

theorem range_of_k (k : ℝ) :
  (∀ x : ℝ, (k^2 - 1) * x^2 - (k + 1) * x + 1 > 0) ↔ (1 ≤ k ∧ k ≤ 5 / 3) := 
sorry

end range_of_k_l119_119949


namespace sum_proof_l119_119232

-- Define the context and assumptions
variables (F S T : ℕ)
axiom sum_of_numbers : F + S + T = 264
axiom first_number_twice_second : F = 2 * S
axiom third_number_one_third_first : T = F / 3
axiom second_number_given : S = 72

-- The theorem to prove the sum is 264 given the conditions
theorem sum_proof : F + S + T = 264 :=
by
  -- Given conditions already imply the theorem, the actual proof follows from these
  sorry

end sum_proof_l119_119232


namespace walls_per_person_l119_119952

theorem walls_per_person (people : ℕ) (rooms : ℕ) (r4_walls r5_walls : ℕ) (total_walls : ℕ) (walls_each_person : ℕ)
  (h1 : people = 5)
  (h2 : rooms = 9)
  (h3 : r4_walls = 5 * 4)
  (h4 : r5_walls = 4 * 5)
  (h5 : total_walls = r4_walls + r5_walls)
  (h6 : walls_each_person = total_walls / people) :
  walls_each_person = 8 := by
  sorry

end walls_per_person_l119_119952


namespace ratio_is_one_half_l119_119708

-- Define the problem conditions as constants
def robert_age_in_2_years : ℕ := 30
def years_until_robert_is_30 : ℕ := 2
def patrick_current_age : ℕ := 14

-- Using the conditions, set up the definitions for the proof
def robert_current_age : ℕ := robert_age_in_2_years - years_until_robert_is_30

-- Define the target ratio
def ratio_of_ages : ℚ := patrick_current_age / robert_current_age

-- Prove that the ratio of Patrick's age to Robert's age is 1/2
theorem ratio_is_one_half : ratio_of_ages = 1 / 2 :=
by
  sorry

end ratio_is_one_half_l119_119708


namespace directrix_of_parabola_l119_119787

theorem directrix_of_parabola (x y : ℝ) : y = 3 * x^2 - 6 * x + 1 → y = -25 / 12 :=
sorry

end directrix_of_parabola_l119_119787


namespace monotonically_decreasing_range_l119_119645

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 3 * x^2 - x + 1
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 6 * x - 1

theorem monotonically_decreasing_range (a : ℝ) :
  (∀ x : ℝ, f' a x ≤ 0) → a ≤ -3 := by
  sorry

end monotonically_decreasing_range_l119_119645


namespace find_angle_BCD_l119_119161

-- Defining the given conditions in the problem
def angleA : ℝ := 100
def angleD : ℝ := 120
def angleE : ℝ := 80
def angleABC : ℝ := 140
def pentagonInteriorAngleSum : ℝ := 540

-- Statement: Prove that the measure of ∠ BCD is 100 degrees given the conditions
theorem find_angle_BCD (h1 : angleA = 100) (h2 : angleD = 120) (h3 : angleE = 80) 
                       (h4 : angleABC = 140) (h5 : pentagonInteriorAngleSum = 540) :
    (angleBCD : ℝ) = 100 :=
sorry

end find_angle_BCD_l119_119161


namespace correct_parameterization_l119_119464

noncomputable def parametrize_curve (t : ℝ) : ℝ × ℝ :=
  (t, t^2)

theorem correct_parameterization : ∀ t : ℝ, ∃ x y : ℝ, parametrize_curve t = (x, y) ∧ y = x^2 :=
by
  intro t
  use t, t^2
  dsimp [parametrize_curve]
  exact ⟨rfl, rfl⟩

end correct_parameterization_l119_119464


namespace urn_contains_specific_balls_after_operations_l119_119455

def initial_red_balls : ℕ := 2
def initial_blue_balls : ℕ := 1
def total_operations : ℕ := 5
def final_red_balls : ℕ := 10
def final_blue_balls : ℕ := 6
def target_probability : ℚ := 16 / 115

noncomputable def urn_proba_result : ℚ := sorry

theorem urn_contains_specific_balls_after_operations :
  urn_proba_result = target_probability := sorry

end urn_contains_specific_balls_after_operations_l119_119455


namespace problem_1_problem_2_l119_119343

theorem problem_1 
  (h1 : 1 < Real.sqrt 2 ∧ Real.sqrt 2 < 2) : 
  Int.floor (5 - Real.sqrt 2) = 3 :=
sorry

theorem problem_2 
  (h2 : Real.sqrt 3 > 1) : 
  abs (1 - 2 * Real.sqrt 3) = 2 * Real.sqrt 3 - 1 :=
sorry

end problem_1_problem_2_l119_119343


namespace axis_of_symmetry_and_vertex_l119_119032

noncomputable def f (x : ℝ) : ℝ := -2 * x^2 + 4 * x + 1

theorem axis_of_symmetry_and_vertex :
  (∃ (a : ℝ), (f a = -2 * (a - 1)^2 + 3) ∧ a = 1) ∧ ∃ v, (v = (1, 3) ∧ ∀ x, f x = -2 * (x - 1)^2 + 3) :=
sorry

end axis_of_symmetry_and_vertex_l119_119032


namespace ratio_of_students_l119_119084

-- Define the conditions
def total_students : Nat := 800
def students_spaghetti : Nat := 320
def students_fettuccine : Nat := 160

-- The proof problem
theorem ratio_of_students (h1 : students_spaghetti = 320) (h2 : students_fettuccine = 160) :
  students_spaghetti / students_fettuccine = 2 := by
  sorry

end ratio_of_students_l119_119084


namespace min_value_of_expression_l119_119370

theorem min_value_of_expression :
  ∃ (a b : ℝ), (∃ (x : ℝ), 1 ≤ x ∧ x ≤ 2 ∧ x^2 + a * x + b - 3 = 0) ∧ a^2 + (b - 4)^2 = 2 :=
sorry

end min_value_of_expression_l119_119370


namespace jane_reading_days_l119_119234

theorem jane_reading_days
  (pages : ℕ)
  (half_pages : ℕ)
  (speed_first_half : ℕ)
  (speed_second_half : ℕ)
  (days_first_half : ℕ)
  (days_second_half : ℕ)
  (total_days : ℕ)
  (h1 : pages = 500)
  (h2 : half_pages = pages / 2)
  (h3 : speed_first_half = 10)
  (h4 : speed_second_half = 5)
  (h5 : days_first_half = half_pages / speed_first_half)
  (h6 : days_second_half = half_pages / speed_second_half)
  (h7 : total_days = days_first_half + days_second_half) :
  total_days = 75 :=
by
  sorry

end jane_reading_days_l119_119234


namespace variance_of_planted_trees_l119_119158

def number_of_groups := 10

def planted_trees : List ℕ := [5, 5, 5, 6, 6, 6, 6, 7, 7, 7]

noncomputable def mean (xs : List ℕ) : ℚ :=
  (xs.sum : ℚ) / (xs.length : ℚ)

noncomputable def variance (xs : List ℕ) : ℚ :=
  let m := mean xs
  (xs.map (λ x => (x - m) ^ 2)).sum / (xs.length : ℚ)

theorem variance_of_planted_trees :
  variance planted_trees = 0.6 := sorry

end variance_of_planted_trees_l119_119158


namespace ellipse_triangle_is_isosceles_right_l119_119231

theorem ellipse_triangle_is_isosceles_right (e : ℝ) (a b c k : ℝ)
  (H1 : e = (c / a))
  (H2 : e = (Real.sqrt 2) / 2)
  (H3 : b^2 = a^2 * (1 - e^2))
  (H4 : a = 2 * k)
  (H5 : b = k * Real.sqrt 2)
  (H6 : c = k * Real.sqrt 2) :
  (4 * k)^2 = (2 * (k * Real.sqrt 2))^2 + (2 * (k * Real.sqrt 2))^2 :=
by
  sorry

end ellipse_triangle_is_isosceles_right_l119_119231


namespace asymptotes_tangent_to_circle_l119_119639

theorem asymptotes_tangent_to_circle {m : ℝ} (hm : m > 0) 
  (hyp_eq : ∀ x y : ℝ, y^2 - (x^2 / m^2) = 1) 
  (circ_eq : ∀ x y : ℝ, x^2 + y^2 - 4 * y + 3 = 0) : 
  m = (Real.sqrt 3) / 3 :=
sorry

end asymptotes_tangent_to_circle_l119_119639


namespace price_of_soda_l119_119484

-- Definitions based on the conditions given in the problem
def initial_amount := 500
def cost_rice := 2 * 20
def cost_wheat_flour := 3 * 25
def remaining_balance := 235
def total_cost := cost_rice + cost_wheat_flour

-- Definition to be proved
theorem price_of_soda : initial_amount - total_cost - remaining_balance = 150 := by
  sorry

end price_of_soda_l119_119484


namespace cricket_player_average_increase_l119_119023

theorem cricket_player_average_increase (total_innings initial_innings next_run : ℕ) (initial_average desired_increase : ℕ) 
(h1 : initial_innings = 10) (h2 : initial_average = 32) (h3 : next_run = 76) : desired_increase = 4 :=
by
  sorry

end cricket_player_average_increase_l119_119023


namespace percentage_invalid_votes_l119_119556

theorem percentage_invalid_votes 
    (total_votes : ℕ)
    (candidate_A_votes : ℕ)
    (candidate_A_percentage : ℝ)
    (total_valid_percentage : ℝ) :
    total_votes = 560000 ∧
    candidate_A_votes = 357000 ∧
    candidate_A_percentage = 0.75 ∧
    total_valid_percentage = 100 - x ∧
    (0.75 * (total_valid_percentage / 100) * 560000 = 357000) →
    x = 15 :=
by
  sorry

end percentage_invalid_votes_l119_119556


namespace catch_up_time_l119_119265

def A_departure_time : ℕ := 8 * 60 -- in minutes
def B_departure_time : ℕ := 6 * 60 -- in minutes
def relative_speed (v : ℕ) : ℕ := 5 * v / 4 -- (2.5v effective) converted to integer math
def initial_distance (v : ℕ) : ℕ := 2 * v * 2 -- 4v distance (B's 2 hours lead)

theorem catch_up_time (v : ℕ) :  A_departure_time + ((initial_distance v * 4) / (relative_speed v - v)) = 1080 :=
by
  sorry

end catch_up_time_l119_119265


namespace line_through_point_hyperbola_l119_119584

theorem line_through_point_hyperbola {x y k : ℝ} : 
  (∃ k : ℝ, ∃ x y : ℝ, y = k * (x - 3) ∧ x^2 / 4 - y^2 = 1 ∧ (1 - 4 * k^2) = 0) → 
  (∃! k : ℝ, (k = 1 / 2) ∨ (k = -1 / 2)) := 
sorry

end line_through_point_hyperbola_l119_119584


namespace inscribed_cone_volume_l119_119839

theorem inscribed_cone_volume
  (H : ℝ) 
  (α : ℝ)
  (h_pos : 0 < H)
  (α_pos : 0 < α ∧ α < π / 2) :
  (1 / 12) * π * H ^ 3 * (Real.sin α) ^ 2 * (Real.sin (2 * α)) ^ 2 = 
  (1 / 3) * π * ((H * Real.sin α * Real.cos α / 2) ^ 2) * (H * (Real.sin α) ^ 2) :=
by sorry

end inscribed_cone_volume_l119_119839


namespace necessary_but_not_sufficient_condition_is_purely_imaginary_l119_119074

noncomputable def is_purely_imaginary (z : ℂ) : Prop :=
  ∃ (b : ℝ), z = ⟨0, b⟩

theorem necessary_but_not_sufficient_condition_is_purely_imaginary (a b : ℝ) (h_imaginary : is_purely_imaginary (⟨a, b⟩)) : 
  (a = 0) ∧ (b ≠ 0) :=
by
  sorry

end necessary_but_not_sufficient_condition_is_purely_imaginary_l119_119074


namespace r_iterated_six_times_l119_119597

def r (θ : ℚ) : ℚ := 1 / (1 - 2 * θ)

theorem r_iterated_six_times (θ : ℚ) : r (r (r (r (r (r θ))))) = θ :=
by sorry

example : r (r (r (r (r (r 10))))) = 10 :=
by rw [r_iterated_six_times 10]

end r_iterated_six_times_l119_119597


namespace midpoint_sum_l119_119416

theorem midpoint_sum (x1 y1 x2 y2 : ℕ) (h₁ : x1 = 4) (h₂ : y1 = 7) (h₃ : x2 = 12) (h₄ : y2 = 19) :
  (x1 + x2) / 2 + (y1 + y2) / 2 = 21 :=
by
  sorry

end midpoint_sum_l119_119416


namespace maria_total_cost_l119_119762

-- Define the costs of the items
def pencil_cost : ℕ := 8
def pen_cost : ℕ := pencil_cost / 2
def eraser_cost : ℕ := 2 * pen_cost

-- Define the total cost
def total_cost : ℕ := pen_cost + pencil_cost + eraser_cost

-- The theorem to prove
theorem maria_total_cost : total_cost = 20 := by
  sorry

end maria_total_cost_l119_119762


namespace product_pattern_l119_119654

theorem product_pattern (m n : ℝ) : 
  m * n = ( ( m + n ) / 2 ) ^ 2 - ( ( m - n ) / 2 ) ^ 2 := 
by 
  sorry

end product_pattern_l119_119654


namespace calculate_rate_l119_119591

-- Definitions corresponding to the conditions in the problem
def bankers_gain (td : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  td * rate * time

-- Given values according to the problem
def BG : ℝ := 7.8
def TD : ℝ := 65
def Time : ℝ := 1
def expected_rate_percentage : ℝ := 12

-- The mathematical proof problem statement in Lean 4
theorem calculate_rate : (BG = bankers_gain TD (expected_rate_percentage / 100) Time) :=
sorry

end calculate_rate_l119_119591


namespace number_of_boys_l119_119475

theorem number_of_boys (n : ℕ)
  (initial_avg_height : ℕ)
  (incorrect_height : ℕ)
  (correct_height : ℕ)
  (actual_avg_height : ℕ)
  (h1 : initial_avg_height = 184)
  (h2 : incorrect_height = 166)
  (h3 : correct_height = 106)
  (h4 : actual_avg_height = 182)
  (h5 : initial_avg_height * n - (incorrect_height - correct_height) = actual_avg_height * n) :
  n = 30 :=
sorry

end number_of_boys_l119_119475


namespace rent_of_first_apartment_l119_119107

theorem rent_of_first_apartment (R : ℝ) :
  let cost1 := R + 260 + (31 * 20 * 0.58)
  let cost2 := 900 + 200 + (21 * 20 * 0.58)
  (cost1 - cost2 = 76) → R = 800 :=
by
  intro h
  sorry

end rent_of_first_apartment_l119_119107


namespace football_goals_in_fifth_match_l119_119848

theorem football_goals_in_fifth_match (G : ℕ) (h1 : (4 / 5 : ℝ) = (4 - G) / 4 + 0.3) : G = 2 :=
by
  sorry

end football_goals_in_fifth_match_l119_119848


namespace find_x_l119_119030

/-- 
Prove that the value of x is 25 degrees, given the following conditions:
1. The sum of the angles in triangle BAC: angle_BAC + 50° + 55° = 180°
2. The angles forming a straight line DAE: 80° + angle_BAC + x = 180°
-/
theorem find_x (angle_BAC : ℝ) (x : ℝ)
  (h1 : angle_BAC + 50 + 55 = 180)
  (h2 : 80 + angle_BAC + x = 180) :
  x = 25 :=
  sorry

end find_x_l119_119030


namespace equation_of_line_parallel_to_x_axis_l119_119065

theorem equation_of_line_parallel_to_x_axis (x: ℝ) :
  ∃ (y: ℝ), (y-2=0) ∧ ∀ (P: ℝ × ℝ), (P = (1, 2)) → P.2 = 2 := 
by
  sorry

end equation_of_line_parallel_to_x_axis_l119_119065


namespace M_inter_N_l119_119754

def M : Set ℝ := {y | ∃ x : ℝ, y = 2^(-x)}
def N : Set ℝ := {y | ∃ x : ℝ, y = Real.sin x}

theorem M_inter_N : M ∩ N = {y | 0 < y ∧ y ≤ 1} :=
by
  sorry

end M_inter_N_l119_119754


namespace sum_of_squares_ge_one_third_l119_119892

theorem sum_of_squares_ge_one_third (a b c : ℝ) (h : a + b + c = 1) : 
  a^2 + b^2 + c^2 ≥ 1/3 := 
by 
  sorry

end sum_of_squares_ge_one_third_l119_119892


namespace polygon_sides_l119_119269

-- Definition of the problem conditions
def interiorAngleSum (n : ℕ) : ℕ := 180 * (n - 2)
def givenAngleSum (n : ℕ) : ℕ := 140 + 145 * (n - 1)

-- Problem statement: proving the number of sides
theorem polygon_sides (n : ℕ) (h : interiorAngleSum n = givenAngleSum n) : n = 10 :=
sorry

end polygon_sides_l119_119269


namespace geometric_sequence_n_l119_119279

-- Definition of the conditions

-- a_1 + a_n = 82
def condition1 (a₁ an : ℕ) : Prop := a₁ + an = 82
-- a_3 * a_{n-2} = 81
def condition2 (a₃ aₙm2 : ℕ) : Prop := a₃ * aₙm2 = 81
-- S_n = 121
def condition3 (Sₙ : ℕ) : Prop := Sₙ = 121

-- Prove n = 5 given the above conditions
theorem geometric_sequence_n (a₁ a₃ an aₙm2 Sₙ n : ℕ)
  (h1 : condition1 a₁ an)
  (h2 : condition2 a₃ aₙm2)
  (h3 : condition3 Sₙ) :
  n = 5 :=
sorry

end geometric_sequence_n_l119_119279


namespace raw_score_is_correct_l119_119413

-- Define the conditions
def points_per_correct : ℝ := 1
def points_subtracted_per_incorrect : ℝ := 0.25
def total_questions : ℕ := 85
def answered_questions : ℕ := 82
def correct_answers : ℕ := 70

-- Define the number of incorrect answers
def incorrect_answers : ℕ := answered_questions - correct_answers
-- Calculate the raw score
def raw_score : ℝ := 
  (correct_answers * points_per_correct) - (incorrect_answers * points_subtracted_per_incorrect)

-- Prove the raw score is 67
theorem raw_score_is_correct : raw_score = 67 := 
by
  sorry

end raw_score_is_correct_l119_119413


namespace consecutive_sunny_days_l119_119316

theorem consecutive_sunny_days (n_sunny_days : ℕ) (n_days_year : ℕ) (days_to_stay : ℕ) (condition1 : n_sunny_days = 350) (condition2 : n_days_year = 365) :
  days_to_stay = 32 :=
by
  sorry

end consecutive_sunny_days_l119_119316


namespace hamburgers_left_over_l119_119731

-- Define the conditions as constants
def hamburgers_made : ℕ := 9
def hamburgers_served : ℕ := 3

-- Prove that the number of hamburgers left over is 6
theorem hamburgers_left_over : hamburgers_made - hamburgers_served = 6 := 
by
  sorry

end hamburgers_left_over_l119_119731


namespace fraction_div_addition_l119_119944

noncomputable def fraction_5_6 : ℚ := 5 / 6
noncomputable def fraction_9_10 : ℚ := 9 / 10
noncomputable def fraction_1_15 : ℚ := 1 / 15
noncomputable def fraction_402_405 : ℚ := 402 / 405

theorem fraction_div_addition :
  (fraction_5_6 / fraction_9_10) + fraction_1_15 = fraction_402_405 :=
by
  sorry

end fraction_div_addition_l119_119944


namespace amy_carl_distance_after_2_hours_l119_119314

-- Conditions
def amy_rate : ℤ := 1
def carl_rate : ℤ := 2
def amy_interval : ℤ := 20
def carl_interval : ℤ := 30
def time_hours : ℤ := 2
def minutes_per_hour : ℤ := 60

-- Derived values
def time_minutes : ℤ := time_hours * minutes_per_hour
def amy_distance : ℤ := time_minutes / amy_interval * amy_rate
def carl_distance : ℤ := time_minutes / carl_interval * carl_rate

-- Question and answer pair
def distance_amy_carl : ℤ := amy_distance + carl_distance
def expected_distance : ℤ := 14

-- The theorem to prove
theorem amy_carl_distance_after_2_hours : distance_amy_carl = expected_distance := by
  sorry

end amy_carl_distance_after_2_hours_l119_119314


namespace rectangle_length_eq_15_l119_119425

theorem rectangle_length_eq_15 (w l s p_rect p_square : ℝ)
    (h_w : w = 9)
    (h_s : s = 12)
    (h_p_square : p_square = 4 * s)
    (h_p_rect : p_rect = 2 * w + 2 * l)
    (h_eq_perimeters : p_square = p_rect) : l = 15 := by
  sorry

end rectangle_length_eq_15_l119_119425


namespace probability_of_same_color_is_correct_l119_119014

def probability_same_color (blue_balls yellow_balls : ℕ) : ℚ :=
  let total_balls := blue_balls + yellow_balls
  let prob_blue := (blue_balls / total_balls : ℚ)
  let prob_yellow := (yellow_balls / total_balls : ℚ)
  (prob_blue ^ 2) + (prob_yellow ^ 2)

theorem probability_of_same_color_is_correct :
  probability_same_color 8 5 = 89 / 169 :=
by 
  sorry

end probability_of_same_color_is_correct_l119_119014


namespace seating_arrangement_l119_119174

def num_ways_to_seat (A B C D E F : Type) (chairs : List (Option Type)) : Nat := sorry

theorem seating_arrangement {A B C D E F : Type} :
  ∀ (chairs : List (Option Type)),
    (A ≠ B ∧ A ≠ C ∧ F ≠ B) → num_ways_to_seat A B C D E F chairs = 28 :=
by
  sorry

end seating_arrangement_l119_119174


namespace last_digit_of_x95_l119_119365

theorem last_digit_of_x95 (x : ℕ) : 
  (x^95 % 10) - (3^58 % 10) = 4 % 10 → (x^95 % 10 = 3) := by
  sorry

end last_digit_of_x95_l119_119365


namespace find_y_l119_119206

theorem find_y (y : ℝ) (h : 9 * y^2 + 36 * y^2 + 9 * y^2 = 1300) : 
  y = Real.sqrt 1300 / Real.sqrt 54 :=
by 
  sorry

end find_y_l119_119206


namespace triangle_inequality_range_l119_119826

theorem triangle_inequality_range (x : ℝ) (h1 : 4 + 5 > x) (h2 : 4 + x > 5) (h3 : 5 + x > 4) :
  1 < x ∧ x < 9 := 
by
  sorry

end triangle_inequality_range_l119_119826


namespace base_eight_to_base_ten_l119_119970

theorem base_eight_to_base_ten : ∃ n : ℕ, 47 = 4 * 8 + 7 ∧ n = 39 :=
by
  sorry

end base_eight_to_base_ten_l119_119970


namespace min_n_for_factorization_l119_119613

theorem min_n_for_factorization (n : ℤ) :
  (∃ A B : ℤ, 6 * A * B = 60 ∧ n = 6 * B + A) → n = 66 :=
sorry

end min_n_for_factorization_l119_119613


namespace jovana_shells_l119_119015

def initial_weight : ℕ := 5
def added_weight : ℕ := 23
def total_weight : ℕ := 28

theorem jovana_shells :
  initial_weight + added_weight = total_weight :=
by
  sorry

end jovana_shells_l119_119015


namespace hyperbola_center_l119_119283

theorem hyperbola_center (x1 y1 x2 y2 : ℝ) (h₁ : x1 = 3) (h₂ : y1 = 2) (h₃ : x2 = 11) (h₄ : y2 = 6) :
  (x1 + x2) / 2 = 7 ∧ (y1 + y2) / 2 = 4 :=
by
  -- Use the conditions h₁, h₂, h₃, and h₄ to substitute values and prove the statement
  sorry

end hyperbola_center_l119_119283


namespace field_trip_fraction_l119_119186

theorem field_trip_fraction (b g : ℕ) (hb : g = b)
  (girls_trip_fraction : ℚ := 4/5)
  (boys_trip_fraction : ℚ := 3/4) :
  girls_trip_fraction * g / (girls_trip_fraction * g + boys_trip_fraction * b) = 16 / 31 :=
by {
  sorry
}

end field_trip_fraction_l119_119186


namespace measureable_weights_count_l119_119927

theorem measureable_weights_count (a b c : ℕ) (ha : a = 1) (hb : b = 3) (hc : c = 9) :
  ∃ s : Finset ℕ, s.card = 13 ∧ ∀ x ∈ s, x ≥ 1 ∧ x ≤ 13 := 
sorry

end measureable_weights_count_l119_119927


namespace josh_500_coins_impossible_l119_119289

theorem josh_500_coins_impossible : ¬ ∃ (x y : ℕ), x + y ≤ 500 ∧ 36 * x + 6 * y + (500 - x - y) = 3564 := 
sorry

end josh_500_coins_impossible_l119_119289


namespace triangle_area_l119_119472

def point := (ℚ × ℚ)

def vertex1 : point := (3, -3)
def vertex2 : point := (3, 4)
def vertex3 : point := (8, -3)

theorem triangle_area :
  let base := (vertex3.1 - vertex1.1 : ℚ)
  let height := (vertex2.2 - vertex1.2 : ℚ)
  (base * height / 2) = 17.5 :=
by
  sorry

end triangle_area_l119_119472


namespace max_successful_free_throws_l119_119774

theorem max_successful_free_throws (a b : ℕ) 
  (h1 : a + b = 105) 
  (h2 : a > 0)
  (h3 : b > 0)
  (ha : a % 3 = 0)
  (hb : b % 5 = 0)
  : (a / 3 + 3 * (b / 5)) ≤ 59 := sorry

end max_successful_free_throws_l119_119774


namespace CanVolume_l119_119874

variable (X Y : Type) [Field X] [Field Y] (V W : X)

theorem CanVolume (mix_ratioX mix_ratioY drawn_volume new_ratioX new_ratioY : ℤ)
  (h1 : mix_ratioX = 5) (h2 : mix_ratioY = 7) (h3 : drawn_volume = 12) 
  (h4 : new_ratioX = 4) (h5 : new_ratioY = 7) :
  V = 72 ∧ W = 72 := 
sorry

end CanVolume_l119_119874


namespace smallest_positive_period_l119_119791

open Real

-- Define conditions
def max_value_condition (b a : ℝ) : Prop := b + a = -1
def min_value_condition (b a : ℝ) : Prop := b - a = -5

-- Define the period of the function
def period (f : ℝ → ℝ) (T : ℝ) : Prop := ∀ x, f (x + T) = f x

-- Main theorem
theorem smallest_positive_period (a b : ℝ) (h1 : a < 0) 
  (h2 : max_value_condition b a) 
  (h3 : min_value_condition b a) : 
  period (fun x => tan ((3 * a + b) * x)) (π / 9) :=
by
  sorry

end smallest_positive_period_l119_119791


namespace old_toilet_water_per_flush_correct_l119_119813

noncomputable def old_toilet_water_per_flush (water_saved : ℕ) (flushes_per_day : ℕ) (days_in_june : ℕ) (reduction_percentage : ℚ) : ℚ :=
  let total_flushes := flushes_per_day * days_in_june
  let water_saved_per_flush := water_saved / total_flushes
  let reduction_factor := reduction_percentage
  let original_water_per_flush := water_saved_per_flush / (1 - reduction_factor)
  original_water_per_flush

theorem old_toilet_water_per_flush_correct :
  old_toilet_water_per_flush 1800 15 30 (80 / 100) = 5 := by
  sorry

end old_toilet_water_per_flush_correct_l119_119813


namespace no_real_roots_of_polynomial_l119_119899

noncomputable def p (x : ℝ) : ℝ := sorry

theorem no_real_roots_of_polynomial (p : ℝ → ℝ) (h_deg : ∃ n : ℕ, n ≥ 1 ∧ ∀ x: ℝ, p x = x^n) :
  (∀ x, p x * p (2 * x^2) = p (3 * x^3 + x)) →
  ¬ ∃ α : ℝ, p α = 0 := sorry

end no_real_roots_of_polynomial_l119_119899


namespace standard_eq_circle_C_equation_line_AB_l119_119060

-- Define the center of circle C and the line l
def center_C : ℝ × ℝ := (2, 1)
def line_l (x y : ℝ) : Prop := x = 3

-- Define the standard equation of circle C
def eq_circle_C (x y : ℝ) : Prop :=
  (x - center_C.1)^2 + (y - center_C.2)^2 = 1

-- Equation of circle O
def eq_circle_O (x y : ℝ) : Prop :=
  x^2 + y^2 = 4

-- Define the condition that circle C intersects with circle O at points A and B
def intersects (x y : ℝ) : Prop :=
  eq_circle_C x y ∧ eq_circle_O x y

-- Define the equation of line AB in general form
def eq_line_AB (x y : ℝ) : Prop :=
  2 * x + y - 4 = 0

-- Prove the standard equation of circle C is (x-2)^2 + (y-1)^2 = 1
theorem standard_eq_circle_C:
  eq_circle_C x y ↔ (x - 2)^2 + (y - 1)^2 = 1 :=
sorry

-- Prove that the equation of line AB is 2x + y - 4 = 0, given the intersection points A and B
theorem equation_line_AB (x y : ℝ) (h : intersects x y) :
  eq_line_AB x y :=
sorry

end standard_eq_circle_C_equation_line_AB_l119_119060


namespace complex_number_purely_imaginary_l119_119716

theorem complex_number_purely_imaginary (m : ℝ) :
  (m^2 - 2 * m - 3 = 0) ∧ (m^2 - 1 ≠ 0) → m = 3 :=
by
  intros h
  sorry

end complex_number_purely_imaginary_l119_119716


namespace find_a_l119_119780

theorem find_a (a : ℝ) :
  (∀ x, x < 2 → 0 < a - 3 * x) ↔ (a = 6) :=
by
  sorry

end find_a_l119_119780


namespace A_share_of_gain_l119_119951

-- Definitions of conditions
variables 
  (x : ℕ) -- Initial investment by A
  (annual_gain : ℕ := 24000) -- Total annual gain
  (A_investment_period : ℕ := 12) -- Months A invested
  (B_investment_period : ℕ := 6) -- Months B invested after 6 months
  (C_investment_period : ℕ := 4) -- Months C invested after 8 months

-- Investment ratios
def A_ratio := x * A_investment_period
def B_ratio := (2 * x) * B_investment_period
def C_ratio := (3 * x) * C_investment_period

-- Proof statement
theorem A_share_of_gain : 
  A_ratio = 12 * x ∧ B_ratio = 12 * x ∧ C_ratio = 12 * x ∧ annual_gain = 24000 →
  annual_gain / 3 = 8000 :=
by
  sorry

end A_share_of_gain_l119_119951


namespace total_number_of_water_filled_jars_l119_119798

theorem total_number_of_water_filled_jars : 
  ∃ (x : ℕ), 28 = x * (1/4 + 1/2 + 1) ∧ 3 * x = 48 :=
by
  sorry

end total_number_of_water_filled_jars_l119_119798


namespace evaluate_expression_l119_119840

variable (b : ℝ) -- assuming b is a real number, (if b should be of different type, modify accordingly)

theorem evaluate_expression (y : ℝ) (h : y = b + 9) : y - b + 5 = 14 :=
by
  sorry

end evaluate_expression_l119_119840


namespace sum_of_first_six_terms_geometric_sequence_l119_119296

-- conditions
def a : ℚ := 1/4
def r : ℚ := 1/4

-- geometric series sum function
def geom_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

-- target sum of first six terms
def S_6 : ℚ := geom_sum a r 6

-- proof statement
theorem sum_of_first_six_terms_geometric_sequence :
  S_6 = 1365 / 4096 :=
by 
  sorry

end sum_of_first_six_terms_geometric_sequence_l119_119296


namespace at_least_two_inequalities_hold_l119_119969

variable {a b c : ℝ}

theorem at_least_two_inequalities_hold (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a + b + c ≥ a * b * c) :
  (2 / a + 3 / b + 6 / c ≥ 6 ∨ 2 / b + 3 / c + 6 / a ≥ 6) ∨
  (2 / b + 3 / c + 6 / a ≥ 6 ∨ 2 / c + 3 / a + 6 / b ≥ 6) ∨
  (2 / c + 3 / a + 6 / b ≥ 6 ∨ 2 / a + 3 / b + 6 / c ≥ 6) :=
  sorry

end at_least_two_inequalities_hold_l119_119969


namespace salary_restoration_l119_119512

theorem salary_restoration (S : ℝ) : 
  let reduced_salary := 0.7 * S
  let restore_factor := 1 / 0.7
  let percentage_increase := restore_factor - 1
  percentage_increase * 100 = 42.857 :=
by
  sorry

end salary_restoration_l119_119512


namespace range_of_m_l119_119977

noncomputable def f (m x : ℝ) : ℝ := x^3 + m * x^2 + (m + 6) * x + 1

theorem range_of_m (m : ℝ) : 
  (∀ x y : ℝ, ∃ c : ℝ, (f m x) ≤ c ∧ (f m y) ≥ (f m x) ∧ ∀ z : ℝ, f m z ≥ f m x ∧ f m z ≤ c) ↔ (m < -3 ∨ m > 6) :=
by
  sorry

end range_of_m_l119_119977


namespace not_integer_fraction_l119_119007

theorem not_integer_fraction (a b : ℕ) (h1 : a > b) (h2 : b > 2) : ¬ (∃ k : ℤ, (2^a + 1) = k * (2^b - 1)) :=
sorry

end not_integer_fraction_l119_119007


namespace sum_of_a_and_b_l119_119363

noncomputable def a : ℕ :=
sorry

noncomputable def b : ℕ :=
sorry

theorem sum_of_a_and_b :
  (100 ≤ a ∧ a ≤ 999) ∧ (1000 ≤ b ∧ b ≤ 9999) ∧ (10000 * a + b = 7 * a * b) ->
  a + b = 1458 :=
by
  sorry

end sum_of_a_and_b_l119_119363


namespace multiplication_of_mixed_number_l119_119330

theorem multiplication_of_mixed_number :
  7 * (9 + 2/5 : ℚ) = 65 + 4/5 :=
by
  -- to start the proof
  sorry

end multiplication_of_mixed_number_l119_119330


namespace true_proposition_l119_119308

-- Define the propositions p and q
def p : Prop := 2 % 2 = 0
def q : Prop := 5 % 2 = 0

-- Define the problem statement
theorem true_proposition (hp : p) (hq : ¬ q) : p ∨ q :=
by
  sorry

end true_proposition_l119_119308


namespace sequence_formula_l119_119200

theorem sequence_formula (a : ℕ → ℕ) (c : ℕ) (h₁ : a 1 = 2) (h₂ : ∀ n, a (n + 1) = a n + c * n) 
(h₃ : a 1 ≠ a 2) (h₄ : a 2 * a 2 = a 1 * a 3) : c = 2 ∧ ∀ n, a n = n^2 - n + 2 :=
by
  sorry

end sequence_formula_l119_119200


namespace quadratic_minimum_val_l119_119452

theorem quadratic_minimum_val (p q x : ℝ) (hp : p > 0) (hq : q > 0) : 
  (∀ x, x^2 - 2 * p * x + 4 * q ≥ p^2 - 4 * q) := 
by
  sorry

end quadratic_minimum_val_l119_119452


namespace similar_triangles_perimeters_and_area_ratios_l119_119441

theorem similar_triangles_perimeters_and_area_ratios
  (m1 m2 : ℝ) (p_sum : ℝ) (ratio_p : ℝ) (ratio_a : ℝ) :
  m1 = 10 →
  m2 = 4 →
  p_sum = 140 →
  ratio_p = 5 / 2 →
  ratio_a = 25 / 4 →
  (∃ (p1 p2 : ℝ), p1 + p2 = p_sum ∧ p1 = (5 / 7) * p_sum ∧ p2 = (2 / 7) * p_sum ∧ ratio_a = (ratio_p)^2) :=
by
  sorry

end similar_triangles_perimeters_and_area_ratios_l119_119441


namespace rectangle_perimeter_ratio_l119_119748

theorem rectangle_perimeter_ratio
    (initial_height : ℕ)
    (initial_width : ℕ)
    (H_initial_height : initial_height = 2)
    (H_initial_width : initial_width = 4)
    (fold1_height : ℕ)
    (fold1_width : ℕ)
    (H_fold1_height : fold1_height = initial_height / 2)
    (H_fold1_width : fold1_width = initial_width)
    (fold2_height : ℕ)
    (fold2_width : ℕ)
    (H_fold2_height : fold2_height = fold1_height)
    (H_fold2_width : fold2_width = fold1_width / 2)
    (cut_height : ℕ)
    (cut_width : ℕ)
    (H_cut_height : cut_height = fold2_height)
    (H_cut_width : cut_width = fold2_width) :
    (2 * (cut_height + cut_width)) / (2 * (fold1_height + fold1_width)) = 3 / 5 := 
    by sorry

end rectangle_perimeter_ratio_l119_119748


namespace matrix_det_problem_l119_119288

-- Define the determinant of a 2x2 matrix
def det (a b c d : ℤ) : ℤ := a * d - b * c

-- State the problem in Lean
theorem matrix_det_problem : 2 * det 5 7 2 3 = 2 := by
  sorry

end matrix_det_problem_l119_119288


namespace gcf_180_270_l119_119711

theorem gcf_180_270 : Int.gcd 180 270 = 90 :=
sorry

end gcf_180_270_l119_119711


namespace inscribed_circle_radius_l119_119681

theorem inscribed_circle_radius (r : ℝ) (R : ℝ) (θ : ℝ) (tangent : ℝ) :
    θ = π / 3 →
    R = 5 →
    tangent = (5 : ℝ) * (Real.sqrt 2 - 1) →
    r * (1 + Real.sqrt 2) = R →
    r = 5 * (Real.sqrt 2 - 1) := 
by sorry

end inscribed_circle_radius_l119_119681


namespace stratified_sampling_middle_schools_l119_119779

theorem stratified_sampling_middle_schools (high_schools : ℕ) (middle_schools : ℕ) (elementary_schools : ℕ) (total_selected : ℕ) 
    (h_high_schools : high_schools = 10) (h_middle_schools : middle_schools = 30) (h_elementary_schools : elementary_schools = 60)
    (h_total_selected : total_selected = 20) : 
    middle_schools * (total_selected / (high_schools + middle_schools + elementary_schools)) = 6 := 
by 
  sorry

end stratified_sampling_middle_schools_l119_119779


namespace total_amount_paid_l119_119958

-- Definitions
def original_aquarium_price : ℝ := 120
def aquarium_discount : ℝ := 0.5
def aquarium_coupon : ℝ := 0.1
def aquarium_sales_tax : ℝ := 0.05

def plants_decorations_price_before_discount : ℝ := 75
def plants_decorations_discount : ℝ := 0.15
def plants_decorations_sales_tax : ℝ := 0.08

def fish_food_price : ℝ := 25
def fish_food_sales_tax : ℝ := 0.06

-- Final result to be proved
theorem total_amount_paid : 
  let discounted_aquarium_price := original_aquarium_price * (1 - aquarium_discount)
  let coupon_aquarium_price := discounted_aquarium_price * (1 - aquarium_coupon)
  let total_aquarium_price := coupon_aquarium_price * (1 + aquarium_sales_tax)
  let discounted_plants_decorations_price := plants_decorations_price_before_discount * (1 - plants_decorations_discount)
  let total_plants_decorations_price := discounted_plants_decorations_price * (1 + plants_decorations_sales_tax)
  let total_fish_food_price := fish_food_price * (1 + fish_food_sales_tax)
  total_aquarium_price + total_plants_decorations_price + total_fish_food_price = 152.05 :=
by 
  sorry

end total_amount_paid_l119_119958


namespace second_number_less_than_twice_first_l119_119683

theorem second_number_less_than_twice_first (x y z : ℤ) (h1 : y = 37) (h2 : x + y = 57) (h3 : y = 2 * x - z) : z = 3 :=
by
  sorry

end second_number_less_than_twice_first_l119_119683


namespace cubic_solution_l119_119451

theorem cubic_solution (x : ℝ) (h : x^3 + (x + 2)^3 + (x + 4)^3 = (x + 6)^3) : x = 6 :=
by
  sorry

end cubic_solution_l119_119451


namespace inequality_proof_l119_119006

theorem inequality_proof
  (a b c : ℝ)
  (h1 : 0 < a)
  (h2 : 0 < b)
  (h3 : 0 < c)
  (h4 : c^2 + a * b = a^2 + b^2) :
  c^2 + a * b ≤ a * c + b * c := sorry

end inequality_proof_l119_119006


namespace find_n_l119_119661

theorem find_n (x : ℝ) (n : ℝ)
  (h1 : Real.log (Real.sin x) + Real.log (Real.cos x) = -2)
  (h2 : Real.log (Real.sin x + Real.cos x) = 1 / 2 * (Real.log n - 2)) :
  n = Real.exp 2 + 2 :=
by
  sorry

end find_n_l119_119661


namespace wood_allocation_l119_119436

theorem wood_allocation (x y : ℝ) (h1 : 50 * x * 4 = 300 * y) (h2 : x + y = 5) : x = 3 :=
by
  sorry

end wood_allocation_l119_119436


namespace binomial_60_3_l119_119671

theorem binomial_60_3 : Nat.choose 60 3 = 34220 := by
  sorry

end binomial_60_3_l119_119671


namespace relationship_among_ys_l119_119720

-- Define the linear function
def linear_function (x : ℝ) (b : ℝ) : ℝ :=
  -2 * x + b

-- Define the points on the graph
def y1 (b : ℝ) : ℝ :=
  linear_function (-2) b

def y2 (b : ℝ) : ℝ :=
  linear_function (-1) b

def y3 (b : ℝ) : ℝ :=
  linear_function 1 b

-- Theorem to prove the relation among y1, y2, y3
theorem relationship_among_ys (b : ℝ) : y1 b > y2 b ∧ y2 b > y3 b :=
by
  sorry

end relationship_among_ys_l119_119720


namespace solution_set_of_tan_eq_two_l119_119163

open Real

theorem solution_set_of_tan_eq_two :
  {x | ∃ k : ℤ, x = k * π + (-1 : ℤ) ^ k * arctan 2} = {x | tan x = 2} :=
by
  sorry

end solution_set_of_tan_eq_two_l119_119163


namespace length_of_train_l119_119902

theorem length_of_train (speed_kmh : ℕ) (time_seconds : ℕ) (h_speed : speed_kmh = 60) (h_time : time_seconds = 36) :
  let time_hours := (time_seconds : ℚ) / 3600
  let distance_km := (speed_kmh : ℚ) * time_hours
  let distance_m := distance_km * 1000
  distance_m = 600 :=
by
  sorry

end length_of_train_l119_119902


namespace ratio_of_men_to_women_l119_119354
open Nat

theorem ratio_of_men_to_women 
  (total_players : ℕ) 
  (players_per_group : ℕ) 
  (extra_women_per_group : ℕ) 
  (H_total_players : total_players = 20) 
  (H_players_per_group : players_per_group = 3) 
  (H_extra_women_per_group : extra_women_per_group = 1) 
  : (7 / 13 : ℝ) = 7 / 13 :=
by
  -- Conditions
  have H1 : total_players = 20 := H_total_players
  have H2 : players_per_group = 3 := H_players_per_group
  have H3 : extra_women_per_group = 1 := H_extra_women_per_group
  -- The correct answer
  sorry

end ratio_of_men_to_women_l119_119354


namespace price_difference_is_99_cents_l119_119837

-- Definitions for the conditions
def list_price : ℚ := 3996 / 100
def discount_super_savers : ℚ := 9
def discount_penny_wise : ℚ := 25 / 100 * list_price

-- Sale prices calculated based on the given conditions
def sale_price_super_savers : ℚ := list_price - discount_super_savers
def sale_price_penny_wise : ℚ := list_price - discount_penny_wise

-- Difference in prices
def price_difference : ℚ := sale_price_super_savers - sale_price_penny_wise

-- Prove that the price difference in cents is 99
theorem price_difference_is_99_cents : price_difference = 99 / 100 := 
by
  sorry

end price_difference_is_99_cents_l119_119837


namespace lattice_point_distance_l119_119391

theorem lattice_point_distance (d : ℝ) : 
  (∃ (r : ℝ), r = 2020 ∧ (∀ (A B C D : ℝ), 
  A = 0 ∧ B = 4040 ∧ C = 2020 ∧ D = 4040) 
  ∧ (∃ (P Q : ℝ), P = 0.25 ∧ Q = 1)) → 
  d = 0.3 := 
by
  sorry

end lattice_point_distance_l119_119391


namespace clara_biked_more_l119_119588

def clara_speed : ℕ := 18
def denise_speed : ℕ := 16
def race_duration : ℕ := 5

def clara_distance := clara_speed * race_duration
def denise_distance := denise_speed * race_duration
def distance_difference := clara_distance - denise_distance

theorem clara_biked_more : distance_difference = 10 := by
  sorry

end clara_biked_more_l119_119588


namespace area_ratio_proof_l119_119640

noncomputable def area_ratio (a b c d : ℝ) (h1 : a / c = 2 / 3) (h2 : b / d = 2 / 3) : ℝ := 
  (a * b) / (c * d)

theorem area_ratio_proof (a b c d : ℝ) (h1 : a / c = 2 / 3) (h2 : b / d = 2 / 3) :
  area_ratio a b c d h1 h2 = 4 / 9 := by
  sorry

end area_ratio_proof_l119_119640


namespace largest_integer_a_can_be_less_than_l119_119767

theorem largest_integer_a_can_be_less_than (a b : ℕ) (h1 : 9 < a) (h2 : 19 < b) (h3 : b < 31) (h4 : a / b = 2 / 3) :
  a < 21 :=
sorry

end largest_integer_a_can_be_less_than_l119_119767


namespace cone_lateral_surface_area_l119_119703

theorem cone_lateral_surface_area (r : ℝ) (theta : ℝ) (h_r : r = 3) (h_theta : theta = 90) : 
  let base_circumference := 2 * Real.pi * r
  let R := 12
  let lateral_surface_area := (1 / 2) * base_circumference * R 
  lateral_surface_area = 36 * Real.pi :=
by
  sorry

end cone_lateral_surface_area_l119_119703


namespace largest_value_l119_119101

theorem largest_value :
  let A := 3 + 1 + 4
  let B := 3 * 1 + 4
  let C := 3 + 1 * 4
  let D := 3 * 1 * 4
  let E := 3 + 0 * 1 + 4
  D > A ∧ D > B ∧ D > C ∧ D > E :=
by
  -- conditions
  let A := 3 + 1 + 4
  let B := 3 * 1 + 4
  let C := 3 + 1 * 4
  let D := 3 * 1 * 4
  let E := 3 + 0 * 1 + 4
  -- sorry to skip the proof
  sorry

end largest_value_l119_119101


namespace total_running_duration_l119_119764

-- Conditions
def speed1 := 15 -- speed during the first part in mph
def time1 := 3 -- time during the first part in hours
def speed2 := 19 -- speed during the second part in mph
def distance2 := 190 -- distance during the second part in miles

-- Initialize
def distance1 := speed1 * time1 -- distance covered in the first part in miles

def time2 := distance2 / speed2 -- time to cover the distance in the second part in hours

-- Total duration
def total_duration := time1 + time2

-- Proof statement
theorem total_running_duration : total_duration = 13 :=
by
  sorry

end total_running_duration_l119_119764


namespace erasers_total_l119_119730

-- Define the initial amount of erasers
def initialErasers : Float := 95.0

-- Define the amount of erasers Marie buys
def boughtErasers : Float := 42.0

-- Define the total number of erasers Marie ends with
def totalErasers : Float := 137.0

-- The theorem that needs to be proven
theorem erasers_total 
  (initial : Float := initialErasers)
  (bought : Float := boughtErasers)
  (total : Float := totalErasers) :
  initial + bought = total :=
sorry

end erasers_total_l119_119730


namespace quadratic_inequality_solution_l119_119561

theorem quadratic_inequality_solution (k : ℝ) :
  (∀ x : ℝ, x^2 - (k - 2) * x - 2 * k + 4 < 0) ↔ (-6 < k ∧ k < 2) :=
by
  sorry

end quadratic_inequality_solution_l119_119561


namespace lines_are_parallel_l119_119536

def line1 (x : ℝ) : ℝ := 2 * x + 1
def line2 (x : ℝ) : ℝ := 2 * x + 5

theorem lines_are_parallel : ∀ x y : ℝ, line1 x = y → line2 x = y → false :=
by
  sorry

end lines_are_parallel_l119_119536


namespace max_surface_area_l119_119212

theorem max_surface_area (l w h : ℕ) (h_conditions : l + w + h = 88) : 
  2 * (l * w + l * h + w * h) ≤ 224 :=
sorry

end max_surface_area_l119_119212


namespace least_sales_needed_not_lose_money_l119_119869

noncomputable def old_salary : ℝ := 75000
noncomputable def new_salary_base : ℝ := 45000
noncomputable def commission_rate : ℝ := 0.15
noncomputable def sale_amount : ℝ := 750

theorem least_sales_needed_not_lose_money : 
  ∃ (n : ℕ), n * (commission_rate * sale_amount) ≥ (old_salary - new_salary_base) ∧ n = 267 := 
by
  -- The proof will show that n = 267 is the least number of sales needed to not lose money.
  existsi 267
  sorry

end least_sales_needed_not_lose_money_l119_119869


namespace solve_system_of_equations_solve_system_of_inequalities_l119_119196

-- For the system of equations
theorem solve_system_of_equations (x y : ℝ) (h1 : 3 * x + 4 * y = 2) (h2 : 2 * x - y = 5) : 
    x = 2 ∧ y = -1 :=
sorry

-- For the system of inequalities
theorem solve_system_of_inequalities (x : ℝ) 
    (h1 : x - 3 * (x - 1) < 7) 
    (h2 : x - 2 ≤ (2 * x - 3) / 3) :
    -2 < x ∧ x ≤ 3 :=
sorry

end solve_system_of_equations_solve_system_of_inequalities_l119_119196


namespace sum_of_squares_multiple_of_five_sum_of_consecutive_squares_multiple_of_five_l119_119089

theorem sum_of_squares_multiple_of_five :
  ( (-1)^2 + 0^2 + 1^2 + 2^2 + 3^2 ) % 5 = 0 :=
by
  sorry

theorem sum_of_consecutive_squares_multiple_of_five 
  (n : ℤ) :
  ((n - 2)^2 + (n - 1)^2 + n^2 + (n + 1)^2 + (n + 2)^2) % 5 = 0 :=
by
  sorry

end sum_of_squares_multiple_of_five_sum_of_consecutive_squares_multiple_of_five_l119_119089


namespace area_union_example_l119_119143

noncomputable def area_union_square_circle (s r : ℝ) : ℝ :=
  let A_square := s ^ 2
  let A_circle := Real.pi * r ^ 2
  let A_overlap := (1 / 4) * A_circle
  A_square + A_circle - A_overlap

theorem area_union_example : (area_union_square_circle 10 10) = 100 + 75 * Real.pi :=
by
  sorry

end area_union_example_l119_119143


namespace original_price_of_dish_l119_119150

variable (P : ℝ)

def john_paid (P : ℝ) : ℝ := 0.9 * P + 0.15 * P
def jane_paid (P : ℝ) : ℝ := 0.9 * P + 0.135 * P

theorem original_price_of_dish (h : john_paid P = jane_paid P + 1.26) : P = 84 := by
  sorry

end original_price_of_dish_l119_119150


namespace factor_of_increase_l119_119075

noncomputable def sum_arithmetic_progression (a1 d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a1 + (n - 1) * d) / 2

theorem factor_of_increase (a1 d n : ℕ) (h1 : a1 > 0) (h2 : (sum_arithmetic_progression a1 (3 * d) n = 2 * sum_arithmetic_progression a1 d n)) :
  sum_arithmetic_progression a1 (4 * d) n = (5 / 2) * sum_arithmetic_progression a1 d n :=
sorry

end factor_of_increase_l119_119075


namespace white_tshirts_per_pack_l119_119877

-- Define the given conditions
def packs_white := 5
def packs_blue := 3
def t_shirts_per_blue_pack := 9
def total_t_shirts := 57

-- Define the total number of blue t-shirts
def total_blue_t_shirts := packs_blue * t_shirts_per_blue_pack

-- Define the variable W for the number of white t-shirts per pack
variable (W : ℕ)

-- Define the total number of white t-shirts
def total_white_t_shirts := packs_white * W

-- State the theorem to prove
theorem white_tshirts_per_pack :
    total_white_t_shirts + total_blue_t_shirts = total_t_shirts → W = 6 :=
by
  sorry

end white_tshirts_per_pack_l119_119877


namespace pb_distance_l119_119241

theorem pb_distance (a b c d PA PD PC PB : ℝ)
  (hPA : PA = 5)
  (hPD : PD = 6)
  (hPC : PC = 7)
  (h1 : a^2 + b^2 = PA^2)
  (h2 : b^2 + c^2 = PC^2)
  (h3 : c^2 + d^2 = PD^2)
  (h4 : d^2 + a^2 = PB^2) :
  PB = Real.sqrt 38 := by
  sorry

end pb_distance_l119_119241


namespace perfect_square_trinomial_l119_119973

variable (x y : ℝ)

theorem perfect_square_trinomial (a : ℝ) :
  (∃ b c : ℝ, 4 * x^2 - (a - 1) * x * y + 9 * y^2 = (b * x + c * y) ^ 2) ↔ 
  (a = 13 ∨ a = -11) := 
by
  sorry

end perfect_square_trinomial_l119_119973


namespace initial_condition_proof_move_to_1_proof_move_to_2_proof_recurrence_relation_proof_p_99_proof_p_100_proof_l119_119134

variable (p : ℕ → ℚ)

-- Given conditions
axiom initial_condition : p 0 = 1
axiom move_to_1 : p 1 = 1 / 2
axiom move_to_2 : p 2 = 3 / 4
axiom recurrence_relation : ∀ n : ℕ, 2 ≤ n → n ≤ 99 → p n - p (n - 1) = - 1 / 2 * (p (n - 1) - p (n - 2))
axiom p_99_cond : p 99 = 2 / 3 - 1 / (3 * 2^99)
axiom p_100_cond : p 100 = 1 / 3 + 1 / (3 * 2^99)

-- Proof that initial conditions are met
theorem initial_condition_proof : p 0 = 1 :=
sorry

theorem move_to_1_proof : p 1 = 1 / 2 :=
sorry

theorem move_to_2_proof : p 2 = 3 / 4 :=
sorry

-- Proof of the recurrence relation
theorem recurrence_relation_proof : ∀ n : ℕ, 2 ≤ n → n ≤ 99 → p n - p (n - 1) = - 1 / 2 * (p (n - 1) - p (n - 2)) :=
sorry

-- Proof of p_99
theorem p_99_proof : p 99 = 2 / 3 - 1 / (3 * 2^99) :=
sorry

-- Proof of p_100
theorem p_100_proof : p 100 = 1 / 3 + 1 / (3 * 2^99) :=
sorry

end initial_condition_proof_move_to_1_proof_move_to_2_proof_recurrence_relation_proof_p_99_proof_p_100_proof_l119_119134


namespace sum_of_reciprocals_of_roots_l119_119818

theorem sum_of_reciprocals_of_roots :
  ∀ (c d : ℝ),
  (6 * c^2 + 5 * c + 7 = 0) → 
  (6 * d^2 + 5 * d + 7 = 0) → 
  (c + d = -5 / 6) → 
  (c * d = 7 / 6) → 
  (1 / c + 1 / d = -5 / 7) :=
by
  intros c d h₁ h₂ h₃ h₄
  sorry

end sum_of_reciprocals_of_roots_l119_119818


namespace meaningful_domain_l119_119881

def is_meaningful (x : ℝ) : Prop :=
  (x - 1) ≠ 0

theorem meaningful_domain (x : ℝ) : is_meaningful x ↔ (x ≠ 1) :=
  sorry

end meaningful_domain_l119_119881


namespace solve_fraction_eq_l119_119246

theorem solve_fraction_eq (x : ℚ) (h : (x^2 + 3 * x + 4) / (x + 3) = x + 6) : x = -7 / 3 :=
sorry

end solve_fraction_eq_l119_119246


namespace max_ab_bc_ca_a2_div_bc_plus_b2_div_ca_plus_c2_div_ab_geq_1_over_2_l119_119802

variable (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
variable (h : a + b + c = 1)

theorem max_ab_bc_ca : ab + bc + ca ≤ 1 / 3 :=
by sorry

theorem a2_div_bc_plus_b2_div_ca_plus_c2_div_ab_geq_1_over_2 :
  (a^2 / (b + c)) + (b^2 / (c + a)) + (c^2 / (a + b)) ≥ 1 / 2 :=
by sorry

end max_ab_bc_ca_a2_div_bc_plus_b2_div_ca_plus_c2_div_ab_geq_1_over_2_l119_119802


namespace product_remainder_mod_5_l119_119763

theorem product_remainder_mod_5 :
  (1024 * 1455 * 1776 * 2018 * 2222) % 5 = 0 := 
sorry

end product_remainder_mod_5_l119_119763


namespace max_ab_condition_l119_119091

-- Define the circles and the tangency condition
def circle1 (a : ℝ) : Set (ℝ × ℝ) := {p | (p.1 - a)^2 + (p.2 + 2)^2 = 4}
def circle2 (b : ℝ) : Set (ℝ × ℝ) := {p | (p.1 + b)^2 + (p.2 + 2)^2 = 1}
def internally_tangent (a b : ℝ) : Prop := (a + b) ^ 2 = 1

-- Define the maximum value condition
def max_ab (a b : ℝ) : ℝ := a * b

-- Main theorem
theorem max_ab_condition {a b : ℝ} (h_tangent : internally_tangent a b) : max_ab a b ≤ 1 / 4 :=
by
  -- Proof steps are not necessary, so we use sorry to end the proof.
  sorry

end max_ab_condition_l119_119091


namespace largest_integral_value_l119_119745

theorem largest_integral_value (x : ℤ) : (1 / 3 : ℚ) < x / 5 ∧ x / 5 < 5 / 8 → x = 3 :=
by
  sorry

end largest_integral_value_l119_119745


namespace expected_pourings_correct_l119_119527

section
  /-- Four glasses are arranged in a row: the first and third contain orange juice, 
      the second and fourth are empty. Valya can take a full glass and pour its 
      contents into one of the two empty glasses each time. -/
  def initial_state : List Bool := [true, false, true, false]
  def target_state : List Bool := [false, true, false, true]

  /-- Define a function to calculate the expected number of pourings required to 
      reach the target state from the initial state given the probabilities of 
      transitions. -/
  noncomputable def expected_number_of_pourings (init : List Bool) (target : List Bool) : ℕ :=
    if init = initial_state ∧ target = target_state then 6 else 0

  /-- Prove that the expected number of pourings required to transition from 
      the initial state [true, false, true, false] to the target state [false, true, false, true] is 6. -/
  theorem expected_pourings_correct :
    expected_number_of_pourings initial_state target_state = 6 :=
  by
    -- Proof omitted
    sorry
end

end expected_pourings_correct_l119_119527


namespace statement_B_l119_119906

variable (Student : Type)
variable (nora : Student)
variable (correctly_answered_all_math_questions : Student → Prop)
variable (received_at_least_B : Student → Prop)

theorem statement_B :
  (∀ s : Student, correctly_answered_all_math_questions s → received_at_least_B s) →
  (¬ received_at_least_B nora → ∃ q : Student, ¬ correctly_answered_all_math_questions q) :=
by
  intros h hn
  sorry

end statement_B_l119_119906


namespace find_angle_A_min_perimeter_l119_119440

theorem find_angle_A (a b c : ℝ) (A B C : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) 
  (h₄ : a > 0 ∧ b > 0 ∧ c > 0) (h5 : b + c * Real.cos A = c + a * Real.cos C) 
  (hTriangle : A + B + C = Real.pi)
  (hSineLaw : Real.sin B = Real.sin C * Real.cos A + Real.sin A * Real.cos C) :
  A = Real.pi / 3 := 
by 
  sorry

theorem min_perimeter (a b c : ℝ) (A : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) 
  (h4 : a > 0 ∧ b > 0 ∧ c > 0 ∧ A = Real.pi / 3)
  (h_area : 1 / 2 * b * c * Real.sin A = Real.sqrt 3)
  (h_cosine : a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * Real.cos A) :
  a + b + c = 6 :=
by 
  sorry

end find_angle_A_min_perimeter_l119_119440


namespace negation_of_exists_l119_119857

theorem negation_of_exists : (¬ ∃ x : ℝ, x > 0 ∧ x^2 > 0) ↔ ∀ x : ℝ, x > 0 → x^2 ≤ 0 :=
by sorry

end negation_of_exists_l119_119857


namespace incorrect_sum_Sn_l119_119148

-- Define the geometric sequence sum formula
def Sn (a r : ℕ) (n : ℕ) : ℕ := a * (1 - r^n) / (1 - r)

-- Define the given values
def S1 : ℕ := 8
def S2 : ℕ := 20
def S3 : ℕ := 36
def S4 : ℕ := 65

-- The main proof statement
theorem incorrect_sum_Sn : 
  ∃ (a r : ℕ), 
  a = 8 ∧ 
  Sn a r 1 = S1 ∧ 
  Sn a r 2 = S2 ∧ 
  Sn a r 3 ≠ S3 ∧ 
  Sn a r 4 = S4 :=
by sorry

end incorrect_sum_Sn_l119_119148


namespace exists_face_with_fewer_than_six_sides_l119_119362

theorem exists_face_with_fewer_than_six_sides
  (N K M : ℕ) 
  (h_euler : N - K + M = 2)
  (h_vertices : M ≤ 2 * K / 3) : 
  ∃ n_i : ℕ, n_i < 6 :=
by
  sorry

end exists_face_with_fewer_than_six_sides_l119_119362


namespace goods_train_speed_l119_119856

def train_speed_km_per_hr (length_of_train length_of_platform time_to_cross : ℕ) : ℕ :=
  let total_distance := length_of_train + length_of_platform
  let speed_m_s := total_distance / time_to_cross
  speed_m_s * 36 / 10

-- Define the conditions given in the problem
def length_of_train : ℕ := 310
def length_of_platform : ℕ := 210
def time_to_cross : ℕ := 26

-- Define the target speed
def target_speed : ℕ := 72

-- The theorem proving the conclusion
theorem goods_train_speed :
  train_speed_km_per_hr length_of_train length_of_platform time_to_cross = target_speed := by
  sorry

end goods_train_speed_l119_119856


namespace total_and_per_suitcase_profit_l119_119867

theorem total_and_per_suitcase_profit
  (num_suitcases : ℕ)
  (purchase_price_per_suitcase : ℕ)
  (total_sales_revenue : ℕ)
  (total_profit : ℕ)
  (profit_per_suitcase : ℕ)
  (h_num_suitcases : num_suitcases = 60)
  (h_purchase_price : purchase_price_per_suitcase = 100)
  (h_total_sales : total_sales_revenue = 8100)
  (h_total_profit : total_profit = total_sales_revenue - num_suitcases * purchase_price_per_suitcase)
  (h_profit_per_suitcase : profit_per_suitcase = total_profit / num_suitcases) :
  total_profit = 2100 ∧ profit_per_suitcase = 35 := by
  sorry

end total_and_per_suitcase_profit_l119_119867


namespace total_goals_other_members_l119_119923

theorem total_goals_other_members (x y : ℕ) (h1 : y = (7 * x) / 15 - 18)
  (h2 : 1 / 3 * x + 1 / 5 * x + 18 + y = x)
  (h3 : ∀ n, 0 ≤ n ∧ n ≤ 3 → ¬(n * 8 > y))
  : y = 24 :=
by
  sorry

end total_goals_other_members_l119_119923


namespace find_a_l119_119833

noncomputable def f (x a : ℝ) : ℝ := x^2 - 2*x + a

theorem find_a :
  (∀ x : ℝ, 0 ≤ f x a) ∧ (∀ y : ℝ, ∃ x : ℝ, y = f x a) ↔ a = 1 := by
  sorry

end find_a_l119_119833


namespace convert_500_to_base5_l119_119211

def base10_to_base5 (n : ℕ) : ℕ :=
  -- A function to convert base 10 to base 5 would be defined here
  sorry

theorem convert_500_to_base5 : base10_to_base5 500 = 4000 := 
by 
  -- The actual proof would go here
  sorry

end convert_500_to_base5_l119_119211


namespace solve_for_n_l119_119861

theorem solve_for_n (n : ℤ) (h : n + (n + 1) + (n + 2) + (n + 3) = 26) : n = 5 :=
by
  sorry

end solve_for_n_l119_119861


namespace lcm_gcd_product_eq_product_12_15_l119_119086

theorem lcm_gcd_product_eq_product_12_15 :
  lcm 12 15 * gcd 12 15 = 12 * 15 :=
sorry

end lcm_gcd_product_eq_product_12_15_l119_119086


namespace number_of_baskets_l119_119208

def apples_per_basket : ℕ := 17
def total_apples : ℕ := 629

theorem number_of_baskets : total_apples / apples_per_basket = 37 :=
  by sorry

end number_of_baskets_l119_119208


namespace continuous_at_4_l119_119057

noncomputable def f (x : ℝ) := 3 * x^2 - 3

theorem continuous_at_4 : 
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, |x - 4| < δ → |f x - f 4| < ε :=
by
  sorry

end continuous_at_4_l119_119057


namespace systematic_sampling_interval_l119_119182

theorem systematic_sampling_interval 
  (N : ℕ) (n : ℕ) (hN : N = 630) (hn : n = 45) :
  N / n = 14 :=
by {
  sorry
}

end systematic_sampling_interval_l119_119182


namespace judge_guilty_cases_l119_119218

theorem judge_guilty_cases :
  let total_cases := 27
  let dismissed_cases := 3
  let remaining_cases := total_cases - dismissed_cases
  let innocent_cases := 3 * remaining_cases / 4
  let delayed_rulings := 2
  remaining_cases - innocent_cases - delayed_rulings = 4 :=
by
  let total_cases := 27
  let dismissed_cases := 3
  let remaining_cases := total_cases - dismissed_cases
  let innocent_cases := 3 * remaining_cases / 4
  let delayed_rulings := 2
  show remaining_cases - innocent_cases - delayed_rulings = 4
  sorry

end judge_guilty_cases_l119_119218


namespace cesaro_lupu_real_analysis_l119_119371

noncomputable def proof_problem (a b c x y z : ℝ) : Prop :=
  (0 < a ∧ a < 1) ∧ (0 < b ∧ b < 1) ∧ (0 < c ∧ c < 1) ∧
  (0 < x) ∧ (0 < y) ∧ (0 < z) ∧
  (a^x = b * c) ∧ (b^y = c * a) ∧ (c^z = a * b) →
  (1 / (2 + x) + 1 / (2 + y) + 1 / (2 + z) ≤ 3 / 4)

theorem cesaro_lupu_real_analysis (a b c x y z : ℝ) :
  proof_problem a b c x y z :=
by sorry

end cesaro_lupu_real_analysis_l119_119371


namespace percentage_of_life_in_accounting_jobs_l119_119042

-- Define the conditions
def years_as_accountant : ℕ := 25
def years_as_manager : ℕ := 15
def lifespan : ℕ := 80

-- Define the proof problem statement
theorem percentage_of_life_in_accounting_jobs :
  (years_as_accountant + years_as_manager) / lifespan * 100 = 50 := 
by sorry

end percentage_of_life_in_accounting_jobs_l119_119042


namespace minimize_cost_l119_119104

noncomputable def shipping_cost (x : ℝ) : ℝ := 5 * x
noncomputable def storage_cost (x : ℝ) : ℝ := 20 / x
noncomputable def total_cost (x : ℝ) : ℝ := shipping_cost x + storage_cost x

theorem minimize_cost : ∃ x : ℝ, x = 2 ∧ total_cost x = 20 :=
by
  use 2
  unfold total_cost
  unfold shipping_cost
  unfold storage_cost
  sorry

end minimize_cost_l119_119104


namespace evaluate_expression_l119_119828

def f (x : ℕ) : ℕ := 4 * x + 2
def g (x : ℕ) : ℕ := 3 * x + 4

theorem evaluate_expression : f (g (f 3)) = 186 := 
by 
  sorry

end evaluate_expression_l119_119828


namespace ratio_of_children_l119_119756

theorem ratio_of_children (C H : ℕ) 
  (hC1 : C / 8 = 16)
  (hC2 : C * (C / 8) = 512)
  (hH : H * 16 = 512) :
  H / C = 1 / 2 :=
by
  sorry

end ratio_of_children_l119_119756


namespace possible_slopes_l119_119159

theorem possible_slopes (k : ℝ) (H_pos : k > 0) :
  (∃ x1 x2 : ℤ, (x1 + x2 : ℝ) = k ∧ (x1 * x2 : ℝ) = -2020) ↔ 
  k = 81 ∨ k = 192 ∨ k = 399 ∨ k = 501 ∨ k = 1008 ∨ k = 2019 := 
by
  sorry

end possible_slopes_l119_119159


namespace meaningful_expression_range_l119_119637

theorem meaningful_expression_range (x : ℝ) (h1 : 3 * x + 2 ≥ 0) (h2 : x ≠ 0) : 
  x ∈ Set.Ico (-2 / 3) 0 ∪ Set.Ioi 0 := 
  sorry

end meaningful_expression_range_l119_119637


namespace green_paint_quarts_l119_119481

theorem green_paint_quarts (x : ℕ) (h : 5 * x = 3 * 15) : x = 9 := 
sorry

end green_paint_quarts_l119_119481


namespace simplify_expression_l119_119286

theorem simplify_expression (x : ℝ) : 3 * (5 - 2 * x) - 2 * (4 + 3 * x) = 7 - 12 * x := by
  sorry

end simplify_expression_l119_119286


namespace valid_vector_parameterizations_of_line_l119_119088

theorem valid_vector_parameterizations_of_line (t : ℝ) :
  (∃ t : ℝ, (∃ x y : ℝ, (x = 1 + t ∧ y = t ∧ y = x - 1)) ∨
            (∃ x y : ℝ, (x = -t ∧ y = -1 - t ∧ y = x - 1)) ∨
            (∃ x y : ℝ, (x = 2 + 0.5 * t ∧ y = 1 + 0.5 * t ∧ y = x - 1))) :=
by sorry

end valid_vector_parameterizations_of_line_l119_119088


namespace seventh_diagram_shaded_triangles_l119_119121

-- Define the factorial function
def fact : ℕ → ℕ
| 0 => 1
| (n + 1) => (n + 1) * fact n

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
| 0 => 1
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

-- The main theorem stating the relationship between the number of shaded sub-triangles and the factorial/Fibonacci sequence
theorem seventh_diagram_shaded_triangles :
  ∃ k : ℕ, (k : ℚ) = (fib 7 : ℚ) / (fact 7 : ℚ) ∧ k = 13 := sorry

end seventh_diagram_shaded_triangles_l119_119121


namespace TeresaTotalMarks_l119_119849

/-- Teresa's scores in various subjects as given conditions -/
def ScienceScore := 70
def MusicScore := 80
def SocialStudiesScore := 85
def PhysicsScore := 1 / 2 * MusicScore

/-- Total marks Teresa scored in all the subjects -/
def TotalMarks := ScienceScore + MusicScore + SocialStudiesScore + PhysicsScore

/-- Proof statement: The total marks scored by Teresa in all subjects is 275. -/
theorem TeresaTotalMarks : TotalMarks = 275 := by
  sorry

end TeresaTotalMarks_l119_119849


namespace arithmetic_sequence_sum_l119_119423

variable {a : ℕ → ℝ} -- The arithmetic sequence {a_n} represented by a function a : ℕ → ℝ

/-- Given that the sum of some terms of an arithmetic sequence is 25, prove the sum of other terms -/
theorem arithmetic_sequence_sum (h : a 3 + a 4 + a 5 + a 6 + a 7 = 25) : a 2 + a 8 = 10 := by
    sorry

end arithmetic_sequence_sum_l119_119423


namespace james_january_income_l119_119281

variable (January February March : ℝ)
variable (h1 : February = 2 * January)
variable (h2 : March = February - 2000)
variable (h3 : January + February + March = 18000)

theorem james_january_income : January = 4000 := by
  sorry

end james_january_income_l119_119281


namespace ice_cost_l119_119759

def people : Nat := 15
def ice_needed_per_person : Nat := 2
def pack_size : Nat := 10
def cost_per_pack : Nat := 3

theorem ice_cost : 
  let total_ice_needed := people * ice_needed_per_person
  let number_of_packs := total_ice_needed / pack_size
  total_ice_needed = 30 ∧ number_of_packs = 3 ∧ number_of_packs * cost_per_pack = 9 :=
by
  let total_ice_needed := people * ice_needed_per_person
  let number_of_packs := total_ice_needed / pack_size
  have h1 : total_ice_needed = 30 := by sorry
  have h2 : number_of_packs = 3 := by sorry
  have h3 : number_of_packs * cost_per_pack = 9 := by sorry
  exact And.intro h1 (And.intro h2 h3)

end ice_cost_l119_119759


namespace billy_points_difference_l119_119328

-- Condition Definitions
def billy_points : ℕ := 7
def friend_points : ℕ := 9

-- Theorem stating the problem and the solution
theorem billy_points_difference : friend_points - billy_points = 2 :=
by 
  sorry

end billy_points_difference_l119_119328


namespace farm_problem_l119_119253

theorem farm_problem (D C : ℕ) (h1 : D + C = 15) (h2 : 2 * D + 4 * C = 42) : C = 6 :=
sorry

end farm_problem_l119_119253


namespace odot_subtraction_l119_119634

-- Define the new operation
def odot (a b : ℚ) : ℚ := (a^3) / (b^2)

-- State the theorem
theorem odot_subtraction :
  ((odot (odot 2 4) 6) - (odot 2 (odot 4 6)) = -81 / 32) :=
by
  sorry

end odot_subtraction_l119_119634


namespace No_of_boxes_in_case_l119_119989

-- Define the conditions
def George_has_total_blocks : ℕ := 12
def blocks_per_box : ℕ := 6
def George_has_boxes : ℕ := George_has_total_blocks / blocks_per_box

-- The theorem to prove
theorem No_of_boxes_in_case : George_has_boxes = 2 :=
by
  sorry

end No_of_boxes_in_case_l119_119989


namespace internet_bill_is_100_l119_119168

theorem internet_bill_is_100 (initial_amount rent paycheck electricity_bill phone_bill final_amount internet_bill : ℝ)
  (h1 : initial_amount = 800)
  (h2 : rent = 450)
  (h3 : paycheck = 1500)
  (h4 : electricity_bill = 117)
  (h5 : phone_bill = 70)
  (h6 : final_amount = 1563)
  (h7 : initial_amount - rent + paycheck - electricity_bill - internet_bill - phone_bill = final_amount) :
  internet_bill = 100 :=
by
  sorry

end internet_bill_is_100_l119_119168


namespace pow_mod_equiv_l119_119526

theorem pow_mod_equiv (h : 5^500 ≡ 1 [MOD 1250]) : 5^15000 ≡ 1 [MOD 1250] := 
by 
  sorry

end pow_mod_equiv_l119_119526


namespace unique_solution_for_2_pow_m_plus_1_eq_n_square_l119_119449

theorem unique_solution_for_2_pow_m_plus_1_eq_n_square (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  2 ^ m + 1 = n ^ 2 → (m = 3 ∧ n = 3) :=
by {
  sorry
}

end unique_solution_for_2_pow_m_plus_1_eq_n_square_l119_119449


namespace munchausen_forest_l119_119127

theorem munchausen_forest (E B : ℕ) (h : B = 10 * E) : B > E := by sorry

end munchausen_forest_l119_119127


namespace Alyssa_puppies_l119_119000

theorem Alyssa_puppies (initial_puppies : ℕ) (given_puppies : ℕ)
  (h_initial : initial_puppies = 7) (h_given : given_puppies = 5) :
  initial_puppies - given_puppies = 2 :=
by
  sorry

end Alyssa_puppies_l119_119000


namespace negation_of_forall_ge_2_l119_119914

theorem negation_of_forall_ge_2 :
  (¬ ∀ x : ℝ, x ≥ 2) = (∃ x₀ : ℝ, x₀ < 2) :=
sorry

end negation_of_forall_ge_2_l119_119914


namespace linear_function_point_l119_119244

theorem linear_function_point (a b : ℝ) (h : b = 2 * a - 1) : 2 * a - b + 1 = 2 :=
by
  sorry

end linear_function_point_l119_119244


namespace max_sum_of_integer_pairs_on_circle_l119_119250

theorem max_sum_of_integer_pairs_on_circle : 
  ∃ (x y : ℤ), x^2 + y^2 = 169 ∧ ∀ (a b : ℤ), a^2 + b^2 = 169 → x + y ≥ a + b :=
sorry

end max_sum_of_integer_pairs_on_circle_l119_119250


namespace owner_overtakes_thief_l119_119334

theorem owner_overtakes_thief :
  let thief_speed_initial := 45 -- kmph
  let discovery_time := 0.5 -- hours
  let owner_speed := 50 -- kmph
  let mud_road_speed := 35 -- kmph
  let mud_road_distance := 30 -- km
  let speed_bumps_speed := 40 -- kmph
  let speed_bumps_distance := 5 -- km
  let traffic_speed := 30 -- kmph
  let head_start_distance := thief_speed_initial * discovery_time
  let mud_road_time := mud_road_distance / mud_road_speed
  let speed_bumps_time := speed_bumps_distance / speed_bumps_speed
  let total_distance_before_traffic := mud_road_distance + speed_bumps_distance
  let total_time_before_traffic := mud_road_time + speed_bumps_time
  let distance_owner_travelled := owner_speed * total_time_before_traffic
  head_start_distance + total_distance_before_traffic < distance_owner_travelled →
  discovery_time + total_time_before_traffic = 1.482 :=
by sorry


end owner_overtakes_thief_l119_119334


namespace find_denominator_x_l119_119035

noncomputable def sum_fractions : ℝ := 
    3.0035428163476343

noncomputable def fraction1 (x : ℝ) : ℝ :=
    2007 / x

noncomputable def fraction2 : ℝ :=
    8001 / 5998

noncomputable def fraction3 : ℝ :=
    2001 / 3999

-- Problem statement in Lean
theorem find_denominator_x (x : ℝ) :
  sum_fractions = fraction1 x + fraction2 + fraction3 ↔ x = 1717 :=
by sorry

end find_denominator_x_l119_119035


namespace like_terms_mn_l119_119586

theorem like_terms_mn (m n : ℕ) (h1 : -2 * x^m * y^2 = 2 * x^3 * y^n) : m * n = 6 :=
by {
  -- Add the statements transforming the assumptions into intermediate steps
  sorry
}

end like_terms_mn_l119_119586


namespace factor_expression_l119_119948

theorem factor_expression (x y z : ℝ) :
  ((x^3 - y^3)^3 + (y^3 - z^3)^3 + (z^3 - x^3)^3) / 
  ((x - y)^3 + (y - z)^3 + (z - x)^3) = 
  ((x^2 + x * y + y^2) * (y^2 + y * z + z^2) * (z^2 + z * x + x^2)) :=
by {
  sorry  -- The proof goes here
}

end factor_expression_l119_119948


namespace sum_of_interiors_l119_119142

theorem sum_of_interiors (n : ℕ) (h : 180 * (n - 2) = 1620) : 180 * ((n + 3) - 2) = 2160 :=
by sorry

end sum_of_interiors_l119_119142


namespace quadratic_equation_unique_solution_l119_119273

theorem quadratic_equation_unique_solution 
  (a c : ℝ) (h1 : ∃ x : ℝ, a * x^2 + 8 * x + c = 0)
  (h2 : a + c = 10)
  (h3 : a < c) :
  (a, c) = (2, 8) := 
sorry

end quadratic_equation_unique_solution_l119_119273


namespace field_area_is_36_square_meters_l119_119261

theorem field_area_is_36_square_meters (side_length : ℕ) (h : side_length = 6) : side_length * side_length = 36 :=
by
  sorry

end field_area_is_36_square_meters_l119_119261


namespace gnuff_tutoring_minutes_l119_119653

theorem gnuff_tutoring_minutes 
  (flat_rate : ℕ) 
  (rate_per_minute : ℕ) 
  (total_paid : ℕ) :
  flat_rate = 20 → 
  rate_per_minute = 7 →
  total_paid = 146 → 
  ∃ minutes : ℕ, minutes = 18 ∧ flat_rate + rate_per_minute * minutes = total_paid := 
by
  -- Assume the necessary steps here
  sorry

end gnuff_tutoring_minutes_l119_119653


namespace find_greatest_natural_number_l119_119369

-- Definitions for terms used in the conditions

def sum_of_squares (m : ℕ) : ℕ :=
  (m * (m + 1) * (2 * m + 1)) / 6

def is_perfect_square (a : ℕ) : Prop :=
  ∃ b : ℕ, b * b = a

-- Conditions defined in Lean terms
def condition1 (n : ℕ) : Prop := n ≤ 2010

def condition2 (n : ℕ) : Prop := 
  let sum1 := sum_of_squares n
  let sum2 := sum_of_squares (2 * n) - sum_of_squares n
  is_perfect_square (sum1 * sum2)

-- Main theorem statement
theorem find_greatest_natural_number : ∃ n, n ≤ 2010 ∧ condition2 n ∧ ∀ m, m ≤ 2010 ∧ condition2 m → m ≤ n := 
by 
  sorry

end find_greatest_natural_number_l119_119369


namespace solve_for_C_l119_119704

-- Given constants and assumptions
def SumOfDigitsFirst (A B : ℕ) := 8 + 4 + A + 5 + 3 + B + 2 + 1
def SumOfDigitsSecond (A B C : ℕ) := 5 + 2 + 7 + A + B + 6 + 0 + C

theorem solve_for_C (A B C : ℕ) 
  (h1 : (SumOfDigitsFirst A B % 9) = 0)
  (h2 : (SumOfDigitsSecond A B C % 9) = 0) 
  : C = 3 :=
sorry

end solve_for_C_l119_119704


namespace solution_set_of_inequality_l119_119870

theorem solution_set_of_inequality :
  { x : ℝ | (x - 3) * (x + 2) < 0 } = { x : ℝ | -2 < x ∧ x < 3 } :=
by
  sorry

end solution_set_of_inequality_l119_119870


namespace smallest_k_exists_l119_119710

open Nat

theorem smallest_k_exists (n m k : ℕ) (hn : n > 0) (hm : 0 < m ∧ m ≤ 5) (hk : k % 3 = 0) :
  (64^k + 32^m > 4^(16 + n^2)) ↔ k = 6 :=
by
  sorry

end smallest_k_exists_l119_119710


namespace expected_messages_xiaoli_l119_119815

noncomputable def expected_greeting_messages (probs : List ℝ) (counts : List ℕ) : ℝ :=
  List.sum (List.zipWith (λ p c => p * c) probs counts)

theorem expected_messages_xiaoli :
  expected_greeting_messages [1, 0.8, 0.5, 0] [8, 15, 14, 3] = 27 :=
by
  -- The proof will use the expected value formula
  sorry

end expected_messages_xiaoli_l119_119815


namespace smallest_points_to_exceed_mean_l119_119108

theorem smallest_points_to_exceed_mean (X y : ℕ) (h_scores : 24 + 17 + 25 = 66) 
  (h_mean_9_gt_mean_6 : X / 6 < (X + 66) / 9) (h_mean_10_gt_22 : (X + 66 + y) / 10 > 22) 
  : y ≥ 24 := by
  sorry

end smallest_points_to_exceed_mean_l119_119108


namespace total_earmuffs_l119_119515

theorem total_earmuffs {a b c : ℕ} (h1 : a = 1346) (h2 : b = 6444) (h3 : c = a + b) : c = 7790 := by
  sorry

end total_earmuffs_l119_119515


namespace shortest_chord_through_point_l119_119905

theorem shortest_chord_through_point
  (correct_length : ℝ)
  (h1 : correct_length = 2 * Real.sqrt 2)
  (circle_eq : ∀ (x y : ℝ), (x - 2)^2 + (y - 2)^2 = 4)
  (passes_point : ∀ (p : ℝ × ℝ), p = (3, 1)) :
  correct_length = 2 * Real.sqrt 2 :=
by {
  -- the proof steps would go here
  sorry
}

end shortest_chord_through_point_l119_119905


namespace graph_shift_cos_function_l119_119999

theorem graph_shift_cos_function (f : ℝ → ℝ) (φ : ℝ) :
  (∀ x, f x = 2 * Real.cos (π * x / 3 + φ)) ∧ 
  (∃ x, f x = 0 ∧ x = 2) ∧ 
  (f 1 > f 3) →
  (∀ x, f x = 2 * Real.cos (π * (x - 1/2) / 3)) :=
by
  sorry

end graph_shift_cos_function_l119_119999


namespace tv_purchase_price_correct_l119_119202

theorem tv_purchase_price_correct (x : ℝ) (h : (1.4 * x * 0.8 - x) = 270) : x = 2250 :=
by
  sorry

end tv_purchase_price_correct_l119_119202


namespace range_of_a_for_maximum_l119_119611

variable {f : ℝ → ℝ}
variable {a : ℝ}

theorem range_of_a_for_maximum (h : ∀ x, deriv f x = a * (x + 1) * (x - a))
  (h_max : ∀ x, f x ≤ f a → x = a) : -1 < a ∧ a < 0 :=
sorry

end range_of_a_for_maximum_l119_119611


namespace vertex_on_x_axis_l119_119744

theorem vertex_on_x_axis (c : ℝ) : (∃ (h : ℝ), (h, 0) = ((-(-8) / (2 * 1)), c - (-8)^2 / (4 * 1))) → c = 16 :=
by
  sorry

end vertex_on_x_axis_l119_119744


namespace sufficient_condition_increasing_l119_119937

noncomputable def f (x a : ℝ) : ℝ := x^2 + 2 * a * x + 1

theorem sufficient_condition_increasing (a : ℝ) :
  (∀ x y : ℝ, 1 < x → x < y → (f x a ≤ f y a)) → a = -1 := sorry

end sufficient_condition_increasing_l119_119937


namespace collinear_vectors_x_eq_neg_two_l119_119319

theorem collinear_vectors_x_eq_neg_two (x : ℝ) (a b : ℝ×ℝ) :
  a = (1, 2) → b = (x, -4) → a.1 * b.2 = a.2 * b.1 → x = -2 :=
by
  intro ha hb hc
  sorry

end collinear_vectors_x_eq_neg_two_l119_119319


namespace problem_1_part1_problem_1_part2_problem_2_l119_119994

open Real

noncomputable def f (x : ℝ) : ℝ := sqrt 3 * sin (2 * x) + 2 + 2 * cos (x) ^ 2

theorem problem_1_part1 : (∃ T > 0, ∀ x, f (x + T) = f x) := sorry

theorem problem_1_part2 : (∀ k : ℤ, ∀ x ∈ Set.Icc (k * π - π / 3) (k * π + π / 6), ∀ y ∈ Set.Icc (k * π - π / 3) (k * π + π / 6), x < y → f x > f y) := sorry

noncomputable def S_triangle (A B C : ℝ) (a b c : ℝ) : ℝ := 1 / 2 * b * c * sin A

theorem problem_2 :
  ∀ (A B C a b c : ℝ), f A = 4 → b = 1 → S_triangle A B C a b c = sqrt 3 / 2 →
    a^2 = b^2 + c^2 - 2 * b * c * cos A → a = sqrt 3 := sorry

end problem_1_part1_problem_1_part2_problem_2_l119_119994


namespace all_non_positive_l119_119628

theorem all_non_positive (n : ℕ) (a : ℕ → ℤ) 
  (h₀ : a 0 = 0) 
  (hₙ : a n = 0) 
  (ineq : ∀ k, 1 ≤ k ∧ k ≤ n - 1 → a (k - 1) - 2 * a k + a (k + 1) ≥ 0) : ∀ k, a k ≤ 0 :=
by 
  sorry

end all_non_positive_l119_119628


namespace base7_to_base10_321_is_162_l119_119094

-- Define the conversion process from a base-7 number to base-10
def convert_base7_to_base10 (n: ℕ) : ℕ :=
  3 * 7^2 + 2 * 7^1 + 1 * 7^0

theorem base7_to_base10_321_is_162 :
  convert_base7_to_base10 321 = 162 :=
by
  sorry

end base7_to_base10_321_is_162_l119_119094


namespace positive_integers_n_l119_119701

theorem positive_integers_n (n a b : ℕ) (h1 : 2 < n) (h2 : n = a ^ 3 + b ^ 3) 
  (h3 : ∀ d, d > 1 ∧ d ∣ n → a ≤ d) (h4 : b ∣ n) : n = 16 ∨ n = 72 ∨ n = 520 :=
sorry

end positive_integers_n_l119_119701


namespace largest_possible_product_is_3886_l119_119728

theorem largest_possible_product_is_3886 :
  ∃ a b c d : ℕ, 5 ≤ a ∧ a ≤ 8 ∧
               5 ≤ b ∧ b ≤ 8 ∧
               5 ≤ c ∧ c ≤ 8 ∧
               5 ≤ d ∧ d ≤ 8 ∧
               a ≠ b ∧ a ≠ c ∧ a ≠ d ∧
               b ≠ c ∧ b ≠ d ∧
               c ≠ d ∧
               (max ((10 * a + b) * (10 * c + d))
                    ((10 * c + b) * (10 * a + d))) = 3886 :=
sorry

end largest_possible_product_is_3886_l119_119728


namespace evaluate_expression_l119_119717

theorem evaluate_expression :
  (Int.floor ((Int.ceil ((11/5:ℚ)^2)) * (19/3:ℚ))) = 31 :=
by
  sorry

end evaluate_expression_l119_119717


namespace range_of_m_l119_119044

theorem range_of_m (m : ℝ) :
  (∃ (x1 x2 : ℝ), (2*x1^2 - 2*x1 + 3*m - 1 = 0 ∧ 2*x2^2 - 2*x2 + 3*m - 1 = 0) ∧ (x1 * x2 > x1 + x2 - 4)) →
  -5/3 < m ∧ m ≤ 1/2 :=
by
  sorry

end range_of_m_l119_119044


namespace larger_solution_of_quadratic_equation_l119_119249

open Nat

theorem larger_solution_of_quadratic_equation :
  ∃! x : ℝ, x * x - 13 * x + 36 = 0 ∧ ∀ y : ℝ, y * y - 13 * y + 36 = 0 → x ≥ y :=
by {
  sorry
}

end larger_solution_of_quadratic_equation_l119_119249


namespace tan_domain_l119_119984

open Real

theorem tan_domain (k : ℤ) (x : ℝ) :
  (∀ k : ℤ, x ≠ (k * π / 2) + (3 * π / 8)) ↔ 
  (∀ k : ℤ, 2 * x - π / 4 ≠ k * π + π / 2) := sorry

end tan_domain_l119_119984


namespace dartboard_central_angle_l119_119604

theorem dartboard_central_angle (A : ℝ) (x : ℝ) (P : ℝ) (h1 : P = 1 / 4) 
    (h2 : A > 0) : (x / 360 = 1 / 4) -> x = 90 :=
by
  sorry

end dartboard_central_angle_l119_119604


namespace japanese_turtle_crane_problem_l119_119284

theorem japanese_turtle_crane_problem (x y : ℕ) (h1 : x + y = 35) (h2 : 2 * x + 4 * y = 94) : x + y = 35 ∧ 2 * x + 4 * y = 94 :=
by
  sorry

end japanese_turtle_crane_problem_l119_119284


namespace division_of_5_parts_division_of_7_parts_division_of_8_parts_l119_119499

-- Problem 1: Primary Division of Square into 5 Equal Parts
theorem division_of_5_parts (x : ℝ) (h : x^2 = 1 / 5) : x = Real.sqrt (1 / 5) :=
sorry

-- Problem 2: Primary Division of Square into 7 Equal Parts
theorem division_of_7_parts (x : ℝ) (hx : 196 * x^3 - 294 * x^2 + 128 * x - 15 = 0) : 
  x = (7 + Real.sqrt 19) / 14 :=
sorry

-- Problem 3: Primary Division of Square into 8 Equal Parts
theorem division_of_8_parts (x : ℝ) (hx : 6 * x^2 - 6 * x + 1 = 0) : 
  x = (3 + Real.sqrt 3) / 6 :=
sorry

end division_of_5_parts_division_of_7_parts_division_of_8_parts_l119_119499


namespace tyler_cd_purchase_l119_119357

theorem tyler_cd_purchase :
  ∀ (initial_cds : ℕ) (given_away_fraction : ℝ) (final_cds : ℕ) (bought_cds : ℕ),
    initial_cds = 21 →
    given_away_fraction = 1 / 3 →
    final_cds = 22 →
    bought_cds = 8 →
    final_cds = initial_cds - initial_cds * given_away_fraction + bought_cds :=
by
  intros
  sorry

end tyler_cd_purchase_l119_119357


namespace points_per_enemy_l119_119409

-- Definitions: total enemies, enemies not destroyed, points earned
def total_enemies : ℕ := 11
def enemies_not_destroyed : ℕ := 3
def points_earned : ℕ := 72

-- To prove: points per enemy
theorem points_per_enemy : points_earned / (total_enemies - enemies_not_destroyed) = 9 := 
by
  sorry

end points_per_enemy_l119_119409


namespace solution_is_correct_l119_119799

noncomputable def satisfies_inequality (x y : ℝ) : Prop := 
  x + 3 * y + 14 ≤ 0

noncomputable def satisfies_equation (x y : ℝ) : Prop := 
  x^4 + 2 * x^2 * y^2 + y^4 + 64 - 20 * x^2 - 20 * y^2 = 8 * x * y

theorem solution_is_correct : satisfies_inequality (-2) (-4) ∧ satisfies_equation (-2) (-4) :=
  by sorry

end solution_is_correct_l119_119799


namespace sum_of_fifth_powers_l119_119131

theorem sum_of_fifth_powers (a b c d : ℝ) (h1 : a + b = c + d) (h2 : a^3 + b^3 = c^3 + d^3) : a^5 + b^5 = c^5 + d^5 := sorry

end sum_of_fifth_powers_l119_119131


namespace product_of_two_large_integers_l119_119801

theorem product_of_two_large_integers :
  ∃ a b : ℕ, a > 2009^182 ∧ b > 2009^182 ∧ 3^2008 + 4^2009 = a * b :=
by { sorry }

end product_of_two_large_integers_l119_119801


namespace first_term_of_geometric_series_l119_119049

variable (a r : ℝ)
variable (h1 : a / (1 - r) = 20)
variable (h2 : a^2 / (1 - r^2) = 80)

theorem first_term_of_geometric_series (a r : ℝ) (h1 : a / (1 - r) = 20) (h2 : a^2 / (1 - r^2) = 80) : 
  a = 20 / 3 :=
  sorry

end first_term_of_geometric_series_l119_119049


namespace quadratic_pos_implies_a_gt_1_l119_119506

theorem quadratic_pos_implies_a_gt_1 {a : ℝ} :
  (∀ x : ℝ, x^2 + 2 * x + a > 0) → a > 1 :=
by
  sorry

end quadratic_pos_implies_a_gt_1_l119_119506


namespace solve_for_x_l119_119446

theorem solve_for_x (x : ℝ) : 0.05 * x + 0.07 * (30 + x) = 15.4 → x = 110.8333333 :=
by
  intro h
  sorry

end solve_for_x_l119_119446


namespace consecutive_vertices_product_l119_119378

theorem consecutive_vertices_product (n : ℕ) (hn : n = 90) :
  ∃ (i : ℕ), 1 ≤ i ∧ i ≤ n ∧ ((i * (i % n + 1)) ≥ 2014) := 
sorry

end consecutive_vertices_product_l119_119378


namespace distinct_positive_and_conditions_l119_119183

theorem distinct_positive_and_conditions (a b : ℕ) (h_distinct: a ≠ b) (h_pos1: 0 < a) (h_pos2: 0 < b) (h_eq: a^3 - b^3 = a^2 - b^2) : 
  ∃ (c : ℕ), c = 9 * a * b ∧ (c = 1 ∨ c = 2 ∨ c = 3) :=
by
  sorry

end distinct_positive_and_conditions_l119_119183


namespace intersection_of_A_and_B_l119_119266

def A : Set ℤ := {1, 2, -3}
def B : Set ℤ := {1, -4, 5}

theorem intersection_of_A_and_B : A ∩ B = {1} :=
by sorry

end intersection_of_A_and_B_l119_119266


namespace solve_equation_l119_119531

def problem_statement : Prop :=
  ∃ x : ℚ, (3 - x) / (x + 2) + (3 * x - 6) / (3 - x) = 2 ∧ x = -7 / 6

theorem solve_equation : problem_statement :=
by {
  sorry
}

end solve_equation_l119_119531


namespace find_m_range_l119_119509

theorem find_m_range
  (m y1 y2 y0 x0 : ℝ)
  (a c : ℝ) (h1 : a ≠ 0)
  (h2 : x0 = -2)
  (h3 : ∀ x, (x, ax^2 + 4*a*x + c) = (m, y1) ∨ (x, ax^2 + 4*a*x + c) = (m + 2, y2) ∨ (x, ax^2 + 4*a*x + c) = (x0, y0))
  (h4 : y0 ≥ y2) (h5 : y2 > y1) :
  m < -3 :=
sorry

end find_m_range_l119_119509


namespace sum_of_first_150_remainder_l119_119544

theorem sum_of_first_150_remainder :
  let n := 150
  let sum := n * (n + 1) / 2
  sum % 5600 = 125 :=
by
  sorry

end sum_of_first_150_remainder_l119_119544


namespace quadratic_function_difference_zero_l119_119129

theorem quadratic_function_difference_zero
  (a b c x1 x2 x3 x4 x5 p q : ℝ)
  (h1 : a ≠ 0)
  (h2 : a * x1^2 + b * x1 + c = 5)
  (h3 : a * (x2 + x3 + x4 + x5)^2 + b * (x2 + x3 + x4 + x5) + c = 5)
  (h4 : x1 ≠ x2 + x3 + x4 + x5)
  (h5 : a * (x1 + x2)^2 + b * (x1 + x2) + c = p)
  (h6 : a * (x3 + x4 + x5)^2 + b * (x3 + x4 + x5) + c = q) :
  p - q = 0 := 
sorry

end quadratic_function_difference_zero_l119_119129


namespace xyz_squared_eq_one_l119_119310

theorem xyz_squared_eq_one (x y z : ℝ) (h_distinct : x ≠ y ∧ y ≠ z ∧ z ≠ x)
    (h_eq : ∃ k, x + (1 / y) = k ∧ y + (1 / z) = k ∧ z + (1 / x) = k) : 
    x^2 * y^2 * z^2 = 1 := 
  sorry

end xyz_squared_eq_one_l119_119310


namespace right_triangle_k_value_l119_119852

theorem right_triangle_k_value (x : ℝ) (k : ℝ) (s : ℝ) 
(h_triangle : 3*x + 4*x + 5*x = k * (1/2 * 3*x * 4*x)) 
(h_square : s = 10) (h_eq_apothems : 4*x = s/2) : 
k = 8 / 5 :=
by {
  sorry
}

end right_triangle_k_value_l119_119852


namespace geometric_sequence_product_l119_119682

variable (a : ℕ → ℝ)

def is_geometric_seq (a : ℕ → ℝ) := ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_product (h_geom : is_geometric_seq a) (h_a6 : a 6 = 3) :
  a 3 * a 4 * a 5 * a 6 * a 7 * a 8 * a 9 = 2187 := by
  sorry

end geometric_sequence_product_l119_119682


namespace avg_growth_rate_l119_119971

theorem avg_growth_rate {a p q x : ℝ} (h_eq : (1 + p) * (1 + q) = (1 + x) ^ 2) : 
  x ≤ (p + q) / 2 := 
by
  sorry

end avg_growth_rate_l119_119971


namespace total_sales_15_days_l119_119894

def edgar_sales (n : ℕ) : ℕ := 3 * n - 1

def clara_sales (n : ℕ) : ℕ := 4 * n

def edgar_total_sales (d : ℕ) : ℕ := (d * (2 + (d * 3 - 1))) / 2

def clara_total_sales (d : ℕ) : ℕ := (d * (4 + (d * 4))) / 2

def total_sales (d : ℕ) : ℕ := edgar_total_sales d + clara_total_sales d

theorem total_sales_15_days : total_sales 15 = 810 :=
by
  sorry

end total_sales_15_days_l119_119894


namespace carrie_hours_per_week_l119_119581

variable (H : ℕ)

def carrie_hourly_wage : ℕ := 8
def cost_of_bike : ℕ := 400
def amount_left_over : ℕ := 720
def weeks_worked : ℕ := 4
def total_earnings : ℕ := cost_of_bike + amount_left_over

theorem carrie_hours_per_week :
  (weeks_worked * H * carrie_hourly_wage = total_earnings) →
  H = 35 := by
  sorry

end carrie_hours_per_week_l119_119581


namespace year_2023_ad_is_written_as_positive_2023_l119_119770

theorem year_2023_ad_is_written_as_positive_2023 :
  (∀ (year : Int), year = -500 → year = -500) → -- This represents the given condition that year 500 BC is -500
  (∀ (year : Int), year > 0) → -- This represents the condition that AD years are postive
  2023 = 2023 := -- The problem conclusion

by
  intros
  trivial -- The solution is quite trivial due to the conditions.

end year_2023_ad_is_written_as_positive_2023_l119_119770


namespace smallest_z_l119_119162

theorem smallest_z 
  (x y z : ℕ) 
  (h_pos_x : x > 0) 
  (h_pos_y : y > 0) 
  (h1 : x + y = z) 
  (h2 : x * y < z^2) 
  (ineq : (27^z) * (5^x) > (3^24) * (2^y)) :
  z = 10 :=
by
  sorry

end smallest_z_l119_119162


namespace cellini_inscription_l119_119520

noncomputable def famous_master_engravings (x: Type) : String :=
  "Эту шкатулку изготовил сын Челлини"

theorem cellini_inscription (x: Type) (created_by_cellini : x) :
  famous_master_engravings x = "Эту шкатулку изготовил сын Челлини" :=
by
  sorry

end cellini_inscription_l119_119520


namespace harrison_croissant_expenditure_l119_119122

-- Define the conditions
def cost_regular_croissant : ℝ := 3.50
def cost_almond_croissant : ℝ := 5.50
def weeks_in_year : ℕ := 52

-- Define the total cost of croissants in a year
def total_cost (cost_regular cost_almond : ℝ) (weeks : ℕ) : ℝ :=
  (weeks * cost_regular) + (weeks * cost_almond)

-- State the proof problem
theorem harrison_croissant_expenditure :
  total_cost cost_regular_croissant cost_almond_croissant weeks_in_year = 468.00 :=
by
  sorry

end harrison_croissant_expenditure_l119_119122


namespace total_math_and_biology_homework_l119_119962

-- Definitions
def math_homework_pages : ℕ := 8
def biology_homework_pages : ℕ := 3

-- Theorem stating the problem to prove
theorem total_math_and_biology_homework :
  math_homework_pages + biology_homework_pages = 11 :=
by
  sorry

end total_math_and_biology_homework_l119_119962


namespace min_value_fraction_sum_l119_119494

theorem min_value_fraction_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h_eq : 1 = 2 * a + b) :
  (1 / a + 1 / b) ≥ 3 + 2 * Real.sqrt 2 :=
sorry

end min_value_fraction_sum_l119_119494


namespace area_of_triangle_ABC_l119_119027

def point := (ℝ × ℝ)

def A : point := (0, 0)
def B : point := (1424233, 2848467)
def C : point := (1424234, 2848469)

def triangle_area (A B C : point) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem area_of_triangle_ABC : triangle_area A B C = 0.50 := by
  sorry

end area_of_triangle_ABC_l119_119027


namespace odd_square_mod_eight_l119_119789

theorem odd_square_mod_eight (k : ℤ) : ((2 * k + 1) ^ 2) % 8 = 1 := 
sorry

end odd_square_mod_eight_l119_119789


namespace factorize_expression_l119_119819

variable {X M N : ℕ}

theorem factorize_expression (x m n : ℕ) : x * m - x * n = x * (m - n) :=
sorry

end factorize_expression_l119_119819


namespace goshawk_nature_reserve_l119_119668

-- Define the problem statement and conditions
def percent_hawks (H W K : ℝ) : Prop :=
  ∃ H W K : ℝ,
    -- Condition 1: 35% of the birds are neither hawks, paddyfield-warblers, nor kingfishers
    1 - (H + W + K) = 0.35 ∧
    -- Condition 2: 40% of the non-hawks are paddyfield-warblers
    W = 0.40 * (1 - H) ∧
    -- Condition 3: There are 25% as many kingfishers as paddyfield-warblers
    K = 0.25 * W ∧
    -- Given all conditions, calculate the percentage of hawks
    H = 0.65

theorem goshawk_nature_reserve :
  ∃ H W K : ℝ,
    1 - (H + W + K) = 0.35 ∧
    W = 0.40 * (1 - H) ∧
    K = 0.25 * W ∧
    H = 0.65 := by
    -- Proof is omitted
    sorry

end goshawk_nature_reserve_l119_119668


namespace solve_abs_eq_l119_119268

theorem solve_abs_eq (x : ℝ) (h : |x - 1| = 2 * x) : x = 1 / 3 :=
by
  sorry

end solve_abs_eq_l119_119268


namespace sum_of_other_endpoint_coordinates_l119_119488

theorem sum_of_other_endpoint_coordinates 
  (A B O : ℝ × ℝ)
  (hA : A = (6, -2)) 
  (hO : O = (3, 5)) 
  (midpoint_formula : (A.1 + B.1) / 2 = O.1 ∧ (A.2 + B.2) / 2 = O.2):
  (B.1 + B.2) = 12 :=
by
  sorry

end sum_of_other_endpoint_coordinates_l119_119488


namespace additional_matches_l119_119054

theorem additional_matches 
  (avg_runs_first_25 : ℕ → ℚ) 
  (avg_runs_additional : ℕ → ℚ) 
  (avg_runs_all : ℚ) 
  (total_matches_first_25 : ℕ) 
  (total_matches_all : ℕ) 
  (total_runs_first_25 : ℚ) 
  (total_runs_all : ℚ) 
  (x : ℕ)
  (h1 : avg_runs_first_25 25 = 45)
  (h2 : avg_runs_additional x = 15)
  (h3 : avg_runs_all = 38.4375)
  (h4 : total_matches_first_25 = 25)
  (h5 : total_matches_all = 32)
  (h6 : total_runs_first_25 = avg_runs_first_25 25 * 25)
  (h7 : total_runs_all = avg_runs_all * 32)
  (h8 : total_runs_first_25 + avg_runs_additional x * x = total_runs_all) :
  x = 7 :=
sorry

end additional_matches_l119_119054


namespace midpoint_P_AB_l119_119080

structure Point := (x : ℝ) (y : ℝ)

def segment_midpoint (P A B : Point) : Prop := P.x = (A.x + B.x) / 2 ∧ P.y = (A.y + B.y) / 2

variables {A D C E P B : Point}
variables (h1 : A.x = D.x ∧ A.y = D.y)
variables (h2 : D.x = C.x ∧ D.y = C.y)
variables (h3 : D.x = P.x ∧ D.y = P.y ∧ P.x = E.x ∧ P.y = E.y)
variables (h4 : B.x = E.x ∧ B.y = E.y)
variables (h5 : A.x = C.x ∧ A.y = C.y)
variables (angle_ADC : ∀ x y : ℝ, (x - A.x)^2 + (y - A.y)^2 = (x - D.x)^2 + (y - D.y)^2 → (x - C.x)^2 + (y - C.y)^2 = (x - D.x)^2 + (y - D.y)^2)
variables (angle_DPE : ∀ x y : ℝ, (x - D.x)^2 + (y - P.y)^2 = (x - P.x)^2 + (y - E.y)^2 → (x - E.x)^2 + (y - E.y)^2 = (x - P.x)^2 + (y - E.y)^2)
variables (angle_BEC : ∀ x y : ℝ, (x - B.x)^2 + (y - E.y)^2 = (x - E.x)^2 + (y - C.y)^2 → (x - B.x)^2 + (y - C.y)^2 = (x - E.x)^2 + (y - C.y)^2)

theorem midpoint_P_AB : segment_midpoint P A B := 
sorry

end midpoint_P_AB_l119_119080


namespace tangent_line_at_P_l119_119209

def tangent_line_eq (x y : ℝ) : ℝ := x - 2 * y + 1

theorem tangent_line_at_P (x y : ℝ) (h : x ^ 2 + y ^ 2 - 4 * x + 2 * y = 0 ∧ (x, y) = (1, 1)) :
    tangent_line_eq x y = 0 := 
sorry

end tangent_line_at_P_l119_119209


namespace prove_distance_uphill_l119_119305

noncomputable def distance_uphill := 
  let flat_speed := 20
  let uphill_speed := 12
  let extra_flat_distance := 30
  let uphill_time (D : ℝ) := D / uphill_speed
  let flat_time (D : ℝ) := (D + extra_flat_distance) / flat_speed
  ∃ D : ℝ, uphill_time D = flat_time D ∧ D = 45

theorem prove_distance_uphill : distance_uphill :=
sorry

end prove_distance_uphill_l119_119305


namespace solution_inequality_l119_119749

open Set

theorem solution_inequality (x : ℝ) : (x > 3 ∨ x < -3) ↔ (x > 9 / x) := by
  sorry

end solution_inequality_l119_119749


namespace roots_of_equations_l119_119535

theorem roots_of_equations (a : ℝ) :
  (∃ x : ℝ, x^2 + 4 * a * x - 4 * a + 3 = 0) ∨
  (∃ x : ℝ, x^2 + (a - 1) * x + a^2 = 0) ∨
  (∃ x : ℝ, x^2 + 2 * a * x - 2 * a = 0) ↔ 
  a ≤ -3 / 2 ∨ a ≥ -1 :=
sorry

end roots_of_equations_l119_119535


namespace hyperbola_condition_l119_119245

theorem hyperbola_condition (m : ℝ) : 
  (∀ x y : ℝ, (m-2) * (m+3) < 0 → (x^2) / (m-2) + (y^2) / (m+3) = 1) ↔ -3 < m ∧ m < 2 :=
by
  sorry

end hyperbola_condition_l119_119245


namespace inverse_function_l119_119100

noncomputable def f (x : ℝ) := 3 - 7 * x + x^2

noncomputable def g (x : ℝ) := (7 + Real.sqrt (37 + 4 * x)) / 2

theorem inverse_function :
  ∀ x : ℝ, f (g x) = x :=
by
  intros x
  sorry

end inverse_function_l119_119100


namespace A_and_B_finish_together_in_11_25_days_l119_119016

theorem A_and_B_finish_together_in_11_25_days (A_rate B_rate : ℝ)
    (hA : A_rate = 1/18) (hB : B_rate = 1/30) :
    1 / (A_rate + B_rate) = 11.25 := by
  sorry

end A_and_B_finish_together_in_11_25_days_l119_119016


namespace one_angle_greater_135_l119_119201

noncomputable def angles_sum_not_form_triangle (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : Prop :=
  ∀ (A B C : ℝ), 
   (A < a + b ∧ A < a + c ∧ A < b + c) →
  (B < a + b ∧ B < a + c ∧ B < b + c) →
  (C < a + b ∧ C < a + c ∧ C < b + c) →
  ∃ α β γ, α > 135 ∧ β < 60 ∧ γ < 60 ∧ α + β + γ = 180

theorem one_angle_greater_135 {a b c : ℝ} (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : angles_sum_not_form_triangle a b c ha hb hc) :
  ∃ α β γ, α > 135 ∧ α + β + γ = 180 :=
sorry

end one_angle_greater_135_l119_119201


namespace silvia_savings_l119_119558

def retail_price : ℝ := 1000
def guitar_center_discount_rate : ℝ := 0.15
def sweetwater_discount_rate : ℝ := 0.10
def guitar_center_shipping_fee : ℝ := 100
def sweetwater_shipping_fee : ℝ := 0

def guitar_center_cost : ℝ := retail_price * (1 - guitar_center_discount_rate) + guitar_center_shipping_fee
def sweetwater_cost : ℝ := retail_price * (1 - sweetwater_discount_rate) + sweetwater_shipping_fee

theorem silvia_savings : guitar_center_cost - sweetwater_cost = 50 := by
  sorry

end silvia_savings_l119_119558


namespace four_digit_number_divisible_by_18_l119_119169

theorem four_digit_number_divisible_by_18 : ∃ n : ℕ, (n % 2 = 0) ∧ (10 + n) % 9 = 0 ∧ n = 8 :=
by
  sorry

end four_digit_number_divisible_by_18_l119_119169


namespace find_sin_B_l119_119133

variables (a b c : ℝ) (A B C : ℝ)

def sin_law_abc (a b : ℝ) (sinA : ℝ) (sinB : ℝ) : Prop := 
  (a / sinA) = (b / sinB)

theorem find_sin_B {a b : ℝ} (sinA : ℝ) 
  (ha : a = 3) 
  (hb : b = 5) 
  (hA : sinA = 1 / 3) :
  ∃ sinB : ℝ, (sinB = 5 / 9) ∧ sin_law_abc a b sinA sinB :=
by
  use 5 / 9
  simp [sin_law_abc, ha, hb, hA]
  sorry

end find_sin_B_l119_119133


namespace charlyn_visible_area_l119_119603

noncomputable def visible_area (side_length vision_distance : ℝ) : ℝ :=
  let outer_rectangles_area := 4 * (side_length * vision_distance)
  let outer_squares_area := 4 * (vision_distance * vision_distance)
  let inner_square_area := 
    let inner_side_length := side_length - 2 * vision_distance
    inner_side_length * inner_side_length
  let total_walk_area := side_length * side_length
  total_walk_area - inner_square_area + outer_rectangles_area + outer_squares_area

theorem charlyn_visible_area :
  visible_area 10 2 = 160 := by
  sorry

end charlyn_visible_area_l119_119603


namespace power_equivalence_l119_119156

theorem power_equivalence (p : ℕ) (hp : 81^10 = 3^p) : p = 40 :=
by {
  -- Proof steps would go here
  sorry
}

end power_equivalence_l119_119156


namespace running_speed_l119_119025

theorem running_speed (side : ℕ) (time_seconds : ℕ) (speed_result : ℕ) 
  (h1 : side = 50) (h2 : time_seconds = 60) (h3 : speed_result = 12) : 
  (4 * side * 3600) / (time_seconds * 1000) = speed_result :=
by
  sorry

end running_speed_l119_119025


namespace number_of_bowls_l119_119873

theorem number_of_bowls (n : ℕ) (h1 : 12 * 8 = 96) (h2 : 6 * n = 96) : n = 16 :=
by
  -- equations from the conditions
  have h3 : 96 = 96 := by sorry
  exact sorry

end number_of_bowls_l119_119873


namespace relay_race_athlete_orders_l119_119404

def athlete_count : ℕ := 4
def cannot_run_first_leg (athlete : ℕ) : Prop := athlete = 1
def cannot_run_fourth_leg (athlete : ℕ) : Prop := athlete = 2

theorem relay_race_athlete_orders : 
  ∃ (number_of_orders : ℕ), number_of_orders = 14 := 
by 
  -- Proof is omitted because it’s not required as per instructions.
  sorry

end relay_race_athlete_orders_l119_119404


namespace remainder_of_sum_divided_by_14_l119_119888

def consecutive_odds : List ℤ := [12157, 12159, 12161, 12163, 12165, 12167, 12169]

def sum_of_consecutive_odds := consecutive_odds.sum

theorem remainder_of_sum_divided_by_14 :
  (sum_of_consecutive_odds % 14) = 7 := by
  sorry

end remainder_of_sum_divided_by_14_l119_119888


namespace avg_of_last_three_l119_119684

-- Define the conditions given in the problem
def avg_5 : Nat := 54
def avg_2 : Nat := 48
def num_list_length : Nat := 5
def first_two_length : Nat := 2

-- State the theorem
theorem avg_of_last_three
    (h_avg5 : 5 * avg_5 = 270)
    (h_avg2 : 2 * avg_2 = 96) :
  (270 - 96) / 3 = 58 :=
sorry

end avg_of_last_three_l119_119684


namespace range_of_x_l119_119502

theorem range_of_x (x y z : ℝ) (h1 : x ≥ y) (h2 : y ≥ z) (h3 : x + y + z = 1) (h4 : x^2 + y^2 + z^2 = 3) : 1 ≤ x ∧ x ≤ 5 / 3 :=
by
  sorry

end range_of_x_l119_119502


namespace salary_spending_l119_119152

theorem salary_spending (S_A S_B : ℝ) (P_A P_B : ℝ) 
  (h1 : S_A = 4500) 
  (h2 : S_A + S_B = 6000)
  (h3 : P_B = 0.85) 
  (h4 : S_A * (1 - P_A) = S_B * (1 - P_B)) : 
  P_A = 0.95 :=
by
  -- Start proofs here
  sorry

end salary_spending_l119_119152


namespace weight_loss_clothes_percentage_l119_119859

theorem weight_loss_clothes_percentage (W : ℝ) : 
  let initial_weight := W
  let weight_after_loss := 0.89 * initial_weight
  let final_weight_with_clothes := 0.9078 * initial_weight
  let added_weight_percentage := (final_weight_with_clothes / weight_after_loss - 1) * 100
  added_weight_percentage = 2 :=
by
  sorry

end weight_loss_clothes_percentage_l119_119859

import Mathlib

namespace NUMINAMATH_GPT_pencils_purchased_l1928_192846

theorem pencils_purchased (total_cost : ℝ) (num_pens : ℕ) (pen_price : ℝ) (pencil_price : ℝ) (num_pencils : ℕ) : 
  total_cost = (num_pens * pen_price) + (num_pencils * pencil_price) → 
  num_pens = 30 → 
  pen_price = 20 → 
  pencil_price = 2 → 
  total_cost = 750 →
  num_pencils = 75 :=
by
  sorry

end NUMINAMATH_GPT_pencils_purchased_l1928_192846


namespace NUMINAMATH_GPT_polynomial_identity_l1928_192814

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def g (x : ℝ) : ℝ := sorry

theorem polynomial_identity (f : ℝ → ℝ) (g : ℝ → ℝ)
  (h : ∀ g : ℝ → ℝ, ∀ x : ℝ, f (g x) = g (f x)) : 
  ∀ x : ℝ, f x = x :=
by
  sorry

end NUMINAMATH_GPT_polynomial_identity_l1928_192814


namespace NUMINAMATH_GPT_tim_score_l1928_192800

def is_prime (n : ℕ) : Prop :=
n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def first_seven_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17]

theorem tim_score :
  (first_seven_primes.sum = 58) :=
by
  sorry

end NUMINAMATH_GPT_tim_score_l1928_192800


namespace NUMINAMATH_GPT_double_acute_angle_l1928_192877

theorem double_acute_angle (α : ℝ) (h : 0 < α ∧ α < π / 2) : 0 < 2 * α ∧ 2 * α < π :=
by
  sorry

end NUMINAMATH_GPT_double_acute_angle_l1928_192877


namespace NUMINAMATH_GPT_tangent_product_l1928_192848

noncomputable def tangent (x : ℝ) : ℝ := Real.tan x

theorem tangent_product : 
  tangent (20 * Real.pi / 180) * 
  tangent (40 * Real.pi / 180) * 
  tangent (60 * Real.pi / 180) * 
  tangent (80 * Real.pi / 180) = 3 :=
by
  -- Definitions and conditions
  have tg60 := Real.tan (60 * Real.pi / 180) = Real.sqrt 3
  
  -- Add tangent addition, subtraction, and triple angle formulas
  -- tangent addition formula
  have tg_add := ∀ x y : ℝ, tangent (x + y) = (tangent x + tangent y) / (1 - tangent x * tangent y)
  -- tangent subtraction formula
  have tg_sub := ∀ x y : ℝ, tangent (x - y) = (tangent x - tangent y) / (1 + tangent x * tangent y)
  -- tangent triple angle formula
  have tg_triple := ∀ α : ℝ, tangent (3 * α) = (3 * tangent α - tangent α^3) / (1 - 3 * tangent α^2)
  
  -- sorry to skip the proof
  sorry


end NUMINAMATH_GPT_tangent_product_l1928_192848


namespace NUMINAMATH_GPT_cassie_nails_l1928_192852

-- Define the number of pets
def num_dogs := 4
def num_parrots := 8
def num_cats := 2
def num_rabbits := 6

-- Define the number of nails/claws/toes per pet
def nails_per_dog := 4 * 4
def common_claws_per_parrot := 2 * 3
def extra_toed_parrot_claws := 2 * 4
def toes_per_cat := 2 * 5 + 2 * 4
def rear_nails_per_rabbit := 2 * 5
def front_nails_per_rabbit := 3 + 4

-- Calculations
def total_dog_nails := num_dogs * nails_per_dog
def total_parrot_claws := 7 * common_claws_per_parrot + extra_toed_parrot_claws
def total_cat_toes := num_cats * toes_per_cat
def total_rabbit_nails := num_rabbits * (rear_nails_per_rabbit + front_nails_per_rabbit)

-- Total nails/claws/toes
def total_nails := total_dog_nails + total_parrot_claws + total_cat_toes + total_rabbit_nails

-- Theorem stating the problem
theorem cassie_nails : total_nails = 252 :=
by
  -- Here we would normally have the proof, but we'll skip it with sorry
  sorry

end NUMINAMATH_GPT_cassie_nails_l1928_192852


namespace NUMINAMATH_GPT_find_cost_price_l1928_192837

variable (CP SP1 SP2 : ℝ)

theorem find_cost_price
    (h1 : SP1 = CP * 0.92)
    (h2 : SP2 = CP * 1.04)
    (h3 : SP2 = SP1 + 140) :
    CP = 1166.67 :=
by
  -- Proof would be filled here
  sorry

end NUMINAMATH_GPT_find_cost_price_l1928_192837


namespace NUMINAMATH_GPT_simplify_and_evaluate_l1928_192894

noncomputable def a := 2 * Real.sin (Real.pi / 4) + (1 / 2) ^ (-1 : ℤ)

theorem simplify_and_evaluate :
  (a^2 - 4) / a / ((4 * a - 4) / a - a) + 2 / (a - 2) = -1 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l1928_192894


namespace NUMINAMATH_GPT_percentage_less_l1928_192899

theorem percentage_less (P T J : ℝ) (hT : T = 0.9375 * P) (hJ : J = 0.8 * T) : (P - J) / P * 100 = 25 :=
by
  sorry

end NUMINAMATH_GPT_percentage_less_l1928_192899


namespace NUMINAMATH_GPT_potatoes_yield_l1928_192831

theorem potatoes_yield (steps_length : ℕ) (steps_width : ℕ) (step_size : ℕ) (yield_per_sqft : ℚ) 
  (h_steps_length : steps_length = 18) 
  (h_steps_width : steps_width = 25) 
  (h_step_size : step_size = 3) 
  (h_yield_per_sqft : yield_per_sqft = 1/3) 
  : (steps_length * step_size) * (steps_width * step_size) * yield_per_sqft = 1350 := 
by 
  sorry

end NUMINAMATH_GPT_potatoes_yield_l1928_192831


namespace NUMINAMATH_GPT_marching_band_l1928_192827

theorem marching_band (total_members brass woodwind percussion : ℕ)
  (h1 : brass + woodwind + percussion = 110)
  (h2 : woodwind = 2 * brass)
  (h3 : percussion = 4 * woodwind) :
  brass = 10 := by
  sorry

end NUMINAMATH_GPT_marching_band_l1928_192827


namespace NUMINAMATH_GPT_average_of_next_seven_consecutive_integers_l1928_192862

theorem average_of_next_seven_consecutive_integers
  (a b : ℕ)
  (hb : b = (a + (a + 1) + (a + 2) + (a + 3) + (a + 4) + (a + 5) + (a + 6)) / 7) :
  ((b + (b + 1) + (b + 2) + (b + 3) + (b + 4) + (b + 5) + (b + 6)) / 7) = a + 6 :=
by
  sorry

end NUMINAMATH_GPT_average_of_next_seven_consecutive_integers_l1928_192862


namespace NUMINAMATH_GPT_least_positive_integer_exists_l1928_192844

theorem least_positive_integer_exists :
  ∃ (x : ℕ), 
    (x % 6 = 5) ∧
    (x % 8 = 7) ∧
    (x % 7 = 6) ∧
    x = 167 :=
by {
  sorry
}

end NUMINAMATH_GPT_least_positive_integer_exists_l1928_192844


namespace NUMINAMATH_GPT_arithmetic_sequence_general_formula_and_extremum_l1928_192860

noncomputable def a (n : ℕ) : ℤ := sorry
def S (n : ℕ) : ℤ := sorry

theorem arithmetic_sequence_general_formula_and_extremum :
  (a 1 + a 4 = 8) ∧ (a 2 * a 3 = 15) →
  (∃ c d : ℤ, (∀ n : ℕ, a n = c * n + d) ∨ (∀ n : ℕ, a n = -c * n + d)) ∧
  ((∃ n_min : ℕ, n_min > 0 ∧ S n_min = 1) ∧ (∃ n_max : ℕ, n_max > 0 ∧ S n_max = 16)) :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_general_formula_and_extremum_l1928_192860


namespace NUMINAMATH_GPT_find_h_l1928_192816

theorem find_h (h : ℤ) (root_condition : (-3)^3 + h * (-3) - 18 = 0) : h = -15 :=
by
  sorry

end NUMINAMATH_GPT_find_h_l1928_192816


namespace NUMINAMATH_GPT_probability_same_color_is_correct_l1928_192821

noncomputable def total_ways_select_plates : ℕ := Nat.choose 11 3
def ways_select_red_plates : ℕ := Nat.choose 6 3
def ways_select_blue_plates : ℕ := Nat.choose 5 3
noncomputable def favorable_outcomes : ℕ := ways_select_red_plates + ways_select_blue_plates
noncomputable def probability_same_color : ℚ := favorable_outcomes / total_ways_select_plates

theorem probability_same_color_is_correct :
  probability_same_color = 2/11 := 
by
  sorry

end NUMINAMATH_GPT_probability_same_color_is_correct_l1928_192821


namespace NUMINAMATH_GPT_small_cone_altitude_l1928_192840

theorem small_cone_altitude (h_f: ℝ) (a_lb: ℝ) (a_ub: ℝ) : 
  h_f = 24 → a_lb = 225 * Real.pi → a_ub = 25 * Real.pi → ∃ h_s, h_s = 12 := 
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_small_cone_altitude_l1928_192840


namespace NUMINAMATH_GPT_find_k_solve_quadratic_l1928_192838

-- Define the conditions
variables (x1 x2 k : ℝ)

-- Given conditions
def quadratic_roots : Prop :=
  x1 + x2 = 6 ∧ x1 * x2 = k

def condition_A (x1 x2 : ℝ) : Prop :=
  x1^2 * x2^2 - x1 - x2 = 115

-- Prove that k = -11 given the conditions
theorem find_k (h1: quadratic_roots x1 x2 k) (h2 : condition_A x1 x2) : k = -11 :=
  sorry

-- Prove the roots of the quadratic equation when k = -11
theorem solve_quadratic (h1 : quadratic_roots x1 x2 (-11)) : 
  x1 = 3 + 2 * Real.sqrt 5 ∧ x2 = 3 - 2 * Real.sqrt 5 ∨ 
  x1 = 3 - 2 * Real.sqrt 5 ∧ x2 = 3 + 2 * Real.sqrt 5 :=
  sorry

end NUMINAMATH_GPT_find_k_solve_quadratic_l1928_192838


namespace NUMINAMATH_GPT_common_divisor_is_19_l1928_192851

theorem common_divisor_is_19 (a d : ℤ) (h1 : d ∣ (35 * a + 57)) (h2 : d ∣ (45 * a + 76)) : d = 19 :=
sorry

end NUMINAMATH_GPT_common_divisor_is_19_l1928_192851


namespace NUMINAMATH_GPT_ChipsEquivalence_l1928_192841

theorem ChipsEquivalence
  (x y : ℕ)
  (h1 : y = x - 2)
  (h2 : 3 * x - 3 = 4 * y - 4) :
  3 * x - 3 = 24 :=
by
  sorry

end NUMINAMATH_GPT_ChipsEquivalence_l1928_192841


namespace NUMINAMATH_GPT_projection_inequality_l1928_192898

-- Define the problem with given Cartesian coordinate system, finite set of points in space, and their orthogonal projections
variable (O_xyz : Type) -- Cartesian coordinate system
variable (S : Finset O_xyz) -- finite set of points in space
variable (S_x S_y S_z : Finset O_xyz) -- sets of orthogonal projections onto the planes

-- Define the orthogonal projections (left as a comment here since detailed implementation is not specified)
-- (In Lean, actual definitions of orthogonal projections would follow mathematical and geometric definitions)

-- State the theorem to be proved
theorem projection_inequality :
  (Finset.card S) ^ 2 ≤ (Finset.card S_x) * (Finset.card S_y) * (Finset.card S_z) := 
sorry

end NUMINAMATH_GPT_projection_inequality_l1928_192898


namespace NUMINAMATH_GPT_combination_seven_choose_three_l1928_192888

-- Define the combination formula
def combination (n k : ℕ) : ℕ :=
  n.choose k

-- Define the problem-specific values
def n : ℕ := 7
def k : ℕ := 3

-- Problem statement: Prove that the number of combinations of 3 toppings from 7 is 35
theorem combination_seven_choose_three : combination 7 3 = 35 :=
  by
    sorry

end NUMINAMATH_GPT_combination_seven_choose_three_l1928_192888


namespace NUMINAMATH_GPT_total_green_peaches_l1928_192879

-- Define the known conditions
def baskets : ℕ := 7
def green_peaches_per_basket : ℕ := 2

-- State the problem and the proof goal
theorem total_green_peaches : baskets * green_peaches_per_basket = 14 := by
  -- Provide a proof here
  sorry

end NUMINAMATH_GPT_total_green_peaches_l1928_192879


namespace NUMINAMATH_GPT_find_angle_A_l1928_192855

variables {A B C a b c : ℝ}
variables {triangle_ABC : (2 * b - c) * (Real.cos A) = a * (Real.cos C)}

theorem find_angle_A (h : (2 * b - c) * (Real.cos A) = a * (Real.cos C)) : A = Real.pi / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_angle_A_l1928_192855


namespace NUMINAMATH_GPT_factor_expression_l1928_192867

theorem factor_expression (x : ℝ) : 72 * x^3 - 250 * x^7 = 2 * x^3 * (36 - 125 * x^4) :=
by
  sorry

end NUMINAMATH_GPT_factor_expression_l1928_192867


namespace NUMINAMATH_GPT_find_k_l1928_192889

theorem find_k {k : ℚ} (h : (3 : ℚ)^3 + 7 * (3 : ℚ)^2 + k * (3 : ℚ) + 23 = 0) : k = -113 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l1928_192889


namespace NUMINAMATH_GPT_g_five_eq_one_l1928_192811

noncomputable def g : ℝ → ℝ := sorry

axiom g_mul (x z : ℝ) : g (x * z) = g x * g z
axiom g_one_ne_zero : g (1) ≠ 0

theorem g_five_eq_one : g (5) = 1 := 
by
  sorry

end NUMINAMATH_GPT_g_five_eq_one_l1928_192811


namespace NUMINAMATH_GPT_lesser_solution_is_minus_15_l1928_192873

noncomputable def lesser_solution : ℤ := -15

theorem lesser_solution_is_minus_15 :
  ∃ x y : ℤ, x^2 + 10 * x - 75 = 0 ∧ y^2 + 10 * y - 75 = 0 ∧ x < y ∧ x = lesser_solution :=
by 
  sorry

end NUMINAMATH_GPT_lesser_solution_is_minus_15_l1928_192873


namespace NUMINAMATH_GPT_A_inter_complement_B_eq_01_l1928_192890

open Set

noncomputable def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x^2 - 2 * x < 0}
def B : Set ℝ := {x | x ≥ 1}
def complement_B : Set ℝ := U \ B

theorem A_inter_complement_B_eq_01 : A ∩ complement_B = (Set.Ioo 0 1) := 
by 
  sorry

end NUMINAMATH_GPT_A_inter_complement_B_eq_01_l1928_192890


namespace NUMINAMATH_GPT_gwen_more_money_from_mom_l1928_192824

def dollars_received_from_mom : ℕ := 7
def dollars_received_from_dad : ℕ := 5

theorem gwen_more_money_from_mom :
  dollars_received_from_mom - dollars_received_from_dad = 2 :=
by
  sorry

end NUMINAMATH_GPT_gwen_more_money_from_mom_l1928_192824


namespace NUMINAMATH_GPT_smallest_positive_period_of_f_minimum_value_of_f_on_interval_l1928_192809

noncomputable def f (x : ℝ) : ℝ :=
  (Real.sqrt 2) * (Real.sin (x / 2)) * (Real.cos (x / 2)) - (Real.sqrt 2) * (Real.sin (x / 2)) ^ 2

theorem smallest_positive_period_of_f : ∀ x : ℝ, f (x + 2 * Real.pi) = f x :=
by sorry

theorem minimum_value_of_f_on_interval : 
  ∃ x ∈ Set.Icc (-Real.pi) 0, 
  f x = -1 - Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_GPT_smallest_positive_period_of_f_minimum_value_of_f_on_interval_l1928_192809


namespace NUMINAMATH_GPT_granola_bars_relation_l1928_192893

theorem granola_bars_relation (x y z : ℕ) (h1 : z = x / (3 * y)) : z = x / (3 * y) :=
by {
    sorry
}

end NUMINAMATH_GPT_granola_bars_relation_l1928_192893


namespace NUMINAMATH_GPT_distinct_triples_l1928_192808

theorem distinct_triples (a b c : ℕ) (h₁: 2 * a - 1 = k₁ * b) (h₂: 2 * b - 1 = k₂ * c) (h₃: 2 * c - 1 = k₃ * a) :
  (a, b, c) = (7, 13, 25) ∨ (a, b, c) = (13, 25, 7) ∨ (a, b, c) = (25, 7, 13) := sorry

end NUMINAMATH_GPT_distinct_triples_l1928_192808


namespace NUMINAMATH_GPT_arithmetic_sequence_property_l1928_192813

variable {a : ℕ → ℕ}

-- Given condition in the problem
def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
∃ d c : ℕ, ∀ n : ℕ, a n = c + n * d

def condition (a : ℕ → ℕ) : Prop := a 4 + a 8 = 16

-- Problem statement
theorem arithmetic_sequence_property (a : ℕ → ℕ)
  (h_arith_seq : arithmetic_sequence a)
  (h_condition : condition a) :
  a 2 + a 6 + a 10 = 24 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_property_l1928_192813


namespace NUMINAMATH_GPT_steve_take_home_pay_l1928_192857

-- Defining the conditions
def annual_salary : ℕ := 40000
def tax_rate : ℝ := 0.20
def healthcare_rate : ℝ := 0.10
def union_dues : ℕ := 800

-- Defining the taxes function
def taxes (salary : ℕ) (rate : ℝ) : ℝ :=
  salary * rate

-- Defining the healthcare function
def healthcare (salary : ℕ) (rate : ℝ) : ℝ :=
  salary * rate

-- Defining the total deductions function
def total_deductions (salary : ℕ) (tax_rate : ℝ) (healthcare_rate : ℝ) (dues : ℕ) : ℝ :=
  (taxes salary tax_rate) + (healthcare salary healthcare_rate) + dues

-- Defining the take-home pay function
def take_home_pay (salary : ℕ) (deductions : ℝ) : ℝ :=
  salary - deductions

-- Using definitions to assert the take-home pay
theorem steve_take_home_pay : take_home_pay annual_salary (total_deductions annual_salary tax_rate healthcare_rate union_dues) = 27200 :=
by
  -- proof has been skipped
  sorry

end NUMINAMATH_GPT_steve_take_home_pay_l1928_192857


namespace NUMINAMATH_GPT_quadratic_roots_condition_l1928_192842

theorem quadratic_roots_condition (a : ℝ) :
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁ < 1 ∧ x₂ > 1 ∧ x₁^2 + (3 * a - 1) * x₁ + a + 8 = 0 ∧
  x₂^2 + (3 * a - 1) * x₂ + a + 8 = 0) → a < -2 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_roots_condition_l1928_192842


namespace NUMINAMATH_GPT_next_two_series_numbers_l1928_192806

theorem next_two_series_numbers :
  ∀ (a : ℕ → ℤ), a 1 = 2 → a 2 = 3 →
    (∀ n, 3 ≤ n → a n = a (n - 1) + a (n - 2) - 5) →
    a 7 = -26 ∧ a 8 = -45 :=
by
  intros a h1 h2 h3
  sorry

end NUMINAMATH_GPT_next_two_series_numbers_l1928_192806


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l1928_192849

theorem necessary_but_not_sufficient_condition (a b : ℤ) :
  (a ≠ 1 ∨ b ≠ 2) → (a + b ≠ 3) ∧ ¬((a + b ≠ 3) → (a ≠ 1 ∨ b ≠ 2)) :=
sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l1928_192849


namespace NUMINAMATH_GPT_max_tan_B_l1928_192830

theorem max_tan_B (A B : ℝ) (h : Real.sin (2 * A + B) = 2 * Real.sin B) : 
  Real.tan B ≤ Real.sqrt 3 / 3 := sorry

end NUMINAMATH_GPT_max_tan_B_l1928_192830


namespace NUMINAMATH_GPT_polynomial_inequality_holds_l1928_192812

def polynomial (x : ℝ) : ℝ := x^6 + 4 * x^5 + 2 * x^4 - 6 * x^3 - 2 * x^2 + 4 * x - 1

theorem polynomial_inequality_holds (x : ℝ) :
  (x ≤ -1 - Real.sqrt 2 ∨ x = (-1 - Real.sqrt 5) / 2 ∨ x ≥ -1 + Real.sqrt 2) →
  polynomial x ≥ 0 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_inequality_holds_l1928_192812


namespace NUMINAMATH_GPT_nut_weights_l1928_192818

noncomputable def part_weights (total_weight : ℝ) (total_parts : ℝ) : ℝ :=
  total_weight / total_parts

theorem nut_weights
  (total_weight : ℝ)
  (parts_almonds parts_walnuts parts_cashews ratio_pistachios_to_almonds : ℝ)
  (total_parts_without_pistachios total_parts_with_pistachios weight_per_part : ℝ)
  (weights_almonds weights_walnuts weights_cashews weights_pistachios : ℝ) :
  parts_almonds = 5 →
  parts_walnuts = 3 →
  parts_cashews = 2 →
  ratio_pistachios_to_almonds = 1 / 4 →
  total_parts_without_pistachios = parts_almonds + parts_walnuts + parts_cashews →
  total_parts_with_pistachios = total_parts_without_pistachios + (parts_almonds * ratio_pistachios_to_almonds) →
  weight_per_part = total_weight / total_parts_with_pistachios →
  weights_almonds = parts_almonds * weight_per_part →
  weights_walnuts = parts_walnuts * weight_per_part →
  weights_cashews = parts_cashews * weight_per_part →
  weights_pistachios = (parts_almonds * ratio_pistachios_to_almonds) * weight_per_part →
  total_weight = 300 →
  weights_almonds = 133.35 ∧
  weights_walnuts = 80.01 ∧
  weights_cashews = 53.34 ∧
  weights_pistachios = 33.34 :=
by
  intros
  sorry

end NUMINAMATH_GPT_nut_weights_l1928_192818


namespace NUMINAMATH_GPT_sum_a_c_e_l1928_192802

theorem sum_a_c_e {a b c d e f : ℝ} 
  (h1 : a / b = 2) 
  (h2 : c / d = 2) 
  (h3 : e / f = 2) 
  (h4 : b + d + f = 5) : 
  a + c + e = 10 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_sum_a_c_e_l1928_192802


namespace NUMINAMATH_GPT_first_day_bacteria_exceeds_200_l1928_192868

noncomputable def N : ℕ → ℕ := λ n => 5 * 3^n

theorem first_day_bacteria_exceeds_200 : ∃ n : ℕ, N n > 200 ∧ ∀ m : ℕ, m < n → N m ≤ 200 :=
by
  sorry

end NUMINAMATH_GPT_first_day_bacteria_exceeds_200_l1928_192868


namespace NUMINAMATH_GPT_find_function_expression_l1928_192882

theorem find_function_expression (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x^2 - 1) = x^4 + 1) :
  ∀ x : ℝ, x ≥ -1 → f x = x^2 + 2*x + 2 :=
sorry

end NUMINAMATH_GPT_find_function_expression_l1928_192882


namespace NUMINAMATH_GPT_simple_interest_fraction_l1928_192823

theorem simple_interest_fraction (P : ℝ) (R T : ℝ) (hR: R = 4) (hT: T = 5) :
  (P * R * T / 100) / P = 1 / 5 := 
by
  sorry

end NUMINAMATH_GPT_simple_interest_fraction_l1928_192823


namespace NUMINAMATH_GPT_nonnegative_integer_count_l1928_192859

def balanced_quaternary_nonnegative_count : Nat :=
  let base := 4
  let max_index := 6
  let valid_digits := [-1, 0, 1]
  let max_sum := (base ^ (max_index + 1) - 1) / (base - 1)
  max_sum + 1

theorem nonnegative_integer_count : balanced_quaternary_nonnegative_count = 5462 := by
  sorry

end NUMINAMATH_GPT_nonnegative_integer_count_l1928_192859


namespace NUMINAMATH_GPT_fit_max_blocks_l1928_192866

/-- Prove the maximum number of blocks of size 1-in x 3-in x 2-in that can fit into a box of size 4-in x 3-in x 5-in is 10. -/
theorem fit_max_blocks :
  ∀ (block_dim box_dim : ℕ → ℕ ),
  block_dim 1 = 1 ∧ block_dim 2 = 3 ∧ block_dim 3 = 2 →
  box_dim 1 = 4 ∧ box_dim 2 = 3 ∧ box_dim 3 = 5 →
  ∃ max_blocks : ℕ, max_blocks = 10 :=
by
  sorry

end NUMINAMATH_GPT_fit_max_blocks_l1928_192866


namespace NUMINAMATH_GPT_exists_perfect_square_in_sequence_of_f_l1928_192828

noncomputable def f (n : ℕ) : ℕ :=
  ⌊(n : ℝ) + Real.sqrt n⌋₊

theorem exists_perfect_square_in_sequence_of_f (m : ℕ) (h : m = 1111) :
  ∃ k, ∃ n, f^[n] m = k * k := 
sorry

end NUMINAMATH_GPT_exists_perfect_square_in_sequence_of_f_l1928_192828


namespace NUMINAMATH_GPT_blue_chips_count_l1928_192820

variable (T : ℕ) (blue_chips : ℕ) (white_chips : ℕ) (green_chips : ℕ)

-- Conditions
def condition1 : Prop := blue_chips = (T / 10)
def condition2 : Prop := white_chips = (T / 2)
def condition3 : Prop := green_chips = 12
def condition4 : Prop := blue_chips + white_chips + green_chips = T

-- Proof problem
theorem blue_chips_count (h1 : condition1 T blue_chips)
                          (h2 : condition2 T white_chips)
                          (h3 : condition3 green_chips)
                          (h4 : condition4 T blue_chips white_chips green_chips) :
  blue_chips = 3 :=
sorry

end NUMINAMATH_GPT_blue_chips_count_l1928_192820


namespace NUMINAMATH_GPT_cylinder_volume_from_cone_l1928_192865

/-- Given the volume of a cone, prove the volume of a cylinder with the same base and height. -/
theorem cylinder_volume_from_cone (V_cone : ℝ) (h : V_cone = 3.6) : 
  ∃ V_cylinder : ℝ, V_cylinder = 0.0108 :=
by
  have V_cylinder := 3 * V_cone
  have V_cylinder_meters := V_cylinder / 1000
  use V_cylinder_meters
  sorry

end NUMINAMATH_GPT_cylinder_volume_from_cone_l1928_192865


namespace NUMINAMATH_GPT_surface_area_cube_l1928_192817

theorem surface_area_cube (a : ℕ) (b : ℕ) (h : a = 2) : b = 54 :=
  by
  sorry

end NUMINAMATH_GPT_surface_area_cube_l1928_192817


namespace NUMINAMATH_GPT_novel_pages_l1928_192895

theorem novel_pages (x : ℕ)
  (h1 : x - ((1 / 6 : ℝ) * x + 10) = (5 / 6 : ℝ) * x - 10)
  (h2 : (5 / 6 : ℝ) * x - 10 - ((1 / 5 : ℝ) * ((5 / 6 : ℝ) * x - 10) + 20) = (2 / 3 : ℝ) * x - 28)
  (h3 : (2 / 3 : ℝ) * x - 28 - ((1 / 4 : ℝ) * ((2 / 3 : ℝ) * x - 28) + 25) = (1 / 2 : ℝ) * x - 46) :
  (1 / 2 : ℝ) * x - 46 = 80 → x = 252 :=
by
  sorry

end NUMINAMATH_GPT_novel_pages_l1928_192895


namespace NUMINAMATH_GPT_number_of_solutions_l1928_192803

-- Defining the conditions for the equation
def isCondition (x : ℝ) : Prop := x ≠ 2 ∧ x ≠ 3

-- Defining the equation
def eqn (x : ℝ) : Prop := (3 * x^2 - 15 * x + 18) / (x^2 - 5 * x + 6) = x - 2

-- Defining the property that we need to prove
def property (x : ℝ) : Prop := eqn x ∧ isCondition x

-- Statement of the proof problem
theorem number_of_solutions : 
  ∃! x : ℝ, property x :=
sorry

end NUMINAMATH_GPT_number_of_solutions_l1928_192803


namespace NUMINAMATH_GPT_area_of_region_bounded_by_sec_and_csc_l1928_192878

theorem area_of_region_bounded_by_sec_and_csc (x y : ℝ) :
  (∃ (x y : ℝ), x = 1 ∧ y = 1 ∧ 0 ≤ x ∧ 0 ≤ y) → 
  (∃ (area : ℝ), area = 1) :=
by 
  sorry

end NUMINAMATH_GPT_area_of_region_bounded_by_sec_and_csc_l1928_192878


namespace NUMINAMATH_GPT_geometric_sequence_fifth_term_l1928_192829

theorem geometric_sequence_fifth_term (a₁ r : ℤ) (n : ℕ) (h_a₁ : a₁ = 5) (h_r : r = -2) (h_n : n = 5) :
  (a₁ * r^(n-1) = 80) :=
by
  rw [h_a₁, h_r, h_n]
  sorry

end NUMINAMATH_GPT_geometric_sequence_fifth_term_l1928_192829


namespace NUMINAMATH_GPT_tileable_by_hook_l1928_192810

theorem tileable_by_hook (m n : ℕ) : 
  (∃ a b : ℕ, m = 3 * a ∧ (n = 4 * b ∨ n = 12 * b) ∨ 
              n = 3 * a ∧ (m = 4 * b ∨ m = 12 * b)) ↔ 12 ∣ (m * n) :=
by
  sorry

end NUMINAMATH_GPT_tileable_by_hook_l1928_192810


namespace NUMINAMATH_GPT_intersection_M_N_l1928_192887

def M : Set ℤ := {-2, -1, 0, 1, 2}
def N : Set ℤ := {x | x^2 + 2*x - 3 ≤ 0}

theorem intersection_M_N :
  M ∩ N = {-2, -1, 0, 1} :=
by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l1928_192887


namespace NUMINAMATH_GPT_intersection_of_P_and_Q_l1928_192892

theorem intersection_of_P_and_Q (P Q : Set ℕ) (h1 : P = {1, 3, 6, 9}) (h2 : Q = {1, 2, 4, 6, 8}) :
  P ∩ Q = {1, 6} :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_P_and_Q_l1928_192892


namespace NUMINAMATH_GPT_star_set_l1928_192853

def A : Set ℝ := {x | 0 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | 1 ≤ x}
def star (A B : Set ℝ) : Set ℝ := {x | (x ∈ A ∪ B) ∧ ¬(x ∈ A ∩ B)}

theorem star_set :
  star A B = {x | (0 ≤ x ∧ x < 1) ∨ (3 < x)} :=
by
  sorry

end NUMINAMATH_GPT_star_set_l1928_192853


namespace NUMINAMATH_GPT_problem_1_problem_2_problem_3_l1928_192881

-- Definition and proof state for problem 1
theorem problem_1 (a b m n : ℕ) (h₀ : a + b * Real.sqrt 3 = (m + n * Real.sqrt 3)^2) : 
  a = m^2 + 3 * n^2 ∧ b = 2 * m * n := by
  sorry

-- Definition and proof state for problem 2
theorem problem_2 (a m n : ℕ) (h₀ : a + 4 * Real.sqrt 3 = (m + n * Real.sqrt 3)^2) : 
  a = 13 ∨ a = 7 := by
  sorry

-- Definition and proof state for problem 3
theorem problem_3 : Real.sqrt (6 + 2 * Real.sqrt 5) = 1 + Real.sqrt 5 := by
  sorry

end NUMINAMATH_GPT_problem_1_problem_2_problem_3_l1928_192881


namespace NUMINAMATH_GPT_min_throws_to_same_sum_l1928_192884

/-- Define the set of possible sums for four six-sided dice --/
def dice_sum_range := {s : ℕ | 4 ≤ s ∧ s ≤ 24}

/-- The total number of possible sums when rolling four six-sided dice --/
def num_possible_sums : ℕ := 24 - 4 + 1

/-- 
  The minimum number of throws required to ensure that the same sum appears at least twice 
  by the Pigeonhole principle.
--/
theorem min_throws_to_same_sum : num_possible_sums + 1 = 22 := by
  sorry

end NUMINAMATH_GPT_min_throws_to_same_sum_l1928_192884


namespace NUMINAMATH_GPT_brenda_trays_l1928_192875

-- Define main conditions
def cookies_per_tray : ℕ := 80
def cookies_per_box : ℕ := 60
def cost_per_box : ℕ := 350
def total_cost : ℕ := 1400  -- Using cents for calculation to avoid float numbers

-- State the problem
theorem brenda_trays :
  (total_cost / cost_per_box) * cookies_per_box / cookies_per_tray = 3 := 
by
  sorry

end NUMINAMATH_GPT_brenda_trays_l1928_192875


namespace NUMINAMATH_GPT_cakes_served_at_lunch_today_l1928_192885

variable (L : ℕ)
variable (dinnerCakes : ℕ) (yesterdayCakes : ℕ) (totalCakes : ℕ)

theorem cakes_served_at_lunch_today :
  (dinnerCakes = 6) → (yesterdayCakes = 3) → (totalCakes = 14) → (L + dinnerCakes + yesterdayCakes = totalCakes) → L = 5 :=
by
  intros h_dinner h_yesterday h_total h_eq
  sorry

end NUMINAMATH_GPT_cakes_served_at_lunch_today_l1928_192885


namespace NUMINAMATH_GPT_inequality_a_squared_plus_b_squared_l1928_192872

variable (a b : ℝ)

theorem inequality_a_squared_plus_b_squared (h : a > b) : a^2 + b^2 > ab := 
sorry

end NUMINAMATH_GPT_inequality_a_squared_plus_b_squared_l1928_192872


namespace NUMINAMATH_GPT_jon_payment_per_visit_l1928_192858

theorem jon_payment_per_visit 
  (visits_per_hour : ℕ) (operating_hours_per_day : ℕ) (income_in_month : ℚ) (days_in_month : ℕ) 
  (visits_per_hour_eq : visits_per_hour = 50) 
  (operating_hours_per_day_eq : operating_hours_per_day = 24) 
  (income_in_month_eq : income_in_month = 3600) 
  (days_in_month_eq : days_in_month = 30) :
  (income_in_month / (visits_per_hour * operating_hours_per_day * days_in_month) : ℚ) = 0.10 := 
by
  sorry

end NUMINAMATH_GPT_jon_payment_per_visit_l1928_192858


namespace NUMINAMATH_GPT_find_f_neg_one_l1928_192876

theorem find_f_neg_one (f : ℝ → ℝ)
  (h_odd : ∀ x, f (-x) = -f x)
  (h_symm : ∀ x, f (4 - x) = -f x)
  (h_f3 : f 3 = 3) :
  f (-1) = 3 := 
sorry

end NUMINAMATH_GPT_find_f_neg_one_l1928_192876


namespace NUMINAMATH_GPT_value_of_expression_l1928_192861

theorem value_of_expression : (180^2 - 150^2) / 30 = 330 := by
  sorry

end NUMINAMATH_GPT_value_of_expression_l1928_192861


namespace NUMINAMATH_GPT_polynomial_roots_l1928_192847

theorem polynomial_roots :
  ∃ (x : ℚ) (y : ℚ) (z : ℚ) (w : ℚ),
    (x = 1) ∧ (y = 1) ∧ (z = -2) ∧ (w = -1/2) ∧
    2*x^4 + x^3 - 6*x^2 + x + 2 = 0 ∧
    2*y^4 + y^3 - 6*y^2 + y + 2 = 0 ∧
    2*z^4 + z^3 - 6*z^2 + z + 2 = 0 ∧
    2*w^4 + w^3 - 6*w^2 + w + 2 = 0 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_roots_l1928_192847


namespace NUMINAMATH_GPT_num_distinct_solutions_l1928_192825

theorem num_distinct_solutions : 
  (∃ x : ℝ, |x - 3| = |x + 5|) ∧ 
  (∀ x1 x2 : ℝ, |x1 - 3| = |x1 + 5| → |x2 - 3| = |x2 + 5| → x1 = x2) := 
  sorry

end NUMINAMATH_GPT_num_distinct_solutions_l1928_192825


namespace NUMINAMATH_GPT_triangle_at_most_one_right_angle_l1928_192805

-- Definition of a triangle with its angles adding up to 180 degrees
def triangle (α β γ : ℝ) : Prop := α + β + γ = 180

-- The main theorem stating that a triangle can have at most one right angle.
theorem triangle_at_most_one_right_angle (α β γ : ℝ) 
  (h₁ : triangle α β γ) 
  (h₂ : α = 90 ∨ β = 90 ∨ γ = 90) : 
  (α = 90 → β ≠ 90 ∧ γ ≠ 90) ∧ 
  (β = 90 → α ≠ 90 ∧ γ ≠ 90) ∧ 
  (γ = 90 → α ≠ 90 ∧ β ≠ 90) :=
sorry

end NUMINAMATH_GPT_triangle_at_most_one_right_angle_l1928_192805


namespace NUMINAMATH_GPT_problem_1_problem_2_l1928_192870

-- Define the propositions p and q
def proposition_p (x a : ℝ) := x^2 - (a + 1/a) * x + 1 < 0
def proposition_q (x : ℝ) := x^2 - 4 * x + 3 ≤ 0

-- Problem 1: Given a = 2 and both p and q are true, find the range of x
theorem problem_1 (a : ℝ) (x : ℝ) (ha : a = 2) (hp : proposition_p x a) (hq : proposition_q x) :
  1 ≤ x ∧ x < 2 :=
sorry

-- Problem 2: Prove that if p is a necessary but not sufficient condition for q, then 3 < a
theorem problem_2 (a : ℝ)
  (h_ns : ∀ x, proposition_q x → proposition_p x a)
  (h_not_s : ∃ x, ¬ (proposition_q x → proposition_p x a)) :
  3 < a :=
sorry

end NUMINAMATH_GPT_problem_1_problem_2_l1928_192870


namespace NUMINAMATH_GPT_average_speeds_equation_l1928_192815

theorem average_speeds_equation (x : ℝ) (hx : 0 < x) : 
  10 / x - 7 / (1.4 * x) = 10 / 60 :=
by
  sorry

end NUMINAMATH_GPT_average_speeds_equation_l1928_192815


namespace NUMINAMATH_GPT_area_is_25_l1928_192883

noncomputable def area_of_square (x : ℝ) : ℝ :=
  let side1 := 5 * x - 20
  let side2 := 25 - 4 * x
  if h : side1 = side2 then 
    side1 * side1
  else 
    0

theorem area_is_25 (x : ℝ) (h_eq : 5 * x - 20 = 25 - 4 * x) : area_of_square x = 25 :=
by
  sorry

end NUMINAMATH_GPT_area_is_25_l1928_192883


namespace NUMINAMATH_GPT_rajesh_walked_distance_l1928_192834

theorem rajesh_walked_distance (H : ℝ) (D_R : ℝ) 
  (h1 : D_R = 4 * H - 10)
  (h2 : H + D_R = 25) :
  D_R = 18 :=
by
  sorry

end NUMINAMATH_GPT_rajesh_walked_distance_l1928_192834


namespace NUMINAMATH_GPT_find_m_to_make_z1_eq_z2_l1928_192845

def z1 (m : ℝ) : ℂ := (2 * m + 7 : ℝ) + (m^2 - 2 : ℂ) * Complex.I
def z2 (m : ℝ) : ℂ := (m^2 - 8 : ℝ) + (4 * m + 3 : ℂ) * Complex.I

theorem find_m_to_make_z1_eq_z2 : 
  ∃ m : ℝ, z1 m = z2 m ∧ m = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_m_to_make_z1_eq_z2_l1928_192845


namespace NUMINAMATH_GPT_range_of_x_range_of_a_l1928_192850

-- Definitions of the conditions
def p (x a : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def q (x : ℝ) : Prop := (x^2 - x - 6 ≤ 0) ∧ (x^2 + 2 * x - 8 > 0)

-- Part (1)
theorem range_of_x (x : ℝ) (h : p x 1 ∧ q x) : 2 < x ∧ x < 3 :=
by sorry

-- Part (2)
theorem range_of_a (a : ℝ) (h : ∀ x, ¬ (p x a) → ¬ (q x)) : 1 < a ∧ a ≤ 2 :=
by sorry

end NUMINAMATH_GPT_range_of_x_range_of_a_l1928_192850


namespace NUMINAMATH_GPT_table_tennis_teams_equation_l1928_192839

-- Variables
variable (x : ℕ)

-- Conditions
def total_matches : ℕ := 28
def teams_playing_equation : Prop := x * (x - 1) = 28 * 2

-- Theorem Statement
theorem table_tennis_teams_equation : teams_playing_equation x :=
sorry

end NUMINAMATH_GPT_table_tennis_teams_equation_l1928_192839


namespace NUMINAMATH_GPT_square_difference_identity_l1928_192826

theorem square_difference_identity (a b : ℕ) : (a - b)^2 = a^2 - 2 * a * b + b^2 :=
  by sorry

lemma evaluate_expression : (101 - 2)^2 = 9801 :=
  by
    have h := square_difference_identity 101 2
    exact h

end NUMINAMATH_GPT_square_difference_identity_l1928_192826


namespace NUMINAMATH_GPT_new_average_contribution_75_l1928_192801

-- Define the conditions given in the problem
def original_contributions : ℝ := 1
def johns_donation : ℝ := 100
def increase_rate : ℝ := 1.5

-- Define a function to calculate the new average contribution size
def new_total_contributions (A : ℝ) := A + johns_donation
def new_average_contribution (A : ℝ) := increase_rate * A

-- Theorem to prove that the new average contribution size is $75
theorem new_average_contribution_75 (A : ℝ) :
  new_total_contributions A / (original_contributions + 1) = increase_rate * A →
  A = 50 →
  new_average_contribution A = 75 :=
by
  intros h1 h2
  rw [new_average_contribution, h2]
  sorry

end NUMINAMATH_GPT_new_average_contribution_75_l1928_192801


namespace NUMINAMATH_GPT_prove_problem_l1928_192896

noncomputable def proof_problem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 1) : Prop :=
  (1 + 1 / x) * (1 + 1 / y) ≥ 9

theorem prove_problem (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 1) : proof_problem x y hx hy h :=
  sorry

end NUMINAMATH_GPT_prove_problem_l1928_192896


namespace NUMINAMATH_GPT_abs_sum_l1928_192871

theorem abs_sum (a b c : ℚ) (h₁ : a = -1/4) (h₂ : b = -2) (h₃ : c = -11/4) :
  |a| + |b| - |c| = -1/2 :=
by {
  sorry
}

end NUMINAMATH_GPT_abs_sum_l1928_192871


namespace NUMINAMATH_GPT_smallest_number_increased_by_seven_divisible_by_37_47_53_l1928_192891

theorem smallest_number_increased_by_seven_divisible_by_37_47_53 : 
  ∃ n : ℕ, (n + 7) % 37 = 0 ∧ (n + 7) % 47 = 0 ∧ (n + 7) % 53 = 0 ∧ n = 92160 :=
by
  sorry

end NUMINAMATH_GPT_smallest_number_increased_by_seven_divisible_by_37_47_53_l1928_192891


namespace NUMINAMATH_GPT_solve_inequality_l1928_192832

open Set

theorem solve_inequality (a x : ℝ) : 
  (x - 2) * (a * x - 2) > 0 → 
  (a = 0 ∧ x < 2) ∨ 
  (a < 0 ∧ (2/a) < x ∧ x < 2) ∨ 
  (0 < a ∧ a < 1 ∧ ((x < 2 ∨ x > 2/a))) ∨ 
  (a = 1 ∧ x ≠ 2) ∨ 
  (a > 1 ∧ ((x < 2/a ∨ x > 2)))
  := sorry

end NUMINAMATH_GPT_solve_inequality_l1928_192832


namespace NUMINAMATH_GPT_bus_capacity_fraction_l1928_192880

theorem bus_capacity_fraction
  (capacity : ℕ)
  (x : ℚ)
  (return_fraction : ℚ)
  (total_people : ℕ)
  (capacity_eq : capacity = 200)
  (return_fraction_eq : return_fraction = 4/5)
  (total_people_eq : total_people = 310)
  (people_first_trip_eq : 200 * x + 200 * 4/5 = 310) :
  x = 3/4 :=
by
  sorry

end NUMINAMATH_GPT_bus_capacity_fraction_l1928_192880


namespace NUMINAMATH_GPT_total_fruits_picked_l1928_192819

theorem total_fruits_picked :
  let sara_pears := 6
  let tim_pears := 5
  let lily_apples := 4
  let max_oranges := 3
  sara_pears + tim_pears + lily_apples + max_oranges = 18 :=
by
  -- skip the proof
  sorry

end NUMINAMATH_GPT_total_fruits_picked_l1928_192819


namespace NUMINAMATH_GPT_alice_minimum_speed_l1928_192836

noncomputable def minimum_speed_to_exceed (d t_bob t_alice : ℝ) (v_bob : ℝ) : ℝ :=
  d / t_alice

theorem alice_minimum_speed (d : ℝ) (v_bob : ℝ) (t_lag : ℝ) (v_alice : ℝ) :
  d = 30 → v_bob = 40 → t_lag = 0.5 → v_alice = d / (d / v_bob - t_lag) → v_alice > 60 :=
by
  intros hd hv hb ht
  rw [hd, hv, hb] at ht
  simp at ht
  sorry

end NUMINAMATH_GPT_alice_minimum_speed_l1928_192836


namespace NUMINAMATH_GPT_problem_angle_magnitude_and_sin_l1928_192864

theorem problem_angle_magnitude_and_sin (
  a b c : ℝ) (A B C : ℝ) 
  (h1 : a = Real.sqrt 7) (h2 : b = 3) 
  (h3 : Real.sqrt 7 * Real.sin B + Real.sin A = 2 * Real.sqrt 3)
  (triangle_is_acute : A > 0 ∧ A < Real.pi / 2 ∧ B > 0 ∧ B < Real.pi / 2 ∧ C > 0 ∧ C < Real.pi / 2) : 
  A = Real.pi / 3 ∧ Real.sin (2 * B + Real.pi / 6) = -1 / 7 :=
by
  sorry

end NUMINAMATH_GPT_problem_angle_magnitude_and_sin_l1928_192864


namespace NUMINAMATH_GPT_problem_xy_l1928_192843

theorem problem_xy (x y : ℝ) (h1 : x + y = 25) (h2 : x^2 * y^3 + y^2 * x^3 = 25) : x * y = 1 :=
by
  sorry

end NUMINAMATH_GPT_problem_xy_l1928_192843


namespace NUMINAMATH_GPT_not_divisible_1961_1963_divisible_1963_1965_l1928_192804

def is_divisible_by_three (n : Nat) : Prop :=
  n % 3 = 0

theorem not_divisible_1961_1963 : ¬ is_divisible_by_three (1961 * 1963) :=
by
  sorry

theorem divisible_1963_1965 : is_divisible_by_three (1963 * 1965) :=
by
  sorry

end NUMINAMATH_GPT_not_divisible_1961_1963_divisible_1963_1965_l1928_192804


namespace NUMINAMATH_GPT_max_sum_of_positives_l1928_192886

theorem max_sum_of_positives (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + y + 1 / x + 1 / y = 5) : x + y ≤ 4 :=
sorry

end NUMINAMATH_GPT_max_sum_of_positives_l1928_192886


namespace NUMINAMATH_GPT_arithmetic_seq_a3_value_l1928_192874

-- Given the arithmetic sequence {a_n}, where
-- a_1 + a_2 + a_3 + a_4 + a_5 = 20
def arithmetic_seq (a : ℕ → ℝ) := ∀ n, a (n + 2) - a (n + 1) = a (n + 1) - a n

theorem arithmetic_seq_a3_value {a : ℕ → ℝ}
    (h_seq : arithmetic_seq a)
    (h_sum : a 1 + a 2 + a 3 + a 4 + a 5 = 20) :
  a 3 = 4 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_seq_a3_value_l1928_192874


namespace NUMINAMATH_GPT_find_percentage_l1928_192822

theorem find_percentage (x p : ℝ) (h1 : 0.25 * x = p * 10 - 30) (h2 : x = 680) : p = 20 := 
sorry

end NUMINAMATH_GPT_find_percentage_l1928_192822


namespace NUMINAMATH_GPT_super_cool_triangles_area_sum_l1928_192863

theorem super_cool_triangles_area_sum : 
  ∃ (areas : List ℕ), 
  (∀ (a b : ℕ), (a * b / 2 = 3 * (a + b)) → 
  (a * b / 2 ∈ areas)) ∧ 
  areas.sum = 471 := 
sorry

end NUMINAMATH_GPT_super_cool_triangles_area_sum_l1928_192863


namespace NUMINAMATH_GPT_hyperbola_equation_l1928_192897

-- Definitions for a given hyperbola
variables {a b : ℝ}
axiom a_pos : a > 0
axiom b_pos : b > 0

-- Definitions for the asymptote condition
axiom point_on_asymptote : (4 : ℝ) = (b / a) * 3

-- Definitions for the focal distance condition
axiom point_circle_intersect : (3 : ℝ)^2 + 4^2 = (a^2 + b^2)

-- The goal is to prove the hyperbola's specific equation
theorem hyperbola_equation : 
  (a^2 = 9 ∧ b^2 = 16) →
  (∃ a b : ℝ, (4 : ℝ)^2 + 3^2 = (a^2 + b^2) ∧ 
               (4 : ℝ) = (b / a) * 3 ∧ 
               ((a^2 = 9) ∧ (b^2 = 16)) ∧ (a > 0) ∧ (b > 0)) :=
sorry

end NUMINAMATH_GPT_hyperbola_equation_l1928_192897


namespace NUMINAMATH_GPT_graph_abs_symmetric_yaxis_l1928_192869

theorem graph_abs_symmetric_yaxis : 
  ∀ x : ℝ, |x| = |(-x)| :=
by
  intro x
  sorry

end NUMINAMATH_GPT_graph_abs_symmetric_yaxis_l1928_192869


namespace NUMINAMATH_GPT_gcd_12012_18018_l1928_192807

theorem gcd_12012_18018 : Int.gcd 12012 18018 = 6006 := 
by
  sorry

end NUMINAMATH_GPT_gcd_12012_18018_l1928_192807


namespace NUMINAMATH_GPT_simplify_expression_eval_l1928_192856

noncomputable def simplify_expression (x : ℝ) : ℝ :=
  ((x / (x - 1) - 1) / ((x^2 - 1) / (x^2 - 2 * x + 1)))

theorem simplify_expression_eval : simplify_expression (Real.sqrt 5 - 1) = (Real.sqrt 5) / 5 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_eval_l1928_192856


namespace NUMINAMATH_GPT_ages_proof_l1928_192833

noncomputable def A : ℝ := 12.1
noncomputable def B : ℝ := 6.1
noncomputable def C : ℝ := 11.3

-- Conditions extracted from the problem
def sum_of_ages (A B C : ℝ) : Prop := A + B + C = 29.5
def specific_age (C : ℝ) : Prop := C = 11.3
def twice_as_old (A B : ℝ) : Prop := A = 2 * B

theorem ages_proof : 
  ∃ (A B C : ℝ), 
    specific_age C ∧ twice_as_old A B ∧ sum_of_ages A B C :=
by
  exists 12.1, 6.1, 11.3
  sorry

end NUMINAMATH_GPT_ages_proof_l1928_192833


namespace NUMINAMATH_GPT_sum_difference_l1928_192835

def sum_even (n : ℕ) : ℕ :=
  (n / 2) * (2 + n)

def sum_odd (n : ℕ) : ℕ :=
  (n / 2) * (1 + (n - 1))

theorem sum_difference : sum_even 100 - sum_odd 99 = 50 :=
by
  sorry

end NUMINAMATH_GPT_sum_difference_l1928_192835


namespace NUMINAMATH_GPT_inequality_hold_l1928_192854

theorem inequality_hold (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a + b + c = 3) :
  1/a^2 + 1/b^2 + 1/c^2 ≥ a^2 + b^2 + c^2 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_inequality_hold_l1928_192854

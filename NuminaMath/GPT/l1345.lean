import Mathlib

namespace NUMINAMATH_GPT_solution_set_of_inequality_l1345_134565

theorem solution_set_of_inequality (x : ℝ) :
  (x - 1) * (x - 2) ≤ 0 ↔ 1 ≤ x ∧ x ≤ 2 := by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1345_134565


namespace NUMINAMATH_GPT_solve_trig_equation_l1345_134559

theorem solve_trig_equation (x : ℝ) : 
  2 * Real.cos (13 * x) + 3 * Real.cos (3 * x) + 3 * Real.cos (5 * x) - 8 * Real.cos x * (Real.cos (4 * x))^3 = 0 ↔ 
  ∃ (k : ℤ), x = (k * Real.pi) / 12 :=
sorry

end NUMINAMATH_GPT_solve_trig_equation_l1345_134559


namespace NUMINAMATH_GPT_sunil_interest_l1345_134531

noncomputable def compound_interest (P r n t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem sunil_interest :
  let A := 19828.80
  let r := 0.08
  let n := 1
  let t := 2
  let P := 19828.80 / (1 + 0.08) ^ 2
  P * (1 + r / n) ^ (n * t) = 19828.80 →
  A - P = 2828.80 :=
by
  sorry

end NUMINAMATH_GPT_sunil_interest_l1345_134531


namespace NUMINAMATH_GPT_simplify_fraction_l1345_134567

theorem simplify_fraction : (90 : ℚ) / (126 : ℚ) = 5 / 7 := 
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l1345_134567


namespace NUMINAMATH_GPT_range_of_a_for_two_critical_points_l1345_134530

noncomputable def f (a x : ℝ) : ℝ := 2 * a^x - Real.exp 1 * x^2 + 18

theorem range_of_a_for_two_critical_points (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ deriv (f a) x1 = 0 ∧ deriv (f a) x2 = 0) ↔ (a ∈ Set.Ioo (1 / Real.exp 1) 1 ∪ Set.Ioo 1 (Real.exp 1)) :=
sorry

end NUMINAMATH_GPT_range_of_a_for_two_critical_points_l1345_134530


namespace NUMINAMATH_GPT_f_div_36_l1345_134570

open Nat

def f (n : ℕ) : ℕ :=
  (2 * n + 7) * 3^n + 9

theorem f_div_36 (n : ℕ) : (f n) % 36 = 0 := 
  sorry

end NUMINAMATH_GPT_f_div_36_l1345_134570


namespace NUMINAMATH_GPT_zoo_problem_l1345_134541

theorem zoo_problem (M B L : ℕ) (h1: 26 ≤ M + B + L) (h2: M + B + L ≤ 32) 
    (h3: M + L > B) (h4: B + L = 2 * M) (h5: M + B = 3 * L + 3) (h6: B = L / 2) : 
    B = 3 :=
by
  sorry

end NUMINAMATH_GPT_zoo_problem_l1345_134541


namespace NUMINAMATH_GPT_solve_fraction_problem_l1345_134529

theorem solve_fraction_problem (n : ℝ) (h : (4 + n) / (7 + n) = 7 / 9) : n = 13 / 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_fraction_problem_l1345_134529


namespace NUMINAMATH_GPT_rotation_volumes_l1345_134575

theorem rotation_volumes (a b c V1 V2 V3 : ℝ) (h : a^2 + b^2 = c^2)
    (hV1 : V1 = (1 / 3) * Real.pi * a^2 * b^2 / c)
    (hV2 : V2 = (1 / 3) * Real.pi * b^2 * a)
    (hV3 : V3 = (1 / 3) * Real.pi * a^2 * b) : 
    (1 / V1^2) = (1 / V2^2) + (1 / V3^2) :=
sorry

end NUMINAMATH_GPT_rotation_volumes_l1345_134575


namespace NUMINAMATH_GPT_fraction_equality_l1345_134586

variables {R : Type*} [Field R] {m n p q : R}

theorem fraction_equality 
  (h1 : m / n = 15)
  (h2 : p / n = 3)
  (h3 : p / q = 1 / 10) :
  m / q = 1 / 2 :=
sorry

end NUMINAMATH_GPT_fraction_equality_l1345_134586


namespace NUMINAMATH_GPT_regular_triangular_prism_properties_l1345_134503

-- Regular triangular pyramid defined
structure RegularTriangularPyramid (height : ℝ) (base_side : ℝ)

-- Regular triangular prism defined
structure RegularTriangularPrism (height : ℝ) (base_side : ℝ) (lateral_area : ℝ)

-- Given data
def pyramid := RegularTriangularPyramid 15 12
def prism_lateral_area := 120

-- Statement of the problem
theorem regular_triangular_prism_properties (h_prism : ℝ) (ratio_lateral_area : ℚ) :
  (h_prism = 10 ∨ h_prism = 5) ∧ (ratio_lateral_area = 1/9 ∨ ratio_lateral_area = 4/9) :=
sorry

end NUMINAMATH_GPT_regular_triangular_prism_properties_l1345_134503


namespace NUMINAMATH_GPT_incorrect_independence_test_conclusion_l1345_134592

-- Definitions for each condition
def independence_test_principle_of_small_probability (A : Prop) : Prop :=
A  -- Statement A: The independence test is based on the principle of small probability.

def independence_test_conclusion_variability (C : Prop) : Prop :=
C  -- Statement C: Different samples may lead to different conclusions in the independence test.

def independence_test_not_the_only_method (D : Prop) : Prop :=
D  -- Statement D: The independence test is not the only method to determine whether two categorical variables are related.

-- Incorrect statement B
def independence_test_conclusion_always_correct (B : Prop) : Prop :=
B  -- Statement B: The conclusion drawn from the independence test is always correct.

-- Prove that statement B is incorrect given conditions A, C, and D
theorem incorrect_independence_test_conclusion (A B C D : Prop) 
  (hA : independence_test_principle_of_small_probability A)
  (hC : independence_test_conclusion_variability C)
  (hD : independence_test_not_the_only_method D) :
  ¬ independence_test_conclusion_always_correct B :=
sorry

end NUMINAMATH_GPT_incorrect_independence_test_conclusion_l1345_134592


namespace NUMINAMATH_GPT_find_x_l1345_134500

theorem find_x (u : ℕ) (h₁ : u = 90) (w : ℕ) (h₂ : w = u + 10)
                (z : ℕ) (h₃ : z = w + 25) (y : ℕ) (h₄ : y = z + 15)
                (x : ℕ) (h₅ : x = y + 3) : x = 143 :=
by {
  -- Proof will be included here
  sorry
}

end NUMINAMATH_GPT_find_x_l1345_134500


namespace NUMINAMATH_GPT_smallest_number_divisible_l1345_134568

   theorem smallest_number_divisible (d n : ℕ) (h₁ : (n + 7) % 11 = 0) (h₂ : (n + 7) % 24 = 0) (h₃ : (n + 7) % d = 0) (h₄ : (n + 7) = 257) : n = 250 :=
   by
     sorry
   
end NUMINAMATH_GPT_smallest_number_divisible_l1345_134568


namespace NUMINAMATH_GPT_largest_of_four_consecutive_primes_l1345_134598

noncomputable def sum_of_primes_is_prime (p1 p2 p3 p4 : ℕ) : Prop :=
  Prime p1 ∧ Prime p2 ∧ Prime p3 ∧ Prime p4 ∧ Prime (p1 + p2 + p3 + p4)

theorem largest_of_four_consecutive_primes :
  ∃ p1 p2 p3 p4, 
  sum_of_primes_is_prime p1 p2 p3 p4 ∧ 
  p1 < p2 ∧ p2 < p3 ∧ p3 < p4 ∧ 
  (p1, p2, p3, p4) = (2, 3, 5, 7) ∧ 
  max p1 (max p2 (max p3 p4)) = 7 :=
by {
  sorry                                 -- solve this in Lean
}

end NUMINAMATH_GPT_largest_of_four_consecutive_primes_l1345_134598


namespace NUMINAMATH_GPT_unique_set_property_l1345_134588

theorem unique_set_property (a b c : ℕ) (h1: 1 < a) (h2: 1 < b) (h3: 1 < c) 
    (gcd_ab_c: (Nat.gcd a b = 1 ∧ Nat.gcd b c = 1 ∧ Nat.gcd a c = 1))
    (property_abc: (a * b) % c = (a * c) % b ∧ (a * c) % b = (b * c) % a) : 
    (a = 2 ∧ b = 3 ∧ c = 5) ∨ 
    (a = 2 ∧ b = 5 ∧ c = 3) ∨ 
    (a = 3 ∧ b = 2 ∧ c = 5) ∨ 
    (a = 3 ∧ b = 5 ∧ c = 2) ∨ 
    (a = 5 ∧ b = 2 ∧ c = 3) ∨ 
    (a = 5 ∧ b = 3 ∧ c = 2) := sorry

end NUMINAMATH_GPT_unique_set_property_l1345_134588


namespace NUMINAMATH_GPT_cube_cut_possible_l1345_134536

theorem cube_cut_possible (a b : ℝ) (unit_a : a = 1) (unit_b : b = 1) : 
  ∃ (cut : ℝ → ℝ → Prop), (∀ x y, cut x y → (∃ q r : ℝ, q > 0 ∧ r > 0 ∧ q * r > 1)) :=
sorry

end NUMINAMATH_GPT_cube_cut_possible_l1345_134536


namespace NUMINAMATH_GPT_trigonometric_identity_l1345_134543

noncomputable def tan_sum (alpha : ℝ) : Prop :=
  Real.tan (alpha + Real.pi / 4) = 2

noncomputable def trigonometric_expression (alpha : ℝ) : ℝ :=
  (Real.sin alpha + 2 * Real.cos alpha) / (Real.sin alpha - 2 * Real.cos alpha)

theorem trigonometric_identity (alpha : ℝ) (h : tan_sum alpha) : 
  trigonometric_expression alpha = -7 / 5 :=
sorry

end NUMINAMATH_GPT_trigonometric_identity_l1345_134543


namespace NUMINAMATH_GPT_find_stream_speed_l1345_134513

variable (D : ℝ) (v : ℝ)

theorem find_stream_speed 
  (h1 : ∀D v, D / (63 - v) = 2 * (D / (63 + v)))
  (h2 : v = 21) :
  true := 
  by
  sorry

end NUMINAMATH_GPT_find_stream_speed_l1345_134513


namespace NUMINAMATH_GPT_sequence_inequality_l1345_134548

def F : ℕ → ℕ
| 0 => 1
| 1 => 1
| 2 => 2
| (n+2) => F (n+1) + F n

theorem sequence_inequality (n : ℕ) :
  (F (n+1) : ℝ)^(1 / n) ≥ 1 + 1 / ((F n : ℝ)^(1 / n)) :=
by
  sorry

end NUMINAMATH_GPT_sequence_inequality_l1345_134548


namespace NUMINAMATH_GPT_problem_l1345_134566

-- Define what it means to be a factor or divisor
def is_factor (a b : ℕ) : Prop := ∃ k : ℕ, b = k * a
def is_divisor (a b : ℕ) : Prop := a ∣ b

-- The specific problem conditions
def statement_A := is_factor 4 28
def statement_B := is_divisor 19 209 ∧ ¬ is_divisor 19 57
def statement_C := ¬ is_divisor 30 90 ∧ ¬ is_divisor 30 76
def statement_D := is_divisor 14 28 ∧ ¬ is_divisor 14 56
def statement_E := is_factor 9 162

-- The proof problem
theorem problem : statement_A ∧ ¬statement_B ∧ ¬statement_C ∧ ¬statement_D ∧ statement_E :=
by 
  -- You would normally provide the proof here
  sorry

end NUMINAMATH_GPT_problem_l1345_134566


namespace NUMINAMATH_GPT_cash_realized_without_brokerage_l1345_134585

theorem cash_realized_without_brokerage
  (C : ℝ)
  (h1 : (1 / 4) * (1 / 100) = 1 / 400)
  (h2 : C + (C / 400) = 108) :
  C = 43200 / 401 :=
by
  sorry

end NUMINAMATH_GPT_cash_realized_without_brokerage_l1345_134585


namespace NUMINAMATH_GPT_no_square_ends_in_2012_l1345_134537

theorem no_square_ends_in_2012 : ¬ ∃ a : ℤ, (a * a) % 10 = 2 := by
  sorry

end NUMINAMATH_GPT_no_square_ends_in_2012_l1345_134537


namespace NUMINAMATH_GPT_constants_sum_l1345_134554

theorem constants_sum (A B C D : ℕ) 
  (h : ∀ n : ℕ, n ≥ 4 → n^4 = A * (n.choose 4) + B * (n.choose 3) + C * (n.choose 2) + D * (n.choose 1)) 
  : A + B + C + D = 75 :=
by
  sorry

end NUMINAMATH_GPT_constants_sum_l1345_134554


namespace NUMINAMATH_GPT_scientific_notation_l1345_134552

theorem scientific_notation :
  686530000 = 6.8653 * 10^8 :=
sorry

end NUMINAMATH_GPT_scientific_notation_l1345_134552


namespace NUMINAMATH_GPT_second_number_removed_l1345_134594

theorem second_number_removed (S : ℝ) (X : ℝ) (h1 : S / 50 = 38) (h2 : (S - 45 - X) / 48 = 37.5) : X = 55 :=
by
  sorry

end NUMINAMATH_GPT_second_number_removed_l1345_134594


namespace NUMINAMATH_GPT_sin_2alpha_over_cos_alpha_sin_beta_value_l1345_134540

variable (α β : ℝ)

-- Given conditions
axiom alpha_pos : 0 < α
axiom alpha_lt_pi_div_2 : α < Real.pi / 2
axiom beta_pos : 0 < β
axiom beta_lt_pi_div_2 : β < Real.pi / 2
axiom cos_alpha_eq : Real.cos α = 3 / 5
axiom cos_beta_plus_alpha_eq : Real.cos (β + α) = 5 / 13

-- The results to prove
theorem sin_2alpha_over_cos_alpha : (Real.sin (2 * α) / (Real.cos α ^ 2 + Real.cos (2 * α)) = 12) :=
sorry

theorem sin_beta_value : (Real.sin β = 16 / 65) :=
sorry


end NUMINAMATH_GPT_sin_2alpha_over_cos_alpha_sin_beta_value_l1345_134540


namespace NUMINAMATH_GPT_customers_added_during_lunch_rush_l1345_134563

noncomputable def initial_customers := 29.0
noncomputable def total_customers_after_lunch_rush := 83.0
noncomputable def expected_customers_added := 54.0

theorem customers_added_during_lunch_rush :
  (total_customers_after_lunch_rush - initial_customers) = expected_customers_added :=
by
  sorry

end NUMINAMATH_GPT_customers_added_during_lunch_rush_l1345_134563


namespace NUMINAMATH_GPT_cost_of_each_gumdrop_l1345_134522

theorem cost_of_each_gumdrop (cents : ℕ) (gumdrops : ℕ) (cost_per_gumdrop : ℕ) : 
  cents = 224 → gumdrops = 28 → cost_per_gumdrop = cents / gumdrops → cost_per_gumdrop = 8 :=
by
  intros h_cents h_gumdrops h_cost
  sorry

end NUMINAMATH_GPT_cost_of_each_gumdrop_l1345_134522


namespace NUMINAMATH_GPT_brad_started_after_maxwell_l1345_134551

theorem brad_started_after_maxwell :
  ∀ (distance maxwell_speed brad_speed maxwell_time : ℕ),
  distance = 94 →
  maxwell_speed = 4 →
  brad_speed = 6 →
  maxwell_time = 10 →
  (distance - maxwell_speed * maxwell_time) / brad_speed = 9 := 
by
  intros distance maxwell_speed brad_speed maxwell_time h_dist h_m_speed h_b_speed h_m_time
  sorry

end NUMINAMATH_GPT_brad_started_after_maxwell_l1345_134551


namespace NUMINAMATH_GPT_walnut_swap_exists_l1345_134505

theorem walnut_swap_exists (n : ℕ) (h_n : n = 2021) :
  ∃ k : ℕ, k < n ∧ ∃ a b : ℕ, a < k ∧ k < b :=
by
  sorry

end NUMINAMATH_GPT_walnut_swap_exists_l1345_134505


namespace NUMINAMATH_GPT_point_B_value_l1345_134556

/-- Given that point A represents the number 7 on a number line
    and point A is moved 3 units to the right to point B,
    prove that point B represents the number 10 -/
theorem point_B_value (A B : ℤ) (h1: A = 7) (h2: B = A + 3) : B = 10 :=
  sorry

end NUMINAMATH_GPT_point_B_value_l1345_134556


namespace NUMINAMATH_GPT_problems_per_worksheet_l1345_134596

theorem problems_per_worksheet (total_worksheets : ℕ) (graded_worksheets : ℕ) (remaining_problems : ℕ) (h1 : total_worksheets = 15) (h2 : graded_worksheets = 7) (h3 : remaining_problems = 24) : (remaining_problems / (total_worksheets - graded_worksheets)) = 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_problems_per_worksheet_l1345_134596


namespace NUMINAMATH_GPT_min_value_condition_l1345_134549

open Real

theorem min_value_condition 
  (m n : ℝ) 
  (h1 : m > 0) 
  (h2 : n > 0) 
  (h3 : 2 * m + n = 1) : 
  (1 / m + 2 / n) ≥ 8 :=
sorry

end NUMINAMATH_GPT_min_value_condition_l1345_134549


namespace NUMINAMATH_GPT_percentage_increase_edge_length_l1345_134560

theorem percentage_increase_edge_length (a a' : ℝ) (h : 6 * (a')^2 = 6 * a^2 + 1.25 * 6 * a^2) : a' = 1.5 * a :=
by sorry

end NUMINAMATH_GPT_percentage_increase_edge_length_l1345_134560


namespace NUMINAMATH_GPT_expression_values_l1345_134581

noncomputable def sign (x : ℝ) : ℝ := 
if x > 0 then 1 else -1

theorem expression_values (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  ∃ v ∈ ({-4, 0, 4} : Set ℝ), 
    sign a + sign b + sign c + sign (a * b * c) = v := by
  sorry

end NUMINAMATH_GPT_expression_values_l1345_134581


namespace NUMINAMATH_GPT_arrange_animals_adjacent_l1345_134578

theorem arrange_animals_adjacent:
  let chickens := 5
  let dogs := 3
  let cats := 6
  let rabbits := 4
  let total_animals := 18
  let group_orderings := 24 -- 4!
  let chicken_orderings := 120 -- 5!
  let dog_orderings := 6 -- 3!
  let cat_orderings := 720 -- 6!
  let rabbit_orderings := 24 -- 4!
  total_animals = chickens + dogs + cats + rabbits →
  chickens > 0 ∧ dogs > 0 ∧ cats > 0 ∧ rabbits > 0 →
  group_orderings * chicken_orderings * dog_orderings * cat_orderings * rabbit_orderings = 17863680 :=
  by intros; sorry

end NUMINAMATH_GPT_arrange_animals_adjacent_l1345_134578


namespace NUMINAMATH_GPT_dante_walk_time_l1345_134509

-- Define conditions and problem
variables (T R : ℝ)

-- Conditions as per the problem statement
def wind_in_favor_condition : Prop := 0.8 * T = 15
def wind_against_condition : Prop := 1.25 * T = 7
def total_walk_time_condition : Prop := 15 + 7 = 22
def total_time_away_condition : Prop := 32 - 22 = 10
def lake_park_restaurant_condition : Prop := 0.8 * R = 10

-- Proof statement
theorem dante_walk_time :
  wind_in_favor_condition T ∧
  wind_against_condition T ∧
  total_walk_time_condition ∧
  total_time_away_condition ∧
  lake_park_restaurant_condition R →
  R = 12.5 :=
by
  intros
  sorry

end NUMINAMATH_GPT_dante_walk_time_l1345_134509


namespace NUMINAMATH_GPT_lamp_cost_l1345_134528

def saved : ℕ := 500
def couch : ℕ := 750
def table : ℕ := 100
def remaining_owed : ℕ := 400

def total_cost_without_lamp : ℕ := couch + table

theorem lamp_cost :
  total_cost_without_lamp - saved + lamp = remaining_owed → lamp = 50 := by
  sorry

end NUMINAMATH_GPT_lamp_cost_l1345_134528


namespace NUMINAMATH_GPT_max_sum_of_factors_l1345_134511

theorem max_sum_of_factors (A B C : ℕ) (h1 : A ≠ B) (h2 : B ≠ C) (h3 : A ≠ C) (h4 : 0 < A) (h5 : 0 < B) (h6 : 0 < C) (h7 : A * B * C = 3003) :
  A + B + C ≤ 45 :=
sorry

end NUMINAMATH_GPT_max_sum_of_factors_l1345_134511


namespace NUMINAMATH_GPT_find_constant_l1345_134504

-- Define the variables: t, x, y, and the constant
variable (t x y constant : ℝ)

-- Conditions
def x_def : x = constant - 2 * t :=
  by sorry

def y_def : y = 2 * t - 2 :=
  by sorry

def x_eq_y_at_t : t = 0.75 → x = y :=
  by sorry

-- Proposition: Prove that the constant in the equation for x is 1
theorem find_constant (ht : t = 0.75) (hx : x = constant - 2 * t) (hy : y = 2 * t - 2) (he : x = y) :
  constant = 1 :=
  by sorry

end NUMINAMATH_GPT_find_constant_l1345_134504


namespace NUMINAMATH_GPT_simplify_and_evaluate_l1345_134595

-- Define the variables
variables (x y : ℝ)

-- Define the expression
def expression := 2 * x * y + (3 * x * y - 2 * y^2) - 2 * (x * y - y^2)

-- Introduce the conditions
theorem simplify_and_evaluate : 
  (x = -1) → (y = 2) → expression x y = -6 := 
by 
  intro hx hy 
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l1345_134595


namespace NUMINAMATH_GPT_travel_A_to_D_l1345_134515

-- Definitions for the number of roads between each pair of cities
def roads_A_to_B : ℕ := 3
def roads_A_to_C : ℕ := 1
def roads_B_to_C : ℕ := 2
def roads_B_to_D : ℕ := 1
def roads_C_to_D : ℕ := 3

-- Theorem stating the total number of ways to travel from A to D visiting each city exactly once
theorem travel_A_to_D : roads_A_to_B * roads_B_to_C * roads_C_to_D + roads_A_to_C * roads_B_to_C * roads_B_to_D = 20 :=
by
  -- Formal proof goes here
  sorry

end NUMINAMATH_GPT_travel_A_to_D_l1345_134515


namespace NUMINAMATH_GPT_relationship_ab_l1345_134519

noncomputable def a : ℝ := Real.log 243 / Real.log 5
noncomputable def b : ℝ := Real.log 27 / Real.log 3

theorem relationship_ab : a = (5 / 3) * b := sorry

end NUMINAMATH_GPT_relationship_ab_l1345_134519


namespace NUMINAMATH_GPT_students_like_apple_chocolate_not_blueberry_l1345_134561

theorem students_like_apple_chocolate_not_blueberry
  (n d a b c abc : ℕ)
  (h1 : n = 50)
  (h2 : d = 15)
  (h3 : a = 25)
  (h4 : b = 20)
  (h5 : c = 10)
  (h6 : abc = 5)
  (h7 : (n - d) = 35)
  (h8 : (55 - (a + b + c - abc)) = 35) :
  (20 - abc) = (15 : ℕ) :=
by
  sorry

end NUMINAMATH_GPT_students_like_apple_chocolate_not_blueberry_l1345_134561


namespace NUMINAMATH_GPT_perpendicular_condition_l1345_134593

theorem perpendicular_condition (m : ℝ) : 
  (2 * (m + 1) * (m - 3) + 2 * (m - 3) = 0) ↔ (m = 3 ∨ m = -3) :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_condition_l1345_134593


namespace NUMINAMATH_GPT_prove_inequality_l1345_134501

noncomputable def a : ℝ := Real.sin (33 * Real.pi / 180)
noncomputable def b : ℝ := Real.cos (55 * Real.pi / 180)
noncomputable def c : ℝ := Real.tan (55 * Real.pi / 180)

theorem prove_inequality : c > b ∧ b > a :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_prove_inequality_l1345_134501


namespace NUMINAMATH_GPT_smallest_product_bdf_l1345_134510

theorem smallest_product_bdf 
  (a b c d e f : ℕ) 
  (h1 : (a + 1) * (c * e) = a * c * e + 3 * b * d * f)
  (h2 : a * (c + 1) * e = a * c * e + 4 * b * d * f)
  (h3 : a * c * (e + 1) = a * c * e + 5 * b * d * f) : 
  b * d * f = 60 := 
sorry

end NUMINAMATH_GPT_smallest_product_bdf_l1345_134510


namespace NUMINAMATH_GPT_hall_volume_proof_l1345_134524

-- Define the given conditions.
def hall_length (l : ℝ) : Prop := l = 18
def hall_width (w : ℝ) : Prop := w = 9
def floor_ceiling_area_eq_wall_area (h l w : ℝ) : Prop := 
  2 * (l * w) = 2 * (l * h) + 2 * (w * h)

-- Define the volume calculation.
def hall_volume (l w h V : ℝ) : Prop := 
  V = l * w * h

-- The main theorem stating that the volume is 972 cubic meters.
theorem hall_volume_proof (l w h V : ℝ) 
  (length : hall_length l) 
  (width : hall_width w) 
  (fc_eq_wa : floor_ceiling_area_eq_wall_area h l w) 
  (volume : hall_volume l w h V) : 
  V = 972 :=
  sorry

end NUMINAMATH_GPT_hall_volume_proof_l1345_134524


namespace NUMINAMATH_GPT_mango_distribution_l1345_134512

theorem mango_distribution (harvested_mangoes : ℕ) (sold_fraction : ℕ) (received_per_neighbor : ℕ)
  (h_harvested : harvested_mangoes = 560)
  (h_sold_fraction : sold_fraction = 2)
  (h_received_per_neighbor : received_per_neighbor = 35) :
  (harvested_mangoes / sold_fraction) = (harvested_mangoes / sold_fraction) / received_per_neighbor :=
by
  sorry

end NUMINAMATH_GPT_mango_distribution_l1345_134512


namespace NUMINAMATH_GPT_points_on_circle_l1345_134589

theorem points_on_circle (t : ℝ) : (∃ (x y : ℝ), x = Real.cos (2 * t) ∧ y = Real.sin (2 * t) ∧ (x^2 + y^2 = 1)) := by
  sorry

end NUMINAMATH_GPT_points_on_circle_l1345_134589


namespace NUMINAMATH_GPT_tangent_line_eq_at_0_max_min_values_l1345_134582

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.cos x - x

theorem tangent_line_eq_at_0 : ∀ x : ℝ, x = 0 → f x = 1 :=
by
  sorry

theorem max_min_values : (∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi / 2 → f 0 ≥ f x) ∧ (f (Real.pi / 2) = -Real.pi / 2) :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_eq_at_0_max_min_values_l1345_134582


namespace NUMINAMATH_GPT_fewer_twos_for_100_l1345_134576

theorem fewer_twos_for_100 : (222 / 2 - 22 / 2) = 100 := by
  sorry

end NUMINAMATH_GPT_fewer_twos_for_100_l1345_134576


namespace NUMINAMATH_GPT_tangent_alpha_l1345_134584

open Real

noncomputable def a (α : ℝ) : ℝ × ℝ := (sin α, 2)
noncomputable def b (α : ℝ) : ℝ × ℝ := (-cos α, 1)

theorem tangent_alpha (α : ℝ) (h : ∀ k : ℝ, a α = (k • b α)) : tan α = -2 := by
  have h1 : sin α / -cos α = 2 := by sorry
  have h2 : tan α = -2 := by sorry
  exact h2

end NUMINAMATH_GPT_tangent_alpha_l1345_134584


namespace NUMINAMATH_GPT_shop_weekly_earnings_l1345_134506

theorem shop_weekly_earnings
  (price_women: ℕ := 18)
  (price_men: ℕ := 15)
  (time_open_hours: ℕ := 12)
  (minutes_per_hour: ℕ := 60)
  (weekly_days: ℕ := 7)
  (sell_rate_women: ℕ := 30)
  (sell_rate_men: ℕ := 40) :
  (time_open_hours * (minutes_per_hour / sell_rate_women) * price_women +
   time_open_hours * (minutes_per_hour / sell_rate_men) * price_men) * weekly_days = 4914 := 
sorry

end NUMINAMATH_GPT_shop_weekly_earnings_l1345_134506


namespace NUMINAMATH_GPT_volume_of_original_cube_l1345_134558

theorem volume_of_original_cube (s : ℝ) (h : (s + 2) * (s - 3) * s - s^3 = 26) : s^3 = 343 := 
sorry

end NUMINAMATH_GPT_volume_of_original_cube_l1345_134558


namespace NUMINAMATH_GPT_average_of_consecutive_integers_l1345_134587

variable (c : ℕ)
variable (d : ℕ)

-- Given condition: d == (c + (c+1) + (c+2) + (c+3) + (c+4) + (c+5) + (c+6)) / 7
def condition1 : Prop := d = (c + (c+1) + (c+2) + (c+3) + (c+4) + (c+5) + (c+6)) / 7

-- The theorem to prove
theorem average_of_consecutive_integers : condition1 c d → 
  (d + 1 + d + 2 + d + 3 + d + 4 + d + 5 + d + 6 + d + 7 + d + 8 + d + 9) / 10 = c + 9 :=
sorry

end NUMINAMATH_GPT_average_of_consecutive_integers_l1345_134587


namespace NUMINAMATH_GPT_distance_A_to_B_is_7km_l1345_134564

theorem distance_A_to_B_is_7km
  (v1 v2 : ℝ) 
  (t_meet_before : ℝ)
  (t1_after_meet t2_after_meet : ℝ)
  (d1_before_meet d2_before_meet : ℝ)
  (d_after_meet : ℝ)
  (h1 : d1_before_meet = d2_before_meet + 1)
  (h2 : t_meet_before = d1_before_meet / v1)
  (h3 : t_meet_before = d2_before_meet / v2)
  (h4 : t1_after_meet = 3 / 4)
  (h5 : t2_after_meet = 4 / 3)
  (h6 : d1_before_meet + v1 * t1_after_meet = d_after_meet)
  (h7 : d2_before_meet + v2 * t2_after_meet = d_after_meet)
  : d_after_meet = 7 := 
sorry

end NUMINAMATH_GPT_distance_A_to_B_is_7km_l1345_134564


namespace NUMINAMATH_GPT_product_of_consecutive_integers_is_perfect_square_l1345_134526

theorem product_of_consecutive_integers_is_perfect_square (n : ℤ) :
    n * (n + 1) * (n + 2) * (n + 3) + 1 = (n * (n + 3) + 1) ^ 2 :=
sorry

end NUMINAMATH_GPT_product_of_consecutive_integers_is_perfect_square_l1345_134526


namespace NUMINAMATH_GPT_constant_term_of_product_l1345_134545

-- Define the polynomials
def poly1 (x : ℝ) : ℝ := 2 * x^3 + 3 * x^2 + 7
def poly2 (x : ℝ) : ℝ := 4 * x^4 + 2 * x^2 + 10

-- Main statement: Prove that the constant term in the expansion of poly1 * poly2 is 70
theorem constant_term_of_product : (poly1 0) * (poly2 0) = 70 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_constant_term_of_product_l1345_134545


namespace NUMINAMATH_GPT_PointNegativeThreeTwo_l1345_134553

def isInSecondQuadrant (x y : ℝ) : Prop :=
  x < 0 ∧ y > 0

theorem PointNegativeThreeTwo:
  isInSecondQuadrant (-3) 2 := by
  sorry

end NUMINAMATH_GPT_PointNegativeThreeTwo_l1345_134553


namespace NUMINAMATH_GPT_find_a_b_extreme_values_l1345_134520

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := (1/3) * x^3 + a * x^2 + b * x - (2/3)

theorem find_a_b_extreme_values : 
  ∃ (a b : ℝ), 
    (a = -2) ∧ 
    (b = 3) ∧ 
    (f 1 (-2) 3 = 2/3) ∧ 
    (f 3 (-2) 3 = -2/3) :=
by
  sorry

end NUMINAMATH_GPT_find_a_b_extreme_values_l1345_134520


namespace NUMINAMATH_GPT_percentage_of_a_l1345_134574

theorem percentage_of_a (x a : ℝ) (paise_in_rupee : ℝ := 100) (a_value : a = 160 * paise_in_rupee) (h : (x / 100) * a = 80) : x = 0.5 :=
by sorry

end NUMINAMATH_GPT_percentage_of_a_l1345_134574


namespace NUMINAMATH_GPT_find_b_l1345_134571

noncomputable def given_c := 3
noncomputable def given_C := Real.pi / 3
noncomputable def given_cos_C := 1 / 2
noncomputable def given_a (b : ℝ) := 2 * b

theorem find_b (b : ℝ) (h1 : given_c = 3) (h2 : given_cos_C = Real.cos (given_C)) (h3 : given_a b = 2 * b) : b = Real.sqrt 3 := 
by
  sorry

end NUMINAMATH_GPT_find_b_l1345_134571


namespace NUMINAMATH_GPT_smallest_positive_integer_l1345_134547

-- We define the integers 3003 and 55555 as given in the conditions
def a : ℤ := 3003
def b : ℤ := 55555

-- The main theorem stating the smallest positive integer that can be written in the form ax + by is 1
theorem smallest_positive_integer (m n : ℤ) : ∃ m n : ℤ, a * m + b * n = 1 :=
by
  -- We need not provide the proof steps here, just state it
  sorry

end NUMINAMATH_GPT_smallest_positive_integer_l1345_134547


namespace NUMINAMATH_GPT_find_number_l1345_134546

theorem find_number (x : ℕ) (h : 5 + x = 20) : x = 15 :=
sorry

end NUMINAMATH_GPT_find_number_l1345_134546


namespace NUMINAMATH_GPT_no_integer_pairs_satisfy_equation_l1345_134579

def equation_satisfaction (m n : ℤ) : Prop :=
  m^3 + 3 * m^2 + 2 * m = 8 * n^3 + 12 * n^2 + 6 * n + 1

theorem no_integer_pairs_satisfy_equation :
  ¬ ∃ (m n : ℤ), equation_satisfaction m n :=
by
  sorry

end NUMINAMATH_GPT_no_integer_pairs_satisfy_equation_l1345_134579


namespace NUMINAMATH_GPT_nick_paints_wall_in_fraction_l1345_134514

theorem nick_paints_wall_in_fraction (nick_paint_time wall_paint_time : ℕ) (h1 : wall_paint_time = 60) (h2 : nick_paint_time = 12) : (nick_paint_time * 1 / wall_paint_time = 1 / 5) :=
by
  sorry

end NUMINAMATH_GPT_nick_paints_wall_in_fraction_l1345_134514


namespace NUMINAMATH_GPT_max_GREECE_val_l1345_134591

variables (V E R I A G C : ℕ)
noncomputable def verify : Prop :=
  (V * 100 + E * 10 + R - (I * 10 + A)) = G^(R^E) * (G * 100 + R * 10 + E + E * 100 + C * 10 + E) ∧
  G ≠ 0 ∧ E ≠ 0 ∧ V ≠ 0 ∧ I ≠ 0 ∧
  V ≠ E ∧ V ≠ R ∧ V ≠ I ∧ V ≠ A ∧ V ≠ G ∧ V ≠ C ∧
  E ≠ R ∧ E ≠ I ∧ E ≠ A ∧ E ≠ G ∧ E ≠ C ∧
  R ≠ I ∧ R ≠ A ∧ R ≠ G ∧ R ≠ C ∧
  I ≠ A ∧ I ≠ G ∧ I ≠ C ∧
  A ≠ G ∧ A ≠ C ∧
  G ≠ C

theorem max_GREECE_val : ∃ V E R I A G C : ℕ, verify V E R I A G C ∧ (G * 100000 + R * 10000 + E * 1000 + E * 100 + C * 10 + E = 196646) :=
sorry

end NUMINAMATH_GPT_max_GREECE_val_l1345_134591


namespace NUMINAMATH_GPT_cost_of_pencil_pen_eraser_l1345_134525

variables {p q r : ℝ}

theorem cost_of_pencil_pen_eraser 
  (h1 : 4 * p + 3 * q + r = 5.40)
  (h2 : 2 * p + 2 * q + 2 * r = 4.60) : 
  p + 2 * q + 3 * r = 4.60 := 
by sorry

end NUMINAMATH_GPT_cost_of_pencil_pen_eraser_l1345_134525


namespace NUMINAMATH_GPT_find_number_of_Persians_l1345_134502

variable (P : ℕ)  -- Number of Persian cats Jamie owns
variable (M : ℕ := 2)  -- Number of Maine Coons Jamie owns (given by conditions)
variable (G_P : ℕ := P / 2)  -- Number of Persian cats Gordon owns, which is half of Jamie's
variable (G_M : ℕ := M + 1)  -- Number of Maine Coons Gordon owns, one more than Jamie's
variable (H_P : ℕ := 0)  -- Number of Persian cats Hawkeye owns, which is 0
variable (H_M : ℕ := G_M - 1)  -- Number of Maine Coons Hawkeye owns, one less than Gordon's

theorem find_number_of_Persians (sum_cats : P + M + G_P + G_M + H_P + H_M = 13) : 
  P = 4 :=
by
  -- Proof can be filled in here
  sorry

end NUMINAMATH_GPT_find_number_of_Persians_l1345_134502


namespace NUMINAMATH_GPT_avery_work_time_l1345_134542

theorem avery_work_time :
  ∀ (t : ℝ),
    (1/2 * t + 1/4 * 1 = 1) → t = 1 :=
by
  intros t h
  sorry

end NUMINAMATH_GPT_avery_work_time_l1345_134542


namespace NUMINAMATH_GPT_cost_per_ounce_l1345_134532

theorem cost_per_ounce (total_cost : ℕ) (num_ounces : ℕ) (h1 : total_cost = 84) (h2 : num_ounces = 12) : (total_cost / num_ounces) = 7 :=
by
  sorry

end NUMINAMATH_GPT_cost_per_ounce_l1345_134532


namespace NUMINAMATH_GPT_problem1_problem2_l1345_134517

-- Definitions of the three conditions given
def condition1 (x y : Nat) : Prop := x > y
def condition2 (y z : Nat) : Prop := y > z
def condition3 (x z : Nat) : Prop := 2 * z > x

-- Problem 1: If the number of teachers is 4, prove the maximum number of female students is 6.
theorem problem1 (z : Nat) (hz : z = 4) : ∃ y : Nat, (∀ x : Nat, condition1 x y → condition2 y z → condition3 x z) ∧ y = 6 :=
by
  sorry

-- Problem 2: Prove the minimum number of people in the group is 12.
theorem problem2 : ∃ z x y : Nat, (condition1 x y ∧ condition2 y z ∧ condition3 x z ∧ z < y ∧ y < x ∧ x < 2 * z) ∧ z = 3 ∧ x = 5 ∧ y = 4 ∧ x + y + z = 12 :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1345_134517


namespace NUMINAMATH_GPT_opposite_of_neg_three_l1345_134583

def opposite (x : Int) : Int := -x

theorem opposite_of_neg_three : opposite (-3) = 3 := by
  -- To be proven using Lean
  sorry

end NUMINAMATH_GPT_opposite_of_neg_three_l1345_134583


namespace NUMINAMATH_GPT_not_possible_for_runners_in_front_l1345_134534

noncomputable def runnerInFrontAtAnyMoment 
  (track_length : ℝ)
  (stands_length : ℝ)
  (runners_speeds : Fin 10 → ℝ) : Prop := 
  ∀ t : ℝ, ∃ i : Fin 10, 
  ∃ n : ℤ, 
  (runners_speeds i * t - n * track_length) % track_length ≤ stands_length

theorem not_possible_for_runners_in_front 
  (track_length stands_length : ℝ)
  (runners_speeds : Fin 10 → ℝ) 
  (h_track : track_length = 1)
  (h_stands : stands_length = 0.1)
  (h_speeds : ∀ i : Fin 10, 20 + i = runners_speeds i) : 
  ¬ runnerInFrontAtAnyMoment track_length stands_length runners_speeds :=
sorry

end NUMINAMATH_GPT_not_possible_for_runners_in_front_l1345_134534


namespace NUMINAMATH_GPT_fifi_pink_hangers_l1345_134573

theorem fifi_pink_hangers :
  ∀ (g b y p : ℕ), 
  g = 4 →
  b = g - 1 →
  y = b - 1 →
  16 = g + b + y + p →
  p = 7 :=
by
  intros
  sorry

end NUMINAMATH_GPT_fifi_pink_hangers_l1345_134573


namespace NUMINAMATH_GPT_elvis_recording_time_l1345_134533

theorem elvis_recording_time :
  ∀ (total_studio_time writing_time_per_song editing_time number_of_songs : ℕ),
  total_studio_time = 300 →
  writing_time_per_song = 15 →
  editing_time = 30 →
  number_of_songs = 10 →
  (total_studio_time - (number_of_songs * writing_time_per_song + editing_time)) / number_of_songs = 12 :=
by
  intros total_studio_time writing_time_per_song editing_time number_of_songs
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_elvis_recording_time_l1345_134533


namespace NUMINAMATH_GPT_total_consumer_installment_credit_l1345_134555

-- Conditions
def auto_instalment_credit (C : ℝ) : ℝ := 0.2 * C
def auto_finance_extends_1_third (auto_installment : ℝ) : ℝ := 57
def student_loans (C : ℝ) : ℝ := 0.15 * C
def credit_card_debt (C : ℝ) (auto_installment : ℝ) : ℝ := 0.25 * C
def other_loans (C : ℝ) : ℝ := 0.4 * C

-- Correct Answer
theorem total_consumer_installment_credit (C : ℝ) :
  auto_instalment_credit C / 3 = auto_finance_extends_1_third (auto_instalment_credit C) ∧
  student_loans C = 80 ∧
  credit_card_debt C (auto_instalment_credit C) = auto_instalment_credit C + 100 ∧
  credit_card_debt C (auto_instalment_credit C) = 271 →
  C = 1084 := 
by
  sorry

end NUMINAMATH_GPT_total_consumer_installment_credit_l1345_134555


namespace NUMINAMATH_GPT_find_y_values_l1345_134507

theorem find_y_values (x : ℝ) (h : x^2 + 9 * (x / (x - 3))^2 = 90) :
  ∃ y, (y = 0 ∨ y = 144 ∨ y = -24) ∧ y = (x - 3)^2 * (x + 4) / (2 * x - 5) :=
by sorry

end NUMINAMATH_GPT_find_y_values_l1345_134507


namespace NUMINAMATH_GPT_smallest_base_l1345_134508

theorem smallest_base (k b : ℕ) (h_k : k = 6) : 64 ^ k > b ^ 16 ↔ b < 5 :=
by
  have h1 : 64 ^ k = 2 ^ (6 * k) := by sorry
  have h2 : 2 ^ (6 * k) > b ^ 16 := by sorry
  exact sorry

end NUMINAMATH_GPT_smallest_base_l1345_134508


namespace NUMINAMATH_GPT_new_students_l1345_134599

theorem new_students (S_i : ℕ) (L : ℕ) (S_f : ℕ) (N : ℕ) 
  (h₁ : S_i = 11) 
  (h₂ : L = 6) 
  (h₃ : S_f = 47) 
  (h₄ : S_f = S_i - L + N) : 
  N = 42 :=
by 
  rw [h₁, h₂, h₃] at h₄
  sorry

end NUMINAMATH_GPT_new_students_l1345_134599


namespace NUMINAMATH_GPT_value_of_a_plus_d_l1345_134597

variable {R : Type} [LinearOrderedField R]
variables {a b c d : R}

theorem value_of_a_plus_d (h1 : a + b = 16) (h2 : b + c = 9) (h3 : c + d = 3) : a + d = 13 := by
  sorry

end NUMINAMATH_GPT_value_of_a_plus_d_l1345_134597


namespace NUMINAMATH_GPT_bridge_length_is_correct_l1345_134527

noncomputable def length_of_bridge (length_of_train : ℝ) (speed_kmh : ℝ) (time_sec : ℝ) : ℝ :=
  let speed_ms := speed_kmh * 1000 / 3600
  let total_distance := speed_ms * time_sec
  total_distance - length_of_train

theorem bridge_length_is_correct :
  length_of_bridge 160 45 30 = 215 := by
  sorry

end NUMINAMATH_GPT_bridge_length_is_correct_l1345_134527


namespace NUMINAMATH_GPT_car_value_reduction_l1345_134569

/-- Jocelyn bought a car 3 years ago at $4000. 
If the car's value has reduced by 30%, calculate the current value of the car. 
Prove that it is equal to $2800. -/
theorem car_value_reduction (initial_value : ℝ) (reduction_percentage : ℝ) (current_value : ℝ) 
  (h_initial : initial_value = 4000)
  (h_reduction : reduction_percentage = 30)
  (h_current : current_value = initial_value - (reduction_percentage / 100) * initial_value) :
  current_value = 2800 :=
by
  -- Formal proof goes here
  sorry

end NUMINAMATH_GPT_car_value_reduction_l1345_134569


namespace NUMINAMATH_GPT_dryer_runtime_per_dryer_l1345_134518

-- Definitions for the given conditions
def washer_cost : ℝ := 4
def dryer_cost_per_10min : ℝ := 0.25
def loads_of_laundry : ℕ := 2
def num_dryers : ℕ := 3
def total_spent : ℝ := 11

-- Statement to prove
theorem dryer_runtime_per_dryer : 
  (2 * washer_cost + ((total_spent - 2 * washer_cost) / dryer_cost_per_10min) * 10) / num_dryers = 40 :=
by
  sorry

end NUMINAMATH_GPT_dryer_runtime_per_dryer_l1345_134518


namespace NUMINAMATH_GPT_min_value_proof_l1345_134580

noncomputable def min_value_expression (α β : ℝ) : ℝ :=
  (3 * Real.cos α + 4 * Real.sin β - 7)^2 + (3 * Real.sin α + 4 * Real.cos β - 12)^2

theorem min_value_proof :
  ∃ α β : ℝ, min_value_expression α β = 48 := by
  sorry

end NUMINAMATH_GPT_min_value_proof_l1345_134580


namespace NUMINAMATH_GPT_geometric_sequence_common_ratio_l1345_134572

theorem geometric_sequence_common_ratio :
  (∃ q : ℝ, 1 + q + q^2 = 13 ∧ (q = 3 ∨ q = -4)) :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_common_ratio_l1345_134572


namespace NUMINAMATH_GPT_working_mom_hours_at_work_l1345_134523

-- Definitions corresponding to the conditions
def hours_awake : ℕ := 16
def work_percentage : ℝ := 0.50

-- The theorem to be proved
theorem working_mom_hours_at_work : work_percentage * hours_awake = 8 :=
by sorry

end NUMINAMATH_GPT_working_mom_hours_at_work_l1345_134523


namespace NUMINAMATH_GPT_average_monthly_growth_rate_proof_profit_in_may_proof_l1345_134544

theorem average_monthly_growth_rate_proof :
  ∃ r : ℝ, 2400 * (1 + r)^2 = 3456 ∧ r = 0.2 := sorry

theorem profit_in_may_proof (r : ℝ) (h_r : r = 0.2) :
  3456 * (1 + r) = 4147.2 := sorry

end NUMINAMATH_GPT_average_monthly_growth_rate_proof_profit_in_may_proof_l1345_134544


namespace NUMINAMATH_GPT_convert_spherical_coordinates_l1345_134557

theorem convert_spherical_coordinates (
  ρ θ φ : ℝ
) (h1 : ρ = 5) (h2 : θ = 3 * Real.pi / 4) (h3 : φ = 9 * Real.pi / 4) : 
ρ = 5 ∧ 0 ≤ 7 * Real.pi / 4 ∧ 7 * Real.pi / 4 < 2 * Real.pi ∧ 0 ≤ Real.pi / 4 ∧ Real.pi / 4 ≤ Real.pi :=
by
  sorry

end NUMINAMATH_GPT_convert_spherical_coordinates_l1345_134557


namespace NUMINAMATH_GPT_evaluation_of_expression_l1345_134550

theorem evaluation_of_expression : (3^2 - 2^2 + 1^2) = 6 :=
by
  sorry

end NUMINAMATH_GPT_evaluation_of_expression_l1345_134550


namespace NUMINAMATH_GPT_total_students_sum_is_90_l1345_134562

theorem total_students_sum_is_90:
  ∃ (x y z : ℕ), 
  (80 * x - 100 = 92 * (x - 5)) ∧
  (75 * y - 150 = 85 * (y - 6)) ∧
  (70 * z - 120 = 78 * (z - 4)) ∧
  (x + y + z = 90) :=
by
  sorry

end NUMINAMATH_GPT_total_students_sum_is_90_l1345_134562


namespace NUMINAMATH_GPT_units_digit_sum_l1345_134577

theorem units_digit_sum (n : ℕ) (h : n > 0) : (35^n % 10) + (93^45 % 10) = 8 :=
by
  -- Since the units digit of 35^n is always 5 
  have h1 : 35^n % 10 = 5 := sorry
  -- Since the units digit of 93^45 is 3 (since 45 mod 4 = 1 and the pattern repeats every 4),
  have h2 : 93^45 % 10 = 3 := sorry
  -- Therefore, combining the units digits
  calc
    (35^n % 10) + (93^45 % 10)
    = 5 + 3 := by rw [h1, h2]
    _ = 8 := by norm_num

end NUMINAMATH_GPT_units_digit_sum_l1345_134577


namespace NUMINAMATH_GPT_ratio_a_to_c_l1345_134539

variable (a b c : ℚ)

theorem ratio_a_to_c (h1 : a / b = 7 / 3) (h2 : b / c = 1 / 5) : a / c = 7 / 15 := 
sorry

end NUMINAMATH_GPT_ratio_a_to_c_l1345_134539


namespace NUMINAMATH_GPT_remainder_when_divided_by_30_l1345_134538

theorem remainder_when_divided_by_30 (y : ℤ)
  (h1 : 4 + y ≡ 9 [ZMOD 8])
  (h2 : 6 + y ≡ 8 [ZMOD 27])
  (h3 : 8 + y ≡ 27 [ZMOD 125]) :
  y ≡ 4 [ZMOD 30] :=
sorry

end NUMINAMATH_GPT_remainder_when_divided_by_30_l1345_134538


namespace NUMINAMATH_GPT_ratio_of_surface_areas_l1345_134521

-- Definitions based on conditions
def side_length_ratio (a b : ℝ) : Prop := b = 6 * a
def surface_area (a : ℝ) : ℝ := 6 * a ^ 2

-- Theorem statement
theorem ratio_of_surface_areas (a b : ℝ) (h : side_length_ratio a b) :
  (surface_area b) / (surface_area a) = 36 := by
  sorry

end NUMINAMATH_GPT_ratio_of_surface_areas_l1345_134521


namespace NUMINAMATH_GPT_geometric_common_ratio_arithmetic_sequence_l1345_134590

theorem geometric_common_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) (h1 : S 3 = a 1 * (1 - q^3) / (1 - q)) (h2 : S 3 = 3 * a 1) :
  q = 2 ∨ q^3 = - (1 / 2) := by
  sorry

theorem arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) (h : S 3 = a 1 * (1 - q^3) / (1 - q))
  (h3 : 2 * S 9 = S 3 + S 6) (h4 : q ≠ 1) :
  a 2 + a 5 = 2 * a 8 := by
  sorry

end NUMINAMATH_GPT_geometric_common_ratio_arithmetic_sequence_l1345_134590


namespace NUMINAMATH_GPT_remaining_amount_to_be_paid_l1345_134516

-- Define the conditions
def deposit_percentage : ℚ := 10 / 100
def deposit_amount : ℚ := 80

-- Define the total purchase price based on the conditions
def total_price : ℚ := deposit_amount / deposit_percentage

-- Define the remaining amount to be paid
def remaining_amount : ℚ := total_price - deposit_amount

-- State the theorem
theorem remaining_amount_to_be_paid : remaining_amount = 720 := by
  sorry

end NUMINAMATH_GPT_remaining_amount_to_be_paid_l1345_134516


namespace NUMINAMATH_GPT_unknown_rate_of_two_towels_l1345_134535

theorem unknown_rate_of_two_towels :
  let x := 325
  let known_cost := (3 * 100) + (5 * 150)
  let total_average_price := 170
  let number_of_towels := 10
  known_cost + (2 * x) = total_average_price * number_of_towels :=
by
  let x := 325
  let known_cost := (3 * 100) + (5 * 150)
  let total_average_price := 170
  let number_of_towels := 10
  show known_cost + (2 * x) = total_average_price * number_of_towels
  sorry

end NUMINAMATH_GPT_unknown_rate_of_two_towels_l1345_134535

import Mathlib

namespace NUMINAMATH_GPT_find_value_of_alpha_beta_plus_alpha_plus_beta_l249_24959

variable (α β : ℝ)

theorem find_value_of_alpha_beta_plus_alpha_plus_beta
  (hα : α^2 + α - 1 = 0)
  (hβ : β^2 + β - 1 = 0)
  (hαβ : α ≠ β) :
  α * β + α + β = -2 := 
by
  sorry

end NUMINAMATH_GPT_find_value_of_alpha_beta_plus_alpha_plus_beta_l249_24959


namespace NUMINAMATH_GPT_crocodile_length_in_meters_l249_24956

-- Definitions based on conditions
def ken_to_cm : ℕ := 180
def shaku_to_cm : ℕ := 30
def ken_to_shaku : ℕ := 6
def cm_to_m : ℕ := 100

-- Lengths given in the problem expressed in ken
def head_to_tail_in_ken (L : ℚ) : Prop := 3 * L = 10
def tail_to_head_in_ken (L : ℚ) : Prop := L = (3 + (2 / ken_to_shaku : ℚ))

-- Final length conversion to meters
def length_in_m (L : ℚ) : ℚ := L * ken_to_cm / cm_to_m

-- The length of the crocodile in meters
theorem crocodile_length_in_meters (L : ℚ) : head_to_tail_in_ken L → tail_to_head_in_ken L → length_in_m L = 6 :=
by
  intros _ _
  sorry

end NUMINAMATH_GPT_crocodile_length_in_meters_l249_24956


namespace NUMINAMATH_GPT_parabola_vertex_trajectory_eq_l249_24980

noncomputable def parabola_vertex_trajectory : Prop :=
  ∀ (m : ℝ), ∃ (x y : ℝ), (y = 2 * m) ∧ (x = -m^2) ∧ (y - 4 * x - 4 * m * y = 0)

theorem parabola_vertex_trajectory_eq :
  (∀ x y : ℝ, (∃ m : ℝ, y = 2 * m ∧ x = -m^2) → y^2 = -4 * x) :=
by
  sorry

end NUMINAMATH_GPT_parabola_vertex_trajectory_eq_l249_24980


namespace NUMINAMATH_GPT_company_bought_14_02_tons_l249_24964

noncomputable def gravel := 5.91
noncomputable def sand := 8.11
noncomputable def total_material := gravel + sand

theorem company_bought_14_02_tons : total_material = 14.02 :=
by 
  sorry

end NUMINAMATH_GPT_company_bought_14_02_tons_l249_24964


namespace NUMINAMATH_GPT_simplify_expression_l249_24917

theorem simplify_expression (z y : ℝ) :
  (4 - 5 * z + 2 * y) - (6 + 7 * z - 3 * y) = -2 - 12 * z + 5 * y :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l249_24917


namespace NUMINAMATH_GPT_solve_for_x_l249_24971

variables {A B C m n x : ℝ}

-- Existing conditions
def A_rate_condition : A = (B + C) / m := sorry
def B_rate_condition : B = (C + A) / n := sorry
def C_rate_condition : C = (A + B) / x := sorry

-- The theorem to be proven
theorem solve_for_x (A_rate_condition : A = (B + C) / m)
                    (B_rate_condition : B = (C + A) / n)
                    (C_rate_condition : C = (A + B) / x)
                    : x = (2 + m + n) / (m * n - 1) := by
  sorry

end NUMINAMATH_GPT_solve_for_x_l249_24971


namespace NUMINAMATH_GPT_find_common_ratio_l249_24904

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a 1 * q ^ n

def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, S n = a 1 * (1 - q ^ n) / (1 - q)

variables (a : ℕ → ℝ) (S : ℕ → ℝ) (a1 q : ℝ)

noncomputable def a_5_condition : Prop :=
  a 5 = 2 * S 4 + 3

noncomputable def a_6_condition : Prop :=
  a 6 = 2 * S 5 + 3

theorem find_common_ratio (h1 : a_5_condition a S) (h2 : a_6_condition a S)
  (hg : geometric_sequence a q) (hs : sum_of_first_n_terms a S q) :
  q = 3 :=
sorry

end NUMINAMATH_GPT_find_common_ratio_l249_24904


namespace NUMINAMATH_GPT_find_ab_l249_24990

theorem find_ab (a b : ℝ) (h1 : a + b = 4) (h2 : a^3 + b^3 = 100) : a * b = -3 :=
by
sorry

end NUMINAMATH_GPT_find_ab_l249_24990


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l249_24936

theorem necessary_but_not_sufficient (x y : ℝ) : 
  (x - y > -1) → (x^3 + x > x^2 * y + y) → 
  ∃ z : ℝ, z - y > -1 ∧ ¬ (z^3 + z > z^2 * y + y) :=
sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l249_24936


namespace NUMINAMATH_GPT_willie_stickers_l249_24989

theorem willie_stickers (initial_stickers : ℕ) (given_stickers : ℕ) (final_stickers : ℕ) 
  (h1 : initial_stickers = 124) 
  (h2 : given_stickers = 43) 
  (h3 : final_stickers = initial_stickers - given_stickers) :
  final_stickers = 81 :=
sorry

end NUMINAMATH_GPT_willie_stickers_l249_24989


namespace NUMINAMATH_GPT_total_chickens_l249_24911

   def number_of_hens := 12
   def hens_to_roosters_ratio := 3
   def chicks_per_hen := 5

   theorem total_chickens (h : number_of_hens = 12)
                          (r : hens_to_roosters_ratio = 3)
                          (c : chicks_per_hen = 5) :
     number_of_hens + (number_of_hens / hens_to_roosters_ratio) + (number_of_hens * chicks_per_hen) = 76 :=
   by
     sorry
   
end NUMINAMATH_GPT_total_chickens_l249_24911


namespace NUMINAMATH_GPT_Yanna_apples_l249_24931

def total_apples_bought (given_to_zenny : ℕ) (given_to_andrea : ℕ) (kept : ℕ) : ℕ :=
  given_to_zenny + given_to_andrea + kept

theorem Yanna_apples {given_to_zenny given_to_andrea kept total : ℕ}:
  given_to_zenny = 18 →
  given_to_andrea = 6 →
  kept = 36 →
  total_apples_bought given_to_zenny given_to_andrea kept = 60 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  rfl

end NUMINAMATH_GPT_Yanna_apples_l249_24931


namespace NUMINAMATH_GPT_ratio_of_volumes_l249_24994

theorem ratio_of_volumes (s : ℝ) (hs : s > 0) :
  let r_s := s / 2
  let r_c := s / 2
  let V_sphere := (4 / 3) * π * (r_s ^ 3)
  let V_cylinder := π * (r_c ^ 2) * s
  let V_total := V_sphere + V_cylinder
  let V_cube := s ^ 3
  V_total / V_cube = (5 * π) / 12 := by {
    -- Given the conditions and expressions
    sorry
  }

end NUMINAMATH_GPT_ratio_of_volumes_l249_24994


namespace NUMINAMATH_GPT_find_a_l249_24974

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x > 0 then x * 2^(x + a) - 1 else - (x * 2^(-x + a) - 1)

theorem find_a (a : ℝ) (h_odd: ∀ x : ℝ, f x a = -f (-x) a)
  (h_pos : ∀ x : ℝ, x > 0 → f x a = x * 2^(x + a) - 1)
  (h_neg : f (-1) a = 3 / 4) :
  a = -3 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l249_24974


namespace NUMINAMATH_GPT_smallest_rectangle_area_l249_24986

-- Definitions based on conditions
def diameter : ℝ := 10
def length : ℝ := diameter
def width : ℝ := diameter + 2

-- Theorem statement
theorem smallest_rectangle_area : (length * width) = 120 :=
by
  -- The proof would go here, but we provide sorry for now
  sorry

end NUMINAMATH_GPT_smallest_rectangle_area_l249_24986


namespace NUMINAMATH_GPT_positive_quadratic_if_and_only_if_l249_24922

variable (a : ℝ)
def p (x : ℝ) : ℝ := a * x^2 + 2 * x + 1

theorem positive_quadratic_if_and_only_if (h : ∀ x : ℝ, p a x > 0) : a > 1 := sorry

end NUMINAMATH_GPT_positive_quadratic_if_and_only_if_l249_24922


namespace NUMINAMATH_GPT_shape_is_cylinder_l249_24906

def is_cylinder (c : ℝ) (r θ z : ℝ) : Prop :=
  c > 0 ∧ r = c ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧ True

theorem shape_is_cylinder (c : ℝ) (r θ z : ℝ) (h : c > 0) :
  is_cylinder c r θ z :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_shape_is_cylinder_l249_24906


namespace NUMINAMATH_GPT_solve_inequality_l249_24996

theorem solve_inequality (a x : ℝ) :
  ((x - a) * (x - 2 * a) < 0) ↔ 
  ((a < 0 ∧ 2 * a < x ∧ x < a) ∨ (a = 0 ∧ false) ∨ (a > 0 ∧ a < x ∧ x < 2 * a)) :=
by sorry

end NUMINAMATH_GPT_solve_inequality_l249_24996


namespace NUMINAMATH_GPT_solve_inequality_l249_24978

noncomputable def g (x : ℝ) : ℝ := (3 * x - 8) * (x - 2) / (x - 1)

theorem solve_inequality : 
  { x : ℝ | g x ≥ 0 } = { x : ℝ | x < 1 } ∪ { x : ℝ | x ≥ 2 } :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l249_24978


namespace NUMINAMATH_GPT_find_other_number_l249_24982

theorem find_other_number (x y : ℕ) (h1 : x + y = 72) (h2 : y = x + 12) (h3 : y = 42) : x = 30 := by
  sorry

end NUMINAMATH_GPT_find_other_number_l249_24982


namespace NUMINAMATH_GPT_gain_percentage_l249_24933

theorem gain_percentage (x : ℝ) (CP : ℝ := 50 * x) (SP : ℝ := 60 * x) (Profit : ℝ := 10 * x) :
  ((Profit / CP) * 100) = 20 := 
by
  sorry

end NUMINAMATH_GPT_gain_percentage_l249_24933


namespace NUMINAMATH_GPT_prove_pqrstu_eq_416_l249_24985

-- Define the condition 1728 * x^4 + 64 = (p * x^3 + q * x^2 + r * x + s) * (t * x + u) + v
def condition (p q r s t u v : ℤ) (x : ℤ) : Prop :=
  1728 * x^4 + 64 = (p * x^3 + q * x^2 + r * x + s) * (t * x + u) + v

-- State the theorem to prove p^2 + q^2 + r^2 + s^2 + t^2 + u^2 + v^2 = 416
theorem prove_pqrstu_eq_416 (p q r s t u v : ℤ) (h : ∀ x, condition p q r s t u v x) : 
  p^2 + q^2 + r^2 + s^2 + t^2 + u^2 + v^2 = 416 :=
sorry

end NUMINAMATH_GPT_prove_pqrstu_eq_416_l249_24985


namespace NUMINAMATH_GPT_arithmetic_sum_S9_l249_24920

variable {a : ℕ → ℝ} -- Define the arithmetic sequence
variable (S : ℕ → ℝ) -- Define the sum of the first n terms
variable (d : ℝ) -- Define the common difference
variable (a_1 : ℝ) -- Define the first term of the sequence

-- Assume the arithmetic sequence properties
axiom arith_seq_def : ∀ n, a (n + 1) = a_1 + n * d

-- Define the sum of the first n terms
axiom sum_first_n_terms : ∀ n, S n = n / 2 * (2 * a_1 + (n - 1) * d)

-- Given condition
axiom given_condition : a 1 + a 7 = 15 - a 4

theorem arithmetic_sum_S9 : S 9 = 45 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_arithmetic_sum_S9_l249_24920


namespace NUMINAMATH_GPT_parabola_intersection_radius_sqr_l249_24919

theorem parabola_intersection_radius_sqr {x y : ℝ} :
  (y = (x - 2)^2) →
  (x - 3 = (y + 2)^2) →
  ∃ r, r^2 = 9 / 2 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_parabola_intersection_radius_sqr_l249_24919


namespace NUMINAMATH_GPT_cars_people_count_l249_24901

-- Define the problem conditions
def cars_people_conditions (x y : ℕ) : Prop :=
  y = 3 * (x - 2) ∧ y = 2 * x + 9

-- Define the theorem stating that there exist numbers of cars and people that satisfy the conditions
theorem cars_people_count (x y : ℕ) : cars_people_conditions x y ↔ (y = 3 * (x - 2) ∧ y = 2 * x + 9) := by
  -- skip the proof
  sorry

end NUMINAMATH_GPT_cars_people_count_l249_24901


namespace NUMINAMATH_GPT_min_sum_squares_l249_24939

noncomputable def distances (P : ℝ) : ℝ :=
  let AP := P
  let BP := |P - 1|
  let CP := |P - 2|
  let DP := |P - 5|
  let EP := |P - 13|
  AP^2 + BP^2 + CP^2 + DP^2 + EP^2

theorem min_sum_squares : ∀ P : ℝ, distances P ≥ 88.2 :=
by
  sorry

end NUMINAMATH_GPT_min_sum_squares_l249_24939


namespace NUMINAMATH_GPT_A_superset_B_l249_24910

open Set

variable (N : Set ℕ)
def A : Set ℕ := {x | ∃ n ∈ N, x = 2 * n}
def B : Set ℕ := {x | ∃ n ∈ N, x = 4 * n}

theorem A_superset_B : A N ⊇ B N :=
by
  -- Proof to be written
  sorry

end NUMINAMATH_GPT_A_superset_B_l249_24910


namespace NUMINAMATH_GPT_repeating_decimal_equals_fraction_l249_24976

theorem repeating_decimal_equals_fraction : 
  let a := 58 / 100
  let r := 1 / 100
  let S := a / (1 - r)
  S = (58 : ℚ) / 99 :=
by
  sorry

end NUMINAMATH_GPT_repeating_decimal_equals_fraction_l249_24976


namespace NUMINAMATH_GPT_geometric_sequence_general_formula_l249_24935

theorem geometric_sequence_general_formula (a : ℕ → ℝ) (h_pos : ∀ n, 0 < a n)
  (h1 : a 1 = 2)
  (h_rec : ∀ n, (a (n + 2))^2 + 4 * (a n)^2 = 4 * (a (n + 1))^2) :
  ∀ n, a n = 2^(n + 1) / 2 := 
sorry

end NUMINAMATH_GPT_geometric_sequence_general_formula_l249_24935


namespace NUMINAMATH_GPT_jensen_meetings_percentage_l249_24937

theorem jensen_meetings_percentage :
  ∃ (first second third total_work_day total_meeting_time : ℕ),
    total_work_day = 600 ∧
    first = 35 ∧
    second = 2 * first ∧
    third = first + second ∧
    total_meeting_time = first + second + third ∧
    (total_meeting_time * 100) / total_work_day = 35 := sorry

end NUMINAMATH_GPT_jensen_meetings_percentage_l249_24937


namespace NUMINAMATH_GPT_algebraic_expression_value_l249_24965

theorem algebraic_expression_value (x : ℝ) (h : 3 * x^2 - 4 * x = 6): 6 * x^2 - 8 * x - 9 = 3 :=
by sorry

end NUMINAMATH_GPT_algebraic_expression_value_l249_24965


namespace NUMINAMATH_GPT_triangle_side_relation_triangle_perimeter_l249_24900

theorem triangle_side_relation (a b c : ℝ) (A B C : ℝ)
  (h1 : a / (Real.sin A) = b / (Real.sin B)) (h2 : a / (Real.sin A) = c / (Real.sin C))
  (h3 : Real.sin C * Real.sin (A - B) = Real.sin B * Real.sin (C - A)) :
  2 * a ^ 2 = b ^ 2 + c ^ 2 := sorry

theorem triangle_perimeter (a b c : ℝ) (A : ℝ) (hcosA : Real.cos A = 25 / 31)
  (h1 : a / (Real.sin A) = b / (Real.sin B)) (h2 : a / (Real.sin A) = c / (Real.sin C))
  (h3 : Real.sin C * Real.sin (A - B) = Real.sin B * Real.sin (C - A)) (ha : a = 5) :
  a + b + c = 14 := sorry

end NUMINAMATH_GPT_triangle_side_relation_triangle_perimeter_l249_24900


namespace NUMINAMATH_GPT_chance_Z_winning_l249_24947

-- Given conditions as Lean definitions
def p_x : ℚ := 1 / (3 + 1)
def p_y : ℚ := 3 / (2 + 3)
def p_z : ℚ := 1 - (p_x + p_y)

-- Theorem statement: Prove the equivalence of the winning ratio for Z
theorem chance_Z_winning : 
  p_z = 3 / (3 + 17) :=
by
  -- Since we include no proof, we use sorry to indicate it
  sorry

end NUMINAMATH_GPT_chance_Z_winning_l249_24947


namespace NUMINAMATH_GPT_correct_option_l249_24979

-- Conditions
def option_A (a : ℕ) : Prop := (a^5)^2 = a^7
def option_B (a : ℕ) : Prop := a + 2 * a = 3 * a^2
def option_C (a : ℕ) : Prop := (2 * a)^3 = 6 * a^3
def option_D (a : ℕ) : Prop := a^6 / a^2 = a^4

-- Theorem statement
theorem correct_option (a : ℕ) : ¬ option_A a ∧ ¬ option_B a ∧ ¬ option_C a ∧ option_D a := by
  sorry

end NUMINAMATH_GPT_correct_option_l249_24979


namespace NUMINAMATH_GPT_smallest_three_digit_multiple_of_13_l249_24970

theorem smallest_three_digit_multiple_of_13 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 13 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 13 = 0 → n ≤ m :=
⟨104, by sorry⟩

end NUMINAMATH_GPT_smallest_three_digit_multiple_of_13_l249_24970


namespace NUMINAMATH_GPT_find_coefficients_l249_24973

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Definitions based on conditions
def A' (A B : V) : V := (3 : ℝ) • (B - A) + A
def B' (B C : V) : V := (3 : ℝ) • (C - B) + C

-- The problem statement
theorem find_coefficients (A A' B B' : V) (p q r : ℝ) 
  (hB : B = (1/4 : ℝ) • A + (3/4 : ℝ) • A') 
  (hC : C = (1/4 : ℝ) • B + (3/4 : ℝ) • B') : 
  ∃ (p q r : ℝ), A = p • A' + q • B + r • B' ∧ p = 4/13 ∧ q = 12/13 ∧ r = 48/13 :=
sorry

end NUMINAMATH_GPT_find_coefficients_l249_24973


namespace NUMINAMATH_GPT_sum_of_coefficients_l249_24912

theorem sum_of_coefficients (p : ℕ) (hp : Nat.Prime p) (a b c : ℕ)
    (f : ℕ → ℕ) (hf : ∀ x, f x = a * x ^ 2 + b * x + c)
    (h_range : 0 < a ∧ a ≤ p ∧ 0 < b ∧ b ≤ p ∧ 0 < c ∧ c ≤ p)
    (h_div : ∀ x, x > 0 → p ∣ (f x)) : 
    a + b + c = 3 * p := 
sorry

end NUMINAMATH_GPT_sum_of_coefficients_l249_24912


namespace NUMINAMATH_GPT_average_of_added_numbers_l249_24972

theorem average_of_added_numbers (sum_twelve : ℕ) (new_sum : ℕ) (x y z : ℕ) 
  (h_sum_twelve : sum_twelve = 12 * 45) 
  (h_new_sum : new_sum = 15 * 60) 
  (h_addition : x + y + z = new_sum - sum_twelve) : 
  (x + y + z) / 3 = 120 :=
by 
  sorry

end NUMINAMATH_GPT_average_of_added_numbers_l249_24972


namespace NUMINAMATH_GPT_first_number_is_48_l249_24938

-- Definitions of the conditions
def ratio (A B : ℕ) := 8 * B = 9 * A
def lcm (A B : ℕ) := Nat.lcm A B = 432

-- The statement to prove
theorem first_number_is_48 (A B : ℕ) (h_ratio : ratio A B) (h_lcm : lcm A B) : A = 48 :=
by
  sorry

end NUMINAMATH_GPT_first_number_is_48_l249_24938


namespace NUMINAMATH_GPT_graphs_intersect_at_one_point_l249_24926

noncomputable def f (x : ℝ) : ℝ := 3 * Real.log x / Real.log 3
noncomputable def g (x : ℝ) : ℝ := Real.log (4 * x) / Real.log 2

theorem graphs_intersect_at_one_point : ∃! x, f x = g x :=
by {
  sorry
}

end NUMINAMATH_GPT_graphs_intersect_at_one_point_l249_24926


namespace NUMINAMATH_GPT_profit_percent_is_approx_6_point_35_l249_24930

noncomputable def selling_price : ℝ := 2552.36
noncomputable def cost_price : ℝ := 2400
noncomputable def profit_amount : ℝ := selling_price - cost_price
noncomputable def profit_percent : ℝ := (profit_amount / cost_price) * 100

theorem profit_percent_is_approx_6_point_35 : abs (profit_percent - 6.35) < 0.01 := sorry

end NUMINAMATH_GPT_profit_percent_is_approx_6_point_35_l249_24930


namespace NUMINAMATH_GPT_profit_23_percent_of_cost_price_l249_24913

-- Define the conditions
variable (C : ℝ) -- Cost price of the turtleneck sweaters
variable (C_nonneg : 0 ≤ C) -- Ensure cost price is non-negative

-- Definitions based on conditions
def SP1 (C : ℝ) : ℝ := 1.20 * C
def SP2 (SP1 : ℝ) : ℝ := 1.25 * SP1
def SPF (SP2 : ℝ) : ℝ := 0.82 * SP2

-- Define the profit calculation
def Profit (C : ℝ) : ℝ := (SPF (SP2 (SP1 C))) - C

-- Statement of the theorem
theorem profit_23_percent_of_cost_price (C : ℝ) (C_nonneg : 0 ≤ C):
  Profit C = 0.23 * C :=
by
  -- The actual proof would go here
  sorry

end NUMINAMATH_GPT_profit_23_percent_of_cost_price_l249_24913


namespace NUMINAMATH_GPT_prove_absolute_value_subtract_power_l249_24941

noncomputable def smallest_absolute_value : ℝ := 0

theorem prove_absolute_value_subtract_power (b : ℝ) 
  (h1 : smallest_absolute_value = 0) 
  (h2 : b * b = 1) : 
  (|smallest_absolute_value - 2| - b ^ 2023 = 1) 
  ∨ (|smallest_absolute_value - 2| - b ^ 2023 = 3) :=
sorry

end NUMINAMATH_GPT_prove_absolute_value_subtract_power_l249_24941


namespace NUMINAMATH_GPT_percent_chemical_a_in_mixture_l249_24969

-- Define the given problem parameters
def percent_chemical_a_in_solution_x : ℝ := 0.30
def percent_chemical_a_in_solution_y : ℝ := 0.40
def proportion_of_solution_x_in_mixture : ℝ := 0.80
def proportion_of_solution_y_in_mixture : ℝ := 1.0 - proportion_of_solution_x_in_mixture

-- Define what we need to prove: the percentage of chemical a in the mixture
theorem percent_chemical_a_in_mixture:
  (percent_chemical_a_in_solution_x * proportion_of_solution_x_in_mixture) + 
  (percent_chemical_a_in_solution_y * proportion_of_solution_y_in_mixture) = 0.32 
:= by sorry

end NUMINAMATH_GPT_percent_chemical_a_in_mixture_l249_24969


namespace NUMINAMATH_GPT_books_read_last_month_l249_24924

namespace BookReading

variable (W : ℕ) -- Number of books William read last month.

-- Conditions
axiom cond1 : ∃ B : ℕ, B = 3 * W -- Brad read thrice as many books as William did last month.
axiom cond2 : W = 2 * 8 -- This month, William read twice as much as Brad, who read 8 books.
axiom cond3 : ∃ (B_prev : ℕ) (B_curr : ℕ), B_prev = 3 * W ∧ B_curr = 8 ∧ W + 16 = B_prev + B_curr + 4 -- Total books equation

theorem books_read_last_month : W = 2 := by
  sorry

end BookReading

end NUMINAMATH_GPT_books_read_last_month_l249_24924


namespace NUMINAMATH_GPT_Sam_has_correct_amount_of_dimes_l249_24927

-- Definitions for initial values and transactions
def initial_dimes := 9
def dimes_from_dad := 7
def dimes_taken_by_mom := 3
def sets_from_sister := 4
def dimes_per_set := 2

-- Definition of the total dimes Sam has now
def total_dimes_now : Nat :=
  initial_dimes + dimes_from_dad - dimes_taken_by_mom + (sets_from_sister * dimes_per_set)

-- Proof statement
theorem Sam_has_correct_amount_of_dimes : total_dimes_now = 21 := by
  sorry

end NUMINAMATH_GPT_Sam_has_correct_amount_of_dimes_l249_24927


namespace NUMINAMATH_GPT_angle_same_terminal_side_l249_24940

theorem angle_same_terminal_side (k : ℤ) : ∃ k : ℤ, -330 = k * 360 + 30 :=
by
  use -1
  sorry

end NUMINAMATH_GPT_angle_same_terminal_side_l249_24940


namespace NUMINAMATH_GPT_jacob_age_proof_l249_24951

theorem jacob_age_proof
  (drew_age maya_age peter_age : ℕ)
  (john_age : ℕ := 30)
  (jacob_age : ℕ) :
  (drew_age = maya_age + 5) →
  (peter_age = drew_age + 4) →
  (john_age = 30 ∧ john_age = 2 * maya_age) →
  (jacob_age + 2 = (peter_age + 2) / 2) →
  jacob_age = 11 :=
by
  sorry

end NUMINAMATH_GPT_jacob_age_proof_l249_24951


namespace NUMINAMATH_GPT_initial_roses_l249_24942

theorem initial_roses (x : ℕ) (h : x - 2 + 32 = 41) : x = 11 :=
sorry

end NUMINAMATH_GPT_initial_roses_l249_24942


namespace NUMINAMATH_GPT_find_common_ratio_l249_24905

variable (a₁ : ℝ) (q : ℝ)

def S₁ (a₁ : ℝ) : ℝ := a₁
def S₃ (a₁ q : ℝ) : ℝ := a₁ + a₁ * q + a₁ * q ^ 2
def a₃ (a₁ q : ℝ) : ℝ := a₁ * q ^ 2

theorem find_common_ratio (h : 2 * S₃ a₁ q = S₁ a₁ + 2 * a₃ a₁ q) : q = -1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_common_ratio_l249_24905


namespace NUMINAMATH_GPT_planes_1_and_6_adjacent_prob_l249_24903

noncomputable def probability_planes_adjacent (total_planes: ℕ) : ℚ :=
  if total_planes = 6 then 1/3 else 0

theorem planes_1_and_6_adjacent_prob :
  probability_planes_adjacent 6 = 1/3 := 
by
  sorry

end NUMINAMATH_GPT_planes_1_and_6_adjacent_prob_l249_24903


namespace NUMINAMATH_GPT_find_a_of_extreme_at_1_l249_24948

noncomputable def f (a x : ℝ) : ℝ := a * x^3 - x - Real.log x

theorem find_a_of_extreme_at_1 :
  (∃ a : ℝ, ∃ f' : ℝ -> ℝ, (f' x = 3 * a * x^2 - 1 - 1/x) ∧ f' 1 = 0) →
  ∃ a : ℝ, a = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_a_of_extreme_at_1_l249_24948


namespace NUMINAMATH_GPT_proof_problem_l249_24954

variable (p1 p2 p3 p4 : Prop)

theorem proof_problem (hp1 : p1) (hp2 : ¬ p2) (hp3 : ¬ p3) (hp4 : p4) :
  (p1 ∧ p4) ∧ (¬ p2 ∨ p3) ∧ (¬ p3 ∨ ¬ p4) := by
  sorry

end NUMINAMATH_GPT_proof_problem_l249_24954


namespace NUMINAMATH_GPT_inequality_always_true_l249_24955

theorem inequality_always_true (a b : ℝ) : a^2 + b^2 ≥ -2 * a * b :=
by sorry

end NUMINAMATH_GPT_inequality_always_true_l249_24955


namespace NUMINAMATH_GPT_part_one_union_sets_l249_24988

theorem part_one_union_sets (a : ℝ) (A B : Set ℝ) :
  (a = 2) →
  A = {x | x^2 - (2*a + 1)*x + a*(a + 1) ≤ 0} →
  B = {x | -2 < x ∧ x < 2} →
  A ∪ B = {x | -2 < x ∧ x ≤ 3} :=
by
  sorry

end NUMINAMATH_GPT_part_one_union_sets_l249_24988


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l249_24991

-- Define the arithmetic sequence properties
def seq : List ℕ := [81, 83, 85, 87, 89, 91, 93, 95, 97, 99]
def first := 81
def last := 99
def common_diff := 2
def n := 10

-- Main theorem statement proving the desired property
theorem arithmetic_sequence_sum :
  2 * (seq.sum) = 1800 := by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l249_24991


namespace NUMINAMATH_GPT_evaluate_expression_l249_24957

-- Define the base and the exponents
def base : ℝ := 64
def exponent1 : ℝ := 0.125
def exponent2 : ℝ := 0.375
def combined_result : ℝ := 8

-- Statement of the problem
theorem evaluate_expression : (base^exponent1) * (base^exponent2) = combined_result := 
by 
  sorry

end NUMINAMATH_GPT_evaluate_expression_l249_24957


namespace NUMINAMATH_GPT_area_enclosed_by_curve_l249_24983

theorem area_enclosed_by_curve :
  let s : ℝ := 3
  let arc_length : ℝ := (3 * Real.pi) / 4
  let octagon_area : ℝ := (1 + Real.sqrt 2) * s^2
  let sector_area : ℝ := (3 / 8) * Real.pi
  let total_area : ℝ := 8 * sector_area + octagon_area
  total_area = 9 + 9 * Real.sqrt 2 + 3 * Real.pi :=
by
  let s := 3
  let arc_length := (3 * Real.pi) / 4
  let r := arc_length / ((3 * Real.pi) / 4)
  have r_eq : r = 1 := by
    sorry
  let full_circle_area := Real.pi * r^2
  let sector_area := (3 / 8) * Real.pi
  have sector_area_eq : sector_area = (3 / 8) * Real.pi := by
    sorry
  let total_sector_area := 8 * sector_area
  have total_sector_area_eq : total_sector_area = 3 * Real.pi := by
    sorry
  let octagon_area := (1 + Real.sqrt 2) * s^2
  have octagon_area_eq : octagon_area = 9 * (1 + Real.sqrt 2) := by
    sorry
  let total_area := total_sector_area + octagon_area
  have total_area_eq : total_area = 9 + 9 * Real.sqrt 2 + 3 * Real.pi := by
    sorry
  exact total_area_eq

end NUMINAMATH_GPT_area_enclosed_by_curve_l249_24983


namespace NUMINAMATH_GPT_arithmetic_sequence_problem_l249_24993

theorem arithmetic_sequence_problem
  (a : ℕ → ℕ)
  (S : ℕ → ℕ)
  (a1 : a 1 = 1)
  (a3 : a 3 = 5)
  (Sn : ∀ n, S n = n * (2 + (n - 1) * 2) / 2)
  (S_diff : ∀ k, S (k + 2) - S k = 36)
  : ∃ k : ℕ, k = 8 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_problem_l249_24993


namespace NUMINAMATH_GPT_correct_propositions_l249_24981

theorem correct_propositions :
  let proposition1 := (∀ A B C : ℝ, C = (A + B) / 2 → C = (A + B) / 2)
  let proposition2 := (∀ a : ℝ, a - |a| = 0 → a ≥ 0)
  let proposition3 := false
  let proposition4 := (∀ a b : ℝ, |a| = |b| → a = -b)
  let proposition5 := (∀ a : ℝ, -a < 0)
  (cond1 : proposition1 = false) →
  (cond2 : proposition2 = false) →
  (cond3 : proposition3 = false) →
  (cond4 : proposition4 = true) →
  (cond5 : proposition5 = false) →
  1 = 1 :=
by
  intros
  sorry

end NUMINAMATH_GPT_correct_propositions_l249_24981


namespace NUMINAMATH_GPT_second_number_is_30_l249_24946

theorem second_number_is_30 
  (A B C : ℝ)
  (h1 : A + B + C = 98)
  (h2 : A / B = 2 / 3)
  (h3 : B / C = 5 / 8) : 
  B = 30 :=
by
  sorry

end NUMINAMATH_GPT_second_number_is_30_l249_24946


namespace NUMINAMATH_GPT_total_spaces_in_game_l249_24915

-- Conditions
def first_turn : ℕ := 8
def second_turn_forward : ℕ := 2
def second_turn_backward : ℕ := 5
def third_turn : ℕ := 6
def total_to_end : ℕ := 37

-- Theorem stating the total number of spaces in the game
theorem total_spaces_in_game : first_turn + second_turn_forward - second_turn_backward + third_turn + (total_to_end - (first_turn + second_turn_forward - second_turn_backward + third_turn)) = total_to_end :=
by sorry

end NUMINAMATH_GPT_total_spaces_in_game_l249_24915


namespace NUMINAMATH_GPT_min_value_f_l249_24923

def f (x y z : ℝ) : ℝ := 
  x^2 + 4 * x * y + 3 * y^2 + 2 * z^2 - 8 * x - 4 * y + 6 * z

theorem min_value_f : ∃ (x y z : ℝ), f x y z = -13.5 :=
  by
  use 1, 1.5, -1.5
  sorry

end NUMINAMATH_GPT_min_value_f_l249_24923


namespace NUMINAMATH_GPT_theon_speed_l249_24943

theorem theon_speed (VTheon VYara D : ℕ) (h1 : VYara = 30) (h2 : D = 90) (h3 : D / VTheon = D / VYara + 3) : VTheon = 15 := by
  sorry

end NUMINAMATH_GPT_theon_speed_l249_24943


namespace NUMINAMATH_GPT_find_y_value_l249_24909

theorem find_y_value :
  (∃ m b : ℝ, (∀ x y : ℝ, (x = 2 ∧ y = 5) ∨ (x = 6 ∧ y = 17) ∨ (x = 10 ∧ y = 29) → y = m * x + b))
  → (∃ y : ℝ, x = 40 → y = 119) := by
  sorry

end NUMINAMATH_GPT_find_y_value_l249_24909


namespace NUMINAMATH_GPT_abs_lt_one_iff_sq_lt_one_l249_24934

variable {x : ℝ}

theorem abs_lt_one_iff_sq_lt_one : |x| < 1 ↔ x^2 < 1 := sorry

end NUMINAMATH_GPT_abs_lt_one_iff_sq_lt_one_l249_24934


namespace NUMINAMATH_GPT_intersection_point_of_lines_l249_24921

theorem intersection_point_of_lines : 
  ∃ (x y : ℝ), (x - 4 * y - 1 = 0) ∧ (2 * x + y - 2 = 0) ∧ (x = 1) ∧ (y = 0) :=
by
  sorry

end NUMINAMATH_GPT_intersection_point_of_lines_l249_24921


namespace NUMINAMATH_GPT_integer_solutions_to_equation_l249_24925

theorem integer_solutions_to_equation :
  {p : ℤ × ℤ | (p.fst^2 - 2 * p.fst * p.snd - 3 * p.snd^2 = 5)} =
  {(4, 1), (2, -1), (-4, -1), (-2, 1)} :=
by {
  sorry
}

end NUMINAMATH_GPT_integer_solutions_to_equation_l249_24925


namespace NUMINAMATH_GPT_find_f_neg_2_l249_24966

noncomputable def f (x : ℝ) : ℝ :=
if x >= 0 then 3^x - 1 else sorry -- we'll define this not for non-negative x properly later

def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = -f x

theorem find_f_neg_2 (hodd : is_odd_function f) (hpos : ∀ x : ℝ, 0 ≤ x → f x = 3^x - 1) :
  f (-2) = -8 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_find_f_neg_2_l249_24966


namespace NUMINAMATH_GPT_find_cost_price_l249_24999

noncomputable def cost_price (CP SP_loss SP_gain : ℝ) : Prop :=
SP_loss = 0.90 * CP ∧
SP_gain = 1.05 * CP ∧
(SP_gain - SP_loss = 225)

theorem find_cost_price (CP : ℝ) (h : cost_price CP (0.90 * CP) (1.05 * CP)) : CP = 1500 :=
by
  sorry

end NUMINAMATH_GPT_find_cost_price_l249_24999


namespace NUMINAMATH_GPT_highest_value_of_a_for_divisibility_l249_24987

/-- Given a number in the format of 365a2_, where 'a' is a digit (0 through 9),
prove that the highest value of 'a' that makes the number divisible by 8 is 9. -/
theorem highest_value_of_a_for_divisibility :
  ∃ (a : ℕ), a ≤ 9 ∧ (∃ (d : ℕ), d < 10 ∧ (365 * 100 + a * 10 + 20 + d) % 8 = 0 ∧ a = 9) :=
sorry

end NUMINAMATH_GPT_highest_value_of_a_for_divisibility_l249_24987


namespace NUMINAMATH_GPT_original_number_divisibility_l249_24902

theorem original_number_divisibility (N : ℤ) : (∃ k : ℤ, N = 9 * k + 3) ↔ (∃ m : ℤ, (N + 3) = 9 * m) := sorry

end NUMINAMATH_GPT_original_number_divisibility_l249_24902


namespace NUMINAMATH_GPT_last_digit_m_is_9_l249_24995

def x (n : ℕ) : ℕ := 2^(2^n) + 1

def m : ℕ := List.foldr Nat.lcm 1 (List.map x (List.range' 2 (1971 - 2 + 1)))

theorem last_digit_m_is_9 : m % 10 = 9 :=
  by
    sorry

end NUMINAMATH_GPT_last_digit_m_is_9_l249_24995


namespace NUMINAMATH_GPT_initial_number_of_persons_l249_24929

-- Define the conditions and the goal
def weight_increase_due_to_new_person : ℝ := 102 - 75
def average_weight_increase (n : ℝ) : ℝ := 4.5 * n

theorem initial_number_of_persons (n : ℝ) (h1 : average_weight_increase n = weight_increase_due_to_new_person) : n = 6 :=
by
  -- Skip the proof with sorry
  sorry

end NUMINAMATH_GPT_initial_number_of_persons_l249_24929


namespace NUMINAMATH_GPT_sum_of_three_consecutive_even_numbers_l249_24997

theorem sum_of_three_consecutive_even_numbers (a : ℤ) (h : a * (a + 2) * (a + 4) = 960) : a + (a + 2) + (a + 4) = 30 := by
  sorry

end NUMINAMATH_GPT_sum_of_three_consecutive_even_numbers_l249_24997


namespace NUMINAMATH_GPT_collinear_points_sum_l249_24950

variables {a b : ℝ}

/-- If the points (1, a, b), (a, b, 3), and (b, 3, a) are collinear, then b + a = 3.
-/
theorem collinear_points_sum (h : ∃ k : ℝ, 
  (a - 1, b - a, 3 - b) = k • (b - 1, 3 - a, a - b)) : b + a = 3 :=
sorry

end NUMINAMATH_GPT_collinear_points_sum_l249_24950


namespace NUMINAMATH_GPT_edge_length_of_cube_l249_24975

theorem edge_length_of_cube (total_cubes : ℕ) (box_edge_length_m : ℝ) (box_edge_length_cm : ℝ) 
  (conversion_factor : ℝ) (edge_length_cm : ℝ) : 
  total_cubes = 8 ∧ box_edge_length_m = 1 ∧ box_edge_length_cm = box_edge_length_m * conversion_factor ∧ conversion_factor = 100 ∧ 
  edge_length_cm = box_edge_length_cm / 2 ↔ edge_length_cm = 50 := 
by 
  sorry

end NUMINAMATH_GPT_edge_length_of_cube_l249_24975


namespace NUMINAMATH_GPT_twice_original_price_l249_24961

theorem twice_original_price (P : ℝ) (h : 377 = 1.30 * P) : 2 * P = 580 :=
by {
  -- proof steps will go here
  sorry
}

end NUMINAMATH_GPT_twice_original_price_l249_24961


namespace NUMINAMATH_GPT_inequality_solution_set_l249_24967

theorem inequality_solution_set (x : ℝ) : 
  ( (x - 1) / (x + 2) > 0 ) ↔ ( x > 1 ∨ x < -2 ) :=
by sorry

end NUMINAMATH_GPT_inequality_solution_set_l249_24967


namespace NUMINAMATH_GPT_inequality_1_inequality_2_inequality_3_inequality_4_l249_24916

-- Definitions of distances
def d_a : ℝ := sorry
def d_b : ℝ := sorry
def d_c : ℝ := sorry
def R_a : ℝ := sorry
def R_b : ℝ := sorry
def R_c : ℝ := sorry
def a : ℝ := sorry
def b : ℝ := sorry
def c : ℝ := sorry

def R : ℝ := sorry -- Circumradius
def r : ℝ := sorry -- Inradius

-- Inequality 1
theorem inequality_1 : a * R_a ≥ c * d_c + b * d_b := 
  sorry

-- Inequality 2
theorem inequality_2 : d_a * R_a + d_b * R_b + d_c * R_c ≥ 2 * (d_a * d_b + d_b * d_c + d_c * d_a) :=
  sorry

-- Inequality 3
theorem inequality_3 : R_a + R_b + R_c ≥ 2 * (d_a + d_b + d_c) :=
  sorry

-- Inequality 4
theorem inequality_4 : R_a * R_b * R_c ≥ (R / (2 * r)) * (d_a + d_b) * (d_b + d_c) * (d_c + d_a) :=
  sorry

end NUMINAMATH_GPT_inequality_1_inequality_2_inequality_3_inequality_4_l249_24916


namespace NUMINAMATH_GPT_find_dimensions_l249_24932

def is_solution (m n r : ℕ) : Prop :=
  ∃ k0 k1 k2 : ℕ, 
    k0 = (m - 2) * (n - 2) * (r - 2) ∧
    k1 = 2 * ((m - 2) * (n - 2) + (n - 2) * (r - 2) + (r - 2) * (m - 2)) ∧
    k2 = 4 * ((m - 2) + (n - 2) + (r - 2)) ∧
    k0 + k2 - k1 = 1985

theorem find_dimensions (m n r : ℕ) (h : m ≤ n ∧ n ≤ r) (hp : 0 < m ∧ 0 < n ∧ 0 < r) : 
  is_solution m n r :=
sorry

end NUMINAMATH_GPT_find_dimensions_l249_24932


namespace NUMINAMATH_GPT_arithmetic_sequence_geometric_l249_24907

noncomputable def sequence_arith_to_geom (a1 d : ℤ) (h_d : d ≠ 0) (n : ℕ) : ℤ :=
a1 + (n - 1) * d

theorem arithmetic_sequence_geometric (a1 d : ℤ) (h_d : d ≠ 0) (n : ℕ) :
  (n = 16)
    ↔ (((a1 + 3 * d) / (a1 + 2 * d) = (a1 + 6 * d) / (a1 + 3 * d)) ∧ 
        ((a1 + 6 * d) / (a1 + 3 * d) = (a1 + (n - 1) * d) / (a1 + 6 * d))) :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_geometric_l249_24907


namespace NUMINAMATH_GPT_lines_parallel_coeff_l249_24984

theorem lines_parallel_coeff (a : ℝ) :
  (∀ x y: ℝ, a * x + 2 * y = 0 → 3 * x + (a + 1) * y + 1 = 0) ↔ (a = -3 ∨ a = 2) :=
by
  sorry

end NUMINAMATH_GPT_lines_parallel_coeff_l249_24984


namespace NUMINAMATH_GPT_bowling_average_decrease_l249_24918

/-- Represents data about the bowler's performance. -/
structure BowlerPerformance :=
(old_average : ℚ)
(last_match_runs : ℚ)
(last_match_wickets : ℕ)
(previous_wickets : ℕ)

/-- Calculates the new total runs given. -/
def new_total_runs (perf : BowlerPerformance) : ℚ :=
  perf.old_average * ↑perf.previous_wickets + perf.last_match_runs

/-- Calculates the new total number of wickets. -/
def new_total_wickets (perf : BowlerPerformance) : ℕ :=
  perf.previous_wickets + perf.last_match_wickets

/-- Calculates the new bowling average. -/
def new_average (perf : BowlerPerformance) : ℚ :=
  new_total_runs perf / ↑(new_total_wickets perf)

/-- Calculates the decrease in the bowling average. -/
def decrease_in_average (perf : BowlerPerformance) : ℚ :=
  perf.old_average - new_average perf

/-- The proof statement to be verified. -/
theorem bowling_average_decrease :
  ∀ (perf : BowlerPerformance),
    perf.old_average = 12.4 →
    perf.last_match_runs = 26 →
    perf.last_match_wickets = 6 →
    perf.previous_wickets = 115 →
    decrease_in_average perf = 0.4 :=
by
  intros
  sorry

end NUMINAMATH_GPT_bowling_average_decrease_l249_24918


namespace NUMINAMATH_GPT_cheese_bread_grams_l249_24953

/-- Each 100 grams of cheese bread costs 3.20 BRL and corresponds to 10 pieces. 
Each person eats, on average, 5 pieces of cheese bread. Including the professor,
there are 16 students, 1 monitor, and 5 parents, making a total of 23 people. 
The precision of the bakery's scale is 100 grams. -/
theorem cheese_bread_grams : (5 * 23 / 10) * 100 = 1200 := 
by
  sorry

end NUMINAMATH_GPT_cheese_bread_grams_l249_24953


namespace NUMINAMATH_GPT_find_number_l249_24908

axiom condition_one (x y : ℕ) : 10 * x + y = 3 * (x + y) + 7
axiom condition_two (x y : ℕ) : x^2 + y^2 - x * y = 10 * x + y

theorem find_number : 
  ∃ (x y : ℕ), (10 * x + y = 37) → (10 * x + y = 3 * (x + y) + 7 ∧ x^2 + y^2 - x * y = 10 * x + y) := 
by 
  sorry

end NUMINAMATH_GPT_find_number_l249_24908


namespace NUMINAMATH_GPT_units_digit_sum_l249_24944

theorem units_digit_sum (h₁ : (24 : ℕ) % 10 = 4) 
                        (h₂ : (42 : ℕ) % 10 = 2) : 
  ((24^3 + 42^3) % 10 = 2) :=
by
  sorry

end NUMINAMATH_GPT_units_digit_sum_l249_24944


namespace NUMINAMATH_GPT_RouteB_quicker_than_RouteA_l249_24928

def RouteA_segment1_time : ℚ := 4 / 40 -- time in hours
def RouteA_segment2_time : ℚ := 4 / 20 -- time in hours
def RouteA_total_time : ℚ := RouteA_segment1_time + RouteA_segment2_time -- total time in hours

def RouteB_segment1_time : ℚ := 6 / 35 -- time in hours
def RouteB_segment2_time : ℚ := 1 / 15 -- time in hours
def RouteB_total_time : ℚ := RouteB_segment1_time + RouteB_segment2_time -- total time in hours

def time_difference_minutes : ℚ := (RouteA_total_time - RouteB_total_time) * 60 -- difference in minutes

theorem RouteB_quicker_than_RouteA : time_difference_minutes = 3.71 := by
  sorry

end NUMINAMATH_GPT_RouteB_quicker_than_RouteA_l249_24928


namespace NUMINAMATH_GPT_initial_books_in_library_l249_24977

theorem initial_books_in_library 
  (initial_books : ℕ)
  (books_taken_out_Tuesday : ℕ := 120)
  (books_returned_Wednesday : ℕ := 35)
  (books_withdrawn_Thursday : ℕ := 15)
  (books_final_count : ℕ := 150)
  : initial_books - books_taken_out_Tuesday + books_returned_Wednesday - books_withdrawn_Thursday = books_final_count → initial_books = 250 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_initial_books_in_library_l249_24977


namespace NUMINAMATH_GPT_cos_seven_pi_over_six_l249_24962

theorem cos_seven_pi_over_six : Real.cos (7 * Real.pi / 6) = -Real.sqrt 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_cos_seven_pi_over_six_l249_24962


namespace NUMINAMATH_GPT_variable_value_l249_24952

theorem variable_value 
  (x : ℝ)
  (a k some_variable : ℝ)
  (eqn1 : (3 * x + 2) * (2 * x - 7) = a * x^2 + k * x + some_variable)
  (eqn2 : a - some_variable + k = 3)
  (a_val : a = 6)
  (k_val : k = -17) :
  some_variable = -14 :=
by
  sorry

end NUMINAMATH_GPT_variable_value_l249_24952


namespace NUMINAMATH_GPT_complete_the_square_l249_24963

theorem complete_the_square : ∀ x : ℝ, x^2 - 6 * x + 4 = 0 → (x - 3)^2 = 5 :=
by
  intro x h
  sorry

end NUMINAMATH_GPT_complete_the_square_l249_24963


namespace NUMINAMATH_GPT_james_weekly_earnings_l249_24945

def rate_per_hour : ℕ := 20
def hours_per_day : ℕ := 8
def days_per_week : ℕ := 4

def daily_earnings : ℕ := rate_per_hour * hours_per_day
def weekly_earnings : ℕ := daily_earnings * days_per_week

theorem james_weekly_earnings : weekly_earnings = 640 := sorry

end NUMINAMATH_GPT_james_weekly_earnings_l249_24945


namespace NUMINAMATH_GPT_vertical_angles_always_equal_l249_24992

theorem vertical_angles_always_equal (a b : ℝ) (h : a = b) : 
  (∀ θ1 θ2, θ1 + θ2 = 180 ∧ θ1 = a ∧ θ2 = b → θ1 = θ2) :=
by 
  intro θ1 θ2 
  intro h 
  sorry

end NUMINAMATH_GPT_vertical_angles_always_equal_l249_24992


namespace NUMINAMATH_GPT_value_of_a_if_perpendicular_l249_24998

theorem value_of_a_if_perpendicular (a l : ℝ) :
  (∀ x y : ℝ, (a + l) * x + 2 * y = 0 → x - a * y = 1 → false) → a = 1 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_value_of_a_if_perpendicular_l249_24998


namespace NUMINAMATH_GPT_f_is_odd_and_periodic_l249_24958

noncomputable def f : ℝ → ℝ := sorry

axiom h1 : ∀ x : ℝ, f (10 + x) = f (10 - x)
axiom h2 : ∀ x : ℝ, f (20 - x) = -f (20 + x)

theorem f_is_odd_and_periodic : 
  (∀ x : ℝ, f (-x) = -f x) ∧ (∃ T : ℝ, T = 40 ∧ ∀ x : ℝ, f (x + T) = f x) :=
by
  sorry

end NUMINAMATH_GPT_f_is_odd_and_periodic_l249_24958


namespace NUMINAMATH_GPT_liangliang_speed_l249_24914

theorem liangliang_speed (d_initial : ℝ) (t : ℝ) (d_final : ℝ) (v_mingming : ℝ) (v_liangliang : ℝ) :
  d_initial = 3000 →
  t = 20 →
  d_final = 2900 →
  v_mingming = 80 →
  (v_liangliang = 85 ∨ v_liangliang = 75) :=
by
  sorry

end NUMINAMATH_GPT_liangliang_speed_l249_24914


namespace NUMINAMATH_GPT_solve_for_y_l249_24968

theorem solve_for_y (x y : ℚ) (h₁ : x - y = 12) (h₂ : 2 * x + y = 10) : y = -14 / 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_y_l249_24968


namespace NUMINAMATH_GPT_f_characterization_l249_24949

noncomputable def op (a b : ℝ) := a * b

noncomputable def ot (a b : ℝ) := a + b

noncomputable def f (x : ℝ) := ot x 2 - op 2 x

-- Prove that f(x) is neither odd nor even and is a decreasing function
theorem f_characterization :
  (∀ x : ℝ, f x = -x + 2) ∧
  (∀ x : ℝ, f (-x) ≠ f x ∧ f (-x) ≠ -f x) ∧
  (∀ x y : ℝ, x < y → f x > f y) := sorry

end NUMINAMATH_GPT_f_characterization_l249_24949


namespace NUMINAMATH_GPT_arithmetic_sequence_nine_l249_24960

variable (a : ℕ → ℝ)
variable (d : ℝ)
-- Define an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) := ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_nine (a : ℕ → ℝ) (d : ℝ) 
  (h_arith : arithmetic_sequence a d)
  (h_cond : a 4 + a 14 = 2) : 
  a 9 = 1 := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_nine_l249_24960

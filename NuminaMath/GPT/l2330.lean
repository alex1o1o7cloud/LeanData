import Mathlib

namespace NUMINAMATH_GPT_sum_of_local_values_l2330_233012

def local_value (digit place_value : ℕ) : ℕ := digit * place_value

theorem sum_of_local_values :
  local_value 2 1000 + local_value 3 100 + local_value 4 10 + local_value 5 1 = 2345 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_local_values_l2330_233012


namespace NUMINAMATH_GPT_minimum_n_for_all_columns_l2330_233093

def triangular_number (n : ℕ) : ℕ :=
  n * (n + 1) / 2

-- Function to check if a given number covers all columns from 0 to 9
def covers_all_columns (n : ℕ) : Bool :=
  let columns := (List.range n).map (λ i => triangular_number i % 10)
  List.range 10 |>.all (λ c => c ∈ columns)

theorem minimum_n_for_all_columns : ∃ n, covers_all_columns n ∧ triangular_number n = 253 :=
by 
  sorry

end NUMINAMATH_GPT_minimum_n_for_all_columns_l2330_233093


namespace NUMINAMATH_GPT_impossible_to_maintain_Gini_l2330_233092

variables (X Y G0 Y' Z : ℝ)
variables (G1 : ℝ)

-- Conditions
axiom initial_Gini : G0 = 0.1
axiom proportion_poor : X = 0.5
axiom income_poor_initial : Y = 0.4
axiom income_poor_half : Y' = 0.2
axiom population_split : ∀ a b c : ℝ, (a + b + c = 1) ∧ (a = b ∧ b = c)
axiom Gini_constant : G1 = G0

-- Equation system representation final value post situation
axiom Gini_post_reform : 
  G1 = (1 / 2 - ((1 / 6) * 0.2 + (1 / 6) * (0.2 + Z) + (1 / 6) * (1 - 0.2 - Z))) / (1 / 2)

-- Proof problem: to prove inconsistency or inability to maintain Gini coefficient given the conditions
theorem impossible_to_maintain_Gini : false :=
sorry

end NUMINAMATH_GPT_impossible_to_maintain_Gini_l2330_233092


namespace NUMINAMATH_GPT_find_angle_C_l2330_233060

theorem find_angle_C (A B C : ℝ) (h1 : |Real.cos A - (Real.sqrt 3 / 2)| + (1 - Real.tan B)^2 = 0) :
  C = 105 :=
by
  sorry

end NUMINAMATH_GPT_find_angle_C_l2330_233060


namespace NUMINAMATH_GPT_proof_expression_value_l2330_233049

theorem proof_expression_value (x y : ℝ) (h : x + 2 * y = 30) : 
  (x / 5 + 2 * y / 3 + 2 * y / 5 + x / 3) = 16 := 
by 
  sorry

end NUMINAMATH_GPT_proof_expression_value_l2330_233049


namespace NUMINAMATH_GPT_hyperbolas_same_asymptotes_l2330_233028

theorem hyperbolas_same_asymptotes (M : ℝ) :
  (∀ x y : ℝ, (x^2 / 9 - y^2 / 16 = 1) ↔ (y^2 / 25 - x^2 / M = 1)) → M = 225 / 16 :=
by
  sorry

end NUMINAMATH_GPT_hyperbolas_same_asymptotes_l2330_233028


namespace NUMINAMATH_GPT_common_ratio_is_two_l2330_233066

-- Given a geometric sequence with specific terms
variable (a : ℕ → ℝ) (q : ℝ)

-- Conditions: all terms are positive, a_2 = 3, a_6 = 48
axiom pos_terms : ∀ n, a n > 0
axiom a2_eq : a 2 = 3
axiom a6_eq : a 6 = 48

-- Question: Prove the common ratio q is 2
theorem common_ratio_is_two :
  (∀ n, a n = a 1 * q ^ (n - 1)) → q = 2 :=
by
  sorry

end NUMINAMATH_GPT_common_ratio_is_two_l2330_233066


namespace NUMINAMATH_GPT_nick_coin_collection_l2330_233033

theorem nick_coin_collection
  (total_coins : ℕ)
  (quarters_coins : ℕ)
  (dimes_coins : ℕ)
  (nickels_coins : ℕ)
  (state_quarters : ℕ)
  (pa_state_quarters : ℕ)
  (roosevelt_dimes : ℕ)
  (h_total : total_coins = 50)
  (h_quarters : quarters_coins = total_coins * 3 / 10)
  (h_dimes : dimes_coins = total_coins * 40 / 100)
  (h_nickels : nickels_coins = total_coins - (quarters_coins + dimes_coins))
  (h_state_quarters : state_quarters = quarters_coins * 2 / 5)
  (h_pa_state_quarters : pa_state_quarters = state_quarters * 3 / 8)
  (h_roosevelt_dimes : roosevelt_dimes = dimes_coins * 75 / 100) :
  pa_state_quarters = 2 ∧ roosevelt_dimes = 15 ∧ nickels_coins = 15 :=
by
  sorry

end NUMINAMATH_GPT_nick_coin_collection_l2330_233033


namespace NUMINAMATH_GPT_ratio_of_girls_to_boys_l2330_233010

theorem ratio_of_girls_to_boys (g b : ℕ) (h1 : g = b + 6) (h2 : g + b = 36) : g / b = 7 / 5 := by sorry

end NUMINAMATH_GPT_ratio_of_girls_to_boys_l2330_233010


namespace NUMINAMATH_GPT_color_fig_l2330_233072

noncomputable def total_colorings (dots : Finset (Fin 9)) (colors : Finset (Fin 4))
  (adj : dots → dots → Prop)
  (diag : dots → dots → Prop) : Nat :=
  -- coloring left triangle
  let left_triangle := 4 * 3 * 2;
  -- coloring middle triangle considering diagonal restrictions
  let middle_triangle := 3 * 2;
  -- coloring right triangle considering same restrictions
  let right_triangle := 3 * 2;
  left_triangle * middle_triangle * middle_triangle

theorem color_fig (dots : Finset (Fin 9)) (colors : Finset (Fin 4))
  (adj : dots → dots → Prop)
  (diag : dots → dots → Prop) :
  total_colorings dots colors adj diag = 864 :=
by
  sorry

end NUMINAMATH_GPT_color_fig_l2330_233072


namespace NUMINAMATH_GPT_nested_expression_evaluation_l2330_233029

theorem nested_expression_evaluation : (3 * (3 * (3 * (3 * (3 * (3 + 2) + 2) + 2) + 2) + 2) + 2) = 1457 :=
by
  sorry

end NUMINAMATH_GPT_nested_expression_evaluation_l2330_233029


namespace NUMINAMATH_GPT_counseling_rooms_l2330_233005

theorem counseling_rooms (n : ℕ) (x : ℕ)
  (h1 : n = 20 * x + 32)
  (h2 : n = 24 * (x - 1)) : x = 14 :=
by
  sorry

end NUMINAMATH_GPT_counseling_rooms_l2330_233005


namespace NUMINAMATH_GPT_original_number_increased_by_40_percent_l2330_233078

theorem original_number_increased_by_40_percent (x : ℝ) (h : 1.40 * x = 700) : x = 500 :=
by
  sorry

end NUMINAMATH_GPT_original_number_increased_by_40_percent_l2330_233078


namespace NUMINAMATH_GPT_discount_difference_l2330_233016

theorem discount_difference :
  ∀ (original_price : ℝ),
  let initial_discount := 0.40
  let subsequent_discount := 0.25
  let claimed_discount := 0.60
  let actual_discount := 1 - (1 - initial_discount) * (1 - subsequent_discount)
  let difference := claimed_discount - actual_discount
  actual_discount = 0.55 ∧ difference = 0.05
:= by
  sorry

end NUMINAMATH_GPT_discount_difference_l2330_233016


namespace NUMINAMATH_GPT_sum_of_x_coordinates_l2330_233086

def line1 (x : ℝ) : ℝ := -3 * x - 5
def line2 (x : ℝ) : ℝ := 2 * x - 3

def has_x_intersect (line : ℝ → ℝ) (y : ℝ) : Prop := ∃ x : ℝ, line x = y

theorem sum_of_x_coordinates :
  (∃ x1 x2 : ℝ, line1 x1 = 2.2 ∧ line2 x2 = 2.2 ∧ x1 + x2 = 0.2) :=
  sorry

end NUMINAMATH_GPT_sum_of_x_coordinates_l2330_233086


namespace NUMINAMATH_GPT_cloth_meters_sold_l2330_233011

-- Conditions as definitions
def total_selling_price : ℝ := 4500
def profit_per_meter : ℝ := 14
def cost_price_per_meter : ℝ := 86

-- The statement of the problem
theorem cloth_meters_sold (SP : ℝ := cost_price_per_meter + profit_per_meter) :
  total_selling_price / SP = 45 := by
  sorry

end NUMINAMATH_GPT_cloth_meters_sold_l2330_233011


namespace NUMINAMATH_GPT_problem_l2330_233026

noncomputable def f (x : ℝ) : ℝ := x^3 - 2 * x^2 - 5 * x + 6

theorem problem (h : f 10 = 756) : f 10 = 756 := 
by 
  sorry

end NUMINAMATH_GPT_problem_l2330_233026


namespace NUMINAMATH_GPT_smallest_largest_number_in_list_l2330_233083

theorem smallest_largest_number_in_list :
  ∃ (a b c d e : ℕ), (a > 0) ∧ (b > 0) ∧ (c > 0) ∧ (d > 0) ∧ (e > 0) ∧ 
  (a + b + c + d + e = 50) ∧ (e - a = 20) ∧ 
  (c = 6) ∧ (b = 6) ∧ 
  (e = 20) :=
by
  sorry

end NUMINAMATH_GPT_smallest_largest_number_in_list_l2330_233083


namespace NUMINAMATH_GPT_machine_sprockets_rate_l2330_233038

theorem machine_sprockets_rate:
  ∀ (h : ℝ), h > 0 → (660 / (h + 10) = (660 / h) * 1/1.1) → (660 / 1.1 / h) = 6 :=
by
  intros h h_pos h_eq
  -- Proof will be here
  sorry

end NUMINAMATH_GPT_machine_sprockets_rate_l2330_233038


namespace NUMINAMATH_GPT_translated_line_value_m_l2330_233077

theorem translated_line_value_m :
  (∀ x y : ℝ, (y = x → y = x + 3) → y = 2 + 3 → ∃ m : ℝ, y = m) :=
by sorry

end NUMINAMATH_GPT_translated_line_value_m_l2330_233077


namespace NUMINAMATH_GPT_Z_4_3_eq_37_l2330_233043

def Z (a b : ℕ) : ℕ :=
  a^2 + a * b + b^2

theorem Z_4_3_eq_37 : Z 4 3 = 37 :=
  by
    sorry

end NUMINAMATH_GPT_Z_4_3_eq_37_l2330_233043


namespace NUMINAMATH_GPT_solve_eq1_solve_eq2_l2330_233074

-- Define the first proof problem
theorem solve_eq1 (x : ℝ) : 2 * x - 3 = 3 * (x + 1) → x = -6 :=
by
  sorry

-- Define the second proof problem
theorem solve_eq2 (x : ℝ) : (1 / 2) * x - (9 * x - 2) / 6 - 2 = 0 → x = -5 / 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_eq1_solve_eq2_l2330_233074


namespace NUMINAMATH_GPT_each_persons_contribution_l2330_233087

def total_cost : ℝ := 67
def coupon : ℝ := 4
def num_people : ℝ := 3

theorem each_persons_contribution :
  (total_cost - coupon) / num_people = 21 := by
  sorry

end NUMINAMATH_GPT_each_persons_contribution_l2330_233087


namespace NUMINAMATH_GPT_maximize_profit_l2330_233006

def cost_per_product : ℝ := 3
def management_fee_per_product : ℝ := 3
def sales_volume (x : ℝ) : ℝ := (12 - x)^2 * 10000
def annual_profit (x : ℝ) : ℝ := (x - cost_per_product - management_fee_per_product) * sales_volume x

theorem maximize_profit :
  (∀ x : ℝ, 9 ≤ x ∧ x ≤ 11 → annual_profit x = x^3 - 30*x^2 + 288*x - 864) ∧
  annual_profit 9 = 27 * 10000 ∧
  (∀ x : ℝ, 9 ≤ x ∧ x ≤ 11 → annual_profit x ≤ annual_profit 9) :=
by
  sorry

end NUMINAMATH_GPT_maximize_profit_l2330_233006


namespace NUMINAMATH_GPT_no_solution_iff_a_leq_8_l2330_233073

theorem no_solution_iff_a_leq_8 (a : ℝ) :
  (¬ ∃ x : ℝ, |x - 5| + |x + 3| < a) ↔ a ≤ 8 := 
sorry

end NUMINAMATH_GPT_no_solution_iff_a_leq_8_l2330_233073


namespace NUMINAMATH_GPT_total_luggage_l2330_233039

theorem total_luggage (ne nb nf : ℕ)
  (leconomy lbusiness lfirst : ℕ)
  (Heconomy : ne = 10) 
  (Hbusiness : nb = 7) 
  (Hfirst : nf = 3)
  (Heconomy_luggage : leconomy = 5)
  (Hbusiness_luggage : lbusiness = 8)
  (Hfirst_luggage : lfirst = 12) : 
  (ne * leconomy + nb * lbusiness + nf * lfirst) = 142 :=
by
  sorry

end NUMINAMATH_GPT_total_luggage_l2330_233039


namespace NUMINAMATH_GPT_angle_C_is_150_degrees_l2330_233002

theorem angle_C_is_150_degrees
  (C D : ℝ)
  (h_supp : C + D = 180)
  (h_C_5D : C = 5 * D) :
  C = 150 :=
by
  sorry

end NUMINAMATH_GPT_angle_C_is_150_degrees_l2330_233002


namespace NUMINAMATH_GPT_geometric_sequence_sum_l2330_233081

-- Let {a_n} be a geometric sequence such that S_2 = 7 and S_6 = 91. Prove that S_4 = 28

-- Define the sum of the first n terms of a geometric sequence
noncomputable def S (n : ℕ) (a1 r : ℝ) : ℝ := a1 * (1 - r^n) / (1 - r)

theorem geometric_sequence_sum (a1 r : ℝ) (h1 : S 2 a1 r = 7) (h2 : S 6 a1 r = 91) :
  S 4 a1 r = 28 := 
by 
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l2330_233081


namespace NUMINAMATH_GPT_find_m_l2330_233098

theorem find_m (m : ℝ) (h1 : (∀ x : ℝ, (x^2 - m) * (x + m) = x^3 + m * (x^2 - x - 12))) (h2 : m ≠ 0) : m = 12 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l2330_233098


namespace NUMINAMATH_GPT_decreasing_interval_l2330_233020

def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 4

theorem decreasing_interval : ∀ x : ℝ, 0 < x ∧ x < 2 → deriv f x < 0 :=
by sorry

end NUMINAMATH_GPT_decreasing_interval_l2330_233020


namespace NUMINAMATH_GPT_sum_of_nonnegative_numbers_eq_10_l2330_233058

theorem sum_of_nonnegative_numbers_eq_10 (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 48)
  (h2 : ab + bc + ca = 26)
  (h3 : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c) : a + b + c = 10 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_nonnegative_numbers_eq_10_l2330_233058


namespace NUMINAMATH_GPT_simplify_expression_l2330_233051

theorem simplify_expression
  (x y : ℝ)
  (h : (x + 2)^3 ≠ (y - 2)^3) :
  ( (x + 2)^3 + (y + x)^3 ) / ( (x + 2)^3 - (y - 2)^3 ) = (2 * x + y + 2) / (x - y + 4) :=
sorry

end NUMINAMATH_GPT_simplify_expression_l2330_233051


namespace NUMINAMATH_GPT_solution_set_ineq_l2330_233061

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x) - 2 * x

theorem solution_set_ineq (x : ℝ) : f (x^2 - 4) + f (3*x) > 0 ↔ x > 1 ∨ x < -4 :=
by sorry

end NUMINAMATH_GPT_solution_set_ineq_l2330_233061


namespace NUMINAMATH_GPT_mass_percentage_C_in_butanoic_acid_is_54_50_l2330_233055

noncomputable def atomic_mass_C : ℝ := 12.01
noncomputable def atomic_mass_H : ℝ := 1.01
noncomputable def atomic_mass_O : ℝ := 16.00

noncomputable def molar_mass_butanoic_acid : ℝ :=
  (4 * atomic_mass_C) + (8 * atomic_mass_H) + (2 * atomic_mass_O)

noncomputable def mass_of_C_in_butanoic_acid : ℝ :=
  4 * atomic_mass_C

noncomputable def mass_percentage_C : ℝ :=
  (mass_of_C_in_butanoic_acid / molar_mass_butanoic_acid) * 100

theorem mass_percentage_C_in_butanoic_acid_is_54_50 :
  mass_percentage_C = 54.50 := by
  sorry

end NUMINAMATH_GPT_mass_percentage_C_in_butanoic_acid_is_54_50_l2330_233055


namespace NUMINAMATH_GPT_determine_radii_l2330_233063

-- Definitions based on conditions from a)
variable (S1 S2 S3 S4 : Type) -- Centers of the circles
variable (dist_S2_S4 : ℝ) (dist_S1_S2 : ℝ) (dist_S2_S3 : ℝ) (dist_S3_S4 : ℝ)
variable (r1 r2 r3 r4 : ℝ) -- Radii of circles k1, k2, k3, and k4
variable (rhombus : Prop) -- Quadrilateral S1S2S3S4 is a rhombus

-- Given conditions
axiom C1 : ∀ t : S1, r1 = 5
axiom C2 : dist_S2_S4 = 24
axiom C3 : rhombus

-- Equivalency to be proven
theorem determine_radii : 
  r2 = 12 ∧ r4 = 12 ∧ r1 = 5 ∧ r3 = 5 :=
sorry

end NUMINAMATH_GPT_determine_radii_l2330_233063


namespace NUMINAMATH_GPT_value_equation_l2330_233042

noncomputable def quarter_value := 25
noncomputable def dime_value := 10
noncomputable def half_dollar_value := 50

theorem value_equation (n : ℕ) :
  25 * quarter_value + 20 * dime_value = 15 * quarter_value + 10 * dime_value + n * half_dollar_value → 
  n = 7 :=
by
  sorry

end NUMINAMATH_GPT_value_equation_l2330_233042


namespace NUMINAMATH_GPT_problem_solution_l2330_233048

variable (α : ℝ)
-- Condition: α in the first quadrant (0 < α < π/2)
variable (h1 : 0 < α ∧ α < Real.pi / 2)
-- Condition: sin α + cos α = sqrt 2
variable (h2 : Real.sin α + Real.cos α = Real.sqrt 2)

theorem problem_solution : Real.tan α + Real.cos α / Real.sin α = 2 :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l2330_233048


namespace NUMINAMATH_GPT_perfect_square_quotient_l2330_233064

theorem perfect_square_quotient (a b : ℕ) (ha : a > 0) (hb : b > 0)
  (h : (a * b + 1) ∣ (a * a + b * b)) : 
  ∃ k : ℕ, (a * a + b * b) = (a * b + 1) * (k * k) := 
sorry

end NUMINAMATH_GPT_perfect_square_quotient_l2330_233064


namespace NUMINAMATH_GPT_Jose_share_land_l2330_233054

theorem Jose_share_land (total_land : ℕ) (num_siblings : ℕ) (total_parts : ℕ) (share_per_person : ℕ) :
  total_land = 20000 → num_siblings = 4 → total_parts = (1 + num_siblings) → share_per_person = (total_land / total_parts) → 
  share_per_person = 4000 :=
by
  sorry

end NUMINAMATH_GPT_Jose_share_land_l2330_233054


namespace NUMINAMATH_GPT_relationship_of_exponents_l2330_233080

theorem relationship_of_exponents (m p r s : ℝ) (u v w t : ℝ) (h1 : m^u = r) (h2 : p^v = r) (h3 : p^w = s) (h4 : m^t = s) : u * v = w * t :=
by
  sorry

end NUMINAMATH_GPT_relationship_of_exponents_l2330_233080


namespace NUMINAMATH_GPT_race_winner_l2330_233030

theorem race_winner
  (faster : String → String → Prop)
  (Minyoung Yoongi Jimin Yuna : String)
  (cond1 : faster Minyoung Yoongi)
  (cond2 : faster Yoongi Jimin)
  (cond3 : faster Yuna Jimin)
  (cond4 : faster Yuna Minyoung) :
  ∀ s, s ≠ Yuna → faster Yuna s :=
by
  sorry

end NUMINAMATH_GPT_race_winner_l2330_233030


namespace NUMINAMATH_GPT_remainder_of_greatest_integer_multiple_of_9_no_repeats_l2330_233047

noncomputable def greatest_integer_multiple_of_9_no_repeats : ℕ :=
  9876543210 -- this should correspond to the greatest number meeting the criteria, but it's identified via more specific logic in practice

theorem remainder_of_greatest_integer_multiple_of_9_no_repeats : 
  (greatest_integer_multiple_of_9_no_repeats % 1000) = 621 := 
  by sorry

end NUMINAMATH_GPT_remainder_of_greatest_integer_multiple_of_9_no_repeats_l2330_233047


namespace NUMINAMATH_GPT_max_min_f_product_of_roots_f_l2330_233040

noncomputable def f (x : ℝ) : ℝ := 
  (Real.log x / Real.log 3 - 3) * (Real.log x / Real.log 3 + 1)

theorem max_min_f
  (x : ℝ) (h : x ∈ Set.Icc (1/27 : ℝ) (1/9 : ℝ)) : 
  (∀ y, y ∈ Set.Icc (1/27 : ℝ) (1/9 : ℝ) → f y ≤ 12)
  ∧ (∀ y, y ∈ Set.Icc (1/27 : ℝ) (1/9 : ℝ) → f y ≥ 5) :=
sorry

theorem product_of_roots_f
  (m α β : ℝ) (h1 : f α + m = 0) (h2 : f β + m = 0) : 
  (Real.log (α * β) / Real.log 3 = 2) → (α * β = 9) :=
sorry

end NUMINAMATH_GPT_max_min_f_product_of_roots_f_l2330_233040


namespace NUMINAMATH_GPT_part1_l2330_233027

theorem part1 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
  (h : (a^2 + b^2 + c^2)^2 > 2 * (a^4 + b^4 + c^4)) : 
  (a + b > c) ∧ (b + c > a) ∧ (c + a > b) :=
sorry

end NUMINAMATH_GPT_part1_l2330_233027


namespace NUMINAMATH_GPT_roots_of_quadratic_eq_l2330_233031

theorem roots_of_quadratic_eq (a b : ℝ) (h1 : a * (-2)^2 + b * (-2) = 6) (h2 : a * 3^2 + b * 3 = 6) :
    ∃ (x1 x2 : ℝ), x1 = -2 ∧ x2 = 3 ∧ ∀ x, a * x^2 + b * x = 6 ↔ (x = x1 ∨ x = x2) :=
by
  use -2, 3
  sorry

end NUMINAMATH_GPT_roots_of_quadratic_eq_l2330_233031


namespace NUMINAMATH_GPT_largest_angle_triangl_DEF_l2330_233007

theorem largest_angle_triangl_DEF (d e f : ℝ) (h1 : d + 3 * e + 3 * f = d^2)
  (h2 : d + 3 * e - 3 * f = -8) : 
  ∃ (F : ℝ), F = 109.47 ∧ (F > 90) := by sorry

end NUMINAMATH_GPT_largest_angle_triangl_DEF_l2330_233007


namespace NUMINAMATH_GPT_new_person_weight_l2330_233000

theorem new_person_weight (avg_inc : Real) (num_persons : Nat) (old_weight new_weight : Real)
  (h1 : avg_inc = 2.5)
  (h2 : num_persons = 8)
  (h3 : old_weight = 40)
  (h4 : num_persons * avg_inc = new_weight - old_weight) :
  new_weight = 60 :=
by
  --proof will be done here
  sorry

end NUMINAMATH_GPT_new_person_weight_l2330_233000


namespace NUMINAMATH_GPT_num_boys_on_playground_l2330_233025

-- Define the conditions using Lean definitions
def num_girls : Nat := 28
def total_children : Nat := 63

-- Define a theorem to prove the number of boys
theorem num_boys_on_playground : total_children - num_girls = 35 :=
by
  -- proof steps would go here
  sorry

end NUMINAMATH_GPT_num_boys_on_playground_l2330_233025


namespace NUMINAMATH_GPT_ball_distribution_l2330_233059

theorem ball_distribution (N a b : ℕ) (h1 : N = 6912) (h2 : N = 100 * a + b) (h3 : a < 100) (h4 : b < 100) : a + b = 81 :=
by
  sorry

end NUMINAMATH_GPT_ball_distribution_l2330_233059


namespace NUMINAMATH_GPT_gcf_palindromes_multiple_of_3_eq_3_l2330_233003

-- Defining a condition that expresses a three-digit palindrome in the form 101a + 10b + a
def is_palindrome (n : ℕ) : Prop :=
∃ a b : ℕ, a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ n = 101 * a + 10 * b + a

-- Defining a condition that n is a multiple of 3
def is_multiple_of_3 (n : ℕ) : Prop :=
n % 3 = 0

-- The Lean statement to prove the greatest common factor of all three-digit palindromes that are multiples of 3
theorem gcf_palindromes_multiple_of_3_eq_3 :
  ∃ gcf : ℕ, gcf = 3 ∧ ∀ n : ℕ, (is_palindrome n ∧ is_multiple_of_3 n) → gcf ∣ n :=
by
  sorry

end NUMINAMATH_GPT_gcf_palindromes_multiple_of_3_eq_3_l2330_233003


namespace NUMINAMATH_GPT_orchid_bushes_after_planting_l2330_233044

def total_orchid_bushes (current_orchids new_orchids : Nat) : Nat :=
  current_orchids + new_orchids

theorem orchid_bushes_after_planting :
  ∀ (current_orchids new_orchids : Nat), current_orchids = 22 → new_orchids = 13 → total_orchid_bushes current_orchids new_orchids = 35 :=
by
  intros current_orchids new_orchids h_current h_new
  rw [h_current, h_new]
  exact rfl

end NUMINAMATH_GPT_orchid_bushes_after_planting_l2330_233044


namespace NUMINAMATH_GPT_express_fraction_l2330_233046

noncomputable def x : ℚ := 0.8571 -- This represents \( x = 0.\overline{8571} \)
noncomputable def y : ℚ := 0.142857 -- This represents \( y = 0.\overline{142857} \)
noncomputable def z : ℚ := 2 + y -- This represents \( 2 + y = 2.\overline{142857} \)

theorem express_fraction :
  (x / z) = (1 / 2) :=
by
  sorry

end NUMINAMATH_GPT_express_fraction_l2330_233046


namespace NUMINAMATH_GPT_first_term_of_arithmetic_sequence_l2330_233037

theorem first_term_of_arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) (a1 : ℝ)
  (h_arith : ∀ n, a n = a1 + ↑n - 1) 
  (h_sum : ∀ n, S n = n / 2 * (2 * a1 + (n - 1))) 
  (h_min : ∀ n, S 2022 ≤ S n) : 
  -2022 < a1 ∧ a1 < -2021 :=
by
  sorry

end NUMINAMATH_GPT_first_term_of_arithmetic_sequence_l2330_233037


namespace NUMINAMATH_GPT_arcsin_one_half_eq_pi_six_l2330_233070

theorem arcsin_one_half_eq_pi_six :
  Real.arcsin (1 / 2) = Real.pi / 6 :=
by
  sorry

end NUMINAMATH_GPT_arcsin_one_half_eq_pi_six_l2330_233070


namespace NUMINAMATH_GPT_tim_biking_time_l2330_233021

theorem tim_biking_time
  (work_days : ℕ := 5) 
  (distance_to_work : ℕ := 20) 
  (weekend_ride : ℕ := 200) 
  (speed : ℕ := 25) 
  (weekly_work_distance := 2 * distance_to_work * work_days)
  (total_distance := weekly_work_distance + weekend_ride) : 
  (total_distance / speed = 16) := 
by
  sorry

end NUMINAMATH_GPT_tim_biking_time_l2330_233021


namespace NUMINAMATH_GPT_problem_counts_correct_pairs_l2330_233009

noncomputable def count_valid_pairs : ℝ :=
  sorry

theorem problem_counts_correct_pairs :
  count_valid_pairs = 128 :=
by
  sorry

end NUMINAMATH_GPT_problem_counts_correct_pairs_l2330_233009


namespace NUMINAMATH_GPT_midlines_tangent_fixed_circle_l2330_233022

-- Definitions of geometric objects and properties
structure Point :=
(x : ℝ) (y : ℝ)

structure Circle :=
(center : Point) (radius : ℝ)

-- Assumptions (conditions)
variable (ω1 ω2 : Circle)
variable (l1 l2 : Point → Prop) -- Representing line equations in terms of points
variable (angle : Point → Prop) -- Representing the given angle sides

-- Tangency conditions
axiom tangency1 : ∀ p : Point, l1 p → p ≠ ω1.center ∧ (ω1.center.x - p.x) ^ 2 + (ω1.center.y - p.y) ^ 2 = ω1.radius ^ 2
axiom tangency2 : ∀ p : Point, l2 p → p ≠ ω2.center ∧ (ω2.center.x - p.x) ^ 2 + (ω2.center.y - p.y) ^ 2 = ω2.radius ^ 2

-- Non-intersecting condition for circles
axiom nonintersecting : (ω1.center.x - ω2.center.x) ^ 2 + (ω1.center.y - ω2.center.y) ^ 2 > (ω1.radius + ω2.radius) ^ 2

-- Conditions for tangent circles and middle line being between them
axiom betweenness : ∀ p, angle p → (ω1.center.y < p.y ∧ p.y < ω2.center.y)

-- Midline definition and fixed circle condition
theorem midlines_tangent_fixed_circle :
  ∃ (O : Point) (d : ℝ), ∀ (T : Point → Prop), 
  (∃ (p1 p2 : Point), l1 p1 ∧ l2 p2 ∧ T p1 ∧ T p2) →
  (∀ (m : Point), T m ↔ ∃ (p1 p2 p3 p4 : Point), T p1 ∧ T p2 ∧ angle p3 ∧ angle p4 ∧ 
  m.x = (p1.x + p2.x + p3.x + p4.x) / 4 ∧ m.y = (p1.y + p2.y + p3.y + p4.y) / 4) → 
  (∀ (m : Point), (m.x - O.x) ^ 2 + (m.y - O.y) ^ 2 = d^2)
:= 
sorry

end NUMINAMATH_GPT_midlines_tangent_fixed_circle_l2330_233022


namespace NUMINAMATH_GPT_employed_females_percentage_l2330_233024

-- Definitions of the conditions
def employment_rate : ℝ := 0.60
def male_employment_rate : ℝ := 0.15

-- The theorem to prove
theorem employed_females_percentage : employment_rate - male_employment_rate = 0.45 := by
  sorry

end NUMINAMATH_GPT_employed_females_percentage_l2330_233024


namespace NUMINAMATH_GPT_quadratic_inequality_solutions_l2330_233036

theorem quadratic_inequality_solutions (a x : ℝ) :
  (x^2 - (2+a)*x + 2*a > 0) → (
    (a < 2  → (x < a ∨ x > 2)) ∧
    (a = 2  → (x ≠ 2)) ∧
    (a > 2  → (x < 2 ∨ x > a))
  ) :=
by sorry

end NUMINAMATH_GPT_quadratic_inequality_solutions_l2330_233036


namespace NUMINAMATH_GPT_zeros_in_Q_l2330_233097

def R_k (k : ℕ) : ℤ := (7^k - 1) / 6

def Q : ℤ := (7^30 - 1) / (7^6 - 1)

def count_zeros (n : ℤ) : ℕ := sorry

theorem zeros_in_Q : count_zeros Q = 470588 :=
by sorry

end NUMINAMATH_GPT_zeros_in_Q_l2330_233097


namespace NUMINAMATH_GPT_range_of_m_l2330_233004

theorem range_of_m (α : ℝ) (m : ℝ) (h1 : π < α ∧ α < 2 * π ∨ 3 * π < α ∧ α < 4 * π) 
(h2 : Real.sin α = (2 * m - 3) / (4 - m)) : 
  -1 < m ∧ m < (3 : ℝ) / 2 :=
  sorry

end NUMINAMATH_GPT_range_of_m_l2330_233004


namespace NUMINAMATH_GPT_number_of_exercise_books_l2330_233090

theorem number_of_exercise_books (pencils pens exercise_books : ℕ) (h_ratio : (14 * pens = 4 * pencils) ∧ (14 * exercise_books = 3 * pencils)) (h_pencils : pencils = 140) : exercise_books = 30 :=
by
  sorry

end NUMINAMATH_GPT_number_of_exercise_books_l2330_233090


namespace NUMINAMATH_GPT_not_perfect_square_T_l2330_233015

noncomputable def operation (x y : ℝ) : ℝ := (x * y + 4) / (x + y)

axiom associative {x y z : ℝ} (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : 0 < z) :
  operation x (operation y z) = operation (operation x y) z

noncomputable def T (n : ℕ) : ℝ :=
  if h : n ≥ 4 then
    (List.range (n - 2)).foldr (λ x acc => operation (x + 3) acc) 3
  else 0

theorem not_perfect_square_T (n : ℕ) (h : n ≥ 4) :
  ¬ (∃ k : ℕ, (96 / (T n - 2) : ℝ) = k ^ 2) :=
sorry

end NUMINAMATH_GPT_not_perfect_square_T_l2330_233015


namespace NUMINAMATH_GPT_geometric_sequence_11th_term_l2330_233068

theorem geometric_sequence_11th_term (a r : ℕ) :
    a * r^4 = 3 →
    a * r^7 = 24 →
    a * r^10 = 192 := by
    sorry

end NUMINAMATH_GPT_geometric_sequence_11th_term_l2330_233068


namespace NUMINAMATH_GPT_cos_neg_570_eq_neg_sqrt3_div_2_l2330_233075

theorem cos_neg_570_eq_neg_sqrt3_div_2 :
  Real.cos (-(570 : ℝ) * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end NUMINAMATH_GPT_cos_neg_570_eq_neg_sqrt3_div_2_l2330_233075


namespace NUMINAMATH_GPT_complement_of_P_in_U_l2330_233057

def universal_set : Set ℝ := Set.univ
def set_P : Set ℝ := { x | x^2 - 5 * x - 6 ≥ 0 }
def complement_in_U (U : Set ℝ) (P : Set ℝ) : Set ℝ := U \ P

theorem complement_of_P_in_U :
  complement_in_U universal_set set_P = { x | -1 < x ∧ x < 6 } :=
by
  sorry

end NUMINAMATH_GPT_complement_of_P_in_U_l2330_233057


namespace NUMINAMATH_GPT_minimum_shots_to_hit_ship_l2330_233032

def is_ship_hit (shots : Finset (Fin 7 × Fin 7)) : Prop :=
  -- Assuming the ship can be represented by any 4 consecutive points in a row
  ∀ r : Fin 7, ∃ c1 c2 c3 c4 : Fin 7, 
    (0 ≤ c1.1 ∧ c1.1 ≤ 6 ∧ c1.1 + 3 = c4.1) ∧
    (0 ≤ c2.1 ∧ c2.1 ≤ 6 ∧ c2.1 = c1.1 + 1) ∧
    (0 ≤ c3.1 ∧ c3.1 ≤ 6 ∧ c3.1 = c1.1 + 2) ∧
    (r, c1) ∈ shots ∧ (r, c2) ∈ shots ∧ (r, c3) ∈ shots ∧ (r, c4) ∈ shots

theorem minimum_shots_to_hit_ship : ∃ shots : Finset (Fin 7 × Fin 7), 
  shots.card = 12 ∧ is_ship_hit shots :=
by 
  sorry

end NUMINAMATH_GPT_minimum_shots_to_hit_ship_l2330_233032


namespace NUMINAMATH_GPT_greatest_possible_perimeter_l2330_233014

noncomputable def max_perimeter (x : ℕ) : ℕ := if 5 * x > 20 ∧ x + 20 > 4 * x ∧ 4 * x + 20 > x then x + 4 * x + 20 else 0

theorem greatest_possible_perimeter : 
  (∀ x : ℕ, 5 * x > 20 ∧ x + 20 > 4 * x ∧ 4 * x + 20 > x → max_perimeter x ≤ 50) ∧ (∃ x : ℕ, 5 * x > 20 ∧ x + 20 > 4 * x ∧ 4 * x + 20 > x ∧ max_perimeter x = 50) :=
by
  sorry

end NUMINAMATH_GPT_greatest_possible_perimeter_l2330_233014


namespace NUMINAMATH_GPT_one_percent_as_decimal_l2330_233023

theorem one_percent_as_decimal : (1 / 100 : ℝ) = 0.01 := 
by 
  sorry

end NUMINAMATH_GPT_one_percent_as_decimal_l2330_233023


namespace NUMINAMATH_GPT_derivative_y_over_x_l2330_233091

noncomputable def x (t : ℝ) : ℝ := (t^2 * Real.log t) / (1 - t^2) + Real.log (Real.sqrt (1 - t^2))
noncomputable def y (t : ℝ) : ℝ := (t / Real.sqrt (1 - t^2)) * Real.arcsin t + Real.log (Real.sqrt (1 - t^2))

theorem derivative_y_over_x (t : ℝ) (ht : t ≠ 0) (h1 : t ≠ 1) (hneg1 : t ≠ -1) : 
  (deriv y t) / (deriv x t) = (Real.arcsin t * Real.sqrt (1 - t^2)) / (2 * t * Real.log t) :=
by
  sorry

end NUMINAMATH_GPT_derivative_y_over_x_l2330_233091


namespace NUMINAMATH_GPT_solve_for_x_l2330_233099

theorem solve_for_x (x : ℝ) (h : 7 - 2 * x = -3) : x = 5 := by
  sorry

end NUMINAMATH_GPT_solve_for_x_l2330_233099


namespace NUMINAMATH_GPT_sum_product_smallest_number_l2330_233065

theorem sum_product_smallest_number (x y : ℝ) (h1 : x + y = 18) (h2 : x * y = 80) : min x y = 8 :=
  sorry

end NUMINAMATH_GPT_sum_product_smallest_number_l2330_233065


namespace NUMINAMATH_GPT_cotangent_positives_among_sequence_l2330_233082

def cotangent_positive_count (n : ℕ) : ℕ :=
  if n ≤ 2019 then
    let count := (n / 4) * 3 + if n % 4 ≠ 0 then (3 + 1 - max 0 ((n % 4) - 1)) else 0
    count
  else 0

theorem cotangent_positives_among_sequence :
  cotangent_positive_count 2019 = 1515 := sorry

end NUMINAMATH_GPT_cotangent_positives_among_sequence_l2330_233082


namespace NUMINAMATH_GPT_length_of_platform_l2330_233062

theorem length_of_platform (l t p : ℝ) (h1 : (l / t) = (l + p) / (5 * t)) : p = 4 * l :=
by
  sorry

end NUMINAMATH_GPT_length_of_platform_l2330_233062


namespace NUMINAMATH_GPT_part1_l2330_233018

noncomputable def P : Set ℝ := {x | (1 / 2) ≤ x ∧ x ≤ 1}
noncomputable def Q (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 1}
def U : Set ℝ := Set.univ
noncomputable def complement_P : Set ℝ := {x | x < (1 / 2)} ∪ {x | x > 1}

theorem part1 (a : ℝ) (h : a = 1) : 
  (complement_P ∩ Q a) = {x | 1 < x ∧ x ≤ 2} :=
sorry

end NUMINAMATH_GPT_part1_l2330_233018


namespace NUMINAMATH_GPT_compute_expression_l2330_233056

theorem compute_expression (w : ℂ) (hw : w = Complex.exp (Complex.I * (6 * Real.pi / 11))) (hwp : w^11 = 1) :
  (w / (1 + w^3) + w^2 / (1 + w^6) + w^3 / (1 + w^9) = -2) :=
sorry

end NUMINAMATH_GPT_compute_expression_l2330_233056


namespace NUMINAMATH_GPT_intersection_P_Q_l2330_233019

-- Definitions based on conditions
def P : Set ℝ := { y | ∃ x : ℝ, y = x + 1 }
def Q : Set ℝ := { y | ∃ x : ℝ, y = 1 - x }

-- Proof statement to show P ∩ Q = Set.univ
theorem intersection_P_Q : P ∩ Q = Set.univ := by
  sorry

end NUMINAMATH_GPT_intersection_P_Q_l2330_233019


namespace NUMINAMATH_GPT_find_correct_r_l2330_233071

noncomputable def ellipse_tangent_circle_intersection : Prop :=
  ∃ (E F : ℝ × ℝ) (r : ℝ), E ∈ { p : ℝ × ℝ | p.1^2 / 4 + p.2^2 / 3 = 1 } ∧
                             F ∈ { p : ℝ × ℝ | p.1^2 / 4 + p.2^2 / 3 = 1 } ∧ 
                             (E ≠ F) ∧
                             ((E.1 - 2)^2 + (E.2 - 3/2)^2 = r^2) ∧
                             ((F.1 - 2)^2 + (F.2 - 3/2)^2 = r^2) ∧
                             r = (Real.sqrt 37) / 37

theorem find_correct_r : ellipse_tangent_circle_intersection :=
sorry

end NUMINAMATH_GPT_find_correct_r_l2330_233071


namespace NUMINAMATH_GPT_percentage_class_takes_lunch_l2330_233076

theorem percentage_class_takes_lunch (total_students boys girls : ℕ)
  (h_total: total_students = 100)
  (h_ratio: boys = 6 * total_students / (6 + 4))
  (h_girls: girls = 4 * total_students / (6 + 4))
  (boys_lunch_ratio : ℝ)
  (girls_lunch_ratio : ℝ)
  (h_boys_lunch_ratio : boys_lunch_ratio = 0.60)
  (h_girls_lunch_ratio : girls_lunch_ratio = 0.40):
  ((boys_lunch_ratio * boys + girls_lunch_ratio * girls) / total_students) * 100 = 52 :=
by
  sorry

end NUMINAMATH_GPT_percentage_class_takes_lunch_l2330_233076


namespace NUMINAMATH_GPT_corrected_mean_l2330_233096

open Real

theorem corrected_mean (n : ℕ) (mu_incorrect : ℝ)
                      (x1 y1 x2 y2 x3 y3 : ℝ)
                      (h1 : mu_incorrect = 41)
                      (h2 : n = 50)
                      (h3 : x1 = 48 ∧ y1 = 23)
                      (h4 : x2 = 36 ∧ y2 = 42)
                      (h5 : x3 = 55 ∧ y3 = 28) :
                      ((mu_incorrect * n + (x1 - y1) + (x2 - y2) + (x3 - y3)) / n = 41.92) :=
by
  sorry

end NUMINAMATH_GPT_corrected_mean_l2330_233096


namespace NUMINAMATH_GPT_find_ab_l2330_233035

theorem find_ab (a b : ℝ) 
  (h1 : a + b = 5) 
  (h2 : a^3 + b^3 = 35) : a * b = 6 := 
by
  sorry

end NUMINAMATH_GPT_find_ab_l2330_233035


namespace NUMINAMATH_GPT_tim_income_percentage_less_l2330_233067

theorem tim_income_percentage_less (M T J : ℝ)
  (h₁ : M = 1.60 * T)
  (h₂ : M = 0.96 * J) :
  100 - (T / J) * 100 = 40 :=
by sorry

end NUMINAMATH_GPT_tim_income_percentage_less_l2330_233067


namespace NUMINAMATH_GPT_ganesh_average_speed_l2330_233034

variable (D : ℝ) -- distance between the two towns in kilometers
variable (V : ℝ) -- average speed from x to y in km/hr

-- Conditions
variable (h1 : V > 0) -- Speed must be positive
variable (h2 : 30 > 0) -- Speed must be positive
variable (h3 : 40 = (2 * D) / ((D / V) + (D / 30))) -- Average speed formula

theorem ganesh_average_speed : V = 60 :=
by {
  sorry
}

end NUMINAMATH_GPT_ganesh_average_speed_l2330_233034


namespace NUMINAMATH_GPT_boat_travel_time_l2330_233052

theorem boat_travel_time (x : ℝ) (T : ℝ) (h0 : 0 ≤ x) (h1 : x ≠ 15.6) 
    (h2 : 96 = (15.6 - x) * T) 
    (h3 : 96 = (15.6 + x) * 5) : 
    T = 8 :=
by 
  sorry

end NUMINAMATH_GPT_boat_travel_time_l2330_233052


namespace NUMINAMATH_GPT_solve_for_x_l2330_233084

theorem solve_for_x :
  ∃ x : ℕ, (12 ^ 3) * (6 ^ x) / 432 = 144 ∧ x = 2 := by
  sorry

end NUMINAMATH_GPT_solve_for_x_l2330_233084


namespace NUMINAMATH_GPT_numbers_divisible_by_three_l2330_233053

theorem numbers_divisible_by_three (a b : ℕ) (h1 : a = 150) (h2 : b = 450) :
  ∃ n : ℕ, ∀ x : ℕ, (a < x) → (x < b) → (x % 3 = 0) → (x = 153 + 3 * (n - 1)) :=
by
  sorry

end NUMINAMATH_GPT_numbers_divisible_by_three_l2330_233053


namespace NUMINAMATH_GPT_eval_expression_l2330_233069

theorem eval_expression : 
  (520 * 0.43 / 0.26 - 217 * (2 + 3/7)) - (31.5 / (12 + 3/5) + 114 * (2 + 1/3) + (61 + 1/2)) = 0.5 := 
by
  sorry

end NUMINAMATH_GPT_eval_expression_l2330_233069


namespace NUMINAMATH_GPT_no_real_solutions_l2330_233094

theorem no_real_solutions :
  ∀ x : ℝ, (2 * x - 6) ^ 2 + 4 ≠ -(x - 3) :=
by
  intro x
  sorry

end NUMINAMATH_GPT_no_real_solutions_l2330_233094


namespace NUMINAMATH_GPT_max_vector_sum_l2330_233013

open Real EuclideanSpace

noncomputable def circle_center : ℝ × ℝ := (3, 0)
noncomputable def radius : ℝ := 2
noncomputable def distance_AB : ℝ := 2 * sqrt 3

theorem max_vector_sum {A B : ℝ × ℝ} 
    (hA_on_circle : dist A circle_center = radius)
    (hB_on_circle : dist B circle_center = radius)
    (hAB_eq : dist A B = distance_AB) :
    (dist (0,0) ((A.1 + B.1, A.2 + B.2))) ≤ 8 :=
by 
  sorry

end NUMINAMATH_GPT_max_vector_sum_l2330_233013


namespace NUMINAMATH_GPT_gcd_lcm_product_l2330_233079

theorem gcd_lcm_product (a b : ℕ) (h1 : a = 24) (h2 : b = 45) : (Int.gcd a b * Nat.lcm a b) = 1080 := by
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_gcd_lcm_product_l2330_233079


namespace NUMINAMATH_GPT_problem_statement_l2330_233085

-- Define the basic problem setup
def defect_rate (p : ℝ) := p = 0.01
def sample_size (n : ℕ) := n = 200

-- Define the binomial distribution
noncomputable def binomial_expectation (n : ℕ) (p : ℝ) := n * p
noncomputable def binomial_variance (n : ℕ) (p : ℝ) := n * p * (1 - p)

-- The actual statement that we will prove
theorem problem_statement (p : ℝ) (n : ℕ) (X : ℕ → ℕ) 
  (h_defect_rate : defect_rate p) 
  (h_sample_size : sample_size n) 
  (h_distribution : ∀ k, X k = (n.choose k) * (p ^ k) * ((1 - p) ^ (n - k))) 
  : binomial_expectation n p = 2 ∧ binomial_variance n p = 1.98 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l2330_233085


namespace NUMINAMATH_GPT_area_of_triangle_POF_l2330_233095

noncomputable def origin : (ℝ × ℝ) := (0, 0)
noncomputable def focus : (ℝ × ℝ) := (Real.sqrt 2, 0)

noncomputable def parabola (x y : ℝ) : Prop :=
  y ^ 2 = 4 * Real.sqrt 2 * x

noncomputable def point_on_parabola (x y : ℝ) : Prop :=
  parabola x y

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

noncomputable def PF_eq_4sqrt2 (x y : ℝ) : Prop :=
  distance x y (Real.sqrt 2) 0 = 4 * Real.sqrt 2

theorem area_of_triangle_POF (x y : ℝ) 
  (h1: point_on_parabola x y)
  (h2: PF_eq_4sqrt2 x y) :
   1 / 2 * distance 0 0 (Real.sqrt 2) 0 * |y| = 2 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_area_of_triangle_POF_l2330_233095


namespace NUMINAMATH_GPT_total_participants_l2330_233001

theorem total_participants (F M : ℕ)
  (h1 : F / 2 = 130)
  (h2 : F / 2 + M / 4 = (F + M) / 3) : 
  F + M = 780 := 
by 
  sorry

end NUMINAMATH_GPT_total_participants_l2330_233001


namespace NUMINAMATH_GPT_rate_per_kg_of_grapes_l2330_233088

theorem rate_per_kg_of_grapes : 
  ∀ (rate_per_kg_grapes : ℕ), 
    (10 * rate_per_kg_grapes + 9 * 55 = 1195) → 
    rate_per_kg_grapes = 70 := 
by
  intros rate_per_kg_grapes h
  sorry

end NUMINAMATH_GPT_rate_per_kg_of_grapes_l2330_233088


namespace NUMINAMATH_GPT_candy_problem_l2330_233089

theorem candy_problem
  (x y m : ℤ)
  (hx : x ≥ 0)
  (hy : y ≥ 0)
  (hxy : x + y = 176)
  (hcond : x - m * (y - 16) = 47)
  (hm : m > 1) :
  x ≥ 131 := 
sorry

end NUMINAMATH_GPT_candy_problem_l2330_233089


namespace NUMINAMATH_GPT_new_volume_of_cylinder_l2330_233041

theorem new_volume_of_cylinder (r h : ℝ) (π : ℝ := Real.pi) (V : ℝ := π * r^2 * h) (hV : V = 15) :
  let r_new := 3 * r
  let h_new := 4 * h
  let V_new := π * (r_new)^2 * h_new
  V_new = 540 :=
by
  sorry

end NUMINAMATH_GPT_new_volume_of_cylinder_l2330_233041


namespace NUMINAMATH_GPT_number_of_classmates_l2330_233017

theorem number_of_classmates (n m : ℕ) (h₁ : n < 100) (h₂ : m = 9)
:(2 ^ 6 - 1) = 63 → 63 / m = 7 := by
  intros 
  sorry

end NUMINAMATH_GPT_number_of_classmates_l2330_233017


namespace NUMINAMATH_GPT_maya_additional_cars_l2330_233008

theorem maya_additional_cars : 
  ∃ n : ℕ, 29 + n ≥ 35 ∧ (29 + n) % 7 = 0 ∧ n = 6 :=
by
  sorry

end NUMINAMATH_GPT_maya_additional_cars_l2330_233008


namespace NUMINAMATH_GPT_smallest_positive_period_monotonically_increasing_interval_minimum_value_a_of_triangle_l2330_233045

noncomputable def f (x : ℝ) := 2 * (Real.cos x)^2 + Real.sin (7 * Real.pi / 6 - 2 * x) - 1

theorem smallest_positive_period :
  ∃ T > 0, ∀ x, f (x + T) = f x := 
by 
  -- Proof omitted
  sorry

theorem monotonically_increasing_interval :
  ∃ k : ℤ, ∀ x y, k * Real.pi - Real.pi / 3 ≤ x ∧ x ≤ k * Real.pi + Real.pi / 6 → 
               k * Real.pi - Real.pi / 3 ≤ y ∧ y ≤  k * Real.pi + Real.pi / 6 →
               x ≤ y → f x ≤ f y := 
by 
  -- Proof omitted
  sorry

theorem minimum_value_a_of_triangle (A B C a b c : ℝ) 
  (h₀ : f A = 1/2) 
  (h₁ : B^2 - C^2 - B * C * Real.cos A - a^2 = 4) :
  a ≥ 2 * Real.sqrt 2 :=
by 
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_smallest_positive_period_monotonically_increasing_interval_minimum_value_a_of_triangle_l2330_233045


namespace NUMINAMATH_GPT_proof_moles_HNO3_proof_molecular_weight_HNO3_l2330_233050

variable (n_CaO : ℕ) (molar_mass_H : ℕ) (molar_mass_N : ℕ) (molar_mass_O : ℕ)

def verify_moles_HNO3 (n_CaO : ℕ) : ℕ :=
  2 * n_CaO

def verify_molecular_weight_HNO3 (molar_mass_H molar_mass_N molar_mass_O : ℕ) : ℕ :=
  molar_mass_H + molar_mass_N + 3 * molar_mass_O

theorem proof_moles_HNO3 :
  n_CaO = 7 →
  verify_moles_HNO3 n_CaO = 14 :=
sorry

theorem proof_molecular_weight_HNO3 :
  molar_mass_H = 101 / 100 ∧ molar_mass_N = 1401 / 100 ∧ molar_mass_O = 1600 / 100 →
  verify_molecular_weight_HNO3 molar_mass_H molar_mass_N molar_mass_O = 6302 / 100 :=
sorry

end NUMINAMATH_GPT_proof_moles_HNO3_proof_molecular_weight_HNO3_l2330_233050

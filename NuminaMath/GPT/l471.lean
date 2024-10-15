import Mathlib

namespace NUMINAMATH_GPT_fish_lives_longer_than_dog_l471_47109

-- Definitions based on conditions
def hamster_lifespan : ℝ := 2.5
def dog_lifespan : ℝ := 4 * hamster_lifespan
def fish_lifespan : ℝ := 12

-- Theorem stating the desired proof
theorem fish_lives_longer_than_dog :
  fish_lifespan - dog_lifespan = 2 := 
sorry

end NUMINAMATH_GPT_fish_lives_longer_than_dog_l471_47109


namespace NUMINAMATH_GPT_string_length_correct_l471_47152

noncomputable def cylinder_circumference : ℝ := 6
noncomputable def cylinder_height : ℝ := 18
noncomputable def number_of_loops : ℕ := 6

noncomputable def height_per_loop : ℝ := cylinder_height / number_of_loops
noncomputable def hypotenuse_per_loop : ℝ := Real.sqrt (cylinder_circumference ^ 2 + height_per_loop ^ 2)
noncomputable def total_string_length : ℝ := number_of_loops * hypotenuse_per_loop

theorem string_length_correct :
  total_string_length = 18 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_GPT_string_length_correct_l471_47152


namespace NUMINAMATH_GPT_no_three_digit_whole_number_solves_log_eq_l471_47181

noncomputable def log_function (b : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log b

theorem no_three_digit_whole_number_solves_log_eq :
  ¬ ∃ n : ℤ, (100 ≤ n ∧ n < 1000) ∧ log_function (3 * n) 10 + log_function (7 * n) 10 = 1 :=
by
  sorry

end NUMINAMATH_GPT_no_three_digit_whole_number_solves_log_eq_l471_47181


namespace NUMINAMATH_GPT_calendar_sum_multiple_of_4_l471_47189

theorem calendar_sum_multiple_of_4 (a : ℕ) : 
  let top_left := a - 1
  let bottom_left := a + 6
  let bottom_right := a + 7
  top_left + a + bottom_left + bottom_right = 4 * (a + 3) :=
by
  sorry

end NUMINAMATH_GPT_calendar_sum_multiple_of_4_l471_47189


namespace NUMINAMATH_GPT_pq_r_sum_l471_47156

theorem pq_r_sum (p q r : ℝ) (h1 : p^3 - 18 * p^2 + 27 * p - 72 = 0) 
                 (h2 : 27 * q^3 - 243 * q^2 + 729 * q - 972 = 0)
                 (h3 : 3 * r = 9) : p + q + r = 18 :=
by
  sorry

end NUMINAMATH_GPT_pq_r_sum_l471_47156


namespace NUMINAMATH_GPT_min_inverse_sum_l471_47111

theorem min_inverse_sum (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : a + b = 4) : 1 ≤ (1/a) + (1/b) :=
by
  sorry

end NUMINAMATH_GPT_min_inverse_sum_l471_47111


namespace NUMINAMATH_GPT_positive_integer_triples_satisfying_conditions_l471_47182

theorem positive_integer_triples_satisfying_conditions :
  ∀ (a b c : ℕ), a^2 + b^2 + c^2 = 2005 ∧ a ≤ b ∧ b ≤ c →
  (a, b, c) = (23, 24, 30) ∨
  (a, b, c) = (12, 30, 31) ∨
  (a, b, c) = (9, 30, 32) ∨
  (a, b, c) = (4, 30, 33) ∨
  (a, b, c) = (15, 22, 36) ∨
  (a, b, c) = (9, 18, 40) ∨
  (a, b, c) = (4, 15, 42) :=
sorry

end NUMINAMATH_GPT_positive_integer_triples_satisfying_conditions_l471_47182


namespace NUMINAMATH_GPT_part_one_part_two_i_part_two_ii_l471_47138

noncomputable def f (x a b : ℝ) : ℝ := x^2 + a * x + b

theorem part_one (a b : ℝ) : 
  f (-a / 2 + 1) a b ≤ f (a^2 + 5 / 4) a b :=
sorry

theorem part_two_i (a b : ℝ) : 
  f 1 a b + f 3 a b - 2 * f 2 a b = 2 :=
sorry

theorem part_two_ii (a b : ℝ) : 
  ¬((|f 1 a b| < 1/2) ∧ (|f 2 a b| < 1/2) ∧ (|f 3 a b| < 1/2)) :=
sorry

end NUMINAMATH_GPT_part_one_part_two_i_part_two_ii_l471_47138


namespace NUMINAMATH_GPT_rugby_team_new_avg_weight_l471_47168

noncomputable def new_average_weight (original_players : ℕ) (original_avg_weight : ℕ) 
  (new_player_weights : List ℕ) : ℚ :=
  let total_original_weight := original_players * original_avg_weight
  let total_new_weight := new_player_weights.foldl (· + ·) 0
  let new_total_weight := total_original_weight + total_new_weight
  let new_total_players := original_players + new_player_weights.length
  (new_total_weight : ℚ) / (new_total_players : ℚ)

theorem rugby_team_new_avg_weight :
  new_average_weight 20 180 [210, 220, 230] = 185.22 := by
  sorry

end NUMINAMATH_GPT_rugby_team_new_avg_weight_l471_47168


namespace NUMINAMATH_GPT_net_distance_from_start_total_distance_driven_fuel_consumption_l471_47149

def driving_distances : List Int := [14, -3, 7, -3, 11, -4, -3, 11, 6, -7, 9]

theorem net_distance_from_start : List.sum driving_distances = 38 := by
  sorry

theorem total_distance_driven : List.sum (List.map Int.natAbs driving_distances) = 78 := by
  sorry

theorem fuel_consumption (fuel_rate : Float) (total_distance : Nat) : total_distance = 78 → total_distance.toFloat * fuel_rate = 7.8 := by
  intros h_total_distance
  rw [h_total_distance]
  norm_num
  sorry

end NUMINAMATH_GPT_net_distance_from_start_total_distance_driven_fuel_consumption_l471_47149


namespace NUMINAMATH_GPT_width_of_wall_is_two_l471_47132

noncomputable def volume_of_brick : ℝ := 20 * 10 * 7.5 / 10^6 -- Volume in cubic meters
def number_of_bricks : ℕ := 27000
noncomputable def volume_of_wall (width : ℝ) : ℝ := 27 * width * 0.75

theorem width_of_wall_is_two :
  ∃ (W : ℝ), volume_of_wall W = number_of_bricks * volume_of_brick ∧ W = 2 :=
by
  sorry

end NUMINAMATH_GPT_width_of_wall_is_two_l471_47132


namespace NUMINAMATH_GPT_trigonometric_identity_l471_47133

theorem trigonometric_identity
  (α : ℝ)
  (h : Real.tan α = 2) :
  (4 * Real.sin α ^ 3 - 2 * Real.cos α) / (5 * Real.cos α + 3 * Real.sin α) = 2 / 5 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l471_47133


namespace NUMINAMATH_GPT_Mr_Kishore_Savings_l471_47147

noncomputable def total_expenses := 
  5000 + 1500 + 4500 + 2500 + 2000 + 6100 + 3500 + 2700

noncomputable def monthly_salary (S : ℝ) := 
  total_expenses + 0.10 * S = S

noncomputable def savings (S : ℝ) := 
  0.10 * S

theorem Mr_Kishore_Savings : 
  ∃ S : ℝ, monthly_salary S ∧ savings S = 3422.22 :=
by
  sorry

end NUMINAMATH_GPT_Mr_Kishore_Savings_l471_47147


namespace NUMINAMATH_GPT_intersection_eq_inter_l471_47121

noncomputable def M : Set ℝ := { x | x^2 < 4 }
noncomputable def N : Set ℝ := { x | x^2 - 2 * x - 3 < 0 }
noncomputable def inter : Set ℝ := { x | -1 < x ∧ x < 2 }

theorem intersection_eq_inter : M ∩ N = inter :=
by sorry

end NUMINAMATH_GPT_intersection_eq_inter_l471_47121


namespace NUMINAMATH_GPT_determinant_A_l471_47187

def A : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![2, -1, 5], ![0, 4, -2], ![3, 0, 1]]

theorem determinant_A : Matrix.det A = -46 := by
  sorry

end NUMINAMATH_GPT_determinant_A_l471_47187


namespace NUMINAMATH_GPT_minimum_value_l471_47101

theorem minimum_value (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z = 6) :
    6 ≤ (x^2 + 2*y^2) / (x + y) + (x^2 + 2*z^2) / (x + z) + (y^2 + 2*z^2) / (y + z) :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_l471_47101


namespace NUMINAMATH_GPT_find_de_over_ef_l471_47108

-- Definitions based on problem conditions
variables {A B C D E F : Type} [AddCommGroup A] [AddCommGroup B] [AddCommGroup C] [AddCommGroup D] [AddCommGroup E] [AddCommGroup F] 
variables (a b c d e f : A) 
variables (α β γ δ : ℝ)

-- Conditions
-- AD:DB = 2:3
def d_def : A := (3 / 5) • a + (2 / 5) • b
-- BE:EC = 1:4
def e_def : A := (4 / 5) • b + (1 / 5) • c
-- Intersection F of DE and AC
def f_def : A := (5 • d) - (10 • e)

-- Target Proof
theorem find_de_over_ef (h_d: d = d_def a b) (h_e: e = e_def b c) (h_f: f = f_def d e):
  DE / EF = 1 / 5 := 
sorry

end NUMINAMATH_GPT_find_de_over_ef_l471_47108


namespace NUMINAMATH_GPT_range_of_a_l471_47195

theorem range_of_a (a : ℝ) (A : Set ℝ) (hA : ∀ x, x ∈ A ↔ a / (x - 1) < 1) (h_not_in : 2 ∉ A) : a ≥ 1 := 
sorry

end NUMINAMATH_GPT_range_of_a_l471_47195


namespace NUMINAMATH_GPT_percentage_reduction_in_price_of_oil_l471_47145

theorem percentage_reduction_in_price_of_oil :
  ∀ P : ℝ, ∀ R : ℝ, P = 800 / (800 / R - 5) ∧ R = 40 →
  (P - R) / P * 100 = 25 := by
  -- Assumptions
  intros P R h
  have hP : P = 800 / (800 / R - 5) := h.1
  have hR : R = 40 := h.2
  -- Result to be proved
  sorry

end NUMINAMATH_GPT_percentage_reduction_in_price_of_oil_l471_47145


namespace NUMINAMATH_GPT_intersection_A_B_l471_47164

def set_A : Set ℝ := {x : ℝ | |x| = x}
def set_B : Set ℝ := {x : ℝ | x^2 + x ≥ 0}
def set_intersection : Set ℝ := {x : ℝ | 0 ≤ x}

theorem intersection_A_B :
  (set_A ∩ set_B) = set_intersection :=
by
  sorry

-- You can verify if the Lean code builds successfully using Lean 4 environment.

end NUMINAMATH_GPT_intersection_A_B_l471_47164


namespace NUMINAMATH_GPT_possible_values_of_n_l471_47116

-- Definitions for the problem
def side_ab (n : ℕ) := 3 * n + 3
def side_ac (n : ℕ) := 2 * n + 10
def side_bc (n : ℕ) := 2 * n + 16

-- Triangle inequality conditions
def triangle_inequality_1 (n : ℕ) : Prop := side_ab n + side_ac n > side_bc n
def triangle_inequality_2 (n : ℕ) : Prop := side_ab n + side_bc n > side_ac n
def triangle_inequality_3 (n : ℕ) : Prop := side_ac n + side_bc n > side_ab n

-- Angle condition simplified (since the more complex one was invalid)
def angle_condition (n : ℕ) : Prop := side_ac n > side_ab n

-- Combined valid n range
def valid_n_range (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 12

-- The theorem to prove
theorem possible_values_of_n (n : ℕ) : triangle_inequality_1 n ∧
                                        triangle_inequality_2 n ∧
                                        triangle_inequality_3 n ∧
                                        angle_condition n ↔
                                        valid_n_range n :=
by
  sorry

end NUMINAMATH_GPT_possible_values_of_n_l471_47116


namespace NUMINAMATH_GPT_math_players_count_l471_47134

-- Define the conditions given in the problem.
def total_players : ℕ := 25
def physics_players : ℕ := 9
def both_subjects_players : ℕ := 5

-- Statement to be proven
theorem math_players_count :
  total_players = physics_players + both_subjects_players + (total_players - physics_players - both_subjects_players) → 
  total_players - physics_players + both_subjects_players = 21 := 
sorry

end NUMINAMATH_GPT_math_players_count_l471_47134


namespace NUMINAMATH_GPT_complex_modulus_proof_l471_47110

noncomputable def complex_modulus_example : ℝ := 
  Complex.abs ⟨3/4, -3⟩

theorem complex_modulus_proof : complex_modulus_example = Real.sqrt 153 / 4 := 
by 
  unfold complex_modulus_example
  sorry

end NUMINAMATH_GPT_complex_modulus_proof_l471_47110


namespace NUMINAMATH_GPT_intersecting_lines_find_m_l471_47153

theorem intersecting_lines_find_m : ∃ m : ℚ, 
  (∃ x y : ℚ, y = 4*x + 2 ∧ y = -3*x - 18 ∧ y = 2*x + m) ↔ m = -26/7 :=
by
  sorry

end NUMINAMATH_GPT_intersecting_lines_find_m_l471_47153


namespace NUMINAMATH_GPT_total_profit_correct_l471_47131

variables (x y : ℝ) -- B's investment and period
variables (B_profit : ℝ) -- profit received by B
variable (A_investment : ℝ) -- A's investment

-- Given conditions
def A_investment_cond := A_investment = 3 * x
def period_cond := 2 * y
def B_profit_given := B_profit = 4500
def total_profit := 7 * B_profit

theorem total_profit_correct :
  (A_investment = 3 * x)
  ∧ (B_profit = 4500)
  ∧ ((6 * x * 2 * y) / (x * y) = 6)
  → total_profit = 31500 :=
by sorry

end NUMINAMATH_GPT_total_profit_correct_l471_47131


namespace NUMINAMATH_GPT_t1_eq_t2_l471_47136

variable (n : ℕ)
variable (s₁ s₂ s₃ : ℝ)
variable (t₁ t₂ : ℝ)
variable (S1 S2 S3 : ℝ)

-- Conditions
axiom h1 : S1 = s₁
axiom h2 : S2 = s₂
axiom h3 : S3 = s₃
axiom h4 : t₁ = s₂^2 - s₁ * s₃
axiom h5 : t₂ = ( (s₁ - s₃) / 2 )^2
axiom h6 : s₁ + s₃ = 2 * s₂

theorem t1_eq_t2 : t₁ = t₂ := by
  sorry

end NUMINAMATH_GPT_t1_eq_t2_l471_47136


namespace NUMINAMATH_GPT_original_selling_price_l471_47178

theorem original_selling_price (C : ℝ) (h : 1.60 * C = 2560) : 1.40 * C = 2240 :=
by
  sorry

end NUMINAMATH_GPT_original_selling_price_l471_47178


namespace NUMINAMATH_GPT_age_of_B_l471_47125

variables (A B C : ℝ)

theorem age_of_B :
  (A + B + C) / 3 = 26 →
  (A + C) / 2 = 29 →
  B = 20 :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_age_of_B_l471_47125


namespace NUMINAMATH_GPT_cost_price_of_toy_l471_47162

theorem cost_price_of_toy (x : ℝ) (selling_price_per_toy : ℝ) (gain : ℝ) 
  (sale_price : ℝ) (number_of_toys : ℕ) (selling_total : ℝ) (gain_condition : ℝ) :
  (selling_total = number_of_toys * selling_price_per_toy) →
  (selling_price_per_toy = x + gain) →
  (gain = gain_condition / number_of_toys) → 
  (gain_condition = 3 * x) →
  selling_total = 25200 → number_of_toys = 18 → x = 1200 :=
by
  sorry

end NUMINAMATH_GPT_cost_price_of_toy_l471_47162


namespace NUMINAMATH_GPT_solve_for_t_l471_47170

theorem solve_for_t (t : ℝ) (h1 : x = 1 - 4 * t) (h2 : y = 2 * t - 2) : x = y → t = 1/2 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_t_l471_47170


namespace NUMINAMATH_GPT_exhibition_admission_fees_ratio_l471_47172

theorem exhibition_admission_fees_ratio
  (a c : ℕ)
  (h1 : 30 * a + 15 * c = 2925)
  (h2 : a % 5 = 0)
  (h3 : c % 5 = 0) :
  (a / 5 = c / 5) :=
by
  sorry

end NUMINAMATH_GPT_exhibition_admission_fees_ratio_l471_47172


namespace NUMINAMATH_GPT_goldfish_cost_discrete_points_l471_47142

def goldfish_cost (n : ℕ) : ℝ :=
  0.25 * n + 5

theorem goldfish_cost_discrete_points :
  ∀ n : ℕ, 5 ≤ n ∧ n ≤ 20 → ∃ k : ℕ, goldfish_cost n = goldfish_cost k ∧ 5 ≤ k ∧ k ≤ 20 :=
by sorry

end NUMINAMATH_GPT_goldfish_cost_discrete_points_l471_47142


namespace NUMINAMATH_GPT_least_positive_angle_l471_47175

theorem least_positive_angle (θ : ℝ) (h : Real.cos (10 * Real.pi / 180) = Real.sin (15 * Real.pi / 180) + Real.sin θ) :
  θ = 32.5 * Real.pi / 180 := 
sorry

end NUMINAMATH_GPT_least_positive_angle_l471_47175


namespace NUMINAMATH_GPT_power_function_inequality_l471_47161

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^a

theorem power_function_inequality (x : ℝ) (a : ℝ) : (x > 1) → (f x a < x) ↔ (a < 1) :=
by
  sorry

end NUMINAMATH_GPT_power_function_inequality_l471_47161


namespace NUMINAMATH_GPT_inequality_proof_l471_47192

theorem inequality_proof (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : x + y ≤ (y^2 / x) + (x^2 / y) :=
sorry

end NUMINAMATH_GPT_inequality_proof_l471_47192


namespace NUMINAMATH_GPT_polar_curve_symmetry_l471_47171

theorem polar_curve_symmetry :
  ∀ (ρ θ : ℝ), ρ = 4 * Real.sin (θ - π / 3) → 
  ∃ k : ℤ, θ = 5 * π / 6 + k * π :=
sorry

end NUMINAMATH_GPT_polar_curve_symmetry_l471_47171


namespace NUMINAMATH_GPT_medium_stores_in_sample_l471_47184

theorem medium_stores_in_sample :
  let total_stores := 300
  let large_stores := 30
  let medium_stores := 75
  let small_stores := 195
  let sample_size := 20
  sample_size * (medium_stores/total_stores) = 5 :=
by
  sorry

end NUMINAMATH_GPT_medium_stores_in_sample_l471_47184


namespace NUMINAMATH_GPT_monomial_properties_l471_47151

def coefficient (m : ℝ) := -3
def degree (x_exp y_exp : ℕ) := x_exp + y_exp

theorem monomial_properties :
  ∀ (x_exp y_exp : ℕ), coefficient (-3) = -3 ∧ degree 2 1 = 3 :=
by
  sorry

end NUMINAMATH_GPT_monomial_properties_l471_47151


namespace NUMINAMATH_GPT_toothpick_sequence_l471_47118

theorem toothpick_sequence (a d n : ℕ) (h1 : a = 6) (h2 : d = 4) (h3 : n = 150) : a + (n - 1) * d = 602 := by
  sorry

end NUMINAMATH_GPT_toothpick_sequence_l471_47118


namespace NUMINAMATH_GPT_solution_set_of_inequality_system_l471_47113

theorem solution_set_of_inequality_system (x : ℝ) :
  (4 * x + 1 > 2) ∧ (1 - 2 * x < 7) ↔ (x > 1 / 4) :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_system_l471_47113


namespace NUMINAMATH_GPT_number_of_women_l471_47183

theorem number_of_women
    (n : ℕ) -- number of men
    (d_m : ℕ) -- number of dances each man had
    (d_w : ℕ) -- number of dances each woman had
    (total_men : n = 15) -- there are 15 men
    (each_man_dances : d_m = 4) -- each man danced with 4 women
    (each_woman_dances : d_w = 3) -- each woman danced with 3 men
    (total_dances : n * d_m = w * d_w): -- total dances are the same when counted from both sides
  w = 20 := sorry -- There should be exactly 20 women.


end NUMINAMATH_GPT_number_of_women_l471_47183


namespace NUMINAMATH_GPT_gcd_8251_6105_l471_47158

theorem gcd_8251_6105 : Nat.gcd 8251 6105 = 37 := by
  sorry

end NUMINAMATH_GPT_gcd_8251_6105_l471_47158


namespace NUMINAMATH_GPT_triangle_side_length_c_l471_47104

theorem triangle_side_length_c (a b : ℝ) (α β γ : ℝ) (h_angle_sum : α + β + γ = 180) (h_angle_eq : 3 * α + 2 * β = 180) (h_a : a = 2) (h_b : b = 3) : 
∃ c : ℝ, c = 4 :=
by
  sorry

end NUMINAMATH_GPT_triangle_side_length_c_l471_47104


namespace NUMINAMATH_GPT_minimum_common_perimeter_l471_47130

namespace IsoscelesTriangles

def integer_sided_isosceles_triangles (a b x : ℕ) :=
  2 * a + 10 * x = 2 * b + 8 * x ∧
  5 * Real.sqrt (a^2 - 25 * x^2) = 4 * Real.sqrt (b^2 - 16 * x^2) ∧
  5 * b = 4 * (b + x)

theorem minimum_common_perimeter : ∃ (a b x : ℕ), 
  integer_sided_isosceles_triangles a b x ∧
  2 * a + 10 * x = 192 :=
by
  sorry

end IsoscelesTriangles

end NUMINAMATH_GPT_minimum_common_perimeter_l471_47130


namespace NUMINAMATH_GPT_todd_initial_gum_l471_47119

-- Define the conditions and the final result
def initial_gum (final_gum: Nat) (given_gum: Nat) : Nat := final_gum - given_gum

theorem todd_initial_gum :
  initial_gum 54 16 = 38 :=
by
  -- Use the initial_gum definition to state the problem
  -- The proof is skipped with sorry
  sorry

end NUMINAMATH_GPT_todd_initial_gum_l471_47119


namespace NUMINAMATH_GPT_fraction_to_decimal_l471_47198

theorem fraction_to_decimal : (3 : ℚ) / 40 = 0.075 :=
by
  sorry

end NUMINAMATH_GPT_fraction_to_decimal_l471_47198


namespace NUMINAMATH_GPT_set_equality_implies_a_value_l471_47140

theorem set_equality_implies_a_value (a : ℤ) : ({2, 3} : Set ℤ) = {2, 2 * a - 1} → a = 2 := 
by
  intro h
  sorry

end NUMINAMATH_GPT_set_equality_implies_a_value_l471_47140


namespace NUMINAMATH_GPT_solve_abs_inequality_l471_47129

theorem solve_abs_inequality (x : ℝ) : x + |2 * x + 3| ≥ 2 ↔ (x ≤ -5 ∨ x ≥ -1/3) :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_abs_inequality_l471_47129


namespace NUMINAMATH_GPT_problems_per_page_l471_47165

theorem problems_per_page (total_problems finished_problems pages_left problems_per_page : ℕ)
  (h1 : total_problems = 40)
  (h2 : finished_problems = 26)
  (h3 : pages_left = 2)
  (h4 : total_problems - finished_problems = pages_left * problems_per_page) :
  problems_per_page = 7 :=
by
  sorry

end NUMINAMATH_GPT_problems_per_page_l471_47165


namespace NUMINAMATH_GPT_cubic_inches_in_two_cubic_feet_l471_47177

theorem cubic_inches_in_two_cubic_feet (conv : 1 = 12) : 2 * (12 * 12 * 12) = 3456 :=
by
  sorry

end NUMINAMATH_GPT_cubic_inches_in_two_cubic_feet_l471_47177


namespace NUMINAMATH_GPT_total_windows_l471_47135

theorem total_windows (installed: ℕ) (hours_per_window: ℕ) (remaining_hours: ℕ) : installed = 8 → hours_per_window = 8 → remaining_hours = 48 → 
  (installed + remaining_hours / hours_per_window) = 14 := by 
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_total_windows_l471_47135


namespace NUMINAMATH_GPT_three_a_ge_two_b_plus_two_l471_47112

theorem three_a_ge_two_b_plus_two (a b : ℕ)
  (h1 : 0 < a)
  (h2 : 0 < b)
  (h3 : (a! * b!) % (a! + b!) = 0) :
  3 * a ≥ 2 * b + 2 :=
sorry

end NUMINAMATH_GPT_three_a_ge_two_b_plus_two_l471_47112


namespace NUMINAMATH_GPT_intersection_value_of_a_l471_47174

theorem intersection_value_of_a (a : ℝ) (A B : Set ℝ) 
  (hA : A = {0, 1, 3})
  (hB : B = {a + 1, a^2 + 2})
  (h_inter : A ∩ B = {1}) : 
  a = 0 :=
by
  sorry

end NUMINAMATH_GPT_intersection_value_of_a_l471_47174


namespace NUMINAMATH_GPT_sum_of_tens_l471_47102

theorem sum_of_tens (n : ℕ) (h : n = 100^10) : (n / 10) = 10^19 := by
  sorry

end NUMINAMATH_GPT_sum_of_tens_l471_47102


namespace NUMINAMATH_GPT_perimeter_of_triangle_l471_47100

theorem perimeter_of_triangle (side_length : ℕ) (num_sides : ℕ) (h1 : side_length = 7) (h2 : num_sides = 3) : 
  num_sides * side_length = 21 :=
by
  sorry

end NUMINAMATH_GPT_perimeter_of_triangle_l471_47100


namespace NUMINAMATH_GPT_cubic_function_properties_l471_47103

noncomputable def f (x : ℝ) : ℝ := x^3 - 6 * x^2 + 9 * x

theorem cubic_function_properties :
  (∀ (x : ℝ), deriv f x = 3 * x^2 - 12 * x + 9) ∧
  (f 1 = 4) ∧ 
  (deriv f 1 = 0) ∧
  (f 3 = 0) ∧ 
  (deriv f 3 = 0) ∧
  (f 0 = 0) :=
by
  sorry

end NUMINAMATH_GPT_cubic_function_properties_l471_47103


namespace NUMINAMATH_GPT_find_operation_l471_47148

theorem find_operation (a b : ℝ) (h_a : a = 0.137) (h_b : b = 0.098) :
  ((a + b) ^ 2 - (a - b) ^ 2) / (a * b) = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_operation_l471_47148


namespace NUMINAMATH_GPT_area_large_square_l471_47141

theorem area_large_square (a b c : ℝ) 
  (h1 : a^2 = b^2 + 32) 
  (h2 : 4*a = 4*c + 16) : a^2 = 100 := 
by {
  sorry
}

end NUMINAMATH_GPT_area_large_square_l471_47141


namespace NUMINAMATH_GPT_cube_root_neg_eight_l471_47167

theorem cube_root_neg_eight : ∃ x : ℝ, x^3 = -8 ∧ x = -2 :=
by {
  sorry
}

end NUMINAMATH_GPT_cube_root_neg_eight_l471_47167


namespace NUMINAMATH_GPT_sum_of_ages_l471_47186

variables (K T1 T2 : ℕ)

theorem sum_of_ages (h1 : K * T1 * T2 = 72) (h2 : T1 = T2) (h3 : T1 < K) : K + T1 + T2 = 14 :=
sorry

end NUMINAMATH_GPT_sum_of_ages_l471_47186


namespace NUMINAMATH_GPT_profit_value_l471_47185

variable (P : ℝ) -- Total profit made by the business in that year.
variable (MaryInvestment : ℝ) -- Mary's investment
variable (MikeInvestment : ℝ) -- Mike's investment
variable (MaryExtra : ℝ) -- Extra money received by Mary

-- Conditions
axiom mary_investment : MaryInvestment = 900
axiom mike_investment : MikeInvestment = 100
axiom mary_received_more : MaryExtra = 1600
axiom profit_shared_equally : (P / 3) / 2 + (MaryInvestment / (MaryInvestment + MikeInvestment)) * (2 * P / 3) 
                           = MikeInvestment / (MaryInvestment + MikeInvestment) * (2 * P / 3) + MaryExtra

-- Statement
theorem profit_value : P = 4000 :=
by
  sorry

end NUMINAMATH_GPT_profit_value_l471_47185


namespace NUMINAMATH_GPT_triple_complement_angle_l471_47166

theorem triple_complement_angle (x : ℝ) (h1 : x = 3 * (90 - x)) : x = 67.5 :=
sorry

end NUMINAMATH_GPT_triple_complement_angle_l471_47166


namespace NUMINAMATH_GPT_total_amount_245_l471_47180

-- Define the conditions and the problem
theorem total_amount_245 (a : ℝ) (x y z : ℝ) (h1 : y = 0.45 * a) (h2 : z = 0.30 * a) (h3 : y = 63) :
  x + y + z = 245 := 
by
  -- Starting the proof (proof steps are unnecessary as per the procedure)
  sorry

end NUMINAMATH_GPT_total_amount_245_l471_47180


namespace NUMINAMATH_GPT_combined_class_average_score_l471_47176

theorem combined_class_average_score
  (avg_A : ℕ := 65) (avg_B : ℕ := 90) (avg_C : ℕ := 77)
  (ratio_A : ℕ := 4) (ratio_B : ℕ := 6) (ratio_C : ℕ := 5) :
  ((avg_A * ratio_A + avg_B * ratio_B + avg_C * ratio_C) / (ratio_A + ratio_B + ratio_C) = 79) :=
by 
  sorry

end NUMINAMATH_GPT_combined_class_average_score_l471_47176


namespace NUMINAMATH_GPT_connie_grandma_birth_year_l471_47115

theorem connie_grandma_birth_year :
  ∀ (B S G : ℕ),
  B = 1932 →
  S = 1936 →
  (S - B) * 2 = (S - G) →
  G = 1928 := 
by
  intros B S G hB hS hGap
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_connie_grandma_birth_year_l471_47115


namespace NUMINAMATH_GPT_germs_left_after_sprays_l471_47163

-- Define the percentages as real numbers
def S1 : ℝ := 0.50 -- 50%
def S2 : ℝ := 0.35 -- 35%
def S3 : ℝ := 0.20 -- 20%
def S4 : ℝ := 0.10 -- 10%

-- Define the overlaps as real numbers
def overlap12 : ℝ := 0.10 -- between S1 and S2
def overlap23 : ℝ := 0.07 -- between S2 and S3
def overlap34 : ℝ := 0.05 -- between S3 and S4
def overlap13 : ℝ := 0.03 -- between S1 and S3
def overlap14 : ℝ := 0.02 -- between S1 and S4

theorem germs_left_after_sprays :
  let total_killed := S1 + S2 + S3 + S4
  let total_overlap := overlap12 + overlap23 + overlap34 + overlap13 + overlap14
  let adjusted_overlap := overlap12 + overlap23 + overlap34
  let effective_killed := total_killed - adjusted_overlap
  let percentage_left := 1.0 - effective_killed
  percentage_left = 0.07 := by
  -- proof steps to be inserted here
  sorry

end NUMINAMATH_GPT_germs_left_after_sprays_l471_47163


namespace NUMINAMATH_GPT_at_least_one_zero_l471_47194

theorem at_least_one_zero (a b : ℝ) : (¬ (a ≠ 0 ∧ b ≠ 0)) → (a = 0 ∨ b = 0) := by
  intro h
  have h' : ¬ ((a ≠ 0) ∧ (b ≠ 0)) := h
  sorry

end NUMINAMATH_GPT_at_least_one_zero_l471_47194


namespace NUMINAMATH_GPT_parabola_intersection_l471_47117

theorem parabola_intersection :
  (∀ x y : ℝ, y = 3 * x^2 - 4 * x + 2 ↔ y = 9 * x^2 + 6 * x + 2) →
  (∃ x1 y1 x2 y2 : ℝ,
    (x1 = 0 ∧ y1 = 2) ∧ (x2 = -5 / 3 ∧ y2 = 17)) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_parabola_intersection_l471_47117


namespace NUMINAMATH_GPT_cost_per_pumpkin_pie_l471_47150

theorem cost_per_pumpkin_pie
  (pumpkin_pies : ℕ)
  (cherry_pies : ℕ)
  (cost_cherry_pie : ℕ)
  (total_profit : ℕ)
  (selling_price : ℕ)
  (total_revenue : ℕ)
  (total_cost : ℕ)
  (cost_pumpkin_pie : ℕ)
  (H1 : pumpkin_pies = 10)
  (H2 : cherry_pies = 12)
  (H3 : cost_cherry_pie = 5)
  (H4 : total_profit = 20)
  (H5 : selling_price = 5)
  (H6 : total_revenue = (pumpkin_pies + cherry_pies) * selling_price)
  (H7 : total_cost = total_revenue - total_profit)
  (H8 : total_cost = pumpkin_pies * cost_pumpkin_pie + cherry_pies * cost_cherry_pie) :
  cost_pumpkin_pie = 3 :=
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_cost_per_pumpkin_pie_l471_47150


namespace NUMINAMATH_GPT_quadratic_roots_expression_l471_47106

theorem quadratic_roots_expression :
  ∀ (x1 x2 : ℝ), (x1^2 - 2 * x1 - 1 = 0) ∧ (x2^2 - 2 * x2 - 1 = 0) →
  (x1 + x2 - x1 * x2 = 3) :=
by
  intros x1 x2 h
  sorry

end NUMINAMATH_GPT_quadratic_roots_expression_l471_47106


namespace NUMINAMATH_GPT_symmetric_point_origin_l471_47143

-- Define the coordinates of point A and the relation of symmetry about the origin
def A : ℝ × ℝ := (2, -1)
def symm_origin (P : ℝ × ℝ) : ℝ × ℝ := (-P.1, -P.2)

-- Theorem statement: Point B is the symmetric point of A about the origin
theorem symmetric_point_origin : symm_origin A = (-2, 1) :=
  sorry

end NUMINAMATH_GPT_symmetric_point_origin_l471_47143


namespace NUMINAMATH_GPT_fraction_of_phone_numbers_begin_with_8_and_end_with_5_l471_47120

theorem fraction_of_phone_numbers_begin_with_8_and_end_with_5 :
  let total_numbers := 7 * 10^7
  let specific_numbers := 10^6
  specific_numbers / total_numbers = 1 / 70 := by
  sorry

end NUMINAMATH_GPT_fraction_of_phone_numbers_begin_with_8_and_end_with_5_l471_47120


namespace NUMINAMATH_GPT_perimeter_change_l471_47126

theorem perimeter_change (s h : ℝ) 
  (h1 : 2 * (1.3 * s + 0.8 * h) = 2 * (s + h)) :
  (2 * (0.8 * s + 1.3 * h) = 1.1 * (2 * (s + h))) :=
by
  sorry

end NUMINAMATH_GPT_perimeter_change_l471_47126


namespace NUMINAMATH_GPT_population_net_increase_l471_47193

-- Define conditions
def birth_rate : ℚ := 5 / 2    -- 5 people every 2 seconds
def death_rate : ℚ := 3 / 2    -- 3 people every 2 seconds
def one_day_in_seconds : ℕ := 86400   -- Number of seconds in one day

-- Define the net increase per second
def net_increase_per_second := birth_rate - death_rate

-- Prove that the net increase in one day is 86400 people given the conditions
theorem population_net_increase :
  net_increase_per_second * one_day_in_seconds = 86400 :=
sorry

end NUMINAMATH_GPT_population_net_increase_l471_47193


namespace NUMINAMATH_GPT_S_40_eq_150_l471_47173

variable {R : Type*} [Field R]

-- Define the sum function for geometric sequences.
noncomputable def geom_sum (a q : R) (n : ℕ) : R :=
  a * (1 - q^n) / (1 - q)

-- Given conditions from the problem.
axiom S_10_eq : ∀ {a q : R}, geom_sum a q 10 = 10
axiom S_30_eq : ∀ {a q : R}, geom_sum a q 30 = 70

-- The main theorem stating S40 = 150 under the given conditions.
theorem S_40_eq_150 {a q : R} (h10 : geom_sum a q 10 = 10) (h30 : geom_sum a q 30 = 70) :
  geom_sum a q 40 = 150 :=
sorry

end NUMINAMATH_GPT_S_40_eq_150_l471_47173


namespace NUMINAMATH_GPT_area_comparison_l471_47169

-- Define the side lengths of the triangles
def a₁ := 17
def b₁ := 17
def c₁ := 12

def a₂ := 17
def b₂ := 17
def c₂ := 16

-- Define the semiperimeters
def s₁ := (a₁ + b₁ + c₁) / 2
def s₂ := (a₂ + b₂ + c₂) / 2

-- Define the areas using Heron's formula
noncomputable def area₁ := (s₁ * (s₁ - a₁) * (s₁ - b₁) * (s₁ - c₁)).sqrt
noncomputable def area₂ := (s₂ * (s₂ - a₂) * (s₂ - b₂) * (s₂ - c₂)).sqrt

-- The theorem to prove
theorem area_comparison : area₁ < area₂ := sorry

end NUMINAMATH_GPT_area_comparison_l471_47169


namespace NUMINAMATH_GPT_david_distance_to_airport_l471_47154

theorem david_distance_to_airport (t : ℝ) (d : ℝ) :
  (35 * (t + 1) = d) ∧ (d - 35 = 50 * (t - 1.5)) → d = 210 :=
by
  sorry

end NUMINAMATH_GPT_david_distance_to_airport_l471_47154


namespace NUMINAMATH_GPT_peaches_total_l471_47191

theorem peaches_total (n P : ℕ) (h1 : P - 6 * n = 57) (h2 : P = 9 * (n - 6) + 3) : P = 273 :=
by
  sorry

end NUMINAMATH_GPT_peaches_total_l471_47191


namespace NUMINAMATH_GPT_vector_magnitude_l471_47179

noncomputable def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem vector_magnitude : 
  let AB := (-1, 2)
  let BC := (x, -5)
  let AC := (AB.1 + BC.1, AB.2 + BC.2)
  dot_product AB BC = -7 → magnitude AC = 5 :=
by sorry

end NUMINAMATH_GPT_vector_magnitude_l471_47179


namespace NUMINAMATH_GPT_find_specific_n_l471_47144

theorem find_specific_n :
  ∀ (n : ℕ), (∃ (a b : ℤ), n^2 = a + b ∧ n^3 = a^2 + b^2) ↔ n = 0 ∨ n = 1 ∨ n = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_specific_n_l471_47144


namespace NUMINAMATH_GPT_cos_120_eq_neg_one_half_l471_47122

theorem cos_120_eq_neg_one_half : Real.cos (120 * Real.pi / 180) = -1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_cos_120_eq_neg_one_half_l471_47122


namespace NUMINAMATH_GPT_NewYearSeasonMarkup_is_25percent_l471_47155

variable (C N : ℝ)
variable (h1 : N >= 0)
variable (h2 : 0.92 * (1 + N) * 1.20 * C = 1.38 * C)

theorem NewYearSeasonMarkup_is_25percent : N = 0.25 :=
  by
  sorry

end NUMINAMATH_GPT_NewYearSeasonMarkup_is_25percent_l471_47155


namespace NUMINAMATH_GPT_complement_union_A_B_l471_47199

def is_element_of_set_A (x : ℤ) : Prop := ∃ k : ℤ, x = 3 * k + 1
def is_element_of_set_B (x : ℤ) : Prop := ∃ k : ℤ, x = 3 * k + 2
def is_element_of_complement_U (x : ℤ) : Prop := ∃ k : ℤ, x = 3 * k

theorem complement_union_A_B :
  {x : ℤ | ¬ (is_element_of_set_A x ∨ is_element_of_set_B x)} = {x : ℤ | is_element_of_complement_U x} :=
by
  sorry

end NUMINAMATH_GPT_complement_union_A_B_l471_47199


namespace NUMINAMATH_GPT_find_parallel_lines_l471_47139

open Real

-- Definitions for the problem conditions
def line1 (a x y : ℝ) : Prop := x + 2 * a * y - 1 = 0
def line2 (a x y : ℝ) : Prop := (2 * a - 1) * x - a * y - 1 = 0

-- Definition of when two lines are parallel in ℝ²
def parallel (l1 l2 : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, (l1 x y → ∃ k, ∀ x' y', l2 x' y' → x = k * x' ∧ y = k * y')

-- Main theorem statement
theorem find_parallel_lines:
  ∀ a : ℝ, (parallel (line1 a) (line2 a)) → (a = 0 ∨ a = 1 / 4) :=
by sorry

end NUMINAMATH_GPT_find_parallel_lines_l471_47139


namespace NUMINAMATH_GPT_algebraic_notation_correct_l471_47124

def exprA : String := "a * 5"
def exprB : String := "a7"
def exprC : String := "3 1/2 x"
def exprD : String := "-7/8 x"

theorem algebraic_notation_correct :
  exprA ≠ "correct" ∧
  exprB ≠ "correct" ∧
  exprC ≠ "correct" ∧
  exprD = "correct" :=
by
  sorry

end NUMINAMATH_GPT_algebraic_notation_correct_l471_47124


namespace NUMINAMATH_GPT_number_is_2250_l471_47188

-- Question: Prove that x = 2250 given the condition.
theorem number_is_2250 (x : ℕ) (h : x / 5 = 75 + x / 6) : x = 2250 := 
sorry

end NUMINAMATH_GPT_number_is_2250_l471_47188


namespace NUMINAMATH_GPT_evaluate_expression_l471_47127

noncomputable def given_expression : ℝ :=
  |8 - 8 * (3 - 12)^2| - |5 - Real.sin 11| + |2^(4 - 2 * 3) / ((3^2) - 7)|

theorem evaluate_expression : given_expression = 634.125009794 := 
  sorry

end NUMINAMATH_GPT_evaluate_expression_l471_47127


namespace NUMINAMATH_GPT_vertical_asymptote_l471_47196

theorem vertical_asymptote (x : ℝ) : 4 * x - 9 = 0 → x = 9 / 4 := by
  sorry

end NUMINAMATH_GPT_vertical_asymptote_l471_47196


namespace NUMINAMATH_GPT_three_digit_numbers_condition_l471_47146

theorem three_digit_numbers_condition (a b c : ℕ) (h1 : 0 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9) (h3 : 0 ≤ c ∧ c ≤ 9) :
  (100 * a + 10 * b + c = 2 * ((10 * a + b) + (10 * a + c) + (10 * b + a) + (10 * b + c) + (10 * c + a) + (10 * c + b)))
  ↔ (100 * a + 10 * b + c = 132 ∨ 100 * a + 10 * b + c = 264 ∨ 100 * a + 10 * b + c = 396) :=
by
  sorry

end NUMINAMATH_GPT_three_digit_numbers_condition_l471_47146


namespace NUMINAMATH_GPT_sin_cos_sixth_l471_47160

theorem sin_cos_sixth (θ : ℝ) (h : Real.sin (2 * θ) = 1 / 3) :
  Real.sin θ ^ 6 + Real.cos θ ^ 6 = 11 / 12 :=
sorry

end NUMINAMATH_GPT_sin_cos_sixth_l471_47160


namespace NUMINAMATH_GPT_garden_roller_diameter_l471_47190

theorem garden_roller_diameter 
  (length : ℝ) 
  (total_area : ℝ) 
  (num_revolutions : ℕ) 
  (pi : ℝ) 
  (A : length = 2)
  (B : total_area = 37.714285714285715)
  (C : num_revolutions = 5)
  (D : pi = 22 / 7) : 
  ∃ d : ℝ, d = 1.2 :=
by
  sorry

end NUMINAMATH_GPT_garden_roller_diameter_l471_47190


namespace NUMINAMATH_GPT_g_minus_1001_l471_47123

def g (x : ℝ) : ℝ := sorry

theorem g_minus_1001 :
  (∀ x y : ℝ, g (x * y) + 2 * x = x * g y + g x) →
  g 1 = 3 →
  g (-1001) = 1005 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_g_minus_1001_l471_47123


namespace NUMINAMATH_GPT_marathons_yards_l471_47137

theorem marathons_yards
  (miles_per_marathon : ℕ)
  (yards_per_marathon : ℕ)
  (miles_in_yard : ℕ)
  (marathons_run : ℕ)
  (total_miles : ℕ)
  (total_yards : ℕ)
  (y : ℕ) :
  miles_per_marathon = 30
  → yards_per_marathon = 520
  → miles_in_yard = 1760
  → marathons_run = 8
  → total_miles = (miles_per_marathon * marathons_run) + (yards_per_marathon * marathons_run) / miles_in_yard
  → total_yards = (yards_per_marathon * marathons_run) % miles_in_yard
  → y = 640 := 
by
  intros
  sorry

end NUMINAMATH_GPT_marathons_yards_l471_47137


namespace NUMINAMATH_GPT_least_M_bench_sections_l471_47197

/--
A single bench section at a community event can hold either 8 adults, 12 children, or 10 teenagers. 
We are to find the smallest positive integer M such that when M bench sections are connected end to end,
an equal number of adults, children, and teenagers seated together will occupy all the bench space.
-/
theorem least_M_bench_sections
  (M : ℕ)
  (hM_pos : M > 0)
  (adults_capacity : ℕ := 8 * M)
  (children_capacity : ℕ := 12 * M)
  (teenagers_capacity : ℕ := 10 * M)
  (h_equal_capacity : adults_capacity = children_capacity ∧ children_capacity = teenagers_capacity) :
  M = 15 := 
sorry

end NUMINAMATH_GPT_least_M_bench_sections_l471_47197


namespace NUMINAMATH_GPT_stamp_book_gcd_l471_47105

theorem stamp_book_gcd (total1 total2 total3 : ℕ) 
    (h1 : total1 = 945) (h2 : total2 = 1260) (h3 : total3 = 630) : 
    ∃ d, d = Nat.gcd (Nat.gcd total1 total2) total3 ∧ d = 315 := 
by
  sorry

end NUMINAMATH_GPT_stamp_book_gcd_l471_47105


namespace NUMINAMATH_GPT_parabola_vertex_b_l471_47114

theorem parabola_vertex_b (a b c p : ℝ) (h₁ : p ≠ 0)
  (h₂ : ∀ x, (x = p → -p = a * (p^2) + b * p + c) ∧ (x = 0 → p = c)) :
  b = - (4 / p) :=
sorry

end NUMINAMATH_GPT_parabola_vertex_b_l471_47114


namespace NUMINAMATH_GPT_fraction_is_seventh_l471_47157

-- Definition of the condition on x being greater by a certain percentage
def x_greater := 1125.0000000000002 / 100

-- Definition of x in terms of the condition
def x := (4 / 7) * (1 + x_greater)

-- Definition of the fraction f
def f := 1 / x

-- Lean theorem statement to prove the fraction is 1/7
theorem fraction_is_seventh (x_greater: ℝ) : (1 / ((4 / 7) * (1 + x_greater))) = 1 / 7 :=
by
  sorry

end NUMINAMATH_GPT_fraction_is_seventh_l471_47157


namespace NUMINAMATH_GPT_sum_primes_between_20_and_40_l471_47128

theorem sum_primes_between_20_and_40 :
  23 + 29 + 31 + 37 = 120 :=
by
  sorry

end NUMINAMATH_GPT_sum_primes_between_20_and_40_l471_47128


namespace NUMINAMATH_GPT_range_of_a_l471_47107

-- Definition for set A
def A : Set (ℝ × ℝ) := { p | ∃ x y, p = (x, y) ∧ y = -|x| - 2 }

-- Definition for set B
def B (a : ℝ) : Set (ℝ × ℝ) := { p | ∃ x y, p = (x, y) ∧ (x - a)^2 + y^2 = a^2 }

-- Statement of the problem in Lean
theorem range_of_a (a : ℝ) : (∀ p, p ∈ A → p ∉ B a) → -2 * Real.sqrt 2 - 2 < a ∧ a < 2 * Real.sqrt 2 + 2 := by
  sorry

end NUMINAMATH_GPT_range_of_a_l471_47107


namespace NUMINAMATH_GPT_cypress_tree_price_l471_47159

def amount_per_cypress_tree (C : ℕ) : Prop :=
  let cabin_price := 129000
  let cash := 150
  let cypress_count := 20
  let pine_count := 600
  let maple_count := 24
  let pine_price := 200
  let maple_price := 300
  let leftover_cash := 350
  let total_amount_raised := cabin_price - cash + leftover_cash
  let total_pine_maple := (pine_count * pine_price) + (maple_count * maple_price)
  let total_cypress := total_amount_raised - total_pine_maple
  let cypress_sale_price := total_cypress / cypress_count
  cypress_sale_price = C

theorem cypress_tree_price : amount_per_cypress_tree 100 :=
by {
  -- Proof skipped
  sorry
}

end NUMINAMATH_GPT_cypress_tree_price_l471_47159

import Mathlib

namespace fraction_n_m_l1162_116294

noncomputable def a (k : ℝ) := 2*k + 1
noncomputable def b (k : ℝ) := 3*k + 2
noncomputable def c (k : ℝ) := 3 - 4*k
noncomputable def S (k : ℝ) := a k + 2*(b k) + 3*(c k)

theorem fraction_n_m : 
  (∀ (k : ℝ), -1/2 ≤ k ∧ k ≤ 3/4 → (S (3/4) = 11 ∧ S (-1/2) = 16)) → 
  11/16 = 11 / 16 :=
by
  sorry

end fraction_n_m_l1162_116294


namespace find_values_of_ABC_l1162_116208

-- Define the given conditions
def condition1 (A B C : ℕ) : Prop := A + B + C = 36
def condition2 (A B C : ℕ) : Prop := 
  (A + B) * 3 * 4 = (B + C) * 2 * 4 ∧ 
  (B + C) * 2 * 4 = (A + C) * 2 * 3

-- State the problem
theorem find_values_of_ABC (A B C : ℕ) 
  (h1 : condition1 A B C) 
  (h2 : condition2 A B C) : 
  A = 12 ∧ B = 4 ∧ C = 20 :=
sorry

end find_values_of_ABC_l1162_116208


namespace cumulative_percentage_decrease_l1162_116221

theorem cumulative_percentage_decrease :
  let original_price := 100
  let first_reduction := original_price * 0.85
  let second_reduction := first_reduction * 0.90
  let third_reduction := second_reduction * 0.95
  let fourth_reduction := third_reduction * 0.80
  let final_price := fourth_reduction
  (original_price - final_price) / original_price * 100 = 41.86 := by
  sorry

end cumulative_percentage_decrease_l1162_116221


namespace cannot_fold_patternD_to_cube_l1162_116296

def patternA : Prop :=
  -- 5 squares arranged in a cross shape
  let squares := 5
  let shape  := "cross"
  squares = 5 ∧ shape = "cross"

def patternB : Prop :=
  -- 4 squares in a straight line
  let squares := 4
  let shape  := "line"
  squares = 4 ∧ shape = "line"

def patternC : Prop :=
  -- 3 squares in an L shape, and 2 squares attached to one end of the L making a T shape
  let squares := 5
  let shape  := "T"
  squares = 5 ∧ shape = "T"

def patternD : Prop :=
  -- 6 squares in a "+" shape with one extra square
  let squares := 7
  let shape  := "plus"
  squares = 7 ∧ shape = "plus"

theorem cannot_fold_patternD_to_cube :
  patternD → ¬ (patternA ∨ patternB ∨ patternC) :=
by
  sorry

end cannot_fold_patternD_to_cube_l1162_116296


namespace no_solution_to_inequalities_l1162_116213

theorem no_solution_to_inequalities :
  ∀ (x y z t : ℝ), 
    ¬ (|x| > |y - z + t| ∧
       |y| > |x - z + t| ∧
       |z| > |x - y + t| ∧
       |t| > |x - y + z|) :=
by
  intro x y z t
  sorry

end no_solution_to_inequalities_l1162_116213


namespace boxes_needed_l1162_116257

-- Define the given conditions

def red_pencils : ℕ := 20
def blue_pencils : ℕ := 2 * red_pencils
def yellow_pencils : ℕ := 40
def green_pencils : ℕ := red_pencils + blue_pencils
def total_pencils : ℕ := red_pencils + blue_pencils + green_pencils + yellow_pencils
def pencils_per_box : ℕ := 20

-- Lean theorem statement to prove the number of boxes needed is 8

theorem boxes_needed : total_pencils / pencils_per_box = 8 :=
by
  -- This is where the proof would go
  sorry

end boxes_needed_l1162_116257


namespace total_stones_l1162_116215

theorem total_stones (x : ℕ) 
  (h1 : x + 6 * x = x * 7 ∧ 7 * x + 6 * x = 2 * x) 
  (h2 : 2 * x = 7 * x - 10) 
  (h3 : 14 * x / 2 = 7 * x) :
  2 * 2 + 14 * 2 + 2 + 7 * 2 + 6 * 2 = 60 := 
by {
  sorry
}

end total_stones_l1162_116215


namespace volume_in_barrel_l1162_116229

def is_divisible (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem volume_in_barrel (x : ℕ) (V : ℕ) (hx : V = 30) 
  (h1 : V = x / 2 + x / 3 + x / 4 + x / 5 + x / 6) 
  (h2 : is_divisible (87 * x) 60) : 
  V = 29 := 
sorry

end volume_in_barrel_l1162_116229


namespace linear_equations_not_always_solvable_l1162_116272

theorem linear_equations_not_always_solvable 
  (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : 
  ¬(∀ x y : ℝ, (a₁ * x + b₁ * y = c₁ ∧ a₂ * x + b₂ * y = c₂) ↔ 
                   a₁ * b₂ - a₂ * b₁ ≠ 0) :=
sorry

end linear_equations_not_always_solvable_l1162_116272


namespace sequence_term_formula_l1162_116227

def sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = 1/2 - 1/2 * a n

def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n ≥ 2, a n = r * a (n - 1)

theorem sequence_term_formula (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n ≥ 1, S n = 1/2 - 1/2 * a n) →
  (S 1 = 1/2 - 1/2 * a 1) →
  a 1 = 1/3 →
  (∀ n ≥ 2, S n = 1/2 - 1/2 * (a n) → S (n - 1) = 1/2 - 1/2 * (a (n - 1)) → a n = 1/3 * a (n-1)) →
  ∀ n, a n = (1/3)^n :=
by
  intro h1 h2 h3 h4
  sorry

end sequence_term_formula_l1162_116227


namespace hair_growth_l1162_116236

theorem hair_growth (initial final : ℝ) (h_init : initial = 18) (h_final : final = 24) : final - initial = 6 :=
by
  sorry

end hair_growth_l1162_116236


namespace total_hours_charged_l1162_116269

variable (K P M : ℕ)

theorem total_hours_charged (h1 : P = 2 * K) (h2 : P = M / 3) (h3 : M = K + 80) : K + P + M = 144 := 
by
  sorry

end total_hours_charged_l1162_116269


namespace vacation_cost_division_l1162_116210

theorem vacation_cost_division (total_cost : ℕ) (cost_per_person3 different_cost : ℤ) (n : ℕ)
  (h1 : total_cost = 375)
  (h2 : cost_per_person3 = total_cost / 3)
  (h3 : different_cost = cost_per_person3 - 50)
  (h4 : different_cost = total_cost / n) :
  n = 5 :=
  sorry

end vacation_cost_division_l1162_116210


namespace rain_third_day_l1162_116250

theorem rain_third_day (rain_day1 rain_day2 rain_day3 : ℕ)
  (h1 : rain_day1 = 4)
  (h2 : rain_day2 = 5 * rain_day1)
  (h3 : rain_day3 = (rain_day1 + rain_day2) - 6) : 
  rain_day3 = 18 := 
by
  -- Proof omitted
  sorry

end rain_third_day_l1162_116250


namespace sample_size_second_grade_l1162_116263

theorem sample_size_second_grade
    (total_students : ℕ)
    (ratio_first : ℕ)
    (ratio_second : ℕ)
    (ratio_third : ℕ)
    (sample_size : ℕ) :
    total_students = 2000 →
    ratio_first = 5 → ratio_second = 3 → ratio_third = 2 →
    sample_size = 20 →
    (20 * (3 / (5 + 3 + 2)) = 6) :=
by
  intros ht hr1 hr2 hr3 hs
  -- The proof would continue from here, but we're finished as the task only requires the statement.
  sorry

end sample_size_second_grade_l1162_116263


namespace find_x_l1162_116292

def magic_constant (a b c d e f g h i : ℤ) : Prop :=
  a + b + c = d + e + f ∧ d + e + f = g + h + i ∧
  a + d + g = b + e + h ∧ b + e + h = c + f + i ∧
  a + e + i = c + e + g

def given_magic_square (x : ℤ) : Prop :=
  magic_constant (4017) (2012) (0) 
                 (4015) (x - 2003) (11) 
                 (2014) (9) (x)

theorem find_x (x : ℤ) (h : given_magic_square x) : x = 4003 :=
by {
  sorry
}

end find_x_l1162_116292


namespace seashells_initial_count_l1162_116226

theorem seashells_initial_count (S : ℕ)
  (h1 : S - 70 = 2 * 55) : S = 180 :=
by
  sorry

end seashells_initial_count_l1162_116226


namespace gym_monthly_income_l1162_116262

-- Define the conditions
def twice_monthly_charge : ℕ := 18
def monthly_charge_per_member : ℕ := 2 * twice_monthly_charge
def number_of_members : ℕ := 300

-- State the goal: the monthly income of the gym
def monthly_income : ℕ := 36 * 300

-- The theorem to prove
theorem gym_monthly_income : monthly_charge_per_member * number_of_members = 10800 :=
by
  sorry

end gym_monthly_income_l1162_116262


namespace water_hyacinth_indicates_connection_l1162_116253

-- Definitions based on the conditions
def universally_interconnected : Prop := 
  ∀ (a b : Type), a ≠ b → ∃ (c : Type), (a ≠ c) ∧ (b ≠ c)

def connections_diverse : Prop := 
  ∀ (a b : Type), a ≠ b → ∃ (f : a → b), ∀ (x y : a), x ≠ y → f x ≠ f y

def connections_created : Prop :=
  ∃ (a b : Type), a ≠ b ∧ (∀ (f : a → b), False)

def connections_humanized : Prop :=
  ∀ (a b : Type), a ≠ b → (∃ c : Type, a = c) ∧ (∃ d : Type, b = d)

-- Problem statement
theorem water_hyacinth_indicates_connection : 
  universally_interconnected ∧ connections_diverse :=
by
  sorry

end water_hyacinth_indicates_connection_l1162_116253


namespace find_carbon_atoms_l1162_116284

variable (n : ℕ)
variable (molecular_weight : ℝ := 124.0)
variable (weight_Cu : ℝ := 63.55)
variable (weight_C : ℝ := 12.01)
variable (weight_O : ℝ := 16.00)
variable (num_Cu : ℕ := 1)
variable (num_O : ℕ := 3)

theorem find_carbon_atoms 
  (h : molecular_weight = (num_Cu * weight_Cu) + (n * weight_C) + (num_O * weight_O)) : 
  n = 1 :=
sorry

end find_carbon_atoms_l1162_116284


namespace e_n_max_value_l1162_116246

def b (n : ℕ) : ℕ := (5^n - 1) / 4

def e (n : ℕ) : ℕ := Nat.gcd (b n) (b (n + 1))

theorem e_n_max_value (n : ℕ) : e n = 1 := 
by sorry

end e_n_max_value_l1162_116246


namespace sequence_condition_satisfies_l1162_116205

def seq_prove_abs_lt_1 (a : ℕ → ℝ) : Prop :=
  (∃ i : ℕ, |a i| < 1)

theorem sequence_condition_satisfies (a : ℕ → ℝ)
  (h1 : a 1 * a 2 < 0)
  (h2 : ∀ n > 2, ∃ i j, 1 ≤ i ∧ i < j ∧ j < n ∧ (∀ k l, 1 ≤ k ∧ k < l ∧ l < n → |a i + a j| ≤ |a k + a l|)) :
  seq_prove_abs_lt_1 a :=
by
  sorry

end sequence_condition_satisfies_l1162_116205


namespace find_u_plus_v_l1162_116218

theorem find_u_plus_v (u v : ℤ) (huv : 0 < v ∧ v < u) (h_area : u * u + 3 * u * v = 451) : u + v = 21 := 
sorry

end find_u_plus_v_l1162_116218


namespace value_of_f_at_1_l1162_116291

def f (x : ℝ) : ℝ := x^2 + |x - 2|

theorem value_of_f_at_1 : f 1 = 2 :=
by sorry

end value_of_f_at_1_l1162_116291


namespace trader_excess_donations_l1162_116248

-- Define the conditions
def profit : ℤ := 1200
def allocation_percentage : ℤ := 60
def family_donation : ℤ := 250
def friends_donation : ℤ := (20 * family_donation) / 100 + family_donation
def total_family_friends_donation : ℤ := family_donation + friends_donation
def local_association_donation : ℤ := 15 * total_family_friends_donation / 10
def total_donations : ℤ := family_donation + friends_donation + local_association_donation
def allocated_amount : ℤ := allocation_percentage * profit / 100

-- Theorem statement (Question)
theorem trader_excess_donations : total_donations - allocated_amount = 655 :=
by
  sorry

end trader_excess_donations_l1162_116248


namespace positive_integer_solutions_count_l1162_116212

theorem positive_integer_solutions_count :
  (∃ (x y z : ℕ), 1 ≤ x ∧ x ≤ y ∧ y ≤ z ∧ x + y + z = 2010) → (336847 = 336847) :=
by {
  sorry
}

end positive_integer_solutions_count_l1162_116212


namespace min_value_of_quadratic_l1162_116238

def quadratic_function (x : ℝ) : ℝ :=
  x^2 - 12 * x + 35

theorem min_value_of_quadratic :
  ∀ x : ℝ, quadratic_function x ≥ quadratic_function 6 :=
by sorry

end min_value_of_quadratic_l1162_116238


namespace no_solution_for_equation_l1162_116299

theorem no_solution_for_equation (x : ℝ) (hx : x ≠ -1) :
  (5 * x + 2) / (x^2 + x) ≠ 3 / (x + 1) := 
sorry

end no_solution_for_equation_l1162_116299


namespace no_power_of_q_l1162_116207

theorem no_power_of_q (n : ℕ) (hn : n > 0) (q : ℕ) (hq : Prime q) : ¬ (∃ k : ℕ, n^q + ((n-1)/2)^2 = q^k) := 
by
  sorry  -- proof steps are not required as per instructions

end no_power_of_q_l1162_116207


namespace partial_fraction_product_is_correct_l1162_116200

-- Given conditions
def fraction_decomposition (x A B C : ℝ) :=
  ( (x^2 + 5 * x - 14) / (x^3 - 3 * x^2 - x + 3) = A / (x - 1) + B / (x - 3) + C / (x + 1) )

-- Statement we want to prove
theorem partial_fraction_product_is_correct (A B C : ℝ) (h : ∀ x : ℝ, fraction_decomposition x A B C) :
  A * B * C = -25 / 2 :=
sorry

end partial_fraction_product_is_correct_l1162_116200


namespace value_of_3a_minus_b_l1162_116216
noncomputable def solveEquation : Type := sorry

theorem value_of_3a_minus_b (a b : ℝ) (h1 : a = 3 + Real.sqrt 15) (h2 : b = 3 - Real.sqrt 15) (h3 : a ≥ b) :
  3 * a - b = 6 + 4 * Real.sqrt 15 :=
sorry

end value_of_3a_minus_b_l1162_116216


namespace Hari_contribution_l1162_116280

theorem Hari_contribution (P T_P T_H : ℕ) (r1 r2 : ℕ) (H : ℕ) :
  P = 3500 → 
  T_P = 12 → 
  T_H = 7 → 
  r1 = 2 → 
  r2 = 3 →
  (P * T_P) * r2 = (H * T_H) * r1 →
  H = 9000 :=
by
  sorry

end Hari_contribution_l1162_116280


namespace find_selling_price_functional_relationship_and_max_find_value_of_a_l1162_116214

section StoreProduct

variable (x : ℕ) (y : ℕ) (a k b : ℝ)

-- Definitions for the given conditions
def cost_price : ℝ := 50
def selling_price := x 
def sales_quantity := y 
def future_cost_increase := a

-- Given points
def point1 : ℝ × ℕ := (55, 90) 
def point2 : ℝ × ℕ := (65, 70)

-- Linear relationship between selling price and sales quantity
def linearfunc := y = k * x + b

-- Proof of the first statement
theorem find_selling_price (k := -2) (b := 200) : 
    (profit = 800 → (x = 60 ∨ x = 90)) :=
by
  -- People prove the theorem here
  sorry

-- Proof for the functional relationship between W and x
theorem functional_relationship_and_max (x := 75) : 
    W = -2*x^2 + 300*x - 10000 ∧ W_max = 1250 :=
by
  -- People prove the theorem here
  sorry

-- Proof for the value of a when the cost price increases
theorem find_value_of_a (cost_increase := 4) : 
    (W'_max = 960 → a = 4) :=
by
  -- People prove the theorem here
  sorry

end StoreProduct

end find_selling_price_functional_relationship_and_max_find_value_of_a_l1162_116214


namespace cos_5theta_l1162_116261

theorem cos_5theta (θ : ℝ) (h : Real.cos θ = 3/5) : Real.cos (5*θ) = -93/3125 :=
sorry

end cos_5theta_l1162_116261


namespace div_seven_and_sum_factors_l1162_116275

theorem div_seven_and_sum_factors (a b c : ℤ) (h : (a = 0 ∨ b = 0 ∨ c = 0) ∧ ¬(a = 0 ∧ b = 0 ∧ c = 0)) :
  ∃ k : ℤ, (a + b + c)^7 - a^7 - b^7 - c^7 = k * 7 * (a + b) * (b + c) * (c + a) :=
by
  sorry

end div_seven_and_sum_factors_l1162_116275


namespace find_r_and_k_l1162_116277

-- Define the line equation
def line (x : ℝ) : ℝ := 5 * x - 7

-- Define the parameterization
def param (t r k : ℝ) : ℝ × ℝ := 
  (r + 3 * t, 2 + k * t)

-- Theorem stating that (r, k) = (9/5, 15) satisfies the given conditions
theorem find_r_and_k 
  (r k : ℝ)
  (H1 : param 0 r k = (r, 2))
  (H2 : line r = 2)
  (H3 : param 1 r k = (r + 3, 2 + k))
  (H4 : line (r + 3) = 2 + k)
  : (r, k) = (9/5, 15) :=
sorry

end find_r_and_k_l1162_116277


namespace min_value_x_plus_4y_l1162_116289

theorem min_value_x_plus_4y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + y = 2 * x * y) : x + 4 * y = 9 / 2 :=
by
  sorry

end min_value_x_plus_4y_l1162_116289


namespace simplify_exponentiation_l1162_116298

theorem simplify_exponentiation (x : ℕ) :
  (x^5 * x^3)^2 = x^16 := 
by {
  sorry -- proof will go here
}

end simplify_exponentiation_l1162_116298


namespace ratio_value_l1162_116245

theorem ratio_value (x y : ℝ) (h : x / y = 2 / 3) : x / (x + y) = 2 / 5 := 
by
  sorry

end ratio_value_l1162_116245


namespace area_y_eq_x2_y_eq_x3_l1162_116285

noncomputable section

open Real

def area_closed_figure_between_curves : ℝ :=
  ∫ x in (0:ℝ)..(1:ℝ), (x^2 - x^3)

theorem area_y_eq_x2_y_eq_x3 :
  area_closed_figure_between_curves = 1 / 12 := by
  sorry

end area_y_eq_x2_y_eq_x3_l1162_116285


namespace digits_count_concatenated_l1162_116249

-- Define the conditions for the digit count of 2^n and 5^n
def digits_count_2n (n p : ℕ) : Prop := 10^(p-1) ≤ 2^n ∧ 2^n < 10^p
def digits_count_5n (n q : ℕ) : Prop := 10^(q-1) ≤ 5^n ∧ 5^n < 10^q

-- The main theorem to prove the number of digits when 2^n and 5^n are concatenated
theorem digits_count_concatenated (n p q : ℕ) 
  (h1 : digits_count_2n n p) 
  (h2 : digits_count_5n n q): 
  p + q = n + 1 := by 
  sorry

end digits_count_concatenated_l1162_116249


namespace equation_of_line_intersection_l1162_116268

theorem equation_of_line_intersection
  (h1 : ∀ x y : ℝ, x^2 + y^2 = 1)
  (h2 : ∀ x y : ℝ, x^2 + y^2 + 2*x - 4*y + 1 = 0) :
  ∀ x y : ℝ, x - 2*y + 1 = 0 :=
by
  sorry

end equation_of_line_intersection_l1162_116268


namespace fraction_exponent_product_l1162_116243

theorem fraction_exponent_product :
  ( (5/6: ℚ)^2 * (2/3: ℚ)^3 = 50/243 ) :=
by
  sorry

end fraction_exponent_product_l1162_116243


namespace possible_values_of_a_l1162_116266

variable (a : ℝ)
def A : Set ℝ := { x | x^2 ≠ 1 }
def B (a : ℝ) : Set ℝ := { x | a * x = 1 }

theorem possible_values_of_a (h : (A ∪ B a) = A) : a = 1 ∨ a = -1 ∨ a = 0 :=
by
  sorry

end possible_values_of_a_l1162_116266


namespace Tim_change_l1162_116265

theorem Tim_change (initial_amount paid_amount : ℕ) (h₀ : initial_amount = 50) (h₁ : paid_amount = 45) : initial_amount - paid_amount = 5 :=
by
  sorry

end Tim_change_l1162_116265


namespace right_triangle_max_area_l1162_116258

theorem right_triangle_max_area
  (a b : ℝ) (h_a_nonneg : 0 ≤ a) (h_b_nonneg : 0 ≤ b)
  (h_right_triangle : a^2 + b^2 = 20^2)
  (h_perimeter : a + b + 20 = 48) :
  (1 / 2) * a * b = 96 :=
by
  sorry

end right_triangle_max_area_l1162_116258


namespace determine_C_plus_D_l1162_116211

theorem determine_C_plus_D (A B C D : ℕ) 
  (hA : A ≠ 0) 
  (h1 : A < 10) (h2 : B < 10) (h3 : C < 10) (h4 : D < 10) 
  (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) :
  (A * 100 + B * 10 + C) * D = A * 1000 + B * 100 + C * 10 + D → 
  C + D = 5 :=
by
    sorry

end determine_C_plus_D_l1162_116211


namespace determine_c_div_d_l1162_116201

theorem determine_c_div_d (x y c d : ℝ) (h1 : 4 * x + 8 * y = c) (h2 : 5 * x - 10 * y = d) (h3 : d ≠ 0) (h4 : x ≠ 0) (h5 : y ≠ 0) : c / d = -4 / 5 :=
by
sorry

end determine_c_div_d_l1162_116201


namespace directrix_of_parabola_l1162_116264

theorem directrix_of_parabola : ∀ (x : ℝ), y = (x^2 - 8*x + 12) / 16 → ∃ (d : ℝ), d = -1/2 := 
sorry

end directrix_of_parabola_l1162_116264


namespace find_alpha_l1162_116232

theorem find_alpha (α : ℝ) (hα : 0 ≤ α ∧ α < 2 * Real.pi) 
  (l1 : ∀ x y : ℝ, x * Real.cos α - y - 1 = 0) 
  (l2 : ∀ x y : ℝ, x + y * Real.sin α + 1 = 0) :
  α = Real.pi / 4 ∨ α = 5 * Real.pi / 4 :=
sorry

end find_alpha_l1162_116232


namespace total_points_other_five_l1162_116244

theorem total_points_other_five
  (x : ℕ) -- total number of points scored by the team
  (d : ℕ) (e : ℕ) (f : ℕ) (y : ℕ) -- points scored by Daniel, Emma, Fiona, and others respectively
  (hd : d = x / 3) -- Daniel scored 1/3 of the team's points
  (he : e = 3 * x / 8) -- Emma scored 3/8 of the team's points
  (hf : f = 18) -- Fiona scored 18 points
  (h_other : ∀ i, 1 ≤ i ∧ i ≤ 5 → y ≤ 15 / 5) -- Other 5 members scored no more than 3 points each
  (h_total : d + e + f + y = x) -- Total points equation
  : y = 14 := sorry -- Final number of points scored by the other 5 members

end total_points_other_five_l1162_116244


namespace find_ABC_l1162_116209

theorem find_ABC :
    ∃ (A B C : ℚ), 
    (∀ x : ℚ, x ≠ 2 ∧ x ≠ 4 ∧ x ≠ 5 → 
        (x^2 - 9) / ((x - 2) * (x - 4) * (x - 5)) = A / (x - 2) + B / (x - 4) + C / (x - 5)) 
    ∧ A = 5 / 3 ∧ B = -7 / 2 ∧ C = 8 / 3 := 
sorry

end find_ABC_l1162_116209


namespace steven_amanda_hike_difference_l1162_116254

variable (Camila_hikes : ℕ)
variable (Camila_weeks : ℕ)
variable (hikes_per_week : ℕ)

def Amanda_hikes (Camila_hikes : ℕ) : ℕ := 8 * Camila_hikes

def Steven_hikes (Camila_hikes : ℕ)(Camila_weeks : ℕ)(hikes_per_week : ℕ) : ℕ :=
  Camila_hikes + Camila_weeks * hikes_per_week

theorem steven_amanda_hike_difference
  (hCamila : Camila_hikes = 7)
  (hWeeks : Camila_weeks = 16)
  (hHikesPerWeek : hikes_per_week = 4) :
  Steven_hikes Camila_hikes Camila_weeks hikes_per_week - Amanda_hikes Camila_hikes = 15 := by
  sorry

end steven_amanda_hike_difference_l1162_116254


namespace max_positive_n_l1162_116223

def a (n : ℕ) : ℤ := 19 - 2 * n

theorem max_positive_n (n : ℕ) (h : a n > 0) : n ≤ 9 :=
by
  sorry

end max_positive_n_l1162_116223


namespace train_speed_proof_l1162_116297

noncomputable def train_speed (L : ℕ) (t : ℝ) (v_m : ℝ) : ℝ :=
  let v_m_m_s := v_m * (1000 / 3600)
  let v_rel := L / t
  v_rel + v_m_m_s

theorem train_speed_proof
  (L : ℕ)
  (t : ℝ)
  (v_m : ℝ)
  (hL : L = 900)
  (ht : t = 53.99568034557235)
  (hv_m : v_m = 3)
  : train_speed L t v_m = 63.0036 :=
  by sorry

end train_speed_proof_l1162_116297


namespace tan_half_alpha_eq_one_third_l1162_116259

open Real

theorem tan_half_alpha_eq_one_third (α : ℝ) (h1 : 5 * sin (2 * α) = 6 * cos α) (h2 : 0 < α ∧ α < π / 2) :
  tan (α / 2) = 1 / 3 :=
by
  sorry

end tan_half_alpha_eq_one_third_l1162_116259


namespace unique_identity_function_l1162_116239

theorem unique_identity_function (f : ℕ+ → ℕ+) :
  (∀ (x y : ℕ+), 
    let a := x 
    let b := f y 
    let c := f (y + f x - 1)
    a + b > c ∧ a + c > b ∧ b + c > a) →
  (∀ x, f x = x) :=
by
  intro h
  sorry

end unique_identity_function_l1162_116239


namespace sum_of_roots_of_quadratic_eq_l1162_116255

theorem sum_of_roots_of_quadratic_eq :
  ∀ x : ℝ, x^2 + 2023 * x - 2024 = 0 → 
  x = -2023 := 
sorry

end sum_of_roots_of_quadratic_eq_l1162_116255


namespace one_plus_x_pow_gt_one_plus_nx_l1162_116235

theorem one_plus_x_pow_gt_one_plus_nx (x : ℝ) (n : ℕ) (hx1 : x > -1) (hx2 : x ≠ 0)
  (hn1 : n ≥ 2) : (1 + x)^n > 1 + n * x :=
sorry

end one_plus_x_pow_gt_one_plus_nx_l1162_116235


namespace circular_arc_sum_l1162_116224

theorem circular_arc_sum (n : ℕ) (h₁ : n > 0) :
  ∀ s : ℕ, (1 ≤ s ∧ s ≤ (n * (n + 1)) / 2) →
  ∃ arc_sum : ℕ, arc_sum = s := 
by
  sorry

end circular_arc_sum_l1162_116224


namespace single_colony_habitat_limit_reach_time_l1162_116206

noncomputable def doubling_time (n : ℕ) : ℕ := 2^n

theorem single_colony_habitat_limit_reach_time :
  ∀ (S : ℕ), ∀ (n : ℕ), doubling_time (n + 1) = S → doubling_time (2 * (n - 1)) = S → n + 1 = 16 :=
by
  intros S n H1 H2
  sorry

end single_colony_habitat_limit_reach_time_l1162_116206


namespace minutes_past_midnight_l1162_116252

-- Definitions for the problem

def degree_per_tick : ℝ := 30
def degree_per_minute_hand : ℝ := 6
def degree_per_hour_hand_hourly : ℝ := 30
def degree_per_hour_hand_minutes : ℝ := 0.5

def condition_minute_hand_degree := 300
def condition_hour_hand_degree := 70

-- Main theorem statement
theorem minutes_past_midnight :
  ∃ (h m: ℝ),
    degree_per_hour_hand_hourly * h + degree_per_hour_hand_minutes * m = condition_hour_hand_degree ∧
    degree_per_minute_hand * m = condition_minute_hand_degree ∧
    h * 60 + m = 110 :=
by
  sorry

end minutes_past_midnight_l1162_116252


namespace range_of_inclination_angle_l1162_116295

theorem range_of_inclination_angle (α : ℝ) (A : ℝ × ℝ) (hA : A = (-2, 0))
  (ellipse_eq : ∀ (x y : ℝ), (x^2 / 2) + y^2 = 1) :
    (0 ≤ α ∧ α < Real.arcsin (Real.sqrt 3 / 3)) ∨ 
    (π - Real.arcsin (Real.sqrt 3 / 3) < α ∧ α < π) :=
sorry

end range_of_inclination_angle_l1162_116295


namespace find_missing_number_l1162_116220

theorem find_missing_number:
  ∃ x : ℕ, (306 / 34) * 15 + x = 405 := sorry

end find_missing_number_l1162_116220


namespace number_of_chairs_in_first_row_l1162_116283

-- Define the number of chairs in each row
def chairs_in_second_row := 23
def chairs_in_third_row := 32
def chairs_in_fourth_row := 41
def chairs_in_fifth_row := 50
def chairs_in_sixth_row := 59

-- Define the pattern increment
def increment := 9

-- Define a function to calculate the number of chairs in a given row, given the increment pattern
def chairs_in_row (n : Nat) : Nat :=
if n = 1 then (chairs_in_second_row - increment)
else if n = 2 then chairs_in_second_row
else if n = 3 then chairs_in_third_row
else if n = 4 then chairs_in_fourth_row
else if n = 5 then chairs_in_fifth_row
else if n = 6 then chairs_in_sixth_row
else chairs_in_second_row + (n - 2) * increment

-- The theorem to prove: The number of chairs in the first row is 14
theorem number_of_chairs_in_first_row : chairs_in_row 1 = 14 :=
  by sorry

end number_of_chairs_in_first_row_l1162_116283


namespace max_value_f_diff_l1162_116270

open Real

noncomputable def f (A ω : ℝ) (x : ℝ) := A * sin (ω * x + π / 6) - 1

theorem max_value_f_diff {A ω : ℝ} (hA : A > 0) (hω : ω > 0)
  (h_sym : (π / 2) = π / (2 * ω))
  (h_initial : f A ω (π / 6) = 1) :
  ∀ (x1 x2 : ℝ), (0 ≤ x1 ∧ x1 ≤ π / 2) ∧ (0 ≤ x2 ∧ x2 ≤ π / 2) →
  (f A ω x1 - f A ω x2 ≤ 3) :=
sorry

end max_value_f_diff_l1162_116270


namespace avg_monthly_bill_over_6_months_l1162_116267

theorem avg_monthly_bill_over_6_months :
  ∀ (avg_first_4_months avg_last_2_months : ℝ), 
  avg_first_4_months = 30 → 
  avg_last_2_months = 24 → 
  (4 * avg_first_4_months + 2 * avg_last_2_months) / 6 = 28 :=
by
  intros
  sorry

end avg_monthly_bill_over_6_months_l1162_116267


namespace range_of_k_l1162_116204

theorem range_of_k (f : ℝ → ℝ) (a : ℝ) (k : ℝ) 
  (h₀ : ∀ x > 0, f x = 2 - 1 / (a - x)^2) 
  (h₁ : ∀ x > 0, k^2 * x + f (1 / 4 * x + 1) > 0) : 
  k ≠ 0 :=
by
  -- proof goes here
  sorry

end range_of_k_l1162_116204


namespace cylinder_original_radius_l1162_116273

theorem cylinder_original_radius
  (r : ℝ)
  (h_original : ℝ := 4)
  (h_increased : ℝ := 3 * h_original)
  (volume_eq : π * (r + 8)^2 * h_original = π * r^2 * h_increased) :
  r = 4 + 4 * Real.sqrt 5 :=
sorry

end cylinder_original_radius_l1162_116273


namespace find_p_l1162_116281

variable (a b p q r1 r2 : ℝ)

-- Given conditions
def roots_eq1 (h_1 : r1 + r2 = -a) (h_2 : r1 * r2 = b) : Prop :=
  -- Using Vieta's Formulas on x^2 + ax + b = 0
  ∀ (r1 r2 : ℝ), r1 + r2 = -a ∧ r1 * r2 = b

def roots_eq2 (r1 r2 : ℝ) (h_3 : r1^2 + r2^2 = -p) (h_4 : r1^2 * r2^2 = q) : Prop :=
  -- Using Vieta's Formulas on x^2 + px + q = 0
  ∀ (r1 r2 : ℝ), r1^2 + r2^2 = -p ∧ r1^2 * r2^2 = q

-- Theorems
theorem find_p (h_1 : r1 + r2 = -a) (h_2 : r1 * r2 = b) (h_3 : r1^2 + r2^2 = -p) :
  p = -a^2 + 2*b := by
  sorry

end find_p_l1162_116281


namespace problem_statement_l1162_116286

-- Define the repeating decimal 0.000272727... as x
noncomputable def repeatingDecimal : ℚ := 3 / 11000

-- Define the given condition for the question
def decimalRepeatsIndefinitely : Prop := 
  repeatingDecimal = 0.0002727272727272727  -- Representation for repeating decimal

-- Definitions of large powers of 10
def ten_pow_5 := 10^5
def ten_pow_3 := 10^3

-- The problem statement
theorem problem_statement : decimalRepeatsIndefinitely →
  (ten_pow_5 - ten_pow_3) * repeatingDecimal = 27 :=
sorry

end problem_statement_l1162_116286


namespace find_a_l1162_116217

theorem find_a (a : ℝ) (h : (2 - -3) / (1 - a) = Real.tan (135 * Real.pi / 180)) : a = 6 :=
sorry

end find_a_l1162_116217


namespace simplify_expression_l1162_116260

theorem simplify_expression (a b : ℝ) (h1 : a ≠ b) (h2 : a ≠ 0) (h3 : b ≠ 0) :
  ( (1/(a-b) - 2 * a * b / (a^3 - a^2 * b + a * b^2 - b^3)) / 
    ((a^2 + a * b) / (a^3 + a^2 * b + a * b^2 + b^3) + 
    b / (a^2 + b^2)) ) = (a - b) / (a + b) :=
by
  sorry

end simplify_expression_l1162_116260


namespace maximum_marks_l1162_116279

theorem maximum_marks (M : ℝ) (P : ℝ) 
  (h1 : P = 0.45 * M) -- 45% of the maximum marks to pass
  (h2 : P = 210 + 40) -- Pradeep's marks plus failed marks

  : M = 556 := 
sorry

end maximum_marks_l1162_116279


namespace probability_at_least_3_out_of_6_babies_speak_l1162_116290

noncomputable def binomial_prob (n : ℕ) (k : ℕ) (p : ℚ) : ℚ :=
  Nat.choose n k * (p^k) * ((1 - p)^(n - k))

noncomputable def prob_at_least_k (total : ℕ) (k : ℕ) (p : ℚ) : ℚ :=
  1 - (Finset.range k).sum (λ i => binomial_prob total i p)

theorem probability_at_least_3_out_of_6_babies_speak :
  prob_at_least_k 6 3 (2/5) = 7120/15625 :=
by
  sorry

end probability_at_least_3_out_of_6_babies_speak_l1162_116290


namespace part1_part2_l1162_116293

theorem part1 (x y : ℝ) (h1 : x + 3 * y = 26) (h2 : 2 * x + y = 22) : x = 8 ∧ y = 6 :=
by
  sorry

theorem part2 (m : ℝ) (h : 8 * m + 6 * (15 - m) ≤ 100) : m ≤ 5 :=
by
  sorry

end part1_part2_l1162_116293


namespace distance_between_X_and_Y_l1162_116242

def distance_XY := 31

theorem distance_between_X_and_Y
  (yolanda_rate : ℕ) (bob_rate : ℕ) (bob_walked : ℕ) (time_difference : ℕ) :
  yolanda_rate = 1 →
  bob_rate = 2 →
  bob_walked = 20 →
  time_difference = 1 →
  distance_XY = bob_walked + (bob_walked / bob_rate + time_difference) * yolanda_rate :=
by
  intros hy hb hbw htd
  sorry

end distance_between_X_and_Y_l1162_116242


namespace machine_A_sprockets_per_hour_l1162_116228

theorem machine_A_sprockets_per_hour :
  ∃ (A : ℝ), 
    (∃ (G : ℝ), 
      (G = 1.10 * A) ∧ 
      (∃ (T : ℝ), 
        (660 = A * (T + 10)) ∧ 
        (660 = G * T) 
      )
    ) ∧ 
    (A = 6) :=
by
  -- Conditions and variables will be introduced here...
  -- Proof can be implemented here
  sorry

end machine_A_sprockets_per_hour_l1162_116228


namespace bricks_needed_l1162_116219

noncomputable def volume (length : ℝ) (width : ℝ) (height : ℝ) : ℝ := length * width * height

theorem bricks_needed
  (brick_length : ℝ)
  (brick_width : ℝ)
  (brick_height : ℝ)
  (wall_length : ℝ)
  (wall_height : ℝ)
  (wall_thickness : ℝ)
  (hl : brick_length = 40)
  (hw : brick_width = 11.25)
  (hh : brick_height = 6)
  (wl : wall_length = 800)
  (wh : wall_height = 600)
  (wt : wall_thickness = 22.5) :
  (volume wall_length wall_height wall_thickness / volume brick_length brick_width brick_height) = 4000 := by
  sorry

end bricks_needed_l1162_116219


namespace acute_angle_sine_l1162_116278

theorem acute_angle_sine (α : ℝ) (h1 : 0 < α) (h2 : α < π / 2) (h3 : Real.sin α = 0.58) : (π / 6) < α ∧ α < (π / 4) :=
by
  sorry

end acute_angle_sine_l1162_116278


namespace solve_system_of_eqns_l1162_116203

theorem solve_system_of_eqns :
  ∃ x y : ℝ, (x^2 + x * y + y = 1 ∧ y^2 + x * y + x = 5) ∧ ((x = -1 ∧ y = 3) ∨ (x = -1 ∧ y = -2)) :=
by
  sorry

end solve_system_of_eqns_l1162_116203


namespace num_ordered_pairs_c_d_l1162_116230

def is_solution (c d x y : ℤ) : Prop :=
  c * x + d * y = 2 ∧ x^2 + y^2 = 65

theorem num_ordered_pairs_c_d : 
  ∃ (S : Finset (ℤ × ℤ)), S.card = 136 ∧ 
  ∀ (c d : ℤ), (c, d) ∈ S ↔ ∃ (x y : ℤ), is_solution c d x y :=
sorry

end num_ordered_pairs_c_d_l1162_116230


namespace glass_bottles_in_second_scenario_l1162_116202

theorem glass_bottles_in_second_scenario
  (G P x : ℕ)
  (h1 : 3 * G = 600)
  (h2 : G = P + 150)
  (h3 : x * G + 5 * P = 1050) :
  x = 4 :=
by 
  -- Proof is omitted
  sorry

end glass_bottles_in_second_scenario_l1162_116202


namespace find_complex_number_l1162_116274

open Complex

theorem find_complex_number (z : ℂ) (hz : z + Complex.abs z = Complex.ofReal 2 + 8 * Complex.I) : 
z = -15 + 8 * Complex.I := by sorry

end find_complex_number_l1162_116274


namespace john_average_speed_l1162_116276

/--
John drove continuously from 8:15 a.m. until 2:05 p.m. of the same day 
and covered a distance of 210 miles. Prove that his average speed in 
miles per hour was 36 mph.
-/
theorem john_average_speed :
  (210 : ℝ) / (((2 - 8) * 60 + 5 - 15) / 60) = 36 := by
  sorry

end john_average_speed_l1162_116276


namespace positive_difference_of_numbers_l1162_116256

theorem positive_difference_of_numbers (x : ℝ) (h : (30 + x) / 2 = 34) : abs (x - 30) = 8 :=
by
  sorry

end positive_difference_of_numbers_l1162_116256


namespace solution_inequality_l1162_116240

theorem solution_inequality (m : ℝ) :
  (∀ x : ℝ, x^2 - (m+3)*x + 3*m < 0 ↔ m ∈ Set.Icc 3 (-1) ∪ Set.Icc 6 7) →
  m = -1/2 ∨ m = 13/2 :=
sorry

end solution_inequality_l1162_116240


namespace cost_of_lunch_l1162_116282

-- Define the conditions: total amount and tip percentage
def total_amount : ℝ := 72.6
def tip_percentage : ℝ := 0.20

-- Define the proof problem
theorem cost_of_lunch (C : ℝ) (h : C + tip_percentage * C = total_amount) : C = 60.5 := 
sorry

end cost_of_lunch_l1162_116282


namespace min_value_of_vectors_l1162_116288

theorem min_value_of_vectors (m n : ℝ) (h1 : m > 0) (h2 : n > 0) 
  (h3 : (m * (n - 2)) + 1 = 0) : (1 / m) + (2 / n) = 2 * Real.sqrt 2 + 3 / 2 :=
by sorry

end min_value_of_vectors_l1162_116288


namespace sin_150_eq_half_l1162_116234

theorem sin_150_eq_half : Real.sin (150 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end sin_150_eq_half_l1162_116234


namespace line_through_point_parallel_to_given_line_l1162_116231

theorem line_through_point_parallel_to_given_line 
  (x y : ℝ) 
  (h₁ : (x, y) = (1, -4)) 
  (h₂ : ∀ m : ℝ, 2 * 1 + 3 * (-4) + m = 0 → m = 10)
  : 2 * x + 3 * y + 10 = 0 :=
sorry

end line_through_point_parallel_to_given_line_l1162_116231


namespace aimee_poll_l1162_116241

theorem aimee_poll (P : ℕ) (h1 : 35 ≤ 100) (h2 : 39 % (P/2) = 39) : P = 120 := 
by sorry

end aimee_poll_l1162_116241


namespace tangent_line_at_point_e_tangent_line_from_origin_l1162_116271

-- Problem 1
theorem tangent_line_at_point_e (x y : ℝ) (h : y = Real.exp x) (h_e : x = Real.exp 1) :
    (Real.exp x) * x - y - Real.exp (x + 1) = 0 :=
sorry

-- Problem 2
theorem tangent_line_from_origin (x y : ℝ) (h : y = Real.exp x) :
    x = 1 →  Real.exp x * x - y = 0 :=
sorry

end tangent_line_at_point_e_tangent_line_from_origin_l1162_116271


namespace find_certain_number_l1162_116247

theorem find_certain_number (x : ℝ) (h : x + 12.952 - 47.95000000000027 = 3854.002) : x = 3889.000 :=
sorry

end find_certain_number_l1162_116247


namespace alex_annual_income_l1162_116287

theorem alex_annual_income (q : ℝ) (B : ℝ)
  (H1 : 0.01 * q * 50000 + 0.01 * (q + 3) * (B - 50000) = 0.01 * (q + 0.5) * B) :
  B = 60000 :=
by sorry

end alex_annual_income_l1162_116287


namespace orange_ratio_l1162_116251

theorem orange_ratio (total_oranges alice_oranges : ℕ) (h_total : total_oranges = 180) (h_alice : alice_oranges = 120) :
  alice_oranges / (total_oranges - alice_oranges) = 2 :=
by
  sorry

end orange_ratio_l1162_116251


namespace maximum_visibility_sum_l1162_116237

theorem maximum_visibility_sum (X Y : ℕ) (h : X + 2 * Y = 30) :
  X * Y ≤ 112 :=
by
  sorry

end maximum_visibility_sum_l1162_116237


namespace arun_remaining_work_days_l1162_116225

noncomputable def arun_and_tarun_work_in_days (W : ℝ) := 10
noncomputable def arun_alone_work_in_days (W : ℝ) := 60
noncomputable def arun_tarun_together_days := 4

theorem arun_remaining_work_days (W : ℝ) :
  (arun_and_tarun_work_in_days W = 10) ∧
  (arun_alone_work_in_days W = 60) ∧
  (let complete_work_days := arun_tarun_together_days;
  let remaining_work := W - (complete_work_days / arun_and_tarun_work_in_days W * W);
  let arun_remaining_days := (remaining_work / W) * arun_alone_work_in_days W;
  arun_remaining_days = 36) :=
sorry

end arun_remaining_work_days_l1162_116225


namespace olivia_worked_hours_on_wednesday_l1162_116222

-- Define the conditions
def hourly_rate := 9
def hours_monday := 4
def hours_friday := 6
def total_earnings := 117
def earnings_monday := hours_monday * hourly_rate
def earnings_friday := hours_friday * hourly_rate
def earnings_wednesday := total_earnings - (earnings_monday + earnings_friday)

-- Define the number of hours worked on Wednesday
def hours_wednesday := earnings_wednesday / hourly_rate

-- The theorem to prove
theorem olivia_worked_hours_on_wednesday : hours_wednesday = 3 :=
by
  -- Skip the proof
  sorry

end olivia_worked_hours_on_wednesday_l1162_116222


namespace petrol_price_l1162_116233

theorem petrol_price (P : ℝ) (h : 0.9 * P = 0.9 * P) : (250 / (0.9 * P) - 250 / P = 5) → P = 5.56 :=
by
  sorry

end petrol_price_l1162_116233

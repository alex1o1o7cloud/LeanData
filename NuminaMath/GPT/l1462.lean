import Mathlib

namespace smallest_positive_period_of_f_max_min_values_of_f_on_interval_l1462_146248

noncomputable def f (x : ℝ) : ℝ := (Real.sin x + Real.cos x) ^ 2 + Real.cos (2 * x)

theorem smallest_positive_period_of_f : ∀ (x : ℝ), f (x + π) = f x :=
by sorry

theorem max_min_values_of_f_on_interval : ∃ (x₁ x₂ : ℝ), 0 ≤ x₁ ∧ x₁ ≤ π / 2 ∧ 0 ≤ x₂ ∧ x₂ ≤ π / 2 ∧
  f x₁ = 0 ∧ f x₂ = 1 + Real.sqrt 2 :=
by sorry

end smallest_positive_period_of_f_max_min_values_of_f_on_interval_l1462_146248


namespace hemisphere_surface_area_l1462_146210

theorem hemisphere_surface_area (r : ℝ) (π : ℝ) (area_base : ℝ) (surface_area_sphere : ℝ) (Q : ℝ) : 
  area_base = 3 ∧ surface_area_sphere = 4 * π * r^2 → Q = 9 :=
by
  sorry

end hemisphere_surface_area_l1462_146210


namespace paul_and_paula_cookies_l1462_146217

-- Define the number of cookies per pack type
def cookies_in_pack (pack : ℕ) : ℕ :=
  match pack with
  | 1 => 15
  | 2 => 30
  | 3 => 45
  | 4 => 60
  | _ => 0

-- Paul's purchase: 2 packs of Pack B and 1 pack of Pack A
def pauls_cookies : ℕ :=
  2 * cookies_in_pack 2 + cookies_in_pack 1

-- Paula's purchase: 1 pack of Pack A and 1 pack of Pack C
def paulas_cookies : ℕ :=
  cookies_in_pack 1 + cookies_in_pack 3

-- Total number of cookies Paul and Paula have
def total_cookies : ℕ :=
  pauls_cookies + paulas_cookies

theorem paul_and_paula_cookies : total_cookies = 135 :=
by
  sorry

end paul_and_paula_cookies_l1462_146217


namespace find_other_number_l1462_146216

-- Given conditions
def sum_of_numbers (x y : ℕ) : Prop := x + y = 72
def number_difference (x y : ℕ) : Prop := x = y + 12
def one_number_is_30 (x : ℕ) : Prop := x = 30

-- Theorem to prove
theorem find_other_number (y : ℕ) : 
  sum_of_numbers y 30 ∧ number_difference 30 y → y = 18 := by
  sorry

end find_other_number_l1462_146216


namespace knights_and_liars_l1462_146284

-- Define the conditions: 
variables (K L : ℕ) 

-- Total number of council members is 101
def total_members : Prop := K + L = 101

-- Inequality conditions
def knight_inequality : Prop := L > (K + L - 1) / 2
def liar_inequality : Prop := K <= (K + L - 1) / 2

-- The theorem we need to prove
theorem knights_and_liars (K L : ℕ) (h1 : total_members K L) (h2 : knight_inequality K L) (h3 : liar_inequality K L) : K = 50 ∧ L = 51 :=
by {
  sorry
}

end knights_and_liars_l1462_146284


namespace exists_x0_in_interval_l1462_146238

noncomputable def f (x : ℝ) : ℝ := (2 : ℝ) / x + Real.log (1 / (x - 1))

theorem exists_x0_in_interval :
  ∃ x0 ∈ Set.Ioo (2 : ℝ) (3 : ℝ), f x0 = 0 := 
sorry  -- Proof is left as an exercise

end exists_x0_in_interval_l1462_146238


namespace children_playing_tennis_l1462_146232

theorem children_playing_tennis
  (Total : ℕ) (S : ℕ) (N : ℕ) (B : ℕ) (T : ℕ) 
  (hTotal : Total = 38) (hS : S = 21) (hN : N = 10) (hB : B = 12) :
  T = 38 - 21 + 12 - 10 :=
by
  sorry

end children_playing_tennis_l1462_146232


namespace sweeties_remainder_l1462_146228

theorem sweeties_remainder (m : ℕ) (h : m % 6 = 4) : (2 * m) % 6 = 2 :=
by {
  sorry
}

end sweeties_remainder_l1462_146228


namespace part1_part2_l1462_146241

def f (x : ℝ) : ℝ := |x - 1| + 2 * |x + 1|

theorem part1 : {x : ℝ | f x ≤ 4} = {x : ℝ | -5 / 3 ≤ x ∧ x ≤ 1} :=
by
  sorry

theorem part2 {a : ℝ} :
  ({x : ℝ | f x ≤ 4} ⊆ {x : ℝ | |x + 3| + |x + a| < x + 6}) ↔ (-4 / 3 < a ∧ a < 2) :=
by
  sorry

end part1_part2_l1462_146241


namespace father_20_bills_count_l1462_146235

-- Defining the conditions from the problem.
variables (mother50 mother20 mother10 father50 father10 : ℕ)
def mother_total := mother50 * 50 + mother20 * 20 + mother10 * 10
def father_total (x : ℕ) := father50 * 50 + x * 20 + father10 * 10

-- Given conditions
axiom mother_given : mother50 = 1 ∧ mother20 = 2 ∧ mother10 = 3
axiom father_given : father50 = 4 ∧ father10 = 1
axiom school_fee : 350 = 350

-- Theorem to prove
theorem father_20_bills_count (x : ℕ) :
  mother_total 1 2 3 + father_total 4 x 1 = 350 → x = 1 :=
by sorry

end father_20_bills_count_l1462_146235


namespace simplify_and_evaluate_expression_l1462_146254

variable (x y : ℝ)

theorem simplify_and_evaluate_expression
  (hx : x = 2)
  (hy : y = -0.5) :
  2 * (2 * x - 3 * y) - (3 * x + 2 * y + 1) = 5 :=
by
  sorry

end simplify_and_evaluate_expression_l1462_146254


namespace intersection_A_B_l1462_146243

noncomputable def A : Set ℝ := { x | (x - 1) / (x + 3) < 0 }
noncomputable def B : Set ℝ := { x | abs x < 2 }

theorem intersection_A_B :
  A ∩ B = { x : ℝ | -2 < x ∧ x < 1 } :=
by
  sorry

end intersection_A_B_l1462_146243


namespace base_conversion_l1462_146253

theorem base_conversion (b : ℕ) (h : 1 * 6^2 + 4 * 6 + 2 = 2 * b^2 + b + 5) : b = 5 :=
by
  sorry

end base_conversion_l1462_146253


namespace sequence_general_term_l1462_146263

theorem sequence_general_term (a : ℕ → ℝ) (h1 : a 1 = 1) (h2 : ∀ n, a (n + 1) = (1 / 2) * a n + 1) :
  ∀ n, a n = 2 - (1 / 2) ^ (n - 1) :=
by
  sorry

end sequence_general_term_l1462_146263


namespace village_population_l1462_146294

noncomputable def number_of_people_in_village
  (vampire_drains_per_week : ℕ)
  (werewolf_eats_per_week : ℕ)
  (weeks : ℕ) : ℕ :=
  let drained := vampire_drains_per_week * weeks
  let eaten := werewolf_eats_per_week * weeks
  drained + eaten

theorem village_population :
  number_of_people_in_village 3 5 9 = 72 := by
  sorry

end village_population_l1462_146294


namespace find_tabitha_age_l1462_146262

-- Define the conditions
variable (age_started : ℕ) (colors_started : ℕ) (years_future : ℕ) (future_colors : ℕ)

-- Let's specify the given problem's conditions:
axiom h1 : age_started = 15          -- Tabitha started at age 15
axiom h2 : colors_started = 2        -- with 2 colors
axiom h3 : years_future = 3          -- in three years
axiom h4 : future_colors = 8         -- she will have 8 different colors

-- The proof problem we need to state:
theorem find_tabitha_age : ∃ age_now : ℕ, age_now = age_started + (future_colors - colors_started) - years_future := by
  sorry

end find_tabitha_age_l1462_146262


namespace total_length_circle_l1462_146274

-- Definitions based on conditions
def num_strips : ℕ := 16
def length_each_strip : ℝ := 10.4
def overlap_each_strip : ℝ := 3.5

-- Theorem stating the total length of the circle-shaped colored tape
theorem total_length_circle : 
  (num_strips * length_each_strip) - (num_strips * overlap_each_strip) = 110.4 := 
by 
  sorry

end total_length_circle_l1462_146274


namespace raul_money_left_l1462_146236

theorem raul_money_left (initial_money : ℕ) (cost_per_comic : ℕ) (number_of_comics : ℕ) (money_left : ℕ)
  (h1 : initial_money = 87)
  (h2 : cost_per_comic = 4)
  (h3 : number_of_comics = 8)
  (h4 : money_left = initial_money - (number_of_comics * cost_per_comic)) :
  money_left = 55 :=
by 
  rw [h1, h2, h3] at h4
  exact h4

end raul_money_left_l1462_146236


namespace odd_two_digit_combinations_l1462_146258

theorem odd_two_digit_combinations (digits : Finset ℕ) (h_digits : digits = {1, 3, 5, 7, 9}) :
  ∃ n : ℕ, n = 20 ∧ (∃ a b : ℕ, a ∈ digits ∧ b ∈ digits ∧ a ≠ b ∧ (10 * a + b) % 2 = 1) :=
by
  sorry

end odd_two_digit_combinations_l1462_146258


namespace area_ADC_calculation_l1462_146286

-- Definitions and assumptions
variables (BD DC : ℝ)
variables (area_ABD area_ADC : ℝ)

-- Given conditions
axiom ratio_BD_DC : BD / DC = 2 / 5
axiom area_ABD_given : area_ABD = 40

-- The theorem to prove
theorem area_ADC_calculation (h1 : BD / DC = 2 / 5) (h2 : area_ABD = 40) :
  area_ADC = 100 :=
sorry

end area_ADC_calculation_l1462_146286


namespace arithmetic_sequence_properties_l1462_146218

-- Definitions and conditions
def S (n : ℕ) : ℤ := -2 * n^2 + 15 * n

-- Statement of the problem as a theorem
theorem arithmetic_sequence_properties :
  (∀ n : ℕ, S (n + 1) - S n = 17 - 4 * (n + 1)) ∧
  (∃ n : ℕ, S n = 28 ∧ ∀ m : ℕ, S m ≤ S n) :=
by {sorry}

end arithmetic_sequence_properties_l1462_146218


namespace sum_first_n_terms_eq_l1462_146247

noncomputable def a_n (n : ℕ) : ℕ := 2 * n - 1

noncomputable def b_n (n : ℕ) : ℕ := 2 ^ (n - 1)

noncomputable def c_n (n : ℕ) : ℕ := a_n n * b_n n

noncomputable def T_n (n : ℕ) : ℕ := (2 * n - 3) * 2 ^ n + 3

theorem sum_first_n_terms_eq (n : ℕ) : 
  (Finset.sum (Finset.range n.succ) (λ k => c_n k) = T_n n) :=
  sorry

end sum_first_n_terms_eq_l1462_146247


namespace rectangle_area_l1462_146205

-- Conditions
def radius : ℝ := 6
def diameter : ℝ := 2 * radius
def width : ℝ := diameter
def ratio_length_to_width : ℝ := 3

-- Given the ratio of the length to the width is 3:1
def length : ℝ := ratio_length_to_width * width

-- Theorem stating the area of the rectangle
theorem rectangle_area :
  let area := length * width
  area = 432 := by
    sorry

end rectangle_area_l1462_146205


namespace production_cost_decrease_l1462_146281

theorem production_cost_decrease (x : ℝ) :
  let initial_production_cost := 50
  let initial_selling_price := 65
  let first_quarter_decrease := 0.10
  let second_quarter_increase := 0.05
  let final_selling_price := initial_selling_price * (1 - first_quarter_decrease) * (1 + second_quarter_increase)
  let original_profit := initial_selling_price - initial_production_cost
  let final_production_cost := initial_production_cost * (1 - x) ^ 2
  (final_selling_price - final_production_cost) = original_profit :=
by
  sorry

end production_cost_decrease_l1462_146281


namespace not_a_factorization_method_l1462_146240

def factorization_methods : Set String := 
  {"Taking out the common factor", "Cross multiplication method", "Formula method", "Group factorization"}

theorem not_a_factorization_method : 
  ¬ ("Addition and subtraction elimination method" ∈ factorization_methods) :=
sorry

end not_a_factorization_method_l1462_146240


namespace tan_beta_eq_neg13_l1462_146280

variables (α β : Real)

theorem tan_beta_eq_neg13 (h1 : Real.tan α = 2) (h2 : Real.tan (α - β) = -3/5) : 
  Real.tan β = -13 := 
by 
  sorry

end tan_beta_eq_neg13_l1462_146280


namespace inv_203_mod_301_exists_l1462_146283

theorem inv_203_mod_301_exists : ∃ b : ℤ, 203 * b % 301 = 1 := sorry

end inv_203_mod_301_exists_l1462_146283


namespace axis_of_symmetry_compare_m_n_range_of_t_for_y1_leq_y2_maximum_value_of_t_l1462_146237

-- (1) Axis of symmetry
theorem axis_of_symmetry (t : ℝ) :
  ∀ x y : ℝ, (y = x^2 - 2*t*x + 1) → (x = t) := sorry

-- (2) Comparison of m and n
theorem compare_m_n (t m n : ℝ) :
  (t - 2)^2 - 2*t*(t - 2) + 1 = m*1 →
  (t + 3)^2 - 2*t*(t + 3) + 1 = n*1 →
  n > m := sorry

-- (3) Range of t for y₁ ≤ y₂
theorem range_of_t_for_y1_leq_y2 (t x1 x2 y1 y2 : ℝ) :
  (-1 ≤ x1) → (x1 < 3) → (x2 = 3) → 
  (y1 = x1^2 - 2*t*x1 + 1) → 
  (y2 = x2^2 - 2*t*x2 + 1) → 
  y1 ≤ y2 →
  t ≤ 1 := sorry

-- (4) Maximum value of t
theorem maximum_value_of_t (t y1 y2 : ℝ) :
  (y1 = (t + 1)^2 - 2*t*(t + 1) + 1) →
  (y2 = (2*t - 4)^2 - 2*t*(2*t - 4) + 1) →
  y1 ≥ y2 →
  t = 5 := sorry

end axis_of_symmetry_compare_m_n_range_of_t_for_y1_leq_y2_maximum_value_of_t_l1462_146237


namespace sqrt_prime_geometric_progression_impossible_l1462_146242

theorem sqrt_prime_geometric_progression_impossible {p1 p2 p3 : ℕ} (hp1 : Nat.Prime p1) (hp2 : Nat.Prime p2) (hp3 : Nat.Prime p3) (hneq12 : p1 ≠ p2) (hneq23 : p2 ≠ p3) (hneq31 : p3 ≠ p1) :
  ¬ ∃ (a r : ℝ) (n1 n2 n3 : ℤ), (a * r^n1 = Real.sqrt p1) ∧ (a * r^n2 = Real.sqrt p2) ∧ (a * r^n3 = Real.sqrt p3) := sorry

end sqrt_prime_geometric_progression_impossible_l1462_146242


namespace triangle_area_eq_40_sqrt_3_l1462_146246

open Real

theorem triangle_area_eq_40_sqrt_3 
  (a : ℝ) (A : ℝ) (b c : ℝ)
  (h1 : a = 14)
  (h2 : A = π / 3) -- 60 degrees in radians
  (h3 : b / c = 8 / 5) :
  1 / 2 * b * c * sin A = 40 * sqrt 3 :=
by
  sorry

end triangle_area_eq_40_sqrt_3_l1462_146246


namespace no_real_satisfies_absolute_value_equation_l1462_146245

theorem no_real_satisfies_absolute_value_equation :
  ∀ x : ℝ, ¬ (|x - 2| = |x - 1| + |x - 5|) :=
by
  sorry

end no_real_satisfies_absolute_value_equation_l1462_146245


namespace ratio_of_board_pieces_l1462_146257

theorem ratio_of_board_pieces (S L : ℕ) (hS : S = 23) (hTotal : S + L = 69) : L / S = 2 :=
by
  sorry

end ratio_of_board_pieces_l1462_146257


namespace fraction_sum_geq_zero_l1462_146203

theorem fraction_sum_geq_zero (a b c : ℝ) (h1 : a > b) (h2 : b > c) : 
  (1 / (a - b) + 1 / (b - c) + 4 / (c - a)) ≥ 0 := 
by 
  sorry

end fraction_sum_geq_zero_l1462_146203


namespace class_students_l1462_146229

theorem class_students :
  ∃ n : ℕ,
    (∃ m : ℕ, 2 * m = n) ∧
    (∃ q : ℕ, 4 * q = n) ∧
    (∃ l : ℕ, 7 * l = n) ∧
    (∀ f : ℕ, f < 6 → n - (n / 2) - (n / 4) - (n / 7) = f) ∧
    n = 28 :=
by
  sorry

end class_students_l1462_146229


namespace solve_for_x_l1462_146207
-- Import the entire Mathlib library

-- Define the condition
def condition (x : ℝ) := (72 - x)^2 = x^2

-- State the theorem
theorem solve_for_x : ∃ x : ℝ, condition x ∧ x = 36 :=
by {
  -- The proof will be provided here
  sorry
}

end solve_for_x_l1462_146207


namespace sale_price_with_50_percent_profit_l1462_146269

theorem sale_price_with_50_percent_profit (CP SP₁ SP₃ : ℝ) 
(h1 : SP₁ - CP = CP - 448) 
(h2 : SP₃ = 1.5 * CP) 
(h3 : SP₃ = 1020) : 
SP₃ = 1020 := 
by 
  sorry

end sale_price_with_50_percent_profit_l1462_146269


namespace length_of_plot_l1462_146252

-- Definitions of the given conditions, along with the question.
def breadth (b : ℝ) : Prop := 2 * (b + 32) + 2 * b = 5300 / 26.50
def length (b : ℝ) := b + 32

theorem length_of_plot (b : ℝ) (h : breadth b) : length b = 66 := by 
  sorry

end length_of_plot_l1462_146252


namespace sequence_problem_l1462_146298

-- Given sequence
variable (P Q R S T U V : ℤ)

-- Given conditions
variable (hR : R = 7)
variable (hPQ : P + Q + R = 21)
variable (hQS : Q + R + S = 21)
variable (hST : R + S + T = 21)
variable (hTU : S + T + U = 21)
variable (hUV : T + U + V = 21)

theorem sequence_problem : P + V = 14 := by
  sorry

end sequence_problem_l1462_146298


namespace eccentricity_of_hyperbola_l1462_146222

noncomputable def hyperbola_eccentricity (a b c : ℝ) (ha : a > 0) (hb : b > 0) (h : (4:ℝ) * a^2 = c^2) : ℝ :=
  c / a

theorem eccentricity_of_hyperbola (a b c : ℝ) (ha : a > 0) (hb : b > 0) (h : (4:ℝ) * a^2 = c^2) :
  hyperbola_eccentricity a b c ha hb h = 2 :=
by
  sorry


end eccentricity_of_hyperbola_l1462_146222


namespace xyz_unique_solution_l1462_146224

theorem xyz_unique_solution (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_eq : x + y^2 + z^3 = x * y * z)
  (h_gcd : z = Nat.gcd x y) : x = 5 ∧ y = 1 ∧ z = 1 :=
by
  sorry

end xyz_unique_solution_l1462_146224


namespace rent_for_additional_hour_l1462_146214

theorem rent_for_additional_hour (x : ℝ) :
  (25 + 10 * x = 125) → (x = 10) :=
by 
  sorry

end rent_for_additional_hour_l1462_146214


namespace third_week_cases_l1462_146231

-- Define the conditions as Lean definitions
def first_week_cases : ℕ := 5000
def second_week_cases : ℕ := first_week_cases / 2
def total_cases_after_three_weeks : ℕ := 9500

-- The statement to be proven
theorem third_week_cases :
  first_week_cases + second_week_cases + 2000 = total_cases_after_three_weeks :=
by
  sorry

end third_week_cases_l1462_146231


namespace sum_of_consecutive_integers_l1462_146268

theorem sum_of_consecutive_integers (n a : ℕ) (h₁ : 2 ≤ n) (h₂ : (n * (2 * a + n - 1)) = 36) :
    ∃! (a' n' : ℕ), 2 ≤ n' ∧ (n' * (2 * a' + n' - 1)) = 36 :=
  sorry

end sum_of_consecutive_integers_l1462_146268


namespace total_earnings_l1462_146244

theorem total_earnings (d_a : ℕ) (h : 57 * d_a + 684 + 380 = 1406) : d_a = 6 :=
by {
  -- The proof will involve algebraic manipulations similar to the solution steps
  sorry
}

end total_earnings_l1462_146244


namespace relationship_of_y_values_l1462_146209

noncomputable def quadratic_function (x : ℝ) (c : ℝ) := x^2 - 6*x + c

theorem relationship_of_y_values (c : ℝ) (y1 y2 y3 : ℝ) :
  quadratic_function 1 c = y1 →
  quadratic_function (2 * Real.sqrt 2) c = y2 →
  quadratic_function 4 c = y3 →
  y3 < y2 ∧ y2 < y1 :=
by
  intros hA hB hC
  sorry

end relationship_of_y_values_l1462_146209


namespace complex_solution_l1462_146250

theorem complex_solution (i z : ℂ) (h : i^2 = -1) (hz : (z - 2 * i) * (2 - i) = 5) : z = 2 + 3 * i :=
sorry

end complex_solution_l1462_146250


namespace complex_number_first_quadrant_l1462_146202

theorem complex_number_first_quadrant (z : ℂ) (h : z = (i - 1) / i) : 
  ∃ x y : ℝ, z = x + y * I ∧ x > 0 ∧ y > 0 := 
sorry

end complex_number_first_quadrant_l1462_146202


namespace fraction_value_l1462_146230

theorem fraction_value : (1 - 1 / 4) / (1 - 1 / 5) = 15 / 16 := sorry

end fraction_value_l1462_146230


namespace sufficient_but_not_necessary_condition_l1462_146264

variables (a b : ℝ)

def p : Prop := a > b ∧ b > 1
def q : Prop := a - b < a^2 - b^2

theorem sufficient_but_not_necessary_condition (h : p a b) : q a b :=
  sorry

end sufficient_but_not_necessary_condition_l1462_146264


namespace seq_a_general_term_seq_b_general_term_inequality_k_l1462_146208

def seq_a (n : ℕ) : ℕ :=
if n = 1 then 2 else 2 * n - 1

def S (n : ℕ) : ℕ := 
match n with
| 0       => 0
| (n + 1) => S n + seq_a (n + 1)

def seq_b (n : ℕ) : ℕ := 3 ^ n

def T (n : ℕ) : ℕ := (3 ^ (n + 1) - 3) / 2

theorem seq_a_general_term (n : ℕ) : seq_a n = if n = 1 then 2 else 2 * n - 1 :=
sorry

theorem seq_b_general_term (n : ℕ) : seq_b n = 3 ^ n :=
sorry

theorem inequality_k (k : ℝ) : (∀ n : ℕ, n > 0 → (T n + 3/2 : ℝ) * k ≥ 3 * n - 6) ↔ k ≥ 2 / 27 :=
sorry

end seq_a_general_term_seq_b_general_term_inequality_k_l1462_146208


namespace period_started_at_7_am_l1462_146227

-- Define the end time of the period
def end_time : ℕ := 16 -- 4 pm in 24-hour format

-- Define the total duration in hours
def duration : ℕ := 9

-- Define the start time of the period
def start_time : ℕ := end_time - duration

-- Prove that the start time is 7 am
theorem period_started_at_7_am : start_time = 7 := by
  sorry

end period_started_at_7_am_l1462_146227


namespace set_union_proof_l1462_146200

  open Set

  def M : Set ℕ := {0, 1, 3}
  def N : Set ℕ := {x | ∃ a, a ∈ M ∧ x = 3 * a}

  theorem set_union_proof : M ∪ N = {0, 1, 3, 9} :=
  by
    sorry
  
end set_union_proof_l1462_146200


namespace cone_lateral_surface_area_l1462_146260

theorem cone_lateral_surface_area (r : ℝ) (V : ℝ) (h : ℝ) (l : ℝ) 
  (h₁ : r = 3)
  (h₂ : V = 12 * Real.pi)
  (h₃ : V = (1 / 3) * Real.pi * r^2 * h)
  (h₄ : l = Real.sqrt (r^2 + h^2)) : 
  ∃ A : ℝ, A = Real.pi * r * l ∧ A = 15 * Real.pi := 
by
  use Real.pi * r * l
  have hr : r = 3 := by exact h₁
  have hV : V = 12 * Real.pi := by exact h₂
  have volume_formula : V = (1 / 3) * Real.pi * r^2 * h := by exact h₃
  have slant_height : l = Real.sqrt (r^2 + h^2) := by exact h₄
  sorry

end cone_lateral_surface_area_l1462_146260


namespace vector_perpendicular_solution_l1462_146211

noncomputable def a (m : ℝ) : ℝ × ℝ := (1, m)
def b : ℝ × ℝ := (3, -2)

def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem vector_perpendicular_solution (m : ℝ) (h : dot_product (a m + b) b = 0) : m = 8 := by
  sorry

end vector_perpendicular_solution_l1462_146211


namespace carlos_blocks_l1462_146295

theorem carlos_blocks (initial_blocks : ℕ) (blocks_given : ℕ) (remaining_blocks : ℕ) 
  (h1 : initial_blocks = 58) (h2 : blocks_given = 21) : remaining_blocks = 37 :=
by
  sorry

end carlos_blocks_l1462_146295


namespace one_three_digit_cube_divisible_by_16_l1462_146221

theorem one_three_digit_cube_divisible_by_16 :
  ∃! (n : ℕ), (100 ≤ n ∧ n < 1000 ∧ ∃ (k : ℕ), n = k^3 ∧ 16 ∣ n) :=
sorry

end one_three_digit_cube_divisible_by_16_l1462_146221


namespace area_of_yard_l1462_146265

def length {w : ℝ} : ℝ := 2 * w + 30

def perimeter {w l : ℝ} (cond_len : l = 2 * w + 30) : Prop := 2 * w + 2 * l = 700

theorem area_of_yard {w l A : ℝ} 
  (cond_len : l = 2 * w + 30) 
  (cond_perim : 2 * w + 2 * l = 700) : 
  A = w * l := 
  sorry

end area_of_yard_l1462_146265


namespace carl_took_4_pink_hard_hats_l1462_146251

-- Define the initial number of hard hats
def initial_pink : ℕ := 26
def initial_green : ℕ := 15
def initial_yellow : ℕ := 24

-- Define the number of hard hats John took
def john_pink : ℕ := 6
def john_green : ℕ := 2 * john_pink
def john_total : ℕ := john_pink + john_green

-- Define the total initial number of hard hats
def total_initial : ℕ := initial_pink + initial_green + initial_yellow

-- Define the number of hard hats remaining after John's removal
def remaining_after_john : ℕ := total_initial - john_total

-- Define the total number of hard hats that remained in the truck
def total_remaining : ℕ := 43

-- Define the number of pink hard hats Carl took away
def carl_pink : ℕ := remaining_after_john - total_remaining

-- State the proof problem
theorem carl_took_4_pink_hard_hats : carl_pink = 4 := by
  sorry

end carl_took_4_pink_hard_hats_l1462_146251


namespace sum_of_a5_a6_l1462_146293

variable (a : ℕ → ℕ)

def S (n : ℕ) : ℕ :=
  n ^ 2 + 2

theorem sum_of_a5_a6 :
  a 5 + a 6 = S 6 - S 4 := by
  sorry

end sum_of_a5_a6_l1462_146293


namespace area_of_triangle_DEF_l1462_146296

-- Definitions of the given conditions
def angle_D : ℝ := 45
def DF : ℝ := 4
def DE : ℝ := DF -- Because it's a 45-45-90 triangle

-- Leam statement proving the area of the triangle
theorem area_of_triangle_DEF : 
  (1 / 2) * DE * DF = 8 := by
  -- Since DE = DF = 4, the area of the triangle can be computed
  sorry

end area_of_triangle_DEF_l1462_146296


namespace B_time_l1462_146249

-- Define the work rates of A, B, and C in terms of how long they take to complete the work
variable (A B C : ℝ)

-- Conditions provided in the problem
axiom A_rate : A = 1 / 3
axiom BC_rate : B + C = 1 / 3
axiom AC_rate : A + C = 1 / 2

-- Prove that B alone will take 6 hours to complete the work
theorem B_time : B = 1 / 6 → (1 / B) = 6 := by
  intro hB
  sorry

end B_time_l1462_146249


namespace A_finish_work_in_6_days_l1462_146219

theorem A_finish_work_in_6_days :
  ∃ (x : ℕ), (1 / (12:ℚ) + 1 / (x:ℚ) = 1 / (4:ℚ)) → x = 6 :=
by
  sorry

end A_finish_work_in_6_days_l1462_146219


namespace nurse_distribution_l1462_146212

theorem nurse_distribution (nurses hospitals : ℕ) (h1 : nurses = 3) (h2 : hospitals = 6) 
  (h3 : ∀ (a b c : ℕ), a = b → b = c → a = c → a ≤ 2) : 
  (hospitals^nurses - hospitals) = 210 := 
by 
  sorry

end nurse_distribution_l1462_146212


namespace water_left_in_bucket_l1462_146292

theorem water_left_in_bucket (initial_amount poured_amount : ℝ) (h1 : initial_amount = 0.8) (h2 : poured_amount = 0.2) : initial_amount - poured_amount = 0.6 := by
  sorry

end water_left_in_bucket_l1462_146292


namespace find_equation_AC_l1462_146213

noncomputable def triangleABC (A B C : (ℝ × ℝ)) : Prop :=
  B = (-2, 0) ∧ 
  ∃ (lineAB : ℝ × ℝ → ℝ), ∀ P, lineAB P = 3 * P.1 - P.2 + 6 

noncomputable def conditions (A B : (ℝ × ℝ)) : Prop :=
  (3 * B.1 - B.2 + 6 = 0) ∧ 
  (B.1 + 3 * B.2 - 26 = 0) ∧
  (A.1 + A.2 - 2 = 0)

noncomputable def equationAC (A C : (ℝ × ℝ)) : Prop :=
  (C.1 - 3 * C.2 + 10 = 0)

theorem find_equation_AC (A B C : (ℝ × ℝ)) (h₁ : triangleABC A B C) (h₂ : conditions A B) : 
  equationAC A C :=
sorry

end find_equation_AC_l1462_146213


namespace min_jugs_needed_to_fill_container_l1462_146299

def min_jugs_to_fill (jug_capacity container_capacity : ℕ) : ℕ :=
  Nat.ceil (container_capacity / jug_capacity)

theorem min_jugs_needed_to_fill_container :
  min_jugs_to_fill 16 200 = 13 :=
by
  -- The proof is omitted.
  sorry

end min_jugs_needed_to_fill_container_l1462_146299


namespace find_principal_amount_l1462_146266

variable (x y : ℝ)

-- conditions given in the problem
def simple_interest_condition : Prop :=
  600 = (x * y * 2) / 100

def compound_interest_condition : Prop :=
  615 = x * ((1 + y / 100)^2 - 1)

-- target statement to be proven
theorem find_principal_amount (h1 : simple_interest_condition x y) (h2 : compound_interest_condition x y) :
  x = 285.7142857 :=
  sorry

end find_principal_amount_l1462_146266


namespace number_of_license_plates_l1462_146220

-- Define the alphabet size and digit size constants.
def num_letters : ℕ := 26
def num_digits : ℕ := 10

-- Define the number of letters in the license plate.
def letters_in_plate : ℕ := 3

-- Define the number of digits in the license plate.
def digits_in_plate : ℕ := 4

-- Calculating the total number of license plates possible as (26^3) * (10^4).
theorem number_of_license_plates : 
  (num_letters ^ letters_in_plate) * (num_digits ^ digits_in_plate) = 175760000 :=
by
  sorry

end number_of_license_plates_l1462_146220


namespace academy_league_total_games_l1462_146261

theorem academy_league_total_games (teams : ℕ) (plays_each_other_twice games_non_conference : ℕ) 
  (h_teams : teams = 8)
  (h_plays_each_other_twice : plays_each_other_twice = 2 * teams * (teams - 1) / 2)
  (h_games_non_conference : games_non_conference = 6 * teams) :
  (plays_each_other_twice + games_non_conference) = 104 :=
by
  sorry

end academy_league_total_games_l1462_146261


namespace value_of_a_l1462_146226

theorem value_of_a (x : ℝ) (h : (1 - x^32) ≠ 0):
  (8 * a / (1 - x^32) = 
   2 / (1 - x) + 2 / (1 + x) + 
   4 / (1 + x^2) + 8 / (1 + x^4) + 
   16 / (1 + x^8) + 32 / (1 + x^16)) → 
  a = 8 := sorry

end value_of_a_l1462_146226


namespace celine_change_l1462_146279

theorem celine_change
  (price_laptop : ℕ)
  (price_smartphone : ℕ)
  (num_laptops : ℕ)
  (num_smartphones : ℕ)
  (total_money : ℕ)
  (h1 : price_laptop = 600)
  (h2 : price_smartphone = 400)
  (h3 : num_laptops = 2)
  (h4 : num_smartphones = 4)
  (h5 : total_money = 3000) :
  total_money - (num_laptops * price_laptop + num_smartphones * price_smartphone) = 200 :=
by
  sorry

end celine_change_l1462_146279


namespace correctStatements_l1462_146234

-- Definitions based on conditions
def isFunctionalRelationshipDeterministic (S1 : Prop) := 
  S1 = true

def isCorrelationNonDeterministic (S2 : Prop) := 
  S2 = true

def regressionAnalysisFunctionalRelation (S3 : Prop) :=
  S3 = false

def regressionAnalysisCorrelation (S4 : Prop) :=
  S4 = true

-- The translated proof problem statement
theorem correctStatements :
  ∀ (S1 S2 S3 S4 : Prop), 
    isFunctionalRelationshipDeterministic S1 →
    isCorrelationNonDeterministic S2 →
    regressionAnalysisFunctionalRelation S3 →
    regressionAnalysisCorrelation S4 →
    (S1 ∧ S2 ∧ ¬S3 ∧ S4) →
    (S1 ∧ S2 ∧ ¬S3 ∧ S4) = (true ∧ true ∧ true ∧ true) :=
by
  intros S1 S2 S3 S4 H1 H2 H3 H4 H5
  sorry

end correctStatements_l1462_146234


namespace power_sum_divisible_by_5_l1462_146289

theorem power_sum_divisible_by_5 (n : ℕ) : (2^(4*n + 1) + 3^(4*n + 1)) % 5 = 0 :=
by
  sorry

end power_sum_divisible_by_5_l1462_146289


namespace perpendicular_lines_l1462_146206

theorem perpendicular_lines (a : ℝ) : 
  ∀ x y : ℝ, 3 * y - x + 4 = 0 → 4 * y + a * x + 5 = 0 → a = 12 :=
by
  sorry

end perpendicular_lines_l1462_146206


namespace min_value_of_algebraic_sum_l1462_146259

theorem min_value_of_algebraic_sum 
  (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h : a + 3 * b = 3) :
  ∃ (min_value : ℝ), min_value = 16 / 3 ∧ (∀ a b, a > 0 → b > 0 → a + 3 * b = 3 → 1 / a + 3 / b ≥ min_value) :=
sorry

end min_value_of_algebraic_sum_l1462_146259


namespace tan_of_alpha_intersects_unit_circle_l1462_146273

theorem tan_of_alpha_intersects_unit_circle (α : ℝ) (hα : ∃ P : ℝ × ℝ, P = (12 / 13, -5 / 13) ∧ ∀ x y : ℝ, P = (x, y) → x^2 + y^2 = 1) : 
  Real.tan α = -5 / 12 :=
by
  -- proof to be completed
  sorry

end tan_of_alpha_intersects_unit_circle_l1462_146273


namespace inequality_proof_l1462_146204

noncomputable def a (x1 x2 x3 x4 x5 : ℝ) := x1 + x2 + x3 + x4 + x5
noncomputable def b (x1 x2 x3 x4 x5 : ℝ) := x1 * x2 + x1 * x3 + x1 * x4 + x1 * x5 + x2 * x3 + x2 * x4 + x2 * x5 + x3 * x4 + x3 * x5 + x4 * x5
noncomputable def c (x1 x2 x3 x4 x5 : ℝ) := x1 * x2 * x3 + x1 * x2 * x4 + x1 * x2 * x5 + x1 * x3 * x4 + x1 * x3 * x5 + x1 * x4 * x5 + x2 * x3 * x4 + x2 * x3 * x5 + x2 * x4 * x5 + x3 * x4 * x5
noncomputable def d (x1 x2 x3 x4 x5 : ℝ) := x1 * x2 * x3 * x4 + x1 * x2 * x3 * x5 + x1 * x2 * x4 * x5 + x1 * x3 * x4 * x5 + x2 * x3 * x4 * x5

theorem inequality_proof (x1 x2 x3 x4 x5 : ℝ) (hx1x2x3x4x5 : x1 * x2 * x3 * x4 * x5 = 1) :
  (1 / a x1 x2 x3 x4 x5) + (1 / b x1 x2 x3 x4 x5) + (1 / c x1 x2 x3 x4 x5) + (1 / d x1 x2 x3 x4 x5) ≤ 3 / 5 := 
sorry

end inequality_proof_l1462_146204


namespace peter_stamps_l1462_146278

theorem peter_stamps (M : ℕ) (h1 : M % 5 = 2) (h2 : M % 11 = 2) (h3 : M % 13 = 2) (h4 : M > 1) : M = 717 :=
by
  -- proof will be filled in
  sorry

end peter_stamps_l1462_146278


namespace walking_rate_on_escalator_l1462_146223

theorem walking_rate_on_escalator (v : ℝ)
  (escalator_speed : ℝ := 12)
  (escalator_length : ℝ := 196)
  (travel_time : ℝ := 14)
  (effective_speed : ℝ := v + escalator_speed)
  (distance_eq : effective_speed * travel_time = escalator_length) :
  v = 2 := by
  sorry

end walking_rate_on_escalator_l1462_146223


namespace combined_area_correct_l1462_146290

noncomputable def breadth : ℝ := 20
noncomputable def length : ℝ := 1.15 * breadth
noncomputable def area_rectangle : ℝ := 460
noncomputable def radius_semicircle : ℝ := breadth / 2
noncomputable def area_semicircle : ℝ := (1/2) * Real.pi * radius_semicircle^2
noncomputable def combined_area : ℝ := area_rectangle + area_semicircle

theorem combined_area_correct : combined_area = 460 + 50 * Real.pi :=
by
  sorry

end combined_area_correct_l1462_146290


namespace greater_of_T_N_l1462_146297

/-- Define an 8x8 board and the number of valid domino placements. -/
def N : ℕ := 12988816

/-- A combinatorial number T representing the number of ways to place 24 dominoes on an 8x8 board. -/
axiom T : ℕ 

/-- We need to prove that T is greater than -N, where N is defined as 12988816. -/
theorem greater_of_T_N : T > - (N : ℤ) := sorry

end greater_of_T_N_l1462_146297


namespace problem_equivalence_l1462_146201

theorem problem_equivalence : 4 * 4^3 - 16^60 / 16^57 = -3840 := by
  sorry

end problem_equivalence_l1462_146201


namespace find_unit_prices_and_evaluate_discount_schemes_l1462_146291

theorem find_unit_prices_and_evaluate_discount_schemes :
  ∃ (x y : ℝ),
    40 * x + 100 * y = 280 ∧
    30 * x + 200 * y = 260 ∧
    x = 6 ∧
    y = 0.4 ∧
    (∀ m : ℝ, m > 200 → 
      (50 * 6 + 0.4 * (m - 50) < 50 * 6 + 0.4 * 200 + 0.4 * 0.8 * (m - 200) ↔ m < 450) ∧
      (50 * 6 + 0.4 * (m - 50) = 50 * 6 + 0.4 * 200 + 0.4 * 0.8 * (m - 200) ↔ m = 450) ∧
      (50 * 6 + 0.4 * (m - 50) > 50 * 6 + 0.4 * 200 + 0.4 * 0.8 * (m - 200) ↔ m > 450)) :=
sorry

end find_unit_prices_and_evaluate_discount_schemes_l1462_146291


namespace adults_wearing_sunglasses_l1462_146225

def total_adults : ℕ := 2400
def one_third_of_adults (total : ℕ) : ℕ := total / 3
def women_wearing_sunglasses (women : ℕ) : ℕ := (15 * women) / 100
def men_wearing_sunglasses (men : ℕ) : ℕ := (12 * men) / 100

theorem adults_wearing_sunglasses : 
  let women := one_third_of_adults total_adults
  let men := total_adults - women
  let women_in_sunglasses := women_wearing_sunglasses women
  let men_in_sunglasses := men_wearing_sunglasses men
  women_in_sunglasses + men_in_sunglasses = 312 :=
by
  sorry

end adults_wearing_sunglasses_l1462_146225


namespace general_term_sum_formula_l1462_146277

-- Conditions for the sequence
variable (a : ℕ → ℤ)
variable (d : ℤ)
variable (S : ℕ → ℤ)

-- Given conditions
axiom a2_eq_5 : a 2 = 5
axiom S4_eq_28 : S 4 = 28

-- The sequence is an arithmetic sequence
axiom arithmetic_sequence : ∀ n : ℕ, a (n + 1) = a n + d

-- Statement 1: Proof that a_n = 4n - 3
theorem general_term (n : ℕ) : a n = 4 * n - 3 :=
by
  sorry

-- Statement 2: Proof that S_n = 2n^2 - n
theorem sum_formula (n : ℕ) : S n = 2 * n^2 - n :=
by
  sorry

end general_term_sum_formula_l1462_146277


namespace hyperbola_asymptotes_l1462_146288

-- Define the data for the hyperbola
def hyperbola_eq (x y : ℝ) : Prop := (y - 1)^2 / 16 - (x + 2)^2 / 25 = 1

-- Define the two equations for the asymptotes
def asymptote1 (x y : ℝ) : Prop := y = 4 / 5 * x + 13 / 5
def asymptote2 (x y : ℝ) : Prop := y = -4 / 5 * x + 13 / 5

-- Theorem stating that the given asymptotes are correct for the hyperbola
theorem hyperbola_asymptotes : 
  (∀ x y : ℝ, hyperbola_eq x y → (asymptote1 x y ∨ asymptote2 x y)) := 
by
  sorry

end hyperbola_asymptotes_l1462_146288


namespace common_root_iff_cond_l1462_146215

theorem common_root_iff_cond (p1 p2 q1 q2 : ℂ) :
  (∃ x : ℂ, x^2 + p1 * x + q1 = 0 ∧ x^2 + p2 * x + q2 = 0) ↔
  (q2 - q1)^2 + (p1 - p2) * (p1 * q2 - q1 * p2) = 0 :=
by
  sorry

end common_root_iff_cond_l1462_146215


namespace shaded_area_inequality_l1462_146233

theorem shaded_area_inequality 
    (A : ℝ) -- All three triangles have the same total area, A.
    {a1 a2 a3 : ℝ} -- a1, a2, a3 are the shaded areas of Triangle I, II, and III respectively.
    (h1 : a1 = A / 6) 
    (h2 : a2 = A / 2) 
    (h3 : a3 = (2 * A) / 3) : 
    a1 ≠ a2 ∧ a1 ≠ a3 ∧ a2 ≠ a3 :=
by
  -- Proof steps would go here, but they are not required as per the instructions
  sorry

end shaded_area_inequality_l1462_146233


namespace brad_amount_l1462_146275

-- Definitions for the conditions
def total_amount (j d b : ℚ) := j + d + b = 68
def josh_twice_brad (j b : ℚ) := j = 2 * b
def josh_three_fourths_doug (j d : ℚ) := j = (3 / 4) * d

-- The theorem we want to prove
theorem brad_amount : ∃ (b : ℚ), (∃ (j d : ℚ), total_amount j d b ∧ josh_twice_brad j b ∧ josh_three_fourths_doug j d) ∧ b = 12 :=
sorry

end brad_amount_l1462_146275


namespace woman_waits_for_man_l1462_146270

noncomputable def man_speed := 5 / 60 -- miles per minute
noncomputable def woman_speed := 15 / 60 -- miles per minute
noncomputable def passed_time := 2 -- minutes

noncomputable def catch_up_time (man_speed woman_speed : ℝ) (passed_time : ℝ) : ℝ :=
  (woman_speed * passed_time) / man_speed

theorem woman_waits_for_man
  (man_speed woman_speed : ℝ)
  (passed_time : ℝ)
  (h_man_speed : man_speed = 5 / 60)
  (h_woman_speed : woman_speed = 15 / 60)
  (h_passed_time : passed_time = 2) :
  catch_up_time man_speed woman_speed passed_time = 6 := 
by
  -- actual proof skipped
  sorry

end woman_waits_for_man_l1462_146270


namespace solve_for_x_l1462_146272

theorem solve_for_x : ∃ x : ℝ, 3 * x - 6 = |(-20 + 5)| ∧ x = 7 := by
  sorry

end solve_for_x_l1462_146272


namespace number_of_passed_candidates_l1462_146285

variables (P F : ℕ) (h1 : P + F = 100)
          (h2 : P * 70 + F * 20 = 100 * 50)
          (h3 : ∀ p, p = P → 70 * p = 70 * P)
          (h4 : ∀ f, f = F → 20 * f = 20 * F)

theorem number_of_passed_candidates (P F : ℕ) (h1 : P + F = 100) 
                                    (h2 : P * 70 + F * 20 = 100 * 50) 
                                    (h3 : ∀ p, p = P → 70 * p = 70 * P) 
                                    (h4 : ∀ f, f = F → 20 * f = 20 * F) : 
  P = 60 :=
sorry

end number_of_passed_candidates_l1462_146285


namespace Caitlin_age_l1462_146267

theorem Caitlin_age (Aunt_Anna_age : ℕ) (Brianna_age : ℕ) (Caitlin_age : ℕ)
    (h1 : Aunt_Anna_age = 48)
    (h2 : Brianna_age = Aunt_Anna_age / 3)
    (h3 : Caitlin_age = Brianna_age - 6) : 
    Caitlin_age = 10 := by 
  -- proof here
  sorry

end Caitlin_age_l1462_146267


namespace presidency_meeting_ways_l1462_146256

theorem presidency_meeting_ways :
  let total_schools := 4
  let members_per_school := 4
  let host_school_choices := total_schools
  let choose_3_from_4 := Nat.choose 4 3
  let choose_1_from_4 := Nat.choose 4 1
  let ways_per_host := choose_3_from_4 * choose_1_from_4 ^ 3
  let total_ways := host_school_choices * ways_per_host
  total_ways = 1024 := by
  sorry

end presidency_meeting_ways_l1462_146256


namespace find_arith_seq_params_l1462_146276

-- Define the arithmetic sequence
def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ := a + n * d

-- The conditions given in the problem
theorem find_arith_seq_params :
  ∃ a d : ℤ, 
  (arithmetic_sequence a d 8) = 5 * (arithmetic_sequence a d 1) ∧
  (arithmetic_sequence a d 12) = 2 * (arithmetic_sequence a d 5) + 5 ∧
  a = 3 ∧
  d = 4 :=
by
  sorry

end find_arith_seq_params_l1462_146276


namespace probability_of_dime_l1462_146239

noncomputable def num_quarters := 12 / 0.25
noncomputable def num_dimes := 8 / 0.10
noncomputable def num_pennies := 5 / 0.01
noncomputable def total_coins := num_quarters + num_dimes + num_pennies

theorem probability_of_dime : (num_dimes / total_coins) = (40 / 314) :=
by
  sorry

end probability_of_dime_l1462_146239


namespace expression_evaluation_l1462_146255

-- Define the numbers and operations
def expr : ℚ := 10 * (1 / 2) * 3 / (1 / 6)

-- Formalize the proof problem
theorem expression_evaluation : expr = 90 := 
by 
  -- Start the proof, which is not required according to the instruction, so we replace it with 'sorry'
  sorry

end expression_evaluation_l1462_146255


namespace sum_of_areas_of_two_squares_l1462_146282

theorem sum_of_areas_of_two_squares (a b : ℕ) (h1 : a = 8) (h2 : b = 10) :
  a * a + b * b = 164 := by
  sorry

end sum_of_areas_of_two_squares_l1462_146282


namespace find_g_of_nine_l1462_146271

noncomputable def g : ℝ → ℝ := sorry

theorem find_g_of_nine (h : ∀ x : ℝ, g (3 ^ x) + x * g (3 ^ (-x)) = x) : g 9 = 2 :=
by
  sorry

end find_g_of_nine_l1462_146271


namespace total_children_correct_l1462_146287

def blocks : ℕ := 9
def children_per_block : ℕ := 6
def total_children : ℕ := blocks * children_per_block

theorem total_children_correct : total_children = 54 := by
  sorry

end total_children_correct_l1462_146287

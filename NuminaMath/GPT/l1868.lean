import Mathlib

namespace intersection_eq_l1868_186891

-- Definitions of sets A and B
def A : Set ℤ := {0, 1}
def B : Set ℤ := {-1, 1}

-- The theorem statement
theorem intersection_eq : A ∩ B = {1} :=
by
  unfold A B
  sorry

end intersection_eq_l1868_186891


namespace waitress_tips_fraction_l1868_186806

theorem waitress_tips_fraction
  (S : ℝ) -- salary
  (T : ℝ) -- tips
  (hT : T = (11 / 4) * S) -- tips are 11/4 of salary
  (I : ℝ) -- total income
  (hI : I = S + T) -- total income is the sum of salary and tips
  : (T / I) = (11 / 15) := -- fraction of income from tips is 11/15
by
  sorry

end waitress_tips_fraction_l1868_186806


namespace mod_6_computation_l1868_186874

theorem mod_6_computation (a b n : ℕ) (h₁ : a ≡ 35 [MOD 6]) (h₂ : b ≡ 16 [MOD 6]) (h₃ : n = 1723) :
  (a ^ n - b ^ n) % 6 = 1 :=
by 
  -- proofs go here
  sorry

end mod_6_computation_l1868_186874


namespace reduced_price_per_kg_of_oil_l1868_186885

theorem reduced_price_per_kg_of_oil
  (P : ℝ)
  (h : (1000 / (0.75 * P) - 1000 / P = 5)) :
  0.75 * (1000 / 15) = 50 := 
sorry

end reduced_price_per_kg_of_oil_l1868_186885


namespace perimeter_region_l1868_186827

theorem perimeter_region (rectangle_height : ℕ) (height_eq_sixteen : rectangle_height = 16) (rect_area_eq : 12 * rectangle_height = 192) (total_area_eq : 12 * rectangle_height - 60 = 132):
  (rectangle_height + 12 + 4 + 6 + 10 * 2) = 54 :=
by
  have h1 : 12 * 16 = 192 := by sorry
  exact sorry


end perimeter_region_l1868_186827


namespace tan_addition_formula_l1868_186875

theorem tan_addition_formula (x : ℝ) (h : Real.tan x = Real.sqrt 3) : 
  Real.tan (x + Real.pi / 3) = -Real.sqrt 3 := 
by 
  sorry

end tan_addition_formula_l1868_186875


namespace num_of_three_digit_integers_greater_than_217_l1868_186843

theorem num_of_three_digit_integers_greater_than_217 : 
  ∃ n : ℕ, n = 82 ∧ ∀ x : ℕ, (217 < x ∧ x < 300) → 200 ≤ x ∧ x ≤ 299 → n = 82 := 
by
  sorry

end num_of_three_digit_integers_greater_than_217_l1868_186843


namespace books_total_l1868_186872

theorem books_total (J T : ℕ) (hJ : J = 10) (hT : T = 38) : J + T = 48 :=
by {
  sorry
}

end books_total_l1868_186872


namespace cave_depth_l1868_186838

theorem cave_depth 
  (total_depth : ℕ) 
  (remaining_depth : ℕ) 
  (h1 : total_depth = 974) 
  (h2 : remaining_depth = 386) : 
  total_depth - remaining_depth = 588 := 
by 
  sorry

end cave_depth_l1868_186838


namespace orthogonal_vectors_l1868_186869

theorem orthogonal_vectors (x : ℝ) :
  (3 * x - 4 * 6 = 0) → x = 8 :=
by
  intro h
  sorry

end orthogonal_vectors_l1868_186869


namespace inverse_prop_function_through_point_l1868_186881

theorem inverse_prop_function_through_point :
  ∀ (f : ℝ → ℝ), (∀ x, f x = k / x) → (f 1 = 2) → (f (-1) = -2) :=
by
  intros f h_inv_prop h_f1
  sorry

end inverse_prop_function_through_point_l1868_186881


namespace arithmetic_sequence_value_l1868_186877

theorem arithmetic_sequence_value 
    (a1 : ℤ) (a2 a3 a4 : ℤ) (a1_a4 : a1 = 18) 
    (b1 b2 b3 : ℤ) 
    (b1_b3 : b3 - b2 = 6 ∧ b2 - b1 = 6 ∧ b2 = 15 ∧ b3 = 21)
    (b1_a3 : a3 = b1 - 6 ∧ a4 = a1 + (a3 - 18) / 3) 
    (c1 c2 c3 c4 : ℝ) 
    (c1_b3 : c1 = a4) 
    (c2 : c2 = -14) 
    (c4 : ∃ m, c4 = b1 - m * (6 :ℝ) + - 0.5) 
    (n : ℝ) : 
    n = -12.5 := by 
  sorry

end arithmetic_sequence_value_l1868_186877


namespace find_n_150_l1868_186894

def special_sum (k n : ℕ) : ℕ := (n * (2 * k + n - 1)) / 2

theorem find_n_150 : ∃ n : ℕ, special_sum 3 n = 150 ∧ n = 15 :=
by
  sorry

end find_n_150_l1868_186894


namespace isabel_ds_games_left_l1868_186839

-- Define the initial number of DS games Isabel had
def initial_ds_games : ℕ := 90

-- Define the number of DS games Isabel gave to her friend
def ds_games_given : ℕ := 87

-- Define a function to calculate the remaining DS games
def remaining_ds_games (initial : ℕ) (given : ℕ) : ℕ := initial - given

-- Statement of the theorem we need to prove
theorem isabel_ds_games_left : remaining_ds_games initial_ds_games ds_games_given = 3 := by
  sorry

end isabel_ds_games_left_l1868_186839


namespace distance_between_trains_l1868_186841

theorem distance_between_trains (d1 d2 : ℝ) (t1 t2 : ℝ) (s1 s2 : ℝ) (x : ℝ) :
  d1 = d2 + 100 →
  s1 = 50 →
  s2 = 40 →
  d1 = s1 * t1 →
  d2 = s2 * t2 →
  t1 = t2 →
  d2 = 400 →
  d1 + d2 = 900 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end distance_between_trains_l1868_186841


namespace find_other_number_l1868_186887

-- Definitions for the given conditions
def A : ℕ := 500
def LCM : ℕ := 3000
def HCF : ℕ := 100

-- Theorem statement: If A = 500, LCM(A, B) = 3000, and HCF(A, B) = 100, then B = 600.
theorem find_other_number (B : ℕ) (h1 : A = 500) (h2 : Nat.lcm A B = 3000) (h3 : Nat.gcd A B = 100) :
  B = 600 :=
by
  sorry

end find_other_number_l1868_186887


namespace one_eighth_of_two_power_36_equals_two_power_x_l1868_186896

theorem one_eighth_of_two_power_36_equals_two_power_x (x : ℕ) :
  (1 / 8) * (2 : ℝ) ^ 36 = (2 : ℝ) ^ x → x = 33 :=
by
  intro h
  sorry

end one_eighth_of_two_power_36_equals_two_power_x_l1868_186896


namespace bottle_caps_per_box_l1868_186813

theorem bottle_caps_per_box (total_caps : ℕ) (total_boxes : ℕ) (h_total_caps : total_caps = 60) (h_total_boxes : total_boxes = 60) :
  (total_caps / total_boxes) = 1 :=
by {
  sorry
}

end bottle_caps_per_box_l1868_186813


namespace greatest_int_with_gcd_3_l1868_186854

theorem greatest_int_with_gcd_3 (n : ℕ) (h1 : n < 150) (h2 : Int.gcd n 24 = 3) : n = 141 := by
  sorry

end greatest_int_with_gcd_3_l1868_186854


namespace compute_g_neg_x_l1868_186859

noncomputable def g (x : ℝ) : ℝ := (x^2 + 3*x + 2) / (x^2 - 3*x + 2)

theorem compute_g_neg_x (x : ℝ) (h : x^2 ≠ 2) : g (-x) = 1 / g x := 
  by sorry

end compute_g_neg_x_l1868_186859


namespace g_5_l1868_186878

variable (g : ℝ → ℝ)

axiom additivity_condition : ∀ (x y : ℝ), g (x + y) = g x + g y
axiom g_1_nonzero : g 1 ≠ 0

theorem g_5 : g 5 = 5 * g 1 :=
by
  sorry

end g_5_l1868_186878


namespace quadratic_solution_set_l1868_186808

theorem quadratic_solution_set (a b c : ℝ) 
  (h : ∀ x : ℝ, ax^2 + bx + c > 0 ↔ x < -2 ∨ x > 3) :
  (a > 0) ∧ 
  (∀ x : ℝ, bx + c > 0 ↔ x < 6) = false ∧ 
  (a + b + c < 0) ∧
  (∀ x : ℝ, cx^2 - bx + a < 0 ↔ x < -1 / 3 ∨ x > 1 / 2) :=
sorry

end quadratic_solution_set_l1868_186808


namespace increasing_exponential_is_necessary_condition_l1868_186834

variable {a : ℝ}

theorem increasing_exponential_is_necessary_condition (h : ∀ x y : ℝ, x < y → a ^ x < a ^ y) :
    (a > 1) ∧ (¬ (a > 2 → a > 1)) :=
by
  sorry

end increasing_exponential_is_necessary_condition_l1868_186834


namespace side_length_of_square_with_circles_l1868_186815

noncomputable def side_length_of_square (radius : ℝ) : ℝ :=
  2 * radius + 2 * radius

theorem side_length_of_square_with_circles 
  (radius : ℝ) 
  (h_radius : radius = 2) 
  (h_tangent : ∀ (P Q : ℝ), P = Q + 2 * radius) :
  side_length_of_square radius = 8 :=
by
  sorry

end side_length_of_square_with_circles_l1868_186815


namespace grid_to_black_probability_l1868_186897

theorem grid_to_black_probability :
  let n := 16
  let p_black_after_rotation := 3 / 4
  (p_black_after_rotation ^ n) = (3 / 4) ^ 16 :=
by
  -- Proof goes here
  sorry

end grid_to_black_probability_l1868_186897


namespace electricity_usage_l1868_186811

theorem electricity_usage 
  (total_usage : ℕ) (saved_cost : ℝ) (initial_cost : ℝ) (peak_cost : ℝ) (off_peak_cost : ℝ) 
  (usage_peak : ℕ) (usage_off_peak : ℕ) :
  total_usage = 100 →
  saved_cost = 3 →
  initial_cost = 0.55 →
  peak_cost = 0.6 →
  off_peak_cost = 0.4 →
  usage_peak + usage_off_peak = total_usage →
  (total_usage * initial_cost - (peak_cost * usage_peak + off_peak_cost * usage_off_peak) = saved_cost) →
  usage_peak = 60 ∧ usage_off_peak = 40 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end electricity_usage_l1868_186811


namespace decreasing_condition_l1868_186850

noncomputable def f (a x : ℝ) : ℝ := x / (Real.log x) + a * x

theorem decreasing_condition (a : ℝ) :
  (∀ x > 1, (Real.log x - 1) / (Real.log x)^2 + a ≤ 0) → a ≤ -1/4 := by
  sorry

end decreasing_condition_l1868_186850


namespace three_digit_numbers_with_distinct_digits_avg_condition_l1868_186886

theorem three_digit_numbers_with_distinct_digits_avg_condition : 
  ∃ (S : Finset (Fin 1000)), 
  (∀ n ∈ S, (n / 100 ≠ (n / 10 % 10) ∧ (n / 100 ≠ n % 10) ∧ (n / 10 % 10 ≠ n % 10))) ∧
  (∀ n ∈ S, ((n / 100 + n % 10) / 2 = n / 10 % 10)) ∧
  (∀ n ∈ S, abs ((n / 100) - (n / 10 % 10)) ≤ 5 ∧ abs ((n / 10 % 10) - (n % 10)) ≤ 5) ∧
  S.card = 120 :=
sorry

end three_digit_numbers_with_distinct_digits_avg_condition_l1868_186886


namespace connie_tickets_l1868_186847

theorem connie_tickets (total_tickets spent_on_koala spent_on_earbuds spent_on_glow_bracelets : ℕ)
  (h1 : total_tickets = 50)
  (h2 : spent_on_koala = total_tickets / 2)
  (h3 : spent_on_earbuds = 10)
  (h4 : total_tickets = spent_on_koala + spent_on_earbuds + spent_on_glow_bracelets) :
  spent_on_glow_bracelets = 15 :=
by
  sorry

end connie_tickets_l1868_186847


namespace arithmetic_sequence_sum_l1868_186889

noncomputable def a_n (a1 d : ℕ) (n : ℕ) : ℕ := a1 + (n - 1) * d

theorem arithmetic_sequence_sum 
  {a1 d : ℕ} (h_pos_d : d > 0) 
  (h_sum : a1 + (a1 + d) + (a1 + 2 * d) = 15) 
  (h_prod : a1 * (a1 + d) * (a1 + 2 * d) = 80) 
  : a_n a1 d 11 + a_n a1 d 12 + a_n a1 d 13 = 105 :=
sorry

end arithmetic_sequence_sum_l1868_186889


namespace amount_pop_spend_l1868_186807

theorem amount_pop_spend
  (total_spent : ℝ)
  (ratio_snap_crackle : ℝ)
  (ratio_crackle_pop : ℝ)
  (spending_eq : total_spent = 150)
  (snap_crackle : ratio_snap_crackle = 2)
  (crackle_pop : ratio_crackle_pop = 3)
  (snap : ℝ)
  (crackle : ℝ)
  (pop : ℝ)
  (snap_eq : snap = ratio_snap_crackle * crackle)
  (crackle_eq : crackle = ratio_crackle_pop * pop)
  (total_eq : snap + crackle + pop = total_spent) :
  pop = 15 := 
by
  sorry

end amount_pop_spend_l1868_186807


namespace handshakesCountIsCorrect_l1868_186873

-- Define the number of gremlins and imps
def numGremlins : ℕ := 30
def numImps : ℕ := 20

-- Define the conditions based on the problem
def handshakesAmongGremlins : ℕ := (numGremlins * (numGremlins - 1)) / 2
def handshakesBetweenImpsAndGremlins : ℕ := numImps * numGremlins

-- Calculate the total handshakes
def totalHandshakes : ℕ := handshakesAmongGremlins + handshakesBetweenImpsAndGremlins

-- Prove that the total number of handshakes equals 1035
theorem handshakesCountIsCorrect : totalHandshakes = 1035 := by
  sorry

end handshakesCountIsCorrect_l1868_186873


namespace sum_even_odd_probability_l1868_186884

theorem sum_even_odd_probability :
  (∀ (a b : ℕ), ∃ (P_even P_odd : ℚ),
    P_even = 1/2 ∧ P_odd = 1/2 ∧
    (a % 2 = 0 ∧ b % 2 = 0 ↔ (a + b) % 2 = 0) ∧
    (a % 2 = 1 ∧ b % 2 = 1 ↔ (a + b) % 2 = 0) ∧
    ((a % 2 = 0 ∧ b % 2 = 1) ∨ (a % 2 = 1 ∧ b % 2 = 0) ↔ (a + b) % 2 = 1)) :=
sorry

end sum_even_odd_probability_l1868_186884


namespace fraction_a_over_d_l1868_186876

-- Defining the given conditions as hypotheses
variables (a b c d : ℚ)

-- Conditions
axiom h1 : a / b = 20
axiom h2 : c / b = 5
axiom h3 : c / d = 1 / 15

-- Goal to prove
theorem fraction_a_over_d : a / d = 4 / 15 :=
by
  sorry

end fraction_a_over_d_l1868_186876


namespace find_gross_salary_l1868_186851

open Real

noncomputable def bill_take_home_salary : ℝ := 40000
noncomputable def property_tax : ℝ := 2000
noncomputable def sales_tax : ℝ := 3000
noncomputable def income_tax_rate : ℝ := 0.10

theorem find_gross_salary (gross_salary : ℝ) :
  bill_take_home_salary = gross_salary - (income_tax_rate * gross_salary + property_tax + sales_tax) →
  gross_salary = 50000 :=
by
  sorry

end find_gross_salary_l1868_186851


namespace skew_lines_sufficient_not_necessary_l1868_186899

-- Definitions for the conditions
def skew_lines (l1 l2 : Type) : Prop := sorry -- Definition of skew lines
def do_not_intersect (l1 l2 : Type) : Prop := sorry -- Definition of not intersecting

-- The main theorem statement
theorem skew_lines_sufficient_not_necessary (l1 l2 : Type) :
  (skew_lines l1 l2) → (do_not_intersect l1 l2) ∧ ¬ (do_not_intersect l1 l2 → skew_lines l1 l2) :=
by
  sorry

end skew_lines_sufficient_not_necessary_l1868_186899


namespace find_a1_a7_l1868_186802

variable {a n : ℕ → ℝ}
variable {d : ℝ}

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ k n, a (k + n) = a k + n * d

theorem find_a1_a7 
  (a1 : ℝ) (d : ℝ)
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a d)
  (h1 : a 3 + a 5 = 14)
  (h2 : a 2 * a 6 = 33) :
  a 1 * a 7 = 13 := 
sorry

end find_a1_a7_l1868_186802


namespace journey_total_time_l1868_186868

noncomputable def total_time (D : ℝ) (r_dist : ℕ → ℕ) (r_time : ℕ → ℕ) (u_speed : ℝ) : ℝ :=
  let dist_uphill := D * (r_dist 1) / (r_dist 1 + r_dist 2 + r_dist 3)
  let t_uphill := (dist_uphill / u_speed)
  let k := t_uphill / (r_time 1)
  (r_time 1 + r_time 2 + r_time 3) * k

theorem journey_total_time :
  total_time 50 (fun n => if n = 1 then 1 else if n = 2 then 2 else 3) 
                (fun n => if n = 1 then 4 else if n = 2 then 5 else 6) 
                3 = 10 + 5/12 :=
by
  sorry

end journey_total_time_l1868_186868


namespace rational_number_theorem_l1868_186824

theorem rational_number_theorem (x y : ℚ) 
  (h1 : |(x + 2017 : ℚ)| + (y - 2017) ^ 2 = 0) : 
  (x / y) ^ 2017 = -1 := 
by
  sorry

end rational_number_theorem_l1868_186824


namespace numbers_not_perfect_squares_or_cubes_l1868_186863

theorem numbers_not_perfect_squares_or_cubes : 
  let total_numbers := 200
  let perfect_squares := 14
  let perfect_cubes := 5
  let sixth_powers := 1
  total_numbers - (perfect_squares + perfect_cubes - sixth_powers) = 182 :=
by
  sorry

end numbers_not_perfect_squares_or_cubes_l1868_186863


namespace min_value_of_expr_l1868_186829

def expr (x y : ℝ) : ℝ := 3 * x^2 + 4 * x * y + 2 * y^2 - 6 * x + 8 * y + 10

theorem min_value_of_expr : ∃ x y : ℝ, expr x y = -2 / 3 :=
by
  sorry

end min_value_of_expr_l1868_186829


namespace g6_eq_16_l1868_186831

-- Definition of the function g that satisfies the given conditions
variable (g : ℝ → ℝ)

-- Given conditions
axiom functional_eq : ∀ x y : ℝ, g (x + y) = g x * g y
axiom g3_eq_4 : g 3 = 4

-- The goal is to prove g(6) = 16
theorem g6_eq_16 : g 6 = 16 := by
  sorry

end g6_eq_16_l1868_186831


namespace smallest_r_l1868_186810

theorem smallest_r {p q r : ℕ} (h1 : p < q) (h2 : q < r) (h3 : 2 * q = p + r) (h4 : r * r = p * q) : r = 5 :=
sorry

end smallest_r_l1868_186810


namespace substitution_modulo_l1868_186879

-- Definitions based on conditions
def total_players := 15
def starting_lineup := 10
def substitutes := 5
def max_substitutions := 2

-- Define the number of substitutions ways for the cases 0, 1, and 2 substitutions
def a_0 := 1
def a_1 := starting_lineup * substitutes
def a_2 := starting_lineup * substitutes * (starting_lineup - 1) * (substitutes - 1)

-- Summing the total number of substitution scenarios
def total_substitution_scenarios := a_0 + a_1 + a_2

-- Theorem statement to verify the result modulo 500
theorem substitution_modulo : total_substitution_scenarios % 500 = 351 := by
  sorry

end substitution_modulo_l1868_186879


namespace product_of_two_numbers_eq_a_mul_100_a_l1868_186853

def product_of_two_numbers (a : ℝ) (b : ℝ) : ℝ := a * b

theorem product_of_two_numbers_eq_a_mul_100_a (a : ℝ) (b : ℝ) (h : a + b = 100) :
    product_of_two_numbers a b = a * (100 - a) :=
by
  sorry

end product_of_two_numbers_eq_a_mul_100_a_l1868_186853


namespace total_cost_sandwiches_and_sodas_l1868_186822

theorem total_cost_sandwiches_and_sodas :
  let price_sandwich : Real := 2.49
  let price_soda : Real := 1.87
  let quantity_sandwich : ℕ := 2
  let quantity_soda : ℕ := 4
  (quantity_sandwich * price_sandwich + quantity_soda * price_soda) = 12.46 := 
by
  sorry

end total_cost_sandwiches_and_sodas_l1868_186822


namespace sum_mod_condition_l1868_186826

theorem sum_mod_condition (a b c : ℤ) (h1 : a * b * c % 7 = 2)
                          (h2 : 3 * c % 7 = 1)
                          (h3 : 4 * b % 7 = (2 + b) % 7) :
                          (a + b + c) % 7 = 3 := by
  sorry

end sum_mod_condition_l1868_186826


namespace find_s_l1868_186835

-- Define the roots of the quadratic equation
variables (a b n r s : ℝ)

-- Conditions from Vieta's formulas
def condition1 : Prop := a + b = n
def condition2 : Prop := a * b = 3

-- Roots of the second quadratic equation
def condition3 : Prop := (a + 1 / b) * (b + 1 / a) = s

-- The theorem statement
theorem find_s
  (h1 : condition1 a b n)
  (h2 : condition2 a b)
  (h3 : condition3 a b s) :
  s = 16 / 3 :=
by
  sorry

end find_s_l1868_186835


namespace union_of_sets_l1868_186893

-- Defining the sets A and B
def A : Set ℕ := {1, 3, 6}
def B : Set ℕ := {1, 2}

-- The theorem we want to prove
theorem union_of_sets : A ∪ B = {1, 2, 3, 6} := by
  sorry

end union_of_sets_l1868_186893


namespace number_of_solutions_in_positive_integers_l1868_186821

theorem number_of_solutions_in_positive_integers (x y : ℕ) (h1 : 3 * x + 4 * y = 806) : 
  ∃ n : ℕ, n = 67 := 
sorry

end number_of_solutions_in_positive_integers_l1868_186821


namespace black_region_area_is_correct_l1868_186858

noncomputable def area_of_black_region : ℕ :=
  let area_large_square := 10 * 10
  let area_first_smaller_square := 4 * 4
  let area_second_smaller_square := 2 * 2
  area_large_square - (area_first_smaller_square + area_second_smaller_square)

theorem black_region_area_is_correct :
  area_of_black_region = 80 :=
by
  sorry

end black_region_area_is_correct_l1868_186858


namespace find_radius_l1868_186817

-- Definitions and conditions
variables (M N r : ℝ) (h1 : M = π * r^2) (h2 : N = 2 * π * r) (h3 : M / N = 25)

-- Theorem statement
theorem find_radius : r = 50 :=
sorry

end find_radius_l1868_186817


namespace longest_side_of_similar_triangle_l1868_186849

-- Define the sides of the original triangle
def a : ℕ := 8
def b : ℕ := 10
def c : ℕ := 12

-- Define the perimeter of the similar triangle
def perimeter_similar_triangle : ℕ := 150

-- Formalize the problem using Lean statement
theorem longest_side_of_similar_triangle :
  ∃ x : ℕ, 8 * x + 10 * x + 12 * x = 150 ∧ 12 * x = 60 :=
by
  sorry

end longest_side_of_similar_triangle_l1868_186849


namespace ordered_pair_and_sum_of_squares_l1868_186882

theorem ordered_pair_and_sum_of_squares :
  ∃ x y : ℚ, 
    6 * x - 48 * y = 2 ∧ 
    3 * y - x = 4 ∧ 
    x ^ 2 + y ^ 2 = 442 / 25 :=
by
  sorry

end ordered_pair_and_sum_of_squares_l1868_186882


namespace problem1_problem2_problem3_problem4_l1868_186844

variable (f : ℝ → ℝ)
variables (H1 : f (-1) = 2) 
          (H2 : ∀ x, x < 0 → f x > 1)
          (H3 : ∀ x y, f (x + y) = f x * f y)

-- (1) Prove f(0) = 1
theorem problem1 : f 0 = 1 := sorry

-- (2) Prove f(-4) = 16
theorem problem2 : f (-4) = 16 := sorry

-- (3) Prove f(x) is strictly decreasing
theorem problem3 : ∀ x y, x < y → f x > f y := sorry

-- (4) Solve f(-4x^2)f(10x) ≥ 1/16
theorem problem4 : { x : ℝ | f (-4 * x ^ 2) * f (10 * x) ≥ 1 / 16 } = { x | x ≤ 1 / 2 ∨ 2 ≤ x } := sorry

end problem1_problem2_problem3_problem4_l1868_186844


namespace cost_per_piece_l1868_186832

variable (totalCost : ℝ) (numberOfPizzas : ℝ) (piecesPerPizza : ℝ)

theorem cost_per_piece (h1 : totalCost = 80) (h2 : numberOfPizzas = 4) (h3 : piecesPerPizza = 5) :
  totalCost / numberOfPizzas / piecesPerPizza = 4 := by
sorry

end cost_per_piece_l1868_186832


namespace reflection_line_coordinates_sum_l1868_186890

theorem reflection_line_coordinates_sum (m b : ℝ)
  (h : ∀ (x y x' y' : ℝ), (x, y) = (-4, 2) → (x', y') = (2, 6) → 
  ∃ (m b : ℝ), y = m * x + b ∧ y' = m * x' + b ∧ ∀ (p q : ℝ), 
  (p, q) = ((x+x')/2, (y+y')/2) → p = ((-4 + 2)/2) ∧ q = ((2 + 6)/2)) :
  m + b = 1 :=
by
  sorry

end reflection_line_coordinates_sum_l1868_186890


namespace proof_equivalent_problem_l1868_186836

variables (a b c : ℝ)
-- Conditions
axiom h1 : a < b
axiom h2 : b < 0
axiom h3 : c > 0

theorem proof_equivalent_problem :
  (a * c < b * c) ∧ (a + b + c < b + c) ∧ (c / a > 1) :=
by
  sorry

end proof_equivalent_problem_l1868_186836


namespace complement_union_correct_l1868_186845

-- Defining the sets U, A, and B
def U : Set ℕ := {1, 2, 3, 4}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3}

-- Theorem statement
theorem complement_union_correct : (U \ A) ∪ B = {2, 3, 4} := by
  sorry

end complement_union_correct_l1868_186845


namespace Jackson_game_time_l1868_186848

/-- Jackson's grade increases by 15 points for every hour he spends studying, 
    and his grade is 45 points, prove that he spends 9 hours playing video 
    games when he spends 3 hours studying and 1/3 of his study time on 
    playing video games. -/
theorem Jackson_game_time (S G : ℕ) (h1 : 15 * S = 45) (h2 : G = 3 * S) : G = 9 :=
by
  sorry

end Jackson_game_time_l1868_186848


namespace range_of_a_l1868_186833

theorem range_of_a (a : ℝ) : (∃ x₀ : ℝ, x₀^2 + (a-1)*x₀ + 1 < 0) ↔ (a < -1 ∨ a > 3) :=
by
  sorry

end range_of_a_l1868_186833


namespace field_area_l1868_186846

theorem field_area (x y : ℕ) (h1 : x + y = 700) (h2 : y - x = (1/5) * ((x + y) / 2)) : x = 315 :=
  sorry

end field_area_l1868_186846


namespace ana_wins_l1868_186803

-- Define the game conditions and state
def game_conditions (n : ℕ) (m : ℕ) : Prop :=
  n < m ∧ m < n^2 ∧ Nat.gcd n m = 1

-- Define the losing condition
def losing_condition (n : ℕ) : Prop :=
  n >= 2016

-- Define the predicate for Ana having a winning strategy
def ana_winning_strategy : Prop :=
  ∃ (strategy : ℕ → ℕ), strategy 3 = 5 ∧
  (∀ n, (¬ losing_condition n) → (losing_condition (strategy n)))

theorem ana_wins : ana_winning_strategy :=
  sorry

end ana_wins_l1868_186803


namespace mila_needs_48_hours_to_earn_as_much_as_agnes_l1868_186856

/-- Definition of the hourly wage for the babysitters and the working hours of Agnes. -/
def mila_hourly_wage : ℝ := 10
def agnes_hourly_wage : ℝ := 15
def agnes_weekly_hours : ℝ := 8
def weeks_in_month : ℝ := 4

/-- Mila needs to work 48 hours in a month to earn as much as Agnes. -/
theorem mila_needs_48_hours_to_earn_as_much_as_agnes :
  ∃ (mila_monthly_hours : ℝ), mila_monthly_hours = 48 ∧ 
  mila_hourly_wage * mila_monthly_hours = agnes_hourly_wage * agnes_weekly_hours * weeks_in_month := 
sorry

end mila_needs_48_hours_to_earn_as_much_as_agnes_l1868_186856


namespace find_q_l1868_186800

theorem find_q (q: ℕ) (h: 81^10 = 3^q) : q = 40 :=
by
  sorry

end find_q_l1868_186800


namespace price_equation_l1868_186805

variable (x : ℝ)

def first_discount (x : ℝ) : ℝ := x - 5

def second_discount (price_after_first_discount : ℝ) : ℝ := 0.8 * price_after_first_discount

theorem price_equation
  (hx : second_discount (first_discount x) = 60) :
  0.8 * (x - 5) = 60 := by
  sorry

end price_equation_l1868_186805


namespace prob_of_nine_correct_is_zero_l1868_186818

-- Define the necessary components and properties of the problem
def is_correct_placement (letter: ℕ) (envelope: ℕ) : Prop := letter = envelope

def is_random_distribution (letters : Fin 10 → Fin 10) : Prop := true

-- State the theorem formally
theorem prob_of_nine_correct_is_zero (f : Fin 10 → Fin 10) :
  is_random_distribution f →
  (∃ (count : ℕ), count = 9 ∧ (∀ i : Fin 10, is_correct_placement i (f i) ↔ i = count)) → false :=
by
  sorry

end prob_of_nine_correct_is_zero_l1868_186818


namespace harvest_bushels_l1868_186867

def num_rows : ℕ := 5
def stalks_per_row : ℕ := 80
def stalks_per_bushel : ℕ := 8

theorem harvest_bushels : (num_rows * stalks_per_row) / stalks_per_bushel = 50 := by
  sorry

end harvest_bushels_l1868_186867


namespace sum_of_coefficients_l1868_186865

theorem sum_of_coefficients (a : ℕ → ℤ) (x : ℂ) :
  (2*x - 1)^10 = a 0 + a 1 * x + a 2 * x^2 + a 3 * x^3 + a 4 * x^4 + 
  a 5 * x^5 + a 6 * x^6 + a 7 * x^7 + a 8 * x^8 + a 9 * x^9 + a 10 * x^10 →
  a 0 = 1 →
  a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 = 20 :=
sorry

end sum_of_coefficients_l1868_186865


namespace find_numbers_l1868_186861

theorem find_numbers (a b : ℝ) (h₁ : a - b = 157) (h₂ : a / b = 2) : a = 314 ∧ b = 157 :=
sorry

end find_numbers_l1868_186861


namespace find_number_l1868_186837

def sum := 555 + 445
def difference := 555 - 445
def quotient := 2 * difference
def remainder := 30
def N : ℕ := 220030

theorem find_number (N : ℕ) : 
  N = sum * quotient + remainder :=
  by
    sorry

end find_number_l1868_186837


namespace complex_number_z_value_l1868_186840

open Complex

theorem complex_number_z_value :
  ∀ (i z : ℂ), i^2 = -1 ∧ z * (1 + i) = 2 * i^2018 → z = -1 + i :=
by
  intros i z h
  have h1 : i^2 = -1 := h.1
  have h2 : z * (1 + i) = 2 * i^2018 := h.2
  sorry

end complex_number_z_value_l1868_186840


namespace largest_prime_factor_of_18_pow_3_plus_15_pow_4_minus_3_pow_7_is_19_l1868_186880

theorem largest_prime_factor_of_18_pow_3_plus_15_pow_4_minus_3_pow_7_is_19 : 
  ∃ p : ℕ, Prime p ∧ p = 19 ∧ ∀ q : ℕ, Prime q → q ∣ (18^3 + 15^4 - 3^7) → q ≤ 19 :=
sorry

end largest_prime_factor_of_18_pow_3_plus_15_pow_4_minus_3_pow_7_is_19_l1868_186880


namespace positive_integer_solution_l1868_186812

theorem positive_integer_solution (n x y : ℕ) (hn : 0 < n) (hx : 0 < x) (hy : 0 < y) :
  y ^ 2 + x * y + 3 * x = n * (x ^ 2 + x * y + 3 * y) → n = 1 :=
sorry

end positive_integer_solution_l1868_186812


namespace ratio_of_p_to_r_l1868_186898

theorem ratio_of_p_to_r (p q r s : ℚ) 
  (h1 : p / q = 5 / 4) 
  (h2 : r / s = 4 / 3) 
  (h3 : s / q = 1 / 8) : 
  p / r = 15 / 2 := 
by 
  sorry

end ratio_of_p_to_r_l1868_186898


namespace hyperbola_condition_l1868_186809

-- Definitions and hypotheses
def is_hyperbola (m n : ℝ) (x y : ℝ) : Prop := m * x^2 - n * y^2 = 1

-- Statement of the problem
theorem hyperbola_condition (m n : ℝ) : (∃ x y : ℝ, is_hyperbola m n x y) ↔ m * n > 0 :=
by sorry

end hyperbola_condition_l1868_186809


namespace cloth_total_selling_price_l1868_186855

theorem cloth_total_selling_price
    (meters : ℕ) (profit_per_meter cost_price_per_meter : ℝ) :
    meters = 92 →
    profit_per_meter = 24 →
    cost_price_per_meter = 83.5 →
    (cost_price_per_meter + profit_per_meter) * meters = 9890 :=
by
  intros
  sorry

end cloth_total_selling_price_l1868_186855


namespace positive_rational_solutions_condition_l1868_186883

-- Definitions used in Lean 4 statement corresponding to conditions in the problem.
variable (a b : ℚ)

-- Lean Statement encapsulating the mathematical proof problem.
theorem positive_rational_solutions_condition :
  ∃ x y : ℚ, x > 0 ∧ y > 0 ∧ x * y = a ∧ x + y = b ↔ (∃ k : ℚ, k^2 = b^2 - 4 * a ∧ k > 0) :=
by
  sorry

end positive_rational_solutions_condition_l1868_186883


namespace statement_is_true_l1868_186819

theorem statement_is_true (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) 
  (h : ∀ x : ℝ, |x + 2| < b → |(3 * x + 2) + 4| < a) : b ≤ a / 3 :=
by
  sorry

end statement_is_true_l1868_186819


namespace sum_fourth_powers_l1868_186862

theorem sum_fourth_powers (a b c : ℝ) 
  (h1 : a + b + c = 1)
  (h2 : a^2 + b^2 + c^2 = 2)
  (h3 : a^3 + b^3 + c^3 = 3) : 
  a^4 + b^4 + c^4 = 25 / 6 :=
by sorry

end sum_fourth_powers_l1868_186862


namespace problem_l1868_186830

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
noncomputable def m : ℝ := sorry
noncomputable def p : ℝ := sorry
noncomputable def r : ℝ := sorry

theorem problem
  (h1 : a^2 - m*a + 3 = 0)
  (h2 : b^2 - m*b + 3 = 0)
  (h3 : a * b = 3)
  (h4 : ∀ x, x^2 - p * x + r = (x - (a + 1 / b)) * (x - (b + 1 / a))) :
  r = 16 / 3 :=
sorry

end problem_l1868_186830


namespace factorial_square_product_l1868_186895

theorem factorial_square_product : (Real.sqrt (Nat.factorial 6 * Nat.factorial 4)) ^ 2 = 17280 := by
  sorry

end factorial_square_product_l1868_186895


namespace annual_subscription_cost_l1868_186825

theorem annual_subscription_cost :
  (10 * 12) * (1 - 0.2) = 96 :=
by
  sorry

end annual_subscription_cost_l1868_186825


namespace number_of_truthful_people_l1868_186804

-- Definitions from conditions
def people := Fin 100
def tells_truth (p : people) : Prop := sorry -- Placeholder definition.

-- Conditions
axiom c1 : ∃ p : people, ¬ tells_truth p
axiom c2 : ∀ p1 p2 : people, p1 ≠ p2 → (tells_truth p1 ∨ tells_truth p2)

-- Goal
theorem number_of_truthful_people : 
  ∃ S : Finset people, S.card = 99 ∧ (∀ p ∈ S, tells_truth p) :=
sorry

end number_of_truthful_people_l1868_186804


namespace a5_b5_equals_c5_d5_cannot_conclude_a4_b4_equals_c4_d4_l1868_186814

-- Definitions of assumptions
variables (a b c d : ℝ)
axiom h1 : a + b = c + d
axiom h2 : a^3 + b^3 = c^3 + d^3

-- Statement for part (a)
theorem a5_b5_equals_c5_d5 : a^5 + b^5 = c^5 + d^5 :=
by sorry

-- Statement for part (b), we need to state that we cannot conclude a^4 + b^4 = c^4 + d^4 under given conditions
theorem cannot_conclude_a4_b4_equals_c4_d4 : ¬ (a^4 + b^4 = c^4 + d^4) :=
by sorry

end a5_b5_equals_c5_d5_cannot_conclude_a4_b4_equals_c4_d4_l1868_186814


namespace triangle_perimeter_l1868_186820

-- Definitions for the conditions
def inscribed_circle_of_triangle_tangent_at (radius : ℝ) (DP : ℝ) (PE : ℝ) : Prop :=
  radius = 27 ∧ DP = 29 ∧ PE = 33

-- Perimeter calculation theorem
theorem triangle_perimeter (r DP PE : ℝ) (h : inscribed_circle_of_triangle_tangent_at r DP PE) : 
  ∃ perimeter : ℝ, perimeter = 774 :=
by
  sorry

end triangle_perimeter_l1868_186820


namespace ratio_of_times_l1868_186871

theorem ratio_of_times (A_work_time B_combined_rate : ℕ) 
  (h1 : A_work_time = 6) 
  (h2 : (1 / (1 / A_work_time + 1 / (B_combined_rate / 2))) = 2) :
  (B_combined_rate : ℝ) / A_work_time = 1 / 2 :=
by
  -- below we add the proof part which we will skip for now with sorry.
  sorry

end ratio_of_times_l1868_186871


namespace intersection_A_B_at_3_range_of_a_l1868_186823

open Set

-- Definitions from the condition
def A (x : ℝ) : Prop := abs x ≥ 2
def B (x a : ℝ) : Prop := (x - 2 * a) * (x + 3) < 0

-- Part (Ⅰ)
theorem intersection_A_B_at_3 :
  let a := 3
  let A := {x : ℝ | abs x ≥ 2}
  let B := {x : ℝ | (x - 6) * (x + 3) < 0}
  {x : ℝ | A x} ∩ {x : ℝ | B x} = {x : ℝ | (-3 < x ∧ x ≤ -2) ∨ (2 ≤ x ∧ x < 6)} :=
by
  sorry

-- Part (Ⅱ)
theorem range_of_a (a : ℝ) (h : a > 0) :
  (∀ x : ℝ, A x ∨ B x a) → a ≥ 1 :=
by
  sorry

end intersection_A_B_at_3_range_of_a_l1868_186823


namespace curve_intersects_at_point_2_3_l1868_186816

open Real

theorem curve_intersects_at_point_2_3 :
  ∃ (t₁ t₂ : ℝ), t₁ ≠ t₂ ∧ 
                 (t₁^2 - 4 = t₂^2 - 4) ∧ 
                 (t₁^3 - 6 * t₁ + 3 = t₂^3 - 6 * t₂ + 3) ∧ 
                 (t₁^2 - 4 = 2) ∧ 
                 (t₁^3 - 6 * t₁ + 3 = 3) :=
by
  sorry

end curve_intersects_at_point_2_3_l1868_186816


namespace discount_price_l1868_186888

theorem discount_price (original_price : ℝ) (discount_rate : ℝ) (current_price : ℝ) 
  (h1 : original_price = 120) 
  (h2 : discount_rate = 0.8) 
  (h3 : current_price = original_price * discount_rate) : 
  current_price = 96 := 
by
  sorry

end discount_price_l1868_186888


namespace greatest_odd_integer_x_l1868_186870

theorem greatest_odd_integer_x (x : ℕ) (h1 : x % 2 = 1) (h2 : x^4 / x^2 < 50) : x ≤ 7 :=
sorry

end greatest_odd_integer_x_l1868_186870


namespace wheel_speed_l1868_186860

def original_circumference_in_miles := 10 / 5280
def time_factor := 3600
def new_time_factor := 3600 - (1/3)

theorem wheel_speed
  (r : ℝ) 
  (original_speed : r * time_factor = original_circumference_in_miles * 3600)
  (new_speed : (r + 5) * (time_factor - 1/10800) = original_circumference_in_miles * 3600) :
  r = 10 :=
sorry

end wheel_speed_l1868_186860


namespace a4_value_l1868_186866

variable (a : ℕ → ℤ)
variable (S : ℕ → ℤ)

-- Condition: The sum of the first n terms of the sequence {a_n} is S_n = n^2 - 1
axiom sum_of_sequence (n : ℕ) : S n = n^2 - 1

-- We need to prove that a_4 = 7
theorem a4_value : a 4 = S 4 - S 3 :=
by 
  -- Proof goes here
  sorry

end a4_value_l1868_186866


namespace integer_solutions_of_equation_l1868_186801

theorem integer_solutions_of_equation :
  ∀ x y : ℤ, 2 * x^3 + x * y - 7 = 0 ↔ (x = -7 ∧ y = -99) ∨ (x = -1 ∧ y = -9) ∨ (x = 1 ∧ y = 5) ∨ (x = 7 ∧ y = -97) := by 
  sorry

end integer_solutions_of_equation_l1868_186801


namespace simplify_evaluate_expression_l1868_186842

noncomputable def a : ℝ := 2 * Real.cos (60 * Real.pi / 180) + 1

theorem simplify_evaluate_expression : (a - (a^2) / (a + 1)) / ((a^2) / ((a^2) - 1)) = 1 / 2 :=
by sorry

end simplify_evaluate_expression_l1868_186842


namespace necessary_but_not_sufficient_l1868_186852

-- Definitions from conditions
def abs_gt_2 (x : ℝ) : Prop := |x| > 2
def x_lt_neg_2 (x : ℝ) : Prop := x < -2

-- Statement to prove
theorem necessary_but_not_sufficient : 
  ∀ x : ℝ, (abs_gt_2 x → x_lt_neg_2 x) ∧ (¬(x_lt_neg_2 x → abs_gt_2 x)) := 
by 
  sorry

end necessary_but_not_sufficient_l1868_186852


namespace z_is_200_percent_of_x_l1868_186857

theorem z_is_200_percent_of_x
  (x y z : ℝ)
  (h1 : 0.45 * z = 1.20 * y)
  (h2 : y = 0.75 * x) :
  z = 2 * x :=
sorry

end z_is_200_percent_of_x_l1868_186857


namespace faye_candies_final_count_l1868_186864

def initialCandies : ℕ := 47
def candiesEaten : ℕ := 25
def candiesReceived : ℕ := 40

theorem faye_candies_final_count : (initialCandies - candiesEaten + candiesReceived) = 62 :=
by
  sorry

end faye_candies_final_count_l1868_186864


namespace sandwiches_with_ten_loaves_l1868_186828

def sandwiches_per_loaf : ℕ := 18 / 3

def num_sandwiches (loaves: ℕ) : ℕ := sandwiches_per_loaf * loaves

theorem sandwiches_with_ten_loaves :
  num_sandwiches 10 = 60 := by
  sorry

end sandwiches_with_ten_loaves_l1868_186828


namespace find_y_l1868_186892

theorem find_y (x y : ℕ) (h1 : x > 0 ∧ y > 0) (h2 : x % y = 9) (h3 : (x:ℝ) / (y:ℝ) = 96.45) : y = 20 :=
by
  sorry

end find_y_l1868_186892

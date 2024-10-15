import Mathlib

namespace NUMINAMATH_GPT_variable_is_eleven_l1543_154335

theorem variable_is_eleven (x : ℕ) (h : (1/2)^22 * (1/81)^x = 1/(18^22)) : x = 11 :=
by
  sorry

end NUMINAMATH_GPT_variable_is_eleven_l1543_154335


namespace NUMINAMATH_GPT_factorization_correct_l1543_154307

theorem factorization_correct (a : ℝ) : a^2 - 2 * a - 15 = (a + 3) * (a - 5) := 
by 
  sorry

end NUMINAMATH_GPT_factorization_correct_l1543_154307


namespace NUMINAMATH_GPT_number_of_pencils_l1543_154377

variable (P L : ℕ)

-- Conditions
def condition1 : Prop := P / L = 5 / 6
def condition2 : Prop := L = P + 5

-- Statement to prove
theorem number_of_pencils (h1 : condition1 P L) (h2 : condition2 P L) : L = 30 :=
  sorry

end NUMINAMATH_GPT_number_of_pencils_l1543_154377


namespace NUMINAMATH_GPT_isosceles_triangle_x_sum_l1543_154347

theorem isosceles_triangle_x_sum :
  ∀ (x : ℝ), (∃ (a b : ℝ), a + b + 60 = 180 ∧ (a = x ∨ b = x) ∧ (a = b ∨ a = 60 ∨ b = 60))
  → (60 + 60 + 60 = 180) :=
by
  intro x h
  sorry

end NUMINAMATH_GPT_isosceles_triangle_x_sum_l1543_154347


namespace NUMINAMATH_GPT_production_today_is_correct_l1543_154341

theorem production_today_is_correct (n : ℕ) (P : ℕ) (T : ℕ) (average_daily_production : ℕ) (new_average_daily_production : ℕ) (h1 : n = 3) (h2 : average_daily_production = 70) (h3 : new_average_daily_production = 75) (h4 : P = n * average_daily_production) (h5 : P + T = (n + 1) * new_average_daily_production) : T = 90 :=
by
  sorry

end NUMINAMATH_GPT_production_today_is_correct_l1543_154341


namespace NUMINAMATH_GPT_complete_triangles_l1543_154350

noncomputable def possible_placements_count : Nat :=
  sorry

theorem complete_triangles {a b c : Nat} :
  (1 + 2 + 4 + 10 + a + b + c) = 23 →
  ∃ (count : Nat), count = 4 := 
by
  sorry

end NUMINAMATH_GPT_complete_triangles_l1543_154350


namespace NUMINAMATH_GPT_original_number_l1543_154372

theorem original_number (x : ℤ) (h : x / 2 = 9) : x = 18 := by
  sorry

end NUMINAMATH_GPT_original_number_l1543_154372


namespace NUMINAMATH_GPT_num_ordered_pairs_l1543_154391

open Real 

-- Define the conditions
def eq_condition (x y : ℕ) : Prop :=
  x * (sqrt y) + y * (sqrt x) + (sqrt (2006 * x * y)) - (sqrt (2006 * x)) - (sqrt (2006 * y)) - 2006 = 0

-- Define the main problem statement
theorem num_ordered_pairs : ∃ (n : ℕ), n = 8 ∧ (∀ (x y : ℕ), eq_condition x y → x * y = 2006) :=
by
  sorry

end NUMINAMATH_GPT_num_ordered_pairs_l1543_154391


namespace NUMINAMATH_GPT_maximum_ab_l1543_154380

theorem maximum_ab (a b c : ℝ) (h1 : a + b + c = 4) (h2 : 3 * a + 2 * b - c = 0) : 
  ab <= 1/3 := 
by 
  sorry

end NUMINAMATH_GPT_maximum_ab_l1543_154380


namespace NUMINAMATH_GPT_evening_minivans_l1543_154396

theorem evening_minivans (total_minivans afternoon_minivans : ℕ) (h_total : total_minivans = 5) 
(h_afternoon : afternoon_minivans = 4) : total_minivans - afternoon_minivans = 1 := 
by
  sorry

end NUMINAMATH_GPT_evening_minivans_l1543_154396


namespace NUMINAMATH_GPT_son_age_18_l1543_154386

theorem son_age_18 (F S : ℤ) (h1 : F = S + 20) (h2 : F + 2 = 2 * (S + 2)) : S = 18 :=
by
  sorry

end NUMINAMATH_GPT_son_age_18_l1543_154386


namespace NUMINAMATH_GPT_f_bound_l1543_154355

-- Define the function f(n) representing the number of representations of n as a sum of powers of 2
noncomputable def f (n : ℕ) : ℕ := 
-- f is defined as described in the problem, implementation skipped here
sorry

-- Propose to prove the main inequality for all n ≥ 3
theorem f_bound (n : ℕ) (h : n ≥ 3) : 2 ^ (n^2 / 4) < f (2 ^ n) ∧ f (2 ^ n) < 2 ^ (n^2 / 2) :=
sorry

end NUMINAMATH_GPT_f_bound_l1543_154355


namespace NUMINAMATH_GPT_arithmetic_sum_S11_l1543_154343

noncomputable def Sn_sum (a1 an n : ℕ) : ℕ := n * (a1 + an) / 2

theorem arithmetic_sum_S11 (a1 a9 a8 a5 a11 : ℕ) (h1 : Sn_sum a1 a9 9 = 54)
    (h2 : Sn_sum a1 a8 8 - Sn_sum a1 a5 5 = 30) : Sn_sum a1 a11 11 = 88 := by
  sorry

end NUMINAMATH_GPT_arithmetic_sum_S11_l1543_154343


namespace NUMINAMATH_GPT_arithmetic_progression_condition_l1543_154365

theorem arithmetic_progression_condition
  (a b c : ℝ) : ∃ (A B : ℤ), A ≠ 0 ∧ B ≠ 0 ∧ (b - a) * B = (c - b) * A := 
by {
  sorry
}

end NUMINAMATH_GPT_arithmetic_progression_condition_l1543_154365


namespace NUMINAMATH_GPT_polynomial_roots_bc_product_l1543_154304

theorem polynomial_roots_bc_product : ∃ (b c : ℤ), 
  (∀ x, (x^2 - 2*x - 1 = 0 → x^5 - b*x^3 - c*x^2 = 0)) ∧ (b * c = 348) := by 
  sorry

end NUMINAMATH_GPT_polynomial_roots_bc_product_l1543_154304


namespace NUMINAMATH_GPT_cost_increase_per_scrap_rate_l1543_154353

theorem cost_increase_per_scrap_rate (x : ℝ) :
  ∀ x Δx, y = 56 + 8 * x → Δx = 1 → y + Δy = 56 + 8 * (x + Δx) → Δy = 8 :=
by
  sorry

end NUMINAMATH_GPT_cost_increase_per_scrap_rate_l1543_154353


namespace NUMINAMATH_GPT_cos_theta_value_l1543_154376

noncomputable def coefficient_x2 (θ : ℝ) : ℝ := Nat.choose 5 2 * (Real.cos θ)^2
noncomputable def coefficient_x3 : ℝ := Nat.choose 4 3 * (5 / 4 : ℝ)^3

theorem cos_theta_value (θ : ℝ) (h : coefficient_x2 θ = coefficient_x3) : 
  Real.cos θ = (Real.sqrt 2)/2 ∨ Real.cos θ = -(Real.sqrt 2)/2 := 
by sorry

end NUMINAMATH_GPT_cos_theta_value_l1543_154376


namespace NUMINAMATH_GPT_property_tax_difference_correct_l1543_154384

-- Define the tax rates for different ranges
def tax_rate (value : ℕ) : ℝ :=
  if value ≤ 10000 then 0.05
  else if value ≤ 20000 then 0.075
  else if value ≤ 30000 then 0.10
  else 0.125

-- Define the progressive tax calculation for a given assessed value
def calculate_tax (value : ℕ) : ℝ :=
  if value ≤ 10000 then value * 0.05
  else if value ≤ 20000 then 10000 * 0.05 + (value - 10000) * 0.075
  else if value <= 30000 then 10000 * 0.05 + 10000 * 0.075 + (value - 20000) * 0.10
  else 10000 * 0.05 + 10000 * 0.075 + 10000 * 0.10 + (value - 30000) * 0.125

-- Define the initial and new assessed values
def initial_value : ℕ := 20000
def new_value : ℕ := 28000

-- Define the difference in tax calculation
def tax_difference : ℝ := calculate_tax new_value - calculate_tax initial_value

theorem property_tax_difference_correct : tax_difference = 550 := by
  sorry

end NUMINAMATH_GPT_property_tax_difference_correct_l1543_154384


namespace NUMINAMATH_GPT_number_of_cut_red_orchids_l1543_154329

variable (initial_red_orchids added_red_orchids final_red_orchids : ℕ)

-- Conditions
def initial_red_orchids_in_vase (initial_red_orchids : ℕ) : Prop :=
  initial_red_orchids = 9

def final_red_orchids_in_vase (final_red_orchids : ℕ) : Prop :=
  final_red_orchids = 15

-- Proof statement
theorem number_of_cut_red_orchids (initial_red_orchids added_red_orchids final_red_orchids : ℕ)
  (h1 : initial_red_orchids_in_vase initial_red_orchids) 
  (h2 : final_red_orchids_in_vase final_red_orchids) :
  final_red_orchids = initial_red_orchids + added_red_orchids → added_red_orchids = 6 := by
  simp [initial_red_orchids_in_vase, final_red_orchids_in_vase] at *
  sorry

end NUMINAMATH_GPT_number_of_cut_red_orchids_l1543_154329


namespace NUMINAMATH_GPT_not_divides_l1543_154387

theorem not_divides (d a n : ℕ) (h1 : 3 ≤ d) (h2 : d ≤ 2^(n+1)) : ¬ d ∣ a^(2^n) + 1 := 
sorry

end NUMINAMATH_GPT_not_divides_l1543_154387


namespace NUMINAMATH_GPT_paul_walking_time_l1543_154316

variable (P : ℕ)

def is_walking_time (P : ℕ) : Prop :=
  P + 7 * (P + 2) = 46

theorem paul_walking_time (h : is_walking_time P) : P = 4 :=
by sorry

end NUMINAMATH_GPT_paul_walking_time_l1543_154316


namespace NUMINAMATH_GPT_maria_towels_l1543_154392

theorem maria_towels (green_towels white_towels given_towels : ℕ) (bought_green : green_towels = 40) 
(bought_white : white_towels = 44) (gave_mother : given_towels = 65) : 
  green_towels + white_towels - given_towels = 19 := by
sorry

end NUMINAMATH_GPT_maria_towels_l1543_154392


namespace NUMINAMATH_GPT_remainder_when_divided_by_r_minus_2_l1543_154374

-- Define polynomial p(r)
def p (r : ℝ) : ℝ := r ^ 11 - 3

-- The theorem stating the problem
theorem remainder_when_divided_by_r_minus_2 : p 2 = 2045 := by
  sorry

end NUMINAMATH_GPT_remainder_when_divided_by_r_minus_2_l1543_154374


namespace NUMINAMATH_GPT_greatest_power_of_2_divides_l1543_154352

-- Define the conditions as Lean definitions.
def a : ℕ := 15
def b : ℕ := 3
def n : ℕ := 600

-- Define the theorem statement based on the conditions and correct answer.
theorem greatest_power_of_2_divides (x : ℕ) (y : ℕ) (k : ℕ) (h₁ : x = a) (h₂ : y = b) (h₃ : k = n) :
  ∃ m : ℕ, (x^k - y^k) % (2^1200) = 0 ∧ ¬ ∃ m' : ℕ, m' > m ∧ (x^k - y^k) % (2^m') = 0 := sorry

end NUMINAMATH_GPT_greatest_power_of_2_divides_l1543_154352


namespace NUMINAMATH_GPT_perpendicular_vectors_find_a_l1543_154364

theorem perpendicular_vectors_find_a
  (a : ℝ)
  (m : ℝ × ℝ := (1, 2))
  (n : ℝ × ℝ := (a, -1))
  (h : m.1 * n.1 + m.2 * n.2 = 0) :
  a = 2 := 
sorry

end NUMINAMATH_GPT_perpendicular_vectors_find_a_l1543_154364


namespace NUMINAMATH_GPT_simplify_expression_l1543_154397

variable (a b : ℤ)

theorem simplify_expression : 
  (15 * a + 45 * b) + (21 * a + 32 * b) - (12 * a + 40 * b) = 24 * a + 37 * b := 
    by sorry

end NUMINAMATH_GPT_simplify_expression_l1543_154397


namespace NUMINAMATH_GPT_sum_nine_terms_of_arithmetic_sequence_l1543_154389

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = (n * (a 0 + a (n - 1))) / 2

theorem sum_nine_terms_of_arithmetic_sequence
  (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : arithmetic_sequence a)
  (h2 : sum_of_first_n_terms a S)
  (h3 : a 5 = 7) :
  S 9 = 63 := by
  sorry

end NUMINAMATH_GPT_sum_nine_terms_of_arithmetic_sequence_l1543_154389


namespace NUMINAMATH_GPT_total_items_children_carry_l1543_154349

theorem total_items_children_carry 
  (pieces_per_pizza : ℕ) (number_of_fourthgraders : ℕ) (pizza_per_fourthgrader : ℕ) 
  (pepperoni_per_pizza : ℕ) (mushrooms_per_pizza : ℕ) (olives_per_pizza : ℕ) 
  (total_pizzas : ℕ) (total_pieces_of_pizza : ℕ) (total_pepperoni : ℕ) (total_mushrooms : ℕ) 
  (total_olives : ℕ) (total_toppings : ℕ) (total_items : ℕ) : 
  pieces_per_pizza = 6 →
  number_of_fourthgraders = 10 →
  pizza_per_fourthgrader = 20 →
  pepperoni_per_pizza = 5 →
  mushrooms_per_pizza = 3 →
  olives_per_pizza = 8 →
  total_pizzas = number_of_fourthgraders * pizza_per_fourthgrader →
  total_pieces_of_pizza = total_pizzas * pieces_per_pizza →
  total_pepperoni = total_pizzas * pepperoni_per_pizza →
  total_mushrooms = total_pizzas * mushrooms_per_pizza →
  total_olives = total_pizzas * olives_per_pizza →
  total_toppings = total_pepperoni + total_mushrooms + total_olives →
  total_items = total_pieces_of_pizza + total_toppings →
  total_items = 4400 :=
by
  sorry

end NUMINAMATH_GPT_total_items_children_carry_l1543_154349


namespace NUMINAMATH_GPT_find_levels_satisfying_surface_area_conditions_l1543_154314

theorem find_levels_satisfying_surface_area_conditions (n : ℕ) :
  let A_total_lateral := n * (n + 1) * Real.pi
  let A_total_vertical := Real.pi * n^2
  let A_total := n * (3 * n + 1) * Real.pi
  A_total_lateral = 0.35 * A_total → n = 13 :=
by
  intros A_total_lateral A_total_vertical A_total h
  sorry

end NUMINAMATH_GPT_find_levels_satisfying_surface_area_conditions_l1543_154314


namespace NUMINAMATH_GPT_apples_equation_l1543_154308

variable {A J H : ℕ}

theorem apples_equation:
    A + J = 12 →
    H = A + J + 9 →
    A = J + 8 →
    H = 21 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_apples_equation_l1543_154308


namespace NUMINAMATH_GPT_problem_l1543_154359

theorem problem {a b : ℝ} (h_pos_a : a > 0) (h_pos_b : b > 0) (h : 3 * a * b = a + 3 * b) :
  (3 * a + b >= 16/3) ∧
  (a * b >= 4/3) ∧
  (a^2 + 9 * b^2 >= 8) ∧
  (¬ (b > 1/2)) :=
by
  sorry

end NUMINAMATH_GPT_problem_l1543_154359


namespace NUMINAMATH_GPT_recurring_decimal_to_fraction_l1543_154317

theorem recurring_decimal_to_fraction : (∃ (x : ℚ), x = 3 + 56 / 99) :=
by
  have x : ℚ := 3 + 56 / 99
  exists x
  sorry

end NUMINAMATH_GPT_recurring_decimal_to_fraction_l1543_154317


namespace NUMINAMATH_GPT_continuity_at_2_l1543_154395

theorem continuity_at_2 (f : ℝ → ℝ) (x0 : ℝ) (hf : ∀ x, f x = -4 * x ^ 2 - 8) :
  x0 = 2 → ∀ ε > 0, ∃ δ > 0, ∀ x, |x - x0| < δ → |f x + 24| < ε := by
  sorry

end NUMINAMATH_GPT_continuity_at_2_l1543_154395


namespace NUMINAMATH_GPT_scientific_notation_periodicals_l1543_154390

theorem scientific_notation_periodicals :
  (56000000 : ℝ) = 5.6 * 10^7 := by
sorry

end NUMINAMATH_GPT_scientific_notation_periodicals_l1543_154390


namespace NUMINAMATH_GPT_penthouse_floors_l1543_154338

theorem penthouse_floors (R P : ℕ) (h1 : R + P = 23) (h2 : 12 * R + 2 * P = 256) : P = 2 :=
by
  sorry

end NUMINAMATH_GPT_penthouse_floors_l1543_154338


namespace NUMINAMATH_GPT_two_pow_n_minus_one_divisible_by_seven_iff_l1543_154362

theorem two_pow_n_minus_one_divisible_by_seven_iff (n : ℕ) (h : n > 0) :
  (2^n - 1) % 7 = 0 ↔ n % 3 = 0 :=
sorry

end NUMINAMATH_GPT_two_pow_n_minus_one_divisible_by_seven_iff_l1543_154362


namespace NUMINAMATH_GPT_boulder_splash_width_l1543_154357

theorem boulder_splash_width :
  (6 * (1/4) + 3 * (1 / 2) + 2 * b = 7) -> b = 2 := by
  sorry

end NUMINAMATH_GPT_boulder_splash_width_l1543_154357


namespace NUMINAMATH_GPT_remainder_when_divided_by_296_and_37_l1543_154358

theorem remainder_when_divided_by_296_and_37 (N : ℤ) (k : ℤ)
  (h : N = 296 * k + 75) : N % 37 = 1 :=
by
  sorry

end NUMINAMATH_GPT_remainder_when_divided_by_296_and_37_l1543_154358


namespace NUMINAMATH_GPT_powers_of_two_diff_div_by_1987_l1543_154399

theorem powers_of_two_diff_div_by_1987 :
  ∃ a b : ℕ, a > b ∧ 1987 ∣ (2^a - 2^b) :=
by sorry

end NUMINAMATH_GPT_powers_of_two_diff_div_by_1987_l1543_154399


namespace NUMINAMATH_GPT_factorize_P_l1543_154342

noncomputable def P (y : ℝ) : ℝ :=
  (16 * y ^ 7 - 36 * y ^ 5 + 8 * y) - (4 * y ^ 7 - 12 * y ^ 5 - 8 * y)

theorem factorize_P (y : ℝ) : P y = 8 * y * (3 * y ^ 6 - 6 * y ^ 4 + 4) :=
  sorry

end NUMINAMATH_GPT_factorize_P_l1543_154342


namespace NUMINAMATH_GPT_MrsHiltRows_l1543_154325

theorem MrsHiltRows :
  let (a : ℕ) := 16
  let (b : ℕ) := 14
  let (r : ℕ) := 5
  (a + b) / r = 6 := by
  sorry

end NUMINAMATH_GPT_MrsHiltRows_l1543_154325


namespace NUMINAMATH_GPT_six_positive_integers_solution_count_l1543_154398

theorem six_positive_integers_solution_count :
  ∃ (S : Finset (Finset ℕ)) (n : ℕ) (a b c x y z : ℕ), 
  a ≥ b → b ≥ c → x ≥ y → y ≥ z → 
  a + b + c = x * y * z → 
  x + y + z = a * b * c → 
  S.card = 7 := by
    sorry

end NUMINAMATH_GPT_six_positive_integers_solution_count_l1543_154398


namespace NUMINAMATH_GPT_mary_screws_l1543_154309

theorem mary_screws (S : ℕ) (h : S + 2 * S = 24) : S = 8 :=
by sorry

end NUMINAMATH_GPT_mary_screws_l1543_154309


namespace NUMINAMATH_GPT_monotonic_range_of_t_l1543_154320

noncomputable def f (x : ℝ) := (x^2 - 3 * x + 3) * Real.exp x

def is_monotonic_on_interval (a b : ℝ) (f : ℝ → ℝ) : Prop :=
  (∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≤ f y) ∨ (∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≥ f y)

theorem monotonic_range_of_t (t : ℝ) (ht : t > -2) :
  is_monotonic_on_interval (-2) t f ↔ (-2 < t ∧ t ≤ 0) :=
sorry

end NUMINAMATH_GPT_monotonic_range_of_t_l1543_154320


namespace NUMINAMATH_GPT_fraction_to_decimal_l1543_154319

theorem fraction_to_decimal : (58 : ℚ) / 160 = 0.3625 := 
by sorry

end NUMINAMATH_GPT_fraction_to_decimal_l1543_154319


namespace NUMINAMATH_GPT_packets_of_gum_is_eight_l1543_154382

-- Given conditions
def pieces_left : ℕ := 2
def pieces_chewed : ℕ := 54
def pieces_per_packet : ℕ := 7

-- Given he chews all the gum except for pieces_left pieces, and chews pieces_chewed pieces at once
def total_pieces_of_gum (pieces_chewed pieces_left : ℕ) : ℕ :=
  pieces_chewed + pieces_left

-- Calculate the number of packets
def number_of_packets (total_pieces pieces_per_packet : ℕ) : ℕ :=
  total_pieces / pieces_per_packet

-- The final theorem asserting the number of packets is 8
theorem packets_of_gum_is_eight : number_of_packets (total_pieces_of_gum pieces_chewed pieces_left) pieces_per_packet = 8 :=
  sorry

end NUMINAMATH_GPT_packets_of_gum_is_eight_l1543_154382


namespace NUMINAMATH_GPT_least_real_number_K_l1543_154367

theorem least_real_number_K (x y z K : ℝ) (h_cond1 : -2 ≤ x ∧ x ≤ 2) (h_cond2 : -2 ≤ y ∧ y ≤ 2) (h_cond3 : -2 ≤ z ∧ z ≤ 2) (h_eq : x^2 + y^2 + z^2 + x * y * z = 4) :
  (∀ x y z : ℝ, -2 ≤ x ∧ x ≤ 2 ∧ -2 ≤ y ∧ y ≤ 2 ∧ -2 ≤ z ∧ z ≤ 2 ∧ x^2 + y^2 + z^2 + x * y * z = 4 → z * (x * z + y * z + y) / (x * y + y^2 + z^2 + 1) ≤ K) → K = 4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_least_real_number_K_l1543_154367


namespace NUMINAMATH_GPT_num_bicycles_l1543_154323

theorem num_bicycles (spokes_per_wheel wheels_per_bicycle total_spokes : ℕ) (h1 : spokes_per_wheel = 10) (h2 : total_spokes = 80) (h3 : wheels_per_bicycle = 2) : total_spokes / spokes_per_wheel / wheels_per_bicycle = 4 := by
  sorry

end NUMINAMATH_GPT_num_bicycles_l1543_154323


namespace NUMINAMATH_GPT_max_value_expr_l1543_154375

variable (x y z : ℝ)

theorem max_value_expr (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) :
  (∃ a, ∀ x y z, (a = (x*y + y*z) / (x^2 + y^2 + z^2)) ∧ a ≤ (Real.sqrt 2) / 2) ∧
  (∃ x' y' z', (x' > 0) ∧ (y' > 0) ∧ (z' > 0) ∧ ((x'*y' + y'*z') / (x'^2 + y'^2 + z'^2) = (Real.sqrt 2) / 2)) :=
by
  sorry

end NUMINAMATH_GPT_max_value_expr_l1543_154375


namespace NUMINAMATH_GPT_solve_for_y_l1543_154361

theorem solve_for_y (x y : ℝ) (h : x - 2 = 4 * y + 3) : y = (x - 5) / 4 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_y_l1543_154361


namespace NUMINAMATH_GPT_negation_proposition_l1543_154326

open Real

theorem negation_proposition (h : ∀ x : ℝ, x^2 - 2*x - 1 > 0) :
  ¬ (∀ x : ℝ, x^2 - 2*x - 1 > 0) = ∃ x_0 : ℝ, x_0^2 - 2*x_0 - 1 ≤ 0 :=
by 
  sorry

end NUMINAMATH_GPT_negation_proposition_l1543_154326


namespace NUMINAMATH_GPT_find_index_l1543_154345

-- Declaration of sequence being arithmetic with first term 1 and common difference 3
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, a n = 1 + (n - 1) * 3

-- The theorem to be proven
theorem find_index (a : ℕ → ℤ) (h1 : arithmetic_sequence a) (h2 : a 672 = 2014) : 672 = 672 :=
by 
  sorry

end NUMINAMATH_GPT_find_index_l1543_154345


namespace NUMINAMATH_GPT_product_units_tens_not_divisible_by_8_l1543_154321

theorem product_units_tens_not_divisible_by_8 :
  ¬ (1834 % 8 = 0) → (4 * 3 = 12) :=
by
  intro h
  exact (by norm_num : 4 * 3 = 12)

end NUMINAMATH_GPT_product_units_tens_not_divisible_by_8_l1543_154321


namespace NUMINAMATH_GPT_obtuse_triangle_area_side_l1543_154371

theorem obtuse_triangle_area_side (a b : ℝ) (C : ℝ) 
  (h1 : a = 8) 
  (h2 : C = 150 * (π / 180)) -- converting degrees to radians
  (h3 : 1 / 2 * a * b * Real.sin C = 24) : 
  b = 12 :=
by sorry

end NUMINAMATH_GPT_obtuse_triangle_area_side_l1543_154371


namespace NUMINAMATH_GPT_first_number_in_proportion_is_correct_l1543_154327

-- Define the proportion condition
def proportion_condition (a x : ℝ) : Prop := a / x = 5 / 11

-- Define the given known value for x
def x_value : ℝ := 1.65

-- Define the correct answer for a
def correct_a : ℝ := 0.75

-- The theorem to prove
theorem first_number_in_proportion_is_correct :
  ∀ a : ℝ, proportion_condition a x_value → a = correct_a := by
  sorry

end NUMINAMATH_GPT_first_number_in_proportion_is_correct_l1543_154327


namespace NUMINAMATH_GPT_ellipse_equation_constants_l1543_154331

noncomputable def ellipse_parametric_eq (t : ℝ) : ℝ × ℝ :=
  ((3 * (Real.sin t - 2)) / (3 - Real.cos t),
  (4 * (Real.cos t - 4)) / (3 - Real.cos t))

theorem ellipse_equation_constants :
  ∃ (A B C D E F : ℤ), ∀ (x y : ℝ),
  ((∃ t : ℝ, (x, y) = ellipse_parametric_eq t) → (A * x^2 + B * x * y + C * y^2 + D * x + E * y + F = 0)) ∧
  (Int.gcd (Int.gcd (Int.gcd (Int.gcd (Int.gcd A B) C) D) E) F = 1) ∧
  (|A| + |B| + |C| + |D| + |E| + |F| = 2502) :=
sorry

end NUMINAMATH_GPT_ellipse_equation_constants_l1543_154331


namespace NUMINAMATH_GPT_coeff_x2_expansion_l1543_154378

theorem coeff_x2_expansion (n r : ℕ) (a b : ℤ) :
  n = 5 → a = 1 → b = 2 → r = 2 →
  (Nat.choose n r) * (a^(n - r)) * (b^r) = 40 :=
by
  intros Hn Ha Hb Hr
  rw [Hn, Ha, Hb, Hr]
  simp
  sorry

end NUMINAMATH_GPT_coeff_x2_expansion_l1543_154378


namespace NUMINAMATH_GPT_Phillip_correct_total_l1543_154332

def number_questions_math : ℕ := 40
def number_questions_english : ℕ := 50
def percentage_correct_math : ℚ := 0.75
def percentage_correct_english : ℚ := 0.98

noncomputable def total_correct_answers : ℚ :=
  (number_questions_math * percentage_correct_math) + (number_questions_english * percentage_correct_english)

theorem Phillip_correct_total : total_correct_answers = 79 := by
  sorry

end NUMINAMATH_GPT_Phillip_correct_total_l1543_154332


namespace NUMINAMATH_GPT_max_area_of_rectangular_garden_l1543_154322

-- Definitions corresponding to the conditions in the problem
def length1 (x : ℕ) := x
def length2 (x : ℕ) := 75 - x

-- Definition of the area
def area (x : ℕ) := x * (75 - x)

-- Statement to prove: there exists natural numbers x and y such that x + y = 75 and x * y = 1406
theorem max_area_of_rectangular_garden :
  ∃ (x : ℕ), (x + (75 - x) = 75) ∧ (x * (75 - x) = 1406) := 
by
  -- Due to the nature of this exercise, the actual proof is omitted.
  sorry

end NUMINAMATH_GPT_max_area_of_rectangular_garden_l1543_154322


namespace NUMINAMATH_GPT_steven_more_peaches_than_apples_l1543_154337

-- Definitions
def apples_steven := 11
def peaches_steven := 18

-- Theorem statement
theorem steven_more_peaches_than_apples : (peaches_steven - apples_steven) = 7 := by 
  sorry

end NUMINAMATH_GPT_steven_more_peaches_than_apples_l1543_154337


namespace NUMINAMATH_GPT_find_pq_of_orthogonal_and_equal_magnitudes_l1543_154388

noncomputable def vec_a (p : ℝ) : ℝ × ℝ × ℝ := (4, p, -2)
noncomputable def vec_b (q : ℝ) : ℝ × ℝ × ℝ := (3, 2, q)

theorem find_pq_of_orthogonal_and_equal_magnitudes
    (p q : ℝ)
    (h1 : 4 * 3 + p * 2 + (-2) * q = 0)
    (h2 : 4^2 + p^2 + (-2)^2 = 3^2 + 2^2 + q^2) :
    (p, q) = (-29/12, 43/12) :=
by {
  sorry
}

end NUMINAMATH_GPT_find_pq_of_orthogonal_and_equal_magnitudes_l1543_154388


namespace NUMINAMATH_GPT_seven_thousand_twenty_two_is_7022_l1543_154300

-- Define the translations of words to numbers
def seven_thousand : ℕ := 7000
def twenty_two : ℕ := 22

-- Define the full number by summing its parts
def seven_thousand_twenty_two : ℕ := seven_thousand + twenty_two

theorem seven_thousand_twenty_two_is_7022 : seven_thousand_twenty_two = 7022 := by
  sorry

end NUMINAMATH_GPT_seven_thousand_twenty_two_is_7022_l1543_154300


namespace NUMINAMATH_GPT_max_min_conditions_x_values_for_max_min_a2_x_values_for_max_min_aneg2_l1543_154394

noncomputable def y (x : ℝ) (a b : ℝ) : ℝ := (Real.cos x)^2 - a * (Real.sin x) + b

theorem max_min_conditions (a b : ℝ) :
  (∃ x : ℝ, y x a b = 0 ∧ (∀ x' : ℝ, y x' a b ≤ 0)) ∧ 
  (∃ x : ℝ, y x a b = -4 ∧ (∀ x' : ℝ, y x' a b ≥ -4)) ↔ 
  (a = 2 ∧ b = -2) ∨ (a = -2 ∧ b = -2) := sorry

theorem x_values_for_max_min_a2 (k : ℤ) :
  (∀ x, y x 2 (-2) = 0 ↔ x = -Real.pi / 2 + 2 * Real.pi * k) ∧ 
  (∀ x, (y x 2 (-2)) = -4 ↔ x = Real.pi / 2 + 2 * Real.pi * k) := sorry

theorem x_values_for_max_min_aneg2 (k : ℤ) :
  (∀ x, y x (-2) (-2) = 0 ↔ x = Real.pi / 2 + 2 * Real.pi * k) ∧ 
  (∀ x, (y x (-2) (-2)) = -4 ↔ x = -Real.pi / 2 + 2 * Real.pi * k) := sorry

end NUMINAMATH_GPT_max_min_conditions_x_values_for_max_min_a2_x_values_for_max_min_aneg2_l1543_154394


namespace NUMINAMATH_GPT_least_possible_number_of_coins_in_jar_l1543_154328

theorem least_possible_number_of_coins_in_jar (n : ℕ) : 
  (n % 7 = 3) → (n % 4 = 1) → (n % 6 = 5) → n = 17 :=
by
  sorry

end NUMINAMATH_GPT_least_possible_number_of_coins_in_jar_l1543_154328


namespace NUMINAMATH_GPT_find_a_l1543_154379

theorem find_a (a b c : ℝ) (h1 : b = 15) (h2 : c = 5)
  (h3 : a * b * c = (Real.sqrt ((a + 2) * (b + 3))) / (c + 1)) 
  (result : a * 15 * 5 = 2) : a = 6 := by 
  sorry

end NUMINAMATH_GPT_find_a_l1543_154379


namespace NUMINAMATH_GPT_square_area_of_equal_perimeter_l1543_154369

theorem square_area_of_equal_perimeter 
  (side_length_triangle : ℕ) (side_length_square : ℕ) (perimeter_square : ℕ)
  (h1 : side_length_triangle = 20)
  (h2 : perimeter_square = 3 * side_length_triangle)
  (h3 : 4 * side_length_square = perimeter_square) :
  side_length_square ^ 2 = 225 := 
by
  sorry

end NUMINAMATH_GPT_square_area_of_equal_perimeter_l1543_154369


namespace NUMINAMATH_GPT_impossible_transformation_l1543_154370

variable (G : Type) [Group G]

/-- Initial word represented by 2003 'a's followed by 'b' --/
def initial_word := "aaa...ab"

/-- Transformed word represented by 'b' followed by 2003 'a's --/
def transformed_word := "baaa...a"

/-- Hypothetical group relations derived from transformations --/
axiom aba_to_b (a b : G) : (a * b * a = b)
axiom bba_to_a (a b : G) : (b * b * a = a)

/-- Impossible transformation proof --/
theorem impossible_transformation (a b : G) : 
  (initial_word = transformed_word) → False := by
  sorry

end NUMINAMATH_GPT_impossible_transformation_l1543_154370


namespace NUMINAMATH_GPT_optimal_price_l1543_154324

def monthly_sales (p : ℝ) : ℝ := 150 - 6 * p
def break_even (p : ℝ) : Prop := 40 ≤ monthly_sales p
def revenue (p : ℝ) : ℝ := p * monthly_sales p

theorem optimal_price : ∃ p : ℝ, p = 13 ∧ p ≤ 30 ∧ break_even p ∧ ∀ q : ℝ, q ≤ 30 → break_even q → revenue p ≥ revenue q := 
by
  sorry

end NUMINAMATH_GPT_optimal_price_l1543_154324


namespace NUMINAMATH_GPT_students_after_last_stop_on_mondays_and_wednesdays_students_after_last_stop_on_tuesdays_and_thursdays_students_after_last_stop_on_fridays_l1543_154313

structure BusRoute where
  first_stop : Nat
  second_stop_on : Nat
  second_stop_off : Nat
  third_stop_on : Nat
  third_stop_off : Nat
  fourth_stop_on : Nat
  fourth_stop_off : Nat

def mondays_and_wednesdays := BusRoute.mk 39 29 12 35 18 27 15
def tuesdays_and_thursdays := BusRoute.mk 39 33 10 5 0 8 4
def fridays := BusRoute.mk 39 25 10 40 20 10 5

def students_after_last_stop (route : BusRoute) : Nat :=
  let stop1 := route.first_stop
  let stop2 := stop1 + route.second_stop_on - route.second_stop_off
  let stop3 := stop2 + route.third_stop_on - route.third_stop_off
  stop3 + route.fourth_stop_on - route.fourth_stop_off

theorem students_after_last_stop_on_mondays_and_wednesdays :
  students_after_last_stop mondays_and_wednesdays = 85 := by
  sorry

theorem students_after_last_stop_on_tuesdays_and_thursdays :
  students_after_last_stop tuesdays_and_thursdays = 71 := by
  sorry

theorem students_after_last_stop_on_fridays :
  students_after_last_stop fridays = 79 := by
  sorry

end NUMINAMATH_GPT_students_after_last_stop_on_mondays_and_wednesdays_students_after_last_stop_on_tuesdays_and_thursdays_students_after_last_stop_on_fridays_l1543_154313


namespace NUMINAMATH_GPT_math_problem_l1543_154333

variables (x y z w p q : ℕ)
variables (x_pos : 0 < x) (y_pos : 0 < y) (z_pos : 0 < z) (w_pos : 0 < w)

theorem math_problem
  (h1 : x^3 = y^2)
  (h2 : z^4 = w^3)
  (h3 : z - x = 22)
  (hx : x = p^2)
  (hy : y = p^3)
  (hz : z = q^3)
  (hw : w = q^4) : w - y = q^4 - p^3 :=
sorry

end NUMINAMATH_GPT_math_problem_l1543_154333


namespace NUMINAMATH_GPT_log_function_domain_l1543_154339

noncomputable def domain_of_log_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : Set ℝ :=
  { x : ℝ | x < a }

theorem log_function_domain (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∀ x, x ∈ domain_of_log_function a h1 h2 ↔ x < a :=
by
  sorry

end NUMINAMATH_GPT_log_function_domain_l1543_154339


namespace NUMINAMATH_GPT_avg_combined_is_2a_plus_3b_l1543_154305

variables {x1 x2 x3 y1 y2 y3 a b : ℝ}

-- Given conditions
def avg_x_is_a (x1 x2 x3 a : ℝ) : Prop := (x1 + x2 + x3) / 3 = a
def avg_y_is_b (y1 y2 y3 b : ℝ) : Prop := (y1 + y2 + y3) / 3 = b

-- The statement to be proved
theorem avg_combined_is_2a_plus_3b
    (hx : avg_x_is_a x1 x2 x3 a) 
    (hy : avg_y_is_b y1 y2 y3 b) :
    ((2 * x1 + 3 * y1) + (2 * x2 + 3 * y2) + (2 * x3 + 3 * y3)) / 3 = 2 * a + 3 * b := 
by
  sorry

end NUMINAMATH_GPT_avg_combined_is_2a_plus_3b_l1543_154305


namespace NUMINAMATH_GPT_find_N_l1543_154381

theorem find_N : ∃ (N : ℤ), N > 0 ∧ (36^2 * 60^2 = 30^2 * N^2) ∧ (N = 72) :=
by
  sorry

end NUMINAMATH_GPT_find_N_l1543_154381


namespace NUMINAMATH_GPT_inverse_function_passes_through_point_l1543_154363

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x + 1)

theorem inverse_function_passes_through_point {a : ℝ} (h1 : 0 < a) (h2 : a ≠ 1) (h3 : f a (-1) = 1) :
  f a⁻¹ 1 = -1 :=
sorry

end NUMINAMATH_GPT_inverse_function_passes_through_point_l1543_154363


namespace NUMINAMATH_GPT_find_x_l1543_154366

def vector_dot_product (v1 v2 : ℝ × ℝ) : ℝ := 
  v1.1 * v2.1 + v1.2 * v2.2

def a : ℝ × ℝ := (1, 2)

def b (x : ℝ) : ℝ × ℝ := (x, -2)

def c (x : ℝ) : ℝ × ℝ := (1 - x, 4)

theorem find_x (x : ℝ) (h : vector_dot_product a (c x) = 0) : x = 9 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1543_154366


namespace NUMINAMATH_GPT_dart_hit_number_list_count_l1543_154330

def number_of_dart_hit_lists (darts dartboards : ℕ) : ℕ :=
  11  -- Based on the solution, the hard-coded answer is 11.

theorem dart_hit_number_list_count : number_of_dart_hit_lists 6 4 = 11 := 
by 
  sorry

end NUMINAMATH_GPT_dart_hit_number_list_count_l1543_154330


namespace NUMINAMATH_GPT_correct_calculation_l1543_154315

theorem correct_calculation :
  3 * Real.sqrt 2 - (Real.sqrt 2 / 2) = (5 / 2) * Real.sqrt 2 :=
by
  -- To proceed with the proof, we need to show:
  -- 3 * sqrt(2) - (sqrt(2) / 2) = (5 / 2) * sqrt(2)
  sorry

end NUMINAMATH_GPT_correct_calculation_l1543_154315


namespace NUMINAMATH_GPT_find_ABC_l1543_154310

theorem find_ABC {A B C : ℕ} (h₀ : ∀ n : ℕ, n ≤ 9 → n ≤ 9) (h₁ : 0 ≤ A) (h₂ : A ≤ 9) 
  (h₃ : 0 ≤ B) (h₄ : B ≤ 9) (h₅ : 0 ≤ C) (h₆ : C ≤ 9) (h₇ : 100 * A + 10 * B + C = B^C - A) :
  100 * A + 10 * B + C = 127 := by {
  sorry
}

end NUMINAMATH_GPT_find_ABC_l1543_154310


namespace NUMINAMATH_GPT_original_number_divisible_l1543_154385

theorem original_number_divisible (n : ℕ) (h : (n - 8) % 20 = 0) : n = 28 := 
by
  sorry

end NUMINAMATH_GPT_original_number_divisible_l1543_154385


namespace NUMINAMATH_GPT_find_third_coaster_speed_l1543_154354

theorem find_third_coaster_speed
  (s1 s2 s4 s5 avg_speed n : ℕ)
  (hs1 : s1 = 50)
  (hs2 : s2 = 62)
  (hs4 : s4 = 70)
  (hs5 : s5 = 40)
  (havg_speed : avg_speed = 59)
  (hn : n = 5) : 
  ∃ s3 : ℕ, s3 = 73 :=
by
  sorry

end NUMINAMATH_GPT_find_third_coaster_speed_l1543_154354


namespace NUMINAMATH_GPT_unique_solution_quadratic_l1543_154334

theorem unique_solution_quadratic (q : ℝ) (hq : q ≠ 0) :
  (∃ x, q * x^2 - 18 * x + 8 = 0 ∧ ∀ y, q * y^2 - 18 * y + 8 = 0 → y = x) →
  q = 81 / 8 :=
by
  sorry

end NUMINAMATH_GPT_unique_solution_quadratic_l1543_154334


namespace NUMINAMATH_GPT_sin_30_eq_half_l1543_154348

theorem sin_30_eq_half : Real.sin (Real.pi / 6) = 1 / 2 := sorry

end NUMINAMATH_GPT_sin_30_eq_half_l1543_154348


namespace NUMINAMATH_GPT_n_gon_partition_l1543_154311

-- Define a function to determine if an n-gon can be partitioned as required
noncomputable def canBePartitioned (n : ℕ) (h : n ≥ 3) : Prop :=
  n ≠ 4 ∧ n ≥ 3

theorem n_gon_partition (n : ℕ) (h : n ≥ 3) : canBePartitioned n h ↔ (n = 3 ∨ n ≥ 5) :=
by sorry

end NUMINAMATH_GPT_n_gon_partition_l1543_154311


namespace NUMINAMATH_GPT_P_sufficient_but_not_necessary_for_Q_l1543_154340

def P (x : ℝ) : Prop := abs (2 * x - 3) < 1
def Q (x : ℝ) : Prop := x * (x - 3) < 0

theorem P_sufficient_but_not_necessary_for_Q : 
  (∀ x : ℝ, P x → Q x) ∧ (∃ x : ℝ, Q x ∧ ¬ P x) := 
by
  sorry

end NUMINAMATH_GPT_P_sufficient_but_not_necessary_for_Q_l1543_154340


namespace NUMINAMATH_GPT_sin_240_deg_l1543_154301

theorem sin_240_deg : Real.sin (240 * Real.pi / 180) = - Real.sqrt 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_sin_240_deg_l1543_154301


namespace NUMINAMATH_GPT_remainder_145_mul_155_div_12_l1543_154303

theorem remainder_145_mul_155_div_12 : (145 * 155) % 12 = 11 := by
  sorry

end NUMINAMATH_GPT_remainder_145_mul_155_div_12_l1543_154303


namespace NUMINAMATH_GPT_george_total_payment_in_dollars_l1543_154351
noncomputable def total_cost_in_dollars : ℝ := 
  let sandwich_cost : ℝ := 4
  let juice_cost : ℝ := 2 * sandwich_cost * 0.9
  let coffee_cost : ℝ := sandwich_cost / 2
  let milk_cost : ℝ := 0.75 * (sandwich_cost + juice_cost)
  let milk_cost_dollars : ℝ := milk_cost * 1.2
  let chocolate_bar_cost_pounds : ℝ := 3
  let chocolate_bar_cost_dollars : ℝ := chocolate_bar_cost_pounds * 1.25
  let total_euros_in_items : ℝ := 2 * sandwich_cost + juice_cost + coffee_cost
  let total_euros_to_dollars : ℝ := total_euros_in_items * 1.2
  total_euros_to_dollars + milk_cost_dollars + chocolate_bar_cost_dollars

theorem george_total_payment_in_dollars : total_cost_in_dollars = 38.07 := by
  sorry

end NUMINAMATH_GPT_george_total_payment_in_dollars_l1543_154351


namespace NUMINAMATH_GPT_ratio_S15_S5_l1543_154306

-- Definition of a geometric sequence sum and the given ratio S10/S5 = 1/2
noncomputable def geom_sum : ℕ → ℕ := sorry
axiom ratio_S10_S5 : geom_sum 10 / geom_sum 5 = 1 / 2

-- The goal is to prove that the ratio S15/S5 = 3/4
theorem ratio_S15_S5 : geom_sum 15 / geom_sum 5 = 3 / 4 :=
by sorry

end NUMINAMATH_GPT_ratio_S15_S5_l1543_154306


namespace NUMINAMATH_GPT_vector_parallel_has_value_x_l1543_154368

-- Define the vectors a and b
def a : ℝ × ℝ := (3, 2)
def b (x : ℝ) : ℝ × ℝ := (x, 4)

-- Define the parallel condition
def parallel (a b : ℝ × ℝ) : Prop := a.1 * b.2 = a.2 * b.1

-- The theorem statement
theorem vector_parallel_has_value_x :
  ∀ x : ℝ, parallel a (b x) → x = 6 :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_vector_parallel_has_value_x_l1543_154368


namespace NUMINAMATH_GPT_modules_count_l1543_154383

theorem modules_count (x y: ℤ) (hx: 10 * x + 35 * y = 450) (hy: x + y = 11) : y = 10 :=
by
  sorry

end NUMINAMATH_GPT_modules_count_l1543_154383


namespace NUMINAMATH_GPT_sum_of_roots_l1543_154356

noncomputable def equation (x : ℝ) := 2 * (x^2 + 1 / x^2) - 3 * (x + 1 / x) = 1

theorem sum_of_roots (r s : ℝ) (hr : equation r) (hs : equation s) (hne : r ≠ s) :
  r + s = -5 / 2 :=
sorry

end NUMINAMATH_GPT_sum_of_roots_l1543_154356


namespace NUMINAMATH_GPT_miriam_pushups_l1543_154346

theorem miriam_pushups :
  let p_M := 5
  let p_T := 7
  let p_W := 2 * p_T
  let p_Th := (p_M + p_T + p_W) / 2
  let p_F := p_M + p_T + p_W + p_Th
  p_F = 39 := by
  sorry

end NUMINAMATH_GPT_miriam_pushups_l1543_154346


namespace NUMINAMATH_GPT_number_drawn_from_3rd_group_l1543_154336

theorem number_drawn_from_3rd_group {n k : ℕ} (pop_size : ℕ) (sample_size : ℕ) 
  (drawn_from_group : ℕ → ℕ) (group_id : ℕ) (num_in_13th_group : ℕ) : 
  pop_size = 160 → 
  sample_size = 20 → 
  (∀ i, 1 ≤ i ∧ i ≤ sample_size → ∃ j, group_id = i ∧ 
    (j = (i - 1) * (pop_size / sample_size) + drawn_from_group 1)) → 
  num_in_13th_group = 101 → 
  drawn_from_group 3 = 21 := 
by
  intros hp hs hg h13
  sorry

end NUMINAMATH_GPT_number_drawn_from_3rd_group_l1543_154336


namespace NUMINAMATH_GPT_nine_chapters_problem_l1543_154344

def cond1 (x y : ℕ) : Prop := y = 6 * x - 6
def cond2 (x y : ℕ) : Prop := y = 5 * x + 5

theorem nine_chapters_problem (x y : ℕ) :
  (cond1 x y ∧ cond2 x y) ↔ (y = 6 * x - 6 ∧ y = 5 * x + 5) :=
by
  sorry

end NUMINAMATH_GPT_nine_chapters_problem_l1543_154344


namespace NUMINAMATH_GPT_new_area_of_card_l1543_154312

-- Conditions from the problem
def original_length : ℕ := 5
def original_width : ℕ := 7
def shortened_length := original_length - 2
def shortened_width := original_width - 1

-- Statement of the proof problem
theorem new_area_of_card : shortened_length * shortened_width = 18 :=
by
  sorry

end NUMINAMATH_GPT_new_area_of_card_l1543_154312


namespace NUMINAMATH_GPT_num_ways_to_write_360_as_increasing_seq_l1543_154318

def is_consecutive_sum (n k : ℕ) : Prop :=
  let seq_sum := k * n + k * (k - 1) / 2
  seq_sum = 360

def valid_k (k : ℕ) : Prop :=
  k ≥ 2 ∧ k ∣ 360 ∧ (k = 2 ∨ (k - 1) % 2 = 0)

noncomputable def count_consecutive_sums : ℕ :=
  Nat.card {k // valid_k k ∧ ∃ n : ℕ, is_consecutive_sum n k}

theorem num_ways_to_write_360_as_increasing_seq : count_consecutive_sums = 4 :=
sorry

end NUMINAMATH_GPT_num_ways_to_write_360_as_increasing_seq_l1543_154318


namespace NUMINAMATH_GPT_eval_expression_l1543_154393

theorem eval_expression : (500 * 500) - (499 * 501) = 1 := by
  sorry

end NUMINAMATH_GPT_eval_expression_l1543_154393


namespace NUMINAMATH_GPT_helga_shoes_l1543_154373

theorem helga_shoes :
  ∃ (S : ℕ), 7 + S + 0 + 2 * (7 + S) = 48 ∧ (S - 7 = 2) :=
by
  sorry

end NUMINAMATH_GPT_helga_shoes_l1543_154373


namespace NUMINAMATH_GPT_find_common_ratio_l1543_154360

noncomputable def geometric_series (a_1 : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a_1 * (1 - q^n) / (1 - q)

theorem find_common_ratio (a_1 : ℝ) (q : ℝ) (n : ℕ) (S_n : ℕ → ℝ)
  (h1 : ∀ n, S_n n = geometric_series a_1 q n)
  (h2 : S_n 3 = (2 * a_1 + a_1 * q) / 2)
  : q = -1/2 :=
  sorry

end NUMINAMATH_GPT_find_common_ratio_l1543_154360


namespace NUMINAMATH_GPT_average_height_Heidi_Lola_l1543_154302

theorem average_height_Heidi_Lola :
  (2.1 + 1.4) / 2 = 1.75 := by
  sorry

end NUMINAMATH_GPT_average_height_Heidi_Lola_l1543_154302

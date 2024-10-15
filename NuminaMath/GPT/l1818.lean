import Mathlib

namespace NUMINAMATH_GPT_am_gm_equality_l1818_181851

theorem am_gm_equality (a b c d : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h5 : a * b * c * d = 1) :
  a^2 + b^2 + c^2 + d^2 + a * b + a * c + a * d + b * c + b * d + c * d ≥ 10 :=
by
  sorry

end NUMINAMATH_GPT_am_gm_equality_l1818_181851


namespace NUMINAMATH_GPT_find_minimum_abs_sum_l1818_181898

noncomputable def minimum_abs_sum (α β γ : ℝ) : ℝ :=
|α| + |β| + |γ|

theorem find_minimum_abs_sum :
  ∃ α β γ : ℝ, α + β + γ = 2 ∧ α * β * γ = 4 ∧
  minimum_abs_sum α β γ = 6 := by
  sorry

end NUMINAMATH_GPT_find_minimum_abs_sum_l1818_181898


namespace NUMINAMATH_GPT_original_annual_pension_l1818_181878

theorem original_annual_pension (k x c d r s : ℝ) (h1 : k * (x + c) ^ (3/4) = k * x ^ (3/4) + r)
  (h2 : k * (x + d) ^ (3/4) = k * x ^ (3/4) + s) :
  k * x ^ (3/4) = (r - s) / (0.75 * (d - c)) :=
by sorry

end NUMINAMATH_GPT_original_annual_pension_l1818_181878


namespace NUMINAMATH_GPT_function_has_property_T_l1818_181817

noncomputable def property_T (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), a ≠ b ∧ (f a ≠ 0) ∧ (f b ≠ 0) ∧ (f a * f b = -1)

theorem function_has_property_T : property_T (fun x => 1 + x * Real.log x) :=
sorry

end NUMINAMATH_GPT_function_has_property_T_l1818_181817


namespace NUMINAMATH_GPT_cos_of_7pi_over_4_l1818_181885

theorem cos_of_7pi_over_4 : Real.cos (7 * Real.pi / 4) = 1 / Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_cos_of_7pi_over_4_l1818_181885


namespace NUMINAMATH_GPT_find_n_constant_term_l1818_181852

-- Given condition as a Lean term
def eq1 (n : ℕ) : ℕ := 2^(2*n) - (2^n + 992)

-- Prove that n = 5 fulfills the condition
theorem find_n : eq1 5 = 0 := by
  sorry

-- Given n = 5, find the constant term in the given expansion
def general_term (n r : ℕ) : ℤ := (-1)^r * (Nat.choose (2*n) r) * (n - 5*r/2)

-- Prove the constant term is 45 when n = 5
theorem constant_term : general_term 5 2 = 45 := by
  sorry

end NUMINAMATH_GPT_find_n_constant_term_l1818_181852


namespace NUMINAMATH_GPT_center_of_circle_l1818_181875

theorem center_of_circle (h k : ℝ) :
  (∀ x y : ℝ, (x - 3) ^ 2 + (y - 4) ^ 2 = 10 ↔ x ^ 2 + y ^ 2 = 6 * x + 8 * y - 15) → 
  h + k = 7 :=
sorry

end NUMINAMATH_GPT_center_of_circle_l1818_181875


namespace NUMINAMATH_GPT_initial_ratio_of_milk_to_water_l1818_181839

theorem initial_ratio_of_milk_to_water (M W : ℕ) (h1 : M + 20 = 3 * W) (h2 : M + W = 40) :
  (M : ℚ) / W = 5 / 3 := by
sorry

end NUMINAMATH_GPT_initial_ratio_of_milk_to_water_l1818_181839


namespace NUMINAMATH_GPT_find_y_l1818_181849

theorem find_y (t : ℝ) (x y : ℝ) (h1 : x = 3 - 2 * t) (h2 : y = 5 * t + 3) (h3 : x = -7) : y = 28 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_y_l1818_181849


namespace NUMINAMATH_GPT_correct_product_l1818_181880

def reverse_digits (n: ℕ) : ℕ :=
  let d1 := n / 10
  let d2 := n % 10
  d2 * 10 + d1

theorem correct_product (a b : ℕ) (h1 : 10 ≤ a ∧ a < 100) (h2 : b > 0) (h3 : reverse_digits a * b = 221) :
  a * b = 527 ∨ a * b = 923 :=
sorry

end NUMINAMATH_GPT_correct_product_l1818_181880


namespace NUMINAMATH_GPT_common_chord_length_proof_l1818_181865

-- Define the first circle equation
def first_circle (x y : ℝ) : Prop := x^2 + y^2 = 50

-- Define the second circle equation
def second_circle (x y : ℝ) : Prop := x^2 + y^2 - 12*x - 6*y + 40 = 0

-- Define the property that the length of the common chord is equal to 2 * sqrt(5)
noncomputable def common_chord_length : ℝ := 2 * Real.sqrt 5

-- The theorem statement
theorem common_chord_length_proof :
  ∀ x y : ℝ, first_circle x y → second_circle x y → common_chord_length = 2 * Real.sqrt 5 :=
by
  intros x y h1 h2
  sorry

end NUMINAMATH_GPT_common_chord_length_proof_l1818_181865


namespace NUMINAMATH_GPT_Carrie_has_50_dollars_left_l1818_181803

/-
Conditions:
1. initial_amount = 91
2. sweater_cost = 24
3. tshirt_cost = 6
4. shoes_cost = 11
-/
def initial_amount : ℕ := 91
def sweater_cost : ℕ := 24
def tshirt_cost : ℕ := 6
def shoes_cost : ℕ := 11

/-
Question:
How much money does Carrie have left?
-/
def total_spent : ℕ := sweater_cost + tshirt_cost + shoes_cost
def money_left : ℕ := initial_amount - total_spent

def proof_statement : Prop := money_left = 50

theorem Carrie_has_50_dollars_left : proof_statement :=
by
  sorry

end NUMINAMATH_GPT_Carrie_has_50_dollars_left_l1818_181803


namespace NUMINAMATH_GPT_sons_ages_l1818_181801

theorem sons_ages (m n : ℕ) (h : m * n + m + n = 34) : 
  (m = 4 ∧ n = 6) ∨ (m = 6 ∧ n = 4) :=
sorry

end NUMINAMATH_GPT_sons_ages_l1818_181801


namespace NUMINAMATH_GPT_part1_part2_part3_l1818_181862

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x - 1 - x - a * x^2

theorem part1 (x : ℝ) : f x 0 ≥ 0 :=
sorry

theorem part2 {a : ℝ} (h : ∀ x ≥ 0, f x a ≥ 0) : a ≤ 1 / 2 :=
sorry

theorem part3 (x : ℝ) (hx : x > 0) : (Real.exp x - 1) * Real.log (x + 1) > x^2 :=
sorry

end NUMINAMATH_GPT_part1_part2_part3_l1818_181862


namespace NUMINAMATH_GPT_max_value_of_d_l1818_181842

theorem max_value_of_d : ∀ (d e : ℕ), (∃ (n : ℕ), n = 70733 + 10^4 * d + e ∧ (∃ (k3 k11 : ℤ), n = 3 * k3 ∧ n = 11 * k11) ∧ d = e ∧ d ≤ 9) → d = 2 :=
by 
  -- Given conditions and goals:
  -- 1. The number has the form 7d7,33e which in numerical form is: n = 70733 + 10^4 * d + e
  -- 2. The number n is divisible by 3 and 11.
  -- 3. d and e are digits (0 ≤ d, e ≤ 9).
  -- 4. To maximize the value of d, ensure that the given conditions hold.
  -- Problem: Prove that the maximum value of d for which this holds is 2.
  sorry

end NUMINAMATH_GPT_max_value_of_d_l1818_181842


namespace NUMINAMATH_GPT_total_markup_l1818_181819

theorem total_markup (p : ℝ) (o : ℝ) (n : ℝ) (m : ℝ) : 
  p = 48 → o = 0.35 → n = 18 → m = o * p + n → m = 34.8 :=
by
  intro hp ho hn hm
  sorry

end NUMINAMATH_GPT_total_markup_l1818_181819


namespace NUMINAMATH_GPT_greg_ate_4_halves_l1818_181810

def greg_ate_halves (total_cookies : ℕ) (brad_halves : ℕ) (left_halves : ℕ) : ℕ :=
  2 * total_cookies - (brad_halves + left_halves)

theorem greg_ate_4_halves : greg_ate_halves 14 6 18 = 4 := by
  sorry

end NUMINAMATH_GPT_greg_ate_4_halves_l1818_181810


namespace NUMINAMATH_GPT_largest_five_digit_number_tens_place_l1818_181843

theorem largest_five_digit_number_tens_place :
  ∀ (n : ℕ), n = 87315 → (n % 100) / 10 = 1 := 
by
  intros n h
  sorry

end NUMINAMATH_GPT_largest_five_digit_number_tens_place_l1818_181843


namespace NUMINAMATH_GPT_vector_on_line_l1818_181897

variable {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (p q : V)

theorem vector_on_line (k : ℝ) (hpq : p ≠ q) :
  ∃ t : ℝ, k • p + (1/2 : ℝ) • q = p + t • (q - p) → k = 1/2 :=
by
  sorry

end NUMINAMATH_GPT_vector_on_line_l1818_181897


namespace NUMINAMATH_GPT_value_of_expression_l1818_181846

theorem value_of_expression : (1 * 2 * 3 * 4 * 5 * 6 : ℚ) / (1 + 2 + 3 + 4 + 5 + 6) = 240 / 7 := 
by 
  sorry

end NUMINAMATH_GPT_value_of_expression_l1818_181846


namespace NUMINAMATH_GPT_largest_number_l1818_181890

-- Define the given numbers
def A : ℝ := 0.986
def B : ℝ := 0.9859
def C : ℝ := 0.98609
def D : ℝ := 0.896
def E : ℝ := 0.8979
def F : ℝ := 0.987

-- State the theorem that F is the largest number among A, B, C, D, and E
theorem largest_number : F > A ∧ F > B ∧ F > C ∧ F > D ∧ F > E := by
  sorry

end NUMINAMATH_GPT_largest_number_l1818_181890


namespace NUMINAMATH_GPT_rice_and_grain_separation_l1818_181825

theorem rice_and_grain_separation (total_weight : ℕ) (sample_size : ℕ) (non_rice_sample : ℕ) (non_rice_in_batch : ℕ) :
  total_weight = 1524 →
  sample_size = 254 →
  non_rice_sample = 28 →
  non_rice_in_batch = total_weight * non_rice_sample / sample_size →
  non_rice_in_batch = 168 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end NUMINAMATH_GPT_rice_and_grain_separation_l1818_181825


namespace NUMINAMATH_GPT_largest_possible_b_l1818_181887

theorem largest_possible_b 
  (V : ℕ)
  (a b c : ℤ)
  (hV : V = 360)
  (h1 : 1 < c)
  (h2 : c < b)
  (h3 : b < a)
  (h4 : a * b * c = V) 
  : b = 12 := 
  sorry

end NUMINAMATH_GPT_largest_possible_b_l1818_181887


namespace NUMINAMATH_GPT_part1_l1818_181808

-- Define the sets P and Q
def P (a : ℝ) : Set ℝ := {x | 3 * a - 10 ≤ x ∧ x < 2 * a + 1}
def Q : Set ℝ := {x | |2 * x - 3| ≤ 7}

-- Define the complement of Q in ℝ
def Q_complement : Set ℝ := {x | x < -2 ∨ x > 5}

-- Define the specific value of a
def a : ℝ := 2

-- Define the specific set P when a = 2
def P_a2 : Set ℝ := {x | -4 ≤ x ∧ x < 5}

-- Define the intersection
def intersection : Set ℝ := {x | -4 ≤ x ∧ x < -2}

theorem part1 : P a ∩ Q_complement = intersection := sorry

end NUMINAMATH_GPT_part1_l1818_181808


namespace NUMINAMATH_GPT_sue_library_inventory_l1818_181823

theorem sue_library_inventory :
  let initial_books := 15
  let initial_movies := 6
  let returned_books := 8
  let returned_movies := initial_movies / 3
  let borrowed_more_books := 9
  let current_books := initial_books - returned_books + borrowed_more_books
  let current_movies := initial_movies - returned_movies
  current_books + current_movies = 20 :=
by
  -- no implementation provided
  sorry

end NUMINAMATH_GPT_sue_library_inventory_l1818_181823


namespace NUMINAMATH_GPT_subsetneq_M_N_l1818_181822

def M : Set ℝ := {-1, 1}
def N : Set ℝ := {x | (x < 0) ∨ (x > 1 / 2)}

theorem subsetneq_M_N : M ⊂ N :=
by
  sorry

end NUMINAMATH_GPT_subsetneq_M_N_l1818_181822


namespace NUMINAMATH_GPT_distance_origin_to_point_l1818_181814

theorem distance_origin_to_point : 
  let origin := (0, 0)
  let point := (8, 15)
  dist origin point = 17 :=
by
  let dist (p1 p2 : ℝ × ℝ) : ℝ :=
    Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  sorry

end NUMINAMATH_GPT_distance_origin_to_point_l1818_181814


namespace NUMINAMATH_GPT_mad_hatter_waiting_time_march_hare_waiting_time_waiting_time_l1818_181895

-- Definitions
def mad_hatter_clock_rate := 5 / 4
def march_hare_clock_rate := 5 / 6
def time_at_dormouse_clock := 5 -- 5:00 PM

-- Real time calculation based on clock rates
def real_time (clock_rate : ℚ) (clock_time : ℚ) : ℚ := clock_time * (1 / clock_rate)

-- Mad Hatter's and March Hare's arrival times in real time
def mad_hatter_real_time := real_time mad_hatter_clock_rate time_at_dormouse_clock
def march_hare_real_time := real_time march_hare_clock_rate time_at_dormouse_clock

-- Theorems to be proved
theorem mad_hatter_waiting_time : mad_hatter_real_time = 4 := sorry
theorem march_hare_waiting_time : march_hare_real_time = 6 := sorry

-- Main theorem
theorem waiting_time : march_hare_real_time - mad_hatter_real_time = 2 := sorry

end NUMINAMATH_GPT_mad_hatter_waiting_time_march_hare_waiting_time_waiting_time_l1818_181895


namespace NUMINAMATH_GPT_problem_1_problem_2_l1818_181884

-- Problem 1:
-- Given: kx^2 - 2x + 3k < 0 and the solution set is {x | x < -3 or x > -1}, prove k = -1/2
theorem problem_1 {k : ℝ} :
  (∀ x : ℝ, (kx^2 - 2*x + 3*k < 0 ↔ x < -3 ∨ x > -1)) → k = -1/2 :=
sorry

-- Problem 2:
-- Given: kx^2 - 2x + 3k < 0 and the solution set is ∅, prove 0 < k ≤ sqrt(3) / 3
theorem problem_2 {k : ℝ} :
  (∀ x : ℝ, ¬ (kx^2 - 2*x + 3*k < 0)) → 0 < k ∧ k ≤ Real.sqrt 3 / 3 :=
sorry

end NUMINAMATH_GPT_problem_1_problem_2_l1818_181884


namespace NUMINAMATH_GPT_line_parabola_intersection_l1818_181827

theorem line_parabola_intersection (k : ℝ) :
  (∀ x y, y = k * x + 1 ∧ y^2 = 4 * x → y = 1 ∧ x = 1 / 4) ∨
  (∀ x y, y = k * x + 1 ∧ y^2 = 4 * x → (k^2 * x^2 + (2 * k - 4) * x + 1 = 0) ∧ (4 * k * k - 16 * k + 16 - 4 * k * k = 0) → k = 1) :=
sorry

end NUMINAMATH_GPT_line_parabola_intersection_l1818_181827


namespace NUMINAMATH_GPT_problem_statement_l1818_181841

-- Define the conditions:
def f (x : ℚ) : ℚ := sorry

axiom f_mul (a b : ℚ) : f (a * b) = f a + f b
axiom f_int (n : ℤ) : f (n : ℚ) = (n : ℚ)

-- The problem statement:
theorem problem_statement : f (8/13) < 0 :=
sorry

end NUMINAMATH_GPT_problem_statement_l1818_181841


namespace NUMINAMATH_GPT_simplify_fraction_l1818_181855

theorem simplify_fraction (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (a^(2*b) * b^a) / (b^(2*a) * a^b) = (a / b)^b := 
by sorry

end NUMINAMATH_GPT_simplify_fraction_l1818_181855


namespace NUMINAMATH_GPT_actors_duration_l1818_181813

-- Definition of conditions
def actors_at_a_time := 5
def total_actors := 20
def total_minutes := 60

-- Main statement to prove
theorem actors_duration : total_minutes / (total_actors / actors_at_a_time) = 15 := 
by
  sorry

end NUMINAMATH_GPT_actors_duration_l1818_181813


namespace NUMINAMATH_GPT_num_isosceles_triangles_l1818_181840

theorem num_isosceles_triangles (a b : ℕ) (h1 : 2 * a + b = 27) (h2 : a > b / 2) : 
  ∃! (n : ℕ), n = 13 :=
by 
  sorry

end NUMINAMATH_GPT_num_isosceles_triangles_l1818_181840


namespace NUMINAMATH_GPT_sale_in_fourth_month_l1818_181892

variable (sale1 sale2 sale3 sale5 sale6 sale4 : ℕ)

def average_sale (total : ℕ) (months : ℕ) : ℕ := total / months

theorem sale_in_fourth_month
  (h1 : sale1 = 6435)
  (h2 : sale2 = 6927)
  (h3 : sale3 = 6855)
  (h5 : sale5 = 6562)
  (h6 : sale6 = 7391)
  (avg : average_sale (sale1 + sale2 + sale3 + sale4 + sale5 + sale6) 6 = 6900) :
  sale4 = 7230 := 
sorry

end NUMINAMATH_GPT_sale_in_fourth_month_l1818_181892


namespace NUMINAMATH_GPT_math_problem_l1818_181836

theorem math_problem
  (p q r : ℕ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r)
  (h1 : p * q + r = 47)
  (h2 : q * r + p = 47)
  (h3 : r * p + q = 47) :
  p + q + r = 48 :=
sorry

end NUMINAMATH_GPT_math_problem_l1818_181836


namespace NUMINAMATH_GPT_proof_abc_identity_l1818_181874

variable {a b c : ℝ}

theorem proof_abc_identity
  (h_ne_a : a ≠ 1) (h_ne_na : a ≠ -1)
  (h_ne_b : b ≠ 1) (h_ne_nb : b ≠ -1)
  (h_ne_c : c ≠ 1) (h_ne_nc : c ≠ -1)
  (habc : a * b + b * c + c * a = 1) :
  a / (1 - a ^ 2) + b / (1 - b ^ 2) + c / (1 - c ^ 2) = (4 * a * b * c) / (1 - a ^ 2) / (1 - b ^ 2) / (1 - c ^ 2) :=
by 
  sorry

end NUMINAMATH_GPT_proof_abc_identity_l1818_181874


namespace NUMINAMATH_GPT_part1_part2_l1818_181877

-- Part 1: Proving the solutions for (x-1)^2 = 49
theorem part1 (x : ℝ) (h : (x - 1)^2 = 49) : x = 8 ∨ x = -6 :=
sorry

-- Part 2: Proving the time for the object to reach the ground
theorem part2 (t : ℝ) (h : 4.9 * t^2 = 10) : t = 10 / 7 :=
sorry

end NUMINAMATH_GPT_part1_part2_l1818_181877


namespace NUMINAMATH_GPT_total_amount_correct_l1818_181886

noncomputable def total_amount_collected
    (single_ticket_price : ℕ)
    (couple_ticket_price : ℕ)
    (total_people : ℕ)
    (couple_tickets_sold : ℕ) : ℕ :=
  let single_tickets_sold := total_people - (couple_tickets_sold * 2)
  let amount_from_couple_tickets := couple_tickets_sold * couple_ticket_price
  let amount_from_single_tickets := single_tickets_sold * single_ticket_price
  amount_from_couple_tickets + amount_from_single_tickets

theorem total_amount_correct :
  total_amount_collected 20 35 128 16 = 2480 := by
  sorry

end NUMINAMATH_GPT_total_amount_correct_l1818_181886


namespace NUMINAMATH_GPT_solve_inequality_l1818_181882

theorem solve_inequality (x : ℝ) (h : x ≠ 1) : (x / (x - 1) ≥ 2 * x) ↔ (x ≤ 0 ∨ (1 < x ∧ x ≤ 3 / 2)) :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l1818_181882


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l1818_181850

variable {m : ℝ}

theorem necessary_but_not_sufficient_condition (h : (∃ x1 x2 : ℝ, (x1 ≠ 0 ∧ x1 = -x2) ∧ (x1^2 + x1 + m^2 - 1 = 0))): 
  0 < m ∧ m < 1 :=
by 
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l1818_181850


namespace NUMINAMATH_GPT_reducible_fraction_least_n_l1818_181818

theorem reducible_fraction_least_n : ∃ n : ℕ, (0 < n) ∧ (n-15 > 0) ∧ (gcd (n-15) (3*n+4) > 1) ∧
  (∀ m : ℕ, (0 < m) ∧ (m-15 > 0) ∧ (gcd (m-15) (3*m+4) > 1) → n ≤ m) :=
by
  sorry

end NUMINAMATH_GPT_reducible_fraction_least_n_l1818_181818


namespace NUMINAMATH_GPT_initial_amount_l1818_181873

theorem initial_amount 
  (M : ℝ)
  (h1 : M * (3 / 5) * (2 / 3) * (3 / 4) * (4 / 7) = 700) : 
  M = 24500 / 6 :=
by sorry

end NUMINAMATH_GPT_initial_amount_l1818_181873


namespace NUMINAMATH_GPT_smallest_rel_prime_210_l1818_181832

theorem smallest_rel_prime_210 : ∃ (y : ℕ), y > 1 ∧ Nat.gcd y 210 = 1 ∧ (∀ z : ℕ, z > 1 ∧ Nat.gcd z 210 = 1 → y ≤ z) ∧ y = 11 :=
by {
  sorry -- proof to be filled in
}

end NUMINAMATH_GPT_smallest_rel_prime_210_l1818_181832


namespace NUMINAMATH_GPT_class_size_l1818_181891

def S : ℝ := 30

theorem class_size (total percent_dogs_videogames percent_dogs_movies number_students_prefer_dogs : ℝ)
  (h1 : percent_dogs_videogames = 0.5)
  (h2 : percent_dogs_movies = 0.1)
  (h3 : number_students_prefer_dogs = 18)
  (h4 : total * (percent_dogs_videogames + percent_dogs_movies) = number_students_prefer_dogs) :
  total = S :=
by
  sorry

end NUMINAMATH_GPT_class_size_l1818_181891


namespace NUMINAMATH_GPT_change_in_expression_l1818_181881

variables (x b : ℝ) (hb : 0 < b)

theorem change_in_expression : (b * x)^2 - 5 - (x^2 - 5) = (b^2 - 1) * x^2 :=
by sorry

end NUMINAMATH_GPT_change_in_expression_l1818_181881


namespace NUMINAMATH_GPT_largest_angle_in_pentagon_l1818_181871

theorem largest_angle_in_pentagon (A B C D E : ℝ) 
    (hA : A = 60) 
    (hB : B = 85) 
    (hCD : C = D) 
    (hE : E = 2 * C + 15) 
    (sum_angles : A + B + C + D + E = 540) : 
    E = 205 := 
by 
    sorry

end NUMINAMATH_GPT_largest_angle_in_pentagon_l1818_181871


namespace NUMINAMATH_GPT_sandwich_and_soda_cost_l1818_181883

theorem sandwich_and_soda_cost:
  let sandwich_cost := 4
  let soda_cost := 1
  let num_sandwiches := 6
  let num_sodas := 10
  let total_cost := num_sandwiches * sandwich_cost + num_sodas * soda_cost
  total_cost = 34 := 
by 
  sorry

end NUMINAMATH_GPT_sandwich_and_soda_cost_l1818_181883


namespace NUMINAMATH_GPT_pairs_of_participants_l1818_181857

theorem pairs_of_participants (n : Nat) (h : n = 12) : (Nat.choose n 2) = 66 := by
  sorry

end NUMINAMATH_GPT_pairs_of_participants_l1818_181857


namespace NUMINAMATH_GPT_workers_cut_down_correct_l1818_181816

def initial_oak_trees : ℕ := 9
def remaining_oak_trees : ℕ := 7
def cut_down_oak_trees : ℕ := initial_oak_trees - remaining_oak_trees

theorem workers_cut_down_correct : cut_down_oak_trees = 2 := by
  sorry

end NUMINAMATH_GPT_workers_cut_down_correct_l1818_181816


namespace NUMINAMATH_GPT_solve_for_x_l1818_181860

theorem solve_for_x (x : ℝ) : (0.25 * x = 0.15 * 1500 - 20) → x = 820 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_solve_for_x_l1818_181860


namespace NUMINAMATH_GPT_a_must_be_negative_l1818_181847

theorem a_must_be_negative (a b : ℝ) (h1 : b > 0) (h2 : a / b < -2 / 3) : a < 0 :=
sorry

end NUMINAMATH_GPT_a_must_be_negative_l1818_181847


namespace NUMINAMATH_GPT_sin_six_theta_l1818_181876

theorem sin_six_theta (θ : ℝ) (h : Complex.exp (Complex.I * θ) = (3 + Complex.I * Real.sqrt 8) / 5) : 
  Real.sin (6 * θ) = - (630 * Real.sqrt 8) / 15625 := by
  sorry

end NUMINAMATH_GPT_sin_six_theta_l1818_181876


namespace NUMINAMATH_GPT_sum_series_eq_l1818_181835

theorem sum_series_eq :
  ∑' n : ℕ, (4 * (n + 1) - 3) / 3^(n + 1) = 3 / 2 :=
sorry

end NUMINAMATH_GPT_sum_series_eq_l1818_181835


namespace NUMINAMATH_GPT_probability_square_not_touching_outer_edge_l1818_181848

theorem probability_square_not_touching_outer_edge :
  let total_squares := 10 * 10
  let perimeter_squares := 10 + 10 + (10 - 2) + (10 - 2)
  let non_perimeter_squares := total_squares - perimeter_squares
  (non_perimeter_squares / total_squares) = (16 / 25) :=
by
  let total_squares := 10 * 10
  let perimeter_squares := 10 + 10 + (10 - 2) + (10 - 2)
  let non_perimeter_squares := total_squares - perimeter_squares
  have h : non_perimeter_squares / total_squares = 16 / 25 := by sorry
  exact h

end NUMINAMATH_GPT_probability_square_not_touching_outer_edge_l1818_181848


namespace NUMINAMATH_GPT_circle_symmetric_to_line_l1818_181859

theorem circle_symmetric_to_line (m : ℝ) :
  (∃ (x y : ℝ), (x^2 + y^2 - m * x + 3 * y + 3 = 0) ∧ (m * x + y - m = 0))
  → m = 3 :=
by
  sorry

end NUMINAMATH_GPT_circle_symmetric_to_line_l1818_181859


namespace NUMINAMATH_GPT_avg_five_probability_l1818_181872

/- Define the set of natural numbers from 1 to 9. -/
def S : Finset ℕ := Finset.range 10 \ {0}

/- Define the binomial coefficient for choosing 7 out of 9. -/
def choose_7_9 : ℕ := Nat.choose 9 7

/- Define the condition for the sum of chosen numbers to be 35. -/
def sum_is_35 (s : Finset ℕ) : Prop := s.sum id = 35

/- Number of ways to choose 3 pairs that sum to 10 and include number 5 - means sum should be 35-/
def ways_3_pairs_and_5 : ℕ := 4

/- Probability calculation. -/
def prob_sum_is_35 : ℚ := (ways_3_pairs_and_5: ℚ) / (choose_7_9: ℚ)

theorem avg_five_probability : prob_sum_is_35 = 1 / 9 := by
  sorry

end NUMINAMATH_GPT_avg_five_probability_l1818_181872


namespace NUMINAMATH_GPT_correct_operations_result_l1818_181806

/-
Pat intended to multiply a number by 8 but accidentally divided by 8.
Pat then meant to add 20 to the result but instead subtracted 20.
After these errors, the final outcome was 12.
Prove that if Pat had performed the correct operations, the final outcome would have been 2068.
-/

theorem correct_operations_result (n : ℕ) (h1 : n / 8 - 20 = 12) : 8 * n + 20 = 2068 :=
by
  sorry

end NUMINAMATH_GPT_correct_operations_result_l1818_181806


namespace NUMINAMATH_GPT_express_y_in_terms_of_x_l1818_181888

variable (x y : ℝ)

theorem express_y_in_terms_of_x (h : x + y = -1) : y = -1 - x := 
by 
  sorry

end NUMINAMATH_GPT_express_y_in_terms_of_x_l1818_181888


namespace NUMINAMATH_GPT_range_of_3t_plus_s_l1818_181802

noncomputable def f : ℝ → ℝ := sorry

def is_increasing (f : ℝ → ℝ) := ∀ x y, x ≤ y → f x ≤ f y

def symmetric_about (f : ℝ → ℝ) (a b : ℝ) := 
  ∀ x, f (x - a) = b - f (a - x)

def satisfies_inequality (s t : ℝ) (f : ℝ → ℝ) := 
  f (s^2 - 2*s) ≥ -f (2*t - t^2)

def in_interval (s : ℝ) := 1 ≤ s ∧ s ≤ 4

theorem range_of_3t_plus_s (f : ℝ → ℝ) :
  is_increasing f ∧ symmetric_about f 3 0 →
  (∀ s t, satisfies_inequality s t f → in_interval s → -2 ≤ 3 * t + s ∧ 3 * t + s ≤ 16) :=
sorry

end NUMINAMATH_GPT_range_of_3t_plus_s_l1818_181802


namespace NUMINAMATH_GPT_least_sugar_pounds_l1818_181861

theorem least_sugar_pounds (f s : ℕ) (hf1 : f ≥ 7 + s / 2) (hf2 : f ≤ 3 * s) : s ≥ 3 :=
by
  have h : (5 * s) / 2 ≥ 7 := sorry
  have s_ge_3 : s ≥ 3 := sorry
  exact s_ge_3

end NUMINAMATH_GPT_least_sugar_pounds_l1818_181861


namespace NUMINAMATH_GPT_find_original_number_l1818_181805

theorem find_original_number (x : ℝ) (h : 0.5 * x = 30) : x = 60 :=
sorry

end NUMINAMATH_GPT_find_original_number_l1818_181805


namespace NUMINAMATH_GPT_ratio_of_larger_to_smaller_l1818_181831

variable {x y : ℝ}

-- Condition for x and y being positive and x > y
axiom x_pos : 0 < x
axiom y_pos : 0 < y
axiom x_gt_y : x > y

-- Condition for sum and difference relationship
axiom sum_diff_relation : x + y = 7 * (x - y)

-- Theorem: Ratio of the larger number to the smaller number is 2
theorem ratio_of_larger_to_smaller : x / y = 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_larger_to_smaller_l1818_181831


namespace NUMINAMATH_GPT_range_of_a_l1818_181826

theorem range_of_a (x a : ℝ) (p : |x - 2| < 3) (q : 0 < x ∧ x < a) :
  (0 < a ∧ a ≤ 5) := 
sorry

end NUMINAMATH_GPT_range_of_a_l1818_181826


namespace NUMINAMATH_GPT_sum_of_roots_l1818_181838

theorem sum_of_roots (x1 x2 : ℝ) (h : x1 * x2 = -3) (hx1 : x1 + x2 = 2) :
  x1 + x2 = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_sum_of_roots_l1818_181838


namespace NUMINAMATH_GPT_max_female_students_min_people_in_group_l1818_181867

-- Problem 1: Given z = 4, the maximum number of female students is 6
theorem max_female_students (x y : ℕ) (h1 : x > y) (h2 : y > 4) (h3 : x < 8) : y <= 6 :=
sorry

-- Problem 2: The minimum number of people in the group is 12
theorem min_people_in_group (x y z : ℕ) (h1 : x > y) (h2 : y > z) (h3 : 2 * z > x) : 12 <= x + y + z :=
sorry

end NUMINAMATH_GPT_max_female_students_min_people_in_group_l1818_181867


namespace NUMINAMATH_GPT_determine_p_l1818_181856

noncomputable def roots (p : ℝ) : ℝ × ℝ :=
  let discr := p ^ 2 - 48
  ((-p + Real.sqrt discr) / 2, (-p - Real.sqrt discr) / 2)

theorem determine_p (p : ℝ) :
  let (x1, x2) := roots p
  (x1 - x2 = 1) → (p = 7 ∨ p = -7) :=
by
  intros
  sorry

end NUMINAMATH_GPT_determine_p_l1818_181856


namespace NUMINAMATH_GPT_johns_height_l1818_181870

theorem johns_height
  (L R J : ℕ)
  (h1 : J = L + 15)
  (h2 : J = R - 6)
  (h3 : L + R = 295) :
  J = 152 :=
by sorry

end NUMINAMATH_GPT_johns_height_l1818_181870


namespace NUMINAMATH_GPT_inequality_proof_l1818_181866

variable (a b c : ℝ)
variable (h_pos : a > 0) (h_pos2 : b > 0) (h_pos3 : c > 0)
variable (h_sum : a + b + c = 1)

theorem inequality_proof :
  (a^2 + b^2 + c^2) * (a / (b + c) + b / (a + c) + c / (a + b)) > 1 / 2 := by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1818_181866


namespace NUMINAMATH_GPT_products_selling_less_than_1000_l1818_181833

theorem products_selling_less_than_1000 (N: ℕ) 
  (total_products: ℕ := 25) 
  (average_price: ℤ := 1200) 
  (min_price: ℤ := 400) 
  (max_price: ℤ := 12000) 
  (total_revenue := total_products * average_price) 
  (revenue_from_expensive: ℤ := max_price):
  12000 + (24 - N) * 1000 + N * 400 = 30000 ↔ N = 10 :=
by
  sorry

end NUMINAMATH_GPT_products_selling_less_than_1000_l1818_181833


namespace NUMINAMATH_GPT_original_number_is_25_l1818_181896

theorem original_number_is_25 (x : ℕ) (h : ∃ n : ℕ, (x^2 - 600)^n = x) : x = 25 :=
sorry

end NUMINAMATH_GPT_original_number_is_25_l1818_181896


namespace NUMINAMATH_GPT_train_length_calculation_l1818_181829

theorem train_length_calculation 
  (bridge_length : ℝ) (crossing_time : ℝ) (train_speed_kmph : ℝ) 
  (h_bridge_length : bridge_length = 150)
  (h_crossing_time : crossing_time = 25) 
  (h_train_speed_kmph : train_speed_kmph = 57.6) : 
  ∃ train_length, train_length = 250 :=
by
  sorry

end NUMINAMATH_GPT_train_length_calculation_l1818_181829


namespace NUMINAMATH_GPT_packs_to_purchase_l1818_181807

theorem packs_to_purchase {n m k : ℕ} (h : 8 * n + 15 * m + 30 * k = 135) : n + m + k = 5 :=
sorry

end NUMINAMATH_GPT_packs_to_purchase_l1818_181807


namespace NUMINAMATH_GPT_max_knights_cannot_be_all_liars_l1818_181879

-- Define the conditions of the problem
structure Student :=
  (is_knight : Bool)
  (statement : String)

-- Define the function to check the truthfulness of statements
def is_truthful (s : Student) (conditions : List Student) : Bool :=
  -- Define how to check the statement based on conditions
  sorry

-- The maximum number of knights
theorem max_knights (N : ℕ) (students : List Student) (cond : ∀ s ∈ students, is_truthful s students = true ↔ s.is_knight) :
  ∃ M, M = N := by
  sorry

-- The school cannot be made up entirely of liars
theorem cannot_be_all_liars (N : ℕ) (students : List Student) (cond : ∀ s ∈ students, ¬is_truthful s students) :
  false := by
  sorry

end NUMINAMATH_GPT_max_knights_cannot_be_all_liars_l1818_181879


namespace NUMINAMATH_GPT_initial_average_age_l1818_181809

theorem initial_average_age (A : ℕ) (h1 : ∀ x : ℕ, 10 * A = 10 * A)
  (h2 : 5 * 17 + 10 * A = 15 * (A + 1)) : A = 14 :=
by 
  sorry

end NUMINAMATH_GPT_initial_average_age_l1818_181809


namespace NUMINAMATH_GPT_line_single_point_not_necessarily_tangent_l1818_181820

-- Define a curve
def curve : Type := ℝ → ℝ

-- Define a line
def line (m b : ℝ) : curve := λ x => m * x + b

-- Define a point of intersection
def intersects_at (l : curve) (c : curve) (x : ℝ) : Prop :=
  l x = c x

-- Define the property of having exactly one common point
def has_single_intersection (l : curve) (c : curve) : Prop :=
  ∃ x, ∀ y ≠ x, l y ≠ c y

-- Define the tangent line property
def is_tangent (l : curve) (c : curve) (x : ℝ) : Prop :=
  ∀ ε > 0, ∃ δ > 0, ∀ h, abs h < δ → abs ((c (x + h) - c x) / h - (l (x + h) - l x) / h) < ε

-- The proof statement: There exists a curve c and a line l such that l has exactly one intersection point with c, but l is not necessarily a tangent to c.
theorem line_single_point_not_necessarily_tangent :
  ∃ c : curve, ∃ l : curve, has_single_intersection l c ∧ ∃ x, ¬ is_tangent l c x :=
sorry

end NUMINAMATH_GPT_line_single_point_not_necessarily_tangent_l1818_181820


namespace NUMINAMATH_GPT_celine_erasers_collected_l1818_181894

theorem celine_erasers_collected (G C J E : ℕ) 
    (hC : C = 2 * G)
    (hJ : J = 4 * G)
    (hE : E = 12 * G)
    (h_total : G + C + J + E = 151) : 
    C = 16 := 
by 
  -- Proof steps skipped, proof body not required as per instructions
  sorry

end NUMINAMATH_GPT_celine_erasers_collected_l1818_181894


namespace NUMINAMATH_GPT_roots_of_quadratic_equation_l1818_181899

theorem roots_of_quadratic_equation (a b c r s : ℝ) 
  (hr : a ≠ 0)
  (h : a * r^2 + b * r - c = 0)
  (h' : a * s^2 + b * s - c = 0)
  :
  (1 / r^2) + (1 / s^2) = (b^2 + 2 * a * c) / c^2 :=
by
  sorry

end NUMINAMATH_GPT_roots_of_quadratic_equation_l1818_181899


namespace NUMINAMATH_GPT_length_of_track_l1818_181889

-- Conditions as definitions
def Janet_runs (m : Nat) := m = 120
def Leah_distance_after_first_meeting (x : Nat) (m : Nat) := m = (x / 2 - 120 + 200)
def Janet_distance_after_first_meeting (x : Nat) (m : Nat) := m = (x - 120 + (x - (x / 2 + 80)))

-- Questions and answers combined in proof statement
theorem length_of_track (x : Nat) (hx : Janet_runs 120) (hy : Leah_distance_after_first_meeting x 280) (hz : Janet_distance_after_first_meeting x (x / 2 - 40)) :
  x = 480 :=
sorry

end NUMINAMATH_GPT_length_of_track_l1818_181889


namespace NUMINAMATH_GPT_smallest_number_l1818_181864

theorem smallest_number:
    let a := 3.25
    let b := 3.26   -- 326% in decimal
    let c := 3.2    -- 3 1/5 in decimal
    let d := 3.75   -- 15/4 in decimal
    c < a ∧ c < b ∧ c < d :=
by
    sorry

end NUMINAMATH_GPT_smallest_number_l1818_181864


namespace NUMINAMATH_GPT_alex_basketball_points_l1818_181828

theorem alex_basketball_points (f t s : ℕ) 
  (h : f + t + s = 40) 
  (points_scored : ℝ := 0.8 * f + 0.3 * t + s) :
  points_scored = 28 :=
sorry

end NUMINAMATH_GPT_alex_basketball_points_l1818_181828


namespace NUMINAMATH_GPT_sum_of_edges_112_l1818_181868

-- Define the problem parameters
def volume (a b c : ℝ) : ℝ := a * b * c
def surface_area (a b c : ℝ) : ℝ := 2 * (a * b + b * c + c * a)
def sum_of_edges (a b c : ℝ) : ℝ := 4 * (a + b + c)

-- The main theorem 
theorem sum_of_edges_112
  (b s : ℝ) (h1 : volume (b / s) b (b * s) = 512)
  (h2 : surface_area (b / s) b (b * s) = 448)
  (h3 : 0 < b ∧ 0 < s) : 
  sum_of_edges (b / s) b (b * s) = 112 :=
sorry

end NUMINAMATH_GPT_sum_of_edges_112_l1818_181868


namespace NUMINAMATH_GPT_expression_equals_500_l1818_181812

theorem expression_equals_500 :
  let A := 5 * 99 + 1
  let B := 100 + 25 * 4
  let C := 88 * 4 + 37 * 4
  let D := 100 * 0 * 5
  C = 500 :=
by
  let A := 5 * 99 + 1
  let B := 100 + 25 * 4
  let C := 88 * 4 + 37 * 4
  let D := 100 * 0 * 5
  sorry

end NUMINAMATH_GPT_expression_equals_500_l1818_181812


namespace NUMINAMATH_GPT_class_2_3_tree_count_total_tree_count_l1818_181844

-- Definitions based on the given conditions
def class_2_5_trees := 142
def class_2_3_trees := class_2_5_trees - 18

-- Statements to be proved
theorem class_2_3_tree_count :
  class_2_3_trees = 124 :=
sorry

theorem total_tree_count :
  class_2_5_trees + class_2_3_trees = 266 :=
sorry

end NUMINAMATH_GPT_class_2_3_tree_count_total_tree_count_l1818_181844


namespace NUMINAMATH_GPT_dog_years_second_year_l1818_181837

theorem dog_years_second_year (human_years : ℕ) :
  15 + human_years + 5 * 8 = 64 →
  human_years = 9 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_dog_years_second_year_l1818_181837


namespace NUMINAMATH_GPT_min_colored_cells_65x65_l1818_181800

def grid_size : ℕ := 65
def total_cells : ℕ := grid_size * grid_size

-- Define a function that calculates the minimum number of colored cells needed
noncomputable def min_colored_cells_needed (N: ℕ) : ℕ := (N * N) / 3

-- The main theorem stating the proof problem
theorem min_colored_cells_65x65 (H: grid_size = 65) : 
  min_colored_cells_needed grid_size = 1408 :=
by {
  sorry
}

end NUMINAMATH_GPT_min_colored_cells_65x65_l1818_181800


namespace NUMINAMATH_GPT_ratio_simplified_l1818_181858

theorem ratio_simplified (total finished : ℕ) (h_total : total = 15) (h_finished : finished = 6) :
  (total - finished) / (Nat.gcd (total - finished) finished) = 3 ∧ finished / (Nat.gcd (total - finished) finished) = 2 := by
  sorry

end NUMINAMATH_GPT_ratio_simplified_l1818_181858


namespace NUMINAMATH_GPT_find_smallest_w_l1818_181815

theorem find_smallest_w (w : ℕ) (h : 0 < w) : 
  (∀ k, k = 2^5 ∨ k = 3^3 ∨ k = 12^2 → (k ∣ (936 * w))) ↔ w = 36 := by 
  sorry

end NUMINAMATH_GPT_find_smallest_w_l1818_181815


namespace NUMINAMATH_GPT_sum_a_b_c_d_eq_nine_l1818_181893

theorem sum_a_b_c_d_eq_nine
  (a b c d : ℤ)
  (h : (Polynomial.X ^ 2 + (Polynomial.C a) * Polynomial.X + Polynomial.C b) *
       (Polynomial.X ^ 2 + (Polynomial.C c) * Polynomial.X + Polynomial.C d) =
       Polynomial.X ^ 4 + 2 * Polynomial.X ^ 3 + Polynomial.X ^ 2 + 11 * Polynomial.X + 6) :
  a + b + c + d = 9 :=
by
  sorry

end NUMINAMATH_GPT_sum_a_b_c_d_eq_nine_l1818_181893


namespace NUMINAMATH_GPT_cos_comp_l1818_181824

open Real

theorem cos_comp {a b c : ℝ} (h1 : a = cos (3 / 2)) (h2 : b = -cos (7 / 4)) (h3 : c = sin (1 / 10)) : 
  a < c ∧ c < b := 
by
  -- Assume the hypotheses
  sorry

end NUMINAMATH_GPT_cos_comp_l1818_181824


namespace NUMINAMATH_GPT_beth_score_l1818_181834

-- Conditions
variables (B : ℕ)  -- Beth's points are some number.
def jan_points := 10 -- Jan scored 10 points.
def judy_points := 8 -- Judy scored 8 points.
def angel_points := 11 -- Angel scored 11 points.

-- First team has 3 more points than the second team
def first_team_points := B + jan_points
def second_team_points := judy_points + angel_points
def first_team_more_than_second := first_team_points = second_team_points + 3

-- Statement: Prove that B = 12
theorem beth_score : first_team_more_than_second → B = 12 :=
by
  -- Proof will be provided here
  sorry

end NUMINAMATH_GPT_beth_score_l1818_181834


namespace NUMINAMATH_GPT_smallest_number_meeting_both_conditions_l1818_181830

theorem smallest_number_meeting_both_conditions :
  ∃ n, (n = 2019) ∧
    (∃ a b c d e f : ℕ,
      n = a^4 + b^4 + c^4 + d^4 + e^4 ∧
      a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
      b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
      c ≠ d ∧ c ≠ e ∧
      d ≠ e ∧
      a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0) ∧
    (∃ x y z u v w : ℕ,
      y = x + 1 ∧ z = x + 2 ∧ u = x + 3 ∧ v = x + 4 ∧ w = x + 5 ∧
      n = x + y + z + u + v + w) ∧
    (¬ ∃ m, m < 2019 ∧
      (∃ a b c d e f : ℕ,
        m = a^4 + b^4 + c^4 + d^4 + e^4 ∧
        a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
        b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
        c ≠ d ∧ c ≠ e ∧
        d ≠ e ∧
        a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0) ∧
      (∃ x y z u v w : ℕ,
        y = x + 1 ∧ z = x + 2 ∧ u = x + 3 ∧ v = x + 4 ∧ w = x + 5 ∧
        m = x + y + z + u + v + w)) :=
by
  sorry

end NUMINAMATH_GPT_smallest_number_meeting_both_conditions_l1818_181830


namespace NUMINAMATH_GPT_ramon_twice_loui_age_in_future_l1818_181845

theorem ramon_twice_loui_age_in_future : 
  ∀ (x : ℕ), 
  (∀ t : ℕ, t = 23 → 
            t * 2 = 46 → 
            ∀ r : ℕ, r = 26 → 
                      26 + x = 46 → 
                      x = 20) := 
by sorry

end NUMINAMATH_GPT_ramon_twice_loui_age_in_future_l1818_181845


namespace NUMINAMATH_GPT_solution_10_digit_divisible_by_72_l1818_181869

def attach_digits_to_divisible_72 : Prop :=
  ∃ (a d : ℕ), (a < 10) ∧ (d < 10) ∧ a * 10^9 + 20222023 * 10 + d = 3202220232 ∧ (3202220232 % 72 = 0)

theorem solution_10_digit_divisible_by_72 : attach_digits_to_divisible_72 :=
  sorry

end NUMINAMATH_GPT_solution_10_digit_divisible_by_72_l1818_181869


namespace NUMINAMATH_GPT_smallest_positive_debt_resolvable_l1818_181811

theorem smallest_positive_debt_resolvable :
  ∃ (p g : ℤ), 400 * p + 280 * g = 800 :=
sorry

end NUMINAMATH_GPT_smallest_positive_debt_resolvable_l1818_181811


namespace NUMINAMATH_GPT_M_intersection_N_l1818_181804

noncomputable def M := {x : ℝ | 0 ≤ x ∧ x < 16}
noncomputable def N := {x : ℝ | x ≥ 1 / 3}

theorem M_intersection_N :
  (M ∩ N) = {x : ℝ | 1 / 3 ≤ x ∧ x < 16} := by
sorry

end NUMINAMATH_GPT_M_intersection_N_l1818_181804


namespace NUMINAMATH_GPT_initial_number_of_persons_l1818_181853

theorem initial_number_of_persons (n : ℕ) (h1 : ∀ n, (2.5 : ℝ) * n = 20) : n = 8 := sorry

end NUMINAMATH_GPT_initial_number_of_persons_l1818_181853


namespace NUMINAMATH_GPT_original_number_div_eq_l1818_181854

theorem original_number_div_eq (h : 204 / 12.75 = 16) : 2.04 / 1.6 = 1.275 :=
by sorry

end NUMINAMATH_GPT_original_number_div_eq_l1818_181854


namespace NUMINAMATH_GPT_value_of_x_l1818_181821

theorem value_of_x (x y : ℕ) (h1 : x / y = 8 / 3) (h2 : y = 27) : x = 72 :=
by
  sorry

end NUMINAMATH_GPT_value_of_x_l1818_181821


namespace NUMINAMATH_GPT_infinite_power_tower_solution_l1818_181863

theorem infinite_power_tower_solution (x : ℝ) (y : ℝ) (h1 : y = x ^ x ^ x ^ x ^ x ^ x ^ x ^ x ^ x ^ x) (h2 : y = 4) : x = Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_infinite_power_tower_solution_l1818_181863

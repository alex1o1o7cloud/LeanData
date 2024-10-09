import Mathlib

namespace zeros_of_g_l1825_182567

theorem zeros_of_g (a b : ℝ) (h : 2 * a + b = 0) :
  (∃ x : ℝ, (b * x^2 - a * x = 0) ∧ (x = 0 ∨ x = -1 / 2)) :=
by
  sorry

end zeros_of_g_l1825_182567


namespace solution_set_equiv_l1825_182550

def solution_set (x : ℝ) : Prop := 2 * x - 6 < 0

theorem solution_set_equiv (x : ℝ) : solution_set x ↔ x < 3 := by
  sorry

end solution_set_equiv_l1825_182550


namespace total_distance_covered_l1825_182566

theorem total_distance_covered :
  ∀ (r j w total : ℝ),
    r = 40 →
    j = (3 / 5) * r →
    w = 5 * j →
    total = r + j + w →
    total = 184 := by
  sorry

end total_distance_covered_l1825_182566


namespace cone_generatrix_length_is_2sqrt2_l1825_182556

noncomputable def cone_generatrix_length (r : ℝ) : ℝ :=
  let C := 2 * Real.pi * r
  let l := (2 * Real.pi * r) / Real.pi
  l

theorem cone_generatrix_length_is_2sqrt2 :
  cone_generatrix_length (Real.sqrt 2) = 2 * Real.sqrt 2 :=
by
  sorry

end cone_generatrix_length_is_2sqrt2_l1825_182556


namespace minimum_additional_small_bottles_needed_l1825_182545

-- Definitions from the problem conditions
def small_bottle_volume : ℕ := 45
def large_bottle_total_volume : ℕ := 600
def initial_volume_in_large_bottle : ℕ := 90

-- The proof problem: How many more small bottles does Jasmine need to fill the large bottle?
theorem minimum_additional_small_bottles_needed : 
  (large_bottle_total_volume - initial_volume_in_large_bottle + small_bottle_volume - 1) / small_bottle_volume = 12 := 
by 
  sorry

end minimum_additional_small_bottles_needed_l1825_182545


namespace direct_proportion_function_l1825_182575

theorem direct_proportion_function (k : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = k * x) (h2 : f 3 = 6) : ∀ x, f x = 2 * x := by
  sorry

end direct_proportion_function_l1825_182575


namespace speed_of_train_A_l1825_182508

noncomputable def train_speed_A (V_B : ℝ) (T_A T_B : ℝ) : ℝ :=
  (T_B / T_A) * V_B

theorem speed_of_train_A : train_speed_A 165 9 4 = 73.33 :=
by
  sorry

end speed_of_train_A_l1825_182508


namespace least_number_modular_l1825_182591

theorem least_number_modular 
  (n : ℕ)
  (h1 : n % 34 = 4)
  (h2 : n % 48 = 6)
  (h3 : n % 5 = 2) : n = 4082 :=
by
  sorry

end least_number_modular_l1825_182591


namespace common_ratio_is_two_l1825_182510

-- Geometric sequence definition
noncomputable def common_ratio (n : ℕ) (a : ℕ → ℝ) : ℝ :=
a 2 / a 1

-- The sequence has 10 terms
def ten_terms (a : ℕ → ℝ) : Prop :=
∀ n, 1 ≤ n ∧ n ≤ 10

-- The product of the odd terms is 2
def product_of_odd_terms (a : ℕ → ℝ) : Prop :=
(a 1) * (a 3) * (a 5) * (a 7) * (a 9) = 2

-- The product of the even terms is 64
def product_of_even_terms (a : ℕ → ℝ) : Prop :=
(a 2) * (a 4) * (a 6) * (a 8) * (a 10) = 64

-- The problem statement to prove that the common ratio q is 2
theorem common_ratio_is_two (a : ℕ → ℝ) (q : ℝ) (h1 : ten_terms a) 
(h2 : product_of_odd_terms a) (h3 : product_of_even_terms a) : q = 2 :=
by {
  sorry
}

end common_ratio_is_two_l1825_182510


namespace capacity_of_smaller_bucket_l1825_182528

theorem capacity_of_smaller_bucket (x : ℕ) (h1 : x < 5) (h2 : 5 - x = 2) : x = 3 := by
  sorry

end capacity_of_smaller_bucket_l1825_182528


namespace quadratic_eq_transformed_l1825_182576

-- Define the given quadratic equation
def quadratic_eq (x : ℝ) : Prop := x^2 - 2 * x - 7 = 0

-- Define the form to transform to using completing the square method
def transformed_eq (x : ℝ) : Prop := (x - 1)^2 = 8

-- The theorem to be proved
theorem quadratic_eq_transformed (x : ℝ) :
  quadratic_eq x → transformed_eq x :=
by
  intros h
  -- here we would use steps of completing the square to transform the equation
  sorry

end quadratic_eq_transformed_l1825_182576


namespace units_digit_7_pow_1995_l1825_182546

theorem units_digit_7_pow_1995 : 
  ∃ a : ℕ, a = 3 ∧ ∀ n : ℕ, (7^n % 10 = a) → ((n % 4) + 1 = 3) := 
by
  sorry

end units_digit_7_pow_1995_l1825_182546


namespace infinite_series_sum_eq_33_div_8_l1825_182503

noncomputable def infinite_series_sum: ℝ :=
  ∑' n: ℕ, n^3 / (3^n : ℝ)

theorem infinite_series_sum_eq_33_div_8:
  infinite_series_sum = 33 / 8 :=
sorry

end infinite_series_sum_eq_33_div_8_l1825_182503


namespace cos_beta_half_l1825_182590

theorem cos_beta_half (α β : ℝ) (hα_ac : 0 < α ∧ α < π / 2) (hβ_ac : 0 < β ∧ β < π / 2) 
  (h1 : Real.tan α = 4 * Real.sqrt 3) (h2 : Real.cos (α + β) = -11 / 14) : 
  Real.cos β = 1 / 2 :=
by
  sorry

end cos_beta_half_l1825_182590


namespace determine_a_l1825_182562

noncomputable def f (x a : ℝ) : ℝ := abs (2 * x - a) + a

theorem determine_a : (∃ a: ℝ, (∀ x: ℝ, -2 ≤ x ∧ x ≤ 3 → f x a ≤ 6) ∧ ∀ x: ℝ, f x a ≤ 6 → -2 ≤ x ∧ x ≤ 3) ↔ a = 1 :=
by
  sorry

end determine_a_l1825_182562


namespace find_real_number_x_l1825_182502

theorem find_real_number_x 
    (x : ℝ) 
    (i : ℂ) 
    (h_imaginary_unit : i*i = -1) 
    (h_equation : (1 - 2*i)*(x + i) = 4 - 3*i) : 
    x = 2 := 
by
  sorry

end find_real_number_x_l1825_182502


namespace quadratic_roots_diff_by_2_l1825_182558

theorem quadratic_roots_diff_by_2 (q : ℝ) (hq : 0 < q) :
  (∃ r1 r2 : ℝ, r1 ≠ r2 ∧ (r1 - r2 = 2 ∨ r2 - r1 = 2) ∧ r1 ^ 2 + (2 * q - 1) * r1 + q = 0 ∧ r2 ^ 2 + (2 * q - 1) * r2 + q = 0) ↔
  q = 1 + (Real.sqrt 7) / 2 :=
sorry

end quadratic_roots_diff_by_2_l1825_182558


namespace inequality_on_abc_l1825_182531

variable (a b c : ℝ)

theorem inequality_on_abc (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0) (h₃ : a + b + c = 1) :
  (a^4 + b^4) / (a^6 + b^6) + (b^4 + c^4) / (b^6 + c^6) + (c^4 + a^4) / (c^6 + a^6) ≤ 1 / (a * b * c) :=
by
  sorry

end inequality_on_abc_l1825_182531


namespace paco_initial_cookies_l1825_182520

theorem paco_initial_cookies (x : ℕ) (h : x - 2 + 36 = 2 + 34) : x = 2 :=
by
-- proof steps will be filled in here
sorry

end paco_initial_cookies_l1825_182520


namespace problem_statement_l1825_182514

-- Definitions of sets S and P
def S : Set ℝ := {x | x^2 - 3 * x - 10 < 0}
def P (a : ℝ) : Set ℝ := {x | a + 1 < x ∧ x < 2 * a + 15}

-- Proof statement
theorem problem_statement (a : ℝ) : 
  (S = {x | -2 < x ∧ x < 5}) ∧ (S ⊆ P a → a ∈ Set.Icc (-5 : ℝ) (-3 : ℝ)) :=
by
  sorry

end problem_statement_l1825_182514


namespace find_divisor_l1825_182596

theorem find_divisor (n x : ℕ) (h1 : n = 3) (h2 : (n / x : ℝ) * 12 = 9): x = 4 := by
  sorry

end find_divisor_l1825_182596


namespace temperature_rise_per_hour_l1825_182538

-- Define the conditions
variables (x : ℕ) -- temperature rise per hour

-- Assume the given conditions
axiom power_outage : (3 : ℕ) * x = (6 : ℕ) * 4

-- State the proposition
theorem temperature_rise_per_hour : x = 8 :=
sorry

end temperature_rise_per_hour_l1825_182538


namespace inequality_division_by_two_l1825_182578

theorem inequality_division_by_two (x y : ℝ) (h : x > y) : (x / 2) > (y / 2) := 
sorry

end inequality_division_by_two_l1825_182578


namespace what_to_do_first_l1825_182582

-- Definition of the conditions
def eat_or_sleep_to_survive (days_without_eat : ℕ) (days_without_sleep : ℕ) : Prop :=
  (days_without_eat = 7 → days_without_sleep ≠ 7) ∨ (days_without_sleep = 7 → days_without_eat ≠ 7)

-- Theorem statement based on the problem and its conditions
theorem what_to_do_first (days_without_eat days_without_sleep : ℕ) :
  days_without_eat = 7 ∨ days_without_sleep = 7 →
  eat_or_sleep_to_survive days_without_eat days_without_sleep :=
by sorry

end what_to_do_first_l1825_182582


namespace marbles_exceed_200_on_sunday_l1825_182547

theorem marbles_exceed_200_on_sunday:
  ∃ n : ℕ, 3 * 2^n > 200 ∧ (n % 7) = 0 :=
by
  sorry

end marbles_exceed_200_on_sunday_l1825_182547


namespace problem_statement_l1825_182527

theorem problem_statement (a b c : ℝ) (h1 : a + b + c = 6) (h2 : a * b + b * c + c * a = 11) (h3 : a * b * c = 6) :
  (a * b / c) + (b * c / a) + (c * a / b) = 49 / 6 := 
by sorry

end problem_statement_l1825_182527


namespace sum_of_solutions_l1825_182513

theorem sum_of_solutions (y x : ℝ) (h1 : y = 7) (h2 : x^2 + y^2 = 100) : 
  x + -x = 0 :=
by
  sorry

end sum_of_solutions_l1825_182513


namespace option_A_incorrect_option_B_incorrect_option_C_incorrect_option_D_correct_l1825_182577

theorem option_A_incorrect (a : ℝ) : (a^2) * (a^3) ≠ a^6 :=
by sorry

theorem option_B_incorrect (a : ℝ) : (a^2)^3 ≠ a^5 :=
by sorry

theorem option_C_incorrect (a : ℝ) : (a^6) / (a^2) ≠ a^3 :=
by sorry

theorem option_D_correct (a b : ℝ) : (a + 2 * b) * (a - 2 * b) = a^2 - 4 * b^2 :=
by sorry

end option_A_incorrect_option_B_incorrect_option_C_incorrect_option_D_correct_l1825_182577


namespace multiple_of_eight_l1825_182541

theorem multiple_of_eight (x y : ℤ) (h : ∀ (k : ℤ), 24 + 16 * k = 8) : ∃ (k : ℤ), x + 16 * y = 8 * k := 
by
  sorry

end multiple_of_eight_l1825_182541


namespace least_integer_nk_l1825_182580

noncomputable def min_nk (k : ℕ) : ℕ :=
  (5 * k + 1) / 2

theorem least_integer_nk (k : ℕ) (S : Fin 5 → Finset ℕ) :
  (∀ j : Fin 5, (S j).card = k) →
  (∀ i : Fin 4, (S i ∩ S (i + 1)).card = 0) →
  (S 4 ∩ S 0).card = 0 →
  (∃ nk, (∃ (U : Finset ℕ), (∀ j : Fin 5, S j ⊆ U) ∧ U.card = nk) ∧ nk = min_nk k) :=
by
  sorry

end least_integer_nk_l1825_182580


namespace number_of_pens_l1825_182557

theorem number_of_pens (num_pencils : ℕ) (total_cost : ℝ) (avg_price_pencil : ℝ) (avg_price_pen : ℝ) : ℕ :=
  sorry

example : number_of_pens 75 690 2 18 = 30 :=
by 
  sorry

end number_of_pens_l1825_182557


namespace y_is_multiple_of_4_y_is_not_multiple_of_8_y_is_not_multiple_of_16_y_is_not_multiple_of_32_l1825_182533

def y := 96 + 144 + 200 + 300 + 600 + 720 + 4800

theorem y_is_multiple_of_4 : y % 4 = 0 := 
by sorry

theorem y_is_not_multiple_of_8 : y % 8 ≠ 0 := 
by sorry

theorem y_is_not_multiple_of_16 : y % 16 ≠ 0 := 
by sorry

theorem y_is_not_multiple_of_32 : y % 32 ≠ 0 := 
by sorry

end y_is_multiple_of_4_y_is_not_multiple_of_8_y_is_not_multiple_of_16_y_is_not_multiple_of_32_l1825_182533


namespace problem_l1825_182568

def is_arithmetic_sequence (a : ℕ → ℕ) :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

variable (a : ℕ → ℕ)
variable (S : ℕ → ℕ)
variable (M m : ℕ)

-- Conditions
axiom h1 : is_arithmetic_sequence a
axiom h2 : a 1 ≥ 1
axiom h3 : a 2 ≤ 5
axiom h4 : a 5 ≥ 8

-- Sum function for arithmetic sequence
axiom h5 : ∀ n : ℕ, S n = n * a 1 + n * (n - 1) / 2 * (a 2 - a 1)

-- Definition of M and m based on S_15
axiom hM : M = max (S 15)
axiom hm : m = min (S 15)

theorem problem (h : S 15 = M + m) : M + m = 600 :=
  sorry

end problem_l1825_182568


namespace machine_probabilities_at_least_one_first_class_component_l1825_182581

theorem machine_probabilities : 
  (∃ (PA PB PC : ℝ), 
  PA * (1 - PB) = 1/4 ∧ 
  PB * (1 - PC) = 1/12 ∧ 
  PA * PC = 2/9 ∧ 
  PA = 1/3 ∧ 
  PB = 1/4 ∧ 
  PC = 2/3) 
:=
sorry

theorem at_least_one_first_class_component : 
  ∃ (PA PB PC : ℝ), 
  PA * (1 - PB) = 1/4 ∧ 
  PB * (1 - PC) = 1/12 ∧ 
  PA * PC = 2/9 ∧ 
  PA = 1/3 ∧ 
  PB = 1/4 ∧ 
  PC = 2/3 ∧ 
  1 - (1 - PA) * (1 - PB) * (1 - PC) = 5/6
:=
sorry

end machine_probabilities_at_least_one_first_class_component_l1825_182581


namespace g_difference_l1825_182597

def g (n : ℕ) : ℚ := (1 / 4) * n * (n + 1) * (n + 2) * (n + 3)

theorem g_difference (s : ℕ) : g s - g (s - 1) = s * (s + 1) * (s + 2) := 
by sorry

end g_difference_l1825_182597


namespace greater_number_l1825_182548

theorem greater_number (x: ℕ) (h1 : 3 * x + 4 * x = 21) : 4 * x = 12 := by
  sorry

end greater_number_l1825_182548


namespace regions_of_diagonals_formula_l1825_182583

def regions_of_diagonals (n : ℕ) : ℕ :=
  ((n - 1) * (n - 2) * (n * n - 3 * n + 12)) / 24

theorem regions_of_diagonals_formula (n : ℕ) (h : 3 ≤ n) :
  ∃ (fn : ℕ), fn = regions_of_diagonals n := by
  sorry

end regions_of_diagonals_formula_l1825_182583


namespace find_a9_l1825_182530

variable (a : ℕ → ℤ)  -- Arithmetic sequence
variable (S : ℕ → ℤ)  -- Sum of the first n terms

-- Conditions provided in the problem
axiom Sum_condition : S 8 = 4 * a 3
axiom Term_condition : a 7 = -2
axiom Sum_def : ∀ n, S n = (n * (a 1 + a n)) / 2

-- Hypothesis for common difference
def common_diff (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Proving that a_9 equals -6 given the conditions
theorem find_a9 (d : ℤ) : common_diff a d → a 9 = -6 :=
by
  intros h
  sorry

end find_a9_l1825_182530


namespace expand_and_simplify_product_l1825_182523

-- Define the initial expression
def initial_expr (x : ℝ) : ℝ := (2 * x^2 - 3 * x + 4) * (2 * x^2 + 3 * x + 4)

-- Define the simplified form
def simplified_expr (x : ℝ) : ℝ := 4 * x^4 + 7 * x^2 + 16

theorem expand_and_simplify_product (x : ℝ) : initial_expr x = simplified_expr x := by
  -- We would provide the proof steps here
  sorry

end expand_and_simplify_product_l1825_182523


namespace solve_for_x_l1825_182542

theorem solve_for_x (x : ℤ) : 27 - 5 = 4 + x → x = 18 :=
by
  intro h
  sorry

end solve_for_x_l1825_182542


namespace remaining_amount_to_be_paid_is_1080_l1825_182512

noncomputable def deposit : ℕ := 120
noncomputable def total_price : ℕ := 10 * deposit
noncomputable def remaining_amount : ℕ := total_price - deposit

theorem remaining_amount_to_be_paid_is_1080 :
  remaining_amount = 1080 :=
by
  sorry

end remaining_amount_to_be_paid_is_1080_l1825_182512


namespace suff_not_nec_cond_l1825_182598

theorem suff_not_nec_cond (a : ℝ) : (a > 6 → a^2 > 36) ∧ (a^2 > 36 → (a > 6 ∨ a < -6)) := by
  sorry

end suff_not_nec_cond_l1825_182598


namespace common_ratio_of_infinite_geometric_series_l1825_182525

noncomputable def first_term : ℝ := 500
noncomputable def series_sum : ℝ := 3125

theorem common_ratio_of_infinite_geometric_series (r : ℝ) (h₀ : first_term / (1 - r) = series_sum) : 
  r = 0.84 := 
by
  sorry

end common_ratio_of_infinite_geometric_series_l1825_182525


namespace fish_total_count_l1825_182519

theorem fish_total_count :
  let num_fishermen : ℕ := 20
  let fish_caught_per_fisherman : ℕ := 400
  let fish_caught_by_twentieth_fisherman : ℕ := 2400
  (19 * fish_caught_per_fisherman + fish_caught_by_twentieth_fisherman) = 10000 :=
by
  sorry

end fish_total_count_l1825_182519


namespace polygon_properties_l1825_182595

theorem polygon_properties
    (each_exterior_angle : ℝ)
    (h1 : each_exterior_angle = 24) :
    ∃ n : ℕ, n = 15 ∧ (180 * (n - 2) = 2340) :=
  by
    sorry

end polygon_properties_l1825_182595


namespace Amy_initial_cupcakes_l1825_182501

def initialCupcakes (packages : ℕ) (cupcakesPerPackage : ℕ) (eaten : ℕ) : ℕ :=
  packages * cupcakesPerPackage + eaten

theorem Amy_initial_cupcakes :
  let packages := 9
  let cupcakesPerPackage := 5
  let eaten := 5
  initialCupcakes packages cupcakesPerPackage eaten = 50 :=
by
  sorry

end Amy_initial_cupcakes_l1825_182501


namespace total_cookies_prepared_l1825_182549

-- State the conditions as definitions
def num_guests : ℕ := 10
def cookies_per_guest : ℕ := 18

-- The theorem stating the problem
theorem total_cookies_prepared (num_guests cookies_per_guest : ℕ) : 
  num_guests * cookies_per_guest = 180 := 
by 
  -- Here, we would have the proof, but we're using sorry to skip it
  sorry

end total_cookies_prepared_l1825_182549


namespace minimum_rounds_l1825_182536

-- Given conditions based on the problem statement
variable (m : ℕ) (hm : m ≥ 17)
variable (players : Fin (2 * m)) -- Representing 2m players
variable (rounds : Fin (2 * m - 1)) -- Representing 2m - 1 rounds
variable (pairs : Fin m → Fin (2 * m) × Fin (2 * m)) -- Pairing for each of the m pairs in each round

-- Statement of the proof problem
theorem minimum_rounds (h1 : ∀ i j, i ≠ j → ∃! (k : Fin m), pairs k = (i, j) ∨ pairs k = (j, i))
(h2 : ∀ k : Fin m, (pairs k).fst ≠ (pairs k).snd)
(h3 : ∀ i j, i ≠ j → ∃ r : Fin (2 * m - 1), (∃ k : Fin m, pairs k = (i, j)) ∧ (∃ k : Fin m, pairs k = (j, i))) :
∃ (n : ℕ), n = m - 1 ∧ ∀ s : Fin 4 → Fin (2 * m), (∀ i j, i ≠ j → ¬ ∃ r : Fin n, ∃ k : Fin m, pairs k = (s i, s j)) ∨ (∃ r1 r2 : Fin n, ∃ i j, i ≠ j ∧ ∃ k1 k2 : Fin m, pairs k1 = (s i, s j) ∧ pairs k2 = (s j, s i)) :=
sorry

end minimum_rounds_l1825_182536


namespace number_of_students_in_third_group_l1825_182517

-- Definitions based on given conditions
def students_group1 : ℕ := 9
def students_group2 : ℕ := 10
def tissues_per_box : ℕ := 40
def total_tissues : ℕ := 1200

-- Define the number of students in the third group as a variable
variable {x : ℕ}

-- Prove that the number of students in the third group is 11
theorem number_of_students_in_third_group (h : 360 + 400 + 40 * x = 1200) : x = 11 :=
by sorry

end number_of_students_in_third_group_l1825_182517


namespace parallel_to_l3_through_P_perpendicular_to_l3_through_P_l1825_182574

-- Define the lines l1, l2, and l3
def l1 (x y : ℝ) : Prop := 2 * x + 3 * y - 5 = 0
def l2 (x y : ℝ) : Prop := x + 2 * y - 3 = 0
def l3 (x y : ℝ) : Prop := 2 * x + y - 5 = 0

-- Define the intersection point P
def P := (1, 1)

-- Define the parallel line equation to l3 passing through P
def parallel_line (x y : ℝ) : Prop := 2 * x + y - 3 = 0

-- Define the perpendicular line equation to l3 passing through P
def perpendicular_line (x y : ℝ) : Prop := x - 2 * y + 1 = 0

-- Prove the parallel line through P is 2x + y - 3 = 0
theorem parallel_to_l3_through_P : 
  ∀ (x y : ℝ), l1 x y → l2 x y → (parallel_line 1 1) := 
by 
  sorry

-- Prove the perpendicular line through P is x - 2y + 1 = 0
theorem perpendicular_to_l3_through_P : 
  ∀ (x y : ℝ), l1 x y → l2 x y → (perpendicular_line 1 1) := 
by 
  sorry

end parallel_to_l3_through_P_perpendicular_to_l3_through_P_l1825_182574


namespace problem_fraction_eq_l1825_182584

theorem problem_fraction_eq (x : ℝ) :
  (x * (3 / 4) * (1 / 2) * 5060 = 759.0000000000001) ↔ (x = 0.4) :=
by
  sorry

end problem_fraction_eq_l1825_182584


namespace geric_initial_bills_l1825_182500

theorem geric_initial_bills (G K J : ℕ) 
  (h1: G = 2 * K)
  (h2: K = J - 2)
  (h3: J - 3 = 7) : G = 16 := 
  by 
  sorry

end geric_initial_bills_l1825_182500


namespace book_weight_l1825_182544

theorem book_weight (total_weight : ℕ) (num_books : ℕ) (each_book_weight : ℕ) 
  (h1 : total_weight = 42) (h2 : num_books = 14) :
  each_book_weight = total_weight / num_books :=
by
  sorry

end book_weight_l1825_182544


namespace factorial_divisibility_l1825_182573

theorem factorial_divisibility 
  {n : ℕ} 
  (hn : bit0 (n.bits.count 1) == 1995) : 
  (2^(n-1995)) ∣ n! := 
sorry

end factorial_divisibility_l1825_182573


namespace distances_product_eq_l1825_182537

-- Define the distances
variables (d_ab d_ac d_bc d_ba d_cb d_ca : ℝ)

-- State the theorem
theorem distances_product_eq : d_ab * d_bc * d_ca = d_ac * d_ba * d_cb :=
sorry

end distances_product_eq_l1825_182537


namespace common_root_polynomials_l1825_182572

theorem common_root_polynomials (a : ℝ) :
  (∃ x : ℝ, x^2 + a * x + 1 = 0 ∧ x^2 + x + a = 0) ↔ (a = 1 ∨ a = -2) :=
by
  sorry

end common_root_polynomials_l1825_182572


namespace tangent_function_property_l1825_182586

noncomputable def f (x : ℝ) (ϕ : ℝ) : ℝ := Real.tan (ϕ - x)

theorem tangent_function_property 
  (ϕ a : ℝ) 
  (h1 : π / 2 < ϕ) 
  (h2 : ϕ < 3 * π / 2) 
  (h3 : f 0 ϕ = 0) 
  (h4 : f (-a) ϕ = 1/2) : 
  f (a + π / 4) ϕ = -3 := by
  sorry

end tangent_function_property_l1825_182586


namespace average_speed_round_trip_l1825_182585

-- Define average speed calculation for round trip

open Real

theorem average_speed_round_trip (S : ℝ) (hS : S > 0) :
  let t1 := S / 6
  let t2 := S / 4
  let total_distance := 2 * S
  let total_time := t1 + t2
  let average_speed := total_distance / total_time
  average_speed = 4.8 :=
  by
    sorry

end average_speed_round_trip_l1825_182585


namespace sum_sqrt_inequality_l1825_182506

theorem sum_sqrt_inequality (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  (3 / 2) * (a + b + c) ≥ (Real.sqrt (a^2 + b * c) + Real.sqrt (b^2 + c * a) + Real.sqrt (c^2 + a * b)) :=
by
  sorry

end sum_sqrt_inequality_l1825_182506


namespace lisa_eggs_l1825_182505

theorem lisa_eggs :
  ∃ x : ℕ, (5 * 52) * (4 * x + 3 + 2) = 3380 ∧ x = 2 :=
by
  sorry

end lisa_eggs_l1825_182505


namespace trig_identity_l1825_182593

theorem trig_identity :
  (Real.cos (105 * Real.pi / 180) * Real.cos (45 * Real.pi / 180) + Real.sin (45 * Real.pi / 180) * Real.sin (105 * Real.pi / 180)) = 1 / 2 :=
  sorry

end trig_identity_l1825_182593


namespace first_folder_number_l1825_182594

theorem first_folder_number (stickers : ℕ) (folders : ℕ) : stickers = 999 ∧ folders = 369 → 100 = 100 :=
by sorry

end first_folder_number_l1825_182594


namespace clock_hands_meeting_duration_l1825_182559

noncomputable def angle_between_clock_hands (h m : ℝ) : ℝ :=
  abs ((30 * h + m / 2) - (6 * m) % 360)

theorem clock_hands_meeting_duration : 
  ∃ n m : ℝ, 0 <= n ∧ n < m ∧ m < 60 ∧ angle_between_clock_hands 5 n = 120 ∧ angle_between_clock_hands 5 m = 120 ∧ m - n = 44 :=
sorry

end clock_hands_meeting_duration_l1825_182559


namespace number_of_grouping_methods_l1825_182560

theorem number_of_grouping_methods : 
  let males := 5
  let females := 3
  let groups := 2
  let select_males := Nat.choose males groups
  let select_females := Nat.choose females groups
  let permute := Nat.factorial groups
  select_males * select_females * permute * permute = 60 :=
by 
  sorry

end number_of_grouping_methods_l1825_182560


namespace minimum_value_x_plus_y_l1825_182526

theorem minimum_value_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x * y * (x - y)^2 = 1) : x + y ≥ 2 :=
sorry

end minimum_value_x_plus_y_l1825_182526


namespace A_subset_B_l1825_182543

def A : Set ℤ := { x : ℤ | ∃ k : ℤ, x = 4 * k + 1 }
def B : Set ℤ := { x : ℤ | ∃ k : ℤ, x = 2 * k - 1 }

theorem A_subset_B : A ⊆ B :=
  sorry

end A_subset_B_l1825_182543


namespace find_a_l1825_182570

open Set

theorem find_a (a : ℝ) (A : Set ℝ) (B : Set ℝ)
  (hA : A = {1, 2})
  (hB : B = {-a, a^2 + 3})
  (hUnion : A ∪ B = {1, 2, 4}) :
  a = -1 :=
sorry

end find_a_l1825_182570


namespace does_not_determine_shape_l1825_182535

-- Definition of a function that checks whether given data determine the shape of a triangle
def determines_shape (data : Type) : Prop := sorry

-- Various conditions about data
def ratio_two_angles_included_side : Type := sorry
def ratios_three_angle_bisectors : Type := sorry
def ratios_three_side_lengths : Type := sorry
def ratio_angle_bisector_opposite_side : Type := sorry
def three_angles : Type := sorry

-- The main theorem stating that the ratio of an angle bisector to its corresponding opposite side does not uniquely determine the shape of a triangle.
theorem does_not_determine_shape :
  ¬determines_shape ratio_angle_bisector_opposite_side := sorry

end does_not_determine_shape_l1825_182535


namespace area_of_large_hexagon_eq_270_l1825_182552

noncomputable def area_large_hexagon (area_shaded : ℝ) (n_small_hexagons_shaded : ℕ) (n_small_hexagons_large : ℕ): ℝ :=
  let area_one_small_hexagon := area_shaded / n_small_hexagons_shaded
  area_one_small_hexagon * n_small_hexagons_large

theorem area_of_large_hexagon_eq_270 :
  area_large_hexagon 180 6 7 = 270 := by
  sorry

end area_of_large_hexagon_eq_270_l1825_182552


namespace sum_of_digits_l1825_182579

theorem sum_of_digits (N : ℕ) (h : N * (N + 1) / 2 = 3003) : (7 + 7) = 14 := by
  sorry

end sum_of_digits_l1825_182579


namespace probability_of_sum_at_least_10_l1825_182551

def total_outcomes : ℕ := 36
def favorable_outcomes : ℕ := 6

theorem probability_of_sum_at_least_10 :
  (favorable_outcomes : ℝ) / (total_outcomes : ℝ) = 1 / 6 := by
  sorry

end probability_of_sum_at_least_10_l1825_182551


namespace fg_of_2_eq_0_l1825_182553

def f (x : ℝ) : ℝ := 4 - x^2
def g (x : ℝ) : ℝ := 3 * x - x^3

theorem fg_of_2_eq_0 : f (g 2) = 0 := by
  sorry

end fg_of_2_eq_0_l1825_182553


namespace solve_equation_l1825_182521

-- Definitions for the variables and the main equation
def equation (x y z : ℤ) : Prop :=
  5 * x^2 + y^2 + 3 * z^2 - 2 * y * z = 30

-- The statement that needs to be proved
theorem solve_equation (x y z : ℤ) :
  equation x y z ↔ (x, y, z) = (1, 5, 0) ∨ (x, y, z) = (1, -5, 0) ∨ (x, y, z) = (-1, 5, 0) ∨ (x, y, z) = (-1, -5, 0) :=
by
  sorry

end solve_equation_l1825_182521


namespace smallest_n_for_geometric_sequence_divisibility_l1825_182507

theorem smallest_n_for_geometric_sequence_divisibility :
  ∃ n : ℕ, (∀ m : ℕ, m < n → ¬ (2 * 10 ^ 6 ∣ (30 ^ (m - 1) * (5 / 6)))) ∧ (2 * 10 ^ 6 ∣ (30 ^ (n - 1) * (5 / 6))) ∧ n = 8 :=
by
  sorry

end smallest_n_for_geometric_sequence_divisibility_l1825_182507


namespace problem1_problem2_l1825_182599

-- Problem 1: Lean 4 Statement
theorem problem1 (n : ℕ) (hn : n > 0) : 20 ∣ (4 * 6^n + 5^(n + 1) - 9) :=
sorry

-- Problem 2: Lean 4 Statement
theorem problem2 : (3^100 % 7) = 4 :=
sorry

end problem1_problem2_l1825_182599


namespace king_plan_feasibility_l1825_182589

-- Create a predicate for the feasibility of the king's plan
def feasible (n : ℕ) : Prop :=
  (n = 6 ∧ true) ∨ (n = 2004 ∧ false)

theorem king_plan_feasibility :
  ∀ n : ℕ, feasible n :=
by
  intro n
  sorry

end king_plan_feasibility_l1825_182589


namespace arithmetic_lemma_l1825_182561

theorem arithmetic_lemma : 45 * 52 + 48 * 45 = 4500 := by
  sorry

end arithmetic_lemma_l1825_182561


namespace eval_F_at_4_f_5_l1825_182564

def f (a : ℤ) : ℤ := 3 * a - 6
def F (a : ℤ) (b : ℤ) : ℤ := 2 * b ^ 2 + 3 * a

theorem eval_F_at_4_f_5 : F 4 (f 5) = 174 := by
  sorry

end eval_F_at_4_f_5_l1825_182564


namespace min_value_expression_l1825_182587

theorem min_value_expression (x y z : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) (hz : 0 ≤ z ∧ z ≤ 1) :
  (min ((1 / ((1 - x) * (1 - y) * (1 - z))) + (1 / ((1 + x) * (1 + y) * (1 + z))) + (x * y * z)) 2) = 2 :=
by 
  sorry

end min_value_expression_l1825_182587


namespace population_in_scientific_notation_l1825_182569

theorem population_in_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), (1 ≤ |a| ∧ |a| < 10) ∧ 1370540000 = a * 10^n ∧ a = 1.37054 ∧ n = 9 :=
by
  sorry

end population_in_scientific_notation_l1825_182569


namespace hearts_total_shaded_area_l1825_182540

theorem hearts_total_shaded_area (A B C D : ℕ) (hA : A = 1) (hB : B = 4) (hC : C = 9) (hD : D = 16) :
  (D - C) + (B - A) = 10 := 
by 
  sorry

end hearts_total_shaded_area_l1825_182540


namespace solve_system_of_equations_l1825_182555

theorem solve_system_of_equations : 
  ∃ (x y : ℝ), 
  (6 * x - 3 * y = -3) ∧ (5 * x - 9 * y = -35) ∧ (x = 2) ∧ (y = 5) :=
by
  sorry

end solve_system_of_equations_l1825_182555


namespace martha_total_cost_l1825_182516

-- Definitions for the conditions
def amount_cheese_needed : ℝ := 1.5 -- in kg
def amount_meat_needed : ℝ := 0.5 -- in kg
def cost_cheese_per_kg : ℝ := 6.0 -- in dollars per kg
def cost_meat_per_kg : ℝ := 8.0 -- in dollars per kg

-- Total cost that needs to be calculated
def total_cost : ℝ :=
  (amount_cheese_needed * cost_cheese_per_kg) +
  (amount_meat_needed * cost_meat_per_kg)

-- Statement of the theorem
theorem martha_total_cost : total_cost = 13 := by
  sorry

end martha_total_cost_l1825_182516


namespace path_to_tile_ratio_l1825_182524

theorem path_to_tile_ratio
  (t p : ℝ) 
  (tiles : ℕ := 400)
  (grid_size : ℕ := 20)
  (total_tile_area : ℝ := (tiles : ℝ) * t^2)
  (total_courtyard_area : ℝ := (grid_size * (t + 2 * p))^2) 
  (tile_area_fraction : ℝ := total_tile_area / total_courtyard_area) : 
  tile_area_fraction = 0.25 → 
  p / t = 0.5 :=
by
  intro h
  sorry

end path_to_tile_ratio_l1825_182524


namespace minimum_a_l1825_182509

noncomputable def f (x : ℝ) : ℝ := abs x + abs (x - 1)
noncomputable def g (x a : ℝ) : ℝ := f x - a

theorem minimum_a (a : ℝ) : (∃ x : ℝ, g x a = 0) ↔ (a ≥ 1) :=
by sorry

end minimum_a_l1825_182509


namespace teddy_bear_cost_l1825_182532

-- Definitions for the given conditions
def num_toys : ℕ := 28
def toy_price : ℕ := 10
def num_teddy_bears : ℕ := 20
def total_money : ℕ := 580

-- The theorem we want to prove
theorem teddy_bear_cost :
  (num_teddy_bears * 15 + num_toys * toy_price = total_money) :=
by
  sorry

end teddy_bear_cost_l1825_182532


namespace quadratic_real_roots_l1825_182515

theorem quadratic_real_roots (m : ℝ) : 
  (∃ x : ℝ, (m - 3) * x^2 - 2 * x + 1 = 0) →
  (m ≤ 4 ∧ m ≠ 3) :=
sorry

end quadratic_real_roots_l1825_182515


namespace find_f_l1825_182563

noncomputable def f : ℝ → ℝ := sorry

theorem find_f (f : ℝ → ℝ) (h₀ : f 0 = 2) 
  (h₁ : ∀ x y : ℝ, f (x * y) = f ((x^2 + y^2) / 2) + (x + y)^2) :
  ∀ x : ℝ, f x = 2 - 2 * x :=
sorry

end find_f_l1825_182563


namespace area_diff_l1825_182522

-- Defining the side lengths of squares
def side_length_small_square : ℕ := 4
def side_length_large_square : ℕ := 10

-- Calculating the areas
def area_small_square : ℕ := side_length_small_square ^ 2
def area_large_square : ℕ := side_length_large_square ^ 2

-- Theorem statement
theorem area_diff (a_small a_large : ℕ) (h1 : a_small = side_length_small_square ^ 2) (h2 : a_large = side_length_large_square ^ 2) : 
  a_large - a_small = 84 :=
by
  sorry

end area_diff_l1825_182522


namespace tan_neg_210_eq_neg_sqrt_3_div_3_l1825_182539

theorem tan_neg_210_eq_neg_sqrt_3_div_3 : Real.tan (-210 * Real.pi / 180) = - (Real.sqrt 3 / 3) :=
by
  sorry

end tan_neg_210_eq_neg_sqrt_3_div_3_l1825_182539


namespace probability_x_gt_9y_in_rectangle_l1825_182518

theorem probability_x_gt_9y_in_rectangle :
  let a := 1007
  let b := 1008
  let area_triangle := (a * a / 18 : ℚ)
  let area_rectangle := (a * b : ℚ)
  area_triangle / area_rectangle = (1 : ℚ) / 18 :=
by
  sorry

end probability_x_gt_9y_in_rectangle_l1825_182518


namespace find_x_in_triangle_l1825_182511

theorem find_x_in_triangle (y z : ℝ) (cos_Y_minus_Z : ℝ) (h1 : y = 7) (h2 : z = 6) (h3 : cos_Y_minus_Z = 1 / 2) : 
    ∃ x : ℝ, x = Real.sqrt 73 :=
by
  existsi Real.sqrt 73
  sorry

end find_x_in_triangle_l1825_182511


namespace compute_expression_l1825_182554

theorem compute_expression : 11 * (1 / 17) * 34 - 3 = 19 :=
by
  sorry

end compute_expression_l1825_182554


namespace warehouse_painted_area_l1825_182504

theorem warehouse_painted_area :
  let length := 8
  let width := 6
  let height := 3.5
  let door_width := 1
  let door_height := 2
  let front_back_area := 2 * (length * height)
  let left_right_area := 2 * (width * height)
  let total_wall_area := front_back_area + left_right_area
  let door_area := door_width * door_height
  let painted_area := total_wall_area - door_area
  painted_area = 96 :=
by
  -- Sorry to skip the actual proof steps
  sorry

end warehouse_painted_area_l1825_182504


namespace evaluate_expression_l1825_182592

theorem evaluate_expression : 
  -3 * 5 - (-4 * -2) + (-15 * -3) / 3 = -8 :=
by
  sorry

end evaluate_expression_l1825_182592


namespace find_a_l1825_182588

noncomputable def f (x a : ℝ) := x^2 + (a + 1) * x + (a + 2)

def g (x a : ℝ) := (a + 1) * x
def h (x a : ℝ) := x^2 + a + 2

def p (a : ℝ) := ∀ x ≥ (a + 1)^2, f x a ≤ x
def q (a : ℝ) := ∀ x, g x a < 0

theorem find_a : 
  (¬p a) → (p a ∨ q a) → a ≥ -1 := sorry

end find_a_l1825_182588


namespace maximum_value_problem_l1825_182534

theorem maximum_value_problem (x : ℝ) (h : 0 < x ∧ x < 4/3) : ∃ M, M = (4 / 3) ∧ ∀ y, 0 < y ∧ y < 4/3 → x * (4 - 3 * x) ≤ M :=
sorry

end maximum_value_problem_l1825_182534


namespace binomial_coefficient_sum_l1825_182565

theorem binomial_coefficient_sum {n : ℕ} (h : (1 : ℝ) + 1 = 128) : n = 7 :=
by
  sorry

end binomial_coefficient_sum_l1825_182565


namespace range_of_m_l1825_182529

variable {f : ℝ → ℝ}

theorem range_of_m 
  (even_f : ∀ x : ℝ, f x = f (-x))
  (mono_f : ∀ {x y : ℝ}, 0 ≤ x → x ≤ y → f x ≤ f y)
  (h : f (m + 1) < f (3 * m - 1)) :
  m > 1 ∨ m < 0 :=
sorry

end range_of_m_l1825_182529


namespace matrix_equation_l1825_182571

open Matrix

-- Define matrix N and the identity matrix I
def N : Matrix (Fin 2) (Fin 2) ℤ := ![![3, 8], ![-4, -2]]
def I : Matrix (Fin 2) (Fin 2) ℤ := 1

-- Scalars p and q
def p : ℤ := 1
def q : ℤ := -26

-- Theorem statement
theorem matrix_equation :
  N * N = p • N + q • I :=
  by
    sorry

end matrix_equation_l1825_182571

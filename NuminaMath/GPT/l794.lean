import Mathlib

namespace NUMINAMATH_GPT_find_a_plus_b_l794_79441

theorem find_a_plus_b (a b : ℝ) 
  (h1 : 1 = a - b) 
  (h2 : 5 = a - b / 5) : a + b = 11 :=
by
  sorry

end NUMINAMATH_GPT_find_a_plus_b_l794_79441


namespace NUMINAMATH_GPT_sequence_a100_gt_14_l794_79446

theorem sequence_a100_gt_14 (a : ℕ → ℝ)
  (h1 : a 1 = 1)
  (h2 : ∀ n, 1 ≤ n → a (n+1) = a n + 1 / a n) :
  a 100 > 14 :=
by sorry

end NUMINAMATH_GPT_sequence_a100_gt_14_l794_79446


namespace NUMINAMATH_GPT_area_percentage_decrease_42_l794_79437

def radius_decrease_factor : ℝ := 0.7615773105863908

noncomputable def area_percentage_decrease : ℝ :=
  let k := radius_decrease_factor
  100 * (1 - k^2)

theorem area_percentage_decrease_42 :
  area_percentage_decrease = 42 := by
  sorry

end NUMINAMATH_GPT_area_percentage_decrease_42_l794_79437


namespace NUMINAMATH_GPT_hose_rate_l794_79419

theorem hose_rate (V : ℝ) (T : ℝ) (r_fixed : ℝ) (total_rate : ℝ) (R : ℝ) :
  V = 15000 ∧ T = 25 ∧ r_fixed = 3 ∧ total_rate = 10 ∧
  (2 * R + 2 * r_fixed = total_rate) → R = 2 :=
by
  -- Given conditions:
  -- Volume V = 15000 gallons
  -- Time T = 25 hours
  -- Rate of fixed hoses r_fixed = 3 gallons per minute each
  -- Total rate of filling the pool total_rate = 10 gallons per minute
  -- Relationship: 2 * rate of first two hoses + 2 * rate of fixed hoses = total rate
  
  sorry

end NUMINAMATH_GPT_hose_rate_l794_79419


namespace NUMINAMATH_GPT_solve_for_a_l794_79417

theorem solve_for_a (a : ℕ) (h : a > 0) (eqn : a / (a + 37) = 925 / 1000) : a = 455 :=
sorry

end NUMINAMATH_GPT_solve_for_a_l794_79417


namespace NUMINAMATH_GPT_probability_one_painted_face_l794_79458

def cube : ℕ := 5
def total_unit_cubes : ℕ := 125
def painted_faces_share_edge : Prop := true
def unit_cubes_with_one_painted_face : ℕ := 41

theorem probability_one_painted_face :
  ∃ (cube : ℕ) (total_unit_cubes : ℕ) (painted_faces_share_edge : Prop) (unit_cubes_with_one_painted_face : ℕ),
  cube = 5 ∧ total_unit_cubes = 125 ∧ painted_faces_share_edge ∧ unit_cubes_with_one_painted_face = 41 →
  (unit_cubes_with_one_painted_face : ℚ) / (total_unit_cubes : ℚ) = 41 / 125 :=
by 
  sorry

end NUMINAMATH_GPT_probability_one_painted_face_l794_79458


namespace NUMINAMATH_GPT_initial_brownies_l794_79477

theorem initial_brownies (B : ℕ) (eaten_by_father : ℕ) (eaten_by_mooney : ℕ) (new_brownies : ℕ) (total_brownies : ℕ) :
  eaten_by_father = 8 →
  eaten_by_mooney = 4 →
  new_brownies = 24 →
  total_brownies = 36 →
  (B - (eaten_by_father + eaten_by_mooney) + new_brownies = total_brownies) →
  B = 24 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_initial_brownies_l794_79477


namespace NUMINAMATH_GPT_arithmetic_series_sum_l794_79483

def first_term (k : ℕ) : ℕ := k^2 + k + 1
def common_difference : ℕ := 1
def number_of_terms (k : ℕ) : ℕ := 2 * k + 3
def nth_term (k n : ℕ) : ℕ := (first_term k) + (n - 1) * common_difference
def sum_of_terms (k : ℕ) : ℕ :=
  let n := number_of_terms k
  let a := first_term k
  let l := nth_term k n
  n * (a + l) / 2

theorem arithmetic_series_sum (k : ℕ) : sum_of_terms k = 2 * k^3 + 7 * k^2 + 10 * k + 6 :=
sorry

end NUMINAMATH_GPT_arithmetic_series_sum_l794_79483


namespace NUMINAMATH_GPT_eggs_per_basket_l794_79409

theorem eggs_per_basket (red_eggs : ℕ) (orange_eggs : ℕ) (min_eggs : ℕ) :
  red_eggs = 30 → orange_eggs = 45 → min_eggs = 5 →
  (∃ k, (30 % k = 0) ∧ (45 % k = 0) ∧ (k ≥ 5) ∧ k = 15) :=
by
  intros h1 h2 h3
  use 15
  sorry

end NUMINAMATH_GPT_eggs_per_basket_l794_79409


namespace NUMINAMATH_GPT_find_P2_l794_79465

def P1 : ℕ := 64
def total_pigs : ℕ := 86

theorem find_P2 : ∃ (P2 : ℕ), P1 + P2 = total_pigs ∧ P2 = 22 :=
by 
  sorry

end NUMINAMATH_GPT_find_P2_l794_79465


namespace NUMINAMATH_GPT_find_q_minus_p_values_l794_79494

theorem find_q_minus_p_values (p q n : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hn : 0 < n) 
    (h : (p * (q + 1) + q * (p + 1)) * (n + 2) = 2 * n * p * q) : 
    q - p = 2 ∨ q - p = 3 ∨ q - p = 5 :=
sorry

end NUMINAMATH_GPT_find_q_minus_p_values_l794_79494


namespace NUMINAMATH_GPT_factorization_count_l794_79495

noncomputable def count_factors (n : ℕ) (a b c : ℕ) : ℕ :=
if 2 ^ a * 2 ^ b * 2 ^ c = n ∧ a + b + c = 10 ∧ a ≥ b ∧ b ≥ c then 1 else 0

noncomputable def total_factorizations : ℕ :=
Finset.sum (Finset.range 11) (fun c => 
  Finset.sum (Finset.Icc c 10) (fun b => 
    Finset.sum (Finset.Icc b 10) (fun a =>
      count_factors 1024 a b c)))

theorem factorization_count : total_factorizations = 14 :=
sorry

end NUMINAMATH_GPT_factorization_count_l794_79495


namespace NUMINAMATH_GPT_coordinates_OQ_quadrilateral_area_range_l794_79461

variables {p : ℝ} (p_pos : 0 < p)
variables {x0 x1 x2 y0 y1 y2 : ℝ} (h_parabola_A : y1^2 = 2*p*x1) (h_parabola_B : y2^2 = 2*p*x2) (h_parabola_M : y0^2 = 2*p*x0)
variables {a : ℝ} (h_focus_x : a = x0 + p) 

variables {FA FM FB : ℝ}
variables (h_arith_seq : ( FM = FA - (FA - FB) / 2 ))

-- Step 1: Prove the coordinates of OQ
theorem coordinates_OQ : (x0 + p, 0) = (a, 0) :=
by
  -- proof will be completed here
  sorry 

variables {x0_val : ℝ} (x0_eq : x0 = 2) {FM_val : ℝ} (FM_eq : FM = 5 / 2)

-- Step 2: Prove the area range of quadrilateral ABB1A1
theorem quadrilateral_area_range : ∀ (p : ℝ), 0 < p →
  ∀ (x0 x1 x2 y1 y2 FM OQ : ℝ), 
    x0 = 2 → FM = 5 / 2 → OQ = 3 → (y1^2 = 2*p*x1) → (y2^2 = 2*p*x2) →
  ( ∃ S : ℝ, 0 < S ∧ S ≤ 10) :=
by
  -- proof will be completed here
  sorry 

end NUMINAMATH_GPT_coordinates_OQ_quadrilateral_area_range_l794_79461


namespace NUMINAMATH_GPT_roots_of_polynomial_l794_79470

theorem roots_of_polynomial (x : ℝ) : x^2 - 4 = 0 ↔ x = 2 ∨ x = -2 :=
by
  sorry

end NUMINAMATH_GPT_roots_of_polynomial_l794_79470


namespace NUMINAMATH_GPT_find_function_that_satisfies_eq_l794_79421

theorem find_function_that_satisfies_eq :
  ∀ (f : ℕ → ℕ), (∀ (m n : ℕ), f (m + f n) = f (f m) + f n) → (∀ n : ℕ, f n = n) :=
by
  intro f
  intro h
  sorry

end NUMINAMATH_GPT_find_function_that_satisfies_eq_l794_79421


namespace NUMINAMATH_GPT_problem1_problem2_l794_79471

theorem problem1 (a b : ℝ) (h1 : a ≥ b) (h2 : b > 0) : 3 * a^3 + 2 * b^3 ≥ 3 * a^2 * b + 2 * a * b^2 := 
by
  sorry

theorem problem2 (a b : ℝ) (h1 : abs a < 1) (h2 : abs b < 1) : abs (1 - a * b) > abs (a - b) := 
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l794_79471


namespace NUMINAMATH_GPT_find_exponent_l794_79456

theorem find_exponent (y : ℕ) (h : (1/8) * (2: ℝ)^36 = (2: ℝ)^y) : y = 33 :=
by sorry

end NUMINAMATH_GPT_find_exponent_l794_79456


namespace NUMINAMATH_GPT_ratio_a_c_l794_79436

variables (a b c d : ℚ)

axiom ratio_a_b : a / b = 5 / 4
axiom ratio_c_d : c / d = 4 / 3
axiom ratio_d_b : d / b = 1 / 8

theorem ratio_a_c : a / c = 15 / 2 :=
by sorry

end NUMINAMATH_GPT_ratio_a_c_l794_79436


namespace NUMINAMATH_GPT_initial_marbles_count_l794_79444

-- Define the conditions
def marbles_given_to_mary : ℕ := 14
def marbles_remaining : ℕ := 50

-- Prove that Dan's initial number of marbles is 64
theorem initial_marbles_count : marbles_given_to_mary + marbles_remaining = 64 := 
by {
  sorry
}

end NUMINAMATH_GPT_initial_marbles_count_l794_79444


namespace NUMINAMATH_GPT_angle_between_diagonal_and_base_l794_79447

theorem angle_between_diagonal_and_base 
  (α : ℝ) (hα1 : 0 < α) (hα2 : α < π / 2) :
  ∃ θ : ℝ, θ = Real.arctan (Real.sin (α / 2)) :=
sorry

end NUMINAMATH_GPT_angle_between_diagonal_and_base_l794_79447


namespace NUMINAMATH_GPT_pinky_pies_count_l794_79408

theorem pinky_pies_count (helen_pies : ℕ) (total_pies : ℕ) (h1 : helen_pies = 56) (h2 : total_pies = 203) : 
  total_pies - helen_pies = 147 := by
  sorry

end NUMINAMATH_GPT_pinky_pies_count_l794_79408


namespace NUMINAMATH_GPT_perpendicular_slope_l794_79431

variable (x y : ℝ)

def line_eq : Prop := 4 * x - 5 * y = 20

theorem perpendicular_slope (x y : ℝ) (h : line_eq x y) : - (1 / (4 / 5)) = -5 / 4 := by
  sorry

end NUMINAMATH_GPT_perpendicular_slope_l794_79431


namespace NUMINAMATH_GPT_enclosed_area_correct_l794_79454

noncomputable def enclosedArea : ℝ := ∫ x in (1 / Real.exp 1)..Real.exp 1, 1 / x

theorem enclosed_area_correct : enclosedArea = 2 := by
  sorry

end NUMINAMATH_GPT_enclosed_area_correct_l794_79454


namespace NUMINAMATH_GPT_sum_odd_implies_parity_l794_79484

theorem sum_odd_implies_parity (a b c: ℤ) (h: (a + b + c) % 2 = 1) : (a^2 + b^2 - c^2 + 2 * a * b) % 2 = 1 := 
sorry

end NUMINAMATH_GPT_sum_odd_implies_parity_l794_79484


namespace NUMINAMATH_GPT_daughter_and_child_weight_l794_79475

variables (M D C : ℝ)

-- Conditions
def condition1 : Prop := M + D + C = 160
def condition2 : Prop := D = 40
def condition3 : Prop := C = (1/5) * M

-- Goal (Question)
def goal : Prop := D + C = 60

theorem daughter_and_child_weight
  (h1 : condition1 M D C)
  (h2 : condition2 D)
  (h3 : condition3 M C) : goal D C :=
by
  sorry

end NUMINAMATH_GPT_daughter_and_child_weight_l794_79475


namespace NUMINAMATH_GPT_geometric_sequence_a3_value_l794_79415

noncomputable def geometric_seq (a : ℕ → ℝ) (r : ℝ) : Prop :=
∀ n, a (n + 1) = a n * r

theorem geometric_sequence_a3_value :
  ∃ a : ℕ → ℝ, ∃ r : ℝ,
  geometric_seq a r ∧
  a 1 = 2 ∧
  (a 3) * (a 5) = 4 * (a 6)^2 →
  a 3 = 1 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_a3_value_l794_79415


namespace NUMINAMATH_GPT_unique_solution_l794_79433

theorem unique_solution (x : ℝ) (hx : x ≥ 0) : 2021 * x = 2022 * x ^ (2021 / 2022) - 1 → x = 1 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_unique_solution_l794_79433


namespace NUMINAMATH_GPT_mean_age_gauss_family_l794_79496

theorem mean_age_gauss_family :
  let ages := [7, 7, 7, 14, 15]
  let sum_ages := List.sum ages
  let number_of_children := List.length ages
  let mean_age := sum_ages / number_of_children
  mean_age = 10 :=
by
  sorry

end NUMINAMATH_GPT_mean_age_gauss_family_l794_79496


namespace NUMINAMATH_GPT_value_of_g_g_2_l794_79405

def g (x : ℝ) : ℝ := 4 * x^2 + 3

theorem value_of_g_g_2 : g (g 2) = 1447 := by
  sorry

end NUMINAMATH_GPT_value_of_g_g_2_l794_79405


namespace NUMINAMATH_GPT_pool_capacity_l794_79466

theorem pool_capacity (C : ℝ) (initial_water : ℝ) :
  0.85 * C - 0.70 * C = 300 → C = 2000 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_pool_capacity_l794_79466


namespace NUMINAMATH_GPT_greatest_common_divisor_of_72_and_m_l794_79410

-- Definitions based on the conditions
def is_power_of_prime (m : ℕ) : Prop :=
  ∃ (p : ℕ) (k : ℕ), Nat.Prime p ∧ m = p^k

-- Main theorem based on the question and conditions
theorem greatest_common_divisor_of_72_and_m (m : ℕ) :
  (Nat.gcd 72 m = 9) ↔ (m = 3^2) ∨ ∃ k, k ≥ 2 ∧ m = 3^k :=
by
  sorry

end NUMINAMATH_GPT_greatest_common_divisor_of_72_and_m_l794_79410


namespace NUMINAMATH_GPT_walk_to_cafe_and_back_time_l794_79449

theorem walk_to_cafe_and_back_time 
  (t_p : ℝ) (d_p : ℝ) (half_dp : ℝ) (pace : ℝ)
  (h1 : t_p = 30) 
  (h2 : d_p = 3) 
  (h3 : half_dp = d_p / 2) 
  (h4 : pace = t_p / d_p) :
  2 * half_dp * pace = 30 :=
by 
  sorry

end NUMINAMATH_GPT_walk_to_cafe_and_back_time_l794_79449


namespace NUMINAMATH_GPT_base_area_of_rect_prism_l794_79432

theorem base_area_of_rect_prism (r : ℝ) (h : ℝ) (V : ℝ) (h_rate : ℝ) (V_rate : ℝ) (conversion : ℝ) :
  V_rate = conversion * V ∧ h_rate = h → ∃ A : ℝ, A = V / h ∧ A = 100 :=
by
  sorry

end NUMINAMATH_GPT_base_area_of_rect_prism_l794_79432


namespace NUMINAMATH_GPT_wrapping_paper_fraction_each_present_l794_79442

theorem wrapping_paper_fraction_each_present (total_fraction : ℚ) (num_presents : ℕ) 
  (H : total_fraction = 3/10) (H1 : num_presents = 3) :
  total_fraction / num_presents = 1/10 :=
by sorry

end NUMINAMATH_GPT_wrapping_paper_fraction_each_present_l794_79442


namespace NUMINAMATH_GPT_distance_between_city_A_and_city_B_l794_79429

noncomputable def eddyTravelTime : ℝ := 3  -- hours
noncomputable def freddyTravelTime : ℝ := 4  -- hours
noncomputable def constantDistance : ℝ := 300  -- km
noncomputable def speedRatio : ℝ := 2  -- Eddy:Freddy

theorem distance_between_city_A_and_city_B (D_B D_C : ℝ) (h1 : D_B = (3 / 2) * D_C) (h2 : D_C = 300) :
  D_B = 450 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_city_A_and_city_B_l794_79429


namespace NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l794_79402

theorem problem1 : -20 - (-14) + (-18) - 13 = -37 := by
  sorry

theorem problem2 : (-3/4 + 1/6 - 5/8) / (-1/24) = 29 := by
  sorry

theorem problem3 : -3^2 + (-3)^2 + 3 * 2 + |(-4)| = 10 := by
  sorry

theorem problem4 : 16 / (-2)^3 - (-1/6) * (-4) + (-1)^2024 = -5/3 := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l794_79402


namespace NUMINAMATH_GPT_chocolate_bar_breaks_l794_79490

-- Definition of the problem as per the conditions
def chocolate_bar (rows : ℕ) (cols : ℕ) : ℕ := rows * cols

-- Statement of the proving problem
theorem chocolate_bar_breaks :
  ∀ (rows cols : ℕ), chocolate_bar rows cols = 40 → rows = 5 → cols = 8 → 
  (rows - 1) + (cols * (rows - 1)) = 39 :=
by
  intros rows cols h_bar h_rows h_cols
  sorry

end NUMINAMATH_GPT_chocolate_bar_breaks_l794_79490


namespace NUMINAMATH_GPT_equations_not_equivalent_l794_79450

variable {X : Type} [Field X]
variable (A B : X → X)

theorem equations_not_equivalent (h1 : ∀ x, A x ^ 2 = B x ^ 2) (h2 : ¬∀ x, A x = B x) :
  (∃ x, A x ≠ B x ∨ A x ≠ -B x) := 
sorry

end NUMINAMATH_GPT_equations_not_equivalent_l794_79450


namespace NUMINAMATH_GPT_least_number_to_subtract_l794_79462

theorem least_number_to_subtract (x : ℕ) (h1 : 997 - x ≡ 3 [MOD 17]) (h2 : 997 - x ≡ 3 [MOD 19]) (h3 : 997 - x ≡ 3 [MOD 23]) : x = 3 :=
by
  sorry

end NUMINAMATH_GPT_least_number_to_subtract_l794_79462


namespace NUMINAMATH_GPT_fifth_group_pythagorean_triples_l794_79448

theorem fifth_group_pythagorean_triples :
  ∃ (a b c : ℕ), (a, b, c) = (11, 60, 61) ∧ a^2 + b^2 = c^2 :=
by
  use 11, 60, 61
  sorry

end NUMINAMATH_GPT_fifth_group_pythagorean_triples_l794_79448


namespace NUMINAMATH_GPT_garden_area_proof_l794_79428

def length_rect : ℕ := 20
def width_rect : ℕ := 18
def area_rect : ℕ := length_rect * width_rect

def side_square1 : ℕ := 4
def area_square1 : ℕ := side_square1 * side_square1

def side_square2 : ℕ := 5
def area_square2 : ℕ := side_square2 * side_square2

def area_remaining : ℕ := area_rect - area_square1 - area_square2

theorem garden_area_proof : area_remaining = 319 := by
  sorry

end NUMINAMATH_GPT_garden_area_proof_l794_79428


namespace NUMINAMATH_GPT_find_roots_l794_79464

theorem find_roots (x : ℝ) : (x^2 + x = 0) ↔ (x = 0 ∨ x = -1) := 
by sorry

end NUMINAMATH_GPT_find_roots_l794_79464


namespace NUMINAMATH_GPT_largest_integer_a_l794_79445

theorem largest_integer_a (x a : ℤ) :
  ∃ x : ℤ, (x - a) * (x - 7) + 3 = 0 → a ≤ 11 :=
sorry

end NUMINAMATH_GPT_largest_integer_a_l794_79445


namespace NUMINAMATH_GPT_chorus_group_membership_l794_79418

theorem chorus_group_membership (n : ℕ) : 
  100 < n ∧ n < 200 →
  n % 3 = 1 ∧ 
  n % 4 = 2 ∧ 
  n % 6 = 4 ∧ 
  n % 8 = 6 →
  n = 118 ∨ n = 142 ∨ n = 166 ∨ n = 190 :=
by
  sorry

end NUMINAMATH_GPT_chorus_group_membership_l794_79418


namespace NUMINAMATH_GPT_hyperbola_equation_l794_79435

noncomputable def h : ℝ := -4
noncomputable def k : ℝ := 2
noncomputable def a : ℝ := 1
noncomputable def c : ℝ := Real.sqrt 2
noncomputable def b : ℝ := 1

theorem hyperbola_equation :
  (h + k + a + b) = 0 := by
  have h := -4
  have k := 2
  have a := 1
  have b := 1
  show (-4 + 2 + 1 + 1) = 0
  sorry

end NUMINAMATH_GPT_hyperbola_equation_l794_79435


namespace NUMINAMATH_GPT_mary_baking_cups_l794_79413

-- Conditions
def flour_needed : ℕ := 9
def sugar_needed : ℕ := 11
def flour_added : ℕ := 4
def sugar_added : ℕ := 0

-- Statement to prove
theorem mary_baking_cups : sugar_needed - (flour_needed - flour_added) = 6 := by
  sorry

end NUMINAMATH_GPT_mary_baking_cups_l794_79413


namespace NUMINAMATH_GPT_sqrt_17_irrational_l794_79482

theorem sqrt_17_irrational : ¬ ∃ (q : ℚ), q * q = 17 := sorry

end NUMINAMATH_GPT_sqrt_17_irrational_l794_79482


namespace NUMINAMATH_GPT_sum_first_n_terms_geom_seq_l794_79430

def geom_seq (n : ℕ) : ℕ :=
match n with
| 0     => 2
| k + 1 => 3 * geom_seq k

def sum_geom_seq (n : ℕ) : ℕ :=
(geom_seq 0) * (3 ^ n - 1) / (3 - 1)

theorem sum_first_n_terms_geom_seq (n : ℕ) :
sum_geom_seq n = 3 ^ n - 1 := by
sorry

end NUMINAMATH_GPT_sum_first_n_terms_geom_seq_l794_79430


namespace NUMINAMATH_GPT_sheela_monthly_income_l794_79497

theorem sheela_monthly_income (deposit : ℝ) (percentage : ℝ) (income : ℝ) 
  (h1 : deposit = 2500) (h2 : percentage = 0.25) (h3 : deposit = percentage * income) :
  income = 10000 := 
by
  -- proof steps would go here
  sorry

end NUMINAMATH_GPT_sheela_monthly_income_l794_79497


namespace NUMINAMATH_GPT_geometric_series_sum_l794_79439

-- Define the terms of the series
def a : ℚ := 1 / 5
def r : ℚ := -1 / 3
def n : ℕ := 6

-- Define the expected sum
def expected_sum : ℚ := 182 / 1215

-- Prove that the sum of the geometric series equals the expected sum
theorem geometric_series_sum : 
  (a * (1 - r^n)) / (1 - r) = expected_sum := 
by
  sorry

end NUMINAMATH_GPT_geometric_series_sum_l794_79439


namespace NUMINAMATH_GPT_three_gorges_scientific_notation_l794_79491

theorem three_gorges_scientific_notation :
  ∃a n : ℝ, (1 ≤ |a| ∧ |a| < 10) ∧ (798.5 * 10^1 = a * 10^n) ∧ a = 7.985 ∧ n = 2 :=
by
  sorry

end NUMINAMATH_GPT_three_gorges_scientific_notation_l794_79491


namespace NUMINAMATH_GPT_mean_of_remaining_four_numbers_l794_79478

theorem mean_of_remaining_four_numbers (a b c d : ℝ) :
  (a + b + c + d + 105) / 5 = 92 → (a + b + c + d) / 4 = 88.75 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_mean_of_remaining_four_numbers_l794_79478


namespace NUMINAMATH_GPT_ten_more_than_twice_number_of_birds_l794_79487

def number_of_birds : ℕ := 20

theorem ten_more_than_twice_number_of_birds :
  10 + 2 * number_of_birds = 50 :=
by
  sorry

end NUMINAMATH_GPT_ten_more_than_twice_number_of_birds_l794_79487


namespace NUMINAMATH_GPT_profit_percentage_l794_79493

-- Define the selling price and the cost price
def SP : ℝ := 100
def CP : ℝ := 86.95652173913044

-- State the theorem for profit percentage
theorem profit_percentage :
  ((SP - CP) / CP) * 100 = 15 :=
by
  sorry

end NUMINAMATH_GPT_profit_percentage_l794_79493


namespace NUMINAMATH_GPT_base_seven_to_ten_l794_79453

theorem base_seven_to_ten :
  (6 * 7^4 + 5 * 7^3 + 2 * 7^2 + 3 * 7^1 + 4 * 7^0) = 16244 :=
by sorry

end NUMINAMATH_GPT_base_seven_to_ten_l794_79453


namespace NUMINAMATH_GPT_correct_division_algorithm_l794_79486

theorem correct_division_algorithm : (-8 : ℤ) / (-4 : ℤ) = (8 : ℤ) / (4 : ℤ) := 
by 
  sorry

end NUMINAMATH_GPT_correct_division_algorithm_l794_79486


namespace NUMINAMATH_GPT_no_root_of_equation_l794_79411

theorem no_root_of_equation : ∀ x : ℝ, x - 8 / (x - 4) ≠ 4 - 8 / (x - 4) :=
by
  intro x
  -- Original equation:
  -- x - 8 / (x - 4) = 4 - 8 / (x - 4)
  -- No valid value of x solves the above equation as shown in the given solution
  sorry

end NUMINAMATH_GPT_no_root_of_equation_l794_79411


namespace NUMINAMATH_GPT_max_value_condition_l794_79460

variable {m n : ℝ}

-- Given conditions
def conditions (m n : ℝ) : Prop :=
  m * n > 0 ∧ m + n = -1

-- Statement of the proof problem
theorem max_value_condition (h : conditions m n) : (1/m + 1/n) ≤ 4 :=
sorry

end NUMINAMATH_GPT_max_value_condition_l794_79460


namespace NUMINAMATH_GPT_int_values_satisfy_condition_l794_79422

theorem int_values_satisfy_condition :
  ∃ (count : ℕ), count = 10 ∧ ∀ (x : ℤ), 6 > Real.sqrt x ∧ Real.sqrt x > 5 ↔ (x ≥ 26 ∧ x ≤ 35) := by
  sorry

end NUMINAMATH_GPT_int_values_satisfy_condition_l794_79422


namespace NUMINAMATH_GPT_problem_l794_79434

-- Definitions for the problem's conditions:
variables {a b c d : ℝ}

-- a and b are roots of x^2 + 68x + 1 = 0
axiom ha : a ^ 2 + 68 * a + 1 = 0
axiom hb : b ^ 2 + 68 * b + 1 = 0

-- c and d are roots of x^2 - 86x + 1 = 0
axiom hc : c ^ 2 - 86 * c + 1 = 0
axiom hd : d ^ 2 - 86 * d + 1 = 0

theorem problem : (a + c) * (b + c) * (a - d) * (b - d) = 2772 :=
sorry

end NUMINAMATH_GPT_problem_l794_79434


namespace NUMINAMATH_GPT_years_between_2000_and_3000_with_property_l794_79425

theorem years_between_2000_and_3000_with_property :
  ∃ n : ℕ, n = 143 ∧
  ∀ Y, 2000 ≤ Y ∧ Y ≤ 3000 → ∃ p q : ℕ, p + q = Y ∧ 2 * p = 5 * q →
  (2 * Y) % 7 = 0 :=
sorry

end NUMINAMATH_GPT_years_between_2000_and_3000_with_property_l794_79425


namespace NUMINAMATH_GPT_camden_dogs_fraction_l794_79407

def number_of_dogs (Justins_dogs : ℕ) (extra_dogs : ℕ) : ℕ := Justins_dogs + extra_dogs
def dogs_from_legs (total_legs : ℕ) (legs_per_dog : ℕ) : ℕ := total_legs / legs_per_dog
def fraction_of_dogs (dogs_camden : ℕ) (dogs_rico : ℕ) : ℚ := dogs_camden / dogs_rico

theorem camden_dogs_fraction (Justins_dogs : ℕ) (extra_dogs : ℕ) (total_legs_camden : ℕ) (legs_per_dog : ℕ) :
  Justins_dogs = 14 →
  extra_dogs = 10 →
  total_legs_camden = 72 →
  legs_per_dog = 4 →
  fraction_of_dogs (dogs_from_legs total_legs_camden legs_per_dog) (number_of_dogs Justins_dogs extra_dogs) = 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_camden_dogs_fraction_l794_79407


namespace NUMINAMATH_GPT_length_of_each_glass_pane_l794_79463

theorem length_of_each_glass_pane (panes : ℕ) (width : ℕ) (total_area : ℕ) 
    (H_panes : panes = 8) (H_width : width = 8) (H_total_area : total_area = 768) : 
    ∃ length : ℕ, length = 12 := by
  sorry

end NUMINAMATH_GPT_length_of_each_glass_pane_l794_79463


namespace NUMINAMATH_GPT_min_number_of_participants_l794_79476

theorem min_number_of_participants :
  ∃ n : ℕ, 
    (∃ x : ℕ, (3 * x + 1 = n) ∧ 
    (∃ y : ℕ, (4 * y + 1 = n) ∧ 
    (∃ z : ℕ, (5 * z + 1 = n)))) ∧
    n = 61 :=
by
  sorry

end NUMINAMATH_GPT_min_number_of_participants_l794_79476


namespace NUMINAMATH_GPT_union_sets_l794_79400

def M : Set ℕ := {0, 1, 3}
def N : Set ℕ := {x | ∃ a ∈ M, x = 3 * a}

theorem union_sets : M ∪ N = {0, 1, 3, 9} :=
by
  sorry

end NUMINAMATH_GPT_union_sets_l794_79400


namespace NUMINAMATH_GPT_blue_eyed_blonds_greater_than_population_proportion_l794_79426

variables {G_B Γ B N : ℝ}

theorem blue_eyed_blonds_greater_than_population_proportion (h : G_B / Γ > B / N) : G_B / B > Γ / N :=
sorry

end NUMINAMATH_GPT_blue_eyed_blonds_greater_than_population_proportion_l794_79426


namespace NUMINAMATH_GPT_always_true_statements_l794_79485

variable (a b c : ℝ)

theorem always_true_statements (h1 : a < 0) (h2 : a < b ∧ b ≤ 0) (h3 : b < c) : 
  (a + b < b + c) ∧ (c / a < 1) :=
by 
  sorry

end NUMINAMATH_GPT_always_true_statements_l794_79485


namespace NUMINAMATH_GPT_range_of_a_l794_79423

variable {α : Type*}

def in_interval (x : ℝ) (a b : ℝ) : Prop := a < x ∧ x < b

def A (a : ℝ) : Set ℝ := {-1, 0, a}

def B : Set ℝ := {x : ℝ | in_interval x 0 1}

theorem range_of_a (a : ℝ) (hA_B_nonempty : (A a ∩ B).Nonempty) : 0 < a ∧ a < 1 := 
sorry

end NUMINAMATH_GPT_range_of_a_l794_79423


namespace NUMINAMATH_GPT_leo_weight_l794_79474

theorem leo_weight 
  (L K E : ℝ)
  (h1 : L + 10 = 1.5 * K)
  (h2 : L + 10 = 0.75 * E)
  (h3 : L + K + E = 210) :
  L = 63.33 := 
sorry

end NUMINAMATH_GPT_leo_weight_l794_79474


namespace NUMINAMATH_GPT_hyperbola_eccentricity_l794_79443

theorem hyperbola_eccentricity (a b x₀ y₀ : ℝ) (h₁ : a > 0) (h₂ : b > 0)
  (h₃ : x₀^2 / a^2 - y₀^2 / b^2 = 1)
  (h₄ : a ≤ x₀ ∧ x₀ ≤ 2 * a)
  (h₅ : x₀ / a^2 * 0 - y₀ / b^2 * b = 1)
  (h₆ : - (a * a / (2 * b)) = 2) :
  (1 + b^2 / a^2 = 3) :=
sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_l794_79443


namespace NUMINAMATH_GPT_number_of_sides_l794_79499

theorem number_of_sides (n : ℕ) : 
  let a_1 := 6 
  let d := 5
  let a_n := a_1 + (n - 1) * d
  a_n = 5 * n + 1 := 
by
  sorry

end NUMINAMATH_GPT_number_of_sides_l794_79499


namespace NUMINAMATH_GPT_trig_identity_proof_l794_79401

theorem trig_identity_proof :
  let sin_95 := Real.cos (Real.pi / 36)
  let sin_65 := Real.cos (5 * Real.pi / 36)
  (Real.sin (Real.pi / 36) * Real.sin (5 * Real.pi / 36) - sin_95 * sin_65) = - (Real.sqrt 3) / 2 :=
by
  let sin_95 := Real.cos (Real.pi / 36)
  let sin_65 := Real.cos (5 * Real.pi / 36)
  sorry

end NUMINAMATH_GPT_trig_identity_proof_l794_79401


namespace NUMINAMATH_GPT_charlie_cost_per_gb_l794_79416

noncomputable def total_data_usage (w1 w2 w3 w4 : ℕ) : ℕ := w1 + w2 + w3 + w4

noncomputable def data_over_limit (total_data usage_limit: ℕ) : ℕ :=
  if total_data > usage_limit then total_data - usage_limit else 0

noncomputable def cost_per_gb (extra_cost data_over_limit: ℕ) : ℕ :=
  if data_over_limit > 0 then extra_cost / data_over_limit else 0

theorem charlie_cost_per_gb :
  let D := 8
  let w1 := 2
  let w2 := 3
  let w3 := 5
  let w4 := 10
  let C := 120
  let total_data := total_data_usage w1 w2 w3 w4
  let data_over := data_over_limit total_data D
  C / data_over = 10 := by
  -- Sorry to skip the proof
  sorry

end NUMINAMATH_GPT_charlie_cost_per_gb_l794_79416


namespace NUMINAMATH_GPT_spiders_make_webs_l794_79404

theorem spiders_make_webs :
  (∀ (s d : ℕ), s = 7 ∧ d = 7 → (∃ w : ℕ, w = s)) ∧
  (∀ (d w : ℕ), w = 1 ∧ d = 7 → (∃ s : ℕ, s = w)) →
  (∀ (s : ℕ), s = 1) :=
by
  sorry

end NUMINAMATH_GPT_spiders_make_webs_l794_79404


namespace NUMINAMATH_GPT_wrench_force_l794_79468

def force_inversely_proportional (f1 f2 : ℝ) (L1 L2 : ℝ) : Prop :=
  f1 * L1 = f2 * L2

theorem wrench_force
  (f1 : ℝ) (L1 : ℝ) (f2 : ℝ) (L2 : ℝ)
  (h1 : L1 = 12) (h2 : f1 = 450) (h3 : L2 = 18) (h_prop : force_inversely_proportional f1 f2 L1 L2) :
  f2 = 300 :=
by
  sorry

end NUMINAMATH_GPT_wrench_force_l794_79468


namespace NUMINAMATH_GPT_trigonometric_identity_l794_79457

theorem trigonometric_identity 
  (α : ℝ) 
  (h : 3 * Real.sin α + Real.cos α = 0) : 
  1 / (Real.cos α ^ 2 + 2 * Real.sin α * Real.cos α) = 10 / 3 := 
sorry

end NUMINAMATH_GPT_trigonometric_identity_l794_79457


namespace NUMINAMATH_GPT_probability_of_top_grade_product_l794_79438

-- Definitions for the problem conditions
def P_B : ℝ := 0.03
def P_C : ℝ := 0.01

-- Given that the sum of all probabilities is 1
axiom sum_of_probabilities (P_A P_B P_C : ℝ) : P_A + P_B + P_C = 1

-- Statement to be proved
theorem probability_of_top_grade_product : ∃ P_A : ℝ, P_A = 1 - P_B - P_C ∧ P_A = 0.96 :=
by
  -- Assuming the proof steps to derive the answer
  sorry

end NUMINAMATH_GPT_probability_of_top_grade_product_l794_79438


namespace NUMINAMATH_GPT_largest_divisor_n4_minus_5n2_plus_6_l794_79424

theorem largest_divisor_n4_minus_5n2_plus_6 :
  ∀ (n : ℤ), (n^4 - 5 * n^2 + 6) % 1 = 0 :=
by
  sorry

end NUMINAMATH_GPT_largest_divisor_n4_minus_5n2_plus_6_l794_79424


namespace NUMINAMATH_GPT_imaginary_part_of_i_mul_root_l794_79403

theorem imaginary_part_of_i_mul_root
  (z : ℂ) (hz : z^2 - 4 * z + 5 = 0) : (i * z).im = 2 := 
sorry

end NUMINAMATH_GPT_imaginary_part_of_i_mul_root_l794_79403


namespace NUMINAMATH_GPT_matrix_det_example_l794_79481

variable (A : Matrix (Fin 2) (Fin 2) ℤ) 
  (hA : A = ![![5, -4], ![2, 3]])

theorem matrix_det_example : Matrix.det A = 23 :=
by
  sorry

end NUMINAMATH_GPT_matrix_det_example_l794_79481


namespace NUMINAMATH_GPT_find_number_l794_79467

theorem find_number (x : ℝ) : (x^2 + 4 = 5 * x) → (x = 4 ∨ x = 1) :=
by
  sorry

end NUMINAMATH_GPT_find_number_l794_79467


namespace NUMINAMATH_GPT_initial_parts_planned_l794_79452

variable (x : ℕ)

theorem initial_parts_planned (x : ℕ) (h : 3 * x + (x + 5) + 100 = 675): x = 142 :=
by sorry

end NUMINAMATH_GPT_initial_parts_planned_l794_79452


namespace NUMINAMATH_GPT_multiplier_for_doberman_puppies_l794_79469

theorem multiplier_for_doberman_puppies 
  (D : ℕ) (S : ℕ) (M : ℝ) 
  (hD : D = 20) 
  (hS : S = 55) 
  (h : D * M + (D - S) = 90) : 
  M = 6.25 := 
by 
  sorry

end NUMINAMATH_GPT_multiplier_for_doberman_puppies_l794_79469


namespace NUMINAMATH_GPT_sufficient_not_necessary_l794_79492

theorem sufficient_not_necessary (a b : ℝ) : (a^2 + b^2 ≤ 2) → (-1 ≤ a * b ∧ a * b ≤ 1) ∧ ¬((-1 ≤ a * b ∧ a * b ≤ 1) → a^2 + b^2 ≤ 2) := 
by
  sorry

end NUMINAMATH_GPT_sufficient_not_necessary_l794_79492


namespace NUMINAMATH_GPT_biology_marks_l794_79427

theorem biology_marks (E M P C: ℝ) (A: ℝ) (N: ℕ) 
  (hE: E = 96) (hM: M = 98) (hP: P = 99) (hC: C = 100) (hA: A = 98.2) (hN: N = 5):
  (E + M + P + C + B) / N = A → B = 98 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_biology_marks_l794_79427


namespace NUMINAMATH_GPT_sample_size_is_30_l794_79498

-- Definitions based on conditions
def total_students : ℕ := 700 + 500 + 300
def students_first_grade : ℕ := 700
def students_sampled_first_grade : ℕ := 14
def sample_size (n : ℕ) : Prop := students_sampled_first_grade = (students_first_grade * n) / total_students

-- Theorem stating the proof problem
theorem sample_size_is_30 : sample_size 30 :=
by
  sorry

end NUMINAMATH_GPT_sample_size_is_30_l794_79498


namespace NUMINAMATH_GPT_self_descriptive_7_digit_first_digit_is_one_l794_79472

theorem self_descriptive_7_digit_first_digit_is_one
  (A B C D E F G : ℕ)
  (h_total : A + B + C + D + E + F + G = 7)
  (h_B : B = 2)
  (h_C : C = 1)
  (h_D : D = 1)
  (h_E : E = 0)
  (h_A_zeroes : A = (if E = 0 then 1 else 0)) :
  A = 1 :=
by
  sorry

end NUMINAMATH_GPT_self_descriptive_7_digit_first_digit_is_one_l794_79472


namespace NUMINAMATH_GPT_proof_problem_l794_79480

theorem proof_problem
  (x y a b c d : ℝ)
  (h1 : |x - 1| + (y + 2)^2 = 0)
  (h2 : a * b = 1)
  (h3 : c + d = 0) :
  (x + y)^3 - (-a * b)^2 + 3 * c + 3 * d = -2 :=
by
  -- The proof steps go here.
  sorry

end NUMINAMATH_GPT_proof_problem_l794_79480


namespace NUMINAMATH_GPT_chris_birthday_days_l794_79473

theorem chris_birthday_days (mod : ℕ → ℕ → ℕ) (day_of_week : ℕ → ℕ) :
  (mod 75 7 = 5) ∧ (mod 30 7 = 2) →
  (day_of_week 0 = 1) →
  (day_of_week 75 = 6) ∧ (day_of_week 30 = 3) := 
sorry

end NUMINAMATH_GPT_chris_birthday_days_l794_79473


namespace NUMINAMATH_GPT_quadratic_function_origin_l794_79406

theorem quadratic_function_origin {a b c : ℝ} :
  (∀ x, y = ax * x + bx * x + c → y = 0 → 0 = c ∧ b = 0) ∨ (c = 0) :=
sorry

end NUMINAMATH_GPT_quadratic_function_origin_l794_79406


namespace NUMINAMATH_GPT_line_through_two_points_l794_79412

theorem line_through_two_points :
  ∀ (A_1 B_1 A_2 B_2 : ℝ),
    (2 * A_1 + 3 * B_1 = 1) →
    (2 * A_2 + 3 * B_2 = 1) →
    (∀ (x y : ℝ), (2 * x + 3 * y = 1) → (x * (B_2 - B_1) + y * (A_1 - A_2) = A_1 * B_2 - A_2 * B_1)) :=
by 
  intros A_1 B_1 A_2 B_2 h1 h2 x y hxy
  sorry

end NUMINAMATH_GPT_line_through_two_points_l794_79412


namespace NUMINAMATH_GPT_no_common_solution_l794_79455

theorem no_common_solution 
  (x : ℝ) 
  (h1 : 8 * x^2 + 6 * x = 5) 
  (h2 : 3 * x + 2 = 0) : 
  False := 
by
  sorry

end NUMINAMATH_GPT_no_common_solution_l794_79455


namespace NUMINAMATH_GPT_find_range_of_a_l794_79414

-- Define the conditions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 - 4 * x + a^2 > 0
def q (a : ℝ) : Prop := a^2 - 5 * a - 6 ≥ 0

-- Define the proposition that one of p or q is true and the other is false
def p_or_q (a : ℝ) : Prop := p a ∨ q a
def not_p_and_q (a : ℝ) : Prop := ¬(p a ∧ q a)

-- Define the range of a
def range_of_a (a : ℝ) : Prop := (2 < a ∧ a < 6) ∨ (-2 ≤ a ∧ a ≤ -1)

-- Theorem statement
theorem find_range_of_a (a : ℝ) : p_or_q a ∧ not_p_and_q a → range_of_a a :=
by
  sorry

end NUMINAMATH_GPT_find_range_of_a_l794_79414


namespace NUMINAMATH_GPT_angle_x_l794_79451

-- Conditions
variable (ABC BAC CDE DCE : ℝ)
variable (h1 : ABC = 70)
variable (h2 : BAC = 50)
variable (h3 : CDE = 90)
variable (h4 : ∃ BCA : ℝ, DCE = BCA ∧ ABC + BAC + BCA = 180)

-- The statement to prove
theorem angle_x (x : ℝ) (h : ∃ BCA : ℝ, (ABC = 70) ∧ (BAC = 50) ∧ (CDE = 90) ∧ (DCE = BCA ∧ ABC + BAC + BCA = 180) ∧ (DCE + x = 90)) :
  x = 30 := by
  sorry

end NUMINAMATH_GPT_angle_x_l794_79451


namespace NUMINAMATH_GPT_base6_div_by_7_l794_79440

theorem base6_div_by_7 (k d : ℕ) (hk : 0 ≤ k ∧ k ≤ 5) (hd : 0 ≤ d ∧ d ≤ 5) (hkd : k = d) : 
  7 ∣ (217 * k + 42 * d) := 
by 
  rw [hkd]
  sorry

end NUMINAMATH_GPT_base6_div_by_7_l794_79440


namespace NUMINAMATH_GPT_largest_integer_divisible_example_1748_largest_n_1748_l794_79459

theorem largest_integer_divisible (n : ℕ) (h : (n + 12) ∣ (n^3 + 160)) : n ≤ 1748 :=
by
  sorry

theorem example_1748 : 1748^3 + 160 = 1760 * 3045738 :=
by
  sorry

theorem largest_n_1748 (n : ℕ) (h : 1748 ≤ n) : (n + 12) ∣ (n^3 + 160) :=
by
  sorry

end NUMINAMATH_GPT_largest_integer_divisible_example_1748_largest_n_1748_l794_79459


namespace NUMINAMATH_GPT_choose_18_4_eq_3060_l794_79479

/-- The number of ways to select 4 members from a group of 18 people (without regard to order). -/
theorem choose_18_4_eq_3060 : Nat.choose 18 4 = 3060 := 
by
  sorry

end NUMINAMATH_GPT_choose_18_4_eq_3060_l794_79479


namespace NUMINAMATH_GPT_sum_of_squares_of_tom_rates_l794_79420

theorem sum_of_squares_of_tom_rates :
  ∃ r b k : ℕ, 3 * r + 4 * b + 2 * k = 104 ∧
               3 * r + 6 * b + 2 * k = 140 ∧
               r^2 + b^2 + k^2 = 440 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_of_tom_rates_l794_79420


namespace NUMINAMATH_GPT_total_additions_in_2_hours_30_minutes_l794_79489

def additions_rate : ℕ := 15000

def time_in_seconds : ℕ := 2 * 3600 + 30 * 60

def total_additions : ℕ := additions_rate * time_in_seconds

theorem total_additions_in_2_hours_30_minutes :
  total_additions = 135000000 :=
by
  -- Non-trivial proof skipped
  sorry

end NUMINAMATH_GPT_total_additions_in_2_hours_30_minutes_l794_79489


namespace NUMINAMATH_GPT_total_viewing_time_amaya_l794_79488

/-- The total viewing time Amaya spent, including rewinding, was 170 minutes. -/
theorem total_viewing_time_amaya 
  (u1 u2 u3 u4 u5 r1 r2 r3 r4 : ℕ)
  (h1 : u1 = 35)
  (h2 : u2 = 45)
  (h3 : u3 = 25)
  (h4 : u4 = 15)
  (h5 : u5 = 20)
  (hr1 : r1 = 5)
  (hr2 : r2 = 7)
  (hr3 : r3 = 10)
  (hr4 : r4 = 8) :
  u1 + u2 + u3 + u4 + u5 + r1 + r2 + r3 + r4 = 170 :=
by
  sorry

end NUMINAMATH_GPT_total_viewing_time_amaya_l794_79488

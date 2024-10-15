import Mathlib

namespace NUMINAMATH_GPT_line_intersects_circle_l1993_199372

theorem line_intersects_circle (k : ℝ) :
  ∃ x y : ℝ, (x^2 + y^2 - 2*y = 0) ∧ (y - 1 = k * (x - 1)) :=
sorry

end NUMINAMATH_GPT_line_intersects_circle_l1993_199372


namespace NUMINAMATH_GPT_betsy_to_cindy_ratio_l1993_199381

-- Definitions based on the conditions
def cindy_time : ℕ := 12
def tina_time : ℕ := cindy_time + 6
def betsy_time : ℕ := tina_time / 3

-- Theorem statement to prove
theorem betsy_to_cindy_ratio :
  (betsy_time : ℚ) / cindy_time = 1 / 2 :=
by sorry

end NUMINAMATH_GPT_betsy_to_cindy_ratio_l1993_199381


namespace NUMINAMATH_GPT_savings_after_20_days_l1993_199355

-- Definitions based on conditions
def daily_earnings : ℕ := 80
def days_worked : ℕ := 20
def total_spent : ℕ := 1360

-- Prove the savings after 20 days
theorem savings_after_20_days : daily_earnings * days_worked - total_spent = 240 :=
by
  sorry

end NUMINAMATH_GPT_savings_after_20_days_l1993_199355


namespace NUMINAMATH_GPT_jacob_age_proof_l1993_199382

-- Definitions based on given conditions
def rehana_current_age : ℕ := 25
def rehana_age_in_five_years : ℕ := rehana_current_age + 5
def phoebe_age_in_five_years : ℕ := rehana_age_in_five_years / 3
def phoebe_current_age : ℕ := phoebe_age_in_five_years - 5
def jacob_current_age : ℕ := 3 * phoebe_current_age / 5

-- Statement of the problem
theorem jacob_age_proof :
  jacob_current_age = 3 :=
by 
  -- Skipping the proof for now
  sorry

end NUMINAMATH_GPT_jacob_age_proof_l1993_199382


namespace NUMINAMATH_GPT_symmetric_circle_equation_l1993_199313

theorem symmetric_circle_equation :
  ∀ (x y : ℝ), (x + 2) ^ 2 + y ^ 2 = 5 → (x - 2) ^ 2 + y ^ 2 = 5 :=
by 
  sorry

end NUMINAMATH_GPT_symmetric_circle_equation_l1993_199313


namespace NUMINAMATH_GPT_geometric_sequence_problem_l1993_199398

noncomputable def geometric_sequence_solution (a_1 a_2 a_3 a_4 a_5 q : ℝ) : Prop :=
  (a_5 - a_1 = 15) ∧
  (a_4 - a_2 = 6) ∧
  (a_3 = 4 ∧ q = 2 ∨ a_3 = -4 ∧ q = 1/2)

theorem geometric_sequence_problem :
  ∃ a_1 a_2 a_3 a_4 a_5 q : ℝ, geometric_sequence_solution a_1 a_2 a_3 a_4 a_5 q :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_problem_l1993_199398


namespace NUMINAMATH_GPT_min_small_containers_needed_l1993_199353

def medium_container_capacity : ℕ := 450
def small_container_capacity : ℕ := 28

theorem min_small_containers_needed : ⌈(medium_container_capacity : ℝ) / small_container_capacity⌉ = 17 :=
by
  sorry

end NUMINAMATH_GPT_min_small_containers_needed_l1993_199353


namespace NUMINAMATH_GPT_part1_part2_part3_l1993_199347

-- Part 1
theorem part1 (a b : ℝ) : 3*(a-b)^2 - 6*(a-b)^2 + 2*(a-b)^2 = -(a-b)^2 :=
sorry

-- Part 2
theorem part2 (x y : ℝ) (h : x^2 - 2*y = 4) : 3*x^2 - 6*y - 21 = -9 :=
sorry

-- Part 3
theorem part3 (a b c d : ℝ) (h1 : a - 5*b = 3) (h2 : 5*b - 3*c = -5) (h3 : 3*c - d = 10) : 
  (a - 3*c) + (5*b - d) - (5*b - 3*c) = 8 :=
sorry

end NUMINAMATH_GPT_part1_part2_part3_l1993_199347


namespace NUMINAMATH_GPT_problem_statement_l1993_199342

noncomputable def a : ℝ := 6 * Real.sqrt 2
noncomputable def b : ℝ := 18 * Real.sqrt 2
noncomputable def c : ℝ := 6 * Real.sqrt 21
noncomputable def d : ℝ := 24 * Real.sqrt 2
noncomputable def e : ℝ := 48 * Real.sqrt 2
noncomputable def N : ℝ := 756 * Real.sqrt 10

axiom condition_a : a^2 + b^2 + c^2 + d^2 + e^2 = 504
axiom positive_numbers : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0

theorem problem_statement : N + a + b + c + d + e = 96 * Real.sqrt 2 + 6 * Real.sqrt 21 + 756 * Real.sqrt 10 :=
by
  -- We'll insert the proof here later
  sorry

end NUMINAMATH_GPT_problem_statement_l1993_199342


namespace NUMINAMATH_GPT_tan_alpha_fraction_l1993_199326

theorem tan_alpha_fraction (α : ℝ) (hα1 : 0 < α ∧ α < (Real.pi / 2)) (hα2 : Real.tan (2 * α) = Real.cos α / (2 - Real.sin α)) :
  Real.tan α = Real.sqrt 15 / 15 :=
by
  sorry

end NUMINAMATH_GPT_tan_alpha_fraction_l1993_199326


namespace NUMINAMATH_GPT_min_distance_point_to_line_l1993_199309

theorem min_distance_point_to_line :
    ∀ (x y : ℝ), (x^2 + y^2 - 6 * x - 4 * y + 12 = 0) -> 
    (3 * x + 4 * y - 2 = 0) -> 
    ∃ d: ℝ, d = 2 :=
by sorry

end NUMINAMATH_GPT_min_distance_point_to_line_l1993_199309


namespace NUMINAMATH_GPT_pepperoni_slices_left_l1993_199323

theorem pepperoni_slices_left :
  ∀ (total_friends : ℕ) (total_slices : ℕ) (cheese_left : ℕ),
    (total_friends = 4) →
    (total_slices = 16) →
    (cheese_left = 7) →
    (∃ p_slices_left : ℕ, p_slices_left = 4) :=
by
  intros total_friends total_slices cheese_left h_friends h_slices h_cheese
  sorry

end NUMINAMATH_GPT_pepperoni_slices_left_l1993_199323


namespace NUMINAMATH_GPT_prime_factors_of_69_l1993_199337

theorem prime_factors_of_69 
  (prime : ℕ → Prop)
  (is_prime : ∀ n, prime n ↔ n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7 ∨ n = 11 ∨ 
                        n = 13 ∨ n = 17 ∨ n = 19 ∨ n = 23)
  (x y : ℕ)
  (h1 : 15 < 69)
  (h2 : 69 < 70)
  (h3 : prime y)
  (h4 : 13 < y)
  (h5 : y < 25)
  (h6 : 69 = x * y)
  : prime x ∧ x = 3 := 
sorry

end NUMINAMATH_GPT_prime_factors_of_69_l1993_199337


namespace NUMINAMATH_GPT_original_price_of_sarees_l1993_199367

theorem original_price_of_sarees (P : ℝ):
  (0.80 * P) * 0.95 = 152 → P = 200 :=
by
  intro h1
  -- You can omit the proof here because the task requires only the statement.
  sorry

end NUMINAMATH_GPT_original_price_of_sarees_l1993_199367


namespace NUMINAMATH_GPT_ellipse_equation_l1993_199357

theorem ellipse_equation (a b c c1 : ℝ)
  (h_hyperbola_eq : ∀ x y, (y^2 / 4 - x^2 / 12 = 1))
  (h_sum_eccentricities : (c / a) + (c1 / 2) = 13 / 5)
  (h_foci_x_axis : c1 = 4) :
  (a = 5 ∧ b = 4 ∧ c = 3) → 
  ∀ x y, (x^2 / 25 + y^2 / 16 = 1) :=
by
  sorry

end NUMINAMATH_GPT_ellipse_equation_l1993_199357


namespace NUMINAMATH_GPT_greatest_integer_c_l1993_199311

theorem greatest_integer_c (c : ℤ) :
  (∀ x : ℝ, x^2 + (c : ℝ) * x + 10 ≠ 0) → c = 6 :=
by
  sorry

end NUMINAMATH_GPT_greatest_integer_c_l1993_199311


namespace NUMINAMATH_GPT_price_of_5_pound_bag_l1993_199350

-- Definitions based on conditions
def price_10_pound_bag : ℝ := 20.42
def price_25_pound_bag : ℝ := 32.25
def min_pounds : ℝ := 65
def max_pounds : ℝ := 80
def total_min_cost : ℝ := 98.77

-- Define the sought price of the 5-pound bag in the hypothesis
variable {price_5_pound_bag : ℝ}

-- The theorem to prove based on the given conditions
theorem price_of_5_pound_bag
  (h₁ : price_10_pound_bag = 20.42)
  (h₂ : price_25_pound_bag = 32.25)
  (h₃ : min_pounds = 65)
  (h₄ : max_pounds = 80)
  (h₅ : total_min_cost = 98.77) :
  price_5_pound_bag = 2.02 :=
sorry

end NUMINAMATH_GPT_price_of_5_pound_bag_l1993_199350


namespace NUMINAMATH_GPT_find_huabei_number_l1993_199376

theorem find_huabei_number :
  ∃ (hua bei sai : ℕ), 
    (hua ≠ 4 ∧ hua ≠ 8 ∧ bei ≠ 4 ∧ bei ≠ 8 ∧ sai ≠ 4 ∧ sai ≠ 8) ∧
    (hua ≠ bei ∧ hua ≠ sai ∧ bei ≠ sai) ∧
    (1 ≤ hua ∧ hua ≤ 9 ∧ 1 ≤ bei ∧ bei ≤ 9 ∧ 1 ≤ sai ∧ sai ≤ 9) ∧
    ((100 * hua + 10 * bei + sai) = 7632) :=
sorry

end NUMINAMATH_GPT_find_huabei_number_l1993_199376


namespace NUMINAMATH_GPT_range_of_k_l1993_199325

-- Define the set M
def M := {x : ℝ | -1 ≤ x ∧ x ≤ 7}

-- Define the set N based on k
def N (k : ℝ) := {x : ℝ | k + 1 ≤ x ∧ x ≤ 2 * k - 1}

-- The main statement to prove
theorem range_of_k (k : ℝ) : M ∩ N k = ∅ → 6 < k :=
by
  -- skipping the proof as instructed
  sorry

end NUMINAMATH_GPT_range_of_k_l1993_199325


namespace NUMINAMATH_GPT_olivia_cookies_total_l1993_199388

def cookies_total (baggie_cookie_count : ℝ) (chocolate_chip_cookies : ℝ) 
                  (baggies_oatmeal_cookies : ℝ) (total_cookies : ℝ) : Prop :=
  let oatmeal_cookies := baggies_oatmeal_cookies * baggie_cookie_count
  oatmeal_cookies + chocolate_chip_cookies = total_cookies

theorem olivia_cookies_total :
  cookies_total 9.0 13.0 3.111111111 41.0 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_olivia_cookies_total_l1993_199388


namespace NUMINAMATH_GPT_largest_multiple_11_lt_neg85_l1993_199314

-- Define the conditions: a multiple of 11 and smaller than -85
def largest_multiple_lt (m n : Int) : Int :=
  let k := (m / n) - 1
  n * k

-- Define our specific problem
theorem largest_multiple_11_lt_neg85 : largest_multiple_lt (-85) 11 = -88 := 
  by
  sorry

end NUMINAMATH_GPT_largest_multiple_11_lt_neg85_l1993_199314


namespace NUMINAMATH_GPT_complex_magnitude_l1993_199360

open Complex

noncomputable def complexZ : ℂ := sorry -- Definition of complex number z

theorem complex_magnitude (z : ℂ) (h : (1 + 2 * Complex.I) * z = -3 + 4 * Complex.I) : abs z = Real.sqrt 5 :=
sorry

end NUMINAMATH_GPT_complex_magnitude_l1993_199360


namespace NUMINAMATH_GPT_abs_inequality_solution_l1993_199338

theorem abs_inequality_solution (x : ℝ) : |2 * x + 1| - 2 * |x - 1| > 0 ↔ x > 1 / 4 :=
sorry

end NUMINAMATH_GPT_abs_inequality_solution_l1993_199338


namespace NUMINAMATH_GPT_largest_cyclic_decimal_l1993_199390

def digits_on_circle := [1, 3, 9, 5, 7, 9, 1, 3, 9, 5, 7, 1]

def max_cyclic_decimal : ℕ := sorry

theorem largest_cyclic_decimal :
  max_cyclic_decimal = 957913 :=
sorry

end NUMINAMATH_GPT_largest_cyclic_decimal_l1993_199390


namespace NUMINAMATH_GPT_prob_first_diamond_second_ace_or_face_l1993_199320

theorem prob_first_diamond_second_ace_or_face :
  let deck_size := 52
  let first_card_diamonds := 13 / deck_size
  let prob_ace_after_diamond := 4 / (deck_size - 1)
  let prob_face_after_diamond := 12 / (deck_size - 1)
  first_card_diamonds * (prob_ace_after_diamond + prob_face_after_diamond) = 68 / 867 :=
by
  let deck_size := 52
  let first_card_diamonds := 13 / deck_size
  let prob_ace_after_diamond := 4 / (deck_size - 1)
  let prob_face_after_diamond := 12 / (deck_size - 1)
  sorry

end NUMINAMATH_GPT_prob_first_diamond_second_ace_or_face_l1993_199320


namespace NUMINAMATH_GPT_sequence_sum_l1993_199336

def arithmetic_seq (a₀ : ℕ) (d : ℕ) : ℕ → ℕ
  | n => a₀ + n * d

def geometric_seq (b₀ : ℕ) (r : ℕ) : ℕ → ℕ
  | n => b₀ * r^(n)

theorem sequence_sum :
  let a : ℕ → ℕ := arithmetic_seq 3 1
  let b : ℕ → ℕ := geometric_seq 1 2
  b (a 0) + b (a 1) + b (a 2) + b (a 3) = 60 :=
  by
    let a : ℕ → ℕ := arithmetic_seq 3 1
    let b : ℕ → ℕ := geometric_seq 1 2
    have h₀ : a 0 = 3 := by rfl
    have h₁ : a 1 = 4 := by rfl
    have h₂ : a 2 = 5 := by rfl
    have h₃ : a 3 = 6 := by rfl
    have hsum : b 3 + b 4 + b 5 + b 6 = 60 := by sorry
    exact hsum

end NUMINAMATH_GPT_sequence_sum_l1993_199336


namespace NUMINAMATH_GPT_problem_statement_l1993_199308

theorem problem_statement (x : ℝ) (h : x = -3) :
  (5 + x * (5 + x) - 5^2) / (x - 5 + x^2) = -26 := by
  rw [h]
  sorry

end NUMINAMATH_GPT_problem_statement_l1993_199308


namespace NUMINAMATH_GPT_number_of_pages_in_book_l1993_199387

-- Define the conditions using variables and hypotheses
variables (P : ℝ) (h1 : 0.30 * P = 150)

-- State the theorem to be proved
theorem number_of_pages_in_book : P = 500 :=
by
  -- Proof would go here, but we use sorry to skip it
  sorry

end NUMINAMATH_GPT_number_of_pages_in_book_l1993_199387


namespace NUMINAMATH_GPT_value_of_x_l1993_199375

theorem value_of_x (x y : ℝ) (h1 : x - y = 10) (h2 : x + y = 8) : x = 9 :=
by
  sorry

end NUMINAMATH_GPT_value_of_x_l1993_199375


namespace NUMINAMATH_GPT_rhombus_diagonal_l1993_199312

/-- Given a rhombus with one diagonal being 11 cm and the area of the rhombus being 88 cm²,
prove that the length of the other diagonal is 16 cm. -/
theorem rhombus_diagonal 
  (d1 : ℝ) (d2 : ℝ) (area : ℝ)
  (h_d1 : d1 = 11)
  (h_area : area = 88)
  (h_area_eq : area = (d1 * d2) / 2) : d2 = 16 :=
sorry

end NUMINAMATH_GPT_rhombus_diagonal_l1993_199312


namespace NUMINAMATH_GPT_nicky_cards_value_l1993_199301

theorem nicky_cards_value 
  (x : ℝ)
  (h : 21 = 2 * x + 5) : 
  x = 8 := by
  sorry

end NUMINAMATH_GPT_nicky_cards_value_l1993_199301


namespace NUMINAMATH_GPT_find_moles_of_NaCl_l1993_199322

-- Define the chemical reaction as an equation
def chemical_reaction (NaCl KNO3 NaNO3 KCl : ℕ) : Prop :=
  NaCl + KNO3 = NaNO3 + KCl

-- Define the problem conditions
def problem_conditions (naCl : ℕ) : Prop :=
  ∃ (kno3 naNo3 kcl : ℕ),
    kno3 = 3 ∧
    naNo3 = 3 ∧
    chemical_reaction naCl kno3 naNo3 kcl

-- Define the goal statement
theorem find_moles_of_NaCl (naCl : ℕ) : problem_conditions naCl → naCl = 3 :=
by
  sorry -- proof to be filled in later

end NUMINAMATH_GPT_find_moles_of_NaCl_l1993_199322


namespace NUMINAMATH_GPT_point_on_same_side_as_l1993_199377

def f (x y : ℝ) : ℝ := 2 * x - y + 1

theorem point_on_same_side_as (x1 y1 : ℝ) (h : f 1 2 > 0) : f 1 0 > 0 := sorry

end NUMINAMATH_GPT_point_on_same_side_as_l1993_199377


namespace NUMINAMATH_GPT_tenth_term_geom_seq_l1993_199383

theorem tenth_term_geom_seq :
  let a := (5 : ℚ)
  let r := (4 / 3 : ℚ)
  let n := 10
  (a * r^(n - 1)) = (1310720 / 19683 : ℚ) :=
by
  sorry

end NUMINAMATH_GPT_tenth_term_geom_seq_l1993_199383


namespace NUMINAMATH_GPT_area_of_inscribed_octagon_l1993_199329

-- Define the given conditions and required proof
theorem area_of_inscribed_octagon (r : ℝ) (h : π * r^2 = 400 * π) :
  let A := r^2 * (1 + Real.sqrt 2)
  A = 20^2 * (1 + Real.sqrt 2) :=
by 
  sorry

end NUMINAMATH_GPT_area_of_inscribed_octagon_l1993_199329


namespace NUMINAMATH_GPT_original_number_is_28_l1993_199356

theorem original_number_is_28 (N : ℤ) :
  (∃ k : ℤ, N - 11 = 17 * k) → N = 28 :=
by
  intro h
  obtain ⟨k, h₁⟩ := h
  have h₂: N = 17 * k + 11 := by linarith
  have h₃: k = 1 := sorry
  linarith [h₃]
 
end NUMINAMATH_GPT_original_number_is_28_l1993_199356


namespace NUMINAMATH_GPT_sum_of_variables_l1993_199304

theorem sum_of_variables (x y z : ℝ) (h₁ : x + y = 1) (h₂ : y + z = 1) (h₃ : z + x = 1) : x + y + z = 3 / 2 := 
sorry

end NUMINAMATH_GPT_sum_of_variables_l1993_199304


namespace NUMINAMATH_GPT_part_a_39x55_5x11_l1993_199332

theorem part_a_39x55_5x11 :
  ¬ (∃ (a1 a2 b1 b2 : ℕ), 
    39 = 5 * a1 + 11 * b1 ∧ 
    55 = 5 * a2 + 11 * b2) := 
  by sorry

end NUMINAMATH_GPT_part_a_39x55_5x11_l1993_199332


namespace NUMINAMATH_GPT_reception_time_l1993_199384

-- Definitions of conditions
def noon : ℕ := 12 * 60 -- define noon in minutes
def rabbit_walk_speed (v : ℕ) : Prop := v > 0
def rabbit_run_speed (v : ℕ) : Prop := 2 * v > 0
def distance (D : ℕ) : Prop := D > 0
def delay (minutes : ℕ) : Prop := minutes = 10

-- Definition of the problem
theorem reception_time (v D : ℕ) (h_v : rabbit_walk_speed v) (h_D : distance D) (h_delay : delay 10) :
  noon + (D / v) * 2 / 3 = 12 * 60 + 40 :=
by sorry

end NUMINAMATH_GPT_reception_time_l1993_199384


namespace NUMINAMATH_GPT_angle_B_l1993_199333

open Set

variables {Point Line : Type}

variable (l m n p : Line)
variable (A B C D : Point)
variable (angle : Point → Point → Point → ℝ)

-- Definitions of the conditions
def parallel (x y : Line) : Prop := sorry
def intersects (x y : Line) (P : Point) : Prop := sorry
def measure_angle (P Q R : Point) : ℝ := sorry

-- Assumptions based on conditions
axiom parallel_lm : parallel l m
axiom intersection_n_l : intersects n l A
axiom angle_A : measure_angle B A D = 140
axiom intersection_p_m : intersects p m C
axiom angle_C : measure_angle A C B = 70
axiom intersection_p_l : intersects p l D
axiom not_parallel_np : ¬ parallel n p

-- Proof goal
theorem angle_B : measure_angle C B D = 140 := sorry

end NUMINAMATH_GPT_angle_B_l1993_199333


namespace NUMINAMATH_GPT_probability_perfect_square_l1993_199310

theorem probability_perfect_square (choose_numbers : Finset (Fin 49)) (ticket : Finset (Fin 49))
  (h_choose_size : choose_numbers.card = 6) 
  (h_ticket_size : ticket.card = 6)
  (h_choose_square : ∃ (n : ℕ), (choose_numbers.prod id = n * n))
  (h_ticket_square : ∃ (m : ℕ), (ticket.prod id = m * m)) :
  ∃ T, (1 / T = 1 / T) :=
by
  sorry

end NUMINAMATH_GPT_probability_perfect_square_l1993_199310


namespace NUMINAMATH_GPT_problem_l1993_199399

theorem problem (x y : ℝ) 
  (h1 : |x + y - 9| = -(2 * x - y + 3) ^ 2) :
  x = 2 ∧ y = 7 :=
sorry

end NUMINAMATH_GPT_problem_l1993_199399


namespace NUMINAMATH_GPT_cos_five_pi_over_six_l1993_199334

theorem cos_five_pi_over_six :
  Real.cos (5 * Real.pi / 6) = -(Real.sqrt 3 / 2) :=
sorry

end NUMINAMATH_GPT_cos_five_pi_over_six_l1993_199334


namespace NUMINAMATH_GPT_arithmetic_progression_15th_term_l1993_199330

theorem arithmetic_progression_15th_term :
  let a := 2
  let d := 3
  let n := 15
  a + (n - 1) * d = 44 :=
by
  let a := 2
  let d := 3
  let n := 15
  sorry

end NUMINAMATH_GPT_arithmetic_progression_15th_term_l1993_199330


namespace NUMINAMATH_GPT_score_order_l1993_199321

variables (L N O P : ℕ)

def conditions : Prop := 
  O = L ∧ 
  N < max O P ∧ 
  P > L

theorem score_order (h : conditions L N O P) : N < O ∧ O < P :=
by
  sorry

end NUMINAMATH_GPT_score_order_l1993_199321


namespace NUMINAMATH_GPT_symmetric_circle_l1993_199317

theorem symmetric_circle
    (x y : ℝ)
    (circle_eq : x^2 + y^2 + 4 * x - 1 = 0) :
    (x - 2)^2 + y^2 = 5 :=
sorry

end NUMINAMATH_GPT_symmetric_circle_l1993_199317


namespace NUMINAMATH_GPT_ellipse_eccentricity_l1993_199324

-- Define the geometric sequence condition and the ellipse properties
theorem ellipse_eccentricity :
  ∀ (a b c e : ℝ), 
  (b^2 = a * c) ∧ (a^2 - c^2 = b^2) ∧ (e = c / a) ∧ (0 < e ∧ e < 1) →
  e = (Real.sqrt 5 - 1) / 2 := 
by 
  sorry

end NUMINAMATH_GPT_ellipse_eccentricity_l1993_199324


namespace NUMINAMATH_GPT_power_mod_eq_remainder_l1993_199352

theorem power_mod_eq_remainder (b m e : ℕ) (hb : b = 17) (hm : m = 23) (he : e = 2090) : 
  b^e % m = 12 := 
  by sorry

end NUMINAMATH_GPT_power_mod_eq_remainder_l1993_199352


namespace NUMINAMATH_GPT_tan_sum_inequality_l1993_199316

noncomputable def pi : ℝ := Real.pi

theorem tan_sum_inequality (x α : ℝ) (hx1 : 0 ≤ x) (hx2 : x ≤ pi / 2) (hα1 : pi / 6 < α) (hα2 : α < pi / 3) :
  Real.tan (pi * (Real.sin x) / (4 * Real.sin α)) + Real.tan (pi * (Real.cos x) / (4 * Real.cos α)) > 1 :=
by
  sorry

end NUMINAMATH_GPT_tan_sum_inequality_l1993_199316


namespace NUMINAMATH_GPT_find_b_l1993_199358

theorem find_b (a b : ℕ) (h1 : a = 105) (h2 : a ^ 3 = 21 * 25 * 315 * b) : b = 7 :=
by
  -- The actual proof would go here
  sorry

end NUMINAMATH_GPT_find_b_l1993_199358


namespace NUMINAMATH_GPT_total_books_l1993_199389

theorem total_books (x : ℕ) (h1 : 3 * x + 2 * x + (3 / 2) * x > 3000) : 
  ∃ (T : ℕ), T = 3 * x + 2 * x + (3 / 2) * x ∧ T > 3000 ∧ T = 3003 := 
by 
  -- Our theorem states there exists an integer T such that the total number of books is 3003.
  sorry

end NUMINAMATH_GPT_total_books_l1993_199389


namespace NUMINAMATH_GPT_calculate_expression_l1993_199344

theorem calculate_expression (a b c d : ℤ) (h1 : 3^0 = 1) (h2 : (-1 / 2 : ℚ)^(-2 : ℤ) = 4) : 
  (202 : ℤ) * 3^0 + (-1 / 2 : ℚ)^(-2 : ℤ) = 206 :=
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l1993_199344


namespace NUMINAMATH_GPT_min_value_expression_l1993_199396

theorem min_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 1) : 
  (x^2 / (1 + x) + y^2 / (1 + y) + z^2 / (1 + z) >= 1 / 4) ∧ (x = 1/3 ∧ y = 1/3 ∧ z = 1/3 → x^2 / (1 + x) + y^2 / (1 + y) + z^2 / (1 + z) = 1 / 4) :=
sorry

end NUMINAMATH_GPT_min_value_expression_l1993_199396


namespace NUMINAMATH_GPT_problem_part_1_problem_part_2_l1993_199362

def f (x m : ℝ) := 2 * x^2 + (2 - m) * x - m
def g (x m : ℝ) := x^2 - x + 2 * m

theorem problem_part_1 (x : ℝ) : f x 1 > 0 ↔ (x > 1/2 ∨ x < -1) :=
by sorry

theorem problem_part_2 {m x : ℝ} (hm : 0 < m) : f x m ≤ g x m ↔ (-3 ≤ x ∧ x ≤ m) :=
by sorry

end NUMINAMATH_GPT_problem_part_1_problem_part_2_l1993_199362


namespace NUMINAMATH_GPT_point_in_third_quadrant_l1993_199361

theorem point_in_third_quadrant (m n : ℝ) (h1 : m > 0) (h2 : n > 0) : (-m < 0) ∧ (-n < 0) :=
by
  sorry

end NUMINAMATH_GPT_point_in_third_quadrant_l1993_199361


namespace NUMINAMATH_GPT_min_value_xy_l1993_199393

theorem min_value_xy (x y : ℝ) (h : 1 / x + 2 / y = Real.sqrt (x * y)) : x * y ≥ 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_min_value_xy_l1993_199393


namespace NUMINAMATH_GPT_sam_distance_when_meeting_l1993_199365

theorem sam_distance_when_meeting :
  ∃ t : ℝ, (35 = 2 * t + 5 * t) ∧ (5 * t = 25) :=
by
  sorry

end NUMINAMATH_GPT_sam_distance_when_meeting_l1993_199365


namespace NUMINAMATH_GPT_relationship_depends_on_b_l1993_199364

theorem relationship_depends_on_b (a b : ℝ) : 
  (a + b > a - b ∨ a + b < a - b ∨ a + b = a - b) ↔ (b > 0 ∨ b < 0 ∨ b = 0) :=
by
  sorry

end NUMINAMATH_GPT_relationship_depends_on_b_l1993_199364


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1993_199306

theorem solution_set_of_inequality (x : ℝ) : (0 < x ∧ x < 1/3) ↔ (1/x > 3) := 
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1993_199306


namespace NUMINAMATH_GPT_book_price_l1993_199307

theorem book_price (P : ℝ) : 
  (3 * 12 * P - 500 = 220) → 
  P = 20 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_book_price_l1993_199307


namespace NUMINAMATH_GPT_rate_of_mangoes_per_kg_l1993_199368

variable (grapes_qty : ℕ := 8)
variable (grapes_rate_per_kg : ℕ := 70)
variable (mangoes_qty : ℕ := 9)
variable (total_amount_paid : ℕ := 1055)

theorem rate_of_mangoes_per_kg :
  (total_amount_paid - grapes_qty * grapes_rate_per_kg) / mangoes_qty = 55 :=
by
  sorry

end NUMINAMATH_GPT_rate_of_mangoes_per_kg_l1993_199368


namespace NUMINAMATH_GPT_tigers_in_zoo_l1993_199370

-- Given definitions
def ratio_lions_tigers := 3 / 4
def number_of_lions := 21
def number_of_tigers := 28

-- Problem statement
theorem tigers_in_zoo : (number_of_lions : ℚ) / 3 * 4 = number_of_tigers := by
  sorry

end NUMINAMATH_GPT_tigers_in_zoo_l1993_199370


namespace NUMINAMATH_GPT_total_profit_calculation_l1993_199369

variables {I_B T_B : ℝ}

-- Conditions as definitions
def investment_A (I_B : ℝ) : ℝ := 3 * I_B
def period_A (T_B : ℝ) : ℝ := 2 * T_B
def profit_B (I_B T_B : ℝ) : ℝ := I_B * T_B
def total_profit (I_B T_B : ℝ) : ℝ := 7 * I_B * T_B

-- To prove
theorem total_profit_calculation
  (h1 : investment_A I_B = 3 * I_B)
  (h2 : period_A T_B = 2 * T_B)
  (h3 : profit_B I_B T_B = 4000)
  : total_profit I_B T_B = 28000 := by
  sorry

end NUMINAMATH_GPT_total_profit_calculation_l1993_199369


namespace NUMINAMATH_GPT_determine_x_l1993_199380

theorem determine_x (x : ℚ) (h : ∀ (y : ℚ), 10 * x * y - 18 * y + x - 2 = 0) : x = 9 / 5 :=
sorry

end NUMINAMATH_GPT_determine_x_l1993_199380


namespace NUMINAMATH_GPT_solve_problem_l1993_199346

def problem_statement : Prop :=
  ⌊ (2011^3 : ℝ) / (2009 * 2010) - (2009^3 : ℝ) / (2010 * 2011) ⌋ = 8

theorem solve_problem : problem_statement := 
  by sorry

end NUMINAMATH_GPT_solve_problem_l1993_199346


namespace NUMINAMATH_GPT_part1_part2_a_eq_1_part2_a_gt_1_part2_a_lt_1_l1993_199348

section part1
variable (x : ℝ)

theorem part1 (a : ℝ) (h : a = 2) : (x + a) * (x - 2 * a + 1) < 0 ↔ -2 < x ∧ x < 3 :=
by
  sorry
end part1

section part2
variable (x a : ℝ)

-- Case: a = 1
theorem part2_a_eq_1 (h : a = 1) : (x - 1) * (x - 2 * a + 1) < 0 ↔ False :=
by
  sorry

-- Case: a > 1
theorem part2_a_gt_1 (h : a > 1) : (x - 1) * (x - 2 * a + 1) < 0 ↔ 1 < x ∧ x < 2 * a - 1 :=
by
  sorry

-- Case: a < 1
theorem part2_a_lt_1 (h : a < 1) : (x - 1) * (x - 2 * a + 1) < 0 ↔ 2 * a - 1 < x ∧ x < 1 :=
by
  sorry
end part2

end NUMINAMATH_GPT_part1_part2_a_eq_1_part2_a_gt_1_part2_a_lt_1_l1993_199348


namespace NUMINAMATH_GPT_salary_of_N_l1993_199379

theorem salary_of_N (total_salary : ℝ) (percent_M_from_N : ℝ) (N_salary : ℝ) : 
  (percent_M_from_N * N_salary + N_salary = total_salary) → (N_salary = 280) :=
by
  sorry

end NUMINAMATH_GPT_salary_of_N_l1993_199379


namespace NUMINAMATH_GPT_train_pass_bridge_in_36_seconds_l1993_199343

def train_length : ℝ := 360 -- meters
def bridge_length : ℝ := 140 -- meters
def train_speed_kmh : ℝ := 50 -- km/h

noncomputable def train_speed_ms : ℝ := train_speed_kmh * (1000 / 3600) -- m/s
noncomputable def total_distance : ℝ := train_length + bridge_length -- meters
noncomputable def passing_time : ℝ := total_distance / train_speed_ms -- seconds

theorem train_pass_bridge_in_36_seconds :
  passing_time = 36 := 
sorry

end NUMINAMATH_GPT_train_pass_bridge_in_36_seconds_l1993_199343


namespace NUMINAMATH_GPT_thief_speed_l1993_199340

theorem thief_speed
  (distance_initial : ℝ := 100 / 1000) -- distance (100 meters converted to kilometers)
  (policeman_speed : ℝ := 10) -- speed of the policeman in km/hr
  (thief_distance : ℝ := 400 / 1000) -- distance thief runs in kilometers (400 meters converted)
  : ∃ V_t : ℝ, V_t = 8 :=
by
  sorry

end NUMINAMATH_GPT_thief_speed_l1993_199340


namespace NUMINAMATH_GPT_sequence_general_formula_l1993_199300

theorem sequence_general_formula (a : ℕ → ℝ) (h₁ : a 1 = 1)
  (h₂ : ∀ n, a (n + 1) = a n / (1 + 2 * a n)) :
  ∀ n, a n = 1 / (2 * n - 1) :=
sorry

end NUMINAMATH_GPT_sequence_general_formula_l1993_199300


namespace NUMINAMATH_GPT_train_length_approx_l1993_199363

noncomputable def speed_kmh_to_ms (v: ℝ) : ℝ :=
  v * (1000 / 3600)

noncomputable def length_of_train (v_kmh: ℝ) (time_s: ℝ) : ℝ :=
  (speed_kmh_to_ms v_kmh) * time_s

theorem train_length_approx (v_kmh: ℝ) (time_s: ℝ) (L: ℝ) 
  (h1: v_kmh = 58) 
  (h2: time_s = 9) 
  (h3: L = length_of_train v_kmh time_s) : 
  |L - 145| < 1 :=
  by sorry

end NUMINAMATH_GPT_train_length_approx_l1993_199363


namespace NUMINAMATH_GPT_simplify_expression_l1993_199328

theorem simplify_expression :
  (3 / 4 : ℚ) * 60 - (8 / 5 : ℚ) * 60 + x = 12 → x = 63 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_simplify_expression_l1993_199328


namespace NUMINAMATH_GPT_darnell_phone_minutes_l1993_199303

theorem darnell_phone_minutes
  (unlimited_cost : ℕ)
  (text_cost : ℕ)
  (call_cost : ℕ)
  (texts_per_dollar : ℕ)
  (minutes_per_dollar : ℕ)
  (total_texts : ℕ)
  (cost_difference : ℕ)
  (alternative_total_cost : ℕ)
  (M : ℕ)
  (text_cost_condition : unlimited_cost - cost_difference = alternative_total_cost)
  (text_formula : M / minutes_per_dollar * call_cost + total_texts / texts_per_dollar * text_cost = alternative_total_cost)
  : M = 60 :=
sorry

end NUMINAMATH_GPT_darnell_phone_minutes_l1993_199303


namespace NUMINAMATH_GPT_evaluate_h_j_l1993_199318

def h (x : ℝ) : ℝ := 3 * x - 4
def j (x : ℝ) : ℝ := x - 2

theorem evaluate_h_j : h (2 + j 3) = 5 := by
  sorry

end NUMINAMATH_GPT_evaluate_h_j_l1993_199318


namespace NUMINAMATH_GPT_smallest_n_for_sqrt_18n_integer_l1993_199359

theorem smallest_n_for_sqrt_18n_integer :
  ∃ n : ℕ, (n > 0) ∧ (∀ m : ℕ, m > 0 → (∃ k : ℕ, k^2 = 18 * m) → n <= m) ∧ (∃ k : ℕ, k^2 = 18 * n) :=
sorry

end NUMINAMATH_GPT_smallest_n_for_sqrt_18n_integer_l1993_199359


namespace NUMINAMATH_GPT_routes_from_A_to_B_l1993_199302

-- Definitions based on conditions given in the problem
variables (A B C D E F : Type)
variables (AB AD AE BC BD CD DE EF : Prop) 

-- Theorem statement
theorem routes_from_A_to_B (route_criteria : AB ∧ AD ∧ AE ∧ BC ∧ BD ∧ CD ∧ DE ∧ EF)
  : ∃ n : ℕ, n = 16 :=
sorry

end NUMINAMATH_GPT_routes_from_A_to_B_l1993_199302


namespace NUMINAMATH_GPT_condition_iff_odd_function_l1993_199386

theorem condition_iff_odd_function (f : ℝ → ℝ) :
  (∀ x, f x + f (-x) = 0) ↔ (∀ x, f (-x) = -f x) :=
by
  sorry

end NUMINAMATH_GPT_condition_iff_odd_function_l1993_199386


namespace NUMINAMATH_GPT_tiles_needed_correct_l1993_199319

noncomputable def tiles_needed (floor_length : ℝ) (floor_width : ℝ) (tile_length_inch : ℝ) (tile_width_inch : ℝ) (border_width : ℝ) : ℝ :=
  let tile_length := tile_length_inch / 12
  let tile_width := tile_width_inch / 12
  let main_length := floor_length - 2 * border_width
  let main_width := floor_width - 2 * border_width
  let main_area := main_length * main_width
  let tile_area := tile_length * tile_width
  main_area / tile_area

theorem tiles_needed_correct :
  tiles_needed 15 20 3 9 1 = 1248 := 
by 
  sorry -- Proof skipped.

end NUMINAMATH_GPT_tiles_needed_correct_l1993_199319


namespace NUMINAMATH_GPT_goldfish_in_each_pond_l1993_199345

variable (x : ℕ)
variable (l1 h1 l2 h2 : ℕ)

-- Conditions
def cond1 : Prop := l1 + h1 = x ∧ l2 + h2 = x
def cond2 : Prop := 4 * l1 = 3 * h1
def cond3 : Prop := 3 * l2 = 5 * h2
def cond4 : Prop := l2 = l1 + 33

theorem goldfish_in_each_pond : cond1 x l1 h1 l2 h2 ∧ cond2 l1 h1 ∧ cond3 l2 h2 ∧ cond4 l1 l2 → 
  x = 168 := 
by 
  sorry

end NUMINAMATH_GPT_goldfish_in_each_pond_l1993_199345


namespace NUMINAMATH_GPT_map_distance_representation_l1993_199341

theorem map_distance_representation
  (cm_to_km_ratio : 15 = 90)
  (km_to_m_ratio : 1000 = 1000) :
  20 * (90 / 15) * 1000 = 120000 := by
  sorry

end NUMINAMATH_GPT_map_distance_representation_l1993_199341


namespace NUMINAMATH_GPT_find_other_number_l1993_199373

def HCF (a b : ℕ) : ℕ := sorry
def LCM (a b : ℕ) : ℕ := sorry

theorem find_other_number (B : ℕ) 
 (h1 : HCF 24 B = 15) 
 (h2 : LCM 24 B = 312) 
 : B = 195 := 
by
  sorry

end NUMINAMATH_GPT_find_other_number_l1993_199373


namespace NUMINAMATH_GPT_rate_per_kg_mangoes_l1993_199394

theorem rate_per_kg_mangoes 
  (weight_grapes : ℕ) 
  (rate_grapes : ℕ) 
  (weight_mangoes : ℕ) 
  (total_paid : ℕ)
  (total_grapes_cost : ℕ)
  (total_mangoes_cost : ℕ)
  (rate_mangoes : ℕ) 
  (h1 : weight_grapes = 14) 
  (h2 : rate_grapes = 54)
  (h3 : weight_mangoes = 10) 
  (h4 : total_paid = 1376) 
  (h5 : total_grapes_cost = weight_grapes * rate_grapes)
  (h6 : total_mangoes_cost = total_paid - total_grapes_cost) 
  (h7 : rate_mangoes = total_mangoes_cost / weight_mangoes):
  rate_mangoes = 62 :=
by
  sorry

end NUMINAMATH_GPT_rate_per_kg_mangoes_l1993_199394


namespace NUMINAMATH_GPT_rose_needs_more_money_l1993_199397

theorem rose_needs_more_money 
    (paintbrush_cost : ℝ)
    (paints_cost : ℝ)
    (easel_cost : ℝ)
    (money_rose_has : ℝ) :
    paintbrush_cost = 2.40 →
    paints_cost = 9.20 →
    easel_cost = 6.50 →
    money_rose_has = 7.10 →
    (paintbrush_cost + paints_cost + easel_cost - money_rose_has) = 11 :=
by
  intros
  sorry

end NUMINAMATH_GPT_rose_needs_more_money_l1993_199397


namespace NUMINAMATH_GPT_smallest_a_such_that_sqrt_50a_is_integer_l1993_199305

theorem smallest_a_such_that_sqrt_50a_is_integer : ∃ a : ℕ, (∀ b : ℕ, (b > 0 ∧ (∃ k : ℕ, 50 * b = k^2)) → (a ≤ b)) ∧ (∃ k : ℕ, 50 * a = k^2) ∧ a = 2 := 
by
  sorry

end NUMINAMATH_GPT_smallest_a_such_that_sqrt_50a_is_integer_l1993_199305


namespace NUMINAMATH_GPT_necessary_and_sufficient_condition_l1993_199349

theorem necessary_and_sufficient_condition (a b : ℝ) (h : a * b ≠ 0) : 
  a - b = 1 ↔ a^3 - b^3 - a * b - a^2 - b^2 = 0 := by
  sorry

end NUMINAMATH_GPT_necessary_and_sufficient_condition_l1993_199349


namespace NUMINAMATH_GPT_problem_one_problem_two_l1993_199371

noncomputable def f (x m : ℝ) : ℝ := x^2 - (m-1) * x + 2 * m

theorem problem_one (m : ℝ) : (∀ x : ℝ, 0 < x → f x m > 0) ↔ (-2 * Real.sqrt 6 + 5 ≤ m ∧ m ≤ 2 * Real.sqrt 6 + 5) :=
by
  sorry

theorem problem_two (m : ℝ) : (∃ x : ℝ, 0 < x ∧ x < 1 ∧ f x m = 0) ↔ (m ∈ Set.Ioo (-2 : ℝ) 0) :=
by
  sorry

end NUMINAMATH_GPT_problem_one_problem_two_l1993_199371


namespace NUMINAMATH_GPT_number_of_boys_is_50_l1993_199374

-- Definitions for conditions:
def total_students : Nat := 100
def boys (x : Nat) : Nat := x
def girls (x : Nat) : Nat := x

-- Theorem statement:
theorem number_of_boys_is_50 (x : Nat) (g : Nat) (h1 : x + g = total_students) (h2 : g = boys x) : boys x = 50 :=
by
  sorry

end NUMINAMATH_GPT_number_of_boys_is_50_l1993_199374


namespace NUMINAMATH_GPT_geometric_series_common_ratio_l1993_199392

theorem geometric_series_common_ratio (a : ℕ → ℚ) (q : ℚ) (h1 : a 1 + a 3 = 10) 
(h2 : a 4 + a 6 = 5 / 4) 
(h_geom : ∀ n : ℕ, a (n + 1) = a n * q) : q = 1 / 2 :=
sorry

end NUMINAMATH_GPT_geometric_series_common_ratio_l1993_199392


namespace NUMINAMATH_GPT_smallest_x_for_multiple_l1993_199315

theorem smallest_x_for_multiple (x : ℕ) (h : x > 0) :
  (450 * x) % 500 = 0 ↔ x = 10 := by
  sorry

end NUMINAMATH_GPT_smallest_x_for_multiple_l1993_199315


namespace NUMINAMATH_GPT_eccentricity_of_ellipse_l1993_199327

theorem eccentricity_of_ellipse (a c : ℝ) (h : 4 * a = 7 * 2 * (a - c)) : 
    c / a = 5 / 7 :=
by {
  sorry
}

end NUMINAMATH_GPT_eccentricity_of_ellipse_l1993_199327


namespace NUMINAMATH_GPT_parallel_lines_l1993_199331

theorem parallel_lines (a : ℝ) :
  (∀ x y : ℝ, a * x + 2 * y + a + 3 = 0) ∧ (∀ x y : ℝ, x + (a + 1) * y + 4 = 0) 
  → a = -2 :=
sorry

end NUMINAMATH_GPT_parallel_lines_l1993_199331


namespace NUMINAMATH_GPT_root_situation_l1993_199366

theorem root_situation (a b : ℝ) : 
  ∃ (m n : ℝ), 
    (x - a) * (x - (a + b)) = 1 → 
    (m < a ∧ a < n) ∨ (n < a ∧ a < m) :=
sorry

end NUMINAMATH_GPT_root_situation_l1993_199366


namespace NUMINAMATH_GPT_largest_power_of_2_divides_n_l1993_199354

def n : ℤ := 17^4 - 13^4

theorem largest_power_of_2_divides_n : ∃ (k : ℕ), 2^4 = k ∧ 2^k ∣ n ∧ ¬ (2^(k + 1) ∣ n) := by
  sorry

end NUMINAMATH_GPT_largest_power_of_2_divides_n_l1993_199354


namespace NUMINAMATH_GPT_asymptote_problem_l1993_199378

-- Definitions for the problem
def r (x : ℝ) : ℝ := -3 * (x + 2) * (x - 1)
def s (x : ℝ) : ℝ := (x + 2) * (x - 4)

-- Assertion to prove
theorem asymptote_problem : r (-1) / s (-1) = 6 / 5 :=
by {
  -- This is where the proof would be carried out
  sorry
}

end NUMINAMATH_GPT_asymptote_problem_l1993_199378


namespace NUMINAMATH_GPT_fraction_relationships_l1993_199395

variable (p r s u : ℚ)

theorem fraction_relationships (h1 : p / r = 8) (h2 : s / r = 5) (h3 : s / u = 1 / 3) :
  u / p = 15 / 8 :=
sorry

end NUMINAMATH_GPT_fraction_relationships_l1993_199395


namespace NUMINAMATH_GPT_calc_expression_l1993_199335

theorem calc_expression :
  15 * (216 / 3 + 36 / 9 + 16 / 25 + 2^2) = 30240 / 25 :=
by
  sorry

end NUMINAMATH_GPT_calc_expression_l1993_199335


namespace NUMINAMATH_GPT_certain_number_is_five_hundred_l1993_199339

theorem certain_number_is_five_hundred (x : ℝ) (h : 0.60 * x = 0.50 * 600) : x = 500 := 
by sorry

end NUMINAMATH_GPT_certain_number_is_five_hundred_l1993_199339


namespace NUMINAMATH_GPT_wire_division_l1993_199351

theorem wire_division (initial_length : ℝ) (num_parts : ℕ) (final_length : ℝ) :
  initial_length = 69.76 ∧ num_parts = 8 ∧
  final_length = (initial_length / num_parts) / num_parts →
  final_length = 1.09 :=
by
  sorry

end NUMINAMATH_GPT_wire_division_l1993_199351


namespace NUMINAMATH_GPT_vertex_on_x_axis_segment_cut_on_x_axis_l1993_199391

-- Define the quadratic function
def quadratic_func (k x : ℝ) : ℝ :=
  (k + 2) * x^2 - 2 * k * x + 3 * k

-- The conditions to prove
theorem vertex_on_x_axis (k : ℝ) :
  (4 * k^2 - 4 * 3 * k * (k + 2) = 0) ↔ (k = 0 ∨ k = -3) :=
sorry

theorem segment_cut_on_x_axis (k : ℝ) :
  ((2 * k / (k + 2))^2 - 12 * k / (k + 2) = 16) ↔ (k = -8/3 ∨ k = -1) :=
sorry

end NUMINAMATH_GPT_vertex_on_x_axis_segment_cut_on_x_axis_l1993_199391


namespace NUMINAMATH_GPT_polyhedron_volume_is_correct_l1993_199385

noncomputable def volume_of_polyhedron : ℕ :=
  let side_length := 12
  let num_squares := 3
  let square_area := side_length * side_length
  let cube_volume := side_length ^ 3
  let polyhedron_volume := cube_volume / 2
  polyhedron_volume

theorem polyhedron_volume_is_correct :
  volume_of_polyhedron = 864 :=
by
  sorry

end NUMINAMATH_GPT_polyhedron_volume_is_correct_l1993_199385

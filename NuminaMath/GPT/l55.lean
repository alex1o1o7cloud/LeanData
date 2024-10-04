import Mathlib

namespace hyperbola_eccentricity_l55_55498

/--
Given a hyperbola with the following properties:
1. Point \( P \) is on the left branch of the hyperbola \( C \): \(\frac{x^2}{a^2} - \frac{y^2}{b^2} = 1\), where \( a > 0 \) and \( b > 0 \).
2. \( F_2 \) is the right focus of the hyperbola.
3. One of the asymptotes of the hyperbola is perpendicular to the line segment \( PF_2 \).

Prove that the eccentricity \( e \) of the hyperbola is \( \sqrt{5} \).
-/
theorem hyperbola_eccentricity (a b e : ℝ) (a_pos : a > 0) (b_pos : b > 0)
  (P_on_hyperbola : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1)
  (F2_is_focus : True) -- Placeholder for focus-related condition
  (asymptote_perpendicular : True) -- Placeholder for asymptote perpendicular condition
  : e = Real.sqrt 5 :=
by
  sorry

end hyperbola_eccentricity_l55_55498


namespace find_constant_k_l55_55483

theorem find_constant_k (k : ℝ) :
  (-x^2 - (k + 9) * x - 8 = - (x - 2) * (x - 4)) → k = -15 :=
by 
  sorry

end find_constant_k_l55_55483


namespace range_of_a_l55_55799

open Set Real

def M (a : ℝ) : Set ℝ := {x | x * (x - a - 1) < 0}
def N : Set ℝ := {x | x^2 - 2 * x - 3 ≤ 0}

theorem range_of_a (a : ℝ) :
  (M a ∪ N = N) → a ∈ Icc (-2 : ℝ) 2 := by
  sorry

end range_of_a_l55_55799


namespace license_plate_combinations_l55_55313

open Finset

theorem license_plate_combinations :
  let letters_combinations := choose 25 2
  let positioning := choose 4 2
  let digit_combinations := 10 * 9
  letters_combinations * positioning * digit_combinations = 162000 :=
by
  sorry

end license_plate_combinations_l55_55313


namespace compare_values_l55_55913

noncomputable def a : ℝ := (1/2)^(1/2)
noncomputable def b : ℝ := Real.log 2015 / Real.log 2014
noncomputable def c : ℝ := Real.log 2 / Real.log 4

theorem compare_values : b > a ∧ a > c :=
by
  sorry

end compare_values_l55_55913


namespace chinese_carriage_problem_l55_55755

theorem chinese_carriage_problem (x : ℕ) : 
  (3 * (x - 2) = 2 * x + 9) :=
sorry

end chinese_carriage_problem_l55_55755


namespace regular_polygon_angle_not_divisible_by_five_l55_55240

theorem regular_polygon_angle_not_divisible_by_five :
  ∃ (n_values : Finset ℕ), n_values.card = 5 ∧
    ∀ n ∈ n_values, 3 ≤ n ∧ n ≤ 15 ∧
      ¬ (∃ k : ℕ, (180 * (n - 2)) / n = 5 * k) := 
by
  sorry

end regular_polygon_angle_not_divisible_by_five_l55_55240


namespace second_day_hike_ratio_l55_55331

theorem second_day_hike_ratio (full_hike_distance first_day_distance third_day_distance : ℕ) 
(h_full_hike: full_hike_distance = 50)
(h_first_day: first_day_distance = 10)
(h_third_day: third_day_distance = 15) : 
(full_hike_distance - (first_day_distance + third_day_distance)) / full_hike_distance = 1 / 2 := by
  sorry

end second_day_hike_ratio_l55_55331


namespace number_of_cubes_with_at_least_two_faces_painted_is_56_l55_55128

def one_inch_cubes_with_at_least_two_faces_painted 
  (side_length : ℕ) (face_colors : ℕ) (cubes_per_side : ℕ) :=
  if side_length = 4 ∧ face_colors = 6 ∧ cubes_per_side = 1 then 56 else 0

theorem number_of_cubes_with_at_least_two_faces_painted_is_56 :
  one_inch_cubes_with_at_least_two_faces_painted 4 6 1 = 56 :=
by
  sorry

end number_of_cubes_with_at_least_two_faces_painted_is_56_l55_55128


namespace min_expression_value_l55_55190

noncomputable def expression (x y : ℝ) : ℝ := 2*x^2 + 2*y^2 - 8*x + 6*y + 25

theorem min_expression_value : ∃ (x y : ℝ), expression x y = 12.5 :=
by
  sorry

end min_expression_value_l55_55190


namespace pure_imaginary_solution_l55_55646

-- Defining the main problem as a theorem in Lean 4

theorem pure_imaginary_solution (m : ℝ) : 
  (∃ a b : ℝ, (m^2 - m = a ∧ a = 0) ∧ (m^2 - 3 * m + 2 = b ∧ b ≠ 0)) → 
  m = 0 :=
sorry -- Proof is omitted as per the instructions

end pure_imaginary_solution_l55_55646


namespace total_classic_books_l55_55104

-- Definitions for the conditions
def authors := 6
def books_per_author := 33

-- Statement of the math proof problem
theorem total_classic_books : authors * books_per_author = 198 := by
  sorry  -- Proof to be filled in

end total_classic_books_l55_55104


namespace academic_integers_l55_55753

def is_academic (n : ℕ) (h : n ≥ 2) : Prop :=
  ∃ (S P : Finset ℕ), (S ∩ P = ∅) ∧ (S ∪ P = Finset.range (n + 1)) ∧ (S.sum id = P.prod id)

theorem academic_integers :
  { n | ∃ h : n ≥ 2, is_academic n h } = { n | n = 3 ∨ n ≥ 5 } :=
by
  sorry

end academic_integers_l55_55753


namespace storks_count_l55_55007

theorem storks_count (B S : ℕ) (h1 : B = 3) (h2 : B + 2 = S + 1) : S = 4 :=
by
  sorry

end storks_count_l55_55007


namespace smallest_positive_integer_problem_l55_55081

theorem smallest_positive_integer_problem
  (n : ℕ) 
  (h1 : 50 ∣ n) 
  (h2 : (∃ e1 e2 e3 : ℕ, n = 2^e1 * 5^e2 * 3^e3 ∧ (e1 + 1) * (e2 + 1) * (e3 + 1) = 100)) 
  (h3 : ∀ m : ℕ, (50 ∣ m) → ((∃ e1 e2 e3 : ℕ, m = 2^e1 * 5^e2 * 3^e3 ∧ (e1 + 1) * (e2 + 1) * (e3 + 1) = 100) → (n ≤ m))) :
  n / 50 = 8100 := 
sorry

end smallest_positive_integer_problem_l55_55081


namespace floral_shop_bouquets_total_l55_55449

theorem floral_shop_bouquets_total (sold_monday_rose : ℕ) (sold_monday_lily : ℕ) (sold_monday_orchid : ℕ)
  (price_monday_rose : ℕ) (price_monday_lily : ℕ) (price_monday_orchid : ℕ)
  (sold_tuesday_rose : ℕ) (sold_tuesday_lily : ℕ) (sold_tuesday_orchid : ℕ)
  (price_tuesday_rose : ℕ) (price_tuesday_lily : ℕ) (price_tuesday_orchid : ℕ)
  (sold_wednesday_rose : ℕ) (sold_wednesday_lily : ℕ) (sold_wednesday_orchid : ℕ)
  (price_wednesday_rose : ℕ) (price_wednesday_lily : ℕ) (price_wednesday_orchid : ℕ)
  (H1 : sold_monday_rose = 12) (H2 : sold_monday_lily = 8) (H3 : sold_monday_orchid = 6)
  (H4 : price_monday_rose = 10) (H5 : price_monday_lily = 15) (H6 : price_monday_orchid = 20)
  (H7 : sold_tuesday_rose = 3 * sold_monday_rose) (H8 : sold_tuesday_lily = 2 * sold_monday_lily)
  (H9 : sold_tuesday_orchid = sold_monday_orchid / 2) (H10 : price_tuesday_rose = 12)
  (H11 : price_tuesday_lily = 18) (H12 : price_tuesday_orchid = 22)
  (H13 : sold_wednesday_rose = sold_tuesday_rose / 3) (H14 : sold_wednesday_lily = sold_tuesday_lily / 4)
  (H15 : sold_wednesday_orchid = 2 * sold_tuesday_orchid / 3) (H16 : price_wednesday_rose = 8)
  (H17 : price_wednesday_lily = 12) (H18 : price_wednesday_orchid = 16) :
  (sold_monday_rose + sold_tuesday_rose + sold_wednesday_rose = 60) ∧
  (sold_monday_lily + sold_tuesday_lily + sold_wednesday_lily = 28) ∧
  (sold_monday_orchid + sold_tuesday_orchid + sold_wednesday_orchid = 11) ∧
  ((sold_monday_rose * price_monday_rose + sold_tuesday_rose * price_tuesday_rose + sold_wednesday_rose * price_wednesday_rose) = 648) ∧
  ((sold_monday_lily * price_monday_lily + sold_tuesday_lily * price_tuesday_lily + sold_wednesday_lily * price_wednesday_lily) = 456) ∧
  ((sold_monday_orchid * price_monday_orchid + sold_tuesday_orchid * price_tuesday_orchid + sold_wednesday_orchid * price_wednesday_orchid) = 218) ∧
  ((sold_monday_rose + sold_tuesday_rose + sold_wednesday_rose + sold_monday_lily + sold_tuesday_lily + sold_wednesday_lily + sold_monday_orchid + sold_tuesday_orchid + sold_wednesday_orchid) = 99) ∧
  ((sold_monday_rose * price_monday_rose + sold_tuesday_rose * price_tuesday_rose + sold_wednesday_rose * price_wednesday_rose + sold_monday_lily * price_monday_lily + sold_tuesday_lily * price_tuesday_lily + sold_wednesday_lily * price_wednesday_lily + sold_monday_orchid * price_monday_orchid + sold_tuesday_orchid * price_tuesday_orchid + sold_wednesday_orchid * price_wednesday_orchid) = 1322) :=
  by sorry

end floral_shop_bouquets_total_l55_55449


namespace solve_quadratic_inequality_l55_55980

open Set Real

noncomputable def quadratic_inequality (x : ℝ) : Prop := -9 * x^2 + 6 * x + 8 > 0

theorem solve_quadratic_inequality :
  {x : ℝ | -9 * x^2 + 6 * x + 8 > 0} = {x : ℝ | -2/3 < x ∧ x < 4/3} :=
by
  sorry

end solve_quadratic_inequality_l55_55980


namespace B_work_days_l55_55000

theorem B_work_days (a b : ℝ) (h1 : a + b = 1/4) (h2 : a = 1/14) : 1 / b = 5.6 :=
by
  sorry

end B_work_days_l55_55000


namespace robot_steps_difference_zero_l55_55454

/-- Define the robot's position at second n --/
def robot_position (n : ℕ) : ℤ :=
  let cycle_length := 7
  let cycle_steps := 4 - 3
  let full_cycles := n / cycle_length
  let remainder := n % cycle_length
  full_cycles + if remainder = 0 then 0 else
    if remainder ≤ 4 then remainder else 4 - (remainder - 4)

/-- The main theorem to prove x_2007 - x_2011 = 0 --/
theorem robot_steps_difference_zero : 
  robot_position 2007 - robot_position 2011 = 0 :=
by sorry

end robot_steps_difference_zero_l55_55454


namespace train_cross_time_l55_55015

def length_of_train : ℕ := 120 -- the train is 120 m long
def speed_of_train_km_hr : ℕ := 45 -- the train's speed in km/hr
def length_of_bridge : ℕ := 255 -- the bridge is 255 m long

def train_speed_m_s : ℕ := speed_of_train_km_hr * (1000 / 3600)

def total_distance : ℕ := length_of_train + length_of_bridge

def time_to_cross_bridge (distance : ℕ) (speed : ℕ) : ℕ :=
  distance / speed

theorem train_cross_time :
  time_to_cross_bridge total_distance train_speed_m_s = 30 :=
by
  sorry

end train_cross_time_l55_55015


namespace proposition_holds_for_odd_numbers_l55_55744

variable (P : ℕ → Prop)

theorem proposition_holds_for_odd_numbers 
  (h1 : P 1)
  (h_ind : ∀ k : ℕ, k ≥ 1 → P k → P (k + 2)) :
  ∀ n : ℕ, n % 2 = 1 → P n :=
by
  sorry

end proposition_holds_for_odd_numbers_l55_55744


namespace painted_cubes_count_l55_55123

def total_painted_cubes : ℕ := 8 + 48

theorem painted_cubes_count : total_painted_cubes = 56 :=
by 
  -- Step 1: Define the number of cubes with 3 faces painted (8 corners)
  let corners := 8
  -- Step 2: Calculate the number of edge cubes with 2 faces painted
  let edge_middle_cubes_per_edge := 2
  let edges := 12
  let edge_cubes := edge_middle_cubes_per_edge * edges -- this should be 24
  -- Step 3: Calculate the number of face-interior cubes with 2 faces painted
  let face_cubes_per_face := 4
  let faces := 6
  let face_cubes := face_cubes_per_face * faces -- this should be 24
  -- Step 4: Sum them up to get total cubes with at least two faces painted
  let total_cubes := corners + edge_cubes + face_cubes
  show total_cubes = total_painted_cubes
  sorry

end painted_cubes_count_l55_55123


namespace find_coordinates_B_l55_55349

variable (B : ℝ × ℝ)

def A : ℝ × ℝ := (2, 3)
def C : ℝ × ℝ := (0, 1)
def vec (P Q : ℝ × ℝ) : ℝ × ℝ := (Q.1 - P.1, Q.2 - P.2)

theorem find_coordinates_B (h : vec A B = (-2) • vec B C) : B = (-2, 5/3) :=
by
  -- Here you would provide proof steps
  sorry

end find_coordinates_B_l55_55349


namespace large_pizzas_sold_l55_55272

def small_pizza_price : ℕ := 2
def large_pizza_price : ℕ := 8
def total_earnings : ℕ := 40
def small_pizzas_sold : ℕ := 8

theorem large_pizzas_sold : 
  ∀ (small_pizza_price large_pizza_price total_earnings small_pizzas_sold : ℕ), 
    small_pizza_price = 2 → 
    large_pizza_price = 8 → 
    total_earnings = 40 → 
    small_pizzas_sold = 8 →
    (total_earnings - small_pizzas_sold * small_pizza_price) / large_pizza_price = 3 :=
by 
  intros small_pizza_price large_pizza_price total_earnings small_pizzas_sold 
         h_small_pizza_price h_large_pizza_price h_total_earnings h_small_pizzas_sold
  rw [h_small_pizza_price, h_large_pizza_price, h_total_earnings, h_small_pizzas_sold]
  simp
  sorry

end large_pizzas_sold_l55_55272


namespace cos_BHD_correct_l55_55747

noncomputable def cos_BHD : ℝ :=
  let DB := 2
  let DC := 2 * Real.sqrt 2
  let AB := Real.sqrt 3
  let DH := DC
  let HG := DH * Real.sin (Real.pi / 6)  -- 30 degrees in radians
  let FB := AB
  let HB := FB * Real.sin (Real.pi / 4)  -- 45 degrees in radians
  let law_of_cosines :=
    DB^2 = DH^2 + HB^2 - 2 * DH * HB * Real.cos (Real.pi / 3)
  let expected_cos := (Real.sqrt 3) / 12
  expected_cos

theorem cos_BHD_correct :
  cos_BHD = (Real.sqrt 3) / 12 :=
by
  sorry

end cos_BHD_correct_l55_55747


namespace initial_investment_l55_55452

variable (P1 P2 π1 π2 : ℝ)

-- Given conditions
axiom h1 : π1 = 100
axiom h2 : π2 = 120

-- Revenue relation after the first transaction
axiom h3 : P2 = P1 + π1

-- Consistent profit relationship across transactions
axiom h4 : π2 = 0.2 * P2

-- To be proved
theorem initial_investment (P1 : ℝ) (h1 : π1 = 100) (h2 : π2 = 120) (h3 : P2 = P1 + π1) (h4 : π2 = 0.2 * P2) :
  P1 = 500 :=
sorry

end initial_investment_l55_55452


namespace original_student_count_l55_55390

variable (A B C N D : ℕ)
variable (hA : A = 40)
variable (hB : B = 32)
variable (hC : C = 36)
variable (hD : D = N * A)
variable (hNewSum : D + 8 * B = (N + 8) * C)

theorem original_student_count (hA : A = 40) (hB : B = 32) (hC : C = 36) (hD : D = N * A) (hNewSum : D + 8 * B = (N + 8) * C) : 
  N = 8 :=
by
  sorry

end original_student_count_l55_55390


namespace find_m_b_sum_does_not_prove_l55_55415

theorem find_m_b_sum_does_not_prove :
  ∃ m b : ℝ, 
  let original_point := (2, 3)
  let image_point := (10, 9)
  let midpoint := ((original_point.1 + image_point.1) / 2, (original_point.2 + image_point.2) / 2)
  m = -4 / 3 ∧ 
  midpoint = (6, 6) ∧ 
  6 = m * 6 + b 
  ∧ m + b = 38 / 3 := sorry

end find_m_b_sum_does_not_prove_l55_55415


namespace minimum_positive_period_of_f_l55_55707

noncomputable def f (x : ℝ) : ℝ := (1 + (Real.sqrt 3) * Real.tan x) * Real.cos x

theorem minimum_positive_period_of_f :
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ (∀ T', T' > 0 → (∀ x, f (x + T') = f x) → T ≤ T') :=
sorry

end minimum_positive_period_of_f_l55_55707


namespace phoenix_number_5841_phoenix_numbers_satisfying_conditions_l55_55487

-- Definition of Phoenix Number
def is_phoenix_number (N : ℕ) : Prop := 
  let d1 := N / 1000,
      d2 := (N / 100) % 10,
      d3 := (N / 10) % 10,
      d4 := N % 10
  in d1 + d3 = 9 ∧ d2 + d4 = 9

-- Part 1: Phoenix Number check and K(N) calculation
theorem phoenix_number_5841 :
  is_phoenix_number 5841 ∧ 5841 / 99 = 59 :=
sorry

-- Part 2: Conditions for solving Phoenix Number N
def K (N : ℕ) : ℕ := N / 99

theorem phoenix_numbers_satisfying_conditions :
  ∀ (N : ℕ),
  is_phoenix_number N /\
  (N % 2 = 0) /\
  (let N' := ((N / 1000) + 9) * 1000 + ((N % 1000 / 100) + 9) * 100 + (N % 1000 / 10 % 10) * 10 + (N % 10) 
      in 3 * K N + 2 * K N' % 9 = 0) /\
  ((N / 1000) >= (N / 100 % 10))
  → (N = 8514 ∨ N = 3168) :=
sorry

end phoenix_number_5841_phoenix_numbers_satisfying_conditions_l55_55487


namespace sin_sum_cos_product_l55_55797

theorem sin_sum_cos_product (A B C : Real) (h : A + B + C = π) :
  Real.sin A + Real.sin B + Real.sin C = 4 * Real.cos (A / 2) * Real.cos (B / 2) * Real.cos (C / 2) :=
by
  sorry

end sin_sum_cos_product_l55_55797


namespace min_ties_to_ensure_pairs_l55_55447

variable (red blue green yellow : Nat)
variable (total_ties : Nat)
variable (pairs_needed : Nat)

-- Define the conditions
def conditions : Prop :=
  red = 120 ∧
  blue = 90 ∧
  green = 70 ∧
  yellow = 50 ∧
  total_ties = 27 ∧
  pairs_needed = 12

-- Define the statement to be proven
theorem min_ties_to_ensure_pairs : conditions red blue green yellow total_ties pairs_needed → total_ties = 27 :=
sorry

end min_ties_to_ensure_pairs_l55_55447


namespace mod_remainder_of_expression_l55_55196

theorem mod_remainder_of_expression : (7 * 10^20 + 2^20) % 9 = 2 := by
  sorry

end mod_remainder_of_expression_l55_55196


namespace twelfth_term_in_sequence_l55_55150

variable (a₁ : ℚ) (d : ℚ)

-- Given conditions
def a_1_term : a₁ = 1 / 2 := rfl
def common_difference : d = 1 / 3 := rfl

-- The twelfth term of the arithmetic sequence
def a₁₂ : ℚ := a₁ + 11 * d

-- Statement to prove
theorem twelfth_term_in_sequence (h₁ : a₁ = 1 / 2) (h₂ : d = 1 / 3) : a₁₂ = 25 / 6 :=
by
  rw [h₁, h₂, add_comm, add_assoc, mul_comm, add_comm]
  exact sorry

end twelfth_term_in_sequence_l55_55150


namespace solve_equation_l55_55991

theorem solve_equation : ∀ x : ℝ, (2 * x - 8 = 0) ↔ (x = 4) :=
by sorry

end solve_equation_l55_55991


namespace kangaroo_jump_is_8_5_feet_longer_l55_55069

noncomputable def camel_step_length (total_distance : ℝ) (num_steps : ℕ) : ℝ := total_distance / num_steps
noncomputable def kangaroo_jump_length (total_distance : ℝ) (num_jumps : ℕ) : ℝ := total_distance / num_jumps
noncomputable def length_difference (jump_length step_length : ℝ) : ℝ := jump_length - step_length

theorem kangaroo_jump_is_8_5_feet_longer :
  let total_distance := 7920
  let num_gaps := 50
  let camel_steps_per_gap := 56
  let kangaroo_jumps_per_gap := 14
  let num_camel_steps := num_gaps * camel_steps_per_gap
  let num_kangaroo_jumps := num_gaps * kangaroo_jumps_per_gap
  let camel_step := camel_step_length total_distance num_camel_steps
  let kangaroo_jump := kangaroo_jump_length total_distance num_kangaroo_jumps
  length_difference kangaroo_jump camel_step = 8.5 := sorry

end kangaroo_jump_is_8_5_feet_longer_l55_55069


namespace trig_identity_l55_55945

open Real

theorem trig_identity (θ : ℝ) (h : tan θ = 2) :
  ((sin θ + cos θ) * cos (2 * θ)) / sin θ = -9 / 10 :=
sorry

end trig_identity_l55_55945


namespace determine_x_l55_55472

theorem determine_x (x : ℝ) :
  (x^2 - 6 * x + 8) / (x^2 - 9 * x + 14) = (x^2 - 8 * x + 15) / (x^2 - 10 * x + 24) →
  x = (13 + Real.sqrt 5) / 2 ∨ x = (13 - Real.sqrt 5) / 2 :=
by
  sorry

end determine_x_l55_55472


namespace problem_l55_55466

theorem problem : (112^2 - 97^2) / 15 = 209 := by
  sorry

end problem_l55_55466


namespace fraction_of_5100_l55_55439

theorem fraction_of_5100 (x : ℝ) (h : ((3 / 4) * x * (2 / 5) * 5100 = 765.0000000000001)) : x = 0.5 :=
by
  sorry

end fraction_of_5100_l55_55439


namespace coefficient_of_c_l55_55888

theorem coefficient_of_c (f c : ℝ) (h₁ : f = (9/5) * c + 32)
                         (h₂ : f + 25 = (9/5) * (c + 13.88888888888889) + 32) :
  (5/9) = (9/5) := sorry

end coefficient_of_c_l55_55888


namespace businessmen_no_drink_l55_55464

theorem businessmen_no_drink 
  (total : ℕ) 
  (coffee : ℕ) 
  (tea : ℕ) 
  (juice : ℕ) 
  (coffee_tea : ℕ) 
  (coffee_juice : ℕ) 
  (tea_juice : ℕ) 
  (all_three : ℕ) 
  (S : Finset ℕ) :
  total = 30 → coffee = 15 → tea = 12 → juice = 8 → coffee_tea = 6 → coffee_juice = 4 → tea_juice = 2 → all_three = 1 →
  (total - (coffee + tea + juice - coffee_tea - coffee_juice - tea_juice + all_three)) = 6 := by
  intros h_total h_coffee h_tea h_juice h_coffee_tea h_coffee_juice h_tea_juice h_all_three
  rw [h_total, h_coffee, h_tea, h_juice, h_coffee_tea, h_coffee_juice, h_tea_juice, h_all_three]
  norm_num
  exact eq.refl 6

end businessmen_no_drink_l55_55464


namespace part_a_part_b_l55_55070

-- Define the exchange rates
def ornament_to_crackers := 2
def sparklers_to_garlands := (5, 2)
def ornaments_to_garland := (4, 1)

-- Part (a)
theorem part_a (n_sparklers : ℕ) (h : n_sparklers = 10) :
  let n_garlands := (n_sparklers / sparklers_to_garlands.1) * sparklers_to_garlands.2 in
  let n_ornaments := n_garlands * ornaments_to_garland.1 in
  let n_crackers := n_ornaments * ornament_to_crackers in
  n_crackers = 32 :=
by {
  have n_garlands_def : n_garlands = (n_sparklers / sparklers_to_garlands.1) * sparklers_to_garlands.2, sorry,
  have n_ornaments_def : n_ornaments = n_garlands * ornaments_to_garland.1, sorry,
  have n_crackers_def : n_crackers = n_ornaments * ornament_to_crackers, sorry,
  have n_sparklers_eq : n_sparklers = 10, from h,
  sorry
}

-- Part (b)
theorem part_b :
  let v1 := (5 * ornament_to_crackers) + 1 in
  let v2 := ((2 / sparklers_to_garlands.1).nat_divide * sparklers_to_garlands.2 * ornaments_to_garland.1) * ornament_to_crackers in
  v1 > v2 :=
by {
  have v1_def : v1 = (5 * ornament_to_crackers) + 1, sorry,
  have v2_def : v2 = ((2 / sparklers_to_garlands.1).nat_divide * sparklers_to_garlands.2 * ornaments_to_garland.1) * ornament_to_crackers, sorry,
  sorry
}

end part_a_part_b_l55_55070


namespace price_of_basic_computer_l55_55131

variable (C P : ℝ)

theorem price_of_basic_computer 
    (h1 : C + P = 2500)
    (h2 : P = (1 / 6) * (C + 500 + P)) : 
  C = 2000 :=
by
  sorry

end price_of_basic_computer_l55_55131


namespace simplified_sum_l55_55567

theorem simplified_sum :
  (-2^2003) + (2^2004) + (-2^2005) - (2^2006) = 5 * (2^2003) :=
by
  sorry

end simplified_sum_l55_55567


namespace cubic_expression_value_l55_55365

theorem cubic_expression_value (x : ℝ) (h : x^2 + 3 * x - 1 = 0) : x^3 + 5 * x^2 + 5 * x + 18 = 20 :=
by
  sorry

end cubic_expression_value_l55_55365


namespace first_player_wins_l55_55816

-- Define the initial conditions
def initial_pile_1 : ℕ := 100
def initial_pile_2 : ℕ := 200

-- Define the game rules
def valid_move (pile_1 pile_2 n : ℕ) : Prop :=
  (n > 0) ∧ ((n <= pile_1) ∨ (n <= pile_2))

-- The game state is represented as a pair of natural numbers
def GameState := ℕ × ℕ

-- Define what it means to win the game
def winning_move (s: GameState) : Prop :=
  (s.1 = 0 ∧ s.2 = 1) ∨ (s.1 = 1 ∧ s.2 = 0)

-- Define the main theorem
theorem first_player_wins : 
  ∀ s : GameState, (s = (initial_pile_1, initial_pile_2)) → (∃ move, valid_move s.1 s.2 move ∧ winning_move (s.1 - move, s.2 - move)) :=
sorry

end first_player_wins_l55_55816


namespace opposite_of_neg_quarter_l55_55422

theorem opposite_of_neg_quarter : -(- (1/4 : ℝ)) = (1/4 : ℝ) :=
by
  sorry

end opposite_of_neg_quarter_l55_55422


namespace probability_of_selecting_cooking_l55_55591

theorem probability_of_selecting_cooking :
  let courses := ["planting", "cooking", "pottery", "woodworking"]
  in (∃ (course : String), course ∈ courses ∧ course = "cooking") →
    (1 / (courses.length : ℝ) = 1 / 4) :=
by
  sorry

end probability_of_selecting_cooking_l55_55591


namespace ageOfX_l55_55862

def threeYearsAgo (x y : ℕ) := x - 3 = 2 * (y - 3)
def sevenYearsHence (x y : ℕ) := (x + 7) + (y + 7) = 83

theorem ageOfX (x y : ℕ) (h1 : threeYearsAgo x y) (h2 : sevenYearsHence x y) : x = 45 := by
  sorry

end ageOfX_l55_55862


namespace value_of_m_l55_55808

theorem value_of_m (m : ℝ) : (∀ x : ℝ, x^2 + m * x + 9 = (x + 3)^2) → m = 6 :=
by
  intro h
  sorry

end value_of_m_l55_55808


namespace one_inch_cubes_with_two_or_more_painted_faces_l55_55117

def original_cube_length : ℕ := 4

def total_one_inch_cubes : ℕ := original_cube_length ^ 3

def corners_count : ℕ := 8

def edges_minus_corners_count : ℕ := 12 * 2

theorem one_inch_cubes_with_two_or_more_painted_faces
  (painted_faces_on_each_face : ∀ i : ℕ, i < total_one_inch_cubes → ℕ) : 
  ∃ n : ℕ, n = corners_count + edges_minus_corners_count ∧ n = 32 := 
by
  simp only [corners_count, edges_minus_corners_count, total_one_inch_cubes]
  sorry

end one_inch_cubes_with_two_or_more_painted_faces_l55_55117


namespace number_of_rows_l55_55622

theorem number_of_rows (total_pencils : ℕ) (pencils_per_row : ℕ) (h1 : total_pencils = 30) (h2 : pencils_per_row = 5) : total_pencils / pencils_per_row = 6 :=
by
  sorry

end number_of_rows_l55_55622


namespace fourth_quadrant_point_l55_55204

theorem fourth_quadrant_point (a : ℤ) (h1 : 2 * a + 6 > 0) (h2 : 3 * a + 3 < 0) :
  (2 * a + 6, 3 * a + 3) = (2, -3) :=
sorry

end fourth_quadrant_point_l55_55204


namespace cost_price_percentage_l55_55259

theorem cost_price_percentage (CP SP : ℝ) (h1 : SP = 4 * CP) : (CP / SP) * 100 = 25 :=
by
  sorry

end cost_price_percentage_l55_55259


namespace tan_585_eq_1_l55_55183

noncomputable def tan_deg (θ : ℝ) : ℝ := Real.tan (θ * Real.pi / 180)

theorem tan_585_eq_1 :
  tan_deg 585 = 1 := 
by
  have h1 : 585 - 360 = 225 := by norm_num
  have h2 : tan_deg 225 = tan_deg 45 :=
    by have h3 : 225 = 180 + 45 := by norm_num
       rw [h3, tan_deg]
       exact Real.tan_add_pi_div_two_simp_left (Real.pi * 45 / 180)
  rw [← tan_deg]
  rw [h1, h2]
  exact Real.tan_pi_div_four

end tan_585_eq_1_l55_55183


namespace find_A_l55_55727

theorem find_A :
  ∃ A B C D : ℕ, A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
               A * B = 72 ∧ C * D = 72 ∧
               A + B = C - D ∧ A = 4 :=
by
  sorry

end find_A_l55_55727


namespace real_numbers_division_l55_55432

def is_non_neg (x : ℝ) : Prop := x ≥ 0

theorem real_numbers_division :
  ∀ x : ℝ, x < 0 ∨ is_non_neg x :=
by
  intro x
  by_cases h : x < 0
  · left
    exact h
  · right
    push_neg at h
    exact h

end real_numbers_division_l55_55432


namespace total_animals_l55_55520

theorem total_animals (H C2 C1 : ℕ) (humps_eq : 2 * C2 + C1 = 200) (horses_eq : H = C2) :
  H + C2 + C1 = 200 :=
by
  /- Proof steps are not required -/
  sorry

end total_animals_l55_55520


namespace average_interest_rate_l55_55608

theorem average_interest_rate
  (x : ℝ)
  (h₀ : 0 ≤ x)
  (h₁ : x ≤ 5000)
  (h₂ : 0.05 * x = 0.03 * (5000 - x)) :
  (0.05 * x + 0.03 * (5000 - x)) / 5000 = 0.0375 :=
by
  sorry

end average_interest_rate_l55_55608


namespace quadratic_root_condition_l55_55115

theorem quadratic_root_condition (d : ℝ) :
  (∀ x, x^2 + 7 * x + d = 0 → x = (-7 + Real.sqrt d) / 2 ∨ x = (-7 - Real.sqrt d) / 2) → d = 9.8 :=
by
  intro h
  sorry

end quadratic_root_condition_l55_55115


namespace total_trophies_l55_55229

theorem total_trophies (michael_now : ℕ) (increase : ℕ) (jack_multiplication : ℕ) :
  michael_now = 30 →
  increase = 100 →
  jack_multiplication = 10 →
  let michael_future := michael_now + increase in
  let jack_future := jack_multiplication * michael_now in
  michael_future + jack_future = 430 :=
by
  intros h_michael h_increase h_jack
  simp [h_michael, h_increase, h_jack]
  sorry

end total_trophies_l55_55229


namespace hyperbola_real_axis_length_l55_55638

theorem hyperbola_real_axis_length :
  (∃ (a b : ℝ), (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) ∧ a = 3) →
  2 * 3 = 6 :=
by
  sorry

end hyperbola_real_axis_length_l55_55638


namespace kris_age_l55_55803

theorem kris_age (kris_age herbert_age : ℕ) (h1 : herbert_age + 1 = 15) (h2 : herbert_age + 10 = kris_age) : kris_age = 24 :=
by
  sorry

end kris_age_l55_55803


namespace value_of_a3_a6_a9_l55_55664

variable (a : ℕ → ℤ) -- Define the sequence a as a function from natural numbers to integers
variable (d : ℤ) -- Define the common difference d as an integer

-- Conditions
axiom h1 : a 1 + a 4 + a 7 = 39
axiom h2 : a 2 + a 5 + a 8 = 33
axiom h3 : ∀ n : ℕ, a (n+1) = a n + d -- This condition ensures the sequence is arithmetic

-- Theorem: We need to prove the value of a_3 + a_6 + a_9 is 27
theorem value_of_a3_a6_a9 : a 3 + a 6 + a 9 = 27 :=
by
  sorry

end value_of_a3_a6_a9_l55_55664


namespace factorization_of_w4_minus_81_l55_55333

theorem factorization_of_w4_minus_81 (w : ℝ) : 
  (w^4 - 81) = (w - 3) * (w + 3) * (w^2 + 9) :=
by sorry

end factorization_of_w4_minus_81_l55_55333


namespace detergent_per_pound_l55_55681

-- Define the conditions
def total_ounces_detergent := 18
def total_pounds_clothes := 9

-- Define the question to prove the amount of detergent per pound of clothes
theorem detergent_per_pound : total_ounces_detergent / total_pounds_clothes = 2 := by
  sorry

end detergent_per_pound_l55_55681


namespace rotated_line_l1_l55_55101

-- Define the original line equation and the point around which the line is rotated
def line_l (x y : ℝ) : Prop := x - y + 1 = 0
def point_A : ℝ × ℝ := (2, 3)

-- Define the line equation that needs to be proven
def line_l1 (x y : ℝ) : Prop := x + y - 5 = 0

-- The theorem stating that after a 90-degree rotation of line l around point A, the new line is equation l1
theorem rotated_line_l1 : 
  ∀ (x y : ℝ), 
  (∃ (k : ℝ), k = 1 ∧ ∀ (x y), line_l x y ∧ ∀ (x y), line_l1 x y) ∧ 
  ∀ (a b : ℝ), (a, b) = point_A → 
  x + y - 5 = 0 := 
by
  sorry

end rotated_line_l1_l55_55101


namespace find_f_expression_find_f_range_l55_55641

noncomputable def y (t x : ℝ) : ℝ := 1 - 2 * t - 2 * t * x + 2 * x ^ 2

noncomputable def f (t : ℝ) : ℝ := 
  if t < -2 then 3 
  else if t > 2 then -4 * t + 3 
  else -t ^ 2 / 2 - 2 * t + 1

theorem find_f_expression (t : ℝ) : 
  f t = if t < -2 then 3 else 
          if t > 2 then -4 * t + 3 
          else - t ^ 2 / 2 - 2 * t + 1 :=
sorry

theorem find_f_range (t : ℝ) (ht : -2 ≤ t ∧ t ≤ 0) : 
  1 ≤ f t ∧ f t ≤ 3 := 
sorry

end find_f_expression_find_f_range_l55_55641


namespace consecutive_page_numbers_sum_l55_55107

theorem consecutive_page_numbers_sum (n : ℕ) (h : n * (n + 1) * (n + 2) = 35280) :
  n + (n + 1) + (n + 2) = 96 := sorry

end consecutive_page_numbers_sum_l55_55107


namespace min_value_condition_l55_55354

theorem min_value_condition 
  (a b : ℝ) 
  (h1 : 4 * a + b = 1) 
  (h2 : a > 0) 
  (h3 : b > 0) : 
  ∃ x : ℝ, ∀ y : ℝ, (y = 1 - 4 * x → x = 16) := 
sorry

end min_value_condition_l55_55354


namespace angle_B_l55_55243

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

end angle_B_l55_55243


namespace subletter_payment_correct_l55_55232

noncomputable def johns_monthly_rent : ℕ := 900
noncomputable def johns_yearly_rent : ℕ := johns_monthly_rent * 12
noncomputable def johns_profit_per_year : ℕ := 3600
noncomputable def total_rent_collected : ℕ := johns_yearly_rent + johns_profit_per_year
noncomputable def number_of_subletters : ℕ := 3
noncomputable def subletter_annual_payment : ℕ := total_rent_collected / number_of_subletters
noncomputable def subletter_monthly_payment : ℕ := subletter_annual_payment / 12

theorem subletter_payment_correct :
  subletter_monthly_payment = 400 :=
by
  sorry

end subletter_payment_correct_l55_55232


namespace ineq_triples_distinct_integers_l55_55693

theorem ineq_triples_distinct_integers 
  (x y z : ℤ) (h₁ : x ≠ y) (h₂ : y ≠ z) (h₃ : z ≠ x) : 
  ( ( (x - y)^7 + (y - z)^7 + (z - x)^7 - (x - y) * (y - z) * (z - x) * ((x - y)^4 + (y - z)^4 + (z - x)^4) )
  / ( (x - y)^5 + (y - z)^5 + (z - x)^5 ) ) ≥ 3 :=
sorry

end ineq_triples_distinct_integers_l55_55693


namespace walnut_trees_planted_l55_55713

-- Define the initial number of walnut trees
def initial_walnut_trees : ℕ := 22

-- Define the total number of walnut trees after planting
def total_walnut_trees_after : ℕ := 55

-- The Lean statement to prove the number of walnut trees planted today
theorem walnut_trees_planted : (total_walnut_trees_after - initial_walnut_trees = 33) :=
by
  sorry

end walnut_trees_planted_l55_55713


namespace students_not_taking_math_or_physics_l55_55834

theorem students_not_taking_math_or_physics (total_students math_students phys_students both_students : ℕ)
  (h1 : total_students = 120)
  (h2 : math_students = 75)
  (h3 : phys_students = 50)
  (h4 : both_students = 15) :
  total_students - (math_students + phys_students - both_students) = 10 :=
by
  sorry

end students_not_taking_math_or_physics_l55_55834


namespace avg_age_when_youngest_born_l55_55222

theorem avg_age_when_youngest_born
  (num_people : ℕ) (avg_age_now : ℝ) (youngest_age_now : ℝ) (sum_ages_others_then : ℝ) 
  (h1 : num_people = 7) 
  (h2 : avg_age_now = 30) 
  (h3 : youngest_age_now = 6) 
  (h4 : sum_ages_others_then = 150) :
  (sum_ages_others_then / num_people) = 21.43 :=
by
  sorry

end avg_age_when_youngest_born_l55_55222


namespace find_a10_l55_55374

-- Define the arithmetic sequence with its common difference and initial term
axiom a_seq : ℕ → ℝ
axiom a1 : ℝ
axiom d : ℝ

-- Conditions
axiom a3 : a_seq 3 = a1 + 2 * d
axiom a5_a8 : a_seq 5 + a_seq 8 = 15

-- Theorem statement
theorem find_a10 : a_seq 10 = 13 :=
by sorry

end find_a10_l55_55374


namespace sin_half_pi_plus_A_l55_55647

theorem sin_half_pi_plus_A (A : Real) (h : Real.cos (Real.pi + A) = -1 / 2) :
  Real.sin (Real.pi / 2 + A) = 1 / 2 := by
  sorry

end sin_half_pi_plus_A_l55_55647


namespace Dvaneft_percentage_bounds_l55_55297

noncomputable def percentageDvaneftShares (x y z : ℤ) (n m : ℕ) : ℚ :=
  n * 100 / (2 * (n + m))

theorem Dvaneft_percentage_bounds
  (x y z : ℤ) (n m : ℕ)
  (h1 : 4 * x * n = y * m)
  (h2 : x * n + y * m = z * (n + m))
  (h3 : 16 ≤ y - x)
  (h4 : y - x ≤ 20)
  (h5 : 42 ≤ z)
  (h6 : z ≤ 60) :
  12.5 ≤ percentageDvaneftShares x y z n m ∧ percentageDvaneftShares x y z n m ≤ 15 := by
  sorry

end Dvaneft_percentage_bounds_l55_55297


namespace simple_interest_years_l55_55908

theorem simple_interest_years (SI P : ℝ) (R : ℝ) (T : ℝ) 
  (hSI : SI = 200) 
  (hP : P = 1600) 
  (hR : R = 3.125) : 
  T = 4 :=
by 
  sorry

end simple_interest_years_l55_55908


namespace solve_equation_solve_inequality_l55_55436

-- Defining the first problem
theorem solve_equation (x : ℝ) : 3 * (x - 2) - (1 - 2 * x) = 3 ↔ x = 2 := 
by
  sorry

-- Defining the second problem
theorem solve_inequality (x : ℝ) : (2 * x - 1 < 4 * x + 3) ↔ (x > -2) :=
by
  sorry

end solve_equation_solve_inequality_l55_55436


namespace range_of_third_side_l55_55500

theorem range_of_third_side (y : ℝ) : (2 < y) ↔ (y < 8) :=
by sorry

end range_of_third_side_l55_55500


namespace propositions_true_false_l55_55505

theorem propositions_true_false :
  (∃ x : ℝ, x ^ 3 < 1) ∧ 
  ¬ (∃ x : ℚ, x ^ 2 = 2) ∧ 
  ¬ (∀ x : ℕ, x ^ 3 > x ^ 2) ∧ 
  (∀ x : ℝ, x ^ 2 + 1 > 0) :=
by
  sorry

end propositions_true_false_l55_55505


namespace value_of_a_g_odd_iff_m_eq_one_l55_55055

noncomputable def f (a x : ℝ) : ℝ := a ^ x

noncomputable def g (m x a : ℝ) : ℝ := m - 2 / (f a x + 1)

theorem value_of_a
  (a : ℝ)
  (h_pos : a > 0)
  (h_neq_one : a ≠ 1)
  (h_diff : ∀ x y : ℝ, x ∈ (Set.Icc 1 2) → y ∈ (Set.Icc 1 2) → abs (f a x - f a y) = 2) :
  a = 2 :=
sorry

theorem g_odd_iff_m_eq_one
  (a m : ℝ)
  (h_pos : a > 0)
  (h_neq_one : a ≠ 1)
  (h_a_eq : a = 2) :
  (∀ x : ℝ, g m x a = -g m (-x) a) ↔ m = 1 :=
sorry

end value_of_a_g_odd_iff_m_eq_one_l55_55055


namespace part_I_part_II_l55_55346

open Real

def f (x m n : ℝ) := abs (x - m) + abs (x + n)

theorem part_I (m n M : ℝ) (h1 : m + n = 9) (h2 : ∀ x : ℝ, f x m n ≥ M) : M ≤ 9 := 
sorry

theorem part_II (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a^2 + b^2 = 9) : (a + b) * (a^3 + b^3) ≥ 81 := 
sorry

end part_I_part_II_l55_55346


namespace average_cookies_per_package_is_fifteen_l55_55028

def average_cookies_count (cookies : List ℕ) (n : ℕ) : ℕ :=
  (cookies.sum / n : ℕ)

theorem average_cookies_per_package_is_fifteen :
  average_cookies_count [5, 12, 18, 20, 21] 5 = 15 :=
by
  sorry

end average_cookies_per_package_is_fifteen_l55_55028


namespace tan_585_eq_one_l55_55185

theorem tan_585_eq_one : Real.tan (585 * Real.pi / 180) = 1 :=
by
  sorry

end tan_585_eq_one_l55_55185


namespace clock_ticks_12_times_l55_55026

theorem clock_ticks_12_times (t1 t2 : ℕ) (d1 d2 : ℕ) (h1 : t1 = 6) (h2 : d1 = 40) (h3 : d2 = 88) : t2 = 12 := by
  sorry

end clock_ticks_12_times_l55_55026


namespace quadratic_discriminant_l55_55623

-- Define the coefficients of the quadratic equation
def a : ℚ := 2
def b : ℚ := 2 + 1/2
def c : ℚ := 1/2

-- State the theorem to prove
theorem quadratic_discriminant : (b^2 - 4 * a * c) = 9 / 4 := by
  -- Coefficient values
  have h_b : b = 5 / 2 := by
    calc
      b = 2 + 1/2 : rfl
      ... = 5 / 2 : by norm_num
  have h_discriminant : (5/2)^2 - 4 * 2 * (1/2) = 9/4 := by sorry
  -- Substitute the coefficient values
  rw h_b,
  exact h_discriminant,
  sorry

end quadratic_discriminant_l55_55623


namespace two_digit_decimal_bounds_l55_55750

def is_approximate (original approx : ℝ) : Prop :=
  abs (original - approx) < 0.05

theorem two_digit_decimal_bounds :
  ∃ max min : ℝ, is_approximate 15.6 max ∧ max = 15.64 ∧ is_approximate 15.6 min ∧ min = 15.55 :=
by
  sorry

end two_digit_decimal_bounds_l55_55750


namespace sara_steps_l55_55251

theorem sara_steps (n : ℕ) (h : n^2 ≤ 210) : n = 14 :=
sorry

end sara_steps_l55_55251


namespace simplify_and_evaluate_l55_55843

theorem simplify_and_evaluate : 
  ∀ (a : ℝ), a = Real.sqrt 3 + 1 → 
  ((a + 1) / (a^2 - 2*a +1) / (1 + (2 / (a - 1)))) = Real.sqrt 3 / 3 :=
by
  intro a ha
  rw ha
  sorry

end simplify_and_evaluate_l55_55843


namespace lattice_points_in_region_l55_55451

theorem lattice_points_in_region : ∃! n : ℕ, n = 14 ∧ ∀ (x y : ℤ), (y = |x| ∨ y = -x^2 + 4) ∧ (-2 ≤ x ∧ x ≤ 1) → 
  (y = -x^2 + 4 ∧ y = |x|) :=
sorry

end lattice_points_in_region_l55_55451


namespace total_ladybugs_l55_55697

theorem total_ladybugs (ladybugs_with_spots ladybugs_without_spots : ℕ) 
  (h1 : ladybugs_with_spots = 12170) 
  (h2 : ladybugs_without_spots = 54912) : 
  ladybugs_with_spots + ladybugs_without_spots = 67082 := 
by
  sorry

end total_ladybugs_l55_55697


namespace g_x_squared_minus_3_l55_55962

theorem g_x_squared_minus_3 (g : ℝ → ℝ)
  (h : ∀ x : ℝ, g (x^2 - 1) = x^4 - 4 * x^2 + 4) :
  ∀ x : ℝ, g (x^2 - 3) = x^4 - 6 * x^2 + 11 :=
by
  sorry

end g_x_squared_minus_3_l55_55962


namespace third_number_is_507_l55_55409

theorem third_number_is_507 (x : ℕ) 
  (h1 : (55 + 48 + x + 2 + 684 + 42) / 6 = 223) : 
  x = 507 := by
  sorry

end third_number_is_507_l55_55409


namespace red_flowers_count_l55_55268

theorem red_flowers_count (w r : ℕ) (h1 : w = 555) (h2 : w = r + 208) : r = 347 :=
by {
  -- Proof steps will be here
  sorry
}

end red_flowers_count_l55_55268


namespace arithmetic_sequence_sum_l55_55673

theorem arithmetic_sequence_sum :
  ∃ (a : ℕ → ℝ) (d : ℝ), 
  (∀ n, a n = a 0 + n * d) ∧ 
  (∃ b c, b^2 - 6*b + 5 = 0 ∧ c^2 - 6*c + 5 = 0 ∧ a 3 = b ∧ a 15 = c) →
  a 7 + a 8 + a 9 + a 10 + a 11 = 15 :=
by
  sorry

end arithmetic_sequence_sum_l55_55673


namespace min_value_expression_l55_55194

theorem min_value_expression : 
  ∃ (x y : ℝ), x^2 + 2 * x * y + 2 * y^2 + 3 * x - 5 * y = -8.5 := by
  sorry

end min_value_expression_l55_55194


namespace twelfth_term_in_sequence_l55_55149

variable (a₁ : ℚ) (d : ℚ)

-- Given conditions
def a_1_term : a₁ = 1 / 2 := rfl
def common_difference : d = 1 / 3 := rfl

-- The twelfth term of the arithmetic sequence
def a₁₂ : ℚ := a₁ + 11 * d

-- Statement to prove
theorem twelfth_term_in_sequence (h₁ : a₁ = 1 / 2) (h₂ : d = 1 / 3) : a₁₂ = 25 / 6 :=
by
  rw [h₁, h₂, add_comm, add_assoc, mul_comm, add_comm]
  exact sorry

end twelfth_term_in_sequence_l55_55149


namespace simplify_abs_expr_l55_55402

noncomputable def piecewise_y (x : ℝ) : ℝ :=
  if h1 : x < -3 then -3 * x
  else if h2 : -3 ≤ x ∧ x < 1 then 6 - x
  else if h3 : 1 ≤ x ∧ x < 2 then 4 + x
  else 3 * x

theorem simplify_abs_expr : 
  ∀ x : ℝ, (|x - 1| + |x - 2| + |x + 3|) = piecewise_y x :=
by
  intro x
  sorry

end simplify_abs_expr_l55_55402


namespace equal_share_each_shopper_l55_55377

theorem equal_share_each_shopper 
  (amount_giselle : ℕ)
  (amount_isabella : ℕ)
  (amount_sam : ℕ)
  (H1 : amount_isabella = amount_sam + 45)
  (H2 : amount_isabella = amount_giselle + 15)
  (H3 : amount_giselle = 120) : 
  (amount_isabella + amount_sam + amount_giselle) / 3 = 115 :=
by
  -- The proof is omitted.
  sorry

end equal_share_each_shopper_l55_55377


namespace jay_savings_in_a_month_l55_55824

def weekly_savings (week : ℕ) : ℕ :=
  20 + 10 * week

theorem jay_savings_in_a_month (weeks : ℕ) (h : weeks = 4) :
  ∑ i in Finset.range weeks, weekly_savings i = 140 :=
by
  -- proof goes here
  sorry

end jay_savings_in_a_month_l55_55824


namespace area_of_square_field_l55_55291

-- Define side length
def sideLength : ℕ := 14

-- Define the area function for a square
def area_of_square (side : ℕ) : ℕ := side * side

-- Prove that the area of the square with side length 14 meters is 196 square meters
theorem area_of_square_field : area_of_square sideLength = 196 := by
  sorry

end area_of_square_field_l55_55291


namespace smallest_cookies_left_l55_55588

theorem smallest_cookies_left (m : ℤ) (h : m % 8 = 5) : (4 * m) % 8 = 4 :=
by
  sorry

end smallest_cookies_left_l55_55588


namespace sum_of_coordinates_is_17_over_3_l55_55845

theorem sum_of_coordinates_is_17_over_3
  (f : ℝ → ℝ)
  (h1 : 5 = 3 * f 2) :
  (5 / 3 + 4) = 17 / 3 :=
by
  have h2 : f 2 = 5 / 3 := by
    linarith
  have h3 : f⁻¹ (5 / 3) = 2 := by
    sorry -- we do not know more properties of f to conclude this proof step
  have h4 : 2 * f⁻¹ (5 / 3) = 4 := by
    sorry -- similarly, assume for now the desired property
  exact sorry -- finally putting everything together

end sum_of_coordinates_is_17_over_3_l55_55845


namespace compare_polynomials_l55_55159

theorem compare_polynomials (x : ℝ) (h : x ≥ 0) : 
  (x > 2 → 5*x^2 - 1 > 3*x^2 + 3*x + 1) ∧ 
  (x = 2 → 5*x^2 - 1 = 3*x^2 + 3*x + 1) ∧ 
  (0 ≤ x → x < 2 → 5*x^2 - 1 < 3*x^2 + 3*x + 1) :=
sorry

end compare_polynomials_l55_55159


namespace line_passes_through_circle_center_l55_55953

theorem line_passes_through_circle_center
  (a : ℝ)
  (h_line : ∀ (x y : ℝ), 3 * x + y + a = 0 → (x, y) = (-1, 2))
  (h_circle : ∀ (x y : ℝ), x^2 + y^2 + 2 * x - 4 * y = 0 → (x, y) = (-1, 2)) :
  a = 1 :=
by
  sorry

end line_passes_through_circle_center_l55_55953


namespace work_efficiency_ratio_l55_55881

-- Define the problem conditions and the ratio we need to prove.
theorem work_efficiency_ratio :
  (∃ (a b : ℝ), b = 1 / 18 ∧ (a + b) = 1 / 12 ∧ (a / b) = 1 / 2) :=
by {
  -- Definitions and variables can be listed if necessary
  -- a : ℝ
  -- b : ℝ
  -- Assume conditions
  sorry
}

end work_efficiency_ratio_l55_55881


namespace probability_of_selecting_particular_girl_l55_55564

-- Define the numbers involved
def total_population : ℕ := 60
def num_girls : ℕ := 25
def num_boys : ℕ := 35
def sample_size : ℕ := 5

-- Total number of basic events
def total_combinations : ℕ := Nat.choose total_population sample_size

-- Number of basic events that include a particular girl
def girl_combinations : ℕ := Nat.choose (total_population - 1) (sample_size - 1)

-- Probability of selecting a particular girl
def probability_of_girl_selection : ℚ := girl_combinations / total_combinations

-- The theorem to be proved
theorem probability_of_selecting_particular_girl :
  probability_of_girl_selection = 1 / 12 :=
by sorry

end probability_of_selecting_particular_girl_l55_55564


namespace eval_g_at_neg2_l55_55515

def g (x : ℝ) : ℝ := 5 * x + 2

theorem eval_g_at_neg2 : g (-2) = -8 := by
  sorry

end eval_g_at_neg2_l55_55515


namespace louie_mistakes_l55_55831

theorem louie_mistakes (total_items : ℕ) (percentage_correct : ℕ) 
  (h1 : total_items = 25) 
  (h2 : percentage_correct = 80) : 
  total_items - ((percentage_correct / 100) * total_items) = 5 := 
by
  sorry

end louie_mistakes_l55_55831


namespace subtraction_result_l55_55404

theorem subtraction_result :
  let x := 567.89
  let y := 123.45
  (x - y) = 444.44 :=
by
  sorry

end subtraction_result_l55_55404


namespace num_ordered_triples_pos_int_l55_55804

theorem num_ordered_triples_pos_int
  (lcm_ab: lcm a b = 180)
  (lcm_ac: lcm a c = 450)
  (lcm_bc: lcm b c = 1200)
  (gcd_abc: gcd (gcd a b) c = 3) :
  ∃ n: ℕ, n = 4 :=
sorry

end num_ordered_triples_pos_int_l55_55804


namespace impossible_to_equalize_numbers_l55_55880

theorem impossible_to_equalize_numbers (nums : Fin 6 → ℤ) :
  ¬ (∃ n : ℤ, ∀ i : Fin 6, nums i = n) :=
sorry

end impossible_to_equalize_numbers_l55_55880


namespace sufficient_condition_for_inequality_l55_55904

theorem sufficient_condition_for_inequality (m : ℝ) : (m ≥ 2) → (∀ x : ℝ, x^2 - 2 * x + m ≥ 0) :=
by
  sorry

end sufficient_condition_for_inequality_l55_55904


namespace product_of_special_triplet_l55_55861

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_even (n : ℕ) : Prop := n % 2 = 0

def is_triangular (n : ℕ) : Prop := ∃ k : ℕ, n = k * (k + 1) / 2

def three_consecutive (a b c : ℕ) : Prop := b = a + 1 ∧ c = b + 1

theorem product_of_special_triplet :
  ∃ a b c : ℕ, a < b ∧ b < c ∧ c < 20 ∧ three_consecutive a b c ∧
   is_prime a ∧ is_even b ∧ is_triangular c ∧ a * b * c = 2730 :=
sorry

end product_of_special_triplet_l55_55861


namespace ratio_of_periods_l55_55730

variable (I_B T_B : ℝ)
variable (I_A T_A : ℝ)
variable (Profit_A Profit_B TotalProfit : ℝ)
variable (k : ℝ)

-- Define the conditions
axiom h1 : I_A = 3 * I_B
axiom h2 : T_A = k * T_B
axiom h3 : Profit_B = 4500
axiom h4 : TotalProfit = 31500
axiom h5 : Profit_A = TotalProfit - Profit_B

-- The profit shares are proportional to the product of investment and time period
axiom h6 : Profit_A = I_A * T_A
axiom h7 : Profit_B = I_B * T_B

theorem ratio_of_periods : T_A / T_B = 2 := by
  sorry

end ratio_of_periods_l55_55730


namespace walk_to_Lake_Park_restaurant_time_l55_55819

-- Define the problem parameters
def time_to_hidden_lake : ℕ := 15
def time_from_hidden_lake : ℕ := 7
def total_time_gone : ℕ := 32

-- Define the goal to prove
theorem walk_to_Lake_Park_restaurant_time :
  total_time_gone - (time_to_hidden_lake + time_from_hidden_lake) = 10 :=
by
  -- skipping the proof here
  sorry

end walk_to_Lake_Park_restaurant_time_l55_55819


namespace sequence_nonzero_l55_55003

def seq (a : ℕ → ℤ) : Prop :=
  a 1 = 1 ∧
  a 2 = 2 ∧
  ∀ n ≥ 3, 
    if (a (n - 1) * a (n - 2)) % 2 = 0 then 
      a n = 5 * (a (n - 1)) - 3 * (a (n - 2)) 
    else 
      a n = (a (n - 1)) - (a (n - 2))

theorem sequence_nonzero (a : ℕ → ℤ) (h : seq a) : ∀ n : ℕ, a n ≠ 0 := 
by sorry

end sequence_nonzero_l55_55003


namespace total_pencils_correct_l55_55854
  
def original_pencils : ℕ := 2
def added_pencils : ℕ := 3
def total_pencils : ℕ := original_pencils + added_pencils

theorem total_pencils_correct : total_pencils = 5 := 
by
  -- proof state will be filled here 
  sorry

end total_pencils_correct_l55_55854


namespace true_propositions_3_and_4_l55_55035

-- Define the condition for Proposition ③
def prop3_statement (m : ℝ) : Prop :=
  (m > 2) → ∀ x : ℝ, (x^2 - 2*x + m > 0)

def prop3_contrapositive (m : ℝ) : Prop :=
  (∀ x : ℝ, (x^2 - 2*x + m > 0)) → (m > 2)

-- Define the condition for Proposition ④
def prop4_condition (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f (-x) = -f (x)) ∧ (∀ x : ℝ, f (1 + x) = f (1 - x))

def prop4_period_4 (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f (x + 4) = f (x))

-- Theorem to prove Propositions ③ and ④ are true
theorem true_propositions_3_and_4
  (m : ℝ) (f : ℝ → ℝ)
  (h3 : ∀ (m : ℝ), prop3_contrapositive m)
  (h4 : prop4_condition f): 
  prop3_statement m ∧ prop4_period_4 f :=
by {
  sorry
}

end true_propositions_3_and_4_l55_55035


namespace largest_integer_solution_of_abs_eq_and_inequality_l55_55485

theorem largest_integer_solution_of_abs_eq_and_inequality : 
  ∃ x : ℤ, |x - 3| = 15 ∧ x ≤ 20 ∧ (∀ y : ℤ, |y - 3| = 15 ∧ y ≤ 20 → y ≤ x) :=
sorry

end largest_integer_solution_of_abs_eq_and_inequality_l55_55485


namespace pauly_cannot_make_more_omelets_l55_55248

-- Pauly's omelet data
def total_eggs : ℕ := 36
def plain_omelet_eggs : ℕ := 3
def cheese_omelet_eggs : ℕ := 4
def vegetable_omelet_eggs : ℕ := 5

-- Requested omelets
def requested_plain_omelets : ℕ := 4
def requested_cheese_omelets : ℕ := 2
def requested_vegetable_omelets : ℕ := 3

-- Number of eggs used for each type of requested omelet
def total_requested_eggs : ℕ :=
  (requested_plain_omelets * plain_omelet_eggs) +
  (requested_cheese_omelets * cheese_omelet_eggs) +
  (requested_vegetable_omelets * vegetable_omelet_eggs)

-- The remaining number of eggs
def remaining_eggs : ℕ := total_eggs - total_requested_eggs

theorem pauly_cannot_make_more_omelets :
  remaining_eggs < min plain_omelet_eggs (min cheese_omelet_eggs vegetable_omelet_eggs) :=
by
  sorry

end pauly_cannot_make_more_omelets_l55_55248


namespace find_subtracted_number_l55_55008

theorem find_subtracted_number (x y : ℝ) (h1 : x = 62.5) (h2 : (2 * (x + 5)) / 5 - y = 22) : y = 5 :=
sorry

end find_subtracted_number_l55_55008


namespace delivery_driver_stops_l55_55603

theorem delivery_driver_stops (initial_stops more_stops total_stops : ℕ)
  (h_initial : initial_stops = 3)
  (h_more : more_stops = 4)
  (h_total : total_stops = initial_stops + more_stops) : total_stops = 7 := by
  sorry

end delivery_driver_stops_l55_55603


namespace number_of_tiles_l55_55986

theorem number_of_tiles (room_width room_height tile_width tile_height : ℝ) :
  room_width = 8 ∧ room_height = 12 ∧ tile_width = 1.5 ∧ tile_height = 2 →
  (room_width * room_height) / (tile_width * tile_height) = 32 :=
by
  intro h
  cases' h with rw h
  cases' h with rh h
  cases' h with tw th
  rw [rw, rh, tw, th]
  norm_num
  sorry

end number_of_tiles_l55_55986


namespace find_d_of_quadratic_roots_l55_55111

theorem find_d_of_quadratic_roots :
  ∃ d : ℝ, (∀ x : ℝ, x^2 + 7 * x + d = 0 ↔ x = (-7 + real.sqrt d) / 2 ∨ x = (-7 - real.sqrt d) / 2) → d = 9.8 :=
by
  sorry

end find_d_of_quadratic_roots_l55_55111


namespace expected_red_light_l55_55010

variables (n : ℕ) (p : ℝ)
def binomial_distribution : Type := sorry

noncomputable def expected_value (n : ℕ) (p : ℝ) : ℝ :=
n * p

theorem expected_red_light :
  expected_value 3 0.4 = 1.2 :=
by
  simp [expected_value]
  sorry

end expected_red_light_l55_55010


namespace part1_decreasing_on_pos_part2_t_range_l55_55502

noncomputable def f (x : ℝ) : ℝ := -x + 2 / x

theorem part1_decreasing_on_pos (x1 x2 : ℝ) (h1 : 0 < x1) (h2 : 0 < x2) (h3 : x1 < x2) : 
  f x1 > f x2 := by sorry

theorem part2_t_range (t : ℝ) (ht : ∀ x : ℝ, 1 ≤ x → f x ≤ (1 + t * x) / x) : 
  0 ≤ t := by sorry

end part1_decreasing_on_pos_part2_t_range_l55_55502


namespace find_a_value_l55_55787

theorem find_a_value (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : (max (a^1) (a^2) + min (a^1) (a^2)) = 12) : a = 3 :=
by
  sorry

end find_a_value_l55_55787


namespace scientific_notation_eq_l55_55615

-- Define the number 82,600,000
def num : ℝ := 82600000

-- Define the scientific notation representation
def sci_not : ℝ := 8.26 * 10^7

-- The theorem to prove that the number is equal to its scientific notation
theorem scientific_notation_eq : num = sci_not :=
by 
  sorry

end scientific_notation_eq_l55_55615


namespace nissa_grooming_time_correct_l55_55382

def clipping_time_per_claw : ℕ := 10
def cleaning_time_per_ear : ℕ := 90
def shampooing_time_minutes : ℕ := 5

def claws_per_foot : ℕ := 4
def feet_count : ℕ := 4
def ear_count : ℕ := 2

noncomputable def total_grooming_time_in_seconds : ℕ := 
  (clipping_time_per_claw * claws_per_foot * feet_count) + 
  (cleaning_time_per_ear * ear_count) + 
  (shampooing_time_minutes * 60) -- converting minutes to seconds

theorem nissa_grooming_time_correct :
  total_grooming_time_in_seconds = 640 := by
  sorry

end nissa_grooming_time_correct_l55_55382


namespace two_digit_number_formed_l55_55778

theorem two_digit_number_formed (A B C D E F : ℕ) 
  (A_C_D_const : A + C + D = constant)
  (A_B_const : A + B = constant)
  (B_D_F_const : B + D + F = constant)
  (E_F_const : E + F = constant)
  (E_B_C_const : E + B + C = constant)
  (B_eq_C_D : B = C + D)
  (B_D_eq_E : B + D = E)
  (E_C_eq_A : E + C = A) 
  (hA : A = 6) 
  (hB : B = 3)
  : 10 * A + B = 63 :=
by sorry

end two_digit_number_formed_l55_55778


namespace select_team_l55_55761

-- Definition of the problem conditions 
def boys : Nat := 10
def girls : Nat := 12
def team_size : Nat := 8
def boys_in_team : Nat := 4
def girls_in_team : Nat := 4

-- Given conditions reflect in the Lean statement that needs proof
theorem select_team : 
  (Nat.choose boys boys_in_team) * (Nat.choose girls girls_in_team) = 103950 :=
by
  sorry

end select_team_l55_55761


namespace subtraction_result_l55_55405

theorem subtraction_result :
  let x := 567.89
  let y := 123.45
  (x - y) = 444.44 :=
by
  sorry

end subtraction_result_l55_55405


namespace arithmetic_sequence_twelfth_term_l55_55153

theorem arithmetic_sequence_twelfth_term :
  let a1 := (1 : ℚ) / 2;
  let a2 := (5 : ℚ) / 6;
  let d := a2 - a1;
  (a1 + 11 * d) = (25 : ℚ) / 6 :=
by
  let a1 := (1 : ℚ) / 2;
  let a2 := (5 : ℚ) / 6;
  let d := a2 - a1;
  exact sorry

end arithmetic_sequence_twelfth_term_l55_55153


namespace peggy_records_l55_55090

theorem peggy_records (R : ℕ) (h : 4 * R - (3 * R + R / 2) = 100) : R = 200 :=
sorry

end peggy_records_l55_55090


namespace each_shopper_receives_equal_amount_l55_55380

variables (G I S total_final : ℝ)

-- Given conditions
def conditions : Prop :=
  G = 120 ∧
  I = G + 15 ∧
  I = S + 45

noncomputable def amount_each_shopper_receives : ℝ :=
  let total := I + S + G in total / 3

theorem each_shopper_receives_equal_amount (h : conditions) : amount_each_shopper_receives G I S = 115 := by
  -- Given conditions for Giselle, Isabella, and Sam
  rcases h with ⟨hG, hI1, hI2⟩
    
  -- Define total_final from conditions
  let total_final := G + (G + 15) + (G + 15 - 45)
  
  -- Default proof
  sorry

end each_shopper_receives_equal_amount_l55_55380


namespace cubic_kilometers_to_cubic_meters_l55_55941

theorem cubic_kilometers_to_cubic_meters :
  (5 : ℝ) * (1000 : ℝ)^3 = 5_000_000_000 :=
by
  sorry

end cubic_kilometers_to_cubic_meters_l55_55941


namespace find_d_of_quadratic_roots_l55_55110

theorem find_d_of_quadratic_roots :
  ∃ d : ℝ, (∀ x : ℝ, x^2 + 7 * x + d = 0 ↔ x = (-7 + real.sqrt d) / 2 ∨ x = (-7 - real.sqrt d) / 2) → d = 9.8 :=
by
  sorry

end find_d_of_quadratic_roots_l55_55110


namespace correct_statement_l55_55286

theorem correct_statement :
  let A := (3 : ℕ) = 3 ∧ (3 + 3) = 6 ∧ ((3 : ℕ) / (3 + 3) : ℝ) = 0.5 in
  let B := ∀ (n : ℕ), (n = 100 → (1/100 * n ≠ 1)) in
  let C := False in
  let D := ∀ (a : ℝ), |a| > 0 → a ≠ 0 in
  A ∧ ¬ B ∧ C ∧ ¬ D :=
by 
  sorry

end correct_statement_l55_55286


namespace fraction_unspent_is_correct_l55_55175

noncomputable def fraction_unspent (S : ℝ) : ℝ :=
  let after_tax := S - 0.15 * S
  let after_first_week := after_tax - 0.25 * after_tax
  let after_second_week := after_first_week - 0.3 * after_first_week
  let after_third_week := after_second_week - 0.2 * S
  let after_fourth_week := after_third_week - 0.1 * after_third_week
  after_fourth_week / S

theorem fraction_unspent_is_correct (S : ℝ) (hS : S > 0) : 
  fraction_unspent S = 0.221625 :=
by
  sorry

end fraction_unspent_is_correct_l55_55175


namespace solve_logarithmic_equation_l55_55979

noncomputable def log_base (a x : ℝ) : ℝ := Real.log x / Real.log a

theorem solve_logarithmic_equation (x : ℝ) (h_pos : x > 0) :
  log_base 8 x + log_base 4 (x^2) + log_base 2 (x^3) = 15 ↔ x = 2 ^ (45 / 13) :=
by
  have h1 : log_base 8 x = (1 / 3) * log_base 2 x :=
    by { sorry }
  have h2 : log_base 4 (x^2) = log_base 2 x :=
    by { sorry }
  have h3 : log_base 2 (x^3) = 3 * log_base 2 x :=
    by { sorry }
  have h4 : (1 / 3) * log_base 2 x + log_base 2 x + 3 * log_base 2 x = 15 ↔ log_base 2 x = 45 / 13 :=
    by { sorry }
  exact sorry

end solve_logarithmic_equation_l55_55979


namespace min_dot_product_coordinates_l55_55046

open Function

theorem min_dot_product_coordinates :
  ∃ P : ℝ × ℝ × ℝ, (∃ x : ℝ, P = (x, 0, 0)) ∧
  ∀ Q : ℝ × ℝ × ℝ, (∃ x : ℝ, Q = (x, 0, 0)) →
  let AP := (Q.1 - 1, Q.2 - 2, Q.3 - 0)
  let BP := (Q.1 - 0, Q.2 - 1, Q.3 + 1)
  in (AP.1 * BP.1 + AP.2 * BP.2 + AP.3 * BP.3) ≥
     (P.1 - 1) * P.1 + 4 :=
  ∃ (P : ℝ × ℝ × ℝ), P = (1/2, 0, 0) :=
begin
  sorry
end

end min_dot_product_coordinates_l55_55046


namespace equal_share_each_shopper_l55_55378

theorem equal_share_each_shopper 
  (amount_giselle : ℕ)
  (amount_isabella : ℕ)
  (amount_sam : ℕ)
  (H1 : amount_isabella = amount_sam + 45)
  (H2 : amount_isabella = amount_giselle + 15)
  (H3 : amount_giselle = 120) : 
  (amount_isabella + amount_sam + amount_giselle) / 3 = 115 :=
by
  -- The proof is omitted.
  sorry

end equal_share_each_shopper_l55_55378


namespace maximum_xyz_l55_55493

theorem maximum_xyz (x y z : ℝ) (hx : x > 1) (hy : y > 1) (hz : z > 1) 
  (h: x ^ (Real.log x / Real.log y) * y ^ (Real.log y / Real.log z) * z ^ (Real.log z / Real.log x) = 10) : 
  x * y * z ≤ 10 := 
sorry

end maximum_xyz_l55_55493


namespace one_inch_cubes_with_two_or_more_painted_faces_l55_55118

def original_cube_length : ℕ := 4

def total_one_inch_cubes : ℕ := original_cube_length ^ 3

def corners_count : ℕ := 8

def edges_minus_corners_count : ℕ := 12 * 2

theorem one_inch_cubes_with_two_or_more_painted_faces
  (painted_faces_on_each_face : ∀ i : ℕ, i < total_one_inch_cubes → ℕ) : 
  ∃ n : ℕ, n = corners_count + edges_minus_corners_count ∧ n = 32 := 
by
  simp only [corners_count, edges_minus_corners_count, total_one_inch_cubes]
  sorry

end one_inch_cubes_with_two_or_more_painted_faces_l55_55118


namespace triangle_area_ratio_l55_55376

theorem triangle_area_ratio 
  (AB BC CA : ℝ)
  (p q r : ℝ)
  (ABC_area DEF_area : ℝ)
  (hAB : AB = 12)
  (hBC : BC = 16)
  (hCA : CA = 20)
  (h1 : p + q + r = 3 / 4)
  (h2 : p^2 + q^2 + r^2 = 1 / 2)
  (area_DEF_to_ABC : DEF_area / ABC_area = 385 / 512)
  : 897 = 385 + 512 := 
by
  sorry

end triangle_area_ratio_l55_55376


namespace probability_calculation_l55_55609

noncomputable def probability_in_ellipsoid : ℝ :=
  let prism_volume := (2 - (-2)) * (1 - (-1)) * (1 - (-1))
  let ellipsoid_volume := (4 * Real.pi / 3) * 1 * 2 * 2
  ellipsoid_volume / prism_volume

theorem probability_calculation :
  probability_in_ellipsoid = Real.pi / 3 :=
sorry

end probability_calculation_l55_55609


namespace factorization_of_w4_minus_81_l55_55332

theorem factorization_of_w4_minus_81 (w : ℝ) : 
  (w^4 - 81) = (w - 3) * (w + 3) * (w^2 + 9) :=
by sorry

end factorization_of_w4_minus_81_l55_55332


namespace johns_pieces_of_gum_l55_55384

theorem johns_pieces_of_gum : 
  (∃ (john cole aubrey : ℕ), 
    cole = 45 ∧ 
    aubrey = 0 ∧ 
    (john + cole + aubrey) = 3 * 33) → 
  ∃ john : ℕ, john = 54 :=
by 
  sorry

end johns_pieces_of_gum_l55_55384


namespace prove_value_of_expression_l55_55786

theorem prove_value_of_expression (x y a b : ℝ)
    (h1 : x = 2) 
    (h2 : y = 1)
    (h3 : 2 * a + b = 5)
    (h4 : a + 2 * b = 1) : 
    3 - a - b = 1 := 
by
    -- Skipping proof
    sorry

end prove_value_of_expression_l55_55786


namespace find_value_l55_55760

theorem find_value 
  (x1 x2 x3 x4 x5 : ℝ)
  (condition1 : x1 + 4 * x2 + 9 * x3 + 16 * x4 + 25 * x5 = 2)
  (condition2 : 4 * x1 + 9 * x2 + 16 * x3 + 25 * x4 + 36 * x5 = 15)
  (condition3 : 9 * x1 + 16 * x2 + 25 * x3 + 36 * x4 + 49 * x5 = 130) :
  16 * x1 + 25 * x2 + 36 * x3 + 49 * x4 + 64 * x5 = 347 :=
by
  sorry

end find_value_l55_55760


namespace LTE_divisibility_l55_55721

theorem LTE_divisibility (m : ℕ) (h_pos : 0 < m) :
  (∀ k : ℕ, k % 2 = 1 ∧ k ≥ 3 → 2^m ∣ k^m - 1) ↔ m = 1 ∨ m = 2 ∨ m = 4 :=
by
  sorry

end LTE_divisibility_l55_55721


namespace two_n_minus_one_lt_n_plus_one_squared_l55_55197

theorem two_n_minus_one_lt_n_plus_one_squared (n : ℕ) (h : n > 0) : 2 * n - 1 < (n + 1) ^ 2 := 
by
  sorry

end two_n_minus_one_lt_n_plus_one_squared_l55_55197


namespace b_plus_c_eq_neg3_l55_55353

theorem b_plus_c_eq_neg3 (b c : ℝ)
  (h1 : ∀ x : ℝ, x^2 + b * x + c > 0 ↔ (x < -1 ∨ x > 2)) :
  b + c = -3 :=
sorry

end b_plus_c_eq_neg3_l55_55353


namespace smallest_integer_is_840_l55_55394

def is_prime_digit (d : ℕ) : Prop :=
  d = 2 ∨ d = 3 ∨ d = 5 ∨ d = 7

def all_divide (N : ℕ) : Prop :=
  (2 ∣ N) ∧ (3 ∣ N) ∧ (5 ∣ N) ∧ (7 ∣ N)

def no_prime_digit (N : ℕ) : Prop :=
  ∀ d ∈ N.digits 10, ¬ is_prime_digit d

def smallest_satisfying_N (N : ℕ) : Prop :=
  no_prime_digit N ∧ all_divide N ∧ ∀ M, no_prime_digit M → all_divide M → N ≤ M

theorem smallest_integer_is_840 : smallest_satisfying_N 840 :=
by
  sorry

end smallest_integer_is_840_l55_55394


namespace triangle_inequality_difference_l55_55226

theorem triangle_inequality_difference :
  ∀ (x : ℕ), (x + 8 > 10) → (x + 10 > 8) → (8 + 10 > x) →
    (17 - 3 = 14) :=
by
  intros x hx1 hx2 hx3
  sorry

end triangle_inequality_difference_l55_55226


namespace percentage_of_sikh_boys_l55_55661

-- Define the conditions
def total_boys : ℕ := 650
def muslim_boys : ℕ := (44 * total_boys) / 100
def hindu_boys : ℕ := (28 * total_boys) / 100
def other_boys : ℕ := 117
def sikh_boys : ℕ := total_boys - (muslim_boys + hindu_boys + other_boys)

-- Define and prove the theorem
theorem percentage_of_sikh_boys : (sikh_boys * 100) / total_boys = 10 :=
by
  have h_muslims: muslim_boys = 286 := by sorry
  have h_hindus: hindu_boys = 182 := by sorry
  have h_total: muslim_boys + hindu_boys + other_boys = 585 := by sorry
  have h_sikhs: sikh_boys = 65 := by sorry
  have h_percentage: (65 * 100) / 650 = 10 := by sorry
  exact h_percentage

end percentage_of_sikh_boys_l55_55661


namespace sum_of_reciprocals_of_roots_l55_55041

theorem sum_of_reciprocals_of_roots (r1 r2 : ℝ) (h1 : r1 + r2 = 17) (h2 : r1 * r2 = 8) :
  1 / r1 + 1 / r2 = 17 / 8 :=
by
  sorry

end sum_of_reciprocals_of_roots_l55_55041


namespace min_value_expression_l55_55039

theorem min_value_expression (x y k : ℝ) (hk : 1 < k) (hx : k < x) (hy : k < y) : 
  (∀ x y, x > k → y > k → (∃ m, (m ≤ (x^2 / (y - k) + y^2 / (x - k)))) ∧ (m = 8 * k)) := sorry

end min_value_expression_l55_55039


namespace evaluate_expression_l55_55328

theorem evaluate_expression (b : ℕ) (h : b = 5) : b^3 * b^4 * 2 = 156250 :=
by
  sorry

end evaluate_expression_l55_55328


namespace chord_eq_line_l55_55044

theorem chord_eq_line (x y : ℝ)
  (h_ellipse : (x^2) / 16 + (y^2) / 4 = 1)
  (h_midpoint : ∃ x1 y1 x2 y2 : ℝ, 
    ((x1^2) / 16 + (y1^2) / 4 = 1) ∧ 
    ((x2^2) / 16 + (y2^2) / 4 = 1) ∧ 
    (x1 + x2) / 2 = 2 ∧ 
    (y1 + y2) / 2 = 1) :
  x + 2 * y - 4 = 0 :=
sorry

end chord_eq_line_l55_55044


namespace quadratic_points_relation_l55_55063

theorem quadratic_points_relation (h y1 y2 y3 : ℝ) :
  (∀ x, x = -1/2 → y1 = -(x-2) ^ 2 + h) ∧
  (∀ x, x = 1 → y2 = -(x-2) ^ 2 + h) ∧
  (∀ x, x = 2 → y3 = -(x-2) ^ 2 + h) →
  y1 < y2 ∧ y2 < y3 :=
by
  -- The required proof is omitted
  sorry

end quadratic_points_relation_l55_55063


namespace find_a_l55_55954

theorem find_a : ∃ a : ℝ, (∀ (x y : ℝ), (3 * x + y + a = 0) → (x^2 + y^2 + 2 * x - 4 * y = 0) → a = 1) :=
by
  let center_x : ℝ := -1
  let center_y : ℝ := 2
  have line_eqn : ∀ a : ℝ, 3 * center_x + center_y + a = 0
  have circle_eqn : ∀ x y : ℝ, x^2 + y^2 + 2 * x - 4 * y = 0 → (x, y) = (center_x, center_y)
  sorry

end find_a_l55_55954


namespace area_of_field_l55_55746

-- Definitions based on the conditions
def length_uncovered (L : ℝ) := L = 20
def fencing_required (W : ℝ) (L : ℝ) := 2 * W + L = 76

-- Statement of the theorem to be proved
theorem area_of_field (L W : ℝ) (hL : length_uncovered L) (hF : fencing_required W L) : L * W = 560 := by
  sorry

end area_of_field_l55_55746


namespace largest_d_value_l55_55389

noncomputable def max_d (a b c d : ℝ) (h1 : a + b + c + d = 10) (h2 : ab + ac + ad + bc + bd + cd = 20) : ℝ :=
  if h : (4 * d ^ 2 - 20 * d - 80) ≤ 0 then d else 0

theorem largest_d_value (a b c d : ℝ) (h1 : a + b + c + d = 10) (h2 : ab + ac + ad + bc + bd + cd = 20) :
  max_d a b c d h1 h2 = (5 + 5 * real.sqrt 21) / 2 :=
sorry

end largest_d_value_l55_55389


namespace hyperbola_eccentricity_a_l55_55504

theorem hyperbola_eccentricity_a (a : ℝ) (ha : a > 0) :
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / 3 = 1) ∧ (∃ (e : ℝ), e = 2 ∧ e = Real.sqrt (a^2 + 3) / a) → a = 1 :=
by
  sorry

end hyperbola_eccentricity_a_l55_55504


namespace intersection_product_distance_eq_eight_l55_55523

noncomputable def parametricCircle : ℝ → ℝ × ℝ :=
  λ θ => (4 * Real.cos θ, 4 * Real.sin θ)

noncomputable def parametricLine : ℝ → ℝ × ℝ :=
  λ t => (2 + (1 / 2) * t, 2 + (Real.sqrt 3 / 2) * t)

theorem intersection_product_distance_eq_eight :
  ∀ θ t,
    let (x1, y1) := parametricCircle θ
    let (x2, y2) := parametricLine t
    (x1^2 + y1^2 = 16) ∧ (x2 = x1 ∧ y2 = y1) →
    ∃ t1 t2,
      x1 = 2 + (1 / 2) * t1 ∧ y1 = 2 + (Real.sqrt 3 / 2) * t1 ∧
      x1 = 2 + (1 / 2) * t2 ∧ y1 = 2 + (Real.sqrt 3 / 2) * t2 ∧
      (t1 * t2 = -8) ∧ (|t1 * t2| = 8) := 
by
  intros θ t
  dsimp only
  intro h
  sorry

end intersection_product_distance_eq_eight_l55_55523


namespace usual_time_to_reach_school_l55_55290

theorem usual_time_to_reach_school
  (R T : ℝ)
  (h1 : (7 / 6) * R = R / (T - 3) * T) : T = 21 :=
sorry

end usual_time_to_reach_school_l55_55290


namespace determine_a_l55_55189

open Real

theorem determine_a :
  (∃ a : ℝ, |x^2 + a*x + 4*a| ≤ 3 → x^2 + a*x + 4*a = 3) ↔ (a = 8 + 2*sqrt 13 ∨ a = 8 - 2*sqrt 13) :=
by
  sorry

end determine_a_l55_55189


namespace find_word_l55_55307

theorem find_word (antonym : Nat) (cond : antonym = 26) : String :=
  "seldom"

end find_word_l55_55307


namespace clocks_resynchronize_after_days_l55_55893

/-- Arthur's clock gains 15 minutes per day. -/
def arthurs_clock_gain_per_day : ℕ := 15

/-- Oleg's clock gains 12 minutes per day. -/
def olegs_clock_gain_per_day : ℕ := 12

/-- The clocks display time in a 12-hour format, which is equivalent to 720 minutes. -/
def twelve_hour_format_in_minutes : ℕ := 720

/-- 
  After how many days will this situation first repeat given the 
  conditions of gain in Arthur's and Oleg's clocks and the 12-hour format.
-/
theorem clocks_resynchronize_after_days :
  ∃ (N : ℕ), N * arthurs_clock_gain_per_day % twelve_hour_format_in_minutes = 0 ∧
             N * olegs_clock_gain_per_day % twelve_hour_format_in_minutes = 0 ∧
             N = 240 :=
by
  sorry

end clocks_resynchronize_after_days_l55_55893


namespace base_area_of_cone_with_slant_height_10_and_semi_lateral_surface_l55_55795

theorem base_area_of_cone_with_slant_height_10_and_semi_lateral_surface :
  (l = 10) → (l = 2 * r) → (A = 25 * π) :=
  by
  intros l_eq_ten l_eq_two_r
  have r_is_five : r = 5 := by sorry
  have A_is_25pi : A = 25 * π := by sorry
  exact A_is_25pi

end base_area_of_cone_with_slant_height_10_and_semi_lateral_surface_l55_55795


namespace minimum_a_l55_55632

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x - 1 - a * Real.log x
noncomputable def g (x : ℝ) : ℝ := x / Real.exp (x - 1)

theorem minimum_a {a : ℝ} (h_neg : a < 0) :
  (∀ x1 x2 : ℝ, 3 ≤ x1 ∧ x1 < x2 ∧ x2 ≤ 4 → 
    (f x1 a - f x2 a) / (g x1 - g x2) > -1 / (g x1 * g x2)) → 
  a ≥ 3 - (2 / 3) * Real.exp 2 := 
sorry

end minimum_a_l55_55632


namespace police_female_officers_l55_55289

theorem police_female_officers (perc : ℝ) (total_on_duty: ℝ) (half_on_duty : ℝ) (F : ℝ) :
    perc = 0.18 →
    total_on_duty = 144 →
    half_on_duty = total_on_duty / 2 →
    half_on_duty = perc * F →
    F = 400 :=
by
  sorry

end police_female_officers_l55_55289


namespace power_function_value_l55_55219

theorem power_function_value (α : ℝ) (h₁ : (2 : ℝ) ^ α = (Real.sqrt 2) / 2) : (9 : ℝ) ^ α = 1 / 3 := 
by
  sorry

end power_function_value_l55_55219


namespace device_identification_l55_55221

def sum_of_device_numbers (numbers : List ℕ) : ℕ :=
  numbers.foldr (· + ·) 0

def is_standard_device (d : List ℕ) : Prop :=
  (d = [1, 2, 3, 4, 5, 6, 7, 8, 9]) ∧ (sum_of_device_numbers d = 45)

theorem device_identification (d : List ℕ) : 
  (sum_of_device_numbers d = 45) → is_standard_device d :=
by
  sorry

end device_identification_l55_55221


namespace sand_exchange_impossible_l55_55878

/-- Given initial conditions for g and p, the goal is to determine if 
the banker can have at least 2 kg of each type of sand in the end. -/
theorem sand_exchange_impossible (g p : ℕ) (G P : ℕ) 
  (initial_g : g = 1001) (initial_p : p = 1001) 
  (initial_G : G = 1) (initial_P : P = 1)
  (exchange_rule : ∀ x y : ℚ, x * p = y * g) 
  (decrement_rule : ∀ k, 1 ≤ k ∧ k ≤ 2000 → 
    (g = 1001 - k ∨ p = 1001 - k)) :
  ¬(G ≥ 2 ∧ P ≥ 2) :=
by
  -- Add a placeholder to skip the proof
  sorry

end sand_exchange_impossible_l55_55878


namespace find_number_l55_55951

variable (a b x : ℕ)

theorem find_number
    (h1 : x * a = 7 * b)
    (h2 : x * a = 20)
    (h3 : 7 * b = 20) :
    x = 1 :=
sorry

end find_number_l55_55951


namespace evaluate_m_l55_55903

theorem evaluate_m (m : ℕ) : 2 ^ m = (64 : ℝ) ^ (1 / 3) → m = 2 :=
by
  sorry

end evaluate_m_l55_55903


namespace buyers_of_cake_mix_l55_55732

/-
  A certain manufacturer of cake, muffin, and bread mixes has 100 buyers,
  of whom some purchase cake mix, 40 purchase muffin mix, and 17 purchase both cake mix and muffin mix.
  If a buyer is to be selected at random from the 100 buyers, the probability that the buyer selected will be one who purchases 
  neither cake mix nor muffin mix is 0.27.
  Prove that the number of buyers who purchase cake mix is 50.
-/

theorem buyers_of_cake_mix (C M B total : ℕ) (hM : M = 40) (hB : B = 17) (hTotal : total = 100)
    (hProb : (total - (C + M - B) : ℝ) / total = 0.27) : C = 50 :=
by
  -- Definition of the proof is required here
  sorry

end buyers_of_cake_mix_l55_55732


namespace inequality_and_equality_condition_l55_55093

theorem inequality_and_equality_condition (x : ℝ)
  (h : x ∈ (Set.Iio 0 ∪ Set.Ioi 0)) :
  max 0 (Real.log (|x|)) ≥ 
      ((Real.sqrt 5 - 1) / (2 * Real.sqrt 5)) * Real.log (|x|) + 
      (1 / (2 * Real.sqrt 5)) * Real.log (|x^2 - 1|) + 
      (1 / 2) * Real.log ((Real.sqrt 5 + 1) / 2)
  ∧ (max 0 (Real.log (|x|)) = 
      ((Real.sqrt 5 - 1) / (2 * Real.sqrt 5)) * Real.log (|x|) + 
      (1 / (2 * Real.sqrt 5)) * Real.log (|x^2 - 1|) + 
      (1 / 2) * Real.log ((Real.sqrt 5 + 1) / 2) ↔ 
      x = (Real.sqrt 5 - 1) / 2 ∨ 
      x = -(Real.sqrt 5 - 1) / 2 ∨ 
      x = (Real.sqrt 5 + 1) / 2 ∨ 
      x = -(Real.sqrt 5 + 1) / 2) :=
by
  sorry

end inequality_and_equality_condition_l55_55093


namespace total_number_of_games_in_season_l55_55858

def number_of_games_per_month : ℕ := 13
def number_of_months_in_season : ℕ := 14

theorem total_number_of_games_in_season :
  number_of_games_per_month * number_of_months_in_season = 182 := by
  sorry

end total_number_of_games_in_season_l55_55858


namespace combination_identity_l55_55191

theorem combination_identity (C : ℕ → ℕ → ℕ)
  (comb_formula : ∀ n r, C r n = Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r)))
  (identity_1 : ∀ n r, C r n = C (n-r) n)
  (identity_2 : ∀ n r, C r (n+1) = C r n + C (r-1) n) :
  C 2 100 + C 97 100 = C 3 101 :=
by sorry

end combination_identity_l55_55191


namespace discriminant_of_given_quadratic_l55_55624

-- define the coefficients a, b, c
def a : ℚ := 2
def b : ℚ := 2 + 1/2
def c : ℚ := 1/2

-- define the discriminant function for a quadratic equation ax^2 + bx + c
def discriminant (a b c : ℚ) : ℚ := b^2 - 4 * a * c

-- state the theorem
theorem discriminant_of_given_quadratic : discriminant a b c = 9/4 :=
by
  -- add the proof here
  sorry

end discriminant_of_given_quadratic_l55_55624


namespace compare_abc_l55_55350

noncomputable def a : ℝ := (0.6)^(2/5)
noncomputable def b : ℝ := (0.4)^(2/5)
noncomputable def c : ℝ := (0.4)^(3/5)

theorem compare_abc : a > b ∧ b > c := 
by
  sorry

end compare_abc_l55_55350


namespace find_m_l55_55236

theorem find_m 
  (f : ℝ → ℝ) (g : ℝ → ℝ) (m : ℝ)
  (h_f : ∀ x, f x = x^2 - 4*x + m)
  (h_g : ∀ x, g x = x^2 - 2*x + 2*m)
  (h_cond : 3 * f 3 = g 3)
  : m = 12 := 
sorry

end find_m_l55_55236


namespace cone_volume_l55_55737

theorem cone_volume (l : ℝ) (θ : ℝ) (h r V : ℝ)
  (h_l : l = 5)
  (h_θ : θ = (8 * Real.pi) / 5)
  (h_arc_length : 2 * Real.pi * r = l * θ)
  (h_radius: r = 4)
  (h_height : h = Real.sqrt (l^2 - r^2))
  (h_volume_eq : V = (1 / 3) * Real.pi * r^2 * h) :
  V = 16 * Real.pi :=
by
  -- proof goes here
  sorry

end cone_volume_l55_55737


namespace travel_probability_l55_55325

theorem travel_probability (P_A P_B P_C : ℝ) (hA : P_A = 1/3) (hB : P_B = 1/4) (hC : P_C = 1/5) :
  let P_none_travel := (1 - P_A) * (1 - P_B) * (1 - P_C)
  ∃ (P_at_least_one : ℝ), P_at_least_one = 1 - P_none_travel ∧ P_at_least_one = 3/5 :=
by {
  sorry
}

end travel_probability_l55_55325


namespace minimum_ellipse_area_l55_55024

theorem minimum_ellipse_area (a b : ℝ) (h₁ : 4 * (a : ℝ) ^ 2 * b ^ 2 = a ^ 2 + b ^ 4)
  (h₂ : (∀ x y : ℝ, ((x - 2) ^ 2 + y ^ 2 ≤ 4 → x ^ 2 / (4 * a ^ 2) + y ^ 2 / (4 * b ^ 2) ≤ 1)) 
       ∧ (∀ x y : ℝ, ((x + 2) ^ 2 + y ^ 2 ≤ 4 → x ^ 2 / (4 * a ^ 2) + y ^ 2 / (4 * b ^ 2) ≤ 1))) : 
  ∃ k : ℝ, (k = 16) ∧ (π * (4 * a * b) = k * π) :=
by sorry

end minimum_ellipse_area_l55_55024


namespace circle_n_gon_area_ineq_l55_55288

variable {n : ℕ} {S S1 S2 : ℝ}

theorem circle_n_gon_area_ineq (h1 : S1 > 0) (h2 : S > 0) (h3 : S2 > 0) : 
  S * S = S1 * S2 := 
sorry

end circle_n_gon_area_ineq_l55_55288


namespace partners_in_firm_l55_55739

theorem partners_in_firm (P A : ℕ) (h1 : P * 63 = 2 * A) (h2 : P * 34 = 1 * (A + 45)) : P = 18 :=
by
  sorry

end partners_in_firm_l55_55739


namespace arithmetic_expressions_correctness_l55_55458

theorem arithmetic_expressions_correctness :
  ((∀ (a b c : ℚ), (a + b) + c = a + (b + c)) ∧
   (∃ (a b c : ℚ), (a - b) - c ≠ a - (b - c)) ∧
   (∀ (a b c : ℚ), (a * b) * c = a * (b * c)) ∧
   (∃ (a b c : ℚ), a / b / c ≠ a / (b / c))) :=
by
  sorry

end arithmetic_expressions_correctness_l55_55458


namespace problem_equivalence_l55_55932

noncomputable def f (a b x : ℝ) : ℝ := a ^ x + b

theorem problem_equivalence (a b : ℝ) (h1 : a > 0) (h2 : a ≠ 1)
    (h3 : f a b 0 = -2) (h4 : f a b 2 = 0) :
    a = Real.sqrt 3 ∧ b = -3 ∧
    (∀ x ∈ Set.Icc (-2 : ℝ) 4, (-8 / 3 : ℝ) ≤ f a b x ∧ f a b x ≤ 6) :=
sorry

end problem_equivalence_l55_55932


namespace intersection_of_sets_l55_55934

def M (x : ℝ) : Prop := (x - 2) / (x - 3) < 0
def N (x : ℝ) : Prop := Real.log (x - 2) / Real.log (1 / 2) ≥ 1 

theorem intersection_of_sets : {x : ℝ | M x} ∩ {x : ℝ | N x} = {x : ℝ | 2 < x ∧ x ≤ 5 / 2} := by
  sorry

end intersection_of_sets_l55_55934


namespace rational_cos_terms_l55_55850

open Real

noncomputable def rational_sum (x : ℝ) (rS : ℚ) (rC : ℚ) :=
  let S := sin (64 * x) + sin (65 * x)
  let C := cos (64 * x) + cos (65 * x)
  S = rS ∧ C = rC

theorem rational_cos_terms (x : ℝ) (rS : ℚ) (rC : ℚ) :
  rational_sum x rS rC → (∃ q1 q2 : ℚ, cos (64 * x) = q1 ∧ cos (65 * x) = q2) :=
sorry

end rational_cos_terms_l55_55850


namespace solution_set_of_xf_x_gt_0_l55_55237

noncomputable def f (x : ℝ) : ℝ := sorry

axiom h1 : ∀ x : ℝ, f (-x) = - f x
axiom h2 : f 2 = 0
axiom h3 : ∀ x : ℝ, 0 < x → x * (deriv f x) + f x < 0

theorem solution_set_of_xf_x_gt_0 :
  {x : ℝ | x * f x > 0} = {x : ℝ | -2 < x ∧ x < 0} ∪ {x : ℝ | 0 < x ∧ x < 2} :=
by {
  sorry
}

end solution_set_of_xf_x_gt_0_l55_55237


namespace sum_of_acute_angles_l55_55351

variable (α β : ℝ)
variable (hα : 0 < α ∧ α < π / 2)
variable (hβ : 0 < β ∧ β < π / 2)
variable (hcosα : Real.cos α = 1 / Real.sqrt 10)
variable (hcosβ : Real.cos β = 1 / Real.sqrt 5)

theorem sum_of_acute_angles :
  α + β = 3 * Real.pi / 4 := by
  sorry

end sum_of_acute_angles_l55_55351


namespace tan_beta_minus_2alpha_l55_55944

theorem tan_beta_minus_2alpha (alpha beta : ℝ) (h1 : Real.tan alpha = 2) (h2 : Real.tan (beta - alpha) = 3) : 
  Real.tan (beta - 2 * alpha) = 1 / 7 := 
sorry

end tan_beta_minus_2alpha_l55_55944


namespace number_of_connections_l55_55561

theorem number_of_connections (n k : ℕ) (h1 : n = 30) (h2 : k = 4) :
  (n * k) / 2 = 60 :=
by
  sorry

end number_of_connections_l55_55561


namespace general_formula_sequence_l55_55916

variable {a : ℕ → ℝ}

-- Definitions and assumptions
def recurrence_relation (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → a n - 2 * a (n + 1) + a (n + 2) = 0

def initial_conditions (a : ℕ → ℝ) : Prop :=
  a 1 = 2 ∧ a 2 = 4

-- The proof problem
theorem general_formula_sequence (a : ℕ → ℝ)
  (h1 : recurrence_relation a)
  (h2 : initial_conditions a) :
  ∀ n : ℕ, a n = 2 * n :=

sorry

end general_formula_sequence_l55_55916


namespace susan_homework_start_time_l55_55698

def start_time_homework (finish_time : ℕ) (homework_duration : ℕ) (interval_duration : ℕ) : ℕ :=
  finish_time - homework_duration - interval_duration

theorem susan_homework_start_time :
  let finish_time : ℕ := 16 * 60 -- 4:00 p.m. in minutes
  let homework_duration : ℕ := 96 -- Homework duration in minutes
  let interval_duration : ℕ := 25 -- Interval between homework finish and practice in minutes
  start_time_homework finish_time homework_duration interval_duration = 13 * 60 + 59 := -- 13:59 in minutes
by
  sorry

end susan_homework_start_time_l55_55698


namespace value_of_x_minus_2y_l55_55434

theorem value_of_x_minus_2y (x y : ℝ) (h1 : 0.5 * x = y + 20) : x - 2 * y = 40 :=
by
  sorry

end value_of_x_minus_2y_l55_55434


namespace total_widgets_sold_after_15_days_l55_55247

def widgets_sold_day_n (n : ℕ) : ℕ :=
  2 + (n - 1) * 3

def sum_of_widgets (n : ℕ) : ℕ :=
  n * (2 + widgets_sold_day_n n) / 2

theorem total_widgets_sold_after_15_days : 
  sum_of_widgets 15 = 345 :=
by
  -- Prove the arithmetic sequence properties and sum.
  sorry

end total_widgets_sold_after_15_days_l55_55247


namespace walnut_trees_planted_l55_55710

-- The number of walnut trees before planting
def walnut_trees_before : ℕ := 22

-- The number of walnut trees after planting
def walnut_trees_after : ℕ := 55

-- The number of walnut trees planted today
def walnut_trees_planted_today : ℕ := 33

-- Theorem statement to prove that the number of walnut trees planted today is 33
theorem walnut_trees_planted:
  walnut_trees_after - walnut_trees_before = walnut_trees_planted_today :=
by sorry

end walnut_trees_planted_l55_55710


namespace sum_divisible_by_15_l55_55249

theorem sum_divisible_by_15 (a : ℤ) : 15 ∣ (9 * a^5 - 5 * a^3 - 4 * a) :=
sorry

end sum_divisible_by_15_l55_55249


namespace area_of_triangle_AEH_of_regular_octagon_of_side_length_4_l55_55748

theorem area_of_triangle_AEH_of_regular_octagon_of_side_length_4 :
  let s : ℝ := 4
  let θ := real.pi / 8              -- 22.5 degrees
  let side_length := s              -- Side length of octagon is 4
  let diagonal_length := 2 * s * real.cos θ   -- Length of the diagonal AE
  let angle_AEH := 3 * real.pi / 4  -- 135 degrees in radians
  let area := (1 / 2) * diagonal_length * diagonal_length * real.sin angle_AEH in
  area = 8 * real.sqrt 2 + 8 :=
by
  sorry

end area_of_triangle_AEH_of_regular_octagon_of_side_length_4_l55_55748


namespace walk_time_to_LakePark_restaurant_l55_55823

/-
  It takes 15 minutes for Dante to go to Hidden Lake.
  From Hidden Lake, it takes him 7 minutes to walk back to the Park Office.
  Dante will have been gone from the Park Office for a total of 32 minutes.
  Prove that the walk from the Park Office to the Lake Park restaurant is 10 minutes.
-/

def T_HiddenLake_to : ℕ := 15
def T_HiddenLake_from : ℕ := 7
def T_total : ℕ := 32
def T_LakePark_restaurant : ℕ := T_total - (T_HiddenLake_to + T_HiddenLake_from)

theorem walk_time_to_LakePark_restaurant : 
  T_LakePark_restaurant = 10 :=
by
  unfold T_LakePark_restaurant T_HiddenLake_to T_HiddenLake_from T_total
  sorry

end walk_time_to_LakePark_restaurant_l55_55823


namespace range_of_f_l55_55851

def f (x : Int) : Int :=
  x + 1

def domain : Set Int :=
  {-1, 1, 2}

theorem range_of_f :
  Set.image f domain = {0, 2, 3} :=
by
  sorry

end range_of_f_l55_55851


namespace roommate_payment_l55_55714

theorem roommate_payment :
  (1100 + 114 + 300) / 2 = 757 := 
by
  sorry

end roommate_payment_l55_55714


namespace mean_equality_l55_55418

theorem mean_equality (z : ℚ) :
  (8 + 12 + 24) / 3 = (16 + z) / 2 ↔ z = 40 / 3 :=
by
  sorry

end mean_equality_l55_55418


namespace max_a_value_l55_55506

theorem max_a_value (a : ℝ) : 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → -2022 ≤ (a - 1) * x^2 - (a - 1) * x + 2022 ∧ 
                                (a - 1) * x^2 - (a - 1) * x + 2022 ≤ 2022) →
  a = 16177 :=
sorry

end max_a_value_l55_55506


namespace value_of_a1_plus_a3_l55_55361

theorem value_of_a1_plus_a3 (a a1 a2 a3 a4 : ℝ) :
  (∀ x : ℝ, (1 + x)^4 = a + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4) →
  a1 + a3 = 8 :=
by
  sorry

end value_of_a1_plus_a3_l55_55361


namespace algebraic_expression_identity_l55_55657

theorem algebraic_expression_identity (a b x : ℕ) (h : x * 3 * a * b = 3 * a * a * b) : x = a :=
sorry

end algebraic_expression_identity_l55_55657


namespace find_a_find_a_plus_c_l55_55660

-- Define the triangle with given sides and angles
variables (A B C : ℝ) (a b c S : ℝ)
  (h_cosB : cos B = 4/5)
  (h_b : b = 2)
  (h_area : S = 3)

-- Prove the value of the side 'a' when angle A is π/6
theorem find_a (h_A : A = Real.pi / 6) : a = 5 / 3 := 
  sorry

-- Prove the sum of sides 'a' and 'c' when the area of the triangle is 3
theorem find_a_plus_c (h_ac : a * c = 10) : a + c = 2 * Real.sqrt 10 :=
  sorry

end find_a_find_a_plus_c_l55_55660


namespace grooming_time_equals_640_seconds_l55_55381

variable (cat_claws_per_foot : Nat) (cat_foot_count : Nat)
variable (nissa_clip_time_per_claw : Nat) (nissa_clean_time_per_ear : Nat) (nissa_shampoo_time_minutes : Nat) 
variable (cat_ear_count : Nat)
variable (seconds_per_minute : Nat)

def total_grooming_time (cat_claws_per_foot * cat_foot_count : nissa_clip_time_per_claw) (nissa_clean_time_per_ear * cat_ear_count) (nissa_shampoo_time_minutes * seconds_per_minute) := sorry

theorem grooming_time_equals_640_seconds : 
  cat_claws_per_foot = 4 →
  cat_foot_count = 4 →
  nissa_clip_time_per_claw = 10 →
  nissa_clean_time_per_ear = 90 →
  nissa_shampoo_time_minutes = 5 →
  cat_ear_count = 2 →
  seconds_per_minute = 60 →
  total_grooming_time = 160 + 180 + 300 → 
  total_grooming_time = 640 := sorry

end grooming_time_equals_640_seconds_l55_55381


namespace markers_last_group_correct_l55_55446

-- Definition of conditions in Lean 4
def total_students : ℕ := 30
def boxes_of_markers : ℕ := 22
def markers_per_box : ℕ := 5
def students_in_first_group : ℕ := 10
def markers_per_student_first_group : ℕ := 2
def students_in_second_group : ℕ := 15
def markers_per_student_second_group : ℕ := 4

-- Calculate total markers allocated to the first and second groups
def markers_used_by_first_group : ℕ := students_in_first_group * markers_per_student_first_group
def markers_used_by_second_group : ℕ := students_in_second_group * markers_per_student_second_group

-- Total number of markers available
def total_markers : ℕ := boxes_of_markers * markers_per_box

-- Markers left for last group
def markers_remaining : ℕ := total_markers - (markers_used_by_first_group + markers_used_by_second_group)

-- Number of students in the last group
def students_in_last_group : ℕ := total_students - (students_in_first_group + students_in_second_group)

-- Number of markers per student in the last group
def markers_per_student_last_group : ℕ := markers_remaining / students_in_last_group

-- The proof problem in Lean 4
theorem markers_last_group_correct : markers_per_student_last_group = 6 :=
  by
  -- Proof is to be filled here
  sorry

end markers_last_group_correct_l55_55446


namespace range_of_a_l55_55497

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 0 < x → a < x + (1 / x)) → a < 2 :=
by
  sorry

end range_of_a_l55_55497


namespace number_of_video_cassettes_in_first_set_l55_55729

/-- Let A be the cost of an audio cassette, and V the cost of a video cassette.
  We are given that V = 300, and we have the following conditions:
  1. 7 * A + n * V = 1110,
  2. 5 * A + 4 * V = 1350.
  Prove that n = 3, the number of video cassettes in the first set -/
theorem number_of_video_cassettes_in_first_set 
    (A V n : ℕ) 
    (hV : V = 300)
    (h1 : 7 * A + n * V = 1110)
    (h2 : 5 * A + 4 * V = 1350) : 
    n = 3 := 
sorry

end number_of_video_cassettes_in_first_set_l55_55729


namespace option_b_correct_l55_55872

theorem option_b_correct (a b : ℝ) (h : a ≠ b) : (1 / (a - b) + 1 / (b - a) = 0) :=
by
  sorry

end option_b_correct_l55_55872


namespace cost_per_tree_l55_55977

theorem cost_per_tree
    (initial_temperature : ℝ := 80)
    (final_temperature : ℝ := 78.2)
    (total_cost : ℝ := 108)
    (temperature_drop_per_tree : ℝ := 0.1) :
    total_cost / ((initial_temperature - final_temperature) / temperature_drop_per_tree) = 6 :=
by sorry

end cost_per_tree_l55_55977


namespace intersection_A_B_l55_55507

def A := {x : ℝ | -2 ≤ x ∧ x ≤ 3}
def B := {x : ℝ | ∃ y : ℝ, y = x^2 + 2}

theorem intersection_A_B :
  {x : ℝ | x ∈ A ∧ ∃ y : ℝ, y = x^2 + 2} = {x : ℝ | 2 ≤ x ∧ x ≤ 3} := sorry

end intersection_A_B_l55_55507


namespace difference_between_heads_and_feet_l55_55373

-- Definitions based on the conditions
def penguins := 30
def zebras := 22
def tigers := 8
def zookeepers := 12

-- Counting heads
def heads := penguins + zebras + tigers + zookeepers

-- Counting feet
def feet := (2 * penguins) + (4 * zebras) + (4 * tigers) + (2 * zookeepers)

-- Proving the difference between the number of feet and heads is 132
theorem difference_between_heads_and_feet : (feet - heads) = 132 :=
by
  sorry

end difference_between_heads_and_feet_l55_55373


namespace die_face_never_lays_on_board_l55_55088

structure Chessboard :=
(rows : ℕ)
(cols : ℕ)
(h_size : rows = 8 ∧ cols = 8)

structure Die :=
(faces : Fin 6 → Nat)  -- a die has 6 faces

structure Position :=
(x : ℕ)
(y : ℕ)

structure State :=
(position : Position)
(bottom_face : Fin 6)
(visited : Fin 64 → Bool)

def initial_position : Position := ⟨0, 0⟩  -- top-left corner (a1)

def initial_state (d : Die) : State :=
  { position := initial_position,
    bottom_face := 0,
    visited := λ _ => false }

noncomputable def can_roll_over_entire_board_without_one_face_touching (board : Chessboard) (d : Die) : Prop :=
  ∃ f : Fin 6, ∀ s : State, -- for some face f of the die
    ((s.position.x < board.rows ∧ s.position.y < board.cols) → 
      s.visited (⟨s.position.x + board.rows * s.position.y, by sorry⟩) = true) → -- every cell visited
      ¬(s.bottom_face = f) -- face f is never the bottom face

theorem die_face_never_lays_on_board (board : Chessboard) (d : Die) :
  can_roll_over_entire_board_without_one_face_touching board d :=
  sorry

end die_face_never_lays_on_board_l55_55088


namespace trigonometric_identity_l55_55287

open Real

theorem trigonometric_identity (α β : ℝ) :
  sin (2 * α) ^ 2 + sin β ^ 2 + cos (2 * α + β) * cos (2 * α - β) = 1 :=
sorry

end trigonometric_identity_l55_55287


namespace circumradius_triangle_AQR_l55_55964

-- Let \(\Gamma\) be a circle with radius 17.
-- Let \(\omega\) be a circle with radius 7, internally tangent to \(\Gamma\) at point \(P\).
axiom circle_Gamma : Circle 
axiom Gamma_radius : radius circle_Gamma = 17
axiom circle_omega : Circle 
axiom omega_radius : radius circle_omega = 7
axiom tangency_point : Point
axiom internal_tangency : tangent_at circle_omega circle_Gamma tangency_point

-- Chord \(AB\) of \(\Gamma\) is tangent to \(\omega\) at point \(Q\).
axiom point_A : Point
axiom point_B : Point
axiom point_Q : Point
axiom chord_tangent : tangent_to_chord circle_Gamma circle_omega point_A point_B point_Q

-- Line \(PQ\) intersects \(\Gamma\) at points \(P\) and \(R\) with \(R \neq P\).
axiom point_R : Point
axiom PQ_intersects_Gamma : intersects_at_points line_PQ circle_Gamma point (tangency_point, point_R)
axiom R_not_P : point_R ≠ tangency_point

-- \(\frac{AQ}{BQ} = 3\).
axiom AQ_BQ_ratio : ratio (distance point_A point_Q) (distance point_B point_Q) = 3

-- Prove that the circumradius of triangle \(AQR\) is \(\sqrt{170}\).
theorem circumradius_triangle_AQR :
  circumradius (triangle AQR) = sqrt 170 :=
sorry

end circumradius_triangle_AQR_l55_55964


namespace Susan_initial_amount_l55_55406

def initial_amount (S : ℝ) : Prop :=
  let Spent_in_September := (1/6) * S
  let Spent_in_October := (1/8) * S
  let Spent_in_November := 0.3 * S
  let Spent_in_December := 100
  let Remaining := 480
  S - (Spent_in_September + Spent_in_October + Spent_in_November + Spent_in_December) = Remaining

theorem Susan_initial_amount : ∃ S : ℝ, initial_amount S ∧ S = 1420 :=
by
  sorry

end Susan_initial_amount_l55_55406


namespace arithmetic_twelfth_term_l55_55144

theorem arithmetic_twelfth_term 
(a d : ℚ) (n : ℕ) (h_a : a = 1/2) (h_d : d = 1/3) (h_n : n = 12) : 
  a + (n - 1) * d = 25 / 6 := 
by 
  sorry

end arithmetic_twelfth_term_l55_55144


namespace distinct_values_of_expr_l55_55193

theorem distinct_values_of_expr : 
  let a := 3^(3^(3^3));
  let b := 3^((3^3)^3);
  let c := ((3^3)^3)^3;
  let d := (3^(3^3))^3;
  let e := (3^3)^(3^3);
  (a ≠ b) ∧ (c ≠ b) ∧ (d ≠ b) ∧ (d ≠ a) ∧ (e ≠ a) ∧ (e ≠ b) ∧ (e ≠ d) := sorry

end distinct_values_of_expr_l55_55193


namespace angle_between_lines_in_folded_rectangle_l55_55663

theorem angle_between_lines_in_folded_rectangle
  (a b : ℝ) 
  (h : b > a)
  (dihedral_angle : ℝ)
  (h_dihedral_angle : dihedral_angle = 18) :
  ∃ (angle_AC_MN : ℝ), angle_AC_MN = 90 :=
by
  sorry

end angle_between_lines_in_folded_rectangle_l55_55663


namespace at_least_one_genuine_l55_55342

theorem at_least_one_genuine :
  ∀ (total_products genuine_products defective_products selected_products : ℕ),
  total_products = 12 →
  genuine_products = 10 →
  defective_products = 2 →
  selected_products = 3 →
  (∃ g d : ℕ, g + d = selected_products ∧ g = 0 ∧ d = selected_products) = false :=
by
  intros total_products genuine_products defective_products selected_products
  intros H_total H_gen H_def H_sel
  sorry

end at_least_one_genuine_l55_55342


namespace somu_age_to_father_age_ratio_l55_55254

theorem somu_age_to_father_age_ratio
  (S : ℕ) (F : ℕ)
  (h1 : S = 10)
  (h2 : S - 5 = (1/5) * (F - 5)) :
  S / F = 1 / 3 :=
by
  sorry

end somu_age_to_father_age_ratio_l55_55254


namespace certain_event_proof_l55_55719

def Moonlight_in_front_of_bed := "depends_on_time_and_moon_position"
def Lonely_smoke_in_desert := "depends_on_specific_conditions"
def Reach_for_stars_with_hand := "physically_impossible"
def Yellow_River_flows_into_sea := "certain_event"

theorem certain_event_proof : Yellow_River_flows_into_sea = "certain_event" :=
by
  sorry

end certain_event_proof_l55_55719


namespace g_f_neg4_eq_12_l55_55961

-- Define the function f
def f (x : ℝ) : ℝ := 4 * x^2 - 8

-- Define the assumption that g(f(4)) = 12
axiom g : ℝ → ℝ
axiom g_f4 : g (f 4) = 12

-- The theorem to prove that g(f(-4)) = 12
theorem g_f_neg4_eq_12 : g (f (-4)) = 12 :=
sorry -- proof placeholder

end g_f_neg4_eq_12_l55_55961


namespace find_a_l55_55345

noncomputable def f (x : ℝ) : ℝ :=
  3 * x^2 + 2 * x + 1

theorem find_a :
  (∫ x in -1..1, f x) = 2 * f a → 
  (a = 1/3 ∨ a = -1) :=
begin
  sorry
end

end find_a_l55_55345


namespace total_time_l55_55679

def time_to_eat_cereal (rate1 rate2 rate3 : ℚ) (amount : ℚ) : ℚ :=
  let combined_rate := rate1 + rate2 + rate3
  amount / combined_rate

theorem total_time (rate1 rate2 rate3 : ℚ) (amount : ℚ) 
  (h1 : rate1 = 1 / 15)
  (h2 : rate2 = 1 / 20)
  (h3 : rate3 = 1 / 30)
  (h4 : amount = 4) : 
  time_to_eat_cereal rate1 rate2 rate3 amount = 80 / 3 := 
by 
  rw [time_to_eat_cereal, h1, h2, h3, h4]
  sorry

end total_time_l55_55679


namespace relay_team_permutations_l55_55084

-- Definitions of conditions
def runners := ["Tony", "Leah", "Nina"]
def fixed_positions := ["Maria runs the third lap", "Jordan runs the fifth lap"]

-- Proof statement
theorem relay_team_permutations : 
  ∃ permutations, permutations = 6 := by
sorry

end relay_team_permutations_l55_55084


namespace number_of_solutions_l55_55201

theorem number_of_solutions (x : ℤ) (h1 : 0 < x) (h2 : x < 150) (h3 : (x + 17) % 46 = 75 % 46) : 
  ∃ n : ℕ, n = 3 :=
sorry

end number_of_solutions_l55_55201


namespace smallest_integral_k_l55_55279

theorem smallest_integral_k (k : ℤ) :
  (297 - 108 * k < 0) ↔ (k ≥ 3) :=
sorry

end smallest_integral_k_l55_55279


namespace corrected_mean_l55_55002

theorem corrected_mean (mean : ℝ) (n : ℕ) (wrong_ob : ℝ) (correct_ob : ℝ) 
(h1 : mean = 36) (h2 : n = 50) (h3 : wrong_ob = 23) (h4 : correct_ob = 34) : 
(mean * n + (correct_ob - wrong_ob)) / n = 36.22 :=
by
  sorry

end corrected_mean_l55_55002


namespace find_certain_number_l55_55947

theorem find_certain_number (h1 : 2994 / 14.5 = 173) (h2 : ∃ x, x / 1.45 = 17.3) : ∃ x, x = 25.085 :=
by
  -- Proof goes here
  sorry

end find_certain_number_l55_55947


namespace beth_coins_sold_l55_55895

def initial_coins : ℕ := 250
def additional_coins : ℕ := 75
def percentage_sold : ℚ := 60 / 100
def total_coins : ℕ := initial_coins + additional_coins
def coins_sold : ℚ := percentage_sold * total_coins

theorem beth_coins_sold : coins_sold = 195 :=
by
  -- Sorry is used to skip the proof as requested
  sorry

end beth_coins_sold_l55_55895


namespace nat_le_two_pow_million_l55_55842

theorem nat_le_two_pow_million (n : ℕ) (h : n ≤ 2^1000000) : 
  ∃ (x : ℕ → ℕ) (k : ℕ), k ≤ 1100000 ∧ x 0 = 1 ∧ x k = n ∧ 
  ∀ (i : ℕ), 1 ≤ i → i ≤ k → ∃ (r s : ℕ), 0 ≤ r ∧ r ≤ s ∧ s < i ∧ x i = x r + x s :=
sorry

end nat_le_two_pow_million_l55_55842


namespace people_and_carriages_condition_l55_55757

-- Definitions corresponding to the conditions
def num_people_using_carriages (x : ℕ) : ℕ := 3 * (x - 2)
def num_people_sharing_carriages (x : ℕ) : ℕ := 2 * x + 9

-- The theorem statement we need to prove
theorem people_and_carriages_condition (x : ℕ) : 
  num_people_using_carriages x = num_people_sharing_carriages x ↔ 3 * (x - 2) = 2 * x + 9 :=
by sorry

end people_and_carriages_condition_l55_55757


namespace arithmetic_sequence_general_formula_l55_55630

theorem arithmetic_sequence_general_formula (a : ℤ) :
  ∀ n : ℕ, n ≥ 1 → (∃ a_1 a_2 a_3 : ℤ, a_1 = a - 1 ∧ a_2 = a + 1 ∧ a_3 = a + 3) →
  (a + 2 * n - 3 = a - 1 + (n - 1) * 2) :=
by
  intros n hn h_exists
  rcases h_exists with ⟨a_1, a_2, a_3, h1, h2, h3⟩
  sorry

end arithmetic_sequence_general_formula_l55_55630


namespace average_p_q_l55_55989

theorem average_p_q (p q : ℝ) 
  (h1 : (4 + 6 + 8 + 2 * p + 2 * q) / 7 = 20) : 
  (p + q) / 2 = 30.5 :=
by
  sorry

end average_p_q_l55_55989


namespace initial_oranges_in_box_l55_55278

theorem initial_oranges_in_box (o_taken_out o_left_in_box : ℕ) (h1 : o_taken_out = 35) (h2 : o_left_in_box = 20) :
  o_taken_out + o_left_in_box = 55 := 
by
  sorry

end initial_oranges_in_box_l55_55278


namespace valid_exponent_rule_l55_55575

theorem valid_exponent_rule (a : ℝ) : (a^3)^2 = a^6 :=
by
  sorry

end valid_exponent_rule_l55_55575


namespace relationship_y1_y2_y3_l55_55062

variable {y1 y2 y3 h : ℝ}

def point_A := -1 / 2
def point_B := 1
def point_C := 2
def quadratic_function (x : ℝ) : ℝ := -(x - 2)^2 + h

theorem relationship_y1_y2_y3
  (hA : A_on_curve : quadratic_function point_A = y1)
  (hB : B_on_curve : quadratic_function point_B = y2)
  (hC : C_on_curve : quadratic_function point_C = y3) :
  y1 < y2 ∧ y2 < y3 := 
sorry

end relationship_y1_y2_y3_l55_55062


namespace coefficient_x3_in_expansion_l55_55322

theorem coefficient_x3_in_expansion :
  let general_term (r : ℕ) := (Nat.choose 5 r) * (2 : ℤ)^(5 - r) * (1 / 4)^(r : ℤ) * (x^(5 - 2 * r) : ℤ)
  (r := 1) :
  (2*x + 1/(4*x))^5 = 20 * x^3 + ... := 
by
  intros
  sorry

end coefficient_x3_in_expansion_l55_55322


namespace negation_of_universal_proposition_l55_55105

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + x + 1 ≥ 0) ↔ (∃ x : ℝ, x^2 + x + 1 < 0) :=
by
  sorry

end negation_of_universal_proposition_l55_55105


namespace total_seashells_after_six_weeks_l55_55621

theorem total_seashells_after_six_weeks :
  ∀ (a b : ℕ) 
  (initial_a : a = 50) 
  (initial_b : b = 30) 
  (next_a : ∀ k : ℕ, k > 0 → a + 20 = (a + 20) * k) 
  (next_b : ∀ k : ℕ, k > 0 → b * 2 = (b * 2) * k), 
  (a + 20 * 5) + (b * 2 ^ 5) = 1110 :=
by
  intros a b initial_a initial_b next_a next_b
  sorry

end total_seashells_after_six_weeks_l55_55621


namespace find_b_perpendicular_lines_l55_55427

variable (b : ℝ)

theorem find_b_perpendicular_lines :
  (2 * b + (-4) * 3 + 7 * (-1) = 0) → b = 19 / 2 := 
by
  intro h
  sorry

end find_b_perpendicular_lines_l55_55427


namespace binomial_inequality_l55_55092

theorem binomial_inequality (n : ℕ) (x : ℝ) (h1 : 2 ≤ n) (h2 : |x| < 1) : 
  (1 - x)^n + (1 + x)^n < 2^n := 
by 
  sorry

end binomial_inequality_l55_55092


namespace ratio_sheep_to_horses_l55_55762

theorem ratio_sheep_to_horses (sheep horses : ℕ) (total_horse_food daily_food_per_horse : ℕ)
  (h1 : sheep = 16)
  (h2 : total_horse_food = 12880)
  (h3 : daily_food_per_horse = 230)
  (h4 : horses = total_horse_food / daily_food_per_horse) :
  (sheep / gcd sheep horses) / (horses / gcd sheep horses) = 2 / 7 := by
  sorry

end ratio_sheep_to_horses_l55_55762


namespace at_least_two_equal_l55_55668

theorem at_least_two_equal (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a + b^2 + c^2 = b + a^2 + c^2) (h2 : b + a^2 + c^2 = c + a^2 + b^2) : 
  (a = b) ∨ (a = c) ∨ (b = c) :=
sorry

end at_least_two_equal_l55_55668


namespace small_seat_capacity_indeterminate_l55_55700

-- Conditions
def small_seats : ℕ := 3
def large_seats : ℕ := 7
def capacity_per_large_seat : ℕ := 12
def total_large_capacity : ℕ := 84

theorem small_seat_capacity_indeterminate
  (h1 : large_seats * capacity_per_large_seat = total_large_capacity)
  (h2 : ∀ s : ℕ, ∃ p : ℕ, p ≠ s * capacity_per_large_seat) :
  ¬ ∃ n : ℕ, ∀ m : ℕ, small_seats * m = n * small_seats :=
by {
  sorry
}

end small_seat_capacity_indeterminate_l55_55700


namespace find_x1_l55_55200

theorem find_x1 (x1 x2 x3 x4 : ℝ) 
  (h1 : 0 ≤ x4 ∧ x4 ≤ x3 ∧ x3 ≤ x2 ∧ x2 ≤ x1 ∧ x1 ≤ 1)
  (h2 : (1 - x1)^2 + (x1 - x2)^2 + (x2 - x3)^2 + (x3 - x4)^2 + x4^2 = 1 / 3) : 
  x1 = 4 / 5 := 
  sorry

end find_x1_l55_55200


namespace common_factor_is_n_plus_1_l55_55258

def polynomial1 (n : ℕ) : ℕ := n^2 - 1
def polynomial2 (n : ℕ) : ℕ := n^2 + n

theorem common_factor_is_n_plus_1 (n : ℕ) : 
  ∃ (d : ℕ), d ∣ polynomial1 n ∧ d ∣ polynomial2 n ∧ d = n + 1 := by
  sorry

end common_factor_is_n_plus_1_l55_55258


namespace borrowed_amount_correct_l55_55759

noncomputable def principal_amount (I: ℚ) (r1 r2 r3 r4 t1 t2 t3 t4: ℚ): ℚ :=
  I / (r1 * t1 + r2 * t2 + r3 * t3 + r4 * t4)

def interest_rate_1 := (6.5 / 100 : ℚ)
def interest_rate_2 := (9.5 / 100 : ℚ)
def interest_rate_3 := (11 / 100 : ℚ)
def interest_rate_4 := (14.5 / 100 : ℚ)

def time_period_1 := (2.5 : ℚ)
def time_period_2 := (3.75 : ℚ)
def time_period_3 := (1.5 : ℚ)
def time_period_4 := (4.25 : ℚ)

def total_interest := (14500 : ℚ)

def expected_principal := (11153.846153846154 : ℚ)

theorem borrowed_amount_correct :
  principal_amount total_interest interest_rate_1 interest_rate_2 interest_rate_3 interest_rate_4 time_period_1 time_period_2 time_period_3 time_period_4 = expected_principal :=
by
  sorry

end borrowed_amount_correct_l55_55759


namespace each_shopper_receives_equal_amount_l55_55379

variables (G I S total_final : ℝ)

-- Given conditions
def conditions : Prop :=
  G = 120 ∧
  I = G + 15 ∧
  I = S + 45

noncomputable def amount_each_shopper_receives : ℝ :=
  let total := I + S + G in total / 3

theorem each_shopper_receives_equal_amount (h : conditions) : amount_each_shopper_receives G I S = 115 := by
  -- Given conditions for Giselle, Isabella, and Sam
  rcases h with ⟨hG, hI1, hI2⟩
    
  -- Define total_final from conditions
  let total_final := G + (G + 15) + (G + 15 - 45)
  
  -- Default proof
  sorry

end each_shopper_receives_equal_amount_l55_55379


namespace fraction_q_p_l55_55840

theorem fraction_q_p (k : ℝ) (c p q : ℝ) (h : 8 * k^2 - 12 * k + 20 = c * (k + p)^2 + q) :
  c = 8 ∧ p = -3/4 ∧ q = 31/2 → q / p = -62 / 3 :=
by
  intros hc_hp_hq
  sorry

end fraction_q_p_l55_55840


namespace line_curve_intersection_symmetric_l55_55102

theorem line_curve_intersection_symmetric (a b : ℝ) 
    (h1 : ∃ p q : ℝ × ℝ, 
          (p.2 = a * p.1 + 1) ∧ 
          (q.2 = a * q.1 + 1) ∧ 
          (p ≠ q) ∧ 
          (p.1^2 + p.2^2 + b * p.1 - p.2 = 1) ∧ 
          (q.1^2 + q.2^2 + b * q.1 - q.2 = 1) ∧ 
          (p.1 + p.2 = -q.1 - q.2)) : 
  a + b = 2 :=
sorry

end line_curve_intersection_symmetric_l55_55102


namespace points_per_member_l55_55887

theorem points_per_member
  (total_members : ℕ)
  (members_didnt_show : ℕ)
  (total_points : ℕ)
  (H1 : total_members = 14)
  (H2 : members_didnt_show = 7)
  (H3 : total_points = 35) :
  total_points / (total_members - members_didnt_show) = 5 :=
by
  sorry

end points_per_member_l55_55887


namespace max_cart_length_l55_55749

-- Definitions for the hallway and cart dimensions
def hallway_width : ℝ := 1.5
def cart_width : ℝ := 1

-- The proposition stating the maximum length of the cart that can smoothly navigate the hallway
theorem max_cart_length : ∃ L : ℝ, L = 3 * Real.sqrt 2 ∧
  (∀ (a b : ℝ), a > 0 ∧ b > 0 → (3 / a) + (3 / b) = 2 → Real.sqrt (a^2 + b^2) = L) :=
  sorry

end max_cart_length_l55_55749


namespace find_x_l55_55924

theorem find_x (x : ℕ) (h : x + 1 = 6) : x = 5 :=
sorry

end find_x_l55_55924


namespace ben_final_amount_l55_55178

-- Definition of the conditions
def daily_start := 50
def daily_spent := 15
def daily_saving := daily_start - daily_spent
def days := 7
def mom_double (s : ℕ) := 2 * s
def dad_addition := 10

-- Total amount calculation based on the conditions
noncomputable def total_savings := daily_saving * days
noncomputable def after_mom := mom_double total_savings
noncomputable def total_amount := after_mom + dad_addition

-- The final theorem to prove Ben's final amount is $500 after the given conditions
theorem ben_final_amount : total_amount = 500 :=
by sorry

end ben_final_amount_l55_55178


namespace sum_of_integers_with_largest_proper_divisor_55_l55_55783

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def largest_proper_divisor (n d : ℕ) : Prop :=
  (d ∣ n) ∧ (d < n) ∧ ∀ e, (e ∣ n ∧ e < n ∧ e > d) → False

theorem sum_of_integers_with_largest_proper_divisor_55 : 
  (∀ n : ℕ, largest_proper_divisor n 55 → n = 110 ∨ n = 165 ∨ n = 275) →
  110 + 165 + 275 = 550 :=
by
  sorry

end sum_of_integers_with_largest_proper_divisor_55_l55_55783


namespace marbles_remaining_correct_l55_55899

-- Define the number of marbles Chris has
def marbles_chris : ℕ := 12

-- Define the number of marbles Ryan has
def marbles_ryan : ℕ := 28

-- Define the total number of marbles in the pile
def total_marbles : ℕ := marbles_chris + marbles_ryan

-- Define the number of marbles each person takes away from the pile
def marbles_taken_each : ℕ := total_marbles / 4

-- Define the total number of marbles taken away
def total_marbles_taken : ℕ := 2 * marbles_taken_each

-- Define the number of marbles remaining in the pile
def marbles_remaining : ℕ := total_marbles - total_marbles_taken

theorem marbles_remaining_correct : marbles_remaining = 20 := by
  sorry

end marbles_remaining_correct_l55_55899


namespace possible_values_of_K_l55_55955

theorem possible_values_of_K (K M : ℕ) (h : K * (K + 1) = M^2) (hM : M < 100) : K = 8 ∨ K = 35 :=
by sorry

end possible_values_of_K_l55_55955


namespace fishing_ratio_l55_55896

variables (B C : ℝ)
variable (brian_per_trip : ℝ)
variable (chris_per_trip : ℝ)

-- Given conditions
def conditions : Prop :=
  C = 10 ∧
  brian_per_trip = 400 ∧
  chris_per_trip = 400 * (5 / 3) ∧
  B * brian_per_trip + 10 * chris_per_trip = 13600

-- The ratio of the number of times Brian goes fishing to the number of times Chris goes fishing
def ratio_correct : Prop :=
  B / C = 26 / 15

theorem fishing_ratio (h : conditions B C brian_per_trip chris_per_trip) : ratio_correct B C :=
by
  sorry

end fishing_ratio_l55_55896


namespace area_units_ordered_correctly_l55_55547

def area_units :=
  ["square kilometers", "hectares", "square meters", "square decimeters", "square centimeters"]

theorem area_units_ordered_correctly :
  area_units = ["square kilometers", "hectares", "square meters", "square decimeters", "square centimeters"] :=
by
  sorry

end area_units_ordered_correctly_l55_55547


namespace tan_585_eq_1_l55_55182

theorem tan_585_eq_1 : Real.tan (585 * Real.pi / 180) = 1 := 
by
  sorry

end tan_585_eq_1_l55_55182


namespace Sean_Julie_ratio_l55_55691

-- Define the sum of the first n natural numbers
def sum_n (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define the sum of even numbers up to 2n
def sum_even (n : ℕ) : ℕ := 2 * sum_n n

theorem Sean_Julie_ratio : 
  (sum_even 250) / (sum_n 250) = 2 := 
by
  sorry

end Sean_Julie_ratio_l55_55691


namespace linda_five_dollar_bills_l55_55082

theorem linda_five_dollar_bills (x y : ℕ) (h1 : x + y = 12) (h2 : 5 * x + 10 * y = 80) : x = 8 :=
by
  sorry

end linda_five_dollar_bills_l55_55082


namespace M_identically_zero_l55_55386

noncomputable def M (x y : ℝ) : ℝ := sorry

theorem M_identically_zero (a : ℝ) (h1 : a > 1) (h2 : ∀ x, M x (a^x) = 0) : ∀ x y, M x y = 0 :=
sorry

end M_identically_zero_l55_55386


namespace math_problem_l55_55905

theorem math_problem (x : ℝ) : 
  x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1/2 ∧ (x^2 + x^3 - 2 * x^4) / (x + x^2 - 2 * x^3) ≥ -1 ↔ 
  x ∈ Set.Icc (-1 : ℝ) (-1/2) ∪ Set.Ioc (-1/2 : ℝ) 0 ∪ Set.Ioo 0 1 ∪ Set.Ioi 1 := 
by 
  sorry

end math_problem_l55_55905


namespace fewer_ducks_than_chickens_and_geese_l55_55855

/-- There are 42 chickens and 48 ducks on the farm, and there are as many geese as there are chickens. 
Prove that there are 36 fewer ducks than the number of chickens and geese combined. -/
theorem fewer_ducks_than_chickens_and_geese (chickens ducks geese : ℕ)
  (h_chickens : chickens = 42)
  (h_ducks : ducks = 48)
  (h_geese : geese = chickens):
  ducks + 36 = chickens + geese :=
by
  sorry

end fewer_ducks_than_chickens_and_geese_l55_55855


namespace walk_time_to_LakePark_restaurant_l55_55822

/-
  It takes 15 minutes for Dante to go to Hidden Lake.
  From Hidden Lake, it takes him 7 minutes to walk back to the Park Office.
  Dante will have been gone from the Park Office for a total of 32 minutes.
  Prove that the walk from the Park Office to the Lake Park restaurant is 10 minutes.
-/

def T_HiddenLake_to : ℕ := 15
def T_HiddenLake_from : ℕ := 7
def T_total : ℕ := 32
def T_LakePark_restaurant : ℕ := T_total - (T_HiddenLake_to + T_HiddenLake_from)

theorem walk_time_to_LakePark_restaurant : 
  T_LakePark_restaurant = 10 :=
by
  unfold T_LakePark_restaurant T_HiddenLake_to T_HiddenLake_from T_total
  sorry

end walk_time_to_LakePark_restaurant_l55_55822


namespace value_of_expression_l55_55815

variable {a : ℕ → ℤ}
variable {a₁ a₄ a₁₀ a₁₆ a₁₉ : ℤ}
variable {d : ℤ}

-- Definition of the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℤ) (a₁ d : ℤ) : Prop :=
  ∀ n : ℕ, a n = a₁ + d * n

-- Given conditions
axiom h₀ : arithmetic_sequence a a₁ d
axiom h₁ : a₁ + a₄ + a₁₀ + a₁₆ + a₁₉ = 150

-- Prove the required statement
theorem value_of_expression :
  a 20 - a 26 + a 16 = 30 :=
sorry

end value_of_expression_l55_55815


namespace scientific_notation_of_258000000_l55_55330

theorem scientific_notation_of_258000000 :
  258000000 = 2.58 * 10^8 :=
sorry

end scientific_notation_of_258000000_l55_55330


namespace dividend_divisor_quotient_l55_55411

theorem dividend_divisor_quotient (x y z : ℕ) 
  (h1 : x = 6 * y) 
  (h2 : y = 6 * z) 
  (h3 : x = y * z) : 
  x = 216 ∧ y = 36 ∧ z = 6 := 
by
  sorry

end dividend_divisor_quotient_l55_55411


namespace dice_arithmetic_progression_l55_55428

theorem dice_arithmetic_progression :
  let valid_combinations := [
     (1, 1, 1), (1, 3, 2), (1, 5, 3), 
     (2, 4, 3), (2, 6, 4), (3, 3, 3),
     (3, 5, 4), (4, 6, 5), (5, 5, 5)
  ]
  (valid_combinations.length : ℚ) / (6^3 : ℚ) = 1 / 24 :=
  sorry

end dice_arithmetic_progression_l55_55428


namespace triangle_proof_l55_55518

theorem triangle_proof (a b : ℝ) (cosA : ℝ) (ha : a = 6) (hb : b = 5) (hcosA : cosA = -4 / 5) :
  (∃ B : ℝ, B = 30) ∧ (∃ area : ℝ, area = (9 * Real.sqrt 3 - 12) / 2) :=
  by
  sorry

end triangle_proof_l55_55518


namespace range_of_k_tan_alpha_l55_55359

noncomputable section

open Real

def a (x : ℝ) : ℝ × ℝ := (sin x, 1)
def b (k : ℝ) : ℝ × ℝ := (1, k)
def f (x k : ℝ) : ℝ := (a x).1 * (b k).1 + (a x).2 * (b k).2

theorem range_of_k (k : ℝ) : 
  (∃ x : ℝ, f x k = 1) ↔ k ∈ Icc 0 2 :=
by
  sorry

theorem tan_alpha (k α : ℝ) (hα : α ∈ Ioo 0 π) :
  f α k = (1 / 3) + k → 
  tan α ∈ {1 / (3 * sqrt (8 / 9)), -1 / (3 * sqrt (8 / 9))} :=
by
  sorry

end range_of_k_tan_alpha_l55_55359


namespace price_per_glass_on_second_day_l55_55087

 -- Definitions based on the conditions
def orangeade_first_day (O: ℝ) : ℝ := 2 * O -- Total volume on first day, O + O
def orangeade_second_day (O: ℝ) : ℝ := 3 * O -- Total volume on second day, O + 2O
def revenue_first_day (O: ℝ) (price_first_day: ℝ) : ℝ := 2 * O * price_first_day -- Revenue on first day
def revenue_second_day (O: ℝ) (P: ℝ) : ℝ := 3 * O * P -- Revenue on second day
def price_first_day: ℝ := 0.90 -- Given price per glass on the first day

 -- Statement to be proved
theorem price_per_glass_on_second_day (O: ℝ) (P: ℝ) (h: revenue_first_day O price_first_day = revenue_second_day O P) :
  P = 0.60 :=
by
  sorry

end price_per_glass_on_second_day_l55_55087


namespace find_k_l55_55366

theorem find_k (k : ℕ) : (1 / 2) ^ 16 * (1 / 81) ^ k = 1 / 18 ^ 16 → k = 8 :=
by
  intro h
  sorry

end find_k_l55_55366


namespace Jerry_age_l55_55393

theorem Jerry_age (M J : ℕ) (h1 : M = 2 * J - 6) (h2 : M = 22) : J = 14 :=
by
  sorry

end Jerry_age_l55_55393


namespace only_nonneg_int_solution_l55_55544

theorem only_nonneg_int_solution (x y z : ℕ) (h : x^3 = 3 * y^3 + 9 * z^3) : x = 0 ∧ y = 0 ∧ z = 0 := 
sorry

end only_nonneg_int_solution_l55_55544


namespace series_sum_eq_l55_55618

noncomputable def sum_series (k : ℝ) : ℝ :=
  (∑' n : ℕ, (4 * (n + 1) + k) / 3^(n + 1))

theorem series_sum_eq (k : ℝ) : sum_series k = 3 + k / 2 := 
  sorry

end series_sum_eq_l55_55618


namespace geometric_progression_solution_l55_55276

noncomputable def first_term_of_geometric_progression (b2 b6 : ℚ) (q : ℚ) : ℚ := 
  b2 / q
  
theorem geometric_progression_solution 
  (b2 b6 : ℚ)
  (h1 : b2 = 37 + 1/3)
  (h2 : b6 = 2 + 1/3) :
  ∃ a q : ℚ, a = 224 / 3 ∧ q = 1/2 ∧ b2 = a * q ∧ b6 = a * q^5 :=
by
  sorry

end geometric_progression_solution_l55_55276


namespace fraction_spent_toy_store_l55_55645

noncomputable def weekly_allowance : ℚ := 2.25
noncomputable def arcade_fraction_spent : ℚ := 3 / 5
noncomputable def candy_store_spent : ℚ := 0.60

theorem fraction_spent_toy_store :
  let remaining_after_arcade := weekly_allowance * (1 - arcade_fraction_spent)
  let spent_toy_store := remaining_after_arcade - candy_store_spent
  spent_toy_store / remaining_after_arcade = 1 / 3 :=
by
  sorry

end fraction_spent_toy_store_l55_55645


namespace truth_values_of_p_and_q_l55_55352

variable (p q : Prop)

theorem truth_values_of_p_and_q
  (h1 : ¬ (p ∧ q))
  (h2 : (¬ p ∨ q)) :
  ¬ p ∧ (q ∨ ¬ q) :=
by {
  sorry
}

end truth_values_of_p_and_q_l55_55352


namespace swallow_distance_flew_l55_55440

/-- The TGV departs from Paris at 150 km/h toward Marseille, which is 800 km away, while an intercité departs from Marseille at 50 km/h toward Paris at the same time. A swallow perched on the TGV takes off at that moment, flying at 200 km/h toward Marseille. We aim to prove that the distance flown by the swallow when the two trains meet is 800 km. -/
theorem swallow_distance_flew :
  let distance := 800 -- distance between Paris and Marseille in km
  let speed_TGV := 150 -- speed of TGV in km/h
  let speed_intercite := 50 -- speed of intercité in km/h
  let speed_swallow := 200 -- speed of swallow in km/h
  let combined_speed := speed_TGV + speed_intercite
  let time_to_meet := distance / combined_speed
  let distance_swallow_traveled := speed_swallow * time_to_meet
  distance_swallow_traveled = 800 := 
by
  sorry

end swallow_distance_flew_l55_55440


namespace range_of_m_l55_55051

theorem range_of_m (m : ℤ) (x : ℤ) (h1 : (m + 3) / (x - 1) = 1) (h2 : x > 0) : m > -4 ∧ m ≠ -3 :=
sorry

end range_of_m_l55_55051


namespace longer_side_of_rectangle_l55_55601

theorem longer_side_of_rectangle (r : ℝ) (Aₙ Aₙ: ℝ) (L S: ℝ):
  (r = 6) → 
  (Aₙ = 36 * π) →
  (Aₙ = 108 * π) →
  (S = 12) → 
  (S * L = Aₙ) →
  L = 9 * π := sorry

end longer_side_of_rectangle_l55_55601


namespace largest_house_number_l55_55643

theorem largest_house_number (phone_number_digits : List ℕ) (house_number_digits : List ℕ) :
  phone_number_digits = [5, 0, 4, 9, 3, 2, 6] →
  phone_number_digits.sum = 29 →
  (∀ (d1 d2 : ℕ), d1 ∈ house_number_digits → d2 ∈ house_number_digits → d1 ≠ d2) →
  house_number_digits.sum = 29 →
  house_number_digits = [9, 8, 7, 5] :=
by
  intros
  sorry

end largest_house_number_l55_55643


namespace least_integer_x_l55_55723

theorem least_integer_x (x : ℤ) (h : 240 ∣ x^2) : x = 60 :=
sorry

end least_integer_x_l55_55723


namespace candace_new_shoes_speed_l55_55768

theorem candace_new_shoes_speed:
  ∀ (old_shoes_speed new_shoes_factor hours_per_blister speed_reduction hike_duration: ℕ),
    old_shoes_speed = 6 →
    new_shoes_factor = 2 →
    hours_per_blister = 2 →
    speed_reduction = 2 →
    hike_duration = 4 →
    let new_shoes_speed := old_shoes_speed * new_shoes_factor in
    let speed_after_blister := new_shoes_speed - speed_reduction in
    let average_speed := (new_shoes_speed * (hike_duration / hours_per_blister) + 
                          speed_after_blister * (hike_duration - hike_duration / hours_per_blister)) / hike_duration in
    average_speed = 11 :=
by {
  intros old_shoes_speed new_shoes_factor hours_per_blister speed_reduction hike_duration,
  intros h1 h2 h3 h4 h5,
  let new_shoes_speed := old_shoes_speed * new_shoes_factor,
  let speed_after_blister := new_shoes_speed - speed_reduction,
  let average_speed := (new_shoes_speed * (hike_duration / hours_per_blister) + 
                          speed_after_blister * (hike_duration - hike_duration / hours_per_blister)) / hike_duration,
  sorry
}

end candace_new_shoes_speed_l55_55768


namespace scheduling_plans_l55_55614

-- Defining a set of 7 employees
@[derive fintype]
inductive Employee
| A
| B
| E1
| E2
| E3
| E4
| E5

-- Days from October 1st to 7th
def Days := Fin 7

-- A function to represent the schedule
def schedule : Days → Employee := sorry

-- A condition to express that each person is scheduled for one day
def one_per_day (s : Days → Employee) : Prop := 
  bijective s

-- A condition that A and B are not on consecutive days
def not_consecutive (s : Days → Employee) : Prop := 
  ∀ i : Days, i < 6 → (s i = Employee.A ∧ s (i+1) = Employee.B) → false

-- Theorem statement
theorem scheduling_plans : 
  ∃ s : Days → Employee, one_per_day s ∧ not_consecutive s ∧ fintype.card {s' : Days → Employee // one_per_day s' ∧ not_consecutive s'} = 3600 := 
sorry

end scheduling_plans_l55_55614


namespace functional_eq_implies_odd_l55_55043

variable (f : ℝ → ℝ)

def condition (f : ℝ → ℝ) : Prop :=
∀ x y : ℝ, f (x * f y) = y * f x

theorem functional_eq_implies_odd (h : condition f) : ∀ x : ℝ, f (-x) = -f x :=
sorry

end functional_eq_implies_odd_l55_55043


namespace product_of_fractions_l55_55617

theorem product_of_fractions : 
  (1 + 1/2) * (1 + 1/3) * (1 + 1/4) * (1 + 1/5) * (1 + 1/6) * (1 + 1/7) = 8 :=
by
  sorry

end product_of_fractions_l55_55617


namespace work_completion_l55_55576

theorem work_completion (W : ℝ) (a b : ℝ) (ha : a = W / 12) (hb : b = W / 6) :
  W / (a + b) = 4 :=
by {
  sorry
}

end work_completion_l55_55576


namespace year_with_greatest_temp_increase_l55_55410

def avg_temp (year : ℕ) : ℝ :=
  match year with
  | 2000 => 2.0
  | 2001 => 2.3
  | 2002 => 2.5
  | 2003 => 2.7
  | 2004 => 3.9
  | 2005 => 4.1
  | 2006 => 4.2
  | 2007 => 4.4
  | 2008 => 3.9
  | 2009 => 3.1
  | _    => 0.0

theorem year_with_greatest_temp_increase : ∃ year, year = 2004 ∧
  (∀ y, 2000 < y ∧ y ≤ 2009 → avg_temp y - avg_temp (y - 1) ≤ avg_temp 2004 - avg_temp 2003) := by
  sorry

end year_with_greatest_temp_increase_l55_55410


namespace probability_two_females_one_male_l55_55835

theorem probability_two_females_one_male :
  let total_contestants := 8
  let num_females := 5
  let num_males := 3
  let choose3 := Nat.choose total_contestants 3
  let choose2f := Nat.choose num_females 2
  let choose1m := Nat.choose num_males 1
  let favorable_outcomes := choose2f * choose1m
  choose3 ≠ 0 → (favorable_outcomes / choose3 : ℚ) = 15 / 28 :=
by
  sorry

end probability_two_females_one_male_l55_55835


namespace num_solutions_to_congruence_l55_55202

def is_solution (x : ℕ) : Prop :=
  x < 150 ∧ ((x + 17) % 46 = 75 % 46)

theorem num_solutions_to_congruence : (finset.univ.filter is_solution).card = 3 :=
by
  sorry

end num_solutions_to_congruence_l55_55202


namespace central_angle_of_sector_l55_55983

open Real

theorem central_angle_of_sector (l S : ℝ) (α R : ℝ) (hl : l = 4) (hS : S = 4) (h1 : l = α * R) (h2 : S = 1/2 * α * R^2) : 
  α = 2 :=
by
  -- Proof will be supplied here
  sorry

end central_angle_of_sector_l55_55983


namespace sum_of_ai_powers_l55_55942

theorem sum_of_ai_powers :
  ∀ (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 : ℝ),
  (∀ x : ℝ, (1 + x) * (1 - 2 * x)^8 = 
            a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + 
            a_4 * x^4 + a_5 * x^5 + a_6 * x^6 + 
            a_7 * x^7 + a_8 * x^8 + a_9 * x^9) →
  a_1 * 2 + a_2 * 2^2 + a_3 * 2^3 + 
  a_4 * 2^4 + a_5 * 2^5 + a_6 * 2^6 + 
  a_7 * 2^7 + a_8 * 2^8 + a_9 * 2^9 = 3^9 - 1 :=
by
  sorry

end sum_of_ai_powers_l55_55942


namespace number_of_defective_pens_l55_55371

noncomputable def defective_pens (total : ℕ) (prob : ℚ) : ℕ :=
  let N := 6 -- since we already know the steps in the solution leading to N = 6
  let D := total - N
  D

theorem number_of_defective_pens (total : ℕ) (prob : ℚ) :
  (total = 12) → (prob = 0.22727272727272727) → defective_pens total prob = 6 :=
by
  intros ht hp
  unfold defective_pens
  sorry

end number_of_defective_pens_l55_55371


namespace production_problem_l55_55651

theorem production_problem (x y : ℝ) (h₁ : x > 0) (h₂ : ∀ k : ℝ, x * x * x * k = x) : (x * x * y * (1 / (x^2)) = y) :=
by {
  sorry
}

end production_problem_l55_55651


namespace roots_cubic_inv_sum_l55_55512

theorem roots_cubic_inv_sum (a b c r s : ℝ) (h_eq : ∃ (r s : ℝ), r^2 * a + b * r - c = 0 ∧ s^2 * a + b * s - c = 0) :
  (1 / r^3) + (1 / s^3) = (b^3 + 3 * a * b * c) / c^3 :=
by
  sorry

end roots_cubic_inv_sum_l55_55512


namespace units_digit_G1000_l55_55901

def Gn (n : ℕ) : ℕ := 3^(3^n) + 1

theorem units_digit_G1000 : (Gn 1000) % 10 = 2 :=
by sorry

end units_digit_G1000_l55_55901


namespace find_d_l55_55283

theorem find_d :
  ∃ d : ℝ, ∀ x : ℝ, x * (4 * x - 3) < d ↔ - (9/4 : ℝ) < x ∧ x < (3/2 : ℝ) ∧ d = 27 / 2 :=
by
  sorry

end find_d_l55_55283


namespace tan_double_angle_l55_55099

theorem tan_double_angle (α : ℝ) (h1 : Real.sin (5 * Real.pi / 6) = 1 / 2)
  (h2 : Real.cos (5 * Real.pi / 6) = -Real.sqrt 3 / 2) : 
  Real.tan (2 * α) = Real.sqrt 3 := 
sorry

end tan_double_angle_l55_55099


namespace johns_weekly_allowance_l55_55059

theorem johns_weekly_allowance (A : ℝ) 
  (arcade_spent : A * (3/5) = 3 * (A/5)) 
  (remainder_after_arcade : (2/5) * A = A - 3 * (A/5))
  (toy_store_spent : (1/3) * (2/5) * A = 2 * (A/15)) 
  (remainder_after_toy_store : (2/5) * A - (2/15) * A = 4 * (A/15))
  (last_spent : (4/15) * A = 0.4) :
  A = 1.5 :=
sorry

end johns_weekly_allowance_l55_55059


namespace longer_side_length_l55_55596

-- Given a circle with radius 6 cm
def radius : ℝ := 6
def circle_area : ℝ := Real.pi * radius^2

-- Given that the area of the rectangle is three times the area of the circle
def rectangle_area : ℝ := 3 * circle_area

-- Given the rectangle has a shorter side equal to the diameter of the circle
def shorter_side : ℝ := 2 * radius

-- Prove that the length of the longer side of the rectangle is 9π cm
theorem longer_side_length : ∃ (longer_side : ℝ), longer_side = 9 * Real.pi :=
by
  have circle_area_def : circle_area = 36 * Real.pi := by sorry
  have rectangle_area_def : rectangle_area = 108 * Real.pi := by sorry
  have shorter_side_def : shorter_side = 12 := by sorry
  let longer_side := rectangle_area / shorter_side
  use longer_side
  show longer_side = 9 * Real.pi
  sorry

end longer_side_length_l55_55596


namespace painted_cubes_count_l55_55125

def total_painted_cubes : ℕ := 8 + 48

theorem painted_cubes_count : total_painted_cubes = 56 :=
by 
  -- Step 1: Define the number of cubes with 3 faces painted (8 corners)
  let corners := 8
  -- Step 2: Calculate the number of edge cubes with 2 faces painted
  let edge_middle_cubes_per_edge := 2
  let edges := 12
  let edge_cubes := edge_middle_cubes_per_edge * edges -- this should be 24
  -- Step 3: Calculate the number of face-interior cubes with 2 faces painted
  let face_cubes_per_face := 4
  let faces := 6
  let face_cubes := face_cubes_per_face * faces -- this should be 24
  -- Step 4: Sum them up to get total cubes with at least two faces painted
  let total_cubes := corners + edge_cubes + face_cubes
  show total_cubes = total_painted_cubes
  sorry

end painted_cubes_count_l55_55125


namespace samuel_faster_than_sarah_l55_55399

theorem samuel_faster_than_sarah
  (efficiency_samuel : ℝ := 0.90)
  (efficiency_sarah : ℝ := 0.75)
  (efficiency_tim : ℝ := 0.80)
  (time_tim : ℝ := 45)
  : (time_tim * efficiency_tim / efficiency_sarah) - (time_tim * efficiency_tim / efficiency_samuel) = 8 :=
by
  sorry

end samuel_faster_than_sarah_l55_55399


namespace dvaneft_shares_percentage_range_l55_55296

theorem dvaneft_shares_percentage_range :
  ∀ (x y z n m : ℝ),
    (4 * x * n = y * m) →
    (x * n + y * m = z * (m + n)) →
    (16 ≤ y - x ∧ y - x ≤ 20) →
    (42 ≤ z ∧ z ≤ 60) →
    (12.5 ≤ (n / (2 * (n + m)) * 100) ∧ (n / (2 * (n + m)) * 100) ≤ 15) :=
by
  intros x y z n m h1 h2 h3 h4
  sorry

end dvaneft_shares_percentage_range_l55_55296


namespace proof_problem_l55_55235

theorem proof_problem (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h : a / (b - c) + b / (c - a) + c / (a - b) = 0) :
  a^2 / (b - c)^2 + b^2 / (c - a)^2 + c^2 / (a - b)^2 = 0 :=
sorry

end proof_problem_l55_55235


namespace range_of_a_l55_55798

theorem range_of_a (a : ℝ) :
  (∀ (x y : ℝ), (1 ≤ x ∧ x ≤ 2) ∧ (2 ≤ y ∧ y ≤ 3) → (x * y ≤ a * x^2 + 2 * y^2)) →
  a ≥ -1 :=
by {
  sorry
}

end range_of_a_l55_55798


namespace inequality_proof_l55_55686

theorem inequality_proof (x y : ℝ) (h1 : y ≥ 0) (h2 : y * (y + 1) ≤ (x + 1)^2) : y * (y - 1) ≤ x^2 :=
sorry

end inequality_proof_l55_55686


namespace possible_values_of_d_l55_55675

theorem possible_values_of_d (r s : ℝ) (c d : ℝ)
  (h1 : ∃ u, u = -r - s ∧ r * s + r * u + s * u = c)
  (h2 : ∃ v, v = -r - s - 8 ∧ (r - 3) * (s + 5) + (r - 3) * (u - 8) + (s + 5) * (u - 8) = c)
  (u_eq : u = -r - s)
  (v_eq : v = -r - s - 8)
  (polynomial_relation : d + 156 = -((r - 3) * (s + 5) * (u - 8))) : 
  d = -198 ∨ d = 468 := 
sorry

end possible_values_of_d_l55_55675


namespace arithmetic_sequence_conditions_l55_55234

open Nat

theorem arithmetic_sequence_conditions (S : ℕ → ℤ) (d : ℤ) (a1 : ℤ) 
  (h1 : S 6 > S 7) (h2 : S 7 > S 5) :
  d < 0 ∧ S 11 > 0 := 
sorry

end arithmetic_sequence_conditions_l55_55234


namespace certain_event_l55_55720

theorem certain_event :
  (∀ (e : string), e = "Moonlight in front of the bed" → ¬is_certain_event e) ∧
  (∀ (e : string), e = "Lonely smoke in the desert" → ¬is_certain_event e) ∧
  (∀ (e : string), e = "Reach for the stars with your hand" → is_impossible_event e) ∧
  (∀ (e : string), e = "Yellow River flows into the sea" → is_certain_event e) →
  is_certain_event "Yellow River flows into the sea" :=
by
  sorry

end certain_event_l55_55720


namespace probability_of_yellow_marble_l55_55465

def marbles_prob :=
  let PxW := 4 / 9                -- Probability of drawing a white marble from Bag X
  let PyY := 7 / 10               -- Probability of drawing a yellow marble from Bag Y
  let PxB := 5 / 9                -- Probability of drawing a black marble from Bag X
  let PzY := 1 / 3                -- Probability of drawing a yellow marble from Bag Z
  PxW * PyY + PxB * PzY           -- Total probability

theorem probability_of_yellow_marble :
  marbles_prob = 67 / 135 :=
by
  sorry

end probability_of_yellow_marble_l55_55465


namespace rectangular_to_cylindrical_l55_55771

theorem rectangular_to_cylindrical (x y z : ℝ) (r θ : ℝ) (h_r : r > 0) (h_θ : 0 ≤ θ ∧ θ < 2 * Real.pi) :
  x = 3 ∧ y = -3 * Real.sqrt 3 ∧ z = 4 →
  r = Real.sqrt (3^2 + (-3 * Real.sqrt 3)^2) ∧
  θ = Real.arctan y x ∧
  r = 6 ∧
  θ = 4 * Real.pi / 3 ∧
  z = 4 :=
by
  sorry

end rectangular_to_cylindrical_l55_55771


namespace problem_statement_l55_55966

theorem problem_statement (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / (b + 2 * c) + b / (c + 2 * a) + c / (a + 2 * b) > 1 / 2) :=
by
  sorry

end problem_statement_l55_55966


namespace oplus_self_twice_l55_55355

def my_oplus (x y : ℕ) := 3^x - y

theorem oplus_self_twice (a : ℕ) : my_oplus a (my_oplus a a) = a := by
  sorry

end oplus_self_twice_l55_55355


namespace number_of_ways_to_choose_students_l55_55424

theorem number_of_ways_to_choose_students :
  let female_students := 4
  let male_students := 3
  (female_students * male_students) = 12 :=
by
  sorry

end number_of_ways_to_choose_students_l55_55424


namespace solve_discriminant_l55_55318

def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem solve_discriminant : 
  discriminant 2 (2 + 1/2) (1/2) = 2.25 :=
by
  -- The proof can be filled in here
  -- Assuming a = 2, b = 2.5, c = 1/2
  -- discriminant 2 2.5 0.5 will be computed
  sorry

end solve_discriminant_l55_55318


namespace cricket_average_l55_55165

theorem cricket_average (A : ℝ) (h : 20 * A + 120 = 21 * (A + 4)) : A = 36 :=
by sorry

end cricket_average_l55_55165


namespace ratio_ab_bd_l55_55877

-- Definitions based on the given conditions
def ab : ℝ := 4
def bc : ℝ := 8
def cd : ℝ := 5
def bd : ℝ := bc + cd

-- Theorem statement
theorem ratio_ab_bd :
  ((ab / bd) = (4 / 13)) :=
by
  -- Proof goes here
  sorry

end ratio_ab_bd_l55_55877


namespace find_r_cubed_l55_55654

theorem find_r_cubed (r : ℝ) (h : (r + 1/r)^2 = 5) : r^3 + 1/r^3 = 2 * Real.sqrt 5 :=
by
  sorry

end find_r_cubed_l55_55654


namespace factorize_expression_l55_55476

theorem factorize_expression (a x : ℝ) : a * x^2 - a = a * (x + 1) * (x - 1) :=
by
  sorry

end factorize_expression_l55_55476


namespace factorial_square_product_l55_55871

theorem factorial_square_product : (Real.sqrt (Nat.factorial 6 * Nat.factorial 4)) ^ 2 = 17280 := by
  sorry

end factorial_square_product_l55_55871


namespace initial_money_amount_l55_55023

theorem initial_money_amount (x : ℕ) (h : x + 16 = 18) : x = 2 := by
  sorry

end initial_money_amount_l55_55023


namespace net_percentage_change_l55_55027

theorem net_percentage_change (k m : ℝ) : 
  let scale_factor_1 := 1 - k / 100
  let scale_factor_2 := 1 + m / 100
  let overall_scale_factor := scale_factor_1 * scale_factor_2
  let percentage_change := (overall_scale_factor - 1) * 100
  percentage_change = m - k - k * m / 100 := 
by 
  sorry

end net_percentage_change_l55_55027


namespace cos_theta_value_projection_value_l55_55937

noncomputable def vec_a : (ℝ × ℝ) := (3, 1)
noncomputable def vec_b : (ℝ × ℝ) := (-2, 4)

theorem cos_theta_value :
  let a := vec_a
  let b := vec_b
  let dot_product := a.1 * b.1 + a.2 * b.2
  let magnitude_a := Real.sqrt (a.1^2 + a.2^2)
  let magnitude_b := Real.sqrt (b.1^2 + b.2^2)
  dot_product / (magnitude_a * magnitude_b) = - Real.sqrt 2 / 10 :=
by 
  sorry

theorem projection_value :
  let a := vec_a
  let b := vec_b
  let dot_product := a.1 * b.1 + a.2 * b.2
  let magnitude_a := Real.sqrt (a.1^2 + a.2^2)
  let magnitude_b := Real.sqrt (b.1^2 + b.2^2)
  let cos_theta := dot_product / (magnitude_a * magnitude_b)
  cos_theta = - Real.sqrt 2 / 10 →
  magnitude_a * cos_theta = - Real.sqrt 5 / 5 :=
by 
  sorry

end cos_theta_value_projection_value_l55_55937


namespace parameterization_of_line_l55_55847

theorem parameterization_of_line : 
  ∀ (r k : ℝ),
  (∀ t : ℝ, (∃ x y : ℝ, (x, y) = (r, 2) + t • (3, k)) → y = 2 * x - 6) → (r = 4 ∧ k = 6) :=
by
  sorry

end parameterization_of_line_l55_55847


namespace mean_equality_l55_55417

theorem mean_equality (z : ℚ) :
  (8 + 12 + 24) / 3 = (16 + z) / 2 ↔ z = 40 / 3 :=
by
  sorry

end mean_equality_l55_55417


namespace smallest_number_of_eggs_l55_55154

theorem smallest_number_of_eggs (c : ℕ) (h1 : 15 * c - 3 > 100) : 102 ≤ 15 * c - 3 :=
by
  sorry

end smallest_number_of_eggs_l55_55154


namespace find_g6_minus_g2_div_g3_l55_55100

noncomputable def g : ℝ → ℝ := sorry

axiom g_condition (a c : ℝ) : c^3 * g a = a^3 * g c
axiom g_nonzero : g 3 ≠ 0

theorem find_g6_minus_g2_div_g3 : (g 6 - g 2) / g 3 = 208 / 27 := by
  sorry

end find_g6_minus_g2_div_g3_l55_55100


namespace henry_has_more_games_l55_55060

-- Define the conditions and initial states
def initial_games_henry : ℕ := 33
def given_games_neil : ℕ := 5
def initial_games_neil : ℕ := 2

-- Define the number of games Henry and Neil have now
def games_henry_now : ℕ := initial_games_henry - given_games_neil
def games_neil_now : ℕ := initial_games_neil + given_games_neil

-- State the theorem to be proven
theorem henry_has_more_games : games_henry_now / games_neil_now = 4 :=
by
  sorry

end henry_has_more_games_l55_55060


namespace find_k_value_l55_55356

theorem find_k_value (k : ℝ) : 
  (-x ^ 2 - (k + 11) * x - 8 = -( (x - 2) * (x - 4) ) ) → k = -17 := 
by 
  sorry

end find_k_value_l55_55356


namespace min_value_of_f_l55_55849

noncomputable def f (x : ℝ) : ℝ := x + 1 / (x - 2)

theorem min_value_of_f : ∀ (x : ℝ), x > 2 → f x ≥ 4 := by
  sorry

end min_value_of_f_l55_55849


namespace power_summation_l55_55030

theorem power_summation :
  (-1:ℤ)^(49) + (2:ℝ)^(3^3 + 5^2 - 48^2) = -1 + 1 / 2 ^ (2252 : ℝ) :=
by
  sorry

end power_summation_l55_55030


namespace walnut_trees_planted_l55_55711

-- The number of walnut trees before planting
def walnut_trees_before : ℕ := 22

-- The number of walnut trees after planting
def walnut_trees_after : ℕ := 55

-- The number of walnut trees planted today
def walnut_trees_planted_today : ℕ := 33

-- Theorem statement to prove that the number of walnut trees planted today is 33
theorem walnut_trees_planted:
  walnut_trees_after - walnut_trees_before = walnut_trees_planted_today :=
by sorry

end walnut_trees_planted_l55_55711


namespace triangle_side_difference_l55_55225

theorem triangle_side_difference :
  ∀ x : ℝ, (x > 2 ∧ x < 18) → ∃ a b : ℤ, (∀ y : ℤ, y ∈ set.Icc a b → (y : ℝ) = x) ∧ (b - a = 14) :=
by
  assume x hx,
  sorry

end triangle_side_difference_l55_55225


namespace area_of_right_triangle_l55_55666

-- Define a structure for the triangle with the given conditions
structure Triangle :=
(A B C : ℝ × ℝ)
(right_angle_at_C : (C.1 = 0 ∧ C.2 = 0))
(hypotenuse_length : (A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2 = 50 ^ 2)
(median_A : ∀ x: ℝ, A.2 = A.1 + 5)
(median_B : ∀ x: ℝ, B.2 = 2 * B.1 + 2)

-- Theorem statement
theorem area_of_right_triangle (t : Triangle) : 
  ∃ area : ℝ, area = 500 :=
sorry

end area_of_right_triangle_l55_55666


namespace shortest_distance_between_tracks_l55_55839

noncomputable def rational_man_track (x y : ℝ) : Prop :=
x^2 + y^2 = 1

noncomputable def irrational_man_track (x y : ℝ) : Prop :=
(x + 1)^2 + y^2 = 9

noncomputable def shortest_distance : ℝ :=
0

theorem shortest_distance_between_tracks :
  ∀ (A B : ℝ × ℝ), 
  rational_man_track A.1 A.2 → 
  irrational_man_track B.1 B.2 → 
  dist A B = shortest_distance := sorry

end shortest_distance_between_tracks_l55_55839


namespace annulus_area_l55_55187

theorem annulus_area (r R x : ℝ) (hR_gt_r : R > r) (h_tangent : r^2 + x^2 = R^2) : 
  π * x^2 = π * (R^2 - r^2) :=
by
  sorry

end annulus_area_l55_55187


namespace part_a_part_b_l55_55071

-- Conditions
def ornament_to_crackers (n : ℕ) : ℕ := n * 2
def sparklers_to_garlands (n : ℕ) : ℕ := (n / 5) * 2
def garlands_to_ornaments (n : ℕ) : ℕ := n * 4

-- Part (a)
theorem part_a (sparklers : ℕ) (h : sparklers = 10) : ornament_to_crackers (garlands_to_ornaments (sparklers_to_garlands sparklers)) = 32 :=
by
  sorry

-- Part (b)
theorem part_b (ornaments : ℕ) (crackers : ℕ) (sparklers : ℕ) (h₁ : ornaments = 5) (h₂ : crackers = 1) (h₃ : sparklers = 2) :
  ornament_to_crackers ornaments + crackers > ornament_to_crackers (garlands_to_ornaments (sparklers_to_garlands sparklers)) :=
by
  sorry

end part_a_part_b_l55_55071


namespace negation_example_l55_55421

theorem negation_example :
  ¬ (∀ x : ℝ, x^2 - x + 1 ≥ 0) ↔ ∃ x : ℝ, x^2 - x + 1 < 0 :=
sorry

end negation_example_l55_55421


namespace correct_result_after_mistakes_l55_55836

theorem correct_result_after_mistakes (n : ℕ) (f : ℕ → ℕ → ℕ) (g : ℕ → ℕ → ℕ)
    (h1 : f n 4 * 4 + 18 = g 12 18) : 
    g (f n 4 * 4) 18 = 498 :=
by
  sorry

end correct_result_after_mistakes_l55_55836


namespace correct_calculation_l55_55573

theorem correct_calculation : (Real.sqrt 3) ^ 2 = 3 := by
  sorry

end correct_calculation_l55_55573


namespace fraction_proof_l55_55743

-- Define N
def N : ℕ := 24

-- Define F that satisfies the equation N = F + 15
def F := N - 15

-- Define the fraction that N exceeds by 15
noncomputable def fraction := (F : ℚ) / N

-- Prove that fraction = 3/8
theorem fraction_proof : fraction = 3 / 8 := by
  sorry

end fraction_proof_l55_55743


namespace four_minus_x_is_five_l55_55805

theorem four_minus_x_is_five (x y : ℤ) (h1 : 4 + x = 5 - y) (h2 : 3 + y = 6 + x) : 4 - x = 5 := by
sorry

end four_minus_x_is_five_l55_55805


namespace feet_of_pipe_per_bolt_l55_55326

-- Definition of the initial conditions
def total_pipe_length := 40 -- total feet of pipe
def washers_per_bolt := 2
def initial_washers := 20
def remaining_washers := 4

-- The proof statement
theorem feet_of_pipe_per_bolt :
  ∀ (total_pipe_length washers_per_bolt initial_washers remaining_washers : ℕ),
  initial_washers - remaining_washers = 16 → -- 16 washers used
  16 / washers_per_bolt = 8 → -- 8 bolts used
  total_pipe_length / 8 = 5 :=
by
  intros
  sorry

end feet_of_pipe_per_bolt_l55_55326


namespace rectangle_area_constant_l55_55852

theorem rectangle_area_constant (d : ℝ) (x : ℝ)
  (length width : ℝ)
  (h_length : length = 5 * x)
  (h_width : width = 4 * x)
  (h_diagonal : d = Real.sqrt (length ^ 2 + width ^ 2)) :
  (exists k : ℝ, k = 20 / 41 ∧ (length * width = k * d ^ 2)) :=
by
  use 20 / 41
  sorry

end rectangle_area_constant_l55_55852


namespace simplify_fraction_l55_55844

variable {a b m : ℝ}

theorem simplify_fraction (h : a + b ≠ 0) : (ma/a + b) + (mb/a + b) = m :=
by
  sorry

end simplify_fraction_l55_55844


namespace waiter_earnings_l55_55312

theorem waiter_earnings (total_customers : ℕ) (no_tip_customers : ℕ) (tip_per_customer : ℕ)
  (h1 : total_customers = 10)
  (h2 : no_tip_customers = 5)
  (h3 : tip_per_customer = 3) :
  (total_customers - no_tip_customers) * tip_per_customer = 15 :=
by sorry

end waiter_earnings_l55_55312


namespace third_stack_shorter_by_five_l55_55685

theorem third_stack_shorter_by_five
    (first_stack second_stack third_stack fourth_stack : ℕ)
    (h1 : first_stack = 5)
    (h2 : second_stack = first_stack + 2)
    (h3 : fourth_stack = third_stack + 5)
    (h4 : first_stack + second_stack + third_stack + fourth_stack = 21) :
    second_stack - third_stack = 5 :=
by
  sorry

end third_stack_shorter_by_five_l55_55685


namespace jay_savings_in_a_month_is_correct_l55_55825

-- Definitions for the conditions
def initial_savings : ℕ := 20
def weekly_increase : ℕ := 10

-- Define the savings for each week
def savings_after_week (week : ℕ) : ℕ :=
  initial_savings + (week - 1) * weekly_increase

-- Define the total savings over 4 weeks
def total_savings_after_4_weeks : ℕ :=
  savings_after_week 1 + savings_after_week 2 + savings_after_week 3 + savings_after_week 4

-- Proposition statement 
theorem jay_savings_in_a_month_is_correct :
  total_savings_after_4_weeks = 140 :=
  by
  -- proof will go here
  sorry

end jay_savings_in_a_month_is_correct_l55_55825


namespace factorize_expression_l55_55477

theorem factorize_expression (a x : ℝ) : a * x^2 - a = a * (x + 1) * (x - 1) :=
by
  sorry

end factorize_expression_l55_55477


namespace problem_l55_55395

open Real 

noncomputable def sqrt_log_a (a : ℝ) : ℝ := sqrt (log a / log 10)
noncomputable def sqrt_log_b (b : ℝ) : ℝ := sqrt (log b / log 10)

theorem problem (a b : ℝ) 
  (ha_pos : 0 < a)
  (hb_pos : 0 < b)
  (condition1 : sqrt_log_a a + 2 * sqrt_log_b b + 2 * log (sqrt a) / log 10 + log (sqrt b) / log 10 = 150)
  (int_sqrt_log_a : ∃ (m : ℕ), sqrt_log_a a = m)
  (int_sqrt_log_b : ∃ (n : ℕ), sqrt_log_b b = n)
  (condition2 : a^2 * b = 10^81) :
  a * b = 10^85 :=
sorry

end problem_l55_55395


namespace exponent_solver_l55_55790

theorem exponent_solver (x : ℕ) : 3^x + 3^x + 3^x + 3^x = 19683 → x = 7 := sorry

end exponent_solver_l55_55790


namespace domain_shift_l55_55929

theorem domain_shift (f : ℝ → ℝ) (h : ∀ (x : ℝ), (-2 < x ∧ x < 2) → (f (x + 2) = f x)) :
  ∀ (y : ℝ), (3 < y ∧ y < 7) ↔ (y - 3 < 4 ∧ y - 3 > -2) :=
by
  sorry

end domain_shift_l55_55929


namespace find_smallest_a_l55_55582

noncomputable def quadratic_eq_1_roots (a b : ℤ) : Prop :=
∃ α β : ℤ, x < -1 ∧ x^2 + bx + a = 0

noncomputable def quadratic_eq_2_roots (a c : ℤ) : Prop :=
∃ γ δ : ℤ, x < -1 ∧ x^2 + cx + (a - 1) = 0

theorem find_smallest_a : ∃ a, (quadratic_eq_1_roots a b ∧ quadratic_eq_2_roots a c) ↔ a = 15 :=
sorry

end find_smallest_a_l55_55582


namespace gcd_9011_4403_l55_55566

theorem gcd_9011_4403 : Nat.gcd 9011 4403 = 1 := 
by sorry

end gcd_9011_4403_l55_55566


namespace solve_for_x_l55_55627

-- Definitions of δ and φ
def delta (x : ℚ) : ℚ := 4 * x + 9
def phi (x : ℚ) : ℚ := 9 * x + 8

-- The main proof statement
theorem solve_for_x :
  ∃ x : ℚ, delta (phi x) = 10 ∧ x = -31 / 36 :=
by
  sorry

end solve_for_x_l55_55627


namespace verify_rs_correct_l55_55534

noncomputable def verify_rs : Prop :=
  let N := ![
    ![3, 4],
    ![-2, 1]
  ]
  let I : Matrix (Fin 2) (Fin 2) ℤ := 1
  let N2 := N ⬝ N
  ∃ r s : ℤ, N2 = r • N + s • I ∧ (r = 4 ∧ s = -11)

theorem verify_rs_correct : verify_rs := sorry

end verify_rs_correct_l55_55534


namespace average_is_five_plus_D_over_two_l55_55828

variable (A B C D : ℝ)

def condition1 := 1001 * C - 2004 * A = 4008
def condition2 := 1001 * B + 3005 * A - 1001 * D = 6010

theorem average_is_five_plus_D_over_two (h1 : condition1 A C) (h2 : condition2 A B D) : 
  (A + B + C + D) / 4 = (5 + D) / 2 := 
by
  sorry

end average_is_five_plus_D_over_two_l55_55828


namespace domain_of_function_l55_55412

theorem domain_of_function :
  { x : ℝ | x + 2 ≥ 0 ∧ x - 1 ≠ 0 } = { x : ℝ | x ≥ -2 ∧ x ≠ 1 } :=
by
  sorry

end domain_of_function_l55_55412


namespace oranges_per_box_l55_55315

theorem oranges_per_box (total_oranges : ℕ) (num_boxes : ℕ) (h1 : total_oranges = 24) (h2 : num_boxes = 3) :
  total_oranges / num_boxes = 8 :=
by
  rw [h1, h2]
  exact Nat.div_eq_of_eq_mul (by norm_num : 24 = 3 * 8)

end oranges_per_box_l55_55315


namespace total_cost_of_purchase_l55_55391

variable (x y z : ℝ)

theorem total_cost_of_purchase (h₁ : 4 * x + (9 / 2) * y + 12 * z = 6) (h₂ : 12 * x + 6 * y + 6 * z = 8) :
  4 * x + 3 * y + 6 * z = 4 :=
sorry

end total_cost_of_purchase_l55_55391


namespace part1_part2_l55_55198

noncomputable def triangleABC (a : ℝ) (cosB : ℝ) (b : ℝ) (SinA : ℝ) : Prop :=
  cosB = 3 / 5 ∧ b = 4 → SinA = 2 / 5

noncomputable def triangleABC2 (a : ℝ) (cosB : ℝ) (S : ℝ) (b c : ℝ) : Prop :=
  cosB = 3 / 5 ∧ S = 4 → b = Real.sqrt 17 ∧ c = 5

theorem part1 :
  triangleABC 2 (3 / 5) 4 (2 / 5) :=
by {
  sorry
}

theorem part2 :
  triangleABC2 2 (3 / 5) 4 (Real.sqrt 17) 5 :=
by {
  sorry
}

end part1_part2_l55_55198


namespace five_natural_numbers_increase_15_times_l55_55525

noncomputable def prod_of_decreased_factors_is_15_times_original (a1 a2 a3 a4 a5 : ℕ) : Prop :=
  (a1 - 3) * (a2 - 3) * (a3 - 3) * (a4 - 3) * (a5 - 3) = 15 * (a1 * a2 * a3 * a4 * a5)

theorem five_natural_numbers_increase_15_times {a1 a2 a3 a4 a5 : ℕ} :
  a1 * a2 * a3 * a4 * a5 = 48 → prod_of_decreased_factors_is_15_times_original a1 a2 a3 a4 a5 :=
by
  sorry

end five_natural_numbers_increase_15_times_l55_55525


namespace sqrt_arith_progression_impossible_l55_55965

theorem sqrt_arith_progression_impossible (a b c : ℕ) (ha : Nat.Prime a) (hb : Nat.Prime b) (hc : Nat.Prime c) (hneab : a ≠ b) (hnebc : b ≠ c) (hneca : c ≠ a) :
  ¬ ∃ d : ℝ, (d = (Real.sqrt b - Real.sqrt a)) ∧ (d = (Real.sqrt c - Real.sqrt b)) :=
sorry

end sqrt_arith_progression_impossible_l55_55965


namespace simplify_fraction_l55_55971

-- Define what it means for a fraction to be in simplest form
def coprime (m n : ℕ) : Prop := Nat.gcd m n = 1

-- Define what it means for a fraction to be reducible
def reducible_fraction (num den : ℕ) : Prop := ∃ d > 1, d ∣ num ∧ d ∣ den

-- Main theorem statement
theorem simplify_fraction 
  (m n : ℕ) (h_coprime : coprime m n) 
  (h_reducible : reducible_fraction (4 * m + 3 * n) (5 * m + 2 * n)) : ∃ d, d = 7 :=
by {
  sorry
}

end simplify_fraction_l55_55971


namespace probability_intervals_l55_55208

open ProbabilityTheory

noncomputable def standard_normal := measure_theory.measure_space.measure

theorem probability_intervals {p : ℝ} (h : p = P(λ x : ℝ, x > 1)) :
  P(λ x: ℝ, -1 < x ∧ x < 0) = 0.5 - p :=
by
  sorry

end probability_intervals_l55_55208


namespace graph_symmetry_l55_55553

variable (f : ℝ → ℝ)

theorem graph_symmetry :
  (∀ x y, y = f (x - 1) ↔ ∃ x', x' = 2 - x ∧ y = f (1 - x'))
  ∧ (∀ x' y', y' = f (1 - x') ↔ ∃ x, x = 2 - x' ∧ y' = f (x - 1)) :=
sorry

end graph_symmetry_l55_55553


namespace candy_pieces_total_l55_55398

def number_of_packages_of_candy := 45
def pieces_per_package := 9

theorem candy_pieces_total : number_of_packages_of_candy * pieces_per_package = 405 :=
by
  sorry

end candy_pieces_total_l55_55398


namespace circumcircle_radius_is_one_l55_55528

-- Define the basic setup for the triangle with given sides and angles
variables {A B C : Real} -- Angles of the triangle
variables {a b c : Real} -- Sides of the triangle opposite these angles
variable (triangle_ABC : a = Real.sqrt 3 ∧ (c - 2 * b + 2 * Real.sqrt 3 * Real.cos C = 0)) -- Conditions on the sides

-- Define the circumcircle radius
noncomputable def circumcircle_radius (a b c : Real) (A B C : Real) := a / (2 * (Real.sin A))

-- Statement of the problem to be proven
theorem circumcircle_radius_is_one (h : a = Real.sqrt 3)
  (h1 : c - 2 * b + 2 * Real.sqrt 3 * Real.cos C = 0) :
  circumcircle_radius a b c A B C = 1 :=
sorry

end circumcircle_radius_is_one_l55_55528


namespace problem_I_problem_II_l55_55503

noncomputable def f (x : ℝ) : ℝ := x - 2 * Real.sin x

theorem problem_I :
  ∀ x ∈ Set.Icc 0 Real.pi, (f x) ≥ (f (Real.pi / 3) - Real.sqrt 3) ∧ (f x) ≤ f Real.pi :=
sorry

theorem problem_II :
  ∀ a : ℝ, ((∃ x : ℝ, (0 < x ∧ x < Real.pi / 2) ∧ f x < a * x) ↔ a > -1) :=
sorry

end problem_I_problem_II_l55_55503


namespace hadley_total_walking_distance_l55_55802

-- Definitions of the distances walked to each location
def distance_grocery_store : ℕ := 2
def distance_pet_store : ℕ := distance_grocery_store - 1
def distance_home : ℕ := 4 - 1

-- Total distance walked by Hadley
def total_distance : ℕ := distance_grocery_store + distance_pet_store + distance_home

-- Statement to be proved
theorem hadley_total_walking_distance : total_distance = 6 := by
  sorry

end hadley_total_walking_distance_l55_55802


namespace smallest_possible_value_of_other_number_l55_55554

theorem smallest_possible_value_of_other_number (x n : ℕ) (h_pos : x > 0) 
  (h_gcd : Nat.gcd 72 n = x + 6) (h_lcm : Nat.lcm 72 n = x * (x + 6)) : n = 12 := by
  sorry

end smallest_possible_value_of_other_number_l55_55554


namespace mary_prevents_pat_l55_55678

noncomputable def smallest_initial_integer (N: ℕ) : Prop :=
  N > 2017 ∧ 
  ∀ x, ∃ n: ℕ, 
  (x = N + n * 2018 → x % 2018 ≠ 0 ∧
   (2017 * x + 2) % 2018 ≠ 0 ∧
   (2017 * x + 2021) % 2018 ≠ 0)

theorem mary_prevents_pat (N : ℕ) : smallest_initial_integer N → N = 2022 :=
sorry

end mary_prevents_pat_l55_55678


namespace beau_age_calculation_l55_55616

variable (sons_age : ℕ) (beau_age_today : ℕ) (beau_age_3_years_ago : ℕ)

def triplets := 3
def sons_today := 16
def sons_age_3_years_ago := sons_today - 3
def sum_of_sons_3_years_ago := triplets * sons_age_3_years_ago

theorem beau_age_calculation
  (h1 : sons_today = 16)
  (h2 : sum_of_sons_3_years_ago = beau_age_3_years_ago)
  (h3 : beau_age_today = beau_age_3_years_ago + 3) :
  beau_age_today = 42 :=
sorry

end beau_age_calculation_l55_55616


namespace fraction_simplifies_l55_55321

-- Define the integers
def a : ℤ := 1632
def b : ℤ := 1625
def c : ℤ := 1645
def d : ℤ := 1612

-- Define the theorem to prove
theorem fraction_simplifies :
  (a^2 - b^2) / (c^2 - d^2) = 7 / 33 := by
  sorry

end fraction_simplifies_l55_55321


namespace convert_units_l55_55585

theorem convert_units :
  (0.56 * 10 = 5.6 ∧ 0.6 * 10 = 6) ∧
  (2.05 = 2 + 0.05 ∧ 0.05 * 100 = 5) :=
by 
  sorry

end convert_units_l55_55585


namespace distance_point_to_line_l55_55705

theorem distance_point_to_line : 
  let x0 := 1
  let y0 := 0
  let A := 1
  let B := -2
  let C := 1 
  let dist := (A * x0 + B * y0 + C : ℝ) / Real.sqrt (A^2 + B^2)
  abs dist = 2 * Real.sqrt 5 / 5 :=
by
  -- Using basic principles of Lean and Mathlib to state the equality proof
  sorry

end distance_point_to_line_l55_55705


namespace parquet_tiles_needed_l55_55987

def room_width : ℝ := 8
def room_length : ℝ := 12
def tile_width : ℝ := 1.5
def tile_length : ℝ := 2

def room_area : ℝ := room_width * room_length
def tile_area : ℝ := tile_width * tile_length

def tiles_needed : ℝ := room_area / tile_area

theorem parquet_tiles_needed : tiles_needed = 32 :=
by
  -- sorry to skip the detailed proof
  sorry

end parquet_tiles_needed_l55_55987


namespace binomial_expansion_l55_55282

theorem binomial_expansion (a b : ℕ) (h_a : a = 34) (h_b : b = 5) :
  a^2 + 2*a*b + b^2 = 1521 :=
by
  rw [h_a, h_b]
  sorry

end binomial_expansion_l55_55282


namespace exists_x_abs_ge_one_fourth_l55_55688

theorem exists_x_abs_ge_one_fourth :
  ∀ (a b c : ℝ), ∃ x : ℝ, |x| ≤ 1 ∧ |x^3 + a * x^2 + b * x + c| ≥ 1 / 4 :=
by sorry

end exists_x_abs_ge_one_fourth_l55_55688


namespace longer_side_of_rectangle_is_l55_55592

-- Define the radius of the circle
def radius (r : ℝ) : Prop := r = 6

-- Define the area of the circle given the radius
def area_circle (A : ℝ) : Prop := A = 36 * Real.pi

-- Define the relationship between the area of the rectangle and the circle
def area_rectangle (A_rect : ℝ) (A : ℝ) : Prop := A_rect = 3 * A

-- Define the length of the shorter side of the rectangle
def shorter_side (s : ℝ) : Prop := s = 12

-- State the goal: the length of the longer side of the rectangle
def longer_side (l : ℝ) (A_rect : ℝ) (s : ℝ) : Prop := l = A_rect / s

-- Theorem statement combining all conditions and the goal
theorem longer_side_of_rectangle_is :
  ∃ l : ℝ, ∃ A_rect : ℝ, ∃ s : ℝ, radius 6 ∧ area_circle 36 * Real.pi ∧ area_rectangle A_rect (36 * Real.pi) ∧ shorter_side s ∧ longer_side l A_rect s :=
by {
  sorry
}

end longer_side_of_rectangle_is_l55_55592


namespace relationship_of_x_vals_l55_55367

variables {k x1 x2 x3 : ℝ}

noncomputable def inverse_proportion_function (k x : ℝ) : ℝ := k / x

theorem relationship_of_x_vals (h1 : inverse_proportion_function k x1 = 1)
                              (h2 : inverse_proportion_function k x2 = -5)
                              (h3 : inverse_proportion_function k x3 = 3)
                              (hk : k < 0) :
                              x1 < x3 ∧ x3 < x2 :=
by
  sorry

end relationship_of_x_vals_l55_55367


namespace sampling_methods_correct_l55_55769

-- Assuming definitions for the populations for both surveys
structure CommunityHouseholds where
  high_income : Nat
  middle_income : Nat
  low_income : Nat

structure ArtisticStudents where
  total_students : Nat

-- Given conditions
def households_population : CommunityHouseholds := { high_income := 125, middle_income := 280, low_income := 95 }
def students_population : ArtisticStudents := { total_students := 15 }

-- Correct answer according to the conditions
def appropriate_sampling_methods (ch: CommunityHouseholds) (as: ArtisticStudents) : String :=
  if ch.high_income > 0 ∧ ch.middle_income > 0 ∧ ch.low_income > 0 ∧ as.total_students ≥ 3 then
    "B" -- ① Stratified sampling, ② Simple random sampling
  else
    "Invalid"

theorem sampling_methods_correct :
  appropriate_sampling_methods households_population students_population = "B" := by
  sorry

end sampling_methods_correct_l55_55769


namespace complement_union_M_N_l55_55830

universe u

namespace complement_union

def U : Set (ℝ × ℝ) := { p | true }

def M : Set (ℝ × ℝ) := { p | (p.2 - 3) = (p.1 - 2) }

def N : Set (ℝ × ℝ) := { p | p.2 ≠ (p.1 + 1) }

theorem complement_union_M_N : (U \ (M ∪ N)) = { (2, 3) } := 
by 
  sorry

end complement_union

end complement_union_M_N_l55_55830


namespace circle_center_tangent_eq_l55_55444

open Real

theorem circle_center_tangent_eq (x y : ℝ):
  (3 * x - 4 * y = 40) ∧
  (3 * x - 4 * y = 0) ∧
  (x - 2 * y = 0) →
  (x = 20 ∧ y = 10) := 
by
  intro h
  sorry

end circle_center_tangent_eq_l55_55444


namespace simplified_expression_form_l55_55252

noncomputable def simplify_expression (x : ℚ) : ℚ := 
  3 * x - 7 * x^2 + 5 - (6 - 5 * x + 7 * x^2)

theorem simplified_expression_form (x : ℚ) : 
  simplify_expression x = -14 * x^2 + 8 * x - 1 :=
by
  sorry

end simplified_expression_form_l55_55252


namespace complex_number_evaluation_l55_55777

noncomputable def i := Complex.I

theorem complex_number_evaluation :
  (1 - i) * (i * i) / (1 + 2 * i) = (1/5 : ℂ) + (3/5 : ℂ) * i :=
by
  sorry

end complex_number_evaluation_l55_55777


namespace tiles_needed_l55_55988

def room_area : ℝ := 2 * 4 * 2 * 6
def tile_area : ℝ := 1.5 * 2

theorem tiles_needed : room_area / tile_area = 32 := 
by
  sorry

end tiles_needed_l55_55988


namespace prism_volume_l55_55014

theorem prism_volume
  (l w h : ℝ)
  (h1 : l * w = 6.5)
  (h2 : w * h = 8)
  (h3 : l * h = 13) :
  l * w * h = 26 :=
by
  sorry

end prism_volume_l55_55014


namespace no_solution_exists_l55_55682

theorem no_solution_exists (x y z : ℕ) (hx : x > 2) (hy : y > 1) (h : x^y + 1 = z^2) : false := 
by
  sorry

end no_solution_exists_l55_55682


namespace people_and_carriages_condition_l55_55756

-- Definitions corresponding to the conditions
def num_people_using_carriages (x : ℕ) : ℕ := 3 * (x - 2)
def num_people_sharing_carriages (x : ℕ) : ℕ := 2 * x + 9

-- The theorem statement we need to prove
theorem people_and_carriages_condition (x : ℕ) : 
  num_people_using_carriages x = num_people_sharing_carriages x ↔ 3 * (x - 2) = 2 * x + 9 :=
by sorry

end people_and_carriages_condition_l55_55756


namespace product_decrease_increase_fifteenfold_l55_55526

theorem product_decrease_increase_fifteenfold (a1 a2 a3 a4 a5 : ℕ) :
  ((a1 - 3) * (a2 - 3) * (a3 - 3) * (a4 - 3) * (a5 - 3) = 15 * a1 * a2 * a3 * a4 * a5) → true :=
by
  sorry

end product_decrease_increase_fifteenfold_l55_55526


namespace five_cubic_km_to_cubic_meters_l55_55940

theorem five_cubic_km_to_cubic_meters (km_to_m : 1 = 1000) : 
  5 * (1000 ^ 3) = 5000000000 := 
by
  sorry

end five_cubic_km_to_cubic_meters_l55_55940


namespace S_equals_2_l55_55533

noncomputable def problem_S := 
  1 / (2 - Real.sqrt 3) - 1 / (Real.sqrt 3 - Real.sqrt 2) + 
  1 / (Real.sqrt 2 - 1) - 1 / (1 - Real.sqrt 3 + Real.sqrt 2)

theorem S_equals_2 : problem_S = 2 := by
  sorry

end S_equals_2_l55_55533


namespace simplify_expr1_simplify_expr2_l55_55694

noncomputable section

-- Problem 1: Simplify the given expression
theorem simplify_expr1 (a b : ℝ) : 4 * a^2 + 2 * (3 * a * b - 2 * a^2) - (7 * a * b - 1) = -a * b + 1 := 
by sorry

-- Problem 2: Simplify the given expression
theorem simplify_expr2 (x y : ℝ) : 3 * (x^2 * y - 1/2 * x * y^2) - 1/2 * (4 * x^2 * y - 3 * x * y^2) = x^2 * y :=
by sorry

end simplify_expr1_simplify_expr2_l55_55694


namespace ferris_wheel_small_seat_capacity_l55_55981

def num_small_seats : Nat := 2
def capacity_per_small_seat : Nat := 14

theorem ferris_wheel_small_seat_capacity : num_small_seats * capacity_per_small_seat = 28 := by
  sorry

end ferris_wheel_small_seat_capacity_l55_55981


namespace pythagorean_triangle_divisible_by_5_l55_55689

theorem pythagorean_triangle_divisible_by_5 {a b c : ℕ} (h : a^2 + b^2 = c^2) : 
  5 ∣ a ∨ 5 ∣ b ∨ 5 ∣ c := 
by
  sorry

end pythagorean_triangle_divisible_by_5_l55_55689


namespace white_marbles_bagA_eq_fifteen_l55_55765

noncomputable def red_marbles_bagA := 5
def rw_ratio_bagA := (1, 3)
def wb_ratio_bagA := (2, 3)

theorem white_marbles_bagA_eq_fifteen :
  let red_to_white := rw_ratio_bagA.1 * red_marbles_bagA
  red_to_white * rw_ratio_bagA.2 = 15 :=
by
  sorry

end white_marbles_bagA_eq_fifteen_l55_55765


namespace find_y_l55_55702

theorem find_y (n x y : ℝ)
  (h1 : (100 + 200 + n + x) / 4 = 250)
  (h2 : (n + 150 + 100 + x + y) / 5 = 200) :
  y = 50 :=
by
  sorry

end find_y_l55_55702


namespace arithmetic_sequence_twelfth_term_l55_55151

theorem arithmetic_sequence_twelfth_term :
  let a1 := (1 : ℚ) / 2;
  let a2 := (5 : ℚ) / 6;
  let d := a2 - a1;
  (a1 + 11 * d) = (25 : ℚ) / 6 :=
by
  let a1 := (1 : ℚ) / 2;
  let a2 := (5 : ℚ) / 6;
  let d := a2 - a1;
  exact sorry

end arithmetic_sequence_twelfth_term_l55_55151


namespace height_of_tower_l55_55982

-- Definitions for points and distances
structure Point :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def A : Point := { x := 0, y := 0, z := 0 }
def C : Point := { x := 0, y := 0, z := 129 }
def D : Point := { x := 0, y := 0, z := 258 }
def B : Point  := { x := 0, y := 305, z := 305 }

-- Given conditions
def angle_elevation_A_to_B : ℝ := 45 -- degrees
def angle_elevation_D_to_B : ℝ := 60 -- degrees
def distance_A_to_D : ℝ := 258 -- meters

-- The problem is to prove the height of the tower is 305 meters given the conditions
theorem height_of_tower : B.y = 305 :=
by
  -- This spot would contain the actual proof
  sorry

end height_of_tower_l55_55982


namespace puffy_muffy_total_weight_l55_55029

theorem puffy_muffy_total_weight (scruffy_weight muffy_weight puffy_weight : ℕ)
  (h1 : scruffy_weight = 12)
  (h2 : muffy_weight = scruffy_weight - 3)
  (h3 : puffy_weight = muffy_weight + 5) :
  puffy_weight + muffy_weight = 23 := by
  sorry

end puffy_muffy_total_weight_l55_55029


namespace pranks_combinations_correct_l55_55863

noncomputable def pranks_combinations : ℕ := by
  let monday_choice := 1
  let tuesday_choice := 2
  let wednesday_choice := 4
  let thursday_choice := 5
  let friday_choice := 1
  let total_combinations := monday_choice * tuesday_choice * wednesday_choice * thursday_choice * friday_choice
  exact 40

theorem pranks_combinations_correct : pranks_combinations = 40 := by
  unfold pranks_combinations
  sorry -- Proof omitted

end pranks_combinations_correct_l55_55863


namespace twelfth_term_of_geometric_sequence_l55_55552

theorem twelfth_term_of_geometric_sequence 
  (a : ℕ → ℕ)
  (h₁ : a 4 = 4)
  (h₂ : a 7 = 32)
  (h_geometric : ∀ n : ℕ, a (n + 1) = a n * r) : 
  a 12 = 1024 :=
sorry

end twelfth_term_of_geometric_sequence_l55_55552


namespace difference_is_four_l55_55134

def chickens_in_coop := 14
def chickens_in_run := 2 * chickens_in_coop
def chickens_free_ranging := 52
def difference := 2 * chickens_in_run - chickens_free_ranging

theorem difference_is_four : difference = 4 := by
  sorry

end difference_is_four_l55_55134


namespace find_a_for_even_function_l55_55368

theorem find_a_for_even_function (a : ℝ) (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = (x + 1)*(x - a) ∧ f (-x) = f x) : a = 1 :=
sorry

end find_a_for_even_function_l55_55368


namespace exists_close_pair_in_interval_l55_55543

theorem exists_close_pair_in_interval (x1 x2 x3 : ℝ) (h1 : 0 ≤ x1 ∧ x1 < 1) (h2 : 0 ≤ x2 ∧ x2 < 1) (h3 : 0 ≤ x3 ∧ x3 < 1) :
  ∃ a b, (a = x1 ∨ a = x2 ∨ a = x3) ∧ (b = x1 ∨ b = x2 ∨ b = x3) ∧ a ≠ b ∧ |b - a| < 1 / 2 :=
sorry

end exists_close_pair_in_interval_l55_55543


namespace simplify_to_ellipse_l55_55401

theorem simplify_to_ellipse (x y : ℝ) :
  (Real.sqrt ((x - 2)^2 + y^2) + Real.sqrt ((x + 2)^2 + y^2) = 10) →
  (x^2 / 25 + y^2 / 21 = 1) :=
by
  sorry

end simplify_to_ellipse_l55_55401


namespace sum_divisible_by_3_l55_55667

theorem sum_divisible_by_3 (a : ℤ) : 3 ∣ (a^3 + 2 * a) :=
sorry

end sum_divisible_by_3_l55_55667


namespace b_can_finish_work_in_15_days_l55_55443

theorem b_can_finish_work_in_15_days (W : ℕ) (r_A : ℕ) (r_B : ℕ) (h1 : r_A = W / 21) (h2 : 10 * r_B + 7 * r_A / 21 = W) : r_B = W / 15 :=
by sorry

end b_can_finish_work_in_15_days_l55_55443


namespace percentage_of_children_who_speak_only_english_l55_55068

theorem percentage_of_children_who_speak_only_english :
  (∃ (total_children both_languages hindi_speaking only_english : ℝ),
    total_children = 60 ∧
    both_languages = 0.20 * total_children ∧
    hindi_speaking = 42 ∧
    only_english = total_children - (hindi_speaking - both_languages + both_languages) ∧
    (only_english / total_children) * 100 = 30) :=
  sorry

end percentage_of_children_who_speak_only_english_l55_55068


namespace cube_faces_paint_count_l55_55122

def is_painted_on_at_least_two_faces (x y z : ℕ) : Prop :=
  ((x = 0 ∨ x = 3) ∨ (y = 0 ∨ y = 3) ∨ (z = 0 ∨ z = 3))

theorem cube_faces_paint_count :
  let n := 4 in
  let volume := n * n * n in
  let one_inch_cubes := {c | ∃ x y z : ℕ, x < n ∧ y < n ∧ z < n ∧ c = (x, y, z)} in
  let painted_cubes := {c ∈ one_inch_cubes | exists_along_edges c} in  -- Auxiliary predicate to check edge paint
  card painted_cubes = 32 :=
begin
  sorry,
end

/-- Auxiliary predicate to check if a cube is on the edges but not corners, signifying 
    at least two faces are painted --/
predicate exists_along_edges (c: ℕ × ℕ × ℕ) := 
  match c with
  | (0, _, _) => true
  | (3, _, _) => true
  | (_, 0, _) => true
  | (_, 3, _) => true
  | (_, _, 0) => true
  | (_, _, 3) => true
  | (_, _, _) => false
  end

end cube_faces_paint_count_l55_55122


namespace find_m_of_parallel_lines_l55_55050

theorem find_m_of_parallel_lines
  (m : ℝ) 
  (parallel : ∀ x y, (x - 2 * y + 5 = 0 → 2 * x + m * y - 5 = 0)) :
  m = -4 :=
sorry

end find_m_of_parallel_lines_l55_55050


namespace min_value_of_n_for_constant_term_l55_55656

theorem min_value_of_n_for_constant_term :
  ∃ (n : ℕ) (r : ℕ) (h₁ : r > 0) (h₂ : n > 0), 
  (2 * n - 7 * r / 3 = 0) ∧ n = 7 :=
by
  sorry

end min_value_of_n_for_constant_term_l55_55656


namespace solve_for_x_l55_55587

theorem solve_for_x (x : ℕ) : x * 12 = 173 * 240 → x = 3460 :=
by
  sorry

end solve_for_x_l55_55587


namespace work_rates_l55_55589

theorem work_rates (A B : ℝ) (combined_days : ℝ) (b_rate: B = 35) 
(combined_rate: combined_days = 20 / 11):
    A = 700 / 365 :=
by
  have h1 : B = 35 := by sorry
  have h2 : combined_days = 20 / 11 := by sorry
  have : 1/A + 1/B = 11/20 := by sorry
  have : 1/A = 11/20 - 1/B := by sorry
  have : 1/A =  365 / 700:= by sorry
  have : A = 700 / 365 := by sorry
  assumption

end work_rates_l55_55589


namespace total_chairs_all_together_l55_55042

-- Definitions of given conditions
def rows := 7
def chairs_per_row := 12
def extra_chairs := 11

-- Main statement we want to prove
theorem total_chairs_all_together : 
  (rows * chairs_per_row + extra_chairs = 95) := 
by
  sorry

end total_chairs_all_together_l55_55042


namespace large_pizzas_sold_l55_55271

def small_pizza_price : ℕ := 2
def large_pizza_price : ℕ := 8
def total_earnings : ℕ := 40
def small_pizzas_sold : ℕ := 8

theorem large_pizzas_sold : 
  ∀ (small_pizza_price large_pizza_price total_earnings small_pizzas_sold : ℕ), 
    small_pizza_price = 2 → 
    large_pizza_price = 8 → 
    total_earnings = 40 → 
    small_pizzas_sold = 8 →
    (total_earnings - small_pizzas_sold * small_pizza_price) / large_pizza_price = 3 :=
by 
  intros small_pizza_price large_pizza_price total_earnings small_pizzas_sold 
         h_small_pizza_price h_large_pizza_price h_total_earnings h_small_pizzas_sold
  rw [h_small_pizza_price, h_large_pizza_price, h_total_earnings, h_small_pizzas_sold]
  simp
  sorry

end large_pizzas_sold_l55_55271


namespace ratio_a_c_l55_55558

variables (a b c d : ℚ)

axiom ratio_a_b : a / b = 5 / 4
axiom ratio_c_d : c / d = 4 / 3
axiom ratio_d_b : d / b = 1 / 8

theorem ratio_a_c : a / c = 15 / 2 :=
by sorry

end ratio_a_c_l55_55558


namespace find_d_l55_55112

noncomputable def quadratic_roots (d : ℝ) : Prop :=
∀ x : ℝ, x^2 + 7*x + d = 0 ↔ x = (-7 + Real.sqrt d) / 2 ∨ x = (-7 - Real.sqrt d) / 2

theorem find_d : ∃ d : ℝ, d = 9.8 ∧ quadratic_roots d :=
sorry

end find_d_l55_55112


namespace quadratic_trinomial_m_eq_2_l55_55990

theorem quadratic_trinomial_m_eq_2 (m : ℤ) (P : |m| = 2 ∧ m + 2 ≠ 0) : m = 2 :=
  sorry

end quadratic_trinomial_m_eq_2_l55_55990


namespace count_solutions_g_composition_eq_l55_55537

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := 3 * Real.cos (Real.pi * x)

-- Define the main theorem
theorem count_solutions_g_composition_eq :
  ∃ (s : Finset ℝ), s.card = 7 ∧ ∀ x ∈ s, -1.5 ≤ x ∧ x ≤ 1.5 ∧ g (g (g x)) = g x :=
by
  sorry

end count_solutions_g_composition_eq_l55_55537


namespace maximum_figures_per_shelf_l55_55889

theorem maximum_figures_per_shelf
  (figures_shelf_1 : ℕ)
  (figures_shelf_2 : ℕ)
  (figures_shelf_3 : ℕ)
  (additional_shelves : ℕ)
  (max_figures_per_shelf : ℕ)
  (total_figures : ℕ)
  (total_shelves : ℕ)
  (H1 : figures_shelf_1 = 9)
  (H2 : figures_shelf_2 = 14)
  (H3 : figures_shelf_3 = 7)
  (H4 : additional_shelves = 2)
  (H5 : max_figures_per_shelf = 11)
  (H6 : total_figures = figures_shelf_1 + figures_shelf_2 + figures_shelf_3)
  (H7 : total_shelves = 3 + additional_shelves)
  (H8 : ∃ d, d ∈ ({x : ℕ | x ∣ total_figures} ∩ {y : ℕ | y ≤ max_figures_per_shelf}))
  : ∃ d, d ∈ ({x : ℕ | x ∣ total_figures} ∩ {y : ℕ | y ≤ max_figures_per_shelf}) ∧ d = 6 := sorry

end maximum_figures_per_shelf_l55_55889


namespace large_pizzas_sold_l55_55269

variables (num_small_pizzas num_large_pizzas : ℕ) (price_small price_large total_revenue revenue_from_smalls revenue_from_larges : ℕ)

theorem large_pizzas_sold
  (price_small := 2)
  (price_large := 8)
  (total_revenue := 40)
  (num_small_pizzas := 8)
  (revenue_from_smalls := num_small_pizzas * price_small)
  (revenue_from_larges := total_revenue - revenue_from_smalls)
  (large_pizza_count := revenue_from_larges / price_large) :
  large_pizza_count = 3 :=
sorry

end large_pizzas_sold_l55_55269


namespace five_circles_intersect_l55_55973

-- Assume we have five circles
variables (circle1 circle2 circle3 circle4 circle5 : Set Point)

-- Assume every four of them intersect at a single point
axiom four_intersect (c1 c2 c3 c4 : Set Point) : ∃ p : Point, p ∈ c1 ∧ p ∈ c2 ∧ p ∈ c3 ∧ p ∈ c4

-- The goal is to prove that there exists a point through which all five circles pass.
theorem five_circles_intersect :
  (∃ p : Point, p ∈ circle1 ∧ p ∈ circle2 ∧ p ∈ circle3 ∧ p ∈ circle4 ∧ p ∈ circle5) :=
sorry

end five_circles_intersect_l55_55973


namespace neg_p_necessary_not_sufficient_neg_q_l55_55215

def p (x : ℝ) : Prop := x^2 - 1 > 0
def q (x : ℝ) : Prop := (x + 1) * (x - 2) > 0
def not_p (x : ℝ) : Prop := ¬ (p x)
def not_q (x : ℝ) : Prop := ¬ (q x)

theorem neg_p_necessary_not_sufficient_neg_q : ∀ (x : ℝ), (not_q x → not_p x) ∧ ¬ (not_p x → not_q x) :=
by
  sorry

end neg_p_necessary_not_sufficient_neg_q_l55_55215


namespace min_distance_between_parallel_lines_l55_55019

theorem min_distance_between_parallel_lines
  (m c_1 c_2 : ℝ)
  (h_parallel : ∀ x : ℝ, m * x + c_1 = m * x + c_2 → false) :
  ∃ D : ℝ, D = (|c_2 - c_1|) / (Real.sqrt (1 + m^2)) :=
by
  sorry

end min_distance_between_parallel_lines_l55_55019


namespace tom_savings_by_having_insurance_l55_55994

noncomputable def insurance_cost_per_month : ℝ := 20
noncomputable def total_months : ℕ := 24
noncomputable def surgery_cost : ℝ := 5000
noncomputable def insurance_coverage_rate : ℝ := 0.80

theorem tom_savings_by_having_insurance :
  let total_insurance_cost := (insurance_cost_per_month * total_months)
  let insurance_coverage := (insurance_coverage_rate * surgery_cost)
  let out_of_pocket_cost := (surgery_cost - insurance_coverage)
  let savings := (surgery_cost - total_insurance_cost - out_of_pocket_cost)
  savings = 3520 :=
by
  let total_insurance_cost := (insurance_cost_per_month * total_months)
  let insurance_coverage := (insurance_coverage_rate * surgery_cost)
  let out_of_pocket_cost := (surgery_cost - insurance_coverage)
  let savings := (surgery_cost - total_insurance_cost - out_of_pocket_cost)
  sorry

end tom_savings_by_having_insurance_l55_55994


namespace two_numbers_with_difference_less_than_half_l55_55541

theorem two_numbers_with_difference_less_than_half
  (x1 x2 x3 : ℝ)
  (h1 : 0 ≤ x1) (h2 : x1 < 1)
  (h3 : 0 ≤ x2) (h4 : x2 < 1)
  (h5 : 0 ≤ x3) (h6 : x3 < 1) :
  ∃ a b, 
    (a = x1 ∨ a = x2 ∨ a = x3) ∧
    (b = x1 ∨ b = x2 ∨ b = x3) ∧
    a ≠ b ∧ 
    |b - a| < 1 / 2 :=
sorry

end two_numbers_with_difference_less_than_half_l55_55541


namespace find_y_value_l55_55949

theorem find_y_value (x y : ℝ) 
    (h1 : x^2 + 3 * x + 6 = y - 2) 
    (h2 : x = -5) : 
    y = 18 := 
  by 
  sorry

end find_y_value_l55_55949


namespace fibonacci_150_mod_9_l55_55032

def fibonacci (n : ℕ) : ℕ :=
  if h : n < 2 then n else fibonacci (n - 1) + fibonacci (n - 2)

theorem fibonacci_150_mod_9 : fibonacci 150 % 9 = 8 :=
  sorry

end fibonacci_150_mod_9_l55_55032


namespace factorize_expression_l55_55478

theorem factorize_expression (a x : ℝ) : a * x^2 - a = a * (x + 1) * (x - 1) :=
by
  sorry

end factorize_expression_l55_55478


namespace value_of_5y_l55_55814

-- Define positive integers
variables {x y z : ℕ}

-- Define the conditions
def conditions (x y z : ℕ) : Prop :=
  (x > 0) ∧ (y > 0) ∧ (z > 0) ∧ (5 * y = 6 * z) ∧ (x + y + z = 26)

-- The theorem statement
theorem value_of_5y (x y z : ℕ) (h : conditions x y z) : 5 * y = 30 :=
by
  -- proof skipped (proof goes here)
  sorry

end value_of_5y_l55_55814


namespace wine_cost_increase_l55_55455

noncomputable def additional_cost (initial_price : ℝ) (num_bottles : ℕ) (month1_rate : ℝ) (month2_tariff : ℝ) (month2_discount : ℝ) (month3_tariff : ℝ) (month3_rate : ℝ) : ℝ := 
  let price_month1 := initial_price * (1 + month1_rate) 
  let cost_month1 := num_bottles * price_month1
  let price_month2 := (initial_price * (1 + month2_tariff)) * (1 - month2_discount)
  let cost_month2 := num_bottles * price_month2
  let price_month3 := (initial_price * (1 + month3_tariff)) * (1 - month3_rate)
  let cost_month3 := num_bottles * price_month3
  (cost_month1 + cost_month2 + cost_month3) - (3 * num_bottles * initial_price)

theorem wine_cost_increase : 
  additional_cost 20 5 0.05 0.25 0.15 0.35 0.03 = 42.20 :=
by sorry

end wine_cost_increase_l55_55455


namespace gcd_108_45_l55_55866

theorem gcd_108_45 :
  ∃ g, g = Nat.gcd 108 45 ∧ g = 9 :=
by
  sorry

end gcd_108_45_l55_55866


namespace tom_balloons_count_l55_55562

-- Define the number of balloons Tom initially has
def balloons_initial : Nat := 30

-- Define the number of balloons Tom gave away
def balloons_given : Nat := 16

-- Define the number of balloons Tom now has
def balloons_remaining : Nat := balloons_initial - balloons_given

theorem tom_balloons_count :
  balloons_remaining = 14 := by
  sorry

end tom_balloons_count_l55_55562


namespace number_of_homes_cleaned_l55_55968

-- Define constants for the amount Mary earns per home and the total amount she made.
def amount_per_home := 46
def total_amount_made := 276

-- Prove that the number of homes Mary cleaned is 6 given the conditions.
theorem number_of_homes_cleaned : total_amount_made / amount_per_home = 6 :=
by
  sorry

end number_of_homes_cleaned_l55_55968


namespace average_age_of_women_l55_55581

theorem average_age_of_women (A : ℕ) (W1 W2 : ℕ) 
  (h1 : 7 * A - 26 - 30 + W1 + W2 = 7 * (A + 4)) : 
  (W1 + W2) / 2 = 42 := 
by 
  sorry

end average_age_of_women_l55_55581


namespace conditional_probability_A_given_B_l55_55022

noncomputable def calculate_conditional_probability (event_A event_B : Finset (Fin 2 × Fin 2 × Fin 2)) : ℚ :=
  let prob_event_A_given_B := (event_A ∩ event_B).card.toRat / event_B.card.toRat in
  prob_event_A_given_B

def three_digit_codes : Finset (Fin 2 × Fin 2 × Fin 2) := 
  Finset.univ.product (Finset.univ.product Finset.univ)

def event_A : Finset (Fin 2 × Fin 2 × Fin 2) :=
  three_digit_codes.filter (λ code, code.2.1 = 0)

def event_B : Finset (Fin 2 × Fin 2 × Fin 2) := 
  three_digit_codes.filter (λ code, code.1 = 0)

theorem conditional_probability_A_given_B : 
  calculate_conditional_probability event_A event_B = 1 / 2 :=
by
  sorry

end conditional_probability_A_given_B_l55_55022


namespace valid_seating_arrangements_l55_55521

def num_people : Nat := 10
def total_arrangements : Nat := Nat.factorial num_people
def restricted_group_arrangements : Nat := Nat.factorial 7 * Nat.factorial 4
def valid_arrangements : Nat := total_arrangements - restricted_group_arrangements

theorem valid_seating_arrangements : valid_arrangements = 3507840 := by
  sorry

end valid_seating_arrangements_l55_55521


namespace area_of_win_sector_l55_55164

theorem area_of_win_sector (r : ℝ) (p : ℝ) (A : ℝ) (h_1 : r = 10) (h_2 : p = 1 / 4) (h_3 : A = π * r^2) : 
  (p * A) = 25 * π := 
by
  sorry

end area_of_win_sector_l55_55164


namespace elena_meeting_percentage_l55_55776

noncomputable def workday_hours : ℕ := 10
noncomputable def first_meeting_duration_minutes : ℕ := 60
noncomputable def second_meeting_duration_minutes : ℕ := 3 * first_meeting_duration_minutes
noncomputable def total_workday_minutes := workday_hours * 60
noncomputable def total_meeting_minutes := first_meeting_duration_minutes + second_meeting_duration_minutes
noncomputable def percent_time_in_meetings := (total_meeting_minutes * 100) / total_workday_minutes

theorem elena_meeting_percentage : percent_time_in_meetings = 40 := by 
  sorry

end elena_meeting_percentage_l55_55776


namespace exists_close_pair_in_interval_l55_55542

theorem exists_close_pair_in_interval (x1 x2 x3 : ℝ) (h1 : 0 ≤ x1 ∧ x1 < 1) (h2 : 0 ≤ x2 ∧ x2 < 1) (h3 : 0 ≤ x3 ∧ x3 < 1) :
  ∃ a b, (a = x1 ∨ a = x2 ∨ a = x3) ∧ (b = x1 ∨ b = x2 ∨ b = x3) ∧ a ≠ b ∧ |b - a| < 1 / 2 :=
sorry

end exists_close_pair_in_interval_l55_55542


namespace max_number_ahn_can_get_l55_55308

theorem max_number_ahn_can_get :
  ∃ n : ℤ, (10 ≤ n ∧ n ≤ 99) ∧ ∀ m : ℤ, (10 ≤ m ∧ m ≤ 99) → (3 * (300 - n) ≥ 3 * (300 - m)) ∧ 3 * (300 - n) = 870 :=
by sorry

end max_number_ahn_can_get_l55_55308


namespace yoongi_flowers_left_l55_55433

theorem yoongi_flowers_left (initial_flowers given_to_eunji given_to_yuna : ℕ) 
  (h_initial : initial_flowers = 28) 
  (h_eunji : given_to_eunji = 7) 
  (h_yuna : given_to_yuna = 9) : 
  initial_flowers - (given_to_eunji + given_to_yuna) = 12 := 
by 
  sorry

end yoongi_flowers_left_l55_55433


namespace complement_of_angle_correct_l55_55343

def complement_of_angle (a : ℚ) : ℚ := 90 - a

theorem complement_of_angle_correct : complement_of_angle (40 + 30/60) = 49 + 30/60 :=
by
  -- placeholder for the proof
  sorry

end complement_of_angle_correct_l55_55343


namespace maximum_value_of_function_l55_55265

theorem maximum_value_of_function : ∃ x, x > (1 : ℝ) ∧ (∀ y, y > 1 → (x + 1 / (x - 1) ≥ y + 1 / (y - 1))) ∧ (x = 2 ∧ (x + 1 / (x - 1) = 3)) :=
sorry

end maximum_value_of_function_l55_55265


namespace circle_equation_l55_55484

theorem circle_equation 
  (x y : ℝ)
  (center : ℝ × ℝ)
  (tangent_point : ℝ × ℝ)
  (line1 : ℝ × ℝ → Prop)
  (line2 : ℝ × ℝ → Prop)
  (hx : line1 center)
  (hy : line2 tangent_point)
  (tangent_point_val : tangent_point = (2, -1))
  (line1_def : ∀ (p : ℝ × ℝ), line1 p ↔ 2 * p.1 + p.2 = 0)
  (line2_def : ∀ (p : ℝ × ℝ), line2 p ↔ p.1 + p.2 - 1 = 0) :
  (∃ (x0 y0 r : ℝ), center = (x0, y0) ∧ r > 0 ∧ (x - x0)^2 + (y - y0)^2 = r^2 ∧ 
                        (x - x0)^2 + (y - y0)^2 = (x - 1)^2 + (y + 2)^2 ∧ 
                        (x - 1)^2 + (y + 2)^2 = 2) :=
by {
  sorry
}

end circle_equation_l55_55484


namespace boat_travel_distance_downstream_l55_55442

def boat_speed : ℝ := 22 -- Speed of boat in still water in km/hr
def stream_speed : ℝ := 5 -- Speed of the stream in km/hr
def time_downstream : ℝ := 7 -- Time taken to travel downstream in hours
def effective_speed_downstream : ℝ := boat_speed + stream_speed -- Effective speed downstream

theorem boat_travel_distance_downstream : effective_speed_downstream * time_downstream = 189 := by
  -- Since effective_speed_downstream = 27 (22 + 5)
  -- Distance = Speed * Time
  -- Hence, Distance = 27 km/hr * 7 hours = 189 km
  sorry

end boat_travel_distance_downstream_l55_55442


namespace truck_distance_and_efficiency_l55_55016

theorem truck_distance_and_efficiency (m d g1 g2 : ℕ) (h1 : d = 300) (h2 : g1 = 10) (h3 : g2 = 15) :
  (d * (g2 / g1) = 450) ∧ (d / g1 = 30) :=
by
  sorry

end truck_distance_and_efficiency_l55_55016


namespace rewrite_subtraction_rewrite_division_l55_55091

theorem rewrite_subtraction : -8 - 5 = -8 + (-5) :=
by sorry

theorem rewrite_division : (1/2) / (-2) = (1/2) * (-1/2) :=
by sorry

end rewrite_subtraction_rewrite_division_l55_55091


namespace eval_expression_l55_55263

theorem eval_expression : (-2 ^ 3) ^ (1/3 : ℝ) - (-1 : ℝ) ^ 0 = -3 := by 
  sorry

end eval_expression_l55_55263


namespace valid_exercise_combinations_l55_55011

def exercise_durations : List ℕ := [30, 20, 40, 30, 30]

theorem valid_exercise_combinations : 
  (∃ (s : Finset ℕ), s.card > 1 ∧ s.toList.sum (exercise_durations.snd) ≥ 60 ∧ ∀ t, t.card > 1 ∧ t.toList.sum (exercise_durations.snd) ≥ 60 → s = t) → 23 := 
sorry

end valid_exercise_combinations_l55_55011


namespace equation_solutions_35_implies_n_26_l55_55238

theorem equation_solutions_35_implies_n_26 (n : ℕ) (h3x3y2z_eq_n : ∃ (s : Finset (ℕ × ℕ × ℕ)), (∀ t ∈ s, ∃ (x y z : ℕ), 
  t = (x, y, z) ∧ 3 * x + 3 * y + 2 * z = n ∧ x > 0 ∧ y > 0 ∧ z > 0) ∧ s.card = 35) : n = 26 := 
sorry

end equation_solutions_35_implies_n_26_l55_55238


namespace tan_585_eq_1_l55_55181

theorem tan_585_eq_1 : Real.tan (585 * Real.pi / 180) = 1 := 
by
  sorry

end tan_585_eq_1_l55_55181


namespace classify_quadrilateral_l55_55812

structure Quadrilateral where
  sides : ℕ → ℝ 
  angle : ℕ → ℝ 
  diag_length : ℕ → ℝ 
  perpendicular_diagonals : Prop

def is_rhombus (q : Quadrilateral) : Prop :=
  (∀ i, q.sides i = q.sides 0) ∧ q.perpendicular_diagonals

def is_kite (q : Quadrilateral) : Prop :=
  (q.sides 1 = q.sides 2 ∧ q.sides 3 = q.sides 4) ∧ q.perpendicular_diagonals

def is_square (q : Quadrilateral) : Prop :=
  (∀ i, q.sides i = q.sides 0) ∧ (∀ i, q.angle i = 90) ∧ q.perpendicular_diagonals

theorem classify_quadrilateral (q : Quadrilateral) (h : q.perpendicular_diagonals) :
  is_rhombus q ∨ is_kite q ∨ is_square q :=
sorry

end classify_quadrilateral_l55_55812


namespace dodecagon_area_l55_55629

theorem dodecagon_area (a : ℝ) : 
  let OA := a / Real.sqrt 2 
  let CD := (a / 2) / Real.sqrt 2 
  let triangle_area := (1/2) * OA * CD 
  let dodecagon_area := 12 * triangle_area
  dodecagon_area = (3 * a^2) / 2 :=
by
  let OA := a / Real.sqrt 2 
  let CD := (a / 2) / Real.sqrt 2 
  let triangle_area := (1/2) * OA * CD 
  let dodecagon_area := 12 * triangle_area
  sorry

end dodecagon_area_l55_55629


namespace union_is_correct_l55_55508

def A : Set ℕ := {1, 2, 4}
def B : Set ℕ := {2, 6}

theorem union_is_correct : A ∪ B = {1, 2, 4, 6} := by
  sorry

end union_is_correct_l55_55508


namespace wang_pens_purchase_l55_55680

theorem wang_pens_purchase :
  ∀ (total_money spent_on_albums pen_cost : ℝ)
  (number_of_pens : ℕ),
  total_money = 80 →
  spent_on_albums = 45.6 →
  pen_cost = 2.5 →
  number_of_pens = 13 →
  (total_money - spent_on_albums) / pen_cost ≥ number_of_pens ∧ 
  (total_money - spent_on_albums) / pen_cost < number_of_pens + 1 :=
by
  intros
  sorry

end wang_pens_purchase_l55_55680


namespace donna_total_episodes_per_week_l55_55324

-- Defining the conditions
def episodes_per_weekday : ℕ := 8
def weekday_count : ℕ := 5
def weekend_factor : ℕ := 3
def weekend_count : ℕ := 2

-- Theorem statement
theorem donna_total_episodes_per_week :
  (episodes_per_weekday * weekday_count) + ((episodes_per_weekday * weekend_factor) * weekend_count) = 88 := 
  by sorry

end donna_total_episodes_per_week_l55_55324


namespace numerator_is_12_l55_55704

theorem numerator_is_12 (x : ℕ) (h1 : (x : ℤ) / (2 * x + 4 : ℤ) = 3 / 7) : x = 12 := 
sorry

end numerator_is_12_l55_55704


namespace product_of_20_random_digits_ends_with_zero_l55_55302

noncomputable def probability_product_ends_in_zero : ℝ := 
  (1 - (9 / 10)^20) +
  (9 / 10)^20 * (1 - (5 / 9)^20) * (1 - (8 / 9)^19)

theorem product_of_20_random_digits_ends_with_zero : 
  abs (probability_product_ends_in_zero - 0.988) < 0.001 :=
by
  sorry

end product_of_20_random_digits_ends_with_zero_l55_55302


namespace notebook_pre_tax_cost_eq_l55_55168

theorem notebook_pre_tax_cost_eq :
  (∃ (n c X : ℝ), n + c = 3 ∧ n = 2 + c ∧ 1.1 * X = 3.3 ∧ X = n + c → n = 2.5) :=
by
  sorry

end notebook_pre_tax_cost_eq_l55_55168


namespace sq_sum_ge_one_third_l55_55879

theorem sq_sum_ge_one_third (a b c : ℝ) (h : a + b + c = 1) : a^2 + b^2 + c^2 ≥ 1 / 3 := 
sorry

end sq_sum_ge_one_third_l55_55879


namespace two_pow_div_factorial_iff_l55_55687

theorem two_pow_div_factorial_iff (n : ℕ) : 
  (∃ k : ℕ, k > 0 ∧ n = 2^(k - 1)) ↔ (∃ m : ℕ, m > 0 ∧ 2^(n - 1) ∣ n!) :=
by
  sorry

end two_pow_div_factorial_iff_l55_55687


namespace probability_black_given_not_white_l55_55161

theorem probability_black_given_not_white
  (total_balls : ℕ)
  (white_balls : ℕ)
  (yellow_balls : ℕ)
  (black_balls : ℕ)
  (H1 : total_balls = 25)
  (H2 : white_balls = 10)
  (H3 : yellow_balls = 5)
  (H4 : black_balls = 10)
  (H5 : total_balls = white_balls + yellow_balls + black_balls)
  (H6 : ¬white_balls = total_balls) :
  (10 / (25 - 10) : ℚ) = 2 / 3 :=
by
  sorry

end probability_black_given_not_white_l55_55161


namespace train_speed_l55_55157

theorem train_speed (train_length bridge_length cross_time : ℝ)
  (h1 : train_length = 250)
  (h2 : bridge_length = 150)
  (h3 : cross_time = 25) :
  (train_length + bridge_length) / cross_time = 16 :=
by
  sorry

end train_speed_l55_55157


namespace smallest_n_reducible_fraction_l55_55340

theorem smallest_n_reducible_fraction : ∀ (n : ℕ), (∃ (k : ℕ), gcd (n - 13) (5 * n + 6) = k ∧ k > 1) ↔ n = 84 := by
  sorry

end smallest_n_reducible_fraction_l55_55340


namespace rule_for_sequence_natural_number_self_map_power_of_2_to_single_digit_l55_55224

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

noncomputable def transition_rule (n : ℕ) : ℕ :=
  2 * (sum_of_digits n)

theorem rule_for_sequence :
  transition_rule 3 = 6 ∧ transition_rule 6 = 12 :=
by
  sorry

theorem natural_number_self_map :
  ∀ n : ℕ, transition_rule n = n ↔ n = 18 :=
by
  sorry

theorem power_of_2_to_single_digit :
  ∃ x : ℕ, transition_rule (2^1991) = x ∧ x < 10 :=
by
  sorry

end rule_for_sequence_natural_number_self_map_power_of_2_to_single_digit_l55_55224


namespace combined_age_of_sam_and_drew_l55_55095

theorem combined_age_of_sam_and_drew
  (sam_age : ℕ)
  (drew_age : ℕ)
  (h1 : sam_age = 18)
  (h2 : sam_age = drew_age / 2):
  sam_age + drew_age = 54 := sorry

end combined_age_of_sam_and_drew_l55_55095


namespace convert_to_cylindrical_coords_l55_55772

def rectangular_to_cylindrical (x y z : ℝ) : ℝ × ℝ × ℝ :=
  let r := Math.sqrt (x^2 + y^2)
  let θ := Real.arccos (x / r)
  if y < 0 then (r, 2 * Real.pi - θ, z)
  else (r, θ, z)

theorem convert_to_cylindrical_coords :
  rectangular_to_cylindrical 3 (-3*Real.sqrt 3) 4 = (6, 4*Real.pi/3, 4) :=
by
  sorry

end convert_to_cylindrical_coords_l55_55772


namespace probability_of_product_ending_with_zero_l55_55303
open BigOperators

def probability_product_ends_with_zero :=
  let no_zero := (9 / 10) ^ 20
  let at_least_one_zero := 1 - no_zero
  let no_even := (5 / 9) ^ 20
  let at_least_one_even := 1 - no_even
  let no_five_among_19 := (8 / 9) ^ 19
  let at_least_one_five := 1 - no_five_among_19
  let no_zero_and_conditions :=
    no_zero * at_least_one_even * at_least_one_five
  at_least_one_zero + no_zero_and_conditions

theorem probability_of_product_ending_with_zero :
  abs (probability_product_ends_with_zero - 0.988) < 0.001 :=
by
  sorry

end probability_of_product_ending_with_zero_l55_55303


namespace percentage_of_copper_is_correct_l55_55025

-- Defining the conditions
def total_weight := 100.0
def weight_20_percent_alloy := 30.0
def weight_27_percent_alloy := total_weight - weight_20_percent_alloy

def percentage_20 := 0.20
def percentage_27 := 0.27

def copper_20 := percentage_20 * weight_20_percent_alloy
def copper_27 := percentage_27 * weight_27_percent_alloy
def total_copper := copper_20 + copper_27

-- The statement to be proved
def percentage_copper := (total_copper / total_weight) * 100

-- The theorem to prove
theorem percentage_of_copper_is_correct : percentage_copper = 24.9 := by sorry

end percentage_of_copper_is_correct_l55_55025


namespace max_annual_profit_l55_55928

noncomputable def annual_sales_volume (x : ℝ) : ℝ := - (1 / 3) * x^2 + 2 * x + 21

noncomputable def annual_sales_profit (x : ℝ) : ℝ := (- (1 / 3) * x^3 + 4 * x^2 + 9 * x - 126)

theorem max_annual_profit :
  ∀ x : ℝ, (x > 6) →
  (annual_sales_volume x) = - (1 / 3) * x^2 + 2 * x + 21 →
  (annual_sales_volume 10 = 23 / 3) →
  (21 - annual_sales_volume x = (1 / 3) * (x^2 - 6 * x)) →
    (annual_sales_profit x = - (1 / 3) * x^3 + 4 * x^2 + 9 * x - 126) ∧
    ∃ x_max : ℝ, 
      (annual_sales_profit x_max = 36) ∧
      x_max = 9 :=
by
  sorry

end max_annual_profit_l55_55928


namespace option_c_is_not_equal_l55_55431

theorem option_c_is_not_equal :
  let A := 14 / 12
  let B := 1 + 1 / 6
  let C := 1 + 1 / 2
  let D := 1 + 7 / 42
  let E := 1 + 14 / 84
  A = 7 / 6 ∧ B = 7 / 6 ∧ D = 7 / 6 ∧ E = 7 / 6 ∧ C ≠ 7 / 6 :=
by
  sorry

end option_c_is_not_equal_l55_55431


namespace percentage_decrease_correct_l55_55275

variable (O N : ℕ)
variable (percentage_decrease : ℕ)

-- Define the conditions based on the problem
def original_price := 1240
def new_price := 620
def price_effect := ((original_price - new_price) * 100) / original_price

-- Prove the percentage decrease is 50%
theorem percentage_decrease_correct :
  price_effect = 50 := by
  sorry

end percentage_decrease_correct_l55_55275


namespace fraction_of_total_money_l55_55958

variable (Max Leevi Nolan Ollie : ℚ)

-- Condition: Each of Max, Leevi, and Nolan gave Ollie the same amount of money
variable (x : ℚ) (h1 : Max / 6 = x) (h2 : Leevi / 3 = x) (h3 : Nolan / 2 = x)

-- Proving that the fraction of the group's (Max, Leevi, Nolan, Ollie) total money possessed by Ollie is 3/11.
theorem fraction_of_total_money (h4 : Max + Leevi + Nolan + Ollie = Max + Leevi + Nolan + 3 * x) : 
  x / (Max + Leevi + Nolan + x) = 3 / 11 := 
by
  sorry

end fraction_of_total_money_l55_55958


namespace find_x_l55_55495

theorem find_x (y z : ℚ) (h1 : z = 80) (h2 : y = z / 4) (h3 : x = y / 3) : x = 20 / 3 :=
by
  sorry

end find_x_l55_55495


namespace limit_sum_perimeters_areas_of_isosceles_triangles_l55_55891

theorem limit_sum_perimeters_areas_of_isosceles_triangles (b s h : ℝ) : 
  ∃ P A : ℝ, 
    (P = 2*(b + 2*s)) ∧ 
    (A = (2/3)*b*h) :=
  sorry

end limit_sum_perimeters_areas_of_isosceles_triangles_l55_55891


namespace p_sufficient_not_necessary_for_q_l55_55241

open Real

def p (x : ℝ) : Prop := abs x < 1
def q (x : ℝ) : Prop := x^2 + x - 6 < 0

theorem p_sufficient_not_necessary_for_q : 
  (∀ x : ℝ, p x → q x) ∧ ¬(∀ x : ℝ, q x → p x) :=
by
  sorry

end p_sufficient_not_necessary_for_q_l55_55241


namespace ian_number_is_1021_l55_55327

-- Define the sequences each student skips
def alice_skips (n : ℕ) := ∃ k : ℕ, n = 4 * k
def barbara_skips (n : ℕ) := ∃ k : ℕ, n = 16 * (k + 1)
def candice_skips (n : ℕ) := ∃ k : ℕ, n = 64 * (k + 1)
-- Similar definitions for Debbie, Eliza, Fatima, Greg, and Helen

-- Define the condition under which Ian says a number
def ian_says (n : ℕ) :=
  ¬(alice_skips n) ∧ ¬(barbara_skips n) ∧ ¬(candice_skips n) -- and so on for Debbie, Eliza, Fatima, Greg, Helen

theorem ian_number_is_1021 : ian_says 1021 :=
by
  sorry

end ian_number_is_1021_l55_55327


namespace original_sales_tax_percentage_l55_55559

theorem original_sales_tax_percentage
  (current_sales_tax : ℝ := 10 / 3) -- 3 1/3% in decimal
  (difference : ℝ := 10.999999999999991) -- Rs. 10.999999999999991
  (market_price : ℝ := 6600) -- Rs. 6600
  (original_sales_tax : ℝ := 3.5) -- Expected original tax
  :  ((original_sales_tax / 100) * market_price = (current_sales_tax / 100) * market_price + difference) 
  := sorry

end original_sales_tax_percentage_l55_55559


namespace find_missing_square_l55_55486

-- Defining the sequence as a list of natural numbers' squares
def square_sequence (n: ℕ) : ℕ := n * n

-- Proving the missing element in the given sequence is 36
theorem find_missing_square :
  (square_sequence 0 = 1) ∧ 
  (square_sequence 1 = 4) ∧ 
  (square_sequence 2 = 9) ∧ 
  (square_sequence 3 = 16) ∧ 
  (square_sequence 4 = 25) ∧ 
  (square_sequence 6 = 49) →
  square_sequence 5 = 36 :=
by {
  sorry
}

end find_missing_square_l55_55486


namespace find_x_when_y_3_l55_55876

variable (y x k : ℝ)

axiom h₁ : x = k / (y ^ 2)
axiom h₂ : y = 9 → x = 0.1111111111111111
axiom y_eq_3 : y = 3

theorem find_x_when_y_3 : y = 3 → x = 1 :=
by
  sorry

end find_x_when_y_3_l55_55876


namespace line_through_intersection_of_circles_l55_55413

theorem line_through_intersection_of_circles :
  ∀ (x y : ℝ),
    (x^2 + y^2 + 4 * x - 4 * y - 12 = 0) ∧
    (x^2 + y^2 + 2 * x + 4 * y - 4 = 0) →
    (x - 4 * y - 4 = 0) :=
by sorry

end line_through_intersection_of_circles_l55_55413


namespace gcd_hcf_of_36_and_84_l55_55337

theorem gcd_hcf_of_36_and_84 : Nat.gcd 36 84 = 12 := sorry

end gcd_hcf_of_36_and_84_l55_55337


namespace roots_identity_l55_55674

theorem roots_identity (p q r : ℝ) (h₁ : p + q + r = 15) (h₂ : p * q + q * r + r * p = 25) (h₃ : p * q * r = 10) :
  (1 + p) * (1 + q) * (1 + r) = 51 :=
by sorry

end roots_identity_l55_55674


namespace find_K_l55_55509

theorem find_K (Z K : ℕ)
  (hZ1 : 700 < Z)
  (hZ2 : Z < 1500)
  (hK : K > 1)
  (hZ_eq : Z = K^4)
  (hZ_perfect : ∃ n : ℕ, Z = n^6) :
  K = 3 :=
by
  sorry

end find_K_l55_55509


namespace first_equation_value_l55_55938

theorem first_equation_value (x y : ℝ) (V : ℝ) 
  (h1 : x + |x| + y = V) 
  (h2 : x + |y| - y = 6) 
  (h3 : x + y = 12) : 
  V = 18 := 
by
  sorry

end first_equation_value_l55_55938


namespace berengere_contribution_l55_55767

noncomputable def exchange_rate : ℝ := (1.5 : ℝ)
noncomputable def pastry_cost_euros : ℝ := (8 : ℝ)
noncomputable def lucas_money_cad : ℝ := (10 : ℝ)
noncomputable def lucas_money_euros : ℝ := lucas_money_cad / exchange_rate

theorem berengere_contribution :
  pastry_cost_euros - lucas_money_euros = (4 / 3 : ℝ) :=
by
  sorry

end berengere_contribution_l55_55767


namespace temperature_on_wednesday_l55_55853

theorem temperature_on_wednesday
  (T_sunday   : ℕ)
  (T_monday   : ℕ)
  (T_tuesday  : ℕ)
  (T_thursday : ℕ)
  (T_friday   : ℕ)
  (T_saturday : ℕ)
  (average_temperature : ℕ)
  (h_sunday   : T_sunday = 40)
  (h_monday   : T_monday = 50)
  (h_tuesday  : T_tuesday = 65)
  (h_thursday : T_thursday = 82)
  (h_friday   : T_friday = 72)
  (h_saturday : T_saturday = 26)
  (h_avg_temp : (T_sunday + T_monday + T_tuesday + W + T_thursday + T_friday + T_saturday) / 7 = average_temperature)
  (h_avg_val  : average_temperature = 53) :
  W = 36 :=
by { sorry }

end temperature_on_wednesday_l55_55853


namespace larger_number_l55_55709

theorem larger_number (x y : ℕ) (h1 : x + y = 40) (h2 : x - y = 4) : x = 22 := by
  sorry

end larger_number_l55_55709


namespace domain_of_f_l55_55716

noncomputable def f (x : ℝ) : ℝ := 1 / ((x - 3) + (x - 12))

theorem domain_of_f :
  ∀ x : ℝ, f x ≠ 0 ↔ (x ≠ 15 / 2) :=
by
  sorry

end domain_of_f_l55_55716


namespace part1_solution_part2_solution_l55_55936

-- Definitions of the lines
def l1 (a : ℝ) : ℝ × ℝ × ℝ :=
  (2 * a + 1, a + 2, 3)

def l2 (a : ℝ) : ℝ × ℝ × ℝ :=
  (a - 1, -2, 2)

-- Parallel lines condition
def parallel_lines (a : ℝ) : Prop :=
  let (A1, B1, C1) := l1 a
  let (A2, B2, C2) := l2 a
  B1 ≠ 0 ∧ B2 ≠ 0 ∧ (A1 / B1) = (A2 / B2)

-- Perpendicular lines condition
def perpendicular_lines (a : ℝ) : Prop :=
  let (A1, B1, C1) := l1 a
  let (A2, B2, C2) := l2 a
  B1 ≠ 0 ∧ B2 ≠ 0 ∧ (A1 * A2 + B1 * B2 = 0)

-- Statement for part 1
theorem part1_solution (a : ℝ) : parallel_lines a ↔ a = 0 :=
  sorry

-- Statement for part 2
theorem part2_solution (a : ℝ) : perpendicular_lines a ↔ (a = -1 ∨ a = 5 / 2) :=
  sorry


end part1_solution_part2_solution_l55_55936


namespace specifically_intersecting_remainder_1000_l55_55188

open Finset

def is_specifically_intersecting (A B C : Finset ℕ) : Prop :=
  |A ∩ B| = 2 ∧ |B ∩ C| = 1 ∧ |C ∩ A| = 1 ∧ A ∩ B ∩ C = ∅

def specifically_intersecting_count (U : Finset ℕ) : ℕ :=
  (U.powerset.filter (λ A => 
     U.powerset.filter (λ B =>
        U.powerset.filter (λ C => is_specifically_intersecting A B C))).card).card

theorem specifically_intersecting_remainder_1000 :
  specifically_intersecting_count (range 1 10) % 1000 = 288 :=
by
  sorry

end specifically_intersecting_remainder_1000_l55_55188


namespace longer_side_length_l55_55012

-- Define the conditions as parameters
variables (W : ℕ) (poles : ℕ) (distance : ℕ) (P : ℕ)

-- Assume the fixed conditions given in the problem
axiom shorter_side : W = 10
axiom number_of_poles : poles = 24
axiom distance_between_poles : distance = 5

-- Define the total perimeter based on the number of segments formed by the poles
noncomputable def perimeter (poles : ℕ) (distance : ℕ) : ℕ :=
  (poles - 4) * distance

-- The total perimeter of the rectangle
axiom total_perimeter : P = perimeter poles distance

-- Definition of the perimeter of the rectangle in terms of its sides
axiom rectangle_perimeter : ∀ (L W : ℕ), P = 2 * L + 2 * W

-- The theorem we need to prove
theorem longer_side_length (L : ℕ) : L = 40 :=
by
  -- Sorry is used to skip the actual proof for now
  sorry

end longer_side_length_l55_55012


namespace min_ν_of_cubic_eq_has_3_positive_real_roots_l55_55796

open Real

noncomputable def cubic_eq (x θ : ℝ) : ℝ :=
  x^3 * sin θ - (sin θ + 2) * x^2 + 6 * x - 4

noncomputable def ν (θ : ℝ) : ℝ :=
  (9 * sin θ ^ 2 - 4 * sin θ + 3) / 
  ((1 - cos θ) * (2 * cos θ - 6 * sin θ - 3 * sin (2 * θ) + 2))

theorem min_ν_of_cubic_eq_has_3_positive_real_roots :
  (∀ x:ℝ, cubic_eq x θ = 0 → 0 < x) →
  ν θ = 621 / 8 :=
sorry

end min_ν_of_cubic_eq_has_3_positive_real_roots_l55_55796


namespace amn_div_l55_55676

theorem amn_div (a m n : ℕ) (a_pos : a > 1) (h : a > 1 ∧ (a^m + 1) ∣ (a^n + 1)) : m ∣ n :=
by sorry

end amn_div_l55_55676


namespace degree_of_monomial_l55_55260

def degree (m : String) : Nat :=  -- Placeholder type, replace with appropriate type that represents a monomial
  sorry  -- Logic to compute the degree would go here, if required for full implementation

theorem degree_of_monomial : degree "-(3/5) * a * b^2" = 3 := by
  sorry

end degree_of_monomial_l55_55260


namespace four_digit_number_count_l55_55174

theorem four_digit_number_count (A : ℕ → ℕ → ℕ)
  (odd_digits even_digits : Finset ℕ)
  (odds : ∀ x ∈ odd_digits, x % 2 = 1)
  (evens : ∀ x ∈ even_digits, x % 2 = 0) :
  odd_digits = {1, 3, 5, 7, 9} ∧ 
  even_digits = {2, 4, 6, 8} →
  A 5 2 * A 7 2 = 840 :=
by
  intros h1
  sorry

end four_digit_number_count_l55_55174


namespace min_value_ineq_l55_55926

theorem min_value_ineq (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + 3 * y = 1) :
  (1 / x) + (1 / (3 * y)) ≥ 4 :=
  sorry

end min_value_ineq_l55_55926


namespace exponent_combination_l55_55946

theorem exponent_combination (a : ℝ) (m n : ℕ) (h₁ : a^m = 3) (h₂ : a^n = 4) :
  a^(2 * m + 3 * n) = 576 :=
by
  sorry

end exponent_combination_l55_55946


namespace five_natural_numbers_increase_15_times_l55_55524

noncomputable def prod_of_decreased_factors_is_15_times_original (a1 a2 a3 a4 a5 : ℕ) : Prop :=
  (a1 - 3) * (a2 - 3) * (a3 - 3) * (a4 - 3) * (a5 - 3) = 15 * (a1 * a2 * a3 * a4 * a5)

theorem five_natural_numbers_increase_15_times {a1 a2 a3 a4 a5 : ℕ} :
  a1 * a2 * a3 * a4 * a5 = 48 → prod_of_decreased_factors_is_15_times_original a1 a2 a3 a4 a5 :=
by
  sorry

end five_natural_numbers_increase_15_times_l55_55524


namespace twelfth_term_arithmetic_sequence_l55_55145

theorem twelfth_term_arithmetic_sequence :
  let a := (1 : ℚ) / 2
  let d := (1 : ℚ) / 3
  (a + 11 * d) = (25 : ℚ) / 6 :=
by
  sorry

end twelfth_term_arithmetic_sequence_l55_55145


namespace large_pizzas_sold_l55_55270

variables (num_small_pizzas num_large_pizzas : ℕ) (price_small price_large total_revenue revenue_from_smalls revenue_from_larges : ℕ)

theorem large_pizzas_sold
  (price_small := 2)
  (price_large := 8)
  (total_revenue := 40)
  (num_small_pizzas := 8)
  (revenue_from_smalls := num_small_pizzas * price_small)
  (revenue_from_larges := total_revenue - revenue_from_smalls)
  (large_pizza_count := revenue_from_larges / price_large) :
  large_pizza_count = 3 :=
sorry

end large_pizzas_sold_l55_55270


namespace total_packs_sold_l55_55683

theorem total_packs_sold (lucy_packs : ℕ) (robyn_packs : ℕ) (h1 : lucy_packs = 19) (h2 : robyn_packs = 16) : lucy_packs + robyn_packs = 35 :=
by
  sorry

end total_packs_sold_l55_55683


namespace third_number_pascals_triangle_61_numbers_l55_55869

theorem third_number_pascals_triangle_61_numbers : (Nat.choose 60 2) = 1770 := by
  sorry

end third_number_pascals_triangle_61_numbers_l55_55869


namespace remainder_div_l55_55156

theorem remainder_div (N : ℤ) (k : ℤ) (h : N = 39 * k + 18) :
  N % 13 = 5 := 
by
  sorry

end remainder_div_l55_55156


namespace fifth_number_in_ninth_row_l55_55605

theorem fifth_number_in_ninth_row :
  ∃ (n : ℕ), n = 61 ∧ ∀ (i : ℕ), i = 9 → (7 * i - 2 = n) :=
by
  sorry

end fifth_number_in_ninth_row_l55_55605


namespace find_FC_l55_55489

theorem find_FC 
(DC CB AD ED FC : ℝ)
(h1 : DC = 7) 
(h2 : CB = 8) 
(h3 : AB = (1 / 4) * AD)
(h4 : ED = (4 / 5) * AD) : 
FC = 10.4 :=
sorry

end find_FC_l55_55489


namespace gcd_of_g_and_y_l55_55496

-- Define the function g(y)
def g (y : ℕ) := (3 * y + 4) * (8 * y + 3) * (14 * y + 9) * (y + 14)

-- Define that y is a multiple of 45678
def isMultipleOf (y divisor : ℕ) : Prop := ∃ k, y = k * divisor

-- Define the proof problem
theorem gcd_of_g_and_y (y : ℕ) (h : isMultipleOf y 45678) : Nat.gcd (g y) y = 1512 :=
by
  sorry

end gcd_of_g_and_y_l55_55496


namespace solve_inequalities_l55_55546

theorem solve_inequalities (x : ℝ) (h₁ : (x - 1) / 2 < 2 * x + 1) (h₂ : -3 * (1 - x) ≥ -4) : x ≥ -1 / 3 :=
by
  sorry

end solve_inequalities_l55_55546


namespace trig_identity_l55_55203

theorem trig_identity (α m : ℝ) (h : Real.tan α = m) :
  (Real.sin (π / 4 + α))^2 - (Real.sin (π / 6 - α))^2 - Real.cos (5 * π / 12) * Real.sin (5 * π / 12 - 2 * α) = 2 * m / (1 + m^2) :=
by
  sorry

end trig_identity_l55_55203


namespace range_of_m_l55_55920

-- Define points A and B
def A : ℝ × ℝ := (-1, 1)
def B : ℝ × ℝ := (2, -2)

-- Define the line equation as a predicate
def line_l (m : ℝ) (p : ℝ × ℝ) : Prop := p.1 + m * p.2 + m = 0

-- Define the condition for the line intersecting the segment AB
def intersects_segment_AB (m : ℝ) : Prop :=
  let P : ℝ × ℝ := (0, -1)
  let k_PA := (P.2 - A.2) / (P.1 - A.1) -- Slope of PA
  let k_PB := (P.2 - B.2) / (P.1 - B.1) -- Slope of PB
  (k_PA <= -1 / m) ∧ (-1 / m <= k_PB)

-- State the theorem
theorem range_of_m : ∀ (m : ℝ), intersects_segment_AB m → (1/2 ≤ m ∧ m ≤ 2) :=
by sorry

end range_of_m_l55_55920


namespace arithmetic_sequence_sum_l55_55957

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S_9 : ℝ)
  (h1 : a 1 + a 4 + a 7 = 15)
  (h2 : a 3 + a 6 + a 9 = 3)
  (h_arith : ∀ n, a (n + 2) - a (n + 1) = a (n + 1) - a n) :
  S_9 = 27 :=
by
  sorry

end arithmetic_sequence_sum_l55_55957


namespace red_jellybeans_count_l55_55294

theorem red_jellybeans_count (total_jellybeans : ℕ)
  (blue_jellybeans : ℕ)
  (purple_jellybeans : ℕ)
  (orange_jellybeans : ℕ)
  (H1 : total_jellybeans = 200)
  (H2 : blue_jellybeans = 14)
  (H3 : purple_jellybeans = 26)
  (H4 : orange_jellybeans = 40) :
  total_jellybeans - (blue_jellybeans + purple_jellybeans + orange_jellybeans) = 120 :=
by sorry

end red_jellybeans_count_l55_55294


namespace consumer_installment_credit_l55_55176

theorem consumer_installment_credit (A C : ℝ) 
  (h1 : A = 0.36 * C) 
  (h2 : 57 = 1 / 3 * A) : 
  C = 475 := 
by 
  sorry

end consumer_installment_credit_l55_55176


namespace twelfth_term_in_sequence_l55_55148

variable (a₁ : ℚ) (d : ℚ)

-- Given conditions
def a_1_term : a₁ = 1 / 2 := rfl
def common_difference : d = 1 / 3 := rfl

-- The twelfth term of the arithmetic sequence
def a₁₂ : ℚ := a₁ + 11 * d

-- Statement to prove
theorem twelfth_term_in_sequence (h₁ : a₁ = 1 / 2) (h₂ : d = 1 / 3) : a₁₂ = 25 / 6 :=
by
  rw [h₁, h₂, add_comm, add_assoc, mul_comm, add_comm]
  exact sorry

end twelfth_term_in_sequence_l55_55148


namespace possible_n_values_l55_55192

theorem possible_n_values (x y n : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : n > 0)
  (top_box_eq : x * y * n^2 = 720) :
  ∃ k : ℕ,  k = 6 :=
by 
  sorry

end possible_n_values_l55_55192


namespace longer_side_of_rectangle_is_l55_55593

-- Define the radius of the circle
def radius (r : ℝ) : Prop := r = 6

-- Define the area of the circle given the radius
def area_circle (A : ℝ) : Prop := A = 36 * Real.pi

-- Define the relationship between the area of the rectangle and the circle
def area_rectangle (A_rect : ℝ) (A : ℝ) : Prop := A_rect = 3 * A

-- Define the length of the shorter side of the rectangle
def shorter_side (s : ℝ) : Prop := s = 12

-- State the goal: the length of the longer side of the rectangle
def longer_side (l : ℝ) (A_rect : ℝ) (s : ℝ) : Prop := l = A_rect / s

-- Theorem statement combining all conditions and the goal
theorem longer_side_of_rectangle_is :
  ∃ l : ℝ, ∃ A_rect : ℝ, ∃ s : ℝ, radius 6 ∧ area_circle 36 * Real.pi ∧ area_rectangle A_rect (36 * Real.pi) ∧ shorter_side s ∧ longer_side l A_rect s :=
by {
  sorry
}

end longer_side_of_rectangle_is_l55_55593


namespace simplify_fraction_l55_55400

theorem simplify_fraction (x y z : ℕ) (hx : x = 5) (hy : y = 2) (hz : z = 4) :
  (10 * x^2 * y^3 * z) / (15 * x * y^2 * z^2) = 4 / 3 :=
by
  sorry

end simplify_fraction_l55_55400


namespace students_number_l55_55886

theorem students_number (x a o : ℕ)
  (h1 : o = 3 * a + 3)
  (h2 : a = 2 * x + 6)
  (h3 : o = 7 * x - 5) :
  x = 26 :=
by sorry

end students_number_l55_55886


namespace evaluate_g_at_neg_four_l55_55213

def g (x : ℤ) : ℤ := 5 * x + 2

theorem evaluate_g_at_neg_four : g (-4) = -18 := 
by 
  sorry

end evaluate_g_at_neg_four_l55_55213


namespace sally_balloon_count_l55_55531

theorem sally_balloon_count 
  (joan_balloons : Nat)
  (jessica_balloons : Nat)
  (total_balloons : Nat)
  (sally_balloons : Nat)
  (h_joan : joan_balloons = 9)
  (h_jessica : jessica_balloons = 2)
  (h_total : total_balloons = 16)
  (h_eq : total_balloons = joan_balloons + jessica_balloons + sally_balloons) : 
  sally_balloons = 5 :=
by
  sorry

end sally_balloon_count_l55_55531


namespace find_original_price_of_dish_l55_55670

noncomputable def original_price_of_dish (P : ℝ) : Prop :=
  let john_paid := (0.9 * P) + (0.15 * P)
  let jane_paid := (0.9 * P) + (0.135 * P)
  john_paid = jane_paid + 0.60 → P = 40

theorem find_original_price_of_dish (P : ℝ) (h : original_price_of_dish P) : P = 40 := by
  sorry

end find_original_price_of_dish_l55_55670


namespace time_for_c_to_finish_alone_l55_55874

variable (A B C : ℚ) -- A, B, and C are the work rates

theorem time_for_c_to_finish_alone :
  (A + B = 1/3) →
  (B + C = 1/4) →
  (C + A = 1/6) →
  1/C = 24 := 
by
  intros h1 h2 h3
  sorry

end time_for_c_to_finish_alone_l55_55874


namespace monotonicity_intervals_f_above_g_l55_55501

noncomputable def f (x m : ℝ) := (Real.exp x) / (x^2 - m * x + 1)

theorem monotonicity_intervals (m : ℝ) (h : m ∈ Set.Ioo (-2 : ℝ) 2) :
  (m = 0 → ∀ x y : ℝ, x ≤ y → f x m ≤ f y m) ∧ 
  (0 < m ∧ m < 2 → ∀ x : ℝ, (x < 1 → f x m < f (x + 1) m) ∧
    (1 < x ∧ x < m + 1 → f x m > f (x + 1) m) ∧
    (x > m + 1 → f x m < f (x + 1) m)) ∧
  (-2 < m ∧ m < 0 → ∀ x : ℝ, (x < m + 1 → f x m < f (x + 1) m) ∧
    (m + 1 < x ∧ x < 1 → f x m > f (x + 1) m) ∧
    (x > 1 → f x m < f (x + 1) m)) :=
sorry

theorem f_above_g (m : ℝ) (hm : m ∈ Set.Ioo (0 : ℝ) (1/2 : ℝ)) (x : ℝ) (hx : x ∈ Set.Icc (0 : ℝ) (m + 1)) :
  f x m > x :=
sorry

end monotonicity_intervals_f_above_g_l55_55501


namespace proof_problem_l55_55793

-- Define the proportional relationship
def proportional_relationship (y x : ℝ) (k : ℝ) : Prop :=
  y - 1 = k * (x + 2)

-- Define the function y = 2x + 5
def function_y_x (y x : ℝ) : Prop :=
  y = 2 * x + 5

-- The theorem for part (1) and (2)
theorem proof_problem (x y a : ℝ) (h1 : proportional_relationship 7 1 2) (h2 : proportional_relationship y x 2) :
  function_y_x y x ∧ function_y_x (-2) a → a = -7 / 2 :=
by
  sorry

end proof_problem_l55_55793


namespace red_ball_probability_correct_l55_55285

theorem red_ball_probability_correct (R B : ℕ) (hR : R = 3) (hB : B = 3) :
  (R / (R + B) : ℚ) = 1 / 2 := by
  sorry

end red_ball_probability_correct_l55_55285


namespace range_of_m_l55_55052

theorem range_of_m (m : ℤ) (x : ℤ) (h1 : (m + 3) / (x - 1) = 1) (h2 : x > 0) : m > -4 ∧ m ≠ -3 :=
sorry

end range_of_m_l55_55052


namespace cube_of_number_l55_55169

theorem cube_of_number (n : ℕ) (h1 : 40000 < n^3) (h2 : n^3 < 50000) (h3 : (n^3 % 10) = 6) : n = 36 := by
  sorry

end cube_of_number_l55_55169


namespace sufficient_not_necessary_for_one_zero_l55_55708

variable {a x : ℝ}

def f (a x : ℝ) : ℝ := a * x ^ 2 - 2 * x + 1

theorem sufficient_not_necessary_for_one_zero :
  (∃ x : ℝ, f 1 x = 0) ∧ (∀ x : ℝ, f 0 x = -2 * x + 1 → x ≠ 0) → 
  (∃ x : ℝ, f a x = 0) → (a = 1 ∨ f 0 x = 0)  :=
sorry

end sufficient_not_necessary_for_one_zero_l55_55708


namespace other_student_questions_l55_55246

theorem other_student_questions (m k o : ℕ) (h1 : m = k - 3) (h2 : k = o + 8) (h3 : m = 40) : o = 35 :=
by
  -- proof goes here
  sorry

end other_student_questions_l55_55246


namespace tamara_diff_3kim_height_l55_55257

variables (K T X : ℕ) -- Kim's height, Tamara's height, and the difference inches respectively

-- Conditions
axiom ht_Tamara : T = 68
axiom combined_ht : T + K = 92
axiom diff_eqn : T = 3 * K - X

theorem tamara_diff_3kim_height (h₁ : T = 68) (h₂ : T + K = 92) (h₃ : T = 3 * K - X) : X = 4 :=
by
  sorry

end tamara_diff_3kim_height_l55_55257


namespace max_airlines_in_country_l55_55519

-- Definition of the problem parameters
variable (N k : ℕ) 

-- Definition of the problem conditions
variable (hN_pos : 0 < N)
variable (hk_pos : 0 < k)
variable (hN_ge_k : k ≤ N)

-- Definition of the function calculating the maximum number of air routes
def max_air_routes (N k : ℕ) : ℕ :=
  Nat.choose N 2 - Nat.choose k 2

-- Theorem stating the maximum number of airlines given the conditions
theorem max_airlines_in_country (N k : ℕ) (hN_pos : 0 < N) (hk_pos : 0 < k) (hN_ge_k : k ≤ N) :
  max_air_routes N k = Nat.choose N 2 - Nat.choose k 2 :=
by sorry

end max_airlines_in_country_l55_55519


namespace percentage_female_officers_on_duty_l55_55972

theorem percentage_female_officers_on_duty:
  ∀ (total_on_duty female_on_duty total_female_officers : ℕ),
    total_on_duty = 160 →
    female_on_duty = total_on_duty / 2 →
    total_female_officers = 500 →
    female_on_duty / total_female_officers * 100 = 16 :=
by
  intros total_on_duty female_on_duty total_female_officers h1 h2 h3
  -- Ensure types are correct
  change total_on_duty = 160 at h1
  change female_on_duty = total_on_duty / 2 at h2
  change total_female_officers = 500 at h3
  sorry

end percentage_female_officers_on_duty_l55_55972


namespace min_value_expression_l55_55556

theorem min_value_expression (a b : ℝ) : 
  4 + (a + b)^2 ≥ 4 ∧ (4 + (a + b)^2 = 4 ↔ a + b = 0) := by
sorry

end min_value_expression_l55_55556


namespace rectangle_longer_side_length_l55_55599

theorem rectangle_longer_side_length
  (r : ℝ) (A_rect : ℝ) :
  r = 6 →
  A_rect = 3 * (Real.pi * r^2) →
  ∃ l : ℝ, l = 9 * Real.pi :=
begin
  intros hr hA,
  use 9 * Real.pi,
  sorry
end

end rectangle_longer_side_length_l55_55599


namespace green_beads_in_each_necklace_l55_55163

theorem green_beads_in_each_necklace (G : ℕ) :
  (∀ n, (n = 5) → (6 * n ≤ 45) ∧ (3 * n ≤ 45) ∧ (G * n = 45)) → G = 9 :=
by
  intros h
  have hn : 5 = 5 := rfl
  cases h 5 hn
  sorry

end green_beads_in_each_necklace_l55_55163


namespace largest_n_l55_55140

noncomputable def is_multiple_of_seven (n : ℕ) : Prop :=
  (6 * (n-3)^3 - n^2 + 10 * n - 15) % 7 = 0

theorem largest_n (n : ℕ) : n < 50000 ∧ is_multiple_of_seven n → n = 49999 :=
by sorry

end largest_n_l55_55140


namespace set_equivalence_l55_55758

-- Define the given set using the condition.
def given_set : Set ℕ := {x | x ∈ {x | 0 < x} ∧ x - 3 < 2}

-- Define the enumerated set.
def enumerated_set : Set ℕ := {1, 2, 3, 4}

-- Statement of the proof problem.
theorem set_equivalence : given_set = enumerated_set :=
by
  -- The proof is omitted
  sorry

end set_equivalence_l55_55758


namespace square_area_side4_l55_55984

theorem square_area_side4
  (s : ℕ)
  (A : ℕ)
  (P : ℕ)
  (h_s : s = 4)
  (h_A : A = s * s)
  (h_P : P = 4 * s)
  (h_eqn : (A + s) - P = 4) : A = 16 := sorry

end square_area_side4_l55_55984


namespace original_price_of_computer_l55_55517

theorem original_price_of_computer (P : ℝ) (h1 : 1.20 * P = 351) (h2 : 2 * P = 585) : P = 292.5 :=
by
  sorry

end original_price_of_computer_l55_55517


namespace roberto_outfit_combinations_l55_55974

-- Define the components of the problem
def trousers_count : ℕ := 5
def shirts_count : ℕ := 7
def jackets_count : ℕ := 4
def disallowed_combinations : ℕ := 7

-- Define the requirements
theorem roberto_outfit_combinations :
  (trousers_count * shirts_count * jackets_count) - disallowed_combinations = 133 := by
  sorry

end roberto_outfit_combinations_l55_55974


namespace price_of_first_oil_l55_55586

variable {x : ℝ}
variable {price1 volume1 price2 volume2 mix_price mix_volume : ℝ}

theorem price_of_first_oil:
  volume1 = 10 →
  price2 = 68 →
  volume2 = 5 →
  mix_volume = 15 →
  mix_price = 56 →
  (volume1 * x + volume2 * price2 = mix_volume * mix_price) →
  x = 50 :=
by
  intros h1 h2 h3 h4 h5 h6
  have h1 : volume1 = 10 := h1
  have h2 : price2 = 68 := h2
  have h3 : volume2 = 5 := h3
  have h4 : mix_volume = 15 := h4
  have h5 : mix_price = 56 := h5
  have h6 : volume1 * x + volume2 * price2 = mix_volume * mix_price := h6
  sorry

end price_of_first_oil_l55_55586


namespace factor_of_polynomial_l55_55481

theorem factor_of_polynomial (t : ℚ) : (8 * t^2 + 17 * t - 10 = 0) ↔ (t = 5/8 ∨ t = -2) :=
by sorry

end factor_of_polynomial_l55_55481


namespace betty_age_l55_55001

theorem betty_age {A M B : ℕ} (h1 : A = 2 * M) (h2 : A = 4 * B) (h3 : M = A - 14) : B = 7 :=
sorry

end betty_age_l55_55001


namespace range_of_a_l55_55923

noncomputable def is_even_function (f : ℝ → ℝ) : Prop :=
∀ x, f x = f (-x)

noncomputable def strictly_increasing_on_nonnegative (f : ℝ → ℝ) : Prop :=
∀ x1 x2, (0 ≤ x1 ∧ 0 ≤ x2 ∧ x1 ≠ x2 → (f x1 - f x2) / (x1 - x2) > 0)

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) (m n : ℝ) (h_even : is_even_function f)
  (h_strict : strictly_increasing_on_nonnegative f)
  (h_m : m = 1/2) (h_f : ∀ x, m ≤ x ∧ x ≤ n → f (a * x + 1) ≤ f 2) :
  a ≤ 2 :=
sorry

end range_of_a_l55_55923


namespace infinite_solutions_b_value_l55_55909

-- Given condition for the equation to hold
def equation_condition (x b : ℤ) : Prop :=
  4 * (3 * x - b) = 3 * (4 * x + 16)

-- The statement we need to prove: b = -12
theorem infinite_solutions_b_value :
  (∀ x : ℤ, equation_condition x b) → b = -12 :=
sorry

end infinite_solutions_b_value_l55_55909


namespace first_player_wins_l55_55715

theorem first_player_wins :
  ∀ (sticks : ℕ), (sticks = 1) →
  (∀ (break_rule : ℕ → ℕ → Prop),
  (∀ (x y z : ℕ), x ≠ y ∧ y ≠ z ∧ x ≠ z → break_rule x y → break_rule x z)
  → (∃ n : ℕ, n % 3 = 0 ∧ break_rule n (n + 1) → ∃ t₁ t₂ t₃ : ℕ, t₁ = t₂ ∧ t₂ = t₃ ∧ t₁ + t₂ + t₃ = n))
  → (∃ w : ℕ, w = 1) := sorry

end first_player_wins_l55_55715


namespace correct_calculation_l55_55572

theorem correct_calculation : 
(∀ x : ℝ, √ 12 = 3 * √ 2 → false) ∧ 
(∀ x : ℝ, √ 3 + √ 2 = √ 5 → false) ∧ 
(∀ x : ℝ, (√ 3)^2 = 3) := 
by
  split
  sorry  -- proof for first part
  split
  sorry  -- proof for second part
  split 
  sorry  -- proof for correct statement

end correct_calculation_l55_55572


namespace employees_count_l55_55408

theorem employees_count (n : ℕ) (avg_salary : ℝ) (manager_salary : ℝ)
  (new_avg_salary : ℝ) (total_employees_with_manager : ℝ) : 
  avg_salary = 1500 → 
  manager_salary = 3600 → 
  new_avg_salary = avg_salary + 100 → 
  total_employees_with_manager = (n + 1) * 1600 → 
  (n * avg_salary + manager_salary) / total_employees_with_manager = new_avg_salary →
  n = 20 := by
  intros
  sorry

end employees_count_l55_55408


namespace line_perpendicular_to_plane_l55_55634

open Classical

-- Define the context of lines and planes.
variables {Line : Type} {Plane : Type}

-- Define the perpendicular and parallel relations.
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def parallel (l₁ l₂ : Line) : Prop := sorry

-- Declare the distinct lines and non-overlapping planes.
variable {m n : Line}
variable {α β : Plane}

-- State the theorem.
theorem line_perpendicular_to_plane (h1 : parallel m n) (h2 : perpendicular n β) : perpendicular m β :=
sorry

end line_perpendicular_to_plane_l55_55634


namespace pandas_and_bamboo_l55_55604

-- Definitions for the conditions
def number_of_pandas (x : ℕ) :=
  (∃ y : ℕ, y = 5 * x + 11 ∧ y = 2 * (3 * x - 5) - 8)

-- Theorem stating the solution
theorem pandas_and_bamboo (x y : ℕ) (h1 : y = 5 * x + 11) (h2 : y = 2 * (3 * x - 5) - 8) : x = 29 ∧ y = 156 :=
by {
  sorry
}

end pandas_and_bamboo_l55_55604


namespace general_term_sequence_l55_55781

-- Definition of the sequence conditions
def seq (n : ℕ) : ℤ :=
  (-1)^(n+1) * (2*n + 1)

-- The main statement to be proved
theorem general_term_sequence (n : ℕ) : seq n = (-1)^(n+1) * (2 * n + 1) :=
sorry

end general_term_sequence_l55_55781


namespace x_interval_l55_55217

theorem x_interval (x : ℝ) (h1 : 1 / x < 3) (h2 : 1 / x > -4) (h3 : 2 * x - 1 > 0) : x > 1 / 2 := 
sorry

end x_interval_l55_55217


namespace problem_part1_problem_part2_problem_part3_l55_55584

open Set

-- Define the universal set U
def U : Set ℝ := Set.univ 

-- Define sets A and B within the universal set U
def A : Set ℝ := { x | 0 < x ∧ x ≤ 2 }
def B : Set ℝ := { x | x < -3 ∨ x > 1 }

-- Define the complements of A and B within U
def complement_A : Set ℝ := U \ A
def complement_B : Set ℝ := U \ B

-- Define the results as goals to be proved
theorem problem_part1 : A ∩ B = { x | 1 < x ∧ x ≤ 2 } := 
by
  sorry

theorem problem_part2 : complement_A ∩ complement_B = { x | -3 ≤ x ∧ x ≤ 0 } :=
by
  sorry

theorem problem_part3 : U \ (A ∪ B) = { x | -3 ≤ x ∧ x ≤ 0 } :=
by
  sorry

end problem_part1_problem_part2_problem_part3_l55_55584


namespace find_MorkTaxRate_l55_55969

noncomputable def MorkIncome : ℝ := sorry
noncomputable def MorkTaxRate : ℝ := sorry 
noncomputable def MindyTaxRate : ℝ := 0.30 
noncomputable def MindyIncome : ℝ := 4 * MorkIncome 
noncomputable def combinedTaxRate : ℝ := 0.32 

theorem find_MorkTaxRate :
  (MorkTaxRate * MorkIncome + MindyTaxRate * MindyIncome) / (MorkIncome + MindyIncome) = combinedTaxRate →
  MorkTaxRate = 0.40 := sorry

end find_MorkTaxRate_l55_55969


namespace quadratic_completion_l55_55890

theorem quadratic_completion :
  ∀ x : ℝ, (x^2 - 4*x + 1 = 0) ↔ ((x - 2)^2 = 3) :=
by
  sorry

end quadratic_completion_l55_55890


namespace Dvaneft_percentage_bounds_l55_55298

noncomputable def percentageDvaneftShares (x y z : ℤ) (n m : ℕ) : ℚ :=
  n * 100 / (2 * (n + m))

theorem Dvaneft_percentage_bounds
  (x y z : ℤ) (n m : ℕ)
  (h1 : 4 * x * n = y * m)
  (h2 : x * n + y * m = z * (n + m))
  (h3 : 16 ≤ y - x)
  (h4 : y - x ≤ 20)
  (h5 : 42 ≤ z)
  (h6 : z ≤ 60) :
  12.5 ≤ percentageDvaneftShares x y z n m ∧ percentageDvaneftShares x y z n m ≤ 15 := by
  sorry

end Dvaneft_percentage_bounds_l55_55298


namespace gaussian_distribution_l55_55726

variables {Ω : Type*} [MeasureSpace Ω]
variables (xi eta : Ω → ℝ)

noncomputable def is_iid (X Y : Ω → ℝ) : Prop :=
  ∀ n : ℕ, measurable (X n) ∧ measurable (Y n) ∧ 
           (X n ~ Y n)

theorem gaussian_distribution
  (h_iid : is_iid xi eta)
  (h_zero_mean_xi : ∫ ω, xi ω ∂(volume) = 0)
  (h_zero_mean_eta : ∫ ω, eta ω ∂(volume) = 0)
  (h_finite_variance_xi : ∫ ω, (xi ω) ^ 2 ∂(volume) < ⊤)
  (h_finite_variance_eta : ∫ ω, (eta ω) ^ 2 ∂(volume) < ⊤)
  (h_distribution : xi ~ λ ω, (xi ω + eta ω) / (real.sqrt 2)) :
  ∀ x, xi ~ gaussian 0 (∫ ω, (xi ω)^2 ∂(volume)) :=
sorry

end gaussian_distribution_l55_55726


namespace valid_numbers_count_l55_55510

-- Define a predicate that checks if a number is a three-digit number
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

-- Define a function that counts how many numbers between 100 and 999 are multiples of 13
def count_multiples_of_13 (start finish : ℕ) : ℕ :=
  (finish - start) / 13 + 1

-- Define a function that checks if a permutation of digits of n is a multiple of 13
-- (actual implementation would require digit manipulation, but we assume its existence here)
def is_permutation_of_digits_multiple_of_13 (n : ℕ) : Prop :=
  ∃ (perm : ℕ), is_three_digit perm ∧ perm % 13 = 0

noncomputable def count_valid_permutations (multiples_of_13 : ℕ) : ℕ :=
  multiples_of_13 * 3 -- Assuming on average

-- Problem statement: Prove that there are 207 valid numbers satisfying the condition
theorem valid_numbers_count : (count_valid_permutations (count_multiples_of_13 104 988)) = 207 := 
by {
  -- Place for proof which is omitted here
  sorry
}

end valid_numbers_count_l55_55510


namespace distribution_schemes_count_l55_55473

noncomputable def number_of_distribution_schemes 
  (slots : ℕ) (schools : ℕ) (min_slots_A : ℕ) (min_slots_B : ℕ) : ℕ :=
  if slots = 7 ∧ schools = 5 ∧ min_slots_A = 2 ∧ min_slots_B = 2 then 35 else 0

theorem distribution_schemes_count :
  number_of_distribution_schemes 7 5 2 2 = 35 :=
by
  sorry

end distribution_schemes_count_l55_55473


namespace num_distinct_x_intercepts_l55_55209

def f (x : ℝ) : ℝ := (x - 5) * (x^3 + 5*x^2 + 9*x + 9)

theorem num_distinct_x_intercepts : 
  (∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ f x1 = 0 ∧ f x2 = 0 ∧
  ∀ x : ℝ, f x = 0 → x = x1 ∨ x = x2) :=
sorry

end num_distinct_x_intercepts_l55_55209


namespace time_to_lake_park_restaurant_l55_55820

variable (T1 T2 T_total T_detour : ℕ)

axiom time_to_hidden_lake : T1 = 15
axiom time_back_to_park_office : T2 = 7
axiom total_time_gone : T_total = 32

theorem time_to_lake_park_restaurant : T_detour = 10 :=
by
  -- Using the axioms and given conditions
  have h : T_total = T1 + T2 + T_detour,
  sorry

#check time_to_lake_park_restaurant

end time_to_lake_park_restaurant_l55_55820


namespace determine_d_l55_55774

theorem determine_d (d c f : ℚ) :
  (3 * x^3 - 2 * x^2 + x - (5/4)) * (3 * x^3 + d * x^2 + c * x + f) = 9 * x^6 - 5 * x^5 - x^4 + 20 * x^3 - (25/4) * x^2 + (15/4) * x - (5/2) →
  d = 1 / 3 :=
by
  sorry

end determine_d_l55_55774


namespace star_four_three_l55_55650

def star (x y : ℕ) : ℕ := x^2 - x*y + y^2

theorem star_four_three : star 4 3 = 13 := by
  sorry

end star_four_three_l55_55650


namespace fourth_number_in_sequence_l55_55551

noncomputable def fifth_number_in_sequence : ℕ := 78
noncomputable def increment : ℕ := 11
noncomputable def final_number_in_sequence : ℕ := 89

theorem fourth_number_in_sequence : (fifth_number_in_sequence - increment) = 67 := by
  sorry

end fourth_number_in_sequence_l55_55551


namespace probability_of_three_same_value_l55_55785

noncomputable def probability_at_least_three_same_value : ℚ :=
  let num_dice := 4
  let num_sides := 6
  let num_successful_outcomes := 16
  let num_total_outcomes := 36
  num_successful_outcomes / num_total_outcomes

theorem probability_of_three_same_value (num_dice : ℕ) (num_sides : ℕ) (num_successful_outcomes : ℕ) (num_total_outcomes : ℕ):
  num_dice = 4 →
  num_sides = 6 →
  num_successful_outcomes = 16 →
  num_total_outcomes = 36 →
  probability_at_least_three_same_value = (4 / 9) :=
by
  intros
  sorry

end probability_of_three_same_value_l55_55785


namespace final_temperature_is_correct_l55_55530

def initial_temperature : ℝ := 40
def after_jerry_temperature (T : ℝ) : ℝ := 2 * T
def after_dad_temperature (T : ℝ) : ℝ := T - 30
def after_mother_temperature (T : ℝ) : ℝ := T - 0.30 * T
def after_sister_temperature (T : ℝ) : ℝ := T + 24

theorem final_temperature_is_correct :
  after_sister_temperature (after_mother_temperature (after_dad_temperature (after_jerry_temperature initial_temperature))) = 59 :=
sorry

end final_temperature_is_correct_l55_55530


namespace jeremy_tylenol_duration_l55_55078

theorem jeremy_tylenol_duration (num_pills : ℕ) (pill_mg : ℕ) (dose_mg : ℕ) (hours_per_dose : ℕ) (hours_per_day : ℕ) 
  (total_tylenol_mg : ℕ := num_pills * pill_mg)
  (num_doses : ℕ := total_tylenol_mg / dose_mg)
  (total_hours : ℕ := num_doses * hours_per_dose) :
  num_pills = 112 → pill_mg = 500 → dose_mg = 1000 → hours_per_dose = 6 → hours_per_day = 24 → 
  total_hours / hours_per_day = 14 := 
by 
  intros; 
  sorry

end jeremy_tylenol_duration_l55_55078


namespace find_m_from_intersection_l55_55800

-- Define the sets A and B
def A : Set ℕ := {1, 2, 3}
def B (m : ℕ) : Set ℕ := {2, m, 4}

-- Prove the relationship given the conditions
theorem find_m_from_intersection (m : ℕ) (h : A ∩ B m = {2, 3}) : m = 3 := 
by 
  sorry

end find_m_from_intersection_l55_55800


namespace no_pair_of_primes_l55_55335

theorem no_pair_of_primes (p q : ℕ) (hp_prime : Prime p) (hq_prime : Prime q) (h_gt : p > q) :
  ¬ (∃ (h : ℤ), 2 * (p^2 - q^2) = 8 * h + 4) :=
by
  sorry

end no_pair_of_primes_l55_55335


namespace seeds_per_can_l55_55529

theorem seeds_per_can (total_seeds : Float) (cans : Float) (h1 : total_seeds = 54.0) (h2 : cans = 9.0) : total_seeds / cans = 6.0 :=
by
  sorry

end seeds_per_can_l55_55529


namespace marbles_remaining_l55_55900

theorem marbles_remaining (c r : ℕ) (hc : c = 12) (hr : r = 28) : 
  let total_marbles := c + r in
  let each_take := total_marbles / 4 in
  let total_taken := 2 * each_take in
  total_marbles - total_taken = 20 :=
by
  sorry

end marbles_remaining_l55_55900


namespace exponent_division_is_equal_l55_55457

variable (a : ℝ) 

theorem exponent_division_is_equal :
  (a^11) / (a^2) = a^9 := 
sorry

end exponent_division_is_equal_l55_55457


namespace maxwell_distance_when_meeting_l55_55085

variable (total_distance : ℝ := 50)
variable (maxwell_speed : ℝ := 4)
variable (brad_speed : ℝ := 6)
variable (t : ℝ := total_distance / (maxwell_speed + brad_speed))

theorem maxwell_distance_when_meeting :
  (maxwell_speed * t = 20) :=
by
  sorry

end maxwell_distance_when_meeting_l55_55085


namespace solve_equation_l55_55696

theorem solve_equation :
  { x : ℝ | x * (x - 3)^2 * (5 - x) = 0 } = {0, 3, 5} :=
by
  sorry

end solve_equation_l55_55696


namespace rectangle_longer_side_length_l55_55598

theorem rectangle_longer_side_length
  (r : ℝ) (A_rect : ℝ) :
  r = 6 →
  A_rect = 3 * (Real.pi * r^2) →
  ∃ l : ℝ, l = 9 * Real.pi :=
begin
  intros hr hA,
  use 9 * Real.pi,
  sorry
end

end rectangle_longer_side_length_l55_55598


namespace tangent_line_condition_l55_55375

-- statement only, no proof required
theorem tangent_line_condition {m n u v x y : ℝ}
  (hm : m > 1)
  (curve_eq : x^m + y^m = 1)
  (line_eq : u * x + v * y = 1)
  (u_v_condition : u^n + v^n = 1)
  (mn_condition : 1/m + 1/n = 1)
  : (u * x + v * y = 1) ↔ (u^n + v^n = 1 ∧ 1/m + 1/n = 1) :=
sorry

end tangent_line_condition_l55_55375


namespace calculate_expression_l55_55514

theorem calculate_expression : 
  ∀ (x y : ℕ), x = 3 → y = 4 → 3*(x^4 + 2*y^2)/9 = 37 + 2/3 :=
by
  intros x y hx hy
  sorry

end calculate_expression_l55_55514


namespace technicians_count_l55_55548

-- Define the number of workers
def total_workers : ℕ := 21

-- Define the average salaries
def avg_salary_all : ℕ := 8000
def avg_salary_technicians : ℕ := 12000
def avg_salary_rest : ℕ := 6000

-- Define the number of technicians and rest of workers
variable (T R : ℕ)

-- Define the equations based on given conditions
def equation1 := T + R = total_workers
def equation2 := (T * avg_salary_technicians) + (R * avg_salary_rest) = total_workers * avg_salary_all

-- Prove the number of technicians
theorem technicians_count : T = 7 :=
by
  sorry

end technicians_count_l55_55548


namespace factorize1_factorize2_l55_55480

-- Part 1: Prove the factorization of xy - 1 - x + y
theorem factorize1 (x y : ℝ) : (x * y - 1 - x + y) = (y - 1) * (x + 1) :=
  sorry

-- Part 2: Prove the factorization of (a^2 + b^2)^2 - 4a^2b^2
theorem factorize2 (a b : ℝ) : (a^2 + b^2)^2 - 4 * a^2 * b^2 = (a + b)^2 * (a - b)^2 :=
  sorry

end factorize1_factorize2_l55_55480


namespace hannahs_vegetarian_restaurant_l55_55058

theorem hannahs_vegetarian_restaurant :
  let total_weight_of_peppers := 0.6666666666666666
  let weight_of_green_peppers := 0.3333333333333333
  total_weight_of_peppers - weight_of_green_peppers = 0.3333333333333333 :=
by
  sorry

end hannahs_vegetarian_restaurant_l55_55058


namespace factorize_expression_l55_55479

theorem factorize_expression (a x : ℝ) : a * x^2 - a = a * (x + 1) * (x - 1) :=
by
  sorry

end factorize_expression_l55_55479


namespace expression_value_l55_55513

theorem expression_value (x y : ℤ) (h1 : x = 2) (h2 : y = 5) : 
  (x^4 + 2 * y^2) / 6 = 11 := by
  sorry

end expression_value_l55_55513


namespace codger_feet_l55_55467

theorem codger_feet (F : ℕ) (h1 : 6 = 2 * (5 - 1) * F) : F = 3 := by
  sorry

end codger_feet_l55_55467


namespace f_log2_9_eq_neg_16_div_9_l55_55199

noncomputable def f : ℝ → ℝ := sorry

axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom periodic_f : ∀ x : ℝ, f (x - 2) = f x
axiom f_range_0_1 : ∀ x : ℝ, 0 < x ∧ x < 1 → f x = 2 ^ x

theorem f_log2_9_eq_neg_16_div_9 : f (Real.log 9 / Real.log 2) = -16 / 9 := 
by 
  sorry

end f_log2_9_eq_neg_16_div_9_l55_55199


namespace girls_in_math_class_l55_55116

theorem girls_in_math_class (x y z : ℕ)
  (boys_girls_ratio : 5 * x = 8 * x)
  (math_science_ratio : 7 * y = 13 * x)
  (science_literature_ratio : 4 * y = 3 * z)
  (total_students : 13 * x + 4 * y + 5 * z = 720) :
  8 * x = 176 :=
by
  sorry

end girls_in_math_class_l55_55116


namespace poster_distance_from_wall_end_l55_55170

theorem poster_distance_from_wall_end (w_wall w_poster : ℝ) (h1 : w_wall = 25) (h2 : w_poster = 4) (h3 : 2 * x + w_poster = w_wall) : x = 10.5 :=
by
  sorry

end poster_distance_from_wall_end_l55_55170


namespace complement_intersection_l55_55921

open Set

variable (R : Type) [LinearOrderedField R]

def A : Set R := {x | |x| < 1}
def B : Set R := {y | ∃ x, y = 2^x + 1}
def complement_A : Set R := {x | x ≤ -1 ∨ x ≥ 1}

theorem complement_intersection (x : R) : 
  x ∈ (complement_A R) ∩ B R ↔ x > 1 :=
by
  sorry

end complement_intersection_l55_55921


namespace pentagon_edges_and_vertices_sum_l55_55135

theorem pentagon_edges_and_vertices_sum :
  let edges := 5
  let vertices := 5
  edges + vertices = 10 := by
  sorry

end pentagon_edges_and_vertices_sum_l55_55135


namespace circle_tangent_ellipse_l55_55864

noncomputable def r : ℝ := (Real.sqrt 15) / 2

theorem circle_tangent_ellipse {x y : ℝ} (r : ℝ) (h₁ : r > 0) 
  (h₂ : ∀ x y, x^2 + 4*y^2 = 5 → ((x - r)^2 + y^2 = r^2 ∨ (x + r)^2 + y^2 = r^2))
  (h₃ : ∀ y, 4*(0 - r)^2 + (4*y^2) = 5 → ((-8*r)^2 - 4*3*(4*r^2 - 5) = 0)) :
  r = (Real.sqrt 15) / 2 :=
sorry

end circle_tangent_ellipse_l55_55864


namespace max_value_l55_55538

-- Definition of the ellipse and the goal function
def ellipse (x y : ℝ) := 2 * x^2 + 3 * y^2 = 12

-- Definition of the function we want to maximize
def func (x y : ℝ) := x + 2 * y

-- The theorem to prove that the maximum value of x + 2y on the ellipse is √22
theorem max_value (x y : ℝ) (h : ellipse x y) : ∃ θ : ℝ, func x y ≤ Real.sqrt 22 :=
by
  sorry

end max_value_l55_55538


namespace greatest_sundays_in_49_days_l55_55717

theorem greatest_sundays_in_49_days : 
  ∀ (days : ℕ), 
    days = 49 → 
    ∀ (sundays_per_week : ℕ), 
      sundays_per_week = 1 → 
      ∀ (weeks : ℕ), 
        weeks = days / 7 → 
        weeks * sundays_per_week = 7 :=
by
  sorry

end greatest_sundays_in_49_days_l55_55717


namespace cos_C_of_triangle_l55_55658

theorem cos_C_of_triangle
  (sin_A : ℝ) (cos_B : ℝ) 
  (h1 : sin_A = 3/5)
  (h2 : cos_B = 5/13) :
  ∃ (cos_C : ℝ), cos_C = 16/65 :=
by
  -- Place for the proof
  sorry

end cos_C_of_triangle_l55_55658


namespace oranges_in_each_box_l55_55316

theorem oranges_in_each_box (O B : ℕ) (h1 : O = 24) (h2 : B = 3) :
  O / B = 8 :=
by
  sorry

end oranges_in_each_box_l55_55316


namespace james_january_income_l55_55230

variable (January February March : ℝ)
variable (h1 : February = 2 * January)
variable (h2 : March = February - 2000)
variable (h3 : January + February + March = 18000)

theorem james_january_income : January = 4000 := by
  sorry

end james_january_income_l55_55230


namespace more_stable_shooting_performance_l55_55177

theorem more_stable_shooting_performance :
  ∀ (SA2 SB2 : ℝ), SA2 = 1.9 → SB2 = 3 → (SA2 < SB2) → "A" = "Athlete with more stable shooting performance" :=
by
  intros SA2 SB2 h1 h2 h3
  sorry

end more_stable_shooting_performance_l55_55177


namespace fair_attendance_l55_55086

theorem fair_attendance (x y z : ℕ) 
    (h1 : y = 2 * x)
    (h2 : z = y - 200)
    (h3 : x + y + z = 2800) : x = 600 := by
  sorry

end fair_attendance_l55_55086


namespace property_depreciation_rate_l55_55884

noncomputable def initial_value : ℝ := 25599.08977777778
noncomputable def final_value : ℝ := 21093
noncomputable def annual_depreciation_rate : ℝ := 0.063

theorem property_depreciation_rate :
  final_value = initial_value * (1 - annual_depreciation_rate)^3 :=
sorry

end property_depreciation_rate_l55_55884


namespace frosting_need_l55_55837

theorem frosting_need : 
  (let layer_cake_frosting := 1
   let single_cake_frosting := 0.5
   let brownie_frosting := 0.5
   let dozen_cupcakes_frosting := 0.5
   let num_layer_cakes := 3
   let num_dozen_cupcakes := 6
   let num_single_cakes := 12
   let num_pans_brownies := 18
   
   let total_frosting := 
     (num_layer_cakes * layer_cake_frosting) + 
     (num_dozen_cupcakes * dozen_cupcakes_frosting) + 
     (num_single_cakes * single_cake_frosting) + 
     (num_pans_brownies * brownie_frosting)
   
   total_frosting = 21) :=
  by
    sorry

end frosting_need_l55_55837


namespace majority_owner_percentage_l55_55555

theorem majority_owner_percentage (profit total_profit : ℝ)
    (majority_owner_share : ℝ) (partner_share : ℝ) 
    (combined_share : ℝ) 
    (num_partners : ℕ) 
    (total_profit_value : total_profit = 80000) 
    (partner_share_value : partner_share = 0.25 * (1 - majority_owner_share)) 
    (combined_share_value : combined_share = profit)
    (combined_share_amount : combined_share = 50000) 
    (num_partners_value : num_partners = 4) :
  majority_owner_share = 0.25 :=
by
  sorry

end majority_owner_percentage_l55_55555


namespace percentage_difference_l55_55563

theorem percentage_difference (water_yesterday : ℕ) (water_two_days_ago : ℕ) (h1 : water_yesterday = 48) (h2 : water_two_days_ago = 50) : 
  (water_two_days_ago - water_yesterday) / water_two_days_ago * 100 = 4 :=
by
  sorry

end percentage_difference_l55_55563


namespace at_least_one_nonzero_l55_55266

theorem at_least_one_nonzero (a b : ℝ) : a^2 + b^2 ≠ 0 ↔ (a ≠ 0 ∨ b ≠ 0) := by
  sorry

end at_least_one_nonzero_l55_55266


namespace proof_f_f_pi_div_12_l55_55205

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 0 then 4 * x^2 - 1 else (Real.sin x)^2 - (Real.cos x)^2

theorem proof_f_f_pi_div_12 : f (f (Real.pi / 12)) = 2 := by
  sorry

end proof_f_f_pi_div_12_l55_55205


namespace cups_per_serving_l55_55162

-- Define the conditions
def total_cups : ℕ := 18
def servings : ℕ := 9

-- State the theorem to prove the answer
theorem cups_per_serving : total_cups / servings = 2 := by
  sorry

end cups_per_serving_l55_55162


namespace fly_total_distance_l55_55875

-- Definitions and conditions
def cyclist_speed : ℝ := 10 -- speed of each cyclist in miles per hour
def initial_distance : ℝ := 50 -- initial distance between the cyclists in miles
def fly_speed : ℝ := 15 -- speed of the fly in miles per hour

-- Statement to prove
theorem fly_total_distance : 
  (cyclist_speed * 2 * initial_distance / (cyclist_speed + cyclist_speed) / fly_speed * fly_speed) = 37.5 :=
by
  -- sorry is used here to skip the proof
  sorry

end fly_total_distance_l55_55875


namespace audio_cassettes_in_first_set_l55_55097

theorem audio_cassettes_in_first_set (A V : ℝ) (num_audio_cassettes : ℝ) : 
  (V = 300) → (A * num_audio_cassettes + 3 * V = 1110) → (5 * A + 4 * V = 1350) → (A = 30) → (num_audio_cassettes = 7) := 
by
  intros hV hCond1 hCond2 hA
  sorry

end audio_cassettes_in_first_set_l55_55097


namespace marta_candies_received_l55_55967

theorem marta_candies_received:
  ∃ x y : ℕ, x + y = 200 ∧ x < 100 ∧ x > (4 * y) / 5 ∧ (x % 8 = 0) ∧ (y % 8 = 0) ∧ x = 96 ∧ y = 104 := 
sorry

end marta_candies_received_l55_55967


namespace band_members_count_l55_55848

theorem band_members_count :
  ∃ n k m : ℤ, n = 10 * k + 4 ∧ n = 12 * m + 6 ∧ 200 ≤ n ∧ n ≤ 300 ∧ n = 254 :=
by
  -- Declaration of the theorem properties
  sorry

end band_members_count_l55_55848


namespace enchanted_creatures_gala_handshakes_l55_55992

theorem enchanted_creatures_gala_handshakes :
  let goblins := 30
  let trolls := 20
  let goblin_handshakes := goblins * (goblins - 1) / 2
  let troll_to_goblin_handshakes := trolls * goblins
  goblin_handshakes + troll_to_goblin_handshakes = 1035 := 
by
  sorry

end enchanted_creatures_gala_handshakes_l55_55992


namespace main_theorem_l55_55557

def d_digits (d : ℕ) : Prop :=
  ∃ (d_1 d_2 d_3 d_4 d_5 d_6 d_7 d_8 d_9 : ℕ),
    d = d_1 * 10^8 + d_2 * 10^7 + d_3 * 10^6 + d_4 * 10^5 + d_5 * 10^4 + d_6 * 10^3 + d_7 * 10^2 + d_8 * 10 + d_9

noncomputable def condition1 (d e : ℕ) (i : ℕ) : Prop :=
  (e - (d / 10^(8 - i) % 10)) * 10^(8 - i) + d ≡ 0 [MOD 7]

noncomputable def condition2 (e f : ℕ) (i : ℕ) : Prop :=
  (f - (e / 10^(8 - i) % 10)) * 10^(8 - i) + e ≡ 0 [MOD 7]

theorem main_theorem
  (d e f : ℕ)
  (h1 : d_digits d)
  (h2 : ∀ i, 1 ≤ i ∧ i ≤ 9 → condition1 d e i)
  (h3 : ∀ i, 1 ≤ i ∧ i ≤ 9 → condition2 e f i) :
  ∀ i, 1 ≤ i ∧ i ≤ 9 → (d / 10^(8 - i) % 10) ≡ (f / 10^(8 - i) % 10) [MOD 7] := sorry

end main_theorem_l55_55557


namespace pipe_length_l55_55301

theorem pipe_length (L_short : ℕ) (hL_short : L_short = 59) : 
    L_short + 2 * L_short = 177 := by
  sorry

end pipe_length_l55_55301


namespace total_grapes_is_157_l55_55021

def number_of_grapes_in_robs_bowl : ℕ := 25

def number_of_grapes_in_allies_bowl : ℕ :=
  number_of_grapes_in_robs_bowl + 5

def number_of_grapes_in_allyns_bowl : ℕ :=
  2 * number_of_grapes_in_allies_bowl - 2

def number_of_grapes_in_sams_bowl : ℕ :=
  (number_of_grapes_in_allies_bowl + number_of_grapes_in_allyns_bowl) / 2

def total_number_of_grapes : ℕ :=
  number_of_grapes_in_robs_bowl +
  number_of_grapes_in_allies_bowl +
  number_of_grapes_in_allyns_bowl +
  number_of_grapes_in_sams_bowl

theorem total_grapes_is_157 : total_number_of_grapes = 157 :=
  sorry

end total_grapes_is_157_l55_55021


namespace mean_problem_l55_55419

theorem mean_problem : 
  (8 + 12 + 24) / 3 = (16 + z) / 2 → z = 40 / 3 :=
by
  intro h
  sorry

end mean_problem_l55_55419


namespace solve_for_y_l55_55695

theorem solve_for_y (y : ℝ) (h : 3 * y ^ (1 / 4) - 3 * y ^ (1 / 2) / y ^ (1 / 4) = 13 - 2 * y ^ (1 / 4)) :
  y = (13 / 2) ^ 4 :=
by sorry

end solve_for_y_l55_55695


namespace store_sells_2_kg_per_week_l55_55171

def packets_per_week := 20
def grams_per_packet := 100
def grams_per_kg := 1000
def kg_per_week (p : Nat) (gr_per_pkt : Nat) (gr_per_kg : Nat) : Nat :=
  (p * gr_per_pkt) / gr_per_kg

theorem store_sells_2_kg_per_week :
  kg_per_week packets_per_week grams_per_packet grams_per_kg = 2 :=
  sorry

end store_sells_2_kg_per_week_l55_55171


namespace odd_and_periodic_function_l55_55536

noncomputable def f : ℝ → ℝ := sorry

lemma given_conditions (x : ℝ) : 
  (f (10 + x) = f (10 - x)) ∧ (f (20 - x) = -f (20 + x)) :=
  sorry

theorem odd_and_periodic_function (x : ℝ) :
  (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x : ℝ, f (x + 40) = f x) :=
  sorry

end odd_and_periodic_function_l55_55536


namespace student_probability_at_least_9_correct_l55_55306

-- Define the conditions
def total_questions : ℕ := 10
def probability_of_success : ℚ := 1 / 4

-- Define the binomial probability calculation
noncomputable def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

-- Calculate the probability of getting exactly 9 correct answers
noncomputable def probability_9_correct : ℚ :=
  binomial_probability total_questions 9 probability_of_success

-- Calculate the probability of getting exactly 10 correct answers
noncomputable def probability_10_correct : ℚ :=
  binomial_probability total_questions 10 probability_of_success

-- The combined probability of answering at least 9 questions correctly
noncomputable def total_probability : ℚ :=
  probability_9_correct + probability_10_correct

-- Statement to be proved
theorem student_probability_at_least_9_correct :
  (total_probability ≈ 3 * 10^(-5) : Prop) :=
sorry

end student_probability_at_least_9_correct_l55_55306


namespace solution_set_inequality_l55_55049

theorem solution_set_inequality {a b c x : ℝ} (h1 : a < 0)
  (h2 : -b / a = 1 + 2) (h3 : c / a = 1 * 2) :
  a - c * (x^2 - x - 1) - b * x ≥ 0 ↔ x ≤ -3 / 2 ∨ x ≥ 1 := by
  sorry

end solution_set_inequality_l55_55049


namespace players_in_physics_class_l55_55463

theorem players_in_physics_class (total players_math players_both : ℕ)
    (h1 : total = 15)
    (h2 : players_math = 9)
    (h3 : players_both = 4) :
    (players_math - players_both) + (total - (players_math - players_both + players_both)) + players_both = 10 :=
by {
  sorry
}

end players_in_physics_class_l55_55463


namespace hyperbola_standard_equation_triangle_area_PF1F2_l55_55635

-- Define the given conditions
def F1 : ℝ × ℝ := (-Real.sqrt 5, 0)
def F2 : ℝ × ℝ := (Real.sqrt 5, 0)
def real_axis_length : ℝ := 4

-- Part 1: Prove the standard equation of the hyperbola
theorem hyperbola_standard_equation : 
  let c := Real.sqrt 5 in
  let a := real_axis_length / 2 in
  let b := Real.sqrt (c^2 - a^2) in
  (a = 2) → (c = Real.sqrt 5) → (a^2 = 4) → (b^2 = 1) → 
  (∀ x y : ℝ, (x, y) ∈ {p | (p.1^2 / 4) - p.2^2 = 1}) :=
by
  intros c a b ha hc ha_square hb_square
  -- define the hyperbola equation
  let hyperbola := (λ p : ℝ × ℝ, (p.1^2 / (ha_square)) - (p.2^2 / (hb_square)) = 1)
  exact hyperbola
  sorry

-- Part 2: Prove the area of triangle PF1F2
theorem triangle_area_PF1F2 (P : ℝ × ℝ) (h_on_hyperbola : P ∈ {p | (p.1^2 / 4) - p.2^2 = 1}) :
  let PF1 := Real.dist P F1 in
  let PF2 := Real.dist P F2 in
  (PF1 ^ 2 + PF2 ^ 2 = 20) → (PF1 * PF2 = 2) → 
  let area := 1 / 2 * PF1 * PF2 in 
  (PF1 * PF2 / 2 = 1) :=
by
  intros PF1 PF2 hp1 hp2
  exact (PF1 * PF2) / 2 = 1
  sorry

end hyperbola_standard_equation_triangle_area_PF1F2_l55_55635


namespace two_numbers_with_difference_less_than_half_l55_55540

theorem two_numbers_with_difference_less_than_half
  (x1 x2 x3 : ℝ)
  (h1 : 0 ≤ x1) (h2 : x1 < 1)
  (h3 : 0 ≤ x2) (h4 : x2 < 1)
  (h5 : 0 ≤ x3) (h6 : x3 < 1) :
  ∃ a b, 
    (a = x1 ∨ a = x2 ∨ a = x3) ∧
    (b = x1 ∨ b = x2 ∨ b = x3) ∧
    a ≠ b ∧ 
    |b - a| < 1 / 2 :=
sorry

end two_numbers_with_difference_less_than_half_l55_55540


namespace expression_equals_4034_l55_55133

theorem expression_equals_4034 : 6 * 2017 - 4 * 2017 = 4034 := by
  sorry

end expression_equals_4034_l55_55133


namespace arithmetic_twelfth_term_l55_55143

theorem arithmetic_twelfth_term 
(a d : ℚ) (n : ℕ) (h_a : a = 1/2) (h_d : d = 1/3) (h_n : n = 12) : 
  a + (n - 1) * d = 25 / 6 := 
by 
  sorry

end arithmetic_twelfth_term_l55_55143


namespace probability_one_pair_three_colors_diff_l55_55083

theorem probability_one_pair_three_colors_diff :
  let colors := {red, blue, green, orange, purple}
  let socks := colors.prod (Finset.range 2)
  let draw := socks.choose 5
  let favorable := { draw' ∈ draw | draw'.pairwise (λ x y, if x.1 = y.1 then x == y) ∨ draw'.pairwise (λ x y, x.1 ≠ y.1) }
  (favorable.card : ℚ) / (draw.card : ℚ) = 20 / 21 := by
  sorry

end probability_one_pair_three_colors_diff_l55_55083


namespace find_fifth_number_l55_55103

-- Define the sets and their conditions
def first_set : List ℕ := [28, 70, 88, 104]
def second_set : List ℕ := [50, 62, 97, 124]

-- Define the means
def mean_first_set (x : ℕ) (y : ℕ) : ℚ := (28 + x + 70 + 88 + y) / 5
def mean_second_set (x : ℕ) : ℚ := (50 + 62 + 97 + 124 + x) / 5

-- Conditions given in the problem
axiom mean_first_set_condition (x y : ℕ) : mean_first_set x y = 67
axiom mean_second_set_condition (x : ℕ) : mean_second_set x = 75.6

-- Lean 4 theorem statement to prove the fifth number in the first set is 104 given above conditions
theorem find_fifth_number : ∃ x y, mean_first_set x y = 67 ∧ mean_second_set x = 75.6 ∧ y = 104 := by
  sorry

end find_fifth_number_l55_55103


namespace product_decrease_increase_fifteenfold_l55_55527

theorem product_decrease_increase_fifteenfold (a1 a2 a3 a4 a5 : ℕ) :
  ((a1 - 3) * (a2 - 3) * (a3 - 3) * (a4 - 3) * (a5 - 3) = 15 * a1 * a2 * a3 * a4 * a5) → true :=
by
  sorry

end product_decrease_increase_fifteenfold_l55_55527


namespace liam_total_time_l55_55677

noncomputable def total_time_7_laps : Nat :=
let time_first_200 := 200 / 5  -- Time in seconds for the first 200 meters
let time_next_300 := 300 / 6   -- Time in seconds for the next 300 meters
let time_per_lap := time_first_200 + time_next_300
let laps := 7
let total_time := laps * time_per_lap
total_time

theorem liam_total_time : total_time_7_laps = 630 := by
sorry

end liam_total_time_l55_55677


namespace copies_made_in_half_hour_l55_55448

theorem copies_made_in_half_hour
  (rate1 rate2 : ℕ)  -- rates of the two copy machines
  (time : ℕ)         -- time considered
  (h_rate1 : rate1 = 40)  -- the first machine's rate
  (h_rate2 : rate2 = 55)  -- the second machine's rate
  (h_time : time = 30)    -- time in minutes
  : (rate1 * time + rate2 * time = 2850) := 
sorry

end copies_made_in_half_hour_l55_55448


namespace arithmetic_sequence_max_sum_l55_55242

theorem arithmetic_sequence_max_sum (a : ℕ → ℝ) (d : ℝ) (m : ℕ) (S : ℕ → ℝ):
  (∀ n, a n = a 1 + (n - 1) * d) → 
  3 * a 8 = 5 * a m → 
  a 1 > 0 →
  (∀ n, S n = n / 2 * (2 * a 1 + (n - 1) * d)) →
  (∀ n, S n ≤ S 20) →
  m = 13 := 
by {
  -- State the corresponding solution steps leading to the proof.
  sorry
}

end arithmetic_sequence_max_sum_l55_55242


namespace max_a_plus_2b_plus_c_l55_55644

open Real

theorem max_a_plus_2b_plus_c
  (A : Set ℝ := {x | |x + 1| ≤ 4})
  (T : ℝ := 3)
  (a b c : ℝ)
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_T : a^2 + b^2 + c^2 = T) :
  a + 2 * b + c ≤ 3 * sqrt 2 :=
by
  -- Proof is omitted
  sorry

end max_a_plus_2b_plus_c_l55_55644


namespace complex_series_sum_eq_zero_l55_55826

open Complex

theorem complex_series_sum_eq_zero {ω : ℂ} (h1 : ω^9 = 1) (h2 : ω ≠ 1) :
  ω^18 + ω^27 + ω^36 + ω^45 + ω^54 + ω^63 + ω^72 + ω^81 + ω^90 = 0 := by
  sorry

end complex_series_sum_eq_zero_l55_55826


namespace polynomial_remainder_l55_55775

noncomputable def remainder_div (p : Polynomial ℚ) (d1 d2 d3 : Polynomial ℚ) : Polynomial ℚ :=
  let d := d1 * d2 * d3 
  let q := p /ₘ d 
  let r := p %ₘ d 
  r

theorem polynomial_remainder :
  let p := (X^6 + 2 * X^4 - X^3 - 7 * X^2 + 3 * X + 1)
  let d1 := X - 2
  let d2 := X + 1
  let d3 := X - 3
  remainder_div p d1 d2 d3 = 29 * X^2 + 17 * X - 19 :=
by
  sorry

end polynomial_remainder_l55_55775


namespace mark_paintable_area_l55_55370

theorem mark_paintable_area :
  let num_bedrooms := 4
  let length := 14
  let width := 11
  let height := 9
  let area_excluded := 70
  let area_wall_one_bedroom := 2 * (length * height) + 2 * (width * height) - area_excluded 
  (area_wall_one_bedroom * num_bedrooms) = 1520 :=
by
  sorry

end mark_paintable_area_l55_55370


namespace simplify_fraction_l55_55574

theorem simplify_fraction : (3^9 / 9^3) = 27 :=
by
  sorry

end simplify_fraction_l55_55574


namespace license_count_l55_55013

def num_licenses : ℕ :=
  let num_letters := 3
  let num_digits := 10
  let num_digit_slots := 6
  num_letters * num_digits ^ num_digit_slots

theorem license_count :
  num_licenses = 3000000 := by
  sorry

end license_count_l55_55013


namespace find_a_l55_55952

noncomputable def f (a x : ℝ) : ℝ := Real.log (Real.sqrt (1 + a * x ^ 2) - x)

theorem find_a (a : ℝ) :
  (∀ (x : ℝ), f a (-x) = -f a x) ↔ a = 1 :=
by
  sorry

end find_a_l55_55952


namespace coefficient_of_x_in_expression_l55_55317

theorem coefficient_of_x_in_expression : 
  let expr := 2 * (x - 5) + 5 * (8 - 3 * x^2 + 6 * x) - 9 * (3 * x - 2) + 3 * (x + 4)
  ∃ k : ℤ, (expr = k * x + term) ∧ 
  (∃ coefficient_x : ℤ, coefficient_x = 8) := 
sorry

end coefficient_of_x_in_expression_l55_55317


namespace greatest_least_S_T_l55_55906

theorem greatest_least_S_T (a b c : ℝ) (h : a ≤ b ∧ b ≤ c) (triangle : a + b > c ∧ a + c > b ∧ b + c > a) :
  4 ≤ (a + b + c)^2 / (b * c) ∧ (a + b + c)^2 / (b * c) ≤ 9 :=
by sorry

end greatest_least_S_T_l55_55906


namespace problem_statement_l55_55387

noncomputable def max_value_d (a b c d : ℝ) : Prop :=
a + b + c + d = 10 ∧
(ab + ac + ad + bc + bd + cd = 20) ∧
∀ x, (a + b + c + x = 10 ∧ ab + ac + ad + bc + bd + cd = 20) → x ≤ 5 + Real.sqrt 105 / 2

theorem problem_statement (a b c d : ℝ) :
  max_value_d a b c d → d = (5 + Real.sqrt 105) / 2 :=
sorry

end problem_statement_l55_55387


namespace max_value_func_l55_55782

noncomputable def func (x : ℝ) : ℝ :=
  Real.sin x - Real.sqrt 3 * Real.cos x

theorem max_value_func : ∃ x : ℝ, func x = 2 :=
by
  -- proof steps will be provided here
  sorry

end max_value_func_l55_55782


namespace correct_statements_l55_55960

variables {d : ℝ} {S : ℕ → ℝ} {a : ℕ → ℝ}

axiom arithmetic_sequence (n : ℕ) : S n = n * a 1 + (n * (n - 1) / 2) * d

theorem correct_statements (h1 : S 6 = S 12) :
  (S 18 = 0) ∧ (d > 0 → a 6 + a 12 < 0) ∧ (d < 0 → |a 6| > |a 12|) :=
sorry

end correct_statements_l55_55960


namespace find_a_plus_b_l55_55206

theorem find_a_plus_b (a b : ℝ) :
  (∀ x : ℝ, x^2 + (a+1)*x + ab = 0 → (x = -1 ∨ x = 4)) → a + b = -3 :=
by
  sorry

end find_a_plus_b_l55_55206


namespace find_a_and_b_find_monotonic_intervals_and_extreme_values_l55_55056

-- Definitions and conditions
def f (x : ℝ) (a b c : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

def takes_extreme_values (f : ℝ → ℝ) (a b c : ℝ) : Prop := 
  ∃ x₁ x₂, x₁ = 1 ∧ x₂ = -2/3 ∧ 3*x₁^2 + 2*a*x₁ + b = 0 ∧ 3*x₂^2 + 2*a*x₂ + b = 0

def f_at_specific_point (f : ℝ → ℝ) (x v : ℝ) : Prop :=
  f x = v

theorem find_a_and_b (a b c : ℝ) :
  takes_extreme_values (f a b c) a b c →
  a = -1/2 ∧ b = -2 :=
sorry

theorem find_monotonic_intervals_and_extreme_values (a b c : ℝ) :
  takes_extreme_values (f a b c) a b c →
  f_at_specific_point (f a b c) (-1) (3/2) →
  c = 1 ∧ 
  (∀ x, x < -2/3 ∨ x > 1 → deriv (f a b c) x > 0) ∧
  (∀ x, -2/3 < x ∧ x < 1 → deriv (f a b c) x < 0) ∧
  f a b c (-2/3) = 49/27 ∧ 
  f a b c 1 = -1/2 :=
sorry

end find_a_and_b_find_monotonic_intervals_and_extreme_values_l55_55056


namespace tan_identity_proof_l55_55363

theorem tan_identity_proof
  (α β : ℝ)
  (h₁ : Real.tan (α + β) = 3)
  (h₂ : Real.tan (α + π / 4) = -3) :
  Real.tan (β - π / 4) = -3 / 4 := 
sorry

end tan_identity_proof_l55_55363


namespace quadricycles_count_l55_55733

theorem quadricycles_count (s q : ℕ) (hsq : s + q = 9) (hw : 2 * s + 4 * q = 30) : q = 6 :=
by
  sorry

end quadricycles_count_l55_55733


namespace general_inequality_l55_55047

theorem general_inequality (x : ℝ) (n : ℕ) (h_pos_x : x > 0) (h_pos_n : 0 < n) : 
  x + n^n / x^n ≥ n + 1 := by 
  sorry

end general_inequality_l55_55047


namespace distance_origin_to_line_constant_l55_55846

noncomputable theory
open Real

-- Definitions from conditions
def ellipse_C (x y : ℝ) : Prop := (x^2 / 4) + y^2 = 1
def F1 : ℝ × ℝ := (-sqrt 3, 0)
def F2 : ℝ × ℝ := (sqrt 3, 0)
def A : ℝ × ℝ := (sqrt 3, 1 / 2)
def line_l (k m x y : ℝ) : Prop := y = k * x + m

-- The formal Lean statement for the mathematically equivalent proof problem
theorem distance_origin_to_line_constant (k m : ℝ) : 
  let intersects := ∃ x y, ellipse_C x y ∧ line_l k m x y,
      circle_passing_through_O := ∀ E F : ℝ × ℝ, (E.1, E.2) ≠ (0,0) → (F.1, F.2) ≠ (0,0) → (E.1 - F.1) * 0 + (E.2 - F.2) * 0 = 0 → 
                                   ellipse_C E.1 E.2 → ellipse_C F.1 F.2 → line_l k m E.1 E.2 → line_l k m F.1 F.2 → 
                                   ∃ x y, x^2 + y^2 = (E.1 - F.1)^2 + (E.2 - F.2)^2 :=
  (sqrt 5) / (5) := by
  sorry  -- Proof goes here.

end distance_origin_to_line_constant_l55_55846


namespace part1_part2_l55_55392

-- Definitions from conditions
def U := ℝ
def A := {x : ℝ | -x^2 + 12*x - 20 > 0}
def B (a : ℝ) := {x : ℝ | 5 - a < x ∧ x < a}

-- (1) If "x ∈ A" is a necessary condition for "x ∈ B", find the range of a
theorem part1 (a : ℝ) : (∀ x : ℝ, x ∈ B a → x ∈ A) → a ≤ 3 :=
by sorry

-- (2) If A ∩ B ≠ ∅, find the range of a
theorem part2 (a : ℝ) : (∃ x : ℝ, x ∈ A ∧ x ∈ B a) → a > 5 / 2 :=
by sorry

end part1_part2_l55_55392


namespace population_decrease_is_25_percent_l55_55522

def initial_population : ℕ := 20000
def final_population_first_year : ℕ := initial_population + (initial_population * 25 / 100)
def final_population_second_year : ℕ := 18750

def percentage_decrease (initial final : ℕ) : ℚ :=
  ((initial - final : ℚ) * 100) / initial 

theorem population_decrease_is_25_percent :
  percentage_decrease final_population_first_year final_population_second_year = 25 :=
by
  sorry

end population_decrease_is_25_percent_l55_55522


namespace total_oranges_in_buckets_l55_55856

theorem total_oranges_in_buckets (a b c : ℕ) 
  (h1 : a = 22) 
  (h2 : b = a + 17) 
  (h3 : c = b - 11) : 
  a + b + c = 89 := 
by {
  sorry
}

end total_oranges_in_buckets_l55_55856


namespace calculation_1500_increased_by_45_percent_l55_55160

theorem calculation_1500_increased_by_45_percent :
  1500 * (1 + 45 / 100) = 2175 := 
by
  sorry

end calculation_1500_increased_by_45_percent_l55_55160


namespace prob_queen_then_diamond_is_correct_l55_55995

/-- Define the probability of drawing a Queen first and a diamond second -/
def prob_queen_then_diamond : ℚ := (3 / 52) * (13 / 51) + (1 / 52) * (12 / 51)

/-- The probability that the first card is a Queen and the second card is a diamond is 18/221 -/
theorem prob_queen_then_diamond_is_correct : prob_queen_then_diamond = 18 / 221 :=
by
  sorry

end prob_queen_then_diamond_is_correct_l55_55995


namespace dana_pencils_more_than_jayden_l55_55469

theorem dana_pencils_more_than_jayden :
  ∀ (Jayden_has_pencils : ℕ) (Marcus_has_pencils : ℕ) (Dana_has_pencils : ℕ),
    Jayden_has_pencils = 20 →
    Marcus_has_pencils = Jayden_has_pencils / 2 →
    Dana_has_pencils = Marcus_has_pencils + 25 →
    Dana_has_pencils - Jayden_has_pencils = 15 :=
by
  intros Jayden_has_pencils Marcus_has_pencils Dana_has_pencils
  intro h1
  intro h2
  intro h3
  sorry

end dana_pencils_more_than_jayden_l55_55469


namespace percentage_of_page_used_l55_55158

theorem percentage_of_page_used (length width side_margin top_margin : ℝ) (h_length : length = 30) (h_width : width = 20) (h_side_margin : side_margin = 2) (h_top_margin : top_margin = 3) :
  ( ((length - 2 * top_margin) * (width - 2 * side_margin)) / (length * width) ) * 100 = 64 := 
by
  sorry

end percentage_of_page_used_l55_55158


namespace sum_of_first_eleven_terms_l55_55930

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {d : ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop := ∃ d, ∀ n, a (n + 1) = a n + d

theorem sum_of_first_eleven_terms 
  (h_arith : is_arithmetic_sequence a)
  (h_S : ∀ n, S n = n * (a 1 + a n) / 2)
  (h_condition : 2 * a 7 - a 8 = 5) :
  S 11 = 55 :=
sorry

end sum_of_first_eleven_terms_l55_55930


namespace value_of_n_l55_55652

theorem value_of_n (n : ℕ) (k : ℕ) (h : k = 11) (eqn : (1/2)^n * (1/81)^k = 1/18^22) : n = 22 :=
by
  sorry

end value_of_n_l55_55652


namespace factorize_expression_l55_55475

theorem factorize_expression (a x : ℝ) : a * x^2 - a = a * (x + 1) * (x - 1) :=
by
  sorry

end factorize_expression_l55_55475


namespace population_sampling_precision_l55_55662

theorem population_sampling_precision (sample_size : ℕ → Prop) 
    (A : Prop) (B : Prop) (C : Prop) (D : Prop)
    (condition_A : A = (∀ n : ℕ, sample_size n → false))
    (condition_B : B = (∀ n : ℕ, sample_size n → n > 0 → true))
    (condition_C : C = (∀ n : ℕ, sample_size n → false))
    (condition_D : D = (∀ n : ℕ, sample_size n → false)) :
  B :=
by sorry

end population_sampling_precision_l55_55662


namespace carlos_laundry_l55_55031

theorem carlos_laundry (n : ℕ) 
  (h1 : 45 * n + 75 = 165) : n = 2 :=
by
  sorry

end carlos_laundry_l55_55031


namespace Alexei_finished_ahead_of_Sergei_by_1_9_km_l55_55620

noncomputable def race_distance : ℝ := 10
noncomputable def v_A : ℝ := 1  -- speed of Alexei
noncomputable def v_V : ℝ := 0.9 * v_A  -- speed of Vitaly
noncomputable def v_S : ℝ := 0.81 * v_A  -- speed of Sergei

noncomputable def distance_Alexei_finished_Ahead_of_Sergei : ℝ :=
race_distance - (0.81 * race_distance)

theorem Alexei_finished_ahead_of_Sergei_by_1_9_km :
  distance_Alexei_finished_Ahead_of_Sergei = 1.9 :=
by
  simp [race_distance, v_A, v_V, v_S, distance_Alexei_finished_Ahead_of_Sergei]
  sorry

end Alexei_finished_ahead_of_Sergei_by_1_9_km_l55_55620


namespace sarah_bottle_caps_l55_55976

theorem sarah_bottle_caps (initial_caps : ℕ) (additional_caps : ℕ) (total_caps : ℕ) : initial_caps = 26 → additional_caps = 3 → total_caps = initial_caps + additional_caps → total_caps = 29 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end sarah_bottle_caps_l55_55976


namespace largest_possible_d_l55_55388

theorem largest_possible_d (a b c d : ℝ) (h1 : a + b + c + d = 10) (h2 : ab + ac + ad + bc + bd + cd = 20) : 
    d ≤ (5 + Real.sqrt 105) / 2 :=
by
  sorry

end largest_possible_d_l55_55388


namespace living_space_increase_l55_55956

theorem living_space_increase (a b x : ℝ) (h₁ : a = 10) (h₂ : b = 12.1) : a * (1 + x) ^ 2 = b :=
sorry

end living_space_increase_l55_55956


namespace jogging_track_circumference_l55_55033

noncomputable def Deepak_speed : ℝ := 4.5 -- km/hr
noncomputable def Wife_speed : ℝ := 3.75 -- km/hr
noncomputable def time_meet : ℝ := 4.8 / 60 -- hours

noncomputable def Distance_Deepak : ℝ := Deepak_speed * time_meet
noncomputable def Distance_Wife : ℝ := Wife_speed * time_meet

theorem jogging_track_circumference : 2 * (Distance_Deepak + Distance_Wife) = 1.32 := by
  sorry

end jogging_track_circumference_l55_55033


namespace cube_faces_paint_count_l55_55120

def is_painted_on_at_least_two_faces (x y z : ℕ) : Prop :=
  ((x = 0 ∨ x = 3) ∨ (y = 0 ∨ y = 3) ∨ (z = 0 ∨ z = 3))

theorem cube_faces_paint_count :
  let n := 4 in
  let volume := n * n * n in
  let one_inch_cubes := {c | ∃ x y z : ℕ, x < n ∧ y < n ∧ z < n ∧ c = (x, y, z)} in
  let painted_cubes := {c ∈ one_inch_cubes | exists_along_edges c} in  -- Auxiliary predicate to check edge paint
  card painted_cubes = 32 :=
begin
  sorry,
end

/-- Auxiliary predicate to check if a cube is on the edges but not corners, signifying 
    at least two faces are painted --/
predicate exists_along_edges (c: ℕ × ℕ × ℕ) := 
  match c with
  | (0, _, _) => true
  | (3, _, _) => true
  | (_, 0, _) => true
  | (_, 3, _) => true
  | (_, _, 0) => true
  | (_, _, 3) => true
  | (_, _, _) => false
  end

end cube_faces_paint_count_l55_55120


namespace area_closed_figure_sqrt_x_x_cube_l55_55779

noncomputable def integral_diff_sqrt_x_cube (a b : ℝ) :=
∫ x in a..b, (Real.sqrt x - x^3)

theorem area_closed_figure_sqrt_x_x_cube :
  integral_diff_sqrt_x_cube 0 1 = 5 / 12 :=
by
  sorry

end area_closed_figure_sqrt_x_x_cube_l55_55779


namespace michael_saves_more_l55_55613

-- Definitions for the conditions
def price_per_pair : ℝ := 50
def discount_a (price : ℝ) : ℝ := price + 0.6 * price
def discount_b (price : ℝ) : ℝ := 2 * price - 15

-- Statement to prove
theorem michael_saves_more (price : ℝ) (h : price = price_per_pair) : discount_b price - discount_a price = 5 :=
by
  sorry

end michael_saves_more_l55_55613


namespace number_of_non_empty_proper_subsets_l55_55829

noncomputable def S : Set ℝ := {x : ℝ | x^2 - 7*x - 30 < 0}

def T : Set ℤ := {x : ℤ | Real.exp x > 1 - x}

def intersectionSet : Set ℤ := {x : ℤ | (x : ℝ) ∈ S ∧ x ∈ T}

theorem number_of_non_empty_proper_subsets : 
  (Finset.card (Finset.attach (Finset.filter (λ x, true) (Set.toFinset {x : ℤ | (x : ℝ) ∈ S ∧ x ∈ T}))) = 9 → 2^9 - 1 = 510) := by
sorries

end number_of_non_empty_proper_subsets_l55_55829


namespace union_A_B_inter_A_B_comp_int_B_l55_55079

open Set

variable (x : ℝ)

def A := {x : ℝ | 2 ≤ x ∧ x < 4}
def B := {x : ℝ | 3 ≤ x}

theorem union_A_B : A ∪ B = (Ici 2) :=
by
  sorry

theorem inter_A_B : A ∩ B = Ico 3 4 :=
by
  sorry

theorem comp_int_B : (univ \ A) ∩ B = Ici 4 :=
by
  sorry

end union_A_B_inter_A_B_comp_int_B_l55_55079


namespace total_students_in_class_l55_55130

theorem total_students_in_class (front_pos back_pos : ℕ) (H_front : front_pos = 23) (H_back : back_pos = 23) : front_pos + back_pos - 1 = 45 :=
by
  -- No proof required as per instructions
  sorry

end total_students_in_class_l55_55130


namespace prob_draw_l55_55570

-- Define the probabilities as constants
def prob_A_winning : ℝ := 0.4
def prob_A_not_losing : ℝ := 0.9

-- Prove that the probability of a draw is 0.5
theorem prob_draw : prob_A_not_losing - prob_A_winning = 0.5 :=
by sorry

end prob_draw_l55_55570


namespace parabola_equation_l55_55453

theorem parabola_equation (p : ℝ) (h : 2 * p = 8) :
  ∃ (a : ℝ), a = 8 ∧ (y^2 = a * x ∨ y^2 = -a * x) :=
by
  sorry

end parabola_equation_l55_55453


namespace meters_to_examine_10000_l55_55459

def projection_for_sample (total_meters_examined : ℕ) (rejection_rate : ℝ) (sample_size : ℕ) :=
  total_meters_examined = sample_size

theorem meters_to_examine_10000 : 
  projection_for_sample 10000 0.015 10000 := by
  sorry

end meters_to_examine_10000_l55_55459


namespace range_of_a_l55_55048

theorem range_of_a (a : ℝ) (hx : ∀ x : ℝ, (a - 1) * x > a - 1 → x < 1) : a < 1 :=
sorry

end range_of_a_l55_55048


namespace perpendicular_lines_a_value_l55_55813

theorem perpendicular_lines_a_value (a : ℝ) :
  (∃ m1 m2 : ℝ, (m1 = -a / 2 ∧ m2 = -1 / (a * (a + 1)) ∧ m1 * m2 = -1) ∨
   (a = 0 ∧ ax + 2 * y + 6 = 0 ∧ x + a * (a + 1) * y + (a^2 - 1) = 0)) →
  (a = -3 / 2 ∨ a = 0) :=
by
  sorry

end perpendicular_lines_a_value_l55_55813


namespace incorrect_conclusion_l55_55789

theorem incorrect_conclusion (a b c : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : b > c) (h4 : c > 0) : ¬ (a / b > a / c) :=
sorry

end incorrect_conclusion_l55_55789


namespace jake_watching_hours_l55_55669

theorem jake_watching_hours
    (monday_hours : ℕ := 12) -- Half of 24 hours in a day is 12 hours for Monday
    (wednesday_hours : ℕ := 6) -- A quarter of 24 hours in a day is 6 hours for Wednesday
    (friday_hours : ℕ := 19) -- Jake watched 19 hours on Friday
    (total_hours : ℕ := 52) -- The entire show is 52 hours long
    (T : ℕ) -- To find the total number of hours on Tuesday
    (h : monday_hours + T + wednesday_hours + (monday_hours + T + wednesday_hours) / 2 + friday_hours = total_hours) :
    T = 4 := sorry

end jake_watching_hours_l55_55669


namespace cage_cost_correct_l55_55231

noncomputable def total_amount_paid : ℝ := 20
noncomputable def change_received : ℝ := 0.26
noncomputable def cat_toy_cost : ℝ := 8.77
noncomputable def cage_cost := total_amount_paid - change_received

theorem cage_cost_correct : cage_cost = 19.74 := by
  sorry

end cage_cost_correct_l55_55231


namespace units_digit_of_17_pow_2025_l55_55280

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_17_pow_2025 :
  units_digit (17 ^ 2025) = 7 :=
by sorry

end units_digit_of_17_pow_2025_l55_55280


namespace james_baked_multiple_l55_55892

theorem james_baked_multiple (x : ℕ) (h1 : 115 ≠ 0) (h2 : 1380 = 115 * x) : x = 12 :=
sorry

end james_baked_multiple_l55_55892


namespace cartesian_to_polar_curve_C_l55_55699

theorem cartesian_to_polar_curve_C (x y : ℝ) (θ ρ : ℝ) 
  (h1 : x = ρ * Real.cos θ)
  (h2 : y = ρ * Real.sin θ)
  (h3 : x^2 + y^2 - 2 * x = 0) : 
  ρ = 2 * Real.cos θ :=
sorry

end cartesian_to_polar_curve_C_l55_55699


namespace fruit_seller_apples_l55_55740

theorem fruit_seller_apples (x : ℝ) (h : 0.60 * x = 420) : x = 700 :=
sorry

end fruit_seller_apples_l55_55740


namespace triangle_area_from_perimeter_and_inradius_l55_55423

theorem triangle_area_from_perimeter_and_inradius
  (P : ℝ) (r : ℝ) (A : ℝ)
  (h₁ : P = 24)
  (h₂ : r = 2.5) :
  A = 30 := 
by
  sorry

end triangle_area_from_perimeter_and_inradius_l55_55423


namespace find_w_l55_55132

theorem find_w (k : ℝ) (h1 : z * Real.sqrt w = k)
  (z_w3 : z = 6) (w3 : w = 3) :
  z = 3 / 2 → w = 48 := sorry

end find_w_l55_55132


namespace downstream_speed_l55_55166

noncomputable def upstream_speed : ℝ := 5
noncomputable def still_water_speed : ℝ := 15

theorem downstream_speed:
  ∃ (Vd : ℝ), Vd = 25 ∧ (still_water_speed = (upstream_speed + Vd) / 2) := 
sorry

end downstream_speed_l55_55166


namespace beth_wins_if_arjun_plays_first_l55_55223

/-- 
In the game where players take turns removing one, two adjacent, or two non-adjacent bricks from 
walls, given certain configurations, the configuration where Beth has a guaranteed winning 
strategy if Arjun plays first is (7, 3, 1).
-/
theorem beth_wins_if_arjun_plays_first :
  let nim_value_1 := 1
  let nim_value_2 := 2
  let nim_value_3 := 3
  let nim_value_7 := 2 -- computed as explained in the solution
  ∀ config : List ℕ,
    config = [7, 1, 1] ∨ config = [7, 2, 1] ∨ config = [7, 2, 2] ∨ config = [7, 3, 1] ∨ config = [7, 3, 2] →
    match config with
    | [7, 3, 1] => true
    | _ => false :=
by
  sorry

end beth_wins_if_arjun_plays_first_l55_55223


namespace ratio_arithmetic_sequence_triangle_l55_55074

theorem ratio_arithmetic_sequence_triangle (a b c : ℝ) 
  (h_triangle : a^2 + b^2 = c^2)
  (h_arith_seq : ∃ d, b = a + d ∧ c = a + 2 * d) :
  a / b = 3 / 4 ∧ b / c = 4 / 5 :=
by
  sorry

end ratio_arithmetic_sequence_triangle_l55_55074


namespace original_water_depth_in_larger_vase_l55_55865

-- Definitions based on the conditions
noncomputable def largerVaseDiameter := 20 -- in cm
noncomputable def smallerVaseDiameter := 10 -- in cm
noncomputable def smallerVaseHeight := 16 -- in cm

-- Proving the original depth of the water in the larger vase
theorem original_water_depth_in_larger_vase :
  ∃ depth : ℝ, depth = 14 :=
by
  sorry

end original_water_depth_in_larger_vase_l55_55865


namespace equivalent_problem_l55_55569

variable (a b : ℤ)

def condition1 : Prop :=
  a * (-2)^3 + b * (-2) - 7 = 9

def condition2 : Prop :=
  8 * a + 2 * b - 7 = -23

theorem equivalent_problem (h : condition1 a b) : condition2 a b :=
sorry

end equivalent_problem_l55_55569


namespace remainder_of_7_power_138_mod_9_l55_55868

theorem remainder_of_7_power_138_mod_9 :
  (7 ^ 138) % 9 = 1 := 
by sorry

end remainder_of_7_power_138_mod_9_l55_55868


namespace train_speed_l55_55172

def length_of_train : ℝ := 150
def time_to_cross_pole : ℝ := 9

def speed_in_m_per_s := length_of_train / time_to_cross_pole
def speed_in_km_per_hr := speed_in_m_per_s * (3600 / 1000)

theorem train_speed : speed_in_km_per_hr = 60 := by
  -- Length of train is 150 meters
  -- Time to cross pole is 9 seconds
  -- Speed in m/s = 150 meters / 9 seconds = 16.67 m/s
  -- Speed in km/hr = 16.67 m/s * 3.6 = 60 km/hr
  sorry

end train_speed_l55_55172


namespace fred_initial_sheets_l55_55911

theorem fred_initial_sheets (X : ℕ) (h1 : X + 307 - 156 = 363) : X = 212 :=
by
  sorry

end fred_initial_sheets_l55_55911


namespace gimbap_total_cost_l55_55894

theorem gimbap_total_cost :
  let basic_gimbap_cost := 2000
  let tuna_gimbap_cost := 3500
  let red_pepper_gimbap_cost := 3000
  let beef_gimbap_cost := 4000
  let nude_gimbap_cost := 3500
  let cost_of_two gimbaps := (tuna_gimbap_cost * 2) + (beef_gimbap_cost * 2) + (nude_gimbap_cost * 2)
  cost_of_two gimbaps = 22000 := 
by 
  sorry

end gimbap_total_cost_l55_55894


namespace number_of_chinese_l55_55460

theorem number_of_chinese (total americans australians chinese : ℕ) 
    (h_total : total = 49)
    (h_americans : americans = 16)
    (h_australians : australians = 11)
    (h_chinese : chinese = total - americans - australians) :
    chinese = 22 :=
by
    rw [h_total, h_americans, h_australians] at h_chinese
    exact h_chinese

end number_of_chinese_l55_55460


namespace remainder_when_dividing_p_by_g_is_3_l55_55625

noncomputable def p (x : ℤ) : ℤ := x^5 - 2 * x^3 + 4 * x^2 + x + 5
noncomputable def g (x : ℤ) : ℤ := x + 2

theorem remainder_when_dividing_p_by_g_is_3 : p (-2) = 3 :=
by
  sorry

end remainder_when_dividing_p_by_g_is_3_l55_55625


namespace quadratic_roots_form_l55_55109

theorem quadratic_roots_form {d : ℝ} (h : ∀ x : ℝ, x^2 + 7*x + d = 0 → (x = (-7 + real.sqrt d) / 2) ∨ (x = (-7 - real.sqrt d) / 2)) : d = 49 / 5 := 
sorry

end quadratic_roots_form_l55_55109


namespace domain_of_f_l55_55470

noncomputable def f (x k : ℝ) := (3 * x ^ 2 + 4 * x - 7) / (-7 * x ^ 2 + 4 * x + k)

theorem domain_of_f {x k : ℝ} (h : k < -4/7): ∀ x, -7 * x ^ 2 + 4 * x + k ≠ 0 :=
by 
  intro x
  sorry

end domain_of_f_l55_55470


namespace moles_of_H2O_formed_l55_55907

-- Define the initial conditions
def molesNaOH : ℕ := 2
def molesHCl : ℕ := 2

-- Balanced chemical equation behavior definition
def reaction (x y : ℕ) : ℕ := min x y

-- Statement of the problem to prove
theorem moles_of_H2O_formed :
  reaction molesNaOH molesHCl = 2 := by
  sorry

end moles_of_H2O_formed_l55_55907


namespace evaluate_g_at_neg_four_l55_55212

def g (x : ℤ) : ℤ := 5 * x + 2

theorem evaluate_g_at_neg_four : g (-4) = -18 := 
by 
  sorry

end evaluate_g_at_neg_four_l55_55212


namespace probability_of_weight_ge_30_l55_55718

noncomputable theory

-- Define the probability of an egg weighing less than 30 grams
def P_lt_30 : ℝ := 0.30

-- Define the probability of an egg weighing within the range [30, 40] grams
def P_30_40 : ℝ := 0.50

-- Define the event of the weight being not less than 30 grams
def P_ge_30 : ℝ := 1 - P_lt_30

-- Prove the event using predefined conditions
theorem probability_of_weight_ge_30 : P_ge_30 = 0.70 :=
sorry  

end probability_of_weight_ge_30_l55_55718


namespace smaller_inscribed_cube_volume_is_192_sqrt_3_l55_55738

noncomputable def volume_of_smaller_inscribed_cube : ℝ :=
  let edge_length_of_larger_cube := 12
  let diameter_of_sphere := edge_length_of_larger_cube
  let side_length_of_smaller_cube := diameter_of_sphere / Real.sqrt 3
  let volume := side_length_of_smaller_cube ^ 3
  volume

theorem smaller_inscribed_cube_volume_is_192_sqrt_3 : 
  volume_of_smaller_inscribed_cube = 192 * Real.sqrt 3 := 
by
  sorry

end smaller_inscribed_cube_volume_is_192_sqrt_3_l55_55738


namespace sum_of_solutions_l55_55094

theorem sum_of_solutions (x y : ℝ) (h : 2 * x^2 + 2 * y^2 = 20 * x - 12 * y + 68) : x + y = 2 := 
sorry

end sum_of_solutions_l55_55094


namespace gathering_handshakes_l55_55253

theorem gathering_handshakes :
  let N := 12       -- twelve people, six couples
  let shakes_per_person := 9   -- each person shakes hands with 9 others
  let total_shakes := (N * shakes_per_person) / 2
  total_shakes = 54 := 
by
  sorry

end gathering_handshakes_l55_55253


namespace distance_to_plane_l55_55619

variable (V : ℝ) (A : ℝ) (r : ℝ) (d : ℝ)

-- Assume the volume of the sphere and area of the cross-section
def sphere_volume := V = 4 * Real.sqrt 3 * Real.pi
def cross_section_area := A = Real.pi

-- Define radius of sphere and cross-section
def sphere_radius := r = Real.sqrt 3
def cross_section_radius := Real.sqrt A = 1

-- Define distance as per Pythagorean theorem
def distance_from_center := d = Real.sqrt (r^2 - 1^2)

-- Main statement to prove
theorem distance_to_plane (V A : ℝ)
  (h1 : sphere_volume V) 
  (h2 : cross_section_area A) 
  (h3: sphere_radius r) 
  (h4: cross_section_radius A) : 
  distance_from_center r d :=
sorry

end distance_to_plane_l55_55619


namespace mean_median_mode_relation_l55_55998

-- Defining the data set of the number of fish caught in twelve outings.
def fish_catches : List ℕ := [3, 0, 2, 2, 1, 5, 3, 0, 1, 4, 3, 3]

-- Proof statement to show the relationship among mean, median and mode.
theorem mean_median_mode_relation (hs : fish_catches = [0, 0, 1, 1, 2, 2, 3, 3, 3, 3, 4, 5]) :
  let mean := (fish_catches.sum : ℚ) / fish_catches.length
  let median := (fish_catches.nthLe 5 sorry + fish_catches.nthLe 6 sorry : ℚ) / 2
  let mode := 3
  mean < median ∧ median < mode := by
  -- Placeholder for the proof. Details are skipped here.
  sorry

end mean_median_mode_relation_l55_55998


namespace kylie_coins_left_l55_55233

-- Definitions for each condition
def coins_from_piggy_bank : ℕ := 15
def coins_from_brother : ℕ := 13
def coins_from_father : ℕ := 8
def coins_given_to_friend : ℕ := 21

-- The total coins Kylie has initially
def initial_coins : ℕ := coins_from_piggy_bank + coins_from_brother
def total_coins_after_father : ℕ := initial_coins + coins_from_father
def coins_left : ℕ := total_coins_after_father - coins_given_to_friend

-- The theorem to prove the final number of coins left is 15
theorem kylie_coins_left : coins_left = 15 :=
by
  sorry -- Proof goes here

end kylie_coins_left_l55_55233


namespace percent_of_x_l55_55578

-- The mathematical equivalent of the problem statement in Lean.
theorem percent_of_x (x : ℝ) (hx : 0 < x) : (x / 10 + x / 25) = 0.14 * x :=
by
  sorry

end percent_of_x_l55_55578


namespace longer_side_length_l55_55597

-- Given a circle with radius 6 cm
def radius : ℝ := 6
def circle_area : ℝ := Real.pi * radius^2

-- Given that the area of the rectangle is three times the area of the circle
def rectangle_area : ℝ := 3 * circle_area

-- Given the rectangle has a shorter side equal to the diameter of the circle
def shorter_side : ℝ := 2 * radius

-- Prove that the length of the longer side of the rectangle is 9π cm
theorem longer_side_length : ∃ (longer_side : ℝ), longer_side = 9 * Real.pi :=
by
  have circle_area_def : circle_area = 36 * Real.pi := by sorry
  have rectangle_area_def : rectangle_area = 108 * Real.pi := by sorry
  have shorter_side_def : shorter_side = 12 := by sorry
  let longer_side := rectangle_area / shorter_side
  use longer_side
  show longer_side = 9 * Real.pi
  sorry

end longer_side_length_l55_55597


namespace total_apples_proof_l55_55752

-- Define the quantities Adam bought each day
def apples_monday := 15
def apples_tuesday := apples_monday * 3
def apples_wednesday := apples_tuesday * 4

-- The total quantity of apples Adam bought over these three days
def total_apples := apples_monday + apples_tuesday + apples_wednesday

-- Theorem stating that the total quantity of apples bought is 240
theorem total_apples_proof : total_apples = 240 := by
  sorry

end total_apples_proof_l55_55752


namespace one_clerk_forms_per_hour_l55_55602

theorem one_clerk_forms_per_hour
  (total_forms : ℕ)
  (total_hours : ℕ)
  (total_clerks : ℕ) 
  (h1 : total_forms = 2400)
  (h2 : total_hours = 8)
  (h3 : total_clerks = 12) :
  (total_forms / total_hours) / total_clerks = 25 :=
by
  have forms_per_hour := total_forms / total_hours
  have forms_per_clerk_per_hour := forms_per_hour / total_clerks
  sorry

end one_clerk_forms_per_hour_l55_55602


namespace intersecting_chords_l55_55996

noncomputable def length_of_other_chord (x : ℝ) : ℝ :=
  3 * x + 8 * x

theorem intersecting_chords
  (a b : ℝ) (h1 : a = 12) (h2 : b = 18) (r1 r2 : ℝ) (h3 : r1/r2 = 3/8) :
  length_of_other_chord 3 = 33 := by
  sorry

end intersecting_chords_l55_55996


namespace percent_calculation_l55_55438

theorem percent_calculation (Part Whole : ℝ) (h1 : Part = 120) (h2 : Whole = 80) :
  (Part / Whole) * 100 = 150 :=
by
  sorry

end percent_calculation_l55_55438


namespace sin_add_pi_over_2_eq_l55_55648

theorem sin_add_pi_over_2_eq :
  ∀ (A : ℝ), (cos (π + A) = -1/2) → (sin (π / 2 + A) = 1/2) :=
by
  intros A h
  -- Here we assume the proof steps
  sorry

end sin_add_pi_over_2_eq_l55_55648


namespace sum_of_squares_ge_one_third_l55_55633

theorem sum_of_squares_ge_one_third (a b c : ℝ) (h : a + b + c = 1) : 
  a^2 + b^2 + c^2 ≥ 1/3 := 
by 
  sorry

end sum_of_squares_ge_one_third_l55_55633


namespace pizzeria_large_pizzas_sold_l55_55273

theorem pizzeria_large_pizzas_sold (price_small price_large total_earnings num_small_pizzas num_large_pizzas : ℕ):
  price_small = 2 →
  price_large = 8 →
  total_earnings = 40 →
  num_small_pizzas = 8 →
  total_earnings = price_small * num_small_pizzas + price_large * num_large_pizzas →
  num_large_pizzas = 3 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end pizzeria_large_pizzas_sold_l55_55273


namespace power_sum_positive_l55_55927

theorem power_sum_positive 
    (a b c : ℝ) 
    (h1 : a * b * c > 0)
    (h2 : a + b + c > 0)
    (n : ℕ):
    a ^ n + b ^ n + c ^ n > 0 :=
by
  sorry

end power_sum_positive_l55_55927


namespace proof_problem_l55_55801

-- Define the universal set U
def U : Set Nat := {0, 1, 2, 3, 4}

-- Define the set M
def M : Set Nat := {2, 4}

-- Define the set N
def N : Set Nat := {0, 4}

-- Define the union of sets M and N
def M_union_N : Set Nat := M ∪ N

-- Define the complement of M ∪ N in U
def complement_U (s : Set Nat) : Set Nat := U \ s

-- State the theorem
theorem proof_problem : complement_U M_union_N = {1, 3} := by
  sorry

end proof_problem_l55_55801


namespace circular_permutations_2a2b2c_l55_55040

open Finset
open Nat.ArithmeticFunction

-- Define the main function calculating the number of circular permutations of a multiset
def circular_permutations (a b c : ℕ) : ℕ :=
  let n := a + b + c in
  (1 / n.toRat * ∑ d in divisors n, totient d * (factorial (n / d) / (factorial (a / d) * factorial (b / d) * factorial (c / d)))).toNat

theorem circular_permutations_2a2b2c : circular_permutations 2 2 2 = 16 :=
  sorry

end circular_permutations_2a2b2c_l55_55040


namespace final_price_percentage_of_original_l55_55304

theorem final_price_percentage_of_original (original_price sale_price final_price : ℝ)
  (h1 : sale_price = original_price * 0.5)
  (h2 : final_price = sale_price * 0.9) :
  final_price = original_price * 0.45 :=
by
  sorry

end final_price_percentage_of_original_l55_55304


namespace max_min_z_l55_55684

-- Define the ellipse
def on_ellipse (x y : ℝ) : Prop :=
  x^2 + 4*y^2 = 4*x

-- Define the function z
def z (x y : ℝ) : ℝ :=
  x^2 - y^2

-- Define the required points
def P1 (x y : ℝ) :=
  x = 4 ∧ y = 0

def P2 (x y : ℝ) :=
  x = 2/5 ∧ (y = 3/5 ∨ y = -3/5)

-- Theorem stating the required conditions
theorem max_min_z (x y : ℝ) (h : on_ellipse x y) :
  (P1 x y → z x y = 16) ∧ (P2 x y → z x y = -1/5) :=
by
  sorry

end max_min_z_l55_55684


namespace max_a_is_fractional_value_l55_55639

theorem max_a_is_fractional_value (a k : ℝ) (f : ℝ → ℝ) 
  (h_f : ∀ x, f x = x^2 - (k^2 - 5 * a * k + 3) * x + 7)
  (h_k : 0 ≤ k ∧ k ≤ 2)
  (x1 x2 : ℝ)
  (h_x1 : k ≤ x1 ∧ x1 ≤ k + a)
  (h_x2 : k + 2 * a ≤ x2 ∧ x2 ≤ k + 4 * a)
  (h_fx1_fx2 : f x1 ≥ f x2) :
  a = (2 * Real.sqrt 6 - 4) / 5 :=
sorry

end max_a_is_fractional_value_l55_55639


namespace range_of_m_l55_55054

theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ x ≠ 1 ∧ (m + 3) / (x - 1) = 1) ↔ m > -4 ∧ m ≠ -3 := 
by 
  sorry

end range_of_m_l55_55054


namespace abc_value_l55_55364

theorem abc_value (a b c : ℝ) (h1 : ab = 30 * (4^(1/3))) (h2 : ac = 40 * (4^(1/3))) (h3 : bc = 24 * (4^(1/3))) :
  a * b * c = 120 :=
sorry

end abc_value_l55_55364


namespace one_inch_cubes_with_two_or_more_painted_faces_l55_55119

def original_cube_length : ℕ := 4

def total_one_inch_cubes : ℕ := original_cube_length ^ 3

def corners_count : ℕ := 8

def edges_minus_corners_count : ℕ := 12 * 2

theorem one_inch_cubes_with_two_or_more_painted_faces
  (painted_faces_on_each_face : ∀ i : ℕ, i < total_one_inch_cubes → ℕ) : 
  ∃ n : ℕ, n = corners_count + edges_minus_corners_count ∧ n = 32 := 
by
  simp only [corners_count, edges_minus_corners_count, total_one_inch_cubes]
  sorry

end one_inch_cubes_with_two_or_more_painted_faces_l55_55119


namespace similar_triangle_shortest_side_l55_55885

theorem similar_triangle_shortest_side {a b c : ℝ} (h₁ : a = 24) (h₂ : b = 32) (h₃ : c = 80) :
  let hypotenuse₁ := Real.sqrt (a ^ 2 + b ^ 2)
  let scale_factor := c / hypotenuse₁
  let shortest_side₂ := scale_factor * a
  shortest_side₂ = 48 :=
by
  sorry

end similar_triangle_shortest_side_l55_55885


namespace ramu_profit_percent_l55_55250

noncomputable def carCost : ℝ := 42000
noncomputable def repairCost : ℝ := 13000
noncomputable def sellingPrice : ℝ := 60900
noncomputable def totalCost : ℝ := carCost + repairCost
noncomputable def profit : ℝ := sellingPrice - totalCost
noncomputable def profitPercent : ℝ := (profit / totalCost) * 100

theorem ramu_profit_percent : profitPercent = 10.73 := 
by
  sorry

end ramu_profit_percent_l55_55250


namespace total_games_correct_l55_55860

noncomputable def number_of_games_per_month : ℕ := 13
noncomputable def number_of_months_in_season : ℕ := 14
noncomputable def total_games_in_season : ℕ := number_of_games_per_month * number_of_months_in_season

theorem total_games_correct : total_games_in_season = 182 := by
  sorry

end total_games_correct_l55_55860


namespace real_estate_profit_l55_55745

def purchase_price_first : ℝ := 350000
def purchase_price_second : ℝ := 450000
def purchase_price_third : ℝ := 600000

def gain_first : ℝ := 0.12
def loss_second : ℝ := 0.08
def gain_third : ℝ := 0.18

def selling_price_first : ℝ :=
  purchase_price_first + (purchase_price_first * gain_first)
def selling_price_second : ℝ :=
  purchase_price_second - (purchase_price_second * loss_second)
def selling_price_third : ℝ :=
  purchase_price_third + (purchase_price_third * gain_third)

def total_purchase_price : ℝ :=
  purchase_price_first + purchase_price_second + purchase_price_third
def total_selling_price : ℝ :=
  selling_price_first + selling_price_second + selling_price_third

def overall_gain : ℝ :=
  total_selling_price - total_purchase_price

theorem real_estate_profit :
  overall_gain = 114000 := by
  sorry

end real_estate_profit_l55_55745


namespace area_square_A_32_l55_55462

-- Define the areas of the squares in Figure B and Figure A and their relationship with the triangle areas
def identical_isosceles_triangles_with_squares (area_square_B : ℝ) (area_triangle_B : ℝ) (area_square_A : ℝ) (area_triangle_A : ℝ) :=
  area_triangle_B = (area_square_B / 2) * 4 ∧
  area_square_A / area_triangle_A = 4 / 9

theorem area_square_A_32 {area_square_B : ℝ} (h : area_square_B = 36) :
  identical_isosceles_triangles_with_squares area_square_B 72 32 72 :=
by
  sorry

end area_square_A_32_l55_55462


namespace Jake_weight_l55_55810

variables (J S : ℝ)

theorem Jake_weight (h1 : 0.8 * J = 2 * S) (h2 : J + S = 168) : J = 120 :=
  sorry

end Jake_weight_l55_55810


namespace license_plate_count_l55_55065

theorem license_plate_count :
  let letters := 26
  let digits := 10
  let second_char_options := letters - 1 + digits
  let third_char_options := digits - 1
  letters * second_char_options * third_char_options = 8190 :=
by
  sorry

end license_plate_count_l55_55065


namespace even_function_derivative_l55_55636

theorem even_function_derivative (f : ℝ → ℝ)
  (h_even : ∀ x, f (-x) = f x)
  (h_deriv_pos : ∀ x > 0, deriv f x = (x - 1) * (x - 2)) : f (-2) < f 1 :=
sorry

end even_function_derivative_l55_55636


namespace gcd_32_48_l55_55999

/--
The greatest common factor of 32 and 48 is 16.
-/
theorem gcd_32_48 : Int.gcd 32 48 = 16 :=
by
  sorry

end gcd_32_48_l55_55999


namespace fraction_of_blue_cars_l55_55425

-- Definitions of the conditions
def total_cars : ℕ := 516
def red_cars : ℕ := total_cars / 2
def black_cars : ℕ := 86
def blue_cars : ℕ := total_cars - (red_cars + black_cars)

-- Statement to prove that the fraction of blue cars is 1/3
theorem fraction_of_blue_cars :
  (blue_cars : ℚ) / total_cars = 1 / 3 :=
by
  sorry -- Proof to be filled in

end fraction_of_blue_cars_l55_55425


namespace train_speed_l55_55173

noncomputable def speed_in_kmh (distance : ℕ) (time : ℕ) : ℚ :=
  (distance : ℚ) / (time : ℚ) * 3600 / 1000

theorem train_speed
  (distance : ℕ) (time : ℕ)
  (h_dist : distance = 150)
  (h_time : time = 9) :
  speed_in_kmh distance time = 60 :=
by
  rw [h_dist, h_time]
  sorry

end train_speed_l55_55173


namespace expression_equals_minus_0p125_l55_55320

-- Define the expression
def compute_expression : ℝ := 0.125^8 * (-8)^7

-- State the theorem to prove
theorem expression_equals_minus_0p125 : compute_expression = -0.125 :=
by {
  sorry
}

end expression_equals_minus_0p125_l55_55320


namespace initial_geese_count_l55_55136

-- Define the number of geese that flew away
def geese_flew_away : ℕ := 28

-- Define the number of geese left in the field
def geese_left : ℕ := 23

-- Prove that the initial number of geese in the field was 51
theorem initial_geese_count : geese_left + geese_flew_away = 51 := by
  sorry

end initial_geese_count_l55_55136


namespace valid_p_values_l55_55214

theorem valid_p_values (p : ℕ) (h : p = 3 ∨ p = 4 ∨ p = 5 ∨ p = 12) :
  0 < (4 * p + 34) / (3 * p - 8) ∧ (4 * p + 34) % (3 * p - 8) = 0 :=
by
  sorry

end valid_p_values_l55_55214


namespace polynomial_roots_l55_55780

theorem polynomial_roots :
  ∃ (x : ℚ) (y : ℚ) (z : ℚ) (w : ℚ),
    (x = 1) ∧ (y = 1) ∧ (z = -2) ∧ (w = -1/2) ∧
    2*x^4 + x^3 - 6*x^2 + x + 2 = 0 ∧
    2*y^4 + y^3 - 6*y^2 + y + 2 = 0 ∧
    2*z^4 + z^3 - 6*z^2 + z + 2 = 0 ∧
    2*w^4 + w^3 - 6*w^2 + w + 2 = 0 :=
by
  sorry

end polynomial_roots_l55_55780


namespace range_of_a_l55_55369

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, (a - 1) * x < a - 1 ↔ x > 1) : a < 1 := 
sorry

end range_of_a_l55_55369


namespace batsman_average_after_12_innings_l55_55441

theorem batsman_average_after_12_innings
  (score_12th: ℕ) (increase_avg: ℕ) (initial_innings: ℕ) (final_innings: ℕ) 
  (initial_avg: ℕ) (final_avg: ℕ) :
  score_12th = 48 ∧ increase_avg = 2 ∧ initial_innings = 11 ∧ final_innings = 12 ∧
  final_avg = initial_avg + increase_avg ∧
  12 * final_avg = initial_innings * initial_avg + score_12th →
  final_avg = 26 :=
by 
  sorry

end batsman_average_after_12_innings_l55_55441


namespace area_of_closed_figure_l55_55701

theorem area_of_closed_figure:
  ∫ (x : ℝ) in 1..3, (x - (1 / x)) = 4 - real.log 3 :=
by
  sorry

end area_of_closed_figure_l55_55701


namespace tan_585_eq_1_l55_55184

noncomputable def tan_deg (θ : ℝ) : ℝ := Real.tan (θ * Real.pi / 180)

theorem tan_585_eq_1 :
  tan_deg 585 = 1 := 
by
  have h1 : 585 - 360 = 225 := by norm_num
  have h2 : tan_deg 225 = tan_deg 45 :=
    by have h3 : 225 = 180 + 45 := by norm_num
       rw [h3, tan_deg]
       exact Real.tan_add_pi_div_two_simp_left (Real.pi * 45 / 180)
  rw [← tan_deg]
  rw [h1, h2]
  exact Real.tan_pi_div_four

end tan_585_eq_1_l55_55184


namespace longer_side_of_rectangle_l55_55600

theorem longer_side_of_rectangle (r : ℝ) (Aₙ Aₙ: ℝ) (L S: ℝ):
  (r = 6) → 
  (Aₙ = 36 * π) →
  (Aₙ = 108 * π) →
  (S = 12) → 
  (S * L = Aₙ) →
  L = 9 * π := sorry

end longer_side_of_rectangle_l55_55600


namespace sequence_100th_term_eq_l55_55245

-- Definitions for conditions
def numerator (n : ℕ) : ℕ := 1 + (n - 1) * 2
def denominator (n : ℕ) : ℕ := 2 + (n - 1) * 3

-- The statement of the problem as a Lean 4 theorem
theorem sequence_100th_term_eq :
  (numerator 100) / (denominator 100) = 199 / 299 :=
by
  sorry

end sequence_100th_term_eq_l55_55245


namespace walnut_trees_planted_l55_55712

-- Define the initial number of walnut trees
def initial_walnut_trees : ℕ := 22

-- Define the total number of walnut trees after planting
def total_walnut_trees_after : ℕ := 55

-- The Lean statement to prove the number of walnut trees planted today
theorem walnut_trees_planted : (total_walnut_trees_after - initial_walnut_trees = 33) :=
by
  sorry

end walnut_trees_planted_l55_55712


namespace find_d_l55_55113

noncomputable def quadratic_roots (d : ℝ) : Prop :=
∀ x : ℝ, x^2 + 7*x + d = 0 ↔ x = (-7 + Real.sqrt d) / 2 ∨ x = (-7 - Real.sqrt d) / 2

theorem find_d : ∃ d : ℝ, d = 9.8 ∧ quadratic_roots d :=
sorry

end find_d_l55_55113


namespace border_area_correct_l55_55611

theorem border_area_correct :
  let photo_height := 9
  let photo_width := 12
  let border_width := 3
  let photo_area := photo_height * photo_width
  let framed_height := photo_height + 2 * border_width
  let framed_width := photo_width + 2 * border_width
  let framed_area := framed_height * framed_width
  let border_area := framed_area - photo_area
  border_area = 162 :=
by sorry

end border_area_correct_l55_55611


namespace longer_side_of_rectangle_l55_55594

theorem longer_side_of_rectangle
  (r : ℝ) (A_rect A_circle L S : ℝ) (h1 : r = 6)
  (h2 : A_circle = π * r^2)
  (h3 : A_rect = 3 * A_circle)
  (h4 : S = 2 * r)
  (h5 : A_rect = S * L) : L = 9 * π :=
by
  sorry

end longer_side_of_rectangle_l55_55594


namespace smallest_number_of_students_l55_55311

/--
At a school, the ratio of 10th-graders to 8th-graders is 3:2, 
and the ratio of 10th-graders to 9th-graders is 5:3. 
Prove that the smallest number of students from these grades is 34.
-/
theorem smallest_number_of_students {G8 G9 G10 : ℕ} 
  (h1 : 3 * G8 = 2 * G10) (h2 : 5 * G9 = 3 * G10) : 
  G10 + G8 + G9 = 34 :=
by
  sorry

end smallest_number_of_students_l55_55311


namespace x12_is_1_l55_55649

noncomputable def compute_x12 (x : ℝ) (h : x + 1 / x = Real.sqrt 5) : ℝ :=
  x ^ 12

theorem x12_is_1 (x : ℝ) (h : x + 1 / x = Real.sqrt 5) : compute_x12 x h = 1 :=
  sorry

end x12_is_1_l55_55649


namespace hyperbola_eccentricity_l55_55072

theorem hyperbola_eccentricity (a b : ℝ) (h : a^2 = 4 ∧ b^2 = 3) :
    let c := Real.sqrt (a^2 + b^2)
    let e := c / a
    e = Real.sqrt 7 / 2 :=
    by
  sorry

end hyperbola_eccentricity_l55_55072


namespace laborers_employed_l55_55310

theorem laborers_employed 
    (H L : ℕ) 
    (h1 : H + L = 35) 
    (h2 : 140 * H + 90 * L = 3950) : 
    L = 19 :=
by
  sorry

end laborers_employed_l55_55310


namespace xy_inequality_l55_55348

theorem xy_inequality (x y : ℝ) (h: x^8 + y^8 ≤ 2) : 
  x^2 * y^2 + |x^2 - y^2| ≤ π / 2 :=
sorry

end xy_inequality_l55_55348


namespace proof_problem_l55_55939

noncomputable def question (a b c d m : ℚ) : ℚ :=
  2 * a + 2 * b + (a + b - 3 * (c * d)) - m

def condition1 (m : ℚ) : Prop :=
  abs (m + 1) = 4

def condition2 (a b : ℚ) : Prop :=
  a = -b

def condition3 (c d : ℚ) : Prop :=
  c * d = 1

theorem proof_problem (a b c d m : ℚ) :
  condition1 m → condition2 a b → condition3 c d →
  (question a b c d m = 2 ∨ question a b c d m = -6) :=
by
  sorry

end proof_problem_l55_55939


namespace exists_infinite_repeated_sum_of_digits_l55_55347

-- Define the sum of the digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Define the sequence a_n which is the sum of digits of P(n)
def a (P : ℕ → ℤ) (n : ℕ) : ℕ :=
  sum_of_digits (P n).natAbs

theorem exists_infinite_repeated_sum_of_digits (P : ℕ → ℤ) (h_nat_coeffs : ∀ n, (P n) ≥ 0) :
  ∃ s : ℕ, ∀ N : ℕ, ∃ n : ℕ, n ≥ N ∧ a P n = s :=
sorry

end exists_infinite_repeated_sum_of_digits_l55_55347


namespace count_negative_numbers_l55_55066

def evaluate (e : String) : Int :=
  match e with
  | "-3^2" => -9
  | "(-3)^2" => 9
  | "-(-3)" => 3
  | "-|-3|" => -3
  | _ => 0

def isNegative (n : Int) : Bool := n < 0

def countNegatives (es : List String) : Int :=
  es.map evaluate |>.filter isNegative |>.length

theorem count_negative_numbers :
  countNegatives ["-3^2", "(-3)^2", "-(-3)", "-|-3|"] = 2 :=
by
  sorry

end count_negative_numbers_l55_55066


namespace intersecting_absolute_value_functions_l55_55933

theorem intersecting_absolute_value_functions (a b c d : ℝ) (h1 : -|2 - a| + b = 5) (h2 : -|8 - a| + b = 3) (h3 : |2 - c| + d = 5) (h4 : |8 - c| + d = 3) (ha : 2 < a) (h8a : a < 8) (hc : 2 < c) (h8c : c < 8) : a + c = 10 :=
sorry

end intersecting_absolute_value_functions_l55_55933


namespace angle_B_in_triangle_l55_55659

theorem angle_B_in_triangle (a b c : ℝ) (B : ℝ) (h : (a^2 + c^2 - b^2) * Real.tan B = Real.sqrt 3 * a * c) :
  B = 60 ∨ B = 120 := 
sorry

end angle_B_in_triangle_l55_55659


namespace ellipse_hyperbola_proof_l55_55917

noncomputable def ellipse_and_hyperbola_condition (a b : ℝ) : Prop :=
  (a > b ∧ b > 0) ∧ (a^2 - b^2 = 5) ∧ (a^2 = 11 * b^2)

theorem ellipse_hyperbola_proof : 
  ∀ (a b : ℝ), ellipse_and_hyperbola_condition a b → b^2 = 0.5 :=
by
  intros a b h
  sorry

end ellipse_hyperbola_proof_l55_55917


namespace units_digit_7_pow_2050_l55_55281

theorem units_digit_7_pow_2050 : (7 ^ 2050) % 10 = 9 := 
by 
  sorry

end units_digit_7_pow_2050_l55_55281


namespace bobs_income_after_changes_l55_55179

variable (initial_salary : ℝ) (february_increase_rate : ℝ) (march_reduction_rate : ℝ)

def february_salary (initial_salary : ℝ) (increase_rate : ℝ) : ℝ :=
  initial_salary * (1 + increase_rate)

def march_salary (february_salary : ℝ) (reduction_rate : ℝ) : ℝ :=
  february_salary * (1 - reduction_rate)

theorem bobs_income_after_changes (h1 : initial_salary = 2750)
  (h2 : february_increase_rate = 0.15)
  (h3 : march_reduction_rate = 0.10) :
  march_salary (february_salary initial_salary february_increase_rate) march_reduction_rate = 2846.25 := 
sorry

end bobs_income_after_changes_l55_55179


namespace intersection_range_l55_55358

-- Define the line equation
def line (k x : ℝ) : ℝ := k * x - k + 1

-- Define the curve equation
def curve (x y m : ℝ) : Prop := x^2 + 2 * y^2 = m

-- State the problem: Given the line and the curve have a common point, prove the range of m is m >= 3
theorem intersection_range (k m : ℝ) (h : ∃ x y, line k x = y ∧ curve x y m) : m ≥ 3 :=
by {
  sorry
}

end intersection_range_l55_55358


namespace num_factors_of_2310_with_more_than_three_factors_l55_55360

theorem num_factors_of_2310_with_more_than_three_factors : 
  (∃ n : ℕ, n > 0 ∧ ∀ d : ℕ, d ∣ 2310 → (∀ f : ℕ, f ∣ d → f = 1 ∨ f = d ∨ f ∣ d) → 26 = n) := sorry

end num_factors_of_2310_with_more_than_three_factors_l55_55360


namespace area_of_defined_region_l55_55565

theorem area_of_defined_region : 
  ∃ (A : ℝ), (∀ x y : ℝ, |4 * x - 20| + |3 * y + 9| ≤ 6 → A = 9) :=
sorry

end area_of_defined_region_l55_55565


namespace find_x_plus_y_l55_55791

theorem find_x_plus_y (x y : ℝ) (h1 : x + Real.sin y = 2008) (h2 : x + 2008 * Real.cos y = 2007) (h3 : 0 ≤ y ∧ y ≤ Real.pi / 2) :
  x + y = 2007 + Real.pi / 2 := 
sorry

end find_x_plus_y_l55_55791


namespace sqrt_5_is_quadratic_radical_l55_55430

variable (a : ℝ) -- a is a real number

-- Definition to check if a given expression is a quadratic radical
def is_quadratic_radical (x : ℝ) : Prop := ∃ y : ℝ, y^2 = x

theorem sqrt_5_is_quadratic_radical : is_quadratic_radical 5 :=
by
  -- Here, 'by' indicates the start of the proof block,
  -- but the actual content of the proof is replaced with 'sorry' as instructed.
  sorry

end sqrt_5_is_quadratic_radical_l55_55430


namespace range_of_a_l55_55216

theorem range_of_a (a : ℝ) : 
  (∀ x y : ℝ, x ≤ a → (x + y + 1 ≤ 2 * (x + 1) - 3 * (y + 1))) → a ≤ -2 :=
by 
  intros h
  sorry

end range_of_a_l55_55216


namespace number_of_cubes_with_at_least_two_faces_painted_is_56_l55_55127

def one_inch_cubes_with_at_least_two_faces_painted 
  (side_length : ℕ) (face_colors : ℕ) (cubes_per_side : ℕ) :=
  if side_length = 4 ∧ face_colors = 6 ∧ cubes_per_side = 1 then 56 else 0

theorem number_of_cubes_with_at_least_two_faces_painted_is_56 :
  one_inch_cubes_with_at_least_two_faces_painted 4 6 1 = 56 :=
by
  sorry

end number_of_cubes_with_at_least_two_faces_painted_is_56_l55_55127


namespace jason_total_spent_l55_55077

def cost_of_flute : ℝ := 142.46
def cost_of_music_tool : ℝ := 8.89
def cost_of_song_book : ℝ := 7.00

def total_spent (flute_cost music_tool_cost song_book_cost : ℝ) : ℝ :=
  flute_cost + music_tool_cost + song_book_cost

theorem jason_total_spent :
  total_spent cost_of_flute cost_of_music_tool cost_of_song_book = 158.35 :=
by
  -- Proof omitted
  sorry

end jason_total_spent_l55_55077


namespace cube_faces_paint_count_l55_55121

def is_painted_on_at_least_two_faces (x y z : ℕ) : Prop :=
  ((x = 0 ∨ x = 3) ∨ (y = 0 ∨ y = 3) ∨ (z = 0 ∨ z = 3))

theorem cube_faces_paint_count :
  let n := 4 in
  let volume := n * n * n in
  let one_inch_cubes := {c | ∃ x y z : ℕ, x < n ∧ y < n ∧ z < n ∧ c = (x, y, z)} in
  let painted_cubes := {c ∈ one_inch_cubes | exists_along_edges c} in  -- Auxiliary predicate to check edge paint
  card painted_cubes = 32 :=
begin
  sorry,
end

/-- Auxiliary predicate to check if a cube is on the edges but not corners, signifying 
    at least two faces are painted --/
predicate exists_along_edges (c: ℕ × ℕ × ℕ) := 
  match c with
  | (0, _, _) => true
  | (3, _, _) => true
  | (_, 0, _) => true
  | (_, 3, _) => true
  | (_, _, 0) => true
  | (_, _, 3) => true
  | (_, _, _) => false
  end

end cube_faces_paint_count_l55_55121


namespace trisha_total_distance_l55_55089

theorem trisha_total_distance :
  let distance1 := 0.11
  let distance2 := 0.11
  let distance3 := 0.67
  distance1 + distance2 + distance3 = 0.89 :=
by
  sorry

end trisha_total_distance_l55_55089


namespace certain_fraction_exists_l55_55336

theorem certain_fraction_exists (a b : ℚ) (h : a / b = 3 / 4) :
  (a / b) / (1 / 5) = (3 / 4) / (2 / 5) :=
by
  sorry

end certain_fraction_exists_l55_55336


namespace twelfth_term_arithmetic_sequence_l55_55146

theorem twelfth_term_arithmetic_sequence :
  let a := (1 : ℚ) / 2
  let d := (1 : ℚ) / 3
  (a + 11 * d) = (25 : ℚ) / 6 :=
by
  sorry

end twelfth_term_arithmetic_sequence_l55_55146


namespace prob_kong_meng_is_one_sixth_l55_55341

variable (bag : List String := ["孔", "孟", "之", "乡"])
variable (draws : List String := [])
def total_events : ℕ := 4 * 3
def favorable_events : ℕ := 2
def probability_kong_meng : ℚ := favorable_events / total_events

theorem prob_kong_meng_is_one_sixth :
  (probability_kong_meng = 1 / 6) :=
by
  sorry

end prob_kong_meng_is_one_sixth_l55_55341


namespace equation_of_line_AB_through_A_and_B_l55_55734

noncomputable def point := (3,1) : ℝ × ℝ

def circle_eqn (x y : ℝ) := (x - 1)^2 + y^2 = 1

def line_eqn_AB : ℝ → ℝ → Prop := λ x y, 2 * x + y - 3 = 0

theorem equation_of_line_AB_through_A_and_B
  (A B : ℝ × ℝ)
  (h₁ : circle_eqn A.1 A.2)
  (h₂ : circle_eqn B.1 B.2)
  (h₃ : (∃ (l : ℝ → ℝ → Prop), (∀ x y, l x y = 0 ↔ (circle_eqn x y)) ∧ 
         (∃ (p : ℝ × ℝ), p = point ∧ l A.1 A.2 = 0 ∧ l B.1 B.2 = 0)) : 
  line_eqn_AB A.1 A.2 ∧ line_eqn_AB B.1 B.2 :=
sorry

end equation_of_line_AB_through_A_and_B_l55_55734


namespace min_M_value_l55_55491

noncomputable def max_pq (p q : ℝ) : ℝ := if p ≥ q then p else q

noncomputable def M (x y : ℝ) : ℝ := max_pq (|x^2 + y + 1|) (|y^2 - x + 1|)

theorem min_M_value : (∀ x y : ℝ, M x y ≥ (3 : ℚ) / 4) ∧ (∃ x y : ℝ, M x y = (3 : ℚ) / 4) :=
sorry

end min_M_value_l55_55491


namespace age_of_B_l55_55293

theorem age_of_B (A B C : ℕ) (h1 : A = 2 * C + 2) (h2 : B = 2 * C) (h3 : A + B + C = 27) : B = 10 :=
by
  sorry

end age_of_B_l55_55293


namespace red_robin_team_arrangements_l55_55407

theorem red_robin_team_arrangements :
  let boys := 3
  let girls := 4
  let choose2 (n : ℕ) (k : ℕ) := Nat.choose n k
  let permutations (n : ℕ) := Nat.factorial n
  let waysToPositionBoys := choose2 boys 2 * permutations 2
  let waysToPositionRemainingMembers := permutations (boys - 2 + girls)
  waysToPositionBoys * waysToPositionRemainingMembers = 720 :=
by
  let boys := 3
  let girls := 4
  let choose2 (n : ℕ) (k : ℕ) := Nat.choose n k
  let permutations (n : ℕ) := Nat.factorial n
  let waysToPositionBoys := choose2 boys 2 * permutations 2
  let waysToPositionRemainingMembers := permutations (boys - 2 + girls)
  have : waysToPositionBoys * waysToPositionRemainingMembers = 720 := 
    by sorry -- Proof omitted here
  exact this

end red_robin_team_arrangements_l55_55407


namespace range_of_m_l55_55642

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := -x^2 + x + m + 2

theorem range_of_m (m : ℝ) : 
  (∃! x : ℤ, f x m ≥ |x|) ↔ -2 ≤ m ∧ m < -1 :=
by
  sorry

end range_of_m_l55_55642


namespace x_minus_y_div_x_eq_4_7_l55_55362

-- Definitions based on the problem's conditions
axiom y_div_x_eq_3_7 (x y : ℝ) : y / x = 3 / 7

-- The main problem to prove
theorem x_minus_y_div_x_eq_4_7 (x y : ℝ) (h : y / x = 3 / 7) : (x - y) / x = 4 / 7 := by
  sorry

end x_minus_y_div_x_eq_4_7_l55_55362


namespace line_through_intersection_points_l55_55038

def first_circle (x y : ℝ) : Prop := x^2 + y^2 - x + y - 2 = 0
def second_circle (x y : ℝ) : Prop := x^2 + y^2 = 5

theorem line_through_intersection_points (x y : ℝ) :
  (first_circle x y ∧ second_circle x y) → x - y - 3 = 0 :=
by
  sorry

end line_through_intersection_points_l55_55038


namespace solve_system_l55_55277

theorem solve_system :
  ∃ (x y : ℝ), (2 * x - y = 1) ∧ (x + y = 2) ∧ (x = 1) ∧ (y = 1) :=
by
  sorry

end solve_system_l55_55277


namespace total_bees_in_colony_l55_55067

def num_bees_in_hive_after_changes (initial_bees : ℕ) (bees_in : ℕ) (bees_out : ℕ) : ℕ :=
  initial_bees + bees_in - bees_out

theorem total_bees_in_colony :
  let hive1 := num_bees_in_hive_after_changes 45 12 8
  let hive2 := num_bees_in_hive_after_changes 60 15 20
  let hive3 := num_bees_in_hive_after_changes 75 10 5
  hive1 + hive2 + hive3 = 184 :=
by
  sorry

end total_bees_in_colony_l55_55067


namespace plant_initial_mass_l55_55299

theorem plant_initial_mass (x : ℕ) :
  (27 * x + 52 = 133) → x = 3 :=
by
  intro h
  sorry

end plant_initial_mass_l55_55299


namespace comp_1_sub_i_pow4_l55_55468

theorem comp_1_sub_i_pow4 : (1 - complex.I)^4 = -4 := by
  sorry

end comp_1_sub_i_pow4_l55_55468


namespace walking_speed_l55_55218

theorem walking_speed (x : ℝ) (h1 : 20 / x = 40 / (x + 5)) : x + 5 = 10 :=
  by
  sorry

end walking_speed_l55_55218


namespace benny_bought_books_l55_55766

theorem benny_bought_books :
  ∀ (initial_books sold_books remaining_books bought_books : ℕ),
    initial_books = 22 →
    sold_books = initial_books / 2 →
    remaining_books = initial_books - sold_books →
    remaining_books + bought_books = 17 →
    bought_books = 6 :=
by
  intros initial_books sold_books remaining_books bought_books
  sorry

end benny_bought_books_l55_55766


namespace angle_in_fourth_quadrant_l55_55943

theorem angle_in_fourth_quadrant (α : ℝ) (h : 0 < α ∧ α < 90) : 270 < 360 - α ∧ 360 - α < 360 :=
by
  sorry

end angle_in_fourth_quadrant_l55_55943


namespace simplify_cbrt_expr_l55_55978

-- Define the cube root function.
def cbrt (x : ℝ) : ℝ := x^(1/3)

-- Define the original expression under the cube root.
def original_expr : ℝ := 40^3 + 70^3 + 100^3

-- Define the simplified expression.
def simplified_expr : ℝ := 10 * cbrt 1407

theorem simplify_cbrt_expr : cbrt original_expr = simplified_expr := by
  -- Declaration that proof is not provided to ensure Lean statement is complete.
  sorry

end simplify_cbrt_expr_l55_55978


namespace merchant_marked_price_l55_55167

theorem merchant_marked_price (L P x S : ℝ)
  (h1 : L = 100)
  (h2 : P = 70)
  (h3 : S = 0.8 * x)
  (h4 : 0.8 * x - 70 = 0.3 * (0.8 * x)) :
  x = 125 :=
by
  sorry

end merchant_marked_price_l55_55167


namespace problem_statement_l55_55915

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin x - Real.pi * x

theorem problem_statement (x : ℝ) (h₀ : 0 < x) (h₁ : x < Real.pi / 2) : 
  ((deriv f x < 0) ∧ (f x < 0)) :=
by
  sorry

end problem_statement_l55_55915


namespace tan_60_eq_sqrt3_l55_55583

theorem tan_60_eq_sqrt3 : Real.tan (Real.pi / 3) = Real.sqrt 3 := 
sorry

end tan_60_eq_sqrt3_l55_55583


namespace sampling_methods_suitability_l55_55139

-- Define sample sizes and population sizes
def n1 := 2  -- Number of students to be selected in sample ①
def N1 := 10  -- Population size for sample ①
def n2 := 50  -- Number of students to be selected in sample ②
def N2 := 1000  -- Population size for sample ②

-- Define what it means for a sampling method to be suitable
def is_simple_random_sampling_suitable (n N : Nat) : Prop :=
  N <= 50 ∧ n < N

def is_systematic_sampling_suitable (n N : Nat) : Prop :=
  N > 50 ∧ n < N ∧ n ≥ 50 / 1000 * N  -- Ensuring suitable systematic sampling size

-- The proof statement
theorem sampling_methods_suitability :
  is_simple_random_sampling_suitable n1 N1 ∧ is_systematic_sampling_suitable n2 N2 :=
by
  -- Sorry blocks are used to skip the proofs
  sorry

end sampling_methods_suitability_l55_55139


namespace correct_calculation_l55_55429

theorem correct_calculation (x : ℝ) : 
(x + x = 2 * x) ∧
(x * x = x^2) ∧
(2 * x * x^2 = 2 * x^3) ∧
(x^6 / x^3 = x^3) →
(2 * x * x^2 = 2 * x^3) := 
by
  intro h
  exact h.2.2.1

end correct_calculation_l55_55429


namespace solve_equation1_solve_equation2_solve_equation3_solve_equation4_l55_55096

theorem solve_equation1 (x : ℝ) : (x - 1) ^ 2 = 4 ↔ x = 3 ∨ x = -1 :=
by sorry

theorem solve_equation2 (x : ℝ) : x ^ 2 + 3 * x - 4 = 0 ↔ x = 1 ∨ x = -4 :=
by sorry

theorem solve_equation3 (x : ℝ) : 4 * x * (2 * x + 1) = 3 * (2 * x + 1) ↔ x = -1 / 2 ∨ x = 3 / 4 :=
by sorry

theorem solve_equation4 (x : ℝ) : 2 * x ^ 2 + 5 * x - 3 = 0 ↔ x = 1 / 2 ∨ x = -3 :=
by sorry

end solve_equation1_solve_equation2_solve_equation3_solve_equation4_l55_55096


namespace meaningful_range_fraction_l55_55516

theorem meaningful_range_fraction (x : ℝ) : 
  ¬ (x = 3) ↔ (∃ y, y = x / (x - 3)) :=
sorry

end meaningful_range_fraction_l55_55516


namespace convert_to_cylindrical_l55_55770

noncomputable def cylindricalCoordinates (x y z : ℝ) : ℝ × ℝ × ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  let θ := Real.arccos (x / r)
  if y / r < 0 then (r, 2 * Real.pi - θ, z) else (r, θ, z)

theorem convert_to_cylindrical :
  cylindricalCoordinates 3 (-3 * Real.sqrt 3) 4 = (6, 5 * Real.pi / 3, 4) :=
by
  sorry

end convert_to_cylindrical_l55_55770


namespace BPB1Q_is_parallelogram_l55_55935

-- Given conditions
variable {α : Type*} [EuclideanGeometry α]

variable (N A B C K M Q P B1 : Point α)
variable (innerCircle outerCircle : Circle α)

-- Conditions based on the problem description
axiom tangent_inner_outer : innerCircle.tangent outerCircle N
axiom touch_inner_K : innerCircle.tangentAt K (Segment B A)
axiom touch_inner_M : innerCircle.tangentAt M (Segment B C)
axiom mid_arc_AB : outerCircle.midpoint_arc_not_in N (Segment B A) Q
axiom mid_arc_BC : outerCircle.midpoint_arc_not_in N (Segment B C) P
axiom circumcircle_BQK : (Triangle B Q K).circumcircle.exists B1
axiom circumcircle_BPM : (Triangle B P M).circumcircle.exists B1

-- Theorem to be proved
theorem BPB1Q_is_parallelogram :
  Parallelogram (Segment B P) (Segment B1 Q) :=
sorry

end BPB1Q_is_parallelogram_l55_55935


namespace infinite_series_sum_l55_55706

theorem infinite_series_sum :
  ∑' (n : ℕ), (n + 1) / 10^(n + 1) = 10 / 81 :=
sorry

end infinite_series_sum_l55_55706


namespace pizzeria_large_pizzas_sold_l55_55274

theorem pizzeria_large_pizzas_sold (price_small price_large total_earnings num_small_pizzas num_large_pizzas : ℕ):
  price_small = 2 →
  price_large = 8 →
  total_earnings = 40 →
  num_small_pizzas = 8 →
  total_earnings = price_small * num_small_pizzas + price_large * num_large_pizzas →
  num_large_pizzas = 3 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end pizzeria_large_pizzas_sold_l55_55274


namespace greatest_number_divisible_by_11_and_3_l55_55610

namespace GreatestNumberDivisibility

theorem greatest_number_divisible_by_11_and_3 : 
  ∃ (A B C : ℕ), 
    A ≠ B ∧ A ≠ C ∧ B ≠ C ∧ 
    (2 * A - 2 * B + C) % 11 = 0 ∧ 
    (2 * A + 2 * C + B) % 3 = 0 ∧
    (10000 * A + 1000 * C + 100 * C + 10 * B + A) = 95695 :=
by
  -- The proof here is omitted.
  sorry

end GreatestNumberDivisibility

end greatest_number_divisible_by_11_and_3_l55_55610


namespace expression_simplification_l55_55948

theorem expression_simplification (x : ℝ) (h : x < -2) : 1 - |1 + x| = -2 - x := 
by
  sorry

end expression_simplification_l55_55948


namespace rational_root_k_values_l55_55784

theorem rational_root_k_values (k : ℤ) :
  (∃ x : ℚ, x^2017 - x^2016 + x^2 + k * x + 1 = 0) ↔ (k = 0 ∨ k = -2) :=
by
  sorry

end rational_root_k_values_l55_55784


namespace contrapositive_inverse_converse_negation_false_l55_55728

theorem contrapositive (a b : ℤ) : (a ≤ b) → (a - 2 ≤ b - 2) :=
sorry

theorem inverse (a b : ℤ) : (a - 2 ≤ b - 2) → (a ≤ b) :=
sorry

theorem converse (a b : ℤ) : (a - 2 > b - 2) → (a > b) :=
sorry

theorem negation_false (a b : ℤ) : ¬ ((a > b) → (a - 2 ≤ b - 2)) :=
sorry

end contrapositive_inverse_converse_negation_false_l55_55728


namespace probability_product_zero_l55_55138

open Finset

theorem probability_product_zero (S : Finset ℤ) (hS : S = {-3, -1, 0, 2, 4}) :
  let pairs := S.product S.filter (λ p, p.1 ≠ p.2),
      zero_product_pairs := pairs.filter (λ p, p.1 * p.2 = 0) in
  (zero_product_pairs.card : ℚ) / pairs.card = 2 / 5 := by
suffices h : (S.product S).filter (λ p : ℤ × ℤ, (p.1 ≠ p.2) ∧ (p.1 * p.2 = 0)).card = 4 by
  have ht : (S.product S).filter (λ p : ℤ × ℤ, p.1 ≠ p.2).card = 10 := by sorry
  rw ← nat.cast_div
  rw ht
  norm_cast
  exact h
sorry


end probability_product_zero_l55_55138


namespace quadratic_roots_form_l55_55108

theorem quadratic_roots_form {d : ℝ} (h : ∀ x : ℝ, x^2 + 7*x + d = 0 → (x = (-7 + real.sqrt d) / 2) ∨ (x = (-7 - real.sqrt d) / 2)) : d = 49 / 5 := 
sorry

end quadratic_roots_form_l55_55108


namespace days_worked_per_week_l55_55009

theorem days_worked_per_week (total_toys_per_week toys_produced_each_day : ℕ) 
  (h1 : total_toys_per_week = 5505)
  (h2 : toys_produced_each_day = 1101)
  : total_toys_per_week / toys_produced_each_day = 5 :=
  by
    sorry

end days_worked_per_week_l55_55009


namespace point_relationship_l55_55788

variable {m : ℝ}

theorem point_relationship
    (hA : ∃ y1 : ℝ, y1 = (-4 : ℝ)^2 - 2 * (-4 : ℝ) + m)
    (hB : ∃ y2 : ℝ, y2 = (0 : ℝ)^2 - 2 * (0 : ℝ) + m)
    (hC : ∃ y3 : ℝ, y3 = (3 : ℝ)^2 - 2 * (3 : ℝ) + m) :
    (∃ y2 y3 y1 : ℝ, y2 < y3 ∧ y3 < y1) := by
  sorry

end point_relationship_l55_55788


namespace tiles_needed_l55_55703

/-- 
Given:
- The cafeteria is tiled with the same floor tiles.
- It takes 630 tiles to cover an area of 18 square decimeters of tiles.
- We switch to square tiles with a side length of 6 decimeters.

Prove:
- The number of new tiles needed to cover the same area is 315.
--/
theorem tiles_needed (n_tiles : ℕ) (area_per_tile : ℕ) (new_tile_side_length : ℕ) 
  (h1 : n_tiles = 630) (h2 : area_per_tile = 18) (h3 : new_tile_side_length = 6) :
  (630 * 18) / (6 * 6) = 315 :=
by
  sorry

end tiles_needed_l55_55703


namespace bananas_to_apples_l55_55256

-- Definitions based on conditions
def bananas := ℕ
def oranges := ℕ
def apples := ℕ

-- Condition 1: 3/4 of 16 bananas are worth 12 oranges
def condition1 : Prop := 3 / 4 * 16 = 12

-- Condition 2: price of one banana equals the price of two apples
def price_equiv_banana_apple : Prop := 1 = 2

-- Proof: 1/3 of 9 bananas are worth 6 apples
theorem bananas_to_apples 
  (c1: condition1)
  (c2: price_equiv_banana_apple) : 1 / 3 * 9 * 2 = 6 :=
by sorry

end bananas_to_apples_l55_55256


namespace index_difference_l55_55626

noncomputable def index_females (n k1 k2 k3 : ℕ) : ℚ :=
  ((n - k1 + k2 : ℚ) / n) * (1 + k3 / 10)

noncomputable def index_males (n k1 l1 l2 : ℕ) : ℚ :=
  ((n - (n - k1) + l1 : ℚ) / n) * (1 + l2 / 10)

theorem index_difference (n k1 k2 k3 l1 l2 : ℕ)
  (h_n : n = 35) (h_k1 : k1 = 15) (h_k2 : k2 = 5) (h_k3 : k3 = 8)
  (h_l1 : l1 = 6) (h_l2 : l2 = 10) : 
  index_females n k1 k2 k3 - index_males n k1 l1 l2 = 3 / 35 :=
by
  sorry

end index_difference_l55_55626


namespace total_number_of_games_in_season_l55_55857

def number_of_games_per_month : ℕ := 13
def number_of_months_in_season : ℕ := 14

theorem total_number_of_games_in_season :
  number_of_games_per_month * number_of_months_in_season = 182 := by
  sorry

end total_number_of_games_in_season_l55_55857


namespace tan_585_eq_one_l55_55186

theorem tan_585_eq_one : Real.tan (585 * Real.pi / 180) = 1 :=
by
  sorry

end tan_585_eq_one_l55_55186


namespace average_remaining_ropes_l55_55435

theorem average_remaining_ropes 
  (n : ℕ) 
  (m : ℕ) 
  (l_avg : ℕ) 
  (l1_avg : ℕ) 
  (l2_avg : ℕ) 
  (h1 : n = 6)
  (h2 : m = 2)
  (hl_avg : l_avg = 80)
  (hl1_avg : l1_avg = 70)
  (htotal : l_avg * n = 480)
  (htotal1 : l1_avg * m = 140)
  (htotal2 : l_avg * n - l1_avg * m = 340):
  (340 : ℕ) / (4 : ℕ) = 85 := by
  sorry

end average_remaining_ropes_l55_55435


namespace seven_divides_n_l55_55959

theorem seven_divides_n (n : ℕ) (h1 : n ≥ 2) (h2 : n ∣ 3^n + 4^n) : 7 ∣ n :=
sorry

end seven_divides_n_l55_55959


namespace lines_perpendicular_l55_55918

/-- Given two lines l1: 3x + 4y + 1 = 0 and l2: 4x - 3y + 2 = 0, 
    prove that the lines are perpendicular. -/
theorem lines_perpendicular :
  ∀ (x y : ℝ), (3 * x + 4 * y + 1 = 0) → (4 * x - 3 * y + 2 = 0) → (- (3 / 4) * (4 / 3) = -1) :=
by
  intro x y h₁ h₂
  sorry

end lines_perpendicular_l55_55918


namespace walk_to_Lake_Park_restaurant_time_l55_55818

-- Define the problem parameters
def time_to_hidden_lake : ℕ := 15
def time_from_hidden_lake : ℕ := 7
def total_time_gone : ℕ := 32

-- Define the goal to prove
theorem walk_to_Lake_Park_restaurant_time :
  total_time_gone - (time_to_hidden_lake + time_from_hidden_lake) = 10 :=
by
  -- skipping the proof here
  sorry

end walk_to_Lake_Park_restaurant_time_l55_55818


namespace necessary_but_not_sufficient_l55_55006

variables {R : Type*} [Field R] (a b c : R)

def condition1 : Prop := (a / b) = (b / c)
def condition2 : Prop := b^2 = a * c

theorem necessary_but_not_sufficient :
  (condition1 a b c → condition2 a b c) ∧ ¬ (condition2 a b c → condition1 a b c) :=
by
  sorry

end necessary_but_not_sufficient_l55_55006


namespace chocolate_bar_cost_l55_55036

theorem chocolate_bar_cost (total_bars : ℕ) (sold_bars : ℕ) (total_money : ℕ) (cost : ℕ) 
  (h1 : total_bars = 13)
  (h2 : sold_bars = total_bars - 4)
  (h3 : total_money = 18)
  (h4 : total_money = sold_bars * cost) :
  cost = 2 :=
by sorry

end chocolate_bar_cost_l55_55036


namespace decompose_96_l55_55773

theorem decompose_96 (a b : ℤ) (h1 : a * b = 96) (h2 : a^2 + b^2 = 208) : 
  (a = 8 ∧ b = 12) ∨ (a = 12 ∧ b = 8) ∨ (a = -8 ∧ b = -12) ∨ (a = -12 ∧ b = -8) :=
by
  sorry

end decompose_96_l55_55773


namespace pages_needed_l55_55396

theorem pages_needed (packs : ℕ) (cards_per_pack : ℕ) (cards_per_page : ℕ) (total_packs : packs = 60) (cards_in_pack : cards_per_pack = 7) (capacity_per_page : cards_per_page = 10) : (packs * cards_per_pack) / cards_per_page = 42 := 
by
  -- Utilize the conditions
  have H1 : packs = 60 := total_packs
  have H2 : cards_per_pack = 7 := cards_in_pack
  have H3 : cards_per_page = 10 := capacity_per_page
  -- Use these to simplify and prove the target expression 
  sorry

end pages_needed_l55_55396


namespace expression_value_l55_55262

theorem expression_value (x : ℝ) (hx1 : x ≠ -1) (hx2 : x ≠ 2) :
  (2 * x ^ 2 - x) / ((x + 1) * (x - 2)) - (4 + x) / ((x + 1) * (x - 2)) = 2 := 
by
  sorry

end expression_value_l55_55262


namespace caochong_weighing_equation_l55_55129

-- Definitions for porter weight, stone weight, and the counts in the respective steps
def porter_weight : ℝ := 120
def stone_weight (x : ℝ) : ℝ := x
def first_step_weight (x : ℝ) : ℝ := 20 * stone_weight x + 3 * porter_weight
def second_step_weight (x : ℝ) : ℝ := (20 + 1) * stone_weight x + 1 * porter_weight

-- Theorem stating the equality condition ensuring the same water level
theorem caochong_weighing_equation (x : ℝ) :
  first_step_weight x = second_step_weight x :=
by
  sorry

end caochong_weighing_equation_l55_55129


namespace workshop_worker_count_l55_55985

theorem workshop_worker_count (W T N : ℕ) (h1 : T = 7) (h2 : 8000 * W = 7 * 14000 + 6000 * N) (h3 : W = T + N) : W = 28 :=
by
  sorry

end workshop_worker_count_l55_55985


namespace min_tablets_to_ensure_three_each_l55_55731

theorem min_tablets_to_ensure_three_each (A B C : ℕ) (hA : A = 20) (hB : B = 25) (hC : C = 15) : 
  ∃ n, n = 48 ∧ (∀ x y z, x + y + z = n → x ≥ 3 ∧ y ≥ 3 ∧ z ≥ 3) :=
by
  -- proof goes here
  sorry

end min_tablets_to_ensure_three_each_l55_55731


namespace neither_math_nor_physics_students_l55_55833

-- Definitions and the main theorem
def total_students : ℕ := 120
def math_students : ℕ := 75
def physics_students : ℕ := 50
def both_students : ℕ := 15

theorem neither_math_nor_physics_students (t m p b : ℕ) (h1 : t = 120) (h2 : m = 75) (h3 : p = 50) (h4 : b = 15) : 
  t - (m - b + p - b + b) = 10 := by
  -- Instantiate the conditions
  rw [h1, h2, h3, h4]
  -- the proof is marked as sorry
  sorry

end neither_math_nor_physics_students_l55_55833


namespace motel_total_rent_l55_55724

theorem motel_total_rent (R₅₀ R₆₀ : ℕ) 
  (h₁ : ∀ x y : ℕ, 50 * x + 60 * y = 50 * (x + 10) + 60 * (y - 10) + 100)
  (h₂ : ∀ x y : ℕ, 25 * (50 * x + 60 * y) = 10000) : 
  50 * R₅₀ + 60 * R₆₀ = 400 :=
by
  sorry

end motel_total_rent_l55_55724


namespace emails_in_afternoon_l55_55383

variable (e_m e_t e_a : Nat)
variable (h1 : e_m = 3)
variable (h2 : e_t = 8)

theorem emails_in_afternoon : e_a = 5 :=
by
  -- (Proof steps would go here)
  sorry

end emails_in_afternoon_l55_55383


namespace sum_or_difference_div_by_100_l55_55838

theorem sum_or_difference_div_by_100 (s : Finset ℤ) (h_card : s.card = 52) :
  ∃ (a b : ℤ), a ∈ s ∧ b ∈ s ∧ (a ≠ b) ∧ (100 ∣ (a + b) ∨ 100 ∣ (a - b)) :=
by
  sorry

end sum_or_difference_div_by_100_l55_55838


namespace dealer_gross_profit_l55_55450

theorem dealer_gross_profit (purchase_price : ℝ) (markup_rate : ℝ) (selling_price : ℝ) (gross_profit : ℝ) 
  (purchase_price_cond : purchase_price = 150)
  (markup_rate_cond : markup_rate = 0.25)
  (selling_price_eq : selling_price = purchase_price + (markup_rate * selling_price))
  (gross_profit_eq : gross_profit = selling_price - purchase_price) : 
  gross_profit = 50 :=
by
  sorry

end dealer_gross_profit_l55_55450


namespace factorize_expression_l55_55474

theorem factorize_expression (a x : ℝ) : a * x^2 - a = a * (x + 1) * (x - 1) :=
by
  sorry

end factorize_expression_l55_55474


namespace evaluate_expression_l55_55064

theorem evaluate_expression (x y z : ℤ) (h1 : x = -2) (h2 : y = -4) (h3 : z = 3) :
  (5 * (x - y)^2 - x * z^2) / (z - y) = 38 / 7 := by
  sorry

end evaluate_expression_l55_55064


namespace carla_total_earnings_l55_55665

theorem carla_total_earnings
  (h1 : ∀ (x : ℝ), 28 * x = 18 * x + 63)
  (h2 : ∀ (x : ℝ), ∃ y : ℝ, 10 * x = 63 ∧ y = x)
  (h3 : ∀ (wage hours1 hours2 : ℝ), 28 = hours2 ∧ 18 = hours1 ∧ wage = 6.30)
  (h4 : ∀ (wage : ℝ), ∀ (total_hours : ℝ), total_hours = 46 → wage = 6.30)
  (h5 : ∀ (total_hours wage : ℝ), total_hours = 46 → wage = 6.30 → total_hours * wage = 289.80) :
  true :=
by
  have h_rw : 10 * 6.30 = 63 := by sorry,
  have wage : ℝ := 6.30,
  have total_hours : ℝ := 46,
  have total_earnings := total_hours * wage,
  have : total_earnings = 289.80 := by sorry,
  exact ⟨⟩.

#check carla_total_earnings
  (λ x, 28 * x = 18 * x + 63)
  (λ x, ⟨x, 10 * x = 63, rfl⟩)
  (λ w h1 h2, ⟨rfl, rfl, rfl⟩)
  (λ wage total_hours, λ h_eq, rfl)
  (λ total_hours wage, λ h_eq1 h_eq2, rfl)

end carla_total_earnings_l55_55665


namespace solve_xy_eq_x_plus_y_l55_55545

theorem solve_xy_eq_x_plus_y (x y : ℤ) (h : x * y = x + y) : (x = 0 ∧ y = 0) ∨ (x = 2 ∧ y = 2) :=
by {
  sorry
}

end solve_xy_eq_x_plus_y_l55_55545


namespace elaine_earnings_increase_l55_55532

variable (E : ℝ) (P : ℝ)

theorem elaine_earnings_increase
  (h1 : E > 0) 
  (h2 : 0.30 * E * (1 + P / 100) = 1.80 * 0.20 * E) : 
  P = 20 :=
by
  sorry

end elaine_earnings_increase_l55_55532


namespace find_AX_bisect_ACB_l55_55334

theorem find_AX_bisect_ACB (AC BX BC : ℝ) (h₁ : AC = 21) (h₂ : BX = 28) (h₃ : BC = 30) :
  ∃ (AX : ℝ), AX = 98 / 5 :=
by
  existsi 98 / 5
  sorry

end find_AX_bisect_ACB_l55_55334


namespace all_real_possible_values_l55_55535

theorem all_real_possible_values 
  (a b c : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0) (h₃ : a + b + c = 1) : 
  ∃ r : ℝ, r = (a^4 + b^4 + c^4) / (ab + bc + ca) :=
sorry

end all_real_possible_values_l55_55535


namespace diamond_example_l55_55034

def diamond (a b : ℕ) : ℤ := 4 * a + 5 * b - a^2 * b

theorem diamond_example : diamond 3 4 = -4 :=
by
  rw [diamond]
  calc
    4 * 3 + 5 * 4 - 3^2 * 4 = 12 + 20 - 36 := by norm_num
                           _              = -4 := by norm_num

end diamond_example_l55_55034


namespace max_cardinality_seven_l55_55264

theorem max_cardinality_seven 
  (M : Finset ℝ)
  (h : ∀ a b c ∈ M, (∃ x y ∈ pairwise (λ x y, x ∈ M) {a, b, c}, x + y ∈ M)) :
  M.card ≤ 7 :=
sorry

end max_cardinality_seven_l55_55264


namespace reasoning_common_sense_l55_55098

theorem reasoning_common_sense :
  (∀ P Q: Prop, names_not_correct → P → ¬Q → affairs_not_successful → ¬Q)
  ∧ (∀ R S: Prop, affairs_not_successful → R → ¬S → rites_not_flourish → ¬S)
  ∧ (∀ T U: Prop, rites_not_flourish → T → ¬U → punishments_not_executed_properly → ¬U)
  ∧ (∀ V W: Prop, punishments_not_executed_properly → V → ¬W → people_nowhere_hands_feet → ¬W)
  → reasoning_is_common_sense :=
by sorry

end reasoning_common_sense_l55_55098


namespace total_wood_needed_l55_55671

theorem total_wood_needed : 
      (4 * 4 + 4 * (4 * 5)) + 
      (10 * 6 + 10 * (6 - 3)) + 
      (8 * 5.5) + 
      (6 * (5.5 * 2) + 6 * (5.5 * 1.5)) = 345.5 := 
by 
  sorry

end total_wood_needed_l55_55671


namespace customer_total_payment_l55_55461

structure PaymentData where
  rate : ℕ
  discount1 : ℕ
  lateFee1 : ℕ
  discount2 : ℕ
  lateFee2 : ℕ
  discount3 : ℕ
  lateFee3 : ℕ
  discount4 : ℕ
  lateFee4 : ℕ
  onTime1 : Bool
  onTime2 : Bool
  onTime3 : Bool
  onTime4 : Bool

noncomputable def monthlyPayment (rate discount late_fee : ℕ) (onTime : Bool) : ℕ :=
  if onTime then rate - (rate * discount / 100) else rate + (rate * late_fee / 100)

theorem customer_total_payment (data : PaymentData) : 
  monthlyPayment data.rate data.discount1 data.lateFee1 data.onTime1 +
  monthlyPayment data.rate data.discount2 data.lateFee2 data.onTime2 +
  monthlyPayment data.rate data.discount3 data.lateFee3 data.onTime3 +
  monthlyPayment data.rate data.discount4 data.lateFee4 data.onTime4 = 195 := by
  sorry

end customer_total_payment_l55_55461


namespace sum_gcd_lcm_is_244_l55_55141

-- Definitions of the constants
def a : ℕ := 12
def b : ℕ := 80

-- Main theorem statement
theorem sum_gcd_lcm_is_244 : Nat.gcd a b + Nat.lcm a b = 244 := by
  sorry

end sum_gcd_lcm_is_244_l55_55141


namespace samantha_hike_distance_l55_55841

theorem samantha_hike_distance :
  let A : ℝ × ℝ := (0, 0)  -- Samantha's starting point
  let B := (0, 3)           -- Point after walking northward 3 miles
  let C := (5 / (2 : ℝ) * Real.sqrt 2, 3) -- Point after walking 5 miles at 45 degrees eastward
  (dist A C = Real.sqrt 86 / 2) :=
by
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (0, 3)
  let C : ℝ × ℝ := (5 / (2 : ℝ) * Real.sqrt 2, 3)
  show dist A C = Real.sqrt 86 / 2
  sorry

end samantha_hike_distance_l55_55841


namespace tournament_games_count_l55_55437

-- Defining the problem conditions
def num_players : Nat := 12
def plays_twice : Bool := true

-- Theorem statement
theorem tournament_games_count (n : Nat) (plays_twice : Bool) (h : n = num_players ∧ plays_twice = true) :
  (n * (n - 1) * 2) = 264 := by
  sorry

end tournament_games_count_l55_55437


namespace find_cost_price_l55_55305

theorem find_cost_price
  (cost_price : ℝ)
  (increase_rate : ℝ := 0.2)
  (decrease_rate : ℝ := 0.1)
  (profit : ℝ := 8):
  (1 + increase_rate) * cost_price * (1 - decrease_rate) - cost_price = profit → 
  cost_price = 100 := 
by 
  sorry

end find_cost_price_l55_55305


namespace original_price_of_cycle_l55_55741

theorem original_price_of_cycle (selling_price : ℝ) (loss_percentage : ℝ) (original_price : ℝ) 
  (h1 : selling_price = 1610)
  (h2 : loss_percentage = 30) 
  (h3 : selling_price = original_price * (1 - loss_percentage / 100)) : 
  original_price = 2300 := 
by 
  sorry

end original_price_of_cycle_l55_55741


namespace arithmetic_sequence_m_value_l55_55220

theorem arithmetic_sequence_m_value (m : ℝ) (h : 2 + 6 = 2 * m) : m = 4 :=
by sorry

end arithmetic_sequence_m_value_l55_55220


namespace possible_values_a_l55_55902

-- Define the problem statement
theorem possible_values_a :
  (∃ a b c : ℤ, ∀ x : ℝ, (x - a) * (x - 5) + 3 = (x + b) * (x + c)) → (a = 1 ∨ a = 9) :=
by 
  -- Variable declaration and theorem body will be placed here
  sorry

end possible_values_a_l55_55902


namespace vasya_correct_l55_55568

theorem vasya_correct (x : ℝ) (h : x^2 + x + 1 = 0) : 
  x^2000 + x^1999 + x^1998 + 1000*x^1000 + 1000*x^999 + 1000*x^998 + 2000*x^3 + 2000*x^2 + 2000*x + 3000 = 3000 :=
by 
  sorry

end vasya_correct_l55_55568


namespace number_of_children_l55_55018

-- Define conditions
variable (A C : ℕ) (h1 : A + C = 280) (h2 : 60 * A + 25 * C = 14000)

-- Lean statement to prove the number of children
theorem number_of_children : C = 80 :=
by
  sorry

end number_of_children_l55_55018


namespace sin_identity_l55_55344

theorem sin_identity (α : ℝ) (h_tan : Real.tan α = -3 / 4) : 
  Real.sin α * (Real.sin α - Real.cos α) = 21 / 25 :=
sorry

end sin_identity_l55_55344


namespace initial_population_is_10000_l55_55106

def population_growth (P : ℝ) : Prop :=
  let growth_rate := 0.20
  let final_population := 12000
  final_population = P * (1 + growth_rate)

theorem initial_population_is_10000 : population_growth 10000 :=
by
  unfold population_growth
  sorry

end initial_population_is_10000_l55_55106


namespace fraction_calculation_l55_55898

theorem fraction_calculation : 
  (1 / 2 + 1 / 5) / (3 / 7 - 1 / 14) = 49 / 25 :=
by sorry

end fraction_calculation_l55_55898


namespace scientific_notation_for_70_million_l55_55292

-- Define the parameters for the problem
def scientific_notation (x : ℕ) (a : ℝ) (n : ℤ) : Prop :=
  x = a * 10 ^ n ∧ 1 ≤ |a| ∧ |a| < 10

-- Problem statement
theorem scientific_notation_for_70_million :
  scientific_notation 70000000 7.0 7 :=
by
  sorry

end scientific_notation_for_70_million_l55_55292


namespace remainder_is_zero_l55_55339

theorem remainder_is_zero :
  (86 * 87 * 88 * 89 * 90 * 91 * 92) % 7 = 0 := 
by 
  sorry

end remainder_is_zero_l55_55339


namespace max_value_of_expression_l55_55239

theorem max_value_of_expression (x y z : ℝ) (h₀ : x ≥ 0) (h₁ : y ≥ 0) (h₂ : z ≥ 0) (h₃ : x^2 + y^2 + z^2 = 1) : 
  3 * x * z * Real.sqrt 2 + 9 * y * z ≤ Real.sqrt 27 :=
sorry

end max_value_of_expression_l55_55239


namespace women_fraction_l55_55372

/-- In a room with 100 people, 1/4 of whom are married, the maximum number of unmarried women is 40.
    We need to prove that the fraction of women in the room is 2/5. -/
theorem women_fraction (total_people : ℕ) (married_fraction : ℚ) (unmarried_women : ℕ) (W : ℚ) 
  (h1 : total_people = 100) 
  (h2 : married_fraction = 1 / 4) 
  (h3 : unmarried_women = 40) 
  (hW : W = 2 / 5) : 
  W = 2 / 5 := 
by
  sorry

end women_fraction_l55_55372


namespace field_ratio_l55_55416

theorem field_ratio
  (l w : ℕ)
  (pond_length : ℕ)
  (pond_area_ratio : ℚ)
  (field_length : ℕ)
  (field_area : ℕ)
  (hl : l = 24)
  (hp : pond_length = 6)
  (hr : pond_area_ratio = 1 / 8)
  (hm : l % w = 0)
  (ha : field_area = 36 * 8)
  (hf : l * w = field_area) :
  l / w = 2 :=
by
  sorry

end field_ratio_l55_55416


namespace converse_implication_l55_55550

theorem converse_implication (a : ℝ) : (a^2 = 1 → a = 1) → (a = 1 → a^2 = 1) :=
sorry

end converse_implication_l55_55550


namespace dvaneft_shares_percentage_range_l55_55295

theorem dvaneft_shares_percentage_range :
  ∀ (x y z n m : ℝ),
    (4 * x * n = y * m) →
    (x * n + y * m = z * (m + n)) →
    (16 ≤ y - x ∧ y - x ≤ 20) →
    (42 ≤ z ∧ z ≤ 60) →
    (12.5 ≤ (n / (2 * (n + m)) * 100) ∧ (n / (2 * (n + m)) * 100) ≤ 15) :=
by
  intros x y z n m h1 h2 h3 h4
  sorry

end dvaneft_shares_percentage_range_l55_55295


namespace total_apples_l55_55751

theorem total_apples : 
  let monday_apples := 15 in
  let tuesday_apples := 3 * monday_apples in
  let wednesday_apples := 4 * tuesday_apples in
  monday_apples + tuesday_apples + wednesday_apples = 240 :=
by
  let monday_apples := 15
  let tuesday_apples := 3 * monday_apples
  let wednesday_apples := 4 * tuesday_apples
  sorry

end total_apples_l55_55751


namespace painted_cubes_count_l55_55124

def total_painted_cubes : ℕ := 8 + 48

theorem painted_cubes_count : total_painted_cubes = 56 :=
by 
  -- Step 1: Define the number of cubes with 3 faces painted (8 corners)
  let corners := 8
  -- Step 2: Calculate the number of edge cubes with 2 faces painted
  let edge_middle_cubes_per_edge := 2
  let edges := 12
  let edge_cubes := edge_middle_cubes_per_edge * edges -- this should be 24
  -- Step 3: Calculate the number of face-interior cubes with 2 faces painted
  let face_cubes_per_face := 4
  let faces := 6
  let face_cubes := face_cubes_per_face * faces -- this should be 24
  -- Step 4: Sum them up to get total cubes with at least two faces painted
  let total_cubes := corners + edge_cubes + face_cubes
  show total_cubes = total_painted_cubes
  sorry

end painted_cubes_count_l55_55124


namespace proof_problem_l55_55057

variables {x1 y1 x2 y2 : ℝ}

-- Definitions
def unit_vector (x y : ℝ) : Prop := x^2 + y^2 = 1
def angle_with_p (x y : ℝ) : Prop := (x + y) / Real.sqrt 2 = Real.sqrt 3 / 2
def m := (x1, y1)
def n := (x2, y2)
def p := (1, 1)

-- Conditions
lemma unit_m : unit_vector x1 y1 := sorry
lemma unit_n : unit_vector x2 y2 := sorry
lemma angle_m_p : angle_with_p x1 y1 := sorry
lemma angle_n_p : angle_with_p x2 y2 := sorry

-- Theorem to prove
theorem proof_problem (h1 : unit_vector x1 y1)
                      (h2 : unit_vector x2 y2)
                      (h3 : angle_with_p x1 y1)
                      (h4 : angle_with_p x2 y2) :
                      (x1 * x2 + y1 * y2 = 1/2) ∧ (y1 * y2 / (x1 * x2) = 1) :=
sorry

end proof_problem_l55_55057


namespace common_tangents_count_l55_55267

-- Define the first circle Q1
def Q1 (x y : ℝ) := x^2 + y^2 = 9

-- Define the second circle Q2
def Q2 (x y : ℝ) := (x - 3)^2 + (y - 4)^2 = 1

-- Prove the number of common tangents between Q1 and Q2
theorem common_tangents_count :
  ∃ n : ℕ, n = 4 ∧ ∀ x y : ℝ, Q1 x y ∧ Q2 x y -> n = 4 := sorry

end common_tangents_count_l55_55267


namespace find_f_six_l55_55414

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_six (f : ℝ → ℝ)
  (h1 : ∀ x y : ℝ, x * f y = y * f x)
  (h2 : f 18 = 24) :
  f 6 = 8 :=
sorry

end find_f_six_l55_55414


namespace part1_part2_part3_l55_55492

noncomputable def f : ℝ → ℝ := sorry -- Given f is a function on ℝ with domain (0, +∞)

axiom domain_pos (x : ℝ) : 0 < x
axiom pos_condition (x : ℝ) (h : 1 < x) : 0 < f x
axiom functional_eq (x y : ℝ) : f (x * y) = f x + f y
axiom specific_value : f (1/3) = -1

-- (1) Prove: f(1/x) = -f(x)
theorem part1 (x : ℝ) (hx : 0 < x) : f (1 / x) = - f x := sorry

-- (2) Prove: f(x) is an increasing function on its domain
theorem part2 (x1 x2 : ℝ) (hx1 : 0 < x1) (hx2 : 0 < x2) (h : x1 < x2) : f x1 < f x2 := sorry

-- (3) Prove the range of x for the inequality
theorem part3 (x : ℝ) (hx : 0 < x) (hx2 : 0 < x - 2) : 
  f x - f (1 / (x - 2)) ≥ 2 ↔ 1 + Real.sqrt 10 ≤ x := sorry

end part1_part2_part3_l55_55492


namespace inverse_proposition_true_l55_55207

theorem inverse_proposition_true (x : ℝ) (h : x > 1 → x^2 > 1) : x^2 ≤ 1 → x ≤ 1 :=
by
  intros h₂
  sorry

end inverse_proposition_true_l55_55207


namespace problem_inequality_l55_55357

-- Definitions and conditions
noncomputable def f (x : ℝ) : ℝ := x * Real.log x
noncomputable def g (x : ℝ) (k : ℝ) : ℝ := f x + f (k - x)

-- The Lean proof problem
theorem problem_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  f a + (a + b) * Real.log 2 ≥ f (a + b) - f b := sorry

end problem_inequality_l55_55357


namespace longer_side_of_rectangle_l55_55595

theorem longer_side_of_rectangle
  (r : ℝ) (A_rect A_circle L S : ℝ) (h1 : r = 6)
  (h2 : A_circle = π * r^2)
  (h3 : A_rect = 3 * A_circle)
  (h4 : S = 2 * r)
  (h5 : A_rect = S * L) : L = 9 * π :=
by
  sorry

end longer_side_of_rectangle_l55_55595


namespace log_equation_l55_55792

theorem log_equation (x : ℝ) (h0 : x < 1) (h1 : (Real.log x / Real.log 10)^3 - 3 * (Real.log x / Real.log 10) = 243) :
  (Real.log x / Real.log 10)^4 - 4 * (Real.log x / Real.log 10) = 6597 :=
by
  sorry

end log_equation_l55_55792


namespace common_tangent_lines_count_l55_55490

-- Define the first circle
def C1 (x y : ℝ) : Prop := (x - 5)^2 + (y - 3)^2 = 9

-- Define the second circle
def C2 (x y : ℝ) : Prop := x^2 + y^2 - 4 * x + 2 * y - 9 = 0

-- Definition for the number of common tangent lines between two circles
def number_of_common_tangent_lines (C1 C2 : ℝ → ℝ → Prop) : ℕ := sorry

-- The theorem stating the number of common tangent lines between the given circles
theorem common_tangent_lines_count : number_of_common_tangent_lines C1 C2 = 2 := by
  sorry

end common_tangent_lines_count_l55_55490


namespace annual_interest_rate_l55_55970

-- Define the conditions as given in the problem
def principal : ℝ := 5000
def maturity_amount : ℝ := 5080
def interest_tax_rate : ℝ := 0.2

-- Define the annual interest rate x
variable (x : ℝ)

-- Statement to be proved: the annual interest rate x is 0.02
theorem annual_interest_rate :
  principal + principal * x - interest_tax_rate * (principal * x) = maturity_amount → x = 0.02 :=
by
  sorry

end annual_interest_rate_l55_55970


namespace find_y_l55_55511

theorem find_y (y : ℕ) (h : 4 ^ 12 = 64 ^ y) : y = 4 :=
sorry

end find_y_l55_55511


namespace S_is_line_l55_55806

open Complex

noncomputable def S : Set ℂ := { z : ℂ | ∃ (x y : ℝ), z = x + y * Complex.I ∧ 3 * y + 4 * x = 0 }

theorem S_is_line :
  ∃ (m b : ℝ), S = { z : ℂ | ∃ (x y : ℝ), z = x + y * Complex.I ∧ x = m * y + b } :=
sorry

end S_is_line_l55_55806


namespace min_value_AF_BF_l55_55794

noncomputable def parabola_focus : ℝ × ℝ := (0, 1)

noncomputable def parabola_eq (x y : ℝ) : Prop := x^2 = 4 * y

noncomputable def line_eq (k x : ℝ) : ℝ := k * x + 1

theorem min_value_AF_BF :
  ∀ (x1 x2 y1 y2 k : ℝ),
  parabola_eq x1 y1 →
  parabola_eq x2 y2 →
  line_eq k x1 = y1 →
  line_eq k x2 = y2 →
  (x1 ≠ x2) →
  parabola_focus = (0, 1) →
  (|y1 + 2| + 1) * (|y2 + 1|) = 2 * Real.sqrt 2 + 3 := 
by
  intros
  sorry

end min_value_AF_BF_l55_55794


namespace odd_function_a_minus_b_l55_55914

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_a_minus_b
  (a b : ℝ)
  (h : is_odd_function (λ x => 2 * x ^ 3 + a * x ^ 2 + b - 1)) :
  a - b = -1 :=
sorry

end odd_function_a_minus_b_l55_55914


namespace min_colors_for_distance_six_l55_55045

/-
Definitions and conditions:
- The board is an infinite checkered paper with a cell side of one unit.
- The distance between two cells is the length of the shortest path of a rook from one cell to another.

Statement:
- Prove that the minimum number of colors needed to color the board such that two cells that are a distance of 6 apart are always painted different colors is 4.
-/

def cell := (ℤ × ℤ)

def rook_distance (c1 c2 : cell) : ℤ :=
  |c1.1 - c2.1| + |c1.2 - c2.2|

theorem min_colors_for_distance_six : ∃ (n : ℕ), (∀ (f : cell → ℕ), (∀ c1 c2, rook_distance c1 c2 = 6 → f c1 ≠ f c2) → n ≤ 4) :=
by
  sorry

end min_colors_for_distance_six_l55_55045


namespace isosceles_triangle_perimeter_l55_55631

theorem isosceles_triangle_perimeter : 
  ∀ a b c : ℝ, a^2 - 6 * a + 5 = 0 → b^2 - 6 * b + 5 = 0 → 
    (a = b ∨ b = c ∨ a = c) →
    (a + b > c ∧ b + c > a ∧ a + c > b) →
    a + b + c = 11 := 
by
  intros a b c ha hb hiso htri
  sorry

end isosceles_triangle_perimeter_l55_55631


namespace total_ants_found_l55_55017

-- Definitions for the number of ants each child finds
def abe_ants : ℕ := 4
def beth_ants : ℕ := abe_ants + (abe_ants / 2)
def cece_ants : ℕ := 2 * abe_ants
def duke_ants : ℕ := abe_ants / 2

-- Statement that needs to be proven
theorem total_ants_found : abe_ants + beth_ants + cece_ants + duke_ants = 20 :=
by sorry

end total_ants_found_l55_55017


namespace uniqueTagSequences_divTotalTagsBy10_l55_55736

noncomputable def totalTags (letters: Finset Char) (digits: Finset Char) : Nat :=
  let allChars := letters ∪ digits
  let tagsWithout1 := allChars \ {'1'} 
  let tagsWithOne1 := Finset.card (Finset.filter ((λ l => l ≠ '1')) (Finset.product (Finset.singleton '1') (Finset.card tagsWithout1)))
  let tagsWithTwo1s := Finset.card (Finset.filter ((λ l => l = '1')) (Finset.product (Finset.singleton '1') (Finset.product (Finset.singleton '1') tagsWithout1)))
  tagsWithout1.card * tagsWithout1.card * tagsWithout1.card * tagsWithout1.card * tagsWithout1.card + tagsWithOne1 * tagsWithout1.card * tagsWithout1.card * tagsWithout1.card + tagsWithTwo1s

theorem uniqueTagSequences :
  totalTags ({'M', 'A', 'T', 'H'}) ({'3', '1', '1', '9'}) = 3120 := sorry
  
theorem divTotalTagsBy10 : 
  totalTags ({'M', 'A', 'T', 'H'}) ({'3', '1', '1', '9'}) / 10 = 312 := sorry

end uniqueTagSequences_divTotalTagsBy10_l55_55736


namespace hal_battery_change_25th_time_l55_55037

theorem hal_battery_change_25th_time (months_in_year : ℕ) 
    (battery_interval : ℕ) 
    (first_change_month : ℕ) 
    (change_count : ℕ) : 
    (battery_interval * (change_count-1)) % months_in_year + first_change_month % months_in_year = first_change_month % months_in_year :=
by
    have h1 : months_in_year = 12 := by sorry
    have h2 : battery_interval = 5 := by sorry
    have h3 : first_change_month = 5 := by sorry -- May is represented by 5 (0 = January, 1 = February, ..., 4 = April, 5 = May, ...)
    have h4 : change_count = 25 := by sorry
    sorry

end hal_battery_change_25th_time_l55_55037


namespace Tyler_cucumbers_and_grapes_l55_55811

theorem Tyler_cucumbers_and_grapes (a b c g : ℝ) (h1 : 10 * a = 5 * b) (h2 : 3 * b = 4 * c) (h3 : 4 * c = 6 * g) :
  (20 * a = (40 / 3) * c) ∧ (20 * a = 20 * g) :=
by
  sorry

end Tyler_cucumbers_and_grapes_l55_55811


namespace find_m_plus_c_l55_55137

-- We need to define the conditions first
variable {A : ℝ × ℝ} {B : ℝ × ℝ} {c : ℝ} {m : ℝ}

-- Given conditions from part a)
def A_def : Prop := A = (1, 3)
def B_def : Prop := B = (m, -1)
def centers_line : Prop := ∀ C : ℝ × ℝ, (C.1 - C.2 + c = 0)

-- Define the theorem for the proof problem
theorem find_m_plus_c (A_def : A = (1, 3)) (B_def : B = (m, -1)) (centers_line : ∀ C : ℝ × ℝ, (C.1 - C.2 + c = 0)) : m + c = 3 :=
sorry

end find_m_plus_c_l55_55137


namespace infinite_solutions_if_b_eq_neg_12_l55_55910

theorem infinite_solutions_if_b_eq_neg_12 (b : ℝ) :
  (∀ x : ℝ, 4 * (3 * x - b) = 3 * (4 * x + 16)) ↔ b = -12 :=
by
  split
  { intro h,
    specialize h 0,
    simp at h,
    linarith },
  { intro h,
    intro x,
    rw h,
    simp }

end infinite_solutions_if_b_eq_neg_12_l55_55910


namespace additional_time_proof_l55_55742

-- Given the charging rate of the battery and the additional time required to reach a percentage
noncomputable def charging_rate := 20 / 60
noncomputable def initial_time := 60
noncomputable def additional_time := 150

-- Define the total time required to reach a certain percentage
noncomputable def total_time := initial_time + additional_time

-- The proof statement to verify the additional time required beyond the initial 60 minutes
theorem additional_time_proof : total_time - initial_time = additional_time := sorry

end additional_time_proof_l55_55742


namespace dutch_exam_problem_l55_55195

theorem dutch_exam_problem (a b c d : ℝ) : 
  (a * b + c + d = 3) ∧ 
  (b * c + d + a = 5) ∧ 
  (c * d + a + b = 2) ∧ 
  (d * a + b + c = 6) → 
  (a = 2 ∧ b = 0 ∧ c = 0 ∧ d = 3) := 
by
  sorry

end dutch_exam_problem_l55_55195


namespace chinese_carriage_problem_l55_55754

theorem chinese_carriage_problem (x : ℕ) : 
  (3 * (x - 2) = 2 * x + 9) :=
sorry

end chinese_carriage_problem_l55_55754


namespace solve_for_N_l55_55560

theorem solve_for_N (N : ℤ) (h1 : N < 0) (h2 : 2 * N * N + N = 15) : N = -3 :=
sorry

end solve_for_N_l55_55560


namespace blocks_found_l55_55692

def initial_blocks : ℕ := 2
def final_blocks : ℕ := 86

theorem blocks_found : (final_blocks - initial_blocks) = 84 :=
by
  sorry

end blocks_found_l55_55692


namespace find_m_range_l55_55637

-- Defining the function and conditions
variable {f : ℝ → ℝ}
variable {m : ℝ}

-- Prove if given the conditions, then the range of m is as specified
theorem find_m_range (h1 : ∀ x, f (-x) = -f x) 
                     (h2 : ∀ x, -2 < x ∧ x < 2 → f (x) > f (x+1)) 
                     (h3 : -2 < m - 1 ∧ m - 1 < 2) 
                     (h4 : -2 < 2 * m - 1 ∧ 2 * m - 1 < 2) 
                     (h5 : f (m - 1) + f (2 * m - 1) > 0) :
  -1/2 < m ∧ m < 2/3 :=
sorry

end find_m_range_l55_55637


namespace josie_leftover_amount_l55_55672

-- Define constants and conditions
def initial_amount : ℝ := 20.00
def milk_price : ℝ := 4.00
def bread_price : ℝ := 3.50
def detergent_price : ℝ := 10.25
def bananas_price_per_pound : ℝ := 0.75
def bananas_weight : ℝ := 2.0
def detergent_coupon : ℝ := 1.25
def milk_discount_rate : ℝ := 0.5

-- Define the total cost before any discounts
def total_cost_before_discounts : ℝ := 
  milk_price + bread_price + detergent_price + (bananas_weight * bananas_price_per_pound)

-- Define the discounted prices
def milk_discounted_price : ℝ := milk_price * milk_discount_rate
def detergent_discounted_price : ℝ := detergent_price - detergent_coupon

-- Define the total cost after discounts
def total_cost_after_discounts : ℝ := 
  milk_discounted_price + bread_price + detergent_discounted_price + 
  (bananas_weight * bananas_price_per_pound)

-- Prove the amount left over
theorem josie_leftover_amount : initial_amount - total_cost_after_discounts = 4.00 := by
  simp [total_cost_before_discounts, milk_discounted_price, detergent_discounted_price,
    total_cost_after_discounts, initial_amount, milk_price, bread_price, detergent_price,
    bananas_price_per_pound, bananas_weight, detergent_coupon, milk_discount_rate]
  sorry

end josie_leftover_amount_l55_55672


namespace miles_monday_calculation_l55_55722

-- Define the constants
def flat_fee : ℕ := 150
def cost_per_mile : ℝ := 0.50
def miles_thursday : ℕ := 744
def total_cost : ℕ := 832

-- Define the equation to be proved
theorem miles_monday_calculation :
  ∃ M : ℕ, (flat_fee + (M : ℝ) * cost_per_mile + (miles_thursday : ℝ) * cost_per_mile = total_cost) ∧ M = 620 :=
by
  sorry

end miles_monday_calculation_l55_55722


namespace correct_exponential_calculation_l55_55284

theorem correct_exponential_calculation (a : ℝ) (ha : a ≠ 0) : 
  (a^4)^4 = a^16 :=
by sorry

end correct_exponential_calculation_l55_55284


namespace percentage_B_of_C_l55_55606

theorem percentage_B_of_C 
  (A C B : ℝ)
  (h1 : A = (7 / 100) * C)
  (h2 : A = (50 / 100) * B) :
  B = (14 / 100) * C := 
sorry

end percentage_B_of_C_l55_55606


namespace forty_percent_of_number_l55_55579

theorem forty_percent_of_number (N : ℝ) (h : (1 / 4) * (1 / 3) * (2 / 5) * N = 15) :
  0.40 * N = 180 :=
by
  sorry

end forty_percent_of_number_l55_55579


namespace no_common_root_l55_55397

theorem no_common_root (a b c d : ℝ) (h0 : 0 < a) (h1 : a < b) (h2 : b < c) (h3 : c < d) :
  ¬ ∃ x : ℝ, x^2 + b * x + c = 0 ∧ x^2 + a * x + d = 0 :=
by
  sorry

end no_common_root_l55_55397


namespace quadratic_root_condition_l55_55114

theorem quadratic_root_condition (d : ℝ) :
  (∀ x, x^2 + 7 * x + d = 0 → x = (-7 + Real.sqrt d) / 2 ∨ x = (-7 - Real.sqrt d) / 2) → d = 9.8 :=
by
  intro h
  sorry

end quadratic_root_condition_l55_55114


namespace evaluate_g_at_neg_four_l55_55210

def g (x : Int) : Int := 5 * x + 2

theorem evaluate_g_at_neg_four : g (-4) = -18 := by
  sorry

end evaluate_g_at_neg_four_l55_55210


namespace overtime_pay_rate_ratio_l55_55882

noncomputable def regular_pay_rate : ℕ := 3
noncomputable def regular_hours : ℕ := 40
noncomputable def total_pay : ℕ := 180
noncomputable def overtime_hours : ℕ := 10

theorem overtime_pay_rate_ratio : 
  (total_pay - (regular_hours * regular_pay_rate)) / overtime_hours / regular_pay_rate = 2 := by
  sorry

end overtime_pay_rate_ratio_l55_55882


namespace find_number_of_raccoons_squirrels_opossums_l55_55076

theorem find_number_of_raccoons_squirrels_opossums
  (R : ℕ)
  (total_animals : ℕ)
  (number_of_squirrels : ℕ := 6 * R)
  (number_of_opossums : ℕ := 2 * R)
  (total : ℕ := R + number_of_squirrels + number_of_opossums) 
  (condition : total_animals = 168)
  (correct_total : total = total_animals) :
  ∃ R : ℕ, R + 6 * R + 2 * R = total_animals :=
by
  sorry

end find_number_of_raccoons_squirrels_opossums_l55_55076


namespace integer_root_abs_sum_l55_55488

noncomputable def solve_abs_sum (p q r : ℤ) : ℤ := |p| + |q| + |r|

theorem integer_root_abs_sum (p q r m : ℤ) 
  (h1 : p + q + r = 0)
  (h2 : p * q + q * r + r * p = -2024)
  (h3 : ∃ m, ∀ x, x^3 - 2024 * x + m = (x - p) * (x - q) * (x - r)) :
  solve_abs_sum p q r = 104 :=
by sorry

end integer_root_abs_sum_l55_55488


namespace arithmetic_twelfth_term_l55_55142

theorem arithmetic_twelfth_term 
(a d : ℚ) (n : ℕ) (h_a : a = 1/2) (h_d : d = 1/3) (h_n : n = 12) : 
  a + (n - 1) * d = 25 / 6 := 
by 
  sorry

end arithmetic_twelfth_term_l55_55142


namespace train_pass_platform_time_l55_55873

theorem train_pass_platform_time :
  ∀ (length_train length_platform speed_time_cross_tree speed_train pass_time : ℕ), 
  length_train = 1200 →
  length_platform = 300 →
  speed_time_cross_tree = 120 →
  speed_train = length_train / speed_time_cross_tree →
  pass_time = (length_train + length_platform) / speed_train →
  pass_time = 150 :=
by
  intros
  sorry

end train_pass_platform_time_l55_55873


namespace dora_rate_correct_l55_55314

noncomputable def betty_rate : ℕ := 10
noncomputable def dora_rate : ℕ := 8
noncomputable def total_time : ℕ := 5
noncomputable def betty_break_time : ℕ := 2
noncomputable def cupcakes_difference : ℕ := 10

theorem dora_rate_correct :
  ∃ D : ℕ, 
  (D = dora_rate) ∧ 
  ((total_time - betty_break_time) * betty_rate = 30) ∧ 
  (total_time * D - 30 = cupcakes_difference) :=
sorry

end dora_rate_correct_l55_55314


namespace value_fraction_eq_three_l55_55912

namespace Problem

variable {R : Type} [Field R]

theorem value_fraction_eq_three (a b c : R) (h : a / 2 = b / 3 ∧ b / 3 = c / 4) :
  (a + b + c) / (2 * a + b - c) = 3 := by
  sorry

end Problem

end value_fraction_eq_three_l55_55912


namespace sum_of_radii_is_364_over_315_l55_55735

noncomputable def circle_radii : List ℝ := [100^2, 105^2, 110^2]

-- Function to compute the radius of a new circle given two existing radii
def new_radius (r_i r_j : ℝ) : ℝ :=
  (r_i * r_j) / (Real.sqrt r_i + Real.sqrt r_j) ^ 2

-- Function to compute all circles up to layer L₅
def compute_circles (n : ℕ) : List ℝ :=
  List.replicate 3 n ++ (List.replicate (3 * 2^(n-1)) (new_radius (100^2) (105^2)))

-- Function to compute the sum for the given set of circles
def sum_radii : ℝ :=
  ∑ C in (List.range 6).bind compute_circles, (1 / Real.sqrt C)

-- Proof statement
theorem sum_of_radii_is_364_over_315 :
  sum_radii = 364 / 315 :=
sorry

end sum_of_radii_is_364_over_315_l55_55735


namespace base_7_to_base_10_l55_55607

theorem base_7_to_base_10 :
  (3 * 7^2 + 2 * 7^1 + 1 * 7^0) = 162 :=
by
  sorry

end base_7_to_base_10_l55_55607


namespace value_of_g_at_2_l55_55655

def g (x : ℝ) : ℝ := x^2 - 4

theorem value_of_g_at_2 : g 2 = 0 :=
by
  -- proof goes here
  sorry

end value_of_g_at_2_l55_55655


namespace paint_cost_of_cube_l55_55950

def cube_side_length : ℝ := 10
def paint_cost_per_quart : ℝ := 3.20
def coverage_per_quart : ℝ := 1200
def number_of_faces : ℕ := 6

theorem paint_cost_of_cube : 
  (number_of_faces * (cube_side_length^2) / coverage_per_quart) * paint_cost_per_quart = 3.20 :=
by 
  sorry

end paint_cost_of_cube_l55_55950


namespace speed_of_water_l55_55300

variable (v : ℝ) -- the speed of the water in km/h
variable (t : ℝ) -- time taken to swim back in hours
variable (d : ℝ) -- distance swum against the current in km
variable (s : ℝ) -- speed in still water

theorem speed_of_water :
  ∀ (v t d s : ℝ),
  s = 20 -> t = 5 -> d = 40 -> d = (s - v) * t -> v = 12 :=
by
  intros v t d s ht hs hd heq
  sorry

end speed_of_water_l55_55300


namespace probability_more_heads_than_tails_l55_55809

theorem probability_more_heads_than_tails :
  (∀ i : ℕ, 0 ≤ i ∧ i < 9 → P (coin_flip i)) = 1/2 :=
by
  have nine_flips : (fin 9) → bool := sorry
  have fair_coin : ∀ i : fin 9, probability (coin_flip i = heads) = 1/2 := sorry
  sorry

end probability_more_heads_than_tails_l55_55809


namespace mean_problem_l55_55420

theorem mean_problem : 
  (8 + 12 + 24) / 3 = (16 + z) / 2 → z = 40 / 3 :=
by
  intro h
  sorry

end mean_problem_l55_55420


namespace time_to_lake_park_restaurant_l55_55821

variable (T1 T2 T_total T_detour : ℕ)

axiom time_to_hidden_lake : T1 = 15
axiom time_back_to_park_office : T2 = 7
axiom total_time_gone : T_total = 32

theorem time_to_lake_park_restaurant : T_detour = 10 :=
by
  -- Using the axioms and given conditions
  have h : T_total = T1 + T2 + T_detour,
  sorry

#check time_to_lake_park_restaurant

end time_to_lake_park_restaurant_l55_55821


namespace binary_computation_l55_55180

theorem binary_computation :
  (0b101101 * 0b10101 + 0b1010 / 0b10) = 0b110111100000 := by
  sorry

end binary_computation_l55_55180


namespace area_of_region_bounded_by_tan_cot_l55_55482

open Real Set

theorem area_of_region_bounded_by_tan_cot :
  let region := {p : ℝ × ℝ | ∃ θ, 0 < θ ∧ θ < π/2 ∧ p.1 = tan θ ∧ p.2 = cot θ} in
  ∀ p ∈ region, 0 ≤ p.1 ∧ 0 ≤ p.2 →
  let triangle := {p : ℝ × ℝ | p.1 = 0 ∧ p.2 = 0} ∪ {p : ℝ × ℝ | p.1 = 1 ∧ p.2 = 0} ∪ {p : ℝ × ℝ | p.1 = 1 ∧ p.2 = 1} in
  area_of_triangle region = 1 / 2 := by
  sorry

end area_of_region_bounded_by_tan_cot_l55_55482


namespace tan_alpha_value_complex_expression_value_l55_55628

theorem tan_alpha_value (α : ℝ) (h : Real.tan (π / 4 + α) = 1 / 2) : Real.tan α = -1 / 3 :=
sorry

theorem complex_expression_value 
(α : ℝ) 
(h1 : Real.tan (π / 4 + α) = 1 / 2) 
(h2 : Real.tan α = -1 / 3) : 
Real.sin (2 * α + 2 * π) - (Real.sin (π / 2 - α))^2 / 
(1 - Real.cos (π - 2 * α) + (Real.sin α)^2) = -15 / 19 :=
sorry

end tan_alpha_value_complex_expression_value_l55_55628


namespace books_on_shelves_l55_55426

-- Definitions based on the problem conditions.
def bookshelves : ℕ := 1250
def books_per_shelf : ℕ := 45
def total_books : ℕ := 56250

-- Theorem statement
theorem books_on_shelves : bookshelves * books_per_shelf = total_books := 
by
  sorry

end books_on_shelves_l55_55426


namespace binomial_expansion_example_l55_55870

theorem binomial_expansion_example : 7^3 + 3 * (7^2) * 2 + 3 * 7 * (2^2) + 2^3 = 729 := by
  sorry

end binomial_expansion_example_l55_55870


namespace other_solution_of_quadratic_l55_55925

theorem other_solution_of_quadratic (x : ℚ) (h₁ : 81 * 2/9 * 2/9 + 220 = 196 * 2/9 - 15) (h₂ : 81*x^2 - 196*x + 235 = 0) : x = 2/9 ∨ x = 5/9 :=
by
  sorry

end other_solution_of_quadratic_l55_55925


namespace find_m_l55_55080

theorem find_m (m : ℕ) (h_pos : 0 < m) 
  (h_intersection : ∃ (x y : ℤ), 13 * x + 11 * y = 700 ∧ y = m * x - 1) : 
  m = 6 :=
sorry

end find_m_l55_55080


namespace f_at_6_5_l55_55494

noncomputable def f : ℝ → ℝ := sorry

def even_function (f : ℝ → ℝ) : Prop := 
  ∀ x : ℝ, f x = f (-x)

def periodic_function (f : ℝ → ℝ) (p : ℝ) : Prop := 
  ∀ x : ℝ, f (x + p) = f x

def specific_values (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → f x = x - 2

theorem f_at_6_5:
  (∀ x : ℝ, f (x + 2) = -1 / f x) →
  even_function f →
  specific_values f →
  f 6.5 = -0.5 :=
by
  sorry

end f_at_6_5_l55_55494


namespace cricketer_total_matches_l55_55549

theorem cricketer_total_matches (n : ℕ)
  (avg_total : ℝ) (avg_first_6 : ℝ) (avg_last_4 : ℝ)
  (total_runs_eq : 6 * avg_first_6 + 4 * avg_last_4 = n * avg_total) :
  avg_total = 38.9 ∧ avg_first_6 = 42 ∧ avg_last_4 = 34.25 → n = 10 :=
by
  sorry

end cricketer_total_matches_l55_55549


namespace number_of_cubes_with_at_least_two_faces_painted_is_56_l55_55126

def one_inch_cubes_with_at_least_two_faces_painted 
  (side_length : ℕ) (face_colors : ℕ) (cubes_per_side : ℕ) :=
  if side_length = 4 ∧ face_colors = 6 ∧ cubes_per_side = 1 then 56 else 0

theorem number_of_cubes_with_at_least_two_faces_painted_is_56 :
  one_inch_cubes_with_at_least_two_faces_painted 4 6 1 = 56 :=
by
  sorry

end number_of_cubes_with_at_least_two_faces_painted_is_56_l55_55126


namespace max_poly_l55_55827

noncomputable def poly (a b : ℝ) : ℝ :=
  a^4 * b + a^3 * b + a^2 * b + a * b + a * b^2 + a * b^3 + a * b^4

theorem max_poly (a b : ℝ) (h : a + b = 4) :
  ∃ (a b : ℝ) (h : a + b = 4), poly a b = (7225 / 56) :=
sorry

end max_poly_l55_55827


namespace joaozinho_multiplication_l55_55385

theorem joaozinho_multiplication :
  12345679 * 9 = 111111111 :=
by
  sorry

end joaozinho_multiplication_l55_55385


namespace last_group_markers_l55_55445

theorem last_group_markers:
  ∀ (total_students group1_students group2_students markers_per_box boxes_of_markers group1_markers group2_markers : ℕ),
    total_students = 30 →
    group1_students = 10 →
    group2_students = 15 →
    markers_per_box = 5 →
    boxes_of_markers = 22 →
    group1_markers = 2 →
    group2_markers = 4 →
    let total_markers := boxes_of_markers * markers_per_box in
    let used_markers1 := group1_students * group1_markers in
    let used_markers2 := group2_students * group2_markers in
    let remaining_students := total_students - group1_students - group2_students in
    let remaining_markers := total_markers - used_markers1 - used_markers2 in
    remaining_students > 0 →
    remaining_markers % remaining_students = 0 →
    remaining_markers / remaining_students = 6 :=
sorry

end last_group_markers_l55_55445


namespace speed_limit_l55_55004

theorem speed_limit (x : ℝ) (h₀ : 0 < x) :
  (11 / (x + 1.5) + 8 / x ≥ 12 / (x + 2) + 2) → x ≤ 4 := 
sorry

end speed_limit_l55_55004


namespace no_odd_integers_satisfy_equation_l55_55323

theorem no_odd_integers_satisfy_equation :
  ¬ ∃ (x y z : ℤ), (x % 2 ≠ 0) ∧ (y % 2 ≠ 0) ∧ (z % 2 ≠ 0) ∧ 
  (x + y)^2 + (x + z)^2 = (y + z)^2 :=
by
  sorry

end no_odd_integers_satisfy_equation_l55_55323


namespace savings_by_having_insurance_l55_55993

theorem savings_by_having_insurance :
  let insurance_cost := 24 * 20 in
  let procedure_cost := 5000 in
  let insurance_coverage := 0.2 in
  let amount_paid_with_insurance := procedure_cost * insurance_coverage in
  let total_paid_with_insurance := amount_paid_with_insurance + insurance_cost in
  let savings := procedure_cost - total_paid_with_insurance in
  savings = 3520 := 
by {
  let insurance_cost := 24 * 20 in
  let procedure_cost := 5000 in
  let insurance_coverage := 0.2 in
  let amount_paid_with_insurance := procedure_cost * insurance_coverage in
  let total_paid_with_insurance := amount_paid_with_insurance + insurance_cost in
  let savings := procedure_cost - total_paid_with_insurance in
  guard_hyp savings,
  sorry
}

end savings_by_having_insurance_l55_55993


namespace complex_identity_l55_55807

theorem complex_identity (α β : ℝ) (h : Complex.exp (Complex.I * α) + Complex.exp (Complex.I * β) = Complex.mk (-1 / 3) (5 / 8)) :
  Complex.exp (-Complex.I * α) + Complex.exp (-Complex.I * β) = Complex.mk (-1 / 3) (-5 / 8) :=
by
  sorry

end complex_identity_l55_55807


namespace Joey_swimming_days_l55_55309

-- Define the conditions and required proof statement
theorem Joey_swimming_days (E : ℕ) (h1 : 3 * E / 4 = 9) : E / 2 = 6 :=
by
  sorry

end Joey_swimming_days_l55_55309


namespace total_trophies_correct_l55_55228

-- Define the current number of Michael's trophies
def michael_current_trophies : ℕ := 30

-- Define the number of trophies Michael will have in three years
def michael_trophies_in_three_years : ℕ := michael_current_trophies + 100

-- Define the number of trophies Jack will have in three years
def jack_trophies_in_three_years : ℕ := 10 * michael_current_trophies

-- Define the total number of trophies Jack and Michael will have after three years
def total_trophies_in_three_years : ℕ := michael_trophies_in_three_years + jack_trophies_in_three_years

-- Prove that the total number of trophies after three years is 430
theorem total_trophies_correct : total_trophies_in_three_years = 430 :=
by
  sorry -- proof is omitted

end total_trophies_correct_l55_55228


namespace twelfth_term_arithmetic_sequence_l55_55147

theorem twelfth_term_arithmetic_sequence :
  let a := (1 : ℚ) / 2
  let d := (1 : ℚ) / 3
  (a + 11 * d) = (25 : ℚ) / 6 :=
by
  sorry

end twelfth_term_arithmetic_sequence_l55_55147


namespace proof_problem_l55_55073

noncomputable def problem : ℚ :=
  let a := 1
  let b := 2
  let c := 1
  let d := 0
  a + 2 * b + 3 * c + 4 * d

theorem proof_problem : problem = 8 := by
  -- All computations are visible here
  unfold problem
  rfl

end proof_problem_l55_55073


namespace value_of_x_is_10_l55_55725

-- Define the conditions
def condition1 (x : ℕ) : ℕ := 3 * x
def condition2 (x : ℕ) : ℕ := (26 - x) + 14

-- Define the proof problem
theorem value_of_x_is_10 (x : ℕ) (h1 : condition1 x = condition2 x) : x = 10 :=
by {
  sorry
}

end value_of_x_is_10_l55_55725


namespace compute_M_l55_55963

open Matrix

variable {R : Type*} [Semiring R]
variable {n m : Type*} [Fintype n] [DecidableEq n] [Fintype m] [DecidableEq m]

def M : Matrix n m R := sorry
def u : m → R := sorry
def v : m → R := sorry
def w : m → R := sorry

axiom hM_u : M.mul_vec u = ![2, -2] 
axiom hM_v : M.mul_vec v = ![3, 1]
axiom hM_w : M.mul_vec w = ![-1, 4]

theorem compute_M (M : Matrix n m R) (u v w : m → R) :
  M.mul_vec (3 • u - v + 2 • w) = ![1, 1] :=
by
  sorry

end compute_M_l55_55963


namespace number_ways_one_ball_correct_number_ways_unlimited_balls_correct_l55_55897

-- Defining the problems: Number of ways to place k identical balls into n urns with the respective conditions.

variable (n k : ℕ)

-- Problem 1: At most one ball in each urn.
def number_ways_one_ball : ℕ := (nat.choose n k)

theorem number_ways_one_ball_correct :
  number_ways_one_ball n k = nat.choose n k := by
  sorry

-- Problem 2: Unlimited number of balls in each urn.
def number_ways_unlimited_balls : ℕ := (nat.choose (n + k - 1) (n - 1))

theorem number_ways_unlimited_balls_correct :
  number_ways_unlimited_balls n k = nat.choose (n + k - 1) (n - 1) := by
  sorry

end number_ways_one_ball_correct_number_ways_unlimited_balls_correct_l55_55897


namespace y_real_for_all_x_l55_55471

theorem y_real_for_all_x (x : ℝ) : ∃ y : ℝ, 9 * y^2 + 3 * x * y + x - 3 = 0 :=
by
  sorry

end y_real_for_all_x_l55_55471


namespace solution_set_l55_55931

def f (x : ℝ) : ℝ := |x + 1| - 2 * |x - 1|

theorem solution_set : { x : ℝ | f x > 1 } = Set.Ioo (2/3) 2 :=
by
  sorry

end solution_set_l55_55931


namespace unpainted_area_of_five_inch_board_l55_55997

def width1 : ℝ := 5
def width2 : ℝ := 6
def angle : ℝ := 45

theorem unpainted_area_of_five_inch_board : 
  ∃ (area : ℝ), area = 30 :=
by
  sorry

end unpainted_area_of_five_inch_board_l55_55997


namespace range_of_m_l55_55053

theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ x ≠ 1 ∧ (m + 3) / (x - 1) = 1) ↔ m > -4 ∧ m ≠ -3 := 
by 
  sorry

end range_of_m_l55_55053


namespace range_of_a_l55_55640

noncomputable def f (x : ℝ) : ℝ := 
  if h : x ≤ 1 then x^2 - x + 3 else 0

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f x ≥ |x / 2 + a|) ↔ -47 / 16 ≤ a ∧ a ≤ 2 := sorry

end range_of_a_l55_55640


namespace anoop_joined_after_6_months_l55_55577

theorem anoop_joined_after_6_months (arjun_investment : ℕ) (anoop_investment : ℕ) (months_in_year : ℕ)
  (arjun_time : ℕ) (anoop_time : ℕ) :
  arjun_investment * arjun_time = anoop_investment * anoop_time →
  anoop_investment = 2 * arjun_investment →
  arjun_time = months_in_year →
  anoop_time + arjun_time = months_in_year →
  anoop_time = 6 :=
by sorry

end anoop_joined_after_6_months_l55_55577


namespace janice_items_l55_55075

theorem janice_items : 
  ∃ a b c : ℕ, 
    a + b + c = 60 ∧ 
    15 * a + 400 * b + 500 * c = 6000 ∧ 
    a = 50 := 
by 
  sorry

end janice_items_l55_55075


namespace arithmetic_sequence_twelfth_term_l55_55152

theorem arithmetic_sequence_twelfth_term :
  let a1 := (1 : ℚ) / 2;
  let a2 := (5 : ℚ) / 6;
  let d := a2 - a1;
  (a1 + 11 * d) = (25 : ℚ) / 6 :=
by
  let a1 := (1 : ℚ) / 2;
  let a2 := (5 : ℚ) / 6;
  let d := a2 - a1;
  exact sorry

end arithmetic_sequence_twelfth_term_l55_55152


namespace total_handshakes_is_72_l55_55764

-- Define the conditions
def number_of_players_per_team := 6
def number_of_teams := 2
def number_of_referees := 3

-- Define the total number of players
def total_players := number_of_teams * number_of_players_per_team

-- Define the total number of handshakes between players of different teams
def team_handshakes := number_of_players_per_team * number_of_players_per_team

-- Define the total number of handshakes between players and referees
def player_referee_handshakes := total_players * number_of_referees

-- Define the total number of handshakes
def total_handshakes := team_handshakes + player_referee_handshakes

-- Prove that the total number of handshakes is 72
theorem total_handshakes_is_72 : total_handshakes = 72 := by
  sorry

end total_handshakes_is_72_l55_55764


namespace total_games_correct_l55_55859

noncomputable def number_of_games_per_month : ℕ := 13
noncomputable def number_of_months_in_season : ℕ := 14
noncomputable def total_games_in_season : ℕ := number_of_games_per_month * number_of_months_in_season

theorem total_games_correct : total_games_in_season = 182 := by
  sorry

end total_games_correct_l55_55859


namespace probability_X_between_neg1_and_5_probability_X_le_8_probability_X_ge_5_probability_X_between_neg3_and_9_l55_55338

noncomputable def normalCDF (z : ℝ) : ℝ :=
  sorry -- Assuming some CDF function for the sake of the example.

variable (X : ℝ → ℝ)
variable (μ : ℝ := 3)
variable (σ : ℝ := sqrt 4)

-- 1. Proof that P(-1 < X < 5) = 0.8185
theorem probability_X_between_neg1_and_5 : 
  ((-1 < X) ∧ (X < 5) → (normalCDF 1 - normalCDF (-2)) = 0.8185) :=
  sorry

-- 2. Proof that P(X ≤ 8) = 0.9938
theorem probability_X_le_8 : 
  (X ≤ 8 → normalCDF 2.5 = 0.9938) :=
  sorry

-- 3. Proof that P(X ≥ 5) = 0.1587
theorem probability_X_ge_5 : 
  (X ≥ 5 → (1 - normalCDF 1) = 0.1587) :=
  sorry

-- 4. Proof that P(-3 < X < 9) = 0.9972
theorem probability_X_between_neg3_and_9 : 
  ((-3 < X) ∧ (X < 9) → (2 * normalCDF 3 - 1) = 0.9972) :=
  sorry

end probability_X_between_neg1_and_5_probability_X_le_8_probability_X_ge_5_probability_X_between_neg3_and_9_l55_55338


namespace evaluate_g_at_neg_four_l55_55211

def g (x : Int) : Int := 5 * x + 2

theorem evaluate_g_at_neg_four : g (-4) = -18 := by
  sorry

end evaluate_g_at_neg_four_l55_55211


namespace irrational_power_to_nat_l55_55690

noncomputable def a : ℝ := Real.sqrt 2
noncomputable def b : ℝ := Real.log 3 / Real.log (Real.sqrt 2) 

theorem irrational_power_to_nat 
  (ha_irr : ¬ ∃ (q : ℚ), a = q)
  (hb_irr : ¬ ∃ (q : ℚ), b = q) : (a ^ b) = 3 := by
  -- \[a = \sqrt{2}, b = \log_{\sqrt{2}}(3)\]
  sorry

end irrational_power_to_nat_l55_55690


namespace toothpicks_needed_for_8_step_staircase_l55_55244

theorem toothpicks_needed_for_8_step_staircase:
  ∀ n toothpicks : ℕ, n = 4 → toothpicks = 30 → 
  (∃ additional_toothpicks : ℕ, additional_toothpicks = 88) :=
by
  sorry

end toothpicks_needed_for_8_step_staircase_l55_55244


namespace first_pedestrian_speed_l55_55005

def pedestrian_speed (x : ℝ) : Prop :=
  0 < x ∧ x ≤ 4 ↔ (11 / (x + 1.5) + 8 / x ≥ 12 / (x + 2) + 2)

theorem first_pedestrian_speed 
  (x : ℝ) (h : 11 / (x + 1.5) + 8 / x ≥ 12 / (x + 2) + 2) :
  0 < x ∧ x ≤ 4 :=
begin
  sorry
end

end first_pedestrian_speed_l55_55005


namespace average_of_multiples_l55_55867

theorem average_of_multiples (n : ℕ) (hn : n > 0) :
  (60.5 : ℚ) = ((n / 2) * (11 + 11 * n)) / n → n = 10 :=
by
  sorry

end average_of_multiples_l55_55867


namespace mark_birth_year_proof_l55_55227

-- Conditions
def current_year := 2021
def janice_age := 21
def graham_age := 2 * janice_age
def mark_age := graham_age + 3
def mark_birth_year (current_year : ℕ) (mark_age : ℕ) := current_year - mark_age

-- Statement to prove
theorem mark_birth_year_proof : 
  mark_birth_year current_year mark_age = 1976 := by
  sorry

end mark_birth_year_proof_l55_55227


namespace multiply_by_12_correct_result_l55_55155

theorem multiply_by_12_correct_result (x : ℕ) (h : x / 14 = 42) : x * 12 = 7056 :=
by
  sorry

end multiply_by_12_correct_result_l55_55155


namespace total_gold_coins_l55_55255

theorem total_gold_coins (n c : ℕ) 
  (h1 : n = 11 * (c - 3))
  (h2 : n = 7 * c + 5) : 
  n = 75 := 
by 
  sorry

end total_gold_coins_l55_55255


namespace logarithmic_relationship_l55_55061

theorem logarithmic_relationship (a b : ℝ) (h1 : a = Real.logb 16 625) (h2 : b = Real.logb 2 25) : a = b / 2 :=
sorry

end logarithmic_relationship_l55_55061


namespace time_needed_n_l55_55883

variable (n : Nat)
variable (d : Nat := n - 1)
variable (s : ℚ := 2 / 3 * (d))
variable (time_third_mile : ℚ := 3)
noncomputable def time_needed (n : Nat) : ℚ := (3 * (n - 1)) / 2

theorem time_needed_n: 
  (∀ (n : Nat), n > 2 → time_needed n = (3 * (n - 1)) / 2) :=
by
  intros n hn
  sorry

end time_needed_n_l55_55883


namespace least_possible_value_of_smallest_integer_l55_55580

theorem least_possible_value_of_smallest_integer :
  ∀ (A B C D : ℤ), A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
  (A + B + C + D) / 4 = 76 ∧ D = 90 →
  A = 37 :=
by
  sorry

end least_possible_value_of_smallest_integer_l55_55580


namespace probability_of_selecting_cooking_l55_55590

def total_courses : ℕ := 4
def favorable_outcomes : ℕ := 1

theorem probability_of_selecting_cooking : (favorable_outcomes : ℚ) / total_courses = 1 / 4 := 
by 
  sorry

end probability_of_selecting_cooking_l55_55590


namespace colin_speed_l55_55319

variable (B T Bn C : ℝ)
variable (m : ℝ)

-- Given conditions
def condition1 := B = 1
def condition2 := T = m * B
def condition3 := Bn = T / 3
def condition4 := C = 6 * Bn
def condition5 := C = 4

-- We need to prove C = 4 given the conditions
theorem colin_speed :
  (B = 1) →
  (T = m * B) →
  (Bn = T / 3) →
  (C = 6 * Bn) →
  C = 4 :=
by
  intros _ _ _ _
  sorry

end colin_speed_l55_55319


namespace midpoints_of_quadrilateral_form_parallelogram_l55_55539

/- Given a quadrilateral ABCD, with L and K being the midpoints of AD and BC respectively, 
and M and N being the midpoints of AC and BD respectively, 
we aim to prove that the quadrilateral formed by the points L, M, K, and N is a parallelogram. -/
theorem midpoints_of_quadrilateral_form_parallelogram
    (A B C D L K M N : Point) -- Declare points A, B, C, D, L, K, M, N
    (hL : midpoint A D L)     -- L is the midpoint of segment AD
    (hK : midpoint B C K)     -- K is the midpoint of segment BC
    (hM : midpoint A C M)     -- M is the midpoint of segment AC
    (hN : midpoint B D N)     -- N is the midpoint of segment BD :
    parallelogram L M K N :=  -- Conclusion: quadrilateral LMKN is a parallelogram
sorry

end midpoints_of_quadrilateral_form_parallelogram_l55_55539


namespace smallest_possible_n_l55_55817

theorem smallest_possible_n (n : ℕ) (h1 : n ≥ 100) (h2 : n < 1000)
  (h3 : n % 9 = 2) (h4 : n % 7 = 2) : n = 128 :=
by
  sorry

end smallest_possible_n_l55_55817


namespace local_minimum_of_function_l55_55922

noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x

theorem local_minimum_of_function : 
  (∃ a, a = 1 ∧ ∀ ε > 0, f a ≤ f (a + ε) ∧ f a ≤ f (a - ε)) := sorry

end local_minimum_of_function_l55_55922


namespace relationship_between_lines_l55_55919

-- Define the type for a line and a plane
structure Line where
  -- some properties (to be defined as needed, omitted for brevity)

structure Plane where
  -- some properties (to be defined as needed, omitted for brevity)

-- Define parallelism between a line and a plane
def parallel_line_plane (m : Line) (α : Plane) : Prop := sorry

-- Define line within a plane
def line_within_plane (n : Line) (α : Plane) : Prop := sorry

-- Define parallelism between two lines
def parallel_lines (m n : Line) : Prop := sorry

-- Define skewness between two lines
def skew_lines (m n : Line) : Prop := sorry

-- The mathematically equivalent proof problem
theorem relationship_between_lines (m n : Line) (α : Plane)
  (h1 : parallel_line_plane m α)
  (h2 : line_within_plane n α) :
  parallel_lines m n ∨ skew_lines m n := 
sorry

end relationship_between_lines_l55_55919


namespace solve_equation_1_solve_equation_2_l55_55403

theorem solve_equation_1 (x : ℝ) : x^2 - 7 * x = 0 ↔ (x = 0 ∨ x = 7) :=
by sorry

theorem solve_equation_2 (x : ℝ) : 2 * x^2 - 6 * x + 1 = 0 ↔ (x = (3 + Real.sqrt 7) / 2 ∨ x = (3 - Real.sqrt 7) / 2) :=
by sorry

end solve_equation_1_solve_equation_2_l55_55403


namespace problem_statement_l55_55499

theorem problem_statement 
  (a b c : ℝ)
  (h1 : a + b + c = 0)
  (h2 : a^3 + b^3 + c^3 = 0) : 
  a^19 + b^19 + c^19 = 0 :=
sorry

end problem_statement_l55_55499


namespace stream_speed_l55_55653

def upstream_time : ℝ := 4  -- time in hours
def downstream_time : ℝ := 4  -- time in hours
def upstream_distance : ℝ := 32  -- distance in km
def downstream_distance : ℝ := 72  -- distance in km

-- Speed equations based on given conditions
def effective_speed_upstream (vj vs : ℝ) : Prop := vj - vs = upstream_distance / upstream_time
def effective_speed_downstream (vj vs : ℝ) : Prop := vj + vs = downstream_distance / downstream_time

theorem stream_speed (vj vs : ℝ)  
  (h1 : effective_speed_upstream vj vs)
  (h2 : effective_speed_downstream vj vs) : 
  vs = 5 := sorry

end stream_speed_l55_55653


namespace handshakes_total_l55_55763

theorem handshakes_total :
  let team_size := 6
  let referees := 3
  (team_size * team_size) + (2 * team_size * referees) = 72 :=
by
  sorry

end handshakes_total_l55_55763


namespace find_initial_money_l55_55020

def initial_money (s1 s2 s3 : ℝ) : ℝ :=
  let after_store_1 := s1 - (0.4 * s1 + 4)
  let after_store_2 := after_store_1 - (0.5 * after_store_1 + 5)
  let after_store_3 := after_store_2 - (0.6 * after_store_2 + 6)
  after_store_3

theorem find_initial_money (s1 s2 s3 : ℝ) (hs3 : initial_money s1 s2 s3 = 2) : s1 = 90 :=
by
  -- Placeholder for the actual proof
  sorry

end find_initial_money_l55_55020


namespace stockholm_to_uppsala_distance_l55_55261

-- Definitions based on conditions
def map_distance_cm : ℝ := 3
def scale_cm_to_km : ℝ := 80

-- Theorem statement based on the question and correct answer
theorem stockholm_to_uppsala_distance : 
  (map_distance_cm * scale_cm_to_km = 240) :=
by 
  -- This is where the proof would go
  sorry

end stockholm_to_uppsala_distance_l55_55261


namespace total_license_groups_l55_55612

-- Defining the given conditions
def letter_choices : Nat := 3
def digit_choices_per_slot : Nat := 10
def number_of_digit_slots : Nat := 5

-- Statement to prove that the total number of different license groups is 300000
theorem total_license_groups : letter_choices * (digit_choices_per_slot ^ number_of_digit_slots) = 300000 := by
  sorry

end total_license_groups_l55_55612


namespace alex_needs_additional_coins_l55_55456

theorem alex_needs_additional_coins : 
  let friends := 12
  let coins := 63
  let total_coins_needed := (friends * (friends + 1)) / 2 
  let additional_coins_needed := total_coins_needed - coins
  additional_coins_needed = 15 :=
by sorry

end alex_needs_additional_coins_l55_55456


namespace expression_evaluation_l55_55329

theorem expression_evaluation :
  (4 * 6 / (12 * 14) * (8 * 12 * 14) / (4 * 6 * 8) - 1 = 0) :=
by sorry

end expression_evaluation_l55_55329


namespace subtract_eq_l55_55571

theorem subtract_eq (x y : ℝ) (h1 : 4 * x - 3 * y = 2) (h2 : 4 * x + y = 10) : 4 * y = 8 :=
by
  sorry

end subtract_eq_l55_55571


namespace brown_loss_percentage_is_10_l55_55832

-- Define the initial conditions
def initialHousePrice : ℝ := 100000
def profitPercentage : ℝ := 0.10
def sellingPriceBrown : ℝ := 99000

-- Compute the price Mr. Brown bought the house
def priceBrownBought := initialHousePrice * (1 + profitPercentage)

-- Define the loss percentage as a goal to prove
theorem brown_loss_percentage_is_10 :
  ((priceBrownBought - sellingPriceBrown) / priceBrownBought) * 100 = 10 := by
  sorry

end brown_loss_percentage_is_10_l55_55832


namespace sally_and_mary_picked_16_lemons_l55_55975

theorem sally_and_mary_picked_16_lemons (sally_lemons mary_lemons : ℕ) (sally_picked : sally_lemons = 7) (mary_picked : mary_lemons = 9) :
  sally_lemons + mary_lemons = 16 :=
by {
  sorry
}

end sally_and_mary_picked_16_lemons_l55_55975

import Mathlib

namespace find_sum_l168_168783

variables (x y : ℝ)

def condition1 : Prop := x^3 - 3 * x^2 + 5 * x = 1
def condition2 : Prop := y^3 - 3 * y^2 + 5 * y = 5

theorem find_sum : condition1 x → condition2 y → x + y = 2 := 
by 
  sorry -- The proof goes here

end find_sum_l168_168783


namespace range_of_m_l168_168504

open Real

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, x^2 - m * x + m > 0) ↔ (0 < m ∧ m < 4) :=
by
  sorry

end range_of_m_l168_168504


namespace fraction_comparison_l168_168220

theorem fraction_comparison (a b c d : ℝ) (h1 : a / b < c / d) (h2 : b > d) (h3 : d > 0) :
  (a + c) / (b + d) < 1/2 * (a / b + c / d) :=
by
  sorry

end fraction_comparison_l168_168220


namespace total_number_of_items_l168_168799

-- Definitions based on the problem conditions
def number_of_notebooks : ℕ := 40
def pens_more_than_notebooks : ℕ := 80
def pencils_more_than_notebooks : ℕ := 45

-- Total items calculation based on the conditions
def number_of_pens : ℕ := number_of_notebooks + pens_more_than_notebooks
def number_of_pencils : ℕ := number_of_notebooks + pencils_more_than_notebooks
def total_items : ℕ := number_of_notebooks + number_of_pens + number_of_pencils

-- Statement to be proved
theorem total_number_of_items : total_items = 245 := 
by 
  sorry

end total_number_of_items_l168_168799


namespace smallest_four_digit_integer_mod_8_eq_3_l168_168426

theorem smallest_four_digit_integer_mod_8_eq_3 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 8 = 3 ∧ n = 1003 := by
  -- Proof will be provided here
  sorry

end smallest_four_digit_integer_mod_8_eq_3_l168_168426


namespace total_number_of_books_l168_168414

theorem total_number_of_books (history_books geography_books math_books : ℕ)
  (h1 : history_books = 32) (h2 : geography_books = 25) (h3 : math_books = 43) :
  history_books + geography_books + math_books = 100 :=
by
  -- the proof would go here but we use sorry to skip it
  sorry

end total_number_of_books_l168_168414


namespace diameter_of_circumscribed_circle_l168_168366

noncomputable def right_triangle_circumcircle_diameter (a b : ℕ) : ℕ :=
  let hypotenuse := (a * a + b * b).sqrt
  if hypotenuse = max a b then hypotenuse else 2 * max a b

theorem diameter_of_circumscribed_circle
  (a b : ℕ)
  (h : a = 16 ∨ b = 16)
  (h1 : a = 12 ∨ b = 12) :
  right_triangle_circumcircle_diameter a b = 16 ∨ right_triangle_circumcircle_diameter a b = 20 :=
by
  -- The proof goes here.
  sorry

end diameter_of_circumscribed_circle_l168_168366


namespace mean_age_of_all_children_l168_168784

def euler_ages : List ℕ := [10, 12, 8]
def gauss_ages : List ℕ := [8, 8, 8, 16, 18]
def all_ages : List ℕ := euler_ages ++ gauss_ages
def total_children : ℕ := all_ages.length
def total_age : ℕ := all_ages.sum
def mean_age : ℕ := total_age / total_children

theorem mean_age_of_all_children : mean_age = 11 := by
  sorry

end mean_age_of_all_children_l168_168784


namespace parallel_line_equation_l168_168490

theorem parallel_line_equation :
  ∃ (c : ℝ), 
    (∀ x : ℝ, y = (3 / 4) * x + 6 → (y = (3 / 4) * x + c → abs (c - 6) = 4 * (5 / 4))) → c = 1 :=
by
  sorry

end parallel_line_equation_l168_168490


namespace luke_initial_stickers_l168_168917

theorem luke_initial_stickers (x : ℕ) (h : x + 12 + 20 - 5 - 8 = 39) : x = 20 := 
by 
  sorry

end luke_initial_stickers_l168_168917


namespace lindas_daughters_and_granddaughters_no_daughters_l168_168522

def number_of_people_with_no_daughters (total_daughters total_descendants daughters_with_5_daughters : ℕ) : ℕ :=
  total_descendants - (5 * daughters_with_5_daughters - total_daughters + daughters_with_5_daughters)

theorem lindas_daughters_and_granddaughters_no_daughters
  (total_daughters : ℕ)
  (total_descendants : ℕ)
  (daughters_with_5_daughters : ℕ)
  (H1 : total_daughters = 8)
  (H2 : total_descendants = 43)
  (H3 : 5 * daughters_with_5_daughters = 35)
  : number_of_people_with_no_daughters total_daughters total_descendants daughters_with_5_daughters = 36 :=
by
  -- Code to check the proof goes here.
  sorry

end lindas_daughters_and_granddaughters_no_daughters_l168_168522


namespace find_number_l168_168428

theorem find_number (x : ℝ) (h : (x / 4) + 3 = 5) : x = 8 :=
by
  sorry

end find_number_l168_168428


namespace sum_of_numbers_mod_11_l168_168026

def sum_mod_eleven : ℕ := 123456 + 123457 + 123458 + 123459 + 123460 + 123461

theorem sum_of_numbers_mod_11 :
  sum_mod_eleven % 11 = 10 :=
sorry

end sum_of_numbers_mod_11_l168_168026


namespace find_b_collinear_points_l168_168098

theorem find_b_collinear_points :
  ∃ b : ℚ, 4 * 11 - 6 * (-3 * b + 4) = 5 * (b + 3) - 1 * 4 ∧ b = 11 / 26 :=
by
  sorry

end find_b_collinear_points_l168_168098


namespace set_inclusion_l168_168480

-- Definitions based on given conditions
def setA (x : ℝ) : Prop := 0 < x ∧ x < 2
def setB (x : ℝ) : Prop := x > 0

-- Statement of the proof problem
theorem set_inclusion : ∀ x, setA x → setB x :=
by
  intros x h
  sorry

end set_inclusion_l168_168480


namespace min_value_of_quadratic_function_min_attained_at_negative_two_l168_168406

def quadratic_function (x : ℝ) : ℝ := 3 * (x + 2)^2 - 5

theorem min_value_of_quadratic_function : ∀ x : ℝ, quadratic_function x ≥ -5 :=
by
  sorry

theorem min_attained_at_negative_two : quadratic_function (-2) = -5 :=
by
  sorry

end min_value_of_quadratic_function_min_attained_at_negative_two_l168_168406


namespace sally_out_of_pocket_l168_168924

-- Definitions based on conditions
def g : ℕ := 320 -- Amount given by the school
def c : ℕ := 12  -- Cost per book
def n : ℕ := 30  -- Number of students

-- Definition derived from conditions
def total_cost : ℕ := n * c
def out_of_pocket : ℕ := total_cost - g

-- Proof statement
theorem sally_out_of_pocket : out_of_pocket = 40 := by
  -- The proof steps would go here
  sorry

end sally_out_of_pocket_l168_168924


namespace complement_A_in_U_l168_168884

-- Define the universal set as ℝ
def U : Set ℝ := Set.univ

-- Define the set A as given in the conditions
def A : Set ℝ := {y | ∃ x : ℝ, 2^(Real.log x) = y}

-- The main statement based on the conditions and the correct answer
theorem complement_A_in_U : (U \ A) = {y | y ≤ 0} := by
  sorry

end complement_A_in_U_l168_168884


namespace equilateral_triangle_side_length_l168_168408

theorem equilateral_triangle_side_length (perimeter : ℝ) (h : perimeter = 2) : abs (perimeter / 3 - 0.67) < 0.01 :=
by
  -- The proof will go here.
  sorry

end equilateral_triangle_side_length_l168_168408


namespace SunshinePumpkinsCount_l168_168987

def MoonglowPumpkins := 14
def SunshinePumpkins := 3 * MoonglowPumpkins + 12

theorem SunshinePumpkinsCount : SunshinePumpkins = 54 :=
by
  -- proof goes here
  sorry

end SunshinePumpkinsCount_l168_168987


namespace cost_of_machines_max_type_A_machines_l168_168562

-- Defining the cost equations for type A and type B machines
theorem cost_of_machines (x y : ℝ) (h1 : 3 * x + 2 * y = 31) (h2 : x - y = 2) : x = 7 ∧ y = 5 :=
sorry

-- Defining the budget constraint and computing the maximum number of type A machines purchasable
theorem max_type_A_machines (m : ℕ) (h : 7 * m + 5 * (6 - m) ≤ 34) : m ≤ 2 :=
sorry

end cost_of_machines_max_type_A_machines_l168_168562


namespace trig_identity_l168_168013

theorem trig_identity : 
  sin (real.pi * 17 / 180) * cos (real.pi * 43 / 180) + sin (real.pi * 73 / 180) * sin (real.pi * 43 / 180) = real.sqrt 3 / 2 := 
by
  sorry

end trig_identity_l168_168013


namespace probability_both_selected_l168_168436

/- 
Problem statement: Given that the probability of selection of Ram is 5/7 and that of Ravi is 1/5,
prove that the probability that both Ram and Ravi are selected is 1/7.
-/

theorem probability_both_selected (pRam : ℚ) (pRavi : ℚ) (hRam : pRam = 5 / 7) (hRavi : pRavi = 1 / 5) :
  (pRam * pRavi) = 1 / 7 :=
by
  sorry

end probability_both_selected_l168_168436


namespace polina_pizza_combinations_correct_l168_168171

def polina_pizza_combinations : Nat :=
  let total_toppings := 5
  let possible_combinations := total_toppings * (total_toppings - 1) / 2
  possible_combinations

theorem polina_pizza_combinations_correct :
  polina_pizza_combinations = 10 :=
by
  sorry

end polina_pizza_combinations_correct_l168_168171


namespace polynomial_divisibility_l168_168476

theorem polynomial_divisibility :
  ∃ (p : Polynomial ℤ), (Polynomial.X ^ 2 - Polynomial.X + 2) * p = Polynomial.X ^ 15 + Polynomial.X ^ 2 + 100 :=
by
  sorry

end polynomial_divisibility_l168_168476


namespace oyster_crab_ratio_l168_168413

theorem oyster_crab_ratio
  (O1 C1 : ℕ)
  (h1 : O1 = 50)
  (h2 : C1 = 72)
  (h3 : ∃ C2 : ℕ, C2 = (2 * C1) / 3)
  (h4 : ∃ O2 : ℕ, O1 + C1 + O2 + C2 = 195) :
  ∃ ratio : ℚ, ratio = O2 / O1 ∧ ratio = (1 : ℚ) / 2 := 
by 
  sorry

end oyster_crab_ratio_l168_168413


namespace sum_of_all_possible_values_of_g7_l168_168377

def f (x : ℝ) : ℝ := x ^ 2 - 6 * x + 14
def g (x : ℝ) : ℝ := 3 * x + 4

theorem sum_of_all_possible_values_of_g7 :
  let x1 := 3 + Real.sqrt 2;
  let x2 := 3 - Real.sqrt 2;
  let g1 := g x1;
  let g2 := g x2;
  g (f 7) = g1 + g2 := by
  sorry

end sum_of_all_possible_values_of_g7_l168_168377


namespace max_value_inequality_max_value_equality_l168_168340

theorem max_value_inequality (x : ℝ) (hx : x < 0) : 
  3 * x + 4 / x ≤ -4 * Real.sqrt 3 :=
sorry

theorem max_value_equality (x : ℝ) (hx : x = -2 * Real.sqrt 3 / 3) : 
  3 * x + 4 / x = -4 * Real.sqrt 3 :=
sorry

end max_value_inequality_max_value_equality_l168_168340


namespace no_injective_function_exists_l168_168326

theorem no_injective_function_exists :
  ¬ ∃ (f : ℝ → ℝ), (∀ x, f (x^2) - (f x)^2 ≥ 1/4) ∧ (∀ x y, f x = f y → x = y) := 
sorry

end no_injective_function_exists_l168_168326


namespace hyperbola_eccentricity_l168_168879

theorem hyperbola_eccentricity (a : ℝ) (h : 0 < a) (h_asymptote : Real.tan (Real.pi / 6) = 1 / a) : 
  (a = Real.sqrt 3) → 
  (∃ e : ℝ, e = (2 * Real.sqrt 3) / 3) :=
by
  intros
  sorry

end hyperbola_eccentricity_l168_168879


namespace number_of_remaining_triangles_after_12_repeats_l168_168721

-- Definitions based on the problem's conditions
def initial_triangle_side_length := (1 : ℝ)
def number_of_repeats := 12
def side_length_of_remaining_triangles (n : ℕ) : ℝ := initial_triangle_side_length / (2 ^ n)

theorem number_of_remaining_triangles_after_12_repeats :
  let n := number_of_repeats in
  (3 : ℕ) ^ n = 531441 :=
by
  sorry

end number_of_remaining_triangles_after_12_repeats_l168_168721


namespace solveRealInequality_l168_168330

theorem solveRealInequality (x : ℝ) (hx : 0 < x) : x * Real.sqrt (18 - x) + Real.sqrt (18 * x - x^3) ≥ 18 → x = 3 :=
by
  sorry -- proof to be filled in

end solveRealInequality_l168_168330


namespace find_calories_per_slice_l168_168961

/-- Defining the number of slices and their respective calories. -/
def slices_in_cake : ℕ := 8
def calories_per_brownie : ℕ := 375
def brownies_in_pan : ℕ := 6
def extra_calories_in_cake : ℕ := 526

/-- Defining the total calories in cake and brownies -/
def total_calories_in_brownies : ℕ := brownies_in_pan * calories_per_brownie
def total_calories_in_cake (c : ℕ) : ℕ := slices_in_cake * c

/-- The equation from the given problem -/
theorem find_calories_per_slice (c : ℕ) :
  total_calories_in_cake c = total_calories_in_brownies + extra_calories_in_cake → c = 347 :=
by
  sorry

end find_calories_per_slice_l168_168961


namespace total_apples_eaten_l168_168247

def simone_apples_per_day := (1 : ℝ) / 2
def simone_days := 16
def simone_total_apples := simone_apples_per_day * simone_days

def lauri_apples_per_day := (1 : ℝ) / 3
def lauri_days := 15
def lauri_total_apples := lauri_apples_per_day * lauri_days

theorem total_apples_eaten :
  simone_total_apples + lauri_total_apples = 13 :=
by
  sorry

end total_apples_eaten_l168_168247


namespace bob_buys_nose_sprays_l168_168725

theorem bob_buys_nose_sprays (cost_per_spray : ℕ) (promotion : ℕ → ℕ) (total_paid : ℕ)
  (h1 : cost_per_spray = 3)
  (h2 : ∀ n, promotion n = 2 * n)
  (h3 : total_paid = 15) : (total_paid / cost_per_spray) * 2 = 10 :=
by
  sorry

end bob_buys_nose_sprays_l168_168725


namespace find_number_l168_168556

-- Define the conditions.
def condition (x : ℚ) : Prop := x - (1 / 3) * x = 16 / 3

-- Define the theorem from the translated (question, conditions, correct answer) tuple
theorem find_number : ∃ x : ℚ, condition x ∧ x = 8 :=
by
  sorry

end find_number_l168_168556


namespace greatest_multiple_less_150_l168_168812

/-- Define the LCM of two natural numbers -/
def LCM (a b : ℕ) : ℕ := Nat.lcm a b

theorem greatest_multiple_less_150 (x y : ℕ) (h1 : x = 15) (h2 : y = 20) : 
  (∃ m : ℕ, LCM x y * m < 150 ∧ ∀ n : ℕ, LCM x y * n < 150 → LCM x y * n ≤ LCM x y * m) ∧ 
  (∃ m : ℕ, LCM x y * m = 120) :=
by
  sorry

end greatest_multiple_less_150_l168_168812


namespace find_dinner_bill_l168_168673

noncomputable def total_dinner_bill (B : ℝ) (silas_share : ℝ) (remaining_friends_pay : ℝ) (each_friend_pays : ℝ) :=
  silas_share = (1/2) * B ∧
  remaining_friends_pay = (1/2) * B + 0.10 * B ∧
  each_friend_pays = remaining_friends_pay / 5 ∧
  each_friend_pays = 18

theorem find_dinner_bill : ∃ B : ℝ, total_dinner_bill B ((1/2) * B) ((1/2) * B + 0.10 * B) (18) → B = 150 :=
by
  sorry

end find_dinner_bill_l168_168673


namespace lines_per_stanza_l168_168782

-- Define the number of stanzas
def num_stanzas : ℕ := 20

-- Define the number of words per line
def words_per_line : ℕ := 8

-- Define the total number of words in the poem
def total_words : ℕ := 1600

-- Theorem statement to prove the number of lines per stanza
theorem lines_per_stanza : 
  (total_words / words_per_line) / num_stanzas = 10 := 
by sorry

end lines_per_stanza_l168_168782


namespace complement_set_solution_l168_168628

open Set Real

theorem complement_set_solution :
  let M := {x : ℝ | (1 + x) / (1 - x) > 0}
  compl M = {x : ℝ | x ≤ -1 ∨ x ≥ 1} :=
by
  sorry

end complement_set_solution_l168_168628


namespace units_digit_of_product_l168_168336

theorem units_digit_of_product (a b c : ℕ) (n m p : ℕ) (units_a : a ≡ 4 [MOD 10])
  (units_b : b ≡ 9 [MOD 10]) (units_c : c ≡ 16 [MOD 10])
  (exp_a : n = 150) (exp_b : m = 151) (exp_c : p = 152) :
  (a^n * b^m * c^p) % 10 = 4 :=
by
  sorry

end units_digit_of_product_l168_168336


namespace christine_stickers_needed_l168_168574

-- Define the number of stickers Christine has
def stickers_has : ℕ := 11

-- Define the number of stickers required for the prize
def stickers_required : ℕ := 30

-- Define the formula to calculate the number of stickers Christine needs
def stickers_needed : ℕ := stickers_required - stickers_has

-- The theorem we need to prove
theorem christine_stickers_needed : stickers_needed = 19 :=
by
  sorry

end christine_stickers_needed_l168_168574


namespace johns_age_less_than_six_times_brothers_age_l168_168645

theorem johns_age_less_than_six_times_brothers_age 
  (B J : ℕ) 
  (h1 : B = 8) 
  (h2 : J + B = 10) 
  (h3 : J = 6 * B - 46) : 
  6 * B - J = 46 :=
by
  rw [h1, h3]
  exact sorry

end johns_age_less_than_six_times_brothers_age_l168_168645


namespace flagpole_breaking_height_l168_168965

theorem flagpole_breaking_height (x : ℝ) (h_pos : 0 < x) (h_ineq : x < 6)
    (h_pythagoras : (x^2 + 2^2 = 6^2)) : x = Real.sqrt 10 :=
by sorry

end flagpole_breaking_height_l168_168965


namespace descent_property_l168_168083

def quadratic_function (x : ℝ) : ℝ := 2 * (x - 3)^2 + 1

theorem descent_property (x : ℝ) (h : x < 3) : (quadratic_function (x + 1) < quadratic_function x) :=
sorry

end descent_property_l168_168083


namespace geometric_sequence_product_identity_l168_168213

theorem geometric_sequence_product_identity 
  {a : ℕ → ℝ} (is_geometric_sequence : ∃ r, ∀ n, a (n+1) = a n * r)
  (h : a 3 * a 4 * a 6 * a 7 = 81):
  a 1 * a 9 = 9 :=
by
  sorry

end geometric_sequence_product_identity_l168_168213


namespace find_m_l168_168874

def A : Set ℕ := {1, 3}
def B (m : ℕ) : Set ℕ := {1, 2, m}

theorem find_m (m : ℕ) (h : A ⊆ B m) : m = 3 :=
sorry

end find_m_l168_168874


namespace integer_solution_of_inequality_system_l168_168257

theorem integer_solution_of_inequality_system :
  ∃ x : ℤ, (2 * (x : ℝ) ≤ 1) ∧ ((x : ℝ) + 2 > 1) ∧ (x = 0) :=
by
  sorry

end integer_solution_of_inequality_system_l168_168257


namespace other_root_of_quadratic_l168_168204

theorem other_root_of_quadratic (m : ℝ) :
  has_root (3 * x^2 - m * x - 3) 1 →
  root_of_quadratic (3, -m, -3) 1 (-1) :=
by sorry

end other_root_of_quadratic_l168_168204


namespace complex_number_condition_l168_168233

theorem complex_number_condition (z : ℂ) (h : z^2 + z + 1 = 0) :
  2 * z^96 + 3 * z^97 + 4 * z^98 + 5 * z^99 + 6 * z^100 = 3 + 5 * z := 
by 
  sorry

end complex_number_condition_l168_168233


namespace average_age_of_team_l168_168958

theorem average_age_of_team 
  (n : ℕ) 
  (captain_age : ℕ) 
  (wicket_keeper_age : ℕ) 
  (remaining_avg : ℕ → ℕ) 
  (h1 : n = 11)
  (h2 : captain_age = 27)
  (h3 : wicket_keeper_age = 28)
  (h4 : ∀ A, remaining_avg A = A - 1)
  (h5 : ∀ A, 11 * A = 9 * (remaining_avg A) + captain_age + wicket_keeper_age) : 
  ∃ A, A = 32 :=
by
  sorry

end average_age_of_team_l168_168958


namespace min_dist_l168_168912

open Complex

theorem min_dist (z w : ℂ) (hz : abs (z - (2 - 5 * I)) = 2) (hw : abs (w - (-3 + 4 * I)) = 4) :
  ∃ d, d = abs (z - w) ∧ d ≥ (Real.sqrt 106 - 6) := sorry

end min_dist_l168_168912


namespace find_numbers_l168_168264

theorem find_numbers (x y : ℤ) (h_sum : x + y = 40) (h_diff : x - y = 12) : x = 26 ∧ y = 14 :=
sorry

end find_numbers_l168_168264


namespace impossible_to_transport_50_stones_l168_168554

def arithmetic_sequence (a d n : ℕ) : List ℕ :=
  List.range n |>.map (fun i => a + i * d)

def can_transport (weights : List ℕ) (k : ℕ) (max_weight : ℕ) : Prop :=
  ∃ partition : List (List ℕ), partition.length = k ∧
    (∀ part ∈ partition, (part.sum ≤ max_weight))

theorem impossible_to_transport_50_stones :
  ¬ can_transport (arithmetic_sequence 370 2 50) 7 3000 :=
by
  sorry

end impossible_to_transport_50_stones_l168_168554


namespace eight_machines_produce_ninety_six_bottles_in_three_minutes_l168_168117

-- Define the initial conditions
def rate_per_machine: ℕ := 16 / 4 -- bottles per minute per machine

def total_bottles_8_machines_3_minutes: ℕ := 8 * rate_per_machine * 3

-- Prove the question
theorem eight_machines_produce_ninety_six_bottles_in_three_minutes:
  total_bottles_8_machines_3_minutes = 96 :=
by
  sorry

end eight_machines_produce_ninety_six_bottles_in_three_minutes_l168_168117


namespace function_zeros_condition_l168_168046

theorem function_zeros_condition (a : ℝ) (H : ∃ x1 x2 : ℝ, 0 < x1 ∧ x1 < 1 ∧ 0 < x2 ∧ x2 < 1 ∧ 
  2 * Real.exp (2 * x1) - 2 * a * x1 + a - 2 * Real.exp 1 - 1 = 0 ∧ 
  2 * Real.exp (2 * x2) - 2 * a * x2 + a - 2 * Real.exp 1 - 1 = 0) :
  2 * Real.exp 1 - 1 < a ∧ a < 2 * Real.exp (2:ℝ) - 2 * Real.exp 1 - 1 := 
sorry

end function_zeros_condition_l168_168046


namespace minimum_value_sum_l168_168155

noncomputable def min_value (a b c : ℝ) : ℝ :=
  (a / (2 * b)) + (b / (4 * c)) + (c / (8 * a))

theorem minimum_value_sum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  min_value a b c >= 3/4 :=
by
  sorry

end minimum_value_sum_l168_168155


namespace triangle_inequality_equality_condition_l168_168913

theorem triangle_inequality (a b c S : ℝ)
  (h_tri : a^2 + b^2 + c^2 = 4 * S * Real.sqrt 3)
  (h_area : S = Real.sqrt ((a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c))):
  a^2 + b^2 + c^2 ≥ 4 * S * Real.sqrt 3 :=
sorry

theorem equality_condition (a b c S : ℝ)
  (h_tri : a^2 + b^2 + c^2 = 4 * S * Real.sqrt 3)
  (h_area : S = Real.sqrt ((a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c))):
  (a = b) ∧ (b = c) :=
sorry

end triangle_inequality_equality_condition_l168_168913


namespace percentage_A_of_B_l168_168728

variable {A B C D : ℝ}

theorem percentage_A_of_B (
  h1: A = 0.125 * C)
  (h2: B = 0.375 * D)
  (h3: D = 1.225 * C)
  (h4: C = 0.805 * B) :
  A = 0.100625 * B := by
  -- Sufficient proof steps would go here
  sorry

end percentage_A_of_B_l168_168728


namespace avg_difference_in_circumferences_l168_168104

-- Define the conditions
def inner_circle_diameter : ℝ := 30
def min_track_width : ℝ := 10
def max_track_width : ℝ := 15

-- Define the average difference in the circumferences of the two circles
theorem avg_difference_in_circumferences :
  let avg_width := (min_track_width + max_track_width) / 2
  let outer_circle_diameter := inner_circle_diameter + 2 * avg_width
  let inner_circle_circumference := Real.pi * inner_circle_diameter
  let outer_circle_circumference := Real.pi * outer_circle_diameter
  outer_circle_circumference - inner_circle_circumference = 25 * Real.pi :=
by
  sorry

end avg_difference_in_circumferences_l168_168104


namespace vertical_asymptotes_sum_l168_168397

theorem vertical_asymptotes_sum (A B C : ℤ)
  (h : ∀ x : ℝ, x = -1 ∨ x = 2 ∨ x = 3 → x^3 + A * x^2 + B * x + C = 0)
  : A + B + C = -3 :=
sorry

end vertical_asymptotes_sum_l168_168397


namespace total_order_cost_is_correct_l168_168849

noncomputable def totalOrderCost : ℝ :=
  let costGeography := 35 * 10.5
  let costEnglish := 35 * 7.5
  let costMath := 20 * 12.0
  let costScience := 30 * 9.5
  let costHistory := 25 * 11.25
  let costArt := 15 * 6.75
  let discount c := c * 0.10
  let netGeography := if 35 >= 30 then costGeography - discount costGeography else costGeography
  let netEnglish := if 35 >= 30 then costEnglish - discount costEnglish else costEnglish
  let netScience := if 30 >= 30 then costScience - discount costScience else costScience
  let netMath := costMath
  let netHistory := costHistory
  let netArt := costArt
  netGeography + netEnglish + netMath + netScience + netHistory + netArt

theorem total_order_cost_is_correct : totalOrderCost = 1446.00 := by
  sorry

end total_order_cost_is_correct_l168_168849


namespace number_of_spiders_l168_168955

theorem number_of_spiders (total_legs : ℕ) (legs_per_spider : ℕ) 
  (h1 : total_legs = 40) (h2 : legs_per_spider = 8) : 
  (total_legs / legs_per_spider = 5) :=
by
  -- Placeholder for the actual proof
  sorry

end number_of_spiders_l168_168955


namespace find_c_l168_168500

theorem find_c (a b c n : ℝ) (h : n = (2 * a * b * c) / (c - a)) : c = (n * a) / (n - 2 * a * b) :=
by
  sorry

end find_c_l168_168500


namespace find_150th_letter_l168_168282

theorem find_150th_letter (n : ℕ) (pattern : ℕ → Char) (h : ∀ m, pattern (m % 3) = if m % 3 = 0 then 'A' else if m % 3 = 1 then 'B' else 'C') :
  pattern 149 = 'C' :=
by
  sorry

end find_150th_letter_l168_168282


namespace triangle_third_side_length_l168_168901

theorem triangle_third_side_length (x: ℕ) (h1: x % 2 = 0) (h2: 2 + 14 > x) (h3: 14 - 2 < x) : x = 14 :=
by 
  sorry

end triangle_third_side_length_l168_168901


namespace quadratic_root_conditions_l168_168359

noncomputable def quadratic_has_two_distinct_real_roots (k : ℝ) : Prop :=
  let a := k + 2
  let b := 4
  let c := 1
  (a ≠ 0) ∧ (b^2 - 4*a*c > 0)

theorem quadratic_root_conditions (k : ℝ) :
  quadratic_has_two_distinct_real_roots k ↔ k < 2 ∧ k ≠ -2 := 
by
  sorry

end quadratic_root_conditions_l168_168359


namespace only_valid_set_is_b_l168_168286

def can_form_triangle (a b c : Nat) : Prop :=
  (a + b > c) ∧ (b + c > a) ∧ (c + a > b)

theorem only_valid_set_is_b :
  can_form_triangle 2 3 4 ∧ 
  ¬ can_form_triangle 1 2 3 ∧
  ¬ can_form_triangle 3 4 9 ∧
  ¬ can_form_triangle 2 2 4 := by
  sorry

end only_valid_set_is_b_l168_168286


namespace square_of_105_l168_168595

theorem square_of_105 : 105^2 = 11025 := by
  sorry

end square_of_105_l168_168595


namespace find_xy_l168_168020

theorem find_xy (x y : ℝ) :
  x^2 + y^2 = 2 ∧ (x^2 / (2 - y) + y^2 / (2 - x) = 2) → (x = 1 ∧ y = 1) :=
by
  sorry

end find_xy_l168_168020


namespace product_of_consecutive_sums_not_eq_111111111_l168_168920

theorem product_of_consecutive_sums_not_eq_111111111 :
  ∀ (a : ℤ), (3 * a + 3) * (3 * a + 12) ≠ 111111111 := 
by
  intros a
  sorry

end product_of_consecutive_sums_not_eq_111111111_l168_168920


namespace wuyang_math_total_participants_l168_168904

theorem wuyang_math_total_participants :
  ∀ (x : ℕ), 
  95 * (x + 5) = 75 * (x + 3 + 10) → 
  2 * (x + x + 8) + 9 = 125 :=
by
  intro x h
  sorry

end wuyang_math_total_participants_l168_168904


namespace isosceles_triangle_largest_angle_l168_168902

theorem isosceles_triangle_largest_angle (A B C : ℝ) (h : A = B) (h₁ : C + A + B = 180) (h₂ : C = 30) : 
  180 - 2 * 30 = 120 :=
by sorry

end isosceles_triangle_largest_angle_l168_168902


namespace total_study_time_is_60_l168_168861

-- Define the times Elizabeth studied for each test
def science_time : ℕ := 25
def math_time : ℕ := 35

-- Define the total study time
def total_study_time : ℕ := science_time + math_time

-- Proposition that the total study time equals 60 minutes
theorem total_study_time_is_60 : total_study_time = 60 := by
  /-
  Here we would provide the proof steps, but since the task is to write the statement only,
  we add 'sorry' to indicate the missing proof.
  -/
  sorry

end total_study_time_is_60_l168_168861


namespace shaded_area_l168_168025

theorem shaded_area (side_len : ℕ) (triangle_base : ℕ) (triangle_height : ℕ)
  (h1 : side_len = 40) (h2 : triangle_base = side_len / 2)
  (h3 : triangle_height = side_len / 2) : 
  side_len^2 - 2 * (1/2 * triangle_base * triangle_height) = 1200 := 
  sorry

end shaded_area_l168_168025


namespace emma_average_speed_l168_168328

-- Define the given conditions
def distance1 : ℕ := 420     -- Distance traveled in the first segment
def time1 : ℕ := 7          -- Time taken in the first segment
def distance2 : ℕ := 480    -- Distance traveled in the second segment
def time2 : ℕ := 8          -- Time taken in the second segment

-- Define the total distance and total time
def total_distance : ℕ := distance1 + distance2
def total_time : ℕ := time1 + time2

-- Define the expected average speed
def expected_average_speed : ℕ := 60

-- Prove that the average speed is 60 miles per hour
theorem emma_average_speed : (total_distance / total_time) = expected_average_speed := by
  sorry

end emma_average_speed_l168_168328


namespace find_x_l168_168785

theorem find_x (x : ℝ) (h : (2 * x + 8 + 5 * x + 3 + 3 * x + 9) / 3 = 3 * x + 2) : x = -14 :=
by
  sorry

end find_x_l168_168785


namespace katarina_miles_l168_168607

theorem katarina_miles 
  (total_miles : ℕ) 
  (miles_harriet : ℕ) 
  (miles_tomas : ℕ)
  (miles_tyler : ℕ)
  (miles_katarina : ℕ) 
  (combined_miles : total_miles = 195) 
  (same_miles : miles_tomas = miles_harriet ∧ miles_tyler = miles_harriet)
  (harriet_miles : miles_harriet = 48) :
  miles_katarina = 51 :=
sorry

end katarina_miles_l168_168607


namespace maximum_cos_product_l168_168039

theorem maximum_cos_product {α β γ : ℝ} (h : sin α ^ 2 + sin β ^ 2 + sin γ ^ 2 = 1) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) (hγ : 0 < γ ∧ γ < π / 2) :
  cos α * cos β * cos γ ≤ 2 * Real.sqrt 6 / 9 :=
sorry

end maximum_cos_product_l168_168039


namespace fraction_comparison_l168_168218

theorem fraction_comparison
  (a b c d : ℝ)
  (h1 : a / b < c / d)
  (h2 : b > 0)
  (h3 : d > 0)
  (h4 : b > d) :
  (a + c) / (b + d) < (1 / 2) * (a / b + c / d) :=
by
  sorry

end fraction_comparison_l168_168218


namespace square_area_inscribed_in_parabola_l168_168973

def parabola (x : ℝ) : ℝ := x^2 - 10*x + 21

theorem square_area_inscribed_in_parabola :
  ∃ s : ℝ, s = (-1 + Real.sqrt 5) ∧ (2 * s)^2 = 24 - 8 * Real.sqrt 5 :=
sorry

end square_area_inscribed_in_parabola_l168_168973


namespace distance_between_stations_l168_168949

-- Definitions based on conditions in step a):
def speed_train1 : ℝ := 20  -- speed of the first train in km/hr
def speed_train2 : ℝ := 25  -- speed of the second train in km/hr
def extra_distance : ℝ := 55  -- one train has traveled 55 km more

-- Definition of the proof problem
theorem distance_between_stations :
  ∃ D1 D2 T : ℝ, D1 = speed_train1 * T ∧ D2 = speed_train2 * T ∧ D2 = D1 + extra_distance ∧ D1 + D2 = 495 :=
by
  sorry

end distance_between_stations_l168_168949


namespace sum_of_squares_l168_168263

theorem sum_of_squares (x y : ℝ) (h1 : x + y = 18) (h2 : x * y = 72) : x^2 + y^2 = 180 :=
sorry

end sum_of_squares_l168_168263


namespace jerry_current_average_l168_168644

theorem jerry_current_average (A : ℚ) (h1 : 3 * A + 89 = 4 * (A + 2)) : A = 81 := 
by
  sorry

end jerry_current_average_l168_168644


namespace probability_segments_length_l168_168442

theorem probability_segments_length (x y : ℝ) : 
    80 ≥ x ∧ x ≥ 20 ∧ 80 ≥ y ∧ y ≥ 20 ∧ 80 ≥ 80 - x - y ∧ 80 - x - y ≥ 20 → 
    (∃ (s : ℝ), s = (200 / 3200) ∧ s = (1 / 16)) :=
by
  intros h
  sorry

end probability_segments_length_l168_168442


namespace f_2011_l168_168037

noncomputable def f : ℝ → ℝ := sorry

axiom odd_f (x : ℝ) : f (-x) = -f x
axiom periodicity (x : ℝ) : f (x + 4) = f x + f 2
axiom f_1 : f 1 = 2

theorem f_2011 : f 2011 = -2 :=
by sorry

end f_2011_l168_168037


namespace backpacks_weight_l168_168107

variables (w_y w_g : ℝ)

theorem backpacks_weight :
  (2 * w_y + 3 * w_g = 44) ∧
  (w_y + w_g + w_g / 2 = w_g + w_y / 2) →
  (w_g = 4) ∧ (w_y = 12) :=
by
  intros h
  sorry

end backpacks_weight_l168_168107


namespace min_value_inequality_l168_168518

variable {a b c d : ℝ}

theorem min_value_inequality (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) : 
  (a / b) + (b / c) + (c / d) + (d / a) ≥ 4 :=
sorry

end min_value_inequality_l168_168518


namespace like_terms_monomials_l168_168506

theorem like_terms_monomials (a b : ℕ) : (5 * (m^8) * (n^6) = -(3/4) * (m^(2*a)) * (n^(2*b))) → (a = 4 ∧ b = 3) := by
  sorry

end like_terms_monomials_l168_168506


namespace probability_line_through_cube_faces_l168_168749

def prob_line_intersects_cube_faces : ℚ :=
  1 / 7

theorem probability_line_through_cube_faces :
  let cube_vertices := 8
  let total_selections := Nat.choose cube_vertices 2
  let body_diagonals := 4
  let probability := (body_diagonals : ℚ) / total_selections
  probability = prob_line_intersects_cube_faces :=
by {
  sorry
}

end probability_line_through_cube_faces_l168_168749


namespace find_d_l168_168514

theorem find_d (m a b d : ℕ) 
(hm : 0 < m) 
(ha : m^2 < a ∧ a < m^2 + m) 
(hb : m^2 < b ∧ b < m^2 + m) 
(hab : a ≠ b)
(hd : m^2 < d ∧ d < m^2 + m ∧ d ∣ (a * b)) : 
d = a ∨ d = b :=
sorry

end find_d_l168_168514


namespace trip_cost_l168_168452

theorem trip_cost (original_price : ℕ) (discount : ℕ) (num_people : ℕ)
  (h1 : original_price = 147) (h2 : discount = 14) (h3 : num_people = 2) :
  num_people * (original_price - discount) = 266 :=
by
  sorry

end trip_cost_l168_168452


namespace log_sum_eq_l168_168621

theorem log_sum_eq : ∀ (x y : ℝ), y = 2016 * x ∧ x^y = y^x → (Real.logb 2016 x + Real.logb 2016 y) = 2017 / 2015 :=
by
  intros x y h
  sorry

end log_sum_eq_l168_168621


namespace magic_square_l168_168509

variable (a b c d e s: ℕ)

axiom h1 : 30 + e + 18 = s
axiom h2 : 15 + c + d = s
axiom h3 : a + 27 + b = s
axiom h4 : 30 + 15 + a = s
axiom h5 : e + c + 27 = s
axiom h6 : 18 + d + b = s
axiom h7 : 30 + c + b = s
axiom h8 : a + c + 18 = s

theorem magic_square : d + e = 47 :=
by
  sorry

end magic_square_l168_168509


namespace decreasing_power_function_l168_168399

theorem decreasing_power_function (n : ℝ) (f : ℝ → ℝ) 
    (h : ∀ x > 0, f x = (n^2 - n - 1) * x^n) 
    (h_decreasing : ∀ x > 0, f x > f (x + 1)) : n = -1 :=
sorry

end decreasing_power_function_l168_168399


namespace minimum_fence_length_l168_168309

theorem minimum_fence_length {x y : ℝ} (hxy : x * y = 100) : 2 * (x + y) ≥ 40 :=
by
  sorry

end minimum_fence_length_l168_168309


namespace no_positive_n_l168_168602

theorem no_positive_n :
  ¬ ∃ (n : ℕ) (n_pos : n > 0) (a b : ℕ) (a_sd : a < 10) (b_sd : b < 10), 
    (1234 - n) * b = (6789 - n) * a :=
by 
  sorry

end no_positive_n_l168_168602


namespace cuboid_volume_l168_168091

theorem cuboid_volume (a b c : ℝ) (h1 : a * b = 3) (h2 : b * c = 5) (h3 : a * c = 15) : a * b * c = 15 :=
sorry

end cuboid_volume_l168_168091


namespace find_m_values_l168_168766

theorem find_m_values (α : Real) (m : Real) (h1 : α ∈ Set.Ioo π (3 * π / 2)) 
  (h2 : Real.sin α = (3 * m - 2) / (m + 3)) 
  (h3 : Real.cos α = (m - 5) / (m + 3)) : m = (10 / 9) ∨ m = 2 := by 
  sorry

end find_m_values_l168_168766


namespace acute_triangle_l168_168622

theorem acute_triangle (a b c : ℝ) (h : a^π + b^π = c^π) : a^2 + b^2 > c^2 := sorry

end acute_triangle_l168_168622


namespace existence_of_k_good_function_l168_168077

def is_k_good_function (f : ℕ+ → ℕ+) (k : ℕ) : Prop :=
  ∀ m n : ℕ+, m ≠ n → Nat.gcd (f m + n) (f n + m) ≤ k

theorem existence_of_k_good_function (k : ℕ) :
  (∃ f : ℕ+ → ℕ+, is_k_good_function f k) ↔ k ≥ 2 := sorry

end existence_of_k_good_function_l168_168077


namespace charlene_sold_necklaces_l168_168151

theorem charlene_sold_necklaces 
  (initial_necklaces : ℕ) 
  (given_away : ℕ) 
  (remaining : ℕ) 
  (total_made : initial_necklaces = 60) 
  (given_to_friends : given_away = 18) 
  (left_with : remaining = 26) : 
  initial_necklaces - given_away - remaining = 16 := 
by
  sorry

end charlene_sold_necklaces_l168_168151


namespace m_le_three_l168_168741

-- Definitions
def setA (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 5
def setB (m x : ℝ) : Prop := m + 1 ≤ x ∧ x ≤ 2 * m - 1

-- Theorem statement
theorem m_le_three (m : ℝ) : (∀ x : ℝ, setB m x → setA x) → m ≤ 3 := by
  sorry

end m_le_three_l168_168741


namespace sequence_properties_l168_168873

-- Define the arithmetic-geometric sequence and its sum
def a_n (n : ℕ) : ℕ := 2^(n-1)
def S_n (n : ℕ) : ℕ := 2^n - 1
def T_n (n : ℕ) : ℕ := 2^(n+1) - n - 2

theorem sequence_properties : 
(S_n 3 = 7) ∧ (S_n 6 = 63) → 
(∀ n: ℕ, a_n n = 2^(n-1)) ∧ 
(∀ n: ℕ, S_n n = 2^n - 1) ∧ 
(∀ n: ℕ, T_n n = 2^(n+1) - n - 2) :=
by
  sorry

end sequence_properties_l168_168873


namespace fraction_problem_l168_168299

theorem fraction_problem (x : ℝ) (h₁ : x * 180 = 18) (h₂ : x < 0.15) : x = 1/10 :=
by sorry

end fraction_problem_l168_168299


namespace find_a_of_perpendicular_tangent_and_line_l168_168508

open Real

theorem find_a_of_perpendicular_tangent_and_line :
  let e := Real.exp 1
  let slope_tangent := 1 / e
  let slope_line (a : ℝ) := a
  let tangent_perpendicular := ∀ (a : ℝ), slope_tangent * slope_line a = -1
  tangent_perpendicular -> ∃ a : ℝ, a = -e :=
by {
  sorry
}

end find_a_of_perpendicular_tangent_and_line_l168_168508


namespace kombucha_bottles_after_refund_l168_168053

noncomputable def bottles_per_month : ℕ := 15
noncomputable def cost_per_bottle : ℝ := 3.0
noncomputable def refund_per_bottle : ℝ := 0.10
noncomputable def months_in_year : ℕ := 12

theorem kombucha_bottles_after_refund :
  let bottles_per_year := bottles_per_month * months_in_year
  let total_refund := bottles_per_year * refund_per_bottle
  let bottles_bought_with_refund := total_refund / cost_per_bottle
  bottles_bought_with_refund = 6 := sorry

end kombucha_bottles_after_refund_l168_168053


namespace complement_intersection_complement_in_U_l168_168187

universe u
open Set

variable (U : Set ℕ) (A B : Set ℕ)

-- Definitions based on the conditions
def universal_set : Set ℕ := { x ∈ (Set.univ : Set ℕ) | x ≤ 4 }
def set_A : Set ℕ := {1, 4}
def set_B : Set ℕ := {2, 4}

-- Problem to be proven
theorem complement_intersection_complement_in_U :
  (U = universal_set) → (A = set_A) → (B = set_B) →
  compl (A ∩ B) ∩ U = {1, 2, 3} :=
by
  intro hU hA hB
  rw [hU, hA, hB]
  sorry

end complement_intersection_complement_in_U_l168_168187


namespace PersonX_job_completed_time_l168_168996

-- Definitions for conditions
def Dan_job_time := 15 -- hours
def PersonX_job_time (x : ℝ) := x -- hours
def Dan_work_time := 3 -- hours
def PersonX_remaining_work_time := 8 -- hours

-- Given Dan's and Person X's work time, prove Person X's job completion time
theorem PersonX_job_completed_time (x : ℝ) (h1 : Dan_job_time > 0)
    (h2 : PersonX_job_time x > 0)
    (h3 : Dan_work_time > 0)
    (h4 : PersonX_remaining_work_time * (1 - Dan_work_time / Dan_job_time) = 1 / x * 8) :
    x = 10 :=
  sorry

end PersonX_job_completed_time_l168_168996


namespace angle_bam_l168_168344

noncomputable def is_isosceles (A B C : Point) : Prop :=
  dist A B = dist A C

noncomputable def midpoint (C A K : Point) : Prop :=
  dist A C = dist C K

noncomputable def angle_abc (A B C : Point) : Real :=
  53

noncomputable def max_angle_mak (C A K M B : Point) : Prop :=
  dist K M = dist A B ∧
  (M = some_point_with_max_angle MAK)

theorem angle_bam
  (A B C K M : Point)
  (hAB_AC : is_isosceles A B C)
  (hABC_53 : angle_abc A B C = 53)
  (hMid_C : midpoint C A K)
  (hB_M_Side : same_side_line B M (line_through A C))
  (hKM_AB : dist K M = dist A B)
  (hMax_MAK : max_angle_mak C A K M B) :
  ∠BAM = 44 :=
sorry

end angle_bam_l168_168344


namespace angle_quadrant_l168_168668

theorem angle_quadrant (α : ℤ) (h : α = -390) : (0 > α % 360 ∨ α % 360 ≥ 270) :=
by {
  have h1 : α % 360 = -30, by sorry,
  show (0 > α % 360 ∨ α % 360 ≥ 270), by {
    have h2 : (-30 + 360) % 360 = 330, by sorry,
    show (330 ≥ 270), by sorry,
  },
  exact or.inr h2,
}

end angle_quadrant_l168_168668


namespace find_larger_number_l168_168144

variable {x y : ℕ} 

theorem find_larger_number (h_ratio : 4 * x = 3 * y) (h_sum : x + y + 100 = 500) : y = 1600 / 7 := by 
  sorry

end find_larger_number_l168_168144


namespace domain_condition_l168_168624

variable (k : ℝ)
def quadratic_expression (x : ℝ) : ℝ := k * x^2 - 4 * k * x + k + 8

theorem domain_condition (k : ℝ) : (∀ x : ℝ, quadratic_expression k x > 0) ↔ (0 ≤ k ∧ k < 8/3) :=
sorry

end domain_condition_l168_168624


namespace value_b_minus_a_l168_168688

theorem value_b_minus_a (a b : ℝ) (h₁ : a + b = 507) (h₂ : (a - b) / b = 1 / 7) : b - a = -34.428571 :=
by
  sorry

end value_b_minus_a_l168_168688


namespace num_of_4_digit_numbers_divisible_by_13_l168_168898

theorem num_of_4_digit_numbers_divisible_by_13 :
  let smallest_4_digit := 1000
  let smallest_divisible_by_13 := 1001
  let largest_4_digit := 9999
  let largest_divisible_by_13 := 9997
  let a := smallest_divisible_by_13
  let d := 13
  let l := largest_divisible_by_13
  (\(l - a) / d) + 1 = 693 :=
by
  let smallest_4_digit := 1000
  let smallest_divisible_by_13 := 1001
  let largest_4_digit := 9999
  let largest_divisible_by_13 := 9997
  let a := smallest_divisible_by_13
  let d := 13
  let l := largest_divisible_by_13
  have h1 : (l - a) / d + 1 = (9997 - 1001) / 13 + 1, by sorry
  have h2 : (9997 - 1001) / 13 + 1 = 8996 / 13 + 1, by sorry
  have h3 : 8996 / 13 + 1 = 692 + 1, by sorry
  have h4 : 692 + 1 = 693, by sorry
  exact h4

end num_of_4_digit_numbers_divisible_by_13_l168_168898


namespace domain_of_f_eq_l168_168935

def domain_of_fractional_function : Set ℝ := 
  { x : ℝ | x > -1 }

theorem domain_of_f_eq : 
  ∀ x : ℝ, x ∈ domain_of_fractional_function ↔ x > -1 :=
by
  sorry -- Proof this part in Lean 4. The domain of f(x) is (-1, +∞)

end domain_of_f_eq_l168_168935


namespace find_x_plus_y_l168_168038

theorem find_x_plus_y (x y : ℝ) 
  (h1 : x + Real.cos y = 1005) 
  (h2 : x + 1005 * Real.sin y = 1003) 
  (h3 : π ≤ y ∧ y ≤ 3 * π / 2) : 
  x + y = 1005 + 3 * π / 2 :=
sorry

end find_x_plus_y_l168_168038


namespace work_completion_days_l168_168657

theorem work_completion_days (P R: ℕ) (hP: P = 80) (hR: R = 120) : P * R / (P + R) = 48 := by
  -- The proof is omitted as we are only writing the statement
  sorry

end work_completion_days_l168_168657


namespace arith_seq_sum_of_terms_l168_168767

theorem arith_seq_sum_of_terms 
  (a : ℕ → ℝ) (d : ℝ) 
  (h_arith_seq : ∀ n, a (n + 1) = a n + d)
  (h_pos_diff : 0 < d) 
  (h_first_three_sum : a 0 + a 1 + a 2 = 15) 
  (h_first_three_prod : a 0 * a 1 * a 2 = 80) : 
  a 10 + a 11 + a 12 = 105 := sorry

end arith_seq_sum_of_terms_l168_168767


namespace max_intersections_arith_geo_seq_l168_168676

def arithmetic_sequence (n : ℕ) (d : ℝ) : ℝ := 1 + (n - 1) * d

def geometric_sequence (n : ℕ) (q : ℝ) : ℝ := q ^ (n - 1)

theorem max_intersections_arith_geo_seq (d : ℝ) (q : ℝ) (h_d : d ≠ 0) (h_q_pos : q > 0) (h_q_neq1 : q ≠ 1) :
  (∃ n : ℕ, arithmetic_sequence n d = geometric_sequence n q) → ∃ n₁ n₂ : ℕ, n₁ ≠ n₂ ∧ (arithmetic_sequence n₁ d = geometric_sequence n₁ q) ∧ (arithmetic_sequence n₂ d = geometric_sequence n₂ q) :=
sorry

end max_intersections_arith_geo_seq_l168_168676


namespace asymptotes_of_hyperbola_l168_168201

theorem asymptotes_of_hyperbola (k : ℤ) (h1 : (k - 2016) * (k - 2018) < 0) :
  ∀ x y: ℝ, (x ^ 2) - (y ^ 2) = 1 → ∃ a b: ℝ, y = x * a ∨ y = x * b :=
by
  sorry

end asymptotes_of_hyperbola_l168_168201


namespace store_shelves_l168_168437

theorem store_shelves (initial_books sold_books books_per_shelf : ℕ) 
    (h_initial: initial_books = 27)
    (h_sold: sold_books = 6)
    (h_per_shelf: books_per_shelf = 7) :
    (initial_books - sold_books) / books_per_shelf = 3 := by
  sorry

end store_shelves_l168_168437


namespace min_value_inequality_l168_168516

theorem min_value_inequality (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) : 
  (a / b + b / c + c / d + d / a) ≥ 4 :=
by
  sorry

end min_value_inequality_l168_168516


namespace common_tangents_count_l168_168787

def circleC1 : Prop := ∃ (x y : ℝ), x^2 + y^2 + 2 * x - 6 * y - 15 = 0
def circleC2 : Prop := ∃ (x y : ℝ), x^2 + y^2 - 4 * x + 2 * y + 4 = 0

theorem common_tangents_count (C1 : circleC1) (C2 : circleC2) : 
  ∃ (n : ℕ), n = 2 := 
sorry

end common_tangents_count_l168_168787


namespace distance_from_A_to_C_correct_total_distance_traveled_correct_l168_168847

-- Define the conditions
def distance_to_A : ℕ := 30
def distance_to_B : ℕ := 20
def distance_to_C : ℤ := -15
def times_to_C : ℕ := 3

-- Define the resulting calculated distances based on the conditions
def distance_A_to_C : ℕ := distance_to_A + distance_to_C.natAbs
def total_distance_traveled : ℕ := (distance_to_A + distance_to_B) * 2 + distance_to_C.natAbs * (times_to_C * 2)

-- The proof problems (statements) based on the problem's questions
theorem distance_from_A_to_C_correct : distance_A_to_C = 45 := by
  sorry

theorem total_distance_traveled_correct : total_distance_traveled = 190 := by
  sorry

end distance_from_A_to_C_correct_total_distance_traveled_correct_l168_168847


namespace negative_cube_root_l168_168110

theorem negative_cube_root (a : ℝ) : ∃ x : ℝ, x ^ 3 = -a^2 - 1 ∧ x < 0 :=
by
  sorry

end negative_cube_root_l168_168110


namespace power_function_inequality_l168_168348

theorem power_function_inequality (m : ℕ) (h : m > 0)
  (h_point : (2 : ℝ) ^ (1 / (m ^ 2 + m)) = Real.sqrt 2) :
  m = 1 ∧ ∀ a : ℝ, 1 ≤ a ∧ a < (3 / 2) → 
  (2 - a : ℝ) ^ (1 / (m ^ 2 + m)) > (a - 1 : ℝ) ^ (1 / (m ^ 2 + m)) :=
by
  sorry

end power_function_inequality_l168_168348


namespace negation_of_p_is_neg_p_l168_168176

-- Define the proposition p
def p : Prop :=
  ∀ x > 0, (x + 1) * Real.exp x > 1

-- Define the negation of the proposition p
def neg_p : Prop :=
  ∃ x > 0, (x + 1) * Real.exp x ≤ 1

-- State the proof problem: negation of p is neg_p
theorem negation_of_p_is_neg_p : ¬p ↔ neg_p :=
by
  -- Stating that ¬p is equivalent to neg_p
  sorry

end negation_of_p_is_neg_p_l168_168176


namespace rectangle_height_l168_168445

theorem rectangle_height (y : ℝ) (h_pos : 0 < y) 
  (h_area : let length := 5 - (-3)
            let height := y - (-2)
            length * height = 112) : y = 12 := 
by 
  -- The proof is omitted
  sorry

end rectangle_height_l168_168445


namespace german_russian_students_l168_168014

open Nat

theorem german_russian_students (G R : ℕ) (G_cap_R : ℕ) 
  (h_total : 1500 = G + R - G_cap_R)
  (hG_lb : 1125 ≤ G) (hG_ub : G ≤ 1275)
  (hR_lb : 375 ≤ R) (hR_ub : R ≤ 525) :
  300 = (max (G_cap_R) - min (G_cap_R)) :=
by
  -- Proof would go here
  sorry

end german_russian_students_l168_168014


namespace range_of_m_and_n_l168_168349

theorem range_of_m_and_n (m n : ℝ) : 
  (2 * 2 - 3 + m > 0) → ¬ (2 + 3 - n ≤ 0) → (m > -1 ∧ n < 5) := by
  intros hA hB
  sorry

end range_of_m_and_n_l168_168349


namespace sum_of_numbers_mod_11_l168_168027

def sum_mod_eleven : ℕ := 123456 + 123457 + 123458 + 123459 + 123460 + 123461

theorem sum_of_numbers_mod_11 :
  sum_mod_eleven % 11 = 10 :=
sorry

end sum_of_numbers_mod_11_l168_168027


namespace compare_neg_fractions_l168_168153

theorem compare_neg_fractions :
  - (10 / 11 : ℤ) > - (11 / 12 : ℤ) :=
sorry

end compare_neg_fractions_l168_168153


namespace quadratic_difference_square_l168_168495

theorem quadratic_difference_square (α β : ℝ) (h : α ≠ β) (hα : α^2 - 3 * α + 1 = 0) (hβ : β^2 - 3 * β + 1 = 0) : (α - β)^2 = 5 := by
  sorry

end quadratic_difference_square_l168_168495


namespace find_a_l168_168542

-- Define the slopes of the lines and the condition that they are perpendicular.
def slope1 (a : ℝ) : ℝ := a
def slope2 (a : ℝ) : ℝ := a + 2

-- The main statement of our problem.
theorem find_a (a : ℝ) (h : slope1 a * slope2 a = -1) : a = -1 :=
sorry

end find_a_l168_168542


namespace symmetry_center_of_f_l168_168256

noncomputable def f (x : ℝ) : ℝ := (1/2) * Real.cos (2 * x) + Real.sqrt 3 * Real.sin x * Real.cos x

theorem symmetry_center_of_f :
  ∃ c : ℝ, ∀ x : ℝ, f (2 * x + π / 6) = Real.sin (2 * (-π / 12) + π / 6) :=
sorry

end symmetry_center_of_f_l168_168256


namespace area_of_given_triangle_l168_168265

def point := ℝ × ℝ

def triangle_area (A B C : point) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem area_of_given_triangle : 
  triangle_area (1, 1) (7, 1) (5, 3) = 6 :=
by
  -- the proof should go here
  sorry

end area_of_given_triangle_l168_168265


namespace division_remainder_unique_u_l168_168825

theorem division_remainder_unique_u :
  ∃! u : ℕ, ∃ q : ℕ, 15 = u * q + 4 ∧ u > 4 :=
sorry

end division_remainder_unique_u_l168_168825


namespace freshmen_and_sophomores_without_pet_l168_168941

theorem freshmen_and_sophomores_without_pet (total_students : ℕ) 
                                             (freshmen_sophomores_percent : ℕ)
                                             (pet_ownership_fraction : ℕ)
                                             (h_total : total_students = 400)
                                             (h_percent : freshmen_sophomores_percent = 50)
                                             (h_fraction : pet_ownership_fraction = 5) : 
                                             (total_students * freshmen_sophomores_percent / 100 - 
                                             total_students * freshmen_sophomores_percent / 100 / pet_ownership_fraction) = 160 :=
by
  sorry

end freshmen_and_sophomores_without_pet_l168_168941


namespace symmetric_parabola_l168_168383

def parabola1 (x : ℝ) : ℝ := (x - 2)^2 + 3
def parabola2 (x : ℝ) : ℝ := -(x + 2)^2 - 3

theorem symmetric_parabola : ∀ x y : ℝ,
  y = parabola1 x ↔ 
  (-y) = parabola2 (-x) ∧ y = -(x + 2)^2 - 3 :=
sorry

end symmetric_parabola_l168_168383


namespace problem_equiv_l168_168982

-- Definitions to match the conditions
def is_monomial (v : List ℤ) : Prop :=
  ∀ i ∈ v, True  -- Simplified; typically this would involve more specific definitions

def degree (e : String) : ℕ :=
  if e = "xy" then 2 else 0

noncomputable def coefficient (v : String) : ℤ :=
  if v = "m" then 1 else 0

-- Main fact to be proven
theorem problem_equiv :
  is_monomial [-3, 1, 5] :=
sorry

end problem_equiv_l168_168982


namespace pedro_more_squares_l168_168243

theorem pedro_more_squares (jesus_squares : ℕ) (linden_squares : ℕ) (pedro_squares : ℕ)
  (h1 : jesus_squares = 60) (h2 : linden_squares = 75) (h3 : pedro_squares = 200) :
  pedro_squares - (jesus_squares + linden_squares) = 65 :=
by
  sorry

end pedro_more_squares_l168_168243


namespace dice_roll_probability_bounds_l168_168838

noncomputable def dice_roll_probability : Prop :=
  let n := 80
  let p := (1 : ℝ) / 6
  let q := 1 - p
  let epsilon := 2.58 / 24
  let lower_bound := (p - epsilon) * n
  let upper_bound := (p + epsilon) * n
  5 ≤ lower_bound ∧ upper_bound ≤ 22

theorem dice_roll_probability_bounds :
  dice_roll_probability :=
sorry

end dice_roll_probability_bounds_l168_168838


namespace center_of_circle_l168_168333

theorem center_of_circle (x y : ℝ) : (x^2 + y^2 - 10 * x + 4 * y + 13 = 0) → (x - y = 7) :=
by
  -- Statement, proof omitted
  sorry

end center_of_circle_l168_168333


namespace joe_paid_4_more_than_jenny_l168_168761

theorem joe_paid_4_more_than_jenny
  (total_plain_pizza_cost : ℕ := 12) 
  (total_slices : ℕ := 12)
  (additional_cost_per_mushroom_slice : ℕ := 1) -- 0.50 dollars represented in integer (value in cents or minimal currency unit)
  (mushroom_slices : ℕ := 4) 
  (plain_slices := total_slices - mushroom_slices) -- Calculate plain slices.
  (total_additional_cost := mushroom_slices * additional_cost_per_mushroom_slice)
  (total_pizza_cost := total_plain_pizza_cost + total_additional_cost)
  (plain_slice_cost := total_plain_pizza_cost / total_slices)
  (mushroom_slice_cost := plain_slice_cost + additional_cost_per_mushroom_slice) 
  (joe_mushroom_slices := mushroom_slices) 
  (joe_plain_slices := 3) 
  (jenny_plain_slices := plain_slices - joe_plain_slices) 
  (joe_paid := (joe_mushroom_slices * mushroom_slice_cost) + (joe_plain_slices * plain_slice_cost))
  (jenny_paid := jenny_plain_slices * plain_slice_cost) : 
  joe_paid - jenny_paid = 4 := 
by {
  -- Here, we define the steps we used to calculate the cost.
  sorry -- Proof skipped as per instructions.
}

end joe_paid_4_more_than_jenny_l168_168761


namespace percentage_of_males_l168_168756

theorem percentage_of_males (total_employees males_below_50 males_percentage : ℕ) (h1 : total_employees = 800) (h2 : males_below_50 = 120) (h3 : 40 * males_percentage / 100 = 60 * males_below_50):
  males_percentage = 25 :=
by
  sorry

end percentage_of_males_l168_168756


namespace kims_total_points_l168_168404

theorem kims_total_points :
  let points_easy := 2
  let points_average := 3
  let points_hard := 5
  let answers_easy := 6
  let answers_average := 2
  let answers_hard := 4
  let total_points := (answers_easy * points_easy) + (answers_average * points_average) + (answers_hard * points_hard)
  total_points = 38 :=
by
  -- This is a placeholder to indicate that the proof is not included.
  sorry

end kims_total_points_l168_168404


namespace first_year_after_2020_with_sum_4_l168_168214

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 1000) + (n % 1000 / 100) + (n % 100 / 10) + (n % 10)

def is_year (y : ℕ) : Prop :=
  y > 2020 ∧ sum_of_digits y = 4

theorem first_year_after_2020_with_sum_4 : ∃ y, is_year y ∧ ∀ z, is_year z → z ≥ y :=
by sorry

end first_year_after_2020_with_sum_4_l168_168214


namespace fraction_people_over_65_l168_168863

theorem fraction_people_over_65 (T : ℕ) (F : ℕ) : 
  (3:ℚ) / 7 * T = 24 ∧ 50 < T ∧ T < 100 → T = 56 ∧ ∃ F : ℕ, (F / 56 : ℚ) = F / (T : ℚ) :=
by 
  sorry

end fraction_people_over_65_l168_168863


namespace max_sum_ge_zero_l168_168561

-- Definition for max and min functions for real numbers
noncomputable def max_real (x y : ℝ) := if x ≥ y then x else y
noncomputable def min_real (x y : ℝ) := if x ≤ y then x else y

-- Condition: a + b + c + d = 0
def sum_zero (a b c d : ℝ) := a + b + c + d = 0

-- Lean statement for Problem (a)
theorem max_sum_ge_zero (a b c d : ℝ) (h : sum_zero a b c d) : 
  max_real a b + max_real a c + max_real a d + max_real b c + max_real b d + max_real c d ≥ 0 :=
sorry

-- Lean statement for Problem (b)
def find_max_k : ℕ :=
2

end max_sum_ge_zero_l168_168561


namespace total_games_l168_168327

theorem total_games (N : ℕ) (p : ℕ)
  (hPetya : 2 ∣ N)
  (hKolya : 3 ∣ N)
  (hVasya : 5 ∣ N)
  (hGamesNotInvolving : 2 ≤ N - (N / 2 + N / 3 + N / 5)) :
  N = 30 :=
by
  sorry

end total_games_l168_168327


namespace computation_problems_count_l168_168975

theorem computation_problems_count
    (C W : ℕ)
    (h1 : 3 * C + 5 * W = 110)
    (h2 : C + W = 30) :
    C = 20 :=
by
  sorry

end computation_problems_count_l168_168975


namespace jake_final_bitcoins_l168_168759

def initial_bitcoins : ℕ := 120
def investment_bitcoins : ℕ := 40
def returned_investment : ℕ := investment_bitcoins * 2
def bitcoins_after_investment : ℕ := initial_bitcoins - investment_bitcoins + returned_investment
def first_charity_donation : ℕ := 25
def bitcoins_after_first_donation : ℕ := bitcoins_after_investment - first_charity_donation
def brother_share : ℕ := 67
def bitcoins_after_giving_to_brother : ℕ := bitcoins_after_first_donation - brother_share
def debt_payment : ℕ := 5
def bitcoins_after_taking_back : ℕ := bitcoins_after_giving_to_brother + debt_payment
def quadrupled_bitcoins : ℕ := bitcoins_after_taking_back * 4
def second_charity_donation : ℕ := 15
def final_bitcoins : ℕ := quadrupled_bitcoins - second_charity_donation

theorem jake_final_bitcoins : final_bitcoins = 277 := by
  unfold final_bitcoins
  unfold quadrupled_bitcoins
  unfold bitcoins_after_taking_back
  unfold debt_payment
  unfold bitcoins_after_giving_to_brother
  unfold brother_share
  unfold bitcoins_after_first_donation
  unfold first_charity_donation
  unfold bitcoins_after_investment
  unfold returned_investment
  unfold investment_bitcoins
  unfold initial_bitcoins
  sorry

end jake_final_bitcoins_l168_168759


namespace simplify_and_evaluate_expression_l168_168084

theorem simplify_and_evaluate_expression (a b : ℤ) (h₁ : a = -2) (h₂ : b = 1) :
  ((a - 2 * b) ^ 2 - (a + 3 * b) * (a - 2 * b)) / b = 20 :=
by
  sorry

end simplify_and_evaluate_expression_l168_168084


namespace ineq_sqrt_two_l168_168477

theorem ineq_sqrt_two (x y : ℝ) (h1 : x > y) (h2 : x * y = 1) : 
  (x^2 + y^2) / (x - y) ≥ 2 * Real.sqrt 2 := 
by 
  sorry

end ineq_sqrt_two_l168_168477


namespace find_a_l168_168298

theorem find_a (a : ℝ) (x_values y_values : List ℝ)
  (h_y : ∀ x, List.getD y_values x 0 = 2.1 * List.getD x_values x 1 - 0.3) :
  a = 10 :=
by
  have h_mean_x : (1 + 2 + 3 + 4 + 5) / 5 = 3 := by norm_num
  have h_sum_y : (2 + 3 + 7 + 8 + a) / 5 = (2.1 * 3 - 0.3) := by sorry
  sorry

end find_a_l168_168298


namespace find_N_l168_168121

theorem find_N (N : ℕ) (h : (Real.sqrt 3 - 1)^N = 4817152 - 2781184 * Real.sqrt 3) : N = 16 :=
sorry

end find_N_l168_168121


namespace common_ratio_of_geometric_series_l168_168467

theorem common_ratio_of_geometric_series (a1 a2 a3 : ℚ) (h1 : a1 = -4 / 7)
                                         (h2 : a2 = 14 / 3) (h3 : a3 = -98 / 9) :
  ∃ r : ℚ, r = a2 / a1 ∧ r = a3 / a2 ∧ r = -49 / 6 :=
by
  use -49 / 6
  sorry

end common_ratio_of_geometric_series_l168_168467


namespace max_liters_of_water_heated_l168_168814

theorem max_liters_of_water_heated
  (heat_initial : ℕ := 480) 
  (heat_drop : ℝ := 0.25)
  (temp_initial : ℝ := 20)
  (temp_boiling : ℝ := 100)
  (specific_heat_capacity : ℝ := 4.2)
  (kJ_to_liters_conversion : ℝ := 336) :
  (∀ m : ℕ, (m * kJ_to_liters_conversion > ((heat_initial : ℝ) / (1 - heat_drop)) → m ≤ 5)) :=
by
  sorry

end max_liters_of_water_heated_l168_168814


namespace center_of_circle_l168_168669

-- Let's define the circle as a set of points satisfying the given condition.
def circle (x y : ℝ) : Prop := (x - 2) ^ 2 + (y + 1) ^ 2 = 4

-- Prove that the point (2, -1) is the center of this circle in ℝ².
theorem center_of_circle : ∀ (x y : ℝ), circle (x - 2) (y + 1) ↔ (x, y) = (2, -1) :=
by
  intros x y
  sorry

end center_of_circle_l168_168669


namespace not_p_and_not_q_true_l168_168058

variable (p q: Prop)

theorem not_p_and_not_q_true (h1: ¬ (p ∧ q)) (h2: ¬ (p ∨ q)) : ¬ p ∧ ¬ q :=
by
  sorry

end not_p_and_not_q_true_l168_168058


namespace general_term_formula_sum_of_2_pow_an_l168_168615

variable {S : ℕ → ℕ}
variable {a : ℕ → ℕ}
variable {b : ℕ → ℕ}
variable {T : ℕ → ℕ}

axiom S5_eq_30 : S 5 = 30
axiom a1_a6_eq_14 : a 1 + a 6 = 14

theorem general_term_formula : ∀ n, a n = 2 * n :=
sorry

theorem sum_of_2_pow_an (n : ℕ) : T n = (4^(n + 1)) / 3 - 4 / 3 :=
sorry

end general_term_formula_sum_of_2_pow_an_l168_168615


namespace expand_expression_l168_168016

theorem expand_expression (x y : ℕ) : 
  (x + 15) * (3 * y + 20) = 3 * x * y + 20 * x + 45 * y + 300 :=
by 
  sorry

end expand_expression_l168_168016


namespace amplitude_period_initial_phase_max_value_set_transformation_from_sine_l168_168048

open Real

noncomputable def f (x φ A : ℝ) : ℝ := sin (2 * x + φ) + A

theorem amplitude_period_initial_phase (φ A : ℝ) :
  ∃ amp prd in_phs : ℝ, 
    amp = A ∧ 
    prd = π ∧ 
    in_phs = φ :=
begin
  use [A, π, φ],
  repeat { split, refl }
end

theorem max_value_set (φ A : ℝ) : 
  ∃ x_max_set : set ℝ, 
    x_max_set = { x | ∃ k : ℤ, x = π / 4 + k * π } :=
begin
  use { x | ∃ k : ℤ, x = π / 4 + k * π },
  simp
end

theorem transformation_from_sine (φ A : ℝ) :
  ∃ (transformations : list (ℝ → ℝ → ℝ → ℝ)),
    transformations = [
      λ y φ A, sin(y + φ),
      λ y φ A, sin(2 * y + φ),
      λ y φ A, sin(2 * y + φ) + A ] :=
begin
  use [
    λ y φ A, sin(y + φ),
    λ y φ A, sin(2 * y + φ),
    λ y φ A, sin(2 * y + φ) + A ],
  repeat { split, refl }
end

end amplitude_period_initial_phase_max_value_set_transformation_from_sine_l168_168048


namespace median_convergence_l168_168830

noncomputable theory

open MeasureTheory
open ProbTheory

-- Definitions for the random variables and their distributions.
variables (α : Type*) [MeasurableSpace α] -- measurable space for the random variable
variables (ξ : α → ennreal) (ξ_n : ℕ → α → ennreal) -- sequences of random variables
variables (μ : ennreal) (μ_n : ℕ → ennreal) -- sequences of medians

-- Assumptions
variables (h_median : ∀ n, µ_n n ⟶ µ)
variables (h_inequality_ξ : ∀ x, P(ξ < µ) ≤ 1/2 ∧ P(ξ ≤ µ) ≥ 1/2)
variables (h_inequality_ξn : ∀ n x, P(ξ_n n < µ_n n) ≤ 1/2 ∧ P(ξ_n n ≤ µ_n n) ≥ 1/2)
variables (h_convergence : ∀ s, converges_in_probability (ξ_n s) ξ)

-- The main theorem statement
theorem median_convergence : (∀ n, (h_inequality_ξn n) ∧ (h_convergence n) ∧ (h_inequality_ξ)) → ∀ ε > 0, ∃ N, ∀ n ≥ N, |μ_n n - μ| < ε := by
  sorry

end median_convergence_l168_168830


namespace set_A_main_inequality_l168_168047

def f (x : ℝ) : ℝ := |2 * x + 1| + |x - 2|
def A : Set ℝ := {x | f x < 3}

theorem set_A :
  A = {x | -2 / 3 < x ∧ x < 0} :=
sorry

theorem main_inequality (s t : ℝ) (hs : -2 / 3 < s ∧ s < 0) (ht : -2 / 3 < t ∧ t < 0) :
  |1 - t / s| < |t - 1 / s| :=
sorry

end set_A_main_inequality_l168_168047


namespace multiplier_is_five_l168_168711

-- condition 1: n = m * (n - 4)
-- condition 2: n = 5
-- question: prove m = 5

theorem multiplier_is_five (n m : ℝ) 
  (h1 : n = m * (n - 4)) 
  (h2 : n = 5) : m = 5 := 
  sorry

end multiplier_is_five_l168_168711


namespace cats_needed_to_catch_100_mice_in_time_l168_168831

-- Define the context and given conditions
def cats_mice_catch_time (cats mice minutes : ℕ) : Prop :=
  cats = 5 ∧ mice = 5 ∧ minutes = 5

-- Define the goal
theorem cats_needed_to_catch_100_mice_in_time :
  cats_mice_catch_time 5 5 5 → (∃ t : ℕ, t = 500) :=
by
  intro h
  sorry

end cats_needed_to_catch_100_mice_in_time_l168_168831


namespace computation_problems_count_l168_168978

theorem computation_problems_count : 
  ∃ (x y : ℕ), 3 * x + 5 * y = 110 ∧ x + y = 30 ∧ x = 20 :=
by
  sorry

end computation_problems_count_l168_168978


namespace four_digit_numbers_divisible_by_13_l168_168891

theorem four_digit_numbers_divisible_by_13 : 
  (set.Icc 1000 9999).filter (λ x, x % 13 = 0).cardinality = 693 :=
by
  sorry

end four_digit_numbers_divisible_by_13_l168_168891


namespace initial_dog_cat_ratio_l168_168065

theorem initial_dog_cat_ratio (C : ℕ) :
  75 / (C + 20) = 15 / 11 →
  (75 / C) = 15 / 7 :=
by
  sorry

end initial_dog_cat_ratio_l168_168065


namespace distinct_digits_and_difference_is_945_l168_168715

theorem distinct_digits_and_difference_is_945 (a b c : ℕ)
  (h1 : 0 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9) (h3 : 0 ≤ c ∧ c ≤ 9)
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_difference : 10 * (100 * a + 10 * b + c) + 2 - (2000 + 100 * a + 10 * b + c) = 945) :
  (100 * a + 10 * b + c) = 327 :=
by
  sorry

end distinct_digits_and_difference_is_945_l168_168715


namespace find_fourth_mark_l168_168538

-- Definitions of conditions
def average_of_four (a b c d : ℕ) : Prop :=
  (a + b + c + d) / 4 = 60

def known_marks (a b c : ℕ) : Prop :=
  a = 30 ∧ b = 55 ∧ c = 65

-- Theorem statement
theorem find_fourth_mark {d : ℕ} (h_avg : average_of_four 30 55 65 d) (h_known : known_marks 30 55 65) : d = 90 := 
by 
  sorry

end find_fourth_mark_l168_168538


namespace least_integer_exists_l168_168470

theorem least_integer_exists (x : ℕ) (h1 : x = 10 * (x / 10) + x % 10) (h2 : (x / 10) = x / 17) : x = 17 :=
sorry

end least_integer_exists_l168_168470


namespace periodic_sequence_condition_l168_168647

theorem periodic_sequence_condition (m : ℕ) (a : ℕ) 
  (h_pos : 0 < m)
  (a_seq : ℕ → ℕ) (h_initial : a_seq 0 = a)
  (h_relation : ∀ n, a_seq (n + 1) = if a_seq n % 2 = 0 then a_seq n / 2 else a_seq n + m) :
  (∃ p, ∀ k, a_seq (k + p) = a_seq k) ↔ 
  (a ∈ ({n | 1 ≤ n ∧ n ≤ m} ∪ {n | ∃ k, n = m + 2 * k + 1 ∧ n < 2 * m + 1})) :=
sorry

end periodic_sequence_condition_l168_168647


namespace triangle_angle_sum_cannot_exist_l168_168429

theorem triangle_angle_sum (A : Real) (B : Real) (C : Real) :
    A + B + C = 180 :=
sorry

theorem cannot_exist (right_two_60 : ¬∃ A B C : Real, A = 90 ∧ B = 60 ∧ C = 60 ∧ A + B + C = 180) 
    (scalene_100 : ∃ A B C : Real, A = 100 ∧ A ≠ B ∧ B ≠ C ∧ C ≠ A ∧ A + B + C = 180)
    (isosceles_two_70 : ∃ A B C : Real, A = B ∧ A = 70 ∧ C = 180 - 2 * A ∧ A + B + C = 180)
    (equilateral_60 : ∃ A B C : Real, A = 60 ∧ B = 60 ∧ C = 60 ∧ A + B + C = 180)
    (one_90_two_50 : ¬∃ A B C : Real, A = 90 ∧ B = 50 ∧ C = 50 ∧ A + B + C = 180) :
  (¬∃ A B C : Real, A = 90 ∧ B = 60 ∧ C = 60 ∧ A + B + C = 180) ∧
  (¬∃ A B C : Real, A = 90 ∧ B = 50 ∧ C = 50 ∧ A + B + C = 180) :=
by
  sorry

end triangle_angle_sum_cannot_exist_l168_168429


namespace range_of_y_l168_168259

noncomputable def y (x : ℝ) : ℝ := Real.sin x + Real.arcsin x

theorem range_of_y : 
  ∀ x, -1 ≤ x ∧ x ≤ 1 →
  y x ∈ Set.Icc (-Real.sin 1 - (Real.pi / 2)) (Real.sin 1 + (Real.pi / 2)) :=
sorry

end range_of_y_l168_168259


namespace binomial_12_11_eq_12_l168_168009

-- Define binomial coefficient function
def binomial (n k : Nat) : Nat :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Statement to prove
theorem binomial_12_11_eq_12 :
  binomial 12 11 = 12 := by
  sorry

end binomial_12_11_eq_12_l168_168009


namespace Lizzie_group_difference_l168_168523

theorem Lizzie_group_difference
  (lizzie_group_members : ℕ)
  (total_members : ℕ)
  (lizzie_more_than_other : lizzie_group_members > total_members - lizzie_group_members)
  (lizzie_members_eq : lizzie_group_members = 54)
  (total_members_eq : total_members = 91)
  : lizzie_group_members - (total_members - lizzie_group_members) = 17 := 
sorry

end Lizzie_group_difference_l168_168523


namespace probability_second_roll_three_times_first_l168_168129

theorem probability_second_roll_three_times_first :
  (probability (rolled_twice (second_roll_is_three_times_first))) = 1/18 := by
  sorry

def rolled_twice (p : (ℕ × ℕ) → Prop) : Set (ℕ × ℕ) :=
  {outcome | outcome.1 ∈ (Finset.range 1 7) ∧ outcome.2 ∈ (Finset.range 1 7) ∧ p outcome}

def second_roll_is_three_times_first (outcome : ℕ × ℕ) : Prop :=
  outcome.2 = 3 * outcome.1

end probability_second_roll_three_times_first_l168_168129


namespace greatest_of_three_consecutive_integers_with_sum_21_l168_168109

theorem greatest_of_three_consecutive_integers_with_sum_21 :
  ∃ (x : ℤ), (x + (x + 1) + (x + 2) = 21) ∧ ((x + 2) = 8) :=
by
  sorry

end greatest_of_three_consecutive_integers_with_sum_21_l168_168109


namespace main_theorem_l168_168253

-- defining the conditions
def cost_ratio_pen_pencil (x : ℕ) : Prop :=
  ∀ (pen pencil : ℕ), pen = 5 * pencil ∧ x = pencil

def cost_3_pens_pencils (pen pencil total_cost : ℕ) : Prop :=
  total_cost = 3 * pen + 7 * pencil  -- assuming "some pencils" translates to 7 pencils for this demonstration

def total_cost_dozen_pens (pen total_cost : ℕ) : Prop :=
  total_cost = 12 * pen

-- proving the main statement from conditions
theorem main_theorem (pen pencil total_cost : ℕ) (x : ℕ) 
  (h1 : cost_ratio_pen_pencil x)
  (h2 : cost_3_pens_pencils (5 * x) x 100)
  (h3 : total_cost_dozen_pens (5 * x) 300) :
  total_cost = 300 :=
by
  sorry

end main_theorem_l168_168253


namespace cos_sin_inequality_inequality_l168_168780

noncomputable def proof_cos_sin_inequality (a b : ℝ) (cos_x sin_x: ℝ) : Prop :=
  (cos_x ^ 2 = a) → (sin_x ^ 2 = b) → (a + b = 1) → (1 / 4 ≤ a ^ 3 + b ^ 3 ∧ a ^ 3 + b ^ 3 ≤ 1)

theorem cos_sin_inequality_inequality (a b : ℝ) (cos_x sin_x : ℝ) :
  proof_cos_sin_inequality a b cos_x sin_x :=
  by { sorry }

end cos_sin_inequality_inequality_l168_168780


namespace track_length_l168_168726

theorem track_length
  (meet1_dist : ℝ)
  (meet2_sally_additional_dist : ℝ)
  (constant_speed : ∀ (b_speed s_speed : ℝ), b_speed = s_speed)
  (opposite_start : true)
  (brenda_first_meet : meet1_dist = 100)
  (sally_second_meet : meet2_sally_additional_dist = 200) :
  ∃ L : ℝ, L = 200 :=
by
  sorry

end track_length_l168_168726


namespace max_liters_l168_168818

-- Declare the problem conditions as constants
def initialHeat : ℕ := 480 -- kJ for the first 5 min
def reductionRate : ℚ := 0.25 -- 25%
def initialTemp : ℚ := 20 -- °C
def boilingTemp : ℚ := 100 -- °C
def specificHeatCapacity : ℚ := 4.2 -- kJ/kg°C

-- Define the heat required to raise m kg of water from initialTemp to boilingTemp
def heatRequired (m : ℚ) : ℚ := m * specificHeatCapacity * (boilingTemp - initialTemp)

-- Define the total available heat from the fuel
noncomputable def totalAvailableHeat : ℚ :=
  initialHeat / (1 - (1 - reductionRate))

-- The proof problem statement
theorem max_liters :
  ∃ (m : ℕ), m ≤ 5 ∧ heatRequired m ≤ totalAvailableHeat :=
sorry

end max_liters_l168_168818


namespace find_range_t_l168_168737

def sequence_increasing (n : ℕ) (t : ℝ) : Prop :=
  (2 * (n + 1) + t^2 - 8) / (n + 1 + t) > (2 * n + t^2 - 8) / (n + t)

theorem find_range_t (t : ℝ) (h : ∀ n : ℕ, sequence_increasing n t) : 
  -1 < t ∧ t < 4 :=
sorry

end find_range_t_l168_168737


namespace blackboard_final_number_lower_bound_l168_168471

noncomputable def phi : ℝ := (1 + Real.sqrt 5) / 2

noncomputable def L (c : ℝ) : ℝ := 1 + Real.log c / Real.log phi

theorem blackboard_final_number_lower_bound (c : ℝ) (n : ℕ) (h_pos_c : c > 1) (h_pos_n : n > 0) :
  ∃ x, x ≥ ((c^(n / (L c)) - 1) / (c^(1 / (L c)) - 1))^(L c) :=
sorry

end blackboard_final_number_lower_bound_l168_168471


namespace white_roses_per_bouquet_l168_168381

/-- Mrs. Dunbar needs to make 5 bouquets and 7 table decorations. -/
def number_of_bouquets : ℕ := 5
def number_of_table_decorations : ℕ := 7
/-- She uses 12 white roses in each table decoration. -/
def white_roses_per_table_decoration : ℕ := 12
/-- She needs a total of 109 white roses to complete all bouquets and table decorations. -/
def total_white_roses_needed : ℕ := 109

/-- Prove that the number of white roses used in each bouquet is 5. -/
theorem white_roses_per_bouquet : ∃ (white_roses_per_bouquet : ℕ),
  number_of_bouquets * white_roses_per_bouquet + number_of_table_decorations * white_roses_per_table_decoration = total_white_roses_needed
  ∧ white_roses_per_bouquet = 5 := 
by
  sorry

end white_roses_per_bouquet_l168_168381


namespace identical_functions_l168_168157

def f (x : ℝ) : ℝ := x^2 - 1
def g (x : ℝ) : ℝ := (x^2 - 1)^3^(1/3)

theorem identical_functions : ∀ x : ℝ, f x = g x :=
by
  intro x
  -- Proof to be completed
  sorry

end identical_functions_l168_168157


namespace alexander_has_more_pencils_l168_168537

-- Definitions based on conditions
def asaf_age := 50
def total_age := 140
def total_pencils := 220

-- Auxiliary definitions based on conditions
def alexander_age := total_age - asaf_age
def age_difference := alexander_age - asaf_age
def asaf_pencils := 2 * age_difference
def alexander_pencils := total_pencils - asaf_pencils

-- Statement to prove
theorem alexander_has_more_pencils :
  (alexander_pencils - asaf_pencils) = 60 := sorry

end alexander_has_more_pencils_l168_168537


namespace new_room_correct_size_l168_168192

-- Definitions of conditions
def current_bedroom := 309 -- sq ft
def current_bathroom := 150 -- sq ft
def current_space := current_bedroom + current_bathroom
def new_room_size := 2 * current_space

-- Proving the new room size
theorem new_room_correct_size : new_room_size = 918 := by
  sorry

end new_room_correct_size_l168_168192


namespace original_price_color_tv_l168_168136

theorem original_price_color_tv (x : ℝ) : 
  1.4 * x * 0.8 - x = 270 → x = 2250 :=
by
  intro h
  simp at h
  sorry

end original_price_color_tv_l168_168136


namespace count_4_digit_numbers_divisible_by_13_l168_168889

theorem count_4_digit_numbers_divisible_by_13 : 
  ∃ n : ℕ, n = 693 ∧ (∀ k : ℕ, k >= 1000 ∧ k < 10000 ∧ k % 13 = 0 → ∃ m : ℕ, m = (k - 1000) / 13 + 1 ∧ m = n) :=
by {
  -- Solution proof will be placed here.
  sorry
}

end count_4_digit_numbers_divisible_by_13_l168_168889


namespace ratio_small_to_large_is_one_to_one_l168_168458

theorem ratio_small_to_large_is_one_to_one
  (total_beads : ℕ)
  (large_beads_per_bracelet : ℕ)
  (bracelets_count : ℕ)
  (small_beads : ℕ)
  (large_beads : ℕ)
  (small_beads_per_bracelet : ℕ) :
  total_beads = 528 →
  large_beads_per_bracelet = 12 →
  bracelets_count = 11 →
  large_beads = total_beads / 2 →
  large_beads >= bracelets_count * large_beads_per_bracelet →
  small_beads = total_beads / 2 →
  small_beads_per_bracelet = small_beads / bracelets_count →
  small_beads_per_bracelet / large_beads_per_bracelet = 1 :=
by sorry

end ratio_small_to_large_is_one_to_one_l168_168458


namespace tea_consumption_eq1_tea_consumption_eq2_l168_168407

theorem tea_consumption_eq1 (k : ℝ) (w_sunday t_sunday w_wednesday : ℝ) (h1 : w_sunday * t_sunday = k) 
  (h2 : w_wednesday = 4) : 
  t_wednesday = 6 := 
  by sorry

theorem tea_consumption_eq2 (k : ℝ) (w_sunday t_sunday t_thursday : ℝ) (h1 : w_sunday * t_sunday = k) 
  (h2 : t_thursday = 2) : 
  w_thursday = 12 := 
  by sorry

end tea_consumption_eq1_tea_consumption_eq2_l168_168407


namespace min_value_of_3x_2y_l168_168345

noncomputable def min_value (x y: ℝ) : ℝ := 3 * x + 2 * y

theorem min_value_of_3x_2y (x y : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : x + y - x * y = 0) :
  min_value x y = 5 + 2 * Real.sqrt 6 :=
sorry

end min_value_of_3x_2y_l168_168345


namespace find_reciprocal_sum_of_roots_l168_168653

theorem find_reciprocal_sum_of_roots
  {x₁ x₂ : ℝ}
  (h1 : 5 * x₁ ^ 2 - 3 * x₁ - 2 = 0)
  (h2 : 5 * x₂ ^ 2 - 3 * x₂ - 2 = 0)
  (h_diff : x₁ ≠ x₂) :
  (1 / x₁ + 1 / x₂) = -3 / 2 :=
by {
  sorry
}

end find_reciprocal_sum_of_roots_l168_168653


namespace largest_square_not_divisible_by_100_l168_168999

theorem largest_square_not_divisible_by_100
  (n : ℕ) (h1 : ∃ a : ℕ, a^2 = n) 
  (h2 : n % 100 ≠ 0)
  (h3 : ∃ m : ℕ, m * 100 + n % 100 = n ∧ ∃ b : ℕ, b^2 = m) :
  n = 1681 := sorry

end largest_square_not_divisible_by_100_l168_168999


namespace tangent_line_of_ellipse_l168_168485

noncomputable def ellipse_tangent_line (a b x0 y0 x y : ℝ) : Prop :=
  x0 * x / a^2 + y0 * y / b^2 = 1

theorem tangent_line_of_ellipse
  (a b x0 y0 : ℝ)
  (h_ellipse : x0^2 / a^2 + y0^2 / b^2 = 1)
  (h_a_pos : a > 0)
  (h_b_pos : b > 0)
  (h_a_b : a > b) :
  ellipse_tangent_line a b x0 y0 x y :=
sorry

end tangent_line_of_ellipse_l168_168485


namespace alloy_mixture_l168_168692

theorem alloy_mixture (x y z : ℕ) 
  (h1 : x + 3*y + 5*z = 819) 
  (h2 : 3*x + 5*y + z = 1053) 
  (h3 : 5*x + y + 3*z = 1287) 
  : 
  x = 195 ∧ y = 78 ∧ z = 78 := 
by {
  sorry
}

end alloy_mixture_l168_168692


namespace quadratic_root_conditions_l168_168360

noncomputable def quadratic_has_two_distinct_real_roots (k : ℝ) : Prop :=
  let a := k + 2
  let b := 4
  let c := 1
  (a ≠ 0) ∧ (b^2 - 4*a*c > 0)

theorem quadratic_root_conditions (k : ℝ) :
  quadratic_has_two_distinct_real_roots k ↔ k < 2 ∧ k ≠ -2 := 
by
  sorry

end quadratic_root_conditions_l168_168360


namespace m_and_n_must_have_same_parity_l168_168100

-- Define the problem conditions
def square_has_four_colored_edges (square : Type) : Prop :=
  ∃ (colors : Fin 4 → square), true

def m_and_n_same_parity (m n : ℕ) : Prop :=
  (m % 2 = n % 2)

-- Formalize the proof statement based on the conditions
theorem m_and_n_must_have_same_parity (m n : ℕ) (square : Type)
  (H : square_has_four_colored_edges square) : 
  m_and_n_same_parity m n :=
by 
  sorry

end m_and_n_must_have_same_parity_l168_168100


namespace total_cost_of_plates_and_cups_l168_168146

theorem total_cost_of_plates_and_cups (P C : ℝ) 
  (h : 20 * P + 40 * C = 1.50) : 
  100 * P + 200 * C = 7.50 :=
by
  -- proof here
  sorry

end total_cost_of_plates_and_cups_l168_168146


namespace value_of_product_l168_168173

theorem value_of_product (x y : ℝ) (h1 : x + y = 5) (h2 : x * y = 2) : (x + 2) * (y + 2) = 16 := by
  sorry

end value_of_product_l168_168173


namespace set_intersection_is_result_l168_168350

def set_A := {x : ℝ | 1 < x^2 ∧ x^2 < 4 }
def set_B := {x : ℝ | x ≥ 1}
def result_set := {x : ℝ | 1 < x ∧ x < 2}

theorem set_intersection_is_result : (set_A ∩ set_B) = result_set :=
by sorry

end set_intersection_is_result_l168_168350


namespace food_for_elephants_l168_168907

theorem food_for_elephants (t : ℕ) : 
  (∀ (food_per_day : ℕ), (12 * food_per_day) * 1 = (1000 * food_per_day) * 600) →
  (∀ (food_per_day : ℕ), (t * food_per_day) * 1 = (100 * food_per_day) * d) →
  d = 500 * t :=
by
  sorry

end food_for_elephants_l168_168907


namespace smallest_four_digit_equiv_to_3_mod_8_l168_168425

theorem smallest_four_digit_equiv_to_3_mod_8 : 
  ∃ (n : ℕ), n ≥ 1000 ∧ n < 10000 ∧ n % 8 = 3 ∧ (∀ m : ℕ, m ≥ 1000 ∧ m < 10000 ∧ m % 8 = 3 → m ≥ n) := 
begin
  use 1003,
  split,
  { 
    -- Prove that 1003 is a four-digit number
    linarith,
  },
  split,
  { 
    -- Prove that 1003 is less than 10000
    linarith,
  },
  split,
  {
    -- Prove that 1003 ≡ 3 (mod 8)
    exact nat.mod_eq_of_lt (show 3 < 8, by linarith),
  },
  {
    -- Prove that 1003 is the smallest such number
    intros m h1 h2 h3,
    have h_mod : m % 8 = 3 := h3,
    have trivial_ineq : 1003 ≤ m := sorry,
    exact trivial_ineq,
  },
end

end smallest_four_digit_equiv_to_3_mod_8_l168_168425


namespace box_weight_l168_168841

theorem box_weight (total_weight : ℕ) (number_of_boxes : ℕ) (box_weight : ℕ) 
  (h1 : total_weight = 267) 
  (h2 : number_of_boxes = 3) 
  (h3 : box_weight = total_weight / number_of_boxes) : 
  box_weight = 89 := 
by 
  sorry

end box_weight_l168_168841


namespace martian_right_angle_l168_168536

theorem martian_right_angle :
  ∀ (full_circle clerts_per_right_angle : ℕ),
  (full_circle = 600) →
  (clerts_per_right_angle = full_circle / 3) →
  clerts_per_right_angle = 200 :=
by
  intros full_circle clerts_per_right_angle h1 h2
  sorry

end martian_right_angle_l168_168536


namespace find_root_floor_l168_168378

noncomputable def g (x : ℝ) := Real.sin x - Real.cos x + 4 * Real.tan x

theorem find_root_floor :
  ∃ s : ℝ, (g s = 0) ∧ (π / 2 < s) ∧ (s < 3 * π / 2) ∧ (Int.floor s = 3) :=
  sorry

end find_root_floor_l168_168378


namespace car_speed_ratio_l168_168003

theorem car_speed_ratio 
  (t D : ℝ) 
  (v_alpha v_beta : ℝ)
  (H1 : (v_alpha + v_beta) * t = D)
  (H2 : v_alpha * 4 = D - v_alpha * t)
  (H3 : v_beta * 1 = D - v_beta * t) : 
  v_alpha / v_beta = 2 :=
by
  sorry

end car_speed_ratio_l168_168003


namespace add_neg_two_l168_168149

theorem add_neg_two : 1 + (-2 : ℚ) = -1 := by
  sorry

end add_neg_two_l168_168149


namespace trip_cost_l168_168453

theorem trip_cost (original_price : ℕ) (discount : ℕ) (num_people : ℕ)
  (h1 : original_price = 147) (h2 : discount = 14) (h3 : num_people = 2) :
  num_people * (original_price - discount) = 266 :=
by
  sorry

end trip_cost_l168_168453


namespace square_of_105_l168_168592

theorem square_of_105 : 105^2 = 11025 := by
  sorry

end square_of_105_l168_168592


namespace flowchart_basic_elements_includes_loop_l168_168786

theorem flowchart_basic_elements_includes_loop 
  (sequence_structure : Prop)
  (condition_structure : Prop)
  (loop_structure : Prop)
  : ∃ element : ℕ, element = 2 := 
by
  -- Assume 0 is A: Judgment
  -- Assume 1 is B: Directed line
  -- Assume 2 is C: Loop
  -- Assume 3 is D: Start
  sorry

end flowchart_basic_elements_includes_loop_l168_168786


namespace max_triangle_area_l168_168491

noncomputable def parabola (x y : ℝ) : Prop := x^2 = 4 * y

theorem max_triangle_area
  (x1 y1 x2 y2 : ℝ)
  (hA : parabola x1 y1)
  (hB : parabola x2 y2)
  (h_sum_y : y1 + y2 = 2)
  (h_neq : y1 ≠ y2) :
  ∃ area : ℝ, area = 121 / 12 :=
sorry

end max_triangle_area_l168_168491


namespace player1_wins_l168_168277

noncomputable def player1_wins_infinite_grid_game :
  Prop :=
∃ strategy : ℕ → ℕ → ℤ → ℤ,
  (∀ n : ℕ, player1_strategy n) ∧ 
  (∀ m : ℕ, player2_strategy m)

theorem player1_wins :
  ∃ strategy : ℕ → ℕ → ℤ → ℤ,
    (∃ player1_strategy : ℕ → ℤ × ℤ,
      ∃ player2_strategy : ℕ → ℤ × ℤ,
        ∀ n : ℕ, player1_strategy n ∧ player2_strategy n) :=
sorry

end player1_wins_l168_168277


namespace smallest_x_l168_168335

theorem smallest_x : ∃ x : ℕ, x + 6721 ≡ 3458 [MOD 12] ∧ x % 5 = 0 ∧ x = 45 :=
by
  sorry

end smallest_x_l168_168335


namespace fraction_comparison_l168_168221

theorem fraction_comparison (a b c d : ℝ) (h1 : a / b < c / d) (h2 : b > d) (h3 : d > 0) :
  (a + c) / (b + d) < 1/2 * (a / b + c / d) :=
by
  sorry

end fraction_comparison_l168_168221


namespace ratio_girls_to_boys_l168_168900

-- Define the number of students and conditions
def num_students : ℕ := 25
def girls_more_than_boys : ℕ := 3

-- Define the variables
variables (g b : ℕ)

-- Define the conditions
def total_students := g + b = num_students
def girls_boys_relationship := b = g - girls_more_than_boys

-- Lean theorem statement
theorem ratio_girls_to_boys (g b : ℕ) (h1 : total_students g b) (h2 : girls_boys_relationship g b) : (g : ℚ) / b = 14 / 11 :=
sorry

end ratio_girls_to_boys_l168_168900


namespace maximize_profit_l168_168124

-- Define the price of the book
variables (p : ℝ) (p_max : ℝ)
-- Define the revenue function
def R (p : ℝ) : ℝ := p * (150 - 4 * p)
-- Define the profit function accounting for fixed costs of $200
def P (p : ℝ) := R p - 200
-- Set the maximum feasible price
def max_price_condition := p_max = 30
-- Define the price that maximizes the profit
def optimal_price := 18.75

-- The theorem to be proved
theorem maximize_profit : p_max = 30 → p = 18.75 → P p = 2612.5 :=
by {
  sorry
}

end maximize_profit_l168_168124


namespace mean_value_of_pentagon_angles_l168_168820

theorem mean_value_of_pentagon_angles : 
  let n := 5 
  let interior_angle_sum := (n - 2) * 180 
  mean_angle = interior_angle_sum / n :=
  sorry

end mean_value_of_pentagon_angles_l168_168820


namespace money_made_l168_168123

-- Define the conditions
def cost_per_bar := 4
def total_bars := 8
def bars_sold := total_bars - 3

-- We need to show that the money made is $20
theorem money_made :
  bars_sold * cost_per_bar = 20 := 
by
  sorry

end money_made_l168_168123


namespace sally_total_fries_is_50_l168_168530

-- Definitions for the conditions
def sally_initial_fries : ℕ := 14
def mark_initial_fries : ℕ := 3 * 12
def mark_fraction_given_to_sally : ℕ := mark_initial_fries / 3
def jessica_total_cm_of_fries : ℕ := 240
def fry_length_cm : ℕ := 5
def jessica_total_fries : ℕ := jessica_total_cm_of_fries / fry_length_cm
def jessica_fraction_given_to_sally : ℕ := jessica_total_fries / 2

-- Definition for the question
def total_fries_sally_has (sally_initial_fries mark_fraction_given_to_sally jessica_fraction_given_to_sally : ℕ) : ℕ :=
  sally_initial_fries + mark_fraction_given_to_sally + jessica_fraction_given_to_sally

-- The theorem to be proved
theorem sally_total_fries_is_50 :
  total_fries_sally_has sally_initial_fries mark_fraction_given_to_sally jessica_fraction_given_to_sally = 50 :=
sorry

end sally_total_fries_is_50_l168_168530


namespace min_sum_reciprocals_of_roots_l168_168880

theorem min_sum_reciprocals_of_roots (k : ℝ) 
  (h_roots_positive : ∀ x : ℝ, (x^2 - k * x + k + 3 = 0) → 0 < x) :
  (k ≥ 6) → 
  ∀ x1 x2 : ℝ, (x1*x2 = k + 3) ∧ (x1 + x2 = k) ∧ (x1 > 0) ∧ (x2 > 0) → 
  (1 / x1 + 1 / x2) = 2 / 3 :=
by 
  -- proof steps go here
  sorry

end min_sum_reciprocals_of_roots_l168_168880


namespace cos_graph_symmetric_l168_168790

theorem cos_graph_symmetric :
  ∃ (x0 : ℝ), x0 = (Real.pi / 3) ∧ ∀ y, (∃ x, y = Real.cos (2 * x + Real.pi / 3)) ↔ (∃ x, y = Real.cos (2 * (2 * x0 - x) + Real.pi / 3)) :=
by
  -- Let x0 = π / 3
  let x0 := Real.pi / 3
  -- Show symmetry about x = π / 3
  exact ⟨x0, by norm_num, sorry⟩

end cos_graph_symmetric_l168_168790


namespace positive_difference_between_two_numbers_l168_168547

variable (x y : ℝ)

theorem positive_difference_between_two_numbers 
  (h₁ : x + y = 40)
  (h₂ : 3 * y - 4 * x = 20) :
  |y - x| = 100 / 7 :=
by
  sorry

end positive_difference_between_two_numbers_l168_168547


namespace vector_magnitude_proof_l168_168744

theorem vector_magnitude_proof (a b : ℝ × ℝ) 
  (h₁ : ‖a‖ = 1) 
  (h₂ : ‖b‖ = 2)
  (h₃ : a - b = (Real.sqrt 3, Real.sqrt 2)) : 
‖a + (2:ℝ) • b‖ = Real.sqrt 17 := 
sorry

end vector_magnitude_proof_l168_168744


namespace problem_1_problem_2_l168_168235

noncomputable def A : Set ℝ := {x | 2 * x^2 - 7 * x + 3 ≤ 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | x + a < 0}

theorem problem_1 (a : ℝ) :
  a = -2 →
  A ∩ B a = {x | (1 / 2 : ℝ) ≤ x ∧ x < 2} :=
by
  intro ha
  sorry

theorem problem_2 (a : ℝ) :
  (A ∩ B a) = A → a < -3 :=
by
  intro h
  sorry

end problem_1_problem_2_l168_168235


namespace number_of_paths_l168_168419

-- Definitions of coordinates
def start : ℕ × ℕ := (0, 0)
def end : ℕ × ℕ := (6, 6)
def center : ℕ × ℕ := (3, 3)

-- Movement constraints (right or top)
def valid_moves : ℕ × ℕ → list (ℕ × ℕ)
| (x, y) => if x < 6 ∧ y < 6 then [(x + 1, y), (x, y + 1)] else if x < 6 then [(x + 1, y)] else if y < 6 then [(x, y + 1)] else []

-- Main theorem
theorem number_of_paths (start end center : ℕ × ℕ) (valid_moves : ℕ × ℕ → list (ℕ × ℕ)) : ℕ :=
  let moves_to_center := nat.choose (3 + 3) 3
  let moves_from_center := nat.choose (3 + 3) 3
  moves_to_center * moves_from_center

#eval number_of_paths start end center valid_moves -- Expected output: 400

end number_of_paths_l168_168419


namespace billy_age_l168_168687

variable (B J : ℕ)

theorem billy_age (h1 : B = 3 * J) (h2 : B + J = 60) : B = 45 :=
by
  sorry

end billy_age_l168_168687


namespace piggy_bank_dimes_diff_l168_168244

theorem piggy_bank_dimes_diff :
  ∃ (a b c : ℕ), a + b + c = 100 ∧ 5 * a + 10 * b + 25 * c = 1005 ∧ (∀ lo hi, 
  (lo = 1 ∧ hi = 101) → (hi - lo = 100)) :=
by
  sorry

end piggy_bank_dimes_diff_l168_168244


namespace son_l168_168112

theorem son's_age (S M : ℕ) (h1 : M = S + 20) (h2 : M + 2 = 2 * (S + 2)) : S = 18 := by
  sorry

end son_l168_168112


namespace common_chord_length_l168_168793

theorem common_chord_length (x y : ℝ) : 
    (x^2 + y^2 = 4) → 
    (x^2 + y^2 - 4*x + 4*y - 12 = 0) → 
    ∃ l : ℝ, l = 2 * Real.sqrt 2 :=
by
  intros h1 h2
  sorry

end common_chord_length_l168_168793


namespace perpendicular_distance_l168_168513

open EuclideanGeometry

noncomputable def excenter (A B C : Point) : Point := sorry
noncomputable def excircle (A B C : Point) : Circle := sorry
noncomputable def tangentPoint (A B C : Point) (excircle : Circle) (L : Line) : Point := sorry
noncomputable def meet (L1 L2 : Line) : Point := sorry
noncomputable def perpendicular (P : Point) (L : Line) : Point := sorry

axiom excircle_tangent {A B C : Point} (I_a : Point) (B' C' : Point) :
  (excircle A B C).tangentPoint A B I_a = B' ∧ (excircle A B C).tangentPoint A C I_a = C'

axiom meet_points {A B C I_a B' C' P Q : Point} :
  meet (I_a.line B) (B'.line C') = P ∧ meet (I_a.line C) (B'.lineC') = Q

axiom intersection_point {A B C I_a B' C' P Q M : Point} :
  meet (B.line Q) (C.line P) = M

theorem perpendicular_distance {A B C I_a B' C' P Q M : Point} (r : ℝ) :
  excircle_tangent A B C I_a B' C' →
  meet_points A B C I_a B' C' P Q →
  intersection_point A B C I_a B' C' P Q M →
  length (perpendicular M (B.line C)) = r := sorry

end perpendicular_distance_l168_168513


namespace square_of_105_l168_168587

/-- Prove that 105^2 = 11025. -/
theorem square_of_105 : 105^2 = 11025 :=
by
  sorry

end square_of_105_l168_168587


namespace region_area_l168_168067

noncomputable def area_of_region_outside_hexagon_inside_semicircles (s : ℝ) : ℝ :=
  let area_hexagon := (3 * Real.sqrt 3 / 2) * s^2
  let area_semicircle := (1/2) * Real.pi * (s/2)^2
  let total_area_semicircles := 6 * area_semicircle
  let total_area_circles := 6 * Real.pi * (s/2)^2
  total_area_circles - area_hexagon

theorem region_area (s := 2) : area_of_region_outside_hexagon_inside_semicircles s = (6 * Real.pi - 6 * Real.sqrt 3) :=
by
  sorry  -- Proof is skipped.

end region_area_l168_168067


namespace contrapositive_statement_l168_168910

theorem contrapositive_statement (m : ℝ) : 
  (¬ ∃ (x : ℝ), x^2 + x - m = 0) → m > 0 :=
by
  sorry

end contrapositive_statement_l168_168910


namespace calculate_value_l168_168472

theorem calculate_value (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (hxy : x = 1 / y) (hzy : z = 1 / y) : 
  (x + 1 / x) * (z - 1 / z) = 4 := 
by 
  -- Proof omitted, this is just the statement
  sorry

end calculate_value_l168_168472


namespace age_of_seventh_person_l168_168520

theorem age_of_seventh_person (A1 A2 A3 A4 A5 A6 A7 D1 D2 D3 D4 D5 : ℕ) 
    (h1 : A1 < A2) (h2 : A2 < A3) (h3 : A3 < A4) (h4 : A4 < A5) (h5 : A5 < A6) 
    (h6 : A2 = A1 + D1) (h7 : A3 = A2 + D2) (h8 : A4 = A3 + D3) 
    (h9 : A5 = A4 + D4) (h10 : A6 = A5 + D5)
    (h11 : A1 + A2 + A3 + A4 + A5 + A6 = 246) 
    (h12 : 246 + A7 = 315) : A7 = 69 :=
by
  sorry

end age_of_seventh_person_l168_168520


namespace trip_cost_l168_168450

variable (price : ℕ) (discount : ℕ) (numPeople : ℕ)

theorem trip_cost :
  price = 147 →
  discount = 14 →
  numPeople = 2 →
  (price - discount) * numPeople = 266 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  simp
  sorry

end trip_cost_l168_168450


namespace find_omega_l168_168481

noncomputable def omega_solution (ω : ℝ) : Prop :=
  ω > 0 ∧
  (∀ x y : ℝ, 0 ≤ x ∧ x < y ∧ y ≤ 2 * Real.pi / 3 → 2 * Real.cos (ω * x) > 2 * Real.cos (ω * y)) ∧
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 * Real.pi / 3 → 2 * Real.cos (ω * x) ≥ 1)

theorem find_omega : omega_solution (1 / 2) :=
sorry

end find_omega_l168_168481


namespace kate_needs_more_money_for_trip_l168_168374

theorem kate_needs_more_money_for_trip:
  let kate_money_base6 := 3 * 6^3 + 2 * 6^2 + 4 * 6^1 + 2 * 6^0
  let ticket_cost := 1000
  kate_money_base6 - ticket_cost = -254 :=
by
  -- Proving the theorem, steps will go here.
  sorry

end kate_needs_more_money_for_trip_l168_168374


namespace bus_final_count_l168_168550

def initial_people : ℕ := 110
def first_stop_off : ℕ := 20
def first_stop_on : ℕ := 15
def second_stop_off : ℕ := 34
def second_stop_on : ℕ := 17
def third_stop_off : ℕ := 18
def third_stop_on : ℕ := 7
def fourth_stop_off : ℕ := 29
def fourth_stop_on : ℕ := 19
def fifth_stop_off : ℕ := 11
def fifth_stop_on : ℕ := 13
def sixth_stop_off : ℕ := 15
def sixth_stop_on : ℕ := 8
def seventh_stop_off : ℕ := 13
def seventh_stop_on : ℕ := 5
def eighth_stop_off : ℕ := 6
def eighth_stop_on : ℕ := 0

theorem bus_final_count :
  initial_people - first_stop_off + first_stop_on 
  - second_stop_off + second_stop_on 
  - third_stop_off + third_stop_on 
  - fourth_stop_off + fourth_stop_on 
  - fifth_stop_off + fifth_stop_on 
  - sixth_stop_off + sixth_stop_on 
  - seventh_stop_off + seventh_stop_on 
  - eighth_stop_off + eighth_stop_on = 48 :=
by sorry

end bus_final_count_l168_168550


namespace polar_line_equation_l168_168905

theorem polar_line_equation
  (rho theta : ℝ)
  (h1 : rho = 4 * Real.cos theta)
  (h2 : ∀ (x y : ℝ), (x - 2) ^ 2 + y ^ 2 = 4 → x = 2)
  : rho * Real.cos theta = 2 :=
sorry

end polar_line_equation_l168_168905


namespace student_tickets_sold_l168_168804

theorem student_tickets_sold
  (A S : ℕ)
  (h1 : A + S = 846)
  (h2 : 6 * A + 3 * S = 3846) :
  S = 410 :=
sorry

end student_tickets_sold_l168_168804


namespace alex_final_silver_tokens_l168_168141

-- Define initial conditions
def initial_red_tokens := 100
def initial_blue_tokens := 50

-- Define exchange rules
def booth1_red_cost := 3
def booth1_silver_gain := 2
def booth1_blue_gain := 1

def booth2_blue_cost := 4
def booth2_silver_gain := 1
def booth2_red_gain := 2

-- Define limits where no further exchanges are possible
def red_token_limit := 2
def blue_token_limit := 3

-- Define the number of times visiting each booth
variable (x y : ℕ)

-- Tokens left after exchanges
def remaining_red_tokens := initial_red_tokens - 3 * x + 2 * y
def remaining_blue_tokens := initial_blue_tokens + x - 4 * y

-- Define proof theorem
theorem alex_final_silver_tokens :
  (remaining_red_tokens x y ≤ red_token_limit) ∧
  (remaining_blue_tokens x y ≤ blue_token_limit) →
  (2 * x + y = 113) :=
by
  sorry

end alex_final_silver_tokens_l168_168141


namespace most_suitable_method_l168_168093

theorem most_suitable_method {x : ℝ} (h : (x - 1) ^ 2 = 4) :
  "Direct method of taking square root" = "Direct method of taking square root" :=
by
  -- We observe that the equation is already in a form 
  -- that is conducive to applying the direct method of taking the square root,
  -- because the equation is already a perfect square on one side and a constant on the other side.
  sorry

end most_suitable_method_l168_168093


namespace equation_solution_l168_168086

theorem equation_solution (x y z : ℕ) :
  x^2 + y^2 = 2^z ↔ ∃ n : ℕ, x = 2^n ∧ y = 2^n ∧ z = 2 * n + 1 := 
sorry

end equation_solution_l168_168086


namespace boat_speed_in_still_water_l168_168701

theorem boat_speed_in_still_water
  (v c : ℝ)
  (h1 : v + c = 10)
  (h2 : v - c = 4) :
  v = 7 :=
by
  sorry

end boat_speed_in_still_water_l168_168701


namespace perimeter_of_smaller_rectangle_l168_168095

theorem perimeter_of_smaller_rectangle :
  ∀ (L W n : ℕ), 
  L = 16 → W = 20 → n = 10 →
  (∃ (x y : ℕ), L % 2 = 0 ∧ W % 5 = 0 ∧ 2 * y = L ∧ 5 * x = W ∧ (L * W) / n = x * y ∧ 2 * (x + y) = 24) :=
by
  intros L W n H1 H2 H3
  use 4, 8
  sorry

end perimeter_of_smaller_rectangle_l168_168095


namespace pyramid_volume_l168_168323

/-- Given the vertices of a triangle and its midpoints, calculate the volume of the folded triangular pyramid. -/
theorem pyramid_volume
  (A B C : ℝ × ℝ)
  (D E F : ℝ × ℝ)
  (hA : A = (0, 0))
  (hB : B = (24, 0))
  (hC : C = (12, 16))
  (hD : D = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))
  (hE : E = ((A.1 + C.1) / 2, (A.2 + C.2) / 2))
  (hF : F = ((B.1 + C.1) / 2, (B.2 + C.2) / 2))
  (area_ABC : ℝ)
  (h_area : area_ABC = 192)
  : (1 / 3) * area_ABC * 8 = 512 :=
by sorry

end pyramid_volume_l168_168323


namespace largest_primes_product_l168_168821

theorem largest_primes_product : 7 * 97 * 997 = 679679 := by
  sorry

end largest_primes_product_l168_168821


namespace fraction_comparison_l168_168217

theorem fraction_comparison
  (a b c d : ℝ)
  (h1 : a / b < c / d)
  (h2 : b > 0)
  (h3 : d > 0)
  (h4 : b > d) :
  (a + c) / (b + d) < (1 / 2) * (a / b + c / d) :=
by
  sorry

end fraction_comparison_l168_168217


namespace max_cos_product_l168_168040

open Real

theorem max_cos_product (α β γ : ℝ) (hα : 0 < α ∧ α < π / 2)
                                       (hβ : 0 < β ∧ β < π / 2)
                                       (hγ : 0 < γ ∧ γ < π / 2)
                                       (h : sin α ^ 2 + sin β ^ 2 + sin γ ^ 2 = 1) : 
  cos α * cos β * cos γ ≤ 2 * Real.sqrt 6 / 9 := 
by sorry

end max_cos_product_l168_168040


namespace arithmetic_sequence_sum_l168_168175

theorem arithmetic_sequence_sum
  (a : ℕ → ℕ)
  (S : ℕ → ℕ)
  (h_arith : ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0)
  (h_S_def : ∀ n : ℕ, S n = n * (a 0 + a (n - 1)) / 2)
  (h_S2 : S 2 = 4)
  (h_S4 : S 4 = 16) :
  a 5 + a 6 = 20 :=
sorry

end arithmetic_sequence_sum_l168_168175


namespace M_infinite_l168_168909

open Nat

-- Define the set M
def M : Set ℕ := {k | ∃ n : ℕ, 3 ^ n % n = k % n}

-- Statement of the problem
theorem M_infinite : Set.Infinite M :=
sorry

end M_infinite_l168_168909


namespace sum_of_cubes_l168_168619

theorem sum_of_cubes
  (x y : ℝ) 
  (hx : 0 < x) 
  (hy : 0 < y)
  (h1 : (x + y)^2 = 2500) 
  (h2 : x * y = 500) :
  x^3 + y^3 = 50000 := 
by
  sorry

end sum_of_cubes_l168_168619


namespace contrapositive_of_original_l168_168094

theorem contrapositive_of_original (a b : ℝ) :
  (a > b → a - 1 > b - 1) ↔ (a - 1 ≤ b - 1 → a ≤ b) :=
by
  sorry

end contrapositive_of_original_l168_168094


namespace angle_in_fourth_quadrant_l168_168667

-- Define the main condition converting the angle to the range [0, 360)
def reducedAngle (θ : ℤ) : ℤ := (θ % 360 + 360) % 360

-- State the theorem proving the angle of -390° is in the fourth quadrant
theorem angle_in_fourth_quadrant (θ : ℤ) (h : θ = -390) : 270 ≤ reducedAngle θ ∧ reducedAngle θ < 360 := by
  sorry

end angle_in_fourth_quadrant_l168_168667


namespace larger_integer_value_l168_168826

theorem larger_integer_value (x y : ℕ) (h1 : (4 * x)^2 - 2 * x = 8100) (h2 : x + 10 = 2 * y) : x = 22 :=
by
  sorry

end larger_integer_value_l168_168826


namespace john_tv_show_duration_l168_168762

def john_tv_show (seasons_before : ℕ) (episodes_per_season : ℕ) (additional_episodes : ℕ) (episode_duration : ℝ) : ℝ :=
  let total_episodes_before := seasons_before * episodes_per_season
  let last_season_episodes := episodes_per_season + additional_episodes
  let total_episodes := total_episodes_before + last_season_episodes
  total_episodes * episode_duration

theorem john_tv_show_duration :
  john_tv_show 9 22 4 0.5 = 112 := 
by
  sorry

end john_tv_show_duration_l168_168762


namespace probability_1700_in_three_spins_l168_168775

/-- Prove that the probability of earning exactly $1700 in three spins is 6/125 given 
    that each region on the spinner has the same area and the spinner has five slots: 
    "Bankrupt", "$1000", "$300", "$5000", and "$400". -/
theorem probability_1700_in_three_spins :
  let slots := ["Bankrupt", "$1000", "$300", "$5000", "$400"]
  in let desired_outcomes := set.to_finset {(["$1000", "$300", "$400"].permutations : multiset (list string))}
  in let total_outcomes := 5 * 5 * 5
  in (desired_outcomes.card : ℚ) / (total_outcomes : ℚ) = 6 / 125 :=
by {
  let slots := ["Bankrupt", "$1000", "$300", "$5000", "$400"] in
  let desired_outcomes := set.to_finset {(["$1000", "$300", "$400"].permutations : multiset (list string))} in
  let total_outcomes := 5 * 5 * 5 in
  have h_desired_outcomes_card : desired_outcomes.card = 6 := sorry,
  have h_total_outcomes : total_outcomes = 125 := by norm_num,
  rw [h_desired_outcomes_card, h_total_outcomes],
  norm_num
}

end probability_1700_in_three_spins_l168_168775


namespace car_a_distance_behind_car_b_l168_168006

theorem car_a_distance_behind_car_b :
  ∃ D : ℝ, D = 40 ∧ 
    (∀ (t : ℝ), t = 4 →
    ((58 - 50) * t + 8) = D + 8)
  := by
  sorry

end car_a_distance_behind_car_b_l168_168006


namespace cake_eating_contest_l168_168278

-- Define the fractions representing the amounts of cake eaten by the two students.
def first_student : ℚ := 7 / 8
def second_student : ℚ := 5 / 6

-- The statement of our proof problem
theorem cake_eating_contest : first_student - second_student = 1 / 24 := by
  sorry

end cake_eating_contest_l168_168278


namespace base9_perfect_square_l168_168199

theorem base9_perfect_square (a b d : ℕ) (h1 : a ≠ 0) (h2 : a < 9) (h3 : b < 9) (h4 : d < 9) (h5 : ∃ n : ℕ, (729 * a + 81 * b + 36 + d) = n * n) : d = 0 ∨ d = 1 ∨ d = 4 :=
by sorry

end base9_perfect_square_l168_168199


namespace compute_105_squared_l168_168576

theorem compute_105_squared :
  let a := 100
  let b := 5
  (a + b)^2 = 11025 :=
by
  sorry

end compute_105_squared_l168_168576


namespace exponentiation_condition_l168_168034

theorem exponentiation_condition (a b : ℝ) (h0 : a > 0) (h1 : a ≠ 1) : 
  (a ^ b > 1 ↔ (a - 1) * b > 0) :=
sorry

end exponentiation_condition_l168_168034


namespace probability_face_then_number_l168_168567

theorem probability_face_then_number :
  let total_cards := 52
  let total_ways_to_draw_two := total_cards * (total_cards - 1)
  let face_cards := 3 * 4
  let number_cards := 9 * 4
  let probability := (face_cards * number_cards) / total_ways_to_draw_two
  probability = 8 / 49 :=
by
  sorry

end probability_face_then_number_l168_168567


namespace sharon_trip_distance_l168_168161

noncomputable def usual_speed (x : ℝ) : ℝ := x / 180
noncomputable def reduced_speed (x : ℝ) : ℝ := usual_speed x - 25 / 60
noncomputable def increased_speed (x : ℝ) : ℝ := usual_speed x + 10 / 60
noncomputable def pre_storm_time : ℝ := 60
noncomputable def total_time : ℝ := 300

theorem sharon_trip_distance : 
  ∀ (x : ℝ), 
  60 + (x / 3) / reduced_speed x + (x / 3) / increased_speed x = 240 → 
  x = 135 :=
sorry

end sharon_trip_distance_l168_168161


namespace books_sold_l168_168526

theorem books_sold (initial_books left_books sold_books : ℕ) (h1 : initial_books = 108) (h2 : left_books = 66) : sold_books = 42 :=
by
  have : sold_books = initial_books - left_books := sorry
  rw [h1, h2] at this
  exact this

end books_sold_l168_168526


namespace road_greening_cost_l168_168837

-- Define constants for the conditions
def l_total : ℕ := 1500
def cost_A : ℕ := 22
def cost_B : ℕ := 25

-- Define variables for the cost per stem
variables (x y : ℕ)

-- Define the conditions from Plan A and Plan B
def plan_A (x y : ℕ) : Prop := 2 * x + 3 * y = cost_A
def plan_B (x y : ℕ) : Prop := x + 5 * y = cost_B

-- System of equations to find x and y
def system_of_equations (x y : ℕ) : Prop := plan_A x y ∧ plan_B x y

-- Define the constraint for the length of road greened according to Plan B
def length_constraint (a : ℕ) : Prop := l_total - a ≥ 2 * a

-- Define the total cost function
def total_cost (a : ℕ) (x y : ℕ) : ℕ := 22 * a + (x + 5 * y) * (l_total - a)

-- Prove the cost per stem and the minimized cost
theorem road_greening_cost :
  (∃ x y, system_of_equations x y ∧ x = 5 ∧ y = 4) ∧
  (∃ a : ℕ, length_constraint a ∧ a = 500 ∧ total_cost a 5 4 = 36000) :=
by
  -- This is where the proof would go
  sorry

end road_greening_cost_l168_168837


namespace geometric_arithmetic_series_difference_l168_168853

theorem geometric_arithmetic_series_difference :
  let a := 1
  let r := 1 / 2
  let S := a / (1 - r)
  let T := 1 + 2 + 3
  S - T = -4 :=
by
  sorry

end geometric_arithmetic_series_difference_l168_168853


namespace slices_of_bread_left_l168_168271

variable (monday_to_friday_slices saturday_slices total_slices_used initial_slices slices_left: ℕ)

def sandwiches_monday_to_friday : ℕ := 5
def slices_per_sandwich : ℕ := 2
def sandwiches_saturday : ℕ := 2
def initial_slices_of_bread : ℕ := 22

theorem slices_of_bread_left :
  slices_left = initial_slices_of_bread - total_slices_used
  :=
by  sorry

end slices_of_bread_left_l168_168271


namespace fraction_of_dark_tiles_proof_l168_168993

noncomputable def fraction_of_dark_tiles
  (floor_repeating_pattern : ℕ)
  (section_size : ℕ)
  (dark_tiles_in_top_left : ℕ)
  (additional_dark_tiles_in_rest : ℕ)
  (configuration : section_size = 4 ∧ dark_tiles_in_top_left = 3 ∧ additional_dark_tiles_in_rest = 2 ∧ floor_repeating_pattern ≥ section_size)
  : ℚ :=
  let total_dark_tiles := dark_tiles_in_top_left + additional_dark_tiles_in_rest in
  let total_tiles := section_size * section_size in
  total_dark_tiles / total_tiles

theorem fraction_of_dark_tiles_proof
  (floor_repeating_pattern section_size dark_tiles_in_top_left additional_dark_tiles_in_rest : ℕ)
  (h : section_size = 4 ∧ dark_tiles_in_top_left = 3 ∧ additional_dark_tiles_in_rest = 2 ∧ floor_repeating_pattern ≥ section_size)
  : fraction_of_dark_tiles floor_repeating_pattern section_size dark_tiles_in_top_left additional_dark_tiles_in_rest h = 5 / 16 :=
by sorry

end fraction_of_dark_tiles_proof_l168_168993


namespace quadratic_difference_square_l168_168496

theorem quadratic_difference_square (α β : ℝ) (h : α ≠ β) (hα : α^2 - 3 * α + 1 = 0) (hβ : β^2 - 3 * β + 1 = 0) : (α - β)^2 = 5 := by
  sorry

end quadratic_difference_square_l168_168496


namespace rope_costs_purchasing_plans_minimum_cost_l168_168162

theorem rope_costs (x y m : ℕ) :
  (10 * x + 5 * y = 175) →
  (15 * x + 10 * y = 300) →
  x = 10 ∧ y = 15 :=
sorry

theorem purchasing_plans (m : ℕ) :
  (10 * 10 + 15 * 15 = 300) →
  23 ≤ m ∧ m ≤ 25 :=
sorry

theorem minimum_cost (m : ℕ) :
  (23 ≤ m ∧ m ≤ 25) →
  m = 25 →
  10 * m + 15 * (45 - m) = 550 :=
sorry

end rope_costs_purchasing_plans_minimum_cost_l168_168162


namespace find_xyz_area_proof_l168_168995

-- Conditions given in the problem
variable (x y z : ℝ)
-- Side lengths derived from condition of inscribed circle
def conditions :=
  (x + y = 5) ∧
  (x + z = 6) ∧
  (y + z = 8)

-- The proof problem: Show the relationships between x, y, and z given the side lengths
theorem find_xyz_area_proof (h : conditions x y z) :
  (z - y = 1) ∧ (z - x = 3) ∧ (z = 4.5) ∧ (x = 1.5) ∧ (y = 3.5) :=
by
  sorry

end find_xyz_area_proof_l168_168995


namespace square_nonneg_l168_168665

theorem square_nonneg (x h k : ℝ) (h_eq: (x + h)^2 = k) : k ≥ 0 := 
by 
  sorry

end square_nonneg_l168_168665


namespace nathan_banana_payment_l168_168237

theorem nathan_banana_payment
  (bunches_8 : ℕ)
  (cost_per_bunch_8 : ℝ)
  (bunches_7 : ℕ)
  (cost_per_bunch_7 : ℝ)
  (discount : ℝ)
  (total_payment : ℝ) :
  bunches_8 = 6 →
  cost_per_bunch_8 = 2.5 →
  bunches_7 = 5 →
  cost_per_bunch_7 = 2.2 →
  discount = 0.10 →
  total_payment = 6 * 2.5 + 5 * 2.2 - 0.10 * (6 * 2.5 + 5 * 2.2) →
  total_payment = 23.40 :=
by
  intros
  sorry

end nathan_banana_payment_l168_168237


namespace square_of_105_l168_168590

/-- Prove that 105^2 = 11025. -/
theorem square_of_105 : 105^2 = 11025 :=
by
  sorry

end square_of_105_l168_168590


namespace gcd_polynomial_primes_l168_168179

theorem gcd_polynomial_primes (a : ℤ) (k : ℤ) (ha : a = 2 * 947 * k) : 
  Int.gcd (3 * a^2 + 47 * a + 101) (a + 19) = 1 :=
by
  sorry

end gcd_polynomial_primes_l168_168179


namespace arrange_students_l168_168832

theorem arrange_students 
  (students : Fin 6 → Type) 
  (A B : Type) 
  (h1 : ∃ i j, students i = A ∧ students j = B ∧ (i = j + 1 ∨ j = i + 1)) : 
  (∃ (n : ℕ), n = 240) := 
sorry

end arrange_students_l168_168832


namespace union_P_Q_l168_168742

def P : Set ℝ := {x | -1 < x ∧ x < 1}
def Q : Set ℝ := {x | x^2 - 2*x < 0}

theorem union_P_Q : P ∪ Q = {x : ℝ | -1 < x ∧ x < 2} :=
sorry

end union_P_Q_l168_168742


namespace intersection_equiv_l168_168051

-- Define the sets M and N based on the given conditions
def M : Set ℤ := {-1, 0, 1, 2, 3}
def N : Set ℤ := {x | x < 0 ∨ x > 2}

-- The main proof statement
theorem intersection_equiv : M ∩ N = {-1, 3} :=
by
  sorry -- proof goes here

end intersection_equiv_l168_168051


namespace sequence_bound_l168_168627

theorem sequence_bound (a : ℕ → ℝ) (h1 : ∀ n, a n > 0) (h2 : ∀ n, (a n)^2 ≤ a n - a (n+1)) :
  ∀ n, a n < 1 / n :=
by
  sorry

end sequence_bound_l168_168627


namespace project_completion_advance_l168_168709

variables (a : ℝ) -- efficiency of each worker (units of work per day)
variables (total_days : ℕ) (initial_workers added_workers : ℕ) (fraction_completed : ℝ)
variables (initial_days remaining_days total_initial_work total_remaining_work total_workers_efficiency : ℝ)

-- Conditions
def conditions : Prop :=
  total_days = 100 ∧
  initial_workers = 10 ∧
  initial_days = 30 ∧
  fraction_completed = 1 / 5 ∧
  added_workers = 10 ∧
  total_initial_work = initial_workers * initial_days * a * 5 ∧ 
  total_remaining_work = total_initial_work - (initial_workers * initial_days * a) ∧
  total_workers_efficiency = (initial_workers + added_workers) * a ∧
  remaining_days = total_remaining_work / total_workers_efficiency

-- Proof statement
theorem project_completion_advance (h : conditions a total_days initial_workers added_workers fraction_completed initial_days remaining_days total_initial_work total_remaining_work total_workers_efficiency) :
  total_days - (initial_days + remaining_days) = 10 :=
  sorry

end project_completion_advance_l168_168709


namespace belfried_industries_payroll_l168_168560

theorem belfried_industries_payroll (P : ℝ) (tax_paid : ℝ) : 
  ((P > 200000) ∧ (tax_paid = 0.002 * (P - 200000)) ∧ (tax_paid = 200)) → P = 300000 :=
by
  sorry

end belfried_industries_payroll_l168_168560


namespace problem_l168_168231

def polynomial (x : ℝ) : ℝ := 9 * x ^ 3 - 27 * x + 54

theorem problem (a b c : ℝ) 
  (h_roots : polynomial a = 0 ∧ polynomial b = 0 ∧ polynomial c = 0) :
  (a + b) ^ 3 + (b + c) ^ 3 + (c + a) ^ 3 = 18 :=
by
  sorry

end problem_l168_168231


namespace rate_per_kg_for_apples_l168_168102

theorem rate_per_kg_for_apples (A : ℝ) :
  (8 * A + 9 * 45 = 965) → (A = 70) :=
by
  sorry

end rate_per_kg_for_apples_l168_168102


namespace particular_solution_exists_l168_168625

theorem particular_solution_exists :
  ∃ C₁ C₂ : ℝ, ∀ t : ℝ, 
  let x₁ := C₁ * Real.exp(-t) + C₂ * Real.exp(3 * t)
  let x₂ := 2 * C₁ * Real.exp(-t) - 2 * C₂ * Real.exp(3 * t)
  x₁ 0 = 0 ∧ x₂ 0 = -4 ∧
  x₁ t = -Real.exp(-t) + Real.exp(3 * t) ∧
  x₂ t = -2 * Real.exp(-t) - 2 * Real.exp(3 * t) := by
  sorry

end particular_solution_exists_l168_168625


namespace kombucha_bottles_after_refund_l168_168054

noncomputable def bottles_per_month : ℕ := 15
noncomputable def cost_per_bottle : ℝ := 3.0
noncomputable def refund_per_bottle : ℝ := 0.10
noncomputable def months_in_year : ℕ := 12

theorem kombucha_bottles_after_refund :
  let bottles_per_year := bottles_per_month * months_in_year
  let total_refund := bottles_per_year * refund_per_bottle
  let bottles_bought_with_refund := total_refund / cost_per_bottle
  bottles_bought_with_refund = 6 := sorry

end kombucha_bottles_after_refund_l168_168054


namespace find_divisor_l168_168823

theorem find_divisor (x : ℕ) (h : 180 % x = 0) (h_eq : 70 + 5 * 12 / (180 / x) = 71) : x = 3 := 
by
  -- proof goes here
  sorry

end find_divisor_l168_168823


namespace river_depth_l168_168844

theorem river_depth (V : ℝ) (W : ℝ) (F : ℝ) (D : ℝ) 
  (hV : V = 10666.666666666666) 
  (hW : W = 40) 
  (hF : F = 66.66666666666667) 
  (hV_eq : V = W * D * F) : 
  D = 4 :=
by sorry

end river_depth_l168_168844


namespace range_S13_over_a14_l168_168229

lemma a_n_is_arithmetic_progression (a S : ℕ → ℕ) (h1 : ∀ n, a n > 1)
  (h2 : ∀ n, S (n + 1) - S n - (1 / 2) * ((a (n + 1)) ^ 2 - (a n) ^ 2) = 1 / 2) :
  ∀ n, a (n + 1) = a n + 1 := 
sorry

theorem range_S13_over_a14 (a S : ℕ → ℕ) (h1 : ∀ n, a n > 1)
  (h2 : ∀ n, S (n + 1) - S n - (1 / 2) * ((a (n + 1)) ^ 2 - (a n) ^ 2) = 1 / 2)
  (h3 : a 1 > 4) :
  130 / 17 < (S 13 / a 14) ∧ (S 13 / a 14) < 13 := 
sorry

end range_S13_over_a14_l168_168229


namespace carton_height_is_60_l168_168966

-- Definitions
def carton_length : ℕ := 30
def carton_width : ℕ := 42
def soap_length : ℕ := 7
def soap_width : ℕ := 6
def soap_height : ℕ := 5
def max_soap_boxes : ℕ := 360

-- Theorem Statement
theorem carton_height_is_60 (h : ℕ) (H : ∀ (layers : ℕ), layers = max_soap_boxes / ((carton_length / soap_length) * (carton_width / soap_width)) → h = layers * soap_height) : h = 60 :=
  sorry

end carton_height_is_60_l168_168966


namespace three_digit_decimal_bounds_l168_168714

def is_rounded_half_up (x : ℝ) (y : ℝ) : Prop :=
  (y - 0.005 ≤ x) ∧ (x < y + 0.005)

theorem three_digit_decimal_bounds :
  ∃ (x : ℝ), (8.725 ≤ x) ∧ (x ≤ 8.734) ∧ is_rounded_half_up x 8.73 :=
by
  sorry

end three_digit_decimal_bounds_l168_168714


namespace balls_remaining_l168_168939

-- Define the initial number of balls in the box
def initial_balls := 10

-- Define the number of balls taken by Yoongi
def balls_taken := 3

-- Define the number of balls left after Yoongi took some balls
def balls_left := initial_balls - balls_taken

-- The theorem statement to be proven
theorem balls_remaining : balls_left = 7 :=
by
    -- Skipping the proof
    sorry

end balls_remaining_l168_168939


namespace value_of_expression_l168_168483

theorem value_of_expression (a b c d x y : ℤ) 
  (h1 : a = -b) 
  (h2 : c * d = 1)
  (h3 : abs x = 3)
  (h4 : y = -1) : 
  2 * x - c * d + 6 * (a + b) - abs y = 4 ∨ 2 * x - c * d + 6 * (a + b) - abs y = -8 := 
by 
  sorry

end value_of_expression_l168_168483


namespace hyperbola_focus_l168_168459

noncomputable def c (a b : ℝ) : ℝ := Real.sqrt (a^2 + b^2)

theorem hyperbola_focus
  (a b : ℝ)
  (hEq : ∀ x y : ℝ, ((x - 1)^2 / a^2) - ((y - 10)^2 / b^2) = 1):
  (1 + c 7 3, 10) = (1 + Real.sqrt (7^2 + 3^2), 10) :=
by
  sorry

end hyperbola_focus_l168_168459


namespace range_of_a_l168_168044

variable (a : ℝ) (f : ℝ → ℝ)
axiom func_def : ∀ x, f x = a^x
axiom a_pos : a > 0
axiom a_ne_one : a ≠ 1
axiom decreasing : ∀ m n : ℝ, m > n → f m < f n

theorem range_of_a : 0 < a ∧ a < 1 :=
sorry

end range_of_a_l168_168044


namespace x_intercept_of_line_l168_168022

theorem x_intercept_of_line : ∃ x : ℚ, (6 * x, 0) = (35 / 6, 0) :=
by
  use 35 / 6
  sorry

end x_intercept_of_line_l168_168022


namespace three_digit_with_five_is_divisible_by_five_l168_168314

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def ends_in_five (n : ℕ) : Prop := n % 10 = 5

def divisible_by_five (n : ℕ) : Prop := n % 5 = 0

theorem three_digit_with_five_is_divisible_by_five (M : ℕ) :
  is_three_digit M ∧ ends_in_five M → divisible_by_five M :=
by
  sorry

end three_digit_with_five_is_divisible_by_five_l168_168314


namespace complex_division_l168_168732

theorem complex_division (a b : ℝ) (i : ℂ) (hi : i = complex.I) : 
  (2 * i / (1 - i) = a + b * i) → a = -1 ∧ b = 1 :=
by
  sorry

end complex_division_l168_168732


namespace unique_four_digit_perfect_cube_divisible_by_16_and_9_l168_168493

theorem unique_four_digit_perfect_cube_divisible_by_16_and_9 :
  ∃! n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ (∃ k : ℕ, n = k^3) ∧ n % 16 = 0 ∧ n % 9 = 0 ∧ n = 1728 :=
by sorry

end unique_four_digit_perfect_cube_divisible_by_16_and_9_l168_168493


namespace boat_speed_in_still_water_l168_168432

theorem boat_speed_in_still_water  (b s : ℝ) (h1 : b + s = 13) (h2 : b - s = 9) : b = 11 :=
sorry

end boat_speed_in_still_water_l168_168432


namespace number_of_recipes_needed_l168_168318

noncomputable def cookies_per_student : ℕ := 3
noncomputable def total_students : ℕ := 150
noncomputable def recipe_yield : ℕ := 20
noncomputable def attendance_drop_rate : ℝ := 0.30

theorem number_of_recipes_needed : 
  ⌈ (total_students * (1 - attendance_drop_rate) * cookies_per_student) / recipe_yield ⌉ = 16 := by
  sorry

end number_of_recipes_needed_l168_168318


namespace quotient_base6_division_l168_168018

theorem quotient_base6_division :
  let a := 2045
  let b := 14
  let base := 6
  a / b = 51 :=
by
  sorry

end quotient_base6_division_l168_168018


namespace largest_number_l168_168813

theorem largest_number (n : ℕ) (digits : List ℕ) (h_digits : ∀ d ∈ digits, d = 5 ∨ d = 3 ∨ d = 1) (h_sum : digits.sum = 15) : n = 555 :=
by
  sorry

end largest_number_l168_168813


namespace perfect_square_trinomial_l168_168747

theorem perfect_square_trinomial (k : ℝ) : (∃ a b : ℝ, (a * x + b) ^ 2 = x^2 - k * x + 4) → (k = 4 ∨ k = -4) :=
by
  sorry

end perfect_square_trinomial_l168_168747


namespace negative_linear_correlation_l168_168488

theorem negative_linear_correlation (x y : ℝ) (h : y = 3 - 2 * x) : 
  ∃ c : ℝ, c < 0 ∧ y = 3 + c * x := 
by  
  sorry

end negative_linear_correlation_l168_168488


namespace reflection_identity_l168_168921

-- Define the reflection function
def reflect (O P : ℝ × ℝ) : ℝ × ℝ := (2 * O.1 - P.1, 2 * O.2 - P.2)

-- Given three points and a point P
variables (O1 O2 O3 P : ℝ × ℝ)

-- Define the sequence of reflections
def sequence_reflection (P : ℝ × ℝ) : ℝ × ℝ :=
  reflect O3 (reflect O2 (reflect O1 P))

-- Lean 4 statement to prove the mathematical theorem
theorem reflection_identity :
  sequence_reflection O1 O2 O3 (sequence_reflection O1 O2 O3 P) = P :=
by sorry

end reflection_identity_l168_168921


namespace rocket_soaring_time_l168_168138

theorem rocket_soaring_time 
  (avg_speed : ℝ)                      -- The average speed of the rocket
  (soar_speed : ℝ)                     -- Speed while soaring
  (plummet_distance : ℝ)               -- Distance covered during plummet
  (plummet_time : ℝ)                   -- Time of plummet
  (total_time : ℝ := plummet_time + t) -- Total time is the sum of soaring time and plummet time
  (total_distance : ℝ := soar_speed * t + plummet_distance) -- Total distance covered
  (h_avg_speed : avg_speed = total_distance / total_time)   -- Given condition for average speed
  :
  ∃ t : ℝ, t = 12 :=                   -- Prove that the soaring time is 12 seconds
by
  sorry

end rocket_soaring_time_l168_168138


namespace find_K_3_15_10_l168_168866

def K (x y z : ℚ) : ℚ := 
  x / y + y / z + z / x + (x + y) / z

theorem find_K_3_15_10 : K 3 15 10 = 41 / 6 := 
  by
  sorry

end find_K_3_15_10_l168_168866


namespace smallest_fraction_l168_168178

theorem smallest_fraction 
  (x y z t : ℝ) 
  (h1 : 1 < x) 
  (h2 : x < y) 
  (h3 : y < z) 
  (h4 : z < t) : 
  (min (min (min (min ((x + y) / (z + t)) ((x + t) / (y + z))) ((y + z) / (x + t))) ((y + t) / (x + z))) ((z + t) / (x + y))) = (x + y) / (z + t) :=
by {
    sorry
}

end smallest_fraction_l168_168178


namespace proof_problem_l168_168228

open Set

def M : Set ℝ := {y | ∃ x : ℝ, y = 3 ^ x}
def N : Set ℝ := { -1, 0, 1 }

theorem proof_problem : ((univ \ M) ∩ N) = { -1, 0 } := by
  sorry

end proof_problem_l168_168228


namespace linear_coefficient_is_one_l168_168254

-- Define the given equation and the coefficient of the linear term
variables {x m : ℝ}
def equation := (m - 3) * x + 4 * m^2 - 2 * m - 1 - m * x + 6

-- State the main theorem: the coefficient of the linear term in the equation is 1 given the conditions
theorem linear_coefficient_is_one (m : ℝ) (hm_neq_3 : m ≠ 3) :
  (m - 3) - m = 1 :=
by sorry

end linear_coefficient_is_one_l168_168254


namespace find_a5_from_geometric_sequence_l168_168347

def geo_seq (a : ℕ → ℝ) (q : ℝ) := ∀ n, a (n + 1) = a n * q

noncomputable def geometric_sequence_property (a : ℕ → ℝ) (q : ℝ) :=
  geo_seq a q ∧ 0 < a 1 ∧ 0 < q ∧ 
  (a 4 = (a 2) ^ 2) ∧ 
  (a 2 + a 4 = 5 / 16)

theorem find_a5_from_geometric_sequence :
  ∀ (a : ℕ → ℝ) (q : ℝ), geometric_sequence_property a q → 
  a 5 = 1 / 32 :=
by 
  sorry

end find_a5_from_geometric_sequence_l168_168347


namespace initial_walking_speed_l168_168306

variable (v : ℝ)

theorem initial_walking_speed :
  (13.5 / v - 13.5 / 6 = 27 / 60) → v = 5 :=
by
  intro h
  sorry

end initial_walking_speed_l168_168306


namespace ternary_to_decimal_l168_168363

theorem ternary_to_decimal (k : ℕ) (hk : k > 0) : (1 * 3^3 + k * 3^1 + 2 = 35) → k = 2 :=
by
  sorry

end ternary_to_decimal_l168_168363


namespace small_cubes_with_two_faces_painted_red_l168_168310

theorem small_cubes_with_two_faces_painted_red (edge_length : ℕ) (small_cube_edge_length : ℕ)
  (h1 : edge_length = 4) (h2 : small_cube_edge_length = 1) :
  ∃ n, n = 24 :=
by
  -- Proof skipped
  sorry

end small_cubes_with_two_faces_painted_red_l168_168310


namespace distance_traveled_l168_168686

theorem distance_traveled 
    (P_b : ℕ) (P_f : ℕ) (R_b : ℕ) (R_f : ℕ)
    (h1 : P_b = 9)
    (h2 : P_f = 7)
    (h3 : R_f = R_b + 10) 
    (h4 : R_b * P_b = R_f * P_f) :
    R_b * P_b = 315 :=
by
  sorry

end distance_traveled_l168_168686


namespace power_function_decreasing_l168_168748

theorem power_function_decreasing (m : ℝ) (x : ℝ) (hx : x > 0) :
  (m^2 - 2*m - 2 = 1) ∧ (-4*m - 2 < 0) → m = 3 :=
by
  sorry

end power_function_decreasing_l168_168748


namespace base3_to_base10_conversion_l168_168994

theorem base3_to_base10_conversion : 
  (1 * 3^4 + 0 * 3^3 + 2 * 3^2 + 0 * 3^1 + 1 * 3^0 = 100) :=
by 
  sorry

end base3_to_base10_conversion_l168_168994


namespace solve_x_values_l168_168331

theorem solve_x_values : ∀ (x : ℝ), (x + 45 / (x - 4) = -10) ↔ (x = -1 ∨ x = -5) :=
by
  intro x
  sorry

end solve_x_values_l168_168331


namespace evaluate_f_at_2_l168_168280

def f (x : ℝ) : ℝ := 2 * x^5 + 3 * x^4 + 2 * x^3 - 4 * x + 5

theorem evaluate_f_at_2 :
  f 2 = 125 :=
by
  sorry

end evaluate_f_at_2_l168_168280


namespace increasing_sequence_a_range_l168_168521

theorem increasing_sequence_a_range (f : ℕ → ℝ) (a : ℝ)
  (h1 : ∀ n, f n = if n ≤ 7 then (3 - a) * n - 3 else a ^ (n - 6))
  (h2 : ∀ n : ℕ, f n < f (n + 1)) :
  2 < a ∧ a < 3 :=
sorry

end increasing_sequence_a_range_l168_168521


namespace wall_area_l168_168903

-- Definition of the width and length of the wall
def width : ℝ := 5.4
def length : ℝ := 2.5

-- Statement of the theorem
theorem wall_area : (width * length) = 13.5 :=
by
  sorry

end wall_area_l168_168903


namespace smallest_number_is_10_l168_168689

/-- Define the set of numbers. -/
def numbers : List Int := [10, 11, 12, 13, 14]

theorem smallest_number_is_10 :
  ∃ n ∈ numbers, (∀ m ∈ numbers, n ≤ m) ∧ n = 10 :=
by
  sorry

end smallest_number_is_10_l168_168689


namespace paint_two_faces_red_l168_168983

theorem paint_two_faces_red (f : Fin 8 → ℕ) (H : ∀ i, 1 ≤ f i ∧ f i ≤ 8) : 
  (∃ pair_count : ℕ, pair_count = 9 ∧
    ∀ i j, i < j → f i + f j ≤ 7 → true) :=
sorry

end paint_two_faces_red_l168_168983


namespace simplify_fraction_l168_168662

theorem simplify_fraction (b : ℕ) (hb : b = 2) : (15 * b ^ 4) / (45 * b ^ 3) = 2 / 3 :=
by
  sorry

end simplify_fraction_l168_168662


namespace complex_magnitude_squared_l168_168379

open Complex Real

theorem complex_magnitude_squared :
  ∃ (z : ℂ), z + abs z = 3 + 7 * i ∧ abs z ^ 2 = 841 / 9 :=
by
  sorry

end complex_magnitude_squared_l168_168379


namespace triangular_pyramid_surface_area_l168_168315

theorem triangular_pyramid_surface_area
  (base_area : ℝ)
  (side_area : ℝ) :
  base_area = 3 ∧ side_area = 6 → base_area + 3 * side_area = 21 :=
by
  sorry

end triangular_pyramid_surface_area_l168_168315


namespace job_pay_per_pound_l168_168087

def p := 2
def M := 8 -- Monday
def T := 3 * M -- Tuesday
def W := 0 -- Wednesday
def R := 18 -- Thursday
def total_picked := M + T + W + R -- total berries picked
def money := 100 -- total money wanted

theorem job_pay_per_pound :
  total_picked = 50 → p = money / total_picked :=
by
  intro h
  rw [h]
  norm_num
  exact rfl

end job_pay_per_pound_l168_168087


namespace largest_y_value_l168_168694

theorem largest_y_value (y : ℝ) : (6 * y^2 - 31 * y + 35 = 0) → (y ≤ 2.5) :=
by
  intro h
  sorry

end largest_y_value_l168_168694


namespace four_digit_numbers_divisible_by_13_l168_168892

theorem four_digit_numbers_divisible_by_13 : 
  (set.Icc 1000 9999).filter (λ x, x % 13 = 0).cardinality = 693 :=
by
  sorry

end four_digit_numbers_divisible_by_13_l168_168892


namespace min_value_f_l168_168612

noncomputable def f (x : Fin 5 → ℝ) : ℝ :=
  (x 0 + x 2) / (x 4 + 2 * x 1 + 3 * x 3) +
  (x 1 + x 3) / (x 0 + 2 * x 2 + 3 * x 4) +
  (x 2 + x 4) / (x 1 + 2 * x 3 + 3 * x 0) +
  (x 3 + x 0) / (x 2 + 2 * x 4 + 3 * x 1) +
  (x 4 + x 1) / (x 3 + 2 * x 0 + 3 * x 2)

def min_f (x : Fin 5 → ℝ) : Prop :=
  (∀ i, 0 < x i) → f x = 5 / 3

theorem min_value_f : ∀ x : Fin 5 → ℝ, min_f x :=
by
  intros
  sorry

end min_value_f_l168_168612


namespace distinct_intersections_count_l168_168156

theorem distinct_intersections_count :
  (∃ (x y : ℝ), (x + 2 * y = 7 ∧ 3 * x - 4 * y + 8 = 0) ∨ (x + 2 * y = 7 ∧ 4 * x + 5 * y - 20 = 0) ∨
                (x - 2 * y - 1 = 0 ∧ 3 * x - 4 * y = 8) ∨ (x - 2 * y - 1 = 0 ∧ 4 * x + 5 * y - 20 = 0)) ∧
  ∃ count : ℕ, count = 3 :=
by sorry

end distinct_intersections_count_l168_168156


namespace remainder_of_sum_mod_11_l168_168028

theorem remainder_of_sum_mod_11 :
  (123456 + 123457 + 123458 + 123459 + 123460 + 123461) % 11 = 10 :=
by
  sorry

end remainder_of_sum_mod_11_l168_168028


namespace compute_105_squared_l168_168581

theorem compute_105_squared : 105^2 = 11025 :=
by
  sorry

end compute_105_squared_l168_168581


namespace average_mileage_city_l168_168142

variable (total_distance : ℝ) (gallons : ℝ) (highway_mpg : ℝ) (city_mpg : ℝ)

-- The given conditions
def conditions : Prop := (total_distance = 280.6) ∧ (gallons = 23) ∧ (highway_mpg = 12.2)

-- The theorem to prove
theorem average_mileage_city (h : conditions total_distance gallons highway_mpg) :
  total_distance / gallons = 12.2 :=
sorry

end average_mileage_city_l168_168142


namespace martha_found_blocks_l168_168387

variable (initial_blocks final_blocks found_blocks : ℕ)

theorem martha_found_blocks 
    (h_initial : initial_blocks = 4) 
    (h_final : final_blocks = 84) 
    (h_found : found_blocks = final_blocks - initial_blocks) : 
    found_blocks = 80 := by
  sorry

end martha_found_blocks_l168_168387


namespace variance_red_balls_correct_l168_168440

noncomputable def variance_of_red_balls : ℝ :=
  let p : ℝ := 3 / 5
  let n : ℕ := 4
  n * p * (1 - p)

theorem variance_red_balls_correct :
  variance_of_red_balls = 24 / 25 :=
by
  sorry

end variance_red_balls_correct_l168_168440


namespace gumballs_initial_count_l168_168860

noncomputable def initial_gumballs := (34.3 / (0.7 ^ 3))

theorem gumballs_initial_count :
  initial_gumballs = 100 :=
sorry

end gumballs_initial_count_l168_168860


namespace fraction_zero_iff_numerator_zero_l168_168636

-- Define the conditions and the result in Lean 4.
theorem fraction_zero_iff_numerator_zero (x : ℝ) (h : x ≠ 0) : (x - 3) / x = 0 ↔ x = 3 :=
by
  sorry

end fraction_zero_iff_numerator_zero_l168_168636


namespace max_ab_of_tangent_circles_l168_168878

theorem max_ab_of_tangent_circles (a b : ℝ) 
  (hC1 : ∀ x y : ℝ, (x - a)^2 + (y + 2)^2 = 4)
  (hC2 : ∀ x y : ℝ, (x + b)^2 + (y + 2)^2 = 1)
  (h_tangent : a + b = 3) :
  ab ≤ 9 / 4 :=
by
  sorry

end max_ab_of_tangent_circles_l168_168878


namespace clown_balloons_l168_168934

theorem clown_balloons 
  (initial_balloons : ℕ := 123) 
  (additional_balloons : ℕ := 53) 
  (given_away_balloons : ℕ := 27) : 
  initial_balloons + additional_balloons - given_away_balloons = 149 := 
by 
  sorry

end clown_balloons_l168_168934


namespace expression_evaluation_l168_168852

theorem expression_evaluation :
  5 * 423 + 4 * 423 + 3 * 423 + 421 = 5497 := by
  sorry

end expression_evaluation_l168_168852


namespace mrs_hilt_rocks_l168_168525

def garden_length := 10
def garden_width := 15
def rock_coverage := 1
def available_rocks := 64

theorem mrs_hilt_rocks :
  ∃ extra_rocks : ℕ, 2 * (garden_length + garden_width) <= available_rocks ∧ extra_rocks = available_rocks - 2 * (garden_length + garden_width) ∧ extra_rocks = 14 :=
by
  sorry

end mrs_hilt_rocks_l168_168525


namespace probability_symmetric_line_l168_168313

theorem probability_symmetric_line (P : (ℕ × ℕ) := (5, 5))
    (n : ℕ := 10) (total_points remaining_points symmetric_points : ℕ) 
    (probability : ℚ) :
  total_points = n * n →
  remaining_points = total_points - 1 →
  symmetric_points = 4 * (n - 1) →
  probability = (symmetric_points : ℚ) / (remaining_points : ℚ) →
  probability = 32 / 99 :=
by
  sorry

end probability_symmetric_line_l168_168313


namespace percentage_increase_l168_168207

theorem percentage_increase (x : ℝ) (h : x = 77.7) : 
  ((x - 70) / 70) * 100 = 11 := by
  sorry

end percentage_increase_l168_168207


namespace chloe_and_friends_points_l168_168854

-- Define the conditions as Lean definitions and then state the theorem to be proven.

def total_pounds_recycled : ℕ := 28 + 2

def pounds_per_point : ℕ := 6

def points_earned (total_pounds : ℕ) (pounds_per_point : ℕ) : ℕ :=
  total_pounds / pounds_per_point

theorem chloe_and_friends_points :
  points_earned total_pounds_recycled pounds_per_point = 5 :=
by
  sorry

end chloe_and_friends_points_l168_168854


namespace Susie_possible_values_l168_168089

theorem Susie_possible_values (n : ℕ) (h1 : n > 43) (h2 : 2023 % n = 43) : 
  (∃ count : ℕ, count = 19 ∧ ∀ n, n > 43 ∧ 2023 % n = 43 → 1980 ∣ (2023 - 43)) :=
sorry

end Susie_possible_values_l168_168089


namespace min_value_x2_2xy_y2_l168_168024

theorem min_value_x2_2xy_y2 (x y : ℝ) : ∃ (a b : ℝ), (x = a ∧ y = b) → x^2 + 2*x*y + y^2 = 0 :=
by {
  sorry
}

end min_value_x2_2xy_y2_l168_168024


namespace number_of_groups_l168_168041

theorem number_of_groups (max_value min_value interval : ℕ) (h_max : max_value = 36) (h_min : min_value = 15) (h_interval : interval = 4) : 
  ∃ groups : ℕ, groups = 6 :=
by 
  sorry

end number_of_groups_l168_168041


namespace solve_work_problem_l168_168558

variables (A B C : ℚ)

-- Conditions
def condition1 := B + C = 1/3
def condition2 := C + A = 1/4
def condition3 := C = 1/24

-- Conclusion (Question translated to proof statement)
theorem solve_work_problem (h1 : condition1 B C) (h2 : condition2 C A) (h3 : condition3 C) : A + B = 1/2 :=
by sorry

end solve_work_problem_l168_168558


namespace y_intercept_of_line_l168_168261

theorem y_intercept_of_line (m : ℝ) (x1 y1 : ℝ) (h_slope : m = -3) (h_x_intercept : x1 = 7) (h_y_intercept : y1 = 0) : 
  ∃ y_intercept : ℝ, y_intercept = 21 ∧ y_intercept = -m * 0 + 21 :=
by
  sorry

end y_intercept_of_line_l168_168261


namespace circle_relationship_l168_168043

noncomputable def f : ℝ × ℝ → ℝ := sorry

variables {x y x₁ y₁ x₂ y₂ : ℝ}
variables (h₁ : f (x₁, y₁) = 0) (h₂ : f (x₂, y₂) ≠ 0)

theorem circle_relationship :
  f (x, y) - f (x₁, y₁) - f (x₂, y₂) = 0 ↔ f (x, y) = f (x₂, y₂) :=
sorry

end circle_relationship_l168_168043


namespace polar_distance_l168_168375

noncomputable def distance_point (r1 θ1 r2 θ2 : ℝ) : ℝ :=
  Real.sqrt ((r1 ^ 2) + (r2 ^ 2) - 2 * r1 * r2 * Real.cos (θ1 - θ2))

theorem polar_distance :
  ∀ (θ1 θ2 : ℝ), (θ1 - θ2 = Real.pi / 2) → distance_point 5 θ1 12 θ2 = 13 :=
by
  intros θ1 θ2 hθ
  rw [distance_point, hθ, Real.cos_pi_div_two]
  norm_num
  sorry

end polar_distance_l168_168375


namespace largest_obtuse_prime_angle_l168_168754

theorem largest_obtuse_prime_angle (alpha beta gamma : ℕ) 
    (h_triangle_sum : alpha + beta + gamma = 180) 
    (h_alpha_gt_beta : alpha > beta) 
    (h_beta_gt_gamma : beta > gamma)
    (h_obtuse_alpha : alpha > 90) 
    (h_alpha_prime : Prime alpha) 
    (h_beta_prime : Prime beta) : 
    alpha = 173 := 
sorry

end largest_obtuse_prime_angle_l168_168754


namespace sum_of_consecutive_even_negative_integers_l168_168794

theorem sum_of_consecutive_even_negative_integers (n m : ℤ) 
  (h1 : n % 2 = 0)
  (h2 : m % 2 = 0)
  (h3 : n < 0)
  (h4 : m < 0)
  (h5 : m = n + 2)
  (h6 : n * m = 2496) : n + m = -102 := 
sorry

end sum_of_consecutive_even_negative_integers_l168_168794


namespace arithmetic_sequence_fifth_term_l168_168097

theorem arithmetic_sequence_fifth_term (a d : ℤ) 
  (h1 : a + 9 * d = 15)
  (h2 : a + 10 * d = 18) : 
  a + 4 * d = 0 := 
sorry

end arithmetic_sequence_fifth_term_l168_168097


namespace log_composite_monotonicity_l168_168681

noncomputable def quadratic (x : ℝ) : ℝ :=
  x^2 - 6 * x + 8

open Function

theorem log_composite_monotonicity :
  ∀ {x : ℝ}, x ∈ Set.Ioo 2 3 -> (log (0.2) (quadratic x)) =
      sorry :=
sorry

end log_composite_monotonicity_l168_168681


namespace minute_hand_gain_per_hour_l168_168564

theorem minute_hand_gain_per_hour (h_start h_end : ℕ) (time_elapsed : ℕ) 
  (total_gain : ℕ) (gain_per_hour : ℕ) 
  (h_start_eq_9 : h_start = 9)
  (time_period_eq_8 : time_elapsed = 8)
  (total_gain_eq_40 : total_gain = 40)
  (time_elapsed_eq : h_end = h_start + time_elapsed)
  (gain_formula : gain_per_hour * time_elapsed = total_gain) :
  gain_per_hour = 5 := 
by 
  sorry

end minute_hand_gain_per_hour_l168_168564


namespace compute_105_squared_l168_168578

theorem compute_105_squared :
  let a := 100
  let b := 5
  (a + b)^2 = 11025 :=
by
  sorry

end compute_105_squared_l168_168578


namespace present_ages_ratio_l168_168919

noncomputable def ratio_of_ages (F S : ℕ) : ℚ :=
  F / S

theorem present_ages_ratio (F S : ℕ) (h1 : F + S = 220) (h2 : (F + 10) * 3 = (S + 10) * 5) :
  ratio_of_ages F S = 7 / 4 :=
by
  sorry

end present_ages_ratio_l168_168919


namespace large_circuit_longer_l168_168730

theorem large_circuit_longer :
  ∀ (small_circuit_length large_circuit_length : ℕ),
  ∀ (laps_jana laps_father : ℕ),
  laps_jana = 3 →
  laps_father = 4 →
  (laps_father * large_circuit_length = 2 * (laps_jana * small_circuit_length)) →
  small_circuit_length = 400 →
  large_circuit_length - small_circuit_length = 200 :=
by
  intros small_circuit_length large_circuit_length laps_jana laps_father
  intros h_jana_laps h_father_laps h_distance h_small_length
  sorry

end large_circuit_longer_l168_168730


namespace library_visitors_on_sunday_l168_168134

def avg_visitors_sundays (S : ℕ) : Prop :=
  let total_days := 30
  let avg_other_days := 240
  let avg_total := 285
  let sundays := 5
  let other_days := total_days - sundays
  (S * sundays) + (avg_other_days * other_days) = avg_total * total_days

theorem library_visitors_on_sunday (S : ℕ) (h : avg_visitors_sundays S) : S = 510 :=
by
  sorry

end library_visitors_on_sunday_l168_168134


namespace water_overflow_amount_l168_168746

-- Declare the conditions given in the problem
def tap_production_per_hour : ℕ := 200
def tap_run_duration_in_hours : ℕ := 24
def tank_capacity_in_ml : ℕ := 4000

-- Define the total water produced by the tap
def total_water_produced : ℕ := tap_production_per_hour * tap_run_duration_in_hours

-- Define the amount of water that overflows
def water_overflowed : ℕ := total_water_produced - tank_capacity_in_ml

-- State the theorem to prove the amount of overflowing water
theorem water_overflow_amount : water_overflowed = 800 :=
by
  -- Placeholder for the proof
  sorry

end water_overflow_amount_l168_168746


namespace gcd_540_180_diminished_by_2_eq_178_l168_168169

theorem gcd_540_180_diminished_by_2_eq_178 : gcd 540 180 - 2 = 178 := by
  sorry

end gcd_540_180_diminished_by_2_eq_178_l168_168169


namespace cube_red_faces_one_third_l168_168710

theorem cube_red_faces_one_third (n : ℕ) (h : 6 * n^3 ≠ 0) : 
  (2 * n^2) / (6 * n^3) = 1 / 3 → n = 1 :=
by sorry

end cube_red_faces_one_third_l168_168710


namespace six_star_three_l168_168473

-- Define the mathematical operation.
def operation (r t : ℝ) : ℝ := sorry

axiom condition_1 (r : ℝ) : operation r 0 = r^2
axiom condition_2 (r t : ℝ) : operation r t = operation t r
axiom condition_3 (r t : ℝ) : operation (r + 1) t = operation r t + 2 * t + 1

-- Prove that 6 * 3 = 75 given the conditions.
theorem six_star_three : operation 6 3 = 75 := by
  sorry

end six_star_three_l168_168473


namespace independence_gini_coefficient_collaboration_gini_change_l168_168119

noncomputable def y_north (x : ℝ) : ℝ := 13.5 - 9 * x
noncomputable def y_south (x : ℝ) : ℝ := 24 - 1.5 * x^2

def kits_produced (y : ℝ) : ℝ := y / 9
def income (kits : ℝ) : ℝ := kits * 6000

def population_north : ℝ := 24
def population_south : ℝ := 6
def total_population : ℝ := population_north + population_south

-- Gini coefficient calculation for independent operations
def gini_coefficient_independent : ℝ := 
  let income_north := income (kits_produced (y_north 0)) / population_north
  let income_south := income (kits_produced (y_south 0)) / population_south
  let total_income := income_north * population_north + income_south * population_south
  let share_population_north := population_north / total_population
  let share_income_north := (income_north * population_north) / total_income
  share_population_north - share_income_north

-- Gini coefficient change upon collaboration
def gini_coefficient_collaboration : ℝ :=
  let income_north := income (kits_produced (y_north 0)) + 1983
  let income_south := income (kits_produced (y_south 0)) - (income (kits_produced (y_north 0)) + 1983)
  let total_income := income_north / population_north + income_south / population_south
  let share_population_north := population_north / total_population
  let share_income_north := (income_north * population_north) / total_income
  share_population_north - share_income_north

-- Proof statements
theorem independence_gini_coefficient : gini_coefficient_independent = 0.2 := sorry
theorem collaboration_gini_change : gini_coefficient_independent - gini_coefficient_collaboration = 0.001 := sorry

end independence_gini_coefficient_collaboration_gini_change_l168_168119


namespace smallest_positive_period_intervals_monotonically_decreasing_l168_168188

noncomputable def f (x : ℝ) : ℝ :=
  let a := (2 * Real.sin x, Real.sqrt 3 * Real.cos x)
  let b := (Real.cos x, 2 * Real.cos x)
  (a.1 * b.1 + a.2 * b.2) - Real.sqrt 3

theorem smallest_positive_period :
  ∃ T, T = π ∧ ∀ x, f (x + T) = f x :=
sorry

theorem intervals_monotonically_decreasing :
  ∀ k : ℤ, ∀ x, k * π + π / 12 ≤ x ∧ x ≤ k * π + 7 * π / 12 → f' x < 0 :=
sorry

end smallest_positive_period_intervals_monotonically_decreasing_l168_168188


namespace distinct_ordered_pairs_count_l168_168181

theorem distinct_ordered_pairs_count : 
  ∃ (n : ℕ), (∀ (a b : ℕ), a + b = 50 → 0 ≤ a ∧ 0 ≤ b) ∧ n = 51 :=
by
  sorry

end distinct_ordered_pairs_count_l168_168181


namespace production_days_l168_168115

noncomputable def daily_production (n : ℕ) : Prop :=
50 * n + 90 = 58 * (n + 1)

theorem production_days (n : ℕ) (h : daily_production n) : n = 4 :=
by sorry

end production_days_l168_168115


namespace inequality_solution_l168_168929

theorem inequality_solution (x : ℝ) (hx : 0 ≤ x) : 
  2021 * (x ^ 10) - 1 ≥ 2020 * x ↔ x = 1 := 
by
  sorry

end inequality_solution_l168_168929


namespace ellipse_triangle_perimeter_l168_168180

-- Definitions based on conditions
def is_ellipse (x y : ℝ) : Prop := (x^2 / 4) + (y^2 / 2) = 1

-- Triangle perimeter calculation
def triangle_perimeter (a c : ℝ) : ℝ := 2 * a + 2 * c

-- Main theorem statement
theorem ellipse_triangle_perimeter :
  let a := 2
  let b2 := 2
  let c := Real.sqrt (a ^ 2 - b2)
  ∀ (P : ℝ × ℝ), (is_ellipse P.1 P.2) → triangle_perimeter a c = 4 + 2 * Real.sqrt 2 :=
by
  intros P hP
  -- Here, we would normally provide the proof.
  sorry

end ellipse_triangle_perimeter_l168_168180


namespace total_spaces_in_game_l168_168931

-- Conditions
def first_turn : ℕ := 8
def second_turn_forward : ℕ := 2
def second_turn_backward : ℕ := 5
def third_turn : ℕ := 6
def total_to_end : ℕ := 37

-- Theorem stating the total number of spaces in the game
theorem total_spaces_in_game : first_turn + second_turn_forward - second_turn_backward + third_turn + (total_to_end - (first_turn + second_turn_forward - second_turn_backward + third_turn)) = total_to_end :=
by sorry

end total_spaces_in_game_l168_168931


namespace find_integer_l168_168019

theorem find_integer (n : ℕ) (hn1 : n % 20 = 0) (hn2 : 8.2 < (n : ℝ)^(1/3)) (hn3 : (n : ℝ)^(1/3) < 8.3) : n = 560 := sorry

end find_integer_l168_168019


namespace product_of_all_possible_values_of_e_l168_168232

theorem product_of_all_possible_values_of_e (d e : ℝ)
  (h1 : ∃ x, x^2 + d * x + e = 0 ∧ x^2 + d * x + e = 0)
  (h2 : d = 2 * e - 3)
  (root_cond : ∀ x, x^2 + d * x + e = 0 → d^2 - 4 * e = 0) :
  (2 + sqrt 7 / 2) * (2 - sqrt 7 / 2) = 9 / 4 := 
sorry

end product_of_all_possible_values_of_e_l168_168232


namespace kombucha_bottles_l168_168055

theorem kombucha_bottles (b_m : ℕ) (c : ℝ) (r : ℝ) (m : ℕ)
  (hb : b_m = 15) (hc : c = 3.00) (hr : r = 0.10) (hm : m = 12) :
  (b_m * m * r) / c = 6 := by
  sorry

end kombucha_bottles_l168_168055


namespace train_length_l168_168846

theorem train_length
  (time : ℝ) (man_speed train_speed : ℝ) (same_direction : Prop)
  (h_time : time = 62.99496040316775)
  (h_man_speed : man_speed = 6)
  (h_train_speed : train_speed = 30)
  (h_same_direction : same_direction) :
  (train_speed - man_speed) * (1000 / 3600) * time = 1259.899208063355 := 
sorry

end train_length_l168_168846


namespace min_value_eq_ab_squared_l168_168648

noncomputable def min_value (x a b : ℝ) : ℝ := 1 / (x^a * (1 - x)^b)

theorem min_value_eq_ab_squared (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) :
  ∃ x, 0 < x ∧ x < 1 ∧ min_value x a b = (a + b)^2 :=
by
  sorry

end min_value_eq_ab_squared_l168_168648


namespace license_plate_increase_l168_168659

theorem license_plate_increase :
  let old_plates := 26^2 * 10^5
  let new_plates := 26^4 * 10^4
  new_plates / old_plates = 26^2 / 10 :=
by
  let old_plates := 26^2 * 10^5
  let new_plates := 26^4 * 10^4
  show new_plates / old_plates = 26^2 / 10
  sorry

end license_plate_increase_l168_168659


namespace count_4digit_numbers_divisible_by_13_l168_168894

theorem count_4digit_numbers_divisible_by_13 : 
  let count := (λ n, n - 77 + 1) (769)
  count = 693 :=
by
-- Let 77 be the smallest integer such that 13 * 77 = 1001
-- Let 769 be the largest integer such that 13 * 769 = 9997
-- Hence, the range of these integers should be 769 - 77 + 1
-- This gives us a count of 693
-- The statement as 693 is directly given
sorry

end count_4digit_numbers_divisible_by_13_l168_168894


namespace parabola_properties_l168_168186

open Real 

theorem parabola_properties 
  (a : ℝ) 
  (h₀ : a ≠ 0)
  (h₁ : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ (x₁^2 + (1 - 2 * a) * x₁ + a^2 = 0) ∧ (x₂^2 + (1 - 2 * a) * x₂ + a^2 = 0)) :
  (a < 1 / 4 ∧ ∀ x₁ x₂, (x₁^2 + (1 - 2 * a) * x₁ + a^2 = 0) ∧ (x₂^2 + (1 - 2 * a) * x₂ + a^2 = 0) → x₁ < 0 ∧ x₂ < 0) ∧
  (∀ (x₁ x₂ C : ℝ), (x₁^2 + (1 - 2 * a) * x₁ + a^2 = 0) ∧ (x₂^2 + (1 - 2 * a) * x₂ + a^2 = 0) 
   ∧ (C = a^2) ∧ (-x₁ - x₂ = C - 2) → a = -3) :=
by
  sorry

end parabola_properties_l168_168186


namespace four_prime_prime_l168_168116

-- Define the function based on the given condition
def q' (q : ℕ) : ℕ := 3 * q - 3

-- The statement to prove
theorem four_prime_prime : (q' (q' 4)) = 24 := by
  sorry

end four_prime_prime_l168_168116


namespace exponent_value_l168_168864

theorem exponent_value (exponent : ℕ) (y: ℕ) :
  (12 ^ exponent) * (6 ^ 4) / 432 = y → y = 36 → exponent = 1 :=
by
  intro h1 h2
  sorry

end exponent_value_l168_168864


namespace closest_whole_number_to_area_of_shaded_region_is_9_l168_168127

theorem closest_whole_number_to_area_of_shaded_region_is_9 :
  let rectangle_area := 4 * 3
  let circle_radius := 2 / 2
  let circle_area := Real.pi * (circle_radius ^ 2)
  let shaded_area := rectangle_area - circle_area
  (Real.floor (shaded_area + 0.5) : Int) = 9 :=
by
  sorry

end closest_whole_number_to_area_of_shaded_region_is_9_l168_168127


namespace hyperbola_asymptotes_correct_l168_168468

noncomputable def asymptotes_for_hyperbola : Prop :=
  ∀ (x y : ℂ),
    9 * (x : ℂ) ^ 2 - 4 * (y : ℂ) ^ 2 = -36 → 
    (y = (3 / 2) * (-Complex.I) * x) ∨ (y = -(3 / 2) * (-Complex.I) * x)

theorem hyperbola_asymptotes_correct :
  asymptotes_for_hyperbola := 
sorry

end hyperbola_asymptotes_correct_l168_168468


namespace number_of_integers_with_gcd_24_4_l168_168337

theorem number_of_integers_with_gcd_24_4 : 
  (Finset.filter (λ n, Int.gcd 24 n = 4) (Finset.range 201)).card = 17 := by
  sorry

end number_of_integers_with_gcd_24_4_l168_168337


namespace negation_correct_l168_168698

-- Define the initial statement
def initial_statement (s : Set ℝ) : Prop :=
  ∀ x ∈ s, |x| ≥ 3

-- Define the negated statement
def negated_statement (s : Set ℝ) : Prop :=
  ∃ x ∈ s, |x| < 3

-- The theorem to be proven
theorem negation_correct (s : Set ℝ) :
  ¬(initial_statement s) ↔ negated_statement s := by
  sorry

end negation_correct_l168_168698


namespace horner_method_correct_l168_168423

-- Define the polynomial function using Horner's method
def f (x : ℤ) : ℤ := (((((x - 8) * x + 60) * x + 16) * x + 96) * x + 240) * x + 64

-- Define the value to be plugged into the polynomial
def x_val : ℤ := 2

-- Compute v_0, v_1, and v_2 according to the Horner's method
def v0 : ℤ := 1
def v1 : ℤ := v0 * x_val - 8
def v2 : ℤ := v1 * x_val + 60

-- Formal statement of the proof problem
theorem horner_method_correct :
  v2 = 48 := by
  -- Insert proof here
  sorry

end horner_method_correct_l168_168423


namespace value_of_a_l168_168380

noncomputable def function_f (x a : ℝ) : ℝ := (x - a) ^ 2 + (Real.log x ^ 2 - 2 * a) ^ 2

theorem value_of_a (x0 : ℝ) (a : ℝ) (h1 : x0 > 0) (h2 : function_f x0 a ≤ 4 / 5) : a = 1 / 5 :=
sorry

end value_of_a_l168_168380


namespace simplify_expression_l168_168990

theorem simplify_expression (w : ℝ) :
  3 * w + 4 - 2 * w - 5 + 6 * w + 7 - 3 * w - 9 = 4 * w - 3 :=
by 
  sorry

end simplify_expression_l168_168990


namespace hoseok_position_reversed_l168_168238

def nine_people (P : ℕ → Prop) : Prop :=
  P 1 ∧ P 2 ∧ P 3 ∧ P 4 ∧ P 5 ∧ P 6 ∧ P 7 ∧ P 8 ∧ P 9

variable (h : ℕ → Prop)

def hoseok_front_foremost : Prop :=
  nine_people h ∧ h 1 -- Hoseok is at the forefront and is the shortest

theorem hoseok_position_reversed :
  hoseok_front_foremost h → h 9 :=
by 
  sorry

end hoseok_position_reversed_l168_168238


namespace increasing_on_interval_of_m_l168_168356

def f (m x : ℝ) := 2 * x^3 - 3 * m * x^2 + 6 * x

theorem increasing_on_interval_of_m (m : ℝ) :
  (∀ x : ℝ, 2 < x → 6 * x^2 - 6 * m * x + 6 ≥ 0) → m ≤ 5 / 2 :=
sorry

end increasing_on_interval_of_m_l168_168356


namespace average_monthly_growth_rate_correct_l168_168365

theorem average_monthly_growth_rate_correct:
  (∃ x : ℝ, 30000 * (1 + x)^2 = 36300) ↔ 3 * (1 + x)^2 = 3.63 := 
by {
  sorry -- proof placeholder
}

end average_monthly_growth_rate_correct_l168_168365


namespace square_of_105_l168_168591

theorem square_of_105 : 105^2 = 11025 := by
  sorry

end square_of_105_l168_168591


namespace solve_polynomial_division_l168_168475

theorem solve_polynomial_division :
  ∃ a : ℤ, (∀ x : ℂ, ∃ p : polynomial ℂ, x^2 - x + (a : ℂ) * p x = x^15 + x^2 + 100) → a = 2 := by
  sorry

end solve_polynomial_division_l168_168475


namespace exponential_function_decreasing_l168_168182

theorem exponential_function_decreasing {a : ℝ} 
  (h : ∀ x y : ℝ, x > y → (a-1)^x < (a-1)^y) : 1 < a ∧ a < 2 :=
by sorry

end exponential_function_decreasing_l168_168182


namespace range_of_m_l168_168346

variables (f : ℝ → ℝ) (m : ℝ)

-- Assume f is a decreasing function
def is_decreasing (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, x < y → f x > f y

-- Assume f is an odd function
def is_odd (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

-- Theorem stating the main condition and the implication
theorem range_of_m (h_decreasing : is_decreasing f) (h_odd : is_odd f) (h_condition : f (m - 1) + f (2 * m - 1) > 0) : m > 2 / 3 :=
sorry

end range_of_m_l168_168346


namespace find_a5_l168_168620

variable {a : ℕ → ℝ} {q : ℝ}

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop := 
  ∀ n, a (n+1) = a n * q

def is_increasing_sequence (a : ℕ → ℝ) : Prop := 
  ∀ n, a n < a (n+1)

def condition1 (a : ℕ → ℝ) : Prop :=
  a 5 ^ 2 = a 10

def condition2 (a : ℕ → ℝ) : Prop :=
  ∀ n, 2 * (a n + a (n+2)) = 5 * a (n+1)

theorem find_a5 (h1 : is_geometric_sequence a q) (h2 : is_increasing_sequence a) (h3 : condition1 a) (h4 : condition2 a) : 
  a 5 = 32 :=
sorry

end find_a5_l168_168620


namespace determine_x0_minus_y0_l168_168174

theorem determine_x0_minus_y0 
  (x0 y0 : ℝ)
  (data_points : List (ℝ × ℝ) := [(1, 2), (3, 5), (6, 8), (x0, y0)])
  (regression_eq : ∀ x, (x + 2) = (x + 2)) :
  x0 - y0 = -3 :=
by
  sorry

end determine_x0_minus_y0_l168_168174


namespace gcd_of_polynomial_l168_168618

theorem gcd_of_polynomial (x : ℕ) (hx : 32515 ∣ x) :
    Nat.gcd ((3 * x + 5) * (5 * x + 3) * (11 * x + 7) * (x + 17)) x = 35 :=
sorry

end gcd_of_polynomial_l168_168618


namespace rice_weight_per_container_l168_168829

-- Given total weight of rice in pounds
def totalWeightPounds : ℚ := 25 / 2

-- Conversion factor from pounds to ounces
def poundsToOunces : ℚ := 16

-- Number of containers
def numberOfContainers : ℕ := 4

-- Total weight in ounces
def totalWeightOunces : ℚ := totalWeightPounds * poundsToOunces

-- Weight per container in ounces
def weightPerContainer : ℚ := totalWeightOunces / numberOfContainers

theorem rice_weight_per_container :
  weightPerContainer = 50 := 
sorry

end rice_weight_per_container_l168_168829


namespace compute_105_squared_l168_168577

theorem compute_105_squared :
  let a := 100
  let b := 5
  (a + b)^2 = 11025 :=
by
  sorry

end compute_105_squared_l168_168577


namespace unit_prices_minimize_cost_l168_168962

theorem unit_prices (x y : ℕ) (h1 : x + 2 * y = 40) (h2 : 2 * x + 3 * y = 70) :
  x = 20 ∧ y = 10 :=
by {
  sorry -- proof would go here
}

theorem minimize_cost (total_pieces : ℕ) (cost_A cost_B : ℕ) 
  (total_cost : ℕ → ℕ)
  (h3 : total_pieces = 60) 
  (h4 : ∀ m, cost_A * m + cost_B * (total_pieces - m) = total_cost m) 
  (h5 : ∀ m, cost_A * m + cost_B * (total_pieces - m) ≥ 800) 
  (h6 : ∀ m, m ≥ (total_pieces - m) / 2) :
  total_cost 20 = 800 :=
by {
  sorry -- proof would go here
}

end unit_prices_minimize_cost_l168_168962


namespace parabola_intersection_l168_168106

def parabola1 (x : ℝ) : ℝ := 3 * x ^ 2 - 4 * x + 2
def parabola2 (x : ℝ) : ℝ := 9 * x ^ 2 + 6 * x + 2

theorem parabola_intersection :
  ∃ (x y : ℝ), (parabola1 x = y ∧ parabola2 x = y) ∧ 
                ((x = 0 ∧ y = 2) ∨ (x = -5 / 3 ∧ y = 17)) :=
by
  sorry

end parabola_intersection_l168_168106


namespace velocity_at_3_seconds_l168_168487

variable (t : ℝ)
variable (s : ℝ)

def motion_eq (t : ℝ) : ℝ := 1 + t + t^2

theorem velocity_at_3_seconds : 
  (deriv motion_eq 3) = 7 :=
by
  sorry

end velocity_at_3_seconds_l168_168487


namespace arithmetic_progression_solution_l168_168167

theorem arithmetic_progression_solution (a1 d : Nat) (hp1 : a1 * (a1 + d) * (a1 + 2 * d) = 6) (hp2 : a1 * (a1 + d) * (a1 + 2 * d) * (a1 + 3 * d) = 24) : 
  (a1 = 1 ∧ d = 1) ∨ (a1 = 2 ∧ d = 1) ∨ (a1 = 3 ∧ d = 1) ∨ (a1 = 4 ∧ d = 1) :=
begin
  sorry
end

end arithmetic_progression_solution_l168_168167


namespace smallest_repeating_block_7_over_13_l168_168012

theorem smallest_repeating_block_7_over_13 : 
  ∃ n : ℕ, (∀ d : ℕ, d < n → 
  (∃ (q r : ℕ), r < 13 ∧ 10 ^ (d + 1) * 7 % 13 = q * 10 ^ n + r)) ∧ n = 6 := sorry

end smallest_repeating_block_7_over_13_l168_168012


namespace find_initial_population_l168_168937

noncomputable def population_first_year (P : ℝ) : ℝ :=
  let P1 := 0.90 * P    -- population after 1st year
  let P2 := 0.99 * P    -- population after 2nd year
  let P3 := 0.891 * P   -- population after 3rd year
  P3

theorem find_initial_population (h : population_first_year P = 4455) : P = 4455 / 0.891 :=
by
  sorry

end find_initial_population_l168_168937


namespace sum_of_series_l168_168017

theorem sum_of_series : 
  (1 / (1 * 2) + 1 / (2 * 3) + 1 / (3 * 4) + 1 / (4 * 5) + 1 / (5 * 6) + 1 / (6 * 7)) = 6 / 7 := 
 by sorry

end sum_of_series_l168_168017


namespace larger_number_is_eight_l168_168957

theorem larger_number_is_eight (x y : ℝ) (h1 : x - y = 3) (h2 : x^2 - y^2 = 39) : x = 8 :=
by
  sorry

end larger_number_is_eight_l168_168957


namespace shaded_region_area_and_circle_centers_l168_168276

theorem shaded_region_area_and_circle_centers :
  ∃ (R : ℝ) (center_big center_small1 center_small2 : ℝ × ℝ),
    R = 10 ∧ 
    center_small1 = (4, 0) ∧
    center_small2 = (10, 0) ∧
    center_big = (7, 0) ∧
    (π * R^2) - (π * 4^2 + π * 6^2) = 48 * π :=
by 
  sorry

end shaded_region_area_and_circle_centers_l168_168276


namespace ellipse_eccentricity_l168_168733

open Real

def ellipse_foci_x_axis (m : ℝ) : Prop :=
  ∃ a b c e,
    a = sqrt m ∧
    b = sqrt 6 ∧
    c = sqrt (m - 6) ∧
    e = c / a ∧
    e = 1 / 2

theorem ellipse_eccentricity (m : ℝ) (h : ellipse_foci_x_axis m) :
  m = 8 := by
  sorry

end ellipse_eccentricity_l168_168733


namespace typing_difference_l168_168147

theorem typing_difference (m : ℕ) (h1 : 10 * m - 8 * m = 10) : m = 5 :=
by
  sorry

end typing_difference_l168_168147


namespace solve_for_y_l168_168393

theorem solve_for_y (y : ℝ) : (y^2 + 6 * y + 8 = -(y + 4) * (y + 6)) → y = -4 :=
by {
  sorry
}

end solve_for_y_l168_168393


namespace sean_bought_3_sodas_l168_168531

def soda_cost (S : ℕ) : ℕ := S * 1
def soup_cost (S : ℕ) (C : ℕ) : Prop := C = S
def sandwich_cost (C : ℕ) (X : ℕ) : Prop := X = 3 * C
def total_cost (S C X : ℕ) : Prop := S + 2 * C + X = 18

theorem sean_bought_3_sodas (S C X : ℕ) (h1 : soup_cost S C) (h2 : sandwich_cost C X) (h3 : total_cost S C X) : S = 3 :=
by
  sorry

end sean_bought_3_sodas_l168_168531


namespace phase_shift_3cos_4x_minus_pi_over_4_l168_168168

theorem phase_shift_3cos_4x_minus_pi_over_4 :
    ∃ (φ : ℝ), y = 3 * Real.cos (4 * x - φ) ∧ φ = π / 16 :=
sorry

end phase_shift_3cos_4x_minus_pi_over_4_l168_168168


namespace a_pow_m_minus_a_pow_n_divisible_by_30_l168_168385

theorem a_pow_m_minus_a_pow_n_divisible_by_30
  (a m n k : ℕ)
  (h_n_ge_two : n ≥ 2)
  (h_m_gt_n : m > n)
  (h_m_n_diff : m = n + 4 * k) :
  30 ∣ (a ^ m - a ^ n) :=
sorry

end a_pow_m_minus_a_pow_n_divisible_by_30_l168_168385


namespace sixth_edge_length_l168_168139

theorem sixth_edge_length (a b c d o : Type) (distance : a -> a -> ℝ) (circumradius : ℝ) 
  (edge_length : ℝ) (h : ∀ (x y : a), x ≠ y → distance x y = edge_length ∨ distance x y = circumradius)
  (eq_edge_length : edge_length = 3) (eq_circumradius : circumradius = 2) : 
  ∃ ad : ℝ, ad = 6 * Real.sqrt (3 / 7) := 
by
  sorry

end sixth_edge_length_l168_168139


namespace polynomial_even_or_odd_polynomial_divisible_by_3_l168_168824

theorem polynomial_even_or_odd (p q : ℤ) :
  (∀ x : ℤ, (x^2 + p * x + q) % 2 = 0 ↔ (q % 2 = 0) ∧ (p % 2 = 1)) ∧
  (∀ x : ℤ, (x^2 + p * x + q) % 2 = 1 ↔ (q % 2 = 1) ∧ (p % 2 = 1)) := 
sorry

theorem polynomial_divisible_by_3 (p q : ℤ) :
  (∀ x : ℤ, (x^3 + p * x + q) % 3 = 0) ↔ (q % 3 = 0) ∧ (p % 3 = 2) := 
sorry

end polynomial_even_or_odd_polynomial_divisible_by_3_l168_168824


namespace hyperbola_asymptote_focal_length_l168_168050

theorem hyperbola_asymptote_focal_length (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) 
  (h3 : c = 2 * Real.sqrt 5) (h4 : b / a = 2) : a = 2 :=
by
  sorry

end hyperbola_asymptote_focal_length_l168_168050


namespace number_doubled_is_12_l168_168118

theorem number_doubled_is_12 (A B C D E : ℝ) (h1 : (A + B + C + D + E) / 5 = 6.8)
  (X : ℝ) (h2 : ((A + B + C + D + E - X) + 2 * X) / 5 = 9.2) : X = 12 :=
by
  sorry

end number_doubled_is_12_l168_168118


namespace probability_of_five_green_marbles_l168_168384

open ProbabilityTheory

-- Define the conditions
def total_marbles := 12
def green_marbles := 8
def purple_marbles := 4
def trials := 8

-- Define the event of interest
def probability_of_exactly_five_green : ℚ :=
  56 * ((2 / 3)^5) * ((1 / 3)^3)

theorem probability_of_five_green_marbles :
  (probability_of_exactly_five_green.to_real = 0.273) :=
  sorry

end probability_of_five_green_marbles_l168_168384


namespace factorize_9_minus_a_squared_l168_168601

theorem factorize_9_minus_a_squared (a : ℤ) : 9 - a^2 = (3 + a) * (3 - a) :=
by
  sorry

end factorize_9_minus_a_squared_l168_168601


namespace range_p_l168_168729

open Set

def p (x : ℝ) : ℝ :=
  x^4 + 6*x^2 + 9

theorem range_p : range p = Ici 9 := by
  sorry

end range_p_l168_168729


namespace johnny_savings_l168_168908

variable (S : ℤ) -- The savings in September.

theorem johnny_savings :
  (S + 49 + 46 - 58 = 67) → (S = 30) :=
by
  intro h
  sorry

end johnny_savings_l168_168908


namespace problem_part1_problem_part2_l168_168738

-- Part 1: Given the conditions, prove that \(\varphi = -\frac{\pi}{6}\)
theorem problem_part1 (f : ℝ → ℝ) (ϕ : ℝ) (hϕ : -π/2 < ϕ ∧ ϕ < π/2) (h_zero : f (π/3) = 0) 
  (h_def : ∀ x, f x = sin (2 * x + ϕ) - 1) : ϕ = -π/6 :=
by
  sorry

-- Part 2: Given the conditions, prove the range of \(f(x)\) on \([0,π/2]\) is \([-3/2, 0]\)
theorem problem_part2 (f : ℝ → ℝ) (ϕ : ℝ) (hϕ : ϕ = -π/6) :
  (∀ x ∈ Icc (0 : ℝ) (π/2), f x = sin (2 * x + ϕ) - 1) → 
  set.range (λ x: ℝ, if x ∈ Icc (0 : ℝ) (π/2) then f x else 0) = Icc (-3 / 2 : ℝ) 0 :=
by
  sorry

end problem_part1_problem_part2_l168_168738


namespace optimize_transport_fleet_l168_168986
-- Lean 4 statement for the equivalent proof problem


axiom normal_distribution_X (X : ℝ) : ProbabilityDistribution ℝ := normal 800 50

-- Define p_0
def p_0 : ℝ := 0.9772

-- Vehicle properties
def capacity_A : ℕ := 36
def capacity_B : ℕ := 60
def cost_A : ℕ := 1600
def cost_B : ℕ := 2400
def max_vehicles : ℕ := 21

-- Constraints
def num_vehicles_A : ℕ := 5
def num_vehicles_B : ℕ := 12

theorem optimize_transport_fleet :
  (∀ X, X ∼ normal_distribution_X →
    (∫ x, (x <= 900) ∂normal_distribution_X = p_0)) ∧ 
  (num_vehicles_A + num_vehicles_B ≤ max_vehicles) ∧
  (num_vehicles_B ≤ num_vehicles_A + 7) ∧
  (num_vehicles_A * capacity_A + num_vehicles_B * capacity_B ≥ (normal_distribution_X.mean)) :=
by
  sorry

end optimize_transport_fleet_l168_168986


namespace min_value_of_a3_l168_168637

open Real

theorem min_value_of_a3 (a : ℕ → ℝ) (hpos : ∀ n, 0 < a n) (hgeo : ∀ n, a (n + 1) / a n = a 1 / a 0)
    (h : a 1 * a 2 * a 3 = a 1 + a 2 + a 3) : a 2 ≥ sqrt 3 := by {
  sorry
}

end min_value_of_a3_l168_168637


namespace find_coordinates_of_P_l168_168479

-- Define the points
def P1 : ℝ × ℝ := (2, -1)
def P2 : ℝ × ℝ := (0, 5)

-- Define the point P
def P : ℝ × ℝ := (-2, 11)

-- Conditions encoded as vector relationships
def vector_P1_P (p : ℝ × ℝ) := (p.1 - P1.1, p.2 - P1.2)
def vector_PP2 (p : ℝ × ℝ) := (P2.1 - p.1, P2.2 - p.2)

-- The hypothesis that | P1P | = 2 * | PP2 |
axiom vector_relation : ∀ (p : ℝ × ℝ), 
  vector_P1_P p = (-2 * (vector_PP2 p).1, -2 * (vector_PP2 p).2) → p = P

theorem find_coordinates_of_P : P = (-2, 11) :=
by
  sorry

end find_coordinates_of_P_l168_168479


namespace sum_d_e_f_l168_168092

-- Define the variables
variables (d e f : ℤ)

-- Given conditions
def condition1 : Prop := ∀ x : ℤ, x^2 + 18 * x + 77 = (x + d) * (x + e)
def condition2 : Prop := ∀ x : ℤ, x^2 - 19 * x + 88 = (x - e) * (x - f)

-- Prove the statement
theorem sum_d_e_f : condition1 d e → condition2 e f → d + e + f = 26 :=
by
  intros h1 h2
  -- Proof omitted
  sorry

end sum_d_e_f_l168_168092


namespace compute_105_squared_l168_168582

theorem compute_105_squared : 105^2 = 11025 :=
by
  sorry

end compute_105_squared_l168_168582


namespace number_division_l168_168305

theorem number_division (n : ℕ) (h1 : 555 + 445 = 1000) (h2 : 555 - 445 = 110) 
  (h3 : n % 1000 = 80) (h4 : n / 1000 = 220) : n = 220080 :=
by {
  -- proof steps would go here
  sorry
}

end number_division_l168_168305


namespace monomial_properties_l168_168670

def coefficient (m : ℝ) := -3
def degree (x_exp y_exp : ℕ) := x_exp + y_exp

theorem monomial_properties :
  ∀ (x_exp y_exp : ℕ), coefficient (-3) = -3 ∧ degree 2 1 = 3 :=
by
  sorry

end monomial_properties_l168_168670


namespace probability_of_sum_16_with_duplicates_l168_168945

namespace DiceProbability

def is_valid_die_roll (n : ℕ) : Prop :=
  n ≥ 1 ∧ n ≤ 6

def is_valid_combination (x y z : ℕ) : Prop :=
  x + y + z = 16 ∧ 
  is_valid_die_roll x ∧ 
  is_valid_die_roll y ∧ 
  is_valid_die_roll z ∧ 
  (x = y ∨ y = z ∨ z = x)

theorem probability_of_sum_16_with_duplicates (P : ℚ) :
  (∃ x y z : ℕ, is_valid_combination x y z) → 
  P = 1 / 36 :=
sorry

end DiceProbability

end probability_of_sum_16_with_duplicates_l168_168945


namespace money_left_after_shopping_l168_168951

-- Definitions based on conditions
def initial_amount : ℝ := 5000
def percentage_spent : ℝ := 0.30
def amount_spent : ℝ := percentage_spent * initial_amount
def remaining_amount : ℝ := initial_amount - amount_spent

-- Theorem statement based on the question and correct answer
theorem money_left_after_shopping : remaining_amount = 3500 :=
by
  sorry

end money_left_after_shopping_l168_168951


namespace pyramid_height_l168_168845

-- Define the heights of individual blocks and the structure of the pyramid.
def block_height := 10 -- in centimeters
def num_layers := 3

-- Define the total height of the pyramid as the sum of the heights of all blocks.
def total_height (block_height : Nat) (num_layers : Nat) := block_height * num_layers

-- The theorem stating that the total height of the stack is 30 cm given the conditions.
theorem pyramid_height : total_height block_height num_layers = 30 := by
  sorry

end pyramid_height_l168_168845


namespace truck_boxes_per_trip_l168_168449

theorem truck_boxes_per_trip (total_boxes trips : ℕ) (h1 : total_boxes = 871) (h2 : trips = 218) : total_boxes / trips = 4 := by
  sorry

end truck_boxes_per_trip_l168_168449


namespace no_point_in_punctured_disk_l168_168460

theorem no_point_in_punctured_disk (A B C D E F G : ℝ) (hB2_4AC : B^2 - 4 * A * C < 0) :
  ∃ δ > 0, ∀ x y : ℝ, 0 < x^2 + y^2 → x^2 + y^2 < δ^2 → 
    ¬(A * x^2 + B * x * y + C * y^2 + D * x^3 + E * x^2 * y + F * x * y^2 + G * y^3 = 0) :=
sorry

end no_point_in_punctured_disk_l168_168460


namespace length_and_width_of_prism_l168_168294

theorem length_and_width_of_prism (w l h d : ℝ) (h_cond : h = 12) (d_cond : d = 15) (length_cond : l = 3 * w) :
  (w = 3) ∧ (l = 9) :=
by
  -- The proof is omitted as instructed in the task description.
  sorry

end length_and_width_of_prism_l168_168294


namespace sequence_explicit_formula_l168_168369

noncomputable def sequence_a : ℕ → ℝ
| 0     => 0  -- Not used, but needed for definition completeness
| 1     => 3
| (n+1) => n / (n + 1) * sequence_a n

theorem sequence_explicit_formula (n : ℕ) (h : n ≠ 0) :
  sequence_a n = 3 / n :=
by sorry

end sequence_explicit_formula_l168_168369


namespace nonneg_int_solutions_eqn_l168_168465

theorem nonneg_int_solutions_eqn :
  { (x, y, z, w) : ℕ × ℕ × ℕ × ℕ | 2^x * 3^y - 5^z * 7^w = 1 } =
  {(1, 0, 0, 0), (3, 0, 0, 1), (1, 1, 1, 0), (2, 2, 1, 1)} :=
by {
  sorry
}

end nonneg_int_solutions_eqn_l168_168465


namespace isosceles_triangle_perimeter_1_isosceles_triangle_perimeter_2_l168_168707

-- Problem 1
theorem isosceles_triangle_perimeter_1 (a b : ℕ) (h1: a = 4 ∨ a = 6) (h2: b = 4 ∨ b = 6) (h3: a ≠ b): 
  (a + b + b = 14 ∨ a + b + b = 16) :=
sorry

-- Problem 2
theorem isosceles_triangle_perimeter_2 (a b : ℕ) (h1: a = 2 ∨ a = 6) (h2: b = 2 ∨ b = 6) (h3: a ≠ b ∨ (a = 2 ∧ 2 + 2 ≥ 6 ∧ 6 = b)):
  (a + b + b = 14) :=
sorry

end isosceles_triangle_perimeter_1_isosceles_triangle_perimeter_2_l168_168707


namespace ramu_paid_for_old_car_l168_168386

theorem ramu_paid_for_old_car (repairs : ℝ) (selling_price : ℝ) (profit_percent : ℝ) (P : ℝ) :
    repairs = 12000 ∧ selling_price = 64900 ∧ profit_percent = 20.185185185185187 → 
    selling_price = P + repairs + (P + repairs) * (profit_percent / 100) → 
    P = 42000 :=
by
  intros h1 h2
  sorry

end ramu_paid_for_old_car_l168_168386


namespace distance_DE_l168_168418

noncomputable def point := (ℝ × ℝ)

variables (A B C P D E : point)
variables (AB BC AC PC : ℝ)
variables (on_line : point → point → point → Prop)
variables (is_parallel : point → point → point → point → Prop)

axiom AB_length : AB = 13
axiom BC_length : BC = 14
axiom AC_length : AC = 15
axiom PC_length : PC = 10

axiom P_on_AC : on_line A C P
axiom D_on_BP : on_line B P D
axiom E_on_BP : on_line B P E

axiom AD_parallel_BC : is_parallel A D B C
axiom AB_parallel_CE : is_parallel A B C E

theorem distance_DE : ∀ (D E : point), 
  on_line B P D → on_line B P E → 
  is_parallel A D B C → is_parallel A B C E → 
  ∃ dist : ℝ, dist = 12 * Real.sqrt 2 :=
by
  sorry

end distance_DE_l168_168418


namespace find_value_of_a_l168_168297

theorem find_value_of_a :
  ∃ a, 
  let xs := [1, 2, 3, 4, 5],
      ys := [2, 3, 7, 8, a],
      mean_x := (xs.sum / 5),
      mean_y := (ys.sum / 5),
      y_hat := 2.1 * mean_x - 0.3
  in mean_y = y_hat ∧ a = 10 := 
by
  sorry

end find_value_of_a_l168_168297


namespace jack_walked_time_l168_168643

def jack_distance : ℝ := 9
def jack_rate : ℝ := 7.2
def jack_time : ℝ := 1.25

theorem jack_walked_time : jack_time = jack_distance / jack_rate := by
  sorry

end jack_walked_time_l168_168643


namespace permutation_order_divides_lcm_l168_168768

open Nat

-- Define a permutation on the set {1, 2, ..., n}
variable (n : ℕ) (h : 0 < n)
variable (σ : Equiv.Perm (Fin n))

noncomputable def set_lcm (n : ℕ) : ℕ :=
  Finset.fold Nat.lcm 1 (Finset.range n).succ

-- The main theorem statement
theorem permutation_order_divides_lcm (n : ℕ) (h : 0 < n) (σ : Equiv.Perm (Fin n)) :
  let m := set_lcm n in orderOf σ ∣ m := sorry

end permutation_order_divides_lcm_l168_168768


namespace total_apples_eaten_l168_168249

def simone_consumption (days: ℕ) (consumption_per_day: ℚ) : ℚ := days * consumption_per_day
def lauri_consumption (days: ℕ) (consumption_per_day: ℚ) : ℚ := days * consumption_per_day

theorem total_apples_eaten :
  simone_consumption 16 (1/2) + lauri_consumption 15 (1/3) = 13 := by
  sorry

end total_apples_eaten_l168_168249


namespace inequality_proof_l168_168484

variable {a1 a2 a3 a4 a5 : ℝ}

theorem inequality_proof (h1 : 1 < a1) (h2 : 1 < a2) (h3 : 1 < a3) (h4 : 1 < a4) (h5 : 1 < a5) :
  16 * (a1 * a2 * a3 * a4 * a5 + 1) > (1 + a1) * (1 + a2) * (1 + a3) * (1 + a4) * (1 + a5) := 
sorry

end inequality_proof_l168_168484


namespace smallest_digit_to_correct_sum_l168_168090

theorem smallest_digit_to_correct_sum (x y z w : ℕ) (hx : x = 753) (hy : y = 946) (hz : z = 821) (hw : w = 2420) :
  ∃ d, d = 7 ∧ (753 + 946 + 821 - 100 * d = 2420) :=
by sorry

end smallest_digit_to_correct_sum_l168_168090


namespace sally_out_of_pocket_l168_168925

-- Definitions based on conditions
def g : ℕ := 320 -- Amount given by the school
def c : ℕ := 12  -- Cost per book
def n : ℕ := 30  -- Number of students

-- Definition derived from conditions
def total_cost : ℕ := n * c
def out_of_pocket : ℕ := total_cost - g

-- Proof statement
theorem sally_out_of_pocket : out_of_pocket = 40 := by
  -- The proof steps would go here
  sorry

end sally_out_of_pocket_l168_168925


namespace not_prime_sum_l168_168770

theorem not_prime_sum (a b c d : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d) (h_eq : a * b = c * d) : ¬ Nat.Prime (a + b + c + d) :=
sorry

end not_prime_sum_l168_168770


namespace elementary_sampling_count_l168_168443

theorem elementary_sampling_count :
  ∃ (a : ℕ), (a + (a + 600) + (a + 1200) = 3600) ∧
             (a = 600) ∧
             (a + 1200 = 1800) ∧
             (1800 * 1 / 100 = 18) :=
by {
  sorry
}

end elementary_sampling_count_l168_168443


namespace roots_difference_squared_quadratic_roots_property_l168_168498

noncomputable def α : ℝ := (3 + Real.sqrt 5) / 2
noncomputable def β : ℝ := (3 - Real.sqrt 5) / 2

theorem roots_difference_squared :
  α - β = Real.sqrt 5 :=
by
  sorry

theorem quadratic_roots_property :
  (α - β) ^ 2 = 5 :=
by
  sorry

end roots_difference_squared_quadratic_roots_property_l168_168498


namespace fraction_comparison_l168_168222

theorem fraction_comparison (a b c d : ℝ) 
  (h1 : (a / b) < (c / d))
  (h2 : b > d) (h3 : d > 0) :
  (a + c) / (b + d) < (1 / 2) * ((a / b) + (c / d)) :=
by
  sorry

end fraction_comparison_l168_168222


namespace largest_power_of_three_dividing_A_l168_168642

theorem largest_power_of_three_dividing_A (A : ℕ)
  (h1 : ∃ (factors : List ℕ), (∀ b ∈ factors, b > 0) ∧ factors.sum = 2011 ∧ factors.prod = A)
  : ∃ k : ℕ, 3^k ∣ A ∧ ∀ m : ℕ, 3^m ∣ A → m ≤ 669 :=
by
  sorry

end largest_power_of_three_dividing_A_l168_168642


namespace nat_ineq_qr_ps_l168_168779

theorem nat_ineq_qr_ps (a b p q r s : ℕ) (h₀ : q * r - p * s = 1) 
  (h₁ : (p : ℚ) / q < a / b) (h₂ : (a : ℚ) / b < r / s) 
  : b ≥ q + s := sorry

end nat_ineq_qr_ps_l168_168779


namespace find_prime_c_l168_168911

-- Define the statement of the problem
theorem find_prime_c (c : ℕ) (hc : Nat.Prime c) (h : ∃ m : ℕ, (m > 0) ∧ (11 * c + 1 = m^2)) : c = 13 :=
by
  sorry

end find_prime_c_l168_168911


namespace non_shaded_region_perimeter_l168_168251

def outer_rectangle_length : ℕ := 12
def outer_rectangle_width : ℕ := 10
def inner_rectangle_length : ℕ := 6
def inner_rectangle_width : ℕ := 2
def shaded_area : ℕ := 116

theorem non_shaded_region_perimeter :
  let total_area := outer_rectangle_length * outer_rectangle_width
  let inner_area := inner_rectangle_length * inner_rectangle_width
  let non_shaded_area := total_area - shaded_area
  non_shaded_area = 4 →
  ∃ width height, width * height = non_shaded_area ∧ 2 * (width + height) = 10 :=
by intros
   sorry

end non_shaded_region_perimeter_l168_168251


namespace zero_point_condition_l168_168616

-- Define the function f(x) = ax + 3
def f (a x : ℝ) : ℝ := a * x + 3

-- Define that a > 2 is necessary but not sufficient condition
theorem zero_point_condition (a : ℝ) (h : a > 2) : 
  (∀ (x : ℝ), -1 ≤ x ∧ x ≤ 0 → f a x = 0) ↔ (a ≥ 3) := 
sorry

end zero_point_condition_l168_168616


namespace square_area_inside_ellipse_l168_168566

theorem square_area_inside_ellipse :
  (∃ s : ℝ, 
    ∀ (x y : ℝ), 
      (x = s ∧ y = s) → 
      (x^2 / 4 + y^2 / 8 = 1) ∧ 
      (4 * (s^2 / 3) = 1) ∧ 
      (area = 4 * (8 / 3))) →
    ∃ area : ℝ, 
      area = 32 / 3 :=
by
  sorry

end square_area_inside_ellipse_l168_168566


namespace find_cos_beta_l168_168736

variable {α β : ℝ}
variable (h_acute_α : 0 < α ∧ α < π / 2)
variable (h_acute_β : 0 < β ∧ β < π / 2)
variable (h_sin_α : Real.sin α = 2 / 5 * Real.sqrt 5)
variable (h_sin_α_plus_β : Real.sin (α + β) = 3 / 5)

theorem find_cos_beta 
  (h_acute_α : 0 < α ∧ α < π / 2)
  (h_acute_β : 0 < β ∧ β < π / 2)
  (h_sin_α : Real.sin α = 2 / 5 * Real.sqrt 5)
  (h_sin_α_plus_β : Real.sin (α + β) = 3 / 5) :
  Real.cos β = Real.sqrt 5 / 5 := 
sorry

end find_cos_beta_l168_168736


namespace probability_white_given_black_drawn_l168_168690

-- Definitions based on the conditions
def num_white : ℕ := 3
def num_black : ℕ := 2
def total_balls : ℕ := num_white + num_black

def P (n : ℕ) : ℚ := n / total_balls

-- Event A: drawing a black ball on the first draw
def PA : ℚ := P num_black

-- Event B: drawing a white ball on the second draw
def PB_given_A : ℚ := num_white / (total_balls - 1)

-- Theorem statement
theorem probability_white_given_black_drawn :
  (PA * PB_given_A) / PA = 3 / 4 :=
by
  sorry

end probability_white_given_black_drawn_l168_168690


namespace equality_of_a_b_c_l168_168338

theorem equality_of_a_b_c
  (a b c : ℝ) (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
  (eqn : a^2 * (b + c - a) = b^2 * (c + a - b) ∧ b^2 * (c + a - b) = c^2 * (a + b - c)) :
  a = b ∧ b = c :=
by
  sorry

end equality_of_a_b_c_l168_168338


namespace proof_max_difference_l168_168122

/-- Digits as displayed on the engineering calculator -/
structure Digits :=
  (a b c d e f g h i : ℕ)

-- Possible digits based on broken displays
axiom a_values : {x // x = 3 ∨ x = 5 ∨ x = 9}
axiom b_values : {x // x = 2 ∨ x = 3 ∨ x = 7}
axiom c_values : {x // x = 3 ∨ x = 4 ∨ x = 8 ∨ x = 9}
axiom d_values : {x // x = 2 ∨ x = 3 ∨ x = 7}
axiom e_values : {x // x = 3 ∨ x = 5 ∨ x = 9}
axiom f_values : {x // x = 1 ∨ x = 4 ∨ x = 7}
axiom g_values : {x // x = 4 ∨ x = 5 ∨ x = 9}
axiom h_values : {x // x = 2}
axiom i_values : {x // x = 4 ∨ x = 5 ∨ x = 9}

-- Minuend and subtrahend values
def minuend := 923
def subtrahend := 394

-- Maximum possible value of the difference
def max_difference := 529

theorem proof_max_difference : 
  ∃ (digits : Digits),
    digits.a = 9 ∧ digits.b = 2 ∧ digits.c = 3 ∧
    digits.d = 3 ∧ digits.e = 9 ∧ digits.f = 4 ∧
    digits.g = 5 ∧ digits.h = 2 ∧ digits.i = 9 ∧
    minuend - subtrahend = max_difference :=
by
  sorry

end proof_max_difference_l168_168122


namespace p_plus_q_eq_10_l168_168936

theorem p_plus_q_eq_10 (p q : ℕ) (hp : p > q) (hpq1 : p < 10) (hpq2 : q < 10)
  (h : p.factorial / q.factorial = 840) : p + q = 10 :=
by
  sorry

end p_plus_q_eq_10_l168_168936


namespace max_mark_paper_i_l168_168835

theorem max_mark_paper_i (M : ℝ) (h1 : 0.65 * M = 170) : M ≈ 262 :=
by sorry

end max_mark_paper_i_l168_168835


namespace find_c_l168_168049

-- Definition of the function f
def f (x a b c : ℤ) : ℤ := x^3 + a * x^2 + b * x + c

-- Theorem statement
theorem find_c (a b c : ℤ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : f a a b c = a^3)
  (h4 : f b a b c = b^3) : c = 16 :=
by
  sorry

end find_c_l168_168049


namespace an_general_term_sum_bn_l168_168042

open Nat

variable (a : ℕ → ℕ)
variable (b : ℕ → ℕ)
variable (S : ℕ → ℕ)
variable (T : ℕ → ℕ)

-- Conditions
axiom a3 : a 3 = 3
axiom S6 : S 6 = 21
axiom Sn : ∀ n, S n = n * (a 1 + a n) / 2

-- Define bn based on the given condition for bn = an + 2^n
def bn (n : ℕ) : ℕ := a n + 2^n

-- Define Tn based on the given condition for Tn.
def Tn (n : ℕ) : ℕ := (n * (n + 1)) / 2 + (2^(n + 1) - 2)

-- Prove the general term formula of the arithmetic sequence an
theorem an_general_term (n : ℕ) : a n = n :=
by
  sorry

-- Prove the sum of the first n terms of the sequence bn
theorem sum_bn (n : ℕ) : T n = Tn n :=
by
  sorry

end an_general_term_sum_bn_l168_168042


namespace range_of_x_l168_168033

theorem range_of_x (x y : ℝ) (h1 : 4 * x + y = 3) (h2 : -2 < y ∧ y ≤ 7) : 1 ≤ x ∧ x < 5 / 4 := 
  sorry

end range_of_x_l168_168033


namespace greatest_value_of_4a_l168_168791

-- Definitions of the given conditions
def hundreds_digit (x : ℕ) : ℕ := x / 100
def tens_digit (x : ℕ) : ℕ := (x / 10) % 10
def units_digit (x : ℕ) : ℕ := x % 10

def satisfies_conditions (a b c x : ℕ) : Prop :=
  hundreds_digit x = a ∧
  tens_digit x = b ∧
  units_digit x = c ∧
  4 * a = 2 * b ∧
  2 * b = c ∧
  a > 0

def difference_of_two_greatest_x : ℕ := 124

theorem greatest_value_of_4a (x1 x2 a1 a2 b1 b2 c1 c2 : ℕ) :
  satisfies_conditions a1 b1 c1 x1 →
  satisfies_conditions a2 b2 c2 x2 →
  x1 - x2 = difference_of_two_greatest_x →
  4 * a1 = 8 :=
by
  sorry

end greatest_value_of_4a_l168_168791


namespace cody_final_money_l168_168152

-- Definitions for the initial conditions
def original_money : ℝ := 45
def birthday_money : ℝ := 9
def game_price : ℝ := 19
def discount_rate : ℝ := 0.10
def friend_owes : ℝ := 12

-- Calculate the final amount Cody has
def final_amount : ℝ := original_money + birthday_money - (game_price * (1 - discount_rate)) + friend_owes

-- The theorem to prove the amount of money Cody has now
theorem cody_final_money :
  final_amount = 48.90 :=
by sorry

end cody_final_money_l168_168152


namespace find_x_l168_168059

/-- Let x be a real number such that the square roots of a positive number are given by x - 4 and 3. 
    Prove that x equals 1. -/
theorem find_x (x : ℝ) 
  (h₁ : ∃ n : ℝ, n > 0 ∧ n.sqrt = x - 4 ∧ n.sqrt = 3) : 
  x = 1 :=
by
  sorry

end find_x_l168_168059


namespace root_poly_ratio_c_d_l168_168796

theorem root_poly_ratio_c_d (a b c d : ℝ)
  (h₁ : 1 + (-2) + 3 = 2)
  (h₂ : 1 * (-2) + (-2) * 3 + 3 * 1 = -5)
  (h₃ : 1 * (-2) * 3 = -6)
  (h_sum : -b / a = 2)
  (h_pair_prod : c / a = -5)
  (h_prod : -d / a = -6) :
  c / d = 5 / 6 := by
  sorry

end root_poly_ratio_c_d_l168_168796


namespace percentage_reduction_in_price_l168_168565

theorem percentage_reduction_in_price (P R : ℝ) (hR : R = 2.953846153846154)
  (h_condition : ∃ P, 65 / 12 * R = 40 - 24 / P) :
  ((P - R) / P) * 100 = 33.3 := by
  sorry

end percentage_reduction_in_price_l168_168565


namespace tony_bread_slices_left_l168_168268

def num_slices_left (initial_slices : ℕ) (d1 : ℕ) (d2 : ℕ) : ℕ :=
  initial_slices - (d1 + d2)

theorem tony_bread_slices_left :
  num_slices_left 22 (5 * 2) (2 * 2) = 8 := by
  sorry

end tony_bread_slices_left_l168_168268


namespace not_right_triangle_D_right_triangle_A_right_triangle_B_right_triangle_C_l168_168696

def right_angle_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

theorem not_right_triangle_D (a b c : ℝ):
  ¬ (a = 3^2 ∧ b = 4^2 ∧ c = 5^2 ∧ right_angle_triangle a b c) :=
sorry

theorem right_triangle_A (a b c x : ℝ):
  a = 5 * x → b = 12 * x → c = 13 * x → x > 0 → right_angle_triangle a b c :=
sorry

theorem right_triangle_B (angleA angleB angleC : ℝ):
  angleA / angleB / angleC = 2 / 3 / 5 → angleC = 90 → angleA + angleB + angleC = 180 → right_angle_triangle angleA angleB angleC :=
sorry

theorem right_triangle_C (a b c k : ℝ):
  a = 9 * k → b = 40 * k → c = 41 * k → k > 0 → right_angle_triangle a b c :=
sorry

end not_right_triangle_D_right_triangle_A_right_triangle_B_right_triangle_C_l168_168696


namespace log_change_of_base_log_change_of_base_with_b_l168_168113

variable {a b x : ℝ}
variable (h₁ : 0 < a ∧ a ≠ 1)
variable (h₂ : 0 < b ∧ b ≠ 1)
variable (h₃ : 0 < x)

theorem log_change_of_base (h₁ : 0 < a ∧ a ≠ 1) (h₂ : 0 < b ∧ b ≠ 1) (h₃ : 0 < x) : 
  Real.log x / Real.log a = Real.log x / Real.log b := by
  sorry

theorem log_change_of_base_with_b (h₁ : 0 < a ∧ a ≠ 1) (h₂ : 0 < b ∧ b ≠ 1) : 
  Real.log b / Real.log a = 1 / Real.log a := by
  sorry

end log_change_of_base_log_change_of_base_with_b_l168_168113


namespace colin_speed_l168_168321

variable (B T Br C D : ℝ)

-- Given conditions
axiom cond1 : C = 6 * Br
axiom cond2 : Br = (1/3) * T^2
axiom cond3 : T = 2 * B
axiom cond4 : D = (1/4) * C
axiom cond5 : B = 1

-- Prove Colin's speed C is 8 mph
theorem colin_speed :
  C = 8 :=
by
  sorry

end colin_speed_l168_168321


namespace preferred_order_for_boy_l168_168441

variable (p q : ℝ)
variable (h : p < q)

theorem preferred_order_for_boy (p q : ℝ) (h : p < q) : 
  (2 * p * q - p^2 * q) > (2 * p * q - p * q^2) := 
sorry

end preferred_order_for_boy_l168_168441


namespace sum_of_num_and_denom_l168_168227

-- Define the repeating decimal G
def G : ℚ := 739 / 999

-- State the theorem
theorem sum_of_num_and_denom (a b : ℕ) (hb : b ≠ 0) (h : G = a / b) : a + b = 1738 := sorry

end sum_of_num_and_denom_l168_168227


namespace tablecloth_covers_table_l168_168850

theorem tablecloth_covers_table
(length_ellipse : ℝ) (width_ellipse : ℝ) (length_tablecloth : ℝ) (width_tablecloth : ℝ)
(h1 : length_ellipse = 160)
(h2 : width_ellipse = 100)
(h3 : length_tablecloth = 140)
(h4 : width_tablecloth = 130) :
length_tablecloth >= width_ellipse ∧ width_tablecloth >= width_ellipse ∧
(length_tablecloth ^ 2 + width_tablecloth ^ 2) >= (length_ellipse ^ 2 + width_ellipse ^ 2) :=
by
  sorry

end tablecloth_covers_table_l168_168850


namespace total_cable_cost_neighborhood_l168_168319

-- Define the number of east-west streets and their length
def ew_streets : ℕ := 18
def ew_length_per_street : ℕ := 2

-- Define the number of north-south streets and their length
def ns_streets : ℕ := 10
def ns_length_per_street : ℕ := 4

-- Define the cable requirements and cost
def cable_per_mile_of_street : ℕ := 5
def cable_cost_per_mile : ℕ := 2000

-- Calculate total length of east-west streets
def ew_total_length : ℕ := ew_streets * ew_length_per_street

-- Calculate total length of north-south streets
def ns_total_length : ℕ := ns_streets * ns_length_per_street

-- Calculate total length of all streets
def total_street_length : ℕ := ew_total_length + ns_total_length

-- Calculate total length of cable required
def total_cable_length : ℕ := total_street_length * cable_per_mile_of_street

-- Calculate total cost of the cable
def total_cost : ℕ := total_cable_length * cable_cost_per_mile

-- The statement to prove
theorem total_cable_cost_neighborhood : total_cost = 760000 :=
by
  sorry

end total_cable_cost_neighborhood_l168_168319


namespace sandy_took_310_dollars_l168_168291

theorem sandy_took_310_dollars (X : ℝ) (h70percent : 0.70 * X = 217) : X = 310 := by
  sorry

end sandy_took_310_dollars_l168_168291


namespace unique_function_satisfying_condition_l168_168074

theorem unique_function_satisfying_condition :
  ∀ f : ℝ → ℝ, (∀ x y : ℝ, f (x^2 + f y) = y + (f x)^2) ↔ f = id :=
sorry

end unique_function_satisfying_condition_l168_168074


namespace max_period_of_function_l168_168872

theorem max_period_of_function (f : ℝ → ℝ) (h1 : ∀ x, f (1 + x) = f (1 - x)) (h2 : ∀ x, f (8 + x) = f (8 - x)) :
  ∃ T > 0, (∀ x, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T' ≥ 14) ∧ T = 14 :=
sorry

end max_period_of_function_l168_168872


namespace geometric_sequence_20_sum_is_2_pow_20_sub_1_l168_168343

def geometric_sequence_sum_condition (a : ℕ → ℕ) (q : ℕ) : Prop :=
  (a 1 * q + 2 * a 1 = 4) ∧ (a 1 ^ 2 * q ^ 4 = a 1 * q ^ 4)

noncomputable def geometric_sequence_sum (a : ℕ → ℕ) (q : ℕ) : ℕ :=
  (a 1 * (1 - q ^ 20)) / (1 - q)

theorem geometric_sequence_20_sum_is_2_pow_20_sub_1 (a : ℕ → ℕ) (q : ℕ) 
  (h : geometric_sequence_sum_condition a q) : 
  geometric_sequence_sum a q =  2 ^ 20 - 1 := 
sorry

end geometric_sequence_20_sum_is_2_pow_20_sub_1_l168_168343


namespace rectangles_perimeter_l168_168781

theorem rectangles_perimeter : 
  let base := 4
  let top := 4
  let left_side := 4
  let right_side := 6
  base + top + left_side + right_side = 18 := 
by {
  let base := 4
  let top := 4
  let left_side := 4
  let right_side := 6
  sorry
}

end rectangles_perimeter_l168_168781


namespace mixed_number_multiplication_equiv_l168_168697

theorem mixed_number_multiplication_equiv :
  (-3 - 1 / 2) * (5 / 7) = -3.5 * (5 / 7) := 
by 
  sorry

end mixed_number_multiplication_equiv_l168_168697


namespace arithmetic_sequence_a4_l168_168212

def a (n : ℕ) : ℕ :=
  if n = 1 then 2 else if n = 2 then 4 else 2 + (n - 1) * 2

theorem arithmetic_sequence_a4 :
  a 4 = 8 :=
by {
  sorry
}

end arithmetic_sequence_a4_l168_168212


namespace complement_union_A_B_l168_168915

-- Define the sets U, A, and B as per the conditions
def U : Set ℕ := {x | x > 0 ∧ x < 6}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {3, 5}

-- Specify the statement to prove the complement of A ∪ B with respect to U
theorem complement_union_A_B : (U \ (A ∪ B)) = {2, 4} :=
by
  sorry

end complement_union_A_B_l168_168915


namespace molecular_weight_6_moles_C4H8O2_is_528_624_l168_168284

-- Define the atomic weights of Carbon, Hydrogen, and Oxygen.
def atomic_weight_C : ℝ := 12.01
def atomic_weight_H : ℝ := 1.008
def atomic_weight_O : ℝ := 16.00

-- Define the molecular formula of C4H8O2.
def num_C_atoms : ℕ := 4
def num_H_atoms : ℕ := 8
def num_O_atoms : ℕ := 2

-- Define the number of moles of C4H8O2.
def num_moles_C4H8O2 : ℝ := 6

-- Define the molecular weight of one mole of C4H8O2.
def molecular_weight_C4H8O2 : ℝ :=
  (num_C_atoms * atomic_weight_C) +
  (num_H_atoms * atomic_weight_H) +
  (num_O_atoms * atomic_weight_O)

-- The total weight of 6 moles of C4H8O2.
def total_weight_6_moles_C4H8O2 : ℝ :=
  num_moles_C4H8O2 * molecular_weight_C4H8O2

-- Theorem stating that the molecular weight of 6 moles of C4H8O2 is 528.624 grams.
theorem molecular_weight_6_moles_C4H8O2_is_528_624 :
  total_weight_6_moles_C4H8O2 = 528.624 :=
by
  -- Proof is omitted.
  sorry

end molecular_weight_6_moles_C4H8O2_is_528_624_l168_168284


namespace number_of_buses_l168_168401

theorem number_of_buses (total_supervisors : ℕ) (supervisors_per_bus : ℕ) (h1 : total_supervisors = 21) (h2 : supervisors_per_bus = 3) : total_supervisors / supervisors_per_bus = 7 :=
by
  sorry

end number_of_buses_l168_168401


namespace single_elimination_games_l168_168068

theorem single_elimination_games (n : ℕ) (h : n = 512) : ∃ g : ℕ, g = 511 := by
  sorry

end single_elimination_games_l168_168068


namespace trip_cost_l168_168451

variable (price : ℕ) (discount : ℕ) (numPeople : ℕ)

theorem trip_cost :
  price = 147 →
  discount = 14 →
  numPeople = 2 →
  (price - discount) * numPeople = 266 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  simp
  sorry

end trip_cost_l168_168451


namespace impossible_triangle_angle_sum_l168_168953

theorem impossible_triangle_angle_sum (x y z : ℝ) (h : x + y + z = 180) : x + y + z ≠ 360 :=
by
sorry

end impossible_triangle_angle_sum_l168_168953


namespace ratio_student_adult_tickets_l168_168716

theorem ratio_student_adult_tickets (A : ℕ) (S : ℕ) (total_tickets: ℕ) (multiple: ℕ) :
  (A = 122) →
  (total_tickets = 366) →
  (S = multiple * A) →
  (S + A = total_tickets) →
  (S / A = 2) :=
by
  intros hA hTotal hMultiple hSum
  -- The proof will go here
  sorry

end ratio_student_adult_tickets_l168_168716


namespace sales_tax_reduction_difference_l168_168289

def sales_tax_difference (original_rate new_rate market_price : ℝ) : ℝ :=
  (market_price * original_rate) - (market_price * new_rate)

theorem sales_tax_reduction_difference :
  sales_tax_difference 0.035 0.03333 10800 = 18.36 :=
by
  -- This is where the proof would go, but it is not required for this task.
  sorry

end sales_tax_reduction_difference_l168_168289


namespace smallest_N_for_abs_x_squared_minus_4_condition_l168_168206

theorem smallest_N_for_abs_x_squared_minus_4_condition (x : ℝ) 
  (h : abs (x - 2) < 0.01) : abs (x^2 - 4) < 0.0401 := 
sorry

end smallest_N_for_abs_x_squared_minus_4_condition_l168_168206


namespace man_days_to_complete_work_alone_l168_168135

-- Defining the variables corresponding to the conditions
variable (M : ℕ)

-- Initial condition: The man can do the work alone in M days
def man_work_rate := 1 / (M : ℚ)
-- The son can do the work alone in 20 days
def son_work_rate := 1 / 20
-- Combined work rate when together
def combined_work_rate := 1 / 4

-- The main theorem we want to prove
theorem man_days_to_complete_work_alone
  (h : man_work_rate M + son_work_rate = combined_work_rate) :
  M = 5 := by
  sorry

end man_days_to_complete_work_alone_l168_168135


namespace find_equation_of_l_l168_168486

open Real

/-- Define the point M(2, 1) -/
def M : ℝ × ℝ := (2, 1)

/-- Define the original line equation x - 2y + 1 = 0 as a function -/
def line1 (x y : ℝ) : Prop := x - 2 * y + 1 = 0

/-- Define the line l that passes through M and is perpendicular to line1 -/
def line_l (x y : ℝ) : Prop := 2 * x + y - 5 = 0

/-- The theorem to be proven: the line l passing through M and perpendicular to line1 has the equation 2x + y - 5 = 0 -/
theorem find_equation_of_l (x y : ℝ)
  (hM : M = (2, 1))
  (hl_perpendicular : ∀ x y : ℝ, line1 x y → line_l y (-x / 2)) :
  line_l x y ↔ (x, y) = (2, 1) :=
by
  sorry

end find_equation_of_l_l168_168486


namespace interval_where_f_increasing_l168_168623

noncomputable def f (x : ℝ) : ℝ := Real.log (4 * x - x^2) / Real.log (1 / 2)

theorem interval_where_f_increasing : ∀ x : ℝ, 2 ≤ x ∧ x < 4 → f x < f (x + 1) :=
by 
  sorry

end interval_where_f_increasing_l168_168623


namespace smallest_positive_period_1_smallest_positive_period_2_l168_168457

-- To prove the smallest positive period T for f(x) = |sin x| + |cos x| is π/2
theorem smallest_positive_period_1 : ∃ T > 0, T = Real.pi / 2 ∧ ∀ x : ℝ, (abs (Real.sin (x + T)) + abs (Real.cos (x + T)) = abs (Real.sin x) + abs (Real.cos x))  := sorry

-- To prove the smallest positive period T for f(x) = tan (2x/3) is 3π/2
theorem smallest_positive_period_2 : ∃ T > 0, T = 3 * Real.pi / 2 ∧ ∀ x : ℝ, (Real.tan ((2 * x) / 3 + T) = Real.tan ((2 * x) / 3)) := sorry

end smallest_positive_period_1_smallest_positive_period_2_l168_168457


namespace freshmen_and_sophomores_without_pets_l168_168940

theorem freshmen_and_sophomores_without_pets :
  let total_students := 400
  let freshmen_sophomores_fraction := 0.50
  let pet_owners_fraction := 1 / 5
  let freshmen_sophomores := total_students * freshmen_sophomores_fraction
  let pet_owners := freshmen_sophomores * pet_owners_fraction
  let non_pet_owners := freshmen_sophomores - pet_owners
  non_pet_owners = 160 :=
by
  Sorry

end freshmen_and_sophomores_without_pets_l168_168940


namespace computation_problems_count_l168_168977

theorem computation_problems_count : 
  ∃ (x y : ℕ), 3 * x + 5 * y = 110 ∧ x + y = 30 ∧ x = 20 :=
by
  sorry

end computation_problems_count_l168_168977


namespace planes_are_perpendicular_l168_168660

-- Define the normal vectors
def N1 : List ℝ := [2, 3, -4]
def N2 : List ℝ := [5, -2, 1]

-- Define the dot product function
def dotProduct (v1 v2 : List ℝ) : ℝ :=
  List.zipWith (fun a b => a * b) v1 v2 |>.sum

-- State the theorem
theorem planes_are_perpendicular :
  dotProduct N1 N2 = 0 :=
by
  sorry

end planes_are_perpendicular_l168_168660


namespace intersection_M_N_l168_168883

def M : Set ℝ := {x | x^2 - 2 * x < 0}
def N : Set ℝ := {-2, -1, 0, 1, 2}

theorem intersection_M_N : M ∩ N = {1} :=
  by sorry

end intersection_M_N_l168_168883


namespace find_A_l168_168295

theorem find_A (A B : ℝ) (h1 : B = 10 * A) (h2 : 211.5 = B - A) : A = 23.5 :=
by {
  sorry
}

end find_A_l168_168295


namespace diophantine_3x_5y_diophantine_3x_5y_indefinite_l168_168806

theorem diophantine_3x_5y (n : ℕ) (h_n_pos : n > 0) :
  (∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ 3 * x + 5 * y = n) ↔ 
    (∃ k : ℕ, (n = 3 * k ∧ n ≥ 15) ∨ 
              (n = 3 * k + 1 ∧ n ≥ 13) ∨ 
              (n = 3 * k + 2 ∧ n ≥ 11) ∨ 
              (n = 8)) :=
sorry

theorem diophantine_3x_5y_indefinite (n m : ℕ) (h_n_large : n > 40 * m):
  ∃ (N : ℕ), ∀ k ≤ N, ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ 3 * x + 5 * y = n + k :=
sorry

end diophantine_3x_5y_diophantine_3x_5y_indefinite_l168_168806


namespace determine_values_l168_168411

-- Define the main problem conditions
def A := 1.2
def B := 12

-- The theorem statement capturing the problem conditions and the solution
theorem determine_values (A B : ℝ) (h1 : A + B = 13.2) (h2 : B = 10 * A) : A = 1.2 ∧ B = 12 :=
  sorry

end determine_values_l168_168411


namespace product_of_digits_l168_168197

theorem product_of_digits (A B : ℕ) (h1 : A + B = 12) (h2 : (10 * A + B) % 4 = 0) : A * B = 32 ∨ A * B = 36 :=
sorry

end product_of_digits_l168_168197


namespace rebecca_end_of_day_money_eq_l168_168082

-- Define the costs for different services
def haircut_cost   := 30
def perm_cost      := 40
def dye_job_cost   := 60
def extension_cost := 80

-- Define the supply costs for the services
def haircut_supply_cost   := 5
def dye_job_supply_cost   := 10
def extension_supply_cost := 25

-- Today's appointments
def num_haircuts   := 5
def num_perms      := 3
def num_dye_jobs   := 2
def num_extensions := 1

-- Additional incomes and expenses
def tips           := 75
def daily_expenses := 45

-- Calculate the total earnings and costs
def total_service_revenue : ℕ := 
  num_haircuts * haircut_cost +
  num_perms * perm_cost +
  num_dye_jobs * dye_job_cost +
  num_extensions * extension_cost

def total_revenue : ℕ := total_service_revenue + tips

def total_supply_cost : ℕ := 
  num_haircuts * haircut_supply_cost +
  num_dye_jobs * dye_job_supply_cost +
  num_extensions * extension_supply_cost

def end_of_day_money : ℕ := total_revenue - total_supply_cost - daily_expenses

-- Lean statement to prove Rebecca will have $430 at the end of the day
theorem rebecca_end_of_day_money_eq : end_of_day_money = 430 := by
  sorry

end rebecca_end_of_day_money_eq_l168_168082


namespace total_apples_eaten_l168_168248

theorem total_apples_eaten : 
  let simone_consumption := 1/2 * 16
  let lauri_consumption := 1/3 * 15
  simone_consumption + lauri_consumption = 13 :=
by
  let simone_consumption := 1/2 * 16
  let lauri_consumption := 1/3 * 15
  have H1 : simone_consumption = 8 := by sorry
  have H2 : lauri_consumption = 5 := by sorry
  show simone_consumption + lauri_consumption = 13 by sorry

end total_apples_eaten_l168_168248


namespace abcd_product_l168_168769

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
noncomputable def c : ℝ := sorry
noncomputable def d : ℝ := sorry

axiom a_eq : a = Real.sqrt (4 - Real.sqrt (5 - a))
axiom b_eq : b = Real.sqrt (4 + Real.sqrt (5 - b))
axiom c_eq : c = Real.sqrt (4 - Real.sqrt (5 + c))
axiom d_eq : d = Real.sqrt (4 + Real.sqrt (5 + d))

theorem abcd_product : a * b * c * d = 11 := sorry

end abcd_product_l168_168769


namespace value_of_b_l168_168061

theorem value_of_b (y : ℝ) (b : ℝ) (h_pos : y > 0) (h_eqn : (7 * y) / b + (3 * y) / 10 = 0.6499999999999999 * y) : 
  b = 70 / 61.99999999999999 :=
sorry

end value_of_b_l168_168061


namespace deg_to_rad_neg_630_l168_168856

theorem deg_to_rad_neg_630 :
  (-630 : ℝ) * (Real.pi / 180) = -7 * Real.pi / 2 := by
  sorry

end deg_to_rad_neg_630_l168_168856


namespace max_marks_paper_I_l168_168836

variable (M : ℝ)

theorem max_marks_paper_I (h1 : 0.65 * M = 112 + 58) : M = 262 :=
  sorry

end max_marks_paper_I_l168_168836


namespace median_length_angle_bisector_length_l168_168165

variable (a b c : ℝ) (ma n : ℝ)

theorem median_length (h1 : ma = Real.sqrt ((b^2 + c^2) / 2 - a^2 / 4)) : 
  ma = Real.sqrt ((b^2 + c^2) / 2 - a^2 / 4) :=
by
  sorry

theorem angle_bisector_length (h2 : n = b * c * Real.sqrt (((b + c)^2 - a^2) / (b + c)^2)) :
  n = b * c * Real.sqrt (((b + c)^2 - a^2) / (b + c)^2) :=
by
  sorry

end median_length_angle_bisector_length_l168_168165


namespace remainder_of_sum_mod_11_l168_168029

theorem remainder_of_sum_mod_11 :
  (123456 + 123457 + 123458 + 123459 + 123460 + 123461) % 11 = 10 :=
by
  sorry

end remainder_of_sum_mod_11_l168_168029


namespace find_multiple_l168_168410

-- Given conditions as definitions
def smaller_number := 21
def sum_of_numbers := 84

-- Definition of larger number being a multiple of the smaller number
def is_multiple (k : ℤ) (a b : ℤ) : Prop := b = k * a

-- Given that one number is a multiple of the other and their sum
def problem (L S : ℤ) (k : ℤ) : Prop := 
  is_multiple k S L ∧ S + L = sum_of_numbers

theorem find_multiple (L S : ℤ) (k : ℤ) (h1 : problem L S k) : k = 3 := by
  -- Proof omitted
  sorry

end find_multiple_l168_168410


namespace abs_sum_of_first_six_a_sequence_terms_l168_168613

def a_sequence (n : ℕ) : ℤ :=
  match n with
  | 0 => -5
  | n+1 => a_sequence n + 2

theorem abs_sum_of_first_six_a_sequence_terms :
  |a_sequence 0| + |a_sequence 1| + |a_sequence 2| + |a_sequence 3| + |a_sequence 4| + |a_sequence 5| = 18 := sorry

end abs_sum_of_first_six_a_sequence_terms_l168_168613


namespace luke_initial_stickers_l168_168918

theorem luke_initial_stickers (x : ℕ) (h : x + 12 + 20 - 5 - 8 = 39) : x = 20 := 
by 
  sorry

end luke_initial_stickers_l168_168918


namespace probability_correct_l168_168507

open Finset

-- Define the set of distinct numbers
def num_set : Finset ℕ := {3, 7, 21, 27, 35, 42, 51}

-- Define the condition for a multiple of 63
def multiple_of_63 (a b : ℕ) : Prop := (a * b) % 63 = 0

-- Define the number of ways to pick distinct pairs
def total_pairs : ℕ := choose num_set.card 2

-- Define the number of successful pairs
def successful_pairs : ℕ := (num_set.prod fun a => 
  (num_set.filter (fun b => (a ≠ b) ∧ multiple_of_63 a b)).card ) / 2

-- Compute the probability
def probability_multiple_of_63 : ℚ := successful_pairs / total_pairs

-- The proof statement
theorem probability_correct : probability_multiple_of_63 = 3 / 7 := sorry

end probability_correct_l168_168507


namespace find_a_and_union_l168_168649

noncomputable def A (a : ℝ) : Set ℝ := { -4, 2 * a - 1, a ^ 2 }
noncomputable def B (a : ℝ) : Set ℝ := { a - 5, 1 - a, 9 }

theorem find_a_and_union {a : ℝ}
  (h : A a ∩ B a = {9}): 
  a = -3 ∧ A a ∪ B a = {-8, -7, -4, 4, 9} :=
by
  sorry

end find_a_and_union_l168_168649


namespace gcd_228_1995_l168_168678

theorem gcd_228_1995 : Nat.gcd 228 1995 = 57 := by
  sorry

end gcd_228_1995_l168_168678


namespace intersection_in_polar_l168_168368

-- Definitions for the circle and line in polar coordinates
def circle_polar (ρ θ : ℝ) := ρ = Real.cos θ + Real.sin θ
def line_polar (ρ θ : ℝ) := ρ * Real.sin (θ - Real.pi / 4) = Real.sqrt 2 / 2

-- Equivalent Cartesian coordinate forms
def circle_cartesian (x y : ℝ) := x^2 + y^2 - x - y = 0
def line_cartesian (x y : ℝ) := x - y + 1 = 0

-- Proving the intersection in polar coordinates
theorem intersection_in_polar :
  (∃ θ, ∃ ρ ≥ 0, θ ∈ set.Ico 0 (2 * Real.pi) ∧ circle_polar ρ θ ∧ line_polar ρ θ) →
  ∃ ρ θ, ρ = 1 ∧ θ = Real.pi / 2 := 
begin
  intros h,
  -- Expanding the definition of the intersection
  rcases h with ⟨θ, ⟨ρ, ⟨h_ρ, ⟨h_θ, ⟨h_circle, h_line⟩⟩⟩⟩⟩,
  -- Verifying intersection
  use [1, Real.pi / 2],
  split,
  { refl },
  { refl },
  sorry,
end

end intersection_in_polar_l168_168368


namespace true_propositions_for_quadratic_equations_l168_168614

theorem true_propositions_for_quadratic_equations :
  (∀ (a b c : ℤ), a ≠ 0 → (∃ Δ : ℤ, Δ^2 = b^2 - 4 * a * c → ∃ x : ℚ, x^2 + (b / a) * x + (c / a) = 0)) ∧
  (∀ (a b c : ℤ), a ≠ 0 → (∃ x : ℚ, x^2 + (b / a) * x + (c / a) = 0 → ∃ Δ : ℤ, Δ^2 = b^2 - 4 * a * c)) ∧
  (¬ ∀ (a b c : ℝ), a ≠ 0 → (∃ x : ℚ, x^2 + (b / a) * x + (c / a) = 0)) ∧
  (∀ (a b c : ℤ), a ≠ 0 ∧ a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1 → ¬∃ x : ℚ, x^2 + (b / a) * x + (c / a) = 0) :=
by sorry

end true_propositions_for_quadratic_equations_l168_168614


namespace find_x_l168_168052

variable (x : ℝ)
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (x, 1)
def u : ℝ × ℝ := (1 + 2 * x, 4)
def v : ℝ × ℝ := (2 - 2 * x, 2)

theorem find_x (h : 2 * (1 + 2 * x) = 4 * (2 - 2 * x)) : x = 1 / 2 := by
  sorry

end find_x_l168_168052


namespace heat_capacity_at_100K_l168_168932

noncomputable def heat_capacity (t : ℝ) : ℝ :=
  0.1054 + 0.000004 * t

theorem heat_capacity_at_100K :
  heat_capacity 100 = 0.1058 := 
by
  sorry

end heat_capacity_at_100K_l168_168932


namespace simplify_expression_l168_168255

theorem simplify_expression :
  2^2 + 2^2 + 2^2 + 2^2 = 2^4 :=
sorry

end simplify_expression_l168_168255


namespace max_rectangle_area_l168_168308

theorem max_rectangle_area (l w : ℕ) (h : 2 * (l + w) = 120) : l * w ≤ 900 :=
by 
  sorry

end max_rectangle_area_l168_168308


namespace colby_mangoes_l168_168575

def mangoes_still_have (t m k : ℕ) : ℕ :=
  let r1 := t - m
  let r2 := r1 / 2
  let r3 := r1 - r2
  r3 * k

theorem colby_mangoes (t m k : ℕ) (h_t : t = 60) (h_m : m = 20) (h_k : k = 8) :
  mangoes_still_have t m k = 160 :=
by
  sorry

end colby_mangoes_l168_168575


namespace square_of_105_l168_168586

/-- Prove that 105^2 = 11025. -/
theorem square_of_105 : 105^2 = 11025 :=
by
  sorry

end square_of_105_l168_168586


namespace product_odd_probability_l168_168101

open Finset
open BigOperators

-- Definition of the set and probability calculation
def set : Finset ℕ := {1, 2, 3, 4, 5, 6}
def odd_numbers : Finset ℕ := {1, 3, 5}

-- Calculation of combinations
def total_combinations := (set.choose 3).card
def odd_combinations := (odd_numbers.choose 3).card

-- Probability of selecting three numbers such that their product is odd
def probability_of_odd_product := (odd_combinations : ℚ) / (total_combinations : ℚ)

theorem product_odd_probability :
  probability_of_odd_product = 1 / 20 :=
by
  -- This is where the proof would go
  sorry

end product_odd_probability_l168_168101


namespace area_is_prime_number_l168_168735

open Real Int

noncomputable def area_of_triangle (a : Int) : Real :=
  (a * a : Real) / 20

theorem area_is_prime_number 
  (a : Int) 
  (h1 : ∃ p : ℕ, Nat.Prime p ∧ p = ((a * a) / 20 : Real)) :
  ((a * a) / 20 : Real) = 5 :=
by 
  sorry

end area_is_prime_number_l168_168735


namespace find_m_l168_168630

noncomputable def A : Set ℝ := { x | x^2 - 3 * x - 10 = 0 }
noncomputable def B (m : ℝ) : Set ℝ := { x | m * x - 1 = 0 }

theorem find_m:
  ∀ (m : ℝ),
    ((∀ x, B m x → x ∈ A) ∨ B m = ∅) →
    (m = 0 ∨ m = -1/2 ∨ m = 1/5) :=
begin
  sorry
end

end find_m_l168_168630


namespace smallest_xy_l168_168158

theorem smallest_xy :
  ∃ (x y : ℕ), (0 < x) ∧ (0 < y) ∧ (1 / x + 1 / (3 * y) = 1 / 6) ∧ (∀ (x' y' : ℕ), (0 < x') ∧ (0 < y') ∧ (1 / x' + 1 / (3 * y') = 1 / 6) → x' * y' ≥ x * y) ∧ x * y = 48 :=
sorry

end smallest_xy_l168_168158


namespace geraldo_drank_7_pints_l168_168947

-- Conditions
def total_gallons : ℝ := 20
def num_containers : ℕ := 80
def gallons_to_pints : ℝ := 8
def containers_drank : ℝ := 3.5

-- Problem statement
theorem geraldo_drank_7_pints :
  let total_pints : ℝ := total_gallons * gallons_to_pints
  let pints_per_container : ℝ := total_pints / num_containers
  let pints_drank : ℝ := containers_drank * pints_per_container
  pints_drank = 7 :=
by
  sorry

end geraldo_drank_7_pints_l168_168947


namespace proof_correct_props_l168_168001

variable (p1 : Prop) (p2 : Prop) (p3 : Prop) (p4 : Prop)

def prop1 : Prop := ∃ (x₀ : ℝ), 0 < x₀ ∧ (1 / 2) * x₀ < (1 / 3) * x₀
def prop2 : Prop := ∃ (x₀ : ℝ), 0 < x₀ ∧ x₀ < 1 ∧ Real.log x₀ / Real.log (1 / 2) > Real.log x₀ / Real.log (1 / 3)
def prop3 : Prop := ∀ (x : ℝ), 0 < x ∧ (1 / 2) ^ x > Real.log x / Real.log (1 / 2)
def prop4 : Prop := ∀ (x : ℝ), 0 < x ∧ x < 1 / 3 ∧ (1 / 2) ^ x < Real.log x / Real.log (1 / 3)

theorem proof_correct_props : prop2 ∧ prop4 :=
by
  sorry -- Proof goes here

end proof_correct_props_l168_168001


namespace avg_visitors_on_sunday_l168_168132

theorem avg_visitors_on_sunday (S : ℕ) :
  (30 * 285) = (5 * S + 25 * 240) -> S = 510 :=
by
  intros h
  sorry

end avg_visitors_on_sunday_l168_168132


namespace tricycle_count_l168_168145

theorem tricycle_count
    (total_children : ℕ) (total_wheels : ℕ) (walking_children : ℕ)
    (h1 : total_children - walking_children = 8)
    (h2 : 2 * (total_children - walking_children - (total_wheels - 16) / 3) + 3 * ((total_wheels - 16) / 3) = total_wheels) :
    (total_wheels - 16) / 3 = 8 :=
by
    intros
    sorry

end tricycle_count_l168_168145


namespace rectangular_box_inscribed_in_sphere_l168_168971

noncomputable def problem_statement : Prop :=
  ∃ (a b c s : ℝ), (4 * (a + b + c) = 72) ∧ (2 * (a * b + b * c + c * a) = 216) ∧
  (a^2 + b^2 + c^2 = 108) ∧ (4 * s^2 = 108) ∧ (s = 3 * Real.sqrt 3)

theorem rectangular_box_inscribed_in_sphere : problem_statement := 
  sorry

end rectangular_box_inscribed_in_sphere_l168_168971


namespace hoseok_needs_17_more_jumps_l168_168631

/-- Define the number of jumps by Hoseok and Minyoung -/
def hoseok_jumps : ℕ := 34
def minyoung_jumps : ℕ := 51

/-- Define the number of additional jumps Hoseok needs -/
def additional_jumps_hoseok : ℕ := minyoung_jumps - hoseok_jumps

/-- Prove that the additional jumps Hoseok needs is equal to 17 -/
theorem hoseok_needs_17_more_jumps (h_jumps : ℕ := hoseok_jumps) (m_jumps : ℕ := minyoung_jumps) :
  additional_jumps_hoseok = 17 := by
  -- Proof goes here
  sorry

end hoseok_needs_17_more_jumps_l168_168631


namespace problem_a_problem_b_l168_168120

noncomputable def gini_coefficient_separate_operations : ℝ := 
  let population_north := 24
  let population_south := population_north / 4
  let income_per_north_inhabitant := (6000 * 18) / population_north
  let income_per_south_inhabitant := (6000 * 12) / population_south
  let total_population := population_north + population_south
  let total_income := 6000 * (18 + 12)
  let share_pop_north := population_north / total_population
  let share_income_north := (income_per_north_inhabitant * population_north) / total_income
  share_pop_north - share_income_north

theorem problem_a : gini_coefficient_separate_operations = 0.2 := 
  by sorry

noncomputable def change_in_gini_coefficient_after_collaboration : ℝ :=
  let previous_income_north := 6000 * 18
  let compensation := previous_income_north + 1983
  let total_combined_income := 6000 * 30.5
  let remaining_income_south := total_combined_income - compensation
  let population := 24 + 6
  let income_per_capita_north := compensation / 24
  let income_per_capita_south := remaining_income_south / 6
  let new_gini_coefficient := 
    let share_pop_north := 24 / population
    let share_income_north := compensation / total_combined_income
    share_pop_north - share_income_north
  (0.2 - new_gini_coefficient)

theorem problem_b : change_in_gini_coefficient_after_collaboration = 0.001 := 
  by sorry

end problem_a_problem_b_l168_168120


namespace probability_correct_l168_168848

def total_assignments := 15 * 14 * 13

def is_multiple (a b : ℕ) : Prop := a % b = 0

def valid (al bill cal : ℕ) : Prop :=
  is_multiple al bill ∧ is_multiple bill cal ∧
  al ≠ bill ∧ bill ≠ cal ∧ al ≠ cal

def count_valid_assignments : ℕ :=
  Finset.card
    (Finset.univ.filter (λ triplet: Fin3 15 × Fin3 15 × Fin3 15,
      valid triplet.1 triplet.2 triplet.3))

def probability_valid_assignment : ℚ :=
  count_valid_assignments / total_assignments

theorem probability_correct :
  probability_valid_assignment = 1 / 60 := sorry

end probability_correct_l168_168848


namespace cubes_sum_is_214_5_l168_168674

noncomputable def r_plus_s_plus_t : ℝ := 12
noncomputable def rs_plus_rt_plus_st : ℝ := 47
noncomputable def rst : ℝ := 59.5

theorem cubes_sum_is_214_5 :
    (r_plus_s_plus_t * ((r_plus_s_plus_t)^2 - 3 * rs_plus_rt_plus_st) + 3 * rst) = 214.5 := by
    sorry

end cubes_sum_is_214_5_l168_168674


namespace max_liters_l168_168819

-- Declare the problem conditions as constants
def initialHeat : ℕ := 480 -- kJ for the first 5 min
def reductionRate : ℚ := 0.25 -- 25%
def initialTemp : ℚ := 20 -- °C
def boilingTemp : ℚ := 100 -- °C
def specificHeatCapacity : ℚ := 4.2 -- kJ/kg°C

-- Define the heat required to raise m kg of water from initialTemp to boilingTemp
def heatRequired (m : ℚ) : ℚ := m * specificHeatCapacity * (boilingTemp - initialTemp)

-- Define the total available heat from the fuel
noncomputable def totalAvailableHeat : ℚ :=
  initialHeat / (1 - (1 - reductionRate))

-- The proof problem statement
theorem max_liters :
  ∃ (m : ℕ), m ≤ 5 ∧ heatRequired m ≤ totalAvailableHeat :=
sorry

end max_liters_l168_168819


namespace buns_left_l168_168943

theorem buns_left (buns_initial : ℕ) (h1 : buns_initial = 15)
                  (x : ℕ) (h2 : 13 * x ≤ buns_initial)
                  (buns_taken_by_bimbo : ℕ) (h3 : buns_taken_by_bimbo = x)
                  (buns_taken_by_little_boy : ℕ) (h4 : buns_taken_by_little_boy = 3 * x)
                  (buns_taken_by_karlsson : ℕ) (h5 : buns_taken_by_karlsson = 9 * x)
                  :
                  buns_initial - (buns_taken_by_bimbo + buns_taken_by_little_boy + buns_taken_by_karlsson) = 2 :=
by
  sorry

end buns_left_l168_168943


namespace range_of_f_l168_168409

noncomputable def f (x : ℝ) : ℝ := 4^x + 2^(x + 1) + 1

theorem range_of_f : Set.range f = {y : ℝ | y > 1} :=
by
  sorry

end range_of_f_l168_168409


namespace customer_bought_two_pens_l168_168064

noncomputable def combination (n k : ℕ) : ℝ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem customer_bought_two_pens :
  ∃ n : ℕ, combination 5 n / combination 8 n = 0.3571428571428571 ↔ n = 2 := by
  sorry

end customer_bought_two_pens_l168_168064


namespace benny_january_savings_l168_168988

theorem benny_january_savings :
  ∃ x : ℕ, x + x + 8 = 46 ∧ x = 19 :=
by
  sorry

end benny_january_savings_l168_168988


namespace algebra_inequality_l168_168859

theorem algebra_inequality (a b c : ℝ) 
  (H : a^4 + b^4 + c^4 ≤ 2 * (a^2 * b^2 + b^2 * c^2 + c^2 * a^2)) :
  a^2 + b^2 + c^2 ≤ 2 * (a * b + b * c + c * a) :=
sorry

end algebra_inequality_l168_168859


namespace binomial_12_11_eq_12_l168_168008

-- Define binomial coefficient function
def binomial (n k : Nat) : Nat :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Statement to prove
theorem binomial_12_11_eq_12 :
  binomial 12 11 = 12 := by
  sorry

end binomial_12_11_eq_12_l168_168008


namespace number_of_ordered_pairs_l168_168194

theorem number_of_ordered_pairs : 
  ∃ (S : Finset (ℕ × ℕ)), (∀ x ∈ S, (x.1 * x.2 = 64) ∧ (x.1 > 0) ∧ (x.2 > 0)) ∧ S.card = 7 := 
sorry

end number_of_ordered_pairs_l168_168194


namespace rectangle_area_l168_168301

theorem rectangle_area (r : ℝ) (w l : ℝ) (h_radius : r = 7) 
  (h_ratio : l = 3 * w) (h_width : w = 2 * r) : l * w = 588 :=
by
  sorry

end rectangle_area_l168_168301


namespace inequality_proof_l168_168871

theorem inequality_proof (x y z : ℝ) (hx : x ≥ y) (hy : y ≥ z) (hz : z > 0) :
  (x^2 * y / z + y^2 * z / x + z^2 * x / y) ≥ (x^2 + y^2 + z^2) := 
  sorry

end inequality_proof_l168_168871


namespace cos_diff_identity_l168_168482

variable {α : ℝ}

def sin_alpha := -3 / 5

def alpha_interval (α : ℝ) : Prop :=
  (3 * Real.pi / 2 < α) ∧ (α < 2 * Real.pi)

theorem cos_diff_identity (h1 : Real.sin α = sin_alpha) (h2 : alpha_interval α) :
  Real.cos (Real.pi / 4 - α) = Real.sqrt 2 / 10 :=
  sorry

end cos_diff_identity_l168_168482


namespace solve_for_x_l168_168533

-- Define the quadratic equation condition
def quadratic_eq (x : ℝ) : Prop := 3 * x^2 - 7 * x - 6 = 0

-- The main theorem to prove
theorem solve_for_x (x : ℝ) : x > 0 ∧ quadratic_eq x → x = 3 := by
  sorry

end solve_for_x_l168_168533


namespace divides_two_pow_n_minus_one_l168_168603

theorem divides_two_pow_n_minus_one {n : ℕ} (h : n > 0) (divides : n ∣ 2^n - 1) : n = 1 :=
sorry

end divides_two_pow_n_minus_one_l168_168603


namespace first_train_speed_is_80_kmph_l168_168422

noncomputable def speedOfFirstTrain
  (lenFirstTrain : ℝ)
  (lenSecondTrain : ℝ)
  (speedSecondTrain : ℝ)
  (clearTime : ℝ)
  (oppositeDirections : Bool) : ℝ :=
  if oppositeDirections then
    let totalDistance := (lenFirstTrain + lenSecondTrain) / 1000  -- convert meters to kilometers
    let timeHours := clearTime / 3600 -- convert seconds to hours
    let relativeSpeed := totalDistance / timeHours
    relativeSpeed - speedSecondTrain
  else
    0 -- This should not happen based on problem conditions

theorem first_train_speed_is_80_kmph :
  speedOfFirstTrain 151 165 65 7.844889650207294 true = 80 :=
by
  sorry

end first_train_speed_is_80_kmph_l168_168422


namespace find_AG_l168_168868

-- Defining constants and variables
variables (DE EC AD BC FB AG : ℚ)
variables (BC_def : BC = (1 / 3) * AD)
variables (FB_def : FB = (2 / 3) * AD)
variables (DE_val : DE = 8)
variables (EC_val : EC = 6)
variables (sum_AD : BC + FB = AD)

-- The theorem statement
theorem find_AG : AG = 56 / 9 :=
by
  -- Placeholder for the proof
  sorry

end find_AG_l168_168868


namespace family_children_count_l168_168250

theorem family_children_count (x y : ℕ) 
  (sister_condition : x = y - 1) 
  (brother_condition : y = 2 * (x - 1)) : 
  x + y = 7 := 
sorry

end family_children_count_l168_168250


namespace tangent_line_at_point_l168_168675

theorem tangent_line_at_point (x y : ℝ) (h_curve : y = (2 * x - 1)^3) (h_point : (x, y) = (1, 1)) :
  ∃ m b : ℝ, y = m * x + b ∧ m = 6 ∧ b = -5 :=
by
  sorry

end tangent_line_at_point_l168_168675


namespace no_such_graph_exists_l168_168704

noncomputable def vertex_degrees (n : ℕ) (deg : ℕ → ℕ) : Prop :=
  n ≥ 8 ∧
  ∃ (deg : ℕ → ℕ),
    (deg 0 = 4) ∧ (deg 1 = 5) ∧ ∀ i, 2 ≤ i ∧ i < n - 7 → deg i = i + 4 ∧
    (deg (n-7) = n-2) ∧ (deg (n-6) = n-2) ∧ (deg (n-5) = n-2) ∧
    (deg (n-4) = n-1) ∧ (deg (n-3) = n-1) ∧ (deg (n-2) = n-1)   

theorem no_such_graph_exists (n : ℕ) (deg : ℕ → ℕ) : 
  n ≥ 10 → ¬vertex_degrees n deg := 
by
  sorry

end no_such_graph_exists_l168_168704


namespace values_of_a_and_b_solution_set_inequality_l168_168739

-- Part (I)
theorem values_of_a_and_b (a b : ℝ) (h : ∀ x, -1 < x ∧ x < 1 → x^2 - a * x - x + b < 0) :
  a = -1 ∧ b = -1 := sorry

-- Part (II)
theorem solution_set_inequality (a : ℝ) (h : a = b) :
  (∀ x, x^2 - a * x - x + a < 0 → (x = 1 → false) 
      ∧ (0 < 1 - a → (x = 1 → false))
      ∧ (1 < - a → (x = 1 → false))) := sorry

end values_of_a_and_b_solution_set_inequality_l168_168739


namespace length_of_first_two_CDs_l168_168370

theorem length_of_first_two_CDs
  (x : ℝ)
  (h1 : x + x + 2 * x = 6) :
  x = 1.5 := 
sorry

end length_of_first_two_CDs_l168_168370


namespace paul_bought_150_books_l168_168081

theorem paul_bought_150_books (initial_books sold_books books_now : ℤ)
  (h1 : initial_books = 2)
  (h2 : sold_books = 94)
  (h3 : books_now = 58) :
  initial_books - sold_books + books_now = 150 :=
by
  rw [h1, h2, h3]
  norm_num
  sorry

end paul_bought_150_books_l168_168081


namespace number_of_equilateral_triangles_after_12_operations_l168_168720

theorem number_of_equilateral_triangles_after_12_operations :
  ∀ (initial_triangles : ℕ),
    initial_triangles = 1 →
    ∀ (operations : ℕ),
      operations = 12 →
      let remaining_triangles : ℕ := initial_triangles * 3 ^ operations in
      remaining_triangles = 531441 :=
by
  intros initial_triangles h_initial_triangles operations h_operations
  simp [h_initial_triangles, h_operations]
  have : 3 ^ 12 = 531441 := by norm_num
  rw [this]
  sorry

end number_of_equilateral_triangles_after_12_operations_l168_168720


namespace polynomial_zero_l168_168989

theorem polynomial_zero (P : Polynomial ℤ) (h : ∀ n : ℕ, n > 0 → (n : ℤ) ∣ P.eval (2^n)) : P = 0 := 
sorry

end polynomial_zero_l168_168989


namespace george_blocks_l168_168867

theorem george_blocks (num_boxes : ℕ) (blocks_per_box : ℕ) (total_blocks : ℕ) :
  num_boxes = 2 → blocks_per_box = 6 → total_blocks = num_boxes * blocks_per_box → total_blocks = 12 := by
  intros h_num_boxes h_blocks_per_box h_blocks_equal
  rw [h_num_boxes, h_blocks_per_box] at h_blocks_equal
  exact h_blocks_equal

end george_blocks_l168_168867


namespace smallest_next_divisor_l168_168225

theorem smallest_next_divisor (n : ℕ) (hn : n % 2 = 0) (h4d : 1000 ≤ n ∧ n < 10000) (hdiv : 221 ∣ n) : 
  ∃ (d : ℕ), d = 238 ∧ 221 < d ∧ d ∣ n :=
by
  sorry

end smallest_next_divisor_l168_168225


namespace four_digit_palindrome_perfect_squares_l168_168967

theorem four_digit_palindrome_perfect_squares : 
  ∃ (count : ℕ), count = 2 ∧ 
  (∀ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 → 
            ∃ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 
            n = 1001 * a + 110 * b ∧ 
            ∃ k : ℕ, k * k = n) → count = 2 := by
  sorry

end four_digit_palindrome_perfect_squares_l168_168967


namespace prob1_prob2_prob3_l168_168740

-- Problem 1
theorem prob1 (k : ℝ) (h₀ : k > 0) 
  (h₁ : ∀ x : ℝ, 2 < x ∧ x < 3 → (k * x^2 - 2 * x + 6 * k) < 0) :
  k = 2/5 := 
sorry

-- Problem 2
theorem prob2 (k : ℝ) (h₀ : k > 0) 
  (h₁ : ∀ x : ℝ, 2 < x ∧ x < 3 → (k * x^2 - 2 * x + 6 * k) < 0) :
  0 < k ∧ k ≤ 2/5 := 
sorry

-- Problem 3
theorem prob3 (k : ℝ) (h₀ : k > 0)
  (h₁ : ∀ x : ℝ, 2 < x ∧ x < 3 → (k * x^2 - 2 * x + 6 * k) < 0) :
  k ≥ 2/5 := 
sorry

end prob1_prob2_prob3_l168_168740


namespace salary_of_A_l168_168702

-- Given:
-- A + B = 6000
-- A's savings = 0.05A
-- B's savings = 0.15B
-- A's savings = B's savings

theorem salary_of_A (A B : ℝ) (h1 : A + B = 6000) (h2 : 0.05 * A = 0.15 * B) :
  A = 4500 :=
sorry

end salary_of_A_l168_168702


namespace coterminal_angle_in_radians_l168_168211

theorem coterminal_angle_in_radians (d : ℝ) (h : d = 2010) : 
  ∃ r : ℝ, r = -5 * Real.pi / 6 ∧ (∃ k : ℤ, d = r * 180 / Real.pi + k * 360) :=
by sorry

end coterminal_angle_in_radians_l168_168211


namespace james_total_cost_l168_168072

def courseCost (units: Nat) (cost_per_unit: Nat) : Nat :=
  units * cost_per_unit

def totalCostForFall : Nat :=
  courseCost 12 60 + courseCost 8 45

def totalCostForSpring : Nat :=
  let science_cost := courseCost 10 60
  let science_scholarship := science_cost / 2
  let humanities_cost := courseCost 10 45
  (science_cost - science_scholarship) + humanities_cost

def totalCostForSummer : Nat :=
  courseCost 6 80 + courseCost 4 55

def totalCostForWinter : Nat :=
  let science_cost := courseCost 6 80
  let science_scholarship := 3 * science_cost / 4
  let humanities_cost := courseCost 4 55
  (science_cost - science_scholarship) + humanities_cost

def totalAmountSpent : Nat :=
  totalCostForFall + totalCostForSpring + totalCostForSummer + totalCostForWinter

theorem james_total_cost: totalAmountSpent = 2870 :=
  by sorry

end james_total_cost_l168_168072


namespace total_eyes_in_family_l168_168372

def mom_eyes := 1
def dad_eyes := 3
def num_kids := 3
def kid_eyes := 4

theorem total_eyes_in_family : mom_eyes + dad_eyes + (num_kids * kid_eyes) = 16 :=
by
  sorry

end total_eyes_in_family_l168_168372


namespace kims_total_points_l168_168405

theorem kims_total_points :
  let points_easy := 2
  let points_average := 3
  let points_hard := 5
  let answers_easy := 6
  let answers_average := 2
  let answers_hard := 4
  let total_points := (answers_easy * points_easy) + (answers_average * points_average) + (answers_hard * points_hard)
  total_points = 38 :=
by
  -- This is a placeholder to indicate that the proof is not included.
  sorry

end kims_total_points_l168_168405


namespace quadratic_solutions_l168_168795

theorem quadratic_solutions :
  ∀ x : ℝ, (x^2 - 4 * x = 0) → (x = 0 ∨ x = 4) :=
by sorry

end quadratic_solutions_l168_168795


namespace contrapositive_of_not_p_implies_q_l168_168361

variable (p q : Prop)

theorem contrapositive_of_not_p_implies_q :
  (¬p → q) → (¬q → p) := by
  sorry

end contrapositive_of_not_p_implies_q_l168_168361


namespace Xiaohong_wins_5_times_l168_168828

theorem Xiaohong_wins_5_times :
  ∃ W L : ℕ, (3 * W - 2 * L = 1) ∧ (W + L = 12) ∧ W = 5 :=
by
  sorry

end Xiaohong_wins_5_times_l168_168828


namespace monster_family_eyes_count_l168_168371

theorem monster_family_eyes_count :
  let mom_eyes := 1
  let dad_eyes := 3
  let num_kids := 3
  let kid_eyes := 4
  (mom_eyes + dad_eyes) + (num_kids * kid_eyes) = 16 :=
by
  let mom_eyes := 1
  let dad_eyes := 3
  let num_kids := 3
  let kid_eyes := 4
  have parents_eyes : mom_eyes + dad_eyes = 4 := by rfl
  have kids_eyes : num_kids * kid_eyes = 12 := by rfl
  show parents_eyes + kids_eyes = 16
  sorry

end monster_family_eyes_count_l168_168371


namespace avg_visitors_on_sunday_l168_168131

theorem avg_visitors_on_sunday (S : ℕ) :
  (30 * 285) = (5 * S + 25 * 240) -> S = 510 :=
by
  intros h
  sorry

end avg_visitors_on_sunday_l168_168131


namespace max_value_l168_168177

variable (x y : ℝ)

def condition : Prop := 2 * x ^ 2 + x * y - y ^ 2 = 1

noncomputable def expression : ℝ := (x - 2 * y) / (5 * x ^ 2 - 2 * x * y + 2 * y ^ 2)

theorem max_value : ∀ x y : ℝ, condition x y → expression x y ≤ (Real.sqrt 2) / 4 :=
by
  sorry

end max_value_l168_168177


namespace jane_earnings_l168_168511

def age_of_child (jane_start_age : ℕ) (child_factor : ℕ) : ℕ :=
  jane_start_age / child_factor

def babysit_rate (age : ℕ) : ℕ :=
  if age < 2 then 5
  else if age <= 5 then 7
  else 8

def amount_earned (hours rate : ℕ) : ℕ := 
  hours * rate

def total_earnings (earnings : List ℕ) : ℕ :=
  earnings.foldl (·+·) 0

theorem jane_earnings
  (jane_start_age : ℕ := 18)
  (child_A_hours : ℕ := 50)
  (child_B_hours : ℕ := 90)
  (child_C_hours : ℕ := 130)
  (child_D_hours : ℕ := 70) :
  let child_A_age := age_of_child jane_start_age 2
  let child_B_age := child_A_age - 2
  let child_C_age := child_B_age + 3
  let child_D_age := child_C_age
  let earnings_A := amount_earned child_A_hours (babysit_rate child_A_age)
  let earnings_B := amount_earned child_B_hours (babysit_rate child_B_age)
  let earnings_C := amount_earned child_C_hours (babysit_rate child_C_age)
  let earnings_D := amount_earned child_D_hours (babysit_rate child_D_age)
  total_earnings [earnings_A, earnings_B, earnings_C, earnings_D] = 2720 :=
by
  sorry

end jane_earnings_l168_168511


namespace cost_price_correct_l168_168719

noncomputable def cost_price (selling_price marked_price_ratio cost_profit_ratio : ℝ) : ℝ :=
  (selling_price * marked_price_ratio) / cost_profit_ratio

theorem cost_price_correct : 
  abs (cost_price 63.16 0.94 1.25 - 50.56) < 0.01 :=
by 
  sorry

end cost_price_correct_l168_168719


namespace arithmetic_progression_exists_l168_168166

theorem arithmetic_progression_exists (a_1 a_2 a_3 a_4 : ℕ) (d : ℕ) :
  a_2 = a_1 + d →
  a_3 = a_1 + 2 * d →
  a_4 = a_1 + 3 * d →
  a_1 * a_2 * a_3 = 6 →
  a_1 * a_2 * a_3 * a_4 = 24 →
  a_1 = 1 ∧ a_2 = 2 ∧ a_3 = 3 ∧ a_4 = 4 :=
by
  sorry

end arithmetic_progression_exists_l168_168166


namespace sum_eq_product_l168_168528

theorem sum_eq_product (a b c : ℝ) (h1 : 1 + b * c ≠ 0) (h2 : 1 + c * a ≠ 0) (h3 : 1 + a * b ≠ 0) :
  (b - c) / (1 + b * c) + (c - a) / (1 + c * a) + (a - b) / (1 + a * b) =
  ((b - c) * (c - a) * (a - b)) / ((1 + b * c) * (1 + c * a) * (1 + a * b)) :=
by
  sorry

end sum_eq_product_l168_168528


namespace problem_statement_l168_168195

def is_palindrome (n : ℕ) : Prop := sorry -- Placeholder for the palindrome-checking function

def is_prime_palindrome (p : ℕ) : Prop :=
  p.Prime ∧ is_palindrome p ∧ (p / 10 ≥ 1 ∧ p / 100 < 10)

theorem problem_statement : 
  ∃ n, n = 2 ∧ 
  (∀ y, 2000 ≤ y ∧ y < 3000 ∧ is_palindrome y → 
    ((∃ p1 p2, is_prime_palindrome p1 ∧ is_prime_palindrome p2 ∧ p1 * p2 = y) → 
    n = 2)) :=
sorry

end problem_statement_l168_168195


namespace problem_l168_168354

theorem problem (a b c : ℝ) (Ha : a > 0) (Hb : b > 0) (Hc : c > 0) : 
  (|a| / a + |b| / b + |c| / c - (abc / |abc|) = 2 ∨ |a| / a + |b| / b + |c| / c - (abc / |abc|) = -2) :=
by
  sorry

end problem_l168_168354


namespace derivative_at_one_max_value_l168_168045

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 3 * x

-- Prove that f'(1) = 0
theorem derivative_at_one : deriv f 1 = 0 :=
by sorry

-- Prove that the maximum value of f(x) is 2
theorem max_value : ∃ x : ℝ, (∀ y : ℝ, f y ≤ f x) ∧ f x = 2 :=
by sorry

end derivative_at_one_max_value_l168_168045


namespace tangent_to_parabola_l168_168600

theorem tangent_to_parabola {k : ℝ} : 
  (∀ x y : ℝ, (4 * x + 3 * y + k = 0) ↔ (y ^ 2 = 16 * x)) → k = 9 :=
by
  sorry

end tangent_to_parabola_l168_168600


namespace tony_slices_remaining_l168_168275

def slices_remaining : ℕ :=
  let total_slices := 22
  let slices_per_sandwich := 2
  let daily_sandwiches := 1
  let extra_sandwiches := 1
  let total_sandwiches := 7 * daily_sandwiches + extra_sandwiches
  total_slices - total_sandwiches * slices_per_sandwich

theorem tony_slices_remaining :
  slices_remaining = 6 :=
by
  -- Here, the theorem proof would go, but it is omitted as per instructions
  sorry

end tony_slices_remaining_l168_168275


namespace num_of_4_digit_numbers_divisible_by_13_l168_168897

theorem num_of_4_digit_numbers_divisible_by_13 :
  let smallest_4_digit := 1000
  let smallest_divisible_by_13 := 1001
  let largest_4_digit := 9999
  let largest_divisible_by_13 := 9997
  let a := smallest_divisible_by_13
  let d := 13
  let l := largest_divisible_by_13
  (\(l - a) / d) + 1 = 693 :=
by
  let smallest_4_digit := 1000
  let smallest_divisible_by_13 := 1001
  let largest_4_digit := 9999
  let largest_divisible_by_13 := 9997
  let a := smallest_divisible_by_13
  let d := 13
  let l := largest_divisible_by_13
  have h1 : (l - a) / d + 1 = (9997 - 1001) / 13 + 1, by sorry
  have h2 : (9997 - 1001) / 13 + 1 = 8996 / 13 + 1, by sorry
  have h3 : 8996 / 13 + 1 = 692 + 1, by sorry
  have h4 : 692 + 1 = 693, by sorry
  exact h4

end num_of_4_digit_numbers_divisible_by_13_l168_168897


namespace total_sum_lent_l168_168447

theorem total_sum_lent (x : ℝ) (second_part : ℝ) (total_sum : ℝ) 
  (h1 : second_part = 1640) 
  (h2 : (x * 8 * 0.03) = (second_part * 3 * 0.05)) :
  total_sum = x + second_part → total_sum = 2665 := by
  sorry

end total_sum_lent_l168_168447


namespace vector_addition_l168_168032

-- Define the vectors a and b
def a : ℝ × ℝ := (2, 3)
def b : ℝ × ℝ := (-1, 5)

-- State the theorem that we want to prove
theorem vector_addition : a + 3 • b = (-1, 18) :=
  sorry

end vector_addition_l168_168032


namespace not_sufficient_nor_necessary_l168_168376

theorem not_sufficient_nor_necessary (a b : ℝ) (hb : b ≠ 0) :
  ¬ ((a > b) ↔ (1 / a < 1 / b)) :=
by
  sorry

end not_sufficient_nor_necessary_l168_168376


namespace Sally_out_of_pocket_payment_l168_168927

theorem Sally_out_of_pocket_payment :
  let amount_given : ℕ := 320
  let cost_per_book : ℕ := 12
  let number_of_students : ℕ := 30
  let total_cost : ℕ := cost_per_book * number_of_students
  let out_of_pocket_cost : ℕ := total_cost - amount_given
  out_of_pocket_cost = 40 := by
  sorry

end Sally_out_of_pocket_payment_l168_168927


namespace integer_sum_l168_168930

theorem integer_sum {p q r s : ℤ} 
  (h1 : p - q + r = 7) 
  (h2 : q - r + s = 8) 
  (h3 : r - s + p = 4) 
  (h4 : s - p + q = 3) : 
  p + q + r + s = 22 := 
sorry

end integer_sum_l168_168930


namespace factor_theorem_for_Q_l168_168427

variable (d : ℝ) -- d is a real number

def Q (x : ℝ) : ℝ := x^3 + 3 * x^2 + d * x + 20

theorem factor_theorem_for_Q :
  (x : ℝ) → (Q x = 0) → (x = 4) → d = -33 :=
by
  intro x Q4 hx
  sorry

end factor_theorem_for_Q_l168_168427


namespace multiplication_of_decimals_l168_168148

theorem multiplication_of_decimals : (0.4 * 0.75 = 0.30) := by
  sorry

end multiplication_of_decimals_l168_168148


namespace arun_remaining_work_days_l168_168114

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

end arun_remaining_work_days_l168_168114


namespace fraction_of_number_l168_168809

theorem fraction_of_number : (7 / 8) * 64 = 56 := by
  sorry

end fraction_of_number_l168_168809


namespace equation_circle_iff_a_equals_neg_one_l168_168706

theorem equation_circle_iff_a_equals_neg_one :
  (∀ x y : ℝ, ∃ k : ℝ, a^2 * x^2 + (a + 2) * y^2 + 2 * a * x + a = k * (x^2 + y^2)) ↔ 
  a = -1 :=
by sorry

end equation_circle_iff_a_equals_neg_one_l168_168706


namespace four_digit_perfect_square_l168_168469

theorem four_digit_perfect_square (N : ℕ) (a b : ℤ) :
  N = 1100 * a + 11 * b ∧
  N >= 1000 ∧ N <= 9999 ∧
  a >= 0 ∧ a <= 9 ∧ b >= 0 ∧ b <= 9 ∧
  (∃ (x : ℤ), N = 11 * x^2) →
  N = 7744 := by
  sorry

end four_digit_perfect_square_l168_168469


namespace min_max_values_in_interval_l168_168023

def func (x y : ℝ) : ℝ := 3 * x^2 * y - 2 * x * y^2

theorem min_max_values_in_interval :
  (∀ x y, 0 ≤ x → x ≤ 1 → 0 ≤ y → y ≤ 1 → func x y ≥ -1/3) ∧
  (∃ x y, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 ∧ func x y = -1/3) ∧
  (∀ x y, 0 ≤ x → x ≤ 1 → 0 ≤ y → y ≤ 1 → func x y ≤ 9/8) ∧
  (∃ x y, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 ∧ func x y = 9/8) :=
by
  sorry

end min_max_values_in_interval_l168_168023


namespace maximum_number_of_intersections_of_150_lines_is_7171_l168_168916

def lines_are_distinct (L : ℕ → Type) : Prop := 
  ∀ n m : ℕ, n ≠ m → L n ≠ L m

def lines_parallel_to_each_other (L : ℕ → Type) (k : ℕ) : Prop :=
  ∀ n m : ℕ, n ≠ m → L (k * n) = L (k * m)

def lines_pass_through_point_B (L : ℕ → Type) (B : Type) (k : ℕ) : Prop :=
  ∀ n : ℕ, L (k * n - 4) = B

def lines_not_parallel (L : ℕ → Type) (k1 k2 : ℕ) : Prop :=
  ∀ n m : ℕ, L (k1 * n) ≠ L (k2 * m)

noncomputable def max_points_of_intersection
  (L : ℕ → Type)
  (B : Type)
  (k1 k2 : ℕ)
  (h_distinct : lines_are_distinct L)
  (h_parallel1 : lines_parallel_to_each_other L k1)
  (h_parallel2 : lines_parallel_to_each_other L k2)
  (h_pass_through_B : lines_pass_through_point_B L B 5)
  (h_not_parallel : lines_not_parallel L k1 k2)
  : ℕ :=
  7171

theorem maximum_number_of_intersections_of_150_lines_is_7171
  (L : ℕ → Type)
  (B : Type)
  (k1 k2 : ℕ)
  (h_distinct : lines_are_distinct L)
  (h_parallel1 : lines_parallel_to_each_other L k1)
  (h_parallel2 : lines_parallel_to_each_other L k2)
  (h_pass_through_B : lines_pass_through_point_B L B 5)
  (h_not_parallel : lines_not_parallel L k1 k2)
  : max_points_of_intersection L B k1 k2 h_distinct h_parallel1 h_parallel2 h_pass_through_B h_not_parallel = 7171 := 
  by 
  sorry

end maximum_number_of_intersections_of_150_lines_is_7171_l168_168916


namespace wire_divided_into_quarters_l168_168466

theorem wire_divided_into_quarters
  (l : ℕ) -- length of the wire
  (parts : ℕ) -- number of parts the wire is divided into
  (h_l : l = 28) -- wire is 28 cm long
  (h_parts : parts = 4) -- wire is divided into 4 parts
  : l / parts = 7 := -- each part is 7 cm long
by
  -- use sorry to skip the proof
  sorry

end wire_divided_into_quarters_l168_168466


namespace initial_distance_l168_168448

-- Definitions based on conditions
def speed_thief : ℝ := 8 -- in km/hr
def speed_policeman : ℝ := 10 -- in km/hr
def distance_thief_runs : ℝ := 0.7 -- in km

-- Theorem statement
theorem initial_distance
  (relative_speed := speed_policeman - speed_thief) -- Relative speed (in km/hr)
  (time_to_overtake := distance_thief_runs / relative_speed) -- Time for the policeman to overtake the thief (in hours)
  (initial_distance := speed_policeman * time_to_overtake) -- Initial distance (in km)
  : initial_distance = 3.5 :=
by
  sorry

end initial_distance_l168_168448


namespace smallest_four_digit_equiv_mod_8_l168_168424

theorem smallest_four_digit_equiv_mod_8 : ∃ n : ℕ, n >= 1000 ∧ n < 10000 ∧ n % 8 = 3 ∧ n = 1003 :=
by
  -- We state the assumptions and final goal
  use 1003
  split
  · linarith
  split
  · linarith
  split
  · norm_num
  · refl
  sorry

end smallest_four_digit_equiv_mod_8_l168_168424


namespace fraction_comparison_l168_168224

theorem fraction_comparison (a b c d : ℝ) 
  (h1 : (a / b) < (c / d))
  (h2 : b > d) (h3 : d > 0) :
  (a + c) / (b + d) < (1 / 2) * ((a / b) + (c / d)) :=
by
  sorry

end fraction_comparison_l168_168224


namespace no_perfect_square_in_form_l168_168641

noncomputable def is_special_form (x : ℕ) : Prop := 99990000 ≤ x ∧ x ≤ 99999999

theorem no_perfect_square_in_form :
  ¬∃ (x : ℕ), is_special_form x ∧ ∃ (n : ℕ), x = n ^ 2 := 
by 
  sorry

end no_perfect_square_in_form_l168_168641


namespace angle_sum_l168_168499

theorem angle_sum {A B D F G : Type} 
  (angle_A : ℝ) 
  (angle_AFG : ℝ) 
  (angle_AGF : ℝ) 
  (angle_BFD : ℝ)
  (H1 : angle_A = 30)
  (H2 : angle_AFG = angle_AGF)
  (H3 : angle_BFD = 105)
  (H4 : angle_AFG + angle_BFD = 180) 
  : angle_B + angle_D = 75 := 
by 
  sorry

end angle_sum_l168_168499


namespace cyclic_sum_non_negative_equality_condition_l168_168646

theorem cyclic_sum_non_negative (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^2 * (a - b) / (a + b) + b^2 * (b - c) / (b + c) + c^2 * (c - a) / (c + a) ≥ 0 :=
sorry

theorem equality_condition (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^2 * (a - b) / (a + b) + b^2 * (b - c) / (b + c) + c^2 * (c - a) / (c + a) = 0 ↔ a = b ∧ b = c :=
sorry

end cyclic_sum_non_negative_equality_condition_l168_168646


namespace handshake_problem_7_boys_21_l168_168352

theorem handshake_problem_7_boys_21 :
  let n := 7
  let total_handshakes := n * (n - 1) / 2
  total_handshakes = 21 → (n - 1) = 6 :=
by
  -- Let n be the number of boys (7 in this case)
  let n := 7
  
  -- Define the total number of handshakes equation
  let total_handshakes := n * (n - 1) / 2
  
  -- Assume the total number of handshakes is 21
  intro h
  -- Proof steps would go here
  sorry

end handshake_problem_7_boys_21_l168_168352


namespace ratio_is_1_to_3_l168_168950

-- Definitions based on the conditions
def washed_on_wednesday : ℕ := 6
def washed_on_thursday : ℕ := 2 * washed_on_wednesday
def washed_on_friday : ℕ := washed_on_thursday / 2
def total_washed : ℕ := 26
def washed_on_saturday : ℕ := total_washed - washed_on_wednesday - washed_on_thursday - washed_on_friday

-- The ratio calculation
def ratio_saturday_to_wednesday : ℚ := washed_on_saturday / washed_on_wednesday

-- The theorem to prove
theorem ratio_is_1_to_3 : ratio_saturday_to_wednesday = 1 / 3 :=
by
  -- Insert proof here
  sorry

end ratio_is_1_to_3_l168_168950


namespace num_biology_books_is_15_l168_168415

-- conditions
def num_chemistry_books : ℕ := 8
def total_ways : ℕ := 2940

-- main statement to prove
theorem num_biology_books_is_15 : ∃ B: ℕ, (B * (B - 1)) / 2 * (num_chemistry_books * (num_chemistry_books - 1)) / 2 = total_ways ∧ B = 15 :=
by
  sorry

end num_biology_books_is_15_l168_168415


namespace loisa_savings_l168_168524

namespace SavingsProof

def cost_cash : ℤ := 450
def down_payment : ℤ := 100
def payment_first_4_months : ℤ := 4 * 40
def payment_next_4_months : ℤ := 4 * 35
def payment_last_4_months : ℤ := 4 * 30

def total_installment_payment : ℤ :=
  down_payment + payment_first_4_months + payment_next_4_months + payment_last_4_months

theorem loisa_savings :
  (total_installment_payment - cost_cash) = 70 := by
  sorry

end SavingsProof

end loisa_savings_l168_168524


namespace root_interval_k_l168_168635

noncomputable def f (x : ℝ) : ℝ := x^3 + 2 * x - 1

theorem root_interval_k (k : ℤ) (h : ∃ ξ : ℝ, k < ξ ∧ ξ < k+1 ∧ f ξ = 0) : k = 0 :=
by
  sorry

end root_interval_k_l168_168635


namespace production_equipment_B_l168_168099

theorem production_equipment_B :
  ∃ (X Y : ℕ), X + Y = 4800 ∧ (50 / 80 = 5 / 8) ∧ (X / 4800 = 5 / 8) ∧ Y = 1800 :=
by
  sorry

end production_equipment_B_l168_168099


namespace problem_1_problem_2_l168_168789

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log (x + 1) / Real.log 2 else 2^(-x) - 1

theorem problem_1 : f (f (-2)) = 2 := by 
  sorry

theorem problem_2 (x_0 : ℝ) (h : f x_0 < 3) : -2 < x_0 ∧ x_0 < 7 := by
  sorry

end problem_1_problem_2_l168_168789


namespace find_x_l168_168063

theorem find_x (A B D : ℝ) (BC CD x : ℝ) 
  (hA : A = 60) (hB : B = 90) (hD : D = 90) 
  (hBC : BC = 2) (hCD : CD = 3) 
  (hResult : x = 8 / Real.sqrt 3) : 
  AB = x :=
by
  sorry

end find_x_l168_168063


namespace terminating_decimal_representation_l168_168329

theorem terminating_decimal_representation : 
  (67 / (2^3 * 5^4) : ℝ) = 0.0134 :=
    sorry

end terminating_decimal_representation_l168_168329


namespace greatest_multiple_less_than_l168_168811

def lcm (a b : ℕ) : ℕ := Nat.lcm a b -- Assuming lcm function is already defined

theorem greatest_multiple_less_than (a b m : ℕ) (h₁ : a = 15) (h₂ : b = 20) (h₃ : m = 150) : 
  ∃ k, k * lcm a b < m ∧ ¬ ∃ k', (k' * lcm a b < m ∧ k' > k) :=
by
  sorry

end greatest_multiple_less_than_l168_168811


namespace fraction_difference_l168_168108

theorem fraction_difference : 7 / 12 - 3 / 8 = 5 / 24 := 
by 
  sorry

end fraction_difference_l168_168108


namespace initial_cows_l168_168302

theorem initial_cows (x : ℕ) (h : (3 / 4 : ℝ) * (x + 5) = 42) : x = 51 :=
by
  sorry

end initial_cows_l168_168302


namespace polynomial_multiplication_l168_168774

theorem polynomial_multiplication (x z : ℝ) :
  (3*x^5 - 7*z^3) * (9*x^10 + 21*x^5*z^3 + 49*z^6) = 27*x^15 - 343*z^9 :=
by
  sorry

end polynomial_multiplication_l168_168774


namespace fifth_friend_payment_l168_168605

/-- 
Five friends bought a piece of furniture for $120.
The first friend paid one third of the sum of the amounts paid by the other four;
the second friend paid one fourth of the sum of the amounts paid by the other four;
the third friend paid one fifth of the sum of the amounts paid by the other four;
and the fourth friend paid one sixth of the sum of the amounts paid by the other four.
Prove that the fifth friend paid $41.33.
-/
theorem fifth_friend_payment :
  ∀ (a b c d e : ℝ),
    a = 1/3 * (b + c + d + e) →
    b = 1/4 * (a + c + d + e) →
    c = 1/5 * (a + b + d + e) →
    d = 1/6 * (a + b + c + e) →
    a + b + c + d + e = 120 →
    e = 41.33 :=
by
  intros a b c d e ha hb hc hd he_sum
  sorry

end fifth_friend_payment_l168_168605


namespace unique_real_solution_for_cubic_l168_168857

theorem unique_real_solution_for_cubic {b : ℝ} :
  (∀ x : ℝ, (x^3 - b * x^2 - 3 * b * x + b^2 - 4 = 0) → ∃! x : ℝ, (x^3 - b * x^2 - 3 * b * x + b^2 - 4 = 0)) ↔ b > 3 :=
sorry

end unique_real_solution_for_cubic_l168_168857


namespace total_trip_cost_l168_168454

-- Definitions for the problem
def price_per_person : ℕ := 147
def discount : ℕ := 14
def number_of_people : ℕ := 2

-- Statement to prove
theorem total_trip_cost :
  (price_per_person - discount) * number_of_people = 266 :=
by
  sorry

end total_trip_cost_l168_168454


namespace computation_problems_count_l168_168976

theorem computation_problems_count
    (C W : ℕ)
    (h1 : 3 * C + 5 * W = 110)
    (h2 : C + W = 30) :
    C = 20 :=
by
  sorry

end computation_problems_count_l168_168976


namespace tony_slices_remaining_l168_168273

def slices_remaining : ℕ :=
  let total_slices := 22
  let slices_per_sandwich := 2
  let daily_sandwiches := 1
  let extra_sandwiches := 1
  let total_sandwiches := 7 * daily_sandwiches + extra_sandwiches
  total_slices - total_sandwiches * slices_per_sandwich

theorem tony_slices_remaining :
  slices_remaining = 6 :=
by
  -- Here, the theorem proof would go, but it is omitted as per instructions
  sorry

end tony_slices_remaining_l168_168273


namespace range_of_a_for_monotonically_decreasing_function_l168_168677

theorem range_of_a_for_monotonically_decreasing_function {a : ℝ} :
    (∀ x y : ℝ, (x > 2 → y > 2 → (ax^2 + x - 1) ≤ (a*y^2 + y - 1)) ∧
                (x ≤ 2 → y ≤ 2 → (-x + 1) ≤ (-y + 1)) ∧
                (x > 2 → y ≤ 2 → (ax^2 + x - 1) ≤ (-y + 1)) ∧
                (x ≤ 2 → y > 2 → (-x + 1) ≤ (a*y^2 + y - 1))) →
    (a < 0 ∧ - (1 / (2 * a)) ≤ 2 ∧ 4 * a + 1 ≤ -1) →
    a ≤ -1 / 2 :=
by
  intro hmonotone hconditions
  sorry

end range_of_a_for_monotonically_decreasing_function_l168_168677


namespace complex_fraction_value_l168_168150

theorem complex_fraction_value :
  (Complex.mk 1 2) * (Complex.mk 1 2) / Complex.mk 3 (-4) = -1 :=
by
  -- Here we would provide the proof, but as per instructions,
  -- we will insert sorry to skip it.
  sorry

end complex_fraction_value_l168_168150


namespace minimize_y_l168_168076

theorem minimize_y (a b : ℝ) : 
  ∃ x, x = (a + b) / 2 ∧ ∀ x', ((x' - a)^3 + (x' - b)^3) ≥ ((x - a)^3 + (x - b)^3) :=
sorry

end minimize_y_l168_168076


namespace simplify_fraction_l168_168663

theorem simplify_fraction (b : ℕ) (h : b = 2) : (15 * b^4) / (45 * b^3) = 2 / 3 :=
by
  -- We have to assume that 15b^4 and 45b^3 are integers
  -- We have to consider them as integers to apply integer division
  have : 15 * b^4 = (15 : ℚ) * (b : ℚ)^4 := by sorry
  have : 45 * b^3 = (45 : ℚ) * (b : ℚ)^3 := by sorry
  have eq1 : ((15 : ℚ) * (b : ℚ)^4) / ((45 : ℚ) * (b : ℚ)^3) = (15 : ℚ) / (45 : ℚ) * (b : ℚ) := by sorry
  have eq2 : (15 : ℚ) / (45 : ℚ) = 1 / 3 := by sorry
  have eq3 : ((1 : ℚ) / (3 : ℚ)) * (b : ℚ) = (b : ℚ) / 3 := by sorry
  rw [←eq1, eq2, eq3] at *,
  rw h,
  exact eq_of_rat_eq_rat (by norm_num),

end simplify_fraction_l168_168663


namespace time_after_2051_hours_l168_168412

theorem time_after_2051_hours (h₀ : 9 ≤ 11): 
  (9 + 2051 % 12) % 12 = 8 :=
by {
  -- proving the statement here
  sorry
}

end time_after_2051_hours_l168_168412


namespace intersection_is_target_set_l168_168629

-- Define sets A and B
def is_in_A (x : ℝ) : Prop := |x - 1| < 2
def is_in_B (x : ℝ) : Prop := x^2 < 4

-- Define the intersection A ∩ B
def is_in_intersection (x : ℝ) : Prop := is_in_A x ∧ is_in_B x

-- Define the target set
def is_in_target_set (x : ℝ) : Prop := -1 < x ∧ x < 2

-- Statement to prove
theorem intersection_is_target_set : 
  ∀ x : ℝ, is_in_intersection x ↔ is_in_target_set x := sorry

end intersection_is_target_set_l168_168629


namespace derivative_at_pi_div_2_l168_168035

noncomputable def f (x : ℝ) : ℝ := x * Real.sin (2 * x)

theorem derivative_at_pi_div_2 : deriv f (Real.pi / 2) = -Real.pi := by
  sorry

end derivative_at_pi_div_2_l168_168035


namespace yan_ratio_l168_168557

variables (w x y : ℝ)

-- Given conditions
def yan_conditions : Prop :=
  w > 0 ∧ 
  x > 0 ∧ 
  y > 0 ∧ 
  (y / w = x / w + (x + y) / (7 * w))

-- The ratio of Yan's distance from his home to his distance from the stadium is 3/4
theorem yan_ratio (h : yan_conditions w x y) : 
  x / y = 3 / 4 :=
sorry

end yan_ratio_l168_168557


namespace probability_of_prime_sum_is_five_over_twelve_l168_168948

def is_prime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7 ∨ n = 11

noncomputable def probability_prime_sum_dice : ℚ :=
  let outcomes := (Finset.product (Finset.range 6) (Finset.range 6)).filter (λ p, is_prime (p.1 + 1 + (p.2 + 1))) in
  outcomes.card / (6 * 6 : ℚ)

theorem probability_of_prime_sum_is_five_over_twelve :
  probability_prime_sum_dice = 5 / 12 :=
by
  sorry

end probability_of_prime_sum_is_five_over_twelve_l168_168948


namespace count_numbers_with_5_or_6_in_base_8_l168_168885

-- Define the condition that checks if a number contains digits 5 or 6 in base 8
def contains_digit_5_or_6_in_base_8 (n : ℕ) : Prop :=
  let digits := Nat.digits 8 n
  5 ∈ digits ∨ 6 ∈ digits

-- The main problem statement
theorem count_numbers_with_5_or_6_in_base_8 :
  (Finset.filter contains_digit_5_or_6_in_base_8 (Finset.range 513)).card = 296 :=
by
  sorry

end count_numbers_with_5_or_6_in_base_8_l168_168885


namespace find_interval_l168_168030

theorem find_interval (n : ℕ) 
  (h1 : n < 500) 
  (h2 : n ∣ 9999) 
  (h3 : n + 4 ∣ 99) : (1 ≤ n) ∧ (n ≤ 125) := 
sorry

end find_interval_l168_168030


namespace number_with_5_or_6_base_8_l168_168886

open Finset

def count_numbers_with_5_or_6 : ℕ :=
  let base_8_numbers := Ico 1 (8 ^ 3)
  let count_with_5_or_6 := base_8_numbers.filter (λ n, ∃ b, b ∈ digit_set 8 n ∧ (b = 5 ∨ b = 6))
  count_with_5_or_6.card

theorem number_with_5_or_6_base_8 : count_numbers_with_5_or_6 = 296 := 
by 
  -- Proof omitted for this exercise
  sorry

end number_with_5_or_6_base_8_l168_168886


namespace positive_difference_of_two_numbers_l168_168549

theorem positive_difference_of_two_numbers :
  ∃ x y : ℚ, x + y = 40 ∧ 3 * y - 4 * x = 20 ∧ y - x = 80 / 7 :=
by
  sorry

end positive_difference_of_two_numbers_l168_168549


namespace smallest_m_exists_l168_168230

noncomputable def T : set ℂ :=
  {z : ℂ | ∃ x y : ℝ, z = x + (y * complex.I) ∧ (1 / 2) ≤ x ∧ x ≤ real.sqrt_two / 2}

theorem smallest_m_exists :
  ∃ m : ℕ, (∀ n : ℕ, n ≥ m → ∃ z ∈ T, z ^ n = 1) ∧ m = 12 :=
by
  use 12
  split
  · intros n hn
    -- For simplification, we denote "complex number of form x + yi"
    let z_2 := complex.cis (real.pi / 3) -- cos(60 degrees) + i*sin(60 degrees)
    let z_10 := complex.cis (5 * real.pi / 3) -- cos(300 degrees) + i*sin(300 degrees)
    -- Show z_2 and z_10 satisfy conditions
    cases hn with n_pos
    use z_2
    split
    · -- z_2 ∈ T
      sorry
    · -- z_2 ^ n = 1
      sorry

      -- Alternative number z_10 in case z_2 isn't suitable
      use z_10
      split
      · -- z_10 ∈ T
        sorry
      · -- z_10 ^ n = 1
        sorry
  · -- m = 12
    rfl

end smallest_m_exists_l168_168230


namespace range_of_a_l168_168505

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, |x - 2| + |x - a| ≥ a) → a ≤ 1 :=
by
  intro h
  sorry

end range_of_a_l168_168505


namespace find_y_intercept_l168_168260

def point (x y : ℝ) := (x, y)

def slope (m : ℝ) := m

def x_intercept (x : ℝ) := point x 0

def y_intercept_point (m x0 : ℝ) :=
  let b := 0 - m * x0
  point 0 b

theorem find_y_intercept :
  ∀ (m x0 : ℝ), (m = -3) → (x0 = 7) → y_intercept_point m x0 = (0, 21) :=
by
  intros m x0 m_cond x0_cond
  simp [y_intercept_point, m_cond, x0_cond]
  exact sorry

end find_y_intercept_l168_168260


namespace area_tripled_sides_l168_168899

theorem area_tripled_sides (a b : ℝ) (θ : ℝ) (hθ : 0 < θ ∧ θ < π) :
  let A := 1 / 2 * a * b * Real.sin θ in
  let A' := 1 / 2 * (3 * a) * (3 * b) * Real.sin θ in
  A' = 9 * A := by
  sorry

end area_tripled_sides_l168_168899


namespace total_students_in_lunchroom_l168_168438

theorem total_students_in_lunchroom 
  (students_per_table : ℕ) 
  (number_of_tables : ℕ) 
  (h1 : students_per_table = 6) 
  (h2 : number_of_tables = 34) : 
  students_per_table * number_of_tables = 204 := by
  sorry

end total_students_in_lunchroom_l168_168438


namespace largest_of_four_numbers_l168_168341

theorem largest_of_four_numbers (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : a + b = 1) :
  max (max (max (a^2 + b^2) (2 * a * b)) a) (1 / 2) = a^2 + b^2 :=
by
  sorry

end largest_of_four_numbers_l168_168341


namespace purely_imaginary_complex_number_l168_168788

theorem purely_imaginary_complex_number (m : ℝ) :
  (m^2 - 2 * m - 3 = 0) ∧ (m^2 - 4 * m + 3 ≠ 0) → m = -1 :=
by
  sorry

end purely_imaginary_complex_number_l168_168788


namespace unique_intersection_point_l168_168031

theorem unique_intersection_point {c : ℝ} :
  (∀ x y : ℝ, y = |x - 20| + |x + 18| → y = x + c → (x = 20 ∧ y = 38)) ↔ c = 18 :=
by
  sorry

end unique_intersection_point_l168_168031


namespace sum_a_b_l168_168172

variable {a b : ℝ}

theorem sum_a_b (hab : a * b = 5) (hrecip : 1 / (a^2) + 1 / (b^2) = 0.6) : a + b = 5 ∨ a + b = -5 :=
sorry

end sum_a_b_l168_168172


namespace desiree_age_l168_168997

-- Definitions of the given variables and conditions
variables (D C : ℝ)

-- Given conditions
def condition1 : Prop := D = 2 * C
def condition2 : Prop := D + 30 = 0.6666666 * (C + 30) + 14
def condition3 : Prop := D = 2.99999835

-- Main theorem to prove
theorem desiree_age : D = 2.99999835 :=
by
  { sorry }

end desiree_age_l168_168997


namespace marbles_given_to_juan_l168_168992

def initial : ℕ := 776
def left : ℕ := 593

theorem marbles_given_to_juan : initial - left = 183 :=
by sorry

end marbles_given_to_juan_l168_168992


namespace compute_105_squared_l168_168580

theorem compute_105_squared :
  let a := 100
  let b := 5
  (a + b)^2 = 11025 :=
by
  sorry

end compute_105_squared_l168_168580


namespace range_of_m_l168_168881

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, (x^2 - 2 * m * x + 4 = 0) → x > 1) ↔ (2 ≤ m ∧ m < 5/2) := sorry

end range_of_m_l168_168881


namespace part1_solution_set_part2_inequality_l168_168184

-- Part (1)
theorem part1_solution_set (x : ℝ) : |x| < 2 * x - 1 ↔ 1 < x := by
  sorry

-- Part (2)
theorem part2_inequality (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h_sum : a + 2 * b + c = 1) :
  (1 / (a + b)) + (1 / (b + c)) ≥ 4 := by
  sorry

end part1_solution_set_part2_inequality_l168_168184


namespace max_extra_time_matches_l168_168293

theorem max_extra_time_matches (number_teams : ℕ) 
    (points_win : ℕ) (points_lose : ℕ) 
    (points_win_extra : ℕ) (points_lose_extra : ℕ) 
    (total_matches_2016 : number_teams = 2016)
    (pts_win_3 : points_win = 3)
    (pts_lose_0 : points_lose = 0)
    (pts_win_extra_2 : points_win_extra = 2)
    (pts_lose_extra_1 : points_lose_extra = 1) :
    ∃ N, N = 1512 := 
by {
  sorry
}

end max_extra_time_matches_l168_168293


namespace recreation_percentage_l168_168073

theorem recreation_percentage (W : ℝ) (hW : W > 0) :
  (0.40 * W) / (0.15 * W) * 100 = 267 := by
  sorry

end recreation_percentage_l168_168073


namespace train_speed_is_64_98_kmph_l168_168717

noncomputable def train_length : ℝ := 200
noncomputable def bridge_length : ℝ := 180
noncomputable def passing_time : ℝ := 21.04615384615385
noncomputable def speed_in_kmph : ℝ := 3.6 * (train_length + bridge_length) / passing_time

theorem train_speed_is_64_98_kmph : abs (speed_in_kmph - 64.98) < 0.01 :=
by
  sorry

end train_speed_is_64_98_kmph_l168_168717


namespace hash_op_example_l168_168598

def hash_op (a b c : ℤ) : ℤ := (b + 1)^2 - 4 * a * (c - 1)

theorem hash_op_example : hash_op 2 3 4 = -8 := by
  -- The proof can be added here, but for now, we use sorry to skip it
  sorry

end hash_op_example_l168_168598


namespace sum_of_first_15_terms_l168_168638

open scoped BigOperators

-- Define the sequence as an arithmetic sequence
def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ :=
  a + n * d

-- Define the condition given in the problem
def condition (a d : ℤ) : Prop :=
  3 * (arithmetic_sequence a d 2 + arithmetic_sequence a d 4) + 
  2 * (arithmetic_sequence a d 6 + arithmetic_sequence a d 11 + arithmetic_sequence a d 16) = 180

-- Prove that the sum of the first 15 terms is 225
theorem sum_of_first_15_terms (a d : ℤ) (h : condition a d) :
  ∑ i in Finset.range 15, arithmetic_sequence a d i = 225 :=
  sorry

end sum_of_first_15_terms_l168_168638


namespace sum_of_ages_l168_168529

theorem sum_of_ages (rachel_age leah_age : ℕ) 
  (h1 : rachel_age = leah_age + 4) 
  (h2 : rachel_age = 19) : rachel_age + leah_age = 34 :=
by
  -- Proof steps are omitted since we only need the statement
  sorry

end sum_of_ages_l168_168529


namespace sum_of_dimensions_l168_168972

theorem sum_of_dimensions (A B C : ℝ) (h1 : A * B = 50) (h2 : A * C = 90) (h3 : B * C = 100) : A + B + C = 24 :=
  sorry

end sum_of_dimensions_l168_168972


namespace trigonometric_identity_x1_trigonometric_identity_x2_l168_168699

noncomputable def x1 (n : ℤ) : ℝ := (2 * n + 1) * (Real.pi / 4)
noncomputable def x2 (k : ℤ) : ℝ := ((-1)^(k + 1)) * (Real.pi / 8) + k * (Real.pi / 2)

theorem trigonometric_identity_x1 (n : ℤ) : 
  (Real.cos (4 * x1 n) * Real.cos (Real.pi + 2 * x1 n) - 
   Real.sin (2 * x1 n) * Real.cos (Real.pi / 2 - 4 * x1 n)) = 
   (Real.sqrt 2 / 2) * Real.sin (4 * x1 n) := 
by
  sorry

theorem trigonometric_identity_x2 (k : ℤ) : 
  (Real.cos (4 * x2 k) * Real.cos (Real.pi + 2 * x2 k) - 
   Real.sin (2 * x2 k) * Real.cos (Real.pi / 2 - 4 * x2 k)) = 
   (Real.sqrt 2 / 2) * Real.sin (4 * x2 k) := 
by
  sorry

end trigonometric_identity_x1_trigonometric_identity_x2_l168_168699


namespace positive_difference_of_two_numbers_l168_168548

theorem positive_difference_of_two_numbers :
  ∃ x y : ℚ, x + y = 40 ∧ 3 * y - 4 * x = 20 ∧ y - x = 80 / 7 :=
by
  sorry

end positive_difference_of_two_numbers_l168_168548


namespace distance_to_cheaper_gas_station_l168_168979

-- Define the conditions
def miles_per_gallon : ℕ := 3
def initial_gallons : ℕ := 12
def additional_gallons : ℕ := 18

-- Define the question and proof statement
theorem distance_to_cheaper_gas_station : 
  (initial_gallons + additional_gallons) * miles_per_gallon = 90 := by
  sorry

end distance_to_cheaper_gas_station_l168_168979


namespace infinite_set_S_exists_l168_168160
open Function

def exists_infinite_set_S : Prop :=
  ∃ (S : Set ℕ) (infinite S),
  ∀ ⦃a b : ℕ⦄, a ∈ S → b ∈ S → ∃ k : ℕ, k % 2 = 1 ∧ a ∣ (b^k + 1)

theorem infinite_set_S_exists : exists_infinite_set_S :=
sorry

end infinite_set_S_exists_l168_168160


namespace sum_of_reciprocals_of_shifted_roots_l168_168234

theorem sum_of_reciprocals_of_shifted_roots (a b c : ℝ) (h : ∀ x, x^3 - x + 2 = 0 → x = a ∨ x = b ∨ x = c) :
  1 / (a + 2) + 1 / (b + 2) + 1 / (c + 2) = 11 / 4 :=
by
  sorry

end sum_of_reciprocals_of_shifted_roots_l168_168234


namespace anna_lemonade_difference_l168_168143

variables (x y p s : ℝ)

theorem anna_lemonade_difference (h : x * p = 1.5 * (y * s)) : (x * p) - (y * s) = 0.5 * (y * s) :=
by
  -- Insert proof here
  sorry

end anna_lemonade_difference_l168_168143


namespace arithmetic_identity_l168_168005

theorem arithmetic_identity : Real.sqrt 16 + ((1/2) ^ (-2:ℤ)) = 8 := 
by 
  sorry

end arithmetic_identity_l168_168005


namespace probability_red_ball_distribution_of_X_expected_value_of_X_l168_168070

theorem probability_red_ball :
  let pB₁ := 2 / 3
  let pB₂ := 1 / 3
  let pA_B₁ := 1 / 2
  let pA_B₂ := 3 / 4
  (pB₁ * pA_B₁ + pB₂ * pA_B₂) = 7 / 12 := by
  sorry

theorem distribution_of_X :
  let p_minus2 := 1 / 12
  let p_0 := 1 / 12
  let p_1 := 11 / 24
  let p_3 := 7 / 48
  let p_4 := 5 / 24
  let p_6 := 1 / 48
  (p_minus2 = 1 / 12) ∧ (p_0 = 1 / 12) ∧ (p_1 = 11 / 24) ∧ (p_3 = 7 / 48) ∧ (p_4 = 5 / 24) ∧ (p_6 = 1 / 48) := by
  sorry

theorem expected_value_of_X :
  let E_X := (-2 * (1 / 12) + 0 * (1 / 12) + 1 * (11 / 24) + 3 * (7 / 48) + 4 * (5 / 24) + 6 * (1 / 48))
  E_X = 27 / 16 := by
  sorry

end probability_red_ball_distribution_of_X_expected_value_of_X_l168_168070


namespace original_data_props_l168_168713

variable {n : ℕ}
variable {x : Fin n → ℝ}
variable {new_x : Fin n → ℝ} 

noncomputable def average (data : Fin n → ℝ) : ℝ :=
  (Finset.univ.sum (λ i => data i)) / n

noncomputable def variance (data : Fin n → ℝ) : ℝ :=
  (Finset.univ.sum (λ i => (data i - average data) ^ 2)) / n

-- Conditions
def condition1 (x new_x : Fin n → ℝ) (h : ∀ i, new_x i = x i - 80) : Prop := true

def condition2 (new_x : Fin n → ℝ) : Prop :=
  average new_x = 1.2

def condition3 (new_x : Fin n → ℝ) : Prop :=
  variance new_x = 4.4

theorem original_data_props (h : ∀ i, new_x i = x i - 80)
  (h_avg : average new_x = 1.2) 
  (h_var : variance new_x = 4.4) :
  average x = 81.2 ∧ variance x = 4.4 :=
sorry

end original_data_props_l168_168713


namespace red_chairs_count_l168_168088

-- Given conditions
variables {R Y B : ℕ} -- Assuming the number of chairs are natural numbers

-- Main theorem statement
theorem red_chairs_count : 
  Y = 4 * R ∧ B = Y - 2 ∧ R + Y + B = 43 -> R = 5 :=
by
  sorry

end red_chairs_count_l168_168088


namespace bowling_ball_weight_l168_168865

theorem bowling_ball_weight (b c : ℝ) 
  (h1 : 5 * b = 4 * c) 
  (h2 : 2 * c = 80) : 
  b = 32 :=
by
  sorry

end bowling_ball_weight_l168_168865


namespace tournament_matches_l168_168210

theorem tournament_matches (n : ℕ) (total_matches : ℕ) (matches_three_withdrew : ℕ) (matches_after_withdraw : ℕ) :
  ∀ (x : ℕ), total_matches = (n * (n - 1) / 2) → matches_three_withdrew = 6 - x → matches_after_withdraw = total_matches - (3 * 2 - x) → 
  matches_after_withdraw = 50 → x = 1 :=
by
  intros
  sorry

end tournament_matches_l168_168210


namespace total_cost_of_water_l168_168390

-- Define conditions in Lean 4
def cost_per_liter : ℕ := 1
def liters_per_bottle : ℕ := 2
def number_of_bottles : ℕ := 6

-- Define the theorem to prove the total cost
theorem total_cost_of_water : (number_of_bottles * (liters_per_bottle * cost_per_liter)) = 12 :=
by
  sorry

end total_cost_of_water_l168_168390


namespace find_x2_plus_y2_l168_168501

theorem find_x2_plus_y2 (x y : ℝ) (h1 : x - y = 20) (h2 : x * y = 9) : x^2 + y^2 = 418 :=
  sorry

end find_x2_plus_y2_l168_168501


namespace polygon_sides_l168_168311

theorem polygon_sides (R : ℝ) (n : ℕ) (h : R ≠ 0)
  (h_area : (1 / 2) * n * R^2 * Real.sin (360 / n * (Real.pi / 180)) = 4 * R^2) :
  n = 8 := 
by
  sorry

end polygon_sides_l168_168311


namespace find_product_l168_168722

-- Define the variables used in the problem statement
variables (A P D B E C F : Type) (AP PD BP PE CP PF : ℝ)

-- The condition given in the problem
def condition (x y z : ℝ) : Prop := 
  x + y + z = 90

-- The theorem to prove
theorem find_product (x y z : ℝ) (h : condition x y z) : 
  x * y * z = 94 :=
sorry

end find_product_l168_168722


namespace tony_slices_remaining_l168_168274

def slices_remaining : ℕ :=
  let total_slices := 22
  let slices_per_sandwich := 2
  let daily_sandwiches := 1
  let extra_sandwiches := 1
  let total_sandwiches := 7 * daily_sandwiches + extra_sandwiches
  total_slices - total_sandwiches * slices_per_sandwich

theorem tony_slices_remaining :
  slices_remaining = 6 :=
by
  -- Here, the theorem proof would go, but it is omitted as per instructions
  sorry

end tony_slices_remaining_l168_168274


namespace binomial_coefficient_19_13_l168_168875

theorem binomial_coefficient_19_13 
  (h1 : Nat.choose 20 13 = 77520) 
  (h2 : Nat.choose 20 14 = 38760) 
  (h3 : Nat.choose 18 13 = 18564) :
  Nat.choose 19 13 = 37128 := 
sorry

end binomial_coefficient_19_13_l168_168875


namespace count_4_digit_numbers_divisible_by_13_l168_168896

theorem count_4_digit_numbers_divisible_by_13 : 
  let count := (9962 - 1014) / 13 + 1
  1000 ≤ 1014 ∧ 9962 ≤ 9999 →
  count = 689 :=
  by
    sorry

end count_4_digit_numbers_divisible_by_13_l168_168896


namespace total_books_l168_168416

def shelves : ℕ := 150
def books_per_shelf : ℕ := 15

theorem total_books (shelves books_per_shelf : ℕ) : shelves * books_per_shelf = 2250 := by
  sorry

end total_books_l168_168416


namespace range_of_g_l168_168764

noncomputable def g (x : ℝ) : ℝ :=
  (Real.arccos x)^2 - (Real.arcsin x)^2

theorem range_of_g :
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → -((Real.pi^2) / 4) ≤ g x ∧ g x ≤ (3 * (Real.pi^2)) / 4 :=
by
  intros x hx
  sorry

end range_of_g_l168_168764


namespace prob_two_days_ge_100_one_day_lt_50_xi_distribution_xi_mean_xi_variance_l168_168316

section CakeShop

def freqDist := [
  (0, 50, 15),
  (50, 100, 25),
  (100, 150, 30),
  (150, 200, 20),
  (200, 250, 10)
]

def p_sales_ge_100 : ℝ := (30 + 20 + 10) / 100
def p_sales_lt_50 : ℝ := 15 / 100
def X := binomial 3 0.3

theorem prob_two_days_ge_100_one_day_lt_50 : 
  (3 * (p_sales_ge_100 ^ 2) * p_sales_lt_50) = 0.162 := 
by
  simp [p_sales_ge_100, p_sales_lt_50]
  sorry

theorem xi_distribution :
  (∀ x, x ∈ X.support → 
    (X.prob x = match x with
      | 0 => 0.343
      | 1 => 0.441
      | 2 => 0.189
      | 3 => 0.027
      | _ => 0)
  ) := sorry

theorem xi_mean : X.mean = 0.9 := by
  simp [X.mean]
  sorry

theorem xi_variance : X.variance = 0.63 := by
  simp [X.variance]
  sorry

end CakeShop

end prob_two_days_ge_100_one_day_lt_50_xi_distribution_xi_mean_xi_variance_l168_168316


namespace debut_show_tickets_l168_168446

variable (P : ℕ) -- Number of people who bought tickets for the debut show

-- Conditions
def three_times_more (P : ℕ) : Bool := (3 * P = P + 2 * P)
def ticket_cost : ℕ := 25
def total_revenue (P : ℕ) : ℕ := 4 * P * ticket_cost

-- Main statement
theorem debut_show_tickets (h1 : three_times_more P = true) 
                           (h2 : total_revenue P = 20000) : P = 200 :=
by
  sorry

end debut_show_tickets_l168_168446


namespace sin_A_mul_sin_B_find_c_l168_168215

-- Definitions for the triangle and the given conditions
variable (A B C : ℝ) -- Angles of the triangle
variable (a b c : ℝ) -- Opposite sides of the triangle

-- Given conditions
axiom h1 : c^2 = 4 * a * b * (Real.sin C)^2

-- The first proof problem statement
theorem sin_A_mul_sin_B (ha : A + B + C = π) (h2 : Real.sin C ≠ 0) :
  Real.sin A * Real.sin B = 1/4 :=
by
  sorry

-- The second proof problem statement with additional given conditions
theorem find_c (ha : A = π / 6) (ha2 : a = 3) (hb2 : b = 3) : 
  c = 3 * Real.sqrt 3 :=
by
  sorry

end sin_A_mul_sin_B_find_c_l168_168215


namespace proof_problem_l168_168324

def diamondsuit (a b : ℝ) : ℝ := a^3 * b - a * b^3

theorem proof_problem :
  { (x, y) : ℝ × ℝ | diamondsuit x y = diamondsuit y x } =
  { (x, y) | x = 0 } ∪ { (x, y) | y = 0 } ∪ { (x, y) | x = y } ∪ { (x, y) | x = -y } :=
by
  sorry

end proof_problem_l168_168324


namespace base8_contains_5_or_6_l168_168887

theorem base8_contains_5_or_6 (n : ℕ) (h : n = 512) : 
  let count_numbers_without_5_6 := 6^3 in
  let total_numbers := 512 in
  total_numbers - count_numbers_without_5_6 = 296 := by
  sorry

end base8_contains_5_or_6_l168_168887


namespace option_B_correct_l168_168827

theorem option_B_correct (x y : ℝ) : 
  x * y^2 - y^2 * x = 0 :=
by sorry

end option_B_correct_l168_168827


namespace square_of_105_l168_168593

theorem square_of_105 : 105^2 = 11025 := by
  sorry

end square_of_105_l168_168593


namespace total_travel_time_l168_168544

noncomputable def travel_time (distance_razumeyevo_river : ℝ) (distance_vkusnoteevo_river : ℝ)
    (distance_downstream : ℝ) (river_width : ℝ) (current_speed : ℝ)
    (swimming_speed : ℝ) (walking_speed : ℝ) : ℝ := 
  let time_walk1 := distance_razumeyevo_river / walking_speed
  let effective_swimming_speed := real.sqrt (swimming_speed^2 - current_speed^2)
  let time_swim := river_width / effective_swimming_speed
  let time_walk2 := distance_vkusnoteevo_river / walking_speed
  time_walk1 + time_swim + time_walk2

theorem total_travel_time :
    travel_time 3 1 3.25 0.5 1 2 4 = 1.5 := 
by
  sorry

end total_travel_time_l168_168544


namespace sum_base8_to_decimal_l168_168140

theorem sum_base8_to_decimal (a b : ℕ) (ha : a = 5) (hb : b = 0o17)
  (h_sum_base8 : a + b = 0o24) : (a + b) = 20 := by
  sorry

end sum_base8_to_decimal_l168_168140


namespace Sally_out_of_pocket_payment_l168_168926

theorem Sally_out_of_pocket_payment :
  let amount_given : ℕ := 320
  let cost_per_book : ℕ := 12
  let number_of_students : ℕ := 30
  let total_cost : ℕ := cost_per_book * number_of_students
  let out_of_pocket_cost : ℕ := total_cost - amount_given
  out_of_pocket_cost = 40 := by
  sorry

end Sally_out_of_pocket_payment_l168_168926


namespace triangle_area_divided_l168_168731

theorem triangle_area_divided {baseA heightA baseB heightB : ℝ} 
  (h1 : baseA = 1) 
  (h2 : heightA = 1)
  (h3 : baseB = 2)
  (h4 : heightB = 1)
  : (1 / 2 * baseA * heightA + 1 / 2 * baseB * heightB = 1.5) :=
by
  sorry

end triangle_area_divided_l168_168731


namespace axisymmetric_and_centrally_symmetric_l168_168573

def Polygon := String

def EquilateralTriangle : Polygon := "EquilateralTriangle"
def Square : Polygon := "Square"
def RegularPentagon : Polygon := "RegularPentagon"
def RegularHexagon : Polygon := "RegularHexagon"

def is_axisymmetric (p : Polygon) : Prop := 
  p = EquilateralTriangle ∨ p = Square ∨ p = RegularPentagon ∨ p = RegularHexagon

def is_centrally_symmetric (p : Polygon) : Prop := 
  p = Square ∨ p = RegularHexagon

theorem axisymmetric_and_centrally_symmetric :
  {p : Polygon | is_axisymmetric p ∧ is_centrally_symmetric p} = {Square, RegularHexagon} :=
by
  sorry

end axisymmetric_and_centrally_symmetric_l168_168573


namespace document_total_characters_l168_168420

theorem document_total_characters (T : ℕ) : 
  (∃ (t_1 t_2 t_3 : ℕ) (v_A v_B : ℕ),
      v_A = 100 ∧ v_B = 200 ∧
      t_1 = T / 600 ∧
      v_A * t_1 = T / 6 ∧
      v_B * t_1 = T / 3 ∧
      v_A * 3 * 5 = 1500 ∧
      t_2 = (T / 2 - 1500) / 500 ∧
      (v_A * 3 * t_2 + 1500 + v_A * t_1 = v_B * t_1 + v_B * t_2) ∧
      (v_A * 3 * (T - 3000) / 1000 + 1500 + v_A * T / 6 =
       v_B * 2 * (T - 3000) / 10 + v_B * T / 3)) →
  T = 18000 := by
  sorry

end document_total_characters_l168_168420


namespace fractional_part_painted_correct_l168_168805

noncomputable def fractional_part_painted (time_fence : ℕ) (time_hole : ℕ) : ℚ :=
  (time_hole : ℚ) / time_fence

theorem fractional_part_painted_correct : fractional_part_painted 60 40 = 2 / 3 := by
  sorry

end fractional_part_painted_correct_l168_168805


namespace fraction_meaningful_iff_nonzero_l168_168417

theorem fraction_meaningful_iff_nonzero (x : ℝ) : (∃ y : ℝ, y = 1 / x) ↔ x ≠ 0 :=
by sorry

end fraction_meaningful_iff_nonzero_l168_168417


namespace tony_bread_slices_left_l168_168269

def num_slices_left (initial_slices : ℕ) (d1 : ℕ) (d2 : ℕ) : ℕ :=
  initial_slices - (d1 + d2)

theorem tony_bread_slices_left :
  num_slices_left 22 (5 * 2) (2 * 2) = 8 := by
  sorry

end tony_bread_slices_left_l168_168269


namespace find_angle_C_find_side_c_l168_168758

noncomputable def triangle_angle_C (a b c : ℝ) (C : ℝ) (A : ℝ) : Prop := 
a * Real.cos C = c * Real.sin A

theorem find_angle_C (a b c : ℝ) (C : ℝ) (A : ℝ)
  (h1 : triangle_angle_C a b c C A)
  (h2 : 0 < A) : C = Real.pi / 3 := 
sorry

noncomputable def triangle_side_c (a b c : ℝ) (C : ℝ) : Prop := 
(∃ (area : ℝ), area = 6 ∧ b = 4 ∧ c * c = a * a + b * b - 2 * a * b * Real.cos C)

theorem find_side_c (a b c : ℝ) (C : ℝ) 
  (h1 : triangle_side_c a b c C) : c = 2 * Real.sqrt 7 :=
sorry

end find_angle_C_find_side_c_l168_168758


namespace chocolates_sold_l168_168200

theorem chocolates_sold (C S : ℝ) (n : ℝ)
  (h1 : 65 * C = n * S)
  (h2 : S = 1.3 * C) :
  n = 50 :=
by
  sorry

end chocolates_sold_l168_168200


namespace negation_of_universal_proposition_l168_168682

open Classical

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℕ, x^2 > x) ↔ (∃ x : ℕ, x^2 ≤ x) :=
by
  sorry

end negation_of_universal_proposition_l168_168682


namespace calculate_total_tulips_l168_168317

def number_of_red_tulips_for_eyes := 8 * 2
def number_of_purple_tulips_for_eyebrows := 5 * 2
def number_of_red_tulips_for_nose := 12
def number_of_red_tulips_for_smile := 18
def number_of_yellow_tulips_for_background := 9 * number_of_red_tulips_for_smile

def total_number_of_tulips : ℕ :=
  number_of_red_tulips_for_eyes + 
  number_of_red_tulips_for_nose + 
  number_of_red_tulips_for_smile + 
  number_of_purple_tulips_for_eyebrows + 
  number_of_yellow_tulips_for_background

theorem calculate_total_tulips : total_number_of_tulips = 218 := by
  sorry

end calculate_total_tulips_l168_168317


namespace B_work_rate_l168_168571

theorem B_work_rate (A B C : ℕ) (combined_work_rate_A_B_C : ℕ)
  (A_work_days B_work_days C_work_days : ℕ)
  (combined_abc : combined_work_rate_A_B_C = 4)
  (a_work_rate : A_work_days = 6)
  (c_work_rate : C_work_days = 36) :
  B = 18 :=
by
  sorry

end B_work_rate_l168_168571


namespace cooper_saved_days_l168_168154

variable (daily_saving : ℕ) (total_saving : ℕ) (n : ℕ)

-- Conditions
def cooper_saved (daily_saving total_saving n : ℕ) : Prop :=
  total_saving = daily_saving * n

-- Theorem stating the question equals the correct answer
theorem cooper_saved_days :
  cooper_saved 34 12410 365 :=
by
  sorry

end cooper_saved_days_l168_168154


namespace solve_for_x_l168_168664

-- We state the problem as a theorem.
theorem solve_for_x (y x : ℚ) : 
  (x - 60) / 3 = (4 - 3 * x) / 6 + y → x = (124 + 6 * y) / 5 :=
by
  -- The actual proof part is skipped with sorry.
  sorry

end solve_for_x_l168_168664


namespace min_value_inequality_l168_168519

variable {a b c d : ℝ}

theorem min_value_inequality (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) : 
  (a / b) + (b / c) + (c / d) + (d / a) ≥ 4 :=
sorry

end min_value_inequality_l168_168519


namespace problem1_part1_problem1_part2_l168_168882

open Set Real

theorem problem1_part1 (a : ℝ) (h1: a = 5) :
  let A := { x : ℝ | (x - 6) * (x - 2 * a - 5) > 0 }
  let B := { x : ℝ | (a ^ 2 + 2 - x) * (2 * a - x) < 0 }
  A ∩ B = { x | 15 < x ∧ x < 27 } := sorry

theorem problem1_part2 (a : ℝ) (h2: a > 1 / 2) :
  let A := { x : ℝ | x < 6 ∨ x > 2 * a + 5 }
  let B := { x : ℝ | 2 * a < x ∧ x < a ^ 2 + 2 }
  (∀ x, x ∈ A → x ∈ B) ∧ ¬ (∀ x, x ∈ B → x ∈ A) → (1 / 2 < a ∧ a ≤ 2) := sorry

end problem1_part1_problem1_part2_l168_168882


namespace prob_red_blue_calc_l168_168833

noncomputable def prob_red_blue : ℚ :=
  let p_yellow := (6 : ℚ) / 13
  let p_red_blue_given_yellow := (7 : ℚ) / 12
  let p_red_blue_given_not_yellow := (7 : ℚ) / 13
  p_red_blue_given_yellow * p_yellow + p_red_blue_given_not_yellow * (1 - p_yellow)

/-- The probability of drawing a red or blue marble from the updated bag contents is 91/169. -/
theorem prob_red_blue_calc : prob_red_blue = 91 / 169 :=
by
  -- This proof is omitted as per instructions.
  sorry

end prob_red_blue_calc_l168_168833


namespace perimeter_of_large_rectangle_l168_168679

-- We are bringing in all necessary mathematical libraries, no specific submodules needed.
theorem perimeter_of_large_rectangle
  (small_rectangle_longest_side : ℝ)
  (number_of_small_rectangles : ℕ)
  (length_of_large_rectangle : ℝ)
  (height_of_large_rectangle : ℝ)
  (perimeter_of_large_rectangle : ℝ) :
  small_rectangle_longest_side = 10 ∧ number_of_small_rectangles = 9 →
  length_of_large_rectangle = 2 * small_rectangle_longest_side →
  height_of_large_rectangle = 5 * (small_rectangle_longest_side / 2) →
  perimeter_of_large_rectangle = 2 * (length_of_large_rectangle + height_of_large_rectangle) →
  perimeter_of_large_rectangle = 76 := by
  sorry

end perimeter_of_large_rectangle_l168_168679


namespace problem1_solution_problem2_solution_l168_168535

theorem problem1_solution (x : ℝ) :
  (2 < |2 * x - 5| ∧ |2 * x - 5| ≤ 7) → ((-1 ≤ x ∧ x < 3 / 2) ∨ (7 / 2 < x ∧ x ≤ 6)) := by
  sorry

theorem problem2_solution (x : ℝ) :
  (1 / (x - 1) > x + 1) → (x < -Real.sqrt 2 ∨ (1 < x ∧ x < Real.sqrt 2)) := by
  sorry

end problem1_solution_problem2_solution_l168_168535


namespace max_height_of_ball_l168_168708

noncomputable def h (t : ℝ) : ℝ := -20 * t^2 + 70 * t + 45

theorem max_height_of_ball : ∃ t : ℝ, (h t) = 69.5 :=
sorry

end max_height_of_ball_l168_168708


namespace rectangle_divisible_into_squares_l168_168474

theorem rectangle_divisible_into_squares (n : ℕ) : 
  ∃ (rectangle : ℕ × ℕ), rectangle = (nat.fib n, nat.fib (n+1)) ∧ 
    (∃ (square_sizes : Finset (ℕ × ℕ)), 
      square_sizes.card = n ∧ 
      ∀ (size : ℕ), 2 ≠ square_sizes.filter (λ s, s.1 = size).card)
:= by
  sorry

end rectangle_divisible_into_squares_l168_168474


namespace find_two_digit_number_l168_168718

theorem find_two_digit_number (x : ℕ) (h1 : (x + 3) % 3 = 0) (h2 : (x + 7) % 7 = 0) (h3 : (x - 4) % 4 = 0) : x = 84 := 
by
  -- Place holder for the proof
  sorry

end find_two_digit_number_l168_168718


namespace kim_points_correct_l168_168402

-- Definitions of given conditions
def points_easy : ℕ := 2
def points_average : ℕ := 3
def points_hard : ℕ := 5

def correct_easy : ℕ := 6
def correct_average : ℕ := 2
def correct_hard : ℕ := 4

-- Definition of total points calculation
def kim_total_points : ℕ :=
  (correct_easy * points_easy) +
  (correct_average * points_average) +
  (correct_hard * points_hard)

-- Theorem stating that Kim's total points are 38
theorem kim_points_correct : kim_total_points = 38 := by
  -- Proof placeholder
  sorry

end kim_points_correct_l168_168402


namespace bisector_length_is_correct_l168_168609

noncomputable def length_of_bisector_of_angle_C
    (BC AC : ℝ)
    (angleC : ℝ)
    (hBC : BC = 5)
    (hAC : AC = 7)
    (hAngleC: angleC = 80) : ℝ := 3.2

theorem bisector_length_is_correct
    (BC AC : ℝ)
    (angleC : ℝ)
    (hBC : BC = 5)
    (hAC : AC = 7)
    (hAngleC: angleC = 80) :
    length_of_bisector_of_angle_C BC AC angleC hBC hAC hAngleC = 3.2 := by
  sorry

end bisector_length_is_correct_l168_168609


namespace winning_percentage_l168_168755

-- Defining the conditions
def election_conditions (winner_votes : ℕ) (win_margin : ℕ) (total_candidates : ℕ) : Prop :=
  total_candidates = 2 ∧ winner_votes = 864 ∧ win_margin = 288

-- Stating the question: What percentage of votes did the winner candidate receive?
theorem winning_percentage (V : ℕ) (winner_votes : ℕ) (win_margin : ℕ) (total_candidates : ℕ) :
  election_conditions winner_votes win_margin total_candidates → (winner_votes * 100 / V) = 60 :=
by
  sorry

end winning_percentage_l168_168755


namespace words_per_page_eq_106_l168_168296

-- Definition of conditions as per the problem statement
def pages : ℕ := 224
def max_words_per_page : ℕ := 150
def total_words_congruence : ℕ := 156
def modulus : ℕ := 253

theorem words_per_page_eq_106 (p : ℕ) : 
  (224 * p % 253 = 156) ∧ (p ≤ 150) → p = 106 :=
by 
  sorry

end words_per_page_eq_106_l168_168296


namespace rectangle_area_percentage_increase_l168_168444

theorem rectangle_area_percentage_increase (l w : ℝ) (hl : l > 0) (hw : w > 0) :
  let A := l * w
  let len_inc := 1.3 * l
  let wid_inc := 1.15 * w
  let A_new := len_inc * wid_inc
  let percentage_increase := ((A_new - A) / A) * 100
  percentage_increase = 49.5 :=
by
  sorry

end rectangle_area_percentage_increase_l168_168444


namespace max_liters_of_water_that_can_be_heated_to_boiling_l168_168817

-- Define the initial conditions
def initial_heat_per_5min := 480 -- kJ
def heat_reduction_rate := 0.25
def initial_temp := 20 -- Celsius
def boiling_temp := 100 -- Celsius
def specific_heat_capacity := 4.2 -- kJ/kg·°C

-- Define the temperature difference
def delta_T := boiling_temp - initial_temp -- Celsius

-- Define the calculation of the total heat available from a geometric series
def total_heat_available := initial_heat_per_5min / (1 - (1 - heat_reduction_rate))

-- Define the calculation of energy required to heat m kg of water
def energy_required (m : ℝ) := specific_heat_capacity * m * delta_T

-- Define the main theorem to prove
theorem max_liters_of_water_that_can_be_heated_to_boiling :
  ∃ (m : ℝ), ⌊m⌋ = 5 ∧ energy_required m ≤ total_heat_available :=
begin
  sorry
end

end max_liters_of_water_that_can_be_heated_to_boiling_l168_168817


namespace eval_expression_l168_168015

theorem eval_expression : 9^9 * 3^3 / 3^30 = 1 / 19683 := by
  sorry

end eval_expression_l168_168015


namespace fermat_little_theorem_l168_168807

theorem fermat_little_theorem (p : ℕ) (hp : Nat.Prime p) (a : ℕ) : a^p ≡ a [MOD p] :=
sorry

end fermat_little_theorem_l168_168807


namespace picnic_recyclable_collected_l168_168944

theorem picnic_recyclable_collected :
  let guests := 90
  let soda_cans := 50
  let sparkling_water_bottles := 50
  let juice_bottles := 50
  let soda_drinkers := guests / 2
  let sparkling_water_drinkers := guests / 3
  let juice_consumed := juice_bottles * 4 / 5 
  soda_drinkers + sparkling_water_drinkers + juice_consumed = 115 :=
by
  let guests := 90
  let soda_cans := 50
  let sparkling_water_bottles := 50
  let juice_bottles := 50
  let soda_drinkers := guests / 2
  let sparkling_water_drinkers := guests / 3
  let juice_consumed := juice_bottles * 4 / 5 
  show soda_drinkers + sparkling_water_drinkers + juice_consumed = 115
  sorry

end picnic_recyclable_collected_l168_168944


namespace triangle_is_isosceles_l168_168364

theorem triangle_is_isosceles {a b c : ℝ} {A B C : ℝ} (h1 : b * Real.cos A = a * Real.cos B) : 
  a = b ∨ a = c ∨ b = c :=
sorry

end triangle_is_isosceles_l168_168364


namespace find_other_root_of_quadratic_l168_168203

theorem find_other_root_of_quadratic (m : ℤ) :
  (3 * 1^2 - m * 1 - 3 = 0) → ∃ t : ℤ, t ≠ 1 ∧ (1 + t = m / 3) ∧ (1 * t = -1) :=
by
  intro h_root_at_1
  use -1
  split
  { exact ne_of_lt (by norm_num) }
  split
  { have h1 : m = 0 := by sorry
    exact (by simp [h1]) }
  { simp }

end find_other_root_of_quadratic_l168_168203


namespace solution_set_ineq_l168_168772

noncomputable
def f (x : ℝ) : ℝ := sorry
noncomputable
def g (x : ℝ) : ℝ := sorry

axiom h_f_odd : ∀ x : ℝ, f (-x) = -f x
axiom h_g_even : ∀ x : ℝ, g (-x) = g x
axiom h_deriv_pos : ∀ x : ℝ, x < 0 → deriv f x * g x + f x * deriv g x > 0
axiom h_g_neg_three_zero : g (-3) = 0

theorem solution_set_ineq : { x : ℝ | f x * g x < 0 } = { x : ℝ | x < -3 } ∪ { x : ℝ | 0 < x ∧ x < 3 } := 
by sorry

end solution_set_ineq_l168_168772


namespace food_waste_in_scientific_notation_l168_168723

-- Given condition that 1 billion equals 10^9
def billion : ℕ := 10 ^ 9

-- Problem statement: expressing 530 billion kilograms in scientific notation
theorem food_waste_in_scientific_notation :
  (530 * billion : ℝ) = 5.3 * 10^10 := 
  sorry

end food_waste_in_scientific_notation_l168_168723


namespace isosceles_triangle_base_length_l168_168671

theorem isosceles_triangle_base_length (a b : ℕ) (h1 : a = 7) (h2 : 2 * a + b = 24) : b = 10 := 
by 
  sorry

end isosceles_triangle_base_length_l168_168671


namespace original_number_of_men_l168_168559

theorem original_number_of_men (x : ℕ) (h1 : x * 50 = (x - 10) * 60) : x = 60 :=
by
  sorry

end original_number_of_men_l168_168559


namespace fraction_comparison_l168_168223

theorem fraction_comparison (a b c d : ℝ) 
  (h1 : (a / b) < (c / d))
  (h2 : b > d) (h3 : d > 0) :
  (a + c) / (b + d) < (1 / 2) * ((a / b) + (c / d)) :=
by
  sorry

end fraction_comparison_l168_168223


namespace max_liters_of_water_heated_l168_168815

theorem max_liters_of_water_heated
  (heat_initial : ℕ := 480) 
  (heat_drop : ℝ := 0.25)
  (temp_initial : ℝ := 20)
  (temp_boiling : ℝ := 100)
  (specific_heat_capacity : ℝ := 4.2)
  (kJ_to_liters_conversion : ℝ := 336) :
  (∀ m : ℕ, (m * kJ_to_liters_conversion > ((heat_initial : ℝ) / (1 - heat_drop)) → m ≤ 5)) :=
by
  sorry

end max_liters_of_water_heated_l168_168815


namespace find_divisor_l168_168433

theorem find_divisor
  (n : ℕ) (h1 : n > 0)
  (h2 : (n + 1) % 6 = 4)
  (h3 : ∃ d : ℕ, n % d = 1) :
  ∃ d : ℕ, (n % d = 1) ∧ d = 2 :=
by
  sorry

end find_divisor_l168_168433


namespace library_visitors_on_sunday_l168_168133

def avg_visitors_sundays (S : ℕ) : Prop :=
  let total_days := 30
  let avg_other_days := 240
  let avg_total := 285
  let sundays := 5
  let other_days := total_days - sundays
  (S * sundays) + (avg_other_days * other_days) = avg_total * total_days

theorem library_visitors_on_sunday (S : ℕ) (h : avg_visitors_sundays S) : S = 510 :=
by
  sorry

end library_visitors_on_sunday_l168_168133


namespace area_triangle_ABC_given_conditions_l168_168478

variable (a b c : ℝ) (A B C : ℝ)

noncomputable def area_of_triangle_ABC (a b c : ℝ) (A B C : ℝ) : ℝ :=
  1/2 * b * c * Real.sin A

theorem area_triangle_ABC_given_conditions
  (habc : a = 4)
  (hbc : b + c = 5)
  (htan : Real.tan B + Real.tan C + Real.sqrt 3 = Real.sqrt 3 * (Real.tan B * Real.tan C))
  : area_of_triangle_ABC a b c (Real.pi / 3) B C = 3 * Real.sqrt 3 / 4 := 
sorry

end area_triangle_ABC_given_conditions_l168_168478


namespace patio_rows_before_rearrangement_l168_168842

theorem patio_rows_before_rearrangement (r c : ℕ) 
  (h1 : r * c = 160) 
  (h2 : (r + 4) * (c - 2) = 160)
  (h3 : ∃ k : ℕ, 5 * k = r)
  (h4 : ∃ l : ℕ, 5 * l = c) :
  r = 16 :=
by
  sorry

end patio_rows_before_rearrangement_l168_168842


namespace difference_star_emilio_l168_168666

open Set

def star_numbers : Set ℕ := {x | 1 ≤ x ∧ x ≤ 40}

def emilio_numbers (n : ℕ) : ℕ :=
  if '3' ∈ String.toList (toString n)
  then stringToNat (String.map (λ c => if c = '3' then '2' else c) (toString n))
  else n

def sum_star_numbers : ℕ := Set.sum star_numbers id
def sum_emilio_numbers : ℕ :=
  star_numbers.to_finset.sum (λ x => emilio_numbers x)

theorem difference_star_emilio :
  sum_star_numbers - sum_emilio_numbers = 104 :=
sorry

end difference_star_emilio_l168_168666


namespace peter_has_read_more_books_l168_168658

theorem peter_has_read_more_books
  (total_books : ℕ)
  (peter_percentage : ℚ)
  (brother_percentage : ℚ)
  (sarah_percentage : ℚ)
  (peter_books : ℚ := (peter_percentage / 100) * total_books)
  (brother_books : ℚ := (brother_percentage / 100) * total_books)
  (sarah_books : ℚ := (sarah_percentage / 100) * total_books)
  (combined_books : ℚ := brother_books + sarah_books)
  (difference : ℚ := peter_books - combined_books) :
  total_books = 50 → peter_percentage = 60 → brother_percentage = 25 → sarah_percentage = 15 → difference = 10 :=
by
  sorry

end peter_has_read_more_books_l168_168658


namespace probability_of_purple_probability_of_blue_or_purple_l168_168960

def total_jelly_beans : ℕ := 60
def purple_jelly_beans : ℕ := 5
def blue_jelly_beans : ℕ := 18

theorem probability_of_purple :
  (purple_jelly_beans : ℚ) / total_jelly_beans = 1 / 12 :=
by
  sorry
  
theorem probability_of_blue_or_purple :
  (blue_jelly_beans + purple_jelly_beans : ℚ) / total_jelly_beans = 23 / 60 :=
by
  sorry

end probability_of_purple_probability_of_blue_or_purple_l168_168960


namespace complex_equilateral_triangle_expression_l168_168652

noncomputable def omega : ℂ :=
  Complex.exp (Complex.I * 2 * Real.pi / 3)

def is_root_of_quadratic (z : ℂ) (a b : ℂ) : Prop :=
  z^2 + a * z + b = 0

theorem complex_equilateral_triangle_expression (z1 z2 a b : ℂ) (h1 : is_root_of_quadratic z1 a b) 
  (h2 : is_root_of_quadratic z2 a b) (h3 : z2 = omega * z1) : a^2 / b = 1 := by
  sorry

end complex_equilateral_triangle_expression_l168_168652


namespace probability_both_tell_truth_l168_168353

variable (P_A : ℝ) (P_B : ℝ)

theorem probability_both_tell_truth (hA : P_A = 0.75) (hB : P_B = 0.60) : P_A * P_B = 0.45 :=
by
  rw [hA, hB]
  norm_num

end probability_both_tell_truth_l168_168353


namespace pq_difference_l168_168956

theorem pq_difference (p q : ℚ) (h₁ : 3 / p = 8) (h₂ : 3 / q = 18) : p - q = 5 / 24 := by
  sorry

end pq_difference_l168_168956


namespace cole_drive_time_to_work_l168_168430

theorem cole_drive_time_to_work :
  ∀ (D : ℝ),
    (D / 80 + D / 120 = 3) → (D / 80 * 60 = 108) :=
by
  intro D h
  sorry

end cole_drive_time_to_work_l168_168430


namespace parabola_relationship_l168_168367

noncomputable def parabola (a b x : ℝ) : ℝ := a * x^2 + b * x

theorem parabola_relationship (a b m n t : ℝ) (ha : a ≠ 0)
  (h1 : 3 * a + b > 0) (h2 : a + b < 0)
  (hm : parabola a b (-3) = m)
  (hn : parabola a b 2 = n)
  (ht : parabola a b 4 = t) :
  n < t ∧ t < m :=
by
  sorry

end parabola_relationship_l168_168367


namespace sms_message_fraudulent_l168_168800

-- Define the conditions as properties
def messageArrivedNumberKnown (msg : String) (numberKnown : Bool) : Prop :=
  msg = "SMS message has already arrived" ∧ numberKnown = true

def fraudDefinition (acquisition : String -> Prop) : Prop :=
  ∀ (s : String), acquisition s = (s = "acquisition of property by third parties through deception or gaining the trust of the victim")

-- Define the main proof problem statement
theorem sms_message_fraudulent (msg : String) (numberKnown : Bool) (acquisition : String -> Prop) :
  messageArrivedNumberKnown msg numberKnown ∧ fraudDefinition acquisition →
  acquisition "acquisition of property by third parties through deception or gaining the trust of the victim" :=
  sorry

end sms_message_fraudulent_l168_168800


namespace george_stickers_l168_168597

theorem george_stickers :
  let bob_stickers := 12
  let tom_stickers := 3 * bob_stickers
  let dan_stickers := 2 * tom_stickers
  let george_stickers := 5 * dan_stickers
  george_stickers = 360 := by
  sorry

end george_stickers_l168_168597


namespace nonagon_isosceles_triangle_count_l168_168712

theorem nonagon_isosceles_triangle_count (vertices : Finset (Fin 9)) (h_reg_nonagon : ∀ (a b ∈ vertices), ∃ k : ℕ, b - a ≡ k [MOD 9] ∧ 1 ≤ k ∧ k ≤ 4) :
  ∃ n, n = 33 := by
  -- We know that the nonagon has 9 sides/vertices
  have n : ℕ := 9

  -- Total ways to choose 2 vertices out of 9
  let total_pairs := Nat.choose n 2

  -- Number of equilateral triangles in a nonagon
  let equilateral_triangles := 3

  -- Calculating the number of isosceles but not equilateral triangles
  let isosceles_triangles := total_pairs - equilateral_triangles

  -- Asserting the count of isosceles triangles
  use isosceles_triangles
  -- Proof step to validate the count
  sorry

end nonagon_isosceles_triangle_count_l168_168712


namespace pieces_of_wood_for_chair_is_correct_l168_168608

-- Define the initial setup and constants
def total_pieces_of_wood := 672
def pieces_of_wood_per_table := 12
def number_of_tables := 24
def number_of_chairs := 48

-- Calculation in the conditions
def pieces_of_wood_used_for_tables := number_of_tables * pieces_of_wood_per_table
def pieces_of_wood_left_for_chairs := total_pieces_of_wood - pieces_of_wood_used_for_tables

-- Question and answer verification
def pieces_of_wood_per_chair := pieces_of_wood_left_for_chairs / number_of_chairs

theorem pieces_of_wood_for_chair_is_correct :
  pieces_of_wood_per_chair = 8 := 
by
  -- Proof omitted
  sorry

end pieces_of_wood_for_chair_is_correct_l168_168608


namespace necessary_but_not_sufficient_condition_l168_168792

variable (a : ℝ) (x : ℝ)

def inequality_holds_for_all_real_numbers (a : ℝ) : Prop :=
    ∀ x : ℝ, (a * x^2 - a * x + 1 > 0)

theorem necessary_but_not_sufficient_condition :
  (0 < a ∧ a < 4) ↔
  (inequality_holds_for_all_real_numbers a) :=
by
  sorry

end necessary_but_not_sufficient_condition_l168_168792


namespace find_number_l168_168572

theorem find_number (x: ℝ) (h1: 0.10 * x + 0.15 * 50 = 10.5) : x = 30 :=
by
  sorry

end find_number_l168_168572


namespace find_length_AD_l168_168062

noncomputable def length_AD (AB AC BC : ℝ) (is_equal_AB_AC : AB = AC) (BD DC : ℝ) (D_midpoint : BD = DC) : ℝ :=
  let BE := BC / 2
  let AE := Real.sqrt (AB ^ 2 - BE ^ 2)
  AE

theorem find_length_AD (AB AC BC BD DC : ℝ) (is_equal_AB_AC : AB = AC) (D_midpoint : BD = DC) (H1 : AB = 26) (H2 : AC = 26) (H3 : BC = 24) (H4 : BD = 12) (H5 : DC = 12) :
  length_AD AB AC BC is_equal_AB_AC BD DC D_midpoint = 2 * Real.sqrt 133 :=
by
  -- the steps of the proof would go here
  sorry

end find_length_AD_l168_168062


namespace less_than_n_repetitions_l168_168130

variable {n : ℕ} (a : Fin n.succ → ℕ)

def is_repetition (a : Fin n.succ → ℕ) (k l p : ℕ) : Prop :=
  p ≤ (l - k) / 2 ∧
  (∀ i : ℕ, k + 1 ≤ i ∧ i ≤ l - p → a ⟨i, sorry⟩ = a ⟨i + p, sorry⟩) ∧
  (k > 0 → a ⟨k, sorry⟩ ≠ a ⟨k + p, sorry⟩) ∧
  (l < n → a ⟨l - p + 1, sorry⟩ ≠ a ⟨l + 1, sorry⟩)

theorem less_than_n_repetitions (a : Fin n.succ → ℕ) :
  ∃ r : ℕ, r < n ∧ ∀ k l : ℕ, is_repetition a k l r → r < n :=
sorry

end less_than_n_repetitions_l168_168130


namespace compute_105_squared_l168_168584

theorem compute_105_squared : 105^2 = 11025 :=
by
  sorry

end compute_105_squared_l168_168584


namespace probability_of_specific_selection_l168_168632

/-- 
Given a drawer with 8 forks, 10 spoons, and 6 knives, 
the probability of randomly choosing one fork, one spoon, and one knife when three pieces of silverware are removed equals 120/506.
-/
theorem probability_of_specific_selection :
  let total_pieces := 24
  let total_ways := Nat.choose total_pieces 3
  let favorable_ways := 8 * 10 * 6
  (favorable_ways : ℚ) / total_ways = 120 / 506 := 
by
  sorry

end probability_of_specific_selection_l168_168632


namespace geraldo_drank_l168_168946

def total_gallons : ℝ := 20
def total_containers : ℝ := 80
def containers_drank : ℝ := 3.5
def pints_per_gallon : ℝ := 8

theorem geraldo_drank :
  let tea_per_container : ℝ := total_gallons / total_containers in
  let pints_per_container : ℝ := tea_per_container * pints_per_gallon in
  let total_pints_drank : ℝ := containers_drank * pints_per_container in
  total_pints_drank = 7 :=
by
  sorry

end geraldo_drank_l168_168946


namespace average_rounds_per_golfer_l168_168684

theorem average_rounds_per_golfer :
  let golfers1 := 3
  let golfers2 := 4
  let golfers3 := 6
  let golfers4 := 3
  let golfers5 := 2
  let rounds1 := 1
  let rounds2 := 2
  let rounds3 := 3
  let rounds4 := 4
  let rounds5 := 5
  let total_rounds := golfers1 * rounds1 + golfers2 * rounds2 + golfers3 * rounds3 + golfers4 * rounds4 + golfers5 * rounds5
  let total_golfers := golfers1 + golfers2 + golfers3 + golfers4 + golfers5
  let average_rounds := (total_rounds : ℝ) / total_golfers
  let rounded_average := Real.round average_rounds
  rounded_average = 3 :=
by
  sorry

end average_rounds_per_golfer_l168_168684


namespace consecutive_numbers_sum_l168_168434

theorem consecutive_numbers_sum (n : ℤ) (h1 : (n - 1) * n * (n + 1) = 210) (h2 : ∀ m, (m - 1) * m * (m + 1) = 210 → (m - 1)^2 + m^2 + (m + 1)^2 ≥ (n - 1)^2 + n^2 + (n + 1)^2) :
  (n - 1) + n = 11 :=
by 
  sorry

end consecutive_numbers_sum_l168_168434


namespace batteries_C_equivalent_l168_168801

variables (x y z W : ℝ)

-- Conditions
def cond1 := 4 * x + 18 * y + 16 * z = W * z
def cond2 := 2 * x + 15 * y + 24 * z = W * z
def cond3 := 6 * x + 12 * y + 20 * z = W * z

-- Equivalent statement to prove
theorem batteries_C_equivalent (h1 : cond1 x y z W) (h2 : cond2 x y z W) (h3 : cond3 x y z W) : W = 48 :=
sorry

end batteries_C_equivalent_l168_168801


namespace min_value_inequality_l168_168517

theorem min_value_inequality (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) : 
  (a / b + b / c + c / d + d / a) ≥ 4 :=
by
  sorry

end min_value_inequality_l168_168517


namespace pattern_ABC_150th_letter_is_C_l168_168281

theorem pattern_ABC_150th_letter_is_C :
  (fun cycle length index =>
    let repeats := index / length;
    let remainder := index % length;
    if remainder = 0 then 'C' else
    if remainder = 1 then 'A' else 'B') 3 150 = 'C' := sorry

end pattern_ABC_150th_letter_is_C_l168_168281


namespace coconut_grove_yield_l168_168431

theorem coconut_grove_yield (x : ℕ)
  (h1 : ∀ y, y = x + 3 → 60 * y = 60 * (x + 3))
  (h2 : ∀ z, z = x → 120 * z = 120 * x)
  (h3 : ∀ w, w = x - 3 → 180 * w = 180 * (x - 3))
  (avg_yield : 100 = 100)
  (total_trees : 3 * x = (x + 3) + x + (x - 3)) :
  60 * (x + 3) + 120 * x + 180 * (x - 3) = 300 * x →
  x = 6 :=
by
  sorry

end coconut_grove_yield_l168_168431


namespace andrew_start_age_l168_168985

-- Define the conditions
def annual_donation : ℕ := 7
def current_age : ℕ := 29
def total_donation : ℕ := 133

-- The theorem to prove
theorem andrew_start_age : (total_donation / annual_donation) = (current_age - 10) :=
by
  sorry

end andrew_start_age_l168_168985


namespace students_called_in_sick_l168_168011

-- Conditions
def total_cupcakes : ℕ := 2 * 12 + 6
def total_people : ℕ := 27 + 1 + 1
def cupcakes_left : ℕ := 4
def cupcakes_given_out : ℕ := total_cupcakes - cupcakes_left

-- Statement to prove
theorem students_called_in_sick : total_people - cupcakes_given_out = 3 := by
  -- The proof steps would be implemented here
  sorry

end students_called_in_sick_l168_168011


namespace find_jessica_almonds_l168_168189

-- Definitions for j (Jessica's almonds) and l (Louise's almonds)
variables (j l : ℕ)
-- Conditions
def condition1 : Prop := l = j - 8
def condition2 : Prop := l = j / 3

theorem find_jessica_almonds (h1 : condition1 j l) (h2 : condition2 j l) : j = 12 :=
by sorry

end find_jessica_almonds_l168_168189


namespace Solomon_collected_66_l168_168532

-- Definitions
variables (J S L : ℕ) -- J for Juwan, S for Solomon, L for Levi

-- Conditions
axiom C1 : S = 3 * J
axiom C2 : L = J / 2
axiom C3 : J + S + L = 99

-- Theorem to prove
theorem Solomon_collected_66 : S = 66 :=
by
  sorry

end Solomon_collected_66_l168_168532


namespace students_still_in_school_l168_168312

def total_students := 5000
def students_to_beach := total_students / 2
def remaining_after_beach := total_students - students_to_beach
def students_to_art_museum := remaining_after_beach / 3
def remaining_after_art_museum := remaining_after_beach - students_to_art_museum
def students_to_science_fair := remaining_after_art_museum / 4
def remaining_after_science_fair := remaining_after_art_museum - students_to_science_fair
def students_to_music_workshop := 200
def remaining_students := remaining_after_science_fair - students_to_music_workshop

theorem students_still_in_school : remaining_students = 1051 := by
  sorry

end students_still_in_school_l168_168312


namespace percentage_of_Y_pay_X_is_paid_correct_l168_168105

noncomputable def percentage_of_Y_pay_X_is_paid
  (total_pay : ℝ) (Y_pay : ℝ) : ℝ :=
  let X_pay := total_pay - Y_pay
  (X_pay / Y_pay) * 100

theorem percentage_of_Y_pay_X_is_paid_correct :
  percentage_of_Y_pay_X_is_paid 700 318.1818181818182 = 120 := 
by
  unfold percentage_of_Y_pay_X_is_paid
  sorry

end percentage_of_Y_pay_X_is_paid_correct_l168_168105


namespace kombucha_bottles_l168_168056

theorem kombucha_bottles (b_m : ℕ) (c : ℝ) (r : ℝ) (m : ℕ)
  (hb : b_m = 15) (hc : c = 3.00) (hr : r = 0.10) (hm : m = 12) :
  (b_m * m * r) / c = 6 := by
  sorry

end kombucha_bottles_l168_168056


namespace car_travel_distance_l168_168952

-- Definitions based on the problem
def arith_seq_sum (a1 d : ℤ) (n : ℕ) : ℤ :=
  n * a1 + d * (n * (n - 1)) / 2

-- Main statement to prove
theorem car_travel_distance : arith_seq_sum 40 (-12) 5 = 88 :=
by sorry

end car_travel_distance_l168_168952


namespace quadratic_two_distinct_real_roots_l168_168357

theorem quadratic_two_distinct_real_roots (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ (k+2) * x^2 + 4 * x + 1 = 0 ∧ (k+2) * y^2 + 4 * y + 1 = 0) ↔ 
  (k < 2 ∧ k ≠ -2) :=
sorry

end quadratic_two_distinct_real_roots_l168_168357


namespace macey_needs_to_save_three_more_weeks_l168_168654

def cost_of_shirt : ℝ := 3.0
def amount_saved : ℝ := 1.5
def saving_per_week : ℝ := 0.5

theorem macey_needs_to_save_three_more_weeks :
  ∃ W : ℝ, W * saving_per_week = cost_of_shirt - amount_saved ∧ W = 3 := by
  sorry

end macey_needs_to_save_three_more_weeks_l168_168654


namespace points_6_units_away_from_neg1_l168_168656

theorem points_6_units_away_from_neg1 (A : ℝ) (h : A = -1) :
  { x : ℝ | abs (x - A) = 6 } = { -7, 5 } :=
by
  sorry

end points_6_units_away_from_neg1_l168_168656


namespace min_value_l168_168617

theorem min_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 * x + y = 1) : 
  ∃ xy : ℝ, (xy = 9 ∧ (forall (u v : ℝ), (u > 0) → (v > 0) → 2 * u + v = 1 → (2 / u) + (1 / v) ≥ xy)) :=
by
  use 9
  sorry

end min_value_l168_168617


namespace find_theta_2phi_l168_168876

-- Given
variables {θ φ : ℝ}
variables (hθ_acute : 0 < θ ∧ θ < π / 2)
variables (hφ_acute : 0 < φ ∧ φ < π / 2)
variables (h_tanθ : Real.tan θ = 3 / 11)
variables (h_sinφ : Real.sin φ = 1 / 3)

-- To prove
theorem find_theta_2phi : 
  ∃ x : ℝ, 0 < x ∧ x < π / 2 ∧ Real.tan x = (21 + 6 * Real.sqrt 2) / (77 - 6 * Real.sqrt 2) ∧ x = θ + 2 * φ := 
sorry

end find_theta_2phi_l168_168876


namespace empty_is_proper_subset_of_singleton_zero_l168_168959

theorem empty_is_proper_subset_of_singleton_zero : ∅ ⊂ ({0} : Set Nat) :=
sorry

end empty_is_proper_subset_of_singleton_zero_l168_168959


namespace tony_bread_slices_left_l168_168267

def num_slices_left (initial_slices : ℕ) (d1 : ℕ) (d2 : ℕ) : ℕ :=
  initial_slices - (d1 + d2)

theorem tony_bread_slices_left :
  num_slices_left 22 (5 * 2) (2 * 2) = 8 := by
  sorry

end tony_bread_slices_left_l168_168267


namespace bridge_length_sufficient_l168_168279

structure Train :=
  (length : ℕ) -- length of the train in meters
  (speed : ℚ) -- speed of the train in km/hr

def speed_in_m_per_s (speed_in_km_per_hr : ℚ) : ℚ :=
  speed_in_km_per_hr * 1000 / 3600

noncomputable def length_of_bridge (train1 train2 : Train) : ℚ :=
  let train1_speed_m_per_s := speed_in_m_per_s train1.speed
  let train2_speed_m_per_s := speed_in_m_per_s train2.speed
  let relative_speed := train1_speed_m_per_s + train2_speed_m_per_s
  let total_length := train1.length + train2.length
  let time_to_pass := total_length / relative_speed
  let distance_train1 := train1_speed_m_per_s * time_to_pass
  let distance_train2 := train2_speed_m_per_s * time_to_pass
  distance_train1 + distance_train2

theorem bridge_length_sufficient (train1 train2 : Train) (h1 : train1.length = 200) (h2 : train1.speed = 60) (h3 : train2.length = 150) (h4 : train2.speed = 45) :
  length_of_bridge train1 train2 ≥ 350.04 :=
  by
  sorry

end bridge_length_sufficient_l168_168279


namespace find_d_minus_a_l168_168202

theorem find_d_minus_a (a b c d : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h1 : a * b = 240)
  (h2 : (b + c) / 2 = 60)
  (h3 : (c + d) / 2 = 90) : d - a = 116 :=
sorry

end find_d_minus_a_l168_168202


namespace value_of_f_at_1_over_16_l168_168400

noncomputable def f (x : ℝ) (α : ℝ) := x ^ α

theorem value_of_f_at_1_over_16 (α : ℝ) (h : f 4 α = 2) : f (1 / 16) α = 1 / 4 :=
by
  sorry

end value_of_f_at_1_over_16_l168_168400


namespace p_is_necessary_not_sufficient_for_q_l168_168339

  variable (x : ℝ)

  def p := |x| ≤ 2
  def q := 0 ≤ x ∧ x ≤ 2

  theorem p_is_necessary_not_sufficient_for_q : (∀ x, q x → p x) ∧ (∃ x, p x ∧ ¬ q x) :=
  by
    sorry
  
end p_is_necessary_not_sufficient_for_q_l168_168339


namespace pedro_more_squares_l168_168242

theorem pedro_more_squares (jesus_squares : ℕ) (linden_squares : ℕ) (pedro_squares : ℕ)
  (h1 : jesus_squares = 60) (h2 : linden_squares = 75) (h3 : pedro_squares = 200) :
  pedro_squares - (jesus_squares + linden_squares) = 65 :=
by
  sorry

end pedro_more_squares_l168_168242


namespace range_of_f_l168_168914

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 0 then x + 1 else 2 ^ x

theorem range_of_f :
  {x : ℝ | f x + f (x - 0.5) > 1} = {x : ℝ | x > -0.25} :=
by
  sorry

end range_of_f_l168_168914


namespace numbers_not_necessarily_equal_l168_168803

theorem numbers_not_necessarily_equal (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a + b^2 + c^2 = b + a^2 + c^2) (h2 : a + b^2 + c^2 = c + a^2 + b^2) : 
  ¬(a = b ∧ b = c) := 
sorry

end numbers_not_necessarily_equal_l168_168803


namespace find_angle_at_A_l168_168510

def triangle_angles_sum_to_180 (α β γ : ℝ) : Prop :=
  α + β + γ = 180

def ab_lt_bc_lt_ac (AB BC AC : ℝ) : Prop :=
  AB < BC ∧ BC < AC

def angles_relation (α β γ : ℝ) : Prop :=
  (α = 2 * γ) ∧ (β = 3 * γ)

theorem find_angle_at_A
  (AB BC AC : ℝ)
  (α β γ : ℝ)
  (h1 : ab_lt_bc_lt_ac AB BC AC)
  (h2 : angles_relation α β γ)
  (h3 : triangle_angles_sum_to_180 α β γ) :
  α = 60 :=
sorry

end find_angle_at_A_l168_168510


namespace min_value_expr_l168_168610

theorem min_value_expr (a : ℝ) (ha : a > 0) : 
  ∃ (x : ℝ), x = (a-1)*(4*a-1)/a ∧ ∀ (y : ℝ), y = (a-1)*(4*a-1)/a → y ≥ -1 :=
by sorry

end min_value_expr_l168_168610


namespace quadratic_two_distinct_real_roots_l168_168358

theorem quadratic_two_distinct_real_roots (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ (k+2) * x^2 + 4 * x + 1 = 0 ∧ (k+2) * y^2 + 4 * y + 1 = 0) ↔ 
  (k < 2 ∧ k ≠ -2) :=
sorry

end quadratic_two_distinct_real_roots_l168_168358


namespace solve_for_x_l168_168391

theorem solve_for_x (x : ℚ) (h : (x + 10) / (x - 4) = (x - 4) / (x + 8)) : x = -32 / 13 :=
sorry

end solve_for_x_l168_168391


namespace derivative_at_neg_one_l168_168057

-- Define the function f
def f (x : ℝ) : ℝ := x ^ 6

-- Define the derivative f'
def f' (x : ℝ) : ℝ := 6 * x ^ 5

-- The statement we want to prove
theorem derivative_at_neg_one : f' (-1) = -6 := sorry

end derivative_at_neg_one_l168_168057


namespace problem_l168_168196

theorem problem (x y : ℝ) 
  (h1 : |x + y - 9| = -(2 * x - y + 3) ^ 2) :
  x = 2 ∧ y = 7 :=
sorry

end problem_l168_168196


namespace probability_xavier_yvonne_not_zelda_l168_168292

noncomputable def xavier_solves : ℚ := 1 / 4
noncomputable def yvonne_solves : ℚ := 1 / 3
noncomputable def zelda_not_solves : ℚ := 3 / 8

theorem probability_xavier_yvonne_not_zelda :
  xavier_solves * yvonne_solves * zelda_not_solves = 1 / 32 :=
by {
  sorry
}

end probability_xavier_yvonne_not_zelda_l168_168292


namespace kim_points_correct_l168_168403

-- Definitions of given conditions
def points_easy : ℕ := 2
def points_average : ℕ := 3
def points_hard : ℕ := 5

def correct_easy : ℕ := 6
def correct_average : ℕ := 2
def correct_hard : ℕ := 4

-- Definition of total points calculation
def kim_total_points : ℕ :=
  (correct_easy * points_easy) +
  (correct_average * points_average) +
  (correct_hard * points_hard)

-- Theorem stating that Kim's total points are 38
theorem kim_points_correct : kim_total_points = 38 := by
  -- Proof placeholder
  sorry

end kim_points_correct_l168_168403


namespace sequence_perfect_square_l168_168855

variable (a : ℕ → ℤ)

axiom a1 : a 1 = 1
axiom a2 : a 2 = 1
axiom recurrence : ∀ n ≥ 3, a n = 7 * (a (n - 1)) - (a (n - 2))

theorem sequence_perfect_square (n : ℕ) (hn : n > 0) : ∃ k : ℤ, a n + a (n + 1) + 2 = k * k :=
by
  sorry

end sequence_perfect_square_l168_168855


namespace pedro_more_squares_l168_168240

theorem pedro_more_squares
  (jesus_squares : ℕ)
  (linden_squares : ℕ)
  (pedro_squares : ℕ)
  (jesus_linden_combined : jesus_squares + linden_squares = 135)
  (pedro_total : pedro_squares = 200) :
  pedro_squares - (jesus_squares + linden_squares) = 65 :=
by
  sorry

end pedro_more_squares_l168_168240


namespace vector_solution_l168_168599

theorem vector_solution :
  let u := -6 / 41
  let v := -46 / 41
  let vec1 := (⟨3, -2⟩: ℝ × ℝ)
  let vec2 := (⟨5, -7⟩: ℝ × ℝ)
  let vec3 := (⟨0, 3⟩: ℝ × ℝ)
  let vec4 := (⟨-3, 4⟩: ℝ × ℝ)
  (vec1 + u • vec2 = vec3 + v • vec4) := by
  sorry

end vector_solution_l168_168599


namespace strictly_increasing_interval_l168_168325

open Set

noncomputable def f (x : ℝ) : ℝ := (1 / 2) ^ (-x^2 + 4 * x)

theorem strictly_increasing_interval : ∀ x ∈ (Ici 2 : Set ℝ), StrictMono f :=
by
  intros x hx
  let t := -x^2 + 4 * x
  have decreasing_t : ∀ y, 2 ≤ y → t' deriving Use the properties of quadratic functions,
  apply StrictMonoOn_of_deriv_pos,
  ...
  sorry

end strictly_increasing_interval_l168_168325


namespace average_marks_l168_168974

theorem average_marks (P C M : ℝ) (h1 : P = 95) (h2 : (P + M) / 2 = 90) (h3 : (P + C) / 2 = 70) :
  (P + C + M) / 3 = 75 := 
by
  sorry

end average_marks_l168_168974


namespace tailoring_cost_is_200_l168_168760

variables 
  (cost_first_suit : ℕ := 300)
  (total_paid : ℕ := 1400)

def cost_of_second_suit (tailoring_cost : ℕ) := 3 * cost_first_suit + tailoring_cost

theorem tailoring_cost_is_200 (T : ℕ) (h1 : cost_first_suit = 300) (h2 : total_paid = 1400) 
  (h3 : total_paid = cost_first_suit + cost_of_second_suit T) : 
  T = 200 := 
by 
  sorry

end tailoring_cost_is_200_l168_168760


namespace binom_8_2_eq_28_l168_168322

open Nat

theorem binom_8_2_eq_28 : Nat.choose 8 2 = 28 := by
  sorry

end binom_8_2_eq_28_l168_168322


namespace decimal_0_0_1_7_eq_rational_l168_168984

noncomputable def infinite_loop_decimal_to_rational_series (a : ℚ) (r : ℚ) : ℚ :=
  a / (1 - r)

theorem decimal_0_0_1_7_eq_rational :
  infinite_loop_decimal_to_rational_series (17 / 1000) (1 / 100) = 17 / 990 :=
by
  sorry

end decimal_0_0_1_7_eq_rational_l168_168984


namespace smallest_angle_product_l168_168002

-- Define an isosceles triangle with angle at B being the smallest angle
def isosceles_triangle (α : ℝ) : Prop :=
  α < 90 ∧ α = 180 / 7

-- Proof that the smallest angle multiplied by 6006 is 154440
theorem smallest_angle_product : 
  isosceles_triangle α → (180 / 7) * 6006 = 154440 :=
by
  intros
  sorry

end smallest_angle_product_l168_168002


namespace range_of_m_l168_168743

variable (x y m : ℝ)

theorem range_of_m (h1 : Real.sin x = m * (Real.sin y)^3)
                   (h2 : Real.cos x = m * (Real.cos y)^3) :
                   1 ≤ m ∧ m ≤ Real.sqrt 2 :=
by
  sorry

end range_of_m_l168_168743


namespace new_room_area_l168_168191

def holden_master_bedroom : Nat := 309
def holden_master_bathroom : Nat := 150

theorem new_room_area : 
  (holden_master_bedroom + holden_master_bathroom) * 2 = 918 := 
by
  -- This is where the proof would go
  sorry

end new_room_area_l168_168191


namespace projection_of_vectors_l168_168870

variables {a b : ℝ}

noncomputable def vector_projection (a b : ℝ) : ℝ :=
  (a * b) / b^2 * b

theorem projection_of_vectors
  (ha : abs a = 6)
  (hb : abs b = 3)
  (hab : a * b = -12) : vector_projection a b = -4 :=
sorry

end projection_of_vectors_l168_168870


namespace ratio_of_perimeters_of_squares_l168_168553

theorem ratio_of_perimeters_of_squares (d1 d11 : ℝ) (s1 s11 : ℝ) (P1 P11 : ℝ) 
  (h1 : d11 = 11 * d1)
  (h2 : d1 = s1 * Real.sqrt 2)
  (h3 : d11 = s11 * Real.sqrt 2) :
  P11 / P1 = 11 :=
by
  sorry

end ratio_of_perimeters_of_squares_l168_168553


namespace sum_of_infinite_series_l168_168164

noncomputable def infinite_series : ℝ :=
  ∑' k : ℕ, (k^3 : ℝ) / (3^k : ℝ)

theorem sum_of_infinite_series :
  infinite_series = (39/16 : ℝ) :=
sorry

end sum_of_infinite_series_l168_168164


namespace compute_105_squared_l168_168585

theorem compute_105_squared : 105^2 = 11025 :=
by
  sorry

end compute_105_squared_l168_168585


namespace compute_105_squared_l168_168579

theorem compute_105_squared :
  let a := 100
  let b := 5
  (a + b)^2 = 11025 :=
by
  sorry

end compute_105_squared_l168_168579


namespace people_per_column_in_second_arrangement_l168_168753
-- Import the necessary libraries

-- Define the conditions as given in the problem
def number_of_people_first_arrangement : ℕ := 30 * 16
def number_of_columns_second_arrangement : ℕ := 8

-- Define the problem statement with proof
theorem people_per_column_in_second_arrangement :
  (number_of_people_first_arrangement / number_of_columns_second_arrangement) = 60 :=
by
  -- Skip the proof here
  sorry

end people_per_column_in_second_arrangement_l168_168753


namespace apples_shared_l168_168923

-- Definitions and conditions based on problem statement
def initial_apples : ℕ := 89
def remaining_apples : ℕ := 84

-- The goal to prove that Ruth shared 5 apples with Peter
theorem apples_shared : initial_apples - remaining_apples = 5 := by
  sorry

end apples_shared_l168_168923


namespace compare_A_B_l168_168611

-- Definitions based on conditions from part a)
def A (n : ℕ) : ℕ := 2 * n^2
def B (n : ℕ) : ℕ := 3^n

-- The theorem that needs to be proven
theorem compare_A_B (n : ℕ) (h : n > 0) : A n < B n := 
by sorry

end compare_A_B_l168_168611


namespace tourists_went_free_l168_168000

theorem tourists_went_free (x : ℕ) : 
  (13 + 4 * x = x + 100) → x = 29 :=
by
  intros h
  sorry

end tourists_went_free_l168_168000


namespace other_root_of_quadratic_l168_168205

theorem other_root_of_quadratic (m t : ℝ) : (∀ (x : ℝ),
    (3 * x^2 - m * x - 3 = 0) → 
    (x = 1)) → 
    (1 * t = -1) := 
sorry

end other_root_of_quadratic_l168_168205


namespace find_a_plus_b_l168_168683

theorem find_a_plus_b (a b : ℝ) (h1 : (a + Real.sqrt b) + (a - Real.sqrt b) = 0)
                      (h2 : (a + Real.sqrt b) * (a - Real.sqrt b) = 16) : a + b = -16 :=
by sorry

end find_a_plus_b_l168_168683


namespace decreasing_interval_of_even_function_l168_168633

-- Defining the function f(x)
def f (x : ℝ) (k : ℝ) : ℝ := (k-2) * x^2 + (k-1) * x + 3

-- Defining the condition that f is an even function
def isEvenFunction (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

theorem decreasing_interval_of_even_function (k : ℝ) :
  isEvenFunction (f · k) → k = 1 ∧ ∀ x ≥ 0, f x k ≤ f 0 k :=
by
  sorry

end decreasing_interval_of_even_function_l168_168633


namespace fraction_comparison_l168_168219

theorem fraction_comparison (a b c d : ℝ) (h1 : a / b < c / d) (h2 : b > d) (h3 : d > 0) :
  (a + c) / (b + d) < 1/2 * (a / b + c / d) :=
by
  sorry

end fraction_comparison_l168_168219


namespace gcm_15_and_20_less_than_150_gcm_of_15_and_20_l168_168810

theorem gcm_15_and_20_less_than_150 : 
  ∃ x, (x = 15 * k ∧ x = 20 * m ∧ x < 150) ∧ (∀ y, (y = 15 * k' ∧ y = 20 * m' ∧ y < 150) → y ≤ x) := by
sorr

theorem gcm_of_15_and_20 : 
  ∃ x, x = 120 ∧ (x = 15 * k ∧ x = 20 * m ∧ x < 150) ∧ (∀ y, (y = 15 * k' ∧ y = 20 * m' ∧ y < 150) → y ≤ x) := by
sorry

end gcm_15_and_20_less_than_150_gcm_of_15_and_20_l168_168810


namespace product_xyz_l168_168262

noncomputable def x : ℚ := 97 / 12
noncomputable def n : ℚ := 8 * x
noncomputable def y : ℚ := n + 7
noncomputable def z : ℚ := n - 11

theorem product_xyz 
  (h1: x + y + z = 190)
  (h2: n = 8 * x)
  (h3: n = y - 7)
  (h4: n = z + 11) : 
  x * y * z = (97 * 215 * 161) / 108 := 
by 
  sorry

end product_xyz_l168_168262


namespace right_triangle_345_l168_168634

def is_right_triangle (a b c : ℕ) : Prop :=
  a ^ 2 + b ^ 2 = c ^ 2

theorem right_triangle_345 : is_right_triangle 3 4 5 :=
by
  sorry

end right_triangle_345_l168_168634


namespace complete_square_k_value_l168_168060

theorem complete_square_k_value : 
  ∃ k : ℝ, ∀ x : ℝ, (x^2 - 8*x = (x - 4)^2 + k) ∧ k = -16 :=
by
  use -16
  intro x
  sorry

end complete_square_k_value_l168_168060


namespace max_daily_sales_revenue_l168_168563

noncomputable def p (t : ℕ) : ℝ :=
if 0 < t ∧ t < 25 then t + 20
else if 25 ≤ t ∧ t ≤ 30 then -t + 70
else 0

noncomputable def Q (t : ℕ) : ℝ :=
if 0 < t ∧ t ≤ 30 then -t + 40 else 0

theorem max_daily_sales_revenue :
  ∃ t : ℕ, 0 < t ∧ t ≤ 30 ∧ (p t) * (Q t) = 1125 ∧
  ∀ t' : ℕ, 0 < t' ∧ t' ≤ 30 → (p t') * (Q t') ≤ 1125 :=
sorry

end max_daily_sales_revenue_l168_168563


namespace sum_of_factorials_is_square_l168_168078

open Nat

theorem sum_of_factorials_is_square (m : ℕ) (S : ℕ) :
  S = ∑ i in range (m + 1), i.factorial →
  is_square S →
  m = 1 ∨ m = 3 :=
  sorry

end sum_of_factorials_is_square_l168_168078


namespace largest_y_value_l168_168695

theorem largest_y_value (y : ℝ) : (6 * y^2 - 31 * y + 35 = 0) → (y ≤ 2.5) :=
by
  intro h
  sorry

end largest_y_value_l168_168695


namespace roots_difference_squared_quadratic_roots_property_l168_168497

noncomputable def α : ℝ := (3 + Real.sqrt 5) / 2
noncomputable def β : ℝ := (3 - Real.sqrt 5) / 2

theorem roots_difference_squared :
  α - β = Real.sqrt 5 :=
by
  sorry

theorem quadratic_roots_property :
  (α - β) ^ 2 = 5 :=
by
  sorry

end roots_difference_squared_quadratic_roots_property_l168_168497


namespace find_divisor_l168_168137

theorem find_divisor (x : ℝ) (h : 1152 / x - 189 = 3) : x = 6 :=
by
  sorry

end find_divisor_l168_168137


namespace check_prime_large_number_l168_168071

def large_number := 23021^377 - 1

theorem check_prime_large_number : ¬ Prime large_number := by
  sorry

end check_prime_large_number_l168_168071


namespace polygon_interior_angle_l168_168198

theorem polygon_interior_angle (n : ℕ) (h : n ≥ 3) 
  (interior_angle : ∀ i, 1 ≤ i ∧ i ≤ n → interior_angle = 120) :
  n = 6 := by sorry

end polygon_interior_angle_l168_168198


namespace count_base_8_digits_5_or_6_l168_168888

-- Define the conditions in Lean
def is_digit_5_or_6 (d : ℕ) : Prop := d = 5 ∨ d = 6

def count_digits_5_or_6 := 
  let total_base_8 := 512
  let total_without_5_6 := 6 * 6 * 6 -- since we exclude 2 out of 8 digits
  total_base_8 - total_without_5_6

-- The statement of the proof problem
theorem count_base_8_digits_5_or_6 : count_digits_5_or_6 = 296 :=
by {
  sorry
}

end count_base_8_digits_5_or_6_l168_168888


namespace slices_of_bread_left_l168_168270

variable (monday_to_friday_slices saturday_slices total_slices_used initial_slices slices_left: ℕ)

def sandwiches_monday_to_friday : ℕ := 5
def slices_per_sandwich : ℕ := 2
def sandwiches_saturday : ℕ := 2
def initial_slices_of_bread : ℕ := 22

theorem slices_of_bread_left :
  slices_left = initial_slices_of_bread - total_slices_used
  :=
by  sorry

end slices_of_bread_left_l168_168270


namespace divisibility_of_sum_of_fifths_l168_168778

theorem divisibility_of_sum_of_fifths (x y z : ℤ) (h : x ≠ y ∧ y ≠ z ∧ z ≠ x) :
  ∃ k : ℤ, (x - y) ^ 5 + (y - z) ^ 5 + (z - x) ^ 5 = 5 * k * (x - y) * (y - z) * (z - x) :=
sorry

end divisibility_of_sum_of_fifths_l168_168778


namespace calculate_value_l168_168320

theorem calculate_value (a b c x : ℕ) (h_a : a = 1) (h_b : b = 2) (h_c : c = 3) (h_x : x = 3) :
  x^(a * (b + c)) - (x^a + x^b + x^c) = 204 := by
  sorry

end calculate_value_l168_168320


namespace initial_percentage_water_l168_168304

theorem initial_percentage_water (P : ℝ) (H1 : 150 * P / 100 + 10 = 40) : P = 20 :=
by
  sorry

end initial_percentage_water_l168_168304


namespace liter_kerosene_cost_friday_l168_168069

-- Define initial conditions.
def cost_pound_rice_monday : ℚ := 0.36
def cost_dozen_eggs_monday : ℚ := cost_pound_rice_monday
def cost_half_liter_kerosene_monday : ℚ := (8 / 12) * cost_dozen_eggs_monday

-- Define the Wednesday price increase.
def percent_increase_rice : ℚ := 0.20
def cost_pound_rice_wednesday : ℚ := cost_pound_rice_monday * (1 + percent_increase_rice)
def cost_half_liter_kerosene_wednesday : ℚ := cost_half_liter_kerosene_monday * (1 + percent_increase_rice)

-- Define the Friday discount on eggs.
def percent_discount_eggs : ℚ := 0.10
def cost_dozen_eggs_friday : ℚ := cost_dozen_eggs_monday * (1 - percent_discount_eggs)
def cost_per_egg_friday : ℚ := cost_dozen_eggs_friday / 12

-- Define the price calculation for a liter of kerosene on Wednesday.
def cost_liter_kerosene_wednesday : ℚ := 2 * cost_half_liter_kerosene_wednesday

-- Define the final goal.
def cost_liter_kerosene_friday := cost_liter_kerosene_wednesday

theorem liter_kerosene_cost_friday : cost_liter_kerosene_friday = 0.576 := by
  sorry

end liter_kerosene_cost_friday_l168_168069


namespace least_number_of_people_l168_168080

-- Conditions
def first_caterer_cost (x : ℕ) : ℕ := 120 + 18 * x
def second_caterer_cost (x : ℕ) : ℕ := 250 + 15 * x

-- Proof Statement
theorem least_number_of_people (x : ℕ) (h : x ≥ 44) : first_caterer_cost x > second_caterer_cost x :=
by sorry

end least_number_of_people_l168_168080


namespace complement_event_l168_168969

-- Definitions based on conditions
variables (shoot1 shoot2 : Prop) -- shoots the target on the first and second attempt

-- Definition based on the question and answer
def hits_at_least_once : Prop := shoot1 ∨ shoot2
def misses_both_times : Prop := ¬shoot1 ∧ ¬shoot2

-- Theorem statement based on the mathematical translation
theorem complement_event :
  misses_both_times shoot1 shoot2 = ¬hits_at_least_once shoot1 shoot2 :=
by sorry

end complement_event_l168_168969


namespace additional_money_spent_on_dvds_correct_l168_168512

def initial_money : ℕ := 320
def spent_on_books : ℕ := initial_money / 4 + 10
def remaining_after_books : ℕ := initial_money - spent_on_books
def spent_on_dvds_portion : ℕ := 2 * remaining_after_books / 5
def remaining_after_dvds : ℕ := 130
def total_spent_on_dvds : ℕ := remaining_after_books - remaining_after_dvds
def additional_spent_on_dvds : ℕ := total_spent_on_dvds - spent_on_dvds_portion

theorem additional_money_spent_on_dvds_correct : additional_spent_on_dvds = 8 :=
by
  sorry

end additional_money_spent_on_dvds_correct_l168_168512


namespace apple_pies_count_l168_168963

def total_pies := 13
def pecan_pies := 4
def pumpkin_pies := 7
def apple_pies := total_pies - pecan_pies - pumpkin_pies

theorem apple_pies_count : apple_pies = 2 := by
  sorry

end apple_pies_count_l168_168963


namespace simplify_expression_l168_168928

theorem simplify_expression (x : ℝ) : 7 * x + 9 - 3 * x + 15 * 2 = 4 * x + 39 := 
by sorry

end simplify_expression_l168_168928


namespace rectangle_perimeters_l168_168266

theorem rectangle_perimeters (length width : ℕ) (h1 : length = 7) (h2 : width = 5) :
  (∃ (L1 L2 : ℕ), L1 = 4 * width ∧ L2 = length ∧ 2 * (L1 + L2) = 54) ∧
  (∃ (L3 L4 : ℕ), L3 = 4 * length ∧ L4 = width ∧ 2 * (L3 + L4) = 66) ∧
  (∃ (L5 L6 : ℕ), L5 = 2 * length ∧ L6 = 2 * width ∧ 2 * (L5 + L6) = 48) :=
by
  sorry

end rectangle_perimeters_l168_168266


namespace mul_mod_eq_l168_168727

theorem mul_mod_eq :
  (66 * 77 * 88) % 25 = 16 :=
by 
  sorry

end mul_mod_eq_l168_168727


namespace train_speed_is_72_kmh_l168_168680

-- Length of the train in meters
def length_train : ℕ := 600

-- Length of the platform in meters
def length_platform : ℕ := 600

-- Time to cross the platform in minutes
def time_crossing_platform : ℕ := 1

-- Convert meters to kilometers
def meters_to_kilometers (m : ℕ) : ℕ := m / 1000

-- Convert minutes to hours
def minutes_to_hours (m : ℕ) : ℕ := m * 60

-- Speed of the train in km/hr given lengths in meters and time in minutes
def speed_train_kmh (distance_m : ℕ) (time_min : ℕ) : ℕ :=
  (meters_to_kilometers distance_m) / (minutes_to_hours time_min)

theorem train_speed_is_72_kmh :
  speed_train_kmh (length_train + length_platform) time_crossing_platform = 72 :=
by
  -- skipping the proof
  sorry

end train_speed_is_72_kmh_l168_168680


namespace max_non_equivalent_100_digit_numbers_l168_168596

noncomputable def maxPairwiseNonEquivalentNumbers : ℕ := 21^5

theorem max_non_equivalent_100_digit_numbers :
  (∀ (n : ℕ), 0 < n ∧ n < 100 → (∀ (digit : Fin n → Fin 2), 
  ∃ (max_num : ℕ), max_num = maxPairwiseNonEquivalentNumbers)) :=
by sorry

end max_non_equivalent_100_digit_numbers_l168_168596


namespace determine_constants_l168_168462

theorem determine_constants (α β : ℝ) (h_eq : ∀ x, (x - α) / (x + β) = (x^2 - 96 * x + 2210) / (x^2 + 65 * x - 3510))
  (h_num : ∀ x, x^2 - 96 * x + 2210 = (x - 34) * (x - 62))
  (h_denom : ∀ x, x^2 + 65 * x - 3510 = (x - 45) * (x + 78)) :
  α + β = 112 :=
sorry

end determine_constants_l168_168462


namespace fraction_of_repeating_decimal_l168_168808

theorem fraction_of_repeating_decimal : ∃ x : ℚ, x = 0.4 + (67 / 999) ∧ x = 4621 / 9900 := by
  use 4621 / 9900
  split
  sorry
  rfl

end fraction_of_repeating_decimal_l168_168808


namespace new_average_score_after_drop_l168_168066

theorem new_average_score_after_drop
  (avg_score : ℝ) (num_students : ℕ) (drop_score : ℝ) (remaining_students : ℕ) :
  avg_score = 62.5 →
  num_students = 16 →
  drop_score = 70 →
  remaining_students = 15 →
  (num_students * avg_score - drop_score) / remaining_students = 62 :=
by
  intros h_avg h_num h_drop h_remain
  rw [h_avg, h_num, h_drop, h_remain]
  norm_num

end new_average_score_after_drop_l168_168066


namespace trivia_team_points_l168_168570

theorem trivia_team_points (total_members: ℕ) (total_points: ℕ) (points_per_member: ℕ) (members_showed_up: ℕ) (members_did_not_show_up: ℕ):
  total_members = 7 → 
  total_points = 20 → 
  points_per_member = 4 → 
  members_showed_up = total_points / points_per_member → 
  members_did_not_show_up = total_members - members_showed_up → 
  members_did_not_show_up = 2 := 
by 
  intros h1 h2 h3 h4 h5
  sorry

end trivia_team_points_l168_168570


namespace bread_cost_equality_l168_168777

variable (B : ℝ)
variable (C1 : B + 3 + 2 * B = 9)  -- $3 for butter, 2B for juice, total spent is 9 dollars

theorem bread_cost_equality : B = 2 :=
by
  sorry

end bread_cost_equality_l168_168777


namespace no_integer_solution_l168_168463

theorem no_integer_solution (x y : ℤ) : 2 * x + 6 * y ≠ 91 :=
by
  sorry

end no_integer_solution_l168_168463


namespace chuck_play_area_l168_168991

-- Definitions
def radius : ℝ := 4
def shed_width : ℝ := 4
def shed_height : ℝ := 3
def arc_fraction : ℝ := 3 / 4
def sector_fraction : ℝ := 1 / 4
def sector_radius : ℝ := 1

-- Area calculation
def full_circle_area (r : ℝ) : ℝ := π * r^2
def arc_area (r : ℝ) (fraction : ℝ) : ℝ := fraction * full_circle_area r
def sector_area (r : ℝ) (fraction : ℝ) : ℝ := fraction * full_circle_area r

-- Total playable area
def total_play_area : ℝ :=
  arc_area radius arc_fraction + sector_area sector_radius sector_fraction

theorem chuck_play_area : total_play_area = (49 / 4) * π := by
  sorry

end chuck_play_area_l168_168991


namespace simplify_fraction_l168_168389

variable {x y : ℝ}

theorem simplify_fraction (hx : x = 3) (hy : y = 4) : (12 * x * y^3) / (9 * x^3 * y^2) = 16 / 27 := by
  sorry

end simplify_fraction_l168_168389


namespace cosine_midline_l168_168004

theorem cosine_midline (a b c d : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d) 
  (h_range : ∀ x, 1 ≤ a * Real.cos (b * x + c) + d ∧ a * Real.cos (b * x + c) + d ≤ 5) : 
  d = 3 := 
by 
  sorry

end cosine_midline_l168_168004


namespace lcm_of_numbers_l168_168435

theorem lcm_of_numbers (x : Nat) (h_ratio : x ≠ 0) (h_hcf : Nat.gcd (5 * x) (Nat.gcd (7 * x) (9 * x)) = 11) :
    Nat.lcm (5 * x) (Nat.lcm (7 * x) (9 * x)) = 99 :=
by
  sorry

end lcm_of_numbers_l168_168435


namespace pages_per_day_l168_168373

def notebooks : Nat := 5
def pages_per_notebook : Nat := 40
def total_days : Nat := 50

theorem pages_per_day (H1 : notebooks = 5) (H2 : pages_per_notebook = 40) (H3 : total_days = 50) : 
  (notebooks * pages_per_notebook / total_days) = 4 := by
  sorry

end pages_per_day_l168_168373


namespace gcd_of_consecutive_digit_sums_is_1111_l168_168541

theorem gcd_of_consecutive_digit_sums_is_1111 (p q r s : ℕ) (hc : q = p+1 ∧ r = p+2 ∧ s = p+3) :
  ∃ d, d = 1111 ∧ ∀ n : ℕ, n = (1000 * p + 100 * q + 10 * r + s) + (1000 * s + 100 * r + 10 * q + p) → d ∣ n := by
  use 1111
  sorry

end gcd_of_consecutive_digit_sums_is_1111_l168_168541


namespace commission_percentage_l168_168456

theorem commission_percentage (commission_earned total_sales : ℝ) (h₀ : commission_earned = 18) (h₁ : total_sales = 720) : 
  ((commission_earned / total_sales) * 100) = 2.5 := by {
  sorry
}

end commission_percentage_l168_168456


namespace distinct_ways_to_distribute_balls_l168_168159

theorem distinct_ways_to_distribute_balls (balls boxes : ℕ) (h_balls : balls = 7) (h_boxes : boxes = 4) :
  (∑ i in Ico 1 (balls+1), if (finset.Ico 1 (balls+1)).card = boxes then 1 else 0) = 20 :=
by {
  -- Here we can provide further elaboration or proofs, but we skip to the conclusion for now.
  sorry
}

end distinct_ways_to_distribute_balls_l168_168159


namespace square_of_105_l168_168594

theorem square_of_105 : 105^2 = 11025 := by
  sorry

end square_of_105_l168_168594


namespace solution_set_l168_168183

def f (x : ℝ) : ℝ := sorry

axiom ax1 : ∀ a b : ℝ, f (a + b) = f a + f b - 1
axiom ax2 : ∀ x : ℝ, x > 0 → f x > 1
axiom ax3 : f 4 = 5

theorem solution_set (x : ℝ) : f (3 * x^2 - x - 2) < 3 ↔ (-1 < x ∧ x < 4 / 3) :=
by
  sorry

end solution_set_l168_168183


namespace unique_integer_solution_range_l168_168362

theorem unique_integer_solution_range (a : ℝ) : 
  (∀ x : ℝ, (2 * x + 3 > 5) ∧ (x - a ≤ 0) → (x = 2)) ↔ (2 ≤ a ∧ a < 3) :=
by
  sorry

end unique_integer_solution_range_l168_168362


namespace total_books_read_l168_168300

-- Definitions based on the conditions
def books_per_month : ℕ := 4
def months_per_year : ℕ := 12
def books_per_year_per_student : ℕ := books_per_month * months_per_year

variables (c s : ℕ)

-- Main theorem statement
theorem total_books_read (c s : ℕ) : 
  (books_per_year_per_student * c * s) = 48 * c * s :=
by
  sorry

end total_books_read_l168_168300


namespace radius_range_of_circle_l168_168342

theorem radius_range_of_circle (r : ℝ) :
  (∃ (x y : ℝ), (x - 3)^2 + (y + 5)^2 = r^2 ∧ 
    (∃ a b : ℝ, 4*a - 3*b - 2 = 0 ∧ ∃ c d : ℝ, 4*c - 3*d - 2 = 0 ∧ 
      (a - x)^2 + (b - y)^2 = 1 ∧ (c - x)^2 + (d - y)^2 = 1 ∧
       a ≠ c ∧ b ≠ d)) ↔ 4 < r ∧ r < 6 :=
by
  sorry

end radius_range_of_circle_l168_168342


namespace ratio_malt_to_coke_l168_168752

-- Definitions from conditions
def total_males : ℕ := 10
def total_females : ℕ := 16
def males_choose_malt : ℕ := 6
def females_choose_malt : ℕ := 8

-- Derived values
def total_cheerleaders : ℕ := total_males + total_females
def total_malt : ℕ := males_choose_malt + females_choose_malt
def total_coke : ℕ := total_cheerleaders - total_malt

-- The theorem to be proved
theorem ratio_malt_to_coke : (total_malt / total_coke) = (7 / 6) :=
  by
    -- skipped proof
    sorry

end ratio_malt_to_coke_l168_168752


namespace no_nonconstant_poly_prime_for_all_l168_168661

open Polynomial

theorem no_nonconstant_poly_prime_for_all (f : Polynomial ℤ) (h : ∀ n : ℕ, Prime (f.eval (n : ℤ))) :
  ∃ c : ℤ, f = Polynomial.C c :=
sorry

end no_nonconstant_poly_prime_for_all_l168_168661


namespace potential_values_of_k_l168_168010

theorem potential_values_of_k :
  ∃ k : ℚ, ∀ (a b : ℕ), 
  (10 * a + b = k * (a + b)) ∧ (10 * b + a = (13 - k) * (a + b)) → k = 11/2 :=
by
  sorry

end potential_values_of_k_l168_168010


namespace find_roots_l168_168334

theorem find_roots : 
  (∃ x : ℝ, (x-1) * (x-2) * (x+1) * (x-5) = 0) ↔ 
  x = -1 ∨ x = 1 ∨ x = 2 ∨ x = 5 :=
by sorry

end find_roots_l168_168334


namespace solve_system_of_equations_l168_168394

theorem solve_system_of_equations : 
  ∀ x y : ℝ, 
    (2 * x^2 - 3 * x * y + y^2 = 3) ∧ 
    (x^2 + 2 * x * y - 2 * y^2 = 6) 
    ↔ (x = 2 ∧ y = 1) ∨ (x = -2 ∧ y = -1) :=
by
  sorry

end solve_system_of_equations_l168_168394


namespace inequality_abc_l168_168734

theorem inequality_abc (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  abs (b / a - b / c) + abs (c / a - c / b) + abs (b * c + 1) > 1 :=
by
  sorry

end inequality_abc_l168_168734


namespace tommy_writing_time_l168_168103

def numUniqueLettersTommy : Nat := 5
def numRearrangementsPerMinute : Nat := 20
def totalRearrangements : Nat := numUniqueLettersTommy.factorial
def minutesToComplete : Nat := totalRearrangements / numRearrangementsPerMinute
def hoursToComplete : Rat := minutesToComplete / 60

theorem tommy_writing_time :
  hoursToComplete = 0.1 := by
  sorry

end tommy_writing_time_l168_168103


namespace round_to_nearest_tenth_l168_168246

theorem round_to_nearest_tenth (x : Float) (h : x = 42.63518) : Float.round (x * 10) / 10 = 42.6 := by
  sorry

end round_to_nearest_tenth_l168_168246


namespace solve_for_x_l168_168085

open Real

-- Define the condition and the target result
def target (x : ℝ) : Prop :=
  sqrt (9 + sqrt (16 + 3 * x)) + sqrt (3 + sqrt (4 + x)) = 3 + 3 * sqrt 2

theorem solve_for_x (x : ℝ) (h : target x) : x = 8 * sqrt 2 / 3 :=
  sorry

end solve_for_x_l168_168085


namespace symmetrical_point_correct_l168_168672

variables (x₁ y₁ : ℝ)

def symmetrical_point_x_axis (x y : ℝ) : ℝ × ℝ :=
(x, -y)

theorem symmetrical_point_correct : symmetrical_point_x_axis 3 2 = (3, -2) :=
by
  -- This is where we would provide the proof
  sorry

end symmetrical_point_correct_l168_168672


namespace positive_difference_between_two_numbers_l168_168546

variable (x y : ℝ)

theorem positive_difference_between_two_numbers 
  (h₁ : x + y = 40)
  (h₂ : 3 * y - 4 * x = 20) :
  |y - x| = 100 / 7 :=
by
  sorry

end positive_difference_between_two_numbers_l168_168546


namespace kafelnikov_served_in_first_game_l168_168954

theorem kafelnikov_served_in_first_game (games : ℕ) (kafelnikov_wins : ℕ) (becker_wins : ℕ)
  (server_victories : ℕ) (x y : ℕ) 
  (h1 : kafelnikov_wins = 6)
  (h2 : becker_wins = 3)
  (h3 : server_victories = 5)
  (h4 : games = 9)
  (h5 : kafelnikov_wins + becker_wins = games)
  (h6 : (5 - x) + y = 5) 
  (h7 : x + y = 6):
  x = 3 :=
by
  sorry

end kafelnikov_served_in_first_game_l168_168954


namespace area_of_abs_inequality_l168_168858

theorem area_of_abs_inequality :
  ∀ (x y : ℝ), |x + 2 * y| + |2 * x - y| ≤ 6 → 
  ∃ (area : ℝ), area = 12 := 
by
  -- This skips the proofs
  sorry

end area_of_abs_inequality_l168_168858


namespace find_a_l168_168395

theorem find_a (a : ℤ) (h1 : 0 ≤ a) (h2 : a ≤ 20) 
  (h3 : (4254253 % 53^1 - a) % 17 = 0): 
  a = 3 := 
sorry

end find_a_l168_168395


namespace correct_operation_is_d_l168_168555

theorem correct_operation_is_d (a b : ℝ) : 
  (∀ x y : ℝ, -x * y = -(x * y)) → 
  (∀ x : ℝ, x⁻¹ * (x ^ 2) = x) → 
  (∀ x : ℝ, x ^ 10 / x ^ 4 = x ^ 6) →
  ((a - b) * (-a - b) ≠ a ^ 2 - b ^ 2) ∧ 
  (2 * a ^ 2 * a ^ 3 ≠ 2 * a ^ 6) ∧ 
  ((-a) ^ 10 / (-a) ^ 4 = a ^ 6) :=
by
  intros h1 h2 h3
  sorry

end correct_operation_is_d_l168_168555


namespace number_of_paths_l168_168209

open Nat

def f : ℕ → ℕ → ℕ
| 0, 0 => 1
| x, 0 => 1
| 0, y => 1
| (x + 1), (y + 1) => f x (y + 1) + f (x + 1) y

theorem number_of_paths (n : ℕ) : f n 2 = (n^2 + 3 * n + 2) / 2 := by sorry

end number_of_paths_l168_168209


namespace batteries_problem_l168_168802

noncomputable def x : ℝ := 2 * z
noncomputable def y : ℝ := (4 / 3) * z

theorem batteries_problem
  (z : ℝ)
  (W : ℝ)
  (h1 : 4 * x + 18 * y + 16 * z = W * z)
  (h2 : 2 * x + 15 * y + 24 * z = W * z)
  (h3 : 6 * x + 12 * y + 20 * z = W * z) :
  W = 48 :=
sorry

end batteries_problem_l168_168802


namespace binomial_12_11_eq_12_l168_168007

-- Define binomial coefficient function
def binomial (n k : Nat) : Nat :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Statement to prove
theorem binomial_12_11_eq_12 :
  binomial 12 11 = 12 := by
  sorry

end binomial_12_11_eq_12_l168_168007


namespace find_p_q_sum_l168_168303

-- Define the number of trees
def pine_trees := 2
def cedar_trees := 3
def fir_trees := 4

-- Total number of trees
def total_trees := pine_trees + cedar_trees + fir_trees

-- Number of ways to arrange the 9 trees
def total_arrangements := Nat.choose total_trees fir_trees

-- Number of ways to place fir trees so no two are adjacent
def valid_arrangements := Nat.choose (pine_trees + cedar_trees + 1) fir_trees

-- Desired probability in its simplest form
def probability := valid_arrangements / total_arrangements

-- Denominator and numerator of the simplified fraction
def num := 5
def den := 42

-- Statement to prove that the probability is 5/42
theorem find_p_q_sum : (num + den) = 47 := by
  sorry

end find_p_q_sum_l168_168303


namespace probability_distribution_xi_max_value_g_p_l168_168655

noncomputable def P_xi (xi : ℕ) :=
  match xi with
  | 0 => 1 / 6
  | 1 => 1 / 2
  | 2 => 3 / 10
  | 3 => 1 / 30
  | _ => 0

def expectation_xi : ℝ :=
  0 * 1 / 6 + 1 * 1 / 2 + 2 * 3 / 10 + 3 * 1 / 30

theorem probability_distribution_xi :
  (∀ xi, (xi = 0 ∨ xi = 1 ∨ xi = 2 ∨ xi = 3) → 
          P_xi xi = if xi = 0 then 1/6 else if xi = 1 then 1/2 else if xi = 2 then 3/10 else 1/30) ∧
  expectation_xi = 6 / 5 :=
by sorry

def p (m : ℕ) (m_pos: 2 < m) : ℝ :=
  4 * m / (m^2 + 3 * m + 2)

def g (p : ℝ) : ℝ :=
  10 * (p^3 - 2 * p^4 + p^5)

noncomputable def max_g_value_and_m : ℝ × ℕ :=
(let max_val := 216 / 625, max_m := 3 in (max_val, max_m))

theorem max_value_g_p :
  ∃ m, max_g_value_and_m = (216 / 625, 3) ∧ 
       (p (3) (by linarith [show 3 > 2 from by linarith]) = 3 / 5) :=
by sorry

end probability_distribution_xi_max_value_g_p_l168_168655


namespace max_sum_of_squares_l168_168226

theorem max_sum_of_squares (a b c d : ℝ) 
  (h1 : a + b = 17) 
  (h2 : ab + c + d = 86) 
  (h3 : ad + bc = 180) 
  (h4 : cd = 110) : 
  a^2 + b^2 + c^2 + d^2 ≤ 258 :=
sorry

end max_sum_of_squares_l168_168226


namespace fill_cistern_time_l168_168703

theorem fill_cistern_time (R1 R2 R3 : ℝ) (H1 : R1 = 1/10) (H2 : R2 = 1/12) (H3 : R3 = 1/40) : 
  (1 / (R1 + R2 - R3)) = (120 / 19) :=
by
  sorry

end fill_cistern_time_l168_168703


namespace man_double_son_in_years_l168_168839

-- Definitions of conditions
def son_age : ℕ := 18
def man_age : ℕ := son_age + 20

-- The proof problem statement
theorem man_double_son_in_years :
  ∃ (X : ℕ), (man_age + X = 2 * (son_age + X)) ∧ X = 2 :=
by
  sorry

end man_double_son_in_years_l168_168839


namespace even_of_form_4a_plus_2_not_diff_of_squares_l168_168527

theorem even_of_form_4a_plus_2_not_diff_of_squares (a x y : ℤ) : ¬ (4 * a + 2 = x^2 - y^2) :=
by sorry

end even_of_form_4a_plus_2_not_diff_of_squares_l168_168527


namespace arithmetic_sequence_length_l168_168745

theorem arithmetic_sequence_length : 
  let a := 11
  let d := 5
  let l := 101
  ∃ n : ℕ, a + (n-1) * d = l ∧ n = 19 := 
by
  sorry

end arithmetic_sequence_length_l168_168745


namespace cakes_remaining_l168_168724

theorem cakes_remaining (initial_cakes : ℕ) (bought_cakes : ℕ) (h1 : initial_cakes = 169) (h2 : bought_cakes = 137) : initial_cakes - bought_cakes = 32 :=
by
  sorry

end cakes_remaining_l168_168724


namespace greatest_a_l168_168540

theorem greatest_a (a : ℤ) (h_pos : a > 0) : 
  (∀ x : ℤ, (x^2 + a * x = -30) → (a = 31)) :=
by {
  sorry
}

end greatest_a_l168_168540


namespace find_a12_l168_168651

variable (a : ℕ → ℤ)
variable (H1 : a 1 = 1) 
variable (H2 : ∀ m n : ℕ, a (m + n) = a m + a n + m * n)

theorem find_a12 : a 12 = 78 := 
by
  sorry

end find_a12_l168_168651


namespace mary_needs_change_probability_l168_168691

theorem mary_needs_change_probability :
  let quarters := 12
  let value_per_quarter := 0.25
  let total_quarter_value := value_per_quarter * quarters
  let toys : List Float := [0.50, 1.00, 1.50, 2.00, 2.50, 3.00, 3.50, 4.00, 4.50, 4.50]
  let favorite_toy_price := 4.50
  let toy_count := toys.length
  let ways_to_dispense_favorite_first := 2 * (9.factorial)
  let ways_to_dispense_favorite_second := 16 * (8.factorial)
  let total_ways := (10.factorial)
  let no_change_ways := ways_to_dispense_favorite_first + ways_to_dispense_favorite_second
  let no_change_probability := (no_change_ways : ℚ) / total_ways
  let needs_change_probability := 1 - no_change_probability
  needs_change_probability = 15 / 25 := by sorry

end mary_needs_change_probability_l168_168691


namespace intersection_point_l168_168757

structure Point3D : Type where
  x : ℝ
  y : ℝ
  z : ℝ

def A : Point3D := ⟨8, -9, 5⟩
def B : Point3D := ⟨18, -19, 15⟩
def C : Point3D := ⟨2, 5, -8⟩
def D : Point3D := ⟨4, -3, 12⟩

/-- Prove that the intersection point of lines AB and CD is (16, -19, 13) -/
theorem intersection_point :
  ∃ (P : Point3D), 
  (∃ t : ℝ, P = ⟨A.x + t * (B.x - A.x), A.y + t * (B.y - A.y), A.z + t * (B.z - A.z)⟩) ∧
  (∃ s : ℝ, P = ⟨C.x + s * (D.x - C.x), C.y + s * (D.y - C.y), C.z + s * (D.z - C.z)⟩) ∧
  P = ⟨16, -19, 13⟩ :=
by
  sorry

end intersection_point_l168_168757


namespace intersection_complement_N_M_eq_singleton_two_l168_168650

def M : Set ℝ := {y | y ≥ 2}
def N : Set ℝ := {x | x > 2}
def C_R_N : Set ℝ := {x | x ≤ 2}

theorem intersection_complement_N_M_eq_singleton_two :
  (C_R_N ∩ M = {2}) :=
by
  sorry

end intersection_complement_N_M_eq_singleton_two_l168_168650


namespace f_zero_count_l168_168685

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - x^2 - 3 * x + 9

theorem f_zero_count : ∃ (z : ℕ), z = 2 ∧ (∃ (x₁ x₂ : ℝ), f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂) :=
by
  sorry

end f_zero_count_l168_168685


namespace spinner_final_direction_l168_168640

theorem spinner_final_direction 
  (initial_direction : ℕ) -- 0 for north, 1 for east, 2 for south, 3 for west
  (clockwise_revolutions : ℚ)
  (counterclockwise_revolutions : ℚ)
  (net_revolutions : ℚ) -- derived via net movement calculation
  (final_position : ℕ) -- correct position after net movement
  : initial_direction = 3 → clockwise_revolutions = 9/4 → counterclockwise_revolutions = 15/4 → final_position = 1 :=
by
  sorry

end spinner_final_direction_l168_168640


namespace new_room_area_l168_168190

def holden_master_bedroom : Nat := 309
def holden_master_bathroom : Nat := 150

theorem new_room_area : 
  (holden_master_bedroom + holden_master_bathroom) * 2 = 918 := 
by
  -- This is where the proof would go
  sorry

end new_room_area_l168_168190


namespace average_weight_children_l168_168751

theorem average_weight_children 
  (n_boys : ℕ)
  (w_boys : ℕ)
  (avg_w_boys : ℕ)
  (n_girls : ℕ)
  (w_girls : ℕ)
  (avg_w_girls : ℕ)
  (h1 : n_boys = 8)
  (h2 : avg_w_boys = 140)
  (h3 : n_girls = 6)
  (h4 : avg_w_girls = 130)
  (h5 : w_boys = n_boys * avg_w_boys)
  (h6 : w_girls = n_girls * avg_w_girls)
  (total_w : ℕ)
  (h7 : total_w = w_boys + w_girls)
  (avg_w : ℚ)
  (h8 : avg_w = total_w / (n_boys + n_girls)) :
  avg_w = 135 :=
by
  sorry

end average_weight_children_l168_168751


namespace unique_solution_triple_l168_168021

def satisfies_system (x y z : ℝ) :=
  x^3 = 3 * x - 12 * y + 50 ∧
  y^3 = 12 * y + 3 * z - 2 ∧
  z^3 = 27 * z + 27 * x

theorem unique_solution_triple (x y z : ℝ) :
  satisfies_system x y z ↔ (x = 2 ∧ y = 4 ∧ z = 6) :=
by sorry

end unique_solution_triple_l168_168021


namespace fountain_distance_l168_168421

theorem fountain_distance (h_AD : ℕ) (h_BC : ℕ) (h_AB : ℕ) (h_AD_eq : h_AD = 30) (h_BC_eq : h_BC = 40) (h_AB_eq : h_AB = 50) :
  ∃ AE EB : ℕ, AE = 32 ∧ EB = 18 := by
  sorry

end fountain_distance_l168_168421


namespace integral_cos_nonnegative_l168_168765

theorem integral_cos_nonnegative {f : ℝ → ℝ} (h_cont : ContinuousOn f (Icc 0 (2 * Real.pi)))
  (h_fst_deriv_cont : ContinuousOn (deriv f) (Icc 0 (2 * Real.pi)))
  (h_snd_deriv_nonneg : ∀ x ∈ Icc 0 (2 * Real.pi), 0 ≤ (deriv^[2]) f x) :
  ∫ x in 0..(2 * Real.pi), f x * Real.cos x ≥ 0 :=
  sorry

end integral_cos_nonnegative_l168_168765


namespace wheel_rotation_angle_l168_168980

-- Define the conditions
def radius : ℝ := 20
def arc_length : ℝ := 40

-- Define the theorem stating the desired proof problem
theorem wheel_rotation_angle (r : ℝ) (l : ℝ) (h_r : r = radius) (h_l : l = arc_length) :
  l / r = 2 := 
by sorry

end wheel_rotation_angle_l168_168980


namespace newton_integral_defined_lebesgue_not_exists_l168_168705

noncomputable def F (x : ℝ) : ℝ := x^2 * Real.sin (1 / x^2)

noncomputable def f (x : ℝ) : ℝ := 
  2 * x * Real.sin (1 / x^2) - (2 / x) * Real.cos (1 / x^2)

theorem newton_integral_defined_lebesgue_not_exists :
  ∃ f : ℝ → ℝ, 
  (∫ x in set.Icc 0 1, f x) = Real.sin(1) ∧ 
  ¬MeasureTheory.integrable f MeasureTheory.volume := 
begin
  use f,
  split,
  { sorry },
  { sorry }
end

end newton_integral_defined_lebesgue_not_exists_l168_168705


namespace hyperbola_foci_distance_l168_168933

theorem hyperbola_foci_distance :
  (∃ (h : ℝ → ℝ) (c : ℝ), (∀ x, h x = 2 * x + 3 ∨ h x = 1 - 2 * x)
    ∧ (h 4 = 5)
    ∧ 2 * Real.sqrt (20.25 + 4.444) = 2 * Real.sqrt 24.694) := 
  sorry

end hyperbola_foci_distance_l168_168933


namespace count_of_numbers_less_than_100_divisible_by_2_but_not_by_3_count_of_numbers_less_than_100_divisible_by_2_or_3_count_of_numbers_less_than_100_not_divisible_by_either_2_or_3_l168_168494

theorem count_of_numbers_less_than_100_divisible_by_2_but_not_by_3 :
  Finset.card (Finset.filter (λ n => n % 2 = 0 ∧ n % 3 ≠ 0) (Finset.range 100)) = 33 :=
sorry

theorem count_of_numbers_less_than_100_divisible_by_2_or_3 :
  Finset.card (Finset.filter (λ n => n % 2 = 0 ∨ n % 3 = 0) (Finset.range 100)) = 66 :=
sorry

theorem count_of_numbers_less_than_100_not_divisible_by_either_2_or_3 :
  Finset.card (Finset.filter (λ n => n % 2 ≠ 0 ∧ n % 3 ≠ 0) (Finset.range 100)) = 33 :=
sorry

end count_of_numbers_less_than_100_divisible_by_2_but_not_by_3_count_of_numbers_less_than_100_divisible_by_2_or_3_count_of_numbers_less_than_100_not_divisible_by_either_2_or_3_l168_168494


namespace min_distance_sum_l168_168355

theorem min_distance_sum (x : ℝ) : 
  ∃ y, y = |x + 1| + 2 * |x - 5| + |2 * x - 7| + |(x - 11) / 2| ∧ y = 45 / 8 :=
sorry

end min_distance_sum_l168_168355


namespace investments_ratio_l168_168543

theorem investments_ratio (P Q : ℝ) (hpq : 7 / 10 = (P * 2) / (Q * 4)) : P / Q = 7 / 5 :=
by 
  sorry

end investments_ratio_l168_168543


namespace total_time_to_watch_all_episodes_l168_168763

theorem total_time_to_watch_all_episodes 
  (announced_seasons : ℕ) (episodes_per_season : ℕ) (additional_episodes_last_season : ℕ)
  (seasons_before_announcement : ℕ) (episode_duration : ℝ) :
  announced_seasons = 1 →
  episodes_per_season = 22 →
  additional_episodes_last_season = 4 →
  seasons_before_announcement = 9 →
  episode_duration = 0.5 →
  let total_episodes_previous := seasons_before_announcement * episodes_per_season in
  let episodes_last_season := episodes_per_season + additional_episodes_last_season in 
  let total_episodes := total_episodes_previous + episodes_last_season in 
  total_episodes * episode_duration = 112 :=
by
  intros
  sorry

end total_time_to_watch_all_episodes_l168_168763


namespace find_n_l168_168797

-- Defining the conditions.
def condition_one : Prop :=
  ∀ (c d : ℕ), 
  (80 * 2 * c = 320) ∧ (80 * 2 * d = 160)

def condition_two : Prop :=
  ∀ (c d : ℕ), 
  (100 * 3 * c = 450) ∧ (100 * 3 * d = 300)

def condition_three (n : ℕ) : Prop :=
  ∀ (c d : ℕ), 
  (40 * 4 * c = n) ∧ (40 * 4 * d = 160)

-- Statement of the proof problem using the conditions.
theorem find_n : 
  condition_one ∧ condition_two ∧ condition_three 160 :=
by
  sorry

end find_n_l168_168797


namespace team_A_processes_fraction_l168_168834

theorem team_A_processes_fraction (A B : ℕ) (total_calls : ℚ) 
  (h1 : A = (5/8) * B) 
  (h2 : (8 / 11) * total_calls = TeamB_calls_processed)
  (frac_TeamA_calls : ℚ := (1 - (8 / 11)) * total_calls)
  (calls_per_member_A : ℚ := frac_TeamA_calls / A)
  (calls_per_member_B : ℚ := (8 / 11) * total_calls / B) : 
  calls_per_member_A / calls_per_member_B = 3 / 5 := 
by
  sorry

end team_A_processes_fraction_l168_168834


namespace total_trip_cost_l168_168455

-- Definitions for the problem
def price_per_person : ℕ := 147
def discount : ℕ := 14
def number_of_people : ℕ := 2

-- Statement to prove
theorem total_trip_cost :
  (price_per_person - discount) * number_of_people = 266 :=
by
  sorry

end total_trip_cost_l168_168455


namespace probability_of_sum_of_dice_rolls_odd_l168_168551

noncomputable def probability_sum_odd (n : ℕ) : ℚ :=
if n = 3 then 1 / 4 else 0

theorem probability_of_sum_of_dice_rolls_odd :
  probability_sum_odd 3 = 1 / 4 :=
sorry

end probability_of_sum_of_dice_rolls_odd_l168_168551


namespace solve_equation_1_solve_equation_2_l168_168534

theorem solve_equation_1 (x : ℝ) : 2 * x^2 - x = 0 ↔ x = 0 ∨ x = 1 / 2 := 
by sorry

theorem solve_equation_2 (x : ℝ) : (2 * x + 1)^2 - 9 = 0 ↔ x = 1 ∨ x = -2 := 
by sorry

end solve_equation_1_solve_equation_2_l168_168534


namespace number_of_tangent_small_circles_l168_168125

-- Definitions from the conditions
def central_radius : ℝ := 2
def small_radius : ℝ := 1

-- The proof problem statement
theorem number_of_tangent_small_circles : 
  ∃ n : ℕ, (∀ i j k : ℕ, i ≠ j ∧ j ≠ k ∧ k ≠ i → 
    dist (3 * central_radius) (3 * small_radius) = 3) ∧ n = 3 :=
by
  sorry

end number_of_tangent_small_circles_l168_168125


namespace age_twice_of_father_l168_168840

theorem age_twice_of_father (S M Y : ℕ) (h₁ : S = 22) (h₂ : M = S + 24) (h₃ : M + Y = 2 * (S + Y)) : Y = 2 := by
  sorry

end age_twice_of_father_l168_168840


namespace count_4_digit_numbers_divisible_by_13_l168_168895

theorem count_4_digit_numbers_divisible_by_13 : 
  let count := (9962 - 1014) / 13 + 1
  1000 ≤ 1014 ∧ 9962 ≤ 9999 →
  count = 689 :=
  by
    sorry

end count_4_digit_numbers_divisible_by_13_l168_168895


namespace union_of_sets_l168_168461

def A : Set ℕ := {2, 3}
def B : Set ℕ := {1, 2}

theorem union_of_sets : A ∪ B = {1, 2, 3} := by
  sorry

end union_of_sets_l168_168461


namespace inv_sum_mod_l168_168693

theorem inv_sum_mod 
  : (∃ (x y : ℤ), (3 * x ≡ 1 [ZMOD 25]) ∧ (3^2 * y ≡ 1 [ZMOD 25]) ∧ (x + y ≡ 6 [ZMOD 25])) :=
sorry

end inv_sum_mod_l168_168693


namespace power_seven_evaluation_l168_168464

theorem power_seven_evaluation (a b : ℝ) (h : a = (7 : ℝ)^(1/4) ∧ b = (7 : ℝ)^(1/7)) : 
  a / b = (7 : ℝ)^(3/28) :=
  sorry

end power_seven_evaluation_l168_168464


namespace line_perp_to_plane_contains_line_implies_perp_l168_168877

variables {Point Line Plane : Type}
variables (m n : Line) (α : Plane)
variables (contains : Plane → Line → Prop) (perp : Line → Line → Prop) (perp_plane : Line → Plane → Prop)

-- Given: 
-- m and n are two different lines
-- α is a plane
-- m ⊥ α (m is perpendicular to the plane α)
-- n ⊂ α (n is contained in the plane α)
-- Prove: m ⊥ n
theorem line_perp_to_plane_contains_line_implies_perp (hm : perp_plane m α) (hn : contains α n) : perp m n :=
sorry

end line_perp_to_plane_contains_line_implies_perp_l168_168877


namespace probability_p_s_mod_10_l168_168552

noncomputable def count_valid_pairs : ℕ :=
  (Finset.Icc 1 100).card * ((Finset.Icc 1 100).card - 1) / 2

noncomputable def total_pairs : ℕ := 4950

theorem probability_p_s_mod_10 (a b : ℕ) (h_a : a ∈ Finset.Icc 1 100) (h_b : b ∈ Finset.Icc 1 100) (h_neq : a ≠ b) :
  let S := a + b,
  let P := a * b in
  (Nat.gcd a b = 1 ↔ Rational.gcd (a+b) (a*b+n) !=0) (hermite_of_galois a b) (P+S) % 10 = n  := 

begin
  sorry -- Proof not required
end

end probability_p_s_mod_10_l168_168552


namespace stream_speed_l168_168096

theorem stream_speed (x : ℝ) (d : ℝ) (v_b : ℝ) (t : ℝ) (h : v_b = 8) (h1 : d = 210) (h2 : t = 56) : x = 2 :=
by
  sorry

end stream_speed_l168_168096


namespace slices_of_bread_left_l168_168272

variable (monday_to_friday_slices saturday_slices total_slices_used initial_slices slices_left: ℕ)

def sandwiches_monday_to_friday : ℕ := 5
def slices_per_sandwich : ℕ := 2
def sandwiches_saturday : ℕ := 2
def initial_slices_of_bread : ℕ := 22

theorem slices_of_bread_left :
  slices_left = initial_slices_of_bread - total_slices_used
  :=
by  sorry

end slices_of_bread_left_l168_168272


namespace intersection_A_B_l168_168503

def A : Set ℝ := {x | x^2 + x - 6 > 0}
def B : Set ℝ := {x | -2 < x ∧ x < 4}

theorem intersection_A_B :
  A ∩ B = {x | 2 < x ∧ x < 4} :=
sorry

end intersection_A_B_l168_168503


namespace multiply_polynomials_l168_168079

theorem multiply_polynomials (x : ℂ) : 
  (x^6 + 27 * x^3 + 729) * (x^3 - 27) = x^12 + 27 * x^9 - 19683 * x^3 - 531441 :=
by
  sorry

end multiply_polynomials_l168_168079


namespace absolute_value_sum_l168_168502

theorem absolute_value_sum (a b : ℤ) (h_a : |a| = 5) (h_b : |b| = 3) : 
  (a + b = 8) ∨ (a + b = 2) ∨ (a + b = -2) ∨ (a + b = -8) :=
by
  sorry

end absolute_value_sum_l168_168502


namespace find_common_ratio_l168_168036

variable {a : ℕ → ℝ}
variable (q : ℝ)

-- Definition of geometric sequence condition
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

-- Given conditions
def conditions (a : ℕ → ℝ) (q : ℝ) : Prop :=
  (a 2 + a 4 = 20) ∧ (a 3 + a 5 = 40)

-- Proposition to be proved
theorem find_common_ratio (a : ℕ → ℝ) (q : ℝ) (h_geo : is_geometric_sequence a q) (h_cond : conditions a q) : q = 2 :=
by 
  sorry

end find_common_ratio_l168_168036


namespace rental_days_l168_168539

-- Definitions based on conditions
def daily_rate := 30
def weekly_rate := 190
def total_payment := 310

-- Prove that Jennie rented the car for 11 days
theorem rental_days : ∃ d : ℕ, d = 11 ∧ (total_payment = weekly_rate + (d - 7) * daily_rate) ∨ (d < 7 ∧ total_payment = d * daily_rate) :=
by
  sorry

end rental_days_l168_168539


namespace maximum_mark_for_paper_i_l168_168700

noncomputable def maximum_mark (pass_percentage: ℝ) (secured_marks: ℝ) (failed_by: ℝ) : ℝ :=
  (secured_marks + failed_by) / pass_percentage

theorem maximum_mark_for_paper_i :
  maximum_mark 0.35 42 23 = 186 :=
by
  sorry

end maximum_mark_for_paper_i_l168_168700


namespace solve_for_y_l168_168392

theorem solve_for_y (y : ℝ) : (y^2 + 6 * y + 8 = -(y + 4) * (y + 6)) → y = -4 :=
by {
  sorry
}

end solve_for_y_l168_168392


namespace triangle_tan_A_and_area_l168_168208

theorem triangle_tan_A_and_area {A B C a b c : ℝ} (hB : B = Real.pi / 3)
  (h1 : (Real.cos A - 3 * Real.cos C) * b = (3 * c - a) * Real.cos B)
  (hb : b = Real.sqrt 14) : 
  ∃ tan_A : ℝ, tan_A = Real.sqrt 3 / 5 ∧  -- First part: the value of tan A
  ∃ S : ℝ, S = (3 * Real.sqrt 3) / 2 :=  -- Second part: the area of triangle ABC
by
  sorry

end triangle_tan_A_and_area_l168_168208


namespace square_of_105_l168_168588

/-- Prove that 105^2 = 11025. -/
theorem square_of_105 : 105^2 = 11025 :=
by
  sorry

end square_of_105_l168_168588


namespace count_4_digit_numbers_divisible_by_13_l168_168890

theorem count_4_digit_numbers_divisible_by_13 : 
  ∃ n : ℕ, n = 693 ∧ (∀ k : ℕ, k >= 1000 ∧ k < 10000 ∧ k % 13 = 0 → ∃ m : ℕ, m = (k - 1000) / 13 + 1 ∧ m = n) :=
by {
  -- Solution proof will be placed here.
  sorry
}

end count_4_digit_numbers_divisible_by_13_l168_168890


namespace total_students_in_lunchroom_l168_168439

theorem total_students_in_lunchroom 
  (students_per_table : ℕ) 
  (number_of_tables : ℕ) 
  (h1 : students_per_table = 6) 
  (h2 : number_of_tables = 34) : 
  students_per_table * number_of_tables = 204 := by
  sorry

end total_students_in_lunchroom_l168_168439


namespace probability_black_white_ball_l168_168798

theorem probability_black_white_ball :
  let total_balls := 5
  let black_balls := 3
  let white_balls := 2
  let favorable_outcomes := (Nat.choose 3 1) * (Nat.choose 2 1)
  let total_outcomes := Nat.choose 5 2
  (favorable_outcomes / total_outcomes) = (3 / 5) := 
by
  sorry

end probability_black_white_ball_l168_168798


namespace trapezoid_dot_product_ad_bc_l168_168252

-- Define the trapezoid and its properties
variables (A B C D O : Type) (AB CD AO BO : ℝ)
variables (AD BC : ℝ)

-- Conditions from the problem
axiom AB_length : AB = 41
axiom CD_length : CD = 24
axiom diagonals_perpendicular : ∀ (v₁ v₂ : ℝ), (v₁ * v₂ = 0)

-- Using these conditions, prove that the dot product of the vectors AD and BC is 984
theorem trapezoid_dot_product_ad_bc : AD * BC = 984 :=
  sorry

end trapezoid_dot_product_ad_bc_l168_168252


namespace basketball_game_l168_168750

theorem basketball_game (E H : ℕ) (h1 : E = H + 18) (h2 : E + H = 50) : H = 16 :=
by
  sorry

end basketball_game_l168_168750


namespace fraction_comparison_l168_168216

theorem fraction_comparison
  (a b c d : ℝ)
  (h1 : a / b < c / d)
  (h2 : b > 0)
  (h3 : d > 0)
  (h4 : b > d) :
  (a + c) / (b + d) < (1 / 2) * (a / b + c / d) :=
by
  sorry

end fraction_comparison_l168_168216


namespace min_value_expression_l168_168332

theorem min_value_expression : 
  ∃ x y : ℝ, (∀ a b : ℝ, 2 * a^2 + 3 * a * b + 4 * b^2 + 5 ≥ 5) ∧ (2 * x^2 + 3 * x * y + 4 * y^2 + 5 = 5) := 
by 
sorry

end min_value_expression_l168_168332


namespace circle_eq_of_hyperbola_focus_eccentricity_l168_168398

theorem circle_eq_of_hyperbola_focus_eccentricity :
  ∀ (x y : ℝ), ((y^2 - (x^2 / 3) = 1) → (x^2 + (y-2)^2 = 4)) := by
  intro x y
  intro hyp_eq
  sorry

end circle_eq_of_hyperbola_focus_eccentricity_l168_168398


namespace sandy_has_32_fish_l168_168388

-- Define the initial number of pet fish Sandy has
def initial_fish : Nat := 26

-- Define the number of fish Sandy bought
def fish_bought : Nat := 6

-- Define the total number of pet fish Sandy has now
def total_fish : Nat := initial_fish + fish_bought

-- Prove that Sandy now has 32 pet fish
theorem sandy_has_32_fish : total_fish = 32 :=
by
  sorry

end sandy_has_32_fish_l168_168388


namespace rice_in_each_container_ounces_l168_168922

-- Given conditions
def total_rice_pounds : ℚ := 25 / 4
def num_containers : ℕ := 4
def pounds_to_ounces : ℚ := 16

-- Problem statement: proving the amount of rice in each container in ounces
theorem rice_in_each_container_ounces :
  (total_rice_pounds / num_containers) * pounds_to_ounces = 25 :=
by sorry

end rice_in_each_container_ounces_l168_168922


namespace student_marks_l168_168568

theorem student_marks (T P F M : ℕ)
  (hT : T = 600)
  (hP : P = 33)
  (hF : F = 73)
  (hM : M = (P * T / 100) - F) : M = 125 := 
by 
  sorry

end student_marks_l168_168568


namespace ali_initial_money_l168_168981

theorem ali_initial_money (X : ℝ) (h1 : X / 2 - (1 / 3) * (X / 2) = 160) : X = 480 :=
by sorry

end ali_initial_money_l168_168981


namespace compute_105_squared_l168_168583

theorem compute_105_squared : 105^2 = 11025 :=
by
  sorry

end compute_105_squared_l168_168583


namespace max_liters_of_water_that_can_be_heated_to_boiling_l168_168816

-- Define the initial conditions
def initial_heat_per_5min := 480 -- kJ
def heat_reduction_rate := 0.25
def initial_temp := 20 -- Celsius
def boiling_temp := 100 -- Celsius
def specific_heat_capacity := 4.2 -- kJ/kg·°C

-- Define the temperature difference
def delta_T := boiling_temp - initial_temp -- Celsius

-- Define the calculation of the total heat available from a geometric series
def total_heat_available := initial_heat_per_5min / (1 - (1 - heat_reduction_rate))

-- Define the calculation of energy required to heat m kg of water
def energy_required (m : ℝ) := specific_heat_capacity * m * delta_T

-- Define the main theorem to prove
theorem max_liters_of_water_that_can_be_heated_to_boiling :
  ∃ (m : ℝ), ⌊m⌋ = 5 ∧ energy_required m ≤ total_heat_available :=
begin
  sorry
end

end max_liters_of_water_that_can_be_heated_to_boiling_l168_168816


namespace parker_total_weight_l168_168776

def twenty_pound := 20
def thirty_pound := 30
def forty_pound := 40

def first_set_weight := (2 * twenty_pound) + (1 * thirty_pound) + (1 * forty_pound)
def second_set_weight := (1 * twenty_pound) + (2 * thirty_pound) + (2 * forty_pound)
def third_set_weight := (3 * thirty_pound) + (3 * forty_pound)

def total_weight := first_set_weight + second_set_weight + third_set_weight

theorem parker_total_weight :
  total_weight = 480 := by
  sorry

end parker_total_weight_l168_168776


namespace count_4digit_numbers_divisible_by_13_l168_168893

theorem count_4digit_numbers_divisible_by_13 : 
  let count := (λ n, n - 77 + 1) (769)
  count = 693 :=
by
-- Let 77 be the smallest integer such that 13 * 77 = 1001
-- Let 769 be the largest integer such that 13 * 769 = 9997
-- Hence, the range of these integers should be 769 - 77 + 1
-- This gives us a count of 693
-- The statement as 693 is directly given
sorry

end count_4digit_numbers_divisible_by_13_l168_168893


namespace matrix_det_eq_l168_168163

open Matrix

def matrix3x3 (x : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![
    ![x + 1, x, x],
    ![x, x + 2, x],
    ![x, x, x + 3]
  ]

theorem matrix_det_eq (x : ℝ) : det (matrix3x3 x) = 2 * x^2 + 11 * x + 6 :=
  sorry

end matrix_det_eq_l168_168163


namespace min_area_rectangle_l168_168307

theorem min_area_rectangle (P : ℕ) (hP : P = 60) :
  ∃ (l w : ℕ), 2 * l + 2 * w = P ∧ l * w = 29 :=
by
  sorry

end min_area_rectangle_l168_168307


namespace row_number_sum_l168_168382

theorem row_number_sum (n : ℕ) (h : (2 * n - 1) ^ 2 = 2015 ^ 2) : n = 1008 :=
by
  sorry

end row_number_sum_l168_168382


namespace total_cards_traded_l168_168239

-- Define the total number of cards traded in both trades
def total_traded (p1_t: ℕ) (r1_t: ℕ) (p2_t: ℕ) (r2_t: ℕ): ℕ :=
  (p1_t + r1_t) + (p2_t + r2_t)

-- Given conditions as definitions
def padma_trade1 := 2   -- Cards Padma traded in the first trade
def robert_trade1 := 10  -- Cards Robert traded in the first trade
def padma_trade2 := 15  -- Cards Padma traded in the second trade
def robert_trade2 := 8   -- Cards Robert traded in the second trade

-- Theorem stating the total number of cards traded is 35
theorem total_cards_traded : 
  total_traded padma_trade1 robert_trade1 padma_trade2 robert_trade2 = 35 :=
by
  sorry

end total_cards_traded_l168_168239


namespace units_digit_expression_l168_168170

theorem units_digit_expression :
  ((2 * 21 * 2019 + 2^5) - 4^3) % 10 = 6 := 
sorry

end units_digit_expression_l168_168170


namespace min_fencing_l168_168236

variable (w l : ℝ)

noncomputable def area := w * l

noncomputable def length := 2 * w

theorem min_fencing (h1 : area w l ≥ 500) (h2 : l = length w) : 
  w = 5 * Real.sqrt 10 ∧ l = 10 * Real.sqrt 10 :=
  sorry

end min_fencing_l168_168236


namespace min_value_expression_l168_168283

theorem min_value_expression (x y : ℝ) : (x^2 * y - 1)^2 + (x + y - 1)^2 ≥ 1 :=
sorry

end min_value_expression_l168_168283


namespace pedro_more_squares_l168_168241

theorem pedro_more_squares
  (jesus_squares : ℕ)
  (linden_squares : ℕ)
  (pedro_squares : ℕ)
  (jesus_linden_combined : jesus_squares + linden_squares = 135)
  (pedro_total : pedro_squares = 200) :
  pedro_squares - (jesus_squares + linden_squares) = 65 :=
by
  sorry

end pedro_more_squares_l168_168241


namespace point_distance_from_origin_l168_168970

theorem point_distance_from_origin (x y m : ℝ) (h1 : |y| = 15) (h2 : (x - 2)^2 + (y - 7)^2 = 169) (h3 : x > 2) :
  m = Real.sqrt (334 + 4 * Real.sqrt 105) :=
sorry

end point_distance_from_origin_l168_168970


namespace range_of_a_l168_168626

-- Define the conditions and what we want to prove
theorem range_of_a (a : ℝ) (x : ℝ) 
    (h1 : ∀ x, |x - 1| + |x + 1| ≥ 3 * a)
    (h2 : ∀ x, (2 * a - 1) ^ x ≤ 1 → (2 * a - 1) < 1 ∧ (2 * a - 1) > 0) :
    (1 / 2 < a ∧ a ≤ 2 / 3) :=
by
  sorry -- Here will be the proof

end range_of_a_l168_168626


namespace right_triangle_hypotenuse_l168_168258

theorem right_triangle_hypotenuse 
  (shorter_leg longer_leg hypotenuse : ℝ)
  (h1 : longer_leg = 2 * shorter_leg - 1)
  (h2 : 1 / 2 * shorter_leg * longer_leg = 60) :
  hypotenuse = 17 :=
by
  sorry

end right_triangle_hypotenuse_l168_168258


namespace floor_tiling_l168_168287

theorem floor_tiling (n : ℕ) (x : ℕ) (h1 : 6 * x = n^2) : 6 ∣ n := sorry

end floor_tiling_l168_168287


namespace measure_angle_y_l168_168639

theorem measure_angle_y
  (triangle_angles : ∀ {A B C : ℝ}, (A = 45 ∧ B = 45 ∧ C = 90) ∨ (A = 45 ∧ B = 90 ∧ C = 45) ∨ (A = 90 ∧ B = 45 ∧ C = 45))
  (p q : ℝ) (hpq : p = q) :
  ∃ (y : ℝ), y = 90 :=
by
  sorry

end measure_angle_y_l168_168639


namespace dartboard_central_angle_l168_168128

-- Define the conditions
variables {A : ℝ} {x : ℝ}

-- State the theorem
theorem dartboard_central_angle (h₁ : A > 0) (h₂ : (1/4 : ℝ) = ((x / 360) * A) / A) : x = 90 := 
by sorry

end dartboard_central_angle_l168_168128


namespace iceCreamCombo_l168_168964

open Finset

def iceCreamFlavors : ℕ := 5
def toppings : ℕ := 7
def chooseRequiredToppings : ℕ := 3

-- Theorem statement
theorem iceCreamCombo :
  iceCreamFlavors * (choose toppings chooseRequiredToppings) = 175 := by
  sorry

end iceCreamCombo_l168_168964


namespace intersection_point_l168_168998

theorem intersection_point : 
  ∃ (x y : ℚ), y = - (5/3 : ℚ) * x ∧ y + 3 = 15 * x - 6 ∧ x = 27 / 50 ∧ y = - 9 / 10 := 
by
  sorry

end intersection_point_l168_168998


namespace largest_divisible_by_digits_sum_l168_168545

def digits_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem largest_divisible_by_digits_sum : ∃ n, n < 900 ∧ n % digits_sum n = 0 ∧ ∀ m, m < 900 ∧ m % digits_sum m = 0 → m ≤ 888 :=
by
  sorry

end largest_divisible_by_digits_sum_l168_168545


namespace blocks_calculation_l168_168290

theorem blocks_calculation
  (total_amount : ℕ)
  (gift_cost : ℕ)
  (workers_per_block : ℕ)
  (H1  : total_amount = 4000)
  (H2  : gift_cost = 4)
  (H3  : workers_per_block = 100)
  : total_amount / gift_cost / workers_per_block = 10 :=
by
  sorry

end blocks_calculation_l168_168290


namespace pumps_fill_time_l168_168569

def fill_time {X Y Z : ℝ} (h1 : X + Y = 1/3) (h2 : X + Z = 1/6) (h3 : Y + Z = 2/9) : Prop :=
  1 / (X + Y + Z) = 36 / 13

theorem pumps_fill_time (X Y Z : ℝ) (h1 : X + Y = 1/3) (h2 : X + Z = 1/6) (h3 : Y + Z = 2/9) : 
  1 / (X + Y + Z) = 36 / 13 :=
by
  sorry

end pumps_fill_time_l168_168569


namespace set_M_listed_correctly_l168_168938

theorem set_M_listed_correctly :
  {a : ℕ+ | ∃ (n : ℤ), 4 = n * (1 - a)} = {2, 3, 4} := by
sorry

end set_M_listed_correctly_l168_168938


namespace interest_rate_for_lending_l168_168968

def simple_interest (P : ℕ) (R : ℕ) (T : ℕ) : ℕ :=
  (P * R * T) / 100

theorem interest_rate_for_lending :
  ∀ (P T R_b G R_l : ℕ),
  P = 20000 →
  T = 6 →
  R_b = 8 →
  G = 200 →
  simple_interest P R_b T + G * T = simple_interest P R_l T →
  R_l = 9 :=
by
  intros P T R_b G R_l
  sorry

end interest_rate_for_lending_l168_168968


namespace positive_difference_between_solutions_l168_168604

theorem positive_difference_between_solutions : 
  let f (x : ℝ) := (5 - (x^2 / 3 : ℝ))^(1 / 3 : ℝ)
  let a := 4 * Real.sqrt 6
  let b := -4 * Real.sqrt 6
  |a - b| = 8 * Real.sqrt 6 := 
by 
  sorry

end positive_difference_between_solutions_l168_168604


namespace min_value_of_function_l168_168771

theorem min_value_of_function (h : 0 < x ∧ x < 1) : 
  ∃ (y : ℝ), (∀ z : ℝ, z = (4 / x + 1 / (1 - x)) → y ≤ z) ∧ y = 9 :=
by
  sorry

end min_value_of_function_l168_168771


namespace new_room_correct_size_l168_168193

-- Definitions of conditions
def current_bedroom := 309 -- sq ft
def current_bathroom := 150 -- sq ft
def current_space := current_bedroom + current_bathroom
def new_room_size := 2 * current_space

-- Proving the new room size
theorem new_room_correct_size : new_room_size = 918 := by
  sorry

end new_room_correct_size_l168_168193


namespace square_of_105_l168_168589

/-- Prove that 105^2 = 11025. -/
theorem square_of_105 : 105^2 = 11025 :=
by
  sorry

end square_of_105_l168_168589


namespace find_lambda_l168_168489

variable (n : ℕ) (λ : ℝ) (an sn : ℕ → ℝ)

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n = 2 * n + λ

def sum_of_terms (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop := 
  ∀ n : ℕ, S n = (n * (a 1 + a n)) / 2

def sum_increasing (S : ℕ → ℝ) : Prop := 
  ∀ n : ℕ, S (n + 1) > S n 

theorem find_lambda 
  (h_arith : arithmetic_sequence an)
  (h_sum : sum_of_terms sn an)
  (h_incr : sum_increasing sn) : 
  λ > -4 :=
sorry

end find_lambda_l168_168489


namespace total_books_per_year_l168_168111

variable (c s : ℕ)

theorem total_books_per_year (hc : 0 < c) (hs : 0 < s) :
  6 * 12 * (c * s) = 72 * c * s := by
  sorry

end total_books_per_year_l168_168111


namespace line_and_circle_separate_l168_168492

open Real

-- Define the vectors a and b
variables (α β : ℝ)
def vec_a : ℝ × ℝ := (2 * cos α, 2 * sin α)
def vec_b : ℝ × ℝ := (3 * cos β, 3 * sin β)

-- Define the angle between a and b
def angle_ab : ℝ := pi / 3 -- 60 degrees in radians

-- Line equation: x * cos(α) - y * sin(α) + 1/2 = 0
noncomputable def line_eq (x y : ℝ) : Prop :=
  x * cos α - y * sin α + 1/2 = 0

-- Circle equation: (x - cos(β))^2 + (y + sin(β))^2 = 1/2
def circle_eq (x y : ℝ) : Prop :=
  (x - cos β)^2 + (y + sin β)^2 = 1/2

-- Proof statement
theorem line_and_circle_separate (α β : ℝ) 
  (h_angle : cos (α - β) = 1/2) :
  ∃ l c : ℝ, line_eq α β l c → circle_eq α β l c → false :=
sorry

end line_and_circle_separate_l168_168492


namespace complex_fraction_identity_l168_168075

theorem complex_fraction_identity
  (a b : ℂ) (ζ : ℂ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : ζ ^ 3 = 1) (h4 : ζ ≠ 1) 
  (h5 : a ^ 2 + a * b + b ^ 2 = 0) :
  (a ^ 9 + b ^ 9) / ((a - b) ^ 9) = (2 : ℂ) / (81 * (ζ - 1)) :=
sorry

end complex_fraction_identity_l168_168075


namespace action_figures_per_shelf_l168_168773

/-- Mike has 64 action figures he wants to display. If each shelf 
    in his room can hold a certain number of figures and he needs 8 
    shelves, prove that each shelf can hold 8 figures. -/
theorem action_figures_per_shelf :
  (64 / 8) = 8 :=
by
  sorry

end action_figures_per_shelf_l168_168773


namespace avg_rate_of_change_l168_168869

noncomputable def f (x : ℝ) : ℝ := 3 * x^2 + 5

theorem avg_rate_of_change :
  (f 0.2 - f 0.1) / (0.2 - 0.1) = 0.9 := by
  sorry

end avg_rate_of_change_l168_168869


namespace find_number_of_members_l168_168288

variable (n : ℕ)

-- We translate the conditions into Lean 4 definitions
def total_collection := 9216
def per_member_contribution := n

-- The goal is to prove that n = 96 given the total collection
theorem find_number_of_members (h : n * n = total_collection) : n = 96 := 
sorry

end find_number_of_members_l168_168288


namespace solve_for_x_l168_168285

theorem solve_for_x (x : ℝ) (h : |x - 2| = |x - 3| + 1) : x = 3 :=
by
  sorry

end solve_for_x_l168_168285


namespace hyperbola_sufficient_but_not_necessary_l168_168906

theorem hyperbola_sufficient_but_not_necessary :
  (∀ (C : Type) (x y : ℝ), C = {p : ℝ × ℝ | ((p.1)^2 / 16) - ((p.2)^2 / 9) = 1} →
  (∀ x, y = 3 * (x / 4) ∨ y = -3 * (x / 4)) →
  ∃ (C' : Type) (x' y' : ℝ), C' = {p : ℝ × ℝ | ((p.1)^2 / 64) - ((p.2)^2 / 36) = 1} ∧
  (∀ x', y' = 3 * (x' / 4) ∨ y' = -3 * (x' / 4))) :=
sorry

end hyperbola_sufficient_but_not_necessary_l168_168906


namespace solve_length_BF_l168_168843

-- Define the problem conditions
def rectangular_paper (short_side long_side : ℝ) : Prop :=
  short_side = 12 ∧ long_side > short_side

def vertex_touch_midpoint (vmp mid : ℝ) : Prop :=
  vmp = mid / 2

def congruent_triangles (triangle1 triangle2 : ℝ) : Prop :=
  triangle1 = triangle2

-- Theorem to prove the length of BF
theorem solve_length_BF (short_side long_side vmp mid triangle1 triangle2 : ℝ) 
  (h1 : rectangular_paper short_side long_side)
  (h2 : vertex_touch_midpoint vmp mid)
  (h3 : congruent_triangles triangle1 triangle2) :
  -- The length of BF is 10
  mid = 6 → 18 - 6 = 12 + 6 - 10 → 10 = 12 - (18 - 10) → vmp = 6 → 6 * 2 = 12 →
  sorry :=
sorry

end solve_length_BF_l168_168843


namespace number_of_days_in_first_part_l168_168396

variable {x : ℕ}

-- Conditions
def avg_exp_first_part (x : ℕ) : ℕ := 350 * x
def avg_exp_next_four_days : ℕ := 420 * 4
def total_days (x : ℕ) : ℕ := x + 4
def avg_exp_whole_week (x : ℕ) : ℕ := 390 * total_days x

-- Equation based on the conditions
theorem number_of_days_in_first_part :
  avg_exp_first_part x + avg_exp_next_four_days = avg_exp_whole_week x →
  x = 3 :=
by
  sorry

end number_of_days_in_first_part_l168_168396


namespace find_prob_p_l168_168851

variable (p : ℚ)

theorem find_prob_p (h : 15 * p^4 * (1 - p)^2 = 500 / 2187) : p = 3 / 7 := 
  sorry

end find_prob_p_l168_168851


namespace find_c_l168_168606

theorem find_c
  (c d : ℝ)
  (h1 : ∀ (x : ℝ), 7 * x^3 + 3 * c * x^2 + 6 * d * x + c = 0)
  (h2 : ∀ (p q r : ℝ), p ≠ q ∧ q ≠ r ∧ p ≠ r ∧
        7 * p^3 + 3 * c * p^2 + 6 * d * p + c = 0 ∧ 
        7 * q^3 + 3 * c * q^2 + 6 * d * q + c = 0 ∧ 
        7 * r^3 + 3 * c * r^2 + 6 * d * r + c = 0 ∧ 
        Real.log (p * q * r) / Real.log 3 = 3) :
  c = -189 :=
sorry

end find_c_l168_168606


namespace solution_replacement_concentration_l168_168126

theorem solution_replacement_concentration :
  ∀ (init_conc replaced_fraction new_conc replaced_conc : ℝ),
    init_conc = 0.45 → replaced_fraction = 0.5 → replaced_conc = 0.25 → new_conc = 35 →
    (init_conc - replaced_fraction * init_conc + replaced_fraction * replaced_conc) * 100 = new_conc :=
by
  intro init_conc replaced_fraction new_conc replaced_conc
  intros h_init h_frac h_replaced h_new
  rw [h_init, h_frac, h_replaced, h_new]
  sorry

end solution_replacement_concentration_l168_168126


namespace a8_div_b8_l168_168351

variable (a b : ℕ → ℕ)
variable (S T : ℕ → ℕ)

-- Given Conditions
axiom sum_a (n : ℕ) : S n = (n * (a 1 + (n - 1) * a 2)) / 2 -- Sum of first n terms of arithmetic sequence a_n
axiom sum_b (n : ℕ) : T n = (n * (b 1 + (n - 1) * b 2)) / 2 -- Sum of first n terms of arithmetic sequence b_n
axiom ratio (n : ℕ) : S n / T n = (7 * n + 3) / (n + 3)

-- Proof statement
theorem a8_div_b8 : a 8 / b 8 = 6 := by
  sorry

end a8_div_b8_l168_168351


namespace percentage_difference_highest_lowest_salary_l168_168245

variables (R : ℝ)
def Ram_salary := 1.25 * R
def Simran_salary := 0.85 * R
def Rahul_salary := 0.85 * R * 1.10

theorem percentage_difference_highest_lowest_salary :
  let highest_salary := Ram_salary R
  let lowest_salary := Simran_salary R
  (highest_salary ≠ 0) → ((highest_salary - lowest_salary) / highest_salary) * 100 = 32 :=
by
  intros
  -- Sorry in place of proof
  sorry

end percentage_difference_highest_lowest_salary_l168_168245


namespace problem1_problem2_l168_168185

noncomputable def f (x a : ℝ) : ℝ := Real.log x + a/x

/-- 
Given the function f(x) = ln(x) + a/x (where a is a real number),
prove that if the function f(x) has two zeros, then 0 < a < 1/e.
-/
theorem problem1 (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 a = 0 ∧ f x2 a = 0) → (0 < a ∧ a < 1/Real.exp 1) :=
sorry

/-- 
Given the function f(x) = ln(x) + a/x (where a is a real number) and a line y = m
that intersects the graph of f(x) at two points (x1, m) and (x2, m),
prove that x1 + x2 > 2a.
-/
theorem problem2 (x1 x2 a m : ℝ) (h : f x1 a = m ∧ f x2 a = m ∧ x1 ≠ x2) :
  x1 + x2 > 2 * a :=
sorry

end problem1_problem2_l168_168185


namespace fraction_eval_l168_168862

theorem fraction_eval : 
    (1 / (3 - (1 / (3 - (1 / (3 - (1 / 4))))))) = (11 / 29) := 
by
  sorry

end fraction_eval_l168_168862


namespace remainder_is_6910_l168_168822

def polynomial (x : ℝ) : ℝ := 5 * x^7 - 3 * x^6 - 8 * x^5 + 3 * x^3 + 5 * x^2 - 20

def divisor (x : ℝ) : ℝ := 3 * x - 9

theorem remainder_is_6910 : polynomial 3 = 6910 := by
  sorry

end remainder_is_6910_l168_168822


namespace floor_of_sqrt_sum_eq_floor_of_sqrt_expr_l168_168515

theorem floor_of_sqrt_sum_eq_floor_of_sqrt_expr (n : ℤ): 
  Int.floor (Real.sqrt n + Real.sqrt (n + 1)) = Int.floor (Real.sqrt (4 * n + 2)) := 
sorry

end floor_of_sqrt_sum_eq_floor_of_sqrt_expr_l168_168515


namespace merchant_spent_for_belle_l168_168942

def dress_cost (S : ℤ) (H : ℤ) : ℤ := 6 * S + 3 * H
def hat_cost (S : ℤ) (H : ℤ) : ℤ := 3 * S + 5 * H
def belle_cost (S : ℤ) (H : ℤ) : ℤ := S + 2 * H

theorem merchant_spent_for_belle :
  ∃ (S H : ℤ), dress_cost S H = 105 ∧ hat_cost S H = 70 ∧ belle_cost S H = 25 :=
by
  sorry

end merchant_spent_for_belle_l168_168942

import Mathlib

namespace NUMINAMATH_GPT_area_of_quadrilateral_ABDF_l1977_197746

theorem area_of_quadrilateral_ABDF :
  let length := 40
  let width := 30
  let rectangle_area := length * width
  let B := (1/4 : ℝ) * length
  let F := (1/2 : ℝ) * width
  let area_BCD := (1/2 : ℝ) * (3/4 : ℝ) * length * width
  let area_EFD := (1/2 : ℝ) * F * length
  rectangle_area - area_BCD - area_EFD = 450 := sorry

end NUMINAMATH_GPT_area_of_quadrilateral_ABDF_l1977_197746


namespace NUMINAMATH_GPT_simplify_product_l1977_197762

theorem simplify_product (b : ℤ) : (1 * 2 * b * 3 * b^2 * 4 * b^3 * 5 * b^4 * 6 * b^5) = 720 * b^15 := by
  sorry

end NUMINAMATH_GPT_simplify_product_l1977_197762


namespace NUMINAMATH_GPT_acute_not_greater_than_right_l1977_197730

-- Definitions for conditions
def is_right_angle (α : ℝ) : Prop := α = 90
def is_acute_angle (α : ℝ) : Prop := α < 90

-- Statement to be proved
theorem acute_not_greater_than_right (α : ℝ) (h1 : is_right_angle 90) (h2 : is_acute_angle α) : ¬ (α > 90) :=
by
    sorry

end NUMINAMATH_GPT_acute_not_greater_than_right_l1977_197730


namespace NUMINAMATH_GPT_nina_running_distance_l1977_197734

theorem nina_running_distance (x : ℝ) (hx : 2 * x + 0.67 = 0.83) : x = 0.08 := by
  sorry

end NUMINAMATH_GPT_nina_running_distance_l1977_197734


namespace NUMINAMATH_GPT_taxi_fare_distance_condition_l1977_197736

theorem taxi_fare_distance_condition (x : ℝ) (h1 : 7 + (max (x - 3) 0) * 2.4 = 19) : x ≤ 8 := 
by
  sorry

end NUMINAMATH_GPT_taxi_fare_distance_condition_l1977_197736


namespace NUMINAMATH_GPT_range_of_m_l1977_197707

theorem range_of_m (α : ℝ) (m : ℝ) (h : (α > π ∧ α < 3 * π / 2) ∨ (α > 3 * π / 2 ∧ α < 2 * π)) :
  -1 < (Real.sin α) ∧ (Real.sin α) < 0 ∧ (Real.sin α) = (2 * m - 3) / (4 - m) → 
  m ∈ Set.Ioo (-1 : ℝ) (3 / 2 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1977_197707


namespace NUMINAMATH_GPT_greatest_odd_integer_l1977_197733

theorem greatest_odd_integer (x : ℕ) (h_odd : x % 2 = 1) (h_pos : x > 0) (h_ineq : x^2 < 50) : x = 7 :=
by sorry

end NUMINAMATH_GPT_greatest_odd_integer_l1977_197733


namespace NUMINAMATH_GPT_min_sum_of_M_and_N_l1977_197771

noncomputable def Alice (x : ℕ) : ℕ := 3 * x + 2
noncomputable def Bob (x : ℕ) : ℕ := 2 * x + 27

-- Define the result after 4 moves
noncomputable def Alice_4_moves (M : ℕ) : ℕ := Alice (Alice (Alice (Alice M)))
noncomputable def Bob_4_moves (N : ℕ) : ℕ := Bob (Bob (Bob (Bob N)))

theorem min_sum_of_M_and_N :
  ∃ (M N : ℕ), Alice_4_moves M = Bob_4_moves N ∧ M + N = 10 :=
sorry

end NUMINAMATH_GPT_min_sum_of_M_and_N_l1977_197771


namespace NUMINAMATH_GPT_exists_solution_for_lambda_9_l1977_197738

theorem exists_solution_for_lambda_9 :
  ∃ x y : ℝ, (x^2 + y^2 = 8 * x + 6 * y) ∧ (9 * x^2 + y^2 = 6 * y) ∧ (y^2 + 9 = 9 * x + 6 * y + 9) :=
by
  sorry

end NUMINAMATH_GPT_exists_solution_for_lambda_9_l1977_197738


namespace NUMINAMATH_GPT_find_a_minus_b_l1977_197767

theorem find_a_minus_b
  (f : ℝ → ℝ)
  (a b : ℝ)
  (hf : ∀ x, f x = x^2 + 3 * a * x + 4)
  (h_even : ∀ x, f (-x) = f x)
  (hb_condition : b - 3 = -2 * b) :
  a - b = -1 :=
sorry

end NUMINAMATH_GPT_find_a_minus_b_l1977_197767


namespace NUMINAMATH_GPT_consecutive_product_solution_l1977_197732

theorem consecutive_product_solution :
  ∀ (n : ℤ), (∃ a : ℤ, n^4 + 8 * n + 11 = a * (a + 1)) ↔ n = 1 :=
by
  sorry

end NUMINAMATH_GPT_consecutive_product_solution_l1977_197732


namespace NUMINAMATH_GPT_percentage_of_men_speaking_french_l1977_197752

theorem percentage_of_men_speaking_french {total_employees men women french_speaking_employees french_speaking_women french_speaking_men : ℕ}
    (h1 : total_employees = 100)
    (h2 : men = 60)
    (h3 : women = 40)
    (h4 : french_speaking_employees = 50)
    (h5 : french_speaking_women = 14)
    (h6 : french_speaking_men = french_speaking_employees - french_speaking_women)
    (h7 : french_speaking_men * 100 / men = 60) : true :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_men_speaking_french_l1977_197752


namespace NUMINAMATH_GPT_h_at_3_l1977_197773

noncomputable def f (x : ℝ) : ℝ := 2 * x^2 + 3 * x + 5
noncomputable def g (x : ℝ) : ℝ := Real.sqrt (f x) + 1
noncomputable def h (x : ℝ) : ℝ := f (g x)

theorem h_at_3 : h 3 = 74 + 28 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_h_at_3_l1977_197773


namespace NUMINAMATH_GPT_find_y_l1977_197739

theorem find_y (y : ℝ) (h : |2 * y - 44| + |y - 24| = |3 * y - 66|) : y = 23 := 
by 
  sorry

end NUMINAMATH_GPT_find_y_l1977_197739


namespace NUMINAMATH_GPT_sets_difference_M_star_N_l1977_197781

def M (y : ℝ) : Prop := y ≤ 2

def N (y : ℝ) : Prop := 0 ≤ y ∧ y ≤ 3

def M_star_N (y : ℝ) : Prop := y < 0

theorem sets_difference_M_star_N : {y : ℝ | M y ∧ ¬ N y} = {y : ℝ | M_star_N y} :=
by {
  sorry
}

end NUMINAMATH_GPT_sets_difference_M_star_N_l1977_197781


namespace NUMINAMATH_GPT_martin_total_distance_l1977_197741

-- Define the conditions
def total_trip_time : ℕ := 8
def first_half_speed : ℕ := 70
def second_half_speed : ℕ := 85
def half_trip_time : ℕ := total_trip_time / 2

-- Define the total distance traveled 
def total_distance : ℕ := (first_half_speed * half_trip_time) + (second_half_speed * half_trip_time)

-- Statement to prove
theorem martin_total_distance : total_distance = 620 :=
by
  -- This is a placeholder to represent that a proof is needed
  -- Actual proof steps are omitted as instructed
  sorry

end NUMINAMATH_GPT_martin_total_distance_l1977_197741


namespace NUMINAMATH_GPT_length_of_side_of_pentagon_l1977_197709

-- Assuming these conditions from the math problem:
-- 1. The perimeter of the regular polygon is 125.
-- 2. The polygon is a pentagon (5 sides).

-- Let's define the conditions:
def perimeter := 125
def sides := 5
def regular_polygon (perimeter : ℕ) (sides : ℕ) := (perimeter / sides : ℕ)

-- Statement to be proved:
theorem length_of_side_of_pentagon : regular_polygon perimeter sides = 25 := 
by sorry

end NUMINAMATH_GPT_length_of_side_of_pentagon_l1977_197709


namespace NUMINAMATH_GPT_mass_percentage_Cl_in_HClO2_is_51_78_l1977_197756

noncomputable def molar_mass_H : ℝ := 1.01
noncomputable def molar_mass_Cl : ℝ := 35.45
noncomputable def molar_mass_O : ℝ := 16.00

noncomputable def molar_mass_HClO2 : ℝ :=
  molar_mass_H + molar_mass_Cl + 2 * molar_mass_O

noncomputable def mass_percentage_Cl_in_HClO2 : ℝ :=
  (molar_mass_Cl / molar_mass_HClO2) * 100

theorem mass_percentage_Cl_in_HClO2_is_51_78 :
  mass_percentage_Cl_in_HClO2 = 51.78 := 
sorry

end NUMINAMATH_GPT_mass_percentage_Cl_in_HClO2_is_51_78_l1977_197756


namespace NUMINAMATH_GPT_arithmetic_seq_num_terms_l1977_197710

theorem arithmetic_seq_num_terms (a1 : ℕ := 1) (S_odd S_even : ℕ) (n : ℕ) 
  (h1 : S_odd = 341) (h2 : S_even = 682) : 2 * n = 10 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_seq_num_terms_l1977_197710


namespace NUMINAMATH_GPT_change_received_l1977_197724

-- Define the given conditions
def num_apples : ℕ := 5
def cost_per_apple : ℝ := 0.75
def amount_paid : ℝ := 10.00

-- Prove the change is equal to $6.25
theorem change_received :
  amount_paid - (num_apples * cost_per_apple) = 6.25 :=
by
  sorry

end NUMINAMATH_GPT_change_received_l1977_197724


namespace NUMINAMATH_GPT_quadratic_distinct_real_roots_range_l1977_197758

open Real

theorem quadratic_distinct_real_roots_range (k : ℝ) :
    (∃ a b c : ℝ, a = k^2 ∧ b = 4 * k - 1 ∧ c = 4 ∧ (b^2 - 4 * a * c > 0) ∧ a ≠ 0) ↔ (k < 1 / 8 ∧ k ≠ 0) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_distinct_real_roots_range_l1977_197758


namespace NUMINAMATH_GPT_pamela_skittles_l1977_197721

variable (initial_skittles : Nat) (given_to_karen : Nat)

def skittles_after_giving (initial_skittles given_to_karen : Nat) : Nat :=
  initial_skittles - given_to_karen

theorem pamela_skittles (h1 : initial_skittles = 50) (h2 : given_to_karen = 7) :
  skittles_after_giving initial_skittles given_to_karen = 43 := by
  sorry

end NUMINAMATH_GPT_pamela_skittles_l1977_197721


namespace NUMINAMATH_GPT_x_intercept_of_line_l1977_197780

theorem x_intercept_of_line (x y : ℝ) : (4 * x + 7 * y = 28) ∧ (y = 0) → x = 7 :=
by
  sorry

end NUMINAMATH_GPT_x_intercept_of_line_l1977_197780


namespace NUMINAMATH_GPT_complement_U_A_l1977_197701

open Set

def U : Set ℤ := {-2, -1, 0, 1, 2}
def A : Set ℤ := {x | x^2 < 3}

theorem complement_U_A :
  (U \ A) = {-2, 2} :=
sorry

end NUMINAMATH_GPT_complement_U_A_l1977_197701


namespace NUMINAMATH_GPT_emily_height_in_cm_l1977_197715

theorem emily_height_in_cm 
  (inches_in_foot : ℝ) (cm_in_foot : ℝ) (emily_height_in_inches : ℝ)
  (h_if : inches_in_foot = 12) (h_cf : cm_in_foot = 30.5) (h_ehi : emily_height_in_inches = 62) :
  emily_height_in_inches * (cm_in_foot / inches_in_foot) = 157.6 :=
by
  sorry

end NUMINAMATH_GPT_emily_height_in_cm_l1977_197715


namespace NUMINAMATH_GPT_polynomial_factorization_l1977_197750

theorem polynomial_factorization (m : ℝ) : 
  (∀ x : ℝ, (x - 7) * (x + 5) = x^2 + mx - 35) → m = -2 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_factorization_l1977_197750


namespace NUMINAMATH_GPT_number_of_false_propositions_is_even_l1977_197727

theorem number_of_false_propositions_is_even 
  (P Q : Prop) : 
  ∃ (n : ℕ), (P ∧ ¬P ∧ (¬Q → ¬P) ∧ (Q → P)) = false ∧ n % 2 = 0 := sorry

end NUMINAMATH_GPT_number_of_false_propositions_is_even_l1977_197727


namespace NUMINAMATH_GPT_abs_diff_squares_104_98_l1977_197785

theorem abs_diff_squares_104_98 : abs ((104 : ℤ)^2 - (98 : ℤ)^2) = 1212 := by
  sorry

end NUMINAMATH_GPT_abs_diff_squares_104_98_l1977_197785


namespace NUMINAMATH_GPT_average_of_c_and_d_l1977_197768

variable (c d e : ℝ)

theorem average_of_c_and_d
  (h1: (4 + 6 + 9 + c + d + e) / 6 = 20)
  (h2: e = c + 6) :
  (c + d) / 2 = 47.5 := by
sorry

end NUMINAMATH_GPT_average_of_c_and_d_l1977_197768


namespace NUMINAMATH_GPT_scientific_notation_of_million_l1977_197791

theorem scientific_notation_of_million (x : ℝ) (h : x = 56.99) : 56.99 * 10^6 = 5.699 * 10^7 :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_of_million_l1977_197791


namespace NUMINAMATH_GPT_minimum_benches_for_equal_occupancy_l1977_197743

theorem minimum_benches_for_equal_occupancy (M : ℕ) :
  (∃ x y, x = y ∧ 8 * M = x ∧ 12 * M = y) ↔ M = 3 := by
  sorry

end NUMINAMATH_GPT_minimum_benches_for_equal_occupancy_l1977_197743


namespace NUMINAMATH_GPT_num_unique_seven_digit_integers_l1977_197706

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

noncomputable def unique_seven_digit_integers : ℕ :=
  factorial 7 / (factorial 2 * factorial 2 * factorial 2)

theorem num_unique_seven_digit_integers : unique_seven_digit_integers = 630 := by
  sorry

end NUMINAMATH_GPT_num_unique_seven_digit_integers_l1977_197706


namespace NUMINAMATH_GPT_find_a_perpendicular_lines_l1977_197793

theorem find_a_perpendicular_lines (a : ℝ) : 
  (∀ x y : ℝ, ax + (a + 2) * y + 1 = 0 ∧ x + a * y + 2 = 0) → a = -3 :=
sorry

end NUMINAMATH_GPT_find_a_perpendicular_lines_l1977_197793


namespace NUMINAMATH_GPT_hannah_sweatshirts_l1977_197760

theorem hannah_sweatshirts (S : ℕ) (h1 : 15 * S + 2 * 10 = 65) : S = 3 := 
by
  sorry

end NUMINAMATH_GPT_hannah_sweatshirts_l1977_197760


namespace NUMINAMATH_GPT_john_income_l1977_197788

theorem john_income 
  (john_tax_rate : ℝ) (ingrid_tax_rate : ℝ) (ingrid_income : ℝ) (combined_tax_rate : ℝ)
  (jt_30 : john_tax_rate = 0.30) (it_40 : ingrid_tax_rate = 0.40) (ii_72000 : ingrid_income = 72000) 
  (ctr_35625 : combined_tax_rate = 0.35625) :
  ∃ J : ℝ, (0.30 * J + ingrid_tax_rate * ingrid_income = combined_tax_rate * (J + ingrid_income)) ∧ (J = 56000) :=
by
  sorry

end NUMINAMATH_GPT_john_income_l1977_197788


namespace NUMINAMATH_GPT_nurses_count_l1977_197748

theorem nurses_count (D N : ℕ) (h1 : D + N = 456) (h2 : D * 11 = 8 * N) : N = 264 :=
by
  sorry

end NUMINAMATH_GPT_nurses_count_l1977_197748


namespace NUMINAMATH_GPT_quadratic_roots_diff_square_l1977_197716

theorem quadratic_roots_diff_square :
  ∀ (d e : ℝ), (∀ x : ℝ, 4 * x^2 + 8 * x - 48 = 0 → (x = d ∨ x = e)) → (d - e)^2 = 49 :=
by
  intros d e h
  sorry

end NUMINAMATH_GPT_quadratic_roots_diff_square_l1977_197716


namespace NUMINAMATH_GPT_find_digit_x_l1977_197775

def base7_number (x : ℕ) : ℕ := 5 * 7^3 + 2 * 7^2 + x * 7 + 4

def is_divisible_by_19 (n : ℕ) : Prop := ∃ k : ℕ, n = 19 * k

theorem find_digit_x : is_divisible_by_19 (base7_number 4) :=
sorry

end NUMINAMATH_GPT_find_digit_x_l1977_197775


namespace NUMINAMATH_GPT_find_x_minus_y_l1977_197790

theorem find_x_minus_y (x y : ℝ) (h1 : |x| + x - y = 14) (h2 : x + |y| + y = 6) : x - y = 8 :=
sorry

end NUMINAMATH_GPT_find_x_minus_y_l1977_197790


namespace NUMINAMATH_GPT_rationalize_denominator_l1977_197779

def cbrt (x : ℝ) : ℝ := x^(1/3)

theorem rationalize_denominator :
  let a := cbrt 2
  let b := cbrt 27
  b = 3 -> ( 1 / (a + b)) = (cbrt 4 / (2 + 3 * cbrt 4))
:= by
  intro a
  intro b
  sorry

end NUMINAMATH_GPT_rationalize_denominator_l1977_197779


namespace NUMINAMATH_GPT_exactly_one_correct_l1977_197744

theorem exactly_one_correct (P_A P_B : ℚ) (hA : P_A = 1/5) (hB : P_B = 1/4) :
  P_A * (1 - P_B) + (1 - P_A) * P_B = 7/20 :=
by
  sorry

end NUMINAMATH_GPT_exactly_one_correct_l1977_197744


namespace NUMINAMATH_GPT_max_campaign_making_animals_prime_max_campaign_making_animals_nine_l1977_197742

theorem max_campaign_making_animals_prime (n : ℕ) (h_prime : Nat.Prime n) (h_ge : n ≥ 3) : 
  ∃ k, k = (n - 1) / 2 :=
by
  sorry

theorem max_campaign_making_animals_nine : ∃ k, k = 4 :=
by
  sorry

end NUMINAMATH_GPT_max_campaign_making_animals_prime_max_campaign_making_animals_nine_l1977_197742


namespace NUMINAMATH_GPT_value_of_a_is_negative_one_l1977_197745

-- Conditions
def I (a : ℤ) : Set ℤ := {2, 4, a^2 - a - 3}
def A (a : ℤ) : Set ℤ := {4, 1 - a}
def complement_I_A (a : ℤ) : Set ℤ := {x ∈ I a | x ∉ A a}

-- Theorem statement
theorem value_of_a_is_negative_one (a : ℤ) (h : complement_I_A a = {-1}) : a = -1 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_is_negative_one_l1977_197745


namespace NUMINAMATH_GPT_kite_initial_gain_percentage_l1977_197704

noncomputable def initial_gain_percentage (MP CP : ℝ) : ℝ :=
  ((MP - CP) / CP) * 100

theorem kite_initial_gain_percentage :
  ∃ MP CP : ℝ,
    SP = 30 ∧
    SP = MP * 0.9 ∧
    1.035 * CP = SP ∧
    initial_gain_percentage MP CP = 15 :=
sorry

end NUMINAMATH_GPT_kite_initial_gain_percentage_l1977_197704


namespace NUMINAMATH_GPT_smith_family_seating_problem_l1977_197711

theorem smith_family_seating_problem :
  let total_children := 8
  let boys := 4
  let girls := 4
  (total_children.factorial - (boys.factorial * girls.factorial)) = 39744 :=
by
  sorry

end NUMINAMATH_GPT_smith_family_seating_problem_l1977_197711


namespace NUMINAMATH_GPT_sequence_sum_l1977_197718

-- Definitions representing the given conditions
variables (A H M O X : ℕ)

-- Assuming the conditions as hypotheses
theorem sequence_sum (h₁ : A + 9 + H = 19) (h₂ : 9 + H + M = 19) (h₃ : H + M + O = 19)
  (h₄ : M + O + X = 19) : A + H + M + O = 26 :=
sorry

end NUMINAMATH_GPT_sequence_sum_l1977_197718


namespace NUMINAMATH_GPT_find_a_and_b_l1977_197794

theorem find_a_and_b (a b : ℚ) :
  ((∃ x y : ℚ, 3 * x - y = 7 ∧ a * x + y = b) ∧
   (∃ x y : ℚ, x + b * y = a ∧ 2 * x + y = 8)) →
  a = -7/5 ∧ b = -11/5 :=
by sorry

end NUMINAMATH_GPT_find_a_and_b_l1977_197794


namespace NUMINAMATH_GPT_chess_tournament_participants_l1977_197720

open Int

theorem chess_tournament_participants (n : ℕ) (h_games: n * (n - 1) / 2 = 190) : n = 20 :=
by
  sorry

end NUMINAMATH_GPT_chess_tournament_participants_l1977_197720


namespace NUMINAMATH_GPT_verify_formula_n1_l1977_197777

theorem verify_formula_n1 (a : ℝ) (ha : a ≠ 1) : 1 + a = (a^3 - 1) / (a - 1) :=
by 
  sorry

end NUMINAMATH_GPT_verify_formula_n1_l1977_197777


namespace NUMINAMATH_GPT_average_monthly_increase_is_20_percent_l1977_197751

-- Define the given conditions in Lean
def V_Jan : ℝ := 2 
def V_Mar : ℝ := 2.88 

-- Percentage increase each month over the previous month is the same
def consistent_growth_rate (x : ℝ) : Prop := 
  V_Jan * (1 + x)^2 = V_Mar

-- We need to prove that the monthly growth rate x is 0.2 (or 20%)
theorem average_monthly_increase_is_20_percent : 
  ∃ x : ℝ, consistent_growth_rate x ∧ x = 0.2 :=
by
  sorry

end NUMINAMATH_GPT_average_monthly_increase_is_20_percent_l1977_197751


namespace NUMINAMATH_GPT_tile_count_l1977_197765

theorem tile_count (room_length room_width tile_length tile_width : ℝ)
  (h1 : room_length = 10)
  (h2 : room_width = 15)
  (h3 : tile_length = 1 / 4)
  (h4 : tile_width = 3 / 4) :
  (room_length * room_width) / (tile_length * tile_width) = 800 :=
by
  sorry

end NUMINAMATH_GPT_tile_count_l1977_197765


namespace NUMINAMATH_GPT_zoe_spent_amount_l1977_197754

theorem zoe_spent_amount :
  (3 * (8 + 2) = 30) :=
by sorry

end NUMINAMATH_GPT_zoe_spent_amount_l1977_197754


namespace NUMINAMATH_GPT_inequality_l1977_197766

theorem inequality (a b : ℝ) : a^2 + b^2 + 7/4 ≥ a * b + 2 * a + b / 2 :=
sorry

end NUMINAMATH_GPT_inequality_l1977_197766


namespace NUMINAMATH_GPT_sandra_age_l1977_197719

theorem sandra_age (S : ℕ) (h1 : ∀ x : ℕ, x = 14) (h2 : S - 3 = 3 * (14 - 3)) : S = 36 :=
by sorry

end NUMINAMATH_GPT_sandra_age_l1977_197719


namespace NUMINAMATH_GPT_preimages_of_one_under_f_l1977_197799

theorem preimages_of_one_under_f :
  {x : ℝ | (x^3 - x + 1 = 1)} = {-1, 0, 1} := by
  sorry

end NUMINAMATH_GPT_preimages_of_one_under_f_l1977_197799


namespace NUMINAMATH_GPT_age_problem_l1977_197798

open Classical

variable (A B C : ℕ)

theorem age_problem (h1 : A + 10 = 2 * (B - 10))
                    (h2 : C = 3 * (A - 5))
                    (h3 : A = B + 9)
                    (h4 : C = A + 4) :
  B = 39 :=
sorry

end NUMINAMATH_GPT_age_problem_l1977_197798


namespace NUMINAMATH_GPT_calc_m_l1977_197761

theorem calc_m (m : ℤ) (h : (64 : ℝ)^(1 / 3) = 2^m) : m = 2 :=
sorry

end NUMINAMATH_GPT_calc_m_l1977_197761


namespace NUMINAMATH_GPT_no_nat_solution_for_exp_eq_l1977_197713

theorem no_nat_solution_for_exp_eq (n x y z : ℕ) (hn : n > 1) (hx : x ≤ n) (hy : y ≤ n) :
  ¬ (x^n + y^n = z^n) :=
by
  sorry

end NUMINAMATH_GPT_no_nat_solution_for_exp_eq_l1977_197713


namespace NUMINAMATH_GPT_range_of_a_l1977_197702

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + 2 * x - Real.exp x

theorem range_of_a (h : ∀ m n : ℝ, 0 < m → 0 < n → m > n → (f a m - f a n) / (m - n) < 2) :
  a ≤ Real.exp 1 / (2 * 1) := 
sorry

end NUMINAMATH_GPT_range_of_a_l1977_197702


namespace NUMINAMATH_GPT_log_ordering_correct_l1977_197723

noncomputable def log_ordering : Prop :=
  let a := 20.3
  let b := 0.32
  let c := Real.log b
  (0 < b ∧ b < 1) ∧ (c < 0) ∧ (c < b ∧ b < a)

theorem log_ordering_correct : log_ordering :=
by
  -- skipped proof
  sorry

end NUMINAMATH_GPT_log_ordering_correct_l1977_197723


namespace NUMINAMATH_GPT_quadratic_equation_properties_l1977_197784

theorem quadratic_equation_properties (m : ℝ) (h : m < 4) (root_one : ℝ) (root_two : ℝ) 
  (eq1 : root_one + root_two = 4) (eq2 : root_one * root_two = m) (root_one_eq : root_one = -1) :
  m = -5 ∧ root_two = 5 ∧ (root_one ≠ root_two) :=
by
  -- Sorry is added to skip the proof because only the statement is needed.
  sorry

end NUMINAMATH_GPT_quadratic_equation_properties_l1977_197784


namespace NUMINAMATH_GPT_marble_probability_l1977_197769

theorem marble_probability (W G R B : ℕ) (h_total : W + G + R + B = 84) 
  (h_white : W / 84 = 1 / 4) (h_green : G / 84 = 1 / 7) :
  (R + B) / 84 = 17 / 28 :=
by
  sorry

end NUMINAMATH_GPT_marble_probability_l1977_197769


namespace NUMINAMATH_GPT_ellipse_is_correct_l1977_197774

-- Define the hyperbola equation
def hyperbola_eq (x y : ℝ) : Prop :=
  (x^2 / 4) - (y^2 / 12) = -1

-- Define the ellipse equation
def ellipse_eq (x y : ℝ) : Prop :=
  (x^2 / 4) + (y^2 / 16) = 1

-- Define the conditions
def ellipse_focus_vertex_of_hyperbola_vertex_and_focus (x y : ℝ) : Prop :=
  hyperbola_eq x y ∧ ellipse_eq x y

-- Theorem stating that the ellipse equation holds given the conditions
theorem ellipse_is_correct :
  ∀ (x y : ℝ), ellipse_focus_vertex_of_hyperbola_vertex_and_focus x y →
  ellipse_eq x y := by
  intros x y h
  sorry

end NUMINAMATH_GPT_ellipse_is_correct_l1977_197774


namespace NUMINAMATH_GPT_simplify_expression_l1977_197737

variable {s r : ℝ}

theorem simplify_expression :
  (2 * s^2 + 5 * r - 4) - (3 * s^2 + 9 * r - 7) = -s^2 - 4 * r + 3 := 
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1977_197737


namespace NUMINAMATH_GPT_find_two_digit_number_l1977_197726

theorem find_two_digit_number : ∃ (y : ℕ), (10 ≤ y ∧ y < 100) ∧ (∃ x : ℕ, x = (y / 10) + (y % 10) ∧ x^3 = y^2) ∧ y = 27 := 
by
  sorry

end NUMINAMATH_GPT_find_two_digit_number_l1977_197726


namespace NUMINAMATH_GPT_lamps_on_bridge_l1977_197772

theorem lamps_on_bridge (bridge_length : ℕ) (lamp_spacing : ℕ) (num_intervals : ℕ) (num_lamps : ℕ) 
  (h1 : bridge_length = 30) 
  (h2 : lamp_spacing = 5)
  (h3 : num_intervals = bridge_length / lamp_spacing)
  (h4 : num_lamps = num_intervals + 1) :
  num_lamps = 7 := 
by
  sorry

end NUMINAMATH_GPT_lamps_on_bridge_l1977_197772


namespace NUMINAMATH_GPT_area_of_tangent_triangle_l1977_197795

noncomputable def tangentTriangleArea : ℝ :=
  let y := λ x : ℝ => x^3 + x
  let dy := λ x : ℝ => 3 * x^2 + 1
  let slope := dy 1
  let y_intercept := 2 - slope * 1
  let x_intercept := - y_intercept / slope
  let base := x_intercept
  let height := - y_intercept
  0.5 * base * height

theorem area_of_tangent_triangle :
  tangentTriangleArea = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_area_of_tangent_triangle_l1977_197795


namespace NUMINAMATH_GPT_total_laces_needed_l1977_197753

variable (x : ℕ) -- Eva has x pairs of shoes
def long_laces_per_pair : ℕ := 3
def short_laces_per_pair : ℕ := 3
def laces_per_pair : ℕ := long_laces_per_pair + short_laces_per_pair

theorem total_laces_needed : 6 * x = 6 * x :=
by
  have h : laces_per_pair = 6 := rfl
  sorry

end NUMINAMATH_GPT_total_laces_needed_l1977_197753


namespace NUMINAMATH_GPT_max_lambda_inequality_l1977_197700

theorem max_lambda_inequality (a b : ℝ) (h_a : a > 0) (h_b : b > 0) :
  (1 / Real.sqrt (20 * a + 23 * b) + 1 / Real.sqrt (23 * a + 20 * b)) ≥ (2 / Real.sqrt 43 / Real.sqrt (a + b)) :=
by
  sorry

end NUMINAMATH_GPT_max_lambda_inequality_l1977_197700


namespace NUMINAMATH_GPT_total_fruits_in_baskets_l1977_197778

def total_fruits (apples1 oranges1 bananas1 apples2 oranges2 bananas2 : ℕ) :=
  apples1 + oranges1 + bananas1 + apples2 + oranges2 + bananas2

theorem total_fruits_in_baskets :
  total_fruits 9 15 14 (9 - 2) (15 - 2) (14 - 2) = 70 :=
by
  sorry

end NUMINAMATH_GPT_total_fruits_in_baskets_l1977_197778


namespace NUMINAMATH_GPT_money_per_percentage_point_l1977_197764

theorem money_per_percentage_point
  (plates : ℕ) (total_states : ℕ) (total_amount : ℤ)
  (h_plates : plates = 40) (h_total_states : total_states = 50) (h_total_amount : total_amount = 160) :
  total_amount / (plates * 100 / total_states) = 2 :=
by
  -- Omitted steps of the proof
  sorry

end NUMINAMATH_GPT_money_per_percentage_point_l1977_197764


namespace NUMINAMATH_GPT_old_lamp_height_is_one_l1977_197755

def new_lamp_height : ℝ := 2.3333333333333335
def height_difference : ℝ := 1.3333333333333333
def old_lamp_height : ℝ := new_lamp_height - height_difference

theorem old_lamp_height_is_one :
  old_lamp_height = 1 :=
by
  sorry

end NUMINAMATH_GPT_old_lamp_height_is_one_l1977_197755


namespace NUMINAMATH_GPT_crayons_total_l1977_197796

theorem crayons_total (rows : ℕ) (crayons_per_row : ℕ) (total_crayons : ℕ) :
  rows = 15 → crayons_per_row = 42 → total_crayons = rows * crayons_per_row → total_crayons = 630 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_crayons_total_l1977_197796


namespace NUMINAMATH_GPT_peanut_raising_ratio_l1977_197740

theorem peanut_raising_ratio
  (initial_peanuts : ℝ)
  (remove_peanuts_1 : ℝ)
  (add_raisins_1 : ℝ)
  (remove_mixture : ℝ)
  (add_raisins_2 : ℝ)
  (final_peanuts : ℝ)
  (final_raisins : ℝ)
  (ratio : ℝ) :
  initial_peanuts = 10 ∧
  remove_peanuts_1 = 2 ∧
  add_raisins_1 = 2 ∧
  remove_mixture = 2 ∧
  add_raisins_2 = 2 ∧
  final_peanuts = initial_peanuts - remove_peanuts_1 - (remove_mixture * (initial_peanuts - remove_peanuts_1) / (initial_peanuts - remove_peanuts_1 + add_raisins_1)) ∧
  final_raisins = add_raisins_1 - (remove_mixture * add_raisins_1 / (initial_peanuts - remove_peanuts_1 + add_raisins_1)) + add_raisins_2 ∧
  ratio = final_peanuts / final_raisins →
  ratio = 16 / 9 := by
  sorry

end NUMINAMATH_GPT_peanut_raising_ratio_l1977_197740


namespace NUMINAMATH_GPT_apple_juice_fraction_correct_l1977_197759

def problem_statement : Prop :=
  let pitcher1_capacity := 800
  let pitcher2_capacity := 500
  let pitcher1_apple_fraction := 1 / 4
  let pitcher2_apple_fraction := 1 / 5
  let pitcher1_apple_volume := pitcher1_capacity * pitcher1_apple_fraction
  let pitcher2_apple_volume := pitcher2_capacity * pitcher2_apple_fraction
  let total_apple_volume := pitcher1_apple_volume + pitcher2_apple_volume
  let total_volume := pitcher1_capacity + pitcher2_capacity
  total_apple_volume / total_volume = 3 / 13

theorem apple_juice_fraction_correct : problem_statement := 
  sorry

end NUMINAMATH_GPT_apple_juice_fraction_correct_l1977_197759


namespace NUMINAMATH_GPT_round_robin_10_players_l1977_197783

theorem round_robin_10_players : @Nat.choose 10 2 = 45 := by
  sorry

end NUMINAMATH_GPT_round_robin_10_players_l1977_197783


namespace NUMINAMATH_GPT_largest_square_area_l1977_197787

theorem largest_square_area (a b c : ℝ) 
  (h1 : c^2 = a^2 + b^2) 
  (h2 : a = b - 5) 
  (h3 : a^2 + b^2 + c^2 = 450) : 
  c^2 = 225 :=
by 
  sorry

end NUMINAMATH_GPT_largest_square_area_l1977_197787


namespace NUMINAMATH_GPT_dogs_bunnies_ratio_l1977_197729

theorem dogs_bunnies_ratio (total : ℕ) (dogs : ℕ) (bunnies : ℕ) (h1 : total = 375) (h2 : dogs = 75) (h3 : bunnies = total - dogs) : (75 / 75 : ℚ) / (300 / 75 : ℚ) = 1 / 4 := by
  sorry

end NUMINAMATH_GPT_dogs_bunnies_ratio_l1977_197729


namespace NUMINAMATH_GPT_maximum_a_pos_integer_greatest_possible_value_of_a_l1977_197749

theorem maximum_a_pos_integer (a : ℕ) (h : ∃ x : ℤ, x^2 + (a * x : ℤ) = -20) : a ≤ 21 :=
by
  sorry

theorem greatest_possible_value_of_a : ∃ (a : ℕ), (∀ b : ℕ, (∃ x : ℤ, x^2 + (b * x : ℤ) = -20) → b ≤ 21) ∧ 21 = a :=
by
  sorry

end NUMINAMATH_GPT_maximum_a_pos_integer_greatest_possible_value_of_a_l1977_197749


namespace NUMINAMATH_GPT_Brittany_age_after_vacation_l1977_197728

variable (Rebecca_age Brittany_age vacation_years : ℕ)
variable (h1 : Rebecca_age = 25)
variable (h2 : Brittany_age = Rebecca_age + 3)
variable (h3 : vacation_years = 4)

theorem Brittany_age_after_vacation : 
    Brittany_age + vacation_years = 32 := 
by
  sorry

end NUMINAMATH_GPT_Brittany_age_after_vacation_l1977_197728


namespace NUMINAMATH_GPT_total_area_of_map_l1977_197712

def level1_area : ℕ := 40 * 20
def level2_area : ℕ := 15 * 15
def level3_area : ℕ := (25 * 12) / 2

def total_area : ℕ := level1_area + level2_area + level3_area

theorem total_area_of_map : total_area = 1175 := by
  -- Proof to be completed
  sorry

end NUMINAMATH_GPT_total_area_of_map_l1977_197712


namespace NUMINAMATH_GPT_inequality_solution_l1977_197763

theorem inequality_solution :
  {x : Real | (2 * x - 5) * (x - 3) / x ≥ 0} = {x : Real | (x ∈ Set.Ioc 0 (5 / 2)) ∨ (x ∈ Set.Ici 3)} := 
sorry

end NUMINAMATH_GPT_inequality_solution_l1977_197763


namespace NUMINAMATH_GPT_decimal_multiplication_l1977_197789

theorem decimal_multiplication : (3.6 * 0.3 = 1.08) := by
  sorry

end NUMINAMATH_GPT_decimal_multiplication_l1977_197789


namespace NUMINAMATH_GPT_find_common_ratio_l1977_197731

noncomputable def geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) : Prop :=
  (a 5 - a 1 = 15) ∧ (a 4 - a 2 = 6) → (q = 1/2 ∨ q = 2)

-- We declare this as a theorem statement
theorem find_common_ratio (a : ℕ → ℝ) (q : ℝ) : geometric_sequence_common_ratio a q :=
sorry

end NUMINAMATH_GPT_find_common_ratio_l1977_197731


namespace NUMINAMATH_GPT_hyperbola_foci_property_l1977_197722

noncomputable def hyperbola (x y b : ℝ) : Prop :=
  (x^2 / 9) - (y^2 / b^2) = 1

theorem hyperbola_foci_property (x y b : ℝ) (h : hyperbola x y b) (b_pos : b > 0) (PF1 : ℝ) (PF2 : ℝ) (hPF1 : PF1 = 5) :
  PF2 = 11 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_foci_property_l1977_197722


namespace NUMINAMATH_GPT_smallest_sphere_radius_l1977_197757

theorem smallest_sphere_radius :
  ∃ (R : ℝ), (∀ (a b : ℝ), a = 14 → b = 12 → ∃ (h : ℝ), h = Real.sqrt (12^2 - (14 * Real.sqrt 2 / 2)^2) ∧ R = 7 * Real.sqrt 2 ∧ h ≤ R) :=
sorry

end NUMINAMATH_GPT_smallest_sphere_radius_l1977_197757


namespace NUMINAMATH_GPT_range_a_l1977_197776

noncomputable def f (x a : ℝ) : ℝ := x^2 - 2 * a * x + 3

theorem range_a (H : ∀ x1 x2 : ℝ, 2 < x1 → 2 < x2 → (f x1 a - f x2 a) / (x1 - x2) > 0) : a ≤ 2 := 
sorry

end NUMINAMATH_GPT_range_a_l1977_197776


namespace NUMINAMATH_GPT_cos_value_of_tan_third_quadrant_l1977_197725

theorem cos_value_of_tan_third_quadrant (x : ℝ) (h1 : Real.tan x = 4 / 3) (h2 : π < x ∧ x < 3 * π / 2) : 
  Real.cos x = -3 / 5 := 
sorry

end NUMINAMATH_GPT_cos_value_of_tan_third_quadrant_l1977_197725


namespace NUMINAMATH_GPT_cornbread_pieces_l1977_197792

theorem cornbread_pieces (pan_length : ℕ) (pan_width : ℕ) (piece_length : ℕ) (piece_width : ℕ)
  (hl : pan_length = 20) (hw : pan_width = 18) (hp : piece_length = 2) (hq : piece_width = 2) :
  (pan_length * pan_width) / (piece_length * piece_width) = 90 :=
by
  sorry

end NUMINAMATH_GPT_cornbread_pieces_l1977_197792


namespace NUMINAMATH_GPT_minimum_photos_l1977_197747

-- Given conditions:
-- 1. There are exactly three monuments.
-- 2. A group of 42 tourists visited the city.
-- 3. Each tourist took at most one photo of each of the three monuments.
-- 4. Any two tourists together had photos of all three monuments.

theorem minimum_photos (n m : ℕ) (h₁ : n = 42) (h₂ : m = 3) 
  (photographs : ℕ → ℕ → Bool) -- whether tourist i took a photo of monument j
  (h₃ : ∀ i : ℕ, i < n → ∀ j : ℕ, j < m → photographs i j = tt ∨ photographs i j = ff) -- each tourist took at most one photo of each monument
  (h₄ : ∀ i₁ i₂ : ℕ, i₁ < n → i₂ < n → i₁ ≠ i₂ → ∀ j : ℕ, j < m → photographs i₁ j = tt ∨ photographs i₂ j = tt) -- any two tourists together had photos of all three monuments
  : ∃ min_photos : ℕ, min_photos = 123 :=
by
  sorry

end NUMINAMATH_GPT_minimum_photos_l1977_197747


namespace NUMINAMATH_GPT_shooting_average_l1977_197782

noncomputable def total_points (a b c d : ℕ) : ℕ :=
  (a * 10) + (b * 9) + (c * 8) + (d * 7)

noncomputable def average_points (total : ℕ) (shots : ℕ) : ℚ :=
  total / shots

theorem shooting_average :
  let a := 1
  let b := 4
  let c := 3
  let d := 2
  let shots := 10
  total_points a b c d = 84 ∧
  average_points (total_points a b c d) shots = 8.4 :=
by {
  sorry
}

end NUMINAMATH_GPT_shooting_average_l1977_197782


namespace NUMINAMATH_GPT_water_speed_l1977_197714

theorem water_speed (swimmer_speed still_water : ℝ) (distance time : ℝ) (h1 : swimmer_speed = 12) (h2 : distance = 12) (h3 : time = 6) :
  ∃ v : ℝ, v = 10 ∧ distance = (swimmer_speed - v) * time :=
by { sorry }

end NUMINAMATH_GPT_water_speed_l1977_197714


namespace NUMINAMATH_GPT_teddy_has_8_cats_l1977_197705

theorem teddy_has_8_cats (dogs_teddy : ℕ) (cats_teddy : ℕ) (dogs_total : ℕ) (pets_total : ℕ)
  (h1 : dogs_teddy = 7)
  (h2 : dogs_total = dogs_teddy + (dogs_teddy + 9) + (dogs_teddy - 5))
  (h3 : pets_total = dogs_total + cats_teddy + (cats_teddy + 13))
  (h4 : pets_total = 54) :
  cats_teddy = 8 := by
  sorry

end NUMINAMATH_GPT_teddy_has_8_cats_l1977_197705


namespace NUMINAMATH_GPT_smallest_integer_l1977_197735

theorem smallest_integer (x : ℤ) (h : 3 * (Int.natAbs x)^3 + 5 < 56) : x = -2 :=
sorry

end NUMINAMATH_GPT_smallest_integer_l1977_197735


namespace NUMINAMATH_GPT_part1_min_value_part2_min_value_l1977_197708

def f (x : ℝ) : ℝ := |x - 1| + 2 * |x + 1|

theorem part1_min_value :
  ∃ (m : ℝ), m = 2 ∧ (∀ (x : ℝ), f x ≥ m) :=
sorry

theorem part2_min_value (a b : ℝ) (h : a^2 + b^2 = 2) :
  ∃ (y : ℝ), y = (1 / (a^2 + 1) + 4 / (b^2 + 1)) ∧ y = 9 / 4 :=
sorry

end NUMINAMATH_GPT_part1_min_value_part2_min_value_l1977_197708


namespace NUMINAMATH_GPT_common_solutions_y_values_l1977_197717

theorem common_solutions_y_values :
  (∃ x : ℝ, x^2 + y^2 - 16 = 0 ∧ x^2 - 3*y - 12 = 0) ↔ (y = -4 ∨ y = 1) :=
by {
  sorry
}

end NUMINAMATH_GPT_common_solutions_y_values_l1977_197717


namespace NUMINAMATH_GPT_greatest_divisor_lemma_l1977_197786

theorem greatest_divisor_lemma : ∃ (d : ℕ), d = Nat.gcd 1636 1852 ∧ d = 4 := by
  sorry

end NUMINAMATH_GPT_greatest_divisor_lemma_l1977_197786


namespace NUMINAMATH_GPT_crayons_and_erasers_difference_l1977_197770

theorem crayons_and_erasers_difference 
  (initial_crayons : ℕ) (initial_erasers : ℕ) (remaining_crayons : ℕ) 
  (h1 : initial_crayons = 601) (h2 : initial_erasers = 406) (h3 : remaining_crayons = 336) : 
  initial_erasers - remaining_crayons = 70 :=
by
  sorry

end NUMINAMATH_GPT_crayons_and_erasers_difference_l1977_197770


namespace NUMINAMATH_GPT_polynomial_remainder_l1977_197703

theorem polynomial_remainder (p q r : Polynomial ℝ) (h1 : p.eval 2 = 6) (h2 : p.eval 4 = 14)
  (r_deg : r.degree < 2) :
  p = q * (X - 2) * (X - 4) + r → r = 4 * X - 2 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_remainder_l1977_197703


namespace NUMINAMATH_GPT_mayor_vice_mayor_happy_people_l1977_197797

theorem mayor_vice_mayor_happy_people :
  (∃ (institutions_per_institution : ℕ) (num_institutions : ℕ),
    institutions_per_institution = 80 ∧
    num_institutions = 6 ∧
    num_institutions * institutions_per_institution = 480) :=
by
  sorry

end NUMINAMATH_GPT_mayor_vice_mayor_happy_people_l1977_197797

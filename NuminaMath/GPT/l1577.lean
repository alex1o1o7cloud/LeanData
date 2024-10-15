import Mathlib

namespace NUMINAMATH_GPT_parity_equivalence_l1577_157794

def p_q_parity_condition (p q : ℕ) : Prop :=
  (p^3 - q^3) % 2 = 0 ↔ (p + q) % 2 = 0

theorem parity_equivalence (p q : ℕ) : p_q_parity_condition p q :=
by sorry

end NUMINAMATH_GPT_parity_equivalence_l1577_157794


namespace NUMINAMATH_GPT_perpendicular_line_sufficient_condition_l1577_157750

theorem perpendicular_line_sufficient_condition (a : ℝ) :
  (-a) * ((a + 2) / 3) = -1 ↔ (a = -3 ∨ a = 1) :=
by {
  sorry
}

#print perpendicular_line_sufficient_condition

end NUMINAMATH_GPT_perpendicular_line_sufficient_condition_l1577_157750


namespace NUMINAMATH_GPT_arithmetic_series_sum_correct_l1577_157790

-- Define the parameters of the arithmetic series
def a : ℤ := -53
def l : ℤ := 3
def d : ℤ := 2

-- Define the number of terms in the series
def n : ℕ := 29

-- The expected sum of the series
def expected_sum : ℤ := -725

-- Define the nth term formula
noncomputable def nth_term (a d : ℤ) (n : ℕ) : ℤ :=
  a + (n - 1) * d

-- Define the sum of the arithmetic series
noncomputable def arithmetic_series_sum (a l : ℤ) (n : ℕ) : ℤ :=
  (n * (a + l)) / 2

-- Statement of the proof problem
theorem arithmetic_series_sum_correct :
  arithmetic_series_sum a l n = expected_sum := by
  sorry

end NUMINAMATH_GPT_arithmetic_series_sum_correct_l1577_157790


namespace NUMINAMATH_GPT_books_shelved_in_fiction_section_l1577_157764

def calculate_books_shelved_in_fiction_section (total_books : ℕ) (remaining_books : ℕ) (books_shelved_in_history : ℕ) (books_shelved_in_children : ℕ) (books_added_back : ℕ) : ℕ :=
  let total_shelved := total_books - remaining_books
  let adjusted_books_shelved_in_children := books_shelved_in_children - books_added_back
  let total_shelved_in_history_and_children := books_shelved_in_history + adjusted_books_shelved_in_children
  total_shelved - total_shelved_in_history_and_children

theorem books_shelved_in_fiction_section:
  calculate_books_shelved_in_fiction_section 51 16 12 8 4 = 19 :=
by 
  -- Definition of the function gives the output directly so proof is trivial.
  rfl

end NUMINAMATH_GPT_books_shelved_in_fiction_section_l1577_157764


namespace NUMINAMATH_GPT_min_value_of_f_l1577_157773

noncomputable def f (x : ℝ) : ℝ := (4 * x^2 - 2 * x + 16) / (2 * x - 1)

theorem min_value_of_f :
  ∃ x : ℝ, x ≥ 1 ∧ f x = 9 ∧ (∀ y : ℝ, y ≥ 1 → f y ≥ 9) :=
by { sorry }

end NUMINAMATH_GPT_min_value_of_f_l1577_157773


namespace NUMINAMATH_GPT_growth_rate_inequality_l1577_157704

theorem growth_rate_inequality (a b x : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) (h_x_pos : x > 0) :
  x ≤ (a + b) / 2 :=
sorry

end NUMINAMATH_GPT_growth_rate_inequality_l1577_157704


namespace NUMINAMATH_GPT_product_of_possible_values_of_x_l1577_157701

theorem product_of_possible_values_of_x : 
  (∀ x, |x - 7| - 5 = 4 → x = 16 ∨ x = -2) -> (16 * -2 = -32) :=
by
  intro h
  have := h 16
  have := h (-2)
  sorry

end NUMINAMATH_GPT_product_of_possible_values_of_x_l1577_157701


namespace NUMINAMATH_GPT_remainder_of_13_plus_x_mod_29_l1577_157759

theorem remainder_of_13_plus_x_mod_29
  (x : ℕ)
  (hx : 8 * x ≡ 1 [MOD 29])
  (hp : 0 < x) : 
  (13 + x) % 29 = 18 :=
sorry

end NUMINAMATH_GPT_remainder_of_13_plus_x_mod_29_l1577_157759


namespace NUMINAMATH_GPT_total_plums_l1577_157772

def alyssa_plums : Nat := 17
def jason_plums : Nat := 10

theorem total_plums : alyssa_plums + jason_plums = 27 := 
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_total_plums_l1577_157772


namespace NUMINAMATH_GPT_smallest_positive_integer_a_l1577_157757

noncomputable def isPerfectSquare (x : ℕ) : Prop :=
  ∃ (k : ℕ), k * k = x

theorem smallest_positive_integer_a :
  ∃ (a : ℕ), 0 < a ∧ (isPerfectSquare (10 + a)) ∧ (isPerfectSquare (10 * a)) ∧ 
  ∀ b : ℕ, 0 < b ∧ (isPerfectSquare (10 + b)) ∧ (isPerfectSquare (10 * b)) → a ≤ b :=
sorry

end NUMINAMATH_GPT_smallest_positive_integer_a_l1577_157757


namespace NUMINAMATH_GPT_units_digit_of_expression_l1577_157746

noncomputable def units_digit (n : ℕ) : ℕ :=
  n % 10

def expr : ℕ := 2 * (1 + 3 + 3^2 + 3^3 + 3^4 + 3^5 + 3^6 + 3^7 + 3^8 + 3^9)

theorem units_digit_of_expression : units_digit expr = 6 :=
by
  sorry

end NUMINAMATH_GPT_units_digit_of_expression_l1577_157746


namespace NUMINAMATH_GPT_flight_relation_not_preserved_l1577_157745

noncomputable def swap_city_flights (cities : Finset ℕ) (flights : ℕ → ℕ → Bool) : Prop := sorry

theorem flight_relation_not_preserved (cities : Finset ℕ) (flights : ℕ → ℕ → Bool) (M N : ℕ) (hM : M ∈ cities) (hN : N ∈ cities) : 
  ¬ swap_city_flights cities flights :=
sorry

end NUMINAMATH_GPT_flight_relation_not_preserved_l1577_157745


namespace NUMINAMATH_GPT_second_smallest_N_prevent_Bananastasia_win_l1577_157719

-- Definition of the set S, as positive integers not divisible by any p^4.
def S : Set ℕ := {n | ∀ p : ℕ, Prime p → ¬ (p ^ 4 ∣ n)}

-- Definition of the game rules and the condition for Anastasia to prevent Bananastasia from winning.
-- N is a value such that for all a in S, it is not possible for Bananastasia to directly win.

theorem second_smallest_N_prevent_Bananastasia_win :
  ∃ N : ℕ, N = 625 ∧ (∀ a ∈ S, N - a ≠ 0 ∧ N - a ≠ 1) :=
by
  sorry

end NUMINAMATH_GPT_second_smallest_N_prevent_Bananastasia_win_l1577_157719


namespace NUMINAMATH_GPT_GCD_is_six_l1577_157769

-- Define the numbers
def a : ℕ := 36
def b : ℕ := 60
def c : ℕ := 90

-- Define the GCD using Lean's gcd function
def GCD_abc : ℕ := Nat.gcd (Nat.gcd a b) c

-- State the theorem that GCD of 36, 60, and 90 is 6
theorem GCD_is_six : GCD_abc = 6 := by
  sorry -- Proof skipped

end NUMINAMATH_GPT_GCD_is_six_l1577_157769


namespace NUMINAMATH_GPT_larger_number_is_588_l1577_157742

theorem larger_number_is_588
  (A B hcf : ℕ)
  (lcm_factors : ℕ × ℕ)
  (hcf_condition : hcf = 42)
  (lcm_factors_condition : lcm_factors = (12, 14))
  (hcf_prop : Nat.gcd A B = hcf)
  (lcm_prop : Nat.lcm A B = hcf * lcm_factors.1 * lcm_factors.2) :
  max (A) (B) = 588 :=
by
  sorry

end NUMINAMATH_GPT_larger_number_is_588_l1577_157742


namespace NUMINAMATH_GPT_problem_solution_l1577_157740

variables {m n : ℝ}

theorem problem_solution (h1 : m^2 - n^2 = m * n) (h2 : m ≠ 0) (h3 : n ≠ 0) :
  (n / m) - (m / n) = -1 :=
sorry

end NUMINAMATH_GPT_problem_solution_l1577_157740


namespace NUMINAMATH_GPT_problem1_problem2_l1577_157725

variables (x y : ℝ)

-- Given Conditions
def given_conditions :=
  (x = 2 + Real.sqrt 3) ∧ (y = 2 - Real.sqrt 3)

-- Problem 1
theorem problem1 (h : given_conditions x y) : x^2 + y^2 = 14 :=
sorry

-- Problem 2
theorem problem2 (h : given_conditions x y) : (x / y) - (y / x) = 8 * Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l1577_157725


namespace NUMINAMATH_GPT_kelly_baking_powder_difference_l1577_157751

theorem kelly_baking_powder_difference :
  let amount_yesterday := 0.4
  let amount_now := 0.3
  amount_yesterday - amount_now = 0.1 :=
by
  -- Definitions for amounts 
  let amount_yesterday := 0.4
  let amount_now := 0.3
  
  -- Applying definitions in the computation
  show amount_yesterday - amount_now = 0.1
  sorry

end NUMINAMATH_GPT_kelly_baking_powder_difference_l1577_157751


namespace NUMINAMATH_GPT_sales_discount_percentage_l1577_157780

theorem sales_discount_percentage :
  ∀ (P N : ℝ) (D : ℝ),
  (N * 1.12 * (P * (1 - D / 100)) = P * N * (1 + 0.008)) → D = 10 :=
by
  intros P N D h
  sorry

end NUMINAMATH_GPT_sales_discount_percentage_l1577_157780


namespace NUMINAMATH_GPT_garden_width_min_5_l1577_157705

theorem garden_width_min_5 (width length : ℝ) (h_length : length = width + 20) (h_area : width * length ≥ 150) :
  width ≥ 5 :=
sorry

end NUMINAMATH_GPT_garden_width_min_5_l1577_157705


namespace NUMINAMATH_GPT_polynomial_quotient_l1577_157727

open Polynomial

noncomputable def dividend : ℤ[X] := 5 * X^4 - 9 * X^3 + 3 * X^2 + 7 * X - 6
noncomputable def divisor : ℤ[X] := X - 1

theorem polynomial_quotient :
  dividend /ₘ divisor = 5 * X^3 - 4 * X^2 + 7 * X + 7 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_quotient_l1577_157727


namespace NUMINAMATH_GPT_chiming_time_is_5_l1577_157708

-- Define the conditions for the clocks
def queen_strikes (h : ℕ) : Prop := (2 * h) % 3 = 0
def king_strikes (h : ℕ) : Prop := (3 * h) % 2 = 0

-- Define the chiming synchronization at the same time condition
def chiming_synchronization (h: ℕ) : Prop :=
  3 * h = 2 * ((2 * h) + 2)

-- The proof statement
theorem chiming_time_is_5 : ∃ h: ℕ, queen_strikes h ∧ king_strikes h ∧ chiming_synchronization h ∧ h = 5 :=
by
  sorry

end NUMINAMATH_GPT_chiming_time_is_5_l1577_157708


namespace NUMINAMATH_GPT_rectangle_length_width_l1577_157753

theorem rectangle_length_width (x y : ℝ) (h1 : 2 * (x + y) = 26) (h2 : x * y = 42) : 
  (x = 7 ∧ y = 6) ∨ (x = 6 ∧ y = 7) :=
by
  sorry

end NUMINAMATH_GPT_rectangle_length_width_l1577_157753


namespace NUMINAMATH_GPT_always_possible_to_rotate_disks_l1577_157792

def labels_are_distinct (a : Fin 20 → ℕ) : Prop :=
  ∀ i j : Fin 20, i ≠ j → a i ≠ a j

def opposite_position (i : Fin 20) (r : Fin 20) : Fin 20 :=
  (i + r) % 20

def no_identical_numbers_opposite (a b : Fin 20 → ℕ) (r : Fin 20) : Prop :=
  ∀ i : Fin 20, a i ≠ b (opposite_position i r)

theorem always_possible_to_rotate_disks (a b : Fin 20 → ℕ) :
  labels_are_distinct a →
  labels_are_distinct b →
  ∃ r : Fin 20, no_identical_numbers_opposite a b r :=
sorry

end NUMINAMATH_GPT_always_possible_to_rotate_disks_l1577_157792


namespace NUMINAMATH_GPT_circle_equation_l1577_157796

theorem circle_equation :
  ∃ r : ℝ, ∀ x y : ℝ,
  ((x - 2) * (x - 2) + (y - 1) * (y - 1) = r * r) ∧
  ((5 - 2) * (5 - 2) + (-2 - 1) * (-2 - 1) = r * r) ∧
  (5 + 2 * -2 - 5 + r * r = 0) :=
sorry

end NUMINAMATH_GPT_circle_equation_l1577_157796


namespace NUMINAMATH_GPT_max_g6_l1577_157721

noncomputable def g (x : ℝ) : ℝ :=
sorry

theorem max_g6 :
  (∀ x, (g x = a * x^2 + b * x + c) ∧ (a ≥ 0) ∧ (b ≥ 0) ∧ (c ≥ 0)) →
  (g 3 = 3) →
  (g 9 = 243) →
  (g 6 ≤ 6) :=
sorry

end NUMINAMATH_GPT_max_g6_l1577_157721


namespace NUMINAMATH_GPT_profitable_year_exists_option2_more_economical_l1577_157783

noncomputable def total_expenses (x : ℕ) : ℝ := 2 * (x:ℝ)^2 + 10 * x  

noncomputable def annual_income (x : ℕ) : ℝ := 50 * x  

def year_profitable (x : ℕ) : Prop := annual_income x > total_expenses x + 98 / 1000

theorem profitable_year_exists : ∃ x : ℕ, year_profitable x ∧ x = 3 := sorry

noncomputable def total_profit (x : ℕ) : ℝ := 
  50 * x - 2 * (x:ℝ)^2 + 10 * x - 98 / 1000 + if x = 10 then 8 else if x = 7 then 26 else 0

theorem option2_more_economical : 
  total_profit 10 = 110 ∧ total_profit 7 = 110 ∧ 7 < 10 :=
sorry

end NUMINAMATH_GPT_profitable_year_exists_option2_more_economical_l1577_157783


namespace NUMINAMATH_GPT_common_ratio_geometric_sequence_l1577_157743

variable (a_n : ℕ → ℝ) (a1 : ℝ) (d : ℝ)

noncomputable def is_arithmetic_sequence (a_n : ℕ → ℝ) (a1 : ℝ) (d : ℝ) : Prop :=
∀ n : ℕ, a_n n = a1 + n * d

noncomputable def forms_geometric_sequence (a_n : ℕ → ℝ) : Prop :=
(a_n 4) / (a_n 0) = (a_n 16) / (a_n 4)

theorem common_ratio_geometric_sequence :
  d ≠ 0 → 
  forms_geometric_sequence (a_n : ℕ → ℝ) →
  is_arithmetic_sequence a_n a1 d →
  ((a_n 4) / (a1) = 9) :=
by
  sorry

end NUMINAMATH_GPT_common_ratio_geometric_sequence_l1577_157743


namespace NUMINAMATH_GPT_julian_younger_than_frederick_by_20_l1577_157726

noncomputable def Kyle: ℕ := 25
noncomputable def Tyson: ℕ := 20
noncomputable def Julian : ℕ := Kyle - 5
noncomputable def Frederick : ℕ := 2 * Tyson

theorem julian_younger_than_frederick_by_20 : Frederick - Julian = 20 :=
by
  sorry

end NUMINAMATH_GPT_julian_younger_than_frederick_by_20_l1577_157726


namespace NUMINAMATH_GPT_initial_number_of_men_l1577_157765

theorem initial_number_of_men (M : ℕ) (h1 : ∃ food : ℕ, food = M * 22) (h2 : ∀ food, food = (M * 20)) (h3 : ∃ food : ℕ, food = ((M + 40) * 19)) : M = 760 := by
  sorry

end NUMINAMATH_GPT_initial_number_of_men_l1577_157765


namespace NUMINAMATH_GPT_dog_weight_ratio_l1577_157761

theorem dog_weight_ratio
  (w7 : ℕ) (r : ℕ) (w13 : ℕ) (w21 : ℕ) (w52 : ℕ):
  (w7 = 6) →
  (w13 = 12 * r) →
  (w21 = 2 * w13) →
  (w52 = w21 + 30) →
  (w52 = 78) →
  r = 2 :=
by 
  sorry

end NUMINAMATH_GPT_dog_weight_ratio_l1577_157761


namespace NUMINAMATH_GPT_sin_cos_bounds_l1577_157767

theorem sin_cos_bounds (w x y z : ℝ)
  (hw : -Real.pi / 2 ≤ w ∧ w ≤ Real.pi / 2)
  (hx : -Real.pi / 2 ≤ x ∧ x ≤ Real.pi / 2)
  (hy : -Real.pi / 2 ≤ y ∧ y ≤ Real.pi / 2)
  (hz : -Real.pi / 2 ≤ z ∧ z ≤ Real.pi / 2)
  (h₁ : Real.sin w + Real.sin x + Real.sin y + Real.sin z = 1)
  (h₂ : Real.cos (2 * w) + Real.cos (2 * x) + Real.cos (2 * y) + Real.cos (2 * z) ≥ 10 / 3) :
  0 ≤ w ∧ w ≤ Real.pi / 6 ∧ 0 ≤ x ∧ x ≤ Real.pi / 6 ∧ 0 ≤ y ∧ y ≤ Real.pi / 6 ∧ 0 ≤ z ∧ z ≤ Real.pi / 6 :=
by
  sorry

end NUMINAMATH_GPT_sin_cos_bounds_l1577_157767


namespace NUMINAMATH_GPT_sample_size_drawn_l1577_157748

theorem sample_size_drawn (sample_size : ℕ) (probability : ℚ) (N : ℚ) 
  (h1 : sample_size = 30) 
  (h2 : probability = 0.25) 
  (h3 : probability = sample_size / N) : 
  N = 120 := by
  sorry

end NUMINAMATH_GPT_sample_size_drawn_l1577_157748


namespace NUMINAMATH_GPT_find_length_AD_l1577_157776

-- Given data and conditions
def triangle_ABC (A B C D : Type) : Prop := sorry
def angle_bisector_AD (A B C D : Type) : Prop := sorry
def length_BD : ℝ := 40
def length_BC : ℝ := 45
def length_AC : ℝ := 36

-- Prove that AD = 320 units
theorem find_length_AD (A B C D : Type)
  (h1 : triangle_ABC A B C D)
  (h2 : angle_bisector_AD A B C D)
  (h3 : length_BD = 40)
  (h4 : length_BC = 45)
  (h5 : length_AC = 36) :
  ∃ x : ℝ, x = 320 :=
sorry

end NUMINAMATH_GPT_find_length_AD_l1577_157776


namespace NUMINAMATH_GPT_john_pre_lunch_drive_l1577_157744

def drive_before_lunch (h : ℕ) : Prop :=
  45 * h + 45 * 3 = 225

theorem john_pre_lunch_drive : ∃ h : ℕ, drive_before_lunch h ∧ h = 2 :=
by
  sorry

end NUMINAMATH_GPT_john_pre_lunch_drive_l1577_157744


namespace NUMINAMATH_GPT_zoe_pictures_l1577_157730

theorem zoe_pictures (pictures_taken : ℕ) (dolphin_show_pictures : ℕ)
  (h1 : pictures_taken = 28) (h2 : dolphin_show_pictures = 16) :
  pictures_taken + dolphin_show_pictures = 44 :=
sorry

end NUMINAMATH_GPT_zoe_pictures_l1577_157730


namespace NUMINAMATH_GPT_rectangle_diagonal_length_l1577_157755

theorem rectangle_diagonal_length (l : ℝ) (L W d : ℝ) 
  (h_ratio : L = 5 * l ∧ W = 2 * l)
  (h_perimeter : 2 * (L + W) = 100) :
  d = (5 * Real.sqrt 290) / 7 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_diagonal_length_l1577_157755


namespace NUMINAMATH_GPT_fraction_zero_solution_l1577_157777

theorem fraction_zero_solution (x : ℝ) (h : (x - 1) / (2 - x) = 0) : x = 1 :=
sorry

end NUMINAMATH_GPT_fraction_zero_solution_l1577_157777


namespace NUMINAMATH_GPT_base_729_base8_l1577_157717

theorem base_729_base8 (b : ℕ) (X Y : ℕ) (h_distinct : X ≠ Y)
  (h_range : b^3 ≤ 729 ∧ 729 < b^4)
  (h_form : 729 = X * b^3 + Y * b^2 + X * b + Y) : b = 8 :=
sorry

end NUMINAMATH_GPT_base_729_base8_l1577_157717


namespace NUMINAMATH_GPT_trigonometric_identity_l1577_157731

theorem trigonometric_identity (α : Real) (h : Real.tan (α / 2) = 4) :
    (6 * Real.sin α - 7 * Real.cos α + 1) / (8 * Real.sin α + 9 * Real.cos α - 1) = -85 / 44 := by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l1577_157731


namespace NUMINAMATH_GPT_total_games_l1577_157784

variable (Ken_games Dave_games Jerry_games : ℕ)

-- The conditions from the problem.
def condition1 : Prop := Ken_games = Dave_games + 5
def condition2 : Prop := Dave_games = Jerry_games + 3
def condition3 : Prop := Jerry_games = 7

-- The final statement to prove
theorem total_games (h1 : condition1 Ken_games Dave_games) 
                    (h2 : condition2 Dave_games Jerry_games) 
                    (h3 : condition3 Jerry_games) : 
  Ken_games + Dave_games + Jerry_games = 32 :=
by
  sorry

end NUMINAMATH_GPT_total_games_l1577_157784


namespace NUMINAMATH_GPT_minimum_groups_l1577_157709

theorem minimum_groups (students : ℕ) (max_group_size : ℕ) (h_students : students = 30) (h_max_group_size : max_group_size = 12) : 
  ∃ least_groups : ℕ, least_groups = 3 :=
by
  sorry

end NUMINAMATH_GPT_minimum_groups_l1577_157709


namespace NUMINAMATH_GPT_prob_at_least_one_solves_l1577_157711

theorem prob_at_least_one_solves (p1 p2 : ℝ) (h1 : 0 ≤ p1) (h2 : p1 ≤ 1) (h3 : 0 ≤ p2) (h4 : p2 ≤ 1) :
  (1 : ℝ) - (1 - p1) * (1 - p2) = 1 - ((1 - p1) * (1 - p2)) :=
by sorry

end NUMINAMATH_GPT_prob_at_least_one_solves_l1577_157711


namespace NUMINAMATH_GPT_sum_of_roots_l1577_157762

theorem sum_of_roots (x : ℝ) (h : x^2 = 10 * x + 16) : x = 10 :=
by 
  -- Rearrange the equation to standard form: x^2 - 10x - 16 = 0
  have eqn : x^2 - 10 * x - 16 = 0 := by sorry
  -- Use the formula for the sum of the roots of a quadratic equation
  -- Prove the sum of the roots is 10
  sorry

end NUMINAMATH_GPT_sum_of_roots_l1577_157762


namespace NUMINAMATH_GPT_furniture_store_revenue_increase_l1577_157714

noncomputable def percentage_increase_in_gross (P R : ℕ) : ℚ :=
  ((0.80 * P) * (1.70 * R) - (P * R)) / (P * R) * 100

theorem furniture_store_revenue_increase (P R : ℕ) :
  percentage_increase_in_gross P R = 36 := 
by
  -- We include the conditions directly in the proof.
  -- Follow theorem from the given solution.
  sorry

end NUMINAMATH_GPT_furniture_store_revenue_increase_l1577_157714


namespace NUMINAMATH_GPT_largest_number_l1577_157785

theorem largest_number (A B C D E : ℝ) (hA : A = 0.998) (hB : B = 0.9899) (hC : C = 0.9) (hD : D = 0.9989) (hE : E = 0.8999) :
  D > A ∧ D > B ∧ D > C ∧ D > E :=
by
  sorry

end NUMINAMATH_GPT_largest_number_l1577_157785


namespace NUMINAMATH_GPT_find_number_l1577_157728

-- Define the given conditions and statement as Lean types
theorem find_number (x : ℝ) :
  (0.3 * x > 0.6 * 50 + 30) -> x = 200 :=
by
  -- Proof here
  sorry

end NUMINAMATH_GPT_find_number_l1577_157728


namespace NUMINAMATH_GPT_polynomial_satisfies_conditions_l1577_157720

noncomputable def f (x y z : ℝ) : ℝ := (x^2 - y^3) * (y^3 - z^6) * (z^6 - x^2)

theorem polynomial_satisfies_conditions :
  (∀ x y z : ℝ, f x (z^2) y + f x (y^2) z = 0) ∧ 
  (∀ x y z : ℝ, f (z^3) y x + f (x^3) y z = 0) :=
by
  sorry

end NUMINAMATH_GPT_polynomial_satisfies_conditions_l1577_157720


namespace NUMINAMATH_GPT_apples_to_grapes_equivalent_l1577_157713

-- Definitions based on the problem conditions
def apples := ℝ
def grapes := ℝ

-- Given conditions
def given_condition : Prop := (3 / 4) * 12 = 9

-- Question to prove
def question : Prop := (1 / 2) * 6 = 3

-- The theorem statement combining given conditions to prove the question
theorem apples_to_grapes_equivalent : given_condition → question := 
by
    intros
    sorry

end NUMINAMATH_GPT_apples_to_grapes_equivalent_l1577_157713


namespace NUMINAMATH_GPT_maximum_ratio_is_2_plus_2_sqrt2_l1577_157793

noncomputable def C1_polar_eq (θ : ℝ) : Prop :=
  ∀ ρ : ℝ, ρ * (Real.cos θ + Real.sin θ) = 1

noncomputable def C2_polar_eq (θ : ℝ) : Prop :=
  ∀ ρ : ℝ, ρ = 4 * Real.cos θ

theorem maximum_ratio_is_2_plus_2_sqrt2 (α : ℝ) (hα : 0 ≤ α ∧ α ≤ Real.pi / 2) :
  ∃ ρA ρB : ℝ, (ρA = 1 / (Real.cos α + Real.sin α)) ∧ (ρB = 4 * Real.cos α) ∧ 
  (4 * Real.cos α * (Real.cos α + Real.sin α) = 2 + 2 * Real.sqrt 2) :=
sorry

end NUMINAMATH_GPT_maximum_ratio_is_2_plus_2_sqrt2_l1577_157793


namespace NUMINAMATH_GPT_problem_part1_problem_part2_l1577_157766

open Complex

noncomputable def E1 := ((1 + I)^2 / (1 + 2 * I)) + ((1 - I)^2 / (2 - I))

theorem problem_part1 : E1 = (6 / 5) - (2 / 5) * I :=
by
  sorry

theorem problem_part2 (x y : ℝ) (h1 : (x / 2) + (y / 5) = 1) (h2 : (x / 2) + (2 * y / 5) = 3) : x = -2 ∧ y = 10 :=
by
  sorry

end NUMINAMATH_GPT_problem_part1_problem_part2_l1577_157766


namespace NUMINAMATH_GPT_original_rent_of_increased_friend_l1577_157788

theorem original_rent_of_increased_friend (avg_rent : ℝ) (new_avg_rent : ℝ) (num_friends : ℝ) (rent_increase_pct : ℝ)
  (total_old_rent : ℝ) (total_new_rent : ℝ) (increase_amount : ℝ) (R : ℝ) :
  avg_rent = 800 ∧ new_avg_rent = 850 ∧ num_friends = 4 ∧ rent_increase_pct = 0.16 ∧
  total_old_rent = num_friends * avg_rent ∧ total_new_rent = num_friends * new_avg_rent ∧
  increase_amount = total_new_rent - total_old_rent ∧ increase_amount = rent_increase_pct * R →
  R = 1250 :=
by
  sorry

end NUMINAMATH_GPT_original_rent_of_increased_friend_l1577_157788


namespace NUMINAMATH_GPT_determine_d_l1577_157779

theorem determine_d (a b c d : ℝ) (h : a^2 + b^2 + c^2 + 2 = d + (a + b + c - d)^(1/3)) : d = 1/2 := by
  sorry

end NUMINAMATH_GPT_determine_d_l1577_157779


namespace NUMINAMATH_GPT_ball_radius_l1577_157729

noncomputable def radius_of_ball (d h : ℝ) : ℝ :=
  let r := d / 2
  (325 / 20 : ℝ)

theorem ball_radius (d h : ℝ) (hd : d = 30) (hh : h = 10) :
  radius_of_ball d h = 16.25 := by
  sorry

end NUMINAMATH_GPT_ball_radius_l1577_157729


namespace NUMINAMATH_GPT_theoretical_yield_H2SO4_l1577_157700

-- Define the theoretical yield calculation problem in terms of moles of reactions and products
theorem theoretical_yield_H2SO4 
  (moles_SO3 : ℝ) (moles_H2O : ℝ) 
  (reaction : moles_SO3 + moles_H2O = 2.0 + 1.5) 
  (limiting_reactant_H2O : moles_H2O = 1.5) : 
  1.5 = moles_H2O * 1 :=
  sorry

end NUMINAMATH_GPT_theoretical_yield_H2SO4_l1577_157700


namespace NUMINAMATH_GPT_fraction_division_l1577_157738

theorem fraction_division :
  (5 : ℚ) / ((13 : ℚ) / 7) = 35 / 13 :=
by
  sorry

end NUMINAMATH_GPT_fraction_division_l1577_157738


namespace NUMINAMATH_GPT_smallest_k_l1577_157741

-- Define the non-decreasing property of digits in a five-digit number
def non_decreasing (n : Fin 5 → ℕ) : Prop :=
  n 0 ≤ n 1 ∧ n 1 ≤ n 2 ∧ n 2 ≤ n 3 ∧ n 3 ≤ n 4

-- Define the overlap property in at least one digit
def overlap (n1 n2 : Fin 5 → ℕ) : Prop :=
  ∃ i : Fin 5, n1 i = n2 i

-- The main theorem stating the problem
theorem smallest_k {N1 Nk : Fin 5 → ℕ} :
  (∀ n : Fin 5 → ℕ, non_decreasing n → overlap N1 n ∨ overlap Nk n) → 
  ∃ (k : Nat), k = 2 :=
sorry

end NUMINAMATH_GPT_smallest_k_l1577_157741


namespace NUMINAMATH_GPT_nina_weeks_to_afford_game_l1577_157786

noncomputable def game_cost : ℝ := 50
noncomputable def sales_tax_rate : ℝ := 0.10
noncomputable def weekly_allowance : ℝ := 10
noncomputable def saving_rate : ℝ := 0.5

noncomputable def total_cost : ℝ := game_cost + (game_cost * sales_tax_rate)
noncomputable def savings_per_week : ℝ := weekly_allowance * saving_rate
noncomputable def weeks_needed : ℝ := total_cost / savings_per_week

theorem nina_weeks_to_afford_game : weeks_needed = 11 := by
  sorry

end NUMINAMATH_GPT_nina_weeks_to_afford_game_l1577_157786


namespace NUMINAMATH_GPT_loan_amounts_l1577_157782

theorem loan_amounts (x y : ℝ) (h1 : x + y = 50) (h2 : 0.1 * x + 0.08 * y = 4.4) : x = 20 ∧ y = 30 := by
  sorry

end NUMINAMATH_GPT_loan_amounts_l1577_157782


namespace NUMINAMATH_GPT_angela_age_in_5_years_l1577_157722

-- Define the variables representing Angela and Beth's ages.
variable (A B : ℕ)

-- State the conditions as hypotheses.
def condition_1 : Prop := A = 4 * B
def condition_2 : Prop := (A - 5) + (B - 5) = 45

-- State the final proposition that Angela will be 49 years old in five years.
theorem angela_age_in_5_years (h1 : condition_1 A B) (h2 : condition_2 A B) : A + 5 = 49 := by
  sorry

end NUMINAMATH_GPT_angela_age_in_5_years_l1577_157722


namespace NUMINAMATH_GPT_p_true_of_and_not_p_false_l1577_157735

variable {p q : Prop}

theorem p_true_of_and_not_p_false (h1 : ¬(p ∧ q)) (h2 : ¬¬p) : p :=
sorry

end NUMINAMATH_GPT_p_true_of_and_not_p_false_l1577_157735


namespace NUMINAMATH_GPT_numbering_tube_contacts_l1577_157734

theorem numbering_tube_contacts {n : ℕ} (hn : n = 7) :
  ∃ (f g : ℕ → ℕ), (∀ k : ℕ, f k = k % n) ∧ (∀ k : ℕ, g k = (n - k) % n) ∧ 
  (∀ m : ℕ, ∃ k : ℕ, f (k + m) % n = g k % n) :=
by
  sorry

end NUMINAMATH_GPT_numbering_tube_contacts_l1577_157734


namespace NUMINAMATH_GPT_B_current_age_l1577_157756

theorem B_current_age (A B : ℕ) (h1 : A = B + 15) (h2 : A - 5 = 2 * (B - 5)) : B = 20 :=
by sorry

end NUMINAMATH_GPT_B_current_age_l1577_157756


namespace NUMINAMATH_GPT_prove_sum_l1577_157747

theorem prove_sum (a b : ℝ) (h1 : a * (a - 4) = 12) (h2 : b * (b - 4) = 12) (h3 : a ≠ b) : a + b = 4 := by
  sorry

end NUMINAMATH_GPT_prove_sum_l1577_157747


namespace NUMINAMATH_GPT_toads_max_l1577_157798

theorem toads_max (n : ℕ) (h₁ : n ≥ 3) : 
  ∃ k : ℕ, k = ⌈ (n : ℝ) / 2 ⌉ ∧ ∀ (labels : Fin n → Fin n) (jumps : Fin n → ℕ), 
  (∀ i, jumps (labels i) = labels i) → ¬ ∃ f : Fin k → Fin n, ∀ i₁ i₂, i₁ ≠ i₂ → f i₁ ≠ f i₂ :=
sorry

end NUMINAMATH_GPT_toads_max_l1577_157798


namespace NUMINAMATH_GPT_rational_solutions_k_l1577_157791

theorem rational_solutions_k (k : ℕ) (h : k > 0) : (∃ x : ℚ, 2 * (k : ℚ) * x^2 + 36 * x + 3 * (k : ℚ) = 0) → k = 6 :=
by
  -- proof to be written
  sorry

end NUMINAMATH_GPT_rational_solutions_k_l1577_157791


namespace NUMINAMATH_GPT_initial_percentage_decrease_l1577_157712

theorem initial_percentage_decrease (P x : ℝ) (h1 : 0 < P) (h2 : 0 ≤ x) (h3 : x ≤ 100) :
  ((P - (x / 100) * P) * 1.50 = P * 1.20) → x = 20 :=
by
  sorry

end NUMINAMATH_GPT_initial_percentage_decrease_l1577_157712


namespace NUMINAMATH_GPT_second_polygon_sides_l1577_157752

theorem second_polygon_sides (s : ℝ) (P : ℝ) (n : ℕ) : 
  (50 * 3 * s = P) ∧ (n * s = P) → n = 150 := 
by {
  sorry
}

end NUMINAMATH_GPT_second_polygon_sides_l1577_157752


namespace NUMINAMATH_GPT_green_face_probability_l1577_157733

def probability_of_green_face (total_faces green_faces : Nat) : ℚ :=
  green_faces / total_faces

theorem green_face_probability :
  let total_faces := 10
  let green_faces := 3
  let blue_faces := 5
  let red_faces := 2
  probability_of_green_face total_faces green_faces = 3/10 :=
by
  sorry

end NUMINAMATH_GPT_green_face_probability_l1577_157733


namespace NUMINAMATH_GPT_rectangle_circle_area_ratio_l1577_157716

noncomputable def area_ratio (w r : ℝ) (h : 3 * w = Real.pi * r) : ℝ :=
  (2 * w^2) / (Real.pi * r^2)

theorem rectangle_circle_area_ratio (w r : ℝ) (h : 3 * w = Real.pi * r) :
  area_ratio w r h = 18 / (Real.pi * Real.pi) :=
by
  sorry

end NUMINAMATH_GPT_rectangle_circle_area_ratio_l1577_157716


namespace NUMINAMATH_GPT_geometric_figure_area_l1577_157775

theorem geometric_figure_area :
  (∀ (z : ℂ),
     (0 < (z.re / 20)) ∧ ((z.re / 20) < 1) ∧ 
     (0 < (z.im / 20)) ∧ ((z.im / 20) < 1) ∧ 
     (0 < (20 / z.re)) ∧ ((20 / z.re) < 1) ∧ 
     (0 < (20 / z.im)) ∧ ((20 / z.im) < 1)) →
     (∃ (area : ℝ), area = 400 - 50 * Real.pi) :=
by
  sorry

end NUMINAMATH_GPT_geometric_figure_area_l1577_157775


namespace NUMINAMATH_GPT_sally_has_more_cards_l1577_157703

def SallyInitial : ℕ := 27
def DanTotal : ℕ := 41
def SallyBought : ℕ := 20
def SallyTotal := SallyInitial + SallyBought

theorem sally_has_more_cards : SallyTotal - DanTotal = 6 := by
  sorry

end NUMINAMATH_GPT_sally_has_more_cards_l1577_157703


namespace NUMINAMATH_GPT_min_distance_l1577_157718

noncomputable def f (x : ℝ) : ℝ := x^2
noncomputable def g (x : ℝ) : ℝ := Real.log x

theorem min_distance :
  ∃ m : ℝ, (∀ x > 0, x ≠ m → (f m - g m) ≤ (f x - g x)) ∧ m = Real.sqrt 2 / 2 :=
by
  sorry

end NUMINAMATH_GPT_min_distance_l1577_157718


namespace NUMINAMATH_GPT_transform_polynomial_l1577_157795

open Real

variable {x y : ℝ}

theorem transform_polynomial 
  (h1 : y = x + 1 / x) 
  (h2 : x^4 + x^3 - 4 * x^2 + x + 1 = 0) : 
  x^2 * (y^2 + y - 6) = 0 := 
sorry

end NUMINAMATH_GPT_transform_polynomial_l1577_157795


namespace NUMINAMATH_GPT_area_of_each_triangle_is_half_l1577_157754

structure Point where
  x : ℝ
  y : ℝ

structure Triangle where
  p1 : Point
  p2 : Point
  p3 : Point

def area (t : Triangle) : ℝ :=
  0.5 * |t.p1.x * (t.p2.y - t.p3.y) + t.p2.x * (t.p3.y - t.p1.y) + t.p3.x * (t.p1.y - t.p2.y)|

def A : Point := { x := 0, y := 0 }
def B : Point := { x := 1, y := 0 }
def C : Point := { x := 1, y := 1 }
def D : Point := { x := 0, y := 1 }
def K : Point := { x := 0.5, y := 1 }
def L : Point := { x := 0, y := 0.5 }
def M : Point := { x := 0.5, y := 0 }
def N : Point := { x := 1, y := 0.5 }

def AKB : Triangle := { p1 := A, p2 := K, p3 := B }
def BLC : Triangle := { p1 := B, p2 := L, p3 := C }
def CMD : Triangle := { p1 := C, p2 := M, p3 := D }
def DNA : Triangle := { p1 := D, p2 := N, p3 := A }

theorem area_of_each_triangle_is_half :
  area AKB = 0.5 ∧ area BLC = 0.5 ∧ area CMD = 0.5 ∧ area DNA = 0.5 := by sorry

end NUMINAMATH_GPT_area_of_each_triangle_is_half_l1577_157754


namespace NUMINAMATH_GPT_avg_growth_rate_proof_l1577_157715

noncomputable def avg_growth_rate_correct_eqn (x : ℝ) : Prop :=
  40 * (1 + x)^2 = 48.4

theorem avg_growth_rate_proof (x : ℝ) 
  (h1 : 40 = avg_working_hours_first_week)
  (h2 : 48.4 = avg_working_hours_third_week) :
  avg_growth_rate_correct_eqn x :=
by 
  sorry

/- Defining the known conditions -/
def avg_working_hours_first_week : ℝ := 40
def avg_working_hours_third_week : ℝ := 48.4

end NUMINAMATH_GPT_avg_growth_rate_proof_l1577_157715


namespace NUMINAMATH_GPT_monthly_rent_calculation_l1577_157789

noncomputable def monthly_rent (purchase_cost : ℕ) (maintenance_pct : ℝ) (annual_taxes : ℕ) (target_roi : ℝ) : ℝ :=
  let annual_return := target_roi * (purchase_cost : ℝ)
  let total_annual_requirement := annual_return + (annual_taxes : ℝ)
  let monthly_requirement := total_annual_requirement / 12
  let actual_rent := monthly_requirement / (1 - maintenance_pct)
  actual_rent

theorem monthly_rent_calculation :
  monthly_rent 12000 0.15 400 0.06 = 109.80 :=
by
  sorry

end NUMINAMATH_GPT_monthly_rent_calculation_l1577_157789


namespace NUMINAMATH_GPT_area_of_region_bounded_by_lines_and_y_axis_l1577_157758

noncomputable def area_of_triangle_bounded_by_lines : ℝ :=
  let y1 (x : ℝ) := 3 * x - 6
  let y2 (x : ℝ) := -2 * x + 18
  let intersection_x := 24 / 5
  let intersection_y := y1 intersection_x
  let base := 18 + 6
  let height := intersection_x
  1 / 2 * base * height

theorem area_of_region_bounded_by_lines_and_y_axis :
  area_of_triangle_bounded_by_lines = 57.6 :=
by
  sorry

end NUMINAMATH_GPT_area_of_region_bounded_by_lines_and_y_axis_l1577_157758


namespace NUMINAMATH_GPT_discount_percentage_correct_l1577_157737

-- Definitions corresponding to the conditions
def number_of_toys : ℕ := 5
def cost_per_toy : ℕ := 3
def total_price_paid : ℕ := 12
def original_price : ℕ := number_of_toys * cost_per_toy
def discount_amount : ℕ := original_price - total_price_paid
def discount_percentage : ℕ := (discount_amount * 100) / original_price

-- Statement of the problem
theorem discount_percentage_correct :
  discount_percentage = 20 := 
  sorry

end NUMINAMATH_GPT_discount_percentage_correct_l1577_157737


namespace NUMINAMATH_GPT_solution_to_inequality_system_l1577_157797

theorem solution_to_inequality_system :
  (∀ x : ℝ, 2 * (x - 1) < x + 2 → (x + 1) / 2 < x → 1 < x ∧ x < 4) :=
by
  intros x h1 h2
  sorry

end NUMINAMATH_GPT_solution_to_inequality_system_l1577_157797


namespace NUMINAMATH_GPT_trihedral_angle_plane_angles_acute_l1577_157787

open Real

-- Define what it means for an angle to be acute
def is_acute (θ : ℝ) : Prop :=
  0 < θ ∧ θ < π / 2

-- Define the given conditions
variable {A B C α β γ : ℝ}
variable (hA : is_acute A)
variable (hB : is_acute B)
variable (hC : is_acute C)

-- State the problem: if dihedral angles are acute, then plane angles are also acute
theorem trihedral_angle_plane_angles_acute :
  is_acute A → is_acute B → is_acute C → is_acute α ∧ is_acute β ∧ is_acute γ :=
sorry

end NUMINAMATH_GPT_trihedral_angle_plane_angles_acute_l1577_157787


namespace NUMINAMATH_GPT_cat_food_finished_on_sunday_l1577_157771

def cat_morning_consumption : ℚ := 1 / 2
def cat_evening_consumption : ℚ := 1 / 3
def total_food : ℚ := 10
def daily_consumption : ℚ := cat_morning_consumption + cat_evening_consumption
def days_to_finish_food (total_food daily_consumption : ℚ) : ℚ :=
  total_food / daily_consumption

theorem cat_food_finished_on_sunday :
  days_to_finish_food total_food daily_consumption = 7 := 
sorry

end NUMINAMATH_GPT_cat_food_finished_on_sunday_l1577_157771


namespace NUMINAMATH_GPT_victors_friend_decks_l1577_157739

theorem victors_friend_decks:
  ∀ (deck_cost : ℕ) (victor_decks : ℕ) (total_spent : ℕ)
  (friend_decks : ℕ),
  deck_cost = 8 →
  victor_decks = 6 →
  total_spent = 64 →
  (victor_decks * deck_cost + friend_decks * deck_cost = total_spent) →
  friend_decks = 2 :=
by
  intros deck_cost victor_decks total_spent friend_decks hc hv ht heq
  sorry

end NUMINAMATH_GPT_victors_friend_decks_l1577_157739


namespace NUMINAMATH_GPT_find_real_solutions_l1577_157710

theorem find_real_solutions (x : ℝ) :
  x^4 + (3 - x)^4 = 146 ↔ x = 1.5 + Real.sqrt 3.4175 ∨ x = 1.5 - Real.sqrt 3.4175 :=
by
  sorry

end NUMINAMATH_GPT_find_real_solutions_l1577_157710


namespace NUMINAMATH_GPT_intersection_A_B_l1577_157768

-- Define the sets A and B
def A : Set ℤ := {1, 3, 5, 7}
def B : Set ℤ := {x | 2 ≤ x ∧ x ≤ 5}

-- The goal is to prove that A ∩ B = {3, 5}
theorem intersection_A_B : A ∩ B = {3, 5} :=
  sorry

end NUMINAMATH_GPT_intersection_A_B_l1577_157768


namespace NUMINAMATH_GPT_total_cost_is_correct_l1577_157799

-- Define the price of pizzas
def pizza_price : ℕ := 5

-- Define the count of triple cheese and meat lovers pizzas
def triple_cheese_pizzas : ℕ := 10
def meat_lovers_pizzas : ℕ := 9

-- Define the special offers
def buy1get1free (count : ℕ) : ℕ := count / 2 + count % 2
def buy2get1free (count : ℕ) : ℕ := (count / 3) * 2 + count % 3

-- Define the cost calculations using the special offers
def cost_triple_cheese : ℕ := buy1get1free triple_cheese_pizzas * pizza_price
def cost_meat_lovers : ℕ := buy2get1free meat_lovers_pizzas * pizza_price

-- Define the total cost calculation
def total_cost : ℕ := cost_triple_cheese + cost_meat_lovers

-- The theorem we need to prove
theorem total_cost_is_correct :
  total_cost = 55 := by
  sorry

end NUMINAMATH_GPT_total_cost_is_correct_l1577_157799


namespace NUMINAMATH_GPT_vertex_of_quadratic_l1577_157781

theorem vertex_of_quadratic (x : ℝ) : 
  (y : ℝ) = -2 * (x + 1) ^ 2 + 3 →
  (∃ vertex_x vertex_y : ℝ, vertex_x = -1 ∧ vertex_y = 3 ∧ y = -2 * (vertex_x + 1) ^ 2 + vertex_y) :=
by
  intro h
  exists -1, 3
  simp [h]
  sorry

end NUMINAMATH_GPT_vertex_of_quadratic_l1577_157781


namespace NUMINAMATH_GPT_find_value_of_expression_l1577_157724

theorem find_value_of_expression (x y z : ℝ)
  (h1 : 12 * x - 9 * y^2 = 7)
  (h2 : 6 * y - 9 * z^2 = -2)
  (h3 : 12 * z - 9 * x^2 = 4) : 
  6 * x^2 + 9 * y^2 + 12 * z^2 = 9 :=
  sorry

end NUMINAMATH_GPT_find_value_of_expression_l1577_157724


namespace NUMINAMATH_GPT_even_function_l1577_157778

theorem even_function (f : ℝ → ℝ) (not_zero : ∃ x, f x ≠ 0) 
  (h : ∀ a b : ℝ, f (a + b) + f (a - b) = 2 * f a + 2 * f b) : 
  ∀ x : ℝ, f (-x) = f x := 
sorry

end NUMINAMATH_GPT_even_function_l1577_157778


namespace NUMINAMATH_GPT_Jean_had_41_candies_at_first_l1577_157763

-- Let total_candies be the initial number of candies Jean had
variable (total_candies : ℕ)
-- Jean gave 18 pieces to a friend
def given_away := 18
-- Jean ate 7 pieces
def eaten := 7
-- Jean has 16 pieces left now
def remaining := 16

-- Calculate the total number of candies initially
def candy_initial (total_candies given_away eaten remaining : ℕ) : Prop :=
  total_candies = remaining + (given_away + eaten)

-- Prove that Jean had 41 pieces of candy initially
theorem Jean_had_41_candies_at_first : candy_initial 41 given_away eaten remaining :=
by
  -- Skipping the proof for now
  sorry

end NUMINAMATH_GPT_Jean_had_41_candies_at_first_l1577_157763


namespace NUMINAMATH_GPT_quadratic_roots_sum_l1577_157774

theorem quadratic_roots_sum (x₁ x₂ m : ℝ) 
  (eq1 : x₁^2 - (2 * m - 2) * x₁ + (m^2 - 2 * m) = 0) 
  (eq2 : x₂^2 - (2 * m - 2) * x₂ + (m^2 - 2 * m) = 0)
  (h : x₁ + x₂ = 10) : m = 6 :=
sorry

end NUMINAMATH_GPT_quadratic_roots_sum_l1577_157774


namespace NUMINAMATH_GPT_q_is_false_given_conditions_l1577_157702

theorem q_is_false_given_conditions
  (h₁: ¬(p ∧ q) = true) 
  (h₂: ¬¬p = true) 
  : q = false := 
sorry

end NUMINAMATH_GPT_q_is_false_given_conditions_l1577_157702


namespace NUMINAMATH_GPT_number_of_planks_needed_l1577_157760

-- Definitions based on conditions
def bed_height : ℕ := 2
def bed_width : ℕ := 2
def bed_length : ℕ := 8
def plank_width : ℕ := 1
def lumber_length : ℕ := 8
def num_beds : ℕ := 10

-- The theorem statement
theorem number_of_planks_needed : (2 * (bed_length / lumber_length) * bed_height) + (2 * ((bed_width * bed_height) / lumber_length) * lumber_length / 4) * num_beds = 60 :=
  by sorry

end NUMINAMATH_GPT_number_of_planks_needed_l1577_157760


namespace NUMINAMATH_GPT_number_of_ways_to_adjust_items_l1577_157707

theorem number_of_ways_to_adjust_items :
  let items_on_upper_shelf := 4
  let items_on_lower_shelf := 8
  let move_items := 2
  let total_ways := Nat.choose items_on_lower_shelf move_items
  total_ways = 840 :=
by
  sorry

end NUMINAMATH_GPT_number_of_ways_to_adjust_items_l1577_157707


namespace NUMINAMATH_GPT_inequality_solution_set_l1577_157749

theorem inequality_solution_set (x : ℝ) :
  x^2 * (x^2 + 2*x + 1) > 2*x * (x^2 + 2*x + 1) ↔
  ((x < -1) ∨ (-1 < x ∧ x < 0) ∨ (2 < x)) :=
sorry

end NUMINAMATH_GPT_inequality_solution_set_l1577_157749


namespace NUMINAMATH_GPT_range_of_a_for_p_range_of_a_for_p_and_q_l1577_157770

variable (a : ℝ)

/-- For any x ∈ ℝ, ax^2 - x + 3 > 0 if and only if a > 1/12 -/
def condition_p : Prop := ∀ x : ℝ, a * x^2 - x + 3 > 0

/-- There exists x ∈ [1, 2] such that 2^x * a ≥ 1 -/
def condition_q : Prop := ∃ x : ℝ, 1 ≤ x ∧ x ≤ 2 ∧ a * 2^x ≥ 1

/-- Theorem (1): The range of values for a such that condition_p holds true is (1/12, +∞) -/
theorem range_of_a_for_p (h : condition_p a) : a > 1/12 :=
sorry

/-- Theorem (2): The range of values for a such that condition_p and condition_q have different truth values is (1/12, 1/4) -/
theorem range_of_a_for_p_and_q (h₁ : condition_p a) (h₂ : ¬condition_q a) : 1/12 < a ∧ a < 1/4 :=
sorry

end NUMINAMATH_GPT_range_of_a_for_p_range_of_a_for_p_and_q_l1577_157770


namespace NUMINAMATH_GPT_mark_bread_time_l1577_157723

def rise_time1 : Nat := 120
def rise_time2 : Nat := 120
def kneading_time : Nat := 10
def baking_time : Nat := 30

def total_time : Nat := rise_time1 + rise_time2 + kneading_time + baking_time

theorem mark_bread_time : total_time = 280 := by
  sorry

end NUMINAMATH_GPT_mark_bread_time_l1577_157723


namespace NUMINAMATH_GPT_chemist_target_temperature_fahrenheit_l1577_157736

noncomputable def kelvinToCelsius (K : ℝ) : ℝ := K - 273.15
noncomputable def celsiusToFahrenheit (C : ℝ) : ℝ := (C * 9 / 5) + 32

theorem chemist_target_temperature_fahrenheit :
  celsiusToFahrenheit (kelvinToCelsius (373.15 - 40)) = 140 :=
by
  sorry

end NUMINAMATH_GPT_chemist_target_temperature_fahrenheit_l1577_157736


namespace NUMINAMATH_GPT_molecular_weights_correct_l1577_157706

-- Define atomic weights
def atomic_weight_Al : Float := 26.98
def atomic_weight_Cl : Float := 35.45
def atomic_weight_K : Float := 39.10

-- Define molecular weight calculations
def molecular_weight_AlCl3 : Float :=
  atomic_weight_Al + 3 * atomic_weight_Cl

def molecular_weight_KCl : Float :=
  atomic_weight_K + atomic_weight_Cl

-- Theorem statement to prove
theorem molecular_weights_correct :
  molecular_weight_AlCl3 = 133.33 ∧ molecular_weight_KCl = 74.55 :=
by
  -- This is where we would normally prove the equivalence
  sorry

end NUMINAMATH_GPT_molecular_weights_correct_l1577_157706


namespace NUMINAMATH_GPT_value_of_expression_l1577_157732

theorem value_of_expression (x : ℝ) (h : 7 * x^2 - 2 * x - 4 = 4 * x + 11) : 
  (5 * x - 7)^2 = 11.63265306 := 
by 
  sorry

end NUMINAMATH_GPT_value_of_expression_l1577_157732

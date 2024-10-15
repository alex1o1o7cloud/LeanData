import Mathlib

namespace NUMINAMATH_GPT_nat_values_of_x_l2424_242429

theorem nat_values_of_x :
  (∃ (x : ℕ), 2^(x - 5) = 2 ∧ x = 6) ∧
  (∃ (x : ℕ), 2^x = 512 ∧ x = 9) ∧
  (∃ (x : ℕ), x^5 = 243 ∧ x = 3) ∧
  (∃ (x : ℕ), x^4 = 625 ∧ x = 5) :=
  by {
    sorry
  }

end NUMINAMATH_GPT_nat_values_of_x_l2424_242429


namespace NUMINAMATH_GPT_integer_solution_system_eq_det_l2424_242419

theorem integer_solution_system_eq_det (a b c d : ℤ) 
  (h : ∀ m n : ℤ, ∃ x y : ℤ, a * x + b * y = m ∧ c * x + d * y = n) : 
  a * d - b * c = 1 ∨ a * d - b * c = -1 :=
by
  sorry

end NUMINAMATH_GPT_integer_solution_system_eq_det_l2424_242419


namespace NUMINAMATH_GPT_vec_a_squared_minus_vec_b_squared_l2424_242469

variable (a b : ℝ × ℝ)
variable (h1 : a + b = (-3, 6))
variable (h2 : a - b = (-3, 2))

theorem vec_a_squared_minus_vec_b_squared : (a.1 * a.1 + a.2 * a.2) - (b.1 * b.1 + b.2 * b.2) = 32 :=
sorry

end NUMINAMATH_GPT_vec_a_squared_minus_vec_b_squared_l2424_242469


namespace NUMINAMATH_GPT_find_k_l2424_242445

noncomputable def vec_a : ℝ × ℝ := (1, 2)
noncomputable def vec_b : ℝ × ℝ := (-3, 2)
noncomputable def vec_k_a_plus_b (k : ℝ) : ℝ × ℝ := (k - 3, 2 * k + 2)
noncomputable def vec_a_minus_3b : ℝ × ℝ := (10, -4)

theorem find_k :
  ∃! k : ℝ, (vec_k_a_plus_b k).1 * vec_a_minus_3b.2 = (vec_k_a_plus_b k).2 * vec_a_minus_3b.1 ∧ k = -1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l2424_242445


namespace NUMINAMATH_GPT_at_least_six_consecutive_heads_l2424_242478

noncomputable def flip_probability : ℚ :=
  let total_outcomes := 2^8
  let successful_outcomes := 7
  successful_outcomes / total_outcomes

theorem at_least_six_consecutive_heads : 
  flip_probability = 7 / 256 :=
by
  sorry

end NUMINAMATH_GPT_at_least_six_consecutive_heads_l2424_242478


namespace NUMINAMATH_GPT_sixth_number_is_eight_l2424_242423

/- 
  The conditions are:
  1. The sequence is an increasing list of consecutive integers.
  2. The 3rd and 4th numbers add up to 11.
  We need to prove that the 6th number is 8.
-/

theorem sixth_number_is_eight (n : ℕ) (h : n + (n + 1) = 11) : (n + 3) = 8 :=
by
  sorry

end NUMINAMATH_GPT_sixth_number_is_eight_l2424_242423


namespace NUMINAMATH_GPT_expression_not_defined_l2424_242481

-- Define the quadratic equation
def quadratic_eq (x : ℝ) : ℝ := x^2 - 25*x + 125

-- Theorem statement that the expression is not defined for specific values of x
theorem expression_not_defined (x : ℝ) : quadratic_eq x = 0 ↔ (x = 5 ∨ x = 20) :=
by
  sorry

end NUMINAMATH_GPT_expression_not_defined_l2424_242481


namespace NUMINAMATH_GPT_find_number_in_parentheses_l2424_242407

theorem find_number_in_parentheses :
  ∃ x : ℝ, 3 + 2 * (x - 3) = 24.16 ∧ x = 13.58 :=
by
  sorry

end NUMINAMATH_GPT_find_number_in_parentheses_l2424_242407


namespace NUMINAMATH_GPT_find_p_q_sum_l2424_242494

variable (P Q x : ℝ)

theorem find_p_q_sum (h : (P / (x - 3)) +  Q * (x + 2) = (-5 * x^2 + 18 * x + 40) / (x - 3)) : P + Q = 20 :=
sorry

end NUMINAMATH_GPT_find_p_q_sum_l2424_242494


namespace NUMINAMATH_GPT_time_saved_calculator_l2424_242476

-- Define the conditions
def time_with_calculator (n : ℕ) : ℕ := 2 * n
def time_without_calculator (n : ℕ) : ℕ := 5 * n
def total_problems : ℕ := 20

-- State the theorem to prove the time saved is 60 minutes
theorem time_saved_calculator : 
  time_without_calculator total_problems - time_with_calculator total_problems = 60 :=
sorry

end NUMINAMATH_GPT_time_saved_calculator_l2424_242476


namespace NUMINAMATH_GPT_continuous_function_nondecreasing_l2424_242408

open Set

variable {α : Type*} [LinearOrder ℝ] [Preorder ℝ]

theorem continuous_function_nondecreasing
  (f : (ℝ)→ ℝ) 
  (h_cont : ContinuousOn f (Ioi 0))
  (h_seq : ∀ x > 0, Monotone (fun n : ℕ => f (n*x))):
  ∀ x y, x ≤ y → f x ≤ f y := 
sorry

end NUMINAMATH_GPT_continuous_function_nondecreasing_l2424_242408


namespace NUMINAMATH_GPT_quadratic_solution_1_quadratic_solution_2_l2424_242483

theorem quadratic_solution_1 (x : ℝ) :
  x^2 + 3 * x - 1 = 0 ↔ (x = (-3 + Real.sqrt 13) / 2) ∨ (x = (-3 - Real.sqrt 13) / 2) :=
by
  sorry

theorem quadratic_solution_2 (x : ℝ) :
  (x - 2)^2 = 2 * (x - 2) ↔ (x = 2) ∨ (x = 4) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_solution_1_quadratic_solution_2_l2424_242483


namespace NUMINAMATH_GPT_max_value_expression_l2424_242455

open Real

theorem max_value_expression (x : ℝ) : 
  ∃ (y : ℝ), y ≤ (x^6 / (x^10 + 3 * x^8 - 5 * x^6 + 10 * x^4 + 25)) ∧
  y = 1 / (5 + 2 * sqrt 30) :=
sorry

end NUMINAMATH_GPT_max_value_expression_l2424_242455


namespace NUMINAMATH_GPT_no_integer_roots_of_polynomial_l2424_242450

theorem no_integer_roots_of_polynomial :
  ¬ ∃ (x : ℤ), x^3 - 3 * x^2 - 10 * x + 20 = 0 :=
by
  sorry

end NUMINAMATH_GPT_no_integer_roots_of_polynomial_l2424_242450


namespace NUMINAMATH_GPT_arithmetic_progression_contains_sixth_power_l2424_242475

theorem arithmetic_progression_contains_sixth_power
  (a h : ℕ) (a_pos : 0 < a) (h_pos : 0 < h)
  (sq : ∃ n : ℕ, a + n * h = k^2)
  (cube : ∃ m : ℕ, a + m * h = l^3) :
  ∃ p : ℕ, ∃ q : ℕ, a + q * h = p^6 := sorry

end NUMINAMATH_GPT_arithmetic_progression_contains_sixth_power_l2424_242475


namespace NUMINAMATH_GPT_hotel_people_per_room_l2424_242438

theorem hotel_people_per_room
  (total_rooms : ℕ := 10)
  (towels_per_person : ℕ := 2)
  (total_towels : ℕ := 60) :
  (total_towels / towels_per_person) / total_rooms = 3 :=
by
  sorry

end NUMINAMATH_GPT_hotel_people_per_room_l2424_242438


namespace NUMINAMATH_GPT_p_or_q_not_necessarily_true_l2424_242417

theorem p_or_q_not_necessarily_true (p q : Prop) (hnp : ¬p) (hpq : ¬(p ∧ q)) : ¬(p ∨ q) ∨ (p ∨ q) :=
by
  sorry

end NUMINAMATH_GPT_p_or_q_not_necessarily_true_l2424_242417


namespace NUMINAMATH_GPT_exists_consecutive_integers_not_sum_of_two_squares_l2424_242421

open Nat

theorem exists_consecutive_integers_not_sum_of_two_squares : 
  ∃ (m : ℕ), ∀ k : ℕ, k < 2017 → ¬(∃ a b : ℤ, (m + k) = a^2 + b^2) := 
sorry

end NUMINAMATH_GPT_exists_consecutive_integers_not_sum_of_two_squares_l2424_242421


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l2424_242449

noncomputable def U : Set ℝ := Set.univ
def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x < 3}
def B : Set ℝ := {x : ℝ | x < -1 ∨ x ≥ 4}

theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | -2 ≤ x ∧ x < -1} :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l2424_242449


namespace NUMINAMATH_GPT_extreme_values_f_range_of_a_l2424_242435

noncomputable def f (x a : ℝ) : ℝ := x^3 - x^2 - x - a
noncomputable def df (x : ℝ) : ℝ := 3 * x^2 - 2 * x - 1

theorem extreme_values_f (a : ℝ) :
  ∃ (x₁ x₂ : ℝ), df x₁ = 0 ∧ df x₂ = 0 ∧ f x₁ a = (5 / 27) - a ∧ f x₂ a = -1 - a :=
sorry

theorem range_of_a (a : ℝ) :
  (∃ (a : ℝ), f (-1/3) a < 0 ∧ f 1 a > 0) ↔ (a < -1 ∨ a > 5 / 27) :=
sorry

end NUMINAMATH_GPT_extreme_values_f_range_of_a_l2424_242435


namespace NUMINAMATH_GPT_Buffy_whiskers_is_40_l2424_242473

def number_of_whiskers (Puffy Scruffy Buffy Juniper : ℕ) : Prop :=
  Puffy = 3 * Juniper ∧
  Puffy = Scruffy / 2 ∧
  Buffy = (Puffy + Scruffy + Juniper) / 3 ∧
  Juniper = 12

theorem Buffy_whiskers_is_40 :
  ∃ (Puffy Scruffy Buffy Juniper : ℕ), 
    number_of_whiskers Puffy Scruffy Buffy Juniper ∧ Buffy = 40 := 
by
  sorry

end NUMINAMATH_GPT_Buffy_whiskers_is_40_l2424_242473


namespace NUMINAMATH_GPT_jerry_added_action_figures_l2424_242442

theorem jerry_added_action_figures (x : ℕ) (h1 : 7 + x - 10 = 8) : x = 11 :=
by
  sorry

end NUMINAMATH_GPT_jerry_added_action_figures_l2424_242442


namespace NUMINAMATH_GPT_roman_numeral_sketching_l2424_242496

/-- Roman numeral sketching problem. -/
theorem roman_numeral_sketching (n : ℕ) (k : ℕ) (students : ℕ) 
  (h1 : ∀ i : ℕ, 1 ≤ i ∧ i ≤ n ∧ i / 1 = i) 
  (h2 : ∀ i : ℕ, i > n → i = n - (i - n)) 
  (h3 : k = 7) 
  (h4 : ∀ r : ℕ, r = (k * n)) : students = 350 :=
by
  sorry

end NUMINAMATH_GPT_roman_numeral_sketching_l2424_242496


namespace NUMINAMATH_GPT_factorize_3a_squared_minus_6a_plus_3_l2424_242460

theorem factorize_3a_squared_minus_6a_plus_3 (a : ℝ) : 
  3 * a^2 - 6 * a + 3 = 3 * (a - 1)^2 :=
by 
  sorry

end NUMINAMATH_GPT_factorize_3a_squared_minus_6a_plus_3_l2424_242460


namespace NUMINAMATH_GPT_solution_set_of_inequality_l2424_242431

open Set

theorem solution_set_of_inequality :
  {x : ℝ | - x ^ 2 - 4 * x + 5 > 0} = {x : ℝ | -5 < x ∧ x < 1} :=
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l2424_242431


namespace NUMINAMATH_GPT_simplify_fraction_l2424_242470

theorem simplify_fraction :
  ((3^2008)^2 - (3^2006)^2) / ((3^2007)^2 - (3^2005)^2) = 9 :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l2424_242470


namespace NUMINAMATH_GPT_shortest_segment_length_l2424_242463

theorem shortest_segment_length :
  let total_length := 1
  let red_dot := 0.618
  let yellow_dot := total_length - red_dot  -- yellow_dot is at the same point after fold
  let first_cut := red_dot  -- Cut the strip at the red dot
  let remaining_strip := red_dot
  let distance_between_red_and_yellow := total_length - 2 * yellow_dot
  let second_cut := distance_between_red_and_yellow
  let shortest_segment := remaining_strip - 2 * distance_between_red_and_yellow
  shortest_segment = 0.146 :=
by
  sorry

end NUMINAMATH_GPT_shortest_segment_length_l2424_242463


namespace NUMINAMATH_GPT_area_of_ABC_l2424_242448

noncomputable def area_of_triangle (AB AC angleB : ℝ) : ℝ :=
  0.5 * AB * AC * Real.sin angleB

theorem area_of_ABC :
  area_of_triangle 5 3 (120 * Real.pi / 180) = (15 * Real.sqrt 3) / 4 :=
by
  sorry

end NUMINAMATH_GPT_area_of_ABC_l2424_242448


namespace NUMINAMATH_GPT_hem_dress_time_l2424_242430

theorem hem_dress_time
  (hem_length_feet : ℕ)
  (stitch_length_inches : ℝ)
  (stitches_per_minute : ℕ)
  (hem_length_inches : ℝ)
  (total_stitches : ℕ)
  (time_minutes : ℝ)
  (h1 : hem_length_feet = 3)
  (h2 : stitch_length_inches = 1 / 4)
  (h3 : stitches_per_minute = 24)
  (h4 : hem_length_inches = 12 * hem_length_feet)
  (h5 : total_stitches = hem_length_inches / stitch_length_inches)
  (h6 : time_minutes = total_stitches / stitches_per_minute) :
  time_minutes = 6 := 
sorry

end NUMINAMATH_GPT_hem_dress_time_l2424_242430


namespace NUMINAMATH_GPT_inequality_1_inequality_2_l2424_242406

noncomputable def f (x : ℝ) : ℝ := abs (x - 1) - abs (x - 2)

theorem inequality_1 (x : ℝ) : f x > 2 * x ↔ x < -1/2 :=
sorry

theorem inequality_2 (t : ℝ) :
  (∃ x : ℝ, f x > t ^ 2 - t + 1) ↔ (0 < t ∧ t < 1) :=
sorry

end NUMINAMATH_GPT_inequality_1_inequality_2_l2424_242406


namespace NUMINAMATH_GPT_total_clothes_washed_l2424_242477

def number_of_clothing_items (Cally Danny Emily shared_socks : ℕ) : ℕ :=
  Cally + Danny + Emily + shared_socks

theorem total_clothes_washed :
  let Cally_clothes := (10 + 5 + 7 + 6 + 3)
  let Danny_clothes := (6 + 8 + 10 + 6 + 4)
  let Emily_clothes := (8 + 6 + 9 + 5 + 2)
  let shared_socks := (3 + 2)
  number_of_clothing_items Cally_clothes Danny_clothes Emily_clothes shared_socks = 100 :=
by
  sorry

end NUMINAMATH_GPT_total_clothes_washed_l2424_242477


namespace NUMINAMATH_GPT_find_m_value_l2424_242411

-- Define the quadratic function
def quadratic (x m : ℝ) : ℝ := x^2 - 6 * x + m

-- Define the condition that the quadratic function has a minimum value of 1
def has_minimum_value_of_one (m : ℝ) : Prop := ∃ x : ℝ, quadratic x m = 1

-- The main theorem statement
theorem find_m_value : ∀ m : ℝ, has_minimum_value_of_one m → m = 10 :=
by sorry

end NUMINAMATH_GPT_find_m_value_l2424_242411


namespace NUMINAMATH_GPT_percent_women_non_union_employees_is_65_l2424_242485

-- Definitions based on the conditions
variables {E : ℝ} -- Denoting the total number of employees as a real number

def percent_men (E : ℝ) : ℝ := 0.56 * E
def percent_union_employees (E : ℝ) : ℝ := 0.60 * E
def percent_non_union_employees (E : ℝ) : ℝ := 0.40 * E
def percent_women_non_union (percent_non_union_employees : ℝ) : ℝ := 0.65 * percent_non_union_employees

-- Theorem statement
theorem percent_women_non_union_employees_is_65 :
  percent_women_non_union (percent_non_union_employees E) / (percent_non_union_employees E) = 0.65 :=
by
  sorry

end NUMINAMATH_GPT_percent_women_non_union_employees_is_65_l2424_242485


namespace NUMINAMATH_GPT_sin_cos_acute_angle_lt_one_l2424_242428

theorem sin_cos_acute_angle_lt_one (α β : ℝ) (a b c : ℝ) (hα : 0 < α ∧ α < π / 2)
  (hβ : 0 < β ∧ β < π / 2) (h_triangle : a^2 + b^2 = c^2) (h_nonzero_c : c ≠ 0) :
  (a / c < 1) ∧ (b / c < 1) :=
by 
  sorry

end NUMINAMATH_GPT_sin_cos_acute_angle_lt_one_l2424_242428


namespace NUMINAMATH_GPT_set_of_x_satisfying_2f_less_than_x_plus_1_l2424_242432

theorem set_of_x_satisfying_2f_less_than_x_plus_1 (f : ℝ → ℝ) 
  (h1 : f 1 = 1) 
  (h2 : ∀ x : ℝ, deriv f x > 1 / 2) :
  { x : ℝ | 2 * f x < x + 1 } = { x : ℝ | x < 1 } :=
by
  sorry

end NUMINAMATH_GPT_set_of_x_satisfying_2f_less_than_x_plus_1_l2424_242432


namespace NUMINAMATH_GPT_girls_on_debate_team_l2424_242405

def number_of_students (groups: ℕ) (group_size: ℕ) : ℕ :=
  groups * group_size

def total_students_debate_team : ℕ :=
  number_of_students 8 9

def number_of_boys : ℕ := 26

def number_of_girls : ℕ :=
  total_students_debate_team - number_of_boys

theorem girls_on_debate_team :
  number_of_girls = 46 :=
by
  sorry

end NUMINAMATH_GPT_girls_on_debate_team_l2424_242405


namespace NUMINAMATH_GPT_students_per_table_l2424_242451

theorem students_per_table (total_students tables students_bathroom students_canteen added_students exchange_students : ℕ) 
  (h1 : total_students = 47)
  (h2 : tables = 6)
  (h3 : students_bathroom = 3)
  (h4 : students_canteen = 3 * students_bathroom)
  (h5 : added_students = 2 * 4)
  (h6 : exchange_students = 3 + 3 + 3) :
  (total_students - (students_bathroom + students_canteen + added_students + exchange_students)) / tables = 3 := 
by 
  sorry

end NUMINAMATH_GPT_students_per_table_l2424_242451


namespace NUMINAMATH_GPT_remainder_43_pow_97_pow_5_plus_109_mod_163_l2424_242425

theorem remainder_43_pow_97_pow_5_plus_109_mod_163 :
    (43 ^ (97 ^ 5) + 109) % 163 = 50 :=
by
  sorry

end NUMINAMATH_GPT_remainder_43_pow_97_pow_5_plus_109_mod_163_l2424_242425


namespace NUMINAMATH_GPT_shaded_areas_are_different_l2424_242479

theorem shaded_areas_are_different :
  let shaded_area_I := 3 / 8
  let shaded_area_II := 1 / 3
  let shaded_area_III := 1 / 2
  (shaded_area_I ≠ shaded_area_II) ∧ (shaded_area_I ≠ shaded_area_III) ∧ (shaded_area_II ≠ shaded_area_III) :=
by
  sorry

end NUMINAMATH_GPT_shaded_areas_are_different_l2424_242479


namespace NUMINAMATH_GPT_monotonically_increasing_interval_l2424_242459

def f (x : ℝ) (a : ℝ) : ℝ := |2 * x + a| + 3

theorem monotonically_increasing_interval (a : ℝ) :
  (∀ x₁ x₂ : ℝ, 1 < x₁ → x₁ < x₂ → f x₁ a ≤ f x₂ a) → a ≥ -2 :=
by
  sorry

end NUMINAMATH_GPT_monotonically_increasing_interval_l2424_242459


namespace NUMINAMATH_GPT_interest_rate_proven_l2424_242420

structure InvestmentProblem where
  P : ℝ  -- Principal amount
  A : ℝ  -- Accumulated amount
  n : ℕ  -- Number of times interest is compounded per year
  t : ℕ  -- Time in years
  rate : ℝ  -- Interest rate per annum (to be proven)

noncomputable def solve_interest_rate (ip : InvestmentProblem) : ℝ :=
  let half_yearly_rate := ip.rate / 2 / 100
  let amount_formula := ip.P * (1 + half_yearly_rate)^(ip.n * ip.t)
  half_yearly_rate

theorem interest_rate_proven :
  ∀ (P A : ℝ) (n t : ℕ), 
  P = 6000 → 
  A = 6615 → 
  n = 2 → 
  t = 1 → 
  solve_interest_rate {P := P, A := A, n := n, t := t, rate := 10.0952} = 10.0952 := 
by 
  intros
  rw [solve_interest_rate]
  sorry

end NUMINAMATH_GPT_interest_rate_proven_l2424_242420


namespace NUMINAMATH_GPT_michael_eggs_count_l2424_242497

def initial_crates : List ℕ := [24, 28, 32, 36, 40, 44]
def wednesday_given : List ℕ := [28, 32, 40]
def thursday_purchases : List ℕ := [50, 45, 55, 60]
def friday_sold : List ℕ := [60, 55]

theorem michael_eggs_count :
  let total_tuesday := initial_crates.sum
  let total_given_wednesday := wednesday_given.sum
  let remaining_wednesday := total_tuesday - total_given_wednesday
  let total_thursday := thursday_purchases.sum
  let total_after_thursday := remaining_wednesday + total_thursday
  let total_sold_friday := friday_sold.sum
  total_after_thursday - total_sold_friday = 199 :=
by
  sorry

end NUMINAMATH_GPT_michael_eggs_count_l2424_242497


namespace NUMINAMATH_GPT_scientific_notation_five_hundred_billion_l2424_242436

theorem scientific_notation_five_hundred_billion :
  500000000000 = 5 * 10^11 := by
  sorry

end NUMINAMATH_GPT_scientific_notation_five_hundred_billion_l2424_242436


namespace NUMINAMATH_GPT_Nell_cards_difference_l2424_242416

-- Definitions
def initial_baseball_cards : ℕ := 438
def initial_ace_cards : ℕ := 18
def given_ace_cards : ℕ := 55
def given_baseball_cards : ℕ := 178

-- Theorem statement
theorem Nell_cards_difference :
  given_baseball_cards - given_ace_cards = 123 := 
by
  sorry

end NUMINAMATH_GPT_Nell_cards_difference_l2424_242416


namespace NUMINAMATH_GPT_Buratino_math_problem_l2424_242413

theorem Buratino_math_problem (x : ℝ) : 4 * x + 15 = 15 * x + 4 → x = 1 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_Buratino_math_problem_l2424_242413


namespace NUMINAMATH_GPT_max_marked_cells_100x100_board_l2424_242495

theorem max_marked_cells_100x100_board : 
  ∃ n, (3 * n + 1 = 100) ∧ (2 * n + 1) * (n + 1) = 2278 :=
by
  sorry

end NUMINAMATH_GPT_max_marked_cells_100x100_board_l2424_242495


namespace NUMINAMATH_GPT_fence_poles_count_l2424_242472

def length_path : ℕ := 900
def length_bridge : ℕ := 42
def distance_between_poles : ℕ := 6

theorem fence_poles_count :
  2 * (length_path - length_bridge) / distance_between_poles = 286 :=
by
  sorry

end NUMINAMATH_GPT_fence_poles_count_l2424_242472


namespace NUMINAMATH_GPT_find_m_l2424_242480

theorem find_m (m x1 x2 : ℝ) (h1 : x1^2 + m * x1 + 5 = 0) (h2 : x2^2 + m * x2 + 5 = 0) (h3 : x1 = 2 * |x2| - 3) : 
  m = -9 / 2 :=
sorry

end NUMINAMATH_GPT_find_m_l2424_242480


namespace NUMINAMATH_GPT_factorization_correct_l2424_242437

theorem factorization_correct (C D : ℤ) (h : 15 = C * D ∧ 48 = 8 * 6 ∧ -56 = -8 * D - 6 * C):
  C * D + C = 18 :=
  sorry

end NUMINAMATH_GPT_factorization_correct_l2424_242437


namespace NUMINAMATH_GPT_geometric_sequence_alpha_5_l2424_242466

theorem geometric_sequence_alpha_5 (α : ℕ → ℝ) (h1 : α 4 * α 5 * α 6 = 27) (h2 : α 4 * α 6 = (α 5) ^ 2) : α 5 = 3 := 
sorry

end NUMINAMATH_GPT_geometric_sequence_alpha_5_l2424_242466


namespace NUMINAMATH_GPT_find_coefficient_of_x_in_expansion_l2424_242492

noncomputable def coefficient_of_x_in_expansion (x : ℤ) : ℤ :=
  (1 / 2 * x - 1) * (2 * x - 1 / x) ^ 6

theorem find_coefficient_of_x_in_expansion :
  coefficient_of_x_in_expansion x = -80 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_coefficient_of_x_in_expansion_l2424_242492


namespace NUMINAMATH_GPT_problem1_problem2_l2424_242461

noncomputable def f (x : ℝ) : ℝ := 1 / (1 + x^2)

theorem problem1 :
  f 1 + f 2 + f 3 + f (1 / 2) + f (1 / 3) = 5 / 2 :=
by
  sorry

theorem problem2 : ∀ x : ℝ, 0 < f x ∧ f x ≤ 1 :=
by
  intro x
  sorry

end NUMINAMATH_GPT_problem1_problem2_l2424_242461


namespace NUMINAMATH_GPT_brand_z_percentage_correct_l2424_242462

noncomputable def percentage_of_brand_z (capacity : ℝ := 1) (brand_z1 : ℝ := 1) (brand_x1 : ℝ := 0) 
(brand_z2 : ℝ := 1/4) (brand_x2 : ℝ := 3/4) (brand_z3 : ℝ := 5/8) (brand_x3 : ℝ := 3/8) 
(brand_z4 : ℝ := 5/16) (brand_x4 : ℝ := 11/16) : ℝ :=
    (brand_z4 / (brand_z4 + brand_x4)) * 100

theorem brand_z_percentage_correct : percentage_of_brand_z = 31.25 := by
  sorry

end NUMINAMATH_GPT_brand_z_percentage_correct_l2424_242462


namespace NUMINAMATH_GPT_decreasing_interval_f_l2424_242447

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * x^2 - Real.log x

theorem decreasing_interval_f :
  ∀ x : ℝ, x > 0 → (f (x) = (1 / 2) * x^2 - Real.log x) →
  (∃ a b : ℝ, 0 < a ∧ a ≤ b ∧ b = 1 ∧ ∀ y, a < y ∧ y ≤ b → f (y) ≤ f (y+1)) := sorry

end NUMINAMATH_GPT_decreasing_interval_f_l2424_242447


namespace NUMINAMATH_GPT_problem_equiv_proof_l2424_242418

theorem problem_equiv_proof : ∀ (i : ℂ), i^2 = -1 → (1 + i^2017) / (1 - i) = i :=
by
  intro i h
  sorry

end NUMINAMATH_GPT_problem_equiv_proof_l2424_242418


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l2424_242415

theorem isosceles_triangle_perimeter :
  ∃ P : ℕ, (P = 15 ∨ P = 18) ∧ ∀ (a b c : ℕ), (a = 7 ∨ b = 7 ∨ c = 7) ∧ (a = 4 ∨ b = 4 ∨ c = 4) → ((a = 7 ∨ a = 4) ∧ (b = 7 ∨ b = 4) ∧ (c = 7 ∨ c = 4)) ∧ P = a + b + c :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l2424_242415


namespace NUMINAMATH_GPT_cost_per_millisecond_l2424_242471

theorem cost_per_millisecond
  (C : ℝ)
  (h1 : 1.07 + (C * 1500) + 5.35 = 40.92) :
  C = 0.023 :=
sorry

end NUMINAMATH_GPT_cost_per_millisecond_l2424_242471


namespace NUMINAMATH_GPT_nn_gt_n1n1_l2424_242409

theorem nn_gt_n1n1 (n : ℕ) (h : n > 1) : n^n > (n + 1)^(n - 1) := 
sorry

end NUMINAMATH_GPT_nn_gt_n1n1_l2424_242409


namespace NUMINAMATH_GPT_wall_area_l2424_242498

-- Define the conditions
variables (R J D : ℕ) (L W : ℝ)
variable (area_regular_tiles : ℝ)
variables (ratio_regular : ℕ) (ratio_jumbo : ℕ) (ratio_diamond : ℕ)
variables (length_ratio_jumbo : ℝ) (width_ratio_jumbo : ℝ)
variables (length_ratio_diamond : ℝ) (width_ratio_diamond : ℝ)
variable (total_area : ℝ)

-- Assign values to the conditions
axiom ratio : ratio_regular = 4 ∧ ratio_jumbo = 2 ∧ ratio_diamond = 1
axiom size_regular : area_regular_tiles = 80
axiom jumbo_tile_ratio : length_ratio_jumbo = 3 ∧ width_ratio_jumbo = 3
axiom diamond_tile_ratio : length_ratio_diamond = 2 ∧ width_ratio_diamond = 0.5

-- Define the statement
theorem wall_area (ratio : ratio_regular = 4 ∧ ratio_jumbo = 2 ∧ ratio_diamond = 1)
    (size_regular : area_regular_tiles = 80)
    (jumbo_tile_ratio : length_ratio_jumbo = 3 ∧ width_ratio_jumbo = 3)
    (diamond_tile_ratio : length_ratio_diamond = 2 ∧ width_ratio_diamond = 0.5):
    total_area = 140 := 
sorry

end NUMINAMATH_GPT_wall_area_l2424_242498


namespace NUMINAMATH_GPT_correct_option_B_l2424_242468

variable {a b x y : ℤ}

def option_A (a : ℤ) : Prop := -a - a = 0
def option_B (x y : ℤ) : Prop := -(x + y) = -x - y
def option_C (b a : ℤ) : Prop := 3 * (b - 2 * a) = 3 * b - 2 * a
def option_D (a : ℤ) : Prop := 8 * a^4 - 6 * a^2 = 2 * a^2

theorem correct_option_B (x y : ℤ) : option_B x y := by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_correct_option_B_l2424_242468


namespace NUMINAMATH_GPT_total_flour_correct_l2424_242439

-- Define the quantities specified in the conditions
def cups_of_flour_already_added : ℕ := 2
def cups_of_flour_to_add : ℕ := 7

-- Define the total cups of flour required by the recipe as a sum of the quantities
def cups_of_flour_required : ℕ := cups_of_flour_already_added + cups_of_flour_to_add

-- Prove that the total cups of flour required is 9
theorem total_flour_correct : cups_of_flour_required = 9 := by
  -- use auto proof placeholder
  rfl

end NUMINAMATH_GPT_total_flour_correct_l2424_242439


namespace NUMINAMATH_GPT_rubles_exchange_l2424_242453

theorem rubles_exchange (x : ℕ) : 
  (3000 * x - 7000 = 2950 * x) → x = 140 := by
  sorry

end NUMINAMATH_GPT_rubles_exchange_l2424_242453


namespace NUMINAMATH_GPT_circle_equation_through_intersections_l2424_242465

theorem circle_equation_through_intersections 
  (h₁ : ∀ x y : ℝ, x^2 + y^2 + 6 * x - 4 = 0 ↔ x^2 + y^2 + 6 * y - 28 = 0)
  (h₂ : ∀ x y : ℝ, x - y - 4 = 0) : 
  ∃ x y : ℝ, (x - 1/2) ^ 2 + (y + 7 / 2) ^ 2 = 89 / 2 :=
by sorry

end NUMINAMATH_GPT_circle_equation_through_intersections_l2424_242465


namespace NUMINAMATH_GPT_find_pairs_l2424_242490

theorem find_pairs (x y : ℤ) (h : 19 / x + 96 / y = (19 * 96) / (x * y)) :
  ∃ m : ℤ, x = 19 * m ∧ y = 96 - 96 * m :=
by
  sorry

end NUMINAMATH_GPT_find_pairs_l2424_242490


namespace NUMINAMATH_GPT_smallest_sum_arith_geo_seq_l2424_242489

theorem smallest_sum_arith_geo_seq (A B C D : ℕ) 
  (h1 : A + B + C + D > 0)
  (h2 : 2 * B = A + C)
  (h3 : 16 * C = 7 * B)
  (h4 : 16 * D = 49 * B) :
  A + B + C + D = 97 :=
sorry

end NUMINAMATH_GPT_smallest_sum_arith_geo_seq_l2424_242489


namespace NUMINAMATH_GPT_symmetric_point_m_eq_one_l2424_242427

theorem symmetric_point_m_eq_one (m : ℝ) (A B : ℝ × ℝ) 
  (hA : A = (-3, 2 * m - 1))
  (hB : B = (-3, -1))
  (symmetric : A.1 = B.1 ∧ A.2 = -B.2) : 
  m = 1 :=
by
  sorry

end NUMINAMATH_GPT_symmetric_point_m_eq_one_l2424_242427


namespace NUMINAMATH_GPT_answer_key_combinations_l2424_242452

theorem answer_key_combinations : 
  (2^3 - 2) * 4^2 = 96 := 
by 
  -- Explanation about why it equals to this multi-step skipped, directly written as sorry.
  sorry

end NUMINAMATH_GPT_answer_key_combinations_l2424_242452


namespace NUMINAMATH_GPT_pow_mod_eq_l2424_242414

theorem pow_mod_eq :
  11 ^ 2023 % 5 = 1 :=
by
  sorry

end NUMINAMATH_GPT_pow_mod_eq_l2424_242414


namespace NUMINAMATH_GPT_investment_growth_theorem_l2424_242467

variable (x : ℝ)

-- Defining the initial and final investments
def initial_investment : ℝ := 800
def final_investment : ℝ := 960

-- Defining the growth equation
def growth_equation (x : ℝ) : Prop := initial_investment * (1 + x) ^ 2 = final_investment

-- The theorem statement that needs to be proven
theorem investment_growth_theorem : growth_equation x := sorry

end NUMINAMATH_GPT_investment_growth_theorem_l2424_242467


namespace NUMINAMATH_GPT_slope_and_intercept_of_line_l2424_242486

theorem slope_and_intercept_of_line :
  ∀ (x y : ℝ), 3 * x + 2 * y + 6 = 0 → y = - (3 / 2) * x - 3 :=
by
  intros x y h
  sorry

end NUMINAMATH_GPT_slope_and_intercept_of_line_l2424_242486


namespace NUMINAMATH_GPT_calc_result_l2424_242491

theorem calc_result : (-3)^2 - (-2)^3 = 17 := 
by
  sorry

end NUMINAMATH_GPT_calc_result_l2424_242491


namespace NUMINAMATH_GPT_find_the_number_l2424_242499

theorem find_the_number (x k : ℕ) (h1 : x / k = 4) (h2 : k = 8) : x = 32 := by
  sorry

end NUMINAMATH_GPT_find_the_number_l2424_242499


namespace NUMINAMATH_GPT_fn_conjecture_l2424_242440

theorem fn_conjecture (f : ℕ → ℝ → ℝ) (x : ℝ) (h_pos : x > 0) :
  (f 1 x = x / (Real.sqrt (1 + x^2))) →
  (∀ n, f (n + 1) x = f 1 (f n x)) →
  (∀ n, f n x = x / (Real.sqrt (1 + n * x ^ 2))) := by
  sorry

end NUMINAMATH_GPT_fn_conjecture_l2424_242440


namespace NUMINAMATH_GPT_area_DEF_l2424_242464

structure Point where
  x : ℝ
  y : ℝ

def D : Point := {x := -3, y := 4}
def E : Point := {x := 1, y := 7}
def F : Point := {x := 3, y := -1}

def area_of_triangle (A B C : Point) : ℝ :=
  0.5 * |(A.x * B.y + B.x * C.y + C.x * A.y - A.y * B.x - B.y * C.x - C.y * A.x)|

theorem area_DEF : area_of_triangle D E F = 16 := by
  sorry

end NUMINAMATH_GPT_area_DEF_l2424_242464


namespace NUMINAMATH_GPT_expression_value_l2424_242487

noncomputable def expression (x b : ℝ) : ℝ :=
  (x / (x + b) + b / (x - b)) / (b / (x + b) - x / (x - b))

theorem expression_value (b x : ℝ) (hb : b ≠ 0) (hx : x ≠ b ∧ x ≠ -b) :
  expression x b = -1 := 
by
  sorry

end NUMINAMATH_GPT_expression_value_l2424_242487


namespace NUMINAMATH_GPT_find_c_l2424_242493

-- Definitions for the conditions
def is_solution (x c : ℝ) : Prop := x^2 + c * x - 36 = 0

theorem find_c (c : ℝ) (h : is_solution (-9) c) : c = 5 :=
sorry

end NUMINAMATH_GPT_find_c_l2424_242493


namespace NUMINAMATH_GPT_John_used_16_bulbs_l2424_242424

variable (X : ℕ)

theorem John_used_16_bulbs
  (h1 : 40 - X = 2 * 12) :
  X = 16 := 
sorry

end NUMINAMATH_GPT_John_used_16_bulbs_l2424_242424


namespace NUMINAMATH_GPT_base_length_of_vessel_l2424_242412

def volume_of_cube (edge : ℝ) := edge^3

def volume_of_displaced_water (L width rise : ℝ) := L * width * rise

theorem base_length_of_vessel (edge width rise L : ℝ) 
  (h1 : edge = 15) (h2 : width = 15) (h3 : rise = 11.25) 
  (h4 : volume_of_displaced_water L width rise = volume_of_cube edge) : 
  L = 20 :=
by
  sorry

end NUMINAMATH_GPT_base_length_of_vessel_l2424_242412


namespace NUMINAMATH_GPT_top_layer_blocks_l2424_242444

theorem top_layer_blocks (x : Nat) (h : x + 3 * x + 9 * x + 27 * x = 40) : x = 1 :=
by
  sorry

end NUMINAMATH_GPT_top_layer_blocks_l2424_242444


namespace NUMINAMATH_GPT_KayleeAgeCorrect_l2424_242403

-- Define Kaylee's current age
def KayleeCurrentAge (k : ℕ) : Prop :=
  (3 * 5 + (7 - k) = 7)

-- State the theorem
theorem KayleeAgeCorrect : ∃ k : ℕ, KayleeCurrentAge k ∧ k = 8 := 
sorry

end NUMINAMATH_GPT_KayleeAgeCorrect_l2424_242403


namespace NUMINAMATH_GPT_number_of_yellow_crayons_l2424_242410

theorem number_of_yellow_crayons : 
  ∃ (R B Y : ℕ), 
  R = 14 ∧ 
  B = R + 5 ∧ 
  Y = 2 * B - 6 ∧ 
  Y = 32 :=
by
  sorry

end NUMINAMATH_GPT_number_of_yellow_crayons_l2424_242410


namespace NUMINAMATH_GPT_modulo_remainder_even_l2424_242404

theorem modulo_remainder_even (k : ℕ) (h : 1 ≤ k ∧ k ≤ 2018) : 
  ((2019 - k)^12 + 2018) % 2019 = (k^12 + 2018) % 2019 := 
by
  sorry

end NUMINAMATH_GPT_modulo_remainder_even_l2424_242404


namespace NUMINAMATH_GPT_kevin_hopping_distance_l2424_242454

theorem kevin_hopping_distance :
  let hop_distance (n : Nat) : ℚ :=
    let factor : ℚ := (3/4 : ℚ)^n
    1/4 * factor
  let total_distance : ℚ :=
    (hop_distance 0 + hop_distance 1 + hop_distance 2 + hop_distance 3 + hop_distance 4 + hop_distance 5)
  total_distance = 39677 / 40960 :=
by
  sorry

end NUMINAMATH_GPT_kevin_hopping_distance_l2424_242454


namespace NUMINAMATH_GPT_target_hit_probability_l2424_242488

-- Defining the probabilities for A, B, and C hitting the target.
def P_A_hit := 1 / 2
def P_B_hit := 1 / 3
def P_C_hit := 1 / 4

-- Defining the probability that A, B, and C miss the target.
def P_A_miss := 1 - P_A_hit
def P_B_miss := 1 - P_B_hit
def P_C_miss := 1 - P_C_hit

-- Calculating the combined probability that none of them hit the target.
def P_none_hit := P_A_miss * P_B_miss * P_C_miss

-- Now, calculating the probability that at least one of them hits the target.
def P_hit := 1 - P_none_hit

-- Statement of the theorem.
theorem target_hit_probability : P_hit = 3 / 4 := by
  sorry

end NUMINAMATH_GPT_target_hit_probability_l2424_242488


namespace NUMINAMATH_GPT_geometric_sequence_a2_a6_l2424_242426

variable (a : ℕ → ℝ) (r : ℝ) (a1 : ℝ)
variable (a_geom_seq : ∀ n, a n = a1 * r^(n-1))
variable (h_a4 : a 4 = 4)

theorem geometric_sequence_a2_a6 : a 2 * a 6 = 16 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_geometric_sequence_a2_a6_l2424_242426


namespace NUMINAMATH_GPT_polygon_sides_l2424_242446

theorem polygon_sides (n : ℕ) (h : 144 * n = 180 * (n - 2)) : n = 10 :=
by { sorry }

end NUMINAMATH_GPT_polygon_sides_l2424_242446


namespace NUMINAMATH_GPT_jina_mascots_l2424_242441

variables (x y z x_new Total : ℕ)

def mascots_problem :=
  (y = 3 * x) ∧
  (x_new = x + 2 * y) ∧
  (z = 2 * y) ∧
  (Total = x_new + y + z) →
  Total = 16 * x

-- The statement only, no proof is required
theorem jina_mascots : mascots_problem x y z x_new Total := sorry

end NUMINAMATH_GPT_jina_mascots_l2424_242441


namespace NUMINAMATH_GPT_ratio_of_distances_l2424_242458

-- Definitions based on conditions in a)
variables (x y w : ℝ) (h_nonneg_x : 0 ≤ x) (h_nonneg_y : 0 ≤ y) (h_nonneg_w : 0 ≤ w)
variables (h_w_ne_zero : w ≠ 0) (h_y_ne_zero : y ≠ 0) (h_eq_times : y / w = x / w + (x + y) / (9 * w))

-- The proof statement
theorem ratio_of_distances (x y w : ℝ) (h_nonneg_x : 0 ≤ x) (h_nonneg_y : 0 ≤ y)
  (h_nonneg_w : 0 ≤ w) (h_w_ne_zero : w ≠ 0) (h_y_ne_zero : y ≠ 0)
  (h_eq_times : y / w = x / w + (x + y) / (9 * w)) :
  x / y = 4 / 5 :=
sorry

end NUMINAMATH_GPT_ratio_of_distances_l2424_242458


namespace NUMINAMATH_GPT_cake_slices_l2424_242474

open Nat

theorem cake_slices (S : ℕ) (h1 : 2 * S - 12 = 10) : S = 8 := by
  sorry

end NUMINAMATH_GPT_cake_slices_l2424_242474


namespace NUMINAMATH_GPT_solve_system_l2424_242402

theorem solve_system (s t : ℚ) (h1 : 7 * s + 6 * t = 156) (h2 : s = t / 2 + 3) : s = 192 / 19 :=
sorry

end NUMINAMATH_GPT_solve_system_l2424_242402


namespace NUMINAMATH_GPT_tan_product_l2424_242443

theorem tan_product :
    (Real.tan (Real.pi / 9)) * (Real.tan (2 * Real.pi / 9)) * (Real.tan (4 * Real.pi / 9)) = 1 / 3 :=
by sorry

end NUMINAMATH_GPT_tan_product_l2424_242443


namespace NUMINAMATH_GPT_prime_even_intersection_l2424_242433

-- Define P as the set of prime numbers
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def P : Set ℕ := { n | is_prime n }

-- Define Q as the set of even numbers
def Q : Set ℕ := { n | n % 2 = 0 }

-- Statement to prove
theorem prime_even_intersection : P ∩ Q = {2} :=
by
  sorry

end NUMINAMATH_GPT_prime_even_intersection_l2424_242433


namespace NUMINAMATH_GPT_soccer_league_equation_l2424_242401

noncomputable def equation_represents_soccer_league (x : ℕ) : Prop :=
  ∀ x : ℕ, (x * (x - 1)) / 2 = 50

theorem soccer_league_equation (x : ℕ) (h : equation_represents_soccer_league x) :
  (x * (x - 1)) / 2 = 50 :=
  by sorry

end NUMINAMATH_GPT_soccer_league_equation_l2424_242401


namespace NUMINAMATH_GPT_total_fertilizer_used_l2424_242484

def daily_fertilizer := 3
def num_days := 12
def extra_final_day := 6

theorem total_fertilizer_used : 
    (daily_fertilizer * num_days + (daily_fertilizer + extra_final_day)) = 45 :=
by
  sorry

end NUMINAMATH_GPT_total_fertilizer_used_l2424_242484


namespace NUMINAMATH_GPT_lcm_18_27_l2424_242422

theorem lcm_18_27 : Nat.lcm 18 27 = 54 :=
by {
  sorry
}

end NUMINAMATH_GPT_lcm_18_27_l2424_242422


namespace NUMINAMATH_GPT_solution_set_inequality_l2424_242400

theorem solution_set_inequality : {x : ℝ | (x - 2) * (1 - 2 * x) ≥ 0} = {x : ℝ | 1/2 ≤ x ∧ x ≤ 2} :=
by
  sorry  -- Proof to be provided

end NUMINAMATH_GPT_solution_set_inequality_l2424_242400


namespace NUMINAMATH_GPT_sum_of_primes_final_sum_l2424_242457

theorem sum_of_primes (p : ℕ) (hp : Nat.Prime p) :
  (¬ ∃ x : ℤ, 5 * (10 * x + 2) ≡ 6 [ZMOD p]) →
  p = 2 ∨ p = 5 :=
sorry

theorem final_sum :
  (∀ p : ℕ, Nat.Prime p → (¬ ∃ x : ℤ, 5 * (10 * x + 2) ≡ 6 [ZMOD p]) → p = 2 ∨ p = 5) →
  (2 + 5 = 7) :=
sorry

end NUMINAMATH_GPT_sum_of_primes_final_sum_l2424_242457


namespace NUMINAMATH_GPT_total_pairs_of_jeans_purchased_l2424_242456

-- Definitions based on the problem conditions
def price_fox : ℝ := 15
def price_pony : ℝ := 18
def discount_save : ℝ := 8.64
def pairs_fox : ℕ := 3
def pairs_pony : ℕ := 2
def sum_discount_rate : ℝ := 0.22
def discount_rate_pony : ℝ := 0.13999999999999993

-- Lean 4 statement to prove the total number of pairs of jeans purchased
theorem total_pairs_of_jeans_purchased :
  pairs_fox + pairs_pony = 5 :=
by
  sorry

end NUMINAMATH_GPT_total_pairs_of_jeans_purchased_l2424_242456


namespace NUMINAMATH_GPT_beavers_working_l2424_242482

theorem beavers_working (a b : ℝ) (h₁ : a = 2.0) (h₂ : b = 1.0) : a + b = 3.0 := 
by 
  rw [h₁, h₂]
  norm_num

end NUMINAMATH_GPT_beavers_working_l2424_242482


namespace NUMINAMATH_GPT_combined_profit_is_14000_l2424_242434

-- Define constants and conditions
def center1_daily_packages : ℕ := 10000
def daily_profit_per_package : ℝ := 0.05
def center2_multiplier : ℕ := 3
def days_per_week : ℕ := 7

-- Define the profit for the first center
def center1_daily_profit : ℝ := center1_daily_packages * daily_profit_per_package

-- Define the packages processed by the second center
def center2_daily_packages : ℕ := center1_daily_packages * center2_multiplier

-- Define the profit for the second center
def center2_daily_profit : ℝ := center2_daily_packages * daily_profit_per_package

-- Define the combined daily profit
def combined_daily_profit : ℝ := center1_daily_profit + center2_daily_profit

-- Define the combined weekly profit
def combined_weekly_profit : ℝ := combined_daily_profit * days_per_week

-- Prove that the combined weekly profit is $14,000
theorem combined_profit_is_14000 : combined_weekly_profit = 14000 := by
  -- You can replace sorry with the steps to solve the proof.
  sorry

end NUMINAMATH_GPT_combined_profit_is_14000_l2424_242434

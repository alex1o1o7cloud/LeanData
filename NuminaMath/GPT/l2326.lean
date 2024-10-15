import Mathlib

namespace NUMINAMATH_GPT_yanna_baked_butter_cookies_in_morning_l2326_232646

-- Define the conditions
def biscuits_morning : ℕ := 40
def biscuits_afternoon : ℕ := 20
def cookies_afternoon : ℕ := 10
def total_more_biscuits : ℕ := 30

-- Define the statement to be proved
theorem yanna_baked_butter_cookies_in_morning (B : ℕ) : 
  (biscuits_morning + biscuits_afternoon = (B + cookies_afternoon) + total_more_biscuits) → B = 20 :=
by
  sorry

end NUMINAMATH_GPT_yanna_baked_butter_cookies_in_morning_l2326_232646


namespace NUMINAMATH_GPT_compute_f_1986_l2326_232630

noncomputable def f : ℕ → ℤ := sorry

axiom f_defined_for_nonneg_integers : ∀ x : ℕ, ∃ y : ℤ, f x = y
axiom f_one : f 1 = 1
axiom f_functional_equation : ∀ (a b : ℕ), f (a + b) = f a + f b - 2 * f (a * b)

theorem compute_f_1986 : f 1986 = 0 :=
  sorry

end NUMINAMATH_GPT_compute_f_1986_l2326_232630


namespace NUMINAMATH_GPT_solve_quadratic_substitution_l2326_232618

theorem solve_quadratic_substitution (x : ℝ) : 
  (x^2 + x)^2 - 4*(x^2 + x) - 12 = 0 ↔ x = -3 ∨ x = 2 := 
by sorry

end NUMINAMATH_GPT_solve_quadratic_substitution_l2326_232618


namespace NUMINAMATH_GPT_intersection_M_N_l2326_232629

open Set

def M := {x : ℝ | x^2 < 4}
def N := {x : ℝ | x^2 - 2*x - 3 < 0}
def intersection_sets := {x : ℝ | -1 < x ∧ x < 2}

theorem intersection_M_N : M ∩ N = intersection_sets :=
  sorry

end NUMINAMATH_GPT_intersection_M_N_l2326_232629


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l2326_232673

theorem sufficient_but_not_necessary_condition
  (a : ℝ) :
  (a = 2 → (a - 1) * (a - 2) = 0)
  ∧ (∃ a : ℝ, (a - 1) * (a - 2) = 0 ∧ a ≠ 2) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l2326_232673


namespace NUMINAMATH_GPT_extreme_values_a_4_find_a_minimum_minus_5_l2326_232681

noncomputable def f (x a : ℝ) : ℝ := 2 * x^2 - a * x + 5

theorem extreme_values_a_4 :
  (∀ x, x ∈ Set.Icc (-1:ℝ) 2 -> f x 4 ≤ 11) ∧ (∃ x, x ∈ Set.Icc (-1:ℝ) 2 ∧ f x 4 = 11) ∧
  (∀ x, x ∈ Set.Icc (-1:ℝ) 2 -> f x 4 ≥ 3) ∧ (∃ x, x ∈ Set.Icc (-1:ℝ) 2 ∧ f x 4 = 3) :=
  sorry

theorem find_a_minimum_minus_5 :
  ∀ (a : ℝ), (∃ x, x ∈ Set.Icc (-1:ℝ) 2 ∧ f x a = -5) -> (a = -12 ∨ a = 9) :=
  sorry

end NUMINAMATH_GPT_extreme_values_a_4_find_a_minimum_minus_5_l2326_232681


namespace NUMINAMATH_GPT_Victor_worked_hours_l2326_232695

theorem Victor_worked_hours (h : ℕ) (pay_rate : ℕ) (total_earnings : ℕ) 
  (H1 : pay_rate = 6) 
  (H2 : total_earnings = 60) 
  (H3 : 2 * (pay_rate * h) = total_earnings): 
  h = 5 := 
by 
  sorry

end NUMINAMATH_GPT_Victor_worked_hours_l2326_232695


namespace NUMINAMATH_GPT_makenna_garden_larger_by_132_l2326_232667

-- Define the dimensions of Karl's garden
def length_karl : ℕ := 22
def width_karl : ℕ := 50

-- Define the dimensions of Makenna's garden including the walking path
def length_makenna_total : ℕ := 30
def width_makenna_total : ℕ := 46
def walking_path_width : ℕ := 1

-- Define the area calculation functions
def area (length : ℕ) (width : ℕ) : ℕ := length * width

-- Calculate the areas
def area_karl : ℕ := area length_karl width_karl
def area_makenna : ℕ := area (length_makenna_total - 2 * walking_path_width) (width_makenna_total - 2 * walking_path_width)

-- Define the theorem to prove
theorem makenna_garden_larger_by_132 :
  area_makenna = area_karl + 132 :=
by
  -- We skip the proof part
  sorry

end NUMINAMATH_GPT_makenna_garden_larger_by_132_l2326_232667


namespace NUMINAMATH_GPT_interest_calculation_years_l2326_232638

theorem interest_calculation_years (P r : ℝ) (diff : ℝ) (n : ℕ) 
  (hP : P = 3600) (hr : r = 0.10) (hdiff : diff = 36) 
  (h_eq : P * (1 + r)^n - P - (P * r * n) = diff) : n = 2 :=
sorry

end NUMINAMATH_GPT_interest_calculation_years_l2326_232638


namespace NUMINAMATH_GPT_Tiffany_bags_l2326_232620

theorem Tiffany_bags (x : ℕ) 
  (h1 : 8 = x + 1) : 
  x = 7 :=
by
  sorry

end NUMINAMATH_GPT_Tiffany_bags_l2326_232620


namespace NUMINAMATH_GPT_find_y_l2326_232661

theorem find_y (x y : ℝ) (h1 : (100 + 200 + 300 + x) / 4 = 250) (h2 : (300 + 150 + 100 + x + y) / 5 = 200) : y = 50 :=
by
  sorry

end NUMINAMATH_GPT_find_y_l2326_232661


namespace NUMINAMATH_GPT_area_shaded_smaller_dodecagon_area_in_circle_l2326_232683

-- Part (a) statement
theorem area_shaded_smaller (dodecagon_area : ℝ) (shaded_area : ℝ) 
  (h : shaded_area = (1 / 12) * dodecagon_area) :
  shaded_area = dodecagon_area / 12 :=
sorry

-- Part (b) statement
theorem dodecagon_area_in_circle (r : ℝ) (A : ℝ) 
  (h : r = 1) (h' : A = (1 / 2) * 12 * r ^ 2 * Real.sin (2 * Real.pi / 12)) :
  A = 3 :=
sorry

end NUMINAMATH_GPT_area_shaded_smaller_dodecagon_area_in_circle_l2326_232683


namespace NUMINAMATH_GPT_linear_function_quadrants_l2326_232611

theorem linear_function_quadrants (k : ℝ) :
  (k - 3 > 0) ∧ (-k + 2 < 0) → k > 3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_linear_function_quadrants_l2326_232611


namespace NUMINAMATH_GPT_patricia_earns_more_than_jose_l2326_232623

noncomputable def jose_final_amount : ℝ :=
  50000 * (1 + 0.04)^2

noncomputable def patricia_final_amount : ℝ :=
  50000 * (1 + 0.01)^8

theorem patricia_earns_more_than_jose :
  patricia_final_amount - jose_final_amount = 63 :=
by
  -- from solution steps
  /-
  jose_final_amount = 50000 * (1 + 0.04)^2 = 54080
  patricia_final_amount = 50000 * (1 + 0.01)^8 ≈ 54143
  patricia_final_amount - jose_final_amount ≈ 63
  -/
  sorry

end NUMINAMATH_GPT_patricia_earns_more_than_jose_l2326_232623


namespace NUMINAMATH_GPT_fujian_provincial_games_distribution_count_l2326_232656

theorem fujian_provincial_games_distribution_count 
  (staff_members : Finset String)
  (locations : Finset String)
  (A B C D E F : String)
  (A_in_B : A ∈ staff_members)
  (B_in_B : B ∈ staff_members)
  (C_in_B : C ∈ staff_members)
  (D_in_B : D ∈ staff_members)
  (E_in_B : E ∈ staff_members)
  (F_in_B : F ∈ staff_members)
  (locations_count : locations.card = 2)
  (staff_count : staff_members.card = 6)
  (must_same_group : ∀ g₁ g₂ : Finset String, A ∈ g₁ → B ∈ g₁ → g₁ ∪ g₂ = staff_members)
  (min_two_people : ∀ g : Finset String, 2 ≤ g.card) :
  ∃ distrib_methods : ℕ, distrib_methods = 22 := 
by
  sorry

end NUMINAMATH_GPT_fujian_provincial_games_distribution_count_l2326_232656


namespace NUMINAMATH_GPT_negation_equiv_l2326_232632

theorem negation_equiv (x : ℝ) : ¬ (x^2 - 1 < 0) ↔ (x^2 - 1 ≥ 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_equiv_l2326_232632


namespace NUMINAMATH_GPT_equi_partite_complex_number_a_l2326_232672

-- A complex number z = 1 + (a-1)i
def z (a : ℝ) : ℂ := ⟨1, a - 1⟩

-- Definition of an equi-partite complex number
def is_equi_partite (z : ℂ) : Prop := z.re = z.im

-- The theorem to prove
theorem equi_partite_complex_number_a (a : ℝ) : is_equi_partite (z a) ↔ a = 2 := 
by
  sorry

end NUMINAMATH_GPT_equi_partite_complex_number_a_l2326_232672


namespace NUMINAMATH_GPT_nested_expression_value_l2326_232604

theorem nested_expression_value : 
  4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4)))))))) = 87380 :=
by 
  sorry

end NUMINAMATH_GPT_nested_expression_value_l2326_232604


namespace NUMINAMATH_GPT_turns_per_minute_l2326_232693

theorem turns_per_minute (x : ℕ) (h₁ : x > 0) (h₂ : 60 / x = (60 / (x + 5)) + 2) :
  60 / x = 6 ∧ 60 / (x + 5) = 4 :=
by sorry

end NUMINAMATH_GPT_turns_per_minute_l2326_232693


namespace NUMINAMATH_GPT_num_second_grade_students_is_80_l2326_232607

def ratio_fst : ℕ := 5
def ratio_snd : ℕ := 4
def ratio_trd : ℕ := 3
def total_students : ℕ := 240

def second_grade : ℕ := (ratio_snd * total_students) / (ratio_fst + ratio_snd + ratio_trd)

theorem num_second_grade_students_is_80 :
  second_grade = 80 := 
sorry

end NUMINAMATH_GPT_num_second_grade_students_is_80_l2326_232607


namespace NUMINAMATH_GPT_fractions_simplify_to_prime_denominator_2023_l2326_232647

def num_fractions_simplifying_to_prime_denominator (n: ℕ) (p q: ℕ) : ℕ :=
  let multiples (m: ℕ) : ℕ := (n - 1) / m
  multiples p + multiples (p * q)

theorem fractions_simplify_to_prime_denominator_2023 :
  num_fractions_simplifying_to_prime_denominator 2023 17 7 = 22 :=
by
  sorry

end NUMINAMATH_GPT_fractions_simplify_to_prime_denominator_2023_l2326_232647


namespace NUMINAMATH_GPT_find_f_4_l2326_232659

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation : ∀ x : ℝ, f x + 3 * f (1 - x) = 4 * x ^ 2

theorem find_f_4 : f 4 = 5.5 :=
by
  sorry

end NUMINAMATH_GPT_find_f_4_l2326_232659


namespace NUMINAMATH_GPT_how_many_strawberries_did_paul_pick_l2326_232671

-- Here, we will define the known quantities
def original_strawberries : Nat := 28
def total_strawberries : Nat := 63

-- The statement to prove
theorem how_many_strawberries_did_paul_pick : total_strawberries - original_strawberries = 35 :=
by
  unfold total_strawberries
  unfold original_strawberries
  calc
    63 - 28 = 35 := by norm_num

end NUMINAMATH_GPT_how_many_strawberries_did_paul_pick_l2326_232671


namespace NUMINAMATH_GPT_max_piece_length_total_pieces_l2326_232621

-- Definitions based on the problem's conditions
def length1 : ℕ := 42
def length2 : ℕ := 63
def gcd_length : ℕ := Nat.gcd length1 length2

-- Theorem statements based on the realized correct answers
theorem max_piece_length (h1 : length1 = 42) (h2 : length2 = 63) :
  gcd_length = 21 := by
  sorry

theorem total_pieces (h1 : length1 = 42) (h2 : length2 = 63) :
  (length1 / gcd_length) + (length2 / gcd_length) = 5 := by
  sorry

end NUMINAMATH_GPT_max_piece_length_total_pieces_l2326_232621


namespace NUMINAMATH_GPT_multiplication_72519_9999_l2326_232642

theorem multiplication_72519_9999 :
  72519 * 9999 = 725117481 :=
by
  sorry

end NUMINAMATH_GPT_multiplication_72519_9999_l2326_232642


namespace NUMINAMATH_GPT_linear_function_does_not_pass_third_quadrant_l2326_232601

/-
Given an inverse proportion function \( y = \frac{a^2 + 1}{x} \), where \( a \) is a constant, and given two points \( (x_1, y_1) \) and \( (x_2, y_2) \) on the same branch of this function, 
with \( b = (x_1 - x_2)(y_1 - y_2) \), prove that the graph of the linear function \( y = bx - b \) does not pass through the third quadrant.
-/

theorem linear_function_does_not_pass_third_quadrant 
  (a x1 x2 : ℝ) 
  (y1 y2 : ℝ)
  (h1 : y1 = (a^2 + 1) / x1) 
  (h2 : y2 = (a^2 + 1) / x2) 
  (h3 : b = (x1 - x2) * (y1 - y2)) : 
  ∃ b, ∀ x y : ℝ, (y = b * x - b) → (x < 0 ∧ y > 0) ∨ (x > 0 ∧ y < 0) :=
by 
  sorry

end NUMINAMATH_GPT_linear_function_does_not_pass_third_quadrant_l2326_232601


namespace NUMINAMATH_GPT_line_not_tangent_if_only_one_common_point_l2326_232699

theorem line_not_tangent_if_only_one_common_point (l p : ℝ) :
  (∃ y, y^2 = 2 * p * l) ∧ ¬ (∃ x : ℝ, y = l ∧ y^2 = 2 * p * x) := 
  sorry

end NUMINAMATH_GPT_line_not_tangent_if_only_one_common_point_l2326_232699


namespace NUMINAMATH_GPT_order_of_logs_l2326_232631

open Real

noncomputable def a := log 10 / log 5
noncomputable def b := log 12 / log 6
noncomputable def c := 1 + log 2 / log 7

theorem order_of_logs : a > b ∧ b > c :=
by
  sorry

end NUMINAMATH_GPT_order_of_logs_l2326_232631


namespace NUMINAMATH_GPT_odd_even_shift_composition_l2326_232679

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def is_even_function_shifted (f : ℝ → ℝ) (shift : ℝ) : Prop :=
  ∀ x : ℝ, f (x + shift) = f (-x + shift)

theorem odd_even_shift_composition
  (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_even_shift : is_even_function_shifted f 3)
  (h_f1 : f 1 = 1) :
  f 6 + f 11 = -1 := by
  sorry

end NUMINAMATH_GPT_odd_even_shift_composition_l2326_232679


namespace NUMINAMATH_GPT_exponent_problem_l2326_232675

theorem exponent_problem (a : ℝ) (m n : ℕ) (h1 : a ^ m = 3) (h2 : a ^ n = 2) : a ^ (m - 2 * n) = 3 / 4 := by
  sorry

end NUMINAMATH_GPT_exponent_problem_l2326_232675


namespace NUMINAMATH_GPT_value_of_expression_l2326_232654

theorem value_of_expression (x y : ℝ) (h₁ : x * y = -3) (h₂ : x + y = -4) :
  x^2 + 3 * x * y + y^2 = 13 :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l2326_232654


namespace NUMINAMATH_GPT_find_breadth_of_rectangle_l2326_232653

noncomputable def breadth_of_rectangle (A : ℝ) (length_to_breadth_ratio : ℝ) (breadth : ℝ) : Prop :=
  A = length_to_breadth_ratio * breadth * breadth → breadth = 20

-- Now we can state the theorem.
theorem find_breadth_of_rectangle (A : ℝ) (length_to_breadth_ratio : ℝ) : breadth_of_rectangle A length_to_breadth_ratio 20 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_find_breadth_of_rectangle_l2326_232653


namespace NUMINAMATH_GPT_pythagorean_triple_fits_l2326_232698

theorem pythagorean_triple_fits 
  (k : ℤ) (n : ℤ) : 
  (∃ k, (n = 5 * k ∨ n = 12 * k ∨ n = 13 * k) ∧ 
      (n = 62 ∨ n = 96 ∨ n = 120 ∨ n = 91 ∨ n = 390)) ↔ 
      (n = 120 ∨ n = 91) := by 
  sorry

end NUMINAMATH_GPT_pythagorean_triple_fits_l2326_232698


namespace NUMINAMATH_GPT_range_of_y_function_l2326_232696

def range_of_function : Set ℝ :=
  {y : ℝ | ∃ (x : ℝ), x ≠ -2 ∧ y = (x^2 + 5*x + 6)/(x+2)}

theorem range_of_y_function :
  range_of_function = {y : ℝ | y ≠ 1} :=
by
  sorry

end NUMINAMATH_GPT_range_of_y_function_l2326_232696


namespace NUMINAMATH_GPT_probability_of_at_least_one_three_l2326_232612

def probability_at_least_one_three_shows : ℚ :=
  let total_outcomes : ℚ := 64
  let favorable_outcomes : ℚ := 15
  favorable_outcomes / total_outcomes

theorem probability_of_at_least_one_three (a b : ℕ) (ha : 1 ≤ a ∧ a ≤ 8) (hb : 1 ≤ b ∧ b ≤ 8) :
    (a = 3 ∨ b = 3) → probability_at_least_one_three_shows = 15 / 64 := by
  sorry

end NUMINAMATH_GPT_probability_of_at_least_one_three_l2326_232612


namespace NUMINAMATH_GPT_inverse_function_fixed_point_l2326_232652

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the condition that graph of y = f(x-1) passes through the point (1, 2)
def passes_through (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  f (a - 1) = b

-- State the main theorem to prove
theorem inverse_function_fixed_point {f : ℝ → ℝ} (h : passes_through f 1 2) :
  ∃ x, x = 2 ∧ f x = 0 :=
sorry

end NUMINAMATH_GPT_inverse_function_fixed_point_l2326_232652


namespace NUMINAMATH_GPT_number_of_odd_palindromes_l2326_232660

def is_palindrome (n : ℕ) : Prop :=
  let d0 := n % 10
  let d1 := (n / 10) % 10
  let d2 := n / 100
  n < 1000 ∧ n >= 100 ∧ d0 = d2

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

theorem number_of_odd_palindromes : ∃ n : ℕ, is_palindrome n ∧ is_odd n → n = 50 :=
by
  sorry

end NUMINAMATH_GPT_number_of_odd_palindromes_l2326_232660


namespace NUMINAMATH_GPT_shortest_fence_length_l2326_232684

-- We define the conditions given in the problem.
def triangle_side_length : ℕ := 50
def number_of_dotted_lines : ℕ := 13

-- We need to prove that the shortest total length of the fences required to protect all the cabbage from goats equals 650 meters.
theorem shortest_fence_length : number_of_dotted_lines * triangle_side_length = 650 :=
by
  -- The proof steps are omitted as per instructions.
  sorry

end NUMINAMATH_GPT_shortest_fence_length_l2326_232684


namespace NUMINAMATH_GPT_pure_acid_total_is_3_8_l2326_232680

/-- Volume of Solution A in liters -/
def volume_A : ℝ := 8

/-- Concentration of Solution A (in decimals, i.e., 20% as 0.20) -/
def concentration_A : ℝ := 0.20

/-- Volume of Solution B in liters -/
def volume_B : ℝ := 5

/-- Concentration of Solution B (in decimals, i.e., 35% as 0.35) -/
def concentration_B : ℝ := 0.35

/-- Volume of Solution C in liters -/
def volume_C : ℝ := 3

/-- Concentration of Solution C (in decimals, i.e., 15% as 0.15) -/
def concentration_C : ℝ := 0.15

/-- Total amount of pure acid in the resulting mixture -/
def total_pure_acid : ℝ :=
  (volume_A * concentration_A) +
  (volume_B * concentration_B) +
  (volume_C * concentration_C)

theorem pure_acid_total_is_3_8 : total_pure_acid = 3.8 := by
  sorry

end NUMINAMATH_GPT_pure_acid_total_is_3_8_l2326_232680


namespace NUMINAMATH_GPT_solution_set_of_inequality_l2326_232627

noncomputable def f : ℝ → ℝ := sorry

axiom ax1 : ∀ (x1 x2 : ℝ), (0 < x1) → (0 < x2) → (x1 ≠ x2) → 
  (x1 * f x2 - x2 * f x1) / (x2 - x1) > 1

axiom ax2 : f 3 = 2

theorem solution_set_of_inequality :
  {x : ℝ | 0 < x ∧ f x < x - 1} = {x : ℝ | 0 < x ∧ x < 3} := by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l2326_232627


namespace NUMINAMATH_GPT_travelers_cross_river_l2326_232617

variables (traveler1 traveler2 traveler3 : ℕ)  -- weights of travelers
variable (raft_capacity : ℕ)  -- maximum carrying capacity of the raft

-- Given conditions
def conditions :=
  traveler1 = 3 ∧ traveler2 = 3 ∧ traveler3 = 5 ∧ raft_capacity = 7

-- Prove that the travelers can all cross the river successfully
theorem travelers_cross_river :
  conditions traveler1 traveler2 traveler3 raft_capacity →
  (traveler1 + traveler2 ≤ raft_capacity) ∧
  (traveler1 ≤ raft_capacity) ∧
  (traveler3 ≤ raft_capacity) ∧
  (traveler1 + traveler2 ≤ raft_capacity) →
  true :=
by
  intros h_conditions h_validity
  sorry

end NUMINAMATH_GPT_travelers_cross_river_l2326_232617


namespace NUMINAMATH_GPT_set_difference_A_B_l2326_232664

-- Defining the sets A and B
def setA : Set ℝ := { x : ℝ | abs (4 * x - 1) > 9 }
def setB : Set ℝ := { x : ℝ | x >= 0 }

-- The theorem stating the result of set difference A - B
theorem set_difference_A_B : (setA \ setB) = { x : ℝ | x > 5/2 } :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_set_difference_A_B_l2326_232664


namespace NUMINAMATH_GPT_sum_of_squares_l2326_232636

theorem sum_of_squares (a b c : ℝ) (h1 : a * b + a * c + b * c = 131) (h2 : a + b + c = 22) : a^2 + b^2 + c^2 = 222 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_l2326_232636


namespace NUMINAMATH_GPT_hyperbola_eccentricity_l2326_232635

theorem hyperbola_eccentricity (a b c e : ℝ) 
  (h_eq1 : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1)
  (h_eq2 : c = Real.sqrt (a^2 + b^2))
  (h_dist : ∀ x, x = b * c / Real.sqrt (a^2 + b^2))
  (h_eq3 : a = b) :
  e = Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_l2326_232635


namespace NUMINAMATH_GPT_stock_profit_percentage_l2326_232616

theorem stock_profit_percentage 
  (total_stock : ℝ) (total_loss : ℝ) (profit_percentage : ℝ) (loss_percentage : ℝ)
  (percentage_sold_at_profit : ℝ) :
  total_stock = 12499.99 →
  total_loss = 500 →
  profit_percentage = 0.20 →
  loss_percentage = 0.10 →
  (0.10 * ((100 - percentage_sold_at_profit) / 100) * 12499.99) - (0.20 * (percentage_sold_at_profit / 100) * 12499.99) = 500 →
  percentage_sold_at_profit = 20 :=
sorry

end NUMINAMATH_GPT_stock_profit_percentage_l2326_232616


namespace NUMINAMATH_GPT_rectangle_max_area_l2326_232608

theorem rectangle_max_area (w : ℝ) (h : ℝ) (hw : h = 2 * w) (perimeter : 2 * (w + h) = 40) :
  w * h = 800 / 9 := 
by
  -- Given: h = 2w and 2(w + h) = 40
  -- We need to prove that the area A = wh = 800/9
  sorry

end NUMINAMATH_GPT_rectangle_max_area_l2326_232608


namespace NUMINAMATH_GPT_calculation_1500_increased_by_45_percent_l2326_232637

theorem calculation_1500_increased_by_45_percent :
  1500 * (1 + 45 / 100) = 2175 := 
by
  sorry

end NUMINAMATH_GPT_calculation_1500_increased_by_45_percent_l2326_232637


namespace NUMINAMATH_GPT_find_third_side_l2326_232610

theorem find_third_side (a b : ℝ) (c : ℕ) 
  (h1 : a = 3.14)
  (h2 : b = 0.67)
  (h_triangle_ineq : a + b > ↑c ∧ a + ↑c > b ∧ b + ↑c > a) : 
  c = 3 := 
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_find_third_side_l2326_232610


namespace NUMINAMATH_GPT_initial_mean_calculated_l2326_232600

theorem initial_mean_calculated (M : ℝ) (h1 : 25 * M - 35 = 25 * 191.4 - 35) : M = 191.4 := 
  sorry

end NUMINAMATH_GPT_initial_mean_calculated_l2326_232600


namespace NUMINAMATH_GPT_rows_seating_8_people_l2326_232645

theorem rows_seating_8_people (x : ℕ) (h₁ : x ≡ 4 [MOD 7]) (h₂ : x ≤ 6) :
  x = 4 := by
  sorry

end NUMINAMATH_GPT_rows_seating_8_people_l2326_232645


namespace NUMINAMATH_GPT_bug_meeting_point_l2326_232649
-- Import the necessary library

-- Define the side lengths of the triangle
variables (DE EF FD : ℝ)
variables (bugs_meet : ℝ)

-- State the conditions and the result
theorem bug_meeting_point
  (h1 : DE = 6)
  (h2 : EF = 8)
  (h3 : FD = 10)
  (h4 : bugs_meet = 1 / 2 * (DE + EF + FD)) :
  bugs_meet - DE = 6 :=
by
  sorry

end NUMINAMATH_GPT_bug_meeting_point_l2326_232649


namespace NUMINAMATH_GPT_marias_workday_end_time_l2326_232643

theorem marias_workday_end_time :
  ∀ (start_time : ℕ) (lunch_time : ℕ) (work_duration : ℕ) (lunch_break : ℕ) (total_work_time : ℕ),
  start_time = 8 ∧ lunch_time = 13 ∧ work_duration = 8 ∧ lunch_break = 1 →
  (total_work_time = work_duration - (lunch_time - start_time - lunch_break)) →
  lunch_time + 1 + (work_duration - (lunch_time - start_time)) = 17 :=
by
  sorry

end NUMINAMATH_GPT_marias_workday_end_time_l2326_232643


namespace NUMINAMATH_GPT_find_number_l2326_232676

theorem find_number (x : ℝ) (h : (5/3) * x = 45) : x = 27 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l2326_232676


namespace NUMINAMATH_GPT_min_n_for_triangle_pattern_l2326_232655

/-- 
There are two types of isosceles triangles with a waist length of 1:
-  Type 1: An acute isosceles triangle with a vertex angle of 30 degrees.
-  Type 2: A right isosceles triangle with a vertex angle of 90 degrees.
They are placed around a point in a clockwise direction in a sequence such that:
- The 1st and 2nd are acute isosceles triangles (30 degrees),
- The 3rd is a right isosceles triangle (90 degrees),
- The 4th and 5th are acute isosceles triangles (30 degrees),
- The 6th is a right isosceles triangle (90 degrees), and so on.

Prove that the minimum value of n such that the nth triangle coincides exactly with
the 1st triangle is 23.
-/
theorem min_n_for_triangle_pattern : ∃ n : ℕ, n = 23 ∧ (∀ m < 23, m ≠ 23) :=
sorry

end NUMINAMATH_GPT_min_n_for_triangle_pattern_l2326_232655


namespace NUMINAMATH_GPT_divides_difference_l2326_232602

theorem divides_difference (n : ℕ) (h_composite : ∃ m k : ℕ, m > 1 ∧ k > 1 ∧ n = m * k) : 
  6 ∣ ((n^2)^3 - n^2) := 
sorry

end NUMINAMATH_GPT_divides_difference_l2326_232602


namespace NUMINAMATH_GPT_geometric_sequence_k_value_l2326_232657

theorem geometric_sequence_k_value
  (k : ℤ)
  (S : ℕ → ℤ)
  (a : ℕ → ℤ)
  (h1 : ∀ n, S n = 3 * 2^n + k)
  (h2 : ∀ n, n ≥ 2 → a n = S n - S (n - 1))
  (h3 : ∃ r, ∀ n, a (n + 1) = r * a n) : k = -3 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_k_value_l2326_232657


namespace NUMINAMATH_GPT_circle_center_radius_l2326_232688

theorem circle_center_radius :
  ∀ (x y : ℝ), (x + 1) ^ 2 + (y - 2) ^ 2 = 9 ↔ (x = -1 ∧ y = 2 ∧ ∃ r : ℝ, r = 3) :=
by
  sorry

end NUMINAMATH_GPT_circle_center_radius_l2326_232688


namespace NUMINAMATH_GPT_area_triangle_ABC_l2326_232692

noncomputable def area_of_triangle_ABC : ℝ :=
  let base_AB : ℝ := 6 - 0
  let height_AB : ℝ := 2 - 0
  let base_BC : ℝ := 6 - 3
  let height_BC : ℝ := 8 - 0
  let base_CA : ℝ := 3 - 0
  let height_CA : ℝ := 8 - 2
  let area_ratio : ℝ := 1 / 2
  let area_I' : ℝ := area_ratio * base_AB * height_AB
  let area_II' : ℝ := area_ratio * 8 * 6
  let area_III' : ℝ := area_ratio * 8 * 3
  let total_small_triangles : ℝ := area_I' + area_II' + area_III'
  let total_area_rectangle : ℝ := 6 * 8
  total_area_rectangle - total_small_triangles

theorem area_triangle_ABC : area_of_triangle_ABC = 6 := 
by
  sorry

end NUMINAMATH_GPT_area_triangle_ABC_l2326_232692


namespace NUMINAMATH_GPT_percentage_error_in_area_l2326_232626

theorem percentage_error_in_area (s : ℝ) (h : s ≠ 0) :
  let s' := 1.02 * s
  let A := s^2
  let A' := s'^2
  ((A' - A) / A) * 100 = 4.04 := by
  sorry

end NUMINAMATH_GPT_percentage_error_in_area_l2326_232626


namespace NUMINAMATH_GPT_remove_one_and_average_l2326_232624

theorem remove_one_and_average (l : List ℕ) (n : ℕ) (avg : ℚ) :
  l = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] →
  avg = 8.5 →
  (l.sum - n : ℚ) = 14 * avg →
  n = 1 :=
by
  intros hlist havg hsum
  sorry

end NUMINAMATH_GPT_remove_one_and_average_l2326_232624


namespace NUMINAMATH_GPT_solve_inequality_l2326_232690

theorem solve_inequality (x : ℝ) : (3 * x - 5) / 2 > 2 * x → x < -5 :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l2326_232690


namespace NUMINAMATH_GPT_GCF_LCM_15_21_14_20_l2326_232651

def GCF (a b : ℕ) : ℕ := Nat.gcd a b
def LCM (a b : ℕ) : ℕ := Nat.lcm a b

theorem GCF_LCM_15_21_14_20 :
  GCF (LCM 15 21) (LCM 14 20) = 35 :=
by
  sorry

end NUMINAMATH_GPT_GCF_LCM_15_21_14_20_l2326_232651


namespace NUMINAMATH_GPT_shaded_region_area_l2326_232691

-- Conditions given in the problem
def diameter (d : ℝ) := d = 4
def length_feet (l : ℝ) := l = 2

-- Proof statement
theorem shaded_region_area (d l : ℝ) (h1 : diameter d) (h2 : length_feet l) : 
  (l * 12 / d * (d / 2)^2 * π = 24 * π) := by
  sorry

end NUMINAMATH_GPT_shaded_region_area_l2326_232691


namespace NUMINAMATH_GPT_total_students_in_college_l2326_232619

theorem total_students_in_college (B G : ℕ) (h_ratio: 8 * G = 5 * B) (h_girls: G = 175) :
  B + G = 455 := 
  sorry

end NUMINAMATH_GPT_total_students_in_college_l2326_232619


namespace NUMINAMATH_GPT_sum_first_100_terms_l2326_232697

def a (n : ℕ) : ℤ := (-1) ^ (n + 1) * n

def S (n : ℕ) : ℤ := Finset.sum (Finset.range n) (λ i => a (i + 1))

theorem sum_first_100_terms : S 100 = -50 := 
by 
  sorry

end NUMINAMATH_GPT_sum_first_100_terms_l2326_232697


namespace NUMINAMATH_GPT_sum_first_3000_terms_l2326_232605

variable {α : Type*}

noncomputable def geometric_sum_1000 (a r : α) [Field α] : α := a * (r ^ 1000 - 1) / (r - 1)
noncomputable def geometric_sum_2000 (a r : α) [Field α] : α := a * (r ^ 2000 - 1) / (r - 1)
noncomputable def geometric_sum_3000 (a r : α) [Field α] : α := a * (r ^ 3000 - 1) / (r - 1)

theorem sum_first_3000_terms 
  {a r : ℝ}
  (h1 : geometric_sum_1000 a r = 1024)
  (h2 : geometric_sum_2000 a r = 2040) :
  geometric_sum_3000 a r = 3048 := 
  sorry

end NUMINAMATH_GPT_sum_first_3000_terms_l2326_232605


namespace NUMINAMATH_GPT_probability_standard_weight_l2326_232639

noncomputable def total_students : ℕ := 500
noncomputable def standard_students : ℕ := 350

theorem probability_standard_weight : (standard_students : ℚ) / (total_students : ℚ) = 7 / 10 :=
by {
  sorry
}

end NUMINAMATH_GPT_probability_standard_weight_l2326_232639


namespace NUMINAMATH_GPT_eq_from_conditions_l2326_232615

theorem eq_from_conditions (a b : ℂ) :
  (1 / (a + b)) ^ 2003 = 1 ∧ (-a + b) ^ 2005 = 1 → a ^ 2003 + b ^ 2004 = 1 := 
by
  sorry

end NUMINAMATH_GPT_eq_from_conditions_l2326_232615


namespace NUMINAMATH_GPT_speed_in_still_water_l2326_232644

/--
A man can row upstream at 55 kmph and downstream at 65 kmph.
Prove that his speed in still water is 60 kmph.
-/
theorem speed_in_still_water (upstream_speed downstream_speed : ℕ) (h_upstream : upstream_speed = 55) (h_downstream : downstream_speed = 65) : 
  (upstream_speed + downstream_speed) / 2 = 60 := by
  sorry

end NUMINAMATH_GPT_speed_in_still_water_l2326_232644


namespace NUMINAMATH_GPT_direction_vector_l1_l2326_232670

theorem direction_vector_l1
  (m : ℝ)
  (l₁ : ∀ x y : ℝ, (m + 3) * x + 4 * y + 3 * m - 5 = 0)
  (l₂ : ∀ x y : ℝ, 2 * x + (m + 6) * y - 8 = 0)
  (h_perp : ((m + 3) * 2 = -4 * (m + 6)))
  : ∃ v : ℝ × ℝ, v = (-1, -1/2) :=
by
  sorry

end NUMINAMATH_GPT_direction_vector_l1_l2326_232670


namespace NUMINAMATH_GPT_order_of_abc_l2326_232609

noncomputable def a : ℝ := (0.3)^3
noncomputable def b : ℝ := (3)^3
noncomputable def c : ℝ := Real.log 0.3 / Real.log 3

theorem order_of_abc : b > a ∧ a > c :=
by
  have ha : a = (0.3)^3 := rfl
  have hb : b = (3)^3 := rfl
  have hc : c = Real.log 0.3 / Real.log 3 := rfl
  sorry

end NUMINAMATH_GPT_order_of_abc_l2326_232609


namespace NUMINAMATH_GPT_soldiers_to_add_l2326_232663

theorem soldiers_to_add (N : ℕ) (add : ℕ) 
    (h1 : N % 7 = 2)
    (h2 : N % 12 = 2)
    (h_add : add = 84 - N) :
    add = 82 :=
by
  sorry

end NUMINAMATH_GPT_soldiers_to_add_l2326_232663


namespace NUMINAMATH_GPT_find_A_l2326_232622

variable (U A CU_A : Set ℕ)

axiom U_is_universal : U = {1, 3, 5, 7, 9}
axiom CU_A_is_complement : CU_A = {5, 7}

theorem find_A (h1 : U = {1, 3, 5, 7, 9}) (h2 : CU_A = {5, 7}) : 
  A = {1, 3, 9} :=
by
  sorry

end NUMINAMATH_GPT_find_A_l2326_232622


namespace NUMINAMATH_GPT_right_angled_triangles_count_l2326_232687

theorem right_angled_triangles_count : 
  ∃ n : ℕ, n = 12 ∧ ∀ (a b c : ℕ), (a = 2016^(1/2)) → (a^2 + b^2 = c^2) →
  (∃ (n k : ℕ), (c - b) = n ∧ (c + b) = k ∧ 2 ∣ n ∧ 2 ∣ k ∧ (n * k = 2016)) :=
by {
  sorry
}

end NUMINAMATH_GPT_right_angled_triangles_count_l2326_232687


namespace NUMINAMATH_GPT_reciprocal_roots_k_value_l2326_232614

theorem reciprocal_roots_k_value :
  ∀ k : ℝ, (∀ r : ℝ, 5.2 * r^2 + 14.3 * r + k = 0 ∧ 5.2 * (1 / r)^2 + 14.3 * (1 / r) + k = 0) →
          k = 5.2 :=
by
  sorry

end NUMINAMATH_GPT_reciprocal_roots_k_value_l2326_232614


namespace NUMINAMATH_GPT_series_sum_eq_l2326_232665

noncomputable def series_sum : Real :=
  ∑' n : ℕ, (4 * (n + 1) + 1) / (((4 * (n + 1) - 1) ^ 3) * ((4 * (n + 1) + 3) ^ 3))

theorem series_sum_eq : series_sum = 1 / 5184 := sorry

end NUMINAMATH_GPT_series_sum_eq_l2326_232665


namespace NUMINAMATH_GPT_second_number_added_is_5_l2326_232674

theorem second_number_added_is_5
  (x : ℕ) (h₁ : x = 3)
  (y : ℕ)
  (h₂ : (x + 1) * (x + 13) = (x + y) * (x + y)) :
  y = 5 :=
sorry

end NUMINAMATH_GPT_second_number_added_is_5_l2326_232674


namespace NUMINAMATH_GPT_ratio_of_blue_to_red_l2326_232677

variable (B : ℕ) -- Number of blue lights

def total_white := 59
def total_colored := total_white - 5
def red_lights := 12
def green_lights := 6

def total_bought := red_lights + green_lights + B

theorem ratio_of_blue_to_red (h : total_bought = total_colored) :
  B / red_lights = 3 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_blue_to_red_l2326_232677


namespace NUMINAMATH_GPT_compute_Q3_Qneg3_l2326_232613

noncomputable def Q (x : ℝ) (a b c m : ℝ) : ℝ :=
  a * x^3 + b * x^2 + c * x + m

theorem compute_Q3_Qneg3 (a b c m : ℝ)
  (h1 : Q 1 a b c m = 3 * m)
  (h2 : Q (-1) a b c m = 4 * m)
  (h3 : Q 0 a b c m = m) :
  Q 3 a b c m + Q (-3) a b c m = 47 * m :=
by
  sorry

end NUMINAMATH_GPT_compute_Q3_Qneg3_l2326_232613


namespace NUMINAMATH_GPT_mixture_concentration_l2326_232648

-- Definitions reflecting the given conditions
def sol1_concentration : ℝ := 0.30
def sol1_volume : ℝ := 8

def sol2_concentration : ℝ := 0.50
def sol2_volume : ℝ := 5

def sol3_concentration : ℝ := 0.70
def sol3_volume : ℝ := 7

-- The proof problem stating that the resulting concentration is 49%
theorem mixture_concentration :
  (sol1_concentration * sol1_volume + sol2_concentration * sol2_volume + sol3_concentration * sol3_volume) /
  (sol1_volume + sol2_volume + sol3_volume) * 100 = 49 :=
by
  sorry

end NUMINAMATH_GPT_mixture_concentration_l2326_232648


namespace NUMINAMATH_GPT_smaller_rectangle_length_ratio_l2326_232625

theorem smaller_rectangle_length_ratio 
  (s : ℝ)
  (h1 : 5 = 5)
  (h2 : ∃ r : ℝ, r = s)
  (h3 : ∀ x : ℝ, x = s)
  (h4 : ∀ y : ℝ, y / 2 = s / 2)
  (h5 : ∀ z : ℝ, z = 3 * s)
  (h6 : ∀ w : ℝ, w = s) :
  ∃ l : ℝ, l / s = 4 :=
sorry

end NUMINAMATH_GPT_smaller_rectangle_length_ratio_l2326_232625


namespace NUMINAMATH_GPT_angle_supplement_complement_l2326_232682

theorem angle_supplement_complement (x : ℝ) 
  (hsupp : 180 - x = 4 * (90 - x)) : 
  x = 60 :=
by
  sorry

end NUMINAMATH_GPT_angle_supplement_complement_l2326_232682


namespace NUMINAMATH_GPT_co_presidents_included_probability_l2326_232650

-- Let the number of students in each club
def club_sizes : List ℕ := [6, 8, 9, 10]

-- Function to calculate binomial coefficient
def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Function to calculate probability for a given club size
noncomputable def co_president_probability (n : ℕ) : ℚ :=
  (choose (n - 2) 2 : ℚ) / (choose n 4)

-- List of probabilities for each club
noncomputable def probabilities : List ℚ :=
  List.map co_president_probability club_sizes

-- Aggregate total probability by averaging the individual probabilities
noncomputable def total_probability : ℚ :=
  (1 / 4 : ℚ) * probabilities.sum

-- The proof problem: proving the total probability equals 119/700
theorem co_presidents_included_probability :
  total_probability = 119 / 700 := by
  sorry

end NUMINAMATH_GPT_co_presidents_included_probability_l2326_232650


namespace NUMINAMATH_GPT_range_of_a_l2326_232689

open Real

namespace PropositionProof

-- Define propositions p and q
def p (a x : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0
def q (x : ℝ) : Prop := (x^2 - x - 6 ≤ 0) ∨ (x^2 + 2*x - 8 > 0)

theorem range_of_a (a : ℝ) (h : a < 0) :
  (¬ ∀ x, ¬ p a x → ∀ x, ¬ q x) ↔ (a ≤ -4 ∨ -2/3 ≤ a ∧ a < 0) :=
sorry

end PropositionProof

end NUMINAMATH_GPT_range_of_a_l2326_232689


namespace NUMINAMATH_GPT_triangle_is_isosceles_l2326_232640

noncomputable def is_isosceles_triangle (A B C a b c : ℝ) : Prop := ∃ (s : ℝ), a = s ∧ b = s

theorem triangle_is_isosceles 
  (A B C a b c : ℝ) 
  (h_sides_angles : a = c ∧ b = c) 
  (h_cos_eq : a * Real.cos B = b * Real.cos A) : 
  is_isosceles_triangle A B C a b c := 
by 
  sorry

end NUMINAMATH_GPT_triangle_is_isosceles_l2326_232640


namespace NUMINAMATH_GPT_find_blue_yarn_count_l2326_232694

def scarves_per_yarn : ℕ := 3
def red_yarn_count : ℕ := 2
def yellow_yarn_count : ℕ := 4
def total_scarves : ℕ := 36

def scarves_from_red_and_yellow : ℕ :=
  red_yarn_count * scarves_per_yarn + yellow_yarn_count * scarves_per_yarn

def blue_scarves : ℕ :=
  total_scarves - scarves_from_red_and_yellow

def blue_yarn_count : ℕ :=
  blue_scarves / scarves_per_yarn

theorem find_blue_yarn_count :
  blue_yarn_count = 6 :=
by 
  sorry

end NUMINAMATH_GPT_find_blue_yarn_count_l2326_232694


namespace NUMINAMATH_GPT_profit_percentage_B_l2326_232641

-- Definitions based on conditions:
def CP_A : ℝ := 150  -- Cost price for A
def profit_percentage_A : ℝ := 0.20  -- Profit percentage for A
def SP_C : ℝ := 225  -- Selling price for C

-- Lean statement for the problem:
theorem profit_percentage_B : (SP_C - (CP_A * (1 + profit_percentage_A))) / (CP_A * (1 + profit_percentage_A)) * 100 = 25 := 
by 
  sorry

end NUMINAMATH_GPT_profit_percentage_B_l2326_232641


namespace NUMINAMATH_GPT_polynomial_composite_l2326_232685

theorem polynomial_composite (x : ℕ) : ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ 4 * x^3 + 6 * x^2 + 4 * x + 1 = a * b :=
by
  sorry

end NUMINAMATH_GPT_polynomial_composite_l2326_232685


namespace NUMINAMATH_GPT_sum_of_three_squares_l2326_232633

theorem sum_of_three_squares (a b : ℝ)
  (h1 : 3 * a + 2 * b = 18)
  (h2 : 2 * a + 3 * b = 22) :
  3 * b = 18 :=
sorry

end NUMINAMATH_GPT_sum_of_three_squares_l2326_232633


namespace NUMINAMATH_GPT_div_floor_factorial_l2326_232666

theorem div_floor_factorial (n q : ℕ) (hn : n ≥ 5) (hq : 2 ≤ q ∧ q ≤ n) :
  q - 1 ∣ (Nat.floor ((Nat.factorial (n - 1)) / q : ℚ)) :=
by
  sorry

end NUMINAMATH_GPT_div_floor_factorial_l2326_232666


namespace NUMINAMATH_GPT_max_page_number_with_given_fives_l2326_232662

theorem max_page_number_with_given_fives (plenty_digit_except_five : ℕ → ℕ) 
  (H0 : ∀ d ≠ 5, ∀ n, plenty_digit_except_five d = n)
  (H5 : plenty_digit_except_five 5 = 30) : ∃ (n : ℕ), n = 154 :=
by {
  sorry
}

end NUMINAMATH_GPT_max_page_number_with_given_fives_l2326_232662


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l2326_232669

noncomputable def is_increasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (∀ x y : ℝ, x < y → (a - 1) * (a ^ x) < (a - 1) * (a ^ y) → a > 1) ∧
  (¬ (∀ c : ℝ, is_increasing_function (λ x => (c - 1) * (c ^ x)) → c > 1)) :=
sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l2326_232669


namespace NUMINAMATH_GPT_apple_distribution_l2326_232628

theorem apple_distribution (total_apples : ℝ)
  (time_anya time_varya time_sveta total_time : ℝ)
  (work_anya work_varya work_sveta : ℝ) :
  total_apples = 10 →
  time_anya = 20 →
  time_varya = 35 →
  time_sveta = 45 →
  total_time = (time_anya + time_varya + time_sveta) →
  work_anya = (total_apples * time_anya / total_time) →
  work_varya = (total_apples * time_varya / total_time) →
  work_sveta = (total_apples * time_sveta / total_time) →
  work_anya = 2 ∧ work_varya = 3.5 ∧ work_sveta = 4.5 := by
  sorry

end NUMINAMATH_GPT_apple_distribution_l2326_232628


namespace NUMINAMATH_GPT_complex_number_pow_two_l2326_232658

theorem complex_number_pow_two (i : ℂ) (hi : i^2 = -1) : (1 + i)^2 = 2 * i :=
by sorry

end NUMINAMATH_GPT_complex_number_pow_two_l2326_232658


namespace NUMINAMATH_GPT_solve_k_equality_l2326_232606

noncomputable def collinear_vectors (e1 e2 : ℝ) (k : ℝ) (AB CB CD : ℝ) : Prop := 
  let BD := (2 * e1 - e2) - (e1 + 3 * e2)
  BD = e1 - 4 * e2 ∧ AB = 2 * e1 + k * e2 ∧ AB = k * BD
  
theorem solve_k_equality (e1 e2 k AB CB CD : ℝ) (h_non_collinear : (e1 ≠ 0 ∨ e2 ≠ 0)) :
  collinear_vectors e1 e2 k AB CB CD → k = -8 :=
by
  intro h_collinear
  sorry

end NUMINAMATH_GPT_solve_k_equality_l2326_232606


namespace NUMINAMATH_GPT_division_remainder_l2326_232678

-- let f(r) = r^15 + r + 1
def f (r : ℝ) : ℝ := r^15 + r + 1

-- let g(r) = r^2 - 1
def g (r : ℝ) : ℝ := r^2 - 1

-- remainder polynomial b(r)
def b (r : ℝ) : ℝ := r + 1

-- Lean statement to prove that polynomial division of f(r) by g(r) 
-- yields the remainder b(r)
theorem division_remainder (r : ℝ) : (f r) % (g r) = b r :=
  sorry

end NUMINAMATH_GPT_division_remainder_l2326_232678


namespace NUMINAMATH_GPT_compound_h_atoms_l2326_232668

theorem compound_h_atoms 
  (weight_H : ℝ) (weight_C : ℝ) (weight_O : ℝ)
  (num_C : ℕ) (num_O : ℕ)
  (total_molecular_weight : ℝ)
  (atomic_weight_H : ℝ) (atomic_weight_C : ℝ) (atomic_weight_O : ℝ)
  (H_w_is_1 : atomic_weight_H = 1)
  (C_w_is_12 : atomic_weight_C = 12)
  (O_w_is_16 : atomic_weight_O = 16)
  (C_atoms_is_1 : num_C = 1)
  (O_atoms_is_3 : num_O = 3)
  (total_mw_is_62 : total_molecular_weight = 62)
  (mw_C : weight_C = num_C * atomic_weight_C)
  (mw_O : weight_O = num_O * atomic_weight_O)
  (mw_CO : weight_C + weight_O = 60)
  (H_weight_contrib : total_molecular_weight - (weight_C + weight_O) = weight_H)
  (H_atoms_calc : weight_H = 2 * atomic_weight_H) :
  2 = 2 :=
by 
  sorry

end NUMINAMATH_GPT_compound_h_atoms_l2326_232668


namespace NUMINAMATH_GPT_max_volume_of_pyramid_PABC_l2326_232634

noncomputable def max_pyramid_volume (PA PB AB BC CA : ℝ) (hPA : PA = 3) (hPB : PB = 3) 
(hAB : AB = 2) (hBC : BC = 2) (hCA : CA = 2) : ℝ :=
  let D := 1 -- Midpoint of segment AB
  let PD : ℝ := Real.sqrt (PA ^ 2 - D ^ 2) -- Distance PD using Pythagorean theorem
  let S_ABC : ℝ := (Real.sqrt 3 / 4) * (AB ^ 2) -- Area of triangle ABC
  let V_PABC : ℝ := (1 / 3) * S_ABC * PD -- Volume of the pyramid
  V_PABC -- Return the volume

theorem max_volume_of_pyramid_PABC : 
  max_pyramid_volume 3 3 2 2 2  (rfl) (rfl) (rfl) (rfl) (rfl) = (2 * Real.sqrt 6) / 3 :=
by
  sorry

end NUMINAMATH_GPT_max_volume_of_pyramid_PABC_l2326_232634


namespace NUMINAMATH_GPT_quadratic_inequality_solutions_l2326_232686

theorem quadratic_inequality_solutions (k : ℝ) :
  (0 < k ∧ k < 16) ↔ ∃ x : ℝ, x^2 - 8*x + k < 0 :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_solutions_l2326_232686


namespace NUMINAMATH_GPT_fair_tickets_more_than_twice_baseball_tickets_l2326_232603

theorem fair_tickets_more_than_twice_baseball_tickets :
  ∃ (fair_tickets baseball_tickets : ℕ), 
    fair_tickets = 25 ∧ baseball_tickets = 56 ∧ 
    fair_tickets + 87 = 2 * baseball_tickets := 
by
  sorry

end NUMINAMATH_GPT_fair_tickets_more_than_twice_baseball_tickets_l2326_232603

import Mathlib

namespace NUMINAMATH_GPT_derivative_at_2_l1906_190638

def f (x : ℝ) : ℝ := x^3 + 2

theorem derivative_at_2 : deriv f 2 = 12 := by
  sorry

end NUMINAMATH_GPT_derivative_at_2_l1906_190638


namespace NUMINAMATH_GPT_wives_identification_l1906_190605

theorem wives_identification (Anna Betty Carol Dorothy MrBrown MrGreen MrWhite MrSmith : ℕ):
  Anna = 2 ∧ Betty = 3 ∧ Carol = 4 ∧ Dorothy = 5 ∧
  (MrBrown = Dorothy ∧ MrGreen = 2 * Carol ∧ MrWhite = 3 * Betty ∧ MrSmith = 4 * Anna) ∧
  (Anna + Betty + Carol + Dorothy + MrBrown + MrGreen + MrWhite + MrSmith = 44) →
  (
    Dorothy = 5 ∧
    Carol = 4 ∧
    Betty = 3 ∧
    Anna = 2 ∧
    MrBrown = 5 ∧
    MrGreen = 8 ∧
    MrWhite = 9 ∧
    MrSmith = 8
  ) :=
by
  intros
  sorry

end NUMINAMATH_GPT_wives_identification_l1906_190605


namespace NUMINAMATH_GPT_multiple_of_michael_trophies_l1906_190682

-- Conditions
def michael_current_trophies : ℕ := 30
def michael_trophies_increse : ℕ := 100
def total_trophies_in_three_years : ℕ := 430

-- Proof statement
theorem multiple_of_michael_trophies (x : ℕ) :
  (michael_current_trophies + michael_trophies_increse) + (michael_current_trophies * x) = total_trophies_in_three_years → x = 10 := 
by
  sorry

end NUMINAMATH_GPT_multiple_of_michael_trophies_l1906_190682


namespace NUMINAMATH_GPT_sum_pqrst_is_neg_15_over_2_l1906_190604

variable (p q r s t x : ℝ)
variable (h1 : p + 2 = x)
variable (h2 : q + 3 = x)
variable (h3 : r + 4 = x)
variable (h4 : s + 5 = x)
variable (h5 : t + 6 = x)
variable (h6 : p + q + r + s + t + 10 = x)

theorem sum_pqrst_is_neg_15_over_2 : p + q + r + s + t = -15 / 2 := by
  sorry

end NUMINAMATH_GPT_sum_pqrst_is_neg_15_over_2_l1906_190604


namespace NUMINAMATH_GPT_select_books_from_corner_l1906_190639

def num_ways_to_select_books (n₁ n₂ k : ℕ) : ℕ :=
  if h₁ : k > n₁ ∧ k > n₂ then 0
  else if h₂ : k > n₂ then 1
  else if h₃ : k > n₁ then Nat.choose n₂ k
  else Nat.choose n₁ k + 2 * Nat.choose n₁ (k-1) * Nat.choose n₂ 1 + Nat.choose n₁ k * 0 +
    (Nat.choose n₂ 1 * Nat.choose n₂ (k-1)) + Nat.choose n₂ k * 1

theorem select_books_from_corner :
  num_ways_to_select_books 3 6 3 = 42 :=
by
  sorry

end NUMINAMATH_GPT_select_books_from_corner_l1906_190639


namespace NUMINAMATH_GPT_identify_value_of_expression_l1906_190633

theorem identify_value_of_expression (x y z : ℝ)
  (h1 : y / (x - y) = x / (y + z))
  (h2 : z^2 = x * (y + z) - y * (x - y)) :
  (y^2 + z^2 - x^2) / (2 * y * z) = 1 / 2 := 
sorry

end NUMINAMATH_GPT_identify_value_of_expression_l1906_190633


namespace NUMINAMATH_GPT_parabola_focus_correct_l1906_190616

-- defining the equation of the parabola as a condition
def parabola (y x : ℝ) : Prop := y^2 = 4 * x

-- defining the focus of the parabola
def focus (x y : ℝ) : Prop := (x, y) = (1, 0)

-- the main theorem statement
theorem parabola_focus_correct (y x : ℝ) (h : parabola y x) : focus 1 0 :=
by
  -- proof steps would go here
  sorry

end NUMINAMATH_GPT_parabola_focus_correct_l1906_190616


namespace NUMINAMATH_GPT_blue_flowers_percentage_l1906_190644

theorem blue_flowers_percentage :
  let total_flowers := 96
  let green_flowers := 9
  let red_flowers := 3 * green_flowers
  let yellow_flowers := 12
  let accounted_flowers := green_flowers + red_flowers + yellow_flowers
  let blue_flowers := total_flowers - accounted_flowers
  (blue_flowers / total_flowers : ℝ) * 100 = 50 :=
by
  sorry

end NUMINAMATH_GPT_blue_flowers_percentage_l1906_190644


namespace NUMINAMATH_GPT_sector_area_l1906_190675

-- Given conditions
variables {l r : ℝ}

-- Definitions (conditions from the problem)
def arc_length (l : ℝ) := l
def radius (r : ℝ) := r

-- Problem statement
theorem sector_area (l r : ℝ) : 
    (1 / 2) * l * r = (1 / 2) * l * r :=
by
  sorry

end NUMINAMATH_GPT_sector_area_l1906_190675


namespace NUMINAMATH_GPT_complement_intersection_in_U_l1906_190629

universe u

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 3, 4, 5}

theorem complement_intersection_in_U : (U \ (A ∩ B)) = {1, 4, 5, 6, 7, 8} :=
by {
  sorry
}

end NUMINAMATH_GPT_complement_intersection_in_U_l1906_190629


namespace NUMINAMATH_GPT_fault_line_movement_year_before_l1906_190610

-- Define the total movement over two years
def total_movement : ℝ := 6.5

-- Define the movement during the past year
def past_year_movement : ℝ := 1.25

-- Define the movement the year before
def year_before_movement : ℝ := total_movement - past_year_movement

-- Prove that the fault line moved 5.25 inches the year before
theorem fault_line_movement_year_before : year_before_movement = 5.25 :=
  by  sorry

end NUMINAMATH_GPT_fault_line_movement_year_before_l1906_190610


namespace NUMINAMATH_GPT_like_terms_sum_l1906_190615

theorem like_terms_sum (m n : ℕ) (a b : ℝ) 
  (h₁ : 5 * a^m * b^3 = 5 * a^m * b^3) 
  (h₂ : -4 * a^2 * b^(n-1) = -4 * a^2 * b^(n-1)) 
  (h₃ : m = 2) (h₄ : 3 = n - 1) : m + n = 6 := by
  sorry

end NUMINAMATH_GPT_like_terms_sum_l1906_190615


namespace NUMINAMATH_GPT_negation_of_proposition_range_of_m_l1906_190691

noncomputable def proposition (m : ℝ) : Prop := ∃ x : ℝ, x^2 + 2 * x - m - 1 < 0

theorem negation_of_proposition (m : ℝ) : ¬ proposition m ↔ ∀ x : ℝ, x^2 + 2 * x - m - 1 ≥ 0 :=
sorry

theorem range_of_m (m : ℝ) : proposition m → m > -2 :=
sorry

end NUMINAMATH_GPT_negation_of_proposition_range_of_m_l1906_190691


namespace NUMINAMATH_GPT_select_7_jury_l1906_190677

theorem select_7_jury (students : Finset ℕ) (jury : Finset ℕ)
  (likes : ℕ → Finset ℕ) (h_students : students.card = 100)
  (h_jury : jury.card = 25) (h_likes : ∀ s ∈ students, (likes s).card = 10) :
  ∃ (selected_jury : Finset ℕ), selected_jury.card = 7 ∧ ∀ s ∈ students, ∃ j ∈ selected_jury, j ∈ (likes s) :=
sorry

end NUMINAMATH_GPT_select_7_jury_l1906_190677


namespace NUMINAMATH_GPT_trees_left_after_typhoon_l1906_190679

-- Define the initial count of trees and the number of trees that died
def initial_trees := 150
def trees_died := 24

-- Define the expected number of trees left
def expected_trees_left := 126

-- The statement to be proven: after trees died, the number of trees left is as expected
theorem trees_left_after_typhoon : (initial_trees - trees_died) = expected_trees_left := by
  sorry

end NUMINAMATH_GPT_trees_left_after_typhoon_l1906_190679


namespace NUMINAMATH_GPT_determine_d_l1906_190687

theorem determine_d (d c f : ℚ) :
  (3 * x^3 - 2 * x^2 + x - (5/4)) * (3 * x^3 + d * x^2 + c * x + f) = 9 * x^6 - 5 * x^5 - x^4 + 20 * x^3 - (25/4) * x^2 + (15/4) * x - (5/2) →
  d = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_determine_d_l1906_190687


namespace NUMINAMATH_GPT_closest_point_on_ellipse_to_line_l1906_190608

theorem closest_point_on_ellipse_to_line :
  ∃ (x y : ℝ), 
    7 * x^2 + 4 * y^2 = 28 ∧ 3 * x - 2 * y - 16 = 0 ∧ (x, y) = (3 / 2, -7 / 4) :=
by
  sorry

end NUMINAMATH_GPT_closest_point_on_ellipse_to_line_l1906_190608


namespace NUMINAMATH_GPT_coffee_expenses_l1906_190648

-- Define amounts consumed and unit costs for French and Columbian roast
def ounces_per_donut_M := 2
def ounces_per_donut_D := 3
def ounces_per_donut_S := ounces_per_donut_D
def ounces_per_pot_F := 12
def ounces_per_pot_C := 15
def cost_per_pot_F := 3
def cost_per_pot_C := 4

-- Define number of donuts consumed
def donuts_M := 8
def donuts_D := 12
def donuts_S := 16

-- Calculate total ounces needed
def total_ounces_F := donuts_M * ounces_per_donut_M
def total_ounces_C := (donuts_D + donuts_S) * ounces_per_donut_D

-- Calculate pots needed, rounding up since partial pots are not allowed
def pots_needed_F := Nat.ceil (total_ounces_F / ounces_per_pot_F)
def pots_needed_C := Nat.ceil (total_ounces_C / ounces_per_pot_C)

-- Calculate total cost
def total_cost := (pots_needed_F * cost_per_pot_F) + (pots_needed_C * cost_per_pot_C)

-- Theorem statement to assert the proof
theorem coffee_expenses : total_cost = 30 := by
  sorry

end NUMINAMATH_GPT_coffee_expenses_l1906_190648


namespace NUMINAMATH_GPT_firing_sequence_hits_submarine_l1906_190628

theorem firing_sequence_hits_submarine (a b : ℕ) (hb : b > 0) : ∃ n : ℕ, (∃ (an bn : ℕ), (an + bn * n) = a + n * b) :=
sorry

end NUMINAMATH_GPT_firing_sequence_hits_submarine_l1906_190628


namespace NUMINAMATH_GPT_area_of_T_shaped_region_l1906_190680

theorem area_of_T_shaped_region :
  let ABCD_area : ℝ := 48
  let EFHG_area : ℝ := 4
  let EFGI_area : ℝ := 8
  let EFCD_area : ℝ := 12
  (ABCD_area - (EFHG_area + EFGI_area + EFCD_area)) = 24 :=
by
  let ABCD_area : ℝ := 48
  let EFHG_area : ℝ := 4
  let EFGI_area : ℝ := 8
  let EFCD_area : ℝ := 12
  exact sorry

end NUMINAMATH_GPT_area_of_T_shaped_region_l1906_190680


namespace NUMINAMATH_GPT_twenty_seven_divides_sum_l1906_190600

theorem twenty_seven_divides_sum (x y z : ℤ) (h : (x - y) * (y - z) * (z - x) = x + y + z) : 27 ∣ x + y + z := sorry

end NUMINAMATH_GPT_twenty_seven_divides_sum_l1906_190600


namespace NUMINAMATH_GPT_brian_distance_more_miles_l1906_190621

variables (s t d m n : ℝ)
-- Mike's distance
variable (hd : d = s * t)
-- Steve's distance condition
variable (hsteve : d + 90 = (s + 6) * (t + 1.5))
-- Brian's distance
variable (hbrian : m = (s + 12) * (t + 3))

theorem brian_distance_more_miles :
  n = m - d → n = 200 :=
sorry

end NUMINAMATH_GPT_brian_distance_more_miles_l1906_190621


namespace NUMINAMATH_GPT_motorcycle_licenses_count_l1906_190690

theorem motorcycle_licenses_count : (3 * (10 ^ 6) = 3000000) :=
by
  sorry -- Proof would go here.

end NUMINAMATH_GPT_motorcycle_licenses_count_l1906_190690


namespace NUMINAMATH_GPT_total_number_of_members_l1906_190686

variables (b g : Nat)
def girls_twice_boys : Prop := g = 2 * b
def boys_twice_remaining_girls (b g : Nat) : Prop := b = 2 * (g - 24)

theorem total_number_of_members (b g : Nat) 
  (h1 : girls_twice_boys b g) 
  (h2 : boys_twice_remaining_girls b g) : 
  b + g = 48 := by
  sorry

end NUMINAMATH_GPT_total_number_of_members_l1906_190686


namespace NUMINAMATH_GPT_average_weight_increase_l1906_190632

theorem average_weight_increase
 (num_persons : ℕ) (weight_increase : ℝ) (replacement_weight : ℝ) (new_weight : ℝ) (weight_difference : ℝ) (avg_weight_increase : ℝ)
 (cond1 : num_persons = 10)
 (cond2 : replacement_weight = 65)
 (cond3 : new_weight = 90)
 (cond4 : weight_difference = new_weight - replacement_weight)
 (cond5 : weight_difference = weight_increase)
 (cond6 : avg_weight_increase = weight_increase / num_persons) :
avg_weight_increase = 2.5 :=
by
  sorry

end NUMINAMATH_GPT_average_weight_increase_l1906_190632


namespace NUMINAMATH_GPT_range_of_x_l1906_190656

-- Define the ceiling function for ease of use.
noncomputable def ceil (x : ℝ) : ℤ := ⌈x⌉

theorem range_of_x (x : ℝ) (h1 : ceil (2 * x + 1) = 5) (h2 : ceil (2 - 3 * x) = -3) :
  (5 / 3 : ℝ) ≤ x ∧ x < 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_x_l1906_190656


namespace NUMINAMATH_GPT_sum_of_number_and_reverse_is_perfect_square_iff_l1906_190658

def is_two_digit (n : ℕ) : Prop :=
  n >= 10 ∧ n < 100

def reverse_of (n : ℕ) : ℕ :=
  let a := n / 10
  let b := n % 10
  10 * b + a

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

theorem sum_of_number_and_reverse_is_perfect_square_iff :
  ∀ n : ℕ, is_two_digit n →
    is_perfect_square (n + reverse_of n) ↔
      n = 29 ∨ n = 38 ∨ n = 47 ∨ n = 56 ∨ n = 65 ∨ n = 74 ∨ n = 83 ∨ n = 92 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_number_and_reverse_is_perfect_square_iff_l1906_190658


namespace NUMINAMATH_GPT_inverse_function_composition_l1906_190630

def g (x : ℝ) : ℝ := 3 * x + 7

noncomputable def g_inv (y : ℝ) : ℝ := (y - 7) / 3

theorem inverse_function_composition : g_inv (g_inv 20) = -8 / 9 := by
  sorry

end NUMINAMATH_GPT_inverse_function_composition_l1906_190630


namespace NUMINAMATH_GPT_integer_roots_l1906_190697

-- Define the polynomial
def polynomial (x : ℤ) : ℤ := x^3 - 4 * x^2 - 7 * x + 10

-- Define the proof problem statement
theorem integer_roots :
  {x : ℤ | polynomial x = 0} = {1, -2, 5} :=
by
  sorry

end NUMINAMATH_GPT_integer_roots_l1906_190697


namespace NUMINAMATH_GPT_simplify_expression_l1906_190620

theorem simplify_expression :
  (3 * (Real.sqrt 3 + Real.sqrt 5)) / (4 * Real.sqrt (3 + Real.sqrt 4))
  = (3 * Real.sqrt 15 + 3 * Real.sqrt 5) / 20 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1906_190620


namespace NUMINAMATH_GPT_relationship_y1_y2_y3_l1906_190636

theorem relationship_y1_y2_y3 (c y1 y2 y3 : ℝ) :
  (y1 = (-(1^2) + 2 * 1 + c))
  ∧ (y2 = (-(2^2) + 2 * 2 + c))
  ∧ (y3 = (-(5^2) + 2 * 5 + c))
  → (y2 > y1 ∧ y1 > y3) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_relationship_y1_y2_y3_l1906_190636


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l1906_190669

variable {S : ℕ → ℕ}

theorem arithmetic_sequence_sum (h1 : S 3 = 15) (h2 : S 9 = 153) : S 6 = 66 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l1906_190669


namespace NUMINAMATH_GPT_caroline_lassis_l1906_190607

theorem caroline_lassis (c : ℕ → ℕ): c 3 = 13 → c 15 = 65 :=
by
  sorry

end NUMINAMATH_GPT_caroline_lassis_l1906_190607


namespace NUMINAMATH_GPT_probability_of_at_least_2_girls_equals_specified_value_l1906_190642

def num_combinations (n k : ℕ) : ℕ :=
  (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

noncomputable def probability_at_least_2_girls : ℚ :=
  let total_committees := num_combinations 24 5
  let all_boys := num_combinations 14 5
  let one_girl_four_boys := num_combinations 10 1 * num_combinations 14 4
  let at_least_2_girls := total_committees - (all_boys + one_girl_four_boys)
  at_least_2_girls / total_committees

theorem probability_of_at_least_2_girls_equals_specified_value :
  probability_at_least_2_girls = 2541 / 3542 := 
sorry

end NUMINAMATH_GPT_probability_of_at_least_2_girls_equals_specified_value_l1906_190642


namespace NUMINAMATH_GPT_sum_of_squares_l1906_190643

theorem sum_of_squares (x y : ℝ) (h1 : x + y = 40) (h2 : x * y = 110) : x^2 + y^2 = 1380 := 
by sorry

end NUMINAMATH_GPT_sum_of_squares_l1906_190643


namespace NUMINAMATH_GPT_find_seventh_term_l1906_190672

variables {a : ℕ → ℝ} {S : ℕ → ℝ} {d : ℝ}

-- Define arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- Define sum of the first n terms of the sequence
def sum_first_n_terms (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n, S n = (n * a 0) + (d * (n * (n - 1)) / 2)

-- Now state the theorem
theorem find_seventh_term
  (h_arith_seq : arithmetic_sequence a d)
  (h_nonzero_d : d ≠ 0)
  (h_sum_five : S 5 = 5)
  (h_squares_eq : a 0 ^ 2 + a 1 ^ 2 = a 2 ^ 2 + a 3 ^ 2) :
  a 6 = 9 :=
sorry

end NUMINAMATH_GPT_find_seventh_term_l1906_190672


namespace NUMINAMATH_GPT_initial_house_cats_l1906_190695

theorem initial_house_cats (H : ℕ) (H_condition : 13 + H - 10 = 8) : H = 5 :=
by
-- sorry provides a placeholder to skip the actual proof
sorry

end NUMINAMATH_GPT_initial_house_cats_l1906_190695


namespace NUMINAMATH_GPT_difference_of_fractions_l1906_190601

theorem difference_of_fractions (a b c : ℝ) (h1 : a = 8000 * (1/2000)) (h2 : b = 8000 * (1/10)) (h3 : c = b - a) : c = 796 := 
sorry

end NUMINAMATH_GPT_difference_of_fractions_l1906_190601


namespace NUMINAMATH_GPT_problem_sum_value_l1906_190627

def letter_value_pattern : List Int := [2, 3, 2, 1, 0, -1, -2, -3, -2, -1]

def char_value (c : Char) : Int :=
  let pos := c.toNat - 'a'.toNat + 1
  letter_value_pattern.get! ((pos - 1) % 10)

def word_value (w : String) : Int :=
  w.data.map char_value |>.sum

theorem problem_sum_value : word_value "problem" = 5 :=
  by sorry

end NUMINAMATH_GPT_problem_sum_value_l1906_190627


namespace NUMINAMATH_GPT_largest_root_ratio_l1906_190619

-- Define the polynomials f(x) and g(x)
def f (x : ℝ) : ℝ := 1 - x - 4 * x^2 + x^4
def g (x : ℝ) : ℝ := 16 - 8 * x - 16 * x^2 + x^4

-- Define the property that x1 is the largest root of f(x) and x2 is the largest root of g(x)
def is_largest_root (p : ℝ → ℝ) (r : ℝ) : Prop := 
  p r = 0 ∧ ∀ x : ℝ, p x = 0 → x ≤ r

-- The main theorem
theorem largest_root_ratio (x1 x2 : ℝ) 
  (hx1 : is_largest_root f x1) 
  (hx2 : is_largest_root g x2) : x2 = 2 * x1 :=
sorry

end NUMINAMATH_GPT_largest_root_ratio_l1906_190619


namespace NUMINAMATH_GPT_solve_prime_equation_l1906_190676

theorem solve_prime_equation (p q r : ℕ) (hp : p.Prime) (hq : q.Prime) (hr : r.Prime) 
(h_eq : p^3 - q^3 = 5 * r) : p = 7 ∧ q = 2 ∧ r = 67 := 
sorry

end NUMINAMATH_GPT_solve_prime_equation_l1906_190676


namespace NUMINAMATH_GPT_identify_infected_person_in_4_tests_l1906_190640

theorem identify_infected_person_in_4_tests :
  (∀ (group : Fin 16 → Bool), ∃ infected : Fin 16, group infected = ff) →
  ∃ (tests_needed : ℕ), tests_needed = 4 :=
by sorry

end NUMINAMATH_GPT_identify_infected_person_in_4_tests_l1906_190640


namespace NUMINAMATH_GPT_at_least_one_misses_l1906_190611

-- Definitions for the given conditions
variables {p q : Prop}

-- Lean 4 statement proving the equivalence
theorem at_least_one_misses (hp : p → false) (hq : q → false) : (¬p ∨ ¬q) :=
by sorry

end NUMINAMATH_GPT_at_least_one_misses_l1906_190611


namespace NUMINAMATH_GPT_find_a_l1906_190654

noncomputable def binomialExpansion (a : ℚ) (x : ℚ) := (x - a / x) ^ 6

theorem find_a (a : ℚ) (A : ℚ) (B : ℚ) (hA : A = 15 * a ^ 2) (hB : B = -20 * a ^ 3) (hB_value : B = 44) :
  a = -22 / 5 :=
by
  sorry -- skipping the proof

end NUMINAMATH_GPT_find_a_l1906_190654


namespace NUMINAMATH_GPT_mila_father_total_pay_l1906_190618

def first_job_pay : ℤ := 2125
def pay_difference : ℤ := 375
def second_job_pay : ℤ := first_job_pay - pay_difference
def total_pay : ℤ := first_job_pay + second_job_pay

theorem mila_father_total_pay :
  total_pay = 3875 := by
  sorry

end NUMINAMATH_GPT_mila_father_total_pay_l1906_190618


namespace NUMINAMATH_GPT_neg_exists_eq_forall_l1906_190603

theorem neg_exists_eq_forall (p : Prop) :
  (∀ x : ℝ, ¬(x^2 + 2*x = 3)) ↔ ¬(∃ x : ℝ, x^2 + 2*x = 3) := 
by
  sorry

end NUMINAMATH_GPT_neg_exists_eq_forall_l1906_190603


namespace NUMINAMATH_GPT_graph_not_in_second_quadrant_l1906_190609

theorem graph_not_in_second_quadrant (b : ℝ) (h : ∀ x < 0, 2^x + b - 1 < 0) : b ≤ 0 :=
sorry

end NUMINAMATH_GPT_graph_not_in_second_quadrant_l1906_190609


namespace NUMINAMATH_GPT_average_weighted_score_l1906_190666

theorem average_weighted_score
  (score1 score2 score3 : ℕ)
  (weight1 weight2 weight3 : ℕ)
  (h_scores : score1 = 90 ∧ score2 = 85 ∧ score3 = 80)
  (h_weights : weight1 = 5 ∧ weight2 = 2 ∧ weight3 = 3) :
  (weight1 * score1 + weight2 * score2 + weight3 * score3) / (weight1 + weight2 + weight3) = 86 := 
by
  sorry

end NUMINAMATH_GPT_average_weighted_score_l1906_190666


namespace NUMINAMATH_GPT_range_of_a_l1906_190606

def p (a : ℝ) : Prop := a ≤ -4 ∨ a ≥ 4
def q (a : ℝ) : Prop := a ≥ -12
def either_p_or_q_but_not_both (a : ℝ) : Prop := (p a ∧ ¬ q a) ∨ (¬ p a ∧ q a)

theorem range_of_a :
  {a : ℝ | either_p_or_q_but_not_both a} = {a : ℝ | (-4 < a ∧ a < 4) ∨ a < -12} :=
sorry

end NUMINAMATH_GPT_range_of_a_l1906_190606


namespace NUMINAMATH_GPT_evaluate_expression_l1906_190668

theorem evaluate_expression : (3 / (2 - (4 / (-5)))) = (15 / 14) :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1906_190668


namespace NUMINAMATH_GPT_range_of_x_l1906_190671

theorem range_of_x (a x : ℝ) (h : 0 ≤ a ∧ a ≤ 4) :
  (x^2 + a * x > 4 * x + a - 3) ↔ (x < -1 ∨ x > 3) :=
by
  sorry

end NUMINAMATH_GPT_range_of_x_l1906_190671


namespace NUMINAMATH_GPT_power_func_passes_point_l1906_190651

noncomputable def power_function (α : ℝ) (x : ℝ) : ℝ := x ^ α

theorem power_func_passes_point (f : ℝ → ℝ) (h : ∃ α : ℝ, ∀ x : ℝ, f x = x ^ α) 
  (h_point : f 9 = 1 / 3) : f 25 = 1 / 5 :=
sorry

end NUMINAMATH_GPT_power_func_passes_point_l1906_190651


namespace NUMINAMATH_GPT_gray_region_area_l1906_190641

noncomputable def area_gray_region : ℝ :=
  let area_rectangle := (12 - 4) * (12 - 4)
  let radius_c := 4
  let radius_d := 4
  let area_quarter_circle_c := 1/4 * Real.pi * radius_c^2
  let area_quarter_circle_d := 1/4 * Real.pi * radius_d^2
  let overlap_area := area_quarter_circle_c + area_quarter_circle_d
  area_rectangle - overlap_area

theorem gray_region_area :
  area_gray_region = 64 - 8 * Real.pi := by
  sorry

end NUMINAMATH_GPT_gray_region_area_l1906_190641


namespace NUMINAMATH_GPT_minimum_cuts_for_11_sided_polygons_l1906_190665

theorem minimum_cuts_for_11_sided_polygons (k : ℕ) :
  (∀ k, (11 * 252 + 3 * (k + 1 - 252) ≤ 4 * k + 4)) ∧ (252 ≤ (k + 1)) ∧ (4 * k + 4 ≥ 11 * 252 + 3 * (k + 1 - 252))
  ∧ (11 * 252 + 3 * (k + 1 - 252) ≤ 4 * k + 4) → (k ≥ 2012) ∧ (k = 2015) := 
sorry

end NUMINAMATH_GPT_minimum_cuts_for_11_sided_polygons_l1906_190665


namespace NUMINAMATH_GPT_range_of_a_l1906_190698

-- Conditions for sets A and B
def SetA := {x : ℝ | -2 ≤ x ∧ x ≤ 2}
def SetB (a : ℝ) := {x : ℝ | a ≤ x ∧ x ≤ a + 2}

-- Main statement to show that A ∪ B = A implies the range of a is [-2, 0]
theorem range_of_a (a : ℝ) : (SetB a ⊆ SetA) → (-2 ≤ a ∧ a ≤ 0) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1906_190698


namespace NUMINAMATH_GPT_coordinates_of_B_l1906_190673

theorem coordinates_of_B (x y : ℝ) (A : ℝ × ℝ) (a : ℝ × ℝ) :
  A = (2, 4) ∧ a = (3, 4) ∧ (x - 2, y - 4) = (2 * a.1, 2 * a.2) → (x, y) = (8, 12) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_coordinates_of_B_l1906_190673


namespace NUMINAMATH_GPT_percentage_of_the_stock_l1906_190652

noncomputable def faceValue : ℝ := 100
noncomputable def yield : ℝ := 0.10
noncomputable def quotedPrice : ℝ := 160

theorem percentage_of_the_stock : 
  (yield * faceValue / quotedPrice * 100 = 6.25) :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_the_stock_l1906_190652


namespace NUMINAMATH_GPT_midpoint_lattice_point_exists_l1906_190650

theorem midpoint_lattice_point_exists (S : Finset (ℤ × ℤ)) (hS : S.card = 5) :
  ∃ (p1 p2 : ℤ × ℤ), p1 ∈ S ∧ p2 ∈ S ∧ p1 ≠ p2 ∧
  (∃ (x_mid y_mid : ℤ), 
    (p1.1 + p2.1) = 2 * x_mid ∧
    (p1.2 + p2.2) = 2 * y_mid) :=
by
  sorry

end NUMINAMATH_GPT_midpoint_lattice_point_exists_l1906_190650


namespace NUMINAMATH_GPT_books_in_june_l1906_190647

-- Definitions
def Book_may : ℕ := 2
def Book_july : ℕ := 10
def Total_books : ℕ := 18

-- Theorem statement
theorem books_in_june : ∃ (Book_june : ℕ), Book_may + Book_june + Book_july = Total_books ∧ Book_june = 6 :=
by
  -- Proof will be here
  sorry

end NUMINAMATH_GPT_books_in_june_l1906_190647


namespace NUMINAMATH_GPT_packs_of_chewing_gum_zero_l1906_190688

noncomputable def frozen_yogurt_price : ℝ := sorry
noncomputable def chewing_gum_price : ℝ := frozen_yogurt_price / 2
noncomputable def packs_of_chewing_gum : ℕ := sorry

theorem packs_of_chewing_gum_zero 
  (F : ℝ) -- Price of a pint of frozen yogurt
  (G : ℝ) -- Price of a pack of chewing gum
  (x : ℕ) -- Number of packs of chewing gum
  (H1 : G = F / 2)
  (H2 : 5 * F + x * G + 25 = 55)
  : x = 0 :=
sorry

end NUMINAMATH_GPT_packs_of_chewing_gum_zero_l1906_190688


namespace NUMINAMATH_GPT_Laura_running_speed_l1906_190681

noncomputable def running_speed (x : ℝ) :=
  let biking_time := 30 / (3 * x + 2)
  let running_time := 10 / x
  let total_time := biking_time + running_time
  total_time = 3

theorem Laura_running_speed : ∃ x : ℝ, running_speed x ∧ abs (x - 6.35) < 0.01 :=
sorry

end NUMINAMATH_GPT_Laura_running_speed_l1906_190681


namespace NUMINAMATH_GPT_inverse_proportion_inequality_l1906_190685

theorem inverse_proportion_inequality :
  ∀ (y : ℝ → ℝ) (y_1 y_2 y_3 : ℝ),
  (∀ x, y x = 7 / x) →
  y (-3) = y_1 →
  y (-1) = y_2 →
  y (2) = y_3 →
  y_2 < y_1 ∧ y_1 < y_3 :=
by
  intros y y_1 y_2 y_3 hy hA hB hC
  sorry

end NUMINAMATH_GPT_inverse_proportion_inequality_l1906_190685


namespace NUMINAMATH_GPT_gopi_turbans_annual_salary_l1906_190694

variable (T : ℕ) (annual_salary_turbans : ℕ)
variable (annual_salary_money : ℕ := 90)
variable (months_worked : ℕ := 9)
variable (total_months_in_year : ℕ := 12)
variable (received_money : ℕ := 55)
variable (turban_price : ℕ := 50)
variable (received_turbans : ℕ := 1)
variable (servant_share_fraction : ℚ := 3 / 4)

theorem gopi_turbans_annual_salary 
    (annual_salary_turbans : ℕ)
    (H : (servant_share_fraction * (annual_salary_money + turban_price * annual_salary_turbans) = received_money + turban_price * received_turbans))
    : annual_salary_turbans = 1 :=
sorry

end NUMINAMATH_GPT_gopi_turbans_annual_salary_l1906_190694


namespace NUMINAMATH_GPT_find_y_in_set_l1906_190655

noncomputable def arithmetic_mean (s : List ℝ) : ℝ :=
  s.sum / s.length

theorem find_y_in_set :
  ∀ (y : ℝ), arithmetic_mean [8, 15, 20, 5, y] = 12 ↔ y = 12 :=
by
  intro y
  unfold arithmetic_mean
  simp [List.sum_cons, List.length_cons]
  sorry

end NUMINAMATH_GPT_find_y_in_set_l1906_190655


namespace NUMINAMATH_GPT_correct_option_is_C_l1906_190674

variable (a b : ℝ)

def option_A : Prop := (a - b) ^ 2 = a ^ 2 - b ^ 2
def option_B : Prop := a ^ 2 + a ^ 2 = a ^ 4
def option_C : Prop := (a ^ 2) ^ 3 = a ^ 6
def option_D : Prop := a ^ 2 * a ^ 2 = a ^ 6

theorem correct_option_is_C : option_C a :=
by
  sorry

end NUMINAMATH_GPT_correct_option_is_C_l1906_190674


namespace NUMINAMATH_GPT_cost_of_first_20_kgs_l1906_190659

theorem cost_of_first_20_kgs (l q : ℕ)
  (h1 : 30 * l + 3 * q = 168)
  (h2 : 30 * l + 6 * q = 186) :
  20 * l = 100 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_first_20_kgs_l1906_190659


namespace NUMINAMATH_GPT_superior_points_in_Omega_l1906_190684

-- Define the set Omega
def Omega : Set (ℝ × ℝ) := { p | let (x, y) := p; x^2 + y^2 ≤ 2008 }

-- Definition of the superior relation
def superior (P P' : ℝ × ℝ) : Prop :=
  let (x, y) := P
  let (x', y') := P'
  x ≤ x' ∧ y ≥ y'

-- Definition of the set of points Q such that no other point in Omega is superior to Q
def Q_set : Set (ℝ × ℝ) :=
  { p | let (x, y) := p; x^2 + y^2 = 2008 ∧ x ≤ 0 ∧ y ≥ 0 }

theorem superior_points_in_Omega :
  { p | p ∈ Omega ∧ ¬ (∃ q ∈ Omega, superior q p) } = Q_set :=
by
  sorry

end NUMINAMATH_GPT_superior_points_in_Omega_l1906_190684


namespace NUMINAMATH_GPT_choir_students_min_l1906_190617

/-- 
  Prove that the minimum number of students in the choir, where the number 
  of students must be a multiple of 9, 10, and 11, is 990. 
-/
theorem choir_students_min (n : ℕ) :
  (∃ n, n > 0 ∧ n % 9 = 0 ∧ n % 10 = 0 ∧ n % 11 = 0) ∧ (∀ m, m > 0 ∧ m % 9 = 0 ∧ m % 10 = 0 ∧ m % 11 = 0 → n ≤ m) → n = 990 :=
by
  sorry

end NUMINAMATH_GPT_choir_students_min_l1906_190617


namespace NUMINAMATH_GPT_problem_part1_problem_part2_l1906_190626

-- Problem statements

theorem problem_part1 (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^3 + b^3 = 2) : 
  (a + b) * (a^5 + b^5) ≥ 4 := 
sorry

theorem problem_part2 (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^3 + b^3 = 2) : 
  a + b ≤ 2 := 
sorry

end NUMINAMATH_GPT_problem_part1_problem_part2_l1906_190626


namespace NUMINAMATH_GPT_value_of_a_l1906_190663

variable (a : ℝ)

noncomputable def f (x : ℝ) := x^2 + 8
noncomputable def g (x : ℝ) := x^2 - 4

theorem value_of_a
  (h0 : a > 0)
  (h1 : f (g a) = 8) : a = 2 :=
by
  -- conditions are used as assumptions
  let f := f
  let g := g
  sorry

end NUMINAMATH_GPT_value_of_a_l1906_190663


namespace NUMINAMATH_GPT_kellan_wax_remaining_l1906_190689

def remaining_wax (initial_A : ℕ) (initial_B : ℕ)
                  (spill_A : ℕ) (spill_B : ℕ)
                  (use_car_A : ℕ) (use_suv_B : ℕ) : ℕ :=
  let remaining_A := initial_A - spill_A - use_car_A
  let remaining_B := initial_B - spill_B - use_suv_B
  remaining_A + remaining_B

theorem kellan_wax_remaining
  (initial_A : ℕ := 10) 
  (initial_B : ℕ := 15)
  (spill_A : ℕ := 3) 
  (spill_B : ℕ := 4)
  (use_car_A : ℕ := 4) 
  (use_suv_B : ℕ := 5) :
  remaining_wax initial_A initial_B spill_A spill_B use_car_A use_suv_B = 9 :=
by sorry

end NUMINAMATH_GPT_kellan_wax_remaining_l1906_190689


namespace NUMINAMATH_GPT_bills_equal_at_80_minutes_l1906_190696

variable (m : ℝ)

def C_U : ℝ := 8 + 0.25 * m
def C_A : ℝ := 12 + 0.20 * m

theorem bills_equal_at_80_minutes (h : C_U m = C_A m) : m = 80 :=
by {
  sorry
}

end NUMINAMATH_GPT_bills_equal_at_80_minutes_l1906_190696


namespace NUMINAMATH_GPT_parabola_hyperbola_tangent_l1906_190662

-- Definitions of the parabola and hyperbola
def parabola (x : ℝ) : ℝ := x^2 + 4
def hyperbola (x y : ℝ) (m : ℝ) : Prop := y^2 - m*x^2 = 1

-- Tangency condition stating that the parabola and hyperbola are tangent implies m = 8 + 2*sqrt(15)
theorem parabola_hyperbola_tangent (m : ℝ) :
  (∀ x y : ℝ, parabola x = y → hyperbola x y m) → m = 8 + 2 * Real.sqrt 15 :=
by
  sorry

end NUMINAMATH_GPT_parabola_hyperbola_tangent_l1906_190662


namespace NUMINAMATH_GPT_studios_total_l1906_190622

section

variable (s1 s2 s3 : ℕ)

theorem studios_total (h1 : s1 = 110) (h2 : s2 = 135) (h3 : s3 = 131) : s1 + s2 + s3 = 376 :=
by
  sorry

end

end NUMINAMATH_GPT_studios_total_l1906_190622


namespace NUMINAMATH_GPT_find_x2_div_c2_add_y2_div_a2_add_z2_div_b2_l1906_190670

variable (a b c x y z : ℝ)

theorem find_x2_div_c2_add_y2_div_a2_add_z2_div_b2 
  (h1 : a * (x / c) + b * (y / a) + c * (z / b) = 5) 
  (h2 : c / x + a / y + b / z = 0) : 
  x^2 / c^2 + y^2 / a^2 + z^2 / b^2 = 25 := 
sorry

end NUMINAMATH_GPT_find_x2_div_c2_add_y2_div_a2_add_z2_div_b2_l1906_190670


namespace NUMINAMATH_GPT_sum_of_roots_is_three_l1906_190692

theorem sum_of_roots_is_three :
  ∀ (x1 x2 : ℝ), (x1^2 - 3 * x1 - 4 = 0) ∧ (x2^2 - 3 * x2 - 4 = 0) → x1 + x2 = 3 :=
by sorry

end NUMINAMATH_GPT_sum_of_roots_is_three_l1906_190692


namespace NUMINAMATH_GPT_avg_length_remaining_wires_l1906_190614

theorem avg_length_remaining_wires (N : ℕ) (avg_length : ℕ) 
    (third_wires_count : ℕ) (third_wires_avg_length : ℕ) 
    (total_length : ℕ := N * avg_length) 
    (third_wires_total_length : ℕ := third_wires_count * third_wires_avg_length) 
    (remaining_wires_count : ℕ := N - third_wires_count) 
    (remaining_wires_total_length : ℕ := total_length - third_wires_total_length) :
    N = 6 → 
    avg_length = 80 → 
    third_wires_count = 2 → 
    third_wires_avg_length = 70 → 
    remaining_wires_count = 4 → 
    remaining_wires_total_length / remaining_wires_count = 85 :=
by 
  intros hN hAvg hThirdCount hThirdAvg hRemainingCount
  sorry

end NUMINAMATH_GPT_avg_length_remaining_wires_l1906_190614


namespace NUMINAMATH_GPT_new_total_lifting_capacity_is_correct_l1906_190657

-- Define the initial lifting capacities and improvements
def initial_clean_and_jerk : ℕ := 80
def initial_snatch : ℕ := 50
def clean_and_jerk_multiplier : ℕ := 2
def snatch_increment_percentage : ℕ := 80

-- Calculated values
def new_clean_and_jerk := initial_clean_and_jerk * clean_and_jerk_multiplier
def snatch_increment := initial_snatch * snatch_increment_percentage / 100
def new_snatch := initial_snatch + snatch_increment
def new_total_lifting_capacity := new_clean_and_jerk + new_snatch

-- Theorem statement to be proven
theorem new_total_lifting_capacity_is_correct :
  new_total_lifting_capacity = 250 := 
sorry

end NUMINAMATH_GPT_new_total_lifting_capacity_is_correct_l1906_190657


namespace NUMINAMATH_GPT_number_of_SUVs_washed_l1906_190625

theorem number_of_SUVs_washed (charge_car charge_truck charge_SUV total_raised : ℕ) (num_trucks num_cars S : ℕ) :
  charge_car = 5 →
  charge_truck = 6 →
  charge_SUV = 7 →
  total_raised = 100 →
  num_trucks = 5 →
  num_cars = 7 →
  total_raised = num_cars * charge_car + num_trucks * charge_truck + S * charge_SUV →
  S = 5 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end NUMINAMATH_GPT_number_of_SUVs_washed_l1906_190625


namespace NUMINAMATH_GPT_find_g2_l1906_190683

theorem find_g2
  (g : ℝ → ℝ)
  (h : ∀ x : ℝ, x ≠ 0 → 4 * g x - 3 * g (1 / x) = x ^ 2) :
  g 2 = 19 / 16 := 
sorry

end NUMINAMATH_GPT_find_g2_l1906_190683


namespace NUMINAMATH_GPT_average_cd_l1906_190660

theorem average_cd (c d : ℝ) (h : (4 + 6 + 8 + c + d) / 5 = 18) : (c + d) / 2 = 36 := 
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_average_cd_l1906_190660


namespace NUMINAMATH_GPT_minimum_m_minus_n_l1906_190634

theorem minimum_m_minus_n (m n : ℕ) (hm : m > n) (h : (9^m) % 100 = (9^n) % 100) : m - n = 10 := 
sorry

end NUMINAMATH_GPT_minimum_m_minus_n_l1906_190634


namespace NUMINAMATH_GPT_product_of_triangle_areas_not_end_in_1988_l1906_190661

theorem product_of_triangle_areas_not_end_in_1988
  (a b c d : ℕ)
  (h1 : a * c = b * d)
  (hp : (a * b * c * d) = (a * c)^2)
  : ¬(∃ k : ℕ, (a * b * c * d) = 10000 * k + 1988) :=
sorry

end NUMINAMATH_GPT_product_of_triangle_areas_not_end_in_1988_l1906_190661


namespace NUMINAMATH_GPT_triangle_formation_l1906_190667

theorem triangle_formation (x₁ x₂ x₃ x₄ : ℝ) 
  (h₁ : x₁ ≠ x₂) (h₂ : x₁ ≠ x₃) (h₃ : x₁ ≠ x₄) (h₄ : x₂ ≠ x₃) (h₅ : x₂ ≠ x₄) (h₆ : x₃ ≠ x₄)
  (h₇ : 0 < x₁) (h₈ : 0 < x₂) (h₉ : 0 < x₃) (h₁₀ : 0 < x₄)
  (h₁₁ : (x₁ + x₂ + x₃ + x₄) * (1/x₁ + 1/x₂ + 1/x₃ + 1/x₄) < 17) :
  (x₁ + x₂ > x₃) ∧ (x₂ + x₃ > x₄) ∧ (x₁ + x₃ > x₂) ∧ 
  (x₁ + x₄ > x₃) ∧ (x₁ + x₂ > x₄) ∧ (x₃ + x₄ > x₁) ∧ 
  (x₂ + x₄ > x₁) ∧ (x₂ + x₃ > x₁) :=
sorry

end NUMINAMATH_GPT_triangle_formation_l1906_190667


namespace NUMINAMATH_GPT_log_identity_proof_l1906_190664

theorem log_identity_proof (lg : ℝ → ℝ) (h1 : lg 50 = lg 2 + lg 25) (h2 : lg 25 = 2 * lg 5) :
  (lg 2)^2 + lg 2 * lg 50 + lg 25 = 2 :=
by sorry

end NUMINAMATH_GPT_log_identity_proof_l1906_190664


namespace NUMINAMATH_GPT_graph_of_equation_is_two_lines_l1906_190646

theorem graph_of_equation_is_two_lines :
  ∀ (x y : ℝ), (x * y - 2 * x + 3 * y - 6 = 0) ↔ ((x + 3 = 0) ∨ (y - 2 = 0)) := 
by
  intro x y
  sorry

end NUMINAMATH_GPT_graph_of_equation_is_two_lines_l1906_190646


namespace NUMINAMATH_GPT_min_pieces_per_orange_l1906_190623

theorem min_pieces_per_orange (oranges : ℕ) (calories_per_orange : ℕ) (people : ℕ) (calories_per_person : ℕ) (pieces_per_orange : ℕ) :
  oranges = 5 →
  calories_per_orange = 80 →
  people = 4 →
  calories_per_person = 100 →
  pieces_per_orange ≥ 4 :=
by
  intro h_oranges h_calories_per_orange h_people h_calories_per_person
  sorry

end NUMINAMATH_GPT_min_pieces_per_orange_l1906_190623


namespace NUMINAMATH_GPT_ratio_of_x_to_y_l1906_190637

variable (x y : ℝ)

theorem ratio_of_x_to_y (h : 3 * x = 0.12 * 250 * y) : x / y = 10 :=
sorry

end NUMINAMATH_GPT_ratio_of_x_to_y_l1906_190637


namespace NUMINAMATH_GPT_geometric_sequence_q_cubed_l1906_190631

noncomputable def S (a_1 q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a_1 else a_1 * (1 - q^n) / (1 - q)

theorem geometric_sequence_q_cubed (a_1 q : ℝ) (h1 : q ≠ 1) (h2 : a_1 ≠ 0)
  (h3 : S a_1 q 3 + S a_1 q 6 = 2 * S a_1 q 9) : q^3 = -1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_q_cubed_l1906_190631


namespace NUMINAMATH_GPT_difference_students_pets_in_all_classrooms_l1906_190613

-- Definitions of the conditions
def students_per_classroom : ℕ := 24
def rabbits_per_classroom : ℕ := 3
def guinea_pigs_per_classroom : ℕ := 2
def number_of_classrooms : ℕ := 5

-- Proof problem statement
theorem difference_students_pets_in_all_classrooms :
  (students_per_classroom * number_of_classrooms) - 
  ((rabbits_per_classroom + guinea_pigs_per_classroom) * number_of_classrooms) = 95 := by
  sorry

end NUMINAMATH_GPT_difference_students_pets_in_all_classrooms_l1906_190613


namespace NUMINAMATH_GPT_smallest_x_l1906_190624

theorem smallest_x (x y : ℕ) (h_pos: x > 0 ∧ y > 0) (h_eq: 8 / 10 = y / (186 + x)) : x = 4 :=
sorry

end NUMINAMATH_GPT_smallest_x_l1906_190624


namespace NUMINAMATH_GPT_percent_calculation_l1906_190699

theorem percent_calculation (Part Whole : ℝ) (hPart : Part = 14) (hWhole : Whole = 70) : 
  (Part / Whole) * 100 = 20 := 
by 
  sorry

end NUMINAMATH_GPT_percent_calculation_l1906_190699


namespace NUMINAMATH_GPT_difference_of_squares_divisible_by_9_l1906_190693

theorem difference_of_squares_divisible_by_9 (a b : ℤ) : ∃ k : ℤ, (3 * a + 2)^2 - (3 * b + 2)^2 = 9 * k := by
  sorry

end NUMINAMATH_GPT_difference_of_squares_divisible_by_9_l1906_190693


namespace NUMINAMATH_GPT_max_value_correct_l1906_190645

noncomputable def max_value_ineq (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : 3 * x + 2 * y + 6 * z = 1) : Prop :=
  x ^ 4 * y ^ 3 * z ^ 2 ≤ 1 / 372008

theorem max_value_correct (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : 3 * x + 2 * y + 6 * z = 1) :
  max_value_ineq x y z h1 h2 h3 h4 :=
sorry

end NUMINAMATH_GPT_max_value_correct_l1906_190645


namespace NUMINAMATH_GPT_minimum_a3_b3_no_exist_a_b_2a_3b_eq_6_l1906_190612

-- Define the conditions once to reuse them for both proof statements.
variables {a b : ℝ} (ha: a > 0) (hb: b > 0) (h: (1/a) + (1/b) = Real.sqrt (a * b))

-- Problem (I)
theorem minimum_a3_b3 (h : (1/a) + (1/b) = Real.sqrt (a * b)) (ha: a > 0) (hb: b > 0) :
  a^3 + b^3 = 4 * Real.sqrt 2 := 
sorry

-- Problem (II)
theorem no_exist_a_b_2a_3b_eq_6 (h : (1/a) + (1/b) = Real.sqrt (a * b)) (ha: a > 0) (hb: b > 0) :
  ¬ ∃ (a b : ℝ), 2 * a + 3 * b = 6 :=
sorry

end NUMINAMATH_GPT_minimum_a3_b3_no_exist_a_b_2a_3b_eq_6_l1906_190612


namespace NUMINAMATH_GPT_roots_sum_cubes_l1906_190678

theorem roots_sum_cubes (a b c d : ℝ) 
  (h_eqn : ∀ x : ℝ, (x = a ∨ x = b ∨ x = c ∨ x = d) → 
    3 * x^4 + 6 * x^3 + 1002 * x^2 + 2005 * x + 4010 = 0) :
  (a + b)^3 + (b + c)^3 + (c + d)^3 + (d + a)^3 = 9362 :=
by { sorry }

end NUMINAMATH_GPT_roots_sum_cubes_l1906_190678


namespace NUMINAMATH_GPT_prices_proof_sales_revenue_proof_l1906_190649

-- Definitions for the prices and quantities
def price_peanut_oil := 50
def price_corn_oil := 40

-- Conditions from the problem
def condition1 (x y : ℕ) : Prop := 20 * x + 30 * y = 2200
def condition2 (x y : ℕ) : Prop := 30 * x + 10 * y = 1900
def purchased_peanut_oil := 50
def selling_price_peanut_oil := 60

-- Proof statement for Part 1
theorem prices_proof : ∃ (x y : ℕ), condition1 x y ∧ condition2 x y ∧ x = price_peanut_oil ∧ y = price_corn_oil :=
sorry

-- Proof statement for Part 2
theorem sales_revenue_proof : ∃ (m : ℕ), (selling_price_peanut_oil * m > price_peanut_oil * purchased_peanut_oil) ∧ m = 42 :=
sorry

end NUMINAMATH_GPT_prices_proof_sales_revenue_proof_l1906_190649


namespace NUMINAMATH_GPT_passed_percentage_l1906_190653

theorem passed_percentage (A B C AB BC AC ABC: ℝ) 
  (hA : A = 0.25) 
  (hB : B = 0.50) 
  (hC : C = 0.30) 
  (hAB : AB = 0.25) 
  (hBC : BC = 0.15) 
  (hAC : AC = 0.10) 
  (hABC : ABC = 0.05) 
  : 100 - (A + B + C - AB - BC - AC + ABC) = 40 := 
by 
  rw [hA, hB, hC, hAB, hBC, hAC, hABC]
  norm_num
  sorry

end NUMINAMATH_GPT_passed_percentage_l1906_190653


namespace NUMINAMATH_GPT_red_box_position_l1906_190635

theorem red_box_position (n : ℕ) (pos_smallest_to_largest : ℕ) (pos_largest_to_smallest : ℕ) 
  (h1 : n = 45) 
  (h2 : pos_smallest_to_largest = 29) 
  (h3 : pos_largest_to_smallest = n - (pos_smallest_to_largest - 1)) :
  pos_largest_to_smallest = 17 := 
  by
    -- This proof is missing; implementation goes here
    sorry

end NUMINAMATH_GPT_red_box_position_l1906_190635


namespace NUMINAMATH_GPT_number_to_match_l1906_190602

def twenty_five_percent_less (x: ℕ) : ℕ := 3 * x / 4

def one_third_more (n: ℕ) : ℕ := 4 * n / 3

theorem number_to_match (n : ℕ) (x : ℕ) 
  (h1 : x = 80) 
  (h2 : one_third_more n = twenty_five_percent_less x) : n = 45 :=
by
  -- Proof is skipped as per the instruction
  sorry

end NUMINAMATH_GPT_number_to_match_l1906_190602

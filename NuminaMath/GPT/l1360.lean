import Mathlib

namespace NUMINAMATH_GPT_non_isosceles_triangle_has_equidistant_incenter_midpoints_l1360_136020

structure Triangle (α : Type*) :=
(a b c : α)
(incenter : α)
(midpoint_a_b : α)
(midpoint_b_c : α)
(midpoint_c_a : α)
(equidistant : Bool)
(non_isosceles : Bool)

-- Define the triangle with the specified properties.
noncomputable def counterexample_triangle : Triangle ℝ :=
{ a := 3,
  b := 4,
  c := 5, 
  incenter := 1, -- incenter length for the right triangle.
  midpoint_a_b := 2.5,
  midpoint_b_c := 2,
  midpoint_c_a := 1.5,
  equidistant := true,    -- midpoints of two sides are equidistant from incenter
  non_isosceles := true } -- the triangle is not isosceles

theorem non_isosceles_triangle_has_equidistant_incenter_midpoints :
  ∃ (T : Triangle ℝ), T.equidistant ∧ T.non_isosceles := by
  use counterexample_triangle
  sorry

end NUMINAMATH_GPT_non_isosceles_triangle_has_equidistant_incenter_midpoints_l1360_136020


namespace NUMINAMATH_GPT_james_ali_difference_l1360_136053

theorem james_ali_difference (J A T : ℝ) (h1 : J = 145) (h2 : T = 250) (h3 : J + A = T) :
  J - A = 40 :=
by
  sorry

end NUMINAMATH_GPT_james_ali_difference_l1360_136053


namespace NUMINAMATH_GPT_proportionate_enlargement_l1360_136040

theorem proportionate_enlargement 
  (original_width original_height new_width : ℕ)
  (h_orig_width : original_width = 3)
  (h_orig_height : original_height = 2)
  (h_new_width : new_width = 12) : 
  ∃ (new_height : ℕ), new_height = 8 :=
by
  -- sorry to skip proof
  sorry

end NUMINAMATH_GPT_proportionate_enlargement_l1360_136040


namespace NUMINAMATH_GPT_proof_problem_l1360_136058

noncomputable def p : Prop := ∃ (α : ℝ), Real.cos (Real.pi - α) = Real.cos α
def q : Prop := ∀ (x : ℝ), x ^ 2 + 1 > 0

theorem proof_problem : p ∨ q := 
by
  sorry

end NUMINAMATH_GPT_proof_problem_l1360_136058


namespace NUMINAMATH_GPT_range_of_a_l1360_136014

noncomputable def f (x a : ℝ) : ℝ := x^2 + 2*x - a
noncomputable def g (x : ℝ) : ℝ := 2*x + 2 * Real.log x
noncomputable def h (x : ℝ) : ℝ := x^2 - 2 * Real.log x

theorem range_of_a (a : ℝ) :
  (∀ x y, (1 / Real.exp 1) ≤ x ∧ x ≤ Real.exp 1 ∧ (1 / Real.exp 1) ≤ y ∧ y ≤ Real.exp 1 ∧ f x a = g x ∧ f y a = g y → x ≠ y) →
  1 < a ∧ a ≤ (1 / Real.exp 2) + 2 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1360_136014


namespace NUMINAMATH_GPT_absolute_value_equation_solution_l1360_136084

theorem absolute_value_equation_solution (x : ℝ) : |x - 30| + |x - 24| = |3 * x - 72| ↔ x = 26 :=
by sorry

end NUMINAMATH_GPT_absolute_value_equation_solution_l1360_136084


namespace NUMINAMATH_GPT_particle_paths_l1360_136066

open Nat

-- Define the conditions of the problem
def move_right (a b : ℕ) : ℕ × ℕ := (a + 1, b)
def move_up (a b : ℕ) : ℕ × ℕ := (a, b + 1)
def move_diagonal (a b : ℕ) : ℕ × ℕ := (a + 1, b + 1)

-- Define a function to count paths without right-angle turns
noncomputable def count_paths (n : ℕ) : ℕ :=
  if n = 6 then 247 else 0

-- The theorem to be proven
theorem particle_paths :
  count_paths 6 = 247 :=
  sorry

end NUMINAMATH_GPT_particle_paths_l1360_136066


namespace NUMINAMATH_GPT_total_floors_l1360_136047

theorem total_floors (P Q R S T X F : ℕ) (h1 : 1 < X) (h2 : X < 50) :
  F = 1 + P - Q + R - S + T + X :=
sorry

end NUMINAMATH_GPT_total_floors_l1360_136047


namespace NUMINAMATH_GPT_find_some_number_l1360_136054

theorem find_some_number (some_number q x y : ℤ) 
  (h1 : x = some_number + 2 * q) 
  (h2 : y = 4 * q + 41) 
  (h3 : q = 7) 
  (h4 : x = y) : 
  some_number = 55 := 
by 
  sorry

end NUMINAMATH_GPT_find_some_number_l1360_136054


namespace NUMINAMATH_GPT_tangent_y_intercept_range_l1360_136060

theorem tangent_y_intercept_range :
  ∀ (x₀ : ℝ), (∃ y₀ : ℝ, y₀ = Real.exp x₀ ∧ (∃ m : ℝ, m = Real.exp x₀ ∧ ∃ b : ℝ, b = Real.exp x₀ * (1 - x₀) ∧ b < 0)) → x₀ > 1 := by
  sorry

end NUMINAMATH_GPT_tangent_y_intercept_range_l1360_136060


namespace NUMINAMATH_GPT_distinct_terms_in_expansion_l1360_136032

theorem distinct_terms_in_expansion:
  (∀ (x y z u v w: ℝ), (x + y + z) * (u + v + w + x + y) = 0 → false) →
  3 * 5 = 15 := by sorry

end NUMINAMATH_GPT_distinct_terms_in_expansion_l1360_136032


namespace NUMINAMATH_GPT_correct_calculation_l1360_136097

def original_number (x : ℕ) : Prop := x + 12 = 48

theorem correct_calculation (x : ℕ) (h : original_number x) : x + 22 = 58 := by
  sorry

end NUMINAMATH_GPT_correct_calculation_l1360_136097


namespace NUMINAMATH_GPT_computation_l1360_136081

theorem computation :
  (13 + 12)^2 - (13 - 12)^2 = 624 :=
by
  sorry

end NUMINAMATH_GPT_computation_l1360_136081


namespace NUMINAMATH_GPT_combined_hits_and_misses_total_l1360_136070

/-
  Prove that given the conditions for each day regarding the number of misses and
  the ratio of misses to hits, the combined total of hits and misses for the 
  three days is 322.
-/

theorem combined_hits_and_misses_total :
  (∀ (H1 : ℕ) (H2 : ℕ) (H3 : ℕ), 
    (2 * H1 = 60) ∧ (3 * H2 = 84) ∧ (5 * H3 = 100) →
    60 + 84 + 100 + H1 + H2 + H3 = 322) :=
by
  sorry

end NUMINAMATH_GPT_combined_hits_and_misses_total_l1360_136070


namespace NUMINAMATH_GPT_base_b_square_of_integer_l1360_136033

theorem base_b_square_of_integer (b : ℕ) (h : b > 4) : ∃ n : ℕ, (n * n) = b^2 + 4 * b + 4 :=
by 
  sorry

end NUMINAMATH_GPT_base_b_square_of_integer_l1360_136033


namespace NUMINAMATH_GPT_coin_flip_probability_l1360_136056

theorem coin_flip_probability : 
  ∀ (prob_tails : ℚ) (seq : List (Bool × ℚ)),
    prob_tails = 1/2 →
    seq = [(true, 1/2), (true, 1/2), (false, 1/2), (false, 1/2)] →
    (seq.map Prod.snd).prod = 0.0625 :=
by 
  intros prob_tails seq htails hseq 
  sorry

end NUMINAMATH_GPT_coin_flip_probability_l1360_136056


namespace NUMINAMATH_GPT_area_of_triangle_with_sides_13_12_5_l1360_136021

theorem area_of_triangle_with_sides_13_12_5 :
  let a := 13
  let b := 12
  let c := 5
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  area = 30 :=
by
  let a := 13
  let b := 12
  let c := 5
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  sorry

end NUMINAMATH_GPT_area_of_triangle_with_sides_13_12_5_l1360_136021


namespace NUMINAMATH_GPT_no_solutions_for_a3_plus_5b3_eq_2016_l1360_136031

theorem no_solutions_for_a3_plus_5b3_eq_2016 (a b : ℤ) : a^3 + 5 * b^3 ≠ 2016 :=
by sorry

end NUMINAMATH_GPT_no_solutions_for_a3_plus_5b3_eq_2016_l1360_136031


namespace NUMINAMATH_GPT_floor_neg_7_over_4_l1360_136052

theorem floor_neg_7_over_4 : (Int.floor (-7 / 4 : ℚ)) = -2 := 
by
  sorry

end NUMINAMATH_GPT_floor_neg_7_over_4_l1360_136052


namespace NUMINAMATH_GPT_initial_peanuts_l1360_136019

-- Definitions based on conditions
def peanuts_added := 8
def total_peanuts_now := 12

-- Statement to prove
theorem initial_peanuts (initial_peanuts : ℕ) (h : initial_peanuts + peanuts_added = total_peanuts_now) : initial_peanuts = 4 :=
sorry

end NUMINAMATH_GPT_initial_peanuts_l1360_136019


namespace NUMINAMATH_GPT_problem_l1360_136006

def Y (a b : ℤ) : ℤ := a^2 - 2 * a * b + b^2
def Z (a b : ℤ) : ℤ := a * b + a + b

theorem problem
  : Z (Y 5 3) (Y 2 1) = 9 := by
  sorry

end NUMINAMATH_GPT_problem_l1360_136006


namespace NUMINAMATH_GPT_probability_of_non_defective_is_seven_ninetyninths_l1360_136015

-- Define the number of total pencils, defective pencils, and the number of pencils selected
def total_pencils : ℕ := 12
def defective_pencils : ℕ := 4
def selected_pencils : ℕ := 5

-- Define the number of ways to choose k elements from n elements (the combination function)
def combination (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Calculate the total number of ways to choose 5 pencils out of 12
def total_ways : ℕ := combination total_pencils selected_pencils

-- Calculate the number of non-defective pencils
def non_defective_pencils : ℕ := total_pencils - defective_pencils

-- Calculate the number of ways to choose 5 non-defective pencils out of 8
def non_defective_ways : ℕ := combination non_defective_pencils selected_pencils

-- Calculate the probability that all 5 chosen pencils are non-defective
def probability_non_defective : ℚ :=
  non_defective_ways / total_ways

-- Prove that this probability equals 7/99
theorem probability_of_non_defective_is_seven_ninetyninths :
  probability_non_defective = 7 / 99 :=
by
  -- The proof is left as an exercise
  sorry

end NUMINAMATH_GPT_probability_of_non_defective_is_seven_ninetyninths_l1360_136015


namespace NUMINAMATH_GPT_valid_two_digit_numbers_l1360_136082

def is_valid_two_digit_number_pair (a b : ℕ) : Prop :=
  0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ a > b ∧ (Nat.gcd (10 * a + b) (10 * b + a) = a^2 - b^2)

theorem valid_two_digit_numbers :
  (is_valid_two_digit_number_pair 2 1 ∨ is_valid_two_digit_number_pair 5 4) ∧
  ∀ a b, is_valid_two_digit_number_pair a b → (a = 2 ∧ b = 1 ∨ a = 5 ∧ b = 4) :=
by
  sorry

end NUMINAMATH_GPT_valid_two_digit_numbers_l1360_136082


namespace NUMINAMATH_GPT_find_original_number_l1360_136048

theorem find_original_number (x : ℤ) : 4 * (3 * x + 29) = 212 → x = 8 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_original_number_l1360_136048


namespace NUMINAMATH_GPT_cannot_determine_exact_insect_l1360_136061

-- Defining the conditions as premises
def insect_legs : ℕ := 6

def total_legs_two_insects (legs_per_insect : ℕ) (num_insects : ℕ) : ℕ :=
  legs_per_insect * num_insects

-- Statement: Proving that given just the number of legs, we cannot determine the exact type of insect
theorem cannot_determine_exact_insect (legs : ℕ) (num_insects : ℕ) (h1 : legs = 6) (h2 : num_insects = 2) (h3 : total_legs_two_insects legs num_insects = 12) :
  ∃ insect_type, insect_type :=
by
  sorry

end NUMINAMATH_GPT_cannot_determine_exact_insect_l1360_136061


namespace NUMINAMATH_GPT_product_of_y_coordinates_l1360_136038

theorem product_of_y_coordinates (k : ℝ) (hk : k > 0) :
    let y1 := 2 + Real.sqrt (k^2 - 64)
    let y2 := 2 - Real.sqrt (k^2 - 64)
    y1 * y2 = 68 - k^2 :=
by 
  sorry

end NUMINAMATH_GPT_product_of_y_coordinates_l1360_136038


namespace NUMINAMATH_GPT_even_perfect_squares_between_50_and_200_l1360_136098

theorem even_perfect_squares_between_50_and_200 : ∃ s : Finset ℕ, 
  (∀ n ∈ s, (n^2 ≥ 50) ∧ (n^2 ≤ 200) ∧ n^2 % 2 = 0) ∧ s.card = 4 := by
  sorry

end NUMINAMATH_GPT_even_perfect_squares_between_50_and_200_l1360_136098


namespace NUMINAMATH_GPT_perfect_square_difference_of_solutions_l1360_136027

theorem perfect_square_difference_of_solutions
  (x y : ℤ) (hx : x > 0) (hy : y > 0) 
  (h : 3 * x^2 + x = 4 * y^2 + y) : ∃ k : ℤ, k^2 = x - y := 
sorry

end NUMINAMATH_GPT_perfect_square_difference_of_solutions_l1360_136027


namespace NUMINAMATH_GPT_range_of_b_l1360_136018

noncomputable def f (x b : ℝ) : ℝ := -1/2 * (x - 2)^2 + b * Real.log (x + 2)
noncomputable def derivative (x b : ℝ) := -(x - 2) + b / (x + 2)

-- Lean theorem statement
theorem range_of_b (b : ℝ) :
  (∀ x > 1, derivative x b ≤ 0) → b ≤ -3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_b_l1360_136018


namespace NUMINAMATH_GPT_min_value_xy_inv_xy_l1360_136075

theorem min_value_xy_inv_xy (x y : ℝ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hxy_sum : x + y = 2) :
  ∃ m : ℝ, m = xy + 4 / xy ∧ m ≥ 5 :=
by
  sorry

end NUMINAMATH_GPT_min_value_xy_inv_xy_l1360_136075


namespace NUMINAMATH_GPT_find_a_l1360_136080

theorem find_a (a : ℝ) (h : (1 - 2016 * a) = 2017) : a = -1 := by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_find_a_l1360_136080


namespace NUMINAMATH_GPT_distribute_candies_l1360_136005

theorem distribute_candies (n : ℕ) (h : ∃ m : ℕ, n = 2^m) : 
  ∀ k : ℕ, ∃ i : ℕ, (1 / 2) * i * (i + 1) % n = k :=
sorry

end NUMINAMATH_GPT_distribute_candies_l1360_136005


namespace NUMINAMATH_GPT_find_a_and_b_minimum_value_of_polynomial_l1360_136089

noncomputable def polynomial_has_maximum (x y a b : ℝ) : Prop :=
  y = a * x ^ 3 + b * x ^ 2 ∧ x = 1 ∧ y = 3

noncomputable def polynomial_minimum_value (y : ℝ) : Prop :=
  y = 0

theorem find_a_and_b (a b x y : ℝ) (h : polynomial_has_maximum x y a b) :
  a = -6 ∧ b = 9 :=
by sorry

theorem minimum_value_of_polynomial (a b y : ℝ) (h : a = -6 ∧ b = 9) :
  polynomial_minimum_value y :=
by sorry

end NUMINAMATH_GPT_find_a_and_b_minimum_value_of_polynomial_l1360_136089


namespace NUMINAMATH_GPT_least_positive_integer_l1360_136013

theorem least_positive_integer (x : ℕ) :
  (∃ k : ℤ, (3 * x + 41) ^ 2 = 53 * k) ↔ x = 4 :=
by
  sorry

end NUMINAMATH_GPT_least_positive_integer_l1360_136013


namespace NUMINAMATH_GPT_exists_city_reaching_all_l1360_136001

variables {City : Type} (canReach : City → City → Prop)

-- Conditions from the problem
axiom reach_itself (A : City) : canReach A A
axiom reach_transitive {A B C : City} : canReach A B → canReach B C → canReach A C
axiom reach_any_two {P Q : City} : ∃ R : City, canReach R P ∧ canReach R Q

-- The proof problem
theorem exists_city_reaching_all (cities : City → Prop) :
  (∀ P Q, P ≠ Q → cities P → cities Q → ∃ R, cities R ∧ canReach R P ∧ canReach R Q) →
  ∃ C, ∀ A, cities A → canReach C A :=
by
  intros H
  sorry

end NUMINAMATH_GPT_exists_city_reaching_all_l1360_136001


namespace NUMINAMATH_GPT_cows_in_group_l1360_136094

variable (c h : ℕ)

/--
In a group of cows and chickens, the number of legs was 20 more than twice the number of heads.
Cows have 4 legs each and chickens have 2 legs each.
Each animal has one head.
-/
theorem cows_in_group (h : ℕ) (hc : 4 * c + 2 * h = 2 * (c + h) + 20) : c = 10 :=
by
  sorry

end NUMINAMATH_GPT_cows_in_group_l1360_136094


namespace NUMINAMATH_GPT_ratio_A_B_l1360_136049

-- Given conditions as definitions
def P_both : ℕ := 500  -- Number of people who purchased both books A and B

def P_only_B : ℕ := P_both / 2  -- Number of people who purchased only book B

def P_only_A : ℕ := 1000  -- Number of people who purchased only book A

-- Total number of people who purchased books
def P_A : ℕ := P_only_A + P_both  -- Total number of people who purchased book A

def P_B : ℕ := P_only_B + P_both  -- Total number of people who purchased book B

-- The ratio of people who purchased book A to book B
theorem ratio_A_B : P_A / P_B = 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_A_B_l1360_136049


namespace NUMINAMATH_GPT_triangle_side_lengths_l1360_136091

theorem triangle_side_lengths (a b c : ℝ) (h1 : a / b = 3 / 4) (h2 : b / c = 4 / 5) (h3 : a + b + c = 60) :
  (a = 15 ∧ b = 20 ∧ c = 25) :=
sorry

end NUMINAMATH_GPT_triangle_side_lengths_l1360_136091


namespace NUMINAMATH_GPT_tony_rope_length_l1360_136071

-- Definitions based on the conditions in the problem
def rope_lengths : List ℝ := [8, 20, 2, 2, 2, 7]
def knot_loss_per_knot : ℝ := 1.2
def number_of_knots : ℕ := 5

-- The final length of the rope after tying all pieces together and losing length per knot
def final_rope_length (lengths : List ℝ) (loss_per_knot : ℝ) (number_of_knots : ℕ) : ℝ :=
  List.sum lengths - (loss_per_knot * number_of_knots)

theorem tony_rope_length :
  final_rope_length rope_lengths knot_loss_per_knot number_of_knots = 35 := by
  sorry

end NUMINAMATH_GPT_tony_rope_length_l1360_136071


namespace NUMINAMATH_GPT_find_number_l1360_136007

theorem find_number
  (P : ℝ) (R : ℝ) (hP : P = 0.0002) (hR : R = 2.4712) :
  (12356 * P = R) := by
  sorry

end NUMINAMATH_GPT_find_number_l1360_136007


namespace NUMINAMATH_GPT_saving_percentage_l1360_136051

variable (I S : Real)

-- Conditions
def cond1 : Prop := S = 0.3 * I -- Man saves 30% of his income

def cond2 : Prop := let income_next_year := 1.3 * I
                    let savings_next_year := 2 * S
                    let expenditure_first_year := I - S
                    let expenditure_second_year := income_next_year - savings_next_year
                    expenditure_first_year + expenditure_second_year = 2 * expenditure_first_year

-- Question
theorem saving_percentage :
  cond1 I S →
  cond2 I S →
  S = 0.3 * I :=
by
  intros
  sorry

end NUMINAMATH_GPT_saving_percentage_l1360_136051


namespace NUMINAMATH_GPT_length_of_train_is_correct_l1360_136034

noncomputable def length_of_train (speed : ℕ) (time : ℕ) : ℕ :=
  (speed * (time / 3600) * 1000)

theorem length_of_train_is_correct : length_of_train 70 36 = 700 := by
  sorry

end NUMINAMATH_GPT_length_of_train_is_correct_l1360_136034


namespace NUMINAMATH_GPT_line_always_passes_fixed_point_l1360_136093

theorem line_always_passes_fixed_point (m : ℝ) :
  m * 1 + (1 - m) * 2 + m - 2 = 0 :=
by
  sorry

end NUMINAMATH_GPT_line_always_passes_fixed_point_l1360_136093


namespace NUMINAMATH_GPT_complement_A_in_B_l1360_136095

-- Define the sets A and B
def A : Set ℕ := {2, 3}
def B : Set ℕ := {0, 1, 2, 3, 4}

-- Define the complement of A in B
def complement (U : Set ℕ) (A : Set ℕ) : Set ℕ := {x ∈ U | x ∉ A}

-- Statement to prove
theorem complement_A_in_B :
  complement B A = {0, 1, 4} := by
  sorry

end NUMINAMATH_GPT_complement_A_in_B_l1360_136095


namespace NUMINAMATH_GPT_cookies_and_sugar_needed_l1360_136050

-- Definitions derived from the conditions
def initial_cookies : ℕ := 24
def initial_flour : ℕ := 3
def initial_sugar : ℝ := 1.5
def flour_needed : ℕ := 5

-- The proof statement
theorem cookies_and_sugar_needed :
  (initial_cookies / initial_flour) * flour_needed = 40 ∧ (initial_sugar / initial_flour) * flour_needed = 2.5 :=
by
  sorry

end NUMINAMATH_GPT_cookies_and_sugar_needed_l1360_136050


namespace NUMINAMATH_GPT_correct_triangle_l1360_136090

-- Define the conditions for the sides of each option
def sides_A := (1, 2, 3)
def sides_B := (3, 4, 5)
def sides_C := (3, 1, 1)
def sides_D := (3, 4, 7)

-- Conditions for forming a triangle
def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Prove the problem statement
theorem correct_triangle : is_triangle 3 4 5 :=
by
  sorry

end NUMINAMATH_GPT_correct_triangle_l1360_136090


namespace NUMINAMATH_GPT_total_money_l1360_136055

theorem total_money (A B C : ℕ) (h1 : A + C = 400) (h2 : B + C = 750) (hC : C = 250) :
  A + B + C = 900 :=
sorry

end NUMINAMATH_GPT_total_money_l1360_136055


namespace NUMINAMATH_GPT_expenditure_should_increase_by_21_percent_l1360_136078

noncomputable def old_income := 100.0
noncomputable def ratio_exp_sav := (3 : ℝ) / (2 : ℝ)
noncomputable def income_increase_percent := 15.0 / 100.0
noncomputable def savings_increase_percent := 6.0 / 100.0
noncomputable def old_expenditure := old_income * (3 / (3 + 2))
noncomputable def old_savings := old_income * (2 / (3 + 2))
noncomputable def new_income := old_income * (1 + income_increase_percent)
noncomputable def new_savings := old_savings * (1 + savings_increase_percent)
noncomputable def new_expenditure := new_income - new_savings
noncomputable def expenditure_increase_percent := ((new_expenditure - old_expenditure) / old_expenditure) * 100

theorem expenditure_should_increase_by_21_percent :
  expenditure_increase_percent = 21 :=
sorry

end NUMINAMATH_GPT_expenditure_should_increase_by_21_percent_l1360_136078


namespace NUMINAMATH_GPT_calculate_expression_l1360_136010

theorem calculate_expression :
  (1.99^2 - 1.98 * 1.99 + 0.99^2 = 1) :=
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l1360_136010


namespace NUMINAMATH_GPT_total_seats_in_theater_l1360_136065

theorem total_seats_in_theater 
    (n : ℕ) 
    (a1 : ℕ)
    (an : ℕ)
    (d : ℕ)
    (h1 : a1 = 12)
    (h2 : d = 2)
    (h3 : an = 48)
    (h4 : an = a1 + (n - 1) * d) :
    (n = 19) →
    (2 * (a1 + an) * n / 2 = 570) :=
by
  intros
  sorry

end NUMINAMATH_GPT_total_seats_in_theater_l1360_136065


namespace NUMINAMATH_GPT_value_of_a_minus_b_l1360_136041

theorem value_of_a_minus_b 
  (a b : ℤ)
  (h1 : 1010 * a + 1014 * b = 1018)
  (h2 : 1012 * a + 1016 * b = 1020) : 
  a - b = -3 :=
sorry

end NUMINAMATH_GPT_value_of_a_minus_b_l1360_136041


namespace NUMINAMATH_GPT_evaluate_division_l1360_136004

theorem evaluate_division : 64 / 0.08 = 800 := by
  sorry

end NUMINAMATH_GPT_evaluate_division_l1360_136004


namespace NUMINAMATH_GPT_teacher_problems_remaining_l1360_136044

theorem teacher_problems_remaining (problems_per_worksheet : Nat) 
                                   (total_worksheets : Nat) 
                                   (graded_worksheets : Nat) 
                                   (remaining_problems : Nat)
  (h1 : problems_per_worksheet = 4)
  (h2 : total_worksheets = 9)
  (h3 : graded_worksheets = 5)
  (h4 : remaining_problems = total_worksheets * problems_per_worksheet - graded_worksheets * problems_per_worksheet) :
  remaining_problems = 16 :=
sorry

end NUMINAMATH_GPT_teacher_problems_remaining_l1360_136044


namespace NUMINAMATH_GPT_two_bedroom_units_l1360_136092

theorem two_bedroom_units (x y : ℕ) (h1 : x + y = 12) (h2 : 360 * x + 450 * y = 4950) : y = 7 :=
by
  sorry

end NUMINAMATH_GPT_two_bedroom_units_l1360_136092


namespace NUMINAMATH_GPT_trains_pass_each_other_time_l1360_136039

theorem trains_pass_each_other_time :
  ∃ t : ℝ, t = 240 / 191.171 := 
sorry

end NUMINAMATH_GPT_trains_pass_each_other_time_l1360_136039


namespace NUMINAMATH_GPT_billy_buys_bottle_l1360_136042

-- Definitions of costs and volumes
def money : ℝ := 10
def cost1 : ℝ := 1
def volume1 : ℝ := 10
def cost2 : ℝ := 2
def volume2 : ℝ := 16
def cost3 : ℝ := 2.5
def volume3 : ℝ := 25
def cost4 : ℝ := 5
def volume4 : ℝ := 50
def cost5 : ℝ := 10
def volume5 : ℝ := 200

-- Statement of the proof problem
theorem billy_buys_bottle : ∃ b : ℕ, b = 1 ∧ cost5 = money := by 
  sorry

end NUMINAMATH_GPT_billy_buys_bottle_l1360_136042


namespace NUMINAMATH_GPT_quadratic_has_two_distinct_real_roots_l1360_136012

theorem quadratic_has_two_distinct_real_roots :
  let a := (1 : ℝ)
  let b := (-5 : ℝ)
  let c := (-1 : ℝ)
  b^2 - 4 * a * c > 0 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_has_two_distinct_real_roots_l1360_136012


namespace NUMINAMATH_GPT_correct_operation_is_a_l1360_136072

theorem correct_operation_is_a (a b : ℝ) : 
  (a^4 * a^3 = a^7) ∧ 
  ((a^2)^3 ≠ a^5) ∧ 
  (3 * a^2 - a^2 ≠ 2) ∧ 
  ((a - b)^2 ≠ a^2 - b^2) := 
by {
  -- Here, you would fill in the proof
  sorry
}

end NUMINAMATH_GPT_correct_operation_is_a_l1360_136072


namespace NUMINAMATH_GPT_train_cross_time_l1360_136000

-- Define the conditions
def train_speed_kmhr := 52
def train_length_meters := 130

-- Conversion factor from km/hr to m/s
def kmhr_to_ms (speed_kmhr : ℕ) : ℕ := (speed_kmhr * 1000) / 3600

-- Speed of the train in m/s
def train_speed_ms := kmhr_to_ms train_speed_kmhr

-- Calculate time to cross the pole
def time_to_cross_pole (distance_m : ℕ) (speed_ms : ℕ) : ℕ := distance_m / speed_ms

-- The theorem to prove
theorem train_cross_time : time_to_cross_pole train_length_meters train_speed_ms = 9 := by sorry

end NUMINAMATH_GPT_train_cross_time_l1360_136000


namespace NUMINAMATH_GPT_jerry_sister_increase_temp_l1360_136002

theorem jerry_sister_increase_temp :
  let T0 := 40
  let T1 := 2 * T0
  let T2 := T1 - 30
  let T3 := T2 - 0.3 * T2
  let T4 := 59
  T4 - T3 = 24 := by
  sorry

end NUMINAMATH_GPT_jerry_sister_increase_temp_l1360_136002


namespace NUMINAMATH_GPT_quadratic_to_vertex_form_l1360_136057

theorem quadratic_to_vertex_form :
  ∀ (x : ℝ), (1/2) * x^2 - 2 * x + 1 = (1/2) * (x - 2)^2 - 1 :=
by
  intro x
  -- full proof omitted
  sorry

end NUMINAMATH_GPT_quadratic_to_vertex_form_l1360_136057


namespace NUMINAMATH_GPT_exists_bounding_constant_M_l1360_136083

variable (α : ℝ) (a : ℕ → ℝ)
variable (hα : α > 1)
variable (h_seq : ∀ n : ℕ, n > 0 →
  a n.succ = a n + (a n / n) ^ α)

theorem exists_bounding_constant_M (h_a1 : 0 < a 1 ∧ a 1 < 1) : 
  ∃ M, ∀ n > 0, a n ≤ M := 
sorry

end NUMINAMATH_GPT_exists_bounding_constant_M_l1360_136083


namespace NUMINAMATH_GPT_current_speed_l1360_136069

-- The main statement of our problem
theorem current_speed (v_with_current v_against_current c man_speed : ℝ) 
  (h1 : v_with_current = man_speed + c) 
  (h2 : v_against_current = man_speed - c) 
  (h_with : v_with_current = 15) 
  (h_against : v_against_current = 9.4) : 
  c = 2.8 :=
by
  sorry

end NUMINAMATH_GPT_current_speed_l1360_136069


namespace NUMINAMATH_GPT_pot_filling_time_l1360_136024

-- Define the given conditions
def drops_per_minute : ℕ := 3
def volume_per_drop : ℕ := 20 -- in ml
def pot_capacity : ℕ := 3000 -- in ml (3 liters * 1000 ml/liter)

-- Define the calculation for the drip rate
def drip_rate_per_minute : ℕ := drops_per_minute * volume_per_drop

-- Define the goal, i.e., how long it will take to fill the pot
def time_to_fill_pot (capacity : ℕ) (rate : ℕ) : ℕ := capacity / rate

-- Proof statement
theorem pot_filling_time :
  time_to_fill_pot pot_capacity drip_rate_per_minute = 50 := 
sorry

end NUMINAMATH_GPT_pot_filling_time_l1360_136024


namespace NUMINAMATH_GPT_max_odd_integers_l1360_136026

theorem max_odd_integers (a b c d e f : ℕ) 
  (hprod : a * b * c * d * e * f % 2 = 0) 
  (hpos_a : 0 < a) (hpos_b : 0 < b) 
  (hpos_c : 0 < c) (hpos_d : 0 < d) 
  (hpos_e : 0 < e) (hpos_f : 0 < f) : 
  ∃ x : ℕ, x ≤ 5 ∧ x = 5 :=
by sorry

end NUMINAMATH_GPT_max_odd_integers_l1360_136026


namespace NUMINAMATH_GPT_cube_inscribed_sphere_volume_l1360_136077

noncomputable def cubeSurfaceArea (a : ℝ) : ℝ := 6 * a^2
noncomputable def sphereVolume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3
noncomputable def inscribedSphereRadius (a : ℝ) : ℝ := a / 2

theorem cube_inscribed_sphere_volume :
  ∀ (a : ℝ), cubeSurfaceArea a = 24 → sphereVolume (inscribedSphereRadius a) = (4 / 3) * Real.pi := 
by 
  intros a h₁
  sorry

end NUMINAMATH_GPT_cube_inscribed_sphere_volume_l1360_136077


namespace NUMINAMATH_GPT_find_y_intercept_l1360_136029

theorem find_y_intercept (m : ℝ) 
  (h1 : ∀ x y : ℝ, y = 2 * x + m)
  (h2 : ∃ x y : ℝ, x = 1 ∧ y = 1 ∧ y = 2 * x + m) : 
  m = -1 := 
sorry

end NUMINAMATH_GPT_find_y_intercept_l1360_136029


namespace NUMINAMATH_GPT_bill_experience_now_l1360_136064

theorem bill_experience_now (B J : ℕ) 
  (h1 : J = 3 * B) 
  (h2 : J + 5 = 2 * (B + 5)) : B + 5 = 10 :=
by
  sorry

end NUMINAMATH_GPT_bill_experience_now_l1360_136064


namespace NUMINAMATH_GPT_mathieu_plot_area_l1360_136030

def total_area (x y : ℕ) : ℕ := x * x

theorem mathieu_plot_area :
  ∃ (x y : ℕ), (x^2 - y^2 = 464) ∧ (x - y = 8) ∧ (total_area x y = 1089) :=
by sorry

end NUMINAMATH_GPT_mathieu_plot_area_l1360_136030


namespace NUMINAMATH_GPT_taxi_ride_cost_l1360_136045

def baseFare : ℝ := 1.50
def costPerMile : ℝ := 0.25
def milesTraveled : ℕ := 5
def totalCost := baseFare + (costPerMile * milesTraveled)

/-- The cost of a 5-mile taxi ride is $2.75. -/
theorem taxi_ride_cost : totalCost = 2.75 := by
  sorry

end NUMINAMATH_GPT_taxi_ride_cost_l1360_136045


namespace NUMINAMATH_GPT_triangle_properties_l1360_136022

theorem triangle_properties (b c : ℝ) (C : ℝ)
  (hb : b = 10)
  (hc : c = 5 * Real.sqrt 6)
  (hC : C = Real.pi / 3) :
  let R := c / (2 * Real.sin C)
  let B := Real.arcsin (b * Real.sin C / c)
  R = 5 * Real.sqrt 2 ∧ B = Real.pi / 4 :=
by
  sorry

end NUMINAMATH_GPT_triangle_properties_l1360_136022


namespace NUMINAMATH_GPT_gcd_7654321_6789012_l1360_136085

theorem gcd_7654321_6789012 : Nat.gcd 7654321 6789012 = 3 := by
  sorry

end NUMINAMATH_GPT_gcd_7654321_6789012_l1360_136085


namespace NUMINAMATH_GPT_miyoung_largest_square_side_l1360_136079

theorem miyoung_largest_square_side :
  ∃ (G : ℕ), G > 0 ∧ ∀ (a b : ℕ), (a = 32) → (b = 74) → (gcd a b = G) → (G = 2) :=
by {
  sorry
}

end NUMINAMATH_GPT_miyoung_largest_square_side_l1360_136079


namespace NUMINAMATH_GPT_average_cups_of_tea_sold_l1360_136073

theorem average_cups_of_tea_sold (x_avg : ℝ) (y_regression : ℝ → ℝ) 
  (h1 : x_avg = 12) (h2 : ∀ x, y_regression x = -2*x + 58) : 
  y_regression x_avg = 34 := by
  sorry

end NUMINAMATH_GPT_average_cups_of_tea_sold_l1360_136073


namespace NUMINAMATH_GPT_find_savings_l1360_136063

-- Definitions of given conditions
def income : ℕ := 10000
def ratio_income_expenditure : ℕ × ℕ := (10, 8)

-- Proving the savings based on given conditions
theorem find_savings (income : ℕ) (ratio_income_expenditure : ℕ × ℕ) :
  let expenditure := (ratio_income_expenditure.2 * income) / ratio_income_expenditure.1
  let savings := income - expenditure
  savings = 2000 :=
by
  sorry

end NUMINAMATH_GPT_find_savings_l1360_136063


namespace NUMINAMATH_GPT_new_area_rhombus_l1360_136068

theorem new_area_rhombus (d1 d2 : ℝ) (h : (d1 * d2) / 2 = 3) : 
  ((5 * d1) * (5 * d2)) / 2 = 75 := 
by
  sorry

end NUMINAMATH_GPT_new_area_rhombus_l1360_136068


namespace NUMINAMATH_GPT_smallest_positive_x_l1360_136046

theorem smallest_positive_x (x : ℕ) (h : 42 * x + 9 ≡ 3 [MOD 15]) : x = 2 :=
sorry

end NUMINAMATH_GPT_smallest_positive_x_l1360_136046


namespace NUMINAMATH_GPT_possible_values_of_K_l1360_136088

theorem possible_values_of_K (K M : ℕ) (h : K * (K + 1) = M^2) (hM : M < 100) : K = 8 ∨ K = 35 :=
by sorry

end NUMINAMATH_GPT_possible_values_of_K_l1360_136088


namespace NUMINAMATH_GPT_age_sum_l1360_136008

variable (b : ℕ)
variable (a : ℕ := b + 2)
variable (c : ℕ := b / 2)

theorem age_sum : b = 10 → a + b + c = 27 :=
by
  intros h
  rw [h]
  sorry

end NUMINAMATH_GPT_age_sum_l1360_136008


namespace NUMINAMATH_GPT_num_factors_2012_l1360_136086

theorem num_factors_2012 : (Nat.factors 2012).length = 6 := by
  sorry

end NUMINAMATH_GPT_num_factors_2012_l1360_136086


namespace NUMINAMATH_GPT_complex_number_sum_l1360_136003

variable (ω : ℂ)
variable (h1 : ω^9 = 1)
variable (h2 : ω ≠ 1)

theorem complex_number_sum :
  ω^20 + ω^24 + ω^28 + ω^32 + ω^36 + ω^40 + ω^44 + ω^48 + ω^52 + ω^56 + ω^60 + ω^64 + ω^68 + ω^72 + ω^76 + ω^80 = ω^2 :=
by sorry

end NUMINAMATH_GPT_complex_number_sum_l1360_136003


namespace NUMINAMATH_GPT_stones_required_to_pave_hall_l1360_136062

theorem stones_required_to_pave_hall :
    let length_hall_m := 36
    let breadth_hall_m := 15
    let length_stone_dm := 3
    let breadth_stone_dm := 5
    let length_hall_dm := length_hall_m * 10
    let breadth_hall_dm := breadth_hall_m * 10
    let area_hall_dm2 := length_hall_dm * breadth_hall_dm
    let area_stone_dm2 := length_stone_dm * breadth_stone_dm
    (area_hall_dm2 / area_stone_dm2) = 3600 :=
by
    -- Definitions
    let length_hall_m := 36
    let breadth_hall_m := 15
    let length_stone_dm := 3
    let breadth_stone_dm := 5

    -- Convert to decimeters
    let length_hall_dm := length_hall_m * 10
    let breadth_hall_dm := breadth_hall_m * 10
    
    -- Calculate areas
    let area_hall_dm2 := length_hall_dm * breadth_hall_dm
    let area_stone_dm2 := length_stone_dm * breadth_stone_dm
    
    -- Calculate number of stones 
    let number_of_stones := area_hall_dm2 / area_stone_dm2

    -- Prove the required number of stones
    have h : number_of_stones = 3600 := sorry
    exact h

end NUMINAMATH_GPT_stones_required_to_pave_hall_l1360_136062


namespace NUMINAMATH_GPT_harly_dogs_final_count_l1360_136016

theorem harly_dogs_final_count (initial_dogs : ℕ) (adopted_percentage : ℕ) (returned_dogs : ℕ) (adoption_rate : adopted_percentage = 40) (initial_count : initial_dogs = 80) (returned_count : returned_dogs = 5) :
  initial_dogs - (initial_dogs * adopted_percentage / 100) + returned_dogs = 53 :=
by
  sorry

end NUMINAMATH_GPT_harly_dogs_final_count_l1360_136016


namespace NUMINAMATH_GPT_seventeen_power_seven_mod_eleven_l1360_136035

-- Define the conditions
def mod_condition : Prop := 17 % 11 = 6

-- Define the main goal (to prove the correct answer)
theorem seventeen_power_seven_mod_eleven (h : mod_condition) : (17^7) % 11 = 8 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_seventeen_power_seven_mod_eleven_l1360_136035


namespace NUMINAMATH_GPT_total_weight_correct_l1360_136059

-- Define the weights given in the problem
def dog_weight_kg := 2 -- weight in kilograms
def dog_weight_g := 600 -- additional grams
def cat_weight_g := 3700 -- weight in grams

-- Convert dog's weight to grams
def dog_weight_total_g : ℕ := dog_weight_kg * 1000 + dog_weight_g

-- Define the total weight of the animals (dog + cat)
def total_weight_animals_g : ℕ := dog_weight_total_g + cat_weight_g

-- Theorem stating that the total weight of the animals is 6300 grams
theorem total_weight_correct : total_weight_animals_g = 6300 := by
  sorry

end NUMINAMATH_GPT_total_weight_correct_l1360_136059


namespace NUMINAMATH_GPT_quadratic_complete_square_l1360_136096

/-- Given quadratic expression, complete the square to find the equivalent form
    and calculate the sum of the coefficients a, h, k. -/
theorem quadratic_complete_square (a h k : ℝ) :
  (∀ x : ℝ, 2 * x^2 - 8 * x + 2 = a * (x - h)^2 + k) → a + h + k = -2 :=
by
  intro h₁
  sorry

end NUMINAMATH_GPT_quadratic_complete_square_l1360_136096


namespace NUMINAMATH_GPT_min_expression_value_2023_l1360_136087

noncomputable def min_expr_val := ∀ x : ℝ, (x + 2) * (x + 3) * (x + 4) * (x + 5) + 2024 ≥ 2023

noncomputable def least_value : ℝ := 2023

theorem min_expression_value_2023 : min_expr_val ∧ (∃ x : ℝ, (x + 2) * (x + 3) * (x + 4) * (x + 5) + 2024 = least_value) := 
by sorry

end NUMINAMATH_GPT_min_expression_value_2023_l1360_136087


namespace NUMINAMATH_GPT_number_of_people_who_purchased_only_book_A_l1360_136017

theorem number_of_people_who_purchased_only_book_A (x y v : ℕ) 
  (h1 : 2 * x = 500)
  (h2 : y = x + 500)
  (h3 : v = 2 * y) : 
  v = 1500 := 
sorry

end NUMINAMATH_GPT_number_of_people_who_purchased_only_book_A_l1360_136017


namespace NUMINAMATH_GPT_tara_spent_more_on_icecream_l1360_136023

def iceCreamCount : ℕ := 19
def yoghurtCount : ℕ := 4
def iceCreamCost : ℕ := 7
def yoghurtCost : ℕ := 1

theorem tara_spent_more_on_icecream :
  (iceCreamCount * iceCreamCost) - (yoghurtCount * yoghurtCost) = 129 := 
  sorry

end NUMINAMATH_GPT_tara_spent_more_on_icecream_l1360_136023


namespace NUMINAMATH_GPT_percentage_increase_of_soda_l1360_136043

variable (C S x : ℝ)

theorem percentage_increase_of_soda
  (h1 : 1.25 * C = 10)
  (h2 : S + x * S = 12)
  (h3 : C + S = 16) :
  x = 0.5 :=
sorry

end NUMINAMATH_GPT_percentage_increase_of_soda_l1360_136043


namespace NUMINAMATH_GPT_identify_urea_decomposing_bacteria_l1360_136076

-- Definitions of different methods
def methodA (culture_medium : Type) := 
  ∃ (urea_only : culture_medium) (phenol_red : culture_medium), phenol_red = urea_only

def methodB (culture_medium : Type) := 
  ∃ (urea_only : culture_medium) (EMB_reagent : culture_medium), EMB_reagent = urea_only

def methodC (culture_medium : Type) := 
  ∃ (urea_only : culture_medium) (Sudan_III : culture_medium), Sudan_III = urea_only

def methodD (culture_medium : Type) := 
  ∃ (urea_only : culture_medium) (Biuret_reagent : culture_medium), Biuret_reagent = urea_only

-- The proof problem statement
theorem identify_urea_decomposing_bacteria (culture_medium : Type) :
  methodA culture_medium :=
sorry

end NUMINAMATH_GPT_identify_urea_decomposing_bacteria_l1360_136076


namespace NUMINAMATH_GPT_integral_abs_x_minus_two_l1360_136067

theorem integral_abs_x_minus_two : ∫ x in (0:ℝ)..4, |x - 2| = 4 := 
by
  sorry

end NUMINAMATH_GPT_integral_abs_x_minus_two_l1360_136067


namespace NUMINAMATH_GPT_IntersectionOfAandB_l1360_136011

def setA : Set ℝ := {x | x < 5}
def setB : Set ℝ := {x | -1 < x}

theorem IntersectionOfAandB : setA ∩ setB = {x | -1 < x ∧ x < 5} :=
sorry

end NUMINAMATH_GPT_IntersectionOfAandB_l1360_136011


namespace NUMINAMATH_GPT_part1_part2_l1360_136025

noncomputable def total_seating_arrangements : ℕ := 840
noncomputable def non_adjacent_4_people_arrangements : ℕ := 24
noncomputable def three_empty_adjacent_arrangements : ℕ := 120

theorem part1 : total_seating_arrangements - non_adjacent_4_people_arrangements = 816 := by
  sorry

theorem part2 : total_seating_arrangements - three_empty_adjacent_arrangements = 720 := by
  sorry

end NUMINAMATH_GPT_part1_part2_l1360_136025


namespace NUMINAMATH_GPT_union_of_sets_l1360_136028

def setA := { x : ℝ | -1 ≤ 2 * x + 1 ∧ 2 * x + 1 ≤ 3 }
def setB := { x : ℝ | (x - 2) / x ≤ 0 }

theorem union_of_sets :
  { x : ℝ | -1 ≤ x ∧ x ≤ 2 } = setA ∪ setB :=
by
  sorry

end NUMINAMATH_GPT_union_of_sets_l1360_136028


namespace NUMINAMATH_GPT_chandler_total_rolls_l1360_136074

-- Definitions based on given conditions
def rolls_sold_grandmother : ℕ := 3
def rolls_sold_uncle : ℕ := 4
def rolls_sold_neighbor : ℕ := 3
def rolls_needed_more : ℕ := 2

-- Total rolls sold so far and needed
def total_rolls_to_sell : ℕ :=
  rolls_sold_grandmother + rolls_sold_uncle + rolls_sold_neighbor + rolls_needed_more

theorem chandler_total_rolls : total_rolls_to_sell = 12 :=
by
  sorry

end NUMINAMATH_GPT_chandler_total_rolls_l1360_136074


namespace NUMINAMATH_GPT_flashes_in_fraction_of_hour_l1360_136037

-- Definitions for the conditions
def flash_interval : ℕ := 6       -- The light flashes every 6 seconds
def hour_in_seconds : ℕ := 3600 -- There are 3600 seconds in an hour
def fraction_of_hour : ℚ := 3/4 -- ¾ of an hour

-- The translated proof problem statement in Lean
theorem flashes_in_fraction_of_hour (interval : ℕ) (sec_in_hour : ℕ) (fraction : ℚ) :
  interval = flash_interval →
  sec_in_hour = hour_in_seconds →
  fraction = fraction_of_hour →
  (fraction * sec_in_hour) / interval = 450 := 
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_flashes_in_fraction_of_hour_l1360_136037


namespace NUMINAMATH_GPT_money_left_correct_l1360_136036

-- Define the initial amount of money John had
def initial_money : ℝ := 10.50

-- Define the amount spent on sweets
def sweets_cost : ℝ := 2.25

-- Define the amount John gave to each friend
def gift_per_friend : ℝ := 2.20

-- Define the total number of friends
def number_of_friends : ℕ := 2

-- Calculate the total gifts given to friends
def total_gifts := gift_per_friend * (number_of_friends : ℝ)

-- Calculate the total amount spent
def total_spent := sweets_cost + total_gifts

-- Define the amount of money left
def money_left := initial_money - total_spent

-- The theorem statement
theorem money_left_correct : money_left = 3.85 := 
by 
  sorry

end NUMINAMATH_GPT_money_left_correct_l1360_136036


namespace NUMINAMATH_GPT_domain_of_function_l1360_136099

theorem domain_of_function :
  {x : ℝ | 2 - x ≥ 0} = {x : ℝ | x ≤ 2} :=
by
  sorry

end NUMINAMATH_GPT_domain_of_function_l1360_136099


namespace NUMINAMATH_GPT_quadratic_inequality_l1360_136009

noncomputable def ax2_plus_bx_c (a b c x : ℝ) : ℝ :=
  a * x^2 + b * x + c

theorem quadratic_inequality (a b c : ℝ) :
  (∀ x : ℝ, |x| ≤ 1 → |ax2_plus_bx_c a b c x| ≤ 1 / 2) →
  ∀ x : ℝ, |x| ≥ 1 → |ax2_plus_bx_c a b c x| ≤ x^2 - 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_l1360_136009

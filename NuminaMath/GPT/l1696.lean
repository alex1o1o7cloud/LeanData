import Mathlib

namespace NUMINAMATH_GPT_percentage_of_Hindu_boys_l1696_169656

theorem percentage_of_Hindu_boys (total_boys : ℕ) (muslim_percentage : ℕ) (sikh_percentage : ℕ)
  (other_community_boys : ℕ) (H : total_boys = 850) (H1 : muslim_percentage = 44) 
  (H2 : sikh_percentage = 10) (H3 : other_community_boys = 153) :
  let muslim_boys := muslim_percentage * total_boys / 100
  let sikh_boys := sikh_percentage * total_boys / 100
  let non_hindu_boys := muslim_boys + sikh_boys + other_community_boys
  let hindu_boys := total_boys - non_hindu_boys
  (hindu_boys * 100 / total_boys : ℚ) = 28 := 
by
  sorry

end NUMINAMATH_GPT_percentage_of_Hindu_boys_l1696_169656


namespace NUMINAMATH_GPT_functional_equality_l1696_169694

theorem functional_equality (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (f y + x ^ 2 + 1) + 2 * x = y + (f (x + 1)) ^ 2) →
  (∀ x : ℝ, f x = x) := 
by
  intro h
  sorry

end NUMINAMATH_GPT_functional_equality_l1696_169694


namespace NUMINAMATH_GPT_closest_point_on_ellipse_l1696_169603

theorem closest_point_on_ellipse : 
  ∃ (x y : ℝ), (7 * x^2 + 4 * y^2 = 28 ∧ 3 * x - 2 * y - 16 = 0) ∧ 
  (∀ (x' y' : ℝ), 7 * x'^2 + 4 * y'^2 = 28 → dist (x, y) (0, 0) ≤ dist (x', y') (0, 0)) :=
sorry

end NUMINAMATH_GPT_closest_point_on_ellipse_l1696_169603


namespace NUMINAMATH_GPT_area_of_picture_l1696_169628

theorem area_of_picture {x y : ℕ} (hx : x > 1) (hy : y > 1) 
  (h : (2 * x + 3) * (y + 2) - x * y = 34) : x * y = 8 := 
by
  sorry

end NUMINAMATH_GPT_area_of_picture_l1696_169628


namespace NUMINAMATH_GPT_parallel_lines_slope_l1696_169626

theorem parallel_lines_slope (m : ℝ) :
  (∀ x y : ℝ, (m + 3) * x + 4 * y + 3 * m - 5 = 0) ∧ (∀ x y : ℝ, 2 * x + (m + 5) * y - 8 = 0) →
  m = -7 :=
by
  intro H
  sorry

end NUMINAMATH_GPT_parallel_lines_slope_l1696_169626


namespace NUMINAMATH_GPT_total_number_of_meetings_proof_l1696_169676

-- Define the conditions in Lean
variable (A B : Type)
variable (starting_time : ℕ)
variable (location_A location_B : A × B)

-- Define speeds
variable (speed_A speed_B : ℕ)

-- Define meeting counts
variable (total_meetings : ℕ)

-- Define A reaches point B 2015 times
variable (A_reaches_B_2015 : Prop)

-- Define that B travels twice as fast as A
axiom speed_ratio : speed_B = 2 * speed_A

-- Define that A reaches point B for the 5th time when B reaches it for the 9th time
axiom meeting_times : A_reaches_B_2015 → (total_meetings = 6044)

-- The Lean statement to prove
theorem total_number_of_meetings_proof : A_reaches_B_2015 → total_meetings = 6044 := by
  sorry

end NUMINAMATH_GPT_total_number_of_meetings_proof_l1696_169676


namespace NUMINAMATH_GPT_necessary_condition_for_line_passes_quadrants_l1696_169651

theorem necessary_condition_for_line_passes_quadrants (m n : ℝ) (h_line : ∀ x : ℝ, x * (m / n) - (1 / n) < 0 ∨ x * (m / n) - (1 / n) > 0) : m * n < 0 :=
by
  sorry

end NUMINAMATH_GPT_necessary_condition_for_line_passes_quadrants_l1696_169651


namespace NUMINAMATH_GPT_part1_monotonicity_when_a_eq_1_part2_range_of_a_l1696_169697

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (x - 1) * (Real.log (x - 2)) - a * (x - 3)

theorem part1_monotonicity_when_a_eq_1 :
  ∀ x, 2 < x → ∀ x1, (2 < x1 → f x 1 ≤ f x1 1) := by
  sorry

theorem part2_range_of_a :
  ∀ a, (∀ x, 3 < x → f x a > 0) → a ≤ 2 := by
  sorry

end NUMINAMATH_GPT_part1_monotonicity_when_a_eq_1_part2_range_of_a_l1696_169697


namespace NUMINAMATH_GPT_sequence_general_term_l1696_169600

theorem sequence_general_term 
  (a : ℕ → ℝ)
  (h₀ : a 1 = 1)
  (h₁ : a 2 = 1 / 3)
  (h₂ : ∀ n : ℕ, 2 ≤ n → a n * a (n - 1) + a n * a (n + 1) = 2 * a (n - 1) * a (n + 1)) :
  ∀ n : ℕ, 1 ≤ n → a n = 1 / (2 * n - 1) := 
by
  sorry

end NUMINAMATH_GPT_sequence_general_term_l1696_169600


namespace NUMINAMATH_GPT_quadratic_root_difference_l1696_169639

theorem quadratic_root_difference (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ + x₂ = 2 ∧ x₁ * x₂ = a ∧ (x₁ - x₂)^2 = 20) → a = -4 := 
by
  sorry

end NUMINAMATH_GPT_quadratic_root_difference_l1696_169639


namespace NUMINAMATH_GPT_f_prime_neg_one_l1696_169633

-- Given conditions and definitions
def f (x : ℝ) (a b c : ℝ) := a * x^4 + b * x^2 + c

def f_prime (x : ℝ) (a b : ℝ) := 4 * a * x^3 + 2 * b * x

-- The theorem we need to prove
theorem f_prime_neg_one (a b c : ℝ) (h : f_prime 1 a b = 2) : f_prime (-1) a b = -2 := by
  sorry

end NUMINAMATH_GPT_f_prime_neg_one_l1696_169633


namespace NUMINAMATH_GPT_problem_real_numbers_l1696_169663

theorem problem_real_numbers (a b : ℝ) (n : ℕ) (h : 2 * a + 3 * b = 12) : 
  ((a / 3) ^ n + (b / 2) ^ n) ≥ 2 := 
sorry

end NUMINAMATH_GPT_problem_real_numbers_l1696_169663


namespace NUMINAMATH_GPT_problem_solution_exists_l1696_169695

theorem problem_solution_exists (x : ℝ) (h : ∃ x, 2 * (3 * 5 - x) - x = -8) : x = 10 :=
sorry

end NUMINAMATH_GPT_problem_solution_exists_l1696_169695


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l1696_169654

theorem isosceles_triangle_perimeter
  (x y : ℝ)
  (h : |x - 3| + (y - 1)^2 = 0)
  (isosceles_triangle : ∃ a b c, (a = x ∧ b = x ∧ c = y) ∨ (a = x ∧ b = y ∧ c = y) ∨ (a = y ∧ b = y ∧ c = x)):
  ∃ perimeter : ℝ, perimeter = 7 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l1696_169654


namespace NUMINAMATH_GPT_poles_on_each_side_l1696_169632

theorem poles_on_each_side (total_poles : ℕ) (sides_equal : ℕ)
  (h1 : total_poles = 104) (h2 : sides_equal = 4) : 
  (total_poles / sides_equal) = 26 :=
by
  sorry

end NUMINAMATH_GPT_poles_on_each_side_l1696_169632


namespace NUMINAMATH_GPT_polynomial_roots_l1696_169611

theorem polynomial_roots (k r : ℝ) (hk_pos : k > 0) 
(h_sum : r + 1 = 2 * k) (h_prod : r * 1 = k) : 
  r = 1 ∧ (∀ x, (x - 1) * (x - 1) = x^2 - 2 * x + 1) := 
by 
  sorry

end NUMINAMATH_GPT_polynomial_roots_l1696_169611


namespace NUMINAMATH_GPT_odd_function_condition_l1696_169621

noncomputable def f (A ω φ : ℝ) (x : ℝ) : ℝ :=
  A * Real.sin (ω * x + φ)

theorem odd_function_condition (A ω : ℝ) (hA : 0 < A) (hω : 0 < ω) (φ : ℝ) :
  (f A ω φ 0 = 0) ↔ (f A ω φ) = fun x => -f A ω φ (-x) := 
by
  sorry

end NUMINAMATH_GPT_odd_function_condition_l1696_169621


namespace NUMINAMATH_GPT_central_angle_measure_l1696_169644

theorem central_angle_measure (α r : ℝ) (h1 : α * r = 2) (h2 : 1/2 * α * r^2 = 2) : α = 1 := 
sorry

end NUMINAMATH_GPT_central_angle_measure_l1696_169644


namespace NUMINAMATH_GPT_cylinder_surface_area_l1696_169610

theorem cylinder_surface_area (a b : ℝ) (h1 : a = 4 * Real.pi) (h2 : b = 8 * Real.pi) :
  (∃ S, S = 32 * Real.pi^2 + 8 * Real.pi ∨ S = 32 * Real.pi^2 + 32 * Real.pi) :=
by
  sorry

end NUMINAMATH_GPT_cylinder_surface_area_l1696_169610


namespace NUMINAMATH_GPT_red_cards_taken_out_l1696_169698

-- Definitions based on the conditions
def total_cards : ℕ := 52
def half_of_total_cards (n : ℕ) := n / 2
def initial_red_cards : ℕ := half_of_total_cards total_cards
def remaining_red_cards : ℕ := 16

-- The statement to prove
theorem red_cards_taken_out : initial_red_cards - remaining_red_cards = 10 := by
  sorry

end NUMINAMATH_GPT_red_cards_taken_out_l1696_169698


namespace NUMINAMATH_GPT_maximum_piles_l1696_169668

theorem maximum_piles (n : ℕ) (h : n = 660) : 
  ∃ m, m = 30 ∧ 
       ∀ (piles : Finset ℕ), (piles.sum id = n) →
       (∀ x ∈ piles, ∀ y ∈ piles, x ≤ y → y < 2 * x) → 
       (piles.card ≤ m) :=
by
  sorry

end NUMINAMATH_GPT_maximum_piles_l1696_169668


namespace NUMINAMATH_GPT_coffee_serving_time_between_1_and_2_is_correct_l1696_169614

theorem coffee_serving_time_between_1_and_2_is_correct
    (x : ℝ)
    (h_pos: 0 < x)
    (h_lt: x < 60) :
    30 + (x / 2) = 360 - (6 * x) → x = 660 / 13 :=
by
  sorry

end NUMINAMATH_GPT_coffee_serving_time_between_1_and_2_is_correct_l1696_169614


namespace NUMINAMATH_GPT_exponential_sequence_term_eq_l1696_169604

-- Definitions for the conditions
variable {α : Type} [CommRing α] (q : α)
def a (n : ℕ) : α := q * (q ^ (n - 1))

-- Statement of the problem
theorem exponential_sequence_term_eq : a q 9 = a q 3 * a q 7 := by
  sorry

end NUMINAMATH_GPT_exponential_sequence_term_eq_l1696_169604


namespace NUMINAMATH_GPT_quadrilateral_diagonals_perpendicular_l1696_169642

def convex_quadrilateral (A B C D : Type) : Prop := sorry -- Assume it’s defined elsewhere 
def tangent_to_all_sides (circle : Type) (A B C D : Type) : Prop := sorry -- Assume it’s properly specified with its conditions elsewhere
def tangent_to_all_extensions (circle : Type) (A B C D : Type) : Prop := sorry -- Same as above

theorem quadrilateral_diagonals_perpendicular
  (A B C D : Type)
  (h_convex : convex_quadrilateral A B C D)
  (incircle excircle : Type)
  (h_incircle : tangent_to_all_sides incircle A B C D)
  (h_excircle : tangent_to_all_extensions excircle A B C D) : 
  (⊥ : Prop) :=  -- statement indicating perpendicularity 
sorry

end NUMINAMATH_GPT_quadrilateral_diagonals_perpendicular_l1696_169642


namespace NUMINAMATH_GPT_problem_statement_l1696_169682

theorem problem_statement : 100 * 29.98 * 2.998 * 1000 = (2998)^2 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1696_169682


namespace NUMINAMATH_GPT_valid_x_y_sum_l1696_169631

-- Setup the initial conditions as variables.
variables (x y : ℕ)

-- Declare the conditions as hypotheses.
theorem valid_x_y_sum (h1 : 0 < x) (h2 : x < 25)
  (h3 : 0 < y) (h4 : y < 25) (h5 : x + y + x * y = 119) :
  x + y = 27 ∨ x + y = 24 ∨ x + y = 21 ∨ x + y = 20 :=
sorry

end NUMINAMATH_GPT_valid_x_y_sum_l1696_169631


namespace NUMINAMATH_GPT_compare_three_and_negfour_l1696_169678

theorem compare_three_and_negfour : 3 > -4 := by
  sorry

end NUMINAMATH_GPT_compare_three_and_negfour_l1696_169678


namespace NUMINAMATH_GPT_cost_of_rusted_side_l1696_169690

-- Define the conditions
def perimeter (s : ℕ) (l : ℕ) : ℕ :=
  2 * s + 2 * l

def long_side (s : ℕ) : ℕ :=
  3 * s

def cost_per_foot : ℕ :=
  5

-- Given these conditions, we prove the cost of replacing one short side.
theorem cost_of_rusted_side (s l : ℕ) (h1 : perimeter s l = 640) (h2 : l = long_side s) : 
  5 * s = 400 :=
by 
  sorry

end NUMINAMATH_GPT_cost_of_rusted_side_l1696_169690


namespace NUMINAMATH_GPT_sum_abc_geq_half_l1696_169677

theorem sum_abc_geq_half (a b c : ℝ) (h_nonneg_a : 0 ≤ a) (h_nonneg_b : 0 ≤ b) (h_nonneg_c : 0 ≤ c) 
(h_abs_sum : |a - b| + |b - c| + |c - a| = 1) : 
a + b + c ≥ 0.5 := 
sorry

end NUMINAMATH_GPT_sum_abc_geq_half_l1696_169677


namespace NUMINAMATH_GPT_avg_tickets_male_l1696_169607

theorem avg_tickets_male (M F : ℕ) (w : ℕ) 
  (h1 : M / F = 1 / 2) 
  (h2 : (M + F) * 66 = M * w + F * 70) 
  : w = 58 := 
sorry

end NUMINAMATH_GPT_avg_tickets_male_l1696_169607


namespace NUMINAMATH_GPT_bob_pennies_l1696_169635

variable (a b : ℕ)

theorem bob_pennies : 
  (b + 2 = 4 * (a - 2)) →
  (b - 3 = 3 * (a + 3)) →
  b = 78 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_bob_pennies_l1696_169635


namespace NUMINAMATH_GPT_principal_amount_l1696_169665

theorem principal_amount (P : ℝ) (CI SI : ℝ) 
  (H1 : CI = P * 0.44) 
  (H2 : SI = P * 0.4) 
  (H3 : CI - SI = 216) : 
  P = 5400 :=
by {
  sorry
}

end NUMINAMATH_GPT_principal_amount_l1696_169665


namespace NUMINAMATH_GPT_parallelogram_height_base_difference_l1696_169650

theorem parallelogram_height_base_difference (A B H : ℝ) (hA : A = 24) (hB : B = 4) (hArea : A = B * H) :
  H - B = 2 := by
  sorry

end NUMINAMATH_GPT_parallelogram_height_base_difference_l1696_169650


namespace NUMINAMATH_GPT_find_initial_men_l1696_169638

noncomputable def initial_men_planned (M : ℕ) : Prop :=
  let initial_days := 10
  let additional_days := 20
  let total_days := initial_days + additional_days
  let men_sent := 25
  let initial_work := M * initial_days
  let remaining_men := M - men_sent
  let remaining_work := remaining_men * total_days
  initial_work = remaining_work 

theorem find_initial_men :
  ∃ M : ℕ, initial_men_planned M ∧ M = 38 :=
by
  have h : initial_men_planned 38 :=
    by
      sorry
  exact ⟨38, h, rfl⟩

end NUMINAMATH_GPT_find_initial_men_l1696_169638


namespace NUMINAMATH_GPT_problem_proof_l1696_169669

theorem problem_proof (A B : ℝ) (h1 : A ≠ B) (h2 : ∀ x : ℝ, (A * (B * x^2 + A)^2 + B) - (B * (A * x^2 + B)^2 + A) = A^2 - B^2) :
  A^2 + B^2 = - (A * B) := 
sorry

end NUMINAMATH_GPT_problem_proof_l1696_169669


namespace NUMINAMATH_GPT_total_wheels_in_storage_l1696_169681

def wheels (n_bicycles n_tricycles n_unicycles n_quadbikes : ℕ) : ℕ :=
  (n_bicycles * 2) + (n_tricycles * 3) + (n_unicycles * 1) + (n_quadbikes * 4)

theorem total_wheels_in_storage :
  let n_bicycles := 24
  let n_tricycles := 14
  let n_unicycles := 10
  let n_quadbikes := 8
  wheels n_bicycles n_tricycles n_unicycles n_quadbikes = 132 :=
by
  let n_bicycles := 24
  let n_tricycles := 14
  let n_unicycles := 10
  let n_quadbikes := 8
  show wheels n_bicycles n_tricycles n_unicycles n_quadbikes = 132
  sorry

end NUMINAMATH_GPT_total_wheels_in_storage_l1696_169681


namespace NUMINAMATH_GPT_alice_average_speed_l1696_169649

def average_speed (distance1 speed1 distance2 speed2 totalDistance totalTime : ℚ) :=
  totalDistance / totalTime

theorem alice_average_speed : 
  let d1 := 45
  let s1 := 15
  let d2 := 15
  let s2 := 45
  let totalDistance := d1 + d2
  let totalTime := (d1 / s1) + (d2 / s2)
  average_speed d1 s1 d2 s2 totalDistance totalTime = 18 :=
by
  sorry

end NUMINAMATH_GPT_alice_average_speed_l1696_169649


namespace NUMINAMATH_GPT_tricycles_count_l1696_169696

theorem tricycles_count (cars bicycles pickup_trucks tricycles : ℕ) (total_tires : ℕ) : 
  cars = 15 →
  bicycles = 3 →
  pickup_trucks = 8 →
  total_tires = 101 →
  4 * cars + 2 * bicycles + 4 * pickup_trucks + 3 * tricycles = total_tires →
  tricycles = 1 :=
by
  sorry

end NUMINAMATH_GPT_tricycles_count_l1696_169696


namespace NUMINAMATH_GPT_contingency_fund_correct_l1696_169662

def annual_donation := 240
def community_pantry_share := (1 / 3 : ℚ)
def local_crisis_fund_share := (1 / 2 : ℚ)
def remaining_share := (1 / 4 : ℚ)

def community_pantry_amount : ℚ := annual_donation * community_pantry_share
def local_crisis_amount : ℚ := annual_donation * local_crisis_fund_share
def remaining_amount : ℚ := annual_donation - community_pantry_amount - local_crisis_amount
def livelihood_amount : ℚ := remaining_amount * remaining_share
def contingency_amount : ℚ := remaining_amount - livelihood_amount

theorem contingency_fund_correct :
  contingency_amount = 30 := by
  -- Proof goes here (to be completed)
  sorry

end NUMINAMATH_GPT_contingency_fund_correct_l1696_169662


namespace NUMINAMATH_GPT_work_done_together_l1696_169661

theorem work_done_together
    (fraction_work_left : ℚ)
    (A_days : ℕ)
    (B_days : ℚ) :
    A_days = 20 →
    fraction_work_left = 2 / 3 →
    4 * (1 / 20 + 1 / B_days) = 1 / 3 →
    B_days = 30 := 
by
  intros hA hfrac heq
  sorry

end NUMINAMATH_GPT_work_done_together_l1696_169661


namespace NUMINAMATH_GPT_minimum_value_of_f_l1696_169641

noncomputable def f (x : ℝ) : ℝ := (1 / 3) * x ^ 3 - x + (1 / 3)

theorem minimum_value_of_f :
  (∃ m : ℝ, ∀ x : ℝ, f x ≤ 1) → (∀ x : ℝ, f 1 = -(1 / 3)) :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_f_l1696_169641


namespace NUMINAMATH_GPT_problem_l1696_169624

theorem problem (q r : ℕ) (hq : 1259 = 23 * q + r) (hq_pos : 0 < q) (hr_pos : 0 < r) :
  q - r ≤ 37 :=
sorry

end NUMINAMATH_GPT_problem_l1696_169624


namespace NUMINAMATH_GPT_solve_for_s_l1696_169623

theorem solve_for_s : ∃ s, (∃ x, 4 * x^2 - 8 * x - 320 = 0) ∧ s = 81 :=
by {
  -- Sorry is used to skip the actual proof.
  sorry
}

end NUMINAMATH_GPT_solve_for_s_l1696_169623


namespace NUMINAMATH_GPT_proof_by_contradiction_example_l1696_169658

theorem proof_by_contradiction_example (a b c : ℝ) (h : a < 3 ∧ b < 3 ∧ c < 3) : a < 1 ∨ b < 1 ∨ c < 1 := 
by
  have h1 : a ≥ 1 ∨ b ≥ 1 ∨ c ≥ 1 := sorry
  sorry

end NUMINAMATH_GPT_proof_by_contradiction_example_l1696_169658


namespace NUMINAMATH_GPT_sum_of_reciprocals_sum_of_square_reciprocals_sum_of_cubic_reciprocals_l1696_169630

variable (p q : ℝ) (x1 x2 : ℝ)

-- Define the condition: Roots of the quadratic equation
def quadratic_equation_condition : Prop :=
  x1^2 + p * x1 + q = 0 ∧ x2^2 + p * x2 + q = 0

-- Define the identities for calculations based on properties of roots
def properties_of_roots : Prop :=
  x1 + x2 = -p ∧ x1 * x2 = q

-- First proof problem
theorem sum_of_reciprocals (h1 : quadratic_equation_condition p q x1 x2) 
                           (h2 : properties_of_roots p q x1 x2) :
  1 / x1 + 1 / x2 = -p / q := 
by sorry

-- Second proof problem
theorem sum_of_square_reciprocals (h1 : quadratic_equation_condition p q x1 x2) 
                                  (h2 : properties_of_roots p q x1 x2) :
  1 / (x1^2) + 1 / (x2^2) = (p^2 - 2*q) / (q^2) := 
by sorry

-- Third proof problem
theorem sum_of_cubic_reciprocals (h1 : quadratic_equation_condition p q x1 x2) 
                                 (h2 : properties_of_roots p q x1 x2) :
  1 / (x1^3) + 1 / (x2^3) = p * (3*q - p^2) / (q^3) := 
by sorry

end NUMINAMATH_GPT_sum_of_reciprocals_sum_of_square_reciprocals_sum_of_cubic_reciprocals_l1696_169630


namespace NUMINAMATH_GPT_negation_is_false_l1696_169605

-- Definitions corresponding to the conditions
def prop (x : ℝ) := x > 0 → x^2 > 0

-- Statement of the proof problem in Lean 4
theorem negation_is_false : ¬(∀ x : ℝ, ¬(x > 0 → x^2 > 0)) = false :=
by {
  sorry
}

end NUMINAMATH_GPT_negation_is_false_l1696_169605


namespace NUMINAMATH_GPT_range_for_m_l1696_169692

def A := { x : ℝ | x^2 - 3 * x - 10 < 0 }
def B (m : ℝ) := { x : ℝ | m + 1 < x ∧ x < 1 - 3 * m }

theorem range_for_m (m : ℝ) (h : ∀ x, x ∈ A ∪ B m ↔ x ∈ B m) : m ≤ -3 := sorry

end NUMINAMATH_GPT_range_for_m_l1696_169692


namespace NUMINAMATH_GPT_green_disks_more_than_blue_l1696_169643

theorem green_disks_more_than_blue 
  (total_disks : ℕ) (blue_ratio yellow_ratio green_ratio red_ratio : ℕ)
  (h1 : total_disks = 132)
  (h2 : blue_ratio = 3)
  (h3 : yellow_ratio = 7)
  (h4 : green_ratio = 8)
  (h5 : red_ratio = 4)
  : 6 * green_ratio - 6 * blue_ratio = 30 :=
by
  sorry

end NUMINAMATH_GPT_green_disks_more_than_blue_l1696_169643


namespace NUMINAMATH_GPT_certain_number_is_50_l1696_169640

theorem certain_number_is_50 (x : ℝ) (h : 0.6 * x = 0.42 * 30 + 17.4) : x = 50 :=
by
  sorry

end NUMINAMATH_GPT_certain_number_is_50_l1696_169640


namespace NUMINAMATH_GPT_conditional_prob_l1696_169608

noncomputable def prob_A := 0.7
noncomputable def prob_AB := 0.4

theorem conditional_prob : prob_AB / prob_A = 4 / 7 :=
by
  sorry

end NUMINAMATH_GPT_conditional_prob_l1696_169608


namespace NUMINAMATH_GPT_maximum_value_product_cube_expression_l1696_169615

theorem maximum_value_product_cube_expression (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 3) :
  (x^3 - x * y^2 + y^3) * (x^3 - x^2 * z + z^3) * (y^3 - y^2 * z + z^3) ≤ 1 :=
sorry

end NUMINAMATH_GPT_maximum_value_product_cube_expression_l1696_169615


namespace NUMINAMATH_GPT_volume_difference_l1696_169684

theorem volume_difference (x1 x2 x3 Vmin Vmax : ℝ)
  (hx1 : 0.5 < x1 ∧ x1 < 1.5)
  (hx2 : 0.5 < x2 ∧ x2 < 1.5)
  (hx3 : 2016.5 < x3 ∧ x3 < 2017.5)
  (rV : 2017 = Nat.floor (x1 * x2 * x3))
  : abs (Vmax - Vmin) = 4035 := 
sorry

end NUMINAMATH_GPT_volume_difference_l1696_169684


namespace NUMINAMATH_GPT_largest_integer_inequality_l1696_169667

theorem largest_integer_inequality (x : ℤ) (h : 10 - 3 * x > 25) : x = -6 :=
sorry

end NUMINAMATH_GPT_largest_integer_inequality_l1696_169667


namespace NUMINAMATH_GPT_company_profit_growth_l1696_169675

theorem company_profit_growth (x : ℝ) (h : 1.6 * (1 + x / 100)^2 = 2.5) : x = 25 :=
sorry

end NUMINAMATH_GPT_company_profit_growth_l1696_169675


namespace NUMINAMATH_GPT_cos_neg_17pi_over_4_l1696_169659

noncomputable def cos_value : ℝ := (Real.pi / 4).cos

theorem cos_neg_17pi_over_4 :
  (Real.cos (-17 * Real.pi / 4)) = cos_value :=
by
  -- Define even property of cosine and angle simplification
  sorry

end NUMINAMATH_GPT_cos_neg_17pi_over_4_l1696_169659


namespace NUMINAMATH_GPT_radian_measure_of_minute_hand_rotation_l1696_169602

theorem radian_measure_of_minute_hand_rotation :
  ∀ (t : ℝ), (t = 10) → (2 * π / 60 * t = -π/3) := by
  sorry

end NUMINAMATH_GPT_radian_measure_of_minute_hand_rotation_l1696_169602


namespace NUMINAMATH_GPT_unique_divisors_form_l1696_169683

theorem unique_divisors_form (n : ℕ) (h₁ : n > 1)
    (h₂ : ∀ d : ℕ, d ∣ n ∧ d > 1 → ∃ a r : ℕ, a > 1 ∧ r > 1 ∧ d = a^r + 1) :
    n = 10 := by
  sorry

end NUMINAMATH_GPT_unique_divisors_form_l1696_169683


namespace NUMINAMATH_GPT_fraction_product_l1696_169689

theorem fraction_product : (2 * (-4)) / (9 * 5) = -8 / 45 :=
  by sorry

end NUMINAMATH_GPT_fraction_product_l1696_169689


namespace NUMINAMATH_GPT_h_inch_approx_l1696_169606

noncomputable def h_cm : ℝ := 14.5 - 2 * 1.7
noncomputable def cm_to_inch (cm : ℝ) : ℝ := cm / 2.54
noncomputable def h_inch : ℝ := cm_to_inch h_cm

theorem h_inch_approx : abs (h_inch - 4.37) < 1e-2 :=
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_h_inch_approx_l1696_169606


namespace NUMINAMATH_GPT_eval_expr1_l1696_169670

theorem eval_expr1 : 
  ( (27 / 8) ^ (-2 / 3) - (49 / 9) ^ 0.5 + (0.008) ^ (-2 / 3) * (2 / 25) ) = 1 / 9 :=
by 
  sorry

end NUMINAMATH_GPT_eval_expr1_l1696_169670


namespace NUMINAMATH_GPT_discount_problem_l1696_169612

theorem discount_problem (n : ℕ) : 
  (∀ x : ℝ, 0 < x → (1 - n / 100 : ℝ) * x < min (0.72 * x) (min (0.6724 * x) (0.681472 * x))) ↔ n ≥ 33 :=
by
  sorry

end NUMINAMATH_GPT_discount_problem_l1696_169612


namespace NUMINAMATH_GPT_sin_square_pi_over_4_l1696_169664

theorem sin_square_pi_over_4 (β : ℝ) (h : Real.sin (2 * β) = 2 / 3) : 
  Real.sin (β + π/4) ^ 2 = 5 / 6 :=
by
  sorry

end NUMINAMATH_GPT_sin_square_pi_over_4_l1696_169664


namespace NUMINAMATH_GPT_relationship_x_y_l1696_169613

variable (a b x y : ℝ)

theorem relationship_x_y (h1: 0 < a) (h2: a < b)
  (hx : x = (Real.sqrt (a + b) - Real.sqrt b))
  (hy : y = (Real.sqrt b - Real.sqrt (b - a))) :
  x < y :=
  sorry

end NUMINAMATH_GPT_relationship_x_y_l1696_169613


namespace NUMINAMATH_GPT_sum_of_series_eq_5_over_16_l1696_169686

theorem sum_of_series_eq_5_over_16 :
  ∑' n : ℕ, (n + 1 : ℝ) / (5 : ℝ)^(n + 1) = 5 / 16 := by
  sorry

end NUMINAMATH_GPT_sum_of_series_eq_5_over_16_l1696_169686


namespace NUMINAMATH_GPT_find_smallest_x_l1696_169699

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m * m = n

theorem find_smallest_x (x: ℕ) (h1: 2 * x = 144) (h2: 3 * x = 216) : x = 72 :=
by
  sorry

end NUMINAMATH_GPT_find_smallest_x_l1696_169699


namespace NUMINAMATH_GPT_smallest_n_value_l1696_169620

theorem smallest_n_value :
  ∃ n, (∀ (sheets : Fin 2000 → Fin 4 → Fin 4),
        (∀ (n : Nat) (h : n ≤ 2000) (a b c d : Fin n) (h' : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d),
          ∃ (i j k : Fin 5), sheets a i = sheets b i ∧ sheets a j = sheets b j ∧ sheets a k = sheets b k → ¬ sheets a i = sheets c i ∧ ¬ sheets b j = sheets c j ∧ ¬ sheets a k = sheets c k)) ↔ n = 25 :=
sorry

end NUMINAMATH_GPT_smallest_n_value_l1696_169620


namespace NUMINAMATH_GPT_max_z_value_l1696_169655

theorem max_z_value (x y z : ℝ) (h : x + y + z = 3) (h' : x * y + y * z + z * x = 2) : z ≤ 5 / 3 :=
  sorry


end NUMINAMATH_GPT_max_z_value_l1696_169655


namespace NUMINAMATH_GPT_sellable_fruit_l1696_169660

theorem sellable_fruit :
  let total_oranges := 30 * 300
  let total_damaged_oranges := total_oranges * 10 / 100
  let sellable_oranges := total_oranges - total_damaged_oranges

  let total_nectarines := 45 * 80
  let nectarines_taken := 5 * 20
  let sellable_nectarines := total_nectarines - nectarines_taken

  let total_apples := 20 * 120
  let bad_apples := 50
  let sellable_apples := total_apples - bad_apples

  sellable_oranges + sellable_nectarines + sellable_apples = 13950 :=
by
  sorry

end NUMINAMATH_GPT_sellable_fruit_l1696_169660


namespace NUMINAMATH_GPT_correct_factorization_l1696_169634

theorem correct_factorization (a b : ℝ) : a^2 - 4 * a * b + 4 * b^2 = (a - 2 * b)^2 :=
by sorry

end NUMINAMATH_GPT_correct_factorization_l1696_169634


namespace NUMINAMATH_GPT_greenwood_school_l1696_169625

theorem greenwood_school (f s : ℕ) (h : (3 / 4) * f = (1 / 3) * s) : s = 3 * f :=
by
  sorry

end NUMINAMATH_GPT_greenwood_school_l1696_169625


namespace NUMINAMATH_GPT_rectangle_area_l1696_169601

def radius : ℝ := 10
def width : ℝ := 2 * radius
def length : ℝ := 3 * width
def area_of_rectangle : ℝ := length * width

theorem rectangle_area : area_of_rectangle = 1200 :=
  by sorry

end NUMINAMATH_GPT_rectangle_area_l1696_169601


namespace NUMINAMATH_GPT_find_original_production_planned_l1696_169636

-- Definition of the problem
variables (x : ℕ)
noncomputable def original_production_planned (x : ℕ) :=
  (6000 / (x + 500)) = (4500 / x)

-- The theorem to prove the original number planned is 1500
theorem find_original_production_planned (x : ℕ) (h : original_production_planned x) : x = 1500 :=
sorry

end NUMINAMATH_GPT_find_original_production_planned_l1696_169636


namespace NUMINAMATH_GPT_diana_shops_for_newborns_l1696_169619

theorem diana_shops_for_newborns (total_children : ℕ) (num_toddlers : ℕ) (teenager_ratio : ℕ) (num_teens : ℕ) (num_newborns : ℕ)
    (h1 : total_children = 40) (h2 : num_toddlers = 6) (h3 : teenager_ratio = 5) (h4 : num_teens = teenager_ratio * num_toddlers) 
    (h5 : num_newborns = total_children - num_teens - num_toddlers) : 
    num_newborns = 4 := sorry

end NUMINAMATH_GPT_diana_shops_for_newborns_l1696_169619


namespace NUMINAMATH_GPT_max_sum_of_arithmetic_sequence_l1696_169691

theorem max_sum_of_arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (S_seq : ∀ n, S n = (n + 1) * a 0 + (n * (n + 1) / 2) * (a 1 - a 0)) 
  (S16_pos : S 16 > 0) (S17_neg : S 17 < 0) : 
  ∃ m, ∀ n, S n ≤ S m ∧ m = 8 := 
sorry

end NUMINAMATH_GPT_max_sum_of_arithmetic_sequence_l1696_169691


namespace NUMINAMATH_GPT_stickers_started_with_l1696_169679

-- Definitions for the conditions
def stickers_given (Emily : ℕ) : Prop := Emily = 7
def stickers_ended_with (Willie_end : ℕ) : Prop := Willie_end = 43

-- The main proof statement
theorem stickers_started_with (Willie_start : ℕ) :
  stickers_given 7 →
  stickers_ended_with 43 →
  Willie_start = 43 - 7 :=
by
  intros h₁ h₂
  sorry

end NUMINAMATH_GPT_stickers_started_with_l1696_169679


namespace NUMINAMATH_GPT_students_recess_time_l1696_169685

def initial_recess : ℕ := 20

def extra_minutes_as (as : ℕ) : ℕ := 4 * as
def extra_minutes_bs (bs : ℕ) : ℕ := 3 * bs
def extra_minutes_cs (cs : ℕ) : ℕ := 2 * cs
def extra_minutes_ds (ds : ℕ) : ℕ := ds
def extra_minutes_es (es : ℕ) : ℤ := - es
def extra_minutes_fs (fs : ℕ) : ℤ := -2 * fs

def total_recess (as bs cs ds es fs : ℕ) : ℤ :=
  initial_recess + 
  (extra_minutes_as as + extra_minutes_bs bs +
  extra_minutes_cs cs + extra_minutes_ds ds +
  extra_minutes_es es + extra_minutes_fs fs : ℤ)

theorem students_recess_time :
  total_recess 10 12 14 5 3 2 = 122 := by sorry

end NUMINAMATH_GPT_students_recess_time_l1696_169685


namespace NUMINAMATH_GPT_find_grade_C_boxes_l1696_169687

theorem find_grade_C_boxes (m n t : ℕ) (h : 2 * t = m + n) (total_boxes : ℕ) (h_total : total_boxes = 420) : t = 140 :=
by
  sorry

end NUMINAMATH_GPT_find_grade_C_boxes_l1696_169687


namespace NUMINAMATH_GPT_relationship_x_a_b_l1696_169645

theorem relationship_x_a_b (x a b : ℝ) (h1 : x < b) (h2 : b < a) (h3 : a < 0) : 
  x^2 > a * b ∧ a * b > a^2 :=
by
  sorry

end NUMINAMATH_GPT_relationship_x_a_b_l1696_169645


namespace NUMINAMATH_GPT_max_triangles_convex_polygon_l1696_169609

theorem max_triangles_convex_polygon (vertices : ℕ) (interior_points : ℕ) (total_points : ℕ) : 
  vertices = 13 ∧ interior_points = 200 ∧ total_points = 213 ∧ (∀ (x y z : ℕ), (x < total_points ∧ y < total_points ∧ z < total_points) → x ≠ y ∧ y ≠ z ∧ x ≠ z) →
  (∃ triangles : ℕ, triangles = 411) :=
by
  sorry

end NUMINAMATH_GPT_max_triangles_convex_polygon_l1696_169609


namespace NUMINAMATH_GPT_calculate_120ab_l1696_169674

variable (a b : ℚ)

theorem calculate_120ab (h1 : 10 * a = 20) (h2 : 6 * b = 20) : 120 * (a * b) = 800 := by
  sorry

end NUMINAMATH_GPT_calculate_120ab_l1696_169674


namespace NUMINAMATH_GPT_g_inv_g_inv_14_l1696_169653

def g (x : ℝ) : ℝ := 5 * x - 3

noncomputable def g_inv (y : ℝ) : ℝ := (y + 3) / 5

theorem g_inv_g_inv_14 : g_inv (g_inv 14) = 32 / 25 :=
by
  sorry

end NUMINAMATH_GPT_g_inv_g_inv_14_l1696_169653


namespace NUMINAMATH_GPT_probability_correct_l1696_169629

namespace ProbabilitySongs

/-- Define the total number of ways to choose 2 out of 4 songs -/ 
def total_ways : ℕ := Nat.choose 4 2

/-- Define the number of ways to choose 2 songs such that neither A nor B is chosen (only C and D can be chosen) -/
def ways_without_AB : ℕ := Nat.choose 2 2

/-- The probability of playing at least one of A and B is calculated via the complementary rule -/
def probability_at_least_one_AB_played : ℚ := 1 - (ways_without_AB / total_ways)

theorem probability_correct : probability_at_least_one_AB_played = 5 / 6 := sorry
end ProbabilitySongs

end NUMINAMATH_GPT_probability_correct_l1696_169629


namespace NUMINAMATH_GPT_certain_event_positive_integers_sum_l1696_169627

theorem certain_event_positive_integers_sum :
  ∀ (a b : ℕ), a > 0 → b > 0 → a + b > 1 :=
by
  intros a b ha hb
  sorry

end NUMINAMATH_GPT_certain_event_positive_integers_sum_l1696_169627


namespace NUMINAMATH_GPT_geometric_sequence_fifth_term_l1696_169666

theorem geometric_sequence_fifth_term (x y : ℝ) (r : ℝ) 
  (h1 : x + y ≠ 0) (h2 : x - y ≠ 0) (h3 : x ≠ 0) (h4 : y ≠ 0)
  (h_ratio_1 : (x - y) / (x + y) = r)
  (h_ratio_2 : (x^2 * y) / (x - y) = r)
  (h_ratio_3 : (x * y^2) / (x^2 * y) = r) :
  (x * y^2 * ((y / x) * r)) = y^3 := 
by 
  sorry

end NUMINAMATH_GPT_geometric_sequence_fifth_term_l1696_169666


namespace NUMINAMATH_GPT_stream_speed_fraction_l1696_169617

theorem stream_speed_fraction (B S : ℝ) (h1 : B = 3 * S) 
  (h2 : (1 / (B - S)) = 2 * (1 / (B + S))) : (S / B) = 1 / 3 :=
sorry

end NUMINAMATH_GPT_stream_speed_fraction_l1696_169617


namespace NUMINAMATH_GPT_sheila_attends_picnic_l1696_169637

theorem sheila_attends_picnic :
  let probRain := 0.30
  let probSunny := 0.50
  let probCloudy := 0.20
  let probAttendIfRain := 0.15
  let probAttendIfSunny := 0.85
  let probAttendIfCloudy := 0.40
  (probRain * probAttendIfRain + probSunny * probAttendIfSunny + probCloudy * probAttendIfCloudy) = 0.55 :=
by sorry

end NUMINAMATH_GPT_sheila_attends_picnic_l1696_169637


namespace NUMINAMATH_GPT_steves_earning_l1696_169657

variable (pounds_picked : ℕ → ℕ) -- pounds picked on day i: 0=Monday, 1=Tuesday, 2=Wednesday, 3=Thursday

def payment_per_pound : ℕ := 2

def total_money_made : ℕ :=
  (pounds_picked 0 * payment_per_pound) + 
  (pounds_picked 1 * payment_per_pound) + 
  (pounds_picked 2 * payment_per_pound) + 
  (pounds_picked 3 * payment_per_pound)

theorem steves_earning 
  (h0 : pounds_picked 0 = 8)
  (h1 : pounds_picked 1 = 3 * pounds_picked 0)
  (h2 : pounds_picked 2 = 0)
  (h3 : pounds_picked 3 = 18) : 
  total_money_made pounds_picked = 100 := by
  sorry

end NUMINAMATH_GPT_steves_earning_l1696_169657


namespace NUMINAMATH_GPT_part1_part2_l1696_169671

-- Conditions and the equation of the circle
def circleCenterLine (a : ℝ) : Prop := ∃ y, y = a + 2
def circleRadius : ℝ := 2
def pointOnCircle (A : ℝ × ℝ) (a : ℝ) : Prop := (A.1 - a)^2 + (A.2 - (a + 2))^2 = circleRadius^2
def tangentToYAxis (a : ℝ) : Prop := abs a = circleRadius

-- Problem 1: Proving the equation of the circle C
def circleEq (x y a : ℝ) : Prop := (x - a)^2 + (y - (a + 2))^2 = circleRadius^2

theorem part1 (a : ℝ) (h : abs a = circleRadius) (h1 : pointOnCircle (2, 2) a) 
    (h2 : circleCenterLine a) : circleEq 2 0 2 := 
sorry

-- Conditions and the properties for Problem 2
def distSquared (P Q : ℝ × ℝ) : ℝ := (P.1 - Q.1)^2 + (P.2 - Q.2)^2
def QCondition (Q : ℝ × ℝ) : Prop := 
  distSquared Q (1, 3) - distSquared Q (1, 1) = 32
def onCircle (Q : ℝ × ℝ) (a : ℝ) : Prop := (Q.1 - a)^2 + (Q.2 - (a + 2))^2 = circleRadius^2

-- Problem 2: Proving the range of the abscissa a
theorem part2 (Q : ℝ × ℝ) (a : ℝ) 
    (hQ : QCondition Q) (hCircle : onCircle Q a) : 
    -3 ≤ a ∧ a ≤ 1 := 
sorry

end NUMINAMATH_GPT_part1_part2_l1696_169671


namespace NUMINAMATH_GPT_remainder_is_3_l1696_169648

theorem remainder_is_3 (x y r : ℕ) (h1 : x = 7 * y + r) (h2 : 2 * x = 18 * y + 2) (h3 : 11 * y - x = 1)
  (hrange : 0 ≤ r ∧ r < 7) : r = 3 := 
sorry

end NUMINAMATH_GPT_remainder_is_3_l1696_169648


namespace NUMINAMATH_GPT_school_bought_50_cartons_of_markers_l1696_169618

theorem school_bought_50_cartons_of_markers
  (n_puzzles : ℕ := 200)  -- the remaining amount after buying pencils
  (cost_per_carton_marker : ℕ := 4)  -- the cost per carton of markers
  :
  (n_puzzles / cost_per_carton_marker = 50) := -- the theorem to prove
by
  -- Provide skeleton proof strategy here
  sorry  -- details of the proof

end NUMINAMATH_GPT_school_bought_50_cartons_of_markers_l1696_169618


namespace NUMINAMATH_GPT_sum_of_first_six_terms_l1696_169647

variable {a_n : ℕ → ℕ}
variable {d : ℕ}

def is_arithmetic_sequence (a_n : ℕ → ℕ) (d : ℕ) : Prop :=
  ∀ n : ℕ, a_n (n + 1) = a_n n + d 

theorem sum_of_first_six_terms (a_3 a_4 : ℕ) (h : a_3 + a_4 = 30) :
  ∃ a_n d, is_arithmetic_sequence a_n d ∧ 
  a_n 3 = a_3 ∧ a_n 4 = a_4 ∧ 
  (3 * (a_n 1 + (a_n 1 + 5 * d))) = 90 := 
sorry

end NUMINAMATH_GPT_sum_of_first_six_terms_l1696_169647


namespace NUMINAMATH_GPT_sin_cos_identity_l1696_169646

theorem sin_cos_identity (x : ℝ) (h : Real.sin x = 4 * Real.cos x) : Real.sin x * Real.cos x = 4 / 17 := 
by
  sorry

end NUMINAMATH_GPT_sin_cos_identity_l1696_169646


namespace NUMINAMATH_GPT_part1_intersection_part2_sufficient_not_necessary_l1696_169673

open Set

-- Definition of sets A and B
def set_A (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ a + 1}
def set_B : Set ℝ := {x | x ≤ -2 ∨ x ≥ 3}

-- Part (1)
theorem part1_intersection (a : ℝ) (h : a = -2) : set_A a ∩ set_B = {x | -3 ≤ x ∧ x ≤ -2} := by
  sorry

-- Part (2)
theorem part2_sufficient_not_necessary (p q : Prop) (hp : ∀ x, set_A a x → set_B x) (h_suff : p → q) (h_not_necess : ¬(q → p)) : set_A a ⊆ set_B → a ∈ Iic (-3) ∪ Ici 4 := by
  sorry

end NUMINAMATH_GPT_part1_intersection_part2_sufficient_not_necessary_l1696_169673


namespace NUMINAMATH_GPT_paul_reading_novel_l1696_169622

theorem paul_reading_novel (x : ℕ) 
  (h1 : x - ((1 / 6) * x + 10) - ((1 / 5) * (x - ((1 / 6) * x + 10)) + 14) - ((1 / 4) * ((x - ((1 / 6) * x + 10) - ((1 / 5) * (x - ((1 / 6) * x + 10)) + 14)) + 16)) = 48) : 
  x = 161 :=
by sorry

end NUMINAMATH_GPT_paul_reading_novel_l1696_169622


namespace NUMINAMATH_GPT_equal_real_roots_of_quadratic_l1696_169688

theorem equal_real_roots_of_quadratic (k : ℝ) :
  (∃ x : ℝ, x^2 + k*x + 4 = 0 ∧ (x-4)*(x-4) = 0) ↔ k = 4 ∨ k = -4 :=
by
  sorry

end NUMINAMATH_GPT_equal_real_roots_of_quadratic_l1696_169688


namespace NUMINAMATH_GPT_combined_weight_is_18442_l1696_169672

noncomputable def combined_weight_proof : ℝ :=
  let elephant_weight_tons := 3
  let donkey_weight_percentage := 0.1
  let giraffe_weight_tons := 1.5
  let hippopotamus_weight_kg := 4000
  let elephant_food_oz := 16
  let donkey_food_lbs := 5
  let giraffe_food_kg := 3
  let hippopotamus_food_g := 5000

  let ton_to_pounds := 2000
  let kg_to_pounds := 2.20462
  let oz_to_pounds := 1 / 16
  let g_to_pounds := 0.00220462

  let elephant_weight_pounds := elephant_weight_tons * ton_to_pounds
  let donkey_weight_pounds := (1 - donkey_weight_percentage) * elephant_weight_pounds
  let giraffe_weight_pounds := giraffe_weight_tons * ton_to_pounds
  let hippopotamus_weight_pounds := hippopotamus_weight_kg * kg_to_pounds

  let elephant_food_pounds := elephant_food_oz * oz_to_pounds
  let giraffe_food_pounds := giraffe_food_kg * kg_to_pounds
  let hippopotamus_food_pounds := hippopotamus_food_g * g_to_pounds

  elephant_weight_pounds + donkey_weight_pounds + giraffe_weight_pounds + hippopotamus_weight_pounds +
  elephant_food_pounds + donkey_food_lbs + giraffe_food_pounds + hippopotamus_food_pounds

theorem combined_weight_is_18442 : combined_weight_proof = 18442 := by
  sorry

end NUMINAMATH_GPT_combined_weight_is_18442_l1696_169672


namespace NUMINAMATH_GPT_bd_ad_ratio_l1696_169693

noncomputable def mass_point_geometry_bd_ad : ℚ := 
  let AT_OVER_ET := 5
  let DT_OVER_CT := 2
  let mass_A := 1
  let mass_D := 3 * mass_A
  let mass_B := mass_A + mass_D
  mass_B / mass_D

theorem bd_ad_ratio (h1 : AT/ET = 5) (h2 : DT/CT = 2) : BD/AD = 4 / 3 :=
by
  have mass_A := 1
  have mass_D := 3
  have mass_B := 4
  have h := mass_B / mass_D
  sorry

end NUMINAMATH_GPT_bd_ad_ratio_l1696_169693


namespace NUMINAMATH_GPT_inverse_fourier_transform_l1696_169680

noncomputable def F (p : ℝ) : ℂ :=
if 0 < p ∧ p < 1 then 1 else 0

noncomputable def f (x : ℝ) : ℂ :=
(1 / Real.sqrt (2 * Real.pi)) * ((1 - Complex.exp (-Complex.I * x)) / (Complex.I * x))

theorem inverse_fourier_transform :
  ∀ x, (f x) = (1 / (Real.sqrt (2 * Real.pi))) * ((1 - Complex.exp (-Complex.I * x)) / (Complex.I * x)) := by
  intros
  sorry

end NUMINAMATH_GPT_inverse_fourier_transform_l1696_169680


namespace NUMINAMATH_GPT_solution_unique_for_alpha_neg_one_l1696_169616

noncomputable def alpha : ℝ := sorry

axiom alpha_nonzero : alpha ≠ 0

def functional_eqn (f : ℝ → ℝ) (x y : ℝ) : Prop :=
  f (f (x + y)) = f (x + y) + f (x) * f (y) + alpha * x * y

theorem solution_unique_for_alpha_neg_one (f : ℝ → ℝ) :
  (alpha = -1 → (∀ x : ℝ, f x = x)) ∧ (alpha ≠ -1 → ¬ ∃ f : ℝ → ℝ, ∀ x y : ℝ, functional_eqn f x y) :=
sorry

end NUMINAMATH_GPT_solution_unique_for_alpha_neg_one_l1696_169616


namespace NUMINAMATH_GPT_guo_can_pay_exact_amount_l1696_169652

-- Define the denominations and total amount Guo has
def note_denominations := [1, 10, 20, 50]
def total_amount := 20000
def cost_computer := 10000

-- The main theorem stating that Guo can pay exactly 10000 yuan
theorem guo_can_pay_exact_amount : ∃ bills : List ℕ, ∀ (b : ℕ), b ∈ bills → b ∈ note_denominations ∧
  bills.sum = cost_computer :=
sorry

end NUMINAMATH_GPT_guo_can_pay_exact_amount_l1696_169652

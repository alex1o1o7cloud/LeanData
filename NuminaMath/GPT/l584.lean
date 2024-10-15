import Mathlib

namespace NUMINAMATH_GPT_total_boys_went_down_slide_l584_58434

-- Definitions according to the conditions given
def boys_went_down_slide1 : ℕ := 22
def boys_went_down_slide2 : ℕ := 13

-- The statement to be proved
theorem total_boys_went_down_slide : boys_went_down_slide1 + boys_went_down_slide2 = 35 := 
by 
  sorry

end NUMINAMATH_GPT_total_boys_went_down_slide_l584_58434


namespace NUMINAMATH_GPT_force_exerted_by_pulley_on_axis_l584_58471

-- Define the basic parameters given in the problem
def m1 : ℕ := 3 -- mass 1 in kg
def m2 : ℕ := 6 -- mass 2 in kg
def g : ℕ := 10 -- acceleration due to gravity in m/s^2

-- From the problem, we know that:
def F1 : ℕ := m1 * g -- gravitational force on mass 1
def F2 : ℕ := m2 * g -- gravitational force on mass 2

-- To find the tension, setup the equations
def a := (F2 - F1) / (m1 + m2) -- solving for acceleration between the masses

def T := (m1 * a) + F1 -- solving for the tension in the rope considering mass 1

-- Define the proof statement to find the force exerted by the pulley on its axis
theorem force_exerted_by_pulley_on_axis : 2 * T = 80 :=
by
  -- Annotations or calculations can go here
  sorry

end NUMINAMATH_GPT_force_exerted_by_pulley_on_axis_l584_58471


namespace NUMINAMATH_GPT_max_even_integers_for_odd_product_l584_58487

theorem max_even_integers_for_odd_product (a b c d e f g : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h5 : 0 < e) (h6 : 0 < f) (h7 : 0 < g) 
  (h_prod_odd : a * b * c * d * e * f * g % 2 = 1) : a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1 ∧ d % 2 = 1 ∧ e % 2 = 1 ∧ f % 2 = 1 ∧ g % 2 = 1 :=
sorry

end NUMINAMATH_GPT_max_even_integers_for_odd_product_l584_58487


namespace NUMINAMATH_GPT_find_a_10_l584_58414

-- Definitions and conditions from the problem
def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

def S (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  (n * (a 1 + a n)) / 2

variable (a : ℕ → ℕ)

-- Conditions given
axiom a_3 : a 3 = 3
axiom S_3 : S a 3 = 6
axiom arithmetic_seq : is_arithmetic_sequence a

-- Proof problem statement
theorem find_a_10 : a 10 = 10 := 
sorry

end NUMINAMATH_GPT_find_a_10_l584_58414


namespace NUMINAMATH_GPT_unique_triple_solution_l584_58416

theorem unique_triple_solution {x y z : ℤ} (hx : x > 1) (hy : y > 1) (hz : z > 1)
  (H1 : x ∣ y * z - 1) (H2 : y ∣ z * x - 1) (H3 : z ∣ x * y - 1) :
  (x, y, z) = (5, 3, 2) :=
sorry

end NUMINAMATH_GPT_unique_triple_solution_l584_58416


namespace NUMINAMATH_GPT_tripod_height_l584_58485

-- Define the conditions of the problem
structure Tripod where
  leg_length : ℝ
  angle_equal : Bool
  top_height : ℝ
  broken_length : ℝ

def m : ℕ := 27
def n : ℕ := 10

noncomputable def h : ℝ := m / Real.sqrt n

theorem tripod_height :
  ∀ (t : Tripod),
  t.leg_length = 6 →
  t.angle_equal = true →
  t.top_height = 3 →
  t.broken_length = 2 →
  (h = m / Real.sqrt n) →
  (⌊m + Real.sqrt n⌋ = 30) :=
by
  intros
  sorry

end NUMINAMATH_GPT_tripod_height_l584_58485


namespace NUMINAMATH_GPT_three_subsets_equal_sum_l584_58479

theorem three_subsets_equal_sum (n : ℕ) (h1 : n ≡ 0 [MOD 3] ∨ n ≡ 2 [MOD 3]) (h2 : 5 ≤ n) :
  ∃ (A B C : Finset ℕ), A ∪ B ∪ C = Finset.range (n + 1) ∧
                        A ∩ B = ∅ ∧ B ∩ C = ∅ ∧ C ∩ A = ∅ ∧
                        A.sum id = B.sum id ∧ B.sum id = C.sum id ∧ C.sum id = A.sum id :=
sorry

end NUMINAMATH_GPT_three_subsets_equal_sum_l584_58479


namespace NUMINAMATH_GPT_minimize_triangle_expression_l584_58433

theorem minimize_triangle_expression :
  ∃ (a b c : ℤ), a < b ∧ b < c ∧ a + b + c = 30 ∧
  ∀ (x y z : ℤ), x < y ∧ y < z ∧ x + y + z = 30 → (z^2 + 18*x + 18*y - 446) ≥ 17 ∧ 
  ∃ (p q r : ℤ), p < q ∧ q < r ∧ p + q + r = 30 ∧ (r^2 + 18*p + 18*q - 446 = 17) := 
sorry

end NUMINAMATH_GPT_minimize_triangle_expression_l584_58433


namespace NUMINAMATH_GPT_find_p_for_quadratic_l584_58420

theorem find_p_for_quadratic (p : ℝ) (h : p ≠ 0) 
  (h_eq : ∀ x : ℝ, p * x^2 - 10 * x + 2 = 0 → x = 5 / p) : p = 12.5 :=
sorry

end NUMINAMATH_GPT_find_p_for_quadratic_l584_58420


namespace NUMINAMATH_GPT_discriminant_of_quadratic_equation_l584_58469

theorem discriminant_of_quadratic_equation :
  let a := 5
  let b := -9
  let c := 4
  (b^2 - 4 * a * c = 1) :=
by {
  sorry
}

end NUMINAMATH_GPT_discriminant_of_quadratic_equation_l584_58469


namespace NUMINAMATH_GPT_gcd_lcm_product_180_l584_58406

theorem gcd_lcm_product_180 (a b : ℕ) (g l : ℕ) (ha : a > 0) (hb : b > 0) (hg : g > 0) (hl : l > 0) 
  (h₁ : g = gcd a b) (h₂ : l = lcm a b) (h₃ : g * l = 180):
  ∃(n : ℕ), n = 8 :=
by
  sorry

end NUMINAMATH_GPT_gcd_lcm_product_180_l584_58406


namespace NUMINAMATH_GPT_no_a_b_not_divide_bn_minus_n_l584_58466

theorem no_a_b_not_divide_bn_minus_n :
  ∀ (a b : ℕ), 0 < a → 0 < b → ∃ (n : ℕ), 0 < n ∧ a ∣ (b^n - n) :=
by
  sorry

end NUMINAMATH_GPT_no_a_b_not_divide_bn_minus_n_l584_58466


namespace NUMINAMATH_GPT_find_angle_APB_l584_58412

-- Definitions based on conditions
def r1 := 2 -- Radius of semicircle SAR
def r2 := 3 -- Radius of semicircle RBT

def angle_AO1S := 70
def angle_BO2T := 40

def angle_AO1R := 180 - angle_AO1S
def angle_BO2R := 180 - angle_BO2T

def angle_PA := 90
def angle_PB := 90

-- Statement of the theorem
theorem find_angle_APB : angle_PA + angle_AO1R + angle_BO2R + angle_PB + 110 = 540 :=
by
  -- Unused in proof: added only to state theorem 
  have _ := angle_PA
  have _ := angle_AO1R
  have _ := angle_BO2R
  have _ := angle_PB
  have _ := 110
  sorry

end NUMINAMATH_GPT_find_angle_APB_l584_58412


namespace NUMINAMATH_GPT_division_of_difference_squared_l584_58426

theorem division_of_difference_squared :
  ((2222 - 2121)^2) / 196 = 52 := 
sorry

end NUMINAMATH_GPT_division_of_difference_squared_l584_58426


namespace NUMINAMATH_GPT_max_n_no_constant_term_l584_58439

theorem max_n_no_constant_term (n : ℕ) (h : n < 10 ∧ n ≠ 3 ∧ n ≠ 6 ∧ n ≠ 9 ∧ n ≠ 2 ∧ n ≠ 5 ∧ n ≠ 8): n ≤ 7 :=
by {
  sorry
}

end NUMINAMATH_GPT_max_n_no_constant_term_l584_58439


namespace NUMINAMATH_GPT_boys_trees_l584_58481

theorem boys_trees (avg_per_person trees_per_girl trees_per_boy : ℕ) :
  avg_per_person = 6 →
  trees_per_girl = 15 →
  (1 / trees_per_boy + 1 / trees_per_girl = 1 / avg_per_person) →
  trees_per_boy = 10 :=
by
  intros h_avg h_girl h_eq
  -- We will provide the proof here eventually
  sorry

end NUMINAMATH_GPT_boys_trees_l584_58481


namespace NUMINAMATH_GPT_total_interest_percentage_l584_58436

theorem total_interest_percentage (inv_total : ℝ) (rate1 rate2 : ℝ) (inv2 : ℝ)
  (h_inv_total : inv_total = 100000)
  (h_rate1 : rate1 = 0.09)
  (h_rate2 : rate2 = 0.11)
  (h_inv2 : inv2 = 24999.999999999996) :
  (rate1 * (inv_total - inv2) + rate2 * inv2) / inv_total * 100 = 9.5 := 
sorry

end NUMINAMATH_GPT_total_interest_percentage_l584_58436


namespace NUMINAMATH_GPT_four_ab_eq_four_l584_58490

theorem four_ab_eq_four {a b : ℝ} (h : a * b = 1) : 4 * a * b = 4 :=
by
  sorry

end NUMINAMATH_GPT_four_ab_eq_four_l584_58490


namespace NUMINAMATH_GPT_find_kn_l584_58413

section
variables (k n : ℝ)

def system_infinite_solutions (k n : ℝ) :=
  ∃ (y : ℝ → ℝ) (x : ℝ → ℝ),
  (∀ y, k * y + x y + n = 0) ∧
  (∀ y, |y - 2| + |y + 1| + |1 - y| + |y + 2| + x y = 0)

theorem find_kn :
  { (k, n) | system_infinite_solutions k n } = {(4, 0), (-4, 0), (2, 4), (-2, 4), (0, 6)} :=
sorry
end

end NUMINAMATH_GPT_find_kn_l584_58413


namespace NUMINAMATH_GPT_company_employee_count_l584_58438

/-- 
 Given the employees are divided into three age groups: A, B, and C, with a ratio of 5:4:1,
 a stratified sampling method is used to draw a sample of size 20 from the population,
 and the probability of selecting both person A and person B from group C is 1/45.
 Prove the total number of employees in the company is 100.
-/
theorem company_employee_count :
  ∃ (total_employees : ℕ),
    (∃ (ratio_A : ℕ) (ratio_B : ℕ) (ratio_C : ℕ),
      ratio_A = 5 ∧ 
      ratio_B = 4 ∧ 
      ratio_C = 1 ∧
      ∃ (sample_size : ℕ), 
        sample_size = 20 ∧
        ∃ (prob_selecting_two_from_C : ℚ),
          prob_selecting_two_from_C = 1 / 45 ∧
          total_employees = 100) :=
sorry

end NUMINAMATH_GPT_company_employee_count_l584_58438


namespace NUMINAMATH_GPT_tourists_left_l584_58451

noncomputable def tourists_remaining {initial remaining poisoned recovered : ℕ} 
  (h1 : initial = 30)
  (h2 : remaining = initial - 2)
  (h3 : poisoned = remaining / 2)
  (h4 : recovered = poisoned / 7)
  (h5 : remaining % 2 = 0) -- ensuring even division for / 2
  (h6 : poisoned % 7 = 0) -- ensuring even division for / 7
  : ℕ :=
  remaining - poisoned + recovered

theorem tourists_left 
  (initial remaining poisoned recovered : ℕ) 
  (h1 : initial = 30)
  (h2 : remaining = initial - 2)
  (h3 : poisoned = remaining / 2)
  (h4 : recovered = poisoned / 7)
  (h5 : remaining % 2 = 0) -- ensuring even division for / 2
  (h6 : poisoned % 7 = 0) -- ensuring even division for / 7
  : tourists_remaining h1 h2 h3 h4 h5 h6 = 16 :=
  by
  sorry

end NUMINAMATH_GPT_tourists_left_l584_58451


namespace NUMINAMATH_GPT_find_percentage_l584_58442

theorem find_percentage (P : ℝ) (h : (P / 100) * 600 = (50 / 100) * 720) : P = 60 :=
by
  sorry

end NUMINAMATH_GPT_find_percentage_l584_58442


namespace NUMINAMATH_GPT_number_of_new_trailer_homes_l584_58495

-- Definitions coming from the conditions
def initial_trailers : ℕ := 30
def initial_avg_age : ℕ := 15
def years_passed : ℕ := 5
def current_avg_age : ℕ := initial_avg_age + years_passed

-- Let 'n' be the number of new trailer homes added five years ago
variable (n : ℕ)

def new_trailer_age : ℕ := years_passed
def total_trailers : ℕ := initial_trailers + n
def total_ages : ℕ := (initial_trailers * current_avg_age) + (n * new_trailer_age)
def combined_avg_age := total_ages / total_trailers

theorem number_of_new_trailer_homes (h : combined_avg_age = 12) : n = 34 := 
sorry

end NUMINAMATH_GPT_number_of_new_trailer_homes_l584_58495


namespace NUMINAMATH_GPT_apples_kilos_first_scenario_l584_58400

noncomputable def cost_per_kilo_oranges : ℝ := 29
noncomputable def cost_per_kilo_apples : ℝ := 29
noncomputable def cost_first_scenario : ℝ := 419
noncomputable def cost_second_scenario : ℝ := 488
noncomputable def kilos_oranges_first_scenario : ℝ := 6
noncomputable def kilos_oranges_second_scenario : ℝ := 5
noncomputable def kilos_apples_second_scenario : ℝ := 7

theorem apples_kilos_first_scenario
  (O A : ℝ) 
  (cost1 cost2 : ℝ) 
  (k_oranges1 k_oranges2 k_apples2 : ℝ) 
  (hO : O = 29) (hA : A = 29) 
  (hCost1 : k_oranges1 * O + x * A = cost1) 
  (hCost2 : k_oranges2 * O + k_apples2 * A = cost2) 
  : x = 8 :=
by
  have hO : O = 29 := sorry
  have hA : A = 29 := sorry
  have h1 : k_oranges1 * O + x * A = cost1 := sorry
  have h2 : k_oranges2 * O + k_apples2 * A = cost2 := sorry
  sorry

end NUMINAMATH_GPT_apples_kilos_first_scenario_l584_58400


namespace NUMINAMATH_GPT_problem_1_problem_2_l584_58437

def f (x : ℝ) : ℝ := x^2 + 4 * x
def g (a : ℝ) : ℝ := |a - 2| + |a + 1|

theorem problem_1 (x : ℝ) :
    (f x ≥ g 3) ↔ (x ≥ 1 ∨ x ≤ -5) :=
  sorry

theorem problem_2 (a : ℝ) :
    (∃ x : ℝ, f x + g a = 0) → (-3 / 2 ≤ a ∧ a ≤ 5 / 2) :=
  sorry

end NUMINAMATH_GPT_problem_1_problem_2_l584_58437


namespace NUMINAMATH_GPT_initial_forks_l584_58428

variables (forks knives spoons teaspoons : ℕ)
variable (F : ℕ)

-- Conditions as given
def num_knives := F + 9
def num_spoons := 2 * (F + 9)
def num_teaspoons := F / 2
def total_cutlery := (F + 2) + (F + 11) + (2 * (F + 9) + 2) + (F / 2 + 2)

-- Problem statement to prove
theorem initial_forks :
  (total_cutlery = 62) ↔ (F = 6) :=
by {
  sorry
}

end NUMINAMATH_GPT_initial_forks_l584_58428


namespace NUMINAMATH_GPT_largest_multiple_of_11_neg_greater_minus_210_l584_58496

theorem largest_multiple_of_11_neg_greater_minus_210 :
  ∃ (x : ℤ), x % 11 = 0 ∧ -x < -210 ∧ ∀ y, y % 11 = 0 ∧ -y < -210 → y ≤ x :=
sorry

end NUMINAMATH_GPT_largest_multiple_of_11_neg_greater_minus_210_l584_58496


namespace NUMINAMATH_GPT_certain_number_is_84_l584_58493

/-
The least number by which 72 must be multiplied in order to produce a multiple of a certain number is 14.
What is that certain number?
-/

theorem certain_number_is_84 (x : ℕ) (h: 72 * 14 % x = 0 ∧ ∀ y : ℕ, 1 ≤ y → y < 14 → 72 * y % x ≠ 0) : x = 84 :=
sorry

end NUMINAMATH_GPT_certain_number_is_84_l584_58493


namespace NUMINAMATH_GPT_product_of_roots_l584_58465

theorem product_of_roots (a b c : ℝ) (h_eq : 24 * a^2 + 36 * a - 648 = 0) : a * c = -27 := 
by
  have h_root_product : (24 * a^2 + 36 * a - 648) = 0 ↔ a = -27 := sorry
  exact sorry

end NUMINAMATH_GPT_product_of_roots_l584_58465


namespace NUMINAMATH_GPT_solve_x_l584_58452

theorem solve_x :
  (1 / 4 - 1 / 6) = 1 / (12 : ℝ) :=
by sorry

end NUMINAMATH_GPT_solve_x_l584_58452


namespace NUMINAMATH_GPT_bailey_chew_toys_l584_58470

theorem bailey_chew_toys (dog_treats rawhide_bones: ℕ) (cards items_per_card : ℕ)
  (h1 : dog_treats = 8)
  (h2 : rawhide_bones = 10)
  (h3 : cards = 4)
  (h4 : items_per_card = 5) :
  ∃ chew_toys : ℕ, chew_toys = 2 :=
by
  sorry

end NUMINAMATH_GPT_bailey_chew_toys_l584_58470


namespace NUMINAMATH_GPT_kody_half_mohamed_years_ago_l584_58473

-- Definitions of initial conditions
def current_age_mohamed : ℕ := 2 * 30
def current_age_kody : ℕ := 32

-- Proof statement
theorem kody_half_mohamed_years_ago : ∃ x : ℕ, (current_age_kody - x) = (1 / 2 : ℕ) * (current_age_mohamed - x) ∧ x = 4 := 
by 
  sorry

end NUMINAMATH_GPT_kody_half_mohamed_years_ago_l584_58473


namespace NUMINAMATH_GPT_pears_picking_total_l584_58407

theorem pears_picking_total :
  let Jason_day1 := 46
  let Keith_day1 := 47
  let Mike_day1 := 12
  let Alicia_day1 := 28
  let Tina_day1 := 33
  let Nicola_day1 := 52

  let Jason_day2 := Jason_day1 / 2
  let Keith_day2 := Keith_day1 / 2
  let Mike_day2 := Mike_day1 / 2
  let Alicia_day2 := 2 * Alicia_day1
  let Tina_day2 := 2 * Tina_day1
  let Nicola_day2 := 2 * Nicola_day1

  let Jason_day3 := (Jason_day1 + Jason_day2) / 2
  let Keith_day3 := (Keith_day1 + Keith_day2) / 2
  let Mike_day3 := (Mike_day1 + Mike_day2) / 2
  let Alicia_day3 := (Alicia_day1 + Alicia_day2) / 2
  let Tina_day3 := (Tina_day1 + Tina_day2) / 2
  let Nicola_day3 := (Nicola_day1 + Nicola_day2) / 2

  let Jason_total := Jason_day1 + Jason_day2 + Jason_day3
  let Keith_total := Keith_day1 + Keith_day2 + Keith_day3
  let Mike_total := Mike_day1 + Mike_day2 + Mike_day3
  let Alicia_total := Alicia_day1 + Alicia_day2 + Alicia_day3
  let Tina_total := Tina_day1 + Tina_day2 + Tina_day3
  let Nicola_total := Nicola_day1 + Nicola_day2 + Nicola_day3

  let overall_total := Jason_total + Keith_total + Mike_total + Alicia_total + Tina_total + Nicola_total

  overall_total = 747 := by
  intro Jason_day1 Jason_day2 Jason_day3 Jason_total
  intro Keith_day1 Keith_day2 Keith_day3 Keith_total
  intro Mike_day1 Mike_day2 Mike_day3 Mike_total
  intro Alicia_day1 Alicia_day2 Alicia_day3 Alicia_total
  intro Tina_day1 Tina_day2 Tina_day3 Tina_total
  intro Nicola_day1 Nicola_day2 Nicola_day3 Nicola_total

  sorry

end NUMINAMATH_GPT_pears_picking_total_l584_58407


namespace NUMINAMATH_GPT_no_such_integers_l584_58482

theorem no_such_integers (a b : ℤ) : 
  ¬ (∃ a b : ℤ, ∃ k₁ k₂ : ℤ, a^5 * b + 3 = k₁^3 ∧ a * b^5 + 3 = k₂^3) :=
by 
  sorry

end NUMINAMATH_GPT_no_such_integers_l584_58482


namespace NUMINAMATH_GPT_trapezoid_circumcircle_radius_l584_58461

theorem trapezoid_circumcircle_radius :
  ∀ (BC AD height midline R : ℝ), 
  (BC / AD = (5 / 12)) →
  (height = 17) →
  (midline = height) →
  (midline = (BC + AD) / 2) →
  (BC = 10) →
  (AD = 24) →
  R = 13 :=
by
  intro BC AD height midline R
  intros h_ratio h_height h_midline_eq_height h_midline_eq_avg_bases h_BC h_AD
  -- Proof would go here, but it's skipped for now.
  sorry

end NUMINAMATH_GPT_trapezoid_circumcircle_radius_l584_58461


namespace NUMINAMATH_GPT_find_real_solutions_l584_58480

variable (x : ℝ)

theorem find_real_solutions :
  (1 / ((x - 1) * (x - 3)) + 1 / ((x - 3) * (x - 5)) + 1 / ((x - 5) * (x - 7)) = 1 / 8) ↔ 
  (x = -2 * Real.sqrt 14 ∨ x = 2 * Real.sqrt 14) := 
sorry

end NUMINAMATH_GPT_find_real_solutions_l584_58480


namespace NUMINAMATH_GPT_number_to_add_l584_58417

theorem number_to_add (a m : ℕ) (h₁ : a = 7844213) (h₂ : m = 549) :
  ∃ n, (a + n) % m = 0 ∧ n = m - (a % m) :=
by
  sorry

end NUMINAMATH_GPT_number_to_add_l584_58417


namespace NUMINAMATH_GPT_seventh_term_l584_58462

def nth_term (n : ℕ) (a : ℝ) : ℝ :=
  (-2) ^ n * a ^ (2 * n - 1)

theorem seventh_term (a : ℝ) : nth_term 7 a = -128 * a ^ 13 :=
by sorry

end NUMINAMATH_GPT_seventh_term_l584_58462


namespace NUMINAMATH_GPT_initial_pencils_count_l584_58449

variables {pencils_taken : ℕ} {pencils_left : ℕ} {initial_pencils : ℕ}

theorem initial_pencils_count 
  (h1 : pencils_taken = 4)
  (h2 : pencils_left = 5) :
  initial_pencils = 9 :=
by 
  sorry

end NUMINAMATH_GPT_initial_pencils_count_l584_58449


namespace NUMINAMATH_GPT_number_of_elements_in_A_l584_58458

theorem number_of_elements_in_A (a b : ℕ) (h1 : a = 3 * b)
  (h2 : a + b - 100 = 500) (h3 : 100 = 100) (h4 : a - 100 = b - 100 + 50) : a = 450 := by
  sorry

end NUMINAMATH_GPT_number_of_elements_in_A_l584_58458


namespace NUMINAMATH_GPT_correct_operation_l584_58483

theorem correct_operation (a b : ℝ) : (a + b) * (a - b) = a^2 - b^2 := by
  sorry

end NUMINAMATH_GPT_correct_operation_l584_58483


namespace NUMINAMATH_GPT_determinant_condition_l584_58446

theorem determinant_condition (a b c d : ℤ)
    (H : ∀ m n : ℤ, ∃ x y : ℤ, a * x + b * y = m ∧ c * x + d * y = n) :
    a * d - b * c = 1 ∨ a * d - b * c = -1 :=
by 
  sorry

end NUMINAMATH_GPT_determinant_condition_l584_58446


namespace NUMINAMATH_GPT_find_x_intercept_l584_58478

-- Define the equation of the line
def line_eq (x y : ℝ) : Prop := 4 * x + 7 * y = 28

-- Define the x-intercept point when y = 0
def x_intercept (x : ℝ) : Prop := line_eq x 0

-- Prove that for the x-intercept, when y = 0, x = 7
theorem find_x_intercept : x_intercept 7 :=
by
  -- proof would go here
  sorry

end NUMINAMATH_GPT_find_x_intercept_l584_58478


namespace NUMINAMATH_GPT_volume_of_sphere_l584_58408

theorem volume_of_sphere
  (a b c : ℝ)
  (h1 : a * b * c = 4 * Real.sqrt 6)
  (h2 : a * b = 2 * Real.sqrt 3)
  (h3 : b * c = 4 * Real.sqrt 3)
  (O_radius : ℝ := Real.sqrt (a^2 + b^2 + c^2) / 2) :
  4 / 3 * Real.pi * O_radius^3 = 32 * Real.pi / 3 := by
  sorry

end NUMINAMATH_GPT_volume_of_sphere_l584_58408


namespace NUMINAMATH_GPT_sum_of_first_six_terms_of_geom_seq_l584_58403

theorem sum_of_first_six_terms_of_geom_seq :
  let a := (1 : ℝ) / 4
  let r := (1 : ℝ) / 4
  let S6 := a * (1 - r^6) / (1 - r)
  S6 = 4095 / 12288 := by
sorry

end NUMINAMATH_GPT_sum_of_first_six_terms_of_geom_seq_l584_58403


namespace NUMINAMATH_GPT_ratio_of_kids_waiting_for_slide_to_swings_final_ratio_of_kids_waiting_l584_58431

-- Define the conditions
def W : ℕ := 3
def wait_time_swing : ℕ := 120 * W
def wait_time_slide (S : ℕ) : ℕ := 15 * S
def wait_diff_condition (S : ℕ) : Prop := wait_time_swing - wait_time_slide S = 270

theorem ratio_of_kids_waiting_for_slide_to_swings (S : ℕ) (h : wait_diff_condition S) : S = 6 :=
by
  -- placeholder proof
  sorry

theorem final_ratio_of_kids_waiting (S : ℕ) (h : wait_diff_condition S) : S / W = 2 :=
by
  -- placeholder proof
  sorry

end NUMINAMATH_GPT_ratio_of_kids_waiting_for_slide_to_swings_final_ratio_of_kids_waiting_l584_58431


namespace NUMINAMATH_GPT_cos_pi_minus_alpha_l584_58423

theorem cos_pi_minus_alpha (α : ℝ) (h : Real.sin (Real.pi / 2 + α) = 1 / 7) : Real.cos (Real.pi - α) = - (1 / 7) := by
  sorry

end NUMINAMATH_GPT_cos_pi_minus_alpha_l584_58423


namespace NUMINAMATH_GPT_prod_eq_diff_squares_l584_58409

variable (a b : ℝ)

theorem prod_eq_diff_squares :
  ( (1 / 4 * a + b) * (b - 1 / 4 * a) = b^2 - (1 / 16 * a^2) ) :=
by
  sorry

end NUMINAMATH_GPT_prod_eq_diff_squares_l584_58409


namespace NUMINAMATH_GPT_distance_traveled_l584_58419

theorem distance_traveled
  (D : ℝ) (T : ℝ)
  (h1 : D = 10 * T)
  (h2 : D + 20 = 14 * T)
  : D = 50 := sorry

end NUMINAMATH_GPT_distance_traveled_l584_58419


namespace NUMINAMATH_GPT_quadratic_equation_correct_form_l584_58456

theorem quadratic_equation_correct_form :
  ∀ (a b c x : ℝ), a = 3 → b = -6 → c = 1 → a * x^2 + c = b * x :=
by
  intros a b c x ha hb hc
  rw [ha, hb, hc]
  sorry

end NUMINAMATH_GPT_quadratic_equation_correct_form_l584_58456


namespace NUMINAMATH_GPT_birch_tree_count_l584_58464

theorem birch_tree_count:
  let total_trees := 8000
  let spruces := 0.12 * total_trees
  let pines := 0.15 * total_trees
  let maples := 0.18 * total_trees
  let cedars := 0.09 * total_trees
  let oaks := spruces + pines
  let calculated_trees := spruces + pines + maples + cedars + oaks
  let birches := total_trees - calculated_trees
  spruces = 960 → pines = 1200 → maples = 1440 → cedars = 720 → oaks = 2160 →
  birches = 1520 :=
by
  intros
  sorry

end NUMINAMATH_GPT_birch_tree_count_l584_58464


namespace NUMINAMATH_GPT_find_k_l584_58463

theorem find_k (x1 x2 : ℝ) (r : ℝ) (h1 : x1 = 3 * r) (h2 : x2 = r) (h3 : x1 + x2 = -8) (h4 : x1 * x2 = k) : k = 12 :=
by
  -- proof steps here
  sorry

end NUMINAMATH_GPT_find_k_l584_58463


namespace NUMINAMATH_GPT_quadruplet_zero_solution_l584_58477

theorem quadruplet_zero_solution (a b c d : ℝ)
  (h1 : (a + b) * (a^2 + b^2) = (c + d) * (c^2 + d^2))
  (h2 : (a + c) * (a^2 + c^2) = (b + d) * (b^2 + d^2))
  (h3 : (a + d) * (a^2 + d^2) = (b + c) * (b^2 + c^2)) :
  a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0 := 
sorry

end NUMINAMATH_GPT_quadruplet_zero_solution_l584_58477


namespace NUMINAMATH_GPT_supplements_of_congruent_angles_are_congruent_l584_58476

-- Define the concept of supplementary angles
def is_supplementary (α β : ℝ) : Prop := α + β = 180

-- Statement of the problem
theorem supplements_of_congruent_angles_are_congruent :
  ∀ {α β γ δ : ℝ},
  is_supplementary α β →
  is_supplementary γ δ →
  β = δ →
  α = γ :=
by
  intros α β γ δ h1 h2 h3
  sorry

end NUMINAMATH_GPT_supplements_of_congruent_angles_are_congruent_l584_58476


namespace NUMINAMATH_GPT_steps_to_school_l584_58489

-- Define the conditions as assumptions
def distance : Float := 900
def step_length : Float := 0.45

-- Define the statement to be proven
theorem steps_to_school (x : Float) : step_length * x = distance → x = 2000 := by
  intro h
  sorry

end NUMINAMATH_GPT_steps_to_school_l584_58489


namespace NUMINAMATH_GPT_valid_paths_in_grid_l584_58444

theorem valid_paths_in_grid : 
  let total_paths := Nat.choose 15 4;
  let paths_through_EF := (Nat.choose 7 2) * (Nat.choose 7 2);
  let valid_paths := total_paths - 2 * paths_through_EF;
  grid_size == (11, 4) ∧
  blocked_segments == [((5, 2), (5, 3)), ((6, 2), (6, 3))] 
  → valid_paths = 483 :=
by
  sorry

end NUMINAMATH_GPT_valid_paths_in_grid_l584_58444


namespace NUMINAMATH_GPT_candy_game_win_l584_58491

def winning_player (A B : ℕ) : String :=
  if (A % B = 0 ∨ B % A = 0) then "Player with forcing checks" else "No inevitable winner"

theorem candy_game_win :
  winning_player 1000 2357 = "Player with forcing checks" :=
by
  sorry

end NUMINAMATH_GPT_candy_game_win_l584_58491


namespace NUMINAMATH_GPT_pyramid_volume_l584_58459

theorem pyramid_volume
  (s : ℝ) (h : ℝ) (base_area : ℝ) (triangular_face_area : ℝ) (surface_area : ℝ)
  (h_base_area : base_area = s * s)
  (h_triangular_face_area : triangular_face_area = (1 / 3) * base_area)
  (h_surface_area : surface_area = base_area + 4 * triangular_face_area)
  (h_surface_area_value : surface_area = 768)
  (h_vol : h = 7.78) :
  (1 / 3) * base_area * h = 853.56 :=
by
  sorry

end NUMINAMATH_GPT_pyramid_volume_l584_58459


namespace NUMINAMATH_GPT_total_price_of_hats_l584_58499

variables (total_hats : ℕ) (blue_hat_cost : ℕ) (green_hat_cost : ℕ) (green_hats : ℕ) (total_price : ℕ)

def total_number_of_hats := 85
def cost_per_blue_hat := 6
def cost_per_green_hat := 7
def number_of_green_hats := 30

theorem total_price_of_hats :
  (number_of_green_hats * cost_per_green_hat) + ((total_number_of_hats - number_of_green_hats) * cost_per_blue_hat) = 540 :=
sorry

end NUMINAMATH_GPT_total_price_of_hats_l584_58499


namespace NUMINAMATH_GPT_range_of_t_max_radius_circle_eq_l584_58422

-- Definitions based on conditions
def circle_equation (x y t : ℝ) := x^2 + y^2 - 2 * x + t^2 = 0

-- Statement for the range of values of t
theorem range_of_t (t : ℝ) (h : ∃ x y : ℝ, circle_equation x y t) : -1 < t ∧ t < 1 := sorry

-- Statement for the equation of the circle when t = 0
theorem max_radius_circle_eq (x y : ℝ) (h : circle_equation x y 0) : (x - 1)^2 + y^2 = 1 := sorry

end NUMINAMATH_GPT_range_of_t_max_radius_circle_eq_l584_58422


namespace NUMINAMATH_GPT_area_of_triangle_l584_58430

theorem area_of_triangle {a b c : ℝ} (S : ℝ) (h1 : (a^2) * (Real.sin C) = 4 * (Real.sin A))
                          (h2 : (a + c)^2 = 12 + b^2)
                          (h3 : S = Real.sqrt ((1/4) * (a^2 * c^2 - ( (a^2 + c^2 - b^2)/2 )^2))) :
  S = Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_area_of_triangle_l584_58430


namespace NUMINAMATH_GPT_integer_1000_column_l584_58432

def column_sequence (n : ℕ) : String :=
  let sequence := ["A", "B", "C", "D", "E", "F", "E", "D", "C", "B"]
  sequence.get! (n % 10)

theorem integer_1000_column : column_sequence 999 = "C" :=
by
  sorry

end NUMINAMATH_GPT_integer_1000_column_l584_58432


namespace NUMINAMATH_GPT_cost_of_each_toy_car_l584_58492

theorem cost_of_each_toy_car (S M C A B : ℕ) (hS : S = 53) (hM : M = 7) (hA : A = 10) (hB : B = 14) 
(hTotalSpent : S - M = C + A + B) (hTotalCars : 2 * C / 2 = 11) : 
C / 2 = 11 :=
by
  rw [hS, hM, hA, hB] at hTotalSpent
  sorry

end NUMINAMATH_GPT_cost_of_each_toy_car_l584_58492


namespace NUMINAMATH_GPT_average_of_remaining_numbers_l584_58424

variable (numbers : List ℝ) (x y : ℝ)

theorem average_of_remaining_numbers
  (h_length_15 : numbers.length = 15)
  (h_avg_15 : (numbers.sum / 15) = 90)
  (h_x : x = 80)
  (h_y : y = 85)
  (h_members : x ∈ numbers ∧ y ∈ numbers) :
  ((numbers.sum - x - y) / 13) = 91.15 :=
sorry

end NUMINAMATH_GPT_average_of_remaining_numbers_l584_58424


namespace NUMINAMATH_GPT_smallest_n_for_perfect_square_and_cube_l584_58441

theorem smallest_n_for_perfect_square_and_cube :
  ∃ n : ℕ, (∃ a : ℕ, 4 * n = a^2) ∧ (∃ b : ℕ, 5 * n = b^3) ∧ n = 125 :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_for_perfect_square_and_cube_l584_58441


namespace NUMINAMATH_GPT_seating_arrangement_l584_58468

variable {M I P A : Prop}

def first_fact : ¬ M := sorry
def second_fact : ¬ A := sorry
def third_fact : ¬ M → I := sorry
def fourth_fact : I → P := sorry

theorem seating_arrangement : ¬ M → (I ∧ P) :=
by
  intros hM
  have hI : I := third_fact hM
  have hP : P := fourth_fact hI
  exact ⟨hI, hP⟩

end NUMINAMATH_GPT_seating_arrangement_l584_58468


namespace NUMINAMATH_GPT_find_interest_rate_l584_58445

noncomputable def interest_rate (A P T : ℚ) : ℚ := (A - P) / (P * T) * 100

theorem find_interest_rate :
  let A := 1120
  let P := 921.0526315789474
  let T := 2.4
  interest_rate A P T = 9 := 
by
  sorry

end NUMINAMATH_GPT_find_interest_rate_l584_58445


namespace NUMINAMATH_GPT_molecular_weight_CO_l584_58450

theorem molecular_weight_CO : 
  let molecular_weight_C := 12.01
  let molecular_weight_O := 16.00
  molecular_weight_C + molecular_weight_O = 28.01 :=
by
  sorry

end NUMINAMATH_GPT_molecular_weight_CO_l584_58450


namespace NUMINAMATH_GPT_pie_difference_l584_58484

theorem pie_difference:
  ∀ (a b c d : ℚ), a = 6 / 7 → b = 3 / 4 → (a - b) = c → c = 3 / 28 :=
by
  sorry

end NUMINAMATH_GPT_pie_difference_l584_58484


namespace NUMINAMATH_GPT_solve_equation_l584_58401

theorem solve_equation (x : ℝ) : 
  (4 * (1 - x)^2 = 25) ↔ (x = -3 / 2 ∨ x = 7 / 2) := 
by 
  sorry

end NUMINAMATH_GPT_solve_equation_l584_58401


namespace NUMINAMATH_GPT_pascal_tenth_number_in_hundred_row_l584_58427

def pascal_row (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem pascal_tenth_number_in_hundred_row :
  pascal_row 99 9 = Nat.choose 99 9 :=
by
  sorry

end NUMINAMATH_GPT_pascal_tenth_number_in_hundred_row_l584_58427


namespace NUMINAMATH_GPT_maximal_n_for_sequence_l584_58411

theorem maximal_n_for_sequence
  (a : ℕ → ℤ)
  (n : ℕ)
  (h1 : ∀ i : ℕ, 1 ≤ i ∧ i ≤ n - 2 → a i + a (i + 1) + a (i + 2) > 0)
  (h2 : ∀ i : ℕ, 1 ≤ i ∧ i ≤ n - 4 → a i + a (i + 1) + a (i + 2) + a (i + 3) + a (i + 4) < 0)
  : n ≤ 9 :=
sorry

end NUMINAMATH_GPT_maximal_n_for_sequence_l584_58411


namespace NUMINAMATH_GPT_part_I_part_II_l584_58429

-- Part (I): If a = 1, prove that q implies p
theorem part_I (x : ℝ) (h : 3 < x ∧ x < 4) : (1 < x) ∧ (x < 4) :=
by sorry

-- Part (II): Prove the range of a for which p is necessary but not sufficient for q
theorem part_II (a : ℝ) (h1 : a > 0) (h2 : ∀ x : ℝ, (a < x ∧ x < 4 * a) → (3 < x ∧ x < 4)) : 1 < a ∧ a ≤ 3 :=
by sorry

end NUMINAMATH_GPT_part_I_part_II_l584_58429


namespace NUMINAMATH_GPT_sum_of_first_9_terms_of_arithmetic_sequence_l584_58494

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = (n * (a 1 + a n)) / 2

theorem sum_of_first_9_terms_of_arithmetic_sequence 
  (h1 : is_arithmetic_sequence a) 
  (h2 : a 2 + a 8 = 18) 
  (h3 : sum_of_first_n_terms a S) :
  S 9 = 81 :=
sorry

end NUMINAMATH_GPT_sum_of_first_9_terms_of_arithmetic_sequence_l584_58494


namespace NUMINAMATH_GPT_second_solution_concentration_l584_58486

def volume1 : ℝ := 5
def concentration1 : ℝ := 0.04
def volume2 : ℝ := 2.5
def concentration_final : ℝ := 0.06
def total_silver1 : ℝ := volume1 * concentration1
def total_volume : ℝ := volume1 + volume2
def total_silver_final : ℝ := total_volume * concentration_final

theorem second_solution_concentration :
  ∃ (C2 : ℝ), total_silver1 + volume2 * C2 = total_silver_final ∧ C2 = 0.1 := 
by 
  sorry

end NUMINAMATH_GPT_second_solution_concentration_l584_58486


namespace NUMINAMATH_GPT_interval_solution_l584_58443

theorem interval_solution (x : ℝ) : 2 ≤ x / (2 * x - 5) ∧ x / (2 * x - 5) < 7 ↔ (35 / 13 : ℝ) < x ∧ x ≤ 10 / 3 :=
by
  sorry

end NUMINAMATH_GPT_interval_solution_l584_58443


namespace NUMINAMATH_GPT_required_percentage_to_pass_l584_58421

theorem required_percentage_to_pass
  (marks_obtained : ℝ)
  (marks_failed_by : ℝ)
  (max_marks : ℝ)
  (passing_marks := marks_obtained + marks_failed_by)
  (required_percentage : ℝ := (passing_marks / max_marks) * 100)
  (h : marks_obtained = 80)
  (h' : marks_failed_by = 40)
  (h'' : max_marks = 200) :
  required_percentage = 60 := 
by
  sorry

end NUMINAMATH_GPT_required_percentage_to_pass_l584_58421


namespace NUMINAMATH_GPT_max_count_larger_than_20_l584_58418

noncomputable def max_larger_than_20 (int_list : List Int) : Nat :=
  (int_list.filter (λ n => n > 20)).length

theorem max_count_larger_than_20 (a1 a2 a3 a4 a5 a6 a7 a8 : Int)
  (h_sum : a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 = 10) :
  ∃ (k : Nat), k = 7 ∧ max_larger_than_20 [a1, a2, a3, a4, a5, a6, a7, a8] = k :=
sorry

end NUMINAMATH_GPT_max_count_larger_than_20_l584_58418


namespace NUMINAMATH_GPT_solve_for_a_l584_58460

-- Given the equation is quadratic, meaning the highest power of x in the quadratic term equals 2
theorem solve_for_a (a : ℚ) : (2 * a - 1 = 2) -> a = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_a_l584_58460


namespace NUMINAMATH_GPT_total_rope_length_l584_58488

theorem total_rope_length 
  (longer_side : ℕ) (shorter_side : ℕ) 
  (h1 : longer_side = 28) (h2 : shorter_side = 22) : 
  2 * longer_side + 2 * shorter_side = 100 := by
  sorry

end NUMINAMATH_GPT_total_rope_length_l584_58488


namespace NUMINAMATH_GPT_total_cars_in_group_l584_58405

theorem total_cars_in_group (C : ℕ)
  (h1 : 37 ≤ C)
  (h2 : ∃ n ≥ 51, n ≤ C)
  (h3 : ∃ n ≤ 49, n + 51 = C - 37) :
  C = 137 :=
by
  sorry

end NUMINAMATH_GPT_total_cars_in_group_l584_58405


namespace NUMINAMATH_GPT_binomial_coefficient_middle_term_l584_58498

theorem binomial_coefficient_middle_term :
  let n := 11
  let sum_odd := 1024
  sum_odd = 2^(n-1) →
  let binom_coef := Nat.choose n (n / 2 - 1)
  binom_coef = 462 :=
by
  intro n
  let n := 11
  intro sum_odd
  let sum_odd := 1024
  intro h
  let binom_coef := Nat.choose n (n / 2 - 1)
  have : binom_coef = 462 := sorry
  exact this

end NUMINAMATH_GPT_binomial_coefficient_middle_term_l584_58498


namespace NUMINAMATH_GPT_max_number_of_pies_l584_58454

def total_apples := 250
def apples_given_to_students := 42
def apples_used_for_juice := 75
def apples_per_pie := 8

theorem max_number_of_pies (h1 : total_apples = 250)
                           (h2 : apples_given_to_students = 42)
                           (h3 : apples_used_for_juice = 75)
                           (h4 : apples_per_pie = 8) :
  ((total_apples - apples_given_to_students - apples_used_for_juice) / apples_per_pie) ≥ 16 :=
by
  sorry

end NUMINAMATH_GPT_max_number_of_pies_l584_58454


namespace NUMINAMATH_GPT_total_money_raised_l584_58435

-- Assume there are 30 students in total
def total_students := 30

-- Assume 10 students raised $20 each
def students_raising_20 := 10
def money_raised_per_20 := 20

-- The rest of the students raised $30 each
def students_raising_30 := total_students - students_raising_20
def money_raised_per_30 := 30

-- Prove that the total amount raised is $800
theorem total_money_raised :
  (students_raising_20 * money_raised_per_20) +
  (students_raising_30 * money_raised_per_30) = 800 :=
by
  sorry

end NUMINAMATH_GPT_total_money_raised_l584_58435


namespace NUMINAMATH_GPT_evaluate_expression_l584_58457

theorem evaluate_expression :
  (2 + 3 / (4 + 5 / (6 + 7 / 8))) = 137 / 52 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l584_58457


namespace NUMINAMATH_GPT_age_of_fourth_child_l584_58425

theorem age_of_fourth_child (c1 c2 c3 c4 : ℕ) (h1 : c1 = 15)
  (h2 : c2 = c1 - 1) (h3 : c3 = c2 - 4)
  (h4 : c4 = c3 - 2) : c4 = 8 :=
by
  sorry

end NUMINAMATH_GPT_age_of_fourth_child_l584_58425


namespace NUMINAMATH_GPT_solve_for_p_l584_58448

variable (p q : ℝ)
noncomputable def binomial_third_term : ℝ := 55 * p^9 * q^2
noncomputable def binomial_fourth_term : ℝ := 165 * p^8 * q^3

theorem solve_for_p (h1 : p + q = 1) (h2 : binomial_third_term p q = binomial_fourth_term p q) : p = 3 / 4 :=
by sorry

end NUMINAMATH_GPT_solve_for_p_l584_58448


namespace NUMINAMATH_GPT_rearrangement_impossible_l584_58447

-- Definition of an 8x8 chessboard's cell numbering.
def cell_number (i j : ℕ) : ℕ := i + j - 1

-- The initial placement of pieces, represented as a permutation on {1, 2, ..., 8}
def initial_placement (p: Fin 8 → Fin 8) := True -- simplify for definition purposes

-- The rearranged placement of pieces
def rearranged_placement (q: Fin 8 → Fin 8) := True -- simplify for definition purposes

-- Condition for each piece: cell number increases
def cell_increase_condition (p q: Fin 8 → Fin 8) : Prop :=
  ∀ i, cell_number (q i).val (i.val + 1) > cell_number (p i).val (i.val + 1)

-- The main theorem to state it's impossible to rearrange under the given conditions and question
theorem rearrangement_impossible 
  (p q: Fin 8 → Fin 8) 
  (h_initial : initial_placement p) 
  (h_rearranged : rearranged_placement q) 
  (h_increase : cell_increase_condition p q) : False := 
sorry

end NUMINAMATH_GPT_rearrangement_impossible_l584_58447


namespace NUMINAMATH_GPT_students_last_year_l584_58474

theorem students_last_year (students_this_year : ℝ) (increase_percent : ℝ) (last_year_students : ℝ) 
  (h1 : students_this_year = 960) 
  (h2 : increase_percent = 0.20) 
  (h3 : students_this_year = last_year_students * (1 + increase_percent)) : 
  last_year_students = 800 :=
by 
  sorry

end NUMINAMATH_GPT_students_last_year_l584_58474


namespace NUMINAMATH_GPT_eddy_travel_time_l584_58455

theorem eddy_travel_time (T : ℝ) (S_e S_f : ℝ) (Freddy_time : ℝ := 4)
  (distance_AB : ℝ := 540) (distance_AC : ℝ := 300) (speed_ratio : ℝ := 2.4) :
  (distance_AB / T = 2.4 * (distance_AC / Freddy_time)) -> T = 3 :=
by
  sorry

end NUMINAMATH_GPT_eddy_travel_time_l584_58455


namespace NUMINAMATH_GPT_number_subtracted_l584_58472

theorem number_subtracted (t k x : ℝ) (h1 : t = (5 / 9) * (k - x)) (h2 : t = 105) (h3 : k = 221) : x = 32 :=
by
  sorry

end NUMINAMATH_GPT_number_subtracted_l584_58472


namespace NUMINAMATH_GPT_Alex_failing_implies_not_all_hw_on_time_l584_58497

-- Definitions based on the conditions provided
variable (Alex_submits_all_hw_on_time : Prop)
variable (Alex_passes_course : Prop)

-- Given condition: Submitting all homework assignments implies passing the course
axiom Mrs_Thompson_statement : Alex_submits_all_hw_on_time → Alex_passes_course

-- The problem: Prove that if Alex failed the course, then he did not submit all homework assignments on time
theorem Alex_failing_implies_not_all_hw_on_time (h : ¬Alex_passes_course) : ¬Alex_submits_all_hw_on_time :=
  by
  sorry

end NUMINAMATH_GPT_Alex_failing_implies_not_all_hw_on_time_l584_58497


namespace NUMINAMATH_GPT_geometric_locus_points_l584_58402

theorem geometric_locus_points :
  (∀ x y : ℝ, (y^2 = x^2) ↔ (y = x ∨ y = -x)) ∧
  (∀ x : ℝ, (x^2 - 2 * x + 1 = 0) ↔ (x = 1)) ∧
  (∀ x y : ℝ, (x^2 + y^2 = 4 * (y - 1)) ↔ (x = 0 ∧ y = 2)) ∧
  (∀ x y : ℝ, (x^2 - 2 * x * y + y^2 = -1) ↔ false) :=
by
  sorry

end NUMINAMATH_GPT_geometric_locus_points_l584_58402


namespace NUMINAMATH_GPT_neither_sufficient_nor_necessary_condition_l584_58467

-- Given conditions
def p (a : ℝ) : Prop := ∃ (x y : ℝ), a * x + y + 1 = 0 ∧ a * x - y + 2 = 0
def q : Prop := ∃ (a : ℝ), a = 1

-- The proof problem
theorem neither_sufficient_nor_necessary_condition : 
  ¬ ((∀ a, p a → q) ∧ (∀ a, q → p a)) :=
sorry

end NUMINAMATH_GPT_neither_sufficient_nor_necessary_condition_l584_58467


namespace NUMINAMATH_GPT_albert_earnings_l584_58453

theorem albert_earnings (E P : ℝ) 
  (h1 : E * 1.20 = 660) 
  (h2 : E * (1 + P) = 693) : 
  P = 0.26 :=
sorry

end NUMINAMATH_GPT_albert_earnings_l584_58453


namespace NUMINAMATH_GPT_complement_of_M_l584_58404

def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x^2 - x > 0}

theorem complement_of_M :
  (U \ M) = {x | 0 ≤ x ∧ x ≤ 1} :=
sorry

end NUMINAMATH_GPT_complement_of_M_l584_58404


namespace NUMINAMATH_GPT_diagonal_not_perpendicular_l584_58475

open Real

theorem diagonal_not_perpendicular (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) 
  (h_a_ne_b : a ≠ b) (h_c_ne_d : c ≠ d) (h_a_ne_c : a ≠ c) (h_b_ne_d : b ≠ d): 
  ¬ ((d - b) * (b - a) = - (c - a) * (d - c)) :=
by
  sorry

end NUMINAMATH_GPT_diagonal_not_perpendicular_l584_58475


namespace NUMINAMATH_GPT_add_fractions_l584_58410

theorem add_fractions (x : ℝ) (h : x ≠ 1) : (1 / (x - 1) + 3 / (x - 1)) = (4 / (x - 1)) :=
by
  sorry

end NUMINAMATH_GPT_add_fractions_l584_58410


namespace NUMINAMATH_GPT_line_intersects_circle_chord_min_length_l584_58415

-- Define the circle C
def C (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 25

-- Define the line L based on parameter m
def L (m x y : ℝ) : Prop := (2 * m + 1) * x + (m + 1) * y - 7 * m - 4 = 0

-- Prove that for any real number m, line L intersects circle C at two points.
theorem line_intersects_circle (m : ℝ) : 
  ∃ x y₁ y₂ : ℝ, y₁ ≠ y₂ ∧ C x y₁ ∧ C x y₂ ∧ L m x y₁ ∧ L m x y₂ :=
sorry

-- Prove the equation of line L in slope-intercept form when the chord cut by circle C has minimum length.
theorem chord_min_length : ∃ (m : ℝ), ∀ x y : ℝ, 
  L m x y ↔ y = 2 * x - 5 :=
sorry

end NUMINAMATH_GPT_line_intersects_circle_chord_min_length_l584_58415


namespace NUMINAMATH_GPT_distance_between_trees_l584_58440

theorem distance_between_trees (num_trees : ℕ) (total_length : ℕ) (num_spaces : ℕ) (distance_per_space : ℕ) 
  (h_num_trees : num_trees = 11) (h_total_length : total_length = 180)
  (h_num_spaces : num_spaces = num_trees - 1) (h_distance_per_space : distance_per_space = total_length / num_spaces) :
  distance_per_space = 18 := 
  by 
    sorry

end NUMINAMATH_GPT_distance_between_trees_l584_58440

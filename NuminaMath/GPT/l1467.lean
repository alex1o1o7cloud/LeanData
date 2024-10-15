import Mathlib

namespace NUMINAMATH_GPT_find_ellipse_eq_product_of_tangent_slopes_l1467_146767

variables {a b : ℝ} {x y x0 y0 : ℝ}

-- Given conditions
def ellipse (a b : ℝ) := a > 0 ∧ b > 0 ∧ a > b ∧ (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 → y = 1 ∧ y = 3 / 2)

def eccentricity (a b : ℝ) := b = (1 / 2) * a

def passes_through (x y : ℝ) := x = 1 ∧ y = 3 / 2

-- Part 1: Prove the equation of the ellipse
theorem find_ellipse_eq (a b : ℝ) (h_ellipse : ellipse a b) (h_eccentricity : eccentricity a b) (h_point : passes_through 1 (3/2)) :
    (x^2) / 4 + (y^2) / 3 = 1 :=
sorry

-- Circle equation definition
def circle (x y : ℝ) := x^2 + y^2 = 7

-- Part 2: Prove the product of the slopes of the tangent lines is constant
theorem product_of_tangent_slopes (P : ℝ × ℝ) (h_circle : circle P.1 P.2) : 
    ∀ k1 k2 : ℝ, (4 - P.1^2) * k1^2 + 6 * P.1 * P.2 * k1 + 3 - P.2^2 = 0 → 
    (4 - P.1^2) * k2^2 + 6 * P.1 * P.2 * k2 + 3 - P.2^2 = 0 → k1 * k2 = -1 :=
sorry

end NUMINAMATH_GPT_find_ellipse_eq_product_of_tangent_slopes_l1467_146767


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_decreasing_l1467_146790

def is_decreasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ x ∈ I, ∀ y ∈ I, x ≤ y → f y ≤ f x

def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 6 * m * x + 6

theorem sufficient_but_not_necessary_decreasing (m : ℝ) :
  m = 1 → is_decreasing_on (f m) (Set.Iic 3) :=
by
  intros h
  rw [h]
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_decreasing_l1467_146790


namespace NUMINAMATH_GPT_line_through_point_parallel_l1467_146786

/-
Given the point P(2, 0) and a line x - 2y + 3 = 0,
prove that the equation of the line passing through 
P and parallel to the given line is 2y - x + 2 = 0.
-/
theorem line_through_point_parallel
  (P : ℝ × ℝ)
  (x y : ℝ)
  (line_eq : x - 2*y + 3 = 0)
  (P_eq : P = (2, 0)) :
  ∃ (a b c : ℝ), a * y - b * x + c = 0 :=
sorry

end NUMINAMATH_GPT_line_through_point_parallel_l1467_146786


namespace NUMINAMATH_GPT_number_of_girls_in_school_l1467_146734

theorem number_of_girls_in_school
  (total_students : ℕ)
  (avg_age_boys avg_age_girls avg_age_school : ℝ)
  (B G : ℕ)
  (h1 : total_students = 640)
  (h2 : avg_age_boys = 12)
  (h3 : avg_age_girls = 11)
  (h4 : avg_age_school = 11.75)
  (h5 : B + G = total_students)
  (h6 : (avg_age_boys * B + avg_age_girls * G = avg_age_school * total_students)) :
  G = 160 :=
by
  sorry

end NUMINAMATH_GPT_number_of_girls_in_school_l1467_146734


namespace NUMINAMATH_GPT_find_a_l1467_146764

theorem find_a (a : ℝ) (h : 3 ∈ ({1, a, a - 2} : Set ℝ)) : a = 5 :=
sorry

end NUMINAMATH_GPT_find_a_l1467_146764


namespace NUMINAMATH_GPT_rectangle_area_l1467_146752

theorem rectangle_area 
  (length_to_width_ratio : Real) 
  (width : Real) 
  (area : Real) 
  (h1 : length_to_width_ratio = 0.875) 
  (h2 : width = 24) 
  (h_area : area = 504) : 
  True := 
sorry

end NUMINAMATH_GPT_rectangle_area_l1467_146752


namespace NUMINAMATH_GPT_negation_of_p_l1467_146758

def f (a x : ℝ) : ℝ := a * x - x - a

theorem negation_of_p :
  (¬ ∀ a > 0, a ≠ 1 → ∃ x : ℝ, f a x = 0) ↔ (∃ a > 0, a ≠ 1 ∧ ¬ ∃ x : ℝ, f a x = 0) :=
by {
  sorry
}

end NUMINAMATH_GPT_negation_of_p_l1467_146758


namespace NUMINAMATH_GPT_compare_fractions_l1467_146728

theorem compare_fractions : (-8 / 21: ℝ) > (-3 / 7: ℝ) :=
sorry

end NUMINAMATH_GPT_compare_fractions_l1467_146728


namespace NUMINAMATH_GPT_find_judes_age_l1467_146717

def jude_age (H : ℕ) (J : ℕ) : Prop :=
  H + 5 = 3 * (J + 5)

theorem find_judes_age : ∃ J : ℕ, jude_age 16 J ∧ J = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_judes_age_l1467_146717


namespace NUMINAMATH_GPT_smallest_class_size_l1467_146722

/--
In a science class, students are separated into five rows for an experiment. 
The class size must be greater than 50. 
Three rows have the same number of students, one row has two more students than the others, 
and another row has three more students than the others.
Prove that the smallest possible class size for this science class is 55.
-/
theorem smallest_class_size (class_size : ℕ) (n : ℕ) 
  (h1 : class_size = 3 * n + (n + 2) + (n + 3))
  (h2 : class_size > 50) :
  class_size = 55 :=
sorry

end NUMINAMATH_GPT_smallest_class_size_l1467_146722


namespace NUMINAMATH_GPT_puppies_brought_in_l1467_146713

open Nat

theorem puppies_brought_in (orig_puppies adopt_rate days total_adopted brought_in_puppies : ℕ) 
  (h_orig : orig_puppies = 3)
  (h_adopt_rate : adopt_rate = 3)
  (h_days : days = 2)
  (h_total_adopted : total_adopted = adopt_rate * days)
  (h_equation : total_adopted = orig_puppies + brought_in_puppies) :
  brought_in_puppies = 3 :=
by
  sorry

end NUMINAMATH_GPT_puppies_brought_in_l1467_146713


namespace NUMINAMATH_GPT_range_of_set_is_8_l1467_146740

theorem range_of_set_is_8 (a b c : ℕ) 
  (h1 : (a + b + c) / 3 = 6) 
  (h2 : b = 6) 
  (h3 : a = 2) 
  : max a (max b c) - min a (min b c) = 8 := 
by sorry

end NUMINAMATH_GPT_range_of_set_is_8_l1467_146740


namespace NUMINAMATH_GPT_find_A_l1467_146794

variable (A B x : ℝ)
variable (hB : B ≠ 0)
variable (h : f (g 2) = 0)
def f := λ x => A * x^3 - B
def g := λ x => B * x^2

theorem find_A (hB : B ≠ 0) (h : (λ x => A * x^3 - B) ((λ x => B * x^2) 2) = 0) : 
  A = 1 / (64 * B^2) :=
  sorry

end NUMINAMATH_GPT_find_A_l1467_146794


namespace NUMINAMATH_GPT_eighth_hexagonal_number_l1467_146750

theorem eighth_hexagonal_number : (8 * (2 * 8 - 1)) = 120 :=
  by
  sorry

end NUMINAMATH_GPT_eighth_hexagonal_number_l1467_146750


namespace NUMINAMATH_GPT_initial_distance_from_lens_l1467_146749

def focal_length := 150 -- focal length F in cm
def screen_shift := 40  -- screen moved by 40 cm

theorem initial_distance_from_lens (d : ℝ) (f : ℝ) (s : ℝ) 
  (h_focal_length : f = focal_length) 
  (h_screen_shift : s = screen_shift) 
  (h_parallel_beam : d = f / 2 ∨ d = 3 * f / 2) : 
  d = 130 ∨ d = 170 := 
by 
  sorry

end NUMINAMATH_GPT_initial_distance_from_lens_l1467_146749


namespace NUMINAMATH_GPT_rotated_line_x_intercept_l1467_146778

theorem rotated_line_x_intercept (x y : ℝ) :
  (∃ (k : ℝ), y = (3 * Real.sqrt 3 + 5) / (2 * Real.sqrt 3) * x) →
  (∃ y : ℝ, 3 * x - 5 * y + 40 = 0) →
  (∃ (x_intercept : ℝ), x_intercept = 0) := 
by
  sorry

end NUMINAMATH_GPT_rotated_line_x_intercept_l1467_146778


namespace NUMINAMATH_GPT_quadratic_single_root_a_l1467_146705

theorem quadratic_single_root_a (a : ℝ) : 
  (∃! x : ℝ, a * x^2 + 2 * x + 1 = 0) → (a = 0 ∨ a = 1) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_single_root_a_l1467_146705


namespace NUMINAMATH_GPT_area_of_EFGH_l1467_146708

variables (EF FG EH HG EG : ℝ)
variables (distEFGH : EF ≠ HG ∧ EG = 5 ∧ EF^2 + FG^2 = 25 ∧ EH^2 + HG^2 = 25)

theorem area_of_EFGH : 
  ∃ EF FG EH HG : ℕ, EF ≠ HG ∧ EG = 5 
  ∧ EF^2 + FG^2 = 25 
  ∧ EH^2 + HG^2 = 25 
  ∧ EF * FG / 2 + EH * HG / 2 = 12 :=
by { sorry }

end NUMINAMATH_GPT_area_of_EFGH_l1467_146708


namespace NUMINAMATH_GPT_solve_inequality_l1467_146742

variables (a b c x α β : ℝ)

theorem solve_inequality 
  (h1 : ∀ x, a * x^2 + b * x + c > 0 ↔ α < x ∧ x < β)
  (h2 : β > α)
  (ha : a < 0)
  (h3 : α + β = -b / a)
  (h4 : α * β = c / a) :
  ∀ x, (c * x^2 + b * x + a < 0 ↔ x < 1 / β ∨ x > 1 / α) := 
  by
    -- A detailed proof would follow here.
    sorry

end NUMINAMATH_GPT_solve_inequality_l1467_146742


namespace NUMINAMATH_GPT_find_least_number_l1467_146766

theorem find_least_number (x : ℕ) :
  (∀ k, 24 ∣ k + 7 → 32 ∣ k + 7 → 36 ∣ k + 7 → 54 ∣ k + 7 → x = k) → 
  x + 7 = Nat.lcm (Nat.lcm (Nat.lcm 24 32) 36) 54 → x = 857 :=
by
  sorry

end NUMINAMATH_GPT_find_least_number_l1467_146766


namespace NUMINAMATH_GPT_students_second_scenario_l1467_146773

def total_students (R : ℕ) : ℕ := 5 * R + 6
def effective_students (R : ℕ) : ℕ := 6 * (R - 3)
def filled_rows (R : ℕ) : ℕ := R - 3
def students_per_row := 6

theorem students_second_scenario:
  ∀ (R : ℕ), R = 24 → total_students R = effective_students R → students_per_row = 6
:= by
  intro R h_eq h_total_eq_effective
  -- Insert proof steps here
  sorry

end NUMINAMATH_GPT_students_second_scenario_l1467_146773


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1467_146744

theorem solution_set_of_inequality:
  {x : ℝ | x^2 - |x-1| - 1 ≤ 0} = {x : ℝ | -2 ≤ x ∧ x ≤ 1} :=
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1467_146744


namespace NUMINAMATH_GPT_find_a_l1467_146756

theorem find_a (α β : ℝ) (h1 : α + β = 10) (h2 : α * β = 20) : (1 / α + 1 / β) = 1 / 2 :=
sorry

end NUMINAMATH_GPT_find_a_l1467_146756


namespace NUMINAMATH_GPT_converse_l1467_146714

variables {x : ℝ}

def P (x : ℝ) : Prop := x < 0
def Q (x : ℝ) : Prop := x^2 > 0

theorem converse (h : Q x) : P x :=
sorry

end NUMINAMATH_GPT_converse_l1467_146714


namespace NUMINAMATH_GPT_loan_difference_is_979_l1467_146770

noncomputable def compounded_interest (P r : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

noncomputable def simple_interest (P r : ℝ) (t : ℕ) : ℝ :=
  P * (1 + r * t)

noncomputable def loan_difference (P : ℝ) : ℝ :=
  let compounded_7_years := compounded_interest P 0.08 12 7
  let half_payment := compounded_7_years / 2
  let remaining_balance := compounded_interest half_payment 0.08 12 8
  let total_compounded := half_payment + remaining_balance
  let total_simple := simple_interest P 0.10 15
  abs (total_compounded - total_simple)

theorem loan_difference_is_979 : loan_difference 15000 = 979 := sorry

end NUMINAMATH_GPT_loan_difference_is_979_l1467_146770


namespace NUMINAMATH_GPT_M_gt_N_l1467_146754

-- Define the variables and conditions
variables (a : ℝ)
def M : ℝ := 5 * a^2 - a + 1
def N : ℝ := 4 * a^2 + a - 1

-- Statement to prove
theorem M_gt_N : M a > N a := by
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_M_gt_N_l1467_146754


namespace NUMINAMATH_GPT_solve_system_of_inequalities_l1467_146718

theorem solve_system_of_inequalities (x : ℝ) :
  (x + 1 < 5) ∧ (2 * x - 1) / 3 ≥ 1 ↔ 2 ≤ x ∧ x < 4 :=
by
  sorry

end NUMINAMATH_GPT_solve_system_of_inequalities_l1467_146718


namespace NUMINAMATH_GPT_sum_of_corners_10x10_l1467_146775

theorem sum_of_corners_10x10 : 
  let top_left := 1
  let top_right := 10
  let bottom_left := 91
  let bottom_right := 100
  (top_left + top_right + bottom_left + bottom_right) = 202 :=
by
  let top_left := 1
  let top_right := 10
  let bottom_left := 91
  let bottom_right := 100
  show top_left + top_right + bottom_left + bottom_right = 202
  sorry

end NUMINAMATH_GPT_sum_of_corners_10x10_l1467_146775


namespace NUMINAMATH_GPT_maximum_value_a3_b3_c3_d3_l1467_146789

noncomputable def max_value (a b c d : ℝ) : ℝ :=
  a^3 + b^3 + c^3 + d^3

theorem maximum_value_a3_b3_c3_d3
  (a b c d : ℝ)
  (h1 : a^2 + b^2 + c^2 + d^2 = 20)
  (h2 : a + b + c + d = 10) :
  max_value a b c d ≤ 500 :=
sorry

end NUMINAMATH_GPT_maximum_value_a3_b3_c3_d3_l1467_146789


namespace NUMINAMATH_GPT_gcd_840_1764_l1467_146795

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end NUMINAMATH_GPT_gcd_840_1764_l1467_146795


namespace NUMINAMATH_GPT_work_problem_l1467_146709

theorem work_problem (days_B : ℝ) (h : (1 / 20) + (1 / days_B) = 1 / 8.571428571428571) : days_B = 15 :=
sorry

end NUMINAMATH_GPT_work_problem_l1467_146709


namespace NUMINAMATH_GPT_biology_marks_l1467_146720

theorem biology_marks (english : ℕ) (math : ℕ) (physics : ℕ) (chemistry : ℕ) (average : ℕ) (biology : ℕ) 
  (h1 : english = 36) 
  (h2 : math = 35) 
  (h3 : physics = 42) 
  (h4 : chemistry = 57) 
  (h5 : average = 45) 
  (h6 : (english + math + physics + chemistry + biology) / 5 = average) : 
  biology = 55 := 
by
  sorry

end NUMINAMATH_GPT_biology_marks_l1467_146720


namespace NUMINAMATH_GPT_blue_chairs_fewer_than_yellow_l1467_146799

theorem blue_chairs_fewer_than_yellow :
  ∀ (red_chairs yellow_chairs chairs_left total_chairs blue_chairs : ℕ),
    red_chairs = 4 →
    yellow_chairs = 2 * red_chairs →
    chairs_left = 15 →
    total_chairs = chairs_left + 3 →
    blue_chairs = total_chairs - (red_chairs + yellow_chairs) →
    yellow_chairs - blue_chairs = 2 :=
by sorry

end NUMINAMATH_GPT_blue_chairs_fewer_than_yellow_l1467_146799


namespace NUMINAMATH_GPT_variance_of_given_data_is_2_l1467_146745

-- Define the data set
def data_set : List ℕ := [198, 199, 200, 201, 202]

-- Define the mean function for a given data set
noncomputable def mean (data : List ℕ) : ℝ :=
  (data.sum : ℝ) / data.length

-- Define the variance function for a given data set
noncomputable def variance (data : List ℕ) : ℝ :=
  let μ := mean data
  (data.map (λ x => (x : ℝ) - μ) |>.map (λ x => x^2)).sum / data.length

-- Proposition that the variance of the given data set is 2
theorem variance_of_given_data_is_2 : variance data_set = 2 := by
  sorry

end NUMINAMATH_GPT_variance_of_given_data_is_2_l1467_146745


namespace NUMINAMATH_GPT_pay_nineteen_rubles_l1467_146774

/-- 
Given a purchase cost of 19 rubles, a customer with only three-ruble bills, 
and a cashier with only five-ruble bills, both having 15 bills each,
prove that it is possible for the customer to pay exactly 19 rubles.
-/
theorem pay_nineteen_rubles (purchase_cost : ℕ) (customer_bills cashier_bills : ℕ) 
  (customer_denomination cashier_denomination : ℕ) (customer_count cashier_count : ℕ) :
  purchase_cost = 19 →
  customer_denomination = 3 →
  cashier_denomination = 5 →
  customer_count = 15 →
  cashier_count = 15 →
  (∃ m n : ℕ, m * customer_denomination - n * cashier_denomination = purchase_cost 
  ∧ m ≤ customer_count ∧ n ≤ cashier_count) :=
by
  intros
  sorry

end NUMINAMATH_GPT_pay_nineteen_rubles_l1467_146774


namespace NUMINAMATH_GPT_christine_wander_time_l1467_146765

-- Definitions based on conditions
def distance : ℝ := 50.0
def speed : ℝ := 6.0

-- The statement to prove
theorem christine_wander_time : (distance / speed) = 8 + 20/60 :=
by
  sorry

end NUMINAMATH_GPT_christine_wander_time_l1467_146765


namespace NUMINAMATH_GPT_admission_charge_l1467_146727

variable (A : ℝ) -- Admission charge in dollars
variable (tour_charge : ℝ)
variable (group1_size : ℕ)
variable (group2_size : ℕ)
variable (total_earnings : ℝ)

-- Given conditions
axiom h1 : tour_charge = 6
axiom h2 : group1_size = 10
axiom h3 : group2_size = 5
axiom h4 : total_earnings = 240
axiom h5 : (group1_size * A + group1_size * tour_charge) + (group2_size * A) = total_earnings

theorem admission_charge : A = 12 :=
by
  sorry

end NUMINAMATH_GPT_admission_charge_l1467_146727


namespace NUMINAMATH_GPT_inverse_function_of_f_l1467_146735

noncomputable def f (x : ℝ) : ℝ := (3 * x + 1) / x
noncomputable def f_inv (x : ℝ) : ℝ := 1 / (x - 3)

theorem inverse_function_of_f:
  ∀ x : ℝ, x ≠ 3 → f (f_inv x) = x ∧ f_inv (f x) = x := by
sorry

end NUMINAMATH_GPT_inverse_function_of_f_l1467_146735


namespace NUMINAMATH_GPT_burn_time_for_structure_l1467_146782

noncomputable def time_to_burn_structure (total_toothpicks : ℕ) (burn_time_per_toothpick : ℕ) (adjacent_corners : Bool) : ℕ :=
  if total_toothpicks = 38 ∧ burn_time_per_toothpick = 10 ∧ adjacent_corners = true then 65 else 0

theorem burn_time_for_structure :
  time_to_burn_structure 38 10 true = 65 :=
sorry

end NUMINAMATH_GPT_burn_time_for_structure_l1467_146782


namespace NUMINAMATH_GPT_find_k_l1467_146796

theorem find_k (k : ℝ) (h1 : k > 0) (h2 : |3 * (k^2 - 9) - 2 * (4 * k - 15) + 2 * (12 - 5 * k)| = 20) : k = 4 := by
  sorry

end NUMINAMATH_GPT_find_k_l1467_146796


namespace NUMINAMATH_GPT_stewart_farm_sheep_l1467_146760

theorem stewart_farm_sheep (S H : ℕ)
  (h1 : S / H = 2 / 7)
  (h2 : H * 230 = 12880) :
  S = 16 :=
by sorry

end NUMINAMATH_GPT_stewart_farm_sheep_l1467_146760


namespace NUMINAMATH_GPT_different_prime_factors_mn_is_five_l1467_146739

theorem different_prime_factors_mn_is_five {m n : ℕ} 
  (m_prime_factors : ∃ (p_1 p_2 p_3 p_4 : ℕ), True)  -- m has 4 different prime factors
  (n_prime_factors : ∃ (q_1 q_2 q_3 : ℕ), True)  -- n has 3 different prime factors
  (gcd_m_n : Nat.gcd m n = 15) : 
  (∃ k : ℕ, k = 5 ∧ (∃ (x_1 x_2 x_3 x_4 x_5 : ℕ), True)) := sorry

end NUMINAMATH_GPT_different_prime_factors_mn_is_five_l1467_146739


namespace NUMINAMATH_GPT_sum_of_first_3n_terms_l1467_146793

def arithmetic_geometric_sequence (n : ℕ) (s : ℕ → ℕ) :=
  (s n = 10) ∧ (s (2 * n) = 30)

theorem sum_of_first_3n_terms (n : ℕ) (s : ℕ → ℕ) :
  arithmetic_geometric_sequence n s → s (3 * n) = 70 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_sum_of_first_3n_terms_l1467_146793


namespace NUMINAMATH_GPT_base_conversion_l1467_146704

noncomputable def b_value : ℝ := Real.sqrt 21

theorem base_conversion (b : ℝ) (h : b = Real.sqrt 21) : 
  (1 * b^2 + 0 * b + 2) = 23 := 
by
  rw [h]
  sorry

end NUMINAMATH_GPT_base_conversion_l1467_146704


namespace NUMINAMATH_GPT_time_for_A_l1467_146788

theorem time_for_A (A B C : ℝ) 
  (h1 : 1/B + 1/C = 1/3) 
  (h2 : 1/A + 1/C = 1/2) 
  (h3 : 1/B = 1/30) : 
  A = 5/2 := 
by
  sorry

end NUMINAMATH_GPT_time_for_A_l1467_146788


namespace NUMINAMATH_GPT_solution_set_g_lt_6_range_of_values_a_l1467_146719

-- Definitions
def f (a x : ℝ) : ℝ := 3 * |x - a| + |3 * x + 1|
def g (x : ℝ) : ℝ := |4 * x - 1| - |x + 2|

-- First part: solution set for g(x) < 6
theorem solution_set_g_lt_6 :
  {x : ℝ | g x < 6} = {x : ℝ | -7/5 < x ∧ x < 3} :=
sorry

-- Second part: range of values for a such that f(x1) and g(x2) are opposite numbers
theorem range_of_values_a (a : ℝ) :
  (∃ x1 x2 : ℝ, f a x1 = -g x2) → -13/12 ≤ a ∧ a ≤ 5/12 :=
sorry

end NUMINAMATH_GPT_solution_set_g_lt_6_range_of_values_a_l1467_146719


namespace NUMINAMATH_GPT_part_I_part_II_l1467_146741

noncomputable def f (a x : ℝ) : ℝ := |a * x - 1| + |x + 2|

theorem part_I (h₁ : ∀ x : ℝ, f 1 x ≤ 5 ↔ -3 ≤ x ∧ x ≤ 2) : True :=
by sorry

theorem part_II (h₂ : ∃ a : ℝ, a > 0 ∧ (∀ x, f a x ≥ 2) ∧ (∀ b : ℝ, b > 0 ∧ (∀ x, f b x ≥ 2) → a ≤ b) ) : True :=
by sorry

end NUMINAMATH_GPT_part_I_part_II_l1467_146741


namespace NUMINAMATH_GPT_jack_marbles_l1467_146738

theorem jack_marbles (initial_marbles share_marbles : ℕ) (h_initial : initial_marbles = 62) (h_share : share_marbles = 33) : 
  initial_marbles - share_marbles = 29 :=
by 
  sorry

end NUMINAMATH_GPT_jack_marbles_l1467_146738


namespace NUMINAMATH_GPT_apples_ratio_l1467_146703

theorem apples_ratio (initial_apples rickis_apples end_apples samsons_apples : ℕ)
(h_initial : initial_apples = 74)
(h_ricki : rickis_apples = 14)
(h_end : end_apples = 32)
(h_samson : initial_apples - rickis_apples - end_apples = samsons_apples) :
  samsons_apples / Nat.gcd samsons_apples rickis_apples = 2 ∧ rickis_apples / Nat.gcd samsons_apples rickis_apples = 1 :=
by
  sorry

end NUMINAMATH_GPT_apples_ratio_l1467_146703


namespace NUMINAMATH_GPT_T_5_value_l1467_146702

noncomputable def T (y : ℝ) (m : ℕ) : ℝ := y^m + (1 / y)^m

theorem T_5_value (y : ℝ) (h : y + 1 / y = 5) : T y 5 = 2525 := 
by {
  sorry
}

end NUMINAMATH_GPT_T_5_value_l1467_146702


namespace NUMINAMATH_GPT_girl_name_correct_l1467_146733

-- The Russian alphabet positions as a Lean list
def russianAlphabet : List (ℕ × Char) := [(1, 'А'), (2, 'Б'), (3, 'В'), (4, 'Г'), (5, 'Д'), (6, 'Е'), (7, 'Ё'), 
                                           (8, 'Ж'), (9, 'З'), (10, 'И'), (11, 'Й'), (12, 'К'), (13, 'Л'), 
                                           (14, 'М'), (15, 'Н'), (16, 'О'), (17, 'П'), (18, 'Р'), (19, 'С'), 
                                           (20, 'Т'), (21, 'У'), (22, 'Ф'), (23, 'Х'), (24, 'Ц'), (25, 'Ч'), 
                                           (26, 'Ш'), (27, 'Щ'), (28, 'Ъ'), (29, 'Ы'), (30, 'Ь'), (31, 'Э'), 
                                           (32, 'Ю'), (33, 'Я')]

-- The sequence of numbers representing the girl's name
def nameSequence : ℕ := 2011533

-- The corresponding name derived from the sequence
def derivedName : String := "ТАНЯ"

-- The equivalence proof statement
theorem girl_name_correct : 
  (nameSequence = 2011533 → derivedName = "ТАНЯ") :=
by
  intro h
  sorry

end NUMINAMATH_GPT_girl_name_correct_l1467_146733


namespace NUMINAMATH_GPT_twelfth_term_geometric_sequence_l1467_146729

theorem twelfth_term_geometric_sequence :
  let a1 := 5
  let r := (2 / 5 : ℝ)
  (a1 * r ^ 11) = (10240 / 48828125 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_twelfth_term_geometric_sequence_l1467_146729


namespace NUMINAMATH_GPT_diameter_of_circular_ground_l1467_146771

noncomputable def radius_of_garden_condition (area_garden : ℝ) (broad_garden : ℝ) : ℝ :=
  let pi_val := Real.pi
  (area_garden / pi_val - broad_garden * broad_garden) / (2 * broad_garden)

-- Given conditions
variable (area_garden : ℝ := 226.19467105846502)
variable (broad_garden : ℝ := 2)

-- Goal to prove: diameter of the circular ground is 34 metres
theorem diameter_of_circular_ground : 2 * radius_of_garden_condition area_garden broad_garden = 34 :=
  sorry

end NUMINAMATH_GPT_diameter_of_circular_ground_l1467_146771


namespace NUMINAMATH_GPT_identity_of_brothers_l1467_146716

theorem identity_of_brothers
  (first_brother_speaks : Prop)
  (second_brother_speaks : Prop)
  (one_tells_truth : first_brother_speaks → ¬ second_brother_speaks)
  (other_tells_truth : ¬first_brother_speaks → second_brother_speaks) :
  first_brother_speaks = false ∧ second_brother_speaks = true :=
by
  sorry

end NUMINAMATH_GPT_identity_of_brothers_l1467_146716


namespace NUMINAMATH_GPT_part1_part2_l1467_146779

-- Definitions
def p (t : ℝ) := ∀ x : ℝ, x^2 + 2 * x + 2 * t - 4 ≠ 0
def q (t : ℝ) := (4 - t > 0) ∧ (t - 2 > 0)

-- Theorem statements
theorem part1 (t : ℝ) (hp : p t) : t > 5 / 2 := sorry

theorem part2 (t : ℝ) (h : p t ∨ q t) (h_and : ¬ (p t ∧ q t)) : (2 < t ∧ t ≤ 5 / 2) ∨ (t ≥ 3) := sorry

end NUMINAMATH_GPT_part1_part2_l1467_146779


namespace NUMINAMATH_GPT_average_monthly_growth_rate_l1467_146753

-- Define the initial and final production quantities
def initial_production : ℝ := 100
def final_production : ℝ := 144

-- Define the average monthly growth rate
def avg_monthly_growth_rate (x : ℝ) : Prop :=
  initial_production * (1 + x)^2 = final_production

-- Statement of the problem to be verified
theorem average_monthly_growth_rate :
  ∃ x : ℝ, avg_monthly_growth_rate x ∧ x = 0.2 :=
by
  sorry

end NUMINAMATH_GPT_average_monthly_growth_rate_l1467_146753


namespace NUMINAMATH_GPT_juice_cost_l1467_146761

theorem juice_cost (J : ℝ) (h1 : 15 * 3 + 25 * 1 + 12 * J = 88) : J = 1.5 :=
by
  sorry

end NUMINAMATH_GPT_juice_cost_l1467_146761


namespace NUMINAMATH_GPT_find_valid_tax_range_l1467_146776

noncomputable def valid_tax_range (t : ℝ) : Prop :=
  let initial_consumption := 200000
  let price_per_cubic_meter := 240
  let consumption_reduction := 2.5 * t * 10^4
  let tax_revenue := (initial_consumption - consumption_reduction) * price_per_cubic_meter * (t / 100)
  tax_revenue >= 900000

theorem find_valid_tax_range (t : ℝ) : 3 ≤ t ∧ t ≤ 5 ↔ valid_tax_range t :=
sorry

end NUMINAMATH_GPT_find_valid_tax_range_l1467_146776


namespace NUMINAMATH_GPT_simplify_powers_of_ten_l1467_146725

theorem simplify_powers_of_ten :
  (10^0.4) * (10^0.5) * (10^0.2) * (10^(-0.6)) * (10^0.5) = 10 := 
by
  sorry

end NUMINAMATH_GPT_simplify_powers_of_ten_l1467_146725


namespace NUMINAMATH_GPT_john_less_than_anna_l1467_146798

theorem john_less_than_anna (J A L T : ℕ) (h1 : A = 50) (h2: L = 3) (h3: T = 82) (h4: T + L = A + J) : A - J = 15 :=
by
  sorry

end NUMINAMATH_GPT_john_less_than_anna_l1467_146798


namespace NUMINAMATH_GPT_circle_center_coordinates_l1467_146757

open Real

noncomputable def circle_center (x y : Real) : Prop := 
  x^2 + y^2 - 4*x + 6*y = 0

theorem circle_center_coordinates :
  ∃ (a b : Real), circle_center a b ∧ a = 2 ∧ b = -3 :=
by
  use 2, -3
  sorry

end NUMINAMATH_GPT_circle_center_coordinates_l1467_146757


namespace NUMINAMATH_GPT_grayson_travels_further_l1467_146759

noncomputable def grayson_first_part_distance : ℝ := 25 * 1
noncomputable def grayson_second_part_distance : ℝ := 20 * 0.5
noncomputable def total_distance_grayson : ℝ := grayson_first_part_distance + grayson_second_part_distance

noncomputable def total_distance_rudy : ℝ := 10 * 3

theorem grayson_travels_further : (total_distance_grayson - total_distance_rudy) = 5 := by
  sorry

end NUMINAMATH_GPT_grayson_travels_further_l1467_146759


namespace NUMINAMATH_GPT_average_next_3_numbers_l1467_146732

theorem average_next_3_numbers 
  (a1 a2 b1 b2 b3 c1 c2 c3 : ℝ)
  (h_avg_total : (a1 + a2 + b1 + b2 + b3 + c1 + c2 + c3) / 8 = 25)
  (h_avg_first2: (a1 + a2) / 2 = 20)
  (h_c1_c2 : c1 + 4 = c2)
  (h_c1_c3 : c1 + 6 = c3)
  (h_c3_value : c3 = 30) :
  (b1 + b2 + b3) / 3 = 26 := 
sorry

end NUMINAMATH_GPT_average_next_3_numbers_l1467_146732


namespace NUMINAMATH_GPT_distance_between_stations_l1467_146736

theorem distance_between_stations
  (time_start_train1 time_meet time_start_train2 : ℕ) -- time in hours (7 a.m., 11 a.m., 8 a.m.)
  (speed_train1 speed_train2 : ℕ) -- speed in kmph (20 kmph, 25 kmph)
  (distance_covered_train1 distance_covered_train2 : ℕ)
  (total_distance : ℕ) :
  time_start_train1 = 7 ∧ time_meet = 11 ∧ time_start_train2 = 8 ∧ speed_train1 = 20 ∧ speed_train2 = 25 ∧
  distance_covered_train1 = (time_meet - time_start_train1) * speed_train1 ∧
  distance_covered_train2 = (time_meet - time_start_train2) * speed_train2 ∧
  total_distance = distance_covered_train1 + distance_covered_train2 →
  total_distance = 155 := by
{
  sorry
}

end NUMINAMATH_GPT_distance_between_stations_l1467_146736


namespace NUMINAMATH_GPT_Jeremy_songs_l1467_146710

theorem Jeremy_songs (songs_yesterday : ℕ) (songs_difference : ℕ) (songs_today : ℕ) (total_songs : ℕ) :
  songs_yesterday = 9 ∧ songs_difference = 5 ∧ songs_today = songs_yesterday + songs_difference ∧ 
  total_songs = songs_yesterday + songs_today → total_songs = 23 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_Jeremy_songs_l1467_146710


namespace NUMINAMATH_GPT_probability_two_boys_l1467_146700

-- Definitions for the conditions
def total_students : ℕ := 4
def boys : ℕ := 3
def girls : ℕ := 1
def select_students : ℕ := 2

-- Combination function definition
def combination (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem probability_two_boys :
  (combination boys select_students) / (combination total_students select_students) = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_probability_two_boys_l1467_146700


namespace NUMINAMATH_GPT_ellipse_equation_l1467_146706

theorem ellipse_equation
  (P : ℝ × ℝ)
  (a b c : ℝ)
  (h1 : a > b ∧ b > 0)
  (h2 : 2 * a = 5 + 3)
  (h3 : (2 * c) ^ 2 = 5 ^ 2 - 3 ^ 2)
  (h4 : P.1 ^ 2 / a ^ 2 + P.2 ^ 2 / b ^ 2 = 1 ∨ P.2 ^ 2 / a ^ 2 + P.1 ^ 2 / b ^ 2 = 1)
  : ((a = 4) ∧ (c = 2) ∧ (b ^ 2 = 12) ∧
    (P.1 ^ 2 / 16 + P.2 ^ 2 / 12 = 1) ∨
    (P.2 ^ 2 / 16 + P.1 ^ 2 / 12 = 1)) :=
sorry

end NUMINAMATH_GPT_ellipse_equation_l1467_146706


namespace NUMINAMATH_GPT_meena_sold_to_stone_l1467_146737

def total_cookies_baked : ℕ := 5 * 12
def cookies_bought_brock : ℕ := 7
def cookies_bought_katy : ℕ := 2 * cookies_bought_brock
def cookies_left : ℕ := 15
def cookies_sold_total : ℕ := total_cookies_baked - cookies_left
def cookies_bought_friends : ℕ := cookies_bought_brock + cookies_bought_katy
def cookies_sold_stone : ℕ := cookies_sold_total - cookies_bought_friends
def dozens_sold_stone : ℕ := cookies_sold_stone / 12

theorem meena_sold_to_stone : dozens_sold_stone = 2 := by
  sorry

end NUMINAMATH_GPT_meena_sold_to_stone_l1467_146737


namespace NUMINAMATH_GPT_find_constants_monotonicity_range_of_k_l1467_146726

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := (b - 2 ^ x) / (2 ^ (x + 1) + a)

theorem find_constants (h_odd : ∀ x : ℝ, f x a b = - f (-x) a b) :
  a = 2 ∧ b = 1 :=
sorry

theorem monotonicity (a : ℝ) (b : ℝ) (h_constants : a = 2 ∧ b = 1) :
  ∀ x y : ℝ, x < y → f y a b ≤ f x a b :=
sorry

theorem range_of_k (a : ℝ) (b : ℝ) (h_constants : a = 2 ∧ b = 1)
  (h_pos : ∀ x : ℝ, x ≥ 1 → f (k * 3^x) a b + f (3^x - 9^x + 2) a b > 0) :
  k < 4 / 3 :=
sorry

end NUMINAMATH_GPT_find_constants_monotonicity_range_of_k_l1467_146726


namespace NUMINAMATH_GPT_simplify_expression_l1467_146731

-- Define the given expressions
def numerator : ℕ := 5^5 + 5^3 + 5
def denominator : ℕ := 5^4 - 2 * 5^2 + 5

-- Define the simplified fraction
def simplified_fraction : ℚ := numerator / denominator

-- Prove that the simplified fraction is equivalent to 651 / 116
theorem simplify_expression : simplified_fraction = 651 / 116 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1467_146731


namespace NUMINAMATH_GPT_pentagon_area_eq_half_l1467_146707

variables {A B C D E : Type*} -- Assume A, B, C, D, E are some points in a plane

-- Assume the given conditions in the problem
variables (angle_A angle_C : ℝ)
variables (AB AE BC CD AC : ℝ)
variables (pentagon_area : ℝ)

-- Assume the constraints from the problem statement
axiom angle_A_eq_90 : angle_A = 90
axiom angle_C_eq_90 : angle_C = 90
axiom AB_eq_AE : AB = AE
axiom BC_eq_CD : BC = CD
axiom AC_eq_1 : AC = 1

theorem pentagon_area_eq_half : pentagon_area = 1 / 2 :=
sorry

end NUMINAMATH_GPT_pentagon_area_eq_half_l1467_146707


namespace NUMINAMATH_GPT_value_of_transformed_product_of_roots_l1467_146755

theorem value_of_transformed_product_of_roots 
  (a b : ℚ)
  (h1 : 3 * a^2 + 4 * a - 7 = 0)
  (h2 : 3 * b^2 + 4 * b - 7 = 0)
  (h3 : a ≠ b) : 
  (a - 2) * (b - 2) = 13 / 3 :=
by
  -- The exact proof would be completed here.
  sorry

end NUMINAMATH_GPT_value_of_transformed_product_of_roots_l1467_146755


namespace NUMINAMATH_GPT_not_all_divisible_by_6_have_prime_neighbors_l1467_146748

theorem not_all_divisible_by_6_have_prime_neighbors :
  ¬ ∀ n : ℕ, (6 ∣ n) → (Prime (n - 1) ∨ Prime (n + 1)) := by
  sorry

end NUMINAMATH_GPT_not_all_divisible_by_6_have_prime_neighbors_l1467_146748


namespace NUMINAMATH_GPT_find_a_l1467_146791

noncomputable def A (a : ℝ) : Set ℝ := {1, 2, a}
noncomputable def B (a : ℝ) : Set ℝ := {1, a^2 - a}

theorem find_a (a : ℝ) : A a ⊇ B a → a = -1 ∨ a = 0 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l1467_146791


namespace NUMINAMATH_GPT_customers_left_tip_l1467_146701

-- Definition of the given conditions
def initial_customers : ℕ := 29
def added_customers : ℕ := 20
def customers_didnt_tip : ℕ := 34

-- Lean 4 statement proving that the number of customers who did leave a tip (answer) equals 15
theorem customers_left_tip : (initial_customers + added_customers - customers_didnt_tip) = 15 :=
by
  sorry

end NUMINAMATH_GPT_customers_left_tip_l1467_146701


namespace NUMINAMATH_GPT_first_complete_row_cover_l1467_146762

def is_shaded_square (n : ℕ) : ℕ := n ^ 2

def row_number (square_number : ℕ) : ℕ :=
  (square_number + 9) / 10 -- ceiling of square_number / 10

theorem first_complete_row_cover : ∃ n, ∀ r : ℕ, 1 ≤ r ∧ r ≤ 10 → ∃ k : ℕ, is_shaded_square k ≤ n ∧ row_number (is_shaded_square k) = r :=
by
  use 100
  intros r h
  sorry

end NUMINAMATH_GPT_first_complete_row_cover_l1467_146762


namespace NUMINAMATH_GPT_solution_set_inequality_l1467_146711

variable (a b c : ℝ)
variable (condition1 : ∀ x : ℝ, ax^2 + bx + c < 0 ↔ x < -1 ∨ 2 < x)

theorem solution_set_inequality (h : a < 0 ∧ b = -a ∧ c = -2 * a) :
  ∀ x : ℝ, (bx^2 + ax - c ≤ 0) ↔ (-1 ≤ x ∧ x ≤ 2) :=
by
  intro x
  sorry

end NUMINAMATH_GPT_solution_set_inequality_l1467_146711


namespace NUMINAMATH_GPT_Vasya_has_larger_amount_l1467_146747

-- Defining the conditions and given data
variables (V P : ℝ)

-- Vasya's profit calculation
def Vasya_profit (V : ℝ) : ℝ := 0.20 * V

-- Petya's profit calculation considering exchange rate increase
def Petya_profit (P : ℝ) : ℝ := 0.2045 * P

-- Proof statement
theorem Vasya_has_larger_amount (h : Vasya_profit V = Petya_profit P) : V > P :=
sorry

end NUMINAMATH_GPT_Vasya_has_larger_amount_l1467_146747


namespace NUMINAMATH_GPT_color_preference_l1467_146783

-- Define the conditions
def total_students := 50
def girls := 30
def boys := 20

def girls_pref_pink := girls / 3
def girls_pref_purple := 2 * girls / 5
def girls_pref_blue := girls - girls_pref_pink - girls_pref_purple

def boys_pref_red := 2 * boys / 5
def boys_pref_green := 3 * boys / 10
def boys_pref_orange := boys - boys_pref_red - boys_pref_green

-- Proof statement
theorem color_preference :
  girls_pref_pink = 10 ∧
  girls_pref_purple = 12 ∧
  girls_pref_blue = 8 ∧
  boys_pref_red = 8 ∧
  boys_pref_green = 6 ∧
  boys_pref_orange = 6 :=
by
  sorry

end NUMINAMATH_GPT_color_preference_l1467_146783


namespace NUMINAMATH_GPT_units_digit_difference_l1467_146746

-- Conditions based on the problem statement
def units_digit_of_power_of_5 (n : ℕ) : ℕ := 5

def units_digit_of_power_of_3 (n : ℕ) : ℕ :=
  match n % 4 with
  | 0     => 1
  | 1     => 3
  | 2     => 9
  | 3     => 7
  | _     => 0  -- impossible due to mod 4

-- Problem statement in Lean as a theorem
theorem units_digit_difference : (5^2019 - 3^2019) % 10 = 8 :=
by
  have h1 : (5^2019 % 10) = units_digit_of_power_of_5 2019 := sorry
  have h2 : (3^2019 % 10) = units_digit_of_power_of_3 2019 := sorry
  -- The core proof step will go here
  sorry

end NUMINAMATH_GPT_units_digit_difference_l1467_146746


namespace NUMINAMATH_GPT_length_first_train_l1467_146769

/-- Let the speeds of two trains be 120 km/hr and 80 km/hr, respectively. 
These trains cross each other in 9 seconds, and the length of the second train is 250.04 meters. 
Prove that the length of the first train is 250 meters. -/
theorem length_first_train
  (FirstTrainSpeed : ℝ := 120)  -- speed of the first train in km/hr
  (SecondTrainSpeed : ℝ := 80)  -- speed of the second train in km/hr
  (TimeToCross : ℝ := 9)        -- time to cross each other in seconds
  (LengthSecondTrain : ℝ := 250.04) -- length of the second train in meters
  : FirstTrainSpeed / 0.36 + SecondTrainSpeed / 0.36 * TimeToCross - LengthSecondTrain = 250 :=
by
  -- omitted proof
  sorry

end NUMINAMATH_GPT_length_first_train_l1467_146769


namespace NUMINAMATH_GPT_manicure_cost_l1467_146777

noncomputable def cost_of_manicure : ℝ := 30

theorem manicure_cost
    (cost_hair_updo : ℝ)
    (total_cost_with_tips : ℝ)
    (tip_rate : ℝ)
    (M : ℝ) :
  cost_hair_updo = 50 →
  total_cost_with_tips = 96 →
  tip_rate = 0.20 →
  (cost_hair_updo + M + tip_rate * cost_hair_updo + tip_rate * M = total_cost_with_tips) →
  M = cost_of_manicure :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_manicure_cost_l1467_146777


namespace NUMINAMATH_GPT_total_company_pay_monthly_l1467_146797

-- Define the given conditions
def hours_josh_works_daily : ℕ := 8
def days_josh_works_weekly : ℕ := 5
def weeks_josh_works_monthly : ℕ := 4
def hourly_rate_josh : ℕ := 9

-- Define Carl's working hours and rate based on the conditions
def hours_carl_works_daily : ℕ := hours_josh_works_daily - 2
def hourly_rate_carl : ℕ := hourly_rate_josh / 2

-- Calculate total hours worked monthly by Josh and Carl
def total_hours_josh_monthly : ℕ := hours_josh_works_daily * days_josh_works_weekly * weeks_josh_works_monthly
def total_hours_carl_monthly : ℕ := hours_carl_works_daily * days_josh_works_weekly * weeks_josh_works_monthly

-- Calculate monthly pay for Josh and Carl
def monthly_pay_josh : ℕ := total_hours_josh_monthly * hourly_rate_josh
def monthly_pay_carl : ℕ := total_hours_carl_monthly * hourly_rate_carl

-- Theorem to prove the total pay for both Josh and Carl in one month
theorem total_company_pay_monthly : monthly_pay_josh + monthly_pay_carl = 1980 := by
  sorry

end NUMINAMATH_GPT_total_company_pay_monthly_l1467_146797


namespace NUMINAMATH_GPT_find_other_divisor_l1467_146715

theorem find_other_divisor (x : ℕ) (h : x ≠ 35) (h1 : 386 % 35 = 1) (h2 : 386 % x = 1) : x = 11 :=
sorry

end NUMINAMATH_GPT_find_other_divisor_l1467_146715


namespace NUMINAMATH_GPT_least_divisor_for_perfect_square_l1467_146787

theorem least_divisor_for_perfect_square : 
  ∃ d : ℕ, (∀ n : ℕ, n > 0 → 16800 / d = n * n) ∧ d = 21 := 
sorry

end NUMINAMATH_GPT_least_divisor_for_perfect_square_l1467_146787


namespace NUMINAMATH_GPT_domain_of_f_l1467_146772

theorem domain_of_f : 
  ∀ x, (2 - x ≥ 0) ∧ (x + 1 > 0) ↔ (-1 < x ∧ x ≤ 2) := by
  sorry

end NUMINAMATH_GPT_domain_of_f_l1467_146772


namespace NUMINAMATH_GPT_sum_possible_values_l1467_146781

theorem sum_possible_values (x : ℤ) (h : ∃ y : ℤ, y = (3 * x + 13) / (x + 6)) :
  ∃ s : ℤ, s = -2 + 8 + 2 + 4 :=
sorry

end NUMINAMATH_GPT_sum_possible_values_l1467_146781


namespace NUMINAMATH_GPT_probability_of_yellow_face_l1467_146730

def total_faces : ℕ := 12
def red_faces : ℕ := 5
def yellow_faces : ℕ := 4
def blue_faces : ℕ := 2
def green_faces : ℕ := 1

theorem probability_of_yellow_face : (yellow_faces : ℚ) / (total_faces : ℚ) = 1 / 3 := by
  sorry

end NUMINAMATH_GPT_probability_of_yellow_face_l1467_146730


namespace NUMINAMATH_GPT_find_triples_solution_l1467_146724

theorem find_triples_solution (x y z : ℕ) (h : x^5 + x^4 + 1 = 3^y * 7^z) :
  (x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1 ∧ y = 1 ∧ z = 0) ∨ (x = 2 ∧ y = 0 ∧ z = 2) :=
by
  sorry

end NUMINAMATH_GPT_find_triples_solution_l1467_146724


namespace NUMINAMATH_GPT_number_of_divisors_of_3003_l1467_146723

theorem number_of_divisors_of_3003 :
  ∃ d, d = 16 ∧ 
  (3003 = 3^1 * 7^1 * 11^1 * 13^1) →
  d = (1 + 1) * (1 + 1) * (1 + 1) * (1 + 1) := 
by 
  sorry

end NUMINAMATH_GPT_number_of_divisors_of_3003_l1467_146723


namespace NUMINAMATH_GPT_large_seat_capacity_l1467_146763

-- Definition of conditions
def num_large_seats : ℕ := 7
def total_capacity_large_seats : ℕ := 84

-- Theorem to prove
theorem large_seat_capacity : total_capacity_large_seats / num_large_seats = 12 :=
by
  sorry

end NUMINAMATH_GPT_large_seat_capacity_l1467_146763


namespace NUMINAMATH_GPT_work_done_together_in_one_day_l1467_146751

-- Defining the conditions
def time_to_finish_a : ℕ := 12
def time_to_finish_b : ℕ := time_to_finish_a / 2

-- Defining the work done in one day
def work_done_by_a_in_one_day : ℚ := 1 / time_to_finish_a
def work_done_by_b_in_one_day : ℚ := 1 / time_to_finish_b

-- The proof statement
theorem work_done_together_in_one_day : 
  work_done_by_a_in_one_day + work_done_by_b_in_one_day = 1 / 4 := by
  sorry

end NUMINAMATH_GPT_work_done_together_in_one_day_l1467_146751


namespace NUMINAMATH_GPT_max_clouds_crossed_by_plane_l1467_146780

-- Define the conditions
def plane_region_divide (num_planes : ℕ) : ℕ :=
  num_planes + 1

-- Hypotheses/Conditions
variable (num_planes : ℕ)
variable (initial_region_clouds : ℕ)
variable (max_crosses : ℕ)

-- The primary statement to be proved
theorem max_clouds_crossed_by_plane : 
  num_planes = 10 → initial_region_clouds = 1 → max_crosses = num_planes + initial_region_clouds →
  max_crosses = 11 := 
by
  -- Placeholder for the actual proof
  intros
  sorry

end NUMINAMATH_GPT_max_clouds_crossed_by_plane_l1467_146780


namespace NUMINAMATH_GPT_no_beverages_l1467_146743

noncomputable def businessmen := 30
def coffee := 15
def tea := 13
def water := 6
def coffee_tea := 7
def tea_water := 3
def coffee_water := 2
def all_three := 1

theorem no_beverages (businessmen coffee tea water coffee_tea tea_water coffee_water all_three):
  businessmen - (coffee + tea + water - coffee_tea - tea_water - coffee_water + all_three) = 7 :=
by sorry

end NUMINAMATH_GPT_no_beverages_l1467_146743


namespace NUMINAMATH_GPT_smallest_value_l1467_146785

theorem smallest_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  ∃ (v : ℝ), (∀ x y : ℝ, 0 < x → 0 < y → v ≤ (16 / x + 108 / y + x * y)) ∧ v = 36 :=
sorry

end NUMINAMATH_GPT_smallest_value_l1467_146785


namespace NUMINAMATH_GPT_solution_set_correct_l1467_146784

theorem solution_set_correct (a b c : ℝ) (h : a < 0) (h1 : ∀ x, (ax^2 + bx + c < 0) ↔ ((x < 1) ∨ (x > 3))) :
  ∀ x, (cx^2 + bx + a > 0) ↔ (1 / 3 < x ∧ x < 1) :=
sorry

end NUMINAMATH_GPT_solution_set_correct_l1467_146784


namespace NUMINAMATH_GPT_max_min_difference_abc_l1467_146712

theorem max_min_difference_abc (a b c : ℝ) (h : a^2 + b^2 + c^2 = 1) :
    let M := 1
    let m := -1/2
    M - m = 3/2 :=
by
  sorry

end NUMINAMATH_GPT_max_min_difference_abc_l1467_146712


namespace NUMINAMATH_GPT_simplify_expression_l1467_146768

noncomputable def proof_problem (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) (h_abc : a * b * c = 1) : Prop :=
  (1 / (1 + a + a * b) + 1 / (1 + b + b * c) + 1 / (1 + c + c * a)) = 1

theorem simplify_expression (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) (h_abc : a * b * c = 1) :
  proof_problem a b c h h_abc :=
by sorry

end NUMINAMATH_GPT_simplify_expression_l1467_146768


namespace NUMINAMATH_GPT_relay_scheme_count_l1467_146792

theorem relay_scheme_count
  (num_segments : ℕ)
  (num_torchbearers : ℕ)
  (first_choices : ℕ)
  (last_choices : ℕ) :
  num_segments = 6 ∧
  num_torchbearers = 6 ∧
  first_choices = 3 ∧
  last_choices = 2 →
  ∃ num_schemes : ℕ, num_schemes = 7776 :=
by
  intro h
  obtain ⟨h_segments, h_torchbearers, h_first_choices, h_last_choices⟩ := h
  exact ⟨7776, sorry⟩

end NUMINAMATH_GPT_relay_scheme_count_l1467_146792


namespace NUMINAMATH_GPT_people_in_each_bus_l1467_146721

-- Definitions and conditions
def num_vans : ℕ := 2
def num_buses : ℕ := 3
def people_per_van : ℕ := 8
def total_people : ℕ := 76

-- Theorem statement to prove the number of people in each bus
theorem people_in_each_bus : (total_people - num_vans * people_per_van) / num_buses = 20 :=
by
    -- The actual proof would go here
    sorry

end NUMINAMATH_GPT_people_in_each_bus_l1467_146721

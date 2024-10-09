import Mathlib

namespace impossible_fifty_pieces_l1268_126811

open Nat

theorem impossible_fifty_pieces :
  ¬ ∃ (m : ℕ), 1 + 3 * m = 50 :=
by
  sorry

end impossible_fifty_pieces_l1268_126811


namespace unique_solution_of_system_l1268_126806

noncomputable def solve_system_of_equations (x1 x2 x3 x4 x5 x6 x7 : ℝ) : Prop :=
  10 * x1 + 3 * x2 + 4 * x3 + x4 + x5 = 0 ∧
  11 * x2 + 2 * x3 + 2 * x4 + 3 * x5 + x6 = 0 ∧
  15 * x3 + 4 * x4 + 5 * x5 + 4 * x6 + x7 = 0 ∧
  2 * x1 + x2 - 3 * x3 + 12 * x4 - 3 * x5 + x6 + x7 = 0 ∧
  6 * x1 - 5 * x2 + 3 * x3 - x4 + 17 * x5 + x6 = 0 ∧
  3 * x1 + 2 * x2 - 3 * x3 + 4 * x4 + x5 - 16 * x6 + 2 * x7 = 0 ∧
  4 * x1 - 8 * x2 + x3 + x4 - 3 * x5 + 19 * x7 = 0

theorem unique_solution_of_system :
  ∀ (x1 x2 x3 x4 x5 x6 x7 : ℝ),
    solve_system_of_equations x1 x2 x3 x4 x5 x6 x7 →
    x1 = 0 ∧ x2 = 0 ∧ x3 = 0 ∧ x4 = 0 ∧ x5 = 0 ∧ x6 = 0 ∧ x7 = 0 :=
by
  intros x1 x2 x3 x4 x5 x6 x7 h
  sorry

end unique_solution_of_system_l1268_126806


namespace average_salary_is_8000_l1268_126843

def average_salary_all_workers (A : ℝ) :=
  let total_workers := 30
  let technicians := 10
  let technician_salary := 12000
  let rest_workers := total_workers - technicians
  let rest_salary := 6000
  let total_salary := (technicians * technician_salary) + (rest_workers * rest_salary)
  A = total_salary / total_workers

theorem average_salary_is_8000 : average_salary_all_workers 8000 :=
by
  sorry

end average_salary_is_8000_l1268_126843


namespace incorrect_inequality_l1268_126851

theorem incorrect_inequality : ¬ (-2 < -3) :=
by {
  -- Proof goes here
  sorry
}

end incorrect_inequality_l1268_126851


namespace larger_square_area_multiple_l1268_126808

theorem larger_square_area_multiple (a b : ℕ) (h : a = 4 * b) :
  (a ^ 2) = 16 * (b ^ 2) :=
sorry

end larger_square_area_multiple_l1268_126808


namespace f_7_eq_neg3_l1268_126809

noncomputable def f : ℝ → ℝ := sorry

axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom periodic_f : ∀ x : ℝ, f (x + 4) = f x
axiom f_interval  : ∀ x : ℝ, 0 < x ∧ x < 2 → f x = -x + 4

theorem f_7_eq_neg3 : f 7 = -3 :=
  sorry

end f_7_eq_neg3_l1268_126809


namespace net_change_correct_l1268_126878
-- Import the necessary library

-- Price calculation function
def price_after_changes (initial_price: ℝ) (changes: List (ℝ -> ℝ)): ℝ :=
  changes.foldl (fun price change => change price) initial_price

-- Define each model's price changes
def modelA_changes: List (ℝ -> ℝ) := [
  fun price => price * 0.9, 
  fun price => price * 1.3, 
  fun price => price * 0.85
]

def modelB_changes: List (ℝ -> ℝ) := [
  fun price => price * 0.85, 
  fun price => price * 1.25, 
  fun price => price * 0.80
]

def modelC_changes: List (ℝ -> ℝ) := [
  fun price => price * 0.80, 
  fun price => price * 1.20, 
  fun price => price * 0.95
]

-- Calculate final prices
def final_price_modelA := price_after_changes 1000 modelA_changes
def final_price_modelB := price_after_changes 1500 modelB_changes
def final_price_modelC := price_after_changes 2000 modelC_changes

-- Calculate net changes
def net_change_modelA := final_price_modelA - 1000
def net_change_modelB := final_price_modelB - 1500
def net_change_modelC := final_price_modelC - 2000

-- Set up theorem
theorem net_change_correct:
  net_change_modelA = -5.5 ∧ net_change_modelB = -225 ∧ net_change_modelC = -176 := by
  -- Proof is skipped
  sorry

end net_change_correct_l1268_126878


namespace expression_simplification_l1268_126877

theorem expression_simplification (a : ℝ) : a * (a + 2) - 2 * a = a ^ 2 :=
by
  sorry

end expression_simplification_l1268_126877


namespace find_corresponding_element_l1268_126844

def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 - p.2, p.1 + p.2)

theorem find_corresponding_element :
  f (-1, 2) = (-3, 1) :=
by
  sorry

end find_corresponding_element_l1268_126844


namespace log_identity_l1268_126865

theorem log_identity : (Real.log 2)^3 + 3 * (Real.log 2) * (Real.log 5) + (Real.log 5)^3 = 1 :=
by
  sorry

end log_identity_l1268_126865


namespace remaining_payment_l1268_126898

theorem remaining_payment (part_payment total_cost : ℝ) (percent_payment : ℝ) 
  (h1 : part_payment = 650) 
  (h2 : percent_payment = 15 / 100) 
  (h3 : part_payment = percent_payment * total_cost) : 
  total_cost - part_payment = 3683.33 := 
by 
  sorry

end remaining_payment_l1268_126898


namespace winnie_keeps_balloons_l1268_126824

theorem winnie_keeps_balloons : 
  let red := 20
  let white := 40
  let green := 70
  let yellow := 90
  let total_balloons := red + white + green + yellow
  let friends := 9
  let remainder := total_balloons % friends
  remainder = 4 :=
by
  let red := 20
  let white := 40
  let green := 70
  let yellow := 90
  let total_balloons := red + white + green + yellow
  let friends := 9
  let remainder := total_balloons % friends
  show remainder = 4
  sorry

end winnie_keeps_balloons_l1268_126824


namespace intersection_A_B_l1268_126893

def A : Set ℝ := {x | 2*x - 1 ≤ 0}
def B : Set ℝ := {x | 0 < x ∧ x < 1}

theorem intersection_A_B : A ∩ B = {x | 0 < x ∧ x ≤ 1/2} := 
by 
  sorry

end intersection_A_B_l1268_126893


namespace line_slope_point_l1268_126834

theorem line_slope_point (m b : ℝ) (h_slope : m = -4) (h_point : ∃ x y : ℝ, (x, y) = (5, 2) ∧ y = m * x + b) : 
  m + b = 18 := by
  sorry

end line_slope_point_l1268_126834


namespace max_rocket_height_l1268_126888

-- Define the quadratic function representing the rocket's height
def rocket_height (t : ℝ) : ℝ := -20 * t^2 + 100 * t + 50

-- State the maximum height problem
theorem max_rocket_height : ∃ t : ℝ, rocket_height t = 175 ∧ ∀ t' : ℝ, rocket_height t' ≤ 175 :=
by
  use 2.5
  sorry -- The proof will show that the maximum height is 175 meters at time t = 2.5 seconds

end max_rocket_height_l1268_126888


namespace find_max_value_l1268_126832

noncomputable def max_value (x y z : ℝ) : ℝ := (x + y) / (x * y * z)

theorem find_max_value (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 2) :
  max_value x y z ≤ 13.5 :=
sorry

end find_max_value_l1268_126832


namespace spotlight_distance_l1268_126828

open Real

-- Definitions for the ellipsoid parameters
def ellipsoid_parameters (a b c : ℝ) : Prop :=
  a^2 = b^2 + c^2 ∧ a - c = 1.5

-- Given conditions as input parameters
variables (a b c : ℝ)
variables (h_a : a = 2.7) -- semi-major axis half length
variables (h_c : c = 1.5) -- focal point distance

-- Prove that the distance from F2 to F1 is 12 cm
theorem spotlight_distance (h : ellipsoid_parameters a b c) : 2 * a - (a - c) = 12 :=
by sorry

end spotlight_distance_l1268_126828


namespace range_of_a_l1268_126858

-- Definitions
def p (x a : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def q (x : ℝ) : Prop := x^2 + 2 * x - 8 > 0

-- Main theorem to prove
theorem range_of_a (a : ℝ) (h : a < 0)
  (h_necessary : ∀ x, ¬ p x a → ¬ q x) 
  (h_not_sufficient : ∃ x, ¬ p x a ∧ q x): 
  a ≤ -4 :=
sorry

end range_of_a_l1268_126858


namespace percentage_decrease_l1268_126850

theorem percentage_decrease (x y : ℝ) :
  let x' := 0.8 * x
  let y' := 0.7 * y
  let original_expr := x^2 * y^3
  let new_expr := (x')^2 * (y')^3
  let perc_decrease := (original_expr - new_expr) / original_expr * 100
  perc_decrease = 78.048 := by
  sorry

end percentage_decrease_l1268_126850


namespace contrapositive_l1268_126826

theorem contrapositive (m : ℝ) :
  (∀ m > 0, ∃ x : ℝ, x^2 + x - m = 0) ↔ (∀ m ≤ 0, ∀ x : ℝ, x^2 + x - m ≠ 0) := by
  sorry

end contrapositive_l1268_126826


namespace train_speed_54_kmh_l1268_126846

theorem train_speed_54_kmh
  (train_length : ℕ)
  (tunnel_length : ℕ)
  (time_seconds : ℕ)
  (total_distance : ℕ := train_length + tunnel_length)
  (speed_mps : ℚ := total_distance / time_seconds)
  (conversion_factor : ℚ := 3.6) :
  train_length = 300 →
  tunnel_length = 1200 →
  time_seconds = 100 →
  speed_mps * conversion_factor = 54 := 
by
  intros h_train_length h_tunnel_length h_time_seconds
  sorry

end train_speed_54_kmh_l1268_126846


namespace det_dilation_matrix_l1268_126841

def E : Matrix (Fin 2) (Fin 2) ℝ := ![![12, 0], ![0, 12]]

theorem det_dilation_matrix : Matrix.det E = 144 := by
  sorry

end det_dilation_matrix_l1268_126841


namespace cubic_polynomial_solution_l1268_126867

theorem cubic_polynomial_solution 
  (p : ℚ → ℚ) 
  (h1 : p 1 = 1)
  (h2 : p 2 = 1 / 4)
  (h3 : p 3 = 1 / 9)
  (h4 : p 4 = 1 / 16)
  (h6 : p 6 = 1 / 36)
  (h0 : p 0 = -1 / 25) : 
  p 5 = 20668 / 216000 :=
sorry

end cubic_polynomial_solution_l1268_126867


namespace domain_of_f_of_f_l1268_126807

noncomputable def f (x : ℝ) : ℝ := (2 * x - 1) / (3 + x)

theorem domain_of_f_of_f :
  {x : ℝ | x ≠ -3 ∧ x ≠ -8 / 5} =
  {x : ℝ | ∃ y : ℝ, f x = y ∧ y ≠ -3 ∧ x ≠ -3} :=
by
  sorry

end domain_of_f_of_f_l1268_126807


namespace problem1_problem2_l1268_126838

variable (x y a b c d : ℝ)
variable (h_a : a ≠ 0) (h_c : c ≠ 0) (h_d : d ≠ 0)

-- Problem 1: Prove (x + y) * (x^2 - x * y + y^2) = x^3 + y^3
theorem problem1 : (x + y) * (x^2 - x * y + y^2) = x^3 + y^3 := sorry

-- Problem 2: Prove ((a^2 * b) / (-c * d^3))^3 / (2 * a / d^3) * (c / (2 * a))^2 = - (a^3 * b^3) / (8 * c * d^6)
theorem problem2 (a b c d : ℝ) (h_a : a ≠ 0) (h_c : c ≠ 0) (h_d : d ≠ 0) : 
  ((a^2 * b) / (-c * d^3))^3 / (2 * a / d^3) * (c / (2 * a))^2 = - (a^3 * b^3) / (8 * c * d^6) := 
  sorry

end problem1_problem2_l1268_126838


namespace cookies_baked_l1268_126819

noncomputable def total_cookies (irin ingrid nell : ℚ) (percentage_ingrid : ℚ) : ℚ :=
  let total_ratio := irin + ingrid + nell
  let proportion_ingrid := ingrid / total_ratio
  let total_cookies := ingrid / (percentage_ingrid / 100)
  total_cookies

theorem cookies_baked (h_ratio: 9.18 + 5.17 + 2.05 = 16.4)
                      (h_percentage : 31.524390243902438 = 31.524390243902438) : 
  total_cookies 9.18 5.17 2.05 31.524390243902438 = 52 :=
by
  -- Placeholder for the proof.
  sorry

end cookies_baked_l1268_126819


namespace sufficient_condition_for_negation_l1268_126836

theorem sufficient_condition_for_negation {A B : Prop} (h : B → A) : ¬ A → ¬ B :=
by
  intro hA
  intro hB
  apply hA
  exact h hB

end sufficient_condition_for_negation_l1268_126836


namespace find_n_l1268_126889

theorem find_n (n : ℕ) (h : (1 + n) / (2 ^ n) = 3 / 16) : n = 5 :=
by sorry

end find_n_l1268_126889


namespace sum_of_fifth_powers_cannot_conclude_fourth_powers_l1268_126862

variable {a b c d : ℝ}

-- Part (a)
theorem sum_of_fifth_powers (h1 : a + b = c + d) (h2 : a^3 + b^3 = c^3 + d^3) : 
  a^5 + b^5 = c^5 + d^5 := sorry

-- Part (b)
theorem cannot_conclude_fourth_powers (h1 : a + b = c + d) (h2 : a^3 + b^3 = c^3 + d^3) : 
  ¬ (a^4 + b^4 = c^4 + d^4) := sorry

end sum_of_fifth_powers_cannot_conclude_fourth_powers_l1268_126862


namespace largest_common_multiple_3_5_l1268_126894

theorem largest_common_multiple_3_5 (n : ℕ) :
  (n < 10000) ∧ (n ≥ 1000) ∧ (n % 3 = 0) ∧ (n % 5 = 0) → n ≤ 9990 :=
sorry

end largest_common_multiple_3_5_l1268_126894


namespace zhou_yu_age_equation_l1268_126864

variable (x : ℕ)

theorem zhou_yu_age_equation (h : x + 3 < 10) : 10 * x + (x + 3) = (x + 3) ^ 2 :=
  sorry

end zhou_yu_age_equation_l1268_126864


namespace candies_eaten_l1268_126899

theorem candies_eaten (A B D : ℕ) 
                      (h1 : 4 * B = 3 * A) 
                      (h2 : 7 * A = 6 * D) 
                      (h3 : A + B + D = 70) :
  A = 24 ∧ B = 18 ∧ D = 28 := 
by
  sorry

end candies_eaten_l1268_126899


namespace more_birds_than_storks_l1268_126815

def initial_storks : ℕ := 5
def initial_birds : ℕ := 3
def additional_birds : ℕ := 4

def total_birds : ℕ := initial_birds + additional_birds

def stork_vs_bird_difference : ℕ := total_birds - initial_storks

theorem more_birds_than_storks : stork_vs_bird_difference = 2 := by
  sorry

end more_birds_than_storks_l1268_126815


namespace kiyiv_first_problem_kiyiv_second_problem_l1268_126897

/-- Let x and y be positive real numbers such that xy ≥ 1.
Prove that x^3 + y^3 + 4xy ≥ x^2 + y^2 + x + y + 2. -/
theorem kiyiv_first_problem (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : 1 ≤ x * y) :
  x^3 + y^3 + 4 * x * y ≥ x^2 + y^2 + x + y + 2 :=
sorry

/-- Let x and y be positive real numbers such that xy ≥ 1.
Prove that 2(x^3 + y^3 + xy + x + y) ≥ 5(x^2 + y^2). -/
theorem kiyiv_second_problem (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : 1 ≤ x * y) :
  2 * (x^3 + y^3 + x * y + x + y) ≥ 5 * (x^2 + y^2) :=
sorry

end kiyiv_first_problem_kiyiv_second_problem_l1268_126897


namespace cost_per_unit_l1268_126803

theorem cost_per_unit 
  (units_per_month : ℕ := 400)
  (selling_price_per_unit : ℝ := 440)
  (profit_requirement : ℝ := 40000)
  (C : ℝ) :
  profit_requirement ≤ (units_per_month * selling_price_per_unit) - (units_per_month * C) → C ≤ 340 :=
by
  sorry

end cost_per_unit_l1268_126803


namespace boys_without_calculators_l1268_126873

theorem boys_without_calculators 
  (total_students : ℕ)
  (total_boys : ℕ)
  (students_with_calculators : ℕ)
  (girls_with_calculators : ℕ)
  (H_total_students : total_students = 30)
  (H_total_boys : total_boys = 20)
  (H_students_with_calculators : students_with_calculators = 25)
  (H_girls_with_calculators : girls_with_calculators = 18) :
  total_boys - (students_with_calculators - girls_with_calculators) = 13 :=
by
  sorry

end boys_without_calculators_l1268_126873


namespace area_of_triangle_bounded_by_lines_l1268_126800

def line1 (x : ℝ) : ℝ := 2 * x + 3
def line2 (x : ℝ) : ℝ := - x + 5

theorem area_of_triangle_bounded_by_lines :
  let x_intercept_line1 := -3 / 2
  let x_intercept_line2 := 5
  let base := x_intercept_line2 - x_intercept_line1
  let intersection_x := 2 / 3
  let intersection_y := line1 intersection_x
  let height := intersection_y
  let area := (1 / 2) * base * height
  area = 169 / 12 := 
by
  sorry

end area_of_triangle_bounded_by_lines_l1268_126800


namespace certain_number_l1268_126869

theorem certain_number (p q x : ℝ) (h1 : 3 / p = x) (h2 : 3 / q = 15) (h3 : p - q = 0.3) : x = 6 :=
sorry

end certain_number_l1268_126869


namespace total_reading_materials_l1268_126821

theorem total_reading_materials 
  (books_per_shelf : ℕ) (magazines_per_shelf : ℕ) (newspapers_per_shelf : ℕ) (graphic_novels_per_shelf : ℕ) 
  (bookshelves : ℕ)
  (h_books : books_per_shelf = 23) 
  (h_magazines : magazines_per_shelf = 61) 
  (h_newspapers : newspapers_per_shelf = 17) 
  (h_graphic_novels : graphic_novels_per_shelf = 29) 
  (h_bookshelves : bookshelves = 37) : 
  (books_per_shelf * bookshelves + magazines_per_shelf * bookshelves + newspapers_per_shelf * bookshelves + graphic_novels_per_shelf * bookshelves) = 4810 := 
by {
  -- Condition definitions are already given; the proof is omitted here.
  sorry
}

end total_reading_materials_l1268_126821


namespace scientific_notation_384000_l1268_126817

theorem scientific_notation_384000 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ |a| ∧ |a| < 10 ∧ 384000 = a * 10 ^ n ∧ 
  a = 3.84 ∧ n = 5 :=
sorry

end scientific_notation_384000_l1268_126817


namespace geom_prog_all_integers_l1268_126820

theorem geom_prog_all_integers (b : ℕ) (r : ℚ) (a c : ℚ) :
  (∀ n : ℕ, ∃ k : ℤ, b * r ^ n = a * n + c) ∧ ∃ b_1 : ℤ, b = b_1 →
  (∀ n : ℕ, ∃ b_n : ℤ, b * r ^ n = b_n) :=
by
  sorry

end geom_prog_all_integers_l1268_126820


namespace positive_difference_in_x_coordinates_l1268_126852

-- Define points for line l
def point_l1 : ℝ × ℝ := (0, 10)
def point_l2 : ℝ × ℝ := (2, 0)

-- Define points for line m
def point_m1 : ℝ × ℝ := (0, 3)
def point_m2 : ℝ × ℝ := (10, 0)

-- Define the proof statement with the given problem
theorem positive_difference_in_x_coordinates :
  let y := 20
  let slope_l := (point_l2.2 - point_l1.2) / (point_l2.1 - point_l1.1)
  let intersection_l_x := (y - point_l1.2) / slope_l + point_l1.1
  let slope_m := (point_m2.2 - point_m1.2) / (point_m2.1 - point_m1.1)
  let intersection_m_x := (y - point_m1.2) / slope_m + point_m1.1
  abs (intersection_l_x - intersection_m_x) = 54.67 := 
  sorry -- Proof goes here

end positive_difference_in_x_coordinates_l1268_126852


namespace fraction_identity_l1268_126891

theorem fraction_identity (m n r t : ℚ) (h1 : m / n = 5 / 3) (h2 : r / t = 8 / 15) : 
  (4 * m * r - 2 * n * t) / (5 * n * t - 9 * m * r) = -14 / 27 :=
by 
  sorry

end fraction_identity_l1268_126891


namespace cos2x_quadratic_eq_specific_values_l1268_126856

variable (a b c x : ℝ)

axiom eqn1 : a * (Real.cos x) ^ 2 + b * Real.cos x + c = 0

noncomputable def quadratic_equation_cos2x 
  (a b c : ℝ) : ℝ × ℝ × ℝ := 
  (a^2, 2*a^2 + 2*a*c - b^2, a^2 + 2*a*c - b^2 + 4*c^2)

theorem cos2x_quadratic_eq 
  (a b c x : ℝ) 
  (h: a * (Real.cos x) ^ 2 + b * Real.cos x + c = 0) :
  (a^2) * (Real.cos (2*x))^2 + 
  (2*a^2 + 2*a*c - b^2) * Real.cos (2*x) + 
  (a^2 + 2*a*c - b^2 + 4*c^2) = 0 :=
sorry

theorem specific_values : 
  quadratic_equation_cos2x 4 2 (-1) = (4, 2, -1) :=
by
  unfold quadratic_equation_cos2x
  simp
  sorry

end cos2x_quadratic_eq_specific_values_l1268_126856


namespace max_servings_l1268_126874

-- Define available chunks for each type of fruit
def available_cantaloupe := 150
def available_honeydew := 135
def available_pineapple := 60
def available_watermelon := 220

-- Define the required chunks per serving for each type of fruit
def chunks_per_serving_cantaloupe := 3
def chunks_per_serving_honeydew := 2
def chunks_per_serving_pineapple := 1
def chunks_per_serving_watermelon := 4

-- Define the minimum required servings
def minimum_servings := 50

-- Prove the greatest number of servings that can be made while maintaining the specific ratio
theorem max_servings : 
  ∀ s : ℕ, 
  s * chunks_per_serving_cantaloupe ≤ available_cantaloupe ∧
  s * chunks_per_serving_honeydew ≤ available_honeydew ∧
  s * chunks_per_serving_pineapple ≤ available_pineapple ∧
  s * chunks_per_serving_watermelon ≤ available_watermelon ∧ 
  s ≥ minimum_servings → 
  s = 50 :=
by
  sorry

end max_servings_l1268_126874


namespace gcd_n_cube_plus_27_n_plus_3_l1268_126845

theorem gcd_n_cube_plus_27_n_plus_3 (n : ℕ) (h : n > 9) : 
  Nat.gcd (n^3 + 27) (n + 3) = n + 3 :=
sorry

end gcd_n_cube_plus_27_n_plus_3_l1268_126845


namespace rock_paper_scissors_l1268_126868

open Nat

-- Definitions based on problem conditions
def personA_movement (x y z : ℕ) : ℤ :=
  3 * (x : ℤ) - 2 * (y : ℤ) + (z : ℤ)

def personB_movement (x y z : ℕ) : ℤ :=
  3 * (y : ℤ) - 2 * (x : ℤ) + (z : ℤ)

def total_rounds (x y z : ℕ) : ℕ :=
  x + y + z

-- Problem statement
theorem rock_paper_scissors (x y z : ℕ) 
  (h1 : total_rounds x y z = 15)
  (h2 : personA_movement x y z = 17)
  (h3 : personB_movement x y z = 2) : x = 7 :=
by
  sorry

end rock_paper_scissors_l1268_126868


namespace binomial_term_is_constant_range_of_a_over_b_l1268_126883

noncomputable def binomial_term (a b : ℝ) (m n : ℤ) (r : ℕ) : ℝ :=
  Nat.choose 12 r * a^(12 - r) * b^r

theorem binomial_term_is_constant
  (a b : ℝ)
  (m n : ℤ)
  (h1: a > 0)
  (h2: b > 0)
  (h3: m ≠ 0)
  (h4: n ≠ 0)
  (h5: 2 * m + n = 0) :
  ∃ r, r = 4 ∧
  (binomial_term a b m n r) = 1 :=
sorry

theorem range_of_a_over_b 
  (a b : ℝ)
  (m n : ℤ)
  (h1: a > 0)
  (h2: b > 0)
  (h3: m ≠ 0)
  (h4: n ≠ 0)
  (h5: 2 * m + n = 0) :
  8 / 5 ≤ a / b ∧ a / b ≤ 9 / 4 :=
sorry

end binomial_term_is_constant_range_of_a_over_b_l1268_126883


namespace solve_quadratic_eq_l1268_126812

theorem solve_quadratic_eq (x : ℝ) (h : x^2 - 4 * x - 1 = 0) : x = 2 + Real.sqrt 5 ∨ x = 2 - Real.sqrt 5 :=
sorry

end solve_quadratic_eq_l1268_126812


namespace nitin_ranks_from_last_l1268_126885

def total_students : ℕ := 75

def math_rank_start : ℕ := 24
def english_rank_start : ℕ := 18

def rank_from_last (total : ℕ) (rank_start : ℕ) : ℕ :=
  total - rank_start + 1

theorem nitin_ranks_from_last :
  rank_from_last total_students math_rank_start = 52 ∧
  rank_from_last total_students english_rank_start = 58 :=
by
  sorry

end nitin_ranks_from_last_l1268_126885


namespace removed_number_is_24_l1268_126871

theorem removed_number_is_24
  (S9 : ℕ) (S8 : ℕ) (avg_9 : ℕ) (avg_8 : ℕ) (h1 : avg_9 = 72) (h2 : avg_8 = 78) (h3 : S9 = avg_9 * 9) (h4 : S8 = avg_8 * 8) :
  S9 - S8 = 24 :=
by
  sorry

end removed_number_is_24_l1268_126871


namespace total_legs_of_passengers_l1268_126892

theorem total_legs_of_passengers :
  ∀ (total_heads cats cat_legs human_heads normal_human_legs one_legged_captain_legs : ℕ),
  total_heads = 15 →
  cats = 7 →
  cat_legs = 4 →
  human_heads = (total_heads - cats) →
  normal_human_legs = 2 →
  one_legged_captain_legs = 1 →
  ((cats * cat_legs) + ((human_heads - 1) * normal_human_legs) + one_legged_captain_legs) = 43 :=
by
  intros total_heads cats cat_legs human_heads normal_human_legs one_legged_captain_legs h1 h2 h3 h4 h5 h6
  sorry

end total_legs_of_passengers_l1268_126892


namespace smallest_integer_l1268_126886

-- Given positive integer M such that
def satisfies_conditions (M : ℕ) : Prop :=
  M % 6 = 5 ∧
  M % 7 = 6 ∧
  M % 8 = 7 ∧
  M % 9 = 8 ∧
  M % 10 = 9 ∧
  M % 11 = 10 ∧
  M % 13 = 12

-- The main theorem to prove
theorem smallest_integer (M : ℕ) (h : satisfies_conditions M) : M = 360359 :=
  sorry

end smallest_integer_l1268_126886


namespace pie_eating_contest_l1268_126842

theorem pie_eating_contest :
  let first_student := (5 : ℚ) / 6
  let second_student := (2 : ℚ) / 3
  let third_student := (3 : ℚ) / 4
  max (max first_student second_student) third_student - 
  min (min first_student second_student) third_student = 1 / 6 :=
by
  let first_student := (5 : ℚ) / 6
  let second_student := (2 : ℚ) / 3
  let third_student := (3 : ℚ) / 4
  sorry

end pie_eating_contest_l1268_126842


namespace cindy_correct_answer_l1268_126818

noncomputable def cindy_number (x : ℝ) : Prop :=
  (x - 10) / 5 = 40

theorem cindy_correct_answer (x : ℝ) (h : cindy_number x) : (x - 4) / 10 = 20.6 :=
by
  -- The proof is omitted as instructed
  sorry

end cindy_correct_answer_l1268_126818


namespace derivative_of_f_eval_deriv_at_pi_over_6_l1268_126879

noncomputable def f (x : Real) : Real := (Real.sin x) ^ 4 + (Real.cos x) ^ 4

theorem derivative_of_f : ∀ x, deriv f x = -Real.sin (4 * x) :=
by
  intro x
  sorry

theorem eval_deriv_at_pi_over_6 : deriv f (Real.pi / 6) = -Real.sqrt 3 / 2 :=
by
  rw [derivative_of_f]
  sorry

end derivative_of_f_eval_deriv_at_pi_over_6_l1268_126879


namespace commutative_star_not_distributive_star_special_case_star_false_no_identity_star_l1268_126854

def star (x y : ℕ) : ℕ := (x + 2) * (y + 2) - 2

theorem commutative_star : ∀ x y : ℕ, star x y = star y x := by
  sorry

theorem not_distributive_star : ∃ x y z : ℕ, star x (y + z) ≠ star x y + star x z := by
  sorry

theorem special_case_star_false : ∀ x : ℕ, star (x - 2) (x + 2) ≠ star x x - 2 := by
  sorry

theorem no_identity_star : ¬∃ e : ℕ, ∀ x : ℕ, star x e = x ∧ star e x = x := by
  sorry

-- Associativity requires further verification and does not have a definitive statement yet.

end commutative_star_not_distributive_star_special_case_star_false_no_identity_star_l1268_126854


namespace jerry_apples_l1268_126805

theorem jerry_apples (J : ℕ) (h1 : 20 + 60 + J = 3 * 2 * 20):
  J = 40 :=
sorry

end jerry_apples_l1268_126805


namespace part_two_l1268_126814

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  x^2 - 2 * x + a * Real.log x

theorem part_two (a : ℝ) (h : a = 4) (m n : ℝ) (hm : 0 < m) (hn : 0 < n)
  (h_cond : (f m a + f n a) / (m^2 * n^2) = 1) : m + n ≥ 3 :=
sorry

end part_two_l1268_126814


namespace total_surface_area_l1268_126872

noncomputable def calculate_surface_area
  (radius : ℝ) (reflective : Bool) : ℝ :=
  let base_area := (radius^2 * Real.pi)
  let curved_surface_area := (4 * Real.pi * (radius^2)) / 2
  let effective_surface_area := if reflective then 2 * curved_surface_area else curved_surface_area
  effective_surface_area

theorem total_surface_area (r : ℝ) (h₁_reflective : Bool) (h₂_reflective : Bool) :
  r = 8 →
  h₁_reflective = false →
  h₂_reflective = true →
  (calculate_surface_area r h₁_reflective + calculate_surface_area r h₂_reflective) = 384 * Real.pi := 
by
  sorry

end total_surface_area_l1268_126872


namespace chocolates_bought_at_cost_price_l1268_126884

variables (C S : ℝ) (n : ℕ)

-- Given conditions
def cost_eq_selling_50 := n * C = 50 * S
def gain_percent := (S - C) / C = 0.30

-- Question to prove
theorem chocolates_bought_at_cost_price (h1 : cost_eq_selling_50 C S n) (h2 : gain_percent C S) : n = 65 :=
sorry

end chocolates_bought_at_cost_price_l1268_126884


namespace well_depth_and_rope_length_l1268_126813

variables (x y : ℝ)

theorem well_depth_and_rope_length :
  (y = x / 4 - 3) ∧ (y = x / 5 + 1) → y = 17 ∧ x = 80 :=
by
  sorry
 
end well_depth_and_rope_length_l1268_126813


namespace constant_value_l1268_126866

theorem constant_value (x y z C : ℤ) (h1 : x = z + 2) (h2 : y = z + 1) (h3 : x > y) (h4 : y > z) (h5 : z = 2) (h6 : 2 * x + 3 * y + 3 * z = 5 * y + C) : C = 8 :=
by
  sorry

end constant_value_l1268_126866


namespace is_isosceles_triangle_l1268_126876

theorem is_isosceles_triangle 
  (a b c : ℝ)
  (A B C : ℝ)
  (h : a * Real.cos B + b * Real.cos C + c * Real.cos A = b * Real.cos A + c * Real.cos B + a * Real.cos C) : 
  (A = B ∨ B = C ∨ A = C) :=
sorry

end is_isosceles_triangle_l1268_126876


namespace solution_set_absolute_value_sum_eq_three_l1268_126848

theorem solution_set_absolute_value_sum_eq_three (m n : ℝ) (h : ∀ x : ℝ, (|2 * x - 3| ≤ 1) ↔ (m ≤ x ∧ x ≤ n)) : m + n = 3 :=
sorry

end solution_set_absolute_value_sum_eq_three_l1268_126848


namespace seth_spent_more_l1268_126804

def cost_ice_cream (cartons : ℕ) (price : ℕ) := cartons * price
def cost_yogurt (cartons : ℕ) (price : ℕ) := cartons * price
def amount_spent (cost_ice : ℕ) (cost_yog : ℕ) := cost_ice - cost_yog

theorem seth_spent_more :
  amount_spent (cost_ice_cream 20 6) (cost_yogurt 2 1) = 118 := by
  sorry

end seth_spent_more_l1268_126804


namespace max_rectangle_perimeter_l1268_126896

theorem max_rectangle_perimeter (n : ℕ) (a b : ℕ) (ha : a * b = 180) (hb: ∀ (a b : ℕ),  6 ∣ (a * b) → a * b = 180): 
  2 * (a + b) ≤ 184 :=
sorry

end max_rectangle_perimeter_l1268_126896


namespace Sam_balloons_correct_l1268_126857

def Fred_balloons : Nat := 10
def Dan_balloons : Nat := 16
def Total_balloons : Nat := 72

def Sam_balloons : Nat := Total_balloons - Fred_balloons - Dan_balloons

theorem Sam_balloons_correct : Sam_balloons = 46 := by 
  have H : Sam_balloons = 72 - 10 - 16 := rfl
  simp at H
  exact H

end Sam_balloons_correct_l1268_126857


namespace unpacked_boxes_l1268_126855

-- Definitions of boxes per case
def boxesPerCaseLemonChalet : Nat := 12
def boxesPerCaseThinMints : Nat := 15
def boxesPerCaseSamoas : Nat := 10
def boxesPerCaseTrefoils : Nat := 18

-- Definitions of boxes sold by Deborah
def boxesSoldLemonChalet : Nat := 31
def boxesSoldThinMints : Nat := 26
def boxesSoldSamoas : Nat := 17
def boxesSoldTrefoils : Nat := 44

-- The theorem stating the number of boxes that will not be packed to a case
theorem unpacked_boxes :
  boxesSoldLemonChalet % boxesPerCaseLemonChalet = 7 ∧
  boxesSoldThinMints % boxesPerCaseThinMints = 11 ∧
  boxesSoldSamoas % boxesPerCaseSamoas = 7 ∧
  boxesSoldTrefoils % boxesPerCaseTrefoils = 8 := 
by
  sorry

end unpacked_boxes_l1268_126855


namespace uki_total_earnings_l1268_126839

-- Define the conditions
def price_cupcake : ℝ := 1.50
def price_cookie : ℝ := 2
def price_biscuit : ℝ := 1
def cupcakes_per_day : ℕ := 20
def cookies_per_day : ℕ := 10
def biscuits_per_day : ℕ := 20
def days : ℕ := 5

-- Prove the total earnings for five days
theorem uki_total_earnings : 
    (cupcakes_per_day * price_cupcake + 
     cookies_per_day * price_cookie + 
     biscuits_per_day * price_biscuit) * days = 350 := 
by
  -- The actual proof will go here, but is omitted for now.
  sorry

end uki_total_earnings_l1268_126839


namespace total_cuts_length_eq_60_l1268_126849

noncomputable def total_length_of_cuts (side_length : ℝ) (num_rectangles : ℕ) : ℝ :=
  if side_length = 36 ∧ num_rectangles = 3 then 60 else 0

theorem total_cuts_length_eq_60 :
  ∀ (side_length : ℝ) (num_rectangles : ℕ),
    side_length = 36 ∧ num_rectangles = 3 →
    total_length_of_cuts side_length num_rectangles = 60 := by
  intros
  simp [total_length_of_cuts]
  sorry

end total_cuts_length_eq_60_l1268_126849


namespace fixed_monthly_fee_l1268_126882

def FebruaryBill (x y : ℝ) : Prop := x + y = 18.72
def MarchBill (x y : ℝ) : Prop := x + 3 * y = 28.08

theorem fixed_monthly_fee (x y : ℝ) (h1 : FebruaryBill x y) (h2 : MarchBill x y) : x = 14.04 :=
by 
  sorry

end fixed_monthly_fee_l1268_126882


namespace how_many_rocks_l1268_126853

section see_saw_problem

-- Conditions
def Jack_weight : ℝ := 60
def Anna_weight : ℝ := 40
def rock_weight : ℝ := 4

-- Theorem statement
theorem how_many_rocks : (Jack_weight - Anna_weight) / rock_weight = 5 :=
by
  -- Proof is omitted, just ensuring the theorem statement
  sorry

end see_saw_problem

end how_many_rocks_l1268_126853


namespace tg_gamma_half_eq_2_div_5_l1268_126829

theorem tg_gamma_half_eq_2_div_5
  (α β γ : ℝ)
  (a b c : ℝ)
  (triangle_angles : α + β + γ = π)
  (tg_half_alpha : Real.tan (α / 2) = 5/6)
  (tg_half_beta : Real.tan (β / 2) = 10/9)
  (ac_eq_2b : a + c = 2 * b):
  Real.tan (γ / 2) = 2 / 5 :=
sorry

end tg_gamma_half_eq_2_div_5_l1268_126829


namespace number_of_students_l1268_126860

theorem number_of_students : 
    ∃ (n : ℕ), 
      (∃ (x : ℕ), 
        (∀ (k : ℕ), x = 4 * k ∧ 5 * x + 1 = n)
      ) ∧ 
      (∃ (y : ℕ), 
        (∀ (k : ℕ), y = 5 * k ∧ 4 * y + 1 = n)
      ) ∧
      n ≤ 30 ∧ 
      n = 21 :=
  sorry

end number_of_students_l1268_126860


namespace count_perfect_cubes_l1268_126802

theorem count_perfect_cubes (a b : ℕ) (h₁ : 200 < a) (h₂ : a < 1500) (h₃ : b = 6^3) :
  (∃! n : ℕ, 200 < n^3 ∧ n^3 < 1500) :=
sorry

end count_perfect_cubes_l1268_126802


namespace simplify_expression_l1268_126801

variable (b : ℝ)

theorem simplify_expression : 3 * b * (3 * b^2 + 2 * b) - 2 * b^2 = 9 * b^3 + 4 * b^2 :=
by
  sorry

end simplify_expression_l1268_126801


namespace obtuse_angle_in_second_quadrant_l1268_126875

-- Let θ be an angle in degrees
def angle_in_first_quadrant (θ : ℝ) : Prop := 0 < θ ∧ θ < 90

def angle_terminal_side_same (θ₁ θ₂ : ℝ) : Prop := θ₁ % 360 = θ₂ % 360

def angle_in_fourth_quadrant (θ : ℝ) : Prop := -360 < θ ∧ θ < 0 ∧ (θ + 360) > 270

def is_obtuse_angle (θ : ℝ) : Prop := 90 < θ ∧ θ < 180

-- Statement D: An obtuse angle is definitely in the second quadrant
theorem obtuse_angle_in_second_quadrant (θ : ℝ) (h : is_obtuse_angle θ) :
  90 < θ ∧ θ < 180 := by
    sorry

end obtuse_angle_in_second_quadrant_l1268_126875


namespace gabby_additional_money_needed_l1268_126837

theorem gabby_additional_money_needed
  (cost_makeup : ℕ := 65)
  (cost_skincare : ℕ := 45)
  (cost_hair_tool : ℕ := 55)
  (initial_savings : ℕ := 35)
  (money_from_mom : ℕ := 20)
  (money_from_dad : ℕ := 30)
  (money_from_chores : ℕ := 25) :
  (cost_makeup + cost_skincare + cost_hair_tool) - (initial_savings + money_from_mom + money_from_dad + money_from_chores) = 55 := 
by
  sorry

end gabby_additional_money_needed_l1268_126837


namespace correct_growth_rate_equation_l1268_126847

noncomputable def numberOfBikesFirstMonth : ℕ := 1000
noncomputable def additionalBikesThirdMonth : ℕ := 440
noncomputable def monthlyGrowthRate (x : ℝ) : Prop :=
  numberOfBikesFirstMonth * (1 + x)^2 = numberOfBikesFirstMonth + additionalBikesThirdMonth

theorem correct_growth_rate_equation (x : ℝ) : monthlyGrowthRate x :=
by
  sorry

end correct_growth_rate_equation_l1268_126847


namespace find_remainder_division_l1268_126890

/--
Given:
1. A dividend of 100.
2. A quotient of 9.
3. A divisor of 11.

Prove: The remainder \( r \) when dividing 100 by 11 is 1.
-/
theorem find_remainder_division :
  ∀ (q d r : Nat), q = 9 → d = 11 → 100 = (d * q + r) → r = 1 :=
by
  intros q d r hq hd hdiv
  -- Proof steps would go here
  sorry

end find_remainder_division_l1268_126890


namespace quadratic_inequality_solution_l1268_126840

theorem quadratic_inequality_solution (a : ℝ) :
  (¬ ∀ x : ℝ, a * x^2 - 2 * a * x + 3 > 0) ↔ (a < 0 ∨ a ≥ 3) :=
sorry

end quadratic_inequality_solution_l1268_126840


namespace joe_lists_count_l1268_126881

def num_options (n : ℕ) (k : ℕ) : ℕ := n ^ k

theorem joe_lists_count : num_options 12 3 = 1728 := by
  unfold num_options
  sorry

end joe_lists_count_l1268_126881


namespace max_positive_integer_difference_l1268_126887

theorem max_positive_integer_difference (x y : ℝ) (hx : 4 < x ∧ x < 8) (hy : 8 < y ∧ y < 12) : ∃ d : ℕ, d = 6 :=
by
  sorry

end max_positive_integer_difference_l1268_126887


namespace james_spends_on_pistachios_per_week_l1268_126830

theorem james_spends_on_pistachios_per_week :
  let cost_per_can := 10
  let ounces_per_can := 5
  let total_ounces_per_5_days := 30
  let days_per_week := 7
  let cost_per_ounce := cost_per_can / ounces_per_can
  let daily_ounces := total_ounces_per_5_days / 5
  let daily_cost := daily_ounces * cost_per_ounce
  daily_cost * days_per_week = 84 :=
by
  sorry

end james_spends_on_pistachios_per_week_l1268_126830


namespace calories_per_person_l1268_126825

theorem calories_per_person (oranges : ℕ) (pieces_per_orange : ℕ) (people : ℕ) (calories_per_orange : ℕ) :
  oranges = 5 →
  pieces_per_orange = 8 →
  people = 4 →
  calories_per_orange = 80 →
  (oranges * pieces_per_orange) / people * ((oranges * calories_per_orange) / (oranges * pieces_per_orange)) = 100 :=
by
  intros h_oranges h_pieces_per_orange h_people h_calories_per_orange
  sorry

end calories_per_person_l1268_126825


namespace divisibility_condition_l1268_126827

theorem divisibility_condition (a p q : ℕ) (hp : p > 0) (ha : a > 0) (hq : q > 0) (h : p ≤ q) :
  (p ∣ a^p ↔ p ∣ a^q) :=
sorry

end divisibility_condition_l1268_126827


namespace molecular_weight_one_mole_l1268_126863

variable (molecular_weight : ℕ → ℕ)

theorem molecular_weight_one_mole (h : molecular_weight 7 = 2856) :
  molecular_weight 1 = 408 :=
sorry

end molecular_weight_one_mole_l1268_126863


namespace john_new_weekly_earnings_l1268_126833

theorem john_new_weekly_earnings :
  ∀ (original_earnings : ℤ) (percentage_increase : ℝ),
  original_earnings = 60 →
  percentage_increase = 66.67 →
  (original_earnings + (percentage_increase / 100 * original_earnings)) = 100 := 
by
  intros original_earnings percentage_increase h_earnings h_percentage
  rw [h_earnings, h_percentage]
  norm_num
  sorry

end john_new_weekly_earnings_l1268_126833


namespace Kylie_coins_left_l1268_126822

-- Definitions based on given conditions
def piggyBank := 30
def brother := 26
def father := 2 * brother
def sofa := 15
def totalCoins := piggyBank + brother + father + sofa
def coinsGivenToLaura := totalCoins / 2
def coinsLeft := totalCoins - coinsGivenToLaura

-- Theorem statement
theorem Kylie_coins_left : coinsLeft = 62 := by sorry

end Kylie_coins_left_l1268_126822


namespace a3_value_l1268_126870

variable {a : ℕ → ℤ} -- Arithmetic sequence as a function from natural numbers to integers
variable {S : ℕ → ℤ} -- Sum of the first n terms

-- Conditions
axiom a1_eq : a 1 = -11
axiom a4_plus_a6_eq : a 4 + a 6 = -6
-- Common difference d
variable {d : ℤ}
axiom d_def : ∀ n, a (n + 1) = a n + d

theorem a3_value : a 3 = -7 := by
  sorry -- Proof not required as per the instructions

end a3_value_l1268_126870


namespace differentiable_function_inequality_l1268_126895

theorem differentiable_function_inequality (f : ℝ → ℝ) (h_diff : Differentiable ℝ f) 
  (h_cond : ∀ x : ℝ, (x - 1) * (deriv f x) ≥ 0) : 
  f 0 + f 2 ≥ 2 * (f 1) :=
sorry

end differentiable_function_inequality_l1268_126895


namespace books_in_either_but_not_both_l1268_126816

theorem books_in_either_but_not_both (shared_books alice_books bob_unique_books : ℕ) 
    (h1 : shared_books = 12) 
    (h2 : alice_books = 26)
    (h3 : bob_unique_books = 8) : 
    (alice_books - shared_books) + bob_unique_books = 22 :=
by
  sorry

end books_in_either_but_not_both_l1268_126816


namespace vehicle_A_no_speed_increase_needed_l1268_126859

noncomputable def V_A := 60 -- Speed of Vehicle A in mph
noncomputable def V_B := 70 -- Speed of Vehicle B in mph
noncomputable def V_C := 50 -- Speed of Vehicle C in mph
noncomputable def dist_AB := 100 -- Initial distance between A and B in ft
noncomputable def dist_AC := 300 -- Initial distance between A and C in ft

theorem vehicle_A_no_speed_increase_needed 
  (V_A V_B V_C : ℝ)
  (dist_AB dist_AC : ℝ)
  (h1 : V_A > V_C)
  (h2 : V_A = 60)
  (h3 : V_B = 70)
  (h4 : V_C = 50)
  (h5 : dist_AB = 100)
  (h6 : dist_AC = 300) : 
  ∀ ΔV : ℝ, ΔV = 0 :=
by
  sorry -- Proof to be filled out

end vehicle_A_no_speed_increase_needed_l1268_126859


namespace actual_distance_traveled_l1268_126831

theorem actual_distance_traveled (D : ℝ) (h : D / 10 = (D + 20) / 20) : D = 20 :=
  sorry

end actual_distance_traveled_l1268_126831


namespace solutionTriangle_l1268_126823

noncomputable def solveTriangle (a b : ℝ) (B : ℝ) : (ℝ × ℝ × ℝ) :=
  let A := 30
  let C := 30
  let c := 2
  (A, C, c)

theorem solutionTriangle :
  solveTriangle 2 (2 * Real.sqrt 3) 120 = (30, 30, 2) :=
by
  sorry

end solutionTriangle_l1268_126823


namespace conic_section_is_hyperbola_l1268_126880

theorem conic_section_is_hyperbola (x y : ℝ) :
  (x - 3)^2 = (3 * y + 4)^2 - 75 → 
  ∃ a b c d e f : ℝ, a * x^2 + b * y^2 + c * x + d * y + e = 0 ∧ a ≠ 0 ∧ b ≠ 0 ∧ a * b < 0 :=
sorry

end conic_section_is_hyperbola_l1268_126880


namespace minimum_value_l1268_126861

theorem minimum_value (a b c : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : c > 2) 
  (h4 : a + b = 1) : 
  ∃ L, L = (3 * a * c / b) + (c / (a * b)) + (6 / (c - 2)) ∧ L = 1 / (a * (1 - a)) := sorry

end minimum_value_l1268_126861


namespace min_cylinder_volume_eq_surface_area_l1268_126810

theorem min_cylinder_volume_eq_surface_area (r h V S : ℝ) (hr : r > 0) (hh : h > 0)
  (hV : V = π * r^2 * h) (hS : S = 2 * π * r^2 + 2 * π * r * h) (heq : V = S) :
  V = 54 * π :=
by
  -- Placeholder for the actual proof
  sorry

end min_cylinder_volume_eq_surface_area_l1268_126810


namespace area_correct_l1268_126835

noncomputable def area_bounded_curves : ℝ := sorry

theorem area_correct :
  ∃ S, S = area_bounded_curves ∧ S = 12 * pi + 16 := sorry

end area_correct_l1268_126835

import Mathlib

namespace NUMINAMATH_GPT_probability_left_red_off_second_blue_on_right_blue_on_l1465_146598

def num_red_lamps : ℕ := 4
def num_blue_lamps : ℕ := 4
def total_lamps : ℕ := num_red_lamps + num_blue_lamps
def num_on : ℕ := 4
def position := Fin total_lamps
def lamp_state := {state // state < (total_lamps.choose num_red_lamps) * (total_lamps.choose num_on)}

def valid_configuration (leftmost : position) (second_left : position) (rightmost : position) (s : lamp_state) : Prop :=
(leftmost.1 = 1 ∧ second_left.1 = 2 ∧ rightmost.1 = 8) ∧ (s.1 =  (((total_lamps - 3).choose 3) * ((total_lamps - 3).choose 2)))

theorem probability_left_red_off_second_blue_on_right_blue_on :
  ∀ (leftmost second_left rightmost : position) (s : lamp_state),
  valid_configuration leftmost second_left rightmost s ->
  ((total_lamps.choose num_red_lamps) * (total_lamps.choose num_on)) = 49 :=
sorry

end NUMINAMATH_GPT_probability_left_red_off_second_blue_on_right_blue_on_l1465_146598


namespace NUMINAMATH_GPT_smallest_prime_sum_l1465_146553

theorem smallest_prime_sum (a b c d : ℕ) (ha : Prime a) (hb : Prime b) (hc : Prime c) (hd : Prime d)
  (hab : a ≠ b) (hac : a ≠ c) (had : a ≠ d) (hbc : b ≠ c) (hbd : b ≠ d) (hcd : c ≠ d)
  (H1 : Prime (a + b + c + d))
  (H2 : Prime (a + b)) (H3 : Prime (a + c)) (H4 : Prime (a + d)) (H5 : Prime (b + c)) (H6 : Prime (b + d)) (H7 : Prime (c + d))
  (H8 : Prime (a + b + c)) (H9 : Prime (a + b + d)) (H10 : Prime (a + c + d)) (H11 : Prime (b + c + d))
  : a + b + c + d = 31 :=
sorry

end NUMINAMATH_GPT_smallest_prime_sum_l1465_146553


namespace NUMINAMATH_GPT_minimum_x_y_sum_l1465_146548

theorem minimum_x_y_sum (x y : ℕ) (hx : x ≠ y) (hx_pos : 0 < x) (hy_pos : 0 < y)
  (h : (1 / (x : ℚ)) + (1 / (y : ℚ)) = 1 / 15) : x + y = 64 :=
  sorry

end NUMINAMATH_GPT_minimum_x_y_sum_l1465_146548


namespace NUMINAMATH_GPT_total_buttons_l1465_146591

theorem total_buttons (green buttons: ℕ) (yellow buttons: ℕ) (blue buttons: ℕ) (total buttons: ℕ) 
(h1: green = 90) (h2: yellow = green + 10) (h3: blue = green - 5) : total = green + yellow + blue → total = 275 :=
by
  sorry

end NUMINAMATH_GPT_total_buttons_l1465_146591


namespace NUMINAMATH_GPT_selling_price_when_profit_equals_loss_l1465_146501

theorem selling_price_when_profit_equals_loss (CP SP Rs_57 : ℕ) (h1: CP = 50) (h2: Rs_57 = 57) (h3: Rs_57 - CP = CP - SP) : 
  SP = 43 := by
  sorry

end NUMINAMATH_GPT_selling_price_when_profit_equals_loss_l1465_146501


namespace NUMINAMATH_GPT_factor_quadratic_l1465_146564

theorem factor_quadratic : ∀ (x : ℝ), 16*x^2 - 40*x + 25 = (4*x - 5)^2 := by
  intro x
  sorry

end NUMINAMATH_GPT_factor_quadratic_l1465_146564


namespace NUMINAMATH_GPT_no_partition_square_isosceles_10deg_l1465_146596

theorem no_partition_square_isosceles_10deg :
  ¬ ∃ (P : ℝ → ℝ → Prop), 
    (∀ x y, P x y → ((x = y) ∨ ((10 * x + 10 * y + 160 * (180 - x - y)) = 9 * 10))) ∧
    (∀ x y, P x 90 → P x y) ∧
    (P 90 90 → False) :=
by
  sorry

end NUMINAMATH_GPT_no_partition_square_isosceles_10deg_l1465_146596


namespace NUMINAMATH_GPT_no_positive_int_squares_l1465_146569

theorem no_positive_int_squares (n : ℕ) (h_pos : 0 < n) :
  ¬ (∃ a b c : ℕ, a ^ 2 = 2 * n ^ 2 + 1 ∧ b ^ 2 = 3 * n ^ 2 + 1 ∧ c ^ 2 = 6 * n ^ 2 + 1) := by
  sorry

end NUMINAMATH_GPT_no_positive_int_squares_l1465_146569


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l1465_146575

theorem necessary_but_not_sufficient (a b x y : ℤ) (ha : 0 < a) (hb : 0 < b) (h1 : x - y > a + b) (h2 : x * y > a * b) : 
  (x > a ∧ y > b) := sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l1465_146575


namespace NUMINAMATH_GPT_factorial_last_nonzero_digit_non_periodic_l1465_146577

def last_nonzero_digit (n : ℕ) : ℕ :=
  -- function to compute last nonzero digit of n!
  sorry

def sequence_periodic (a : ℕ → ℕ) (T : ℕ) : Prop :=
  ∀ n, a n = a (n + T)

theorem factorial_last_nonzero_digit_non_periodic : ¬ ∃ T, sequence_periodic last_nonzero_digit T :=
  sorry

end NUMINAMATH_GPT_factorial_last_nonzero_digit_non_periodic_l1465_146577


namespace NUMINAMATH_GPT_reciprocal_2023_l1465_146576

def reciprocal (x : ℕ) := 1 / x

theorem reciprocal_2023 : reciprocal 2023 = 1 / 2023 :=
by
  sorry

end NUMINAMATH_GPT_reciprocal_2023_l1465_146576


namespace NUMINAMATH_GPT_find_a_for_even_function_l1465_146584

theorem find_a_for_even_function (a : ℝ) (h : ∀ x : ℝ, x^3 * (a * 2^x - 2^(-x)) = (-x)^3 * (a * 2^(-x) - 2^x)) : a = 1 :=
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_find_a_for_even_function_l1465_146584


namespace NUMINAMATH_GPT_sum_a1_to_a14_equals_zero_l1465_146504

theorem sum_a1_to_a14_equals_zero 
  (a a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 : ℝ) 
  (h1 : (1 + x - x^2)^3 * (1 - 2 * x^2)^4 = a + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4 + a5 * x^5 + a6 * x^6 + a7 * x^7 + a8 * x^8 + a9 * x^9 + a10 * x^10 + a11 * x^11 + a12 * x^12 + a13 * x^13 + a14 * x^14) 
  (h2 : a + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 = 1) 
  (h3 : a = 1) : 
  a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 = 0 := by
  sorry

end NUMINAMATH_GPT_sum_a1_to_a14_equals_zero_l1465_146504


namespace NUMINAMATH_GPT_no_nat_nums_satisfying_l1465_146565

theorem no_nat_nums_satisfying (x y z k : ℕ) (hx : x < k) (hy : y < k) : x^k + y^k ≠ z^k :=
by
  sorry

end NUMINAMATH_GPT_no_nat_nums_satisfying_l1465_146565


namespace NUMINAMATH_GPT_problem_statement_l1465_146545

noncomputable def M (x y : ℝ) : ℝ := max x y
noncomputable def m (x y : ℝ) : ℝ := min x y

theorem problem_statement {p q r s t : ℝ} (h1 : p < q) (h2 : q < r) (h3 : r < s) (h4 : s < t) :
  M (M p (m q r)) (m s (m p t)) = q :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1465_146545


namespace NUMINAMATH_GPT_greatest_common_divisor_of_98_and_n_l1465_146585

theorem greatest_common_divisor_of_98_and_n (n : ℕ) (h1 : ∃ (d : Finset ℕ),  d = {1, 7, 49} ∧ ∀ x ∈ d, x ∣ 98 ∧ x ∣ n) :
  ∃ (g : ℕ), g = 49 :=
by
  sorry

end NUMINAMATH_GPT_greatest_common_divisor_of_98_and_n_l1465_146585


namespace NUMINAMATH_GPT_negation_of_forall_x_squared_nonnegative_l1465_146593

theorem negation_of_forall_x_squared_nonnegative :
  ¬ (∀ x : ℝ, x^2 ≥ 0) ↔ ∃ x : ℝ, x^2 < 0 :=
sorry

end NUMINAMATH_GPT_negation_of_forall_x_squared_nonnegative_l1465_146593


namespace NUMINAMATH_GPT_rectangle_x_is_18_l1465_146597

-- Definitions for the conditions
def rectangle (a b x : ℕ) : Prop := 
  (a = 2 * b) ∧
  (x = 2 * (a + b)) ∧
  (x = a * b)

-- Theorem to prove the equivalence of the conditions and the answer \( x = 18 \)
theorem rectangle_x_is_18 : ∀ a b x : ℕ, rectangle a b x → x = 18 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_x_is_18_l1465_146597


namespace NUMINAMATH_GPT_second_tray_holds_l1465_146514

-- The conditions and the given constants
variables (x : ℕ) (h1 : 2 * x - 20 = 500)

-- The theorem proving the number of cups the second tray holds is 240 
theorem second_tray_holds (h2 : x = 260) : x - 20 = 240 := by
  sorry

end NUMINAMATH_GPT_second_tray_holds_l1465_146514


namespace NUMINAMATH_GPT_cylinder_h_over_r_equals_one_l1465_146525

theorem cylinder_h_over_r_equals_one
  (A : ℝ) (r h : ℝ)
  (h_surface_area : A = 2 * π * r^2 + 2 * π * r * h)
  (V : ℝ := π * r^2 * h)
  (max_V : ∀ r' h', (A = 2 * π * r'^2 + 2 * π * r' * h') → (π * r'^2 * h' ≤ V) → (r' = r ∧ h' = h)) :
  h / r = 1 := by
sorry

end NUMINAMATH_GPT_cylinder_h_over_r_equals_one_l1465_146525


namespace NUMINAMATH_GPT_polygon_interior_angles_sum_l1465_146566

theorem polygon_interior_angles_sum (n : ℕ) (h : (n - 2) * 180 = 720) : n = 6 := 
by sorry

end NUMINAMATH_GPT_polygon_interior_angles_sum_l1465_146566


namespace NUMINAMATH_GPT_proportion_x_l1465_146512

theorem proportion_x (x : ℝ) (h : 3 / 12 = x / 16) : x = 4 :=
sorry

end NUMINAMATH_GPT_proportion_x_l1465_146512


namespace NUMINAMATH_GPT_number_x_is_divided_by_l1465_146533

-- Define the conditions
variable (x y n : ℕ)
variable (cond1 : x = n * y + 4)
variable (cond2 : 2 * x = 8 * 3 * y + 3)
variable (cond3 : 13 * y - x = 1)

-- Define the statement to be proven
theorem number_x_is_divided_by : n = 11 :=
by
  sorry

end NUMINAMATH_GPT_number_x_is_divided_by_l1465_146533


namespace NUMINAMATH_GPT_no_solution_inequalities_l1465_146561

theorem no_solution_inequalities (a : ℝ) : 
  (∀ x : ℝ, ¬ (x > 3 ∧ x < a)) ↔ (a ≤ 3) :=
by
  sorry

end NUMINAMATH_GPT_no_solution_inequalities_l1465_146561


namespace NUMINAMATH_GPT_translation_coordinates_l1465_146526

theorem translation_coordinates
  (a b : ℝ)
  (h₁ : 4 = a + 2)
  (h₂ : -3 = b - 6) :
  (a, b) = (2, 3) :=
by
  sorry

end NUMINAMATH_GPT_translation_coordinates_l1465_146526


namespace NUMINAMATH_GPT_smallest_percent_increase_is_100_l1465_146556

-- The values for each question
def prize_values : List ℕ := [150, 300, 450, 900, 1800, 3600, 7200, 14400, 28800, 57600, 115200, 230400, 460800, 921600, 1843200]

-- Definition of percent increase calculation
def percent_increase (old new : ℕ) : ℕ :=
  ((new - old : ℕ) * 100) / old

-- Lean theorem statement
theorem smallest_percent_increase_is_100 :
  percent_increase (prize_values.get! 5) (prize_values.get! 6) = 100 ∧
  percent_increase (prize_values.get! 7) (prize_values.get! 8) = 100 ∧
  percent_increase (prize_values.get! 9) (prize_values.get! 10) = 100 ∧
  percent_increase (prize_values.get! 10) (prize_values.get! 11) = 100 ∧
  percent_increase (prize_values.get! 13) (prize_values.get! 14) = 100 :=
by
  sorry

end NUMINAMATH_GPT_smallest_percent_increase_is_100_l1465_146556


namespace NUMINAMATH_GPT_max_area_garden_l1465_146534

/-- Given a rectangular garden with a total perimeter of 480 feet and one side twice as long as another,
    prove that the maximum area of the garden is 12800 square feet. -/
theorem max_area_garden (l w : ℝ) (h1 : l = 2 * w) (h2 : 2 * l + 2 * w = 480) : l * w = 12800 := 
sorry

end NUMINAMATH_GPT_max_area_garden_l1465_146534


namespace NUMINAMATH_GPT_complement_of_A_in_U_l1465_146531

-- Conditions definitions
def U : Set ℕ := {x | x ≤ 5}
def A : Set ℕ := {x | 2 * x - 5 < 0}

-- Theorem stating the question and the correct answer
theorem complement_of_A_in_U :
  U \ A = {x | 3 ≤ x ∧ x ≤ 5} :=
by
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_complement_of_A_in_U_l1465_146531


namespace NUMINAMATH_GPT_triangle_expression_negative_l1465_146571

theorem triangle_expression_negative {a b c : ℝ} (habc : a > 0 ∧ b > 0 ∧ c > 0) (triangle_ineq1 : a + b > c) (triangle_ineq2 : a + c > b) (triangle_ineq3 : b + c > a) :
  a^2 + b^2 - c^2 - 2 * a * b < 0 :=
sorry

end NUMINAMATH_GPT_triangle_expression_negative_l1465_146571


namespace NUMINAMATH_GPT_sequence_sum_S15_S22_S31_l1465_146588

def sequence_sum (n : ℕ) : ℤ :=
  match n with
  | 0     => 0
  | m + 1 => sequence_sum m + (-1)^m * (3 * (m + 1) - 1)

theorem sequence_sum_S15_S22_S31 :
  sequence_sum 15 + sequence_sum 22 - sequence_sum 31 = -57 := 
sorry

end NUMINAMATH_GPT_sequence_sum_S15_S22_S31_l1465_146588


namespace NUMINAMATH_GPT_count_three_digit_values_with_double_sum_eq_six_l1465_146530

def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

def is_three_digit (x : ℕ) : Prop := 
  100 ≤ x ∧ x < 1000

theorem count_three_digit_values_with_double_sum_eq_six :
  ∃ count : ℕ, is_three_digit count ∧ (
    (∀ x, is_three_digit x → sum_of_digits (sum_of_digits x) = 6) ↔ count = 30
  ) :=
sorry

end NUMINAMATH_GPT_count_three_digit_values_with_double_sum_eq_six_l1465_146530


namespace NUMINAMATH_GPT_ratio_of_lengths_l1465_146595

theorem ratio_of_lengths (l1 l2 l3 : ℝ)
    (h1 : l2 = (1/2) * (l1 + l3))
    (h2 : l2^3 = (6/13) * (l1^3 + l3^3)) :
    l1 / l3 = 7 / 5 := by
  sorry

end NUMINAMATH_GPT_ratio_of_lengths_l1465_146595


namespace NUMINAMATH_GPT_average_of_w_x_z_eq_one_sixth_l1465_146539

open Real

variable {w x y z t : ℝ}

theorem average_of_w_x_z_eq_one_sixth
  (h1 : 3 / w + 3 / x + 3 / z = 3 / (y + t))
  (h2 : w * x * z = y + t)
  (h3 : w * z + x * t + y * z = 3 * w + 3 * x + 3 * z) :
  (w + x + z) / 3 = 1 / 6 :=
by 
  sorry

end NUMINAMATH_GPT_average_of_w_x_z_eq_one_sixth_l1465_146539


namespace NUMINAMATH_GPT_number_of_exchanges_l1465_146519

theorem number_of_exchanges (n : ℕ) (hz_initial : ℕ) (hl_initial : ℕ) 
  (hz_decrease : ℕ) (hl_decrease : ℕ) (k : ℕ) :
  hz_initial = 200 →
  hl_initial = 20 →
  hz_decrease = 6 →
  hl_decrease = 1 →
  k = 11 →
  (hz_initial - n * hz_decrease) = k * (hl_initial - n * hl_decrease) →
  n = 4 := 
sorry

end NUMINAMATH_GPT_number_of_exchanges_l1465_146519


namespace NUMINAMATH_GPT_other_number_is_286_l1465_146581

theorem other_number_is_286 (a b hcf lcm : ℕ) (h_hcf : hcf = 26) (h_lcm : lcm = 2310) (h_one_num : a = 210) 
  (rel : lcm * hcf = a * b) : b = 286 :=
by
  sorry

end NUMINAMATH_GPT_other_number_is_286_l1465_146581


namespace NUMINAMATH_GPT_three_digit_solutions_modulo_l1465_146518

def three_digit_positive_integers (x : ℕ) : Prop :=
  100 ≤ x ∧ x ≤ 999

theorem three_digit_solutions_modulo (h : ∃ x : ℕ, three_digit_positive_integers x ∧ 
  (2597 * x + 763) % 17 = 1459 % 17) : 
  ∃ (count : ℕ), count = 53 :=
by sorry

end NUMINAMATH_GPT_three_digit_solutions_modulo_l1465_146518


namespace NUMINAMATH_GPT_problem_1_problem_2_l1465_146523

noncomputable def a : ℝ := sorry
def m : ℝ := sorry
def n : ℝ := sorry
def k : ℝ := sorry

theorem problem_1 (h1 : a^m = 2) (h2 : a^n = 4) (h3 : a^k = 32) (h4 : a ≠ 0) : 
  a^(3*m + 2*n - k) = 4 := 
sorry

theorem problem_2 (h1 : a^m = 2) (h2 : a^n = 4) (h3 : a^k = 32) (h4 : a ≠ 0) : 
  k - 3*m - n = 0 := 
sorry

end NUMINAMATH_GPT_problem_1_problem_2_l1465_146523


namespace NUMINAMATH_GPT_lana_total_spending_l1465_146549

theorem lana_total_spending (ticket_price : ℕ) (tickets_friends : ℕ) (tickets_extra : ℕ)
  (H1 : ticket_price = 6)
  (H2 : tickets_friends = 8)
  (H3 : tickets_extra = 2) :
  ticket_price * (tickets_friends + tickets_extra) = 60 :=
by
  sorry

end NUMINAMATH_GPT_lana_total_spending_l1465_146549


namespace NUMINAMATH_GPT_bologna_sandwiches_l1465_146537

variable (C B P : ℕ)

theorem bologna_sandwiches (h1 : C = 1) (h2 : B = 7) (h3 : P = 8)
                          (h4 : C + B + P = 16) (h5 : 80 / 16 = 5) :
                          B * 5 = 35 :=
by
  -- omit the proof part
  sorry

end NUMINAMATH_GPT_bologna_sandwiches_l1465_146537


namespace NUMINAMATH_GPT_other_number_l1465_146578

theorem other_number (a b : ℝ) (h : a = 0.650) (h2 : a = b + 0.525) : b = 0.125 :=
sorry

end NUMINAMATH_GPT_other_number_l1465_146578


namespace NUMINAMATH_GPT_num_points_on_ellipse_with_area_l1465_146510

-- Define the line equation
def line_eq (x y : ℝ) : Prop := (x / 4) + (y / 3) = 1

-- Define the ellipse equation
def ellipse_eq (x y : ℝ) : Prop := (x^2) / 16 + (y^2) / 9 = 1

-- Define the area condition for the triangle
def area_condition (xA yA xB yB xP yP : ℝ) : Prop :=
  abs (xA * (yB - yP) + xB * (yP - yA) + xP * (yA - yB)) = 6

-- Define the main theorem statement
theorem num_points_on_ellipse_with_area (P : ℝ × ℝ) (A B : ℝ × ℝ) :
  ∃ P1 P2 : ℝ × ℝ, 
    (ellipse_eq P1.1 P1.2) ∧ 
    (ellipse_eq P2.1 P2.2) ∧ 
    (area_condition A.1 A.2 B.1 B.2 P1.1 P1.2) ∧ 
    (area_condition A.1 A.2 B.1 B.2 P2.1 P2.2) ∧ 
    P1 ≠ P2 := sorry

end NUMINAMATH_GPT_num_points_on_ellipse_with_area_l1465_146510


namespace NUMINAMATH_GPT_find_a_b_l1465_146554

noncomputable def f (x : ℝ) (a b : ℝ) := x^3 + a * x + b

theorem find_a_b 
  (a b : ℝ) 
  (h_tangent : ∀ x y, y = 2 * x - 5 → y = f 1 a b - 3) 
  : a = -1 ∧ b = -3 :=
by 
{
  sorry
}

end NUMINAMATH_GPT_find_a_b_l1465_146554


namespace NUMINAMATH_GPT_taxi_ride_cost_l1465_146574

noncomputable def fixed_cost : ℝ := 2.00
noncomputable def cost_per_mile : ℝ := 0.30
noncomputable def distance_traveled : ℝ := 8

theorem taxi_ride_cost :
  fixed_cost + (cost_per_mile * distance_traveled) = 4.40 := by
  sorry

end NUMINAMATH_GPT_taxi_ride_cost_l1465_146574


namespace NUMINAMATH_GPT_rest_days_in_1200_days_l1465_146567

noncomputable def rest_days_coinciding (n : ℕ) : ℕ :=
  if h : n > 0 then (n / 6) else 0

theorem rest_days_in_1200_days :
  rest_days_coinciding 1200 = 200 :=
by
  sorry

end NUMINAMATH_GPT_rest_days_in_1200_days_l1465_146567


namespace NUMINAMATH_GPT_imaginary_part_of_z_l1465_146522

open Complex

theorem imaginary_part_of_z :
  ∃ z: ℂ, (3 - 4 * I) * z = abs (4 + 3 * I) ∧ z.im = 4 / 5 :=
by
  sorry

end NUMINAMATH_GPT_imaginary_part_of_z_l1465_146522


namespace NUMINAMATH_GPT_second_smallest_packs_hot_dogs_l1465_146546

theorem second_smallest_packs_hot_dogs (n : ℕ) :
  (∃ k : ℕ, n = 5 * k + 3) →
  n > 0 →
  ∃ m : ℕ, m < n ∧ (∃ k2 : ℕ, m = 5 * k2 + 3) →
  n = 8 :=
by
  sorry

end NUMINAMATH_GPT_second_smallest_packs_hot_dogs_l1465_146546


namespace NUMINAMATH_GPT_problem_1_problem_2_l1465_146524

def f (x a : ℝ) : ℝ := abs (2 * x - a) + abs (2 * x + 3)
def g (x : ℝ) : ℝ := abs (2 * x - 3) + 2

theorem problem_1 (x : ℝ) :
  abs (g x) < 5 → 0 < x ∧ x < 3 :=
sorry

theorem problem_2 (a : ℝ) :
  (∀ x1 : ℝ, ∃ x2 : ℝ, f x1 a = g x2) →
  (a ≥ -1 ∨ a ≤ -5) :=
sorry

end NUMINAMATH_GPT_problem_1_problem_2_l1465_146524


namespace NUMINAMATH_GPT_number_of_days_worked_l1465_146516

theorem number_of_days_worked (total_toys_per_week : ℕ) (toys_per_day : ℕ) (h₁ : total_toys_per_week = 6000) (h₂ : toys_per_day = 1500) : (total_toys_per_week / toys_per_day) = 4 :=
by
  sorry

end NUMINAMATH_GPT_number_of_days_worked_l1465_146516


namespace NUMINAMATH_GPT_capacity_of_new_bucket_l1465_146506

def number_of_old_buckets : ℕ := 26
def capacity_of_old_bucket : ℝ := 13.5
def total_volume : ℝ := number_of_old_buckets * capacity_of_old_bucket
def number_of_new_buckets : ℕ := 39

theorem capacity_of_new_bucket :
  total_volume / number_of_new_buckets = 9 :=
sorry

end NUMINAMATH_GPT_capacity_of_new_bucket_l1465_146506


namespace NUMINAMATH_GPT_gcd_of_three_numbers_l1465_146562

theorem gcd_of_three_numbers : Nat.gcd 16434 (Nat.gcd 24651 43002) = 3 := by
  sorry

end NUMINAMATH_GPT_gcd_of_three_numbers_l1465_146562


namespace NUMINAMATH_GPT_totalCroissants_is_18_l1465_146538

def jorgeCroissants : ℕ := 7
def giulianaCroissants : ℕ := 5
def matteoCroissants : ℕ := 6

def totalCroissants : ℕ := jorgeCroissants + giulianaCroissants + matteoCroissants

theorem totalCroissants_is_18 : totalCroissants = 18 := by
  -- Proof will be provided here
  sorry

end NUMINAMATH_GPT_totalCroissants_is_18_l1465_146538


namespace NUMINAMATH_GPT_final_image_of_F_is_correct_l1465_146500

-- Define the initial F position as a struct
structure Position where
  base : (ℝ × ℝ)
  stem : (ℝ × ℝ)

-- Function to rotate a point 90 degrees counterclockwise around the origin
def rotate90 (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, p.1)

-- Function to reflect a point in the x-axis
def reflectX (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

-- Function to rotate a point by 180 degrees around the origin (half turn)
def rotate180 (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, -p.2)

-- Define the initial state of F
def initialFPosition : Position := {
  base := (-1, 0),  -- Base along the negative x-axis
  stem := (0, -1)   -- Stem along the negative y-axis
}

-- Perform all transformations on the Position of F
def transformFPosition (pos : Position) : Position :=
  let afterRotation90 := Position.mk (rotate90 pos.base) (rotate90 pos.stem)
  let afterReflectionX := Position.mk (reflectX afterRotation90.base) (reflectX afterRotation90.stem)
  let finalPosition := Position.mk (rotate180 afterReflectionX.base) (rotate180 afterReflectionX.stem)
  finalPosition

-- Define the target final position we expect
def finalFPosition : Position := {
  base := (0, 1),   -- Base along the positive y-axis
  stem := (1, 0)    -- Stem along the positive x-axis
}

-- The theorem statement: After the transformations, the position of F
-- should match the final expected position
theorem final_image_of_F_is_correct :
  transformFPosition initialFPosition = finalFPosition := by
  sorry

end NUMINAMATH_GPT_final_image_of_F_is_correct_l1465_146500


namespace NUMINAMATH_GPT_gravity_anomaly_l1465_146502

noncomputable def gravity_anomaly_acceleration
  (α : ℝ) (v₀ : ℝ) (g : ℝ) (S : ℝ) (g_a : ℝ) : Prop :=
  α = 30 ∧ v₀ = 10 ∧ g = 10 ∧ S = 3 * Real.sqrt 3 → g_a = 250

theorem gravity_anomaly (α v₀ g S g_a : ℝ) : gravity_anomaly_acceleration α v₀ g S g_a :=
by
  intro h
  sorry

end NUMINAMATH_GPT_gravity_anomaly_l1465_146502


namespace NUMINAMATH_GPT_sqrt_product_l1465_146520

open Real

theorem sqrt_product :
  sqrt 54 * sqrt 48 * sqrt 6 = 72 * sqrt 3 := by
  sorry

end NUMINAMATH_GPT_sqrt_product_l1465_146520


namespace NUMINAMATH_GPT_people_not_in_pool_l1465_146532

-- Define families and their members
def karen_donald_family : ℕ := 2 + 6
def tom_eva_family : ℕ := 2 + 4
def luna_aidan_family : ℕ := 2 + 5
def isabel_jake_family : ℕ := 2 + 3

-- Total number of people
def total_people : ℕ := karen_donald_family + tom_eva_family + luna_aidan_family + isabel_jake_family

-- Number of legs in the pool and people in the pool
def legs_in_pool : ℕ := 34
def people_in_pool : ℕ := legs_in_pool / 2

-- People not in the pool: people who went to store and went to bed
def store_people : ℕ := 2
def bed_people : ℕ := 3
def not_available_people : ℕ := store_people + bed_people

-- Prove (given conditions) number of people not in the pool
theorem people_not_in_pool : total_people - people_in_pool - not_available_people = 4 :=
by
  -- ...proof steps or "sorry"
  sorry

end NUMINAMATH_GPT_people_not_in_pool_l1465_146532


namespace NUMINAMATH_GPT_sum_c_2017_l1465_146521

def a (n : ℕ) : ℕ := 3 * n + 1

def b (n : ℕ) : ℕ := 4^(n-1)

def c (n : ℕ) : ℕ := if n = 1 then 7 else 3 * 4^(n-1)

theorem sum_c_2017 : (Finset.range 2017).sum c = 4^2017 + 3 :=
by
  -- definitions and required assumptions
  sorry

end NUMINAMATH_GPT_sum_c_2017_l1465_146521


namespace NUMINAMATH_GPT_largest_square_side_l1465_146513

theorem largest_square_side (width length : ℕ) (h_width : width = 63) (h_length : length = 42) : 
  Nat.gcd width length = 21 :=
by
  rw [h_width, h_length]
  sorry

end NUMINAMATH_GPT_largest_square_side_l1465_146513


namespace NUMINAMATH_GPT_parabola_constant_c_l1465_146594

theorem parabola_constant_c (b c : ℝ): 
  (∀ x : ℝ, y = x^2 + b * x + c) ∧ 
  (10 = 2^2 + b * 2 + c) ∧ 
  (31 = 4^2 + b * 4 + c) → 
  c = -3 :=
by
  sorry

end NUMINAMATH_GPT_parabola_constant_c_l1465_146594


namespace NUMINAMATH_GPT_probability_same_color_pair_l1465_146544

theorem probability_same_color_pair : 
  let total_shoes := 28
  let black_pairs := 8
  let brown_pairs := 4
  let gray_pairs := 2
  total_shoes = 2 * (black_pairs + brown_pairs + gray_pairs) → 
  ∃ (prob : ℚ), prob = 7 / 32 := by
  sorry

end NUMINAMATH_GPT_probability_same_color_pair_l1465_146544


namespace NUMINAMATH_GPT_solve_equation_l1465_146511

theorem solve_equation (x : ℝ) (hx : x ≠ 0 ∧ x ≠ -1) : (2 / x = 1 / (x + 1)) ↔ (x = -2) :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_equation_l1465_146511


namespace NUMINAMATH_GPT_sum_of_perimeters_l1465_146563

theorem sum_of_perimeters (a : ℝ) : 
    ∑' n : ℕ, (3 * a) * (1/3)^n = 9 * a / 2 :=
by sorry

end NUMINAMATH_GPT_sum_of_perimeters_l1465_146563


namespace NUMINAMATH_GPT_range_of_a_l1465_146570

theorem range_of_a (a : ℝ) : (∀ (x : ℝ), (a-2) * x^2 + 4 * (a-2) * x - 4 < 0) ↔ (1 < a ∧ a ≤ 2) :=
by sorry

end NUMINAMATH_GPT_range_of_a_l1465_146570


namespace NUMINAMATH_GPT_total_pages_in_book_l1465_146590

theorem total_pages_in_book 
    (pages_read : ℕ) (pages_left : ℕ) 
    (h₁ : pages_read = 11) 
    (h₂ : pages_left = 6) : 
    pages_read + pages_left = 17 := 
by 
    sorry

end NUMINAMATH_GPT_total_pages_in_book_l1465_146590


namespace NUMINAMATH_GPT_ratio_of_adults_to_children_closest_to_one_l1465_146589

theorem ratio_of_adults_to_children_closest_to_one (a c : ℕ) 
  (h₁ : 25 * a + 12 * c = 1950) 
  (h₂ : a ≥ 1) 
  (h₃ : c ≥ 1) : (a : ℚ) / (c : ℚ) = 27 / 25 := 
by 
  sorry

end NUMINAMATH_GPT_ratio_of_adults_to_children_closest_to_one_l1465_146589


namespace NUMINAMATH_GPT_line_passes_through_fixed_point_l1465_146592

theorem line_passes_through_fixed_point :
  ∀ (k : ℝ), (k + 1) * (-1) - (2 * k - 1) * (1) + 3 * k = 0 :=
by
  intro k
  sorry

end NUMINAMATH_GPT_line_passes_through_fixed_point_l1465_146592


namespace NUMINAMATH_GPT_range_of_k_l1465_146583

theorem range_of_k (x y k : ℝ) 
  (h1 : 2 * x + y = k + 1) 
  (h2 : x + 2 * y = 2) 
  (h3 : x + y < 0) : 
  k < -3 :=
sorry

end NUMINAMATH_GPT_range_of_k_l1465_146583


namespace NUMINAMATH_GPT_min_shift_sine_l1465_146540

theorem min_shift_sine (φ : ℝ) (hφ : φ > 0) :
    (∃ k : ℤ, 2 * φ + π / 3 = 2 * k * π) → φ = 5 * π / 6 :=
sorry

end NUMINAMATH_GPT_min_shift_sine_l1465_146540


namespace NUMINAMATH_GPT_positive_number_property_l1465_146528

theorem positive_number_property (x : ℝ) (h : x > 0) (hx : (x / 100) * x = 9) : x = 30 := by
  sorry

end NUMINAMATH_GPT_positive_number_property_l1465_146528


namespace NUMINAMATH_GPT_trajectory_of_moving_circle_l1465_146568

noncomputable def ellipse_trajectory_eq (x y : ℝ) : Prop :=
  (x^2)/25 + (y^2)/9 = 1

theorem trajectory_of_moving_circle
  (x y : ℝ)
  (A : ℝ × ℝ)
  (C : ℝ × ℝ)
  (radius_C : ℝ)
  (hC : (x + 4)^2 + y^2 = 100)
  (hA : A = (4, 0))
  (radius_C_eq : radius_C = 10) :
  ellipse_trajectory_eq x y :=
sorry

end NUMINAMATH_GPT_trajectory_of_moving_circle_l1465_146568


namespace NUMINAMATH_GPT_pasture_feeding_l1465_146559

-- The definitions corresponding to the given conditions
def portion_per_cow_per_day := 1

def food_needed (cows : ℕ) (days : ℕ) : ℕ := cows * days

def growth_rate (food10for20 : ℕ) (food15for10 : ℕ) (days10_20 : ℕ) : ℕ :=
  (food10for20 - food15for10) / days10_20

def food_growth_rate := growth_rate (food_needed 10 20) (food_needed 15 10) 10

def new_grass_feed_cows_per_day := food_growth_rate / portion_per_cow_per_day

def original_grass := (food_needed 10 20) - (food_growth_rate * 20)

def days_to_feed_30_cows := original_grass / (30 - new_grass_feed_cows_per_day)

-- The statement we want to prove
theorem pasture_feeding :
  new_grass_feed_cows_per_day = 5 ∧ days_to_feed_30_cows = 4 := by
  sorry

end NUMINAMATH_GPT_pasture_feeding_l1465_146559


namespace NUMINAMATH_GPT_sum_of_squares_l1465_146555

/-- 
Given two real numbers x and y, if their product is 120 and their sum is 23, 
then the sum of their squares is 289.
-/
theorem sum_of_squares (x y : ℝ) (h₁ : x * y = 120) (h₂ : x + y = 23) :
  x^2 + y^2 = 289 :=
sorry

end NUMINAMATH_GPT_sum_of_squares_l1465_146555


namespace NUMINAMATH_GPT_power_equation_value_l1465_146505

theorem power_equation_value (n : ℕ) (h : n = 20) : n ^ (n / 2) = 102400000000000000000 := by
  sorry

end NUMINAMATH_GPT_power_equation_value_l1465_146505


namespace NUMINAMATH_GPT_train_speed_l1465_146586

theorem train_speed
  (length_of_train : ℝ) 
  (time_to_cross : ℝ) 
  (train_length_is_140 : length_of_train = 140)
  (time_is_6 : time_to_cross = 6) :
  (length_of_train / time_to_cross) = 23.33 :=
sorry

end NUMINAMATH_GPT_train_speed_l1465_146586


namespace NUMINAMATH_GPT_profit_percent_is_25_l1465_146509

noncomputable def SP : ℝ := sorry
noncomputable def CP : ℝ := 0.80 * SP
noncomputable def Profit : ℝ := SP - CP
noncomputable def ProfitPercent : ℝ := (Profit / CP) * 100

theorem profit_percent_is_25 :
  ProfitPercent = 25 :=
by
  sorry

end NUMINAMATH_GPT_profit_percent_is_25_l1465_146509


namespace NUMINAMATH_GPT_inequality_chain_l1465_146508

theorem inequality_chain (a : ℝ) (h : a - 1 > 0) : -a < -1 ∧ -1 < 1 ∧ 1 < a := by
  sorry

end NUMINAMATH_GPT_inequality_chain_l1465_146508


namespace NUMINAMATH_GPT_survivor_probability_l1465_146552

noncomputable def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem survivor_probability :
  let total_people := 20
  let tribe_size := 10
  let droppers := 3
  let total_ways := choose total_people droppers
  let tribe_ways := choose tribe_size droppers
  let same_tribe_ways := 2 * tribe_ways
  let probability := same_tribe_ways / total_ways
  probability = 20 / 95 :=
by
  let total_people := 20
  let tribe_size := 10
  let droppers := 3
  let total_ways := choose total_people droppers
  let tribe_ways := choose tribe_size droppers
  let same_tribe_ways := 2 * tribe_ways
  let probability := same_tribe_ways / total_ways
  have : probability = 20 / 95 := sorry
  exact this

end NUMINAMATH_GPT_survivor_probability_l1465_146552


namespace NUMINAMATH_GPT_binomial_coeff_sum_l1465_146527

-- Define the problem: compute the numerical sum of the binomial coefficients
theorem binomial_coeff_sum (a b : ℕ) (h_a1 : a = 1) (h_b1 : b = 1) : 
  (a + b) ^ 8 = 256 :=
by
  -- Therefore, the sum must be 256
  sorry

end NUMINAMATH_GPT_binomial_coeff_sum_l1465_146527


namespace NUMINAMATH_GPT_ab_value_l1465_146507

theorem ab_value (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 30) (h4 : 3 * a * b + 5 * a = 4 * b + 180) : a * b = 29 :=
sorry

end NUMINAMATH_GPT_ab_value_l1465_146507


namespace NUMINAMATH_GPT_tan_of_cos_l1465_146517

theorem tan_of_cos (α : ℝ) (h_cos : Real.cos α = -4 / 5) (h_alpha : 0 < α ∧ α < Real.pi) : 
  Real.tan α = -3 / 4 :=
sorry

end NUMINAMATH_GPT_tan_of_cos_l1465_146517


namespace NUMINAMATH_GPT_ratio_first_term_l1465_146580

theorem ratio_first_term (x : ℝ) (h1 : 60 / 100 = x / 25) : x = 15 := 
sorry

end NUMINAMATH_GPT_ratio_first_term_l1465_146580


namespace NUMINAMATH_GPT_vector_magnitude_problem_l1465_146572

open Real

noncomputable def magnitude (x : ℝ × ℝ) : ℝ := sqrt (x.1 ^ 2 + x.2 ^ 2)

theorem vector_magnitude_problem (a b : ℝ × ℝ) 
  (h_a : a = (1, 3))
  (h_perp : (a.1 + b.1, a.2 + b.2) • (a.1 - b.1, a.2 - b.2) = 0) :
  magnitude b = sqrt 10 := 
sorry

end NUMINAMATH_GPT_vector_magnitude_problem_l1465_146572


namespace NUMINAMATH_GPT_probability_one_each_l1465_146582

-- Define the counts of letters
def total_letters : ℕ := 11
def cybil_count : ℕ := 5
def ronda_count : ℕ := 5
def andy_initial_count : ℕ := 1

-- Define the probability calculation
def probability_one_from_cybil_and_one_from_ronda : ℚ :=
  (cybil_count / total_letters) * (ronda_count / (total_letters - 1)) +
  (ronda_count / total_letters) * (cybil_count / (total_letters - 1))

theorem probability_one_each (total_letters cybil_count ronda_count andy_initial_count : ℕ) :
  probability_one_from_cybil_and_one_from_ronda = 5 / 11 := sorry

end NUMINAMATH_GPT_probability_one_each_l1465_146582


namespace NUMINAMATH_GPT_find_incorrect_expression_l1465_146560

variable {x y : ℚ}

theorem find_incorrect_expression
  (h : x / y = 5 / 6) :
  ¬ (
    (x + 3 * y) / x = 23 / 5
  ) := by
  sorry

end NUMINAMATH_GPT_find_incorrect_expression_l1465_146560


namespace NUMINAMATH_GPT_sin_225_cos_225_l1465_146543

noncomputable def sin_225_eq_neg_sqrt2_div_2 : Prop :=
  Real.sin (225 * Real.pi / 180) = -Real.sqrt 2 / 2

noncomputable def cos_225_eq_neg_sqrt2_div_2 : Prop :=
  Real.cos (225 * Real.pi / 180) = -Real.sqrt 2 / 2

theorem sin_225 : sin_225_eq_neg_sqrt2_div_2 := by
  sorry

theorem cos_225 : cos_225_eq_neg_sqrt2_div_2 := by
  sorry

end NUMINAMATH_GPT_sin_225_cos_225_l1465_146543


namespace NUMINAMATH_GPT_infinitely_many_MTRP_numbers_l1465_146547

def sum_of_digits (n : ℕ) : ℕ := 
n.digits 10 |>.sum

def is_MTRP_number (m n : ℕ) : Prop :=
  n % m = 1 ∧ sum_of_digits (n^2) ≥ sum_of_digits n

theorem infinitely_many_MTRP_numbers (m : ℕ) : 
  ∀ N : ℕ, ∃ n > N, is_MTRP_number m n :=
by sorry

end NUMINAMATH_GPT_infinitely_many_MTRP_numbers_l1465_146547


namespace NUMINAMATH_GPT_problem_solution_l1465_146529

noncomputable def verify_solution (x y z : ℝ) : Prop :=
  x = 12 ∧ y = 10 ∧ z = 8 →
  (x > 4) ∧ (y > 4) ∧ (z > 4) →
  ( ( (x + 3)^2 / (y + z - 3) ) + 
    ( (y + 5)^2 / (z + x - 5) ) + 
    ( (z + 7)^2 / (x + y - 7) ) = 45)

theorem problem_solution :
  verify_solution 12 10 8 := by
  sorry

end NUMINAMATH_GPT_problem_solution_l1465_146529


namespace NUMINAMATH_GPT_cookies_with_flour_l1465_146550

theorem cookies_with_flour (x: ℕ) (c1: ℕ) (c2: ℕ) (h: c1 = 18 ∧ c2 = 2 ∧ x = 9 * 5):
  x = 45 :=
by
  obtain ⟨h1, h2, h3⟩ := h
  sorry

end NUMINAMATH_GPT_cookies_with_flour_l1465_146550


namespace NUMINAMATH_GPT_ratio_senior_junior_l1465_146557

theorem ratio_senior_junior
  (J S : ℕ)
  (h1 : ∃ k : ℕ, S = k * J)
  (h2 : (3 / 8) * S + (1 / 4) * J = (1 / 3) * (S + J)) :
  S = 2 * J :=
by
  -- The proof is to be provided
  sorry

end NUMINAMATH_GPT_ratio_senior_junior_l1465_146557


namespace NUMINAMATH_GPT_july_birth_percentage_l1465_146573

theorem july_birth_percentage (total : ℕ) (july : ℕ) (h1 : total = 150) (h2 : july = 18) : (july : ℚ) / total * 100 = 12 := sorry

end NUMINAMATH_GPT_july_birth_percentage_l1465_146573


namespace NUMINAMATH_GPT_inequality_example_l1465_146579

variable (a b : ℝ)

theorem inequality_example (h1 : a > 1/2) (h2 : b > 1/2) : a + 2 * b - 5 * a * b < 1/4 :=
by
  sorry

end NUMINAMATH_GPT_inequality_example_l1465_146579


namespace NUMINAMATH_GPT_sum_first_8_terms_l1465_146535

variable {α : Type*} [LinearOrderedField α]

-- Define the arithmetic sequence
def arithmetic_sequence (a_1 d : α) (n : ℕ) : α := a_1 + (n - 1) * d

-- Define the sum of the first n terms of the arithmetic sequence
def sum_arithmetic_sequence (a_1 d : α) (n : ℕ) : α :=
  (n * (2 * a_1 + (n - 1) * d)) / 2

-- Define the given condition
variable (a_1 d : α)
variable (h : arithmetic_sequence a_1 d 3 = 20 - arithmetic_sequence a_1 d 6)

-- Statement of the problem
theorem sum_first_8_terms : sum_arithmetic_sequence a_1 d 8 = 80 :=
by
  sorry

end NUMINAMATH_GPT_sum_first_8_terms_l1465_146535


namespace NUMINAMATH_GPT_sin_law_ratio_l1465_146536

theorem sin_law_ratio {A B C : ℝ} {a b c : ℝ} (hA : a = 1) (hSinA : Real.sin A = 1 / 3) :
  (a + b + c) / (Real.sin A + Real.sin B + Real.sin C) = 3 := 
  sorry

end NUMINAMATH_GPT_sin_law_ratio_l1465_146536


namespace NUMINAMATH_GPT_odd_function_decreasing_function_max_min_values_on_interval_l1465_146515

variable (f : ℝ → ℝ)

axiom func_additive : ∀ x y : ℝ, f (x + y) = f x + f y
axiom func_negative_for_positive : ∀ x : ℝ, (0 < x) → f x < 0
axiom func_value_at_one : f 1 = -2

theorem odd_function : ∀ x : ℝ, f (-x) = -f x := by
  have f_zero : f 0 = 0 := by sorry
  sorry

theorem decreasing_function : ∀ x₁ x₂ : ℝ, (x₁ < x₂) → f x₁ > f x₂ := by sorry

theorem max_min_values_on_interval :
  (f (-3) = 6) ∧ (f 3 = -6) := by sorry

end NUMINAMATH_GPT_odd_function_decreasing_function_max_min_values_on_interval_l1465_146515


namespace NUMINAMATH_GPT_complement_intersection_l1465_146542

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {2, 4}
def N : Set ℕ := {3, 5}

theorem complement_intersection (hU: U = {1, 2, 3, 4, 5}) (hM: M = {2, 4}) (hN: N = {3, 5}) : 
  (U \ M) ∩ N = {3, 5} := 
by 
  sorry

end NUMINAMATH_GPT_complement_intersection_l1465_146542


namespace NUMINAMATH_GPT_similar_triangles_l1465_146558

-- Define the similarity condition between the triangles
theorem similar_triangles (x : ℝ) (h₁ : 12 / x = 9 / 6) : x = 8 := 
by sorry

end NUMINAMATH_GPT_similar_triangles_l1465_146558


namespace NUMINAMATH_GPT_range_of_a_l1465_146541

theorem range_of_a (x y a : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (∀ (x y : ℝ), 0 < x → 0 < y → (y / 4 - (Real.cos x)^2) ≥ a * (Real.sin x) - 9 / y) ↔ (-3 ≤ a ∧ a ≤ 3) :=
sorry

end NUMINAMATH_GPT_range_of_a_l1465_146541


namespace NUMINAMATH_GPT_equation_relationship_linear_l1465_146587

theorem equation_relationship_linear 
  (x y : ℕ)
  (h1 : (x, y) = (0, 200) ∨ (x, y) = (1, 160) ∨ (x, y) = (2, 120) ∨ (x, y) = (3, 80) ∨ (x, y) = (4, 40)) :
  y = 200 - 40 * x :=
  sorry

end NUMINAMATH_GPT_equation_relationship_linear_l1465_146587


namespace NUMINAMATH_GPT_max_value_of_expression_l1465_146551

theorem max_value_of_expression (x y z : ℝ) (h₀ : x ≥ 0) (h₁ : y ≥ 0) (h₂ : z ≥ 0) (h₃ : x^2 + y^2 + z^2 = 1) : 
  3 * x * z * Real.sqrt 2 + 9 * y * z ≤ Real.sqrt 27 :=
sorry

end NUMINAMATH_GPT_max_value_of_expression_l1465_146551


namespace NUMINAMATH_GPT_find_smallest_c_plus_d_l1465_146599

noncomputable def smallest_c_plus_d (c d : ℝ) :=
  c + d

theorem find_smallest_c_plus_d (c d : ℝ) (hc : 0 < c) (hd : 0 < d)
  (h1 : c ^ 2 ≥ 12 * d)
  (h2 : 9 * d ^ 2 ≥ 4 * c) :
  smallest_c_plus_d c d = 16 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_smallest_c_plus_d_l1465_146599


namespace NUMINAMATH_GPT_more_birds_than_storks_l1465_146503

-- Defining the initial number of birds
def initial_birds : ℕ := 2

-- Defining the number of birds that joined
def additional_birds : ℕ := 5

-- Defining the number of storks that joined
def storks : ℕ := 4

-- Defining the total number of birds
def total_birds : ℕ := initial_birds + additional_birds

-- Defining the problem statement in Lean 4
theorem more_birds_than_storks : (total_birds - storks) = 3 := by
  sorry

end NUMINAMATH_GPT_more_birds_than_storks_l1465_146503

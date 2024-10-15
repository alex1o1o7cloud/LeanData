import Mathlib

namespace NUMINAMATH_GPT_count_three_digit_integers_with_remainder_3_div_7_l1640_164024

theorem count_three_digit_integers_with_remainder_3_div_7 :
  ∃ n, (100 ≤ 7 * n + 3 ∧ 7 * n + 3 < 1000) ∧
  ∀ m, (100 ≤ 7 * m + 3 ∧ 7 * m + 3 < 1000) → m - n < 142 - 14 + 1 :=
by
  sorry

end NUMINAMATH_GPT_count_three_digit_integers_with_remainder_3_div_7_l1640_164024


namespace NUMINAMATH_GPT_range_of_x_for_direct_above_inverse_l1640_164029

-- The conditions
def is_intersection_point (p : ℝ × ℝ) (k1 k2 : ℝ) : Prop :=
  let (x, y) := p
  y = k1 * x ∧ y = k2 / x

-- The main proof that we need to show
theorem range_of_x_for_direct_above_inverse :
  (∃ k1 k2 : ℝ, is_intersection_point (2, -1/3) k1 k2) →
  {x : ℝ | -1/6 * x > -2/(3 * x)} = {x : ℝ | x < -2 ∨ (0 < x ∧ x < 2)} :=
by
  intros
  sorry

end NUMINAMATH_GPT_range_of_x_for_direct_above_inverse_l1640_164029


namespace NUMINAMATH_GPT_multiply_powers_same_base_l1640_164043

theorem multiply_powers_same_base (a : ℝ) : a^3 * a = a^4 :=
by
  sorry

end NUMINAMATH_GPT_multiply_powers_same_base_l1640_164043


namespace NUMINAMATH_GPT_proof_complex_magnitude_z_l1640_164028

noncomputable def complex_magnitude_z : Prop :=
  ∀ (z : ℂ),
    (z * (Complex.cos (Real.pi / 9) + Complex.sin (Real.pi / 9) * Complex.I) ^ 6 = 2) →
    Complex.abs z = 2

theorem proof_complex_magnitude_z : complex_magnitude_z :=
by
  intros z h
  sorry

end NUMINAMATH_GPT_proof_complex_magnitude_z_l1640_164028


namespace NUMINAMATH_GPT_students_on_couch_per_room_l1640_164071

def total_students : ℕ := 30
def total_rooms : ℕ := 6
def students_per_bed : ℕ := 2
def beds_per_room : ℕ := 2
def students_in_beds_per_room : ℕ := beds_per_room * students_per_bed

theorem students_on_couch_per_room :
  (total_students / total_rooms) - students_in_beds_per_room = 1 := by
  sorry

end NUMINAMATH_GPT_students_on_couch_per_room_l1640_164071


namespace NUMINAMATH_GPT_problem_l1640_164093

theorem problem (a b c d e : ℤ) 
  (h1 : a - b + c - e = 7)
  (h2 : b - c + d + e = 8)
  (h3 : c - d + a - e = 4)
  (h4 : d - a + b + e = 3) :
  a + b + c + d + e = 22 := by
  sorry

end NUMINAMATH_GPT_problem_l1640_164093


namespace NUMINAMATH_GPT_general_term_min_value_S_n_l1640_164042

-- Definitions and conditions according to the problem statement
variable (d : ℤ) (a₁ : ℤ) (n : ℕ)

def a_n (n : ℕ) : ℤ := a₁ + (n - 1) * d
def S_n (n : ℕ) : ℤ := n * (2 * a₁ + (n - 1) * d) / 2

-- Given conditions
axiom positive_common_difference : 0 < d
axiom a3_a4_product : a_n 3 * a_n 4 = 117
axiom a2_a5_sum : a_n 2 + a_n 5 = -22

-- Proof 1: General term of the arithmetic sequence
theorem general_term : a_n n = 4 * (n : ℤ) - 25 :=
  by sorry

-- Proof 2: Minimum value of the sum of the first n terms
theorem min_value_S_n : S_n 6 = -66 :=
  by sorry

end NUMINAMATH_GPT_general_term_min_value_S_n_l1640_164042


namespace NUMINAMATH_GPT_largest_unique_k_l1640_164015

theorem largest_unique_k (n : ℕ) :
  (∀ k : ℤ, (8:ℚ)/15 < n / (n + k) ∧ n / (n + k) < 7/13 → False) ∧
  (∃ k : ℤ, (8:ℚ)/15 < n / (n + k) ∧ n / (n + k) < 7/13) → n = 112 :=
by sorry

end NUMINAMATH_GPT_largest_unique_k_l1640_164015


namespace NUMINAMATH_GPT_algebraic_expression_value_l1640_164036

theorem algebraic_expression_value (a : ℝ) (h : a^2 - 4 * a + 3 = 0) : -2 * a^2 + 8 * a - 5 = 1 := 
by 
  sorry 

end NUMINAMATH_GPT_algebraic_expression_value_l1640_164036


namespace NUMINAMATH_GPT_male_students_outnumber_female_students_l1640_164030

-- Define the given conditions
def total_students : ℕ := 928
def male_students : ℕ := 713
def female_students : ℕ := total_students - male_students

-- The theorem to be proven
theorem male_students_outnumber_female_students :
  male_students - female_students = 498 :=
by
  sorry

end NUMINAMATH_GPT_male_students_outnumber_female_students_l1640_164030


namespace NUMINAMATH_GPT_tan_product_pi_8_l1640_164038

theorem tan_product_pi_8 :
  (Real.tan (π / 8)) * (Real.tan (3 * π / 8)) * (Real.tan (5 * π / 8)) * (Real.tan (7 * π / 8)) = 1 :=
sorry

end NUMINAMATH_GPT_tan_product_pi_8_l1640_164038


namespace NUMINAMATH_GPT_minimum_sticks_broken_n12_can_form_square_n15_l1640_164033

-- Define the total length function
def total_length (n : ℕ) : ℕ := (n * (n + 1)) / 2

-- For n = 12, prove that at least 2 sticks need to be broken to form a square
theorem minimum_sticks_broken_n12 : ∀ (n : ℕ), n = 12 → total_length n % 4 ≠ 0 → 2 = 2 := 
by 
  intros n h1 h2
  sorry

-- For n = 15, prove that a square can be directly formed
theorem can_form_square_n15 : ∀ (n : ℕ), n = 15 → total_length n % 4 = 0 := 
by 
  intros n h1
  sorry

end NUMINAMATH_GPT_minimum_sticks_broken_n12_can_form_square_n15_l1640_164033


namespace NUMINAMATH_GPT_max_value_of_expression_l1640_164069

theorem max_value_of_expression (x y z : ℝ) (h : 3 * x + 4 * y + 2 * z = 12) :
  x^2 * y + x^2 * z + y * z^2 ≤ 3 := sorry

end NUMINAMATH_GPT_max_value_of_expression_l1640_164069


namespace NUMINAMATH_GPT_chickens_pigs_legs_l1640_164045

variable (x : ℕ)

-- Define the conditions
def sum_chickens_pigs (x : ℕ) : Prop := x + (70 - x) = 70
def total_legs (x : ℕ) : Prop := 2 * x + 4 * (70 - x) = 196

-- Main theorem to prove the given mathematical statement
theorem chickens_pigs_legs (x : ℕ) (h1 : sum_chickens_pigs x) (h2 : total_legs x) : (2 * x + 4 * (70 - x) = 196) :=
by sorry

end NUMINAMATH_GPT_chickens_pigs_legs_l1640_164045


namespace NUMINAMATH_GPT_fewerEmployeesAbroadThanInKorea_l1640_164049

def totalEmployees : Nat := 928
def employeesInKorea : Nat := 713
def employeesAbroad : Nat := totalEmployees - employeesInKorea

theorem fewerEmployeesAbroadThanInKorea :
  employeesInKorea - employeesAbroad = 498 :=
by
  sorry

end NUMINAMATH_GPT_fewerEmployeesAbroadThanInKorea_l1640_164049


namespace NUMINAMATH_GPT_vicentes_total_cost_l1640_164085

def total_cost (rice_bought cost_per_kg_rice meat_bought cost_per_lb_meat : Nat) : Nat :=
  (rice_bought * cost_per_kg_rice) + (meat_bought * cost_per_lb_meat)

theorem vicentes_total_cost :
  let rice_bought := 5
  let cost_per_kg_rice := 2
  let meat_bought := 3
  let cost_per_lb_meat := 5
  total_cost rice_bought cost_per_kg_rice meat_bought cost_per_lb_meat = 25 :=
by
  intros
  sorry

end NUMINAMATH_GPT_vicentes_total_cost_l1640_164085


namespace NUMINAMATH_GPT_length_of_AB_l1640_164086

-- Define the ellipse equation
def ellipse (x y : ℝ) : Prop := (x^2) / 25 + (y^2) / 16 = 1

-- Define the line perpendicular to the x-axis passing through the right focus of the ellipse
def line_perpendicular_y_axis_through_focus (y : ℝ) : Prop := true

-- Define the right focus of the ellipse
def right_focus : ℝ × ℝ := (3, 0)

-- Statement to prove the length of the line segment AB
theorem length_of_AB : 
  ∃ A B : ℝ × ℝ, 
  (ellipse A.1 A.2 ∧ ellipse B.1 B.2) ∧ 
  (A.1 = 3 ∧ B.1 = 3) ∧
  (|A.2 - B.2| = 2 * 16 / 5) :=
sorry

end NUMINAMATH_GPT_length_of_AB_l1640_164086


namespace NUMINAMATH_GPT_transformed_area_l1640_164075

noncomputable def area_transformation (f : ℝ → ℝ) (x1 x2 x3 : ℝ)
  (h : (1 / 2 * ((x2 - x1) * ((3 * f x3) - (3 * f x1))) - 1 / 2 * ((x3 - x2) * ((3 * f x1) - (3 * f x2)))) = 27) : Prop :=
  1 / 2 * ((0.5 * x2 - 0.5 * x1) * (3 * f (2 * x3) - 3 * f (2 * x1)) - 1 / 2 * (0.5 * x3 - 0.5 * x2) * (3 * f (2 * x1) - 3 * f (2 * x2))) = 40.5

theorem transformed_area
  (f : ℝ → ℝ) (x1 x2 x3 : ℝ)
  (h : 1 / 2 * ((x2 - x1) * (f x3 - f x1) - (x3 - x2) * (f x1 - f x2)) = 27) :
  1 / 2 * ((0.5 * x2 - 0.5 * x1) * (3 * f (2 * x3) - 3 * f (2 * x1)) - 1 / 2 * (0.5 * x3 - 0.5 * x2) * (3 * f (2 * x1) - 3 * f (2 * x2))) = 40.5 := sorry

end NUMINAMATH_GPT_transformed_area_l1640_164075


namespace NUMINAMATH_GPT_product_of_two_numbers_l1640_164002

variable (x y : ℝ)

-- conditions
def condition1 : Prop := x + y = 23
def condition2 : Prop := x - y = 7

-- target
theorem product_of_two_numbers {x y : ℝ} 
  (h1 : condition1 x y) 
  (h2 : condition2 x y) : 
  x * y = 120 := 
sorry

end NUMINAMATH_GPT_product_of_two_numbers_l1640_164002


namespace NUMINAMATH_GPT_candidate_lost_by_1650_votes_l1640_164003

theorem candidate_lost_by_1650_votes (total_votes : ℕ) (pct_candidate : ℝ) (pct_rival : ℝ) : 
  total_votes = 5500 → 
  pct_candidate = 0.35 → 
  pct_rival = 0.65 → 
  ((pct_rival * total_votes) - (pct_candidate * total_votes)) = 1650 := 
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_candidate_lost_by_1650_votes_l1640_164003


namespace NUMINAMATH_GPT_daily_earnings_from_oil_refining_l1640_164025

-- Definitions based on conditions
def daily_earnings_from_mining : ℝ := 3000000
def monthly_expenses : ℝ := 30000000
def fine : ℝ := 25600000
def profit_percentage : ℝ := 0.01
def months_in_year : ℝ := 12
def days_in_month : ℝ := 30

-- The question translated as a Lean theorem statement
theorem daily_earnings_from_oil_refining : ∃ O : ℝ, O = 5111111.11 ∧ 
  fine = profit_percentage * months_in_year * 
    (days_in_month * (daily_earnings_from_mining + O) - monthly_expenses) :=
sorry

end NUMINAMATH_GPT_daily_earnings_from_oil_refining_l1640_164025


namespace NUMINAMATH_GPT_max_acceptable_ages_l1640_164084

noncomputable def acceptable_ages (avg_age std_dev : ℕ) : ℕ :=
  let lower_limit := avg_age - 2 * std_dev
  let upper_limit := avg_age + 2 * std_dev
  upper_limit - lower_limit + 1

theorem max_acceptable_ages : acceptable_ages 40 10 = 41 :=
by
  sorry

end NUMINAMATH_GPT_max_acceptable_ages_l1640_164084


namespace NUMINAMATH_GPT_problem_solution_l1640_164017

noncomputable def circle_constant : ℝ := Real.pi
noncomputable def natural_base : ℝ := Real.exp 1

theorem problem_solution (π : ℝ) (e : ℝ) (h₁ : π = Real.pi) (h₂ : e = Real.exp 1) :
  π * Real.log e / Real.log 3 > 3 * Real.log e / Real.log π := by
  sorry

end NUMINAMATH_GPT_problem_solution_l1640_164017


namespace NUMINAMATH_GPT_jane_mean_score_l1640_164099

def quiz_scores : List ℕ := [85, 90, 95, 80, 100]

def total_scores : ℕ := quiz_scores.length

def sum_scores : ℕ := quiz_scores.sum

def mean_score : ℕ := sum_scores / total_scores

theorem jane_mean_score : mean_score = 90 := by
  sorry

end NUMINAMATH_GPT_jane_mean_score_l1640_164099


namespace NUMINAMATH_GPT_average_price_of_pig_l1640_164064

theorem average_price_of_pig :
  ∀ (total_cost total_cost_hens total_cost_pigs : ℕ) (num_hens num_pigs avg_price_hen avg_price_pig : ℕ),
  num_hens = 10 →
  num_pigs = 3 →
  total_cost = 1200 →
  avg_price_hen = 30 →
  total_cost_hens = num_hens * avg_price_hen →
  total_cost_pigs = total_cost - total_cost_hens →
  avg_price_pig = total_cost_pigs / num_pigs →
  avg_price_pig = 300 :=
by
  intros total_cost total_cost_hens total_cost_pigs num_hens num_pigs avg_price_hen avg_price_pig h_num_hens h_num_pigs h_total_cost h_avg_price_hen h_total_cost_hens h_total_cost_pigs h_avg_price_pig
  sorry

end NUMINAMATH_GPT_average_price_of_pig_l1640_164064


namespace NUMINAMATH_GPT_least_number_divisible_by_12_leaves_remainder_4_is_40_l1640_164091

theorem least_number_divisible_by_12_leaves_remainder_4_is_40 :
  ∃ n : ℕ, (∀ k : ℕ, n = 12 * k + 4) ∧ (∀ m : ℕ, (∀ k : ℕ, m = 12 * k + 4) → n ≤ m) ∧ n = 40 :=
by
  sorry

end NUMINAMATH_GPT_least_number_divisible_by_12_leaves_remainder_4_is_40_l1640_164091


namespace NUMINAMATH_GPT_hole_digging_problem_l1640_164034

theorem hole_digging_problem
  (total_distance : ℕ)
  (original_interval : ℕ)
  (new_interval : ℕ)
  (original_holes : ℕ)
  (new_holes : ℕ)
  (lcm_interval : ℕ)
  (common_holes : ℕ)
  (new_holes_to_be_dug : ℕ)
  (original_holes_discarded : ℕ)
  (h1 : total_distance = 3000)
  (h2 : original_interval = 50)
  (h3 : new_interval = 60)
  (h4 : original_holes = total_distance / original_interval + 1)
  (h5 : new_holes = total_distance / new_interval + 1)
  (h6 : lcm_interval = Nat.lcm original_interval new_interval)
  (h7 : common_holes = total_distance / lcm_interval + 1)
  (h8 : new_holes_to_be_dug = new_holes - common_holes)
  (h9 : original_holes_discarded = original_holes - common_holes) :
  new_holes_to_be_dug = 40 ∧ original_holes_discarded = 50 :=
sorry

end NUMINAMATH_GPT_hole_digging_problem_l1640_164034


namespace NUMINAMATH_GPT_Sandy_age_l1640_164052

variable (S M : ℕ)

def condition1 (S M : ℕ) : Prop := M = S + 18
def condition2 (S M : ℕ) : Prop := S * 9 = M * 7

theorem Sandy_age (h1 : condition1 S M) (h2 : condition2 S M) : S = 63 := sorry

end NUMINAMATH_GPT_Sandy_age_l1640_164052


namespace NUMINAMATH_GPT_find_x_l1640_164001

theorem find_x (y z : ℚ) (h1 : z = 80) (h2 : y = z / 4) (h3 : x = y / 3) : x = 20 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1640_164001


namespace NUMINAMATH_GPT_smallest_base_l1640_164061

theorem smallest_base : ∃ b : ℕ, (b^2 ≤ 120 ∧ 120 < b^3) ∧ ∀ n : ℕ, (n^2 ≤ 120 ∧ 120 < n^3) → b ≤ n :=
by sorry

end NUMINAMATH_GPT_smallest_base_l1640_164061


namespace NUMINAMATH_GPT_cube_inequality_sufficient_and_necessary_l1640_164050

theorem cube_inequality_sufficient_and_necessary (a b : ℝ) :
  (a > b ↔ a^3 > b^3) := 
sorry

end NUMINAMATH_GPT_cube_inequality_sufficient_and_necessary_l1640_164050


namespace NUMINAMATH_GPT_james_tylenol_daily_intake_l1640_164087

def tylenol_per_tablet : ℕ := 375
def tablets_per_dose : ℕ := 2
def hours_per_dose : ℕ := 6
def hours_per_day : ℕ := 24

theorem james_tylenol_daily_intake :
  (hours_per_day / hours_per_dose) * (tablets_per_dose * tylenol_per_tablet) = 3000 := by
  sorry

end NUMINAMATH_GPT_james_tylenol_daily_intake_l1640_164087


namespace NUMINAMATH_GPT_total_slices_left_is_14_l1640_164004

-- Define the initial conditions
def large_pizza_slices : ℕ := 12
def small_pizza_slices : ℕ := 8
def hawaiian_pizza (num_large : ℕ) : ℕ := num_large * large_pizza_slices
def cheese_pizza (num_large : ℕ) : ℕ := num_large * large_pizza_slices
def pepperoni_pizza (num_small : ℕ) : ℕ := num_small * small_pizza_slices

-- Number of large pizzas ordered (Hawaiian and cheese)
def num_large_pizzas : ℕ := 2

-- Number of small pizzas received in promotion
def num_small_pizzas : ℕ := 1

-- Slices eaten by each person
def dean_slices (hawaiian_slices : ℕ) : ℕ := hawaiian_slices / 2
def frank_slices : ℕ := 3
def sammy_slices (cheese_slices : ℕ) : ℕ := cheese_slices / 3
def nancy_cheese_slices : ℕ := 2
def nancy_pepperoni_slice : ℕ := 1
def olivia_slices : ℕ := 2

-- Total slices eaten from each pizza
def total_hawaiian_slices_eaten (hawaiian_slices : ℕ) : ℕ := dean_slices hawaiian_slices + frank_slices
def total_cheese_slices_eaten (cheese_slices : ℕ) : ℕ := sammy_slices cheese_slices + nancy_cheese_slices
def total_pepperoni_slices_eaten : ℕ := nancy_pepperoni_slice + olivia_slices

-- Total slices left over
def total_slices_left (hawaiian_slices : ℕ) (cheese_slices : ℕ) (pepperoni_slices : ℕ) : ℕ := 
  (hawaiian_slices - total_hawaiian_slices_eaten hawaiian_slices) + 
  (cheese_slices - total_cheese_slices_eaten cheese_slices) + 
  (pepperoni_slices - total_pepperoni_slices_eaten)

-- The actual Lean 4 statement to be verified
theorem total_slices_left_is_14 : total_slices_left (hawaiian_pizza num_large_pizzas) (cheese_pizza num_large_pizzas) (pepperoni_pizza num_small_pizzas) = 14 := 
  sorry

end NUMINAMATH_GPT_total_slices_left_is_14_l1640_164004


namespace NUMINAMATH_GPT_find_teacher_age_l1640_164035

theorem find_teacher_age (S T : ℕ) (h1 : S / 19 = 20) (h2 : (S + T) / 20 = 21) : T = 40 :=
sorry

end NUMINAMATH_GPT_find_teacher_age_l1640_164035


namespace NUMINAMATH_GPT_line_equation_l1640_164057

-- Given conditions
variables (k x x0 y y0 : ℝ)
variable (line_passes_through : ∀ x0 y0, y0 = k * x0 + l)
variable (M0 : (ℝ × ℝ))

-- Main statement we need to prove
theorem line_equation (k x x0 y y0 : ℝ) (M0 : (ℝ × ℝ)) (line_passes_through : ∀ x0 y0, y0 = k * x0 + l) :
  y - y0 = k * (x - x0) :=
sorry

end NUMINAMATH_GPT_line_equation_l1640_164057


namespace NUMINAMATH_GPT_union_of_A_and_B_l1640_164070

-- Define the sets A and B
def A := {x : ℝ | 0 < x ∧ x < 16}
def B := {y : ℝ | -1 < y ∧ y < 4}

-- Prove that A ∪ B = (-1, 16)
theorem union_of_A_and_B : A ∪ B = {z : ℝ | -1 < z ∧ z < 16} :=
by sorry

end NUMINAMATH_GPT_union_of_A_and_B_l1640_164070


namespace NUMINAMATH_GPT_native_answer_l1640_164076

-- Define properties to represent native types
inductive NativeType
| normal
| zombie
| half_zombie

-- Define the function that determines the response of a native
def response (native : NativeType) : String :=
  match native with
  | NativeType.normal => "да"
  | NativeType.zombie => "да"
  | NativeType.half_zombie => "да"

-- Define the main theorem
theorem native_answer (native : NativeType) : response native = "да" :=
by sorry

end NUMINAMATH_GPT_native_answer_l1640_164076


namespace NUMINAMATH_GPT_circle_center_and_radius_locus_of_midpoint_l1640_164041

-- Part 1: Prove the equation of the circle C:
theorem circle_center_and_radius (a b r: ℝ) (hc: a + b = 2):
  (4 - a)^2 + b^2 = r^2 →
  (2 - a)^2 + (2 - b)^2 = r^2 →
  a = 2 ∧ b = 0 ∧ r = 2 := by
  sorry

-- Part 2: Prove the locus of the midpoint M:
theorem locus_of_midpoint (x y : ℝ) :
  ∃ (x1 y1 : ℝ), (x1 - 2)^2 + y1^2 = 4 ∧ x = (x1 + 5) / 2 ∧ y = y1 / 2 →
  x^2 - 7*x + y^2 + 45/4 = 0 := by
  sorry

end NUMINAMATH_GPT_circle_center_and_radius_locus_of_midpoint_l1640_164041


namespace NUMINAMATH_GPT_smallest_positive_integer_l1640_164096

theorem smallest_positive_integer (k : ℕ) :
  (∃ k : ℕ, ((2^4 ∣ 1452 * k) ∧ (3^3 ∣ 1452 * k) ∧ (13^3 ∣ 1452 * k))) → 
  k = 676 := 
sorry

end NUMINAMATH_GPT_smallest_positive_integer_l1640_164096


namespace NUMINAMATH_GPT_divide_54_degree_angle_l1640_164089

theorem divide_54_degree_angle :
  ∃ (angle_div : ℝ), angle_div = 54 / 3 :=
by
  sorry

end NUMINAMATH_GPT_divide_54_degree_angle_l1640_164089


namespace NUMINAMATH_GPT_quadratic_roots_equal_l1640_164010

theorem quadratic_roots_equal (m : ℝ) :
  (∃ x : ℝ, x^2 - 4 * x + m - 1 = 0 ∧ (∀ y : ℝ, y^2 - 4*y + m-1 = 0 → y = x)) ↔ (m = 5 ∧ (∀ x, x^2 - 4 * x + 4 = 0 ↔ x = 2)) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_roots_equal_l1640_164010


namespace NUMINAMATH_GPT_sqrt_of_16_l1640_164047

theorem sqrt_of_16 : Real.sqrt 16 = 4 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_of_16_l1640_164047


namespace NUMINAMATH_GPT_pineapple_total_cost_correct_l1640_164082

-- Define the conditions
def pineapple_cost : ℝ := 1.25
def num_pineapples : ℕ := 12
def shipping_cost : ℝ := 21.00

-- Calculate total cost
noncomputable def total_pineapple_cost : ℝ := pineapple_cost * num_pineapples
noncomputable def total_cost : ℝ := total_pineapple_cost + shipping_cost
noncomputable def cost_per_pineapple : ℝ := total_cost / num_pineapples

-- The proof problem
theorem pineapple_total_cost_correct : cost_per_pineapple = 3 := by
  -- The proof will be filled in here
  sorry

end NUMINAMATH_GPT_pineapple_total_cost_correct_l1640_164082


namespace NUMINAMATH_GPT_find_m_n_l1640_164090

theorem find_m_n : ∃ (m n : ℕ), 2^n + 1 = m^2 ∧ m = 3 ∧ n = 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_m_n_l1640_164090


namespace NUMINAMATH_GPT_team_a_vs_team_b_l1640_164014

noncomputable def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem team_a_vs_team_b (P1 P2 : ℝ) :
  let n_a := 5
  let x_a := 4
  let p_a := 0.5
  let n_b := 5
  let x_b := 3
  let p_b := 1/3
  let P1 := binomial_probability n_a x_a p_a
  let P2 := binomial_probability n_b x_b p_b
  P1 < P2 := by sorry

end NUMINAMATH_GPT_team_a_vs_team_b_l1640_164014


namespace NUMINAMATH_GPT_total_crayons_l1640_164044

theorem total_crayons (crayons_per_child : ℕ) (number_of_children : ℕ) (h1 : crayons_per_child = 3) (h2 : number_of_children = 6) : 
  crayons_per_child * number_of_children = 18 := by
  sorry

end NUMINAMATH_GPT_total_crayons_l1640_164044


namespace NUMINAMATH_GPT_integer_roots_condition_l1640_164039

theorem integer_roots_condition (n : ℕ) (hn : n > 0) :
  (∃ x : ℤ, x^2 - 4 * x + n = 0) ↔ (n = 3 ∨ n = 4) := 
by
  sorry

end NUMINAMATH_GPT_integer_roots_condition_l1640_164039


namespace NUMINAMATH_GPT_three_digit_divisible_by_11_l1640_164022

theorem three_digit_divisible_by_11
  (x y z : ℕ) (h1 : y = x + z) : (100 * x + 10 * y + z) % 11 = 0 :=
by
  sorry

end NUMINAMATH_GPT_three_digit_divisible_by_11_l1640_164022


namespace NUMINAMATH_GPT_pair_opposites_example_l1640_164063

theorem pair_opposites_example :
  (-5)^2 = 25 ∧ -((5)^2) = -25 →
  (∀ a b : ℕ, (|-4|)^2 = 4^2 → 4^2 = 16 → |-4|^2 = 16) →
  (-3)^2 = 9 ∧ 3^2 = 9 →
  (-(|-2|)^2 = -4 ∧ -2^2 = -4) →
  25 = -(-25) :=
by
  sorry

end NUMINAMATH_GPT_pair_opposites_example_l1640_164063


namespace NUMINAMATH_GPT_right_triangle_construction_condition_l1640_164072

theorem right_triangle_construction_condition
  (b s : ℝ) 
  (h_b_pos : b > 0)
  (h_s_pos : s > 0)
  (h_perimeter : ∃ (AC BC AB : ℝ), AC = b ∧ AC + BC + AB = 2 * s ∧ (AC^2 + BC^2 = AB^2)) :
  b < s := 
sorry

end NUMINAMATH_GPT_right_triangle_construction_condition_l1640_164072


namespace NUMINAMATH_GPT_cost_of_orchestra_seat_l1640_164008

-- Define the variables according to the conditions in the problem
def orchestra_ticket_count (y : ℕ) : Prop := (2 * y + 115 = 355)
def total_ticket_cost (x y : ℕ) : Prop := (120 * x + 235 * 8 = 3320)
def balcony_ticket_relation (y : ℕ) : Prop := (y + 115 = 355 - y)

-- Main theorem statement: Prove that the cost of a seat in the orchestra is 12 dollars
theorem cost_of_orchestra_seat : ∃ x y : ℕ, orchestra_ticket_count y ∧ total_ticket_cost x y ∧ (x = 12) :=
by sorry

end NUMINAMATH_GPT_cost_of_orchestra_seat_l1640_164008


namespace NUMINAMATH_GPT_initial_population_l1640_164053

theorem initial_population (rate_decrease : ℝ) (population_after_2_years : ℝ) (P : ℝ) : 
  rate_decrease = 0.1 → 
  population_after_2_years = 8100 → 
  ((1 - rate_decrease) ^ 2) * P = population_after_2_years → 
  P = 10000 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_initial_population_l1640_164053


namespace NUMINAMATH_GPT_area_of_square_field_l1640_164027

-- Define side length
def side_length : ℕ := 20

-- Theorem statement about the area of the square field
theorem area_of_square_field : (side_length * side_length) = 400 := by
  sorry

end NUMINAMATH_GPT_area_of_square_field_l1640_164027


namespace NUMINAMATH_GPT_necessary_condition_not_sufficient_condition_l1640_164007

variable (x : ℝ)

def quadratic_condition : Prop := x^2 - 3 * x + 2 > 0
def interval_condition : Prop := x < 1 ∨ x > 4

theorem necessary_condition : interval_condition x → quadratic_condition x := by sorry

theorem not_sufficient_condition : ¬ (quadratic_condition x → interval_condition x) := by sorry

end NUMINAMATH_GPT_necessary_condition_not_sufficient_condition_l1640_164007


namespace NUMINAMATH_GPT_problem_statement_l1640_164088

theorem problem_statement (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) 
  (h4 : x^3 + (1 / (y + 2016)) = y^3 + (1 / (z + 2016))) 
  (h5 : y^3 + (1 / (z + 2016)) = z^3 + (1 / (x + 2016))) : 
  (x^3 + y^3 + z^3) / (x * y * z) = 3 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1640_164088


namespace NUMINAMATH_GPT_function_passes_through_point_l1640_164074

noncomputable def func_graph (a : ℝ) (x : ℝ) : ℝ := a ^ (x - 1) + 2

theorem function_passes_through_point (a : ℝ) (h0 : a > 0) (h1 : a ≠ 1) :
  func_graph a 1 = 3 :=
by
  -- Proof logic is omitted
  sorry

end NUMINAMATH_GPT_function_passes_through_point_l1640_164074


namespace NUMINAMATH_GPT_molecular_weight_of_barium_iodide_l1640_164012

-- Define the atomic weights
def atomic_weight_of_ba : ℝ := 137.33
def atomic_weight_of_i : ℝ := 126.90

-- Define the molecular weight calculation for Barium iodide
def molecular_weight_of_bai2 : ℝ := atomic_weight_of_ba + 2 * atomic_weight_of_i

-- The main theorem to prove
theorem molecular_weight_of_barium_iodide : molecular_weight_of_bai2 = 391.13 := by
  -- we are given that atomic_weight_of_ba = 137.33 and atomic_weight_of_i = 126.90
  -- hence, molecular_weight_of_bai2 = 137.33 + 2 * 126.90
  -- simplifying this, we get
  -- molecular_weight_of_bai2 = 137.33 + 253.80 = 391.13
  sorry

end NUMINAMATH_GPT_molecular_weight_of_barium_iodide_l1640_164012


namespace NUMINAMATH_GPT_solve_z_l1640_164011

open Complex

theorem solve_z (z : ℂ) (h : z^2 = 3 - 4 * I) : z = 1 - 2 * I ∨ z = -1 + 2 * I :=
by
  sorry

end NUMINAMATH_GPT_solve_z_l1640_164011


namespace NUMINAMATH_GPT_feasible_test_for_rhombus_l1640_164098

def is_rhombus (paper : Type) : Prop :=
  true -- Placeholder for the actual definition of a rhombus

def method_A (paper : Type) : Prop :=
  -- Placeholder for the condition "Measure if the four internal angles are equal"
  true

def method_B (paper : Type) : Prop :=
  -- Placeholder for the condition "Measure if the two diagonals are equal"
  true

def method_C (paper : Type) : Prop :=
  -- Placeholder for the condition "Measure if the distance from the intersection of the two diagonals to the four vertices is equal"
  true

def method_D (paper : Type) : Prop :=
  -- Placeholder for the condition "Fold the paper along the two diagonals separately and see if the parts on both sides of the diagonals coincide completely each time"
  true

theorem feasible_test_for_rhombus (paper : Type) : is_rhombus paper → method_D paper :=
by
  intro h_rhombus
  sorry

end NUMINAMATH_GPT_feasible_test_for_rhombus_l1640_164098


namespace NUMINAMATH_GPT_distinct_remainders_sum_quotient_l1640_164078

theorem distinct_remainders_sum_quotient :
  let sq_mod_7 (n : Nat) := (n * n) % 7
  let distinct_remainders := List.eraseDup ([sq_mod_7 1, sq_mod_7 2, sq_mod_7 3, sq_mod_7 4, sq_mod_7 5])
  let s := List.sum distinct_remainders
  s / 7 = 1 :=
by
  sorry

end NUMINAMATH_GPT_distinct_remainders_sum_quotient_l1640_164078


namespace NUMINAMATH_GPT_represent_same_function_l1640_164019

noncomputable def f1 (x : ℝ) : ℝ := (x^3 + x) / (x^2 + 1)
def f2 (x : ℝ) : ℝ := x

theorem represent_same_function : ∀ x : ℝ, f1 x = f2 x := 
by
  sorry

end NUMINAMATH_GPT_represent_same_function_l1640_164019


namespace NUMINAMATH_GPT_sum_of_three_numbers_l1640_164059

theorem sum_of_three_numbers (x y z : ℝ) (h1 : x + y = 31) (h2 : y + z = 41) (h3 : z + x = 55) :
  x + y + z = 63.5 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_three_numbers_l1640_164059


namespace NUMINAMATH_GPT_polynomial_transformable_l1640_164066

theorem polynomial_transformable (a b c d : ℝ) :
  (∃ A B : ℝ, ∀ z : ℝ, z^4 + A * z^2 + B = (z + a/4)^4 + a * (z + a/4)^3 + b * (z + a/4)^2 + c * (z + a/4) + d) ↔ a^3 - 4 * a * b + 8 * c = 0 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_transformable_l1640_164066


namespace NUMINAMATH_GPT_total_days_on_island_correct_l1640_164013

-- Define the first, second, and third expeditions
def firstExpedition : ℕ := 3

def secondExpedition (a : ℕ) : ℕ := a + 2

def thirdExpedition (b : ℕ) : ℕ := 2 * b

-- Define the total duration in weeks
def totalWeeks : ℕ := firstExpedition + secondExpedition firstExpedition + thirdExpedition (secondExpedition firstExpedition)

-- Define the total days spent on the island
def totalDays (weeks : ℕ) : ℕ := weeks * 7

-- Prove that the total number of days spent is 126
theorem total_days_on_island_correct : totalDays totalWeeks = 126 := 
  by
    sorry

end NUMINAMATH_GPT_total_days_on_island_correct_l1640_164013


namespace NUMINAMATH_GPT_linear_increase_y_l1640_164051

-- Progressively increase x and track y

theorem linear_increase_y (Δx Δy : ℝ) (x_increase : Δx = 4) (y_increase : Δy = 10) :
  12 * (Δy / Δx) = 30 := by
  sorry

end NUMINAMATH_GPT_linear_increase_y_l1640_164051


namespace NUMINAMATH_GPT_range_of_m_l1640_164067

theorem range_of_m (m : ℝ) :
  (∀ x y : ℝ, (x^2 : ℝ) / (2 - m) + (y^2 : ℝ) / (m - 1) = 1 → 2 - m < 0 ∧ m - 1 > 0) →
  (∀ Δ : ℝ, Δ = 16 * (m - 2) ^ 2 - 16 → Δ < 0 → 1 < m ∧ m < 3) →
  (∀ (p q : Prop), p ∨ q ∧ ¬ q → p ∧ ¬ q) →
  m ≥ 3 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_range_of_m_l1640_164067


namespace NUMINAMATH_GPT_cyclist_speed_l1640_164092

theorem cyclist_speed (c d : ℕ) (h1 : d = c + 5) (hc : c ≠ 0) (hd : d ≠ 0)
    (H1 : ∀ tC tD : ℕ, 80 = c * tC → 120 = d * tD → tC = tD) : c = 10 := by
  sorry

end NUMINAMATH_GPT_cyclist_speed_l1640_164092


namespace NUMINAMATH_GPT_greatest_int_satisfying_inequality_l1640_164097

theorem greatest_int_satisfying_inequality : 
  ∃ m : ℤ, (∀ x : ℤ, x - 5 > 4 * x - 1 → x ≤ -2) ∧ (∀ k : ℤ, k < -2 → k - 5 > 4 * k - 1) :=
by
  sorry

end NUMINAMATH_GPT_greatest_int_satisfying_inequality_l1640_164097


namespace NUMINAMATH_GPT_student_departments_l1640_164073

variable {Student : Type}
variable (Anna Vika Masha : Student)

-- Let Department be an enumeration type representing the three departments
inductive Department
| Literature : Department
| History : Department
| Biology : Department

open Department

variables (isLit : Student → Prop) (isHist : Student → Prop) (isBio : Student → Prop)

-- Conditions
axiom cond1 : isLit Anna → ¬isHist Masha
axiom cond2 : ¬isHist Vika → isLit Anna
axiom cond3 : ¬isLit Masha → isBio Vika

-- Target conclusion
theorem student_departments :
  isHist Vika ∧ isLit Masha ∧ isBio Anna :=
sorry

end NUMINAMATH_GPT_student_departments_l1640_164073


namespace NUMINAMATH_GPT_general_term_l1640_164083

noncomputable def S (n : ℕ) : ℕ := sorry
noncomputable def a (n : ℕ) : ℕ := sorry

axiom S2 : S 2 = 4
axiom a_recurrence : ∀ n : ℕ, n > 0 → a (n + 1) = 2 * S n + 1

theorem general_term (n : ℕ) : a n = 3 ^ (n - 1) :=
by
  sorry

end NUMINAMATH_GPT_general_term_l1640_164083


namespace NUMINAMATH_GPT_chemistry_more_than_physics_l1640_164080

variables (M P C x : ℤ)

-- Condition 1: The total marks in mathematics and physics is 50
def condition1 : Prop := M + P = 50

-- Condition 2: The average marks in mathematics and chemistry together is 35
def condition2 : Prop := (M + C) / 2 = 35

-- Condition 3: The score in chemistry is some marks more than that in physics
def condition3 : Prop := C = P + x

theorem chemistry_more_than_physics :
  condition1 M P ∧ condition2 M C ∧ (∃ x : ℤ, condition3 P C x ∧ x = 20) :=
sorry

end NUMINAMATH_GPT_chemistry_more_than_physics_l1640_164080


namespace NUMINAMATH_GPT_box_weight_no_apples_l1640_164054

variable (initialWeight : ℕ) (halfWeight : ℕ) (totalWeight : ℕ)
variable (boxWeight : ℕ)

-- Given conditions
axiom initialWeight_def : initialWeight = 9
axiom halfWeight_def : halfWeight = 5
axiom appleWeight_consistent : ∃ w : ℕ, ∀ n : ℕ, n * w = totalWeight

-- Question: How many kilograms does the empty box weigh?
theorem box_weight_no_apples : (initialWeight - totalWeight) = boxWeight :=
by
  -- The proof steps are omitted as indicated by the 'sorry' placeholder.
  sorry

end NUMINAMATH_GPT_box_weight_no_apples_l1640_164054


namespace NUMINAMATH_GPT_center_radius_sum_l1640_164031

theorem center_radius_sum (a b r : ℝ) (h : ∀ x y : ℝ, (x^2 - 8*x - 4*y = -y^2 + 2*y + 13) ↔ (x - 4)^2 + (y - 3)^2 = 38) :
  a = 4 ∧ b = 3 ∧ r = Real.sqrt 38 → a + b + r = 7 + Real.sqrt 38 :=
by
  sorry

end NUMINAMATH_GPT_center_radius_sum_l1640_164031


namespace NUMINAMATH_GPT_Phoenix_roots_prod_l1640_164005

def Phoenix_eqn (a b c : ℝ) : Prop :=
  a ≠ 0 ∧ a + b + c = 0

theorem Phoenix_roots_prod {m n : ℝ} (hPhoenix : Phoenix_eqn 1 m n)
  (hEqualRoots : (m^2 - 4 * n) = 0) : m * n = -2 :=
by sorry

end NUMINAMATH_GPT_Phoenix_roots_prod_l1640_164005


namespace NUMINAMATH_GPT_total_yards_of_fabric_l1640_164020

theorem total_yards_of_fabric (cost_checkered : ℝ) (cost_plain : ℝ) (price_per_yard : ℝ)
  (h1 : cost_checkered = 75) (h2 : cost_plain = 45) (h3 : price_per_yard = 7.50) :
  (cost_checkered / price_per_yard) + (cost_plain / price_per_yard) = 16 := 
by
  sorry

end NUMINAMATH_GPT_total_yards_of_fabric_l1640_164020


namespace NUMINAMATH_GPT_digits_count_of_special_numbers_l1640_164095

theorem digits_count_of_special_numbers
  (n : ℕ)
  (h1 : 8^n = 28672) : n = 5 := 
by
  sorry

end NUMINAMATH_GPT_digits_count_of_special_numbers_l1640_164095


namespace NUMINAMATH_GPT_first_term_geometric_progression_l1640_164032

theorem first_term_geometric_progression (S : ℝ) (sum_first_two_terms : ℝ) (a : ℝ) (r : ℝ) :
  S = 8 → sum_first_two_terms = 5 →
  (a = 8 * (1 - (Real.sqrt 6) / 4)) ∨ (a = 8 * (1 + (Real.sqrt 6) / 4)) :=
by
  sorry

end NUMINAMATH_GPT_first_term_geometric_progression_l1640_164032


namespace NUMINAMATH_GPT_probability_of_three_blue_beans_l1640_164006

-- Define the conditions
def red_jellybeans : ℕ := 10 
def blue_jellybeans : ℕ := 10 
def total_jellybeans : ℕ := red_jellybeans + blue_jellybeans 
def draws : ℕ := 3 

-- Define the events
def P_first_blue : ℚ := blue_jellybeans / total_jellybeans 
def P_second_blue : ℚ := (blue_jellybeans - 1) / (total_jellybeans - 1) 
def P_third_blue : ℚ := (blue_jellybeans - 2) / (total_jellybeans - 2) 
def P_all_three_blue : ℚ := P_first_blue * P_second_blue * P_third_blue 

-- Define the correct answer
def correct_probability : ℚ := 1 / 9.5 

-- State the theorem
theorem probability_of_three_blue_beans : 
  P_all_three_blue = correct_probability := 
sorry

end NUMINAMATH_GPT_probability_of_three_blue_beans_l1640_164006


namespace NUMINAMATH_GPT_students_solved_both_l1640_164077

theorem students_solved_both (total_students solved_set_problem solved_function_problem both_problems_wrong: ℕ) 
  (h1: total_students = 50)
  (h2 : solved_set_problem = 40)
  (h3 : solved_function_problem = 31)
  (h4 : both_problems_wrong = 4) :
  (solved_set_problem + solved_function_problem - x + both_problems_wrong = total_students) → x = 25 := by
  sorry

end NUMINAMATH_GPT_students_solved_both_l1640_164077


namespace NUMINAMATH_GPT_midpoint_coords_l1640_164046

noncomputable def F1 : (ℝ × ℝ) := (-2 * Real.sqrt 2, 0)
noncomputable def F2 : (ℝ × ℝ) := (2 * Real.sqrt 2, 0)
def major_axis_length : ℝ := 6
def line_eq (x y : ℝ) : Prop := x - y + 2 = 0

noncomputable def ellipse_eq (x y : ℝ) : Prop :=
  let a := 3
  let b := 1
  (x^2) / (a^2) + y^2 / (b^2) = 1

theorem midpoint_coords :
  ∃ (A B : ℝ × ℝ), ellipse_eq A.1 A.2 ∧ ellipse_eq B.1 B.2 ∧ line_eq A.1 A.2 ∧ line_eq B.1 B.2 →
  (A.1 + B.1) / 2 = -9 / 5 ∧ (A.2 + B.2) / 2 = 1 / 5 :=
by
  sorry

end NUMINAMATH_GPT_midpoint_coords_l1640_164046


namespace NUMINAMATH_GPT_tan_triple_angle_formula_l1640_164094

variable (θ : ℝ)
variable (h : Real.tan θ = 4)

theorem tan_triple_angle_formula : Real.tan (3 * θ) = 52 / 47 :=
by
  sorry  -- Proof is omitted

end NUMINAMATH_GPT_tan_triple_angle_formula_l1640_164094


namespace NUMINAMATH_GPT_parker_added_dumbbells_l1640_164037

def initial_dumbbells : Nat := 4
def weight_per_dumbbell : Nat := 20
def total_weight_used : Nat := 120

theorem parker_added_dumbbells :
  (total_weight_used - (initial_dumbbells * weight_per_dumbbell)) / weight_per_dumbbell = 2 := by
  sorry

end NUMINAMATH_GPT_parker_added_dumbbells_l1640_164037


namespace NUMINAMATH_GPT_cos_B_find_b_l1640_164081

theorem cos_B (a b c : ℝ) (h1 : 2 * b = a + c) (h2 : 7 * a = 3 * c) :
  Real.cos (Real.arccos ((a^2 + c^2 - b^2) / (2 * a * c))) = 11 / 14 := by
  sorry

theorem find_b (a b c : ℝ) (h1 : 2 * b = a + c) (h2 : 7 * a = 3 * c)
  (area : ℝ := 15 * Real.sqrt 3 / 4)
  (h3 : (1/2) * a * c * Real.sin (Real.arccos ((a^2 + c^2 - b^2) / (2 * a * c))) = area) :
  b = 5 := by
  sorry

end NUMINAMATH_GPT_cos_B_find_b_l1640_164081


namespace NUMINAMATH_GPT_largest_n_l1640_164026

theorem largest_n (n x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  n^2 = x^2 + y^2 + z^2 + 2 * x * y + 2 * y * z + 2 * z * x + 6 * x + 6 * y + 6 * z - 18 →
  n ≤ 3 := 
by 
  sorry

end NUMINAMATH_GPT_largest_n_l1640_164026


namespace NUMINAMATH_GPT_equal_powers_equal_elements_l1640_164065

theorem equal_powers_equal_elements
  (a : Fin 17 → ℕ)
  (h : ∀ i : Fin 17, a i ^ a (i + 1) % 17 = a ((i + 1) % 17) ^ a ((i + 2) % 17) % 17)
  : ∀ i j : Fin 17, a i = a j :=
by
  sorry

end NUMINAMATH_GPT_equal_powers_equal_elements_l1640_164065


namespace NUMINAMATH_GPT_jacobs_hourly_wage_l1640_164056

theorem jacobs_hourly_wage (jake_total_earnings : ℕ) (jake_days : ℕ) (hours_per_day : ℕ) (jake_thrice_jacob : ℕ) 
    (h_total_jake : jake_total_earnings = 720) 
    (h_jake_days : jake_days = 5) 
    (h_hours_per_day : hours_per_day = 8)
    (h_jake_thrice_jacob : jake_thrice_jacob = 3) 
    (jacob_hourly_wage : ℕ) :
  jacob_hourly_wage = 6 := 
by
  sorry

end NUMINAMATH_GPT_jacobs_hourly_wage_l1640_164056


namespace NUMINAMATH_GPT_original_price_of_shirt_l1640_164000

theorem original_price_of_shirt (discounted_price : ℝ) (discount_percentage : ℝ) 
  (h_discounted_price : discounted_price = 780) (h_discount_percentage : discount_percentage = 0.20) 
  : (discounted_price / (1 - discount_percentage) = 975) := by
  sorry

end NUMINAMATH_GPT_original_price_of_shirt_l1640_164000


namespace NUMINAMATH_GPT_amount_paid_correct_l1640_164021

-- Defining the conditions and constants
def hourly_rate : ℕ := 60
def hours_per_day : ℕ := 3
def total_days : ℕ := 14

-- The proof statement
theorem amount_paid_correct : hourly_rate * hours_per_day * total_days = 2520 := by
  sorry

end NUMINAMATH_GPT_amount_paid_correct_l1640_164021


namespace NUMINAMATH_GPT_tim_buys_loaves_l1640_164048

theorem tim_buys_loaves (slices_per_loaf : ℕ) (paid : ℕ) (change : ℕ) (price_per_slice_cents : ℕ) 
    (h1 : slices_per_loaf = 20) 
    (h2 : paid = 2 * 20) 
    (h3 : change = 16) 
    (h4 : price_per_slice_cents = 40) : 
    (paid - change) / (slices_per_loaf * price_per_slice_cents / 100) = 3 := 
by 
  -- proof omitted 
  sorry

end NUMINAMATH_GPT_tim_buys_loaves_l1640_164048


namespace NUMINAMATH_GPT_initial_jellybeans_l1640_164058

theorem initial_jellybeans (J : ℕ) :
    (∀ x y : ℕ, x = 24 → y = 12 →
    (J - x - y + ((x + y) / 2) = 72) → J = 90) :=
by
  intros x y hx hy h
  rw [hx, hy] at h
  sorry

end NUMINAMATH_GPT_initial_jellybeans_l1640_164058


namespace NUMINAMATH_GPT_find_x_l1640_164040

def vector := (ℝ × ℝ)

def a (x : ℝ) : vector := (x, 2)
def b : vector := (1, -1)

-- Dot product of two vectors
def dot_product (v1 v2 : vector) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Orthogonality condition rewritten in terms of dot product
def orthogonal (v1 v2 : vector) : Prop := dot_product v1 v2 = 0

-- Main theorem to prove
theorem find_x (x : ℝ) (h : orthogonal ((a x).1 - b.1, (a x).2 - b.2) b) : x = 4 :=
by sorry

end NUMINAMATH_GPT_find_x_l1640_164040


namespace NUMINAMATH_GPT_determine_m_l1640_164055

theorem determine_m (x y m : ℝ) 
  (h1 : 3 * x + 2 * y = 4 * m - 5) 
  (h2 : 2 * x + 3 * y = m) 
  (h3 : x + y = 2) : 
  m = 3 :=
sorry

end NUMINAMATH_GPT_determine_m_l1640_164055


namespace NUMINAMATH_GPT_probability_of_D_l1640_164068

theorem probability_of_D (pA pB pC pD : ℚ)
  (hA : pA = 1/4)
  (hB : pB = 1/3)
  (hC : pC = 1/6)
  (hTotal : pA + pB + pC + pD = 1) : pD = 1/4 :=
by
  have hTotal_before_D : pD = 1 - (pA + pB + pC) := by sorry
  sorry

end NUMINAMATH_GPT_probability_of_D_l1640_164068


namespace NUMINAMATH_GPT_factorial_expression_l1640_164018

open Nat

theorem factorial_expression :
  7 * (6!) + 6 * (5!) + 2 * (5!) = 6000 :=
by
  sorry

end NUMINAMATH_GPT_factorial_expression_l1640_164018


namespace NUMINAMATH_GPT_apples_more_than_oranges_l1640_164062

-- Definitions based on conditions
def total_fruits : ℕ := 301
def apples : ℕ := 164

-- Statement to prove
theorem apples_more_than_oranges : (apples - (total_fruits - apples)) = 27 :=
by
  sorry

end NUMINAMATH_GPT_apples_more_than_oranges_l1640_164062


namespace NUMINAMATH_GPT_unique_solution_triple_l1640_164079

theorem unique_solution_triple (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (xy / z : ℚ) + (yz / x) + (zx / y) = 3 → (x = 1 ∧ y = 1 ∧ z = 1) := 
by 
  sorry

end NUMINAMATH_GPT_unique_solution_triple_l1640_164079


namespace NUMINAMATH_GPT_value_of_expression_l1640_164016

theorem value_of_expression (a b c : ℝ) (h : a * (-2)^5 + b * (-2)^3 + c * (-2) - 5 = 7) :
  a * 2^5 + b * 2^3 + c * 2 - 5 = -17 :=
by sorry

end NUMINAMATH_GPT_value_of_expression_l1640_164016


namespace NUMINAMATH_GPT_train_cross_time_in_seconds_l1640_164023

-- Definitions based on conditions
def train_speed_kph : ℚ := 60
def train_length_m : ℚ := 450

-- Statement: prove that the time to cross the pole is 27 seconds
theorem train_cross_time_in_seconds (train_speed_kph train_length_m : ℚ) :
  train_speed_kph = 60 →
  train_length_m = 450 →
  (train_length_m / (train_speed_kph * 1000 / 3600)) = 27 :=
by
  intros h_speed h_length
  rw [h_speed, h_length]
  sorry

end NUMINAMATH_GPT_train_cross_time_in_seconds_l1640_164023


namespace NUMINAMATH_GPT_good_tipper_bill_amount_l1640_164009

theorem good_tipper_bill_amount {B : ℝ} 
    (h₁ : 0.05 * B + 1/20 ≥ 0.20 * B) 
    (h₂ : 0.15 * B = 3.90) : 
    B = 26.00 := 
by 
  sorry

end NUMINAMATH_GPT_good_tipper_bill_amount_l1640_164009


namespace NUMINAMATH_GPT_circle_passing_origin_l1640_164060

theorem circle_passing_origin (a b r : ℝ) :
  ((a^2 + b^2 = r^2) ↔ (∃ (x y : ℝ), (x-a)^2 + (y-b)^2 = r^2 ∧ x = 0 ∧ y = 0)) :=
by
  sorry

end NUMINAMATH_GPT_circle_passing_origin_l1640_164060

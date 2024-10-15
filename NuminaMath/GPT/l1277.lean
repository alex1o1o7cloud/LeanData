import Mathlib

namespace NUMINAMATH_GPT_f_difference_l1277_127764

noncomputable def f (n : ℕ) : ℝ :=
  (6 + 4 * Real.sqrt 3) / 12 * ((1 + Real.sqrt 3) / 2)^n + 
  (6 - 4 * Real.sqrt 3) / 12 * ((1 - Real.sqrt 3) / 2)^n

theorem f_difference (n : ℕ) : f (n + 1) - f n = (Real.sqrt 3 - 3) / 4 * f n :=
  sorry

end NUMINAMATH_GPT_f_difference_l1277_127764


namespace NUMINAMATH_GPT_radii_of_circles_l1277_127794

theorem radii_of_circles
  (r s : ℝ)
  (h_ratio : r / s = 9 / 4)
  (h_right_triangle : ∃ (a b c : ℝ), a = 3 ∧ b = 4 ∧ c = 5 ∧ a^2 + b^2 = c^2)
  (h_tangent : (r + s)^2 = (r - s)^2 + 12^2) :
   r = 20 / 47 ∧ s = 45 / 47 :=
by
  sorry

end NUMINAMATH_GPT_radii_of_circles_l1277_127794


namespace NUMINAMATH_GPT_model_tower_height_l1277_127704

theorem model_tower_height (real_height : ℝ) (real_volume : ℝ) (model_volume : ℝ) (h_real : real_height = 60) (v_real : real_volume = 200000) (v_model : model_volume = 0.2) :
  real_height / (real_volume / model_volume)^(1/3) = 0.6 :=
by
  rw [h_real, v_real, v_model]
  norm_num
  sorry

end NUMINAMATH_GPT_model_tower_height_l1277_127704


namespace NUMINAMATH_GPT_second_reduction_is_18_point_1_percent_l1277_127783

noncomputable def second_reduction_percentage (P : ℝ) : ℝ :=
  let first_price := 0.91 * P
  let second_price := 0.819 * P
  let R := (first_price - second_price) / first_price
  R * 100

theorem second_reduction_is_18_point_1_percent (P : ℝ) : second_reduction_percentage P = 18.1 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_second_reduction_is_18_point_1_percent_l1277_127783


namespace NUMINAMATH_GPT_largest_value_of_number_l1277_127788

theorem largest_value_of_number 
  (v w x y z : ℝ)
  (h1 : v + w + x + y + z = 8)
  (h2 : v^2 + w^2 + x^2 + y^2 + z^2 = 16) :
  ∃ (m : ℝ), m = 2.4 ∧ (m = v ∨ m = w ∨ m = x ∨ m = y ∨ m = z) :=
sorry

end NUMINAMATH_GPT_largest_value_of_number_l1277_127788


namespace NUMINAMATH_GPT_largest_possible_perimeter_l1277_127752

noncomputable def max_perimeter_triangle : ℤ :=
  let a : ℤ := 7
  let b : ℤ := 9
  let x : ℤ := 15
  a + b + x

theorem largest_possible_perimeter (x : ℤ) (h1 : 7 + 9 > x) (h2 : 7 + x > 9) (h3 : 9 + x > 7) : max_perimeter_triangle = 31 := by
  sorry

end NUMINAMATH_GPT_largest_possible_perimeter_l1277_127752


namespace NUMINAMATH_GPT_symmetric_point_is_correct_l1277_127721

/-- A point in 2D Cartesian coordinates -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Defining the point P with given coordinates -/
def P : Point := {x := 2, y := 3}

/-- Defining the symmetry of a point with respect to the origin -/
def symmetric_origin (p : Point) : Point :=
  { x := -p.x, y := -p.y }

/-- States that the symmetric point of P (2, 3) with respect to the origin is (-2, -3) -/
theorem symmetric_point_is_correct :
  symmetric_origin P = {x := -2, y := -3} :=
by
  sorry

end NUMINAMATH_GPT_symmetric_point_is_correct_l1277_127721


namespace NUMINAMATH_GPT_sum_of_number_and_reverse_l1277_127700

theorem sum_of_number_and_reverse (a b : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ 9) (h3 : 0 ≤ b) (h4 : b ≤ 9) 
  (h5 : (10 * a + b) - (10 * b + a) = 7 * (a + b)) : 
  (10 * a + b) + (10 * b + a) = 99 := 
sorry

end NUMINAMATH_GPT_sum_of_number_and_reverse_l1277_127700


namespace NUMINAMATH_GPT_problem_1_problem_2_l1277_127715

open Set -- to work with sets conveniently

noncomputable section -- to allow the use of real numbers and other non-constructive elements

-- Define U as the set of all real numbers
def U : Set ℝ := univ

-- Define M as the set of all x such that y = sqrt(x - 2)
def M : Set ℝ := {x : ℝ | ∃ y : ℝ, y = Real.sqrt (x - 2) }

-- Define N as the set of all x such that x < 1 or x > 3
def N : Set ℝ := {x : ℝ | x < 1 ∨ x > 3}

-- Statement to prove (1)
theorem problem_1 : M ∪ N = {x : ℝ | x < 1 ∨ x ≥ 2} := sorry

-- Statement to prove (2)
theorem problem_2 : M ∩ (compl N) = {x : ℝ | 2 ≤ x ∧ x ≤ 3} := sorry

end NUMINAMATH_GPT_problem_1_problem_2_l1277_127715


namespace NUMINAMATH_GPT_arithmetic_sequence_count_l1277_127730

theorem arithmetic_sequence_count :
  ∃ n : ℕ, 2 + (n-1) * 5 = 2507 ∧ n = 502 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_count_l1277_127730


namespace NUMINAMATH_GPT_smallest_number_greater_than_300_divided_by_25_has_remainder_24_l1277_127718

theorem smallest_number_greater_than_300_divided_by_25_has_remainder_24 :
  ∃ x : ℕ, (x > 300) ∧ (x % 25 = 24) ∧ (x = 324) := by
  sorry

end NUMINAMATH_GPT_smallest_number_greater_than_300_divided_by_25_has_remainder_24_l1277_127718


namespace NUMINAMATH_GPT_units_digit_of_result_is_eight_l1277_127778

def three_digit_number_reverse_subtract (a b c : ℕ) : ℕ :=
  let original := 100 * a + 10 * b + c
  let reversed := 100 * c + 10 * b + a
  original - reversed

theorem units_digit_of_result_is_eight (a b c : ℕ) (h : a = c + 2) :
  (three_digit_number_reverse_subtract a b c) % 10 = 8 :=
by
  sorry

end NUMINAMATH_GPT_units_digit_of_result_is_eight_l1277_127778


namespace NUMINAMATH_GPT_solution_set_l1277_127728

noncomputable def f : ℝ → ℝ
| x => if x < 2 then 2 * Real.exp (x - 1) else Real.log (x^2 - 1) / Real.log 3

theorem solution_set (x : ℝ) : 
  ((x > 1 ∧ x < 2 ∨ x > Real.sqrt 10)) ↔ f x > 2 :=
sorry

end NUMINAMATH_GPT_solution_set_l1277_127728


namespace NUMINAMATH_GPT_total_money_tshirts_l1277_127795

-- Conditions
def price_per_tshirt : ℕ := 62
def num_tshirts_sold : ℕ := 183

-- Question: prove the total money made from selling the t-shirts
theorem total_money_tshirts :
  num_tshirts_sold * price_per_tshirt = 11346 := 
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_total_money_tshirts_l1277_127795


namespace NUMINAMATH_GPT_remainder_of_sum_is_five_l1277_127729

theorem remainder_of_sum_is_five (a b c d : ℕ) (ha : a % 15 = 11) (hb : b % 15 = 12) (hc : c % 15 = 13) (hd : d % 15 = 14) :
  (a + b + c + d) % 15 = 5 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_sum_is_five_l1277_127729


namespace NUMINAMATH_GPT_negation_example_l1277_127799

variable (x : ℤ)

theorem negation_example : (¬ ∀ x : ℤ, |x| ≠ 3) ↔ (∃ x : ℤ, |x| = 3) :=
by
  sorry

end NUMINAMATH_GPT_negation_example_l1277_127799


namespace NUMINAMATH_GPT_sphere_surface_area_ratio_l1277_127791

theorem sphere_surface_area_ratio (V1 V2 : ℝ) (h1 : V1 = (4 / 3) * π * (r1^3))
  (h2 : V2 = (4 / 3) * π * (r2^3)) (h3 : V1 / V2 = 1 / 27) :
  (4 * π * r1^2) / (4 * π * r2^2) = 1 / 9 := 
sorry

end NUMINAMATH_GPT_sphere_surface_area_ratio_l1277_127791


namespace NUMINAMATH_GPT_circle_symmetric_eq_l1277_127717

theorem circle_symmetric_eq :
  ∀ (x y : ℝ), (x^2 + y^2 + 2 * x - 2 * y + 1 = 0) → (x - y + 3 = 0) → 
  (∃ (a b : ℝ), (a + 2)^2 + (b - 2)^2 = 1) :=
by
  intros x y hc hl
  sorry

end NUMINAMATH_GPT_circle_symmetric_eq_l1277_127717


namespace NUMINAMATH_GPT_add_and_round_58_29_l1277_127734

def add_and_round_to_nearest_ten (a b : ℕ) : ℕ :=
  let sum := a + b
  let rounded_sum := if sum % 10 < 5 then sum - (sum % 10) else sum + (10 - sum % 10)
  rounded_sum

theorem add_and_round_58_29 : add_and_round_to_nearest_ten 58 29 = 90 := by
  sorry

end NUMINAMATH_GPT_add_and_round_58_29_l1277_127734


namespace NUMINAMATH_GPT_find_projection_l1277_127732

noncomputable def a : ℝ × ℝ := (-3, 2)
noncomputable def b : ℝ × ℝ := (5, -1)
noncomputable def p : ℝ × ℝ := (21/73, 56/73)
noncomputable def d : ℝ × ℝ := (8, -3)

theorem find_projection :
  ∃ t : ℝ, (t * d.1 - a.1, t * d.2 + a.2) = p ∧
          (p.1 - a.1) * d.1 + (p.2 - a.2) * d.2 = 0 :=
by
  sorry

end NUMINAMATH_GPT_find_projection_l1277_127732


namespace NUMINAMATH_GPT_min_value_of_a_plus_b_l1277_127731

theorem min_value_of_a_plus_b (a b c : ℝ) (C : ℝ) 
  (hC : C = 60) 
  (h : (a + b)^2 - c^2 = 4) : 
  a + b ≥ (4 * Real.sqrt 3) / 3 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_a_plus_b_l1277_127731


namespace NUMINAMATH_GPT_increase_average_by_3_l1277_127714

theorem increase_average_by_3 (x : ℕ) (average_initial : ℕ := 32) (matches_initial : ℕ := 10) (score_11th_match : ℕ := 65) :
  (matches_initial * average_initial + score_11th_match = 11 * (average_initial + x)) → x = 3 := 
sorry

end NUMINAMATH_GPT_increase_average_by_3_l1277_127714


namespace NUMINAMATH_GPT_largest_n_unique_k_l1277_127741

theorem largest_n_unique_k (n : ℕ) (h : ∃ k : ℕ, (9 / 17 : ℚ) < n / (n + k) ∧ n / (n + k) < (8 / 15 : ℚ) ∧ ∀ k' : ℕ, ((9 / 17 : ℚ) < n / (n + k') ∧ n / (n + k') < (8 / 15 : ℚ)) → k' = k) : n = 72 :=
sorry

end NUMINAMATH_GPT_largest_n_unique_k_l1277_127741


namespace NUMINAMATH_GPT_necessary_and_sufficient_condition_l1277_127790

theorem necessary_and_sufficient_condition (p q : Prop) 
  (hpq : p → q) (hqp : q → p) : 
  (p ↔ q) :=
by 
  sorry

end NUMINAMATH_GPT_necessary_and_sufficient_condition_l1277_127790


namespace NUMINAMATH_GPT_blue_marbles_count_l1277_127768

theorem blue_marbles_count
  (total_marbles : ℕ)
  (yellow_marbles : ℕ)
  (red_marbles : ℕ)
  (blue_marbles : ℕ)
  (yellow_probability : ℚ)
  (total_marbles_eq : yellow_marbles = 6)
  (yellow_probability_eq : yellow_probability = 1 / 4)
  (red_marbles_eq : red_marbles = 11)
  (total_marbles_def : total_marbles = yellow_marbles * 4)
  (blue_marbles_def : blue_marbles = total_marbles - red_marbles - yellow_marbles) :
  blue_marbles = 7 :=
sorry

end NUMINAMATH_GPT_blue_marbles_count_l1277_127768


namespace NUMINAMATH_GPT_cash_still_missing_l1277_127725

theorem cash_still_missing (c : ℝ) (h : c > 0) :
  (1 : ℝ) - (8 / 9) = (1 / 9 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_cash_still_missing_l1277_127725


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1277_127787

noncomputable def f (x : ℝ) : ℝ :=
if x < 0 then x^2 - x else Real.log (x + 1) / Real.log 2

theorem solution_set_of_inequality :
  { x : ℝ | f x ≥ 2 } = { x : ℝ | x ∈ Set.Iic (-1) } ∪ { x : ℝ | x ∈ Set.Ici 3 } :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1277_127787


namespace NUMINAMATH_GPT_parabola_vertex_origin_through_point_l1277_127742

theorem parabola_vertex_origin_through_point :
  (∃ p, p > 0 ∧ x^2 = 2 * p * y ∧ (x, y) = (-4, 4) → x^2 = 4 * y) ∨
  (∃ p, p > 0 ∧ y^2 = -2 * p * x ∧ (x, y) = (-4, 4) → y^2 = -4 * x) :=
sorry

end NUMINAMATH_GPT_parabola_vertex_origin_through_point_l1277_127742


namespace NUMINAMATH_GPT_committee_selection_count_l1277_127745

-- Definition of the problem condition: Club of 12 people, one specific person must always be on the committee.
def club_size : ℕ := 12
def committee_size : ℕ := 4
def specific_person_included : ℕ := 1

-- Number of ways to choose 3 members from the other 11 people
def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem committee_selection_count : choose 11 3 = 165 := 
  sorry

end NUMINAMATH_GPT_committee_selection_count_l1277_127745


namespace NUMINAMATH_GPT_fewest_keystrokes_One_to_410_l1277_127709

noncomputable def fewest_keystrokes (start : ℕ) (target : ℕ) : ℕ :=
if target = 410 then 10 else sorry

theorem fewest_keystrokes_One_to_410 : fewest_keystrokes 1 410 = 10 :=
by
  sorry

end NUMINAMATH_GPT_fewest_keystrokes_One_to_410_l1277_127709


namespace NUMINAMATH_GPT_interval_length_l1277_127722

theorem interval_length (x : ℝ) :
  (1/x > 1/2) ∧ (Real.sin x > 1/2) → (2 - Real.pi / 6 = 1.48) :=
by
  sorry

end NUMINAMATH_GPT_interval_length_l1277_127722


namespace NUMINAMATH_GPT_gcd_positive_ints_l1277_127784

theorem gcd_positive_ints (a b c d : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) 
  (hdiv : (a^2 + b^2) ∣ (a * c + b * d)) : 
  Nat.gcd (c^2 + d^2) (a^2 + b^2) > 1 := 
sorry

end NUMINAMATH_GPT_gcd_positive_ints_l1277_127784


namespace NUMINAMATH_GPT_intersection_of_complement_l1277_127759

open Set

variable (U : Set ℤ) (A B : Set ℤ)

def complement (U A : Set ℤ) : Set ℤ := U \ A

theorem intersection_of_complement (hU : U = {-1, 0, 1, 2, 3, 4})
  (hA : A = {1, 2, 3, 4}) (hB : B = {0, 2}) :
  (complement U A) ∩ B = {0} :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_complement_l1277_127759


namespace NUMINAMATH_GPT_vertical_distance_rotated_square_l1277_127701

-- Lean 4 statement for the mathematically equivalent proof problem
theorem vertical_distance_rotated_square
  (side_length : ℝ)
  (n : ℕ)
  (rot_angle : ℝ)
  (orig_line_height before_rotation : ℝ)
  (diagonal_length : ℝ)
  (lowered_distance : ℝ)
  (highest_point_drop : ℝ)
  : side_length = 2 →
    n = 4 →
    rot_angle = 45 →
    orig_line_height = 1 →
    diagonal_length = side_length * (2:ℝ)^(1/2) →
    lowered_distance = (diagonal_length / 2) - orig_line_height →
    highest_point_drop = lowered_distance →
    2 = 2 :=
    sorry

end NUMINAMATH_GPT_vertical_distance_rotated_square_l1277_127701


namespace NUMINAMATH_GPT_find_N_l1277_127771

theorem find_N (N x : ℝ) (h1 : N / (1 + 4 / x) = 1) (h2 : x = 0.5) : N = 9 := 
by 
  sorry

end NUMINAMATH_GPT_find_N_l1277_127771


namespace NUMINAMATH_GPT_calculate_expression_l1277_127773

theorem calculate_expression : (3.65 - 1.25) * 2 = 4.80 := 
by 
  sorry

end NUMINAMATH_GPT_calculate_expression_l1277_127773


namespace NUMINAMATH_GPT_total_amount_l1277_127733

noncomputable def mark_amount : ℝ := 5 / 8

noncomputable def carolyn_amount : ℝ := 7 / 20

theorem total_amount : mark_amount + carolyn_amount = 0.975 := by
  sorry

end NUMINAMATH_GPT_total_amount_l1277_127733


namespace NUMINAMATH_GPT_average_of_solutions_l1277_127716

-- Define the quadratic equation condition
def quadratic_eq : Prop := ∃ x : ℂ, 3*x^2 - 4*x + 1 = 0

-- State the theorem
theorem average_of_solutions : quadratic_eq → (∃ avg : ℂ, avg = 2 / 3) :=
by
  sorry

end NUMINAMATH_GPT_average_of_solutions_l1277_127716


namespace NUMINAMATH_GPT_cat_can_pass_through_gap_l1277_127785

theorem cat_can_pass_through_gap (R : ℝ) (h : ℝ) (π : ℝ) (hπ : π = Real.pi)
  (L₀ : ℝ) (L₁ : ℝ)
  (hL₀ : L₀ = 2 * π * R)
  (hL₁ : L₁ = L₀ + 1)
  (hL₁' : L₁ = 2 * π * (R + h)) :
  h = 1 / (2 * π) :=
by
  sorry

end NUMINAMATH_GPT_cat_can_pass_through_gap_l1277_127785


namespace NUMINAMATH_GPT_cube_face_sum_l1277_127707

theorem cube_face_sum (a b c d e f : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) (h5 : e > 0) (h6 : f > 0) :
  (a * b * c + a * e * c + a * b * f + a * e * f + d * b * c + d * e * c + d * b * f + d * e * f = 1287) →
  (a + d + b + e + c + f = 33) :=
by
  sorry

end NUMINAMATH_GPT_cube_face_sum_l1277_127707


namespace NUMINAMATH_GPT_emails_in_morning_and_afternoon_l1277_127724

-- Conditions
def morning_emails : Nat := 5
def afternoon_emails : Nat := 8

-- Theorem statement
theorem emails_in_morning_and_afternoon : morning_emails + afternoon_emails = 13 := by
  -- Proof goes here, but adding sorry for now
  sorry

end NUMINAMATH_GPT_emails_in_morning_and_afternoon_l1277_127724


namespace NUMINAMATH_GPT_linear_dependence_condition_l1277_127705

theorem linear_dependence_condition (k : ℝ) :
  (∃ (a b : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧ (a * 1 + b * 4 = 0) ∧ (a * 2 + b * k = 0) ∧ (a * 1 + b * 2 = 0)) ↔ k = 8 := 
by sorry

end NUMINAMATH_GPT_linear_dependence_condition_l1277_127705


namespace NUMINAMATH_GPT_robert_total_interest_l1277_127774

theorem robert_total_interest
  (inheritance : ℕ)
  (part1 part2 : ℕ)
  (rate1 rate2 : ℝ)
  (time : ℝ) :
  inheritance = 4000 →
  part2 = 1800 →
  part1 = inheritance - part2 →
  rate1 = 0.05 →
  rate2 = 0.065 →
  time = 1 →
  (part1 * rate1 * time + part2 * rate2 * time) = 227 :=
by
  intros
  sorry

end NUMINAMATH_GPT_robert_total_interest_l1277_127774


namespace NUMINAMATH_GPT_sqrt_prod_simplified_l1277_127777

open Real

variable (x : ℝ)

theorem sqrt_prod_simplified (hx : 0 ≤ x) : sqrt (50 * x) * sqrt (18 * x) * sqrt (8 * x) = 30 * x * sqrt (2 * x) :=
by
  sorry

end NUMINAMATH_GPT_sqrt_prod_simplified_l1277_127777


namespace NUMINAMATH_GPT_pete_ten_dollar_bills_l1277_127739

theorem pete_ten_dollar_bills (owes dollars bills: ℕ) (bill_value_per_bottle : ℕ) (num_bottles : ℕ) (ten_dollar_bills : ℕ):
  owes = 90 →
  dollars = 40 →
  bill_value_per_bottle = 5 →
  num_bottles = 20 →
  dollars + (num_bottles * bill_value_per_bottle) + (ten_dollar_bills * 10) = owes →
  ten_dollar_bills = 4 :=
by
  sorry

end NUMINAMATH_GPT_pete_ten_dollar_bills_l1277_127739


namespace NUMINAMATH_GPT_polynomial_evaluation_l1277_127727

theorem polynomial_evaluation (P : ℕ → ℝ) (n : ℕ) 
  (h_degree : ∀ k : ℕ, k ≤ n → P k = k / (k + 1)) 
  (h_poly : ∀ k : ℕ, ∃ a : ℝ, P k = a * k ^ n) : 
  P (n + 1) = (n + 1 + (-1) ^ (n + 1)) / (n + 2) :=
by 
  sorry

end NUMINAMATH_GPT_polynomial_evaluation_l1277_127727


namespace NUMINAMATH_GPT_percent_nonunion_part_time_women_l1277_127789

noncomputable def percent (part: ℚ) (whole: ℚ) : ℚ := part / whole * 100

def employees : ℚ := 100
def men_ratio : ℚ := 54 / 100
def women_ratio : ℚ := 46 / 100
def full_time_men_ratio : ℚ := 70 / 100
def part_time_men_ratio : ℚ := 30 / 100
def full_time_women_ratio : ℚ := 60 / 100
def part_time_women_ratio : ℚ := 40 / 100
def union_full_time_ratio : ℚ := 60 / 100
def union_part_time_ratio : ℚ := 50 / 100

def men := employees * men_ratio
def women := employees * women_ratio
def full_time_men := men * full_time_men_ratio
def part_time_men := men * part_time_men_ratio
def full_time_women := women * full_time_women_ratio
def part_time_women := women * part_time_women_ratio
def total_full_time := full_time_men + full_time_women
def total_part_time := part_time_men + part_time_women

def union_full_time := total_full_time * union_full_time_ratio
def union_part_time := total_part_time * union_part_time_ratio
def nonunion_full_time := total_full_time - union_full_time
def nonunion_part_time := total_part_time - union_part_time

def nonunion_part_time_women_ratio : ℚ := 50 / 100
def nonunion_part_time_women := part_time_women * nonunion_part_time_women_ratio

theorem percent_nonunion_part_time_women : 
  percent nonunion_part_time_women nonunion_part_time = 52.94 :=
by
  sorry

end NUMINAMATH_GPT_percent_nonunion_part_time_women_l1277_127789


namespace NUMINAMATH_GPT_remainder_of_b97_is_52_l1277_127746

def b (n : ℕ) : ℕ := 7^n + 9^n

theorem remainder_of_b97_is_52 : (b 97) % 81 = 52 := 
sorry

end NUMINAMATH_GPT_remainder_of_b97_is_52_l1277_127746


namespace NUMINAMATH_GPT_ttakjis_count_l1277_127736

theorem ttakjis_count (n : ℕ) (initial_residual new_residual total_ttakjis : ℕ) :
  initial_residual = 36 → 
  new_residual = 3 → 
  total_ttakjis = n^2 + initial_residual → 
  total_ttakjis = (n + 1)^2 + new_residual → 
  total_ttakjis = 292 :=
by
  sorry

end NUMINAMATH_GPT_ttakjis_count_l1277_127736


namespace NUMINAMATH_GPT_raking_yard_time_l1277_127740

theorem raking_yard_time (your_rate : ℚ) (brother_rate : ℚ) (combined_rate : ℚ) (combined_time : ℚ) :
  your_rate = 1 / 30 ∧ 
  brother_rate = 1 / 45 ∧ 
  combined_rate = your_rate + brother_rate ∧ 
  combined_time = 1 / combined_rate → 
  combined_time = 18 := 
by 
  sorry

end NUMINAMATH_GPT_raking_yard_time_l1277_127740


namespace NUMINAMATH_GPT_nonnegative_integer_with_divisors_is_multiple_of_6_l1277_127770

-- Definitions as per conditions in (a)
def has_two_distinct_divisors_with_distance (n : ℕ) : Prop := ∃ d1 d2 : ℕ,
  d1 ≠ d2 ∧ d1 ∣ n ∧ d2 ∣ n ∧
  (d1:ℚ) - n / 3 = n / 3 - (d2:ℚ)

-- Main statement to prove as derived in (c)
theorem nonnegative_integer_with_divisors_is_multiple_of_6 (n : ℕ) :
  n > 0 ∧ has_two_distinct_divisors_with_distance n → ∃ k : ℕ, n = 6 * k :=
by
  sorry

end NUMINAMATH_GPT_nonnegative_integer_with_divisors_is_multiple_of_6_l1277_127770


namespace NUMINAMATH_GPT_train_speed_l1277_127797

theorem train_speed (distance_meters : ℝ) (time_seconds : ℝ) :
  distance_meters = 180 →
  time_seconds = 17.998560115190784 →
  ((distance_meters / 1000) / (time_seconds / 3600)) = 36.00360072014403 :=
by 
  intros h1 h2
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_train_speed_l1277_127797


namespace NUMINAMATH_GPT_min_time_proof_l1277_127757

/-
  Problem: 
  Given 5 colored lights that each can shine in one of the colors {red, orange, yellow, green, blue},
  and the colors are all different, and the interval between two consecutive flashes is 5 seconds.
  Define the ordered shining of these 5 lights once as a "flash", where each flash lasts 5 seconds.
  We need to show that the minimum time required to achieve all different flashes (120 flashes) is equal to 1195 seconds.
-/

def min_time_required : Nat :=
  let num_flashes := 5 * 4 * 3 * 2 * 1
  let flash_time := 5 * num_flashes
  let interval_time := 5 * (num_flashes - 1)
  flash_time + interval_time

theorem min_time_proof : min_time_required = 1195 := by
  sorry

end NUMINAMATH_GPT_min_time_proof_l1277_127757


namespace NUMINAMATH_GPT_find_n_l1277_127756

noncomputable def positive_geometric_sequence := ℕ → ℝ

def is_geometric_sequence (a : positive_geometric_sequence) :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

def conditions (a : positive_geometric_sequence) :=
  is_geometric_sequence a ∧
  a 0 * a 1 * a 2 = 4 ∧
  a 3 * a 4 * a 5 = 12 ∧
  ∃ n : ℕ, a (n - 1) * a n * a (n + 1) = 324

theorem find_n (a : positive_geometric_sequence) (h : conditions a) : ∃ n : ℕ, n = 14 :=
by
  sorry

end NUMINAMATH_GPT_find_n_l1277_127756


namespace NUMINAMATH_GPT_gcd_231_154_l1277_127726

def find_gcd (a b : ℕ) : ℕ := Nat.gcd a b

theorem gcd_231_154 : find_gcd 231 154 = 77 := by
  sorry

end NUMINAMATH_GPT_gcd_231_154_l1277_127726


namespace NUMINAMATH_GPT_grandfather_grandson_ages_l1277_127780

theorem grandfather_grandson_ages :
  ∃ (x y a b : ℕ), 
    70 < x ∧ 
    x < 80 ∧ 
    x - a = 10 * (y - a) ∧ 
    x + b = 8 * (y + b) ∧ 
    x = 71 ∧ 
    y = 8 :=
by
  sorry

end NUMINAMATH_GPT_grandfather_grandson_ages_l1277_127780


namespace NUMINAMATH_GPT_audit_options_correct_l1277_127735

-- Define the initial number of ORs and GTUs
def initial_ORs : ℕ := 13
def initial_GTUs : ℕ := 15

-- Define the number of ORs and GTUs visited in the first week
def visited_ORs : ℕ := 2
def visited_GTUs : ℕ := 3

-- Calculate the remaining ORs and GTUs
def remaining_ORs : ℕ := initial_ORs - visited_ORs
def remaining_GTUs : ℕ := initial_GTUs - visited_GTUs

-- Calculate the number of ways to choose 2 ORs from remaining ORs
def choose_ORs : ℕ := Nat.choose remaining_ORs 2

-- Calculate the number of ways to choose 3 GTUs from remaining GTUs
def choose_GTUs : ℕ := Nat.choose remaining_GTUs 3

-- The final function to calculate the number of options
def number_of_options : ℕ := choose_ORs * choose_GTUs

-- The proof statement asserting the number of options is 12100
theorem audit_options_correct : number_of_options = 12100 := by
    sorry -- Proof will be filled in here

end NUMINAMATH_GPT_audit_options_correct_l1277_127735


namespace NUMINAMATH_GPT_largest_divisor_prime_cube_diff_l1277_127703

theorem largest_divisor_prime_cube_diff (p : ℕ) (hp_prime : Nat.Prime p) (hp_ge5 : p ≥ 5) : 
  ∃ k, k = 12 ∧ ∀ n, n ∣ (p^3 - p) ↔ n ∣ 12 :=
by
  sorry

end NUMINAMATH_GPT_largest_divisor_prime_cube_diff_l1277_127703


namespace NUMINAMATH_GPT_trig_proof_l1277_127762

noncomputable def trig_problem (α : ℝ) (h : Real.tan α = 3) : Prop :=
  Real.cos (α + Real.pi / 4) ^ 2 - Real.cos (α - Real.pi / 4) ^ 2 = -3 / 5

theorem trig_proof (α : ℝ) (h : Real.tan α = 3) : Real.cos (α + Real.pi / 4) ^ 2 - Real.cos (α - Real.pi / 4) ^ 2 = -3 / 5 :=
by
  sorry

end NUMINAMATH_GPT_trig_proof_l1277_127762


namespace NUMINAMATH_GPT_find_x_to_be_2_l1277_127786

variable (x : ℝ)

def a := (2, x)
def b := (3, x + 1)

theorem find_x_to_be_2 (h : a x = b x) : x = 2 := by
  sorry

end NUMINAMATH_GPT_find_x_to_be_2_l1277_127786


namespace NUMINAMATH_GPT_annie_overtakes_bonnie_l1277_127712

-- Define the conditions
def track_circumference : ℝ := 300
def bonnie_speed (v : ℝ) : ℝ := v
def annie_speed (v : ℝ) : ℝ := 1.5 * v

-- Define the statement for proving the number of laps completed by Annie when she first overtakes Bonnie
theorem annie_overtakes_bonnie (v t : ℝ) : 
  bonnie_speed v * t = track_circumference * 2 → 
  annie_speed v * t = track_circumference * 3 :=
by
  sorry

end NUMINAMATH_GPT_annie_overtakes_bonnie_l1277_127712


namespace NUMINAMATH_GPT_nicky_catches_up_time_l1277_127713

theorem nicky_catches_up_time
  (head_start : ℕ := 12)
  (cristina_speed : ℕ := 5)
  (nicky_speed : ℕ := 3)
  (head_start_distance : ℕ := nicky_speed * head_start)
  (time_to_catch_up : ℕ := 36 / 2) -- 36 is the head start distance of 36 meters
  (total_time : ℕ := time_to_catch_up + head_start)  -- Total time Nicky runs before Cristina catches up
  : total_time = 30 := sorry

end NUMINAMATH_GPT_nicky_catches_up_time_l1277_127713


namespace NUMINAMATH_GPT_cricket_player_avg_runs_l1277_127749

theorem cricket_player_avg_runs (A : ℝ) :
  (13 * A + 92 = 14 * (A + 5)) → A = 22 :=
by
  intro h1
  have h2 : 13 * A + 92 = 14 * A + 70 := by sorry
  have h3 : 92 - 70 = 14 * A - 13 * A := by sorry
  sorry

end NUMINAMATH_GPT_cricket_player_avg_runs_l1277_127749


namespace NUMINAMATH_GPT_extra_people_needed_l1277_127751

theorem extra_people_needed 
  (initial_people : ℕ) 
  (initial_time : ℕ) 
  (final_time : ℕ) 
  (work_done : ℕ) 
  (all_paint_same_rate : initial_people * initial_time = work_done) :
  initial_people = 8 →
  initial_time = 3 →
  final_time = 2 →
  work_done = 24 →
  ∃ extra_people : ℕ, extra_people = 4 :=
by
  sorry

end NUMINAMATH_GPT_extra_people_needed_l1277_127751


namespace NUMINAMATH_GPT_cos_identity_l1277_127779

open Real

theorem cos_identity
  (θ : ℝ)
  (h1 : cos ((5 * π) / 12 + θ) = 3 / 5)
  (h2 : -π < θ ∧ θ < -π / 2) :
  cos ((π / 12) - θ) = -4 / 5 :=
by
  sorry

end NUMINAMATH_GPT_cos_identity_l1277_127779


namespace NUMINAMATH_GPT_arith_seq_sum_of_terms_l1277_127754

theorem arith_seq_sum_of_terms 
  (a : ℕ → ℝ) (d : ℝ) 
  (h_arith_seq : ∀ n, a (n + 1) = a n + d)
  (h_pos_diff : 0 < d) 
  (h_first_three_sum : a 0 + a 1 + a 2 = 15) 
  (h_first_three_prod : a 0 * a 1 * a 2 = 80) : 
  a 10 + a 11 + a 12 = 105 := sorry

end NUMINAMATH_GPT_arith_seq_sum_of_terms_l1277_127754


namespace NUMINAMATH_GPT_derivative_y_l1277_127758

noncomputable def y (x : ℝ) : ℝ :=
  (Real.sqrt (9 * x^2 - 12 * x + 5)) * Real.arctan (3 * x - 2) - 
  Real.log (3 * x - 2 + Real.sqrt (9 * x^2 - 12 * x + 5))

theorem derivative_y (x : ℝ) :
  ∃ (f' : ℝ → ℝ), deriv y x = f' x ∧ f' x = (9 * x - 6) * Real.arctan (3 * x - 2) / 
  Real.sqrt (9 * x^2 - 12 * x + 5) :=
sorry

end NUMINAMATH_GPT_derivative_y_l1277_127758


namespace NUMINAMATH_GPT_problem_correct_choice_l1277_127711

-- Definitions of the propositions
def p : Prop := ∃ n : ℕ, 3 = 2 * n + 1
def q : Prop := ∃ n : ℕ, 5 = 2 * n

-- The problem statement
theorem problem_correct_choice : p ∨ q :=
sorry

end NUMINAMATH_GPT_problem_correct_choice_l1277_127711


namespace NUMINAMATH_GPT_fraction_in_between_l1277_127723

variable {r u s v : ℤ}

/-- Assumes r, u, s, v be positive integers such that su - rv = 1 --/
theorem fraction_in_between (h1 : r > 0) (h2 : u > 0) (h3 : s > 0) (h4 : v > 0) (h5 : s * u - r * v = 1) :
  ∀ ⦃x num denom : ℤ⦄, r * denom = num * u → s * denom = (num + 1) * v → r * v ≤ num * denom - 1 / u * v * denom
   ∧ num * denom - 1 / u * v * denom ≤ s * v :=
sorry

end NUMINAMATH_GPT_fraction_in_between_l1277_127723


namespace NUMINAMATH_GPT_system_of_inequalities_l1277_127750

theorem system_of_inequalities :
  ∃ (a b : ℤ), 
  (11 > 2 * a - b) ∧ 
  (25 > 2 * b - a) ∧ 
  (42 < 3 * b - a) ∧ 
  (46 < 2 * a + b) ∧ 
  (a = 14) ∧ 
  (b = 19) := 
sorry

end NUMINAMATH_GPT_system_of_inequalities_l1277_127750


namespace NUMINAMATH_GPT_calculate_b_50_l1277_127796

def sequence_b : ℕ → ℤ
| 0 => sorry -- This case is not used.
| 1 => 3
| (n + 2) => sequence_b (n + 1) + 3 * (n + 1) + 1

theorem calculate_b_50 : sequence_b 50 = 3727 := 
by
    sorry

end NUMINAMATH_GPT_calculate_b_50_l1277_127796


namespace NUMINAMATH_GPT_arith_seq_problem_l1277_127738

noncomputable def arithmetic_sequence (a : ℕ → ℝ) (a1 d : ℝ) : Prop :=
  ∀ n, a n = a1 + (n - 1) * d

theorem arith_seq_problem 
  (a : ℕ → ℝ) (a1 d : ℝ)
  (h1 : arithmetic_sequence a a1 d)
  (h2 : a 1 + 3 * a 8 + a 15 = 120) : 
  3 * a 9 - a 11 = 48 :=
by 
  sorry

end NUMINAMATH_GPT_arith_seq_problem_l1277_127738


namespace NUMINAMATH_GPT_discount_rate_on_pony_jeans_l1277_127708

theorem discount_rate_on_pony_jeans (F P : ℝ) 
  (h1 : F + P = 25)
  (h2 : 45 * F + 36 * P = 900) :
  P = 25 :=
by
  sorry

end NUMINAMATH_GPT_discount_rate_on_pony_jeans_l1277_127708


namespace NUMINAMATH_GPT_log_neq_x_minus_one_l1277_127748

theorem log_neq_x_minus_one (x : ℝ) (h₁ : 0 < x) : Real.log x ≠ x - 1 :=
sorry

end NUMINAMATH_GPT_log_neq_x_minus_one_l1277_127748


namespace NUMINAMATH_GPT_flat_fee_first_night_l1277_127798

-- Given conditions
variable (f n : ℝ)
axiom alice_cost : f + 3 * n = 245
axiom bob_cost : f + 5 * n = 350

-- Main theorem to prove
theorem flat_fee_first_night : f = 87.5 := by sorry

end NUMINAMATH_GPT_flat_fee_first_night_l1277_127798


namespace NUMINAMATH_GPT_condition_for_equation_l1277_127767

theorem condition_for_equation (a b c : ℕ) (ha : 0 < a ∧ a < 20) (hb : 0 < b ∧ b < 20) (hc : 0 < c ∧ c < 20) :
  (20 * a + b) * (20 * a + c) = 400 * a^2 + 200 * a + b * c ↔ b + c = 10 :=
by
  sorry

end NUMINAMATH_GPT_condition_for_equation_l1277_127767


namespace NUMINAMATH_GPT_malcolm_red_lights_bought_l1277_127753

-- Define the problem's parameters and conditions
variable (R : ℕ) (B : ℕ := 3 * R) (G : ℕ := 6)
variable (initial_white_lights : ℕ := 59) (remaining_colored_lights : ℕ := 5)

-- The total number of colored lights that he still needs to replace the white lights
def total_colored_lights_needed : ℕ := initial_white_lights - remaining_colored_lights

-- Total colored lights bought so far
def total_colored_lights_bought : ℕ := R + B + G

-- The main theorem to prove that Malcolm bought 12 red lights
theorem malcolm_red_lights_bought (h : total_colored_lights_bought = total_colored_lights_needed) :
  R = 12 := by
  sorry

end NUMINAMATH_GPT_malcolm_red_lights_bought_l1277_127753


namespace NUMINAMATH_GPT_circle_eq_l1277_127782

theorem circle_eq (A B C : ℝ × ℝ)
  (hA : A = (2, 0))
  (hB : B = (4, 0))
  (hC : C = (0, 2)) :
  ∃ (h: ℝ), (x - 3) ^ 2 + (y - 3) ^ 2 = h :=
by 
  use 10
  -- additional steps to rigorously prove the result would go here
  sorry

end NUMINAMATH_GPT_circle_eq_l1277_127782


namespace NUMINAMATH_GPT_velvet_needed_for_box_l1277_127719

theorem velvet_needed_for_box :
  let long_side_area := 2 * (8 * 6)
  let short_side_area := 2 * (5 * 6)
  let top_and_bottom_area := 2 * 40
  long_side_area + short_side_area + top_and_bottom_area = 236 := by
{
  let long_side_area := 2 * (8 * 6)
  let short_side_area := 2 * (5 * 6)
  let top_and_bottom_area := 2 * 40
  sorry
}

end NUMINAMATH_GPT_velvet_needed_for_box_l1277_127719


namespace NUMINAMATH_GPT_speed_conversion_l1277_127776

theorem speed_conversion (speed_m_s : ℚ) (conversion_factor : ℚ) :
  speed_m_s = 8 / 26 → conversion_factor = 3.6 →
  speed_m_s * conversion_factor = 1.1077 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  sorry

end NUMINAMATH_GPT_speed_conversion_l1277_127776


namespace NUMINAMATH_GPT_abigail_saving_period_l1277_127737

-- Define the conditions
def amount_saved_each_month : ℕ := 4000
def total_amount_saved : ℕ := 48000

-- State the theorem
theorem abigail_saving_period : total_amount_saved / amount_saved_each_month = 12 := by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_abigail_saving_period_l1277_127737


namespace NUMINAMATH_GPT_standard_eq_circle_l1277_127744

noncomputable def circle_eq (x y : ℝ) (r : ℝ) : Prop :=
  (x - 4)^2 + (y - 4)^2 = 16 ∨ (x - 1)^2 + (y + 1)^2 = 1

theorem standard_eq_circle {x y : ℝ}
  (h1 : 5 * x - 3 * y = 8)
  (h2 : abs x = abs y) :
  ∃ r : ℝ, circle_eq x y r :=
by {
  sorry
}

end NUMINAMATH_GPT_standard_eq_circle_l1277_127744


namespace NUMINAMATH_GPT_find_a_range_for_two_distinct_roots_l1277_127769

def f (x : ℝ) : ℝ := x^3 - 3 * x + 5

theorem find_a_range_for_two_distinct_roots :
  ∀ (a : ℝ), 3 ≤ a ∧ a ≤ 7 → ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ f x1 = a ∧ f x2 = a :=
by
  -- The proof will be here
  sorry

end NUMINAMATH_GPT_find_a_range_for_two_distinct_roots_l1277_127769


namespace NUMINAMATH_GPT_complex_sum_cubics_eq_zero_l1277_127743

-- Define the hypothesis: omega is a nonreal root of x^3 = 1
def is_nonreal_root_of_cubic (ω : ℂ) : Prop :=
  ω^3 = 1 ∧ ω ≠ 1

-- Now state the theorem to prove the expression evaluates to 0
theorem complex_sum_cubics_eq_zero (ω : ℂ) (h : is_nonreal_root_of_cubic ω) :
  (2 - 2*ω + 2*ω^2)^3 + (2 + 2*ω - 2*ω^2)^3 = 0 :=
by
  -- This is where the proof would go. 
  sorry

end NUMINAMATH_GPT_complex_sum_cubics_eq_zero_l1277_127743


namespace NUMINAMATH_GPT_original_volume_l1277_127792

variable (V : ℝ)

theorem original_volume (h1 : (1/4) * V = V₁)
                       (h2 : (1/4) * V₁ = V₂)
                       (h3 : (1/3) * V₂ = 0.4) : 
                       V = 19.2 := 
by 
  sorry

end NUMINAMATH_GPT_original_volume_l1277_127792


namespace NUMINAMATH_GPT_sum_of_perimeters_triangles_l1277_127761

theorem sum_of_perimeters_triangles (a : ℕ → ℕ) (side_length : ℕ) (P : ℕ → ℕ):
  (∀ n : ℕ, a 0 = side_length ∧ P 0 = 3 * a 0) →
  (∀ n : ℕ, a (n + 1) = a n / 2 ∧ P (n + 1) = 3 * a (n + 1)) →
  (side_length = 45) →
  ∑' n, P n = 270 :=
by
  -- the proof would continue here
  sorry

end NUMINAMATH_GPT_sum_of_perimeters_triangles_l1277_127761


namespace NUMINAMATH_GPT_product_of_two_numbers_l1277_127793

theorem product_of_two_numbers (x y : ℝ) (h1 : x + y = 27) (h2 : x - y = 9) : x * y = 162 := 
by {
  sorry
}

end NUMINAMATH_GPT_product_of_two_numbers_l1277_127793


namespace NUMINAMATH_GPT_clare_bought_loaves_l1277_127775

-- Define the given conditions
def initial_amount : ℕ := 47
def remaining_amount : ℕ := 35
def cost_per_loaf : ℕ := 2
def cost_per_carton : ℕ := 2
def number_of_cartons : ℕ := 2

-- Required to prove the number of loaves of bread bought by Clare
theorem clare_bought_loaves (initial_amount remaining_amount cost_per_loaf cost_per_carton number_of_cartons : ℕ) 
    (h1 : initial_amount = 47) 
    (h2 : remaining_amount = 35) 
    (h3 : cost_per_loaf = 2) 
    (h4 : cost_per_carton = 2) 
    (h5 : number_of_cartons = 2) : 
    (initial_amount - remaining_amount - cost_per_carton * number_of_cartons) / cost_per_loaf = 4 :=
by sorry

end NUMINAMATH_GPT_clare_bought_loaves_l1277_127775


namespace NUMINAMATH_GPT_probability_of_perfect_square_is_correct_l1277_127763

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def probability_perfect_square (p : ℚ) : ℚ :=
  let less_than_equal_60 := 7 * p
  let greater_than_60 := 4 * 4 * p
  less_than_equal_60 + greater_than_60

theorem probability_of_perfect_square_is_correct :
  let p : ℚ := 1 / 300
  probability_perfect_square p = 23 / 300 :=
sorry

end NUMINAMATH_GPT_probability_of_perfect_square_is_correct_l1277_127763


namespace NUMINAMATH_GPT_total_amount_of_money_l1277_127772

theorem total_amount_of_money (P1 : ℝ) (interest_total : ℝ)
  (hP1 : P1 = 299.99999999999994) (hInterest : interest_total = 144) :
  ∃ T : ℝ, T = 3000 :=
by
  sorry

end NUMINAMATH_GPT_total_amount_of_money_l1277_127772


namespace NUMINAMATH_GPT_true_root_30_40_l1277_127766

noncomputable def u (x : ℝ) : ℝ := Real.sqrt (x + 15)
noncomputable def original_eqn (x : ℝ) : Prop := u x - 3 / (u x) = 4

theorem true_root_30_40 : ∃ (x : ℝ), 30 < x ∧ x < 40 ∧ original_eqn x :=
by
  sorry

end NUMINAMATH_GPT_true_root_30_40_l1277_127766


namespace NUMINAMATH_GPT_slope_at_A_is_7_l1277_127710

def curve (x : ℝ) : ℝ := x^2 + 3 * x

def point_A : ℝ × ℝ := (2, 10)

theorem slope_at_A_is_7 : (deriv curve 2) = 7 := 
by
  sorry

end NUMINAMATH_GPT_slope_at_A_is_7_l1277_127710


namespace NUMINAMATH_GPT_tenth_digit_of_expression_l1277_127747

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

def tenth_digit (n : ℕ) : ℕ := (n / 10) % 10

theorem tenth_digit_of_expression : 
  tenth_digit ((factorial 5 * factorial 5 - factorial 5 * factorial 3) / 5) = 3 :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_tenth_digit_of_expression_l1277_127747


namespace NUMINAMATH_GPT_red_peaches_count_l1277_127706

/-- Math problem statement:
There are some red peaches and 16 green peaches in the basket.
There is 1 more red peach than green peaches in the basket.
Prove that the number of red peaches in the basket is 17.
--/

-- Let G be the number of green peaches and R be the number of red peaches.
def G : ℕ := 16
def R : ℕ := G + 1

theorem red_peaches_count : R = 17 := by
  sorry

end NUMINAMATH_GPT_red_peaches_count_l1277_127706


namespace NUMINAMATH_GPT_day_before_yesterday_l1277_127702

theorem day_before_yesterday (day_after_tomorrow_is_monday : String) : String :=
by
  have tomorrow := "Sunday"
  have today := "Saturday"
  exact today

end NUMINAMATH_GPT_day_before_yesterday_l1277_127702


namespace NUMINAMATH_GPT_max_remainder_is_8_l1277_127720

theorem max_remainder_is_8 (d q r : ℕ) (h1 : d = 9) (h2 : q = 6) (h3 : r < d) : 
  r ≤ (d - 1) :=
by 
  sorry

end NUMINAMATH_GPT_max_remainder_is_8_l1277_127720


namespace NUMINAMATH_GPT_angle_C_ne_5pi_over_6_l1277_127765

-- Define the triangle ∆ABC
variables (A B C : ℝ)

-- Assume the conditions provided
axiom condition_1 : 3 * Real.sin A + 4 * Real.cos B = 6
axiom condition_2 : 3 * Real.cos A + 4 * Real.sin B = 1

-- State that the size of angle C cannot be 5π/6
theorem angle_C_ne_5pi_over_6 : C ≠ 5 * Real.pi / 6 :=
sorry

end NUMINAMATH_GPT_angle_C_ne_5pi_over_6_l1277_127765


namespace NUMINAMATH_GPT_quadratic_real_roots_l1277_127755

theorem quadratic_real_roots (a: ℝ) :
  ∀ x: ℝ, (a-6) * x^2 - 8 * x + 9 = 0 ↔ (a ≤ 70/9 ∧ a ≠ 6) :=
  sorry

end NUMINAMATH_GPT_quadratic_real_roots_l1277_127755


namespace NUMINAMATH_GPT_remaining_problems_l1277_127760

-- Define the conditions
def worksheets_total : ℕ := 15
def worksheets_graded : ℕ := 7
def problems_per_worksheet : ℕ := 3

-- Define the proof goal
theorem remaining_problems : (worksheets_total - worksheets_graded) * problems_per_worksheet = 24 :=
by
  sorry

end NUMINAMATH_GPT_remaining_problems_l1277_127760


namespace NUMINAMATH_GPT_jordan_rectangle_length_l1277_127781

variables (L : ℝ)

-- Condition: Carol's rectangle measures 12 inches by 15 inches.
def carol_area : ℝ := 12 * 15

-- Condition: Jordan's rectangle has the same area as Carol's rectangle.
def jordan_area : ℝ := carol_area

-- Condition: Jordan's rectangle is 20 inches wide.
def jordan_width : ℝ := 20

-- Proposition: Length of Jordan's rectangle == 9 inches.
theorem jordan_rectangle_length : L * jordan_width = jordan_area → L = 9 := 
by
  intros h
  sorry

end NUMINAMATH_GPT_jordan_rectangle_length_l1277_127781

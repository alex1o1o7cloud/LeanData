import Mathlib

namespace P_intersect_Q_empty_l714_71447

def is_element_of_P (x : ℝ) : Prop :=
  ∃ (k : ℤ), x = k / 2 + 1 / 4

def is_element_of_Q (x : ℝ) : Prop :=
  ∃ (k : ℤ), x = k / 2 + 1 / 2

theorem P_intersect_Q_empty : ∀ x, is_element_of_P x → is_element_of_Q x → false :=
by
  intro x hP hQ
  sorry

end P_intersect_Q_empty_l714_71447


namespace find_two_digit_numbers_l714_71409

theorem find_two_digit_numbers :
  ∀ (a b : ℕ), (1 ≤ a ∧ a ≤ 9) → (0 ≤ b ∧ b ≤ 9) → (10 * a + b = 3 * a * b) → (10 * a + b = 15 ∨ 10 * a + b = 24) :=
by
  intros
  sorry

end find_two_digit_numbers_l714_71409


namespace percentage_non_honda_red_cars_l714_71421

theorem percentage_non_honda_red_cars 
  (total_cars : ℕ)
  (honda_cars : ℕ)
  (toyota_cars : ℕ)
  (ford_cars : ℕ)
  (other_cars : ℕ)
  (perc_red_honda : ℕ)
  (perc_red_toyota : ℕ)
  (perc_red_ford : ℕ)
  (perc_red_other : ℕ)
  (perc_total_red : ℕ)
  (hyp_total_cars : total_cars = 900)
  (hyp_honda_cars : honda_cars = 500)
  (hyp_toyota_cars : toyota_cars = 200)
  (hyp_ford_cars : ford_cars = 150)
  (hyp_other_cars : other_cars = 50)
  (hyp_perc_red_honda : perc_red_honda = 90)
  (hyp_perc_red_toyota : perc_red_toyota = 75)
  (hyp_perc_red_ford : perc_red_ford = 30)
  (hyp_perc_red_other : perc_red_other = 20)
  (hyp_perc_total_red : perc_total_red = 60) :
  (205 / 400) * 100 = 51.25 := 
by {
  sorry
}

end percentage_non_honda_red_cars_l714_71421


namespace madeline_has_five_boxes_l714_71441

theorem madeline_has_five_boxes 
    (total_crayons_per_box : ℕ)
    (not_used_fraction1 : ℚ)
    (not_used_fraction2 : ℚ)
    (used_fraction2 : ℚ)
    (total_boxes_not_used : ℚ)
    (total_unused_crayons : ℕ)
    (unused_in_last_box : ℚ)
    (total_boxes : ℕ) :
    total_crayons_per_box = 24 →
    not_used_fraction1 = 5 / 8 →
    not_used_fraction2 = 1 / 3 →
    used_fraction2 = 2 / 3 →
    total_boxes_not_used = 4 →
    total_unused_crayons = 70 →
    total_boxes = 5 :=
by
  -- Insert proof here
  sorry

end madeline_has_five_boxes_l714_71441


namespace determine_gallons_l714_71443

def current_amount : ℝ := 7.75
def desired_total : ℝ := 14.75
def needed_to_add (x : ℝ) : Prop := desired_total = current_amount + x

theorem determine_gallons : needed_to_add 7 :=
by
  sorry

end determine_gallons_l714_71443


namespace intersection_condition_l714_71438

noncomputable def M : Set ℝ := {x | -1 ≤ x ∧ x < 2}
noncomputable def N (k : ℝ) : Set ℝ := {x | x ≤ k}

theorem intersection_condition (k : ℝ) (h : M ⊆ N k) : k ≥ 2 :=
  sorry

end intersection_condition_l714_71438


namespace rect_tiling_l714_71479

theorem rect_tiling (a b : ℕ) : ∃ (w h : ℕ), w = max 1 (2 * a) ∧ h = 2 * b ∧ (∃ f : ℕ → ℕ → (ℕ × ℕ), ∀ i j, (i < w ∧ j < h → f i j = (a, b))) := sorry

end rect_tiling_l714_71479


namespace angle_of_inclination_range_l714_71474

theorem angle_of_inclination_range (a : ℝ) :
  (∃ m : ℝ, ax + (a + 1)*m + 2 = 0 ∧ (m < 0 ∨ m > 1)) ↔ (a < -1/2 ∨ a > 0) := sorry

end angle_of_inclination_range_l714_71474


namespace animals_left_in_barn_l714_71475

-- Define the conditions
def num_pigs : Nat := 156
def num_cows : Nat := 267
def num_sold : Nat := 115

-- Define the question
def num_left := num_pigs + num_cows - num_sold

-- State the theorem
theorem animals_left_in_barn : num_left = 308 :=
by
  sorry

end animals_left_in_barn_l714_71475


namespace math_solution_l714_71431

noncomputable def math_problem (x y z : ℝ) : Prop :=
  (0 ≤ x) ∧ (0 ≤ y) ∧ (0 ≤ z) ∧ (x + y + z = 1) → 
  (x^2 * y^2 + y^2 * z^2 + z^2 * x^2 + x^2 * y^2 * z^2 ≤ 1 / 16)

theorem math_solution (x y z : ℝ) :
  math_problem x y z := 
by
  sorry

end math_solution_l714_71431


namespace directrix_of_parabola_l714_71444

-- Define the given condition: the equation of the parabola
def given_parabola (x : ℝ) : ℝ := 4 * x ^ 2

-- State the theorem to be proven
theorem directrix_of_parabola : 
  (∀ x : ℝ, given_parabola x = 4 * x ^ 2) → 
  (y = -1 / 16) :=
sorry

end directrix_of_parabola_l714_71444


namespace fraction_absent_l714_71413

theorem fraction_absent (total_students present_students : ℕ) (h1 : total_students = 28) (h2 : present_students = 20) : 
  (total_students - present_students) / total_students = 2 / 7 :=
by
  sorry

end fraction_absent_l714_71413


namespace roger_current_money_l714_71473

def roger_initial_money : ℕ := 16
def roger_birthday_money : ℕ := 28
def roger_spent_money : ℕ := 25

theorem roger_current_money : roger_initial_money + roger_birthday_money - roger_spent_money = 19 := 
by sorry

end roger_current_money_l714_71473


namespace blocks_fit_into_box_l714_71494

theorem blocks_fit_into_box :
  let box_height := 8
  let box_width := 10
  let box_length := 12
  let block_height := 3
  let block_width := 2
  let block_length := 4
  let box_volume := box_height * box_width * box_length
  let block_volume := block_height * block_width * block_length
  let num_blocks := box_volume / block_volume
  num_blocks = 40 :=
by
  sorry

end blocks_fit_into_box_l714_71494


namespace weight_of_newcomer_l714_71429

theorem weight_of_newcomer (avg_old W_initial : ℝ) 
  (h_weight_range : 400 ≤ W_initial ∧ W_initial ≤ 420)
  (h_avg_increase : avg_old + 3.5 = (W_initial - 47 + W_new) / 6)
  (h_person_replaced : 47 = 47) :
  W_new = 68 := 
sorry

end weight_of_newcomer_l714_71429


namespace exists_sum_of_divisibles_l714_71487

theorem exists_sum_of_divisibles : ∃ (a b: ℕ), a + b = 316 ∧ (13 ∣ a) ∧ (11 ∣ b) :=
by
  existsi 52
  existsi 264
  sorry

end exists_sum_of_divisibles_l714_71487


namespace handshake_problem_l714_71434

-- Defining the necessary elements:
def num_people : Nat := 12
def num_handshakes_per_person : Nat := num_people - 2

-- Defining the total number of handshakes. Each handshake is counted twice.
def total_handshakes : Nat := (num_people * num_handshakes_per_person) / 2

-- The theorem statement:
theorem handshake_problem : total_handshakes = 60 :=
by
  sorry

end handshake_problem_l714_71434


namespace min_height_box_l714_71451

noncomputable def min_height (x : ℝ) : ℝ :=
  if h : x ≥ (5 : ℝ) then x + 5 else 0

theorem min_height_box (x : ℝ) (hx : 3*x^2 + 10*x - 65 ≥ 0) : min_height x = 10 :=
by
  sorry

end min_height_box_l714_71451


namespace haleigh_needs_46_leggings_l714_71491

-- Define the number of each type of animal
def num_dogs : ℕ := 4
def num_cats : ℕ := 3
def num_spiders : ℕ := 2
def num_parrot : ℕ := 1

-- Define the number of legs each type of animal has
def legs_dog : ℕ := 4
def legs_cat : ℕ := 4
def legs_spider : ℕ := 8
def legs_parrot : ℕ := 2

-- Define the total number of legs function
def total_leggings (d c s p : ℕ) (ld lc ls lp : ℕ) : ℕ :=
  d * ld + c * lc + s * ls + p * lp

-- The statement to be proven
theorem haleigh_needs_46_leggings : total_leggings num_dogs num_cats num_spiders num_parrot legs_dog legs_cat legs_spider legs_parrot = 46 := by
  sorry

end haleigh_needs_46_leggings_l714_71491


namespace time_for_Harish_to_paint_alone_l714_71411

theorem time_for_Harish_to_paint_alone (H : ℝ) (h1 : H > 0) (h2 :  (1 / 6 + 1 / H) = 1 / 2 ) : H = 3 :=
sorry

end time_for_Harish_to_paint_alone_l714_71411


namespace part1_part2_l714_71464

def unitPrices (x : ℕ) (y : ℕ) : Prop :=
  (20 * x = 16 * (y + 20)) ∧ (x = y + 20)

def maxBoxes (a : ℕ) : Prop :=
  ∀ b, (100 * a + 80 * b ≤ 4600) → (a + b = 50)

theorem part1 (x : ℕ) :
  unitPrices x (x - 20) → x = 100 ∧ (x - 20 = 80) :=
by
  sorry

theorem part2 :
  maxBoxes 30 :=
by
  sorry

end part1_part2_l714_71464


namespace linear_eq_m_minus_2n_zero_l714_71405

theorem linear_eq_m_minus_2n_zero (m n : ℕ) (x y : ℝ) 
  (h1 : 2 * x ^ (m - 1) + 3 * y ^ (2 * n - 1) = 7)
  (h2 : m - 1 = 1) (h3 : 2 * n - 1 = 1) : 
  m - 2 * n = 0 := 
sorry

end linear_eq_m_minus_2n_zero_l714_71405


namespace expr_D_is_diff_of_squares_l714_71472

-- Definitions for the expressions
def expr_A (a b : ℤ) : ℤ := (a + 2 * b) * (-a - 2 * b)
def expr_B (m n : ℤ) : ℤ := (2 * m - 3 * n) * (3 * n - 2 * m)
def expr_C (x y : ℤ) : ℤ := (2 * x - 3 * y) * (3 * x + 2 * y)
def expr_D (a b : ℤ) : ℤ := (a - b) * (-b - a)

-- Theorem stating that Expression D can be calculated using the difference of squares formula
theorem expr_D_is_diff_of_squares (a b : ℤ) : expr_D a b = a^2 - b^2 :=
by sorry

end expr_D_is_diff_of_squares_l714_71472


namespace steve_has_7_fewer_b_berries_l714_71495

-- Define the initial number of berries Stacy has
def stacy_initial_berries : ℕ := 32

-- Define the number of berries Steve takes from Stacy
def steve_takes : ℕ := 4

-- Define the initial number of berries Steve has
def steve_initial_berries : ℕ := 21

-- Using the given conditions, prove that Steve has 7 fewer berries compared to Stacy's initial amount
theorem steve_has_7_fewer_b_berries :
  stacy_initial_berries - (steve_initial_berries + steve_takes) = 7 := 
by
  sorry

end steve_has_7_fewer_b_berries_l714_71495


namespace marked_price_percentage_l714_71412

theorem marked_price_percentage
  (CP MP SP : ℝ)
  (h_profit : SP = 1.08 * CP)
  (h_discount : SP = 0.8307692307692308 * MP) :
  MP = CP * 1.3 :=
by sorry

end marked_price_percentage_l714_71412


namespace hotel_cost_l714_71415

/--
Let the total cost of the hotel be denoted as x dollars.
Initially, the cost for each of the original four colleagues is x / 4.
After three more colleagues joined, the cost per person becomes x / 7.
Given that the amount paid by each of the original four decreased by 15,
prove that the total cost of the hotel is 140 dollars.
-/
theorem hotel_cost (x : ℕ) (h : x / 4 - 15 = x / 7) : x = 140 := 
by
  sorry

end hotel_cost_l714_71415


namespace sequence_difference_l714_71492

theorem sequence_difference
  (a : ℕ → ℤ)
  (h : ∀ n : ℕ, a (n + 1) - a n - n = 0) :
  a 2017 - a 2016 = 2016 :=
by
  sorry

end sequence_difference_l714_71492


namespace geometric_sequence_l714_71454

theorem geometric_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) (h1 : a 1 = 1) 
  (h2 : (3 * S 1, 2 * S 2, S 3) = (3 * S 1, 2 * S 2, S 3) ∧ (4 * S 2 = 3 * S 1 + S 3)) 
  (hq_pos : q ≠ 0) 
  (hq : ∀ n, a (n + 1) = a n * q):
  ∀ n, a n = 3^(n-1) :=
by
  sorry

end geometric_sequence_l714_71454


namespace find_x9_y9_l714_71422

theorem find_x9_y9 (x y : ℝ) (h1 : x^3 + y^3 = 7) (h2 : x^6 + y^6 = 49) : x^9 + y^9 = 343 :=
by
  sorry

end find_x9_y9_l714_71422


namespace frisbee_sales_total_receipts_l714_71471

theorem frisbee_sales_total_receipts 
  (total_frisbees : ℕ) 
  (price_3_frisbee : ℕ) 
  (price_4_frisbee : ℕ) 
  (sold_3 : ℕ) 
  (sold_4 : ℕ) 
  (total_receipts : ℕ) 
  (h1 : total_frisbees = 60) 
  (h2 : price_3_frisbee = 3)
  (h3 : price_4_frisbee = 4) 
  (h4 : sold_3 + sold_4 = total_frisbees) 
  (h5 : sold_4 ≥ 24)
  (h6 : total_receipts = sold_3 * price_3_frisbee + sold_4 * price_4_frisbee) :
  total_receipts = 204 :=
sorry

end frisbee_sales_total_receipts_l714_71471


namespace count_L_shapes_l714_71483

theorem count_L_shapes (m n : ℕ) (hm : 1 ≤ m) (hn : 1 ≤ n) : 
  ∃ k, k = 4 * (m - 1) * (n - 1) :=
by
  sorry

end count_L_shapes_l714_71483


namespace symmetry_condition_l714_71458

open Real

noncomputable def f (x : ℝ) : ℝ := 2 * sin (x + π / 3)

theorem symmetry_condition (ϕ : ℝ) (hϕ : |ϕ| ≤ π / 2)
    (hxy: ∀ x : ℝ, f (x + ϕ) = f (-x + ϕ)) : ϕ = π / 6 :=
by
  -- Since the problem specifically asks for the statement only and not the proof steps,
  -- a "sorry" is used to skip the proof content.
  sorry

end symmetry_condition_l714_71458


namespace sum_of_consecutive_integers_with_product_812_l714_71456

theorem sum_of_consecutive_integers_with_product_812 : ∃ (x : ℕ), (x * (x + 1) = 812) ∧ (x + (x + 1) = 57) :=
by 
  sorry

end sum_of_consecutive_integers_with_product_812_l714_71456


namespace plane_divided_by_n_lines_l714_71401

-- Definition of the number of regions created by n lines in a plane
def regions (n : ℕ) : ℕ :=
  if n = 0 then 1 else (n * (n + 1)) / 2 + 1 -- Using the given formula directly

-- Theorem statement to prove the formula holds
theorem plane_divided_by_n_lines (n : ℕ) : 
  regions n = (n * (n + 1)) / 2 + 1 :=
sorry

end plane_divided_by_n_lines_l714_71401


namespace last_two_digits_of_7_pow_10_l714_71414

theorem last_two_digits_of_7_pow_10 :
  (7 ^ 10) % 100 = 49 := by
  sorry

end last_two_digits_of_7_pow_10_l714_71414


namespace triangle_median_length_l714_71466

variable (XY XZ XM YZ : ℝ)

theorem triangle_median_length :
  XY = 6 →
  XZ = 8 →
  XM = 5 →
  YZ = 10 := by
  sorry

end triangle_median_length_l714_71466


namespace leak_time_to_empty_tank_l714_71419

theorem leak_time_to_empty_tank :
  let rateA := 1 / 2  -- rate at which pipe A fills the tank (tanks per hour)
  let rateB := 2 / 3  -- rate at which pipe B fills the tank (tanks per hour)
  let combined_rate_without_leak := rateA + rateB  -- combined rate without leak
  let combined_rate_with_leak := 1 / 1.75  -- combined rate with leak (tanks per hour)
  let leak_rate := combined_rate_without_leak - combined_rate_with_leak  -- rate of the leak (tanks per hour)
  60 / leak_rate = 100.8 :=  -- time to empty the tank by the leak (minutes)
    by sorry

end leak_time_to_empty_tank_l714_71419


namespace find_n_divisible_by_35_l714_71485

-- Define the five-digit number for some digit n
def num (n : ℕ) : ℕ := 80000 + n * 1000 + 975

-- Define the conditions
def divisible_by_5 (d : ℕ) : Prop := d % 5 = 0
def divisible_by_7 (d : ℕ) : Prop := d % 7 = 0
def divisible_by_35 (d : ℕ) : Prop := divisible_by_5 d ∧ divisible_by_7 d

-- Statement of the problem for proving given conditions and the correct answer
theorem find_n_divisible_by_35 : ∃ (n : ℕ), (num n % 35 = 0) ∧ n = 6 := by
  sorry

end find_n_divisible_by_35_l714_71485


namespace minimum_value_inequality_equality_condition_exists_l714_71463

theorem minimum_value_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  6 * c / (3 * a + b) + 6 * a / (b + 3 * c) + 2 * b / (a + c) ≥ 12 := by
  sorry

theorem equality_condition_exists : 
  ∃ a b c : ℝ, 0 < a ∧ 0 < b ∧ 0 < c ∧ (6 * c / (3 * a + b) + 6 * a / (b + 3 * c) + 2 * b / (a + c) = 12) := by
  sorry

end minimum_value_inequality_equality_condition_exists_l714_71463


namespace largest_value_among_expressions_l714_71465

def expA : ℕ := 3 + 1 + 2 + 4
def expB : ℕ := 3 * 1 + 2 + 4
def expC : ℕ := 3 + 1 * 2 + 4
def expD : ℕ := 3 + 1 + 2 * 4
def expE : ℕ := 3 * 1 * 2 * 4

theorem largest_value_among_expressions :
  expE > expA ∧ expE > expB ∧ expE > expC ∧ expE > expD :=
by
  -- Proof will go here
  sorry

end largest_value_among_expressions_l714_71465


namespace small_cubes_for_larger_cube_l714_71493

theorem small_cubes_for_larger_cube (VL VS : ℕ) (h : VL = 125 * VS) : (VL / VS = 125) :=
by {
    sorry
}

end small_cubes_for_larger_cube_l714_71493


namespace evaluate_expression_l714_71499

theorem evaluate_expression (x : ℝ) (h1 : x ≠ -2) (h2 : x ≠ 3) :
  (3 * x ^ 2 - 2 * x + 1) / ((x + 2) * (x - 3)) - (x ^ 2 - 5 * x + 6) / ((x + 2) * (x - 3)) =
  (2 * x ^ 2 + 3 * x - 5) / ((x + 2) * (x - 3)) :=
by
  sorry

end evaluate_expression_l714_71499


namespace project_completion_time_l714_71437

-- Definitions for conditions
def a_rate : ℚ := 1 / 20
def b_rate : ℚ := 1 / 30
def combined_rate : ℚ := a_rate + b_rate

-- Total days to complete the project
def total_days (x : ℚ) : Prop :=
  (x - 5) * a_rate + x * b_rate = 1

-- The theorem to be proven
theorem project_completion_time : ∃ (x : ℚ), total_days x ∧ x = 15 := by
  sorry

end project_completion_time_l714_71437


namespace quadratic_equiv_original_correct_transformation_l714_71445

theorem quadratic_equiv_original :
  (5 + 3*Real.sqrt 2) * x^2 + (3 + Real.sqrt 2) * x - 3 = 
  (7 + 4 * Real.sqrt 3) * x^2 + (2 + Real.sqrt 3) * x - 2 :=
sorry

theorem correct_transformation :
  ∃ r : ℝ, r = (9 / 7) - (4 * Real.sqrt 2 / 7) ∧ 
  ((5 + 3 * Real.sqrt 2) * x^2 + (3 + Real.sqrt 2) * x - 3) = 0 :=
sorry

end quadratic_equiv_original_correct_transformation_l714_71445


namespace problem1_problem2_l714_71468

theorem problem1 (x : ℝ) (a : ℝ) (h : a = 1) (hp : a < x ∧ x < 3 * a) (hq : 2 < x ∧ x < 3) : 2 < x ∧ x < 3 := 
by
  sorry

theorem problem2 (x : ℝ) (a : ℝ) (hp : 0 < a ∧ a < x ∧ x < 3 * a) (hq : 2 < x ∧ x < 3) (hsuff : ∀ (a x : ℝ), (2 < x ∧ x < 3) → a < x ∧ x < 3 * a) : 1 ≤ a ∧ a ≤ 2 := 
by
  sorry

end problem1_problem2_l714_71468


namespace find_x_l714_71432

theorem find_x (x : ℝ) : 
  let a := (4, 2)
  let b := (x, 3)
  (a.1 * b.1 + a.2 * b.2 = 0) → x = -3 / 2 :=
by
  intros a b h
  sorry

end find_x_l714_71432


namespace length_of_AC_l714_71478

-- Define the conditions: lengths and angle
def AB : ℝ := 10
def BC : ℝ := 10
def CD : ℝ := 15
def DA : ℝ := 15
def angle_ADC : ℝ := 120

-- Prove the length of diagonal AC is 15*sqrt(3)
theorem length_of_AC : 
  (CD ^ 2 + DA ^ 2 - 2 * CD * DA * Real.cos (angle_ADC * Real.pi / 180)) = (15 * Real.sqrt 3) ^ 2 :=
by
  sorry

end length_of_AC_l714_71478


namespace circle_center_tangent_lines_l714_71477

theorem circle_center_tangent_lines 
    (center : ℝ × ℝ)
    (h1 : 3 * center.1 + 4 * center.2 = 10)
    (h2 : center.1 = 3 * center.2) : 
    center = (30 / 13, 10 / 13) := 
by {
  sorry
}

end circle_center_tangent_lines_l714_71477


namespace ratio_of_common_differences_l714_71410

variable (x y d1 d2 : ℝ)

theorem ratio_of_common_differences (d1_nonzero : d1 ≠ 0) (d2_nonzero : d2 ≠ 0) 
  (seq1 : x + 4 * d1 = y) (seq2 : x + 5 * d2 = y) : d1 / d2 = 5 / 4 := 
sorry

end ratio_of_common_differences_l714_71410


namespace remainder_when_divided_by_x_minus_1_is_minus_2_l714_71427

def p (x : ℝ) : ℝ := x^4 - 2*x^2 + 4*x - 5

theorem remainder_when_divided_by_x_minus_1_is_minus_2 : (p 1) = -2 := 
by 
  -- Proof not required
  sorry

end remainder_when_divided_by_x_minus_1_is_minus_2_l714_71427


namespace mass_percentage_H_calculation_l714_71460

noncomputable def molar_mass_CaH2 : ℝ := 42.09
noncomputable def molar_mass_H2O : ℝ := 18.015
noncomputable def molar_mass_H2SO4 : ℝ := 98.079

noncomputable def moles_CaH2 : ℕ := 3
noncomputable def moles_H2O : ℕ := 4
noncomputable def moles_H2SO4 : ℕ := 2

noncomputable def mass_H_CaH2 : ℝ := 3 * 2 * 1.008
noncomputable def mass_H_H2O : ℝ := 4 * 2 * 1.008
noncomputable def mass_H_H2SO4 : ℝ := 2 * 2 * 1.008

noncomputable def total_mass_H : ℝ :=
  mass_H_CaH2 + mass_H_H2O + mass_H_H2SO4

noncomputable def total_mass_mixture : ℝ :=
  (moles_CaH2 * molar_mass_CaH2) + (moles_H2O * molar_mass_H2O) + (moles_H2SO4 * molar_mass_H2SO4)

noncomputable def mass_percentage_H : ℝ :=
  (total_mass_H / total_mass_mixture) * 100

theorem mass_percentage_H_calculation :
  abs (mass_percentage_H - 4.599) < 0.001 :=
by
  sorry

end mass_percentage_H_calculation_l714_71460


namespace minimum_value_of_x_plus_y_l714_71486

theorem minimum_value_of_x_plus_y
  (x y : ℝ)
  (h1 : x > 0) 
  (h2 : y > 0) 
  (h3 : (1 / y) + (4 / x) = 1) : 
  x + y = 9 :=
sorry

end minimum_value_of_x_plus_y_l714_71486


namespace distribute_pencils_l714_71480

theorem distribute_pencils (n : ℕ) (k : ℕ) (h1 : n = 8) (h2 : k = 4) :
  (∃ ways : ℕ, ways = Nat.choose (n - 1) (k - 1) ∧ ways = 35) :=
by
  sorry

end distribute_pencils_l714_71480


namespace tteokbokki_cost_l714_71476

theorem tteokbokki_cost (P : ℝ) (h1 : P / 2 - P * (3 / 16) = 2500) : P / 2 = 4000 :=
by
  sorry

end tteokbokki_cost_l714_71476


namespace min_value_y_l714_71482

theorem min_value_y (x : ℝ) (hx : x > 2) : 
  ∃ x, x > 2 ∧ (∀ y, y = (x^2 - 4*x + 8) / (x - 2) → y ≥ 4 ∧ y = 4 ↔ x = 4) :=
sorry

end min_value_y_l714_71482


namespace intersection_eq_l714_71439

def M := {x : ℝ | x < 1}
def N := {x : ℝ | Real.log x / Real.log 2 < 1}

theorem intersection_eq : {x : ℝ | x ∈ M ∧ x ∈ N} = {x : ℝ | 0 < x ∧ x < 1} :=
by
  sorry

end intersection_eq_l714_71439


namespace problem_M_l714_71449

theorem problem_M (M : ℤ) (h : 1989 + 1991 + 1993 + 1995 + 1997 + 1999 + 2001 = 14000 - M) : M = 35 :=
by
  sorry

end problem_M_l714_71449


namespace value_of_expression_l714_71417

variables (u v w : ℝ)

theorem value_of_expression (h1 : u = 3 * v) (h2 : w = 5 * u) : 2 * v + u + w = 20 * v :=
by sorry

end value_of_expression_l714_71417


namespace find_r_l714_71424

variable (n : ℕ) (q r : ℝ)

-- n must be a positive natural number
axiom n_pos : n > 0

-- q is a positive real number and not equal to 1
axiom q_pos : q > 0
axiom q_ne_one : q ≠ 1

-- Define the sequence sum S_n according to the problem statement
def S_n (n : ℕ) (q r : ℝ) : ℝ := q^n + r

-- The goal is to prove that the correct value of r is -1
theorem find_r : r = -1 :=
sorry

end find_r_l714_71424


namespace tom_spend_l714_71489

def theater_cost (seat_count : ℕ) (sqft_per_seat : ℕ) (cost_per_sqft : ℕ) (construction_multiplier : ℕ) (partner_percentage : ℝ) : ℝ :=
  let total_sqft := seat_count * sqft_per_seat
  let land_cost := total_sqft * cost_per_sqft
  let construction_cost := construction_multiplier * land_cost
  let total_cost := land_cost + construction_cost
  let partner_contribution := partner_percentage * (total_cost : ℝ)
  total_cost - partner_contribution

theorem tom_spend (partner_percentage : ℝ) :
  theater_cost 500 12 5 2 partner_percentage = 54000 :=
sorry

end tom_spend_l714_71489


namespace rectangular_field_area_l714_71440

theorem rectangular_field_area (a b c : ℕ) (h1 : a = 15) (h2 : c = 17)
  (h3 : a * a + b * b = c * c) : a * b = 120 := by
  sorry

end rectangular_field_area_l714_71440


namespace professional_tax_correct_l714_71455

-- Define the total income and professional deductions
def total_income : ℝ := 50000
def professional_deductions : ℝ := 35000

-- Define the tax rates
def tax_rate_ndfl : ℝ := 0.13
def tax_rate_simplified_income : ℝ := 0.06
def tax_rate_simplified_income_minus_exp : ℝ := 0.15
def tax_rate_professional_income : ℝ := 0.04

-- Define the expected tax amount
def expected_tax_professional_income : ℝ := 2000

-- Define a function to calculate the professional income tax for self-employed individuals
def calculate_professional_income_tax (income : ℝ) (rate : ℝ) : ℝ :=
  income * rate

-- Define the main theorem to assert the correctness of the tax calculation
theorem professional_tax_correct :
  calculate_professional_income_tax total_income tax_rate_professional_income = expected_tax_professional_income :=
by
  sorry

end professional_tax_correct_l714_71455


namespace laptop_sticker_price_l714_71423

theorem laptop_sticker_price (x : ℝ) (h1 : 0.8 * x - 120 = y) (h2 : 0.7 * x = z) (h3 : y + 25 = z) : x = 950 :=
sorry

end laptop_sticker_price_l714_71423


namespace ratio_men_to_women_on_team_l714_71442

theorem ratio_men_to_women_on_team (M W : ℕ) 
  (h1 : W = M + 6) 
  (h2 : M + W = 24) : 
  M / W = 3 / 5 := 
by 
  sorry

end ratio_men_to_women_on_team_l714_71442


namespace min_value_of_squares_attains_min_value_l714_71488

theorem min_value_of_squares (a b c t : ℝ) (h : a + b + c = t) :
  (a^2 + b^2 + c^2) ≥ (t^2 / 3) :=
sorry

theorem attains_min_value (a b c t : ℝ) (h : a = t / 3 ∧ b = t / 3 ∧ c = t / 3) :
  (a^2 + b^2 + c^2) = (t^2 / 3) :=
sorry

end min_value_of_squares_attains_min_value_l714_71488


namespace solve_y_l714_71498

theorem solve_y : ∃ y : ℚ, 2 * y + 3 * y = 600 - (4 * y + 5 * y + 100) ∧ y = 250 / 7 := by
  sorry

end solve_y_l714_71498


namespace yellow_more_than_green_l714_71453

-- Given conditions
def G : ℕ := 90               -- Number of green buttons
def B : ℕ := 85               -- Number of blue buttons
def T : ℕ := 275              -- Total number of buttons
def Y : ℕ := 100              -- Number of yellow buttons (derived from conditions)

-- Mathematically equivalent proof problem
theorem yellow_more_than_green : (90 + 100 + 85 = 275) → (100 - 90 = 10) :=
by sorry

end yellow_more_than_green_l714_71453


namespace law_school_student_count_l714_71418

theorem law_school_student_count 
    (business_students : ℕ)
    (sibling_pairs : ℕ)
    (selection_probability : ℚ)
    (L : ℕ)
    (h1 : business_students = 500)
    (h2 : sibling_pairs = 30)
    (h3 : selection_probability = 7.500000000000001e-5) :
    L = 8000 :=
by
  sorry

end law_school_student_count_l714_71418


namespace unoccupied_volume_in_container_l714_71402

-- defining constants
def side_length_container := 12
def side_length_ice_cube := 3
def number_of_ice_cubes := 8
def water_fill_fraction := 3 / 4

-- defining volumes
def volume_container := side_length_container ^ 3
def volume_water := volume_container * water_fill_fraction
def volume_ice_cube := side_length_ice_cube ^ 3
def total_volume_ice := volume_ice_cube * number_of_ice_cubes
def volume_unoccupied := volume_container - (volume_water + total_volume_ice)

-- The theorem to be proved
theorem unoccupied_volume_in_container : volume_unoccupied = 216 := by
  -- Proof steps will go here
  sorry

end unoccupied_volume_in_container_l714_71402


namespace percentage_time_in_park_l714_71400

/-- Define the number of trips Laura takes to the park. -/
def number_of_trips : ℕ := 6

/-- Define time spent at the park per trip in hours. -/
def time_at_park_per_trip : ℝ := 2

/-- Define time spent walking per trip in hours. -/
def time_walking_per_trip : ℝ := 0.5

/-- Define the total time for all trips. -/
def total_time_for_all_trips : ℝ := (time_at_park_per_trip + time_walking_per_trip) * number_of_trips

/-- Define the total time spent in the park for all trips. -/
def total_time_in_park : ℝ := time_at_park_per_trip * number_of_trips

/-- Prove that the percentage of the total time spent in the park is 80%. -/
theorem percentage_time_in_park : total_time_in_park / total_time_for_all_trips * 100 = 80 :=
by
  sorry

end percentage_time_in_park_l714_71400


namespace intersection_M_N_l714_71452

-- Definitions of the sets M and N
def M : Set ℤ := {-3, -2, -1}
def N : Set ℤ := { x | -2 < x ∧ x < 3 }

-- The theorem stating that the intersection of M and N is {-1}
theorem intersection_M_N : M ∩ N = {-1} := by
  sorry

end intersection_M_N_l714_71452


namespace greatest_integer_y_l714_71420

theorem greatest_integer_y (y : ℤ) : abs (3 * y - 4) ≤ 21 → y ≤ 8 :=
by
  sorry

end greatest_integer_y_l714_71420


namespace quadratic_has_two_distinct_real_roots_l714_71407

theorem quadratic_has_two_distinct_real_roots :
  ∀ x : ℝ, ∃ r1 r2 : ℝ, r1 ≠ r2 ∧ (x^2 - 2 * x - 6 = 0 ∧ x = r1 ∨ x = r2) :=
by sorry

end quadratic_has_two_distinct_real_roots_l714_71407


namespace question_solution_l714_71450

noncomputable def segment_ratio : (ℝ × ℝ) :=
  let m := 7
  let n := 2
  let x := - (2 / (m - n))
  let y := 7 / (m - n)
  (x, y)

theorem question_solution : segment_ratio = (-2/5, 7/5) :=
  by
  -- prove that the pair (x, y) calculated using given m and n equals (-2/5, 7/5)
  sorry

end question_solution_l714_71450


namespace speed_of_second_half_l714_71459

theorem speed_of_second_half (total_time : ℕ) (speed_first_half : ℕ) (total_distance : ℕ)
  (h1 : total_time = 15) (h2 : speed_first_half = 21) (h3 : total_distance = 336) :
  2 * total_distance / total_time - speed_first_half * (total_time / 2) / (total_time / 2) = 24 :=
by
  -- Proof omitted
  sorry

end speed_of_second_half_l714_71459


namespace evaluate_expression_l714_71403

theorem evaluate_expression : (4^150 * 9^152) / 6^301 = 27 / 2 := 
by 
  -- skipping the actual proof
  sorry

end evaluate_expression_l714_71403


namespace min_dist_sum_l714_71428

theorem min_dist_sum (x y : ℝ) :
  let M := (1, 3)
  let N := (7, 5)
  let P_on_M := (x - 1)^2 + (y - 3)^2 = 1
  let Q_on_N := (x - 7)^2 + (y - 5)^2 = 4
  let A_on_x_axis := y = 0
  ∃ (P Q : ℝ × ℝ), P_on_M ∧ Q_on_N ∧ ∀ A : ℝ × ℝ, A_on_x_axis → (|dist A P| + |dist A Q|) = 7 := 
sorry

end min_dist_sum_l714_71428


namespace tangent_line_equation_at_1_l714_71436

-- Define the function f and the point of tangency
def f (x : ℝ) : ℝ := x^2 + 2 * x
def p : ℝ × ℝ := (1, f 1)

-- Statement of the theorem
theorem tangent_line_equation_at_1 :
  ∃ a b c : ℝ, (∀ x y : ℝ, y = f x → y - p.2 = a * (x - p.1)) ∧
               4 * (p.1 : ℝ) - (p.2 : ℝ) - 1 = 0 :=
by
  -- Skipping the proof
  sorry

end tangent_line_equation_at_1_l714_71436


namespace train_A_start_time_l714_71484

theorem train_A_start_time :
  let distance := 155 -- km
  let speed_A := 20 -- km/h
  let speed_B := 25 -- km/h
  let start_B := 8 -- a.m.
  let meet_time := 11 -- a.m.
  let travel_time_B := meet_time - start_B -- time in hours for train B from 8 a.m. to 11 a.m.
  let distance_B := speed_B * travel_time_B -- distance covered by train B
  let distance_A := distance - distance_B -- remaining distance covered by train A
  let travel_time_A := distance_A / speed_A -- time for train A to cover its distance
  let start_A := meet_time - travel_time_A -- start time for train A
  start_A = 7 := by
  sorry

end train_A_start_time_l714_71484


namespace fraction_of_fractions_l714_71426

theorem fraction_of_fractions : (1 / 8) / (1 / 3) = 3 / 8 := by
  sorry

end fraction_of_fractions_l714_71426


namespace max_items_sum_l714_71448

theorem max_items_sum (m n : ℕ) (h : 5 * m + 17 * n = 203) : m + n ≤ 31 :=
sorry

end max_items_sum_l714_71448


namespace sum_of_coeff_l714_71457

theorem sum_of_coeff (x y : ℕ) (n : ℕ) (h : 2 * x + y = 3) : (2 * x + y) ^ n = 3^n := 
by
  sorry

end sum_of_coeff_l714_71457


namespace average_children_with_children_l714_71497

theorem average_children_with_children (total_families : ℕ) (average_children : ℚ) (childless_families : ℕ) :
  total_families = 12 →
  average_children = 3 →
  childless_families = 3 →
  (total_families * average_children) / (total_families - childless_families) = 4.0 :=
by
  intros h1 h2 h3
  sorry

end average_children_with_children_l714_71497


namespace Koschei_no_equal_coins_l714_71433

theorem Koschei_no_equal_coins (a : Fin 6 → ℕ)
  (initial_condition : a 0 = 1 ∧ a 1 = 0 ∧ a 2 = 0 ∧ a 3 = 0 ∧ a 4 = 0 ∧ a 5 = 0) :
  ¬ ( ∃ k : ℕ, ( ( ∀ i : Fin 6, a i = k ) ) ) :=
by
  sorry

end Koschei_no_equal_coins_l714_71433


namespace product_of_two_numbers_l714_71404

theorem product_of_two_numbers (a b : ℝ) (h1 : a + b = 70) (h2 : a - b = 10) : a * b = 1200 := 
by
  sorry

end product_of_two_numbers_l714_71404


namespace problem1_problem2_l714_71496

-- Problem 1 Proof Statement
theorem problem1 : Real.sin (30 * Real.pi / 180) + abs (-1) - (Real.sqrt 3 - Real.pi) ^ 0 = 1 / 2 := 
  by sorry

-- Problem 2 Proof Statement
theorem problem2 (x: ℝ) (hx : x ≠ 2) : (2 * x - 3) / (x - 2) - (x - 1) / (x - 2) = 1 := 
  by sorry

end problem1_problem2_l714_71496


namespace largest_divisor_60_36_divisible_by_3_l714_71425

theorem largest_divisor_60_36_divisible_by_3 : 
  ∃ x, (x ∣ 60) ∧ (x ∣ 36) ∧ (3 ∣ x) ∧ (∀ y, (y ∣ 60) → (y ∣ 36) → (3 ∣ y) → y ≤ x) ∧ x = 12 :=
sorry

end largest_divisor_60_36_divisible_by_3_l714_71425


namespace parabola_directrix_l714_71406

theorem parabola_directrix (x : ℝ) :
  (∃ y : ℝ, y = (x^2 - 8*x + 12) / 16) →
  ∃ directrix : ℝ, directrix = -17 / 4 :=
by
  sorry

end parabola_directrix_l714_71406


namespace number_of_square_tiles_l714_71481

/-- A box contains a collection of triangular tiles, square tiles, and pentagonal tiles. 
    There are a total of 30 tiles in the box and a total of 100 edges. 
    We need to show that the number of square tiles is 10. --/
theorem number_of_square_tiles (a b c : ℕ) (h1 : a + b + c = 30) (h2 : 3 * a + 4 * b + 5 * c = 100) : b = 10 := by
  sorry

end number_of_square_tiles_l714_71481


namespace matrix_C_power_50_l714_71462

open Matrix

theorem matrix_C_power_50 (C : Matrix (Fin 2) (Fin 2) ℤ) 
  (hC : C = !![3, 2; -8, -5]) : 
  C^50 = !![1, 0; 0, 1] :=
by {
  -- External proof omitted.
  sorry
}

end matrix_C_power_50_l714_71462


namespace specific_values_exist_l714_71467

def expr_equal_for_specific_values (a b c : ℝ) : Prop :=
  a + b^2 * c = (a^2 + b) * (a + c)

theorem specific_values_exist :
  ∃ a b c : ℝ, expr_equal_for_specific_values a b c :=
sorry

end specific_values_exist_l714_71467


namespace job_completion_time_l714_71408

def time (hours : ℕ) (minutes : ℕ) : ℕ := hours * 60 + minutes

noncomputable def start_time : ℕ := time 9 45
noncomputable def half_completion_time : ℕ := time 13 0  -- 1:00 PM in 24-hour time format

theorem job_completion_time :
  ∃ finish_time, finish_time = time 16 15 ∧
  (half_completion_time - start_time) * 2 = finish_time - start_time :=
by
  sorry

end job_completion_time_l714_71408


namespace lucas_notation_sum_l714_71470

-- Define what each representation in Lucas's notation means
def lucasValue : String → Int
| "0" => 0
| s => -((s.length) - 1)

-- Define the question as a Lean theorem
theorem lucas_notation_sum :
  lucasValue "000" + lucasValue "0000" = lucasValue "000000" :=
by
  sorry

end lucas_notation_sum_l714_71470


namespace find_rate_percent_l714_71469

theorem find_rate_percent
  (P : ℝ) (SI : ℝ) (T : ℝ) (R : ℝ) 
  (hP : P = 1600)
  (hSI : SI = 200)
  (hT : T = 4)
  (hSI_eq : SI = (P * R * T) / 100) :
  R = 3.125 :=
by {
  sorry
}

end find_rate_percent_l714_71469


namespace tournament_start_count_l714_71490

theorem tournament_start_count (x : ℝ) (h1 : (0.1 * x = 30)) : x = 300 :=
by
  sorry

end tournament_start_count_l714_71490


namespace fruits_total_l714_71461

def remaining_fruits (frank_apples susan_blueberries henry_apples karen_grapes : ℤ) : ℤ :=
  let frank_remaining := 36 - (36 / 3)
  let susan_remaining := 120 - (120 / 2)
  let henry_collected := 2 * 120
  let henry_after_eating := henry_collected - (henry_collected / 4)
  let henry_remaining := henry_after_eating - (henry_after_eating / 10)
  let karen_collected := henry_collected / 2
  let karen_after_spoilage := karen_collected - (15 * karen_collected / 100)
  let karen_after_giving_away := karen_after_spoilage - (karen_after_spoilage / 3)
  let karen_remaining := karen_after_giving_away - (Int.sqrt karen_after_giving_away)
  frank_remaining + susan_remaining + henry_remaining + karen_remaining

theorem fruits_total : remaining_fruits 36 120 240 120 = 254 :=
by sorry

end fruits_total_l714_71461


namespace range_of_m_l714_71446

theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ x ≠ 1 ∧ (m + 3) / (x - 1) = 1) ↔ m > -4 ∧ m ≠ -3 := 
by 
  sorry

end range_of_m_l714_71446


namespace neg_exists_equiv_forall_l714_71430

theorem neg_exists_equiv_forall :
  (¬ ∃ x : ℝ, x^2 - x + 4 < 0) ↔ (∀ x : ℝ, x^2 - x + 4 ≥ 0) :=
by
  sorry

end neg_exists_equiv_forall_l714_71430


namespace jessy_initial_reading_plan_l714_71416

theorem jessy_initial_reading_plan (x : ℕ) (h : (7 * (3 * x + 2) = 140)) : x = 6 :=
sorry

end jessy_initial_reading_plan_l714_71416


namespace largest_common_value_l714_71435

theorem largest_common_value :
  ∃ (a : ℕ), (∃ (n m : ℕ), a = 4 + 5 * n ∧ a = 5 + 10 * m) ∧ a < 1000 ∧ a = 994 :=
by {
  sorry
}

end largest_common_value_l714_71435

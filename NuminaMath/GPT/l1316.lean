import Mathlib

namespace NUMINAMATH_GPT_length_of_picture_frame_l1316_131602

theorem length_of_picture_frame (P W : ℕ) (hP : P = 30) (hW : W = 10) : ∃ L : ℕ, 2 * (L + W) = P ∧ L = 5 :=
by
  sorry

end NUMINAMATH_GPT_length_of_picture_frame_l1316_131602


namespace NUMINAMATH_GPT_rex_lesson_schedule_l1316_131652

-- Define the total lessons and weeks
def total_lessons : ℕ := 40
def weeks_completed : ℕ := 6
def weeks_remaining : ℕ := 4

-- Define the proof statement
theorem rex_lesson_schedule : (weeks_completed + weeks_remaining) * 4 = total_lessons := by
  -- Proof placeholder, to be filled in 
  sorry

end NUMINAMATH_GPT_rex_lesson_schedule_l1316_131652


namespace NUMINAMATH_GPT_minimum_m_n_sum_l1316_131694

theorem minimum_m_n_sum:
  ∃ (m n : ℕ), 0 < m ∧ 0 < n ∧ 90 * m = n ^ 3 ∧ m + n = 330 :=
sorry

end NUMINAMATH_GPT_minimum_m_n_sum_l1316_131694


namespace NUMINAMATH_GPT_problem1_problem2_l1316_131638

-- Proof problem 1
theorem problem1 : (-3)^2 / 3 + abs (-7) + 3 * (-1/3) = 3 :=
by
  sorry

-- Proof problem 2
theorem problem2 : (-1) ^ 2022 - ( (-1/4) - (-1/3) ) / (-1/12) = 2 :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1316_131638


namespace NUMINAMATH_GPT_production_average_l1316_131645

theorem production_average (n : ℕ) (P : ℕ) (h1 : P / n = 50) (h2 : (P + 90) / (n + 1) = 54) : n = 9 :=
sorry

end NUMINAMATH_GPT_production_average_l1316_131645


namespace NUMINAMATH_GPT_containers_needed_l1316_131623

-- Define the conditions: 
def weight_in_pounds : ℚ := 25 / 2
def ounces_per_pound : ℚ := 16
def ounces_per_container : ℚ := 50

-- Define the total weight in ounces
def total_weight_in_ounces := weight_in_pounds * ounces_per_pound

-- Theorem statement: Number of containers.
theorem containers_needed : total_weight_in_ounces / ounces_per_container = 4 := 
by
  -- Write the proof here
  sorry

end NUMINAMATH_GPT_containers_needed_l1316_131623


namespace NUMINAMATH_GPT_cos_difference_simplification_l1316_131609

theorem cos_difference_simplification :
  let x := Real.cos (20 * Real.pi / 180)
  let y := Real.cos (40 * Real.pi / 180)
  x - y = -1 / (2 * Real.sqrt 5) :=
sorry

end NUMINAMATH_GPT_cos_difference_simplification_l1316_131609


namespace NUMINAMATH_GPT_max_x_plus_2y_l1316_131620

theorem max_x_plus_2y (x y : ℝ) (h : 3 * (x^2 + y^2) = x + y) : x + 2 * y ≤ 3 :=
sorry

end NUMINAMATH_GPT_max_x_plus_2y_l1316_131620


namespace NUMINAMATH_GPT_print_time_l1316_131650

-- Define the conditions
def pages : ℕ := 345
def rate : ℕ := 23
def expected_minutes : ℕ := 15

-- State the problem as a theorem
theorem print_time (pages rate : ℕ) : (pages / rate = 15) :=
by
  sorry

end NUMINAMATH_GPT_print_time_l1316_131650


namespace NUMINAMATH_GPT_test_score_after_preparation_l1316_131657

-- Define the conditions in Lean 4
def score (k t : ℝ) : ℝ := k * t^2

theorem test_score_after_preparation (k t : ℝ)
    (h1 : score k 2 = 90) (h2 : k = 22.5) :
    score k 3 = 202.5 :=
by
  sorry

end NUMINAMATH_GPT_test_score_after_preparation_l1316_131657


namespace NUMINAMATH_GPT_rectangular_field_perimeter_l1316_131627

-- Definitions for conditions
def width : ℕ := 75
def length : ℕ := (7 * width) / 5
def perimeter (L W : ℕ) : ℕ := 2 * (L + W)

-- Statement to prove
theorem rectangular_field_perimeter : perimeter length width = 360 := by
  sorry

end NUMINAMATH_GPT_rectangular_field_perimeter_l1316_131627


namespace NUMINAMATH_GPT_bus_children_problem_l1316_131684

theorem bus_children_problem :
  ∃ X, 5 - 63 + X = 14 ∧ X - 63 = 9 :=
by 
  sorry

end NUMINAMATH_GPT_bus_children_problem_l1316_131684


namespace NUMINAMATH_GPT_range_of_a_l1316_131632

open Real

noncomputable def f (a e x : ℝ) : ℝ := 2 * a^x - e * x^2

theorem range_of_a (a e x₁ x₂ : ℝ) (h_a_pos : 0 < a) (h_a_ne_one : a ≠ 1) (h_x₁ : f a e x₁ = f a e x + f a e 1) (h_min : deriv (f a e) x₁ = 0) (h_max : deriv (f a e) x₂ = 0) (h_x₁_lt_x₂ : x₁ < x₂) :
  1 / exp 1 < a ∧ a < 1 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1316_131632


namespace NUMINAMATH_GPT_problem_statement_l1316_131660

theorem problem_statement 
  (a b c : ℤ)
  (h1 : (5 * a + 2) ^ (1/3) = 3)
  (h2 : (3 * a + b - 1) ^ (1/2) = 4)
  (h3 : c = Int.floor (Real.sqrt 13))
  : a = 5 ∧ b = 2 ∧ c = 3 ∧ Real.sqrt (3 * a - b + c) = 4 := 
by 
  sorry

end NUMINAMATH_GPT_problem_statement_l1316_131660


namespace NUMINAMATH_GPT_max_xyz_eq_one_l1316_131605

noncomputable def max_xyz (x y z : ℝ) : ℝ :=
  if h_cond : 0 < x ∧ 0 < y ∧ 0 < z ∧ (x * y + z ^ 2 = (x + z) * (y + z)) ∧ (x + y + z = 3) then
    x * y * z
  else
    0

theorem max_xyz_eq_one : ∀ (x y z : ℝ), 0 < x → 0 < y → 0 < z → 
  (x * y + z ^ 2 = (x + z) * (y + z)) → (x + y + z = 3) → max_xyz x y z ≤ 1 :=
by
  intros x y z hx hy hz h1 h2
  -- Proof is omitted here
  sorry

end NUMINAMATH_GPT_max_xyz_eq_one_l1316_131605


namespace NUMINAMATH_GPT_range_of_a_l1316_131693

theorem range_of_a (a : ℝ) (x : ℝ) : (∃ x, x^2 - a*x - a ≤ -3) → (a ≤ -6 ∨ a ≥ 2) :=
sorry

end NUMINAMATH_GPT_range_of_a_l1316_131693


namespace NUMINAMATH_GPT_inequality_xy_l1316_131668

-- Defining the constants and conditions
variables {x y : ℝ}

-- Main theorem to prove the inequality and find pairs for equality
theorem inequality_xy (h : (x + 1) * (y + 2) = 8) :
  (xy - 10)^2 ≥ 64 ∧ ((xy - 10)^2 = 64 → (x, y) = (1, 2) ∨ (x, y) = (-3, -6)) :=
sorry

end NUMINAMATH_GPT_inequality_xy_l1316_131668


namespace NUMINAMATH_GPT_paul_oil_change_rate_l1316_131646

theorem paul_oil_change_rate (P : ℕ) (h₁ : 8 * (P + 3) = 40) : P = 2 :=
by
  sorry

end NUMINAMATH_GPT_paul_oil_change_rate_l1316_131646


namespace NUMINAMATH_GPT_quadratic_coefficients_l1316_131647

theorem quadratic_coefficients (b c : ℝ) :
  (∀ x : ℝ, |x + 4| = 3 ↔ x^2 + bx + c = 0) → (b = 8 ∧ c = 7) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_coefficients_l1316_131647


namespace NUMINAMATH_GPT_length_of_BC_is_eight_l1316_131648

theorem length_of_BC_is_eight (a : ℝ) (h_area : (1 / 2) * (2 * a) * a^2 = 64) : 2 * a = 8 := 
by { sorry }

end NUMINAMATH_GPT_length_of_BC_is_eight_l1316_131648


namespace NUMINAMATH_GPT_max_tiles_on_floor_l1316_131683

   -- Definitions corresponding to conditions
   def tile_length_1 : ℕ := 35
   def tile_width_1 : ℕ := 30
   def tile_length_2 : ℕ := 30
   def tile_width_2 : ℕ := 35
   def floor_length : ℕ := 1000
   def floor_width : ℕ := 210

   -- Conditions:
   -- 1. Tiles do not overlap.
   -- 2. Tiles are placed with edges jutting against each other on all edges.
   -- 3. A tile can be placed in any orientation so long as its edges are parallel to the edges of the floor.
   -- 4. No tile should overshoot any edge of the floor.

   theorem max_tiles_on_floor :
     let tiles_orientation_1 := (floor_length / tile_length_1) * (floor_width / tile_width_1)
     let tiles_orientation_2 := (floor_length / tile_length_2) * (floor_width / tile_width_2)
     max tiles_orientation_1 tiles_orientation_2 = 198 :=
   by {
     -- The actual proof handling is skipped, as per instructions.
     sorry
   }
   
end NUMINAMATH_GPT_max_tiles_on_floor_l1316_131683


namespace NUMINAMATH_GPT_nuts_distributive_problem_l1316_131622

theorem nuts_distributive_problem (x y : ℕ) (h1 : 70 ≤ x + y) (h2 : x + y ≤ 80) (h3 : (3 / 4 : ℚ) * x + (1 / 5 : ℚ) * (y + (1 / 4 : ℚ) * x) = (x : ℚ) + 1) :
  x = 36 ∧ y = 41 :=
by
  sorry

end NUMINAMATH_GPT_nuts_distributive_problem_l1316_131622


namespace NUMINAMATH_GPT_nine_b_equals_eighteen_l1316_131629

theorem nine_b_equals_eighteen (a b : ℝ) (h1 : 6 * a + 3 * b = 0) (h2 : a = b - 3) : 9 * b = 18 :=
  sorry

end NUMINAMATH_GPT_nine_b_equals_eighteen_l1316_131629


namespace NUMINAMATH_GPT_difference_eq_neg_subtrahend_implies_minuend_zero_l1316_131608

theorem difference_eq_neg_subtrahend_implies_minuend_zero {x y : ℝ} (h : x - y = -y) : x = 0 :=
sorry

end NUMINAMATH_GPT_difference_eq_neg_subtrahend_implies_minuend_zero_l1316_131608


namespace NUMINAMATH_GPT_valentines_initial_l1316_131658

theorem valentines_initial (gave_away : ℕ) (left_over : ℕ) (initial : ℕ) : 
  gave_away = 8 → left_over = 22 → initial = gave_away + left_over → initial = 30 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_valentines_initial_l1316_131658


namespace NUMINAMATH_GPT_gcd_gt_one_l1316_131661

-- Defining the given conditions and the statement to prove
theorem gcd_gt_one (a b x y : ℕ) (h : (a^2 + b^2) ∣ (a * x + b * y)) : 
  Nat.gcd (x^2 + y^2) (a^2 + b^2) > 1 := 
sorry

end NUMINAMATH_GPT_gcd_gt_one_l1316_131661


namespace NUMINAMATH_GPT_birch_trees_count_l1316_131614

-- Definitions based on the conditions
def total_trees : ℕ := 4000
def percentage_spruce : ℕ := 10
def percentage_pine : ℕ := 13

def count_spruce : ℕ := (percentage_spruce * total_trees) / 100
def count_pine : ℕ := (percentage_pine * total_trees) / 100
def count_oak : ℕ := count_spruce + count_pine

def count_birch : ℕ := total_trees - (count_spruce + count_pine + count_oak)

-- The theorem to be proven
theorem birch_trees_count :
  count_birch = 2160 := by
  sorry

end NUMINAMATH_GPT_birch_trees_count_l1316_131614


namespace NUMINAMATH_GPT_right_triangle_area_l1316_131610

theorem right_triangle_area {a r R : ℝ} (hR : R = (5 / 2) * r) (h_leg : ∃ BC, BC = a) :
  (∃ area, area = (2 * a^2 / 3) ∨ area = (3 * a^2 / 8)) :=
sorry

end NUMINAMATH_GPT_right_triangle_area_l1316_131610


namespace NUMINAMATH_GPT_Andy_and_Carlos_tie_for_first_l1316_131612

def AndyLawnArea (A : ℕ) := 3 * A
def CarlosLawnArea (A : ℕ) := A / 4
def BethMowingRate := 90
def CarlosMowingRate := BethMowingRate / 3
def AndyMowingRate := BethMowingRate * 4

theorem Andy_and_Carlos_tie_for_first (A : ℕ) (hA_nonzero : 0 < A) :
  (AndyLawnArea A / AndyMowingRate) = (CarlosLawnArea A / CarlosMowingRate) ∧
  (AndyLawnArea A / AndyMowingRate) < (A / BethMowingRate) :=
by
  unfold AndyLawnArea CarlosLawnArea BethMowingRate CarlosMowingRate AndyMowingRate
  sorry

end NUMINAMATH_GPT_Andy_and_Carlos_tie_for_first_l1316_131612


namespace NUMINAMATH_GPT_arithmetic_geometric_inequality_l1316_131626

theorem arithmetic_geometric_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  (a + b) / 2 ≥ Real.sqrt (a * b) := 
sorry

end NUMINAMATH_GPT_arithmetic_geometric_inequality_l1316_131626


namespace NUMINAMATH_GPT_triangle_inequality_l1316_131642

theorem triangle_inequality (a b c p q r : ℝ) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_sum_zero : p + q + r = 0) : 
  a^2 * p * q + b^2 * q * r + c^2 * r * p ≤ 0 := 
sorry

end NUMINAMATH_GPT_triangle_inequality_l1316_131642


namespace NUMINAMATH_GPT_avg_diff_condition_l1316_131617

variable (a b c : ℝ)

theorem avg_diff_condition (h1 : (a + b) / 2 = 110) (h2 : (b + c) / 2 = 150) : a - c = -80 :=
by
  sorry

end NUMINAMATH_GPT_avg_diff_condition_l1316_131617


namespace NUMINAMATH_GPT_smallest_four_digit_divisible_by_8_with_3_even_1_odd_l1316_131666

theorem smallest_four_digit_divisible_by_8_with_3_even_1_odd : ∃ (n : ℕ), 1000 ≤ n ∧ n < 10000 ∧ n % 8 = 0 ∧ 
  (∃ d1 d2 d3 d4, n = d1 * 1000 + d2 * 100 + d3 * 10 + d4 ∧ 
    (d1 % 2 = 0) ∧ (d2 % 2 = 0 ∨ d2 % 2 ≠ 0) ∧ 
    (d3 % 2 = 0) ∧ (d4 % 2 = 0 ∨ d4 % 2 ≠ 0) ∧ 
    (d2 % 2 ≠ 0 ∨ d4 % 2 ≠ 0) ) ∧ n = 1248 :=
by
  sorry

end NUMINAMATH_GPT_smallest_four_digit_divisible_by_8_with_3_even_1_odd_l1316_131666


namespace NUMINAMATH_GPT_cage_cost_correct_l1316_131697

noncomputable def total_amount_paid : ℝ := 20
noncomputable def change_received : ℝ := 0.26
noncomputable def cat_toy_cost : ℝ := 8.77
noncomputable def cage_cost := total_amount_paid - change_received

theorem cage_cost_correct : cage_cost = 19.74 := by
  sorry

end NUMINAMATH_GPT_cage_cost_correct_l1316_131697


namespace NUMINAMATH_GPT_no_third_number_for_lcm_l1316_131634

theorem no_third_number_for_lcm (a : ℕ) : ¬ (Nat.lcm (Nat.lcm 23 46) a = 83) :=
sorry

end NUMINAMATH_GPT_no_third_number_for_lcm_l1316_131634


namespace NUMINAMATH_GPT_trapezium_other_side_length_l1316_131664

theorem trapezium_other_side_length (a h Area : ℕ) (a_eq : a = 4) (h_eq : h = 6) (Area_eq : Area = 27) : 
  ∃ (b : ℕ), b = 5 := 
by
  sorry

end NUMINAMATH_GPT_trapezium_other_side_length_l1316_131664


namespace NUMINAMATH_GPT_sum_difference_4041_l1316_131663

def sum_of_first_n_integers (n : ℕ) : ℕ := n * (n + 1) / 2

theorem sum_difference_4041 :
  sum_of_first_n_integers 2021 - sum_of_first_n_integers 2019 = 4041 :=
by
  sorry

end NUMINAMATH_GPT_sum_difference_4041_l1316_131663


namespace NUMINAMATH_GPT_how_many_more_cups_of_sugar_l1316_131628

def required_sugar : ℕ := 11
def required_flour : ℕ := 9
def added_flour : ℕ := 12
def added_sugar : ℕ := 10

theorem how_many_more_cups_of_sugar :
  required_sugar - added_sugar = 1 :=
by
  sorry

end NUMINAMATH_GPT_how_many_more_cups_of_sugar_l1316_131628


namespace NUMINAMATH_GPT_calculate_smaller_sphere_radius_l1316_131699

noncomputable def smaller_sphere_radius (r1 r2 r3 r4 : ℝ) : ℝ := 
  if h : r1 = 2 ∧ r2 = 2 ∧ r3 = 3 ∧ r4 = 3 then 
    6 / 11 
  else 
    0

theorem calculate_smaller_sphere_radius :
  smaller_sphere_radius 2 2 3 3 = 6 / 11 :=
by
  sorry

end NUMINAMATH_GPT_calculate_smaller_sphere_radius_l1316_131699


namespace NUMINAMATH_GPT_sixty_percent_of_fifty_minus_thirty_percent_of_thirty_l1316_131671

theorem sixty_percent_of_fifty_minus_thirty_percent_of_thirty : 
  (60 / 100 : ℝ) * 50 - (30 / 100 : ℝ) * 30 = 21 :=
by
  sorry

end NUMINAMATH_GPT_sixty_percent_of_fifty_minus_thirty_percent_of_thirty_l1316_131671


namespace NUMINAMATH_GPT_barrels_oil_total_l1316_131673

theorem barrels_oil_total :
  let A := 3 / 4
  let B := A + 1 / 10
  A + B = 8 / 5 := by
  sorry

end NUMINAMATH_GPT_barrels_oil_total_l1316_131673


namespace NUMINAMATH_GPT_f_is_odd_f_is_monotone_l1316_131669

noncomputable def f (k x : ℝ) : ℝ := x + k / x

-- Proving f(x) is odd
theorem f_is_odd (k : ℝ) (hk : k ≠ 0) : ∀ x : ℝ, f k (-x) = -f k x :=
by
  intro x
  sorry

-- Proving f(x) is monotonically increasing on [sqrt(k), +∞) for k > 0
theorem f_is_monotone (k : ℝ) (hk : k > 0) : ∀ x1 x2 : ℝ, 
  x1 ∈ Set.Ici (Real.sqrt k) → x2 ∈ Set.Ici (Real.sqrt k) → x1 < x2 → f k x1 < f k x2 :=
by
  intro x1 x2 hx1 hx2 hlt
  sorry

end NUMINAMATH_GPT_f_is_odd_f_is_monotone_l1316_131669


namespace NUMINAMATH_GPT_solve_for_a_when_diamond_eq_6_l1316_131698

def diamond (a b : ℝ) : ℝ := 3 * a - 2 * b^2

theorem solve_for_a_when_diamond_eq_6 (a : ℝ) : diamond a 3 = 6 → a = 8 :=
by
  intros h
  simp [diamond] at h
  sorry

end NUMINAMATH_GPT_solve_for_a_when_diamond_eq_6_l1316_131698


namespace NUMINAMATH_GPT_find_xy_such_that_product_is_fifth_power_of_prime_l1316_131695

theorem find_xy_such_that_product_is_fifth_power_of_prime
  (x y : ℕ) (p : ℕ) (hp : Nat.Prime p)
  (hx_pos : 0 < x) (hy_pos : 0 < y)
  (h_eq : (x^2 + y) * (y^2 + x) = p^5) :
  (x = 2 ∧ y = 5) ∨ (x = 5 ∧ y = 2) :=
sorry

end NUMINAMATH_GPT_find_xy_such_that_product_is_fifth_power_of_prime_l1316_131695


namespace NUMINAMATH_GPT_no_four_distinct_numbers_l1316_131688

theorem no_four_distinct_numbers (x y : ℝ) (h : x ≠ y ∧ 
    (x^(10:ℕ) + (x^(9:ℕ)) * y + (x^(8:ℕ)) * (y^(2:ℕ)) + 
    (x^(7:ℕ)) * (y^(3:ℕ)) + (x^(6:ℕ)) * (y^(4:ℕ)) + 
    (x^(5:ℕ)) * (y^(5:ℕ)) + (x^(4:ℕ)) * (y^(6:ℕ)) + 
    (x^(3:ℕ)) * (y^(7:ℕ)) + (x^(2:ℕ)) * (y^(8:ℕ)) + 
    (x^(1:ℕ)) * (y^(9:ℕ)) + (y^(10:ℕ)) = 1)) : False :=
by
  sorry

end NUMINAMATH_GPT_no_four_distinct_numbers_l1316_131688


namespace NUMINAMATH_GPT_processing_times_maximum_salary_l1316_131672

def monthly_hours : ℕ := 8 * 25
def base_salary : ℕ := 800
def earnings_per_A : ℕ := 16
def earnings_per_B : ℕ := 12

theorem processing_times :
  ∃ (x y : ℕ),
    x + 3 * y = 5 ∧ 2 * x + 5 * y = 9 ∧ x = 2 ∧ y = 1 :=
by
  sorry

theorem maximum_salary :
  ∃ (a b W : ℕ),
    a ≥ 50 ∧ 
    b = monthly_hours - 2 * a ∧ 
    W = base_salary + earnings_per_A * a + earnings_per_B * b ∧ 
    a = 50 ∧ 
    b = 100 ∧ 
    W = 2800 :=
by
  sorry

end NUMINAMATH_GPT_processing_times_maximum_salary_l1316_131672


namespace NUMINAMATH_GPT_students_solved_only_B_l1316_131616

variable (A B C : Prop)
variable (n x y b c d : ℕ)

-- Conditions given in the problem
axiom h1 : n = 25
axiom h2 : x + y + b + c + d = n
axiom h3 : b + d = 2 * (c + d)
axiom h4 : x = y + 1
axiom h5 : x + b + c = 2 * (b + c)

-- Theorem to be proved
theorem students_solved_only_B : b = 6 :=
by
  sorry

end NUMINAMATH_GPT_students_solved_only_B_l1316_131616


namespace NUMINAMATH_GPT_cheryl_material_left_l1316_131659

-- Conditions
def initial_material_type1 (m1 : ℚ) : Prop := m1 = 2/9
def initial_material_type2 (m2 : ℚ) : Prop := m2 = 1/8
def used_material (u : ℚ) : Prop := u = 0.125

-- Define the total material bought
def total_material (m1 m2 : ℚ) : ℚ := m1 + m2

-- Define the material left
def material_left (t u : ℚ) : ℚ := t - u

-- The target theorem
theorem cheryl_material_left (m1 m2 u : ℚ) 
  (h1 : initial_material_type1 m1)
  (h2 : initial_material_type2 m2)
  (h3 : used_material u) : 
  material_left (total_material m1 m2) u = 2/9 :=
by
  sorry

end NUMINAMATH_GPT_cheryl_material_left_l1316_131659


namespace NUMINAMATH_GPT_infinitely_many_triples_no_triples_l1316_131631

theorem infinitely_many_triples :
  ∃ (m n p : ℕ), ∃ (k : ℕ), m > 0 ∧ n > 0 ∧ p > 0 ∧ 4 * m * n - m - n = p ^ 2 - 1 := 
sorry

theorem no_triples :
  ¬∃ (m n p : ℕ), m > 0 ∧ n > 0 ∧ p > 0 ∧ 4 * m * n - m - n = p ^ 2 := 
sorry

end NUMINAMATH_GPT_infinitely_many_triples_no_triples_l1316_131631


namespace NUMINAMATH_GPT_f_evaluation_l1316_131655

def f (a b c : ℚ) : ℚ := a^2 + 2 * b * c

theorem f_evaluation :
  f 1 23 76 + f 23 76 1 + f 76 1 23 = 10000 := by
  sorry

end NUMINAMATH_GPT_f_evaluation_l1316_131655


namespace NUMINAMATH_GPT_annual_pension_l1316_131675

theorem annual_pension (c d r s x k : ℝ) (hc : c ≠ 0) (hd : d ≠ c)
  (h1 : k * (x + c) ^ (3 / 2) = k * x ^ (3 / 2) + r)
  (h2 : k * (x + d) ^ (3 / 2) = k * x ^ (3 / 2) + s) :
  k * x ^ (3 / 2) = 4 * r^2 / (9 * c^2) :=
by
  sorry

end NUMINAMATH_GPT_annual_pension_l1316_131675


namespace NUMINAMATH_GPT_isosceles_triangle_formed_by_lines_l1316_131651

theorem isosceles_triangle_formed_by_lines :
  let P1 := (1/4, 4)
  let P2 := (-3/2, -3)
  let P3 := (2, -3)
  let d12 := ((1/4 + 3/2)^2 + (4 + 3)^2)
  let d13 := ((1/4 - 2)^2 + (4 + 3)^2)
  let d23 := ((-3/2 - 2)^2)
  (d12 = d13) ∧ (d12 ≠ d23) → 
  ∃ (A B C : ℝ × ℝ), 
    A = P1 ∧ B = P2 ∧ C = P3 ∧ 
    ((dist A B = dist A C) ∧ (dist B C ≠ dist A B)) :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_formed_by_lines_l1316_131651


namespace NUMINAMATH_GPT_total_songs_purchased_is_162_l1316_131678

variable (c_country : ℕ) (c_pop : ℕ) (c_jazz : ℕ) (c_rock : ℕ)
variable (s_country : ℕ) (s_pop : ℕ) (s_jazz : ℕ) (s_rock : ℕ)

-- Setting up the conditions
def num_country_albums := 6
def num_pop_albums := 2
def num_jazz_albums := 4
def num_rock_albums := 3

-- Number of songs per album
def country_album_songs := 9
def pop_album_songs := 9
def jazz_album_songs := 12
def rock_album_songs := 14

theorem total_songs_purchased_is_162 :
  num_country_albums * country_album_songs +
  num_pop_albums * pop_album_songs +
  num_jazz_albums * jazz_album_songs +
  num_rock_albums * rock_album_songs = 162 := by
  sorry

end NUMINAMATH_GPT_total_songs_purchased_is_162_l1316_131678


namespace NUMINAMATH_GPT_arithmetic_seq_condition_l1316_131633

def sum_first_n_terms (a d : ℕ) (n : ℕ) : ℕ := 
  n * a + (n * (n - 1) / 2) * d

theorem arithmetic_seq_condition (a2 : ℕ) (S3 S9 : ℕ) :
  a2 = 1 → 
  (∃ d, (d > 4 ∧ S3 = 3 * a2 + (3 * (3 - 1) / 2) * d ∧ S9 = 9 * a2 + (9 * (9 - 1) / 2) * d) → (S3 + S9) > 93) ↔ 
  (∃ d, (S3 + S9 = sum_first_n_terms a2 d 3 + sum_first_n_terms a2 d 9 ∧ (sum_first_n_terms a2 d 3 + sum_first_n_terms a2 d 9) > 93 → d > 3 ∧ a2 + d > 5)) :=
by 
  sorry

end NUMINAMATH_GPT_arithmetic_seq_condition_l1316_131633


namespace NUMINAMATH_GPT_glens_speed_is_37_l1316_131607

/-!
# Problem Statement
Glen and Hannah drive at constant speeds toward each other on a highway. Glen drives at a certain speed G km/h. At some point, they pass by each other, and keep driving away from each other, maintaining their constant speeds. 
Glen is 130 km away from Hannah at 6 am and again at 11 am. Hannah is driving at 15 kilometers per hour.
Prove that Glen's speed is 37 km/h.
-/

def glens_speed (G : ℝ) : Prop :=
  ∃ G: ℝ, 
    (∃ H_speed : ℝ, H_speed = 15) ∧ -- Hannah's speed
    (∃ distance : ℝ, distance = 130) ∧ -- distance at 6 am and 11 am
    G + 15 = 260 / 5 -- derived equation from conditions

theorem glens_speed_is_37 : glens_speed 37 :=
by {
  sorry -- proof to be filled in
}

end NUMINAMATH_GPT_glens_speed_is_37_l1316_131607


namespace NUMINAMATH_GPT_nancy_kept_chips_l1316_131654

def nancy_initial_chips : ℕ := 22
def chips_given_to_brother : ℕ := 7
def chips_given_to_sister : ℕ := 5

theorem nancy_kept_chips : nancy_initial_chips - (chips_given_to_brother + chips_given_to_sister) = 10 :=
by
  sorry

end NUMINAMATH_GPT_nancy_kept_chips_l1316_131654


namespace NUMINAMATH_GPT_total_points_each_team_l1316_131670

def score_touchdown := 7
def score_field_goal := 3
def score_safety := 2

def team_hawks_first_match_score := 3 * score_touchdown + 2 * score_field_goal + score_safety
def team_eagles_first_match_score := 5 * score_touchdown + 4 * score_field_goal
def team_hawks_second_match_score := 4 * score_touchdown + 3 * score_field_goal
def team_falcons_second_match_score := 6 * score_touchdown + 2 * score_safety

def total_score_hawks := team_hawks_first_match_score + team_hawks_second_match_score
def total_score_eagles := team_eagles_first_match_score
def total_score_falcons := team_falcons_second_match_score

theorem total_points_each_team :
  total_score_hawks = 66 ∧ total_score_eagles = 47 ∧ total_score_falcons = 46 :=
by
  unfold total_score_hawks team_hawks_first_match_score team_hawks_second_match_score
           total_score_eagles team_eagles_first_match_score
           total_score_falcons team_falcons_second_match_score
           score_touchdown score_field_goal score_safety
  sorry

end NUMINAMATH_GPT_total_points_each_team_l1316_131670


namespace NUMINAMATH_GPT_distinct_integers_sum_l1316_131677

theorem distinct_integers_sum {p q r s t : ℤ} 
    (h1 : (8 - p) * (8 - q) * (8 - r) * (8 - s) * (8 - t) = 120)
    (h2 : p ≠ q) (h3 : p ≠ r) (h4 : p ≠ s) (h5 : p ≠ t) 
    (h6 : q ≠ r) (h7 : q ≠ s) (h8 : q ≠ t) 
    (h9 : r ≠ s) (h10 : r ≠ t) (h11 : s ≠ t) : 
  p + q + r + s + t = 35 := 
sorry

end NUMINAMATH_GPT_distinct_integers_sum_l1316_131677


namespace NUMINAMATH_GPT_hyperbola_eccentricity_l1316_131665

theorem hyperbola_eccentricity (m : ℝ) (h : 0 < m) :
  ∃ e, e = Real.sqrt (1 + m) ∧ e > Real.sqrt 2 → m > 1 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_l1316_131665


namespace NUMINAMATH_GPT_copies_made_in_half_hour_l1316_131601

theorem copies_made_in_half_hour :
  let copies_per_minute_machine1 := 40
  let copies_per_minute_machine2 := 55
  let time_minutes := 30
  (copies_per_minute_machine1 * time_minutes) + (copies_per_minute_machine2 * time_minutes) = 2850 := by
    sorry

end NUMINAMATH_GPT_copies_made_in_half_hour_l1316_131601


namespace NUMINAMATH_GPT_agatha_initial_money_60_l1316_131640

def Agatha_initial_money (spent_frame : ℕ) (spent_front_wheel: ℕ) (left_over: ℕ) : ℕ :=
  spent_frame + spent_front_wheel + left_over

theorem agatha_initial_money_60 :
  Agatha_initial_money 15 25 20 = 60 :=
by
  -- This line assumes $15 on frame, $25 on wheel, $20 left translates to a total of $60.
  sorry

end NUMINAMATH_GPT_agatha_initial_money_60_l1316_131640


namespace NUMINAMATH_GPT_general_admission_price_l1316_131639

theorem general_admission_price :
  ∃ x : ℝ,
    ∃ G V : ℕ,
      VIP_price = 45 ∧ Total_tickets_sold = 320 ∧ Total_revenue = 7500 ∧ VIP_tickets_less = 276 ∧
      G + V = Total_tickets_sold ∧ V = G - VIP_tickets_less ∧ 45 * V + x * G = Total_revenue ∧ x = 21.85 :=
sorry

end NUMINAMATH_GPT_general_admission_price_l1316_131639


namespace NUMINAMATH_GPT_geometric_sequence_condition_l1316_131618

variable (a_1 : ℝ) (q : ℝ)

noncomputable def geometric_sum (n : ℕ) : ℝ :=
if q = 1 then a_1 * n else a_1 * (1 - q^n) / (1 - q)

theorem geometric_sequence_condition (a_1 : ℝ) (q : ℝ) :
  (a_1 > 0) ↔ (geometric_sum a_1 q 2017 > 0) :=
by sorry

end NUMINAMATH_GPT_geometric_sequence_condition_l1316_131618


namespace NUMINAMATH_GPT_shirt_selling_price_l1316_131696

theorem shirt_selling_price (x : ℝ)
  (cost_price : x = 80)
  (initial_shirts_sold : ∃ s : ℕ, s = 30)
  (profit_per_shirt : ∃ p : ℝ, p = 50)
  (additional_shirts_per_dollar_decrease : ∃ a : ℕ, a = 2)
  (target_daily_profit : ∃ t : ℝ, t = 2000) :
  (x = 105 ∨ x = 120) := 
sorry

end NUMINAMATH_GPT_shirt_selling_price_l1316_131696


namespace NUMINAMATH_GPT_find_value_of_2_minus_c_l1316_131656

theorem find_value_of_2_minus_c (c d : ℤ) (h1 : 5 + c = 6 - d) (h2 : 3 + d = 8 + c) : 2 - c = -1 := 
by
  sorry

end NUMINAMATH_GPT_find_value_of_2_minus_c_l1316_131656


namespace NUMINAMATH_GPT_probability_at_least_one_exceeds_one_dollar_l1316_131649

noncomputable def prob_A : ℚ := 2 / 3
noncomputable def prob_B : ℚ := 1 / 2
noncomputable def prob_C : ℚ := 1 / 4

theorem probability_at_least_one_exceeds_one_dollar :
  (1 - ((1 - prob_A) * (1 - prob_B) * (1 - prob_C))) = 7 / 8 :=
by
  -- The proof can be conducted here
  sorry

end NUMINAMATH_GPT_probability_at_least_one_exceeds_one_dollar_l1316_131649


namespace NUMINAMATH_GPT_multiply_polynomials_l1316_131662

variables {R : Type*} [CommRing R] -- Define R as a commutative ring
variable (x : R) -- Define variable x in R

theorem multiply_polynomials : (2 * x) * (5 * x^2) = 10 * x^3 := 
sorry -- Placeholder for the proof

end NUMINAMATH_GPT_multiply_polynomials_l1316_131662


namespace NUMINAMATH_GPT_solve_for_q_l1316_131641

theorem solve_for_q (q : ℕ) : 16^4 = (8^3 / 2 : ℕ) * 2^(16 * q) → q = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_q_l1316_131641


namespace NUMINAMATH_GPT_range_of_a_l1316_131624

structure PropositionP (a : ℝ) : Prop :=
  (h : 2 * a + 1 > 5)

structure PropositionQ (a : ℝ) : Prop :=
  (h : -1 ≤ a ∧ a ≤ 3)

theorem range_of_a (a : ℝ) (hp : PropositionP a ∨ PropositionQ a) (hq : ¬(PropositionP a ∧ PropositionQ a)) :
  (-1 ≤ a ∧ a ≤ 2) ∨ (a > 3) :=
sorry

end NUMINAMATH_GPT_range_of_a_l1316_131624


namespace NUMINAMATH_GPT_percentage_of_white_chips_l1316_131667

theorem percentage_of_white_chips (T : ℕ) (h1 : 3 = 10 * T / 100) (h2 : 12 = 12): (15 / T * 100) = 50 := by
  sorry

end NUMINAMATH_GPT_percentage_of_white_chips_l1316_131667


namespace NUMINAMATH_GPT_find_b_l1316_131630

variable (p q r b : ℤ)

-- Conditions
def condition1 : Prop := p - q = 2
def condition2 : Prop := p - r = 1

-- The main statement to prove
def problem_statement : Prop :=
  b = (r - q) * ((p - q)^2 + (p - q) * (p - r) + (p - r)^2) → b = 7

theorem find_b (h1 : condition1 p q) (h2 : condition2 p r) (h3 : problem_statement p q r b) : b = 7 :=
sorry

end NUMINAMATH_GPT_find_b_l1316_131630


namespace NUMINAMATH_GPT_star_five_three_l1316_131643

def star (a b : ℤ) : ℤ := 4 * a - 2 * b

theorem star_five_three : star 5 3 = 14 := by
  sorry

end NUMINAMATH_GPT_star_five_three_l1316_131643


namespace NUMINAMATH_GPT_jason_money_in_usd_l1316_131680

noncomputable def jasonTotalInUSD : ℝ :=
  let init_quarters_value := 49 * 0.25
  let init_dimes_value    := 32 * 0.10
  let init_nickels_value  := 18 * 0.05
  let init_euros_in_usd   := 22.50 * 1.20
  let total_initial       := init_quarters_value + init_dimes_value + init_nickels_value + init_euros_in_usd

  let dad_quarters_value  := 25 * 0.25
  let dad_dimes_value     := 15 * 0.10
  let dad_nickels_value   := 10 * 0.05
  let dad_euros_in_usd    := 12 * 1.20
  let total_additional    := dad_quarters_value + dad_dimes_value + dad_nickels_value + dad_euros_in_usd

  total_initial + total_additional

theorem jason_money_in_usd :
  jasonTotalInUSD = 66 := 
sorry

end NUMINAMATH_GPT_jason_money_in_usd_l1316_131680


namespace NUMINAMATH_GPT_mixed_groups_count_l1316_131682

theorem mixed_groups_count (total_children : ℕ) (total_groups : ℕ)
  (group_size : ℕ) (total_boy_boy_photos : ℕ)
  (total_girl_girl_photos : ℕ)
  (total_photos : ℕ)
  (each_group_photos : ℕ)
  (total_mixed_photos : ℕ)
  (mixed_group_count : ℕ):
  total_children = 300 ∧
  total_groups = 100 ∧
  group_size = 3 ∧
  total_boy_boy_photos = 100 ∧
  total_girl_girl_photos = 56 ∧
  each_group_photos = 3 ∧
  total_photos = 100 * each_group_photos ∧
  total_mixed_photos = total_photos - total_boy_boy_photos - total_girl_girl_photos ∧
  mixed_group_count = total_mixed_photos / 2 →
  mixed_group_count = 72 :=
by
  sorry

end NUMINAMATH_GPT_mixed_groups_count_l1316_131682


namespace NUMINAMATH_GPT_selling_price_per_sweater_correct_l1316_131676

-- Definitions based on the problem's conditions
def balls_of_yarn_per_sweater := 4
def cost_per_ball_of_yarn := 6
def number_of_sweaters := 28
def total_gain := 308

-- Defining the required selling price per sweater
def total_cost_of_yarn : Nat := balls_of_yarn_per_sweater * cost_per_ball_of_yarn * number_of_sweaters
def total_revenue : Nat := total_cost_of_yarn + total_gain
def selling_price_per_sweater : ℕ := total_revenue / number_of_sweaters

theorem selling_price_per_sweater_correct :
  selling_price_per_sweater = 35 :=
  by
  sorry

end NUMINAMATH_GPT_selling_price_per_sweater_correct_l1316_131676


namespace NUMINAMATH_GPT_factorize_problem_1_factorize_problem_2_l1316_131604

theorem factorize_problem_1 (a b : ℝ) : -3 * a ^ 3 + 12 * a ^ 2 * b - 12 * a * b ^ 2 = -3 * a * (a - 2 * b) ^ 2 := 
sorry

theorem factorize_problem_2 (m n : ℝ) : 9 * (m + n) ^ 2 - (m - n) ^ 2 = 4 * (2 * m + n) * (m + 2 * n) := 
sorry

end NUMINAMATH_GPT_factorize_problem_1_factorize_problem_2_l1316_131604


namespace NUMINAMATH_GPT_range_of_alpha_l1316_131690

open Real

theorem range_of_alpha 
  (α : ℝ) (k : ℤ) :
  (sin α > 0) ∧ (cos α < 0) ∧ (sin α > cos α) →
  (∃ k : ℤ, (2 * k * π + π / 2 < α ∧ α < 2 * k * π + π) ∨ 
  (2 * k * π + (3 * π / 2) < α ∧ α < 2 * k * π + 2 * π)) := 
by 
  sorry

end NUMINAMATH_GPT_range_of_alpha_l1316_131690


namespace NUMINAMATH_GPT_solution_l1316_131600

noncomputable def problem (x : ℝ) : Prop :=
  0 < x ∧ (1/2 * (4 * x^2 - 1) = (x^2 - 50 * x - 20) * (x^2 + 25 * x + 10))

theorem solution (x : ℝ) (h : problem x) : x = 26 + Real.sqrt 677 :=
by
  sorry

end NUMINAMATH_GPT_solution_l1316_131600


namespace NUMINAMATH_GPT_distinct_real_pairs_l1316_131679

theorem distinct_real_pairs (x y : ℝ) (h1 : x ≠ y) (h2 : x^100 - y^100 = 2^99 * (x - y)) (h3 : x^200 - y^200 = 2^199 * (x - y)) :
  (x = 2 ∧ y = 0) ∨ (x = 0 ∧ y = 2) :=
sorry

end NUMINAMATH_GPT_distinct_real_pairs_l1316_131679


namespace NUMINAMATH_GPT_probability_of_b_l1316_131613

noncomputable def P : ℕ → ℝ := sorry

axiom P_a : P 0 = 0.15
axiom P_a_and_b : P 1 = 0.15
axiom P_neither_a_nor_b : P 2 = 0.6

theorem probability_of_b : P 3 = 0.4 := 
by
  sorry

end NUMINAMATH_GPT_probability_of_b_l1316_131613


namespace NUMINAMATH_GPT_students_in_all_classes_l1316_131625

theorem students_in_all_classes (total_students : ℕ) (students_photography : ℕ) (students_music : ℕ) (students_theatre : ℕ) (students_dance : ℕ) (students_at_least_two : ℕ) (students_in_all : ℕ) :
  total_students = 30 →
  students_photography = 15 →
  students_music = 18 →
  students_theatre = 12 →
  students_dance = 10 →
  students_at_least_two = 18 →
  students_in_all = 4 :=
by
  intros
  sorry

end NUMINAMATH_GPT_students_in_all_classes_l1316_131625


namespace NUMINAMATH_GPT_geometric_sequence_a6_l1316_131687

theorem geometric_sequence_a6 :
  ∃ (a : ℕ → ℝ), (∀ n, a n > 0) ∧ (∀ n, a (n + 1) = 2 * a n) ∧ (a 4 * a 10 = 16) → (a 6 = 2) :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_a6_l1316_131687


namespace NUMINAMATH_GPT_find_f1_l1316_131685

theorem find_f1 (f : ℝ → ℝ)
  (h : ∀ x, x ≠ 1 / 2 → f x + f ((x + 2) / (1 - 2 * x)) = x) :
  f 1 = 7 / 6 :=
sorry

end NUMINAMATH_GPT_find_f1_l1316_131685


namespace NUMINAMATH_GPT_find_f_2_l1316_131611

theorem find_f_2 (f : ℕ → ℕ) (h : ∀ x, f (x + 1) = 2 * x + 3) : f 2 = 5 :=
sorry

end NUMINAMATH_GPT_find_f_2_l1316_131611


namespace NUMINAMATH_GPT_inequality_solution_l1316_131606

theorem inequality_solution (x : ℝ) : 
  (3 - (1 / (3 * x + 4)) < 5) ↔ (x < -4 / 3) := 
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l1316_131606


namespace NUMINAMATH_GPT_Q_divisible_by_P_Q_divisible_by_P_squared_Q_not_divisible_by_P_cubed_l1316_131691

def Q (x : ℂ) (n : ℕ) : ℂ := (x + 1)^n + x^n + 1
def P (x : ℂ) : ℂ := x^2 + x + 1

-- Part a) Q(x) is divisible by P(x) if and only if n ≡ 2 (mod 6) or n ≡ 4 (mod 6)
theorem Q_divisible_by_P (x : ℂ) (n : ℕ) : 
  (Q x n) % (P x) = 0 ↔ (n % 6 = 2 ∨ n % 6 = 4) := sorry

-- Part b) Q(x) is divisible by P(x)^2 if and only if n ≡ 4 (mod 6)
theorem Q_divisible_by_P_squared (x : ℂ) (n : ℕ) : 
  (Q x n) % (P x)^2 = 0 ↔ n % 6 = 4 := sorry

-- Part c) Q(x) is never divisible by P(x)^3
theorem Q_not_divisible_by_P_cubed (x : ℂ) (n : ℕ) : 
  (Q x n) % (P x)^3 ≠ 0 := sorry

end NUMINAMATH_GPT_Q_divisible_by_P_Q_divisible_by_P_squared_Q_not_divisible_by_P_cubed_l1316_131691


namespace NUMINAMATH_GPT_first_term_to_common_difference_ratio_l1316_131603

theorem first_term_to_common_difference_ratio (a d : ℝ) 
  (h : (14 / 2) * (2 * a + 13 * d) = 3 * (7 / 2) * (2 * a + 6 * d)) :
  a / d = 4 :=
by
  sorry

end NUMINAMATH_GPT_first_term_to_common_difference_ratio_l1316_131603


namespace NUMINAMATH_GPT_three_pow_1000_mod_seven_l1316_131621

theorem three_pow_1000_mod_seven : (3 ^ 1000) % 7 = 4 := 
by 
  -- proof omitted
  sorry

end NUMINAMATH_GPT_three_pow_1000_mod_seven_l1316_131621


namespace NUMINAMATH_GPT_probability_is_correct_l1316_131636

variables (total_items truckA_first_class truckA_second_class truckB_first_class truckB_second_class brokenA brokenB remaining_items : ℕ)

-- Setting up the problem according to the given conditions
def conditions := (total_items = 10) ∧ 
                  (truckA_first_class = 2) ∧ (truckA_second_class = 2) ∧ 
                  (truckB_first_class = 4) ∧ (truckB_second_class = 2) ∧ 
                  (brokenA = 1) ∧ (brokenB = 1) ∧
                  (remaining_items = 8)

-- Calculating the probability of selecting a first-class item from the remaining items
def probability_of_first_class : ℚ :=
  1/3 * 1/2 + 1/6 * 5/8 + 1/3 * 5/8 + 1/6 * 3/4

-- The theorem to be proved
theorem probability_is_correct : 
  conditions total_items truckA_first_class truckA_second_class truckB_first_class truckB_second_class brokenA brokenB remaining_items →
  probability_of_first_class = 29/48 :=
sorry

end NUMINAMATH_GPT_probability_is_correct_l1316_131636


namespace NUMINAMATH_GPT_altitude_line_eq_circumcircle_eq_l1316_131635

noncomputable def point := ℝ × ℝ

noncomputable def A : point := (5, 1)
noncomputable def B : point := (1, 3)
noncomputable def C : point := (4, 4)

theorem altitude_line_eq : ∃ (k b : ℝ), (k = 2 ∧ b = -4) ∧ (∀ x y : ℝ, y = k * x + b ↔ 2 * x - y - 4 = 0) :=
sorry

theorem circumcircle_eq : ∃ (h k r : ℝ), (h = 3 ∧ k = 2 ∧ r = 5) ∧ (∀ x y : ℝ, (x - h)^2 + (y - k)^2 = r ↔ (x - 3)^2 + (y - 2)^2 = 5) :=
sorry

end NUMINAMATH_GPT_altitude_line_eq_circumcircle_eq_l1316_131635


namespace NUMINAMATH_GPT_bride_groom_couples_sum_l1316_131692

def wedding_reception (total_guests : ℕ) (friends : ℕ) (couples_guests : ℕ) : Prop :=
  total_guests - friends = couples_guests

theorem bride_groom_couples_sum (B G : ℕ) (total_guests : ℕ) (friends : ℕ) (couples_guests : ℕ) 
  (h1 : total_guests = 180) (h2 : friends = 100) (h3 : wedding_reception total_guests friends couples_guests) 
  (h4 : couples_guests = 80) : B + G = 40 := 
  by
  sorry

end NUMINAMATH_GPT_bride_groom_couples_sum_l1316_131692


namespace NUMINAMATH_GPT_max_students_l1316_131619

-- Definitions for the conditions
noncomputable def courses := ["Mathematics", "Physics", "Biology", "Music", "History", "Geography"]

def most_preferred (ranking : List String) : Prop :=
  "Mathematics" ∈ (ranking.take 2) ∨ "Mathematics" ∈ (ranking.take 3)

def least_preferred (ranking : List String) : Prop :=
  "Music" ∉ ranking.drop (ranking.length - 2)

def preference_constraints (ranking : List String) : Prop :=
  ranking.indexOf "History" < ranking.indexOf "Geography" ∧
  ranking.indexOf "Physics" < ranking.indexOf "Biology"

def all_rankings_unique (rankings : List (List String)) : Prop :=
  ∀ (r₁ r₂ : List String), r₁ ≠ r₂ → r₁ ∈ rankings → r₂ ∈ rankings → r₁ ≠ r₂

-- The goal statement
theorem max_students : 
  ∃ (rankings : List (List String)), 
  (∀ r ∈ rankings, most_preferred r) ∧
  (∀ r ∈ rankings, least_preferred r) ∧
  (∀ r ∈ rankings, preference_constraints r) ∧
  all_rankings_unique rankings ∧
  rankings.length = 44 :=
sorry

end NUMINAMATH_GPT_max_students_l1316_131619


namespace NUMINAMATH_GPT_player_A_winning_strategy_l1316_131689

-- Define the game state and the player's move
inductive Move
| single (index : Nat) : Move
| double (index : Nat) : Move

-- Winning strategy prop
def winning_strategy (n : Nat) (first_player : Bool) : Prop :=
  ∀ moves : List Move, moves.length ≤ n → (first_player → false) → true

-- Main theorem stating that player A always has a winning strategy
theorem player_A_winning_strategy (n : Nat) (h : n ≥ 1) : winning_strategy n true := 
by 
  -- directly prove the statement
  sorry

end NUMINAMATH_GPT_player_A_winning_strategy_l1316_131689


namespace NUMINAMATH_GPT_simplify_fraction_l1316_131637

variable (k : ℤ)

theorem simplify_fraction (a b : ℤ)
  (hk : a = 2)
  (hb : b = 4) :
  (6 * k + 12) / 3 = 2 * k + 4 ∧ (a : ℚ) / (b : ℚ) = 1 / 2 := 
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l1316_131637


namespace NUMINAMATH_GPT_phone_number_fraction_l1316_131644

theorem phone_number_fraction : 
  let total_valid_numbers := 6 * (10^6)
  let valid_numbers_with_conditions := 10^5
  valid_numbers_with_conditions / total_valid_numbers = 1 / 60 :=
by sorry

end NUMINAMATH_GPT_phone_number_fraction_l1316_131644


namespace NUMINAMATH_GPT_jackson_grade_increase_per_hour_l1316_131686

-- Define the necessary variables
variables (v s p G : ℕ)

-- The conditions from the problem
def study_condition1 : v = 9 := sorry
def study_condition2 : s = v / 3 := sorry
def grade_starts_at_zero : G = s * p := sorry
def final_grade : G = 45 := sorry

-- The final problem statement to prove
theorem jackson_grade_increase_per_hour :
  p = 15 :=
by
  -- Add our sorry to indicate the partial proof
  sorry

end NUMINAMATH_GPT_jackson_grade_increase_per_hour_l1316_131686


namespace NUMINAMATH_GPT_part_a_part_b_l1316_131674

-- Define a polygon type
structure Polygon :=
  (sides : ℕ)
  (area : ℝ)
  (grid_size : ℕ)

-- Define a function to verify drawable polygon
def DrawablePolygon (p : Polygon) : Prop :=
  ∃ (n : ℕ), p.grid_size = n ∧ p.area = n ^ 2

-- Part (a): 20-sided polygon with an area of 9
theorem part_a : DrawablePolygon {sides := 20, area := 9, grid_size := 3} :=
by
  sorry

-- Part (b): 100-sided polygon with an area of 49
theorem part_b : DrawablePolygon {sides := 100, area := 49, grid_size := 7} :=
by
  sorry

end NUMINAMATH_GPT_part_a_part_b_l1316_131674


namespace NUMINAMATH_GPT_one_over_a_plus_one_over_b_eq_neg_one_l1316_131653

theorem one_over_a_plus_one_over_b_eq_neg_one
  (a b : ℝ) (h_distinct : a ≠ b)
  (h_eq : a / b + a = b / a + b) :
  1 / a + 1 / b = -1 :=
by
  sorry

end NUMINAMATH_GPT_one_over_a_plus_one_over_b_eq_neg_one_l1316_131653


namespace NUMINAMATH_GPT_cost_of_article_l1316_131615

-- Definitions for conditions
def gain_340 (C G : ℝ) : Prop := 340 = C + G
def gain_360 (C G : ℝ) : Prop := 360 = C + G + 0.05 * C

-- Theorem to be proven
theorem cost_of_article (C G : ℝ) (h1 : gain_340 C G) (h2 : gain_360 C G) : C = 400 :=
by sorry

end NUMINAMATH_GPT_cost_of_article_l1316_131615


namespace NUMINAMATH_GPT_range_of_b_plus_c_l1316_131681

noncomputable def func (b c x : ℝ) : ℝ := x^2 + b*x + c * 3^x

theorem range_of_b_plus_c {b c : ℝ} (h1 : ∃ x, func b c x = 0)
  (h2 : ∀ x, (func b c x = 0 ↔ func b c (func b c x) = 0)) : 
  0 ≤ b + c ∧ b + c < 4 :=
by
  sorry

end NUMINAMATH_GPT_range_of_b_plus_c_l1316_131681

import Mathlib

namespace NUMINAMATH_GPT_money_difference_l1085_108571

def share_ratio (w x y z : ℝ) (k : ℝ) : Prop :=
  w = k ∧ x = 6 * k ∧ y = 2 * k ∧ z = 4 * k

theorem money_difference (k : ℝ) (h : k = 375) : 
  ∀ w x y z : ℝ, share_ratio w x y z k → (x - y) = 1500 := 
by
  intros w x y z h_ratio
  rw [share_ratio] at h_ratio
  have h_w : w = k := h_ratio.1
  have h_x : x = 6 * k := h_ratio.2.1
  have h_y : y = 2 * k := h_ratio.2.2.1
  rw [h_x, h_y]
  rw [h] at h_x h_y
  sorry

end NUMINAMATH_GPT_money_difference_l1085_108571


namespace NUMINAMATH_GPT_cos_225_eq_neg_sqrt2_div2_l1085_108558

theorem cos_225_eq_neg_sqrt2_div2 : Real.cos (225 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_GPT_cos_225_eq_neg_sqrt2_div2_l1085_108558


namespace NUMINAMATH_GPT_evaluate_expression_l1085_108533

def a : ℚ := 7/3

theorem evaluate_expression :
  (4 * a^2 - 11 * a + 7) * (2 * a - 3) = 140 / 27 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1085_108533


namespace NUMINAMATH_GPT_initial_amount_l1085_108598

theorem initial_amount (M : ℝ) (h1 : M * 2 - 50 > 0) (h2 : (M * 2 - 50) * 2 - 60 > 0) 
(h3 : ((M * 2 - 50) * 2 - 60) * 2 - 70 > 0) 
(h4 : (((M * 2 - 50) * 2 - 60) * 2 - 70) * 2 - 80 = 0) : M = 53.75 := 
sorry

end NUMINAMATH_GPT_initial_amount_l1085_108598


namespace NUMINAMATH_GPT_ratio_of_m_l1085_108579

theorem ratio_of_m (a b m m1 m2 : ℚ) 
  (h1 : a^2 - 2*a + (3/m) = 0)
  (h2 : a + b = 2 - 2/m)
  (h3 : a * b = 3/m)
  (h4 : (a/b) + (b/a) = 3/2) 
  (h5 : 8 * m^2 - 31 * m + 8 = 0)
  (h6 : m1 + m2 = 31/8)
  (h7 : m1 * m2 = 1) :
  (m1/m2) + (m2/m1) = 833/64 :=
sorry

end NUMINAMATH_GPT_ratio_of_m_l1085_108579


namespace NUMINAMATH_GPT_base_conversion_problem_l1085_108530

def base_to_dec (base : ℕ) (digits : List ℕ) : ℕ :=
  digits.foldr (λ x acc => x + base * acc) 0

theorem base_conversion_problem : 
  (base_to_dec 8 [2, 5, 3] : ℝ) / (base_to_dec 4 [1, 3] : ℝ) + 
  (base_to_dec 5 [1, 3, 2] : ℝ) / (base_to_dec 3 [2, 3] : ℝ) = 28.67 := by
  sorry

end NUMINAMATH_GPT_base_conversion_problem_l1085_108530


namespace NUMINAMATH_GPT_product_check_l1085_108501

theorem product_check : 
  (1200 < 31 * 53 ∧ 31 * 53 < 2400) ∧ 
  ¬ (1200 < 32 * 84 ∧ 32 * 84 < 2400) ∧ 
  ¬ (1200 < 63 * 54 ∧ 63 * 54 < 2400) ∧ 
  (1200 < 72 * 24 ∧ 72 * 24 < 2400) :=
by 
  sorry

end NUMINAMATH_GPT_product_check_l1085_108501


namespace NUMINAMATH_GPT_skyscraper_anniversary_l1085_108520

theorem skyscraper_anniversary (built_years_ago : ℕ) (anniversary_years : ℕ) (years_before : ℕ) :
    built_years_ago = 100 → anniversary_years = 200 → years_before = 5 → 
    (anniversary_years - years_before) - built_years_ago = 95 := by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_skyscraper_anniversary_l1085_108520


namespace NUMINAMATH_GPT_find_t_max_value_of_xyz_l1085_108529

-- Problem (1)
theorem find_t (t : ℝ) (x : ℝ) (h1 : |2 * x + t| - t ≤ 8) (sol_set : -5 ≤ x ∧ x ≤ 4) : t = 1 :=
sorry

-- Problem (2)
theorem max_value_of_xyz (x y z : ℝ) (h2 : x^2 + (1/4) * y^2 + (1/9) * z^2 = 2) : x + y + z ≤ 2 * Real.sqrt 7 :=
sorry

end NUMINAMATH_GPT_find_t_max_value_of_xyz_l1085_108529


namespace NUMINAMATH_GPT_min_value_expression_l1085_108580

theorem min_value_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 1) : 
  ∃ c, ∀ x y, 0 < x → 0 < y → x + y = 1 → c = 9 ∧ ((1 / x) + (4 / y)) ≥ 9 := 
sorry

end NUMINAMATH_GPT_min_value_expression_l1085_108580


namespace NUMINAMATH_GPT_arts_school_probability_l1085_108508

theorem arts_school_probability :
  let cultural_courses := 3
  let arts_courses := 3
  let total_periods := 6
  let total_arrangements := Nat.factorial total_periods
  let no_adjacent_more_than_one_separator := (72 + 216 + 144)
  (no_adjacent_more_than_one_separator : ℝ) / (total_arrangements : ℝ) = (3 / 5 : ℝ) := 
by 
  sorry

end NUMINAMATH_GPT_arts_school_probability_l1085_108508


namespace NUMINAMATH_GPT_range_of_m_for_common_point_l1085_108526

-- Define the quadratic function
def quadratic_function (x m : ℝ) : ℝ :=
  -x^2 - 2 * x + m

-- Define the condition for a common point with the x-axis (i.e., it has real roots)
def has_common_point_with_x_axis (m : ℝ) : Prop :=
  ∃ x : ℝ, quadratic_function x m = 0

-- The theorem statement
theorem range_of_m_for_common_point : ∀ m : ℝ, has_common_point_with_x_axis m ↔ m ≥ -1 := 
sorry

end NUMINAMATH_GPT_range_of_m_for_common_point_l1085_108526


namespace NUMINAMATH_GPT_a_eq_b_if_b2_ab_1_divides_a2_ab_1_l1085_108500

theorem a_eq_b_if_b2_ab_1_divides_a2_ab_1 (a b : ℕ) (ha_pos : a > 0) (hb_pos : b > 0)
  (h : b^2 + a * b + 1 ∣ a^2 + a * b + 1) : a = b :=
by
  sorry

end NUMINAMATH_GPT_a_eq_b_if_b2_ab_1_divides_a2_ab_1_l1085_108500


namespace NUMINAMATH_GPT_length_of_rectangle_l1085_108546

-- Define the conditions as given in the problem
variables (width : ℝ) (perimeter : ℝ) (length : ℝ)

-- The conditions provided
def conditions : Prop :=
  width = 15 ∧ perimeter = 70 ∧ perimeter = 2 * (length + width)

-- The statement to prove: the length of the rectangle is 20 feet
theorem length_of_rectangle {width perimeter length : ℝ} (h : conditions width perimeter length) : length = 20 :=
by 
  -- This is where the proof steps would go
  sorry

end NUMINAMATH_GPT_length_of_rectangle_l1085_108546


namespace NUMINAMATH_GPT_card_arrangement_bound_l1085_108576

theorem card_arrangement_bound : 
  ∀ (cards : ℕ) (cells : ℕ), cards = 1000 → cells = 1994 → 
  ∃ arrangements : ℕ, arrangements = cells - cards + 1 ∧ arrangements < 500000 :=
by {
  sorry
}

end NUMINAMATH_GPT_card_arrangement_bound_l1085_108576


namespace NUMINAMATH_GPT_geometric_triangle_q_range_l1085_108503

theorem geometric_triangle_q_range (a : ℝ) (q : ℝ) (h : 0 < q) 
  (h1 : a + q * a > (q ^ 2) * a)
  (h2 : q * a + (q ^ 2) * a > a)
  (h3 : a + (q ^ 2) * a > q * a) : 
  q ∈ Set.Ioo ((Real.sqrt 5 - 1) / 2) ((1 + Real.sqrt 5) / 2) :=
sorry

end NUMINAMATH_GPT_geometric_triangle_q_range_l1085_108503


namespace NUMINAMATH_GPT_minimum_basketballs_sold_l1085_108590

theorem minimum_basketballs_sold :
  ∃ (F B K : ℕ), F + B + K = 180 ∧ 3 * F + 5 * B + 10 * K = 800 ∧ F > B ∧ B > K ∧ K = 2 :=
by
  sorry

end NUMINAMATH_GPT_minimum_basketballs_sold_l1085_108590


namespace NUMINAMATH_GPT_part_a_part_b_l1085_108547

def square_side_length : ℝ := 10
def square_area (side_length : ℝ) : ℝ := side_length * side_length
def triangle_area (base : ℝ) (height : ℝ) : ℝ := 0.5 * base * height

-- Part (a)
theorem part_a :
  let side_length := square_side_length
  let square := square_area side_length
  let triangle := triangle_area side_length side_length
  square - triangle = 50 := by
  sorry

-- Part (b)
theorem part_b :
  let side_length := square_side_length
  let square := square_area side_length
  let small_triangle_area := square / 8
  2 * small_triangle_area = 25 := by
  sorry

end NUMINAMATH_GPT_part_a_part_b_l1085_108547


namespace NUMINAMATH_GPT_sum_problem3_equals_50_l1085_108548

-- Assume problem3_condition is a placeholder for the actual conditions described in problem 3
-- and sum_problem3 is a placeholder for the sum of elements described in problem 3.

axiom problem3_condition : Prop
axiom sum_problem3 : ℕ

theorem sum_problem3_equals_50 (h : problem3_condition) : sum_problem3 = 50 :=
sorry

end NUMINAMATH_GPT_sum_problem3_equals_50_l1085_108548


namespace NUMINAMATH_GPT_cube_surface_area_l1085_108512

/-- Given a cube with a space diagonal of 6, the surface area is 72. -/
theorem cube_surface_area (s : ℝ) (h : s * Real.sqrt 3 = 6) : 6 * s^2 = 72 :=
by
  sorry

end NUMINAMATH_GPT_cube_surface_area_l1085_108512


namespace NUMINAMATH_GPT_geometric_series_first_term_l1085_108595

theorem geometric_series_first_term (a r : ℝ)
  (h1 : a / (1 - r) = 20)
  (h2 : a^2 / (1 - r^2) = 80) :
  a = 20 / 3 :=
by
  sorry

end NUMINAMATH_GPT_geometric_series_first_term_l1085_108595


namespace NUMINAMATH_GPT_choir_singers_joined_final_verse_l1085_108531

theorem choir_singers_joined_final_verse (total_singers : ℕ) (first_verse_fraction : ℚ)
  (second_verse_fraction : ℚ) (initial_remaining : ℕ) (second_verse_joined : ℕ) : 
  total_singers = 30 → 
  first_verse_fraction = 1 / 2 → 
  second_verse_fraction = 1 / 3 → 
  initial_remaining = total_singers / 2 → 
  second_verse_joined = initial_remaining / 3 → 
  (total_singers - (initial_remaining + second_verse_joined)) = 10 := 
by
  intros
  sorry

end NUMINAMATH_GPT_choir_singers_joined_final_verse_l1085_108531


namespace NUMINAMATH_GPT_giant_spider_leg_cross_sectional_area_l1085_108518

theorem giant_spider_leg_cross_sectional_area :
  let previous_spider_weight := 6.4
  let weight_multiplier := 2.5
  let pressure := 4
  let num_legs := 8

  let giant_spider_weight := weight_multiplier * previous_spider_weight
  let weight_per_leg := giant_spider_weight / num_legs
  let cross_sectional_area := weight_per_leg / pressure

  cross_sectional_area = 0.5 :=
by 
  sorry

end NUMINAMATH_GPT_giant_spider_leg_cross_sectional_area_l1085_108518


namespace NUMINAMATH_GPT_salary_recovery_l1085_108542

theorem salary_recovery (S : ℝ) : 
  (0.80 * S) + (0.25 * (0.80 * S)) = S :=
by
  sorry

end NUMINAMATH_GPT_salary_recovery_l1085_108542


namespace NUMINAMATH_GPT_customers_who_left_tip_l1085_108523

-- Define the initial number of customers
def initial_customers : ℕ := 39

-- Define the additional number of customers during lunch rush
def additional_customers : ℕ := 12

-- Define the number of customers who didn't leave a tip
def no_tip_customers : ℕ := 49

-- Prove the number of customers who did leave a tip
theorem customers_who_left_tip : (initial_customers + additional_customers) - no_tip_customers = 2 := by
  sorry

end NUMINAMATH_GPT_customers_who_left_tip_l1085_108523


namespace NUMINAMATH_GPT_number_of_games_in_division_l1085_108543

theorem number_of_games_in_division (P Q : ℕ) (h1 : P > 2 * Q) (h2 : Q > 6) (schedule_eq : 4 * P + 5 * Q = 82) : 4 * P = 52 :=
by sorry

end NUMINAMATH_GPT_number_of_games_in_division_l1085_108543


namespace NUMINAMATH_GPT_gcd_g_values_l1085_108569

def g (x : ℤ) : ℤ := x^2 - 2 * x + 2023

theorem gcd_g_values : gcd (g 102) (g 103) = 1 := by
  sorry

end NUMINAMATH_GPT_gcd_g_values_l1085_108569


namespace NUMINAMATH_GPT_parallelogram_side_length_l1085_108559

theorem parallelogram_side_length 
  (s : ℝ) 
  (A : ℝ)
  (angle : ℝ)
  (adj1 adj2 : ℝ) 
  (h : adj1 = s) 
  (h1 : adj2 = 2 * s) 
  (h2 : angle = 30)
  (h3 : A = 8 * Real.sqrt 3): 
  s = 2 * Real.sqrt 2 :=
by
  -- sorry to skip proofs
  sorry

end NUMINAMATH_GPT_parallelogram_side_length_l1085_108559


namespace NUMINAMATH_GPT_max_distance_of_MN_l1085_108550

noncomputable def curve_C_polar (θ : ℝ) : ℝ := 2 * Real.cos θ

noncomputable def curve_C_cartesian (x y : ℝ) := x^2 + y^2 - 2 * x

noncomputable def line_l (t : ℝ) : ℝ × ℝ :=
  ( -1 + (Real.sqrt 5 / 5) * t, (2 * Real.sqrt 5 / 5) * t)

def point_M : ℝ × ℝ := (0, 2)

noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

noncomputable def center_C : ℝ × ℝ := (1, 0)

theorem max_distance_of_MN :
  ∃ N : ℝ × ℝ, 
  ∀ (θ : ℝ), N = (curve_C_polar θ * Real.cos θ, curve_C_polar θ * Real.sin θ) →
  distance point_M N ≤ Real.sqrt 5 + 1 :=
sorry

end NUMINAMATH_GPT_max_distance_of_MN_l1085_108550


namespace NUMINAMATH_GPT_y1_y2_positive_l1085_108556

theorem y1_y2_positive 
  (x1 x2 x3 : ℝ)
  (y1 y2 y3 : ℝ)
  (h_line1 : y1 = -2 * x1 + 3)
  (h_line2 : y2 = -2 * x2 + 3)
  (h_line3 : y3 = -2 * x3 + 3)
  (h_order : x1 < x2 ∧ x2 < x3)
  (h_product_neg : x2 * x3 < 0) :
  y1 * y2 > 0 :=
by
  sorry

end NUMINAMATH_GPT_y1_y2_positive_l1085_108556


namespace NUMINAMATH_GPT_candy_per_smaller_bag_l1085_108594

-- Define the variables and parameters
def george_candy : ℕ := 648
def friends : ℕ := 3
def total_people : ℕ := friends + 1
def smaller_bags : ℕ := 8

-- Define the theorem
theorem candy_per_smaller_bag : (george_candy / total_people) / smaller_bags = 20 :=
by
  -- Assume the proof steps, not required to actually complete
  sorry

end NUMINAMATH_GPT_candy_per_smaller_bag_l1085_108594


namespace NUMINAMATH_GPT_no_integer_solutions_l1085_108502

theorem no_integer_solutions (a b c : ℤ) : ¬ (a^2 + b^2 = 8 * c + 6) :=
sorry

end NUMINAMATH_GPT_no_integer_solutions_l1085_108502


namespace NUMINAMATH_GPT_max_sum_of_three_integers_with_product_24_l1085_108517

theorem max_sum_of_three_integers_with_product_24 : ∃ (a b c : ℤ), (a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a * b * c = 24 ∧ a + b + c = 15) :=
by
  sorry

end NUMINAMATH_GPT_max_sum_of_three_integers_with_product_24_l1085_108517


namespace NUMINAMATH_GPT_intersection_of_M_and_N_l1085_108538

def M : Set ℤ := {0, 1}
def N : Set ℤ := {-1, 0}

theorem intersection_of_M_and_N : M ∩ N = {0} := by
  sorry

end NUMINAMATH_GPT_intersection_of_M_and_N_l1085_108538


namespace NUMINAMATH_GPT_solve_equation_l1085_108593

theorem solve_equation : ∃ x : ℝ, 2 * x + 1 = 0 ∧ x = -1 / 2 := by
  sorry

end NUMINAMATH_GPT_solve_equation_l1085_108593


namespace NUMINAMATH_GPT_remainder_equiv_l1085_108597

theorem remainder_equiv (x : ℤ) (h : ∃ k : ℤ, x = 95 * k + 31) : ∃ m : ℤ, x = 19 * m + 12 := 
sorry

end NUMINAMATH_GPT_remainder_equiv_l1085_108597


namespace NUMINAMATH_GPT_medium_size_shoes_initially_stocked_l1085_108525

variable {M : ℕ}  -- The number of medium-size shoes initially stocked

noncomputable def initial_pairs_eq (M : ℕ) := 22 + M + 24
noncomputable def shoes_sold (M : ℕ) := initial_pairs_eq M - 13

theorem medium_size_shoes_initially_stocked :
  shoes_sold M = 83 → M = 26 :=
by
  sorry

end NUMINAMATH_GPT_medium_size_shoes_initially_stocked_l1085_108525


namespace NUMINAMATH_GPT_cost_of_each_item_number_of_purchasing_plans_l1085_108582

-- Question 1: Cost of each item
theorem cost_of_each_item : 
  ∃ (x y : ℕ), 
    (10 * x + 5 * y = 2000) ∧ 
    (5 * x + 3 * y = 1050) ∧ 
    (x = 150) ∧ 
    (y = 100) :=
by
    sorry

-- Question 2: Number of different purchasing plans
theorem number_of_purchasing_plans : 
  (∀ (a b : ℕ), 
    (150 * a + 100 * b = 4000) → 
    (a ≥ 12) → 
    (b ≥ 12) → 
    (4 = 4)) :=
by
    sorry

end NUMINAMATH_GPT_cost_of_each_item_number_of_purchasing_plans_l1085_108582


namespace NUMINAMATH_GPT_min_value_fraction_l1085_108519

theorem min_value_fraction (x y : ℝ) (h₁ : x + y = 4) (h₂ : x > y) (h₃ : y > 0) : (∃ z : ℝ, z = (2 / (x - y)) + (1 / y) ∧ z = 2) :=
by
  sorry

end NUMINAMATH_GPT_min_value_fraction_l1085_108519


namespace NUMINAMATH_GPT_total_weight_of_three_packages_l1085_108562

theorem total_weight_of_three_packages (a b c d : ℝ)
  (h1 : a + b = 162)
  (h2 : b + c = 164)
  (h3 : c + a = 168) :
  a + b + c = 247 :=
sorry

end NUMINAMATH_GPT_total_weight_of_three_packages_l1085_108562


namespace NUMINAMATH_GPT_find_vertical_shift_l1085_108566

theorem find_vertical_shift (A B C D : ℝ) (h1 : ∀ x, -3 ≤ A * Real.cos (B * x + C) + D ∧ A * Real.cos (B * x + C) + D ≤ 5) :
  D = 1 :=
by
  -- Here's where the proof would go
  sorry

end NUMINAMATH_GPT_find_vertical_shift_l1085_108566


namespace NUMINAMATH_GPT_cos_alpha_plus_beta_l1085_108574

theorem cos_alpha_plus_beta (α β : ℝ) (hα : Complex.exp (Complex.I * α) = 4 / 5 + Complex.I * 3 / 5)
  (hβ : Complex.exp (Complex.I * β) = -5 / 13 + Complex.I * 12 / 13) : 
  Real.cos (α + β) = -7 / 13 :=
  sorry

end NUMINAMATH_GPT_cos_alpha_plus_beta_l1085_108574


namespace NUMINAMATH_GPT_ratio_of_parts_l1085_108513

theorem ratio_of_parts (N : ℝ) (h1 : (1/4) * (2/5) * N = 14) (h2 : 0.40 * N = 168) : (2/5) * N / N = 1 / 2.5 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_parts_l1085_108513


namespace NUMINAMATH_GPT_probability_point_between_X_and_Z_l1085_108506

theorem probability_point_between_X_and_Z (XW XZ YW : ℝ) (h1 : XW = 4 * XZ) (h2 : XW = 8 * YW) :
  (XZ / XW) = 1 / 4 := by
  sorry

end NUMINAMATH_GPT_probability_point_between_X_and_Z_l1085_108506


namespace NUMINAMATH_GPT_minimum_valid_N_exists_l1085_108537

theorem minimum_valid_N_exists (N : ℝ) (a : ℕ → ℕ) :
  (∀ n : ℕ, a n > 0) →
  (∀ n : ℕ, a n < a (n+1)) →
  (∀ n : ℕ, (a (2*n - 1) + a (2*n)) / a n = N) →
  N ≥ 4 :=
by
  sorry

end NUMINAMATH_GPT_minimum_valid_N_exists_l1085_108537


namespace NUMINAMATH_GPT_trig_expression_evaluation_l1085_108570

theorem trig_expression_evaluation (θ : ℝ) (h : Real.tan θ = 2) :
  Real.sin θ ^ 2 + (Real.sin θ * Real.cos θ) - 2 * (Real.cos θ ^ 2) = 4 / 5 := 
by
  sorry

end NUMINAMATH_GPT_trig_expression_evaluation_l1085_108570


namespace NUMINAMATH_GPT_graph_not_pass_second_quadrant_l1085_108509

theorem graph_not_pass_second_quadrant (a b : ℝ) (h1 : a > 1) (h2 : b < -1) :
  ¬ ∃ (x : ℝ), y = a^x + b ∧ x < 0 ∧ y > 0 :=
by
  sorry

end NUMINAMATH_GPT_graph_not_pass_second_quadrant_l1085_108509


namespace NUMINAMATH_GPT_midpoint_product_l1085_108551

theorem midpoint_product (x y z : ℤ) 
  (h1 : (2 + x) / 2 = 4) 
  (h2 : (10 + y) / 2 = 6) 
  (h3 : (5 + z) / 2 = 3) : 
  x * y * z = 12 := 
by
  sorry

end NUMINAMATH_GPT_midpoint_product_l1085_108551


namespace NUMINAMATH_GPT_range_of_m_l1085_108514

theorem range_of_m (x y m : ℝ) (h1 : x > 0) (h2 : y > 0)
  (h3 : (2 / x) + (3 / y) = 1)
  (h4 : 3 * x + 2 * y > m^2 + 2 * m) :
  -6 < m ∧ m < 4 :=
sorry

end NUMINAMATH_GPT_range_of_m_l1085_108514


namespace NUMINAMATH_GPT_zoo_animal_count_l1085_108563

def tiger_enclosures : ℕ := 4
def zebra_enclosures_per_tiger_enclosures : ℕ := 2
def zebra_enclosures : ℕ := tiger_enclosures * zebra_enclosures_per_tiger_enclosures
def giraffe_enclosures_per_zebra_enclosures : ℕ := 3
def giraffe_enclosures : ℕ := zebra_enclosures * giraffe_enclosures_per_zebra_enclosures
def tigers_per_enclosure : ℕ := 4
def zebras_per_enclosure : ℕ := 10
def giraffes_per_enclosure : ℕ := 2

def total_animals_in_zoo : ℕ := 
    (tiger_enclosures * tigers_per_enclosure) + 
    (zebra_enclosures * zebras_per_enclosure) + 
    (giraffe_enclosures * giraffes_per_enclosure)

theorem zoo_animal_count : total_animals_in_zoo = 144 := 
by
  -- proof would go here
  sorry

end NUMINAMATH_GPT_zoo_animal_count_l1085_108563


namespace NUMINAMATH_GPT_consecutive_composite_numbers_bound_l1085_108505

theorem consecutive_composite_numbers_bound (n : ℕ) (hn: 0 < n) :
  ∃ (seq : Fin n → ℕ), (∀ i, ¬ Nat.Prime (seq i)) ∧ (∀ i, seq i < 4^(n+1)) :=
sorry

end NUMINAMATH_GPT_consecutive_composite_numbers_bound_l1085_108505


namespace NUMINAMATH_GPT_diorama_time_subtraction_l1085_108532

theorem diorama_time_subtraction (P B X : ℕ) (h1 : B = 3 * P - X) (h2 : B = 49) (h3 : P + B = 67) : X = 5 :=
by
  sorry

end NUMINAMATH_GPT_diorama_time_subtraction_l1085_108532


namespace NUMINAMATH_GPT_number_of_machines_in_first_scenario_l1085_108545

noncomputable def machine_work_rate (R : ℝ) (hours_per_job : ℝ) : Prop :=
  (6 * R * 8 = 1)

noncomputable def machines_first_scenario (M : ℝ) (R : ℝ) (hours_per_job_first : ℝ) : Prop :=
  (M * R * hours_per_job_first = 1)

theorem number_of_machines_in_first_scenario (M : ℝ) (R : ℝ) :
  machine_work_rate R 8 ∧ machines_first_scenario M R 6 -> M = 8 :=
sorry

end NUMINAMATH_GPT_number_of_machines_in_first_scenario_l1085_108545


namespace NUMINAMATH_GPT_andrew_total_payment_l1085_108575

-- Given conditions
def quantity_of_grapes := 14
def rate_per_kg_grapes := 54
def quantity_of_mangoes := 10
def rate_per_kg_mangoes := 62

-- Calculations
def cost_of_grapes := quantity_of_grapes * rate_per_kg_grapes
def cost_of_mangoes := quantity_of_mangoes * rate_per_kg_mangoes
def total_amount_paid := cost_of_grapes + cost_of_mangoes

-- Theorem to prove
theorem andrew_total_payment : total_amount_paid = 1376 := by
  sorry

end NUMINAMATH_GPT_andrew_total_payment_l1085_108575


namespace NUMINAMATH_GPT_hyperbola_solution_l1085_108573

noncomputable def hyperbola_eq (x y : ℝ) : Prop :=
  y^2 - x^2 / 3 = 1

theorem hyperbola_solution :
  ∃ x y : ℝ,
    (∃ c : ℝ, c = 2) ∧
    (∃ a : ℝ, a = 1) ∧
    (∃ n : ℝ, n = 1) ∧
    (∃ b : ℝ, b^2 = 3) ∧
    (∃ m : ℝ, m = -3) ∧
    hyperbola_eq x y := sorry

end NUMINAMATH_GPT_hyperbola_solution_l1085_108573


namespace NUMINAMATH_GPT_smallest_mult_to_cube_l1085_108572

def y : ℕ := 2^3 * 3^4 * 4^5 * 5^6 * 6^7 * 7^8 * 8^9 * 9^10

theorem smallest_mult_to_cube (n : ℕ) (h : ∃ n, ∃ k, n * y = k^3) : n = 4500 := 
  sorry

end NUMINAMATH_GPT_smallest_mult_to_cube_l1085_108572


namespace NUMINAMATH_GPT_smaller_angle_formed_by_hands_at_3_15_l1085_108540

def degrees_per_hour : ℝ := 30
def degrees_per_minute : ℝ := 6
def hour_hand_degrees_per_minute : ℝ := 0.5

def minute_position (minute : ℕ) : ℝ :=
  minute * degrees_per_minute

def hour_position (hour : ℕ) (minute : ℕ) : ℝ :=
  hour * degrees_per_hour + minute * hour_hand_degrees_per_minute

theorem smaller_angle_formed_by_hands_at_3_15 : 
  minute_position 15 = 90 ∧ 
  hour_position 3 15 = 97.5 →
  abs (hour_position 3 15 - minute_position 15) = 7.5 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_smaller_angle_formed_by_hands_at_3_15_l1085_108540


namespace NUMINAMATH_GPT_total_weight_of_family_l1085_108527

theorem total_weight_of_family (M D C : ℕ) (h1 : D + C = 60) (h2 : C = 1 / 5 * M) (h3 : D = 40) : M + D + C = 160 :=
sorry

end NUMINAMATH_GPT_total_weight_of_family_l1085_108527


namespace NUMINAMATH_GPT_find_smallest_k_l1085_108510

variable (k : ℕ)

theorem find_smallest_k :
  (∀ a : ℝ, 0 ≤ a ∧ a ≤ 1 → (∀ n : ℕ, n > 0 → a^k * (1-a)^n < 1 / (n+1)^3)) ↔ k = 4 :=
sorry

end NUMINAMATH_GPT_find_smallest_k_l1085_108510


namespace NUMINAMATH_GPT_positive_integer_solutions_l1085_108564

theorem positive_integer_solutions : 
  (∀ x : ℤ, ((1 + 2 * (x:ℝ)) / 4 - (1 - 3 * (x:ℝ)) / 10 > -1 / 5) ∧ (3 * (x:ℝ) - 1 < 2 * ((x:ℝ) + 1)) → (x = 1 ∨ x = 2)) :=
by 
  sorry

end NUMINAMATH_GPT_positive_integer_solutions_l1085_108564


namespace NUMINAMATH_GPT_train_pass_time_l1085_108554

theorem train_pass_time (train_length : ℕ) (platform_length : ℕ) (speed : ℕ) (h1 : train_length = 50) (h2 : platform_length = 100) (h3 : speed = 15) : 
  (train_length + platform_length) / speed = 10 :=
by
  sorry

end NUMINAMATH_GPT_train_pass_time_l1085_108554


namespace NUMINAMATH_GPT_temperature_at_noon_l1085_108577

-- Definitions of the given conditions.
def morning_temperature : ℝ := 4
def temperature_drop : ℝ := 10

-- The theorem statement that needs to be proven.
theorem temperature_at_noon : morning_temperature - temperature_drop = -6 :=
by
  -- The proof can be filled in by solving the stated theorem.
  sorry

end NUMINAMATH_GPT_temperature_at_noon_l1085_108577


namespace NUMINAMATH_GPT_Henry_has_four_Skittles_l1085_108539

-- Defining the initial amount of Skittles Bridget has
def Bridget_initial := 4

-- Defining the final amount of Skittles Bridget has after receiving all of Henry's Skittles
def Bridget_final := 8

-- Defining the amount of Skittles Henry has
def Henry_Skittles := Bridget_final - Bridget_initial

-- The proof statement to be proven
theorem Henry_has_four_Skittles : Henry_Skittles = 4 := by
  sorry

end NUMINAMATH_GPT_Henry_has_four_Skittles_l1085_108539


namespace NUMINAMATH_GPT_luke_number_of_rounds_l1085_108578

variable (points_per_round total_points : ℕ)

theorem luke_number_of_rounds 
  (h1 : points_per_round = 3)
  (h2 : total_points = 78) : 
  total_points / points_per_round = 26 := 
by 
  sorry

end NUMINAMATH_GPT_luke_number_of_rounds_l1085_108578


namespace NUMINAMATH_GPT_problem_solution_l1085_108522

def is_quadratic (y : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), ∀ x, y x = a * x^2 + b * x + c

def not_quadratic_func := 
  let yA := fun x => -2 * x^2
  let yB := fun x => 2 * (x - 1)^2 + 1
  let yC := fun x => (x - 3)^2 - x^2
  let yD := fun a => a * (8 - a)
  (¬ is_quadratic yC) ∧ (is_quadratic yA) ∧ (is_quadratic yB) ∧ (is_quadratic yD)

theorem problem_solution : not_quadratic_func := 
sorry

end NUMINAMATH_GPT_problem_solution_l1085_108522


namespace NUMINAMATH_GPT_total_frogs_in_ponds_l1085_108585

def pondA_frogs := 32
def pondB_frogs := pondA_frogs / 2

theorem total_frogs_in_ponds : pondA_frogs + pondB_frogs = 48 := by
  sorry

end NUMINAMATH_GPT_total_frogs_in_ponds_l1085_108585


namespace NUMINAMATH_GPT_rectangle_width_l1085_108565

theorem rectangle_width (w l A : ℕ) 
  (h1 : l = 3 * w)
  (h2 : A = l * w)
  (h3 : A = 108) : 
  w = 6 := 
sorry

end NUMINAMATH_GPT_rectangle_width_l1085_108565


namespace NUMINAMATH_GPT_average_height_correct_l1085_108536

noncomputable def initially_calculated_average_height 
  (num_students : ℕ) (incorrect_height correct_height : ℕ) 
  (actual_average : ℝ) 
  (A : ℝ) : Prop :=
  let incorrect_sum := num_students * A
  let height_difference := incorrect_height - correct_height
  let actual_sum := num_students * actual_average
  incorrect_sum = actual_sum + height_difference

theorem average_height_correct 
  (num_students : ℕ) (incorrect_height correct_height : ℕ) 
  (actual_average : ℝ) :
  initially_calculated_average_height num_students incorrect_height correct_height actual_average 175 :=
by {
  sorry
}

end NUMINAMATH_GPT_average_height_correct_l1085_108536


namespace NUMINAMATH_GPT_find_abc_sum_l1085_108561

theorem find_abc_sum (a b c : ℤ) (h1 : a - 2 * b = 4) (h2 : a * b + c^2 - 1 = 0) :
  a + b + c = 5 ∨ a + b + c = 3 ∨ a + b + c = -1 ∨ a + b + c = -3 :=
  sorry

end NUMINAMATH_GPT_find_abc_sum_l1085_108561


namespace NUMINAMATH_GPT_sin_half_angle_correct_l1085_108583

noncomputable def sin_half_angle (theta : ℝ) (h1 : Real.sin theta = 3 / 5) (h2 : 5 * Real.pi / 2 < theta ∧ theta < 3 * Real.pi) : ℝ :=
  -3 * Real.sqrt 10 / 10

theorem sin_half_angle_correct (theta : ℝ) (h1 : Real.sin theta = 3 / 5) (h2 : 5 * Real.pi / 2 < theta ∧ theta < 3 * Real.pi) :
  sin_half_angle theta h1 h2 = Real.sin (theta / 2) :=
by
  sorry

end NUMINAMATH_GPT_sin_half_angle_correct_l1085_108583


namespace NUMINAMATH_GPT_parabola_focus_l1085_108535

theorem parabola_focus (x y : ℝ) (h : y = 4 * x^2) : (0, 1) = (0, 1) :=
by 
  -- key steps would go here
  sorry

end NUMINAMATH_GPT_parabola_focus_l1085_108535


namespace NUMINAMATH_GPT_hari_contribution_l1085_108541

theorem hari_contribution (c_p: ℕ) (m_p: ℕ) (ratio_p: ℕ) 
                          (m_h: ℕ) (ratio_h: ℕ) (profit_ratio_p: ℕ) (profit_ratio_h: ℕ) 
                          (c_h: ℕ) : 
  (c_p = 3780) → 
  (m_p = 12) → 
  (ratio_p = 2) → 
  (m_h = 7) → 
  (ratio_h = 3) → 
  (profit_ratio_p = 2) →
  (profit_ratio_h = 3) →
  (c_p * m_p * profit_ratio_h) = (c_h * m_h * profit_ratio_p) → 
  c_h = 9720 :=
by
  intros
  sorry

end NUMINAMATH_GPT_hari_contribution_l1085_108541


namespace NUMINAMATH_GPT_tree_height_at_2_years_l1085_108584

theorem tree_height_at_2_years (h : ℕ → ℚ) (h5 : h 5 = 243) 
  (h_rec : ∀ n, h (n - 1) = h n / 3) : h 2 = 9 :=
  sorry

end NUMINAMATH_GPT_tree_height_at_2_years_l1085_108584


namespace NUMINAMATH_GPT_find_cost_price_of_clock_l1085_108507

namespace ClockCost

variable (C : ℝ)

def cost_price_each_clock (n : ℝ) (gain1 : ℝ) (gain2 : ℝ) (uniform_gain : ℝ) (price_difference : ℝ) :=
  let selling_price1 := 40 * C * (1 + gain1)
  let selling_price2 := 50 * C * (1 + gain2)
  let uniform_selling_price := n * C * (1 + uniform_gain)
  selling_price1 + selling_price2 - uniform_selling_price = price_difference

theorem find_cost_price_of_clock (C : ℝ) (h : cost_price_each_clock C 90 0.10 0.20 0.15 40) : C = 80 :=
  sorry

end ClockCost

end NUMINAMATH_GPT_find_cost_price_of_clock_l1085_108507


namespace NUMINAMATH_GPT_movie_of_the_year_condition_l1085_108516

theorem movie_of_the_year_condition (total_lists : ℕ) (fraction : ℚ) (num_lists : ℕ) 
  (h1 : total_lists = 775) (h2 : fraction = 1 / 4) (h3 : num_lists = ⌈fraction * total_lists⌉) : 
  num_lists = 194 :=
by
  -- Using the conditions given,
  -- total_lists = 775,
  -- fraction = 1 / 4,
  -- num_lists = ⌈fraction * total_lists⌉
  -- We need to show num_lists = 194.
  sorry

end NUMINAMATH_GPT_movie_of_the_year_condition_l1085_108516


namespace NUMINAMATH_GPT_angus_total_investment_l1085_108560

variable (x T : ℝ)

theorem angus_total_investment (h1 : 0.03 * x + 0.05 * 6000 = 660) (h2 : T = x + 6000) : T = 18000 :=
by
  sorry

end NUMINAMATH_GPT_angus_total_investment_l1085_108560


namespace NUMINAMATH_GPT_curve_intersects_every_plane_l1085_108581

theorem curve_intersects_every_plane (A B C D : ℝ) (h : A ≠ 0 ∨ B ≠ 0 ∨ C ≠ 0) :
  ∃ t : ℝ, A * t + B * t^3 + C * t^5 + D = 0 :=
by
  sorry

end NUMINAMATH_GPT_curve_intersects_every_plane_l1085_108581


namespace NUMINAMATH_GPT_large_circle_radius_l1085_108549

theorem large_circle_radius (s : ℝ) (r : ℝ) (R : ℝ)
  (side_length : s = 6)
  (coverage : ∀ (x y : ℝ), (x - y)^2 + (x - y)^2 = (2 * R)^2) :
  R = 3 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_large_circle_radius_l1085_108549


namespace NUMINAMATH_GPT_problem_statement_l1085_108528

theorem problem_statement (x : ℝ) (h : x^2 - 3 * x + 1 = 0) : x^2 + 1 / x^2 = 7 :=
sorry

end NUMINAMATH_GPT_problem_statement_l1085_108528


namespace NUMINAMATH_GPT_domain_range_g_l1085_108524

variable (f : ℝ → ℝ) 

noncomputable def g (x : ℝ) := 2 - f (x + 1)

theorem domain_range_g :
  (∀ x, 0 ≤ x → x ≤ 3 → 0 ≤ f x → f x ≤ 1) →
  (∀ x, -1 ≤ x → x ≤ 2) ∧ (∀ y, 1 ≤ y → y ≤ 2) :=
sorry

end NUMINAMATH_GPT_domain_range_g_l1085_108524


namespace NUMINAMATH_GPT_simplest_quadratic_radical_problem_l1085_108587

/-- The simplest quadratic radical -/
def simplest_quadratic_radical (r : ℝ) : Prop :=
  ((∀ a b : ℝ, r = a * b → b = 1 ∧ a = r) ∧ (∀ a b : ℝ, r ≠ a / b))

theorem simplest_quadratic_radical_problem :
  (simplest_quadratic_radical (Real.sqrt 6)) ∧ 
  (¬ simplest_quadratic_radical (Real.sqrt 8)) ∧ 
  (¬ simplest_quadratic_radical (Real.sqrt (1/3))) ∧ 
  (¬ simplest_quadratic_radical (Real.sqrt 4)) :=
by
  sorry

end NUMINAMATH_GPT_simplest_quadratic_radical_problem_l1085_108587


namespace NUMINAMATH_GPT_circle_center_sum_l1085_108596

theorem circle_center_sum (x y : ℝ) (h : (x - 2)^2 + (y + 1)^2 = 15) : x + y = 1 :=
sorry

end NUMINAMATH_GPT_circle_center_sum_l1085_108596


namespace NUMINAMATH_GPT_boat_speed_in_still_water_l1085_108586

-- Definitions and conditions
def Vs : ℕ := 5  -- Speed of the stream in km/hr
def distance : ℕ := 135  -- Distance traveled in km
def time : ℕ := 5  -- Time in hours

-- Statement to prove
theorem boat_speed_in_still_water : 
  ((distance = (Vb + Vs) * time) -> Vb = 22) :=
by
  sorry

end NUMINAMATH_GPT_boat_speed_in_still_water_l1085_108586


namespace NUMINAMATH_GPT_goldfish_below_surface_l1085_108592

theorem goldfish_below_surface (Toby_counts_at_surface : ℕ) (percentage_at_surface : ℝ) (total_goldfish : ℕ) (below_surface : ℕ) :
    (Toby_counts_at_surface = 15 ∧ percentage_at_surface = 0.25 ∧ Toby_counts_at_surface = percentage_at_surface * total_goldfish ∧ below_surface = total_goldfish - Toby_counts_at_surface) →
    below_surface = 45 :=
by
  sorry

end NUMINAMATH_GPT_goldfish_below_surface_l1085_108592


namespace NUMINAMATH_GPT_length_AE_l1085_108553

/-- Given points A, B, C, D, and E on a plane with distances:
  - CA = 12,
  - AB = 8,
  - BC = 4,
  - CD = 5,
  - DB = 3,
  - BE = 6,
  - ED = 3.
  Prove that AE = sqrt 113.
--/
theorem length_AE (A B C D E : ℝ × ℝ)
  (h1 : dist C A = 12)
  (h2 : dist A B = 8)
  (h3 : dist B C = 4)
  (h4 : dist C D = 5)
  (h5 : dist D B = 3)
  (h6 : dist B E = 6)
  (h7 : dist E D = 3) : 
  dist A E = Real.sqrt 113 := 
  by 
    sorry

end NUMINAMATH_GPT_length_AE_l1085_108553


namespace NUMINAMATH_GPT_x_coordinate_l1085_108544

theorem x_coordinate (x : ℝ) (y : ℝ) :
  (∃ m : ℝ, m = (0 + 6) / (4 + 8) ∧
            y + 6 = m * (x + 8) ∧
            y = 3) →
  x = 10 :=
by
  sorry

end NUMINAMATH_GPT_x_coordinate_l1085_108544


namespace NUMINAMATH_GPT_complex_div_imaginary_unit_eq_l1085_108599

theorem complex_div_imaginary_unit_eq :
  (∀ i : ℂ, i^2 = -1 → (1 / (1 + i)) = ((1 - i) / 2)) :=
by
  intro i
  intro hi
  /- The proof will be inserted here -/
  sorry

end NUMINAMATH_GPT_complex_div_imaginary_unit_eq_l1085_108599


namespace NUMINAMATH_GPT_inequality_proofs_l1085_108568

def sinSumInequality (A B C ε : ℝ) : Prop :=
  ε * (Real.sin A + Real.sin B + Real.sin C) ≤ Real.sin A * Real.sin B * Real.sin C + 1 + ε^3

def sinProductInequality (A B C ε : ℝ) : Prop :=
  (1 + ε + Real.sin A) * (1 + ε + Real.sin B) * (1 + ε + Real.sin C) ≥ 9 * ε * (Real.sin A + Real.sin B + Real.sin C)

theorem inequality_proofs (A B C ε : ℝ) (hA : 0 ≤ A ∧ A ≤ Real.pi) (hB : 0 ≤ B ∧ B ≤ Real.pi) 
  (hC : 0 ≤ C ∧ C ≤ Real.pi) (hε : ε ≥ 1) :
  sinSumInequality A B C ε ∧ sinProductInequality A B C ε :=
by
  sorry

end NUMINAMATH_GPT_inequality_proofs_l1085_108568


namespace NUMINAMATH_GPT_right_triangle_set_l1085_108588

theorem right_triangle_set:
  (1^2 + 2^2 = (Real.sqrt 5)^2) ∧
  ¬ (6^2 + 8^2 = 9^2) ∧
  ¬ ((Real.sqrt 3)^2 + (Real.sqrt 2)^2 = 5^2) ∧
  ¬ ((3^2)^2 + (4^2)^2 = (5^2)^2)  :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_set_l1085_108588


namespace NUMINAMATH_GPT_range_of_m_l1085_108555

theorem range_of_m (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 / x + 1 / y = 1) : 
  ∀ m : ℝ, (x + 2 * y > m) ↔ (m < 8) :=
by 
  sorry

end NUMINAMATH_GPT_range_of_m_l1085_108555


namespace NUMINAMATH_GPT_make_fraction_meaningful_l1085_108504

theorem make_fraction_meaningful (x : ℝ) : (x - 1) ≠ 0 ↔ x ≠ 1 :=
by
  sorry

end NUMINAMATH_GPT_make_fraction_meaningful_l1085_108504


namespace NUMINAMATH_GPT_pythagorean_triplet_l1085_108534

theorem pythagorean_triplet (k : ℕ) :
  let a := k
  let b := 2 * k - 2
  let c := 2 * k - 1
  (a * b) ^ 2 + c ^ 2 = (2 * k ^ 2 - 2 * k + 1) ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_pythagorean_triplet_l1085_108534


namespace NUMINAMATH_GPT_count_three_digit_odd_increasing_order_l1085_108567

theorem count_three_digit_odd_increasing_order : 
  ∃ n : ℕ, n = 10 ∧
  ∀ a b c : ℕ, (100 * a + 10 * b + c) % 2 = 1 ∧ a < b ∧ b < c ∧ 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ 1 ≤ c ∧ c ≤ 9 → 
    (100 * a + 10 * b + c) % 2 = 1 := 
sorry

end NUMINAMATH_GPT_count_three_digit_odd_increasing_order_l1085_108567


namespace NUMINAMATH_GPT_polynomial_complete_square_l1085_108591

theorem polynomial_complete_square :
  ∃ a h k : ℝ, (∀ x : ℝ, 4 * x^2 - 12 * x + 1 = a * (x - h)^2 + k) ∧ a + h + k = -2.5 := by
  sorry

end NUMINAMATH_GPT_polynomial_complete_square_l1085_108591


namespace NUMINAMATH_GPT_password_probability_l1085_108589

theorem password_probability :
  let even_digits := [0, 2, 4, 6, 8]
  let vowels := ['A', 'E', 'I', 'O', 'U']
  let non_zero_digits := [1, 2, 3, 4, 5, 6, 7, 8, 9]
  (even_digits.length / 10) * (vowels.length / 26) * (non_zero_digits.length / 10) = 9 / 52 :=
by
  sorry

end NUMINAMATH_GPT_password_probability_l1085_108589


namespace NUMINAMATH_GPT_loss_percentage_25_l1085_108521

variable (C S : ℝ)
variable (h : 15 * C = 20 * S)

theorem loss_percentage_25 (h : 15 * C = 20 * S) : (C - S) / C * 100 = 25 := by
  sorry

end NUMINAMATH_GPT_loss_percentage_25_l1085_108521


namespace NUMINAMATH_GPT_a_4_value_l1085_108515

-- Definitions and Theorem
variable {α : Type*} [LinearOrderedField α]

noncomputable def geometric_seq (a₀ : α) (q : α) (n : ℕ) : α := a₀ * q ^ n

theorem a_4_value (a₁ : α) (q : α) (h : geometric_seq a₁ q 1 * geometric_seq a₁ q 2 * geometric_seq a₁ q 6 = 8) : 
  geometric_seq a₁ q 3 = 2 :=
sorry

end NUMINAMATH_GPT_a_4_value_l1085_108515


namespace NUMINAMATH_GPT_average_effective_increase_correct_l1085_108552

noncomputable def effective_increase (initial_price: ℕ) (price_increase_percent: ℕ) (discount_percent: ℕ) : ℕ :=
let increased_price := initial_price + (initial_price * price_increase_percent / 100)
let final_price := increased_price - (increased_price * discount_percent / 100)
(final_price - initial_price) * 100 / initial_price

noncomputable def average_effective_increase : ℕ :=
let increase1 := effective_increase 300 10 5
let increase2 := effective_increase 450 15 7
let increase3 := effective_increase 600 20 10
(increase1 + increase2 + increase3) / 3

theorem average_effective_increase_correct :
  average_effective_increase = 6483 / 100 :=
by
  sorry

end NUMINAMATH_GPT_average_effective_increase_correct_l1085_108552


namespace NUMINAMATH_GPT_find_A_range_sinB_sinC_l1085_108557

-- Given conditions in a triangle
variable (a b c : ℝ)
variable (A B C : ℝ)
variable (h_cos_eq : a * Real.cos C + c * Real.cos A = 2 * b * Real.cos A)

-- Angle A verification
theorem find_A (h_sum_angles : A + B + C = Real.pi) : A = Real.pi / 3 :=
  sorry

-- Range of sin B + sin C
theorem range_sinB_sinC (h_sum_angles : A + B + C = Real.pi) :
  (0 < B ∧ B < 2 * Real.pi / 3) →
  Real.sin B + Real.sin C ∈ Set.Ioo (Real.sqrt 3 / 2) (Real.sqrt 3) :=
  sorry

end NUMINAMATH_GPT_find_A_range_sinB_sinC_l1085_108557


namespace NUMINAMATH_GPT_dual_cassette_recorder_price_l1085_108511

theorem dual_cassette_recorder_price :
  ∃ (x y : ℝ),
    (x - 0.05 * x = 380) ∧
    (y = x + 0.08 * x) ∧ 
    (y = 432) :=
by
  -- sorry to skip the proof.
  sorry

end NUMINAMATH_GPT_dual_cassette_recorder_price_l1085_108511

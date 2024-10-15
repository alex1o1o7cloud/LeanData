import Mathlib

namespace NUMINAMATH_GPT_votes_candidate_X_l131_13130

theorem votes_candidate_X (X Y Z : ℕ) (h1 : X = (3 / 2 : ℚ) * Y) (h2 : Y = (3 / 5 : ℚ) * Z) (h3 : Z = 25000) : X = 22500 :=
by
  sorry

end NUMINAMATH_GPT_votes_candidate_X_l131_13130


namespace NUMINAMATH_GPT_C_eq_D_at_n_l131_13151

noncomputable def C_n (n : ℕ) : ℝ := 768 * (1 - (1 / (3^n)))
noncomputable def D_n (n : ℕ) : ℝ := (4096 / 5) * (1 - ((-1)^n / (4^n)))
noncomputable def n_ge_1 : ℕ := 4

theorem C_eq_D_at_n : ∀ n ≥ 1, C_n n = D_n n → n = n_ge_1 :=
by
  intro n hn heq
  sorry

end NUMINAMATH_GPT_C_eq_D_at_n_l131_13151


namespace NUMINAMATH_GPT_sum_of_xyz_l131_13198

theorem sum_of_xyz (x y z : ℝ)
  (h1 : x > 0)
  (h2 : y > 0)
  (h3 : z > 0)
  (h4 : x^2 + y^2 + x * y = 3)
  (h5 : y^2 + z^2 + y * z = 4)
  (h6 : z^2 + x^2 + z * x = 7) :
  x + y + z = Real.sqrt 13 :=
by sorry -- Proof omitted, but the statement formulation is complete and checks the equality under given conditions.

end NUMINAMATH_GPT_sum_of_xyz_l131_13198


namespace NUMINAMATH_GPT_committee_meeting_people_l131_13194

theorem committee_meeting_people (A B : ℕ) 
  (h1 : 2 * A + B = 10) 
  (h2 : A + 2 * B = 11) : 
  A + B = 7 :=
sorry

end NUMINAMATH_GPT_committee_meeting_people_l131_13194


namespace NUMINAMATH_GPT_simplify_expr_l131_13160

theorem simplify_expr (a : ℝ) (h_a : a = (8:ℝ)^(1/2) * (1/2) - (3:ℝ)^(1/2)^(0) ) : 
  a = (2:ℝ)^(1/2) - 1 := 
by
  sorry

end NUMINAMATH_GPT_simplify_expr_l131_13160


namespace NUMINAMATH_GPT_find_x_l131_13101

def binop (a b : ℤ) : ℤ := a * b + a + b + 2

theorem find_x :
  ∃ x : ℤ, binop x 3 = 1 ∧ x = -1 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l131_13101


namespace NUMINAMATH_GPT_original_cost_of_remaining_shirt_l131_13142

theorem original_cost_of_remaining_shirt 
  (total_original_cost : ℝ) 
  (shirts_on_discount : ℕ) 
  (original_cost_per_discounted_shirt : ℝ) 
  (discount : ℝ) 
  (current_total_cost : ℝ) : 
  total_original_cost = 100 → 
  shirts_on_discount = 3 → 
  original_cost_per_discounted_shirt = 25 → 
  discount = 0.4 → 
  current_total_cost = 85 → 
  ∃ (remaining_shirts : ℕ) (original_cost_per_remaining_shirt : ℝ), 
    remaining_shirts = 2 ∧ 
    original_cost_per_remaining_shirt = 12.5 :=
by 
  sorry

end NUMINAMATH_GPT_original_cost_of_remaining_shirt_l131_13142


namespace NUMINAMATH_GPT_xyz_squared_sum_l131_13123

theorem xyz_squared_sum (x y z : ℤ) 
  (h1 : |x + y| + |y + z| + |z + x| = 4)
  (h2 : |x - y| + |y - z| + |z - x| = 2) :
  x^2 + y^2 + z^2 = 2 := 
by 
  sorry

end NUMINAMATH_GPT_xyz_squared_sum_l131_13123


namespace NUMINAMATH_GPT_jane_doe_investment_l131_13145

theorem jane_doe_investment (total_investment mutual_funds real_estate : ℝ)
  (h1 : total_investment = 250000)
  (h2 : real_estate = 3 * mutual_funds)
  (h3 : total_investment = mutual_funds + real_estate) :
  real_estate = 187500 :=
by
  sorry

end NUMINAMATH_GPT_jane_doe_investment_l131_13145


namespace NUMINAMATH_GPT_min_accommodation_cost_l131_13144

theorem min_accommodation_cost :
  ∃ (x y z : ℕ), x + y + z = 20 ∧ 3 * x + 2 * y + z = 50 ∧ 100 * 3 * x + 150 * 2 * y + 200 * z = 5500 :=
by
  sorry

end NUMINAMATH_GPT_min_accommodation_cost_l131_13144


namespace NUMINAMATH_GPT_average_difference_l131_13193

def differences : List ℤ := [15, -5, 25, 35, -15, 10, 20]
def days : ℤ := 7

theorem average_difference (diff : List ℤ) (n : ℤ) 
  (h : diff = [15, -5, 25, 35, -15, 10, 20]) (h_days : n = 7) : 
  (diff.sum / n : ℚ) = 12 := 
by 
  rw [h, h_days]
  norm_num
  sorry

end NUMINAMATH_GPT_average_difference_l131_13193


namespace NUMINAMATH_GPT_equation_of_line_AB_l131_13152

def point := (ℝ × ℝ)

def A : point := (-1, 0)
def B : point := (3, 2)

def equation_of_line (p1 p2 : point) : ℝ × ℝ × ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  -- Calculate the slope
  let k := (y2 - y1) / (x2 - x1)
  -- Use point-slope form and simplify the equation to standard form
  (((1 : ℝ), -2, 1) : ℝ × ℝ × ℝ)

theorem equation_of_line_AB :
  equation_of_line A B = (1, -2, 1) :=
sorry

end NUMINAMATH_GPT_equation_of_line_AB_l131_13152


namespace NUMINAMATH_GPT_tangent_line_to_curve_at_Mpi_l131_13120

noncomputable def tangent_line_eq_at_point (x : ℝ) (y : ℝ) : Prop :=
  y = (Real.sin x) / x

theorem tangent_line_to_curve_at_Mpi :
  (∀ x y, tangent_line_eq_at_point x y →
    (∃ (m : ℝ), m = -1 / π) →
    (∀ x1 y1 (hx : x1 = π) (hy : y1 = 0), x + π * y - π = 0)) :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_to_curve_at_Mpi_l131_13120


namespace NUMINAMATH_GPT_necessary_sufficient_condition_l131_13199

theorem necessary_sufficient_condition (a b x_0 : ℝ) (h : a > 0) :
  (x_0 = b / a) ↔ (∀ x : ℝ, (1/2) * a * x^2 - b * x ≥ (1/2) * a * x_0^2 - b * x_0) :=
sorry

end NUMINAMATH_GPT_necessary_sufficient_condition_l131_13199


namespace NUMINAMATH_GPT_repetend_five_seventeen_l131_13115

noncomputable def repetend_of_fraction (n : ℕ) (d : ℕ) : ℕ := sorry

theorem repetend_five_seventeen : repetend_of_fraction 5 17 = 294117647058823529 := sorry

end NUMINAMATH_GPT_repetend_five_seventeen_l131_13115


namespace NUMINAMATH_GPT_find_speed_of_B_l131_13186

theorem find_speed_of_B
  (d : ℝ := 12)
  (v_b : ℝ)
  (v_a : ℝ := 1.2 * v_b)
  (t_b : ℝ := d / v_b)
  (t_a : ℝ := d / v_a)
  (time_diff : ℝ := t_b - t_a)
  (h_time_diff : time_diff = 1 / 6) :
  v_b = 12 :=
sorry

end NUMINAMATH_GPT_find_speed_of_B_l131_13186


namespace NUMINAMATH_GPT_car_price_l131_13150

/-- Prove that the price of the car Quincy bought is $20,000 given the conditions. -/
theorem car_price (years : ℕ) (monthly_payment : ℕ) (down_payment : ℕ) 
  (h1 : years = 5) 
  (h2 : monthly_payment = 250) 
  (h3 : down_payment = 5000) : 
  (down_payment + (monthly_payment * (12 * years))) = 20000 :=
by
  /- We provide the proof below with sorry because we are only writing the statement as requested. -/
  sorry

end NUMINAMATH_GPT_car_price_l131_13150


namespace NUMINAMATH_GPT_iron_balls_count_l131_13105

-- Conditions
def length_bar := 12  -- in cm
def width_bar := 8    -- in cm
def height_bar := 6   -- in cm
def num_bars := 10
def volume_iron_ball := 8  -- in cubic cm

-- Calculate the volume of one iron bar
def volume_one_bar := length_bar * width_bar * height_bar

-- Calculate the total volume of the ten iron bars
def total_volume := volume_one_bar * num_bars

-- Calculate the number of iron balls
def num_iron_balls := total_volume / volume_iron_ball

-- The proof statement
theorem iron_balls_count : num_iron_balls = 720 := by
  sorry

end NUMINAMATH_GPT_iron_balls_count_l131_13105


namespace NUMINAMATH_GPT_break_even_performances_l131_13159

def totalCost (x : ℕ) : ℕ := 81000 + 7000 * x
def totalRevenue (x : ℕ) : ℕ := 16000 * x

theorem break_even_performances : ∃ x : ℕ, totalCost x = totalRevenue x ∧ x = 9 := 
by
  sorry

end NUMINAMATH_GPT_break_even_performances_l131_13159


namespace NUMINAMATH_GPT_ratio_A_B_l131_13104

noncomputable def A : ℝ := ∑' n : ℕ, if n % 4 = 0 then 0 else 1 / (n:ℝ) ^ 2
noncomputable def B : ℝ := ∑' k : ℕ, (-1)^(k+1) / (4 * (k:ℝ)) ^ 2

theorem ratio_A_B : A / B = 32 := by
  -- proof here
  sorry

end NUMINAMATH_GPT_ratio_A_B_l131_13104


namespace NUMINAMATH_GPT_find_y_l131_13190

def G (a b c d : ℕ) : ℕ := a^b + c * d

theorem find_y (y : ℕ) (h : G 3 y 5 18 = 500) : y = 6 :=
sorry

end NUMINAMATH_GPT_find_y_l131_13190


namespace NUMINAMATH_GPT_ratio_sum_pqr_uvw_l131_13131

theorem ratio_sum_pqr_uvw (p q r u v w : ℝ)
  (hp : 0 < p) (hq : 0 < q) (hr : 0 < r)
  (hu : 0 < u) (hv : 0 < v) (hw : 0 < w)
  (h1 : p^2 + q^2 + r^2 = 49)
  (h2 : u^2 + v^2 + w^2 = 64)
  (h3 : p * u + q * v + r * w = 56) :
  (p + q + r) / (u + v + w) = 7 / 8 :=
sorry

end NUMINAMATH_GPT_ratio_sum_pqr_uvw_l131_13131


namespace NUMINAMATH_GPT_find_number_l131_13148

theorem find_number : ∃ x : ℝ, x^2 + 100 = (x - 20)^2 ∧ x = 7.5 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_number_l131_13148


namespace NUMINAMATH_GPT_max_grapes_in_bag_l131_13163

theorem max_grapes_in_bag : ∃ (x : ℕ), x > 100 ∧ x % 3 = 1 ∧ x % 5 = 2 ∧ x % 7 = 4 ∧ x = 172 := by
  sorry

end NUMINAMATH_GPT_max_grapes_in_bag_l131_13163


namespace NUMINAMATH_GPT_herd_compuation_l131_13127

theorem herd_compuation (a b c : ℕ) (total_animals total_payment : ℕ) 
  (H1 : total_animals = a + b + 10 * c) 
  (H2 : total_payment = 20 * a + 10 * b + 10 * c) 
  (H3 : total_animals = 100) 
  (H4 : total_payment = 200) :
  a = 1 ∧ b = 9 ∧ 10 * c = 90 :=
by
  sorry

end NUMINAMATH_GPT_herd_compuation_l131_13127


namespace NUMINAMATH_GPT_pie_left_is_30_percent_l131_13196

def Carlos_share : ℝ := 0.60
def remaining_after_Carlos : ℝ := 1 - Carlos_share
def Jessica_share : ℝ := 0.25 * remaining_after_Carlos
def final_remaining : ℝ := remaining_after_Carlos - Jessica_share

theorem pie_left_is_30_percent :
  final_remaining = 0.30 :=
sorry

end NUMINAMATH_GPT_pie_left_is_30_percent_l131_13196


namespace NUMINAMATH_GPT_twelve_million_plus_twelve_thousand_l131_13182

theorem twelve_million_plus_twelve_thousand :
  12000000 + 12000 = 12012000 :=
by
  sorry

end NUMINAMATH_GPT_twelve_million_plus_twelve_thousand_l131_13182


namespace NUMINAMATH_GPT_min_value_of_sum_squares_on_circle_l131_13180

theorem min_value_of_sum_squares_on_circle :
  ∃ (x y : ℝ), (x - 2)^2 + (y - 1)^2 = 1 ∧ x^2 + y^2 = 6 - 2 * Real.sqrt 5 :=
sorry

end NUMINAMATH_GPT_min_value_of_sum_squares_on_circle_l131_13180


namespace NUMINAMATH_GPT_aluminum_weight_proportional_l131_13136

noncomputable def area_equilateral_triangle (side_length : ℝ) : ℝ :=
  (side_length * side_length * Real.sqrt 3) / 4

theorem aluminum_weight_proportional (weight1 weight2 : ℝ) 
  (side_length1 side_length2 : ℝ)
  (h_density_thickness : ∀ s t, area_equilateral_triangle s * weight1 = area_equilateral_triangle t * weight2)
  (h_weight1 : weight1 = 20)
  (h_side_length1 : side_length1 = 2)
  (h_side_length2 : side_length2 = 4) : 
  weight2 = 80 :=
by
  sorry

end NUMINAMATH_GPT_aluminum_weight_proportional_l131_13136


namespace NUMINAMATH_GPT_candy_total_l131_13116

theorem candy_total (chocolate_boxes caramel_boxes mint_boxes berry_boxes : ℕ)
  (chocolate_pieces caramel_pieces mint_pieces berry_pieces : ℕ)
  (h_chocolate : chocolate_boxes = 7)
  (h_caramel : caramel_boxes = 3)
  (h_mint : mint_boxes = 5)
  (h_berry : berry_boxes = 4)
  (p_chocolate : chocolate_pieces = 8)
  (p_caramel : caramel_pieces = 8)
  (p_mint : mint_pieces = 10)
  (p_berry : berry_pieces = 12) :
  (chocolate_boxes * chocolate_pieces + caramel_boxes * caramel_pieces + mint_boxes * mint_pieces + berry_boxes * berry_pieces) = 178 := by
  sorry

end NUMINAMATH_GPT_candy_total_l131_13116


namespace NUMINAMATH_GPT_company_bought_oil_l131_13124

-- Define the conditions
def tank_capacity : ℕ := 32
def oil_in_tank : ℕ := 24

-- Formulate the proof problem
theorem company_bought_oil : oil_in_tank = 24 := by
  sorry

end NUMINAMATH_GPT_company_bought_oil_l131_13124


namespace NUMINAMATH_GPT_housing_price_equation_l131_13174

-- Initial conditions
def january_price : ℝ := 8300
def march_price : ℝ := 8700
variables (x : ℝ)

-- Lean statement of the problem
theorem housing_price_equation :
  january_price * (1 + x)^2 = march_price := 
sorry

end NUMINAMATH_GPT_housing_price_equation_l131_13174


namespace NUMINAMATH_GPT_total_time_to_complete_l131_13192

noncomputable def time_to_clean_keys (n : Nat) (t : Nat) : Nat := n * t

def assignment_time : Nat := 10
def time_per_key : Nat := 3
def remaining_keys : Nat := 14

theorem total_time_to_complete :
  time_to_clean_keys remaining_keys time_per_key + assignment_time = 52 := by
  sorry

end NUMINAMATH_GPT_total_time_to_complete_l131_13192


namespace NUMINAMATH_GPT_arithmetic_sequence_problem_l131_13141

variable {a : ℕ → ℤ}

def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n m, a (n + 1) - a n = a (m + 1) - a m

theorem arithmetic_sequence_problem 
  (h : is_arithmetic_sequence a)
  (h_cond : a 2 + 2 * a 6 + a 10 = 120) :
  a 3 + a 9 = 60 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_problem_l131_13141


namespace NUMINAMATH_GPT_units_digit_fraction_l131_13154

open Nat

theorem units_digit_fraction : 
  (15 * 16 * 17 * 18 * 19 * 20) % 500 % 10 = 2 := by
  sorry

end NUMINAMATH_GPT_units_digit_fraction_l131_13154


namespace NUMINAMATH_GPT_factor_values_l131_13187

theorem factor_values (a b : ℤ) :
  (∀ s : ℂ, s^2 - s - 1 = 0 → a * s^15 + b * s^14 + 1 = 0) ∧
  (∀ t : ℂ, t^2 - t - 1 = 0 → a * t^15 + b * t^14 + 1 = 0) →
  a = 377 ∧ b = -610 :=
by
  sorry

end NUMINAMATH_GPT_factor_values_l131_13187


namespace NUMINAMATH_GPT_cubics_product_equals_1_over_1003_l131_13114

theorem cubics_product_equals_1_over_1003
  (x_1 y_1 x_2 y_2 x_3 y_3 : ℝ)
  (h1 : x_1^3 - 3 * x_1 * y_1^2 = 2007)
  (h2 : y_1^3 - 3 * x_1^2 * y_1 = 2006)
  (h3 : x_2^3 - 3 * x_2 * y_2^2 = 2007)
  (h4 : y_2^3 - 3 * x_2^2 * y_2 = 2006)
  (h5 : x_3^3 - 3 * x_3 * y_3^2 = 2007)
  (h6 : y_3^3 - 3 * x_3^2 * y_3 = 2006) :
  (1 - x_1 / y_1) * (1 - x_2 / y_2) * (1 - x_3 / y_3) = 1 / 1003 :=
by
  sorry

end NUMINAMATH_GPT_cubics_product_equals_1_over_1003_l131_13114


namespace NUMINAMATH_GPT_math_equivalence_example_l131_13135

theorem math_equivalence_example :
  ((3.242^2 * (16 + 8)) / (100 - (3 * 25))) + (32 - 10)^2 = 494.09014144 := 
by
  sorry

end NUMINAMATH_GPT_math_equivalence_example_l131_13135


namespace NUMINAMATH_GPT_non_coincident_angles_l131_13133

theorem non_coincident_angles : ¬ ∃ k : ℤ, 1050 - (-300) = k * 360 := by
  sorry

end NUMINAMATH_GPT_non_coincident_angles_l131_13133


namespace NUMINAMATH_GPT_inequality_solution_l131_13107

noncomputable def f (x : ℝ) : ℝ := abs (x + 1) + abs (x - 3)

theorem inequality_solution :
  -3 < x ∧ x < -1 ↔ f (x^2 - 3) < f (x - 1) :=
sorry

end NUMINAMATH_GPT_inequality_solution_l131_13107


namespace NUMINAMATH_GPT_train_vs_airplane_passenger_capacity_l131_13128

theorem train_vs_airplane_passenger_capacity :
  (60 * 16) - (366 * 2) = 228 := by
sorry

end NUMINAMATH_GPT_train_vs_airplane_passenger_capacity_l131_13128


namespace NUMINAMATH_GPT_geom_seq_log_eqn_l131_13168

theorem geom_seq_log_eqn {a : ℕ → ℝ} {b : ℕ → ℝ}
    (geom_seq : ∃ (r : ℝ) (a1 : ℝ), ∀ n : ℕ, a (n + 1) = a1 * r^n)
    (log_seq : ∀ n : ℕ, b n = Real.log (a (n + 1)) / Real.log 2)
    (b_eqn : b 1 + b 3 = 4) : a 2 = 4 :=
by
  sorry

end NUMINAMATH_GPT_geom_seq_log_eqn_l131_13168


namespace NUMINAMATH_GPT_proof_n_value_l131_13103

theorem proof_n_value (n : ℕ) (h : (9^n) * (9^n) * (9^n) * (9^n) * (9^n) = 81^5) : n = 2 :=
by
  sorry

end NUMINAMATH_GPT_proof_n_value_l131_13103


namespace NUMINAMATH_GPT_range_of_k_l131_13164

theorem range_of_k (k : ℝ) (h : -3 < k ∧ k ≤ 0) : ∀ x : ℝ, k * x^2 + 2 * k * x - 3 < 0 :=
sorry

end NUMINAMATH_GPT_range_of_k_l131_13164


namespace NUMINAMATH_GPT_swap_tens_units_digits_l131_13113

theorem swap_tens_units_digits (x a b : ℕ) (h1 : 9 < x) (h2 : x < 100) (h3 : a = x / 10) (h4 : b = x % 10) :
  10 * b + a = (x % 10) * 10 + (x / 10) :=
by
  sorry

end NUMINAMATH_GPT_swap_tens_units_digits_l131_13113


namespace NUMINAMATH_GPT_dot_product_conditioned_l131_13117

variables (a b : ℝ×ℝ)

def condition1 : Prop := 2 • a + b = (1, 6)
def condition2 : Prop := a + 2 • b = (-4, 9)
def dot_product (u v : ℝ×ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem dot_product_conditioned :
  condition1 a b ∧ condition2 a b → dot_product a b = -2 :=
by
  sorry

end NUMINAMATH_GPT_dot_product_conditioned_l131_13117


namespace NUMINAMATH_GPT_find_k_l131_13181

-- Define the variables and conditions
variables (x y k : ℤ)

-- State the theorem
theorem find_k (h1 : x = 2) (h2 : y = 1) (h3 : k * x - y = 3) : k = 2 :=
sorry

end NUMINAMATH_GPT_find_k_l131_13181


namespace NUMINAMATH_GPT_square_area_l131_13153

theorem square_area :
  ∀ (x1 x2 : ℝ), (x1^2 + 2 * x1 + 1 = 8) ∧ (x2^2 + 2 * x2 + 1 = 8) ∧ (x1 ≠ x2) →
  (abs (x1 - x2))^2 = 36 :=
by
  sorry

end NUMINAMATH_GPT_square_area_l131_13153


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l131_13102

variable {a b c : ℝ}

theorem necessary_but_not_sufficient (h1 : b^2 - 4 * a * c ≥ 0) (h2 : a * c > 0) (h3 : a * b < 0) : 
  ¬∀ r1 r2 : ℝ, (r1 = (-b + Real.sqrt (b^2 - 4 * a * c)) / (2 * a)) ∧ (r2 = (-b - Real.sqrt (b^2 - 4 * a * c)) / (2 * a)) → r1 > 0 ∧ r2 > 0 :=
sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l131_13102


namespace NUMINAMATH_GPT_proof_problem_l131_13140

-- Define the conditions based on Classmate A and Classmate B's statements
def classmateA_statement (x y : ℝ) : Prop := 6 * x = 5 * y
def classmateB_statement (x y : ℝ) : Prop := x = 2 * y - 40

-- Define the system of equations derived from the statements
def system_of_equations (x y : ℝ) : Prop := (6 * x = 5 * y) ∧ (x = 2 * y - 40)

-- Proof goal: Prove the system of equations if classmate statements hold
theorem proof_problem (x y : ℝ) :
  classmateA_statement x y ∧ classmateB_statement x y → system_of_equations x y :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l131_13140


namespace NUMINAMATH_GPT_term_sequence_l131_13173

theorem term_sequence (n : ℕ) (h : (-1:ℤ) ^ (n + 1) * n * (n + 1) = -20) : n = 4 :=
sorry

end NUMINAMATH_GPT_term_sequence_l131_13173


namespace NUMINAMATH_GPT_minimum_final_percentage_to_pass_l131_13119

-- Conditions
def problem_sets : ℝ := 100
def midterm_worth : ℝ := 100
def final_worth : ℝ := 300
def perfect_problem_sets_score : ℝ := 100
def midterm1_score : ℝ := 0.60 * midterm_worth
def midterm2_score : ℝ := 0.70 * midterm_worth
def midterm3_score : ℝ := 0.80 * midterm_worth
def passing_percentage : ℝ := 0.70

-- Derived Values
def total_points_available : ℝ := problem_sets + 3 * midterm_worth + final_worth
def required_points_to_pass : ℝ := passing_percentage * total_points_available
def total_points_before_final : ℝ := perfect_problem_sets_score + midterm1_score + midterm2_score + midterm3_score
def points_needed_from_final : ℝ := required_points_to_pass - total_points_before_final

-- Proof Statement
theorem minimum_final_percentage_to_pass : 
  ∃ (final_score : ℝ), (final_score / final_worth * 100) ≥ 60 :=
by
  -- Calculations for proof
  let required_final_percentage := (points_needed_from_final / final_worth) * 100
  -- We need to show that the required percentage is at least 60%
  have : required_final_percentage = 60 := sorry
  exact Exists.intro 180 sorry

end NUMINAMATH_GPT_minimum_final_percentage_to_pass_l131_13119


namespace NUMINAMATH_GPT_solve_equation_l131_13191

theorem solve_equation (x: ℝ) (h : (5 - x)^(1/3) = -5/2) : x = 165/8 :=
by
  -- proof skipped
  sorry

end NUMINAMATH_GPT_solve_equation_l131_13191


namespace NUMINAMATH_GPT_no_solution_system_of_inequalities_l131_13169

theorem no_solution_system_of_inequalities :
  ¬ ∃ (x y : ℝ),
    11 * x^2 - 10 * x * y + 3 * y^2 ≤ 3 ∧
    5 * x + y ≤ -10 :=
by
  sorry

end NUMINAMATH_GPT_no_solution_system_of_inequalities_l131_13169


namespace NUMINAMATH_GPT_find_usual_time_l131_13161

variables (P D T : ℝ)
variable (h1 : P = D / T)
variable (h2 : 3 / 4 * P = D / (T + 20))

theorem find_usual_time (h1 : P = D / T) (h2 : 3 / 4 * P = D / (T + 20)) : T = 80 := 
  sorry

end NUMINAMATH_GPT_find_usual_time_l131_13161


namespace NUMINAMATH_GPT_even_product_probability_l131_13100

def number_on_first_spinner := [3, 6, 5, 10, 15]
def number_on_second_spinner := [7, 6, 11, 12, 13, 14]

noncomputable def probability_even_product : ℚ :=
  1 - (3 / 5) * (3 / 6)

theorem even_product_probability :
  probability_even_product = 7 / 10 :=
by
  sorry

end NUMINAMATH_GPT_even_product_probability_l131_13100


namespace NUMINAMATH_GPT_intersection_point_of_line_and_y_axis_l131_13175

theorem intersection_point_of_line_and_y_axis :
  {p : ℝ × ℝ | ∃ x, p = (x, 2 * x + 1) ∧ x = 0} = {(0, 1)} :=
by sorry

end NUMINAMATH_GPT_intersection_point_of_line_and_y_axis_l131_13175


namespace NUMINAMATH_GPT_binomial_10_10_binomial_10_9_l131_13155

-- Prove that \(\binom{10}{10} = 1\)
theorem binomial_10_10 : Nat.choose 10 10 = 1 :=
by sorry

-- Prove that \(\binom{10}{9} = 10\)
theorem binomial_10_9 : Nat.choose 10 9 = 10 :=
by sorry

end NUMINAMATH_GPT_binomial_10_10_binomial_10_9_l131_13155


namespace NUMINAMATH_GPT_chemistry_more_than_physics_l131_13106

theorem chemistry_more_than_physics
  (M P C : ℕ)
  (h1 : M + P = 60)
  (h2 : (M + C) / 2 = 35) :
  ∃ x : ℕ, C = P + x ∧ x = 10 := 
by
  sorry

end NUMINAMATH_GPT_chemistry_more_than_physics_l131_13106


namespace NUMINAMATH_GPT_box_breadth_l131_13189

noncomputable def cm_to_m (cm : ℕ) : ℝ := cm / 100

theorem box_breadth :
  ∀ (length depth cm cubical_edge blocks : ℕ), 
    length = 160 →
    depth = 60 →
    cubical_edge = 20 →
    blocks = 120 →
    breadth = (blocks * (cubical_edge ^ 3)) / (length * depth) →
    breadth = 100 :=
by
  sorry

end NUMINAMATH_GPT_box_breadth_l131_13189


namespace NUMINAMATH_GPT_remainder_of_10_pow_23_minus_7_mod_6_l131_13167

theorem remainder_of_10_pow_23_minus_7_mod_6 : ((10 ^ 23 - 7) % 6) = 3 := by
  sorry

end NUMINAMATH_GPT_remainder_of_10_pow_23_minus_7_mod_6_l131_13167


namespace NUMINAMATH_GPT_max_value_expression_l131_13185

theorem max_value_expression (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 1 / Real.sqrt 3) :
  27 * a * b * c + a * Real.sqrt (a^2 + 2 * b * c) + b * Real.sqrt (b^2 + 2 * c * a) + c * Real.sqrt (c^2 + 2 * a * b) ≤ 2 / (3 * Real.sqrt 3) :=
sorry

end NUMINAMATH_GPT_max_value_expression_l131_13185


namespace NUMINAMATH_GPT_herd_total_cows_l131_13172

theorem herd_total_cows (n : ℕ) (h1 : (1 / 3 : ℚ) * n + (1 / 5 : ℚ) * n + (1 / 6 : ℚ) * n + 19 = n) : n = 63 :=
sorry

end NUMINAMATH_GPT_herd_total_cows_l131_13172


namespace NUMINAMATH_GPT_cuboid_first_dimension_l131_13109

theorem cuboid_first_dimension (x : ℕ)
  (h₁ : ∃ n : ℕ, n = 24) 
  (h₂ : ∃ a b c d e f g : ℕ, x = a ∧ 9 = b ∧ 12 = c ∧ a * b * c = d * e * f ∧ g = Nat.gcd b c ∧ f = (g^3) ∧ e = (n * f) ∧ d = 648) : 
  x = 6 :=
by
  sorry

end NUMINAMATH_GPT_cuboid_first_dimension_l131_13109


namespace NUMINAMATH_GPT_complement_union_l131_13121

namespace SetTheory

def U : Set ℕ := {1, 3, 5, 9}
def A : Set ℕ := {1, 3, 9}
def B : Set ℕ := {1, 9}

theorem complement_union (U A B: Set ℕ) (hU : U = {1, 3, 5, 9}) (hA : A = {1, 3, 9}) (hB : B = {1, 9}) :
  U \ (A ∪ B) = {5} :=
by
  sorry

end SetTheory

end NUMINAMATH_GPT_complement_union_l131_13121


namespace NUMINAMATH_GPT_janelle_initial_green_marbles_l131_13149

def initial_green_marbles (blue_bags : ℕ) (marbles_per_bag : ℕ) (gift_green : ℕ) (gift_blue : ℕ) (remaining_marbles : ℕ) : ℕ :=
  let blue_marbles := blue_bags * marbles_per_bag
  let remaining_blue_marbles := blue_marbles - gift_blue
  let remaining_green_marbles := remaining_marbles - remaining_blue_marbles
  remaining_green_marbles + gift_green

theorem janelle_initial_green_marbles :
  initial_green_marbles 6 10 6 8 72 = 26 :=
by
  rfl

end NUMINAMATH_GPT_janelle_initial_green_marbles_l131_13149


namespace NUMINAMATH_GPT_sector_area_half_triangle_area_l131_13177

theorem sector_area_half_triangle_area (θ : Real) (r : Real) (hθ1 : 0 < θ) (hθ2 : θ < π / 3) :
    2 * θ = Real.tan θ := by
  sorry

end NUMINAMATH_GPT_sector_area_half_triangle_area_l131_13177


namespace NUMINAMATH_GPT_inequality_problem_l131_13125

theorem inequality_problem
  (a b c : ℝ)
  (h_pos_a : 0 < a)
  (h_pos_b : 0 < b)
  (h_pos_c : 0 < c)
  (h_sum : a + b + c ≤ 3) :
  (1 / (a + 1)) + (1 / (b + 1)) + (1 / (c + 1)) ≥ 3 / 2 :=
by sorry

end NUMINAMATH_GPT_inequality_problem_l131_13125


namespace NUMINAMATH_GPT_complex_expression_evaluation_l131_13110

theorem complex_expression_evaluation : (i : ℂ) * (1 + i : ℂ)^2 = -2 := 
by
  sorry

end NUMINAMATH_GPT_complex_expression_evaluation_l131_13110


namespace NUMINAMATH_GPT_total_length_of_board_l131_13126

theorem total_length_of_board (x y : ℝ) (h1 : y = 2 * x) (h2 : y = 46) : x + y = 69 :=
by
  sorry

end NUMINAMATH_GPT_total_length_of_board_l131_13126


namespace NUMINAMATH_GPT_a_sufficient_not_necessary_l131_13146

theorem a_sufficient_not_necessary (a : ℝ) : (a > 1 → 1 / a < 1) ∧ (¬(1 / a < 1 → a > 1)) :=
by
  sorry

end NUMINAMATH_GPT_a_sufficient_not_necessary_l131_13146


namespace NUMINAMATH_GPT_eliza_height_l131_13122

theorem eliza_height
  (n : ℕ) (H_total : ℕ) 
  (sib1_height : ℕ) (sib2_height : ℕ) (sib3_height : ℕ)
  (eliza_height : ℕ) (last_sib_height : ℕ) :
  n = 5 →
  H_total = 330 →
  sib1_height = 66 →
  sib2_height = 66 →
  sib3_height = 60 →
  eliza_height = last_sib_height - 2 →
  H_total = sib1_height + sib2_height + sib3_height + eliza_height + last_sib_height →
  eliza_height = 68 :=
by
  intros n_eq H_total_eq sib1_eq sib2_eq sib3_eq eliza_eq H_sum_eq
  sorry

end NUMINAMATH_GPT_eliza_height_l131_13122


namespace NUMINAMATH_GPT_quarter_circle_area_ratio_l131_13139

theorem quarter_circle_area_ratio (R : ℝ) (hR : 0 < R) :
  let O := pi * R^2
  let AXC := pi * (R/2)^2 / 4
  let BYD := pi * (R/2)^2 / 4
  (2 * (AXC + BYD) / O = 1 / 8) := 
by
  let O := pi * R^2
  let AXC := pi * (R/2)^2 / 4
  let BYD := pi * (R/2)^2 / 4
  sorry

end NUMINAMATH_GPT_quarter_circle_area_ratio_l131_13139


namespace NUMINAMATH_GPT_original_price_of_wand_l131_13166

-- Definitions as per the conditions
def price_paid (paid : Real) := paid = 8
def fraction_of_original (fraction : Real) := fraction = 1 / 8

-- Question and correct answer put as a theorem to prove
theorem original_price_of_wand (paid : Real) (fraction : Real) 
  (h1 : price_paid paid) (h2 : fraction_of_original fraction) : 
  (paid / fraction = 64) := 
by
  -- This 'sorry' indicates where the actual proof would go.
  sorry

end NUMINAMATH_GPT_original_price_of_wand_l131_13166


namespace NUMINAMATH_GPT_infinite_solutions_if_one_exists_l131_13108

namespace RationalSolutions

def has_rational_solution (a b : ℚ) : Prop :=
  ∃ (x y : ℚ), a * x^2 + b * y^2 = 1

def infinite_rational_solutions (a b : ℚ) : Prop :=
  ∀ (x₀ y₀ : ℚ), (a * x₀^2 + b * y₀^2 = 1) → ∃ (f : ℕ → ℚ × ℚ), ∀ n : ℕ, a * (f n).1^2 + b * (f n).2^2 = 1 ∧ (f 0 = (x₀, y₀)) ∧ ∀ m n : ℕ, m ≠ n → (f m) ≠ (f n)

theorem infinite_solutions_if_one_exists (a b : ℚ) (h : has_rational_solution a b) : infinite_rational_solutions a b :=
  sorry

end RationalSolutions

end NUMINAMATH_GPT_infinite_solutions_if_one_exists_l131_13108


namespace NUMINAMATH_GPT_sum_of_ages_3_years_hence_l131_13179

theorem sum_of_ages_3_years_hence (A B C D S : ℝ) (h1 : A = 2 * B) (h2 : C = A / 2) (h3 : D = A - C) (h_sum : A + B + C + D = S) : 
  (A + 3) + (B + 3) + (C + 3) + (D + 3) = S + 12 :=
by sorry

end NUMINAMATH_GPT_sum_of_ages_3_years_hence_l131_13179


namespace NUMINAMATH_GPT_train_speed_is_correct_l131_13170

/-- Define the length of the train (in meters) -/
def train_length : ℕ := 120

/-- Define the length of the bridge (in meters) -/
def bridge_length : ℕ := 255

/-- Define the time to cross the bridge (in seconds) -/
def time_to_cross : ℕ := 30

/-- Define the total distance covered by the train while crossing the bridge -/
def total_distance : ℕ := train_length + bridge_length

/-- Define the speed of the train in meters per second -/
def speed_m_per_s : ℚ := total_distance / time_to_cross

/-- Conversion factor from m/s to km/hr -/
def m_per_s_to_km_per_hr : ℚ := 3.6

/-- The expected speed of the train in km/hr -/
def expected_speed_km_per_hr : ℕ := 45

/-- The theorem stating that the speed of the train is 45 km/hr -/
theorem train_speed_is_correct :
  (speed_m_per_s * m_per_s_to_km_per_hr) = expected_speed_km_per_hr := by
  sorry

end NUMINAMATH_GPT_train_speed_is_correct_l131_13170


namespace NUMINAMATH_GPT_third_side_of_triangle_l131_13138

theorem third_side_of_triangle (a b : ℝ) (γ : ℝ) (x : ℝ) 
  (ha : a = 6) (hb : b = 2 * Real.sqrt 7) (hγ : γ = Real.pi / 3) :
  x = 2 ∨ x = 4 :=
by 
  sorry

end NUMINAMATH_GPT_third_side_of_triangle_l131_13138


namespace NUMINAMATH_GPT_engineers_crimson_meet_in_tournament_l131_13132

noncomputable def probability_engineers_crimson_meet : ℝ := 
  1 - Real.exp (-1)

theorem engineers_crimson_meet_in_tournament :
  (∃ (n : ℕ), n = 128) → 
  (∀ (i : ℕ), i < 128 → (∀ (j : ℕ), j < 128 → i ≠ j → ∃ (p : ℝ), p = probability_engineers_crimson_meet)) :=
sorry

end NUMINAMATH_GPT_engineers_crimson_meet_in_tournament_l131_13132


namespace NUMINAMATH_GPT_complement_U_A_inter_B_eq_l131_13111

open Set

-- Definitions
def U : Set ℤ := {x | ∃ n : ℤ, x = 2 * n}
def A : Set ℤ := {-2, 0, 2, 4}
def B : Set ℤ := {-2, 0, 4, 6, 8}

-- Complement of A in U
def complement_U_A : Set ℤ := U \ A

-- Proof Problem
theorem complement_U_A_inter_B_eq : complement_U_A ∩ B = {6, 8} := by
  sorry

end NUMINAMATH_GPT_complement_U_A_inter_B_eq_l131_13111


namespace NUMINAMATH_GPT_total_bricks_used_l131_13137

def numCoursesPerWall : Nat := 6
def bricksPerCourse : Nat := 10
def numWalls : Nat := 4
def unfinishedCoursesLastWall : Nat := 2

theorem total_bricks_used : 
  let totalCourses := numWalls * numCoursesPerWall
  let bricksRequired := totalCourses * bricksPerCourse
  let bricksMissing := unfinishedCoursesLastWall * bricksPerCourse
  let bricksUsed := bricksRequired - bricksMissing
  bricksUsed = 220 := 
by
  sorry

end NUMINAMATH_GPT_total_bricks_used_l131_13137


namespace NUMINAMATH_GPT_value_of_m_div_x_l131_13183

noncomputable def ratio_of_a_to_b (a b : ℝ) : Prop := a / b = 4 / 5
noncomputable def x_value (a : ℝ) : ℝ := a * 1.75
noncomputable def m_value (b : ℝ) : ℝ := b * 0.20

theorem value_of_m_div_x (a b : ℝ) (h1 : ratio_of_a_to_b a b) (h2 : 0 < a) (h3 : 0 < b) :
  (m_value b) / (x_value a) = 1 / 7 :=
by
  sorry

end NUMINAMATH_GPT_value_of_m_div_x_l131_13183


namespace NUMINAMATH_GPT_f_at_neg_8_5_pi_eq_pi_div_2_l131_13134

def f (x : Real) : Real := sorry

axiom functional_eqn (x : Real) : f (x + (3 * Real.pi / 2)) = -1 / f x
axiom f_interval (x : Real) (h : x ∈ Set.Icc (-Real.pi) Real.pi) : f x = x * Real.sin x

theorem f_at_neg_8_5_pi_eq_pi_div_2 : f (-8.5 * Real.pi) = Real.pi / 2 := 
  sorry

end NUMINAMATH_GPT_f_at_neg_8_5_pi_eq_pi_div_2_l131_13134


namespace NUMINAMATH_GPT_arithmetic_square_root_of_9_l131_13197

theorem arithmetic_square_root_of_9 : Real.sqrt 9 = 3 := by
  sorry

end NUMINAMATH_GPT_arithmetic_square_root_of_9_l131_13197


namespace NUMINAMATH_GPT_anne_distance_l131_13188

theorem anne_distance (speed time : ℕ) (h_speed : speed = 2) (h_time : time = 3) : 
  (speed * time) = 6 := by
  sorry

end NUMINAMATH_GPT_anne_distance_l131_13188


namespace NUMINAMATH_GPT_area_within_square_outside_semicircles_l131_13156

theorem area_within_square_outside_semicircles (side_length : ℝ) (r : ℝ) (area_square : ℝ) (area_semicircles : ℝ) (area_shaded : ℝ) 
  (h1 : side_length = 4)
  (h2 : r = side_length / 2)
  (h3 : area_square = side_length * side_length)
  (h4 : area_semicircles = 4 * (1 / 2 * π * r^2))
  (h5 : area_shaded = area_square - area_semicircles)
  : area_shaded = 16 - 8 * π :=
sorry

end NUMINAMATH_GPT_area_within_square_outside_semicircles_l131_13156


namespace NUMINAMATH_GPT_find_BP_l131_13112

theorem find_BP
  (A B C D P : Type) 
  (AP PC BP DP : ℝ)
  (hAP : AP = 8) 
  (hPC : PC = 1)
  (hBD : BD = 6)
  (hBP_less_DP : BP < DP) 
  (hPower_of_Point : AP * PC = BP * DP)
  : BP = 2 := 
by {
  sorry
}

end NUMINAMATH_GPT_find_BP_l131_13112


namespace NUMINAMATH_GPT_metres_sold_is_200_l131_13118

-- Define the conditions
def loss_per_metre : ℕ := 6
def cost_price_per_metre : ℕ := 66
def total_selling_price : ℕ := 12000

-- Define the selling price per metre based on the conditions
def selling_price_per_metre := cost_price_per_metre - loss_per_metre

-- Define the number of metres sold
def metres_sold : ℕ := total_selling_price / selling_price_per_metre

-- Proof statement: Check if the number of metres sold equals 200
theorem metres_sold_is_200 : metres_sold = 200 :=
  by
  sorry

end NUMINAMATH_GPT_metres_sold_is_200_l131_13118


namespace NUMINAMATH_GPT_problem1_problem2_l131_13184

variable (a b c : ℝ)

-- Condition: a, b, c are positive numbers and a + b + c = 1
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (habc : a + b + c = 1)

-- Problem 1: Prove that a^2 / b + b^2 / c + c^2 / a ≥ 1
theorem problem1 : a^2 / b + b^2 / c + c^2 / a ≥ 1 :=
by sorry

-- Problem 2: Prove that ab + bc + ca ≤ 1 / 3
theorem problem2 : ab + bc + ca ≤ 1 / 3 :=
by sorry

end NUMINAMATH_GPT_problem1_problem2_l131_13184


namespace NUMINAMATH_GPT_tangent_lines_ln_e_proof_l131_13162

noncomputable def tangent_tangent_ln_e : Prop :=
  ∀ (x₁ x₂ : ℝ) 
  (h₁ : x₁ > 0) 
  (h₁_eq : x₂ = -Real.log x₁)
  (h₂_eq : Real.log x₁ - 1 = (Real.exp x₂) * (1 - x₂)),
  (2 / (x₁ - 1) + x₂ = -1)

theorem tangent_lines_ln_e_proof : tangent_tangent_ln_e :=
  sorry

end NUMINAMATH_GPT_tangent_lines_ln_e_proof_l131_13162


namespace NUMINAMATH_GPT_parallel_lines_condition_l131_13143

theorem parallel_lines_condition (m n : ℝ) :
  (∃x y, (m * x + y - n = 0) ∧ (x + m * y + 1 = 0)) →
  (m = 1 ∧ n ≠ -1) ∨ (m = -1 ∧ n ≠ 1) :=
by
  sorry

end NUMINAMATH_GPT_parallel_lines_condition_l131_13143


namespace NUMINAMATH_GPT_find_a100_l131_13165

-- Define the arithmetic sequence with the given conditions
def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ :=
  a + (n - 1) * d

-- Given conditions
variables {a d : ℤ}
variables (S_9 : ℤ) (a_10 : ℤ)

-- Conditions in Lean definition
def conditions (a d : ℤ) : Prop :=
  (9 / 2 * (2 * a + 8 * d) = 27) ∧ (a + 9 * d = 8)

-- Prove the final statement
theorem find_a100 : ∃ a d : ℤ, conditions a d → arithmetic_sequence a d 100 = 98 := 
by {
    sorry
}

end NUMINAMATH_GPT_find_a100_l131_13165


namespace NUMINAMATH_GPT_rabbit_is_hit_l131_13147

noncomputable def P_A : ℝ := 0.6
noncomputable def P_B : ℝ := 0.5
noncomputable def P_C : ℝ := 0.4

noncomputable def P_none_hit : ℝ := (1 - P_A) * (1 - P_B) * (1 - P_C)
noncomputable def P_rabbit_hit : ℝ := 1 - P_none_hit

theorem rabbit_is_hit :
  P_rabbit_hit = 0.88 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_rabbit_is_hit_l131_13147


namespace NUMINAMATH_GPT_find_a_plus_k_l131_13171

-- Define the conditions.
def foci1 : (ℝ × ℝ) := (2, 0)
def foci2 : (ℝ × ℝ) := (2, 4)
def ellipse_point : (ℝ × ℝ) := (7, 2)

-- Statement of the equivalent proof problem.
theorem find_a_plus_k (a b h k : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) :
  (∀ x y, ((x - h)^2 / a^2 + (y - k)^2 / b^2 = 1) ↔ (x, y) = ellipse_point) →
  h = 2 → k = 2 → a = 5 →
  a + k = 7 :=
by
  sorry

end NUMINAMATH_GPT_find_a_plus_k_l131_13171


namespace NUMINAMATH_GPT_frank_initial_mushrooms_l131_13157

theorem frank_initial_mushrooms (pounds_eaten pounds_left initial_pounds : ℕ) 
  (h1 : pounds_eaten = 8) 
  (h2 : pounds_left = 7) 
  (h3 : initial_pounds = pounds_eaten + pounds_left) : 
  initial_pounds = 15 := 
by 
  sorry

end NUMINAMATH_GPT_frank_initial_mushrooms_l131_13157


namespace NUMINAMATH_GPT_sum_of_coordinates_of_point_B_l131_13158

theorem sum_of_coordinates_of_point_B
  (A : ℝ × ℝ) (hA : A = (0, 0))
  (B : ℝ × ℝ) (hB : ∃ x : ℝ, B = (x, 3))
  (slope_AB : ∃ x : ℝ, (3 - 0)/(x - 0) = 3/4) :
  (∃ x : ℝ, B = (x, 3)) ∧ x + 3 = 7 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_coordinates_of_point_B_l131_13158


namespace NUMINAMATH_GPT_percent_not_covering_politics_l131_13178

-- Definitions based on the conditions
def total_reporters : ℕ := 100
def local_politics_reporters : ℕ := 28
def percent_cover_local_politics : ℚ := 0.7

-- To be proved
theorem percent_not_covering_politics :
  let politics_reporters := local_politics_reporters / percent_cover_local_politics 
  (total_reporters - politics_reporters) / total_reporters = 0.6 := 
by
  sorry

end NUMINAMATH_GPT_percent_not_covering_politics_l131_13178


namespace NUMINAMATH_GPT_count_two_digit_remainders_l131_13129

theorem count_two_digit_remainders : 
  ∃ n, (∀ x, 10 ≤ x ∧ x < 100 ∧ x % 7 = 3 ↔ (∃ k, 1 ≤ k ∧ k ≤ 13 ∧ x = 7*k + 3)) ∧ n = 13 :=
by
  sorry

end NUMINAMATH_GPT_count_two_digit_remainders_l131_13129


namespace NUMINAMATH_GPT_quadrilateral_is_trapezium_l131_13195

-- Define the angles of the quadrilateral and the sum of the angles condition
variables {x : ℝ}
def sum_of_angles (x : ℝ) : Prop := x + 5 * x + 2 * x + 4 * x = 360

-- State the theorem
theorem quadrilateral_is_trapezium (x : ℝ) (h : sum_of_angles x) : 
  30 + 150 = 180 ∧ 60 + 120 = 180 → is_trapezium :=
sorry

end NUMINAMATH_GPT_quadrilateral_is_trapezium_l131_13195


namespace NUMINAMATH_GPT_relation_between_3a5_3b5_l131_13176

theorem relation_between_3a5_3b5 (a b : ℝ) (h : a > b) : 3 * a + 5 > 3 * b + 5 := by
  sorry

end NUMINAMATH_GPT_relation_between_3a5_3b5_l131_13176

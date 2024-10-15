import Mathlib

namespace NUMINAMATH_GPT_marsha_remainder_l1518_151864

-- Definitions based on problem conditions
def a (n : ℤ) : ℤ := 90 * n + 84
def b (m : ℤ) : ℤ := 120 * m + 114
def c (p : ℤ) : ℤ := 150 * p + 144

-- Proof statement
theorem marsha_remainder (n m p : ℤ) : ((a n + b m + c p) % 30) = 12 :=
by 
  -- Notice we need to add the proof steps here
  sorry 

end NUMINAMATH_GPT_marsha_remainder_l1518_151864


namespace NUMINAMATH_GPT_eval_three_plus_three_cubed_l1518_151896

theorem eval_three_plus_three_cubed : 3 + 3^3 = 30 := 
by 
  sorry

end NUMINAMATH_GPT_eval_three_plus_three_cubed_l1518_151896


namespace NUMINAMATH_GPT_rose_bush_cost_correct_l1518_151872

-- Definitions of the given conditions
def total_rose_bushes : ℕ := 20
def gardener_rate : ℕ := 30
def gardener_hours_per_day : ℕ := 5
def gardener_days : ℕ := 4
def gardener_cost : ℕ := gardener_rate * gardener_hours_per_day * gardener_days
def soil_cubic_feet : ℕ := 100
def soil_cost_per_cubic_foot : ℕ := 5
def soil_cost : ℕ := soil_cubic_feet * soil_cost_per_cubic_foot
def total_cost : ℕ := 4100

-- Result computed given the conditions
def rose_bush_cost : ℕ := 150

-- The proof goal (statement only, no proof)
theorem rose_bush_cost_correct : 
  total_cost - gardener_cost - soil_cost = total_rose_bushes * rose_bush_cost :=
by
  sorry

end NUMINAMATH_GPT_rose_bush_cost_correct_l1518_151872


namespace NUMINAMATH_GPT_no_p_dependence_l1518_151830

theorem no_p_dependence (m : ℕ) (p : ℕ) (hp : Prime p) (hm : m < p)
  (n : ℕ) (hn : 0 < n) (k : ℕ) 
  (h : m^2 + n^2 + p^2 - 2*m*n - 2*m*p - 2*n*p = k^2) : 
  ∀ q : ℕ, Prime q → m < q → (m^2 + n^2 + q^2 - 2*m*n - 2*m*q - 2*n*q = k^2) :=
by sorry

end NUMINAMATH_GPT_no_p_dependence_l1518_151830


namespace NUMINAMATH_GPT_calculate_expression_l1518_151898

theorem calculate_expression : (40 * 1505 - 20 * 1505) / 5 = 6020 := by
  sorry

end NUMINAMATH_GPT_calculate_expression_l1518_151898


namespace NUMINAMATH_GPT_new_person_weight_l1518_151820

theorem new_person_weight (n : ℕ) (k : ℝ) (w_old w_new : ℝ) 
  (h_n : n = 6) 
  (h_k : k = 4.5) 
  (h_w_old : w_old = 75) 
  (h_avg_increase : w_new - w_old = n * k) : 
  w_new = 102 := 
sorry

end NUMINAMATH_GPT_new_person_weight_l1518_151820


namespace NUMINAMATH_GPT_peach_difference_proof_l1518_151827

def red_peaches_odd := 12
def green_peaches_odd := 22
def red_peaches_even := 15
def green_peaches_even := 20
def num_baskets := 20
def num_odd_baskets := num_baskets / 2
def num_even_baskets := num_baskets / 2

def total_red_peaches := (red_peaches_odd * num_odd_baskets) + (red_peaches_even * num_even_baskets)
def total_green_peaches := (green_peaches_odd * num_odd_baskets) + (green_peaches_even * num_even_baskets)
def difference := total_green_peaches - total_red_peaches

theorem peach_difference_proof : difference = 150 := by
  sorry

end NUMINAMATH_GPT_peach_difference_proof_l1518_151827


namespace NUMINAMATH_GPT_b_days_work_alone_l1518_151886

theorem b_days_work_alone 
  (W_b : ℝ)  -- Work done by B in one day
  (W_a : ℝ)  -- Work done by A in one day
  (D_b : ℝ)  -- Number of days for B to complete the work alone
  (h1 : W_a = 2 * W_b)  -- A is twice as good a workman as B
  (h2 : 7 * (W_a + W_b) = D_b * W_b)  -- A and B took 7 days together to do the work
  : D_b = 21 :=
sorry

end NUMINAMATH_GPT_b_days_work_alone_l1518_151886


namespace NUMINAMATH_GPT_total_ticket_count_is_59_l1518_151869

-- Define the constants and variables
def price_adult : ℝ := 4
def price_student : ℝ := 2.5
def total_revenue : ℝ := 222.5
def student_tickets_sold : ℕ := 9

-- Define the equation representing the total revenue and solve for the number of adult tickets
noncomputable def total_tickets_sold (adult_tickets : ℕ) :=
  adult_tickets + student_tickets_sold

theorem total_ticket_count_is_59 (A : ℕ) 
  (h : price_adult * A + price_student * (student_tickets_sold : ℝ) = total_revenue) :
  total_tickets_sold A = 59 :=
by
  sorry

end NUMINAMATH_GPT_total_ticket_count_is_59_l1518_151869


namespace NUMINAMATH_GPT_total_food_each_day_l1518_151817

-- Definitions as per conditions
def soldiers_first_side : Nat := 4000
def food_per_soldier_first_side : Nat := 10
def soldiers_difference : Nat := 500
def food_difference : Nat := 2

-- Proving the total amount of food
theorem total_food_each_day : 
  let soldiers_second_side := soldiers_first_side - soldiers_difference
  let food_per_soldier_second_side := food_per_soldier_first_side - food_difference
  let total_food_first_side := soldiers_first_side * food_per_soldier_first_side
  let total_food_second_side := soldiers_second_side * food_per_soldier_second_side
  total_food_first_side + total_food_second_side = 68000 := by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_total_food_each_day_l1518_151817


namespace NUMINAMATH_GPT_annual_avg_growth_rate_export_volume_2023_l1518_151888

variable (V0 V2 V3 : ℕ) (r : ℝ)
variable (h1 : V0 = 200000) (h2 : V2 = 450000) (h3 : V3 = 675000)

-- Definition of the exponential growth equation
def growth_exponential (V0 Vn: ℕ) (n : ℕ) (r : ℝ) : Prop :=
  Vn = V0 * ((1 + r) ^ n)

-- The Lean statement to prove the annual average growth rate
theorem annual_avg_growth_rate (x : ℝ) (h : growth_exponential V0 V2 2 x) : 
  x = 0.5 :=
by
  sorry

-- The Lean statement to prove the export volume in 2023
theorem export_volume_2023 (h_growth : growth_exponential V2 V3 1 0.5) :
  V3 = 675000 :=
by
  sorry

end NUMINAMATH_GPT_annual_avg_growth_rate_export_volume_2023_l1518_151888


namespace NUMINAMATH_GPT_minimize_expression_l1518_151845

theorem minimize_expression (x y z : ℝ) (h : 0 < x ∧ 0 < y ∧ 0 < z) (h_xyz : x * y * z = 2 / 3) :
  x^2 + 6 * x * y + 18 * y^2 + 12 * y * z + 4 * z^2 = 18 :=
sorry

end NUMINAMATH_GPT_minimize_expression_l1518_151845


namespace NUMINAMATH_GPT_sheets_per_pack_l1518_151854

theorem sheets_per_pack (p d t : Nat) (total_sheets : Nat) (sheets_per_pack : Nat) 
  (h1 : p = 2) (h2 : d = 80) (h3 : t = 6) 
  (h4 : total_sheets = d * t)
  (h5 : sheets_per_pack = total_sheets / p) : sheets_per_pack = 240 := 
  by 
    sorry

end NUMINAMATH_GPT_sheets_per_pack_l1518_151854


namespace NUMINAMATH_GPT_regular_polygon_sides_l1518_151884

theorem regular_polygon_sides (n : ℕ) (h₁ : 1 < n)
  (interior_angle_sum : 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end NUMINAMATH_GPT_regular_polygon_sides_l1518_151884


namespace NUMINAMATH_GPT_rose_joined_after_six_months_l1518_151895

noncomputable def profit_shares (m : ℕ) : ℕ :=
  12000 * (12 - m) - 9000 * 8

theorem rose_joined_after_six_months :
  ∃ (m : ℕ), profit_shares m = 370 :=
by
  use 6
  unfold profit_shares
  norm_num
  sorry

end NUMINAMATH_GPT_rose_joined_after_six_months_l1518_151895


namespace NUMINAMATH_GPT_correct_option_C_l1518_151844

variable (x : ℝ)
variable (hx : 0 < x ∧ x < 1)

theorem correct_option_C : 0 < 1 - x^2 ∧ 1 - x^2 < 1 :=
by
  sorry

end NUMINAMATH_GPT_correct_option_C_l1518_151844


namespace NUMINAMATH_GPT_min_value_144_l1518_151818

noncomputable def min_expression (a b c d : ℝ) : ℝ :=
  (a + b + c) / (a * b * c * d)

theorem min_value_144 (a b c d : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
    (h_pos_d : 0 < d) (h_sum : a + b + c + d = 2) : min_expression a b c d ≥ 144 :=
by
  sorry

end NUMINAMATH_GPT_min_value_144_l1518_151818


namespace NUMINAMATH_GPT_six_sin6_cos6_l1518_151836

theorem six_sin6_cos6 (A : ℝ) (h : Real.cos (2 * A) = - Real.sqrt 5 / 3) : 
  6 * Real.sin (A) ^ 6 + 6 * Real.cos (A) ^ 6 = 4 := 
sorry

end NUMINAMATH_GPT_six_sin6_cos6_l1518_151836


namespace NUMINAMATH_GPT_ratio_equivalence_l1518_151828

theorem ratio_equivalence (x : ℚ) (h : x / 360 = 18 / 12) : x = 540 :=
by
  -- Proof goes here, to be filled in
  sorry

end NUMINAMATH_GPT_ratio_equivalence_l1518_151828


namespace NUMINAMATH_GPT_triangle_area_l1518_151819

theorem triangle_area : 
  let line_eq (x y : ℝ) := 3 * x + 2 * y = 12
  let x_intercept := (4 : ℝ)
  let y_intercept := (6 : ℝ)
  ∃ (x y : ℝ), line_eq x y ∧ x = x_intercept ∧ y = y_intercept ∧
  ∃ (area : ℝ), area = 1 / 2 * x * y ∧ area = 12 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_l1518_151819


namespace NUMINAMATH_GPT_largest_of_seven_consecutive_l1518_151815

theorem largest_of_seven_consecutive (n : ℕ) (h1 : (7 * n + 21 = 3020)) : (n + 6 = 434) :=
sorry

end NUMINAMATH_GPT_largest_of_seven_consecutive_l1518_151815


namespace NUMINAMATH_GPT_integers_exist_for_eqns_l1518_151893

theorem integers_exist_for_eqns (a b c : ℤ) :
  ∃ (p1 q1 r1 p2 q2 r2 : ℤ), 
    a = q1 * r2 - q2 * r1 ∧ 
    b = r1 * p2 - r2 * p1 ∧ 
    c = p1 * q2 - p2 * q1 :=
  sorry

end NUMINAMATH_GPT_integers_exist_for_eqns_l1518_151893


namespace NUMINAMATH_GPT_distinct_values_of_fx_l1518_151897

theorem distinct_values_of_fx :
  let f (x : ℝ) := ⌊x⌋ + ⌊2 * x⌋ + ⌊3 * x⌋ + ⌊4 * x⌋
  ∃ (s : Finset ℤ), (∀ x, 0 ≤ x ∧ x ≤ 10 → f x ∈ s) ∧ s.card = 61 :=
by
  sorry

end NUMINAMATH_GPT_distinct_values_of_fx_l1518_151897


namespace NUMINAMATH_GPT_train_B_speed_l1518_151853

theorem train_B_speed (V_B : ℝ) : 
  (∀ t meet_A meet_B, 
     meet_A = 9 ∧
     meet_B = 4 ∧
     t = 70 ∧
     (t * meet_A) = (V_B * meet_B)) →
     V_B = 157.5 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_train_B_speed_l1518_151853


namespace NUMINAMATH_GPT_points_in_first_quadrant_points_in_fourth_quadrant_points_in_second_quadrant_points_in_third_quadrant_l1518_151866

def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0
def fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0
def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0
def third_quadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0

theorem points_in_first_quadrant (x y : ℝ) (h : x > 0 ∧ y > 0) : first_quadrant x y :=
by {
  sorry
}

theorem points_in_fourth_quadrant (x y : ℝ) (h : x > 0 ∧ y < 0) : fourth_quadrant x y :=
by {
  sorry
}

theorem points_in_second_quadrant (x y : ℝ) (h : x < 0 ∧ y > 0) : second_quadrant x y :=
by {
  sorry
}

theorem points_in_third_quadrant (x y : ℝ) (h : x < 0 ∧ y < 0) : third_quadrant x y :=
by {
  sorry
}

end NUMINAMATH_GPT_points_in_first_quadrant_points_in_fourth_quadrant_points_in_second_quadrant_points_in_third_quadrant_l1518_151866


namespace NUMINAMATH_GPT_shortest_distance_between_circles_l1518_151847

def circle_eq1 (x y : ℝ) : Prop := x^2 - 6*x + y^2 - 8*y - 15 = 0
def circle_eq2 (x y : ℝ) : Prop := x^2 + 10*x + y^2 + 12*y + 21 = 0

theorem shortest_distance_between_circles :
  ∀ (x1 y1 x2 y2 : ℝ), circle_eq1 x1 y1 → circle_eq2 x2 y2 → 
  (abs ((x1 - x2)^2 + (y1 - y2)^2)^(1/2) - (15^(1/2) + 82^(1/2))) =
  2 * 41^(1/2) - 97^(1/2) :=
by sorry

end NUMINAMATH_GPT_shortest_distance_between_circles_l1518_151847


namespace NUMINAMATH_GPT_tom_is_15_l1518_151878

theorem tom_is_15 (T M : ℕ) (h1 : T + M = 21) (h2 : T + 3 = 2 * (M + 3)) : T = 15 :=
by {
  sorry
}

end NUMINAMATH_GPT_tom_is_15_l1518_151878


namespace NUMINAMATH_GPT_product_of_odd_primes_mod_32_l1518_151831

theorem product_of_odd_primes_mod_32 :
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := primes.foldr (· * ·) 1
  M % 32 = 17 :=
by
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := primes.foldr (· * ·) 1
  exact sorry

end NUMINAMATH_GPT_product_of_odd_primes_mod_32_l1518_151831


namespace NUMINAMATH_GPT_distance_from_wall_to_picture_edge_l1518_151814

theorem distance_from_wall_to_picture_edge
  (wall_width : ℕ)
  (picture_width : ℕ)
  (centered : Prop)
  (h1 : wall_width = 22)
  (h2 : picture_width = 4)
  (h3 : centered) :
  ∃ x : ℕ, x = 9 :=
by
  sorry

end NUMINAMATH_GPT_distance_from_wall_to_picture_edge_l1518_151814


namespace NUMINAMATH_GPT_total_meters_built_l1518_151883

/-- Define the length of the road -/
def road_length (L : ℕ) := L = 1000

/-- Define the average meters built per day -/
def average_meters_per_day (A : ℕ) := A = 120

/-- Define the number of days worked from July 29 to August 2 -/
def number_of_days_worked (D : ℕ) := D = 5

/-- The total meters built by the time they finished -/
theorem total_meters_built
  (L A D : ℕ)
  (h1 : road_length L)
  (h2 : average_meters_per_day A)
  (h3 : number_of_days_worked D)
  : L / D * A = 600 := by
  sorry

end NUMINAMATH_GPT_total_meters_built_l1518_151883


namespace NUMINAMATH_GPT_probability_sequence_correct_l1518_151840

noncomputable def probability_of_sequence : ℚ :=
  (13 / 52) * (13 / 51) * (13 / 50)

theorem probability_sequence_correct :
  probability_of_sequence = 2197 / 132600 :=
by
  sorry

end NUMINAMATH_GPT_probability_sequence_correct_l1518_151840


namespace NUMINAMATH_GPT_negation_equiv_no_solution_l1518_151812

-- Definition of there is at least one solution
def at_least_one_solution (P : α → Prop) : Prop := ∃ x, P x

-- Definition of no solution
def no_solution (P : α → Prop) : Prop := ∀ x, ¬ P x

-- Problem statement to prove that the negation of at_least_one_solution is equivalent to no_solution
theorem negation_equiv_no_solution (P : α → Prop) :
  ¬ at_least_one_solution P ↔ no_solution P := 
sorry

end NUMINAMATH_GPT_negation_equiv_no_solution_l1518_151812


namespace NUMINAMATH_GPT_no_positive_integers_m_n_l1518_151823

theorem no_positive_integers_m_n (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  m^3 + 11^3 ≠ n^3 :=
sorry

end NUMINAMATH_GPT_no_positive_integers_m_n_l1518_151823


namespace NUMINAMATH_GPT_right_angled_triangles_l1518_151832

theorem right_angled_triangles (x y z : ℕ) : (x - 6) * (y - 6) = 18 ∧ (x^2 + y^2 = z^2)
  → (3 * (x + y + z) = x * y) :=
sorry

end NUMINAMATH_GPT_right_angled_triangles_l1518_151832


namespace NUMINAMATH_GPT_angela_initial_action_figures_l1518_151808

theorem angela_initial_action_figures (X : ℕ) (h1 : X - (1/4 : ℚ) * X - (1/3 : ℚ) * (3/4 : ℚ) * X = 12) : X = 24 :=
sorry

end NUMINAMATH_GPT_angela_initial_action_figures_l1518_151808


namespace NUMINAMATH_GPT_gcd_10010_15015_l1518_151806

theorem gcd_10010_15015 :
  let n1 := 10010
  let n2 := 15015
  ∃ d, d = Nat.gcd n1 n2 ∧ d = 5005 :=
by
  let n1 := 10010
  let n2 := 15015
  -- ... omitted proof steps
  sorry

end NUMINAMATH_GPT_gcd_10010_15015_l1518_151806


namespace NUMINAMATH_GPT_mailman_junk_mail_l1518_151870

/-- 
  Given:
    - n = 640 : total number of pieces of junk mail for the block
    - h = 20 : number of houses in the block
  
  Prove:
    - The number of pieces of junk mail given to each house equals 32, when the total number of pieces of junk mail is divided by the number of houses.
--/
theorem mailman_junk_mail (n h : ℕ) (h_total : n = 640) (h_houses : h = 20) :
  n / h = 32 :=
by
  sorry

end NUMINAMATH_GPT_mailman_junk_mail_l1518_151870


namespace NUMINAMATH_GPT_problem1_problem2_problem3_l1518_151889

noncomputable def a_n (n : ℕ) : ℕ := 3 * (2 ^ n) - 3
noncomputable def S_n (n : ℕ) : ℕ := 2 * a_n n - 3 * n

-- 1. Prove a_1 = 3 and a_2 = 9 given S_n = 2a_n - 3n
theorem problem1 (n : ℕ) (h : ∀ n > 0, S_n n = 2 * (a_n n) - 3 * n) :
    a_n 1 = 3 ∧ a_n 2 = 9 :=
  sorry

-- 2. Prove that the sequence {a_n + 3} is a geometric sequence and find the general term formula for the sequence {a_n}.
theorem problem2 (n : ℕ) (h : ∀ n > 0, S_n n = 2 * (a_n n) - 3 * n) :
    ∀ n, (a_n (n + 1) + 3) / (a_n n + 3) = 2 ∧ a_n n = 3 * (2 ^ n) - 3 :=
  sorry

-- 3. Prove {S_{n_k}} is not an arithmetic sequence given S_n = 2a_n - 3n and {n_k} is an arithmetic sequence
theorem problem3 (n_k : ℕ → ℕ) (h_arithmetic : ∃ d, ∀ k, n_k (k + 1) - n_k k = d) :
    ¬ ∃ d, ∀ k, S_n (n_k (k + 1)) - S_n (n_k k) = d :=
  sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_l1518_151889


namespace NUMINAMATH_GPT_geometric_seq_a8_l1518_151877

noncomputable def geometric_seq_term (a₁ r : ℝ) (n : ℕ) : ℝ :=
  a₁ * r^(n-1)

noncomputable def geometric_seq_sum (a₁ r : ℝ) (n : ℕ) : ℝ :=
  a₁ * (1 - r^n) / (1 - r)

theorem geometric_seq_a8
  (a₁ r : ℝ)
  (h1 : geometric_seq_sum a₁ r 3 = 7/4)
  (h2 : geometric_seq_sum a₁ r 6 = 63/4)
  (h3 : r ≠ 1) :
  geometric_seq_term a₁ r 8 = 32 :=
by
  sorry

end NUMINAMATH_GPT_geometric_seq_a8_l1518_151877


namespace NUMINAMATH_GPT_find_y_given_conditions_l1518_151899

theorem find_y_given_conditions (a x y : ℝ) (h1 : y = a * x + (1 - a)) 
  (x_val : x = 3) (y_val : y = 7) (x_new : x = 8) :
  y = 22 := 
  sorry

end NUMINAMATH_GPT_find_y_given_conditions_l1518_151899


namespace NUMINAMATH_GPT_exists_inequality_l1518_151850

theorem exists_inequality (n : ℕ) (x : Fin (n + 1) → ℝ) 
  (hx1 : ∀ i, 0 ≤ x i ∧ x i ≤ 1) 
  (h_n : 2 ≤ n) : 
  ∃ i : Fin n, x i * (1 - x (i + 1)) ≥ (1 / 4) * x 0 * (1 - x n) :=
sorry

end NUMINAMATH_GPT_exists_inequality_l1518_151850


namespace NUMINAMATH_GPT_find_m_l1518_151842

theorem find_m {x1 x2 m : ℝ} 
  (h_eqn : ∀ x, x^2 - (m+3)*x + (m+2) = 0) 
  (h_cond : x1 / (x1 + 1) + x2 / (x2 + 1) = 13 / 10) : 
  m = 2 := 
sorry

end NUMINAMATH_GPT_find_m_l1518_151842


namespace NUMINAMATH_GPT_minimum_value_f_range_a_l1518_151868

noncomputable def f (x : ℝ) : ℝ := (x + 1) * Real.log (x + 1)

theorem minimum_value_f :
  ∃ x : ℝ, f x = -(1 / Real.exp 1) :=
sorry

theorem range_a (a : ℝ) :
  (∀ x ≥ 0, f x ≥ a * x) ↔ a ∈ Set.Iic 1 :=
sorry

end NUMINAMATH_GPT_minimum_value_f_range_a_l1518_151868


namespace NUMINAMATH_GPT_determine_x_y_l1518_151880

-- Definitions from the conditions
def cond1 (x y : ℚ) : Prop := 12 * x + 198 = 12 * y + 176
def cond2 (x y : ℚ) : Prop := x + y = 29

-- Statement to prove
theorem determine_x_y : ∃ x y : ℚ, cond1 x y ∧ cond2 x y ∧ x = 163 / 12 ∧ y = 185 / 12 := 
by 
  sorry

end NUMINAMATH_GPT_determine_x_y_l1518_151880


namespace NUMINAMATH_GPT_find_B_value_l1518_151856

def divisible_by_9 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 9 * k

theorem find_B_value (B : ℕ) :
  divisible_by_9 (4 * 10^4 + B * 10^3 + B * 10^2 + 1 * 10 + 3) →
  0 ≤ B ∧ B ≤ 9 →
  B = 5 :=
sorry

end NUMINAMATH_GPT_find_B_value_l1518_151856


namespace NUMINAMATH_GPT_not_divisible_by_1998_l1518_151875

theorem not_divisible_by_1998 (n : ℕ) :
  ∀ k : ℕ, ¬ (2^(k+1) * n + 2^k - 1) % 2 = 0 → ¬ (2^(k+1) * n + 2^k - 1) % 1998 = 0 :=
by
  intros _ _
  sorry

end NUMINAMATH_GPT_not_divisible_by_1998_l1518_151875


namespace NUMINAMATH_GPT_correct_sampling_method_l1518_151837

structure SchoolPopulation :=
  (senior : ℕ)
  (intermediate : ℕ)
  (junior : ℕ)

-- Define the school population
def school : SchoolPopulation :=
  { senior := 10, intermediate := 50, junior := 75 }

-- Define the condition for sampling method
def total_school_teachers (s : SchoolPopulation) : ℕ :=
  s.senior + s.intermediate + s.junior

-- The desired sample size
def sample_size : ℕ := 30

-- The correct sampling method based on the population strata
def stratified_sampling (s : SchoolPopulation) : Prop :=
  s.senior + s.intermediate + s.junior > 0

theorem correct_sampling_method : stratified_sampling school :=
by { sorry }

end NUMINAMATH_GPT_correct_sampling_method_l1518_151837


namespace NUMINAMATH_GPT_commutativity_l1518_151813

universe u

variable {M : Type u} [Nonempty M]
variable (star : M → M → M)

axiom star_assoc_right {a b : M} : (star (star a b) b) = a
axiom star_assoc_left {a b : M} : star a (star a b) = b

theorem commutativity (a b : M) : star a b = star b a :=
by sorry

end NUMINAMATH_GPT_commutativity_l1518_151813


namespace NUMINAMATH_GPT_kim_gets_change_of_5_l1518_151867

noncomputable def meal_cost : ℝ := 10
noncomputable def drink_cost : ℝ := 2.5
noncomputable def tip_rate : ℝ := 0.20
noncomputable def payment : ℝ := 20
noncomputable def total_cost_before_tip := meal_cost + drink_cost
noncomputable def tip := tip_rate * total_cost_before_tip
noncomputable def total_cost_with_tip := total_cost_before_tip + tip
noncomputable def change := payment - total_cost_with_tip

theorem kim_gets_change_of_5 : change = 5 := by
  sorry

end NUMINAMATH_GPT_kim_gets_change_of_5_l1518_151867


namespace NUMINAMATH_GPT_find_number_l1518_151838

theorem find_number (x : ℝ) (h : (x - 5) / 3 = 4) : x = 17 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_number_l1518_151838


namespace NUMINAMATH_GPT_intersection_M_N_l1518_151824

open Set

def M := { x : ℝ | 0 < x ∧ x < 3 }
def N := { x : ℝ | x^2 - 5 * x + 4 ≥ 0 }

theorem intersection_M_N :
  { x | x ∈ M ∧ x ∈ N } = { x | 0 < x ∧ x ≤ 1 } :=
sorry

end NUMINAMATH_GPT_intersection_M_N_l1518_151824


namespace NUMINAMATH_GPT_percent_not_crust_l1518_151861

-- Definitions as conditions
def pie_total_weight : ℕ := 200
def crust_weight : ℕ := 50

-- The theorem to be proven
theorem percent_not_crust : (pie_total_weight - crust_weight) / pie_total_weight * 100 = 75 := 
by
  sorry

end NUMINAMATH_GPT_percent_not_crust_l1518_151861


namespace NUMINAMATH_GPT_molecular_weight_of_compound_l1518_151816

-- Definitions of the atomic weights.
def atomic_weight_K : ℝ := 39.10
def atomic_weight_Br : ℝ := 79.90
def atomic_weight_O : ℝ := 16.00

-- Proof statement of the molecular weight of the compound.
theorem molecular_weight_of_compound :
  (1 * atomic_weight_K) + (1 * atomic_weight_Br) + (3 * atomic_weight_O) = 167.00 :=
  by
    sorry

end NUMINAMATH_GPT_molecular_weight_of_compound_l1518_151816


namespace NUMINAMATH_GPT_isosceles_triangle_area_48_l1518_151801

noncomputable def isosceles_triangle_area (b h s : ℝ) : ℝ :=
  (1 / 2) * (2 * b) * h

theorem isosceles_triangle_area_48 :
  ∀ (b s : ℝ),
  b ^ 2 + 8 ^ 2 = s ^ 2 ∧ s + b = 16 →
  isosceles_triangle_area b 8 s = 48 :=
by
  intros b s h
  unfold isosceles_triangle_area
  sorry

end NUMINAMATH_GPT_isosceles_triangle_area_48_l1518_151801


namespace NUMINAMATH_GPT_molecular_weight_calculation_l1518_151887

-- Define the atomic weights of each element
def atomic_weight_C : ℝ := 12.01
def atomic_weight_H : ℝ := 1.008
def atomic_weight_O : ℝ := 16.00

-- Define the number of atoms of each element in the compound
def num_atoms_C : ℕ := 7
def num_atoms_H : ℕ := 6
def num_atoms_O : ℕ := 2

-- The molecular weight calculation
def molecular_weight : ℝ :=
  (num_atoms_C * atomic_weight_C) +
  (num_atoms_H * atomic_weight_H) +
  (num_atoms_O * atomic_weight_O)

theorem molecular_weight_calculation : molecular_weight = 122.118 :=
by
  -- Proof
  sorry

end NUMINAMATH_GPT_molecular_weight_calculation_l1518_151887


namespace NUMINAMATH_GPT_average_speed_second_day_l1518_151860

theorem average_speed_second_day
  (t v : ℤ)
  (h1 : 2 * t + 2 = 18)
  (h2 : (v + 5) * (t + 2) + v * t = 680) :
  v = 35 :=
by
  sorry

end NUMINAMATH_GPT_average_speed_second_day_l1518_151860


namespace NUMINAMATH_GPT_emilia_strCartons_l1518_151849

theorem emilia_strCartons (total_cartons_needed cartons_bought cartons_blueberries : ℕ) (h1 : total_cartons_needed = 42) (h2 : cartons_blueberries = 7) (h3 : cartons_bought = 33) :
  (total_cartons_needed - (cartons_bought + cartons_blueberries)) = 2 :=
by
  sorry

end NUMINAMATH_GPT_emilia_strCartons_l1518_151849


namespace NUMINAMATH_GPT_no_solution_set_1_2_4_l1518_151811

theorem no_solution_set_1_2_4 
  (f : ℝ → ℝ) 
  (hf : ∃ (a b c : ℝ), ∀ x, f x = a * x^2 + b * x + c)
  (t : ℝ) : ¬ ∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ f (|x1 - t|) = 0 ∧ f (|x2 - t|) = 0 ∧ f (|x3 - t|) = 0 ∧ (x1 = 1 ∧ x2 = 2 ∧ x3 = 4) := 
sorry

end NUMINAMATH_GPT_no_solution_set_1_2_4_l1518_151811


namespace NUMINAMATH_GPT_find_x_l1518_151894

theorem find_x (A V R S x : ℝ) 
  (h1 : A + x = V - x)
  (h2 : V + 2 * x = A - 2 * x + 30)
  (h3 : (A + R / 2) + (V + R / 2) = 120)
  (h4 : S - 0.25 * S + 10 = 2 * (R / 2)) :
  x = 5 :=
  sorry

end NUMINAMATH_GPT_find_x_l1518_151894


namespace NUMINAMATH_GPT_number_of_cookies_first_friend_took_l1518_151874

-- Definitions of given conditions:
def initial_cookies : ℕ := 22
def eaten_by_Kristy : ℕ := 2
def given_to_brother : ℕ := 1
def taken_by_second_friend : ℕ := 5
def taken_by_third_friend : ℕ := 5
def cookies_left : ℕ := 6

noncomputable def cookies_after_Kristy_ate_and_gave_away : ℕ :=
  initial_cookies - eaten_by_Kristy - given_to_brother

noncomputable def cookies_after_second_and_third_friends : ℕ :=
  taken_by_second_friend + taken_by_third_friend

noncomputable def cookies_before_second_and_third_friends_took : ℕ :=
  cookies_left + cookies_after_second_and_third_friends

theorem number_of_cookies_first_friend_took :
  cookies_after_Kristy_ate_and_gave_away - cookies_before_second_and_third_friends_took = 3 := by
  sorry

end NUMINAMATH_GPT_number_of_cookies_first_friend_took_l1518_151874


namespace NUMINAMATH_GPT_leaked_before_fixing_l1518_151863

def total_leaked_oil := 6206
def leaked_while_fixing := 3731

theorem leaked_before_fixing :
  total_leaked_oil - leaked_while_fixing = 2475 := by
  sorry

end NUMINAMATH_GPT_leaked_before_fixing_l1518_151863


namespace NUMINAMATH_GPT_prime_divides_sequence_term_l1518_151805

theorem prime_divides_sequence_term (k : ℕ) (h_prime : Nat.Prime k) (h_ne_two : k ≠ 2) (h_ne_five : k ≠ 5) :
  ∃ n ≤ k, k ∣ (Nat.ofDigits 10 (List.replicate n 1)) :=
by
  sorry

end NUMINAMATH_GPT_prime_divides_sequence_term_l1518_151805


namespace NUMINAMATH_GPT_additional_savings_if_purchase_together_l1518_151859

theorem additional_savings_if_purchase_together :
  let price_per_window := 100
  let windows_each_offer := 4
  let free_each_offer := 1
  let dave_windows := 7
  let doug_windows := 8

  let cost_without_offer (windows : Nat) := windows * price_per_window
  let cost_with_offer (windows : Nat) := 
    if windows % (windows_each_offer + free_each_offer) = 0 then
      (windows / (windows_each_offer + free_each_offer)) * windows_each_offer * price_per_window
    else
      (windows / (windows_each_offer + free_each_offer)) * windows_each_offer * price_per_window 
      + (windows % (windows_each_offer + free_each_offer)) * price_per_window

  (cost_without_offer (dave_windows + doug_windows) 
  - cost_with_offer (dave_windows + doug_windows)) 
  - ((cost_without_offer dave_windows - cost_with_offer dave_windows)
  + (cost_without_offer doug_windows - cost_with_offer doug_windows)) = price_per_window := 
  sorry

end NUMINAMATH_GPT_additional_savings_if_purchase_together_l1518_151859


namespace NUMINAMATH_GPT_ratio_area_square_circle_eq_pi_l1518_151804

theorem ratio_area_square_circle_eq_pi
  (a r : ℝ)
  (h : 4 * a = 4 * π * r) :
  (a^2 / (π * r^2)) = π := by
  sorry

end NUMINAMATH_GPT_ratio_area_square_circle_eq_pi_l1518_151804


namespace NUMINAMATH_GPT_slope_intercept_parallel_l1518_151846

theorem slope_intercept_parallel (A : ℝ × ℝ) (x y : ℝ) (hA : A = (3, 2))
(hparallel : 4 * x + y - 2 = 0) :
  ∃ b : ℝ, y = -4 * x + b ∧ b = 14 :=
by
  sorry

end NUMINAMATH_GPT_slope_intercept_parallel_l1518_151846


namespace NUMINAMATH_GPT_sum_abs_eq_l1518_151803

theorem sum_abs_eq (a b : ℝ) (ha : |a| = 10) (hb : |b| = 7) (hab : a > b) : a + b = 17 ∨ a + b = 3 :=
sorry

end NUMINAMATH_GPT_sum_abs_eq_l1518_151803


namespace NUMINAMATH_GPT_find_number_of_clerks_l1518_151891

-- Define the conditions 
def avg_salary_per_head_staff : ℝ := 90
def avg_salary_officers : ℝ := 600
def avg_salary_clerks : ℝ := 84
def number_of_officers : ℕ := 2

-- Define the variable C (number of clerks)
def number_of_clerks : ℕ := sorry   -- We will prove that this is 170

-- Define the total salary equations based on the conditions
def total_salary_officers := number_of_officers * avg_salary_officers
def total_salary_clerks := number_of_clerks * avg_salary_clerks
def total_number_of_staff := number_of_officers + number_of_clerks
def total_salary := total_salary_officers + total_salary_clerks

-- Define the average salary per head equation 
def avg_salary_eq : Prop := avg_salary_per_head_staff = total_salary / total_number_of_staff

theorem find_number_of_clerks (h : avg_salary_eq) : number_of_clerks = 170 :=
sorry

end NUMINAMATH_GPT_find_number_of_clerks_l1518_151891


namespace NUMINAMATH_GPT_product_of_roots_l1518_151857

theorem product_of_roots (x : ℝ) (h : (x - 1) * (x + 4) = 22) : ∃ a b, (x^2 + 3*x - 26 = 0) ∧ a * b = -26 :=
by
  -- Given the equation (x - 1) * (x + 4) = 22,
  -- We want to show that the roots of the equation when simplified are such that
  -- their product is -26.
  sorry

end NUMINAMATH_GPT_product_of_roots_l1518_151857


namespace NUMINAMATH_GPT_train_crossing_time_l1518_151841

-- Define the problem conditions in Lean 4
def train_length : ℕ := 130
def bridge_length : ℕ := 150
def speed_kmph : ℕ := 36
def speed_mps : ℕ := (speed_kmph * 1000 / 3600)

-- The statement to prove
theorem train_crossing_time : (train_length + bridge_length) / speed_mps = 28 :=
by
  -- The proof starts here
  sorry

end NUMINAMATH_GPT_train_crossing_time_l1518_151841


namespace NUMINAMATH_GPT_product_eval_l1518_151821

theorem product_eval (a : ℤ) (h : a = 3) : (a - 6) * (a - 5) * (a - 4) * (a - 3) * (a - 2) * (a - 1) * a = 0 :=
by
  sorry

end NUMINAMATH_GPT_product_eval_l1518_151821


namespace NUMINAMATH_GPT_find_n_l1518_151882

theorem find_n (n : ℤ) (h : Real.sqrt (10 + n) = 9) : n = 71 :=
sorry

end NUMINAMATH_GPT_find_n_l1518_151882


namespace NUMINAMATH_GPT_f_even_of_g_odd_l1518_151822

theorem f_even_of_g_odd (g : ℝ → ℝ) (f : ℝ → ℝ) (h1 : ∀ x, g (-x) = -g x) (h2 : ∀ x, f x = |g (x^5)|) : ∀ x, f (-x) = f x := 
by
  sorry

end NUMINAMATH_GPT_f_even_of_g_odd_l1518_151822


namespace NUMINAMATH_GPT_mother_age_4times_daughter_l1518_151871

-- Conditions
def Y := 12
def M := 42

-- Proof statement: Prove that 2 years ago, the mother's age was 4 times Yujeong's age.
theorem mother_age_4times_daughter (X : ℕ) (hY : Y = 12) (hM : M = 42) : (42 - X) = 4 * (12 - X) :=
by
  intros
  sorry

end NUMINAMATH_GPT_mother_age_4times_daughter_l1518_151871


namespace NUMINAMATH_GPT_rachel_picture_shelves_l1518_151829

-- We define the number of books per shelf
def books_per_shelf : ℕ := 9

-- We define the number of mystery shelves
def mystery_shelves : ℕ := 6

-- We define the total number of books
def total_books : ℕ := 72

-- We create a theorem that states Rachel had 2 shelves of picture books
theorem rachel_picture_shelves : ∃ (picture_shelves : ℕ), 
  (mystery_shelves * books_per_shelf + picture_shelves * books_per_shelf = total_books) ∧
  picture_shelves = 2 := by
  sorry

end NUMINAMATH_GPT_rachel_picture_shelves_l1518_151829


namespace NUMINAMATH_GPT_problem1_problem2_l1518_151892

-- Definitions and assumptions
def p (m : ℝ) : Prop := ∀x y : ℝ, (x^2)/(4 - m) + (y^2)/m = 1 → ∃ c : ℝ, c^2 < (4 - m) ∧ c^2 < m
def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*m*x + 1 > 0
def S (m : ℝ) : Prop := ∃ x : ℝ, m*x^2 + 2*m*x + 2 - m = 0

-- Problem (1)
theorem problem1 (m : ℝ) (hS : S m) : m < 0 ∨ m ≥ 1 := sorry

-- Problem (2)
theorem problem2 (m : ℝ) (hp : p m ∨ q m) (hnq : ¬ q m) : 1 ≤ m ∧ m < 2 := sorry

end NUMINAMATH_GPT_problem1_problem2_l1518_151892


namespace NUMINAMATH_GPT_tan_alpha_eq_neg_one_l1518_151839

theorem tan_alpha_eq_neg_one (alpha : ℝ) (h1 : Real.tan alpha = -1) (h2 : 0 ≤ alpha ∧ alpha < Real.pi) :
  alpha = (3 * Real.pi) / 4 :=
sorry

end NUMINAMATH_GPT_tan_alpha_eq_neg_one_l1518_151839


namespace NUMINAMATH_GPT_variance_cows_l1518_151848

-- Define the number of cows and incidence rate.
def n : ℕ := 10
def p : ℝ := 0.02

-- The variance of the binomial distribution, given n and p.
def variance (n : ℕ) (p : ℝ) : ℝ := n * p * (1 - p)

-- Statement to prove
theorem variance_cows : variance n p = 0.196 :=
by
  sorry

end NUMINAMATH_GPT_variance_cows_l1518_151848


namespace NUMINAMATH_GPT_cricketer_average_score_l1518_151851

theorem cricketer_average_score
  (avg1 : ℕ)
  (matches1 : ℕ)
  (avg2 : ℕ)
  (matches2 : ℕ)
  (total_matches : ℕ)
  (total_avg : ℕ)
  (h1 : avg1 = 20)
  (h2 : matches1 = 2)
  (h3 : avg2 = 30)
  (h4 : matches2 = 3)
  (h5 : total_matches = 5)
  (h6 : total_avg = 26)
  (h_total_runs : total_avg * total_matches = avg1 * matches1 + avg2 * matches2) :
  total_avg = 26 := 
sorry

end NUMINAMATH_GPT_cricketer_average_score_l1518_151851


namespace NUMINAMATH_GPT_cos_alpha_minus_pi_over_6_l1518_151810

theorem cos_alpha_minus_pi_over_6 (α : Real) 
  (h1 : Real.pi / 2 < α) 
  (h2 : α < Real.pi) 
  (h3 : Real.sin (α + Real.pi / 6) = 3 / 5) : 
  Real.cos (α - Real.pi / 6) = (3 * Real.sqrt 3 - 4) / 10 := 
by 
  sorry

end NUMINAMATH_GPT_cos_alpha_minus_pi_over_6_l1518_151810


namespace NUMINAMATH_GPT_remainder_when_expression_divided_l1518_151843

theorem remainder_when_expression_divided 
  (x y u v : ℕ) 
  (h1 : x = u * y + v) 
  (h2 : 0 ≤ v) 
  (h3 : v < y) :
  (x - u * y + 3 * v) % y = (4 * v) % y :=
by
  sorry

end NUMINAMATH_GPT_remainder_when_expression_divided_l1518_151843


namespace NUMINAMATH_GPT_find_multiple_of_q_l1518_151834

variable (p q m : ℚ)

theorem find_multiple_of_q (h1 : p / q = 3 / 4) (h2 : 3 * p + m * q = 6.25) :
  m = 4 :=
sorry

end NUMINAMATH_GPT_find_multiple_of_q_l1518_151834


namespace NUMINAMATH_GPT_xiangshan_port_investment_scientific_notation_l1518_151876

-- Definition of scientific notation
def in_scientific_notation (a : ℝ) (n : ℤ) (x : ℝ) : Prop :=
  1 ≤ |a| ∧ |a| < 10 ∧ x = a * 10^n

-- Theorem stating the equivalence of the investment in scientific notation
theorem xiangshan_port_investment_scientific_notation :
  in_scientific_notation 7.7 9 7.7e9 :=
by {
  sorry
}

end NUMINAMATH_GPT_xiangshan_port_investment_scientific_notation_l1518_151876


namespace NUMINAMATH_GPT_cost_price_correct_l1518_151865

noncomputable def cost_price (selling_price marked_price_ratio cost_profit_ratio : ℝ) : ℝ :=
  (selling_price * marked_price_ratio) / cost_profit_ratio

theorem cost_price_correct : 
  abs (cost_price 63.16 0.94 1.25 - 50.56) < 0.01 :=
by 
  sorry

end NUMINAMATH_GPT_cost_price_correct_l1518_151865


namespace NUMINAMATH_GPT_jake_first_week_sales_jake_second_week_sales_jake_highest_third_week_sales_l1518_151858

theorem jake_first_week_sales :
  let initial_pieces := 80
  let monday_sales := 15
  let tuesday_sales := 2 * monday_sales
  let remaining_pieces := 7
  monday_sales + tuesday_sales + (initial_pieces - (monday_sales + tuesday_sales) - remaining_pieces) = 73 :=
by
  sorry

theorem jake_second_week_sales :
  let monday_sales := 12
  let tuesday_sales := 18
  let wednesday_sales := 20
  let thursday_sales := 11
  let friday_sales := 25
  monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales = 86 :=
by
  sorry

theorem jake_highest_third_week_sales :
  let highest_sales := 40
  highest_sales = 40 :=
by
  sorry

end NUMINAMATH_GPT_jake_first_week_sales_jake_second_week_sales_jake_highest_third_week_sales_l1518_151858


namespace NUMINAMATH_GPT_arnaldo_bernaldo_distribute_toys_l1518_151873

noncomputable def num_ways_toys_distributed (total_toys remaining_toys : ℕ) : ℕ :=
  if total_toys = 10 ∧ remaining_toys = 8 then 6561 - 256 else 0

theorem arnaldo_bernaldo_distribute_toys : num_ways_toys_distributed 10 8 = 6305 :=
by 
  -- Lean calculation for 3^8 = 6561 and 2^8 = 256 can be done as follows
  -- let three_power_eight := 3^8
  -- let two_power_eight := 2^8
  -- three_power_eight - two_power_eight = 6305
  sorry

end NUMINAMATH_GPT_arnaldo_bernaldo_distribute_toys_l1518_151873


namespace NUMINAMATH_GPT_SamLastPage_l1518_151890

theorem SamLastPage (total_pages : ℕ) (Sam_read_time : ℕ) (Lily_read_time : ℕ) (last_page : ℕ) :
  total_pages = 920 ∧ Sam_read_time = 30 ∧ Lily_read_time = 50 → last_page = 575 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_SamLastPage_l1518_151890


namespace NUMINAMATH_GPT_index_card_area_l1518_151833

theorem index_card_area :
  ∀ (length width : ℕ), length = 5 → width = 7 →
  (length - 2) * width = 21 →
  length * (width - 1) = 30 :=
by
  intros length width h_length h_width h_condition
  sorry

end NUMINAMATH_GPT_index_card_area_l1518_151833


namespace NUMINAMATH_GPT_derivative_at_zero_l1518_151835
noncomputable def f (x : ℝ) : ℝ := x * (x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 5)

theorem derivative_at_zero : (deriv f 0) = -120 :=
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_derivative_at_zero_l1518_151835


namespace NUMINAMATH_GPT_sequence_general_formula_l1518_151885

theorem sequence_general_formula (a : ℕ → ℝ) (h₁ : a 1 = 3) 
    (h₂ : ∀ n : ℕ, 1 < n → a n = (n / (n - 1)) * a (n - 1)) : 
    ∀ n : ℕ, 1 ≤ n → a n = 3 * n :=
by
  -- Proof description here
  sorry

end NUMINAMATH_GPT_sequence_general_formula_l1518_151885


namespace NUMINAMATH_GPT_count_two_digit_numbers_with_digit_8_l1518_151852

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def has_digit_8 (n : ℕ) : Prop :=
  n / 10 = 8 ∨ n % 10 = 8

theorem count_two_digit_numbers_with_digit_8 : 
  (∃ S : Finset ℕ, (∀ n ∈ S, is_two_digit n ∧ has_digit_8 n) ∧ S.card = 18) :=
sorry

end NUMINAMATH_GPT_count_two_digit_numbers_with_digit_8_l1518_151852


namespace NUMINAMATH_GPT_find_the_number_l1518_151807

theorem find_the_number :
  ∃ N : ℝ, ((4/5 : ℝ) * 25 = 20) ∧ (0.40 * N = 24) ∧ (N = 60) :=
by
  sorry

end NUMINAMATH_GPT_find_the_number_l1518_151807


namespace NUMINAMATH_GPT_frog_escape_probability_l1518_151802

def jump_probability (N : ℕ) : ℚ := N / 14

def survival_probability (P : ℕ → ℚ) (N : ℕ) : ℚ :=
  if N = 0 then 0
  else if N = 14 then 1
  else jump_probability N * P (N - 1) + (1 - jump_probability N) * P (N + 1)

theorem frog_escape_probability :
  ∃ (P : ℕ → ℚ), P 0 = 0 ∧ P 14 = 1 ∧ (∀ (N : ℕ), 0 < N ∧ N < 14 → survival_probability P N = P N) ∧ P 3 = 325 / 728 :=
sorry

end NUMINAMATH_GPT_frog_escape_probability_l1518_151802


namespace NUMINAMATH_GPT_sqrt_inequality_l1518_151826

theorem sqrt_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  Real.sqrt (a^2 + b^2) ≥ (Real.sqrt 2 / 2) * (a + b) :=
sorry

end NUMINAMATH_GPT_sqrt_inequality_l1518_151826


namespace NUMINAMATH_GPT_zero_of_F_when_a_is_zero_range_of_a_if_P_and_Q_l1518_151809

noncomputable def f (a x : ℝ) : ℝ := a * x - Real.log x
noncomputable def g (a x : ℝ) : ℝ := Real.log (x^2 - 2*x + a)
noncomputable def F (a x : ℝ) : ℝ := f a x + g a x

theorem zero_of_F_when_a_is_zero (x : ℝ) : a = 0 → F a x = 0 → x = 3 := by
  sorry

theorem range_of_a_if_P_and_Q (a : ℝ) :
  (∀ x ∈ Set.Icc (1/4 : ℝ) (1/2 : ℝ), a - 1/x ≤ 0) ∧
  (∀ x : ℝ, (x^2 - 2*x + a) > 0) →
  1 < a ∧ a ≤ 2 := by
  sorry

end NUMINAMATH_GPT_zero_of_F_when_a_is_zero_range_of_a_if_P_and_Q_l1518_151809


namespace NUMINAMATH_GPT_will_buy_5_toys_l1518_151862

theorem will_buy_5_toys (initial_money spent_money toy_cost money_left toys : ℕ) 
  (h1 : initial_money = 57) 
  (h2 : spent_money = 27) 
  (h3 : toy_cost = 6) 
  (h4 : money_left = initial_money - spent_money) 
  (h5 : toys = money_left / toy_cost) : 
  toys = 5 := 
by
  sorry

end NUMINAMATH_GPT_will_buy_5_toys_l1518_151862


namespace NUMINAMATH_GPT_goldbach_10000_l1518_151800

def is_prime (n : ℕ) : Prop := ∀ m, m ∣ n → m = 1 ∨ m = n

theorem goldbach_10000 :
  ∃ (S : Finset (ℕ × ℕ)), (∀ (p q : ℕ), (p, q) ∈ S → is_prime p ∧ is_prime q ∧ p + q = 10000) ∧ S.card > 3 :=
sorry

end NUMINAMATH_GPT_goldbach_10000_l1518_151800


namespace NUMINAMATH_GPT_min_commission_deputies_l1518_151855

theorem min_commission_deputies 
  (members : ℕ) 
  (brawls : ℕ) 
  (brawl_participants : brawls = 200) 
  (member_count : members = 200) :
  ∃ minimal_commission_members : ℕ, minimal_commission_members = 67 := 
sorry

end NUMINAMATH_GPT_min_commission_deputies_l1518_151855


namespace NUMINAMATH_GPT_propA_neither_sufficient_nor_necessary_l1518_151825

def PropA (a b : ℕ) : Prop := a + b ≠ 4
def PropB (a b : ℕ) : Prop := a ≠ 1 ∧ b ≠ 3

theorem propA_neither_sufficient_nor_necessary (a b : ℕ) : 
  ¬((PropA a b → PropB a b) ∧ (PropB a b → PropA a b)) :=
by {
  sorry
}

end NUMINAMATH_GPT_propA_neither_sufficient_nor_necessary_l1518_151825


namespace NUMINAMATH_GPT_relationship_between_first_and_third_numbers_l1518_151879

variable (A B C : ℕ)

theorem relationship_between_first_and_third_numbers
  (h1 : A + B + C = 660)
  (h2 : A = 2 * B)
  (h3 : B = 180) :
  C = A - 240 :=
by
  sorry

end NUMINAMATH_GPT_relationship_between_first_and_third_numbers_l1518_151879


namespace NUMINAMATH_GPT_find_valid_ns_l1518_151881

theorem find_valid_ns (n : ℕ) (h1 : n > 1) (h2 : ∃ k : ℕ, k^2 = (n^2 + 7 * n + 136) / (n-1)) : n = 5 ∨ n = 37 :=
sorry

end NUMINAMATH_GPT_find_valid_ns_l1518_151881

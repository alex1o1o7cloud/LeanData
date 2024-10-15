import Mathlib

namespace NUMINAMATH_GPT_solve_congruence_y37_x3_11_l1886_188672

theorem solve_congruence_y37_x3_11 (p : ℕ) (hp_pr : Nat.Prime p) (hp_le100 : p ≤ 100) : 
  ∃ (x y : ℕ), y^37 ≡ x^3 + 11 [MOD p] := 
sorry

end NUMINAMATH_GPT_solve_congruence_y37_x3_11_l1886_188672


namespace NUMINAMATH_GPT_semicircle_problem_l1886_188636

open Real

theorem semicircle_problem (r : ℝ) (N : ℕ)
  (h1 : True) -- condition 1: There are N small semicircles each with radius r.
  (h2 : True) -- condition 2: The diameter of the large semicircle is 2Nr.
  (h3 : (N * (π * r^2) / 2) / ((π * (N^2 * r^2) / 2) - (N * (π * r^2) / 2)) = (1 : ℝ) / 12) -- given ratio A / B = 1 / 12 
  : N = 13 :=
sorry

end NUMINAMATH_GPT_semicircle_problem_l1886_188636


namespace NUMINAMATH_GPT_daughter_work_alone_12_days_l1886_188663

/-- Given a man, his wife, and their daughter working together on a piece of work. The man can complete the work in 4 days, the wife in 6 days, and together with their daughter, they can complete it in 2 days. Prove that the daughter alone would take 12 days to complete the work. -/
theorem daughter_work_alone_12_days (h1 : (1/4 : ℝ) + (1/6) + D = 1/2) : D = 1/12 :=
by
  sorry

end NUMINAMATH_GPT_daughter_work_alone_12_days_l1886_188663


namespace NUMINAMATH_GPT_rhombus_area_l1886_188611

theorem rhombus_area (side_length : ℝ) (d1_diff_d2 : ℝ) 
  (h_side_length : side_length = Real.sqrt 104) 
  (h_d1_diff_d2 : d1_diff_d2 = 10) : 
  (1 / 2) * (2 * Real.sqrt 104 - d1_diff_d2) * (d1_diff_d2 + 2 * Real.sqrt 104) = 79.17 :=
by
  sorry

end NUMINAMATH_GPT_rhombus_area_l1886_188611


namespace NUMINAMATH_GPT_twelve_times_y_plus_three_half_quarter_l1886_188637

theorem twelve_times_y_plus_three_half_quarter (y : ℝ) : 
  (1 / 2) * (1 / 4) * (12 * y + 3) = (3 * y) / 2 + 3 / 8 :=
by sorry

end NUMINAMATH_GPT_twelve_times_y_plus_three_half_quarter_l1886_188637


namespace NUMINAMATH_GPT_right_triangle_inequality_l1886_188660

theorem right_triangle_inequality (a b c : ℝ) (h1 : a^2 + b^2 = c^2) (h2 : b > a) (h3 : b / a < 2) :
  a^2 / (b^2 + c^2) + b^2 / (a^2 + c^2) > 4 / 9 :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_inequality_l1886_188660


namespace NUMINAMATH_GPT_g_675_eq_42_l1886_188689

theorem g_675_eq_42 
  (g : ℕ → ℕ) 
  (h_mul : ∀ x y : ℕ, x > 0 → y > 0 → g (x * y) = g x + g y) 
  (h_g15 : g 15 = 18) 
  (h_g45 : g 45 = 24) : g 675 = 42 :=
by
  sorry

end NUMINAMATH_GPT_g_675_eq_42_l1886_188689


namespace NUMINAMATH_GPT_exists_multiple_l1886_188615

theorem exists_multiple (n : ℕ) (a : Fin (n + 1) → ℕ) 
  (h : ∀ i, a i > 0) 
  (h2 : ∀ i, a i ≤ 2 * n) : 
  ∃ i j : Fin (n + 1), i ≠ j ∧ (a i ∣ a j ∨ a j ∣ a i) :=
by
sorry

end NUMINAMATH_GPT_exists_multiple_l1886_188615


namespace NUMINAMATH_GPT_scale_model_height_is_correct_l1886_188679

noncomputable def height_of_scale_model (h_real : ℝ) (V_real : ℝ) (V_scale : ℝ) : ℝ :=
  h_real / (V_real / V_scale)^(1/3:ℝ)

theorem scale_model_height_is_correct :
  height_of_scale_model 90 500000 0.2 = 0.66 :=
by
  sorry

end NUMINAMATH_GPT_scale_model_height_is_correct_l1886_188679


namespace NUMINAMATH_GPT_other_number_in_product_l1886_188633

theorem other_number_in_product (w : ℕ) (n : ℕ) (hw_pos : 0 < w) (n_factor : Nat.lcm (2^5) (Nat.gcd  864 w) = 2^5 * 3^3) (h_w : w = 144) : n = 6 :=
by
  -- proof would go here
  sorry

end NUMINAMATH_GPT_other_number_in_product_l1886_188633


namespace NUMINAMATH_GPT_alice_paid_24_percent_l1886_188694

theorem alice_paid_24_percent (P : ℝ) (h1 : P > 0) :
  let MP := 0.60 * P
  let price_paid := 0.40 * MP
  (price_paid / P) * 100 = 24 :=
by
  sorry

end NUMINAMATH_GPT_alice_paid_24_percent_l1886_188694


namespace NUMINAMATH_GPT_maximum_value_when_t_is_2_solve_for_t_when_maximum_value_is_2_l1886_188668

def f (x : ℝ) (t : ℝ) : ℝ := abs (2 * x - 1) - abs (t * x + 3)

theorem maximum_value_when_t_is_2 :
  ∃ x : ℝ, (f x 2) ≤ 4 ∧ ∀ y : ℝ, (f y 2) ≤ (f x 2) := sorry

theorem solve_for_t_when_maximum_value_is_2 :
  ∃ t : ℝ, t > 0 ∧ (∀ x : ℝ, (f x t) ≤ 2 ∧ (∃ y : ℝ, (f y t) = 2)) → t = 6 := sorry

end NUMINAMATH_GPT_maximum_value_when_t_is_2_solve_for_t_when_maximum_value_is_2_l1886_188668


namespace NUMINAMATH_GPT_triangle_inequality_l1886_188616

theorem triangle_inequality (a b c : ℝ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) : True :=
  sorry

end NUMINAMATH_GPT_triangle_inequality_l1886_188616


namespace NUMINAMATH_GPT_evaluation_l1886_188626
-- Import the entire Mathlib library

-- Define the operations triangle and nabla
def triangle (a b : ℕ) : ℕ := 3 * a + 2 * b
def nabla (a b : ℕ) : ℕ := 2 * a + 3 * b

-- The proof statement
theorem evaluation : triangle 2 (nabla 3 4) = 42 :=
by
  -- Provide a placeholder for the proof
  sorry

end NUMINAMATH_GPT_evaluation_l1886_188626


namespace NUMINAMATH_GPT_percentage_of_managers_l1886_188666

theorem percentage_of_managers (P : ℝ) :
  (200 : ℝ) * (P / 100) - 99.99999999999991 = 0.98 * (200 - 99.99999999999991) →
  P = 99 := 
sorry

end NUMINAMATH_GPT_percentage_of_managers_l1886_188666


namespace NUMINAMATH_GPT_john_total_cost_l1886_188664

-- Define the costs and usage details
def base_cost : ℕ := 25
def cost_per_text_cent : ℕ := 10
def cost_per_extra_minute_cent : ℕ := 15
def included_hours : ℕ := 20
def texts_sent : ℕ := 150
def hours_talked : ℕ := 22

-- Prove that the total cost John had to pay is $58
def total_cost_john : ℕ :=
  let base_cost_dollars := base_cost
  let text_cost_dollars := (texts_sent * cost_per_text_cent) / 100
  let extra_minutes := (hours_talked - included_hours) * 60
  let extra_minutes_cost_dollars := (extra_minutes * cost_per_extra_minute_cent) / 100
  base_cost_dollars + text_cost_dollars + extra_minutes_cost_dollars

theorem john_total_cost (h1 : base_cost = 25)
                        (h2 : cost_per_text_cent = 10)
                        (h3 : cost_per_extra_minute_cent = 15)
                        (h4 : included_hours = 20)
                        (h5 : texts_sent = 150)
                        (h6 : hours_talked = 22) : 
  total_cost_john = 58 := by
  sorry

end NUMINAMATH_GPT_john_total_cost_l1886_188664


namespace NUMINAMATH_GPT_geometric_sequence_a1_l1886_188699

theorem geometric_sequence_a1 (a1 a2 a3 S3 : ℝ) (q : ℝ)
  (h1 : S3 = a1 + (1 / 2) * a2)
  (h2 : a3 = (1 / 4))
  (h3 : S3 = a1 * (1 + q + q^2))
  (h4 : a2 = a1 * q)
  (h5 : a3 = a1 * q^2) :
  a1 = 1 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_a1_l1886_188699


namespace NUMINAMATH_GPT_astronaut_total_days_l1886_188675

-- Definitions of the regular and leap seasons.
def regular_season_days := 49
def leap_season_days := 51

-- Definition of the number of days in different types of years.
def days_in_regular_year := 2 * regular_season_days + 3 * leap_season_days
def days_in_first_3_years := 2 * regular_season_days + 3 * (leap_season_days + 1)
def days_in_years_7_to_9 := 2 * regular_season_days + 3 * (leap_season_days + 2)

-- Calculation for visits.
def first_visit := regular_season_days
def second_visit := 2 * regular_season_days + 3 * (leap_season_days + 1)
def third_visit := 3 * (2 * regular_season_days + 3 * (leap_season_days + 1))
def fourth_visit := 4 * days_in_regular_year + 3 * days_in_first_3_years + 3 * days_in_years_7_to_9

-- Total days spent.
def total_days := first_visit + second_visit + third_visit + fourth_visit

-- The proof statement.
theorem astronaut_total_days : total_days = 3578 :=
by
  -- We place a sorry here to skip the proof.
  sorry

end NUMINAMATH_GPT_astronaut_total_days_l1886_188675


namespace NUMINAMATH_GPT_min_value_x_plus_inv_x_l1886_188678

theorem min_value_x_plus_inv_x (x : ℝ) (hx : x > 0) : ∃ y, (y = x + 1/x) ∧ (∀ z, z = x + 1/x → z ≥ 2) :=
by
  sorry

end NUMINAMATH_GPT_min_value_x_plus_inv_x_l1886_188678


namespace NUMINAMATH_GPT_sequence_solution_l1886_188639

theorem sequence_solution (a : ℕ → ℝ) (h1 : a 1 = 2) (h2 : a 2 = 1)
  (h_rec : ∀ n ≥ 2, 2 / a n = 1 / a (n + 1) + 1 / a (n - 1)) :
  ∀ n, a n = 2 / n :=
by
  sorry

end NUMINAMATH_GPT_sequence_solution_l1886_188639


namespace NUMINAMATH_GPT_total_plums_picked_l1886_188623

-- Conditions
def Melanie_plums : ℕ := 4
def Dan_plums : ℕ := 9
def Sally_plums : ℕ := 3

-- Proof statement
theorem total_plums_picked : Melanie_plums + Dan_plums + Sally_plums = 16 := by
  sorry

end NUMINAMATH_GPT_total_plums_picked_l1886_188623


namespace NUMINAMATH_GPT_supermarket_problem_l1886_188608

-- Define that type A costs x yuan and type B costs y yuan
def cost_price_per_item (x y : ℕ) : Prop :=
  (10 * x + 8 * y = 880) ∧ (2 * x + 5 * y = 380)

-- Define purchasing plans with the conditions described
def purchasing_plans (a : ℕ) : Prop :=
  ∀ a : ℕ, 24 ≤ a ∧ a ≤ 26

theorem supermarket_problem : 
  (∃ x y, cost_price_per_item x y ∧ x = 40 ∧ y = 60) ∧ 
  (∃ n, purchasing_plans n ∧ n = 3) :=
by
  sorry

end NUMINAMATH_GPT_supermarket_problem_l1886_188608


namespace NUMINAMATH_GPT_number_of_non_empty_proper_subsets_of_A_l1886_188685

noncomputable def A : Set ℤ := { x : ℤ | -1 < x ∧ x ≤ 2 }

theorem number_of_non_empty_proper_subsets_of_A : 
  (∃ (A : Set ℤ), A = { x : ℤ | -1 < x ∧ x ≤ 2 }) → 
  ∃ (n : ℕ), n = 6 := by
  sorry

end NUMINAMATH_GPT_number_of_non_empty_proper_subsets_of_A_l1886_188685


namespace NUMINAMATH_GPT_locus_of_Q_l1886_188671

def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2/a^2 + y^2/b^2 = 1

def A_vertice (a b : ℝ) (x y : ℝ) : Prop :=
  (x = a ∧ y = 0) ∨ (x = -a ∧ y = 0)

def chord_parallel_y_axis (x : ℝ) : Prop :=
  -- Assuming chord's x coordinate is given
  True

def lines_intersect_at_Q (a b Qx Qy : ℝ) : Prop :=
  ∃ x y : ℝ, ellipse a b x y ∧
  A_vertice a b x y ∧
  chord_parallel_y_axis x ∧
  (
    ( (Qy - y) / (Qx - (-a)) = (Qy - 0) / (Qx - a) ) ∨ -- A'P slope-comp
    ( (Qy - (-y)) / (Qx - a) = (Qy - 0) / (Qx - (-a)) ) -- AP' slope-comp
  )

theorem locus_of_Q (a b Qx Qy : ℝ) :
  (lines_intersect_at_Q a b Qx Qy) →
  (Qx^2 / a^2 - Qy^2 / b^2 = 1) := by
  sorry

end NUMINAMATH_GPT_locus_of_Q_l1886_188671


namespace NUMINAMATH_GPT_intersection_on_circle_l1886_188659

def parabola1 (X : ℝ) : ℝ := X^2 + X - 41
def parabola2 (Y : ℝ) : ℝ := Y^2 + Y - 40

theorem intersection_on_circle (X Y : ℝ) :
  parabola1 X = Y ∧ parabola2 Y = X → X^2 + Y^2 = 81 :=
by {
  sorry
}

end NUMINAMATH_GPT_intersection_on_circle_l1886_188659


namespace NUMINAMATH_GPT_katie_flour_l1886_188617

theorem katie_flour (x : ℕ) (h1 : x + (x + 2) = 8) : x = 3 := 
by
  sorry

end NUMINAMATH_GPT_katie_flour_l1886_188617


namespace NUMINAMATH_GPT_probability_of_winning_five_tickets_l1886_188605

def probability_of_winning_one_ticket := 1 / 10000000
def number_of_tickets_bought := 5

theorem probability_of_winning_five_tickets : 
  (number_of_tickets_bought * probability_of_winning_one_ticket) = 5 / 10000000 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_winning_five_tickets_l1886_188605


namespace NUMINAMATH_GPT_percentage_of_Y_pay_X_is_paid_correct_l1886_188607

noncomputable def percentage_of_Y_pay_X_is_paid
  (total_pay : ℝ) (Y_pay : ℝ) : ℝ :=
  let X_pay := total_pay - Y_pay
  (X_pay / Y_pay) * 100

theorem percentage_of_Y_pay_X_is_paid_correct :
  percentage_of_Y_pay_X_is_paid 700 318.1818181818182 = 120 := 
by
  unfold percentage_of_Y_pay_X_is_paid
  sorry

end NUMINAMATH_GPT_percentage_of_Y_pay_X_is_paid_correct_l1886_188607


namespace NUMINAMATH_GPT_sum_of_remainders_11111k_43210_eq_141_l1886_188661

theorem sum_of_remainders_11111k_43210_eq_141 :
  (List.sum (List.map (fun k => (11111 * k + 43210) % 31) [0, 1, 2, 3, 4, 5])) = 141 :=
by
  -- Proof is omitted: sorry
  sorry

end NUMINAMATH_GPT_sum_of_remainders_11111k_43210_eq_141_l1886_188661


namespace NUMINAMATH_GPT_set_operation_empty_l1886_188643

-- Definition of the universal set I, and sets P and Q with the given properties
variable {I : Set ℕ} -- Universal set
variable {P Q : Set ℕ} -- Non-empty sets with P ⊂ Q ⊂ I
variable (hPQ : P ⊂ Q) (hQI : Q ⊂ I)

-- Prove the set operation expression that results in the empty set
theorem set_operation_empty :
  ∃ (P Q : Set ℕ), P ⊂ Q ∧ Q ⊂ I ∧ P ≠ ∅ ∧ Q ≠ ∅ → 
  P ∩ (I \ Q) = ∅ :=
by
  sorry

end NUMINAMATH_GPT_set_operation_empty_l1886_188643


namespace NUMINAMATH_GPT_volume_ratio_l1886_188635

def cube_volume (side_length : ℝ) : ℝ :=
  side_length ^ 3

theorem volume_ratio : 
  let a := (4 : ℝ) / 12   -- 4 inches converted to feet
  let b := (2 : ℝ)       -- 2 feet
  cube_volume a / cube_volume b = 1 / 216 :=
by
  sorry

end NUMINAMATH_GPT_volume_ratio_l1886_188635


namespace NUMINAMATH_GPT_eval_fraction_l1886_188658

theorem eval_fraction : (3 : ℚ) / (2 - 5 / 4) = 4 := 
by 
  sorry

end NUMINAMATH_GPT_eval_fraction_l1886_188658


namespace NUMINAMATH_GPT_freshman_count_630_l1886_188606

-- Define the variables and conditions
variables (f o j s : ℕ)
variable (total_students : ℕ)

-- Define the ratios given in the problem
def freshmen_to_sophomore : Prop := f = (5 * o) / 4
def sophomore_to_junior : Prop := j = (8 * o) / 7
def junior_to_senior : Prop := s = (7 * j) / 9

-- Total number of students condition
def total_students_condition : Prop := f + o + j + s = total_students

theorem freshman_count_630
  (h1 : freshmen_to_sophomore f o)
  (h2 : sophomore_to_junior o j)
  (h3 : junior_to_senior j s)
  (h4 : total_students_condition f o j s 2158) :
  f = 630 :=
sorry

end NUMINAMATH_GPT_freshman_count_630_l1886_188606


namespace NUMINAMATH_GPT_evaluate_expression_l1886_188625

theorem evaluate_expression (a b : ℕ) (h1 : a = 3) (h2 : b = 2) : (a^b)^b + (b^a)^a = 593 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1886_188625


namespace NUMINAMATH_GPT_range_a_l1886_188650

theorem range_a (a : ℝ) : 
  (∃ x : ℝ, 0 < x ∧ x < 1 ∧ a * x^2 - x - 1 = 0 ∧ 
  ∀ y : ℝ, (0 < y ∧ y < 1 ∧ a * y^2 - y - 1 = 0 → y = x)) ↔ a > 2 :=
by
  sorry

end NUMINAMATH_GPT_range_a_l1886_188650


namespace NUMINAMATH_GPT_quadratic_trinomial_bound_l1886_188682

theorem quadratic_trinomial_bound (a b : ℤ) (f : ℝ → ℝ)
  (h_def : ∀ x : ℝ, f x = x^2 + a * x + b)
  (h_bound : ∀ x : ℝ, f x ≥ -9 / 10) :
  ∀ x : ℝ, f x ≥ -1 / 4 :=
sorry

end NUMINAMATH_GPT_quadratic_trinomial_bound_l1886_188682


namespace NUMINAMATH_GPT_complement_of_M_in_U_l1886_188632

namespace SetComplements

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 3, 5}
def complement_U_M : Set ℕ := U \ M

theorem complement_of_M_in_U :
  complement_U_M = {2, 4, 6} :=
by
  sorry

end SetComplements

end NUMINAMATH_GPT_complement_of_M_in_U_l1886_188632


namespace NUMINAMATH_GPT_cylinder_radius_and_remaining_space_l1886_188600

theorem cylinder_radius_and_remaining_space 
  (cone_radius : ℝ) (cone_height : ℝ) 
  (cylinder_radius : ℝ) (cylinder_height : ℝ) :
  cone_radius = 8 →
  cone_height = 20 →
  cylinder_height = 2 * cylinder_radius →
  (20 - 2 * cylinder_radius) / cylinder_radius = 20 / 8 →
  (cylinder_radius = 40 / 9 ∧ (cone_height - cylinder_height) = 100 / 9) :=
by
  intros cone_radius_8 cone_height_20 cylinder_height_def similarity_eq
  sorry

end NUMINAMATH_GPT_cylinder_radius_and_remaining_space_l1886_188600


namespace NUMINAMATH_GPT_nearest_multiple_to_457_divisible_by_11_l1886_188630

theorem nearest_multiple_to_457_divisible_by_11 : ∃ n : ℤ, (n % 11 = 0) ∧ (abs (457 - n) = 5) :=
by
  sorry

end NUMINAMATH_GPT_nearest_multiple_to_457_divisible_by_11_l1886_188630


namespace NUMINAMATH_GPT_total_voters_in_districts_l1886_188640

theorem total_voters_in_districts : 
  ∀ (D1 D2 D3 : ℕ),
  (D1 = 322) →
  (D2 = D3 - 19) →
  (D3 = 2 * D1) →
  (D1 + D2 + D3 = 1591) :=
by
  intros D1 D2 D3 h1 h2 h3
  sorry

end NUMINAMATH_GPT_total_voters_in_districts_l1886_188640


namespace NUMINAMATH_GPT_money_needed_l1886_188609

def phone_cost : ℕ := 1300
def mike_fraction : ℚ := 0.4

theorem money_needed : mike_fraction * phone_cost + 780 = phone_cost := by
  sorry

end NUMINAMATH_GPT_money_needed_l1886_188609


namespace NUMINAMATH_GPT_Bill_threw_more_sticks_l1886_188642

-- Definitions based on the given conditions
def Ted_sticks : ℕ := 10
def Ted_rocks : ℕ := 10
def Ted_double_Bill_rocks (R : ℕ) : Prop := Ted_rocks = 2 * R
def Bill_total_objects (S R : ℕ) : Prop := S + R = 21

-- The theorem stating Bill throws 6 more sticks than Ted
theorem Bill_threw_more_sticks (S R : ℕ) (h1 : Ted_double_Bill_rocks R) (h2 : Bill_total_objects S R) : S - Ted_sticks = 6 :=
by
  -- Definitions and conditions are loaded here
  sorry

end NUMINAMATH_GPT_Bill_threw_more_sticks_l1886_188642


namespace NUMINAMATH_GPT_find_a_l1886_188692

theorem find_a (a : ℤ) :
  (∃! x : ℤ, |a * x + a + 2| < 2) ↔ a = 3 ∨ a = -3 := 
sorry

end NUMINAMATH_GPT_find_a_l1886_188692


namespace NUMINAMATH_GPT_quadratic_roots_l1886_188618

theorem quadratic_roots (m : ℝ) : 
  (m > 0 → ∃ a b : ℝ, a ≠ b ∧ (a^2 + a - 2 = m) ∧ (b^2 + b - 2 = m)) ∧ 
  ¬(m = 0 ∧ ∃ a : ℝ, (a^2 + a - 2 = m) ∧ (a^2 + a - 2 = m)) ∧ 
  ¬(m < 0 ∧ ¬ ∃ a b : ℝ, (a^2 + a - 2 = m) ∧ (b^2 + b - 2 = m) ) ∧ 
  ¬(∀ m, ∃ a : ℝ, (a^2 + a - 2 = m)) :=
by 
  sorry

end NUMINAMATH_GPT_quadratic_roots_l1886_188618


namespace NUMINAMATH_GPT_compute_nested_operation_l1886_188681

def my_op (a b : ℚ) : ℚ := (a^2 - b^2) / (1 - a * b)

theorem compute_nested_operation : my_op 1 (my_op 2 (my_op 3 4)) = -18 := by
  sorry

end NUMINAMATH_GPT_compute_nested_operation_l1886_188681


namespace NUMINAMATH_GPT_jimin_initial_candies_l1886_188670

theorem jimin_initial_candies : 
  let candies_given_to_yuna := 25
  let candies_given_to_sister := 13
  candies_given_to_yuna + candies_given_to_sister = 38 := 
  by 
    sorry

end NUMINAMATH_GPT_jimin_initial_candies_l1886_188670


namespace NUMINAMATH_GPT_rent_change_percent_l1886_188687

open Real

noncomputable def elaine_earnings_last_year (E : ℝ) : ℝ :=
E

noncomputable def elaine_rent_last_year (E : ℝ) : ℝ :=
0.2 * E

noncomputable def elaine_earnings_this_year (E : ℝ) : ℝ :=
1.15 * E

noncomputable def elaine_rent_this_year (E : ℝ) : ℝ :=
0.25 * (1.15 * E)

noncomputable def rent_percentage_change (E : ℝ) : ℝ :=
(elaine_rent_this_year E) / (elaine_rent_last_year E) * 100

theorem rent_change_percent (E : ℝ) :
  rent_percentage_change E = 143.75 :=
by
  sorry

end NUMINAMATH_GPT_rent_change_percent_l1886_188687


namespace NUMINAMATH_GPT_negation_of_exists_cube_pos_l1886_188601

theorem negation_of_exists_cube_pos :
  (¬ (∃ x : ℝ, x^3 > 0)) ↔ (∀ x : ℝ, x^3 ≤ 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_exists_cube_pos_l1886_188601


namespace NUMINAMATH_GPT_pencils_left_proof_l1886_188662

noncomputable def total_pencils_left (a d : ℕ) : ℕ :=
  let total_initial_pencils : ℕ := 30
  let total_pencils_given_away : ℕ := 15 * a + 105 * d
  total_initial_pencils - total_pencils_given_away

theorem pencils_left_proof (a d : ℕ) :
  total_pencils_left a d = 30 - (15 * a + 105 * d) :=
by
  sorry

end NUMINAMATH_GPT_pencils_left_proof_l1886_188662


namespace NUMINAMATH_GPT_sylvia_carla_together_time_l1886_188680

-- Define the conditions
def sylviaRate := 1 / 45
def carlaRate := 1 / 30

-- Define the combined work rate and the time taken to complete the job together
def combinedRate := sylviaRate + carlaRate
def timeTogether := 1 / combinedRate

-- Theorem stating the desired result
theorem sylvia_carla_together_time : timeTogether = 18 := by
  sorry

end NUMINAMATH_GPT_sylvia_carla_together_time_l1886_188680


namespace NUMINAMATH_GPT_geometric_sequence_b_l1886_188654

theorem geometric_sequence_b (b : ℝ) (r : ℝ) (hb : b > 0)
  (h1 : 10 * r = b)
  (h2 : b * r = 10 / 9)
  (h3 : (10 / 9) * r = 10 / 81) :
  b = 10 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_b_l1886_188654


namespace NUMINAMATH_GPT_mushrooms_gigi_cut_l1886_188621

-- Definitions based on conditions in part a)
def pieces_per_mushroom := 4
def kenny_sprinkled := 38
def karla_sprinkled := 42
def pieces_remaining := 8

-- The total number of pieces is the sum of Kenny's, Karla's, and the remaining pieces.
def total_pieces := kenny_sprinkled + karla_sprinkled + pieces_remaining

-- The number of mushrooms GiGi cut up at the beginning is total_pieces divided by pieces_per_mushroom.
def mushrooms_cut := total_pieces / pieces_per_mushroom

-- The theorem to be proved.
theorem mushrooms_gigi_cut (h1 : pieces_per_mushroom = 4)
                           (h2 : kenny_sprinkled = 38)
                           (h3 : karla_sprinkled = 42)
                           (h4 : pieces_remaining = 8)
                           (h5 : total_pieces = kenny_sprinkled + karla_sprinkled + pieces_remaining)
                           (h6 : mushrooms_cut = total_pieces / pieces_per_mushroom) :
  mushrooms_cut = 22 :=
by
  sorry

end NUMINAMATH_GPT_mushrooms_gigi_cut_l1886_188621


namespace NUMINAMATH_GPT_sandy_hours_per_day_l1886_188627

theorem sandy_hours_per_day (total_hours : ℕ) (days : ℕ) (H : total_hours = 45 ∧ days = 5) : total_hours / days = 9 :=
by
  sorry

end NUMINAMATH_GPT_sandy_hours_per_day_l1886_188627


namespace NUMINAMATH_GPT_arithmetic_sequence_S12_l1886_188603

theorem arithmetic_sequence_S12 (S : ℕ → ℝ) (a : ℕ → ℝ) 
  (h1 : ∀ n : ℕ, S n = n * (a 1 + a n) / 2)
  (hS4 : S 4 = 25) (hS8 : S 8 = 100) : S 12 = 225 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_S12_l1886_188603


namespace NUMINAMATH_GPT_janet_more_cards_than_brenda_l1886_188673

theorem janet_more_cards_than_brenda : ∀ (J B M : ℕ), M = 2 * J → J + B + M = 211 → M = 150 - 40 → J - B = 9 :=
by
  intros J B M h1 h2 h3
  sorry

end NUMINAMATH_GPT_janet_more_cards_than_brenda_l1886_188673


namespace NUMINAMATH_GPT_abs_neg_six_l1886_188695

theorem abs_neg_six : abs (-6) = 6 := by
  sorry

end NUMINAMATH_GPT_abs_neg_six_l1886_188695


namespace NUMINAMATH_GPT_strong_2013_l1886_188644

theorem strong_2013 :
  ∃ x : ℕ, x > 0 ∧ (x ^ (2013 * x) + 1) % (2 ^ 2013) = 0 :=
sorry

end NUMINAMATH_GPT_strong_2013_l1886_188644


namespace NUMINAMATH_GPT_cosine_angle_between_vectors_l1886_188676

noncomputable def vector_cosine (a b : ℝ × ℝ) : ℝ :=
  let dot_product := (a.1 * b.1 + a.2 * b.2)
  let magnitude_a := Real.sqrt (a.1 ^ 2 + a.2 ^ 2)
  let magnitude_b := Real.sqrt (b.1 ^ 2 + b.2 ^ 2)
  dot_product / (magnitude_a * magnitude_b)

theorem cosine_angle_between_vectors : ∀ (k : ℝ), 
  let a := (3, 1)
  let b := (1, 3)
  let c := (k, -2)
  (3 - k) / 3 = 1 →
  vector_cosine a c = Real.sqrt 5 / 5 := by
  intros
  sorry

end NUMINAMATH_GPT_cosine_angle_between_vectors_l1886_188676


namespace NUMINAMATH_GPT_intersection_A_B_l1886_188697

noncomputable def set_A : Set ℝ := { x | 2 ≤ x ∧ x < 4 }
noncomputable def set_B : Set ℝ := { x | 3 ≤ x }

theorem intersection_A_B :
  set_A ∩ set_B = { x | 3 ≤ x ∧ x < 4 } := 
sorry

end NUMINAMATH_GPT_intersection_A_B_l1886_188697


namespace NUMINAMATH_GPT_man_is_older_by_24_l1886_188612

-- Define the conditions as per the given problem
def present_age_son : ℕ := 22
def present_age_man (M : ℕ) : Prop := M + 2 = 2 * (present_age_son + 2)

-- State the problem: Prove that the man is 24 years older than his son
theorem man_is_older_by_24 (M : ℕ) (h : present_age_man M) : M - present_age_son = 24 := 
sorry

end NUMINAMATH_GPT_man_is_older_by_24_l1886_188612


namespace NUMINAMATH_GPT_difference_length_breadth_l1886_188690

theorem difference_length_breadth (B L A : ℕ) (h1 : B = 11) (h2 : A = 21 * B) (h3 : A = L * B) :
  L - B = 10 :=
by
  sorry

end NUMINAMATH_GPT_difference_length_breadth_l1886_188690


namespace NUMINAMATH_GPT_rectangle_area_l1886_188629

theorem rectangle_area :
  ∃ (L B : ℝ), (L - B = 23) ∧ (2 * (L + B) = 206) ∧ (L * B = 2520) :=
sorry

end NUMINAMATH_GPT_rectangle_area_l1886_188629


namespace NUMINAMATH_GPT_total_seniors_is_161_l1886_188652

def total_students : ℕ := 240

def percentage_statistics : ℚ := 0.45
def percentage_geometry : ℚ := 0.35
def percentage_calculus : ℚ := 0.20

def percentage_stats_and_calc : ℚ := 0.10
def percentage_geom_and_calc : ℚ := 0.05

def percentage_seniors_statistics : ℚ := 0.90
def percentage_seniors_geometry : ℚ := 0.60
def percentage_seniors_calculus : ℚ := 0.80

def students_in_statistics : ℚ := percentage_statistics * total_students
def students_in_geometry : ℚ := percentage_geometry * total_students
def students_in_calculus : ℚ := percentage_calculus * total_students

def students_in_stats_and_calc : ℚ := percentage_stats_and_calc * students_in_statistics
def students_in_geom_and_calc : ℚ := percentage_geom_and_calc * students_in_geometry

def unique_students_in_statistics : ℚ := students_in_statistics - students_in_stats_and_calc
def unique_students_in_geometry : ℚ := students_in_geometry - students_in_geom_and_calc
def unique_students_in_calculus : ℚ := students_in_calculus - students_in_stats_and_calc - students_in_geom_and_calc

def seniors_in_statistics : ℚ := percentage_seniors_statistics * unique_students_in_statistics
def seniors_in_geometry : ℚ := percentage_seniors_geometry * unique_students_in_geometry
def seniors_in_calculus : ℚ := percentage_seniors_calculus * unique_students_in_calculus

def total_seniors : ℚ := seniors_in_statistics + seniors_in_geometry + seniors_in_calculus

theorem total_seniors_is_161 : total_seniors = 161 :=
by
  sorry

end NUMINAMATH_GPT_total_seniors_is_161_l1886_188652


namespace NUMINAMATH_GPT_math_problem_l1886_188657

theorem math_problem : 1012^2 - 992^2 - 1009^2 + 995^2 = 12024 := sorry

end NUMINAMATH_GPT_math_problem_l1886_188657


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l1886_188634

noncomputable def given_expression (a : ℝ) : ℝ :=
  (a - 1 - (2 * a - 1) / (a + 1)) / ((a^2 - 4 * a + 4) / (a + 1))

theorem simplify_and_evaluate_expression (a : ℝ) (h : a = 2 + Real.sqrt 3) :
  given_expression a = (2 * Real.sqrt 3 + 3) / 3 :=
by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l1886_188634


namespace NUMINAMATH_GPT_unique_four_digit_number_l1886_188684

theorem unique_four_digit_number (N : ℕ) (a : ℕ) (x : ℕ) :
  (N = 1000 * a + x) ∧ (N = 7 * x) ∧ (100 ≤ x ∧ x ≤ 999) ∧ (1 ≤ a ∧ a ≤ 9) →
  N = 3500 :=
by sorry

end NUMINAMATH_GPT_unique_four_digit_number_l1886_188684


namespace NUMINAMATH_GPT_arithmetic_sequence_geometric_sequence_l1886_188653

-- Arithmetic sequence proof problem
theorem arithmetic_sequence (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : ∀ n, n ≥ 2 → a n - a (n - 1) = 2) :
  ∀ n, a n = 2 * n - 1 :=
by 
  sorry

-- Geometric sequence proof problem
theorem geometric_sequence (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : ∀ n, n ≥ 2 → a n / a (n - 1) = 2) :
  ∀ n, a n = 2 ^ (n - 1) :=
by 
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_geometric_sequence_l1886_188653


namespace NUMINAMATH_GPT_choose_bar_chart_for_comparisons_l1886_188691

/-- 
To easily compare the quantities of various items, one should choose a bar chart 
based on the characteristics of statistical charts.
-/
theorem choose_bar_chart_for_comparisons 
  (chart_type: Type) 
  (is_bar_chart: chart_type → Prop)
  (is_ideal_chart_for_comparison: chart_type → Prop)
  (bar_chart_ideal: ∀ c, is_bar_chart c → is_ideal_chart_for_comparison c) 
  (comparison_chart : chart_type) 
  (h: is_bar_chart comparison_chart): 
  is_ideal_chart_for_comparison comparison_chart := 
by
  exact bar_chart_ideal comparison_chart h

end NUMINAMATH_GPT_choose_bar_chart_for_comparisons_l1886_188691


namespace NUMINAMATH_GPT_points_in_groups_l1886_188674

theorem points_in_groups (n1 n2 : ℕ) (h_total : n1 + n2 = 28) 
  (h_lines_diff : (n1*(n1 - 1) / 2) - (n2*(n2 - 1) / 2) = 81) : 
  (n1 = 17 ∧ n2 = 11) ∨ (n1 = 11 ∧ n2 = 17) :=
by
  sorry

end NUMINAMATH_GPT_points_in_groups_l1886_188674


namespace NUMINAMATH_GPT_coaches_meet_together_l1886_188628

theorem coaches_meet_together (e s n a : ℕ)
  (h₁ : e = 5) (h₂ : s = 3) (h₃ : n = 9) (h₄ : a = 8) :
  Nat.lcm (Nat.lcm e s) (Nat.lcm n a) = 360 :=
by
  sorry

end NUMINAMATH_GPT_coaches_meet_together_l1886_188628


namespace NUMINAMATH_GPT_sector_area_l1886_188619

theorem sector_area (α r : ℝ) (hα : α = π / 3) (hr : r = 2) : 
  1 / 2 * α * r^2 = 2 * π / 3 := 
by 
  rw [hα, hr] 
  simp 
  sorry

end NUMINAMATH_GPT_sector_area_l1886_188619


namespace NUMINAMATH_GPT_total_crew_members_l1886_188620

def num_islands : ℕ := 3
def ships_per_island : ℕ := 12
def crew_per_ship : ℕ := 24

theorem total_crew_members : num_islands * ships_per_island * crew_per_ship = 864 := by
  sorry

end NUMINAMATH_GPT_total_crew_members_l1886_188620


namespace NUMINAMATH_GPT_intersection_sets_l1886_188667

def setA : Set ℝ := { x | -1 ≤ 2 * x + 1 ∧ 2 * x + 1 ≤ 3 }
def setB : Set ℝ := { x | (x - 3) / (2 * x) ≤ 0 }

theorem intersection_sets (x : ℝ) : x ∈ setA ∧ x ∈ setB ↔ 0 < x ∧ x ≤ 1 := by
  sorry

end NUMINAMATH_GPT_intersection_sets_l1886_188667


namespace NUMINAMATH_GPT_yuna_solved_problems_l1886_188665

def yuna_problems_per_day : ℕ := 8
def days_per_week : ℕ := 7
def yuna_weekly_problems : ℕ := 56

theorem yuna_solved_problems :
  yuna_problems_per_day * days_per_week = yuna_weekly_problems := by
  sorry

end NUMINAMATH_GPT_yuna_solved_problems_l1886_188665


namespace NUMINAMATH_GPT_time_to_pass_tree_l1886_188656

noncomputable def length_of_train : ℝ := 275
noncomputable def speed_in_kmh : ℝ := 90
noncomputable def speed_in_m_per_s : ℝ := speed_in_kmh * (5 / 18)

theorem time_to_pass_tree : (length_of_train / speed_in_m_per_s) = 11 :=
by {
  sorry
}

end NUMINAMATH_GPT_time_to_pass_tree_l1886_188656


namespace NUMINAMATH_GPT_ratio_x_y_l1886_188614

theorem ratio_x_y (x y : ℤ) (h : (8 * x - 5 * y) * 3 = (11 * x - 3 * y) * 2) :
  x / y = 9 / 2 := by
  sorry

end NUMINAMATH_GPT_ratio_x_y_l1886_188614


namespace NUMINAMATH_GPT_perpendicular_OP_CD_l1886_188602

variables {Point : Type}

-- Definitions of all the points involved
variables (A B C D P O : Point)
-- Definitions for distances / lengths
variables (dist : Point → Point → ℝ)
-- Definitions for relationships
variables (circumcenter : Point → Point → Point → Point)
variables (perpendicular : Point → Point → Point → Point → Prop)

-- Segment meet condition
variables (meet_at : Point → Point → Point → Prop)

-- Assuming the given conditions
theorem perpendicular_OP_CD 
  (meet : meet_at A C P)
  (meet' : meet_at B D P)
  (h1 : dist P A = dist P D)
  (h2 : dist P B = dist P C)
  (hO : circumcenter P A B = O) :
  perpendicular O P C D :=
sorry

end NUMINAMATH_GPT_perpendicular_OP_CD_l1886_188602


namespace NUMINAMATH_GPT_solve_equations_l1886_188638

theorem solve_equations (x : ℝ) (h1 : x^2 - 9 = 0) (h2 : (-x)^3 = (-8)^2) : x = 3 ∨ x = -3 ∨ x = -4 :=
by 
  sorry

end NUMINAMATH_GPT_solve_equations_l1886_188638


namespace NUMINAMATH_GPT_strawberries_jam_profit_l1886_188613

noncomputable def betty_strawberries : ℕ := 25
noncomputable def matthew_strawberries : ℕ := betty_strawberries + 30
noncomputable def natalie_strawberries : ℕ := matthew_strawberries / 3  -- Integer division rounds down
noncomputable def total_strawberries : ℕ := betty_strawberries + matthew_strawberries + natalie_strawberries
noncomputable def strawberries_per_jar : ℕ := 12
noncomputable def jars_of_jam : ℕ := total_strawberries / strawberries_per_jar  -- Integer division rounds down
noncomputable def money_per_jar : ℕ := 6
noncomputable def total_money_made : ℕ := jars_of_jam * money_per_jar

theorem strawberries_jam_profit :
  total_money_made = 48 := by
  sorry

end NUMINAMATH_GPT_strawberries_jam_profit_l1886_188613


namespace NUMINAMATH_GPT_problem_statement_l1886_188688

noncomputable def even_increasing (f : ℝ → ℝ) :=
  ∀ x, f x = f (-x) ∧ ∀ x y, x < y → f x < f y

theorem problem_statement {f : ℝ → ℝ} (hf_even_incr : even_increasing f)
  (x1 x2 : ℝ) (hx1_gt_0 : x1 > 0) (hx2_lt_0 : x2 < 0) (hf_lt : f x1 < f x2) : x1 + x2 > 0 :=
sorry

end NUMINAMATH_GPT_problem_statement_l1886_188688


namespace NUMINAMATH_GPT_ending_number_divisible_by_3_l1886_188645

theorem ending_number_divisible_by_3 : 
∃ n : ℕ, (∀ k : ℕ, (10 + k * 3) ≤ n → (10 + k * 3) % 3 = 0) ∧ 
       (∃ c : ℕ, c = 12 ∧ (n - 10) / 3 + 1 = c) ∧ 
       n = 45 := 
sorry

end NUMINAMATH_GPT_ending_number_divisible_by_3_l1886_188645


namespace NUMINAMATH_GPT_roots_of_polynomial_l1886_188604

theorem roots_of_polynomial :
  {x : ℝ | x^10 - 5*x^8 + 4*x^6 - 64*x^4 + 320*x^2 - 256 = 0} = {x | x = 1 ∨ x = -1 ∨ x = 2 ∨ x = -2} :=
by
  sorry

end NUMINAMATH_GPT_roots_of_polynomial_l1886_188604


namespace NUMINAMATH_GPT_sum_alternating_series_l1886_188655

theorem sum_alternating_series :
  (Finset.sum (Finset.range 2023) (λ k => (-1)^(k + 1))) = -1 := 
by
  sorry

end NUMINAMATH_GPT_sum_alternating_series_l1886_188655


namespace NUMINAMATH_GPT_rectangular_prism_volume_l1886_188669

theorem rectangular_prism_volume
  (l w h : ℝ)
  (h1 : l * w = 15)
  (h2 : w * h = 10)
  (h3 : l * h = 6) :
  l * w * h = 30 := by
  sorry

end NUMINAMATH_GPT_rectangular_prism_volume_l1886_188669


namespace NUMINAMATH_GPT_find_children_and_coins_l1886_188624

def condition_for_child (k m remaining_coins : ℕ) : Prop :=
  ∃ (received_coins : ℕ), (received_coins = k + remaining_coins / 7 ∧ received_coins * 7 = 7 * k + remaining_coins)

def valid_distribution (n m : ℕ) : Prop :=
  ∀ k (hk : 1 ≤ k ∧ k ≤ n),
  ∃ remaining_coins,
    condition_for_child k m remaining_coins

theorem find_children_and_coins :
  ∃ n m, valid_distribution n m ∧ n = 6 ∧ m = 36 :=
sorry

end NUMINAMATH_GPT_find_children_and_coins_l1886_188624


namespace NUMINAMATH_GPT_area_of_enclosed_shape_l1886_188622

noncomputable def enclosed_area : ℝ :=
  ∫ x in (0 : ℝ)..(2 : ℝ), (4 * x - x^3)

theorem area_of_enclosed_shape : enclosed_area = 4 := by
  sorry

end NUMINAMATH_GPT_area_of_enclosed_shape_l1886_188622


namespace NUMINAMATH_GPT_train_speed_correct_l1886_188649

noncomputable def train_speed (train_length : ℝ) (bridge_length : ℝ) (time_seconds : ℝ) : ℝ :=
  (train_length + bridge_length) / time_seconds

theorem train_speed_correct :
  train_speed (400 : ℝ) (300 : ℝ) (45 : ℝ) = 700 / 45 :=
by
  sorry

end NUMINAMATH_GPT_train_speed_correct_l1886_188649


namespace NUMINAMATH_GPT_johns_sixth_quiz_score_l1886_188631

theorem johns_sixth_quiz_score
  (score1 score2 score3 score4 score5 : ℕ)
  (h1 : score1 = 85)
  (h2 : score2 = 90)
  (h3 : score3 = 88)
  (h4 : score4 = 92)
  (h5 : score5 = 95)
  : (∃ score6 : ℕ, (score1 + score2 + score3 + score4 + score5 + score6) / 6 = 90) :=
by
  use 90
  sorry

end NUMINAMATH_GPT_johns_sixth_quiz_score_l1886_188631


namespace NUMINAMATH_GPT_fraction_of_water_in_mixture_l1886_188696

theorem fraction_of_water_in_mixture (r : ℚ) (h : r = 2 / 3) : (3 / (2 + 3) : ℚ) = 3 / 5 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_water_in_mixture_l1886_188696


namespace NUMINAMATH_GPT_initial_tomato_count_l1886_188698

variable (T : ℝ)
variable (H1 : T - (1 / 4 * T + 20 + 40) = 15)

theorem initial_tomato_count : T = 100 :=
by
  sorry

end NUMINAMATH_GPT_initial_tomato_count_l1886_188698


namespace NUMINAMATH_GPT_club_planning_committee_l1886_188647

theorem club_planning_committee : Nat.choose 20 3 = 1140 := 
by sorry

end NUMINAMATH_GPT_club_planning_committee_l1886_188647


namespace NUMINAMATH_GPT_fractions_equivalence_l1886_188677

theorem fractions_equivalence (k : ℝ) (h : k ≠ -5) : (k + 3) / (k + 5) = 3 / 5 ↔ k = 0 := 
by 
  sorry

end NUMINAMATH_GPT_fractions_equivalence_l1886_188677


namespace NUMINAMATH_GPT_B_joined_with_54000_l1886_188683

theorem B_joined_with_54000 :
  ∀ (x : ℕ),
    (36000 * 12) / (x * 4) = 2 → x = 54000 :=
by 
  intro x h
  sorry

end NUMINAMATH_GPT_B_joined_with_54000_l1886_188683


namespace NUMINAMATH_GPT_middle_segment_proportion_l1886_188651

theorem middle_segment_proportion (a b c : ℝ) (h_a : a = 1) (h_b : b = 3) :
  (a / c = c / b) → c = Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_middle_segment_proportion_l1886_188651


namespace NUMINAMATH_GPT_car_speed_second_hour_l1886_188646

theorem car_speed_second_hour
  (v1 : ℕ) (avg_speed : ℕ) (time : ℕ) (v2 : ℕ)
  (h1 : v1 = 90)
  (h2 : avg_speed = 70)
  (h3 : time = 2) :
  v2 = 50 :=
by
  sorry

end NUMINAMATH_GPT_car_speed_second_hour_l1886_188646


namespace NUMINAMATH_GPT_theo_cookies_l1886_188610

theorem theo_cookies (cookies_per_time times_per_day total_cookies total_months : ℕ) (h1 : cookies_per_time = 13) (h2 : times_per_day = 3) (h3 : total_cookies = 2340) (h4 : total_months = 3) : (total_cookies / total_months) / (cookies_per_time * times_per_day) = 20 := 
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_theo_cookies_l1886_188610


namespace NUMINAMATH_GPT_interestDifference_l1886_188648

noncomputable def simpleInterest (P R T : ℝ) : ℝ :=
  (P * R * T) / 100

noncomputable def compoundInterest (P R T : ℝ) : ℝ :=
  P * (1 + R / 100)^T - P

theorem interestDifference (P R T : ℝ) (hP : P = 500) (hR : R = 20) (hT : T = 2) :
  compoundInterest P R T - simpleInterest P R T = 120 := by
  sorry

end NUMINAMATH_GPT_interestDifference_l1886_188648


namespace NUMINAMATH_GPT_red_tulips_for_smile_l1886_188693

/-
Problem Statement:
Anna wants to plant red and yellow tulips in the shape of a smiley face. Given the following conditions:
1. Anna needs 8 red tulips for each eye.
2. She needs 9 times the number of red tulips in the smile to make the yellow background of the face.
3. The total number of tulips needed is 196.

Prove:
The number of red tulips needed for the smile is 18.
-/

-- Defining the conditions
def red_tulips_per_eye : Nat := 8
def total_tulips : Nat := 196
def yellow_multiplier : Nat := 9

-- Proving the number of red tulips for the smile
theorem red_tulips_for_smile (R : Nat) :
  2 * red_tulips_per_eye + R + yellow_multiplier * R = total_tulips → R = 18 :=
by
  sorry

end NUMINAMATH_GPT_red_tulips_for_smile_l1886_188693


namespace NUMINAMATH_GPT_total_pens_left_l1886_188686

def initial_blue_pens := 9
def removed_blue_pens := 4
def initial_black_pens := 21
def removed_black_pens := 7
def initial_red_pens := 6

def remaining_blue_pens := initial_blue_pens - removed_blue_pens
def remaining_black_pens := initial_black_pens - removed_black_pens
def remaining_red_pens := initial_red_pens

def total_remaining_pens := remaining_blue_pens + remaining_black_pens + remaining_red_pens

theorem total_pens_left : total_remaining_pens = 25 :=
by
  -- Proof will be provided here
  sorry

end NUMINAMATH_GPT_total_pens_left_l1886_188686


namespace NUMINAMATH_GPT_perpendicular_lines_condition_l1886_188641

variable {A1 B1 C1 A2 B2 C2 : ℝ}

theorem perpendicular_lines_condition :
  (∀ x y : ℝ, A1 * x + B1 * y + C1 = 0) ∧ (∀ x y : ℝ, A2 * x + B2 * y + C2 = 0) → 
  (A1 * A2) / (B1 * B2) = -1 := 
sorry

end NUMINAMATH_GPT_perpendicular_lines_condition_l1886_188641

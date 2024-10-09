import Mathlib

namespace total_pupils_in_school_l784_78441

theorem total_pupils_in_school (girls boys : ℕ) (h_girls : girls = 542) (h_boys : boys = 387) : girls + boys = 929 := by
  sorry

end total_pupils_in_school_l784_78441


namespace flea_can_visit_all_points_l784_78433

def flea_maximum_jump (max_point : ℕ) : ℕ :=
  1006

theorem flea_can_visit_all_points (n : ℕ) (max_point : ℕ) (h_nonneg_max_point : 0 ≤ max_point) (h_segment : max_point = 2013) :
  n ≤ flea_maximum_jump max_point :=
by
  sorry

end flea_can_visit_all_points_l784_78433


namespace measure_8_liters_possible_l784_78488

-- Define the types for buckets
structure Bucket :=
  (capacity : ℕ)
  (water : ℕ := 0)

-- Initial state with a 10-liter bucket and a 6-liter bucket, both empty
def B10_init := Bucket.mk 10 0
def B6_init := Bucket.mk 6 0

-- Define a function to check if we can measure 8 liters in B10
def can_measure_8_liters (B10 B6 : Bucket) : Prop :=
  (B10.water = 8 ∧ B10.capacity = 10 ∧ B6.capacity = 6)

-- The statement to prove there exists a sequence of operations to measure 8 liters in B10
theorem measure_8_liters_possible : ∃ (B10 B6 : Bucket), can_measure_8_liters B10 B6 :=
by
  -- Proof omitted
  sorry

end measure_8_liters_possible_l784_78488


namespace average_last_three_l784_78470

/-- The average of the last three numbers is 65, given that the average of six numbers is 60
  and the average of the first three numbers is 55. -/
theorem average_last_three (a b c d e f : ℝ) (h1 : (a + b + c + d + e + f) / 6 = 60) (h2 : (a + b + c) / 3 = 55) :
  (d + e + f) / 3 = 65 :=
by
  sorry

end average_last_three_l784_78470


namespace textbook_profit_l784_78418

theorem textbook_profit (cost_price selling_price : ℕ) (h1 : cost_price = 44) (h2 : selling_price = 55) :
  (selling_price - cost_price) = 11 := by
  sorry

end textbook_profit_l784_78418


namespace introduce_people_no_three_same_acquaintances_l784_78432

theorem introduce_people_no_three_same_acquaintances (n : ℕ) :
  ∃ f : ℕ → ℕ, (∀ i, i < n → f i ≤ n - 1) ∧ (∀ i j k, i < n → j < n → k < n → i ≠ j → j ≠ k → i ≠ k → ¬(f i = f j ∧ f j = f k)) := 
sorry

end introduce_people_no_three_same_acquaintances_l784_78432


namespace find_r_l784_78420

theorem find_r (a b m p r : ℝ) (h1 : a * b = 4)
  (h2 : ∃ (q w : ℝ), (a + 2 / b = q ∧ b + 2 / a = w) ∧ q * w = r) :
  r = 9 :=
sorry

end find_r_l784_78420


namespace t_n_minus_n_even_l784_78492

noncomputable def number_of_nonempty_subsets_with_integer_average (n : ℕ) : ℕ := 
  sorry

theorem t_n_minus_n_even (N : ℕ) (hN : N > 1) :
  ∃ T_n, T_n = number_of_nonempty_subsets_with_integer_average N ∧ (T_n - N) % 2 = 0 :=
by
  sorry

end t_n_minus_n_even_l784_78492


namespace apricot_trees_count_l784_78404

theorem apricot_trees_count (peach_trees apricot_trees : ℕ) 
  (h1 : peach_trees = 300) 
  (h2 : peach_trees = 2 * apricot_trees + 30) : 
  apricot_trees = 135 := 
by 
  sorry

end apricot_trees_count_l784_78404


namespace book_pages_l784_78417

theorem book_pages (P D : ℕ) 
  (h1 : P = 23 * D + 9) 
  (h2 : ∃ D, P = 23 * (D + 1) - 14) : 
  P = 32 :=
by sorry

end book_pages_l784_78417


namespace find_g3_l784_78479

theorem find_g3 (g : ℝ → ℝ) (h : ∀ x : ℝ, g (3 ^ x) + 2 * x * g (3 ^ (-x)) = 1) : 
  g 3 = 1 / 5 := 
sorry

end find_g3_l784_78479


namespace min_focal_length_hyperbola_l784_78455

theorem min_focal_length_hyperbola (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b - c = 2) : 
  2*c ≥ 4 + 4 * Real.sqrt 2 := 
sorry

end min_focal_length_hyperbola_l784_78455


namespace find_f_42_div_17_l784_78419

def f : ℚ → ℤ := sorry

theorem find_f_42_div_17 : 
  (∀ x y : ℚ, x ≠ y → (x * y = 1 ∨ x + y = 1) → f x * f y = -1) → 
  f 0 = 1 →
  f (42 / 17) = -1 :=
sorry

end find_f_42_div_17_l784_78419


namespace x_lt_y_l784_78413

theorem x_lt_y (n : ℕ) (h_n : n > 2) (x y : ℝ) (h_x_pos : 0 < x) (h_y_pos : 0 < y) (h_x : x ^ n = x + 1) (h_y : y ^ (n + 1) = y ^ 3 + 1) : x < y :=
sorry

end x_lt_y_l784_78413


namespace abs_diff_of_prod_and_sum_l784_78480

theorem abs_diff_of_prod_and_sum (m n : ℝ) (h1 : m * n = 8) (h2 : m + n = 6) : |m - n| = 2 :=
by
  -- The proof is not required as per the instructions.
  sorry

end abs_diff_of_prod_and_sum_l784_78480


namespace find_annual_pension_l784_78490

variable (P k x a b p q : ℝ) (h1 : k * Real.sqrt (x + a) = k * Real.sqrt x + p)
                                   (h2 : k * Real.sqrt (x + b) = k * Real.sqrt x + q)

theorem find_annual_pension (h_nonzero_proportionality_constant : k ≠ 0) 
(h_year_difference : a ≠ b) : 
P = (a * q ^ 2 - b * p ^ 2) / (2 * (b * p - a * q)) := 
by
  sorry

end find_annual_pension_l784_78490


namespace range_of_x_l784_78416

theorem range_of_x (x : ℝ) : -2 * x + 3 ≤ 6 → x ≥ -3 / 2 :=
sorry

end range_of_x_l784_78416


namespace missing_digit_divisible_by_11_l784_78457

theorem missing_digit_divisible_by_11 (A : ℕ) (h : 1 ≤ A ∧ A ≤ 9) (div_11 : (100 + 10 * A + 2) % 11 = 0) : A = 3 :=
sorry

end missing_digit_divisible_by_11_l784_78457


namespace monthly_income_l784_78464

def average_expenditure_6_months (expenditure_6_months : ℕ) (average : ℕ) : Prop :=
  average = expenditure_6_months / 6

def expenditure_next_4_months (expenditure_4_months : ℕ) (monthly_expense : ℕ) : Prop :=
  expenditure_4_months = 4 * monthly_expense

def cleared_debt_and_saved (income_4_months : ℕ) (debt : ℕ) (savings : ℕ)  (condition : ℕ) : Prop :=
  income_4_months = debt + savings + condition

theorem monthly_income 
(income : ℕ) 
(avg_6m_exp : ℕ) 
(exp_4m : ℕ) 
(debt: ℕ) 
(savings: ℕ )
(condition: ℕ) 
    (h1 : average_expenditure_6_months avg_6m_exp 85) 
    (h2 : expenditure_next_4_months exp_4m 60) 
    (h3 : cleared_debt_and_saved (income * 4) debt savings 30) 
    (h4 : income * 6 < 6 * avg_6m_exp) 
    : income = 78 :=
sorry

end monthly_income_l784_78464


namespace prove_expression_value_l784_78451

theorem prove_expression_value (a b c d : ℝ) (h1 : a + b = 0) (h2 : c = -1) (h3 : d = 1 ∨ d = -1) :
  2 * a + 2 * b - c * d = 1 ∨ 2 * a + 2 * b - c * d = -1 := 
by sorry

end prove_expression_value_l784_78451


namespace quadratic_polynomial_discriminant_l784_78412

def P (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem quadratic_polynomial_discriminant (a b c : ℝ) (h₀ : a ≠ 0)
  (h₁ : ∃ x : ℝ, P a b c x = x - 2 ∧ (discriminant a (b - 1) (c + 2) = 0))
  (h₂ : ∃ x : ℝ, P a b c x = 1 - x / 2 ∧ (discriminant a (b + 1 / 2) (c - 1) = 0)) :
  discriminant a b c = -1 / 2 :=
sorry

end quadratic_polynomial_discriminant_l784_78412


namespace domain_of_sqrt_cos_minus_half_correct_l784_78442

noncomputable def domain_of_sqrt_cos_minus_half (x : ℝ) : Prop :=
  ∃ (k : ℤ), 2 * (k : ℝ) * Real.pi - Real.pi / 3 ≤ x ∧ x ≤ 2 * (k : ℝ) * Real.pi + Real.pi / 3

theorem domain_of_sqrt_cos_minus_half_correct :
  ∀ x, (∃ (k : ℤ), 2 * (k : ℝ) * Real.pi - Real.pi / 3 ≤ x ∧ x ≤ 2 * (k : ℝ) * Real.pi + Real.pi / 3) ↔
    ∃ (k : ℤ), 2 * (k : ℝ) * Real.pi - Real.pi / 3 ≤ x ∧ x ≤ 2 * (k : ℝ) * Real.pi + Real.pi / 3 :=
by sorry

end domain_of_sqrt_cos_minus_half_correct_l784_78442


namespace face_sum_l784_78473

theorem face_sum (a b c d e f : ℕ) (h : (a + d) * (b + e) * (c + f) = 1008) : 
  a + b + c + d + e + f = 173 :=
by
  sorry

end face_sum_l784_78473


namespace sin_510_eq_1_div_2_l784_78423

theorem sin_510_eq_1_div_2 : Real.sin (510 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end sin_510_eq_1_div_2_l784_78423


namespace four_digit_number_sum_of_digits_2023_l784_78476

theorem four_digit_number_sum_of_digits_2023 (a b c d : ℕ) (ha : a < 10) (hb : b < 10) (hc : c < 10) (hd : d < 10) :
  1000 * a + 100 * b + 10 * c + d = a + b + c + d + 2023 → 
  (1000 * a + 100 * b + 10 * c + d = 1997 ∨ 1000 * a + 100 * b + 10 * c + d = 2015) :=
by
  sorry

end four_digit_number_sum_of_digits_2023_l784_78476


namespace ellen_golf_cart_trips_l784_78408

def patrons_from_cars : ℕ := 12
def patrons_from_bus : ℕ := 27
def patrons_per_cart : ℕ := 3

theorem ellen_golf_cart_trips : (patrons_from_cars + patrons_from_bus) / patrons_per_cart = 13 := by
  sorry

end ellen_golf_cart_trips_l784_78408


namespace sum_of_three_equal_expressions_l784_78467

-- Definitions of variables and conditions
variables (a b c d e f g h i S : ℤ)
variable (ha : a = 4)
variable (hg : g = 13)
variable (hh : h = 6)
variable (heq1 : a + b + c + d = S)
variable (heq2 : d + e + f + g = S)
variable (heq3 : g + h + i = S)

-- Main statement we want to prove
theorem sum_of_three_equal_expressions : S = 19 + i :=
by
  -- substitution steps and equality reasoning would be carried out here
  sorry

end sum_of_three_equal_expressions_l784_78467


namespace student_failed_by_l784_78461

-- Conditions
def total_marks : ℕ := 440
def passing_percentage : ℝ := 0.50
def marks_obtained : ℕ := 200

-- Calculate passing marks
noncomputable def passing_marks : ℝ := passing_percentage * total_marks

-- Definition of the problem to be proved
theorem student_failed_by : passing_marks - marks_obtained = 20 := 
by
  sorry

end student_failed_by_l784_78461


namespace line_through_point_outside_plane_l784_78454

-- Definitions based on conditions
variable {Point Line Plane : Type}
variable (P : Point) (a : Line) (α : Plane)

-- Define the conditions
variable (passes_through : Point → Line → Prop)
variable (outside_of : Point → Plane → Prop)

-- State the theorem
theorem line_through_point_outside_plane :
  (passes_through P a) ∧ (¬ outside_of P α) :=
sorry

end line_through_point_outside_plane_l784_78454


namespace area_of_EFGH_l784_78410

-- Definitions based on given conditions
def shorter_side : ℝ := 4
def longer_side : ℝ := 8
def smaller_rectangle_area : ℝ := shorter_side * longer_side
def larger_rectangle_width : ℝ := longer_side
def larger_rectangle_height : ℝ := 2 * longer_side

-- Theorem stating the area of the larger rectangle
theorem area_of_EFGH : larger_rectangle_width * larger_rectangle_height = 128 := by
  -- Proof goes here
  sorry

end area_of_EFGH_l784_78410


namespace porter_monthly_earnings_l784_78482

-- Definitions
def regular_daily_rate : ℝ := 8
def days_per_week : ℕ := 5
def overtime_rate : ℝ := 1.5
def tax_deduction_rate : ℝ := 0.10
def insurance_deduction_rate : ℝ := 0.05
def weeks_per_month : ℕ := 4

-- Intermediate Calculations
def regular_weekly_earnings := regular_daily_rate * days_per_week
def extra_day_rate := regular_daily_rate * overtime_rate
def total_weekly_earnings := regular_weekly_earnings + extra_day_rate
def total_monthly_earnings_before_deductions := total_weekly_earnings * weeks_per_month

-- Deductions
def tax_deduction := total_monthly_earnings_before_deductions * tax_deduction_rate
def insurance_deduction := total_monthly_earnings_before_deductions * insurance_deduction_rate
def total_deductions := tax_deduction + insurance_deduction
def total_monthly_earnings_after_deductions := total_monthly_earnings_before_deductions - total_deductions

-- Theorem Statement
theorem porter_monthly_earnings : total_monthly_earnings_after_deductions = 176.80 := by
  sorry

end porter_monthly_earnings_l784_78482


namespace prod_mod_11_remainder_zero_l784_78452

theorem prod_mod_11_remainder_zero : (108 * 110) % 11 = 0 := 
by sorry

end prod_mod_11_remainder_zero_l784_78452


namespace theater_seat_count_l784_78434

theorem theater_seat_count :
  ∃ n : ℕ, n < 60 ∧ n % 9 = 5 ∧ n % 6 = 3 ∧ n = 41 :=
sorry

end theater_seat_count_l784_78434


namespace find_f_at_3_l784_78472

noncomputable def f : ℝ → ℝ := sorry

axiom functional_eq : ∀ (x : ℝ), x ≠ 2 / 3 → f x + f ((x + 2) / (2 - 3 * x)) = 2 * x

theorem find_f_at_3 : f 3 = 3 :=
by {
  sorry
}

end find_f_at_3_l784_78472


namespace friend_gain_percentage_l784_78495

noncomputable def gain_percentage (original_cost_price sold_price_friend : ℝ) : ℝ :=
  ((sold_price_friend - (original_cost_price - 0.12 * original_cost_price)) / (original_cost_price - 0.12 * original_cost_price)) * 100

theorem friend_gain_percentage (original_cost_price sold_price_friend gain_pct : ℝ) 
  (H1 : original_cost_price = 51136.36) 
  (H2 : sold_price_friend = 54000) 
  (H3 : gain_pct = 20) : 
  gain_percentage original_cost_price sold_price_friend = gain_pct := 
by
  sorry

end friend_gain_percentage_l784_78495


namespace negation_of_exists_leq_l784_78487

theorem negation_of_exists_leq (x : ℝ) : ¬ (∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) ↔ ∀ x : ℝ, x^2 + 2*x + 2 > 0 :=
by
  sorry

end negation_of_exists_leq_l784_78487


namespace price_of_first_variety_l784_78485

theorem price_of_first_variety (P : ℝ) (h1 : 1 * P + 1 * 135 + 2 * 175.5 = 153 * 4) : P = 126 :=
sorry

end price_of_first_variety_l784_78485


namespace total_supervisors_l784_78481

def buses : ℕ := 7
def supervisors_per_bus : ℕ := 3

theorem total_supervisors : buses * supervisors_per_bus = 21 := 
by
  have h : buses * supervisors_per_bus = 21 := by sorry
  exact h

end total_supervisors_l784_78481


namespace circle_general_eq_l784_78462
noncomputable def center_line (x : ℝ) := -4 * x
def tangent_line (x : ℝ) := 1 - x

def is_circle (center : ℝ × ℝ) (radius : ℝ) :=
  ∃ (x y : ℝ), (x - center.1)^2 + (y - center.2)^2 = radius^2

def is_on_line (p : ℝ × ℝ) := (p.2 = center_line p.1)

def is_tangent_at_p (center : ℝ × ℝ) (p : ℝ × ℝ) (r : ℝ) :=
  is_circle center r ∧ p.2 = tangent_line p.1 ∧ (center.1 - p.1)^2 + (center.2 - p.2)^2 = r^2

theorem circle_general_eq :
  ∀ (center : ℝ × ℝ), is_on_line center →
  ∀ r, is_tangent_at_p center (3, -2) r →
  ∀ x y : ℝ, (x - center.1)^2 + (y - center.2)^2 = r^2 →
  x^2 + y^2 - 2 * x + 8 * y + 9 = 0 := by
  sorry

end circle_general_eq_l784_78462


namespace imag_part_z_l784_78446

theorem imag_part_z {z : ℂ} (h : i * (z - 3) = -1 + 3 * i) : z.im = 1 :=
sorry

end imag_part_z_l784_78446


namespace arithmetic_seq_a7_l784_78439

theorem arithmetic_seq_a7 (a : ℕ → ℤ) (d : ℤ)
  (h1 : a 1 = 2)
  (h2 : a 3 + a 5 = 10)
  (h3 : ∀ n, a (n + 1) = a n + d) : a 7 = 8 :=
sorry

end arithmetic_seq_a7_l784_78439


namespace smallest_b_l784_78477

theorem smallest_b (a b : ℕ) (hp : a > 0) (hq : b > 0) (h1 : a - b = 8) (h2 : Nat.gcd ((a^3 + b^3) / (a + b)) (a * b) = 8) : b = 4 :=
sorry

end smallest_b_l784_78477


namespace max_min_diff_eq_four_l784_78448

noncomputable def f (a x : ℝ) : ℝ := x^3 + 3 * |x - a|

theorem max_min_diff_eq_four (a : ℝ) (h_a : a ≥ 2) : 
    let M := max (f a (-1)) (f a 1)
    let m := min (f a (-1)) (f a 1)
    M - m = 4 :=
by
  sorry

end max_min_diff_eq_four_l784_78448


namespace beaker_filling_l784_78437

theorem beaker_filling (C : ℝ) (hC : 0 < C) :
    let small_beaker_salt := (1/2) * C
    let large_beaker_capacity := 5 * C
    let large_beaker_fresh := large_beaker_capacity / 5
    let large_beaker_total_fill := large_beaker_fresh + small_beaker_salt
    (large_beaker_total_fill / large_beaker_capacity) = 3 / 10 :=
by
    let small_beaker_salt := (1/2) * C
    let large_beaker_capacity := 5 * C
    let large_beaker_fresh := large_beaker_capacity / 5
    let large_beaker_total_fill := large_beaker_fresh + small_beaker_salt
    show (large_beaker_total_fill / large_beaker_capacity) = 3 / 10
    sorry

end beaker_filling_l784_78437


namespace sixth_largest_divisor_correct_l784_78489

noncomputable def sixth_largest_divisor_of_4056600000 : ℕ :=
  50707500

theorem sixth_largest_divisor_correct : sixth_largest_divisor_of_4056600000 = 50707500 :=
sorry

end sixth_largest_divisor_correct_l784_78489


namespace jacket_cost_correct_l784_78459

-- Definitions based on given conditions
def total_cost : ℝ := 33.56
def cost_shorts : ℝ := 13.99
def cost_shirt : ℝ := 12.14
def cost_jacket : ℝ := 7.43

-- Formal statement of the proof problem in Lean 4
theorem jacket_cost_correct :
  total_cost = cost_shorts + cost_shirt + cost_jacket :=
by
  sorry

end jacket_cost_correct_l784_78459


namespace fireflies_remaining_l784_78406

theorem fireflies_remaining
  (initial_fireflies : ℕ)
  (fireflies_joined : ℕ)
  (fireflies_flew_away : ℕ)
  (h_initial : initial_fireflies = 3)
  (h_joined : fireflies_joined = 12 - 4)
  (h_flew_away : fireflies_flew_away = 2)
  : initial_fireflies + fireflies_joined - fireflies_flew_away = 9 := by
  sorry

end fireflies_remaining_l784_78406


namespace depth_of_tank_proof_l784_78440

-- Definitions based on conditions
def length_of_tank : ℝ := 25
def width_of_tank : ℝ := 12
def cost_per_sq_meter : ℝ := 0.75
def total_cost : ℝ := 558

-- The depth of the tank to be proven as 6 meters
def depth_of_tank : ℝ := 6

-- Area of the tanks for walls and bottom
def plastered_area (d : ℝ) : ℝ := 2 * (length_of_tank * d) + 2 * (width_of_tank * d) + (length_of_tank * width_of_tank)

-- Final cost calculation
def plastering_cost (d : ℝ) : ℝ := cost_per_sq_meter * (plastered_area d)

-- Statement to be proven in Lean 4
theorem depth_of_tank_proof : plastering_cost depth_of_tank = total_cost :=
by
  sorry

end depth_of_tank_proof_l784_78440


namespace decreasing_function_range_of_a_l784_78422

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3 * a - 1) * x + 4 * a else Real.log x / Real.log a

theorem decreasing_function_range_of_a :
  (∀ x y : ℝ, x < y → f a y ≤ f a x) ↔ (1/7 ≤ a ∧ a < 1/3) :=
by
  sorry

end decreasing_function_range_of_a_l784_78422


namespace percentage_to_pass_l784_78474

theorem percentage_to_pass (score shortfall max_marks : ℕ) (h_score : score = 212) (h_shortfall : shortfall = 13) (h_max_marks : max_marks = 750) :
  (score + shortfall) / max_marks * 100 = 30 :=
by
  sorry

end percentage_to_pass_l784_78474


namespace find_extrema_of_f_l784_78402

noncomputable def f (x : ℝ) : ℝ := (x^4 + x^2 + 5) / (x^2 + 1)^2

theorem find_extrema_of_f :
  (∀ x : ℝ, f x ≤ 5) ∧ (∃ x : ℝ, f x = 5) ∧ (∀ x : ℝ, f x ≥ 0.95) ∧ (∃ x : ℝ, f x = 0.95) :=
by {
  sorry
}

end find_extrema_of_f_l784_78402


namespace solve_monetary_prize_problem_l784_78475

def monetary_prize_problem : Prop :=
  ∃ (P x y : ℝ), 
    P = x + y + 30000 ∧
    x = (1/2) * P - (3/22) * (y + 30000) ∧
    y = (1/4) * P + (1/56) * x ∧
    P = 95000 ∧
    x = 40000 ∧
    y = 25000

theorem solve_monetary_prize_problem : monetary_prize_problem :=
  sorry

end solve_monetary_prize_problem_l784_78475


namespace inequality_holds_l784_78443

variable {a b c : ℝ}

theorem inequality_holds (h1 : 0 < a) (h2 : a < b) (h3 : b < 1) : (a - b) * c ^ 2 ≤ 0 :=
sorry

end inequality_holds_l784_78443


namespace find_angle_x_l784_78428

theorem find_angle_x (A B C : Type) (angle_ABC angle_CAB x : ℝ) 
  (h1 : angle_ABC = 40) 
  (h2 : angle_CAB = 120)
  (triangle_sum : x + angle_ABC + (180 - angle_CAB) = 180) : 
  x = 80 :=
by 
  -- actual proof goes here
  sorry

end find_angle_x_l784_78428


namespace initial_investment_calculation_l784_78435

theorem initial_investment_calculation
  (x : ℝ)  -- initial investment at 5% per annum
  (h₁ : x * 0.05 + 4000 * 0.08 = (x + 4000) * 0.06) :
  x = 8000 :=
by
  -- skip the proof
  sorry

end initial_investment_calculation_l784_78435


namespace bus_interval_three_buses_l784_78449

theorem bus_interval_three_buses (T : ℕ) (h : T = 21) : (T * 2) / 3 = 14 :=
by
  sorry

end bus_interval_three_buses_l784_78449


namespace greatest_radius_l784_78400

theorem greatest_radius (A : ℝ) (hA : A < 60 * Real.pi) : ∃ r : ℕ, r = 7 ∧ (r : ℝ) * (r : ℝ) < 60 :=
by
  sorry

end greatest_radius_l784_78400


namespace cuboid_surface_area_cuboid_volume_not_unique_l784_78421

theorem cuboid_surface_area
    (a b c p q : ℝ)
    (h1 : a + b + c = p)
    (h2 : a^2 + b^2 + c^2 = q^2) :
    2 * (a * b + b * c + a * c) = p^2 - q^2 :=
by
  sorry

theorem cuboid_volume_not_unique
    (a b c p q v1 v2 : ℝ)
    (h1 : a + b + c = p)
    (h2 : a^2 + b^2 + c^2 = q^2)
    : ¬ (∀ (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ), 
          a₁ + b₁ + c₁ = p ∧ a₁^2 + b₁^2 + c₁^2 = q^2 →
          a₂ + b₂ + c₂ = p ∧ a₂^2 + b₂^2 + c₂^2 = q^2 →
          (a₁ * b₁ * c₁ = a₂ * b₂ * c₂)) :=
by
  -- Provide counterexamples (4, 4, 7) and (3, 6, 6) for p = 15, q = 9
  sorry

end cuboid_surface_area_cuboid_volume_not_unique_l784_78421


namespace hcf_of_two_numbers_l784_78496
-- Importing the entire Mathlib library for mathematical functions

-- Define the two numbers and the conditions given in the problem
variables (x y : ℕ)

-- State the conditions as hypotheses
def conditions (h1 : x + y = 45) (h2 : Nat.lcm x y = 120) (h3 : (1 / (x : ℚ)) + (1 / (y : ℚ)) = 11 / 120) : Prop :=
  True

-- State the theorem we want to prove
theorem hcf_of_two_numbers (x y : ℕ)
  (h1 : x + y = 45)
  (h2 : Nat.lcm x y = 120)
  (h3 : (1 / (x : ℚ)) + (1 / (y : ℚ)) = 11 / 120) : Nat.gcd x y = 1 :=
  sorry

end hcf_of_two_numbers_l784_78496


namespace max_students_l784_78453

theorem max_students : 
  ∃ x : ℕ, x < 100 ∧ x % 9 = 4 ∧ x % 7 = 3 ∧ ∀ y : ℕ, (y < 100 ∧ y % 9 = 4 ∧ y % 7 = 3) → y ≤ x := 
by
  sorry

end max_students_l784_78453


namespace max_f_of_sin_bounded_l784_78401

theorem max_f_of_sin_bounded (x : ℝ) : (∀ y, -1 ≤ Real.sin y ∧ Real.sin y ≤ 1) → ∃ m, (∀ z, (1 + 2 * Real.sin z) ≤ m) ∧ (∀ n, (∀ z, (1 + 2 * Real.sin z) ≤ n) → m ≤ n) :=
by
  sorry

end max_f_of_sin_bounded_l784_78401


namespace tutors_meet_in_360_days_l784_78427

noncomputable def lcm_four_days : ℕ := Nat.lcm (Nat.lcm 5 6) (Nat.lcm 8 9)

theorem tutors_meet_in_360_days :
  lcm_four_days = 360 := 
by
  -- The proof steps are omitted.
  sorry

end tutors_meet_in_360_days_l784_78427


namespace geometric_sequence_proof_l784_78491

theorem geometric_sequence_proof (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) (n : ℕ) 
    (h1 : a 2 = 8) 
    (h2 : S 3 = 28) 
    (h3 : ∀ n, S n = a 1 * (1 - q^n) / (1 - q)) 
    (h4 : ∀ n, a n = a 1 * q^(n-1)) 
    (h5 : q > 1) :
    (∀ n, a n = 2^(n + 1)) ∧ (∀ n, (a n)^2 > S n + 7) := sorry

end geometric_sequence_proof_l784_78491


namespace log15_12_eq_l784_78444

-- Goal: Define the constants and statement per the identified conditions and goal
variable (a b : ℝ)
#check Real.log
#check Real.logb

-- Math conditions
def lg2_eq_a := Real.log 2 = a
def lg3_eq_b := Real.log 3 = b

-- Math proof problem statement
theorem log15_12_eq : lg2_eq_a a → lg3_eq_b b → Real.logb 15 12 = (2 * a + b) / (1 - a + b) :=
by intros h1 h2; sorry

end log15_12_eq_l784_78444


namespace bucket_problem_l784_78411

theorem bucket_problem 
  (C : ℝ) -- original capacity of the bucket
  (N : ℕ) -- number of buckets required to fill the tank with the original bucket size
  (h : N * C = 25 * (2/5) * C) : 
  N = 10 :=
by
  sorry

end bucket_problem_l784_78411


namespace side_face_area_l784_78425

noncomputable def box_lengths (l w h : ℕ) : Prop :=
  (w * h = (1 / 2) * l * w ∧
   l * w = (3 / 2) * l * h ∧
   l * w * h = 5184 ∧
   2 * (l + h) = (6 / 5) * 2 * (l + w))

theorem side_face_area :
  ∃ (l w h : ℕ), box_lengths l w h ∧ l * h = 384 := by
  sorry

end side_face_area_l784_78425


namespace Y_4_3_l784_78431

def Y (x y : ℝ) : ℝ := x^2 - 3 * x * y + y^2

theorem Y_4_3 : Y 4 3 = -11 :=
by
  -- This line is added to skip the proof and focus on the statement.
  sorry

end Y_4_3_l784_78431


namespace sweets_remaining_l784_78494

def num_cherry := 30
def num_strawberry := 40
def num_pineapple := 50

def half (n : Nat) := n / 2

def num_eaten_cherry := half num_cherry
def num_eaten_strawberry := half num_strawberry
def num_eaten_pineapple := half num_pineapple

def num_given_away := 5

def total_initial := num_cherry + num_strawberry + num_pineapple

def total_eaten := num_eaten_cherry + num_eaten_strawberry + num_eaten_pineapple

def total_remaining_after_eating := total_initial - total_eaten
def total_remaining := total_remaining_after_eating - num_given_away

theorem sweets_remaining : total_remaining = 55 := by
  sorry

end sweets_remaining_l784_78494


namespace find_larger_number_l784_78409

theorem find_larger_number (x y : ℤ) (h1 : x - y = 7) (h2 : x + y = 41) : x = 24 :=
by sorry

end find_larger_number_l784_78409


namespace sequence_a8_equals_neg2_l784_78445

theorem sequence_a8_equals_neg2 (a : ℕ → ℝ) (h1 : a 1 = 1) 
  (h2 : ∀ n : ℕ, a n * a (n + 1) = -2) 
  : a 8 = -2 :=
sorry

end sequence_a8_equals_neg2_l784_78445


namespace solution_set_of_quadratic_inequality_l784_78465

theorem solution_set_of_quadratic_inequality :
  {x : ℝ | x^2 + 2 * x - 3 < 0} = {x : ℝ | -3 < x ∧ x < 1} :=
sorry

end solution_set_of_quadratic_inequality_l784_78465


namespace gcd_8251_6105_l784_78486

theorem gcd_8251_6105 :
  Nat.gcd 8251 6105 = 37 :=
by
  sorry

end gcd_8251_6105_l784_78486


namespace crates_second_trip_l784_78471

theorem crates_second_trip
  (x y : Nat) 
  (h1 : x + y = 12)
  (h2 : x = 5) :
  y = 7 :=
by
  sorry

end crates_second_trip_l784_78471


namespace smallest_bdf_value_l784_78456

open Nat

theorem smallest_bdf_value
  (a b c d e f : ℕ)
  (h1 : (a + 1) * c * e = a * c * e + 3 * b * d * f)
  (h2 : a * (c + 1) * e = a * c * e + 4 * b * d * f)
  (h3 : a * c * (e + 1) = a * c * e + 5 * b * d * f) :
  (b * d * f = 60) :=
sorry

end smallest_bdf_value_l784_78456


namespace decoration_sets_count_l784_78405

/-- 
Prove the number of different decoration sets that can be purchased for $120 dollars,
where each balloon costs $4, each ribbon costs $6, and the number of balloons must be even,
is exactly 2.
-/
theorem decoration_sets_count : 
  ∃ n : ℕ, n = 2 ∧ 
  (∃ (b r : ℕ), 
    4 * b + 6 * r = 120 ∧ 
    b % 2 = 0 ∧ 
    ∃ (i j : ℕ), 
      i ≠ j ∧ 
      (4 * i + 6 * (120 - 4 * i) / 6 = 120) ∧ 
      (4 * j + 6 * (120 - 4 * j) / 6 = 120) 
  )
:= sorry

end decoration_sets_count_l784_78405


namespace circle_range_of_t_max_radius_t_value_l784_78429

open Real

theorem circle_range_of_t {t x y : ℝ} :
  (x^2 + y^2 - 2 * (t + 3) * x + 2 * (1 - 4 * t^2) * y + 16*t^4 + 9 = 0) →
  (- (1:ℝ)/7 < t ∧ t < 1) :=
by
  sorry

theorem max_radius_t_value {t x y : ℝ} :
  (x^2 + y^2 - 2 * (t + 3) * x + 2 * (1 - 4 * t^2) * y + 16*t^4 + 9 = 0) →
  (- (1:ℝ)/7 < t ∧ t < 1) →
  (∃ r, r^2 = -7*t^2 + 6*t + 1) →
  t = 3 / 7 :=
by
  sorry

end circle_range_of_t_max_radius_t_value_l784_78429


namespace solution_inequality_l784_78468

noncomputable def solution_set (a : ℝ) (x : ℝ) := x < (1 - a) / (1 + a)

theorem solution_inequality 
  (a : ℝ) 
  (h1 : a^3 < a) 
  (h2 : a < a^2) :
  ∀ (x : ℝ), x + a > 1 - a * x ↔ solution_set a x :=
sorry

end solution_inequality_l784_78468


namespace simplify_sqrt_eight_l784_78483

theorem simplify_sqrt_eight : Real.sqrt 8 = 2 * Real.sqrt 2 :=
by
  -- Given that 8 can be factored into 4 * 2 and the property sqrt(a * b) = sqrt(a) * sqrt(b)
  sorry

end simplify_sqrt_eight_l784_78483


namespace total_tomato_seeds_l784_78403

theorem total_tomato_seeds (mike_morning mike_afternoon : ℕ) 
  (ted_morning : mike_morning = 50) 
  (ted_afternoon : mike_afternoon = 60) 
  (ted_morning_eq : 2 * mike_morning = 100) 
  (ted_afternoon_eq : mike_afternoon - 20 = 40)
  (total_seeds : mike_morning + mike_afternoon + (2 * mike_morning) + (mike_afternoon - 20) = 250) : 
  (50 + 60 + 100 + 40 = 250) :=
sorry

end total_tomato_seeds_l784_78403


namespace expression_divisible_by_24_l784_78498

theorem expression_divisible_by_24 (n : ℕ) (hn : 0 < n) : ∃ k : ℕ, (n + 7)^2 - (n - 5)^2 = 24 * k := by
  sorry

end expression_divisible_by_24_l784_78498


namespace compare_neg_fractions_l784_78493

theorem compare_neg_fractions : (- (3 : ℚ) / 5) > (- (3 : ℚ) / 4) := sorry

end compare_neg_fractions_l784_78493


namespace problem_log_inequality_l784_78407

noncomputable def f (x m : ℝ) := x - |x + 2| - |x - 3| - m

theorem problem (m : ℝ) (h1 : ∀ x : ℝ, (1 / m) - 4 ≥ f x m) :
  m > 0 :=
sorry

noncomputable def log_base (b x : ℝ) := Real.log x / Real.log b

theorem log_inequality (m : ℝ) (h2 : m > 0) :
  log_base (m + 1) (m + 2) > log_base (m + 2) (m + 3) :=
sorry

end problem_log_inequality_l784_78407


namespace ball_falls_in_middle_pocket_l784_78497

theorem ball_falls_in_middle_pocket (p q : ℕ) (hp : p % 2 = 1) (hq : q % 2 = 1) : 
  ∃ k : ℕ, (k * p) % (2 * q) = 0 :=
by
  sorry

end ball_falls_in_middle_pocket_l784_78497


namespace find_k_l784_78426

theorem find_k (k : ℝ) : (∃ x y : ℝ, y = -2 * x + 4 ∧ y = k * x ∧ y = x + 2) → k = 4 :=
by
  sorry

end find_k_l784_78426


namespace cloth_cost_price_per_metre_l784_78478

theorem cloth_cost_price_per_metre (total_metres : ℕ) (total_price : ℕ) (loss_per_metre : ℕ) :
  total_metres = 300 → total_price = 18000 → loss_per_metre = 5 → (total_price / total_metres + loss_per_metre) = 65 :=
by
  intros
  sorry

end cloth_cost_price_per_metre_l784_78478


namespace find_x_l784_78430

def x_condition (x : ℤ) : Prop :=
  (120 ≤ x ∧ x ≤ 150) ∧ (x % 5 = 2) ∧ (x % 6 = 5)

theorem find_x :
  ∃ x : ℤ, x_condition x ∧ x = 137 :=
by
  sorry

end find_x_l784_78430


namespace sum_of_squares_l784_78469

def positive_integers (x y z : ℕ) : Prop :=
  x > 0 ∧ y > 0 ∧ z > 0

def sum_of_values (x y z : ℕ) : Prop :=
  x + y + z = 24 ∧ Int.gcd x y + Int.gcd y z + Int.gcd z x = 10

theorem sum_of_squares (x y z : ℕ) (h1 : positive_integers x y z) (h2 : sum_of_values x y z) :
  x^2 + y^2 + z^2 = 296 :=
by sorry

end sum_of_squares_l784_78469


namespace total_rainfall_l784_78436

theorem total_rainfall
  (monday : ℝ)
  (tuesday : ℝ)
  (wednesday : ℝ)
  (h_monday : monday = 0.17)
  (h_tuesday : tuesday = 0.42)
  (h_wednesday : wednesday = 0.08) :
  monday + tuesday + wednesday = 0.67 :=
by
  sorry

end total_rainfall_l784_78436


namespace find_value_sum_l784_78447

noncomputable def f : ℝ → ℝ
  := sorry

axiom odd_function (x : ℝ) : f (-x) = -f x
axiom periodic_function (x : ℝ) : f (x + 3) = f x
axiom value_at_minus_one : f (-1) = 1

theorem find_value_sum :
  f 1 + f 2 + f 3 + f 4 + f 5 + f 6 + f 7 + f 8 + f 9 + f 10 = -1 :=
by
  sorry

end find_value_sum_l784_78447


namespace cos_pi_div_4_minus_alpha_l784_78499

theorem cos_pi_div_4_minus_alpha (α : ℝ) (h : Real.sin (α + π/4) = 5/13) : 
  Real.cos (π/4 - α) = 5/13 :=
by
  sorry

end cos_pi_div_4_minus_alpha_l784_78499


namespace find_p_if_parabola_axis_tangent_to_circle_l784_78466

theorem find_p_if_parabola_axis_tangent_to_circle :
  ∀ (p : ℝ), 0 < p →
    (∃ (C : ℝ × ℝ) (r : ℝ), 
      (C = (2, 0)) ∧ (r = 3) ∧ (dist (C.1 + p / 2, C.2) (C.1, C.2) = r) 
    ) → p = 2 :=
by
  intro p hp h
  rcases h with ⟨C, r, hC, hr, h_dist⟩ 
  have h_eq : C = (2, 0) := hC
  have hr_eq : r = 3 := hr
  rw [h_eq, hr_eq] at h_dist
  sorry

end find_p_if_parabola_axis_tangent_to_circle_l784_78466


namespace sum_of_f10_values_l784_78415

noncomputable def f : ℕ → ℝ := sorry

axiom f_cond1 : f 1 = 4

axiom f_cond2 : ∀ (m n : ℕ), m ≥ n → f (m + n) + f (m - n) = (f (2 * m) + f (2 * n)) / 2

theorem sum_of_f10_values : f 10 = 400 :=
sorry

end sum_of_f10_values_l784_78415


namespace remainder_of_6_power_700_mod_72_l784_78460

theorem remainder_of_6_power_700_mod_72 : (6^700) % 72 = 0 :=
by
  sorry

end remainder_of_6_power_700_mod_72_l784_78460


namespace trevor_pages_l784_78484

theorem trevor_pages (p1 p2 p3 : ℕ) (h1 : p1 = 72) (h2 : p2 = 72) (h3 : p3 = p1 + 4) : 
    p1 + p2 + p3 = 220 := 
by 
    sorry

end trevor_pages_l784_78484


namespace add_to_37_eq_52_l784_78414

theorem add_to_37_eq_52 (x : ℕ) (h : 37 + x = 52) : x = 15 := by
  sorry

end add_to_37_eq_52_l784_78414


namespace percent_of_rs_600_l784_78463

theorem percent_of_rs_600 : (600 * 0.25 = 150) :=
by
  sorry

end percent_of_rs_600_l784_78463


namespace minimum_value_l784_78438

noncomputable def smallest_value_expression (x y : ℝ) := x^4 + y^4 - x^2 * y - x * y^2

theorem minimum_value (x y : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : x + y ≤ 1) :
  (smallest_value_expression x y) ≥ -1 / 8 :=
sorry

end minimum_value_l784_78438


namespace quadratic_function_choice_l784_78450

-- Define what it means to be a quadratic function
def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x : ℝ, f x = a * x^2 + b * x + c

-- Define the given equations as functions
def f_A (x : ℝ) : ℝ := 3 * x
def f_B (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c
def f_C (x : ℝ) : ℝ := (x - 1)^2
def f_D (x : ℝ) : ℝ := 2

-- State the Lean theorem statement
theorem quadratic_function_choice : is_quadratic f_C := sorry

end quadratic_function_choice_l784_78450


namespace negation_of_quadratic_inequality_l784_78424

-- Definitions
def quadratic_inequality (a : ℝ) : Prop := ∃ x : ℝ, x * x + a * x + 1 < 0

-- Theorem statement
theorem negation_of_quadratic_inequality (a : ℝ) : ¬ (quadratic_inequality a) ↔ ∀ x : ℝ, x * x + a * x + 1 ≥ 0 :=
by sorry

end negation_of_quadratic_inequality_l784_78424


namespace solution_set_inequality_l784_78458

theorem solution_set_inequality (x : ℝ) : 
  (x - 3) * (x - 1) > 0 ↔ x < 1 ∨ x > 3 :=
sorry

end solution_set_inequality_l784_78458

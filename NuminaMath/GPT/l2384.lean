import Mathlib

namespace range_of_a_l2384_238432

theorem range_of_a (a : ℝ) (h : ∀ x, a ≤ x ∧ x ≤ a + 2 → |x + a| ≥ 2 * |x|) : a ≤ -3 / 2 := 
by
  sorry

end range_of_a_l2384_238432


namespace daleyza_contracted_units_l2384_238483

variable (units_building1 : ℕ)
variable (units_building2 : ℕ)
variable (units_building3 : ℕ)

def total_units (units_building1 units_building2 units_building3 : ℕ) : ℕ :=
  units_building1 + units_building2 + units_building3

theorem daleyza_contracted_units :
  units_building1 = 4000 →
  units_building2 = 2 * units_building1 / 5 →
  units_building3 = 120 * units_building2 / 100 →
  total_units units_building1 units_building2 units_building3 = 7520 :=
by
  intros h1 h2 h3
  unfold total_units
  rw [h1, h2, h3]
  sorry

end daleyza_contracted_units_l2384_238483


namespace solve_recurrence_relation_l2384_238478

noncomputable def a_n (n : ℕ) : ℝ := 2 * 4^n - 2 * n + 2
noncomputable def b_n (n : ℕ) : ℝ := 2 * 4^n + 2 * n - 2

theorem solve_recurrence_relation :
  a_n 0 = 4 ∧ b_n 0 = 0 ∧
  (∀ n : ℕ, a_n (n + 1) = 3 * a_n n + b_n n - 4) ∧
  (∀ n : ℕ, b_n (n + 1) = 2 * a_n n + 2 * b_n n + 2) :=
by
  sorry

end solve_recurrence_relation_l2384_238478


namespace lloyd_normal_hours_l2384_238496

-- Definitions based on the conditions
def regular_rate : ℝ := 3.50
def overtime_rate : ℝ := 1.5 * regular_rate
def total_hours_worked : ℝ := 10.5
def total_earnings : ℝ := 42
def normal_hours_worked (h : ℝ) : Prop := 
  h * regular_rate + (total_hours_worked - h) * overtime_rate = total_earnings

-- The theorem to prove
theorem lloyd_normal_hours : ∃ h : ℝ, normal_hours_worked h ∧ h = 7.5 := sorry

end lloyd_normal_hours_l2384_238496


namespace compute_fraction_square_l2384_238435

theorem compute_fraction_square : 6 * (3 / 7) ^ 2 = 54 / 49 :=
by 
  sorry

end compute_fraction_square_l2384_238435


namespace remi_water_intake_l2384_238493

def bottle_capacity := 20
def daily_refills := 3
def num_days := 7
def spill1 := 5
def spill2 := 8

def daily_intake := daily_refills * bottle_capacity
def total_intake_without_spill := daily_intake * num_days
def total_spill := spill1 + spill2
def total_intake_with_spill := total_intake_without_spill - total_spill

theorem remi_water_intake : total_intake_with_spill = 407 := 
by
  -- Provide proof here
  sorry

end remi_water_intake_l2384_238493


namespace exponential_simplification_l2384_238419

theorem exponential_simplification : 
  (10^0.25) * (10^0.25) * (10^0.5) * (10^0.5) * (10^0.75) * (10^0.75) = 1000 := 
by 
  sorry

end exponential_simplification_l2384_238419


namespace solve_for_x_l2384_238448

theorem solve_for_x (x : ℕ) : 8 * 4^x = 2048 → x = 4 := by
  sorry

end solve_for_x_l2384_238448


namespace find_stream_speed_l2384_238412

-- Define the problem based on the provided conditions
theorem find_stream_speed (b s : ℝ) (h1 : b + s = 250 / 7) (h2 : b - s = 150 / 21) : s = 14.28 :=
by
  sorry

end find_stream_speed_l2384_238412


namespace odd_function_value_at_2_l2384_238427

variable (f : ℝ → ℝ)
variable (h_odd : ∀ x : ℝ, f (-x) = -f x)

theorem odd_function_value_at_2 : f (-2) + f (2) = 0 :=
by
  sorry

end odd_function_value_at_2_l2384_238427


namespace factor_expression_l2384_238498

theorem factor_expression (x : ℝ) : 16 * x ^ 2 + 8 * x = 8 * x * (2 * x + 1) :=
by
  -- Problem: Completely factor the expression
  -- Given Condition
  -- Conclusion
  sorry

end factor_expression_l2384_238498


namespace average_minutes_per_day_is_correct_l2384_238489
-- Import required library for mathematics

-- Define the conditions
def sixth_grade_minutes := 10
def seventh_grade_minutes := 12
def eighth_grade_minutes := 8
def sixth_grade_ratio := 3
def eighth_grade_ratio := 1/2

-- We use noncomputable since we'll rely on some real number operations that are not trivially computable.
noncomputable def total_minutes_per_week (s : ℝ) : ℝ :=
  sixth_grade_minutes * (sixth_grade_ratio * s) * 2 + 
  seventh_grade_minutes * s * 2 + 
  eighth_grade_minutes * (eighth_grade_ratio * s) * 1

noncomputable def total_students (s : ℝ) : ℝ :=
  sixth_grade_ratio * s + s + eighth_grade_ratio * s

noncomputable def average_minutes_per_day : ℝ :=
  (total_minutes_per_week 1) / (total_students 1 / 5)

theorem average_minutes_per_day_is_correct : average_minutes_per_day = 176 / 9 :=
by
  sorry

end average_minutes_per_day_is_correct_l2384_238489


namespace line_circle_no_intersection_l2384_238428

theorem line_circle_no_intersection :
  ∀ (x y : ℝ), (5 * x + 8 * y = 10) → ¬ (x^2 + y^2 = 1) :=
by
  intro x y hline hcirc
  -- Proof omitted
  sorry

end line_circle_no_intersection_l2384_238428


namespace find_x_squared_plus_y_squared_plus_z_squared_l2384_238409

theorem find_x_squared_plus_y_squared_plus_z_squared
  (x y z : ℤ)
  (h1 : x + y + z = 3)
  (h2 : x^3 + y^3 + z^3 = 3) :
  x^2 + y^2 + z^2 = 57 :=
by
  sorry

end find_x_squared_plus_y_squared_plus_z_squared_l2384_238409


namespace factorial_division_identity_l2384_238439

theorem factorial_division_identity: (Nat.factorial 10) / ((Nat.factorial 7) * (Nat.factorial 3)) = 120 := by
  sorry

end factorial_division_identity_l2384_238439


namespace composite_expression_l2384_238471

theorem composite_expression (n : ℕ) (h : n > 1) : ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ 3^(2*n+1) - 2^(2*n+1) - 6^n = a * b :=
sorry

end composite_expression_l2384_238471


namespace trigonometric_identity_l2384_238497

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 1) : 
  1 - 2 * Real.sin α * Real.cos α - 3 * (Real.cos α)^2 = -3 / 2 :=
sorry

end trigonometric_identity_l2384_238497


namespace quadratic_inequality_solution_l2384_238457

variables {x p q : ℝ}

theorem quadratic_inequality_solution
  (h1 : ∀ x, x^2 + p * x + q < 0 ↔ -1/2 < x ∧ x < 1/3) : 
  ∀ x, q * x^2 + p * x + 1 > 0 ↔ -2 < x ∧ x < 3 :=
by sorry

end quadratic_inequality_solution_l2384_238457


namespace cos_difference_l2384_238466

theorem cos_difference (α β : ℝ) (h_α_acute : 0 < α ∧ α < π / 2)
                      (h_β_acute : 0 < β ∧ β < π / 2)
                      (h_cos_α : Real.cos α = 1 / 3)
                      (h_cos_sum : Real.cos (α + β) = -1 / 3) :
  Real.cos (α - β) = 23 / 27 := 
sorry

end cos_difference_l2384_238466


namespace three_mathematicians_same_language_l2384_238414

theorem three_mathematicians_same_language
  (M : Fin 9 → Finset string)
  (h1 : ∀ i j k : Fin 9, ∃ lang, i ≠ j → i ≠ k → j ≠ k → lang ∈ M i ∧ lang ∈ M j)
  (h2 : ∀ i : Fin 9, (M i).card ≤ 3)
  : ∃ lang ∈ ⋃ i, M i, ∃ (A B C : Fin 9), A ≠ B → A ≠ C → B ≠ C → lang ∈ M A ∧ lang ∈ M B ∧ lang ∈ M C :=
sorry

end three_mathematicians_same_language_l2384_238414


namespace pencils_bought_l2384_238454

theorem pencils_bought (payment change pencil_cost glue_cost : ℕ)
  (h_payment : payment = 1000)
  (h_change : change = 100)
  (h_pencil_cost : pencil_cost = 210)
  (h_glue_cost : glue_cost = 270) :
  (payment - change - glue_cost) / pencil_cost = 3 :=
by sorry

end pencils_bought_l2384_238454


namespace triangle_side_length_BC_49_l2384_238433

theorem triangle_side_length_BC_49
  (angle_A : ℝ)
  (AC : ℝ)
  (area_ABC : ℝ)
  (h1 : angle_A = 60)
  (h2 : AC = 16)
  (h3 : area_ABC = 220 * Real.sqrt 3) : 
  ∃ (BC : ℝ), BC = 49 :=
by
  sorry

end triangle_side_length_BC_49_l2384_238433


namespace solution_comparison_l2384_238421

theorem solution_comparison (a a' b b' k : ℝ) (h1 : a ≠ 0) (h2 : a' ≠ 0) (h3 : 0 < k) :
  (k * b * a') > (a * b') :=
sorry

end solution_comparison_l2384_238421


namespace sum_of_geometric_ratios_l2384_238467

theorem sum_of_geometric_ratios (k a2 a3 b2 b3 p r : ℝ)
  (h_seq1 : a2 = k * p)
  (h_seq2 : a3 = k * p^2)
  (h_seq3 : b2 = k * r)
  (h_seq4 : b3 = k * r^2)
  (h_diff : a3 - b3 = 3 * (a2 - b2) - k) :
  p + r = 2 :=
by
  sorry

end sum_of_geometric_ratios_l2384_238467


namespace product_mod_5_l2384_238463

theorem product_mod_5 : (3 * 13 * 23 * 33 * 43 * 53 * 63 * 73 * 83 * 93) % 5 = 4 := 
by 
  sorry

end product_mod_5_l2384_238463


namespace find_roots_of_star_eq_l2384_238416

def star (a b : ℝ) : ℝ := a^2 - b^2

theorem find_roots_of_star_eq :
  (star (star 2 3) x = 9) ↔ (x = 4 ∨ x = -4) :=
by
  sorry

end find_roots_of_star_eq_l2384_238416


namespace original_expenditure_beginning_month_l2384_238402

theorem original_expenditure_beginning_month (A E : ℝ)
  (h1 : E = 35 * A)
  (h2 : E + 84 = 42 * (A - 1))
  (h3 : E + 124 = 37 * (A + 1))
  (h4 : E + 154 = 40 * (A + 1)) :
  E = 630 := 
sorry

end original_expenditure_beginning_month_l2384_238402


namespace hyperbola_eccentricity_l2384_238464

theorem hyperbola_eccentricity (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : b = a) 
  (h₄ : ∀ c, (c = Real.sqrt (a^2 + b^2)) → (b * c / Real.sqrt (a^2 + b^2) = a)) :
  (Real.sqrt (2) = (c / a)) :=
by
  sorry

end hyperbola_eccentricity_l2384_238464


namespace part1_part2_l2384_238423

variables (a b c d m : Real) 

-- Condition: a and b are opposite numbers
def opposite_numbers (a b : Real) : Prop := a = -b

-- Condition: c and d are reciprocals
def reciprocals (c d : Real) : Prop := c = 1 / d

-- Condition: |m| = 3
def absolute_value_three (m : Real) : Prop := abs m = 3

-- Statement for part 1
theorem part1 (h1 : opposite_numbers a b) (h2 : reciprocals c d) (h3 : absolute_value_three m) :
  a + b = 0 ∧ c * d = 1 ∧ (m = 3 ∨ m = -3) :=
by
  sorry

-- Statement for part 2
theorem part2 (h1 : opposite_numbers a b) (h2 : reciprocals c d) (h3 : absolute_value_three m) (h4 : m < 0) :
  m^3 + c * d + (a + b) / m = -26 :=
by
  sorry

end part1_part2_l2384_238423


namespace trig_expression_value_l2384_238449

theorem trig_expression_value {θ : Real} (h : Real.tan θ = 2) :
  (2 * Real.sin θ - Real.cos θ) / (Real.sin θ + 2 * Real.cos θ) = 3 / 4 := 
by
  sorry

end trig_expression_value_l2384_238449


namespace child_l2384_238430

-- Definitions of the given conditions
def total_money : ℕ := 35
def adult_ticket_cost : ℕ := 8
def number_of_children : ℕ := 9

-- Statement of the math proof problem
theorem child's_ticket_cost : ∃ C : ℕ, total_money - adult_ticket_cost = C * number_of_children ∧ C = 3 :=
by
  sorry

end child_l2384_238430


namespace calculate_star_value_l2384_238438

def custom_operation (a b : ℕ) : ℕ :=
  (a + b)^3

theorem calculate_star_value : custom_operation 3 5 = 512 :=
by
  sorry

end calculate_star_value_l2384_238438


namespace two_digit_plus_one_multiple_of_3_4_5_6_7_l2384_238453

theorem two_digit_plus_one_multiple_of_3_4_5_6_7 (n : ℕ) (h1 : 10 ≤ n) (h2 : n < 100) :
  (∃ m : ℕ, (m = n - 1 ∧ m % 3 = 0 ∧ m % 4 = 0 ∧ m % 5 = 0 ∧ m % 6 = 0 ∧ m % 7 = 0)) → False :=
sorry

end two_digit_plus_one_multiple_of_3_4_5_6_7_l2384_238453


namespace mary_has_34_lambs_l2384_238474

def mary_lambs (initial_lambs : ℕ) (lambs_with_babies : ℕ) (babies_per_lamb : ℕ) (traded_lambs : ℕ) (found_lambs : ℕ): ℕ :=
  initial_lambs + (lambs_with_babies * babies_per_lamb) - traded_lambs + found_lambs

theorem mary_has_34_lambs :
  mary_lambs 12 4 3 5 15 = 34 :=
by
  -- This line is in place of the actual proof.
  sorry

end mary_has_34_lambs_l2384_238474


namespace compare_abc_case1_compare_abc_case2_compare_abc_case3_l2384_238462

variable (a : ℝ)
variable (b : ℝ := (1 / 2) * (a + 3 / a))
variable (c : ℝ := (1 / 2) * (b + 3 / b))

-- First condition: if \(a > \sqrt{3}\), then \(a > b > c\)
theorem compare_abc_case1 (h1 : a > 0) (h2 : a > Real.sqrt 3) : a > b ∧ b > c := sorry

-- Second condition: if \(a = \sqrt{3}\), then \(a = b = c\)
theorem compare_abc_case2 (h1 : a > 0) (h2 : a = Real.sqrt 3) : a = b ∧ b = c := sorry

-- Third condition: if \(0 < a < \sqrt{3}\), then \(a < c < b\)
theorem compare_abc_case3 (h1 : a > 0) (h2 : a < Real.sqrt 3) : a < c ∧ c < b := sorry

end compare_abc_case1_compare_abc_case2_compare_abc_case3_l2384_238462


namespace camping_trip_percentage_l2384_238477

theorem camping_trip_percentage (T : ℝ)
  (h1 : 16 / 100 ≤ 1)
  (h2 : T - 16 / 100 ≤ 1)
  (h3 : T = 64 / 100) :
  T = 64 / 100 := by
  sorry

end camping_trip_percentage_l2384_238477


namespace Bill_order_combinations_l2384_238481

def donut_combinations (num_donuts num_kinds : ℕ) : ℕ :=
  Nat.choose (num_donuts + num_kinds - 1) (num_kinds - 1)

theorem Bill_order_combinations : donut_combinations 10 5 = 126 :=
by
  -- This would be the place to insert the proof steps, but we're using sorry as the placeholder.
  sorry

end Bill_order_combinations_l2384_238481


namespace eccentricity_of_ellipse_l2384_238488

theorem eccentricity_of_ellipse 
  (a b c m n : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : m > 0) 
  (h4 : n > 0) 
  (ellipse_eq : ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 -> (m^2 + n^2 > x^2 + y^2))
  (hyperbola_eq : ∀ x y : ℝ, x^2 / m^2 - y^2 / n^2 = 1 -> (m^2 + n^2 > x^2 - y^2))
  (same_foci: ∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 <= 1 → x^2 / m^2 - y^2 / n^2 = 1)
  (geometric_mean : c^2 = a * m)
  (arithmetic_mean : 2 * n^2 = 2 * m^2 + c^2) : 
  (c / a = 1 / 2) :=
sorry

end eccentricity_of_ellipse_l2384_238488


namespace size_of_angle_C_max_value_of_a_add_b_l2384_238492

variable (A B C a b c : ℝ)
variable (h₀ : 0 < A ∧ A < π / 2)
variable (h₁ : 0 < B ∧ B < π / 2)
variable (h₂ : 0 < C ∧ C < π / 2)
variable (h₃ : a = 2 * c * sin A / sqrt 3)
variable (h₄ : a * a + b * b - 2 * a * b * cos (π / 3) = c * c)

theorem size_of_angle_C (h₅: a ≠ 0):
  C = π / 3 :=
by sorry

theorem max_value_of_a_add_b (h₆: c = 2):
  a + b ≤ 4 :=
by sorry

end size_of_angle_C_max_value_of_a_add_b_l2384_238492


namespace no_solution_condition_l2384_238417

theorem no_solution_condition (b : ℝ) : (∀ x : ℝ, 4 * (3 * x - b) ≠ 3 * (4 * x + 16)) ↔ b = -12 := 
by
  sorry

end no_solution_condition_l2384_238417


namespace school_cases_of_water_l2384_238401

theorem school_cases_of_water (bottles_per_case bottles_used_first_game bottles_left_after_second_game bottles_used_second_game : ℕ)
  (h1 : bottles_per_case = 20)
  (h2 : bottles_used_first_game = 70)
  (h3 : bottles_left_after_second_game = 20)
  (h4 : bottles_used_second_game = 110) :
  let total_bottles_used := bottles_used_first_game + bottles_used_second_game
  let total_bottles_initial := total_bottles_used + bottles_left_after_second_game
  let number_of_cases := total_bottles_initial / bottles_per_case
  number_of_cases = 10 :=
by
  -- The proof goes here
  sorry

end school_cases_of_water_l2384_238401


namespace sum_of_731_and_one_fifth_l2384_238408

theorem sum_of_731_and_one_fifth :
  (7.31 + (1 / 5) = 7.51) :=
sorry

end sum_of_731_and_one_fifth_l2384_238408


namespace equation_of_the_line_l2384_238469

theorem equation_of_the_line (a b : ℝ) :
    ((a - b = 5) ∧ (9 / a + 4 / b = 1)) → 
    ( (2 * 9 + 3 * 4 - 30 = 0) ∨ (2 * 9 - 3 * 4 - 6 = 0) ∨ (9 - 4 - 5 = 0)) :=
  by
    sorry

end equation_of_the_line_l2384_238469


namespace charts_per_associate_professor_l2384_238445

-- Definitions
def A : ℕ := 3
def B : ℕ := 4
def C : ℕ := 1

-- Conditions based on the given problem
axiom h1 : 2 * A + B = 10
axiom h2 : A * C + 2 * B = 11
axiom h3 : A + B = 7

-- The theorem to be proven
theorem charts_per_associate_professor : C = 1 := by
  sorry

end charts_per_associate_professor_l2384_238445


namespace kopeechka_items_l2384_238434

theorem kopeechka_items (a n : ℕ) (hn : n * (100 * a + 99) = 20083) : n = 17 ∨ n = 117 :=
sorry

end kopeechka_items_l2384_238434


namespace number_of_white_balls_l2384_238470

-- Definitions based on the problem conditions
def total_balls : Nat := 120
def red_freq : ℝ := 0.15
def black_freq : ℝ := 0.45

-- Result to prove
theorem number_of_white_balls :
  let red_balls := total_balls * red_freq
  let black_balls := total_balls * black_freq
  total_balls - red_balls - black_balls = 48 :=
by
  sorry

end number_of_white_balls_l2384_238470


namespace find_f3_l2384_238460

def f (a b c x : ℝ) : ℝ := a * x^5 + b * x^3 + c * x + 6

theorem find_f3 (a b c : ℝ) (h : f a b c (-3) = -12) : f a b c 3 = 24 :=
by
  sorry

end find_f3_l2384_238460


namespace find_a_l2384_238494

theorem find_a (a : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, f x = abs (2 * x - a) + a)
  (h2 : ∀ x : ℝ, f x ≤ 6 ↔ -2 ≤ x ∧ x ≤ 3) : 
  a = 1 := by
  sorry

end find_a_l2384_238494


namespace possible_values_of_x_l2384_238459

theorem possible_values_of_x (x : ℕ) (h1 : ∃ k : ℕ, k * k = 8 - x) (h2 : 1 ≤ x ∧ x ≤ 8) :
  x = 4 ∨ x = 7 ∨ x = 8 :=
by
  sorry

end possible_values_of_x_l2384_238459


namespace relationship_between_heights_is_correlated_l2384_238472

theorem relationship_between_heights_is_correlated :
  (∃ r : ℕ, (r = 1 ∨ r = 2 ∨ r = 3 ∨ r = 4) ∧ r = 2) := by
  sorry

end relationship_between_heights_is_correlated_l2384_238472


namespace show_spiders_l2384_238461

noncomputable def spiders_found (ants : ℕ) (ladybugs_initial : ℕ) (ladybugs_fly_away : ℕ) (total_insects_remaining : ℕ) : ℕ :=
  let ladybugs_remaining := ladybugs_initial - ladybugs_fly_away
  let insects_observed := ants + ladybugs_remaining
  total_insects_remaining - insects_observed

theorem show_spiders
  (ants : ℕ := 12)
  (ladybugs_initial : ℕ := 8)
  (ladybugs_fly_away : ℕ := 2)
  (total_insects_remaining : ℕ := 21) :
  spiders_found ants ladybugs_initial ladybugs_fly_away total_insects_remaining = 3 := by
  sorry

end show_spiders_l2384_238461


namespace simplify_expression_l2384_238440

theorem simplify_expression :
  ((3 + 4 + 5 + 6) ^ 2 / 4) + ((3 * 6 + 9) ^ 2 / 3) = 324 := 
  sorry

end simplify_expression_l2384_238440


namespace cost_of_each_new_shirt_l2384_238487

theorem cost_of_each_new_shirt (pants_cost shorts_cost shirts_cost : ℕ)
  (pants_sold shorts_sold shirts_sold : ℕ) (money_left : ℕ) (new_shirts : ℕ)
  (h₁ : pants_cost = 5) (h₂ : shorts_cost = 3) (h₃ : shirts_cost = 4)
  (h₄ : pants_sold = 3) (h₅ : shorts_sold = 5) (h₆ : shirts_sold = 5)
  (h₇ : money_left = 30) (h₈ : new_shirts = 2) :
  (pants_cost * pants_sold + shorts_cost * shorts_sold + shirts_cost * shirts_sold - money_left) / new_shirts = 10 :=
by sorry

end cost_of_each_new_shirt_l2384_238487


namespace length_of_each_cut_section_xiao_hong_age_l2384_238431

theorem length_of_each_cut_section (x : ℝ) (h : 60 - 2 * x = 10) : x = 25 := sorry

theorem xiao_hong_age (y : ℝ) (h : 2 * y + 10 = 30) : y = 10 := sorry

end length_of_each_cut_section_xiao_hong_age_l2384_238431


namespace probability_red_or_white_l2384_238485

noncomputable def total_marbles : ℕ := 50
noncomputable def blue_marbles : ℕ := 5
noncomputable def red_marbles : ℕ := 9
noncomputable def white_marbles : ℕ := total_marbles - (blue_marbles + red_marbles)

theorem probability_red_or_white : 
  (red_marbles + white_marbles) / total_marbles = 9 / 10 :=
by sorry

end probability_red_or_white_l2384_238485


namespace solve_inequalities_l2384_238411

theorem solve_inequalities (a b : ℝ) :
  (∀ x : ℝ, -1 < x ∧ x < 3 → x - a < 1 ∧ x - 2 * b > 3) ↔ (a = 2 ∧ b = -2) := 
  by 
    sorry

end solve_inequalities_l2384_238411


namespace exists_coeff_less_than_neg_one_l2384_238400

theorem exists_coeff_less_than_neg_one 
  (P : Polynomial ℤ)
  (h1 : P.eval 1 = 0)
  (h2 : P.eval 2 = 0) :
  ∃ i, P.coeff i < -1 := sorry

end exists_coeff_less_than_neg_one_l2384_238400


namespace coordinates_of_focus_with_greater_x_coordinate_l2384_238403

noncomputable def focus_of_ellipse_with_greater_x_coordinate : (ℝ × ℝ) :=
  let center : ℝ × ℝ := (3, -2)
  let a : ℝ := 3 -- semi-major axis length
  let b : ℝ := 2 -- semi-minor axis length
  let c : ℝ := Real.sqrt (a^2 - b^2)
  let focus_x : ℝ := 3 + c
  (focus_x, -2)

theorem coordinates_of_focus_with_greater_x_coordinate :
  focus_of_ellipse_with_greater_x_coordinate = (3 + Real.sqrt 5, -2) := 
sorry

end coordinates_of_focus_with_greater_x_coordinate_l2384_238403


namespace no_perf_square_of_prime_three_digit_l2384_238407

theorem no_perf_square_of_prime_three_digit {A B C : ℕ} (h_prime: Prime (100 * A + 10 * B + C)) : ¬ ∃ n : ℕ, B^2 - 4 * A * C = n^2 :=
by
  sorry

end no_perf_square_of_prime_three_digit_l2384_238407


namespace find_b_minus_a_l2384_238405

theorem find_b_minus_a (a b : ℝ) (h : ∀ x : ℝ, 0 ≤ x → 
  0 ≤ x^4 - x^3 + a * x + b ∧ x^4 - x^3 + a * x + b ≤ (x^2 - 1)^2) : 
  b - a = 2 :=
sorry

end find_b_minus_a_l2384_238405


namespace part_one_part_two_l2384_238456

variable {x : ℝ}

def setA (a : ℝ) : Set ℝ := {x | 0 < a * x + 1 ∧ a * x + 1 ≤ 5}
def setB : Set ℝ := {x | -1 / 2 < x ∧ x ≤ 2}

theorem part_one (a : ℝ) (h : a = 1) : setB ⊆ setA a :=
by
  sorry

theorem part_two (a : ℝ) : (setA a ⊆ setB) ↔ (a < -8 ∨ a ≥ 2) :=
by
  sorry

end part_one_part_two_l2384_238456


namespace monthly_income_l2384_238479

variable {I : ℝ} -- George's monthly income

def donated_to_charity (I : ℝ) := 0.60 * I -- 60% of the income left
def paid_in_taxes (I : ℝ) := 0.75 * donated_to_charity I -- 75% of the remaining income after donation
def saved_for_future (I : ℝ) := 0.80 * paid_in_taxes I -- 80% of the remaining income after taxes
def expenses (I : ℝ) := saved_for_future I - 125 -- Remaining income after groceries and transportation expenses
def remaining_for_entertainment := 150 -- $150 left for entertainment and miscellaneous expenses

theorem monthly_income : I = 763.89 := 
by
  -- Using the conditions of the problem
  sorry

end monthly_income_l2384_238479


namespace sequence_AMS_ends_in_14_l2384_238418

def start := 3
def add_two (x : ℕ) := x + 2
def multiply_three (x : ℕ) := x * 3
def subtract_one (x : ℕ) := x - 1

theorem sequence_AMS_ends_in_14 : 
  subtract_one (multiply_three (add_two start)) = 14 :=
by
  -- The proof would go here if required.
  sorry

end sequence_AMS_ends_in_14_l2384_238418


namespace handshakes_count_l2384_238458

-- Define the parameters
def teams : ℕ := 3
def players_per_team : ℕ := 7
def referees : ℕ := 3

-- Calculate handshakes among team members
def handshakes_among_teams :=
  let unique_handshakes_per_team := players_per_team * 2 * players_per_team / 2
  unique_handshakes_per_team * teams

-- Calculate handshakes between players and referees
def players_shake_hands_with_referees :=
  teams * players_per_team * referees

-- Calculate total handshakes
def total_handshakes :=
  handshakes_among_teams + players_shake_hands_with_referees

-- Proof statement
theorem handshakes_count : total_handshakes = 210 := by
  sorry

end handshakes_count_l2384_238458


namespace count_valid_n_le_30_l2384_238429

theorem count_valid_n_le_30 :
  ∀ n : ℕ, (0 < n ∧ n ≤ 30) → (n! * 2) % (n * (n + 1)) = 0 := by
  sorry

end count_valid_n_le_30_l2384_238429


namespace urn_marbles_100_white_l2384_238420

theorem urn_marbles_100_white 
(initial_white initial_black final_white final_black : ℕ) 
(h_initial : initial_white = 150 ∧ initial_black = 50)
(h_operations : 
  (∀ n, (initial_white - 3 * n + 2 * n = final_white ∧ initial_black + n = final_black) ∨
  (initial_white - 2 * n - 1 = initial_white ∧ initial_black = final_black) ∨
  (initial_white - 1 * n - 2 = final_white ∧ initial_black - 1 * n = final_black) ∨
  (initial_white - 3 * n + 2 = final_white ∧ initial_black + 1 * n = final_black)) →
  ((initial_white = 150 ∧ initial_black = 50) →
   ∃ m: ℕ, final_white = 100)) :
∃ n: ℕ, initial_white - 3 * n + 2 * n = 100 ∧ initial_black + n = final_black :=
sorry

end urn_marbles_100_white_l2384_238420


namespace strictly_increasing_0_to_e_l2384_238465

noncomputable def ln (x : ℝ) : ℝ := Real.log x

noncomputable def f (x : ℝ) : ℝ := ln x / x

theorem strictly_increasing_0_to_e :
  ∀ x : ℝ, 0 < x ∧ x < Real.exp 1 → 0 < (1 - ln x) / (x^2) :=
by
  sorry

end strictly_increasing_0_to_e_l2384_238465


namespace correct_removal_of_parentheses_l2384_238415

theorem correct_removal_of_parentheses (x : ℝ) : (1/3) * (6 * x - 3) = 2 * x - 1 :=
by sorry

end correct_removal_of_parentheses_l2384_238415


namespace shaina_chocolate_amount_l2384_238450

variable (total_chocolate : ℚ) (num_piles : ℕ) (fraction_kept : ℚ)
variable (eq_total_chocolate : total_chocolate = 72 / 7)
variable (eq_num_piles : num_piles = 6)
variable (eq_fraction_kept : fraction_kept = 1 / 3)

theorem shaina_chocolate_amount :
  (total_chocolate / num_piles) * (1 - fraction_kept) = 8 / 7 :=
by
  sorry

end shaina_chocolate_amount_l2384_238450


namespace train_speed_is_300_kmph_l2384_238441

noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  (length / 1000) / (time / 3600)

theorem train_speed_is_300_kmph :
  train_speed 1250 15 = 300 := by
  sorry

end train_speed_is_300_kmph_l2384_238441


namespace parabola_translation_left_by_two_units_l2384_238451

/-- 
The parabola y = x^2 + 4x + 5 is obtained by translating the parabola y = x^2 + 1. 
Prove that this translation is 2 units to the left.
-/
theorem parabola_translation_left_by_two_units :
  ∀ x : ℝ, (x^2 + 4*x + 5) = ((x+2)^2 + 1) :=
by
  intro x
  sorry

end parabola_translation_left_by_two_units_l2384_238451


namespace intersection_in_quadrant_II_l2384_238424

theorem intersection_in_quadrant_II (x y : ℝ) 
  (h1: y ≥ -2 * x + 3) 
  (h2: y ≤ 3 * x + 6) 
  (h_intersection: x = -3 / 5 ∧ y = 21 / 5) :
  x < 0 ∧ y > 0 := 
sorry

end intersection_in_quadrant_II_l2384_238424


namespace f_has_two_zeros_l2384_238455

def f (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 1

theorem f_has_two_zeros : ∃ (x1 x2 : ℝ), f x1 = 0 ∧ f x2 = 0 ∧ x1 ≠ x2 := 
by
  sorry

end f_has_two_zeros_l2384_238455


namespace fraction_value_l2384_238475

variable (a b : ℚ)  -- Variables a and b are rational numbers

theorem fraction_value (h : a / 4 = b / 3) : (a - b) / b = 1 / 3 := by
  sorry

end fraction_value_l2384_238475


namespace g_sum_eq_neg_one_l2384_238491

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

-- Main theorem to prove g(1) + g(-1) = -1 given the conditions
theorem g_sum_eq_neg_one
  (h1 : ∀ x y : ℝ, f (x - y) = f x * g y - g x * f y)
  (h2 : f (-2) = f 1)
  (h3 : f 1 ≠ 0) :
  g 1 + g (-1) = -1 :=
sorry

end g_sum_eq_neg_one_l2384_238491


namespace max_p_l2384_238476

theorem max_p (p q r s t u v w : ℕ)
  (h1 : p + q + r + s = 35)
  (h2 : q + r + s + t = 35)
  (h3 : r + s + t + u = 35)
  (h4 : s + t + u + v = 35)
  (h5 : t + u + v + w = 35)
  (h6 : q + v = 14) :
  p ≤ 20 :=
sorry

end max_p_l2384_238476


namespace calculate_expression_l2384_238404

theorem calculate_expression :
  150 * (150 - 4) - (150 * 150 - 8 + 2^3) = -600 :=
by
  sorry

end calculate_expression_l2384_238404


namespace solve_equation_l2384_238495

theorem solve_equation (x : ℝ) (h : (x^2 - x + 2) / (x - 1) = x + 3) : x = 5 / 3 := 
by sorry

end solve_equation_l2384_238495


namespace value_of_f_at_3_l2384_238486

def f (x : ℝ) := 2 * x - 1

theorem value_of_f_at_3 : f 3 = 5 := by
  sorry

end value_of_f_at_3_l2384_238486


namespace max_value_of_d_l2384_238468

-- Define the conditions
variable (a b c d : ℝ) (h_sum : a + b + c + d = 10) 
          (h_prod_sum : ab + ac + ad + bc + bd + cd = 20)

-- Define the theorem statement
theorem max_value_of_d : 
  d ≤ (5 + Real.sqrt 105) / 2 :=
sorry

end max_value_of_d_l2384_238468


namespace merry_go_round_cost_per_child_l2384_238425

-- Definitions
def num_children := 5
def ferris_wheel_cost_per_child := 5
def num_children_on_ferris_wheel := 3
def ice_cream_cost_per_cone := 8
def ice_cream_cones_per_child := 2
def total_spent := 110

-- Totals
def ferris_wheel_total_cost := num_children_on_ferris_wheel * ferris_wheel_cost_per_child
def ice_cream_total_cost := num_children * ice_cream_cones_per_child * ice_cream_cost_per_cone
def merry_go_round_total_cost := total_spent - ferris_wheel_total_cost - ice_cream_total_cost

-- Final proof statement
theorem merry_go_round_cost_per_child : 
  merry_go_round_total_cost / num_children = 3 :=
by
  -- We skip the actual proof here
  sorry

end merry_go_round_cost_per_child_l2384_238425


namespace A_union_B_eq_B_l2384_238452

-- Define set A
def A : Set ℝ := {-1, 0, 1}

-- Define set B
def B : Set ℝ := {y | ∃ x : ℝ, y = Real.sin x}

-- The proof problem
theorem A_union_B_eq_B : A ∪ B = B := 
  sorry

end A_union_B_eq_B_l2384_238452


namespace range_of_a_l2384_238437

theorem range_of_a (a : ℝ) :
  (∀ x : ℕ, 0 < x ∧ 3*x + a ≤ 2 → x = 1 ∨ x = 2) ↔ (-7 < a ∧ a ≤ -4) :=
sorry

end range_of_a_l2384_238437


namespace distance_to_SFL_is_81_l2384_238473

variable (Speed : ℝ)
variable (Time : ℝ)

def distance_to_SFL (Speed : ℝ) (Time : ℝ) := Speed * Time

theorem distance_to_SFL_is_81 : distance_to_SFL 27 3 = 81 :=
by
  sorry

end distance_to_SFL_is_81_l2384_238473


namespace range_of_d_l2384_238446

theorem range_of_d (a_1 d : ℝ) (h : (a_1 + 2 * d) * (a_1 + 3 * d) + 1 = 0) :
  d ∈ Set.Iic (-2) ∪ Set.Ici 2 :=
sorry

end range_of_d_l2384_238446


namespace alicia_satisfaction_l2384_238444

theorem alicia_satisfaction (t : ℚ) (h_sat : t * (12 - t) = (4 - t) * (2 * t + 2)) : t = 2 :=
by
  sorry

end alicia_satisfaction_l2384_238444


namespace ab_eq_six_l2384_238442

theorem ab_eq_six (a b : ℚ) (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end ab_eq_six_l2384_238442


namespace exists_square_with_digit_sum_2002_l2384_238436

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem exists_square_with_digit_sum_2002 :
  ∃ (n : ℕ), sum_of_digits (n^2) = 2002 :=
sorry

end exists_square_with_digit_sum_2002_l2384_238436


namespace remaining_slices_after_weekend_l2384_238499

theorem remaining_slices_after_weekend 
  (initial_pies : ℕ) (slices_per_pie : ℕ) (rebecca_initial_slices : ℕ) 
  (family_fraction : ℚ) (sunday_evening_slices : ℕ) : 
  initial_pies = 2 → 
  slices_per_pie = 8 → 
  rebecca_initial_slices = 2 → 
  family_fraction = 0.5 → 
  sunday_evening_slices = 2 → 
  (initial_pies * slices_per_pie 
   - rebecca_initial_slices 
   - family_fraction * (initial_pies * slices_per_pie - rebecca_initial_slices) 
   - sunday_evening_slices) = 5 :=
by 
  intros initial_pies_eq slices_per_pie_eq rebecca_initial_slices_eq family_fraction_eq sunday_evening_slices_eq
  sorry

end remaining_slices_after_weekend_l2384_238499


namespace max_principals_in_10_years_l2384_238490

theorem max_principals_in_10_years :
  ∀ (term_length : ℕ) (P : ℕ → Prop),
  (∀ n, P n → 3 ≤ n ∧ n ≤ 5) → 
  ∃ (n : ℕ), (n ≤ 10 / 3 ∧ P n) ∧ n = 3 :=
by
  sorry

end max_principals_in_10_years_l2384_238490


namespace infinite_series_converges_l2384_238410

open BigOperators

noncomputable def problem : ℝ :=
  ∑' n : ℕ, if n > 0 then (3 * n - 2) / (n * (n + 1) * (n + 3)) else 0

theorem infinite_series_converges : problem = 61 / 24 :=
sorry

end infinite_series_converges_l2384_238410


namespace first_prize_ticket_numbers_l2384_238447

theorem first_prize_ticket_numbers :
  {n : ℕ | n < 10000 ∧ (n % 1000 = 418)} = {418, 1418, 2418, 3418, 4418, 5418, 6418, 7418, 8418, 9418} :=
by
  sorry

end first_prize_ticket_numbers_l2384_238447


namespace lcm_gcd_product_l2384_238406

def a : ℕ := 20 -- Defining the first number as 20
def b : ℕ := 90 -- Defining the second number as 90

theorem lcm_gcd_product : Nat.lcm a b * Nat.gcd a b = 1800 := 
by 
  -- Computation and proof steps would go here
  sorry -- Replace with actual proof

end lcm_gcd_product_l2384_238406


namespace equilibrium_mass_l2384_238443

variable (l m2 S g : ℝ) (m1 : ℝ)

-- Given conditions
def length_of_rod : ℝ := 0.5 -- length l in meters
def mass_of_rod : ℝ := 2 -- mass m2 in kg
def distance_S : ℝ := 0.1 -- distance S in meters
def gravity : ℝ := 9.8 -- gravitational acceleration in m/s^2

-- Equivalence statement
theorem equilibrium_mass (h1 : l = length_of_rod)
                         (h2 : m2 = mass_of_rod)
                         (h3 : S = distance_S)
                         (h4 : g = gravity) :
  m1 = 10 := sorry

end equilibrium_mass_l2384_238443


namespace find_a_of_square_roots_l2384_238426

theorem find_a_of_square_roots (a : ℤ) (n : ℤ) (h₁ : 2 * a + 1 = n) (h₂ : a + 5 = n) : a = 4 :=
by
  -- proof goes here
  sorry

end find_a_of_square_roots_l2384_238426


namespace original_number_of_candies_l2384_238422

theorem original_number_of_candies (x : ℝ) (h₀ : x * (0.7 ^ 3) = 40) : x = 117 :=
by 
  sorry

end original_number_of_candies_l2384_238422


namespace stuffed_animal_cost_l2384_238482

variables 
  (M S A A_single C : ℝ)
  (Coupon_discount : ℝ)
  (Maximum_budget : ℝ)

noncomputable def conditions : Prop :=
  M = 6 ∧
  M = 3 * S ∧
  M = A / 4 ∧
  A_single = A / 2 ∧
  C = A_single / 2 ∧
  C = 2 * S ∧
  Coupon_discount = 0.10 ∧
  Maximum_budget = 30

theorem stuffed_animal_cost (h : conditions M S A A_single C Coupon_discount Maximum_budget) :
  A_single = 12 :=
sorry

end stuffed_animal_cost_l2384_238482


namespace length_AD_l2384_238484

open Real

-- Define the properties of the quadrilateral
variable (A B C D: Point)
variable (angle_ABC angle_BCD: ℝ)
variable (AB BC CD: ℝ)

-- Given conditions
axiom angle_ABC_eq_135 : angle_ABC = 135 * π / 180
axiom angle_BCD_eq_120 : angle_BCD = 120 * π / 180
axiom AB_eq_sqrt_6 : AB = sqrt 6
axiom BC_eq_5_minus_sqrt_3 : BC = 5 - sqrt 3
axiom CD_eq_6 : CD = 6

-- The theorem to prove
theorem length_AD {AD : ℝ} (h : True) :
  AD = 2 * sqrt 19 :=
sorry

end length_AD_l2384_238484


namespace necessary_but_not_sufficient_l2384_238413

theorem necessary_but_not_sufficient (m : ℝ) :
  (∀ x : ℝ, x^2 + m * x + 1 > 0) ↔ -2 < m ∧ m < 2 → m < 2 :=
by
  sorry

end necessary_but_not_sufficient_l2384_238413


namespace find_x_l2384_238480

variable (x y : ℝ)

theorem find_x (h1 : 0 < x) (h2 : 0 < y) (h3 : 5 * x^2 + 10 * x * y = x^3 + 2 * x^2 * y) : x = 5 := by
  sorry

end find_x_l2384_238480

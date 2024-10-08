import Mathlib

namespace bus_capacity_l35_35095

def left_side_seats : ℕ := 15
def seats_difference : ℕ := 3
def people_per_seat : ℕ := 3
def back_seat_capacity : ℕ := 7

theorem bus_capacity : left_side_seats + (left_side_seats - seats_difference) * people_per_seat + back_seat_capacity = 88 := 
by
  sorry

end bus_capacity_l35_35095


namespace quadratic_equation_root_zero_l35_35384

/-- Given that x = -3 is a root of the quadratic equation x^2 + 3x + k = 0,
    prove that the other root of the equation is 0 and k = 0. -/
theorem quadratic_equation_root_zero (k : ℝ) (h : -3^2 + 3 * -3 + k = 0) :
  (∀ t : ℝ, t^2 + 3 * t + k = 0 → t = 0) ∧ k = 0 :=
sorry

end quadratic_equation_root_zero_l35_35384


namespace starting_number_divisible_by_3_count_l35_35848

-- Define a predicate for divisibility by 3
def divisible_by_3 (n : ℕ) : Prop :=
  n % 3 = 0

-- Define the main theorem
theorem starting_number_divisible_by_3_count : 
  ∃ n : ℕ, (∀ m, n ≤ m ∧ m ≤ 50 → divisible_by_3 m → ∃ s, (m = n + 3 * s) ∧ s < 13) ∧
           (∀ k : ℕ, (divisible_by_3 k) → n ≤ k ∧ k ≤ 50 → m = 12) :=
sorry

end starting_number_divisible_by_3_count_l35_35848


namespace remainder_polynomial_division_l35_35842

noncomputable def remainder_division : Polynomial ℝ := 
  (Polynomial.X ^ 4 + Polynomial.X ^ 3 - 4 * Polynomial.X + 1) % (Polynomial.X ^ 3 - 1)

theorem remainder_polynomial_division :
  remainder_division = -3 * Polynomial.X + 2 :=
by
  sorry

end remainder_polynomial_division_l35_35842


namespace probability_of_drawing_white_ball_l35_35816

-- Define initial conditions
def initial_balls : ℕ := 6
def total_balls_after_white : ℕ := initial_balls + 1
def number_of_white_balls : ℕ := 1
def number_of_total_balls : ℕ := total_balls_after_white

-- Define the probability of drawing a white ball
def probability_of_white : ℚ := number_of_white_balls / number_of_total_balls

-- Statement to be proved
theorem probability_of_drawing_white_ball :
  probability_of_white = 1 / 7 :=
by
  sorry

end probability_of_drawing_white_ball_l35_35816


namespace tree_height_l35_35534

theorem tree_height (future_height : ℕ) (growth_per_year : ℕ) (years : ℕ) (inches_per_foot : ℕ) :
  future_height = 1104 →
  growth_per_year = 5 →
  years = 8 →
  inches_per_foot = 12 →
  (future_height / inches_per_foot - growth_per_year * years) = 52 := 
by
  intros h1 h2 h3 h4
  sorry

end tree_height_l35_35534


namespace polynomial_root_sum_l35_35863

theorem polynomial_root_sum 
  (c d : ℂ) 
  (h1 : c + d = 6) 
  (h2 : c * d = 10) 
  (h3 : c^2 - 6 * c + 10 = 0) 
  (h4 : d^2 - 6 * d + 10 = 0) : 
  c^3 + c^5 * d^3 + c^3 * d^5 + d^3 = 16156 := 
by sorry

end polynomial_root_sum_l35_35863


namespace magnitude_2a_sub_b_l35_35543

def a : ℝ × ℝ := (1, 3)
def b : ℝ × ℝ := (-1, 2)

theorem magnitude_2a_sub_b : (‖(2 * a.1 - b.1, 2 * a.2 - b.2)‖ = 5) :=
by {
  sorry
}

end magnitude_2a_sub_b_l35_35543


namespace initial_earning_members_l35_35385

theorem initial_earning_members (n T : ℕ)
  (h₁ : T = n * 782)
  (h₂ : T - 1178 = (n - 1) * 650) :
  n = 14 :=
by sorry

end initial_earning_members_l35_35385


namespace true_proposition_l35_35422

open Real

def proposition_p : Prop := ∃ x : ℝ, sin x < 1
def proposition_q : Prop := ∀ x : ℝ, exp (abs x) ≥ 1

theorem true_proposition : proposition_p ∧ proposition_q :=
by
  -- These definitions are directly from conditions.
  sorry

end true_proposition_l35_35422


namespace minimum_value_of_reciprocal_sum_l35_35023

theorem minimum_value_of_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h : 2 * a * (-1) - b * 2 + 2 = 0) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 2 * a * (-1) - b * 2 + 2 = 0 ∧ (a + b = 1) ∧ (a = 1/2 ∧ b = 1/2) ∧ (1/a + 1/b = 4) :=
by
  sorry

end minimum_value_of_reciprocal_sum_l35_35023


namespace find_two_angles_of_scalene_obtuse_triangle_l35_35958

def is_scalene (a b c : ℝ) : Prop := a ≠ b ∧ b ≠ c ∧ a ≠ c
def is_obtuse (a : ℝ) : Prop := a > 90
def is_triangle (a b c : ℝ) : Prop := a + b + c = 180

theorem find_two_angles_of_scalene_obtuse_triangle
  (a b c : ℝ)
  (ha : is_obtuse a) (h_scalene : is_scalene a b c) 
  (h_sum : is_triangle a b c) 
  (ha_val : a = 108)
  (h_half : b = 2 * c) :
  b = 48 ∧ c = 24 :=
by
  sorry

end find_two_angles_of_scalene_obtuse_triangle_l35_35958


namespace gcd_sum_equality_l35_35567

theorem gcd_sum_equality (n : ℕ) : 
  (Nat.gcd 6 n + Nat.gcd 8 (2 * n) = 10) ↔ 
  (∃ t : ℤ, n = 12 * t + 4 ∨ n = 12 * t + 6 ∨ n = 12 * t + 8) :=
by
  sorry

end gcd_sum_equality_l35_35567


namespace sum_of_last_two_digits_of_9pow25_plus_11pow25_eq_0_l35_35390

theorem sum_of_last_two_digits_of_9pow25_plus_11pow25_eq_0 :
  (9^25 + 11^25) % 100 = 0 := 
sorry

end sum_of_last_two_digits_of_9pow25_plus_11pow25_eq_0_l35_35390


namespace solve_nat_pairs_l35_35760

theorem solve_nat_pairs (n m : ℕ) :
  m^2 + 2 * 3^n = m * (2^(n+1) - 1) ↔ (n = 3 ∧ m = 6) ∨ (n = 3 ∧ m = 9) :=
by sorry

end solve_nat_pairs_l35_35760


namespace sector_area_l35_35833

theorem sector_area (alpha : ℝ) (r : ℝ) (h_alpha : alpha = Real.pi / 3) (h_r : r = 2) : 
  (1 / 2) * (alpha * r) * r = (2 * Real.pi) / 3 := 
by
  sorry

end sector_area_l35_35833


namespace certain_event_birthday_example_l35_35951
-- Import the necessary library

-- Define the problem with conditions
def certain_event_people_share_birthday (num_days : ℕ) (num_people : ℕ) : Prop :=
  num_people > num_days

-- Define a specific instance based on the given problem
theorem certain_event_birthday_example : certain_event_people_share_birthday 365 400 :=
by
  sorry

end certain_event_birthday_example_l35_35951


namespace parabola_focus_l35_35957

theorem parabola_focus (a : ℝ) (p : ℝ) (x y : ℝ) :
  a = -3 ∧ p = 6 →
  (y^2 = -2 * p * x) → 
  (y^2 = -12 * x) := 
by sorry

end parabola_focus_l35_35957


namespace pure_imaginary_number_l35_35115

theorem pure_imaginary_number (a : ℝ) (ha : (1 + a) / (1 + a^2) = 0) : a = -1 :=
sorry

end pure_imaginary_number_l35_35115


namespace ordered_triple_l35_35511

theorem ordered_triple (a b c : ℝ) (h1 : 4 < a) (h2 : 4 < b) (h3 : 4 < c) 
  (h_eq : (a + 3)^2 / (b + c - 3) + (b + 5)^2 / (c + a - 5) + (c + 7)^2 / (a + b - 7) = 45) 
  : (a, b, c) = (12, 10, 8) :=
  sorry

end ordered_triple_l35_35511


namespace percentage_sold_is_80_l35_35970

-- Definitions corresponding to conditions
def first_day_houses : Nat := 20
def items_per_house : Nat := 2
def total_items_sold : Nat := 104

-- Calculate the houses visited on the second day
def second_day_houses : Nat := 2 * first_day_houses

-- Calculate items sold on the first day
def items_sold_first_day : Nat := first_day_houses * items_per_house

-- Calculate items sold on the second day
def items_sold_second_day : Nat := total_items_sold - items_sold_first_day

-- Calculate houses sold to on the second day
def houses_sold_to_second_day : Nat := items_sold_second_day / items_per_house

-- Percentage calculation
def percentage_sold_second_day : Nat := (houses_sold_to_second_day * 100) / second_day_houses

-- Theorem proving that James sold to 80% of the houses on the second day
theorem percentage_sold_is_80 : percentage_sold_second_day = 80 := by
  sorry

end percentage_sold_is_80_l35_35970


namespace fraction_spent_toy_store_l35_35406

noncomputable def weekly_allowance : ℚ := 2.25
noncomputable def arcade_fraction_spent : ℚ := 3 / 5
noncomputable def candy_store_spent : ℚ := 0.60

theorem fraction_spent_toy_store :
  let remaining_after_arcade := weekly_allowance * (1 - arcade_fraction_spent)
  let spent_toy_store := remaining_after_arcade - candy_store_spent
  spent_toy_store / remaining_after_arcade = 1 / 3 :=
by
  sorry

end fraction_spent_toy_store_l35_35406


namespace anya_kolya_apples_l35_35081

theorem anya_kolya_apples (A K : ℕ) (h1 : A = (K * 100) / (A + K)) (h2 : K = (A * 100) / (A + K)) : A = 50 ∧ K = 50 :=
sorry

end anya_kolya_apples_l35_35081


namespace problem_solution_l35_35049

theorem problem_solution (x y : ℝ) (h1 : y = x / (3 * x + 1)) (hx : x ≠ 0) (hy : y ≠ 0) :
    (x - y + 3 * x * y) / (x * y) = 6 := by
  sorry

end problem_solution_l35_35049


namespace sum_of_roots_3x2_minus_12x_plus_12_eq_4_l35_35787

def sum_of_roots_quadratic (a b : ℚ) (h : a ≠ 0) : ℚ := -b / a

theorem sum_of_roots_3x2_minus_12x_plus_12_eq_4 :
  sum_of_roots_quadratic 3 (-12) (by norm_num) = 4 :=
sorry

end sum_of_roots_3x2_minus_12x_plus_12_eq_4_l35_35787


namespace statement_C_is_incorrect_l35_35234

noncomputable def g (x : ℝ) : ℝ := (2 * x + 3) / (x - 2)

theorem statement_C_is_incorrect : g (-2) ≠ 0 :=
by
  sorry

end statement_C_is_incorrect_l35_35234


namespace paper_pieces_l35_35956

theorem paper_pieces (n : ℕ) (h1 : 20 = 2 * n - 8) : n^2 + 20 = 216 := 
by
  sorry

end paper_pieces_l35_35956


namespace distance_between_x_intercepts_l35_35342

theorem distance_between_x_intercepts 
  (s1 s2 : ℝ) (P : ℝ × ℝ)
  (h1 : s1 = 2) 
  (h2 : s2 = -4) 
  (hP : P = (8, 20)) :
  let l1_x_intercept := (0 - (20 - P.2)) / s1 + P.1
  let l2_x_intercept := (0 - (20 - P.2)) / s2 + P.1
  abs (l1_x_intercept - l2_x_intercept) = 15 := 
sorry

end distance_between_x_intercepts_l35_35342


namespace triangle_is_equilateral_l35_35856

-- Define the triangle
structure Triangle :=
  (A B C : ℝ)

-- Define a triangle's circumradius and inradius
structure TriangleProperties :=
  (circumradius : ℝ)
  (inradius : ℝ)
  (circumcenter_incenter_sq_distance : ℝ) -- OI^2 = circumradius^2 - 2*circumradius*inradius

noncomputable def circumcenter_incenter_coincide (T : Triangle) (P : TriangleProperties) : Prop :=
  P.circumcenter_incenter_sq_distance = 0

theorem triangle_is_equilateral
  (T : Triangle)
  (P : TriangleProperties)
  (hR : P.circumradius = 2 * P.inradius)
  (hOI : circumcenter_incenter_coincide T P) :
  ∃ (R r : ℝ), T = {A := 1 * r, B := 1 * r, C := 1 * r} :=
by sorry

end triangle_is_equilateral_l35_35856


namespace equal_roots_B_value_l35_35670

theorem equal_roots_B_value (B : ℝ) :
  (∀ k : ℝ, (2 * k * x^2 + B * x + 2 = 0) → (k = 1 → (B^2 - 4 * (2 * 1) * 2 = 0))) → B = 4 ∨ B = -4 :=
by
  sorry

end equal_roots_B_value_l35_35670


namespace problem_2_8_3_4_7_2_2_l35_35426

theorem problem_2_8_3_4_7_2_2 : 2 + 8 * 3 - 4 + 7 * 2 / 2 = 29 := by
  sorry

end problem_2_8_3_4_7_2_2_l35_35426


namespace expression_value_at_2_l35_35151

theorem expression_value_at_2 : (2^2 + 3 * 2 - 4) = 6 :=
by 
  sorry

end expression_value_at_2_l35_35151


namespace sum_of_roots_of_qubic_polynomial_l35_35756

noncomputable def Q (a b c d : ℝ) (x : ℝ) : ℝ :=
  a * x^3 + b * x^2 + c * x + d

theorem sum_of_roots_of_qubic_polynomial (a b c d : ℝ) 
  (h₁ : ∀ x : ℝ, Q a b c d (x^4 + x) ≥ Q a b c d (x^3 + 1))
  (h₂ : Q a b c d 1 = 0) : 
  -b / a = 3 / 2 :=
sorry

end sum_of_roots_of_qubic_polynomial_l35_35756


namespace range_of_x_max_y_over_x_l35_35791

-- Define the circle and point P(x,y) on the circle
def CircleEquation (x y : ℝ) : Prop := (x - 4)^2 + (y - 3)^2 = 9

theorem range_of_x (x y : ℝ) (h : CircleEquation x y) : 1 ≤ x ∧ x ≤ 7 :=
sorry

theorem max_y_over_x (x y : ℝ) (h : CircleEquation x y) : ∀ k : ℝ, (k = y / x) → 0 ≤ k ∧ k ≤ (24 / 7) :=
sorry

end range_of_x_max_y_over_x_l35_35791


namespace smallest_k_satisfying_condition_l35_35513

def is_smallest_prime_greater_than (n : ℕ) (p : ℕ) : Prop :=
  Nat.Prime p ∧ n < p ∧ ∀ q, Nat.Prime q ∧ q > n → q >= p

def is_divisible_by (m k : ℕ) : Prop := k % m = 0

theorem smallest_k_satisfying_condition :
  ∃ k, is_smallest_prime_greater_than 19 23 ∧ is_divisible_by 3 k ∧ 64 ^ k > 4 ^ (19 * 23) ∧ (∀ k' < k, is_divisible_by 3 k' → 64 ^ k' ≤ 4 ^ (19 * 23)) :=
by
  sorry

end smallest_k_satisfying_condition_l35_35513


namespace circle_radius_l35_35575

theorem circle_radius (r : ℝ) (h : 3 * 2 * Real.pi * r = 2 * Real.pi * r^2) : r = 3 :=
by
  sorry

end circle_radius_l35_35575


namespace sum_int_values_l35_35414

theorem sum_int_values (sum : ℤ) : 
  (∀ n : ℤ, (20 % (2 * n - 1) = 0) → sum = 2) :=
by
  sorry

end sum_int_values_l35_35414


namespace principal_amount_l35_35293

theorem principal_amount (SI R T : ℕ) (P : ℕ) : SI = 160 ∧ R = 5 ∧ T = 4 → P = 800 :=
by
  sorry

end principal_amount_l35_35293


namespace swap_square_digit_l35_35102

theorem swap_square_digit (n : ℕ) (h1 : n ≥ 10 ∧ n < 100) : 
  ∃ (x y : ℕ), n = 10 * x + y ∧ (x < 10 ∧ y < 10) ∧ (y * 100 + x * 10 + y^2 + 20 * x * y - 1) = n * n + 2 * n + 1 :=
by 
    sorry

end swap_square_digit_l35_35102


namespace choosing_one_student_is_50_l35_35617

-- Define the number of male students and female students
def num_male_students : Nat := 26
def num_female_students : Nat := 24

-- Define the total number of ways to choose one student
def total_ways_to_choose_one_student : Nat := num_male_students + num_female_students

-- Theorem statement proving the total number of ways to choose one student is 50
theorem choosing_one_student_is_50 : total_ways_to_choose_one_student = 50 := by
  sorry

end choosing_one_student_is_50_l35_35617


namespace solve_for_y_l35_35417

theorem solve_for_y (y : ℝ) : (y^2 + 6 * y + 8 = -(y + 4) * (y + 6)) → y = -4 :=
by {
  sorry
}

end solve_for_y_l35_35417


namespace tree_planting_equation_l35_35117

theorem tree_planting_equation (x : ℝ) (h : x > 0) : 
  180 / x - 180 / (1.5 * x) = 2 :=
sorry

end tree_planting_equation_l35_35117


namespace correct_option_A_l35_35392

theorem correct_option_A : 
  (∀ a : ℝ, a^3 * a^4 = a^7) ∧ 
  ¬ (∀ a : ℝ, a^6 / a^2 = a^3) ∧ 
  ¬ (∀ a : ℝ, a^4 - a^2 = a^2) ∧ 
  ¬ (∀ a b : ℝ, (a - b)^2 = a^2 - b^2) :=
by
  /- omitted proofs -/
  sorry

end correct_option_A_l35_35392


namespace male_students_count_l35_35077

theorem male_students_count :
  ∃ (N M : ℕ), 
  (N % 4 = 2) ∧ 
  (N % 5 = 1) ∧ 
  (N = M + 15) ∧ 
  (15 > M) ∧ 
  (M = 11) :=
sorry

end male_students_count_l35_35077


namespace jane_reads_pages_l35_35898

theorem jane_reads_pages (P : ℕ) (h1 : 7 * (P + 10) = 105) : P = 5 := by
  sorry

end jane_reads_pages_l35_35898


namespace line_through_perpendicular_l35_35802

theorem line_through_perpendicular (x y : ℝ) :
  (∃ (k : ℝ), (2 * x - y + 3 = 0) ∧ k = - 1 / 2) →
  (∃ (a b c : ℝ), (a * (-1) + b * 1 + c = 0) ∧ a = 1 ∧ b = 2 ∧ c = -1) :=
by
  sorry

end line_through_perpendicular_l35_35802


namespace rhombus_area_l35_35231

theorem rhombus_area (d1 d2 : ℝ) (h1 : d1 = 6) (h2 : d2 = 8) : 
  (1 / 2) * d1 * d2 = 24 :=
by {
  sorry
}

end rhombus_area_l35_35231


namespace unique_solution_of_abc_l35_35812

theorem unique_solution_of_abc (a b c : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) 
  (h_lt_ab_c : a < b) (h_lt_b_c: b < c) (h_eq_abc : a * b + b * c + c * a = a * b * c) : 
  a = 2 ∧ b = 3 ∧ c = 6 :=
by {
  -- Proof skipped, only the statement is provided.
  sorry
}

end unique_solution_of_abc_l35_35812


namespace range_of_a_l35_35389

theorem range_of_a {a : ℝ} (h : (a^2) / 4 + 1 / 2 < 1) : -Real.sqrt 2 < a ∧ a < Real.sqrt 2 :=
sorry

end range_of_a_l35_35389


namespace lina_walk_probability_l35_35027

/-- Total number of gates -/
def num_gates : ℕ := 20

/-- Distance between adjacent gates in feet -/
def gate_distance : ℕ := 50

/-- Maximum distance in feet Lina can walk to be within the desired range -/
def max_walk_distance : ℕ := 200

/-- Number of gates Lina can move within the max walk distance -/
def max_gates_within_distance : ℕ := max_walk_distance / gate_distance

/-- Total possible gate pairs for initial and new gate selection -/
def total_possible_pairs : ℕ := num_gates * (num_gates - 1)

/-- Total number of favorable gate pairs where walking distance is within the allowed range -/
def total_favorable_pairs : ℕ :=
  let edge_favorable (g : ℕ) := if g = 1 ∨ g = num_gates then 4
                                else if g = 2 ∨ g = num_gates - 1 then 5
                                else if g = 3 ∨ g = num_gates - 2 then 6
                                else if g = 4 ∨ g = num_gates - 3 then 7 else 8
  (edge_favorable 1) + (edge_favorable 2) + (edge_favorable 3) +
  (edge_favorable 4) + (num_gates - 8) * 8

/-- Probability that Lina walks 200 feet or less expressed as a reduced fraction -/
def probability_within_distance : ℚ :=
  (total_favorable_pairs : ℚ) / (total_possible_pairs : ℚ)

/-- p and q components of the fraction representing the probability -/
def p := 7
def q := 19

/-- Sum of p and q -/
def p_plus_q : ℕ := p + q

theorem lina_walk_probability : p_plus_q = 26 := by sorry

end lina_walk_probability_l35_35027


namespace solve_seating_problem_l35_35800

-- Define the conditions of the problem
def valid_seating_arrangements (n : ℕ) : Prop :=
  (∃ (x y : ℕ), x < y ∧ x + 1 < y ∧ y < n ∧ 
    (n ≥ 5 ∧ y - x - 1 > 0)) ∧
  (∃! (x' y' : ℕ), x' < y' ∧ x' + 1 < y' ∧ y' < n ∧ 
    (n ≥ 5 ∧ y' - x' - 1 > 0))

-- State the theorem
theorem solve_seating_problem : ∃ n : ℕ, valid_seating_arrangements n ∧ n = 5 :=
by
  sorry

end solve_seating_problem_l35_35800


namespace determine_b_l35_35949

noncomputable def has_exactly_one_real_solution (b : ℝ) : Prop :=
  ∃ x : ℝ, x^4 - b*x^3 - 3*b*x + b^2 - 2 = 0 ∧ ∀ y : ℝ, y ≠ x → y^4 - b*y^3 - 3*b*y + b^2 - 2 ≠ 0

theorem determine_b (b : ℝ) :
  has_exactly_one_real_solution b → b < 7 / 4 :=
by
  sorry

end determine_b_l35_35949


namespace linear_condition_l35_35130

theorem linear_condition (a : ℝ) : a ≠ 0 ↔ ∃ (x y : ℝ), ax + y = -1 :=
by
  sorry

end linear_condition_l35_35130


namespace Vasya_fraction_impossible_l35_35556

theorem Vasya_fraction_impossible
  (a b n : ℕ) (h_ab : a < b) (h_na : n < a) (h_nb : n < b)
  (h1 : (a + n) / (b + n) > 3 * a / (2 * b))
  (h2 : (a - n) / (b - n) > a / (2 * b)) : false :=
by
  sorry

end Vasya_fraction_impossible_l35_35556


namespace not_possible_to_create_3_piles_l35_35402

def similar_sizes (a b : ℝ) : Prop := a / b ≤ Real.sqrt 2

theorem not_possible_to_create_3_piles (x : ℝ) (hx : 0 < x) : ¬ ∃ (y z w : ℝ), 
  y + z + w = x ∧ 
  similar_sizes y z ∧ similar_sizes z w ∧ similar_sizes y w := 
by 
  sorry

end not_possible_to_create_3_piles_l35_35402


namespace original_numbers_correct_l35_35520

noncomputable def restore_original_numbers : List ℕ :=
  let T : ℕ := 5
  let EL : ℕ := 12
  let EK : ℕ := 19
  let LA : ℕ := 26
  let SS : ℕ := 33
  [T, EL, EK, LA, SS]

theorem original_numbers_correct :
  restore_original_numbers = [5, 12, 19, 26, 33] :=
by
  sorry

end original_numbers_correct_l35_35520


namespace question_1_question_2_question_3_l35_35498

theorem question_1 (m : ℝ) : 
  (∀ x : ℝ, (m + 1) * x^2 - (m - 1) * x + (m - 1) < 1) ↔ 
    m < (1 - 2 * Real.sqrt 7) / 3 := sorry

theorem question_2 (m : ℝ) : 
  ∀ x : ℝ, (m + 1) * x^2 - (m - 1) * x + (m - 1) ≥ (m + 1) * x := sorry

theorem question_3 (m : ℝ) : 
  (∀ x ∈ Set.Icc (-1/2 : ℝ) (1/2), (m + 1) * x^2 - (m - 1) * x + (m - 1) ≥ 0) ↔ 
    m ≥ 1 := sorry

end question_1_question_2_question_3_l35_35498


namespace smallest_consecutive_natural_number_sum_l35_35965

theorem smallest_consecutive_natural_number_sum (a n : ℕ) (hn : n > 1) (h : n * a + (n * (n - 1)) / 2 = 2016) :
  ∃ a, a = 1 :=
by
  sorry

end smallest_consecutive_natural_number_sum_l35_35965


namespace sleepySquirrelNutsPerDay_l35_35175

def twoBusySquirrelsNutsPerDay : ℕ := 2 * 30
def totalDays : ℕ := 40
def totalNuts : ℕ := 3200

theorem sleepySquirrelNutsPerDay 
  (s  : ℕ) 
  (h₁ : 2 * 30 * totalDays + s * totalDays = totalNuts) 
  : s = 20 := 
  sorry

end sleepySquirrelNutsPerDay_l35_35175


namespace smaller_angle_is_85_l35_35104

-- Conditions
def isParallelogram (α β : ℝ) : Prop :=
  α + β = 180

def angleExceedsBy10 (α β : ℝ) : Prop :=
  β = α + 10

-- Proof Problem
theorem smaller_angle_is_85 (α β : ℝ)
  (h1 : isParallelogram α β)
  (h2 : angleExceedsBy10 α β) :
  α = 85 :=
by
  sorry

end smaller_angle_is_85_l35_35104


namespace sum_single_digits_l35_35764

theorem sum_single_digits (P Q R : ℕ) (hP : P ≠ Q) (hQ : Q ≠ R) (hR : R ≠ P)
  (h1 : R + R = 10)
  (h_sum : ∃ (P Q R : ℕ), P * 100 + 70 + R + 390 + R = R * 100 + Q * 10) :
  P + Q + R = 13 := 
sorry

end sum_single_digits_l35_35764


namespace intersection_points_of_curve_with_axes_l35_35012

theorem intersection_points_of_curve_with_axes :
  (∃ t : ℝ, (-2 + 5 * t = 0) ∧ (1 - 2 * t = 1/5)) ∧
  (∃ t : ℝ, (1 - 2 * t = 0) ∧ (-2 + 5 * t = 1/2)) :=
by {
  -- Proving the intersection points with the coordinate axes
  sorry
}

end intersection_points_of_curve_with_axes_l35_35012


namespace right_building_shorter_l35_35623

-- Define the conditions as hypotheses
def middle_building_height : ℕ := 100
def left_building_height : ℕ := (80 * middle_building_height) / 100
def combined_height_left_middle : ℕ := left_building_height + middle_building_height
def total_height : ℕ := 340
def right_building_height : ℕ := total_height - combined_height_left_middle

-- Define the statement we need to prove
theorem right_building_shorter :
  combined_height_left_middle - right_building_height = 20 :=
by sorry

end right_building_shorter_l35_35623


namespace natasha_average_speed_l35_35329

theorem natasha_average_speed :
  ∀ (time_up time_down : ℝ) (speed_up : ℝ),
  time_up = 4 →
  time_down = 2 →
  speed_up = 2.25 →
  (2 * (time_up * speed_up) / (time_up + time_down) = 3) :=
by
  intros time_up time_down speed_up h_time_up h_time_down h_speed_up
  rw [h_time_up, h_time_down, h_speed_up]
  sorry

end natasha_average_speed_l35_35329


namespace probability_two_students_next_to_each_other_l35_35680

theorem probability_two_students_next_to_each_other : (2 * Nat.factorial 9) / Nat.factorial 10 = 1 / 5 :=
by
  sorry

end probability_two_students_next_to_each_other_l35_35680


namespace circle_reflection_l35_35021

/-- The reflection of a point over the line y = -x results in swapping the x and y coordinates 
and changing their signs. Given a circle with center (3, -7), the reflected center should be (7, -3). -/
theorem circle_reflection (x y : ℝ) (h : (x, y) = (3, -7)) : (y, -x) = (7, -3) :=
by
  -- since the problem is stated to skip the proof, we use sorry
  sorry

end circle_reflection_l35_35021


namespace percent_neither_condition_l35_35443

namespace TeachersSurvey

variables (Total HighBloodPressure HeartTrouble Both: ℕ)

theorem percent_neither_condition :
  Total = 150 → HighBloodPressure = 90 → HeartTrouble = 50 → Both = 30 →
  (HighBloodPressure + HeartTrouble - Both) = 110 →
  ((Total - (HighBloodPressure + HeartTrouble - Both)) * 100 / Total) = 2667 / 100 :=
by
  intros hTotal hBP hHT hBoth hUnion
  sorry

end TeachersSurvey

end percent_neither_condition_l35_35443


namespace initial_number_is_11_l35_35995

theorem initial_number_is_11 :
  ∃ (N : ℤ), ∃ (k : ℤ), N - 11 = 17 * k ∧ N = 11 :=
by
  sorry

end initial_number_is_11_l35_35995


namespace insured_fraction_l35_35000

theorem insured_fraction (premium : ℝ) (rate : ℝ) (insured_value : ℝ) (original_value : ℝ)
  (h₁ : premium = 910)
  (h₂ : rate = 0.013)
  (h₃ : insured_value = premium / rate)
  (h₄ : original_value = 87500) :
  insured_value / original_value = 4 / 5 :=
by
  sorry

end insured_fraction_l35_35000


namespace remainder_twice_original_l35_35298

def findRemainder (N : ℕ) (D : ℕ) (r : ℕ) : ℕ :=
  2 * N % D

theorem remainder_twice_original
  (N : ℕ) (D : ℕ)
  (hD : D = 367)
  (hR : N % D = 241) :
  findRemainder N D 2 = 115 := by
  sorry

end remainder_twice_original_l35_35298


namespace no_real_roots_of_equation_l35_35165

theorem no_real_roots_of_equation :
  (∃ x : ℝ, 2 * Real.cos (x / 2) = 10^x + 10^(-x) + 1) -> False :=
by
  sorry

end no_real_roots_of_equation_l35_35165


namespace wheel_distance_3_revolutions_l35_35146

theorem wheel_distance_3_revolutions (r : ℝ) (n : ℝ) (circumference : ℝ) (total_distance : ℝ) :
  r = 2 →
  n = 3 →
  circumference = 2 * Real.pi * r →
  total_distance = n * circumference →
  total_distance = 12 * Real.pi := by
  intros
  sorry

end wheel_distance_3_revolutions_l35_35146


namespace coeff_x2_in_expansion_l35_35814

theorem coeff_x2_in_expansion : 
  (2 : ℚ) - (1 / x) * ((1 + x)^6)^(2 : ℤ) = (10 : ℚ) :=
by sorry

end coeff_x2_in_expansion_l35_35814


namespace sum_of_ages_of_sarahs_friends_l35_35692

noncomputable def sum_of_ages (a b c : ℕ) : ℕ := a + b + c

theorem sum_of_ages_of_sarahs_friends (a b c : ℕ) (h_distinct : ∀ (x y z : ℕ), x ≠ y ∧ y ≠ z ∧ x ≠ z)
  (h_single_digits : ∀ (x y z : ℕ), x < 10 ∧ y < 10 ∧ z < 10)
  (h_product_36 : ∃ (x y : ℕ), x * y = 36 ∧ x ≠ y)
  (h_factor_36 : ∀ (x y z : ℕ), x ∣ 36 ∧ y ∣ 36 ∧ z ∣ 36) :
  ∃ (a b c : ℕ), sum_of_ages a b c = 16 := 
sorry

end sum_of_ages_of_sarahs_friends_l35_35692


namespace find_equation_of_BC_l35_35434

theorem find_equation_of_BC :
  ∃ (BC : ℝ → ℝ → Prop), 
  (∀ x y, (BC x y ↔ 2 * x - y + 5 = 0)) :=
sorry

end find_equation_of_BC_l35_35434


namespace not_periodic_l35_35273

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sin x + Real.sin (a * x)

theorem not_periodic {a : ℝ} (ha : Irrational a) : ¬ ∃ T : ℝ, T ≠ 0 ∧ ∀ x : ℝ, f a (x + T) = f a x :=
  sorry

end not_periodic_l35_35273


namespace length_of_arc_l35_35210

theorem length_of_arc (C : ℝ) (θ : ℝ) (DE : ℝ) (c_circ : C = 100) (angle : θ = 120) :
  DE = 100 / 3 :=
by
  -- Place the actual proof here.
  sorry

end length_of_arc_l35_35210


namespace unicorn_rope_problem_l35_35025

/-
  A unicorn is tethered by a 24-foot golden rope to the base of a sorcerer's cylindrical tower
  whose radius is 10 feet. The rope is attached to the tower at ground level and to the unicorn
  at a height of 6 feet. The unicorn has pulled the rope taut, and the end of the rope is 6 feet
  from the nearest point on the tower.
  The length of the rope that is touching the tower is given as:
  ((96 - sqrt(36)) / 6) feet,
  where 96, 36, and 6 are positive integers, and 6 is prime.
  We need to prove that the sum of these integers is 138.
-/
theorem unicorn_rope_problem : 
  let d := 96
  let e := 36
  let f := 6
  d + e + f = 138 := by
  sorry

end unicorn_rope_problem_l35_35025


namespace smallest_gcd_value_l35_35964

theorem smallest_gcd_value (m n : ℕ) (h1 : m > 0) (h2 : n > 0) (h3 : Nat.gcd m n = 8) : Nat.gcd (8 * m) (12 * n) = 32 :=
by
  sorry

end smallest_gcd_value_l35_35964


namespace john_flights_of_stairs_l35_35882

theorem john_flights_of_stairs (x : ℕ) : 
    let flight_height := 10
    let rope_height := flight_height / 2
    let ladder_height := rope_height + 10
    let total_height := 70
    10 * x + rope_height + ladder_height = total_height → x = 5 :=
by
    intro h
    sorry

end john_flights_of_stairs_l35_35882


namespace ryan_total_commuting_time_l35_35264

def biking_time : ℕ := 30
def bus_time : ℕ := biking_time + 10
def bus_commutes : ℕ := 3
def total_bus_time : ℕ := bus_time * bus_commutes
def friend_time : ℕ := biking_time - (2 * biking_time / 3)
def total_commuting_time : ℕ := biking_time + total_bus_time + friend_time

theorem ryan_total_commuting_time :
  total_commuting_time = 160 :=
by
  sorry

end ryan_total_commuting_time_l35_35264


namespace average_gas_mileage_round_trip_l35_35819

-- Definition of the problem conditions

def distance_to_home : ℕ := 120
def distance_back : ℕ := 120
def mileage_to_home : ℕ := 30
def mileage_back : ℕ := 20

-- Theorem that we need to prove
theorem average_gas_mileage_round_trip
  (d1 d2 : ℕ) (m1 m2 : ℕ)
  (h1 : d1 = distance_to_home)
  (h2 : d2 = distance_back)
  (h3 : m1 = mileage_to_home)
  (h4 : m2 = mileage_back) :
  (d1 + d2) / ((d1 / m1) + (d2 / m2)) = 24 :=
by
  sorry

end average_gas_mileage_round_trip_l35_35819


namespace value_of_k_h_10_l35_35901

def h (x : ℝ) : ℝ := 4 * x - 5
def k (x : ℝ) : ℝ := 2 * x + 6

theorem value_of_k_h_10 : k (h 10) = 76 := by
  -- We provide only the statement as required, skipping the proof
  sorry

end value_of_k_h_10_l35_35901


namespace swans_in_10_years_l35_35232

def doubling_time := 2
def initial_swans := 15
def periods := 10 / doubling_time

theorem swans_in_10_years : 
  (initial_swans * 2 ^ periods) = 480 := 
by
  sorry

end swans_in_10_years_l35_35232


namespace remainder_when_xyz_divided_by_9_is_0_l35_35656

theorem remainder_when_xyz_divided_by_9_is_0
  (x y z : ℕ)
  (hx : x < 9)
  (hy : y < 9)
  (hz : z < 9)
  (h1 : x + 3 * y + 2 * z ≡ 0 [MOD 9])
  (h2 : 3 * x + 2 * y + z ≡ 5 [MOD 9])
  (h3 : 2 * x + y + 3 * z ≡ 5 [MOD 9]) :
  (x * y * z) % 9 = 0 := by
  sorry

end remainder_when_xyz_divided_by_9_is_0_l35_35656


namespace least_number_to_add_l35_35283

theorem least_number_to_add (x : ℕ) (h1 : (1789 + x) % 6 = 0) (h2 : (1789 + x) % 4 = 0) (h3 : (1789 + x) % 3 = 0) : x = 7 := 
sorry

end least_number_to_add_l35_35283


namespace y_coordinate_equidistant_l35_35259

theorem y_coordinate_equidistant : ∃ y : ℝ, (∀ A B : ℝ × ℝ, A = (-3, 0) → B = (-2, 5) → dist (0, y) A = dist (0, y) B) ∧ y = 2 :=
by
  sorry

end y_coordinate_equidistant_l35_35259


namespace expression_equals_5000_l35_35226

theorem expression_equals_5000 :
  12 * 171 + 29 * 9 + 171 * 13 + 29 * 16 = 5000 :=
by
  sorry

end expression_equals_5000_l35_35226


namespace minimum_value_l35_35698

/-- 
Given \(a > 0\), \(b > 0\), and \(a + 2b = 1\),
prove that the minimum value of \(\frac{2}{a} + \frac{1}{b}\) is 8.
-/
theorem minimum_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + 2 * b = 1) : 
  (∀ a b : ℝ, (a > 0) → (b > 0) → (a + 2 * b = 1) → (∃ c : ℝ, c = 8 ∧ ∀ x y : ℝ, (x = a) → (y = b) → (c ≤ (2 / x) + (1 / y)))) :=
sorry

end minimum_value_l35_35698


namespace min_value_of_y_l35_35233

theorem min_value_of_y {y : ℤ} (h : ∃ x : ℤ, y^2 = (0 ^ 2 + 1 ^ 2 + 2 ^ 2 + 3 ^ 2 + 4 ^ 2 + 5 ^ 2 + (-1) ^ 2 + (-2) ^ 2 + (-3) ^ 2 + (-4) ^ 2 + (-5) ^ 2)) :
  y = -11 :=
by sorry

end min_value_of_y_l35_35233


namespace remainder_7_pow_700_div_100_l35_35694

theorem remainder_7_pow_700_div_100 : (7 ^ 700) % 100 = 1 := 
  by sorry

end remainder_7_pow_700_div_100_l35_35694


namespace four_digit_even_numbers_count_and_sum_l35_35843

variable (digits : Set ℕ) (used_once : ∀ d ∈ digits, d ≤ 6 ∧ d ≥ 1)

theorem four_digit_even_numbers_count_and_sum
  (hyp : digits = {1, 2, 3, 4, 5, 6}) :
  ∃ (N M : ℕ), 
    (N = 180 ∧ M = 680040) := 
sorry

end four_digit_even_numbers_count_and_sum_l35_35843


namespace simplify_abs_expression_l35_35404

theorem simplify_abs_expression (a b : ℝ) (h1 : a > 0) (h2 : a * b < 0) :
  |a - 2 * b + 5| + |-3 * a + 2 * b - 2| = 4 * a - 4 * b + 7 := by
  sorry

end simplify_abs_expression_l35_35404


namespace arithmetic_evaluation_l35_35885

theorem arithmetic_evaluation :
  -10 * 3 - (-4 * -2) + (-12 * -4) / 2 = -14 :=
by
  sorry

end arithmetic_evaluation_l35_35885


namespace triangle_area_l35_35759

theorem triangle_area (base height : ℝ) (h_base : base = 8.4) (h_height : height = 5.8) :
  0.5 * base * height = 24.36 := by
  sorry

end triangle_area_l35_35759


namespace exists_nat_a_b_l35_35192

theorem exists_nat_a_b (n : ℕ) (hn : 0 < n) : 
∃ a b : ℕ, 1 ≤ b ∧ b ≤ n ∧ |a - b * Real.sqrt 2| ≤ 1 / n :=
by
  -- The proof steps would be filled here.
  sorry

end exists_nat_a_b_l35_35192


namespace largest_three_digit_multiple_of_4_and_5_l35_35412

theorem largest_three_digit_multiple_of_4_and_5 : 
  ∃ (n : ℕ), n < 1000 ∧ n ≥ 100 ∧ n % 4 = 0 ∧ n % 5 = 0 ∧ n = 980 :=
by
  sorry

end largest_three_digit_multiple_of_4_and_5_l35_35412


namespace sector_area_l35_35906

theorem sector_area (r l : ℝ) (h1 : l + 2 * r = 10) (h2 : l = 2 * r) : 
  (1 / 2) * l * r = 25 / 4 :=
by 
  sorry

end sector_area_l35_35906


namespace find_triplets_l35_35795

theorem find_triplets (a k m : ℕ) (hpos_a : 0 < a) (hpos_k : 0 < k) (hpos_m : 0 < m) (h_eq : k + a^k = m + 2 * a^m) :
  ∃ t : ℕ, 0 < t ∧ (a = 1 ∧ k = t + 1 ∧ m = t) :=
by
  sorry

end find_triplets_l35_35795


namespace bees_multiple_l35_35343

theorem bees_multiple (bees_day1 bees_day2 : ℕ) (h1 : bees_day1 = 144) (h2 : bees_day2 = 432) :
  bees_day2 / bees_day1 = 3 :=
by
  sorry

end bees_multiple_l35_35343


namespace B_share_is_correct_l35_35088

open Real

noncomputable def total_money : ℝ := 10800
noncomputable def ratio_A : ℝ := 0.5
noncomputable def ratio_B : ℝ := 1.5
noncomputable def ratio_C : ℝ := 2.25
noncomputable def ratio_D : ℝ := 3.5
noncomputable def ratio_E : ℝ := 4.25
noncomputable def total_ratio : ℝ := ratio_A + ratio_B + ratio_C + ratio_D + ratio_E
noncomputable def value_per_part : ℝ := total_money / total_ratio
noncomputable def B_share : ℝ := ratio_B * value_per_part

theorem B_share_is_correct : B_share = 1350 := by 
  sorry

end B_share_is_correct_l35_35088


namespace book_page_count_l35_35252

theorem book_page_count:
  (∃ (total_pages : ℕ), 
    (∃ (days_read : ℕ) (pages_per_day : ℕ), 
      days_read = 12 ∧ 
      pages_per_day = 8 ∧ 
      (days_read * pages_per_day) = 2 * (total_pages / 3)) 
  ↔ total_pages = 144) :=
by 
  sorry

end book_page_count_l35_35252


namespace top_and_bottom_area_each_l35_35804

def long_side_area : ℕ := 2 * 8 * 6
def short_side_area : ℕ := 2 * 5 * 6
def total_sides_area : ℕ := long_side_area + short_side_area
def total_needed_area : ℕ := 236
def top_and_bottom_area : ℕ := total_needed_area - total_sides_area

theorem top_and_bottom_area_each :
  top_and_bottom_area / 2 = 40 := by
  sorry

end top_and_bottom_area_each_l35_35804


namespace digits_problem_solution_l35_35437

def digits_proof_problem (E F G H : ℕ) : Prop :=
  (E, F, G) = (5, 0, 5) → H = 0

theorem digits_problem_solution 
  (E F G H : ℕ)
  (h1 : F + E = E ∨ F + E = E + 10)
  (h2 : E ≠ 0)
  (h3 : E = 5)
  (h4 : 5 + G = H)
  (h5 : 5 - G = 0) :
  H = 0 := 
by {
  sorry -- proof goes here
}

end digits_problem_solution_l35_35437


namespace greatest_value_of_b_l35_35980

noncomputable def solution : ℝ :=
  (3 + Real.sqrt 21) / 2

theorem greatest_value_of_b :
  ∀ b : ℝ, b^2 - 4 * b + 3 < -b + 6 → b ≤ solution :=
by
  intro b
  intro h
  sorry

end greatest_value_of_b_l35_35980


namespace water_left_after_four_hours_l35_35177

def initial_water : ℕ := 40
def water_loss_per_hour : ℕ := 2
def water_added_hour3 : ℕ := 1
def water_added_hour4 : ℕ := 3

theorem water_left_after_four_hours :
    initial_water - 2 * 2 - 2 + water_added_hour3 - 2 + water_added_hour4 - 2 = 36 := 
by
    sorry

end water_left_after_four_hours_l35_35177


namespace member_sum_or_double_exists_l35_35789

theorem member_sum_or_double_exists (n : ℕ) (k : ℕ) (P: ℕ → ℕ) (m: ℕ) 
  (h_mem : n = 1978)
  (h_countries : m = 6) : 
  ∃ k, (∃ i j, P i + P j = k ∧ P i = P j)
    ∨ (∃ i, 2 * P i = k) :=
sorry

end member_sum_or_double_exists_l35_35789


namespace males_listen_l35_35415

theorem males_listen (males_dont_listen females_listen total_listen total_dont_listen : ℕ) 
  (h1 : males_dont_listen = 70)
  (h2 : females_listen = 75)
  (h3 : total_listen = 180)
  (h4 : total_dont_listen = 120) :
  ∃ m, m = 105 :=
by {
  sorry
}

end males_listen_l35_35415


namespace edward_lives_left_l35_35599

theorem edward_lives_left : 
  let initial_lives := 50
  let stage1_loss := 18
  let stage1_gain := 7
  let stage2_loss := 10
  let stage2_gain := 5
  let stage3_loss := 13
  let stage3_gain := 2
  let final_lives := initial_lives - stage1_loss + stage1_gain - stage2_loss + stage2_gain - stage3_loss + stage3_gain
  final_lives = 23 :=
by
  sorry

end edward_lives_left_l35_35599


namespace quadratic_no_real_roots_probability_l35_35942

theorem quadratic_no_real_roots_probability :
  (1 : ℝ) - 1 / 4 - 0 = 3 / 4 :=
by
  sorry

end quadratic_no_real_roots_probability_l35_35942


namespace chord_length_is_correct_l35_35799

noncomputable def length_of_chord {ρ θ : Real} 
 (h_line : ρ * Real.sin (π / 6 - θ) = 2) 
 (h_curve : ρ = 4 * Real.cos θ) : Real :=
  2 * Real.sqrt 3

theorem chord_length_is_correct {ρ θ : Real} 
 (h_line : ρ * Real.sin (π / 6 - θ) = 2) 
 (h_curve : ρ = 4 * Real.cos θ) : 
 length_of_chord h_line h_curve = 2 * Real.sqrt 3 :=
sorry

end chord_length_is_correct_l35_35799


namespace land_remaining_is_correct_l35_35182

def lizzie_covered : ℕ := 250
def other_covered : ℕ := 265
def total_land : ℕ := 900
def land_remaining : ℕ := total_land - (lizzie_covered + other_covered)

theorem land_remaining_is_correct : land_remaining = 385 := 
by
  sorry

end land_remaining_is_correct_l35_35182


namespace vector_subtraction_l35_35505

def vector_a : ℝ × ℝ := (3, 5)
def vector_b : ℝ × ℝ := (-2, 1)

theorem vector_subtraction :
  vector_a - 2 • vector_b = (7, 3) :=
sorry

end vector_subtraction_l35_35505


namespace binom_coeff_divisibility_l35_35867

theorem binom_coeff_divisibility (p : ℕ) (hp : Prime p) : Nat.choose (2 * p) p - 2 ≡ 0 [MOD p^2] := 
sorry

end binom_coeff_divisibility_l35_35867


namespace smallest_positive_integer_remainder_l35_35068

theorem smallest_positive_integer_remainder
  (b : ℕ) (h1 : b % 4 = 3) (h2 : b % 6 = 5) :
  b = 11 := by
  sorry

end smallest_positive_integer_remainder_l35_35068


namespace eggs_division_l35_35479

theorem eggs_division (n_students n_eggs : ℕ) (h_students : n_students = 9) (h_eggs : n_eggs = 73):
  n_eggs / n_students = 8 ∧ n_eggs % n_students = 1 :=
by
  rw [h_students, h_eggs]
  exact ⟨rfl, rfl⟩

end eggs_division_l35_35479


namespace not_perfect_cube_of_cond_l35_35862

open Int

theorem not_perfect_cube_of_cond (n : ℤ) (h₁ : 0 < n) (k : ℤ) 
  (h₂ : n^5 + n^3 + 2 * n^2 + 2 * n + 2 = k ^ 3) : 
  ¬ ∃ m : ℤ, 2 * n^2 + n + 2 = m ^ 3 :=
sorry

end not_perfect_cube_of_cond_l35_35862


namespace popsicle_sum_l35_35509

-- Gino has 63 popsicle sticks
def gino_popsicle_sticks : Nat := 63

-- I have 50 popsicle sticks
def my_popsicle_sticks : Nat := 50

-- The sum of our popsicle sticks
def total_popsicle_sticks : Nat := gino_popsicle_sticks + my_popsicle_sticks

-- Prove that the total is 113
theorem popsicle_sum : total_popsicle_sticks = 113 :=
by
  -- Proof goes here
  sorry

end popsicle_sum_l35_35509


namespace find_r_in_arithmetic_sequence_l35_35523

-- Define an arithmetic sequence
def is_arithmetic_sequence (a b c d e f : ℤ) : Prop :=
  (b - a = c - b) ∧ (c - b = d - c) ∧ (d - c = e - d) ∧ (e - d = f - e)

-- Define the given problem
theorem find_r_in_arithmetic_sequence :
  ∃ r : ℤ, ∀ p q s : ℤ, is_arithmetic_sequence 23 p q r s 59 → r = 41 :=
by
  sorry

end find_r_in_arithmetic_sequence_l35_35523


namespace intersection_domain_range_l35_35266

-- Define domain and function
def domain : Set ℝ := {-1, 0, 1}
def f (x : ℝ) : ℝ := |x|

-- Prove the theorem
theorem intersection_domain_range :
  let range : Set ℝ := {y | ∃ x ∈ domain, f x = y}
  let A : Set ℝ := domain
  let B : Set ℝ := range 
  A ∩ B = {0, 1} :=
by
  -- The proof is skipped with sorry
  sorry

end intersection_domain_range_l35_35266


namespace rectangle_perimeter_l35_35098

-- Definitions of the conditions
def side_length_square : ℕ := 75  -- side length of the square in mm
def height_sum (x y z : ℕ) : Prop := x + y + z = side_length_square  -- sum of heights of the rectangles

-- Perimeter definition
def perimeter (h : ℕ) (w : ℕ) : ℕ := 2 * (h + w)

-- Statement of the problem
theorem rectangle_perimeter (x y z : ℕ) (h_sum : height_sum x y z)
  (h1 : perimeter x side_length_square = (perimeter y side_length_square + perimeter z side_length_square) / 2)
  : perimeter x side_length_square = 200 := by
  sorry

end rectangle_perimeter_l35_35098


namespace f_is_odd_l35_35733

open Real

def f (x : ℝ) : ℝ := x^3 + x

theorem f_is_odd : ∀ x : ℝ, f (-x) = -f x :=
by
  intro x
  sorry

end f_is_odd_l35_35733


namespace solve_quadratic_eq1_solve_quadratic_eq2_l35_35699

-- Define the first equation
theorem solve_quadratic_eq1 (x : ℝ) : x^2 - 6 * x - 6 = 0 ↔ x = 3 + Real.sqrt 15 ∨ x = 3 - Real.sqrt 15 := by
  sorry

-- Define the second equation
theorem solve_quadratic_eq2 (x : ℝ) : 2 * x^2 - 3 * x + 1 = 0 ↔ x = 1 ∨ x = 1 / 2 := by
  sorry

end solve_quadratic_eq1_solve_quadratic_eq2_l35_35699


namespace area_of_pentagon_correct_l35_35262

noncomputable def area_of_pentagon : ℝ :=
  let AB := 5
  let BC := 3
  let BD := 3
  let AC := Real.sqrt (AB^2 - BC^2)
  let AD := Real.sqrt (AB^2 - BD^2)
  let EC := 1
  let FD := 2
  let AE := AC - EC
  let AF := AD - FD
  let sin_alpha := BC / AB
  let cos_alpha := AC / AB
  let sin_2alpha := 2 * sin_alpha * cos_alpha
  let area_ABC := 0.5 * AB * BC
  let area_AEF := 0.5 * AE * AF * sin_2alpha
  2 * area_ABC - area_AEF

theorem area_of_pentagon_correct :
  area_of_pentagon = 9.12 := sorry

end area_of_pentagon_correct_l35_35262


namespace sum_smallest_largest_eq_2y_l35_35318

theorem sum_smallest_largest_eq_2y (n : ℕ) (y a : ℕ) 
  (h1 : 2 * a + 2 * (n - 1) / n = y) : 
  2 * y = (2 * a + 2 * (n - 1)) := 
sorry

end sum_smallest_largest_eq_2y_l35_35318


namespace find_interest_rate_l35_35314

-- conditions
def P : ℝ := 6200
def t : ℕ := 10

def interest (P : ℝ) (r : ℝ) (t : ℕ) : ℝ := P * r * t
def I : ℝ := P - 3100

-- problem statement
theorem find_interest_rate (r : ℝ) :
  interest P r t = I → r = 0.05 :=
by
  sorry

end find_interest_rate_l35_35314


namespace combined_sale_price_correct_l35_35718

-- Define constants for purchase costs of items A, B, and C.
def purchase_cost_A : ℝ := 650
def purchase_cost_B : ℝ := 350
def purchase_cost_C : ℝ := 400

-- Define profit percentages for items A, B, and C.
def profit_percentage_A : ℝ := 0.40
def profit_percentage_B : ℝ := 0.25
def profit_percentage_C : ℝ := 0.30

-- Define the desired sale prices for items A, B, and C based on profit margins.
def sale_price_A : ℝ := purchase_cost_A * (1 + profit_percentage_A)
def sale_price_B : ℝ := purchase_cost_B * (1 + profit_percentage_B)
def sale_price_C : ℝ := purchase_cost_C * (1 + profit_percentage_C)

-- Calculate the combined sale price for all three items.
def combined_sale_price : ℝ := sale_price_A + sale_price_B + sale_price_C

-- The theorem stating that the combined sale price for all three items is $1867.50.
theorem combined_sale_price_correct :
  combined_sale_price = 1867.50 := 
sorry

end combined_sale_price_correct_l35_35718


namespace fraction_zero_solution_l35_35873

theorem fraction_zero_solution (x : ℝ) (h₁ : x - 1 = 0) (h₂ : x + 3 ≠ 0) : x = 1 :=
by
  sorry

end fraction_zero_solution_l35_35873


namespace trips_per_student_l35_35370

theorem trips_per_student
  (num_students : ℕ := 5)
  (chairs_per_trip : ℕ := 5)
  (total_chairs : ℕ := 250)
  (T : ℕ) :
  num_students * chairs_per_trip * T = total_chairs → T = 10 :=
by
  intro h
  sorry

end trips_per_student_l35_35370


namespace roots_difference_squared_quadratic_roots_property_l35_35857

noncomputable def α : ℝ := (3 + Real.sqrt 5) / 2
noncomputable def β : ℝ := (3 - Real.sqrt 5) / 2

theorem roots_difference_squared :
  α - β = Real.sqrt 5 :=
by
  sorry

theorem quadratic_roots_property :
  (α - β) ^ 2 = 5 :=
by
  sorry

end roots_difference_squared_quadratic_roots_property_l35_35857


namespace total_apples_for_bobbing_l35_35007

theorem total_apples_for_bobbing (apples_per_bucket : ℕ) (buckets : ℕ) (total_apples : ℕ) : 
  apples_per_bucket = 9 → buckets = 7 → total_apples = apples_per_bucket * buckets → total_apples = 63 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end total_apples_for_bobbing_l35_35007


namespace preferred_apples_percentage_l35_35563

theorem preferred_apples_percentage (A B C O G : ℕ) (total freq_apples : ℕ)
  (hA : A = 70) (hB : B = 50) (hC: C = 30) (hO: O = 50) (hG: G = 40)
  (htotal : total = A + B + C + O + G)
  (hfa : freq_apples = A) :
  (freq_apples / total : ℚ) * 100 = 29 :=
by sorry

end preferred_apples_percentage_l35_35563


namespace intersection_value_unique_l35_35991

theorem intersection_value_unique (x : ℝ) :
  (∃ y : ℝ, y = 8 / (x^2 + 4) ∧ x + y = 2) → x = 0 :=
by
  sorry

end intersection_value_unique_l35_35991


namespace population_weight_of_500_students_l35_35337

-- Definitions
def number_of_students : ℕ := 500
def number_of_selected_students : ℕ := 60

-- Conditions
def condition1 := number_of_students = 500
def condition2 := number_of_selected_students = 60

-- Statement
theorem population_weight_of_500_students : 
  condition1 → condition2 → 
  (∃ p, p = "the weight of the 500 students") := by
  intros _ _
  existsi "the weight of the 500 students"
  rfl

end population_weight_of_500_students_l35_35337


namespace mod_pow_sub_eq_l35_35917

theorem mod_pow_sub_eq : 
  (45^1537 - 25^1537) % 8 = 4 := 
by
  have h1 : 45 % 8 = 5 := by norm_num
  have h2 : 25 % 8 = 1 := by norm_num
  sorry

end mod_pow_sub_eq_l35_35917


namespace remainder_7n_mod_4_l35_35011

theorem remainder_7n_mod_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 := by
  sorry

end remainder_7n_mod_4_l35_35011


namespace convert_deg_to_rad_l35_35358

theorem convert_deg_to_rad (deg : ℝ) (h : deg = -630) : deg * (Real.pi / 180) = -7 * Real.pi / 2 :=
by
  rw [h]
  simp
  sorry

end convert_deg_to_rad_l35_35358


namespace fraction_inequality_l35_35596

-- Given the conditions
variables {c x y : ℝ} (h1 : c > x) (h2 : x > y) (h3 : y > 0)

-- Prove that \frac{x}{c-x} > \frac{y}{c-y}
theorem fraction_inequality (h4 : c > 0) : (x / (c - x)) > (y / (c - y)) :=
by {
  sorry  -- Proof to be completed
}

end fraction_inequality_l35_35596


namespace problem_statement_l35_35688

def h (x : ℝ) : ℝ := 3 * x + 2
def k (x : ℝ) : ℝ := 2 * x - 3

theorem problem_statement : (h (k (h 3))) / (k (h (k 3))) = 59 / 19 := by
  sorry

end problem_statement_l35_35688


namespace subtraction_verification_l35_35195

theorem subtraction_verification : 888888888888 - 111111111111 = 777777777777 :=
by
  sorry

end subtraction_verification_l35_35195


namespace inequality_problem_l35_35349

theorem inequality_problem (x : ℝ) (h_denom : 2 * x^2 + 2 * x + 1 ≠ 0) : 
  -4 ≤ (x^2 - 2*x - 3)/(2*x^2 + 2*x + 1) ∧ (x^2 - 2*x - 3)/(2*x^2 + 2*x + 1) ≤ 1 :=
sorry

end inequality_problem_l35_35349


namespace force_for_18_inch_wrench_l35_35205

theorem force_for_18_inch_wrench (F : ℕ → ℕ → ℕ) : 
  (∀ L : ℕ, ∃ k : ℕ, F 300 12 = F (F L k) L) → 
  ((F 12 300) = 3600) → 
  (∀ k : ℕ, F (F 6 k) 6 = 3600) → 
  (∀ k : ℕ, F (F 18 k) 18 = 3600) → 
  (F 18 200 = 3600) :=
by
  sorry

end force_for_18_inch_wrench_l35_35205


namespace animal_population_l35_35089

def total_population (L P E : ℕ) : ℕ :=
L + P + E

theorem animal_population 
    (L P E : ℕ) 
    (h1 : L = 2 * P) 
    (h2 : E = (L + P) / 2) 
    (h3 : L = 200) : 
  total_population L P E = 450 := 
  by 
    sorry

end animal_population_l35_35089


namespace unique_intersection_point_l35_35403

theorem unique_intersection_point {c : ℝ} :
  (∀ x y : ℝ, y = |x - 20| + |x + 18| → y = x + c → (x = 20 ∧ y = 38)) ↔ c = 18 :=
by
  sorry

end unique_intersection_point_l35_35403


namespace bricks_in_wall_l35_35646

theorem bricks_in_wall (h : ℕ) 
  (brenda_rate : ℕ := h / 8)
  (brandon_rate : ℕ := h / 12)
  (combined_rate : ℕ := (5 * h) / 24)
  (decreased_combined_rate : ℕ := combined_rate - 15)
  (work_time : ℕ := 6) :
  work_time * decreased_combined_rate = h → h = 360 := by
  intros h_eq
  sorry

end bricks_in_wall_l35_35646


namespace find_f2_l35_35765

variable (f g : ℝ → ℝ) (a : ℝ)

-- Definitions based on conditions
def is_odd (f : ℝ → ℝ) := ∀ x, f (-x) = -f x
def is_even (g : ℝ → ℝ) := ∀ x, g (-x) = g x
def equation (f g : ℝ → ℝ) (a : ℝ) := ∀ x, f x + g x = a^x - a^(-x) + 2

-- Lean statement for the proof problem
theorem find_f2
  (h1 : is_odd f)
  (h2 : is_even g)
  (h3 : equation f g a)
  (h4 : g 2 = a) : f 2 = 15 / 4 :=
by
  sorry

end find_f2_l35_35765


namespace molecular_weight_correct_l35_35181

def atomic_weight_N : ℝ := 14.01
def atomic_weight_O : ℝ := 16.00
def num_N_in_N2O3 : ℕ := 2
def num_O_in_N2O3 : ℕ := 3

def molecular_weight_N2O3 : ℝ :=
  (num_N_in_N2O3 * atomic_weight_N) + (num_O_in_N2O3 * atomic_weight_O)

theorem molecular_weight_correct :
  molecular_weight_N2O3 = 76.02 := by
  sorry

end molecular_weight_correct_l35_35181


namespace combinations_15_3_l35_35400

def num_combinations (n k : ℕ) : ℕ := n.choose k

theorem combinations_15_3 :
  num_combinations 15 3 = 455 :=
sorry

end combinations_15_3_l35_35400


namespace park_trees_after_planting_l35_35977

theorem park_trees_after_planting (current_trees trees_today trees_tomorrow : ℕ)
  (h1 : current_trees = 7)
  (h2 : trees_today = 5)
  (h3 : trees_tomorrow = 4) :
  current_trees + trees_today + trees_tomorrow = 16 :=
by
  sorry

end park_trees_after_planting_l35_35977


namespace odd_function_b_value_f_monotonically_increasing_l35_35069

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := Real.log (Real.sqrt (4 * x^2 + b) + 2 * x)

-- part (1): Prove that if y = f(x) is an odd function, then b = 1
theorem odd_function_b_value :
  (∀ x : ℝ, f x b + f (-x) b = 0) → b = 1 := sorry

-- part (2): Prove that y = f(x) is monotonically increasing for all x in ℝ given b = 1
theorem f_monotonically_increasing (b : ℝ) :
  b = 1 → ∀ x1 x2 : ℝ, x1 < x2 → f x1 b < f x2 b := sorry

end odd_function_b_value_f_monotonically_increasing_l35_35069


namespace sufficient_not_necessary_p_q_l35_35652

theorem sufficient_not_necessary_p_q {m : ℝ} 
  (hp : ∀ x, (x^2 - 8*x - 20 ≤ 0) → (-2 ≤ x ∧ x ≤ 10))
  (hq : ∀ x, ((x - 1 - m) * (x - 1 + m) ≤ 0) → (1 - m ≤ x ∧ x ≤ 1 + m))
  (m_pos : 0 < m)  :
  (∀ x, (x - 1 - m) * (x - 1 + m) ≤ 0 → x^2 - 8*x - 20 ≤ 0) ∧ ¬ (∀ x, x^2 - 8*x - 20 ≤ 0 → (x - 1 - m) * (x - 1 + m) ≤ 0) →
  m ≤ 3 :=
sorry

end sufficient_not_necessary_p_q_l35_35652


namespace find_x0_l35_35601

-- Define a function f with domain [0, 3] and its inverse
variable {f : ℝ → ℝ}

-- Assume conditions for the inverse function
axiom f_inv_1 : ∀ x, 0 ≤ x ∧ x < 1 → 1 ≤ f x ∧ f x < 2
axiom f_inv_2 : ∀ x, 2 < x ∧ x ≤ 4 → 0 ≤ f x ∧ f x < 1

-- Domain condition
variables (x : ℝ) (hf_domain : 0 ≤ x ∧ x ≤ 3)

-- The main theorem
theorem find_x0 : (∃ x0: ℝ, f x0 = x0) → x = 2 :=
  sorry

end find_x0_l35_35601


namespace sin_tan_condition_l35_35394

theorem sin_tan_condition (x : ℝ) (h : Real.sin x = (Real.sqrt 2) / 2) : ¬((∀ x, Real.sin x = (Real.sqrt 2) / 2 → Real.tan x = 1) ∧ (∀ x, Real.tan x = 1 → Real.sin x = (Real.sqrt 2) / 2)) :=
sorry

end sin_tan_condition_l35_35394


namespace basic_astrophysics_budget_percent_l35_35735

theorem basic_astrophysics_budget_percent
  (total_degrees : ℝ := 360)
  (astrophysics_degrees : ℝ := 108) :
  (astrophysics_degrees / total_degrees) * 100 = 30 := by
  sorry

end basic_astrophysics_budget_percent_l35_35735


namespace odd_and_symmetric_f_l35_35017

open Real

noncomputable def f (A ϕ : ℝ) (x : ℝ) := A * sin (x + ϕ)

theorem odd_and_symmetric_f (A ϕ : ℝ) (hA : A > 0) (hmin : f A ϕ (π / 4) = -1) : 
  ∃ g : ℝ → ℝ, g x = -A * sin x ∧ (∀ x, g (-x) = -g x) ∧ (∀ x, g (π / 2 - x) = g (π / 2 + x)) :=
sorry

end odd_and_symmetric_f_l35_35017


namespace quadratic_function_increases_l35_35239

-- Define the quadratic function
def quadratic_function (x : ℝ) : ℝ := 2 * x ^ 2 - 4 * x + 5

-- Prove that for x > 1, the function value y increases as x increases
theorem quadratic_function_increases (x : ℝ) (h : x > 1) : 
  quadratic_function x > quadratic_function 1 :=
sorry

end quadratic_function_increases_l35_35239


namespace possible_k_values_l35_35037

theorem possible_k_values :
  (∃ k b a c : ℤ, b = 2020 + k ∧ a * (c ^ 2) = (2020 + k) ∧ 
  (k = -404 ∨ k = -1010)) :=
sorry

end possible_k_values_l35_35037


namespace floor_negative_fraction_l35_35369

theorem floor_negative_fraction : (Int.floor (-7 / 4 : ℚ)) = -2 := by
  sorry

end floor_negative_fraction_l35_35369


namespace minimum_employees_needed_l35_35409

theorem minimum_employees_needed
  (n_W : ℕ) (n_A : ℕ) (n_S : ℕ)
  (n_WA : ℕ) (n_AS : ℕ) (n_SW : ℕ)
  (n_WAS : ℕ)
  (h_W : n_W = 115)
  (h_A : n_A = 92)
  (h_S : n_S = 60)
  (h_WA : n_WA = 32)
  (h_AS : n_AS = 20)
  (h_SW : n_SW = 10)
  (h_WAS : n_WAS = 5) :
  n_W + n_A + n_S - (n_WA - n_WAS) - (n_AS - n_WAS) - (n_SW - n_WAS) + 2 * n_WAS = 225 :=
by
  sorry

end minimum_employees_needed_l35_35409


namespace danivan_drugstore_end_of_week_inventory_l35_35776

-- Define the initial conditions in Lean
def initial_inventory := 4500
def sold_monday := 2445
def sold_tuesday := 900
def sold_wednesday_to_sunday := 50 * 5
def supplier_delivery := 650

-- Define the statement of the proof problem
theorem danivan_drugstore_end_of_week_inventory :
  initial_inventory - (sold_monday + sold_tuesday + sold_wednesday_to_sunday) + supplier_delivery = 1555 :=
by
  sorry

end danivan_drugstore_end_of_week_inventory_l35_35776


namespace length_of_bridge_l35_35836

noncomputable def train_length : ℝ := 155
noncomputable def train_speed_km_hr : ℝ := 45
noncomputable def crossing_time_seconds : ℝ := 30

noncomputable def train_speed_m_s : ℝ := train_speed_km_hr * 1000 / 3600

noncomputable def total_distance : ℝ := train_speed_m_s * crossing_time_seconds

noncomputable def bridge_length : ℝ := total_distance - train_length

theorem length_of_bridge : bridge_length = 220 := by
  sorry

end length_of_bridge_l35_35836


namespace concentration_third_flask_l35_35714

-- Definitions based on the conditions in the problem
def first_flask_acid := 10
def second_flask_acid := 20
def third_flask_acid := 30
def concentration_first_flask := 0.05
def concentration_second_flask := 70 / 300

-- Problem statement in Lean
theorem concentration_third_flask (W1 W2 : ℝ) (h1 : 10 / (10 + W1) = 0.05)
 (h2 : 20 / (20 + W2) = 70 / 300):
  (30 / (30 + (W1 + W2))) * 100 = 10.5 := 
sorry

end concentration_third_flask_l35_35714


namespace chairs_left_after_selling_l35_35438

-- Definitions based on conditions
def chairs_before_selling : ℕ := 15
def difference_after_selling : ℕ := 12

-- Theorem statement based on the question
theorem chairs_left_after_selling : (chairs_before_selling - 3 = difference_after_selling) → (chairs_before_selling - difference_after_selling = 3) := by
  intro h
  sorry

end chairs_left_after_selling_l35_35438


namespace sequence_general_term_l35_35992

theorem sequence_general_term 
  (x : ℕ → ℝ)
  (h1 : x 1 = 2)
  (h2 : x 2 = 3)
  (h3 : ∀ m ≥ 1, x (2*m+1) = x (2*m) + x (2*m-1))
  (h4 : ∀ m ≥ 2, x (2*m) = x (2*m-1) + 2*x (2*m-2)) :
  ∀ m, (x (2*m-1) = ((3 - Real.sqrt 2) / 4) * (2 + Real.sqrt 2) ^ m + ((3 + Real.sqrt 2) / 4) * (2 - Real.sqrt 2) ^ m ∧ 
          x (2*m) = ((1 + 2 * Real.sqrt 2) / 4) * (2 + Real.sqrt 2) ^ m + ((1 - 2 * Real.sqrt 2) / 4) * (2 - Real.sqrt 2) ^ m) :=
sorry

end sequence_general_term_l35_35992


namespace tangent_line_ln_x_xsq_l35_35683

theorem tangent_line_ln_x_xsq (x y : ℝ) (h_curve : y = Real.log x + x^2) (h_point : (x, y) = (1, 1)) :
  3 * x - y - 2 = 0 :=
sorry

end tangent_line_ln_x_xsq_l35_35683


namespace inverse_proposition_l35_35737

-- Definition of the proposition
def complementary_angles_on_same_side (l m : Line) : Prop := sorry
def parallel_lines (l m : Line) : Prop := sorry

-- The original proposition
def original_proposition (l m : Line) : Prop := complementary_angles_on_same_side l m → parallel_lines l m

-- The statement of the proof problem
theorem inverse_proposition (l m : Line) :
  (complementary_angles_on_same_side l m → parallel_lines l m) →
  (parallel_lines l m → complementary_angles_on_same_side l m) := sorry

end inverse_proposition_l35_35737


namespace coordinates_C_l35_35562

theorem coordinates_C 
  (A : ℝ × ℝ) (B : ℝ × ℝ) (C : ℝ × ℝ) 
  (hA : A = (-1, 3)) 
  (hB : B = (11, 7))
  (hBC_AB : (C.1 - B.1, C.2 - B.2) = (2 / 3) • (B.1 - A.1, B.2 - A.2)) :
  C = (19, 29 / 3) :=
sorry

end coordinates_C_l35_35562


namespace mark_more_hours_than_kate_l35_35223

-- Definitions for the problem
variable (K : ℕ)  -- K is the number of hours charged by Kate
variable (P : ℕ)  -- P is the number of hours charged by Pat
variable (M : ℕ)  -- M is the number of hours charged by Mark

-- Conditions
def total_hours := K + P + M = 216
def pat_kate_relation := P = 2 * K
def pat_mark_relation := P = (1 / 3) * M

-- The statement to be proved
theorem mark_more_hours_than_kate (K P M : ℕ) (h1 : total_hours K P M)
  (h2 : pat_kate_relation K P) (h3 : pat_mark_relation P M) :
  (M - K = 120) :=
by
  sorry

end mark_more_hours_than_kate_l35_35223


namespace sum_of_squares_eq_2_l35_35594

theorem sum_of_squares_eq_2 (a b : ℝ) 
  (h : (a^2 + b^2) * (a^2 + b^2 + 4) = 12) : a^2 + b^2 = 2 :=
by sorry

end sum_of_squares_eq_2_l35_35594


namespace money_last_weeks_l35_35420

-- Define the amounts of money earned and spent per week
def money_mowing : ℕ := 5
def money_weed_eating : ℕ := 58
def weekly_spending : ℕ := 7

-- Define the total money earned
def total_money : ℕ := money_mowing + money_weed_eating

-- Define the number of weeks the money will last
def weeks_last (total : ℕ) (weekly : ℕ) : ℕ := total / weekly

-- Theorem stating the number of weeks the money will last
theorem money_last_weeks : weeks_last total_money weekly_spending = 9 := by
  sorry

end money_last_weeks_l35_35420


namespace reassemble_square_with_hole_l35_35321

theorem reassemble_square_with_hole 
  (a b c d k1 k2 : ℝ)
  (h1 : a = b)
  (h2 : c = d)
  (h3 : k1 = k2) :
  ∃ (f gh ef gh' : ℝ), 
    f = a - c ∧
    gh = b - d ∧
    ef = f ∧
    gh' = gh := 
by sorry

end reassemble_square_with_hole_l35_35321


namespace perimeter_triangle_APR_l35_35138

-- Define given lengths
def AB := 24
def AC := AB
def AP := 8
def AR := AP

-- Define lengths calculated from conditions 
def PB := AB - AP
def RC := AC - AR

-- Define properties from the tangent intersection at Q
def PQ := PB
def QR := RC
def PR := PQ + QR

-- Calculate the perimeter
def perimeter_APR := AP + PR + AR

-- Proof of the problem statement
theorem perimeter_triangle_APR : perimeter_APR = 48 :=
by
  -- Calculations already given in the statement
  sorry

end perimeter_triangle_APR_l35_35138


namespace median_possible_values_l35_35173

theorem median_possible_values (S : Finset ℤ)
  (h : S.card = 10)
  (h_contains : {5, 7, 12, 15, 18, 21} ⊆ S) :
  ∃! n : ℕ, n = 5 :=
by
   sorry

end median_possible_values_l35_35173


namespace articles_bought_l35_35930

theorem articles_bought (C : ℝ) (N : ℝ) (h1 : (N * C) = (30 * ((5 / 3) * C))) : N = 50 :=
by
  sorry

end articles_bought_l35_35930


namespace angle_in_quadrants_l35_35029

theorem angle_in_quadrants (α : ℝ) (hα : 0 < α ∧ α < π / 2) (k : ℤ) :
  (∃ i : ℤ, k = 2 * i + 1 ∧ π < (2 * i + 1) * π + α ∧ (2 * i + 1) * π + α < 3 * π / 2) ∨
  (∃ i : ℤ, k = 2 * i ∧ 0 < 2 * i * π + α ∧ 2 * i * π + α < π / 2) :=
sorry

end angle_in_quadrants_l35_35029


namespace max_value_of_m_l35_35202

theorem max_value_of_m {a b c : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h_sum : a + b + c = 12) (h_prod : a * b + b * c + c * a = 20) :
  ∃ m, m = min (a * b) (min (b * c) (c * a)) ∧ m = 12 :=
by
  sorry

end max_value_of_m_l35_35202


namespace gallons_of_soup_l35_35911

def bowls_per_minute : ℕ := 5
def ounces_per_bowl : ℕ := 10
def serving_time_minutes : ℕ := 15
def ounces_per_gallon : ℕ := 128

theorem gallons_of_soup :
  (5 * 10 * 15 / 128) = 6 :=
by
  sorry

end gallons_of_soup_l35_35911


namespace rocco_piles_of_quarters_proof_l35_35521

-- Define the value of a pile of different types of coins
def pile_value (coin_value : ℕ) (num_coins_in_pile : ℕ) : ℕ :=
  coin_value * num_coins_in_pile

-- Define the number of piles for different coins
def num_piles_of_dimes : ℕ := 6
def num_piles_of_nickels : ℕ := 9
def num_piles_of_pennies : ℕ := 5
def num_coins_in_pile : ℕ := 10

-- Define the total value of each type of coin
def value_of_a_dime : ℕ := 10  -- in cents
def value_of_a_nickel : ℕ := 5  -- in cents
def value_of_a_penny : ℕ := 1  -- in cents
def value_of_a_quarter : ℕ := 25  -- in cents

-- Define the total money Rocco has in cents
def total_money : ℕ := 2100  -- since $21 = 2100 cents

-- Calculate the value of all piles of each type of coin
def total_dimes_value : ℕ := num_piles_of_dimes * (pile_value value_of_a_dime num_coins_in_pile)
def total_nickels_value : ℕ := num_piles_of_nickels * (pile_value value_of_a_nickel num_coins_in_pile)
def total_pennies_value : ℕ := num_piles_of_pennies * (pile_value value_of_a_penny num_coins_in_pile)

-- Calculate the value of the quarters
def value_of_quarters : ℕ := total_money - (total_dimes_value + total_nickels_value + total_pennies_value)
def num_piles_of_quarters : ℕ := value_of_quarters / 250 -- since each pile of quarters is worth 250 cents

-- Theorem to prove
theorem rocco_piles_of_quarters_proof : num_piles_of_quarters = 4 := by
  sorry

end rocco_piles_of_quarters_proof_l35_35521


namespace part_a_part_b_l35_35629

theorem part_a (α : ℝ) (h_irr : Irrational α) (a b : ℝ) (h_lt : a < b) :
  ∃ (m n : ℤ), a < m * α - n ∧ m * α - n < b :=
sorry

theorem part_b (α : ℝ) (h_irr : Irrational α) (a b : ℝ) (h_lt : a < b) :
  ∃ (m n : ℕ), a < m * α - n ∧ m * α - n < b :=
sorry

end part_a_part_b_l35_35629


namespace negation_of_existence_l35_35907

theorem negation_of_existence (h: ∃ x : ℝ, 0 < x ∧ (Real.log x + x - 1 ≤ 0)) :
  ¬ (∀ x : ℝ, 0 < x → ¬ (Real.log x + x - 1 ≤ 0)) :=
sorry

end negation_of_existence_l35_35907


namespace average_calculation_l35_35640

def average (a b c : ℚ) : ℚ := (a + b + c) / 3
def pairAverage (a b : ℚ) : ℚ := (a + b) / 2

theorem average_calculation :
  average (average (pairAverage 2 2) 3 1) (pairAverage 1 2) 1 = 3 / 2 := sorry

end average_calculation_l35_35640


namespace right_triangle_leg_length_l35_35427

theorem right_triangle_leg_length (a b c : ℕ) (h_c : c = 13) (h_a : a = 12) (h_pythagorean : a^2 + b^2 = c^2) :
  b = 5 := 
by {
  -- Provide a placeholder for the proof
  sorry
}

end right_triangle_leg_length_l35_35427


namespace inequality_positive_real_l35_35093

theorem inequality_positive_real (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z = 1) :
  (x^2 + y^2) / z + (y^2 + z^2) / x + (z^2 + x^2) / y ≥ 2 :=
sorry

end inequality_positive_real_l35_35093


namespace new_volume_eq_7352_l35_35429

variable (l w h : ℝ)

-- Given conditions
def volume_eq : Prop := l * w * h = 5184
def surface_area_eq : Prop := l * w + w * h + h * l = 972
def edge_sum_eq : Prop := l + w + h = 54

-- Question: New volume when dimensions are increased by two inches
def new_volume : ℝ := (l + 2) * (w + 2) * (h + 2)

-- Correct Answer: Prove that the new volume equals 7352
theorem new_volume_eq_7352 (h_vol : volume_eq l w h) (h_surf : surface_area_eq l w h) (h_edge : edge_sum_eq l w h) 
    : new_volume l w h = 7352 :=
by
  -- Proof omitted
  sorry

#check new_volume_eq_7352

end new_volume_eq_7352_l35_35429


namespace percentage_proof_l35_35213

/-- Lean 4 statement proving the percentage -/
theorem percentage_proof :
  ∃ P : ℝ, (800 - (P / 100) * 8000) = 796 ∧ P = 0.05 :=
by
  use 0.05
  sorry

end percentage_proof_l35_35213


namespace rational_terms_count_l35_35308

noncomputable def number_of_rational_terms (n : ℕ) (x : ℝ) : ℕ :=
  -- The count of rational terms in the expansion
  17

theorem rational_terms_count (n : ℕ) (x : ℝ) :
  (number_of_rational_terms 100 x) = 17 := by
  sorry

end rational_terms_count_l35_35308


namespace find_xy_l35_35716

theorem find_xy (x y : ℝ) :
  (x - 8) ^ 2 + (y - 9) ^ 2 + (x - y) ^ 2 = 1 / 3 ↔ 
  (x = 25 / 3 ∧ y = 26 / 3) :=
by
  sorry

end find_xy_l35_35716


namespace Dawn_sold_glasses_l35_35555

variable (x : ℕ)

def Bea_price_per_glass : ℝ := 0.25
def Dawn_price_per_glass : ℝ := 0.28
def Bea_glasses_sold : ℕ := 10
def Bea_extra_earnings : ℝ := 0.26
def Bea_total_earnings : ℝ := Bea_glasses_sold * Bea_price_per_glass
def Dawn_total_earnings (x : ℕ) : ℝ := x * Dawn_price_per_glass

theorem Dawn_sold_glasses :
  Bea_total_earnings - Bea_extra_earnings = Dawn_total_earnings x → x = 8 :=
by
  sorry

end Dawn_sold_glasses_l35_35555


namespace find_percentage_of_male_students_l35_35062

def percentage_of_male_students (M F : ℝ) : Prop :=
  M + F = 1 ∧ 0.40 * M + 0.60 * F = 0.52

theorem find_percentage_of_male_students (M F : ℝ) (h1 : M + F = 1) (h2 : 0.40 * M + 0.60 * F = 0.52) : M = 0.40 :=
by
  sorry

end find_percentage_of_male_students_l35_35062


namespace animals_remaining_correct_l35_35083

-- Definitions from the conditions
def initial_cows : ℕ := 184
def initial_dogs : ℕ := initial_cows / 2

def cows_sold : ℕ := initial_cows / 4
def remaining_cows : ℕ := initial_cows - cows_sold

def dogs_sold : ℕ := (3 * initial_dogs) / 4
def remaining_dogs : ℕ := initial_dogs - dogs_sold

def total_remaining_animals : ℕ := remaining_cows + remaining_dogs

-- Theorem to be proved
theorem animals_remaining_correct : total_remaining_animals = 161 := 
by
  sorry

end animals_remaining_correct_l35_35083


namespace fraction_of_total_amount_l35_35090

theorem fraction_of_total_amount (p q r : ℕ) (h1 : p + q + r = 4000) (h2 : r = 1600) :
  r / (p + q + r) = 2 / 5 :=
by
  sorry

end fraction_of_total_amount_l35_35090


namespace find_r_for_f_of_3_eq_0_l35_35734

noncomputable def f (x r : ℝ) : ℝ := 3 * x^4 - 2 * x^3 + x^2 - 4 * x + r

theorem find_r_for_f_of_3_eq_0 : ∃ r : ℝ, f 3 r = 0 ∧ r = -186 := by
  sorry

end find_r_for_f_of_3_eq_0_l35_35734


namespace translate_line_upwards_l35_35328

theorem translate_line_upwards (x y y' : ℝ) (h : y = -2 * x) (t : y' = y + 4) : y' = -2 * x + 4 :=
by
  sorry

end translate_line_upwards_l35_35328


namespace polynomial_value_l35_35564

theorem polynomial_value (y : ℝ) (h : 4 * y^2 - 2 * y + 5 = 7) : 2 * y^2 - y + 1 = 2 :=
by
  sorry

end polynomial_value_l35_35564


namespace race_dead_heat_l35_35727

theorem race_dead_heat 
  (L Vb : ℝ) 
  (speed_a : ℝ := (16/15) * Vb)
  (speed_c : ℝ := (20/15) * Vb) 
  (time_a : ℝ := L / speed_a)
  (time_b : ℝ := L / Vb)
  (time_c : ℝ := L / speed_c) :
  (1 / (16 / 15) = 3 / 4) → 
  (1 - 3 / 4) = 1 / 4 :=
by 
  sorry

end race_dead_heat_l35_35727


namespace option_D_is_linear_equation_with_two_variables_l35_35206

def is_linear_equation (eq : String) : Prop :=
  match eq with
  | "3x - 6 = x" => false
  | "x = 5 / y - 1" => false
  | "2x - 3y = x^2" => false
  | "3x = 2y" => true
  | _ => false

theorem option_D_is_linear_equation_with_two_variables :
  is_linear_equation "3x = 2y" = true := by
  sorry

end option_D_is_linear_equation_with_two_variables_l35_35206


namespace intersection_complement_l35_35141

open Set

variable (U P Q : Set ℕ)
variable (H_U : U = {1, 2, 3, 4, 5, 6})
variable (H_P : P = {1, 2, 3, 4})
variable (H_Q : Q = {3, 4, 5})

theorem intersection_complement (hU : U = {1, 2, 3, 4, 5, 6}) (hP : P = {1, 2, 3, 4}) (hQ : Q = {3, 4, 5}) :
  P ∩ (U \ Q) = {1, 2} := by
  sorry

end intersection_complement_l35_35141


namespace total_paths_A_to_C_via_B_l35_35883

-- Define the conditions
def steps_from_A_to_B : Nat := 6
def steps_from_B_to_C : Nat := 6
def right_moves_A_to_B : Nat := 4
def down_moves_A_to_B : Nat := 2
def right_moves_B_to_C : Nat := 3
def down_moves_B_to_C : Nat := 3

-- Define binomial coefficient function
def binom (n k : Nat) : Nat :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Calculate the number of paths for each segment
def paths_A_to_B : Nat := binom steps_from_A_to_B down_moves_A_to_B
def paths_B_to_C : Nat := binom steps_from_B_to_C down_moves_B_to_C

-- Theorem stating the total number of distinct paths
theorem total_paths_A_to_C_via_B : paths_A_to_B * paths_B_to_C = 300 :=
by
  sorry

end total_paths_A_to_C_via_B_l35_35883


namespace train_speed_l35_35830

theorem train_speed :
  ∃ V : ℝ,
    (∃ L : ℝ, L = V * 18) ∧ 
    (∃ L : ℝ, L + 260 = V * 31) ∧ 
    V * 3.6 = 72 := by
  sorry

end train_speed_l35_35830


namespace smallest_n_such_that_no_n_digit_is_11_power_l35_35322

theorem smallest_n_such_that_no_n_digit_is_11_power (log_11 : Real) (h : log_11 = 1.0413) : 
  ∃ n > 1, ∀ k : ℕ, ¬ (10 ^ (n - 1) ≤ 11 ^ k ∧ 11 ^ k < 10 ^ n) :=
sorry

end smallest_n_such_that_no_n_digit_is_11_power_l35_35322


namespace Dawn_hourly_earnings_l35_35446

theorem Dawn_hourly_earnings :
  let t_per_painting := 2 
  let num_paintings := 12
  let total_earnings := 3600
  let total_time := t_per_painting * num_paintings
  let hourly_wage := total_earnings / total_time
  hourly_wage = 150 := by
  sorry

end Dawn_hourly_earnings_l35_35446


namespace count_integer_b_for_log_b_256_l35_35494

theorem count_integer_b_for_log_b_256 :
  (∃ b : ℕ, b > 1 ∧ ∃ n : ℕ, n > 0 ∧ b ^ n = 256) ∧ 
  (∀ b : ℕ, (b > 1 ∧ ∃ n : ℕ, n > 0 ∧ b ^ n = 256) → (b = 2 ∨ b = 4 ∨ b = 16 ∨ b = 256)) :=
by sorry

end count_integer_b_for_log_b_256_l35_35494


namespace cone_radius_l35_35139

theorem cone_radius (CSA : ℝ) (l : ℝ) (r : ℝ) (h_CSA : CSA = 989.6016858807849) (h_l : l = 15) :
    r = 21 :=
by
  sorry

end cone_radius_l35_35139


namespace spectators_count_l35_35133

theorem spectators_count (total_wristbands : ℕ) (wristbands_per_person : ℕ) (h1 : total_wristbands = 250) (h2 : wristbands_per_person = 2) : (total_wristbands / wristbands_per_person = 125) :=
by
  sorry

end spectators_count_l35_35133


namespace convert_BFACE_to_decimal_l35_35928

def hex_BFACE : ℕ := 11 * 16^4 + 15 * 16^3 + 10 * 16^2 + 12 * 16^1 + 14 * 16^0

theorem convert_BFACE_to_decimal : hex_BFACE = 785102 := by
  sorry

end convert_BFACE_to_decimal_l35_35928


namespace quadrilateral_area_24_l35_35207

open Classical

noncomputable def quad_area (a b : ℤ) (h : a > b ∧ b > 0) : ℤ :=
let P := (a, b)
let Q := (2*b, a)
let R := (-a, -b)
let S := (-2*b, -a)
-- The proved area
24

theorem quadrilateral_area_24 (a b : ℤ) (h : a > b ∧ b > 0) :
  quad_area a b h = 24 :=
sorry

end quadrilateral_area_24_l35_35207


namespace system_nonzero_solution_l35_35610

-- Definition of the game setup and conditions
def initial_equations (a b c : ℤ) (x y z : ℤ) : Prop :=
  (a * x + b * y + c * z = 0) ∧
  (a * x + b * y + c * z = 0) ∧
  (a * x + b * y + c * z = 0)

-- The main proposition statement in Lean
theorem system_nonzero_solution :
  ∀ (a b c : ℤ), ∃ (x y z : ℤ), x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0 ∧ initial_equations a b c x y z :=
by
  sorry

end system_nonzero_solution_l35_35610


namespace f_one_zero_range_of_a_l35_35376

variable (f : ℝ → ℝ) (a : ℝ)

-- Conditions
def odd_function : Prop := ∀ x : ℝ, x ≠ 0 → f (-x) = -f x
def increasing_on_pos : Prop := ∀ x y : ℝ, 0 < x → x < y → f x < f y
def f_neg_one_zero : Prop := f (-1) = 0
def f_a_minus_half_neg : Prop := f (a - 1/2) < 0

-- Questions
theorem f_one_zero (h1 : odd_function f) (h2 : increasing_on_pos f) (h3 : f_neg_one_zero f) : f 1 = 0 := 
sorry

theorem range_of_a (h1 : odd_function f) (h2 : increasing_on_pos f) (h3 : f_neg_one_zero f) (h4 : f_a_minus_half_neg f a) :
  1/2 < a ∧ a < 3/2 ∨ a < -1/2 :=
sorry

end f_one_zero_range_of_a_l35_35376


namespace angle_between_line_and_plane_l35_35372

-- Define the conditions
def angle_direct_vector_normal_vector (direction_vector_angle : ℝ) := direction_vector_angle = 120

-- Define the goal to prove
theorem angle_between_line_and_plane (direction_vector_angle : ℝ) :
  angle_direct_vector_normal_vector direction_vector_angle → direction_vector_angle = 120 → 90 - (180 - direction_vector_angle) = 30 :=
by
  intros h_angle_eq angle_120
  sorry

end angle_between_line_and_plane_l35_35372


namespace find_b_l35_35057

theorem find_b (b p : ℝ) 
  (h1 : 3 * p + 15 = 0)
  (h2 : 15 * p + 3 = b) :
  b = -72 :=
by
  sorry

end find_b_l35_35057


namespace min_value_y_l35_35289

noncomputable def y (x : ℝ) := x^4 - 4*x + 3

theorem min_value_y : ∃ x ∈ Set.Icc (-2 : ℝ) 3, y x = 0 ∧ ∀ x' ∈ Set.Icc (-2 : ℝ) 3, y x' ≥ 0 :=
by
  sorry

end min_value_y_l35_35289


namespace river_width_l35_35695

def bridge_length : ℕ := 295
def additional_length : ℕ := 192
def total_width : ℕ := 487

theorem river_width (h1 : bridge_length = 295) (h2 : additional_length = 192) : bridge_length + additional_length = total_width := by
  sorry

end river_width_l35_35695


namespace average_percent_increase_per_year_l35_35561

-- Definitions and conditions
def initialPopulation : ℕ := 175000
def finalPopulation : ℕ := 297500
def numberOfYears : ℕ := 10

-- Statement to prove
theorem average_percent_increase_per_year : 
  ((finalPopulation - initialPopulation) / numberOfYears : ℚ) / initialPopulation * 100 = 7 := by
  sorry

end average_percent_increase_per_year_l35_35561


namespace avg_class_weight_l35_35805

def num_students_A : ℕ := 24
def num_students_B : ℕ := 16
def avg_weight_A : ℕ := 40
def avg_weight_B : ℕ := 35

/-- Theorem: The average weight of the whole class is 38 kg --/
theorem avg_class_weight :
  (num_students_A * avg_weight_A + num_students_B * avg_weight_B) / (num_students_A + num_students_B) = 38 :=
by
  -- Proof goes here
  sorry

end avg_class_weight_l35_35805


namespace solution_set_inequality_l35_35566

   theorem solution_set_inequality (a : ℝ) : (∀ x : ℝ, x^2 - 2*a*x + a > 0) ↔ (0 < a ∧ a < 1) :=
   sorry
   
end solution_set_inequality_l35_35566


namespace intersection_of_A_and_B_l35_35658

noncomputable def A : Set ℝ := {-2, -1, 0, 1}
noncomputable def B : Set ℝ := {x | x^2 - 1 ≤ 0}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0, 1} := 
by
  sorry

end intersection_of_A_and_B_l35_35658


namespace cups_per_larger_crust_l35_35149

theorem cups_per_larger_crust
  (initial_crusts : ℕ)
  (initial_flour : ℚ)
  (new_crusts : ℕ)
  (constant_flour : ℚ)
  (h1 : initial_crusts * (initial_flour / initial_crusts) = initial_flour )
  (h2 : new_crusts * (constant_flour / new_crusts) = constant_flour )
  (h3 : initial_flour = constant_flour)
  : (constant_flour / new_crusts) = (8 / 10) :=
by 
  sorry

end cups_per_larger_crust_l35_35149


namespace sum_of_squares_of_roots_l35_35491

theorem sum_of_squares_of_roots (s₁ s₂ : ℝ) (h1 : s₁ + s₂ = 10) (h2 : s₁ * s₂ = 9) : 
  s₁^2 + s₂^2 = 82 := by
  sorry

end sum_of_squares_of_roots_l35_35491


namespace initial_quantity_of_gummy_worms_l35_35204

theorem initial_quantity_of_gummy_worms (x : ℕ) (h : x / 2^4 = 4) : x = 64 :=
sorry

end initial_quantity_of_gummy_worms_l35_35204


namespace ratio_eq_two_l35_35407

theorem ratio_eq_two (a b c d : ℤ) (h1 : b * c + a * d = 1) (h2 : a * c + 2 * b * d = 1) : 
  (a^2 + c^2 : ℚ) / (b^2 + d^2) = 2 :=
sorry

end ratio_eq_two_l35_35407


namespace simplify_expression_simplify_and_evaluate_evaluate_expression_l35_35391

theorem simplify_expression (a b : ℝ) : 8 * (a + b) + 6 * (a + b) - 2 * (a + b) = 12 * (a + b) := 
by sorry

theorem simplify_and_evaluate (x y : ℝ) (h : x + y = 1/2) : 
  9 * (x + y)^2 + 3 * (x + y) + 7 * (x + y)^2 - 7 * (x + y) = 2 := 
by sorry

theorem evaluate_expression (x y : ℝ) (h : x^2 - 2 * y = 4) : -3 * x^2 + 6 * y + 2 = -10 := 
by sorry

end simplify_expression_simplify_and_evaluate_evaluate_expression_l35_35391


namespace time_between_ticks_at_6_oclock_l35_35482

theorem time_between_ticks_at_6_oclock (ticks6 ticks12 intervals6 intervals12 total_time12: ℕ) (time_per_tick : ℕ) :
  ticks6 = 6 →
  ticks12 = 12 →
  total_time12 = 66 →
  intervals12 = ticks12 - 1 →
  time_per_tick = total_time12 / intervals12 →
  intervals6 = ticks6 - 1 →
  (time_per_tick * intervals6) = 30 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end time_between_ticks_at_6_oclock_l35_35482


namespace find_shortest_side_of_triangle_l35_35288

def Triangle (A B C : Type) := true -- Dummy definition for a triangle

structure Segments :=
(BD DE EC : ℝ)

def angle_ratios (AD AE : ℝ) (r1 r2 : ℕ) := true -- Dummy definition for angle ratios

def triangle_conditions (ABC : Type) (s : Segments) (r1 r2 : ℕ)
  (h1 : angle_ratios AD AE r1 r2)
  (h2 : s.BD = 4)
  (h3 : s.DE = 2)
  (h4 : s.EC = 5) : Prop := True

noncomputable def shortestSide (ABC : Type) (s : Segments) (r1 r2 : ℕ) : ℝ := 
  if true then sorry else 0 -- Placeholder for the shortest side length function

theorem find_shortest_side_of_triangle (ABC : Type) (s : Segments)
  (h1 : angle_ratios AD AE 2 3) (h2 : angle_ratios AE AD 1 1)
  (h3 : s.BD = 4) (h4 : s.DE = 2) (h5 : s.EC = 5) :
  shortestSide ABC s 2 3 = 30 / 11 :=
sorry

end find_shortest_side_of_triangle_l35_35288


namespace area_new_rectangle_greater_than_square_l35_35246

theorem area_new_rectangle_greater_than_square (a b : ℝ) (h : a > b) : 
  (2 * (a + b) * (2 * b + a) / 3) > ((a + b) * (a + b)) := 
sorry

end area_new_rectangle_greater_than_square_l35_35246


namespace prob_A_wins_correct_l35_35103

noncomputable def prob_A_wins : ℚ :=
  let outcomes : ℕ := 3^3
  let win_one_draw_two : ℕ := 3
  let win_two_other : ℕ := 6
  let win_all : ℕ := 1
  let total_wins : ℕ := win_one_draw_two + win_two_other + win_all
  total_wins / outcomes

theorem prob_A_wins_correct :
  prob_A_wins = 10/27 :=
by
  sorry

end prob_A_wins_correct_l35_35103


namespace desired_on_time_departure_rate_l35_35549

theorem desired_on_time_departure_rate :
  let first_late := 1
  let on_time_flights_next := 3
  let additional_on_time_flights := 2
  let total_on_time_flights := on_time_flights_next + additional_on_time_flights
  let total_flights := first_late + on_time_flights_next + additional_on_time_flights
  let on_time_departure_rate := (total_on_time_flights : ℚ) / (total_flights : ℚ) * 100
  on_time_departure_rate > 83.33 :=
by
  sorry

end desired_on_time_departure_rate_l35_35549


namespace cartesian_equation_of_line_l35_35387

theorem cartesian_equation_of_line (t x y : ℝ)
  (h1 : x = 1 + t / 2)
  (h2 : y = 2 + (Real.sqrt 3 / 2) * t) :
  Real.sqrt 3 * x - y + 2 - Real.sqrt 3 = 0 :=
sorry

end cartesian_equation_of_line_l35_35387


namespace total_sleep_time_is_correct_l35_35015

-- Define the sleeping patterns of the animals
def cougar_sleep_even_days : ℕ := 4
def cougar_sleep_odd_days : ℕ := 6
def zebra_sleep_more : ℕ := 2

-- Define the distribution of even and odd days in a week
def even_days_in_week : ℕ := 3
def odd_days_in_week : ℕ := 4

-- Define the total weekly sleep time for the cougar
def cougar_total_weekly_sleep : ℕ := 
  (cougar_sleep_even_days * even_days_in_week) + 
  (cougar_sleep_odd_days * odd_days_in_week)

-- Define the total weekly sleep time for the zebra
def zebra_total_weekly_sleep : ℕ := 
  ((cougar_sleep_even_days + zebra_sleep_more) * even_days_in_week) + 
  ((cougar_sleep_odd_days + zebra_sleep_more) * odd_days_in_week)

-- Define the total weekly sleep time for both the cougar and the zebra
def total_weekly_sleep : ℕ := 
  cougar_total_weekly_sleep + zebra_total_weekly_sleep

-- Prove that the total weekly sleep time for both animals is 86 hours
theorem total_sleep_time_is_correct : total_weekly_sleep = 86 :=
by
  -- skipping proof
  sorry

end total_sleep_time_is_correct_l35_35015


namespace rohan_monthly_salary_l35_35265

theorem rohan_monthly_salary :
  ∃ S : ℝ, 
    (0.4 * S) + (0.2 * S) + (0.1 * S) + (0.1 * S) + 1000 = S :=
by
  sorry

end rohan_monthly_salary_l35_35265


namespace julieta_total_cost_l35_35931

variable (initial_backpack_price : ℕ)
variable (initial_binder_price : ℕ)
variable (backpack_price_increase : ℕ)
variable (binder_price_reduction : ℕ)
variable (discount_rate : ℕ)
variable (num_binders : ℕ)

def calculate_total_cost (initial_backpack_price initial_binder_price backpack_price_increase binder_price_reduction discount_rate num_binders : ℕ) : ℝ :=
  let new_backpack_price := initial_backpack_price + backpack_price_increase
  let new_binder_price := initial_binder_price - binder_price_reduction
  let total_bindable_cost := min num_binders ((num_binders + 1) / 2 * new_binder_price)
  let total_pre_discount := new_backpack_price + total_bindable_cost
  let discount_amount := total_pre_discount * discount_rate / 100
  let total_price := total_pre_discount - discount_amount
  total_price

theorem julieta_total_cost
  (initial_backpack_price : ℕ)
  (initial_binder_price : ℕ)
  (backpack_price_increase : ℕ)
  (binder_price_reduction : ℕ)
  (discount_rate : ℕ)
  (num_binders : ℕ)
  (h_initial_backpack : initial_backpack_price = 50)
  (h_initial_binder : initial_binder_price = 20)
  (h_backpack_inc : backpack_price_increase = 5)
  (h_binder_red : binder_price_reduction = 2)
  (h_discount : discount_rate = 10)
  (h_num_binders : num_binders = 3) :
  calculate_total_cost initial_backpack_price initial_binder_price backpack_price_increase binder_price_reduction discount_rate num_binders = 81.90 :=
by
  sorry

end julieta_total_cost_l35_35931


namespace cubic_expression_value_l35_35340

theorem cubic_expression_value (a b c : ℝ) 
  (h1 : a + b + c = 13) 
  (h2 : ab + ac + bc = 32) : 
  a^3 + b^3 + c^3 - 3 * a * b * c = 949 := 
by
  sorry

end cubic_expression_value_l35_35340


namespace fifth_number_in_10th_row_l35_35941

theorem fifth_number_in_10th_row : 
  ∀ (n : ℕ), (∃ (a : ℕ), ∀ (m : ℕ), 1 ≤ m ∧ m ≤ 10 → (m = 10 → a = 67)) :=
by
  sorry

end fifth_number_in_10th_row_l35_35941


namespace total_items_correct_l35_35697

-- Defining the number of each type of items ordered by Betty
def slippers := 6
def lipstick := 4
def hair_color := 8

-- The total number of items ordered by Betty
def total_items := slippers + lipstick + hair_color

-- The statement asserting that the total number of items is 18
theorem total_items_correct : total_items = 18 := 
by 
  -- sorry allows us to skip the proof
  sorry

end total_items_correct_l35_35697


namespace product_of_primes_is_even_l35_35704

-- Define the conditions for P and Q to cover P, Q, P-Q, and P+Q being prime and positive
def is_prime (n : ℕ) : Prop := ¬ (n = 0 ∨ n = 1) ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem product_of_primes_is_even {P Q : ℕ} (hP : is_prime P) (hQ : is_prime Q) 
  (hPQ_diff : is_prime (P - Q)) (hPQ_sum : is_prime (P + Q)) 
  (hPosP : P > 0) (hPosQ : Q > 0) 
  (hPosPQ_diff : P - Q > 0) (hPosPQ_sum : P + Q > 0) : 
  ∃ k : ℕ, P * Q * (P - Q) * (P + Q) = 2 * k := 
sorry

end product_of_primes_is_even_l35_35704


namespace z_neq_5_for_every_k_l35_35338

theorem z_neq_5_for_every_k (z : ℕ) (h₁ : z = 5) :
  ¬ (∀ k : ℕ, k ≥ 1 → ∃ n : ℕ, n ≥ 1 ∧ (∃ m, n ^ 9 % 10 ^ k = z * (10 ^ m))) :=
by
  intro h
  sorry

end z_neq_5_for_every_k_l35_35338


namespace min_value_expression_l35_35131

open Real

theorem min_value_expression : ∀ x y : ℝ, x^2 + y^2 - 8*x + 6*y + 25 ≥ 0 :=
by
  intro x y
  sorry

end min_value_expression_l35_35131


namespace quadratic_solutions_l35_35861

-- Definition of the conditions given in the problem
def quadratic_axis_symmetry (b : ℝ) : Prop :=
  -b / 2 = 2

def equation_solutions (x b : ℝ) : Prop :=
  x^2 + b*x - 5 = 2*x - 13

-- The math proof problem statement in Lean 4
theorem quadratic_solutions (b : ℝ) (x1 x2 : ℝ) :
  quadratic_axis_symmetry b →
  equation_solutions x1 b →
  equation_solutions x2 b →
  (x1 = 2 ∧ x2 = 4) ∨ (x1 = 4 ∧ x2 = 2) :=
by
  sorry

end quadratic_solutions_l35_35861


namespace geometric_sequence_general_formula_l35_35859

noncomputable def a_n (n : ℕ) : ℝ := 2^n

theorem geometric_sequence_general_formula :
  (∀ n : ℕ, 2 * (a_n n + a_n (n + 2)) = 5 * a_n (n + 1)) →
  (a_n 5 ^ 2 = a_n 10) →
  ∀ n : ℕ, a_n n = 2 ^ n := 
by 
  sorry

end geometric_sequence_general_formula_l35_35859


namespace functional_equation_solution_exists_l35_35399

noncomputable def f : ℝ → ℝ := sorry

theorem functional_equation_solution_exists (f : ℝ → ℝ) (h : ∀ x y, 0 < x → 0 < y → f x * f y = 2 * f (x + y * f x)) :
  ∃ c : ℝ, ∀ x, 0 < x → f x = x + c := 
sorry

end functional_equation_solution_exists_l35_35399


namespace count_multiples_of_30_l35_35832

theorem count_multiples_of_30 (a b n : ℕ) (h1 : a = 900) (h2 : b = 27000) 
    (h3 : ∃ n, 30 * n = a) (h4 : ∃ n, 30 * n = b) : 
    (b - a) / 30 + 1 = 871 := 
by
    sorry

end count_multiples_of_30_l35_35832


namespace hexagon_angle_R_l35_35725

theorem hexagon_angle_R (F I G U R E : ℝ) 
  (h1 : F = I ∧ I = R ∧ R = E)
  (h2 : G + U = 180) 
  (sum_angles_hexagon : F + I + G + U + R + E = 720) : 
  R = 135 :=
by sorry

end hexagon_angle_R_l35_35725


namespace work_completion_in_6_days_l35_35661

-- Definitions for the work rates of a, b, and c.
def work_rate_a_b : ℚ := 1 / 8
def work_rate_a : ℚ := 1 / 16
def work_rate_c : ℚ := 1 / 24

-- The theorem to prove that a, b, and c together can complete the work in 6 days.
theorem work_completion_in_6_days : 
  (1 / (work_rate_a_b - work_rate_a)) + work_rate_c = 1 / 6 :=
by
  sorry

end work_completion_in_6_days_l35_35661


namespace sum_numbers_l35_35478

theorem sum_numbers :
  2345 + 3452 + 4523 + 5234 + 3245 + 2453 + 4532 + 5324 = 8888 := by
  sorry

end sum_numbers_l35_35478


namespace two_distinct_solutions_exist_l35_35047

theorem two_distinct_solutions_exist :
  ∃ (a1 b1 c1 d1 e1 a2 b2 c2 d2 e2 : ℕ), 
    1 ≤ a1 ∧ a1 ≤ 9 ∧ 1 ≤ b1 ∧ b1 ≤ 9 ∧ 1 ≤ c1 ∧ c1 ≤ 9 ∧ 1 ≤ d1 ∧ d1 ≤ 9 ∧ 1 ≤ e1 ∧ e1 ≤ 9 ∧
    1 ≤ a2 ∧ a2 ≤ 9 ∧ 1 ≤ b2 ∧ b2 ≤ 9 ∧ 1 ≤ c2 ∧ c2 ≤ 9 ∧ 1 ≤ d2 ∧ d2 ≤ 9 ∧ 1 ≤ e2 ∧ e2 ≤ 9 ∧
    (b1 - d1 = 2) ∧ (d1 - a1 = 3) ∧ (a1 - c1 = 1) ∧
    (b2 - d2 = 2) ∧ (d2 - a2 = 3) ∧ (a2 - c2 = 1) ∧
    ¬ (a1 = a2 ∧ b1 = b2 ∧ c1 = c2 ∧ d1 = d2 ∧ e1 = e2) :=
by
  sorry

end two_distinct_solutions_exist_l35_35047


namespace solve_for_x_l35_35912

theorem solve_for_x (x : ℚ) : (3 * x / 7 - 2 = 12) → (x = 98 / 3) :=
by
  intro h
  sorry

end solve_for_x_l35_35912


namespace parallel_planes_perpendicular_planes_l35_35713

variables {A1 B1 C1 D1 A2 B2 C2 D2 : ℝ}

-- Parallelism Condition
theorem parallel_planes (h₁ : A1 ≠ 0) (h₂ : B1 ≠ 0) (h₃ : C1 ≠ 0) (h₄ : A2 ≠ 0) (h₅ : B2 ≠ 0) (h₆ : C2 ≠ 0) :
  (A1 / A2 = B1 / B2 ∧ B1 / B2 = C1 / C2) ↔ (∃ k : ℝ, (A1 = k * A2) ∧ (B1 = k * B2) ∧ (C1 = k * C2)) :=
sorry

-- Perpendicularity Condition
theorem perpendicular_planes :
  A1 * A2 + B1 * B2 + C1 * C2 = 0 :=
sorry

end parallel_planes_perpendicular_planes_l35_35713


namespace fraction_of_red_knights_magical_l35_35304

def total_knights : ℕ := 28
def red_fraction : ℚ := 3 / 7
def magical_fraction : ℚ := 1 / 4
def red_magical_to_blue_magical_ratio : ℚ := 3

theorem fraction_of_red_knights_magical :
  let red_knights := red_fraction * total_knights
  let blue_knights := total_knights - red_knights
  let total_magical := magical_fraction * total_knights
  let red_magical_fraction := 21 / 52
  let blue_magical_fraction := red_magical_fraction / red_magical_to_blue_magical_ratio
  red_knights * red_magical_fraction + blue_knights * blue_magical_fraction = total_magical :=
by
  sorry

end fraction_of_red_knights_magical_l35_35304


namespace calculate_weight_of_first_batch_jelly_beans_l35_35306

theorem calculate_weight_of_first_batch_jelly_beans (J : ℝ)
    (h1 : 16 = 8 * (J * 4)) : J = 2 := 
  sorry

end calculate_weight_of_first_batch_jelly_beans_l35_35306


namespace derivative_of_reciprocal_at_one_l35_35573

noncomputable def f (x : ℝ) : ℝ := 1 / x

theorem derivative_of_reciprocal_at_one : (deriv f 1) = -1 :=
by {
    sorry
}

end derivative_of_reciprocal_at_one_l35_35573


namespace convert_base_10_to_base_7_l35_35904

def base10_to_base7 (n : ℕ) : ℕ := 
  match n with
  | 5423 => 21545
  | _ => 0

theorem convert_base_10_to_base_7 : base10_to_base7 5423 = 21545 := by
  rfl

end convert_base_10_to_base_7_l35_35904


namespace sin_double_angle_l35_35160

theorem sin_double_angle (x : ℝ) (h : Real.cos (π / 4 - x) = 3 / 5) : Real.sin (2 * x) = -7 / 25 :=
by
  sorry

end sin_double_angle_l35_35160


namespace max_value_of_expression_l35_35868

variable (a b c : ℝ)

theorem max_value_of_expression : 
  ∃ θ : ℝ, a * Real.cos θ + b * Real.sin θ + c = Real.sqrt (a^2 + b^2) + c := by
sorry

end max_value_of_expression_l35_35868


namespace value_of_a_l35_35532

def hyperbolaFociSharedEllipse : Prop :=
  ∃ a > 0, 
    (∃ c h k : ℝ, c = 3 ∧ (h, k) = (3, 0) ∨ (h, k) = (-3, 0)) ∧ 
    ∃ x y : ℝ, ((x^2) / 4) - ((y^2) / 5) = 1 ∧ ((x^2) / (a^2)) + ((y^2) / 16) = 1

theorem value_of_a : ∃ a > 0, hyperbolaFociSharedEllipse ∧ a = 5 :=
by
  sorry

end value_of_a_l35_35532


namespace triangle_sides_l35_35155

noncomputable def sides (a b c : ℝ) : Prop :=
  (a = Real.sqrt (427 / 3)) ∧
  (b = Real.sqrt (427 / 3) + 3/2) ∧
  (c = Real.sqrt (427 / 3) - 3/2)

theorem triangle_sides (a b c : ℝ) (h1 : b - c = 3) (h2 : ∃ d : ℝ, d = 10)
  (h3 : ∃ BD CD : ℝ, CD - BD = 12 ∧ BD + CD = a ∧ 
    a = 2 * (BD + 12 / 2)) :
  sides a b c :=
  sorry

end triangle_sides_l35_35155


namespace fibonacci_periodicity_l35_35460

-- Definitions for p-arithmetic and Fibonacci sequence
def is_prime (p : ℕ) := Nat.Prime p
def sqrt_5_extractable (p : ℕ) : Prop := ∃ k : ℕ, p = 5 * k + 1 ∨ p = 5 * k - 1

-- Definitions of Fibonacci sequences and properties
def fibonacci : ℕ → ℕ
| 0     => 0
| 1     => 1
| (n+2) => fibonacci n + fibonacci (n + 1)

-- Main theorem
theorem fibonacci_periodicity (p : ℕ) (r : ℕ) (h_prime : is_prime p) (h_not_2_or_5 : p ≠ 2 ∧ p ≠ 5)
    (h_period : r = (p+1) ∨ r = (p-1)) (h_div : (sqrt_5_extractable p → r ∣ (p - 1)) ∧ (¬ sqrt_5_extractable p → r ∣ (p + 1)))
    : (fibonacci (p+1) % p = 0 ∨ fibonacci (p-1) % p = 0) := by
          sorry

end fibonacci_periodicity_l35_35460


namespace green_pill_cost_l35_35258

-- Given conditions
def days := 21
def total_cost := 903
def cost_difference := 2
def daily_cost := total_cost / days

-- Statement to prove
theorem green_pill_cost : (∃ (y : ℝ), y + (y - cost_difference) = daily_cost ∧ y = 22.5) :=
by
  sorry

end green_pill_cost_l35_35258


namespace highest_probability_face_l35_35466

theorem highest_probability_face :
  let faces := 6
  let face_3 := 3
  let face_2 := 2
  let face_1 := 1
  (face_3 / faces > face_2 / faces) ∧ (face_2 / faces > face_1 / faces) →
  (face_3 / faces > face_1 / faces) →
  (face_3 = 3) :=
by {
  sorry
}

end highest_probability_face_l35_35466


namespace actual_distance_between_cities_l35_35067

-- Define the scale and distance on the map as constants
def distance_on_map : ℝ := 20
def scale_inch_miles : ℝ := 12  -- Because 1 inch = 12 miles derived from the scale 0.5 inches = 6 miles

-- Define the actual distance calculation
def actual_distance (distance_inch : ℝ) (scale : ℝ) : ℝ :=
  distance_inch * scale

-- Example theorem to prove the actual distance between the cities
theorem actual_distance_between_cities :
  actual_distance distance_on_map scale_inch_miles = 240 := by
  sorry

end actual_distance_between_cities_l35_35067


namespace tomatoes_grew_in_absence_l35_35357

def initial_tomatoes : ℕ := 36
def multiplier : ℕ := 100
def total_tomatoes_after_vacation : ℕ := initial_tomatoes * multiplier

theorem tomatoes_grew_in_absence : 
  total_tomatoes_after_vacation - initial_tomatoes = 3564 :=
by
  -- skipped proof with 'sorry'
  sorry

end tomatoes_grew_in_absence_l35_35357


namespace find_a_plus_b_l35_35143

-- Given points A and B, where A(1, a) and B(b, -2) are symmetric with respect to the origin.
variables (a b : ℤ)

-- Definition for symmetry conditions
def symmetric_wrt_origin (x1 y1 x2 y2 : ℤ) :=
  x2 = -x1 ∧ y2 = -y1

-- The main theorem
theorem find_a_plus_b :
  symmetric_wrt_origin 1 a b (-2) → a + b = 1 :=
by
  sorry

end find_a_plus_b_l35_35143


namespace scenario_a_scenario_b_l35_35469

-- Define the chessboard and the removal function
def is_adjacent (x1 y1 x2 y2 : ℕ) : Prop :=
  (x1 = x2 ∧ (y1 = y2 + 1 ∨ y1 + 1 = y2)) ∨ (y1 = y2 ∧ (x1 = x2 + 1 ∨ x1 + 1 = x2))

def is_square (x y : ℕ) : Prop :=
  1 ≤ x ∧ x ≤ 8 ∧ 1 ≤ y ∧ y ≤ 8

-- Define a Hamiltonian path on the chessboard
inductive HamiltonianPath : (ℕ × ℕ) → (ℕ → (ℕ × ℕ)) → ℕ → Prop
| empty : Π (start : ℕ × ℕ) (path : ℕ → (ℕ × ℕ)), HamiltonianPath start path 0
| step : Π (start : ℕ × ℕ) (path : ℕ → (ℕ × ℕ)) (n : ℕ),
    is_adjacent (path n).1 (path n).2 (path (n+1)).1 (path (n+1)).2 →
    HamiltonianPath start path n →
    (is_square (path (n + 1)).1 (path (n + 1)).2 ∧ ¬ (∃ m < n + 1, path m = path (n + 1))) →
    HamiltonianPath start path (n + 1)

-- State the main theorems
theorem scenario_a : 
  ∃ (path : ℕ → (ℕ × ℕ)),
    HamiltonianPath (3, 2) path 62 ∧
    (∀ n, path n ≠ (2, 2)) := sorry

theorem scenario_b :
  ¬ ∃ (path : ℕ → (ℕ × ℕ)),
    HamiltonianPath (3, 2) path 61 ∧
    (∀ n, path n ≠ (2, 2) ∧ path n ≠ (7, 7)) := sorry

end scenario_a_scenario_b_l35_35469


namespace sum_of_ages_l35_35748

variable (a b c : ℕ)

theorem sum_of_ages (h1 : a = 20 + b + c) (h2 : a^2 = 2000 + (b + c)^2) : a + b + c = 80 := 
by
  sorry

end sum_of_ages_l35_35748


namespace find_multiple_of_t_l35_35515

theorem find_multiple_of_t (k t x y : ℝ) (h1 : x = 1 - k * t) (h2 : y = 2 * t - 2) :
  t = 0.5 → x = y → k = 4 :=
by
  intros ht hxy
  sorry

end find_multiple_of_t_l35_35515


namespace part1_part2_l35_35624

noncomputable def f (x : ℝ) : ℝ := |2 * x - 1| + |5 - x|

theorem part1 : ∃ m, m = 9 / 2 ∧ ∀ x, f x ≥ m :=
sorry

theorem part2 (a b : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : a + b = 3) : 
  (1 / (a + 1) + 1 / (b + 2)) ≥ 2 / 3 :=
sorry

end part1_part2_l35_35624


namespace find_other_root_of_quadratic_l35_35172

theorem find_other_root_of_quadratic (m x_1 x_2 : ℝ) 
  (h_root1 : x_1 = 1) (h_eqn : ∀ x, x^2 - 4 * x + m = 0) : x_2 = 3 :=
by
  sorry

end find_other_root_of_quadratic_l35_35172


namespace card_trick_l35_35763

/-- A magician is able to determine the fifth card from a 52-card deck using a prearranged 
    communication system between the magician and the assistant, thus no supernatural 
    abilities are required. -/
theorem card_trick (deck : Finset ℕ) (h_deck : deck.card = 52) (chosen_cards : Finset ℕ)
  (h_chosen : chosen_cards.card = 5) (shown_cards : Finset ℕ) (h_shown : shown_cards.card = 4)
  (fifth_card : ℕ) (h_fifth_card : fifth_card ∈ chosen_cards \ shown_cards) :
  ∃ (prearranged_system : (Finset ℕ) → (Finset ℕ) → ℕ),
    ∀ (remaining : Finset ℕ), remaining.card = 1 → 
    prearranged_system shown_cards remaining = fifth_card := 
sorry

end card_trick_l35_35763


namespace roots_of_varying_signs_l35_35597

theorem roots_of_varying_signs :
  (∃ x : ℝ, (4 * x^2 - 8 = 40 ∧ x != 0) ∧
           (∃ y : ℝ, (3 * y - 2)^2 = (y + 2)^2 ∧ y != 0) ∧
           (∃ z1 z2 : ℝ, z1 ≠ z2 ∧ (z1 = 0 ∨ z2 = 0) ∧ x^3 - 8 * x^2 + 13 * x + 10 = 0)) :=
sorry

end roots_of_varying_signs_l35_35597


namespace sequence_formula_l35_35295

theorem sequence_formula (a : ℕ → ℝ) (h1 : a 1 = 2) (h2 : ∀ n : ℕ, a (n + 1) = 3 * a n - 2) :
  ∀ n : ℕ, a n = 3^(n - 1) + 1 :=
by sorry

end sequence_formula_l35_35295


namespace correct_product_l35_35929

theorem correct_product : 0.125 * 5.12 = 0.64 := sorry

end correct_product_l35_35929


namespace both_sports_l35_35254

-- Definitions based on the given conditions
def total_members := 80
def badminton_players := 48
def tennis_players := 46
def neither_players := 7

-- The theorem to be proved
theorem both_sports : (badminton_players + tennis_players - (total_members - neither_players)) = 21 := by
  sorry

end both_sports_l35_35254


namespace solve_inequality_l35_35359

open Set Real

def condition1 (x : ℝ) : Prop := 6 * x + 2 < (x + 2) ^ 2
def condition2 (x : ℝ) : Prop := (x + 2) ^ 2 < 8 * x + 4

theorem solve_inequality (x : ℝ) : condition1 x ∧ condition2 x ↔ x ∈ Ioo (2 + Real.sqrt 2) 4 := by
  sorry

end solve_inequality_l35_35359


namespace tangent_line_inv_g_at_0_l35_35971

noncomputable def g (x : ℝ) := Real.log x

theorem tangent_line_inv_g_at_0 
  (h₁ : ∀ x, g x = Real.log x) 
  (h₂ : ∀ x, x > 0): 
  ∃ m b, (∀ x y, y = g⁻¹ x → y - m * x = b) ∧ 
         (m = 1) ∧ 
         (b = 1) ∧ 
         (∀ x y, x - y + 1 = 0) := 
by
  sorry

end tangent_line_inv_g_at_0_l35_35971


namespace better_fit_model_l35_35946

-- Define the residual sums of squares
def RSS_1 : ℝ := 152.6
def RSS_2 : ℝ := 159.8

-- Define the statement that the model with RSS_1 is the better fit
theorem better_fit_model : RSS_1 < RSS_2 → RSS_1 = 152.6 :=
by
  sorry

end better_fit_model_l35_35946


namespace fifth_segment_student_l35_35382

variable (N : ℕ) (n : ℕ) (second_segment_student : ℕ)

def sampling_interval (N n : ℕ) : ℕ := N / n

def initial_student (second_segment_student interval : ℕ) : ℕ := second_segment_student - interval

def student_number (initial_student interval : ℕ) (segment : ℕ) : ℕ :=
  initial_student + (segment - 1) * interval

theorem fifth_segment_student (N n : ℕ) (second_segment_student : ℕ) (hN : N = 700) (hn : n = 50) (hsecond : second_segment_student = 20) :
  student_number (initial_student second_segment_student (sampling_interval N n)) (sampling_interval N n) 5 = 62 := by
  sorry

end fifth_segment_student_l35_35382


namespace area_of_triangle_l35_35201

-- Define the coordinates of the vertices
def A : ℝ × ℝ := (0, 3)
def B : ℝ × ℝ := (7, -1)
def C : ℝ × ℝ := (2, 6)

-- Define the function to calculate the area of the triangle formed by three vertices
def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

-- The theorem statement that the area of the triangle with given vertices is 14.5
theorem area_of_triangle : triangle_area A B C = 14.5 :=
by 
  -- Skipping the proof part
  sorry

end area_of_triangle_l35_35201


namespace crayons_selection_l35_35164

theorem crayons_selection : 
  ∃ (n : ℕ), n = Nat.choose 14 4 ∧ n = 1001 := by
  sorry

end crayons_selection_l35_35164


namespace triangle_inequality_inequality_l35_35952

theorem triangle_inequality_inequality {a b c : ℝ}
  (h1 : a + b > c) (h2 : b + c > a) (h3 : a + c > b) :
  3 * (b + c - a) * (c + a - b) * (a + b - c) ≤ a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) :=
sorry

end triangle_inequality_inequality_l35_35952


namespace correct_option_is_D_l35_35798

noncomputable def option_A := 230
noncomputable def option_B := [251, 260]
noncomputable def option_B_average := 256
noncomputable def option_C := [21, 212, 256]
noncomputable def option_C_average := 163
noncomputable def option_D := [210, 240, 250]
noncomputable def option_D_average := 233

theorem correct_option_is_D :
  ∃ (correct_option : String), correct_option = "D" :=
  sorry

end correct_option_is_D_l35_35798


namespace veranda_width_l35_35016

-- Defining the conditions as given in the problem
def room_length : ℝ := 21
def room_width : ℝ := 12
def veranda_area : ℝ := 148

-- The main statement to prove
theorem veranda_width :
  ∃ (w : ℝ), (21 + 2*w) * (12 + 2*w) - 21 * 12 = 148 ∧ w = 2 :=
by
  sorry

end veranda_width_l35_35016


namespace triangle_cosine_identity_l35_35416

open Real

variables {A B C a b c : ℝ}

theorem triangle_cosine_identity (h : b = (a + c) / 2) : cos (A - C) + 4 * cos B = 3 :=
sorry

end triangle_cosine_identity_l35_35416


namespace BKINGTON_appears_first_on_eighth_line_l35_35059

-- Define the cycle lengths for letters and digits
def cycle_letters : ℕ := 8
def cycle_digits : ℕ := 4

-- Define the problem statement
theorem BKINGTON_appears_first_on_eighth_line :
  Nat.lcm cycle_letters cycle_digits = 8 := by
  sorry

end BKINGTON_appears_first_on_eighth_line_l35_35059


namespace division_problem_l35_35326

theorem division_problem :
  250 / (5 + 12 * 3^2) = 250 / 113 :=
by sorry

end division_problem_l35_35326


namespace multiplication_equation_l35_35236

-- Define the given conditions
def multiplier : ℕ := 6
def product : ℕ := 168
def multiplicand : ℕ := product - 140

-- Lean statement for the proof
theorem multiplication_equation : multiplier * multiplicand = product := by
  sorry

end multiplication_equation_l35_35236


namespace odd_function_expression_on_negative_domain_l35_35051

theorem odd_function_expression_on_negative_domain
  (f : ℝ → ℝ)
  (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_pos : ∀ x : ℝ, 0 < x → f x = x * (x - 1))
  (x : ℝ)
  (h_neg : x < 0)
  : f x = x * (x + 1) :=
sorry

end odd_function_expression_on_negative_domain_l35_35051


namespace julia_stairs_less_than_third_l35_35260

theorem julia_stairs_less_than_third (J1 : ℕ) (T : ℕ) (T_total : ℕ) (J : ℕ) 
  (hJ1 : J1 = 1269) (hT : T = 1269 / 3) (hT_total : T_total = 1685) (hTotal : J1 + J = T_total) : 
  T - J = 7 := 
by
  sorry

end julia_stairs_less_than_third_l35_35260


namespace candies_left_l35_35354

-- Defining the given conditions
def initial_candies : Nat := 30
def eaten_candies : Nat := 23

-- Define the target statement to prove
theorem candies_left : initial_candies - eaten_candies = 7 := by
  sorry

end candies_left_l35_35354


namespace prove_a_5_l35_35194

noncomputable def a_5_proof : Prop :=
  ∀ (a : ℕ → ℝ) (q : ℝ),
    (∀ n, a n > 0) → 
    (a 1 + 2 * a 2 = 4) →
    ((a 1)^2 * q^6 = 4 * a 1 * q^2 * a 1 * q^6) →
    a 5 = 1 / 8

theorem prove_a_5 : a_5_proof := sorry

end prove_a_5_l35_35194


namespace number_of_f3_and_sum_of_f3_l35_35870

noncomputable def f : ℝ → ℝ := sorry
variable (a : ℝ)

theorem number_of_f3_and_sum_of_f3 (hf : ∀ x y : ℝ, f (f x - y) = f x + f (f y - f a) + x) :
  (∃! c : ℝ, f 3 = c) ∧ (∃ s : ℝ, (∀ c, f 3 = c → s = c) ∧ s = 3) :=
sorry

end number_of_f3_and_sum_of_f3_l35_35870


namespace vicki_donated_fraction_l35_35648

/-- Given Jeff had 300 pencils and donated 30% of them, and Vicki had twice as many pencils as Jeff originally 
    had, and there are 360 pencils remaining altogether after both donations,
    prove that Vicki donated 3/4 of her pencils. -/
theorem vicki_donated_fraction : 
  let jeff_pencils := 300
  let jeff_donated := jeff_pencils * 0.30
  let jeff_remaining := jeff_pencils - jeff_donated
  let vicki_pencils := 2 * jeff_pencils
  let total_remaining := 360
  let vicki_remaining := total_remaining - jeff_remaining
  let vicki_donated := vicki_pencils - vicki_remaining
  vicki_donated / vicki_pencils = 3 / 4 :=
by
  -- Proof needs to be inserted here
  sorry

end vicki_donated_fraction_l35_35648


namespace convert_deg_to_rad_l35_35032

theorem convert_deg_to_rad (deg : ℝ) (π : ℝ) (h : deg = 50) : (deg * (π / 180) = 5 / 18 * π) :=
by
  -- Conditions
  sorry

end convert_deg_to_rad_l35_35032


namespace percentage_of_full_marks_D_l35_35888

theorem percentage_of_full_marks_D (full_marks a b c d : ℝ)
  (h_full_marks : full_marks = 500)
  (h_a : a = 360)
  (h_a_b : a = b - 0.10 * b)
  (h_b_c : b = c + 0.25 * c)
  (h_c_d : c = d - 0.20 * d) :
  d / full_marks * 100 = 80 :=
by
  sorry

end percentage_of_full_marks_D_l35_35888


namespace find_all_real_solutions_l35_35775

theorem find_all_real_solutions (x : ℝ) :
    (1 / ((x - 1) * (x - 2))) + (1 / ((x - 2) * (x - 3))) + (1 / ((x - 3) * (x - 4))) + (1 / ((x - 4) * (x - 5))) = 1 / 4 →
    x = 1 ∨ x = 5 :=
by
  sorry

end find_all_real_solutions_l35_35775


namespace four_digit_number_l35_35013

-- Definitions of the conditions
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

-- Statement of the theorem
theorem four_digit_number (x y : ℕ) (hx : is_two_digit x) (hy : is_two_digit y) :
    (100 * x + y) = 1000 * x + y := sorry

end four_digit_number_l35_35013


namespace fraction_eq_l35_35598

def at_op (a b : ℝ) : ℝ := a * b - a * b^2
def hash_op (a b : ℝ) : ℝ := a^2 + b - a^2 * b

theorem fraction_eq :
  (at_op 8 3) / (hash_op 8 3) = 48 / 125 :=
by sorry

end fraction_eq_l35_35598


namespace inclination_angle_range_l35_35228

theorem inclination_angle_range (k : ℝ) (α : ℝ) (h1 : -1 ≤ k) (h2 : k < 1)
  (h3 : k = Real.tan α) (h4 : 0 ≤ α) (h5 : α < 180) :
  (0 ≤ α ∧ α < 45) ∨ (135 ≤ α ∧ α < 180) :=
sorry

end inclination_angle_range_l35_35228


namespace initial_weight_of_beef_l35_35169

theorem initial_weight_of_beef (W : ℝ) 
  (stage1 : W' = 0.70 * W) 
  (stage2 : W'' = 0.80 * W') 
  (stage3 : W''' = 0.50 * W'') 
  (final_weight : W''' = 315) : 
  W = 1125 := by 
  sorry

end initial_weight_of_beef_l35_35169


namespace triangle_angle_contradiction_l35_35363

theorem triangle_angle_contradiction (a b c : ℝ) (h₁ : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h₂ : a + b + c = 180) (h₃ : 60 < a ∧ 60 < b ∧ 60 < c) : false :=
by
  sorry

end triangle_angle_contradiction_l35_35363


namespace greatest_natural_number_exists_l35_35626

noncomputable def sum_of_squares (n : ℕ) : ℕ :=
    n * (n + 1) * (2 * n + 1) / 6

noncomputable def squared_sum_from_to (a b : ℕ) : ℕ :=
    sum_of_squares b - sum_of_squares (a - 1)

noncomputable def is_perfect_square (n : ℕ) : Prop :=
    ∃ k, k * k = n

theorem greatest_natural_number_exists :
    ∃ n : ℕ, n = 1921 ∧ n ≤ 2008 ∧ 
    is_perfect_square ((sum_of_squares n) * (squared_sum_from_to (n + 1) (2 * n))) :=
by
  sorry

end greatest_natural_number_exists_l35_35626


namespace solution_point_satisfies_inequalities_l35_35779

theorem solution_point_satisfies_inequalities:
  let x := -1/3
  let y := 2/3
  11 * x^2 + 8 * x * y + 8 * y^2 ≤ 3 ∧ x - 4 * y ≤ -3 :=
by
  let x := -1/3
  let y := 2/3
  sorry

end solution_point_satisfies_inequalities_l35_35779


namespace problem_1_problem_2_problem_3_l35_35823

-- Simplified and combined statements for clarity
theorem problem_1 (f : ℝ → ℝ) (h_even : ∀ x, f x = f (-x)) 
  (h_cond : ∀ x ≤ 0, f x = Real.logb (1/2) (-x + 1)) : 
  f 3 + f (-1) = -3 := sorry

theorem problem_2 (f : ℝ → ℝ) (h_even : ∀ x, f x = f (-x)) 
  (h_cond : ∀ x ≤ 0, f x = Real.logb (1/2) (-x + 1)) : 
  ∀ x, f x = if x ≤ 0 then Real.logb (1/2) (-x + 1) else Real.logb (1/2) (x + 1) := sorry

theorem problem_3 (f : ℝ → ℝ) (h_cond : ∀ x ≤ 0, f x = Real.logb (1/2) (-x + 1))
  (h_cond_ev : ∀ x, f x = f (-x)) (a : ℝ) : 
  f (a - 1) < -1 ↔ a ∈ ((Set.Iio 0) ∪ (Set.Ioi 2)) := sorry

end problem_1_problem_2_problem_3_l35_35823


namespace contrapositive_of_x_squared_eq_one_l35_35291

theorem contrapositive_of_x_squared_eq_one (x : ℝ) 
  (h : x^2 = 1 → x = 1 ∨ x = -1) : (x ≠ 1 ∧ x ≠ -1) → x^2 ≠ 1 :=
by
  sorry

end contrapositive_of_x_squared_eq_one_l35_35291


namespace total_number_of_legs_is_40_l35_35457

-- Define the number of octopuses Carson saw.
def number_of_octopuses := 5

-- Define the number of legs per octopus.
def legs_per_octopus := 8

-- Define the total number of octopus legs Carson saw.
def total_octopus_legs : Nat := number_of_octopuses * legs_per_octopus

-- Prove that the total number of octopus legs Carson saw is 40.
theorem total_number_of_legs_is_40 : total_octopus_legs = 40 := by
  sorry

end total_number_of_legs_is_40_l35_35457


namespace parabola_focus_l35_35113

theorem parabola_focus (a : ℝ) : (∀ x : ℝ, y = a * x^2) ∧ ∃ f : ℝ × ℝ, f = (0, 1) → a = (1/4) := 
sorry

end parabola_focus_l35_35113


namespace speed_of_current_l35_35932

variable (c : ℚ) -- Speed of the current in miles per hour
variable (d : ℚ) -- Distance to the certain point in miles

def boat_speed := 16 -- Boat's speed relative to water in mph
def upstream_time := (20:ℚ) / 60 -- Time upstream in hours 
def downstream_time := (15:ℚ) / 60 -- Time downstream in hours

theorem speed_of_current (h1 : d = (boat_speed - c) * upstream_time)
                         (h2 : d = (boat_speed + c) * downstream_time) :
    c = 16 / 7 :=
  by
  sorry

end speed_of_current_l35_35932


namespace exists_x0_l35_35109

noncomputable def f (x : Real) (a : Real) : Real :=
  Real.exp x - a * Real.sin x

theorem exists_x0 (a : Real) (h : a = 1) :
  ∃ x0 ∈ Set.Ioo (-Real.pi / 2) 0, 1 < f x0 a ∧ f x0 a < Real.sqrt 2 :=
  sorry

end exists_x0_l35_35109


namespace doves_eggs_l35_35462

theorem doves_eggs (initial_doves total_doves : ℕ) (fraction_hatched : ℚ) (E : ℕ)
  (h_initial_doves : initial_doves = 20)
  (h_total_doves : total_doves = 65)
  (h_fraction_hatched : fraction_hatched = 3/4)
  (h_after_hatching : total_doves = initial_doves + fraction_hatched * E * initial_doves) :
  E = 3 :=
by
  -- The proof would go here.
  sorry

end doves_eggs_l35_35462


namespace bird_wings_l35_35592

theorem bird_wings (P Pi C : ℕ) (h_total_money : 4 * 50 = 200)
  (h_total_cost : 30 * P + 20 * Pi + 15 * C = 200)
  (h_P_ge : P ≥ 1) (h_Pi_ge : Pi ≥ 1) (h_C_ge : C ≥ 1) :
  2 * (P + Pi + C) = 24 :=
sorry

end bird_wings_l35_35592


namespace right_triangle_side_lengths_l35_35002

theorem right_triangle_side_lengths (a S : ℝ) (b c : ℝ)
  (h1 : S = b + c)
  (h2 : c^2 = a^2 + b^2) :
  b = (S^2 - a^2) / (2 * S) ∧ c = (S^2 + a^2) / (2 * S) :=
by
  sorry

end right_triangle_side_lengths_l35_35002


namespace find_multiple_of_t_l35_35199

variable (t : ℝ)
variable (x y : ℝ)

theorem find_multiple_of_t (h1 : x = 1 - 4 * t)
  (h2 : ∃ m : ℝ, y = m * t - 2)
  (h3 : t = 0.5)
  (h4 : x = y) : ∃ m : ℝ, (m = 2) :=
by
  sorry

end find_multiple_of_t_l35_35199


namespace intersect_point_one_l35_35324

theorem intersect_point_one (k : ℝ) : 
  (∀ y : ℝ, (x = -3 * y^2 - 2 * y + 4 ↔ x = k)) ↔ k = 13 / 3 := 
by
  sorry

end intersect_point_one_l35_35324


namespace range_of_m_l35_35962

def P (m : ℝ) : Prop := m^2 - 4 > 0
def Q (m : ℝ) : Prop := 16 * (m - 2)^2 - 16 < 0

theorem range_of_m (m : ℝ) : ¬(P m ∧ Q m) ∧ (P m ∨ Q m) ↔ (m < -2 ∨ (1 < m ∧ m ≤ 2) ∨ m ≥ 3) :=
by
  sorry

end range_of_m_l35_35962


namespace negation_of_existential_l35_35039

theorem negation_of_existential (x : ℝ) : ¬(∃ x : ℝ, x^2 - 2 * x + 3 > 0) ↔ ∀ x : ℝ, x^2 - 2 * x + 3 ≤ 0 := 
by
  sorry

end negation_of_existential_l35_35039


namespace sheep_count_l35_35504

theorem sheep_count (cows sheep shepherds : ℕ) 
  (h_cows : cows = 12) 
  (h_ears : 2 * cows < sheep) 
  (h_legs : sheep < 4 * cows) 
  (h_shepherds : sheep = 12 * shepherds) :
  sheep = 36 :=
by {
  sorry
}

end sheep_count_l35_35504


namespace alex_needs_additional_coins_l35_35267

theorem alex_needs_additional_coins : 
  let friends := 12
  let coins := 63
  let total_coins_needed := (friends * (friends + 1)) / 2 
  let additional_coins_needed := total_coins_needed - coins
  additional_coins_needed = 15 :=
by sorry

end alex_needs_additional_coins_l35_35267


namespace integer_for_all_n_l35_35280

theorem integer_for_all_n
  (x y : ℝ)
  (f : ℕ → ℤ)
  (h : ∀ n : ℕ, 1 ≤ n ∧ n ≤ 4 → f n = ((x^n - y^n) / (x - y))) :
  ∀ n : ℕ, 0 < n → f n = ((x^n - y^n) / (x - y)) :=
by sorry

end integer_for_all_n_l35_35280


namespace train_speed_l35_35542

theorem train_speed (length_of_train time_to_cross : ℝ) (h_length : length_of_train = 800) (h_time : time_to_cross = 12) : (length_of_train / time_to_cross) = 66.67 :=
by
  sorry

end train_speed_l35_35542


namespace cupcake_difference_l35_35360

def betty_rate : ℕ := 10
def dora_rate : ℕ := 8
def total_hours : ℕ := 5
def betty_break_hours : ℕ := 2

theorem cupcake_difference :
  (dora_rate * total_hours) - (betty_rate * (total_hours - betty_break_hours)) = 10 :=
by
  sorry

end cupcake_difference_l35_35360


namespace max_f_theta_l35_35554

noncomputable def determinant (a b c d : ℝ) : ℝ := a*d - b*c

noncomputable def f (θ : ℝ) : ℝ :=
  determinant (Real.sin θ) (Real.cos θ) (-1) 1

theorem max_f_theta :
  ∀ θ : ℝ, 0 < θ ∧ θ < (Real.pi / 3) →
  f θ ≤ (Real.sqrt 6 + Real.sqrt 2) / 4 :=
by
  sorry

end max_f_theta_l35_35554


namespace daisy_count_per_bouquet_l35_35749

-- Define the conditions
def roses_per_bouquet := 12
def total_bouquets := 20
def rose_bouquets := 10
def daisy_bouquets := total_bouquets - rose_bouquets
def total_flowers_sold := 190
def total_roses_sold := rose_bouquets * roses_per_bouquet
def total_daisies_sold := total_flowers_sold - total_roses_sold

-- Define the problem: prove that the number of daisies per bouquet is 7
theorem daisy_count_per_bouquet : total_daisies_sold / daisy_bouquets = 7 := by
  sorry

end daisy_count_per_bouquet_l35_35749


namespace f_1982_l35_35560

-- Define the function f and the essential properties and conditions
def f : ℕ → ℕ := sorry

axiom f_nonneg (n : ℕ) : f n ≥ 0
axiom f_add_property (m n : ℕ) : f (m + n) - f m - f n = 0 ∨ f (m + n) - f m - f n = 1
axiom f_2 : f 2 = 0
axiom f_3_pos : f 3 > 0
axiom f_9999 : f 9999 = 3333

-- Statement of the theorem we want to prove
theorem f_1982 : f 1982 = 660 := 
  by sorry

end f_1982_l35_35560


namespace equivalent_resistance_A_B_l35_35148

-- Parameters and conditions
def resistor_value : ℝ := 5 -- in MΩ
def num_resistors : ℕ := 4
def has_bridging_wire : Prop := true
def negligible_wire_resistance : Prop := true

-- Problem: Prove the equivalent resistance (R_eff) between points A and B is 5 MΩ.
theorem equivalent_resistance_A_B : 
  ∀ (R : ℝ) (n : ℕ) (bridge : Prop) (negligible_wire : Prop),
    R = 5 → n = 4 → bridge → negligible_wire → R = 5 :=
by sorry

end equivalent_resistance_A_B_l35_35148


namespace part1_part2_l35_35576

section
variable (k : ℝ)

/-- Part 1: Range of k -/
def discriminant_eqn (k : ℝ) := (2 * k - 1) ^ 2 - 4 * (k ^ 2 - 1)

theorem part1 (h : discriminant_eqn k ≥ 0) : k ≤ 5 / 4 :=
by sorry

/-- Part 2: Value of k when x₁ and x₂ satisfy the given condition -/
def x1_x2_eqn (k x1 x2 : ℝ) := x1 ^ 2 + x2 ^ 2 = 16 + x1 * x2

def vieta (k : ℝ) (x1 x2 : ℝ) :=
  x1 + x2 = 1 - 2 * k ∧ x1 * x2 = k ^ 2 - 1

theorem part2 (x1 x2 : ℝ) (h1 : vieta k x1 x2) (h2 : x1_x2_eqn k x1 x2) : k = -2 :=
by sorry

end

end part1_part2_l35_35576


namespace cuboid_surface_area_increase_l35_35824

variables (L W H : ℝ)
def SA_original (L W H : ℝ) : ℝ := 2 * (L * W + L * H + W * H)

def SA_new (L W H : ℝ) : ℝ := 2 * ((1.50 * L) * (1.70 * W) + (1.50 * L) * (1.80 * H) + (1.70 * W) * (1.80 * H))

theorem cuboid_surface_area_increase :
  (SA_new L W H - SA_original L W H) / SA_original L W H * 100 = 315.5 :=
by
  sorry

end cuboid_surface_area_increase_l35_35824


namespace Xia_shared_stickers_l35_35948

def stickers_shared (initial remaining sheets_per_sheet : ℕ) : ℕ :=
  initial - (remaining * sheets_per_sheet)

theorem Xia_shared_stickers :
  stickers_shared 150 5 10 = 100 :=
by
  sorry

end Xia_shared_stickers_l35_35948


namespace isosceles_triangle_perimeter_l35_35668

variable (a b : ℕ) 

theorem isosceles_triangle_perimeter (h1 : a = 3) (h2 : b = 6) : 
  ∃ P, (a = 3 ∧ b = 6 ∧ P = 15 ∨ b = 3 ∧ a = 6 ∧ P = 15) := by
  use 15
  sorry

end isosceles_triangle_perimeter_l35_35668


namespace problem_statement_l35_35945

noncomputable def f (x : ℝ) : ℝ := x + 1 / x - Real.sqrt 2

theorem problem_statement (x : ℝ) (h₁ : x ∈ Set.Ioc (Real.sqrt 2 / 2) 1) :
  Real.sqrt 2 / 2 < f (f x) ∧ f (f x) < x :=
by
  sorry

end problem_statement_l35_35945


namespace rectangle_area_l35_35193

theorem rectangle_area (w l : ℝ) (h1 : l = 3 * w) (h2 : 2 * (w + l) = 72) : w * l = 243 := by
  sorry

end rectangle_area_l35_35193


namespace physics_marks_l35_35238

theorem physics_marks (P C M : ℕ) 
  (h1 : P + C + M = 195)
  (h2 : P + M = 180)
  (h3 : P + C = 140) :
  P = 125 :=
by {
  sorry
}

end physics_marks_l35_35238


namespace part1_part2_l35_35715

noncomputable def choose (n : ℕ) (k : ℕ) : ℕ :=
  n.choose k

theorem part1 :
  let internal_medicine_doctors := 12
  let surgeons := 8
  let total_doctors := internal_medicine_doctors + surgeons
  let team_size := 5
  let doctors_left := total_doctors - 1 - 1 -- as one internal medicine must participate and one surgeon cannot
  choose doctors_left (team_size - 1) = 3060 := by
  sorry

theorem part2 :
  let internal_medicine_doctors := 12
  let surgeons := 8
  let total_doctors := internal_medicine_doctors + surgeons
  let team_size := 5
  let only_internal_medicine := choose internal_medicine_doctors team_size
  let only_surgeons := choose surgeons team_size
  let total_ways := choose total_doctors team_size
  total_ways - only_internal_medicine - only_surgeons = 14656 := by
  sorry

end part1_part2_l35_35715


namespace sum_of_relatively_prime_integers_l35_35979

theorem sum_of_relatively_prime_integers (x y : ℕ) (h1 : 0 < x) (h2 : 0 < y)
  (h3 : x * y + x + y = 154) (h4 : Nat.gcd x y = 1) (h5 : x < 30) (h6 : y < 30) : 
  x + y = 34 :=
sorry -- proof

end sum_of_relatively_prime_integers_l35_35979


namespace original_savings_l35_35622

-- Given conditions:
def total_savings (s : ℝ) : Prop :=
  1 / 4 * s = 230

-- Theorem statement: 
theorem original_savings (s : ℝ) (h : total_savings s) : s = 920 :=
sorry

end original_savings_l35_35622


namespace neg_exists_n_sq_gt_two_pow_n_l35_35484

open Classical

theorem neg_exists_n_sq_gt_two_pow_n :
  (¬ ∃ n : ℕ, n^2 > 2^n) ↔ (∀ n : ℕ, n^2 ≤ 2^n) := by
  sorry

end neg_exists_n_sq_gt_two_pow_n_l35_35484


namespace total_volume_of_drink_l35_35889

theorem total_volume_of_drink :
  ∀ (total_ounces : ℝ),
    (∀ orange_juice watermelon_juice grape_juice : ℝ,
      orange_juice = 0.25 * total_ounces →
      watermelon_juice = 0.4 * total_ounces →
      grape_juice = 0.35 * total_ounces →
      grape_juice = 105 →
      total_ounces = 300) :=
by
  intros total_ounces orange_juice watermelon_juice grape_juice ho hw hg hg_eq
  sorry

end total_volume_of_drink_l35_35889


namespace number_of_pines_l35_35106

theorem number_of_pines (total_trees : ℕ) (T B S : ℕ) (h_total : total_trees = 101)
  (h_poplar_spacing : ∀ T1 T2, T1 ≠ T2 → (T2 - T1) > 1)
  (h_birch_spacing : ∀ B1 B2, B1 ≠ B2 → (B2 - B1) > 2)
  (h_pine_spacing : ∀ S1 S2, S1 ≠ S2 → (S2 - S1) > 3) :
  S = 25 ∨ S = 26 :=
sorry

end number_of_pines_l35_35106


namespace sixth_employee_salary_l35_35033

-- We define the salaries of the five employees
def salaries : List ℝ := [1000, 2500, 3100, 1500, 2000]

-- The mean of the salaries of these 5 employees and another employee
def mean_salary : ℝ := 2291.67

-- The number of employees
def number_of_employees : ℝ := 6

-- The total salary of the first five employees
def total_salary_5 : ℝ := salaries.sum

-- The total salary based on the given mean and number of employees
def total_salary_all : ℝ := mean_salary * number_of_employees

-- The statement to prove: The salary of the sixth employee
theorem sixth_employee_salary :
  total_salary_all - total_salary_5 = 3650.02 := 
  sorry

end sixth_employee_salary_l35_35033


namespace largest_divisor_of_m_p1_l35_35272

theorem largest_divisor_of_m_p1 (m : ℕ) (h1 : m > 0) (h2 : 72 ∣ m^3) : 6 ∣ m :=
sorry

end largest_divisor_of_m_p1_l35_35272


namespace find_k_l35_35352

theorem find_k
  (k x1 x2 : ℝ)
  (h1 : x1^2 - 3*x1 + k = 0)
  (h2 : x2^2 - 3*x2 + k = 0)
  (h3 : x1 = 2 * x2) :
  k = 2 :=
sorry

end find_k_l35_35352


namespace grandma_can_give_cherry_exists_better_grand_strategy_l35_35996

variable (Packet1 : Finset String) (Packet2 : Finset String) (Packet3 : Finset String)
variable (isCabbage : String → Prop) (isCherry : String → Prop)
variable (wholePie : String → Prop)

-- Conditions
axiom Packet1_cond : ∀ p ∈ Packet1, isCabbage p
axiom Packet2_cond : ∀ p ∈ Packet2, isCherry p
axiom Packet3_cond_cabbage : ∃ p ∈ Packet3, isCabbage p
axiom Packet3_cond_cherry : ∃ p ∈ Packet3, isCherry p

-- Question (a)
theorem grandma_can_give_cherry (h1 : ∃ p1 ∈ Packet3, wholePie p1 ∧ isCherry p1 ∨
    ∃ p2 ∈ Packet1, wholePie p2 ∧ (∃ q ∈ Packet2 ∪ Packet3, isCherry q ∧ wholePie q) ∨
    ∃ p3 ∈ Packet2, wholePie p3 ∧ isCherry p3) :
  ∃ grand_strategy, grand_strategy = (2 / 3) * (1 : ℝ) :=
by
  sorry

-- Question (b)
theorem exists_better_grand_strategy (h2 : ∃ p ∈ Packet3, wholePie p ∧ isCherry p ∨
    ∃ p2 ∈ Packet1, wholePie p2 ∧ (∃ q ∈ Packet2 ∪ Packet3, isCherry q ∧ wholePie q) ∨
    ∃ p3 ∈ Packet2, wholePie p3 ∧ isCherry p3) :
  ∃ grand_strategy, grand_strategy > (2 / 3) * (1 : ℝ) :=
by
  sorry

end grandma_can_give_cherry_exists_better_grand_strategy_l35_35996


namespace correct_subtraction_result_l35_35235

-- Definitions based on the problem conditions
def initial_two_digit_number (X Y : ℕ) : ℕ := X * 10 + Y

-- Lean statement that expresses the proof problem
theorem correct_subtraction_result (X Y : ℕ) (H1 : initial_two_digit_number X Y = 99) (H2 : 57 = 57) :
  99 - 57 = 42 :=
by
  sorry

end correct_subtraction_result_l35_35235


namespace findLineEquation_l35_35807

-- Define the point P
structure Point where
  x : ℝ
  y : ℝ

-- Define a function to represent the hyperbola condition
def isOnHyperbola (pt : Point) : Prop :=
  pt.x ^ 2 - 4 * pt.y ^ 2 = 4

-- Define midpoint condition for points A and B
def isMidpoint (P A B : Point) : Prop :=
  P.x = (A.x + B.x) / 2 ∧ P.y = (A.y + B.y) / 2

-- Define points
def P : Point := ⟨8, 1⟩
def A : Point := sorry
def B : Point := sorry

-- Statement to prove
theorem findLineEquation :
  isOnHyperbola A ∧ isOnHyperbola B ∧ isMidpoint P A B →
  ∃ m b, (∀ pt : Point, pt.y = m * pt.x + b ↔ pt.x = 8 ∧ pt.y = 1) ∧ (m = 2) ∧ (b = -15) :=
by
  sorry

end findLineEquation_l35_35807


namespace find_y_l35_35615

theorem find_y : ∃ y : ℝ, 1.5 * y - 10 = 35 ∧ y = 30 :=
by
  sorry

end find_y_l35_35615


namespace smallest_positive_multiple_l35_35925

theorem smallest_positive_multiple (a : ℕ) (k : ℕ) (h : 17 * a ≡ 7 [MOD 101]) : 
  ∃ k, k = 17 * 42 := 
sorry

end smallest_positive_multiple_l35_35925


namespace solve_for_x_l35_35667

theorem solve_for_x (x : ℝ) (h : 2 * x - 3 * x + 4 * x = 150) : x = 50 :=
by
  sorry

end solve_for_x_l35_35667


namespace net_change_of_Toronto_Stock_Exchange_l35_35134

theorem net_change_of_Toronto_Stock_Exchange :
  let monday := -150
  let tuesday := 106
  let wednesday := -47
  let thursday := 182
  let friday := -210
  (monday + tuesday + wednesday + thursday + friday) = -119 :=
by
  let monday := -150
  let tuesday := 106
  let wednesday := -47
  let thursday := 182
  let friday := -210
  have h : (monday + tuesday + wednesday + thursday + friday) = -119 := sorry
  exact h

end net_change_of_Toronto_Stock_Exchange_l35_35134


namespace foci_on_x_axis_l35_35440

theorem foci_on_x_axis (k : ℝ) : (∃ a b : ℝ, ∀ x y : ℝ, (x^2)/(3 - k) + (y^2)/(1 + k) = 1) ↔ -1 < k ∧ k < 1 :=
by
  sorry

end foci_on_x_axis_l35_35440


namespace total_high_sulfur_samples_l35_35251

-- Define the conditions as given in the problem
def total_samples : ℕ := 143
def heavy_oil_freq : ℚ := 2 / 11
def light_low_sulfur_freq : ℚ := 7 / 13
def no_low_sulfur_in_heavy_oil : Prop := ∀ (x : ℕ), (x / total_samples = heavy_oil_freq) → false

-- Define total high-sulfur samples
def num_heavy_oil := heavy_oil_freq * total_samples
def num_light_oil := total_samples - num_heavy_oil
def num_light_low_sulfur_oil := light_low_sulfur_freq * num_light_oil
def num_light_high_sulfur_oil := num_light_oil - num_light_low_sulfur_oil

-- Now state that we need to prove the total number of high-sulfur samples
theorem total_high_sulfur_samples : num_light_high_sulfur_oil + num_heavy_oil = 80 :=
by
  sorry

end total_high_sulfur_samples_l35_35251


namespace real_root_quadratic_l35_35649

theorem real_root_quadratic (b : ℝ) : 
  (∃ x : ℝ, x^2 + b * x + 9 = 0) ↔ b ≤ -6 ∨ b ≥ 6 := 
sorry

end real_root_quadratic_l35_35649


namespace log5_of_15625_l35_35696

-- Define the logarithm function in base 5
def log_base_5 (n : ℕ) : ℕ := sorry

-- State the theorem with the given condition and conclude the desired result
theorem log5_of_15625 : log_base_5 15625 = 6 :=
by sorry

end log5_of_15625_l35_35696


namespace smallest_integral_k_no_real_roots_l35_35004

theorem smallest_integral_k_no_real_roots :
  ∃ k : ℤ, (∀ x : ℝ, 2 * x * (k * x - 4) - x^2 + 6 ≠ 0) ∧ 
           (∀ j : ℤ, j < k → (∃ x : ℝ, 2 * x * (j * x - 4) - x^2 + 6 = 0)) ∧
           k = 2 :=
by sorry

end smallest_integral_k_no_real_roots_l35_35004


namespace plane_equation_l35_35433

-- We will create a structure for 3D points to use in our problem
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the problem conditions and the equation we want to prove
def containsPoint (p: Point3D) : Prop := p.x = 1 ∧ p.y = 4 ∧ p.z = -8

def onLine (p: Point3D) : Prop := 
  ∃ t : ℝ, 
    (p.x = 4 * t + 2) ∧ 
    (p.y = - t - 1) ∧ 
    (p.z = 5 * t + 3)

def planeEq (p: Point3D) : Prop := 
  -4 * p.x + 2 * p.y - 5 * p.z + 3 = 0

-- Now state the theorem
theorem plane_equation (p: Point3D) : 
  containsPoint p ∨ onLine p → planeEq p := 
  sorry

end plane_equation_l35_35433


namespace one_over_m_add_one_over_n_l35_35100

theorem one_over_m_add_one_over_n (m n : ℕ) (h_sum : m + n = 80) (h_hcf : Nat.gcd m n = 6) (h_lcm : Nat.lcm m n = 210) : 
  1 / (m:ℚ) + 1 / (n:ℚ) = 1 / 15.75 :=
by
  sorry

end one_over_m_add_one_over_n_l35_35100


namespace primes_between_4900_8100_l35_35094

theorem primes_between_4900_8100 :
  ∃ (count : ℕ),
  count = 5 ∧ ∀ n : ℤ, 70 < n ∧ n < 90 ∧ (n * n > 4900 ∧ n * n < 8100 ∧ Prime n) → count = 5 :=
by
  sorry

end primes_between_4900_8100_l35_35094


namespace find_a_of_odd_function_l35_35215

noncomputable def f (a : ℝ) (x : ℝ) := 1 + a / (2^x + 1)

theorem find_a_of_odd_function (a : ℝ) (h : ∀ x : ℝ, f a x = -f a (-x)) : a = -2 :=
by
  sorry

end find_a_of_odd_function_l35_35215


namespace cistern_fill_time_l35_35166

-- Define the filling rate and emptying rate as given conditions.
def R_fill : ℚ := 1 / 5
def R_empty : ℚ := 1 / 9

-- Define the net rate when both taps are opened simultaneously.
def R_net : ℚ := R_fill - R_empty

-- The total time to fill the cistern when both taps are opened.
def fill_time := 1 / R_net

-- Prove that the total time to fill the cistern is 11.25 hours.
theorem cistern_fill_time : fill_time = 11.25 := 
by 
    -- We include sorry to bypass the actual proof. This will allow the code to compile.
    sorry

end cistern_fill_time_l35_35166


namespace cost_of_chlorine_l35_35754

/--
Gary has a pool that is 10 feet long, 8 feet wide, and 6 feet deep.
He needs to buy one quart of chlorine for every 120 cubic feet of water.
Chlorine costs $3 per quart.
Prove that the total cost of chlorine Gary spends is $12.
-/
theorem cost_of_chlorine:
  let length := 10
  let width := 8
  let depth := 6
  let volume := length * width * depth
  let chlorine_per_cubic_feet := 1 / 120
  let chlorine_needed := volume * chlorine_per_cubic_feet
  let cost_per_quart := 3
  let total_cost := chlorine_needed * cost_per_quart
  total_cost = 12 :=
by
  sorry

end cost_of_chlorine_l35_35754


namespace perfect_number_mod_9_l35_35892

theorem perfect_number_mod_9 (N : ℕ) (hN : ∃ p, N = 2^(p-1) * (2^p - 1) ∧ Nat.Prime (2^p - 1)) (hN_ne_6 : N ≠ 6) : ∃ n : ℕ, N = 9 * n + 1 :=
by
  sorry

end perfect_number_mod_9_l35_35892


namespace largest_unrepresentable_n_l35_35986

theorem largest_unrepresentable_n (a b : ℕ) (ha : 1 < a) (hb : 1 < b) : ∃ n, ¬ ∃ x y : ℕ, n = 7 * a + 5 * b ∧ n = 47 :=
  sorry

end largest_unrepresentable_n_l35_35986


namespace fox_can_eat_80_fox_cannot_eat_65_l35_35710
-- import the required library

-- Define the conditions for the problem.
def total_candies := 100
def piles := 3
def fox_eat_equalize (fox: ℕ) (pile1: ℕ) (pile2: ℕ): ℕ :=
  if pile1 = pile2 then fox + pile1 else fox + pile2 - pile1

-- Statement for part (a)
theorem fox_can_eat_80: ∃ c₁ c₂ c₃: ℕ, (c₁ + c₂ + c₃ = total_candies) ∧ 
  (∃ x: ℕ, (fox_eat_equalize (c₁ + c₂ + c₃ - x) c₁ c₂ = 80) ∨ 
              (fox_eat_equalize x c₁ c₂  = 80)) :=
sorry

-- Statement for part (b)
theorem fox_cannot_eat_65: ¬ (∃ c₁ c₂ c₃: ℕ, (c₁ + c₂ + c₃ = total_candies) ∧ 
  (∃ x: ℕ, (fox_eat_equalize (c₁ + c₂ + c₃ - x) c₁ c₂ = 65) ∨ 
              (fox_eat_equalize x c₁ c₂  = 65))) :=
sorry

end fox_can_eat_80_fox_cannot_eat_65_l35_35710


namespace value_of_V3_l35_35063

def f (x : ℝ) : ℝ := 3 * x^5 + 8 * x^4 - 3 * x^3 + 5 * x^2 + 12 * x - 6

def horner (a : ℝ) : ℝ :=
  let V0 := 3
  let V1 := V0 * a + 8
  let V2 := V1 * a - 3
  let V3 := V2 * a + 5
  V3

theorem value_of_V3 : horner 2 = 55 :=
  by
    simp [horner]
    sorry

end value_of_V3_l35_35063


namespace evaluate_expression_l35_35728

theorem evaluate_expression : -(16 / 4 * 7 - 50 + 5 * 7) = -13 :=
by
  sorry

end evaluate_expression_l35_35728


namespace percentage_increase_l35_35309

theorem percentage_increase (total_capacity : ℝ) (additional_water : ℝ) (percentage_capacity : ℝ) (current_water : ℝ) : 
    additional_water + current_water = percentage_capacity * total_capacity →
    percentage_capacity = 0.70 →
    total_capacity = 1857.1428571428573 →
    additional_water = 300 →
    current_water = ((percentage_capacity * total_capacity) - additional_water) →
    (additional_water / current_water) * 100 = 30 :=
by
    sorry

end percentage_increase_l35_35309


namespace dot_product_is_one_l35_35375

def vec_a : ℝ × ℝ := (1, 1)
def vec_b : ℝ × ℝ := (-1, 2)

noncomputable def dot_product (v1 v2 : ℝ × ℝ) : ℝ := (v1.1 * v2.1) + (v1.2 * v2.2)

theorem dot_product_is_one : dot_product vec_a vec_b = 1 :=
by sorry

end dot_product_is_one_l35_35375


namespace mike_total_spending_is_497_50_l35_35176

def rose_bush_price : ℝ := 75
def rose_bush_count : ℕ := 6
def rose_bush_discount : ℝ := 0.10
def friend_rose_bushes : ℕ := 2
def tax_rose_bushes : ℝ := 0.05

def aloe_price : ℝ := 100
def aloe_count : ℕ := 2
def tax_aloe : ℝ := 0.07

def calculate_total_cost_for_mike : ℝ :=
  let total_rose_bush_cost := rose_bush_price * rose_bush_count
  let discount := total_rose_bush_cost * rose_bush_discount
  let cost_after_discount := total_rose_bush_cost - discount
  let sales_tax_rose_bushes := tax_rose_bushes * cost_after_discount
  let cost_rose_bushes_after_tax := cost_after_discount + sales_tax_rose_bushes

  let total_aloe_cost := aloe_price * aloe_count
  let sales_tax_aloe := tax_aloe * total_aloe_cost

  let total_cost_friend_rose_bushes := friend_rose_bushes * (rose_bush_price - (rose_bush_price * rose_bush_discount))
  let sales_tax_friend_rose_bushes := tax_rose_bushes * total_cost_friend_rose_bushes
  let total_cost_friend := total_cost_friend_rose_bushes + sales_tax_friend_rose_bushes

  let total_mike_rose_bushes := cost_rose_bushes_after_tax - total_cost_friend

  let total_cost_mike_aloe := total_aloe_cost + sales_tax_aloe

  total_mike_rose_bushes + total_cost_mike_aloe

theorem mike_total_spending_is_497_50 : calculate_total_cost_for_mike = 497.50 := by
  sorry

end mike_total_spending_is_497_50_l35_35176


namespace complement_intersection_l35_35878

open Set

variable (U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8})
variable (A : Set ℕ := {2, 5, 8})
variable (B : Set ℕ := {1, 3, 5, 7})

theorem complement_intersection (CUA : Set ℕ := {1, 3, 4, 6, 7}) :
  (CUA ∩ B) = {1, 3, 7} := by
  sorry

end complement_intersection_l35_35878


namespace correct_option_B_l35_35732

theorem correct_option_B (x : ℝ) : (1 - x)^2 = 1 - 2 * x + x^2 :=
sorry

end correct_option_B_l35_35732


namespace smallest_number_groups_l35_35074

theorem smallest_number_groups :
  ∃ x : ℕ, (∀ y : ℕ, (y % 12 = 0 ∧ y % 20 = 0 ∧ y % 6 = 0) → y ≥ x) ∧ 
           (x % 12 = 0 ∧ x % 20 = 0 ∧ x % 6 = 0) ∧ x = 60 :=
by
  sorry

end smallest_number_groups_l35_35074


namespace inverse_prop_l35_35296

theorem inverse_prop (x : ℝ) : x < 0 → x^2 > 0 :=
by
  sorry

end inverse_prop_l35_35296


namespace number_of_boys_l35_35467

-- Definitions from the problem conditions
def trees : ℕ := 29
def trees_per_boy : ℕ := 3

-- Prove the number of boys is 10
theorem number_of_boys : (trees / trees_per_boy) + 1 = 10 :=
by sorry

end number_of_boys_l35_35467


namespace domain_of_f_l35_35356

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt (x - 3)) / (abs (x + 1) - 5)

theorem domain_of_f :
  {x : ℝ | x - 3 ≥ 0 ∧ abs (x + 1) - 5 ≠ 0} = {x : ℝ | (3 ≤ x ∧ x < 4) ∨ (4 < x)} :=
by
  sorry

end domain_of_f_l35_35356


namespace product_gcd_lcm_eq_1296_l35_35781

theorem product_gcd_lcm_eq_1296 : (Int.gcd 24 54) * (Int.lcm 24 54) = 1296 := by
  sorry

end product_gcd_lcm_eq_1296_l35_35781


namespace compute_product_fraction_l35_35770

theorem compute_product_fraction :
  ( ((3 : ℚ)^4 - 1) / ((3 : ℚ)^4 + 1) *
    ((4 : ℚ)^4 - 1) / ((4 : ℚ)^4 + 1) * 
    ((5 : ℚ)^4 - 1) / ((5 : ℚ)^4 + 1) *
    ((6 : ℚ)^4 - 1) / ((6 : ℚ)^4 + 1) *
    ((7 : ℚ)^4 - 1) / ((7 : ℚ)^4 + 1)
  ) = (25 / 210) := 
  sorry

end compute_product_fraction_l35_35770


namespace isosceles_triangle_perimeter_l35_35730

theorem isosceles_triangle_perimeter {a b : ℕ} (h₁ : a = 4) (h₂ : b = 9) (h₃ : ∀ x y z : ℕ, 
  (x = a ∧ y = a ∧ z = b) ∨ (x = b ∧ y = b ∧ z = a) → 
  (x + y > z ∧ x + z > y ∧ y + z > x)) : 
  (a = 4 ∧ b = 9) → a + a + b = 22 :=
by sorry

end isosceles_triangle_perimeter_l35_35730


namespace kim_total_water_drank_l35_35275

noncomputable def total_water_kim_drank : Float :=
  let water_from_bottle := 1.5 * 32
  let water_from_can := 12
  let shared_bottle := (3 / 5) * 32
  water_from_bottle + water_from_can + shared_bottle

theorem kim_total_water_drank :
  total_water_kim_drank = 79.2 :=
by
  -- Proof skipped
  sorry

end kim_total_water_drank_l35_35275


namespace sin_of_angle_l35_35723

theorem sin_of_angle (α : ℝ) (x y : ℝ) (h1 : x = -3) (h2 : y = -4) (r : ℝ) (hr : r = Real.sqrt (x^2 + y^2)) : 
  Real.sin α = -4 / r := 
by
  -- Definitions
  let y := -4
  let x := -3
  let r := Real.sqrt (x^2 + y^2)
  -- Proof
  sorry

end sin_of_angle_l35_35723


namespace tan_alpha_minus_pi_six_l35_35250

variable (α β : Real)

axiom tan_alpha_minus_beta : Real.tan (α - β) = 2 / 3
axiom tan_pi_six_minus_beta : Real.tan ((Real.pi / 6) - β) = 1 / 2

theorem tan_alpha_minus_pi_six : Real.tan (α - (Real.pi / 6)) = 1 / 8 :=
by
  sorry

end tan_alpha_minus_pi_six_l35_35250


namespace four_digit_number_conditions_l35_35362

-- Define the needed values based on the problem conditions
def first_digit := 1
def second_digit := 3
def third_digit := 4
def last_digit := 9

def number := 1349

-- State the theorem
theorem four_digit_number_conditions :
  (second_digit = 3 * first_digit) ∧ 
  (last_digit = 3 * second_digit) ∧ 
  (number = 1349) :=
by
  -- This is where the proof would go
  sorry

end four_digit_number_conditions_l35_35362


namespace chip_credit_card_balance_l35_35924

-- Conditions
def initial_balance : Float := 50.00
def first_interest_rate : Float := 0.20
def additional_charge : Float := 20.00
def second_interest_rate : Float := 0.20

-- Question
def current_balance : Float :=
  let first_interest_fee := initial_balance * first_interest_rate
  let balance_after_first_interest := initial_balance + first_interest_fee
  let balance_before_second_interest := balance_after_first_interest + additional_charge
  let second_interest_fee := balance_before_second_interest * second_interest_rate
  balance_before_second_interest + second_interest_fee

-- Correct Answer
def expected_balance : Float := 96.00

-- Proof Problem Statement
theorem chip_credit_card_balance : current_balance = expected_balance := by
  sorry

end chip_credit_card_balance_l35_35924


namespace simplify_expression_l35_35315

theorem simplify_expression : 3000 * 3000^3000 = 3000^(3001) := 
by 
  sorry

end simplify_expression_l35_35315


namespace age_difference_l35_35687

theorem age_difference (A B C : ℕ) (h : A + B = B + C + 13) : A = C + 13 :=
by
  sorry

end age_difference_l35_35687


namespace math_problem_solution_l35_35261

noncomputable def math_problem (a b c d : ℝ) (h1 : a^2 + b^2 - c^2 - d^2 = 0) (h2 : a^2 - b^2 - c^2 + d^2 = (56 / 53) * (b * c + a * d)) : ℝ :=
  (a * b + c * d) / (b * c + a * d)

theorem math_problem_solution (a b c d : ℝ) (h1 : a^2 + b^2 - c^2 - d^2 = 0) (h2 : a^2 - b^2 - c^2 + d^2 = (56 / 53) * (b * c + a * d)) :
  math_problem a b c d h1 h2 = 45 / 53 := sorry

end math_problem_solution_l35_35261


namespace exists_infinitely_many_primes_dividing_form_l35_35916

theorem exists_infinitely_many_primes_dividing_form (a : ℕ) (ha : 0 < a) :
  ∃ᶠ p in Filter.atTop, ∃ n : ℕ, p ∣ 2^(2*n) + a := 
sorry

end exists_infinitely_many_primes_dividing_form_l35_35916


namespace exists_intersecting_line_l35_35701

/-- Represents a segment as a pair of endpoints in a 2D plane. -/
structure Segment where
  x : ℝ
  y1 : ℝ
  y2 : ℝ

open Segment

/-- Given several parallel segments with the property that for any three of these segments, 
there exists a line that intersects all three of them, prove that 
there is a line that intersects all the segments. -/
theorem exists_intersecting_line (segments : List Segment)
  (h : ∀ s1 s2 s3 : Segment, s1 ∈ segments → s2 ∈ segments → s3 ∈ segments → 
       ∃ a b : ℝ, (s1.y1 <= a * s1.x + b) ∧ (a * s1.x + b <= s1.y2) ∧ 
                   (s2.y1 <= a * s2.x + b) ∧ (a * s2.x + b <= s2.y2) ∧ 
                   (s3.y1 <= a * s3.x + b) ∧ (a * s3.x + b <= s3.y2)) :
  ∃ a b : ℝ, ∀ s : Segment, s ∈ segments → (s.y1 <= a * s.x + b) ∧ (a * s.x + b <= s.y2) := 
sorry

end exists_intersecting_line_l35_35701


namespace tony_pool_filling_time_l35_35780

theorem tony_pool_filling_time
  (J S T : ℝ)
  (hJ : J = 1 / 30)
  (hS : S = 1 / 45)
  (hCombined : J + S + T = 1 / 15) :
  T = 1 / 90 :=
by
  -- the setup for proof would be here
  sorry

end tony_pool_filling_time_l35_35780


namespace number_of_seats_in_nth_row_l35_35111

theorem number_of_seats_in_nth_row (n : ℕ) :
    ∃ m : ℕ, m = 3 * n + 15 :=
by
  sorry

end number_of_seats_in_nth_row_l35_35111


namespace parallel_lines_slope_condition_l35_35877

theorem parallel_lines_slope_condition (b : ℝ) (x y : ℝ) :
    (∀ (x y : ℝ), 3 * y - 3 * b = 9 * x) →
    (∀ (x y : ℝ), y + 2 = (b + 9) * x) →
    b = -6 :=
by
    sorry

end parallel_lines_slope_condition_l35_35877


namespace suff_not_necc_condition_l35_35212

theorem suff_not_necc_condition (x : ℝ) : (x=2) → ((x-2) * (x+5) = 0) ∧ ¬((x-2) * (x+5) = 0 → x=2) :=
by {
  sorry
}

end suff_not_necc_condition_l35_35212


namespace part1_part2_l35_35014

def f (x : ℝ) : ℝ := |2 * x + 1| - |x - 1|

theorem part1 : {x : ℝ | f x < 2} = {x : ℝ | -4 < x ∧ x < 2 / 3} :=
by
  sorry

theorem part2 : ∀ a : ℝ, (∃ x : ℝ, f x ≤ a - a^2 / 2) → (-1 ≤ a ∧ a ≤ 3) :=
by
  sorry

end part1_part2_l35_35014


namespace Cody_age_is_14_l35_35053

variable (CodyGrandmotherAge CodyAge : ℕ)

theorem Cody_age_is_14 (h1 : CodyGrandmotherAge = 6 * CodyAge) (h2 : CodyGrandmotherAge = 84) : CodyAge = 14 := by
  sorry

end Cody_age_is_14_l35_35053


namespace max_num_triangles_for_right_triangle_l35_35834

-- Define a right triangle on graph paper
def right_triangle (n : ℕ) : Prop :=
  ∀ (a b : ℕ), 0 ≤ a ∧ a ≤ n ∧ 0 ≤ b ∧ b ≤ n

-- Define maximum number of triangles that can be formed within the triangle
def max_triangles (n : ℕ) : ℕ :=
  if h : n = 7 then 28 else 0  -- Given n = 7, the max number is 28

-- Define the theorem to be proven
theorem max_num_triangles_for_right_triangle :
  right_triangle 7 → max_triangles 7 = 28 :=
by
  intro h
  -- Proof goes here
  sorry

end max_num_triangles_for_right_triangle_l35_35834


namespace awards_distribution_l35_35028

theorem awards_distribution :
  let num_awards := 6
  let num_students := 3 
  let min_awards_per_student := 2
  (num_awards = 6 ∧ num_students = 3 ∧ min_awards_per_student = 2) →
  ∃ (ways : ℕ), ways = 15 :=
by
  sorry

end awards_distribution_l35_35028


namespace range_of_ω_l35_35129

noncomputable def f (x : ℝ) (ω : ℝ) (ϕ : ℝ) : ℝ :=
  Real.cos (ω * x + ϕ)

theorem range_of_ω :
  ∀ (ω : ℝ) (ϕ : ℝ),
    (0 < ω) →
    (-π ≤ ϕ) →
    (ϕ ≤ 0) →
    (∀ x, f x ω ϕ = -f (-x) ω ϕ) →
    (∀ x1 x2, (x1 < x2) → (-π/4 ≤ x1 ∧ x1 ≤ 3*π/16) ∧ (-π/4 ≤ x2 ∧ x2 ≤ 3*π/16) → f x1 ω ϕ ≤ f x2 ω ϕ) →
    (0 < ω ∧ ω ≤ 2) :=
by
  sorry

end range_of_ω_l35_35129


namespace points_on_opposite_sides_l35_35602

theorem points_on_opposite_sides (a : ℝ) :
  (3 * 3 - 2 * 1 + a) * (3 * (-4) - 2 * 6 + a) < 0 ↔ -7 < a ∧ a < 24 :=
by sorry

end points_on_opposite_sides_l35_35602


namespace chandler_tickets_total_cost_l35_35085

theorem chandler_tickets_total_cost :
  let movie_ticket_cost := 30
  let num_movie_tickets := 8
  let num_football_tickets := 5
  let num_concert_tickets := 3
  let num_theater_tickets := 4
  let theater_ticket_cost := 40
  let discount := 0.10
  let total_movie_cost := num_movie_tickets * movie_ticket_cost
  let football_ticket_cost := total_movie_cost / 2
  let total_football_cost := num_football_tickets * football_ticket_cost
  let concert_ticket_cost := football_ticket_cost - 10
  let total_concert_cost := num_concert_tickets * concert_ticket_cost
  let discounted_theater_ticket_cost := theater_ticket_cost * (1 - discount)
  let total_theater_cost := num_theater_tickets * discounted_theater_ticket_cost
  let total_cost := total_movie_cost + total_football_cost + total_concert_cost + total_theater_cost
  total_cost = 1314 := by
  sorry

end chandler_tickets_total_cost_l35_35085


namespace twelve_div_one_fourth_eq_48_l35_35269

theorem twelve_div_one_fourth_eq_48 : 12 / (1 / 4) = 48 := by
  -- We know that dividing by a fraction is equivalent to multiplying by its reciprocal
  sorry

end twelve_div_one_fourth_eq_48_l35_35269


namespace trajectory_of_moving_circle_l35_35355

-- Define the two given circles C1 and C2
def C1 : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 + 2)^2 + p.2^2 = 1}
def C2 : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 2)^2 + p.2^2 = 81}

-- Define a moving circle P with center P_center and radius r
structure Circle (α : Type) := 
(center : α × α) 
(radius : ℝ)

def isExternallyTangentTo (P : Circle ℝ) (C : Set (ℝ × ℝ)) :=
  ∃ k ∈ C, (P.center.1 - k.1)^2 + (P.center.2 - k.2)^2 = (P.radius + 1)^2

def isInternallyTangentTo (P : Circle ℝ) (C : Set (ℝ × ℝ)) :=
  ∃ k ∈ C, (P.center.1 - k.1)^2 + (P.center.2 - k.2)^2 = (9 - P.radius)^2

-- Formulate the problem statement
theorem trajectory_of_moving_circle :
  ∀ P : Circle ℝ, 
  isExternallyTangentTo P C1 → 
  isInternallyTangentTo P C2 → 
  (P.center.1^2 / 25 + P.center.2^2 / 21 = 1) := 
sorry

end trajectory_of_moving_circle_l35_35355


namespace gcd_lcm_sum_l35_35071

-- Define the necessary components: \( A \) as the greatest common factor and \( B \) as the least common multiple of 16, 32, and 48
def A := Int.gcd (Int.gcd 16 32) 48
def B := Int.lcm (Int.lcm 16 32) 48

-- Statement that needs to be proved
theorem gcd_lcm_sum : A + B = 112 := by
  sorry

end gcd_lcm_sum_l35_35071


namespace least_integer_greater_than_sqrt_500_l35_35665

theorem least_integer_greater_than_sqrt_500 : ∃ n : ℤ, n = 23 ∧ ∀ m : ℤ, (m ≤ 23 → m^2 ≤ 500) → (m < 23 ∧ m > 0 → (m + 1)^2 > 500) :=
by
  sorry

end least_integer_greater_than_sqrt_500_l35_35665


namespace problem_solution_inf_problem_solution_prime_l35_35608

-- Definitions based on the given conditions and problem statement
def is_solution_inf (m : ℕ) : Prop := 3^m ∣ 2^(3^m) + 1

def is_solution_prime (n : ℕ) : Prop := n.Prime ∧ n ∣ 2^n + 1

-- Lean statement for the math proof problem
theorem problem_solution_inf : ∀ m : ℕ, m ≥ 0 → is_solution_inf m := sorry

theorem problem_solution_prime : ∀ n : ℕ, n.Prime → is_solution_prime n → n = 3 := sorry

end problem_solution_inf_problem_solution_prime_l35_35608


namespace arithmetic_difference_l35_35993

variables (p q r : ℝ)

theorem arithmetic_difference (h1 : (p + q) / 2 = 10) (h2 : (q + r) / 2 = 27) : r - p = 34 :=
by
  sorry

end arithmetic_difference_l35_35993


namespace count_congruent_to_5_mod_7_l35_35196

theorem count_congruent_to_5_mod_7 (n : ℕ) :
  (∀ x : ℕ, 1 ≤ x ∧ x ≤ 300 ∧ x % 7 = 5) → ∃ count : ℕ, count = 43 := by
  sorry

end count_congruent_to_5_mod_7_l35_35196


namespace division_quotient_proof_l35_35186

theorem division_quotient_proof (x : ℕ) (larger_number : ℕ) (h1 : larger_number - x = 1365)
    (h2 : larger_number = 1620) (h3 : larger_number % x = 15) : larger_number / x = 6 :=
by
  sorry

end division_quotient_proof_l35_35186


namespace total_juice_sold_3_days_l35_35974

def juice_sales_problem (V_L V_M V_S : ℕ) (d1 d2 d3 : ℕ) :=
  (d1 = V_L + 4 * V_M) ∧ 
  (d2 = 2 * V_L + 6 * V_S) ∧ 
  (d3 = V_L + 3 * V_M + 3 * V_S) ∧
  (d1 = d2) ∧
  (d2 = d3)

theorem total_juice_sold_3_days (V_L V_M V_S d1 d2 d3 : ℕ) 
  (h : juice_sales_problem V_L V_M V_S d1 d2 d3) 
  (h_VM : V_M = 3) 
  (h_VL : V_L = 6) : 
  3 * d1 = 54 := 
by 
  -- Proof will be filled in
  sorry

end total_juice_sold_3_days_l35_35974


namespace solve_for_a_l35_35468

theorem solve_for_a (a : ℝ) : 
  (∀ x : ℝ, (x + 1) * (x^2 - 5 * a * x + a) = x^3 + (1 - 5 * a) * x^2 - 4 * a * x + a) →
  (1 - 5 * a = 0) →
  a = 1 / 5 := 
by
  intro h₁ h₂
  sorry

end solve_for_a_l35_35468


namespace line_x_intercept_l35_35157

theorem line_x_intercept (P Q : ℝ × ℝ) (hP : P = (2, 3)) (hQ : Q = (6, 7)) :
  ∃ x, (x, 0) = (-1, 0) ∧ ∃ (m : ℝ), m = (Q.2 - P.2) / (Q.1 - P.1) ∧ ∀ (x y : ℝ), y = m * (x - P.1) + P.2 := 
  sorry

end line_x_intercept_l35_35157


namespace tangent_line_at_origin_is_minus_3x_l35_35056

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a * x^2 + (a - 3) * x
noncomputable def f_prime (a : ℝ) (x : ℝ) : ℝ := 3 * x^2 + 2 * a * x + (a - 3)

theorem tangent_line_at_origin_is_minus_3x (a : ℝ) (h : ∀ x : ℝ, f_prime a x = f_prime a (-x)) : 
  (f_prime 0 0 = -3) → ∀ x : ℝ, (f a x = -3 * x) :=
by
  sorry

end tangent_line_at_origin_is_minus_3x_l35_35056


namespace laura_park_time_l35_35853

theorem laura_park_time
  (T : ℝ) -- Time spent at the park each trip in hours
  (walk_time : ℝ := 0.5) -- Time spent walking to and from the park each trip in hours
  (trips : ℕ := 6) -- Total number of trips
  (park_time_percentage : ℝ := 0.80) -- Percentage of total time spent at the park
  (total_park_time_eq : trips * T = park_time_percentage * (trips * (T + walk_time))) :
  T = 2 :=
by
  sorry

end laura_park_time_l35_35853


namespace marys_mother_bought_3_pounds_of_beef_l35_35200

-- Define the variables and constants
def total_paid : ℝ := 16
def cost_of_chicken : ℝ := 2 * 1  -- 2 pounds of chicken
def cost_per_pound_beef : ℝ := 4
def cost_of_oil : ℝ := 1
def shares : ℝ := 3  -- Mary and her two friends

theorem marys_mother_bought_3_pounds_of_beef:
  total_paid - (cost_of_chicken / shares) - cost_of_oil = 3 * cost_per_pound_beef :=
by
  -- the proof goes here
  sorry

end marys_mother_bought_3_pounds_of_beef_l35_35200


namespace eq_x_in_terms_of_y_l35_35614

theorem eq_x_in_terms_of_y (x y : ℝ) (h : 2 * x + y = 5) : x = (5 - y) / 2 := by
  sorry

end eq_x_in_terms_of_y_l35_35614


namespace exposed_surface_area_hemisphere_l35_35140

-- Given conditions
def radius : ℝ := 10
def height_above_liquid : ℝ := 5

-- The attempt to state the problem as a proposition
theorem exposed_surface_area_hemisphere : 
  (π * radius ^ 2) + (π * radius * height_above_liquid) = 200 * π :=
by
  sorry

end exposed_surface_area_hemisphere_l35_35140


namespace different_colors_probability_l35_35902

noncomputable def differentColorProbability : ℚ :=
  let redChips := 7
  let greenChips := 5
  let totalChips := redChips + greenChips
  let probRedThenGreen := (redChips / totalChips) * (greenChips / totalChips)
  let probGreenThenRed := (greenChips / totalChips) * (redChips / totalChips)
  (probRedThenGreen + probGreenThenRed)

theorem different_colors_probability :
  differentColorProbability = 35 / 72 :=
by sorry

end different_colors_probability_l35_35902


namespace geometric_series_sum_y_equals_nine_l35_35976

theorem geometric_series_sum_y_equals_nine : 
  (∑' n : ℕ, (1 / 3) ^ n) * (∑' n : ℕ, (-1 / 3) ^ n) = ∑' n : ℕ, (1 / (9 ^ n)) :=
by
  sorry

end geometric_series_sum_y_equals_nine_l35_35976


namespace problem_statement_l35_35900

noncomputable def a : ℝ := 2 * Real.log 0.99
noncomputable def b : ℝ := Real.log 0.98
noncomputable def c : ℝ := Real.sqrt 0.96 - 1

theorem problem_statement : a > b ∧ b > c := by
  sorry

end problem_statement_l35_35900


namespace product_units_digit_mod_10_l35_35441

theorem product_units_digit_mod_10
  (u1 u2 u3 : ℕ)
  (hu1 : u1 = 2583 % 10)
  (hu2 : u2 = 7462 % 10)
  (hu3 : u3 = 93215 % 10) :
  ((2583 * 7462 * 93215) % 10) = 0 :=
by
  have h_units1 : u1 = 3 := by sorry
  have h_units2 : u2 = 2 := by sorry
  have h_units3 : u3 = 5 := by sorry
  have h_produce_units : ((3 * 2 * 5) % 10) = 0 := by sorry
  exact h_produce_units

end product_units_digit_mod_10_l35_35441


namespace initial_volume_of_mixture_l35_35741

theorem initial_volume_of_mixture 
  (V : ℝ)
  (h1 : 0 < V) 
  (h2 : 0.20 * V = 0.15 * (V + 5)) :
  V = 15 :=
by 
  -- proof steps 
  sorry

end initial_volume_of_mixture_l35_35741


namespace calories_in_200_grams_is_137_l35_35178

-- Define the grams of ingredients used.
def lemon_juice_grams := 100
def sugar_grams := 100
def water_grams := 400

-- Define the calories per 100 grams of each ingredient.
def lemon_juice_calories_per_100_grams := 25
def sugar_calories_per_100_grams := 386
def water_calories_per_100_grams := 0

-- Calculate the total calories in the entire lemonade mixture.
def total_calories : Nat :=
  (lemon_juice_grams * lemon_juice_calories_per_100_grams / 100) + 
  (sugar_grams * sugar_calories_per_100_grams / 100) +
  (water_grams * water_calories_per_100_grams / 100)

-- Calculate the total weight of the lemonade mixture.
def total_weight : Nat := lemon_juice_grams + sugar_grams + water_grams

-- Calculate the caloric density (calories per gram).
def caloric_density := total_calories / total_weight

-- Calculate the calories in 200 grams of lemonade.
def calories_in_200_grams := (caloric_density * 200)

-- The theorem to prove
theorem calories_in_200_grams_is_137 : calories_in_200_grams = 137 :=
by sorry

end calories_in_200_grams_is_137_l35_35178


namespace area_of_pentagon_eq_fraction_l35_35248

theorem area_of_pentagon_eq_fraction (w : ℝ) (h : ℝ) (fold_x : ℝ) (fold_y : ℝ)
    (hw3 : h = 3 * w)
    (hfold : fold_x = fold_y)
    (hx : fold_x ^ 2 + fold_y ^ 2 = 3 ^ 2)
    (hx_dist : fold_x = 4 / 3) :
  (3 * (1 / 2) + fold_x / 2) / (3 * w) = 13 / 18 := 
by 
  sorry

end area_of_pentagon_eq_fraction_l35_35248


namespace min_value_expr_l35_35162

theorem min_value_expr (m n : ℝ) (h : m - n^2 = 8) : m^2 - 3 * n^2 + m - 14 ≥ 58 :=
sorry

end min_value_expr_l35_35162


namespace feed_mixture_hay_calculation_l35_35065

theorem feed_mixture_hay_calculation
  (hay_Stepan_percent oats_Pavel_percent corn_mixture_percent : ℝ)
  (hay_Stepan_mass_Stepan hay_Pavel_mass_Pavel total_mixture_mass : ℝ):
  hay_Stepan_percent = 0.4 ∧
  oats_Pavel_percent = 0.26 ∧
  (∃ (x : ℝ), 
  x > 0 ∧ 
  hay_Pavel_percent =  0.74 - x ∧ 
  0.15 * x + 0.25 * x = 0.3 * total_mixture_mass ∧
  hay_Stepan_mass_Stepan = 0.40 * 150 ∧
  hay_Pavel_mass_Pavel = (0.74 - x) * 250 ∧ 
  total_mixture_mass = 150 + 250) → 
  hay_Stepan_mass_Stepan + hay_Pavel_mass_Pavel = 170 := 
by
  intro h
  obtain ⟨h1, h2, ⟨x, hx1, hx2, hx3, hx4, hx5, hx6⟩⟩ := h
  /- proof -/
  sorry

end feed_mixture_hay_calculation_l35_35065


namespace min_value_expression_l35_35187

variable (a b m n : ℝ)

-- Conditions: a, b, m, n are positive, a + b = 1, mn = 2
def conditions (a b m n : ℝ) : Prop := 
  0 < a ∧ 0 < b ∧ 0 < m ∧ 0 < n ∧ a + b = 1 ∧ m * n = 2

-- Statement to prove: The minimum value of (am + bn) * (bm + an) is 2
theorem min_value_expression (a b m n : ℝ) (h : conditions a b m n) : 
  ∃ c : ℝ, c = 2 ∧ (∀ (x y z w : ℝ), conditions x y z w → (x * z + y * w) * (y * z + x * w) ≥ c) :=
by
  sorry

end min_value_expression_l35_35187


namespace alia_markers_l35_35058

theorem alia_markers (S A a : ℕ) (h1 : S = 60) (h2 : A = S / 3) (h3 : a = 2 * A) : a = 40 :=
by
  -- Proof omitted
  sorry

end alia_markers_l35_35058


namespace sum_and_product_of_roots_l35_35752

theorem sum_and_product_of_roots (m p : ℝ) 
    (h₁ : ∀ α β : ℝ, (3 * α^2 - m * α + p = 0 ∧ 3 * β^2 - m * β + p = 0) → α + β = 9)
    (h₂ : ∀ α β : ℝ, (3 * α^2 - m * α + p = 0 ∧ 3 * β^2 - m * β + p = 0) → α * β = 14) :
    m + p = 69 := 
sorry

end sum_and_product_of_roots_l35_35752


namespace winning_percentage_l35_35690

noncomputable def total_votes (votes_winner votes_margin : ℕ) : ℕ :=
  votes_winner + (votes_winner - votes_margin)

noncomputable def percentage_votes (votes_winner total_votes : ℕ) : ℝ :=
  (votes_winner : ℝ) / (total_votes : ℝ) * 100

theorem winning_percentage
  (votes_winner : ℕ)
  (votes_margin : ℕ)
  (h_winner : votes_winner = 775)
  (h_margin : votes_margin = 300) :
  percentage_votes votes_winner (total_votes votes_winner votes_margin) = 62 :=
sorry

end winning_percentage_l35_35690


namespace number_of_pens_sold_l35_35784

variables (C N : ℝ) (gain_percentage : ℝ) (gain : ℝ)

-- Defining conditions given in the problem
def trader_gain_cost_pens (C N : ℝ) : ℝ := 30 * C
def gain_percentage_condition (gain_percentage : ℝ) : Prop := gain_percentage = 0.30
def gain_condition (C N : ℝ) : Prop := (0.30 * N * C) = 30 * C

-- Defining the theorem to prove
theorem number_of_pens_sold
  (h_gain_percentage : gain_percentage_condition gain_percentage)
  (h_gain : gain_condition C N) :
  N = 100 :=
sorry

end number_of_pens_sold_l35_35784


namespace sqrt_mul_l35_35689

theorem sqrt_mul (h₁ : 0 ≤ 3) (h₂ : 0 ≤ 12) : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_mul_l35_35689


namespace partial_fraction_sum_equals_251_l35_35851

theorem partial_fraction_sum_equals_251 (p q r A B C : ℝ) :
  (p ≠ q) ∧ (p ≠ r) ∧ (q ≠ r) ∧ 
  (A ≠ 0) ∧ (B ≠ 0) ∧ (C ≠ 0) ∧
  (∀ s : ℝ, (s ≠ p) ∧ (s ≠ q) ∧ (s ≠ r) →
  1 / (s^3 - 24*s^2 + 151*s - 650) = A / (s - p) + B / (s - q) + C / (s - r)) →
  (p + q + r = 24) →
  (p * q + p * r + q * r = 151) →
  (p * q * r = 650) →
  (1 / A + 1 / B + 1 / C = 251) :=
by
  sorry

end partial_fraction_sum_equals_251_l35_35851


namespace FC_value_l35_35451

variables (DC CB AB AD ED FC CA BD : ℝ)

-- Set the conditions as variables
variable (h_DC : DC = 10)
variable (h_CB : CB = 12)
variable (h_AB : AB = (1/3) * AD)
variable (h_ED : ED = (2/3) * AD)
variable (h_BD : BD = 22)
variable (BD_eq : BD = DC + CB)
variable (CA_eq : CA = CB + AB)

-- Define the relationship for the final result
def find_FC (DC CB AB AD ED FC CA BD : ℝ) := FC = (ED * CA) / AD

-- The main statement to be proven
theorem FC_value : 
  find_FC DC CB AB (33 : ℝ) (22 : ℝ) FC (23 : ℝ) (22 : ℝ) → 
  FC = (506/33) :=
by 
  intros h
  sorry

end FC_value_l35_35451


namespace Ksyusha_time_to_school_l35_35080

/-- Ksyusha runs twice as fast as she walks (both speeds are constant).
On Tuesday, Ksyusha walked a distance twice the distance she ran, and she reached school in 30 minutes.
On Wednesday, Ksyusha ran twice the distance she walked.
Prove that it took her 24 minutes to get from home to school on Wednesday. -/
theorem Ksyusha_time_to_school
  (S v : ℝ)                  -- S for distance unit, v for walking speed
  (htuesday : 2 * (2 * S / v) + S / (2 * v) = 30) -- Condition on Tuesday's travel
  :
  (S / v + 2 * S / (2 * v) = 24)                 -- Claim about Wednesday's travel time
:=
  sorry

end Ksyusha_time_to_school_l35_35080


namespace permutation_value_l35_35792

theorem permutation_value (n : ℕ) (h : n * (n - 1) = 12) : n = 4 :=
by
  sorry

end permutation_value_l35_35792


namespace platform_length_l35_35481

theorem platform_length (train_length : ℕ) (pole_time : ℕ) (platform_time : ℕ) (V : ℕ) (L : ℕ)
  (h_train_length : train_length = 500)
  (h_pole_time : pole_time = 50)
  (h_platform_time : platform_time = 100)
  (h_speed : V = train_length / pole_time)
  (h_platform_distance : V * platform_time = train_length + L) : 
  L = 500 := 
sorry

end platform_length_l35_35481


namespace otimes_identity_l35_35583

-- Define the operation ⊗
def otimes (k l : ℝ) : ℝ := k^2 - l^2

-- The goal is to show k ⊗ (k ⊗ k) = k^2 for any real number k
theorem otimes_identity (k : ℝ) : otimes k (otimes k k) = k^2 :=
by sorry

end otimes_identity_l35_35583


namespace num_triangles_with_perimeter_9_l35_35378

theorem num_triangles_with_perimeter_9 : 
  ∃! (S : Finset (ℕ × ℕ × ℕ)), 
  S.card = 6 ∧ 
  (∀ (a b c : ℕ), (a, b, c) ∈ S → a + b + c = 9 ∧ a + b > c ∧ b + c > a ∧ a + c > b ∧ a ≤ b ∧ b ≤ c) := 
sorry

end num_triangles_with_perimeter_9_l35_35378


namespace smallest_x_absolute_value_l35_35960

theorem smallest_x_absolute_value : ∃ x : ℤ, |x + 3| = 15 ∧ ∀ y : ℤ, |y + 3| = 15 → x ≤ y :=
sorry

end smallest_x_absolute_value_l35_35960


namespace dessert_distribution_l35_35537

theorem dessert_distribution 
  (mini_cupcakes : ℕ) 
  (donut_holes : ℕ) 
  (total_desserts : ℕ) 
  (students : ℕ) 
  (h1 : mini_cupcakes = 14)
  (h2 : donut_holes = 12) 
  (h3 : students = 13)
  (h4 : total_desserts = mini_cupcakes + donut_holes)
  : total_desserts / students = 2 :=
by sorry

end dessert_distribution_l35_35537


namespace min_sum_ab_l35_35043

theorem min_sum_ab (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_ab : a * b = a + b + 3) : 
  a + b ≥ 6 := 
sorry

end min_sum_ab_l35_35043


namespace bounded_region_area_l35_35611

theorem bounded_region_area : 
  (∀ x y : ℝ, (y^2 + 4*x*y + 50*|x| = 500) → (x ≥ 0 ∧ y = 25 - 4*x) ∨ (x ≤ 0 ∧ y = -12.5 - 4*x)) →
  ∃ (A : ℝ), A = 156.25 :=
by
  sorry

end bounded_region_area_l35_35611


namespace data_variance_l35_35755

def data : List ℝ := [9.8, 9.9, 10.1, 10, 10.2]

noncomputable def mean (data : List ℝ) : ℝ :=
  data.sum / data.length

noncomputable def variance (data : List ℝ) : ℝ :=
  (data.map (λ x => (x - mean data) ^ 2)).sum / data.length

theorem data_variance : variance data = 0.02 := by
  sorry

end data_variance_l35_35755


namespace sally_cut_red_orchids_l35_35953

-- Definitions and conditions
def initial_red_orchids := 9
def orchids_in_vase_after_cutting := 15

-- Problem statement
theorem sally_cut_red_orchids : (orchids_in_vase_after_cutting - initial_red_orchids) = 6 := by
  sorry

end sally_cut_red_orchids_l35_35953


namespace parallelogram_height_l35_35797

theorem parallelogram_height (area base height : ℝ) 
  (h_area : area = 336) 
  (h_base : base = 14) 
  (h_formula : area = base * height) : 
  height = 24 := 
by 
  sorry

end parallelogram_height_l35_35797


namespace relationship_y_values_l35_35168

theorem relationship_y_values (m y1 y2 y3 : ℝ) 
  (h1 : y1 = -3 * (-3 : ℝ)^2 - 12 * (-3 : ℝ) + m)
  (h2 : y2 = -3 * (-2 : ℝ)^2 - 12 * (-2 : ℝ) + m)
  (h3 : y3 = -3 * (1 : ℝ)^2 - 12 * (1 : ℝ) + m) :
  y2 > y1 ∧ y1 > y3 :=
sorry

end relationship_y_values_l35_35168


namespace total_students_in_class_l35_35778

/-- 
There are 208 boys in the class.
There are 69 more girls than boys.
The total number of students in the class is the sum of boys and girls.
Prove that the total number of students in the graduating class is 485.
-/
theorem total_students_in_class (boys girls : ℕ) (h1 : boys = 208) (h2 : girls = boys + 69) : 
  boys + girls = 485 :=
by
  sorry

end total_students_in_class_l35_35778


namespace range_of_a_l35_35910

noncomputable def f (x : ℝ) : ℝ := x + 1 / Real.exp x

theorem range_of_a (a : ℝ) : (∀ x : ℝ, f x > a * x) ↔ (1 - Real.exp 1 < a ∧ a ≤ 1) := by
  sorry

end range_of_a_l35_35910


namespace no_real_solution_for_inequality_l35_35435

theorem no_real_solution_for_inequality :
  ¬ ∃ a : ℝ, ∃ x : ℝ, ∀ b : ℝ, |x^2 + 4*a*x + 5*a| ≤ 3 :=
by
  sorry

end no_real_solution_for_inequality_l35_35435


namespace solve_equation_l35_35913

theorem solve_equation (x : ℝ) (h : x ≠ 2 / 3) :
  (6 * x + 2) / (3 * x^2 + 6 * x - 4) = (3 * x) / (3 * x - 2) ↔ (x = (Real.sqrt 6) / 3 ∨ x = -(Real.sqrt 6) / 3) :=
by sorry

end solve_equation_l35_35913


namespace universal_negation_example_l35_35975

theorem universal_negation_example :
  (∀ x : ℝ, x^2 - 3 * x + 1 ≤ 0) →
  (¬ (∀ x : ℝ, x^2 - 3 * x + 1 ≤ 0) = (∃ x : ℝ, x^2 - 3 * x + 1 > 0)) :=
by
  intro h
  sorry

end universal_negation_example_l35_35975


namespace part1_solution_part2_solution_l35_35895

open Real

noncomputable def f (x : ℝ) : ℝ := abs (x - 1) + abs (x - 3)

theorem part1_solution : ∀ x, f x ≤ 4 ↔ (0 ≤ x) ∧ (x ≤ 4) :=
by
  intro x
  sorry

theorem part2_solution : ∀ m, (∀ x, f x > m^2 + m) ↔ (-2 < m) ∧ (m < 1) :=
by
  intro m
  sorry

end part1_solution_part2_solution_l35_35895


namespace train_length_l35_35890

theorem train_length (L : ℝ) (V : ℝ)
  (h1 : V = L / 8)
  (h2 : V = (L + 273) / 20) :
  L = 182 :=
  by
  sorry

end train_length_l35_35890


namespace keychain_arrangement_count_l35_35312

-- Definitions of the keys
inductive Key
| house
| car
| office
| other1
| other2

-- Function to count the number of distinct arrangements on a keychain
noncomputable def distinct_keychain_arrangements : ℕ :=
  sorry -- This will be the placeholder for the proof

-- The ultimate theorem stating the solution
theorem keychain_arrangement_count : distinct_keychain_arrangements = 2 :=
  sorry -- This will be the placeholder for the proof

end keychain_arrangement_count_l35_35312


namespace jordan_rectangle_width_l35_35653

theorem jordan_rectangle_width :
  ∀ (areaC areaJ : ℕ) (lengthC widthC lengthJ widthJ : ℕ), 
    (areaC = lengthC * widthC) →
    (areaJ = lengthJ * widthJ) →
    (areaC = areaJ) →
    (lengthC = 5) →
    (widthC = 24) →
    (lengthJ = 3) →
    widthJ = 40 :=
by
  intros areaC areaJ lengthC widthC lengthJ widthJ
  intro hAreaC
  intro hAreaJ
  intro hEqualArea
  intro hLengthC
  intro hWidthC
  intro hLengthJ
  sorry

end jordan_rectangle_width_l35_35653


namespace minimum_value_w_l35_35018

theorem minimum_value_w : ∃ (x y : ℝ), ∀ w, w = 3 * x^2 + 3 * y^2 + 9 * x - 6 * y + 30 → w ≥ 20.25 :=
sorry

end minimum_value_w_l35_35018


namespace jack_received_more_emails_l35_35386

-- Definitions representing the conditions
def morning_emails : ℕ := 6
def afternoon_emails : ℕ := 8

-- The theorem statement
theorem jack_received_more_emails : afternoon_emails - morning_emails = 2 := 
by 
  sorry

end jack_received_more_emails_l35_35386


namespace scientific_notation_of_1_300_000_l35_35122

-- Define the condition: 1.3 million equals 1,300,000
def one_point_three_million : ℝ := 1300000

-- The theorem statement for the question
theorem scientific_notation_of_1_300_000 :
  one_point_three_million = 1.3 * 10^6 :=
sorry

end scientific_notation_of_1_300_000_l35_35122


namespace percentage_of_water_in_nectar_l35_35203

-- Define the necessary conditions and variables
def weight_of_nectar : ℝ := 1.7 -- kg
def weight_of_honey : ℝ := 1 -- kg
def honey_water_percentage : ℝ := 0.15 -- 15%

noncomputable def water_in_honey : ℝ := weight_of_honey * honey_water_percentage -- Water content in 1 kg of honey

noncomputable def total_water_in_nectar : ℝ := water_in_honey + (weight_of_nectar - weight_of_honey) -- Total water content in nectar

-- The theorem to prove
theorem percentage_of_water_in_nectar :
    (total_water_in_nectar / weight_of_nectar) * 100 = 50 := 
by 
    -- Skipping the proof by using sorry as it is not required
    sorry

end percentage_of_water_in_nectar_l35_35203


namespace larger_segment_length_l35_35456

open Real

theorem larger_segment_length (a b c : ℝ) (h : a = 50 ∧ b = 110 ∧ c = 120) :
  ∃ x : ℝ, x = 100 ∧ (∃ h : ℝ, a^2 = x^2 + h^2 ∧ b^2 = (c - x)^2 + h^2) :=
by
  sorry

end larger_segment_length_l35_35456


namespace smaller_base_length_trapezoid_l35_35558

variable (p q a b : ℝ)
variable (h : p < q)
variable (angle_ratio : ∃ α, ((2 * α) : ℝ) = α + (α : ℝ))

theorem smaller_base_length_trapezoid :
  b = (p^2 + a * p - q^2) / p :=
sorry

end smaller_base_length_trapezoid_l35_35558


namespace one_third_of_four_l35_35811

theorem one_third_of_four (h : 1/6 * 20 = 15) : 1/3 * 4 = 10 :=
sorry

end one_third_of_four_l35_35811


namespace mass_percentage_of_nitrogen_in_N2O5_l35_35854

noncomputable def atomic_mass_N : ℝ := 14.01
noncomputable def atomic_mass_O : ℝ := 16.00
noncomputable def molar_mass_N2O5 : ℝ := (2 * atomic_mass_N) + (5 * atomic_mass_O)

theorem mass_percentage_of_nitrogen_in_N2O5 : 
  (2 * atomic_mass_N / molar_mass_N2O5 * 100) = 25.94 := 
by 
  sorry

end mass_percentage_of_nitrogen_in_N2O5_l35_35854


namespace extra_flour_l35_35927

-- Define the conditions
def recipe_flour : ℝ := 7.0
def mary_flour : ℝ := 9.0

-- Prove the number of extra cups of flour Mary puts in
theorem extra_flour : mary_flour - recipe_flour = 2 :=
by
  sorry

end extra_flour_l35_35927


namespace relationship_y1_y2_l35_35465

theorem relationship_y1_y2 (y1 y2 : ℤ) 
  (h1 : y1 = 2 * -3 + 1) 
  (h2 : y2 = 2 * 4 + 1) : y1 < y2 :=
by {
  sorry -- Proof goes here
}

end relationship_y1_y2_l35_35465


namespace roots_cubic_roots_sum_of_squares_l35_35344

variables {R : Type*} [CommRing R] {p q r s t : R}

theorem roots_cubic_roots_sum_of_squares (h1 : r + s + t = p) (h2 : r * s + r * t + s * t = q) :
  r^2 + s^2 + t^2 = p^2 - 2 * q :=
sorry

end roots_cubic_roots_sum_of_squares_l35_35344


namespace fraction_sum_eq_decimal_l35_35632

theorem fraction_sum_eq_decimal : (2 / 5) + (2 / 50) + (2 / 500) = 0.444 := by
  sorry

end fraction_sum_eq_decimal_l35_35632


namespace cubes_side_length_l35_35489

theorem cubes_side_length (s : ℝ) (h : 2 * (s * s + s * 2 * s + s * 2 * s) = 10) : s = 1 :=
by
  sorry

end cubes_side_length_l35_35489


namespace find_a_values_l35_35998

theorem find_a_values (a t t₁ t₂ : ℝ) :
  (t^2 + (a - 6) * t + (9 - 3 * a) = 0) ∧
  (t₁ = 4 * t₂) ∧
  (t₁ + t₂ = 6 - a) ∧
  (t₁ * t₂ = 9 - 3 * a)
  ↔ (a = -2 ∨ a = 2) := sorry

end find_a_values_l35_35998


namespace inverse_proportion_point_l35_35921

theorem inverse_proportion_point (a : ℝ) (h : (a, 7) ∈ {p : ℝ × ℝ | ∃ x y, y = 14 / x ∧ p = (x, y)}) : a = 2 :=
by
  sorry

end inverse_proportion_point_l35_35921


namespace trip_length_is_440_l35_35858

noncomputable def total_trip_length (d : ℝ) : Prop :=
  55 * 0.02 * (d - 40) = d

theorem trip_length_is_440 :
  total_trip_length 440 :=
by
  sorry

end trip_length_is_440_l35_35858


namespace total_savings_calculation_l35_35185

theorem total_savings_calculation
  (income : ℕ)
  (ratio_income_to_expenditure : ℕ)
  (ratio_expenditure_to_income : ℕ)
  (tax_rate : ℚ)
  (investment_rate : ℚ)
  (expenditure : ℕ)
  (taxes : ℚ)
  (investments : ℚ)
  (total_savings : ℚ)
  (h_income : income = 17000)
  (h_ratio : ratio_income_to_expenditure / ratio_expenditure_to_income = 5 / 4)
  (h_tax_rate : tax_rate = 0.15)
  (h_investment_rate : investment_rate = 0.1)
  (h_expenditure : expenditure = (income / 5) * 4)
  (h_taxes : taxes = 0.15 * income)
  (h_investments : investments = 0.1 * income)
  (h_total_savings : total_savings = income - (expenditure + taxes + investments)) :
  total_savings = 900 :=
by
  sorry

end total_savings_calculation_l35_35185


namespace green_block_weight_l35_35864

theorem green_block_weight
    (y : ℝ)
    (g : ℝ)
    (h1 : y = 0.6)
    (h2 : y = g + 0.2) :
    g = 0.4 :=
by
  sorry

end green_block_weight_l35_35864


namespace greatest_integer_le_x_squared_div_50_l35_35087

-- Define the conditions as given in the problem
def trapezoid (b h : ℝ) (x : ℝ) : Prop :=
  let baseDifference := 50
  let longerBase := b + baseDifference
  let midline := (b + longerBase) / 2
  let heightRatioFactor := 2
  let xSquared := 6875
  let regionAreaRatio := 2 / 1 -- represented as 2
  (let areaRatio := (b + midline) / (b + baseDifference / 2)
   areaRatio = regionAreaRatio) ∧
  (x = Real.sqrt xSquared) ∧
  (b = 50)

-- Define the theorem that captures the question
theorem greatest_integer_le_x_squared_div_50 (b h x : ℝ) (h_trapezoid : trapezoid b h x) :
  ⌊ (x^2) / 50 ⌋ = 137 :=
by sorry

end greatest_integer_le_x_squared_div_50_l35_35087


namespace operation_result_l35_35152

theorem operation_result (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) (h_sum : a + b = 12) (h_prod : a * b = 32) 
: (1 / a : ℚ) + (1 / b) = 3 / 8 := by
  sorry

end operation_result_l35_35152


namespace find_value_of_a_l35_35319

theorem find_value_of_a (a : ℝ) (h : 2 - a = 0) : a = 2 :=
by {
  sorry
}

end find_value_of_a_l35_35319


namespace find_multiple_l35_35810

-- Definitions and given conditions
def total_seats : ℤ := 387
def first_class_seats : ℤ := 77

-- The statement we need to prove
theorem find_multiple (m : ℤ) :
  (total_seats = first_class_seats + (m * first_class_seats + 2)) → m = 4 :=
by
  sorry

end find_multiple_l35_35810


namespace vector_dot_product_l35_35076

def dot_product (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2

noncomputable def sin_deg (x : ℝ) : ℝ := Real.sin (x * Real.pi / 180)
noncomputable def cos_deg (x : ℝ) : ℝ := Real.cos (x * Real.pi / 180)

theorem vector_dot_product :
  let a := (sin_deg 55, sin_deg 35)
  let b := (sin_deg 25, sin_deg 65)
  dot_product a b = (Real.sqrt 3) / 2 :=
by
  sorry

end vector_dot_product_l35_35076


namespace probability_of_exactly_one_hitting_l35_35431

variable (P_A_hitting B_A_hitting : ℝ)

theorem probability_of_exactly_one_hitting (hP_A : P_A_hitting = 0.6) (hP_B : B_A_hitting = 0.6) :
  ((P_A_hitting * (1 - B_A_hitting)) + ((1 - P_A_hitting) * B_A_hitting)) = 0.48 := 
by 
  sorry

end probability_of_exactly_one_hitting_l35_35431


namespace largest_n_for_perfect_square_l35_35609

theorem largest_n_for_perfect_square :
  ∃ n : ℕ, 4 ^ 27 + 4 ^ 500 + 4 ^ n = k ^ 2 ∧ ∀ m : ℕ, 4 ^ 27 + 4 ^ 500 + 4 ^ m = l ^ 2 → m ≤ n  → n = 972 :=
sorry

end largest_n_for_perfect_square_l35_35609


namespace negation_of_forall_x_gt_1_l35_35790

theorem negation_of_forall_x_gt_1 : ¬(∀ x : ℝ, x^2 > 1) ↔ (∃ x : ℝ, x^2 ≤ 1) := by
  sorry

end negation_of_forall_x_gt_1_l35_35790


namespace problem1_problem2_l35_35180

variables {a b : ℝ}

-- Given conditions
def condition1 : a + b = 2 := sorry
def condition2 : a * b = -1 := sorry

-- Proof for a^2 + b^2 = 6
theorem problem1 (h1 : a + b = 2) (h2 : a * b = -1) : a^2 + b^2 = 6 :=
by sorry

-- Proof for (a - b)^2 = 8
theorem problem2 (h1 : a + b = 2) (h2 : a * b = -1) : (a - b)^2 = 8 :=
by sorry

end problem1_problem2_l35_35180


namespace equation_has_one_solution_l35_35010

theorem equation_has_one_solution : ∀ x : ℝ, x - 6 / (x - 2) = 4 - 6 / (x - 2) ↔ x = 4 :=
by {
  -- proof goes here
  sorry
}

end equation_has_one_solution_l35_35010


namespace angle_sum_in_triangle_l35_35815

theorem angle_sum_in_triangle (A B C : ℝ) (h₁ : A + B = 90) (h₂ : A + B + C = 180) : C = 90 := by
  sorry

end angle_sum_in_triangle_l35_35815


namespace rectangle_square_problem_l35_35279

theorem rectangle_square_problem
  (m n x : ℕ)
  (h : 2 * (m + n) + 2 * x = m * n)
  (h2 : m * n - x^2 = 2 * (m + n)) :
  x = 2 ∧ ((m = 3 ∧ n = 10) ∨ (m = 6 ∧ n = 4)) :=
by {
  -- Proof goes here
  sorry
}

end rectangle_square_problem_l35_35279


namespace min_rectangles_to_cover_minimum_number_of_rectangles_required_l35_35650

-- Definitions based on the conditions
def corners_type1 : Nat := 12
def corners_type2 : Nat := 12

theorem min_rectangles_to_cover (type1_corners type2_corners : Nat) (h1 : type1_corners = corners_type1) (h2 : type2_corners = corners_type2) : Nat :=
12

theorem minimum_number_of_rectangles_required (type1_corners type2_corners : Nat) (h1 : type1_corners = corners_type1) (h2 : type2_corners = corners_type2) :
  min_rectangles_to_cover type1_corners type2_corners h1 h2 = 12 := by
  sorry

end min_rectangles_to_cover_minimum_number_of_rectangles_required_l35_35650


namespace tim_younger_than_jenny_l35_35691

def tim_age : ℕ := 5
def rommel_age : ℕ := 3 * tim_age
def jenny_age : ℕ := rommel_age + 2
def combined_ages_rommel_jenny : ℕ := rommel_age + jenny_age
def uncle_age : ℕ := 2 * combined_ages_rommel_jenny
noncomputable def aunt_age : ℝ := (uncle_age + jenny_age : ℕ) / 2

theorem tim_younger_than_jenny : jenny_age - tim_age = 12 :=
by {
  -- Placeholder proof
  sorry
}

end tim_younger_than_jenny_l35_35691


namespace average_speed_is_35_l35_35136

-- Given constants
def distance : ℕ := 210
def speed_difference : ℕ := 5
def time_difference : ℕ := 1

-- Definition of time for planned speed and actual speed
def planned_time (x : ℕ) : ℚ := distance / (x - speed_difference)
def actual_time (x : ℕ) : ℚ := distance / x

-- Main theorem to be proved
theorem average_speed_is_35 (x : ℕ) (h : (planned_time x - actual_time x) = time_difference) : x = 35 :=
sorry

end average_speed_is_35_l35_35136


namespace sum_of_powers_l35_35580

theorem sum_of_powers : (-1: ℤ) ^ 2006 - (-1) ^ 2007 + 1 ^ 2008 + 1 ^ 2009 - 1 ^ 2010 = 3 := by
  sorry

end sum_of_powers_l35_35580


namespace value_of_m_l35_35869

theorem value_of_m (m : ℝ) (h1 : m^2 - 2 * m - 1 = 2) (h2 : m ≠ 3) : m = -1 :=
sorry

end value_of_m_l35_35869


namespace smallest_four_digit_multiple_of_18_l35_35075

theorem smallest_four_digit_multiple_of_18 : 
  ∃ (n : ℕ), 999 < n ∧ n < 10000 ∧ 18 ∣ n ∧ ∀ m : ℕ, 999 < m ∧ m < 10000 ∧ 18 ∣ m → n ≤ m ∧ n = 1008 := 
sorry

end smallest_four_digit_multiple_of_18_l35_35075


namespace cyclist_average_rate_l35_35963

noncomputable def average_rate_round_trip (D : ℝ) : ℝ :=
  let time_to_travel := D / 10
  let time_to_return := D / 9
  let total_distance := 2 * D
  let total_time := time_to_travel + time_to_return
  (total_distance / total_time)

theorem cyclist_average_rate (D : ℝ) (hD : D > 0) :
  average_rate_round_trip D = 180 / 19 :=
by
  sorry

end cyclist_average_rate_l35_35963


namespace value_of_expression_l35_35019

variable {a b : ℝ}
variables (h1 : ∀ x, 3 * x^2 + 9 * x - 18 = 0 → x = a ∨ x = b)

theorem value_of_expression : (3 * a - 2) * (6 * b - 9) = 27 :=
by
  sorry

end value_of_expression_l35_35019


namespace find_n_in_geom_series_l35_35423

noncomputable def geom_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem find_n_in_geom_series :
  ∃ n : ℕ, geom_sum 1 (1/2) n = 31 / 16 :=
sorry

end find_n_in_geom_series_l35_35423


namespace average_income_N_O_l35_35777

variable (M N O : ℝ)

-- Condition declaration
def condition1 : Prop := M + N = 10100
def condition2 : Prop := M + O = 10400
def condition3 : Prop := M = 4000

-- Theorem statement
theorem average_income_N_O (h1 : condition1 M N) (h2 : condition2 M O) (h3 : condition3 M) :
  (N + O) / 2 = 6250 :=
sorry

end average_income_N_O_l35_35777


namespace geometric_number_difference_l35_35512

theorem geometric_number_difference : 
  ∀ (a b c : ℕ), 8 = a → b ≠ c → (∃ k : ℕ, 8 ≠ k ∧ b = k ∧ c = k * k / 8) → (10^2 * a + 10 * b + c = 842) ∧ (10^2 * a + 10 * b + c = 842) → (10^2 * a + 10 * b + c) - (10^2 * a + 10 * b + c) = 0 :=
by
  intro a b c
  intro ha hb
  intro hk
  intro hseq
  sorry

end geometric_number_difference_l35_35512


namespace student_weekly_allowance_l35_35574

theorem student_weekly_allowance (A : ℝ) (h1 : (4 / 15) * A = 1) : A = 3.75 :=
by
  sorry

end student_weekly_allowance_l35_35574


namespace simplify_sqrt_product_l35_35237

theorem simplify_sqrt_product (x : ℝ) :
  Real.sqrt (45 * x) * Real.sqrt (20 * x) * Real.sqrt (28 * x) * Real.sqrt (5 * x) =
  60 * x^2 * Real.sqrt 35 :=
by
  sorry

end simplify_sqrt_product_l35_35237


namespace max_product_of_two_integers_whose_sum_is_300_l35_35616

theorem max_product_of_two_integers_whose_sum_is_300 :
  ∃ (a b : ℤ), a + b = 300 ∧ a * b = 22500 :=
by
  sorry

end max_product_of_two_integers_whose_sum_is_300_l35_35616


namespace year_2023_not_lucky_l35_35026

def is_valid_date (month day year : ℕ) : Prop :=
  month * day = year % 100

def is_lucky_year (year : ℕ) : Prop :=
  ∃ month day, month ≤ 12 ∧ day ≤ 31 ∧ is_valid_date month day year

theorem year_2023_not_lucky : ¬ is_lucky_year 2023 :=
by sorry

end year_2023_not_lucky_l35_35026


namespace johnson_oldest_child_age_l35_35936

/-- The average age of the three Johnson children is 10 years. 
    The two younger children are 6 years old and 8 years old. 
    Prove that the age of the oldest child is 16 years. -/
theorem johnson_oldest_child_age :
  ∃ x : ℕ, (6 + 8 + x) / 3 = 10 ∧ x = 16 :=
by
  sorry

end johnson_oldest_child_age_l35_35936


namespace slope_intercept_parallel_line_l35_35973

def is_parallel (m1 m2 : ℝ) : Prop :=
  m1 = m2

theorem slope_intercept_parallel_line (A : ℝ × ℝ) (hA₁ : A.1 = 3) (hA₂ : A.2 = 2) 
  (m : ℝ) (h_parallel : is_parallel m (-4)) : ∃ b : ℝ, ∀ x y : ℝ, y = -4 * x + b :=
by
  use 14
  intro x y
  sorry

end slope_intercept_parallel_line_l35_35973


namespace cat_toy_cost_l35_35547

-- Define the conditions
def cost_of_cage : ℝ := 11.73
def total_cost_of_purchases : ℝ := 21.95

-- Define the proof statement
theorem cat_toy_cost : (total_cost_of_purchases - cost_of_cage) = 10.22 := by
  sorry

end cat_toy_cost_l35_35547


namespace evie_collected_shells_for_6_days_l35_35828

theorem evie_collected_shells_for_6_days (d : ℕ) (h1 : 10 * d - 2 = 58) : d = 6 := by
  sorry

end evie_collected_shells_for_6_days_l35_35828


namespace divisor_of_136_l35_35107

theorem divisor_of_136 (d : ℕ) (h : 136 = 9 * d + 1) : d = 15 := 
by {
  -- Since the solution steps are skipped, we use sorry to indicate a placeholder.
  sorry
}

end divisor_of_136_l35_35107


namespace first_investment_percentage_l35_35729

variable (P : ℝ)
variable (x : ℝ := 1400)  -- investment amount in the first investment
variable (y : ℝ := 600)   -- investment amount at 8 percent
variable (income_difference : ℝ := 92)
variable (total_investment : ℝ := 2000)
variable (rate_8_percent : ℝ := 0.08)
variable (exceed_by : ℝ := 92)

theorem first_investment_percentage :
  P * x - rate_8_percent * y = exceed_by →
  total_investment = x + y →
  P = 0.10 :=
by
  -- Solution steps can be filled here if needed
  sorry

end first_investment_percentage_l35_35729


namespace odds_burning_out_during_second_period_l35_35310

def odds_burning_out_during_first_period := 1 / 3
def odds_not_burning_out_first_period := 1 - odds_burning_out_during_first_period
def odds_not_burning_out_next_period := odds_not_burning_out_first_period / 2

theorem odds_burning_out_during_second_period :
  (1 - odds_not_burning_out_next_period) = 2 / 3 := by
  sorry

end odds_burning_out_during_second_period_l35_35310


namespace pool_cannot_be_filled_l35_35899

noncomputable def pool := 48000 -- Pool capacity in gallons
noncomputable def hose_rate := 3 -- Rate of each hose in gallons per minute
noncomputable def number_of_hoses := 6 -- Number of hoses
noncomputable def leakage_rate := 18 -- Leakage rate in gallons per minute

theorem pool_cannot_be_filled : 
  (number_of_hoses * hose_rate - leakage_rate <= 0) -> False :=
by
  -- Skipping the proof with 'sorry' as per instructions
  sorry

end pool_cannot_be_filled_l35_35899


namespace triangle_perimeter_l35_35663

noncomputable def smallest_perimeter (a b c : ℕ) : ℕ :=
  a + b + c

theorem triangle_perimeter (a b c : ℕ) (A B C : ℝ) (h1 : A = 2 * B) 
  (h2 : C > π / 2) (h3 : a^2 = b * (b + c)) (h4 : ∃ m n : ℕ, b = m^2 ∧ b + c = n^2 ∧ a = m * n) :
  smallest_perimeter 28 16 33 = 77 :=
by sorry

end triangle_perimeter_l35_35663


namespace walnuts_amount_l35_35459

theorem walnuts_amount (w : ℝ) (total_nuts : ℝ) (almonds : ℝ) (h1 : total_nuts = 0.5) (h2 : almonds = 0.25) (h3 : w + almonds = total_nuts) : w = 0.25 :=
by
  sorry

end walnuts_amount_l35_35459


namespace second_class_students_count_l35_35630

theorem second_class_students_count 
    (x : ℕ)
    (h1 : 12 * 40 = 480)
    (h2 : ∀ x, x * 60 = 60 * x)
    (h3 : (12 + x) * 54 = 480 + 60 * x) : 
    x = 28 :=
by
  sorry

end second_class_students_count_l35_35630


namespace yankees_mets_ratio_l35_35302

-- Given conditions
def num_mets_fans : ℕ := 104
def total_fans : ℕ := 390
def ratio_mets_to_redsox : ℚ := 4 / 5

-- Definitions
def num_redsox_fans (M : ℕ) := (5 / 4) * M
def num_yankees_fans (Y M B : ℕ) := (total_fans - M - B)

-- Theorem statement
theorem yankees_mets_ratio (Y M B : ℕ)
  (h1 : M = num_mets_fans)
  (h2 : Y + M + B = total_fans)
  (h3 : (M : ℚ) / (B : ℚ) = ratio_mets_to_redsox) :
  (Y : ℚ) / (M : ℚ) = 3 / 2 :=
sorry

end yankees_mets_ratio_l35_35302


namespace determinant_tan_matrix_l35_35493

theorem determinant_tan_matrix (B C : ℝ) (h : B + C = 3 * π / 4) :
  Matrix.det ![
    ![Real.tan (π / 4), 1, 1],
    ![1, Real.tan B, 1],
    ![1, 1, Real.tan C]
  ] = 1 :=
by
  sorry

end determinant_tan_matrix_l35_35493


namespace unique_solution_for_system_l35_35679

theorem unique_solution_for_system (a : ℝ) :
  (∀ x y z : ℝ, x^2 + y^2 + z^2 + 4 * y = 0 ∧ x + a * y + a * z - a = 0 →
    (a = 2 ∨ a = -2)) :=
by
  intros x y z h
  sorry

end unique_solution_for_system_l35_35679


namespace millet_percentage_in_mix_l35_35244

def contribution_millet_brandA (percA mixA : ℝ) := percA * mixA
def contribution_millet_brandB (percB mixB : ℝ) := percB * mixB

theorem millet_percentage_in_mix
  (percA : ℝ) (percB : ℝ) (mixA : ℝ) (mixB : ℝ)
  (h1 : percA = 0.40) (h2 : percB = 0.65) (h3 : mixA = 0.60) (h4 : mixB = 0.40) :
  (contribution_millet_brandA percA mixA + contribution_millet_brandB percB mixB = 0.50) :=
by
  sorry

end millet_percentage_in_mix_l35_35244


namespace addition_results_in_perfect_square_l35_35881

theorem addition_results_in_perfect_square : ∃ n: ℕ, n * n = 4440 + 49 :=
by
  sorry

end addition_results_in_perfect_square_l35_35881


namespace student_average_marks_l35_35997

theorem student_average_marks 
(P C M : ℕ) 
(h1 : (P + M) / 2 = 90) 
(h2 : (P + C) / 2 = 70) 
(h3 : P = 65) : 
  (P + C + M) / 3 = 85 :=
  sorry

end student_average_marks_l35_35997


namespace sine_wave_solution_l35_35751

theorem sine_wave_solution (a b c : ℝ) (h_pos_a : a > 0) 
  (h_amp : a = 3) 
  (h_period : (2 * Real.pi) / b = Real.pi) 
  (h_peak : (Real.pi / (2 * b)) - (c / b) = Real.pi / 6) : 
  a = 3 ∧ b = 2 ∧ c = Real.pi / 6 :=
by
  -- Lean code to construct the proof will appear here
  sorry

end sine_wave_solution_l35_35751


namespace isPossible_l35_35241

structure Person where
  firstName : String
  patronymic : String
  surname : String

def conditions (people : List Person) : Prop :=
  people.length = 4 ∧
  ∀ p1 p2 p3 : Person, 
    p1 ∈ people → p2 ∈ people → p3 ∈ people →
    p1 ≠ p2 → p2 ≠ p3 → p1 ≠ p3 →
    p1.firstName ≠ p2.firstName ∨ p2.firstName ≠ p3.firstName ∨ p1.firstName ≠ p3.firstName ∧
    p1.patronymic ≠ p2.patronymic ∨ p2.patronymic ≠ p3.patronymic ∨ p1.patronymic ≠ p3.patronymic ∧
    p1.surname ≠ p2.surname ∨ p2.surname ≠ p3.surname ∨ p1.surname ≠ p3.surname ∧
  ∀ p1 p2 : Person, 
    p1 ∈ people → p2 ∈ people →
    p1 ≠ p2 →
    p1.firstName = p2.firstName ∨ p1.patronymic = p2.patronymic ∨ p1.surname = p2.surname

theorem isPossible : ∃ people : List Person, conditions people := by
  sorry

end isPossible_l35_35241


namespace find_k_value_l35_35361

theorem find_k_value (x y k : ℝ) 
  (h1 : 2 * x + y = 1) 
  (h2 : x + 2 * y = k - 2) 
  (h3 : x - y = 2) : 
  k = 1 := 
by
  sorry

end find_k_value_l35_35361


namespace scientific_notation_of_308000000_l35_35557

theorem scientific_notation_of_308000000 :
  ∃ (a : ℝ) (n : ℤ), (a = 3.08) ∧ (n = 8) ∧ (308000000 = a * 10 ^ n) :=
by
  sorry

end scientific_notation_of_308000000_l35_35557


namespace remainder_when_divided_by_6_l35_35197

theorem remainder_when_divided_by_6 (n : ℕ) (h : n % 12 = 8) : n % 6 = 2 :=
by sorry

end remainder_when_divided_by_6_l35_35197


namespace min_value_a_l35_35323

theorem min_value_a (a : ℝ) : 
  (∀ x y : ℝ, (3 * x - 5 * y) ≥ 0 → x > 0 → y > 0 → (1 - a) * x ^ 2 + 2 * x * y - a * y ^ 2 ≤ 0) ↔ a ≥ 55 / 34 := 
by 
  sorry

end min_value_a_l35_35323


namespace sum_solution_equation_l35_35317

theorem sum_solution_equation (n : ℚ) : (∃ x : ℚ, (n / x = 3 - n) ∧ (x = 1 / (n + (3 - n)))) → n = 3 / 4 := by
  intros h
  sorry

end sum_solution_equation_l35_35317


namespace range_neg_square_l35_35073

theorem range_neg_square (x : ℝ) (hx : -3 ≤ x ∧ x ≤ 1) : 
  -9 ≤ -x^2 ∧ -x^2 ≤ 0 :=
sorry

end range_neg_square_l35_35073


namespace sequence_general_formula_l35_35411

theorem sequence_general_formula (a : ℕ → ℕ) :
  (a 1 = 1) ∧ (a 2 = 2) ∧ (a 3 = 4) ∧ (a 4 = 8) ∧ (a 5 = 16) → ∀ n : ℕ, n > 0 → a n = 2^(n-1) :=
by
  intros h n hn
  sorry

end sequence_general_formula_l35_35411


namespace area_ratio_parallelogram_to_triangle_l35_35447

variables {A B C D R E : Type*}
variables (s_AB s_AD : ℝ)

-- Given AR = 2/3 AB and AE = 1/3 AD
axiom AR_proportion : s_AB > 0 → s_AB * (2/3) = s_AB
axiom AE_proportion : s_AD > 0 → s_AD * (1/3) = s_AD

-- Given the relationship, we need to prove
theorem area_ratio_parallelogram_to_triangle (hAB : s_AB > 0) (hAD : s_AD > 0) :
  ∃ (S_ABCD S_ARE : ℝ), S_ABCD / S_ARE = 9 :=
by {
  sorry
}

end area_ratio_parallelogram_to_triangle_l35_35447


namespace diameter_of_circle_given_radius_l35_35005

theorem diameter_of_circle_given_radius (radius: ℝ) (h: radius = 7): 
  2 * radius = 14 :=
by
  rw [h]
  sorry

end diameter_of_circle_given_radius_l35_35005


namespace solve_r_l35_35548

def E (a : ℝ) (b : ℝ) (c : ℕ) : ℝ := a * b^c

theorem solve_r : ∃ (r : ℝ), E r r 5 = 1024 ∧ r = 2^(5/3) :=
by
  sorry

end solve_r_l35_35548


namespace carter_family_children_l35_35198

variable (f m x y : ℕ)

theorem carter_family_children 
  (avg_family : (3 * y + m + x * y) / (2 + x) = 25)
  (avg_mother_children : (m + x * y) / (1 + x) = 18)
  (father_age : f = 3 * y)
  (simplest_case : y = x) :
  x = 8 :=
by
  -- Proof to be provided
  sorry

end carter_family_children_l35_35198


namespace average_brown_mms_l35_35281

def brown_mms_bag_1 := 9
def brown_mms_bag_2 := 12
def brown_mms_bag_3 := 8
def brown_mms_bag_4 := 8
def brown_mms_bag_5 := 3

def total_brown_mms : ℕ := brown_mms_bag_1 + brown_mms_bag_2 + brown_mms_bag_3 + brown_mms_bag_4 + brown_mms_bag_5

theorem average_brown_mms :
  (total_brown_mms / 5) = 8 := by
  rw [total_brown_mms]
  norm_num
  sorry

end average_brown_mms_l35_35281


namespace two_divides_a_squared_minus_a_three_divides_a_cubed_minus_a_l35_35635

theorem two_divides_a_squared_minus_a (a : ℤ) : ∃ k₁ : ℤ, a^2 - a = 2 * k₁ :=
sorry

theorem three_divides_a_cubed_minus_a (a : ℤ) : ∃ k₂ : ℤ, a^3 - a = 3 * k₂ :=
sorry

end two_divides_a_squared_minus_a_three_divides_a_cubed_minus_a_l35_35635


namespace tom_candy_pieces_l35_35939

def total_boxes : ℕ := 14
def give_away_boxes : ℕ := 8
def pieces_per_box : ℕ := 3

theorem tom_candy_pieces : (total_boxes - give_away_boxes) * pieces_per_box = 18 := 
by 
  sorry

end tom_candy_pieces_l35_35939


namespace smallest_two_digit_multiple_of_17_smallest_four_digit_multiple_of_17_l35_35064

theorem smallest_two_digit_multiple_of_17 : ∃ m, 10 ≤ m ∧ m < 100 ∧ 17 ∣ m ∧ ∀ n, 10 ≤ n ∧ n < 100 ∧ 17 ∣ n → m ≤ n :=
by
  sorry

theorem smallest_four_digit_multiple_of_17 : ∃ m, 1000 ≤ m ∧ m < 10000 ∧ 17 ∣ m ∧ ∀ n, 1000 ≤ n ∧ n < 10000 ∧ 17 ∣ n → m ≤ n :=
by
  sorry

end smallest_two_digit_multiple_of_17_smallest_four_digit_multiple_of_17_l35_35064


namespace proof_problem_l35_35461

noncomputable def f (x : ℝ) : ℝ := (x + 1) / (x - 1)

theorem proof_problem (x : ℝ) (h : x ≠ 1 ∧ x ≠ -1) : f x * f (-x) = 1 := 
by 
  sorry

end proof_problem_l35_35461


namespace percentage_increase_after_lawnmower_l35_35644

-- Definitions from conditions
def initial_daily_yards := 8
def weekly_yards_after_lawnmower := 84
def days_in_week := 7

-- Problem statement
theorem percentage_increase_after_lawnmower : 
  ((weekly_yards_after_lawnmower / days_in_week - initial_daily_yards) / initial_daily_yards) * 100 = 50 := 
by 
  sorry

end percentage_increase_after_lawnmower_l35_35644


namespace girls_with_short_hair_count_l35_35826

-- Definitions based on the problem's conditions
def TotalPeople := 55
def Boys := 30
def FractionLongHair : ℚ := 3 / 5

-- The statement to prove
theorem girls_with_short_hair_count :
  (TotalPeople - Boys) - (TotalPeople - Boys) * FractionLongHair = 10 :=
by
  sorry

end girls_with_short_hair_count_l35_35826


namespace basketball_shots_l35_35066

theorem basketball_shots (total_points total_3pt_shots: ℕ) 
  (h1: total_points = 26) 
  (h2: total_3pt_shots = 4) 
  (h3: ∀ points_from_3pt_shots, points_from_3pt_shots = 3 * total_3pt_shots) :
  let points_from_3pt_shots := 3 * total_3pt_shots
  let points_from_2pt_shots := total_points - points_from_3pt_shots
  let total_2pt_shots := points_from_2pt_shots / 2
  total_2pt_shots + total_3pt_shots = 11 :=
by
  sorry

end basketball_shots_l35_35066


namespace sin_double_angle_tangent_identity_l35_35817

theorem sin_double_angle_tangent_identity (x : ℝ) 
  (h : Real.tan (x + Real.pi / 4) = 2) : 
  Real.sin (2 * x) = 3 / 5 :=
by
  -- proof is omitted
  sorry

end sin_double_angle_tangent_identity_l35_35817


namespace empty_boxes_count_l35_35972

-- Definitions based on conditions:
def large_box_contains (B : Type) : ℕ := 1
def initial_small_boxes (B : Type) : ℕ := 10
def non_empty_boxes (B : Type) : ℕ := 6
def additional_smaller_boxes_in_non_empty (B : Type) (b : B) : ℕ := 10
def non_empty_small_boxes := 5

-- Proving that the number of empty boxes is 55 given the conditions:
theorem empty_boxes_count (B : Type) : 
  large_box_contains B = 1 ∧
  initial_small_boxes B = 10 ∧
  non_empty_boxes B = 6 ∧
  (∃ b : B, additional_smaller_boxes_in_non_empty B b = 10) →
  (initial_small_boxes B - non_empty_small_boxes + non_empty_small_boxes * additional_smaller_boxes_in_non_empty B) = 55 :=
by 
  sorry

end empty_boxes_count_l35_35972


namespace emily_total_spent_l35_35702

-- Define the given conditions.
def cost_per_flower : ℕ := 3
def num_roses : ℕ := 2
def num_daisies : ℕ := 2

-- Calculate the total number of flowers and the total cost.
def total_flowers : ℕ := num_roses + num_daisies
def total_cost : ℕ := total_flowers * cost_per_flower

-- Statement: Prove that Emily spent 12 dollars.
theorem emily_total_spent : total_cost = 12 := by
  sorry

end emily_total_spent_l35_35702


namespace piglet_steps_count_l35_35144

theorem piglet_steps_count (u v L : ℝ) (h₁ : (L * u) / (u + v) = 66) (h₂ : (L * u) / (u - v) = 198) : L = 99 :=
sorry

end piglet_steps_count_l35_35144


namespace solve_inequality_system_l35_35955

theorem solve_inequality_system (x : ℝ) :
  (8 * x - 3 ≤ 13) ∧ ((x - 1) / 3 - 2 < x - 1) → -2 < x ∧ x ≤ 2 :=
by
  intros h
  sorry

end solve_inequality_system_l35_35955


namespace fish_count_l35_35720

theorem fish_count (T : ℕ) :
  (T > 10 ∧ T ≤ 18) ∧ ((T > 18 ∧ T > 15 ∧ ¬(T > 10)) ∨ (¬(T > 18) ∧ T > 15 ∧ T > 10) ∨ (T > 18 ∧ ¬(T > 15) ∧ T > 10)) →
  T = 16 ∨ T = 17 ∨ T = 18 :=
sorry

end fish_count_l35_35720


namespace total_unique_working_games_l35_35297

-- Define the given conditions
def initial_games_from_friend := 25
def non_working_games_from_friend := 12

def games_from_garage_sale := 15
def non_working_games_from_garage_sale := 8
def duplicate_games := 3

-- Calculate the number of working games from each source
def working_games_from_friend := initial_games_from_friend - non_working_games_from_friend
def total_garage_sale_games := games_from_garage_sale - non_working_games_from_garage_sale
def unique_working_games_from_garage_sale := total_garage_sale_games - duplicate_games

-- Theorem statement
theorem total_unique_working_games : 
  working_games_from_friend + unique_working_games_from_garage_sale = 17 := by
  sorry

end total_unique_working_games_l35_35297


namespace first_discount_percentage_l35_35072

theorem first_discount_percentage (normal_price sale_price : ℝ) (second_discount : ℝ) (first_discount : ℝ) :
  normal_price = 149.99999999999997 →
  sale_price = 108 →
  second_discount = 0.20 →
  (1 - second_discount) * (1 - first_discount) * normal_price = sale_price →
  first_discount = 0.10 :=
by
  intros
  sorry

end first_discount_percentage_l35_35072


namespace cost_of_game_l35_35253

theorem cost_of_game
  (number_of_ice_creams : ℕ) 
  (price_per_ice_cream : ℕ)
  (total_sold : number_of_ice_creams = 24)
  (price : price_per_ice_cream = 5) :
  (number_of_ice_creams * price_per_ice_cream) / 2 = 60 :=
by
  sorry

end cost_of_game_l35_35253


namespace arithmetic_sequence_properties_l35_35003

-- Defining the arithmetic sequence and the conditions
variable {a : ℕ → ℤ}
variable {d : ℤ}
noncomputable def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) - a n = d

-- Given conditions
variable (h1 : a 5 = 10)
variable (h2 : a 1 + a 2 + a 3 = 3)

-- The theorem to prove
theorem arithmetic_sequence_properties :
  is_arithmetic_sequence a d → a 1 = -2 ∧ d = 3 :=
sorry

end arithmetic_sequence_properties_l35_35003


namespace class_mean_score_l35_35042

theorem class_mean_score:
  ∀ (n: ℕ) (m: ℕ) (a b: ℕ),
  n + m = 50 →
  n * a = 3400 →
  m * b = 750 →
  a = 85 →
  b = 75 →
  (n * a + m * b) / (n + m) = 83 :=
by
  intros n m a b h1 h2 h3 h4 h5
  sorry

end class_mean_score_l35_35042


namespace trigonometric_identity_l35_35528

theorem trigonometric_identity
  (θ : ℝ)
  (h1 : θ > -π/2)
  (h2 : θ < 0)
  (h3 : Real.tan θ = -2) :
  (Real.sin θ)^2 / (Real.cos (2 * θ) + 2) = 4 / 7 :=
sorry

end trigonometric_identity_l35_35528


namespace farmer_pomelos_dozen_l35_35507

theorem farmer_pomelos_dozen (pomelos_last_week : ℕ) (boxes_last_week : ℕ) (boxes_this_week : ℕ) :
  pomelos_last_week = 240 → boxes_last_week = 10 → boxes_this_week = 20 →
  (pomelos_last_week / boxes_last_week) * boxes_this_week / 12 = 40 := 
by
  intro h1 h2 h3
  sorry

end farmer_pomelos_dozen_l35_35507


namespace solve_equation_l35_35669

theorem solve_equation (x : ℚ) :
  (1 / (x + 2) + 3 * x / (x + 2) + 4 / (x + 2) = 1) → x = -3 / 2 :=
by
  sorry

end solve_equation_l35_35669


namespace xia_sheets_left_l35_35923

def stickers_left (initial : ℕ) (shared : ℕ) (per_sheet : ℕ) : ℕ :=
  (initial - shared) / per_sheet

theorem xia_sheets_left :
  stickers_left 150 100 10 = 5 :=
by
  sorry

end xia_sheets_left_l35_35923


namespace solution_set_of_inequality_l35_35894

theorem solution_set_of_inequality :
  {x : ℝ | |2 * x + 1| > 3} = {x : ℝ | x < -2} ∪ {x : ℝ | x > 1} :=
by sorry

end solution_set_of_inequality_l35_35894


namespace distance_at_40_kmph_l35_35114

theorem distance_at_40_kmph (x : ℝ) (h1 : x / 40 + (250 - x) / 60 = 5) : x = 100 := 
by
  sorry

end distance_at_40_kmph_l35_35114


namespace math_proof_problem_l35_35539

theorem math_proof_problem : (10^8 / (2 * 10^5) - 50) = 450 := 
  by
  sorry

end math_proof_problem_l35_35539


namespace water_level_drop_recording_l35_35303

theorem water_level_drop_recording (rise6_recorded: Int): 
    (rise6_recorded = 6) → (6 = -rise6_recorded) :=
by
  sorry

end water_level_drop_recording_l35_35303


namespace complement_A_in_U_l35_35421

-- Define the universal set as ℝ
def U : Set ℝ := Set.univ

-- Define the set A as given in the conditions
def A : Set ℝ := {y | ∃ x : ℝ, 2^(Real.log x) = y}

-- The main statement based on the conditions and the correct answer
theorem complement_A_in_U : (U \ A) = {y | y ≤ 0} := by
  sorry

end complement_A_in_U_l35_35421


namespace largest_possible_perimeter_l35_35031

noncomputable def max_perimeter (a b c: ℕ) : ℕ := 2 * (a + b + c - 6)

theorem largest_possible_perimeter :
  ∃ (a b c : ℕ), (a = c) ∧ ((a - 2) * (b - 2) = 8) ∧ (max_perimeter a b c = 42) := by
  sorry

end largest_possible_perimeter_l35_35031


namespace minimize_y_at_x_l35_35463

-- Define the function y
def y (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := 3 * (x - a) ^ 2 + (x - b) ^ 2 

-- State the theorem
theorem minimize_y_at_x (a b : ℝ) : ∃ x : ℝ, (∀ x' : ℝ, y x' a b ≥ y ((3 * a + b) / 4) a b) :=
by
  sorry

end minimize_y_at_x_l35_35463


namespace ice_cream_cone_cost_is_5_l35_35803

noncomputable def cost_of_ice_cream_cone (x : ℝ) : Prop := 
  let total_cost_of_cones := 15 * x
  let total_cost_of_puddings := 5 * 2
  let extra_spent_on_cones := total_cost_of_cones - total_cost_of_puddings
  extra_spent_on_cones = 65

theorem ice_cream_cone_cost_is_5 : ∃ x : ℝ, cost_of_ice_cream_cone x ∧ x = 5 :=
by 
  use 5
  unfold cost_of_ice_cream_cone
  simp
  sorry

end ice_cream_cone_cost_is_5_l35_35803


namespace base_height_l35_35084

-- Define the height of the sculpture and the combined height.
def sculpture_height : ℚ := 2 + 10 / 12
def total_height : ℚ := 3 + 2 / 3

-- We want to prove that the base height is 5/6 feet.
theorem base_height :
  total_height - sculpture_height = 5 / 6 :=
by
  sorry

end base_height_l35_35084


namespace pirate_prob_l35_35374

def probability_treasure_no_traps := 1 / 3
def probability_traps_no_treasure := 1 / 6
def probability_neither := 1 / 2

theorem pirate_prob : (70 : ℝ) * ((1 / 3)^4 * (1 / 2)^4) = 35 / 648 := by
  sorry

end pirate_prob_l35_35374


namespace fraction_value_l35_35214

theorem fraction_value (x : ℝ) (h : x + 1/x = 3) : x^2 / (x^4 + x^2 + 1) = 1/8 :=
by sorry

end fraction_value_l35_35214


namespace floor_area_difference_l35_35271

noncomputable def area_difference (r_outer : ℝ) (n : ℕ) (r_inner : ℝ) : ℝ :=
  let outer_area := Real.pi * r_outer^2
  let inner_area := n * Real.pi * r_inner^2
  outer_area - inner_area

theorem floor_area_difference :
  ∀ (r_outer : ℝ) (n : ℕ) (r_inner : ℝ), 
  n = 8 ∧ r_outer = 40 ∧ r_inner = 40 / (2*Real.sqrt 2 + 1) →
  ⌊area_difference r_outer n r_inner⌋ = 1150 :=
by
  intros
  sorry

end floor_area_difference_l35_35271


namespace mixed_number_division_l35_35822

theorem mixed_number_division :
  (5 + 1 / 2 - (2 + 2 / 3)) / (1 + 1 / 5 + 3 + 1 / 4) = 0 + 170 / 267 := 
by
  sorry

end mixed_number_division_l35_35822


namespace exists_disjoint_nonempty_subsets_with_equal_sum_l35_35531

theorem exists_disjoint_nonempty_subsets_with_equal_sum :
  ∀ (A : Finset ℕ), (A.card = 11) → (∀ a ∈ A, 1 ≤ a ∧ a ≤ 100) →
  ∃ (B C : Finset ℕ), B ≠ ∅ ∧ C ≠ ∅ ∧ B ∩ C = ∅ ∧ (B ∪ C ⊆ A) ∧ (B.sum id = C.sum id) :=
by
  sorry

end exists_disjoint_nonempty_subsets_with_equal_sum_l35_35531


namespace solution_set_of_inequality_l35_35052

theorem solution_set_of_inequality :
  { x : ℝ | 2 * x^2 - x - 3 > 0 } = { x : ℝ | x > 3 / 2 ∨ x < -1 } :=
sorry

end solution_set_of_inequality_l35_35052


namespace students_with_two_skills_l35_35452

theorem students_with_two_skills :
  ∀ (n_students n_chess n_puzzles n_code : ℕ),
  n_students = 120 →
  n_chess = n_students - 50 →
  n_puzzles = n_students - 75 →
  n_code = n_students - 40 →
  (n_chess + n_puzzles + n_code - n_students) = 75 :=
by 
  sorry

end students_with_two_skills_l35_35452


namespace vector_b_norm_range_l35_35758

variable (a b : ℝ × ℝ)
variable (norm_a : ‖a‖ = 1)
variable (norm_sum : ‖a + b‖ = 2)

theorem vector_b_norm_range : 1 ≤ ‖b‖ ∧ ‖b‖ ≤ 3 :=
sorry

end vector_b_norm_range_l35_35758


namespace faculty_after_reduction_is_correct_l35_35581

-- Define the original number of faculty members
def original_faculty : ℝ := 253.25

-- Define the reduction percentage as a decimal
def reduction_percentage : ℝ := 0.23

-- Calculate the reduction amount
def reduction_amount : ℝ := original_faculty * reduction_percentage

-- Define the rounded reduction amount
def rounded_reduction_amount : ℝ := 58.25

-- Calculate the number of professors after the reduction
def professors_after_reduction : ℝ := original_faculty - rounded_reduction_amount

-- Statement to be proven: the number of professors after the reduction is 195
theorem faculty_after_reduction_is_correct : professors_after_reduction = 195 := by
  sorry

end faculty_after_reduction_is_correct_l35_35581


namespace gambler_target_win_percentage_l35_35242

-- Define the initial conditions
def initial_games_played : ℕ := 20
def initial_win_rate : ℚ := 0.40

def additional_games_played : ℕ := 20
def additional_win_rate : ℚ := 0.80

-- Define the proof problem statement
theorem gambler_target_win_percentage 
  (initial_wins : ℚ := initial_win_rate * initial_games_played)
  (additional_wins : ℚ := additional_win_rate * additional_games_played)
  (total_games_played : ℕ := initial_games_played + additional_games_played)
  (total_wins : ℚ := initial_wins + additional_wins) :
  ((total_wins / total_games_played) * 100 : ℚ) = 60 := 
by
  -- Skipping the proof steps
  sorry

end gambler_target_win_percentage_l35_35242


namespace find_discounts_l35_35959

variables (a b c : ℝ)
variables (x y z : ℝ)

theorem find_discounts (h1 : 1.1 * a - x * a = 0.99 * a)
                       (h2 : 1.12 * b - y * b = 0.99 * b)
                       (h3 : 1.15 * c - z * c = 0.99 * c) : 
x = 0.11 ∧ y = 0.13 ∧ z = 0.16 := 
sorry

end find_discounts_l35_35959


namespace polygon_sides_eq_seven_l35_35184

theorem polygon_sides_eq_seven (n d : ℕ) (h1 : d = (n * (n - 3)) / 2) (h2 : d = 2 * n) : n = 7 := 
by
  sorry

end polygon_sides_eq_seven_l35_35184


namespace num_ways_4x4_proof_l35_35365

-- Define a function that represents the number of ways to cut a 2x2 square
noncomputable def num_ways_2x2_cut : ℕ := 4

-- Define a function that represents the number of ways to cut a 3x3 square
noncomputable def num_ways_3x3_cut (ways_2x2 : ℕ) : ℕ :=
  ways_2x2 * 4

-- Define a function that represents the number of ways to cut a 4x4 square
noncomputable def num_ways_4x4_cut (ways_3x3 : ℕ) : ℕ :=
  ways_3x3 * 4

-- Prove the final number of ways to cut the 4x4 square into 3 L-shaped pieces and 1 small square
theorem num_ways_4x4_proof : num_ways_4x4_cut (num_ways_3x3_cut num_ways_2x2_cut) = 64 := by
  sorry

end num_ways_4x4_proof_l35_35365


namespace combined_room_size_l35_35123

theorem combined_room_size (M J S : ℝ) 
  (h1 : M + J + S = 800) 
  (h2 : J = M + 100) 
  (h3 : S = M - 50) : 
  J + S = 550 := 
by
  sorry

end combined_room_size_l35_35123


namespace cricket_average_score_l35_35035

theorem cricket_average_score (A : ℝ)
    (h1 : 3 * 30 = 90)
    (h2 : 5 * 26 = 130) :
    2 * A + 90 = 130 → A = 20 :=
by
  intros h
  linarith

end cricket_average_score_l35_35035


namespace gcd_proof_l35_35454

noncomputable def gcd_problem : Prop :=
  let a := 765432
  let b := 654321
  Nat.gcd a b = 111111

theorem gcd_proof : gcd_problem := by
  sorry

end gcd_proof_l35_35454


namespace distinct_triangles_from_tetrahedron_l35_35383

theorem distinct_triangles_from_tetrahedron (tetrahedron_vertices : Finset α)
  (h_tet : tetrahedron_vertices.card = 4) : 
  ∃ (triangles : Finset (Finset α)), triangles.card = 4 ∧ (∀ triangle ∈ triangles, triangle.card = 3 ∧ triangle ⊆ tetrahedron_vertices) :=
by
  -- Proof omitted
  sorry

end distinct_triangles_from_tetrahedron_l35_35383


namespace train_passing_time_l35_35700

/-- The problem defines a train of length 110 meters traveling at 40 km/hr, 
    passing a man who is running at 5 km/hr in the opposite direction.
    We want to prove that the time it takes for the train to pass the man is 8.8 seconds. -/
theorem train_passing_time :
  ∀ (train_length : ℕ) (train_speed man_speed : ℕ), 
  train_length = 110 → train_speed = 40 → man_speed = 5 →
  (∃ time : ℚ, time = 8.8) :=
by
  intros train_length train_speed man_speed h_train_length h_train_speed h_man_speed
  sorry

end train_passing_time_l35_35700


namespace factor_x4_minus_64_l35_35045

theorem factor_x4_minus_64 :
  ∀ x : ℝ, (x^4 - 64 = (x - 2 * Real.sqrt 2) * (x + 2 * Real.sqrt 2) * (x^2 + 8)) :=
by
  intro x
  sorry

end factor_x4_minus_64_l35_35045


namespace interest_payment_frequency_l35_35664

theorem interest_payment_frequency (i : ℝ) (EAR : ℝ) (n : ℕ)
  (h1 : i = 0.10) (h2 : EAR = 0.1025) :
  (1 + i / n)^n = 1 + EAR → n = 2 :=
by
  intros
  sorry

end interest_payment_frequency_l35_35664


namespace trigonometric_value_existence_l35_35990

noncomputable def can_be_value_of_tan (n : ℝ) : Prop :=
∃ θ : ℝ, Real.tan θ = n

noncomputable def can_be_value_of_cot (n : ℝ) : Prop :=
∃ θ : ℝ, 1 / Real.tan θ = n

def can_be_value_of_sin (n : ℝ) : Prop :=
|n| ≤ 1 ∧ ∃ θ : ℝ, Real.sin θ = n

def can_be_value_of_cos (n : ℝ) : Prop :=
|n| ≤ 1 ∧ ∃ θ : ℝ, Real.cos θ = n

def can_be_value_of_sec (n : ℝ) : Prop :=
|n| ≥ 1 ∧ ∃ θ : ℝ, 1 / Real.cos θ = n

def can_be_value_of_csc (n : ℝ) : Prop :=
|n| ≥ 1 ∧ ∃ θ : ℝ, 1 / Real.sin θ = n

theorem trigonometric_value_existence (n : ℝ) : 
  can_be_value_of_tan n ∧ 
  can_be_value_of_cot n ∧ 
  can_be_value_of_sin n ∧ 
  can_be_value_of_cos n ∧ 
  can_be_value_of_sec n ∧ 
  can_be_value_of_csc n := 
sorry

end trigonometric_value_existence_l35_35990


namespace remaining_pens_l35_35533

theorem remaining_pens (blue_initial black_initial red_initial green_initial purple_initial : ℕ)
                        (blue_removed black_removed red_removed green_removed purple_removed : ℕ) :
  blue_initial = 15 → black_initial = 27 → red_initial = 12 → green_initial = 10 → purple_initial = 8 →
  blue_removed = 8 → black_removed = 9 → red_removed = 3 → green_removed = 5 → purple_removed = 6 →
  blue_initial - blue_removed + black_initial - black_removed + red_initial - red_removed +
  green_initial - green_removed + purple_initial - purple_removed = 41 :=
by
  intros
  sorry

end remaining_pens_l35_35533


namespace problem1_solution_problem2_solution_l35_35110

-- Problem 1
theorem problem1_solution (x y : ℝ) (h1 : y = 2 * x - 3) (h2 : 2 * x + y = 5) : 
  x = 2 ∧ y = 1 :=
  sorry

-- Problem 2
theorem problem2_solution (x y : ℝ) (h1 : 3 * x + 4 * y = 5) (h2 : 5 * x - 2 * y = 17) : 
  x = 3 ∧ y = -1 :=
  sorry

end problem1_solution_problem2_solution_l35_35110


namespace range_of_a_l35_35156

def discriminant (a : ℝ) : ℝ := 4 * a^2 - 16
def P (a : ℝ) : Prop := discriminant a < 0
def Q (a : ℝ) : Prop := 5 - 2 * a > 1

theorem range_of_a (a : ℝ) (h1 : P a ∨ Q a) (h2 : ¬ (P a ∧ Q a)) : a ≤ -2 := by
  sorry

end range_of_a_l35_35156


namespace cookies_left_for_Monica_l35_35135

-- Definitions based on the conditions
def total_cookies : ℕ := 30
def father_cookies : ℕ := 10
def mother_cookies : ℕ := father_cookies / 2
def brother_cookies : ℕ := mother_cookies + 2

-- Statement for the theorem
theorem cookies_left_for_Monica : total_cookies - (father_cookies + mother_cookies + brother_cookies) = 8 := by
  -- The proof goes here
  sorry

end cookies_left_for_Monica_l35_35135


namespace david_presents_l35_35286

variables (C B E : ℕ)

def total_presents (C B E : ℕ) : ℕ := C + B + E

theorem david_presents :
  C = 60 →
  B = 3 * E →
  E = (C / 2) - 10 →
  total_presents C B E = 140 :=
by
  intros hC hB hE
  sorry

end david_presents_l35_35286


namespace smallest_y_square_l35_35919

theorem smallest_y_square (y n : ℕ) (h1 : y = 10) (h2 : ∀ m : ℕ, (∃ z : ℕ, m * y = z^2) ↔ (m = n)) : n = 10 :=
sorry

end smallest_y_square_l35_35919


namespace distance_points_lt_2_over_3_r_l35_35638

theorem distance_points_lt_2_over_3_r (r : ℝ) (h_pos_r : 0 < r) (points : Fin 17 → ℝ × ℝ)
  (h_points_in_circle : ∀ i, (points i).1 ^ 2 + (points i).2 ^ 2 < r ^ 2) :
  ∃ i j : Fin 17, i ≠ j ∧ (dist (points i) (points j) < 2 * r / 3) :=
by
  sorry

end distance_points_lt_2_over_3_r_l35_35638


namespace area_of_quadrilateral_EFGH_l35_35079

-- Define the properties of rectangle ABCD and the areas
def rectangle (A B C D : Type) := 
  ∃ (area : ℝ), area = 48

-- Define the positions of the points E, G, F, H
def points_positions (A D C B E G F H : Type) :=
  ∃ (one_third : ℝ) (two_thirds : ℝ), one_third = 1/3 ∧ two_thirds = 2/3

-- Define the area calculation for quadrilateral EFGH
def area_EFGH (area_ABCD : ℝ) (one_third : ℝ) : ℝ :=
  (one_third * one_third) * area_ABCD

-- The proof statement that area of EFGH is 5 1/3 square meters
theorem area_of_quadrilateral_EFGH 
  (A B C D E F G H : Type)
  (area_ABCD : ℝ)
  (one_third : ℝ) :
  rectangle A B C D →
  points_positions A D C B E G F H →
  area_ABCD = 48 →
  one_third = 1/3 →
  area_EFGH area_ABCD one_third = 16/3 :=
by
  intros h1 h2 h3 h4
  have h5 : area_EFGH area_ABCD one_third = 16/3 :=
  sorry
  exact h5

end area_of_quadrilateral_EFGH_l35_35079


namespace length_of_room_l35_35243

theorem length_of_room (L : ℝ) 
  (h_width : 12 > 0) 
  (h_veranda_width : 2 > 0) 
  (h_area_veranda : (L + 4) * 16 - L * 12 = 140) : 
  L = 19 := 
by
  sorry

end length_of_room_l35_35243


namespace arithmetic_sqrt_of_9_l35_35190

theorem arithmetic_sqrt_of_9 : ∃ y : ℝ, y ^ 2 = 9 ∧ y ≥ 0 ∧ y = 3 := by
  sorry

end arithmetic_sqrt_of_9_l35_35190


namespace min_value_problem_l35_35588

noncomputable def min_value (a b c d e f : ℝ) := (2 / a) + (3 / b) + (9 / c) + (16 / d) + (25 / e) + (36 / f)

theorem min_value_problem 
  (a b c d e f : ℝ)
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) (h5 : e > 0) (h6 : f > 0) 
  (h_sum : a + b + c + d + e + f = 10) : 
  min_value a b c d e f >= (329 + 38 * Real.sqrt 6) / 10 := 
sorry

end min_value_problem_l35_35588


namespace squirrel_rise_per_circuit_l35_35225

noncomputable def rise_per_circuit
    (height : ℕ)
    (circumference : ℕ)
    (distance : ℕ) :=
    height / (distance / circumference)

theorem squirrel_rise_per_circuit : rise_per_circuit 25 3 15 = 5 :=
by
  sorry

end squirrel_rise_per_circuit_l35_35225


namespace num_integers_for_polynomial_negative_l35_35603

open Int

theorem num_integers_for_polynomial_negative :
  ∃ (set_x : Finset ℤ), set_x.card = 12 ∧ ∀ x ∈ set_x, (x^4 - 65 * x^2 + 64) < 0 :=
by
  sorry

end num_integers_for_polynomial_negative_l35_35603


namespace evaluate_f_5_minus_f_neg_5_l35_35222

def f (x : ℝ) : ℝ := x^4 + x^2 + 5 * x + 3

theorem evaluate_f_5_minus_f_neg_5 : f 5 - f (-5) = 50 := 
  by
    sorry

end evaluate_f_5_minus_f_neg_5_l35_35222


namespace chase_cardinals_count_l35_35396

variable (gabrielle_robins : Nat)
variable (gabrielle_cardinals : Nat)
variable (gabrielle_blue_jays : Nat)
variable (chase_robins : Nat)
variable (chase_blue_jays : Nat)
variable (chase_cardinals : Nat)

variable (gabrielle_total : Nat)
variable (chase_total : Nat)

variable (percent_more : Nat)

axiom gabrielle_robins_def : gabrielle_robins = 5
axiom gabrielle_cardinals_def : gabrielle_cardinals = 4
axiom gabrielle_blue_jays_def : gabrielle_blue_jays = 3

axiom chase_robins_def : chase_robins = 2
axiom chase_blue_jays_def : chase_blue_jays = 3

axiom gabrielle_total_def : gabrielle_total = gabrielle_robins + gabrielle_cardinals + gabrielle_blue_jays
axiom chase_total_def : chase_total = chase_robins + chase_blue_jays + chase_cardinals
axiom percent_more_def : percent_more = 20

axiom gabrielle_more_birds : gabrielle_total = Nat.ceil ((chase_total * (100 + percent_more)) / 100)

theorem chase_cardinals_count : chase_cardinals = 5 := by sorry

end chase_cardinals_count_l35_35396


namespace difference_in_total_cost_l35_35782

theorem difference_in_total_cost
  (item_price : ℝ := 15)
  (tax_rate1 : ℝ := 0.08)
  (tax_rate2 : ℝ := 0.072)
  (discount : ℝ := 0.005)
  (correct_difference : ℝ := 0.195) :
  let discounted_tax_rate := tax_rate2 - discount
  let total_price_with_tax_rate1 := item_price * (1 + tax_rate1)
  let total_price_with_discounted_tax_rate := item_price * (1 + discounted_tax_rate)
  total_price_with_tax_rate1 - total_price_with_discounted_tax_rate = correct_difference := by
  sorry

end difference_in_total_cost_l35_35782


namespace one_angle_not_greater_than_60_l35_35274

theorem one_angle_not_greater_than_60 (A B C : ℝ) (h : A + B + C = 180) : A ≤ 60 ∨ B ≤ 60 ∨ C ≤ 60 := 
sorry

end one_angle_not_greater_than_60_l35_35274


namespace xyz_product_neg4_l35_35827

theorem xyz_product_neg4 (x y z : ℝ) (h1 : x + 2 / y = 2) (h2 : y + 2 / z = 2) : x * y * z = -4 :=
by {
  sorry
}

end xyz_product_neg4_l35_35827


namespace sampling_interval_is_100_l35_35517

-- Define the total number of numbers (N), the number of samples to be taken (k), and the condition for systematic sampling.
def N : ℕ := 2005
def k : ℕ := 20

-- Define the concept of systematic sampling interval
def sampling_interval (N k : ℕ) : ℕ := N / k

-- The proof that the sampling interval is 100 when 2005 numbers are sampled as per the systematic sampling method.
theorem sampling_interval_is_100 (N k : ℕ) 
  (hN : N = 2005) 
  (hk : k = 20) 
  (h1 : N % k ≠ 0) : 
  sampling_interval (N - (N % k)) k = 100 :=
by
  -- Initialization
  sorry

end sampling_interval_is_100_l35_35517


namespace find_total_cows_l35_35445

-- Definitions as per the conditions
variables (D C L H : ℕ)

-- Condition 1: Total number of legs
def total_legs : ℕ := 2 * D + 4 * C

-- Condition 2: Total number of heads
def total_heads : ℕ := D + C

-- Condition 3: Legs are 28 more than twice the number of heads
def legs_heads_relation : Prop := total_legs D C = 2 * total_heads D C + 28

-- The theorem to prove
theorem find_total_cows (h : legs_heads_relation D C) : C = 14 :=
sorry

end find_total_cows_l35_35445


namespace length_of_AX_l35_35453

theorem length_of_AX 
  (A B C X : Type) 
  (AB AC BC AX BX : ℕ) 
  (hx : AX + BX = AB)
  (h_angle_bisector : AC * BX = BC * AX)
  (h_AB : AB = 40)
  (h_BC : BC = 35)
  (h_AC : AC = 21) : 
  AX = 15 :=
by
  sorry

end length_of_AX_l35_35453


namespace number_of_cows_l35_35287

-- Define conditions
def total_bags_consumed_by_some_cows := 45
def bags_consumed_by_one_cow := 1

-- State the theorem to prove the number of cows
theorem number_of_cows (h1 : total_bags_consumed_by_some_cows = 45) (h2 : bags_consumed_by_one_cow = 1) : 
  total_bags_consumed_by_some_cows / bags_consumed_by_one_cow = 45 :=
by
  -- Proof goes here
  sorry

end number_of_cows_l35_35287


namespace maddie_watched_138_on_monday_l35_35621

-- Define the constants and variables from the problem statement
def total_episodes : ℕ := 8
def minutes_per_episode : ℕ := 44
def watched_thursday : ℕ := 21
def watched_friday_episodes : ℕ := 2
def watched_weekend : ℕ := 105

-- Calculate the total minutes watched from all episodes
def total_minutes : ℕ := total_episodes * minutes_per_episode

-- Calculate the minutes watched on Friday
def watched_friday : ℕ := watched_friday_episodes * minutes_per_episode

-- Calculate the total minutes watched on weekdays excluding Monday
def watched_other_days : ℕ := watched_thursday + watched_friday + watched_weekend

-- Statement to prove that Maddie watched 138 minutes on Monday
def minutes_watched_on_monday : ℕ := total_minutes - watched_other_days

-- The final statement for proof in Lean 4
theorem maddie_watched_138_on_monday : minutes_watched_on_monday = 138 := by
  -- This theorem should be proved using the above definitions and calculations, proof skipped with sorry
  sorry

end maddie_watched_138_on_monday_l35_35621


namespace number_of_lines_l35_35284

-- Define points A and B
def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (3, 1)

-- Define the distances from the points
def d_A : ℝ := 1
def d_B : ℝ := 2

-- A theorem stating the number of lines under the given conditions
theorem number_of_lines (A B : ℝ × ℝ) (d_A d_B : ℝ) (hA : A = (1, 2)) (hB : B = (3, 1)) (hdA : d_A = 1) (hdB : d_B = 2) :
  ∃ n : ℕ, n = 2 :=
by {
  sorry
}

end number_of_lines_l35_35284


namespace cistern_fill_time_l35_35516

theorem cistern_fill_time (F E : ℝ) (hF : F = 1/3) (hE : E = 1/6) : (1 / (F - E)) = 6 :=
by sorry

end cistern_fill_time_l35_35516


namespace tangent_length_to_circle_l35_35671

-- Definitions capturing the conditions
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4 * x - 2 * y + 1 = 0
def line_l (x y a : ℝ) : Prop := x + a * y - 1 = 0
def point_A (a : ℝ) : ℝ × ℝ := (-4, a)

-- Main theorem statement proving the question against the answer
theorem tangent_length_to_circle (a : ℝ) (x y : ℝ) (hC : circle_C x y) (hl : line_l 2 1 a) :
  (a = -1) -> (point_A a = (-4, -1)) -> ∃ b : ℝ, b = 6 := 
sorry

end tangent_length_to_circle_l35_35671


namespace flooring_sq_ft_per_box_l35_35492

/-- The problem statement converted into a Lean theorem -/
theorem flooring_sq_ft_per_box
  (living_room_length : ℕ)
  (living_room_width : ℕ)
  (flooring_installed : ℕ)
  (additional_boxes : ℕ)
  (correct_answer : ℕ) 
  (h1 : living_room_length = 16)
  (h2 : living_room_width = 20)
  (h3 : flooring_installed = 250)
  (h4 : additional_boxes = 7)
  (h5 : correct_answer = 10) :
  
  (living_room_length * living_room_width - flooring_installed) / additional_boxes = correct_answer :=
by 
  sorry

end flooring_sq_ft_per_box_l35_35492


namespace smallest_int_neither_prime_nor_square_no_prime_lt_70_l35_35572

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def has_no_prime_factor_less_than (n : ℕ) (k : ℕ) : Prop :=
  ∀ p : ℕ, is_prime p → p < k → ¬ p ∣ n

theorem smallest_int_neither_prime_nor_square_no_prime_lt_70
  (n : ℕ) : 
  n = 5183 ∧ ¬ is_prime n ∧ ¬ is_square n ∧ has_no_prime_factor_less_than n 70 ∧
  (∀ m : ℕ, 0 < m → m < 5183 →
    ¬ (¬ is_prime m ∧ ¬ is_square m ∧ has_no_prime_factor_less_than m 70)) :=
by sorry

end smallest_int_neither_prime_nor_square_no_prime_lt_70_l35_35572


namespace number_of_solutions_l35_35746

theorem number_of_solutions (x y: ℕ) (hx : 0 < x) (hy : 0 < y) :
    (1 / (x + 1) + 1 / y + 1 / ((x + 1) * y) = 1 / 1991) →
    ∃! (n : ℕ), n = 64 :=
by
  sorry

end number_of_solutions_l35_35746


namespace M_diff_N_l35_35341

def A : Set ℝ := sorry
def B : Set ℝ := sorry

def M := {x : ℝ | -3 ≤ x ∧ x ≤ 1}
def N := {y : ℝ | ∃ x : ℝ, y = x^2 ∧ -1 ≤ x ∧ x ≤ 1}

-- Definition of set subtraction
def set_diff (A B : Set ℝ) := {x : ℝ | x ∈ A ∧ x ∉ B}

-- Given problem statement
theorem M_diff_N : set_diff M N = {x : ℝ | -3 ≤ x ∧ x < 0} := 
by
  sorry

end M_diff_N_l35_35341


namespace find_f_2016_l35_35040

noncomputable def f : ℝ → ℝ :=
sorry

axiom f_0_eq_2016 : f 0 = 2016

axiom f_x_plus_2_minus_f_x_leq : ∀ x : ℝ, f (x + 2) - f x ≤ 3 * 2 ^ x

axiom f_x_plus_6_minus_f_x_geq : ∀ x : ℝ, f (x + 6) - f x ≥ 63 * 2 ^ x

theorem find_f_2016 : f 2016 = 2015 + 2 ^ 2020 :=
sorry

end find_f_2016_l35_35040


namespace number_of_truthful_dwarfs_l35_35987

def total_dwarfs := 10
def hands_raised_vanilla := 10
def hands_raised_chocolate := 5
def hands_raised_fruit := 1
def total_hands_raised := hands_raised_vanilla + hands_raised_chocolate + hands_raised_fruit
def extra_hands := total_hands_raised - total_dwarfs
def liars := extra_hands
def truthful := total_dwarfs - liars

theorem number_of_truthful_dwarfs : truthful = 4 :=
by sorry

end number_of_truthful_dwarfs_l35_35987


namespace possible_values_a_l35_35292

-- Define the problem statement
theorem possible_values_a :
  (∃ a b c : ℤ, ∀ x : ℝ, (x - a) * (x - 5) + 3 = (x + b) * (x + c)) → (a = 1 ∨ a = 9) :=
by 
  -- Variable declaration and theorem body will be placed here
  sorry

end possible_values_a_l35_35292


namespace correct_weight_misread_l35_35471

theorem correct_weight_misread : 
  ∀ (x : ℝ) (n : ℝ) (avg1 : ℝ) (avg2 : ℝ) (misread : ℝ),
  n = 20 → avg1 = 58.4 → avg2 = 59 → misread = 56 → 
  (n * avg2 - n * avg1 + misread) = x → 
  x = 68 :=
by
  intros x n avg1 avg2 misread
  intros h1 h2 h3 h4 h5
  sorry

end correct_weight_misread_l35_35471


namespace trapezoid_area_l35_35551

theorem trapezoid_area (AD BC AC : ℝ) (BD : ℝ) 
  (hAD : AD = 24) 
  (hBC : BC = 8) 
  (hAC : AC = 13) 
  (hBD : BD = 5 * Real.sqrt 17) : 
  (1 / 2 * (AD + BC) * Real.sqrt (AC^2 - (BC + (AD - BC) / 2)^2)) = 80 :=
by
  sorry

end trapezoid_area_l35_35551


namespace total_surface_area_of_prism_l35_35703

-- Define the conditions of the problem
def sphere_radius (R : ℝ) := R > 0
def prism_circumscribed_around_sphere (R : ℝ) := True  -- Placeholder as the concept assertion, actual geometry handling not needed here
def prism_height (R : ℝ) := 2 * R

-- Define the main theorem to be proved
theorem total_surface_area_of_prism (R : ℝ) (hR : sphere_radius R) (hCircumscribed : prism_circumscribed_around_sphere R) (hHeight : prism_height R = 2 * R) : 
  ∃ (S : ℝ), S = 12 * R^2 * Real.sqrt 3 :=
sorry

end total_surface_area_of_prism_l35_35703


namespace find_a5_l35_35582

variable {a : ℕ → ℝ} {q : ℝ}

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop := 
  ∀ n, a (n+1) = a n * q

def is_increasing_sequence (a : ℕ → ℝ) : Prop := 
  ∀ n, a n < a (n+1)

def condition1 (a : ℕ → ℝ) : Prop :=
  a 5 ^ 2 = a 10

def condition2 (a : ℕ → ℝ) : Prop :=
  ∀ n, 2 * (a n + a (n+2)) = 5 * a (n+1)

theorem find_a5 (h1 : is_geometric_sequence a q) (h2 : is_increasing_sequence a) (h3 : condition1 a) (h4 : condition2 a) : 
  a 5 = 32 :=
sorry

end find_a5_l35_35582


namespace shaded_area_of_circles_l35_35211

theorem shaded_area_of_circles :
  let R := 10
  let r1 := R / 2
  let r2 := R / 2
  (π * R^2 - (π * r1^2 + π * r1^2 + π * r2^2)) = 25 * π :=
by
  sorry

end shaded_area_of_circles_l35_35211


namespace ratio_of_areas_of_concentric_circles_l35_35216

theorem ratio_of_areas_of_concentric_circles
  (Q : Type)
  (r₁ r₂ : ℝ)
  (C₁ C₂ : ℝ)
  (h₀ : r₁ > 0 ∧ r₂ > 0)
  (h₁ : C₁ = 2 * π * r₁)
  (h₂ : C₂ = 2 * π * r₂)
  (h₃ : (60 / 360) * C₁ = (30 / 360) * C₂) :
  (π * r₁^2) / (π * r₂^2) = 1 / 4 :=
by
  sorry

end ratio_of_areas_of_concentric_circles_l35_35216


namespace train_speed_in_km_hr_l35_35719

-- Definitions based on conditions
def train_length : ℝ := 150  -- meters
def crossing_time : ℝ := 6  -- seconds

-- Definition for conversion factor
def meters_per_second_to_km_per_hour (speed_mps : ℝ) : ℝ := speed_mps * 3.6

-- Main theorem
theorem train_speed_in_km_hr : meters_per_second_to_km_per_hour (train_length / crossing_time) = 90 :=
by
  sorry

end train_speed_in_km_hr_l35_35719


namespace simplify_complex_fraction_l35_35444

theorem simplify_complex_fraction :
  (⟨3, 5⟩ : ℂ) / (⟨-2, 7⟩ : ℂ) = (29 / 53) - (31 / 53) * I :=
by sorry

end simplify_complex_fraction_l35_35444


namespace distance_between_trees_l35_35726

theorem distance_between_trees (n : ℕ) (len : ℝ) (d : ℝ) 
  (h1 : n = 26) 
  (h2 : len = 400) 
  (h3 : len / (n - 1) = d) : 
  d = 16 :=
by
  sorry

end distance_between_trees_l35_35726


namespace original_game_start_player_wins_modified_game_start_player_wins_l35_35529

def divisor_game_condition (num : ℕ) := ∀ d : ℕ, d ∣ num → ∀ x : ℕ, x ∣ d → x = d ∨ x = 1
def modified_divisor_game_condition (num d_prev : ℕ) := ∀ d : ℕ, d ∣ num → d ≠ d_prev → ∃ k l : ℕ, d = k * l ∧ k ≠ 1 ∧ l ≠ 1 ∧ k ≤ l

/-- Prove that if the starting player plays wisely, they will always win the original game. -/
theorem original_game_start_player_wins : ∀ d : ℕ, divisor_game_condition 1000 → d = 100 → (∃ p : ℕ, p != 1000) := 
sorry

/-- What happens if the game is modified such that a divisor cannot be mentioned if it has fewer divisors than any previously mentioned number? -/
theorem modified_game_start_player_wins : ∀ d_prev : ℕ, modified_divisor_game_condition 1000 d_prev → d_prev = 100 → (∃ p : ℕ, p != 1000) := 
sorry

end original_game_start_player_wins_modified_game_start_player_wins_l35_35529


namespace xy_solutions_l35_35569

theorem xy_solutions : 
  ∀ (x y : ℕ), 0 < x → 0 < y →
  (xy ^ 2 + 7) ∣ (x^2 * y + x) →
  (x, y) = (7, 1) ∨ (x, y) = (14, 1) ∨ (x, y) = (35, 1) ∨ (x, y) = (7, 2) ∨ (∃ k : ℕ, x = 7 * k ∧ y = 7) :=
by
  sorry

end xy_solutions_l35_35569


namespace max_marks_tests_l35_35559

theorem max_marks_tests :
  ∃ (T1 T2 T3 T4 : ℝ),
    0.30 * T1 = 80 + 40 ∧
    0.40 * T2 = 105 + 35 ∧
    0.50 * T3 = 150 + 50 ∧
    0.60 * T4 = 180 + 60 ∧
    T1 = 400 ∧
    T2 = 350 ∧
    T3 = 400 ∧
    T4 = 400 :=
by
    sorry

end max_marks_tests_l35_35559


namespace desired_interest_rate_l35_35620

theorem desired_interest_rate 
  (F : ℝ) -- Face value of each share
  (D : ℝ) -- Dividend rate
  (M : ℝ) -- Market value of each share
  (annual_dividend : ℝ := (D / 100) * F) -- Annual dividend per share
  (desired_interest_rate : ℝ := (annual_dividend / M) * 100) -- Desired interest rate
  (F_eq : F = 44) -- Given Face value
  (D_eq : D = 9) -- Given Dividend rate
  (M_eq : M = 33) -- Given Market value
  : desired_interest_rate = 12 := 
by
  sorry

end desired_interest_rate_l35_35620


namespace correct_mean_of_values_l35_35347

variable (n : ℕ) (mu_incorrect : ℝ) (incorrect_value : ℝ) (correct_value : ℝ) (mu_correct : ℝ)

theorem correct_mean_of_values
  (h1 : n = 30)
  (h2 : mu_incorrect = 150)
  (h3 : incorrect_value = 135)
  (h4 : correct_value = 165)
  : mu_correct = 151 :=
by
  let S_incorrect := mu_incorrect * n
  let S_correct := S_incorrect - incorrect_value + correct_value
  let mu_correct := S_correct / n
  sorry

end correct_mean_of_values_l35_35347


namespace simplify_fraction_l35_35954

open Complex

theorem simplify_fraction :
  (7 + 9 * I) / (3 - 4 * I) = 2.28 + 2.2 * I := 
by {
    -- We know that this should be true based on the provided solution,
    -- but we will place a placeholder here for the actual proof.
    sorry
}

end simplify_fraction_l35_35954


namespace plastic_bag_estimation_l35_35325

theorem plastic_bag_estimation (a b c d e f : ℕ) (class_size : ℕ) (h1 : a = 33) 
  (h2 : b = 25) (h3 : c = 28) (h4 : d = 26) (h5 : e = 25) (h6 : f = 31) (h_class_size : class_size = 45) :
  let count := a + b + c + d + e + f
  let average := count / 6
  average * class_size = 1260 := by
{ 
  sorry 
}

end plastic_bag_estimation_l35_35325


namespace exists_abc_gcd_equation_l35_35036

theorem exists_abc_gcd_equation (n : ℕ) : ∃ a b c : ℤ, n = Int.gcd a b * (c^2 - a*b) + Int.gcd b c * (a^2 - b*c) + Int.gcd c a * (b^2 - c*a) := sorry

end exists_abc_gcd_equation_l35_35036


namespace elberta_money_l35_35840

theorem elberta_money (GrannySmith Anjou Elberta : ℝ)
  (h_granny : GrannySmith = 100)
  (h_anjou : Anjou = 1 / 4 * GrannySmith)
  (h_elberta : Elberta = Anjou + 5) : Elberta = 30 := by
  sorry

end elberta_money_l35_35840


namespace time_wandered_l35_35351

-- Definitions and Hypotheses
def distance : ℝ := 4
def speed : ℝ := 2

-- Proof statement
theorem time_wandered : distance / speed = 2 := by
  sorry

end time_wandered_l35_35351


namespace oreo_solution_l35_35761

noncomputable def oreo_problem : Prop :=
∃ (m : ℤ), (11 + m * 11 + 3 = 36) → m = 2

theorem oreo_solution : oreo_problem :=
sorry

end oreo_solution_l35_35761


namespace sum_divisible_by_ten_l35_35530

    -- Given conditions
    def is_natural_number (n : ℕ) : Prop := true

    -- Sum S as defined in the conditions
    def S (n : ℕ) : ℕ := n ^ 2 + (n + 1) ^ 2 + (n + 2) ^ 2 + (n + 3) ^ 2

    -- The equivalent math proof problem in Lean 4 statement
    theorem sum_divisible_by_ten (n : ℕ) : S n % 10 = 0 ↔ n % 5 = 1 := by
      sorry
    
end sum_divisible_by_ten_l35_35530


namespace range_of_set_l35_35277

-- Define the conditions
variable (a b c : ℝ)
variable (h1 : a ≤ b)
variable (h2 : b ≤ c)
variable (mean_condition : (a + b + c) / 3 = 5)
variable (median_condition : b = 5)
variable (smallest_condition : a = 2)

-- The theorem to prove
theorem range_of_set : c - a = 6 := by
  sorry

end range_of_set_l35_35277


namespace find_a_value_l35_35519

theorem find_a_value (a : ℤ) (h1 : 0 < a) (h2 : a < 13) (h3 : (53^2017 + a) % 13 = 0) : a = 12 :=
by
  -- proof steps
  sorry

end find_a_value_l35_35519


namespace largest_value_x_l35_35091

-- Definition of the conditions
def equation (x : ℚ) : Prop :=
  (16 * x^2 - 40 * x + 15) / (4 * x - 3) + 7 * x = 8 * x - 2

-- Statement of the proof 
theorem largest_value_x : ∀ x : ℚ, equation x → x ≤ 9 / 4 := sorry

end largest_value_x_l35_35091


namespace price_on_friday_is_correct_l35_35655

-- Define initial price on Tuesday
def price_on_tuesday : ℝ := 50

-- Define the percentage increase on Wednesday (20%)
def percentage_increase : ℝ := 0.20

-- Define the percentage discount on Friday (15%)
def percentage_discount : ℝ := 0.15

-- Define the price on Wednesday after the increase
def price_on_wednesday : ℝ := price_on_tuesday * (1 + percentage_increase)

-- Define the price on Friday after the discount
def price_on_friday : ℝ := price_on_wednesday * (1 - percentage_discount)

-- Theorem statement to prove that the price on Friday is 51 dollars
theorem price_on_friday_is_correct : price_on_friday = 51 :=
by
  sorry

end price_on_friday_is_correct_l35_35655


namespace min_value_of_expression_l35_35631

/-- Given the area of △ ABC is 2, and the sides opposite to angles A, B, C are a, b, c respectively,
    prove that the minimum value of a^2 + 2b^2 + 3c^2 is 8 * sqrt(11). -/
theorem min_value_of_expression
  (a b c : ℝ)
  (h₁ : 1/2 * b * c * Real.sin A = 2) :
  a^2 + 2 * b^2 + 3 * c^2 ≥ 8 * Real.sqrt 11 :=
sorry

end min_value_of_expression_l35_35631


namespace three_digit_number_unchanged_upside_down_l35_35844

theorem three_digit_number_unchanged_upside_down (n : ℕ) :
  (n >= 100 ∧ n <= 999) ∧ (∀ d ∈ [n / 100, (n / 10) % 10, n % 10], d = 0 ∨ d = 8) ->
  n = 888 ∨ n = 808 :=
by
  sorry

end three_digit_number_unchanged_upside_down_l35_35844


namespace max_expression_value_l35_35944

theorem max_expression_value : 
  ∃ a b c d e f : ℕ, 1 ≤ a ∧ a ≤ 6 ∧
                   1 ≤ b ∧ b ≤ 6 ∧
                   1 ≤ c ∧ c ≤ 6 ∧
                   1 ≤ d ∧ d ≤ 6 ∧
                   1 ≤ e ∧ e ≤ 6 ∧
                   1 ≤ f ∧ f ≤ 6 ∧
                   a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
                   b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
                   c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
                   d ≠ e ∧ d ≠ f ∧
                   e ≠ f ∧
                   (f * (a * d + b * c) / (b * d * e) = 14) :=
sorry

end max_expression_value_l35_35944


namespace find_new_ratio_l35_35536

def initial_ratio (H C : ℕ) : Prop := H = 6 * C

def transaction (H C : ℕ) : Prop :=
  H - 15 = (C + 15) + 70

def new_ratio (H C : ℕ) : Prop := H - 15 = 3 * (C + 15)

theorem find_new_ratio (H C : ℕ) (h1 : initial_ratio H C) (h2 : transaction H C) : 
  new_ratio H C :=
sorry

end find_new_ratio_l35_35536


namespace maximum_value_l35_35227

noncomputable def p : ℝ := 1 + 1/2 + 1/2^2 + 1/2^3 + 1/2^4 + 1/2^5

theorem maximum_value (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) 
  (h_constraint : (x - 1)^2 + (y - 1)^2 + (z - 1)^2 = 27) : 
  x^p + y^p + z^p ≤ 40.4 :=
sorry

end maximum_value_l35_35227


namespace parabola_x_intercepts_count_l35_35501

theorem parabola_x_intercepts_count :
  ∃! x, ∃ y, x = -3 * y^2 + 2 * y + 2 ∧ y = 0 :=
by
  sorry

end parabola_x_intercepts_count_l35_35501


namespace arithmetic_sequence_sum_l35_35388

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : ∀ n, S n = n * (a 1 + a n) / 2)
  (h2 : a 5 + a 6 = 18) :
  S 10 = 90 :=
sorry

end arithmetic_sequence_sum_l35_35388


namespace length_of_LM_l35_35345

-- Definitions of the conditions
variable (P Q R L M : Type)
variable (b : Real) (PR_area : Real) (PR_base : Real)
variable (PR_base_eq : PR_base = 15)
variable (crease_parallel : Parallel L M)
variable (projected_area_fraction : Real)
variable (projected_area_fraction_eq : projected_area_fraction = 0.25 * PR_area)

-- Theorem statement to prove the length of LM
theorem length_of_LM : ∀ (LM_length : Real), (LM_length = 7.5) :=
sorry

end length_of_LM_l35_35345


namespace train_length_l35_35268

theorem train_length
    (V : ℝ) -- train speed in m/s
    (L : ℝ) -- length of the train in meters
    (H1 : L = V * 18) -- condition: train crosses signal pole in 18 sec
    (H2 : L + 333.33 = V * 38) -- condition: train crosses platform in 38 sec
    (V_pos : 0 < V) -- additional condition: speed must be positive
    : L = 300 :=
by
-- here goes the proof which is not required for our task
sorry

end train_length_l35_35268


namespace coprime_count_l35_35442

theorem coprime_count (n : ℕ) (h : n = 56700000) : 
  ∃ m, m = 12960000 ∧ ∀ i < n, Nat.gcd i n = 1 → i < m :=
by
  sorry

end coprime_count_l35_35442


namespace amount_left_after_spending_l35_35813

-- Define the initial amount and percentage spent
def initial_amount : ℝ := 500
def percentage_spent : ℝ := 0.30

-- Define the proof statement that the amount left is 350
theorem amount_left_after_spending : 
  (initial_amount - (percentage_spent * initial_amount)) = 350 :=
by
  sorry

end amount_left_after_spending_l35_35813


namespace intersection_equiv_l35_35154

open Set

def A : Set ℝ := { x | 2 * x < 2 + x }
def B : Set ℝ := { x | 5 - x > 8 - 4 * x }

theorem intersection_equiv : A ∩ B = { x : ℝ | 1 < x ∧ x < 2 } := 
by 
  sorry

end intersection_equiv_l35_35154


namespace bc_over_ad_l35_35116

-- Define the rectangular prism
structure RectangularPrism :=
(length width height : ℝ)

-- Define the problem parameters
def B : RectangularPrism := ⟨2, 4, 5⟩

-- Define the volume form of S(r)
def volume (a b c d : ℝ) (r : ℝ) : ℝ := a * r^3 + b * r^2 + c * r + d

-- Prove that the relationship holds
theorem bc_over_ad (a b c d : ℝ) (r : ℝ) (h_a : a = (4 * π) / 3) (h_b : b = 11 * π) (h_c : c = 76) (h_d : d = 40) :
  (b * c) / (a * d) = 15.67 := by
  sorry

end bc_over_ad_l35_35116


namespace carrie_first_day_miles_l35_35585

theorem carrie_first_day_miles
  (x : ℕ)
  (h1 : ∀ y : ℕ, y = x + 124) -- Second day
  (h2 : ∀ y : ℕ, y = 159) -- Third day
  (h3 : ∀ y : ℕ, y = 189) -- Fourth day
  (h4 : ∀ z : ℕ, z = 106) -- Phone charge interval
  (h5 : ∀ n : ℕ, n = 7) -- Number of charges
  (h_total : 106 * 7 = x + (x + 124) + 159 + 189)
  : x = 135 :=
by sorry

end carrie_first_day_miles_l35_35585


namespace best_fitting_model_l35_35300

theorem best_fitting_model :
  ∀ R1 R2 R3 R4 : ℝ, 
  R1 = 0.21 → R2 = 0.80 → R3 = 0.50 → R4 = 0.98 → 
  abs (R4 - 1) < abs (R1 - 1) ∧ abs (R4 - 1) < abs (R2 - 1) 
    ∧ abs (R4 - 1) < abs (R3 - 1) :=
by
  intros R1 R2 R3 R4 hR1 hR2 hR3 hR4
  rw [hR1, hR2, hR3, hR4]
  exact sorry

end best_fitting_model_l35_35300


namespace handshake_count_l35_35643

theorem handshake_count (couples : ℕ) (people : ℕ) (total_handshakes : ℕ) :
  couples = 6 →
  people = 2 * couples →
  total_handshakes = (people * (people - 1)) / 2 - couples →
  total_handshakes = 60 :=
by
  intros h_couples h_people h_handshakes
  sorry

end handshake_count_l35_35643


namespace choose_most_suitable_l35_35327

def Survey := ℕ → Bool
structure Surveys :=
  (A B C D : Survey)
  (census_suitable : Survey)

theorem choose_most_suitable (s : Surveys) :
  s.census_suitable = s.C :=
sorry

end choose_most_suitable_l35_35327


namespace intersection_points_of_segments_l35_35366

noncomputable def num_intersection_points (A B C : Point) (P : Fin 60 → Point) (Q : Fin 50 → Point) : ℕ :=
  3000

theorem intersection_points_of_segments (A B C : Point) (P : Fin 60 → Point) (Q : Fin 50 → Point) :
  num_intersection_points A B C P Q = 3000 :=
  by sorry

end intersection_points_of_segments_l35_35366


namespace strawberries_weight_l35_35821

theorem strawberries_weight (total_weight apples_weight oranges_weight grapes_weight strawberries_weight : ℕ) 
  (h_total : total_weight = 10)
  (h_apples : apples_weight = 3)
  (h_oranges : oranges_weight = 1)
  (h_grapes : grapes_weight = 3) 
  (h_sum : total_weight = apples_weight + oranges_weight + grapes_weight + strawberries_weight) :
  strawberries_weight = 3 :=
by
  sorry

end strawberries_weight_l35_35821


namespace triangle_is_isosceles_l35_35257

theorem triangle_is_isosceles 
  (a b c : ℝ)
  (h : a^2 - b^2 + a * c - b * c = 0)
  (h_tri : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a)
  : a = b := 
sorry

end triangle_is_isosceles_l35_35257


namespace restaurant_bill_l35_35645

theorem restaurant_bill 
  (salisbury_steak : ℝ := 16.00)
  (chicken_fried_steak : ℝ := 18.00)
  (mozzarella_sticks : ℝ := 8.00)
  (caesar_salad : ℝ := 6.00)
  (bowl_chili : ℝ := 7.00)
  (chocolate_lava_cake : ℝ := 7.50)
  (cheesecake : ℝ := 6.50)
  (iced_tea : ℝ := 3.00)
  (soda : ℝ := 3.50)
  (half_off_meal : ℝ := 0.5)
  (dessert_discount : ℝ := 0.1)
  (tip_percent : ℝ := 0.2)
  (sales_tax : ℝ := 0.085) :
  let total : ℝ :=
    (salisbury_steak * half_off_meal) +
    (chicken_fried_steak * half_off_meal) +
    mozzarella_sticks +
    caesar_salad +
    bowl_chili +
    (chocolate_lava_cake * (1 - dessert_discount)) +
    (cheesecake * (1 - dessert_discount)) +
    iced_tea +
    soda
  let total_with_tax : ℝ := total * (1 + sales_tax)
  let final_total : ℝ := total_with_tax * (1 + tip_percent)
  final_total = 73.04 :=
by
  sorry

end restaurant_bill_l35_35645


namespace two_distinct_solutions_diff_l35_35041

theorem two_distinct_solutions_diff (a b : ℝ) (h1 : a ≠ b) (h2 : a > b)
  (h3 : ∀ x, (x = a ∨ x = b) ↔ (6 * x - 18) / (x^2 + 3 * x - 18) = x + 3) :
  a - b = 3 :=
by
  -- Proof will be provided here.
  sorry

end two_distinct_solutions_diff_l35_35041


namespace ratio_of_years_taught_l35_35982

-- Definitions based on given conditions
def C : ℕ := 4
def A : ℕ := 2 * C
def total_years (S : ℕ) : Prop := C + A + S = 52

-- Proof statement
theorem ratio_of_years_taught (S : ℕ) (h : total_years S) : 
  S / A = 5 / 1 :=
by
  sorry

end ratio_of_years_taught_l35_35982


namespace part1_part2_l35_35183

-- Define the complex number z in terms of m
def z (m : ℝ) : ℂ := (2 * m^2 - 3 * m - 2) + (m^2 - 3 * m + 2) * Complex.I

-- State the condition where z is a purely imaginary number
def purelyImaginary (m : ℝ) : Prop := 2 * m^2 - 3 * m - 2 = 0 ∧ m^2 - 3 * m + 2 ≠ 0

-- State the condition where z is in the second quadrant.
def inSecondQuadrant (m : ℝ) : Prop := 2 * m^2 - 3 * m - 2 < 0 ∧ m^2 - 3 * m + 2 > 0

-- Part 1: Prove that m = -1/2 given that z is purely imaginary.
theorem part1 : purelyImaginary m → m = -1/2 :=
sorry

-- Part 2: Prove the range of m for z in the second quadrant.
theorem part2 : inSecondQuadrant m → -1/2 < m ∧ m < 1 :=
sorry

end part1_part2_l35_35183


namespace shopper_saved_percentage_l35_35774

-- Definition of the problem conditions
def amount_saved : ℝ := 4
def amount_spent : ℝ := 36

-- Lean 4 statement to prove the percentage saved
theorem shopper_saved_percentage : (amount_saved / (amount_spent + amount_saved)) * 100 = 10 := by
  sorry

end shopper_saved_percentage_l35_35774


namespace price_reduction_is_not_10_yuan_l35_35514

theorem price_reduction_is_not_10_yuan (current_price original_price : ℝ)
  (CurrentPrice : current_price = 45)
  (Reduction : current_price = 0.9 * original_price)
  (TenPercentReduction : 0.1 * original_price = 10) :
  (original_price - current_price) ≠ 10 := by
  sorry

end price_reduction_is_not_10_yuan_l35_35514


namespace danny_bottle_caps_after_collection_l35_35773

-- Definitions for the conditions
def initial_bottle_caps : ℕ := 69
def bottle_caps_thrown : ℕ := 60
def bottle_caps_found : ℕ := 58

-- Theorem stating the proof problem
theorem danny_bottle_caps_after_collection : 
  initial_bottle_caps - bottle_caps_thrown + bottle_caps_found = 67 :=
by {
  -- Placeholder for proof
  sorry
}

end danny_bottle_caps_after_collection_l35_35773


namespace michael_pays_106_l35_35847

def num_cats : ℕ := 2
def num_dogs : ℕ := 3
def num_parrots : ℕ := 1
def num_fish : ℕ := 4

def cost_per_cat : ℕ := 13
def cost_per_dog : ℕ := 18
def cost_per_parrot : ℕ := 10
def cost_per_fish : ℕ := 4

def total_cost : ℕ :=
  (num_cats * cost_per_cat) +
  (num_dogs * cost_per_dog) +
  (num_parrots * cost_per_parrot) +
  (num_fish * cost_per_fish)

theorem michael_pays_106 : total_cost = 106 := by
  sorry

end michael_pays_106_l35_35847


namespace largest_divisor_l35_35330

theorem largest_divisor (A B : ℕ) (h : 24 = A * B + 4) : A ≤ 20 :=
sorry

end largest_divisor_l35_35330


namespace distance_covered_l35_35981

-- Definitions
def speed : ℕ := 150  -- Speed in km/h
def time : ℕ := 8  -- Time in hours

-- Proof statement
theorem distance_covered : speed * time = 1200 := 
by
  sorry

end distance_covered_l35_35981


namespace tip_customers_count_l35_35413

-- Definitions and given conditions
def initial_customers : ℕ := 29
def added_customers : ℕ := 20
def no_tip_customers : ℕ := 34

-- Total customers computation
def total_customers : ℕ := initial_customers + added_customers

-- Lean 4 statement for proof problem
theorem tip_customers_count : (total_customers - no_tip_customers) = 15 := by
  sorry

end tip_customers_count_l35_35413


namespace smallest_positive_perfect_square_divisible_by_2_and_5_l35_35866

theorem smallest_positive_perfect_square_divisible_by_2_and_5 :
  ∃ n : ℕ, n > 0 ∧ (∃ k : ℕ, n = k^2) ∧ (2 ∣ n) ∧ (5 ∣ n) ∧ n = 100 :=
by
  sorry

end smallest_positive_perfect_square_divisible_by_2_and_5_l35_35866


namespace cube_sphere_volume_ratio_l35_35835

theorem cube_sphere_volume_ratio (s : ℝ) (r : ℝ) (h : r = (Real.sqrt 3 * s) / 2):
  (s^3) / ((4 / 3) * Real.pi * r^3) = (2 * Real.sqrt 3) / Real.pi :=
by
  sorry

end cube_sphere_volume_ratio_l35_35835


namespace Liam_savings_after_trip_and_bills_l35_35367

theorem Liam_savings_after_trip_and_bills :
  let trip_cost := 7000
  let bills_cost := 3500
  let monthly_savings := 500
  let years := 2
  let total_savings := monthly_savings * 12 * years
  total_savings - bills_cost - trip_cost = 1500 := by
  let trip_cost := 7000
  let bills_cost := 3500
  let monthly_savings := 500
  let years := 2
  let total_savings := monthly_savings * 12 * years
  sorry

end Liam_savings_after_trip_and_bills_l35_35367


namespace lucas_earnings_l35_35989

-- Declare constants and definitions given in the problem
def dollars_per_window : ℕ := 3
def windows_per_floor : ℕ := 5
def floors : ℕ := 4
def penalty_amount : ℕ := 2
def days_per_period : ℕ := 4
def total_days : ℕ := 12

-- Definition of the number of total windows
def total_windows : ℕ := windows_per_floor * floors

-- Initial earnings before penalties
def initial_earnings : ℕ := total_windows * dollars_per_window

-- Number of penalty periods
def penalty_periods : ℕ := total_days / days_per_period

-- Total penalty amount
def total_penalty : ℕ := penalty_periods * penalty_amount

-- Final earnings after penalties
def final_earnings : ℕ := initial_earnings - total_penalty

-- Proof problem: correct amount Lucas' father will pay
theorem lucas_earnings : final_earnings = 54 :=
by
  sorry

end lucas_earnings_l35_35989


namespace number_of_zeros_of_h_l35_35675

noncomputable def f (x : ℝ) : ℝ := 2 * x
noncomputable def g (x : ℝ) : ℝ := 3 - x^2
noncomputable def h (x : ℝ) : ℝ := f x - g x

theorem number_of_zeros_of_h : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ h x1 = 0 ∧ h x2 = 0 ∧ ∀ x, h x = 0 → (x = x1 ∨ x = x2) :=
by
  sorry

end number_of_zeros_of_h_l35_35675


namespace desired_average_l35_35903

variable (avg_4_tests : ℕ)
variable (score_5th_test : ℕ)

theorem desired_average (h1 : avg_4_tests = 78) (h2 : score_5th_test = 88) : (4 * avg_4_tests + score_5th_test) / 5 = 80 :=
by
  sorry

end desired_average_l35_35903


namespace craig_apples_total_l35_35020

-- Defining the conditions
def initial_apples_craig : ℝ := 20.0
def apples_from_eugene : ℝ := 7.0

-- Defining the total number of apples Craig will have
noncomputable def total_apples_craig : ℝ := initial_apples_craig + apples_from_eugene

-- The theorem stating that Craig will have 27.0 apples.
theorem craig_apples_total : total_apples_craig = 27.0 := by
  -- Proof here
  sorry

end craig_apples_total_l35_35020


namespace g_neg_six_eq_neg_twenty_l35_35636

theorem g_neg_six_eq_neg_twenty (g : ℤ → ℤ)
    (h1 : g 1 - 1 > 0)
    (h2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x)
    (h3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3) :
  g (-6) = -20 := 
sorry

end g_neg_six_eq_neg_twenty_l35_35636


namespace area_of_triangle_A2B2C2_l35_35150

noncomputable def area_DA1B1 : ℝ := 15 / 4
noncomputable def area_DA1C1 : ℝ := 10
noncomputable def area_DB1C1 : ℝ := 6
noncomputable def area_DA2B2 : ℝ := 40
noncomputable def area_DA2C2 : ℝ := 30
noncomputable def area_DB2C2 : ℝ := 50

theorem area_of_triangle_A2B2C2 : ∃ area : ℝ, 
  area = (50 * Real.sqrt 2) ∧ 
  (area_DA1B1 = 15/4 ∧ 
  area_DA1C1 = 10 ∧ 
  area_DB1C1 = 6 ∧ 
  area_DA2B2 = 40 ∧ 
  area_DA2C2 = 30 ∧ 
  area_DB2C2 = 50) := 
by
  sorry

end area_of_triangle_A2B2C2_l35_35150


namespace ratio_fraction_l35_35994

theorem ratio_fraction (A B C : ℕ) (h1 : 7 * B = 3 * A) (h2 : 6 * C = 5 * B) :
  (C : ℚ) / (A : ℚ) = 5 / 14 ∧ (A : ℚ) / (C : ℚ) = 14 / 5 :=
by
  sorry

end ratio_fraction_l35_35994


namespace initial_passengers_l35_35693

theorem initial_passengers (P : ℕ) (H1 : P - 263 + 419 = 725) : P = 569 :=
by
  sorry

end initial_passengers_l35_35693


namespace school_total_students_l35_35872

theorem school_total_students (T G : ℕ) (h1 : 80 + G = T) (h2 : G = (80 * T) / 100) : T = 400 :=
by
  sorry

end school_total_students_l35_35872


namespace quadratic_integer_roots_l35_35571

theorem quadratic_integer_roots (a b x : ℤ) :
  (∀ x₁ x₂ : ℤ, x₁ + x₂ = -b / a ∧ x₁ * x₂ = b / a → (x₁ = x₂ ∧ x₁ = -2 ∧ b = 4 * a) ∨ (x = -1 ∧ a = 0 ∧ b ≠ 0) ∨ (x = 0 ∧ a ≠ 0 ∧ b = 0)) :=
sorry

end quadratic_integer_roots_l35_35571


namespace jack_total_cost_l35_35684

def plan_base_cost : ℕ := 25

def cost_per_text : ℕ := 8

def free_hours : ℕ := 25

def cost_per_extra_minute : ℕ := 10

def texts_sent : ℕ := 150

def hours_talked : ℕ := 26

def total_cost (base_cost : ℕ) (texts_sent : ℕ) (cost_per_text : ℕ) (hours_talked : ℕ) 
               (free_hours : ℕ) (cost_per_extra_minute : ℕ) : ℕ :=
  base_cost + (texts_sent * cost_per_text) / 100 + 
  ((hours_talked - free_hours) * 60 * cost_per_extra_minute) / 100

theorem jack_total_cost : 
  total_cost plan_base_cost texts_sent cost_per_text hours_talked free_hours cost_per_extra_minute = 43 :=
by
  sorry

end jack_total_cost_l35_35684


namespace sum_of_squares_of_consecutive_integers_is_perfect_square_l35_35125

theorem sum_of_squares_of_consecutive_integers_is_perfect_square (x : ℤ) :
  ∃ k : ℤ, k ^ 2 = x ^ 2 + (x + 1) ^ 2 + (x ^ 2 * (x + 1) ^ 2) :=
by
  use (x^2 + x + 1)
  sorry

end sum_of_squares_of_consecutive_integers_is_perfect_square_l35_35125


namespace necessary_and_sufficient_condition_l35_35188

variables {f g : ℝ → ℝ}

theorem necessary_and_sufficient_condition (f g : ℝ → ℝ)
  (hdom : ∀ x : ℝ, true)
  (hst : ∀ y : ℝ, true) :
  (∀ x : ℝ, f x > g x) ↔ (∀ x : ℝ, ¬ (x ∈ {x : ℝ | f x ≤ g x})) :=
by sorry

end necessary_and_sufficient_condition_l35_35188


namespace find_x_eq_l35_35255

-- Given conditions
variables (c b θ : ℝ)

-- The proof problem
theorem find_x_eq :
  ∃ x : ℝ, x^2 + c^2 * (Real.sin θ)^2 = (b - x)^2 ∧
          x = (b^2 - c^2 * (Real.sin θ)^2) / (2 * b) :=
by
    sorry

end find_x_eq_l35_35255


namespace number_of_slices_per_package_l35_35552

-- Define the problem's conditions
def packages_of_bread := 2
def slices_per_package_of_ham := 8
def packages_of_ham := 2
def leftover_slices_of_bread := 8
def total_ham_slices := packages_of_ham * slices_per_package_of_ham
def total_ham_required_bread := total_ham_slices * 2
def total_initial_bread_slices (B : ℕ) := packages_of_bread * B
def total_bread_used (B : ℕ) := total_ham_required_bread
def slices_leftover (B : ℕ) := total_initial_bread_slices B - total_bread_used B

-- Specify the goal
theorem number_of_slices_per_package (B : ℕ) (h : total_initial_bread_slices B = total_bread_used B + leftover_slices_of_bread) : B = 20 :=
by
  -- Use the provided conditions along with the hypothesis
  -- of the initial bread slices equation equating to used and leftover slices
  sorry

end number_of_slices_per_package_l35_35552


namespace new_boarders_l35_35263

theorem new_boarders (init_boarders : ℕ) (init_day_students : ℕ) (ratio_b : ℕ) (ratio_d : ℕ) (ratio_new_b : ℕ) (ratio_new_d : ℕ) (x : ℕ) :
    init_boarders = 240 →
    ratio_b = 8 →
    ratio_d = 17 →
    ratio_new_b = 3 →
    ratio_new_d = 7 →
    init_day_students = (init_boarders * ratio_d) / ratio_b →
    (ratio_new_b * init_day_students) = ratio_new_d * (init_boarders + x) →
    x = 21 :=
by sorry

end new_boarders_l35_35263


namespace original_number_l35_35961

theorem original_number (x : ℝ) (h : 1.40 * x = 1680) : x = 1200 :=
by {
  sorry -- We will skip the actual proof steps here.
}

end original_number_l35_35961


namespace lcm_first_ten_integers_l35_35270

theorem lcm_first_ten_integers : Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 := by
  sorry

end lcm_first_ten_integers_l35_35270


namespace y_increase_by_20_l35_35485

-- Define the conditions
def relationship (Δx Δy : ℕ) : Prop :=
  Δy = (11 * Δx) / 5

-- The proof problem statement
theorem y_increase_by_20 : relationship 5 11 → relationship 20 44 :=
by
  intros h
  sorry

end y_increase_by_20_l35_35485


namespace race_participants_minimum_l35_35332

theorem race_participants_minimum : ∃ (n : ℕ), 
  (∃ (x : ℕ), n = 3 * x + 1) ∧ 
  (∃ (y : ℕ), n = 4 * y + 1) ∧ 
  (∃ (z : ℕ), n = 5 * z + 1) ∧ 
  n = 61 :=
by
  sorry

end race_participants_minimum_l35_35332


namespace radhika_total_games_l35_35577

-- Define the conditions
def giftsOnChristmas := 12
def giftsOnBirthday := 8
def alreadyOwned := (giftsOnChristmas + giftsOnBirthday) / 2
def totalGifts := giftsOnChristmas + giftsOnBirthday
def expectedTotalGames := totalGifts + alreadyOwned

-- Define the proof statement
theorem radhika_total_games : 
  giftsOnChristmas = 12 ∧ giftsOnBirthday = 8 ∧ alreadyOwned = 10 
  ∧ totalGifts = 20 ∧ expectedTotalGames = 30 :=
by 
  sorry

end radhika_total_games_l35_35577


namespace number_of_people_eating_both_l35_35855

variable (A B C : Nat)

theorem number_of_people_eating_both (hA : A = 13) (hB : B = 19) (hC : C = B - A) : C = 6 :=
by 
  sorry

end number_of_people_eating_both_l35_35855


namespace average_wage_correct_l35_35967

def male_workers : ℕ := 20
def female_workers : ℕ := 15
def child_workers : ℕ := 5

def male_wage : ℕ := 25
def female_wage : ℕ := 20
def child_wage : ℕ := 8

def total_amount_paid_per_day : ℕ := 
  (male_workers * male_wage) + (female_workers * female_wage) + (child_workers * child_wage)

def total_number_of_workers : ℕ := 
  male_workers + female_workers + child_workers

def average_wage_per_day : ℕ := 
  total_amount_paid_per_day / total_number_of_workers

theorem average_wage_correct : 
  average_wage_per_day = 21 := by 
  sorry

end average_wage_correct_l35_35967


namespace engineer_last_name_is_smith_l35_35299

/-- Given these conditions:
 1. Businessman Robinson and a conductor live in Sheffield.
 2. Businessman Jones and a stoker live in Leeds.
 3. Businessman Smith and the railroad engineer live halfway between Leeds and Sheffield.
 4. The conductor’s namesake earns $10,000 a year.
 5. The engineer earns exactly 1/3 of what the businessman who lives closest to him earns.
 6. Railroad worker Smith beats the stoker at billiards.
 
We need to prove that the last name of the engineer is Smith. -/
theorem engineer_last_name_is_smith
  (lives_in_Sheffield_Robinson : Prop)
  (lives_in_Sheffield_conductor : Prop)
  (lives_in_Leeds_Jones : Prop)
  (lives_in_Leeds_stoker : Prop)
  (lives_in_halfway_Smith : Prop)
  (lives_in_halfway_engineer : Prop)
  (conductor_namesake_earns_10000 : Prop)
  (engineer_earns_one_third_closest_bizman : Prop)
  (railway_worker_Smith_beats_stoker_at_billiards : Prop) :
  (engineer_last_name = "Smith") :=
by
  -- Proof will go here
  sorry

end engineer_last_name_is_smith_l35_35299


namespace unknown_number_l35_35969

theorem unknown_number (x : ℝ) (h : 7^8 - 6/x + 9^3 + 3 + 12 = 95) : x = 1 / 960908.333 :=
sorry

end unknown_number_l35_35969


namespace GCD_40_48_l35_35189

theorem GCD_40_48 : Int.gcd 40 48 = 8 :=
by sorry

end GCD_40_48_l35_35189


namespace pages_per_comic_l35_35230

variable {comics_initial : ℕ} -- initially 5 untorn comics in the box
variable {comics_final : ℕ}   -- now there are 11 comics in the box
variable {pages_found : ℕ}    -- found 150 pages on the floor
variable {comics_assembled : ℕ} -- comics assembled from the found pages

theorem pages_per_comic (h1 : comics_initial = 5) (h2 : comics_final = 11) 
      (h3 : pages_found = 150) (h4 : comics_assembled = comics_final - comics_initial) :
      (pages_found / comics_assembled = 25) := 
sorry

end pages_per_comic_l35_35230


namespace solve_for_x_and_compute_value_l35_35167

theorem solve_for_x_and_compute_value (x : ℝ) (h : 5 * x - 3 = 15 * x + 15) : 6 * (x + 5) = 19.2 := by
  sorry

end solve_for_x_and_compute_value_l35_35167


namespace total_servings_daily_l35_35224

def cost_per_serving : ℕ := 14
def price_A : ℕ := 20
def price_B : ℕ := 18
def total_revenue : ℕ := 1120
def total_profit : ℕ := 280

theorem total_servings_daily (x y : ℕ) (h1 : price_A * x + price_B * y = total_revenue)
                             (h2 : (price_A - cost_per_serving) * x + (price_B - cost_per_serving) * y = total_profit) :
                             x + y = 60 := sorry

end total_servings_daily_l35_35224


namespace power_division_l35_35579

theorem power_division (a : ℝ) (h : a ≠ 0) : ((-a)^6) / (a^3) = a^3 := by
  sorry

end power_division_l35_35579


namespace red_apples_ordered_l35_35740

variable (R : ℕ)

theorem red_apples_ordered (h : R + 32 = 2 + 73) : R = 43 := by
  sorry

end red_apples_ordered_l35_35740


namespace solve_for_f_sqrt_2_l35_35786

theorem solve_for_f_sqrt_2 (f : ℝ → ℝ) (h : ∀ x, f x = 2 / (2 - x)) : f (Real.sqrt 2) = 2 + Real.sqrt 2 :=
by
  sorry

end solve_for_f_sqrt_2_l35_35786


namespace pyr_sphere_ineq_l35_35535

open Real

theorem pyr_sphere_ineq (h a : ℝ) (R r : ℝ) 
  (h_pos : h > 0) (a_pos : a > 0) 
  (pyr_in_sphere : ∀ h a : ℝ, R = (2*a^2 + h^2) / (2*h))
  (pyr_circ_sphere : ∀ h a : ℝ, r = (a * h) / (sqrt (h^2 + a^2) + a)) :
  R ≥ (sqrt 2 + 1) * r := 
sorry

end pyr_sphere_ineq_l35_35535


namespace lollipop_cases_l35_35553

theorem lollipop_cases (total_cases : ℕ) (chocolate_cases : ℕ) (lollipop_cases : ℕ) 
  (h1 : total_cases = 80) (h2 : chocolate_cases = 25) : lollipop_cases = 55 :=
by
  sorry

end lollipop_cases_l35_35553


namespace degree_not_determined_from_characteristic_l35_35348

def characteristic (P : Polynomial ℝ) : Set ℝ := sorry -- define this characteristic function

noncomputable def P₁ : Polynomial ℝ := Polynomial.X -- polynomial x
noncomputable def P₂ : Polynomial ℝ := Polynomial.X ^ 3 -- polynomial x^3

theorem degree_not_determined_from_characteristic (A : Polynomial ℝ → Set ℝ)
  (h₁ : A P₁ = A P₂) : 
  ¬∀ P : Polynomial ℝ, ∃ n : ℕ, P.degree = n → A P = A P -> P.degree = n :=
sorry

end degree_not_determined_from_characteristic_l35_35348


namespace arithmetic_mean_solution_l35_35526

/-- Given the arithmetic mean of six expressions is 30, prove the values of x and y are as follows. -/
theorem arithmetic_mean_solution (x y : ℝ) (h : ((2 * x - y) + 20 + (3 * x + y) + 16 + (x + 5) + (y + 8)) / 6 = 30) (hy : y = 10) : 
  x = 18.5 :=
by {
  sorry
}

end arithmetic_mean_solution_l35_35526


namespace number_of_math_books_l35_35009

-- Definitions for conditions
variables (M H : ℕ)

-- Given conditions as a Lean proposition
def conditions : Prop :=
  M + H = 80 ∧ 4 * M + 5 * H = 368

-- The theorem to prove
theorem number_of_math_books (M H : ℕ) (h : conditions M H) : M = 32 :=
by sorry

end number_of_math_books_l35_35009


namespace fifth_element_is_17_l35_35476

-- Define the sequence pattern based on given conditions
def seq : ℕ → ℤ 
| 0 => 5    -- first element
| 1 => -8   -- second element
| n + 2 => seq n + 3    -- each following element is calculated by adding 3 to the two positions before

-- Additional condition: the sign of sequence based on position
def seq_sign : ℕ → ℤ
| n => if n % 2 = 0 then 1 else -1

-- The final adjusted sequence based on the above observations
def final_seq (n : ℕ) : ℤ := seq n * seq_sign n

-- Assert the expected outcome for the 5th element
theorem fifth_element_is_17 : final_seq 4 = 17 :=
by
  sorry

end fifth_element_is_17_l35_35476


namespace function_properties_l35_35331

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x^2 + a*x + b

theorem function_properties (a b : ℝ) (h : (a - 1) ^ 2 - 4 * b < 0) : 
  (∀ x : ℝ, f x a b > x) ∧ (∀ x : ℝ, f (f x a b) a b > x) ∧ (a + b > 0) :=
by
  sorry

end function_properties_l35_35331


namespace sum_of_first_3_geometric_terms_eq_7_l35_35538

theorem sum_of_first_3_geometric_terms_eq_7 
  (a : ℕ → ℝ)
  (r : ℝ)
  (h_geom : ∀ n, a (n+1) = a n * r)
  (h_ratio_gt_1 : r > 1)
  (h_eq : (a 0 + a 2 = 5) ∧ (a 0 * a 2 = 4)) 
  : (a 0 + a 1 + a 2) = 7 := 
by
  sorry

end sum_of_first_3_geometric_terms_eq_7_l35_35538


namespace value_of_y_l35_35506

theorem value_of_y (x y z : ℕ) (h_positive_x : 0 < x) (h_positive_y : 0 < y) (h_positive_z : 0 < z)
    (h_sum : x + y + z = 37) (h_eq : 4 * x = 6 * z) : y = 32 :=
sorry

end value_of_y_l35_35506


namespace transformed_roots_l35_35654

theorem transformed_roots (b c : ℝ) (h₁ : (Polynomial.C 1 * Polynomial.X^2 + Polynomial.C b * Polynomial.X + Polynomial.C c).roots = {2, -3}) :
  (Polynomial.C 1 * (Polynomial.X - Polynomial.C 4)^2 + Polynomial.C b * (Polynomial.X - Polynomial.C 4) + Polynomial.C c).roots = {1, 6} :=
by
  sorry

end transformed_roots_l35_35654


namespace dot_product_eq_neg20_l35_35628

def a : ℝ × ℝ := (3, -1)
def b : ℝ × ℝ := (-5, 5)

def dot_product (x y : ℝ × ℝ) : ℝ :=
x.1 * y.1 + x.2 * y.2

theorem dot_product_eq_neg20 :
  dot_product a b = -20 :=
by
  sorry

end dot_product_eq_neg20_l35_35628


namespace initial_slices_ham_l35_35721

def total_sandwiches : ℕ := 50
def slices_per_sandwich : ℕ := 3
def additional_slices_needed : ℕ := 119

-- Calculate the total number of slices needed to make 50 sandwiches.
def total_slices_needed : ℕ := total_sandwiches * slices_per_sandwich

-- Prove the initial number of slices of ham Anna has.
theorem initial_slices_ham : total_slices_needed - additional_slices_needed = 31 := by
  sorry

end initial_slices_ham_l35_35721


namespace symmetric_point_exists_l35_35208

-- Define the point M
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the original point M
def M : Point3D := { x := 3, y := 3, z := 3 }

-- Define the parametric form of the line
def line (t : ℝ) : Point3D := { x := 1 - t, y := 1.5, z := 3 + t }

-- Define the point M' that we want to prove is symmetrical to M with respect to the line
def symmPoint : Point3D := { x := 1, y := 0, z := 1 }

-- The theorem that we need to prove, ensuring M' is symmetrical to M with respect to the given line
theorem symmetric_point_exists : ∃ t, line t = symmPoint ∧ 
  (∀ M_0 : Point3D, M_0.x = (M.x + symmPoint.x) / 2 ∧ M_0.y = (M.y + symmPoint.y) / 2 ∧ M_0.z = (M.z + symmPoint.z) / 2)
  → line t = M_0
  → M_0 = { x := 2, y := 1.5, z := 2 } := 
by
  sorry

end symmetric_point_exists_l35_35208


namespace proof_l_shaped_area_l35_35879

-- Define the overall rectangle dimensions
def overall_length : ℕ := 10
def overall_width : ℕ := 7

-- Define the dimensions of the removed rectangle
def removed_length : ℕ := overall_length - 3
def removed_width : ℕ := overall_width - 2

-- Calculate the areas
def overall_area : ℕ := overall_length * overall_width
def removed_area : ℕ := removed_length * removed_width
def l_shaped_area : ℕ := overall_area - removed_area

-- The theorem to be proved
theorem proof_l_shaped_area : l_shaped_area = 35 := by
  sorry

end proof_l_shaped_area_l35_35879


namespace arithmetic_identity_l35_35660

theorem arithmetic_identity : 72 * 989 - 12 * 989 = 59340 := by
  sorry

end arithmetic_identity_l35_35660


namespace circle_tangent_x_axis_l35_35673

theorem circle_tangent_x_axis (x y : ℝ) (h_center : (x, y) = (-3, 4)) (h_tangent : y = 4) :
  ∃ r : ℝ, r = 4 ∧ (∀ x y, (x + 3)^2 + (y - 4)^2 = 16) :=
sorry

end circle_tangent_x_axis_l35_35673


namespace min_balls_draw_l35_35712

def box1_red := 40
def box1_green := 30
def box1_yellow := 25
def box1_blue := 15

def box2_red := 35
def box2_green := 25
def box2_yellow := 20

def min_balls_to_draw_to_get_20_balls_of_single_color (totalRed totalGreen totalYellow totalBlue : ℕ) : ℕ :=
  let maxNoColor :=
    (min totalRed 19) + (min totalGreen 19) + (min totalYellow 19) + (min totalBlue 15)
  maxNoColor + 1

theorem  min_balls_draw {r1 r2 g1 g2 y1 y2 b1 : ℕ} :
  r1 = box1_red -> g1 = box1_green -> y1 = box1_yellow -> b1 = box1_blue ->
  r2 = box2_red -> g2 = box2_green -> y2 = box2_yellow ->
  min_balls_to_draw_to_get_20_balls_of_single_color (r1 + r2) (g1 + g2) (y1 + y2) b1 = 73 :=
by
  intros
  unfold min_balls_to_draw_to_get_20_balls_of_single_color
  sorry

end min_balls_draw_l35_35712


namespace bus_speed_calculation_l35_35607

noncomputable def bus_speed_excluding_stoppages : ℝ :=
  let effective_speed_with_stoppages := 50 -- kmph
  let stoppage_time_in_minutes := 13.125 -- minutes per hour
  let stoppage_time_in_hours := stoppage_time_in_minutes / 60 -- convert to hours
  let effective_moving_time := 1 - stoppage_time_in_hours -- effective moving time in one hour
  let bus_speed := (effective_speed_with_stoppages * 60) / (60 - stoppage_time_in_minutes) -- calculate bus speed
  bus_speed

theorem bus_speed_calculation : bus_speed_excluding_stoppages = 64 := by
  sorry

end bus_speed_calculation_l35_35607


namespace parabola_focus_l35_35908

-- Define the equation of the parabola
def parabola_eq (x y : ℝ) : Prop := y^2 = -8 * x

-- Define the coordinates of the focus
def focus (x y : ℝ) : Prop := x = -2 ∧ y = 0

-- The Lean statement that needs to be proved
theorem parabola_focus : ∀ (x y : ℝ), parabola_eq x y → focus x y :=
by
  intros x y h
  sorry

end parabola_focus_l35_35908


namespace like_terms_monomials_l35_35666

theorem like_terms_monomials (a b : ℕ) (x y : ℝ) (c : ℝ) (H1 : x^(a+1) * y^3 = c * y^b * x^2) : a = 1 ∧ b = 3 :=
by
  -- Proof will be provided here
  sorry

end like_terms_monomials_l35_35666


namespace total_marbles_l35_35678

-- Definitions based on given conditions
def ratio_white := 2
def ratio_purple := 3
def ratio_red := 5
def ratio_blue := 4
def ratio_green := 6
def blue_marbles := 24

-- Definition of sum of ratio parts
def sum_of_ratio_parts := ratio_white + ratio_purple + ratio_red + ratio_blue + ratio_green

-- Definition of ratio of blue marbles to total
def ratio_blue_to_total := ratio_blue / sum_of_ratio_parts

-- Proof goal: total number of marbles
theorem total_marbles : blue_marbles / ratio_blue_to_total = 120 := by
  sorry

end total_marbles_l35_35678


namespace average_mark_of_excluded_students_l35_35880

theorem average_mark_of_excluded_students (N A A_remaining N_excluded N_remaining T T_remaining T_excluded A_excluded : ℝ)
  (hN : N = 33) 
  (hA : A = 90) 
  (hA_remaining : A_remaining = 95)
  (hN_excluded : N_excluded = 3) 
  (hN_remaining : N_remaining = N - N_excluded) 
  (hT : T = N * A) 
  (hT_remaining : T_remaining = N_remaining * A_remaining) 
  (hT_eq : T = T_excluded + T_remaining) : 
  A_excluded = T_excluded / N_excluded :=
by
  have hTN : N = 33 := hN
  have hTA : A = 90 := hA
  have hTAR : A_remaining = 95 := hA_remaining
  have hTN_excluded : N_excluded = 3 := hN_excluded
  have hNrem : N_remaining = N - N_excluded := hN_remaining
  have hT_sum : T = N * A := hT
  have hTRem : T_remaining = N_remaining * A_remaining := hT_remaining
  have h_sum_eq : T = T_excluded + T_remaining := hT_eq
  sorry -- proof yet to be constructed

end average_mark_of_excluded_students_l35_35880


namespace carl_sold_each_watermelon_for_3_l35_35546

def profit : ℕ := 105
def final_watermelons : ℕ := 18
def starting_watermelons : ℕ := 53
def sold_watermelons : ℕ := starting_watermelons - final_watermelons
def price_per_watermelon : ℕ := profit / sold_watermelons

theorem carl_sold_each_watermelon_for_3 :
  price_per_watermelon = 3 :=
by
  sorry

end carl_sold_each_watermelon_for_3_l35_35546


namespace find_mn_l35_35886

theorem find_mn (sec_x_plus_tan_x : ℝ) (sec_tan_eq : sec_x_plus_tan_x = 24 / 7) :
  ∃ (m n : ℕ) (h : Int.gcd m n = 1), (∃ y, y = (m:ℝ) / (n:ℝ) ∧ (y^2)*527^2 - 2*y*527*336 + 336^2 = 1) ∧
  m + n = boxed_mn :=
by
  sorry

end find_mn_l35_35886


namespace problem_l35_35637

noncomputable def consecutive_integers_sum (x : ℕ) : ℕ :=
  x + (x + 1)

theorem problem (x : ℕ) (hx : x * (x + 1) = 812) : consecutive_integers_sum x = 57 := by
  sorry

end problem_l35_35637


namespace P_on_x_axis_Q_max_y_PQR_90_deg_PQS_PQT_45_deg_l35_35294

-- Conditions
def center_C : (ℝ × ℝ) := (6, 8)
def radius : ℝ := 10
def circle_eq (x y : ℝ) : Prop := (x - 6)^2 + (y - 8)^2 = 100
def origin_O : (ℝ × ℝ) := (0, 0)

-- (a) Point of intersection of the circle with the x-axis
def point_P : (ℝ × ℝ) := (12, 0)
theorem P_on_x_axis : circle_eq (point_P.1) (point_P.2) ∧ point_P.2 = 0 := sorry

-- (b) Point on the circle with maximum y-coordinate
def point_Q : (ℝ × ℝ) := (6, 18)
theorem Q_max_y : circle_eq (point_Q.1) (point_Q.2) ∧ ∀ y : ℝ, (circle_eq 6 y → y ≤ 18) := sorry

-- (c) Point on the circle such that ∠PQR = 90°
def point_R : (ℝ × ℝ) := (0, 16)
theorem PQR_90_deg : circle_eq (point_R.1) (point_R.2) ∧
  ∃ Q : (ℝ × ℝ), circle_eq (Q.1) (Q.2) ∧ (Q = point_Q) ∧ (point_P.1 - point_R.1) * (Q.1 - point_Q.1) + (point_P.2 - point_R.2) * (Q.2 - point_Q.2) = 0 := sorry

-- (d) Two points on the circle such that ∠PQS = ∠PQT = 45°
def point_S : (ℝ × ℝ) := (14, 14)
def point_T : (ℝ × ℝ) := (-2, 2)
theorem PQS_PQT_45_deg : circle_eq (point_S.1) (point_S.2) ∧ circle_eq (point_T.1) (point_T.2) ∧
  ∃ Q : (ℝ × ℝ), circle_eq (Q.1) (Q.2) ∧ (Q = point_Q) ∧
  ((point_P.1 - Q.1) * (point_S.1 - Q.1) + (point_P.2 - Q.2) * (point_S.2 - Q.2) =
  (point_P.1 - Q.1) * (point_T.1 - Q.1) + (point_P.2 - Q.2) * (point_T.2 - Q.2)) := sorry

end P_on_x_axis_Q_max_y_PQR_90_deg_PQS_PQT_45_deg_l35_35294


namespace investment_amount_correct_l35_35048

noncomputable def investment_problem : Prop :=
  let initial_investment_rubles : ℝ := 10000
  let initial_exchange_rate : ℝ := 50
  let annual_return_rate : ℝ := 0.12
  let end_year_exchange_rate : ℝ := 80
  let currency_conversion_commission : ℝ := 0.05
  let broker_profit_commission_rate : ℝ := 0.3

  -- Computations
  let initial_investment_dollars := initial_investment_rubles / initial_exchange_rate
  let profit_dollars := initial_investment_dollars * annual_return_rate
  let total_dollars := initial_investment_dollars + profit_dollars
  let broker_commission_dollars := profit_dollars * broker_profit_commission_rate
  let post_commission_dollars := total_dollars - broker_commission_dollars
  let amount_in_rubles_before_conversion_commission := post_commission_dollars * end_year_exchange_rate
  let conversion_commission := amount_in_rubles_before_conversion_commission * currency_conversion_commission
  let final_amount_rubles := amount_in_rubles_before_conversion_commission - conversion_commission

  -- Proof goal
  final_amount_rubles = 16476.8

theorem investment_amount_correct : investment_problem := by {
  sorry
}

end investment_amount_correct_l35_35048


namespace number_of_chocolates_bought_l35_35503

theorem number_of_chocolates_bought (C S : ℝ) 
  (h1 : ∃ n : ℕ, n * C = 21 * S) 
  (h2 : (S - C) / C * 100 = 66.67) : 
  ∃ n : ℕ, n = 35 := 
by
  sorry

end number_of_chocolates_bought_l35_35503


namespace dollar_neg3_4_eq_neg27_l35_35618

-- Define the operation $$
def dollar (a b : ℤ) : ℤ := a * (b + 1) + a * b

-- Theorem stating the value of (-3) $$ 4
theorem dollar_neg3_4_eq_neg27 : dollar (-3) 4 = -27 := 
by
  sorry

end dollar_neg3_4_eq_neg27_l35_35618


namespace values_of_m_zero_rain_l35_35595

def f (x y : ℝ) : ℝ := abs (x^3 + 2*x^2*y - 5*x*y^2 - 6*y^3)

theorem values_of_m_zero_rain :
  {m : ℝ | ∀ x : ℝ, f x (m * x) = 0} = {-1, 1/2, -1/3} :=
sorry

end values_of_m_zero_rain_l35_35595


namespace a_2_is_minus_1_l35_35121
open Nat

variable (a S : ℕ → ℤ)

-- Conditions
axiom sum_first_n (n : ℕ) (hn : n > 0) : 2 * S n - n * a n = n
axiom S_20 : S 20 = -360

-- The problem statement to prove
theorem a_2_is_minus_1 : a 2 = -1 :=
by 
  sorry

end a_2_is_minus_1_l35_35121


namespace f_properties_l35_35432

noncomputable def f : ℝ → ℝ := sorry -- we define f as a noncomputable function for generality 

-- Given conditions as Lean hypotheses
axiom functional_eq : ∀ x y : ℝ, f x + f y = 2 * f ((x + y) / 2) * f ((x - y) / 2)
axiom not_always_zero : ¬(∀ x : ℝ, f x = 0)

-- The statement we need to prove
theorem f_properties : f 0 = 1 ∧ (∀ x : ℝ, f (-x) = f x) := 
  by 
    sorry

end f_properties_l35_35432


namespace total_points_correct_l35_35220

-- Define the scores
def Marius (Darius : ℕ) : ℕ := Darius + 3
def Matt (Darius : ℕ) : ℕ := Darius + 5

-- Define the total points function
def total_points (Darius : ℕ) : ℕ :=
  Darius + Marius Darius + Matt Darius

-- Specific value for Darius's score
def Darius_score : ℕ := 10

-- The theorem that proves the total score is 38 given Darius's score
theorem total_points_correct :
  total_points Darius_score = 38 :=
by
  sorry

end total_points_correct_l35_35220


namespace num_satisfying_inequality_l35_35934

theorem num_satisfying_inequality : ∃ (s : Finset ℤ), (∀ n ∈ s, (n + 4) * (n - 8) ≤ 0) ∧ s.card = 13 := by
  sorry

end num_satisfying_inequality_l35_35934


namespace simon_removes_exactly_180_silver_coins_l35_35783

theorem simon_removes_exactly_180_silver_coins :
  ∀ (initial_total_coins initial_gold_percentage final_gold_percentage : ℝ) 
  (initial_silver_coins final_total_coins final_silver_coins silver_coins_removed : ℕ),
  initial_total_coins = 200 → 
  initial_gold_percentage = 0.02 →
  final_gold_percentage = 0.2 →
  initial_silver_coins = (initial_total_coins * (1 - initial_gold_percentage)) → 
  final_total_coins = (4 / final_gold_percentage) →
  final_silver_coins = (final_total_coins - 4) →
  silver_coins_removed = (initial_silver_coins - final_silver_coins) →
  silver_coins_removed = 180 :=
by
  intros initial_total_coins initial_gold_percentage final_gold_percentage 
         initial_silver_coins final_total_coins final_silver_coins silver_coins_removed
  sorry

end simon_removes_exactly_180_silver_coins_l35_35783


namespace total_cost_is_correct_l35_35473

-- Define the costs as constants
def marbles_cost : ℝ := 9.05
def football_cost : ℝ := 4.95
def baseball_cost : ℝ := 6.52

-- Assert that the total cost is correct
theorem total_cost_is_correct : marbles_cost + football_cost + baseball_cost = 20.52 :=
by sorry

end total_cost_is_correct_l35_35473


namespace compare_logs_l35_35483

noncomputable def a : ℝ := Real.log 2 / Real.log 5
noncomputable def b : ℝ := Real.log 3 / Real.log 8
noncomputable def c : ℝ := 1 / 2

theorem compare_logs : a < c ∧ c < b := by
  sorry

end compare_logs_l35_35483


namespace selling_price_conditions_met_l35_35796

-- Definitions based on the problem conditions
def initial_selling_price : ℝ := 50
def purchase_price : ℝ := 40
def initial_volume : ℝ := 500
def decrease_rate : ℝ := 10
def desired_profit : ℝ := 8000
def max_total_cost : ℝ := 10000

-- Definition for the selling price
def selling_price : ℝ := 80

-- Condition: Cost is below $10000 for the valid selling price
def valid_item_count (x : ℝ) : ℝ := initial_volume - decrease_rate * (x - initial_selling_price)

-- Cost calculation function
def total_cost (x : ℝ) : ℝ := purchase_price * valid_item_count x

-- Profit calculation function 
def profit (x : ℝ) : ℝ := (x - purchase_price) * valid_item_count x

-- Main theorem statement
theorem selling_price_conditions_met : 
  profit selling_price = desired_profit ∧ total_cost selling_price < max_total_cost :=
by
  sorry

end selling_price_conditions_met_l35_35796


namespace cricket_player_innings_l35_35339

theorem cricket_player_innings (n : ℕ) (h1 : 35 * n = 35 * n) (h2 : 35 * n + 79 = 39 * (n + 1)) : n = 10 := by
  sorry

end cricket_player_innings_l35_35339


namespace equation_of_line_passing_through_center_and_perpendicular_to_l_l35_35747

theorem equation_of_line_passing_through_center_and_perpendicular_to_l (a : ℝ) : 
  let C_center := (-2, 1)
  let l_slope := 1
  let m_slope := -1
  ∃ (b : ℝ), ∀ x y : ℝ, (x + y + 1 = 0) := 
by 
  let C_center := (-2, 1)
  let l_slope := 1
  let m_slope := -1
  use 1
  sorry

end equation_of_line_passing_through_center_and_perpendicular_to_l_l35_35747


namespace choir_average_age_solution_l35_35938

noncomputable def choir_average_age (avg_f avg_m avg_c : ℕ) (n_f n_m n_c : ℕ) : ℕ :=
  (n_f * avg_f + n_m * avg_m + n_c * avg_c) / (n_f + n_m + n_c)

def choir_average_age_problem : Prop :=
  let avg_f := 32
  let avg_m := 38
  let avg_c := 10
  let n_f := 12
  let n_m := 18
  let n_c := 5
  choir_average_age avg_f avg_m avg_c n_f n_m n_c = 32

theorem choir_average_age_solution : choir_average_age_problem := by
  sorry

end choir_average_age_solution_l35_35938


namespace equidistant_point_quadrants_l35_35108

theorem equidistant_point_quadrants (x y : ℝ) (h : 4 * x + 3 * y = 12) :
  (x > 0 ∧ y = 0) ∨ (x < 0 ∧ y > 0) :=
sorry

end equidistant_point_quadrants_l35_35108


namespace card_game_total_l35_35837

theorem card_game_total (C E O : ℝ) (h1 : E = (11 / 20) * C) (h2 : O = (9 / 20) * C) (h3 : E = O + 50) : C = 500 :=
sorry

end card_game_total_l35_35837


namespace number_of_balls_selected_is_three_l35_35137

-- Definitions of conditions
def total_balls : ℕ := 100
def odd_balls_selected : ℕ := 2
def even_balls_selected : ℕ := 1
def probability_first_ball_odd : ℚ := 2 / 3

-- The number of balls selected
def balls_selected := odd_balls_selected + even_balls_selected

-- Statement of the proof problem
theorem number_of_balls_selected_is_three 
(h1 : total_balls = 100)
(h2 : odd_balls_selected = 2)
(h3 : even_balls_selected = 1)
(h4 : probability_first_ball_odd = 2 / 3) :
  balls_selected = 3 :=
sorry

end number_of_balls_selected_is_three_l35_35137


namespace arithmetic_sequence_common_difference_l35_35744

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ)
  (h1 : a 1 = 30)
  (h2 : ∀ n, S n = n * (a 1 + (n - 1) / 2 * d))
  (h3 : S 12 = S 19) :
  d = -2 :=
by
  sorry

end arithmetic_sequence_common_difference_l35_35744


namespace percentage_material_B_in_final_mixture_l35_35785

-- Conditions
def percentage_material_A_in_Solution_X : ℝ := 20
def percentage_material_B_in_Solution_X : ℝ := 80
def percentage_material_A_in_Solution_Y : ℝ := 30
def percentage_material_B_in_Solution_Y : ℝ := 70
def percentage_material_A_in_final_mixture : ℝ := 22

-- Goal
theorem percentage_material_B_in_final_mixture :
  100 - percentage_material_A_in_final_mixture = 78 := by
  sorry

end percentage_material_B_in_final_mixture_l35_35785


namespace pushups_difference_l35_35350

theorem pushups_difference :
  let David_pushups := 44
  let Zachary_pushups := 35
  David_pushups - Zachary_pushups = 9 :=
by
  -- Here we define the push-ups counts
  let David_pushups := 44
  let Zachary_pushups := 35
  -- We need to show that David did 9 more push-ups than Zachary.
  show David_pushups - Zachary_pushups = 9
  sorry

end pushups_difference_l35_35350


namespace max_sum_sqrt_expr_max_sum_sqrt_expr_attained_l35_35449

open Real

theorem max_sum_sqrt_expr (a b c : ℝ) (h₀ : a ≥ 0) (h₁ : b ≥ 0) (h₂ : c ≥ 0) (h_sum : a + b + c = 8) :
  sqrt (3 * a^2 + 1) + sqrt (3 * b^2 + 1) + sqrt (3 * c^2 + 1) ≤ sqrt 201 :=
  sorry

theorem max_sum_sqrt_expr_attained : sqrt (3 * (8/3)^2 + 1) + sqrt (3 * (8/3)^2 + 1) + sqrt (3 * (8/3)^2 + 1) = sqrt 201 :=
  sorry

end max_sum_sqrt_expr_max_sum_sqrt_expr_attained_l35_35449


namespace lenny_has_39_left_l35_35587

/-- Define the initial amount Lenny has -/
def initial_amount : ℕ := 84

/-- Define the amount Lenny spent on video games -/
def spent_on_video_games : ℕ := 24

/-- Define the amount Lenny spent at the grocery store -/
def spent_on_groceries : ℕ := 21

/-- Define the total amount Lenny spent -/
def total_spent : ℕ := spent_on_video_games + spent_on_groceries

/-- Calculate the amount Lenny has left -/
def amount_left (initial amount_spent : ℕ) : ℕ :=
  initial - amount_spent

/-- The statement of our mathematical equivalent proof problem
  Prove that Lenny has $39 left given the initial amount,
  and the amounts spent on video games and groceries.
-/
theorem lenny_has_39_left :
  amount_left initial_amount total_spent = 39 :=
by
  -- Leave the proof as 'sorry' for now
  sorry

end lenny_has_39_left_l35_35587


namespace find_a_l35_35395

theorem find_a (a b c : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) 
  (h1 : a^2 / b = 1) (h2 : b^2 / c = 2) (h3 : c^2 / a = 3) : 
  a = 12^(1/7 : ℝ) :=
by
  sorry

end find_a_l35_35395


namespace rational_sum_eq_one_l35_35745

theorem rational_sum_eq_one (a b : ℚ) (h : |3 - a| + (b + 2)^2 = 0) : a + b = 1 := 
by
  sorry

end rational_sum_eq_one_l35_35745


namespace maria_strawberries_l35_35381

theorem maria_strawberries (S : ℕ) :
  (21 = 8 + 9 + S) → (S = 4) :=
by
  intro h
  sorry

end maria_strawberries_l35_35381


namespace shaded_region_volume_l35_35145

theorem shaded_region_volume :
  let r1 := 4   -- radius of the first cylinder
  let h1 := 2   -- height of the first cylinder
  let r2 := 1   -- radius of the second cylinder
  let h2 := 5   -- height of the second cylinder
  let V1 := π * r1^2 * h1 -- volume of the first cylinder
  let V2 := π * r2^2 * h2 -- volume of the second cylinder
  V1 + V2 = 37 * π :=
by
  sorry

end shaded_region_volume_l35_35145


namespace avg_of_last_11_eq_41_l35_35490

def sum_of_first_11 : ℕ := 11 * 48
def sum_of_all_21 : ℕ := 21 * 44
def eleventh_number : ℕ := 55

theorem avg_of_last_11_eq_41 (S1 S : ℕ) :
  S1 = sum_of_first_11 →
  S = sum_of_all_21 →
  (S - S1 + eleventh_number) / 11 = 41 :=
by
  sorry

end avg_of_last_11_eq_41_l35_35490


namespace z_is_46_percent_less_than_y_l35_35933

variable (w e y z : ℝ)

-- Conditions
def w_is_60_percent_of_e := w = 0.60 * e
def e_is_60_percent_of_y := e = 0.60 * y
def z_is_150_percent_of_w := z = w * 1.5000000000000002

-- Proof Statement
theorem z_is_46_percent_less_than_y (h1 : w_is_60_percent_of_e w e)
                                    (h2 : e_is_60_percent_of_y e y)
                                    (h3 : z_is_150_percent_of_w z w) :
                                    100 - (z / y * 100) = 46 :=
by
  sorry

end z_is_46_percent_less_than_y_l35_35933


namespace volume_of_displaced_water_square_of_displaced_water_volume_l35_35586

-- Definitions for the conditions
def cube_side_length : ℝ := 10
def displaced_water_volume : ℝ := cube_side_length ^ 3
def displaced_water_volume_squared : ℝ := displaced_water_volume ^ 2

-- The Lean theorem statements proving the equivalence
theorem volume_of_displaced_water : displaced_water_volume = 1000 := by
  sorry

theorem square_of_displaced_water_volume : displaced_water_volume_squared = 1000000 := by
  sorry

end volume_of_displaced_water_square_of_displaced_water_volume_l35_35586


namespace total_number_of_soccer_games_l35_35966

theorem total_number_of_soccer_games (teams : ℕ)
  (regular_games_per_team : ℕ)
  (promotional_games_per_team : ℕ)
  (h1 : teams = 15)
  (h2 : regular_games_per_team = 14)
  (h3 : promotional_games_per_team = 2) :
  ((teams * regular_games_per_team) / 2 + (teams * promotional_games_per_team) / 2) = 120 :=
by
  sorry

end total_number_of_soccer_games_l35_35966


namespace problem1_problem2_l35_35353

-- Problem 1: (-3xy)² * 4x² = 36x⁴y²
theorem problem1 (x y : ℝ) : ((-3 * x * y) ^ 2) * (4 * x ^ 2) = 36 * x ^ 4 * y ^ 2 := by
  sorry

-- Problem 2: (x + 2)(2x - 3) = 2x² + x - 6
theorem problem2 (x : ℝ) : (x + 2) * (2 * x - 3) = 2 * x ^ 2 + x - 6 := by
  sorry

end problem1_problem2_l35_35353


namespace f_for_negative_x_l35_35612

noncomputable def f (x : ℝ) : ℝ :=
if h : x > 0 then x * abs (x - 2) else 0  -- only assume the given case for x > 0

theorem f_for_negative_x (x : ℝ) (h : x < 0) : 
  f x = x * abs (x + 2) :=
by
  -- Sorry block to bypass the proof
  sorry

end f_for_negative_x_l35_35612


namespace M_lt_N_l35_35124

/-- M is the coefficient of x^4 y^2 in the expansion of (x^2 + x + 2y)^5 -/
def M : ℕ := 120

/-- N is the sum of the coefficients in the expansion of (3/x - x)^7 -/
def N : ℕ := 128

/-- The relationship between M and N -/
theorem M_lt_N : M < N := by 
  dsimp [M, N]
  sorry

end M_lt_N_l35_35124


namespace greatest_prime_factor_of_221_l35_35061

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def greatest_prime_factor (n : ℕ) (p : ℕ) : Prop := 
  is_prime p ∧ p ∣ n ∧ ∀ q : ℕ, is_prime q → q ∣ n → q ≤ p

theorem greatest_prime_factor_of_221 : greatest_prime_factor 221 17 := by
  sorry

end greatest_prime_factor_of_221_l35_35061


namespace new_class_mean_l35_35305

theorem new_class_mean :
  let students1 := 45
  let mean1 := 80
  let students2 := 4
  let mean2 := 85
  let students3 := 1
  let score3 := 90
  let total_students := students1 + students2 + students3
  let total_score := (students1 * mean1) + (students2 * mean2) + (students3 * score3)
  let class_mean := total_score / total_students
  class_mean = 80.6 := 
by
  sorry

end new_class_mean_l35_35305


namespace simplify_fraction_eq_one_over_thirty_nine_l35_35502

theorem simplify_fraction_eq_one_over_thirty_nine :
  let a1 := (1 / 3)^1
  let a2 := (1 / 3)^2
  let a3 := (1 / 3)^3
  (1 / (1 / a1 + 1 / a2 + 1 / a3)) = 1 / 39 :=
by
  sorry

end simplify_fraction_eq_one_over_thirty_nine_l35_35502


namespace sphere_radius_geometric_mean_l35_35219

-- Definitions from conditions
variable (r R ρ : ℝ)
variable (r_nonneg : 0 ≤ r)
variable (R_relation : R = 3 * r)
variable (ρ_relation : ρ = Real.sqrt 3 * r)

-- Problem statement
theorem sphere_radius_geometric_mean (tetrahedron : Prop):
  ρ * ρ = R * r :=
by
  sorry

end sphere_radius_geometric_mean_l35_35219


namespace student_chose_number_l35_35793

theorem student_chose_number (x : ℕ) (h : 2 * x - 138 = 112) : x = 125 :=
by
  sorry

end student_chose_number_l35_35793


namespace min_weight_of_automobile_l35_35127

theorem min_weight_of_automobile (ferry_weight_tons: ℝ) (auto_max_weight: ℝ) 
  (max_autos: ℝ) (ferry_weight_pounds: ℝ) (min_auto_weight: ℝ) : 
  ferry_weight_tons = 50 → 
  auto_max_weight = 3200 → 
  max_autos = 62.5 → 
  ferry_weight_pounds = ferry_weight_tons * 2000 → 
  min_auto_weight = ferry_weight_pounds / max_autos → 
  min_auto_weight = 1600 :=
by
  intros
  sorry

end min_weight_of_automobile_l35_35127


namespace lily_spent_on_shirt_l35_35218

theorem lily_spent_on_shirt (S : ℝ) (initial_balance : ℝ) (final_balance : ℝ) : 
  initial_balance = 55 → 
  final_balance = 27 → 
  55 - S - 3 * S = 27 → 
  S = 7 := 
by
  intros h1 h2 h3
  sorry

end lily_spent_on_shirt_l35_35218


namespace find_a8_l35_35285

-- Define the arithmetic sequence aₙ
def arithmetic_sequence (a₁ d : ℕ) (n : ℕ) := a₁ + (n - 1) * d

-- The given condition
def condition (a₁ d : ℕ) :=
  arithmetic_sequence a₁ d 2 + arithmetic_sequence a₁ d 7 + arithmetic_sequence a₁ d 15 = 12

-- The value we want to prove
def a₈ (a₁ d : ℕ ) : ℕ :=
  arithmetic_sequence a₁ d 8

theorem find_a8 (a₁ d : ℕ) (h : condition a₁ d) : a₈ a₁ d = 4 :=
  sorry

end find_a8_l35_35285


namespace train_speed_faster_l35_35470

-- The Lean statement of the problem
theorem train_speed_faster (Vs : ℝ) (L : ℝ) (T : ℝ) (Vf : ℝ) :
  Vs = 36 ∧ L = 340 ∧ T = 17 ∧ (Vf - Vs) * (5 / 18) = L / T → Vf = 108 :=
by 
  intros 
  sorry

end train_speed_faster_l35_35470


namespace length_of_plot_l35_35852

theorem length_of_plot (total_poles : ℕ) (distance : ℕ) (one_side : ℕ) (other_side : ℕ) 
  (poles_distance_condition : total_poles = 28) 
  (fencing_condition : distance = 10) 
  (side_condition : one_side = 50) 
  (rectangular_condition : total_poles = (2 * (one_side / distance) + 2 * (other_side / distance))) :
  other_side = 120 :=
by
  sorry

end length_of_plot_l35_35852


namespace solve_square_l35_35430

theorem solve_square:
  ∃ (square: ℚ), 
    ((13/5) - ((17/2) - square) / (7/2)) / (1 / ((61/20) + (89/20))) = 2 → 
    square = 1/3 :=
  sorry

end solve_square_l35_35430


namespace total_cantaloupes_l35_35311

def cantaloupes (fred : ℕ) (tim : ℕ) := fred + tim

theorem total_cantaloupes : cantaloupes 38 44 = 82 := by
  sorry

end total_cantaloupes_l35_35311


namespace expected_volunteers_by_2022_l35_35364

noncomputable def initial_volunteers : ℕ := 1200
noncomputable def increase_2021 : ℚ := 0.15
noncomputable def increase_2022 : ℚ := 0.30

theorem expected_volunteers_by_2022 :
  (initial_volunteers * (1 + increase_2021) * (1 + increase_2022)) = 1794 := 
by
  sorry

end expected_volunteers_by_2022_l35_35364


namespace students_more_than_pets_l35_35591

-- Definition of given conditions
def num_students_per_classroom := 20
def num_rabbits_per_classroom := 2
def num_goldfish_per_classroom := 3
def num_classrooms := 5

-- Theorem stating the proof problem
theorem students_more_than_pets :
  let total_students := num_students_per_classroom * num_classrooms
  let total_pets := (num_rabbits_per_classroom + num_goldfish_per_classroom) * num_classrooms
  total_students - total_pets = 75 := by
  sorry

end students_more_than_pets_l35_35591


namespace c_geq_one_l35_35046

variable {α : Type*} [LinearOrderedField α]

theorem c_geq_one
  (a : ℕ → α)
  (c : α)
  (h1 : ∀ i : ℕ, 0 < i → 0 ≤ a i ∧ a i ≤ c)
  (h2 : ∀ i j : ℕ, 0 < i → 0 < j → i ≠ j → |a i - a j| ≥ 1 / (i + j)) :
  c ≥ 1 :=
sorry

end c_geq_one_l35_35046


namespace benny_missed_games_l35_35839

def total_games : ℕ := 39
def attended_games : ℕ := 14
def missed_games : ℕ := total_games - attended_games

theorem benny_missed_games : missed_games = 25 := by
  sorry

end benny_missed_games_l35_35839


namespace alices_number_l35_35642

theorem alices_number :
  ∃ (m : ℕ), (180 ∣ m) ∧ (45 ∣ m) ∧ (1000 ≤ m) ∧ (m ≤ 3000) ∧
    (m = 1260 ∨ m = 1440 ∨ m = 1620 ∨ m = 1800 ∨ m = 1980 ∨
     m = 2160 ∨ m = 2340 ∨ m = 2520 ∨ m = 2700 ∨ m = 2880) :=
by
  sorry

end alices_number_l35_35642


namespace equation_transformation_correct_l35_35101

theorem equation_transformation_correct :
  ∀ (x : ℝ), 
  6 * ((x - 1) / 2 - 1) = 6 * ((3 * x + 1) / 3) → 
  (3 * (x - 1) - 6 = 2 * (3 * x + 1)) :=
by
  intro x
  intro h
  sorry

end equation_transformation_correct_l35_35101


namespace valerie_initial_money_l35_35142

theorem valerie_initial_money (n m C_s C_l L I : ℕ) 
  (h1 : n = 3) (h2 : m = 1) (h3 : C_s = 8) (h4 : C_l = 12) (h5 : L = 24) :
  I = (n * C_s) + (m * C_l) + L :=
  sorry

end valerie_initial_money_l35_35142


namespace percent_increase_perimeter_third_triangle_l35_35950

noncomputable def side_length_first : ℝ := 4
noncomputable def side_length_second : ℝ := 2 * side_length_first
noncomputable def side_length_third : ℝ := 2 * side_length_second

noncomputable def perimeter (s : ℝ) : ℝ := 3 * s

noncomputable def percent_increase (initial_perimeter final_perimeter : ℝ) : ℝ := 
  ((final_perimeter - initial_perimeter) / initial_perimeter) * 100

theorem percent_increase_perimeter_third_triangle :
  percent_increase (perimeter side_length_first) (perimeter side_length_third) = 300 := 
sorry

end percent_increase_perimeter_third_triangle_l35_35950


namespace kelly_needs_more_apples_l35_35540

theorem kelly_needs_more_apples (initial_apples : ℕ) (total_apples : ℕ) (needed_apples : ℕ) :
  initial_apples = 128 → total_apples = 250 → needed_apples = total_apples - initial_apples → needed_apples = 122 :=
by
  intros h_initial h_total h_needed
  rw [h_initial, h_total] at h_needed
  exact h_needed

end kelly_needs_more_apples_l35_35540


namespace infinite_impossible_values_of_d_l35_35766

theorem infinite_impossible_values_of_d 
  (pentagon_perimeter square_perimeter : ℕ) 
  (d : ℕ) 
  (h1 : pentagon_perimeter = 5 * (d + ((square_perimeter) / 4)) )
  (h2 : square_perimeter > 0)
  (h3 : pentagon_perimeter - square_perimeter = 2023) :
  ∀ n : ℕ, n > 404 → ¬∃ d : ℕ, d = n :=
by {
  sorry
}

end infinite_impossible_values_of_d_l35_35766


namespace roots_in_arithmetic_progression_l35_35738

theorem roots_in_arithmetic_progression (a b c : ℝ) :
  (∃ x1 x2 x3 : ℝ, (x2 = (x1 + x3) / 2) ∧ (x1 + x2 + x3 = -a) ∧ (x1 * x3 + x2 * (x1 + x3) = b) ∧ (x1 * x2 * x3 = -c)) ↔ 
  (27 * c = 3 * a * b - 2 * a^3 ∧ 3 * b ≤ a^2) :=
sorry

end roots_in_arithmetic_progression_l35_35738


namespace find_angle_at_A_l35_35217

def triangle_angles_sum_to_180 (α β γ : ℝ) : Prop :=
  α + β + γ = 180

def ab_lt_bc_lt_ac (AB BC AC : ℝ) : Prop :=
  AB < BC ∧ BC < AC

def angles_relation (α β γ : ℝ) : Prop :=
  (α = 2 * γ) ∧ (β = 3 * γ)

theorem find_angle_at_A
  (AB BC AC : ℝ)
  (α β γ : ℝ)
  (h1 : ab_lt_bc_lt_ac AB BC AC)
  (h2 : angles_relation α β γ)
  (h3 : triangle_angles_sum_to_180 α β γ) :
  α = 60 :=
sorry

end find_angle_at_A_l35_35217


namespace vincent_total_laundry_loads_l35_35450

theorem vincent_total_laundry_loads :
  let wednesday_loads := 6
  let thursday_loads := 2 * wednesday_loads
  let friday_loads := thursday_loads / 2
  let saturday_loads := wednesday_loads / 3
  let total_loads := wednesday_loads + thursday_loads + friday_loads + saturday_loads
  total_loads = 26 :=
by {
  let wednesday_loads := 6
  let thursday_loads := 2 * wednesday_loads
  let friday_loads := thursday_loads / 2
  let saturday_loads := wednesday_loads / 3
  let total_loads := wednesday_loads + thursday_loads + friday_loads + saturday_loads
  show total_loads = 26
  sorry
}

end vincent_total_laundry_loads_l35_35450


namespace man_completion_time_l35_35024

theorem man_completion_time (w_time : ℕ) (efficiency_increase : ℚ) (m_time : ℕ) :
  w_time = 40 → efficiency_increase = 1.25 → m_time = (w_time : ℚ) / efficiency_increase → m_time = 32 :=
by
  sorry

end man_completion_time_l35_35024


namespace license_plates_count_correct_l35_35044

/-- Calculate the number of five-character license plates. -/
def count_license_plates : Nat :=
  let num_consonants := 20
  let num_vowels := 6
  let num_digits := 10
  num_consonants^2 * num_vowels^2 * num_digits

theorem license_plates_count_correct :
  count_license_plates = 144000 :=
by
  sorry

end license_plates_count_correct_l35_35044


namespace total_pumped_volume_l35_35119

def powerJetA_rate : ℕ := 360
def powerJetB_rate : ℕ := 540
def powerJetA_time : ℕ := 30
def powerJetB_time : ℕ := 45

def pump_volume (rate : ℕ) (minutes : ℕ) : ℕ :=
  rate * (minutes / 60)

theorem total_pumped_volume : 
  pump_volume powerJetA_rate powerJetA_time + pump_volume powerJetB_rate powerJetB_time = 585 := 
by
  sorry

end total_pumped_volume_l35_35119


namespace sally_sours_total_l35_35985

theorem sally_sours_total (cherry_sours lemon_sours orange_sours total_sours : ℕ) 
    (h1 : cherry_sours = 32)
    (h2 : 5 * cherry_sours = 4 * lemon_sours)
    (h3 : orange_sours = total_sours / 4)
    (h4 : cherry_sours + lemon_sours + orange_sours = total_sours) : 
    total_sours = 96 :=
by
  rw [h1] at h2
  have h5 : lemon_sours = 40 := by linarith
  rw [h1, h5] at h4
  have h6 : orange_sours = total_sours / 4 := by assumption
  rw [h6] at h4
  have h7 : 72 + total_sours / 4 = total_sours := by linarith
  sorry

end sally_sours_total_l35_35985


namespace mrs_lovely_class_l35_35527

-- Define the number of students in Mrs. Lovely's class
def number_of_students (g b : ℕ) : ℕ := g + b

theorem mrs_lovely_class (g b : ℕ): 
  (b = g + 3) →
  (500 - 10 = g * g + b * b) →
  number_of_students g b = 23 :=
by
  sorry

end mrs_lovely_class_l35_35527


namespace sum_of_squares_not_divisible_by_5_or_13_l35_35436

-- Definition of the set T
def T (n : ℤ) : ℤ :=
  (n - 1)^2 + n^2 + (n + 1)^2 + (n + 2)^2

-- The theorem to prove
theorem sum_of_squares_not_divisible_by_5_or_13 (n : ℤ) :
  ¬ (T n % 5 = 0) ∧ ¬ (T n % 13 = 0) :=
by
  sorry

end sum_of_squares_not_divisible_by_5_or_13_l35_35436


namespace arithmetic_sum_of_11_terms_l35_35568

variable {α : Type*} [LinearOrderedField α] (a : ℕ → α) (d : α)

def arithmetic_sequence (a : ℕ → α) (a₁ : α) (d : α) : Prop :=
∀ n, a n = a₁ + n * d

def sum_first_n_terms (a : ℕ → α) (n : ℕ) : α :=
(n + 1) * (a 0 + a n) / 2

theorem arithmetic_sum_of_11_terms
  (a₁ d : α)
  (a : ℕ → α)
  (h_seq : arithmetic_sequence a a₁ d)
  (h_cond : a 8 = (1 / 2) * a 11 + 3) :
  sum_first_n_terms a 10 = 66 := by
  sorry

end arithmetic_sum_of_11_terms_l35_35568


namespace polynomial_self_composition_l35_35935

theorem polynomial_self_composition {p : Polynomial ℝ} {n : ℕ} (hn : 0 < n) :
  (∀ x, p.eval (p.eval x) = (p.eval x) ^ n) ↔ p = Polynomial.X ^ n :=
by sorry

end polynomial_self_composition_l35_35935


namespace jane_exercises_per_day_l35_35455

-- Conditions
variables (total_hours : ℕ) (total_weeks : ℕ) (days_per_week : ℕ)
variable (goal_achieved : total_hours = 40 ∧ total_weeks = 8 ∧ days_per_week = 5)

-- Statement
theorem jane_exercises_per_day : ∃ hours_per_day : ℕ, hours_per_day = (total_hours / total_weeks) / days_per_week :=
by
  sorry

end jane_exercises_per_day_l35_35455


namespace line_intersects_parabola_exactly_one_point_l35_35158

theorem line_intersects_parabola_exactly_one_point (k : ℝ) :
  (∃ y : ℝ, -3 * y^2 - 4 * y + 10 = k) ∧
  (∀ y z : ℝ, -3 * y^2 - 4 * y + 10 = k ∧ -3 * z^2 - 4 * z + 10 = k → y = z) 
  → k = 34 / 3 :=
by
  sorry

end line_intersects_parabola_exactly_one_point_l35_35158


namespace apples_harvested_l35_35711

variable (A P : ℕ)
variable (h₁ : P = 3 * A) (h₂ : P - A = 120)

theorem apples_harvested : A = 60 := 
by
  -- proof will go here
  sorry

end apples_harvested_l35_35711


namespace paving_cost_is_16500_l35_35674

-- Define the given conditions
def length : ℝ := 5.5
def width : ℝ := 3.75
def rate_per_sq_meter : ℝ := 800

-- Define the area calculation
def area (L W : ℝ) : ℝ := L * W

-- Define the cost calculation
def cost (A rate : ℝ) : ℝ := A * rate

-- The theorem to prove that the cost of paving the floor is 16500
theorem paving_cost_is_16500 : cost (area length width) rate_per_sq_meter = 16500 :=
by
  -- Proof is omitted here
  sorry

end paving_cost_is_16500_l35_35674


namespace machine_value_depletion_rate_l35_35128

theorem machine_value_depletion_rate :
  ∃ r : ℝ, 700 * (1 - r)^2 = 567 ∧ r = 0.1 := 
by
  sorry

end machine_value_depletion_rate_l35_35128


namespace number_of_footballs_l35_35487

theorem number_of_footballs (x y : ℕ) (h1 : x + y = 20) (h2 : 6 * x + 3 * y = 96) : x = 12 :=
by {
  sorry
}

end number_of_footballs_l35_35487


namespace ball_hits_ground_in_2_72_seconds_l35_35229

noncomputable def height_at_time (t : ℝ) : ℝ :=
  -16 * t^2 - 30 * t + 200

theorem ball_hits_ground_in_2_72_seconds :
  ∃ t : ℝ, t = 2.72 ∧ height_at_time t = 0 :=
by
  use 2.72
  sorry

end ball_hits_ground_in_2_72_seconds_l35_35229


namespace pqr_value_l35_35891

noncomputable def complex_numbers (p q r : ℂ) := p * q + 5 * q = -20 ∧ q * r + 5 * r = -20 ∧ r * p + 5 * p = -20

theorem pqr_value (p q r : ℂ) (h : complex_numbers p q r) : p * q * r = 80 := by
  sorry

end pqr_value_l35_35891


namespace sum_of_a_and_b_l35_35078

def otimes (x y : ℝ) : ℝ := x * (1 - y)

variable (a b : ℝ)

theorem sum_of_a_and_b :
  ({ x : ℝ | (x - a) * (1 - (x - b)) > 0 } = { x : ℝ | 2 < x ∧ x < 3 }) →
  a + b = 4 :=
by
  intro h
  have h_eq : ∀ x, (x - a) * ((1 : ℝ) - (x - b)) = (x - a) * (x - (b + 1)) := sorry
  have h_ineq : ∀ x, (x - a) * (x - (b + 1)) > 0 ↔ 2 < x ∧ x < 3 := sorry
  have h_set_eq : { x | (x - a) * ((1 : ℝ) - (x - b)) > 0 } = { x | 2 < x ∧ x < 3 } := sorry
  have h_roots_2_3 : (2 - a) * (2 - (b + 1)) = 0 ∧ (3 - a) * (3 - (b + 1)) = 0 := sorry
  have h_2_eq : 2 - a = 0 ∨ 2 - (b + 1) = 0 := sorry
  have h_3_eq : 3 - a = 0 ∨ 3 - (b + 1) = 0 := sorry
  have h_a_2 : a = 2 ∨ b + 1 = 2 := sorry
  have h_b_2 : b = 2 - 1 := sorry
  have h_a_3 : a = 3 ∨ b + 1 = 3 := sorry
  have h_b_3 : b = 3 - 1 := sorry
  sorry

end sum_of_a_and_b_l35_35078


namespace seating_arrangement_l35_35850

theorem seating_arrangement (n : ℕ) (h1 : n * 9 + (100 - n) * 10 = 100) : n = 10 :=
by sorry

end seating_arrangement_l35_35850


namespace inequality_solution_l35_35276

/-- Define conditions and state the corresponding theorem -/
theorem inequality_solution (a x : ℝ) (h : a < 0) : ax - 1 > 0 ↔ x < 1 / a :=
by sorry

end inequality_solution_l35_35276


namespace tomatoes_planted_each_kind_l35_35871

-- Definitions derived from Conditions
def total_rows : ℕ := 10
def spaces_per_row : ℕ := 15
def kinds_of_tomatoes : ℕ := 3
def kinds_of_cucumbers : ℕ := 5
def cucumbers_per_kind : ℕ := 4
def potatoes : ℕ := 30
def available_spaces : ℕ := 85

-- Theorem statement with the question and answer derived from the problem
theorem tomatoes_planted_each_kind : (kinds_of_tomatoes * (total_rows * spaces_per_row - Available_spaces - (kinds_of_cucumbers * cucumbers_per_kind + potatoes)) / kinds_of_tomatoes) = 5 :=
by 
  sorry

end tomatoes_planted_each_kind_l35_35871


namespace find_k_value_l35_35739

theorem find_k_value (x k : ℝ) (h : x = 2) (h_sol : (k / (x - 3)) - (1 / (3 - x)) = 1) : k = -2 :=
by
  -- sorry to suppress the actual proof
  sorry

end find_k_value_l35_35739


namespace slope_angle_of_line_l35_35147

theorem slope_angle_of_line (x y : ℝ) (θ : ℝ) : (x - y + 3 = 0) → θ = 45 := 
sorry

end slope_angle_of_line_l35_35147


namespace find_a3_l35_35099

-- Definitions from conditions
def arithmetic_sum (a1 a3 : ℕ) := (3 / 2) * (a1 + a3)
def common_difference := 2
def S3 := 12

-- Theorem to prove that a3 = 6
theorem find_a3 (a1 a3 : ℕ) (h₁ : arithmetic_sum a1 a3 = S3) (h₂ : a3 = a1 + common_difference * 2) : a3 = 6 :=
by
  sorry

end find_a3_l35_35099


namespace wheels_travel_distance_l35_35008

noncomputable def total_horizontal_distance (R₁ R₂ : ℝ) : ℝ :=
  2 * Real.pi * R₁ + 2 * Real.pi * R₂

theorem wheels_travel_distance (R₁ R₂ : ℝ) (h₁ : R₁ = 2) (h₂ : R₂ = 3) :
  total_horizontal_distance R₁ R₂ = 10 * Real.pi :=
by
  rw [total_horizontal_distance, h₁, h₂]
  sorry

end wheels_travel_distance_l35_35008


namespace second_term_of_geometric_series_l35_35522

noncomputable def geometric_series_second_term (a r : ℝ) (S : ℝ) : ℝ :=
a * r

theorem second_term_of_geometric_series 
  (a r S : ℝ) 
  (h1 : r = 1 / 4) 
  (h2 : S = 10) 
  (h3 : S = a / (1 - r)) 
  : geometric_series_second_term a r S = 1.875 :=
by
  sorry

end second_term_of_geometric_series_l35_35522


namespace factorize_square_difference_l35_35161

theorem factorize_square_difference (x: ℝ):
  x^2 - 4 = (x + 2) * (x - 2) := by
  -- Using the difference of squares formula a^2 - b^2 = (a + b)(a - b)
  sorry

end factorize_square_difference_l35_35161


namespace darnel_jogging_l35_35677

variable (j s : ℝ)

theorem darnel_jogging :
  s = 0.875 ∧ s = j + 0.125 → j = 0.750 :=
by
  intros h
  have h1 : s = 0.875 := h.1
  have h2 : s = j + 0.125 := h.2
  sorry

end darnel_jogging_l35_35677


namespace intersection_of_sets_l35_35428

-- Definitions of sets A and B based on given conditions
def setA : Set ℤ := {x | x + 2 = 0}
def setB : Set ℤ := {x | x^2 - 4 = 0}

-- The theorem to prove the intersection of A and B
theorem intersection_of_sets : setA ∩ setB = {-2} := by
  sorry

end intersection_of_sets_l35_35428


namespace angle_bisector_triangle_inequality_l35_35410

theorem angle_bisector_triangle_inequality (AB AC D BD CD x : ℝ) (hAB : AB = 10) (hCD : CD = 3) (h_angle_bisector : BD = 30 / x)
  (h_triangle_inequality_1 : x + (BD + CD) > AB)
  (h_triangle_inequality_2 : AB + (BD + CD) > x)
  (h_triangle_inequality_3 : AB + x > BD + CD) :
  (3 < x) ∧ (x < 15) ∧ (3 + 15 = (18 : ℝ)) :=
by
  sorry

end angle_bisector_triangle_inequality_l35_35410


namespace jason_cuts_lawns_l35_35486

theorem jason_cuts_lawns 
  (time_per_lawn: ℕ)
  (total_cutting_time_hours: ℕ)
  (total_cutting_time_minutes: ℕ)
  (total_yards_cut: ℕ) : 
  time_per_lawn = 30 → 
  total_cutting_time_hours = 8 → 
  total_cutting_time_minutes = total_cutting_time_hours * 60 → 
  total_yards_cut = total_cutting_time_minutes / time_per_lawn → 
  total_yards_cut = 16 :=
by
  intros
  sorry

end jason_cuts_lawns_l35_35486


namespace number_greater_by_l35_35613

def question (a b : Int) : Int := a + b

theorem number_greater_by (a b : Int) : question a b = -11 :=
  by
    sorry

-- Use specific values from the provided problem:
example : question -5 -6 = -11 :=
  by
    sorry

end number_greater_by_l35_35613


namespace first_and_second_bags_l35_35707

def bags_apples (A B C : ℕ) : Prop :=
  (A + B + C = 24) ∧ (B + C = 18) ∧ (A + C = 19)

theorem first_and_second_bags (A B C : ℕ) (h : bags_apples A B C) :
  A + B = 11 :=
sorry

end first_and_second_bags_l35_35707


namespace total_profit_l35_35708

variable (InvestmentA InvestmentB InvestmentTimeA InvestmentTimeB ShareA : ℝ)
variable (hA : InvestmentA = 150)
variable (hB : InvestmentB = 200)
variable (hTimeA : InvestmentTimeA = 12)
variable (hTimeB : InvestmentTimeB = 6)
variable (hShareA : ShareA = 60)

theorem total_profit (TotalProfit : ℝ) :
  (ShareA / 3) * 5 = TotalProfit := 
by
  sorry

end total_profit_l35_35708


namespace towels_per_person_l35_35915

-- Define the conditions
def num_rooms : ℕ := 10
def people_per_room : ℕ := 3
def total_towels : ℕ := 60

-- Define the total number of people
def total_people : ℕ := num_rooms * people_per_room

-- Define the proposition to prove
theorem towels_per_person : total_towels / total_people = 2 :=
by sorry

end towels_per_person_l35_35915


namespace geometric_sequence_characterization_l35_35983

theorem geometric_sequence_characterization (a : ℕ → ℝ) :
  (∀ n, a (n + 1) / a n = (a (n + 2) - a (n + 1)) / (a (n + 1) - a n)) →
  (∃ r : ℝ, ∀ n, a (n + 1) = r * a n) :=
by
  sorry

end geometric_sequence_characterization_l35_35983


namespace chelsea_cupcakes_time_l35_35647

theorem chelsea_cupcakes_time
  (batches : ℕ)
  (bake_time_per_batch : ℕ)
  (ice_time_per_batch : ℕ)
  (total_time : ℕ)
  (h1 : batches = 4)
  (h2 : bake_time_per_batch = 20)
  (h3 : ice_time_per_batch = 30)
  (h4 : total_time = (bake_time_per_batch + ice_time_per_batch) * batches) :
  total_time = 200 :=
  by
  -- The proof statement here
  -- The proof would go here, but we skip it for now
  sorry

end chelsea_cupcakes_time_l35_35647


namespace michael_exceeds_suresh_by_36_5_l35_35472

noncomputable def shares_total : ℝ := 730
noncomputable def punith_ratio_to_michael : ℝ := 3 / 4
noncomputable def michael_ratio_to_suresh : ℝ := 3.5 / 3

theorem michael_exceeds_suresh_by_36_5 :
  ∃ P M S : ℝ, P + M + S = shares_total
  ∧ (P / M = punith_ratio_to_michael)
  ∧ (M / S = michael_ratio_to_suresh)
  ∧ (M - S = 36.5) :=
by
  sorry

end michael_exceeds_suresh_by_36_5_l35_35472


namespace min_b_for_quadratic_factorization_l35_35672

theorem min_b_for_quadratic_factorization : ∃ b : ℕ, b = 84 ∧ ∃ p q : ℤ, p + q = b ∧ p * q = 1760 :=
by
  sorry

end min_b_for_quadratic_factorization_l35_35672


namespace correct_operation_l35_35316

variable {R : Type*} [CommRing R] (x y : R)

theorem correct_operation : x * (1 + y) = x + x * y :=
by sorry

end correct_operation_l35_35316


namespace radius_of_sphere_eq_l35_35788

theorem radius_of_sphere_eq (r : ℝ) : 
  (4 / 3) * Real.pi * r^3 = 4 * Real.pi * r^2 → r = 3 :=
by
  sorry

end radius_of_sphere_eq_l35_35788


namespace build_wall_30_persons_l35_35831

-- Defining the conditions
def work_rate (persons : ℕ) (days : ℕ) : ℚ := 1 / (persons * days)

-- Total work required to build the wall by 8 persons in 42 days
def total_work : ℚ := work_rate 8 42 * 8 * 42

-- Work rate for 30 persons
def combined_work_rate (persons : ℕ) : ℚ := persons * work_rate 8 42

-- Days required for 30 persons to complete the same work
def days_required (persons : ℕ) (work : ℚ) : ℚ := work / combined_work_rate persons

-- Expected result is 11.2 days for 30 persons
theorem build_wall_30_persons : days_required 30 total_work = 11.2 := 
by
  sorry

end build_wall_30_persons_l35_35831


namespace problem1_problem2_problem3_l35_35918

-- Problem 1
theorem problem1 (x y : ℝ) : 4 * x^2 - y^4 = (2 * x + y^2) * (2 * x - y^2) :=
by
  -- proof omitted
  sorry

-- Problem 2
theorem problem2 (x y : ℝ) : 8 * x^2 - 24 * x * y + 18 * y^2 = 2 * (2 * x - 3 * y)^2 :=
by
  -- proof omitted
  sorry

-- Problem 3
theorem problem3 (x y : ℝ) : (x - y) * (3 * x + 1) - 2 * (x^2 - y^2) - (y - x)^2 = (x - y) * (1 - y) :=
by
  -- proof omitted
  sorry

end problem1_problem2_problem3_l35_35918


namespace min_product_value_max_product_value_l35_35762

open Real

noncomputable def min_cos_sin_product (x y z : ℝ) : ℝ :=
  if x ≥ y ∧ y ≥ z ∧ z ≥ π / 12 ∧ x + y + z = π / 2 then
    cos x * sin y * cos z
  else 0

noncomputable def max_cos_sin_product (x y z : ℝ) : ℝ :=
  if x ≥ y ∧ y ≥ z ∧ z ≥ π / 12 ∧ x + y + z = π / 2 then
    cos x * sin y * cos z
  else 0

theorem min_product_value :
  ∃ (x y z : ℝ), x ≥ y ∧ y ≥ z ∧ z ≥ π / 12 ∧ x + y + z = π / 2 ∧ min_cos_sin_product x y z = 1 / 8 :=
sorry

theorem max_product_value :
  ∃ (x y z : ℝ), x ≥ y ∧ y ≥ z ∧ z ≥ π / 12 ∧ x + y + z = π / 2 ∧ max_cos_sin_product x y z = (2 + sqrt 3) / 8 :=
sorry

end min_product_value_max_product_value_l35_35762


namespace total_courses_attended_l35_35510

-- Define the number of courses attended by Max
def maxCourses : ℕ := 40

-- Define the number of courses attended by Sid (four times as many as Max)
def sidCourses : ℕ := 4 * maxCourses

-- Define the total number of courses attended by both Max and Sid
def totalCourses : ℕ := maxCourses + sidCourses

-- The proof statement
theorem total_courses_attended : totalCourses = 200 := by
  sorry

end total_courses_attended_l35_35510


namespace simplify_eval_expression_l35_35524

theorem simplify_eval_expression (a : ℝ) (h : a^2 + 2 * a - 1 = 0) :
  ((a^2 - 1) / (a^2 - 2 * a + 1) - 1 / (1 - a)) / (1 / (a^2 - a)) = 1 :=
  sorry

end simplify_eval_expression_l35_35524


namespace two_digit_number_exists_l35_35545

theorem two_digit_number_exists (x : ℕ) (h1 : 1 ≤ x) (h2 : x ≤ 9) :
  (9 * x + 8) * (80 - 9 * x) = 1855 → (9 * x + 8 = 35 ∨ 9 * x + 8 = 53) := by
  sorry

end two_digit_number_exists_l35_35545


namespace solve_inequality_l35_35550

theorem solve_inequality (x : ℝ) : 
  (x - 2) / (x + 1) ≤ 2 ↔ x ∈ Set.Iic (-4) ∪ Set.Ioi (-1) := 
sorry

end solve_inequality_l35_35550


namespace max_d_is_9_l35_35829

-- Define the 6-digit number of the form 8d8, 45e
def num (d e : ℕ) : ℕ :=
  800000 + 10000 * d + 800 + 450 + e

-- Define the conditions: the number is a multiple of 45, 0 ≤ d, e ≤ 9
def conditions (d e : ℕ) : Prop :=
  0 ≤ d ∧ d ≤ 9 ∧ 0 ≤ e ∧ e ≤ 9 ∧
  (num d e) % 45 = 0

-- Define the maximum value of d
noncomputable def max_d : ℕ :=
  9

-- The theorem statement to be proved
theorem max_d_is_9 :
  ∀ (d e : ℕ), conditions d e → d ≤ max_d :=
by
  sorry

end max_d_is_9_l35_35829


namespace domain_of_f_l35_35398

noncomputable def f (x : ℝ) : ℝ := (Real.tan (2 * x)) / Real.sqrt (x - x^2)

theorem domain_of_f :
  { x : ℝ | ∃ k : ℤ, 2*x ≠ k*π + π/2 ∧ x ∈ (Set.Ioo 0 (π/4) ∪ Set.Ioo (π/4) 1) } = 
  { x : ℝ | x ∈ Set.Ioo 0 (π/4) ∪ Set.Ioo (π/4) 1 } :=
sorry

end domain_of_f_l35_35398


namespace avg_of_sequence_is_x_l35_35377

noncomputable def sum_naturals (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem avg_of_sequence_is_x (x : ℝ) :
  let n := 100
  let sum := sum_naturals n
  (sum + x) / (n + 1) = 50 * x → 
  x = 5050 / 5049 :=
by
  intro n sum h
  exact sorry

end avg_of_sequence_is_x_l35_35377


namespace round_trip_time_l35_35096

def boat_speed_still_water : ℝ := 16
def stream_speed : ℝ := 2
def distance_to_place : ℝ := 7560

theorem round_trip_time : (distance_to_place / (boat_speed_still_water + stream_speed) + distance_to_place / (boat_speed_still_water - stream_speed)) = 960 := by
  sorry

end round_trip_time_l35_35096


namespace cistern_length_is_correct_l35_35397

-- Definitions for the conditions mentioned in the problem
def cistern_width : ℝ := 6
def water_depth : ℝ := 1.25
def wet_surface_area : ℝ := 83

-- The length of the cistern to be proven
def cistern_length : ℝ := 8

-- Theorem statement that length of the cistern must be 8 meters given the conditions
theorem cistern_length_is_correct :
  ∃ (L : ℝ), (wet_surface_area = (L * cistern_width) + (2 * L * water_depth) + (2 * cistern_width * water_depth)) ∧ L = cistern_length :=
  sorry

end cistern_length_is_correct_l35_35397


namespace total_votes_cast_correct_l35_35860

noncomputable def total_votes_cast : Nat :=
  let total_valid_votes : Nat := 1050
  let spoiled_votes : Nat := 325
  total_valid_votes + spoiled_votes

theorem total_votes_cast_correct :
  total_votes_cast = 1375 := by
  sorry

end total_votes_cast_correct_l35_35860


namespace unique_a_value_l35_35905

theorem unique_a_value (a : ℝ) :
  let M := { x : ℝ | x^2 = 2 }
  let N := { x : ℝ | a * x = 1 }
  N ⊆ M ↔ (a = 0 ∨ a = -Real.sqrt 2 / 2 ∨ a = Real.sqrt 2 / 2) :=
by
  sorry

end unique_a_value_l35_35905


namespace range_of_y_l35_35055

-- Define the vectors
def a : ℝ × ℝ := (1, -2)
def b (y : ℝ) : ℝ × ℝ := (4, y)

-- Define the vector sum
def a_plus_b (y : ℝ) : ℝ × ℝ := (a.1 + (b y).1, a.2 + (b y).2)

-- Define the dot product
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Prove the angle between a and a + b is acute and y ≠ -8
theorem range_of_y (y : ℝ) :
  (dot_product a (a_plus_b y) > 0) ↔ (y < 4.5 ∧ y ≠ -8) :=
by
  sorry

end range_of_y_l35_35055


namespace man_l35_35439

theorem man's_salary (S : ℝ)
  (h1 : S * (1/5 + 1/10 + 3/5) = 9/10 * S)
  (h2 : S - 9/10 * S = 14000) :
  S = 140000 :=
by
  sorry

end man_l35_35439


namespace sufficient_condition_implication_l35_35054

theorem sufficient_condition_implication {A B : Prop}
  (h : (¬A → ¬B) ∧ (B → A)): (B → A) ∧ (A → ¬¬A ∧ ¬A → ¬B) :=
by
  -- Note: We would provide the proof here normally, but we skip it for now.
  sorry

end sufficient_condition_implication_l35_35054


namespace solution_of_system_of_inequalities_l35_35627

theorem solution_of_system_of_inequalities (p : ℝ) :
  19 * p < 10 ∧ p > 0.5 ↔ (1 / 2) < p ∧ p < (10 / 19) :=
by
  sorry

end solution_of_system_of_inequalities_l35_35627


namespace find_fencing_cost_l35_35209

theorem find_fencing_cost
  (d : ℝ) (cost_per_meter : ℝ) (π : ℝ)
  (h1 : d = 22)
  (h2 : cost_per_meter = 2.50)
  (hπ : π = Real.pi) :
  (cost_per_meter * (π * d) = 172.80) :=
sorry

end find_fencing_cost_l35_35209


namespace surface_area_spherical_segment_l35_35408

-- Definitions based on given conditions
variables {R h : ℝ}

-- The theorem to be proven
theorem surface_area_spherical_segment (h_pos : 0 < h) (R_pos : 0 < R)
  (planes_not_intersect_sphere : h < 2 * R) :
  S = 2 * π * R * h := by
  sorry

end surface_area_spherical_segment_l35_35408


namespace small_circle_to_large_circle_ratio_l35_35495

theorem small_circle_to_large_circle_ratio (a b : ℝ) (h : π * b^2 - π * a^2 = 3 * π * a^2) :
  a / b = 1 / 2 :=
sorry

end small_circle_to_large_circle_ratio_l35_35495


namespace city_council_vote_l35_35336

theorem city_council_vote :
  ∀ (x y x' y' m : ℕ),
    x + y = 350 →
    y > x →
    y - x = m →
    x' - y' = 2 * m →
    x' + y' = 350 →
    x' = (10 * y) / 9 →
    x' - x = 66 :=
by
  intros x y x' y' m h1 h2 h3 h4 h5 h6
  -- proof goes here
  sorry

end city_council_vote_l35_35336


namespace carolyn_total_monthly_practice_l35_35724

-- Define the constants and relationships given in the problem
def daily_piano_practice : ℕ := 20
def times_violin_practice : ℕ := 3
def days_week : ℕ := 6
def weeks_month : ℕ := 4
def daily_violin_practice : ℕ := daily_piano_practice * times_violin_practice
def total_daily_practice : ℕ := daily_piano_practice + daily_violin_practice
def weekly_practice_time : ℕ := total_daily_practice * days_week
def monthly_practice_time : ℕ := weekly_practice_time * weeks_month

-- The proof statement with the final result
theorem carolyn_total_monthly_practice : monthly_practice_time = 1920 := by
  sorry

end carolyn_total_monthly_practice_l35_35724


namespace solve_for_y_solve_for_x_l35_35943

variable (x y : ℝ)

theorem solve_for_y (h : 2 * x + 3 * y - 4 = 0) : y = (4 - 2 * x) / 3 := 
sorry

theorem solve_for_x (h : 2 * x + 3 * y - 4 = 0) : x = (4 - 3 * y) / 2 := 
sorry

end solve_for_y_solve_for_x_l35_35943


namespace sum_of_two_primes_l35_35174

theorem sum_of_two_primes (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (h_sum : p + q = 93) : p * q = 178 := 
sorry

end sum_of_two_primes_l35_35174


namespace typing_lines_in_10_minutes_l35_35771

def programmers := 10
def total_lines_in_60_minutes := 60
def total_minutes := 60
def target_minutes := 10

theorem typing_lines_in_10_minutes :
  (total_lines_in_60_minutes / total_minutes) * programmers * target_minutes = 100 :=
by sorry

end typing_lines_in_10_minutes_l35_35771


namespace solve_for_t_l35_35419

variables (V0 V g a t S : ℝ)

-- Given conditions
def velocity_eq : Prop := V = (g + a) * t + V0
def displacement_eq : Prop := S = (1/2) * (g + a) * t^2 + V0 * t

-- The theorem to prove
theorem solve_for_t (h1 : velocity_eq V0 V g a t)
                    (h2 : displacement_eq V0 g a t S) :
  t = 2 * S / (V + V0) :=
sorry

end solve_for_t_l35_35419


namespace gray_region_area_l35_35247

theorem gray_region_area (r : ℝ) (h1 : r > 0) (h2 : 3 * r - r = 3) : 
  (π * (3 * r) * (3 * r) - π * r * r) = 18 * π := by
  sorry

end gray_region_area_l35_35247


namespace vasya_average_not_exceed_4_l35_35240

variable (a b c d e : ℕ) 

-- Total number of grades
def total_grades : ℕ := a + b + c + d + e

-- Initial average condition
def initial_condition : Prop := 
  (a + 2 * b + 3 * c + 4 * d + 5 * e) < 3 * (total_grades a b c d e)

-- New average condition after grade changes
def changed_average (a b c d e : ℕ) : ℚ := 
  ((2 * b + 3 * (a + c) + 4 * d + 5 * e) : ℚ) / (total_grades a b c d e)

-- Proof problem to show the new average grade does not exceed 4
theorem vasya_average_not_exceed_4 (h : initial_condition a b c d e) : 
  (changed_average 0 b (c + a) d e) ≤ 4 := 
sorry

end vasya_average_not_exceed_4_l35_35240


namespace possible_values_l35_35126

def seq_condition (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, a n + a (n + 1) = 2 * a (n + 2) * a (n + 3) + 2016

theorem possible_values (a : ℕ → ℤ) (h : seq_condition a) :
  (a 1, a 2) = (0, 2016) ∨
  (a 1, a 2) = (-14, 70) ∨
  (a 1, a 2) = (-69, 15) ∨
  (a 1, a 2) = (-2015, 1) ∨
  (a 1, a 2) = (2016, 0) ∨
  (a 1, a 2) = (70, -14) ∨
  (a 1, a 2) = (15, -69) ∨
  (a 1, a 2) = (1, -2015) :=
sorry

end possible_values_l35_35126


namespace sequence_periodic_l35_35633

noncomputable def exists_N (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, n ≥ 1 → a (n+2) = abs (a (n+1)) - a n

theorem sequence_periodic (a : ℕ → ℝ) (h : exists_N a) :
  ∃ N : ℕ, ∀ n : ℕ, n ≥ N → a (n+9) = a n :=
sorry

end sequence_periodic_l35_35633


namespace star_equiv_l35_35750

variable {m n x y : ℝ}

def star (m n : ℝ) : ℝ := (3 * m - 2 * n) ^ 2

theorem star_equiv (x y : ℝ) : star ((3 * x - 2 * y) ^ 2) ((2 * y - 3 * x) ^ 2) = (3 * x - 2 * y) ^ 4 := 
by
  sorry

end star_equiv_l35_35750


namespace boats_seating_problem_l35_35768

theorem boats_seating_problem 
  (total_boats : ℕ) (total_people : ℕ) 
  (big_boat_seats : ℕ) (small_boat_seats : ℕ) 
  (b s : ℕ) 
  (h1 : total_boats = 12) 
  (h2 : total_people = 58) 
  (h3 : big_boat_seats = 6) 
  (h4 : small_boat_seats = 4) 
  (h5 : b + s = 12) 
  (h6 : b * 6 + s * 4 = 58) 
  : b = 5 ∧ s = 7 :=
sorry

end boats_seating_problem_l35_35768


namespace focus_parabola_l35_35606

theorem focus_parabola (f : ℝ) (d : ℝ) (y : ℝ) :
  (∀ y, ((- (1 / 8) * y^2 - f) ^ 2 + y^2 = (- (1 / 8) * y^2 - d) ^ 2)) → 
  (d - f = 4) → 
  (f^2 = d^2) → 
  f = -2 :=
by
  sorry

end focus_parabola_l35_35606


namespace no_real_roots_iff_k_lt_neg_one_l35_35477

theorem no_real_roots_iff_k_lt_neg_one (k : ℝ) :
  (∀ x : ℝ, x^2 - 2 * x - k ≠ 0) ↔ k < -1 :=
by sorry

end no_real_roots_iff_k_lt_neg_one_l35_35477


namespace permutations_of_three_digit_numbers_from_set_l35_35282

theorem permutations_of_three_digit_numbers_from_set {digits : Finset ℕ} (h : digits = {1, 2, 3, 4, 5}) :
  ∃ n : ℕ, n = (Finset.card digits) * (Finset.card digits - 1) * (Finset.card digits - 2) ∧ n = 60 :=
by
  sorry

end permutations_of_three_digit_numbers_from_set_l35_35282


namespace increasing_g_on_neg_l35_35278

variable {R : Type*} [LinearOrderedField R]

-- Assumptions: 
-- 1. f is an increasing function on R
-- 2. (h_neg : ∀ x : R, f x < 0)

theorem increasing_g_on_neg (f : R → R) (h_inc : ∀ x y : R, x < y → f x < f y) (h_neg : ∀ x : R, f x < 0) :
  ∀ x y : R, x < y → x < 0 → y < 0 → (x^2 * f x < y^2 * f y) :=
by
  sorry

end increasing_g_on_neg_l35_35278


namespace exists_y_with_7_coprimes_less_than_20_l35_35256

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1
def connection (a b : ℕ) : ℚ := Nat.lcm a b / (a * b)

theorem exists_y_with_7_coprimes_less_than_20 :
  ∃ y : ℕ, y < 20 ∧ (∃ x : ℕ, connection y x = 1) ∧ (Nat.totient y = 7) :=
by
  sorry

end exists_y_with_7_coprimes_less_than_20_l35_35256


namespace max_min_K_max_min_2x_plus_y_l35_35940

noncomputable def circle_equation (x y : ℝ) : Prop := x^2 + (y-1)^2 = 1

theorem max_min_K (x y : ℝ) (h : circle_equation x y) : 
  - (Real.sqrt 3) / 3 ≤ (y - 1) / (x - 2) ∧ (y - 1) / (x - 2) ≤ (Real.sqrt 3) / 3 :=
by sorry

theorem max_min_2x_plus_y (x y : ℝ) (h : circle_equation x y) :
  1 - Real.sqrt 5 ≤ 2 * x + y ∧ 2 * x + y ≤ 1 + Real.sqrt 5 :=
by sorry

end max_min_K_max_min_2x_plus_y_l35_35940


namespace div_by_37_l35_35541

theorem div_by_37 : (333^555 + 555^333) % 37 = 0 :=
by sorry

end div_by_37_l35_35541


namespace hyperbola_C2_equation_constant_ratio_kAM_kBN_range_of_w_kAM_kBN_l35_35320

-- Definitions for conditions of the problem
def ellipse_C1 (x y : ℝ) (b : ℝ) : Prop := (x^2) / 4 + (y^2) / (b^2) = 1

def is_sister_conic_section (e1 e2 : ℝ) : Prop :=
  e1 * e2 = Real.sqrt 15 / 4

def hyperbola_C2 (x y : ℝ) : Prop := (x^2) / 4 - y^2 = 1

variable {b : ℝ} (hb : 0 < b ∧ b < 2)
variable {e1 e2 : ℝ} (heccentricities : is_sister_conic_section e1 e2)

theorem hyperbola_C2_equation :
  ∃ (x y : ℝ), ellipse_C1 x y b → hyperbola_C2 x y := sorry

theorem constant_ratio_kAM_kBN (G : ℝ × ℝ) :
  G = (4,0) → 
  ∀ (M N : ℝ × ℝ) (kAM kBN : ℝ), 
  (kAM / kBN = -1/3) := sorry

theorem range_of_w_kAM_kBN (kAM kBN : ℝ) :
  ∃ (w : ℝ),
  w = kAM^2 + (2 / 3) * kBN →
  (w ∈ Set.Icc (-3 / 4) (-11 / 36) ∪ Set.Icc (13 / 36) (5 / 4)) := sorry

end hyperbola_C2_equation_constant_ratio_kAM_kBN_range_of_w_kAM_kBN_l35_35320


namespace Tino_has_correct_jellybeans_total_jellybeans_l35_35497

-- Define the individuals and their amounts of jellybeans
def Arnold_jellybeans := 5
def Lee_jellybeans := 2 * Arnold_jellybeans
def Tino_jellybeans := Lee_jellybeans + 24
def Joshua_jellybeans := 3 * Arnold_jellybeans

-- Verify Tino's jellybean count
theorem Tino_has_correct_jellybeans : Tino_jellybeans = 34 :=
by
  -- Unfold definitions and perform calculations
  sorry

-- Verify the total jellybean count
theorem total_jellybeans : (Arnold_jellybeans + Lee_jellybeans + Tino_jellybeans + Joshua_jellybeans) = 64 :=
by
  -- Unfold definitions and perform calculations
  sorry

end Tino_has_correct_jellybeans_total_jellybeans_l35_35497


namespace abs_inequality_solution_set_l35_35884

theorem abs_inequality_solution_set :
  { x : ℝ | abs (2 - x) < 5 } = { x : ℝ | -3 < x ∧ x < 7 } :=
by
  sorry

end abs_inequality_solution_set_l35_35884


namespace number_of_ordered_pairs_lcm_232848_l35_35659

theorem number_of_ordered_pairs_lcm_232848 :
  let count_pairs :=
    let pairs_1 := 9
    let pairs_2 := 7
    let pairs_3 := 5
    let pairs_4 := 3
    pairs_1 * pairs_2 * pairs_3 * pairs_4
  count_pairs = 945 :=
by
  sorry

end number_of_ordered_pairs_lcm_232848_l35_35659


namespace negation_of_existential_statement_l35_35464

theorem negation_of_existential_statement (x : ℚ) :
  ¬ (∃ x : ℚ, x^2 = 3) ↔ ∀ x : ℚ, x^2 ≠ 3 :=
by sorry

end negation_of_existential_statement_l35_35464


namespace cos_A_eq_sqrt3_div3_of_conditions_l35_35050

noncomputable def given_conditions
  (a b c : ℝ)
  (A B C : ℝ)
  (h1 : (Real.sqrt 3 * b - c) * Real.cos A = a * Real.cos C)
  (h2 : a ≠ 0) 
  (h3 : b ≠ 0) 
  (h4 : c ≠ 0) 
  (h5 : A ≠ 0) 
  (h6 : B ≠ 0) 
  (h7 : C ≠ 0) : Prop :=
  (Real.cos A = Real.sqrt 3 / 3)

theorem cos_A_eq_sqrt3_div3_of_conditions
  (a b c : ℝ)
  (A B C : ℝ)
  (h1 : (Real.sqrt 3 * b - c) * Real.cos A = a * Real.cos C)
  (h2 : a ≠ 0) 
  (h3 : b ≠ 0) 
  (h4 : c ≠ 0) 
  (h5 : A ≠ 0) 
  (h6 : B ≠ 0) 
  (h7 : C ≠ 0) :
  Real.cos A = Real.sqrt 3 / 3 :=
sorry

end cos_A_eq_sqrt3_div3_of_conditions_l35_35050


namespace tap_B_fills_remaining_pool_l35_35159

theorem tap_B_fills_remaining_pool :
  ∀ (flow_A flow_B : ℝ) (t_A t_B : ℕ),
  flow_A = 7.5 / 100 →  -- A fills 7.5% of the pool per hour
  flow_B = 5 / 100 →    -- B fills 5% of the pool per hour
  t_A = 2 →             -- A is open for 2 hours during the second phase
  t_A * flow_A = 15 / 100 →  -- A fills 15% of the pool in 2 hours
  4 * (flow_A + flow_B) = 50 / 100 →  -- A and B together fill 50% of the pool in 4 hours
  (100 / 100 - 50 / 100 - 15 / 100) / flow_B = t_B →  -- remaining pool filled only by B
  t_B = 7 := sorry    -- Prove that t_B is 7

end tap_B_fills_remaining_pool_l35_35159


namespace necessary_but_not_sufficient_condition_l35_35006

theorem necessary_but_not_sufficient_condition (x : ℝ) :
  ((x + 2) * (x - 3) < 0 → |x - 1| < 2) ∧ (¬(|x - 1| < 2 → (x + 2) * (x - 3) < 0)) :=
by
  sorry

end necessary_but_not_sufficient_condition_l35_35006


namespace chick_hit_count_l35_35313

theorem chick_hit_count :
  ∃ x y z : ℕ,
    9 * x + 5 * y + 2 * z = 61 ∧
    x + y + z = 10 ∧
    x ≥ 1 ∧
    y ≥ 1 ∧
    z ≥ 1 ∧
    x = 5 :=
by
  sorry

end chick_hit_count_l35_35313


namespace inequality_problem_l35_35772

open Real

theorem inequality_problem {a b c d : ℝ} (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : 0 ≤ d) (h_ac : a * b + b * c + c * d + d * a = 1) :
  a^3 / (b + c + d) + b^3 / (a + c + d) + c^3 / (a + b + d) + d^3 / (a + b + c) ≥ 1 / 3 := 
by 
  sorry

end inequality_problem_l35_35772


namespace hawkeye_charged_4_times_l35_35846

variables (C B L S : ℝ) (N : ℕ)
def hawkeye_charging_problem : Prop :=
  C = 3.5 ∧ B = 20 ∧ L = 6 ∧ S = B - L ∧ N = (S / C) → N = 4 

theorem hawkeye_charged_4_times : hawkeye_charging_problem C B L S N :=
by {
  repeat { sorry }
}

end hawkeye_charged_4_times_l35_35846


namespace problem_part1_problem_part2_l35_35333

open Real

variables {α : ℝ}

theorem problem_part1 (h1 : sin α - cos α = sqrt 10 / 5) (h2 : α > pi ∧ α < 2 * pi) :
  sin α * cos α = 3 / 10 := sorry

theorem problem_part2 (h1 : sin α - cos α = sqrt 10 / 5) (h2 : α > pi ∧ α < 2 * pi) (h3 : sin α * cos α = 3 / 10) :
  sin α + cos α = - (2 * sqrt 10 / 5) := sorry

end problem_part1_problem_part2_l35_35333


namespace pipe_A_fill_time_l35_35743

theorem pipe_A_fill_time (x : ℝ) (h₁ : x > 0) (h₂ : 1 / x + 1 / 15 = 1 / 6) : x = 10 :=
by
  sorry

end pipe_A_fill_time_l35_35743


namespace total_population_l35_35605

variables (b g t : ℕ)

theorem total_population (h1 : b = 4 * g) (h2 : g = 5 * t) : b + g + t = 26 * t :=
sorry

end total_population_l35_35605


namespace probability_not_win_l35_35301

theorem probability_not_win (n : ℕ) (h : 1 - 1 / (n : ℝ) = 0.9375) : n = 16 :=
sorry

end probability_not_win_l35_35301


namespace original_number_of_movies_l35_35418

theorem original_number_of_movies (x : ℕ) (dvd blu_ray : ℕ)
  (h1 : dvd = 17 * x)
  (h2 : blu_ray = 4 * x)
  (h3 : 17 * x / (4 * x - 4) = 9 / 2) :
  dvd + blu_ray = 378 := by
  sorry

end original_number_of_movies_l35_35418


namespace total_board_length_l35_35705

-- Defining the lengths of the pieces of the board
def shorter_piece_length : ℕ := 23
def longer_piece_length : ℕ := 2 * shorter_piece_length

-- Stating the theorem that the total length of the board is 69 inches
theorem total_board_length : shorter_piece_length + longer_piece_length = 69 :=
by
  -- The proof is omitted for now
  sorry

end total_board_length_l35_35705


namespace number_wall_problem_l35_35685

theorem number_wall_problem (m : ℤ) : 
  ((m + 5) + 16 + 18 = 56) → (m = 17) :=
by
  sorry

end number_wall_problem_l35_35685


namespace isabella_houses_l35_35448

theorem isabella_houses (green yellow red : ℕ)
  (h1 : green = 3 * yellow)
  (h2 : yellow = red - 40)
  (h3 : green = 90) :
  green + red = 160 := 
by sorry

end isabella_houses_l35_35448


namespace f_2012_eq_3_l35_35544

noncomputable def f (a b α β : ℝ) (x : ℝ) : ℝ := a * Real.sin (Real.pi * x + α) + b * Real.cos (Real.pi * x + β) + 4

theorem f_2012_eq_3 
  (a b α β : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hα : α ≠ 0) (hβ : β ≠ 0) 
  (h : f a b α β 2011 = 5) : 
  f a b α β 2012 = 3 :=
by
  sorry

end f_2012_eq_3_l35_35544


namespace smallest_n_l35_35926

theorem smallest_n (n : ℕ) (h : 5 * n ≡ 850 [MOD 26]) : n = 14 :=
by
  sorry

end smallest_n_l35_35926


namespace pythagorean_theorem_l35_35984

-- Definitions from the conditions
variables {a b c : ℝ}
-- Assuming a right triangle with legs a, b and hypotenuse c
def is_right_triangle (a b c : ℝ) : Prop :=
  c^2 = a^2 + b^2

-- Statement of the theorem:
theorem pythagorean_theorem (a b c : ℝ) (h : is_right_triangle a b c) : c^2 = a^2 + b^2 :=
sorry

end pythagorean_theorem_l35_35984


namespace calc_difference_l35_35947

theorem calc_difference :
  let a := (7/12 : ℚ) * 450
  let b := (3/5 : ℚ) * 320
  let c := (5/9 : ℚ) * 540
  let d := b + c
  d - a = 229.5 := by
  -- declare the variables and provide their values
  sorry

end calc_difference_l35_35947


namespace quadratic_equal_real_roots_l35_35112

theorem quadratic_equal_real_roots (a : ℝ) :
  (∃ x : ℝ, x^2 - a * x + 1 = 0 ∧ (x = a*x / 2)) ↔ a = 2 ∨ a = -2 :=
by sorry

end quadratic_equal_real_roots_l35_35112


namespace truck_license_combinations_l35_35914

theorem truck_license_combinations :
  let letter_choices := 3
  let digit_choices := 10
  let number_of_digits := 6
  letter_choices * (digit_choices ^ number_of_digits) = 3000000 :=
by
  sorry

end truck_license_combinations_l35_35914


namespace common_divisor_seven_l35_35221

-- Definition of numbers A, B, and C based on given conditions
def A (m n : ℤ) : ℤ := n^2 + 2 * m * n + 3 * m^2 + 2
def B (m n : ℤ) : ℤ := 2 * n^2 + 3 * m * n + m^2 + 2
def C (m n : ℤ) : ℤ := 3 * n^2 + m * n + 2 * m^2 + 1

-- The proof statement ensuring A, B and C have a common divisor of 7
theorem common_divisor_seven (m n : ℤ) : ∃ d : ℤ, d > 1 ∧ d ∣ A m n ∧ d ∣ B m n ∧ d ∣ C m n → d = 7 :=
by
  sorry

end common_divisor_seven_l35_35221


namespace value_of_a_minus_c_l35_35662

theorem value_of_a_minus_c
  (a b c d : ℝ) 
  (h1 : (a + d + b + d) / 2 = 80)
  (h2 : (b + d + c + d) / 2 = 180)
  (h3 : d = 2 * (a - b)) :
  a - c = -200 := sorry

end value_of_a_minus_c_l35_35662


namespace pipeA_fill_time_l35_35105

variable (t : ℕ) -- t is the time in minutes for Pipe A to fill the tank

-- Conditions
def pipeA_duration (t : ℕ) : Prop :=
  t > 0

def pipeB_duration (t : ℕ) : Prop :=
  t / 3 > 0

def combined_rate (t : ℕ) : Prop :=
  3 * (1 / (4 / t)) = t

-- Problem
theorem pipeA_fill_time (h1 : pipeA_duration t) (h2 : pipeB_duration t) (h3 : combined_rate t) : t = 12 :=
sorry

end pipeA_fill_time_l35_35105


namespace solution_A_l35_35118

def P : Set ℕ := {1, 2, 3, 4}

theorem solution_A (A : Set ℕ) (h1 : A ⊆ P) 
  (h2 : ∀ x ∈ A, 2 * x ∉ A) 
  (h3 : ∀ x ∈ (P \ A), 2 * x ∉ (P \ A)): 
    A = {2} ∨ A = {1, 4} ∨ A = {2, 3} ∨ A = {1, 3, 4} :=
sorry

end solution_A_l35_35118


namespace student_correct_answers_l35_35641

theorem student_correct_answers (C I : ℕ) 
  (h1 : C + I = 100) 
  (h2 : C - 2 * I = 61) : 
  C = 87 :=
by
  sorry

end student_correct_answers_l35_35641


namespace M_plus_N_eq_2_l35_35307

noncomputable def M : ℝ := 1^5 + 2^4 * 3^3 - (4^2 / 5^1)
noncomputable def N : ℝ := 1^5 - 2^4 * 3^3 + (4^2 / 5^1)

theorem M_plus_N_eq_2 : M + N = 2 := by
  sorry

end M_plus_N_eq_2_l35_35307


namespace line_intersects_circle_l35_35794

theorem line_intersects_circle (k : ℝ) (h1 : k = 2) (radius : ℝ) (center_distance : ℝ) (eq_roots : ∀ x, x^2 - k * x + 1 = 0) :
  radius = 5 → center_distance = k → k < radius :=
by
  intros hradius hdistance
  have h_root_eq : k = 2 := h1
  have h_rad : radius = 5 := hradius
  have h_dist : center_distance = k := hdistance
  have kval : k = 2 := h1
  simp [kval, hradius, hdistance, h_rad, h_dist]
  sorry

end line_intersects_circle_l35_35794


namespace length_of_bridge_is_correct_l35_35809

noncomputable def train_length : ℝ := 150
noncomputable def crossing_time : ℝ := 29.997600191984642
noncomputable def train_speed_kmph : ℝ := 36
noncomputable def kmph_to_mps (v : ℝ) : ℝ := (v * 1000) / 3600
noncomputable def train_speed_mps : ℝ := kmph_to_mps train_speed_kmph
noncomputable def total_distance : ℝ := train_speed_mps * crossing_time
noncomputable def bridge_length : ℝ := total_distance - train_length

theorem length_of_bridge_is_correct :
  bridge_length = 149.97600191984642 := by
  sorry

end length_of_bridge_is_correct_l35_35809


namespace ratio_of_edges_l35_35742

theorem ratio_of_edges
  (V₁ V₂ : ℝ)
  (a b : ℝ)
  (hV : V₁ / V₂ = 8 / 1)
  (hV₁ : V₁ = a^3)
  (hV₂ : V₂ = b^3) :
  a / b = 2 / 1 := 
by 
  sorry

end ratio_of_edges_l35_35742


namespace find_x_in_sequence_l35_35371

theorem find_x_in_sequence :
  ∃ x y z : ℤ, 
    (z - 1 = 0) ∧ (y - z = -1) ∧ (x - y = 1) ∧ x = 1 :=
by
  sorry

end find_x_in_sequence_l35_35371


namespace ellipse_equation_l35_35634

noncomputable def point := (ℝ × ℝ)

theorem ellipse_equation (a b : ℝ) (P Q : point) (h1 : a > b) (h2: b > 0) (e : ℝ) (h3 : e = 1/2)
  (h4 : P = (2, 3)) (h5 : Q = (2, -3))
  (h6 : (P.1^2)/(a^2) + (P.2^2)/(b^2) = 1) (h7 : (Q.1^2)/(a^2) + (Q.2^2)/(b^2) = 1) :
  (∀ x y: ℝ, (x^2/16 + y^2/12 = 1) ↔ (x^2/a^2 + y^2/b^2 = 1)) :=
sorry

end ellipse_equation_l35_35634


namespace fitness_center_cost_effectiveness_l35_35651

noncomputable def f (x : ℝ) : ℝ := 5 * x

noncomputable def g (x : ℝ) : ℝ :=
  if 15 ≤ x ∧ x ≤ 30 then 90 
  else 2 * x + 30

def cost_comparison (x : ℝ) (h1 : 15 ≤ x) (h2 : x ≤ 40) : Prop :=
  (15 ≤ x ∧ x < 18 → f x < g x) ∧
  (x = 18 → f x = g x) ∧
  (18 < x ∧ x ≤ 40 → f x > g x)

theorem fitness_center_cost_effectiveness (x : ℝ) (h1 : 15 ≤ x) (h2 : x ≤ 40) : cost_comparison x h1 h2 :=
by
  sorry

end fitness_center_cost_effectiveness_l35_35651


namespace min_value_of_quadratic_function_min_attained_at_negative_two_l35_35604

def quadratic_function (x : ℝ) : ℝ := 3 * (x + 2)^2 - 5

theorem min_value_of_quadratic_function : ∀ x : ℝ, quadratic_function x ≥ -5 :=
by
  sorry

theorem min_attained_at_negative_two : quadratic_function (-2) = -5 :=
by
  sorry

end min_value_of_quadratic_function_min_attained_at_negative_two_l35_35604


namespace displacement_representation_l35_35368

def represents_north (d : ℝ) : Prop := d > 0

theorem displacement_representation (d : ℝ) (h : represents_north 80) : represents_north d ↔ d > 0 :=
by trivial

example (h : represents_north 80) : 
  ∀ d, d = -50 → ¬ represents_north d ∧ abs d = 50 → ∃ s, s = "south" :=
sorry

end displacement_representation_l35_35368


namespace norris_savings_l35_35290

theorem norris_savings:
  ∀ (N : ℕ), 
  (29 + 25 + N = 85) → N = 31 :=
by
  intros N h
  sorry

end norris_savings_l35_35290


namespace find_ratio_l35_35801

theorem find_ratio (f : ℝ → ℝ) (h : ∀ a b : ℝ, b^2 * f a = a^2 * f b) (h3 : f 3 ≠ 0) :
  (f 7 - f 3) / f 3 = 40 / 9 :=
sorry

end find_ratio_l35_35801


namespace number_of_hens_l35_35082

theorem number_of_hens (H C : ℕ) (h1 : H + C = 46) (h2 : 2 * H + 4 * C = 136) : H = 24 :=
by
  sorry

end number_of_hens_l35_35082


namespace molly_more_minutes_than_xanthia_l35_35838

-- Define the constants: reading speeds and book length
def xanthia_speed := 80  -- pages per hour
def molly_speed := 40    -- pages per hour
def book_length := 320   -- pages

-- Define the times taken to read the book in hours
def xanthia_time := book_length / xanthia_speed
def molly_time := book_length / molly_speed

-- Define the time difference in minutes
def time_difference_minutes := (molly_time - xanthia_time) * 60

theorem molly_more_minutes_than_xanthia : time_difference_minutes = 240 := 
by {
  -- Here the proof would go, but we'll leave it as a sorry for now.
  sorry
}

end molly_more_minutes_than_xanthia_l35_35838


namespace gcd_proof_l35_35920

def gcd_10010_15015 := Nat.gcd 10010 15015 = 5005

theorem gcd_proof : gcd_10010_15015 :=
by
  sorry

end gcd_proof_l35_35920


namespace arithmetic_sequence_sum_l35_35508

variable (a : ℕ → ℝ)
variable (d : ℝ)

noncomputable def arithmetic_sequence := ∀ n : ℕ, a n = a 0 + n * d

theorem arithmetic_sequence_sum (h₁ : a 1 + a 2 = 3) (h₂ : a 3 + a 4 = 5) :
  a 7 + a 8 = 9 :=
by
  sorry

end arithmetic_sequence_sum_l35_35508


namespace common_number_is_eight_l35_35480

theorem common_number_is_eight (a b c d e f g : ℝ) 
  (h1 : (a + b + c + d) / 4 = 7)
  (h2 : (d + e + f + g) / 4 = 9)
  (h3 : (a + b + c + d + e + f + g) / 7 = 8) :
  d = 8 :=
by
sorry

end common_number_is_eight_l35_35480


namespace calculate_expression_l35_35887

theorem calculate_expression (y : ℝ) : (20 * y^3) * (7 * y^2) * (1 / (2 * y)^3) = 17.5 * y^2 :=
by
  sorry

end calculate_expression_l35_35887


namespace tulip_price_correct_l35_35191

-- Initial conditions
def first_day_tulips : ℕ := 30
def first_day_roses : ℕ := 20
def second_day_tulips : ℕ := 60
def second_day_roses : ℕ := 40
def third_day_tulips : ℕ := 6
def third_day_roses : ℕ := 16
def rose_price : ℝ := 3
def total_revenue : ℝ := 420

-- Question: What is the price of one tulip?
def tulip_price (T : ℝ) : ℝ :=
    first_day_tulips * T + first_day_roses * rose_price +
    second_day_tulips * T + second_day_roses * rose_price +
    third_day_tulips * T + third_day_roses * rose_price

-- Proof problem statement
theorem tulip_price_correct (T : ℝ) : tulip_price T = total_revenue → T = 2 :=
by
  sorry

end tulip_price_correct_l35_35191


namespace find_c_d_l35_35496

theorem find_c_d (c d : ℝ) (h1 : c ≠ 0) (h2 : d ≠ 0)
  (h3 : ∀ x : ℝ, x^2 + c * x + d = 0 → (x = c ∧ x = d)) :
  c = 1 ∧ d = -2 :=
by
  sorry

end find_c_d_l35_35496


namespace prove_k_range_l35_35600

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := a * x - b * Real.log x

theorem prove_k_range (a b k : ℝ) (h1 : a - b = 1) (h2 : f 1 a b = 2) :
  (∀ x ≥ 1, f x a b ≥ k * x) → k ≤ 2 - 1 / Real.exp 1 :=
by
  sorry

end prove_k_range_l35_35600


namespace find_b_l35_35030

theorem find_b (b : ℚ) (H : ∃ x y : ℚ, x = 3 ∧ y = -7 ∧ b * x + (b - 1) * y = b + 3) : 
  b = 4 / 5 := 
by
  sorry

end find_b_l35_35030


namespace find_ab_exponent_l35_35937

theorem find_ab_exponent (a b : ℝ) 
  (h : |a - 2| + (b + 1 / 2)^2 = 0) : 
  a^2022 * b^2023 = -1 / 2 := 
sorry

end find_ab_exponent_l35_35937


namespace solve_problem_l35_35153

theorem solve_problem (a b c : ℤ) (h1 : 1 < a) (h2 : a < b) (h3 : b < c)
    (h4 : (a-1) * (b-1) * (c-1) ∣ a * b * c - 1) :
    (a = 3 ∧ b = 5 ∧ c = 15) ∨ (a = 2 ∧ b = 4 ∧ c = 8) :=
sorry

end solve_problem_l35_35153


namespace four_digit_perfect_square_is_1156_l35_35373

theorem four_digit_perfect_square_is_1156 :
  ∃ (N : ℕ), (N ≥ 1000) ∧ (N < 10000) ∧ (∀ a, a ∈ [N / 1000, (N % 1000) / 100, (N % 100) / 10, N % 10] → a < 7) 
              ∧ (∃ n : ℕ, N = n * n) ∧ (∃ m : ℕ, (N + 3333 = m * m)) ∧ (N = 1156) :=
by
  sorry

end four_digit_perfect_square_is_1156_l35_35373


namespace certain_number_divisibility_l35_35681

theorem certain_number_divisibility (n : ℕ) (p : ℕ) (h : p = 1) (h2 : 4864 * 9 * n % 12 = 0) : n = 43776 :=
by {
  sorry
}

end certain_number_divisibility_l35_35681


namespace quadratic_real_equal_roots_l35_35578

theorem quadratic_real_equal_roots (k : ℝ) :
  (∃ x : ℝ, 3 * x^2 - k * x + 2 * x + 15 = 0 ∧ ∀ y : ℝ, (3 * y^2 - k * y + 2 * y + 15 = 0 → y = x)) ↔
  (k = 6 * Real.sqrt 5 + 2 ∨ k = -6 * Real.sqrt 5 + 2) :=
by
  sorry

end quadratic_real_equal_roots_l35_35578


namespace sum_of_interior_angles_l35_35682

theorem sum_of_interior_angles (h_triangle : ∀ (a b c : ℝ), a + b + c = 180)
    (h_quadrilateral : ∀ (a b c d : ℝ), a + b + c + d = 360) :
  (∀ (n : ℕ), n ≥ 3 → ∀ (angles : Fin n → ℝ), (Finset.univ.sum angles) = (n-2) * 180) :=
by
  intro n h_n angles
  sorry

end sum_of_interior_angles_l35_35682


namespace fraction_of_top_10_lists_l35_35686

theorem fraction_of_top_10_lists (total_members : ℝ) (min_top_10_lists : ℝ) (fraction : ℝ) 
  (h1 : total_members = 765) (h2 : min_top_10_lists = 191.25) : 
    min_top_10_lists / total_members = fraction := by
  have h3 : fraction = 0.25 := by sorry
  rw [h1, h2, h3]
  sorry

end fraction_of_top_10_lists_l35_35686


namespace triangle_angle_ratio_l35_35978

theorem triangle_angle_ratio (a b c : ℝ) (h₁ : a + b + c = 180)
  (h₂ : b = 2 * a) (h₃ : c = 3 * a) : a = 30 ∧ b = 60 ∧ c = 90 :=
by
  sorry

end triangle_angle_ratio_l35_35978


namespace different_colors_of_roads_leading_out_l35_35584

-- Define the city with intersections and streets
variables (n : ℕ) -- number of intersections
variables (c₁ c₂ c₃ : ℕ) -- number of external roads of each color

-- Conditions
axiom intersections_have_three_streets : ∀ (i : ℕ), i < n → (∀ (color : ℕ), color < 3 → exists (s : ℕ → ℕ), s color < n ∧ s color ≠ s ((color + 1) % 3) ∧ s color ≠ s ((color + 2) % 3))
axiom streets_colored_differently : ∀ (i : ℕ), i < n → (∀ (color1 color2 : ℕ), color1 < 3 → color2 < 3 → color1 ≠ color2 → exists (s1 s2 : ℕ → ℕ), s1 color1 < n ∧ s2 color2 < n ∧ s1 color1 ≠ s2 color2)

-- Problem Statement
theorem different_colors_of_roads_leading_out (h₁ : n % 2 = 0) (h₂ : c₁ + c₂ + c₃ = 3) : c₁ = 1 ∧ c₂ = 1 ∧ c₃ = 1 :=
by sorry

end different_colors_of_roads_leading_out_l35_35584


namespace coconut_to_almond_ratio_l35_35097

-- Conditions
def number_of_coconut_candles (C : ℕ) : Prop :=
  ∃ L A : ℕ, L = 2 * C ∧ A = 10

-- Question
theorem coconut_to_almond_ratio (C : ℕ) (h : number_of_coconut_candles C) :
  ∃ r : ℚ, r = C / 10 := by
  sorry

end coconut_to_almond_ratio_l35_35097


namespace find_B_and_distance_l35_35896

noncomputable def pointA : ℝ × ℝ := (2, 4)

noncomputable def pointB : ℝ × ℝ := (-(1 + Real.sqrt 385) / 8, (-(1 + Real.sqrt 385) / 8) ^ 2)

noncomputable def distanceToOrigin (p : ℝ × ℝ) : ℝ :=
  Real.sqrt (p.1 ^ 2 + p.2 ^ 2)

theorem find_B_and_distance :
  (pointA.snd = pointA.fst ^ 2) ∧
  (pointB.snd = (-(1 + Real.sqrt 385) / 8) ^ 2) ∧
  (distanceToOrigin pointB = Real.sqrt ((-(1 + Real.sqrt 385) / 8) ^ 2 + (-(1 + Real.sqrt 385) / 8) ^ 4)) :=
  sorry

end find_B_and_distance_l35_35896


namespace box_cost_is_550_l35_35120

noncomputable def cost_of_dryer_sheets (loads_per_week : ℕ) (sheets_per_load : ℕ)
                                        (sheets_per_box : ℕ) (annual_savings : ℝ) : ℝ :=
  let sheets_per_week := loads_per_week * sheets_per_load
  let sheets_per_year := sheets_per_week * 52
  let boxes_per_year := sheets_per_year / sheets_per_box
  annual_savings / boxes_per_year

theorem box_cost_is_550 (h1 : 4 = 4)
                        (h2 : 1 = 1)
                        (h3 : 104 = 104)
                        (h4 : 11 = 11) :
  cost_of_dryer_sheets 4 1 104 11 = 5.50 :=
by
  sorry

end box_cost_is_550_l35_35120


namespace combined_cost_is_correct_l35_35767

-- Definitions based on the conditions
def dryer_cost : ℕ := 150
def washer_cost : ℕ := 3 * dryer_cost
def combined_cost : ℕ := dryer_cost + washer_cost

-- Statement to be proved
theorem combined_cost_is_correct : combined_cost = 600 :=
by
  sorry

end combined_cost_is_correct_l35_35767


namespace find_three_digit_numbers_l35_35909

theorem find_three_digit_numbers :
  ∃ A, (100 ≤ A ∧ A ≤ 999) ∧ (A^2 % 1000 = A) ↔ (A = 376) ∨ (A = 625) :=
by
  sorry

end find_three_digit_numbers_l35_35909


namespace centimeters_per_inch_l35_35424

theorem centimeters_per_inch (miles_per_map_inch : ℝ) (cm_measured : ℝ) (approx_miles : ℝ) (miles_per_inch : ℝ) (inches_from_cm : ℝ) : 
  miles_per_map_inch = 16 →
  inches_from_cm = 18.503937007874015 →
  miles_per_map_inch = 24 / 1.5 →
  approx_miles = 296.06299212598424 →
  cm_measured = 47 →
  (cm_measured / inches_from_cm) = 2.54 :=
by
  sorry

end centimeters_per_inch_l35_35424


namespace train_crossing_time_l35_35874

def train_length : ℝ := 140
def bridge_length : ℝ := 235.03
def speed_kmh : ℝ := 45

noncomputable def speed_mps : ℝ := speed_kmh * (1000 / 3600)
noncomputable def total_distance : ℝ := train_length + bridge_length

theorem train_crossing_time :
  (total_distance / speed_mps) = 30.0024 :=
by
  sorry

end train_crossing_time_l35_35874


namespace smallest_initial_number_l35_35380

theorem smallest_initial_number (N : ℕ) (h₁ : N ≤ 999) (h₂ : 27 * N - 240 ≥ 1000) : N = 46 :=
by {
    sorry
}

end smallest_initial_number_l35_35380


namespace team_with_at_least_one_girl_l35_35092

noncomputable def choose (n m : ℕ) := Nat.factorial n / (Nat.factorial m * Nat.factorial (n - m))

theorem team_with_at_least_one_girl (total_boys total_girls select : ℕ) (h_boys : total_boys = 5) (h_girls : total_girls = 5) (h_select : select = 3) :
  (choose (total_boys + total_girls) select) - (choose total_boys select) = 110 := 
by
  sorry

end team_with_at_least_one_girl_l35_35092


namespace polynomial_addition_l35_35825

variable (x : ℝ)

def p := 3 * x^4 + 2 * x^3 - 5 * x^2 + 9 * x - 2
def q := -3 * x^4 - 5 * x^3 + 7 * x^2 - 9 * x + 4

theorem polynomial_addition : p x + q x = -3 * x^3 + 2 * x^2 + 2 := by
  sorry

end polynomial_addition_l35_35825


namespace polynomial_roots_unique_b_c_l35_35334

theorem polynomial_roots_unique_b_c :
    ∀ (r : ℝ), (r ^ 2 - 2 * r - 1 = 0) → (r ^ 5 - 29 * r - 12 = 0) :=
by
    sorry

end polynomial_roots_unique_b_c_l35_35334


namespace product_of_solutions_l35_35849

theorem product_of_solutions (a b c x : ℝ) (h1 : -x^2 - 4 * x + 10 = 0) :
  x * (-4 - x) = -10 :=
by
  sorry

end product_of_solutions_l35_35849


namespace min_value_of_expression_l35_35657

noncomputable def smallest_value (a b c : ℕ) : ℤ :=
  3 * a - 2 * a * b + a * c

theorem min_value_of_expression : ∃ (a b c : ℕ), 0 < a ∧ a < 7 ∧ 0 < b ∧ b ≤ 3 ∧ 0 < c ∧ c ≤ 4 ∧ smallest_value a b c = -12 := by
  sorry

end min_value_of_expression_l35_35657


namespace division_in_base_5_l35_35335

noncomputable def base5_quick_divide : nat := sorry

theorem division_in_base_5 (a b quotient : ℕ) (h1 : a = 1324) (h2 : b = 12) (h3 : quotient = 111) :
  ∃ c : ℕ, c = quotient ∧ a / b = quotient :=
by
  sorry

end division_in_base_5_l35_35335


namespace range_of_a_l35_35245

open Real

theorem range_of_a (a b c : ℝ) (h₁ : a ≠ 0) (h₂ : a - b + c = 3) (h₃ : a + b + c = 1) (h₄ : 0 < c ∧ c < 1) : 1 < a ∧ a < 2 :=
by
  sorry

end range_of_a_l35_35245


namespace envelope_weight_l35_35893

-- Define the conditions as constants
def total_weight_kg : ℝ := 7.48
def num_envelopes : ℕ := 880
def kg_to_g_conversion : ℝ := 1000

-- Calculate the total weight in grams
def total_weight_g : ℝ := total_weight_kg * kg_to_g_conversion

-- Define the expected weight of one envelope in grams
def expected_weight_one_envelope_g : ℝ := 8.5

-- The proof statement
theorem envelope_weight :
  total_weight_g / num_envelopes = expected_weight_one_envelope_g := by
  sorry

end envelope_weight_l35_35893


namespace solve_arithmetic_series_l35_35038

theorem solve_arithmetic_series : 
  25^2 - 23^2 + 21^2 - 19^2 + 17^2 - 15^2 + 13^2 - 11^2 + 9^2 - 7^2 + 5^2 - 3^2 + 1^2 = 338 :=
by sorry

end solve_arithmetic_series_l35_35038


namespace simplify_expression_l35_35590

theorem simplify_expression (x y : ℝ) (h : (x + 2)^2 + abs (y - 1/2) = 0) :
  (x - 2*y)*(x + 2*y) - (x - 2*y)^2 = -6 :=
by
  -- Proof will be provided here
  sorry

end simplify_expression_l35_35590


namespace cows_milk_production_l35_35163

variable (p q r s t : ℕ)

theorem cows_milk_production
  (h : p * r > 0)  -- Assuming p and r are positive to avoid division by zero
  (produce : p * r * q ≠ 0) -- Additional assumption to ensure non-zero q
  (h_cows : q = p * r * (q / (p * r))) 
  : s * t * q / (p * r) = s * t * (q / (p * r)) :=
by
  sorry

end cows_milk_production_l35_35163


namespace parabola_directrix_l35_35001

theorem parabola_directrix (x y : ℝ) (h : y = 8 * x^2) : y = -1 / 32 :=
sorry

end parabola_directrix_l35_35001


namespace missed_angle_l35_35845

def sum_interior_angles (n : ℕ) : ℝ :=
  (n - 2) * 180

theorem missed_angle :
  ∃ (n : ℕ), sum_interior_angles n = 3060 ∧ 3060 - 2997 = 63 :=
by {
  sorry
}

end missed_angle_l35_35845


namespace jerry_reaches_five_probability_l35_35625

noncomputable def probability_move_reaches_five_at_some_point : ℚ :=
  let num_heads_needed := 7
  let num_tails_needed := 3
  let total_tosses := 10
  let num_ways_to_choose_heads := Nat.choose total_tosses num_heads_needed
  let total_possible_outcomes : ℚ := 2^total_tosses
  let prob_reach_4 := num_ways_to_choose_heads / total_possible_outcomes
  let prob_reach_5_at_some_point := 2 * prob_reach_4
  prob_reach_5_at_some_point

theorem jerry_reaches_five_probability :
  probability_move_reaches_five_at_some_point = 15 / 64 := by
  sorry

end jerry_reaches_five_probability_l35_35625


namespace final_result_l35_35393

noncomputable def f : ℝ → ℝ := sorry
def a : ℕ → ℝ := sorry
def S : ℕ → ℝ := sorry

axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom f_periodic : ∀ x : ℝ, f (3 + x) = f x
axiom f_half_periodic : ∀ x : ℝ, f (3 / 2 - x) = f x
axiom f_value_neg2 : f (-2) = -3

axiom a1_value : a 1 = -1
axiom S_n : ∀ n : ℕ, S n = 2 * a n + n

theorem final_result : f (a 5) + f (a 6) = 3 :=
sorry

end final_result_l35_35393


namespace father_son_age_ratio_l35_35820

theorem father_son_age_ratio :
  ∃ S : ℕ, (45 = S + 15 * 2) ∧ (45 / S = 3) := 
sorry

end father_son_age_ratio_l35_35820


namespace sum_of_integers_from_1_to_10_l35_35717

theorem sum_of_integers_from_1_to_10 :
  (Finset.range 11).sum id = 55 :=
sorry

end sum_of_integers_from_1_to_10_l35_35717


namespace distance_to_lake_l35_35731

theorem distance_to_lake (d : ℝ) :
  ¬ (d ≥ 10) → ¬ (d ≤ 9) → d ≠ 7 → d ∈ Set.Ioo 9 10 :=
by
  intros h1 h2 h3
  sorry

end distance_to_lake_l35_35731


namespace find_angle_CDB_l35_35968

variables (A B C D E : Type)
variables [LinearOrderedField A] [LinearOrderedField B] [LinearOrderedField C] [LinearOrderedField D] [LinearOrderedField E]

noncomputable def angle := ℝ -- Define type for angles

variables (AB AD AC ACB ACD : angle)
variables (BAD BEA CDB : ℝ)

-- Define the given angles and conditions in Lean
axiom AB_eq_AD : AB = AD
axiom angle_ACD_eq_angle_ACB : AC = ACD
axiom angle_BAD_eq_140 : BAD = 140
axiom angle_BEA_eq_110 : BEA = 110

theorem find_angle_CDB (AB_eq_AD : AB = AD)
                       (angle_ACD_eq_angle_ACB : AC = ACD)
                       (angle_BAD_eq_140 : BAD = 140)
                       (angle_BEA_eq_110 : BEA = 110) :
                       CDB = 50 :=
by
  sorry

end find_angle_CDB_l35_35968


namespace executiveCommittee_ways_l35_35034

noncomputable def numberOfWaysToFormCommittee (totalMembers : ℕ) (positions : ℕ) : ℕ :=
Nat.choose (totalMembers - 1) (positions - 1)

theorem executiveCommittee_ways : numberOfWaysToFormCommittee 30 5 = 25839 := 
by
  -- skipping the proof as it's not required
  sorry

end executiveCommittee_ways_l35_35034


namespace probability_event_occurring_exactly_once_l35_35589

theorem probability_event_occurring_exactly_once
  (P : ℝ)
  (h1 : ∀ n : ℕ, P ≥ 0 ∧ P ≤ 1) -- Probabilities are valid for all trials
  (h2 : (1 - (1 - P)^3) = 63 / 64) : -- Given condition for at least once
  (3 * P * (1 - P)^2 = 9 / 64) := 
by
  -- Here you would provide the proof steps using the conditions given.
  sorry

end probability_event_occurring_exactly_once_l35_35589


namespace find_number_l35_35474

theorem find_number (x : ℝ) (h : x - (3 / 5) * x = 56) : x = 140 := 
by {
  -- The proof would be written here,
  -- but it is indicated to skip it using "sorry"
  sorry
}

end find_number_l35_35474


namespace train_passes_man_in_15_seconds_l35_35499

theorem train_passes_man_in_15_seconds
  (length_of_train : ℝ)
  (speed_of_train : ℝ)
  (speed_of_man : ℝ)
  (direction_opposite : Bool)
  (h1 : length_of_train = 275)
  (h2 : speed_of_train = 60)
  (h3 : speed_of_man = 6)
  (h4 : direction_opposite = true) : 
  ∃ t : ℝ, t = 15 :=
by
  sorry

end train_passes_man_in_15_seconds_l35_35499


namespace sum_first_95_odds_equals_9025_l35_35249

-- Define the nth odd positive integer
def nth_odd (n : ℕ) : ℕ := 2 * n - 1

-- Define the sum of the first n odd positive integers
def sum_first_n_odds (n : ℕ) : ℕ := n^2

-- State the theorem to be proved
theorem sum_first_95_odds_equals_9025 : sum_first_n_odds 95 = 9025 :=
by
  -- We provide a placeholder for the proof
  sorry

end sum_first_95_odds_equals_9025_l35_35249


namespace exists_equal_sum_pairs_l35_35988

theorem exists_equal_sum_pairs (n : ℕ) (hn : n > 2009) :
  ∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  a ≤ 2009 ∧ b ≤ 2009 ∧ c ≤ 2009 ∧ d ≤ 2009 ∧
  (1 / a + 1 / b : ℝ) = 1 / c + 1 / d :=
sorry

end exists_equal_sum_pairs_l35_35988


namespace free_fall_height_and_last_second_distance_l35_35806

theorem free_fall_height_and_last_second_distance :
  let time := 11
  let initial_distance := 4.9
  let increment := 9.8
  let total_height := (initial_distance * time + increment * (time * (time - 1)) / 2)
  let last_second_distance := initial_distance + increment * (time - 1)
  total_height = 592.9 ∧ last_second_distance = 102.9 :=
by
  sorry

end free_fall_height_and_last_second_distance_l35_35806


namespace proof_N_union_complement_M_eq_235_l35_35060

open Set

theorem proof_N_union_complement_M_eq_235 :
  let U := ({1,2,3,4,5} : Set ℕ)
  let M := ({1, 4} : Set ℕ)
  let N := ({2, 5} : Set ℕ)
  N ∪ (U \ M) = ({2, 3, 5} : Set ℕ) :=
by
  sorry

end proof_N_union_complement_M_eq_235_l35_35060


namespace domain_of_f_l35_35525

noncomputable def f (x : ℝ) := (Real.sqrt (x + 3)) / x

theorem domain_of_f :
  { x : ℝ | x ≥ -3 ∧ x ≠ 0 } = { x : ℝ | ∃ y, f y ≠ 0 } :=
by
  sorry

end domain_of_f_l35_35525


namespace box_filling_rate_l35_35808

theorem box_filling_rate (l w h t : ℝ) (hl : l = 7) (hw : w = 6) (hh : h = 2) (ht : t = 21) : 
  (l * w * h) / t = 4 := by
  sorry

end box_filling_rate_l35_35808


namespace find_marks_in_biology_l35_35875

/-- 
David's marks in various subjects and his average marks are given.
This statement proves David's marks in Biology assuming the conditions provided.
--/
theorem find_marks_in_biology
  (english : ℕ) (math : ℕ) (physics : ℕ) (chemistry : ℕ) (avg_marks : ℕ)
  (total_subjects : ℕ)
  (h_english : english = 91)
  (h_math : math = 65)
  (h_physics : physics = 82)
  (h_chemistry : chemistry = 67)
  (h_avg_marks : avg_marks = 78)
  (h_total_subjects : total_subjects = 5)
  : ∃ (biology : ℕ), biology = 85 := 
by
  sorry

end find_marks_in_biology_l35_35875


namespace find_x_given_k_l35_35676

-- Define the equation under consideration
def equation (x : ℝ) : Prop := (x - 3) / (x - 4) = (x - 5) / (x - 8)

theorem find_x_given_k {k : ℝ} (h : k = 7) : ∀ x : ℝ, x ≠ 4 ∧ x ≠ 8 → equation x → x = 2 :=
by
  intro x hx h_eq
  sorry

end find_x_given_k_l35_35676


namespace cost_price_l35_35841

theorem cost_price (MP SP C : ℝ) (h1 : MP = 74.21875)
  (h2 : SP = MP - 0.20 * MP)
  (h3 : SP = 1.25 * C) : C = 47.5 :=
by
  sorry

end cost_price_l35_35841


namespace quadratic_equation_divisible_by_x_minus_one_l35_35170

theorem quadratic_equation_divisible_by_x_minus_one (a b c : ℝ) (h1 : (x - 1) ∣ (a * x * x + b * x + c)) (h2 : c = 2) :
  (a = 1 ∧ b = -3 ∧ c = 2) → a * x * x + b * x + c = x^2 - 3 * x + 2 :=
by
  sorry

end quadratic_equation_divisible_by_x_minus_one_l35_35170


namespace proof_problem_l35_35753

-- Define the conditions: n is a positive integer and (n(n + 1) / 3) is a square
def problem_condition (n : ℕ) : Prop :=
  ∃ m : ℕ, n > 0 ∧ (n * (n + 1)) = 3 * m^2

-- Define the proof problem: given the condition, n is a multiple of 3, n+1 and n/3 are squares
theorem proof_problem (n : ℕ) (h : problem_condition n) : 
  (∃ a : ℕ, n = 3 * a^2) ∧ 
  (∃ b : ℕ, n + 1 = b^2) ∧ 
  (∃ c : ℕ, n = 3 * c^2) :=
sorry

end proof_problem_l35_35753


namespace value_of_a_l35_35706

theorem value_of_a (a : ℝ) : (a^2 - 4) / (a - 2) = 0 → a ≠ 2 → a = -2 :=
by 
  intro h1 h2
  sorry

end value_of_a_l35_35706


namespace fraction_subtraction_l35_35458

theorem fraction_subtraction (a : ℝ) (h : a ≠ 0) : 1 / a - 3 / a = -2 / a := 
by
  sorry

end fraction_subtraction_l35_35458


namespace days_of_earning_l35_35722

theorem days_of_earning (T D d : ℕ) (hT : T = 165) (hD : D = 33) (h : d = T / D) :
  d = 5 :=
by sorry

end days_of_earning_l35_35722


namespace GCF_LCM_example_l35_35171

/-- Greatest Common Factor (GCF) definition -/
def GCF (a b : ℕ) : ℕ := a.gcd b

/-- Least Common Multiple (LCM) definition -/
def LCM (a b : ℕ) : ℕ := a.lcm b

/-- Main theorem statement to prove -/
theorem GCF_LCM_example : 
  GCF (LCM 9 21) (LCM 8 15) = 3 := by
  sorry

end GCF_LCM_example_l35_35171


namespace beads_per_necklace_correct_l35_35565
-- Importing the necessary library.

-- Defining the given number of necklaces and total beads.
def number_of_necklaces : ℕ := 11
def total_beads : ℕ := 308

-- Stating the proof goal as a theorem.
theorem beads_per_necklace_correct : (total_beads / number_of_necklaces) = 28 := 
by
  sorry

end beads_per_necklace_correct_l35_35565


namespace range_f_compare_sizes_final_comparison_l35_35922

noncomputable def f (x : ℝ) := |2 * x - 1| + |x + 1|

theorem range_f :
  {y : ℝ | ∃ x : ℝ, f x = y} = {y : ℝ | y ∈ Set.Ici (3 / 2)} :=
sorry

theorem compare_sizes (a : ℝ) (ha : a ≥ 3 / 2) :
  |a - 1| + |a + 1| > 3 / (2 * a) ∧ 3 / (2 * a) > 7 / 2 - 2 * a :=
sorry

theorem final_comparison (a : ℝ) (ha : a ≥ 3 / 2) :
  |a - 1| + |a + 1| > 3 / (2 * a) ∧ 3 / (2 * a) > 7 / 2 - 2 * a :=
by
  exact compare_sizes a ha

end range_f_compare_sizes_final_comparison_l35_35922


namespace compare_two_and_neg_three_l35_35179

theorem compare_two_and_neg_three (h1 : 2 > 0) (h2 : -3 < 0) : 2 > -3 :=
by
  sorry

end compare_two_and_neg_three_l35_35179


namespace percentage_increase_first_to_second_l35_35488

theorem percentage_increase_first_to_second (D1 D2 D3 : ℕ) (h1 : D2 = 12)
  (h2 : D3 = D2 + Nat.div (D2 * 25) 100) (h3 : D1 + D2 + D3 = 37) :
  Nat.div ((D2 - D1) * 100) D1 = 20 := by
  sorry

end percentage_increase_first_to_second_l35_35488


namespace ellipse_eccentricity_l35_35897

theorem ellipse_eccentricity (a c : ℝ) (h : 2 * a = 2 * (2 * c)) : (c / a) = 1 / 2 :=
by
  sorry

end ellipse_eccentricity_l35_35897


namespace river_flow_rate_l35_35999

variables (d w : ℝ) (V : ℝ)

theorem river_flow_rate (h₁ : d = 4) (h₂ : w = 40) (h₃ : V = 10666.666666666666) :
  ((V / 60) / (d * w) * 3.6) = 4 :=
by sorry

end river_flow_rate_l35_35999


namespace option_costs_more_cost_effective_x30_more_cost_effective_plan_x30_l35_35425

def racket_price : ℕ := 80
def ball_price : ℕ := 20
def discount_rate : ℕ := 90

def option_1_cost (n_rackets : ℕ) : ℕ :=
  n_rackets * racket_price

def option_2_cost (n_rackets : ℕ) (n_balls : ℕ) : ℕ :=
  (discount_rate * (n_rackets * racket_price + n_balls * ball_price)) / 100

-- Part 1: Express in Algebraic Terms
theorem option_costs (n_rackets : ℕ) (n_balls : ℕ) :
  option_1_cost n_rackets = 1600 ∧ option_2_cost n_rackets n_balls = 1440 + 18 * n_balls := 
by
  sorry

-- Part 2: For x = 30, determine more cost-effective option
theorem more_cost_effective_x30 (x : ℕ) (h : x = 30) :
  option_1_cost 20 < option_2_cost 20 x := 
by
  sorry

-- Part 3: More cost-effective Plan for x = 30
theorem more_cost_effective_plan_x30 :
  1600 + (discount_rate * (10 * ball_price)) / 100 < option_2_cost 20 30 :=
by
  sorry

end option_costs_more_cost_effective_x30_more_cost_effective_plan_x30_l35_35425


namespace distinct_real_roots_imply_sum_greater_than_two_l35_35865

noncomputable def function_f (x: ℝ) : ℝ := abs (Real.log x)

theorem distinct_real_roots_imply_sum_greater_than_two {k α β : ℝ} 
  (h₁ : function_f α = k) 
  (h₂ : function_f β = k) 
  (h₃ : α ≠ β) 
  (h4 : 0 < α ∧ α < 1)
  (h5 : 1 < β) :
  (1 / α) + (1 / β) > 2 :=
sorry

end distinct_real_roots_imply_sum_greater_than_two_l35_35865


namespace find_x_l35_35379

theorem find_x (a x : ℤ) (h1 : -6 * a^2 = x * (4 * a + 2)) (h2 : a = 1) : x = -1 :=
sorry

end find_x_l35_35379


namespace log_cosine_range_l35_35401

noncomputable def log_base_three (a : ℝ) : ℝ := Real.log a / Real.log 3

theorem log_cosine_range (x : ℝ) (hx : x ∈ Set.Ioo (Real.pi / 2) (7 * Real.pi / 6)) :
    ∃ y, y = log_base_three (1 - 2 * Real.cos x) ∧ y ∈ Set.Ioc 0 1 :=
by
  sorry

end log_cosine_range_l35_35401


namespace customer_payment_l35_35736

noncomputable def cost_price : ℝ := 4090.9090909090905
noncomputable def markup : ℝ := 0.32
noncomputable def selling_price : ℝ := cost_price * (1 + markup)

theorem customer_payment :
  selling_price = 5400 :=
by
  unfold selling_price
  unfold cost_price
  unfold markup
  sorry

end customer_payment_l35_35736


namespace f_decreasing_interval_triangle_abc_l35_35132

noncomputable def f (x : Real) : Real := 2 * (Real.sin x)^2 + Real.cos ((Real.pi) / 3 - 2 * x)

theorem f_decreasing_interval :
  ∃ (a b : Real), a = Real.pi / 3 ∧ b = 5 * Real.pi / 6 ∧ 
  ∀ x y, (a ≤ x ∧ x < y ∧ y ≤ b) → f y ≤ f x := 
sorry

variables {a b c : Real} (A B C : Real) 

theorem triangle_abc (h1 : A = Real.pi / 3) 
    (h2 : f A = 2)
    (h3 : a = 2 * b)
    (h4 : Real.sin C = 2 * Real.sin B):
  a / b = Real.sqrt 3 := 
sorry

end f_decreasing_interval_triangle_abc_l35_35132


namespace class_8_3_final_score_is_correct_l35_35709

def class_8_3_singing_quality : ℝ := 92
def class_8_3_spirit : ℝ := 80
def class_8_3_coordination : ℝ := 70

def final_score (singing_quality spirit coordination : ℝ) : ℝ :=
  0.4 * singing_quality + 0.3 * spirit + 0.3 * coordination

theorem class_8_3_final_score_is_correct :
  final_score class_8_3_singing_quality class_8_3_spirit class_8_3_coordination = 81.8 :=
by
  sorry

end class_8_3_final_score_is_correct_l35_35709


namespace abs_value_solutions_l35_35639

theorem abs_value_solutions (y : ℝ) :
  |4 * y - 5| = 39 ↔ (y = 11 ∨ y = -8.5) :=
by
  sorry

end abs_value_solutions_l35_35639


namespace binom_product_l35_35086

open Nat

-- Define binomial coefficient
def binom (n k : ℕ) : ℕ := n.choose k

theorem binom_product :
  binom 10 3 * binom 8 3 = 6720 := by
  sorry

end binom_product_l35_35086


namespace fifth_root_of_unity_l35_35876

noncomputable def expression (x : ℂ) := 
  2 * x + 1 / (1 + x) + x / (1 + x^2) + x^2 / (1 + x^3) + x^3 / (1 + x^4)

theorem fifth_root_of_unity (x : ℂ) (hx : x^5 = 1) : 
  (expression x = 4) ∨ (expression x = -1 + Real.sqrt 5) ∨ (expression x = -1 - Real.sqrt 5) :=
sorry

end fifth_root_of_unity_l35_35876


namespace arithmetic_sequence_a1_value_l35_35475

theorem arithmetic_sequence_a1_value (a : ℕ → ℤ) (d : ℤ) 
  (h1 : a 3 = -6) 
  (h2 : a 7 = a 5 + 4) 
  (h_seq : ∀ n, a (n+1) = a n + d) : 
  a 1 = -10 := 
by
  sorry

end arithmetic_sequence_a1_value_l35_35475


namespace sum_of_numbers_of_large_cube_l35_35405

def sum_faces_of_die := 1 + 2 + 3 + 4 + 5 + 6

def number_of_dice := 125

def number_of_faces_per_die := 6

def total_exposed_faces (side_length: ℕ) : ℕ := 6 * (side_length * side_length)

theorem sum_of_numbers_of_large_cube (side_length : ℕ) (dice_count : ℕ) 
    (sum_per_die : ℕ) (opposite_face_sum : ℕ) :
    dice_count = 125 →
    total_exposed_faces side_length = 150 →
    sum_per_die = 21 →
    (∀ f1 f2, (f1 + f2 = opposite_face_sum)) →
    dice_count * sum_per_die = 2625 →
    (210 ≤ dice_count * sum_per_die ∧ dice_count * sum_per_die ≤ 840) :=
by 
  intro h_dice_count
  intro h_exposed_faces
  intro h_sum_per_die
  intro h_opposite_faces
  intro h_total_sum
  sorry

end sum_of_numbers_of_large_cube_l35_35405


namespace grooming_time_correct_l35_35769

def time_to_groom_poodle : ℕ := 30
def time_to_groom_terrier : ℕ := time_to_groom_poodle / 2
def number_of_poodles : ℕ := 3
def number_of_terriers : ℕ := 8

def total_grooming_time : ℕ :=
  (number_of_poodles * time_to_groom_poodle) + (number_of_terriers * time_to_groom_terrier)

theorem grooming_time_correct :
  total_grooming_time = 210 :=
by
  sorry

end grooming_time_correct_l35_35769


namespace arrangement_with_A_in_middle_arrangement_with_A_at_end_B_not_at_end_arrangement_with_A_B_adjacent_not_adjacent_to_C_l35_35070

-- Proof Problem 1
theorem arrangement_with_A_in_middle (products : Finset ℕ) (A : ℕ) (hA : A ∈ products) (arrangements : Finset (Fin 5 → ℕ)) :
  5 ∈ products ∧ (∀ a ∈ arrangements, a (Fin.mk 2 sorry) = A) →
  arrangements.card = 24 :=
by sorry

-- Proof Problem 2
theorem arrangement_with_A_at_end_B_not_at_end (products : Finset ℕ) (A B : ℕ) (hA : A ∈ products) (hB : B ∈ products) (arrangements : Finset (Fin 5 → ℕ)) :
  (5 ∈ products ∧ (∀ a ∈ arrangements, (a 0 = A ∨ a 4 = A) ∧ (a 1 ≠ B ∧ a 2 ≠ B ∧ a 3 ≠ B))) →
  arrangements.card = 36 :=
by sorry

-- Proof Problem 3
theorem arrangement_with_A_B_adjacent_not_adjacent_to_C (products : Finset ℕ) (A B C : ℕ) (hA : A ∈ products) (hB : B ∈ products) (hC : C ∈ products) (arrangements : Finset (Fin 5 → ℕ)) :
  (5 ∈ products ∧ (∀ a ∈ arrangements, ((a 0 = A ∧ a 1 = B) ∨ (a 1 = A ∧ a 2 = B) ∨ (a 2 = A ∧ a 3 = B) ∨ (a 3 = A ∧ a 4 = B)) ∧
   (a 0 ≠ A ∧ a 1 ≠ B ∧ a 2 ≠ C))) →
  arrangements.card = 36 :=
by sorry

end arrangement_with_A_in_middle_arrangement_with_A_at_end_B_not_at_end_arrangement_with_A_B_adjacent_not_adjacent_to_C_l35_35070


namespace gibi_percentage_is_59_l35_35593

-- Define the conditions
def max_score := 700
def avg_score := 490
def jigi_percent := 55
def mike_percent := 99
def lizzy_percent := 67

def jigi_score := (jigi_percent * max_score) / 100
def mike_score := (mike_percent * max_score) / 100
def lizzy_score := (lizzy_percent * max_score) / 100

def total_score := 4 * avg_score
def gibi_score := total_score - (jigi_score + mike_score + lizzy_score)

def gibi_percent := (gibi_score * 100) / max_score

-- The proof goal
theorem gibi_percentage_is_59 : gibi_percent = 59 := by
  sorry

end gibi_percentage_is_59_l35_35593


namespace tagged_fish_in_second_catch_l35_35757

theorem tagged_fish_in_second_catch (N : ℕ) (initially_tagged second_catch : ℕ)
  (h1 : N = 1250)
  (h2 : initially_tagged = 50)
  (h3 : second_catch = 50) :
  initially_tagged / N * second_catch = 2 :=
by
  sorry

end tagged_fish_in_second_catch_l35_35757


namespace total_units_in_building_l35_35022

theorem total_units_in_building (x y : ℕ) (cost_1_bedroom cost_2_bedroom total_cost : ℕ)
  (h1 : cost_1_bedroom = 360) (h2 : cost_2_bedroom = 450)
  (h3 : total_cost = 4950) (h4 : y = 7) (h5 : total_cost = cost_1_bedroom * x + cost_2_bedroom * y) :
  x + y = 12 :=
sorry

end total_units_in_building_l35_35022


namespace inequality_reciprocal_l35_35570

theorem inequality_reciprocal (a b c : ℝ) (h1 : a > b) (h2 : b > c) : 1 / (b - c) > 1 / (a - c) :=
sorry

end inequality_reciprocal_l35_35570


namespace circle_center_sum_l35_35518

theorem circle_center_sum (x y : ℝ) :
  (x^2 + y^2 = 10*x - 12*y + 40) →
  x + y = -1 :=
by {
  sorry
}

end circle_center_sum_l35_35518


namespace factorize_expression_l35_35500

theorem factorize_expression (a : ℝ) : 2 * a^2 - 8 = 2 * (a + 2) * (a - 2) :=
by
  sorry

end factorize_expression_l35_35500


namespace find_intersection_sums_l35_35346

noncomputable def cubic_expression (x : ℝ) : ℝ := x^3 - 4 * x^2 + 5 * x - 2
noncomputable def linear_expression (x : ℝ) : ℝ := -x / 2 + 1

theorem find_intersection_sums :
  (∃ x1 x2 x3 y1 y2 y3,
    cubic_expression x1 = linear_expression x1 ∧
    cubic_expression x2 = linear_expression x2 ∧
    cubic_expression x3 = linear_expression x3 ∧
    (x1 + x2 + x3 = 4) ∧ (y1 + y2 + y3 = 1)) :=
sorry

end find_intersection_sums_l35_35346


namespace last_two_digits_7_pow_2011_l35_35619

noncomputable def pow_mod_last_two_digits (n : ℕ) : ℕ :=
  (7^n) % 100

theorem last_two_digits_7_pow_2011 : pow_mod_last_two_digits 2011 = 43 :=
by
  sorry

end last_two_digits_7_pow_2011_l35_35619


namespace program_outputs_all_divisors_l35_35818

/--
  The function of the program is to output all divisors of \( n \), 
  given the initial conditions and operations in the program.
 -/
theorem program_outputs_all_divisors (n : ℕ) :
  ∀ I : ℕ, (1 ≤ I ∧ I ≤ n) → (∃ S : ℕ, (n % I = 0 ∧ S = I)) :=
by
  sorry

end program_outputs_all_divisors_l35_35818

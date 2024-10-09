import Mathlib

namespace races_needed_to_declare_winner_l1754_175432

noncomputable def total_sprinters : ℕ := 275
noncomputable def sprinters_per_race : ℕ := 7
noncomputable def sprinters_advance : ℕ := 2
noncomputable def sprinters_eliminated : ℕ := 5

theorem races_needed_to_declare_winner :
  (total_sprinters - 1 + sprinters_eliminated) / sprinters_eliminated = 59 :=
by
  sorry

end races_needed_to_declare_winner_l1754_175432


namespace quadratic_equal_real_roots_l1754_175467

theorem quadratic_equal_real_roots (m : ℝ) :
  (∃ x : ℝ, x^2 + x + m = 0 ∧ ∀ y : ℝ, y^2 + y + m = 0 → y = x) ↔ m = 1/4 :=
by sorry

end quadratic_equal_real_roots_l1754_175467


namespace eval_expr_l1754_175415

theorem eval_expr : 2 + 3 * 4 - 5 / 5 + 7 = 20 := by
  sorry

end eval_expr_l1754_175415


namespace arrange_polynomial_descending_l1754_175487

variable (a b : ℤ)

def polynomial := -a + 3 * a^5 * b^3 + 5 * a^3 * b^5 - 9 + 4 * a^2 * b^2 

def rearranged_polynomial := 3 * a^5 * b^3 + 5 * a^3 * b^5 + 4 * a^2 * b^2 - a - 9

theorem arrange_polynomial_descending :
  polynomial a b = rearranged_polynomial a b :=
sorry

end arrange_polynomial_descending_l1754_175487


namespace invalid_transformation_of_equation_l1754_175412

theorem invalid_transformation_of_equation (x y m : ℝ) (h : x = y) :
  (m = 0 → (x = y → x / m = y / m)) = false :=
by
  sorry

end invalid_transformation_of_equation_l1754_175412


namespace total_height_correct_l1754_175476

def height_of_stairs : ℕ := 10

def num_flights : ℕ := 3

def height_of_all_stairs : ℕ := height_of_stairs * num_flights

def height_of_rope : ℕ := height_of_all_stairs / 2

def extra_height_of_ladder : ℕ := 10

def height_of_ladder : ℕ := height_of_rope + extra_height_of_ladder

def total_height_climbed : ℕ := height_of_all_stairs + height_of_rope + height_of_ladder

theorem total_height_correct : total_height_climbed = 70 := by
  sorry

end total_height_correct_l1754_175476


namespace classroom_gpa_l1754_175488

theorem classroom_gpa (n : ℕ) (x : ℝ)
  (h1 : n > 0)
  (h2 : (1/3 : ℝ) * n * 45 + (2/3 : ℝ) * n * x = n * 55) : x = 60 :=
by
  sorry

end classroom_gpa_l1754_175488


namespace find_abc_sol_l1754_175453

theorem find_abc_sol (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (1 / ↑a + 1 / ↑b + 1 / ↑c = 1) →
  (a = 2 ∧ b = 3 ∧ c = 6) ∨ 
  (a = 2 ∧ b = 4 ∧ c = 4) ∨
  (a = 3 ∧ b = 3 ∧ c = 3) :=
by
  sorry

end find_abc_sol_l1754_175453


namespace factor_difference_of_squares_l1754_175456

theorem factor_difference_of_squares (t : ℝ) : t^2 - 144 = (t - 12) * (t + 12) :=
by
  sorry

end factor_difference_of_squares_l1754_175456


namespace least_number_remainder_l1754_175451

open Nat

theorem least_number_remainder (n : ℕ) :
  (n ≡ 4 [MOD 5]) →
  (n ≡ 4 [MOD 6]) →
  (n ≡ 4 [MOD 9]) →
  (n ≡ 4 [MOD 12]) →
  n = 184 :=
by
  intros h1 h2 h3 h4
  sorry

end least_number_remainder_l1754_175451


namespace milk_total_correct_l1754_175406

def chocolate_milk : Nat := 2
def strawberry_milk : Nat := 15
def regular_milk : Nat := 3
def total_milk : Nat := chocolate_milk + strawberry_milk + regular_milk

theorem milk_total_correct : total_milk = 20 := by
  sorry

end milk_total_correct_l1754_175406


namespace Heather_total_distance_walked_l1754_175464

theorem Heather_total_distance_walked :
  let d1 := 0.645
  let d2 := 1.235
  let d3 := 0.875
  let d4 := 1.537
  let d5 := 0.932
  (d1 + d2 + d3 + d4 + d5) = 5.224 := 
by
  sorry -- Proof goes here

end Heather_total_distance_walked_l1754_175464


namespace math_equivalence_proof_problem_l1754_175480

-- Define the initial radii in L0
def r1 := 50^2
def r2 := 53^2

-- Define the formula for constructing a new circle in subsequent layers
def next_radius (r1 r2 : ℕ) : ℕ :=
  (r1 * r2) / ((Nat.sqrt r1 + Nat.sqrt r2)^2)

-- Compute the sum of reciprocals of the square roots of the radii 
-- of all circles up to and including layer L6
def sum_of_reciprocals_of_square_roots_up_to_L6 : ℚ :=
  let initial_sum := (1 / (50 : ℚ)) + (1 / (53 : ℚ))
  (127 * initial_sum) / (50 * 53)

theorem math_equivalence_proof_problem : 
  sum_of_reciprocals_of_square_roots_up_to_L6 = 13021 / 2650 := 
sorry

end math_equivalence_proof_problem_l1754_175480


namespace total_perimeter_of_compound_shape_l1754_175499

-- Definitions of the conditions from the original problem
def triangle1_side : ℝ := 10
def triangle2_side : ℝ := 6
def shared_side : ℝ := 6

-- A theorem to represent the mathematically equivalent proof problem
theorem total_perimeter_of_compound_shape 
  (t1s : ℝ := triangle1_side) 
  (t2s : ℝ := triangle2_side)
  (ss : ℝ := shared_side) : 
  t1s = 10 ∧ t2s = 6 ∧ ss = 6 → 3 * t1s + 3 * t2s - ss = 42 := 
by
  sorry

end total_perimeter_of_compound_shape_l1754_175499


namespace product_of_two_numbers_l1754_175477

theorem product_of_two_numbers (x y : ℝ) (h1 : x^2 + y^2 = 289) (h2 : x + y = 23) : x * y = 120 :=
sorry

end product_of_two_numbers_l1754_175477


namespace num_distinct_intersections_l1754_175402

def linear_eq1 (x y : ℝ) := x + 2 * y - 10
def linear_eq2 (x y : ℝ) := x - 4 * y + 8
def linear_eq3 (x y : ℝ) := 2 * x - y - 1
def linear_eq4 (x y : ℝ) := 5 * x + 3 * y - 15

theorem num_distinct_intersections (n : ℕ) :
  (∀ x y : ℝ, linear_eq1 x y = 0 ∨ linear_eq2 x y = 0) ∧ 
  (∀ x y : ℝ, linear_eq3 x y = 0 ∨ linear_eq4 x y = 0) →
  n = 3 :=
  sorry

end num_distinct_intersections_l1754_175402


namespace initial_number_of_children_l1754_175417

-- Define the initial conditions
variables {X : ℕ} -- Initial number of children on the bus
variables (got_off got_on children_after : ℕ)
variables (H1 : got_off = 10)
variables (H2 : got_on = 5)
variables (H3 : children_after = 16)

-- Define the theorem to be proved
theorem initial_number_of_children (H : X - got_off + got_on = children_after) : X = 21 :=
by sorry

end initial_number_of_children_l1754_175417


namespace stack_crates_height_l1754_175479

theorem stack_crates_height :
  ∀ a b c : ℕ, (3 * a + 4 * b + 5 * c = 50) ∧ (a + b + c = 12) → false :=
by
  sorry

end stack_crates_height_l1754_175479


namespace rate_of_grapes_l1754_175423

theorem rate_of_grapes (G : ℝ) (H : 8 * G + 9 * 50 = 1010) : G = 70 := by
  sorry

end rate_of_grapes_l1754_175423


namespace smallest_int_remainder_two_l1754_175434

theorem smallest_int_remainder_two (m : ℕ) (hm : m > 1)
  (h3 : m % 3 = 2)
  (h4 : m % 4 = 2)
  (h5 : m % 5 = 2)
  (h6 : m % 6 = 2)
  (h7 : m % 7 = 2) :
  m = 422 :=
sorry

end smallest_int_remainder_two_l1754_175434


namespace determinant_zero_implies_sum_neg_nine_l1754_175405

theorem determinant_zero_implies_sum_neg_nine
  (x y : ℝ)
  (h1 : x ≠ y)
  (h2 : x * y = 1)
  (h3 : (Matrix.det ![
    ![1, 5, 8], 
    ![3, x, y], 
    ![3, y, x]
  ]) = 0) : 
  x + y = -9 := 
sorry

end determinant_zero_implies_sum_neg_nine_l1754_175405


namespace inequality_x4_y4_l1754_175492

theorem inequality_x4_y4 (x y : ℝ) : x^4 + y^4 + 8 ≥ 8 * x * y := 
by {
  sorry
}

end inequality_x4_y4_l1754_175492


namespace correct_divisor_l1754_175440

theorem correct_divisor :
  ∀ (D : ℕ), (D = 12 * 63) → (D = x * 36) → (x = 21) := 
by 
  intros D h1 h2
  sorry

end correct_divisor_l1754_175440


namespace jack_pays_back_total_l1754_175450

noncomputable def principal : ℝ := 1200
noncomputable def rate : ℝ := 0.10
noncomputable def interest : ℝ := principal * rate
noncomputable def total : ℝ := principal + interest

theorem jack_pays_back_total : total = 1320 := by
  sorry

end jack_pays_back_total_l1754_175450


namespace polynomial_simplification_l1754_175491

theorem polynomial_simplification (p : ℝ) :
  (4 * p^4 + 2 * p^3 - 7 * p + 3) + (5 * p^3 - 8 * p^2 + 3 * p + 2) = 
  4 * p^4 + 7 * p^3 - 8 * p^2 - 4 * p + 5 :=
by
  sorry

end polynomial_simplification_l1754_175491


namespace segment_ratios_correct_l1754_175433

noncomputable def compute_segment_ratios : (ℕ × ℕ) :=
  let ratio := 20 / 340;
  let gcd := Nat.gcd 1 17;
  if (ratio = 1 / 17) ∧ (gcd = 1) then (1, 17) else (0, 0) 

theorem segment_ratios_correct : 
  compute_segment_ratios = (1, 17) := 
by
  sorry

end segment_ratios_correct_l1754_175433


namespace vector_parallel_solution_l1754_175446

-- Define the vectors and the condition
def a (m : ℝ) := (2 * m + 1, 3)
def b (m : ℝ) := (2, m)

-- The proof problem statement
theorem vector_parallel_solution (m : ℝ) :
  (2 * m + 1) * m = 3 * 2 ↔ m = 3 / 2 ∨ m = -2 :=
by
  sorry

end vector_parallel_solution_l1754_175446


namespace nancy_games_this_month_l1754_175495

-- Define the variables and conditions from the problem
def went_games_last_month : ℕ := 8
def plans_games_next_month : ℕ := 7
def total_games : ℕ := 24

-- Let's calculate the games this month and state the theorem
def games_last_and_next : ℕ := went_games_last_month + plans_games_next_month
def games_this_month : ℕ := total_games - games_last_and_next

-- The theorem statement
theorem nancy_games_this_month : games_this_month = 9 := by
  -- Proof is omitted for the sake of brevity
  sorry

end nancy_games_this_month_l1754_175495


namespace sin_3x_over_4_period_l1754_175408

noncomputable def sine_period (b : ℝ) : ℝ :=
  (2 * Real.pi) / b

theorem sin_3x_over_4_period :
  sine_period (3/4) = (8 * Real.pi) / 3 :=
by
  sorry

end sin_3x_over_4_period_l1754_175408


namespace scientific_notation_280000_l1754_175454

theorem scientific_notation_280000 : 
  ∃ n: ℝ, n * 10^5 = 280000 ∧ n = 2.8 :=
by
-- our focus is on the statement outline, thus we use sorry to skip the proof part
  sorry

end scientific_notation_280000_l1754_175454


namespace min_value_of_quadratic_l1754_175445

noncomputable def f (x : ℝ) : ℝ := 3 * x^2 + 6 * x + 9

theorem min_value_of_quadratic : ∃ (x : ℝ), f x = 6 :=
by sorry

end min_value_of_quadratic_l1754_175445


namespace parabola_slopes_l1754_175414

theorem parabola_slopes (k : ℝ) (A B : ℝ × ℝ) (C : ℝ × ℝ) 
    (hC : C = (0, -2)) (hA : A.1^2 = 2 * A.2) (hB : B.1^2 = 2 * B.2) 
    (hA_eq : A.2 = k * A.1 + 2) (hB_eq : B.2 = k * B.1 + 2) :
  ((C.2 - A.2) / (C.1 - A.1))^2 + ((C.2 - B.2) / (C.1 - B.1))^2 - 2 * k^2 = 8 := 
sorry

end parabola_slopes_l1754_175414


namespace overall_average_marks_is_57_l1754_175483

-- Define the number of students and average mark per class
def students_class_A := 26
def avg_marks_class_A := 40

def students_class_B := 50
def avg_marks_class_B := 60

def students_class_C := 35
def avg_marks_class_C := 55

def students_class_D := 45
def avg_marks_class_D := 65

-- Define the total marks per class
def total_marks_class_A := students_class_A * avg_marks_class_A
def total_marks_class_B := students_class_B * avg_marks_class_B
def total_marks_class_C := students_class_C * avg_marks_class_C
def total_marks_class_D := students_class_D * avg_marks_class_D

-- Define the grand total of marks
def grand_total_marks := total_marks_class_A + total_marks_class_B + total_marks_class_C + total_marks_class_D

-- Define the total number of students
def total_students := students_class_A + students_class_B + students_class_C + students_class_D

-- Define the overall average marks
def overall_avg_marks := grand_total_marks / total_students

-- The target theorem we want to prove
theorem overall_average_marks_is_57 : overall_avg_marks = 57 := by
  sorry

end overall_average_marks_is_57_l1754_175483


namespace negation_of_proposition_l1754_175466

theorem negation_of_proposition (a b : ℝ) (h : a > b → a^2 > b^2) : a ≤ b → a^2 ≤ b^2 :=
by
  sorry

end negation_of_proposition_l1754_175466


namespace championship_outcomes_l1754_175496

theorem championship_outcomes :
  ∀ (students events : ℕ), students = 4 → events = 3 → students ^ events = 64 :=
by
  intros students events h_students h_events
  rw [h_students, h_events]
  exact rfl

end championship_outcomes_l1754_175496


namespace equation1_solution_equation2_solution_l1754_175475

variable (x : ℝ)

theorem equation1_solution :
  ((2 * x - 5) / 6 - (3 * x + 1) / 2 = 1) → (x = -2) :=
by
  sorry

theorem equation2_solution :
  (3 * x - 7 * (x - 1) = 3 - 2 * (x + 3)) → (x = 5) :=
by
  sorry

end equation1_solution_equation2_solution_l1754_175475


namespace sequence_formula_l1754_175452

theorem sequence_formula (a : ℕ → ℕ) (h₀ : a 1 = 2) 
  (h₁ : ∀ n : ℕ, a (n + 1) = a n ^ 2 - n * a n + 1) :
  ∀ n : ℕ, a n = n + 1 :=
by
  sorry

end sequence_formula_l1754_175452


namespace right_triangle_perimeter_l1754_175498

theorem right_triangle_perimeter (area leg1 : ℕ) (h_area : area = 180) (h_leg1 : leg1 = 30) :
  ∃ leg2 hypotenuse perimeter, 
    (2 * area = leg1 * leg2) ∧ 
    (hypotenuse^2 = leg1^2 + leg2^2) ∧ 
    (perimeter = leg1 + leg2 + hypotenuse) ∧ 
    (perimeter = 42 + 2 * Real.sqrt 261) :=
by
  sorry

end right_triangle_perimeter_l1754_175498


namespace isosceles_triangle_perimeter_l1754_175494

-- Define the given conditions
def is_isosceles_triangle (a b c : ℕ) : Prop :=
  (a = b ∨ b = c ∨ c = a)

def valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

-- Theorem based on the problem statement and conditions
theorem isosceles_triangle_perimeter (a b : ℕ) (P : is_isosceles_triangle a b 5) (Q : is_isosceles_triangle b a 10) :
  valid_triangle a b 5 → valid_triangle b a 10 → a + b + 5 = 25 :=
by sorry

end isosceles_triangle_perimeter_l1754_175494


namespace betty_needs_more_flies_l1754_175458

-- Definitions for the number of flies consumed by the frog each day
def fliesMonday : ℕ := 3
def fliesTuesday : ℕ := 2
def fliesWednesday : ℕ := 4
def fliesThursday : ℕ := 5
def fliesFriday : ℕ := 1
def fliesSaturday : ℕ := 2
def fliesSunday : ℕ := 3

-- Definition for the total number of flies eaten by the frog in a week
def totalFliesEaten : ℕ :=
  fliesMonday + fliesTuesday + fliesWednesday + fliesThursday + fliesFriday + fliesSaturday + fliesSunday

-- Definitions for the number of flies caught by Betty
def fliesMorning : ℕ := 5
def fliesAfternoon : ℕ := 6
def fliesEscaped : ℕ := 1

-- Definition for the total number of flies caught by Betty considering the escape
def totalFliesCaught : ℕ := fliesMorning + fliesAfternoon - fliesEscaped

-- Lean 4 statement to prove the number of additional flies Betty needs to catch
theorem betty_needs_more_flies : 
  totalFliesEaten - totalFliesCaught = 10 := 
by
  sorry

end betty_needs_more_flies_l1754_175458


namespace find_a4_l1754_175409

noncomputable def S : ℕ → ℤ
| 0 => 0
| 1 => -1
| n+1 => 3 * S n + 2^(n+1) - 3

def a : ℕ → ℤ
| 0 => 0
| 1 => -1
| n+1 => 3 * a n + 2^n

theorem find_a4 (h1 : ∀ n ≥ 2, S n = 3 * S (n - 1) + 2^n - 3) (h2 : a 1 = -1) : a 4 = 11 :=
by
  sorry

end find_a4_l1754_175409


namespace find_ravish_marks_l1754_175436

-- Define the data according to the conditions.
def max_marks : ℕ := 200
def passing_percentage : ℕ := 40
def failed_by : ℕ := 40

-- The main theorem we need to prove.
theorem find_ravish_marks (max_marks : ℕ) (passing_percentage : ℕ) (failed_by : ℕ) 
  (passing_marks := (max_marks * passing_percentage) / 100)
  (ravish_marks := passing_marks - failed_by) 
  : ravish_marks = 40 := by sorry

end find_ravish_marks_l1754_175436


namespace solution_for_a_if_fa_eq_a_l1754_175441

noncomputable def f (x : ℝ) : ℝ := (x^2 - x - 2) / (x - 2)

theorem solution_for_a_if_fa_eq_a (a : ℝ) (h : f a = a) : a = -1 :=
sorry

end solution_for_a_if_fa_eq_a_l1754_175441


namespace factorize_poly_l1754_175419

-- Statement of the problem
theorem factorize_poly (x : ℝ) : x^2 - 3 * x = x * (x - 3) :=
sorry

end factorize_poly_l1754_175419


namespace line_equation_mb_l1754_175459

theorem line_equation_mb (b m : ℤ) (h_b : b = -2) (h_m : m = 5) : m * b = -10 :=
by
  rw [h_b, h_m]
  norm_num

end line_equation_mb_l1754_175459


namespace measure_of_angle_C_l1754_175474

variable (a b c : ℝ) (S : ℝ)

-- Conditions
axiom triangle_sides : a > 0 ∧ b > 0 ∧ c > 0
axiom area_equation : S = (Real.sqrt 3 / 4) * (a^2 + b^2 - c^2)

-- The problem
theorem measure_of_angle_C (h₁: a > 0) (h₂: b > 0) (h₃: c > 0) (h₄: S = (Real.sqrt 3 / 4) * (a^2 + b^2 - c^2)) :
  ∃ C : ℝ, C = Real.arctan (Real.sqrt 3) ∧ C = Real.pi / 3 :=
by
  sorry

end measure_of_angle_C_l1754_175474


namespace net_profit_positive_max_average_net_profit_l1754_175447

def initial_investment : ℕ := 720000
def first_year_expense : ℕ := 120000
def annual_expense_increase : ℕ := 40000
def annual_sales : ℕ := 500000

def net_profit (n : ℕ) : ℕ := annual_sales - (first_year_expense + (n-1) * annual_expense_increase)
def average_net_profit (y n : ℕ) : ℕ := y / n

theorem net_profit_positive (n : ℕ) : net_profit n > 0 :=
sorry -- prove when net profit is positive

theorem max_average_net_profit (n : ℕ) : 
∀ m, average_net_profit (net_profit m) m ≤ average_net_profit (net_profit n) n :=
sorry -- prove when the average net profit is maximized

end net_profit_positive_max_average_net_profit_l1754_175447


namespace hard_candy_food_colouring_l1754_175430

theorem hard_candy_food_colouring :
  (∀ lollipop_colour hard_candy_count total_food_colouring lollipop_count hard_candy_food_total_per_lollipop,
    lollipop_colour = 5 →
    lollipop_count = 100 →
    hard_candy_count = 5 →
    total_food_colouring = 600 →
    hard_candy_food_total_per_lollipop = lollipop_colour * lollipop_count →
    total_food_colouring - hard_candy_food_total_per_lollipop = hard_candy_count * hard_candy_food_total_per_candy →
    hard_candy_food_total_per_candy = 20) :=
by
  sorry

end hard_candy_food_colouring_l1754_175430


namespace range_of_m_l1754_175461

noncomputable def f (x : ℝ) : ℝ :=
  if x >= -1 then x^2 + 3*x + 5 else (1/2)^x

theorem range_of_m (m : ℝ) : (∀ x : ℝ, f x > m^2 - m) ↔ -1 ≤ m ∧ m ≤ 2 := sorry

end range_of_m_l1754_175461


namespace payment_difference_correct_l1754_175439

noncomputable def prove_payment_difference (x : ℕ) (h₀ : x > 0) : Prop :=
  180 / x - 180 / (x + 2) = 3

theorem payment_difference_correct (x : ℕ) (h₀ : x > 0) : prove_payment_difference x h₀ :=
  by
    sorry

end payment_difference_correct_l1754_175439


namespace total_bees_l1754_175455

theorem total_bees 
    (B : ℕ) 
    (h1 : (1/5 : ℚ) * B + (1/3 : ℚ) * B + (2/5 : ℚ) * B + 1 = B) : 
    B = 15 := sorry

end total_bees_l1754_175455


namespace solution_to_problem_l1754_175493

theorem solution_to_problem
  {x y z : ℝ}
  (h1 : xy / (x + y) = 1 / 3)
  (h2 : yz / (y + z) = 1 / 5)
  (h3 : zx / (z + x) = 1 / 6) :
  xyz / (xy + yz + zx) = 1 / 7 :=
by sorry

end solution_to_problem_l1754_175493


namespace alice_catch_up_time_l1754_175469

def alice_speed : ℝ := 45
def tom_speed : ℝ := 15
def initial_distance : ℝ := 4
def minutes_per_hour : ℝ := 60

theorem alice_catch_up_time :
  (initial_distance / (alice_speed - tom_speed)) * minutes_per_hour = 8 :=
by
  sorry

end alice_catch_up_time_l1754_175469


namespace boy_actual_height_is_236_l1754_175435

def actual_height (n : ℕ) (incorrect_avg correct_avg wrong_height : ℕ) : ℕ :=
  let incorrect_total := n * incorrect_avg
  let correct_total := n * correct_avg
  let diff := incorrect_total - correct_total
  wrong_height + diff

theorem boy_actual_height_is_236 :
  ∀ (n incorrect_avg correct_avg wrong_height actual_height : ℕ),
  n = 35 → 
  incorrect_avg = 183 → 
  correct_avg = 181 → 
  wrong_height = 166 → 
  actual_height = wrong_height + (n * incorrect_avg - n * correct_avg) →
  actual_height = 236 :=
by
  intros n incorrect_avg correct_avg wrong_height actual_height hn hic hg hw ha
  rw [hn, hic, hg, hw] at ha
  -- At this point, we would normally proceed to prove the statement.
  -- However, as per the requirements, we just include "sorry" to skip the proof.
  sorry

end boy_actual_height_is_236_l1754_175435


namespace largest_side_of_enclosure_l1754_175421

theorem largest_side_of_enclosure (l w : ℕ) (h1 : 2 * l + 2 * w = 180) (h2 : l * w = 1800) : max l w = 60 := 
by 
  sorry

end largest_side_of_enclosure_l1754_175421


namespace ten_times_ten_thousand_ten_times_one_million_ten_times_ten_million_tens_of_thousands_in_hundred_million_l1754_175490

theorem ten_times_ten_thousand : 10 * 10000 = 100000 :=
by sorry

theorem ten_times_one_million : 10 * 1000000 = 10000000 :=
by sorry

theorem ten_times_ten_million : 10 * 10000000 = 100000000 :=
by sorry

theorem tens_of_thousands_in_hundred_million : 100000000 / 10000 = 10000 :=
by sorry

end ten_times_ten_thousand_ten_times_one_million_ten_times_ten_million_tens_of_thousands_in_hundred_million_l1754_175490


namespace intersection_eq_l1754_175422

noncomputable def A : Set ℝ := { x | x < 2 }
noncomputable def B : Set ℝ := {-1, 0, 1, 2}

theorem intersection_eq : A ∩ B = {-1, 0, 1} :=
by
  sorry

end intersection_eq_l1754_175422


namespace smallest_sum_l1754_175428

theorem smallest_sum (x y : ℕ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hxy_neq : x ≠ y) 
  (h_fraction : (1 : ℚ) / x + (1 : ℚ) / y = 1 / 15) : x + y = 64 :=
sorry

end smallest_sum_l1754_175428


namespace problem_1_problem_2_l1754_175482

def line1 (x y : ℝ) : Prop := 3 * x + 4 * y - 11 = 0
def line2 (x y : ℝ) : Prop := 2 * x + 3 * y - 8 = 0
def line3 (x y : ℝ) : Prop := 3 * x + 2 * y + 5 = 0
def point_M : ℝ × ℝ := (1, 2)
def point_P : ℝ × ℝ := (3, 1)

def line_l1 (x y : ℝ) : Prop := x + 2 * y - 5 = 0
def line_l2 (x y : ℝ) : Prop := 2 * x - 3 * y + 4 = 0

theorem problem_1 :
  (line1 point_M.1 point_M.2) ∧ (line2 point_M.1 point_M.2) → 
  (line_l1 point_P.1 point_P.2) ∧ (line_l1 point_M.1 point_M.2) :=
by 
  sorry

theorem problem_2 :
  (line1 point_M.1 point_M.2) ∧ (line2 point_M.1 point_M.2) →
  (∀ (x y : ℝ), line_l2 x y ↔ line3 x y) :=
by
  sorry

end problem_1_problem_2_l1754_175482


namespace dinosaur_count_l1754_175416

theorem dinosaur_count (h : ℕ) (l : ℕ) (H1 : h = 1) (H2 : l = 3) (total_hl : ℕ) (H3 : total_hl = 20) :
  ∃ D : ℕ, 4 * D = total_hl := 
by
  use 5
  sorry

end dinosaur_count_l1754_175416


namespace probability_not_e_after_n_spins_l1754_175420

theorem probability_not_e_after_n_spins
    (S : Type)
    (e b c d : S)
    (p_e : ℝ)
    (p_b : ℝ)
    (p_c : ℝ)
    (p_d : ℝ) :
    (p_e = 0.25) →
    (p_b = 0.25) →
    (p_c = 0.25) →
    (p_d = 0.25) →
    (1 - p_e)^2 = 0.5625 :=
by
  sorry

end probability_not_e_after_n_spins_l1754_175420


namespace range_of_a_l1754_175401

noncomputable def g (x : ℝ) : ℝ := abs (x-1) - abs (x-2)

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, ¬ (g x ≥ a^2 + a + 1)) ↔ (a < -1 ∨ a > 0) :=
by
  sorry

end range_of_a_l1754_175401


namespace grandson_age_l1754_175427

variable (G F : ℕ)

-- Define the conditions given in the problem
def condition1 := F = 6 * G
def condition2 := (F + 4) + (G + 4) = 78

-- The theorem to prove
theorem grandson_age : condition1 G F → condition2 G F → G = 10 :=
by
  intros h1 h2
  sorry

end grandson_age_l1754_175427


namespace factorial_not_multiple_of_57_l1754_175465

theorem factorial_not_multiple_of_57 (n : ℕ) (h : ¬ (57 ∣ n!)) : n < 19 := 
sorry

end factorial_not_multiple_of_57_l1754_175465


namespace price_of_A_correct_l1754_175400

noncomputable def A_price : ℝ := 25

theorem price_of_A_correct (H1 : 6000 / A_price - 4800 / (1.2 * A_price) = 80) 
                           (H2 : ∀ B_price : ℝ, B_price = 1.2 * A_price) : A_price = 25 := 
by
  sorry

end price_of_A_correct_l1754_175400


namespace number_of_factors_l1754_175449

theorem number_of_factors (b n : ℕ) (hb1 : b = 6) (hn1 : n = 15) (hb2 : b > 0) (hb3 : b ≤ 15) (hn2 : n > 0) (hn3 : n ≤ 15) :
  let factors := (15 + 1) * (15 + 1)
  factors = 256 :=
by
  sorry

end number_of_factors_l1754_175449


namespace sequence_length_l1754_175484

theorem sequence_length :
  ∃ n : ℕ, ∀ (a_1 : ℤ) (d : ℤ) (a_n : ℤ), a_1 = -6 → d = 4 → a_n = 50 → a_n = a_1 + (n - 1) * d ∧ n = 15 :=
by
  sorry

end sequence_length_l1754_175484


namespace complex_div_eq_l1754_175410

theorem complex_div_eq (z1 z2 : ℂ) (h1 : z1 = 3 - i) (h2 : z2 = 2 + i) :
  z1 / z2 = 1 - i := by
  sorry

end complex_div_eq_l1754_175410


namespace find_a_n_l1754_175429

theorem find_a_n (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h₁ : ∀ n, a n > 0)
  (h₂ : ∀ n, S n = 1/2 * (a n + 1 / (a n))) :
  ∀ n, a n = Real.sqrt n - Real.sqrt (n - 1) :=
by
  sorry

end find_a_n_l1754_175429


namespace exp_inequality_l1754_175462

theorem exp_inequality (n : ℕ) (h : 0 < n) : 2 ≤ (1 + 1 / (n : ℝ)) ^ n ∧ (1 + 1 / (n : ℝ)) ^ n < 3 :=
sorry

end exp_inequality_l1754_175462


namespace line_does_not_pass_first_quadrant_l1754_175481

open Real

theorem line_does_not_pass_first_quadrant (a b : ℝ) (h₁ : a > 0) (h₂ : b < 0) : 
  ¬∃ x y : ℝ, (x > 0) ∧ (y > 0) ∧ (ax + y - b = 0) :=
sorry

end line_does_not_pass_first_quadrant_l1754_175481


namespace delta_minus2_3_eq_minus14_l1754_175425

def delta (a b : Int) : Int := a * b^2 + b + 1

theorem delta_minus2_3_eq_minus14 : delta (-2) 3 = -14 :=
by
  sorry

end delta_minus2_3_eq_minus14_l1754_175425


namespace find_question_mark_l1754_175486

noncomputable def c1 : ℝ := (5568 / 87)^(1/3)
noncomputable def c2 : ℝ := (72 * 2)^(1/2)
noncomputable def sum_c1_c2 : ℝ := c1 + c2

theorem find_question_mark : sum_c1_c2 = 16 → 256 = 16^2 :=
by
  sorry

end find_question_mark_l1754_175486


namespace rounding_estimate_lt_exact_l1754_175448

variable (a b c a' b' c' : ℕ)

theorem rounding_estimate_lt_exact (ha : a' ≤ a) (hb : b' ≥ b) (hc : c' ≤ c) (hb_pos : b > 0) (hb'_pos : b' > 0) :
  (a':ℚ) / (b':ℚ) + (c':ℚ) < (a:ℚ) / (b:ℚ) + (c:ℚ) :=
sorry

end rounding_estimate_lt_exact_l1754_175448


namespace collinear_dot_probability_computation_l1754_175471

def collinear_dot_probability : ℚ := 12 / Nat.choose 25 5

theorem collinear_dot_probability_computation :
  collinear_dot_probability = 12 / 53130 :=
by
  -- This is where the proof steps would be if provided.
  sorry

end collinear_dot_probability_computation_l1754_175471


namespace x_in_interval_l1754_175407

theorem x_in_interval (x : ℝ) (h : x = (1 / x) * (-x) + 2) : 0 < x ∧ x ≤ 2 :=
by
  -- Place the proof here
  sorry

end x_in_interval_l1754_175407


namespace expand_expression_l1754_175426

theorem expand_expression (x : ℝ) : (x + 3) * (2 * x ^ 2 - x + 4) = 2 * x ^ 3 + 5 * x ^ 2 + x + 12 :=
by
  sorry

end expand_expression_l1754_175426


namespace binomial_expansion_value_l1754_175413

theorem binomial_expansion_value : 
  105^3 - 3 * 105^2 + 3 * 105 - 1 = 1124864 := by
  sorry

end binomial_expansion_value_l1754_175413


namespace triangle_perimeter_l1754_175444

theorem triangle_perimeter (L R B : ℕ) (hL : L = 12) (hR : R = L + 2) (hB : B = 24) : L + R + B = 50 :=
by
  -- proof steps go here
  sorry

end triangle_perimeter_l1754_175444


namespace dirocks_rectangular_fence_count_l1754_175403

/-- Dirock's backyard problem -/
def grid_side : ℕ := 32

def rock_placement (i j : ℕ) : Prop := (i % 3 = 0) ∧ (j % 3 = 0)

noncomputable def dirocks_rectangular_fence_ways : ℕ :=
  sorry

theorem dirocks_rectangular_fence_count : dirocks_rectangular_fence_ways = 1920 :=
sorry

end dirocks_rectangular_fence_count_l1754_175403


namespace a_and_b_finish_work_in_72_days_l1754_175457

noncomputable def work_rate_A_B {A B C : ℝ} 
  (h1 : B + C = 1 / 24) 
  (h2 : A + C = 1 / 36) 
  (h3 : A + B + C = 1 / 16.000000000000004) : ℝ :=
  A + B

theorem a_and_b_finish_work_in_72_days {A B C : ℝ} 
  (h1 : B + C = 1 / 24) 
  (h2 : A + C = 1 / 36) 
  (h3 : A + B + C = 1 / 16.000000000000004) : 
  work_rate_A_B h1 h2 h3 = 1 / 72 :=
sorry

end a_and_b_finish_work_in_72_days_l1754_175457


namespace one_fourth_of_7point2_is_9div5_l1754_175463

theorem one_fourth_of_7point2_is_9div5 : (7.2 / 4 : ℚ) = 9 / 5 := 
by sorry

end one_fourth_of_7point2_is_9div5_l1754_175463


namespace remainder_of_sum_of_consecutive_days_l1754_175431

theorem remainder_of_sum_of_consecutive_days :
  (100045 + 100046 + 100047 + 100048 + 100049 + 100050 + 100051 + 100052) % 5 = 3 :=
by
  sorry

end remainder_of_sum_of_consecutive_days_l1754_175431


namespace shorten_ellipse_parametric_form_l1754_175411

theorem shorten_ellipse_parametric_form :
  ∀ (θ : ℝ), 
  ∃ (x' y' : ℝ),
    x' = 4 * Real.cos θ ∧ y' = 2 * Real.sin θ ∧
    (∃ (x y : ℝ),
      x' = 2 * x ∧ y' = y ∧
      x = 2 * Real.cos θ ∧ y = 2 * Real.sin θ) :=
by
  sorry

end shorten_ellipse_parametric_form_l1754_175411


namespace angle_terminal_side_equiv_l1754_175497

theorem angle_terminal_side_equiv (α : ℝ) (k : ℤ) :
  (∃ k : ℤ, α = 30 + k * 360) ↔ (∃ β : ℝ, β = 30 ∧ α % 360 = β % 360) :=
by
  sorry

end angle_terminal_side_equiv_l1754_175497


namespace trapezoid_division_areas_l1754_175424

open Classical

variable (area_trapezoid : ℝ) (base1 base2 : ℝ)
variable (triangle1 triangle2 triangle3 triangle4 : ℝ)

theorem trapezoid_division_areas 
  (h1 : area_trapezoid = 3) 
  (h2 : base1 = 1) 
  (h3 : base2 = 2) 
  (h4 : triangle1 = 1 / 3)
  (h5 : triangle2 = 2 / 3)
  (h6 : triangle3 = 2 / 3)
  (h7 : triangle4 = 4 / 3) :
  triangle1 + triangle2 + triangle3 + triangle4 = area_trapezoid :=
by
  sorry

end trapezoid_division_areas_l1754_175424


namespace exactly_2_std_devs_less_than_mean_l1754_175485

noncomputable def mean : ℝ := 14.5
noncomputable def std_dev : ℝ := 1.5
noncomputable def value : ℝ := mean - 2 * std_dev

theorem exactly_2_std_devs_less_than_mean : value = 11.5 := by
  sorry

end exactly_2_std_devs_less_than_mean_l1754_175485


namespace negation_of_implication_l1754_175443

theorem negation_of_implication (x : ℝ) :
  ¬ (x ≠ 3 ∧ x ≠ 2 → x^2 - 5 * x + 6 ≠ 0) ↔ (x = 3 ∨ x = 2 → x^2 - 5 * x + 6 = 0) := 
by {
  sorry
}

end negation_of_implication_l1754_175443


namespace trick_deck_cost_l1754_175438

theorem trick_deck_cost :
  (∃ x : ℝ, 4 * x + 4 * x = 72) → ∃ x : ℝ, x = 9 := sorry

end trick_deck_cost_l1754_175438


namespace sequence_formula_l1754_175460

theorem sequence_formula (a : ℕ → ℕ) (h₀ : a 1 = 1)
  (h₁ : ∀ n : ℕ, a (n + 1) - 2 * a n + 3 = 0) :
  ∀ n : ℕ, a n = 3 - 2^n :=
by
  sorry

end sequence_formula_l1754_175460


namespace tempo_insured_fraction_l1754_175437

theorem tempo_insured_fraction (premium : ℝ) (rate : ℝ) (original_value : ℝ) (h1 : premium = 300) (h2 : rate = 0.03) (h3 : original_value = 14000) : 
  premium / rate / original_value = 5 / 7 :=
by 
  sorry

end tempo_insured_fraction_l1754_175437


namespace johns_total_working_hours_l1754_175442

theorem johns_total_working_hours (d h t : Nat) (h_d : d = 5) (h_h : h = 8) : t = d * h := by
  rewrite [h_d, h_h]
  sorry

end johns_total_working_hours_l1754_175442


namespace sin_cos_relation_l1754_175404

theorem sin_cos_relation 
  (α β : Real) 
  (h : 2 * Real.sin α - Real.cos β = 2) 
  : Real.sin α + 2 * Real.cos β = 1 ∨ Real.sin α + 2 * Real.cos β = -1 := 
sorry

end sin_cos_relation_l1754_175404


namespace trapezoid_median_l1754_175418

theorem trapezoid_median {BC AD : ℝ} (h AC CD : ℝ) (h_nonneg : h = 2) (AC_eq_CD : AC = 4) (BC_eq_0 : BC = 0) 
: (AD = 4 * Real.sqrt 3) → (median = 3 * Real.sqrt 3) := by
  sorry

end trapezoid_median_l1754_175418


namespace n_leq_84_l1754_175468

theorem n_leq_84 (n : ℕ) (hn : 0 < n) (h: (1 / 2 + 1 / 3 + 1 / 7 + 1 / ↑n : ℚ).den ≤ 1): n ≤ 84 :=
sorry

end n_leq_84_l1754_175468


namespace percentage_vanilla_orders_l1754_175473

theorem percentage_vanilla_orders 
  (V C : ℕ) 
  (h1 : V = 2 * C) 
  (h2 : V + C = 220) 
  (h3 : C = 22) : 
  (V * 100) / 220 = 20 := 
by 
  sorry

end percentage_vanilla_orders_l1754_175473


namespace smallest_k_sum_sequence_l1754_175489

theorem smallest_k_sum_sequence (n : ℕ) (h : 100 = (n + 1) * (2 * 9 + n) / 2) : k = 9 := 
sorry

end smallest_k_sum_sequence_l1754_175489


namespace smallest_n_l1754_175478

theorem smallest_n (n : ℕ) (h1: n ≥ 100) (h2: n ≤ 999) 
  (h3: (n + 5) % 8 = 0) (h4: (n - 8) % 5 = 0) : 
  n = 123 :=
sorry

end smallest_n_l1754_175478


namespace point_quadrant_l1754_175470

theorem point_quadrant (x y : ℝ) (h : (x + y)^2 = x^2 + y^2 - 2) : 
  ((x < 0 ∧ y > 0) ∨ (x > 0 ∧ y < 0)) :=
by
  sorry

end point_quadrant_l1754_175470


namespace ellipse_eccentricity_range_of_ratio_l1754_175472

-- The setup conditions
variables {a b c : ℝ}
variables (a_pos : a > 0) (b_pos : b > 0) (a_gt_b : a > b) (h1 : a^2 - b^2 = c^2)
variables (M : ℝ) (m : ℝ)
variables (hM : M = a + c) (hm : m = a - c) (hMm : M * m = 3 / 4 * a^2)

-- Proof statement for the eccentricity of the ellipse
theorem ellipse_eccentricity : c / a = 1 / 2 := by
  sorry

-- The setup for the second part
variables {S1 S2 : ℝ}
variables (ellipse_eq : ∀ x y : ℝ, (x^2 / (4 * c^2) + y^2 / (3 * c^2) = 1) → x + y = 0)
variables (range_S : S1 / S2 > 9)

-- Proof statement for the range of the given ratio
theorem range_of_ratio : 0 < (2 * S1 * S2) / (S1^2 + S2^2) ∧ (2 * S1 * S2) / (S1^2 + S2^2) < 9 / 41 := by
  sorry

end ellipse_eccentricity_range_of_ratio_l1754_175472

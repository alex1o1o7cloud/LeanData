import Mathlib

namespace NUMINAMATH_GPT_multiple_of_distance_l175_17553

namespace WalkProof

variable (H R M : ℕ)

/-- Rajesh walked 10 kilometers less than a certain multiple of the distance that Hiro walked. 
    Together they walked 25 kilometers. Rajesh walked 18 kilometers. 
    Prove that the multiple of the distance Hiro walked that Rajesh walked less than is 4. -/
theorem multiple_of_distance (h1 : R = M * H - 10) 
                             (h2 : H + R = 25)
                             (h3 : R = 18) :
                             M = 4 :=
by
  sorry

end WalkProof

end NUMINAMATH_GPT_multiple_of_distance_l175_17553


namespace NUMINAMATH_GPT_catherine_initial_pens_l175_17518

-- Defining the conditions
def equal_initial_pencils_and_pens (P : ℕ) : Prop := true
def pens_given_away_per_friend : ℕ := 8
def pencils_given_away_per_friend : ℕ := 6
def number_of_friends : ℕ := 7
def remaining_pens_and_pencils : ℕ := 22

-- The total number of items given away
def total_pens_given_away : ℕ := pens_given_away_per_friend * number_of_friends
def total_pencils_given_away : ℕ := pencils_given_away_per_friend * number_of_friends

-- The problem statement in Lean 4
theorem catherine_initial_pens (P : ℕ) 
  (h1 : equal_initial_pencils_and_pens P)
  (h2 : P - total_pens_given_away + P - total_pencils_given_away = remaining_pens_and_pencils) : 
  P = 60 :=
sorry

end NUMINAMATH_GPT_catherine_initial_pens_l175_17518


namespace NUMINAMATH_GPT_polygon_sides_l175_17564

-- Definition of the problem conditions
def interiorAngleSum (n : ℕ) : ℕ := 180 * (n - 2)
def givenAngleSum (n : ℕ) : ℕ := 140 + 145 * (n - 1)

-- Problem statement: proving the number of sides
theorem polygon_sides (n : ℕ) (h : interiorAngleSum n = givenAngleSum n) : n = 10 :=
sorry

end NUMINAMATH_GPT_polygon_sides_l175_17564


namespace NUMINAMATH_GPT_james_january_income_l175_17531

variable (January February March : ℝ)
variable (h1 : February = 2 * January)
variable (h2 : March = February - 2000)
variable (h3 : January + February + March = 18000)

theorem james_january_income : January = 4000 := by
  sorry

end NUMINAMATH_GPT_james_january_income_l175_17531


namespace NUMINAMATH_GPT_parabola_vertex_f_l175_17505

theorem parabola_vertex_f (d e f : ℝ) (h_vertex : ∀ y, (d * (y - 3)^2 + 5) = (d * y^2 + e * y + f))
  (h_point : d * (6 - 3)^2 + 5 = 2) : f = 2 :=
by
  sorry

end NUMINAMATH_GPT_parabola_vertex_f_l175_17505


namespace NUMINAMATH_GPT_sara_spent_correct_amount_on_movies_l175_17502

def cost_ticket : ℝ := 10.62
def num_tickets : ℕ := 2
def cost_rented_movie : ℝ := 1.59
def cost_purchased_movie : ℝ := 13.95

def total_amount_spent : ℝ :=
  num_tickets * cost_ticket + cost_rented_movie + cost_purchased_movie

theorem sara_spent_correct_amount_on_movies :
  total_amount_spent = 36.78 :=
sorry

end NUMINAMATH_GPT_sara_spent_correct_amount_on_movies_l175_17502


namespace NUMINAMATH_GPT_age_of_new_teacher_l175_17538

theorem age_of_new_teacher (sum_of_20_teachers : ℕ)
  (avg_age_20_teachers : ℕ)
  (total_teachers_after_new_teacher : ℕ)
  (new_avg_age_after_new_teacher : ℕ)
  (h1 : sum_of_20_teachers = 20 * 49)
  (h2 : avg_age_20_teachers = 49)
  (h3 : total_teachers_after_new_teacher = 21)
  (h4 : new_avg_age_after_new_teacher = 48) :
  ∃ (x : ℕ), x = 28 :=
by
  sorry

end NUMINAMATH_GPT_age_of_new_teacher_l175_17538


namespace NUMINAMATH_GPT_probability_is_8point64_percent_l175_17516

/-- Define the probabilities based on given conditions -/
def p_excel : ℝ := 0.45
def p_night_shift_given_excel : ℝ := 0.32
def p_no_weekend_given_night_shift : ℝ := 0.60

/-- Calculate the combined probability -/
def combined_probability :=
  p_excel * p_night_shift_given_excel * p_no_weekend_given_night_shift

theorem probability_is_8point64_percent :
  combined_probability = 0.0864 :=
by
  -- We will skip the proof for now
  sorry

end NUMINAMATH_GPT_probability_is_8point64_percent_l175_17516


namespace NUMINAMATH_GPT_sum_proof_l175_17510

-- Define the context and assumptions
variables (F S T : ℕ)
axiom sum_of_numbers : F + S + T = 264
axiom first_number_twice_second : F = 2 * S
axiom third_number_one_third_first : T = F / 3
axiom second_number_given : S = 72

-- The theorem to prove the sum is 264 given the conditions
theorem sum_proof : F + S + T = 264 :=
by
  -- Given conditions already imply the theorem, the actual proof follows from these
  sorry

end NUMINAMATH_GPT_sum_proof_l175_17510


namespace NUMINAMATH_GPT_rick_books_division_l175_17595

theorem rick_books_division (books_per_group initial_books final_groups : ℕ) 
  (h_initial : initial_books = 400) 
  (h_books_per_group : books_per_group = 25) 
  (h_final_groups : final_groups = 16) : 
  ∃ divisions : ℕ, (divisions = 4) ∧ 
    ∃ f : ℕ → ℕ, 
    (f 0 = initial_books) ∧ 
    (f divisions = books_per_group * final_groups) ∧ 
    (∀ n, 1 ≤ n → n ≤ divisions → f n = f (n - 1) / 2) := 
by 
  sorry

end NUMINAMATH_GPT_rick_books_division_l175_17595


namespace NUMINAMATH_GPT_number_of_baskets_l175_17506

def apples_per_basket : ℕ := 17
def total_apples : ℕ := 629

theorem number_of_baskets : total_apples / apples_per_basket = 37 :=
  by sorry

end NUMINAMATH_GPT_number_of_baskets_l175_17506


namespace NUMINAMATH_GPT_necessary_and_sufficient_condition_l175_17542

theorem necessary_and_sufficient_condition (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (|a + b| = |a| + |b|) ↔ (a * b > 0) :=
sorry

end NUMINAMATH_GPT_necessary_and_sufficient_condition_l175_17542


namespace NUMINAMATH_GPT_part_a_part_b_l175_17547

-- Definition of the function f and the condition it satisfies
variable (f : ℕ → ℕ)
variable (k n : ℕ)

theorem part_a (h1 : ∀ k n : ℕ, (k * f n) ≤ f (k * n) ∧ f (k * n) ≤ (k * f n) + k - 1)
  (a b : ℕ) :
  f a + f b ≤ f (a + b) ∧ f (a + b) ≤ f a + f b + 1 :=
by
  exact sorry  -- Proof to be supplied

theorem part_b (h1 : ∀ k n : ℕ, (k * f n) ≤ f (k * n) ∧ f (k * n) ≤ (k * f n) + k - 1)
  (h2 : ∀ n : ℕ, f (2007 * n) ≤ 2007 * f n + 200) :
  ∃ c : ℕ, f (2007 * c) = 2007 * f c :=
by
  exact sorry  -- Proof to be supplied

end NUMINAMATH_GPT_part_a_part_b_l175_17547


namespace NUMINAMATH_GPT_find_a_plus_b_l175_17545

theorem find_a_plus_b (a b : ℝ) (h1 : 2 * a = -6) (h2 : a^2 - b = 4) : a + b = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_a_plus_b_l175_17545


namespace NUMINAMATH_GPT_flour_per_new_bread_roll_l175_17572

theorem flour_per_new_bread_roll (p1 f1 p2 f2 c : ℚ)
  (h1 : p1 = 40)
  (h2 : f1 = 1 / 8)
  (h3 : p2 = 25)
  (h4 : c = p1 * f1)
  (h5 : c = p2 * f2) :
  f2 = 1 / 5 :=
by
  sorry

end NUMINAMATH_GPT_flour_per_new_bread_roll_l175_17572


namespace NUMINAMATH_GPT_amy_carl_distance_after_2_hours_l175_17555

-- Conditions
def amy_rate : ℤ := 1
def carl_rate : ℤ := 2
def amy_interval : ℤ := 20
def carl_interval : ℤ := 30
def time_hours : ℤ := 2
def minutes_per_hour : ℤ := 60

-- Derived values
def time_minutes : ℤ := time_hours * minutes_per_hour
def amy_distance : ℤ := time_minutes / amy_interval * amy_rate
def carl_distance : ℤ := time_minutes / carl_interval * carl_rate

-- Question and answer pair
def distance_amy_carl : ℤ := amy_distance + carl_distance
def expected_distance : ℤ := 14

-- The theorem to prove
theorem amy_carl_distance_after_2_hours : distance_amy_carl = expected_distance := by
  sorry

end NUMINAMATH_GPT_amy_carl_distance_after_2_hours_l175_17555


namespace NUMINAMATH_GPT_catch_up_time_l175_17536

def A_departure_time : ℕ := 8 * 60 -- in minutes
def B_departure_time : ℕ := 6 * 60 -- in minutes
def relative_speed (v : ℕ) : ℕ := 5 * v / 4 -- (2.5v effective) converted to integer math
def initial_distance (v : ℕ) : ℕ := 2 * v * 2 -- 4v distance (B's 2 hours lead)

theorem catch_up_time (v : ℕ) :  A_departure_time + ((initial_distance v * 4) / (relative_speed v - v)) = 1080 :=
by
  sorry

end NUMINAMATH_GPT_catch_up_time_l175_17536


namespace NUMINAMATH_GPT_first_train_speed_l175_17526

theorem first_train_speed:
  ∃ v : ℝ, 
    (∀ t : ℝ, t = 1 → (v * t) + (4 * v) = 200) ∧ 
    (∀ t : ℝ, t = 4 → 50 * t = 200) → 
    v = 40 :=
by {
 sorry
}

end NUMINAMATH_GPT_first_train_speed_l175_17526


namespace NUMINAMATH_GPT_man_is_older_by_22_l175_17515

/-- 
Given the present age of the son is 20 years and in two years the man's age will be 
twice the age of his son, prove that the man is 22 years older than his son.
-/
theorem man_is_older_by_22 (S M : ℕ) (h1 : S = 20) (h2 : M + 2 = 2 * (S + 2)) : M - S = 22 :=
by
  sorry  -- Proof will be provided here

end NUMINAMATH_GPT_man_is_older_by_22_l175_17515


namespace NUMINAMATH_GPT_smallest_positive_period_max_min_values_l175_17508

noncomputable def f (x a : ℝ) : ℝ :=
  (Real.cos x) * (2 * Real.sqrt 3 * Real.sin x - Real.cos x) + a * Real.sin x ^ 2

theorem smallest_positive_period (a : ℝ) (h : f (Real.pi / 12) a = 0) : 
  ∃ T : ℝ, T > 0 ∧ (∀ x, f (x + T) a = f x a) ∧ (∀ ε > 0, ε < T → ∃ y, y < T ∧ f y a ≠ f 0 a) := 
sorry

theorem max_min_values (a : ℝ) (h : f (Real.pi / 12) a = 0) :
  (∀ x ∈ Set.Icc (-Real.pi / 6) (Real.pi / 4), f x a ≤ Real.sqrt 3) ∧ 
  (∀ x ∈ Set.Icc (-Real.pi / 6) (Real.pi / 4), -2 ≤ f x a) := 
sorry

end NUMINAMATH_GPT_smallest_positive_period_max_min_values_l175_17508


namespace NUMINAMATH_GPT_triangle_properties_l175_17503

noncomputable def triangle_side_lengths (m1 m2 m3 : ℝ) : Prop :=
  ∃ a b c s,
    m1 = 20 ∧
    m2 = 24 ∧
    m3 = 30 ∧
    a = 36.28 ∧
    b = 30.24 ∧
    c = 24.19 ∧
    s = 362.84

theorem triangle_properties :
  triangle_side_lengths 20 24 30 :=
by
  sorry

end NUMINAMATH_GPT_triangle_properties_l175_17503


namespace NUMINAMATH_GPT_max_sum_of_integer_pairs_on_circle_l175_17500

theorem max_sum_of_integer_pairs_on_circle : 
  ∃ (x y : ℤ), x^2 + y^2 = 169 ∧ ∀ (a b : ℤ), a^2 + b^2 = 169 → x + y ≥ a + b :=
sorry

end NUMINAMATH_GPT_max_sum_of_integer_pairs_on_circle_l175_17500


namespace NUMINAMATH_GPT_collinear_vectors_x_eq_neg_two_l175_17581

theorem collinear_vectors_x_eq_neg_two (x : ℝ) (a b : ℝ×ℝ) :
  a = (1, 2) → b = (x, -4) → a.1 * b.2 = a.2 * b.1 → x = -2 :=
by
  intro ha hb hc
  sorry

end NUMINAMATH_GPT_collinear_vectors_x_eq_neg_two_l175_17581


namespace NUMINAMATH_GPT_field_area_is_36_square_meters_l175_17543

theorem field_area_is_36_square_meters (side_length : ℕ) (h : side_length = 6) : side_length * side_length = 36 :=
by
  sorry

end NUMINAMATH_GPT_field_area_is_36_square_meters_l175_17543


namespace NUMINAMATH_GPT_evaluate_ceiling_expression_l175_17580

theorem evaluate_ceiling_expression:
  (Int.ceil ((23 : ℚ) / 9 - Int.ceil ((35 : ℚ) / 23)))
  / (Int.ceil ((35 : ℚ) / 9 + Int.ceil ((9 * 23 : ℚ) / 35))) = 1 / 12 := by
  sorry

end NUMINAMATH_GPT_evaluate_ceiling_expression_l175_17580


namespace NUMINAMATH_GPT_tan_alpha_solution_l175_17587

variable (α : ℝ)
variable (h₀ : 0 < α ∧ α < π)
variable (h₁ : Real.sin α + Real.cos α = 7 / 13)

theorem tan_alpha_solution : Real.tan α = -12 / 5 := 
by
  sorry

end NUMINAMATH_GPT_tan_alpha_solution_l175_17587


namespace NUMINAMATH_GPT_number_of_students_l175_17585

theorem number_of_students (N T : ℕ) (h1 : T = 80 * N)
  (h2 : (T - 100) / (N - 5) = 90) : N = 35 := 
by 
  sorry

end NUMINAMATH_GPT_number_of_students_l175_17585


namespace NUMINAMATH_GPT_tony_initial_money_l175_17574

theorem tony_initial_money (ticket_cost hotdog_cost money_left initial_money : ℕ) 
  (h_ticket : ticket_cost = 8)
  (h_hotdog : hotdog_cost = 3) 
  (h_left : money_left = 9)
  (h_spent : initial_money = ticket_cost + hotdog_cost + money_left) :
  initial_money = 20 := 
by 
  sorry

end NUMINAMATH_GPT_tony_initial_money_l175_17574


namespace NUMINAMATH_GPT_coffee_on_Thursday_coffee_on_Friday_average_coffee_l175_17586

noncomputable def coffee_consumption (k h : ℝ) : ℝ := k / h

theorem coffee_on_Thursday : coffee_consumption 24 4 = 6 :=
by sorry

theorem coffee_on_Friday : coffee_consumption 24 10 = 2.4 :=
by sorry

theorem average_coffee : 
  (coffee_consumption 24 8 + coffee_consumption 24 4 + coffee_consumption 24 10) / 3 = 3.8 :=
by sorry

end NUMINAMATH_GPT_coffee_on_Thursday_coffee_on_Friday_average_coffee_l175_17586


namespace NUMINAMATH_GPT_isosceles_triangles_l175_17523

noncomputable def is_isosceles_triangle (a b c : ℝ) : Prop :=
  a = b ∨ b = c ∨ a = c

theorem isosceles_triangles (a b c : ℝ) (h : a ≥ b ∧ b ≥ c ∧ c > 0)
    (H : ∀ n : ℕ, a ^ n + b ^ n > c ^ n ∧ b ^ n + c ^ n > a ^ n ∧ c ^ n + a ^ n > b ^ n) :
    is_isosceles_triangle a b c :=
  sorry

end NUMINAMATH_GPT_isosceles_triangles_l175_17523


namespace NUMINAMATH_GPT_find_angle_C_l175_17535

variable (a b c : ℝ)
variable (A B C : ℝ)
variable (triangle_ABC : Type)

-- Given conditions
axiom ten_a_cos_B_eq_three_b_cos_A : 10 * a * Real.cos B = 3 * b * Real.cos A
axiom cos_A_value : Real.cos A = 5 * Real.sqrt 26 / 26

-- Required to prove
theorem find_angle_C : C = 3 * Real.pi / 4 := by
  sorry

end NUMINAMATH_GPT_find_angle_C_l175_17535


namespace NUMINAMATH_GPT_geometric_sequence_n_l175_17522

-- Definition of the conditions

-- a_1 + a_n = 82
def condition1 (a₁ an : ℕ) : Prop := a₁ + an = 82
-- a_3 * a_{n-2} = 81
def condition2 (a₃ aₙm2 : ℕ) : Prop := a₃ * aₙm2 = 81
-- S_n = 121
def condition3 (Sₙ : ℕ) : Prop := Sₙ = 121

-- Prove n = 5 given the above conditions
theorem geometric_sequence_n (a₁ a₃ an aₙm2 Sₙ n : ℕ)
  (h1 : condition1 a₁ an)
  (h2 : condition2 a₃ aₙm2)
  (h3 : condition3 Sₙ) :
  n = 5 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_n_l175_17522


namespace NUMINAMATH_GPT_billy_points_difference_l175_17579

-- Condition Definitions
def billy_points : ℕ := 7
def friend_points : ℕ := 9

-- Theorem stating the problem and the solution
theorem billy_points_difference : friend_points - billy_points = 2 :=
by 
  sorry

end NUMINAMATH_GPT_billy_points_difference_l175_17579


namespace NUMINAMATH_GPT_unit_digit_3_pow_2012_sub_1_l175_17509

theorem unit_digit_3_pow_2012_sub_1 :
  (3 ^ 2012 - 1) % 10 = 0 :=
sorry

end NUMINAMATH_GPT_unit_digit_3_pow_2012_sub_1_l175_17509


namespace NUMINAMATH_GPT_at_least_one_non_negative_l175_17575

variable (x : ℝ)
def a : ℝ := x^2 - 1
def b : ℝ := 2*x + 2

theorem at_least_one_non_negative (x : ℝ) : ¬ (a x < 0 ∧ b x < 0) :=
by
  sorry

end NUMINAMATH_GPT_at_least_one_non_negative_l175_17575


namespace NUMINAMATH_GPT_sum_of_a_for_unique_solution_l175_17578

theorem sum_of_a_for_unique_solution (a : ℝ) (h : (a + 12)^2 - 384 = 0) : 
  let a1 := -12 + 16 * Real.sqrt 6
  let a2 := -12 - 16 * Real.sqrt 6
  a1 + a2 = -24 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_a_for_unique_solution_l175_17578


namespace NUMINAMATH_GPT_inequality_proof_l175_17559

theorem inequality_proof (x y : ℝ) (h : x^8 + y^8 ≤ 1) : 
  x^12 - y^12 + 2 * x^6 * y^6 ≤ π / 2 :=
sorry

end NUMINAMATH_GPT_inequality_proof_l175_17559


namespace NUMINAMATH_GPT_Julie_initial_savings_l175_17593

theorem Julie_initial_savings (P r : ℝ) 
  (h1 : 100 = P * r * 2) 
  (h2 : 105 = P * (1 + r) ^ 2 - P) : 
  2 * P = 1000 :=
by
  sorry

end NUMINAMATH_GPT_Julie_initial_savings_l175_17593


namespace NUMINAMATH_GPT_geometric_sequence_sum_l175_17573

theorem geometric_sequence_sum (q a₁ : ℝ) (hq : q > 1) (h₁ : a₁ + a₁ * q^3 = 18) (h₂ : a₁^2 * q^3 = 32) :
  (a₁ * (1 - q^8) / (1 - q) = 510) :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l175_17573


namespace NUMINAMATH_GPT_problem_l175_17560

theorem problem (a b c : ℤ) :
  (∀ x : ℤ, x^2 + 19 * x + 88 = (x + a) * (x + b)) →
  (∀ x : ℤ, x^2 - 23 * x + 132 = (x - b) * (x - c)) →
  a + b + c = 31 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_problem_l175_17560


namespace NUMINAMATH_GPT_problem_statement_l175_17576

theorem problem_statement (a b : ℝ) (h1 : a^3 - b^3 = 2) (h2 : a^5 - b^5 ≥ 4) : a^2 + b^2 ≥ 2 := 
sorry

end NUMINAMATH_GPT_problem_statement_l175_17576


namespace NUMINAMATH_GPT_tv_purchase_price_correct_l175_17507

theorem tv_purchase_price_correct (x : ℝ) (h : (1.4 * x * 0.8 - x) = 270) : x = 2250 :=
by
  sorry

end NUMINAMATH_GPT_tv_purchase_price_correct_l175_17507


namespace NUMINAMATH_GPT_quadratic_c_over_b_l175_17557

theorem quadratic_c_over_b :
  ∃ (b c : ℤ), (x^2 + 500 * x + 1000 = (x + b)^2 + c) ∧ (c / b = -246) :=
by sorry

end NUMINAMATH_GPT_quadratic_c_over_b_l175_17557


namespace NUMINAMATH_GPT_equal_area_intersection_l175_17563

variable (p q r s : ℚ)
noncomputable def intersection_point (x y : ℚ) : Prop :=
  4 * x + 5 * p / q = 12 * p / q ∧ 8 * y = p 

theorem equal_area_intersection :
  intersection_point p q r s /\
  p + q + r + s = 60 := 
by 
  sorry

end NUMINAMATH_GPT_equal_area_intersection_l175_17563


namespace NUMINAMATH_GPT_ellipse_triangle_is_isosceles_right_l175_17511

theorem ellipse_triangle_is_isosceles_right (e : ℝ) (a b c k : ℝ)
  (H1 : e = (c / a))
  (H2 : e = (Real.sqrt 2) / 2)
  (H3 : b^2 = a^2 * (1 - e^2))
  (H4 : a = 2 * k)
  (H5 : b = k * Real.sqrt 2)
  (H6 : c = k * Real.sqrt 2) :
  (4 * k)^2 = (2 * (k * Real.sqrt 2))^2 + (2 * (k * Real.sqrt 2))^2 :=
by
  sorry

end NUMINAMATH_GPT_ellipse_triangle_is_isosceles_right_l175_17511


namespace NUMINAMATH_GPT_solve_equation_real_l175_17569

theorem solve_equation_real (x : ℝ) (h : (x ^ 2 - x + 1) * (3 * x ^ 2 - 10 * x + 3) = 20 * x ^ 2) :
    x = (5 + Real.sqrt 21) / 2 ∨ x = (5 - Real.sqrt 21) / 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_real_l175_17569


namespace NUMINAMATH_GPT_area_of_triangle_ABC_is_24_l175_17588

-- Define the vertices of the triangle
def A : ℝ × ℝ := (-2, 3)
def B : ℝ × ℝ := (6, 1)
def C : ℝ × ℝ := (10, 6)

-- Define the area calculation
def triangleArea (A B C : ℝ × ℝ) : ℝ :=
  let v := (A.1 - C.1, A.2 - C.2)
  let w := (B.1 - C.1, B.2 - C.2)
  0.5 * |(v.1 * w.2 - v.2 * w.1)|

theorem area_of_triangle_ABC_is_24 :
  triangleArea A B C = 24 := by
  sorry

end NUMINAMATH_GPT_area_of_triangle_ABC_is_24_l175_17588


namespace NUMINAMATH_GPT_sum_of_first_six_terms_geometric_sequence_l175_17561

-- conditions
def a : ℚ := 1/4
def r : ℚ := 1/4

-- geometric series sum function
def geom_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

-- target sum of first six terms
def S_6 : ℚ := geom_sum a r 6

-- proof statement
theorem sum_of_first_six_terms_geometric_sequence :
  S_6 = 1365 / 4096 :=
by 
  sorry

end NUMINAMATH_GPT_sum_of_first_six_terms_geometric_sequence_l175_17561


namespace NUMINAMATH_GPT_billy_age_is_45_l175_17519

variable (Billy_age Joe_age : ℕ)

-- Given conditions
def condition1 := Billy_age = 3 * Joe_age
def condition2 := Billy_age + Joe_age = 60
def condition3 := Billy_age > 60 / 2

-- Prove Billy's age is 45
theorem billy_age_is_45 (h1 : condition1 Billy_age Joe_age) (h2 : condition2 Billy_age Joe_age) (h3 : condition3 Billy_age) : Billy_age = 45 :=
by
  sorry

end NUMINAMATH_GPT_billy_age_is_45_l175_17519


namespace NUMINAMATH_GPT_multiplication_of_mixed_number_l175_17527

theorem multiplication_of_mixed_number :
  7 * (9 + 2/5 : ℚ) = 65 + 4/5 :=
by
  -- to start the proof
  sorry

end NUMINAMATH_GPT_multiplication_of_mixed_number_l175_17527


namespace NUMINAMATH_GPT_sum_of_first_twelve_multiples_of_18_l175_17513

-- Given conditions
def sum_of_first_n_positives (n : ℕ) : ℕ := n * (n + 1) / 2

def first_twelve_multiples_sum (k : ℕ) : ℕ := k * (sum_of_first_n_positives 12)

-- The question to prove
theorem sum_of_first_twelve_multiples_of_18 : first_twelve_multiples_sum 18 = 1404 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_first_twelve_multiples_of_18_l175_17513


namespace NUMINAMATH_GPT_xyz_squared_eq_one_l175_17552

theorem xyz_squared_eq_one (x y z : ℝ) (h_distinct : x ≠ y ∧ y ≠ z ∧ z ≠ x)
    (h_eq : ∃ k, x + (1 / y) = k ∧ y + (1 / z) = k ∧ z + (1 / x) = k) : 
    x^2 * y^2 * z^2 = 1 := 
  sorry

end NUMINAMATH_GPT_xyz_squared_eq_one_l175_17552


namespace NUMINAMATH_GPT_problem_a_problem_b_l175_17504

-- Define necessary elements for the problem
def is_divisible_by_seven (n : ℕ) : Prop := n % 7 = 0

-- Define the method to check divisibility by seven
noncomputable def check_divisibility_by_seven (n : ℕ) : ℕ :=
  let last_digit := n % 10
  let remaining_digits := n / 10
  remaining_digits - 2 * last_digit

-- Problem a: Prove that 4578 is divisible by 7
theorem problem_a : is_divisible_by_seven 4578 :=
  sorry

-- Problem b: Prove that there are 13 three-digit numbers of the form AB5 divisible by 7
theorem problem_b : ∃ (count : ℕ), count = 13 ∧ (∀ a b : ℕ, a ≠ 0 ∧ 1 ≤ a ∧ a < 10 ∧ 0 ≤ b ∧ b < 10 → is_divisible_by_seven (100 * a + 10 * b + 5) → count = count + 1) :=
  sorry

end NUMINAMATH_GPT_problem_a_problem_b_l175_17504


namespace NUMINAMATH_GPT_pushups_total_l175_17532

theorem pushups_total (z d e : ℕ)
  (hz : z = 44) 
  (hd : d = z + 58) 
  (he : e = 2 * d) : 
  z + d + e = 350 := by
  sorry

end NUMINAMATH_GPT_pushups_total_l175_17532


namespace NUMINAMATH_GPT_sequence_sum_is_25_div_3_l175_17530

noncomputable def sum_of_arithmetic_sequence (a n d : ℝ) : ℝ := (n / 2) * (2 * a + (n - 1) * d)

theorem sequence_sum_is_25_div_3 (a d : ℝ)
  (h1 : a + 4 * d = 1)
  (h2 : 3 * a + 15 * d = 2 * a + 8 * d) :
  sum_of_arithmetic_sequence a 10 d = 25 / 3 := by
  sorry

end NUMINAMATH_GPT_sequence_sum_is_25_div_3_l175_17530


namespace NUMINAMATH_GPT_initial_contribution_l175_17589

theorem initial_contribution (j k l : ℝ)
  (h1 : j + k + l = 1200)
  (h2 : j - 200 + 3 * (k + l) = 1800) :
  j = 800 :=
sorry

end NUMINAMATH_GPT_initial_contribution_l175_17589


namespace NUMINAMATH_GPT_consecutive_sunny_days_l175_17517

theorem consecutive_sunny_days (n_sunny_days : ℕ) (n_days_year : ℕ) (days_to_stay : ℕ) (condition1 : n_sunny_days = 350) (condition2 : n_days_year = 365) :
  days_to_stay = 32 :=
by
  sorry

end NUMINAMATH_GPT_consecutive_sunny_days_l175_17517


namespace NUMINAMATH_GPT_rooks_control_chosen_squares_l175_17520

theorem rooks_control_chosen_squares (n : Nat) 
  (chessboard : Fin (2 * n) × Fin (2 * n)) 
  (chosen_squares : Finset (Fin (2 * n) × Fin (2 * n))) 
  (h : chosen_squares.card = 3 * n) :
  ∃ rooks : Finset (Fin (2 * n) × Fin (2 * n)), rooks.card = n ∧
  ∀ (square : Fin (2 * n) × Fin (2 * n)), square ∈ chosen_squares → 
  (square ∈ rooks ∨ ∃ (rook : Fin (2 * n) × Fin (2 * n)) (hr : rook ∈ rooks), 
  rook.1 = square.1 ∨ rook.2 = square.2) :=
sorry

end NUMINAMATH_GPT_rooks_control_chosen_squares_l175_17520


namespace NUMINAMATH_GPT_water_consumption_correct_l175_17514

theorem water_consumption_correct (w n r : ℝ) 
  (hw : w = 21428) 
  (hn : n = 26848.55) 
  (hr : r = 302790.13) :
  w = 21428 ∧ n = 26848.55 ∧ r = 302790.13 :=
by 
  sorry

end NUMINAMATH_GPT_water_consumption_correct_l175_17514


namespace NUMINAMATH_GPT_find_k_l175_17584

variables (k : ℝ)
def vector_a : ℝ × ℝ := (1, 2)
def vector_b : ℝ × ℝ := (-3, 2)
def vector_k_a_plus_b (k : ℝ) : ℝ × ℝ := (k*1 + (-3), k*2 + 2)
def vector_a_minus_2b : ℝ × ℝ := (1 - 2*(-3), 2 - 2*2)

theorem find_k (h : (vector_k_a_plus_b k).fst * (vector_a_minus_2b).snd = (vector_k_a_plus_b k).snd * (vector_a_minus_2b).fst) : k = -1/2 :=
sorry

end NUMINAMATH_GPT_find_k_l175_17584


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l175_17537

def A : Set ℤ := {1, 2, -3}
def B : Set ℤ := {1, -4, 5}

theorem intersection_of_A_and_B : A ∩ B = {1} :=
by sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l175_17537


namespace NUMINAMATH_GPT_max_XG_l175_17524

theorem max_XG :
  ∀ (G X Y Z : ℝ),
    Y - X = 5 ∧ Z - Y = 3 ∧ (1 / G + 1 / (G - 5) + 1 / (G - 8) = 0) →
    G = 20 / 3 :=
by
  sorry

end NUMINAMATH_GPT_max_XG_l175_17524


namespace NUMINAMATH_GPT_janet_freelancer_income_difference_l175_17554

theorem janet_freelancer_income_difference :
  let hours_per_week := 40
  let current_job_hourly_rate := 30
  let freelancer_hourly_rate := 40
  let fica_taxes_per_week := 25
  let healthcare_premiums_per_month := 400
  let weeks_per_month := 4
  
  let current_job_weekly_income := hours_per_week * current_job_hourly_rate
  let current_job_monthly_income := current_job_weekly_income * weeks_per_month
  
  let freelancer_weekly_income := hours_per_week * freelancer_hourly_rate
  let freelancer_monthly_income := freelancer_weekly_income * weeks_per_month
  
  let freelancer_monthly_fica_taxes := fica_taxes_per_week * weeks_per_month
  let freelancer_total_additional_costs := freelancer_monthly_fica_taxes + healthcare_premiums_per_month
  
  let freelancer_net_monthly_income := freelancer_monthly_income - freelancer_total_additional_costs
  
  freelancer_net_monthly_income - current_job_monthly_income = 1100 :=
by
  sorry

end NUMINAMATH_GPT_janet_freelancer_income_difference_l175_17554


namespace NUMINAMATH_GPT_solve_abs_eq_l175_17556

theorem solve_abs_eq (x : ℝ) (h : |x - 1| = 2 * x) : x = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_abs_eq_l175_17556


namespace NUMINAMATH_GPT_model_distance_comparison_l175_17539

theorem model_distance_comparison (m h c x y z : ℝ) (hm : 0 < m) (hh : 0 < h) (hc : 0 < c) (hz : 0 < z) (hx : 0 < x) (hy : 0 < y)
    (h_eq : (x - c) * z = (y - c) * (z + m) + h) :
    (if h > c * m then (x * z > y * (z + m))
     else if h < c * m then (x * z < y * (z + m))
     else (h = c * m → x * z = y * (z + m))) :=
by
  sorry

end NUMINAMATH_GPT_model_distance_comparison_l175_17539


namespace NUMINAMATH_GPT_rectangular_solid_surface_area_l175_17525

theorem rectangular_solid_surface_area (a b c : ℕ) (h_a_prime : Nat.Prime a) (h_b_prime : Nat.Prime b) (h_c_prime : Nat.Prime c) 
  (volume_eq : a * b * c = 273) :
  2 * (a * b + b * c + c * a) = 302 := 
sorry

end NUMINAMATH_GPT_rectangular_solid_surface_area_l175_17525


namespace NUMINAMATH_GPT_hyperbola_center_l175_17533

theorem hyperbola_center (x1 y1 x2 y2 : ℝ) (h₁ : x1 = 3) (h₂ : y1 = 2) (h₃ : x2 = 11) (h₄ : y2 = 6) :
  (x1 + x2) / 2 = 7 ∧ (y1 + y2) / 2 = 4 :=
by
  -- Use the conditions h₁, h₂, h₃, and h₄ to substitute values and prove the statement
  sorry

end NUMINAMATH_GPT_hyperbola_center_l175_17533


namespace NUMINAMATH_GPT_middle_aged_participating_l175_17592

-- Definitions of the given conditions
def total_employees : Nat := 1200
def ratio (elderly middle_aged young : Nat) := elderly = 1 ∧ middle_aged = 5 ∧ young = 6
def selected_employees : Nat := 36

-- The stratified sampling condition implies
def stratified_sampling (elderly middle_aged young : Nat) (total : Nat) (selected : Nat) :=
  (elderly + middle_aged + young = total) ∧
  (selected = 36)

-- The proof statement
theorem middle_aged_participating (elderly middle_aged young : Nat) (total : Nat) (selected : Nat) 
  (h_ratio : ratio elderly middle_aged young) 
  (h_total : total = total_employees)
  (h_sampled : stratified_sampling elderly middle_aged young (elderly + middle_aged + young) selected) : 
  selected * middle_aged / (elderly + middle_aged + young) = 15 := 
by sorry

end NUMINAMATH_GPT_middle_aged_participating_l175_17592


namespace NUMINAMATH_GPT_sum_of_coefficients_l175_17577

-- Definition of the binomial coefficient
def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem sum_of_coefficients (n : ℕ) (hn1 : 5 < n) (hn2 : n < 7)
  (coeff_cond : binom n 3 > binom n 2 ∧ binom n 3 > binom n 4) :
  (1 + 1)^n = 64 :=
by
  have h : n = 6 :=
    by sorry -- provided conditions force n to be 6
  show 2^n = 64
  rw [h]
  exact rfl

end NUMINAMATH_GPT_sum_of_coefficients_l175_17577


namespace NUMINAMATH_GPT_strawberries_to_grapes_ratio_l175_17521

-- Define initial conditions
def initial_grapes : ℕ := 100
def fruits_left : ℕ := 96

-- Define the number of strawberries initially
def strawberries_init (S : ℕ) : Prop :=
  (S - (2 * (1/5) * S) = fruits_left - initial_grapes + ((2 * (1/5)) * initial_grapes))

-- Define the ratio problem in Lean
theorem strawberries_to_grapes_ratio (S : ℕ) (h : strawberries_init S) : (S / initial_grapes = 3 / 5) :=
sorry

end NUMINAMATH_GPT_strawberries_to_grapes_ratio_l175_17521


namespace NUMINAMATH_GPT_cylinder_height_l175_17534

theorem cylinder_height (r h : ℝ) (SA : ℝ) (h₀ : r = 3) (h₁ : SA = 36 * Real.pi) (h₂ : SA = 2 * Real.pi * r^2 + 2 * Real.pi * r * h) : h = 3 :=
by
  -- The proof will be constructed here
  sorry

end NUMINAMATH_GPT_cylinder_height_l175_17534


namespace NUMINAMATH_GPT_shift_right_inverse_exp_eq_ln_l175_17528

variable (f : ℝ → ℝ)

theorem shift_right_inverse_exp_eq_ln :
  (∀ x, f (x - 1) = Real.log x) → ∀ x, f x = Real.log (x + 1) :=
by
  sorry

end NUMINAMATH_GPT_shift_right_inverse_exp_eq_ln_l175_17528


namespace NUMINAMATH_GPT_find_middle_number_l175_17590

namespace Problem

-- Define the three numbers x, y, z
variables (x y z : ℕ)

-- Given conditions from the problem
def condition1 (h1 : x + y = 18) := x + y = 18
def condition2 (h2 : x + z = 23) := x + z = 23
def condition3 (h3 : y + z = 27) := y + z = 27
def condition4 (h4 : x < y ∧ y < z) := x < y ∧ y < z

-- Statement to prove:
theorem find_middle_number (h1 : x + y = 18) (h2 : x + z = 23) (h3 : y + z = 27) (h4 : x < y ∧ y < z) : 
  y = 11 :=
by
  sorry

end Problem

end NUMINAMATH_GPT_find_middle_number_l175_17590


namespace NUMINAMATH_GPT_cube_surface_area_l175_17599

theorem cube_surface_area (V : ℝ) (hV : V = 125) : ∃ A : ℝ, A = 25 :=
by
  sorry

end NUMINAMATH_GPT_cube_surface_area_l175_17599


namespace NUMINAMATH_GPT_polynomial_coeff_sum_l175_17546

/-- 
Given that the product of the polynomials (4x^2 - 6x + 5)(8 - 3x) can be written as
ax^3 + bx^2 + cx + d, prove that 9a + 3b + c + d = 19.
-/
theorem polynomial_coeff_sum :
  ∃ a b c d : ℝ, 
  (∀ x : ℝ, (4 * x^2 - 6 * x + 5) * (8 - 3 * x) = a * x^3 + b * x^2 + c * x + d) ∧
  9 * a + 3 * b + c + d = 19 :=
sorry

end NUMINAMATH_GPT_polynomial_coeff_sum_l175_17546


namespace NUMINAMATH_GPT_second_largest_div_second_smallest_l175_17597

theorem second_largest_div_second_smallest : 
  let a := 10
  let b := 11
  let c := 12
  ∃ second_smallest second_largest, 
    second_smallest = b ∧ second_largest = b ∧ second_largest / second_smallest = 1 := 
by
  let a := 10
  let b := 11
  let c := 12
  use b
  use b
  exact ⟨rfl, rfl, rfl⟩

end NUMINAMATH_GPT_second_largest_div_second_smallest_l175_17597


namespace NUMINAMATH_GPT_total_amount_spent_l175_17548

def price_per_deck (n : ℕ) : ℝ :=
if n <= 3 then 8 else if n <= 6 then 7 else 6

def promotion_price (price : ℝ) : ℝ :=
price * 0.5

def total_cost (decks_victor decks_friend : ℕ) : ℝ :=
let cost_victor :=
  if decks_victor % 2 = 0 then
    let pairs := decks_victor / 2
    price_per_deck decks_victor * pairs + promotion_price (price_per_deck decks_victor) * pairs
  else sorry
let cost_friend :=
  if decks_friend = 2 then
    price_per_deck decks_friend + promotion_price (price_per_deck decks_friend)
  else sorry
cost_victor + cost_friend

theorem total_amount_spent : total_cost 6 2 = 43.5 := sorry

end NUMINAMATH_GPT_total_amount_spent_l175_17548


namespace NUMINAMATH_GPT_proportion_red_MMs_l175_17598

theorem proportion_red_MMs (R B : ℝ) (h1 : R + B = 1) 
  (h2 : R * (4 / 5) = B * (1 / 6)) :
  R = 5 / 29 :=
by
  sorry

end NUMINAMATH_GPT_proportion_red_MMs_l175_17598


namespace NUMINAMATH_GPT_arthur_additional_muffins_l175_17540

/-- Define the number of muffins Arthur has already baked -/
def muffins_baked : ℕ := 80

/-- Define the multiplier for the total output Arthur wants -/
def desired_multiplier : ℝ := 2.5

/-- Define the equation representing the total desired muffins -/
def total_muffins : ℝ := muffins_baked * desired_multiplier

/-- Define the number of additional muffins Arthur needs to bake -/
def additional_muffins : ℝ := total_muffins - muffins_baked

theorem arthur_additional_muffins : additional_muffins = 120 := by
  sorry

end NUMINAMATH_GPT_arthur_additional_muffins_l175_17540


namespace NUMINAMATH_GPT_parallel_lines_eq_a2_l175_17562

theorem parallel_lines_eq_a2
  (a : ℝ)
  (h : ∀ x y : ℝ, x + a * y - 1 = 0 → (a - 1) * x + a * y + 1 = 0)
  : a = 2 := 
  sorry

end NUMINAMATH_GPT_parallel_lines_eq_a2_l175_17562


namespace NUMINAMATH_GPT_angle_ACD_measure_l175_17571

theorem angle_ACD_measure {ABD BAE ABC ACD : ℕ} 
  (h1 : ABD = 125) 
  (h2 : BAE = 95) 
  (h3 : ABC = 180 - ABD) 
  (h4 : ABD + ABC = 180 ) : 
  ACD = 180 - (BAE + ABC) :=
by 
  sorry

end NUMINAMATH_GPT_angle_ACD_measure_l175_17571


namespace NUMINAMATH_GPT_total_fish_catch_l175_17529

noncomputable def Johnny_fishes : ℕ := 8
noncomputable def Sony_fishes : ℕ := 4 * Johnny_fishes
noncomputable def total_fishes : ℕ := Sony_fishes + Johnny_fishes

theorem total_fish_catch : total_fishes = 40 := by
  sorry

end NUMINAMATH_GPT_total_fish_catch_l175_17529


namespace NUMINAMATH_GPT_inequality_proof_l175_17565

open Real

theorem inequality_proof 
  (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_cond : a^2 + b^2 + c^2 = 3) :
  (a / (a + 5) + b / (b + 5) + c / (c + 5) ≤ 1 / 2) :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l175_17565


namespace NUMINAMATH_GPT_intersection_of_lines_l175_17544

theorem intersection_of_lines :
  ∃ x y : ℚ, 3 * y = -2 * x + 6 ∧ 2 * y = 6 * x - 4 ∧ x = 12 / 11 ∧ y = 14 / 11 := by
  sorry

end NUMINAMATH_GPT_intersection_of_lines_l175_17544


namespace NUMINAMATH_GPT_prove_distance_uphill_l175_17550

noncomputable def distance_uphill := 
  let flat_speed := 20
  let uphill_speed := 12
  let extra_flat_distance := 30
  let uphill_time (D : ℝ) := D / uphill_speed
  let flat_time (D : ℝ) := (D + extra_flat_distance) / flat_speed
  ∃ D : ℝ, uphill_time D = flat_time D ∧ D = 45

theorem prove_distance_uphill : distance_uphill :=
sorry

end NUMINAMATH_GPT_prove_distance_uphill_l175_17550


namespace NUMINAMATH_GPT_find_remaining_area_l175_17582

theorem find_remaining_area 
    (base_RST : ℕ) 
    (height_RST : ℕ) 
    (base_RSC : ℕ) 
    (height_RSC : ℕ) 
    (area_RST : ℕ := (1 / 2) * base_RST * height_RST) 
    (area_RSC : ℕ := (1 / 2) * base_RSC * height_RSC) 
    (remaining_area : ℕ := area_RST - area_RSC) 
    (h_base_RST : base_RST = 5) 
    (h_height_RST : height_RST = 4) 
    (h_base_RSC : base_RSC = 1) 
    (h_height_RSC : height_RSC = 4) : 
    remaining_area = 8 := 
by 
  sorry

end NUMINAMATH_GPT_find_remaining_area_l175_17582


namespace NUMINAMATH_GPT_simplify_expression_l175_17512

theorem simplify_expression (x : ℝ) : 3 * (5 - 2 * x) - 2 * (4 + 3 * x) = 7 - 12 * x := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l175_17512


namespace NUMINAMATH_GPT_find_number_l175_17567

theorem find_number (x : ℝ) : (5 / 3) * x = 45 → x = 27 := by
  sorry

end NUMINAMATH_GPT_find_number_l175_17567


namespace NUMINAMATH_GPT_greatest_number_in_consecutive_multiples_l175_17591

theorem greatest_number_in_consecutive_multiples (s : Set ℕ) (h₁ : ∃ m : ℕ, s = {n | ∃ k < 100, n = 8 * (m + k)} ∧ m = 14) :
  (∃ n ∈ s, ∀ x ∈ s, x ≤ n) →
  ∃ n ∈ s, n = 904 :=
by
  sorry

end NUMINAMATH_GPT_greatest_number_in_consecutive_multiples_l175_17591


namespace NUMINAMATH_GPT_matrix_det_problem_l175_17568

-- Define the determinant of a 2x2 matrix
def det (a b c d : ℤ) : ℤ := a * d - b * c

-- State the problem in Lean
theorem matrix_det_problem : 2 * det 5 7 2 3 = 2 := by
  sorry

end NUMINAMATH_GPT_matrix_det_problem_l175_17568


namespace NUMINAMATH_GPT_quadratic_equation_unique_solution_l175_17583

theorem quadratic_equation_unique_solution 
  (a c : ℝ) (h1 : ∃ x : ℝ, a * x^2 + 8 * x + c = 0)
  (h2 : a + c = 10)
  (h3 : a < c) :
  (a, c) = (2, 8) := 
sorry

end NUMINAMATH_GPT_quadratic_equation_unique_solution_l175_17583


namespace NUMINAMATH_GPT_total_people_in_class_l175_17501

-- Define the number of people based on their interests
def likes_both: Nat := 5
def only_baseball: Nat := 2
def only_football: Nat := 3
def likes_neither: Nat := 6

-- Define the total number of people in the class
def total_people := likes_both + only_baseball + only_football + likes_neither

-- Theorem statement
theorem total_people_in_class : total_people = 16 :=
by
  -- Proof is skipped
  sorry

end NUMINAMATH_GPT_total_people_in_class_l175_17501


namespace NUMINAMATH_GPT_cube_root_of_sum_of_powers_l175_17594

theorem cube_root_of_sum_of_powers :
  ∃ (x : ℝ), x = 16 * (4 ^ (1 / 3)) ∧ x = (4^6 + 4^6 + 4^6 + 4^6) ^ (1 / 3) :=
by
  sorry

end NUMINAMATH_GPT_cube_root_of_sum_of_powers_l175_17594


namespace NUMINAMATH_GPT_xiaolong_correct_answers_l175_17551

/-- There are 50 questions in the exam. Correct answers earn 3 points each,
incorrect answers deduct 1 point each, and unanswered questions score 0 points.
Xiaolong scored 120 points. Prove that the maximum number of questions 
Xiaolong answered correctly is 42. -/
theorem xiaolong_correct_answers :
  ∃ (x y : ℕ), 3 * x - y = 120 ∧ x + y = 48 ∧ x ≤ 50 ∧ y ≤ 50 ∧ x = 42 :=
by
  sorry

end NUMINAMATH_GPT_xiaolong_correct_answers_l175_17551


namespace NUMINAMATH_GPT_josh_500_coins_impossible_l175_17566

theorem josh_500_coins_impossible : ¬ ∃ (x y : ℕ), x + y ≤ 500 ∧ 36 * x + 6 * y + (500 - x - y) = 3564 := 
sorry

end NUMINAMATH_GPT_josh_500_coins_impossible_l175_17566


namespace NUMINAMATH_GPT_japanese_turtle_crane_problem_l175_17541

theorem japanese_turtle_crane_problem (x y : ℕ) (h1 : x + y = 35) (h2 : 2 * x + 4 * y = 94) : x + y = 35 ∧ 2 * x + 4 * y = 94 :=
by
  sorry

end NUMINAMATH_GPT_japanese_turtle_crane_problem_l175_17541


namespace NUMINAMATH_GPT_area_difference_l175_17596

-- Setting up the relevant conditions and entities
def side_red := 8
def length_yellow := 10
def width_yellow := 5

-- Definition of areas
def area_red := side_red * side_red
def area_yellow := length_yellow * width_yellow

-- The theorem we need to prove
theorem area_difference :
  area_red - area_yellow = 14 :=
by
  -- We skip the proof here due to the instruction
  sorry

end NUMINAMATH_GPT_area_difference_l175_17596


namespace NUMINAMATH_GPT_expression_evaluation_l175_17549

theorem expression_evaluation : (6 * 111) - (2 * 111) = 444 :=
by
  sorry

end NUMINAMATH_GPT_expression_evaluation_l175_17549


namespace NUMINAMATH_GPT_true_proposition_l175_17558

-- Define the propositions p and q
def p : Prop := 2 % 2 = 0
def q : Prop := 5 % 2 = 0

-- Define the problem statement
theorem true_proposition (hp : p) (hq : ¬ q) : p ∨ q :=
by
  sorry

end NUMINAMATH_GPT_true_proposition_l175_17558


namespace NUMINAMATH_GPT_baker_made_cakes_l175_17570

theorem baker_made_cakes (sold_cakes left_cakes total_cakes : ℕ) (h1 : sold_cakes = 108) (h2 : left_cakes = 59) :
  total_cakes = sold_cakes + left_cakes → total_cakes = 167 := by
  intro h
  rw [h1, h2] at h
  exact h

-- The proof part is omitted since only the statement is required

end NUMINAMATH_GPT_baker_made_cakes_l175_17570

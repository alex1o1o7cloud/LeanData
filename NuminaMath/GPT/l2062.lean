import Mathlib

namespace points_on_same_line_l2062_206254

theorem points_on_same_line (k : ℤ) : 
  (∃ m b : ℤ, ∀ p : ℤ × ℤ, p = (1, 4) ∨ p = (3, -2) ∨ p = (6, k / 3) → p.2 = m * p.1 + b) ↔ k = -33 :=
by
  sorry

end points_on_same_line_l2062_206254


namespace constant_term_of_binomial_expansion_l2062_206220

noncomputable def constant_in_binomial_expansion (a : ℝ) : ℝ := 
  if h : a = ∫ (x : ℝ) in (0)..(1), 2 * x 
  then ((1 : ℝ) - (a : ℝ)^(-1 : ℝ))^6
  else 0

theorem constant_term_of_binomial_expansion : 
  ∃ a : ℝ, (a = ∫ (x : ℝ) in (0)..(1), 2 * x) → constant_in_binomial_expansion a = (15 : ℝ) := sorry

end constant_term_of_binomial_expansion_l2062_206220


namespace least_positive_integer_with_12_factors_is_972_l2062_206252

theorem least_positive_integer_with_12_factors_is_972 : ∃ k : ℕ, (∀ n : ℕ, (∃ (d : ℕ), d * k = n) ↔ n = 12) ∧ k = 972 := sorry

end least_positive_integer_with_12_factors_is_972_l2062_206252


namespace problem_statement_l2062_206245

def a : ℤ := 2020
def b : ℤ := 2022

theorem problem_statement : b^3 - a * b^2 - a^2 * b + a^3 = 16168 := by
  sorry

end problem_statement_l2062_206245


namespace symmetric_points_subtraction_l2062_206262

theorem symmetric_points_subtraction (a b : ℝ) (h1 : -2 = -a) (h2 : b = -3) : a - b = 5 :=
by {
  sorry
}

end symmetric_points_subtraction_l2062_206262


namespace product_of_tangents_l2062_206233

theorem product_of_tangents : 
  (Real.tan (Real.pi / 8) * Real.tan (3 * Real.pi / 8) * 
   Real.tan (5 * Real.pi / 8) * Real.tan (7 * Real.pi / 8) = -2 * Real.sqrt 2) :=
sorry

end product_of_tangents_l2062_206233


namespace not_cube_of_sum_l2062_206265

theorem not_cube_of_sum (a b : ℕ) : ¬ ∃ (k : ℤ), a^3 + b^3 + 4 = k^3 :=
by
  sorry

end not_cube_of_sum_l2062_206265


namespace quadratic_has_distinct_real_roots_l2062_206292

theorem quadratic_has_distinct_real_roots {k : ℝ} (hk : k < 0) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x₁^2 - x₁ + k = 0) ∧ (x₂^2 - x₂ + k = 0) :=
by
  -- Proof goes here.
  sorry

end quadratic_has_distinct_real_roots_l2062_206292


namespace least_prime_factor_of_expr_l2062_206236

theorem least_prime_factor_of_expr : ∀ n : ℕ, n = 11^5 - 11^2 → (∃ p : ℕ, Nat.Prime p ∧ p ≤ 2 ∧ p ∣ n) :=
by
  intros n h
  -- here will be proof steps, currently skipped
  sorry

end least_prime_factor_of_expr_l2062_206236


namespace min_arithmetic_series_sum_l2062_206229

-- Definitions from the conditions
def arithmetic_sequence (a1 : ℤ) (d : ℤ) (n : ℕ) : ℤ := a1 + (n - 1) * d
def arithmetic_series_sum (a1 : ℤ) (d : ℤ) (n : ℕ) : ℤ := n * (a1 + (n-1) * d / 2)

-- Theorem statement
theorem min_arithmetic_series_sum (a2 a7 : ℤ) (h1 : a2 = -7) (h2 : a7 = 3) :
  ∃ n, (n * (a2 + (n - 1) * 2 / 2) = (n * n) - 10 * n) ∧
  (∀ m, n* (a2 + (m - 1) * 2 / 2) ≥ n * (n * n - 10 * n)) :=
sorry

end min_arithmetic_series_sum_l2062_206229


namespace arithmetic_mean_fraction_l2062_206289

theorem arithmetic_mean_fraction :
  let a := (3 : ℚ) / 4
  let b := (5 : ℚ) / 6
  let c := (9 : ℚ) / 10
  (1 / 3) * (a + b + c) = 149 / 180 :=
by 
  sorry

end arithmetic_mean_fraction_l2062_206289


namespace quadratic_negative_roots_pq_value_l2062_206215

theorem quadratic_negative_roots_pq_value (r : ℝ) :
  (∃ p q : ℝ, p = -87 ∧ q = -23 ∧ x^2 - (r + 7)*x + r + 87 = 0 ∧ p < r ∧ r < q)
  → ((-87)^2 + (-23)^2 = 8098) :=
by
  sorry

end quadratic_negative_roots_pq_value_l2062_206215


namespace steve_pie_difference_l2062_206218

-- Definitions of conditions
def apple_pie_days : Nat := 3
def cherry_pie_days : Nat := 2
def pies_per_day : Nat := 12

-- Theorem statement
theorem steve_pie_difference : 
  (apple_pie_days * pies_per_day) - (cherry_pie_days * pies_per_day) = 12 := 
by
  sorry

end steve_pie_difference_l2062_206218


namespace grasshopper_position_after_100_jumps_l2062_206272

theorem grasshopper_position_after_100_jumps :
  let start_pos := 1
  let jumps (n : ℕ) := n
  let total_positions := 6
  let total_distance := (100 * (100 + 1)) / 2
  (start_pos + (total_distance % total_positions)) % total_positions = 5 :=
by
  sorry

end grasshopper_position_after_100_jumps_l2062_206272


namespace tan_diff_pi_over_4_l2062_206258

theorem tan_diff_pi_over_4 (α : ℝ) (hα1 : π < α) (hα2 : α < 3 / 2 * π) (hcos : Real.cos α = -4 / 5) :
  Real.tan (π / 4 - α) = 1 / 7 := by
  sorry

end tan_diff_pi_over_4_l2062_206258


namespace rita_canoe_trip_distance_l2062_206266

theorem rita_canoe_trip_distance 
  (D : ℝ)
  (h_upstream : ∃ t1, t1 = D / 3)
  (h_downstream : ∃ t2, t2 = D / 9)
  (h_total_time : ∃ t1 t2, t1 + t2 = 8) :
  D = 18 :=
by
  sorry

end rita_canoe_trip_distance_l2062_206266


namespace line_always_passes_fixed_point_l2062_206208

theorem line_always_passes_fixed_point : ∀ (m : ℝ), (m-1)*(-2) - 1 + (2*m-1) = 0 :=
by
  intro m
  -- Calculations can be done here to prove the theorem straightforwardly.
  sorry

end line_always_passes_fixed_point_l2062_206208


namespace colten_chickens_l2062_206276

variable (Colten Skylar Quentin : ℕ)

def chicken_problem_conditions :=
  (Skylar = 3 * Colten - 4) ∧
  (Quentin = 6 * Skylar + 17) ∧
  (Colten + Skylar + Quentin = 383)

theorem colten_chickens (h : chicken_problem_conditions Colten Skylar Quentin) : Colten = 37 :=
sorry

end colten_chickens_l2062_206276


namespace women_at_each_table_l2062_206299

/-- A waiter had 5 tables, each with 3 men and some women, and a total of 40 customers.
    Prove that there are 5 women at each table. -/
theorem women_at_each_table (W : ℕ) (total_customers : ℕ) (men_per_table : ℕ) (tables : ℕ)
  (h1 : total_customers = 40) (h2 : men_per_table = 3) (h3 : tables = 5) :
  (W * tables + men_per_table * tables = total_customers) → (W = 5) :=
by
  sorry

end women_at_each_table_l2062_206299


namespace no_integer_solutions_for_2891_l2062_206248

theorem no_integer_solutions_for_2891 (x y : ℤ) : ¬ (x^3 - 3 * x * y^2 + y^3 = 2891) :=
sorry

end no_integer_solutions_for_2891_l2062_206248


namespace solve_inequality_range_of_m_l2062_206290

noncomputable def f (x : ℝ) : ℝ := abs (x - 2)
noncomputable def g (x m : ℝ) : ℝ := - abs (x + 3) + m

theorem solve_inequality (x a : ℝ) :
  (f x + a - 1 > 0) ↔
  (a = 1 → x ≠ 2) ∧
  (a > 1 → true) ∧
  (a < 1 → x < a + 1 ∨ x > 3 - a) := by sorry

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, f x ≥ g x m) ↔ m < 5 := by sorry

end solve_inequality_range_of_m_l2062_206290


namespace complex_quadrant_l2062_206267

theorem complex_quadrant (z : ℂ) (h : z = (↑(1/2) : ℂ) + (↑(1/2) : ℂ) * I ) : 
  0 < z.re ∧ 0 < z.im :=
by {
sorry -- Proof goes here
}

end complex_quadrant_l2062_206267


namespace max_value_of_x2_plus_y2_l2062_206210

noncomputable def max_value (x y : ℝ) : ℝ := x^2 + y^2

theorem max_value_of_x2_plus_y2 {x y : ℝ} (h : 5*x^2 + 4*y^2 = 10*x) : max_value x y ≤ 4 := sorry

end max_value_of_x2_plus_y2_l2062_206210


namespace discriminant_eq_perfect_square_l2062_206283

variables (a b c t : ℝ)

-- Conditions
axiom a_nonzero : a ≠ 0
axiom t_root : a * t^2 + b * t + c = 0

-- Goal
theorem discriminant_eq_perfect_square :
  (b^2 - 4 * a * c) = (2 * a * t + b)^2 :=
by
  -- Conditions and goal are stated, proof to be filled.
  sorry

end discriminant_eq_perfect_square_l2062_206283


namespace table_runner_combined_area_l2062_206214

theorem table_runner_combined_area
    (table_area : ℝ) (cover_percentage : ℝ) (area_two_layers : ℝ) (area_three_layers : ℝ) (A : ℝ) :
    table_area = 175 →
    cover_percentage = 0.8 →
    area_two_layers = 24 →
    area_three_layers = 28 →
    A = (cover_percentage * table_area - area_two_layers - area_three_layers) + area_two_layers + 2 * area_three_layers →
    A = 168 :=
by
  intros h_table_area h_cover_percentage h_area_two_layers h_area_three_layers h_A
  sorry

end table_runner_combined_area_l2062_206214


namespace pizza_slices_have_both_cheese_and_bacon_l2062_206226

theorem pizza_slices_have_both_cheese_and_bacon:
  ∀ (total_slices cheese_slices bacon_slices n : ℕ),
  total_slices = 15 →
  cheese_slices = 8 →
  bacon_slices = 13 →
  (total_slices = cheese_slices + bacon_slices - n) →
  n = 6 :=
by {
  -- proof skipped
  sorry
}

end pizza_slices_have_both_cheese_and_bacon_l2062_206226


namespace pairs_equality_l2062_206231

-- Define all the pairs as given in the problem.
def pairA_1 : ℤ := - (2^7)
def pairA_2 : ℤ := (-2)^7
def pairB_1 : ℤ := - (3^2)
def pairB_2 : ℤ := (-3)^2
def pairC_1 : ℤ := -3 * (2^3)
def pairC_2 : ℤ := - (3^2) * 2
def pairD_1 : ℤ := -((-3)^2)
def pairD_2 : ℤ := -((-2)^3)

-- The problem statement.
theorem pairs_equality :
  pairA_1 = pairA_2 ∧ ¬ (pairB_1 = pairB_2) ∧ ¬ (pairC_1 = pairC_2) ∧ ¬ (pairD_1 = pairD_2) := by
  sorry

end pairs_equality_l2062_206231


namespace javier_needs_10_dozen_l2062_206253

def javier_goal : ℝ := 96
def cost_per_dozen : ℝ := 2.40
def selling_price_per_donut : ℝ := 1

theorem javier_needs_10_dozen : (javier_goal / ((selling_price_per_donut - (cost_per_dozen / 12)) * 12)) = 10 :=
by
  sorry

end javier_needs_10_dozen_l2062_206253


namespace total_students_is_45_l2062_206257

theorem total_students_is_45
  (students_burgers : ℕ) 
  (total_students : ℕ) 
  (hb : students_burgers = 30) 
  (ht : total_students = 45) : 
  total_students = 45 :=
by
  sorry

end total_students_is_45_l2062_206257


namespace remainder_addition_l2062_206200

theorem remainder_addition (k m : ℤ) (x y : ℤ) (h₁ : x = 124 * k + 13) (h₂ : y = 186 * m + 17) :
  ((x + y + 19) % 62) = 49 :=
by {
  sorry
}

end remainder_addition_l2062_206200


namespace johns_family_total_members_l2062_206251

theorem johns_family_total_members (n_f : ℕ) (h_f : n_f = 10) (n_m : ℕ) (h_m : n_m = (13 * n_f) / 10) :
  n_f + n_m = 23 := by
  rw [h_f, h_m]
  norm_num
  sorry

end johns_family_total_members_l2062_206251


namespace part_a_part_b_l2062_206294

noncomputable section

open Real

theorem part_a (x y z : ℝ) (hx : x ≠ 1) (hy : y ≠ 1) (hz : z ≠ 1) (hxyz : x * y * z = 1) :
  (x^2 / (x-1)^2) + (y^2 / (y-1)^2) + (z^2 / (z-1)^2) ≥ 1 :=
sorry

theorem part_b : ∃ (infinitely_many : ℕ → (ℚ × ℚ × ℚ)), 
  ∀ n, ((infinitely_many n).1.1 ≠ 1) ∧ ((infinitely_many n).1.2 ≠ 1) ∧ ((infinitely_many n).2 ≠ 1) ∧ 
  ((infinitely_many n).1.1 * (infinitely_many n).1.2 * (infinitely_many n).2 = 1) ∧ 
  ((infinitely_many n).1.1^2 / ((infinitely_many n).1.1 - 1)^2 + 
   (infinitely_many n).1.2^2 / ((infinitely_many n).1.2 - 1)^2 + 
   (infinitely_many n).2^2 / ((infinitely_many n).2 - 1)^2 = 1) :=
sorry

end part_a_part_b_l2062_206294


namespace econ_not_feasible_l2062_206278

theorem econ_not_feasible (x y p q: ℕ) (h_xy : 26 * x + 29 * y = 687) (h_pq : 27 * p + 31 * q = 687) : p + q ≥ x + y := by
  sorry

end econ_not_feasible_l2062_206278


namespace arithmetic_sequence_150th_term_l2062_206255

open Nat

-- Define the nth term of an arithmetic sequence
def nth_term_arithmetic (a1 d n : ℕ) : ℕ :=
  a1 + (n - 1) * d

-- Theorem to prove
theorem arithmetic_sequence_150th_term (a1 d n : ℕ) (h1 : a1 = 3) (h2 : d = 7) (h3 : n = 150) :
  nth_term_arithmetic a1 d n = 1046 :=
by
  sorry

end arithmetic_sequence_150th_term_l2062_206255


namespace total_pizza_pieces_l2062_206234

-- Definitions based on the conditions
def pieces_per_pizza : Nat := 6
def pizzas_per_student : Nat := 20
def number_of_students : Nat := 10

-- Statement of the theorem
theorem total_pizza_pieces :
  pieces_per_pizza * pizzas_per_student * number_of_students = 1200 :=
by
  -- Placeholder for the proof
  sorry

end total_pizza_pieces_l2062_206234


namespace basketball_scores_l2062_206263

theorem basketball_scores (n : ℕ) (h : n = 7) : 
  ∃ (k : ℕ), k = 8 :=
by {
  sorry
}

end basketball_scores_l2062_206263


namespace parabola_directrix_eq_l2062_206225

def parabola_directrix (p : ℝ) : ℝ := -p

theorem parabola_directrix_eq (x y p : ℝ) (h : y ^ 2 = 8 * x) (hp : 2 * p = 8) : 
  parabola_directrix p = -2 :=
by
  sorry

end parabola_directrix_eq_l2062_206225


namespace fraction_of_males_on_time_l2062_206261

theorem fraction_of_males_on_time (A : ℕ) :
  (2 / 9 : ℚ) * A = (2 / 9 : ℚ) * A → 
  (2 / 3 : ℚ) * A = (2 / 3 : ℚ) * A → 
  (5 / 6 : ℚ) * ((1 / 3 : ℚ) * A) = (5 / 6 : ℚ) * ((1 / 3 : ℚ) * A) → 
  ((7 / 9 : ℚ) * A - (5 / 18 : ℚ) * A) / ((2 / 3 : ℚ) * A) = (1 / 2 : ℚ) :=
by
  intros h1 h2 h3
  sorry

end fraction_of_males_on_time_l2062_206261


namespace ratio_of_m_l2062_206224

theorem ratio_of_m (a b m m1 m2 : ℝ)
  (h1 : a * m^2 + b * m + c = 0)
  (h2 : (a / b + b / a) = 3 / 7)
  (h3 : a + b = (3 * m - 2) / m)
  (h4 : a * b = 7 / m)
  (h5 : (a + b)^2 = ab / (m * (7/ m)) - 2) :
  (m1 + m2 = 21) ∧ (m1 * m2 = 4) → 
  (m1/m2 + m2/m1 = 108.25) := sorry

end ratio_of_m_l2062_206224


namespace full_tank_cost_l2062_206219

-- Definitions from the conditions
def total_liters_given := 36
def total_cost_given := 18
def tank_capacity := 64

-- Hypothesis based on the conditions
def price_per_liter := total_cost_given / total_liters_given

-- Conclusion we need to prove
theorem full_tank_cost: price_per_liter * tank_capacity = 32 :=
  sorry

end full_tank_cost_l2062_206219


namespace parkway_school_students_l2062_206291

theorem parkway_school_students (total_boys total_soccer soccer_boys_percentage girls_not_playing_soccer : ℕ)
  (h1 : total_boys = 320)
  (h2 : total_soccer = 250)
  (h3 : soccer_boys_percentage = 86)
  (h4 : girls_not_playing_soccer = 95)
  (h5 : total_soccer * soccer_boys_percentage / 100 = 215) :
  total_boys + total_soccer - (total_soccer * soccer_boys_percentage / 100) + girls_not_playing_soccer = 450 :=
by
  sorry

end parkway_school_students_l2062_206291


namespace total_grandchildren_l2062_206238

-- Define the conditions 
def daughters := 5
def sons := 4
def children_per_daughter := 8 + 7
def children_per_son := 6 + 3

-- State the proof problem
theorem total_grandchildren : daughters * children_per_daughter + sons * children_per_son = 111 :=
by
  sorry

end total_grandchildren_l2062_206238


namespace minimum_value_S15_minus_S10_l2062_206273

theorem minimum_value_S15_minus_S10 (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h_geom_seq : ∀ n, S (n + 1) = S n * a (n + 1))
  (h_pos_terms : ∀ n, a n > 0)
  (h_arith_seq : S 10 - 2 * S 5 = 3)
  (h_geom_sub_seq : (S 10 - S 5) * (S 10 - S 5) = S 5 * (S 15 - S 10)) :
  ∃ m, m = 12 ∧ (S 15 - S 10) ≥ m := sorry

end minimum_value_S15_minus_S10_l2062_206273


namespace laborer_savings_l2062_206274

theorem laborer_savings
  (monthly_expenditure_first6 : ℕ := 70)
  (monthly_expenditure_next4 : ℕ := 60)
  (monthly_income : ℕ := 69)
  (expenditure_first6 := 6 * monthly_expenditure_first6)
  (income_first6 := 6 * monthly_income)
  (debt : ℕ := expenditure_first6 - income_first6)
  (expenditure_next4 := 4 * monthly_expenditure_next4)
  (income_next4 := 4 * monthly_income)
  (savings : ℕ := income_next4 - (expenditure_next4 + debt)) :
  savings = 30 := 
by
  sorry

end laborer_savings_l2062_206274


namespace neg_proposition_l2062_206207

theorem neg_proposition :
  (¬(∀ x : ℕ, x^3 > x^2)) ↔ (∃ x : ℕ, x^3 ≤ x^2) := 
sorry

end neg_proposition_l2062_206207


namespace pipe_filling_problem_l2062_206280

theorem pipe_filling_problem (x : ℝ) (h : (2 / 15) * x + (1 / 20) * (10 - x) = 1) : x = 6 :=
sorry

end pipe_filling_problem_l2062_206280


namespace percent_increase_first_quarter_l2062_206281

theorem percent_increase_first_quarter (P : ℝ) (X : ℝ) (h1 : P > 0) 
  (end_of_second_quarter : P * 1.8 = P*(1 + X / 100) * 1.44) : 
  X = 25 :=
by
  sorry

end percent_increase_first_quarter_l2062_206281


namespace evaluate_expression_l2062_206269

theorem evaluate_expression (a : ℚ) (h : a = 4 / 3) : 
  (6 * a ^ 2 - 15 * a + 5) * (3 * a - 4) = 0 := by
  sorry

end evaluate_expression_l2062_206269


namespace nonagon_isosceles_triangle_count_l2062_206270

theorem nonagon_isosceles_triangle_count (N : ℕ) (hN : N = 9) : 
  ∃(k : ℕ), k = 30 := 
by 
  have h := hN
  sorry      -- Solution steps would go here if we were proving it

end nonagon_isosceles_triangle_count_l2062_206270


namespace calculate_discount_l2062_206213

theorem calculate_discount
  (original_cost : ℝ)
  (amount_spent : ℝ)
  (h1 : original_cost = 35.00)
  (h2 : amount_spent = 18.00) :
  original_cost - amount_spent = 17.00 :=
by
  sorry

end calculate_discount_l2062_206213


namespace walking_west_negation_l2062_206277

theorem walking_west_negation (distance_east distance_west : Int) (h_east : distance_east = 6) (h_west : distance_west = -10) : 
    (10 : Int) = - distance_west := by
  sorry

end walking_west_negation_l2062_206277


namespace ezekiel_third_day_hike_l2062_206203

-- Ezekiel's total hike distance
def total_distance : ℕ := 50

-- Distance covered on the first day
def first_day_distance : ℕ := 10

-- Distance covered on the second day
def second_day_distance : ℕ := total_distance / 2

-- Distance remaining for the third day
def third_day_distance : ℕ := total_distance - first_day_distance - second_day_distance

-- The distance Ezekiel had to hike on the third day
theorem ezekiel_third_day_hike : third_day_distance = 15 := by
  sorry

end ezekiel_third_day_hike_l2062_206203


namespace find_q_zero_l2062_206268

-- Assuming the polynomials p, q, and r are defined, and their relevant conditions are satisfied.

def constant_term (f : ℕ → ℝ) : ℝ := f 0

theorem find_q_zero (p q r : ℕ → ℝ)
  (h : p * q = r)
  (h_p_const : constant_term p = 5)
  (h_r_const : constant_term r = -10) :
  q 0 = -2 :=
sorry

end find_q_zero_l2062_206268


namespace closest_to_sin_2016_deg_is_neg_half_l2062_206244

/-- Given the value of \( \sin 2016^\circ \), show that the closest number from the given options is \( -\frac{1}{2} \).
Options:
A: \( \frac{11}{2} \)
B: \( -\frac{1}{2} \)
C: \( \frac{\sqrt{2}}{2} \)
D: \( -1 \)
-/
theorem closest_to_sin_2016_deg_is_neg_half :
  let sin_2016 := Real.sin (2016 * Real.pi / 180)
  |sin_2016 - (-1 / 2)| < |sin_2016 - 11 / 2| ∧
  |sin_2016 - (-1 / 2)| < |sin_2016 - Real.sqrt 2 / 2| ∧
  |sin_2016 - (-1 / 2)| < |sin_2016 - (-1)| :=
by
  sorry

end closest_to_sin_2016_deg_is_neg_half_l2062_206244


namespace volume_of_Q_3_l2062_206293

noncomputable def Q (n : ℕ) : ℚ :=
  match n with
  | 0 => 1
  | 1 => 2       -- 1 + 1
  | 2 => 2 + 3 / 16
  | 3 => (2 + 3 / 16) + 3 / 64
  | _ => sorry -- This handles cases n >= 4, which we don't need.

theorem volume_of_Q_3 : Q 3 = 143 / 64 := by
  sorry

end volume_of_Q_3_l2062_206293


namespace coating_profit_l2062_206286

theorem coating_profit (x y : ℝ) (h1 : 0.6 * x + 0.9 * (150 - x) ≤ 120)
  (h2 : 0.7 * x + 0.4 * (150 - x) ≤ 90) :
  (50 ≤ x ∧ x ≤ 100) → (y = -50 * x + 75000) → (x = 50 → y = 72500) :=
by
  intros hx hy hx_val
  sorry

end coating_profit_l2062_206286


namespace geometric_sequence_a7_value_l2062_206211

noncomputable def geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, a (n + 1) = r * a n

theorem geometric_sequence_a7_value (a : ℕ → ℝ) (r : ℝ) (S : ℕ → ℝ) :
  (∀ n, 0 < a n) →
  (geometric_sequence a r) →
  (S 4 = 3 * S 2) →
  (a 3 = 2) →
  (S n = a 1 + a 1 * r + a 1 * r^2 + a 1 * r^3) →
  a 7 = 8 :=
by
  sorry

end geometric_sequence_a7_value_l2062_206211


namespace find_value_of_ratio_l2062_206275

theorem find_value_of_ratio (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x / y + y / x = 4) :
  (x + 2 * y) / (x - 2 * y) = Real.sqrt 33 / 3 := 
  sorry

end find_value_of_ratio_l2062_206275


namespace avg_weight_of_a_b_c_l2062_206284

theorem avg_weight_of_a_b_c (A B C : ℝ) (h1 : (A + B) / 2 = 40) (h2 : (B + C) / 2 = 43) (h3 : B = 31) :
  (A + B + C) / 3 = 45 :=
by
  sorry

end avg_weight_of_a_b_c_l2062_206284


namespace problem_l2062_206235

theorem problem (a b c k : ℝ) (h : a ≠ b ∧ b ≠ c ∧ c ≠ a) (hk : k ≠ 0)
  (h1 : a / (b - c) + b / (c - a) + c / (a - b) = 0) :
  a / (k * (b - c)^2) + b / (k * (c - a)^2) + c / (k * (a - b)^2) = 0 :=
by
  sorry

end problem_l2062_206235


namespace area_of_rectangle_l2062_206297

theorem area_of_rectangle (side_small_squares : ℝ) (side_smaller_square : ℝ) (side_larger_square : ℝ) 
  (h_small_squares : side_small_squares ^ 2 = 4) 
  (h_smaller_square : side_smaller_square ^ 2 = 1) 
  (h_larger_square : side_larger_square = 2 * side_smaller_square) :
  let horizontal_length := 2 * side_small_squares
  let vertical_length := side_small_squares + side_smaller_square
  let area := horizontal_length * vertical_length
  area = 12 
:= by 
  sorry

end area_of_rectangle_l2062_206297


namespace chess_tournament_games_l2062_206260

def num_games (n : Nat) : Nat := n * (n - 1) * 2

theorem chess_tournament_games : num_games 7 = 84 :=
by
  sorry

end chess_tournament_games_l2062_206260


namespace rebecca_less_than_toby_l2062_206241

-- Define the conditions
variable (x : ℕ) -- Thomas worked x hours
variable (tobyHours : ℕ := 2 * x - 10) -- Toby worked 10 hours less than twice what Thomas worked
variable (rebeccaHours : ℕ := 56) -- Rebecca worked 56 hours

-- Define the total hours worked in one week
axiom total_hours_worked : x + tobyHours + rebeccaHours = 157

-- The proof goal
theorem rebecca_less_than_toby : tobyHours - rebeccaHours = 8 := 
by
  -- (proof steps would go here)
  sorry

end rebecca_less_than_toby_l2062_206241


namespace factorization_exists_l2062_206298

-- Define the polynomial f(x)
def f (x : ℚ) : ℚ := x^4 + x^3 + x^2 + x + 12

-- Definition for polynomial g(x)
def g (a : ℤ) (x : ℚ) : ℚ := x^2 + a*x + 3

-- Definition for polynomial h(x)
def h (b : ℤ) (x : ℚ) : ℚ := x^2 + b*x + 4

-- The main statement to prove
theorem factorization_exists :
  ∃ (a b : ℤ), (∀ x, f x = (g a x) * (h b x)) :=
by
  sorry

end factorization_exists_l2062_206298


namespace expectation_equality_variance_inequality_l2062_206285

noncomputable def X1_expectation : ℚ :=
  2 * (2 / 5 : ℚ)

noncomputable def X1_variance : ℚ :=
  2 * (2 / 5) * (1 - 2 / 5)

noncomputable def P_X2_0 : ℚ :=
  (3 * 2) / (5 * 4)

noncomputable def P_X2_1 : ℚ :=
  (2 * 3) / (5 * 4)

noncomputable def P_X2_2 : ℚ :=
  (2 * 1) / (5 * 4)

noncomputable def X2_expectation : ℚ :=
  0 * P_X2_0 + 1 * P_X2_1 + 2 * P_X2_2

noncomputable def X2_variance : ℚ :=
  P_X2_0 * (0 - X2_expectation)^2 + P_X2_1 * (1 - X2_expectation)^2 + P_X2_2 * (2 - X2_expectation)^2

theorem expectation_equality : X1_expectation = X2_expectation :=
  by sorry

theorem variance_inequality : X1_variance > X2_variance :=
  by sorry

end expectation_equality_variance_inequality_l2062_206285


namespace dividend_calculation_l2062_206206

theorem dividend_calculation :
  ∀ (divisor quotient remainder : ℝ), 
  divisor = 37.2 → 
  quotient = 14.61 → 
  remainder = 0.67 → 
  (divisor * quotient + remainder) = 544.042 :=
by
  intros divisor quotient remainder h_div h_qt h_rm
  sorry

end dividend_calculation_l2062_206206


namespace sin_double_angle_identity_l2062_206247

theorem sin_double_angle_identity (α : ℝ) (h : Real.cos α = 1 / 4) : 
  Real.sin (π / 2 - 2 * α) = -7 / 8 :=
by 
  sorry

end sin_double_angle_identity_l2062_206247


namespace find_b_l2062_206204

theorem find_b (a b : ℕ) (h1 : a = 105) (h2 : a ^ 3 = 21 * 25 * 15 * b) : b = 147 :=
sorry

end find_b_l2062_206204


namespace last_two_digits_of_1976_pow_100_l2062_206288

theorem last_two_digits_of_1976_pow_100 :
  (1976 ^ 100) % 100 = 76 :=
by
  sorry

end last_two_digits_of_1976_pow_100_l2062_206288


namespace water_level_drop_l2062_206264

theorem water_level_drop :
  (∀ x : ℝ, x > 0 → (x = 4) → (x > 0 → x = 4)) →
  ∃ y : ℝ, y < 0 ∧ (y = -1) :=
by
  sorry

end water_level_drop_l2062_206264


namespace find_a_b_l2062_206202

theorem find_a_b (a b : ℤ) (h: 4 * a^2 + 3 * b^2 + 10 * a * b = 144) :
    (a = 2 ∧ b = 4) :=
by {
  sorry
}

end find_a_b_l2062_206202


namespace math_problem_l2062_206237

theorem math_problem : 12 - (- 18) + (- 7) - 15 = 8 :=
by
  sorry

end math_problem_l2062_206237


namespace largest_constant_C_l2062_206279

theorem largest_constant_C (C : ℝ) : C = 2 / Real.sqrt 3 ↔ ∀ (x y z : ℝ), x^2 + y^2 + 2 * z^2 + 1 ≥ C * (x + y + z) :=
by
  sorry

end largest_constant_C_l2062_206279


namespace quadratic_real_roots_l2062_206232

theorem quadratic_real_roots (k : ℝ) : 
  (∀ x : ℝ, (2 * x^2 + 4 * x + k - 1 = 0) → ∃ x : ℝ, 2 * x^2 + 4 * x + k - 1 = 0) → 
  k ≤ 3 :=
by
  intro h
  have h_discriminant : 16 - 8 * k >= 0 := sorry
  linarith

end quadratic_real_roots_l2062_206232


namespace sticks_form_triangle_l2062_206243

theorem sticks_form_triangle (a b c d e : ℝ) 
  (h1 : 2 < a) (h2 : a < 8)
  (h3 : 2 < b) (h4 : b < 8)
  (h5 : 2 < c) (h6 : c < 8)
  (h7 : 2 < d) (h8 : d < 8)
  (h9 : 2 < e) (h10 : e < 8) :
  ∃ x y z, 
    (x = a ∨ x = b ∨ x = c ∨ x = d ∨ x = e) ∧
    (y = a ∨ y = b ∨ y = c ∨ y = d ∨ y = e) ∧
    (z = a ∨ z = b ∨ z = c ∨ z = d ∨ z = e) ∧
    x ≠ y ∧ x ≠ z ∧ y ≠ z ∧
    x + y > z ∧ x + z > y ∧ y + z > x :=
by sorry

end sticks_form_triangle_l2062_206243


namespace sum_of_a_and_b_is_24_l2062_206287

theorem sum_of_a_and_b_is_24 
  (a b : ℕ) 
  (h_a_pos : a > 0) 
  (h_b_gt_one : b > 1) 
  (h_maximal : ∀ (a' b' : ℕ), (a' > 0) → (b' > 1) → (a'^b' < 500) → (a'^b' ≤ a^b)) :
  a + b = 24 := 
sorry

end sum_of_a_and_b_is_24_l2062_206287


namespace problem_3000_mod_1001_l2062_206221

theorem problem_3000_mod_1001 : (300 ^ 3000 - 1) % 1001 = 0 := 
by
  have h1: (300 ^ 3000) % 7 = 1 := sorry
  have h2: (300 ^ 3000) % 11 = 1 := sorry
  have h3: (300 ^ 3000) % 13 = 1 := sorry
  sorry

end problem_3000_mod_1001_l2062_206221


namespace area_of_square_l2062_206212

theorem area_of_square (r : ℝ) (b : ℝ) (ℓ : ℝ) (area_rect : ℝ) 
    (h₁ : ℓ = 2 / 3 * r) 
    (h₂ : r = b) 
    (h₃ : b = 13) 
    (h₄ : area_rect = 598) 
    (h₅ : area_rect = ℓ * b) : 
    r^2 = 4761 := 
sorry

end area_of_square_l2062_206212


namespace win_sector_area_l2062_206259

theorem win_sector_area (r : ℝ) (P : ℝ) (h0 : r = 8) (h1 : P = 3 / 8) :
    let area_total := Real.pi * r ^ 2
    let area_win := P * area_total
    area_win = 24 * Real.pi :=
by 
  sorry

end win_sector_area_l2062_206259


namespace tangent_vertical_y_axis_iff_a_gt_0_l2062_206223

theorem tangent_vertical_y_axis_iff_a_gt_0 {a : ℝ} (f : ℝ → ℝ) 
    (hf : ∀ x > 0, f x = a * x^2 - Real.log x)
    (h_tangent_vertical : ∃ x > 0, (deriv f x) = 0) :
    a > 0 := 
sorry

end tangent_vertical_y_axis_iff_a_gt_0_l2062_206223


namespace candy_distribution_impossible_l2062_206296

theorem candy_distribution_impossible :
  ∀ (candies : Fin 6 → ℕ),
  (candies 0 = 0 ∧ candies 1 = 1 ∧ candies 2 = 0 ∧ candies 3 = 0 ∧ candies 4 = 0 ∧ candies 5 = 1) →
  (∀ t, ∃ i, (i < 6) ∧ candies ((i+t)%6) = candies ((i+t+1)%6)) →
  ∃ (i : Fin 6), candies i ≠ candies ((i + 1) % 6) :=
by
  sorry

end candy_distribution_impossible_l2062_206296


namespace sin_alpha_l2062_206256

noncomputable def f (x : ℝ) : ℝ := Real.cos (x / 2 - Real.pi / 4)

theorem sin_alpha (α : ℝ) (h : f α = 1 / 3) : Real.sin α = -7 / 9 :=
by 
  sorry

end sin_alpha_l2062_206256


namespace geometric_sequence_terms_l2062_206228

theorem geometric_sequence_terms
  (a_3 : ℝ) (a_4 : ℝ)
  (h1 : a_3 = 12)
  (h2 : a_4 = 18) :
  ∃ (a_1 a_2 : ℝ) (q: ℝ), 
    a_1 = 16 / 3 ∧ a_2 = 8 ∧ a_3 = a_1 * q^2 ∧ a_4 = a_1 * q^3 := 
by
  sorry

end geometric_sequence_terms_l2062_206228


namespace triangle_angle_sum_l2062_206209

open scoped Real

theorem triangle_angle_sum (A B C : ℝ) 
  (hA : A = 25) (hB : B = 55) : C = 100 :=
by
  have h1 : A + B + C = 180 := sorry
  rw [hA, hB] at h1
  linarith

end triangle_angle_sum_l2062_206209


namespace major_axis_length_l2062_206295

def length_of_major_axis 
  (tangent_x : ℝ) (f1 : ℝ × ℝ) (f2 : ℝ × ℝ) : ℝ :=
  sorry

theorem major_axis_length 
  (hx_tangent : (4, 0) = (4, 0)) 
  (foci : (4, 2 + 2 * Real.sqrt 2) = (4, 2 + 2 * Real.sqrt 2) ∧ 
         (4, 2 - 2 * Real.sqrt 2) = (4, 2 - 2 * Real.sqrt 2)) :
  length_of_major_axis 4 
  (4, 2 + 2 * Real.sqrt 2) (4, 2 - 2 * Real.sqrt 2) = 4 :=
sorry

end major_axis_length_l2062_206295


namespace total_newspapers_collected_l2062_206205

-- Definitions based on the conditions
def Chris_collected : ℕ := 42
def Lily_collected : ℕ := 23

-- The proof statement
theorem total_newspapers_collected :
  Chris_collected + Lily_collected = 65 := by
  sorry

end total_newspapers_collected_l2062_206205


namespace simplify_expression_l2062_206217

def expression (x y : ℤ) : ℤ := 
  ((2 * x + y) * (2 * x - y) - (2 * x - 3 * y)^2) / (-2 * y)

theorem simplify_expression {x y : ℤ} (hx : x = 1) (hy : y = -2) :
  expression x y = -16 :=
by 
  -- This proof will involve algebraic manipulation and substitution.
  sorry

end simplify_expression_l2062_206217


namespace range_of_sum_l2062_206242

theorem range_of_sum (a b c d : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) (h₄ : d > 0) :
  1 < (a / (a + b + d) + b / (a + b + c) + c / (b + c + d) + d / (a + c + d)) ∧
  (a / (a + b + d) + b / (a + b + c) + c / (b + c + d) + d / (a + c + d)) < 2 :=
sorry

end range_of_sum_l2062_206242


namespace value_of_composed_operations_l2062_206249

def op1 (x : ℝ) : ℝ := 9 - x
def op2 (x : ℝ) : ℝ := x - 9

theorem value_of_composed_operations : op2 (op1 15) = -15 :=
by
  sorry

end value_of_composed_operations_l2062_206249


namespace line_through_point_trangle_area_line_with_given_slope_l2062_206230

theorem line_through_point_trangle_area (k : ℝ) (b : ℝ) : 
  (∃ k, (∀ x y, y = k * (x + 3) + 4 ∧ (1 / 2) * (abs (3 * k + 4) * abs (-4 / k - 3)) = 3)) → 
  (∃ k₁ k₂, k₁ = -2/3 ∧ k₂ = -8/3 ∧ 
    (∀ x y, y = k₁ * (x + 3) + 4 → 2 * x + 3 * y - 6 = 0) ∧ 
    (∀ x y, y = k₂ * (x + 3) + 4 → 8 * x + 3 * y + 12 = 0)) := 
sorry

theorem line_with_given_slope (b : ℝ) : 
  (∀ x y, y = (1 / 6) * x + b) → (1 / 2) * abs (6 * b * b) = 3 → 
  (b = 1 ∨ b = -1) → (∀ x y, (b = 1 → x - 6 * y + 6 = 0 ∧ b = -1 → x - 6 * y - 6 = 0)) := 
sorry

end line_through_point_trangle_area_line_with_given_slope_l2062_206230


namespace words_per_page_l2062_206282

theorem words_per_page (p : ℕ) (h1 : 150 * p ≡ 210 [MOD 221]) (h2 : p ≤ 90) : p = 90 :=
sorry

end words_per_page_l2062_206282


namespace nonnegative_difference_of_roots_l2062_206271

theorem nonnegative_difference_of_roots :
  ∀ (x : ℝ), x^2 + 40 * x + 300 = -50 → (∃ a b : ℝ, x^2 + 40 * x + 350 = 0 ∧ x = a ∧ x = b ∧ |a - b| = 25) := 
by 
sorry

end nonnegative_difference_of_roots_l2062_206271


namespace coordinates_of_C_are_correct_l2062_206240

noncomputable section 

def Point := (ℝ × ℝ)

def A : Point := (1, 3)
def B : Point := (13, 9)

def vector_AB (A B : Point) : Point :=
  (B.1 - A.1, B.2 - A.2)

def scalar_mult (s : ℝ) (v : Point) : Point :=
  (s * v.1, s * v.2)

def add_vectors (v1 v2 : Point) : Point :=
  (v1.1 + v2.1, v1.2 + v2.2)

def C : Point :=
  let AB := vector_AB A B
  add_vectors B (scalar_mult (1 / 2) AB)

theorem coordinates_of_C_are_correct : C = (19, 12) := by sorry

end coordinates_of_C_are_correct_l2062_206240


namespace binary_arithmetic_l2062_206239

theorem binary_arithmetic :
  let a := 0b1101
  let b := 0b0110
  let c := 0b1011
  let d := 0b1001
  a + b - c + d = 0b10001 := by
sorry

end binary_arithmetic_l2062_206239


namespace batsman_percentage_running_between_wickets_l2062_206222

def boundaries : Nat := 6
def runs_per_boundary : Nat := 4
def sixes : Nat := 4
def runs_per_six : Nat := 6
def no_balls : Nat := 8
def runs_per_no_ball : Nat := 1
def wide_balls : Nat := 5
def runs_per_wide_ball : Nat := 1
def leg_byes : Nat := 2
def runs_per_leg_bye : Nat := 1
def total_score : Nat := 150

def runs_from_boundaries : Nat := boundaries * runs_per_boundary
def runs_from_sixes : Nat := sixes * runs_per_six
def runs_not_off_bat : Nat := no_balls * runs_per_no_ball + wide_balls * runs_per_wide_ball + leg_byes * runs_per_leg_bye

def runs_running_between_wickets : Nat := total_score - runs_not_off_bat - runs_from_boundaries - runs_from_sixes

def percentage_runs_running_between_wickets : Float := 
  (runs_running_between_wickets.toFloat / total_score.toFloat) * 100

theorem batsman_percentage_running_between_wickets : percentage_runs_running_between_wickets = 58 := sorry

end batsman_percentage_running_between_wickets_l2062_206222


namespace make_tea_time_efficiently_l2062_206246

theorem make_tea_time_efficiently (minutes_kettle minutes_boil minutes_teapot minutes_teacups minutes_tea_leaves total_estimate total_time : ℕ)
  (h1 : minutes_kettle = 1)
  (h2 : minutes_boil = 15)
  (h3 : minutes_teapot = 1)
  (h4 : minutes_teacups = 1)
  (h5 : minutes_tea_leaves = 2)
  (h6 : total_estimate = 20)
  (h_total_time : total_time = minutes_kettle + minutes_boil) :
  total_time = 16 :=
by
  sorry

end make_tea_time_efficiently_l2062_206246


namespace loop_execution_count_l2062_206250

theorem loop_execution_count : 
  ∀ (a b : ℤ), a = 2 → b = 20 → (b - a + 1) = 19 :=
by
  intros a b ha hb
  rw [ha, hb]
  -- Here, we explicitly compute (20 - 2 + 1) = 19
  exact rfl

end loop_execution_count_l2062_206250


namespace inequation_proof_l2062_206201

theorem inequation_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a^2 + b^2 + c^2 = 1) :
  (a / (1 - a^2)) + (b / (1 - b^2)) + (c / (1 - c^2)) ≥ (3 * Real.sqrt 3 / 2) :=
by
  sorry

end inequation_proof_l2062_206201


namespace average_weight_of_all_children_l2062_206227

theorem average_weight_of_all_children 
    (boys_weight_avg : ℕ)
    (number_of_boys : ℕ)
    (girls_weight_avg : ℕ)
    (number_of_girls : ℕ)
    (tall_boy_weight : ℕ)
    (ht1 : boys_weight_avg = 155)
    (ht2 : number_of_boys = 8)
    (ht3 : girls_weight_avg = 130)
    (ht4 : number_of_girls = 6)
    (ht5 : tall_boy_weight = 175)
    : (boys_weight_avg * (number_of_boys - 1) + tall_boy_weight + girls_weight_avg * number_of_girls) / (number_of_boys + number_of_girls) = 146 :=
by
  sorry

end average_weight_of_all_children_l2062_206227


namespace r_amount_l2062_206216

-- Let p, q, and r be the amounts of money p, q, and r have, respectively
variables (p q r : ℝ)

-- Given conditions: p + q + r = 5000 and r = (2 / 3) * (p + q)
theorem r_amount (h1 : p + q + r = 5000) (h2 : r = (2 / 3) * (p + q)) :
  r = 2000 :=
sorry

end r_amount_l2062_206216

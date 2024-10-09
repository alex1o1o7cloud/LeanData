import Mathlib

namespace y1_lt_y2_l447_44758

theorem y1_lt_y2 (x1 x2 : ℝ) (h1 : x1 < 0) (h2 : 0 < x2) :
  (6 / x1) < (6 / x2) :=
by
  sorry

end y1_lt_y2_l447_44758


namespace point_transformation_l447_44750

theorem point_transformation (a b : ℝ) :
  let P := (a, b)
  let P₁ := (2 * 2 - a, 2 * 3 - b) -- Rotate P 180° counterclockwise around (2, 3)
  let P₂ := (P₁.2, P₁.1)           -- Reflect P₁ about the line y = x
  P₂ = (5, -4) → a - b = 7 :=
by
  intros
  sorry

end point_transformation_l447_44750


namespace profit_calculation_more_profitable_method_l447_44771

def profit_end_of_month (x : ℝ) : ℝ :=
  0.3 * x - 900

def profit_beginning_of_month (x : ℝ) : ℝ :=
  0.26 * x

theorem profit_calculation (x : ℝ) (h₁ : profit_end_of_month x = 0.3 * x - 900)
  (h₂ : profit_beginning_of_month x = 0.26 * x) :
  profit_end_of_month x = 0.3 * x - 900 ∧ profit_beginning_of_month x = 0.26 * x :=
by 
  sorry

theorem more_profitable_method (x : ℝ) (hx : x = 20000)
  (h_beg : profit_beginning_of_month x = 0.26 * x)
  (h_end : profit_end_of_month x = 0.3 * x - 900) :
  profit_beginning_of_month x > profit_end_of_month x ∧ profit_beginning_of_month x = 5200 :=
by 
  sorry

end profit_calculation_more_profitable_method_l447_44771


namespace episodes_per_season_before_loss_l447_44728

-- Define the given conditions
def initial_total_seasons : ℕ := 12 + 14
def episodes_lost_per_season : ℕ := 2
def remaining_episodes : ℕ := 364
def total_episodes_lost : ℕ := 12 * episodes_lost_per_season + 14 * episodes_lost_per_season
def initial_total_episodes : ℕ := remaining_episodes + total_episodes_lost

-- Define the theorem to prove
theorem episodes_per_season_before_loss : initial_total_episodes / initial_total_seasons = 16 :=
by
  sorry

end episodes_per_season_before_loss_l447_44728


namespace simplify_expr_l447_44785

-- Define the expression
def expr := |-4^2 + 7|

-- State the theorem
theorem simplify_expr : expr = 9 :=
by sorry

end simplify_expr_l447_44785


namespace largest_root_range_l447_44783

theorem largest_root_range (b_0 b_1 b_2 b_3 : ℝ)
  (hb_0 : |b_0| ≤ 3) (hb_1 : |b_1| ≤ 3) (hb_2 : |b_2| ≤ 3) (hb_3 : |b_3| ≤ 3) :
  ∃ s : ℝ, (∃ x : ℝ, x ^ 4 + b_3 * x ^ 3 + b_2 * x ^ 2 + b_1 * x + b_0 = 0 ∧ x > 0 ∧ s = x) ∧ 3 < s ∧ s < 4 := 
sorry

end largest_root_range_l447_44783


namespace students_taking_neither_l447_44726

def total_students : ℕ := 1200
def music_students : ℕ := 60
def art_students : ℕ := 80
def sports_students : ℕ := 30
def music_and_art_students : ℕ := 25
def music_and_sports_students : ℕ := 15
def art_and_sports_students : ℕ := 20
def all_three_students : ℕ := 10

theorem students_taking_neither :
  total_students - (music_students + art_students + sports_students 
  - music_and_art_students - music_and_sports_students - art_and_sports_students 
  + all_three_students) = 1080 := sorry

end students_taking_neither_l447_44726


namespace prime_factor_of_T_l447_44738

-- Define constants and conditions
def x : ℕ := 2021
def T : ℕ := Nat.sqrt ((x + x) + (x - x) + (x * x) + (x / x))

-- Define what needs to be proved
theorem prime_factor_of_T : ∃ p : ℕ, Nat.Prime p ∧ Nat.factorization T p > 0 ∧ (∀ q : ℕ, Nat.Prime q ∧ Nat.factorization T q > 0 → q ≤ p) :=
sorry

end prime_factor_of_T_l447_44738


namespace inequality_solution_l447_44756

theorem inequality_solution (x : ℝ) : 
  (7 - 2 * (x + 1) ≥ 1 - 6 * x) ∧ ((1 + 2 * x) / 3 > x - 1) ↔ (-1 ≤ x ∧ x < 4) := 
by
  sorry

end inequality_solution_l447_44756


namespace hilton_final_marbles_l447_44766

def initial_marbles : ℕ := 26
def marbles_found : ℕ := 6
def marbles_lost : ℕ := 10
def marbles_from_lori := 2 * marbles_lost

def final_marbles := initial_marbles + marbles_found - marbles_lost + marbles_from_lori

theorem hilton_final_marbles : final_marbles = 42 := sorry

end hilton_final_marbles_l447_44766


namespace first_proof_l447_44740

def triangular (n : ℕ) : ℕ :=
  (n * (n + 1)) / 2

def covers_all_columns (k : ℕ) : Prop :=
  ∀ c : ℕ, (c < 10) → (∃ m : ℕ, m ≤ k ∧ (triangular m) % 10 = c)

theorem first_proof (k : ℕ) (h : covers_all_columns 28) : 
  triangular k = 435 :=
sorry

end first_proof_l447_44740


namespace simplify_expression_l447_44779

variable (a b : ℝ) (hab_pos : 0 < a ∧ 0 < b)
variable (h : a^3 - b^3 = a - b)

theorem simplify_expression 
  (a b : ℝ) (hab_pos : 0 < a ∧ 0 < b) (h : a^3 - b^3 = a - b) : 
  (a / b - b / a + 1 / (a * b)) = 2 * (1 / (a * b)) - 1 := 
sorry

end simplify_expression_l447_44779


namespace minimize_payment_l447_44768

theorem minimize_payment :
  ∀ (bd_A td_A bd_B td_B bd_C td_C : ℕ),
    bd_A = 42 → td_A = 36 →
    bd_B = 48 → td_B = 41 →
    bd_C = 54 → td_C = 47 →
    ∃ (S : ℕ), S = 36 ∧ 
      (S = bd_A - (bd_A - td_A)) ∧
      (S < bd_B - (bd_B - td_B)) ∧
      (S < bd_C - (bd_C - td_C)) := 
by {
  sorry
}

end minimize_payment_l447_44768


namespace units_digit_k_squared_plus_2_k_l447_44780

noncomputable def k : ℕ := 2009^2 + 2^2009 - 3

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_k_squared_plus_2_k : units_digit (k^2 + 2^k) = 1 := by
  sorry

end units_digit_k_squared_plus_2_k_l447_44780


namespace sqrt_nested_expression_l447_44713

theorem sqrt_nested_expression : 
  Real.sqrt (32 * Real.sqrt (16 * Real.sqrt (8 * Real.sqrt 4))) = 16 := 
by
  sorry

end sqrt_nested_expression_l447_44713


namespace james_total_beverages_l447_44746

-- Define the initial quantities
def initial_sodas := 4 * 10 + 12
def initial_juice_boxes := 3 * 8 + 5
def initial_water_bottles := 2 * 15
def initial_energy_drinks := 7

-- Define the consumption rates
def mon_to_wed_sodas := 3 * 3
def mon_to_wed_juice_boxes := 2 * 3
def mon_to_wed_water_bottles := 1 * 3

def thu_to_sun_sodas := 2 * 4
def thu_to_sun_juice_boxes := 4 * 4
def thu_to_sun_water_bottles := 1 * 4
def thu_to_sun_energy_drinks := 1 * 4

-- Define total beverages consumed
def total_consumed_sodas := mon_to_wed_sodas + thu_to_sun_sodas
def total_consumed_juice_boxes := mon_to_wed_juice_boxes + thu_to_sun_juice_boxes
def total_consumed_water_bottles := mon_to_wed_water_bottles + thu_to_sun_water_bottles
def total_consumed_energy_drinks := thu_to_sun_energy_drinks

-- Define total beverages consumed by the end of the week
def total_beverages_consumed := total_consumed_sodas + total_consumed_juice_boxes + total_consumed_water_bottles + total_consumed_energy_drinks

-- The theorem statement to prove
theorem james_total_beverages : total_beverages_consumed = 50 :=
  by sorry

end james_total_beverages_l447_44746


namespace cos_double_angle_l447_44700

theorem cos_double_angle (α : ℝ) (h : Real.cos α = -Real.sqrt 3 / 2) : Real.cos (2 * α) = 1 / 2 :=
by
  sorry

end cos_double_angle_l447_44700


namespace midpoint_sum_l447_44734

theorem midpoint_sum (x y : ℝ) (h1 : (x + 0) / 2 = 2) (h2 : (y + 9) / 2 = 4) : x + y = 3 := by
  sorry

end midpoint_sum_l447_44734


namespace find_x_plus_y_l447_44702

theorem find_x_plus_y (x y : ℝ) (h1 : x + Real.sin y = 2008) (h2 : x + 2008 * Real.cos y = 2007) (h3 : 0 ≤ y ∧ y ≤ Real.pi / 2) :
  x + y = 2007 + Real.pi / 2 :=
by
  sorry

end find_x_plus_y_l447_44702


namespace sum_of_x_and_y_greater_equal_twice_alpha_l447_44708

theorem sum_of_x_and_y_greater_equal_twice_alpha (x y α : ℝ) 
  (h : Real.sqrt (1 + x) + Real.sqrt (1 + y) = 2 * Real.sqrt (1 + α)) :
  x + y ≥ 2 * α :=
sorry

end sum_of_x_and_y_greater_equal_twice_alpha_l447_44708


namespace midpoint_line_l447_44757

theorem midpoint_line (a : ℝ) (P Q M : ℝ × ℝ) (hP : P = (a, 5 * a + 3)) (hQ : Q = (3, -2))
  (hM : M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)) : M.2 = 5 * M.1 - 7 := 
sorry

end midpoint_line_l447_44757


namespace bus_speed_l447_44769

theorem bus_speed (distance time : ℝ) (h_distance : distance = 201) (h_time : time = 3) : 
  distance / time = 67 :=
by
  sorry

end bus_speed_l447_44769


namespace percentage_of_filled_seats_l447_44741

theorem percentage_of_filled_seats (total_seats vacant_seats : ℕ) (h_total : total_seats = 600) (h_vacant : vacant_seats = 240) :
  (total_seats - vacant_seats) * 100 / total_seats = 60 :=
by
  sorry

end percentage_of_filled_seats_l447_44741


namespace proof_statements_l447_44782

theorem proof_statements (m : ℝ) (x y : ℝ)
  (h1 : 2 * x + y = 4 - m)
  (h2 : x - 2 * y = 3 * m) :
  (m = 1 → (x = 9 / 5 ∧ y = -3 / 5)) ∧
  (3 * x - y = 4 + 2 * m) ∧
  ¬(∃ (m' : ℝ), (8 + m') / 5 < 0 ∧ (4 - 7 * m') / 5 < 0) :=
sorry

end proof_statements_l447_44782


namespace initial_number_divisible_by_15_l447_44714

theorem initial_number_divisible_by_15 (N : ℕ) (h : (N - 7) % 15 = 0) : N = 22 := 
by
  sorry

end initial_number_divisible_by_15_l447_44714


namespace positive_integers_divisors_of_2_to_the_n_plus_1_l447_44705

theorem positive_integers_divisors_of_2_to_the_n_plus_1:
  ∀ n : ℕ, 0 < n → (n^2 ∣ 2^n + 1) ↔ (n = 1 ∨ n = 3) :=
by
  sorry

end positive_integers_divisors_of_2_to_the_n_plus_1_l447_44705


namespace augmented_matrix_solution_l447_44749

theorem augmented_matrix_solution (c1 c2 : ℚ) 
    (h1 : 2 * (3 : ℚ) + 3 * (5 : ℚ) = c1)
    (h2 : (5 : ℚ) = c2) : 
    c1 - c2 = 16 := 
by 
  sorry

end augmented_matrix_solution_l447_44749


namespace ab_value_l447_44765

theorem ab_value (a b : ℝ) (h1 : a^2 + b^2 = 2) (h2 : a + b = 3) : a * b = 7/2 :=
by
  sorry

end ab_value_l447_44765


namespace log_base_property_l447_44731

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log x / log a

theorem log_base_property
  (a : ℝ)
  (ha_pos : a > 0)
  (ha_ne_one : a ≠ 1)
  (hf9 : f a 9 = 2) :
  f a (3^a) = 3 :=
by
  sorry

end log_base_property_l447_44731


namespace rectangle_dimensions_l447_44704

theorem rectangle_dimensions (w l : ℝ) 
  (h1 : l = 3 * w) 
  (h2 : 2 * (l + w) = 2 * l * w) : 
  w = 4 / 3 ∧ l = 4 := 
by
  sorry

end rectangle_dimensions_l447_44704


namespace james_farmer_walk_distance_l447_44737

theorem james_farmer_walk_distance (d : ℝ) :
  ∃ d : ℝ,
    (∀ w : ℝ, (w = 300 + 50 → d = 20) ∧ 
             (w' = w * 1.30 ∧ w'' = w' * 1.20 → w'' = 546)) :=
by
  sorry

end james_farmer_walk_distance_l447_44737


namespace dot_product_example_l447_44715

def vector := ℝ × ℝ

-- Define the dot product function
def dot_product (v1 v2 : vector) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem dot_product_example : dot_product (-1, 0) (0, 2) = 0 := by
  sorry

end dot_product_example_l447_44715


namespace maximize_probability_remove_6_l447_44751

-- Definitions
def integers_list : List Int := [-2, -1, 0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12] -- After removing 6
def initial_list : List Int := [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

-- Theorem Statement
theorem maximize_probability_remove_6 :
  ∀ (n : Int),
  n ∈ initial_list →
  n ≠ 6 →
  ∃ (a b : Int), a ∈ integers_list ∧ b ∈ integers_list ∧ a ≠ b ∧ a + b = 12 → False :=
by
  intros n hn hn6
  -- Placeholder for proof
  sorry

end maximize_probability_remove_6_l447_44751


namespace total_votes_l447_44761

noncomputable def total_votes_proof : Prop :=
  ∃ T A : ℝ, 
    A = 0.40 * T ∧ 
    T = A + (A + 70) ∧ 
    T = 350

theorem total_votes : total_votes_proof :=
sorry

end total_votes_l447_44761


namespace xiao_peach_days_l447_44752

theorem xiao_peach_days :
  ∀ (xiao_ming_apples xiao_ming_pears xiao_ming_peaches : ℕ)
    (xiao_hong_apples xiao_hong_pears xiao_hong_peaches : ℕ)
    (both_eat_apples both_eat_pears : ℕ)
    (one_eats_apple_other_eats_pear : ℕ),
    xiao_ming_apples = 4 →
    xiao_ming_pears = 6 →
    xiao_ming_peaches = 8 →
    xiao_hong_apples = 5 →
    xiao_hong_pears = 7 →
    xiao_hong_peaches = 6 →
    both_eat_apples = 3 →
    both_eat_pears = 2 →
    one_eats_apple_other_eats_pear = 3 →
    ∃ (both_eat_peaches_days : ℕ),
      both_eat_peaches_days = 4 := 
sorry

end xiao_peach_days_l447_44752


namespace min_value_inequality_l447_44735

theorem min_value_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  6 * x / (2 * y + z) + 3 * y / (x + 2 * z) + 9 * z / (x + y) ≥ 83 :=
sorry

end min_value_inequality_l447_44735


namespace sum_of_squares_and_cube_unique_l447_44789

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m * m = n

theorem sum_of_squares_and_cube_unique : 
  ∃! (a b c : ℕ), a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ is_perfect_square a ∧ is_perfect_square b ∧ is_perfect_cube c ∧ a + b + c = 100 :=
sorry

end sum_of_squares_and_cube_unique_l447_44789


namespace burmese_python_eats_alligators_l447_44763

theorem burmese_python_eats_alligators (snake_length : ℝ) (alligator_length : ℝ) (alligator_per_week : ℝ) (total_alligators : ℝ) :
  snake_length = 1.4 → alligator_length = 0.5 → alligator_per_week = 1 → total_alligators = 88 →
  (total_alligators / alligator_per_week) * 7 = 616 := by
  intros
  sorry

end burmese_python_eats_alligators_l447_44763


namespace other_root_of_quadratic_l447_44796

theorem other_root_of_quadratic (a : ℝ) :
  (∀ x, (x^2 + 2*x - a) = 0 → x = -3) → (∃ z, z = 1 ∧ (z^2 + 2*z - a) = 0) :=
by
  sorry

end other_root_of_quadratic_l447_44796


namespace box_neg2_0_3_eq_10_div_9_l447_44781

def box (a b c : ℤ) : ℚ :=
  a^b - b^c + c^a

theorem box_neg2_0_3_eq_10_div_9 : box (-2) 0 3 = 10 / 9 :=
by
  sorry

end box_neg2_0_3_eq_10_div_9_l447_44781


namespace intersection_of_sets_l447_44772

open Set

theorem intersection_of_sets :
  let A := {x : ℤ | |x| < 3}
  let B := {x : ℤ | |x| > 1}
  A ∩ B = ({-2, 2} : Set ℤ) := by
  sorry

end intersection_of_sets_l447_44772


namespace unique_triplet_l447_44727

theorem unique_triplet (a b p : ℕ) (hp : Nat.Prime p) (ha : 0 < a) (hb : 0 < b) :
  (1 / (p : ℚ) = 1 / (a^2 : ℚ) + 1 / (b^2 : ℚ)) → (a = 2 ∧ b = 2 ∧ p = 2) :=
by
  sorry

end unique_triplet_l447_44727


namespace triangle_inequality_equality_condition_l447_44744

theorem triangle_inequality (a b c S : ℝ)
  (h_tri : a^2 + b^2 + c^2 = 4 * S * Real.sqrt 3)
  (h_area : S = Real.sqrt ((a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c))):
  a^2 + b^2 + c^2 ≥ 4 * S * Real.sqrt 3 :=
sorry

theorem equality_condition (a b c S : ℝ)
  (h_tri : a^2 + b^2 + c^2 = 4 * S * Real.sqrt 3)
  (h_area : S = Real.sqrt ((a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c))):
  (a = b) ∧ (b = c) :=
sorry

end triangle_inequality_equality_condition_l447_44744


namespace exists_x1_x2_l447_44711

noncomputable def f (a x : ℝ) := a * x + Real.log x

theorem exists_x1_x2 (a : ℝ) (h : a < 0) :
  ∃ x1 x2 : ℝ, 0 < x1 ∧ x1 < x2 ∧ f a x1 ≥ f a x2 :=
by
  sorry

end exists_x1_x2_l447_44711


namespace math_problem_l447_44732

def is_polynomial (expr : String) : Prop := sorry
def is_monomial (expr : String) : Prop := sorry
def is_cubic (expr : String) : Prop := sorry
def is_quintic (expr : String) : Prop := sorry
def correct_option_C : String := "C"

theorem math_problem :
  ¬ is_polynomial "8 - 2 / z" ∧
  ¬ (is_monomial "-x^2yz" ∧ is_cubic "-x^2yz") ∧
  is_polynomial "x^2 - 3xy^2 + 2x^2y^3 - 1" ∧
  is_quintic "x^2 - 3xy^2 + 2x^2y^3 - 1" ∧
  ¬ is_monomial "5b / x" →
  correct_option_C = "C" := sorry

end math_problem_l447_44732


namespace problem1_l447_44798

theorem problem1 (x y : ℝ) (h1 : x + y = 4) (h2 : 2 * x - y = 5) : 
  x = 3 ∧ y = 1 := sorry

end problem1_l447_44798


namespace possible_values_expression_l447_44793

theorem possible_values_expression 
  (a b : ℝ) 
  (h₁ : a^2 = 16) 
  (h₂ : |b| = 3) 
  (h₃ : ab < 0) : 
  (a - b)^2 + a * b^2 = 85 ∨ (a - b)^2 + a * b^2 = 13 := 
by 
  sorry

end possible_values_expression_l447_44793


namespace double_rooms_booked_l447_44722

theorem double_rooms_booked (S D : ℕ) 
  (h1 : S + D = 260) 
  (h2 : 35 * S + 60 * D = 14000) : 
  D = 196 :=
by
  sorry

end double_rooms_booked_l447_44722


namespace right_triangle_one_leg_div_by_3_l447_44725

theorem right_triangle_one_leg_div_by_3 {a b c : ℕ} (a_pos : 0 < a) (b_pos : 0 < b) 
  (h : a^2 + b^2 = c^2) : 3 ∣ a ∨ 3 ∣ b := 
by 
  sorry

end right_triangle_one_leg_div_by_3_l447_44725


namespace restore_axes_with_parabola_l447_44790

-- Define the given parabola y = x^2
def parabola (x : ℝ) : ℝ := x^2

-- Problem: Prove that you can restore the coordinate axes using the given parabola and tools.
theorem restore_axes_with_parabola : 
  ∃ O X Y : ℝ × ℝ, 
  (∀ x, parabola x = (x, x^2).snd) ∧ 
  (X.fst = 0 ∧ Y.snd = 0) ∧
  (O = (0,0)) :=
sorry

end restore_axes_with_parabola_l447_44790


namespace initial_money_l447_44794

theorem initial_money (x : ℝ) (cupcake_cost total_cookie_cost total_cost money_left : ℝ) 
  (h1 : cupcake_cost = 10 * 1.5) 
  (h2 : total_cookie_cost = 5 * 3)
  (h3 : total_cost = cupcake_cost + total_cookie_cost)
  (h4 : money_left = 30)
  (h5 : 3 * x = total_cost + money_left) 
  : x = 20 := 
sorry

end initial_money_l447_44794


namespace part1_part2_l447_44786

open BigOperators

noncomputable def seq (a : ℕ → ℝ) : Prop :=
  (∀ n : ℕ, n ≠ 0 → a n > 0) ∧
  (a 1 = 2) ∧
  (∀ n : ℕ, n ≠ 0 → (n + 1) * (a (n + 1)) ^ 2 = n * (a n) ^ 2 + a n)

theorem part1 (a : ℕ → ℝ) (h : seq a)
  (n : ℕ) (hn : n ≠ 0) 
  : 1 < a (n+1) ∧ a (n+1) < a n :=
sorry

theorem part2 (a : ℕ → ℝ) (h : seq a)
  : ∑ k in Finset.range 2022 \ {0}, (a (k+1))^2 / (k+1)^2 < 2 :=
sorry

end part1_part2_l447_44786


namespace range_of_x_for_expression_meaningful_l447_44721

theorem range_of_x_for_expression_meaningful (x : ℝ) :
  (x - 1 > 0 ∧ x ≠ 1) ↔ x > 1 :=
by
  sorry

end range_of_x_for_expression_meaningful_l447_44721


namespace find_intersection_l447_44743

noncomputable def setM : Set ℝ := {x : ℝ | x^2 ≤ 9}
noncomputable def setN : Set ℝ := {x : ℝ | x ≤ 1}
noncomputable def intersection : Set ℝ := {x : ℝ | -3 ≤ x ∧ x ≤ 1}

theorem find_intersection (x : ℝ) : (x ∈ setM ∧ x ∈ setN) ↔ (x ∈ intersection) := 
by sorry

end find_intersection_l447_44743


namespace sin_225_eq_neg_sqrt2_over_2_l447_44720

theorem sin_225_eq_neg_sqrt2_over_2 : Real.sin (225 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by sorry

end sin_225_eq_neg_sqrt2_over_2_l447_44720


namespace evaluate_expression_l447_44716

theorem evaluate_expression :
  (2 * 10^3)^3 = 8 * 10^9 :=
by
  sorry

end evaluate_expression_l447_44716


namespace probability_selecting_girl_l447_44710

def boys : ℕ := 3
def girls : ℕ := 1
def total_candidates : ℕ := boys + girls
def favorable_outcomes : ℕ := girls

theorem probability_selecting_girl : 
  ∃ p : ℚ, p = (favorable_outcomes : ℚ) / (total_candidates : ℚ) ∧ p = 1 / 4 :=
sorry

end probability_selecting_girl_l447_44710


namespace stickers_after_loss_l447_44724

-- Conditions
def stickers_per_page : ℕ := 20
def initial_pages : ℕ := 12
def lost_pages : ℕ := 1

-- Problem statement
theorem stickers_after_loss : (initial_pages - lost_pages) * stickers_per_page = 220 := by
  sorry

end stickers_after_loss_l447_44724


namespace trigonometric_identity_proof_l447_44784

noncomputable def a : ℝ := -35 / 6 * Real.pi

theorem trigonometric_identity_proof :
  (2 * Real.sin (Real.pi + a) * Real.cos (Real.pi - a) - Real.cos (Real.pi + a)) / 
  (1 + Real.sin a ^ 2 + Real.sin (Real.pi - a) - Real.cos (Real.pi + a) ^ 2) = Real.sqrt 3 := 
by
  sorry

end trigonometric_identity_proof_l447_44784


namespace find_a_l447_44709

theorem find_a (a : ℝ) (h₁ : a > 1) (h₂ : (∀ x : ℝ, a^3 = 8)) : a = 2 :=
by
  sorry

end find_a_l447_44709


namespace tan_eq_example_l447_44730

theorem tan_eq_example (x : ℝ) (hx : Real.tan (3 * x) * Real.tan (5 * x) = Real.tan (7 * x) * Real.tan (9 * x)) : x = 30 * Real.pi / 180 :=
  sorry

end tan_eq_example_l447_44730


namespace can_lids_per_box_l447_44788

/-- Aaron initially has 14 can lids, and after adding can lids from 3 boxes,
he has a total of 53 can lids. How many can lids are in each box? -/
theorem can_lids_per_box (initial : ℕ) (total : ℕ) (boxes : ℕ) (h₀ : initial = 14) (h₁ : total = 53) (h₂ : boxes = 3) :
  (total - initial) / boxes = 13 :=
by
  sorry

end can_lids_per_box_l447_44788


namespace chimney_bricks_l447_44701

variable (h : ℕ)

/-- Brenda would take 8 hours to build a chimney alone. 
    Brandon would take 12 hours to build it alone. 
    When they work together, their efficiency is diminished by 15 bricks per hour due to their chatting. 
    If they complete the chimney in 6 hours when working together, then the total number of bricks in the chimney is 360. -/
theorem chimney_bricks
  (h : ℕ)
  (Brenda_rate : ℕ)
  (Brandon_rate : ℕ)
  (effective_rate : ℕ)
  (completion_time : ℕ)
  (h_eq : Brenda_rate = h / 8)
  (h_eq_alt : Brandon_rate = h / 12)
  (effective_rate_eq : effective_rate = (Brenda_rate + Brandon_rate) - 15)
  (completion_eq : 6 * effective_rate = h) :
  h = 360 := by 
  sorry

end chimney_bricks_l447_44701


namespace factorize_problem1_factorize_problem2_l447_44764

-- Problem 1: Factorization of 4x^2 - 16
theorem factorize_problem1 (x : ℝ) : 4 * x^2 - 16 = 4 * (x - 2) * (x + 2) :=
by
  sorry

-- Problem 2: Factorization of a^2b - 4ab + 4b
theorem factorize_problem2 (a b : ℝ) : a^2 * b - 4 * a * b + 4 * b = b * (a - 2) ^ 2 :=
by
  sorry

end factorize_problem1_factorize_problem2_l447_44764


namespace kolya_win_l447_44703

theorem kolya_win : ∀ stones : ℕ, stones = 100 → (∃ strategy : (ℕ → ℕ × ℕ), ∀ opponent_strategy : (ℕ → ℕ × ℕ), true → true) :=
by
  sorry

end kolya_win_l447_44703


namespace fraction_to_decimal_l447_44787

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 := by
  sorry

end fraction_to_decimal_l447_44787


namespace value_of_m_l447_44717

theorem value_of_m (m : ℤ) (h1 : abs m = 2) (h2 : m - 2 ≠ 0) : m = -2 :=
by
  sorry

end value_of_m_l447_44717


namespace rightmost_three_digits_of_7_pow_2023_l447_44742

theorem rightmost_three_digits_of_7_pow_2023 :
  (7 ^ 2023) % 1000 = 637 :=
sorry

end rightmost_three_digits_of_7_pow_2023_l447_44742


namespace taco_castle_num_dodge_trucks_l447_44739

theorem taco_castle_num_dodge_trucks
  (D F T V H C : ℕ)
  (hV : V = 5)
  (h1 : F = D / 3)
  (h2 : F = 2 * T)
  (h3 : V = T / 2)
  (h4 : H = 3 * F / 4)
  (h5 : C = 2 * H / 3) :
  D = 60 :=
by
  sorry

end taco_castle_num_dodge_trucks_l447_44739


namespace smallest_number_of_marbles_l447_44775

theorem smallest_number_of_marbles :
  ∃ (r w b g y : ℕ), 
  (r + w + b + g + y = 13) ∧ 
  (r ≥ 5) ∧
  (r - 4 = 5 * w) ∧
  ((r - 3) * (r - 4) = 20 * w * b) ∧
  sorry := sorry

end smallest_number_of_marbles_l447_44775


namespace find_k_l447_44762

/- Definitions for vectors -/
def vector_a : ℝ × ℝ := (1, 2)
def vector_b : ℝ × ℝ := (-3, 2)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

/- Prove that if ka + b is perpendicular to a, then k = -1/5 -/
theorem find_k (k : ℝ) : 
  dot_product (k • (1, 2) + (-3, 2)) (1, 2) = 0 → 
  k = -1 / 5 := 
  sorry

end find_k_l447_44762


namespace problem_solution_l447_44759

theorem problem_solution (A B : ℝ) (h : ∀ x, x ≠ 3 → (A / (x - 3)) + B * (x + 2) = (-4 * x^2 + 14 * x + 38) / (x - 3)) : 
  A + B = 46 :=
sorry

end problem_solution_l447_44759


namespace ratio_of_tagged_fish_is_1_over_25_l447_44753

-- Define the conditions
def T70 : ℕ := 70  -- Number of tagged fish first caught and tagged
def T50 : ℕ := 50  -- Total number of fish caught in the second sample
def t2 : ℕ := 2    -- Number of tagged fish in the second sample

-- State the theorem/question
theorem ratio_of_tagged_fish_is_1_over_25 : (t2 / T50) = 1 / 25 :=
by
  sorry

end ratio_of_tagged_fish_is_1_over_25_l447_44753


namespace imaginaria_city_population_l447_44719

theorem imaginaria_city_population (a b c : ℕ) (h₁ : a^2 + 225 = b^2 + 1) (h₂ : b^2 + 1 + 75 = c^2) : 5 ∣ a^2 :=
by
  sorry

end imaginaria_city_population_l447_44719


namespace fraction_ratio_l447_44745

theorem fraction_ratio
  (m n p q r : ℚ)
  (h1 : m / n = 20)
  (h2 : p / n = 4)
  (h3 : p / q = 1 / 5)
  (h4 : m / r = 10) :
  r / q = 1 / 10 :=
by
  sorry

end fraction_ratio_l447_44745


namespace value_of_a_squared_plus_b_squared_plus_2ab_l447_44754

theorem value_of_a_squared_plus_b_squared_plus_2ab (a b : ℝ) (h : a + b = -1) :
  a^2 + b^2 + 2 * a * b = 1 :=
by sorry

end value_of_a_squared_plus_b_squared_plus_2ab_l447_44754


namespace jemma_grasshoppers_l447_44799

-- Definitions corresponding to the conditions
def grasshoppers_on_plant : ℕ := 7
def baby_grasshoppers : ℕ := 2 * 12

-- Theorem statement equivalent to the problem
theorem jemma_grasshoppers : grasshoppers_on_plant + baby_grasshoppers = 31 :=
by
  sorry

end jemma_grasshoppers_l447_44799


namespace fourth_intersection_point_l447_44770

def intersect_curve_circle : Prop :=
  let curve_eq (x y : ℝ) : Prop := x * y = 1
  let circle_intersects_points (h k s : ℝ) : Prop :=
    ∃ (x1 y1 x2 y2 x3 y3 : ℝ), 
    (x1, y1) = (3, (1 : ℝ) / 3) ∧ 
    (x2, y2) = (-4, -(1 : ℝ) / 4) ∧ 
    (x3, y3) = ((1 : ℝ) / 6, 6) ∧ 
    (x1 - h)^2 + (y1 - k)^2 = s^2 ∧
    (x2 - h)^2 + (y2 - k)^2 = s^2 ∧
    (x3 - h)^2 + (y3 - k)^2 = s^2 
  let fourth_point_of_intersection (x y : ℝ) : Prop := 
    x = -(1 : ℝ) / 2 ∧ 
    y = -2
  curve_eq 3 ((1 : ℝ) / 3) ∧
  curve_eq (-4) (-(1 : ℝ) / 4) ∧
  curve_eq ((1 : ℝ) / 6) 6 ∧
  ∃ h k s, circle_intersects_points h k s →
  ∃ (x4 y4 : ℝ), curve_eq x4 y4 ∧
  fourth_point_of_intersection x4 y4

theorem fourth_intersection_point :
  intersect_curve_circle := by
  sorry

end fourth_intersection_point_l447_44770


namespace total_cost_of_suits_l447_44723

theorem total_cost_of_suits : 
    ∃ o t : ℕ, o = 300 ∧ t = 3 * o + 200 ∧ o + t = 1400 :=
by
  sorry

end total_cost_of_suits_l447_44723


namespace triangle_angle_contradiction_l447_44755

theorem triangle_angle_contradiction (α β γ : ℝ) (h1 : α + β + γ = 180) (h2 : α > 60) (h3 : β > 60) (h4 : γ > 60) : false :=
sorry

end triangle_angle_contradiction_l447_44755


namespace linear_transformation_proof_l447_44706

theorem linear_transformation_proof (x : ℝ) (h1 : 0 ≤ x ∧ x ≤ 1) :
  ∃ (k b : ℝ), k = 4 ∧ b = -1 ∧ (y = k * x + b ∧ -1 ≤ y ∧ y ≤ 3) :=
by
  sorry

end linear_transformation_proof_l447_44706


namespace rectangular_region_area_l447_44733

-- Definitions based on conditions
variable (w : ℝ) -- length of the shorter sides
variable (l : ℝ) -- length of the longer side
variable (total_fence_length : ℝ) -- total length of the fence

-- Given conditions as hypotheses
theorem rectangular_region_area
  (h1 : l = 2 * w) -- The length of the side opposite the wall is twice the length of each of the other two fenced sides
  (h2 : w + w + l = total_fence_length) -- The total length of the fence is 40 feet
  (h3 : total_fence_length = 40) -- total fence length of 40 feet
: (w * l) = 200 := -- The area of the rectangular region is 200 square feet
sorry

end rectangular_region_area_l447_44733


namespace days_taken_to_complete_work_l447_44797

-- Conditions
def work_rate_B : ℚ := 1 / 33
def work_rate_A : ℚ := 2 * work_rate_B
def combined_work_rate : ℚ := work_rate_A + work_rate_B

-- Proof statement
theorem days_taken_to_complete_work : combined_work_rate ≠ 0 → 1 / combined_work_rate = 11 :=
by
  sorry

end days_taken_to_complete_work_l447_44797


namespace minimum_value_of_quadratic_function_l447_44712

def quadratic_function (a x : ℝ) : ℝ :=
  4 * x ^ 2 - 4 * a * x + (a ^ 2 - 2 * a + 2)

def min_value_in_interval (f : ℝ → ℝ) (a : ℝ) (interval : Set ℝ) (min_val : ℝ) : Prop :=
  ∀ x ∈ interval, f x ≥ min_val ∧ ∃ y ∈ interval, f y = min_val

theorem minimum_value_of_quadratic_function :
  ∃ a : ℝ, min_value_in_interval (quadratic_function a) a {x | 0 ≤ x ∧ x ≤ 1} 2 ↔ (a = 0 ∨ a = 3 + Real.sqrt 5) :=
by
  sorry

end minimum_value_of_quadratic_function_l447_44712


namespace point_P_outside_circle_l447_44736

theorem point_P_outside_circle (a b : ℝ) (h : ∃ x y : ℝ, a * x + b * y = 1 ∧ x^2 + y^2 = 1) :
  a^2 + b^2 > 1 :=
sorry

end point_P_outside_circle_l447_44736


namespace total_revenue_4706_l447_44792

noncomputable def totalTicketRevenue (seats : ℕ) (show2pm : ℕ × ℕ) (show5pm : ℕ × ℕ) (show8pm : ℕ × ℕ) : ℕ :=
  let revenue2pm := show2pm.1 * 4 + (seats - show2pm.1) * 6
  let revenue5pm := show5pm.1 * 5 + (seats - show5pm.1) * 8
  let revenue8pm := show8pm.1 * 7 + (show8pm.2 - show8pm.1) * 10
  revenue2pm + revenue5pm + revenue8pm

theorem total_revenue_4706 :
  totalTicketRevenue 250 (135, 250) (160, 250) (98, 225) = 4706 :=
by
  unfold totalTicketRevenue
  -- We provide the proof steps here in a real proof scenario.
  -- We are focusing on the statement formulation only.
  sorry

end total_revenue_4706_l447_44792


namespace power_equivalence_l447_44778

theorem power_equivalence (K : ℕ) : 32^2 * 4^5 = 2^K ↔ K = 20 :=
by sorry

end power_equivalence_l447_44778


namespace exists_smallest_n_l447_44718

theorem exists_smallest_n :
  ∃ n : ℕ, (n^2 + 20 * n + 19) % 2019 = 0 ∧ n = 2000 :=
sorry

end exists_smallest_n_l447_44718


namespace molar_weight_of_BaF2_l447_44777

theorem molar_weight_of_BaF2 (Ba_weight : Real) (F_weight : Real) (num_moles : ℕ) 
    (Ba_weight_val : Ba_weight = 137.33) (F_weight_val : F_weight = 18.998) 
    (num_moles_val : num_moles = 6) 
    : (137.33 + 2 * 18.998) * 6 = 1051.956 := 
by
  sorry

end molar_weight_of_BaF2_l447_44777


namespace plane_equation_l447_44748

variable (x y z : ℝ)

def pointA : ℝ × ℝ × ℝ := (3, 0, 0)
def normalVector : ℝ × ℝ × ℝ := (2, -3, 1)

theorem plane_equation : 
  ∃ a b c d, normalVector = (a, b, c) ∧ pointA = (x, y, z) ∧ a * (x - 3) + b * y + c * z = d ∧ d = -6 := 
  sorry

end plane_equation_l447_44748


namespace WangLei_is_13_l447_44767

-- We need to define the conditions and question in Lean 4
def WangLei_age (x : ℕ) : Prop :=
  3 * x - 8 = 31

theorem WangLei_is_13 : ∃ x : ℕ, WangLei_age x ∧ x = 13 :=
by
  use 13
  unfold WangLei_age
  sorry

end WangLei_is_13_l447_44767


namespace lines_divide_circle_into_four_arcs_l447_44707

theorem lines_divide_circle_into_four_arcs (a b : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 = 1 → y = x + a ∨ y = x + b) →
  a^2 + b^2 = 2 :=
by
  sorry

end lines_divide_circle_into_four_arcs_l447_44707


namespace abc_geq_inequality_l447_44774

open Real

theorem abc_geq_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a * b * c ≥ (a + b - c) * (b + c - a) * (c + a - b) :=
by
  sorry

end abc_geq_inequality_l447_44774


namespace trig_identity_1_trig_identity_2_l447_44776

noncomputable def point := ℚ × ℚ

namespace TrigProblem

open Real

def point_on_terminal_side (α : ℝ) (p : point) : Prop :=
  let (x, y) := p
  ∃ r : ℝ, r = sqrt (x^2 + y^2) ∧ x/r = cos α ∧ y/r = sin α

theorem trig_identity_1 {α : ℝ} (h : point_on_terminal_side α (-4, 3)) :
  (sin (π / 2 + α) - cos (π + α)) / (sin (π / 2 - α) - sin (π - α)) = 8 / 7 :=
sorry

theorem trig_identity_2 {α : ℝ} (h : point_on_terminal_side α (-4, 3)) :
  sin α * cos α = -12 / 25 :=
sorry

end TrigProblem

end trig_identity_1_trig_identity_2_l447_44776


namespace find_n_l447_44729

noncomputable def tangent_line_problem (x0 : ℝ) (n : ℕ) : Prop :=
(x0 ∈ Set.Ioo (Real.sqrt n) (Real.sqrt (n + 1))) ∧
(∃ m : ℝ, 0 < m ∧ m < 1 ∧ (2 * x0 = 1 / m) ∧ (x0^2 = (Real.log m - 1)))

theorem find_n (x0 : ℝ) (n : ℕ) :
  tangent_line_problem x0 n → n = 2 :=
sorry

end find_n_l447_44729


namespace guitar_price_proof_l447_44760

def total_guitar_price (x : ℝ) : Prop :=
  0.20 * x = 240 → x = 1200

theorem guitar_price_proof (x : ℝ) (h : 0.20 * x = 240) : x = 1200 :=
by
  sorry

end guitar_price_proof_l447_44760


namespace part_a_part_b_l447_44773

theorem part_a (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z ≥ 3) :
  ¬ (1/x + 1/y + 1/z ≤ 3) :=
sorry

theorem part_b (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z ≤ 3) :
  1/x + 1/y + 1/z ≥ 3 :=
sorry

end part_a_part_b_l447_44773


namespace angle4_is_35_l447_44795

theorem angle4_is_35
  (angle1 angle2 angle3 angle4 angle5 angle6 : ℝ)
  (h1 : angle1 + angle2 = 180)
  (h2 : angle3 = angle4)
  (ha : angle1 = 50)
  (h_opposite : angle5 = 60)
  (triangle_sum : angle1 + angle5 + angle6 = 180)
  (supplementary_angle : angle2 + angle6 = 180) :
  angle4 = 35 :=
by
  sorry

end angle4_is_35_l447_44795


namespace avg_speed_BC_60_mph_l447_44747

theorem avg_speed_BC_60_mph 
  (d_AB : ℕ) (d_BC : ℕ) (avg_speed_total : ℚ) (time_ratio : ℚ) (t_AB : ℕ) :
  d_AB = 120 ∧ d_BC = 60 ∧ avg_speed_total = 45 ∧ time_ratio = 3 ∧
  t_AB = 3 → (d_BC / (t_AB / time_ratio) = 60) :=
by
  sorry

end avg_speed_BC_60_mph_l447_44747


namespace a_cubed_plus_b_cubed_gt_a_sq_b_plus_ab_sq_l447_44791

theorem a_cubed_plus_b_cubed_gt_a_sq_b_plus_ab_sq {a b : ℝ} (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a ≠ b) : a^3 + b^3 > a^2 * b + a * b^2 := by
  sorry

end a_cubed_plus_b_cubed_gt_a_sq_b_plus_ab_sq_l447_44791

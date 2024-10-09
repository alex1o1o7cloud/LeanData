import Mathlib

namespace sector_angle_radian_measure_l2077_207758

theorem sector_angle_radian_measure (r l : ℝ) (h1 : r = 1) (h2 : l = 2) : l / r = 2 := by
  sorry

end sector_angle_radian_measure_l2077_207758


namespace find_y_l2077_207747

noncomputable def inverse_proportion_y_value (x y k : ℝ) : Prop :=
  (x * y = k) ∧ (x + y = 52) ∧ (x = 3 * y) ∧ (x = -10) → (y = -50.7)

theorem find_y (x y k : ℝ) (h : inverse_proportion_y_value x y k) : y = -50.7 :=
  sorry

end find_y_l2077_207747


namespace chess_game_problem_l2077_207768

-- Mathematical definitions based on the conditions
def petr_wins : ℕ := 6
def petr_draws : ℕ := 2
def karel_points : ℤ := 9
def points_for_win : ℕ := 3
def points_for_loss : ℕ := 2
def points_for_draw : ℕ := 0

-- Defining the final statement to prove
theorem chess_game_problem :
    ∃ (total_games : ℕ) (leader : String), total_games = 15 ∧ leader = "Karel" := 
by
  -- Placeholder for proof
  sorry

end chess_game_problem_l2077_207768


namespace correct_calculation_l2077_207736

theorem correct_calculation (x y : ℝ) : (x * y^2) ^ 2 = x^2 * y^4 :=
by
  sorry

end correct_calculation_l2077_207736


namespace sum_cube_eq_l2077_207786

theorem sum_cube_eq (a b c : ℝ) (h : a + b + c = 0) : a^3 + b^3 + c^3 = 3 * a * b * c :=
by 
  sorry

end sum_cube_eq_l2077_207786


namespace max_a_plus_b_l2077_207773

theorem max_a_plus_b (a b c d e : ℕ) (h1 : a < b) (h2 : b < c) (h3 : c < d) (h4 : d < e) (h5 : a + 2*b + 3*c + 4*d + 5*e = 300) : a + b ≤ 35 :=
sorry

end max_a_plus_b_l2077_207773


namespace sequence_property_l2077_207798

theorem sequence_property {m : ℤ} (h_m : |m| ≥ 2) (a : ℕ → ℤ)
  (h_nonzero : ¬(a 1 = 0 ∧ a 2 = 0))
  (h_rec : ∀ n : ℕ, a (n+2) = a (n+1) - m * a n)
  (r s : ℕ) (h_r_s : r > s ∧ s ≥ 2) (h_eq : a r = a s ∧ a s = a 1) :
  r - s ≥ |m| := sorry

end sequence_property_l2077_207798


namespace angle_of_inclination_l2077_207792

theorem angle_of_inclination (x y : ℝ) (θ : ℝ) :
  (x - y - 1 = 0) → θ = 45 :=
by
  sorry

end angle_of_inclination_l2077_207792


namespace no_linear_factor_l2077_207738

theorem no_linear_factor : ∀ x y z : ℤ,
  ¬ ∃ a b c : ℤ, a*x + b*y + c*z + (x^2 - y^2 + z^2 - 2*y*z + 2*x - 3*y + z) = 0 :=
by sorry

end no_linear_factor_l2077_207738


namespace smallest_X_l2077_207732

theorem smallest_X (T : ℕ) (hT_digits : ∀ d, d ∈ T.digits 10 → d = 0 ∨ d = 1) (hX_int : ∃ (X : ℕ), T = 20 * X) : ∃ T, ∀ X, X = T / 20 → X = 55 :=
by
  sorry

end smallest_X_l2077_207732


namespace intersection_M_N_l2077_207733

def M : Set ℝ := {x | x < 2}
def N : Set ℝ := {x | -1 < x ∧ x < 3}

theorem intersection_M_N :
  M ∩ N = {x | -1 < x ∧ x <2} := by
  sorry

end intersection_M_N_l2077_207733


namespace linda_five_dollar_bills_l2077_207746

theorem linda_five_dollar_bills :
  ∃ (x y : ℕ), x + y = 15 ∧ 5 * x + 10 * y = 100 ∧ x = 10 :=
by
  sorry

end linda_five_dollar_bills_l2077_207746


namespace sum_of_coefficients_l2077_207772

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℤ) :
  (2 * x + 1)^5 = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4 + a₅ * x^5 →
  a₀ = 1 →
  a₁ + a₂ + a₃ + a₄ + a₅ = 3^5 - 1 :=
by
  intros h_expand h_a₀
  sorry

end sum_of_coefficients_l2077_207772


namespace first_shaded_square_ensuring_all_columns_l2077_207796

def triangular_number (n : ℕ) : ℕ :=
  n * (n + 1) / 2

def shaded_squares_in_columns (k : ℕ) : Prop :=
  ∀ j : ℕ, j < 7 → ∃ n : ℕ, triangular_number n % 7 = j ∧ triangular_number n ≤ k

theorem first_shaded_square_ensuring_all_columns:
  shaded_squares_in_columns 55 :=
by
  sorry

end first_shaded_square_ensuring_all_columns_l2077_207796


namespace martha_initial_blocks_l2077_207791

theorem martha_initial_blocks (final_blocks : ℕ) (found_blocks : ℕ) (initial_blocks : ℕ) : 
  final_blocks = initial_blocks + found_blocks → 
  final_blocks = 84 →
  found_blocks = 80 → 
  initial_blocks = 4 :=
by
  intros h1 h2 h3
  sorry

end martha_initial_blocks_l2077_207791


namespace matrix_vector_subtraction_l2077_207707

open Matrix

variable {α : Type*} [AddCommGroup α] [Module ℝ α]

def matrix_mul_vector (M : Matrix (Fin 2) (Fin 2) ℝ) (v : Fin 2 → ℝ) : Fin 2 → ℝ :=
  M.mulVec v

theorem matrix_vector_subtraction (M : Matrix (Fin 2) (Fin 2) ℝ) (v w : Fin 2 → ℝ)
  (hv : matrix_mul_vector M v = ![4, 6])
  (hw : matrix_mul_vector M w = ![5, -4]) :
  matrix_mul_vector M (v - (2 : ℝ) • w) = ![-6, 14] :=
sorry

end matrix_vector_subtraction_l2077_207707


namespace units_digit_is_valid_l2077_207727

theorem units_digit_is_valid (n : ℕ) : 
  (∃ k : ℕ, (k^3 % 10 = n)) → 
  (n = 2 ∨ n = 3 ∨ n = 7 ∨ n = 8 ∨ n = 9) :=
by sorry

end units_digit_is_valid_l2077_207727


namespace interest_earned_l2077_207720

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r) ^ n

theorem interest_earned (P : ℝ) (r : ℝ) (n : ℕ) (A : ℝ) (I : ℝ):
  P = 1500 → r = 0.12 → n = 4 →
  A = compound_interest P r n →
  I = A - P →
  I = 862.2 :=
by
  intros hP hr hn hA hI
  sorry

end interest_earned_l2077_207720


namespace vinnie_tips_l2077_207785

variable (Paul Vinnie : ℕ)

def tips_paul := 14
def more_vinnie_than_paul := 16

theorem vinnie_tips :
  Vinnie = tips_paul + more_vinnie_than_paul :=
by
  unfold tips_paul more_vinnie_than_paul -- unfolding defined values
  exact sorry

end vinnie_tips_l2077_207785


namespace scientific_notation_000073_l2077_207706

theorem scientific_notation_000073 : 0.000073 = 7.3 * 10^(-5) := by
  sorry

end scientific_notation_000073_l2077_207706


namespace find_constant_l2077_207793

theorem find_constant (x1 x2 : ℝ) (C : ℝ) :
  x1 - x2 = 5.5 ∧
  x1 + x2 = -5 / 2 ∧
  x1 * x2 = C / 2 →
  C = -12 :=
by
  -- proof goes here
  sorry

end find_constant_l2077_207793


namespace winnie_servings_l2077_207708

theorem winnie_servings:
  ∀ (x : ℝ), 
  (2 / 5) * x + (21 / 25) * x = 82 →
  x = 30 :=
by
  sorry

end winnie_servings_l2077_207708


namespace younger_person_age_l2077_207703

theorem younger_person_age 
  (y e : ℕ)
  (h1 : e = y + 20)
  (h2 : e - 4 = 5 * (y - 4)) : 
  y = 9 := 
sorry

end younger_person_age_l2077_207703


namespace divisibility_of_f_by_cubic_factor_l2077_207709

noncomputable def f (x : ℂ) (m n : ℕ) : ℂ := x^(3 * m + 2) + (-x^2 - 1)^(3 * n + 1) + 1

theorem divisibility_of_f_by_cubic_factor (m n : ℕ) : ∀ x : ℂ, x^2 + x + 1 = 0 → f x m n = 0 :=
by
  sorry

end divisibility_of_f_by_cubic_factor_l2077_207709


namespace gcd_of_three_numbers_l2077_207783

theorem gcd_of_three_numbers : 
  let a := 4560
  let b := 6080
  let c := 16560
  gcd (gcd a b) c = 80 := 
by {
  -- placeholder for the proof
  sorry
}

end gcd_of_three_numbers_l2077_207783


namespace number_of_children_l2077_207705

theorem number_of_children (total_oranges : ℕ) (oranges_per_child : ℕ) (h1 : oranges_per_child = 3) (h2 : total_oranges = 12) : total_oranges / oranges_per_child = 4 :=
by
  sorry

end number_of_children_l2077_207705


namespace difference_of_squares_example_l2077_207700

theorem difference_of_squares_example :
  (262^2 - 258^2 = 2080) :=
by {
  sorry -- placeholder for the actual proof
}

end difference_of_squares_example_l2077_207700


namespace initial_average_l2077_207799

variable (A : ℝ)
variables (nums : Fin 5 → ℝ)
variables (h_sum : 5 * A = nums 0 + nums 1 + nums 2 + nums 3 + nums 4)
variables (h_num : nums 0 = 12)
variables (h_new_avg : (5 * A + 12) / 5 = 9.2)

theorem initial_average :
  A = 6.8 :=
sorry

end initial_average_l2077_207799


namespace range_of_t_l2077_207781

noncomputable def f (a x : ℝ) : ℝ :=
  a / x - x + a * Real.log x

noncomputable def g (a x : ℝ) : ℝ :=
  f a x + 1/2 * x^2 - (a - 1) * x - a / x

theorem range_of_t (a x₁ x₂ t : ℝ) (h1 : f a x₁ = f a x₂) (h2 : x₁ + x₂ = a)
  (h3 : x₁ * x₂ = a) (h4 : a > 4) (h5 : g a x₁ + g a x₂ > t * (x₁ + x₂)) :
  t < Real.log 4 - 3 :=
  sorry

end range_of_t_l2077_207781


namespace incorrect_operation_l2077_207787

theorem incorrect_operation 
    (x y : ℝ) :
    (x - y) / (x + y) = (y - x) / (y + x) ↔ False := 
by 
  sorry

end incorrect_operation_l2077_207787


namespace hiker_final_distance_l2077_207797

theorem hiker_final_distance :
  let east := 24
  let north := 7
  let west := 15
  let south := 5
  let net_east := east - west
  let net_north := north - south
  net_east = 9 ∧ net_north = 2 →
  Real.sqrt ((net_east)^2 + (net_north)^2) = Real.sqrt 85 :=
by
  intros
  sorry

end hiker_final_distance_l2077_207797


namespace center_of_circle_param_eq_l2077_207719

theorem center_of_circle_param_eq (θ : ℝ) : 
  (∃ c : ℝ × ℝ, ∀ θ, 
    ∃ (x y : ℝ), 
      (x = 2 + 2 * Real.cos θ) ∧ 
      (y = 2 * Real.sin θ) ∧ 
      (x - c.1)^2 + y^2 = 4) 
  ↔ 
  c = (2, 0) :=
by
  sorry

end center_of_circle_param_eq_l2077_207719


namespace eddies_sister_pies_per_day_l2077_207759

theorem eddies_sister_pies_per_day 
  (Eddie_daily : ℕ := 3) 
  (Mother_daily : ℕ := 8) 
  (total_days : ℕ := 7)
  (total_pies : ℕ := 119) :
  ∃ (S : ℕ), S = 6 ∧ (Eddie_daily * total_days + Mother_daily * total_days + S * total_days = total_pies) :=
by
  sorry

end eddies_sister_pies_per_day_l2077_207759


namespace zookeeper_fish_total_l2077_207713

def fish_given : ℕ := 19
def fish_needed : ℕ := 17

theorem zookeeper_fish_total : fish_given + fish_needed = 36 :=
by
  sorry

end zookeeper_fish_total_l2077_207713


namespace simplify_expression_l2077_207721

open Real

-- Define the given expression as a function of x
noncomputable def given_expression (x : ℝ) (hx : 0 < x) : ℝ :=
  sqrt (2 * (1 + sqrt (1 + ( (x^4 - 1) / (2 * x^2) )^2)))

-- Define the expected simplified expression
noncomputable def expected_expression (x : ℝ) (hx : 0 < x) : ℝ :=
  (x^2 + 1) / x

-- Proof statement to verify the simplification
theorem simplify_expression (x : ℝ) (hx : 0 < x) :
  given_expression x hx = expected_expression x hx :=
sorry

end simplify_expression_l2077_207721


namespace remaining_calories_proof_l2077_207717

def volume_of_rectangular_block (length width height : ℝ) : ℝ :=
  length * width * height

def volume_of_cube (side : ℝ) : ℝ :=
  side * side * side

def remaining_volume (initial_volume eaten_volume : ℝ) : ℝ :=
  initial_volume - eaten_volume

def remaining_calories (remaining_volume calorie_density : ℝ) : ℝ :=
  remaining_volume * calorie_density

theorem remaining_calories_proof :
  let calorie_density := 110
  let original_length := 4
  let original_width := 8
  let original_height := 2
  let cube_side := 2
  let original_volume := volume_of_rectangular_block original_length original_width original_height
  let eaten_volume := volume_of_cube cube_side
  let remaining_vol := remaining_volume original_volume eaten_volume
  let resulting_calories := remaining_calories remaining_vol calorie_density
  resulting_calories = 6160 := by
  repeat { sorry }

end remaining_calories_proof_l2077_207717


namespace jimmy_paid_total_l2077_207774

-- Data for the problem
def pizza_cost : ℕ := 12
def delivery_charge : ℕ := 2
def park_distance : ℕ := 100
def park_pizzas : ℕ := 3
def building_distance : ℕ := 2000
def building_pizzas : ℕ := 2
def house_distance : ℕ := 800
def house_pizzas : ℕ := 4
def community_center_distance : ℕ := 1500
def community_center_pizzas : ℕ := 5
def office_distance : ℕ := 300
def office_pizzas : ℕ := 1
def bus_stop_distance : ℕ := 1200
def bus_stop_pizzas : ℕ := 3

def cost (distance pizzas : ℕ) : ℕ := 
  let base_cost := pizzas * pizza_cost
  if distance > 1000 then base_cost + delivery_charge else base_cost

def total_cost : ℕ :=
  cost park_distance park_pizzas +
  cost building_distance building_pizzas +
  cost house_distance house_pizzas +
  cost community_center_distance community_center_pizzas +
  cost office_distance office_pizzas +
  cost bus_stop_distance bus_stop_pizzas

theorem jimmy_paid_total : total_cost = 222 :=
  by
    -- Proof omitted
    sorry

end jimmy_paid_total_l2077_207774


namespace original_ribbon_length_l2077_207777

theorem original_ribbon_length :
  ∃ x : ℝ, 
    (∀ a b : ℝ, 
       a = x - 18 ∧ 
       b = x - 12 ∧ 
       b = 2 * a → x = 24) :=
by
  sorry

end original_ribbon_length_l2077_207777


namespace households_used_both_brands_l2077_207794

theorem households_used_both_brands 
  (total_households : ℕ)
  (neither_AB : ℕ)
  (only_A : ℕ)
  (h3 : ∀ (both : ℕ), ∃ (only_B : ℕ), only_B = 3 * both)
  (h_sum : ∀ (both : ℕ), neither_AB + only_A + both + (3 * both) = total_households) :
  ∃ (both : ℕ), both = 10 :=
by 
  sorry

end households_used_both_brands_l2077_207794


namespace smallest_prime_sum_l2077_207737

open Nat

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_sum_of_distinct_primes (n k : ℕ) (s : List ℕ) : Prop :=
  s.length = k ∧ (∀ x ∈ s, is_prime x) ∧ (∀ (x y : ℕ), x ≠ y → x ∈ s → y ∈ s → x ≠ y) ∧ s.sum = n

theorem smallest_prime_sum :
  (is_prime 61) ∧ 
  (∃ s2, is_sum_of_distinct_primes 61 2 s2) ∧ 
  (∃ s3, is_sum_of_distinct_primes 61 3 s3) ∧ 
  (∃ s4, is_sum_of_distinct_primes 61 4 s4) ∧ 
  (∃ s5, is_sum_of_distinct_primes 61 5 s5) ∧ 
  (∃ s6, is_sum_of_distinct_primes 61 6 s6) :=
by
  sorry

end smallest_prime_sum_l2077_207737


namespace rectangle_length_to_width_ratio_l2077_207744

variables (s : ℝ)

-- Given conditions
def small_square_side := s
def large_square_side := 3 * s
def rectangle_length := large_square_side
def rectangle_width := large_square_side - 2 * small_square_side

-- Theorem to prove the ratio of the length to the width of the rectangle
theorem rectangle_length_to_width_ratio : 
  ∃ (r : ℝ), r = rectangle_length s / rectangle_width s ∧ r = 3 := 
by
  sorry

end rectangle_length_to_width_ratio_l2077_207744


namespace count_edge_cubes_l2077_207716

/-- 
A cube is painted red on all faces and then cut into 27 equal smaller cubes.
Prove that the number of smaller cubes that are painted on only 2 faces is 12. 
-/
theorem count_edge_cubes (c : ℕ) (inner : ℕ)  (edge : ℕ) (face : ℕ) :
  (c = 27 ∧ inner = 1 ∧ edge = 12 ∧ face = 6) → edge = 12 :=
by
  -- Given the conditions from the problem statement
  sorry

end count_edge_cubes_l2077_207716


namespace school_starts_at_8_l2077_207749

def minutes_to_time (minutes : ℕ) : ℕ × ℕ :=
  let hour := minutes / 60
  let minute := minutes % 60
  (hour, minute)

def add_minutes_to_time (h : ℕ) (m : ℕ) (added_minutes : ℕ) : ℕ × ℕ :=
  minutes_to_time ((h * 60 + m) + added_minutes)

def subtract_minutes_from_time (h : ℕ) (m : ℕ) (subtracted_minutes : ℕ) : ℕ × ℕ :=
  minutes_to_time ((h * 60 + m) - subtracted_minutes)

theorem school_starts_at_8 : True := by
  let normal_commute := 30
  let red_light_stops := 3 * 4
  let construction_delay := 10
  let total_additional_time := red_light_stops + construction_delay
  let total_commute_time := normal_commute + total_additional_time
  let depart_time := (7, 15)
  let arrival_time := add_minutes_to_time depart_time.1 depart_time.2 total_commute_time
  let start_time := subtract_minutes_from_time arrival_time.1 arrival_time.2 7

  have : start_time = (8, 0) := by
    sorry

  exact trivial

end school_starts_at_8_l2077_207749


namespace math_proof_problem_l2077_207704

noncomputable def problem_statement : Prop :=
  ∃ (x : ℝ), (x > 12) ∧ ((x - 5) / 12 = 5 / (x - 12)) ∧ (x = 17)

theorem math_proof_problem : problem_statement :=
by
  sorry

end math_proof_problem_l2077_207704


namespace union_of_A_and_B_l2077_207763

def A : Set ℕ := {1, 2, 4}
def B : Set ℕ := {2, 4, 5}

theorem union_of_A_and_B : A ∪ B = {1, 2, 4, 5} := 
by
  sorry

end union_of_A_and_B_l2077_207763


namespace linear_function_no_fourth_quadrant_l2077_207725

theorem linear_function_no_fourth_quadrant (k : ℝ) (hk : k > 2) : 
  ∀ x (hx : x > 0), (k-2) * x + k ≥ 0 :=
by
  sorry

end linear_function_no_fourth_quadrant_l2077_207725


namespace range_of_a_l2077_207726

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, x < a → x ^ 2 > 1 ∧ ¬(x ^ 2 > 1 → x < a)) : a ≤ -1 :=
sorry

end range_of_a_l2077_207726


namespace edric_hours_per_day_l2077_207730

/--
Edric's monthly salary is $576. He works 6 days a week for 4 weeks in a month and 
his hourly rate is $3. Prove that Edric works 8 hours in a day.
-/
theorem edric_hours_per_day (m : ℕ) (r : ℕ) (d : ℕ) (w : ℕ)
  (h_m : m = 576) (h_r : r = 3) (h_d : d = 6) (h_w : w = 4) :
  (m / r) / (d * w) = 8 := by
    sorry

end edric_hours_per_day_l2077_207730


namespace proof_x_y_l2077_207743

noncomputable def x_y_problem (x y : ℝ) : Prop :=
  (x^2 = 9) ∧ (|y| = 4) ∧ (x < y) → (x - y = -1 ∨ x - y = -7)

theorem proof_x_y (x y : ℝ) : x_y_problem x y :=
by
  sorry

end proof_x_y_l2077_207743


namespace tangent_line_eqn_l2077_207757

theorem tangent_line_eqn (r x0 y0 : ℝ) (h : x0^2 + y0^2 = r^2) : 
  ∃ a b c : ℝ, a = x0 ∧ b = y0 ∧ c = r^2 ∧ (a*x + b*y = c) :=
sorry

end tangent_line_eqn_l2077_207757


namespace bucket_weight_full_l2077_207788

theorem bucket_weight_full (c d : ℝ) (x y : ℝ) 
  (h1 : x + (1 / 3) * y = c) 
  (h2 : x + (3 / 4) * y = d) : 
  x + y = (-3 * c + 8 * d) / 5 :=
sorry

end bucket_weight_full_l2077_207788


namespace sum_reciprocals_geom_seq_l2077_207770

theorem sum_reciprocals_geom_seq (a₁ q : ℝ) (h_pos_a₁ : 0 < a₁) (h_pos_q : 0 < q)
    (h_sum : a₁ + a₁ * q + a₁ * q^2 + a₁ * q^3 = 9)
    (h_prod : a₁^4 * q^6 = 81 / 4) :
    (1 / a₁) + (1 / (a₁ * q)) + (1 / (a₁ * q^2)) + (1 / (a₁ * q^3)) = 2 :=
by
  sorry

end sum_reciprocals_geom_seq_l2077_207770


namespace probability_at_least_two_boys_one_girl_l2077_207776

-- Define what constitutes a family of four children
def family := {s : Fin 4 → Bool // ∃ (b g : Fin 4), b ≠ g}

-- Define the probability equation
noncomputable def probability_of_boy_or_girl : ℚ := 1 / 2

-- Define what it means to have at least two boys and one girl
def at_least_two_boys_one_girl (f : family) : Prop :=
  ∃ (count_boys count_girls : ℕ), count_boys + count_girls = 4 
  ∧ count_boys ≥ 2 
  ∧ count_girls ≥ 1

-- Calculate the probability
theorem probability_at_least_two_boys_one_girl : 
  (∃ (f : family), at_least_two_boys_one_girl f) →
  probability_of_boy_or_girl ^ 4 * ( (6 / 16 : ℚ) + (4 / 16 : ℚ) + (1 / 16 : ℚ) ) = 11 / 16 :=
by
  sorry

end probability_at_least_two_boys_one_girl_l2077_207776


namespace number_of_results_l2077_207734

theorem number_of_results (n : ℕ)
  (avg_all : (summation : ℤ) → summation / n = 42)
  (avg_first_5 : (sum_first_5 : ℤ) → sum_first_5 / 5 = 49)
  (avg_last_7 : (sum_last_7 : ℤ) → sum_last_7 / 7 = 52)
  (fifth_result : (r5 : ℤ) → r5 = 147) :
  n = 11 :=
by
  -- Conditions
  let sum_first_5 := 5 * 49
  let sum_last_7 := 7 * 52
  let summed_results := sum_first_5 + sum_last_7 - 147
  let sum_all := 42 * n 
  -- Since sum of all results = 42n
  exact sorry

end number_of_results_l2077_207734


namespace line_intersection_l2077_207789

noncomputable def line1 (t : ℚ) : ℚ × ℚ := (1 - 2 * t, 4 + 3 * t)
noncomputable def line2 (u : ℚ) : ℚ × ℚ := (5 + u, 2 + 6 * u)

theorem line_intersection :
  ∃ t u : ℚ, line1 t = (21 / 5, -4 / 5) ∧ line2 u = (21 / 5, -4 / 5) :=
sorry

end line_intersection_l2077_207789


namespace basketball_problem_l2077_207755

theorem basketball_problem :
  ∃ x y : ℕ, (3 + x + y = 14) ∧ (3 * 3 + 2 * x + y = 28) ∧ (x = 8) ∧ (y = 3) :=
by
  sorry

end basketball_problem_l2077_207755


namespace equation_equivalence_and_rst_l2077_207710

theorem equation_equivalence_and_rst 
  (a x y c : ℝ) 
  (r s t : ℤ) 
  (h1 : r = 3) 
  (h2 : s = 1) 
  (h3 : t = 5)
  (h_eq1 : a^8 * x * y - a^7 * y - a^6 * x = a^5 * (c^5 - 1)) :
  (a^r * x - a^s) * (a^t * y - a^3) = a^5 * c^5 ∧ r * s * t = 15 :=
by
  sorry

end equation_equivalence_and_rst_l2077_207710


namespace luigi_pizza_cost_l2077_207752

theorem luigi_pizza_cost (num_pizzas pieces_per_pizza cost_per_piece : ℕ) 
  (h1 : num_pizzas = 4) 
  (h2 : pieces_per_pizza = 5) 
  (h3 : cost_per_piece = 4) :
  num_pizzas * pieces_per_pizza * cost_per_piece / pieces_per_pizza = 80 := by
  sorry

end luigi_pizza_cost_l2077_207752


namespace solve_equation_l2077_207728

theorem solve_equation (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ -2) (h3 : x ≠ -1) :
  -x^2 = (4 * x + 2) / (x^2 + 3 * x + 2) ↔ x = -1 :=
by
  sorry

end solve_equation_l2077_207728


namespace cone_volume_divided_by_pi_l2077_207782

noncomputable def arc_length (r : ℝ) (θ : ℝ) : ℝ := (θ / 360) * 2 * Real.pi * r

noncomputable def sector_to_cone_radius (arc_len : ℝ) : ℝ := arc_len / (2 * Real.pi)

noncomputable def cone_height (r_base : ℝ) (slant_height : ℝ) : ℝ :=
  Real.sqrt (slant_height ^ 2 - r_base ^ 2)

noncomputable def cone_volume (r_base : ℝ) (height : ℝ) : ℝ :=
  (1 / 3) * Real.pi * r_base ^ 2 * height

theorem cone_volume_divided_by_pi (r slant_height θ : ℝ) (h : slant_height = 15 ∧ θ = 270):
  cone_volume (sector_to_cone_radius (arc_length r θ)) (cone_height (sector_to_cone_radius (arc_length r θ)) slant_height) / Real.pi = (453.515625 * Real.sqrt 10.9375) :=
by
  sorry

end cone_volume_divided_by_pi_l2077_207782


namespace verify_final_weights_l2077_207701

-- Define the initial weights
def initial_bench_press : ℝ := 500
def initial_squat : ℝ := 400
def initial_deadlift : ℝ := 600

-- Define the weight adjustment transformations for each exercise
def transform_bench_press (w : ℝ) : ℝ :=
  let w1 := w * 0.20
  let w2 := w1 * 1.60
  let w3 := w2 * 0.80
  let w4 := w3 * 3
  w4

def transform_squat (w : ℝ) : ℝ :=
  let w1 := w * 0.50
  let w2 := w1 * 1.40
  let w3 := w2 * 2
  w3

def transform_deadlift (w : ℝ) : ℝ :=
  let w1 := w * 0.70
  let w2 := w1 * 1.80
  let w3 := w2 * 0.60
  let w4 := w3 * 1.50
  w4

-- The final calculated weights for verification
def final_bench_press : ℝ := 384
def final_squat : ℝ := 560
def final_deadlift : ℝ := 680.4

-- Statement of the problem: prove that the transformed weights are as calculated
theorem verify_final_weights : 
  transform_bench_press initial_bench_press = final_bench_press ∧ 
  transform_squat initial_squat = final_squat ∧ 
  transform_deadlift initial_deadlift = final_deadlift := 
by 
  sorry

end verify_final_weights_l2077_207701


namespace equilateral_triangle_ratio_l2077_207715

theorem equilateral_triangle_ratio (s : ℝ) (h : s = 6) :
  let perimeter := 3 * s
  let area := (s * s * Real.sqrt 3) / 4
  perimeter / area = (2 * Real.sqrt 3) / 3 :=
by
  sorry

end equilateral_triangle_ratio_l2077_207715


namespace sequence_property_exists_l2077_207784

theorem sequence_property_exists :
  ∃ a₁ a₂ a₃ a₄ : ℝ, 
  a₂ - a₁ = a₃ - a₂ ∧ a₃ - a₂ = a₄ - a₃ ∧
  (a₃ / a₁ = a₄ / a₃) ∧ ∃ r : ℝ, r ≠ 0 ∧ a₁ = -4 * r ∧ a₂ = -3 * r ∧ a₃ = -2 * r ∧ a₄ = -r :=
by
  sorry

end sequence_property_exists_l2077_207784


namespace find_f_2015_l2077_207718

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f_periodic_2 : ∀ x : ℝ, f x * f (x + 2) = 13
axiom f_at_1 : f 1 = 2

theorem find_f_2015 : f 2015 = 13 / 2 :=
by
  sorry

end find_f_2015_l2077_207718


namespace find_S5_l2077_207762

-- Assuming the sequence is geometric and defining the conditions
variables {a : ℕ → ℝ} {q : ℝ}

-- Definitions of the conditions based on the problem
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n+1) = a n * q

def condition_1 (a : ℕ → ℝ) (q : ℝ) : Prop :=
  a 2 * a 5 = 3 * a 3

def condition_2 (a : ℕ → ℝ) (q : ℝ) : Prop :=
  (a 4 + 9 * a 7) / 2 = 2

-- Sum of the first n terms of a geometric sequence
noncomputable def S_n (a : ℕ → ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a 0 * (1 - q ^ n) / (1 - q)

-- The theorem stating the final goal
theorem find_S5 (a : ℕ → ℝ) (q : ℝ) (h_seq : geometric_sequence a q) 
    (h1 : condition_1 a q) (h2 : condition_2 a q) : S_n a q 5 = 121 :=
by
  -- This adds "sorry" to bypass the actual proof
  sorry

end find_S5_l2077_207762


namespace average_score_bounds_l2077_207767

/-- Problem data definitions -/
def n_100 : ℕ := 2
def n_90_99 : ℕ := 9
def n_80_89 : ℕ := 17
def n_70_79 : ℕ := 28
def n_60_69 : ℕ := 36
def n_50_59 : ℕ := 7
def n_48 : ℕ := 1

def sum_scores_min : ℕ := (100 * n_100 + 90 * n_90_99 + 80 * n_80_89 + 70 * n_70_79 + 60 * n_60_69 + 50 * n_50_59 + 48)
def sum_scores_max : ℕ := (100 * n_100 + 99 * n_90_99 + 89 * n_80_89 + 79 * n_70_79 + 69 * n_60_69 + 59 * n_50_59 + 48)
def total_people : ℕ := n_100 + n_90_99 + n_80_89 + n_70_79 + n_60_69 + n_50_59 + n_48

/-- Prove the minimum and maximum average scores. -/
theorem average_score_bounds :
  (sum_scores_min / total_people : ℚ) = 68.88 ∧
  (sum_scores_max / total_people : ℚ) = 77.61 :=
by
  sorry

end average_score_bounds_l2077_207767


namespace simplify_power_multiplication_l2077_207790

theorem simplify_power_multiplication (x : ℝ) : (-x) ^ 3 * (-x) ^ 2 = -x ^ 5 :=
by sorry

end simplify_power_multiplication_l2077_207790


namespace largest_fraction_l2077_207795

theorem largest_fraction :
  let A := (5 : ℚ) / 11
  let B := (6 : ℚ) / 13
  let C := (18 : ℚ) / 37
  let D := (101 : ℚ) / 202
  let E := (200 : ℚ) / 399
  E > A ∧ E > B ∧ E > C ∧ E > D := by
  sorry

end largest_fraction_l2077_207795


namespace find_abcd_abs_eq_one_l2077_207711

noncomputable def non_zero_real (r : ℝ) := r ≠ 0

theorem find_abcd_abs_eq_one
  (a b c d : ℝ)
  (h1 : a ≠ 0)
  (h2 : b ≠ 0)
  (h3 : c ≠ 0)
  (h4 : d ≠ 0)
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_eq : a^2 + (1/b) = b^2 + (1/c) ∧ b^2 + (1/c) = c^2 + (1/d) ∧ c^2 + (1/d) = d^2 + (1/a)) :
  |a * b * c * d| = 1 :=
sorry

end find_abcd_abs_eq_one_l2077_207711


namespace find_z_l2077_207729

-- Definitions from the problem statement
variables {x y z : ℤ}
axiom consecutive (h1: x = z + 2) (h2: y = z + 1) : true
axiom ordered (h3: x > y) (h4: y > z) : true
axiom equation (h5: 2 * x + 3 * y + 3 * z = 5 * y + 8) : true

-- The proof goal
theorem find_z (h1: x = z + 2) (h2: y = z + 1) (h3: x > y) (h4: y > z) (h5: 2 * x + 3 * y + 3 * z = 5 * y + 8) : z = 2 :=
by 
  sorry

end find_z_l2077_207729


namespace student_marks_problem_l2077_207760

-- Define the variables
variables (M P C X : ℕ)

-- State the conditions
-- Condition 1: M + P = 70
def condition1 : Prop := M + P = 70

-- Condition 2: C = P + X
def condition2 : Prop := C = P + X

-- Condition 3: (M + C) / 2 = 45
def condition3 : Prop := (M + C) / 2 = 45

-- The theorem stating the problem
theorem student_marks_problem (h1 : condition1 M P) (h2 : condition2 C P X) (h3 : condition3 M C) : X = 20 :=
by sorry

end student_marks_problem_l2077_207760


namespace remainder_when_doubling_l2077_207761

theorem remainder_when_doubling:
  ∀ (n k : ℤ), n = 30 * k + 16 → (2 * n) % 15 = 2 :=
by
  intros n k h
  sorry

end remainder_when_doubling_l2077_207761


namespace problem_l2077_207775

def S (n : ℕ) : ℤ := n^2 - 4 * n + 1

def a (n : ℕ) : ℤ :=
  if n = 1 then S 1 else S n - S (n - 1)

def sum_abs_a_10 : ℤ :=
  (|a 1| + |a 2| + |a 3| + |a 4| + |a 5| + |a 6| + |a 7| + |a 8| + |a 9| + |a 10|)

theorem problem : sum_abs_a_10 = 67 := by
  sorry

end problem_l2077_207775


namespace prove_seq_properties_l2077_207714

theorem prove_seq_properties (a b : ℕ → ℕ) (S T : ℕ → ℕ) (h_increasing : ∀ n, a n < a (n + 1))
  (h_sum : ∀ n, 2 * S n = a n ^ 2 + n)
  (h_b : ∀ n, b n = a (n + 1) * 2 ^ n)
  : (∀ n, a n = n) ∧ (∀ n, T n = n * 2 ^ (n + 1)) :=
sorry

end prove_seq_properties_l2077_207714


namespace fraction_cubed_equality_l2077_207742

-- Constants for the problem
def A : ℝ := 81000
def B : ℝ := 9000

-- Problem statement
theorem fraction_cubed_equality : (A^3) / (B^3) = 729 :=
by
  sorry

end fraction_cubed_equality_l2077_207742


namespace remainder_123456789012_div_252_l2077_207765

theorem remainder_123456789012_div_252 :
  (∃ x : ℕ, 123456789012 % 252 = x ∧ x = 204) :=
sorry

end remainder_123456789012_div_252_l2077_207765


namespace sum_two_numbers_in_AP_and_GP_equals_20_l2077_207723

theorem sum_two_numbers_in_AP_and_GP_equals_20 :
  ∃ a b : ℝ, 
    (a > 0) ∧ (b > 0) ∧ 
    (4 < a) ∧ (a < b) ∧ 
    (4 + (a - 4) = a) ∧ (4 + 2 * (a - 4) = b) ∧
    (a * (b / a) = b) ∧ (b * (b / a) = 16) ∧ 
    a + b = 20 :=
by
  sorry

end sum_two_numbers_in_AP_and_GP_equals_20_l2077_207723


namespace little_john_spent_on_sweets_l2077_207722

theorem little_john_spent_on_sweets:
  let initial_amount := 10.10
  let amount_given_to_each_friend := 2.20
  let amount_left := 2.45
  let total_given_to_friends := 2 * amount_given_to_each_friend
  let amount_before_sweets := initial_amount - total_given_to_friends
  let amount_spent_on_sweets := amount_before_sweets - amount_left
  amount_spent_on_sweets = 3.25 :=
by
  sorry

end little_john_spent_on_sweets_l2077_207722


namespace positive_difference_of_two_numbers_l2077_207739

theorem positive_difference_of_two_numbers 
  (a b : ℝ) 
  (h₁ : a + b = 10) 
  (h₂ : a^2 - b^2 = 40) : 
  |a - b| = 4 := 
sorry

end positive_difference_of_two_numbers_l2077_207739


namespace remainder_3012_div_96_l2077_207750

theorem remainder_3012_div_96 : 3012 % 96 = 36 :=
by 
  sorry

end remainder_3012_div_96_l2077_207750


namespace points_on_planes_of_cubic_eq_points_on_planes_of_quintic_eq_l2077_207756

-- Problem 1: Prove that if \(x^3 + y^3 + z^3 = (x + y + z)^3\), the points lie on the planes \(x + y = 0\), \(y + z = 0\), \(z + x = 0\).
theorem points_on_planes_of_cubic_eq (x y z : ℝ) (h : x^3 + y^3 + z^3 = (x + y + z)^3) :
  x + y = 0 ∨ y + z = 0 ∨ z + x = 0 :=
sorry

-- Problem 2: Prove that if \(x^5 + y^5 + z^5 = (x + y + z)^5\), the points lie on the planes \(x + y = 0\), \(y + z = 0\), \(z + x = 0\).
theorem points_on_planes_of_quintic_eq (x y z : ℝ) (h : x^5 + y^5 + z^5 = (x + y + z)^5) :
  x + y = 0 ∨ y + z = 0 ∨ z + x = 0 :=
sorry

end points_on_planes_of_cubic_eq_points_on_planes_of_quintic_eq_l2077_207756


namespace Jim_paycheck_correct_l2077_207741

noncomputable def Jim_paycheck_after_deductions (gross_pay : ℝ) (retirement_percentage : ℝ) (tax_deduction : ℝ) : ℝ :=
  gross_pay - (gross_pay * retirement_percentage) - tax_deduction

theorem Jim_paycheck_correct :
  Jim_paycheck_after_deductions 1120 0.25 100 = 740 :=
by sorry

end Jim_paycheck_correct_l2077_207741


namespace line_quadrant_conditions_l2077_207712

theorem line_quadrant_conditions (k b : ℝ) 
  (H1 : ∃ x : ℝ, x > 0 ∧ k * x + b > 0)
  (H3 : ∃ x : ℝ, x < 0 ∧ k * x + b < 0)
  (H4 : ∃ x : ℝ, x > 0 ∧ k * x + b < 0) : k > 0 ∧ b < 0 :=
sorry

end line_quadrant_conditions_l2077_207712


namespace largest_constant_l2077_207745

def equation_constant (c d : ℝ) : ℝ :=
  5 * c + (d - 12)^2

theorem largest_constant : ∃ constant : ℝ, (∀ c, c ≤ 47) → (∀ d, equation_constant 47 d = constant) → constant = 235 := 
by
  sorry

end largest_constant_l2077_207745


namespace solve_polynomial_relation_l2077_207771

--Given Conditions
def polynomial_relation (x y : ℤ) : Prop := y^3 = x^3 + 8 * x^2 - 6 * x + 8 

--Proof Problem
theorem solve_polynomial_relation : ∃ (x y : ℤ), (polynomial_relation x y) ∧ 
  ((y = 11 ∧ x = 9) ∨ (y = 2 ∧ x = 0)) :=
by 
  sorry

end solve_polynomial_relation_l2077_207771


namespace arcsin_one_half_eq_pi_six_l2077_207764

theorem arcsin_one_half_eq_pi_six : Real.arcsin (1 / 2) = π / 6 :=
by
  sorry

end arcsin_one_half_eq_pi_six_l2077_207764


namespace ratio_Mandy_to_Pamela_l2077_207751

-- Definitions based on conditions in the problem
def exam_items : ℕ := 100
def Lowella_correct : ℕ := (35 * exam_items) / 100  -- 35% of 100
def Pamela_correct : ℕ := Lowella_correct + (20 * Lowella_correct) / 100 -- 20% more than Lowella
def Mandy_score : ℕ := 84

-- The proof problem statement
theorem ratio_Mandy_to_Pamela : Mandy_score / Pamela_correct = 2 := by
  sorry

end ratio_Mandy_to_Pamela_l2077_207751


namespace part_a_l2077_207778

theorem part_a (p : ℕ → ℕ → ℝ) (m : ℕ) (hm : m ≥ 1) : p m 0 = (3 / 4) * p (m-1) 0 + (1 / 2) * p (m-1) 2 + (1 / 8) * p (m-1) 4 :=
by
  sorry

end part_a_l2077_207778


namespace tan_double_angle_l2077_207780

theorem tan_double_angle (α : ℝ) (h : Real.tan α = 1 / 3) : Real.tan (2 * α) = 3 / 4 := 
by
  sorry

end tan_double_angle_l2077_207780


namespace total_length_figure2_l2077_207769

-- Define the initial lengths of each segment in Figure 1.
def initial_length_horizontal1 := 5
def initial_length_vertical1 := 10
def initial_length_horizontal2 := 4
def initial_length_vertical2 := 3
def initial_length_horizontal3 := 3
def initial_length_vertical3 := 5
def initial_length_horizontal4 := 4
def initial_length_vertical_sum := 10 + 3 + 5

-- Define the transformations.
def bottom_length := initial_length_horizontal1
def rightmost_vertical_length := initial_length_vertical1 - 2
def top_horizontal_length := initial_length_horizontal2 - 3
def leftmost_vertical_length := initial_length_vertical1

-- Define the total length in Figure 2 as a theorem to be proved.
theorem total_length_figure2:
  bottom_length + rightmost_vertical_length + top_horizontal_length + leftmost_vertical_length = 24 := by
  sorry

end total_length_figure2_l2077_207769


namespace range_of_f_when_a_eq_2_max_value_implies_a_l2077_207735

-- first part
theorem range_of_f_when_a_eq_2 (x : ℝ) (hx : -2 ≤ x ∧ x ≤ 3) :
  (∀ y, (y = x^2 + 3*x - 3) → (y ≥ -21/4 ∧ y ≤ 15)) :=
by sorry

-- second part
theorem max_value_implies_a (a : ℝ) (hx : ∀ x, -1 ≤ x ∧ x ≤ 3 → x^2 + (2*a - 1)*x - 3 ≤ 1) :
  a = -1 ∨ a = -1 / 3 :=
by sorry

end range_of_f_when_a_eq_2_max_value_implies_a_l2077_207735


namespace f_zero_eq_one_f_positive_f_increasing_f_range_x_l2077_207754

noncomputable def f : ℝ → ℝ := sorry
axiom f_condition1 : f 0 ≠ 0
axiom f_condition2 : ∀ x : ℝ, x > 0 → f x > 1
axiom f_condition3 : ∀ a b : ℝ, f (a + b) = f a * f b

theorem f_zero_eq_one : f 0 = 1 :=
sorry

theorem f_positive : ∀ x : ℝ, f x > 0 :=
sorry

theorem f_increasing : ∀ x y : ℝ, x < y → f x < f y :=
sorry

theorem f_range_x (x : ℝ) (h : f x * f (2 * x - x^2) > 1) : x ∈ { x : ℝ | f x > 1 ∧ f (2 * x - x^2) > 1 } :=
sorry

end f_zero_eq_one_f_positive_f_increasing_f_range_x_l2077_207754


namespace ratio_of_boys_to_girls_l2077_207748

theorem ratio_of_boys_to_girls {T G B : ℕ} (h1 : (2/3 : ℚ) * G = (1/4 : ℚ) * T) (h2 : T = G + B) : (B : ℚ) / G = 5 / 3 :=
by
  sorry

end ratio_of_boys_to_girls_l2077_207748


namespace smallest_integer_in_ratio_l2077_207779

theorem smallest_integer_in_ratio (a b c : ℕ) 
    (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
    (h_sum : a + b + c = 100) 
    (h_ratio : c = 5 * a / 2 ∧ b = 3 * a / 2) : 
    a = 20 := 
by
  sorry

end smallest_integer_in_ratio_l2077_207779


namespace students_who_chose_water_l2077_207766

-- Defining the conditions
def percent_juice : ℚ := 75 / 100
def percent_water : ℚ := 25 / 100
def students_who_chose_juice : ℚ := 90
def ratio_water_to_juice : ℚ := percent_water / percent_juice  -- This should equal 1/3

-- The theorem we need to prove
theorem students_who_chose_water : students_who_chose_juice * ratio_water_to_juice = 30 := 
by
  sorry

end students_who_chose_water_l2077_207766


namespace find_number_l2077_207724

theorem find_number (N : ℝ) (h : 0.15 * 0.30 * 0.50 * N = 108) : N = 4800 :=
by
  sorry

end find_number_l2077_207724


namespace remaining_bananas_l2077_207753

def original_bananas : ℕ := 46
def removed_bananas : ℕ := 5

theorem remaining_bananas : original_bananas - removed_bananas = 41 := by
  sorry

end remaining_bananas_l2077_207753


namespace roots_squared_sum_l2077_207702

theorem roots_squared_sum (x1 x2 : ℝ) (h₁ : x1^2 - 5 * x1 + 3 = 0) (h₂ : x2^2 - 5 * x2 + 3 = 0) :
  x1^2 + x2^2 = 19 :=
by
  sorry

end roots_squared_sum_l2077_207702


namespace kim_money_l2077_207731

theorem kim_money (S P K A : ℝ) (h1 : K = 1.40 * S) (h2 : S = 0.80 * P) (h3 : A = 1.25 * (S + K)) (h4 : S + P + A = 3.60) : K = 0.96 :=
by
  sorry

end kim_money_l2077_207731


namespace radius_of_circle_l2077_207740

noncomputable def circle_radius {k : ℝ} (hk : k > -6) : ℝ := 6 * Real.sqrt 2 + 6

theorem radius_of_circle (k : ℝ) (hk : k > -6)
  (tangent_y_eq_x : ∀ (P : ℝ × ℝ), P.1 = 0 → P.2 = k → dist P (0, P.2) = 6 * Real.sqrt 2 + 6)
  (tangent_y_eq_negx : ∀ (P : ℝ × ℝ), P.1 = 0 → P.2 = k → dist P (0, -P.2) = 6 * Real.sqrt 2 + 6)
  (tangent_y_eq_neg6 : ∀ (P : ℝ × ℝ), P.1 = 0 → P.2 = k → dist P (0, -6) = 6 * Real.sqrt 2 + 6) :
  circle_radius hk = 6 * Real.sqrt 2 + 6 :=
by
  sorry

end radius_of_circle_l2077_207740

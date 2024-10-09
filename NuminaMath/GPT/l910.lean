import Mathlib

namespace principal_amount_correct_l910_91012

-- Define the given conditions and quantities
def P : ℝ := 1054.76
def final_amount : ℝ := 1232.0
def rate1 : ℝ := 0.05
def rate2 : ℝ := 0.07
def rate3 : ℝ := 0.04

-- Define the statement we want to prove
theorem principal_amount_correct :
  final_amount = P * (1 + rate1) * (1 + rate2) * (1 + rate3) :=
sorry

end principal_amount_correct_l910_91012


namespace no_same_last_four_digits_of_powers_of_five_and_six_l910_91016

theorem no_same_last_four_digits_of_powers_of_five_and_six : 
  ¬ ∃ (n m : ℕ), 0 < n ∧ 0 < m ∧ (5 ^ n % 10000 = 6 ^ m % 10000) := 
by 
  sorry

end no_same_last_four_digits_of_powers_of_five_and_six_l910_91016


namespace m_range_l910_91050

/-- Given a point (x, y) on the circle x^2 + (y - 1)^2 = 2, show that the real number m,
such that x + y + m ≥ 0, must satisfy m ≥ 1. -/
theorem m_range (x y m : ℝ) (h₁ : x^2 + (y - 1)^2 = 2) (h₂ : x + y + m ≥ 0) : m ≥ 1 :=
sorry

end m_range_l910_91050


namespace number_of_red_balls_l910_91077

def total_balls : ℕ := 50
def frequency_red_ball : ℝ := 0.7

theorem number_of_red_balls :
  ∃ n : ℕ, n = (total_balls : ℝ) * frequency_red_ball ∧ n = 35 :=
by
  sorry

end number_of_red_balls_l910_91077


namespace solve_equation_l910_91017

noncomputable def equation (x : ℝ) : Prop :=
  -2 * x ^ 3 = (5 * x ^ 2 + 2) / (2 * x - 1)

theorem solve_equation (x : ℝ) :
  equation x ↔ (x = (1 + Real.sqrt 17) / 4 ∨ x = (1 - Real.sqrt 17) / 4) :=
by
  sorry

end solve_equation_l910_91017


namespace compute_f_sum_l910_91007

noncomputable def f : ℝ → ℝ := sorry -- placeholder for f(x)

variables (x : ℝ)

-- Conditions
axiom odd_function : ∀ x, f (-x) = -f x
axiom periodic_function : ∀ x, f (x + 2) = f x
axiom f_definition : ∀ x, 0 < x ∧ x < 1 → f x = x^2

-- Prove the main statement
theorem compute_f_sum : f (-3 / 2) + f 1 = 3 / 4 :=
by
  sorry

end compute_f_sum_l910_91007


namespace combined_alloy_force_l910_91047

-- Define the masses and forces exerted by Alloy A and Alloy B
def mass_A : ℝ := 6
def force_A : ℝ := 30
def mass_B : ℝ := 3
def force_B : ℝ := 10

-- Define the combined mass and force
def combined_mass : ℝ := mass_A + mass_B
def combined_force : ℝ := force_A + force_B

-- Theorem statement
theorem combined_alloy_force :
  combined_force = 40 :=
by
  -- The proof is omitted.
  sorry

end combined_alloy_force_l910_91047


namespace integer_roots_count_l910_91003

theorem integer_roots_count (b c d e f : ℚ) :
  ∃ (n : ℕ), (n = 0 ∨ n = 1 ∨ n = 2 ∨ n = 4 ∨ n = 5) ∧
  (∃ (r : ℕ → ℤ), ∀ i, i < n → (∀ z : ℤ, (∃ m, z = r m) → (z^5 + b * z^4 + c * z^3 + d * z^2 + e * z + f = 0))) :=
sorry

end integer_roots_count_l910_91003


namespace parallelogram_angle_sum_l910_91032

theorem parallelogram_angle_sum (ABCD : Type) (A B C D : ABCD) 
  (angle : ABCD → ℝ) (h_parallelogram : true) (h_B : angle B = 60) :
  ¬ (angle C + angle A = 180) :=
sorry

end parallelogram_angle_sum_l910_91032


namespace factorize_x2_add_2x_sub_3_l910_91076

theorem factorize_x2_add_2x_sub_3 :
  (x^2 + 2 * x - 3) = (x + 3) * (x - 1) :=
by
  sorry

end factorize_x2_add_2x_sub_3_l910_91076


namespace tenth_term_arithmetic_sequence_l910_91005

theorem tenth_term_arithmetic_sequence :
  let a_1 := (1 : ℝ) / 2
  let a_2 := (5 : ℝ) / 6
  let d := a_2 - a_1
  (a_1 + 9 * d) = 7 / 2 := 
by
  sorry

end tenth_term_arithmetic_sequence_l910_91005


namespace intersection_M_N_l910_91062

noncomputable def M := {x : ℕ | x < 6}
noncomputable def N := {x : ℕ | x^2 - 11 * x + 18 < 0}
noncomputable def intersection := {x : ℕ | x ∈ M ∧ x ∈ N}

theorem intersection_M_N : intersection = {3, 4, 5} := by
  sorry

end intersection_M_N_l910_91062


namespace correct_age_equation_l910_91027

variable (x : ℕ)

def age_older_brother (x : ℕ) : ℕ := 2 * x
def age_younger_brother_six_years_ago (x : ℕ) : ℕ := x - 6
def age_older_brother_six_years_ago (x : ℕ) : ℕ := 2 * x - 6

theorem correct_age_equation (h1 : age_younger_brother_six_years_ago x + age_older_brother_six_years_ago x = 15) :
  (x - 6) + (2 * x - 6) = 15 :=
by
  sorry

end correct_age_equation_l910_91027


namespace total_coins_l910_91066

-- Definitions for the conditions
def number_of_nickels := 13
def number_of_quarters := 8

-- Statement of the proof problem
theorem total_coins : number_of_nickels + number_of_quarters = 21 :=
by
  sorry

end total_coins_l910_91066


namespace helicopter_rental_cost_l910_91072

noncomputable def rentCost (hours_per_day : ℕ) (days : ℕ) (cost_per_hour : ℕ) : ℕ :=
  hours_per_day * days * cost_per_hour

theorem helicopter_rental_cost :
  rentCost 2 3 75 = 450 := 
by
  sorry

end helicopter_rental_cost_l910_91072


namespace original_number_is_600_l910_91067

theorem original_number_is_600 (x : Real) (h : x * 1.10 = 660) : x = 600 := by
  sorry

end original_number_is_600_l910_91067


namespace number_of_blue_lights_l910_91052

-- Conditions
def total_colored_lights : Nat := 95
def red_lights : Nat := 26
def yellow_lights : Nat := 37
def blue_lights : Nat := total_colored_lights - (red_lights + yellow_lights)

-- Statement we need to prove
theorem number_of_blue_lights : blue_lights = 32 := by
  sorry

end number_of_blue_lights_l910_91052


namespace solve_system_of_equations_l910_91028

theorem solve_system_of_equations (x y z : ℝ) :
  (x * y + 1 = 2 * z) →
  (y * z + 1 = 2 * x) →
  (z * x + 1 = 2 * y) →
  (x = 1 ∧ y = 1 ∧ z = 1) ∨ 
  ((x = -2 ∧ y = -2 ∧ z = 5/2) ∨
   (x = 5/2 ∧ y = -2 ∧ z = -2) ∨ 
   (x = -2 ∧ y = 5/2 ∧ z = -2)) :=
sorry

end solve_system_of_equations_l910_91028


namespace purely_imaginary_iff_x_equals_one_l910_91079

theorem purely_imaginary_iff_x_equals_one (x : ℝ) :
  ((x^2 - 1) + (x + 1) * Complex.I).re = 0 → x = 1 :=
by
  sorry

end purely_imaginary_iff_x_equals_one_l910_91079


namespace simplified_sum_l910_91030

def exp1 := -( -1 ^ 2006 )
def exp2 := -( -1 ^ 2007 )
def exp3 := -( 1 ^ 2008 )
def exp4 := -( -1 ^ 2009 )

theorem simplified_sum : 
  exp1 + exp2 + exp3 + exp4 = 0 := 
by 
  sorry

end simplified_sum_l910_91030


namespace jail_time_calculation_l910_91098

def total_arrests (arrests_per_day : ℕ) (cities : ℕ) (days : ℕ) : ℕ := 
  arrests_per_day * cities * days

def jail_time_before_trial (arrests : ℕ) (days_before_trial : ℕ) : ℕ := 
  days_before_trial * arrests

def jail_time_after_trial (arrests : ℕ) (weeks_after_trial : ℕ) : ℕ := 
  weeks_after_trial * arrests

def combined_jail_time (weeks_before_trial : ℕ) (weeks_after_trial : ℕ) : ℕ := 
  weeks_before_trial + weeks_after_trial

noncomputable def total_jail_time_in_weeks : ℕ := 
  let arrests := total_arrests 10 21 30
  let weeks_before_trial := jail_time_before_trial arrests 4 / 7
  let weeks_after_trial := jail_time_after_trial arrests 1
  combined_jail_time weeks_before_trial weeks_after_trial

theorem jail_time_calculation : 
  total_jail_time_in_weeks = 9900 :=
sorry

end jail_time_calculation_l910_91098


namespace norm_photos_l910_91096

-- Define variables for the number of photos taken by Lisa, Mike, and Norm.
variables {L M N : ℕ}

-- Define the given conditions as hypotheses.
def condition1 (L M N : ℕ) : Prop := L + M = M + N - 60
def condition2 (L N : ℕ) : Prop := N = 2 * L + 10

-- State the problem in Lean: we want to prove that the number of photos Norm took is 110.
theorem norm_photos (L M N : ℕ) (h1 : condition1 L M N) (h2 : condition2 L N) : N = 110 :=
by
  sorry

end norm_photos_l910_91096


namespace sector_perimeter_l910_91088

theorem sector_perimeter (A θ r: ℝ) (hA : A = 2) (hθ : θ = 4) (hArea : A = (1/2) * r^2 * θ) : (2 * r + r * θ) = 6 :=
by 
  sorry

end sector_perimeter_l910_91088


namespace cyclic_sum_non_negative_equality_condition_l910_91086

theorem cyclic_sum_non_negative (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^2 * (a - b) / (a + b) + b^2 * (b - c) / (b + c) + c^2 * (c - a) / (c + a) ≥ 0 :=
sorry

theorem equality_condition (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^2 * (a - b) / (a + b) + b^2 * (b - c) / (b + c) + c^2 * (c - a) / (c + a) = 0 ↔ a = b ∧ b = c :=
sorry

end cyclic_sum_non_negative_equality_condition_l910_91086


namespace distance_they_both_run_l910_91092

theorem distance_they_both_run
  (D : ℝ)
  (A_time : D / 28 = A_speed)
  (B_time : D / 32 = B_speed)
  (A_beats_B : A_speed * 28 = B_speed * 28 + 16) :
  D = 128 := 
sorry

end distance_they_both_run_l910_91092


namespace find_n_l910_91093

theorem find_n (n : ℕ) (h : (17 + 98 + 39 + 54 + n) / 5 = n) : n = 52 :=
by
  sorry

end find_n_l910_91093


namespace contrapositive_even_contrapositive_not_even_l910_91039

theorem contrapositive_even (x y : ℤ) : 
  (∃ a b : ℤ, x = 2*a ∧ y = 2*b)  → (∃ c : ℤ, x + y = 2*c) :=
sorry

theorem contrapositive_not_even (x y : ℤ) :
  (¬ ∃ c : ℤ, x + y = 2*c) → (¬ ∃ a b : ℤ, x = 2*a ∧ y = 2*b) :=
sorry

end contrapositive_even_contrapositive_not_even_l910_91039


namespace new_average_production_l910_91013

theorem new_average_production (n : ℕ) (daily_avg : ℕ) (today_prod : ℕ) (new_avg : ℕ) 
  (h1 : daily_avg = 50) 
  (h2 : today_prod = 95) 
  (h3 : n = 8) 
  (h4 : new_avg = (daily_avg * n + today_prod) / (n + 1)) : 
  new_avg = 55 := 
sorry

end new_average_production_l910_91013


namespace jordan_travel_distance_heavy_traffic_l910_91073

theorem jordan_travel_distance_heavy_traffic (x : ℝ) (h1 : x / 20 + x / 10 + x / 6 = 7 / 6) : 
  x = 3.7 :=
by
  sorry

end jordan_travel_distance_heavy_traffic_l910_91073


namespace faye_money_left_is_30_l910_91040

-- Definitions and conditions
def initial_money : ℝ := 20
def mother_gave (initial : ℝ) : ℝ := 2 * initial
def cost_of_cupcakes : ℝ := 10 * 1.5
def cost_of_cookies : ℝ := 5 * 3

-- Calculate the total money Faye has left
def total_money_left (initial : ℝ) (mother_gave_ : ℝ) (cost_cupcakes : ℝ) (cost_cookies : ℝ) : ℝ :=
  initial + mother_gave_ - (cost_cupcakes + cost_cookies)

-- Theorem stating the money left
theorem faye_money_left_is_30 :
  total_money_left initial_money (mother_gave initial_money) cost_of_cupcakes cost_of_cookies = 30 :=
by sorry

end faye_money_left_is_30_l910_91040


namespace min_sum_sequence_n_l910_91006

theorem min_sum_sequence_n (S : ℕ → ℤ) (h : ∀ n, S n = n * n - 48 * n) : 
  ∃ n, n = 24 ∧ ∀ m, S n ≤ S m :=
by
  sorry

end min_sum_sequence_n_l910_91006


namespace raj_is_older_than_ravi_l910_91089

theorem raj_is_older_than_ravi
  (R V H L x : ℕ)
  (h1 : R = V + x)
  (h2 : H = V - 2)
  (h3 : R = 3 * L)
  (h4 : H * 2 = 3 * L)
  (h5 : 20 = (4 * H) / 3) :
  x = 13 :=
by
  sorry

end raj_is_older_than_ravi_l910_91089


namespace a_plus_b_l910_91075

-- Definitions and conditions
def f (x : ℝ) (a b : ℝ) := a * x + b
def g (x : ℝ) := 3 * x - 7

theorem a_plus_b (a b : ℝ) (h : ∀ x : ℝ, g (f x a b) = 4 * x + 5) : a + b = 16 / 3 :=
by
  sorry

end a_plus_b_l910_91075


namespace sum_of_distinct_integers_eq_36_l910_91025

theorem sum_of_distinct_integers_eq_36
  (p q r s t : ℤ)
  (hpq : p ≠ q) (hpr : p ≠ r) (hps : p ≠ s) (hpt : p ≠ t)
  (hqr : q ≠ r) (hqs : q ≠ s) (hqt : q ≠ t)
  (hrs : r ≠ s) (hrt : r ≠ t)
  (hst : s ≠ t)
  (h : (8 - p) * (8 - q) * (8 - r) * (8 - s) * (8 - t) = 80) :
  p + q + r + s + t = 36 :=
by
  sorry

end sum_of_distinct_integers_eq_36_l910_91025


namespace grid_points_circumference_l910_91031

def numGridPointsOnCircumference (R : ℝ) : ℕ := sorry

def isInteger (x : ℝ) : Prop := ∃ (n : ℤ), x = n

theorem grid_points_circumference (R : ℝ) (h : numGridPointsOnCircumference R = 1988) : 
  isInteger R ∨ isInteger (Real.sqrt 2 * R) :=
by
  sorry

end grid_points_circumference_l910_91031


namespace max_empty_squares_l910_91087

theorem max_empty_squares (board_size : ℕ) (total_cells : ℕ) 
  (initial_cockroaches : ℕ) (adjacent : ℕ → ℕ → Prop) 
  (different : ℕ → ℕ → Prop) :
  board_size = 8 → total_cells = 64 → initial_cockroaches = 2 →
  (∀ s : ℕ, s < total_cells → ∃ s1 s2 : ℕ, adjacent s s1 ∧ 
              adjacent s s2 ∧ 
              different s1 s2) →
  ∃ max_empty_cells : ℕ, max_empty_cells = 24 :=
by
  intros h_board_size h_total_cells h_initial_cockroaches h_moves
  sorry

end max_empty_squares_l910_91087


namespace minimum_max_abs_x2_sub_2xy_l910_91021

theorem minimum_max_abs_x2_sub_2xy {y : ℝ} :
  ∃ y : ℝ, (∀ x ∈ (Set.Icc 0 1), abs (x^2 - 2*x*y) ≥ 0) ∧
           (∀ y' ∈ Set.univ, (∀ x ∈ (Set.Icc 0 1), abs (x^2 - 2*x*y') ≥ abs (x^2 - 2*x*y))) :=
sorry

end minimum_max_abs_x2_sub_2xy_l910_91021


namespace customers_added_l910_91084

theorem customers_added (x : ℕ) (h : 29 + x = 49) : x = 20 := by
  sorry

end customers_added_l910_91084


namespace solve_system_of_equations_l910_91004

theorem solve_system_of_equations (x y : ℝ) :
  16 * x^3 + 4 * x = 16 * y + 5 ∧ 16 * y^3 + 4 * y = 16 * x + 5 → x = y ∧ 16 * x^3 - 12 * x - 5 = 0 :=
by
  sorry

end solve_system_of_equations_l910_91004


namespace ones_digit_of_8_pow_47_l910_91059

theorem ones_digit_of_8_pow_47 :
  (8^47) % 10 = 2 :=
by
  sorry

end ones_digit_of_8_pow_47_l910_91059


namespace quadratic_roots_new_equation_l910_91033

theorem quadratic_roots_new_equation (a b c x1 x2 : ℝ) 
  (h1 : a ≠ 0) 
  (h2 : x1 + x2 = -b / a) 
  (h3 : x1 * x2 = c / a) : 
  ∃ (a' b' c' : ℝ), a' * x^2 + b' * x + c' = 0 ∧ a' = a^2 ∧ b' = 3 * a * b ∧ c' = 2 * b^2 + a * c :=
sorry

end quadratic_roots_new_equation_l910_91033


namespace probability_Z_l910_91091

variable (p_X p_Y p_Z p_W : ℚ)

def conditions :=
  (p_X = 1/4) ∧ (p_Y = 1/3) ∧ (p_W = 1/6) ∧ (p_X + p_Y + p_Z + p_W = 1)

theorem probability_Z (h : conditions p_X p_Y p_Z p_W) : p_Z = 1/4 :=
by
  obtain ⟨hX, hY, hW, hSum⟩ := h
  sorry

end probability_Z_l910_91091


namespace siblings_total_weight_l910_91097

/-- Given conditions:
Antonio's weight: 50 kilograms.
Antonio's sister weighs 12 kilograms less than Antonio.
Antonio's backpack weight: 5 kilograms.
Antonio's sister's backpack weight: 3 kilograms.
Marco's weight: 30 kilograms.
Marco's stuffed animal weight: 2 kilograms.
Prove that the total weight of the three siblings including additional weights is 128 kilograms.
-/
theorem siblings_total_weight :
  let antonio_weight := 50
  let antonio_sister_weight := antonio_weight - 12
  let antonio_backpack_weight := 5
  let antonio_sister_backpack_weight := 3
  let marco_weight := 30
  let marco_stuffed_animal_weight := 2
  antonio_weight + antonio_backpack_weight +
  antonio_sister_weight + antonio_sister_backpack_weight +
  marco_weight + marco_stuffed_animal_weight = 128 :=
by
  sorry

end siblings_total_weight_l910_91097


namespace a_n_divisible_by_2013_a_n_minus_207_is_cube_l910_91074

theorem a_n_divisible_by_2013 (n : ℕ) (h : n ≥ 1) : 2013 ∣ (4 ^ (6 ^ n) + 1943) :=
by sorry

theorem a_n_minus_207_is_cube (n : ℕ) : (∃ k : ℕ, 4 ^ (6 ^ n) + 1736 = k^3) ↔ (n = 1) :=
by sorry

end a_n_divisible_by_2013_a_n_minus_207_is_cube_l910_91074


namespace number_of_storks_joined_l910_91081

theorem number_of_storks_joined (initial_birds : ℕ) (initial_storks : ℕ) (total_birds_and_storks : ℕ) 
    (h1 : initial_birds = 3) (h2 : initial_storks = 4) (h3 : total_birds_and_storks = 13) : 
    (total_birds_and_storks - (initial_birds + initial_storks)) = 6 := 
by
  sorry

end number_of_storks_joined_l910_91081


namespace avg_visitors_on_sunday_l910_91090

theorem avg_visitors_on_sunday (S : ℕ) :
  (30 * 285) = (5 * S + 25 * 240) -> S = 510 :=
by
  intros h
  sorry

end avg_visitors_on_sunday_l910_91090


namespace derivative_f_at_pi_l910_91044

noncomputable def f (x : ℝ) : ℝ := Real.sin x - Real.cos x

theorem derivative_f_at_pi : (deriv f π) = -1 := 
by
  sorry

end derivative_f_at_pi_l910_91044


namespace angle_PQRS_l910_91068

theorem angle_PQRS (P Q R S : ℝ) (h1 : P = 3 * Q) (h2 : P = 4 * R) (h3 : P = 6 * S) (h4 : P + Q + R + S = 360) : 
  P = 206 := 
by
  sorry

end angle_PQRS_l910_91068


namespace average_length_of_strings_l910_91042

theorem average_length_of_strings : 
  let length1 := 2
  let length2 := 5
  let length3 := 3
  let total_length := length1 + length2 + length3 
  let average_length := total_length / 3
  average_length = 10 / 3 :=
by
  let length1 := 2
  let length2 := 5
  let length3 := 3
  let total_length := length1 + length2 + length3
  let average_length := total_length / 3
  have h1 : total_length = 10 := by rfl
  have h2 : average_length = 10 / 3 := by rfl
  exact h2

end average_length_of_strings_l910_91042


namespace min_val_x_2y_l910_91026

noncomputable def min_x_2y (x y : ℝ) : ℝ :=
  x + 2 * y

theorem min_val_x_2y : 
  ∀ (x y : ℝ), (x > 0) → (y > 0) → (1 / (x + 2) + 1 / (y + 2) = 1 / 3) → 
  min_x_2y x y ≥ 3 + 6 * Real.sqrt 2 :=
by
  intros x y x_pos y_pos eqn
  sorry

end min_val_x_2y_l910_91026


namespace jennys_wedding_guests_l910_91069

noncomputable def total_guests (C S : ℕ) : ℕ := C + S

theorem jennys_wedding_guests :
  ∃ (C S : ℕ), (S = 3 * C) ∧
               (18 * C + 25 * S = 1860) ∧
               (total_guests C S = 80) :=
sorry

end jennys_wedding_guests_l910_91069


namespace concert_total_revenue_l910_91083

def adult_ticket_price : ℕ := 26
def child_ticket_price : ℕ := adult_ticket_price / 2
def num_adults : ℕ := 183
def num_children : ℕ := 28

def revenue_from_adults : ℕ := num_adults * adult_ticket_price
def revenue_from_children : ℕ := num_children * child_ticket_price
def total_revenue : ℕ := revenue_from_adults + revenue_from_children

theorem concert_total_revenue :
  total_revenue = 5122 :=
by
  -- proof can be filled in here
  sorry

end concert_total_revenue_l910_91083


namespace shaded_area_isosceles_right_triangle_l910_91043

theorem shaded_area_isosceles_right_triangle (y : ℝ) :
  (∃ (x : ℝ), 2 * x^2 = y^2) ∧
  (∃ (A : ℝ), A = (1 / 2) * (y^2 / 2)) ∧
  (∃ (shaded_area : ℝ), shaded_area = (1 / 2) * (y^2 / 4)) →
  (shaded_area = y^2 / 8) :=
sorry

end shaded_area_isosceles_right_triangle_l910_91043


namespace minimum_value_expression_l910_91054

theorem minimum_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x^2 + 5*x + 2) * (y^2 + 5*y + 2) * (z^2 + 5*z + 2) / (x * y * z) ≥ 343 :=
sorry

end minimum_value_expression_l910_91054


namespace solveExpression_l910_91020

noncomputable def evaluateExpression : ℝ := (Real.sqrt 3) / Real.sin (Real.pi / 9) - 1 / Real.sin (7 * Real.pi / 18)

theorem solveExpression : evaluateExpression = 4 :=
by sorry

end solveExpression_l910_91020


namespace factory_dolls_per_day_l910_91095

-- Define the number of normal dolls made per day
def N : ℝ := 4800

-- Define the total number of dolls made per day as 1.33 times the number of normal dolls
def T : ℝ := 1.33 * N

-- The theorem statement to prove the factory makes 6384 dolls per day
theorem factory_dolls_per_day : T = 6384 :=
by
  -- Proof here
  sorry

end factory_dolls_per_day_l910_91095


namespace exists_odd_digit_div_by_five_power_l910_91037

theorem exists_odd_digit_div_by_five_power (n : ℕ) (h : 0 < n) : ∃ (k : ℕ), 
  (∃ (m : ℕ), k = m * 5^n) ∧ 
  (∀ (d : ℕ), (d = (k / (10^(n-1))) % 10) → d % 2 = 1) :=
sorry

end exists_odd_digit_div_by_five_power_l910_91037


namespace number_is_48_l910_91056

theorem number_is_48 (x : ℝ) (h : (1/4) * x + 15 = 27) : x = 48 :=
by sorry

end number_is_48_l910_91056


namespace area_of_circle_l910_91051

def circle_area (x y : ℝ) : Prop := x^2 + y^2 - 8 * x + 18 * y = -45

theorem area_of_circle :
  (∃ x y : ℝ, circle_area x y) → ∃ A : ℝ, A = 52 * Real.pi :=
by
  sorry

end area_of_circle_l910_91051


namespace students_catching_up_on_homework_l910_91023

def total_students : ℕ := 24
def silent_reading_students : ℕ := total_students / 2
def board_games_students : ℕ := total_students / 3

theorem students_catching_up_on_homework : 
  total_students - (silent_reading_students + board_games_students) = 4 := by
  sorry

end students_catching_up_on_homework_l910_91023


namespace find_a_l910_91094

theorem find_a (a : ℝ) (h : ∀ B: ℝ × ℝ, (B = (a, 0)) → (2 - 0) * (0 - 2) = (4 - 2) * (2 - a)) : a = 4 :=
by
  sorry

end find_a_l910_91094


namespace unicorn_rope_problem_l910_91002

theorem unicorn_rope_problem
  (d e f : ℕ)
  (h_prime_f : Prime f)
  (h_d : d = 75)
  (h_e : e = 450)
  (h_f : f = 3)
  : d + e + f = 528 := by
  sorry

end unicorn_rope_problem_l910_91002


namespace inequality_proof_l910_91019

open Real

theorem inequality_proof (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_product : a * b * c = 1) :
  1 / (a^3 * (b + c)) + 1 / (b^3 * (a + c)) + 1 / (c^3 * (a + b)) ≥ 3 / 2 :=
by {
  sorry
}

end inequality_proof_l910_91019


namespace number_of_students_l910_91010

theorem number_of_students
    (average_marks : ℕ)
    (wrong_mark : ℕ)
    (correct_mark : ℕ)
    (correct_average_marks : ℕ)
    (h1 : average_marks = 100)
    (h2 : wrong_mark = 50)
    (h3 : correct_mark = 10)
    (h4 : correct_average_marks = 96)
  : ∃ n : ℕ, (100 * n - 40) / n = 96 ∧ n = 10 :=
by
  sorry

end number_of_students_l910_91010


namespace find_divisor_l910_91058

theorem find_divisor 
  (dividend : ℤ)
  (quotient : ℤ)
  (remainder : ℤ)
  (divisor : ℤ)
  (h : dividend = (divisor * quotient) + remainder)
  (h_dividend : dividend = 474232)
  (h_quotient : quotient = 594)
  (h_remainder : remainder = -968) :
  divisor = 800 :=
sorry

end find_divisor_l910_91058


namespace fish_weight_l910_91046

variables (H T X : ℝ)
-- Given conditions
def tail_weight : Prop := X = 1
def head_weight : Prop := H = X + 0.5 * T
def torso_weight : Prop := T = H + X

theorem fish_weight (H T X : ℝ) 
  (h_tail : tail_weight X)
  (h_head : head_weight H T X)
  (h_torso : torso_weight H T X) : 
  H + T + X = 8 :=
sorry

end fish_weight_l910_91046


namespace crows_cannot_be_on_same_tree_l910_91018

theorem crows_cannot_be_on_same_tree :
  (∀ (trees : ℕ) (crows : ℕ),
   trees = 22 ∧ crows = 22 →
   (∀ (positions : ℕ → ℕ),
    (∀ i, 1 ≤ positions i ∧ positions i ≤ 2) →
    ∀ (move : (ℕ → ℕ) → (ℕ → ℕ)),
    (∀ (pos : ℕ → ℕ) (i : ℕ),
     move pos i = pos i + positions (i + 1) ∨ move pos i = pos i - positions (i + 1)) →
    (∀ (pos : ℕ → ℕ) (i : ℕ),
     pos i % trees = (move pos i) % trees) →
    ¬ (∃ (final_pos : ℕ → ℕ),
      (∀ i, final_pos i = 0 ∨ final_pos i = 22) ∧
      (∀ i j, final_pos i = final_pos j)
    )
  )
) :=
sorry

end crows_cannot_be_on_same_tree_l910_91018


namespace robot_cost_l910_91055

theorem robot_cost (num_friends : ℕ) (total_tax change start_money : ℝ) (h_friends : num_friends = 7) (h_tax : total_tax = 7.22) (h_change : change = 11.53) (h_start : start_money = 80) :
  let spent_money := start_money - change
  let cost_robots := spent_money - total_tax
  let cost_per_robot := cost_robots / num_friends
  cost_per_robot = 8.75 :=
by
  sorry

end robot_cost_l910_91055


namespace geo_sequence_arithmetic_l910_91071

variable {d : ℝ} (hd : d ≠ 0)
variable {a : ℕ → ℝ} (ha : ∀ n, a (n+1) = a n + d)

-- Hypothesis that a_5, a_9, a_15 form a geometric sequence
variable (hgeo : a 9 ^ 2 = (a 9 - 4 * d) * (a 9 + 6 * d))

theorem geo_sequence_arithmetic (hd : d ≠ 0) (ha : ∀ n, a (n + 1) = a n + d) (hgeo : a 9 ^ 2 = (a 9 - 4 * d) * (a 9 + 6 * d)) :
  a 15 / a 9 = 3 / 2 :=
by
  sorry

end geo_sequence_arithmetic_l910_91071


namespace melanie_total_value_l910_91029

-- Define the initial number of dimes Melanie had
def initial_dimes : ℕ := 7

-- Define the number of dimes given by her dad
def dimes_from_dad : ℕ := 8

-- Define the number of dimes given by her mom
def dimes_from_mom : ℕ := 4

-- Calculate the total number of dimes Melanie has now
def total_dimes : ℕ := initial_dimes + dimes_from_dad + dimes_from_mom

-- Define the value of each dime in dollars
def value_per_dime : ℝ := 0.10

-- Calculate the total value of dimes in dollars
def total_value_in_dollars : ℝ := total_dimes * value_per_dime

-- The theorem states that the total value in dollars is 1.90
theorem melanie_total_value : total_value_in_dollars = 1.90 := 
by
  -- Using the established definitions, the goal follows directly.
  sorry

end melanie_total_value_l910_91029


namespace no_common_points_l910_91022

theorem no_common_points : 
  ∀ (x y : ℝ), ¬(x^2 + y^2 = 9 ∧ x^2 + y^2 = 4) := 
by
  sorry

end no_common_points_l910_91022


namespace base8_to_base10_4513_l910_91057

theorem base8_to_base10_4513 : (4 * 8^3 + 5 * 8^2 + 1 * 8^1 + 3 * 8^0 = 2379) :=
by
  sorry

end base8_to_base10_4513_l910_91057


namespace cake_shop_problem_l910_91009

theorem cake_shop_problem :
  ∃ (N n K : ℕ), (N - n * K = 6) ∧ (N = (n - 1) * 8 + 1) ∧ (N = 97) :=
by
  sorry

end cake_shop_problem_l910_91009


namespace unique_elements_condition_l910_91078

theorem unique_elements_condition (x : ℝ) : 
  (1 ≠ x ∧ x ≠ x^2 ∧ 1 ≠ x^2) ↔ (x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1) :=
by 
  sorry

end unique_elements_condition_l910_91078


namespace find_fx2_l910_91063

theorem find_fx2 (f : ℝ → ℝ) (x : ℝ) (h : f (x - 1) = x ^ 2) : f (x ^ 2) = (x ^ 2 + 1) ^ 2 := by
  sorry

end find_fx2_l910_91063


namespace bobby_final_paycheck_correct_l910_91064

def bobby_salary : ℕ := 450
def federal_tax_rate : ℚ := 1/3
def state_tax_rate : ℚ := 0.08
def health_insurance_deduction : ℕ := 50
def life_insurance_deduction : ℕ := 20
def city_parking_fee : ℕ := 10

def final_paycheck_amount : ℚ :=
  let federal_taxes := federal_tax_rate * bobby_salary
  let state_taxes := state_tax_rate * bobby_salary
  let total_deductions := federal_taxes + state_taxes + health_insurance_deduction + life_insurance_deduction + city_parking_fee
  bobby_salary - total_deductions

theorem bobby_final_paycheck_correct : final_paycheck_amount = 184 := by
  sorry

end bobby_final_paycheck_correct_l910_91064


namespace floor_plus_ceil_eq_seven_l910_91008

theorem floor_plus_ceil_eq_seven (x : ℝ) :
  (⌊x⌋ + ⌈x⌉ = 7) ↔ (3 < x ∧ x < 4) :=
sorry

end floor_plus_ceil_eq_seven_l910_91008


namespace daily_rental_cost_l910_91085

def daily_cost (x : ℝ) (miles : ℝ) (cost_per_mile : ℝ) : ℝ :=
  x + miles * cost_per_mile

theorem daily_rental_cost (x : ℝ) (miles : ℝ) (cost_per_mile : ℝ) (total_budget : ℝ) 
  (h : daily_cost x miles cost_per_mile = total_budget) : x = 30 :=
by
  let constant_miles := 200
  let constant_cost_per_mile := 0.23
  let constant_budget := 76
  sorry

end daily_rental_cost_l910_91085


namespace range_of_y_l910_91053

theorem range_of_y (y : ℝ) (h1 : y < 0) (h2 : Int.ceil y * Int.floor y = 72) : 
  -9 < y ∧ y < -8 :=
sorry

end range_of_y_l910_91053


namespace tim_minus_tom_l910_91014

def sales_tax_rate : ℝ := 0.07
def original_price : ℝ := 120.00
def discount_rate : ℝ := 0.25
def city_tax_rate : ℝ := 0.05

noncomputable def tim_total : ℝ :=
  let price_with_tax := original_price * (1 + sales_tax_rate)
  price_with_tax * (1 - discount_rate)

noncomputable def tom_total : ℝ :=
  let discounted_price := original_price * (1 - discount_rate)
  let price_with_sales_tax := discounted_price * (1 + sales_tax_rate)
  price_with_sales_tax * (1 + city_tax_rate)

theorem tim_minus_tom : tim_total - tom_total = -4.82 := 
by sorry

end tim_minus_tom_l910_91014


namespace length_increase_percentage_l910_91049

theorem length_increase_percentage
  (L W : ℝ)
  (A : ℝ := L * W)
  (A' : ℝ := 1.30000000000000004 * A)
  (new_length : ℝ := L * (1 + x / 100))
  (new_width : ℝ := W / 2)
  (area_equiv : new_length * new_width = A')
  (x : ℝ) :
  1 + x / 100 = 2.60000000000000008 :=
by
  -- Proof goes here
  sorry

end length_increase_percentage_l910_91049


namespace S_n_min_at_5_min_nS_n_is_neg_49_l910_91061

variable {S_n : ℕ → ℝ}
variable {a_1 d : ℝ}

-- Conditions
axiom sum_first_n_terms (n : ℕ) : S_n n = n / 2 * (2 * a_1 + (n - 1) * d)

axiom S_10 : S_n 10 = 0
axiom S_15 : S_n 15 = 25

-- Proving the following statements
theorem S_n_min_at_5 :
  (∀ n, S_n n ≥ S_n 5) :=
sorry

theorem min_nS_n_is_neg_49 :
  (∀ n, n * S_n n ≥ -49) :=
sorry

end S_n_min_at_5_min_nS_n_is_neg_49_l910_91061


namespace optimal_play_winner_l910_91035

theorem optimal_play_winner (n : ℕ) (h : n > 1) : (n % 2 = 0) ↔ (first_player_wins: Bool) :=
  sorry

end optimal_play_winner_l910_91035


namespace activity_popularity_order_l910_91011

theorem activity_popularity_order
  (dodgeball : ℚ := 13 / 40)
  (picnic : ℚ := 9 / 30)
  (swimming : ℚ := 7 / 20)
  (crafts : ℚ := 3 / 15) :
  (swimming > dodgeball ∧ dodgeball > picnic ∧ picnic > crafts) :=
by 
  sorry

end activity_popularity_order_l910_91011


namespace two_numbers_sum_l910_91070

theorem two_numbers_sum (N1 N2 : ℕ) (h1 : N1 % 10^5 = 0) (h2 : N2 % 10^5 = 0) 
  (h3 : N1 ≠ N2) (h4 : (Nat.divisors N1).card = 42) (h5 : (Nat.divisors N2).card = 42) : 
  N1 + N2 = 700000 := 
by
  sorry

end two_numbers_sum_l910_91070


namespace pick_theorem_l910_91048

def lattice_polygon (vertices : List (ℤ × ℤ)) : Prop :=
  ∀ v ∈ vertices, ∃ i j : ℤ, v = (i, j)

variables {n m : ℕ}
variables {A : ℤ}
variables {vertices : List (ℤ × ℤ)}

def lattice_point_count_inside (vertices : List (ℤ × ℤ)) : ℕ :=
  -- Placeholder for the actual logic to count inside points
  sorry

def lattice_point_count_boundary (vertices : List (ℤ × ℤ)) : ℕ :=
  -- Placeholder for the actual logic to count boundary points
  sorry

theorem pick_theorem (h : lattice_polygon vertices) :
  lattice_point_count_inside vertices = n → 
  lattice_point_count_boundary vertices = m → 
  A = n + m / 2 - 1 :=
sorry

end pick_theorem_l910_91048


namespace hash_triple_l910_91001

def hash (N : ℝ) : ℝ := 0.5 * (N^2) + 1

theorem hash_triple  : hash (hash (hash 4)) = 862.125 :=
by {
  sorry
}

end hash_triple_l910_91001


namespace heptagon_angle_sum_l910_91038

theorem heptagon_angle_sum 
  (angle_A angle_B angle_C angle_D angle_E angle_F angle_G : ℝ) 
  (h : angle_A + angle_B + angle_C + angle_D + angle_E + angle_F + angle_G = 540) :
  angle_A + angle_B + angle_C + angle_D + angle_E + angle_F + angle_G = 540 :=
by
  sorry

end heptagon_angle_sum_l910_91038


namespace first_tier_price_level_is_10000_l910_91024

noncomputable def first_tier_price_level (P : ℝ) : Prop :=
  ∀ (car_price : ℝ), car_price = 30000 → (P ≤ car_price ∧ 
    (0.25 * P + 0.15 * (car_price - P)) = 5500)

theorem first_tier_price_level_is_10000 :
  first_tier_price_level 10000 :=
by
  sorry

end first_tier_price_level_is_10000_l910_91024


namespace mean_proportional_of_segments_l910_91045

theorem mean_proportional_of_segments (a b c : ℝ) (a_val : a = 2) (b_val : b = 6) :
  c = 2 * Real.sqrt 3 ↔ c*c = a * b := by
  sorry

end mean_proportional_of_segments_l910_91045


namespace remainder_of_3056_mod_32_l910_91034

theorem remainder_of_3056_mod_32 : 3056 % 32 = 16 := by
  sorry

end remainder_of_3056_mod_32_l910_91034


namespace managers_non_managers_ratio_l910_91015

theorem managers_non_managers_ratio
  (M N : ℕ)
  (h_ratio : M / N > 7 / 24)
  (h_max_non_managers : N = 27) :
  ∃ M, 8 ≤ M ∧ M / 27 > 7 / 24 :=
by
  sorry

end managers_non_managers_ratio_l910_91015


namespace arithmetic_mean_of_numbers_l910_91065

theorem arithmetic_mean_of_numbers (n : ℕ) (h : n > 1) :
  let one_special_number := (1 / n) + (2 / n ^ 2)
  let other_numbers := (n - 1) * 1
  (other_numbers + one_special_number) / n = 1 + 2 / n ^ 2 :=
by
  sorry

end arithmetic_mean_of_numbers_l910_91065


namespace gcd_of_three_numbers_l910_91041

-- Define the given numbers
def a := 72
def b := 120
def c := 168

-- Define the GCD function and prove the required statement
theorem gcd_of_three_numbers : Nat.gcd (Nat.gcd a b) c = 24 := by
  -- Intermediate steps and their justifications would go here in the proof, but we are putting sorry
  sorry

end gcd_of_three_numbers_l910_91041


namespace root_exists_between_0_and_1_l910_91099

theorem root_exists_between_0_and_1 (a b c : ℝ) (m : ℝ) (hm : 0 < m)
  (h : a / (m + 2) + b / (m + 1) + c / m = 0) : 
  ∃ x : ℝ, 0 < x ∧ x < 1 ∧ a * x ^ 2 + b * x + c = 0 :=
by
  sorry

end root_exists_between_0_and_1_l910_91099


namespace total_legs_correct_l910_91060

-- Number of animals
def horses : ℕ := 2
def dogs : ℕ := 5
def cats : ℕ := 7
def turtles : ℕ := 3
def goat : ℕ := 1

-- Total number of animals
def total_animals : ℕ := horses + dogs + cats + turtles + goat

-- Total number of legs
def total_legs : ℕ := total_animals * 4

theorem total_legs_correct : total_legs = 72 := by
  -- proof skipped
  sorry

end total_legs_correct_l910_91060


namespace sum_abs_values_l910_91082

theorem sum_abs_values (a b : ℝ) (h₁ : abs a = 4) (h₂ : abs b = 7) (h₃ : a < b) : a + b = 3 ∨ a + b = 11 :=
by
  sorry

end sum_abs_values_l910_91082


namespace right_triangle_roots_l910_91080

theorem right_triangle_roots (m a b c : ℝ) 
  (h_eq : ∀ x, x^2 - (2 * m + 1) * x + m^2 + m = 0)
  (h_roots : a^2 - (2 * m + 1) * a + m^2 + m = 0 ∧ b^2 - (2 * m + 1) * b + m^2 + m = 0)
  (h_triangle : a^2 + b^2 = c^2)
  (h_c : c = 5) : 
  m = 3 :=
by sorry

end right_triangle_roots_l910_91080


namespace sum_of_coordinates_of_D_l910_91000

/--
Given points A = (4,8), B = (2,4), C = (6,6), and D = (a,b) in the first quadrant, if the quadrilateral formed by joining the midpoints of the segments AB, BC, CD, and DA is a square with sides inclined at 45 degrees to the x-axis, then the sum of the coordinates of point D is 6.
-/
theorem sum_of_coordinates_of_D 
  (a b : ℝ)
  (h_quadrilateral : ∃ A B C D : Prod ℝ ℝ, 
    A = (4, 8) ∧ B = (2, 4) ∧ C = (6, 6) ∧ D = (a, b) ∧ 
    ∃ M1 M2 M3 M4 : Prod ℝ ℝ,
    M1 = ((4 + 2) / 2, (8 + 4) / 2) ∧ M2 = ((2 + 6) / 2, (4 + 6) / 2) ∧ 
    M3 = (M2.1 + 1, M2.2 - 1) ∧ M4 = (M3.1 + 1, M3.2 + 1) ∧ 
    M3 = ((a + 6) / 2, (b + 6) / 2) ∧ M4 = ((a + 4) / 2, (b + 8) / 2)
  ) : 
  a + b = 6 := sorry

end sum_of_coordinates_of_D_l910_91000


namespace rectangle_area_is_12_l910_91036

noncomputable def rectangle_area_proof (w l y : ℝ) : Prop :=
  l = 3 * w ∧ 2 * (l + w) = 16 ∧ (l^2 + w^2 = y^2) → l * w = 12

theorem rectangle_area_is_12 (y : ℝ) : ∃ (w l : ℝ), rectangle_area_proof w l y :=
by
  -- Introducing variables
  exists 2
  exists 6
  -- Constructing proof steps (skipped here with sorry)
  sorry

end rectangle_area_is_12_l910_91036

import Mathlib

namespace necessary_but_not_sufficient_l242_242261

theorem necessary_but_not_sufficient (x y : ℝ) : 
  (x - y > -1) → (x^3 + x > x^2 * y + y) → 
  ∃ z : ℝ, z - y > -1 ∧ ¬ (z^3 + z > z^2 * y + y) :=
sorry

end necessary_but_not_sufficient_l242_242261


namespace number_of_strikers_l242_242801

theorem number_of_strikers 
  (goalies defenders midfielders strikers : ℕ) 
  (h1 : goalies = 3) 
  (h2 : defenders = 10) 
  (h3 : midfielders = 2 * defenders) 
  (h4 : goalies + defenders + midfielders + strikers = 40) : 
  strikers = 7 := 
sorry

end number_of_strikers_l242_242801


namespace james_weekly_earnings_l242_242294

def rate_per_hour : ℕ := 20
def hours_per_day : ℕ := 8
def days_per_week : ℕ := 4

def daily_earnings : ℕ := rate_per_hour * hours_per_day
def weekly_earnings : ℕ := daily_earnings * days_per_week

theorem james_weekly_earnings : weekly_earnings = 640 := sorry

end james_weekly_earnings_l242_242294


namespace rectangle_length_l242_242031

theorem rectangle_length (P L W : ℕ) (h1 : P = 48) (h2 : L = 2 * W) (h3 : P = 2 * L + 2 * W) : L = 16 := by
  sorry

end rectangle_length_l242_242031


namespace alcohol_water_ratio_l242_242006

theorem alcohol_water_ratio 
  (P_alcohol_pct : ℝ) (Q_alcohol_pct : ℝ) 
  (P_volume : ℝ) (Q_volume : ℝ) 
  (mixture_alcohol : ℝ) (mixture_water : ℝ)
  (h1 : P_alcohol_pct = 62.5)
  (h2 : Q_alcohol_pct = 87.5)
  (h3 : P_volume = 4)
  (h4 : Q_volume = 4)
  (ha : mixture_alcohol = (P_volume * (P_alcohol_pct / 100)) + (Q_volume * (Q_alcohol_pct / 100)))
  (hm : mixture_water = (P_volume + Q_volume) - mixture_alcohol) :
  mixture_alcohol / mixture_water = 3 :=
by
  sorry

end alcohol_water_ratio_l242_242006


namespace kids_bike_wheels_l242_242462

theorem kids_bike_wheels
  (x : ℕ) 
  (h1 : 7 * 2 + 11 * x = 58) :
  x = 4 :=
sorry

end kids_bike_wheels_l242_242462


namespace total_regular_and_diet_soda_bottles_l242_242811

-- Definitions from the conditions
def regular_soda_bottles := 49
def diet_soda_bottles := 40

-- The statement to prove
theorem total_regular_and_diet_soda_bottles :
  regular_soda_bottles + diet_soda_bottles = 89 :=
by
  sorry

end total_regular_and_diet_soda_bottles_l242_242811


namespace international_news_duration_l242_242469

theorem international_news_duration
  (total_duration : ℕ := 30)
  (national_news : ℕ := 12)
  (sports : ℕ := 5)
  (weather_forecasts : ℕ := 2)
  (advertising : ℕ := 6) :
  total_duration - national_news - sports - weather_forecasts - advertising = 5 :=
by
  sorry

end international_news_duration_l242_242469


namespace f_characterization_l242_242278

noncomputable def op (a b : ℝ) := a * b

noncomputable def ot (a b : ℝ) := a + b

noncomputable def f (x : ℝ) := ot x 2 - op 2 x

-- Prove that f(x) is neither odd nor even and is a decreasing function
theorem f_characterization :
  (∀ x : ℝ, f x = -x + 2) ∧
  (∀ x : ℝ, f (-x) ≠ f x ∧ f (-x) ≠ -f x) ∧
  (∀ x y : ℝ, x < y → f x > f y) := sorry

end f_characterization_l242_242278


namespace simplify_expression_l242_242674

theorem simplify_expression (x y : ℝ) :
  5 * x - 3 * y + 9 * x ^ 2 + 8 - (4 - 5 * x + 3 * y - 9 * x ^ 2) = 18 * x ^ 2 + 10 * x - 6 * y + 4 :=
by
  sorry

end simplify_expression_l242_242674


namespace number_of_action_figures_bought_l242_242182

-- Definitions of conditions
def cost_of_board_game : ℕ := 2
def cost_per_action_figure : ℕ := 7
def total_spent : ℕ := 30

-- The problem to prove
theorem number_of_action_figures_bought : 
  ∃ (n : ℕ), total_spent - cost_of_board_game = n * cost_per_action_figure ∧ n = 4 :=
by
  sorry

end number_of_action_figures_bought_l242_242182


namespace trip_time_l242_242045

theorem trip_time (distance half_dist speed1 speed2 : ℝ) 
  (h_distance : distance = 360) 
  (h_half_distance : half_dist = distance / 2) 
  (h_speed1 : speed1 = 50) 
  (h_speed2 : speed2 = 45) : 
  (half_dist / speed1 + half_dist / speed2) = 7.6 := 
by
  -- Simplify the expressions based on provided conditions
  sorry

end trip_time_l242_242045


namespace intersection_eq_1_2_l242_242682

-- Define the set M
def M : Set ℝ := {y : ℝ | -2 ≤ y ∧ y ≤ 2}

-- Define the set N
def N : Set ℝ := {x : ℝ | 1 < x}

-- The intersection of M and N
def intersection : Set ℝ := { x : ℝ | 1 < x ∧ x ≤ 2 }

-- Our goal is to prove that M ∩ N = (1, 2]
theorem intersection_eq_1_2 : (M ∩ N) = (Set.Ioo 1 2) :=
by
  sorry

end intersection_eq_1_2_l242_242682


namespace polynomial_factorization_l242_242244

theorem polynomial_factorization (x : ℝ) : x - x^3 = x * (1 - x) * (1 + x) := 
by sorry

end polynomial_factorization_l242_242244


namespace hyperbola_equation_l242_242155

theorem hyperbola_equation {a b : ℝ} (ha : a > 0) (hb : b > 0) :
  (∀ {x y : ℝ}, x^2 / 12 + y^2 / 4 = 1 → True) →
  (∀ {x y : ℝ}, x^2 / a^2 - y^2 / b^2 = 1 → True) →
  (∀ {x y : ℝ}, y = Real.sqrt 3 * x → True) →
  (∃ k : ℝ, 4 < k ∧ k < 12 ∧ 2 = 12 - k ∧ 6 = k - 4) →
  a = 2 ∧ b = 6 := by
  intros h_ellipse h_hyperbola h_asymptote h_k
  sorry

end hyperbola_equation_l242_242155


namespace min_value_f_l242_242279

def f (x y z : ℝ) : ℝ := 
  x^2 + 4 * x * y + 3 * y^2 + 2 * z^2 - 8 * x - 4 * y + 6 * z

theorem min_value_f : ∃ (x y z : ℝ), f x y z = -13.5 :=
  by
  use 1, 1.5, -1.5
  sorry

end min_value_f_l242_242279


namespace find_ab_l242_242500

theorem find_ab (a b : ℝ) (h1 : a - b = 26) (h2 : a + b = 15) :
  a = 41 / 2 ∧ b = 11 / 2 :=
sorry

end find_ab_l242_242500


namespace tomatoes_eaten_l242_242713

theorem tomatoes_eaten (initial_tomatoes : ℕ) (remaining_tomatoes : ℕ) (portion_eaten : ℚ)
  (h_init : initial_tomatoes = 21)
  (h_rem : remaining_tomatoes = 14)
  (h_portion : portion_eaten = 1/3) :
  initial_tomatoes - remaining_tomatoes = (portion_eaten * initial_tomatoes) :=
by
  sorry

end tomatoes_eaten_l242_242713


namespace smallest_unpayable_amount_l242_242761

theorem smallest_unpayable_amount :
  ∀ (coins_1p coins_2p coins_3p coins_4p coins_5p : ℕ), 
    coins_1p = 1 → 
    coins_2p = 2 → 
    coins_3p = 3 → 
    coins_4p = 4 → 
    coins_5p = 5 → 
    ∃ (x : ℕ), x = 56 ∧ 
    ¬ (∃ (a b c d e : ℕ), a * 1 + b * 2 + c * 3 + d * 4 + e * 5 = x ∧ 
    a ≤ coins_1p ∧
    b ≤ coins_2p ∧
    c ≤ coins_3p ∧
    d ≤ coins_4p ∧
    e ≤ coins_5p) :=
by {
  -- Here we skip the actual proof
  sorry
}

end smallest_unpayable_amount_l242_242761


namespace train_length_correct_l242_242018

noncomputable def train_length (v_kmph : ℝ) (t_sec : ℝ) (bridge_length : ℝ) : ℝ :=
  let v_mps := v_kmph / 3.6
  let total_distance := v_mps * t_sec
  total_distance - bridge_length

theorem train_length_correct : train_length 72 12.099 132 = 109.98 :=
by
  sorry

end train_length_correct_l242_242018


namespace prime_divides_sum_diff_l242_242067

theorem prime_divides_sum_diff
  (a b c p : ℕ) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hc : c ≠ 0) 
  (hp : p.Prime) 
  (h1 : p ∣ (100 * a + 10 * b + c)) 
  (h2 : p ∣ (100 * c + 10 * b + a)) 
  : p ∣ (a + b + c) ∨ p ∣ (a - b + c) ∨ p ∣ (a - c) :=
by
  sorry

end prime_divides_sum_diff_l242_242067


namespace twice_original_price_l242_242287

theorem twice_original_price (P : ℝ) (h : 377 = 1.30 * P) : 2 * P = 580 :=
by {
  -- proof steps will go here
  sorry
}

end twice_original_price_l242_242287


namespace minimum_3x_4y_l242_242176

theorem minimum_3x_4y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 3 * y = 5 * x * y) : 3 * x + 4 * y ≥ 5 :=
by
  sorry

end minimum_3x_4y_l242_242176


namespace T_n_lt_1_l242_242673

open Nat

def a (n : ℕ) : ℕ := 2^n

def b (n : ℕ) : ℕ := 2^n - 1

def c (n : ℕ) : ℚ := (a n : ℚ) / ((b n : ℚ) * (b (n + 1) : ℚ))

noncomputable def T (n : ℕ) : ℚ := (Finset.range (n + 1)).sum c

theorem T_n_lt_1 (n : ℕ) : T n < 1 := by
  sorry

end T_n_lt_1_l242_242673


namespace distance_interval_l242_242832

theorem distance_interval (d : ℝ) :
  (d < 8) ∧ (d > 7) ∧ (d > 5) ∧ (d ≠ 3) ↔ (7 < d ∧ d < 8) :=
by
  sorry

end distance_interval_l242_242832


namespace equivalent_polar_point_representation_l242_242214

/-- Representation of a point in polar coordinates -/
structure PolarPoint :=
  (r : ℝ)
  (θ : ℝ)

theorem equivalent_polar_point_representation :
  ∀ (p1 p2 : PolarPoint), p1 = PolarPoint.mk (-1) (5 * Real.pi / 6) →
    (p2 = PolarPoint.mk 1 (11 * Real.pi / 6) → p1.r + Real.pi = p2.r ∧ p1.θ = p2.θ) :=
by
  intros p1 p2 h1 h2
  sorry

end equivalent_polar_point_representation_l242_242214


namespace polynomial_of_degree_2_l242_242154

noncomputable def polynomialSeq (n : ℕ) : Prop :=
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → 
    ∃ (f_k f_k1 f_k2 : Polynomial ℝ),
      f_k ≠ Polynomial.C 0 ∧ (f_k * f_k1 = f_k1.comp f_k2)

theorem polynomial_of_degree_2 (n : ℕ) (h : n ≥ 3) :
  polynomialSeq n → 
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → 
    ∃ f : Polynomial ℝ, f = Polynomial.X ^ 2 :=
sorry

end polynomial_of_degree_2_l242_242154


namespace part_one_part_two_l242_242001

def M (n : ℤ) : ℤ := n - 3
def M_frac (n : ℚ) : ℚ := - (1 / n^2)

theorem part_one 
    : M 28 * M_frac (1/5) = -1 :=
by {
  sorry
}

theorem part_two 
    : -1 / M 39 / (- M_frac (1/6)) = -1 :=
by {
  sorry
}

end part_one_part_two_l242_242001


namespace transform_sequence_zero_l242_242360

theorem transform_sequence_zero 
  (n : ℕ) 
  (a : ℕ → ℝ) 
  (h_nonempty : n > 0) :
  ∃ k : ℕ, k ≤ n ∧ ∀ k' ≤ k, ∃ α : ℝ, (∀ i, i < n → |a i - α| = 0) := 
sorry

end transform_sequence_zero_l242_242360


namespace solve_abc_values_l242_242686

theorem solve_abc_values (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a + 1/b = 5)
  (h2 : b + 1/c = 2)
  (h3 : c + 1/a = 8/3) :
  abc = 1 ∨ abc = 37/3 :=
sorry

end solve_abc_values_l242_242686


namespace sum_of_non_visible_faces_l242_242439

theorem sum_of_non_visible_faces
    (d1 d2 d3 d4 : Fin 6 → Nat)
    (visible_faces : List Nat)
    (hv : visible_faces = [1, 2, 3, 4, 4, 5, 5, 6]) :
    let total_sum := 4 * 21
    let visible_sum := List.sum visible_faces
    total_sum - visible_sum = 54 := by
  sorry

end sum_of_non_visible_faces_l242_242439


namespace tan_product_eq_three_l242_242615

noncomputable def tan_pi_over_9 : ℝ := Real.tan (Real.pi / 9)
noncomputable def tan_2pi_over_9 : ℝ := Real.tan (2 * Real.pi / 9)
noncomputable def tan_4pi_over_9 : ℝ := Real.tan (4 * Real.pi / 9)

theorem tan_product_eq_three : tan_pi_over_9 * tan_2pi_over_9 * tan_4pi_over_9 = 3 := by
    sorry

end tan_product_eq_three_l242_242615


namespace number_of_students_in_both_ball_and_track_l242_242180

variable (total studentsSwim studentsTrack studentsBall bothSwimTrack bothSwimBall bothTrackBall : ℕ)
variable (noAllThree : Prop)

theorem number_of_students_in_both_ball_and_track
  (h_total : total = 26)
  (h_swim : studentsSwim = 15)
  (h_track : studentsTrack = 8)
  (h_ball : studentsBall = 14)
  (h_both_swim_track : bothSwimTrack = 3)
  (h_both_swim_ball : bothSwimBall = 3)
  (h_no_all_three : noAllThree) :
  bothTrackBall = 5 := by
  sorry

end number_of_students_in_both_ball_and_track_l242_242180


namespace positive_difference_of_squares_and_product_l242_242090

theorem positive_difference_of_squares_and_product (x y : ℕ) 
  (h1 : x + y = 60) (h2 : x - y = 16) :
  x^2 - y^2 = 960 ∧ x * y = 836 :=
by sorry

end positive_difference_of_squares_and_product_l242_242090


namespace paper_thickness_after_folding_five_times_l242_242619

-- Definitions of initial conditions
def initial_thickness : ℝ := 0.1
def num_folds : ℕ := 5

-- Target thickness after folding
def final_thickness (init_thickness : ℝ) (folds : ℕ) : ℝ :=
  (2 ^ folds) * init_thickness

-- Statement of the theorem
theorem paper_thickness_after_folding_five_times :
  final_thickness initial_thickness num_folds = 3.2 :=
by
  -- The proof (the implementation is replaced with sorry)
  sorry

end paper_thickness_after_folding_five_times_l242_242619


namespace hexagon_classroom_students_l242_242461

-- Define the number of sleeping students
def num_sleeping_students (students_detected : Nat → Nat) :=
  students_detected 2 + students_detected 3 + students_detected 6

-- Define the condition that the sum of snore-o-meter readings is 7
def snore_o_meter_sum (students_detected : Nat → Nat) :=
  2 * students_detected 2 + 3 * students_detected 3 + 6 * students_detected 6 = 7

-- Proof that the number of sleeping students is 3 given the conditions
theorem hexagon_classroom_students : 
  ∀ (students_detected : Nat → Nat), snore_o_meter_sum students_detected → num_sleeping_students students_detected = 3 :=
by
  intro students_detected h
  sorry

end hexagon_classroom_students_l242_242461


namespace prop_false_iff_a_lt_neg_13_over_2_l242_242745

theorem prop_false_iff_a_lt_neg_13_over_2 :
  (¬ ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 + a * x + 9 ≥ 0) ↔ a < -13 / 2 := 
sorry

end prop_false_iff_a_lt_neg_13_over_2_l242_242745


namespace find_value_of_alpha_beta_plus_alpha_plus_beta_l242_242280

variable (α β : ℝ)

theorem find_value_of_alpha_beta_plus_alpha_plus_beta
  (hα : α^2 + α - 1 = 0)
  (hβ : β^2 + β - 1 = 0)
  (hαβ : α ≠ β) :
  α * β + α + β = -2 := 
by
  sorry

end find_value_of_alpha_beta_plus_alpha_plus_beta_l242_242280


namespace distinct_rational_numbers_l242_242223

theorem distinct_rational_numbers (m : ℚ) :
  abs m < 100 ∧ (∃ x : ℤ, 4 * x^2 + m * x + 15 = 0) → 
  ∃ n : ℕ, n = 48 :=
sorry

end distinct_rational_numbers_l242_242223


namespace cos_seven_pi_over_six_l242_242263

theorem cos_seven_pi_over_six : Real.cos (7 * Real.pi / 6) = -Real.sqrt 3 / 2 :=
by
  sorry

end cos_seven_pi_over_six_l242_242263


namespace isosceles_triangle_angle_between_vectors_l242_242141

theorem isosceles_triangle_angle_between_vectors 
  (α β γ : ℝ) 
  (h1: α + β + γ = 180)
  (h2: α = 120) 
  (h3: β = γ):
  180 - β = 150 :=
sorry

end isosceles_triangle_angle_between_vectors_l242_242141


namespace minimum_at_neg_one_l242_242435

noncomputable def f (x : Real) : Real := x * Real.exp x

theorem minimum_at_neg_one : 
  ∃ c : Real, c = -1 ∧ ∀ x : Real, f c ≤ f x := sorry

end minimum_at_neg_one_l242_242435


namespace rem_neg_one_third_quarter_l242_242725

noncomputable def rem (x y : ℝ) : ℝ :=
  x - y * ⌊x / y⌋

theorem rem_neg_one_third_quarter :
  rem (-1/3) (1/4) = 1/6 :=
by
  sorry

end rem_neg_one_third_quarter_l242_242725


namespace right_triangle_midpoints_distances_l242_242808

theorem right_triangle_midpoints_distances (a b : ℝ) 
  (hXON : 19^2 = a^2 + (b/2)^2)
  (hYOM : 22^2 = b^2 + (a/2)^2) :
  a^2 + b^2 = 676 :=
by
  sorry

end right_triangle_midpoints_distances_l242_242808


namespace distributive_laws_fail_for_all_l242_242326

def has_op_hash (a b : ℝ) : ℝ := a + 2 * b

theorem distributive_laws_fail_for_all (x y z : ℝ) : 
  ¬ (∀ x y z, has_op_hash x (y + z) = has_op_hash x y + has_op_hash x z) ∧
  ¬ (∀ x y z, x + has_op_hash y z = has_op_hash (x + y) (x + z)) ∧
  ¬ (∀ x y z, has_op_hash x (has_op_hash y z) = has_op_hash (has_op_hash x y) (has_op_hash x z)) := 
sorry

end distributive_laws_fail_for_all_l242_242326


namespace liquid_X_percentage_36_l242_242520

noncomputable def liquid_X_percentage (m : ℕ) (pX : ℕ) (m_evaporate : ℕ) (m_add : ℕ) (p_add : ℕ) : ℕ :=
  let m_X_initial := (pX * m / 100)
  let m_water_initial := ((100 - pX) * m / 100)
  let m_X_after_evaporation := m_X_initial
  let m_water_after_evaporation := m_water_initial - m_evaporate
  let m_X_additional := (p_add * m_add / 100)
  let m_water_additional := ((100 - p_add) * m_add / 100)
  let m_X_new := m_X_after_evaporation + m_X_additional
  let m_water_new := m_water_after_evaporation + m_water_additional
  let m_total_new := m_X_new + m_water_new
  (m_X_new * 100 / m_total_new)

theorem liquid_X_percentage_36 :
  liquid_X_percentage 10 30 2 2 30 = 36 := by
  sorry

end liquid_X_percentage_36_l242_242520


namespace student_B_speed_l242_242399

theorem student_B_speed (d : ℝ) (ratio : ℝ) (t_diff : ℝ) (sB : ℝ) : 
  d = 12 → ratio = 1.2 → t_diff = 1/6 → 
  (d / sB - t_diff = d / (ratio * sB)) → 
  sB = 12 :=
by
  intros h1 h2 h3 h4
  sorry

end student_B_speed_l242_242399


namespace bike_route_length_l242_242568

theorem bike_route_length (u1 u2 u3 l1 l2 : ℕ) (h1 : u1 = 4) (h2 : u2 = 7) (h3 : u3 = 2) (h4 : l1 = 6) (h5 : l2 = 7) :
  u1 + u2 + u3 + u1 + u2 + u3 + l1 + l2 + l1 + l2 = 52 := 
by
  sorry

end bike_route_length_l242_242568


namespace pies_sold_l242_242631

-- Define the conditions in Lean
def num_cakes : ℕ := 453
def price_per_cake : ℕ := 12
def total_earnings : ℕ := 6318
def price_per_pie : ℕ := 7

-- Define the problem
theorem pies_sold (P : ℕ) (h1 : num_cakes * price_per_cake + P * price_per_pie = total_earnings) : P = 126 := 
by 
  sorry

end pies_sold_l242_242631


namespace point_P_path_length_l242_242809

/-- A rectangle PQRS in the plane with points P Q R S, where PQ = RS = 2 and QR = SP = 6. 
    The rectangle is rotated 90 degrees twice: first about point R and then 
    about the new position of point S after the first rotation. 
    The goal is to prove that the length of the path P travels is (3 + sqrt 10) * pi. -/
theorem point_P_path_length :
  ∀ (P Q R S : ℝ × ℝ), 
    dist P Q = 2 ∧ dist Q R = 6 ∧ dist R S = 2 ∧ dist S P = 6 →
    ∃ path_length : ℝ, path_length = (3 + Real.sqrt 10) * Real.pi :=
by
  sorry

end point_P_path_length_l242_242809


namespace lily_milk_remaining_l242_242213

def lilyInitialMilk : ℚ := 4
def milkGivenAway : ℚ := 7 / 3
def milkLeft : ℚ := 5 / 3

theorem lily_milk_remaining : lilyInitialMilk - milkGivenAway = milkLeft := by
  sorry

end lily_milk_remaining_l242_242213


namespace calculate_large_exponent_l242_242109

theorem calculate_large_exponent : (1307 * 1307)^3 = 4984209203082045649 :=
by {
   sorry
}

end calculate_large_exponent_l242_242109


namespace taller_cycle_shadow_length_l242_242378

theorem taller_cycle_shadow_length 
  (h_taller : ℝ) (h_shorter : ℝ) (shadow_shorter : ℝ) (shadow_taller : ℝ) 
  (h_taller_val : h_taller = 2.5) 
  (h_shorter_val : h_shorter = 2) 
  (shadow_shorter_val : shadow_shorter = 4)
  (similar_triangles : h_taller / shadow_taller = h_shorter / shadow_shorter) :
  shadow_taller = 5 := 
by 
  sorry

end taller_cycle_shadow_length_l242_242378


namespace find_radius_of_sector_l242_242342

noncomputable def radius_of_sector (P : ℝ) (θ : ℝ) : ℝ :=
  P / (Real.pi + 2)

theorem find_radius_of_sector :
  radius_of_sector 144 180 = 144 / (Real.pi + 2) :=
by
  unfold radius_of_sector
  sorry

end find_radius_of_sector_l242_242342


namespace find_A_l242_242566

noncomputable def A_value (A B C : ℝ) := (A = 1/4) 

theorem find_A : 
  ∀ (A B C : ℝ),
  (∀ x : ℝ, x ≠ 1 → x ≠ 3 → (1 / (x^3 - 3*x^2 - 13*x + 15) = A / (x - 1) + B / (x - 3) + C / (x - 3)^2)) →
  A_value A B C :=
by 
  sorry

end find_A_l242_242566


namespace find_two_digit_number_l242_242145

theorem find_two_digit_number : 
  ∃ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ (b = 0 ∨ b = 5) ∧ (10 * a + b = 5 * (a + b)) ∧ (10 * a + b = 45) :=
by
  sorry

end find_two_digit_number_l242_242145


namespace cylinder_surface_area_and_volume_l242_242387

noncomputable def cylinder_total_surface_area (r h : ℝ) : ℝ :=
  2 * Real.pi * r^2 + 2 * Real.pi * r * h

noncomputable def cylinder_volume (r h : ℝ) : ℝ :=
  Real.pi * r^2 * h

theorem cylinder_surface_area_and_volume (r h : ℝ) (hr : r = 5) (hh : h = 15) :
  cylinder_total_surface_area r h = 200 * Real.pi ∧ cylinder_volume r h = 375 * Real.pi :=
by
  sorry -- Proof omitted

end cylinder_surface_area_and_volume_l242_242387


namespace arccos_cos_three_l242_242127

-- Defining the problem conditions
def three_radians : ℝ := 3

-- Main statement to prove
theorem arccos_cos_three : Real.arccos (Real.cos three_radians) = three_radians := 
sorry

end arccos_cos_three_l242_242127


namespace find_some_value_l242_242369

theorem find_some_value (m n : ℝ) (some_value : ℝ) 
  (h₁ : m = n / 2 - 2 / 5)
  (h₂ : m + 2 = (n + some_value) / 2 - 2 / 5) :
  some_value = 4 := 
sorry

end find_some_value_l242_242369


namespace product_lcm_gcd_l242_242525

def a : ℕ := 6
def b : ℕ := 8

theorem product_lcm_gcd : Nat.lcm a b * Nat.gcd a b = 48 := by
  sorry

end product_lcm_gcd_l242_242525


namespace edge_length_of_cube_l242_242262

theorem edge_length_of_cube (total_cubes : ℕ) (box_edge_length_m : ℝ) (box_edge_length_cm : ℝ) 
  (conversion_factor : ℝ) (edge_length_cm : ℝ) : 
  total_cubes = 8 ∧ box_edge_length_m = 1 ∧ box_edge_length_cm = box_edge_length_m * conversion_factor ∧ conversion_factor = 100 ∧ 
  edge_length_cm = box_edge_length_cm / 2 ↔ edge_length_cm = 50 := 
by 
  sorry

end edge_length_of_cube_l242_242262


namespace kw_price_approx_4266_percent_l242_242019

noncomputable def kw_price_percentage (A B C D E : ℝ) (hA : KW = 1.5 * A) (hB : KW = 2 * B) (hC : KW = 2.5 * C) (hD : KW = 2.25 * D) (hE : KW = 3 * E) : ℝ :=
  let total_assets := A + B + C + D + E
  let price_kw := 1.5 * A
  (price_kw / total_assets) * 100

theorem kw_price_approx_4266_percent (A B C D E KW : ℝ)
  (hA : KW = 1.5 * A) (hB : KW = 2 * B) (hC : KW = 2.5 * C) (hD : KW = 2.25 * D) (hE : KW = 3 * E)
  (hB_from_A : B = 0.75 * A) (hC_from_A : C = 0.6 * A) (hD_from_A : D = 0.6667 * A) (hE_from_A : E = 0.5 * A) :
  abs ((kw_price_percentage A B C D E hA hB hC hD hE) - 42.66) < 1 :=
by sorry

end kw_price_approx_4266_percent_l242_242019


namespace sum_of_quotient_and_reciprocal_l242_242352

theorem sum_of_quotient_and_reciprocal (x y : ℝ) (h1 : x + y = 45) (h2 : x * y = 500) : 
    (x / y + y / x) = 41 / 20 := 
sorry

end sum_of_quotient_and_reciprocal_l242_242352


namespace email_count_first_day_l242_242456

theorem email_count_first_day (E : ℕ) 
  (h1 : ∃ E, E + E / 2 + E / 4 + E / 8 = 30) : E = 16 :=
by
  sorry

end email_count_first_day_l242_242456


namespace intersection_result_l242_242762

noncomputable def A : Set ℝ := { x | x^2 - 5*x - 6 < 0 }
noncomputable def B : Set ℝ := { x | 2022^x > Real.sqrt 2022 }
noncomputable def intersection : Set ℝ := { x | A x ∧ B x }

theorem intersection_result : intersection = Set.Ioo (1/2 : ℝ) 6 := by
  sorry

end intersection_result_l242_242762


namespace sum_of_fractions_l242_242781

variable {a : ℝ}

theorem sum_of_fractions (h : a ≠ 0) : (3 / a + 2 / a) = 5 / a := 
by sorry

end sum_of_fractions_l242_242781


namespace sin_15_cos_15_l242_242594

theorem sin_15_cos_15 : (Real.sin (15 * Real.pi / 180)) * (Real.cos (15 * Real.pi / 180)) = 1 / 4 := by
  sorry

end sin_15_cos_15_l242_242594


namespace solve_for_y_l242_242282

theorem solve_for_y (x y : ℚ) (h₁ : x - y = 12) (h₂ : 2 * x + y = 10) : y = -14 / 3 :=
by
  sorry

end solve_for_y_l242_242282


namespace present_age_of_son_l242_242147

variable (S M : ℕ)

-- Conditions
def condition1 : Prop := M = S + 32
def condition2 : Prop := M + 2 = 2 * (S + 2)

-- Theorem stating the required proof
theorem present_age_of_son : condition1 S M ∧ condition2 S M → S = 30 := by
  sorry

end present_age_of_son_l242_242147


namespace average_of_added_numbers_l242_242286

theorem average_of_added_numbers (sum_twelve : ℕ) (new_sum : ℕ) (x y z : ℕ) 
  (h_sum_twelve : sum_twelve = 12 * 45) 
  (h_new_sum : new_sum = 15 * 60) 
  (h_addition : x + y + z = new_sum - sum_twelve) : 
  (x + y + z) / 3 = 120 :=
by 
  sorry

end average_of_added_numbers_l242_242286


namespace first_group_men_l242_242455

theorem first_group_men (x : ℕ) (days1 days2 : ℝ) (men2 : ℕ) (h1 : days1 = 25) (h2 : days2 = 17.5) (h3 : men2 = 20) (h4 : x * days1 = men2 * days2) : x = 14 := 
by
  sorry

end first_group_men_l242_242455


namespace number_of_regular_pencils_l242_242746

def cost_eraser : ℝ := 0.8
def cost_regular : ℝ := 0.5
def cost_short : ℝ := 0.4
def num_eraser : ℕ := 200
def num_short : ℕ := 35
def total_revenue : ℝ := 194

theorem number_of_regular_pencils (num_regular : ℕ) :
  (num_eraser * cost_eraser) + (num_short * cost_short) + (num_regular * cost_regular) = total_revenue → 
  num_regular = 40 :=
by
  sorry

end number_of_regular_pencils_l242_242746


namespace num_trains_encountered_l242_242675

noncomputable def train_travel_encounters : ℕ := 5

theorem num_trains_encountered (start_time : ℕ) (duration : ℕ) (daily_departure : ℕ) 
  (train_journey_duration : ℕ) (daily_start_interval : ℕ) 
  (end_time : ℕ) (number_encountered : ℕ) :
  (train_journey_duration = 3 * 24 * 60 + 30) → -- 3 days and 30 minutes in minutes
  (daily_start_interval = 24 * 60) →             -- interval between daily train starts (in minutes)
  (number_encountered = 5) :=
by
  sorry

end num_trains_encountered_l242_242675


namespace liangliang_speed_l242_242298

theorem liangliang_speed (d_initial : ℝ) (t : ℝ) (d_final : ℝ) (v_mingming : ℝ) (v_liangliang : ℝ) :
  d_initial = 3000 →
  t = 20 →
  d_final = 2900 →
  v_mingming = 80 →
  (v_liangliang = 85 ∨ v_liangliang = 75) :=
by
  sorry

end liangliang_speed_l242_242298


namespace linear_function_no_first_quadrant_l242_242113

theorem linear_function_no_first_quadrant : 
  ¬ ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ y = -3 * x - 2 := by
  sorry

end linear_function_no_first_quadrant_l242_242113


namespace max_sum_factors_of_60_exists_max_sum_factors_of_60_l242_242783

theorem max_sum_factors_of_60 (d Δ : ℕ) (h : d * Δ = 60) : (d + Δ) ≤ 61 :=
sorry

theorem exists_max_sum_factors_of_60 : ∃ d Δ : ℕ, d * Δ = 60 ∧ d + Δ = 61 :=
sorry

end max_sum_factors_of_60_exists_max_sum_factors_of_60_l242_242783


namespace sqrt_x_plus_inv_sqrt_x_eq_sqrt_52_l242_242332

variable (x : ℝ) (h1 : x > 0) (h2 : x + 1/x = 50)

theorem sqrt_x_plus_inv_sqrt_x_eq_sqrt_52 : (Real.sqrt x + 1 / Real.sqrt x = Real.sqrt 52) :=
by
  sorry

end sqrt_x_plus_inv_sqrt_x_eq_sqrt_52_l242_242332


namespace cos_x_plus_2y_eq_one_l242_242218

theorem cos_x_plus_2y_eq_one (x y a : ℝ) 
  (hx : -Real.pi / 4 ≤ x ∧ x ≤ Real.pi / 4)
  (hy : -Real.pi / 4 ≤ y ∧ y ≤ Real.pi / 4)
  (h_eq1 : x^3 + Real.sin x - 2 * a = 0)
  (h_eq2 : 4 * y^3 + (1 / 2) * Real.sin (2 * y) + a = 0) : 
  Real.cos (x + 2 * y) = 1 := 
sorry -- Proof goes here

end cos_x_plus_2y_eq_one_l242_242218


namespace product_pass_rate_l242_242680

variable {a b : ℝ} (h_a : 0 ≤ a ∧ a ≤ 1) (h_b : 0 ≤ b ∧ b ≤ 1) (h_indep : true)

theorem product_pass_rate : (1 - a) * (1 - b) = 
((1 - a) * (1 - b)) :=
by
  sorry

end product_pass_rate_l242_242680


namespace inequality_1_inequality_2_inequality_3_inequality_4_l242_242257

-- Definitions of distances
def d_a : ℝ := sorry
def d_b : ℝ := sorry
def d_c : ℝ := sorry
def R_a : ℝ := sorry
def R_b : ℝ := sorry
def R_c : ℝ := sorry
def a : ℝ := sorry
def b : ℝ := sorry
def c : ℝ := sorry

def R : ℝ := sorry -- Circumradius
def r : ℝ := sorry -- Inradius

-- Inequality 1
theorem inequality_1 : a * R_a ≥ c * d_c + b * d_b := 
  sorry

-- Inequality 2
theorem inequality_2 : d_a * R_a + d_b * R_b + d_c * R_c ≥ 2 * (d_a * d_b + d_b * d_c + d_c * d_a) :=
  sorry

-- Inequality 3
theorem inequality_3 : R_a + R_b + R_c ≥ 2 * (d_a + d_b + d_c) :=
  sorry

-- Inequality 4
theorem inequality_4 : R_a * R_b * R_c ≥ (R / (2 * r)) * (d_a + d_b) * (d_b + d_c) * (d_c + d_a) :=
  sorry

end inequality_1_inequality_2_inequality_3_inequality_4_l242_242257


namespace calculate_value_expression_l242_242030

theorem calculate_value_expression :
  3000 * (3000 ^ 3000 + 3000 ^ 2999) = 3001 * 3000 ^ 3000 := 
by
  sorry

end calculate_value_expression_l242_242030


namespace length_real_axis_l242_242419

theorem length_real_axis (x y : ℝ) : 
  (x^2 / 4 - y^2 / 12 = 1) → 4 = 4 :=
by
  intro h
  sorry

end length_real_axis_l242_242419


namespace simplify_expression_l242_242296

theorem simplify_expression (z y : ℝ) :
  (4 - 5 * z + 2 * y) - (6 + 7 * z - 3 * y) = -2 - 12 * z + 5 * y :=
by
  sorry

end simplify_expression_l242_242296


namespace set_union_eq_l242_242822

open Set

noncomputable def A : Set ℤ := {x | x^2 - x = 0}
def B : Set ℤ := {-1, 0}
def C : Set ℤ := {-1, 0, 1}

theorem set_union_eq :
  A ∪ B = C :=
by {
  sorry
}

end set_union_eq_l242_242822


namespace smallest_two_digit_product_12_l242_242080

theorem smallest_two_digit_product_12 : 
  ∃ (n : ℕ), (10 ≤ n ∧ n < 100) ∧ (∃ (a b : ℕ), (1 ≤ a ∧ a < 10) ∧ (1 ≤ b ∧ b < 10) ∧ (a * b = 12) ∧ (n = 10 * a + b)) ∧
  (∀ m : ℕ, (10 ≤ m ∧ m < 100) ∧ (∃ (c d : ℕ), (1 ≤ c ∧ c < 10) ∧ (1 ≤ d ∧ d < 10) ∧ (c * d = 12) ∧ (m = 10 * c + d)) → n ≤ m) ↔ n = 26 :=
by
  sorry

end smallest_two_digit_product_12_l242_242080


namespace solve_inequalities_l242_242088

theorem solve_inequalities {x : ℝ} :
  (3 * x + 1) / 2 > x ∧ (4 * (x - 2) ≤ x - 5) ↔ (-1 < x ∧ x ≤ 1) :=
by sorry

end solve_inequalities_l242_242088


namespace quadratic_inequality_solution_l242_242062

theorem quadratic_inequality_solution (m : ℝ) :
  {x : ℝ | (x - m) * (x - (m + 1)) > 0} = {x | x < m ∨ x > m + 1} := sorry

end quadratic_inequality_solution_l242_242062


namespace depth_of_water_in_cistern_l242_242691

-- Define the given constants
def length_cistern : ℝ := 6
def width_cistern : ℝ := 5
def total_wet_area : ℝ := 57.5

-- Define the area of the bottom of the cistern
def area_bottom (length : ℝ) (width : ℝ) : ℝ := length * width

-- Define the area of the longer sides of the cistern in contact with water
def area_long_sides (length : ℝ) (depth : ℝ) : ℝ := 2 * length * depth

-- Define the area of the shorter sides of the cistern in contact with water
def area_short_sides (width : ℝ) (depth : ℝ) : ℝ := 2 * width * depth

-- Define the total wet surface area based on depth of the water
def total_wet_surface_area (length : ℝ) (width : ℝ) (depth : ℝ) : ℝ := 
    area_bottom length width + area_long_sides length depth + area_short_sides width depth

-- Define the proof statement
theorem depth_of_water_in_cistern : ∃ h : ℝ, h = 1.25 ∧ total_wet_surface_area length_cistern width_cistern h = total_wet_area := 
by
  use 1.25
  sorry

end depth_of_water_in_cistern_l242_242691


namespace inequality_always_true_l242_242304

theorem inequality_always_true (a b : ℝ) : a^2 + b^2 ≥ -2 * a * b :=
by sorry

end inequality_always_true_l242_242304


namespace min_x_prime_factorization_sum_eq_31_l242_242314

theorem min_x_prime_factorization_sum_eq_31
    (x y a b c d : ℕ)
    (hx : x > 0)
    (hy : y > 0)
    (h : 7 * x^5 = 11 * y^13)
    (hx_prime_fact : ∃ a c b d : ℕ, x = a^c * b^d) :
    a + b + c + d = 31 :=
by
 sorry
 
end min_x_prime_factorization_sum_eq_31_l242_242314


namespace avg_waiting_time_l242_242441

theorem avg_waiting_time : 
  let P_G := 1 / 3      -- Probability of green light
  let P_red := 2 / 3    -- Probability of red light
  let E_T_given_G := 0  -- Expected time given green light
  let E_T_given_red := 1 -- Expected time given red light
  (E_T_given_G * P_G) + (E_T_given_red * P_red) = 2 / 3
:= by
  sorry

end avg_waiting_time_l242_242441


namespace meaningful_expression_l242_242203

theorem meaningful_expression (x : ℝ) : (∃ y : ℝ, y = 1 / (Real.sqrt (x - 1))) → x > 1 :=
by sorry

end meaningful_expression_l242_242203


namespace milan_long_distance_bill_l242_242760

theorem milan_long_distance_bill
  (monthly_fee : ℝ := 2)
  (per_minute_cost : ℝ := 0.12)
  (minutes_used : ℕ := 178) :
  ((minutes_used : ℝ) * per_minute_cost + monthly_fee = 23.36) :=
by
  sorry

end milan_long_distance_bill_l242_242760


namespace blue_marbles_l242_242647

theorem blue_marbles (r b : ℕ) (h_ratio : 3 * b = 5 * r) (h_red : r = 18) : b = 30 := by
  -- proof
  sorry

end blue_marbles_l242_242647


namespace subtraction_of_tenths_l242_242229

theorem subtraction_of_tenths (a b : ℝ) (n : ℕ) (h1 : a = (1 / 10) * 6000) (h2 : b = (1 / 10 / 100) * 6000) : (a - b) = 594 := by
sorry

end subtraction_of_tenths_l242_242229


namespace find_sister_candy_l242_242528

/-- Define Katie's initial amount of candy -/
def Katie_candy : ℕ := 10

/-- Define the amount of candy eaten the first night -/
def eaten_candy : ℕ := 9

/-- Define the amount of candy left after the first night -/
def remaining_candy : ℕ := 7

/-- Define the number of candies Katie's sister had -/
def sister_candy (S : ℕ) : Prop :=
  Katie_candy + S - eaten_candy = remaining_candy

/-- Theorem stating that Katie's sister had 6 pieces of candy -/
theorem find_sister_candy : ∃ S, sister_candy S ∧ S = 6 :=
by
  sorry

end find_sister_candy_l242_242528


namespace find_arithmetic_sequence_l242_242095

theorem find_arithmetic_sequence (a d : ℝ) : 
(a - d) + a + (a + d) = 6 ∧ (a - d) * a * (a + d) = -10 → 
  (a = 2 ∧ d = 3 ∨ a = 2 ∧ d = -3) :=
by
  sorry

end find_arithmetic_sequence_l242_242095


namespace remainder_3_pow_89_plus_5_mod_7_l242_242052

theorem remainder_3_pow_89_plus_5_mod_7 :
  (3^1 % 7 = 3) ∧ (3^2 % 7 = 2) ∧ (3^3 % 7 = 6) ∧ (3^4 % 7 = 4) ∧ (3^5 % 7 = 5) ∧ (3^6 % 7 = 1) →
  ((3^89 + 5) % 7 = 3) :=
by
  intros h
  sorry

end remainder_3_pow_89_plus_5_mod_7_l242_242052


namespace relationship_between_a_and_b_l242_242473

variable (a b : ℝ)

def in_interval (x : ℝ) := 0 < x ∧ x < 1

theorem relationship_between_a_and_b 
  (ha : in_interval a)
  (hb : in_interval b)
  (h : (1 - a) * b > 1 / 4) : a < b :=
sorry

end relationship_between_a_and_b_l242_242473


namespace eliza_received_12_almonds_l242_242790

theorem eliza_received_12_almonds (y : ℕ) (h1 : y - 8 = y / 3) : y = 12 :=
sorry

end eliza_received_12_almonds_l242_242790


namespace Bridget_Skittles_Final_l242_242048

def Bridget_Skittles_initial : Nat := 4
def Henry_Skittles_initial : Nat := 4

def Bridget_Receives_Henry_Skittles : Nat :=
  Bridget_Skittles_initial + Henry_Skittles_initial

theorem Bridget_Skittles_Final :
  Bridget_Receives_Henry_Skittles = 8 :=
by
  -- Proof steps here, assuming sorry for now
  sorry

end Bridget_Skittles_Final_l242_242048


namespace sum_a_b_eq_4_l242_242131

-- Define the problem conditions
variables (a b : ℝ)

-- State the conditions
def condition1 : Prop := 2 * a = 8
def condition2 : Prop := a^2 - b = 16

-- State the theorem
theorem sum_a_b_eq_4 (h1 : condition1 a) (h2 : condition2 a b) : a + b = 4 :=
by sorry

end sum_a_b_eq_4_l242_242131


namespace parabola_intersection_radius_sqr_l242_242276

theorem parabola_intersection_radius_sqr {x y : ℝ} :
  (y = (x - 2)^2) →
  (x - 3 = (y + 2)^2) →
  ∃ r, r^2 = 9 / 2 :=
by
  intros h1 h2
  sorry

end parabola_intersection_radius_sqr_l242_242276


namespace raja_monthly_income_l242_242524

theorem raja_monthly_income (X : ℝ) 
  (h1 : 0.1 * X = 5000) : X = 50000 :=
sorry

end raja_monthly_income_l242_242524


namespace total_tomatoes_l242_242563

def tomatoes_first_plant : Nat := 2 * 12
def tomatoes_second_plant : Nat := (tomatoes_first_plant / 2) + 5
def tomatoes_third_plant : Nat := tomatoes_second_plant + 2

theorem total_tomatoes :
  (tomatoes_first_plant + tomatoes_second_plant + tomatoes_third_plant) = 60 := by
  sorry

end total_tomatoes_l242_242563


namespace leila_yards_l242_242758

variable (mile_yards : ℕ := 1760)
variable (marathon_miles : ℕ := 28)
variable (marathon_yards : ℕ := 1500)
variable (marathons_ran : ℕ := 15)

theorem leila_yards (m y : ℕ) (h1 : marathon_miles = 28) (h2 : marathon_yards = 1500) (h3 : mile_yards = 1760) (h4 : marathons_ran = 15) (hy : 0 ≤ y ∧ y < mile_yards) :
  y = 1200 :=
sorry

end leila_yards_l242_242758


namespace sixty_five_percent_of_40_minus_four_fifths_of_25_l242_242546

theorem sixty_five_percent_of_40_minus_four_fifths_of_25 : 
  (0.65 * 40) - (0.8 * 25) = 6 := 
by
  sorry

end sixty_five_percent_of_40_minus_four_fifths_of_25_l242_242546


namespace part1_part2_l242_242513

theorem part1 (m n : ℕ) (h1 : m > n) (h2 : Nat.gcd m n + Nat.lcm m n = m + n) : n ∣ m := 
sorry

theorem part2 (m n : ℕ) (h1 : m > n) (h2 : Nat.gcd m n + Nat.lcm m n = m + n)
(h3 : m - n = 10) : (m, n) = (11, 1) ∨ (m, n) = (12, 2) ∨ (m, n) = (15, 5) ∨ (m, n) = (20, 10) := 
sorry

end part1_part2_l242_242513


namespace log_one_third_nine_l242_242687

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log_one_third_nine : log_base (1/3) 9 = -2 := by
  sorry

end log_one_third_nine_l242_242687


namespace incorrect_value_in_polynomial_progression_l242_242346

noncomputable def polynomial_values (x : ℕ) : ℕ :=
  match x with
  | 0 => 1
  | 1 => 9
  | 2 => 35
  | 3 => 99
  | 4 => 225
  | 5 => 441
  | 6 => 784
  | 7 => 1296
  | _ => 0  -- This is a dummy value just to complete the function

theorem incorrect_value_in_polynomial_progression :
  ¬ (∃ (a b c d : ℝ), ∀ x : ℕ,
    polynomial_values x = (a * x ^ 3 + b * x ^ 2 + c * x + d + if x ≤ 7 then 0 else 1)) :=
by
  intro h
  sorry

end incorrect_value_in_polynomial_progression_l242_242346


namespace smallest_rel_prime_210_l242_242539

theorem smallest_rel_prime_210 : ∃ x : ℕ, x > 1 ∧ gcd x 210 = 1 ∧ ∀ y : ℕ, y > 1 ∧ gcd y 210 = 1 → x ≤ y := 
by
  sorry

end smallest_rel_prime_210_l242_242539


namespace no_three_distinct_nat_numbers_sum_prime_l242_242664

theorem no_three_distinct_nat_numbers_sum_prime:
  ¬∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 0 < a ∧ 0 < b ∧ 0 < c ∧ 
  Nat.Prime (a + b) ∧ Nat.Prime (a + c) ∧ Nat.Prime (b + c) := 
sorry

end no_three_distinct_nat_numbers_sum_prime_l242_242664


namespace second_number_is_30_l242_242251

theorem second_number_is_30 
  (A B C : ℝ)
  (h1 : A + B + C = 98)
  (h2 : A / B = 2 / 3)
  (h3 : B / C = 5 / 8) : 
  B = 30 :=
by
  sorry

end second_number_is_30_l242_242251


namespace fraction_product_l242_242058

theorem fraction_product : (1 / 4 : ℚ) * (2 / 5) * (3 / 6) = (1 / 20) := by
  sorry

end fraction_product_l242_242058


namespace interest_rate_proof_l242_242453

noncomputable def compound_interest_rate (P A : ℝ) (t n : ℕ) : ℝ :=
  (((A / P)^(1 / (n * t))) - 1) * n

theorem interest_rate_proof :
  ∀ P A : ℝ, ∀ t n : ℕ, P = 1093.75 → A = 1183 → t = 2 → n = 1 →
  compound_interest_rate P A t n = 0.0399 :=
by
  intros P A t n hP hA ht hn
  rw [hP, hA, ht, hn]
  unfold compound_interest_rate
  sorry

end interest_rate_proof_l242_242453


namespace circles_intersect_l242_242320

def circle1 : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 2}
def circle2 : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 + 4 * p.2 + 3 = 0}

theorem circles_intersect : ∃ (p : ℝ × ℝ), p ∈ circle1 ∧ p ∈ circle2 :=
by
  sorry

end circles_intersect_l242_242320


namespace total_interest_l242_242561

variable (P R : ℝ)

-- Given condition: Simple interest on sum of money is Rs. 700 after 10 years
def interest_10_years (P R : ℝ) : Prop := (P * R * 10) / 100 = 700

-- Principal is trebled after 5 years
def interest_5_years_treble (P R : ℝ) : Prop := (15 * P * R) / 100 = 105

-- The final interest is the sum of interest for the first 10 years and next 5 years post trebling the principal
theorem total_interest (P R : ℝ) (h1: interest_10_years P R) (h2: interest_5_years_treble P R) : 
  (700 + 105 = 805) := 
  by 
  sorry

end total_interest_l242_242561


namespace blue_dress_difference_l242_242820

theorem blue_dress_difference 
(total_space : ℕ)
(red_dresses : ℕ)
(blue_dresses : ℕ)
(h1 : total_space = 200)
(h2 : red_dresses = 83)
(h3 : blue_dresses = total_space - red_dresses) :
blue_dresses - red_dresses = 34 :=
by
  rw [h1, h2] at h3
  sorry -- Proof details go here.

end blue_dress_difference_l242_242820


namespace number_of_seven_banana_bunches_l242_242652

theorem number_of_seven_banana_bunches (total_bananas : ℕ) (eight_banana_bunches : ℕ) (seven_banana_bunches : ℕ) : 
    total_bananas = 83 → 
    eight_banana_bunches = 6 → 
    (∃ n : ℕ, seven_banana_bunches = n) → 
    8 * eight_banana_bunches + 7 * seven_banana_bunches = total_bananas → 
    seven_banana_bunches = 5 := by
  sorry

end number_of_seven_banana_bunches_l242_242652


namespace possible_rectangular_arrays_l242_242616

theorem possible_rectangular_arrays (n : ℕ) (h : n = 48) :
  ∃ (m k : ℕ), m * k = n ∧ 2 ≤ m ∧ 2 ≤ k :=
sorry

end possible_rectangular_arrays_l242_242616


namespace composite_function_properties_l242_242150

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem composite_function_properties
  (a b c : ℝ)
  (h_a_nonzero : a ≠ 0)
  (h_no_real_roots : ∀ x : ℝ, f a b c x ≠ x) :
  (∀ x : ℝ, f a b c (f a b c x) ≠ x) ∧
  (a > 0 → ∀ x : ℝ, f a b c (f a b c x) > x) :=
by sorry

end composite_function_properties_l242_242150


namespace event_A_muffins_correct_event_B_muffins_correct_event_C_muffins_correct_l242_242112

-- Event A
def total_muffins_needed_A := 200
def arthur_muffins_A := 35
def beatrice_muffins_A := 48
def charles_muffins_A := 29
def total_muffins_baked_A := arthur_muffins_A + beatrice_muffins_A + charles_muffins_A
def additional_muffins_needed_A := total_muffins_needed_A - total_muffins_baked_A

-- Event B
def total_muffins_needed_B := 150
def arthur_muffins_B := 20
def beatrice_muffins_B := 35
def charles_muffins_B := 25
def total_muffins_baked_B := arthur_muffins_B + beatrice_muffins_B + charles_muffins_B
def additional_muffins_needed_B := total_muffins_needed_B - total_muffins_baked_B

-- Event C
def total_muffins_needed_C := 250
def arthur_muffins_C := 45
def beatrice_muffins_C := 60
def charles_muffins_C := 30
def total_muffins_baked_C := arthur_muffins_C + beatrice_muffins_C + charles_muffins_C
def additional_muffins_needed_C := total_muffins_needed_C - total_muffins_baked_C

-- Proof Statements
theorem event_A_muffins_correct : additional_muffins_needed_A = 88 := by
  sorry

theorem event_B_muffins_correct : additional_muffins_needed_B = 70 := by
  sorry

theorem event_C_muffins_correct : additional_muffins_needed_C = 115 := by
  sorry

end event_A_muffins_correct_event_B_muffins_correct_event_C_muffins_correct_l242_242112


namespace find_divisor_l242_242140

theorem find_divisor {x y : ℤ} (h1 : (x - 5) / y = 7) (h2 : (x - 24) / 10 = 3) : y = 7 :=
by
  sorry

end find_divisor_l242_242140


namespace largest_multiple_of_7_whose_negation_greater_than_neg80_l242_242787

theorem largest_multiple_of_7_whose_negation_greater_than_neg80 : ∃ (n : ℤ), n = 77 ∧ (∃ (k : ℤ), n = k * 7) ∧ (-n > -80) :=
by
  sorry

end largest_multiple_of_7_whose_negation_greater_than_neg80_l242_242787


namespace sufficient_condition_for_equation_l242_242813

theorem sufficient_condition_for_equation (x y z : ℤ) (h1 : x = y + 1) (h2 : z = y) :
    x * (x - y) + y * (y - z) + z * (z - x) = 1 :=
by
  -- Proof omitted
  sorry

end sufficient_condition_for_equation_l242_242813


namespace alcohol_added_l242_242776

theorem alcohol_added (x : ℝ) :
  let initial_solution_volume := 40
  let initial_alcohol_percentage := 0.05
  let initial_alcohol_volume := initial_solution_volume * initial_alcohol_percentage
  let additional_water := 6.5
  let final_solution_volume := initial_solution_volume + x + additional_water
  let final_alcohol_percentage := 0.11
  let final_alcohol_volume := final_solution_volume * final_alcohol_percentage
  initial_alcohol_volume + x = final_alcohol_volume → x = 3.5 :=
by
  intros
  sorry

end alcohol_added_l242_242776


namespace translation_of_exponential_l242_242357

noncomputable def translated_function (a : ℝ × ℝ) (f : ℝ → ℝ) : ℝ → ℝ :=
  λ x => f (x - a.1) + a.2

theorem translation_of_exponential :
  translated_function (2, 3) (λ x => Real.exp x) = λ x => Real.exp (x - 2) + 3 :=
by
  sorry

end translation_of_exponential_l242_242357


namespace find_dimensions_l242_242260

def is_solution (m n r : ℕ) : Prop :=
  ∃ k0 k1 k2 : ℕ, 
    k0 = (m - 2) * (n - 2) * (r - 2) ∧
    k1 = 2 * ((m - 2) * (n - 2) + (n - 2) * (r - 2) + (r - 2) * (m - 2)) ∧
    k2 = 4 * ((m - 2) + (n - 2) + (r - 2)) ∧
    k0 + k2 - k1 = 1985

theorem find_dimensions (m n r : ℕ) (h : m ≤ n ∧ n ≤ r) (hp : 0 < m ∧ 0 < n ∧ 0 < r) : 
  is_solution m n r :=
sorry

end find_dimensions_l242_242260


namespace smallest_b_l242_242020

theorem smallest_b
  (a b : ℕ)
  (h_pos : 0 < b)
  (h_diff : a - b = 8)
  (h_gcd : Nat.gcd ((a^3 + b^3) / (a + b)) (a * b) = 16) :
  b = 4 := sorry

end smallest_b_l242_242020


namespace min_value_of_x_l242_242645

open Real

-- Defining the conditions
def condition1 (x : ℝ) : Prop := x > 0
def condition2 (x : ℝ) : Prop := log x ≥ 2 * log 3 + (1/3) * log x

-- Statement of the theorem
theorem min_value_of_x (x : ℝ) (h1 : condition1 x) (h2 : condition2 x) : x ≥ 27 :=
sorry

end min_value_of_x_l242_242645


namespace find_a_l242_242445

theorem find_a (a : ℝ) 
  (line_through : ∃ (p1 p2 : ℝ × ℝ), p1 = (a-2, -1) ∧ p2 = (-a-2, 1)) 
  (perpendicular : ∀ (l1 l2 : ℝ × ℝ), l1 = (2, 3) → l2 = (-1/a, 1) → false) : 
  a = -2/3 :=
by 
  sorry

end find_a_l242_242445


namespace arithmetic_sequence_a11_l242_242354

theorem arithmetic_sequence_a11 (a : ℕ → ℕ) 
  (h1 : a 1 = 1)
  (h2 : ∀ n, a (n + 2) - a n = 6) : 
  a 11 = 31 := 
sorry

end arithmetic_sequence_a11_l242_242354


namespace total_cookies_l242_242137

-- Conditions
def Paul_cookies : ℕ := 45
def Paula_cookies : ℕ := Paul_cookies - 3

-- Question and Answer
theorem total_cookies : Paul_cookies + Paula_cookies = 87 := by
  sorry

end total_cookies_l242_242137


namespace steve_speed_ratio_l242_242148

/-- Define the distance from Steve's house to work. -/
def distance_to_work := 30

/-- Define the total time spent on the road by Steve. -/
def total_time_on_road := 6

/-- Define Steve's speed on the way back from work. -/
def speed_back := 15

/-- Calculate the ratio of Steve's speed on the way back to his speed on the way to work. -/
theorem steve_speed_ratio (v : ℝ) (h_v_pos : v > 0) 
    (h1 : distance_to_work / v + distance_to_work / speed_back = total_time_on_road) :
    speed_back / v = 2 := 
by
  -- We will provide the proof here
  sorry

end steve_speed_ratio_l242_242148


namespace range_of_m_l242_242512

def f (m x : ℝ) : ℝ := 2 * x^3 - 3 * m * x^2 + 6 * x

def f_derivative_nonnegative_on_interval (m : ℝ) : Prop :=
  ∀ x : ℝ, 1 < x → 6 * x^2 - 6 * m * x + 6 ≥ 0

theorem range_of_m (m : ℝ) : f_derivative_nonnegative_on_interval m ↔ m ≤ 2 :=
by
  sorry

end range_of_m_l242_242512


namespace rectangle_to_cylinder_max_volume_ratio_l242_242623

/-- Given a rectangle with a perimeter of 12 and converting it into a cylinder 
with the height being the same as the width of the rectangle, prove that the 
ratio of the circumference of the cylinder's base to its height when the volume 
is maximized is 2:1. -/
theorem rectangle_to_cylinder_max_volume_ratio : 
  ∃ (x : ℝ), (2 * x + 2 * (6 - x)) = 12 → 2 * (6 - x) / x = 2 :=
sorry

end rectangle_to_cylinder_max_volume_ratio_l242_242623


namespace arithmetic_sequence_value_l242_242611

theorem arithmetic_sequence_value (a : ℝ) 
  (h1 : 2 * (2 * a + 1) = (a - 1) + (a + 4)) : a = 1 / 2 := 
by 
  sorry

end arithmetic_sequence_value_l242_242611


namespace probability_of_triangle_segments_from_15gon_l242_242082

/-- A proof problem that calculates the probability that three randomly selected segments 
    from a regular 15-gon inscribed in a circle form a triangle with positive area. -/
theorem probability_of_triangle_segments_from_15gon : 
  let n := 15
  let total_segments := (n * (n - 1)) / 2 
  let total_combinations := total_segments * (total_segments - 1) * (total_segments - 2) / 6 
  let valid_probability := 943 / 1365
  valid_probability = (total_combinations - count_violating_combinations) / total_combinations :=
sorry

end probability_of_triangle_segments_from_15gon_l242_242082


namespace solve_for_a_l242_242766

theorem solve_for_a (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b)
(h_eq_exponents : a ^ b = b ^ a) (h_b_equals_3a : b = 3 * a) : a = Real.sqrt 3 :=
sorry

end solve_for_a_l242_242766


namespace value_of_m_l242_242186

noncomputable def f (x a b : ℝ) : ℝ := x^3 + a * x^2 + b * x

theorem value_of_m (a b m : ℝ) (h₀ : m ≠ 0)
  (h₁ : 3 * m^2 + 2 * a * m + b = 0)
  (h₂ : m^2 + a * m + b = 0)
  (h₃ : ∃ x, f x a b = 1/2) :
  m = 3/2 :=
by
  sorry

end value_of_m_l242_242186


namespace price_before_tax_l242_242111

theorem price_before_tax (P : ℝ) (h : 1.15 * P = 1955) : P = 1700 :=
by sorry

end price_before_tax_l242_242111


namespace smithtown_left_handed_women_percentage_l242_242139

theorem smithtown_left_handed_women_percentage
    (x y : ℕ)
    (H1 : 3 * x + x = 4 * x)
    (H2 : 3 * y + 2 * y = 5 * y)
    (H3 : 4 * x = 5 * y) :
    (x / (4 * x)) * 100 = 25 :=
by sorry

end smithtown_left_handed_women_percentage_l242_242139


namespace prob_simultaneous_sequences_l242_242134

-- Definitions for coin probabilities
def prob_heads_A : ℝ := 0.3
def prob_tails_A : ℝ := 0.7
def prob_heads_B : ℝ := 0.4
def prob_tails_B : ℝ := 0.6

-- Definitions for required sequences
def seq_TTH_A : ℝ := prob_tails_A * prob_tails_A * prob_heads_A
def seq_HTT_B : ℝ := prob_heads_B * prob_tails_B * prob_tails_B

-- Main assertion
theorem prob_simultaneous_sequences :
  seq_TTH_A * seq_HTT_B = 0.021168 :=
by
  sorry

end prob_simultaneous_sequences_l242_242134


namespace train_length_l242_242699

theorem train_length (speed_kmph : ℕ) (time_seconds : ℕ) (length_meters : ℕ)
  (h1 : speed_kmph = 72)
  (h2 : time_seconds = 14)
  (h3 : length_meters = speed_kmph * 1000 * time_seconds / 3600)
  : length_meters = 280 := by
  sorry

end train_length_l242_242699


namespace smallest_even_number_of_seven_l242_242371

-- Conditions: The sum of seven consecutive even numbers is 700.
-- We need to prove that the smallest of these numbers is 94.

theorem smallest_even_number_of_seven (n : ℕ) (hn : 7 * n = 700) :
  ∃ (a b c d e f g : ℕ), 
  (2 * a + 4 * b + 6 * c + 8 * d + 10 * e + 12 * f + 14 * g = 700) ∧ 
  (a = b - 1) ∧ (b = c - 1) ∧ (c = d - 1) ∧ (d = e - 1) ∧ (e = f - 1) ∧ 
  (f = g - 1) ∧ (g = 100) ∧ (a = 94) :=
by
  -- This is the theorem statement. 
  sorry

end smallest_even_number_of_seven_l242_242371


namespace principal_amount_correct_l242_242330

noncomputable def initial_amount (A : ℝ) (R : ℝ) (T : ℝ) : ℝ :=
  (A * 100) / (R * T + 100)

theorem principal_amount_correct : initial_amount 950 9.230769230769232 5 = 650 := by
  sorry

end principal_amount_correct_l242_242330


namespace triangle_y_values_l242_242457

theorem triangle_y_values (y : ℕ) :
  (8 + 11 > y^2) ∧ (y^2 + 8 > 11) ∧ (y^2 + 11 > 8) ↔ y = 2 ∨ y = 3 ∨ y = 4 :=
by
  sorry

end triangle_y_values_l242_242457


namespace profit_23_percent_of_cost_price_l242_242297

-- Define the conditions
variable (C : ℝ) -- Cost price of the turtleneck sweaters
variable (C_nonneg : 0 ≤ C) -- Ensure cost price is non-negative

-- Definitions based on conditions
def SP1 (C : ℝ) : ℝ := 1.20 * C
def SP2 (SP1 : ℝ) : ℝ := 1.25 * SP1
def SPF (SP2 : ℝ) : ℝ := 0.82 * SP2

-- Define the profit calculation
def Profit (C : ℝ) : ℝ := (SPF (SP2 (SP1 C))) - C

-- Statement of the theorem
theorem profit_23_percent_of_cost_price (C : ℝ) (C_nonneg : 0 ≤ C):
  Profit C = 0.23 * C :=
by
  -- The actual proof would go here
  sorry

end profit_23_percent_of_cost_price_l242_242297


namespace traders_fabric_sales_l242_242061

theorem traders_fabric_sales (x y : ℕ) : 
  x + y = 85 ∧
  x = y + 5 ∧
  60 = x * (60 / y) ∧
  30 = y * (30 / x) →
  (x, y) = (25, 20) :=
by {
  sorry
}

end traders_fabric_sales_l242_242061


namespace find_x_l242_242740

def sum_sequence (a b : ℕ) : ℕ :=
  (b * (2 * a + b - 1)) / 2  -- Sum of an arithmetic progression

theorem find_x (x : ℕ) (h1 : sum_sequence x 10 = 65) : x = 2 :=
by {
  -- the proof goes here
  sorry
}

end find_x_l242_242740


namespace centroid_sum_of_squares_l242_242110

theorem centroid_sum_of_squares (a b c : ℝ) 
  (h : 1 / Real.sqrt (1 / a^2 + 1 / b^2 + 1 / c^2) = 2) : 
  1 / (a / 3) ^ 2 + 1 / (b / 3) ^ 2 + 1 / (c / 3) ^ 2 = 9 / 4 :=
by
  sorry

end centroid_sum_of_squares_l242_242110


namespace y_intercept_of_line_l242_242491

theorem y_intercept_of_line (x y : ℝ) (h : 4 * x + 6 * y = 24) : y = 4 := by
  sorry

end y_intercept_of_line_l242_242491


namespace no_positive_integer_satisfies_l242_242782

theorem no_positive_integer_satisfies : ¬ ∃ n : ℕ, 0 < n ∧ (20 * n + 2) ∣ (2003 * n + 2002) :=
by sorry

end no_positive_integer_satisfies_l242_242782


namespace distance_A_B_l242_242784

theorem distance_A_B 
  (perimeter_small_square : ℝ)
  (area_large_square : ℝ)
  (h1 : perimeter_small_square = 8)
  (h2 : area_large_square = 64) :
  let side_small_square := perimeter_small_square / 4
  let side_large_square := Real.sqrt area_large_square
  let horizontal_distance := side_small_square + side_large_square
  let vertical_distance := side_large_square - side_small_square
  let distance_AB := Real.sqrt (horizontal_distance^2 + vertical_distance^2)
  distance_AB = 11.7 :=
  by sorry

end distance_A_B_l242_242784


namespace range_of_m_l242_242078

noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  x^3 - 6 * x^2 + 9 * x + m

theorem range_of_m (m : ℝ) :
  (∃ a b c : ℝ, a < b ∧ b < c ∧ f m a = 0 ∧ f m b = 0 ∧ f m c = 0) ↔ -4 < m ∧ m < 0 :=
by
  sorry

end range_of_m_l242_242078


namespace average_of_seven_consecutive_l242_242618

theorem average_of_seven_consecutive (
  a : ℤ 
  ) (c : ℤ) 
  (h1 : c = (a + 1 + a + 2 + a + 3 + a + 4 + a + 5 + a + 6 + a + 7) / 7) : 
  (c + (c + 1) + (c + 2) + (c + 3) + (c + 4) + (c + 5) + (c + 6)) / 7 = a + 7 := 
by 
  sorry

end average_of_seven_consecutive_l242_242618


namespace cockatiel_weekly_consumption_is_50_l242_242824

def boxes_bought : ℕ := 3
def boxes_existing : ℕ := 5
def grams_per_box : ℕ := 225
def parrot_weekly_consumption : ℕ := 100
def weeks_supply : ℕ := 12

def total_boxes : ℕ := boxes_bought + boxes_existing
def total_birdseed_grams : ℕ := total_boxes * grams_per_box
def parrot_total_consumption : ℕ := parrot_weekly_consumption * weeks_supply
def cockatiel_total_consumption : ℕ := total_birdseed_grams - parrot_total_consumption
def cockatiel_weekly_consumption : ℕ := cockatiel_total_consumption / weeks_supply

theorem cockatiel_weekly_consumption_is_50 :
  cockatiel_weekly_consumption = 50 := by
  -- Proof goes here
  sorry

end cockatiel_weekly_consumption_is_50_l242_242824


namespace hyperbola_equation_l242_242794

theorem hyperbola_equation
  (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (e : ℝ) (he : e = 2 * Real.sqrt 3 / 3)
  (dist_from_origin : ∀ A B : ℝ × ℝ, A = (0, -b) ∧ B = (a, 0) →
    abs (a * b) / Real.sqrt (a^2 + b^2) = Real.sqrt 3 / 2) :
  (a^2 = 3 ∧ b^2 = 1) → (∀ x y : ℝ, (x^2 / 3 - y^2 = 1)) := 
sorry

end hyperbola_equation_l242_242794


namespace geometric_sequence_sum_n5_l242_242802

def geometric_sum (a₁ q : ℕ) (n : ℕ) : ℕ :=
  a₁ * (1 - q ^ n) / (1 - q)

theorem geometric_sequence_sum_n5 (a₁ q : ℕ) (n : ℕ) (h₁ : a₁ = 3) (h₂ : q = 4) (h₃ : n = 5) : 
  geometric_sum a₁ q n = 1023 :=
by
  sorry

end geometric_sequence_sum_n5_l242_242802


namespace prove_a_value_l242_242704

theorem prove_a_value (a : ℝ) (h : (a - 2) * 0^2 + 0 + a^2 - 4 = 0) : a = -2 := 
by
  sorry

end prove_a_value_l242_242704


namespace find_n_l242_242773

noncomputable def f (x : ℤ) : ℤ := sorry -- f is some polynomial with integer coefficients

theorem find_n (n : ℤ) (h1 : f 1 = -1) (h4 : f 4 = 2) (h8 : f 8 = 34) (hn : f n = n^2 - 4 * n - 18) : n = 3 ∨ n = 6 :=
sorry

end find_n_l242_242773


namespace coefficient_6th_term_expansion_l242_242527

-- Define the binomial coefficient
def binom : ℕ → ℕ → ℕ
| n, k => if k > n then 0 else Nat.choose n k

-- Define the coefficient of the general term of binomial expansion
def binomial_coeff (n r : ℕ) : ℤ := (-1)^r * binom n r

-- Define the theorem to show the coefficient of the 6th term in the expansion of (x-1)^10
theorem coefficient_6th_term_expansion :
  binomial_coeff 10 5 = -binom 10 5 :=
by sorry

end coefficient_6th_term_expansion_l242_242527


namespace sum_of_roots_l242_242157

theorem sum_of_roots (f : ℝ → ℝ) :
  (∀ x : ℝ, f (3 + x) = f (3 - x)) →
  (∃ (S : Finset ℝ), S.card = 6 ∧ ∀ x ∈ S, f x = 0) →
  (∃ (S : Finset ℝ), S.sum id = 18) :=
by
  sorry

end sum_of_roots_l242_242157


namespace derivative_of_my_function_l242_242649

variable (x : ℝ)

noncomputable def my_function : ℝ :=
  (Real.cos (Real.sin 3))^2 + (Real.sin (29 * x))^2 / (29 * Real.cos (58 * x))

theorem derivative_of_my_function :
  deriv my_function x = Real.tan (58 * x) / Real.cos (58 * x) := 
sorry

end derivative_of_my_function_l242_242649


namespace positive_number_percent_l242_242220

theorem positive_number_percent (x : ℝ) (h : 0.01 * x^2 = 9) (hx : 0 < x) : x = 30 :=
sorry

end positive_number_percent_l242_242220


namespace max_satiated_pikes_l242_242103

-- Define the total number of pikes
def total_pikes : ℕ := 30

-- Define the condition for satiation
def satiated_condition (eats : ℕ) : Prop := eats ≥ 3

-- Define the number of pikes eaten by each satiated pike
def eaten_by_satiated_pike : ℕ := 3

-- Define the theorem to find the maximum number of satiated pikes
theorem max_satiated_pikes (s : ℕ) : 
  (s * eaten_by_satiated_pike < total_pikes) → s ≤ 9 :=
by
  sorry

end max_satiated_pikes_l242_242103


namespace discount_percent_l242_242733

theorem discount_percent
  (MP CP SP : ℝ)
  (h1 : CP = 0.55 * MP)
  (gainPercent : ℝ)
  (h2 : gainPercent = 54.54545454545454 / 100)
  (h3 : (SP - CP) / CP = gainPercent)
  : ((MP - SP) / MP) * 100 = 15 := by
  sorry

end discount_percent_l242_242733


namespace largest_consecutive_sum_55_l242_242172

theorem largest_consecutive_sum_55 : 
  ∃ k n : ℕ, k > 0 ∧ (∃ n : ℕ, 55 = k * n + (k * (k - 1)) / 2 ∧ k = 5) :=
sorry

end largest_consecutive_sum_55_l242_242172


namespace net_marble_change_l242_242392

/-- Josh's initial number of marbles. -/
def initial_marbles : ℕ := 20

/-- Number of marbles Josh lost. -/
def lost_marbles : ℕ := 16

/-- Number of marbles Josh found. -/
def found_marbles : ℕ := 8

/-- Number of marbles Josh traded away. -/
def traded_away_marbles : ℕ := 5

/-- Number of marbles Josh received in a trade. -/
def received_in_trade_marbles : ℕ := 9

/-- Number of marbles Josh gave away. -/
def gave_away_marbles : ℕ := 3

/-- Number of marbles Josh received from his cousin. -/
def received_from_cousin_marbles : ℕ := 4

/-- Final number of marbles Josh has after all transactions. -/
def final_marbles : ℕ :=
  initial_marbles - lost_marbles + found_marbles - traded_away_marbles + received_in_trade_marbles
  - gave_away_marbles + received_from_cousin_marbles

theorem net_marble_change : (final_marbles : ℤ) - (initial_marbles : ℤ) = -3 := 
by
  sorry

end net_marble_change_l242_242392


namespace initial_number_of_persons_l242_242255

-- Define the conditions and the goal
def weight_increase_due_to_new_person : ℝ := 102 - 75
def average_weight_increase (n : ℝ) : ℝ := 4.5 * n

theorem initial_number_of_persons (n : ℝ) (h1 : average_weight_increase n = weight_increase_due_to_new_person) : n = 6 :=
by
  -- Skip the proof with sorry
  sorry

end initial_number_of_persons_l242_242255


namespace complete_the_square_l242_242264

theorem complete_the_square : ∀ x : ℝ, x^2 - 6 * x + 4 = 0 → (x - 3)^2 = 5 :=
by
  intro x h
  sorry

end complete_the_square_l242_242264


namespace max_value_of_f_l242_242136

noncomputable def f (theta x : ℝ) : ℝ :=
  (Real.cos theta)^2 - 2 * x * Real.cos theta - 1

noncomputable def M (x : ℝ) : ℝ :=
  if 0 <= x then 
    2 * x
  else 
    -2 * x

theorem max_value_of_f {x : ℝ} : 
  ∃ theta : ℝ, Real.cos theta ∈ [-1, 1] ∧ f theta x = M x :=
by
  sorry

end max_value_of_f_l242_242136


namespace polynomial_perfect_square_l242_242200

theorem polynomial_perfect_square (x : ℝ) :
  (x + 1) * (x + 2) * (x + 3) * (x + 4) + 1 = (x^2 + 5 * x + 5)^2 :=
by 
  sorry

end polynomial_perfect_square_l242_242200


namespace solve_inequalities_l242_242069

theorem solve_inequalities (x : ℝ) : (x + 1 > 0 ∧ x - 3 < 2) ↔ (-1 < x ∧ x < 5) :=
by sorry

end solve_inequalities_l242_242069


namespace gondor_laptops_wednesday_l242_242582

/-- Gondor's phone repair earnings per unit -/
def phone_earning : ℕ := 10

/-- Gondor's laptop repair earnings per unit -/
def laptop_earning : ℕ := 20

/-- Number of phones repaired on Monday -/
def phones_monday : ℕ := 3

/-- Number of phones repaired on Tuesday -/
def phones_tuesday : ℕ := 5

/-- Number of laptops repaired on Thursday -/
def laptops_thursday : ℕ := 4

/-- Total earnings of Gondor -/
def total_earnings : ℕ := 200

/-- Number of laptops repaired on Wednesday, which we need to prove equals 2 -/
def laptops_wednesday : ℕ := 2

theorem gondor_laptops_wednesday : 
    (phones_monday * phone_earning + phones_tuesday * phone_earning + 
    laptops_thursday * laptop_earning + laptops_wednesday * laptop_earning = total_earnings) :=
by
    sorry

end gondor_laptops_wednesday_l242_242582


namespace measure_45_minutes_l242_242493

-- Definitions of the conditions
structure Conditions where
  lighter : Prop
  strings : ℕ
  burn_time : ℕ → ℕ
  non_uniform_burn : Prop

-- We can now state the problem in Lean
theorem measure_45_minutes (c : Conditions) (h1 : c.lighter) (h2 : c.strings = 2)
  (h3 : ∀ s, s < 2 → c.burn_time s = 60) (h4 : c.non_uniform_burn) :
  ∃ t, t = 45 := 
sorry

end measure_45_minutes_l242_242493


namespace min_sugar_l242_242754

theorem min_sugar (f s : ℝ) (h₁ : f ≥ 8 + (3/4) * s) (h₂ : f ≤ 2 * s) : s ≥ 32 / 5 :=
sorry

end min_sugar_l242_242754


namespace area_of_triangle_ABC_sinA_value_l242_242814

noncomputable def cosC := 3 / 4
noncomputable def sinC := Real.sqrt (1 - cosC ^ 2)
noncomputable def a := 1
noncomputable def b := 2
noncomputable def c := Real.sqrt (a ^ 2 + b ^ 2 - 2 * a * b * cosC)
noncomputable def area := (1 / 2) * a * b * sinC
noncomputable def sinA := (a * sinC) / c

theorem area_of_triangle_ABC : area = Real.sqrt 7 / 4 :=
by sorry

theorem sinA_value : sinA = Real.sqrt 14 / 8 :=
by sorry

end area_of_triangle_ABC_sinA_value_l242_242814


namespace shaded_area_calculation_l242_242241

-- Define the dimensions of the grid and the size of each square
def gridWidth : ℕ := 9
def gridHeight : ℕ := 7
def squareSize : ℕ := 2

-- Define the number of 2x2 squares horizontally and vertically
def numSquaresHorizontally : ℕ := gridWidth / squareSize
def numSquaresVertically : ℕ := gridHeight / squareSize

-- Define the area of one 2x2 square and one shaded triangle within it
def squareArea : ℕ := squareSize * squareSize
def shadedTriangleArea : ℕ := squareArea / 2

-- Define the total number of 2x2 squares
def totalNumSquares : ℕ := numSquaresHorizontally * numSquaresVertically

-- Define the total area of shaded regions
def totalShadedArea : ℕ := totalNumSquares * shadedTriangleArea

-- The theorem to be proved
theorem shaded_area_calculation : totalShadedArea = 24 := by
  sorry    -- Placeholder for the proof

end shaded_area_calculation_l242_242241


namespace min_value_of_expression_l242_242607

theorem min_value_of_expression (x y : ℝ) : (2 * x * y - 3) ^ 2 + (x - y) ^ 2 ≥ 1 :=
sorry

end min_value_of_expression_l242_242607


namespace distinct_solutions_of_transformed_eq_l242_242008

open Function

variable {R : Type} [Field R]

def cubic_func (a b c d : R) (x : R) : R := a*x^3 + b*x^2 + c*x + d

noncomputable def three_distinct_roots {a b c d : R} (f : R → R)
  (h : ∀ x, f x = a*x^3 + b*x^2 + c*x + d) : Prop :=
∃ α β γ, α ≠ β ∧ β ≠ γ ∧ γ ≠ α ∧ f α = 0 ∧ f β = 0 ∧ f γ = 0

theorem distinct_solutions_of_transformed_eq
  {a b c d : R} (h : ∃ α β γ, α ≠ β ∧ β ≠ γ ∧ γ ≠ α ∧ (cubic_func a b c d α) = 0 ∧ (cubic_func a b c d β) = 0 ∧ (cubic_func a b c d γ) = 0) :
  ∃ p q, p ≠ q ∧ (4 * (cubic_func a b c d p) * (3 * a * p + b) = (3 * a * p^2 + 2 * b * p + c)^2) ∧ 
              (4 * (cubic_func a b c d q) * (3 * a * q + b) = (3 * a * q^2 + 2 * b * q + c)^2) := sorry

end distinct_solutions_of_transformed_eq_l242_242008


namespace least_consecutive_odd_integers_l242_242405

theorem least_consecutive_odd_integers (x : ℤ)
  (h : (x + (x + 2) + (x + 4) + (x + 6) + (x + 8) + (x + 10) + (x + 12) + (x + 14) = 8 * 414)) :
  x = 407 :=
by
  sorry

end least_consecutive_odd_integers_l242_242405


namespace find_some_number_eq_0_3_l242_242714

theorem find_some_number_eq_0_3 (X : ℝ) (h : 2 * ((3.6 * 0.48 * 2.50) / (X * 0.09 * 0.5)) = 1600.0000000000002) :
  X = 0.3 :=
by sorry

end find_some_number_eq_0_3_l242_242714


namespace area_of_triangle_PDE_l242_242316

noncomputable def length (a b : Point) : ℝ := -- define length between two points
sorry

def distance_from_line (P D E : Point) : ℝ := -- define perpendicular distance from P to line DE
sorry

structure Point :=
(x : ℝ)
(y : ℝ)

def area_triangle (P D E : Point) : ℝ :=
0.5 -- define area given conditions

theorem area_of_triangle_PDE (D E : Point) (hD_E : D ≠ E) :
  { P : Point | area_triangle P D E = 0.5 } =
  { P : Point | distance_from_line P D E = 1 / (length D E) } :=
sorry

end area_of_triangle_PDE_l242_242316


namespace volume_of_larger_prism_is_correct_l242_242418

noncomputable def volume_of_larger_solid : ℝ :=
  let A := (0, 0, 0)
  let B := (2, 0, 0)
  let C := (2, 2, 0)
  let D := (0, 2, 0)
  let E := (0, 0, 2)
  let F := (2, 0, 2)
  let G := (2, 2, 2)
  let H := (0, 2, 2)
  let P := (1, 1, 1)
  let Q := (1, 0, 1)
  
  -- Assume the plane equation here divides the cube into equal halves
  -- Calculate the volume of one half of the cube
  let volume := 2 -- This represents the volume of the larger solid

  volume

theorem volume_of_larger_prism_is_correct :
  volume_of_larger_solid = 2 :=
sorry

end volume_of_larger_prism_is_correct_l242_242418


namespace legs_total_l242_242039

def number_of_legs_bee := 6
def number_of_legs_spider := 8
def number_of_bees := 5
def number_of_spiders := 2
def total_legs := number_of_bees * number_of_legs_bee + number_of_spiders * number_of_legs_spider

theorem legs_total : total_legs = 46 := by
  sorry

end legs_total_l242_242039


namespace greatest_value_of_squares_l242_242770

theorem greatest_value_of_squares (a b c d : ℝ)
  (h1 : a + b = 18)
  (h2 : ab + c + d = 85)
  (h3 : ad + bc = 170)
  (h4 : cd = 105) :
  a^2 + b^2 + c^2 + d^2 ≤ 308 :=
sorry

end greatest_value_of_squares_l242_242770


namespace sum_tenth_powers_l242_242126

theorem sum_tenth_powers (a b : ℝ) (h1 : a + b = 1) (h2 : a^2 + b^2 = 3) (h3 : a^3 + b^3 = 4) (h4 : a^4 + b^4 = 7) (h5 : a^5 + b^5 = 11) : a^10 + b^10 = 123 :=
  sorry

end sum_tenth_powers_l242_242126


namespace major_axis_length_l242_242446

theorem major_axis_length {r : ℝ} (h_r : r = 1) (h_major : ∃ (minor_axis : ℝ), minor_axis = 2 * r ∧ 1.5 * minor_axis = major_axis) : major_axis = 3 :=
by
  sorry

end major_axis_length_l242_242446


namespace num_white_balls_l242_242059

theorem num_white_balls (W : ℕ) (h : (W : ℝ) / (6 + W) = 0.45454545454545453) : W = 5 :=
by
  sorry

end num_white_balls_l242_242059


namespace company_annual_income_l242_242668

variable {p a : ℝ}

theorem company_annual_income (h : 280 * p + (a - 280) * (p + 2) = a * (p + 0.25)) : a = 320 := 
sorry

end company_annual_income_l242_242668


namespace copper_zinc_ratio_l242_242431

theorem copper_zinc_ratio (total_weight : ℝ) (zinc_weight : ℝ) 
  (h_total_weight : total_weight = 70) (h_zinc_weight : zinc_weight = 31.5) : 
  (70 - 31.5) / 31.5 = 77 / 63 :=
by
  have h_copper_weight : total_weight - zinc_weight = 38.5 :=
    by rw [h_total_weight, h_zinc_weight]; norm_num
  sorry

end copper_zinc_ratio_l242_242431


namespace probability_of_green_ball_l242_242072

-- Definitions according to the conditions.
def containerA : ℕ × ℕ := (4, 6) -- 4 red balls, 6 green balls
def containerB : ℕ × ℕ := (6, 4) -- 6 red balls, 4 green balls
def containerC : ℕ × ℕ := (6, 4) -- 6 red balls, 4 green balls

-- Proving the probability of selecting a green ball.
theorem probability_of_green_ball :
  let pA := 1 / 3
  let pB := 1 / 3
  let pC := 1 / 3
  let pGreenA := (containerA.2 : ℚ) / (containerA.1 + containerA.2)
  let pGreenB := (containerB.2 : ℚ) / (containerB.1 + containerB.2)
  let pGreenC := (containerC.2 : ℚ) / (containerC.1 + containerC.2)
  pA * pGreenA + pB * pGreenB + pC * pGreenC = 7 / 15
  :=
by
  -- Formal proof will be filled in here.
  sorry

end probability_of_green_ball_l242_242072


namespace Mina_additional_miles_l242_242807

theorem Mina_additional_miles:
  let distance1 := 20 -- distance in miles for the first part of the trip
  let speed1 := 40 -- speed in mph for the first part of the trip
  let speed2 := 60 -- speed in mph for the second part of the trip
  let avg_speed := 55 -- average speed needed for the entire trip in mph
  let distance2 := (distance1 / speed1 + (avg_speed * (distance1 / speed1)) / (speed1 - avg_speed * speed1 / speed2)) * speed2 -- formula to find the additional distance
  distance2 = 90 :=
by {
  sorry
}

end Mina_additional_miles_l242_242807


namespace VerifyMultiplicationProperties_l242_242153

theorem VerifyMultiplicationProperties (α : Type) [Semiring α] :
  ((∀ x y z : α, (x * y) * z = x * (y * z)) ∧
   (∀ x y : α, x * y = y * x) ∧
   (∀ x y z : α, x * (y + z) = x * y + x * z) ∧
   (∃ e : α, ∀ x : α, x * e = x)) := by
  sorry

end VerifyMultiplicationProperties_l242_242153


namespace percent_sparrows_not_pigeons_l242_242570

-- Definitions of percentages
def crows_percent : ℝ := 0.20
def sparrows_percent : ℝ := 0.40
def pigeons_percent : ℝ := 0.15
def doves_percent : ℝ := 0.25

-- The statement to prove
theorem percent_sparrows_not_pigeons :
  (sparrows_percent / (1 - pigeons_percent)) = 0.47 :=
by
  sorry

end percent_sparrows_not_pigeons_l242_242570


namespace integer_solutions_to_quadratic_inequality_l242_242026

theorem integer_solutions_to_quadratic_inequality :
  {x : ℤ | (x^2 + 6 * x + 8) * (x^2 - 4 * x + 3) < 0} = {-3, 2} :=
by
  sorry

end integer_solutions_to_quadratic_inequality_l242_242026


namespace garage_motorcycles_l242_242318

theorem garage_motorcycles (bicycles cars motorcycles total_wheels : ℕ)
  (hb : bicycles = 20)
  (hc : cars = 10)
  (hw : total_wheels = 90)
  (wb : bicycles * 2 = 40)
  (wc : cars * 4 = 40)
  (wm : motorcycles * 2 = total_wheels - (bicycles * 2 + cars * 4)) :
  motorcycles = 5 := 
  by 
  sorry

end garage_motorcycles_l242_242318


namespace no_polyhedron_with_surface_2015_l242_242719

/--
It is impossible to glue together 1 × 1 × 1 cubes to form a polyhedron whose surface area is 2015.
-/
theorem no_polyhedron_with_surface_2015 (n k : ℕ) : 6 * n - 2 * k ≠ 2015 :=
by
  sorry

end no_polyhedron_with_surface_2015_l242_242719


namespace inequlity_proof_l242_242421

theorem inequlity_proof (a b : ℝ) : a^2 + a * b + b^2 ≥ 3 * (a + b - 1) := 
  sorry

end inequlity_proof_l242_242421


namespace jasmine_percentage_is_approx_l242_242573

noncomputable def initial_solution_volume : ℝ := 80
noncomputable def initial_jasmine_percent : ℝ := 0.10
noncomputable def initial_lemon_percent : ℝ := 0.05
noncomputable def initial_orange_percent : ℝ := 0.03
noncomputable def added_jasmine_volume : ℝ := 8
noncomputable def added_water_volume : ℝ := 12
noncomputable def added_lemon_volume : ℝ := 6
noncomputable def added_orange_volume : ℝ := 7

noncomputable def initial_jasmine_volume := initial_solution_volume * initial_jasmine_percent
noncomputable def initial_lemon_volume := initial_solution_volume * initial_lemon_percent
noncomputable def initial_orange_volume := initial_solution_volume * initial_orange_percent
noncomputable def initial_water_volume := initial_solution_volume - (initial_jasmine_volume + initial_lemon_volume + initial_orange_volume)

noncomputable def new_jasmine_volume := initial_jasmine_volume + added_jasmine_volume
noncomputable def new_water_volume := initial_water_volume + added_water_volume
noncomputable def new_lemon_volume := initial_lemon_volume + added_lemon_volume
noncomputable def new_orange_volume := initial_orange_volume + added_orange_volume
noncomputable def new_total_volume := new_jasmine_volume + new_water_volume + new_lemon_volume + new_orange_volume

noncomputable def new_jasmine_percent := (new_jasmine_volume / new_total_volume) * 100

theorem jasmine_percentage_is_approx :
  abs (new_jasmine_percent - 14.16) < 0.01 := sorry

end jasmine_percentage_is_approx_l242_242573


namespace area_of_circle_portion_l242_242427

theorem area_of_circle_portion :
  (∀ x y : ℝ, (x^2 + 6 * x + y^2 = 50) → y ≤ x - 3 → y ≤ 0 → (y^2 + (x + 3)^2 ≤ 59)) →
  (∃ area : ℝ, area = (59 * Real.pi / 4)) :=
by
  sorry

end area_of_circle_portion_l242_242427


namespace josephine_total_milk_l242_242376

-- Define the number of containers and the amount of milk they hold
def cnt_1 : ℕ := 3
def qty_1 : ℚ := 2

def cnt_2 : ℕ := 2
def qty_2 : ℚ := 0.75

def cnt_3 : ℕ := 5
def qty_3 : ℚ := 0.5

-- Define the total amount of milk sold
def total_milk_sold : ℚ := cnt_1 * qty_1 + cnt_2 * qty_2 + cnt_3 * qty_3

theorem josephine_total_milk : total_milk_sold = 10 := by
  -- This is the proof placeholder
  sorry

end josephine_total_milk_l242_242376


namespace correct_operation_l242_242226

theorem correct_operation (a b : ℝ) :
  ¬ (a^2 + a^3 = a^6) ∧
  ¬ ((a*b)^2 = a*(b^2)) ∧
  ¬ ((a + b)^2 = a^2 + b^2) ∧
  ((a + b)*(a - b) = a^2 - b^2) := 
by
  sorry

end correct_operation_l242_242226


namespace cos_beta_l242_242193

theorem cos_beta (α β : ℝ) (hα : 0 < α ∧ α < π/2) (hβ : 0 < β ∧ β < π/2) 
  (h_cos_α : Real.cos α = 3/5) (h_cos_alpha_plus_beta : Real.cos (α + β) = -5/13) : 
  Real.cos β = 33/65 :=
by
  sorry

end cos_beta_l242_242193


namespace minimize_distance_postman_l242_242763

-- Let x be a function that maps house indices to coordinates.
def optimalPostOfficeLocation (n: ℕ) (x : ℕ → ℝ) : ℝ :=
  if n % 2 = 1 then 
    x (n / 2 + 1)
  else 
    x (n / 2)

theorem minimize_distance_postman (n: ℕ) (x : ℕ → ℝ)
  (h_sorted : ∀ i j, i < j → x i < x j) :
  optimalPostOfficeLocation n x = if n % 2 = 1 then 
    x (n / 2 + 1)
  else 
    x (n / 2) := 
  sorry

end minimize_distance_postman_l242_242763


namespace work_completion_time_of_x_l242_242077

def totalWork := 1  -- We can normalize W to 1 unit to simplify the problem

theorem work_completion_time_of_x (W : ℝ) (Wx Wy : ℝ) 
  (hx : 8 * Wx + 16 * Wy = W)
  (hy : Wy = W / 20) :
  Wx = W / 40 :=
by
  -- The proof goes here, but we just put sorry for now.
  sorry

end work_completion_time_of_x_l242_242077


namespace root_of_quadratic_l242_242105

theorem root_of_quadratic (m : ℝ) (h : 3*1^2 - 1 + m = 0) : m = -2 :=
by {
  sorry
}

end root_of_quadratic_l242_242105


namespace group4_exceeds_group2_group4_exceeds_group3_l242_242319

-- Define conditions
def score_group1 : Int := 100
def score_group2 : Int := 150
def score_group3 : Int := -400
def score_group4 : Int := 350
def score_group5 : Int := -100

-- Theorem 1: Proving Group 4 exceeded Group 2 by 200 points
theorem group4_exceeds_group2 :
  score_group4 - score_group2 = 200 := by
  sorry

-- Theorem 2: Proving Group 4 exceeded Group 3 by 750 points
theorem group4_exceeds_group3 :
  score_group4 - score_group3 = 750 := by
  sorry

end group4_exceeds_group2_group4_exceeds_group3_l242_242319


namespace diagonal_length_of_rectangular_prism_l242_242516

-- Define the dimensions of the rectangular prism
variables (a b c : ℕ) (a_pos : a = 12) (b_pos : b = 15) (c_pos : c = 8)

-- Define the theorem statement
theorem diagonal_length_of_rectangular_prism : 
  ∃ d : ℝ, d = Real.sqrt (a ^ 2 + b ^ 2 + c ^ 2) ∧ d = Real.sqrt 433 := 
by
  -- Note that the proof is intentionally omitted
  sorry

end diagonal_length_of_rectangular_prism_l242_242516


namespace Emily_average_speed_l242_242833

noncomputable def Emily_run_distance : ℝ := 10

noncomputable def speed_first_uphill : ℝ := 4
noncomputable def distance_first_uphill : ℝ := 2

noncomputable def speed_first_downhill : ℝ := 6
noncomputable def distance_first_downhill : ℝ := 1

noncomputable def speed_flat_ground : ℝ := 5
noncomputable def distance_flat_ground : ℝ := 3

noncomputable def speed_second_uphill : ℝ := 4.5
noncomputable def distance_second_uphill : ℝ := 2

noncomputable def speed_second_downhill : ℝ := 6
noncomputable def distance_second_downhill : ℝ := 2

noncomputable def break_first : ℝ := 5 / 60
noncomputable def break_second : ℝ := 7 / 60
noncomputable def break_third : ℝ := 3 / 60

noncomputable def time_first_uphill : ℝ := distance_first_uphill / speed_first_uphill
noncomputable def time_first_downhill : ℝ := distance_first_downhill / speed_first_downhill
noncomputable def time_flat_ground : ℝ := distance_flat_ground / speed_flat_ground
noncomputable def time_second_uphill : ℝ := distance_second_uphill / speed_second_uphill
noncomputable def time_second_downhill : ℝ := distance_second_downhill / speed_second_downhill

noncomputable def total_running_time : ℝ := time_first_uphill + time_first_downhill + time_flat_ground + time_second_uphill + time_second_downhill
noncomputable def total_break_time : ℝ := break_first + break_second + break_third
noncomputable def total_time : ℝ := total_running_time + total_break_time

noncomputable def average_speed : ℝ := Emily_run_distance / total_time

theorem Emily_average_speed : abs (average_speed - 4.36) < 0.01 := by
  sorry

end Emily_average_speed_l242_242833


namespace find_a_l242_242303

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x > 0 then x * 2^(x + a) - 1 else - (x * 2^(-x + a) - 1)

theorem find_a (a : ℝ) (h_odd: ∀ x : ℝ, f x a = -f (-x) a)
  (h_pos : ∀ x : ℝ, x > 0 → f x a = x * 2^(x + a) - 1)
  (h_neg : f (-1) a = 3 / 4) :
  a = -3 :=
by
  sorry

end find_a_l242_242303


namespace tangent_line_at_P_l242_242551

noncomputable def tangent_line (x : ℝ) (y : ℝ) := (8 * x - y - 12 = 0)

def curve (x : ℝ) := x^3 - x^2

def derivative (f : ℝ → ℝ) (x : ℝ) := 3 * x^2 - 2 * x

theorem tangent_line_at_P :
    tangent_line 2 4 :=
by
  sorry

end tangent_line_at_P_l242_242551


namespace number_of_blue_marbles_l242_242518

-- Definitions based on the conditions
def total_marbles : ℕ := 20
def red_marbles : ℕ := 9
def probability_red_or_white : ℚ := 0.7

-- The question to prove: the number of blue marbles (B)
theorem number_of_blue_marbles (B W : ℕ) (h1 : B + W + red_marbles = total_marbles)
  (h2: (red_marbles + W : ℚ) / total_marbles = probability_red_or_white) : 
  B = 6 := 
by
  sorry

end number_of_blue_marbles_l242_242518


namespace positive_difference_of_solutions_l242_242430

theorem positive_difference_of_solutions :
  let a := 1
  let b := -6
  let c := -28
  let discriminant := b^2 - 4 * a * c
  let solution1 := 3 + (Real.sqrt discriminant) / 2
  let solution2 := 3 - (Real.sqrt discriminant) / 2
  have h_discriminant : discriminant = 148 := by sorry
  Real.sqrt 148 = 2 * Real.sqrt 37 :=
 sorry

end positive_difference_of_solutions_l242_242430


namespace foodAdditivesPercentage_l242_242192

-- Define the given percentages
def microphotonicsPercentage : ℕ := 14
def homeElectronicsPercentage : ℕ := 24
def microorganismsPercentage : ℕ := 29
def industrialLubricantsPercentage : ℕ := 8

-- Define degrees representing basic astrophysics
def basicAstrophysicsDegrees : ℕ := 18

-- Define the total degrees in a circle
def totalDegrees : ℕ := 360

-- Define the total budget percentage
def totalBudgetPercentage : ℕ := 100

-- Prove that the remaining percentage for food additives is 20%
theorem foodAdditivesPercentage :
  let basicAstrophysicsPercentage := (basicAstrophysicsDegrees * totalBudgetPercentage) / totalDegrees
  let totalKnownPercentage := microphotonicsPercentage + homeElectronicsPercentage + microorganismsPercentage + industrialLubricantsPercentage + basicAstrophysicsPercentage
  totalBudgetPercentage - totalKnownPercentage = 20 :=
by
  let basicAstrophysicsPercentage := (basicAstrophysicsDegrees * totalBudgetPercentage) / totalDegrees
  let totalKnownPercentage := microphotonicsPercentage + homeElectronicsPercentage + microorganismsPercentage + industrialLubricantsPercentage + basicAstrophysicsPercentage
  sorry

end foodAdditivesPercentage_l242_242192


namespace variable_value_l242_242284

theorem variable_value 
  (x : ℝ)
  (a k some_variable : ℝ)
  (eqn1 : (3 * x + 2) * (2 * x - 7) = a * x^2 + k * x + some_variable)
  (eqn2 : a - some_variable + k = 3)
  (a_val : a = 6)
  (k_val : k = -17) :
  some_variable = -14 :=
by
  sorry

end variable_value_l242_242284


namespace max_value_of_quadratic_on_interval_l242_242465

theorem max_value_of_quadratic_on_interval : 
  ∃ (x : ℝ), -2 ≤ x ∧ x ≤ 2 ∧ (∀ y, (∃ x, -2 ≤ x ∧ x ≤ 2 ∧ y = (x + 1)^2 - 4) → y ≤ 5) :=
sorry

end max_value_of_quadratic_on_interval_l242_242465


namespace find_a5_l242_242502

variable {α : Type*} [Field α]

def geometric_seq (a : α) (q : α) (n : ℕ) : α := a * q ^ (n - 1)

theorem find_a5 (a q : α) 
  (h1 : geometric_seq a q 2 = 4)
  (h2 : geometric_seq a q 6 * geometric_seq a q 7 = 16 * geometric_seq a q 9) :
  geometric_seq a q 5 = 32 ∨ geometric_seq a q 5 = -32 :=
by
  -- Proof is omitted as per instructions
  sorry

end find_a5_l242_242502


namespace percentage_y_less_than_x_l242_242653

variable (x y : ℝ)

-- given condition
axiom hyp : x = 11 * y

-- proof problem: Prove that the percentage y is less than x is (10/11) * 100
theorem percentage_y_less_than_x (x y : ℝ) (hyp : x = 11 * y) : 
  (x - y) / x * 100 = (10 / 11) * 100 :=
by
  sorry

end percentage_y_less_than_x_l242_242653


namespace find_number_l242_242219

theorem find_number (N : ℝ) (h : (1 / 2) * (3 / 5) * N = 36) : N = 120 :=
by
  sorry

end find_number_l242_242219


namespace largest_integer_satisfying_inequality_l242_242541

theorem largest_integer_satisfying_inequality : ∃ (x : ℤ), (5 * x - 4 < 3 - 2 * x) ∧ (∀ (y : ℤ), (5 * y - 4 < 3 - 2 * y) → y ≤ x) ∧ x = 0 :=
by
  sorry

end largest_integer_satisfying_inequality_l242_242541


namespace charge_difference_l242_242413

theorem charge_difference (cost_x cost_y : ℝ) (num_copies : ℕ) (hx : cost_x = 1.25) (hy : cost_y = 2.75) (hn : num_copies = 40) : 
  num_copies * cost_y - num_copies * cost_x = 60 := by
  sorry

end charge_difference_l242_242413


namespace total_spaces_in_game_l242_242256

-- Conditions
def first_turn : ℕ := 8
def second_turn_forward : ℕ := 2
def second_turn_backward : ℕ := 5
def third_turn : ℕ := 6
def total_to_end : ℕ := 37

-- Theorem stating the total number of spaces in the game
theorem total_spaces_in_game : first_turn + second_turn_forward - second_turn_backward + third_turn + (total_to_end - (first_turn + second_turn_forward - second_turn_backward + third_turn)) = total_to_end :=
by sorry

end total_spaces_in_game_l242_242256


namespace marathons_total_distance_l242_242706

theorem marathons_total_distance :
  ∀ (m y : ℕ),
  (26 + 385 / 1760 : ℕ) = 26 ∧ 385 % 1760 = 385 →
  15 * 26 + 15 * 385 / 1760 = m + 495 / 1760 ∧
  15 * 385 % 1760 = 495 →
  0 ≤ 495 ∧ 495 < 1760 →
  y = 495 := by
  intros
  sorry

end marathons_total_distance_l242_242706


namespace percent_value_quarters_l242_242599

noncomputable def value_in_cents (dimes quarters nickels : ℕ) : ℕ := 
  (dimes * 10) + (quarters * 25) + (nickels * 5)

noncomputable def percent_in_quarters (quarters total_value : ℕ) : ℚ := 
  (quarters * 25 : ℚ) / total_value * 100

theorem percent_value_quarters 
  (h_dimes : ℕ := 80) 
  (h_quarters : ℕ := 30) 
  (h_nickels : ℕ := 40) 
  (h_total_value := value_in_cents h_dimes h_quarters h_nickels) : 
  percent_in_quarters h_quarters h_total_value = 42.86 :=
by sorry

end percent_value_quarters_l242_242599


namespace positive_quadratic_if_and_only_if_l242_242247

variable (a : ℝ)
def p (x : ℝ) : ℝ := a * x^2 + 2 * x + 1

theorem positive_quadratic_if_and_only_if (h : ∀ x : ℝ, p a x > 0) : a > 1 := sorry

end positive_quadratic_if_and_only_if_l242_242247


namespace clothes_prices_l242_242146

theorem clothes_prices (total_cost : ℕ) (shirt_more : ℕ) (trousers_price : ℕ) (shirt_price : ℕ)
  (h1 : total_cost = 185)
  (h2 : shirt_more = 5)
  (h3 : shirt_price = 2 * trousers_price + shirt_more)
  (h4 : total_cost = shirt_price + trousers_price) : 
  trousers_price = 60 ∧ shirt_price = 125 :=
  by sorry

end clothes_prices_l242_242146


namespace polynomial_identity_l242_242027

theorem polynomial_identity : 
  ∀ x : ℝ, 
    5 * x^3 - 32 * x^2 + 75 * x - 71 = 
    5 * (x - 2)^3 + (-2) * (x - 2)^2 + 7 * (x - 2) - 9 :=
by 
  sorry

end polynomial_identity_l242_242027


namespace evaluate_expression_l242_242054

theorem evaluate_expression : 6^4 - 4 * 6^3 + 6^2 - 2 * 6 + 1 = 457 := 
  by
    sorry

end evaluate_expression_l242_242054


namespace quadratic_roots_l242_242595

-- Definitions based on the conditions provided.
def condition1 (x y : ℝ) : Prop := x^2 - 6 * x + 9 = -(abs (y - 1))

-- The main theorem we want to prove.
theorem quadratic_roots (x y : ℝ) (h : condition1 x y) : (a : ℝ) → (a - 3) * (a - 1) = a^2 - 4 * a + 3 :=
  by sorry

end quadratic_roots_l242_242595


namespace min_moves_to_find_treasure_l242_242194

theorem min_moves_to_find_treasure (cells : List ℕ) (h1 : cells = [5, 5, 5]) : 
  ∃ n, n = 2 ∧ (∀ moves, moves ≥ n → true) := sorry

end min_moves_to_find_treasure_l242_242194


namespace triangle_at_most_one_obtuse_l242_242537

theorem triangle_at_most_one_obtuse (A B C : ℝ) (h1 : A + B + C = 180) (h2 : A > 90 → B + C < 90) (h3 : B > 90 → A + C < 90) (h4 : C > 90 → A + B < 90) :
  ¬ (A > 90 ∧ B > 90 ∨ B > 90 ∧ C > 90 ∨ A > 90 ∧ C > 90) :=
by
  sorry

end triangle_at_most_one_obtuse_l242_242537


namespace counterexample_to_strict_inequality_l242_242383

theorem counterexample_to_strict_inequality :
  ∃ (a1 a2 b1 b2 c1 c2 d1 d2 : ℕ),
  (0 < a1) ∧ (0 < a2) ∧ (0 < b1) ∧ (0 < b2) ∧ (0 < c1) ∧ (0 < c2) ∧ (0 < d1) ∧ (0 < d2) ∧
  (a1 * b2 < a2 * b1) ∧ (c1 * d2 < c2 * d1) ∧ ¬ (a1 + c1) * (b2 + d2) < (a2 + c2) * (b1 + d1) :=
sorry

end counterexample_to_strict_inequality_l242_242383


namespace weight_of_lightest_weight_l242_242160

theorem weight_of_lightest_weight (x : ℕ) (y : ℕ) (h1 : 0 < y ∧ y < 9)
  (h2 : (10 : ℕ) * x + 45 - (x + y) = 2022) : x = 220 := by
  sorry

end weight_of_lightest_weight_l242_242160


namespace stratified_sampling_l242_242744

variable (H M L total_sample : ℕ)
variable (H_fams M_fams L_fams : ℕ)

-- Conditions
def community : Prop := H_fams = 150 ∧ M_fams = 360 ∧ L_fams = 90
def total_population : Prop := H_fams + M_fams + L_fams = 600
def sample_size : Prop := total_sample = 100

-- Statement
theorem stratified_sampling (H_fams M_fams L_fams : ℕ) (total_sample : ℕ)
  (h_com : community H_fams M_fams L_fams)
  (h_total_pop : total_population H_fams M_fams L_fams)
  (h_sample_size : sample_size total_sample)
  : H = 25 ∧ M = 60 ∧ L = 15 :=
by
  sorry

end stratified_sampling_l242_242744


namespace conic_is_parabola_l242_242437

-- Define the main equation
def main_equation (x y : ℝ) : Prop :=
  y^4 - 6 * x^2 = 3 * y^2 - 2

-- Definition of parabola condition
def is_parabola (x y : ℝ) : Prop :=
  ∃ a b c : ℝ, y^2 = a * x + b ∧ a ≠ 0

-- The theorem statement.
theorem conic_is_parabola :
  ∀ x y : ℝ, main_equation x y → is_parabola x y :=
by
  intros x y h
  sorry

end conic_is_parabola_l242_242437


namespace largest_possible_length_d_l242_242482

theorem largest_possible_length_d (a b c d : ℝ) 
  (h1 : a + b + c + d = 2) 
  (h2 : a ≤ b)
  (h3 : b ≤ c)
  (h4 : c ≤ d) 
  (h5 : d < a + b + c) : 
  d < 1 :=
sorry

end largest_possible_length_d_l242_242482


namespace ratio_of_a_to_c_l242_242637

theorem ratio_of_a_to_c (a b c d : ℚ)
  (h1 : a / b = 5 / 4) 
  (h2 : c / d = 4 / 3) 
  (h3 : d / b = 1 / 5) : a / c = 75 / 16 := 
sorry

end ratio_of_a_to_c_l242_242637


namespace Ron_book_picking_times_l242_242588

theorem Ron_book_picking_times (couples members : ℕ) (weeks people : ℕ) (Ron wife picks_per_year : ℕ) 
  (h1 : couples = 3) 
  (h2 : members = 5) 
  (h3 : Ron = 1) 
  (h4 : wife = 1) 
  (h5 : weeks = 52) 
  (h6 : people = 2 * couples + members + Ron + wife) 
  (h7 : picks_per_year = weeks / people) 
  : picks_per_year = 4 :=
by
  -- Definition steps can be added here if needed, currently immediate from conditions h1 to h7
  sorry

end Ron_book_picking_times_l242_242588


namespace find_x_l242_242676

noncomputable def f (x : ℝ) := (30 : ℝ) / (x + 5)
noncomputable def h (x : ℝ) := 4 * (f⁻¹ x)

theorem find_x (x : ℝ) (hx : h x = 20) : x = 3 :=
by 
  -- Conditions
  let f_inv := f⁻¹
  have h_def : h x = 4 * f_inv x := rfl
  have f_def : f x = (30 : ℝ) / (x + 5) := rfl
  -- Needed Proof Steps
  sorry

end find_x_l242_242676


namespace price_of_brand_X_pen_l242_242164

variable (P : ℝ)

theorem price_of_brand_X_pen :
  (∀ (n : ℕ), n = 12 → 6 * P + 6 * 2.20 = 42 - 13.20) →
  P = 4.80 :=
by
  intro h₁
  have h₂ := h₁ 12 rfl
  sorry

end price_of_brand_X_pen_l242_242164


namespace segment_halving_1M_l242_242377

noncomputable def segment_halving_sum (k : ℕ) : ℕ :=
  3^k + 1

theorem segment_halving_1M : segment_halving_sum 1000000 = 3^1000000 + 1 :=
by
  sorry

end segment_halving_1M_l242_242377


namespace find_k_value_l242_242370

theorem find_k_value : 
  (∃ (x y k : ℝ), x = -6.8 ∧ 
  (y = 0.25 * x + 10) ∧ 
  (k = -3 * x + y) ∧ 
  k = 32.1) :=
sorry

end find_k_value_l242_242370


namespace find_b_of_sin_l242_242580

theorem find_b_of_sin (a b c d : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0)
                       (h_period : (2 * Real.pi) / b = Real.pi / 2) : b = 4 := by
  sorry

end find_b_of_sin_l242_242580


namespace person_speed_in_kmph_l242_242070

-- Define the distance in meters
def distance_meters : ℕ := 300

-- Define the time in minutes
def time_minutes : ℕ := 4

-- Function to convert distance from meters to kilometers
def meters_to_kilometers (m : ℕ) : ℕ := m / 1000

-- Function to convert time from minutes to hours
def minutes_to_hours (min : ℕ) : ℚ := min / 60

-- Define the expected speed in km/h
def expected_speed : ℚ := 4.5

-- Proof statement
theorem person_speed_in_kmph : 
  meters_to_kilometers distance_meters / minutes_to_hours time_minutes = expected_speed :=
by 
  -- This is where the steps to verify the theorem would be located, currently omitted for the sake of the statement.
  sorry

end person_speed_in_kmph_l242_242070


namespace f_zero_one_and_odd_l242_242081

noncomputable def f : ℝ → ℝ := sorry

axiom f_def (a b : ℝ) : f (a * b) = a * f b + b * f a
axiom f_not_zero : ∃ x : ℝ, f x ≠ 0

theorem f_zero_one_and_odd :
  f 0 = 0 ∧ f 1 = 0 ∧ ∀ x : ℝ, f (-x) = -f x :=
by
  sorry

end f_zero_one_and_odd_l242_242081


namespace or_is_true_given_p_true_q_false_l242_242367

theorem or_is_true_given_p_true_q_false (p q : Prop) (hp : p) (hq : ¬q) : p ∨ q :=
by
  sorry

end or_is_true_given_p_true_q_false_l242_242367


namespace four_digit_perfect_square_l242_242717

theorem four_digit_perfect_square (N : ℕ) (a b : ℤ) :
  N = 1100 * a + 11 * b ∧
  N >= 1000 ∧ N <= 9999 ∧
  a >= 0 ∧ a <= 9 ∧ b >= 0 ∧ b <= 9 ∧
  (∃ (x : ℤ), N = 11 * x^2) →
  N = 7744 := by
  sorry

end four_digit_perfect_square_l242_242717


namespace min_packs_needed_l242_242542

-- Define pack sizes
def pack_sizes : List ℕ := [6, 12, 24, 30]

-- Define the total number of cans needed
def total_cans : ℕ := 150

-- Define the minimum number of packs needed to buy exactly 150 cans of soda
theorem min_packs_needed : ∃ packs : List ℕ, (∀ p ∈ packs, p ∈ pack_sizes) ∧ List.sum packs = total_cans ∧ packs.length = 5 := by
  sorry

end min_packs_needed_l242_242542


namespace gcd_of_set_B_is_five_l242_242014

def is_in_set_B (n : ℕ) : Prop :=
  ∃ x : ℕ, n = (x - 2) + (x - 1) + x + (x + 1) + (x + 2)

theorem gcd_of_set_B_is_five : ∃ d, (∀ n, is_in_set_B n → d ∣ n) ∧ d = 5 :=
by 
  sorry

end gcd_of_set_B_is_five_l242_242014


namespace triangle_perimeter_l242_242394

theorem triangle_perimeter (A B C : Type) 
  (x : ℝ) 
  (a b c : ℝ) 
  (h₁ : a = x + 1) 
  (h₂ : b = x) 
  (h₃ : c = x - 1) 
  (α β γ : ℝ) 
  (angle_condition : α = 2 * γ) 
  (law_of_sines : a / Real.sin α = c / Real.sin γ)
  (law_of_cosines : Real.cos γ = ((a^2 + b^2 - c^2) / (2 * b * a))) :
  a + b + c = 15 :=
  by
  sorry

end triangle_perimeter_l242_242394


namespace batteries_manufactured_l242_242232

theorem batteries_manufactured (gather_time create_time : Nat) (robots : Nat) (hours : Nat) (total_batteries : Nat) :
  gather_time = 6 →
  create_time = 9 →
  robots = 10 →
  hours = 5 →
  total_batteries = (hours * 60 / (gather_time + create_time)) * robots →
  total_batteries = 200 :=
by
  intros h_gather h_create h_robots h_hours h_batteries
  simp [h_gather, h_create, h_robots, h_hours] at h_batteries
  exact h_batteries

end batteries_manufactured_l242_242232


namespace radius_of_circle_l242_242550

theorem radius_of_circle (r : ℝ) (h : π * r^2 = 64 * π) : r = 8 :=
by
  sorry

end radius_of_circle_l242_242550


namespace max_investment_at_7_percent_l242_242642

variables (x y : ℝ)

theorem max_investment_at_7_percent 
  (h1 : x + y = 25000)
  (h2 : 0.07 * x + 0.12 * y ≥ 2450) : 
  x ≤ 11000 :=
sorry

end max_investment_at_7_percent_l242_242642


namespace printer_z_time_l242_242085

theorem printer_z_time (T_X T_Y T_Z : ℝ) (hZX_Y : T_X = 2.25 * (T_Y + T_Z)) 
  (hX : T_X = 15) (hY : T_Y = 10) : T_Z = 20 :=
by
  rw [hX, hY] at hZX_Y
  sorry

end printer_z_time_l242_242085


namespace math_problem_l242_242411

theorem math_problem (x : ℝ) (h : x + (1 / x) = 5) : x^2 + (1 / x^2) = 23 :=
sorry

end math_problem_l242_242411


namespace inequality_proof_l242_242388

variable (a b c d e p q : ℝ)

theorem inequality_proof
  (h₀ : 0 < p)
  (h₁ : p ≤ a) (h₂ : a ≤ q)
  (h₃ : p ≤ b) (h₄ : b ≤ q)
  (h₅ : p ≤ c) (h₆ : c ≤ q)
  (h₇ : p ≤ d) (h₈ : d ≤ q)
  (h₉ : p ≤ e) (h₁₀ : e ≤ q) :
  (a + b + c + d + e) * (1 / a + 1 / b + 1 / c + 1 / d + 1 / e) 
  ≤ 25 + 6 * (Real.sqrt (p / q) - Real.sqrt (q / p))^2 := 
by
  sorry -- The actual proof will be filled here

end inequality_proof_l242_242388


namespace find_digits_l242_242047

def divisible_45z_by_8 (z : ℕ) : Prop :=
  45 * z % 8 = 0

def sum_digits_divisible_by_9 (x y z : ℕ) : Prop :=
  (1 + 3 + x + y + 4 + 5 + z) % 9 = 0

def alternating_sum_digits_divisible_by_11 (x y z : ℕ) : Prop :=
  (1 - 3 + x - y + 4 - 5 + z) % 11 = 0

theorem find_digits (x y z : ℕ) (h_div8 : divisible_45z_by_8 z) (h_div9 : sum_digits_divisible_by_9 x y z) (h_div11 : alternating_sum_digits_divisible_by_11 x y z) :
  x = 2 ∧ y = 3 ∧ z = 6 := 
sorry

end find_digits_l242_242047


namespace incorrect_equation_l242_242010

noncomputable def x : ℂ := (-1 + Real.sqrt 3 * Complex.I) / 2
noncomputable def y : ℂ := (-1 - Real.sqrt 3 * Complex.I) / 2

theorem incorrect_equation : x^9 + y^9 ≠ -1 := sorry

end incorrect_equation_l242_242010


namespace expression_value_l242_242128

theorem expression_value (x : ℤ) (hx : x = 1729) : abs (abs (abs x + x) + abs x) + x = 6916 :=
by
  rw [hx]
  sorry

end expression_value_l242_242128


namespace min_value_2x_minus_y_l242_242079

open Real

theorem min_value_2x_minus_y : ∀ (x y : ℝ), |x| ≤ y ∧ y ≤ 2 → ∃ (c : ℝ), c = 2 * x - y ∧ ∀ z, z = 2 * x - y → z ≥ -6 := sorry

end min_value_2x_minus_y_l242_242079


namespace angle_E_in_quadrilateral_EFGH_l242_242151

theorem angle_E_in_quadrilateral_EFGH 
  (angle_E angle_F angle_G angle_H : ℝ) 
  (h1 : angle_E = 2 * angle_F)
  (h2 : angle_E = 3 * angle_G)
  (h3 : angle_E = 6 * angle_H)
  (sum_angles : angle_E + angle_F + angle_G + angle_H = 360) : 
  angle_E = 180 :=
by
  sorry

end angle_E_in_quadrilateral_EFGH_l242_242151


namespace geometric_sequence_general_formula_l242_242275

theorem geometric_sequence_general_formula (a : ℕ → ℝ) (h_pos : ∀ n, 0 < a n)
  (h1 : a 1 = 2)
  (h_rec : ∀ n, (a (n + 2))^2 + 4 * (a n)^2 = 4 * (a (n + 1))^2) :
  ∀ n, a n = 2^(n + 1) / 2 := 
sorry

end geometric_sequence_general_formula_l242_242275


namespace cats_in_village_l242_242385

theorem cats_in_village (C : ℕ) (h1 : 1 / 3 * C = (1 / 4) * (1 / 3) * C)
  (h2 : (1 / 12) * C = 10) : C = 120 :=
sorry

end cats_in_village_l242_242385


namespace find_y_l242_242819

theorem find_y (x y : ℕ) (h₀ : x > 0) (h₁ : y > 0) (h₂ : ∃ q : ℕ, x = q * y + 9) (h₃ : x / y = 96 + 3 / 20) : y = 60 :=
sorry

end find_y_l242_242819


namespace maria_trip_time_l242_242827

theorem maria_trip_time 
(s_highway : ℕ) (s_mountain : ℕ) (d_highway : ℕ) (d_mountain : ℕ) (t_mountain : ℕ) (t_break : ℕ) : 
  (s_highway = 4 * s_mountain) -> 
  (t_mountain = d_mountain / s_mountain) -> 
  t_mountain = 40 -> 
  t_break = 15 -> 
  d_highway = 100 -> 
  d_mountain = 20 ->
  s_mountain = d_mountain / t_mountain -> 
  s_highway = 4 * s_mountain -> 
  d_highway / s_highway = 50 ->
  40 + 50 + 15 = 105 := 
by 
  sorry

end maria_trip_time_l242_242827


namespace bottles_have_200_mL_l242_242795

def liters_to_milliliters (liters : ℕ) : ℕ :=
  liters * 1000

def total_milliliters (liters : ℕ) : ℕ :=
  liters_to_milliliters liters

def milliliters_per_bottle (total_mL : ℕ) (num_bottles : ℕ) : ℕ :=
  total_mL / num_bottles

theorem bottles_have_200_mL (num_bottles : ℕ) (total_oil_liters : ℕ) (h1 : total_oil_liters = 4) (h2 : num_bottles = 20) :
  milliliters_per_bottle (total_milliliters total_oil_liters) num_bottles = 200 := 
by
  sorry

end bottles_have_200_mL_l242_242795


namespace number_of_x_values_l242_242628

theorem number_of_x_values : 
  (∃ x_values : Finset ℕ, (∀ x ∈ x_values, 10 ≤ x ∧ x < 25) ∧ x_values.card = 15) :=
by
  sorry

end number_of_x_values_l242_242628


namespace books_read_last_month_l242_242273

namespace BookReading

variable (W : ℕ) -- Number of books William read last month.

-- Conditions
axiom cond1 : ∃ B : ℕ, B = 3 * W -- Brad read thrice as many books as William did last month.
axiom cond2 : W = 2 * 8 -- This month, William read twice as much as Brad, who read 8 books.
axiom cond3 : ∃ (B_prev : ℕ) (B_curr : ℕ), B_prev = 3 * W ∧ B_curr = 8 ∧ W + 16 = B_prev + B_curr + 4 -- Total books equation

theorem books_read_last_month : W = 2 := by
  sorry

end BookReading

end books_read_last_month_l242_242273


namespace MischiefConventionHandshakes_l242_242050

theorem MischiefConventionHandshakes :
  let gremlins := 30
  let imps := 25
  let reconciled_imps := 10
  let non_reconciled_imps := imps - reconciled_imps
  let handshakes_among_gremlins := (gremlins * (gremlins - 1)) / 2
  let handshakes_among_imps := (reconciled_imps * (reconciled_imps - 1)) / 2
  let handshakes_between_gremlins_and_imps := gremlins * imps
  handshakes_among_gremlins + handshakes_among_imps + handshakes_between_gremlins_and_imps = 1230 := by
  sorry

end MischiefConventionHandshakes_l242_242050


namespace initial_cost_renting_car_l242_242204

theorem initial_cost_renting_car
  (initial_cost : ℝ)
  (miles_monday : ℝ := 620)
  (miles_thursday : ℝ := 744)
  (cost_per_mile : ℝ := 0.50)
  (total_spent : ℝ := 832)
  (total_miles : ℝ := miles_monday + miles_thursday)
  (expected_initial_cost : ℝ := 150) :
  total_spent = initial_cost + cost_per_mile * total_miles → initial_cost = expected_initial_cost :=
by
  sorry

end initial_cost_renting_car_l242_242204


namespace charcoal_drawings_correct_l242_242179

-- Define the constants based on the problem conditions
def total_drawings : ℕ := 120
def colored_pencils : ℕ := 35
def blending_markers : ℕ := 22
def pastels : ℕ := 15
def watercolors : ℕ := 12

-- Calculate the total number of charcoal drawings
def charcoal_drawings : ℕ := total_drawings - (colored_pencils + blending_markers + pastels + watercolors)

-- The theorem we want to prove is that the number of charcoal drawings is 36
theorem charcoal_drawings_correct : charcoal_drawings = 36 :=
by
  -- The proof goes here (we skip it with 'sorry')
  sorry

end charcoal_drawings_correct_l242_242179


namespace weight_of_replaced_person_l242_242497

theorem weight_of_replaced_person 
  (avg_increase : ℝ)
  (num_persons : ℕ)
  (new_person_weight : ℝ)
  (weight_increase : ℝ)
  (new_person_might_be_90_kg : new_person_weight = 90)
  (average_increase_by_3_5_kg : avg_increase = 3.5)
  (group_of_8_persons : num_persons = 8)
  (total_weight_increase_formula : weight_increase = num_persons * avg_increase)
  (weight_of_replaced_person : ℝ)
  (weight_difference_formula : weight_of_replaced_person = new_person_weight - weight_increase) :
  weight_of_replaced_person = 62 :=
sorry

end weight_of_replaced_person_l242_242497


namespace principal_argument_of_z_l242_242829

-- Mathematical definitions based on provided conditions
noncomputable def theta : ℝ := Real.arctan (5 / 12)

-- The complex number z defined in the problem
noncomputable def z : ℂ := (Real.cos (2 * theta) + Real.sin (2 * theta) * Complex.I) / (239 + Complex.I)

-- Lean statement to prove the argument of z
theorem principal_argument_of_z : Complex.arg z = Real.pi / 4 :=
by
  sorry

end principal_argument_of_z_l242_242829


namespace pizza_slices_per_pizza_l242_242696

theorem pizza_slices_per_pizza (h : ∀ (mrsKaplanSlices bobbySlices pizzas : ℕ), 
  mrsKaplanSlices = 3 ∧ mrsKaplanSlices = bobbySlices / 4 ∧ pizzas = 2 → bobbySlices / pizzas = 6) : 
  ∃ (bobbySlices pizzas : ℕ), bobbySlices / pizzas = 6 :=
by
  existsi (3 * 4)
  existsi 2
  sorry

end pizza_slices_per_pizza_l242_242696


namespace transformation_C_factorization_l242_242639

open Function

theorem transformation_C_factorization (a b : ℤ) :
  (a - 1) * (b - 1) = ab - a - b + 1 :=
by sorry

end transformation_C_factorization_l242_242639


namespace common_factor_extraction_l242_242423

-- Define the polynomial
def poly (a b c : ℝ) := 8 * a^3 * b^2 + 12 * a^3 * b * c - 4 * a^2 * b

-- Define the common factor
def common_factor (a b : ℝ) := 4 * a^2 * b

-- State the theorem
theorem common_factor_extraction (a b c : ℝ) :
  ∃ p : ℝ, poly a b c = common_factor a b * p := by
  sorry

end common_factor_extraction_l242_242423


namespace students_wanted_fruit_l242_242511

theorem students_wanted_fruit (red_apples green_apples extra_fruit : ℕ)
  (h_red : red_apples = 42)
  (h_green : green_apples = 7)
  (h_extra : extra_fruit = 40) :
  red_apples + green_apples + extra_fruit - (red_apples + green_apples) = 40 :=
by
  sorry

end students_wanted_fruit_l242_242511


namespace sum_of_four_consecutive_even_numbers_l242_242818

theorem sum_of_four_consecutive_even_numbers (n : ℤ) (h : n^2 + (n + 2)^2 + (n + 4)^2 + (n + 6)^2 = 344) :
  n + (n + 2) + (n + 4) + (n + 6) = 36 := sorry

end sum_of_four_consecutive_even_numbers_l242_242818


namespace second_smallest_three_digit_in_pascal_triangle_l242_242097

theorem second_smallest_three_digit_in_pascal_triangle (m n : ℕ) :
  (∀ k : ℕ, ∃! r c : ℕ, r ≥ c ∧ r.choose c = k) →
  (∃! r : ℕ, r ≥ 2 ∧ 100 = r.choose 1) →
  (m = 101 ∧ n = 101) :=
by
  sorry

end second_smallest_three_digit_in_pascal_triangle_l242_242097


namespace first_number_is_48_l242_242271

-- Definitions of the conditions
def ratio (A B : ℕ) := 8 * B = 9 * A
def lcm (A B : ℕ) := Nat.lcm A B = 432

-- The statement to prove
theorem first_number_is_48 (A B : ℕ) (h_ratio : ratio A B) (h_lcm : lcm A B) : A = 48 :=
by
  sorry

end first_number_is_48_l242_242271


namespace grade_point_average_l242_242065

theorem grade_point_average (X : ℝ) (GPA_rest : ℝ) (GPA_whole : ℝ) 
  (h1 : GPA_rest = 66) (h2 : GPA_whole = 64) 
  (h3 : (1 / 3) * X + (2 / 3) * GPA_rest = GPA_whole) : X = 60 :=
sorry

end grade_point_average_l242_242065


namespace valid_third_side_l242_242663

-- Define a structure for the triangle with given sides
structure Triangle where
  a : ℝ
  b : ℝ
  x : ℝ

-- Define the conditions using the triangle inequality theorem
def valid_triangle (T : Triangle) : Prop :=
  T.a + T.x > T.b ∧ T.b + T.x > T.a ∧ T.a + T.b > T.x

-- Given values of a and b, and the condition on x
def specific_triangle : Triangle :=
  { a := 4, b := 9, x := 6 }

-- Statement to prove valid_triangle holds for specific_triangle
theorem valid_third_side : valid_triangle specific_triangle :=
by
  -- Import or assumptions about inequalities can be skipped or replaced by sorry
  sorry

end valid_third_side_l242_242663


namespace smallest_multiple_of_seven_gt_neg50_l242_242475

theorem smallest_multiple_of_seven_gt_neg50 : ∃ (n : ℤ), n % 7 = 0 ∧ n > -50 ∧ ∀ (m : ℤ), m % 7 = 0 → m > -50 → n ≤ m :=
sorry

end smallest_multiple_of_seven_gt_neg50_l242_242475


namespace fraction_of_ponies_with_horseshoes_l242_242574

theorem fraction_of_ponies_with_horseshoes 
  (P H : ℕ) 
  (h1 : H = P + 4) 
  (h2 : H + P ≥ 164) 
  (x : ℚ)
  (h3 : ∃ (n : ℕ), n = (5 / 8) * (x * P)) :
  x = 1 / 10 := by
  sorry

end fraction_of_ponies_with_horseshoes_l242_242574


namespace greatest_two_digit_product_is_12_l242_242810

theorem greatest_two_digit_product_is_12 : 
  ∃ (n : ℕ), (∃ (d1 d2 : ℕ), n = 10 * d1 + d2 ∧ d1 * d2 = 12 ∧ 10 ≤ n ∧ n < 100) ∧ 
              ∀ (m : ℕ), (∃ (e1 e2 : ℕ), m = 10 * e1 + e2 ∧ e1 * e2 = 12 ∧ 10 ≤ m ∧ m < 100) → m ≤ n :=
sorry

end greatest_two_digit_product_is_12_l242_242810


namespace gavin_shirts_l242_242526

theorem gavin_shirts (t g b : ℕ) (h_total : t = 23) (h_green : g = 17) (h_blue : b = t - g) : b = 6 :=
by sorry

end gavin_shirts_l242_242526


namespace vegan_non_soy_fraction_l242_242343

theorem vegan_non_soy_fraction (total_menu : ℕ) (vegan_dishes soy_free_vegan_dish : ℕ) 
  (h1 : vegan_dishes = 6) (h2 : vegan_dishes = total_menu / 3) (h3 : soy_free_vegan_dish = vegan_dishes - 5) :
  (soy_free_vegan_dish / total_menu = 1 / 18) :=
by
  sorry

end vegan_non_soy_fraction_l242_242343


namespace problem_proof_l242_242053

-- Definition of the function f
def f (x : ℝ) : ℝ := 2 * x + 2 - x

-- Condition given in the problem
axiom h : ∃ a : ℝ, f a = 3

-- Theorem statement
theorem problem_proof : ∃ a : ℝ, f a = 3 → f (2 * a) = 7 :=
by
  sorry

end problem_proof_l242_242053


namespace number_of_dozens_l242_242690

theorem number_of_dozens (x : Nat) (h : x = 16 * (3 * 4)) : x / 12 = 16 :=
by
  sorry

end number_of_dozens_l242_242690


namespace scientific_notation_of_00000065_l242_242033

theorem scientific_notation_of_00000065:
  (6.5 * 10^(-7)) = 0.00000065 :=
by
  -- Proof goes here
  sorry

end scientific_notation_of_00000065_l242_242033


namespace fraction_of_groups_with_a_and_b_l242_242242

/- Definitions based on the conditions -/
def total_persons : ℕ := 6
def group_size : ℕ := 3
def person_a : ℕ := 1  -- arbitrary assignment for simplicity
def person_b : ℕ := 2  -- arbitrary assignment for simplicity

/- Hypotheses based on conditions -/
axiom six_persons (n : ℕ) : n = total_persons
axiom divided_into_two_groups (grp_size : ℕ) : grp_size = group_size
axiom a_and_b_included (a b : ℕ) : a = person_a ∧ b = person_b

/- The theorem to prove -/
theorem fraction_of_groups_with_a_and_b
    (total_groups : ℕ := Nat.choose total_persons group_size)
    (groups_with_a_b : ℕ := Nat.choose 4 1) :
    groups_with_a_b / total_groups = 1 / 5 :=
by
    sorry

end fraction_of_groups_with_a_and_b_l242_242242


namespace max_value_inequality_am_gm_inequality_l242_242448

-- Given conditions and goals as Lean statements
theorem max_value_inequality (x : ℝ) : (|x - 1| + |x - 2| ≥ 1) := sorry

theorem am_gm_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : (1/a) + (1/(2*b)) + (1/(3*c)) = 1) : (a + 2*b + 3*c) ≥ 9 := sorry

end max_value_inequality_am_gm_inequality_l242_242448


namespace tire_mileage_problem_l242_242429

/- Definitions -/
def total_miles : ℕ := 45000
def enhancement_ratio : ℚ := 1.2
def total_tire_miles : ℚ := 180000

/- Question as theorem -/
theorem tire_mileage_problem
  (x y : ℚ)
  (h1 : y = enhancement_ratio * x)
  (h2 : 4 * x + y = total_tire_miles) :
  (x = 34615 ∧ y = 41538) :=
sorry

end tire_mileage_problem_l242_242429


namespace takeoff_run_length_l242_242488

theorem takeoff_run_length
  (t : ℕ) (h_t : t = 15)
  (v_kmh : ℕ) (h_v : v_kmh = 100)
  (uniform_acc : Prop) :
  ∃ S : ℕ, S = 208 := by
  sorry

end takeoff_run_length_l242_242488


namespace expression_evaluation_l242_242585

theorem expression_evaluation :
  (1007 * (((7/4 : ℚ) / (3/4) + (3 / (9/4)) + (1/3)) /
    ((1 + 2 + 3 + 4 + 5) * 5 - 22)) / 19) = (4 : ℚ) :=
by
  sorry

end expression_evaluation_l242_242585


namespace number_of_vip_children_l242_242206

theorem number_of_vip_children (total_attendees children_percentage children_vip_percentage : ℕ) :
  total_attendees = 400 →
  children_percentage = 75 →
  children_vip_percentage = 20 →
  (total_attendees * children_percentage / 100) * children_vip_percentage / 100 = 60 :=
by
  intros h_total h_children_pct h_vip_pct
  sorry

end number_of_vip_children_l242_242206


namespace company_bought_14_02_tons_l242_242265

noncomputable def gravel := 5.91
noncomputable def sand := 8.11
noncomputable def total_material := gravel + sand

theorem company_bought_14_02_tons : total_material = 14.02 :=
by 
  sorry

end company_bought_14_02_tons_l242_242265


namespace negation_problem_l242_242693

variable {a b c : ℝ}

theorem negation_problem (h : a + b + c = 3 → a^2 + b^2 + c^2 ≥ 3) : 
  a + b + c ≠ 3 → a^2 + b^2 + c^2 < 3 :=
sorry

end negation_problem_l242_242693


namespace range_of_m_l242_242709

theorem range_of_m (m : ℝ) : (∃ x : ℝ, x^2 + 2 * x - m - 1 = 0) → m ≥ -2 := 
by
  sorry

end range_of_m_l242_242709


namespace cos_double_angle_l242_242116

theorem cos_double_angle (θ : ℝ) (h : Real.cos θ = 3/5) : Real.cos (2 * θ) = -7/25 :=
by
  sorry

end cos_double_angle_l242_242116


namespace prism_closed_polygonal_chain_impossible_l242_242464

theorem prism_closed_polygonal_chain_impossible
  (lateral_edges : ℕ)
  (base_edges : ℕ)
  (total_edges : ℕ)
  (h_lateral : lateral_edges = 171)
  (h_base : base_edges = 171)
  (h_total : total_edges = 513)
  (h_total_sum : total_edges = 2 * base_edges + lateral_edges) :
  ¬ (∃ f : Fin 513 → (ℝ × ℝ × ℝ), (f 513 = f 0) ∧
    ∀ i, ( f (i + 1) - f i = (1, 0, 0) ∨ f (i + 1) - f i = (0, 1, 0) ∨ f (i + 1) - f i = (0, 0, 1) ∨ f (i + 1) - f i = (0, 0, -1) )) :=
by
  sorry

end prism_closed_polygonal_chain_impossible_l242_242464


namespace inequality_solution_l242_242767

def solution_set_of_inequality (x : ℝ) : Prop :=
  x * (x - 1) < 0

theorem inequality_solution :
  { x : ℝ | solution_set_of_inequality x } = { x : ℝ | 0 < x ∧ x < 1 } :=
by
  sorry

end inequality_solution_l242_242767


namespace lisa_goal_l242_242366

theorem lisa_goal 
  (total_quizzes : ℕ) 
  (target_percentage : ℝ) 
  (completed_quizzes : ℕ) 
  (earned_A : ℕ) 
  (remaining_quizzes : ℕ) : 
  total_quizzes = 40 → 
  target_percentage = 0.9 → 
  completed_quizzes = 25 → 
  earned_A = 20 → 
  remaining_quizzes = (total_quizzes - completed_quizzes) → 
  (earned_A + remaining_quizzes ≥ target_percentage * total_quizzes) → 
  remaining_quizzes - (total_quizzes * target_percentage - earned_A) = 0 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end lisa_goal_l242_242366


namespace sum_of_central_squares_is_34_l242_242361

-- Defining the parameters and conditions
def is_adjacent (i j : ℕ) : Prop := 
  (i = j + 1 ∨ i = j - 1 ∨ i = j + 4 ∨ i = j - 4)

def valid_matrix (M : Fin 4 → Fin 4 → ℕ) : Prop := 
  ∀ (i j : Fin 4), 
  i < 3 ∧ j < 3 → is_adjacent (M i j) (M (i + 1) j) ∧ is_adjacent (M i j) (M i (j + 1))

def corners_sum_to_34 (M : Fin 4 → Fin 4 → ℕ) : Prop :=
  M 0 0 + M 0 3 + M 3 0 + M 3 3 = 34

-- Stating the proof problem
theorem sum_of_central_squares_is_34 :
  ∃ (M : Fin 4 → Fin 4 → ℕ), valid_matrix M ∧ corners_sum_to_34 M → 
  (M 1 1 + M 1 2 + M 2 1 + M 2 2 = 34) :=
by
  sorry

end sum_of_central_squares_is_34_l242_242361


namespace vitya_catches_up_in_5_minutes_l242_242386

noncomputable def catch_up_time (s : ℝ) : ℝ :=
  let initial_distance := 20 * s
  let vitya_speed := 5 * s
  let mom_speed := s
  let relative_speed := vitya_speed - mom_speed
  initial_distance / relative_speed

theorem vitya_catches_up_in_5_minutes (s : ℝ) (h : s > 0) :
  catch_up_time s = 5 :=
by
  -- Proof is here.
  sorry

end vitya_catches_up_in_5_minutes_l242_242386


namespace expression_simplification_l242_242627

theorem expression_simplification (x y : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : 2 * x + y / 2 ≠ 0) :
  (2 * x + y / 2)⁻¹ * ((2 * x)⁻¹ + (y / 2)⁻¹) = (x * y)⁻¹ := 
sorry

end expression_simplification_l242_242627


namespace root_relationship_specific_root_five_l242_242803

def f (x : ℝ) : ℝ := x^3 - 6 * x^2 - 39 * x - 10
def g (x : ℝ) : ℝ := x^3 + x^2 - 20 * x - 50

theorem root_relationship :
  ∃ (x_0 : ℝ), g x_0 = 0 ∧ f (2 * x_0) = 0 :=
sorry

theorem specific_root_five :
  g 5 = 0 ∧ f 10 = 0 :=
sorry

end root_relationship_specific_root_five_l242_242803


namespace chinese_characters_digits_l242_242816

theorem chinese_characters_digits:
  ∃ (a b g s t : ℕ), -- Chinese characters represented by digits
    -- Different characters represent different digits
    a ≠ b ∧ a ≠ g ∧ a ≠ s ∧ a ≠ t ∧
    b ≠ g ∧ b ≠ s ∧ b ≠ t ∧
    g ≠ s ∧ g ≠ t ∧
    s ≠ t ∧
    -- Equation: 业步高 * 业步高 = 高升抬步高
    (a * 100 + b * 10 + g) * (a * 100 + b * 10 + g) = (g * 10000 + s * 1000 + t * 100 + b * 10 + g) :=
by {
  -- We need to prove that the number represented by "高升抬步高" is 50625.
  sorry
}

end chinese_characters_digits_l242_242816


namespace sum_of_parts_l242_242075

variable (x y : ℤ)
variable (h1 : x + y = 60)
variable (h2 : y = 45)

theorem sum_of_parts : 10 * x + 22 * y = 1140 :=
by
  sorry

end sum_of_parts_l242_242075


namespace power_mod_equiv_l242_242323

theorem power_mod_equiv :
  2^1000 % 17 = 1 := by
  sorry

end power_mod_equiv_l242_242323


namespace ahmed_final_score_requirement_l242_242737

-- Define the given conditions
def total_assignments : ℕ := 9
def ahmed_initial_grade : ℕ := 91
def emily_initial_grade : ℕ := 92
def sarah_initial_grade : ℕ := 94
def final_assignment_weight := true -- Assuming each assignment has the same weight
def min_passing_score : ℕ := 70
def max_score : ℕ := 100
def emily_final_score : ℕ := 90

noncomputable def ahmed_min_final_score : ℕ := 98

-- The proof statement
theorem ahmed_final_score_requirement :
  let ahmed_initial_points := ahmed_initial_grade * total_assignments
  let emily_initial_points := emily_initial_grade * total_assignments
  let sarah_initial_points := sarah_initial_grade * total_assignments
  let emily_final_total := emily_initial_points + emily_final_score
  let sarah_final_total := sarah_initial_points + min_passing_score
  let ahmed_final_total_needed := sarah_final_total + 1
  let ahmed_needed_score := ahmed_final_total_needed - ahmed_initial_points
  ahmed_needed_score = ahmed_min_final_score :=
by
  sorry

end ahmed_final_score_requirement_l242_242737


namespace positive_square_root_of_256_l242_242199

theorem positive_square_root_of_256 (y : ℝ) (hy_pos : y > 0) (hy_squared : y^2 = 256) : y = 16 :=
by
  sorry

end positive_square_root_of_256_l242_242199


namespace problem1_solution_problem2_solution_l242_242170

-- Problem 1: Prove that x = 1 given 6x - 7 = 4x - 5
theorem problem1_solution (x : ℝ) (h : 6 * x - 7 = 4 * x - 5) : x = 1 := by
  sorry


-- Problem 2: Prove that x = -1 given (3x - 1) / 4 - 1 = (5x - 7) / 6
theorem problem2_solution (x : ℝ) (h : (3 * x - 1) / 4 - 1 = (5 * x - 7) / 6) : x = -1 := by
  sorry

end problem1_solution_problem2_solution_l242_242170


namespace min_value_of_f_l242_242610

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x + Real.sin (2 * x)

theorem min_value_of_f : 
  ∃ x : ℝ, ∀ y : ℝ, f x ≤ f y :=
sorry

end min_value_of_f_l242_242610


namespace ring_cost_l242_242353

theorem ring_cost (total_cost : ℕ) (rings : ℕ) (h1 : total_cost = 24) (h2 : rings = 2) : total_cost / rings = 12 :=
by
  sorry

end ring_cost_l242_242353


namespace find_exp_l242_242379

noncomputable def a : ℝ := sorry
noncomputable def m : ℤ := sorry
noncomputable def n : ℤ := sorry

axiom a_m_eq_six : a ^ m = 6
axiom a_n_eq_six : a ^ n = 6

theorem find_exp : a ^ (2 * m - n) = 6 :=
by
  sorry

end find_exp_l242_242379


namespace equivalent_modulo_l242_242165

theorem equivalent_modulo:
  123^2 * 947 % 60 = 3 :=
by
  sorry

end equivalent_modulo_l242_242165


namespace peaches_division_l242_242364

theorem peaches_division (n k r : ℕ) 
  (h₁ : 100 = n * k + 10)
  (h₂ : 1000 = n * k * 11 + r) :
  r = 10 :=
by sorry

end peaches_division_l242_242364


namespace distinct_integers_sum_441_l242_242099

-- Define the variables and conditions
variables (a b c d : ℕ)

-- State the conditions: a, b, c, d are distinct positive integers and their product is 441
def distinct_positive_integers (a b c d : ℕ) : Prop := a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d 
def positive_integers (a b c d : ℕ) : Prop := a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0

-- Define the main statement to be proved
theorem distinct_integers_sum_441 (a b c d : ℕ) (h_distinct : distinct_positive_integers a b c d) 
(h_positive : positive_integers a b c d) 
(h_product : a * b * c * d = 441) : a + b + c + d = 32 :=
by
  sorry

end distinct_integers_sum_441_l242_242099


namespace area_of_square_with_perimeter_40_l242_242521

theorem area_of_square_with_perimeter_40 (P : ℝ) (s : ℝ) (A : ℝ) 
  (hP : P = 40) (hs : s = P / 4) (hA : A = s^2) : A = 100 :=
by
  sorry

end area_of_square_with_perimeter_40_l242_242521


namespace range_of_expression_l242_242402

theorem range_of_expression (x y : ℝ) (h1 : x * y = 1) (h2 : 3 ≥ x ∧ x ≥ 4 * y ∧ 4 * y > 0) :
  ∃ A B, A = 4 ∧ B = 5 ∧ ∀ z, z = (x^2 + 4 * y^2) / (x - 2 * y) → 4 ≤ z ∧ z ≤ 5 :=
by
  sorry

end range_of_expression_l242_242402


namespace cheese_bread_grams_l242_242309

/-- Each 100 grams of cheese bread costs 3.20 BRL and corresponds to 10 pieces. 
Each person eats, on average, 5 pieces of cheese bread. Including the professor,
there are 16 students, 1 monitor, and 5 parents, making a total of 23 people. 
The precision of the bakery's scale is 100 grams. -/
theorem cheese_bread_grams : (5 * 23 / 10) * 100 = 1200 := 
by
  sorry

end cheese_bread_grams_l242_242309


namespace solve_for_x_l242_242283

variables {A B C m n x : ℝ}

-- Existing conditions
def A_rate_condition : A = (B + C) / m := sorry
def B_rate_condition : B = (C + A) / n := sorry
def C_rate_condition : C = (A + B) / x := sorry

-- The theorem to be proven
theorem solve_for_x (A_rate_condition : A = (B + C) / m)
                    (B_rate_condition : B = (C + A) / n)
                    (C_rate_condition : C = (A + B) / x)
                    : x = (2 + m + n) / (m * n - 1) := by
  sorry

end solve_for_x_l242_242283


namespace regular_octagon_interior_angle_l242_242734

theorem regular_octagon_interior_angle : 
  (∀ (n : ℕ), n = 8 → ∀ (sum_of_interior_angles : ℕ), sum_of_interior_angles = (n - 2) * 180 → ∀ (each_angle : ℕ), each_angle = sum_of_interior_angles / n → each_angle = 135) :=
  sorry

end regular_octagon_interior_angle_l242_242734


namespace compound_interest_rate_l242_242752

theorem compound_interest_rate : 
  let P := 14800
  let interest := 4265.73
  let A := 19065.73
  let t := 2
  let n := 1
  let r := 0.13514
  (P : ℝ) * (1 + r)^t = A :=
by
-- Here we will provide the steps of the proof
sorry

end compound_interest_rate_l242_242752


namespace evening_customers_l242_242556

-- Define the conditions
def matinee_price : ℕ := 5
def evening_price : ℕ := 7
def opening_night_price : ℕ := 10
def popcorn_price : ℕ := 10
def num_matinee_customers : ℕ := 32
def num_opening_night_customers : ℕ := 58
def total_revenue : ℕ := 1670

-- Define the number of evening customers as a variable
variable (E : ℕ)

-- Prove that the number of evening customers E equals 40 given the conditions
theorem evening_customers :
  5 * num_matinee_customers +
  7 * E +
  10 * num_opening_night_customers +
  10 * (num_matinee_customers + E + num_opening_night_customers) / 2 = total_revenue
  → E = 40 :=
by
  intro h
  sorry

end evening_customers_l242_242556


namespace total_shirts_l242_242235

def hazel_shirts : ℕ := 6
def razel_shirts : ℕ := 2 * hazel_shirts

theorem total_shirts : hazel_shirts + razel_shirts = 18 := by
  sorry

end total_shirts_l242_242235


namespace M_subset_N_l242_242503

def M : Set ℝ := {x | ∃ k : ℤ, x = (k / 2) * 180 + 45}
def N : Set ℝ := {x | ∃ k : ℤ, x = (k / 4) * 180 + 45}

theorem M_subset_N : M ⊆ N :=
sorry

end M_subset_N_l242_242503


namespace part1_part2_l242_242552

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := abs (2 * x - a) + a
def g (x : ℝ) : ℝ := abs (2 * x - 1)

theorem part1 (x : ℝ) : f x 2 ≤ 6 ↔ -1 ≤ x ∧ x ≤ 3 := by
  sorry

theorem part2 (a : ℝ) : 2 ≤ a ↔ ∀ (x : ℝ), f x a + g x ≥ 3 := by
  sorry

end part1_part2_l242_242552


namespace reflection_proof_l242_242092

def reflect_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

noncomputable def initial_point : ℝ × ℝ := (3, -3)
noncomputable def reflected_over_y_axis := reflect_y initial_point
noncomputable def reflected_over_x_axis := reflect_x reflected_over_y_axis

theorem reflection_proof : reflected_over_x_axis = (-3, 3) :=
  by
    -- proof goes here
    sorry

end reflection_proof_l242_242092


namespace cost_of_old_car_l242_242498

theorem cost_of_old_car (C_old C_new : ℝ): 
  C_new = 2 * C_old → 
  1800 + 2000 = C_new → 
  C_old = 1900 :=
by
  intros H1 H2
  sorry

end cost_of_old_car_l242_242498


namespace fraction_addition_l242_242025

def fraction_sum : ℚ := (2 : ℚ)/5 + (3 : ℚ)/8

theorem fraction_addition : fraction_sum = 31/40 := by
  sorry

end fraction_addition_l242_242025


namespace find_f_at_3_l242_242083

theorem find_f_at_3 (f : ℤ → ℤ) (h : ∀ x : ℤ, f (2 * x + 1) = x ^ 2 - 2 * x) : f 3 = -1 :=
by {
  -- Proof would go here.
  sorry
}

end find_f_at_3_l242_242083


namespace max_value_of_inverse_l242_242681

noncomputable def f (x y z : ℝ) : ℝ := (1/4) * x^2 + 2 * y^2 + 16 * z^2

theorem max_value_of_inverse (x y z a b c : ℝ) (h : a + b + c = 1) (pos_intercepts : a > 0 ∧ b > 0 ∧ c > 0)
  (point_on_plane : (x/a + y/b + z/c = 1)) (pos_points : x > 0 ∧ y > 0 ∧ z > 0) :
  ∀ (k : ℕ), 21 ≤ k → k < (f x y z)⁻¹ :=
sorry

end max_value_of_inverse_l242_242681


namespace correct_operation_l242_242063

theorem correct_operation : (a : ℕ) →
  (a^2 * a^3 = a^5) ∧
  (2 * a + 4 ≠ 6 * a) ∧
  ((2 * a)^2 ≠ 2 * a^2) ∧
  (a^3 / a^3 ≠ a) := sorry

end correct_operation_l242_242063


namespace trapezoid_area_l242_242009

variable (x y : ℝ)

def condition1 : Prop := abs (y - 3 * x) ≥ abs (2 * y + x) ∧ -1 ≤ y - 3 ∧ y - 3 ≤ 1

def condition2 : Prop := (2 * y + y - y + 3 * x) * (2 * y + x + y - 3 * x) ≤ 0 ∧ 2 ≤ y ∧ y ≤ 4

theorem trapezoid_area (h1 : condition1 x y) (h2 : condition2 x y) :
  let A := (3, 2)
  let B := (-1/2, 2)
  let C := (-1, 4)
  let D := (6, 4)
  let S := (1/2) * (2 * (7 + 3.5))
  S = 10.5 :=
sorry

end trapezoid_area_l242_242009


namespace range_of_f_l242_242240

-- Define the function f
def f (x : ℕ) : ℤ := 2 * (x : ℤ) - 3

-- Define the domain
def domain : Finset ℕ := {1, 2, 3, 4, 5}

-- Define the expected range
def expected_range : Finset ℤ := {-1, 1, 3, 5, 7}

-- Prove the range of f given the domain
theorem range_of_f : domain.image f = expected_range :=
  sorry

end range_of_f_l242_242240


namespace Sam_has_correct_amount_of_dimes_l242_242253

-- Definitions for initial values and transactions
def initial_dimes := 9
def dimes_from_dad := 7
def dimes_taken_by_mom := 3
def sets_from_sister := 4
def dimes_per_set := 2

-- Definition of the total dimes Sam has now
def total_dimes_now : Nat :=
  initial_dimes + dimes_from_dad - dimes_taken_by_mom + (sets_from_sister * dimes_per_set)

-- Proof statement
theorem Sam_has_correct_amount_of_dimes : total_dimes_now = 21 := by
  sorry

end Sam_has_correct_amount_of_dimes_l242_242253


namespace solution_x_x_sub_1_eq_x_l242_242622

theorem solution_x_x_sub_1_eq_x (x : ℝ) : x * (x - 1) = x ↔ (x = 0 ∨ x = 2) :=
by {
  sorry
}

end solution_x_x_sub_1_eq_x_l242_242622


namespace antonella_toonies_l242_242114

theorem antonella_toonies (L T : ℕ) (h1 : L + T = 10) (h2 : L + 2 * T = 14) : T = 4 :=
by
  sorry

end antonella_toonies_l242_242114


namespace problem_1_problem_2_l242_242612

-- Definitions of sets A and B
def A : Set ℝ := { x : ℝ | x^2 - 2 * x - 3 ≤ 0 }
def B (m : ℝ) : Set ℝ := { x : ℝ | m - 1 ≤ x ∧ x ≤ m + 1 }

-- Problem 1: Prove that if A ∩ B = [1, 3], then m = 2
theorem problem_1 (m : ℝ) (h : (A ∩ B m) = {x : ℝ | 1 ≤ x ∧ x ≤ 3}) : m = 2 :=
sorry

-- Problem 2: Prove that if A ⊆ complement ℝ B m, then m > 4 or m < -2
theorem problem_2 (m : ℝ) (h : A ⊆ { x : ℝ | x < m - 1 ∨ x > m + 1 }) : m > 4 ∨ m < -2 :=
sorry

end problem_1_problem_2_l242_242612


namespace roots_product_l242_242697

theorem roots_product (x1 x2 : ℝ) (h : ∀ x : ℝ, x^2 - 4 * x + 1 = 0 → x = x1 ∨ x = x2) : x1 * x2 = 1 :=
sorry

end roots_product_l242_242697


namespace log_expression_l242_242650

section log_problem

variable (log : ℝ → ℝ)
variable (m n : ℝ)

-- Assume the properties of logarithms:
-- 1. log(m^n) = n * log(m)
axiom log_pow (m : ℝ) (n : ℝ) : log (m ^ n) = n * log m
-- 2. log(m * n) = log(m) + log(n)
axiom log_mul (m n : ℝ) : log (m * n) = log m + log n
-- 3. log(1) = 0
axiom log_one : log 1 = 0

theorem log_expression : log 5 * log 2 + log (2 ^ 2) - log 2 = 0 := by
  sorry

end log_problem

end log_expression_l242_242650


namespace fifth_term_is_19_l242_242730

-- Define the first term and the common difference
def a₁ : Int := 3
def d : Int := 4

-- Define the formula for the nth term in the arithmetic sequence
def arithmetic_sequence (n : Int) : Int :=
  a₁ + (n - 1) * d

-- Define the Lean 4 statement proving that the 5th term is 19
theorem fifth_term_is_19 : arithmetic_sequence 5 = 19 :=
by
  sorry -- Proof to be filled in

end fifth_term_is_19_l242_242730


namespace simplify_expr_l242_242024

-- Define the expression
def expr (a : ℝ) := 4 * a ^ 2 * (3 * a - 1)

-- State the theorem
theorem simplify_expr (a : ℝ) : expr a = 12 * a ^ 3 - 4 * a ^ 2 := 
by 
  sorry

end simplify_expr_l242_242024


namespace rectangles_divided_into_13_squares_l242_242679

theorem rectangles_divided_into_13_squares (m n : ℕ) (h : m * n = 13) : 
  (m = 1 ∧ n = 13) ∨ (m = 13 ∧ n = 1) :=
sorry

end rectangles_divided_into_13_squares_l242_242679


namespace total_tiles_count_l242_242084

theorem total_tiles_count (n total_tiles: ℕ) 
  (h1: total_tiles - n^2 = 36) 
  (h2: total_tiles - (n + 1)^2 = 3) : total_tiles = 292 :=
by {
  sorry
}

end total_tiles_count_l242_242084


namespace evaluate_expression_l242_242701

theorem evaluate_expression :
  (⌈(19 / 7 : ℚ) - ⌈(35 / 19 : ℚ)⌉⌉ / ⌈(35 / 7 : ℚ) + ⌈((7 * 19) / 35 : ℚ)⌉⌉) = (1 / 9 : ℚ) :=
by
  sorry

end evaluate_expression_l242_242701


namespace value_of_expression_l242_242553

variable (x1 x2 : ℝ)

def sum_roots (x1 x2 : ℝ) : Prop := x1 + x2 = 3
def product_roots (x1 x2 : ℝ) : Prop := x1 * x2 = -4

theorem value_of_expression (h1 : sum_roots x1 x2) (h2 : product_roots x1 x2) : 
  x1^2 - 4*x1 - x2 + 2*x1*x2 = -7 :=
by sorry

end value_of_expression_l242_242553


namespace prove_absolute_value_subtract_power_l242_242249

noncomputable def smallest_absolute_value : ℝ := 0

theorem prove_absolute_value_subtract_power (b : ℝ) 
  (h1 : smallest_absolute_value = 0) 
  (h2 : b * b = 1) : 
  (|smallest_absolute_value - 2| - b ^ 2023 = 1) 
  ∨ (|smallest_absolute_value - 2| - b ^ 2023 = 3) :=
sorry

end prove_absolute_value_subtract_power_l242_242249


namespace y_pow_x_eq_x_pow_y_l242_242028

theorem y_pow_x_eq_x_pow_y (n : ℕ) (hn : 0 < n) :
    let x := (1 + 1 / (n : ℝ)) ^ n
    let y := (1 + 1 / (n : ℝ)) ^ (n + 1)
    y ^ x = x ^ y := 
    sorry

end y_pow_x_eq_x_pow_y_l242_242028


namespace books_about_outer_space_l242_242584

variable (x : ℕ)

theorem books_about_outer_space :
  160 + 48 + 16 * x = 224 → x = 1 :=
by
  intro h
  sorry

end books_about_outer_space_l242_242584


namespace ellipse_sum_l242_242237

-- Define the givens
def h : ℤ := -3
def k : ℤ := 5
def a : ℤ := 7
def b : ℤ := 4

-- State the theorem to be proven
theorem ellipse_sum : h + k + a + b = 13 := by
  sorry

end ellipse_sum_l242_242237


namespace arithmetic_sequence_sum_l242_242064

theorem arithmetic_sequence_sum (a : ℕ → ℕ) (S : ℕ → ℕ) (h1 : a 8 = 8)
  (h2 : ∀ n, S n = n * (a 1 + a n) / 2) (h3 : a 1 + a 15 = 2 * a 8) :
  S 15 = 120 := sorry

end arithmetic_sequence_sum_l242_242064


namespace bee_count_l242_242224

theorem bee_count (initial_bees additional_bees : ℕ) (h_init : initial_bees = 16) (h_add : additional_bees = 9) :
  initial_bees + additional_bees = 25 :=
by
  sorry

end bee_count_l242_242224


namespace smallest_not_prime_nor_square_no_prime_factor_lt_60_correct_l242_242579

noncomputable def smallest_not_prime_nor_square_no_prime_factor_lt_60 : ℕ :=
  4087

theorem smallest_not_prime_nor_square_no_prime_factor_lt_60_correct :
  ∀ n : ℕ, 
    (n > 0) → 
    (¬ Prime n) →
    (¬ ∃ k : ℕ, k * k = n) →
    (∀ p : ℕ, Prime p → p ∣ n → p ≥ 60) →
    n ≥ 4087 :=
sorry

end smallest_not_prime_nor_square_no_prime_factor_lt_60_correct_l242_242579


namespace f_is_odd_and_periodic_l242_242252

noncomputable def f : ℝ → ℝ := sorry

axiom h1 : ∀ x : ℝ, f (10 + x) = f (10 - x)
axiom h2 : ∀ x : ℝ, f (20 - x) = -f (20 + x)

theorem f_is_odd_and_periodic : 
  (∀ x : ℝ, f (-x) = -f x) ∧ (∃ T : ℝ, T = 40 ∧ ∀ x : ℝ, f (x + T) = f x) :=
by
  sorry

end f_is_odd_and_periodic_l242_242252


namespace total_volume_stacked_dice_l242_242189

def die_volume (width length height : ℕ) : ℕ := 
  width * length * height

def total_dice (horizontal vertical layers : ℕ) : ℕ := 
  horizontal * vertical * layers

theorem total_volume_stacked_dice :
  let width := 1
  let length := 1
  let height := 1
  let horizontal := 7
  let vertical := 5
  let layers := 3
  let single_die_volume := die_volume width length height
  let num_dice := total_dice horizontal vertical layers
  single_die_volume * num_dice = 105 :=
by
  sorry  -- proof to be provided

end total_volume_stacked_dice_l242_242189


namespace distance_is_18_l242_242557

noncomputable def distance_walked (x t d : ℝ) : Prop :=
  let faster := (x + 1) * (3 * t / 4) = d
  let slower := (x - 1) * (t + 3) = d
  let normal := x * t = d
  faster ∧ slower ∧ normal

theorem distance_is_18 : 
  ∃ (x t : ℝ), distance_walked x t 18 :=
by
  sorry

end distance_is_18_l242_242557


namespace max_sheep_pen_area_l242_242362

theorem max_sheep_pen_area :
  ∃ x y : ℝ, 15 * 2 = 30 ∧ (x + 2 * y = 30) ∧
  (x > 0 ∧ y > 0) ∧
  (x * y = 112) := by
  sorry

end max_sheep_pen_area_l242_242362


namespace divisibility_of_binomial_l242_242685

theorem divisibility_of_binomial (p : ℕ) (n : ℕ) (hp : Nat.Prime p) (hn : n > 1) :
    (∀ x : ℕ, 1 ≤ x ∧ x ≤ n-1 → p ∣ Nat.choose n x) ↔ ∃ m : ℕ, n = p^m := sorry

end divisibility_of_binomial_l242_242685


namespace largest_k_inequality_l242_242672

theorem largest_k_inequality {a b c : ℝ} (h1 : a ≤ b) (h2 : b ≤ c) (h3 : ab + bc + ca = 0) (h4 : abc = 1) :
  |a + b| ≥ 4 * |c| :=
sorry

end largest_k_inequality_l242_242672


namespace calculation_result_l242_242348

theorem calculation_result:
  (-1:ℤ)^3 - 8 / (-2) + 4 * abs (-5) = 23 := by
  sorry

end calculation_result_l242_242348


namespace sum_of_numbers_in_ratio_with_lcm_l242_242108

theorem sum_of_numbers_in_ratio_with_lcm (a b : ℕ) (h_lcm : Nat.lcm a b = 36) (h_ratio : a / Nat.gcd a b = 2 ∧ b / Nat.gcd a b = 3) : a + b = 30 :=
sorry

end sum_of_numbers_in_ratio_with_lcm_l242_242108


namespace highest_probability_highspeed_rail_l242_242459

def total_balls : ℕ := 10
def beidou_balls : ℕ := 3
def tianyan_balls : ℕ := 2
def highspeed_rail_balls : ℕ := 5

theorem highest_probability_highspeed_rail :
  (highspeed_rail_balls : ℚ) / total_balls > (beidou_balls : ℚ) / total_balls ∧
  (highspeed_rail_balls : ℚ) / total_balls > (tianyan_balls : ℚ) / total_balls :=
by {
  -- Proof skipped
  sorry
}

end highest_probability_highspeed_rail_l242_242459


namespace price_alloy_per_kg_l242_242415

-- Defining the costs of the two metals.
def cost_metal1 : ℝ := 68
def cost_metal2 : ℝ := 96

-- Defining the mixture ratio.
def ratio : ℝ := 1

-- The proposition that the price per kg of the alloy is 82 Rs.
theorem price_alloy_per_kg (C1 C2 r : ℝ) (hC1 : C1 = 68) (hC2 : C2 = 96) (hr : r = 1) :
  (C1 + C2) / (r + r) = 82 :=
by
  sorry

end price_alloy_per_kg_l242_242415


namespace impossibility_triplet_2002x2002_grid_l242_242506

theorem impossibility_triplet_2002x2002_grid: 
  ∀ (M : Matrix ℕ (Fin 2002) (Fin 2002)),
    (∀ i j : Fin 2002, ∃ (r1 r2 r3 : Fin 2002), 
      (M i r1 > 0 ∧ M i r2 > 0 ∧ M i r3 > 0) ∨ 
      (M r1 j > 0 ∧ M r2 j > 0 ∧ M r3 j > 0)) →
    ¬ (∀ i j : Fin 2002, ∃ (a b c : ℕ), 
      M i j = a ∧ a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ 
      (∃ (r1 r2 r3 : Fin 2002), 
        (M i r1 = a ∨ M i r1 = b ∨ M i r1 = c) ∧ 
        (M i r2 = a ∨ M i r2 = b ∨ M i r2 = c) ∧ 
        (M i r3 = a ∨ M i r3 = b ∨ M i r3 = c)) ∨
      (∃ (c1 c2 c3 : Fin 2002), 
        (M c1 j = a ∨ M c1 j = b ∨ M c1 j = c) ∧ 
        (M c2 j = a ∨ M c2 j = b ∨ M c2 j = c) ∧ 
        (M c3 j = a ∨ M c3 j = b ∨ M c3 j = c)))
:= sorry

end impossibility_triplet_2002x2002_grid_l242_242506


namespace one_third_percent_of_150_l242_242487

theorem one_third_percent_of_150 : (1/3) * (150 / 100) = 0.5 := by
  sorry

end one_third_percent_of_150_l242_242487


namespace union_of_intervals_l242_242624

open Set

theorem union_of_intervals :
  let M := { x : ℝ | 1 < x ∧ x ≤ 3 }
  let N := { x : ℝ | 2 < x ∧ x ≤ 5 }
  M ∪ N = { x : ℝ | 1 < x ∧ x ≤ 5 } :=
by
  let M := { x : ℝ | 1 < x ∧ x ≤ 3 }
  let N := { x : ℝ | 2 < x ∧ x ≤ 5 }
  sorry

end union_of_intervals_l242_242624


namespace intersection_M_N_l242_242395

def M := {x : ℝ | -2 ≤ x ∧ x ≤ 2}
def N := {x : ℝ | x^2 - 3*x ≤ 0}

theorem intersection_M_N : M ∩ N = {x : ℝ | 0 ≤ x ∧ x ≤ 2} :=
by
  sorry

end intersection_M_N_l242_242395


namespace max_grain_mass_l242_242529

def platform_length : ℝ := 10
def platform_width : ℝ := 5
def grain_density : ℝ := 1200
def angle_of_repose : ℝ := 45
def max_mass : ℝ := 175000

theorem max_grain_mass :
  let height_of_pile := platform_width / 2
  let volume_of_prism := platform_length * platform_width * height_of_pile
  let volume_of_pyramid := (1 / 3) * (platform_width * height_of_pile) * height_of_pile
  let total_volume := volume_of_prism + 2 * volume_of_pyramid
  let calculated_mass := total_volume * grain_density
  calculated_mass = max_mass :=
by {
  sorry
}

end max_grain_mass_l242_242529


namespace delivery_cost_l242_242205

theorem delivery_cost (base_fee : ℕ) (limit : ℕ) (extra_fee : ℕ) 
(item_weight : ℕ) (total_cost : ℕ) 
(h1 : base_fee = 13) (h2 : limit = 5) (h3 : extra_fee = 2) 
(h4 : item_weight = 7) (h5 : total_cost = 17) : 
  total_cost = base_fee + (item_weight - limit) * extra_fee := 
by
  sorry

end delivery_cost_l242_242205


namespace cyclic_quadrilateral_diameter_l242_242029

theorem cyclic_quadrilateral_diameter
  (AB BC CD DA : ℝ)
  (h1 : AB = 25)
  (h2 : BC = 39)
  (h3 : CD = 52)
  (h4 : DA = 60) : 
  ∃ D : ℝ, D = 65 :=
by
  sorry

end cyclic_quadrilateral_diameter_l242_242029


namespace cone_volume_l242_242750

theorem cone_volume (S r : ℝ) : 
  ∃ V : ℝ, V = (1 / 3) * S * r :=
by
  sorry

end cone_volume_l242_242750


namespace selling_price_correct_l242_242071

def meters_of_cloth : ℕ := 45
def profit_per_meter : ℝ := 12
def cost_price_per_meter : ℝ := 88
def total_selling_price : ℝ := 4500

theorem selling_price_correct :
  (cost_price_per_meter * meters_of_cloth) + (profit_per_meter * meters_of_cloth) = total_selling_price :=
by
  sorry

end selling_price_correct_l242_242071


namespace find_annual_interest_rate_l242_242771

noncomputable def annual_interest_rate (P A n t : ℝ) : ℝ :=
  2 * ((A / P)^(1 / (n * t)) - 1)

theorem find_annual_interest_rate :
  Π (P A : ℝ) (n t : ℕ), P = 600 → A = 760 → n = 2 → t = 4 →
  annual_interest_rate P A n t = 0.06020727 :=
by
  intros P A n t hP hA hn ht
  rw [hP, hA, hn, ht]
  unfold annual_interest_rate
  sorry

end find_annual_interest_rate_l242_242771


namespace train_length_equals_sixty_two_point_five_l242_242087

-- Defining the conditions
noncomputable def calculate_train_length (speed_faster_train : ℝ) (speed_slower_train : ℝ) (time_seconds : ℝ) : ℝ :=
  let relative_speed_kmh := speed_faster_train - speed_slower_train
  let relative_speed_ms := (relative_speed_kmh * 5) / 18
  let distance_covered := relative_speed_ms * time_seconds
  distance_covered / 2

theorem train_length_equals_sixty_two_point_five :
  calculate_train_length 46 36 45 = 62.5 :=
sorry

end train_length_equals_sixty_two_point_five_l242_242087


namespace gcd_of_72_120_168_l242_242534

theorem gcd_of_72_120_168 : Nat.gcd (Nat.gcd 72 120) 168 = 24 := 
by
  sorry

end gcd_of_72_120_168_l242_242534


namespace units_digit_G_1000_l242_242636

def G (n : ℕ) : ℕ := 3 ^ (3 ^ n) + 1

theorem units_digit_G_1000 : G 1000 % 10 = 2 :=
by
  sorry

end units_digit_G_1000_l242_242636


namespace combined_speed_in_still_water_l242_242036

theorem combined_speed_in_still_water 
  (U1 D1 U2 D2 : ℝ) 
  (hU1 : U1 = 30) 
  (hD1 : D1 = 60) 
  (hU2 : U2 = 40) 
  (hD2 : D2 = 80) 
  : (U1 + D1) / 2 + (U2 + D2) / 2 = 105 := 
by 
  sorry

end combined_speed_in_still_water_l242_242036


namespace benny_cards_left_l242_242505

theorem benny_cards_left (n : ℕ) : ℕ :=
  (n + 4) / 2

end benny_cards_left_l242_242505


namespace sum_of_roots_l242_242578

theorem sum_of_roots (y1 y2 k m : ℝ) (h1 : y1 ≠ y2) (h2 : 5 * y1^2 - k * y1 = m) (h3 : 5 * y2^2 - k * y2 = m) : 
  y1 + y2 = k / 5 := 
by
  sorry

end sum_of_roots_l242_242578


namespace units_digit_sum_l242_242291

theorem units_digit_sum (h₁ : (24 : ℕ) % 10 = 4) 
                        (h₂ : (42 : ℕ) % 10 = 2) : 
  ((24^3 + 42^3) % 10 = 2) :=
by
  sorry

end units_digit_sum_l242_242291


namespace epicenter_distance_l242_242756

noncomputable def distance_from_epicenter (v1 v2 Δt: ℝ) : ℝ :=
  Δt / ((1 / v2) - (1 / v1))

theorem epicenter_distance : 
  distance_from_epicenter 5.94 3.87 11.5 = 128 := 
by
  -- The proof will use calculations shown in the solution.
  sorry

end epicenter_distance_l242_242756


namespace minimal_side_length_of_room_l242_242601

theorem minimal_side_length_of_room (a b : ℕ) (h₁ : a = 6) (h₂ : b = 8) :
  ∃ S : ℕ, S = 10 :=
by {
  sorry
}

end minimal_side_length_of_room_l242_242601


namespace harper_water_duration_l242_242398

theorem harper_water_duration
  (half_bottle_per_day : ℝ)
  (bottles_per_case : ℕ)
  (cost_per_case : ℝ)
  (total_spending : ℝ)
  (cases_bought : ℕ)
  (days_per_case : ℕ)
  (total_days : ℕ) :
  half_bottle_per_day = 1/2 →
  bottles_per_case = 24 →
  cost_per_case = 12 →
  total_spending = 60 →
  cases_bought = total_spending / cost_per_case →
  days_per_case = bottles_per_case * 2 →
  total_days = days_per_case * cases_bought →
  total_days = 240 :=
by
  -- The proof will be added here
  sorry

end harper_water_duration_l242_242398


namespace smallest_number_exceeding_triangle_perimeter_l242_242630

theorem smallest_number_exceeding_triangle_perimeter (a b : ℕ) (a_eq_7 : a = 7) (b_eq_21 : b = 21) :
  ∃ P : ℕ, (∀ c : ℝ, 14 < c ∧ c < 28 → a + b + c < P) ∧ P = 56 := by
  sorry

end smallest_number_exceeding_triangle_perimeter_l242_242630


namespace problem_statement_l242_242231

theorem problem_statement
  (c d : ℕ)
  (h_factorization : ∀ x, x^2 - 18 * x + 72 = (x - c) * (x - d))
  (h_c_nonnegative : c ≥ 0)
  (h_d_nonnegative : d ≥ 0)
  (h_c_greater_d : c > d) :
  4 * d - c = 12 :=
sorry

end problem_statement_l242_242231


namespace slips_with_3_l242_242797

theorem slips_with_3 (x : ℤ) 
    (h1 : 15 > 0) 
    (h2 : 3 > 0 ∧ 9 > 0) 
    (h3 : (3 * x + 9 * (15 - x)) / 15 = 5) : 
    x = 10 := 
sorry

end slips_with_3_l242_242797


namespace chance_Z_winning_l242_242254

-- Given conditions as Lean definitions
def p_x : ℚ := 1 / (3 + 1)
def p_y : ℚ := 3 / (2 + 3)
def p_z : ℚ := 1 - (p_x + p_y)

-- Theorem statement: Prove the equivalence of the winning ratio for Z
theorem chance_Z_winning : 
  p_z = 3 / (3 + 17) :=
by
  -- Since we include no proof, we use sorry to indicate it
  sorry

end chance_Z_winning_l242_242254


namespace largest_proper_fraction_smallest_improper_fraction_smallest_mixed_number_l242_242755

-- Definitions based on conditions
def isProperFraction (n d : ℕ) : Prop := n < d
def isImproperFraction (n d : ℕ) : Prop := n ≥ d
def isMixedNumber (w n d : ℕ) : Prop := w > 0 ∧ isProperFraction n d

-- Fractional part is 1/9, meaning all fractions considered have part = 1/9
def fractionalPart := 1 / 9

-- Lean 4 statements to verify the correct answers
theorem largest_proper_fraction : ∃ n d : ℕ, fractionalPart = n / d ∧ isProperFraction n d ∧ (n, d) = (8, 9) := sorry

theorem smallest_improper_fraction : ∃ n d : ℕ, fractionalPart = n / d ∧ isImproperFraction n d ∧ (n, d) = (9, 9) := sorry

theorem smallest_mixed_number : ∃ w n d : ℕ, fractionalPart = n / d ∧ isMixedNumber w n d ∧ ((w, n, d) = (1, 1, 9) ∨ (w, n, d) = (10, 9)) := sorry

end largest_proper_fraction_smallest_improper_fraction_smallest_mixed_number_l242_242755


namespace collinear_points_sum_l242_242290

variables {a b : ℝ}

/-- If the points (1, a, b), (a, b, 3), and (b, 3, a) are collinear, then b + a = 3.
-/
theorem collinear_points_sum (h : ∃ k : ℝ, 
  (a - 1, b - a, 3 - b) = k • (b - 1, 3 - a, a - b)) : b + a = 3 :=
sorry

end collinear_points_sum_l242_242290


namespace length_of_the_bridge_l242_242638

-- Conditions
def train_length : ℝ := 80
def train_speed_kmh : ℝ := 45
def crossing_time_seconds : ℝ := 30

-- Conversion factor
def km_to_m : ℝ := 1000
def hr_to_s : ℝ := 3600

-- Calculation
noncomputable def train_speed_ms : ℝ := train_speed_kmh * km_to_m / hr_to_s
noncomputable def total_distance : ℝ := train_speed_ms * crossing_time_seconds
noncomputable def bridge_length : ℝ := total_distance - train_length

-- Proof statement
theorem length_of_the_bridge : bridge_length = 295 :=
by
  sorry

end length_of_the_bridge_l242_242638


namespace sacks_harvested_per_section_l242_242517

theorem sacks_harvested_per_section (total_sacks : ℕ) (sections : ℕ) (sacks_per_section : ℕ) 
  (h1 : total_sacks = 360) 
  (h2 : sections = 8) 
  (h3 : total_sacks = sections * sacks_per_section) :
  sacks_per_section = 45 :=
by sorry

end sacks_harvested_per_section_l242_242517


namespace power_of_7_mod_8_l242_242451

theorem power_of_7_mod_8 : 7^123 % 8 = 7 :=
by sorry

end power_of_7_mod_8_l242_242451


namespace challenge_Jane_l242_242564

def is_vowel (c : Char) : Prop :=
  c = 'A' ∨ c = 'E' ∨ c = 'I' ∨ c = 'O' ∨ c = 'U'

def is_even (n : ℕ) : Prop :=
  n % 2 = 0

def card_pairs : List (Char ⊕ ℕ) :=
  [Sum.inl 'A', Sum.inl 'T', Sum.inl 'U', Sum.inr 5, Sum.inr 8, Sum.inr 10, Sum.inr 14]

def Jane_claim (c : Char ⊕ ℕ) : Prop :=
  match c with
  | Sum.inl v => is_vowel v → ∃ n, Sum.inr n ∈ card_pairs ∧ is_even n
  | Sum.inr n => false

theorem challenge_Jane (cards : List (Char ⊕ ℕ)) (h : card_pairs = cards) :
  ∃ c ∈ cards, c = Sum.inr 5 ∧ ¬Jane_claim (Sum.inr 5) :=
sorry

end challenge_Jane_l242_242564


namespace simpleInterest_500_l242_242565

def simpleInterest (P R T : ℝ) : ℝ := P * R * T

theorem simpleInterest_500 :
  simpleInterest 10000 0.05 1 = 500 :=
by
  sorry

end simpleInterest_500_l242_242565


namespace units_digit_m_squared_plus_2_pow_m_l242_242710

-- Define the value of m
def m : ℕ := 2023^2 + 2^2023

-- Define the property we need to prove
theorem units_digit_m_squared_plus_2_pow_m :
  ((m^2 + 2^m) % 10) = 7 :=
by
  sorry

end units_digit_m_squared_plus_2_pow_m_l242_242710


namespace minimum_value_x_squared_plus_y_squared_l242_242495

-- We define our main proposition in Lean
theorem minimum_value_x_squared_plus_y_squared (x y : ℝ) 
  (h : (x + 5)^2 + (y - 12)^2 = 196) : x^2 + y^2 ≥ 169 :=
sorry

end minimum_value_x_squared_plus_y_squared_l242_242495


namespace complement_of_A_with_respect_to_U_l242_242222

-- Definitions
def U : Set ℤ := {-2, -1, 1, 3, 5}
def A : Set ℤ := {-1, 3}

-- Statement of the problem
theorem complement_of_A_with_respect_to_U :
  (U \ A) = {-2, 1, 5} := 
by
  sorry

end complement_of_A_with_respect_to_U_l242_242222


namespace product_of_all_possible_values_l242_242191

theorem product_of_all_possible_values (x : ℝ) :
  (|16 / x + 4| = 3) → ((x = -16 ∨ x = -16 / 7) →
  (x_1 = -16 ∧ x_2 = -16 / 7) →
  (x_1 * x_2 = 256 / 7)) :=
sorry

end product_of_all_possible_values_l242_242191


namespace freddy_call_duration_l242_242621

theorem freddy_call_duration (total_cost : ℕ) (local_cost_per_minute : ℕ) (international_cost_per_minute : ℕ) (local_duration : ℕ)
  (total_cost_eq : total_cost = 1000) -- cost in cents
  (local_cost_eq : local_cost_per_minute = 5)
  (international_cost_eq : international_cost_per_minute = 25)
  (local_duration_eq : local_duration = 45) :
  (total_cost - local_duration * local_cost_per_minute) / international_cost_per_minute = 31 :=
by
  sorry

end freddy_call_duration_l242_242621


namespace simplify_fraction_l242_242481

theorem simplify_fraction :
  ((1 / 4) + (1 / 6)) / ((3 / 8) - (1 / 3)) = 10 := by
  sorry

end simplify_fraction_l242_242481


namespace minimum_abs_a_plus_b_l242_242208

theorem minimum_abs_a_plus_b {a b : ℤ} (h1 : |a| < |b|) (h2 : |b| ≤ 4) : ∃ (a b : ℤ), |a| + b = -4 :=
by
  sorry

end minimum_abs_a_plus_b_l242_242208


namespace periodic_decimal_to_fraction_l242_242472

theorem periodic_decimal_to_fraction : (0.7 + 0.32 : ℝ) == (1013 / 990 : ℝ) := by
  sorry

end periodic_decimal_to_fraction_l242_242472


namespace leak_time_to_empty_cistern_l242_242665

theorem leak_time_to_empty_cistern :
  (1/6 - 1/8) = 1/24 → (1 / (1/24)) = 24 := by
sorry

end leak_time_to_empty_cistern_l242_242665


namespace sum_of_coefficients_l242_242817

theorem sum_of_coefficients (a a_1 a_2 a_3 a_4 a_5 a_6 : ℤ) :
  (∀ x : ℤ, (1 + x)^6 = a + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6) →
  a = 1 →
  a_1 + a_2 + a_3 + a_4 + a_5 + a_6 = 63 :=
by
  intros h ha
  sorry

end sum_of_coefficients_l242_242817


namespace solve_for_k_l242_242035

theorem solve_for_k (k : ℚ) : 
  (∃ x : ℚ, (3 * x - 6 = 0) ∧ (2 * x - 5 * k = 11)) → k = -7/5 :=
by 
  intro h
  cases' h with x hx
  have hx1 : x = 2 := by linarith
  have hx2 : x = 11 / 2 + 5 / 2 * k := by linarith
  linarith

end solve_for_k_l242_242035


namespace pascal_sum_difference_l242_242463

open BigOperators

noncomputable def a_i (i : ℕ) := Nat.choose 3005 i
noncomputable def b_i (i : ℕ) := Nat.choose 3006 i
noncomputable def c_i (i : ℕ) := Nat.choose 3007 i

theorem pascal_sum_difference :
  (∑ i in Finset.range 3007, (b_i i) / (c_i i)) - (∑ i in Finset.range 3006, (a_i i) / (b_i i)) = 1 / 2 := by
  sorry

end pascal_sum_difference_l242_242463


namespace bacteria_growth_time_l242_242667

theorem bacteria_growth_time (n0 : ℕ) (n : ℕ) (rate : ℕ) (time_step : ℕ) (final : ℕ)
  (h0 : n0 = 200)
  (h1 : rate = 3)
  (h2 : time_step = 5)
  (h3 : n = n0 * rate ^ final)
  (h4 : n = 145800) :
  final = 30 := 
sorry

end bacteria_growth_time_l242_242667


namespace find_x_when_y_is_sqrt_8_l242_242315

theorem find_x_when_y_is_sqrt_8
  (x y : ℝ)
  (h : ∀ x y : ℝ, (x^2 * y^4 = 1600) ↔ (x = 10 ∧ y = 2)) :
  x = 5 :=
by
  sorry

end find_x_when_y_is_sqrt_8_l242_242315


namespace fabric_delivered_on_monday_amount_l242_242474

noncomputable def cost_per_yard : ℝ := 2
noncomputable def earnings : ℝ := 140

def fabric_delivered_on_monday (x : ℝ) : Prop :=
  let tuesday := 2 * x
  let wednesday := (1 / 4) * tuesday
  let total_yards := x + tuesday + wednesday
  let total_earnings := total_yards * cost_per_yard
  total_earnings = earnings

theorem fabric_delivered_on_monday_amount : ∃ x : ℝ, fabric_delivered_on_monday x ∧ x = 20 :=
by sorry

end fabric_delivered_on_monday_amount_l242_242474


namespace remainder_when_divided_by_9_l242_242739

noncomputable def base12_to_dec (x : ℕ) : ℕ :=
  (1 * 12^3) + (5 * 12^2) + (3 * 12) + 4
  
theorem remainder_when_divided_by_9 : base12_to_dec (1534) % 9 = 2 := by
  sorry

end remainder_when_divided_by_9_l242_242739


namespace max_sum_abc_l242_242608

theorem max_sum_abc (a b c : ℝ) (h : a + b + c = a^2 + b^2 + c^2) : a + b + c ≤ 3 :=
sorry

end max_sum_abc_l242_242608


namespace max_value_harmonic_series_l242_242143

theorem max_value_harmonic_series (k l m : ℕ) (hk : 0 < k) (hl : 0 < l) (hm : 0 < m)
  (h : 1/k + 1/l + 1/m < 1) : 
  (1/2 + 1/3 + 1/7) = 41/42 := 
sorry

end max_value_harmonic_series_l242_242143


namespace stationery_difference_l242_242401

theorem stationery_difference :
  let georgia := 25
  let lorene := 3 * georgia
  lorene - georgia = 50 :=
by
  let georgia := 25
  let lorene := 3 * georgia
  show lorene - georgia = 50
  sorry

end stationery_difference_l242_242401


namespace max_sum_of_arithmetic_sequence_l242_242073

theorem max_sum_of_arithmetic_sequence 
  (a : ℕ → ℤ) 
  (S : ℕ → ℤ)
  (h_arithmetic : ∀ n, a (n + 1) - a n = a 1 - a 0)
  (h_a3 : a 3 = 7)
  (h_a1_a7 : a 1 + a 7 = 10)
  (h_S : ∀ n, S n = (n * (2 * a 1 + (n - 1) * (a 1 - a 0))) / 2) :
  ∃ n, S n = S 6 ∧ (∀ m, S m ≤ S 6) :=
sorry

end max_sum_of_arithmetic_sequence_l242_242073


namespace triangle_inequality_range_x_l242_242625

theorem triangle_inequality_range_x (x : ℝ) :
  let a := 3;
  let b := 8;
  let c := 1 + 2 * x;
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a) ↔ (2 < x ∧ x < 5) :=
by
  sorry

end triangle_inequality_range_x_l242_242625


namespace Yanna_apples_l242_242310

def total_apples_bought (given_to_zenny : ℕ) (given_to_andrea : ℕ) (kept : ℕ) : ℕ :=
  given_to_zenny + given_to_andrea + kept

theorem Yanna_apples {given_to_zenny given_to_andrea kept total : ℕ}:
  given_to_zenny = 18 →
  given_to_andrea = 6 →
  kept = 36 →
  total_apples_bought given_to_zenny given_to_andrea kept = 60 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  rfl

end Yanna_apples_l242_242310


namespace least_number_subtracted_l242_242522

theorem least_number_subtracted (n m k : ℕ) (h1 : n = 3830) (h2 : k = 15) (h3 : n % k = m) (h4 : m = 5) : 
  (n - m) % k = 0 :=
by
  sorry

end least_number_subtracted_l242_242522


namespace original_price_of_coat_l242_242345

theorem original_price_of_coat (P : ℝ) (h : 0.70 * P = 350) : P = 500 :=
sorry

end original_price_of_coat_l242_242345


namespace budget_for_bulbs_l242_242202

theorem budget_for_bulbs (num_crocus_bulbs : ℕ) (cost_per_crocus : ℝ) (budget : ℝ)
  (h1 : num_crocus_bulbs = 22)
  (h2 : cost_per_crocus = 0.35)
  (h3 : budget = num_crocus_bulbs * cost_per_crocus) :
  budget = 7.70 :=
sorry

end budget_for_bulbs_l242_242202


namespace expression_equals_eight_l242_242775

theorem expression_equals_eight
  (a b c : ℝ)
  (h1 : a + b = 2 * c)
  (h2 : b + c = 2 * a)
  (h3 : a + c = 2 * b) :
  (a + b) * (b + c) * (a + c) / (a * b * c) = 8 := by
  sorry

end expression_equals_eight_l242_242775


namespace interest_rate_l242_242198

noncomputable def compoundInterest (P : ℕ) (r : ℕ) (t : ℕ) : ℚ :=
  P * ((1 + r / 100 : ℚ) ^ t) - P

noncomputable def simpleInterest (P : ℕ) (r : ℕ) (t : ℕ) : ℚ :=
  P * r * t / 100

theorem interest_rate (P t : ℕ) (D : ℚ) (r : ℕ) :
  P = 10000 → t = 2 → D = 49 →
  compoundInterest P r t - simpleInterest P r t = D → r = 7 := by
  sorry

end interest_rate_l242_242198


namespace handshakes_total_l242_242726

def num_couples : ℕ := 15
def total_people : ℕ := 30
def men : ℕ := 15
def women : ℕ := 15
def youngest_man_handshakes : ℕ := 0
def men_handshakes : ℕ := (14 * 13) / 2
def men_women_handshakes : ℕ := 15 * 14

theorem handshakes_total : men_handshakes + men_women_handshakes = 301 :=
by
  -- Proof goes here
  sorry

end handshakes_total_l242_242726


namespace RouteB_quicker_than_RouteA_l242_242258

def RouteA_segment1_time : ℚ := 4 / 40 -- time in hours
def RouteA_segment2_time : ℚ := 4 / 20 -- time in hours
def RouteA_total_time : ℚ := RouteA_segment1_time + RouteA_segment2_time -- total time in hours

def RouteB_segment1_time : ℚ := 6 / 35 -- time in hours
def RouteB_segment2_time : ℚ := 1 / 15 -- time in hours
def RouteB_total_time : ℚ := RouteB_segment1_time + RouteB_segment2_time -- total time in hours

def time_difference_minutes : ℚ := (RouteA_total_time - RouteB_total_time) * 60 -- difference in minutes

theorem RouteB_quicker_than_RouteA : time_difference_minutes = 3.71 := by
  sorry

end RouteB_quicker_than_RouteA_l242_242258


namespace alice_needs_136_life_vests_l242_242694

-- Definitions from the problem statement
def num_classes : ℕ := 4
def students_per_class : ℕ := 40
def instructors_per_class : ℕ := 10
def life_vest_probability : ℝ := 0.40

-- Calculate the total number of people
def total_people := num_classes * (students_per_class + instructors_per_class)

-- Calculate the expected number of students with life vests
def students_with_life_vests := (students_per_class : ℝ) * life_vest_probability
def total_students_with_life_vests := num_classes * students_with_life_vests

-- Calculate the number of life vests needed
def life_vests_needed := total_people - total_students_with_life_vests

-- Proof statement (missing the actual proof)
theorem alice_needs_136_life_vests : life_vests_needed = 136 := by
  sorry

end alice_needs_136_life_vests_l242_242694


namespace regular_price_of_fish_l242_242641

theorem regular_price_of_fish (discounted_price_per_quarter_pound : ℝ)
  (discount : ℝ) (hp1 : discounted_price_per_quarter_pound = 2) (hp2 : discount = 0.4) :
  ∃ x : ℝ, x = (40 / 3) :=
by
  sorry

end regular_price_of_fish_l242_242641


namespace min_value_expression_l242_242743

noncomputable def min_expression (a b c : ℝ) : ℝ :=
  (9 / a) + (16 / b) + (25 / c)

theorem min_value_expression :
  ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 → a + b + c = 6 →
  min_expression a b c ≥ 18 :=
by
  intro a b c ha hb hc habc
  sorry

end min_value_expression_l242_242743


namespace find_k_l242_242331

def f (a b c x : Int) : Int := a * x^2 + b * x + c

theorem find_k (a b c k : Int)
  (h₁ : f a b c 2 = 0)
  (h₂ : 100 < f a b c 7 ∧ f a b c 7 < 110)
  (h₃ : 120 < f a b c 8 ∧ f a b c 8 < 130)
  (h₄ : 6000 * k < f a b c 100 ∧ f a b c 100 < 6000 * (k + 1)) :
  k = 0 := 
sorry

end find_k_l242_242331


namespace smallest_n_for_sqrt_50n_is_integer_l242_242337

theorem smallest_n_for_sqrt_50n_is_integer :
  ∃ n : ℕ, n > 0 ∧ (∃ k : ℕ, (50 * n) = k * k) ∧ n = 2 :=
by
  sorry

end smallest_n_for_sqrt_50n_is_integer_l242_242337


namespace analytical_expression_smallest_positive_period_min_value_max_value_l242_242468

noncomputable def P (x : ℝ) : ℝ × ℝ :=
  (Real.cos (2 * x) + 1, 1)

noncomputable def Q (x : ℝ) : ℝ × ℝ :=
  (1, Real.sqrt 3 * Real.sin (2 * x) + 1)

noncomputable def f (x : ℝ) : ℝ :=
  (P x).1 * (Q x).1 + (P x).2 * (Q x).2

theorem analytical_expression (x : ℝ) : 
  f x = 2 * Real.sin (2 * x + Real.pi / 6) + 2 :=
sorry

theorem smallest_positive_period : 
  ∀ x : ℝ, f (x + Real.pi) = f x :=
sorry

theorem min_value : 
  ∃ x : ℝ, f x = 0 :=
sorry

theorem max_value : 
  ∃ y : ℝ, f y = 4 :=
sorry

end analytical_expression_smallest_positive_period_min_value_max_value_l242_242468


namespace decreasing_function_range_l242_242757

theorem decreasing_function_range (k : ℝ) : (∀ x : ℝ, k + 2 < 0) ↔ k < -2 :=
by
  sorry

end decreasing_function_range_l242_242757


namespace F_2_f_3_equals_341_l242_242449

def f (a : ℕ) : ℕ := a^2 - 2
def F (a b : ℕ) : ℕ := b^3 - a

theorem F_2_f_3_equals_341 : F 2 (f 3) = 341 := by
  sorry

end F_2_f_3_equals_341_l242_242449


namespace prove_interest_rates_equal_l242_242576

noncomputable def interest_rates_equal : Prop :=
  let initial_savings := 1000
  let savings_simple := initial_savings / 2
  let savings_compound := initial_savings / 2
  let simple_interest_earned := 100
  let compound_interest_earned := 105
  let time := 2
  let r_s := simple_interest_earned / (savings_simple * time)
  let r_c := (compound_interest_earned / savings_compound + 1) ^ (1 / time) - 1
  r_s = r_c

theorem prove_interest_rates_equal : interest_rates_equal :=
  sorry

end prove_interest_rates_equal_l242_242576


namespace order_of_exponents_l242_242559

theorem order_of_exponents (p q r : ℕ) (hp : p = 2^3009) (hq : q = 3^2006) (hr : r = 5^1003) : r < p ∧ p < q :=
by {
  sorry -- Proof will go here
}

end order_of_exponents_l242_242559


namespace abs_lt_one_iff_sq_lt_one_l242_242274

variable {x : ℝ}

theorem abs_lt_one_iff_sq_lt_one : |x| < 1 ↔ x^2 < 1 := sorry

end abs_lt_one_iff_sq_lt_one_l242_242274


namespace x_minus_q_eq_three_l242_242571

theorem x_minus_q_eq_three (x q : ℝ) (h1 : |x - 3| = q) (h2 : x > 3) : x - q = 3 :=
by 
  sorry

end x_minus_q_eq_three_l242_242571


namespace probability_one_marble_each_color_l242_242496

theorem probability_one_marble_each_color :
  let total_marbles := 9
  let total_ways := Nat.choose total_marbles 3
  let favorable_ways := 3 * 3 * 3
  let probability := favorable_ways / total_ways
  probability = 9 / 28 :=
by
  sorry

end probability_one_marble_each_color_l242_242496


namespace find_coefficients_l242_242292

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Definitions based on conditions
def A' (A B : V) : V := (3 : ℝ) • (B - A) + A
def B' (B C : V) : V := (3 : ℝ) • (C - B) + C

-- The problem statement
theorem find_coefficients (A A' B B' : V) (p q r : ℝ) 
  (hB : B = (1/4 : ℝ) • A + (3/4 : ℝ) • A') 
  (hC : C = (1/4 : ℝ) • B + (3/4 : ℝ) • B') : 
  ∃ (p q r : ℝ), A = p • A' + q • B + r • B' ∧ p = 4/13 ∧ q = 12/13 ∧ r = 48/13 :=
sorry

end find_coefficients_l242_242292


namespace find_sum_xyz_l242_242613

-- Define the problem
def system_of_equations (x y z : ℝ) : Prop :=
  x^2 + x * y + y^2 = 27 ∧
  y^2 + y * z + z^2 = 9 ∧
  z^2 + z * x + x^2 = 36

-- The main theorem to be proved
theorem find_sum_xyz (x y z : ℝ) (h : system_of_equations x y z) : 
  x * y + y * z + z * x = 18 :=
sorry

end find_sum_xyz_l242_242613


namespace roof_area_l242_242480

theorem roof_area (w l : ℕ) (h1 : l = 4 * w) (h2 : l - w = 42) : l * w = 784 :=
by
  sorry

end roof_area_l242_242480


namespace greatest_integer_leq_l242_242152

theorem greatest_integer_leq (a b : ℝ) (ha : a = 5^150) (hb : b = 3^150) (c d : ℝ) (hc : c = 5^147) (hd : d = 3^147):
  ⌊ (a + b) / (c + d) ⌋ = 124 := 
sorry

end greatest_integer_leq_l242_242152


namespace no_negative_roots_but_at_least_one_positive_root_l242_242158

def f (x : ℝ) : ℝ := x^6 - 3 * x^5 - 6 * x^3 - x + 8

theorem no_negative_roots_but_at_least_one_positive_root :
  (∀ x : ℝ, x < 0 → f x ≠ 0) ∧ (∃ x : ℝ, x > 0 ∧ f x = 0) :=
by {
  sorry
}

end no_negative_roots_but_at_least_one_positive_root_l242_242158


namespace jane_rejects_percent_l242_242788

theorem jane_rejects_percent :
  -- Declare the conditions as hypotheses
  ∀ (P : ℝ) (J : ℝ) (john_frac_reject : ℝ) (total_reject_percent : ℝ) (jane_inspect_frac : ℝ),
  john_frac_reject = 0.005 →
  total_reject_percent = 0.0075 →
  jane_inspect_frac = 5 / 6 →
  -- Given the rejection equation
  (john_frac_reject * (1 / 6) * P + (J / 100) * jane_inspect_frac * P = total_reject_percent * P) →
  -- Prove that Jane rejected 0.8% of the products she inspected
  J = 0.8 :=
by {
  sorry
}

end jane_rejects_percent_l242_242788


namespace find_a_n_l242_242228

def S (n : ℕ) : ℕ := 2^(n+1) - 1

def a_n (n : ℕ) : ℕ :=
  if n = 1 then 3 else 2^n

theorem find_a_n (n : ℕ) : a_n n = if n = 1 then 3 else 2^n :=
  sorry

end find_a_n_l242_242228


namespace solve_inequality_l242_242432

theorem solve_inequality :
  {x : ℝ | (1 / (x * (x + 1)) - 1 / ((x + 1) * (x + 2)) < 1 / 2)} =
  {x : ℝ | (x < -2) ∨ (-1 < x ∧ x < 0) ∨ (1 < x)} :=
by
  sorry

end solve_inequality_l242_242432


namespace total_visible_legs_l242_242634

-- Defining the conditions
def num_crows : ℕ := 4
def num_pigeons : ℕ := 3
def num_flamingos : ℕ := 5
def num_sparrows : ℕ := 8

def legs_per_crow : ℕ := 2
def legs_per_pigeon : ℕ := 2
def legs_per_flamingo : ℕ := 3
def legs_per_sparrow : ℕ := 2

-- Formulating the theorem that we need to prove
theorem total_visible_legs :
  (num_crows * legs_per_crow) +
  (num_pigeons * legs_per_pigeon) +
  (num_flamingos * legs_per_flamingo) +
  (num_sparrows * legs_per_sparrow) = 45 := by sorry

end total_visible_legs_l242_242634


namespace domain_of_function_y_eq_sqrt_2x_3_div_x_2_l242_242547

def domain (x : ℝ) : Prop :=
  (2 * x - 3 ≥ 0) ∧ (x ≠ 2)

theorem domain_of_function_y_eq_sqrt_2x_3_div_x_2 :
  ∀ x : ℝ, domain x ↔ ((x ≥ 3 / 2) ∧ (x ≠ 2)) :=
by
  sorry

end domain_of_function_y_eq_sqrt_2x_3_div_x_2_l242_242547


namespace total_passengers_per_day_l242_242234

-- Define the conditions
def airplanes : ℕ := 5
def rows_per_airplane : ℕ := 20
def seats_per_row : ℕ := 7
def flights_per_day : ℕ := 2

-- Define the proof problem
theorem total_passengers_per_day : 
  (airplanes * rows_per_airplane * seats_per_row * flights_per_day) = 1400 := 
by 
  sorry

end total_passengers_per_day_l242_242234


namespace average_age_of_other_9_students_l242_242221

variable (total_students : ℕ) (total_average_age : ℝ) (group1_students : ℕ) (group1_average_age : ℝ) (age_student12 : ℝ) (group2_students : ℕ)

theorem average_age_of_other_9_students 
  (h1 : total_students = 16) 
  (h2 : total_average_age = 16) 
  (h3 : group1_students = 5) 
  (h4 : group1_average_age = 14) 
  (h5 : age_student12 = 42) 
  (h6 : group2_students = 9) : 
  (group1_students * group1_average_age + group2_students * 16 + age_student12) / total_students = total_average_age := by
  sorry

end average_age_of_other_9_students_l242_242221


namespace intersection_with_y_axis_is_correct_l242_242531

theorem intersection_with_y_axis_is_correct (x y : ℝ) (h : y = 5 * x + 1) (hx : x = 0) : y = 1 :=
by
  sorry

end intersection_with_y_axis_is_correct_l242_242531


namespace Daniel_had_more_than_200_marbles_at_day_6_l242_242834

noncomputable def marbles (k : ℕ) : ℕ :=
  5 * 2^k

theorem Daniel_had_more_than_200_marbles_at_day_6 :
  ∃ k : ℕ, marbles k > 200 ∧ ∀ m < k, marbles m ≤ 200 :=
by
  sorry

end Daniel_had_more_than_200_marbles_at_day_6_l242_242834


namespace cos_alpha_in_fourth_quadrant_l242_242066

theorem cos_alpha_in_fourth_quadrant (α : ℝ) (P : ℝ × ℝ) (h_angle_quadrant : α > 3 * Real.pi / 2 ∧ α < 2 * Real.pi)
(h_point : P = (Real.sqrt 5, 2)) (h_sin : Real.sin α = (Real.sqrt 2 / 4) * 2) :
  Real.cos α = Real.sqrt 10 / 4 :=
sorry

end cos_alpha_in_fourth_quadrant_l242_242066


namespace count_solutions_eq_4_l242_242476

theorem count_solutions_eq_4 :
  ∀ x : ℝ, (x^2 - 5)^2 = 16 → x = 3 ∨ x = -3 ∨ x = 1 ∨ x = -1  := sorry

end count_solutions_eq_4_l242_242476


namespace range_a_for_false_proposition_l242_242161

theorem range_a_for_false_proposition :
  {a : ℝ | ¬ ∃ x : ℝ, x^2 + (1 - a) * x < 0} = {1} :=
sorry

end range_a_for_false_proposition_l242_242161


namespace first_box_oranges_l242_242688

theorem first_box_oranges (x : ℕ) (h : x + (x + 2) + (x + 4) + (x + 6) + (x + 8) + (x + 10) + (x + 12) = 120) : x = 11 :=
sorry

end first_box_oranges_l242_242688


namespace find_f_neg_2_l242_242267

noncomputable def f (x : ℝ) : ℝ :=
if x >= 0 then 3^x - 1 else sorry -- we'll define this not for non-negative x properly later

def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = -f x

theorem find_f_neg_2 (hodd : is_odd_function f) (hpos : ∀ x : ℝ, 0 ≤ x → f x = 3^x - 1) :
  f (-2) = -8 :=
by
  -- Proof omitted
  sorry

end find_f_neg_2_l242_242267


namespace original_total_price_l242_242051

-- Definitions of the original prices
def original_price_candy_box : ℕ := 10
def original_price_soda : ℕ := 6
def original_price_chips : ℕ := 4
def original_price_chocolate_bar : ℕ := 2

-- Mathematical problem statement
theorem original_total_price :
  original_price_candy_box + original_price_soda + original_price_chips + original_price_chocolate_bar = 22 :=
by
  sorry

end original_total_price_l242_242051


namespace train_crosses_platform_in_15_seconds_l242_242720

-- Definitions based on conditions
def length_of_train : ℝ := 330 -- in meters
def tunnel_length : ℝ := 1200 -- in meters
def time_to_cross_tunnel : ℝ := 45 -- in seconds
def platform_length : ℝ := 180 -- in meters

-- Definition based on the solution but directly asserting the correct answer.
def time_to_cross_platform : ℝ := 15 -- in seconds

-- Lean statement
theorem train_crosses_platform_in_15_seconds :
  (length_of_train + platform_length) / ((length_of_train + tunnel_length) / time_to_cross_tunnel) = time_to_cross_platform :=
by
  sorry

end train_crosses_platform_in_15_seconds_l242_242720


namespace find_a_l242_242660

-- Definitions and theorem statement
def A (a : ℝ) : Set ℝ := {2, a^2 - a + 1}
def B (a : ℝ) : Set ℝ := {3, a + 3}
def C (a : ℝ) : Set ℝ := {3}

theorem find_a (a : ℝ) : A a ∩ B a = C a → a = 2 :=
by
  sorry

end find_a_l242_242660


namespace max_value_inequality_l242_242765

theorem max_value_inequality (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (hxyz : x + y + z = 3) :
  (x^2 + x * y + y^2) * (y^2 + y * z + z^2) * (z^2 + z * x + x^2) ≤ 27 := 
sorry

end max_value_inequality_l242_242765


namespace mark_speed_l242_242454

theorem mark_speed
  (chris_speed : ℕ)
  (distance_to_school : ℕ)
  (mark_total_distance : ℕ)
  (mark_time_longer : ℕ)
  (chris_speed_eq : chris_speed = 3)
  (distance_to_school_eq : distance_to_school = 9)
  (mark_total_distance_eq : mark_total_distance = 15)
  (mark_time_longer_eq : mark_time_longer = 2) :
  mark_total_distance / (distance_to_school / chris_speed + mark_time_longer) = 3 := 
by
  sorry 

end mark_speed_l242_242454


namespace total_flowers_correct_l242_242753

def rosa_original_flowers : ℝ := 67.5
def andre_gifted_flowers : ℝ := 90.75
def total_flowers (rosa : ℝ) (andre : ℝ) : ℝ := rosa + andre

theorem total_flowers_correct : total_flowers rosa_original_flowers andre_gifted_flowers = 158.25 :=
by 
  rw [total_flowers]
  sorry

end total_flowers_correct_l242_242753


namespace winning_candidate_percentage_l242_242424

theorem winning_candidate_percentage 
    (votes_winner : ℕ)
    (votes_total : ℕ)
    (votes_majority : ℕ)
    (H1 : votes_total = 900)
    (H2 : votes_majority = 360)
    (H3 : votes_winner - (votes_total - votes_winner) = votes_majority) :
    (votes_winner : ℕ) * 100 / (votes_total : ℕ) = 70 := by
    sorry

end winning_candidate_percentage_l242_242424


namespace one_thirds_in_fraction_l242_242689

theorem one_thirds_in_fraction : (11 / 5) / (1 / 3) = 33 / 5 := by
  sorry

end one_thirds_in_fraction_l242_242689


namespace jinsu_third_attempt_kicks_l242_242400

theorem jinsu_third_attempt_kicks
  (hoseok_kicks : ℕ) (jinsu_first_attempt : ℕ) (jinsu_second_attempt : ℕ) (required_kicks : ℕ) :
  hoseok_kicks = 48 →
  jinsu_first_attempt = 15 →
  jinsu_second_attempt = 15 →
  required_kicks = 19 →
  jinsu_first_attempt + jinsu_second_attempt + required_kicks > hoseok_kicks :=
by
  sorry

end jinsu_third_attempt_kicks_l242_242400


namespace determine_asymptotes_l242_242393

noncomputable def asymptotes_of_hyperbola (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : Prop :=
  (2 * a = 2 * Real.sqrt 2) ∧ (2 * b = 2) → 
  (∀ x y : ℝ, (y = x * (Real.sqrt 2 / 2) ∨ y = -x * (Real.sqrt 2 / 2)))

theorem determine_asymptotes (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (2 * a = 2 * Real.sqrt 2) ∧ (2 * b = 2) → 
  asymptotes_of_hyperbola a b ha hb :=
by
  intros h
  sorry

end determine_asymptotes_l242_242393


namespace infinite_equal_pairs_of_equal_terms_l242_242350

theorem infinite_equal_pairs_of_equal_terms {a : ℤ → ℤ}
  (h : ∀ n, a n = (a (n - 1) + a (n + 1)) / 4)
  (i j : ℤ) (hij : a i = a j) :
  ∃ (infinitely_many_pairs : ℕ → ℤ × ℤ), ∀ k, a (infinitely_many_pairs k).1 = a (infinitely_many_pairs k).2 :=
sorry

end infinite_equal_pairs_of_equal_terms_l242_242350


namespace trig_identity_l242_242215

theorem trig_identity (α : ℝ) (h : Real.tan α = 2 / 3) : 
  Real.sin (2 * α) - Real.cos (π - 2 * α) = 17 / 13 :=
by
  sorry

end trig_identity_l242_242215


namespace triangle_sides_fraction_sum_eq_one_l242_242195

theorem triangle_sides_fraction_sum_eq_one
  (a b c : ℝ)
  (h : a^2 + b^2 = c^2 + a * b) :
  a / (b + c) + b / (c + a) = 1 :=
sorry

end triangle_sides_fraction_sum_eq_one_l242_242195


namespace chickens_cheaper_than_buying_eggs_l242_242381

theorem chickens_cheaper_than_buying_eggs :
  ∃ W, W ≥ 80 ∧ 80 + W ≤ 2 * W :=
by
  sorry

end chickens_cheaper_than_buying_eggs_l242_242381


namespace steven_seeds_l242_242738

def average_seeds (fruit: String) : Nat :=
  match fruit with
  | "apple" => 6
  | "pear" => 2
  | "grape" => 3
  | "orange" => 10
  | "watermelon" => 300
  | _ => 0

def fruits := [("apple", 2), ("pear", 3), ("grape", 5), ("orange", 1), ("watermelon", 2)]

def required_seeds := 420

def total_seeds (fruit_list : List (String × Nat)) : Nat :=
  fruit_list.foldr (fun (fruit_qty : String × Nat) acc =>
    acc + (average_seeds fruit_qty.fst) * fruit_qty.snd) 0

theorem steven_seeds : total_seeds fruits - required_seeds = 223 := by
  sorry

end steven_seeds_l242_242738


namespace find_side_length_l242_242598

def hollow_cube_formula (n : ℕ) : ℕ :=
  6 * n^2 - (n^2 + 4 * (n - 2))

theorem find_side_length :
  ∃ n : ℕ, hollow_cube_formula n = 98 ∧ n = 9 :=
by
  sorry

end find_side_length_l242_242598


namespace trig_identity_l242_242589

theorem trig_identity : (Real.cos (15 * Real.pi / 180))^2 - (Real.sin (15 * Real.pi / 180))^2 = Real.sqrt 3 / 2 :=
by
  sorry

end trig_identity_l242_242589


namespace find_x_l242_242138

theorem find_x (a b x : ℝ) (h : ∀ a b, a * b = a + 2 * b) (H : 3 * (4 * x) = 6) : x = -5 / 4 :=
by
  sorry

end find_x_l242_242138


namespace razorback_shop_revenue_from_jerseys_zero_l242_242657

theorem razorback_shop_revenue_from_jerseys_zero:
  let num_tshirts := 20
  let num_jerseys := 64
  let revenue_per_tshirt := 215
  let total_revenue_tshirts := 4300
  let total_revenue := total_revenue_tshirts
  let revenue_from_jerseys := total_revenue - total_revenue_tshirts
  revenue_from_jerseys = 0 := by
  sorry

end razorback_shop_revenue_from_jerseys_zero_l242_242657


namespace find_a_of_extreme_at_1_l242_242302

noncomputable def f (a x : ℝ) : ℝ := a * x^3 - x - Real.log x

theorem find_a_of_extreme_at_1 :
  (∃ a : ℝ, ∃ f' : ℝ -> ℝ, (f' x = 3 * a * x^2 - 1 - 1/x) ∧ f' 1 = 0) →
  ∃ a : ℝ, a = 2 / 3 :=
by
  sorry

end find_a_of_extreme_at_1_l242_242302


namespace train_speed_in_km_per_hr_l242_242572

variables (L : ℕ) (t : ℕ) (train_speed : ℕ)

-- Conditions
def length_of_train : ℕ := 1050
def length_of_platform : ℕ := 1050
def crossing_time : ℕ := 1

-- Given calculation of speed in meters per minute
def speed_in_m_per_min : ℕ := (length_of_train + length_of_platform) / crossing_time

-- Conversion units
def meters_to_kilometers (m : ℕ) : ℕ := m / 1000
def minutes_to_hours (min : ℕ) : ℕ := min / 60

-- Speed in km/hr
def speed_in_km_per_hr : ℕ := speed_in_m_per_min * (meters_to_kilometers 1000) * (minutes_to_hours 60)

theorem train_speed_in_km_per_hr : speed_in_km_per_hr = 35 :=
by {
  -- We will include the proof steps here, but for now, we just assert with sorry.
  sorry
}

end train_speed_in_km_per_hr_l242_242572


namespace sphere_volume_ratio_l242_242509

theorem sphere_volume_ratio (r1 r2 : ℝ) (S1 S2 V1 V2 : ℝ) 
(h1 : S1 = 4 * Real.pi * r1^2)
(h2 : S2 = 4 * Real.pi * r2^2)
(h3 : V1 = (4 / 3) * Real.pi * r1^3)
(h4 : V2 = (4 / 3) * Real.pi * r2^3)
(h_surface_ratio : S1 / S2 = 2 / 3) :
V1 / V2 = (2 * Real.sqrt 6) / 9 :=
by
  sorry

end sphere_volume_ratio_l242_242509


namespace jacob_age_proof_l242_242301

theorem jacob_age_proof
  (drew_age maya_age peter_age : ℕ)
  (john_age : ℕ := 30)
  (jacob_age : ℕ) :
  (drew_age = maya_age + 5) →
  (peter_age = drew_age + 4) →
  (john_age = 30 ∧ john_age = 2 * maya_age) →
  (jacob_age + 2 = (peter_age + 2) / 2) →
  jacob_age = 11 :=
by
  sorry

end jacob_age_proof_l242_242301


namespace standard_circle_equation_l242_242119

theorem standard_circle_equation (x y : ℝ) :
  ∃ (h k r : ℝ), h = 2 ∧ k = -1 ∧ r = 3 ∧ (x - h)^2 + (y - k + 1)^2 = r^2 :=
by
  use 2, -1, 3
  simp
  sorry

end standard_circle_equation_l242_242119


namespace percent_chemical_a_in_mixture_l242_242293

-- Define the given problem parameters
def percent_chemical_a_in_solution_x : ℝ := 0.30
def percent_chemical_a_in_solution_y : ℝ := 0.40
def proportion_of_solution_x_in_mixture : ℝ := 0.80
def proportion_of_solution_y_in_mixture : ℝ := 1.0 - proportion_of_solution_x_in_mixture

-- Define what we need to prove: the percentage of chemical a in the mixture
theorem percent_chemical_a_in_mixture:
  (percent_chemical_a_in_solution_x * proportion_of_solution_x_in_mixture) + 
  (percent_chemical_a_in_solution_y * proportion_of_solution_y_in_mixture) = 0.32 
:= by sorry

end percent_chemical_a_in_mixture_l242_242293


namespace smallest_three_digit_multiple_of_13_l242_242259

theorem smallest_three_digit_multiple_of_13 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 13 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 13 = 0 → n ≤ m :=
⟨104, by sorry⟩

end smallest_three_digit_multiple_of_13_l242_242259


namespace perpendicular_vectors_k_value_l242_242183

theorem perpendicular_vectors_k_value (k : ℝ) (a b: ℝ × ℝ)
  (h_a : a = (-1, 3)) (h_b : b = (1, k)) (h_perp : (a.1 * b.1 + a.2 * b.2) = 0) :
  k = 1 / 3 :=
by
  sorry

end perpendicular_vectors_k_value_l242_242183


namespace mr_bodhi_adds_twenty_sheep_l242_242436

def cows : ℕ := 20
def foxes : ℕ := 15
def zebras : ℕ := 3 * foxes
def required_total : ℕ := 100

def sheep := required_total - (cows + foxes + zebras)

theorem mr_bodhi_adds_twenty_sheep : sheep = 20 :=
by
  -- Proof for the theorem is not required and is thus replaced with sorry.
  sorry

end mr_bodhi_adds_twenty_sheep_l242_242436


namespace ratio_third_first_l242_242684

theorem ratio_third_first (A B C : ℕ) (h1 : A + B + C = 110) (h2 : A = 2 * B) (h3 : B = 30) :
  C / A = 1 / 3 :=
by
  sorry

end ratio_third_first_l242_242684


namespace max_value_of_x_and_y_l242_242404

theorem max_value_of_x_and_y (x y : ℕ) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (h : (x - 4) * (x - 10) = 2 ^ y) : x + y ≤ 16 :=
sorry

end max_value_of_x_and_y_l242_242404


namespace solve_Diamond_l242_242133

theorem solve_Diamond :
  ∀ (Diamond : ℕ), (Diamond * 7 + 4 = Diamond * 8 + 1) → Diamond = 3 :=
by
  intros Diamond h
  sorry

end solve_Diamond_l242_242133


namespace right_triangle_leg_length_l242_242590

theorem right_triangle_leg_length (a b c : ℕ) (h₁ : a = 8) (h₂ : c = 17) (h₃ : a^2 + b^2 = c^2) : b = 15 := 
by
  sorry

end right_triangle_leg_length_l242_242590


namespace StatementA_incorrect_l242_242227

def f (n : ℕ) : ℕ := (n.factorial)^2

def g (x : ℕ) : ℕ := f (x + 1) / f x

theorem StatementA_incorrect (x : ℕ) (h : x = 1) : g x ≠ 4 := sorry

end StatementA_incorrect_l242_242227


namespace perimeter_of_region_l242_242333

noncomputable def side_length : ℝ := 2 / Real.pi

noncomputable def semicircle_perimeter : ℝ := 2

theorem perimeter_of_region (s : ℝ) (p : ℝ) (h1 : s = 2 / Real.pi) (h2 : p = 2) :
  4 * (p / 2) = 4 :=
by
  sorry

end perimeter_of_region_l242_242333


namespace total_emeralds_l242_242335

theorem total_emeralds (D R E : ℕ) 
  (h1 : 2 * D + 2 * E + 2 * R = 6)
  (h2 : R = D + 15) : 
  E = 12 :=
by
  -- Proof omitted
  sorry

end total_emeralds_l242_242335


namespace point_outside_circle_l242_242799

theorem point_outside_circle (D E F x0 y0 : ℝ) (h : (x0 + D / 2)^2 + (y0 + E / 2)^2 > (D^2 + E^2 - 4 * F) / 4) :
  x0^2 + y0^2 + D * x0 + E * y0 + F > 0 :=
sorry

end point_outside_circle_l242_242799


namespace cos_seven_pi_over_six_l242_242523

theorem cos_seven_pi_over_six :
  Real.cos (7 * Real.pi / 6) = - (Real.sqrt 3) / 2 :=
sorry

end cos_seven_pi_over_six_l242_242523


namespace orchard_yield_correct_l242_242530

-- Definitions for conditions
def gala3YrTreesYield : ℕ := 10 * 120
def gala2YrTreesYield : ℕ := 10 * 150
def galaTotalYield : ℕ := gala3YrTreesYield + gala2YrTreesYield

def fuji4YrTreesYield : ℕ := 5 * 180
def fuji5YrTreesYield : ℕ := 5 * 200
def fujiTotalYield : ℕ := fuji4YrTreesYield + fuji5YrTreesYield

def redhaven6YrTreesYield : ℕ := 15 * 50
def redhaven4YrTreesYield : ℕ := 15 * 60
def redhavenTotalYield : ℕ := redhaven6YrTreesYield + redhaven4YrTreesYield

def elberta2YrTreesYield : ℕ := 5 * 70
def elberta3YrTreesYield : ℕ := 5 * 75
def elberta5YrTreesYield : ℕ := 5 * 80
def elbertaTotalYield : ℕ := elberta2YrTreesYield + elberta3YrTreesYield + elberta5YrTreesYield

def appleTotalYield : ℕ := galaTotalYield + fujiTotalYield
def peachTotalYield : ℕ := redhavenTotalYield + elbertaTotalYield
def orchardTotalYield : ℕ := appleTotalYield + peachTotalYield

-- Theorem to prove
theorem orchard_yield_correct : orchardTotalYield = 7375 := 
by sorry

end orchard_yield_correct_l242_242530


namespace train_passes_jogger_in_37_seconds_l242_242002

noncomputable def jogger_speed_kmph : ℝ := 9
noncomputable def train_speed_kmph : ℝ := 45
noncomputable def jogger_lead_m : ℝ := 250
noncomputable def train_length_m : ℝ := 120

noncomputable def jogger_speed_mps : ℝ := jogger_speed_kmph * 1000 / 3600
noncomputable def train_speed_mps : ℝ := train_speed_kmph * 1000 / 3600
noncomputable def relative_speed_mps : ℝ := train_speed_mps - jogger_speed_mps
noncomputable def total_distance_m : ℝ := jogger_lead_m + train_length_m

theorem train_passes_jogger_in_37_seconds :
  total_distance_m / relative_speed_mps = 37 := by
  sorry

end train_passes_jogger_in_37_seconds_l242_242002


namespace num_divisors_fact8_l242_242692

-- Helper function to compute factorial
def factorial (n : ℕ) : ℕ :=
  if hn : n = 0 then 1 else n * factorial (n - 1)

-- Defining 8!
def fact8 := factorial 8

-- Prime factorization related definitions
def prime_factors_8! := (2 ^ 7) * (3 ^ 2) * (5 ^ 1) * (7 ^ 1)
def number_of_divisors (n : ℕ) := n.factors.length

-- Statement of the theorem
theorem num_divisors_fact8 : number_of_divisors fact8 = 96 := 
sorry

end num_divisors_fact8_l242_242692


namespace ratio_of_distances_l242_242322

theorem ratio_of_distances 
  (x : ℝ) -- distance walked by the first lady
  (h1 : 4 + x = 12) -- combined total distance walked is 12 miles 
  (h2 : ¬(x < 0)) -- distance cannot be negative
  (h3 : 4 ≠ 0) : -- the second lady walked 4 miles which is not zero
  x / 4 = 2 := -- the ratio of the distances is 2
by
  sorry

end ratio_of_distances_l242_242322


namespace laurie_shells_l242_242382

def alan_collected : ℕ := 48
def ben_collected (alan : ℕ) : ℕ := alan / 4
def laurie_collected (ben : ℕ) : ℕ := ben * 3

theorem laurie_shells (a : ℕ) (b : ℕ) (l : ℕ) (h1 : alan_collected = a)
  (h2 : ben_collected a = b) (h3 : laurie_collected b = l) : l = 36 := 
by
  sorry

end laurie_shells_l242_242382


namespace simplify_expression_and_evaluate_evaluate_expression_at_one_l242_242118

theorem simplify_expression_and_evaluate (x : ℝ)
  (h1 : x ≠ -2) (h2 : x ≠ 2) (h3 : x ≠ 3) :
  ( ((x^2 - 2*x) / (x^2 - 4*x + 4) - 3 / (x - 2)) / ((x - 3) / (x^2 - 4)) ) = x + 2 :=
by {
  sorry
}

theorem evaluate_expression_at_one :
  ( ((1^2 - 2*1) / (1^2 - 4*1 + 4) - 3 / (1 - 2)) / ((1 - 3) / (1^2 - 4)) ) = 3 :=
by {
  sorry
}

end simplify_expression_and_evaluate_evaluate_expression_at_one_l242_242118


namespace power_function_value_at_minus_two_l242_242329

-- Define the power function assumption and points
variable (f : ℝ → ℝ)
variable (hf : f (1 / 2) = 8)

-- Prove that the given condition implies the required result
theorem power_function_value_at_minus_two : f (-2) = -1 / 8 := 
by {
  -- proof to be filled here
  sorry
}

end power_function_value_at_minus_two_l242_242329


namespace proof_problem_l242_242306

variable (p1 p2 p3 p4 : Prop)

theorem proof_problem (hp1 : p1) (hp2 : ¬ p2) (hp3 : ¬ p3) (hp4 : p4) :
  (p1 ∧ p4) ∧ (¬ p2 ∨ p3) ∧ (¬ p3 ∨ ¬ p4) := by
  sorry

end proof_problem_l242_242306


namespace complete_the_square_l242_242372

theorem complete_the_square (y : ℝ) : (y^2 + 12*y + 40) = (y + 6)^2 + 4 := by
  sorry

end complete_the_square_l242_242372


namespace initial_books_in_library_l242_242295

theorem initial_books_in_library 
  (initial_books : ℕ)
  (books_taken_out_Tuesday : ℕ := 120)
  (books_returned_Wednesday : ℕ := 35)
  (books_withdrawn_Thursday : ℕ := 15)
  (books_final_count : ℕ := 150)
  : initial_books - books_taken_out_Tuesday + books_returned_Wednesday - books_withdrawn_Thursday = books_final_count → initial_books = 250 :=
by
  intros h
  sorry

end initial_books_in_library_l242_242295


namespace combine_syllables_to_computer_l242_242042

/-- Conditions provided in the problem -/
def first_syllable : String := "ком" -- A big piece of a snowman
def second_syllable : String := "пьют" -- Something done by elephants at a watering hole
def third_syllable : String := "ер" -- The old name of the hard sign

/-- The result obtained by combining the three syllables should be "компьютер" -/
theorem combine_syllables_to_computer :
  (first_syllable ++ second_syllable ++ third_syllable) = "компьютер" :=
by
  -- Proof to be provided
  sorry

end combine_syllables_to_computer_l242_242042


namespace age_ratio_five_years_later_l242_242715

theorem age_ratio_five_years_later (my_age : ℕ) (son_age : ℕ) (h1 : my_age = 45) (h2 : son_age = 15) :
  (my_age + 5) / gcd (my_age + 5) (son_age + 5) = 5 ∧ (son_age + 5) / gcd (my_age + 5) (son_age + 5) = 2 :=
by
  sorry

end age_ratio_five_years_later_l242_242715


namespace baseball_tickets_l242_242101

theorem baseball_tickets (B : ℕ) 
  (h1 : 25 = 2 * B + 6) : B = 9 :=
sorry

end baseball_tickets_l242_242101


namespace probability_three_defective_before_two_good_correct_l242_242658

noncomputable def probability_three_defective_before_two_good 
  (total_items : ℕ) 
  (good_items : ℕ) 
  (defective_items : ℕ) 
  (sequence_length : ℕ) : ℚ := 
  -- We will skip the proof part and just acknowledge the result as mentioned
  (1 / 55 : ℚ)

theorem probability_three_defective_before_two_good_correct :
  probability_three_defective_before_two_good 12 9 3 5 = 1 / 55 := 
by sorry

end probability_three_defective_before_two_good_correct_l242_242658


namespace optimality_theorem_l242_242201

def sequence_1 := "[[[a1, a2], a3], a4]" -- 22 symbols sequence
def sequence_2 := "[[a1, a2], [a3, a4]]" -- 16 symbols sequence

def optimal_sequence := sequence_2

theorem optimality_theorem : optimal_sequence = "[[a1, a2], [a3, a4]]" :=
by
  sorry

end optimality_theorem_l242_242201


namespace parabola_equation_and_orthogonality_l242_242336

theorem parabola_equation_and_orthogonality 
  (p : ℝ) (h_p_pos : p > 0) 
  (F : ℝ × ℝ) (h_focus : F = (p / 2, 0)) 
  (A B : ℝ × ℝ) (y : ℝ → ℝ) (C : ℝ × ℝ) 
  (h_parabola : ∀ (x y : ℝ), y^2 = 2 * p * x) 
  (h_line : ∀ (x : ℝ), y x = x - 8) 
  (h_intersect : ∃ x, y x = 0)
  (h_intersection_points : ∃ (x1 x2 : ℝ), y x1 = 0 ∧ y x2 = 0)
  (O : ℝ × ℝ) (h_origin : O = (0, 0)) 
  (h_vector_relation : 3 * F.fst = C.fst - F.fst)
  (h_C_x_axis : C = (8, 0)) :
  (p = 4 → y^2 = 8 * x) ∧ 
  (∀ (A B : ℝ × ℝ), (A.snd * B.snd = -64) ∧ 
  ((A.fst = (A.snd)^2 / 8) ∧ (B.fst = (B.snd)^2 / 8)) → 
  (A.fst * B.fst + A.snd * B.snd = 0)) := 
sorry

end parabola_equation_and_orthogonality_l242_242336


namespace value_of_x_l242_242022

theorem value_of_x (x y : ℝ) (h : x / (x - 1) = (y^2 + 2 * y + 3) / (y^2 + 2 * y + 2))  : 
  x = y^2 + 2 * y + 3 := 
by 
  sorry

end value_of_x_l242_242022


namespace total_games_friends_l242_242805

def new_friends_games : ℕ := 88
def old_friends_games : ℕ := 53

theorem total_games_friends :
  new_friends_games + old_friends_games = 141 :=
by
  sorry

end total_games_friends_l242_242805


namespace find_y_l242_242207

theorem find_y (x y : ℕ) (h1 : 24 * x = 173 * y) (h2 : 173 * y = 1730) : y = 10 :=
by 
  -- Proof is skipped
  sorry

end find_y_l242_242207


namespace tank_leak_time_l242_242355

/--
The rate at which the tank is filled without a leak is R = 1/5 tank per hour.
The effective rate with the leak is 1/6 tank per hour.
Prove that the time it takes for the leak to empty the full tank is 30 hours.
-/
theorem tank_leak_time (R : ℝ) (L : ℝ) (h1 : R = 1 / 5) (h2 : R - L = 1 / 6) :
  1 / L = 30 :=
by
  sorry

end tank_leak_time_l242_242355


namespace julian_notes_problem_l242_242038

theorem julian_notes_problem (x y : ℤ) (h1 : 3 * x + 4 * y = 151) (h2 : x = 19 ∨ y = 19) :
  x = 25 ∨ y = 25 := 
by
  sorry

end julian_notes_problem_l242_242038


namespace division_powers_5_half_division_powers_6_3_division_powers_formula_division_powers_combination_l242_242791

def f (n : ℕ) (a : ℚ) : ℚ := a ^ (2 - n)

theorem division_powers_5_half : f 5 (1/2) = 8 := by
  -- skip the proof
  sorry

theorem division_powers_6_3 : f 6 3 = 1/81 := by
  -- skip the proof
  sorry

theorem division_powers_formula (n : ℕ) (a : ℚ) (h : n > 0) : f n a = a^(2 - n) := by
  -- skip the proof
  sorry

theorem division_powers_combination : f 5 (1/3) * f 4 3 * f 5 (1/2) + f 5 (-1/4) / f 6 (-1/2) = 20 := by
  -- skip the proof
  sorry

end division_powers_5_half_division_powers_6_3_division_powers_formula_division_powers_combination_l242_242791


namespace range_of_m_l242_242177

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := 9 * x - m

theorem range_of_m (H : ∃ (x_0 : ℝ), x_0 ≠ 0 ∧ f 0 x_0 = f 0 x_0) : 0 < m ∧ m < 1 / 2 :=
sorry

end range_of_m_l242_242177


namespace chessboard_colorings_l242_242603

-- Definitions based on conditions
def valid_chessboard_colorings_count : ℕ :=
  2 ^ 33

-- Theorem statement with the question, conditions, and the correct answer
theorem chessboard_colorings : 
  valid_chessboard_colorings_count = 2 ^ 33 := by
  sorry

end chessboard_colorings_l242_242603


namespace solve_system_l242_242412

noncomputable def system_solutions (x y z : ℤ) : Prop :=
  x^3 + y^3 + z^3 = 8 ∧
  x^2 + y^2 + z^2 = 22 ∧
  (1 / x + 1 / y + 1 / z = - (z / (x * y)))

theorem solve_system :
  ∀ (x y z : ℤ), system_solutions x y z ↔ 
    (x = 3 ∧ y = 2 ∧ z = -3) ∨
    (x = -3 ∧ y = 2 ∧ z = 3) ∨
    (x = 2 ∧ y = 3 ∧ z = -3) ∨
    (x = 2 ∧ y = -3 ∧ z = 3) := by
  sorry

end solve_system_l242_242412


namespace linear_equation_solution_l242_242626

theorem linear_equation_solution (m : ℝ) (x : ℝ) (h : |m| - 2 = 1) (h_ne : m ≠ 3) :
  (2 * m - 6) * x^(|m|-2) = m^2 ↔ x = -(3/4) :=
by
  sorry

end linear_equation_solution_l242_242626


namespace fernanda_total_time_to_finish_l242_242086

noncomputable def fernanda_days_to_finish_audiobooks
  (num_audiobooks : ℕ) (hours_per_audiobook : ℕ) (hours_listened_per_day : ℕ) : ℕ :=
num_audiobooks * hours_per_audiobook / hours_listened_per_day

-- Definitions based on the conditions
def num_audiobooks : ℕ := 6
def hours_per_audiobook : ℕ := 30
def hours_listened_per_day : ℕ := 2

-- Statement to prove
theorem fernanda_total_time_to_finish :
  fernanda_days_to_finish_audiobooks num_audiobooks hours_per_audiobook hours_listened_per_day = 90 := 
sorry

end fernanda_total_time_to_finish_l242_242086


namespace rational_inequality_solution_l242_242577

theorem rational_inequality_solution (x : ℝ) (h : x ≠ 4) :
  (4 < x ∧ x ≤ 5) ↔ (x - 2) / (x - 4) ≤ 3 :=
sorry

end rational_inequality_solution_l242_242577


namespace xiangming_payment_methods_count_l242_242428

def xiangming_payment_methods : Prop :=
  ∃ x y z : ℕ, 
    x + y + z ≤ 10 ∧ 
    x + 2 * y + 5 * z = 18 ∧ 
    ((x > 0 ∧ y > 0) ∨ (x > 0 ∧ z > 0) ∨ (y > 0 ∧ z > 0))

theorem xiangming_payment_methods_count : 
  xiangming_payment_methods → ∃! n, n = 11 :=
by sorry

end xiangming_payment_methods_count_l242_242428


namespace part1_solution_set_part2_range_a_l242_242173

def f (x a : ℝ) : ℝ := |x - a^2| + |x - 2*a + 1|

theorem part1_solution_set (x : ℝ) (h : f x 2 ≥ 4) : 
  x ≤ 3/2 ∨ x ≥ 11/2 :=
sorry

theorem part2_range_a (a : ℝ) (h : ∃ x, f x a ≥ 4) : 
  a ≤ -1 ∨ a ≥ 3 :=
sorry

end part1_solution_set_part2_range_a_l242_242173


namespace combined_mpg_l242_242532

theorem combined_mpg (ray_mpg tom_mpg ray_miles tom_miles : ℕ) 
  (h1 : ray_mpg = 50) (h2 : tom_mpg = 8) 
  (h3 : ray_miles = 100) (h4 : tom_miles = 200) : 
  (ray_miles + tom_miles) / ((ray_miles / ray_mpg) + (tom_miles / tom_mpg)) = 100 / 9 :=
by
  sorry

end combined_mpg_l242_242532


namespace find_f_2010_l242_242742

noncomputable def f (x : ℝ) : ℝ := sorry

axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom functional_eq : ∀ x : ℝ, f (x + 6) = f x + f (3 - x)

theorem find_f_2010 : f 2010 = 0 := sorry

end find_f_2010_l242_242742


namespace marsha_first_package_miles_l242_242721

noncomputable def total_distance (x : ℝ) : ℝ := x + 28 + 14

noncomputable def earnings (x : ℝ) : ℝ := total_distance x * 2

theorem marsha_first_package_miles : ∃ x : ℝ, earnings x = 104 ∧ x = 10 :=
by
  use 10
  sorry

end marsha_first_package_miles_l242_242721


namespace broken_seashells_count_l242_242823

def total_seashells : Nat := 6
def unbroken_seashells : Nat := 2
def broken_seashells : Nat := total_seashells - unbroken_seashells

theorem broken_seashells_count :
  broken_seashells = 4 :=
by
  -- The proof would go here, but for now, we use 'sorry' to denote it.
  sorry

end broken_seashells_count_l242_242823


namespace find_prime_pairs_l242_242325

def is_prime (n : ℕ) : Prop := Nat.Prime n

def valid_pair (p q : ℕ) : Prop := 
  p < 2023 ∧ q < 2023 ∧ 
  p ∣ q^2 + 8 ∧ q ∣ p^2 + 8

theorem find_prime_pairs : 
  ∀ (p q : ℕ), is_prime p → is_prime q → valid_pair p q → 
    (p = 2 ∧ q = 2) ∨ 
    (p = 17 ∧ q = 3) ∨ 
    (p = 11 ∧ q = 5) :=
by 
  sorry

end find_prime_pairs_l242_242325


namespace real_roots_of_quadratic_l242_242089

theorem real_roots_of_quadratic (m : ℝ) :
  (∃ x : ℝ, m * x^2 - 4 * x + 3 = 0) ↔ m ≤ 4 / 3 :=
by
  sorry

end real_roots_of_quadratic_l242_242089


namespace exists_pos_ints_l242_242375

open Nat

noncomputable def f (a : ℕ) : ℕ :=
  a^2 + 3 * a + 2

noncomputable def g (b c : ℕ) : ℕ :=
  b^2 - b + 3 * c^2 + 3 * c

theorem exists_pos_ints (a : ℕ) (ha : 0 < a) :
  ∃ (b c : ℕ), 0 < b ∧ 0 < c ∧ f a = g b c :=
sorry

end exists_pos_ints_l242_242375


namespace range_of_m_l242_242004

-- Definitions based on the given conditions
def setA : Set ℝ := {x | -3 ≤ x ∧ x ≤ 4}
def setB (m : ℝ) : Set ℝ := {x | 2 * m - 1 < x ∧ x < m + 1}

-- Lean statement of the problem
theorem range_of_m (m : ℝ) (h : setB m ⊆ setA) : m ≥ -1 :=
sorry  -- proof is not required

end range_of_m_l242_242004


namespace neg_p_sufficient_for_neg_q_l242_242507

def p (x : ℝ) : Prop := |2 * x - 3| > 1
def q (x : ℝ) : Prop := x^2 + x - 6 > 0

theorem neg_p_sufficient_for_neg_q :
  (∀ x, ¬ p x → ¬ q x) ∧ ¬ (∀ x, ¬ q x → ¬ p x) :=
by
  -- Placeholder to indicate skipping the proof
  sorry

end neg_p_sufficient_for_neg_q_l242_242507


namespace arithmetic_sequence_30th_term_l242_242056

-- Definitions
def a₁ : ℤ := 8
def d : ℤ := -3
def n : ℕ := 30

-- The statement to be proved
theorem arithmetic_sequence_30th_term :
  a₁ + (n - 1) * d = -79 :=
by
  sorry

end arithmetic_sequence_30th_term_l242_242056


namespace person_speed_l242_242060

namespace EscalatorProblem

/-- The speed of the person v_p walking on the moving escalator is 3 ft/sec given the conditions -/
theorem person_speed (v_p : ℝ) 
  (escalator_speed : ℝ := 12) 
  (escalator_length : ℝ := 150) 
  (time_taken : ℝ := 10) :
  escalator_length = (v_p + escalator_speed) * time_taken → v_p = 3 := 
by sorry

end EscalatorProblem

end person_speed_l242_242060


namespace factorization_correct_l242_242662

theorem factorization_correct : ∀ x : ℝ, (x^2 - 2*x - 9 = 0) → ((x-1)^2 = 10) :=
by 
  intros x h
  sorry

end factorization_correct_l242_242662


namespace not_possible_2018_people_in_2019_minutes_l242_242671

-- Definitions based on conditions
def initial_people (t : ℕ) : ℕ := 0
def changed_people (x y : ℕ) : ℕ := 2 * x - y

theorem not_possible_2018_people_in_2019_minutes :
  ¬ ∃ (x y : ℕ), (x + y = 2019) ∧ (2 * x - y = 2018) :=
by
  sorry

end not_possible_2018_people_in_2019_minutes_l242_242671


namespace ratio_of_areas_l242_242358

-- Definition of sides and given condition
variables {a b c d : ℝ}
-- Given condition in the problem.
axiom condition : a / c = 3 / 5 ∧ b / d = 3 / 5

-- Statement of the theorem to be proved in Lean 4
theorem ratio_of_areas (h : a / c = 3 / 5) (h' : b / d = 3 / 5) : (a * b) / (c * d) = 9 / 25 :=
by sorry

end ratio_of_areas_l242_242358


namespace count_valid_three_digit_numbers_l242_242057

theorem count_valid_three_digit_numbers : 
  let is_valid (a b c : ℕ) := 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ b = (a + c) / 2 ∧ (a + c) % 2 = 0
  ∃ n : ℕ, (∀ a b c : ℕ, is_valid a b c → n = 45) :=
sorry

end count_valid_three_digit_numbers_l242_242057


namespace find_root_floor_l242_242723

noncomputable def g (x : ℝ) := Real.sin x - Real.cos x + 4 * Real.tan x

theorem find_root_floor :
  ∃ s : ℝ, (g s = 0) ∧ (π / 2 < s) ∧ (s < 3 * π / 2) ∧ (Int.floor s = 3) :=
  sorry

end find_root_floor_l242_242723


namespace forest_enclosure_l242_242043

theorem forest_enclosure
  (n : ℕ)
  (a : Fin n → ℝ)
  (h_a_lt_100 : ∀ i, a i < 100)
  (d : Fin n → Fin n → ℝ)
  (h_dist : ∀ i j, i < j → d i j ≤ (a i) - (a j)) :
  ∃ f : ℝ, f = 200 :=
by
  -- The proof goes here
  sorry

end forest_enclosure_l242_242043


namespace john_pays_more_than_jane_l242_242632

noncomputable def original_price : ℝ := 34.00
noncomputable def discount : ℝ := 0.10
noncomputable def tip_percent : ℝ := 0.15

noncomputable def discounted_price : ℝ := original_price - (discount * original_price)
noncomputable def john_tip : ℝ := tip_percent * original_price
noncomputable def john_total : ℝ := discounted_price + john_tip
noncomputable def jane_tip : ℝ := tip_percent * discounted_price
noncomputable def jane_total : ℝ := discounted_price + jane_tip

theorem john_pays_more_than_jane : john_total - jane_total = 0.51 := by
  sorry

end john_pays_more_than_jane_l242_242632


namespace integer_solutions_to_equation_l242_242307

theorem integer_solutions_to_equation :
  {p : ℤ × ℤ | (p.fst^2 - 2 * p.fst * p.snd - 3 * p.snd^2 = 5)} =
  {(4, 1), (2, -1), (-4, -1), (-2, 1)} :=
by {
  sorry
}

end integer_solutions_to_equation_l242_242307


namespace parcel_cost_guangzhou_shanghai_l242_242102

theorem parcel_cost_guangzhou_shanghai (x y : ℕ) :
  (x + 2 * y = 10 ∧ x + 3 * (y + 3) + 2 = 23) →
  (x = 6 ∧ y = 2 ∧ (6 + 4 * 2 = 14)) := by
  sorry

end parcel_cost_guangzhou_shanghai_l242_242102


namespace total_fish_caught_l242_242471

-- Definitions based on conditions
def brenden_morning_fish := 8
def brenden_fish_thrown_back := 3
def brenden_afternoon_fish := 5
def dad_fish := 13

-- Theorem representing the main question and its answer
theorem total_fish_caught : 
  (brenden_morning_fish + brenden_afternoon_fish - brenden_fish_thrown_back) + dad_fish = 23 :=
by
  sorry -- Proof goes here

end total_fish_caught_l242_242471


namespace topsoil_cost_is_112_l242_242217

noncomputable def calculate_topsoil_cost (length width depth_in_inches : ℝ) (cost_per_cubic_foot : ℝ) : ℝ :=
  let depth_in_feet := depth_in_inches / 12
  let volume := length * width * depth_in_feet
  volume * cost_per_cubic_foot

theorem topsoil_cost_is_112 :
  calculate_topsoil_cost 8 4 6 7 = 112 :=
by
  sorry

end topsoil_cost_is_112_l242_242217


namespace seq_geom_prog_l242_242162

theorem seq_geom_prog (a : ℕ → ℝ) (b : ℝ) (h_pos_b : 0 < b)
  (h_pos_a : ∀ n, 0 < a n)
  (h_recurrence : ∀ n, a (n + 2) = (b + 1) * a n * a (n + 1)) :
  (∃ r, ∀ n, a (n + 1) = r * a n) ↔ a 0 = a 1 :=
sorry

end seq_geom_prog_l242_242162


namespace multiply_and_simplify_fractions_l242_242458

theorem multiply_and_simplify_fractions :
  (2 / 3) * (4 / 7) * (9 / 11) = 24 / 77 := 
by
  sorry

end multiply_and_simplify_fractions_l242_242458


namespace hexagon_angles_l242_242515

theorem hexagon_angles (a e : ℝ) (h1 : a = e - 60) (h2 : 4 * a + 2 * e = 720) :
  e = 160 :=
by
  sorry

end hexagon_angles_l242_242515


namespace max_apples_discarded_l242_242093

theorem max_apples_discarded (n : ℕ) : n % 7 ≤ 6 := by
  sorry

end max_apples_discarded_l242_242093


namespace infinitely_many_primes_l242_242540

theorem infinitely_many_primes : ∀ (p : ℕ) (h_prime : Nat.Prime p), ∃ (q : ℕ), Nat.Prime q ∧ q > p :=
by
  sorry

end infinitely_many_primes_l242_242540


namespace rainfall_difference_correct_l242_242479

def rainfall_difference (monday_rain : ℝ) (tuesday_rain : ℝ) : ℝ :=
  monday_rain - tuesday_rain

theorem rainfall_difference_correct : rainfall_difference 0.9 0.2 = 0.7 :=
by
  simp [rainfall_difference]
  sorry

end rainfall_difference_correct_l242_242479


namespace find_number_l242_242567

theorem find_number (x : ℝ) (h : 160 = 3.2 * x) : x = 50 :=
by 
  sorry

end find_number_l242_242567


namespace percentage_sales_tax_on_taxable_purchases_l242_242804

-- Definitions
def total_cost : ℝ := 30
def tax_free_cost : ℝ := 24.7
def tax_rate : ℝ := 0.06

-- Statement to prove
theorem percentage_sales_tax_on_taxable_purchases :
  (tax_rate * (total_cost - tax_free_cost)) / total_cost * 100 = 1 := by
  sorry

end percentage_sales_tax_on_taxable_purchases_l242_242804


namespace percentage_increase_in_expenses_l242_242328

theorem percentage_increase_in_expenses:
  ∀ (S : ℝ) (original_save_percentage new_savings : ℝ), 
  S = 5750 → 
  original_save_percentage = 0.20 →
  new_savings = 230 →
  (original_save_percentage * S - new_savings) / (S - original_save_percentage * S) * 100 = 20 :=
by
  intros S original_save_percentage new_savings HS Horiginal_save_percentage Hnew_savings
  rw [HS, Horiginal_save_percentage, Hnew_savings]
  sorry

end percentage_increase_in_expenses_l242_242328


namespace house_cost_l242_242666

-- Definitions of given conditions
def annual_salary : ℝ := 150000
def saving_rate : ℝ := 0.10
def downpayment_rate : ℝ := 0.20
def years_saving : ℝ := 6

-- Given the conditions, calculate annual savings and total savings after 6 years
def annual_savings : ℝ := annual_salary * saving_rate
def total_savings : ℝ := annual_savings * years_saving

-- Total savings represents 20% of the house cost
def downpayment : ℝ := total_savings

-- Prove the total cost of the house
theorem house_cost (downpayment : ℝ) (downpayment_rate : ℝ) : ℝ :=
  downpayment / downpayment_rate

lemma house_cost_correct : house_cost downpayment downpayment_rate = 450000 :=
by
  -- the proof would go here
  sorry

end house_cost_l242_242666


namespace units_digit_n_l242_242409

theorem units_digit_n (m n : ℕ) (h1 : m * n = 31^8) (h2 : m % 10 = 7) : n % 10 = 3 := 
sorry

end units_digit_n_l242_242409


namespace evaluate_fraction_l242_242106

theorem evaluate_fraction : (35 / 0.07) = 500 := 
by
  sorry

end evaluate_fraction_l242_242106


namespace P_cubed_plus_7_is_composite_l242_242562

theorem P_cubed_plus_7_is_composite (P : ℕ) (h_prime_P : Nat.Prime P) (h_prime_P3_plus_5 : Nat.Prime (P^3 + 5)) : ¬ Nat.Prime (P^3 + 7) ∧ (P^3 + 7).factors.length > 1 :=
by
  sorry

end P_cubed_plus_7_is_composite_l242_242562


namespace simplify_expression_l242_242548

variable (a : ℝ)

theorem simplify_expression : 2 * a * (2 * a ^ 2 + a) - a ^ 2 = 4 * a ^ 3 + a ^ 2 := 
  sorry

end simplify_expression_l242_242548


namespace gcd_factorials_l242_242712

noncomputable def factorial : ℕ → ℕ
| 0     => 1
| (n+1) => (n+1) * factorial n

theorem gcd_factorials (n m : ℕ) (hn : n = 8) (hm : m = 10) :
  Nat.gcd (factorial n) (factorial m) = 40320 := by
  sorry

end gcd_factorials_l242_242712


namespace production_movie_count_l242_242558

theorem production_movie_count
  (LJ_annual : ℕ)
  (H1 : LJ_annual = 220)
  (H2 : ∀ n, n = 275 → n = LJ_annual + (LJ_annual * 25 / 100))
  (years : ℕ)
  (H3 : years = 5) :
  (LJ_annual + 275) * years = 2475 :=
by {
  sorry
}

end production_movie_count_l242_242558


namespace graphs_intersect_at_one_point_l242_242300

noncomputable def f (x : ℝ) : ℝ := 3 * Real.log x / Real.log 3
noncomputable def g (x : ℝ) : ℝ := Real.log (4 * x) / Real.log 2

theorem graphs_intersect_at_one_point : ∃! x, f x = g x :=
by {
  sorry
}

end graphs_intersect_at_one_point_l242_242300


namespace cards_per_layer_l242_242171

theorem cards_per_layer (total_decks : ℕ) (cards_per_deck : ℕ) (layers : ℕ) (h_decks : total_decks = 16) (h_cards_per_deck : cards_per_deck = 52) (h_layers : layers = 32) :
  total_decks * cards_per_deck / layers = 26 :=
by {
  -- To skip the proof
  sorry
}

end cards_per_layer_l242_242171


namespace sandcastle_ratio_l242_242005

-- Definitions based on conditions in a)
def sandcastles_on_marks_beach : ℕ := 20
def towers_per_sandcastle_marks_beach : ℕ := 10
def towers_per_sandcastle_jeffs_beach : ℕ := 5
def total_combined_sandcastles_and_towers : ℕ := 580

-- The main statement to prove
theorem sandcastle_ratio : 
  ∃ (J : ℕ), 
  (sandcastles_on_marks_beach + (towers_per_sandcastle_marks_beach * sandcastles_on_marks_beach) + J + (towers_per_sandcastle_jeffs_beach * J) = total_combined_sandcastles_and_towers) ∧ 
  (J / sandcastles_on_marks_beach = 3) :=
by 
  sorry

end sandcastle_ratio_l242_242005


namespace find_ac_bc_val_l242_242187

variable (a b c d : ℚ)
variable (h_neq : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
variable (h1 : (a + c) * (a + d) = 1)
variable (h2 : (b + c) * (b + d) = 1)

theorem find_ac_bc_val : (a + c) * (b + c) = -1 := 
by 
  sorry

end find_ac_bc_val_l242_242187


namespace average_temperature_correct_l242_242778

-- Definition of the daily temperatures
def daily_temperatures : List ℕ := [51, 64, 61, 59, 48, 63, 55]

-- Define the number of days
def number_of_days : ℕ := 7

-- Prove the average temperature calculation
theorem average_temperature_correct :
  ((List.sum daily_temperatures : ℚ) / number_of_days : ℚ) = 57.3 :=
by
  sorry

end average_temperature_correct_l242_242778


namespace area_excluding_hole_l242_242815

def area_large_rectangle (x : ℝ) : ℝ :=
  (2 * x + 9) * (x + 6)

def area_square_hole (x : ℝ) : ℝ :=
  (x - 1) * (x - 1)

theorem area_excluding_hole (x : ℝ) : 
  area_large_rectangle x - area_square_hole x = x^2 + 23 * x + 53 :=
by
  sorry

end area_excluding_hole_l242_242815


namespace arithmetic_sequence_nine_l242_242281

variable (a : ℕ → ℝ)
variable (d : ℝ)
-- Define an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) := ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_nine (a : ℕ → ℝ) (d : ℝ) 
  (h_arith : arithmetic_sequence a d)
  (h_cond : a 4 + a 14 = 2) : 
  a 9 = 1 := 
sorry

end arithmetic_sequence_nine_l242_242281


namespace smallest_k_multiple_of_180_l242_242185

def sum_of_squares (k : ℕ) : ℕ :=
  (k * (k + 1) * (2 * k + 1)) / 6

def divisible_by_180 (n : ℕ) : Prop :=
  n % 180 = 0

theorem smallest_k_multiple_of_180 :
  ∃ k : ℕ, k > 0 ∧ divisible_by_180 (sum_of_squares k) ∧ ∀ m : ℕ, m > 0 ∧ divisible_by_180 (sum_of_squares m) → k ≤ m :=
sorry

end smallest_k_multiple_of_180_l242_242185


namespace team_A_processes_fraction_l242_242444

theorem team_A_processes_fraction (A B : ℕ) (total_calls : ℚ) 
  (h1 : A = (5/8) * B) 
  (h2 : (8 / 11) * total_calls = TeamB_calls_processed)
  (frac_TeamA_calls : ℚ := (1 - (8 / 11)) * total_calls)
  (calls_per_member_A : ℚ := frac_TeamA_calls / A)
  (calls_per_member_B : ℚ := (8 / 11) * total_calls / B) : 
  calls_per_member_A / calls_per_member_B = 3 / 5 := 
by
  sorry

end team_A_processes_fraction_l242_242444


namespace number_of_terms_in_product_l242_242230

theorem number_of_terms_in_product 
  (a b c d e f g h i : ℕ) :
  (a + b + c + d) * (e + f + g + h + i) = 20 :=
sorry

end number_of_terms_in_product_l242_242230


namespace intersection_S_T_eq_T_l242_242477

-- Define the sets S and T
def S : Set ℤ := { s : ℤ | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t : ℤ | ∃ n : ℤ, t = 4 * n + 1 }

-- State the theorem
theorem intersection_S_T_eq_T : S ∩ T = T :=
sorry

end intersection_S_T_eq_T_l242_242477


namespace intersection_point_of_lines_l242_242246

theorem intersection_point_of_lines : 
  ∃ (x y : ℝ), (x - 4 * y - 1 = 0) ∧ (2 * x + y - 2 = 0) ∧ (x = 1) ∧ (y = 0) :=
by
  sorry

end intersection_point_of_lines_l242_242246


namespace total_rainfall_l242_242830

theorem total_rainfall
  (r₁ r₂ : ℕ)
  (T t₁ : ℕ)
  (H1 : r₁ = 30)
  (H2 : r₂ = 15)
  (H3 : T = 45)
  (H4 : t₁ = 20) :
  r₁ * t₁ + r₂ * (T - t₁) = 975 := by
  sorry

end total_rainfall_l242_242830


namespace crocodile_length_in_meters_l242_242305

-- Definitions based on conditions
def ken_to_cm : ℕ := 180
def shaku_to_cm : ℕ := 30
def ken_to_shaku : ℕ := 6
def cm_to_m : ℕ := 100

-- Lengths given in the problem expressed in ken
def head_to_tail_in_ken (L : ℚ) : Prop := 3 * L = 10
def tail_to_head_in_ken (L : ℚ) : Prop := L = (3 + (2 / ken_to_shaku : ℚ))

-- Final length conversion to meters
def length_in_m (L : ℚ) : ℚ := L * ken_to_cm / cm_to_m

-- The length of the crocodile in meters
theorem crocodile_length_in_meters (L : ℚ) : head_to_tail_in_ken L → tail_to_head_in_ken L → length_in_m L = 6 :=
by
  intros _ _
  sorry

end crocodile_length_in_meters_l242_242305


namespace fill_pool_time_l242_242798

theorem fill_pool_time (R : ℝ) (T : ℝ) (hSlowerPipe : R = 1 / 9) (hFasterPipe : 1.25 * R = 1.25 / 9)
                     (hCombinedRate : 2.25 * R = 2.25 / 9) : T = 4 := by
  sorry

end fill_pool_time_l242_242798


namespace sum_of_fractions_limit_one_l242_242365

theorem sum_of_fractions_limit_one :
  (∑' (a : ℕ), ∑' (b : ℕ), (1 : ℝ) / ((a + 1) : ℝ) ^ (b + 1)) = 1 := 
sorry

end sum_of_fractions_limit_one_l242_242365


namespace smallest_term_of_bn_div_an_is_four_l242_242683

theorem smallest_term_of_bn_div_an_is_four
  (a : ℕ → ℚ)
  (b : ℕ → ℚ)
  (S : ℕ → ℚ)
  (h1 : a 1 = 1)
  (h2 : ∀ n, a n * a (n + 1) = 2 * S n)
  (h3 : b 1 = 16)
  (h4 : ∀ n, b (n + 1) - b n = 2 * n) :
  ∃ n : ℕ, ∀ m : ℕ, (m ≠ 4 → b m / a m > b 4 / a 4) ∧ (n = 4) := sorry

end smallest_term_of_bn_div_an_is_four_l242_242683


namespace total_chickens_l242_242288

   def number_of_hens := 12
   def hens_to_roosters_ratio := 3
   def chicks_per_hen := 5

   theorem total_chickens (h : number_of_hens = 12)
                          (r : hens_to_roosters_ratio = 3)
                          (c : chicks_per_hen = 5) :
     number_of_hens + (number_of_hens / hens_to_roosters_ratio) + (number_of_hens * chicks_per_hen) = 76 :=
   by
     sorry
   
end total_chickens_l242_242288


namespace smaller_package_contains_correct_number_of_cupcakes_l242_242835

-- Define the conditions
def number_of_packs_large : ℕ := 4
def cupcakes_per_large_pack : ℕ := 15
def total_children : ℕ := 100
def needed_packs_small : ℕ := 4

-- Define the total cupcakes bought initially
def total_cupcakes_bought : ℕ := number_of_packs_large * cupcakes_per_large_pack

-- Define the total additional cupcakes needed
def additional_cupcakes_needed : ℕ := total_children - total_cupcakes_bought

-- Define the number of cupcakes per smaller package
def cupcakes_per_small_pack : ℕ := additional_cupcakes_needed / needed_packs_small

-- The theorem statement to prove
theorem smaller_package_contains_correct_number_of_cupcakes :
  cupcakes_per_small_pack = 10 :=
by
  -- This is where the proof would go
  sorry

end smaller_package_contains_correct_number_of_cupcakes_l242_242835


namespace train_length_l242_242620

theorem train_length (speed_kmph : ℝ) (time_sec : ℝ) (length_train : ℝ) 
  (h_speed : speed_kmph = 50)
  (h_time : time_sec = 18) 
  (h_length : length_train = 250) : 
  (speed_kmph * 1000 / 3600) * time_sec = length_train :=
by 
  rw [h_speed, h_time, h_length]
  sorry

end train_length_l242_242620


namespace fraction_of_sy_not_declared_major_l242_242055

-- Conditions
variables (T : ℝ) -- Total number of students
variables (first_year : ℝ) -- Fraction of first-year students
variables (second_year : ℝ) -- Fraction of second-year students
variables (decl_fy_major : ℝ) -- Fraction of first-year students who have declared a major
variables (decl_sy_major : ℝ) -- Fraction of second-year students who have declared a major

-- Definitions from conditions
def fraction_first_year_students := 1 / 2
def fraction_second_year_students := 1 / 2
def fraction_fy_declared_major := 1 / 5
def fraction_sy_declared_major := 4 * fraction_fy_declared_major

-- Hollow statement
theorem fraction_of_sy_not_declared_major :
  first_year = fraction_first_year_students →
  second_year = fraction_second_year_students →
  decl_fy_major = fraction_fy_declared_major →
  decl_sy_major = fraction_sy_declared_major →
  (1 - decl_sy_major) * second_year = 1 / 10 :=
by
  sorry

end fraction_of_sy_not_declared_major_l242_242055


namespace reduced_price_tickets_first_week_l242_242508

theorem reduced_price_tickets_first_week (total_tickets sold_at_full_price : ℕ) 
  (condition1 : total_tickets = 25200) 
  (condition2 : sold_at_full_price = 16500)
  (condition3 : ∃ R, total_tickets = R + 5 * R) : 
  ∃ R : ℕ, R = 3300 := 
by sorry

end reduced_price_tickets_first_week_l242_242508


namespace total_tickets_sold_l242_242486

theorem total_tickets_sold (n : ℕ) 
  (h1 : n * n = 1681) : 
  2 * n = 82 :=
by
  sorry

end total_tickets_sold_l242_242486


namespace expression_is_perfect_cube_l242_242764

theorem expression_is_perfect_cube {x y z : ℝ} (h : x + y + z = 0) :
  ∃ m : ℝ, 
    (x^2 * y^2 + y^2 * z^2 + z^2 * x^2) * 
    (x^3 * y * z + x * y^3 * z + x * y * z^3) *
    (x^3 * y^2 * z + x^3 * y * z^2 + x^2 * y^3 * z + x * y^3 * z^2 + x^2 * y * z^3 + x * y^2 * z^3) =
    m ^ 3 := 
by 
  sorry

end expression_is_perfect_cube_l242_242764


namespace find_g_at_7_l242_242046

noncomputable def g (x : ℝ) (a b c : ℝ) : ℝ := a * x^7 - b * x^3 + c * x - 4

theorem find_g_at_7 (a b c : ℝ) (h_symm : ∀ x : ℝ, g x a b c + g (-x) a b c = -8) (h_neg7: g (-7) a b c = 12) :
  g 7 a b c = -20 :=
by
  sorry

end find_g_at_7_l242_242046


namespace equation_of_curve_C_range_of_m_l242_242324

theorem equation_of_curve_C (x y m : ℝ) (hx : x ≠ 0) (hm : m > 1) (k1 k2 : ℝ) 
  (h_k1 : k1 = (y - 1) / x) (h_k2 : k2 = (y + 1) / (2 * x))
  (h_prod : k1 * k2 = -1 / m^2) :
  (x^2) / (m^2) + (y^2) = 1 := 
sorry

theorem range_of_m (m : ℝ) :
  (1 < m ∧ m ≤ Real.sqrt 3)
  ∨ (m < 1 ∨ m > Real.sqrt 3) :=
sorry

end equation_of_curve_C_range_of_m_l242_242324


namespace initial_roses_l242_242250

theorem initial_roses (x : ℕ) (h : x - 2 + 32 = 41) : x = 11 :=
sorry

end initial_roses_l242_242250


namespace ratio_of_crates_l242_242390

/-
  Gabrielle sells eggs. On Monday she sells 5 crates of eggs. On Tuesday she sells 2 times as many
  crates of eggs as Monday. On Wednesday she sells 2 fewer crates than Tuesday. On Thursday she sells
  some crates of eggs. She sells a total of 28 crates of eggs for the 4 days. Prove the ratio of the 
  number of crates she sells on Thursday to the number she sells on Tuesday is 1/2.
-/

theorem ratio_of_crates 
    (mon_crates : ℕ) 
    (tue_crates : ℕ) 
    (wed_crates : ℕ) 
    (thu_crates : ℕ) 
    (total_crates : ℕ) 
    (h_mon : mon_crates = 5) 
    (h_tue : tue_crates = 2 * mon_crates) 
    (h_wed : wed_crates = tue_crates - 2) 
    (h_total : total_crates = mon_crates + tue_crates + wed_crates + thu_crates) 
    (h_total_val : total_crates = 28): 
  (thu_crates / tue_crates : ℚ) = 1 / 2 := 
by 
  sorry

end ratio_of_crates_l242_242390


namespace num_positive_integers_n_l242_242245

theorem num_positive_integers_n (n : ℕ) : 
  (∃ n, ( ∃ k : ℕ, n = 2015 * k^2 ∧ ∃ m, m^2 = 2015 * n) ∧ 
          (∃ k : ℕ, n = 2015 * k^2 ∧  ∃ l : ℕ, 2 * 2015 * k^2 = l * (1 + k^2)))
  →
  n = 5 := sorry

end num_positive_integers_n_l242_242245


namespace string_length_l242_242040

def cylindrical_post_circumference : ℝ := 6
def cylindrical_post_height : ℝ := 15
def loops : ℝ := 3

theorem string_length :
  (cylindrical_post_height / loops)^2 + cylindrical_post_circumference^2 = 61 → 
  loops * Real.sqrt 61 = 3 * Real.sqrt 61 :=
by
  sorry

end string_length_l242_242040


namespace lcm_of_8_and_15_l242_242656

theorem lcm_of_8_and_15 : Nat.lcm 8 15 = 120 :=
by
  sorry

end lcm_of_8_and_15_l242_242656


namespace inequality_solution_set_l242_242268

theorem inequality_solution_set (x : ℝ) : 
  ( (x - 1) / (x + 2) > 0 ) ↔ ( x > 1 ∨ x < -2 ) :=
by sorry

end inequality_solution_set_l242_242268


namespace real_roots_of_f_l242_242021

noncomputable def f (x : ℝ) : ℝ := x^4 - 3 * x^3 + 3 * x^2 - x - 6

theorem real_roots_of_f :
  {x | f x = 0} = {-1, 1, 2, 3} :=
sorry

end real_roots_of_f_l242_242021


namespace weightlifter_total_weight_l242_242774

theorem weightlifter_total_weight (weight_one_hand : ℕ) (num_hands : ℕ) (condition: weight_one_hand = 8 ∧ num_hands = 2) :
  2 * weight_one_hand = 16 :=
by
  sorry

end weightlifter_total_weight_l242_242774


namespace theon_speed_l242_242289

theorem theon_speed (VTheon VYara D : ℕ) (h1 : VYara = 30) (h2 : D = 90) (h3 : D / VTheon = D / VYara + 3) : VTheon = 15 := by
  sorry

end theon_speed_l242_242289


namespace bowling_average_decrease_l242_242269

/-- Represents data about the bowler's performance. -/
structure BowlerPerformance :=
(old_average : ℚ)
(last_match_runs : ℚ)
(last_match_wickets : ℕ)
(previous_wickets : ℕ)

/-- Calculates the new total runs given. -/
def new_total_runs (perf : BowlerPerformance) : ℚ :=
  perf.old_average * ↑perf.previous_wickets + perf.last_match_runs

/-- Calculates the new total number of wickets. -/
def new_total_wickets (perf : BowlerPerformance) : ℕ :=
  perf.previous_wickets + perf.last_match_wickets

/-- Calculates the new bowling average. -/
def new_average (perf : BowlerPerformance) : ℚ :=
  new_total_runs perf / ↑(new_total_wickets perf)

/-- Calculates the decrease in the bowling average. -/
def decrease_in_average (perf : BowlerPerformance) : ℚ :=
  perf.old_average - new_average perf

/-- The proof statement to be verified. -/
theorem bowling_average_decrease :
  ∀ (perf : BowlerPerformance),
    perf.old_average = 12.4 →
    perf.last_match_runs = 26 →
    perf.last_match_wickets = 6 →
    perf.previous_wickets = 115 →
    decrease_in_average perf = 0.4 :=
by
  intros
  sorry

end bowling_average_decrease_l242_242269


namespace geometric_sequence_a6_l242_242796

noncomputable def a_sequence (n : ℕ) : ℝ := 1 * 2^(n-1)

theorem geometric_sequence_a6 (S : ℕ → ℝ)
  (h1 : S 10 = 3 * S 5)
  (h2 : ∀ n, S n = (1 - 2^n) / (1 - 2))
  (h3 : a_sequence 1 = 1) :
  a_sequence 6 = 2 := by
  sorry

end geometric_sequence_a6_l242_242796


namespace quadratic_variation_y_l242_242604

theorem quadratic_variation_y (k : ℝ) (x y : ℝ) (h1 : y = k * x^2) (h2 : (25 : ℝ) = k * (5 : ℝ)^2) :
  y = 25 :=
by
sorry

end quadratic_variation_y_l242_242604


namespace math_proof_l242_242122

def exponentiation_result := -1 ^ 4
def negative_exponentiation_result := (-2) ^ 3
def absolute_value_result := abs (-3 - 1)
def division_result := 16 / negative_exponentiation_result
def multiplication_result := division_result * absolute_value_result
def final_result := exponentiation_result + multiplication_result

theorem math_proof : final_result = -9 := by
  -- To be proved
  sorry

end math_proof_l242_242122


namespace max_boxes_in_large_box_l242_242389

def max_boxes (l_L w_L h_L : ℕ) (l_S w_S h_S : ℕ) : ℕ :=
  (l_L * w_L * h_L) / (l_S * w_S * h_S)

theorem max_boxes_in_large_box :
  let l_L := 8 * 100 -- converted to cm
  let w_L := 7 * 100 -- converted to cm
  let h_L := 6 * 100 -- converted to cm
  let l_S := 4
  let w_S := 7
  let h_S := 6
  max_boxes l_L w_L h_L l_S w_S h_S = 2000000 :=
by {
  let l_L := 800 -- converted to cm
  let w_L := 700 -- converted to cm
  let h_L := 600 -- converted to cm
  let l_S := 4
  let w_S := 7
  let h_S := 6
  trivial
}

end max_boxes_in_large_box_l242_242389


namespace arithmetic_sequence_S10_l242_242772

-- Definition of an arithmetic sequence and the corresponding sums S_n.
def is_arithmetic_sequence (S : ℕ → ℕ) : Prop :=
  ∃ d, ∀ n, S (n + 1) = S n + d

theorem arithmetic_sequence_S10 
  (S : ℕ → ℕ)
  (h1 : S 1 = 10)
  (h2 : S 2 = 20)
  (h_arith : is_arithmetic_sequence S) :
  S 10 = 100 :=
sorry

end arithmetic_sequence_S10_l242_242772


namespace geometric_series_sum_l242_242635

theorem geometric_series_sum :
  let a := 2
  let r := 2
  let n := 11
  let S := a * (r^n - 1) / (r - 1)
  S = 4094 := by
  sorry

end geometric_series_sum_l242_242635


namespace min_distance_mn_l242_242016

noncomputable def f (x : ℝ) : ℝ := x^2
noncomputable def g (x : ℝ) : ℝ := Real.log x

theorem min_distance_mn : ∃ m > 0, ∀ x > 0, |f x - g x| = 1/2 + 1/2 * Real.log 2 :=
by
  sorry

end min_distance_mn_l242_242016


namespace profit_percent_is_approx_6_point_35_l242_242270

noncomputable def selling_price : ℝ := 2552.36
noncomputable def cost_price : ℝ := 2400
noncomputable def profit_amount : ℝ := selling_price - cost_price
noncomputable def profit_percent : ℝ := (profit_amount / cost_price) * 100

theorem profit_percent_is_approx_6_point_35 : abs (profit_percent - 6.35) < 0.01 := sorry

end profit_percent_is_approx_6_point_35_l242_242270


namespace inequality_abc_l242_242538

variable {a b c : ℝ}

theorem inequality_abc (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) (h₄ : a * b * c = 8) :
  (a - 2) / (a + 1) + (b - 2) / (b + 1) + (c - 2) / (c + 1) ≤ 0 := by
  sorry

end inequality_abc_l242_242538


namespace yellow_yarns_count_l242_242703

theorem yellow_yarns_count (total_scarves red_yarn_count blue_yarn_count yellow_yarns scarves_per_yarn : ℕ) 
  (h1 : 3 = scarves_per_yarn)
  (h2 : red_yarn_count = 2)
  (h3 : blue_yarn_count = 6)
  (h4 : total_scarves = 36)
  :
  yellow_yarns = 4 :=
by 
  sorry

end yellow_yarns_count_l242_242703


namespace arithmetic_sum_S9_l242_242277

variable {a : ℕ → ℝ} -- Define the arithmetic sequence
variable (S : ℕ → ℝ) -- Define the sum of the first n terms
variable (d : ℝ) -- Define the common difference
variable (a_1 : ℝ) -- Define the first term of the sequence

-- Assume the arithmetic sequence properties
axiom arith_seq_def : ∀ n, a (n + 1) = a_1 + n * d

-- Define the sum of the first n terms
axiom sum_first_n_terms : ∀ n, S n = n / 2 * (2 * a_1 + (n - 1) * d)

-- Given condition
axiom given_condition : a 1 + a 7 = 15 - a 4

theorem arithmetic_sum_S9 : S 9 = 45 :=
by
  -- Proof omitted
  sorry

end arithmetic_sum_S9_l242_242277


namespace probability_of_satisfaction_l242_242748

-- Definitions for the conditions given in the problem
def dissatisfied_customers_leave_negative_review_probability : ℝ := 0.8
def satisfied_customers_leave_positive_review_probability : ℝ := 0.15
def negative_reviews : ℕ := 60
def positive_reviews : ℕ := 20
def expected_satisfaction_probability : ℝ := 0.64

-- The problem to prove
theorem probability_of_satisfaction :
  ∃ p : ℝ, (dissatisfied_customers_leave_negative_review_probability * (1 - p) = negative_reviews / (negative_reviews + positive_reviews)) ∧
           (satisfied_customers_leave_positive_review_probability * p = positive_reviews / (negative_reviews + positive_reviews)) ∧
           p = expected_satisfaction_probability := 
by
  sorry

end probability_of_satisfaction_l242_242748


namespace lizette_overall_average_is_94_l242_242629

-- Defining the given conditions
def third_quiz_score : ℕ := 92
def first_two_quizzes_average : ℕ := 95
def total_quizzes : ℕ := 3

-- Calculating total points from the conditions
def total_points : ℕ := first_two_quizzes_average * 2 + third_quiz_score

-- Defining the overall average to prove
def overall_average : ℕ := total_points / total_quizzes

-- The theorem stating Lizette's overall average after taking the third quiz
theorem lizette_overall_average_is_94 : overall_average = 94 := by
  sorry

end lizette_overall_average_is_94_l242_242629


namespace calculate_expression_l242_242368

theorem calculate_expression :
  (-0.125) ^ 2009 * (8 : ℝ) ^ 2009 = -1 :=
sorry

end calculate_expression_l242_242368


namespace avg_age_increase_l242_242356

theorem avg_age_increase 
    (student_count : ℕ) (avg_student_age : ℕ) (teacher_age : ℕ) (new_count : ℕ) (new_avg_age : ℕ) (age_increase : ℕ)
    (hc1 : student_count = 23)
    (hc2 : avg_student_age = 22)
    (hc3 : teacher_age = 46)
    (hc4 : new_count = student_count + 1)
    (hc5 : new_avg_age = ((avg_student_age * student_count + teacher_age) / new_count))
    (hc6 : age_increase = new_avg_age - avg_student_age) :
  age_increase = 1 := 
sorry

end avg_age_increase_l242_242356


namespace line_intersects_circle_l242_242438

-- Definitions
def radius : ℝ := 5
def distance_to_center : ℝ := 3

-- Theorem statement
theorem line_intersects_circle (r : ℝ) (d : ℝ) (h_r : r = radius) (h_d : d = distance_to_center) : d < r :=
by
  rw [h_r, h_d]
  exact sorry

end line_intersects_circle_l242_242438


namespace remainder_3a_plus_b_l242_242184

theorem remainder_3a_plus_b (p q : ℤ) (a b : ℤ)
  (h1 : a = 98 * p + 92)
  (h2 : b = 147 * q + 135) :
  ((3 * a + b) % 49) = 19 := by
sorry

end remainder_3a_plus_b_l242_242184


namespace add_pure_water_to_achieve_solution_l242_242426

theorem add_pure_water_to_achieve_solution
  (w : ℝ) (h_salt_content : 0.15 * 40 = 6) (h_new_concentration : 6 / (40 + w) = 0.1) :
  w = 20 :=
sorry

end add_pure_water_to_achieve_solution_l242_242426


namespace max_t_squared_value_l242_242724

noncomputable def max_t_squared (R : ℝ) : ℝ :=
  let PR_QR_sq_sum := 4 * R^2
  let max_PR_QR_prod := 2 * R^2
  PR_QR_sq_sum + 2 * max_PR_QR_prod

theorem max_t_squared_value (R : ℝ) : max_t_squared R = 8 * R^2 :=
  sorry

end max_t_squared_value_l242_242724


namespace geometric_sequence_n_value_l242_242074

theorem geometric_sequence_n_value (a : ℕ → ℝ) (q : ℝ) (n : ℕ) 
  (h1 : a 3 + a 6 = 36) 
  (h2 : a 4 + a 7 = 18)
  (h3 : a n = 1/2) :
  n = 9 :=
sorry

end geometric_sequence_n_value_l242_242074


namespace find_number_l242_242123

theorem find_number (x : ℝ) (h : (3 / 4) * (1 / 2) * (2 / 5) * x = 753.0000000000001) : 
  x = 5020.000000000001 :=
by 
  sorry

end find_number_l242_242123


namespace trader_sold_23_bags_l242_242484

theorem trader_sold_23_bags
    (initial_stock : ℕ) (restocked : ℕ) (final_stock : ℕ) (x : ℕ)
    (h_initial : initial_stock = 55)
    (h_restocked : restocked = 132)
    (h_final : final_stock = 164)
    (h_equation : initial_stock - x + restocked = final_stock) :
    x = 23 :=
by
    -- Here will be the proof of the theorem
    sorry

end trader_sold_23_bags_l242_242484


namespace linear_equation_in_one_variable_proof_l242_242510

noncomputable def is_linear_equation_in_one_variable (eq : String) : Prop :=
  eq = "3x = 2x" ∨ eq = "ax + b = 0"

theorem linear_equation_in_one_variable_proof :
  is_linear_equation_in_one_variable "3x = 2x" ∧ ¬is_linear_equation_in_one_variable "3x - (4 + 3x) = 2"
  ∧ ¬is_linear_equation_in_one_variable "x + y = 1" ∧ ¬is_linear_equation_in_one_variable "x^2 + 1 = 5" :=
by
  sorry

end linear_equation_in_one_variable_proof_l242_242510


namespace algebraic_expression_value_l242_242266

theorem algebraic_expression_value (x : ℝ) (h : 3 * x^2 - 4 * x = 6): 6 * x^2 - 8 * x - 9 = 3 :=
by sorry

end algebraic_expression_value_l242_242266


namespace range_of_f_l242_242188

noncomputable def f (x : ℝ) : ℝ := if x < 1 then 3 * x - 1 else 2 * x ^ 2

theorem range_of_f (a : ℝ) : (f (f a) = 2 * (f a) ^ 2) ↔ (a ≥ 2 / 3 ∨ a = 1 / 2) := 
  sorry

end range_of_f_l242_242188


namespace total_cookies_and_brownies_l242_242494

-- Define the conditions
def bagsOfCookies : ℕ := 272
def cookiesPerBag : ℕ := 45
def bagsOfBrownies : ℕ := 158
def browniesPerBag : ℕ := 32

-- Define the total cookies, total brownies, and total items
def totalCookies := bagsOfCookies * cookiesPerBag
def totalBrownies := bagsOfBrownies * browniesPerBag
def totalItems := totalCookies + totalBrownies

-- State the theorem to prove
theorem total_cookies_and_brownies : totalItems = 17296 := by
  sorry

end total_cookies_and_brownies_l242_242494


namespace tangent_lines_diff_expected_l242_242117

noncomputable def tangent_lines_diff (a : ℝ) (k1 k2 : ℝ) : Prop :=
  let curve (x : ℝ) := a * x + 2 * Real.log (|x|)
  let deriv (x : ℝ) := a + 2 / x
  -- Tangent conditions at some x1 > 0 for k1
  (∃ x1 : ℝ, 0 < x1 ∧ k1 = deriv x1 ∧ curve x1 = k1 * x1)
  -- Tangent conditions at some x2 < 0 for k2
  ∧ (∃ x2 : ℝ, x2 < 0 ∧ k2 = deriv x2 ∧ curve x2 = k2 * x2)
  -- The lines' slopes relations
  ∧ k1 > k2

theorem tangent_lines_diff_expected (a k1 k2 : ℝ) (h : tangent_lines_diff a k1 k2) :
  k1 - k2 = 4 / Real.exp 1 :=
sorry

end tangent_lines_diff_expected_l242_242117


namespace sum_of_solutions_eq_320_l242_242735

theorem sum_of_solutions_eq_320 :
  ∃ (S : Finset ℝ), 
  (∀ x ∈ S, 0 < x ∧ x < 180 ∧ (1 + (Real.sin x / Real.sin (4 * x)) = (Real.sin (3 * x) / Real.sin (2 * x)))) 
  ∧ S.sum id = 320 :=
by {
  sorry
}

end sum_of_solutions_eq_320_l242_242735


namespace return_journey_time_l242_242011

-- Define the conditions
def walking_speed : ℕ := 100 -- meters per minute
def walking_time : ℕ := 36 -- minutes
def running_speed : ℕ := 3 -- meters per second

-- Define derived values from conditions
def distance_walked : ℕ := walking_speed * walking_time -- meters
def running_speed_minute : ℕ := running_speed * 60 -- meters per minute

-- Statement of the problem
theorem return_journey_time :
  (distance_walked / running_speed_minute) = 20 := by
  sorry

end return_journey_time_l242_242011


namespace intersection_M_N_l242_242334

theorem intersection_M_N :
  let M := { x : ℝ | abs x ≤ 2 }
  let N := {-1, 0, 2, 3}
  M ∩ N = {-1, 0, 2} :=
by
  sorry

end intersection_M_N_l242_242334


namespace repeating_decimal_sum_l242_242700

def repeating_decimal_to_fraction (d : ℕ) (n : ℕ) : ℚ := n / ((10^d) - 1)

theorem repeating_decimal_sum : 
  repeating_decimal_to_fraction 1 2 + repeating_decimal_to_fraction 2 2 + repeating_decimal_to_fraction 4 2 = 2474646 / 9999 := 
sorry

end repeating_decimal_sum_l242_242700


namespace terry_lunch_combo_l242_242793

theorem terry_lunch_combo :
  let lettuce_options : ℕ := 2
  let tomato_options : ℕ := 3
  let olive_options : ℕ := 4
  let soup_options : ℕ := 2
  (lettuce_options * tomato_options * olive_options * soup_options = 48) := 
by
  sorry

end terry_lunch_combo_l242_242793


namespace lg_sum_geometric_seq_l242_242238

noncomputable def geometric_sequence (a : ℕ → ℝ) := ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem lg_sum_geometric_seq (a : ℕ → ℝ) (h1 : geometric_sequence a) (h2 : a 2 * a 5 * a 8 = 1) :
  Real.log (a 4) + Real.log (a 6) = 0 := 
sorry

end lg_sum_geometric_seq_l242_242238


namespace quadratic_value_range_l242_242076

theorem quadratic_value_range (y : ℝ) (h : y^3 - 6 * y^2 + 11 * y - 6 < 0) : 
  1 ≤ y^2 - 4 * y + 5 ∧ y^2 - 4 * y + 5 ≤ 2 := 
sorry

end quadratic_value_range_l242_242076


namespace student_l242_242741

noncomputable def allowance_after_video_games (A : ℝ) : ℝ := (3 / 7) * A

noncomputable def allowance_after_comic_books (remaining_after_video_games : ℝ) : ℝ := (3 / 5) * remaining_after_video_games

noncomputable def allowance_after_trading_cards (remaining_after_comic_books : ℝ) : ℝ := (5 / 8) * remaining_after_comic_books

noncomputable def last_allowance (remaining_after_trading_cards : ℝ) : ℝ := remaining_after_trading_cards

theorem student's_monthly_allowance (A : ℝ) (h1 : last_allowance (allowance_after_trading_cards (allowance_after_comic_books (allowance_after_video_games A))) = 1.20) :
  A = 7.47 := 
sorry

end student_l242_242741


namespace entree_cost_14_l242_242800

-- Define the conditions as given in part a)
def total_cost (e d : ℕ) : Prop := e + d = 23
def entree_more (e d : ℕ) : Prop := e = d + 5

-- The theorem to be proved
theorem entree_cost_14 (e d : ℕ) (h1 : total_cost e d) (h2 : entree_more e d) : e = 14 := 
by 
  sorry

end entree_cost_14_l242_242800


namespace paper_cut_square_l242_242363

noncomputable def proof_paper_cut_square : Prop :=
  ∃ x : ℝ, 1 < x ∧ x < 2 ∧ ((2 * x - 2 = 2 - x) ∨ (2 * (2 * x - 2) = 2 - x)) ∧ (x = 1.2 ∨ x = 1.5)

theorem paper_cut_square : proof_paper_cut_square :=
sorry

end paper_cut_square_l242_242363


namespace problem_solution_set_l242_242434

variable {a b c : ℝ}

theorem problem_solution_set (h_condition : ∀ x, 1 ≤ x → x ≤ 2 → a * x^2 - b * x + c ≥ 0) : 
  { x : ℝ | c * x^2 + b * x + a ≤ 0 } = { x : ℝ | x ≤ -1 } ∪ { x | -1/2 ≤ x } :=
by 
  sorry

end problem_solution_set_l242_242434


namespace simplify_expression_correct_l242_242483

variable {R : Type} [CommRing R]

def simplify_expression (x : R) : R :=
  2 * x^2 * (4 * x^3 - 3 * x + 1) - 7 * (x^3 - 3 * x^2 + 2 * x - 8)

theorem simplify_expression_correct (x : R) : 
  simplify_expression x = 8 * x^5 + 0 * x^4 - 13 * x^3 + 23 * x^2 - 14 * x + 56 :=
by
  sorry

end simplify_expression_correct_l242_242483


namespace compare_negatives_l242_242677

theorem compare_negatives : -3 < -2 :=
by {
  -- Placeholder for proof
  sorry
}

end compare_negatives_l242_242677


namespace num_teachers_l242_242492

variable (num_students : ℕ) (ticket_cost : ℕ) (total_cost : ℕ)

theorem num_teachers (h1 : num_students = 20) (h2 : ticket_cost = 5) (h3 : total_cost = 115) :
  (total_cost / ticket_cost - num_students = 3) :=
by
  sorry

end num_teachers_l242_242492


namespace sum_of_digits_inequality_l242_242406

def sum_of_digits (n : ℕ) : ℕ := -- Definition of the sum of digits function
  -- This should be defined, for demonstration we use a placeholder
  sorry

theorem sum_of_digits_inequality (n : ℕ) (h : n > 0) :
  sum_of_digits n ≤ 8 * sum_of_digits (8 * n) :=
sorry

end sum_of_digits_inequality_l242_242406


namespace T7_value_l242_242780

-- Define the geometric sequence a_n
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q > 0, ∀ n, a (n + 1) = a n * q

-- Define the even function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - (2 * a + 1) * x + 2 * a

-- The main theorem statement
theorem T7_value (a : ℕ → ℝ) (a2 a6 : ℝ) (a_val : ℝ) (q : ℝ) (T7 : ℝ) 
  (h1 : is_geometric_sequence a) 
  (h2 : a 2 = a2)
  (h3 : a 6 = a6)
  (h4 : a2 - 2 = f a_val 0)
  (h5 : a6 - 3 = f a_val 0)
  (h6 : q > 1)
  (h7 : T7 = a 1 * a 2 * a 3 * a 4 * a 5 * a 6 * a 7) : 
  T7 = 128 :=
sorry

end T7_value_l242_242780


namespace number_of_nickels_is_3_l242_242384

-- Defining the problem conditions
def total_coins := 8
def total_value := 53 -- in cents
def at_least_one_penny := 1
def at_least_one_nickel := 1
def at_least_one_dime := 1

-- Stating the proof problem
theorem number_of_nickels_is_3 : ∃ (pennies nickels dimes : Nat), 
  pennies + nickels + dimes = total_coins ∧ 
  pennies ≥ at_least_one_penny ∧ 
  nickels ≥ at_least_one_nickel ∧ 
  dimes ≥ at_least_one_dime ∧ 
  pennies + 5 * nickels + 10 * dimes = total_value ∧ 
  nickels = 3 := sorry

end number_of_nickels_is_3_l242_242384


namespace heating_rate_l242_242640

/-- 
 Andy is making fudge. He needs to raise the temperature of the candy mixture from 60 degrees to 240 degrees. 
 Then, he needs to cool it down to 170 degrees. The candy heats at a certain rate and cools at a rate of 7 degrees/minute.
 It takes 46 minutes for the candy to be done. Prove that the heating rate is 5 degrees per minute.
-/
theorem heating_rate (initial_temp heating_temp cooling_temp : ℝ) (cooling_rate total_time : ℝ) 
  (h1 : initial_temp = 60) (h2 : heating_temp = 240) (h3 : cooling_temp = 170) 
  (h4 : cooling_rate = 7) (h5 : total_time = 46) : 
  ∃ (H : ℝ), H = 5 :=
by 
  -- We declare here that the rate H exists and is 5 degrees per minute.
  let H : ℝ := 5
  existsi H
  sorry

end heating_rate_l242_242640


namespace smallest_arithmetic_mean_divisible_product_l242_242044

theorem smallest_arithmetic_mean_divisible_product :
  ∃ (n : ℕ), (∀ i : ℕ, i < 9 → Nat.gcd ((n + i) * 1111) 1111 = 1111) ∧ (n + 4 = 97) := 
  sorry

end smallest_arithmetic_mean_divisible_product_l242_242044


namespace liter_kerosene_cost_friday_l242_242007

-- Define initial conditions.
def cost_pound_rice_monday : ℚ := 0.36
def cost_dozen_eggs_monday : ℚ := cost_pound_rice_monday
def cost_half_liter_kerosene_monday : ℚ := (8 / 12) * cost_dozen_eggs_monday

-- Define the Wednesday price increase.
def percent_increase_rice : ℚ := 0.20
def cost_pound_rice_wednesday : ℚ := cost_pound_rice_monday * (1 + percent_increase_rice)
def cost_half_liter_kerosene_wednesday : ℚ := cost_half_liter_kerosene_monday * (1 + percent_increase_rice)

-- Define the Friday discount on eggs.
def percent_discount_eggs : ℚ := 0.10
def cost_dozen_eggs_friday : ℚ := cost_dozen_eggs_monday * (1 - percent_discount_eggs)
def cost_per_egg_friday : ℚ := cost_dozen_eggs_friday / 12

-- Define the price calculation for a liter of kerosene on Wednesday.
def cost_liter_kerosene_wednesday : ℚ := 2 * cost_half_liter_kerosene_wednesday

-- Define the final goal.
def cost_liter_kerosene_friday := cost_liter_kerosene_wednesday

theorem liter_kerosene_cost_friday : cost_liter_kerosene_friday = 0.576 := by
  sorry

end liter_kerosene_cost_friday_l242_242007


namespace problem_inequality_l242_242197

theorem problem_inequality (a b : ℝ) (n : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_ab : 1 < a * b) (h_n : 2 ≤ n) :
  (a + b)^n > a^n + b^n + 2^n - 2 :=
sorry

end problem_inequality_l242_242197


namespace symmetric_scanning_codes_count_l242_242485

structure Grid (n : ℕ) :=
  (cells : Fin n × Fin n → Bool)

def is_symmetric_90 (g : Grid 8) : Prop :=
  ∀ i j, g.cells (i, j) = g.cells (7 - j, i)

def is_symmetric_reflection_mid_side (g : Grid 8) : Prop :=
  ∀ i j, g.cells (i, j) = g.cells (7 - i, j) ∧ g.cells (i, j) = g.cells (i, 7 - j)

def is_symmetric_reflection_diagonal (g : Grid 8) : Prop :=
  ∀ i j, g.cells (i, j) = g.cells (j, i)

def has_at_least_one_black_and_one_white (g : Grid 8) : Prop :=
  ∃ i j, g.cells (i, j) ∧ ∃ i j, ¬g.cells (i, j)

noncomputable def count_symmetric_scanning_codes : ℕ :=
  (sorry : ℕ)

theorem symmetric_scanning_codes_count : count_symmetric_scanning_codes = 62 :=
  sorry

end symmetric_scanning_codes_count_l242_242485


namespace opposite_terminal_sides_l242_242023

theorem opposite_terminal_sides (α β : ℝ) (k : ℤ) (h : ∃ k : ℤ, α = β + 180 + k * 360) :
  α = β + 180 + k * 360 :=
by sorry

end opposite_terminal_sides_l242_242023


namespace right_rectangular_prism_volume_l242_242380

theorem right_rectangular_prism_volume (x y z : ℝ) 
  (h1 : x * y = 72) (h2 : y * z = 75) (h3 : x * z = 80) : 
  x * y * z = 657 :=
sorry

end right_rectangular_prism_volume_l242_242380


namespace first_train_takes_4_hours_less_l242_242094

-- Definitions of conditions
def distance: ℝ := 425.80645161290323
def speed_first_train: ℝ := 75
def speed_second_train: ℝ := 44

-- Lean statement to prove the correct answer
theorem first_train_takes_4_hours_less:
  (distance / speed_second_train) - (distance / speed_first_train) = 4 := 
  by
    -- Skip the actual proof
    sorry

end first_train_takes_4_hours_less_l242_242094


namespace jenni_age_l242_242466

theorem jenni_age 
    (B J : ℤ)
    (h1 : B + J = 70)
    (h2 : B - J = 32) : 
    J = 19 :=
by
  sorry

end jenni_age_l242_242466


namespace gain_percentage_l242_242308

theorem gain_percentage (x : ℝ) (CP : ℝ := 50 * x) (SP : ℝ := 60 * x) (Profit : ℝ := 10 * x) :
  ((Profit / CP) * 100) = 20 := 
by
  sorry

end gain_percentage_l242_242308


namespace bart_total_pages_l242_242583

theorem bart_total_pages (total_spent : ℝ) (cost_per_notepad : ℝ) (pages_per_notepad : ℕ)
  (h1 : total_spent = 10) (h2 : cost_per_notepad = 1.25) (h3 : pages_per_notepad = 60) :
  total_spent / cost_per_notepad * pages_per_notepad = 480 :=
by
  sorry

end bart_total_pages_l242_242583


namespace line_ellipse_intersection_l242_242514

-- Define the problem conditions and the proof problem statement.
theorem line_ellipse_intersection (k m : ℝ) : 
  (∀ x y, y - k * x - 1 = 0 → ((x^2 / 5) + (y^2 / m) = 1)) →
  (m ≥ 1) ∧ (m ≠ 5) ∧ (m < 5 ∨ m > 5) :=
sorry

end line_ellipse_intersection_l242_242514


namespace find_s_l242_242707

theorem find_s (s : Real) (h : ⌊s⌋ + s = 15.4) : s = 7.4 :=
sorry

end find_s_l242_242707


namespace find_values_of_m_l242_242698

theorem find_values_of_m (m : ℤ) (h₁ : m > 2022) (h₂ : (2022 + m) ∣ (2022 * m)) : 
  m = 1011 ∨ m = 2022 :=
sorry

end find_values_of_m_l242_242698


namespace class_total_students_l242_242718

theorem class_total_students (x y : ℕ)
  (initial_absent : y = (1/6) * x)
  (after_sending_chalk : y = (1/5) * (x - 1)) :
  x + y = 7 :=
by
  sorry

end class_total_students_l242_242718


namespace sum_of_cubes_ages_l242_242211

theorem sum_of_cubes_ages (d t h : ℕ) 
  (h1 : 4 * d + t = 3 * h) 
  (h2 : 4 * h ^ 2 = 2 * d ^ 2 + t ^ 2) 
  (h3 : Nat.gcd d (Nat.gcd t h) = 1)
  : d ^ 3 + t ^ 3 + h ^ 3 = 155557 :=
sorry

end sum_of_cubes_ages_l242_242211


namespace union_sets_l242_242654

def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {x | ∃ a ∈ A, x = 2^a}

theorem union_sets : A ∪ B = {0, 1, 2, 4} := by
  sorry

end union_sets_l242_242654


namespace initial_candles_count_l242_242068

section

variable (C : ℝ)
variable (h_Alyssa : C / 2 = C / 2)
variable (h_Chelsea : C / 2 - 0.7 * (C / 2) = 6)

theorem initial_candles_count : C = 40 := 
by sorry

end

end initial_candles_count_l242_242068


namespace number_of_girls_l242_242646

variable (G : ℕ) -- Number of girls in the school
axiom boys_count : G + 807 = 841 -- Given condition

theorem number_of_girls : G = 34 :=
by
  sorry

end number_of_girls_l242_242646


namespace simplify_sqrt_expression_l242_242596

theorem simplify_sqrt_expression :
  ( (Real.sqrt 112 + Real.sqrt 567) / Real.sqrt 175) = 13 / 5 := by
  -- conditions for simplification
  have h1 : Real.sqrt 112 = 4 * Real.sqrt 7 := sorry
  have h2 : Real.sqrt 567 = 9 * Real.sqrt 7 := sorry
  have h3 : Real.sqrt 175 = 5 * Real.sqrt 7 := sorry
  
  -- Use the conditions to simplify the expression
  rw [h1, h2, h3]
  -- Further simplification to achieve the result 13 / 5
  sorry

end simplify_sqrt_expression_l242_242596


namespace max_pots_l242_242452

theorem max_pots (x y z : ℕ) (h₁ : 3 * x + 4 * y + 9 * z = 100) (h₂ : 1 ≤ x) (h₃ : 1 ≤ y) (h₄ : 1 ≤ z) : 
  z ≤ 10 :=
sorry

end max_pots_l242_242452


namespace one_fourth_difference_l242_242347

theorem one_fourth_difference :
  (1 / 4) * ((9 * 5) - (7 + 3)) = 35 / 4 :=
by sorry

end one_fourth_difference_l242_242347


namespace difference_between_m_and_n_l242_242433

theorem difference_between_m_and_n (m n : ℕ) (hm : m > 0) (hn : n > 0) (h : 10 * 2^m = 2^n + 2^(n + 2)) :
  n - m = 1 :=
sorry

end difference_between_m_and_n_l242_242433


namespace gasVolume_at_20_l242_242569

variable (V : ℕ → ℕ)

/-- Given conditions:
 1. The gas volume expands by 3 cubic centimeters for every 5 degree rise in temperature.
 2. The volume is 30 cubic centimeters when the temperature is 30 degrees.
  -/
def gasVolume : Prop :=
  (∀ T ΔT, ΔT = 5 → V (T + ΔT) = V T + 3) ∧ V 30 = 30

theorem gasVolume_at_20 :
  gasVolume V → V 20 = 24 :=
by
  intro h
  -- Proof steps would go here.
  sorry

end gasVolume_at_20_l242_242569


namespace find_tangent_line_at_neg1_l242_242359

noncomputable def tangent_line (x : ℝ) : ℝ := 2 * x^2 + 3

theorem find_tangent_line_at_neg1 :
  let x := -1
  let m := 4 * x
  let y := 2 * x^2 + 3
  let tangent := y + m * (x - x)
  tangent = -4 * x + 1 :=
by
  sorry

end find_tangent_line_at_neg1_l242_242359


namespace find_d_l242_242467

open Real

-- Define the given conditions
variable (a b c d e : ℝ)

axiom cond1 : 3 * (a^2 + b^2 + c^2) + 4 = 2 * d + sqrt (a + b + c - d + e)
axiom cond2 : e = 1

-- Define the theorem stating that d = 7/4 under the given conditions
theorem find_d : d = 7/4 := by
  sorry

end find_d_l242_242467


namespace alice_bob_not_both_l242_242410

-- Define the group of 8 students
def total_students : ℕ := 8

-- Define the committee size
def committee_size : ℕ := 5

-- Calculate the total number of unrestricted committees
def total_committees : ℕ := Nat.choose total_students committee_size

-- Calculate the number of committees where both Alice and Bob are included
def alice_bob_committees : ℕ := Nat.choose (total_students - 2) (committee_size - 2)

-- Calculate the number of committees where Alice and Bob are not both included
def not_both_alice_bob : ℕ := total_committees - alice_bob_committees

-- Now state the theorem we want to prove
theorem alice_bob_not_both : not_both_alice_bob = 36 :=
by
  sorry

end alice_bob_not_both_l242_242410


namespace certain_number_example_l242_242727

theorem certain_number_example (x : ℝ) 
    (h1 : 213 * 16 = 3408)
    (h2 : 0.16 * x = 0.3408) : 
    x = 2.13 := 
by 
  sorry

end certain_number_example_l242_242727


namespace sum_of_x_and_y_l242_242606

theorem sum_of_x_and_y (x y : ℝ) 
  (h1 : (x - 1) ^ 3 + 1997 * (x - 1) = -1)
  (h2 : (y - 1) ^ 3 + 1997 * (y - 1) = 1) : 
  x + y = 2 :=
by
  sorry

end sum_of_x_and_y_l242_242606


namespace Toms_dog_age_in_6_years_l242_242768

-- Let's define the conditions
variables (B D : ℕ)
axiom h1 : B = 4 * D
axiom h2 : B + 6 = 30

-- Now we state the theorem
theorem Toms_dog_age_in_6_years :
  D + 6 = 12 :=
by
  sorry

end Toms_dog_age_in_6_years_l242_242768


namespace fraction_upgraded_sensors_l242_242124

theorem fraction_upgraded_sensors (N U : ℕ) (h1 : N = U / 3) (h2 : U = 3 * N) : 
  (U : ℚ) / (24 * N + U) = 1 / 9 := by
  sorry

end fraction_upgraded_sensors_l242_242124


namespace functional_eq_l242_242708

noncomputable def f (x : ℝ) : ℝ := sorry 

theorem functional_eq {f : ℝ → ℝ} (h1 : ∀ x, x * (f (x + 1) - f x) = f x) (h2 : ∀ x y, |f x - f y| ≤ |x - y|) :
  ∃ k : ℝ, ∀ x > 0, f x = k * x :=
sorry

end functional_eq_l242_242708


namespace value_of_expression_l242_242591

theorem value_of_expression : (20 * 24) / (2 * 0 + 2 * 4) = 60 := sorry

end value_of_expression_l242_242591


namespace train_length_is_360_l242_242049

-- Conditions from the problem
variable (speed_kmph : ℕ) (time_sec : ℕ) (platform_length_m : ℕ)

-- Definitions to be used for the conditions
def speed_ms (speed_kmph : ℕ) : ℤ := (speed_kmph * 1000) / 3600 -- Speed in m/s
def total_distance (speed_ms : ℤ) (time_sec : ℕ) : ℤ := speed_ms * (time_sec : ℤ) -- Total distance covered
def train_length (total_distance : ℤ) (platform_length : ℤ) : ℤ := total_distance - platform_length -- Length of the train

-- Assertion statement
theorem train_length_is_360 : train_length (total_distance (speed_ms speed_kmph) time_sec) platform_length_m = 360 := 
  by sorry

end train_length_is_360_l242_242049


namespace total_students_in_class_l242_242678

theorem total_students_in_class : 
  ∀ (total_candies students_candies : ℕ), 
    total_candies = 901 → students_candies = 53 → 
    students_candies * (total_candies / students_candies) = total_candies ∧ 
    total_candies % students_candies = 0 → 
    total_candies / students_candies = 17 := 
by 
  sorry

end total_students_in_class_l242_242678


namespace find_k_l242_242313

variable (x y z k : ℝ)

def fractions_are_equal : Prop := (9 / (x + y) = k / (x + z) ∧ k / (x + z) = 15 / (z - y))

theorem find_k (h : fractions_are_equal x y z k) : k = 24 := by
  sorry

end find_k_l242_242313


namespace jensen_meetings_percentage_l242_242299

theorem jensen_meetings_percentage :
  ∃ (first second third total_work_day total_meeting_time : ℕ),
    total_work_day = 600 ∧
    first = 35 ∧
    second = 2 * first ∧
    third = first + second ∧
    total_meeting_time = first + second + third ∧
    (total_meeting_time * 100) / total_work_day = 35 := sorry

end jensen_meetings_percentage_l242_242299


namespace solve_for_x_l242_242178

theorem solve_for_x (x : ℝ) (h : x + 2 = 7) : x = 5 := 
by
  sorry

end solve_for_x_l242_242178


namespace chickens_and_rabbits_l242_242225

theorem chickens_and_rabbits (c r : ℕ) 
    (h1 : c = 2 * r - 5)
    (h2 : 2 * c + r = 92) : ∃ c r : ℕ, (c = 2 * r - 5) ∧ (2 * c + r = 92) := 
by 
    -- proof steps
    sorry

end chickens_and_rabbits_l242_242225


namespace balls_in_each_package_l242_242728

theorem balls_in_each_package (x : ℕ) (h : 21 * x = 399) : x = 19 :=
by
  sorry

end balls_in_each_package_l242_242728


namespace sum_of_x_y_l242_242536

theorem sum_of_x_y (m x y : ℝ) (h₁ : x + m = 4) (h₂ : y - 3 = m) : x + y = 7 :=
sorry

end sum_of_x_y_l242_242536


namespace two_points_same_color_at_distance_one_l242_242181

theorem two_points_same_color_at_distance_one (color : ℝ × ℝ → ℕ) (h : ∀p : ℝ × ℝ, color p < 3) :
  ∃ (p q : ℝ × ℝ), dist p q = 1 ∧ color p = color q :=
sorry

end two_points_same_color_at_distance_one_l242_242181


namespace ax_by_powers_l242_242374

theorem ax_by_powers (a b x y : ℝ) (h1 : a * x + b * y = 5) 
                      (h2: a * x^2 + b * y^2 = 11)
                      (h3: a * x^3 + b * y^3 = 25)
                      (h4: a * x^4 + b * y^4 = 59) : 
                      a * x^5 + b * y^5 = 145 := 
by 
  -- Include the proof steps here if needed 
  sorry

end ax_by_powers_l242_242374


namespace ancient_china_pentatonic_scale_l242_242175

theorem ancient_china_pentatonic_scale (a : ℝ) (h : a * (2/3) * (4/3) * (2/3) = 32) : a = 54 :=
by
  sorry

end ancient_china_pentatonic_scale_l242_242175


namespace greatest_divisor_with_sum_of_digits_four_l242_242554

/-- Define the given numbers -/
def a := 4665
def b := 6905

/-- Define the sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

/-- Define the greatest number n that divides both a and b, leaving the same remainder and having a sum of digits equal to 4 -/
theorem greatest_divisor_with_sum_of_digits_four :
  ∃ (n : ℕ), (∀ (d : ℕ), (d ∣ a - b ∧ sum_of_digits d = 4) → d ≤ n) ∧ (n ∣ a - b) ∧ (sum_of_digits n = 4) ∧ n = 40 := sorry

end greatest_divisor_with_sum_of_digits_four_l242_242554


namespace largest_x_value_satisfies_largest_x_value_l242_242422

theorem largest_x_value (x : ℚ) (h : Real.sqrt (3 * x) = 5 * x) : x ≤ 3 / 25 :=
sorry

theorem satisfies_largest_x_value : 
  ∃ x : ℚ, Real.sqrt (3 * x) = 5 * x ∧ x = 3 / 25 :=
sorry

end largest_x_value_satisfies_largest_x_value_l242_242422


namespace contradiction_method_conditions_l242_242470

theorem contradiction_method_conditions :
  (using_judgments_contrary_to_conclusion ∧ using_conditions_of_original_proposition ∧ using_axioms_theorems_definitions) =
  (needed_conditions_method_of_contradiction) :=
sorry

end contradiction_method_conditions_l242_242470


namespace florida_vs_georgia_license_plates_l242_242655

theorem florida_vs_georgia_license_plates :
  26 ^ 4 * 10 ^ 3 - 26 ^ 3 * 10 ^ 3 = 439400000 := by
  -- proof is omitted as directed
  sorry

end florida_vs_georgia_license_plates_l242_242655


namespace sum_of_positive_odd_divisors_of_90_l242_242142

-- Define the conditions: the odd divisors of 45
def odd_divisors_45 : List ℕ := [1, 3, 5, 9, 15, 45]

-- Define a function to sum the elements of a list
def sum_list (l : List ℕ) : ℕ := l.foldr (· + ·) 0

-- Now, state the theorem
theorem sum_of_positive_odd_divisors_of_90 : sum_list odd_divisors_45 = 78 := by
  sorry

end sum_of_positive_odd_divisors_of_90_l242_242142


namespace range_of_a_l242_242000

noncomputable def f : ℝ → ℝ := sorry

variables (a : ℝ)
variable (is_even : ∀ x : ℝ, f (x) = f (-x)) -- f is even
variable (monotonic_incr : ∀ x y : ℝ, 0 ≤ x → x ≤ y → f x ≤ f y) -- f is monotonically increasing in [0, +∞)

theorem range_of_a
  (h : f (Real.log a / Real.log 2) + f (Real.log (1/a) / Real.log 2) ≤ 2 * f 1) : 
  1 / 2 ≤ a ∧ a ≤ 2 :=
by
  sorry

end range_of_a_l242_242000


namespace train_speed_l242_242560

theorem train_speed (length : ℝ) (time : ℝ) (h_length : length = 300) (h_time : time = 15) : 
  (length / time) * 3.6 = 72 :=
by
  sorry

end train_speed_l242_242560


namespace problem1_problem2_l242_242499

-- (Problem 1)
def A : Set ℝ := {x | x^2 + 2 * x < 0}
def B : Set ℝ := {x | x ≥ -1}
def complement_A : Set ℝ := {x | x ≤ -2 ∨ x ≥ 0}
def intersection_complement_A_B : Set ℝ := {x | x ≥ 0}

theorem problem1 : (complement_A ∩ B) = intersection_complement_A_B :=
by
  sorry

-- (Problem 2)
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < 2 * a + 1}

theorem problem2 {a : ℝ} : (C a ⊆ A) ↔ (a ≤ -1 / 2) :=
by
  sorry

end problem1_problem2_l242_242499


namespace abs_neg_three_l242_242129

theorem abs_neg_three : abs (-3) = 3 := 
by
  sorry

end abs_neg_three_l242_242129


namespace triangle_area_l242_242747

theorem triangle_area (a b c : ℝ) (h1 : a / b = 3 / 4) (h2 : b / c = 4 / 5) (h3 : a + b + c = 60) : 
  (1/2) * a * b = 150 :=
by
  sorry

end triangle_area_l242_242747


namespace find_x_l242_242831

/--
Given the following conditions:
1. The sum of angles around a point is 360 degrees.
2. The angles are 7x, 6x, 3x, and (2x + y).
3. y = 2x.

Prove that x = 18 degrees.
-/
theorem find_x (x y : ℝ) (h : 18 * x + y = 360) (h_y : y = 2 * x) : x = 18 :=
by
  sorry

end find_x_l242_242831


namespace find_roots_l242_242460

open Real

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_roots 
  (h_symm : ∀ x : ℝ, f (2 + x) = f (2 - x))
  (h_three_roots : ∃ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ f a = 0 ∧ f b = 0 ∧ f c = 0)
  (h_zero_root : f 0 = 0) :
  ∃ a b : ℝ, a = 2 ∧ b = 4 ∧ f a = 0 ∧ f b = 0 :=
sorry

end find_roots_l242_242460


namespace min_sum_squares_l242_242272

noncomputable def distances (P : ℝ) : ℝ :=
  let AP := P
  let BP := |P - 1|
  let CP := |P - 2|
  let DP := |P - 5|
  let EP := |P - 13|
  AP^2 + BP^2 + CP^2 + DP^2 + EP^2

theorem min_sum_squares : ∀ P : ℝ, distances P ≥ 88.2 :=
by
  sorry

end min_sum_squares_l242_242272


namespace quarters_initial_l242_242670

-- Define the given conditions
def candies_cost_dimes : Nat := 4 * 3
def candies_cost_cents : Nat := candies_cost_dimes * 10
def lollipop_cost_quarters : Nat := 1
def lollipop_cost_cents : Nat := lollipop_cost_quarters * 25
def total_spent_cents : Nat := candies_cost_cents + lollipop_cost_cents
def money_left_cents : Nat := 195
def total_initial_money_cents : Nat := money_left_cents + total_spent_cents
def dimes_count : Nat := 19
def dimes_value_cents : Nat := dimes_count * 10

-- Prove that the number of quarters initially is 6
theorem quarters_initial (quarters_count : Nat) (h : quarters_count * 25 = total_initial_money_cents - dimes_value_cents) : quarters_count = 6 :=
by
  sorry

end quarters_initial_l242_242670


namespace factor_correct_l242_242121

noncomputable def factor_expr (x : ℝ) : ℝ :=
  75 * x^3 - 225 * x^10
  
noncomputable def factored_form (x : ℝ) : ℝ :=
  75 * x^3 * (1 - 3 * x^7)

theorem factor_correct (x : ℝ): 
  factor_expr x = factored_form x :=
by
  -- Proof omitted
  sorry

end factor_correct_l242_242121


namespace monthly_payment_l242_242425

theorem monthly_payment (price : ℝ) (discount_rate : ℝ) (down_payment : ℝ) (months : ℕ) (monthly_payment : ℝ) :
  price = 480 ∧ discount_rate = 0.05 ∧ down_payment = 150 ∧ months = 3 ∧
  monthly_payment = (price * (1 - discount_rate) - down_payment) / months →
  monthly_payment = 102 :=
by
  sorry

end monthly_payment_l242_242425


namespace min_value_of_sum_of_squares_l242_242769

theorem min_value_of_sum_of_squares (a b c : ℝ) (h : a + b + c = 1) : a^2 + b^2 + c^2 ≥ 1 / 3 :=
by
  sorry

end min_value_of_sum_of_squares_l242_242769


namespace tangent_line_slope_through_origin_l242_242351

theorem tangent_line_slope_through_origin :
  (∃ a : ℝ, (a^3 + a + 16 = (3 * a^2 + 1) * a ∧ a = 2)) →
  (3 * (2 : ℝ)^2 + 1 = 13) :=
by
  intro h
  -- Detailed proof goes here
  sorry

end tangent_line_slope_through_origin_l242_242351


namespace conditional_two_exits_one_effective_l242_242327

def conditional_structure (decide : Bool) : Prop :=
  if decide then True else False

theorem conditional_two_exits_one_effective (decide : Bool) :
  conditional_structure decide ↔ True :=
by
  sorry

end conditional_two_exits_one_effective_l242_242327


namespace motorbike_speed_l242_242592

noncomputable def speed_of_motorbike 
  (V_train : ℝ) 
  (t_overtake : ℝ) 
  (train_length_m : ℝ) : ℝ :=
  V_train - (train_length_m / 1000) * (3600 / t_overtake)

theorem motorbike_speed : 
  speed_of_motorbike 100 80 800.064 = 63.99712 :=
by
  -- this is where the proof steps would go
  sorry

end motorbike_speed_l242_242592


namespace terminal_side_of_half_angle_quadrant_l242_242440

def is_angle_in_third_quadrant (α : ℝ) (k : ℤ) : Prop :=
  k * 360 + 180 < α ∧ α < k * 360 + 270

def is_terminal_side_of_half_angle_in_quadrant (α : ℝ) : Prop :=
  (∃ n : ℤ, n * 360 + 90 < α / 2 ∧ α / 2 < n * 360 + 135) ∨ 
  (∃ n : ℤ, n * 360 + 270 < α / 2 ∧ α / 2 < n * 360 + 315)

theorem terminal_side_of_half_angle_quadrant (α : ℝ) (k : ℤ) :
  is_angle_in_third_quadrant α k → is_terminal_side_of_half_angle_in_quadrant α := 
sorry

end terminal_side_of_half_angle_quadrant_l242_242440


namespace problem1_problem2_l242_242408

-- Problem (1)
theorem problem1 : (Real.sqrt 12 + (-1 / 3)⁻¹ + (-2)^2 = 2 * Real.sqrt 3 + 1) :=
  sorry

-- Problem (2)
theorem problem2 (a : Real) (h : a ≠ 2) :
  (2 * a / (a^2 - 4) / (1 + (a - 2) / (a + 2)) = 1 / (a - 2)) :=
  sorry

end problem1_problem2_l242_242408


namespace negation_of_forall_ge_zero_l242_242729

theorem negation_of_forall_ge_zero :
  (¬ (∀ x : ℝ, x^2 ≥ 0)) ↔ (∃ x : ℝ, x^2 < 0) :=
sorry

end negation_of_forall_ge_zero_l242_242729


namespace cube_greater_l242_242104

theorem cube_greater (a b : ℝ) (h : a > b) : a^3 > b^3 := 
sorry

end cube_greater_l242_242104


namespace Jerome_money_left_l242_242447

-- Definitions based on conditions
def J_half := 43              -- Half of Jerome's money
def to_Meg := 8               -- Amount Jerome gave to Meg
def to_Bianca := to_Meg * 3   -- Amount Jerome gave to Bianca

-- Total initial amount of Jerome's money
def J_initial : ℕ := J_half * 2

-- Amount left after giving money to Meg
def after_Meg : ℕ := J_initial - to_Meg

-- Amount left after giving money to Bianca
def after_Bianca : ℕ := after_Meg - to_Bianca

-- Statement to be proved
theorem Jerome_money_left : after_Bianca = 54 :=
by
  sorry

end Jerome_money_left_l242_242447


namespace tangent_product_value_l242_242159

theorem tangent_product_value (A B : ℝ) (hA : A = 20) (hB : B = 25) :
    (1 + Real.tan (A * Real.pi / 180)) * (1 + Real.tan (B * Real.pi / 180)) = 2 :=
sorry

end tangent_product_value_l242_242159


namespace phase_and_initial_phase_theorem_l242_242341

open Real

noncomputable def phase_and_initial_phase (x : ℝ) : ℝ := 3 * sin (-x + π / 6)

theorem phase_and_initial_phase_theorem :
  ∃ φ : ℝ, ∃ ψ : ℝ,
    ∀ x : ℝ, phase_and_initial_phase x = 3 * sin (x + φ) ∧
    (φ = 5 * π / 6) ∧ (ψ = φ) :=
sorry

end phase_and_initial_phase_theorem_l242_242341


namespace negation_of_proposition_l242_242396

theorem negation_of_proposition :
  (¬ ∃ x : ℝ, x^2 - x - 1 = 0) ↔ (∀ x : ℝ, x^2 - x - 1 ≠ 0) :=
by sorry

end negation_of_proposition_l242_242396


namespace sam_after_joan_took_marbles_l242_242695

theorem sam_after_joan_took_marbles
  (original_yellow : ℕ)
  (marbles_taken_by_joan : ℕ)
  (remaining_yellow : ℕ)
  (h1 : original_yellow = 86)
  (h2 : marbles_taken_by_joan = 25)
  (h3 : remaining_yellow = original_yellow - marbles_taken_by_joan) :
  remaining_yellow = 61 :=
by
  sorry

end sam_after_joan_took_marbles_l242_242695


namespace second_concert_attendance_l242_242633

def first_concert_attendance : ℕ := 65899
def additional_people : ℕ := 119

theorem second_concert_attendance : first_concert_attendance + additional_people = 66018 := 
by 
  -- Proof is not discussed here, only the statement is required.
sorry

end second_concert_attendance_l242_242633


namespace spinsters_count_l242_242034

theorem spinsters_count (S C : ℕ) (h1 : S / C = 2 / 9) (h2 : C = S + 42) : S = 12 := by
  sorry

end spinsters_count_l242_242034


namespace marco_might_need_at_least_n_tables_n_tables_are_sufficient_l242_242032
open Function

variables (n : ℕ) (friends_sticker_sets : Fin n → Finset (Fin n))

-- Each friend is missing exactly one unique sticker
def each_friend_missing_one_unique_sticker :=
  ∀ i : Fin n, ∃ j : Fin n, friends_sticker_sets i = (Finset.univ \ {j})

-- A pair of friends is wholesome if their combined collection has all stickers
def is_wholesome_pair (i j : Fin n) :=
  ∀ s : Fin n, s ∈ friends_sticker_sets i ∨ s ∈ friends_sticker_sets j

-- Main problem statements
-- Problem 1: Marco might need to reserve at least n different tables
theorem marco_might_need_at_least_n_tables 
  (h : each_friend_missing_one_unique_sticker n friends_sticker_sets) : 
  ∃ i j : Fin n, i ≠ j ∧ is_wholesome_pair n friends_sticker_sets i j :=
sorry

-- Problem 2: n tables will always be enough for Marco to achieve his goal
theorem n_tables_are_sufficient
  (h : each_friend_missing_one_unique_sticker n friends_sticker_sets) :
  ∃ arrangement : Fin n → Fin n, ∀ i j, i ≠ j → arrangement i ≠ arrangement j :=
sorry

end marco_might_need_at_least_n_tables_n_tables_are_sufficient_l242_242032


namespace find_top_row_number_l242_242174

theorem find_top_row_number (x z : ℕ) (h1 : 8 = x * 2) (h2 : 16 = 2 * z)
  (h3 : 56 = 8 * 7) (h4 : 112 = 16 * 7) : x = 4 :=
by sorry

end find_top_row_number_l242_242174


namespace number_and_sum_of_g3_l242_242120

-- Define the function g with its conditions
variable (g : ℝ → ℝ)
variable (h : ∀ x y : ℝ, g (x * g y - x) = 2 * x * y + g x)

-- Define the problem parameters
def n : ℕ := sorry -- Number of possible values of g(3)
def s : ℝ := sorry -- Sum of all possible values of g(3)

-- The main statement to be proved
theorem number_and_sum_of_g3 : n * s = 0 := sorry

end number_and_sum_of_g3_l242_242120


namespace triangle_properties_l242_242828

open Real

variables (A B C a b c : ℝ) (triangle_obtuse triangle_right triangle_acute : Prop)

-- Declaration of properties 
def sin_gt (A B : ℝ) := sin A > sin B
def tan_product_lt (A C : ℝ) := tan A * tan C < 1
def cos_squared_eq (A B C : ℝ) := cos A ^ 2 + cos B ^ 2 - cos C ^ 2 = 1

theorem triangle_properties :
  (sin_gt A B → A > B) ∧
  (triangle_obtuse → tan_product_lt A C) ∧
  (cos_squared_eq A B C → triangle_right) :=
  by sorry

end triangle_properties_l242_242828


namespace arithmetic_expression_l242_242135

theorem arithmetic_expression :
  7 / 2 - 3 - 5 + 3 * 4 = 7.5 :=
by {
  -- We state the main equivalence to be proven
  sorry
}

end arithmetic_expression_l242_242135


namespace events_equally_likely_iff_N_eq_18_l242_242600

variable (N : ℕ)

-- Define the number of combinations in the draws
def C (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Define the sums of selecting balls
noncomputable def S_63_10 (N: ℕ) : ℕ := sorry -- placeholder definition
noncomputable def S_44_8 (N: ℕ) : ℕ := sorry 

-- Condition for the events being equally likely
theorem events_equally_likely_iff_N_eq_18 : 
  (S_63_10 N * C N 8 = S_44_8 N * C N 10) ↔ N = 18 :=
sorry

end events_equally_likely_iff_N_eq_18_l242_242600


namespace evaluate_expression_l242_242285

-- Define the base and the exponents
def base : ℝ := 64
def exponent1 : ℝ := 0.125
def exponent2 : ℝ := 0.375
def combined_result : ℝ := 8

-- Statement of the problem
theorem evaluate_expression : (base^exponent1) * (base^exponent2) = combined_result := 
by 
  sorry

end evaluate_expression_l242_242285


namespace largest_number_is_y_l242_242168

def x := 8.1235
def y := 8.12355555555555 -- 8.123\overline{5}
def z := 8.12345454545454 -- 8.123\overline{45}
def w := 8.12345345345345 -- 8.12\overline{345}
def v := 8.12345234523452 -- 8.1\overline{2345}

theorem largest_number_is_y : y > x ∧ y > z ∧ y > w ∧ y > v :=
by
-- Proof steps would go here.
sorry

end largest_number_is_y_l242_242168


namespace selection_ways_l242_242478

/-- There are a total of 70 ways to select 3 people from 4 teachers and 5 students,
with the condition that there must be at least one teacher and one student among the selected. -/
theorem selection_ways (teachers students : ℕ) (T : 4 = teachers) (S : 5 = students) :
  ∃ (ways : ℕ), ways = 70 := by
  sorry

end selection_ways_l242_242478


namespace mass_of_man_l242_242826

def density_of_water : ℝ := 1000  -- kg/m³
def boat_length : ℝ := 4  -- meters
def boat_breadth : ℝ := 2  -- meters
def sinking_depth : ℝ := 0.01  -- meters (1 cm)

theorem mass_of_man
  (V : ℝ := boat_length * boat_breadth * sinking_depth)
  (m : ℝ := V * density_of_water) :
  m = 80 :=
by
  sorry

end mass_of_man_l242_242826


namespace base_length_of_isosceles_triangle_l242_242751

-- Definitions based on given conditions
def is_isosceles (a b : ℕ) (c : ℕ) :=
a = b ∧ c = c

def side_length : ℕ := 6
def perimeter : ℕ := 20

-- Theorem to prove the base length
theorem base_length_of_isosceles_triangle (b : ℕ) (h1 : 2 * side_length + b = perimeter) :
  b = 8 :=
sorry

end base_length_of_isosceles_triangle_l242_242751


namespace range_of_m_l242_242340

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 1 then 2^x 
else if 1 < x ∧ x ≤ 2 then Real.log (x - 1) 
else 0 -- function is not defined outside the given range

theorem range_of_m (m : ℝ) : (∀ x : ℝ, 
  (x ≤ 1 → 2^x ≤ 4 - m * x) ∧ 
  (1 < x ∧ x ≤ 2 → Real.log (x - 1) ≤ 4 - m * x)) → 
  0 ≤ m ∧ m ≤ 2 := 
sorry

end range_of_m_l242_242340


namespace repeating_decimal_equals_fraction_l242_242248

theorem repeating_decimal_equals_fraction : 
  let a := 58 / 100
  let r := 1 / 100
  let S := a / (1 - r)
  S = (58 : ℚ) / 99 :=
by
  sorry

end repeating_decimal_equals_fraction_l242_242248


namespace malou_average_score_l242_242443

def quiz1_score := 91
def quiz2_score := 90
def quiz3_score := 92

def sum_of_scores := quiz1_score + quiz2_score + quiz3_score
def number_of_quizzes := 3

theorem malou_average_score : sum_of_scores / number_of_quizzes = 91 :=
by
  sorry

end malou_average_score_l242_242443


namespace quadratic_inequality_solution_l242_242167

theorem quadratic_inequality_solution (k : ℝ) :
  (∀ x : ℝ, k * x^2 + k * x - (3 / 4) < 0) ↔ -3 < k ∧ k ≤ 0 :=
by
  sorry

end quadratic_inequality_solution_l242_242167


namespace derivative_equals_l242_242236

noncomputable def func (x : ℝ) : ℝ :=
  (3 / (8 * Real.sqrt 2) * Real.log ((Real.sqrt 2 + Real.tanh x) / (Real.sqrt 2 - Real.tanh x)))
  - (Real.tanh x / (4 * (2 - (Real.tanh x)^2)))

theorem derivative_equals :
  ∀ x : ℝ, deriv func x = 1 / (2 + (Real.cosh x)^2)^2 :=
by {
  sorry
}

end derivative_equals_l242_242236


namespace sum_of_roots_of_quadratic_l242_242587

theorem sum_of_roots_of_quadratic (m n : ℝ) (h1 : m = 2 * n) (h2 : ∀ x : ℝ, x ^ 2 + m * x + n = 0) :
    m + n = 3 / 2 :=
sorry

end sum_of_roots_of_quadratic_l242_242587


namespace sum_of_coefficients_l242_242311

theorem sum_of_coefficients (p : ℕ) (hp : Nat.Prime p) (a b c : ℕ)
    (f : ℕ → ℕ) (hf : ∀ x, f x = a * x ^ 2 + b * x + c)
    (h_range : 0 < a ∧ a ≤ p ∧ 0 < b ∧ b ≤ p ∧ 0 < c ∧ c ≤ p)
    (h_div : ∀ x, x > 0 → p ∣ (f x)) : 
    a + b + c = 3 * p := 
sorry

end sum_of_coefficients_l242_242311


namespace tops_count_l242_242489

def price_eq (C T : ℝ) : Prop := 3 * C + 6 * T = 1500 ∧ C + 12 * T = 1500

def tops_to_buy (C T : ℝ) (num_tops : ℝ) : Prop := 500 = 100 * num_tops

theorem tops_count (C T num_tops : ℝ) (h1 : price_eq C T) (h2 : tops_to_buy C T num_tops) : num_tops = 5 :=
by
  sorry

end tops_count_l242_242489


namespace min_x2_y2_of_product_eq_zero_l242_242716

theorem min_x2_y2_of_product_eq_zero (x y : ℝ) (h : (x + 8) * (y - 8) = 0) : x^2 + y^2 = 64 :=
sorry

end min_x2_y2_of_product_eq_zero_l242_242716


namespace work_completion_time_l242_242659

theorem work_completion_time 
    (A B : ℝ) 
    (h1 : A = 2 * B) 
    (h2 : (A + B) * 18 = 1) : 
    1 / A = 27 := 
by 
    sorry

end work_completion_time_l242_242659


namespace binom_18_7_l242_242581

theorem binom_18_7 : Nat.choose 18 7 = 31824 := by sorry

end binom_18_7_l242_242581


namespace quadratic_root_form_l242_242397

theorem quadratic_root_form {a b : ℂ} (h : 6 * a ^ 2 - 5 * a + 18 = 0 ∧ a.im = 0 ∧ b.im = 0) : 
  a + b^2 = (467:ℚ) / 144 :=
by
  sorry

end quadratic_root_form_l242_242397


namespace angle_same_terminal_side_l242_242312

theorem angle_same_terminal_side (k : ℤ) : ∃ k : ℤ, -330 = k * 360 + 30 :=
by
  use -1
  sorry

end angle_same_terminal_side_l242_242312


namespace abs_diff_roots_quad_eq_l242_242806

theorem abs_diff_roots_quad_eq : 
  ∀ (r1 r2 : ℝ), 
  (r1 * r2 = 12) ∧ (r1 + r2 = 7) → |r1 - r2| = 1 :=
by
  intro r1 r2 h
  sorry

end abs_diff_roots_quad_eq_l242_242806


namespace find_extra_factor_l242_242555

theorem find_extra_factor (w : ℕ) (h1 : w > 0) (h2 : w = 156) (h3 : ∃ (k : ℕ), (2^5 * 13^2) ∣ (936 * w))
  : 3 ∣ w := sorry

end find_extra_factor_l242_242555


namespace obtuse_angled_triangles_in_polygon_l242_242504

/-- The number of obtuse-angled triangles formed by the vertices of a regular polygon with 2n+1 sides -/
theorem obtuse_angled_triangles_in_polygon (n : ℕ) : 
  (2 * n + 1) * (n * (n - 1)) / 2 = (2 * n + 1) * (n * (n - 1)) / 2 :=
by
  sorry

end obtuse_angled_triangles_in_polygon_l242_242504


namespace smallest_five_digit_multiple_of_18_l242_242166

theorem smallest_five_digit_multiple_of_18 : ∃ n : ℕ, 10000 ≤ n ∧ n ≤ 99999 ∧ n % 18 = 0 ∧ ∀ m : ℕ, 10000 ≤ m ∧ m ≤ 99999 ∧ m % 18 = 0 → n ≤ m :=
  sorry

end smallest_five_digit_multiple_of_18_l242_242166


namespace garden_snake_length_l242_242196

theorem garden_snake_length :
  ∀ (garden_snake boa_constrictor : ℝ),
    boa_constrictor * 7.0 = garden_snake →
    boa_constrictor = 1.428571429 →
    garden_snake = 10.0 :=
by
  intros garden_snake boa_constrictor H1 H2
  sorry

end garden_snake_length_l242_242196


namespace stone_length_is_correct_l242_242017

variable (length_m width_m : ℕ)
variable (num_stones : ℕ)
variable (width_stone dm : ℕ)

def length_of_each_stone (length_m : ℕ) (width_m : ℕ) (num_stones : ℕ) (width_stone : ℕ) : ℕ :=
  let length_dm := length_m * 10
  let width_dm := width_m * 10
  let area_hall := length_dm * width_dm
  let area_stone := width_stone * 5
  (area_hall / num_stones) / width_stone

theorem stone_length_is_correct :
  length_of_each_stone 36 15 5400 5 = 2 := by
  sorry

end stone_length_is_correct_l242_242017


namespace unique_y_star_l242_242705

def star (x y : ℝ) : ℝ := 5 * x - 4 * y + 2 * x * y

theorem unique_y_star :
  ∃! y : ℝ, star 4 y = 20 :=
by 
  sorry

end unique_y_star_l242_242705


namespace consecutive_roots_prime_q_l242_242586

theorem consecutive_roots_prime_q (p q : ℤ) (h1 : Prime q)
  (h2 : ∃ x1 x2 : ℤ, 
    x1 ≠ x2 ∧ 
    (x1 = x2 + 1 ∨ x1 = x2 - 1) ∧ 
    x1 + x2 = p ∧ 
    x1 * x2 = q) : (p = 3 ∨ p = -3) ∧ q = 2 :=
by
  sorry

end consecutive_roots_prime_q_l242_242586


namespace g_at_seven_equals_92_l242_242209

def g (n : ℕ) : ℕ := n^2 + 2*n + 29

theorem g_at_seven_equals_92 : g 7 = 92 :=
by
  sorry

end g_at_seven_equals_92_l242_242209


namespace smallest_angle_in_triangle_l242_242669

theorem smallest_angle_in_triangle (k : ℕ) 
  (h1 : 3 * k + 4 * k + 5 * k = 180) : 
  3 * k = 45 := 
by sorry

end smallest_angle_in_triangle_l242_242669


namespace find_percentage_loss_l242_242190

theorem find_percentage_loss 
  (P : ℝ)
  (initial_marbles remaining_marbles : ℝ)
  (h1 : initial_marbles = 100)
  (h2 : remaining_marbles = 20)
  (h3 : (initial_marbles - initial_marbles * P / 100) / 2 = remaining_marbles) :
  P = 60 :=
by
  sorry

end find_percentage_loss_l242_242190


namespace maximize_expression_l242_242100

theorem maximize_expression
  (a b c : ℝ)
  (h1 : a ≥ 0)
  (h2 : b ≥ 0)
  (h3 : c ≥ 0)
  (h_sum : a + b + c = 3) :
  (a^2 - a * b + b^2) * (a^2 - a * c + c^2) * (b^2 - b * c + c^2) ≤ 729 / 432 := 
sorry

end maximize_expression_l242_242100


namespace carrie_savings_l242_242013

-- Define the original prices and discount rates
def deltaPrice : ℝ := 850
def deltaDiscount : ℝ := 0.20
def unitedPrice : ℝ := 1100
def unitedDiscount : ℝ := 0.30

-- Calculate discounted prices
def deltaDiscountAmount : ℝ := deltaPrice * deltaDiscount
def unitedDiscountAmount : ℝ := unitedPrice * unitedDiscount

def deltaDiscountedPrice : ℝ := deltaPrice - deltaDiscountAmount
def unitedDiscountedPrice : ℝ := unitedPrice - unitedDiscountAmount

-- Define the savings by choosing the cheaper flight
def savingsByChoosingCheaperFlight : ℝ := unitedDiscountedPrice - deltaDiscountedPrice

-- The theorem stating the amount saved
theorem carrie_savings : savingsByChoosingCheaperFlight = 90 :=
by
  sorry

end carrie_savings_l242_242013


namespace valid_interval_for_k_l242_242414

theorem valid_interval_for_k :
  ∀ k : ℝ, (∀ x : ℝ, x^2 - 8*x + k < 0 → 0 < k ∧ k < 16) :=
by
  sorry

end valid_interval_for_k_l242_242414


namespace jori_water_left_l242_242545

theorem jori_water_left (initial_gallons used_gallons : ℚ) (h1 : initial_gallons = 3) (h2 : used_gallons = 11 / 4) :
  initial_gallons - used_gallons = 1 / 4 :=
by
  sorry

end jori_water_left_l242_242545


namespace num_students_is_92_l242_242736

noncomputable def total_students (S : ℕ) : Prop :=
  let remaining := S - 20
  let biking := (5/8 : ℚ) * remaining
  let walking := (3/8 : ℚ) * remaining
  walking = 27

theorem num_students_is_92 : total_students 92 :=
by
  let remaining := 92 - 20
  let biking := (5/8 : ℚ) * remaining
  let walking := (3/8 : ℚ) * remaining
  have walk_eq : walking = 27 := by sorry
  exact walk_eq

end num_students_is_92_l242_242736


namespace find_t_l242_242825

variable (a t : ℝ)

def f (x : ℝ) : ℝ := a * x + 19

theorem find_t (h1 : f a 3 = 7) (h2 : f a t = 15) : t = 1 :=
by
  sorry

end find_t_l242_242825


namespace solve_for_a_l242_242450

def star (a b : ℤ) : ℤ := 3 * a - b^3

theorem solve_for_a (a : ℤ) : star a 3 = 18 → a = 15 := by
  intro h₁
  sorry

end solve_for_a_l242_242450


namespace correct_decimal_product_l242_242339

theorem correct_decimal_product : (0.125 * 3.2 = 4.0) :=
sorry

end correct_decimal_product_l242_242339


namespace evaluate_expression_l242_242777

-- Definition of the conditions
def a : ℕ := 15
def b : ℕ := 19
def c : ℕ := 13

-- Problem statement
theorem evaluate_expression :
  (225 * (1 / a - 1 / b) + 361 * (1 / b - 1 / c) + 169 * (1 / c - 1 / a))
  /
  (a * (1 / b - 1 / c) + b * (1 / c - 1 / a) + c * (1 / a - 1 / b)) = a + b + c :=
by
  sorry

end evaluate_expression_l242_242777


namespace nat_pow_eq_iff_divides_l242_242617

theorem nat_pow_eq_iff_divides (a b n : ℕ) (h : ∀ k : ℕ, k ≠ b → (b - k) ∣ (a - k^n)) : a = b^n :=
sorry

end nat_pow_eq_iff_divides_l242_242617


namespace find_a9_l242_242732

variable {a : ℕ → ℝ} 
variable {q : ℝ}

-- Conditions
def geom_seq (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

def a3_eq_1 (a : ℕ → ℝ) : Prop := 
  a 3 = 1

def a5_a6_a7_eq_8 (a : ℕ → ℝ) : Prop := 
  a 5 * a 6 * a 7 = 8

-- Theorem to prove
theorem find_a9 {a : ℕ → ℝ} {q : ℝ} 
  (geom : geom_seq a q)
  (ha3 : a3_eq_1 a)
  (ha5a6a7 : a5_a6_a7_eq_8 a) : a 9 = 4 := 
sorry

end find_a9_l242_242732


namespace work_completion_l242_242661

theorem work_completion (A B C : ℚ) (hA : A = 1/21) (hB : B = 1/6) 
    (hCombined : A + B + C = 1/3.36) : C = 1/12 := by
  sorry

end work_completion_l242_242661


namespace wire_ratio_bonnie_roark_l242_242349

-- Definitions from the conditions
def bonnie_wire_length : ℕ := 12 * 8
def bonnie_volume : ℕ := 8 ^ 3
def roark_cube_side : ℕ := 2
def roark_cube_volume : ℕ := roark_cube_side ^ 3
def num_roark_cubes : ℕ := bonnie_volume / roark_cube_volume
def roark_wire_length_per_cube : ℕ := 12 * roark_cube_side
def roark_total_wire_length : ℕ := num_roark_cubes * roark_wire_length_per_cube

-- Statement to prove
theorem wire_ratio_bonnie_roark : 
  ((bonnie_wire_length : ℚ) / roark_total_wire_length) = (1 / 16) :=
by
  sorry

end wire_ratio_bonnie_roark_l242_242349


namespace pages_remaining_total_l242_242317

-- Define the conditions
def total_pages_book1 : ℕ := 563
def read_pages_book1 : ℕ := 147

def total_pages_book2 : ℕ := 849
def read_pages_book2 : ℕ := 389

def total_pages_book3 : ℕ := 700
def read_pages_book3 : ℕ := 134

-- The theorem to be proved
theorem pages_remaining_total :
  (total_pages_book1 - read_pages_book1) + 
  (total_pages_book2 - read_pages_book2) + 
  (total_pages_book3 - read_pages_book3) = 1442 := 
by
  sorry

end pages_remaining_total_l242_242317


namespace number_subtracted_from_15n_l242_242391

theorem number_subtracted_from_15n (m n : ℕ) (h_pos_n : 0 < n) (h_pos_m : 0 < m) (h_eq : m = 15 * n - 1) (h_remainder : m % 5 = 4) : 1 = 1 :=
by
  sorry

end number_subtracted_from_15n_l242_242391


namespace stratified_sampling_medium_supermarkets_l242_242605

theorem stratified_sampling_medium_supermarkets
  (large_supermarkets : ℕ)
  (medium_supermarkets : ℕ)
  (small_supermarkets : ℕ)
  (sample_size : ℕ)
  (total_supermarkets : ℕ)
  (medium_proportion : ℚ) :
  large_supermarkets = 200 →
  medium_supermarkets = 400 →
  small_supermarkets = 1400 →
  sample_size = 100 →
  total_supermarkets = large_supermarkets + medium_supermarkets + small_supermarkets →
  medium_proportion = (medium_supermarkets : ℚ) / (total_supermarkets : ℚ) →
  medium_supermarkets_to_sample = sample_size * medium_proportion →
  medium_supermarkets_to_sample = 20 :=
sorry

end stratified_sampling_medium_supermarkets_l242_242605


namespace find_angle_B_l242_242711

-- Conditions
variable (A B C a b : ℝ)
variable (h1 : a = Real.sqrt 6)
variable (h2 : b = Real.sqrt 3)
variable (h3 : b + a * (Real.sin C - Real.cos C) = 0)

-- Target
theorem find_angle_B : B = Real.pi / 6 :=
sorry

end find_angle_B_l242_242711


namespace finite_odd_divisors_condition_l242_242519

theorem finite_odd_divisors_condition (k : ℕ) (hk : 0 < k) :
  (∃ N : ℕ, ∀ n : ℕ, n > N → ¬ (n % 2 = 1 ∧ n ∣ k^n + 1)) ↔ (∃ c : ℕ, k + 1 = 2^c) :=
by sorry

end finite_odd_divisors_condition_l242_242519


namespace solve_for_a_l242_242416

noncomputable def parabola (a b c : ℚ) (x : ℚ) := a * x^2 + b * x + c

theorem solve_for_a (a b c : ℚ) (h1 : parabola a b c 2 = 5) (h2 : parabola a b c 1 = 2) : 
  a = -3 :=
by
  -- Given: y = ax^2 + bx + c with vertex (2,5) and point (1,2)
  have eq1 : a * (2:ℚ)^2 + b * (2:ℚ) + c = 5 := h1
  have eq2 : a * (1:ℚ)^2 + b * (1:ℚ) + c = 2 := h2

  -- Combine information to find a
  sorry

end solve_for_a_l242_242416


namespace problem_statement_l242_242037

theorem problem_statement (m : ℝ) : 
  (¬ (∃ x : ℝ, x^2 + 2 * x + m ≤ 0)) → m > 1 :=
by
  sorry

end problem_statement_l242_242037


namespace find_intercept_l242_242533

theorem find_intercept (avg_height : ℝ) (avg_shoe_size : ℝ) (a : ℝ)
  (h1 : avg_height = 170)
  (h2 : avg_shoe_size = 40) 
  (h3 : 3 * avg_shoe_size + a = avg_height) : a = 50 := 
by
  sorry

end find_intercept_l242_242533


namespace compute_fraction_mul_l242_242785

theorem compute_fraction_mul :
  (1 / 3) ^ 2 * (1 / 8) = 1 / 72 :=
by
  sorry

end compute_fraction_mul_l242_242785


namespace another_divisor_l242_242239

theorem another_divisor (n : ℕ) (h1 : n = 44402) (h2 : ∀ d ∈ [12, 48, 74, 100], (n + 2) % d = 0) : 
  199 ∣ (n + 2) := 
by 
  sorry

end another_divisor_l242_242239


namespace ratio_of_ages_l242_242344

variable (F S : ℕ)

-- Condition 1: The product of father's age and son's age is 756
def cond1 := F * S = 756

-- Condition 2: The ratio of their ages after 6 years will be 2
def cond2 := (F + 6) / (S + 6) = 2

-- Theorem statement: The current ratio of the father's age to the son's age is 7:3
theorem ratio_of_ages (h1 : cond1 F S) (h2 : cond2 F S) : F / S = 7 / 3 :=
sorry

end ratio_of_ages_l242_242344


namespace meals_without_restrictions_l242_242731

theorem meals_without_restrictions (total_clients vegan kosher gluten_free halal dairy_free nut_free vegan_kosher vegan_gluten_free kosher_gluten_free halal_dairy_free gluten_free_nut_free vegan_halal_gluten_free kosher_dairy_free_nut_free : ℕ) 
  (h_tc : total_clients = 80)
  (h_vegan : vegan = 15)
  (h_kosher : kosher = 18)
  (h_gluten_free : gluten_free = 12)
  (h_halal : halal = 10)
  (h_dairy_free : dairy_free = 8)
  (h_nut_free : nut_free = 4)
  (h_vegan_kosher : vegan_kosher = 5)
  (h_vegan_gluten_free : vegan_gluten_free = 6)
  (h_kosher_gluten_free : kosher_gluten_free = 3)
  (h_halal_dairy_free : halal_dairy_free = 4)
  (h_gluten_free_nut_free : gluten_free_nut_free = 2)
  (h_vegan_halal_gluten_free : vegan_halal_gluten_free = 2)
  (h_kosher_dairy_free_nut_free : kosher_dairy_free_nut_free = 1) : 
  (total_clients - (vegan + kosher + gluten_free + halal + dairy_free + nut_free 
  - vegan_kosher - vegan_gluten_free - kosher_gluten_free - halal_dairy_free - gluten_free_nut_free 
  + vegan_halal_gluten_free + kosher_dairy_free_nut_free) = 30) :=
by {
  -- solution steps here
  sorry
}

end meals_without_restrictions_l242_242731


namespace arc_length_parametric_curve_l242_242149

noncomputable def arcLength (x y : ℝ → ℝ) (t1 t2 : ℝ) : ℝ :=
  ∫ t in t1..t2, Real.sqrt ((deriv x t)^2 + (deriv y t)^2)

theorem arc_length_parametric_curve :
    (∫ t in (0 : ℝ)..(3 * Real.pi), 
        Real.sqrt ((deriv (fun t => (t ^ 2 - 2) * Real.sin t + 2 * t * Real.cos t) t) ^ 2 +
                   (deriv (fun t => (2 - t ^ 2) * Real.cos t + 2 * t * Real.sin t) t) ^ 2)) =
    9 * Real.pi ^ 3 :=
by
  -- The proof is omitted
  sorry

end arc_length_parametric_curve_l242_242149


namespace sum_of_series_eq_one_third_l242_242648

theorem sum_of_series_eq_one_third :
  ∑' k : ℕ, (2^k / (8^k - 1)) = 1 / 3 :=
sorry

end sum_of_series_eq_one_third_l242_242648


namespace men_earnings_l242_242321

-- Definitions based on given problem conditions
variables (M rm W rw B rb X : ℝ)
variables (h1 : 5 > 0) (h2 : X > 0) (h3 : 8 > 0) -- positive quantities
variables (total_earnings : 5 * M * rm + X * W * rw + 8 * B * rb = 180)

-- The theorem we want to prove
theorem men_earnings (h1 : 5 > 0) (h2 : X > 0) (h3 : 8 > 0) (total_earnings : 5 * M * rm + X * W * rw + 8 * B * rb = 180) : 
  ∃ men_earnings : ℝ, men_earnings = 5 * M * rm :=
by 
  -- Proof is omitted
  exact Exists.intro (5 * M * rm) rfl

end men_earnings_l242_242321


namespace tuna_per_customer_l242_242812

noncomputable def total_customers := 100
noncomputable def total_tuna := 10
noncomputable def weight_per_tuna := 200
noncomputable def customers_without_fish := 20

theorem tuna_per_customer : (total_tuna * weight_per_tuna) / (total_customers - customers_without_fish) = 25 := by
  sorry

end tuna_per_customer_l242_242812


namespace distance_to_city_l242_242115

variable (d : ℝ)  -- Define d as a real number

theorem distance_to_city (h1 : ¬ (d ≥ 13)) (h2 : ¬ (d ≤ 10)) :
  10 < d ∧ d < 13 :=
by
  -- Here we will formalize the proof in Lean syntax
  sorry

end distance_to_city_l242_242115


namespace eccentricity_of_hyperbola_l242_242593

theorem eccentricity_of_hyperbola (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) (h_asymp : 3 * a + b = 0) :
    let c := Real.sqrt (a^2 + b^2)
    let e := c / a
    e = Real.sqrt 10 :=
by
  sorry

end eccentricity_of_hyperbola_l242_242593


namespace units_digit_of_power_ends_in_nine_l242_242609

theorem units_digit_of_power_ends_in_nine (n : ℕ) (h : (3^n) % 10 = 9) : n % 4 = 2 :=
sorry

end units_digit_of_power_ends_in_nine_l242_242609


namespace initial_number_proof_l242_242644

def initial_number : ℕ := 7899665
def result : ℕ := 7899593
def factor1 : ℕ := 12
def factor2 : ℕ := 3
def factor3 : ℕ := 2

def certain_value : ℕ := (factor1 * factor2) * factor3

theorem initial_number_proof :
  initial_number - certain_value = result := by
  sorry

end initial_number_proof_l242_242644


namespace brainiacs_like_both_l242_242210

theorem brainiacs_like_both
  (R M B : ℕ)
  (h1 : R = 2 * M)
  (h2 : R + M - B = 96)
  (h3 : M - B = 20) : B = 18 := by
  sorry

end brainiacs_like_both_l242_242210


namespace railway_original_stations_l242_242420

theorem railway_original_stations (m n : ℕ) (hn : n > 1) (h : n * (2 * m - 1 + n) = 58) : m = 14 :=
by
  sorry

end railway_original_stations_l242_242420


namespace smallest_value_A_plus_B_plus_C_plus_D_l242_242651

variable (A B C D : ℤ)

-- Given conditions in Lean statement form
def isArithmeticSequence (A B C : ℤ) : Prop :=
  B - A = C - B

def isGeometricSequence (B C D : ℤ) : Prop :=
  (C / B : ℚ) = 4 / 3 ∧ (D / C : ℚ) = C / B

def givenConditions (A B C D : ℤ) : Prop :=
  isArithmeticSequence A B C ∧ isGeometricSequence B C D

-- The proof problem to validate the smallest possible value
theorem smallest_value_A_plus_B_plus_C_plus_D (h : givenConditions A B C D) :
  A + B + C + D = 43 :=
sorry

end smallest_value_A_plus_B_plus_C_plus_D_l242_242651


namespace max_value_of_function_neg_x_l242_242130

theorem max_value_of_function_neg_x (x : ℝ) (h : x < 0) : 
  ∃ y, (y = 2 * x + 2 / x) ∧ y ≤ -4 := sorry

end max_value_of_function_neg_x_l242_242130


namespace area_square_II_l242_242243

theorem area_square_II (a b : ℝ) :
  let diag_I := 2 * (a + b)
  let area_I := (a + b) * (a + b) * 2
  let area_II := area_I * 3
  area_II = 6 * (a + b) ^ 2 :=
by
  sorry

end area_square_II_l242_242243


namespace annie_has_12_brownies_left_l242_242597

noncomputable def initial_brownies := 100
noncomputable def portion_for_admin := (3 / 5 : ℚ) * initial_brownies
noncomputable def leftover_after_admin := initial_brownies - portion_for_admin
noncomputable def portion_for_carl := (1 / 4 : ℚ) * leftover_after_admin
noncomputable def leftover_after_carl := leftover_after_admin - portion_for_carl
noncomputable def portion_for_simon := 3
noncomputable def leftover_after_simon := leftover_after_carl - portion_for_simon
noncomputable def portion_for_friends := (2 / 3 : ℚ) * leftover_after_simon
noncomputable def each_friend_get := portion_for_friends / 5
noncomputable def total_given_to_friends := each_friend_get * 5
noncomputable def final_brownies := leftover_after_simon - total_given_to_friends

theorem annie_has_12_brownies_left : final_brownies = 12 := by
  sorry

end annie_has_12_brownies_left_l242_242597


namespace weight_of_new_student_l242_242096

theorem weight_of_new_student (W x y z : ℝ) (h : (W - x - y + z = W - 40)) : z = 40 - (x + y) :=
by
  sorry

end weight_of_new_student_l242_242096


namespace average_salary_of_technicians_l242_242012

theorem average_salary_of_technicians
  (total_workers : ℕ)
  (average_salary_all : ℕ)
  (average_salary_non_technicians : ℕ)
  (num_technicians : ℕ)
  (num_non_technicians : ℕ)
  (h1 : total_workers = 21)
  (h2 : average_salary_all = 8000)
  (h3 : average_salary_non_technicians = 6000)
  (h4 : num_technicians = 7)
  (h5 : num_non_technicians = 14) :
  (average_salary_all * total_workers - average_salary_non_technicians * num_non_technicians) / num_technicians = 12000 :=
by
  sorry

end average_salary_of_technicians_l242_242012


namespace find_x_squared_plus_y_squared_l242_242107

variables (x y : ℝ)

theorem find_x_squared_plus_y_squared (h1 : x - y = 20) (h2 : x * y = 9) :
  x^2 + y^2 = 418 :=
sorry

end find_x_squared_plus_y_squared_l242_242107


namespace pieces_per_box_l242_242098

theorem pieces_per_box (total_pieces : ℕ) (boxes : ℕ) (h_total : total_pieces = 3000) (h_boxes : boxes = 6) :
  total_pieces / boxes = 500 := by
  sorry

end pieces_per_box_l242_242098


namespace small_pizza_slices_correct_l242_242779

-- Defining the total number of people involved
def people_count : ℕ := 3

-- Defining the number of slices each person can eat
def slices_per_person : ℕ := 12

-- Calculating the total number of slices needed based on the number of people and slices per person
def total_slices_needed : ℕ := people_count * slices_per_person

-- Defining the number of slices in a large pizza
def large_pizza_slices : ℕ := 14

-- Defining the number of large pizzas ordered
def large_pizzas_count : ℕ := 2

-- Calculating the total number of slices provided by the large pizzas
def total_large_pizza_slices : ℕ := large_pizza_slices * large_pizzas_count

-- Defining the number of slices in a small pizza
def small_pizza_slices : ℕ := 8

-- Total number of slices provided needs to be at least the total slices needed
theorem small_pizza_slices_correct :
  total_slices_needed ≤ total_large_pizza_slices + small_pizza_slices := by
  sorry

end small_pizza_slices_correct_l242_242779


namespace range_of_f_l242_242549

noncomputable def f (x : ℝ) : ℝ := (1/3) ^ (x^2 - 2*x)

theorem range_of_f : Set.Ioo 0 3 ∪ {3} = { y | ∃ x, f x = y } :=
by
  sorry

end range_of_f_l242_242549


namespace eval_p_positive_int_l242_242144

theorem eval_p_positive_int (p : ℕ) : 
  (∃ n : ℕ, n > 0 ∧ (4 * p + 20) = n * (3 * p - 6)) ↔ p = 3 ∨ p = 4 ∨ p = 15 ∨ p = 28 := 
by sorry

end eval_p_positive_int_l242_242144


namespace arithmetic_seq_max_n_l242_242614

def arithmetic_seq_max_sum (a : ℕ → ℝ) (d : ℝ) (S : ℕ → ℝ) : Prop :=
  (a 1 > 0) ∧ (3 * (a 1 + 4 * d) = 5 * (a 1 + 7 * d)) ∧
  (∀ n, S n = n * a 1 + (n * (n - 1) / 2) * d) ∧
  (S 12 = -72 * d)

theorem arithmetic_seq_max_n
  (a : ℕ → ℝ) (d : ℝ) (S : ℕ → ℝ) : 
  arithmetic_seq_max_sum a d S → n = 12 :=
by
  sorry

end arithmetic_seq_max_n_l242_242614


namespace find_k_l242_242373

theorem find_k (k : ℝ) (A B : ℝ × ℝ) 
  (hA : A = (2, 3)) (hB : B = (4, k)) 
  (hAB_parallel : A.2 = B.2) : k = 3 := 
by 
  have hA_def : A = (2, 3) := hA 
  have hB_def : B = (4, k) := hB 
  have parallel_condition: A.2 = B.2 := hAB_parallel
  simp at parallel_condition
  sorry

end find_k_l242_242373


namespace unique_positive_b_for_discriminant_zero_l242_242041

theorem unique_positive_b_for_discriminant_zero (c : ℝ) : 
  (∃! b : ℝ, b > 0 ∧ (b^2 + 1/b^2)^2 - 4 * c = 0) → c = 1 :=
by
  sorry

end unique_positive_b_for_discriminant_zero_l242_242041


namespace find_n_l242_242543

theorem find_n (n : ℤ) (h : (1 : ℤ)^2 + 3 * 1 + n = 0) : n = -4 :=
sorry

end find_n_l242_242543


namespace transformation_maps_segment_l242_242169

variables (C D : ℝ × ℝ) (C' D' : ℝ × ℝ)

def reflect_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

theorem transformation_maps_segment :
  reflect_x (reflect_y (3, -2)) = (-3, 2) ∧ reflect_x (reflect_y (4, -5)) = (-4, 5) :=
by {
  sorry
}

end transformation_maps_segment_l242_242169


namespace probability_getting_wet_l242_242003

theorem probability_getting_wet 
  (P_R : ℝ := 1/2)
  (P_notT : ℝ := 1/2)
  (h1 : 0 ≤ P_R ∧ P_R ≤ 1)
  (h2 : 0 ≤ P_notT ∧ P_notT ≤ 1) 
  : P_R * P_notT = 1/4 := 
by
  -- Proof that the probability of getting wet equals 1/4
  sorry

end probability_getting_wet_l242_242003


namespace sin_cos_from_tan_in_second_quadrant_l242_242163

theorem sin_cos_from_tan_in_second_quadrant (α : ℝ) 
  (h1 : Real.tan α = -2) 
  (h2 : α ∈ Set.Ioo (π / 2) π) : 
  Real.sin α = 2 * Real.sqrt 5 / 5 ∧ Real.cos α = -Real.sqrt 5 / 5 :=
by
  sorry

end sin_cos_from_tan_in_second_quadrant_l242_242163


namespace ten_digit_number_contains_repeated_digit_l242_242490

open Nat

theorem ten_digit_number_contains_repeated_digit
  (n : ℕ)
  (h1 : 10^9 ≤ n^2 + 1)
  (h2 : n^2 + 1 < 10^10) :
  ∃ d1 d2 : ℕ, d1 ≠ d2 ∧ (d1 ∈ (digits 10 (n^2 + 1))) ∧ (d2 ∈ (digits 10 (n^2 + 1))) :=
sorry

end ten_digit_number_contains_repeated_digit_l242_242490


namespace option_C_is_quadratic_l242_242442

-- Definitions based on conditions
def option_A (x : ℝ) : Prop := x^2 + (1/x^2) = 0
def option_B (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0
def option_C (x : ℝ) : Prop := (x - 1) * (x + 2) = 1
def option_D (x y : ℝ) : Prop := 3 * x^2 - 2 * x * y - 5 * y^2 = 0

-- Statement to prove option C is a quadratic equation in one variable.
theorem option_C_is_quadratic : ∀ x : ℝ, (option_C x) → (∃ a b c : ℝ, a ≠ 0 ∧ a * x^2 + b * x + c = 0) :=
by
  intros x hx
  -- To be proven
  sorry

end option_C_is_quadratic_l242_242442


namespace problem_example_l242_242643

theorem problem_example (a : ℕ) (H1 : a ∈ ({a, b, c} : Set ℕ)) (H2 : 0 ∈ ({x | x^2 ≠ 0} : Set ℕ)) :
  a ∈ ({a, b, c} : Set ℕ) ∧ 0 ∈ ({x | x^2 ≠ 0} : Set ℕ) :=
by
  sorry

end problem_example_l242_242643


namespace sin_double_angle_eq_half_l242_242216

theorem sin_double_angle_eq_half (α : ℝ) (h1 : 0 < α ∧ α < π / 2)
  (h2 : Real.sin (π / 2 + 2 * α) = Real.cos (π / 4 - α)) : 
  Real.sin (2 * α) = 1 / 2 :=
by
  sorry

end sin_double_angle_eq_half_l242_242216


namespace min_value_of_f_l242_242233

noncomputable def f (x a : ℝ) := Real.exp (x - a) - Real.log (x + a) - 1

theorem min_value_of_f (a : ℝ) : 
  (0 < a) → (∃ x : ℝ, f x a = 0) ↔ a = 1 / 2 :=
by
  sorry

end min_value_of_f_l242_242233


namespace parabola_through_P_l242_242015

-- Define the point P
def P : ℝ × ℝ := (4, -2)

-- Define a condition function for equations y^2 = a*x
def satisfies_y_eq_ax (a : ℝ) : Prop := 
  ∃ x y, (x, y) = P ∧ y^2 = a * x

-- Define a condition function for equations x^2 = b*y
def satisfies_x_eq_by (b : ℝ) : Prop := 
  ∃ x y, (x, y) = P ∧ x^2 = b * y

-- Lean's theorem statement
theorem parabola_through_P : satisfies_y_eq_ax 1 ∨ satisfies_x_eq_by (-8) :=
sorry

end parabola_through_P_l242_242015


namespace modulus_z_eq_one_l242_242722

noncomputable def imaginary_unit : ℂ := Complex.I

noncomputable def z : ℂ := (1 - imaginary_unit) / (1 + imaginary_unit) 

theorem modulus_z_eq_one : Complex.abs z = 1 := 
sorry

end modulus_z_eq_one_l242_242722


namespace train_speed_in_km_per_hr_l242_242091

/-- Given the length of a train and a bridge, and the time taken for the train to cross the bridge, prove the speed of the train in km/hr -/
theorem train_speed_in_km_per_hr
  (train_length : ℕ)  -- 100 meters
  (bridge_length : ℕ) -- 275 meters
  (crossing_time : ℕ) -- 30 seconds
  (conversion_factor : ℝ) -- 1 m/s = 3.6 km/hr
  (h_train_length : train_length = 100)
  (h_bridge_length : bridge_length = 275)
  (h_crossing_time : crossing_time = 30)
  (h_conversion_factor : conversion_factor = 3.6) : 
  (train_length + bridge_length) / crossing_time * conversion_factor = 45 := 
sorry

end train_speed_in_km_per_hr_l242_242091


namespace cars_with_neither_feature_l242_242403

theorem cars_with_neither_feature 
  (total_cars : ℕ) 
  (power_steering : ℕ) 
  (power_windows : ℕ) 
  (both_features : ℕ) 
  (h1 : total_cars = 65) 
  (h2 : power_steering = 45) 
  (h3 : power_windows = 25) 
  (h4 : both_features = 17)
  : total_cars - (power_steering + power_windows - both_features) = 12 :=
by
  sorry

end cars_with_neither_feature_l242_242403


namespace minimum_value_g_l242_242535

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 
  if a > 1 then 
    a * (-1/a) + 1 
  else 
    if 0 < a then 
      a^2 + 1 
    else 
      0  -- adding a default value to make it computable

theorem minimum_value_g (a : ℝ) (m : ℝ) : 0 < a ∧ a < 2 ∧ ∃ x₀, f x₀ a = m → m ≥ 5 / 2 :=
by
  sorry

end minimum_value_g_l242_242535


namespace simplify_complex_expression_l242_242407

open Complex

theorem simplify_complex_expression :
  let a := (4 : ℂ) + 6 * I
  let b := (4 : ℂ) - 6 * I
  ((a / b) - (b / a) = (24 * I) / 13) := by
  sorry

end simplify_complex_expression_l242_242407


namespace rectangle_probability_l242_242544

theorem rectangle_probability (m n : ℕ) (h_m : m = 1003^2) (h_n : n = 1003 * 2005) :
  (1 - (m / n)) = 1002 / 2005 :=
by
  sorry

end rectangle_probability_l242_242544


namespace time_for_A_and_C_to_complete_work_l242_242501

variable (A_rate B_rate C_rate : ℝ)

theorem time_for_A_and_C_to_complete_work
  (hA : A_rate = 1 / 4)
  (hBC : 1 / 3 = B_rate + C_rate)
  (hB : B_rate = 1 / 12) :
  1 / (A_rate + C_rate) = 2 :=
by
  -- Here would be the proof logic
  sorry

end time_for_A_and_C_to_complete_work_l242_242501


namespace joe_dropped_score_l242_242789

theorem joe_dropped_score (A B C D : ℕ) (h1 : (A + B + C + D) / 4 = 60) (h2 : (A + B + C) / 3 = 65) :
  min A (min B (min C D)) = D → D = 45 :=
by sorry

end joe_dropped_score_l242_242789


namespace correct_translation_of_tradition_l242_242786

def is_adjective (s : String) : Prop :=
  s = "传统的"

def is_correct_translation (s : String) (translation : String) : Prop :=
  s = "传统的" → translation = "traditional"

theorem correct_translation_of_tradition : 
  is_adjective "传统的" ∧ is_correct_translation "传统的" "traditional" :=
by
  sorry

end correct_translation_of_tradition_l242_242786


namespace trader_total_discount_correct_l242_242156

theorem trader_total_discount_correct :
  let CP_A := 200
  let CP_B := 150
  let CP_C := 100
  let MSP_A := CP_A + 0.50 * CP_A
  let MSP_B := CP_B + 0.50 * CP_B
  let MSP_C := CP_C + 0.50 * CP_C
  let SP_A := 0.99 * CP_A
  let SP_B := 0.97 * CP_B
  let SP_C := 0.98 * CP_C
  let discount_A := MSP_A - SP_A
  let discount_B := MSP_B - SP_B
  let discount_C := MSP_C - SP_C
  let total_discount := discount_A + discount_B + discount_C
  total_discount = 233.5 := by sorry

end trader_total_discount_correct_l242_242156


namespace find_side2_l242_242125

-- Define the given conditions
def perimeter : ℕ := 160
def side1 : ℕ := 40
def side3 : ℕ := 70

-- Define the second side as a variable
def side2 : ℕ := perimeter - side1 - side3

-- State the theorem to be proven
theorem find_side2 : side2 = 50 := by
  -- We skip the proof here with sorry
  sorry

end find_side2_l242_242125


namespace coin_flip_probability_l242_242602

/--
Suppose we flip five coins simultaneously: a penny, a nickel, a dime, a quarter, and a half-dollar.
What is the probability that the penny and dime both come up heads, and the half-dollar comes up tails?
-/

theorem coin_flip_probability :
  let outcomes := 2^5
  let success := 1 * 1 * 1 * 2 * 2
  success / outcomes = (1 : ℚ) / 8 :=
by
  /- Proof goes here -/
  sorry

end coin_flip_probability_l242_242602


namespace initial_number_of_girls_l242_242759

theorem initial_number_of_girls (b g : ℕ) (h1 : b = 3 * (g - 20)) (h2 : 7 * (b - 54) = g - 20) : g = 39 :=
sorry

end initial_number_of_girls_l242_242759


namespace find_hourly_rate_l242_242792

theorem find_hourly_rate (x : ℝ) (h1 : 40 * x + 10.75 * 16 = 622) : x = 11.25 :=
sorry

end find_hourly_rate_l242_242792


namespace volume_relation_l242_242132

-- Definitions for points and geometry structures
structure Point3D :=
(x : ℝ) (y : ℝ) (z : ℝ)

structure Tetrahedron :=
(A B C D : Point3D)

-- Volume function for Tetrahedron
noncomputable def volume (t : Tetrahedron) : ℝ := sorry

-- Given conditions
variable {A B C D D1 A1 B1 C1 : Point3D} 

-- D_1 is the centroid of triangle ABC
axiom centroid_D1 (A B C D1 : Point3D) : D1 = Point3D.mk ((A.x + B.x + C.x) / 3) ((A.y + B.y + C.y) / 3) ((A.z + B.z + C.z) / 3)

-- Line through A parallel to DD_1 intersects plane BCD at A1
axiom A1_condition (A B C D D1 A1 : Point3D) : sorry
-- Line through B parallel to DD_1 intersects plane ACD at B1
axiom B1_condition (A B C D D1 B1 : Point3D) : sorry
-- Line through C parallel to DD_1 intersects plane ABD at C1
axiom C1_condition (A B C D D1 C1 : Point3D) : sorry

-- Volume relation to be proven
theorem volume_relation (t1 t2 : Tetrahedron) (h : t1.A = A ∧ t1.B = B ∧ t1.C = C ∧ t1.D = D ∧
                                                t2.A = A1 ∧ t2.B = B1 ∧ t2.C = C1 ∧ t2.D = D1) :
  volume t1 = 2 * volume t2 := 
sorry

end volume_relation_l242_242132


namespace standard_equation_of_ellipse_l242_242575

theorem standard_equation_of_ellipse :
  ∀ (m n : ℝ), 
    (m > 0 ∧ n > 0) →
    (∃ (c : ℝ), c^2 = m^2 - n^2 ∧ c = 2) →
    (∃ (e : ℝ), e = c / m ∧ e = 1 / 2) →
    (m = 4 ∧ n = 2 * Real.sqrt 3) →
    (∀ x y : ℝ, (x^2 / 16 + y^2 / 12 = 1)) :=
by
  intros m n hmn hc he hm_eq hn_eq
  sorry

end standard_equation_of_ellipse_l242_242575


namespace initial_assessed_value_l242_242702

theorem initial_assessed_value (V : ℝ) (tax_rate : ℝ) (new_value : ℝ) (tax_increase : ℝ) 
  (h1 : tax_rate = 0.10) 
  (h2 : new_value = 28000) 
  (h3 : tax_increase = 800) 
  (h4 : tax_rate * new_value = tax_rate * V + tax_increase) : 
  V = 20000 :=
by
  sorry

end initial_assessed_value_l242_242702


namespace john_beats_per_minute_l242_242338

theorem john_beats_per_minute :
  let hours_per_day := 2
  let days := 3
  let total_beats := 72000
  let minutes_per_hour := 60
  total_beats / (days * hours_per_day * minutes_per_hour) = 200 := 
by 
  sorry

end john_beats_per_minute_l242_242338


namespace stephanie_gas_payment_l242_242212

variables (electricity_bill : ℕ) (gas_bill : ℕ) (water_bill : ℕ) (internet_bill : ℕ)
variables (electricity_paid : ℕ) (gas_paid_fraction : ℚ) (water_paid_fraction : ℚ) (internet_paid : ℕ)
variables (additional_gas_payment : ℕ) (remaining_payment : ℕ) (expected_remaining : ℕ)

def stephanie_budget : Prop :=
  electricity_bill = 60 ∧
  electricity_paid = 60 ∧
  gas_bill = 40 ∧
  gas_paid_fraction = 3/4 ∧
  water_bill = 40 ∧
  water_paid_fraction = 1/2 ∧
  internet_bill = 25 ∧
  internet_paid = 4 * 5 ∧
  remaining_payment = 30 ∧
  expected_remaining = 
    (gas_bill - gas_paid_fraction * gas_bill) +
    (water_bill - water_paid_fraction * water_bill) + 
    (internet_bill - internet_paid) - 
    additional_gas_payment ∧
  expected_remaining = remaining_payment

theorem stephanie_gas_payment : additional_gas_payment = 5 :=
by sorry

end stephanie_gas_payment_l242_242212


namespace ascending_order_proof_l242_242417

noncomputable def frac1 : ℚ := 1 / 2
noncomputable def frac2 : ℚ := 3 / 4
noncomputable def frac3 : ℚ := 1 / 5
noncomputable def dec1 : ℚ := 0.25
noncomputable def dec2 : ℚ := 0.42

theorem ascending_order_proof :
  frac3 < dec1 ∧ dec1 < dec2 ∧ dec2 < frac1 ∧ frac1 < frac2 :=
by {
  -- The proof will show the conversions mentioned in solution steps
  sorry
}

end ascending_order_proof_l242_242417


namespace correct_choice_for_games_l242_242821
  
-- Define the problem context
def games_preferred (question : String) (answer : String) :=
  question = "Which of the two computer games did you prefer?" ∧
  answer = "Actually I didn’t like either of them."

-- Define the proof that the correct choice is 'either of them'
theorem correct_choice_for_games (question : String) (answer : String) :
  games_preferred question answer → answer = "either of them" :=
by
  -- Provided statement and proof assumptions
  intro h
  cases h
  exact sorry -- Proof steps will be here
  -- Here, the conclusion should be derived from given conditions

end correct_choice_for_games_l242_242821


namespace perp_bisector_chord_l242_242749

theorem perp_bisector_chord (x y : ℝ) :
  (2 * x + 3 * y + 1 = 0) ∧ (x^2 + y^2 - 2 * x + 4 * y = 0) → 
  ∃ k l m : ℝ, (3 * x - 2 * y - 7 = 0) :=
by
  sorry

end perp_bisector_chord_l242_242749

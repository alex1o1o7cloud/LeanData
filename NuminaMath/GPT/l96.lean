import Mathlib

namespace meaningful_fraction_range_l96_9661

theorem meaningful_fraction_range (x : ℝ) : (3 - x) ≠ 0 ↔ x ≠ 3 :=
by sorry

end meaningful_fraction_range_l96_9661


namespace initial_students_proof_l96_9617

def initial_students (e : ℝ) (transferred : ℝ) (left : ℝ) : ℝ :=
  e + transferred + left

theorem initial_students_proof : initial_students 28 10 4 = 42 :=
  by
    -- This is where the proof would go, but we use 'sorry' to skip it.
    sorry

end initial_students_proof_l96_9617


namespace sum_of_digits_eq_4_l96_9618

theorem sum_of_digits_eq_4 (A B C D X Y : ℕ) (h1 : A + B + C + D = 22) (h2 : B + D = 9) (h3 : X = 1) (h4 : Y = 3) :
    X + Y = 4 :=
by
  sorry

end sum_of_digits_eq_4_l96_9618


namespace pie_eating_fraction_l96_9695

theorem pie_eating_fraction :
  (1 / 3 + 1 / 3^2 + 1 / 3^3 + 1 / 3^4 + 1 / 3^5 + 1 / 3^6 + 1 / 3^7) = 1093 / 2187 := 
sorry

end pie_eating_fraction_l96_9695


namespace train_speed_correct_l96_9620

def train_length : ℝ := 110
def bridge_length : ℝ := 142
def crossing_time : ℝ := 12.598992080633549
def expected_speed : ℝ := 20.002

theorem train_speed_correct :
  (train_length + bridge_length) / crossing_time = expected_speed :=
by
  sorry

end train_speed_correct_l96_9620


namespace cornelia_age_l96_9649

theorem cornelia_age :
  ∃ C : ℕ, 
  (∃ K : ℕ, K = 30 ∧ (C + 20 = 2 * (K + 20))) ∧
  ((K - 5)^2 = 3 * (C - 5)) := by
  sorry

end cornelia_age_l96_9649


namespace a_b_total_money_l96_9680

variable (A B : ℝ)

theorem a_b_total_money (h1 : (4 / 15) * A = (2 / 5) * 484) (h2 : B = 484) : A + B = 1210 := by
  sorry

end a_b_total_money_l96_9680


namespace longer_side_is_40_l96_9657

-- Given the conditions
variable (small_rect_width : ℝ) (small_rect_length : ℝ)
variable (num_rects : ℕ)

-- Conditions 
axiom rect_width_is_10 : small_rect_width = 10
axiom length_is_twice_width : small_rect_length = 2 * small_rect_width
axiom four_rectangles : num_rects = 4

-- Prove length of the longer side of the large rectangle
theorem longer_side_is_40 :
  small_rect_width = 10 → small_rect_length = 2 * small_rect_width → num_rects = 4 →
  (2 * small_rect_length) = 40 := sorry

end longer_side_is_40_l96_9657


namespace prime_geq_7_div_240_l96_9665

theorem prime_geq_7_div_240 (p : ℕ) (hp : Nat.Prime p) (h7 : p ≥ 7) : 240 ∣ p^4 - 1 :=
sorry

end prime_geq_7_div_240_l96_9665


namespace molly_total_swim_l96_9628

variable (meters_saturday : ℕ) (meters_sunday : ℕ)

theorem molly_total_swim (h1 : meters_saturday = 45) (h2 : meters_sunday = 28) : meters_saturday + meters_sunday = 73 := by
  sorry

end molly_total_swim_l96_9628


namespace binomial_22_5_computation_l96_9601

theorem binomial_22_5_computation (h1 : Nat.choose 20 3 = 1140) (h2 : Nat.choose 20 4 = 4845) (h3 : Nat.choose 20 5 = 15504) :
    Nat.choose 22 5 = 26334 := by
  sorry

end binomial_22_5_computation_l96_9601


namespace find_m_n_value_l96_9600

theorem find_m_n_value (x m n : ℝ) 
  (h1 : x - 3 * m < 0) 
  (h2 : n - 2 * x < 0) 
  (h3 : -1 < x)
  (h4 : x < 3) 
  : (m + n) ^ 2023 = -1 :=
sorry

end find_m_n_value_l96_9600


namespace sin_315_eq_neg_sqrt_2_div_2_l96_9689

theorem sin_315_eq_neg_sqrt_2_div_2 : Real.sin (315 * (Real.pi / 180)) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_315_eq_neg_sqrt_2_div_2_l96_9689


namespace pos_diff_of_solutions_abs_eq_20_l96_9622

theorem pos_diff_of_solutions_abs_eq_20 : ∀ (x1 x2 : ℝ), (|x1 + 5| = 20 ∧ |x2 + 5| = 20) → x1 - x2 = 40 :=
  by
    intros x1 x2 h
    sorry

end pos_diff_of_solutions_abs_eq_20_l96_9622


namespace pain_subsided_days_l96_9608

-- Define the problem conditions in Lean
variable (x : ℕ) -- the number of days it takes for the pain to subside

-- Condition 1: The injury takes 5 times the pain subsiding period to fully heal
def injury_healing_days := 5 * x

-- Condition 2: James waits an additional 3 days after the injury is fully healed
def workout_waiting_days := injury_healing_days + 3

-- Condition 3: James waits another 3 weeks (21 days) before lifting heavy
def total_days_until_lifting_heavy := workout_waiting_days + 21

-- Given the total days until James can lift heavy is 39 days, prove x = 3
theorem pain_subsided_days : 
    total_days_until_lifting_heavy x = 39 → x = 3 := by
  sorry

end pain_subsided_days_l96_9608


namespace camera_value_l96_9664

variables (V : ℝ)

def rental_fee_per_week (V : ℝ) := 0.1 * V
def total_rental_fee(V : ℝ) := 4 * rental_fee_per_week V
def johns_share_of_fee(V : ℝ) := 0.6 * (0.4 * total_rental_fee V)

theorem camera_value (h : johns_share_of_fee V = 1200): 
  V = 5000 :=
by
  sorry

end camera_value_l96_9664


namespace linear_if_abs_k_eq_1_l96_9684

theorem linear_if_abs_k_eq_1 (k : ℤ) : |k| = 1 ↔ (k = 1 ∨ k = -1) := by
  sorry

end linear_if_abs_k_eq_1_l96_9684


namespace find_initial_balance_l96_9606

-- Define the initial balance (X)
def initial_balance (X : ℝ) := 
  ∃ (X : ℝ), (X / 2 + 30 + 50 - 20 = 160)

theorem find_initial_balance (X : ℝ) (h : initial_balance X) : 
  X = 200 :=
sorry

end find_initial_balance_l96_9606


namespace average_sitting_time_per_student_l96_9603

def total_travel_time_in_minutes : ℕ := 152
def number_of_seats : ℕ := 5
def number_of_students : ℕ := 8

theorem average_sitting_time_per_student :
  (total_travel_time_in_minutes * number_of_seats) / number_of_students = 95 := 
by
  sorry

end average_sitting_time_per_student_l96_9603


namespace weight_of_each_bag_of_food_l96_9639

theorem weight_of_each_bag_of_food
  (horses : ℕ)
  (feedings_per_day : ℕ)
  (pounds_per_feeding : ℕ)
  (days : ℕ)
  (bags : ℕ)
  (total_food_in_pounds : ℕ)
  (h1 : horses = 25)
  (h2 : feedings_per_day = 2)
  (h3 : pounds_per_feeding = 20)
  (h4 : days = 60)
  (h5 : bags = 60)
  (h6 : total_food_in_pounds = horses * (feedings_per_day * pounds_per_feeding) * days) :
  total_food_in_pounds / bags = 1000 :=
by
  sorry

end weight_of_each_bag_of_food_l96_9639


namespace evaluate_f_a_plus_1_l96_9653

variable (a : ℝ)  -- The variable a is a real number.

def f (x : ℝ) : ℝ := x^2 + 1  -- The function f is defined as x^2 + 1.

theorem evaluate_f_a_plus_1 : f (a + 1) = a^2 + 2 * a + 2 := by
  -- Provide the proof here
  sorry

end evaluate_f_a_plus_1_l96_9653


namespace solution_set_l96_9636

noncomputable def domain := Set.Ioo (-(Real.pi / 2)) (Real.pi / 2)
def f : ℝ → ℝ := sorry
axiom f_domain : ∀ x, x ∈ domain → x ∈ Set.Ioo (-(Real.pi / 2)) (Real.pi / 2)
axiom f_odd : ∀ x, f x + f (-x) = 0
def f' : ℝ → ℝ := sorry
axiom derivative_condition : ∀ x, 0 < x ∧ x < Real.pi / 2 → f' x * Real.cos x + f x * Real.sin x < 0

theorem solution_set :
  {x | f x < Real.sqrt 2 * f (Real.pi / 4) * Real.cos x} = {x | Real.pi / 4 < x ∧ x < Real.pi / 2} :=
sorry

end solution_set_l96_9636


namespace cube_difference_l96_9633

theorem cube_difference (x y : ℕ) (h₁ : x + y = 64) (h₂ : x - y = 16) : x^3 - y^3 = 50176 := by
  sorry

end cube_difference_l96_9633


namespace total_cost_fencing_l96_9650

-- Define the conditions
def length : ℝ := 75
def breadth : ℝ := 25
def cost_per_meter : ℝ := 26.50

-- Define the perimeter of the rectangular plot
def perimeter : ℝ := 2 * length + 2 * breadth

-- Define the total cost of fencing
def total_cost : ℝ := perimeter * cost_per_meter

-- The theorem statement
theorem total_cost_fencing : total_cost = 5300 := 
by 
  -- This is the statement we want to prove
  sorry

end total_cost_fencing_l96_9650


namespace number_of_pupils_l96_9660

theorem number_of_pupils
  (pupil_mark_wrong : ℕ)
  (pupil_mark_correct : ℕ)
  (average_increase : ℚ)
  (n : ℕ)
  (h1 : pupil_mark_wrong = 73)
  (h2 : pupil_mark_correct = 45)
  (h3 : average_increase = 1/2)
  (h4 : 28 / n = average_increase) : n = 56 := 
sorry

end number_of_pupils_l96_9660


namespace prove_n_eq_1_l96_9663

-- Definitions of the given conditions
def is_prime (x : ℕ) : Prop := Nat.Prime x

variable {p q r n : ℕ}
variable (hp : is_prime p) (hq : is_prime q) (hr : is_prime r)
variable (hn_pos : n > 0)
variable (h_eq : p^n + q^n = r^2)

-- Statement to prove
theorem prove_n_eq_1 : n = 1 :=
  sorry

end prove_n_eq_1_l96_9663


namespace number_of_red_dresses_l96_9627

-- Define context for Jane's dress shop problem
def dresses_problem (R B : Nat) : Prop :=
  R + B = 200 ∧ B = R + 34

-- Prove that the number of red dresses (R) should be 83
theorem number_of_red_dresses : ∃ R B : Nat, dresses_problem R B ∧ R = 83 :=
by
  sorry

end number_of_red_dresses_l96_9627


namespace cost_of_five_plastic_chairs_l96_9654

theorem cost_of_five_plastic_chairs (C T : ℕ) (h1 : 3 * C = T) (h2 : T + 2 * C = 55) : 5 * C = 55 :=
by {
  sorry
}

end cost_of_five_plastic_chairs_l96_9654


namespace derivative_at_pi_over_4_l96_9648

noncomputable def f (x : ℝ) : ℝ := x * Real.sin x

theorem derivative_at_pi_over_4 :
  deriv f (π / 4) = (Real.sqrt 2 / 2) + (Real.sqrt 2 * π / 8) :=
by
  -- Since the focus is only on the statement, the proof is not required.
  sorry

end derivative_at_pi_over_4_l96_9648


namespace volume_of_rectangular_prism_l96_9696

-- Given conditions translated into Lean definitions
variables (AB AD AC1 AA1 : ℕ)

def rectangular_prism_properties : Prop :=
  AB = 2 ∧ AD = 2 ∧ AC1 = 3 ∧ AA1 = 1

-- The mathematical volume of the rectangular prism
def volume (AB AD AA1 : ℕ) := AB * AD * AA1

-- Prove that given the conditions, the volume of the rectangular prism is 4
theorem volume_of_rectangular_prism (h : rectangular_prism_properties AB AD AC1 AA1) : volume AB AD AA1 = 4 :=
by
  sorry

#check volume_of_rectangular_prism

end volume_of_rectangular_prism_l96_9696


namespace zeros_of_f_l96_9605

noncomputable def f (a b c x : ℝ) : ℝ := (x - a) * (x - b) + (x - b) * (x - c) + (x - c) * (x - a)

theorem zeros_of_f (a b c : ℝ) (h1 : a < b) (h2 : b < c) :
  (∃ x, a < x ∧ x < b ∧ f a b c x = 0) ∧ (∃ y, b < y ∧ y < c ∧ f a b c y = 0) :=
by
  sorry

end zeros_of_f_l96_9605


namespace find_n_value_l96_9621

theorem find_n_value (x y : ℕ) : x = 3 → y = 1 → n = x - y^(x - y) → x > y → n + x * y = 5 := by sorry

end find_n_value_l96_9621


namespace determine_hyperbola_eq_l96_9630

def hyperbola_eq (a b : ℝ) : Prop :=
  ∀ x y : ℝ, (x^2) / (a^2) - (y^2) / (b^2) = 1

def asymptote_condition (a b : ℝ) : Prop :=
  b / a = 3 / 4

def focus_condition (a b : ℝ) : Prop :=
  a^2 + b^2 = 25

theorem determine_hyperbola_eq : 
  ∃ a b : ℝ, 
  (a > 0) ∧ (b > 0) ∧ asymptote_condition a b ∧ focus_condition a b ∧ hyperbola_eq 4 3 :=
sorry

end determine_hyperbola_eq_l96_9630


namespace quadratic_always_positive_if_and_only_if_l96_9626

theorem quadratic_always_positive_if_and_only_if :
  (∀ x : ℝ, x^2 + m * x + m + 3 > 0) ↔ (-2 < m ∧ m < 6) :=
by sorry

end quadratic_always_positive_if_and_only_if_l96_9626


namespace part1_part2_l96_9629

-- Part (1): Solution set of the inequality
theorem part1 (x : ℝ) : (|x - 1| + |x + 1| ≤ 8 - x^2) ↔ (-2 ≤ x) ∧ (x ≤ 2) :=
by
  sorry

-- Part (2): Range of real number t
theorem part2 (t : ℝ) (m n : ℝ) (x : ℝ) (h1 : m + n = 4) (h2 : m > 0) (h3 : n > 0) :  
  |x-t| + |x+t| = (4 * m^2 + n) / (m * n) → t ≥ 9 / 8 ∨ t ≤ -9 / 8 :=
by
  sorry

end part1_part2_l96_9629


namespace hypotenuse_not_5_cm_l96_9613

theorem hypotenuse_not_5_cm (a b c : ℝ) (h₀ : a + b = 8) (h₁ : a^2 + b^2 = c^2) : c ≠ 5 := by
  sorry

end hypotenuse_not_5_cm_l96_9613


namespace sum_of_powers_l96_9681

theorem sum_of_powers : 5^5 + 5^5 + 5^5 + 5^5 = 4 * 5^5 :=
by
  sorry

end sum_of_powers_l96_9681


namespace preston_high_school_teachers_l96_9670

theorem preston_high_school_teachers 
  (num_students : ℕ)
  (classes_per_student : ℕ)
  (classes_per_teacher : ℕ)
  (students_per_class : ℕ)
  (teachers_per_class : ℕ)
  (H : num_students = 1500)
  (C : classes_per_student = 6)
  (T : classes_per_teacher = 5)
  (S : students_per_class = 30)
  (P : teachers_per_class = 1) : 
  (num_students * classes_per_student / students_per_class / classes_per_teacher = 60) :=
by sorry

end preston_high_school_teachers_l96_9670


namespace smallest_b_for_factoring_l96_9647

theorem smallest_b_for_factoring (b : ℕ) : 
  (∃ r s : ℤ, x^2 + b*x + (1200 : ℤ) = (x + r)*(x + s) ∧ b = r + s ∧ r * s = 1200) →
  b = 70 := 
sorry

end smallest_b_for_factoring_l96_9647


namespace max_expression_value_l96_9615

theorem max_expression_value (a b c d e f g h k : ℤ)
  (ha : (a = 1 ∨ a = -1)) (hb : (b = 1 ∨ b = -1))
  (hc : (c = 1 ∨ c = -1)) (hd : (d = 1 ∨ d = -1))
  (he : (e = 1 ∨ e = -1)) (hf : (f = 1 ∨ f = -1))
  (hg : (g = 1 ∨ g = -1)) (hh : (h = 1 ∨ h = -1))
  (hk : (k = 1 ∨ k = -1)) :
  a * e * k - a * f * h + b * f * g - b * d * k + c * d * h - c * e * g ≤ 4 := sorry

end max_expression_value_l96_9615


namespace final_color_all_blue_l96_9643

-- Definitions based on the problem's initial conditions
def initial_blue_sheep : ℕ := 22
def initial_red_sheep : ℕ := 18
def initial_green_sheep : ℕ := 15

-- The final problem statement: prove that all sheep end up being blue
theorem final_color_all_blue (B R G : ℕ) 
  (hB : B = initial_blue_sheep) 
  (hR : R = initial_red_sheep) 
  (hG : G = initial_green_sheep) 
  (interaction : ∀ (B R G : ℕ), (B > 0 ∨ R > 0 ∨ G > 0) → (R ≡ G [MOD 3])) :
  ∃ b, b = B + R + G ∧ R = 0 ∧ G = 0 ∧ b % 3 = 1 ∧ B = b :=
by
  -- Proof to be provided
  sorry

end final_color_all_blue_l96_9643


namespace seunghyo_daily_dosage_l96_9637

theorem seunghyo_daily_dosage (total_medicine : ℝ) (daily_fraction : ℝ) (correct_dosage : ℝ) :
  total_medicine = 426 → daily_fraction = 0.06 → correct_dosage = 25.56 →
  total_medicine * daily_fraction = correct_dosage :=
by
  intros ht hf hc
  simp [ht, hf, hc]
  sorry

end seunghyo_daily_dosage_l96_9637


namespace percentage_of_girls_with_dogs_l96_9624

theorem percentage_of_girls_with_dogs (students total_students : ℕ)
(h_total_students : total_students = 100)
(girls boys : ℕ)
(h_half_students : girls = total_students / 2 ∧ boys = total_students / 2)
(boys_with_dogs : ℕ)
(h_boys_with_dogs : boys_with_dogs = boys / 10)
(total_with_dogs : ℕ)
(h_total_with_dogs : total_with_dogs = 15)
(girls_with_dogs : ℕ)
(h_girls_with_dogs : girls_with_dogs = total_with_dogs - boys_with_dogs)
: (girls_with_dogs * 100 / girls = 20) :=
by
  sorry

end percentage_of_girls_with_dogs_l96_9624


namespace bottom_row_bricks_l96_9672

theorem bottom_row_bricks {x : ℕ} 
  (c1 : ∀ i, i < 5 → (x - i) > 0)
  (c2 : x + (x - 1) + (x - 2) + (x - 3) + (x - 4) = 100) : 
  x = 22 := 
by
  sorry

end bottom_row_bricks_l96_9672


namespace expected_number_of_edges_same_color_3x3_l96_9662

noncomputable def expected_edges_same_color (board_size : ℕ) (blackened_count : ℕ) : ℚ :=
  let total_pairs := 12       -- 6 horizontal pairs + 6 vertical pairs
  let prob_both_white := 1 / 6
  let prob_both_black := 5 / 18
  let prob_same_color := prob_both_white + prob_both_black
  total_pairs * prob_same_color

theorem expected_number_of_edges_same_color_3x3 :
  expected_edges_same_color 3 5 = 16 / 3 :=
by
  sorry

end expected_number_of_edges_same_color_3x3_l96_9662


namespace reflect_P_across_x_axis_l96_9688

def point_reflection_over_x_axis (P : ℝ × ℝ) : ℝ × ℝ :=
  (P.1, -P.2)

theorem reflect_P_across_x_axis : 
  point_reflection_over_x_axis (-3, 1) = (-3, -1) :=
  by
    sorry

end reflect_P_across_x_axis_l96_9688


namespace solve_inequality_l96_9691

open Real

theorem solve_inequality (f : ℝ → ℝ)
  (h_cos : ∀ x, 0 ≤ x ∧ x ≤ π / 2 → f (cos x) ≥ 0) :
  ∀ k : ℤ, ∀ x, (2 * ↑k * π ≤ x ∧ x ≤ 2 * ↑k * π + π) → f (sin x) ≥ 0 :=
by
  intros k x hx
  sorry

end solve_inequality_l96_9691


namespace sqrt_nested_expr_l96_9674

theorem sqrt_nested_expr (x : ℝ) (hx : 0 ≤ x) : 
  (x * (x * (x * x)^(1 / 2))^(1 / 2))^(1 / 2) = (x^7)^(1 / 4) :=
sorry

end sqrt_nested_expr_l96_9674


namespace find_g_neg_one_l96_9602

theorem find_g_neg_one (g : ℝ → ℝ) (h : ∀ x : ℝ, x ≠ 1 / 2 → g x + g ((x + 2) / (2 - 4 * x)) = 3 * x) : 
  g (-1) = - 3 / 2 := 
sorry

end find_g_neg_one_l96_9602


namespace coefficient_a7_l96_9679

theorem coefficient_a7 (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 : ℝ) (x : ℝ) 
  (h : x^9 = a_0 + a_1 * (x - 1) + a_2 * (x - 1)^2 + a_3 * (x - 1)^3 
          + a_4 * (x - 1)^4 + a_5 * (x - 1)^5 + a_6 * (x - 1)^6 + a_7 * (x - 1)^7 
          + a_8 * (x - 1)^8 + a_9 * (x - 1)^9) : 
  a_7 = 36 := 
by
  sorry

end coefficient_a7_l96_9679


namespace range_of_a_l96_9646

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x + 1| + |x - 2| ≥ a) → a ≤ 3 :=
by
  intro h
  sorry

end range_of_a_l96_9646


namespace intersection_eq_l96_9609

def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {x | x^2 ≤ 2}

theorem intersection_eq : A ∩ B = {-1, 0, 1} := by
  sorry

end intersection_eq_l96_9609


namespace intersect_point_sum_l96_9607

theorem intersect_point_sum (a' b' : ℝ) (x y : ℝ) 
    (h1 : x = (1 / 3) * y + a')
    (h2 : y = (1 / 3) * x + b')
    (h3 : x = 2)
    (h4 : y = 4) : 
    a' + b' = 4 :=
by
  sorry

end intersect_point_sum_l96_9607


namespace transformation_of_95_squared_l96_9638

theorem transformation_of_95_squared :
  (9.5 : ℝ) ^ 2 = (10 : ℝ) ^ 2 - 2 * (10 : ℝ) * (0.5 : ℝ) + (0.5 : ℝ) ^ 2 :=
by
  sorry

end transformation_of_95_squared_l96_9638


namespace smallest_denominator_between_l96_9659

theorem smallest_denominator_between :
  ∃ (a b : ℕ), b > 0 ∧ a < b ∧ 6 / 17 < (a : ℚ) / b ∧ (a : ℚ) / b < 9 / 25 ∧ (∀ (c d : ℕ), d > 0 → c < d → 6 / 17 < (c : ℚ) / d → (c : ℚ) / d < 9 / 25 → b ≤ d) ∧ a = 5 ∧ b = 14 :=
by
  existsi 5
  existsi 14
  sorry

end smallest_denominator_between_l96_9659


namespace range_of_m_l96_9683

noncomputable def f (m x : ℝ) : ℝ := m * (x - 2 * m) * (x + m + 3)
noncomputable def g (x : ℝ) : ℝ := 2^x - 2

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, f m x < 0 ∨ g x < 0) ∧ (∃ x : ℝ, x < -4 ∧ f m x * g x < 0) → (-4 < m ∧ m < -2) :=
by
  sorry

end range_of_m_l96_9683


namespace distinct_pen_distribution_l96_9698

theorem distinct_pen_distribution :
  ∃! (a b c d : ℕ), a + b + c + d = 10 ∧
                    1 ≤ a ∧ 1 ≤ b ∧ 1 ≤ c ∧ 1 ≤ d ∧
                    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d :=
sorry

end distinct_pen_distribution_l96_9698


namespace speed_of_stream_l96_9699

theorem speed_of_stream (downstream_speed upstream_speed : ℝ) (h1 : downstream_speed = 14) (h2 : upstream_speed = 8) :
  (downstream_speed - upstream_speed) / 2 = 3 :=
by
  rw [h1, h2]
  norm_num

end speed_of_stream_l96_9699


namespace intersection_empty_l96_9667

def A : Set ℝ := {x | x^2 + 2 * x - 3 < 0}
def B : Set ℝ := {-3, 1, 2}

theorem intersection_empty : A ∩ B = ∅ := 
by
  sorry

end intersection_empty_l96_9667


namespace total_earnings_correct_l96_9604

noncomputable def total_earnings : ℝ :=
  let earnings1 := 12 * (2 + 15 / 60)
  let earnings2 := 15 * (1 + 40 / 60)
  let earnings3 := 10 * (3 + 10 / 60)
  earnings1 + earnings2 + earnings3

theorem total_earnings_correct : total_earnings = 83.75 := by
  sorry

end total_earnings_correct_l96_9604


namespace find_m_correct_l96_9616

structure Triangle where
  X : ℝ × ℝ
  Y : ℝ × ℝ
  Z : ℝ × ℝ
  XY_length : dist X Y = 80
  XZ_length : dist X Z = 100
  YZ_length : dist Y Z = 120

noncomputable def find_m (t : Triangle) : ℝ :=
  let s := (80 + 100 + 120) / 2
  let A := 1 / 2 * 80 * 100
  let r1 := A / s
  let r2 := r1 / 2
  let r3 := r1 / 4
  let O2 := ((40 / 3), 50 + (40 / 3))
  let O3 := (40 + (20 / 3), (20 / 3))
  let O2O3 := dist O2 O3
  let m := (O2O3^2) / 10
  m

theorem find_m_correct (t : Triangle) : find_m t = 610 := sorry

end find_m_correct_l96_9616


namespace sequence_product_mod_five_l96_9652

theorem sequence_product_mod_five : 
  let seq := List.range 20 |>.map (λ k => 10 * k + 3)
  seq.prod % 5 = 1 := 
by
  sorry

end sequence_product_mod_five_l96_9652


namespace jean_total_cost_l96_9685

theorem jean_total_cost 
  (num_pants : ℕ)
  (original_price_per_pant : ℝ)
  (discount_rate : ℝ)
  (tax_rate : ℝ)
  (num_pants_eq : num_pants = 10)
  (original_price_per_pant_eq : original_price_per_pant = 45)
  (discount_rate_eq : discount_rate = 0.2)
  (tax_rate_eq : tax_rate = 0.1) : 
  ∃ total_cost : ℝ, total_cost = 396 :=
by
  sorry

end jean_total_cost_l96_9685


namespace sam_bought_cards_l96_9619

-- Define the initial number of baseball cards Dan had.
def dan_initial_cards : ℕ := 97

-- Define the number of baseball cards Dan has after selling some to Sam.
def dan_remaining_cards : ℕ := 82

-- Prove that the number of baseball cards Sam bought is 15.
theorem sam_bought_cards : (dan_initial_cards - dan_remaining_cards) = 15 :=
by
  sorry

end sam_bought_cards_l96_9619


namespace sum_digits_single_digit_l96_9697

theorem sum_digits_single_digit (n : ℕ) (h : n = 2^100) : (n % 9) = 7 := 
sorry

end sum_digits_single_digit_l96_9697


namespace dot_product_is_ten_l96_9666

-- Define the vectors
def a : ℝ × ℝ := (1, -2)
def b (m : ℝ) : ℝ × ℝ := (2, m)

-- Define the condition that the vectors are parallel
def parallel (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 / v2.1 = v1.2 / v2.2

-- The main theorem statement
theorem dot_product_is_ten (m : ℝ) (h : parallel a (b m)) : 
  a.1 * (b m).1 + a.2 * (b m).2 = 10 := by
  sorry

end dot_product_is_ten_l96_9666


namespace volume_water_needed_l96_9634

noncomputable def radius_sphere : ℝ := 0.5
noncomputable def radius_cylinder : ℝ := 1
noncomputable def height_cylinder : ℝ := 2

theorem volume_water_needed :
  let volume_sphere := (4 / 3) * Real.pi * (radius_sphere ^ 3)
  let total_volume_spheres := 4 * volume_sphere
  let volume_cylinder := Real.pi * (radius_cylinder ^ 2) * height_cylinder
  volume_cylinder - total_volume_spheres = (4 * Real.pi) / 3 :=
by
  let volume_sphere := (4 / 3) * Real.pi * (radius_sphere ^ 3)
  let total_volume_spheres := 4 * volume_sphere
  let volume_cylinder := Real.pi * (radius_cylinder ^ 2) * height_cylinder
  have h : volume_cylinder - total_volume_spheres = (4 * Real.pi) / 3 := sorry
  exact h

end volume_water_needed_l96_9634


namespace minimum_surface_area_of_combined_cuboids_l96_9686

noncomputable def cuboid_combinations (l w h : ℕ) (n : ℕ) : ℕ :=
sorry

theorem minimum_surface_area_of_combined_cuboids :
  ∃ n, cuboid_combinations 2 1 3 3 = 4 ∧ n = 42 :=
sorry

end minimum_surface_area_of_combined_cuboids_l96_9686


namespace Sarah_brother_apples_l96_9678

theorem Sarah_brother_apples (n : Nat) (h1 : 45 = 5 * n) : n = 9 := 
  sorry

end Sarah_brother_apples_l96_9678


namespace time_for_B_alone_l96_9625

theorem time_for_B_alone (r_A r_B r_C : ℚ)
  (h1 : r_A + r_B = 1/3)
  (h2 : r_B + r_C = 2/7)
  (h3 : r_A + r_C = 1/4) :
  1/r_B = 168/31 :=
by
  sorry

end time_for_B_alone_l96_9625


namespace socks_total_is_51_l96_9611

-- Define initial conditions for John and Mary
def john_initial_socks : Nat := 33
def john_thrown_away_socks : Nat := 19
def john_new_socks : Nat := 13

def mary_initial_socks : Nat := 20
def mary_thrown_away_socks : Nat := 6
def mary_new_socks : Nat := 10

-- Define the total socks function
def total_socks (john_initial john_thrown john_new mary_initial mary_thrown mary_new : Nat) : Nat :=
  (john_initial - john_thrown + john_new) + (mary_initial - mary_thrown + mary_new)

-- Statement to prove
theorem socks_total_is_51 : 
  total_socks john_initial_socks john_thrown_away_socks john_new_socks 
              mary_initial_socks mary_thrown_away_socks mary_new_socks = 51 := 
by
  sorry

end socks_total_is_51_l96_9611


namespace range_of_a_l96_9644

theorem range_of_a (a : ℝ) : (∃ x : ℝ, |x + 1| + |x - a| < 4) ↔ (-5 < a ∧ a < 3) :=
by
  sorry

end range_of_a_l96_9644


namespace geometric_sequence_a6_l96_9635

theorem geometric_sequence_a6 : 
  ∀ (a : ℕ → ℚ), (∀ n, a n ≠ 0) → a 1 = 3 → (∀ n, 2 * a (n+1) - a n = 0) → a 6 = 3 / 32 :=
by
  intros a h1 h2 h3
  sorry

end geometric_sequence_a6_l96_9635


namespace analytical_expression_l96_9645

theorem analytical_expression (k : ℝ) (h : k ≠ 0) (x y : ℝ) (hx : x = 4) (hy : y = 6) 
  (eqn : y = k * x) : y = (3 / 2) * x :=
by {
  sorry
}

end analytical_expression_l96_9645


namespace duty_pairing_impossible_l96_9641

theorem duty_pairing_impossible :
  ∀ (m n : ℕ), 29 * m + 32 * n ≠ 29 * 32 := 
by 
  sorry

end duty_pairing_impossible_l96_9641


namespace drone_height_l96_9651

theorem drone_height (TR TS TU : ℝ) (UR : TU^2 + TR^2 = 180^2) (US : TU^2 + TS^2 = 150^2) (RS : TR^2 + TS^2 = 160^2) : 
  TU = Real.sqrt 14650 :=
by
  sorry

end drone_height_l96_9651


namespace sqrt_four_eq_two_or_neg_two_l96_9614

theorem sqrt_four_eq_two_or_neg_two (x : ℝ) : x^2 = 4 → (x = 2 ∨ x = -2) :=
sorry

end sqrt_four_eq_two_or_neg_two_l96_9614


namespace problem_six_circles_l96_9692

noncomputable def six_circles_centers : List (ℝ × ℝ) := [(1,1), (1,3), (3,1), (3,3), (5,1), (5,3)]

noncomputable def slope_of_line_dividing_circles := (2 : ℝ)

def gcd_is_1 (p q r : ℕ) : Prop := Nat.gcd (Nat.gcd p q) r = 1

theorem problem_six_circles (p q r : ℕ) (h_gcd : gcd_is_1 p q r)
  (h_line_eq : ∀ x y, y = slope_of_line_dividing_circles * x - 3 → px = qy + r) :
  p^2 + q^2 + r^2 = 14 :=
sorry

end problem_six_circles_l96_9692


namespace value_of_x_l96_9632

theorem value_of_x (x : ℝ) (h1 : |x| - 1 = 0) (h2 : x - 1 ≠ 0) : x = -1 := 
sorry

end value_of_x_l96_9632


namespace mirror_area_correct_l96_9610

-- Given conditions
def outer_length : ℕ := 80
def outer_width : ℕ := 60
def frame_width : ℕ := 10

-- Deriving the dimensions of the mirror
def mirror_length : ℕ := outer_length - 2 * frame_width
def mirror_width : ℕ := outer_width - 2 * frame_width

-- Statement: Prove that the area of the mirror is 2400 cm^2
theorem mirror_area_correct : mirror_length * mirror_width = 2400 := by
  -- Proof should go here
  sorry

end mirror_area_correct_l96_9610


namespace tan_of_fourth_quadrant_l96_9671

theorem tan_of_fourth_quadrant (α : ℝ) (h₁ : Real.sin α = -5 / 13) (h₂ : α > 3 * Real.pi / 2 ∧ α < 2 * Real.pi) : Real.tan α = -5 / 12 :=
sorry

end tan_of_fourth_quadrant_l96_9671


namespace handshake_count_l96_9690

theorem handshake_count (num_companies : ℕ) (num_representatives : ℕ) 
  (total_handshakes : ℕ) (h1 : num_companies = 5) (h2 : num_representatives = 5)
  (h3 : total_handshakes = (num_companies * num_representatives * 
   (num_companies * num_representatives - 1 - (num_representatives - 1)) / 2)) :
  total_handshakes = 250 :=
by
  rw [h1, h2] at h3
  exact h3

end handshake_count_l96_9690


namespace max_S_n_of_arithmetic_seq_l96_9677

theorem max_S_n_of_arithmetic_seq (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h1 : ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a 1 + n * d)
  (h2 : ∀ n : ℕ, S n = n * (a 1 + a n) / 2)
  (h3 : a 1 + a 3 + a 5 = 15)
  (h4 : a 2 + a 4 + a 6 = 0) : 
  ∃ n : ℕ, S n = 40 ∧ (∀ m : ℕ, S m ≤ 40) :=
sorry

end max_S_n_of_arithmetic_seq_l96_9677


namespace max_geometric_sequence_terms_l96_9668

theorem max_geometric_sequence_terms (a r : ℝ) (n : ℕ) (h_r : r > 1) 
    (h_seq : ∀ (k : ℕ), 1 ≤ k ∧ k ≤ n → 100 ≤ a * r^(k-1) ∧ a * r^(k-1) ≤ 1000) :
  n ≤ 6 :=
sorry

end max_geometric_sequence_terms_l96_9668


namespace students_neither_play_football_nor_cricket_l96_9623

theorem students_neither_play_football_nor_cricket
  (total_students football_players cricket_players both_players : ℕ)
  (h_total : total_students = 470)
  (h_football : football_players = 325)
  (h_cricket : cricket_players = 175)
  (h_both : both_players = 80) :
  (total_students - (football_players + cricket_players - both_players)) = 50 :=
by
  sorry

end students_neither_play_football_nor_cricket_l96_9623


namespace find_m_range_l96_9682

noncomputable def range_m (a b c m : ℝ) (h1 : a > b) (h2 : b > c) 
  (h3 : 1 / (a - b) + m / (b - c) ≥ 9 / (a - c)) : Prop :=
  m ≥ 4

-- Here is the theorem statement
theorem find_m_range (a b c m : ℝ) (h1 : a > b) (h2 : b > c) 
  (h3 : 1 / (a - b) + m / (b - c) ≥ 9 / (a - c)) : range_m a b c m h1 h2 h3 :=
sorry

end find_m_range_l96_9682


namespace inequality_solution_l96_9656

theorem inequality_solution (a b : ℝ)
  (h : ∀ x : ℝ, 0 ≤ x → (x^2 + 1 ≥ a * x + b ∧ a * x + b ≥ (3 / 2) * x^(2 / 3) )) :
  (2 - Real.sqrt 2) / 4 ≤ b ∧ b ≤ (2 + Real.sqrt 2) / 4 ∧
  (1 / Real.sqrt (2 * b)) ≤ a ∧ a ≤ 2 * Real.sqrt (1 - b) :=
  sorry

end inequality_solution_l96_9656


namespace problem_solution_l96_9676

def eq_A (x : ℝ) : Prop := 2 * x = 7
def eq_B (x y : ℝ) : Prop := x^2 + y = 5
def eq_C (x : ℝ) : Prop := x = 1 / x + 1
def eq_D (x : ℝ) : Prop := x^2 + x = 4

def is_quadratic (eq : ℝ → Prop) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x : ℝ, eq x ↔ a * x^2 + b * x + c = 0

theorem problem_solution : is_quadratic eq_D := by
  sorry

end problem_solution_l96_9676


namespace total_simple_interest_l96_9694

noncomputable def simple_interest (P R T : ℝ) : ℝ := P * R * T / 100

theorem total_simple_interest : simple_interest 2500 10 4 = 1000 := 
by
  sorry

end total_simple_interest_l96_9694


namespace money_sum_l96_9655

theorem money_sum (A B : ℕ) (h₁ : (1 / 3 : ℝ) * A = (1 / 4 : ℝ) * B) (h₂ : B = 484) : A + B = 847 := by
  sorry

end money_sum_l96_9655


namespace angle_sum_l96_9669

-- Define the angles in the isosceles triangles
def angle_BAC := 40
def angle_EDF := 50

-- Using the property of isosceles triangles to calculate other angles
def angle_ABC := (180 - angle_BAC) / 2
def angle_DEF := (180 - angle_EDF) / 2

-- Since AD is parallel to CE, angles DAC and ACB are equal as are ADE and DEF
def angle_DAC := angle_ABC
def angle_ADE := angle_DEF

-- The theorem to be proven
theorem angle_sum :
  angle_DAC + angle_ADE = 135 :=
by
  sorry

end angle_sum_l96_9669


namespace q_implies_not_p_l96_9642

-- Define the conditions p and q
def p (x : ℝ) := x < -1
def q (x : ℝ) := x^2 - x - 2 > 0

-- Prove that q implies ¬p
theorem q_implies_not_p (x : ℝ) : q x → ¬ p x := by
  intros hq hp
  -- Provide the steps of logic here
  sorry

end q_implies_not_p_l96_9642


namespace problem_solution_l96_9631

def satisfies_conditions (x y : ℚ) : Prop :=
  (3 * x + y = 6) ∧ (x + 3 * y = 6)

theorem problem_solution :
  ∃ (x y : ℚ), satisfies_conditions x y ∧ 3 * x^2 + 5 * x * y + 3 * y^2 = 24.75 :=
by
  sorry

end problem_solution_l96_9631


namespace smallest_integer_inequality_l96_9658

theorem smallest_integer_inequality:
  ∃ x : ℤ, (2 * x < 3 * x - 10) ∧ ∀ y : ℤ, (2 * y < 3 * y - 10) → y ≥ 11 := by
  sorry

end smallest_integer_inequality_l96_9658


namespace board_transformation_l96_9612

def transformation_possible (a b : ℕ) : Prop :=
  6 ∣ (a * b)

theorem board_transformation (a b : ℕ) (h₁ : 2 ≤ a) (h₂ : 2 ≤ b) : 
  transformation_possible a b ↔ 6 ∣ (a * b) := by
  sorry

end board_transformation_l96_9612


namespace compute_expression_l96_9675

theorem compute_expression : ((-5) * 3) - (7 * (-2)) + ((-4) * (-6)) = 23 := by
  sorry

end compute_expression_l96_9675


namespace bridge_length_is_correct_l96_9687

noncomputable def train_length : ℝ := 135
noncomputable def train_speed_kmh : ℝ := 45
noncomputable def bridge_crossing_time : ℝ := 30

noncomputable def train_speed_ms : ℝ := (train_speed_kmh * 1000) / 3600
noncomputable def total_distance_crossed : ℝ := train_speed_ms * bridge_crossing_time
noncomputable def bridge_length : ℝ := total_distance_crossed - train_length

theorem bridge_length_is_correct : bridge_length = 240 := by
  sorry

end bridge_length_is_correct_l96_9687


namespace parallelogram_height_l96_9693

theorem parallelogram_height (A : ℝ) (b : ℝ) (h : ℝ) (h1 : A = 320) (h2 : b = 20) :
  h = A / b → h = 16 := by
  sorry

end parallelogram_height_l96_9693


namespace minimum_value_expression_l96_9673

theorem minimum_value_expression (p q r : ℝ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) :
  4 ≤ (5 * r) / (3 * p + q) + (5 * p) / (q + 3 * r) + (2 * q) / (p + r) :=
by sorry

end minimum_value_expression_l96_9673


namespace machine_C_time_l96_9640

theorem machine_C_time (T_c : ℝ) :
  (1 / 4 + 1 / 2 + 1 / T_c = 11 / 12) → T_c = 6 :=
by
  sorry

end machine_C_time_l96_9640

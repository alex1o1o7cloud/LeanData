import Mathlib

namespace find_x_l279_27926

def custom_op (a b : ℤ) : ℤ := 2 * a + 3 * b

theorem find_x : ∃ x : ℤ, custom_op 5 (custom_op 7 x) = -4 ∧ x = -56 / 9 := by
  sorry

end find_x_l279_27926


namespace validate_shots_statistics_l279_27974

-- Define the scores and their frequencies
def scores : List ℕ := [6, 7, 8, 9, 10]
def times : List ℕ := [4, 10, 11, 9, 6]

-- Condition 1: Calculate the mode
def mode := 8

-- Condition 2: Calculate the median
def median := 8

-- Condition 3: Calculate the 35th percentile
def percentile_35 := ¬(35 * 40 / 100 = 7)

-- Condition 4: Calculate the average
def average := 8.075

theorem validate_shots_statistics :
  mode = 8
  ∧ median = 8
  ∧ percentile_35
  ∧ average = 8.075 :=
by
  sorry

end validate_shots_statistics_l279_27974


namespace problem_l279_27902

theorem problem (x y z : ℝ) (h : (x - z) ^ 2 - 4 * (x - y) * (y - z) = 0) : z + x - 2 * y = 0 :=
sorry

end problem_l279_27902


namespace find_value_l279_27925

theorem find_value (x : ℝ) (h : Real.cos x - 3 * Real.sin x = 2) : 2 * Real.sin x + 3 * Real.cos x = -7 / 3 := 
sorry

end find_value_l279_27925


namespace minimum_students_ans_q1_correctly_l279_27911

variable (Total Students Q1 Q2 Q1_and_Q2 : ℕ)
variable (did_not_take_test: Student → Bool)

-- Given Conditions
def total_students := 40
def students_ans_q2_correctly := 29
def students_not_taken_test := 10
def students_ans_both_correctly := 29

theorem minimum_students_ans_q1_correctly (H1: Q2 - students_not_taken_test == 30)
                                           (H2: Q1_and_Q2 + students_not_taken_test == total_students)
                                           (H3: Q1_and_Q2 == students_ans_q2_correctly):
  Q1 ≥ 29 := by
  sorry

end minimum_students_ans_q1_correctly_l279_27911


namespace remainder_7_pow_4_div_100_l279_27989

theorem remainder_7_pow_4_div_100 : (7 ^ 4) % 100 = 1 := 
by
  sorry

end remainder_7_pow_4_div_100_l279_27989


namespace circle_tangent_to_x_axis_at_origin_l279_27943

theorem circle_tangent_to_x_axis_at_origin
  (D E F : ℝ)
  (h : ∀ (x y : ℝ), x^2 + y^2 + D*x + E*y + F = 0 → 
      (x = 0 → y = 0)) :
  D = 0 ∧ E ≠ 0 ∧ F = 0 :=
sorry

end circle_tangent_to_x_axis_at_origin_l279_27943


namespace train_length_is_50_meters_l279_27991

theorem train_length_is_50_meters
  (L : ℝ)
  (equal_length : ∀ (a b : ℝ), a = L ∧ b = L → a + b = 2 * L)
  (speed_faster_train : ℝ := 46) -- km/hr
  (speed_slower_train : ℝ := 36) -- km/hr
  (relative_speed : ℝ := speed_faster_train - speed_slower_train)
  (relative_speed_km_per_sec : ℝ := relative_speed / 3600) -- converting km/hr to km/sec
  (time : ℝ := 36) -- seconds
  (distance_covered : ℝ := 2 * L)
  (distance_eq : distance_covered = relative_speed_km_per_sec * time):
  L = 50 / 1000 :=
by 
  -- We will prove it as per the derived conditions
  sorry

end train_length_is_50_meters_l279_27991


namespace range_of_a_minimum_value_of_b_l279_27948

def is_fixed_point (f : ℝ → ℝ) (x₀ : ℝ) : Prop := f x₀ = x₀

noncomputable def f (a b x : ℝ) : ℝ := a * x^2 + (2 * b - 1) * x + b - 2
noncomputable def g (a x : ℝ) : ℝ := -x + a / (3 * a^2 - 2 * a + 1)

theorem range_of_a (h : ∀ b : ℝ, ∃ x1 x2 : ℝ, is_fixed_point (f a b) x1 ∧ is_fixed_point (f a b) x2) : 0 < a ∧ a < 4 :=
sorry

theorem minimum_value_of_b (hx1 : is_fixed_point (f a b) x₁) (hx2 : is_fixed_point (f a b) x₂)
  (hm : g a ((x₁ + x₂) / 2) = (x₁ + x₂) / 2) (ha : 0 < a ∧ a < 4) : b ≥ 3/4 :=
sorry

end range_of_a_minimum_value_of_b_l279_27948


namespace rectangle_side_greater_than_twelve_l279_27947

theorem rectangle_side_greater_than_twelve (a b : ℕ) (h1 : a ≠ b) (h2 : a * b = 6 * (a + b)) : a > 12 ∨ b > 12 :=
sorry

end rectangle_side_greater_than_twelve_l279_27947


namespace square_fold_distance_l279_27923

noncomputable def distance_from_A (area : ℝ) (visible_equal : Bool) : ℝ :=
  if area = 18 ∧ visible_equal then 2 * Real.sqrt 6 else 0

theorem square_fold_distance (area : ℝ) (visible_equal : Bool) :
  area = 18 → visible_equal → distance_from_A area visible_equal = 2 * Real.sqrt 6 :=
by
  sorry

end square_fold_distance_l279_27923


namespace find_minimum_value_l279_27936

open Real

noncomputable def g (x : ℝ) : ℝ := 
  x + x / (x^2 + 2) + x * (x + 3) / (x^2 + 3) + 3 * (x + 1) / (x * (x^2 + 3))

theorem find_minimum_value (x : ℝ) (hx : x > 0) : g x ≥ 4 := 
sorry

end find_minimum_value_l279_27936


namespace cos_alpha_minus_11pi_div_12_eq_neg_2_div_3_l279_27955

theorem cos_alpha_minus_11pi_div_12_eq_neg_2_div_3
  (α : ℝ)
  (h : Real.sin (7 * Real.pi / 12 + α) = 2 / 3) :
  Real.cos (α - 11 * Real.pi / 12) = -(2 / 3) :=
by
  sorry

end cos_alpha_minus_11pi_div_12_eq_neg_2_div_3_l279_27955


namespace factor_poly_l279_27956

theorem factor_poly (x : ℝ) : (75 * x^3 - 300 * x^7) = 75 * x^3 * (1 - 4 * x^4) :=
by sorry

end factor_poly_l279_27956


namespace angle_at_3_15_l279_27905

-- Define the measurements and conditions
def hour_hand_position (hour min : ℕ) : ℝ := 
  30 * hour + 0.5 * min

def minute_hand_position (min : ℕ) : ℝ := 
  6 * min

def angle_between_hands (hour min : ℕ) : ℝ := 
  abs (minute_hand_position min - hour_hand_position hour min)

-- Theorem statement in Lean 4
theorem angle_at_3_15 : angle_between_hands 3 15 = 7.5 :=
by sorry

end angle_at_3_15_l279_27905


namespace derivative_ln_div_x_l279_27928

noncomputable def f (x : ℝ) := (Real.log x) / x

theorem derivative_ln_div_x (x : ℝ) (h : x ≠ 0) : deriv f x = (1 - Real.log x) / (x^2) :=
by
  sorry

end derivative_ln_div_x_l279_27928


namespace ratio_of_a_to_b_l279_27900

theorem ratio_of_a_to_b (a b : ℝ) (h1 : 0.5 / 100 * a = 85) (h2 : 0.75 / 100 * b = 150) : a / b = 17 / 20 :=
by {
  -- Proof will go here
  sorry
}

end ratio_of_a_to_b_l279_27900


namespace cone_surface_area_volume_ineq_l279_27907

theorem cone_surface_area_volume_ineq
  (A V r a m : ℝ)
  (hA : A = π * r * (r + a))
  (hV : V = (1/3) * π * r^2 * m)
  (hPythagoras : a^2 = r^2 + m^2) :
  A^3 ≥ 72 * π * V^2 := 
by
  sorry

end cone_surface_area_volume_ineq_l279_27907


namespace expected_value_of_8_sided_die_l279_27960

-- Define the expected value function for the given win calculation rule.
def expected_value := (1/8 : ℚ) * (7 + 6 + 5 + 4 + 3 + 2 + 1 + 0)

-- Formal statement of the proof problem.
theorem expected_value_of_8_sided_die : 
  expected_value = 3.50 :=
by
  sorry

end expected_value_of_8_sided_die_l279_27960


namespace simplify_and_evaluate_l279_27903

variable (x y : ℝ)
variable (condition_x : x = 1/3)
variable (condition_y : y = -6)

theorem simplify_and_evaluate :
  3 * x^2 * y - (6 * x * y^2 - 2 * (x * y + (3/2) * x^2 * y)) + 2 * (3 * x * y^2 - x * y) = -4 :=
by
  rw [condition_x, condition_y]
  sorry

end simplify_and_evaluate_l279_27903


namespace garden_snake_is_10_inches_l279_27920

-- Define the conditions from the problem statement
def garden_snake_length (garden_snake boa_constrictor : ℝ) : Prop :=
  boa_constrictor = 7 * garden_snake

def boa_constrictor_length (boa_constrictor : ℝ) : Prop :=
  boa_constrictor = 70

-- Prove the length of the garden snake
theorem garden_snake_is_10_inches : ∃ (garden_snake : ℝ), garden_snake_length garden_snake 70 ∧ garden_snake = 10 :=
by {
  sorry
}

end garden_snake_is_10_inches_l279_27920


namespace percentage_error_in_area_l279_27930

theorem percentage_error_in_area (s : ℝ) (h : s > 0) :
  let s' := 1.02 * s
  let A := s ^ 2
  let A' := s' ^ 2
  let error := A' - A
  let percent_error := (error / A) * 100
  percent_error = 4.04 := by
  sorry

end percentage_error_in_area_l279_27930


namespace event_complementary_and_mutually_exclusive_l279_27996

def students : Finset (String × String) := 
  { ("boy", "1"), ("boy", "2"), ("boy", "3"), ("girl", "1"), ("girl", "2") }

def event_at_least_one_girl (s : Finset (String × String)) : Prop :=
  ∃ x ∈ s, (x.1 = "girl")

def event_all_boys (s : Finset (String × String)) : Prop :=
  ∀ x ∈ s, (x.1 = "boy")

def two_students (s : Finset (String × String)) : Prop :=
  s.card = 2

theorem event_complementary_and_mutually_exclusive :
  ∀ s: Finset (String × String), two_students s → 
  (event_at_least_one_girl s ↔ ¬ event_all_boys s) ∧ 
  (event_all_boys s ↔ ¬ event_at_least_one_girl s) :=
sorry

end event_complementary_and_mutually_exclusive_l279_27996


namespace selling_price_A_count_purchasing_plans_refund_amount_l279_27969

-- Problem 1
theorem selling_price_A (last_revenue this_revenue last_price this_price cars_sold : ℝ) 
    (last_revenue_eq : last_revenue = 1) (this_revenue_eq : this_revenue = 0.9)
    (diff_eq : last_price = this_price + 1)
    (same_cars : cars_sold ≠ 0) :
    this_price = 9 := by
  sorry

-- Problem 2
theorem count_purchasing_plans (cost_A cost_B total_cars min_cost max_cost : ℝ)
    (cost_A_eq : cost_A = 0.75) (cost_B_eq : cost_B = 0.6)
    (total_cars_eq : total_cars = 15) (min_cost_eq : min_cost = 0.99)
    (max_cost_eq : max_cost = 1.05) :
    ∃ n : ℕ, n = 5 := by
  sorry

-- Problem 3
theorem refund_amount (refund_A refund_B revenue_A revenue_B cost_A cost_B total_profits a : ℝ)
    (revenue_B_eq : revenue_B = 0.8) (cost_A_eq : cost_A = 0.75)
    (cost_B_eq : cost_B = 0.6) (total_profits_eq : total_profits = 30 - 15 * a) :
    a = 0.5 := by
  sorry

end selling_price_A_count_purchasing_plans_refund_amount_l279_27969


namespace determine_f_value_l279_27933

-- Define initial conditions
def parabola_eqn (d e f : ℝ) (y : ℝ) : ℝ := d * y^2 + e * y + f
def vertex : (ℝ × ℝ) := (2, -3)
def point_on_parabola : (ℝ × ℝ) := (7, 0)

-- Prove that f = 7 given the conditions
theorem determine_f_value (d e f : ℝ) :
  (parabola_eqn d e f (vertex.snd) = vertex.fst) ∧
  (parabola_eqn d e f (point_on_parabola.snd) = point_on_parabola.fst) →
  f = 7 := 
by
  sorry 

end determine_f_value_l279_27933


namespace quadratic_equation_unique_solution_l279_27992

theorem quadratic_equation_unique_solution (a b x k : ℝ) (h : a = 8) (h₁ : b = 36) (h₂ : k = 40.5) : 
  (8*x^2 + 36*x + 40.5 = 0) ∧ x = -2.25 :=
by {
  sorry
}

end quadratic_equation_unique_solution_l279_27992


namespace commission_rate_l279_27942

theorem commission_rate (old_salary new_base_salary sale_amount : ℝ) (required_sales : ℕ) (condition: (old_salary = 75000) ∧ (new_base_salary = 45000) ∧ (sale_amount = 750) ∧ (required_sales = 267)) :
  ∃ commission_rate : ℝ, abs (commission_rate - 0.14981) < 0.0001 :=
by
  sorry

end commission_rate_l279_27942


namespace negation_P_l279_27914

-- Define the condition that x is a real number
variable (x : ℝ)

-- Define the proposition P
def P := ∀ (x : ℝ), x ≥ 2

-- Define the negation of P
def not_P := ∃ (x : ℝ), x < 2

-- Theorem stating the equivalence of the negation of P
theorem negation_P : ¬P ↔ not_P := by
  sorry

end negation_P_l279_27914


namespace group_size_l279_27946

noncomputable def total_cost : ℤ := 13500
noncomputable def cost_per_person : ℤ := 900

theorem group_size : total_cost / cost_per_person = 15 :=
by {
  sorry
}

end group_size_l279_27946


namespace star_eq_zero_iff_x_eq_5_l279_27954

/-- Define the operation * on real numbers -/
def star (a b : ℝ) : ℝ := a^2 - 2*a*b + b^2

/-- Proposition stating that x = 5 is the solution to (x - 4) * 1 = 0 -/
theorem star_eq_zero_iff_x_eq_5 (x : ℝ) : (star (x-4) 1 = 0) ↔ x = 5 :=
by
  sorry

end star_eq_zero_iff_x_eq_5_l279_27954


namespace age_equation_correct_l279_27921

-- Define the main theorem
theorem age_equation_correct (x : ℕ) (h1 : ∀ (b : ℕ), b = 2 * x) (h2 : ∀ (b4 s4 : ℕ), b4 = b - 4 ∧ s4 = x - 4 ∧ b4 = 3 * s4) : 
  2 * x - 4 = 3 * (x - 4) :=
by
  sorry

end age_equation_correct_l279_27921


namespace distance_between_A_and_mrs_A_l279_27949

-- Define the initial conditions
def speed_mr_A : ℝ := 30 -- Mr. A's speed in kmph
def speed_mrs_A : ℝ := 10 -- Mrs. A's speed in kmph
def speed_bee : ℝ := 60 -- The bee's speed in kmph
def distance_bee_traveled : ℝ := 180 -- Distance traveled by the bee in km

-- Define the proven statement
theorem distance_between_A_and_mrs_A : 
  distance_bee_traveled / speed_bee * (speed_mr_A + speed_mrs_A) = 120 := 
by 
  sorry

end distance_between_A_and_mrs_A_l279_27949


namespace jean_pages_written_l279_27973

theorem jean_pages_written:
  (∀ d : ℕ, 150 * d = 900 → d * 2 = 12) :=
by
  sorry

end jean_pages_written_l279_27973


namespace smallest_total_cashews_l279_27980

noncomputable def first_monkey_final (c1 c2 c3 : ℕ) : ℕ :=
  (2 * c1) / 3 + c2 / 6 + (4 * c3) / 18

noncomputable def second_monkey_final (c1 c2 c3 : ℕ) : ℕ :=
  c1 / 6 + (2 * c2) / 6 + (4 * c3) / 18

noncomputable def third_monkey_final (c1 c2 c3 : ℕ) : ℕ :=
  c1 / 6 + (2 * c2) / 6 + c3 / 9

theorem smallest_total_cashews : ∃ (c1 c2 c3 : ℕ), ∃ y : ℕ,
  3 * y = first_monkey_final c1 c2 c3 ∧
  2 * y = second_monkey_final c1 c2 c3 ∧
  y = third_monkey_final c1 c2 c3 ∧
  c1 + c2 + c3 = 630 :=
sorry

end smallest_total_cashews_l279_27980


namespace finance_charge_rate_l279_27952

theorem finance_charge_rate (original_balance total_payment finance_charge_rate : ℝ)
    (h1 : original_balance = 150)
    (h2 : total_payment = 153)
    (h3 : finance_charge_rate = ((total_payment - original_balance) / original_balance) * 100) :
    finance_charge_rate = 2 :=
by
  rw [h1, h2] at h3
  norm_num at h3
  exact h3

end finance_charge_rate_l279_27952


namespace max_sin_a_l279_27994

theorem max_sin_a (a b : ℝ)
  (h1 : b = Real.pi / 2 - a)
  (h2 : Real.cos (a + b) = Real.cos a + Real.cos b) :
  Real.sin a ≤ Real.sqrt 2 / 2 :=
sorry

end max_sin_a_l279_27994


namespace negation_of_proposition_l279_27970

variable (x : ℝ)

theorem negation_of_proposition (h : ∃ x : ℝ, x^2 + x - 1 < 0) : ¬ (∀ x : ℝ, x^2 + x - 1 ≥ 0) :=
sorry

end negation_of_proposition_l279_27970


namespace three_pow_1000_mod_seven_l279_27981

theorem three_pow_1000_mod_seven : (3 ^ 1000) % 7 = 4 := 
by 
  -- proof omitted
  sorry

end three_pow_1000_mod_seven_l279_27981


namespace expression_eval_l279_27935

theorem expression_eval :
  -14 - (-2) ^ 3 * (1 / 4) - 16 * (1 / 2 - 1 / 4 + 3 / 8) = -22 := by
  sorry

end expression_eval_l279_27935


namespace robert_saves_5_dollars_l279_27968

theorem robert_saves_5_dollars :
  let original_price := 50
  let promotion_c_discount (price : ℕ) := price * 20 / 100
  let promotion_d_discount (price : ℕ) := 15
  let cost_promotion_c := original_price + (original_price - promotion_c_discount original_price)
  let cost_promotion_d := original_price + (original_price - promotion_d_discount original_price)
  (cost_promotion_c - cost_promotion_d) = 5 :=
by
  sorry

end robert_saves_5_dollars_l279_27968


namespace ganesh_ram_together_l279_27962

theorem ganesh_ram_together (G R S : ℝ) (h1 : G + R + S = 1 / 16) (h2 : S = 1 / 48) : (G + R) = 1 / 24 :=
by
  sorry

end ganesh_ram_together_l279_27962


namespace san_antonio_bus_passes_4_austin_buses_l279_27904

theorem san_antonio_bus_passes_4_austin_buses :
  ∀ (hourly_austin_buses : ℕ → ℕ) (every_50_minute_san_antonio_buses : ℕ → ℕ) (trip_time : ℕ),
    (∀ h : ℕ, hourly_austin_buses (h) = (h * 60)) →
    (∀ m : ℕ, every_50_minute_san_antonio_buses (m) = (m * 60 + 50)) →
    trip_time = 240 →
    ∃ num_buses_passed : ℕ, num_buses_passed = 4 :=
by
  sorry

end san_antonio_bus_passes_4_austin_buses_l279_27904


namespace snow_volume_l279_27951

-- Define the dimensions of the sidewalk and the snow depth
def length : ℝ := 20
def width : ℝ := 2
def depth : ℝ := 0.5

-- Define the volume calculation
def volume (l w d : ℝ) : ℝ := l * w * d

-- The theorem to prove
theorem snow_volume : volume length width depth = 20 := 
by
  sorry

end snow_volume_l279_27951


namespace smaller_circle_radius_l279_27957

theorem smaller_circle_radius :
  ∀ (R r : ℝ), R = 10 ∧ (4 * r = 2 * R) → r = 5 :=
by
  intro R r
  intro h
  have h1 : R = 10 := h.1
  have h2 : 4 * r = 2 * R := h.2
  -- Use the conditions to eventually show r = 5
  sorry

end smaller_circle_radius_l279_27957


namespace largest_among_four_l279_27901

theorem largest_among_four (a b : ℝ) (h : 0 < a ∧ a < b ∧ a + b = 1) :
  a^2 + b^2 = max (max (max a (1/2)) (2*a*b)) (a^2 + b^2) :=
by
  sorry

end largest_among_four_l279_27901


namespace cost_of_article_l279_27997

-- Definitions for conditions
def gain_340 (C G : ℝ) : Prop := 340 = C + G
def gain_360 (C G : ℝ) : Prop := 360 = C + G + 0.05 * C

-- Theorem to be proven
theorem cost_of_article (C G : ℝ) (h1 : gain_340 C G) (h2 : gain_360 C G) : C = 400 :=
by sorry

end cost_of_article_l279_27997


namespace work_problem_l279_27913

theorem work_problem (x : ℝ) (h1 : x > 0) 
                      (h2 : (2 * (1 / 4 + 1 / x) + 2 * (1 / x) = 1)) : 
                      x = 8 := sorry

end work_problem_l279_27913


namespace geometric_sequence_condition_l279_27998

variable (a_1 : ℝ) (q : ℝ)

noncomputable def geometric_sum (n : ℕ) : ℝ :=
if q = 1 then a_1 * n else a_1 * (1 - q^n) / (1 - q)

theorem geometric_sequence_condition (a_1 : ℝ) (q : ℝ) :
  (a_1 > 0) ↔ (geometric_sum a_1 q 2017 > 0) :=
by sorry

end geometric_sequence_condition_l279_27998


namespace correct_limiting_reagent_and_yield_l279_27950

noncomputable def balanced_reaction_theoretical_yield : Prop :=
  let Fe2O3_initial : ℕ := 4
  let CaCO3_initial : ℕ := 10
  let moles_Fe2O3_needed_for_CaCO3 := Fe2O3_initial * (6 / 2)
  let limiting_reagent := if CaCO3_initial < moles_Fe2O3_needed_for_CaCO3 then true else false
  let theoretical_yield := (CaCO3_initial * (3 / 6))
  limiting_reagent = true ∧ theoretical_yield = 5

theorem correct_limiting_reagent_and_yield : balanced_reaction_theoretical_yield :=
by
  sorry

end correct_limiting_reagent_and_yield_l279_27950


namespace modulus_of_z_l279_27917

-- Definitions of the problem conditions
def z := Complex.mk 1 (-1)

-- Statement of the math proof problem
theorem modulus_of_z : Complex.abs z = Real.sqrt 2 := by
  sorry -- Proof placeholder

end modulus_of_z_l279_27917


namespace max_truthful_students_l279_27932

def count_students (n : ℕ) : ℕ :=
  (n * (n + 1)) / 2

theorem max_truthful_students : count_students 2015 = 2031120 :=
by sorry

end max_truthful_students_l279_27932


namespace max_rectangle_area_l279_27964

-- Define the perimeter and side lengths of the rectangle
def perimeter := 60
def L (x : ℝ) := x
def W (x : ℝ) := 30 - x

-- Define the area of the rectangle
def area (x : ℝ) := L x * W x

-- State the theorem to prove the maximum area is 225 square feet
theorem max_rectangle_area : ∃ x : ℝ, 0 ≤ x ∧ x ≤ 30 ∧ area x = 225 :=
by
  sorry

end max_rectangle_area_l279_27964


namespace cheapest_store_for_60_balls_l279_27995

def cost_store_A (n : ℕ) (price_per_ball : ℕ) (free_per_10 : ℕ) : ℕ :=
  if n < 10 then n * price_per_ball
  else (n / 10) * 10 * price_per_ball + (n % 10) * price_per_ball * (n / (10 + free_per_10))

def cost_store_B (n : ℕ) (discount : ℕ) (price_per_ball : ℕ) : ℕ :=
  n * (price_per_ball - discount)

def cost_store_C (n : ℕ) (price_per_ball : ℕ) (cashback_threshold cashback_amt : ℕ) : ℕ :=
  let initial_cost := n * price_per_ball
  let cashback := (initial_cost / cashback_threshold) * cashback_amt
  initial_cost - cashback

theorem cheapest_store_for_60_balls
  (price_per_ball discount free_per_10 cashback_threshold cashback_amt : ℕ) :
  cost_store_A 60 price_per_ball free_per_10 = 1250 →
  cost_store_B 60 discount price_per_ball = 1200 →
  cost_store_C 60 price_per_ball cashback_threshold cashback_amt = 1290 →
  min (cost_store_A 60 price_per_ball free_per_10) (min (cost_store_B 60 discount price_per_ball) (cost_store_C 60 price_per_ball cashback_threshold cashback_amt))
  = 1200 :=
by
  sorry

end cheapest_store_for_60_balls_l279_27995


namespace john_age_l279_27978

/-!
# John’s Current Age Proof
Given the following condition:
1. 9 years from now, John will be 3 times as old as he was 11 years ago.
Prove that John is currently 21 years old.
-/

def john_current_age (x : ℕ) : Prop :=
  (x + 9 = 3 * (x - 11)) → (x = 21)

-- Proof Statement
theorem john_age : john_current_age 21 :=
by
  sorry

end john_age_l279_27978


namespace find_b_eq_neg_three_l279_27916

theorem find_b_eq_neg_three (b : ℝ) (h : (2 - b) / 5 = -(2 * b + 1) / 5) : b = -3 :=
by
  sorry

end find_b_eq_neg_three_l279_27916


namespace expression_evaluation_l279_27999

theorem expression_evaluation (x y : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : x^2 = 1 / y^2) :
  (x^2 - 4 / x^2) * (y^2 + 4 / y^2) = x^4 - 16 / x^4 :=
by
  sorry

end expression_evaluation_l279_27999


namespace max_x_plus_2y_l279_27984

theorem max_x_plus_2y (x y : ℝ) (h : 3 * (x^2 + y^2) = x + y) : x + 2 * y ≤ 3 :=
sorry

end max_x_plus_2y_l279_27984


namespace sum_in_base5_correct_l279_27966

-- Defining the integers
def num1 : ℕ := 210
def num2 : ℕ := 72

-- Summing the integers
def sum : ℕ := num1 + num2

-- Converting the resulting sum to base 5
def to_base5 (n : ℕ) : String :=
  let rec aux (n : ℕ) (acc : List Char) : List Char :=
    if n < 5 then Char.ofNat (n + 48) :: acc
    else aux (n / 5) (Char.ofNat (n % 5 + 48) :: acc)
  String.mk (aux n [])

-- The expected sum in base 5
def expected_sum_base5 : String := "2062"

-- The Lean theorem to be proven
theorem sum_in_base5_correct : to_base5 sum = expected_sum_base5 :=
by
  sorry

end sum_in_base5_correct_l279_27966


namespace number_of_parrots_in_each_cage_l279_27965

theorem number_of_parrots_in_each_cage (num_cages : ℕ) (total_birds : ℕ) (parrots_per_cage parakeets_per_cage : ℕ)
    (h1 : num_cages = 9)
    (h2 : parrots_per_cage = parakeets_per_cage)
    (h3 : total_birds = 36)
    (h4 : total_birds = num_cages * (parrots_per_cage + parakeets_per_cage)) :
  parrots_per_cage = 2 :=
by
  sorry

end number_of_parrots_in_each_cage_l279_27965


namespace birch_trees_count_l279_27983

-- Definitions based on the conditions
def total_trees : ℕ := 4000
def percentage_spruce : ℕ := 10
def percentage_pine : ℕ := 13

def count_spruce : ℕ := (percentage_spruce * total_trees) / 100
def count_pine : ℕ := (percentage_pine * total_trees) / 100
def count_oak : ℕ := count_spruce + count_pine

def count_birch : ℕ := total_trees - (count_spruce + count_pine + count_oak)

-- The theorem to be proven
theorem birch_trees_count :
  count_birch = 2160 := by
  sorry

end birch_trees_count_l279_27983


namespace original_number_l279_27967

theorem original_number (x : ℝ) (h1 : 268 * x = 19832) (h2 : 2.68 * x = 1.9832) : x = 74 :=
sorry

end original_number_l279_27967


namespace polynomial_value_at_2_l279_27908

-- Define the polynomial f(x)
def f (x : ℝ) : ℝ := x^6 - 12*x^5 + 60*x^4 - 160*x^3 + 240*x^2 - 192*x + 64

-- Define the transformation rules for each v_i according to Horner's Rule
def v0 : ℝ := 1
def v1 (x : ℝ) : ℝ := (v0 * x) - 12
def v2 (x : ℝ) : ℝ := (v1 x * x) + 60
def v3 (x : ℝ) : ℝ := (v2 x * x) - 160

-- State the theorem to be proven
theorem polynomial_value_at_2 : v3 2 = -80 := 
by 
  -- Since this is just a Lean 4 statement, we include sorry to defer proof
  sorry

end polynomial_value_at_2_l279_27908


namespace area_of_square_l279_27958

-- Conditions: Points A (5, -2) and B (5, 3) are adjacent corners of a square.
def A : ℝ × ℝ := (5, -2)
def B : ℝ × ℝ := (5, 3)

-- The statement to prove that the area of the square formed by these points is 25.
theorem area_of_square : (∃ s : ℝ, s = Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)) → s^2 = 25 :=
sorry

end area_of_square_l279_27958


namespace max_xyz_eq_one_l279_27985

noncomputable def max_xyz (x y z : ℝ) : ℝ :=
  if h_cond : 0 < x ∧ 0 < y ∧ 0 < z ∧ (x * y + z ^ 2 = (x + z) * (y + z)) ∧ (x + y + z = 3) then
    x * y * z
  else
    0

theorem max_xyz_eq_one : ∀ (x y z : ℝ), 0 < x → 0 < y → 0 < z → 
  (x * y + z ^ 2 = (x + z) * (y + z)) → (x + y + z = 3) → max_xyz x y z ≤ 1 :=
by
  intros x y z hx hy hz h1 h2
  -- Proof is omitted here
  sorry

end max_xyz_eq_one_l279_27985


namespace negate_proposition_l279_27977

def p (x : ℝ) : Prop := x^2 + x - 6 > 0
def q (x : ℝ) : Prop := x > 2 ∨ x < -3

def neg_p (x : ℝ) : Prop := x^2 + x - 6 ≤ 0
def neg_q (x : ℝ) : Prop := -3 ≤ x ∧ x ≤ 2

theorem negate_proposition (x : ℝ) :
  (¬ (p x → q x)) ↔ (neg_p x → neg_q x) :=
by unfold p q neg_p neg_q; apply sorry

end negate_proposition_l279_27977


namespace baby_plants_produced_l279_27918

theorem baby_plants_produced (baby_plants_per_time: ℕ) (times_per_year: ℕ) (years: ℕ) (total_babies: ℕ) :
  baby_plants_per_time = 2 ∧ times_per_year = 2 ∧ years = 4 ∧ total_babies = baby_plants_per_time * times_per_year * years → 
  total_babies = 16 :=
by
  intro h
  rcases h with ⟨h1, h2, h3, h4⟩
  sorry

end baby_plants_produced_l279_27918


namespace arithmetic_sequence_sum_l279_27945

-- Define the arithmetic sequence and the given conditions
def arithmetic_seq (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Define the specific values for the sequence a_1 = 2 and a_2 + a_3 = 13
variables {a : ℕ → ℤ} (d : ℤ)
axiom h1 : a 1 = 2
axiom h2 : a 2 + a 3 = 13

-- Conclude the value of a_4 + a_5 + a_6
theorem arithmetic_sequence_sum : a 4 + a 5 + a 6 = 42 :=
sorry

end arithmetic_sequence_sum_l279_27945


namespace total_ticket_sales_l279_27961

def ticket_price : Type := 
  ℕ → ℕ

def total_individual_sales (student_count adult_count child_count senior_count : ℕ) (prices : ticket_price) : ℝ :=
  (student_count * prices 6 + adult_count * prices 8 + child_count * prices 4 + senior_count * prices 7)

def total_group_sales (group_student_count group_adult_count group_child_count group_senior_count : ℕ) (prices : ticket_price) : ℝ :=
  let total_price := (group_student_count * prices 6 + group_adult_count * prices 8 + group_child_count * prices 4 + group_senior_count * prices 7)
  if (group_student_count + group_adult_count + group_child_count + group_senior_count) > 10 then 
    total_price - 0.10 * total_price 
  else 
    total_price

theorem total_ticket_sales
  (prices : ticket_price)
  (student_count adult_count child_count senior_count : ℕ)
  (group_student_count group_adult_count group_child_count group_senior_count : ℕ)
  (total_sales : ℝ) :
  student_count = 20 →
  adult_count = 12 →
  child_count = 15 →
  senior_count = 10 →
  group_student_count = 5 →
  group_adult_count = 8 →
  group_child_count = 10 →
  group_senior_count = 9 →
  prices 6 = 6 →
  prices 8 = 8 →
  prices 4 = 4 →
  prices 7 = 7 →
  total_sales = (total_individual_sales student_count adult_count child_count senior_count prices) + (total_group_sales group_student_count group_adult_count group_child_count group_senior_count prices) →
  total_sales = 523.30 := by
  sorry

end total_ticket_sales_l279_27961


namespace inequality_solution_l279_27915

variable {a b : ℝ}

theorem inequality_solution
  (h1 : a < 0)
  (h2 : -1 < b ∧ b < 0) :
  ab > ab^2 ∧ ab^2 > a := 
sorry

end inequality_solution_l279_27915


namespace range_of_a_l279_27910

variable (a : ℝ)
def p : Prop := ∀ x : ℝ, 1 ≤ x ∧ x ≤ 12 → x^2 - a ≥ 0
def q : Prop := ∃ x₀ : ℝ, x₀^2 + (a - 1) * x₀ + 1 < 0

theorem range_of_a (hpq : p a ∨ q a) (hpnq : ¬p a ∧ ¬q a) : 
  (-1 ≤ a ∧ a ≤ 1) ∨ (a > 3) :=
sorry

end range_of_a_l279_27910


namespace integer_roots_of_polynomial_l279_27988

theorem integer_roots_of_polynomial (a₂ a₁ : ℤ) :
  {x : ℤ | (x^3 + a₂ * x^2 + a₁ * x - 18 = 0)} ⊆ {-18, -9, -6, -3, -2, -1, 1, 2, 3, 6, 9, 18} :=
by sorry

end integer_roots_of_polynomial_l279_27988


namespace diving_competition_scores_l279_27940

theorem diving_competition_scores (A B C D E : ℝ) (hA : 1 ≤ A ∧ A ≤ 10)
  (hB : 1 ≤ B ∧ B ≤ 10) (hC : 1 ≤ C ∧ C ≤ 10) (hD : 1 ≤ D ∧ D ≤ 10) 
  (hE : 1 ≤ E ∧ E ≤ 10) (degree_of_difficulty : ℝ) (h_diff : degree_of_difficulty = 3.2)
  (point_value : ℝ) (h_point_value : point_value = 79.36) :
  A = max A (max B (max C (max D E))) →
  E = min A (min B (min C (min D E))) →
  (B + C + D) = (point_value / degree_of_difficulty) :=
by sorry

end diving_competition_scores_l279_27940


namespace initial_tests_count_l279_27979

theorem initial_tests_count (n S : ℕ)
  (h1 : S = 35 * n)
  (h2 : (S - 20) / (n - 1) = 40) :
  n = 4 := 
sorry

end initial_tests_count_l279_27979


namespace negation_of_p_l279_27922

-- Define the proposition p
def p : Prop := ∃ n : ℕ, 2^n > 100

-- Goal is to show the negation of p
theorem negation_of_p : (¬ p) = (∀ n : ℕ, 2^n ≤ 100) :=
by
  sorry

end negation_of_p_l279_27922


namespace pipe_p_fills_cistern_in_12_minutes_l279_27971

theorem pipe_p_fills_cistern_in_12_minutes :
  (∃ (t : ℝ), 
    ∀ (q_fill_rate p_fill_rate : ℝ), 
      q_fill_rate = 1 / 15 ∧ 
      t > 0 ∧ 
      (4 * (1 / t + q_fill_rate) + 6 * q_fill_rate = 1) → t = 12) :=
sorry

end pipe_p_fills_cistern_in_12_minutes_l279_27971


namespace probability_of_b_l279_27993

noncomputable def P : ℕ → ℝ := sorry

axiom P_a : P 0 = 0.15
axiom P_a_and_b : P 1 = 0.15
axiom P_neither_a_nor_b : P 2 = 0.6

theorem probability_of_b : P 3 = 0.4 := 
by
  sorry

end probability_of_b_l279_27993


namespace expenditure_record_l279_27975

/-- Lean function to represent the condition and the proof problem -/
theorem expenditure_record (income expenditure : Int) (h_income : income = 500) (h_recorded_income : income = 500) (h_expenditure : expenditure = 200) : expenditure = -200 := 
by
  sorry

end expenditure_record_l279_27975


namespace sum_of_first_ten_primes_ending_in_3_is_671_l279_27959

noncomputable def sum_of_first_ten_primes_ending_in_3 : ℕ :=
  3 + 13 + 23 + 43 + 53 + 73 + 83 + 103 + 113 + 163

theorem sum_of_first_ten_primes_ending_in_3_is_671 :
  sum_of_first_ten_primes_ending_in_3 = 671 :=
by
  sorry

end sum_of_first_ten_primes_ending_in_3_is_671_l279_27959


namespace solutions_equation1_solutions_equation2_l279_27941

-- Definition for the first equation
def equation1 (x : ℝ) : Prop := 4 * x^2 - 9 = 0

-- Definition for the second equation
def equation2 (x : ℝ) : Prop := 2 * x^2 - 3 * x - 5 = 0

theorem solutions_equation1 (x : ℝ) :
  equation1 x ↔ (x = 3 / 2 ∨ x = -3 / 2) := 
  by sorry

theorem solutions_equation2 (x : ℝ) :
  equation2 x ↔ (x = 1 ∨ x = 5 / 2) := 
  by sorry

end solutions_equation1_solutions_equation2_l279_27941


namespace inequality_for_positive_n_and_x_l279_27938

theorem inequality_for_positive_n_and_x (n : ℕ) (x : ℝ) (hn : n > 0) (hx : x > 0) :
  (x^(2 * n - 1) - 1) / (2 * n - 1) ≤ (x^(2 * n) - 1) / (2 * n) :=
by sorry

end inequality_for_positive_n_and_x_l279_27938


namespace cookies_with_five_cups_l279_27919

theorem cookies_with_five_cups (cookies_per_four_cups : ℕ) (flour_for_four_cups : ℕ) (flour_for_five_cups : ℕ) (h : 24 / 4 = cookies_per_four_cups / 5) :
  cookies_per_four_cups = 30 :=
by
  sorry

end cookies_with_five_cups_l279_27919


namespace eldest_child_age_l279_27912

variable (y m e : Nat)

theorem eldest_child_age :
  (m - y = 3) →
  (e = 3 * y) →
  (e = y + m + 2) →
  (e = 15) :=
by
  intros h1 h2 h3
  sorry

end eldest_child_age_l279_27912


namespace taxi_fare_l279_27934

theorem taxi_fare (x : ℝ) (h : x > 6) : 
  let starting_price := 6
  let mid_distance_fare := (6 - 2) * 2.4
  let long_distance_fare := (x - 6) * 3.6
  let total_fare := starting_price + mid_distance_fare + long_distance_fare
  total_fare = 3.6 * x - 6 :=
by
  sorry

end taxi_fare_l279_27934


namespace avg_diff_condition_l279_27976

variable (a b c : ℝ)

theorem avg_diff_condition (h1 : (a + b) / 2 = 110) (h2 : (b + c) / 2 = 150) : a - c = -80 :=
by
  sorry

end avg_diff_condition_l279_27976


namespace distance_between_centers_l279_27924

-- Declare radii of the circles and the shortest distance between points on the circles
def R := 28
def r := 12
def d := 10

-- Define the problem to prove the distance between the centers
theorem distance_between_centers (R r d : ℝ) (hR : R = 28) (hr : r = 12) (hd : d = 10) : 
  ∀ OO1 : ℝ, OO1 = 6 :=
by sorry

end distance_between_centers_l279_27924


namespace number_of_outcomes_for_champions_l279_27990

def num_events : ℕ := 3
def num_competitors : ℕ := 6
def total_possible_outcomes : ℕ := num_competitors ^ num_events

theorem number_of_outcomes_for_champions :
  total_possible_outcomes = 216 :=
by
  sorry

end number_of_outcomes_for_champions_l279_27990


namespace problem_statement_l279_27909

theorem problem_statement (f : ℝ → ℝ) (a b c m : ℝ)
  (h_cond1 : ∀ x, f x = -x^2 + a * x + b)
  (h_range : ∀ y, y ∈ Set.range f ↔ y ≤ 0)
  (h_ineq_sol : ∀ x, ((-x^2 + a * x + b > c - 1) ↔ (m - 4 < x ∧ x < m + 1))) :
  (b = -(1/4) * (2 * m - 3)^2) ∧ (c = -(21 / 4)) := sorry

end problem_statement_l279_27909


namespace proof_problem_l279_27906

-- Proposition B: ∃ x ∈ ℝ, x^2 - 3*x + 3 < 0
def propB : Prop := ∃ x : ℝ, x^2 - 3 * x + 3 < 0

-- Proposition D: ∀ x ∈ ℝ, x^2 - a*x + 1 = 0 has real solutions
def propD (a : ℝ) : Prop := ∀ x : ℝ, ∃ (x1 x2 : ℝ), x^2 - a * x + 1 = 0

-- Negation of Proposition B: ∀ x ∈ ℝ, x^2 - 3 * x + 3 ≥ 0
def neg_propB : Prop := ∀ x : ℝ, x^2 - 3 * x + 3 ≥ 0

-- Negation of Proposition D: ∃ a ∈ ℝ, ∃ x ∈ ℝ, ∄ (x1 x2 : ℝ), x^2 - a * x + 1 = 0
def neg_propD : Prop := ∃ a : ℝ, ∀ x : ℝ, ¬ ∃ (x1 x2 : ℝ), x^2 - a * x + 1 = 0 

-- The main theorem combining the results based on the solutions.
theorem proof_problem : neg_propB ∧ neg_propD :=
by
  sorry

end proof_problem_l279_27906


namespace find_circle_equation_l279_27972

-- Define the conditions on the circle
def passes_through_points (center : ℝ × ℝ) (radius : ℝ) : Prop :=
  ∃ c : ℝ × ℝ, ∃ r : ℝ, (c = center ∧ r = radius) ∧ 
  dist (0, 2) c = r ∧ dist (0, 4) c = r

def lies_on_line (center : ℝ × ℝ) : Prop :=
  2 * center.1 - center.2 - 1 = 0

-- Define the problem
theorem find_circle_equation :
  ∃ (center : ℝ × ℝ) (radius : ℝ), 
  passes_through_points center radius ∧ lies_on_line center ∧ 
  (∀ x y : ℝ, (x - center.1)^2 + (y - center.2)^2 = radius^2 
  ↔ (x - 2)^2 + (y - 3)^2 = 5) :=
sorry

end find_circle_equation_l279_27972


namespace distance_between_foci_of_hyperbola_l279_27944

open Real

-- Definitions based on the given conditions
def asymptote1 (x : ℝ) : ℝ := x + 3
def asymptote2 (x : ℝ) : ℝ := -x + 5
def hyperbola_passes_through (x y : ℝ) : Prop := x = 4 ∧ y = 6
noncomputable def hyperbola_centre : (ℝ × ℝ) := (1, 4)

-- Definition of the hyperbola and the proof problem
theorem distance_between_foci_of_hyperbola (x y : ℝ) (hx : asymptote1 x = y) (hy : asymptote2 x = y) (hpass : hyperbola_passes_through 4 6) :
  2 * (sqrt (5 + 5)) = 2 * sqrt 10 :=
by
  sorry

end distance_between_foci_of_hyperbola_l279_27944


namespace necessary_sufficient_condition_l279_27953

theorem necessary_sufficient_condition 
  (a b : ℝ) : 
  a * |a + b| < |a| * (a + b) ↔ (a < 0 ∧ b > -a) :=
sorry

end necessary_sufficient_condition_l279_27953


namespace nuts_distributive_problem_l279_27982

theorem nuts_distributive_problem (x y : ℕ) (h1 : 70 ≤ x + y) (h2 : x + y ≤ 80) (h3 : (3 / 4 : ℚ) * x + (1 / 5 : ℚ) * (y + (1 / 4 : ℚ) * x) = (x : ℚ) + 1) :
  x = 36 ∧ y = 41 :=
by
  sorry

end nuts_distributive_problem_l279_27982


namespace mary_needs_more_sugar_l279_27929

def recipe_sugar := 14
def sugar_already_added := 2
def sugar_needed := recipe_sugar - sugar_already_added

theorem mary_needs_more_sugar : sugar_needed = 12 := by
  sorry

end mary_needs_more_sugar_l279_27929


namespace tan_C_in_triangle_l279_27927

theorem tan_C_in_triangle
  (A B C : ℝ)
  (cos_A : Real.cos A = 4/5)
  (tan_A_minus_B : Real.tan (A - B) = -1/2) :
  Real.tan C = 11/2 := 
sorry

end tan_C_in_triangle_l279_27927


namespace simplify_expression_l279_27931

variable (x : ℝ)

theorem simplify_expression :
  2 * x * (4 * x^2 - 3 * x + 1) - 4 * (2 * x^2 - 3 * x + 5) =
  8 * x^3 - 14 * x^2 + 14 * x - 20 := 
  sorry

end simplify_expression_l279_27931


namespace coin_prob_not_unique_l279_27963

theorem coin_prob_not_unique (p : ℝ) (w : ℝ) (h1 : 0 ≤ p ∧ p ≤ 1) (h2 : w = 144 / 625) :
  ¬ ∃! p, (∃ w, w = 10 * p^3 * (1 - p)^2 ∧ w = 144 / 625) :=
by
  sorry

end coin_prob_not_unique_l279_27963


namespace slant_height_base_plane_angle_l279_27939

noncomputable def angle_between_slant_height_and_base_plane (R : ℝ) : ℝ :=
  Real.arcsin ((Real.sqrt 13 - 1) / 3)

theorem slant_height_base_plane_angle (R : ℝ) (h : R = R) : angle_between_slant_height_and_base_plane R = Real.arcsin ((Real.sqrt 13 - 1) / 3) :=
by
  -- Here we assume that the mathematical conditions and transformations hold true.
  -- According to the solution steps provided:
  -- We found that γ = arcsin ((sqrt(13) - 1) / 3)
  sorry

end slant_height_base_plane_angle_l279_27939


namespace calculate_final_amount_l279_27937

def calculate_percentage (percentage : ℝ) (amount : ℝ) : ℝ :=
  percentage * amount

theorem calculate_final_amount :
  let A := 3000
  let B := 0.20
  let C := 0.35
  let D := 0.05
  D * (C * (B * A)) = 10.50 := by
    sorry

end calculate_final_amount_l279_27937


namespace polynomial_mod_p_zero_l279_27987

def is_zero_mod_p (p : ℕ) [Fact (Nat.Prime p)] (f : (List ℕ → ℤ)) : Prop :=
  ∀ (x : List ℕ), f x % p = 0

theorem polynomial_mod_p_zero
  (p : ℕ) [Fact (Nat.Prime p)]
  (n : ℕ) 
  (f : (List ℕ → ℤ)) 
  (h : ∀ (x : List ℕ), f x % p = 0) 
  (g : (List ℕ → ℤ)) :
  (∀ (x : List ℕ), g x % p = 0) := sorry

end polynomial_mod_p_zero_l279_27987


namespace inequality_solution_l279_27986

theorem inequality_solution (x : ℝ) : 
  (3 - (1 / (3 * x + 4)) < 5) ↔ (x < -4 / 3) := 
by
  sorry

end inequality_solution_l279_27986

import Mathlib

namespace NUMINAMATH_GPT_number_of_possible_a_values_l89_8953

-- Define the function f(x)
def f (a x : ℝ) := abs (x + 1) + abs (a * x + 1)

-- Define the condition for the minimum value
def minimum_value_of_f (a : ℝ) := ∃ x : ℝ, f a x = (3 / 2)

-- The proof problem statement
theorem number_of_possible_a_values : 
  (∃ (a1 a2 a3 a4 : ℝ),
    minimum_value_of_f a1 ∧
    minimum_value_of_f a2 ∧
    minimum_value_of_f a3 ∧
    minimum_value_of_f a4 ∧
    a1 ≠ a2 ∧ a1 ≠ a3 ∧ a1 ≠ a4 ∧ a2 ≠ a3 ∧ a2 ≠ a4 ∧ a3 ≠ a4) :=
sorry

end NUMINAMATH_GPT_number_of_possible_a_values_l89_8953


namespace NUMINAMATH_GPT_Q_cannot_be_log_x_l89_8951

def P : Set ℝ := {y | y ≥ 0}

theorem Q_cannot_be_log_x (Q : Set ℝ) :
  (P ∩ Q = Q) → Q ≠ {y | ∃ x, y = Real.log x} :=
by
  sorry

end NUMINAMATH_GPT_Q_cannot_be_log_x_l89_8951


namespace NUMINAMATH_GPT_david_initial_money_l89_8960

theorem david_initial_money (S X : ℕ) (h1 : S - 800 = 500) (h2 : X = S + 500) : X = 1800 :=
by
  sorry

end NUMINAMATH_GPT_david_initial_money_l89_8960


namespace NUMINAMATH_GPT_largest_band_members_l89_8980

theorem largest_band_members
  (p q m : ℕ)
  (h1 : p * q + 3 = m)
  (h2 : (q + 1) * (p + 2) = m)
  (h3 : m < 120) :
  m = 119 :=
sorry

end NUMINAMATH_GPT_largest_band_members_l89_8980


namespace NUMINAMATH_GPT_stock_AB_increase_factor_l89_8901

-- Define the conditions as mathematical terms
def stock_A_initial := 300
def stock_B_initial := 300
def stock_C_initial := 300
def stock_C_final := stock_C_initial / 2
def total_final := 1350
def AB_combined_initial := stock_A_initial + stock_B_initial
def AB_combined_final := total_final - stock_C_final

-- The statement to prove that the factor by which stocks A and B increased in value is 2.
theorem stock_AB_increase_factor :
  AB_combined_final / AB_combined_initial = 2 :=
  by
    sorry

end NUMINAMATH_GPT_stock_AB_increase_factor_l89_8901


namespace NUMINAMATH_GPT_find_cashew_kilos_l89_8954

variables (x : ℕ)

def cashew_cost_per_kilo := 210
def peanut_cost_per_kilo := 130
def total_weight := 5
def peanuts_weight := 2
def avg_price_per_kilo := 178

-- Given conditions
def cashew_total_cost := cashew_cost_per_kilo * x
def peanut_total_cost := peanut_cost_per_kilo * peanuts_weight
def total_price := total_weight * avg_price_per_kilo

theorem find_cashew_kilos (h1 : cashew_total_cost + peanut_total_cost = total_price) : x = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_cashew_kilos_l89_8954


namespace NUMINAMATH_GPT_ribbon_each_box_fraction_l89_8930

-- Define the conditions
def total_ribbon_used : ℚ := 5 / 12
def number_of_boxes : ℕ := 5
def ribbon_per_box : ℚ := total_ribbon_used / number_of_boxes

-- Statement to be proved
theorem ribbon_each_box_fraction :
  ribbon_per_box = 1 / 12 :=
  sorry

end NUMINAMATH_GPT_ribbon_each_box_fraction_l89_8930


namespace NUMINAMATH_GPT_part1_part2_l89_8925

noncomputable def probability_A_receives_one_red_envelope : ℚ :=
  sorry

theorem part1 (P_A1 : ℚ) (P_not_A1 : ℚ) (P_A2 : ℚ) (P_not_A2 : ℚ) :
  P_A1 = 1/3 ∧ P_not_A1 = 2/3 ∧ P_A2 = 1/3 ∧ P_not_A2 = 2/3 →
  probability_A_receives_one_red_envelope = 4/9 :=
sorry

noncomputable def probability_B_receives_at_least_10_yuan : ℚ :=
  sorry

theorem part2 (P_B1 : ℚ) (P_not_B1 : ℚ) (P_B2 : ℚ) (P_not_B2 : ℚ) (P_B3 : ℚ) (P_not_B3 : ℚ) :
  P_B1 = 1/3 ∧ P_not_B1 = 2/3 ∧ P_B2 = 1/3 ∧ P_not_B2 = 2/3 ∧ P_B3 = 1/3 ∧ P_not_B3 = 2/3 →
  probability_B_receives_at_least_10_yuan = 11/27 :=
sorry

end NUMINAMATH_GPT_part1_part2_l89_8925


namespace NUMINAMATH_GPT_lunch_break_duration_l89_8914

/-- Define the total recess time as a sum of two 15-minute breaks and one 20-minute break. -/
def total_recess_time : ℕ := 15 + 15 + 20

/-- Define the total time spent outside of class. -/
def total_outside_class_time : ℕ := 80

/-- Prove that the lunch break is 30 minutes long. -/
theorem lunch_break_duration : total_outside_class_time - total_recess_time = 30 :=
by
  sorry

end NUMINAMATH_GPT_lunch_break_duration_l89_8914


namespace NUMINAMATH_GPT_geometric_sequence_divisible_by_ten_million_l89_8988

theorem geometric_sequence_divisible_by_ten_million 
  (a1 a2 : ℝ)
  (h1 : a1 = 1 / 2)
  (h2 : a2 = 50) :
  ∀ n : ℕ, (n ≥ 5) → (∃ k : ℕ, (a1 * (a2 / a1)^(n - 1)) = k * 10^7) :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_divisible_by_ten_million_l89_8988


namespace NUMINAMATH_GPT_possible_c_value_l89_8923

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) (c : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

theorem possible_c_value (a b c : ℝ) 
  (h1 : f (-1) a b c = f (-2) a b c) 
  (h2 : f (-2) a b c = f (-3) a b c) 
  (h3 : 0 ≤ f (-1) a b c) 
  (h4 : f (-1) a b c ≤ 3) : 
  6 ≤ c ∧ c ≤ 9 := sorry

end NUMINAMATH_GPT_possible_c_value_l89_8923


namespace NUMINAMATH_GPT_negate_exists_real_l89_8970

theorem negate_exists_real (h : ¬ ∃ x : ℝ, x^2 - 2 ≤ 0) : ∀ x : ℝ, x^2 - 2 > 0 :=
by
  sorry

end NUMINAMATH_GPT_negate_exists_real_l89_8970


namespace NUMINAMATH_GPT_compare_P_Q_l89_8936

-- Define the structure of the number a with 2010 digits of 1
def a := 10^2010 - 1

-- Define P and Q based on a
def P := 24 * a^2
def Q := 24 * a^2 + 4 * a

-- Define the theorem to compare P and Q
theorem compare_P_Q : Q > P := by
  sorry

end NUMINAMATH_GPT_compare_P_Q_l89_8936


namespace NUMINAMATH_GPT_sum_of_variables_l89_8959

theorem sum_of_variables (x y z : ℝ) (h : x^2 + y^2 + z^2 - 2 * x + 4 * y - 6 * z + 14 = 0) : x + y + z = 2 :=
sorry

end NUMINAMATH_GPT_sum_of_variables_l89_8959


namespace NUMINAMATH_GPT_arithmetic_sequence_common_difference_l89_8972

theorem arithmetic_sequence_common_difference
  (a₁ d : ℝ)
  (h_a4 : a₁ + 3 * d = -2)
  (h_sum : 10 * a₁ + 45 * d = 65) :
  d = 17 / 3 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_common_difference_l89_8972


namespace NUMINAMATH_GPT_installation_rates_l89_8977

variables (units_total : ℕ) (teamA_units : ℕ) (teamB_units : ℕ) (team_units_gap : ℕ)
variables (rate_teamA : ℕ) (rate_teamB : ℕ)

-- Conditions
def conditions : Prop :=
  units_total = 140 ∧
  teamA_units = 80 ∧
  teamB_units = units_total - teamA_units ∧
  team_units_gap = 5 ∧
  rate_teamA = rate_teamB + team_units_gap

-- Question to prove
def solution : Prop :=
  rate_teamB = 15 ∧ rate_teamA = 20

-- Statement of the proof
theorem installation_rates (units_total : ℕ) (teamA_units : ℕ) (teamB_units : ℕ) (team_units_gap : ℕ) (rate_teamA : ℕ) (rate_teamB : ℕ) :
  conditions units_total teamA_units teamB_units team_units_gap rate_teamA rate_teamB →
  solution rate_teamA rate_teamB :=
sorry

end NUMINAMATH_GPT_installation_rates_l89_8977


namespace NUMINAMATH_GPT_base_circumference_of_cone_l89_8992

theorem base_circumference_of_cone (r : ℝ) (theta : ℝ) (C : ℝ) 
  (h_radius : r = 6)
  (h_theta : theta = 180)
  (h_C : C = 2 * Real.pi * r) :
  (theta / 360) * C = 6 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_base_circumference_of_cone_l89_8992


namespace NUMINAMATH_GPT_subtract_500_from_sum_of_calculations_l89_8957

theorem subtract_500_from_sum_of_calculations (x : ℕ) (h : 423 - x = 421) : 
  (421 + 423 * x) - 500 = 767 := 
by
  sorry

end NUMINAMATH_GPT_subtract_500_from_sum_of_calculations_l89_8957


namespace NUMINAMATH_GPT_percentage_of_loss_l89_8987

theorem percentage_of_loss
    (CP SP : ℝ)
    (h1 : CP = 1200)
    (h2 : SP = 1020)
    (Loss : ℝ)
    (h3 : Loss = CP - SP)
    (Percentage_of_Loss : ℝ)
    (h4 : Percentage_of_Loss = (Loss / CP) * 100) :
  Percentage_of_Loss = 15 := by
  sorry

end NUMINAMATH_GPT_percentage_of_loss_l89_8987


namespace NUMINAMATH_GPT_proposition_form_l89_8947

-- Definitions based on the conditions
def p : Prop := (12 % 4 = 0)
def q : Prop := (12 % 3 = 0)

-- Problem statement to prove
theorem proposition_form : p ∧ q :=
by
  sorry

end NUMINAMATH_GPT_proposition_form_l89_8947


namespace NUMINAMATH_GPT_triangle_altitude_l89_8942

theorem triangle_altitude (b : ℕ) (h : ℕ) (area : ℕ) (h_area : area = 800) (h_base : b = 40) (h_formula : area = (1 / 2) * b * h) : h = 40 :=
by
  sorry

end NUMINAMATH_GPT_triangle_altitude_l89_8942


namespace NUMINAMATH_GPT_isosceles_trapezoid_ratio_l89_8943

theorem isosceles_trapezoid_ratio (a b d_E d_G : ℝ) (h1 : a > b)
  (h2 : (1/2) * b * d_G = 3) (h3 : (1/2) * a * d_E = 7)
  (h4 : (1/2) * (a + b) * (d_E + d_G) = 24) :
  (a / b) = 7 / 3 :=
sorry

end NUMINAMATH_GPT_isosceles_trapezoid_ratio_l89_8943


namespace NUMINAMATH_GPT_car_pedestrian_speed_ratio_l89_8985

theorem car_pedestrian_speed_ratio
  (L : ℝ) -- Length of the bridge
  (v_p v_c : ℝ) -- Speed of pedestrian and car
  (h1 : (4 / 9) * L / v_p = (5 / 9) * L / v_p + (5 / 9) * L / v_c) -- Initial meet at bridge start
  (h2 : (4 / 9) * L / v_p = (8 / 9) * L / v_c) -- If pedestrian continues to walk
  : v_c / v_p = 9 :=
sorry

end NUMINAMATH_GPT_car_pedestrian_speed_ratio_l89_8985


namespace NUMINAMATH_GPT_gcd_largest_value_l89_8999

/-- Given two positive integers x and y such that x + y = 780,
    this definition states that the largest possible value of gcd(x, y) is 390. -/
theorem gcd_largest_value (x y : ℕ) (hx : x > 0) (hy : y > 0) (h : x + y = 780) : ∃ d, d = Nat.gcd x y ∧ d = 390 :=
sorry

end NUMINAMATH_GPT_gcd_largest_value_l89_8999


namespace NUMINAMATH_GPT_number_of_boys_in_class_l89_8973

theorem number_of_boys_in_class (B : ℕ) (G : ℕ) (hG : G = 10) (h_combinations : (G * B * (B - 1)) / 2 = 1050) :
    B = 15 :=
by
  sorry

end NUMINAMATH_GPT_number_of_boys_in_class_l89_8973


namespace NUMINAMATH_GPT_Alan_collected_48_shells_l89_8950

def Laurie_shells : ℕ := 36
def Ben_shells : ℕ := Laurie_shells / 3
def Alan_shells : ℕ := 4 * Ben_shells

theorem Alan_collected_48_shells :
  Alan_shells = 48 :=
by
  sorry

end NUMINAMATH_GPT_Alan_collected_48_shells_l89_8950


namespace NUMINAMATH_GPT_min_sum_of_factors_l89_8971

theorem min_sum_of_factors (a b c : ℕ) (h1 : a * b * c = 2310) : a + b + c ≥ 42 :=
sorry

end NUMINAMATH_GPT_min_sum_of_factors_l89_8971


namespace NUMINAMATH_GPT_group_purchase_cheaper_l89_8946

-- Define the initial conditions
def initial_price : ℕ := 10
def bulk_price : ℕ := 7
def delivery_cost : ℕ := 100
def group_size : ℕ := 50

-- Define the costs for individual and group purchases
def individual_cost : ℕ := initial_price
def group_cost : ℕ := bulk_price + (delivery_cost / group_size)

-- Statement to prove: cost per participant in a group purchase is less than cost per participant in individual purchases
theorem group_purchase_cheaper : group_cost < individual_cost := by
  sorry

end NUMINAMATH_GPT_group_purchase_cheaper_l89_8946


namespace NUMINAMATH_GPT_water_usage_difference_l89_8967

variable (a b : ℝ)
variable (ha : a ≠ 0)
variable (hb : b ≠ 0)
variable (ha_plus_4 : a + 4 ≠ 0)

theorem water_usage_difference :
  b / a - b / (a + 4) = 4 * b / (a * (a + 4)) :=
by
  sorry

end NUMINAMATH_GPT_water_usage_difference_l89_8967


namespace NUMINAMATH_GPT_miss_hilt_apples_l89_8966

theorem miss_hilt_apples (h : ℕ) (a_per_hour : ℕ) (total_apples : ℕ) 
    (H1 : a_per_hour = 5) (H2 : total_apples = 15) (H3 : total_apples = h * a_per_hour) : 
  h = 3 :=
by
  sorry

end NUMINAMATH_GPT_miss_hilt_apples_l89_8966


namespace NUMINAMATH_GPT_value_of_3_over_x_l89_8986

theorem value_of_3_over_x (x : ℝ) (hx : 1 - 6 / x + 9 / x^2 - 4 / x^3 = 0) : 
  (3 / x = 3 ∨ 3 / x = 3 / 4) :=
  sorry

end NUMINAMATH_GPT_value_of_3_over_x_l89_8986


namespace NUMINAMATH_GPT_correct_number_of_students_answered_both_l89_8978

def students_enrolled := 25
def answered_q1_correctly := 22
def answered_q2_correctly := 20
def not_taken_test := 3

def students_answered_both_questions_correctly : Nat :=
  let students_took_test := students_enrolled - not_taken_test
  let b := answered_q2_correctly
  b

theorem correct_number_of_students_answered_both :
  students_answered_both_questions_correctly = answered_q2_correctly :=
by {
  -- this space is for the proof, we are currently not required to provide it
  sorry
}

end NUMINAMATH_GPT_correct_number_of_students_answered_both_l89_8978


namespace NUMINAMATH_GPT_unique_two_digit_number_l89_8908

-- Definition of the problem in Lean
def is_valid_number (n : ℕ) : Prop :=
  n % 4 = 1 ∧ n % 17 = 1 ∧ 10 ≤ n ∧ n ≤ 99

theorem unique_two_digit_number :
  ∃! n : ℕ, is_valid_number n ∧ n = 69 :=
by
  sorry

end NUMINAMATH_GPT_unique_two_digit_number_l89_8908


namespace NUMINAMATH_GPT_coordinate_minimizes_z_l89_8940

-- Definitions for conditions
def in_first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

def equation_holds (x y : ℝ) : Prop := (1 / x) + (1 / (2 * y)) + (3 / (2 * x * y)) = 1

def z_def (x y : ℝ) : ℝ := x * y

-- Statement
theorem coordinate_minimizes_z (x y : ℝ) (h1 : in_first_quadrant x y) (h2 : equation_holds x y) :
    z_def x y = 9 / 2 ∧ (x = 3 ∧ y = 3 / 2) :=
    sorry

end NUMINAMATH_GPT_coordinate_minimizes_z_l89_8940


namespace NUMINAMATH_GPT_focus_of_curve_is_4_0_l89_8910

noncomputable def is_focus (p : ℝ × ℝ) (curve : ℝ × ℝ → Prop) : Prop :=
  ∃ c : ℝ, ∀ x y : ℝ, curve (x, y) ↔ (y^2 = -16 * c * (x - 4))

def curve (p : ℝ × ℝ) : Prop := p.2^2 = -16 * p.1 + 64

theorem focus_of_curve_is_4_0 : is_focus (4, 0) curve :=
by
sorry

end NUMINAMATH_GPT_focus_of_curve_is_4_0_l89_8910


namespace NUMINAMATH_GPT_number_of_students_l89_8916

theorem number_of_students 
  (N : ℕ)
  (avg_age : ℕ → ℕ)
  (h1 : avg_age N = 15)
  (h2 : avg_age 5 = 12)
  (h3 : avg_age 9 = 16)
  (h4 : N = 15 ∧ avg_age 1 = 21) : 
  N = 15 :=
by
  sorry

end NUMINAMATH_GPT_number_of_students_l89_8916


namespace NUMINAMATH_GPT_find_reciprocal_sum_of_roots_l89_8915

theorem find_reciprocal_sum_of_roots
  {x₁ x₂ : ℝ}
  (h1 : 5 * x₁ ^ 2 - 3 * x₁ - 2 = 0)
  (h2 : 5 * x₂ ^ 2 - 3 * x₂ - 2 = 0)
  (h_diff : x₁ ≠ x₂) :
  (1 / x₁ + 1 / x₂) = -3 / 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_reciprocal_sum_of_roots_l89_8915


namespace NUMINAMATH_GPT_vertex_angle_measure_l89_8991

-- Definitions for Lean Proof
def isosceles_triangle (α β γ : ℝ) : Prop := (α = β) ∨ (α = γ) ∨ (β = γ)
def exterior_angle (interior exterior : ℝ) : Prop := interior + exterior = 180

-- Conditions from the problem
variables (α β γ : ℝ)
variable (ext_angle : ℝ := 110)

-- Lean 4 statement: The measure of the vertex angle is 70° or 40°
theorem vertex_angle_measure :
  isosceles_triangle α β γ ∧
  (exterior_angle γ ext_angle ∨ exterior_angle α ext_angle ∨ exterior_angle β ext_angle) →
  (γ = 70 ∨ γ = 40) :=
by
  sorry

end NUMINAMATH_GPT_vertex_angle_measure_l89_8991


namespace NUMINAMATH_GPT_minimum_money_lost_l89_8998

-- Define the conditions and setup the problem

def check_amount : ℕ := 1270
def T_used (F : ℕ) : Σ' T, (T = F + 1 ∨ T = F - 1) :=
sorry

def money_used (T F : ℕ) : ℕ := 10 * T + 50 * F

def total_bills_used (T F : ℕ) : Prop := T + F = 15

theorem minimum_money_lost : (∃ T F, (T = F + 1 ∨ T = F - 1) ∧ T + F = 15 ∧ (check_amount - (10 * T + 50 * F) = 800)) :=
sorry

end NUMINAMATH_GPT_minimum_money_lost_l89_8998


namespace NUMINAMATH_GPT_yearly_payment_split_evenly_l89_8938

def monthly_cost : ℤ := 14
def split_cost (cost : ℤ) := cost / 2
def total_yearly_cost (monthly_payment : ℤ) := monthly_payment * 12

theorem yearly_payment_split_evenly (h : split_cost monthly_cost = 7) :
  total_yearly_cost (split_cost monthly_cost) = 84 :=
by
  -- Here we use the hypothesis h which simplifies the proof.
  sorry

end NUMINAMATH_GPT_yearly_payment_split_evenly_l89_8938


namespace NUMINAMATH_GPT_race_distance_l89_8911

theorem race_distance (Va Vb Vc : ℝ) (D : ℝ) :
    (Va / Vb = 10 / 9) →
    (Va / Vc = 80 / 63) →
    (Vb / Vc = 8 / 7) →
    (D - 100) / D = 7 / 8 → 
    D = 700 :=
by
  intros h1 h2 h3 h4 
  sorry

end NUMINAMATH_GPT_race_distance_l89_8911


namespace NUMINAMATH_GPT_exists_large_p_l89_8919

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.sin (Real.pi * x)

theorem exists_large_p (d : ℝ) (h : d > 0) : ∃ p : ℝ, ∀ x : ℝ, |f (x + p) - f x| < d ∧ ∃ M : ℝ, M > 0 ∧ p > M :=
by {
  sorry
}

end NUMINAMATH_GPT_exists_large_p_l89_8919


namespace NUMINAMATH_GPT_length_of_fence_l89_8989

theorem length_of_fence (side_length : ℕ) (h : side_length = 28) : 4 * side_length = 112 :=
by
  sorry

end NUMINAMATH_GPT_length_of_fence_l89_8989


namespace NUMINAMATH_GPT_stickers_left_after_giving_away_l89_8935

/-- Willie starts with 36 stickers and gives 7 to Emily. 
    We want to prove that Willie ends up with 29 stickers. -/
theorem stickers_left_after_giving_away (init_stickers : ℕ) (given_away : ℕ) (end_stickers : ℕ) : 
  init_stickers = 36 ∧ given_away = 7 → end_stickers = init_stickers - given_away → end_stickers = 29 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_stickers_left_after_giving_away_l89_8935


namespace NUMINAMATH_GPT_trapezoid_circle_tangent_ratio_l89_8924

/-- Given trapezoid EFGH with specified side lengths,
    where EF is parallel to GH, and a circle with
    center Q on EF tangent to FG and HE,
    the ratio EQ : QF is 12 : 37. -/
theorem trapezoid_circle_tangent_ratio :
  ∀ (EF FG GH HE : ℝ) (EQ QF : ℝ),
  EF = 40 → FG = 25 → GH = 12 → HE = 35 →
  ∃ (Q : ℝ) (EQ QF : ℝ),
  EQ + QF = EF ∧ EQ / QF = 12 / 37 ∧ gcd 12 37 = 1 :=
by
  sorry

end NUMINAMATH_GPT_trapezoid_circle_tangent_ratio_l89_8924


namespace NUMINAMATH_GPT_range_of_x_l89_8952

noncomputable def f (x : ℝ) : ℝ := sorry -- Define f to satisfy given conditions later

theorem range_of_x (hf_odd : ∀ x : ℝ, f (-x) = - f x)
                   (hf_inc_mono_neg : ∀ x y : ℝ, x ≤ y → y ≤ 0 → f x ≤ f y)
                   (h_ineq : f 1 + f (Real.log x - 2) < 0) : (0 < x) ∧ (x < 10) :=
by
  sorry

end NUMINAMATH_GPT_range_of_x_l89_8952


namespace NUMINAMATH_GPT_scientific_notation_of_510000000_l89_8949

theorem scientific_notation_of_510000000 :
  (510000000 : ℝ) = 5.1 * 10^8 := 
sorry

end NUMINAMATH_GPT_scientific_notation_of_510000000_l89_8949


namespace NUMINAMATH_GPT_min_colors_for_grid_coloring_l89_8982

theorem min_colors_for_grid_coloring : ∃c : ℕ, c = 4 ∧ (∀ (color : ℕ × ℕ → ℕ), 
  (∀ i j : ℕ, i < 5 ∧ j < 5 → 
     ((i < 4 → color (i, j) ≠ color (i+1, j+1)) ∧ 
      (j < 4 → color (i, j) ≠ color (i+1, j-1))) ∧ 
     ((i > 0 → color (i, j) ≠ color (i-1, j-1)) ∧ 
      (j > 0 → color (i, j) ≠ color (i-1, j+1)))) → 
  c = 4) :=
sorry

end NUMINAMATH_GPT_min_colors_for_grid_coloring_l89_8982


namespace NUMINAMATH_GPT_domain_of_f_univ_l89_8948

noncomputable def f (x : ℝ) : ℝ := (x^2 - 2 * x + 1)^(1 / 3) + (9 - x^2)^(1 / 3)

theorem domain_of_f_univ : ∀ x : ℝ, true :=
by
  intro x
  sorry

end NUMINAMATH_GPT_domain_of_f_univ_l89_8948


namespace NUMINAMATH_GPT_find_x_floor_l89_8905

theorem find_x_floor : ∃ (x : ℚ), (⌊x⌋ : ℚ) + x = 29 / 4 ∧ x = 29 / 4 := 
by
  sorry

end NUMINAMATH_GPT_find_x_floor_l89_8905


namespace NUMINAMATH_GPT_b_is_square_of_positive_integer_l89_8933

theorem b_is_square_of_positive_integer 
  (a b : ℕ) (ha : 0 < a) (hb : 0 < b) 
  (h : b^2 = a^2 + ab + b) : 
  ∃ k : ℕ, b = k^2 := 
by 
  sorry

end NUMINAMATH_GPT_b_is_square_of_positive_integer_l89_8933


namespace NUMINAMATH_GPT_calculate_A_plus_B_l89_8993

theorem calculate_A_plus_B (A B : ℝ) (h1 : A ≠ B) 
  (h2 : ∀ x : ℝ, (A * (B * x^2 + A * x + 1)^2 + B * (B * x^2 + A * x + 1) + 1) 
                - (B * (A * x^2 + B * x + 1)^2 + A * (A * x^2 + B * x + 1) + 1) 
                = x^4 + 5 * x^3 + x^2 - 4 * x) : A + B = 0 :=
by
  sorry

end NUMINAMATH_GPT_calculate_A_plus_B_l89_8993


namespace NUMINAMATH_GPT_area_square_field_l89_8997

-- Define the side length of the square
def side_length : ℕ := 12

-- Define the area of the square with the given side length
def area_of_square (side : ℕ) : ℕ := side * side

-- The theorem to state and prove
theorem area_square_field : area_of_square side_length = 144 :=
by
  sorry

end NUMINAMATH_GPT_area_square_field_l89_8997


namespace NUMINAMATH_GPT_inequality_solution_b_range_l89_8922

-- Given conditions
variables (a b : ℝ)

def condition1 : Prop := (1 - a < 0) ∧ (a = 3)
def condition2 : Prop := ∀ (x : ℝ), (3 * x^2 + b * x + 3) ≥ 0

-- Assertions to be proved
theorem inequality_solution (a : ℝ) (ha : condition1 a) : 
  ∀ (x : ℝ), (2 * x^2 + (2 - a) * x - a > 0) ↔ (x < -1 ∨ x > 3/2) :=
sorry

theorem b_range (a : ℝ) (hb : condition1 a) : 
  condition2 b ↔ (-6 ≤ b ∧ b ≤ 6) :=
sorry

end NUMINAMATH_GPT_inequality_solution_b_range_l89_8922


namespace NUMINAMATH_GPT_quotient_is_36_l89_8928

-- Conditions
def divisor := 85
def remainder := 26
def dividend := 3086

-- The Question and Answer (proof required)
theorem quotient_is_36 (quotient : ℕ) (h : dividend = (divisor * quotient) + remainder) : quotient = 36 := by 
  sorry

end NUMINAMATH_GPT_quotient_is_36_l89_8928


namespace NUMINAMATH_GPT_find_constants_l89_8912

theorem find_constants (A B C : ℝ) :
  (∀ x : ℝ, x ≠ 2 ∧ x ≠ 3 ∧ x ≠ 5 → (x^2 - 7) / ((x - 2) * (x - 3) * (x - 5)) = A / (x - 2) + B / (x - 3) + C / (x - 5))
  ↔ (A = -1 ∧ B = -1 ∧ C = 3) :=
by
  sorry

end NUMINAMATH_GPT_find_constants_l89_8912


namespace NUMINAMATH_GPT_base_of_isosceles_triangle_l89_8921

theorem base_of_isosceles_triangle (a b side equil_perim iso_perim : ℕ) 
  (h1 : equil_perim = 60)
  (h2 : 3 * side = equil_perim)
  (h3 : iso_perim = 50)
  (h4 : 2 * side + b = iso_perim)
  : b = 10 :=
by
  sorry

end NUMINAMATH_GPT_base_of_isosceles_triangle_l89_8921


namespace NUMINAMATH_GPT_ruth_gave_janet_53_stickers_l89_8902

-- Definitions: Janet initially has 3 stickers, after receiving more from Ruth, she has 56 stickers in total.
def janet_initial : ℕ := 3
def janet_total : ℕ := 56

-- The statement to prove: Ruth gave Janet 53 stickers.
def stickers_from_ruth (initial: ℕ) (total: ℕ) : ℕ :=
  total - initial

theorem ruth_gave_janet_53_stickers : stickers_from_ruth janet_initial janet_total = 53 :=
by sorry

end NUMINAMATH_GPT_ruth_gave_janet_53_stickers_l89_8902


namespace NUMINAMATH_GPT_latest_first_pump_time_l89_8920

theorem latest_first_pump_time 
  (V : ℝ) -- Volume of the pool
  (x y : ℝ) -- Productivity of first and second pumps respectively
  (t : ℝ) -- Time of operation of the first pump until the second pump is turned on
  (h1 : 2*x + 2*y = V/2) -- Condition from 10 AM to 12 PM
  (h2 : 5*x + 5*y = V/2) -- Condition from 12 PM to 5 PM
  (h3 : t*x + 2*x + 2*y = V/2) -- Condition for early morning until 12 PM
  (hx_pos : 0 < x) -- Assume productivity of first pump is positive
  (hy_pos : 0 < y) -- Assume productivity of second pump is positive
  : t ≥ 3 :=
by
  -- The proof goes here...
  sorry

end NUMINAMATH_GPT_latest_first_pump_time_l89_8920


namespace NUMINAMATH_GPT_not_all_crows_gather_on_one_tree_l89_8913

theorem not_all_crows_gather_on_one_tree :
  ∀ (crows : Fin 6 → ℕ), 
  (∀ i, crows i = 1) →
  (∀ t1 t2, abs (t1 - t2) = 1 → crows t1 = crows t1 - 1 ∧ crows t2 = crows t2 + 1) →
  ¬(∃ i, crows i = 6 ∧ (∀ j ≠ i, crows j = 0)) :=
by
  sorry

end NUMINAMATH_GPT_not_all_crows_gather_on_one_tree_l89_8913


namespace NUMINAMATH_GPT_fibonacci_sum_of_squares_l89_8932

def fibonacci : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fibonacci (n + 1) + fibonacci n

theorem fibonacci_sum_of_squares (n : ℕ) (hn : n ≥ 1) :
  (Finset.range n).sum (λ i => (fibonacci (i + 1))^2) = fibonacci n * fibonacci (n + 1) :=
sorry

end NUMINAMATH_GPT_fibonacci_sum_of_squares_l89_8932


namespace NUMINAMATH_GPT_abs_equation_solution_l89_8955

theorem abs_equation_solution (x : ℝ) (h : |x - 3| = 2 * x + 4) : x = -1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_abs_equation_solution_l89_8955


namespace NUMINAMATH_GPT_triangle_area_is_rational_l89_8994

-- Definition of the area of a triangle given vertices with integer coordinates
def triangle_area (x1 x2 x3 y1 y2 y3 : ℤ) : ℚ :=
0.5 * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

-- The theorem stating that the area of a triangle formed by points with integer coordinates is rational
theorem triangle_area_is_rational (x1 x2 x3 y1 y2 y3 : ℤ) :
  ∃ (area : ℚ), area = triangle_area x1 x2 x3 y1 y2 y3 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_is_rational_l89_8994


namespace NUMINAMATH_GPT_intersection_claim_union_claim_l89_8965

def A : Set ℝ := {x | (x - 2) * (x + 5) < 0}
def B : Set ℝ := {x | x^2 - 2 * x - 3 ≥ 0}
def U : Set ℝ := Set.univ

-- Claim 1: Prove that A ∩ B = {x | -5 < x ∧ x ≤ -1}
theorem intersection_claim : A ∩ B = {x | -5 < x ∧ x ≤ -1} :=
by
  sorry

-- Claim 2: Prove that A ∪ (U \ B) = {x | -5 < x ∧ x < 3}
theorem union_claim : A ∪ (U \ B) = {x | -5 < x ∧ x < 3} :=
by
  sorry

end NUMINAMATH_GPT_intersection_claim_union_claim_l89_8965


namespace NUMINAMATH_GPT_complement_of_A_l89_8961

open Set

theorem complement_of_A (U : Set ℕ) (A : Set ℕ) (hU : U = {1, 2, 3, 4, 5, 6}) (hA : A = {3, 4, 5}) :
  (U \ A) = {1, 2, 6} :=
by
  sorry

end NUMINAMATH_GPT_complement_of_A_l89_8961


namespace NUMINAMATH_GPT_total_age_l89_8990

theorem total_age (a b c : ℕ) (h1 : a = b + 2) (h2 : b = 2 * c) (h3 : b = 8) : a + b + c = 22 :=
by
  sorry

end NUMINAMATH_GPT_total_age_l89_8990


namespace NUMINAMATH_GPT_problem_solution_l89_8984

noncomputable def f (x : ℝ) : ℝ := x / (Real.cos x)

variables (x1 x2 x3 : ℝ)

axiom a1 : |x1| < (Real.pi / 2)
axiom a2 : |x2| < (Real.pi / 2)
axiom a3 : |x3| < (Real.pi / 2)

axiom h1 : f x1 + f x2 ≥ 0
axiom h2 : f x2 + f x3 ≥ 0
axiom h3 : f x3 + f x1 ≥ 0

theorem problem_solution : f (x1 + x2 + x3) ≥ 0 := sorry

end NUMINAMATH_GPT_problem_solution_l89_8984


namespace NUMINAMATH_GPT_Anita_should_buy_more_cartons_l89_8975

def Anita_needs (total_needed : ℕ) : Prop :=
total_needed = 26

def Anita_has (strawberries blueberries : ℕ) : Prop :=
strawberries = 10 ∧ blueberries = 9

def additional_cartons (total_needed strawberries blueberries : ℕ) : ℕ :=
total_needed - (strawberries + blueberries)

theorem Anita_should_buy_more_cartons :
  ∀ (total_needed strawberries blueberries : ℕ),
    Anita_needs total_needed →
    Anita_has strawberries blueberries →
    additional_cartons total_needed strawberries blueberries = 7 :=
by
  intros total_needed strawberries blueberries Hneeds Hhas
  sorry

end NUMINAMATH_GPT_Anita_should_buy_more_cartons_l89_8975


namespace NUMINAMATH_GPT_superior_sequences_count_l89_8934

noncomputable def number_of_superior_sequences (n : ℕ) : ℕ :=
  Nat.choose (2 * n + 1) (n + 1) * 2^n

theorem superior_sequences_count (n : ℕ) (h : 2 ≤ n) 
  (x : Fin (n + 1) → ℤ)
  (h1 : ∀ i, 0 ≤ i ∧ i ≤ n → |x i| ≤ n)
  (h2 : ∀ i j, 0 ≤ i ∧ i < j ∧ j ≤ n → x i ≠ x j)
  (h3 : ∀ (i j k : Nat), 0 ≤ i ∧ i < j ∧ j < k ∧ k ≤ n → 
    max (|x k - x i|) (|x k - x j|) = 
    (|x i - x j| + |x j - x k| + |x k - x i|) / 2) :
  number_of_superior_sequences n = Nat.choose (2 * n + 1) (n + 1) * 2^n :=
sorry

end NUMINAMATH_GPT_superior_sequences_count_l89_8934


namespace NUMINAMATH_GPT_probability_of_event_l89_8927

open Set Real

noncomputable def probability_event_interval (x : ℝ) : Prop :=
  1 ≤ 2 * x - 1 ∧ 2 * x - 1 ≤ 3

noncomputable def interval := Icc (0 : ℝ) (3 : ℝ)

noncomputable def event_probability := 1 / 3

theorem probability_of_event :
  ∀ x ∈ interval, probability_event_interval x → (event_probability) = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_event_l89_8927


namespace NUMINAMATH_GPT_Nancy_picked_l89_8909

def Alyssa_picked : ℕ := 42
def Total_picked : ℕ := 59

theorem Nancy_picked : Total_picked - Alyssa_picked = 17 := by
  sorry

end NUMINAMATH_GPT_Nancy_picked_l89_8909


namespace NUMINAMATH_GPT_find_ab_l89_8981

theorem find_ab (a b : ℤ) :
  (∀ x : ℤ, x^3 + a * x^2 + b * x + 5 % (x - 1) = 7) ∧ (∀ x : ℤ, x^3 + a * x^2 + b * x + 5 % (x + 1) = 9) →
  (a, b) = (3, -2) := 
by
  sorry

end NUMINAMATH_GPT_find_ab_l89_8981


namespace NUMINAMATH_GPT_num_non_fiction_books_l89_8918

-- Definitions based on the problem conditions
def num_fiction_configurations : ℕ := 24
def total_configurations : ℕ := 36

-- Non-computable definition for factorial
noncomputable def factorial (n : ℕ) : ℕ :=
if n = 0 then 1 else n * factorial (n - 1)

-- Theorem to prove the number of new non-fiction books
theorem num_non_fiction_books (n : ℕ) :
  num_fiction_configurations * factorial n = total_configurations → n = 2 :=
by
  sorry

end NUMINAMATH_GPT_num_non_fiction_books_l89_8918


namespace NUMINAMATH_GPT_sum_ages_l89_8968

theorem sum_ages (x : ℕ) (h_triple : True) (h_sons_age : ∀ a, a ∈ [16, 16, 16]) (h_beau_age : 42 = 42) :
  3 * (16 - x) = 42 - x → x = 3 := by
  sorry

end NUMINAMATH_GPT_sum_ages_l89_8968


namespace NUMINAMATH_GPT_sum_of_u_and_v_l89_8917

theorem sum_of_u_and_v (u v : ℤ) (h1 : 1 ≤ v) (h2 : v < u) (h3 : u^2 + v^2 = 500) : u + v = 20 := by
  sorry

end NUMINAMATH_GPT_sum_of_u_and_v_l89_8917


namespace NUMINAMATH_GPT_graphs_symmetric_y_axis_l89_8941

theorem graphs_symmetric_y_axis : ∀ (x : ℝ), (-x) ∈ { y | y = 3^(-x) } ↔ x ∈ { y | y = 3^x } :=
by
  intro x
  sorry

end NUMINAMATH_GPT_graphs_symmetric_y_axis_l89_8941


namespace NUMINAMATH_GPT_find_f_2010_l89_8939

noncomputable def f (a b α β : ℝ) (x : ℝ) :=
  a * Real.sin (Real.pi * x + α) + b * Real.cos (Real.pi * x + β) + 4

theorem find_f_2010 {a b α β : ℝ} (h : f a b α β 2009 = 5) : f a b α β 2010 = 3 :=
sorry

end NUMINAMATH_GPT_find_f_2010_l89_8939


namespace NUMINAMATH_GPT_pentagon_area_l89_8963

/-- This Lean statement represents the problem of finding the y-coordinate of vertex C
    in a pentagon with given vertex positions and specific area constraint. -/
theorem pentagon_area (y : ℝ) 
  (h_sym : true) -- The pentagon ABCDE has a vertical line of symmetry
  (h_A : (0, 0) = (0, 0)) -- A(0,0)
  (h_B : (0, 5) = (0, 5)) -- B(0, 5)
  (h_C : (3, y) = (3, y)) -- C(3, y)
  (h_D : (6, 5) = (6, 5)) -- D(6, 5)
  (h_E : (6, 0) = (6, 0)) -- E(6, 0)
  (h_area : 50 = 50) -- The total area of the pentagon is 50 square units
  : y = 35 / 3 :=
sorry

end NUMINAMATH_GPT_pentagon_area_l89_8963


namespace NUMINAMATH_GPT_sin_cos_sum_l89_8956

open Real

theorem sin_cos_sum : sin (47 : ℝ) * cos (43 : ℝ) + cos (47 : ℝ) * sin (43 : ℝ) = 1 :=
by
  sorry

end NUMINAMATH_GPT_sin_cos_sum_l89_8956


namespace NUMINAMATH_GPT_sum_of_three_integers_l89_8974

theorem sum_of_three_integers (a b c : ℕ) (h_pos: a > 0 ∧ b > 0 ∧ c > 0) 
  (h_diff: a ≠ b ∧ b ≠ c ∧ c ≠ a) (h_prod: a * b * c = 5^4) : a + b + c = 131 :=
sorry

end NUMINAMATH_GPT_sum_of_three_integers_l89_8974


namespace NUMINAMATH_GPT_obtuse_angle_half_in_first_quadrant_l89_8979

-- Define α to be an obtuse angle
variable {α : ℝ}

-- The main theorem we want to prove
theorem obtuse_angle_half_in_first_quadrant (h_obtuse : (π / 2) < α ∧ α < π) :
  0 < α / 2 ∧ α / 2 < π / 2 :=
  sorry

end NUMINAMATH_GPT_obtuse_angle_half_in_first_quadrant_l89_8979


namespace NUMINAMATH_GPT_no_int_solutions_x2_minus_3y2_eq_17_l89_8983

theorem no_int_solutions_x2_minus_3y2_eq_17 : 
  ∀ (x y : ℤ), (x^2 - 3 * y^2 ≠ 17) := 
by
  intros x y
  sorry

end NUMINAMATH_GPT_no_int_solutions_x2_minus_3y2_eq_17_l89_8983


namespace NUMINAMATH_GPT_rabbit_carrots_l89_8907

theorem rabbit_carrots (h_r h_f x : ℕ) (H1 : 5 * h_r = x) (H2 : 6 * h_f = x) (H3 : h_r = h_f + 2) : x = 60 :=
by
  sorry

end NUMINAMATH_GPT_rabbit_carrots_l89_8907


namespace NUMINAMATH_GPT_solution_of_inequality_l89_8906

-- Let us define the inequality and the solution set
def inequality (x : ℝ) := (x - 1)^2023 - 2^2023 * x^2023 ≤ x + 1
def solution_set (x : ℝ) := x ≥ -1

-- The theorem statement to prove that the solution set matches the inequality
theorem solution_of_inequality :
  {x : ℝ | inequality x} = {x : ℝ | solution_set x} := sorry

end NUMINAMATH_GPT_solution_of_inequality_l89_8906


namespace NUMINAMATH_GPT_ratio_female_male_l89_8903

theorem ratio_female_male (f m : ℕ) 
  (h1 : (50 * f) / f = 50) 
  (h2 : (30 * m) / m = 30) 
  (h3 : (50 * f + 30 * m) / (f + m) = 35) : 
  f / m = 1 / 3 := 
by
  sorry

end NUMINAMATH_GPT_ratio_female_male_l89_8903


namespace NUMINAMATH_GPT_problem_I_problem_II_l89_8904

variable (t a : ℝ)

-- Problem (I)
theorem problem_I (h1 : a = 1) (h2 : t^2 - 5 * a * t + 4 * a^2 < 0) (h3 : (t - 2) * (t - 6) < 0) : 2 < t ∧ t < 4 := 
by 
  sorry   -- Proof omitted as per instructions

-- Problem (II)
theorem problem_II (h1 : (t - 2) * (t - 6) < 0 → t^2 - 5 * a * t + 4 * a^2 < 0) : 3 / 2 ≤ a ∧ a ≤ 2 :=
by 
  sorry   -- Proof omitted as per instructions

end NUMINAMATH_GPT_problem_I_problem_II_l89_8904


namespace NUMINAMATH_GPT_P_iff_q_l89_8996

variables (a b c: ℝ)

def P : Prop := a * c < 0
def q : Prop := ∃ α β : ℝ, α * β < 0 ∧ a * α^2 + b * α + c = 0 ∧ a * β^2 + b * β + c = 0

theorem P_iff_q : P a c ↔ q a b c := 
sorry

end NUMINAMATH_GPT_P_iff_q_l89_8996


namespace NUMINAMATH_GPT_factors_2310_l89_8958

theorem factors_2310 : ∃ (S : Finset ℕ), (∀ p ∈ S, Nat.Prime p) ∧ S.card = 5 ∧ (2310 = S.prod id) :=
by
  sorry

end NUMINAMATH_GPT_factors_2310_l89_8958


namespace NUMINAMATH_GPT_integers_a_b_c_d_arbitrarily_large_l89_8944

theorem integers_a_b_c_d_arbitrarily_large (n : ℤ) : 
  ∃ (a b c d : ℤ), (a^2 + b^2 + c^2 + d^2 = a * b * c + a * b * d + a * c * d + b * c * d) ∧ 
    min (min a b) (min c d) ≥ n := 
by sorry

end NUMINAMATH_GPT_integers_a_b_c_d_arbitrarily_large_l89_8944


namespace NUMINAMATH_GPT_polygon_is_decagon_l89_8926

-- Definitions based on conditions
def exterior_angles_sum (x : ℕ) : ℝ := 360

def interior_angles_sum (x : ℕ) : ℝ := 4 * exterior_angles_sum x

def interior_sum_formula (n : ℕ) : ℝ := (n - 2) * 180

-- Mathematically equivalent proof problem
theorem polygon_is_decagon (n : ℕ) (h1 : exterior_angles_sum n = 360)
  (h2 : interior_angles_sum n = 4 * exterior_angles_sum n)
  (h3 : interior_sum_formula n = interior_angles_sum n) : n = 10 :=
sorry

end NUMINAMATH_GPT_polygon_is_decagon_l89_8926


namespace NUMINAMATH_GPT_fraction_to_decimal_l89_8929

theorem fraction_to_decimal : (3 : ℚ) / 80 = 0.0375 :=
by
  sorry

end NUMINAMATH_GPT_fraction_to_decimal_l89_8929


namespace NUMINAMATH_GPT_find_a_l89_8900

noncomputable def f (a x : ℝ) : ℝ := a * x^2 + 2 * a * x + 1

theorem find_a
  (a : ℝ)
  (h₁ : ∀ x ∈ Set.Icc (-3 : ℝ) (2 : ℝ), f a x ≤ 4)
  (h₂ : ∃ x ∈ Set.Icc (-3 : ℝ) (2 : ℝ), f a x = 4) :
  a = -3 ∨ a = 3 / 8 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l89_8900


namespace NUMINAMATH_GPT_Tony_fills_pool_in_90_minutes_l89_8995

def minutes (r : ℚ) : ℚ := 1 / r

theorem Tony_fills_pool_in_90_minutes (J S T : ℚ) 
  (hJ : J = 1 / 30)       -- Jim's rate in pools per minute
  (hS : S = 1 / 45)       -- Sue's rate in pools per minute
  (h_combined : J + S + T = 1 / 15) -- Combined rate of all three

  : minutes T = 90 :=     -- Tony can fill the pool alone in 90 minutes
by sorry

end NUMINAMATH_GPT_Tony_fills_pool_in_90_minutes_l89_8995


namespace NUMINAMATH_GPT_work_finish_in_3_days_l89_8976

-- Define the respective rates of work
def A_rate := 1/4
def B_rate := 1/14
def C_rate := 1/7

-- Define the duration they start working together
def initial_duration := 2
def after_C_joining := 1 -- time after C joins before A leaves

-- From the third day, consider A leaving the job
theorem work_finish_in_3_days :
  (initial_duration * (A_rate + B_rate)) + 
  (after_C_joining * (A_rate + B_rate + C_rate)) + 
  ((1 : ℝ) - after_C_joining) * (B_rate + C_rate) >= 1 :=
by
  sorry

end NUMINAMATH_GPT_work_finish_in_3_days_l89_8976


namespace NUMINAMATH_GPT_mushrooms_left_l89_8937

-- Define the initial amount of mushrooms.
def init_mushrooms : ℕ := 15

-- Define the amount of mushrooms eaten.
def eaten_mushrooms : ℕ := 8

-- Define the resulting amount of mushrooms.
def remaining_mushrooms (init : ℕ) (eaten : ℕ) : ℕ := init - eaten

-- The proof statement
theorem mushrooms_left : remaining_mushrooms init_mushrooms eaten_mushrooms = 7 :=
by
    sorry

end NUMINAMATH_GPT_mushrooms_left_l89_8937


namespace NUMINAMATH_GPT_integer_value_of_a_l89_8964

theorem integer_value_of_a (a x y z k : ℤ) :
  (x = k) ∧ (y = 4 * k) ∧ (z = 5 * k) ∧ (y = 9 * a^2 - 2 * a - 8) ∧ (z = 10 * a + 2) → a = 5 :=
by 
  sorry

end NUMINAMATH_GPT_integer_value_of_a_l89_8964


namespace NUMINAMATH_GPT_final_problem_l89_8931

def problem1 : Prop :=
  ∃ (x y : ℝ), 10 * x + 20 * y = 3000 ∧ 8 * x + 24 * y = 2800 ∧ x = 200 ∧ y = 50

def problem2 : Prop :=
  ∀ (m : ℕ), 10 ≤ m ∧ m ≤ 12 ∧ 
  200 * m + 50 * (40 - m) ≤ 3800 ∧ 
  (40 - m) ≤ 3 * m →
  (m = 10 ∧ (40 - m) = 30) ∨ 
  (m = 11 ∧ (40 - m) = 29) ∨ 
  (m = 12 ∧ (40 - m) = 28)

theorem final_problem : problem1 ∧ problem2 :=
by
  sorry

end NUMINAMATH_GPT_final_problem_l89_8931


namespace NUMINAMATH_GPT_min_a_minus_b_when_ab_eq_156_l89_8962

theorem min_a_minus_b_when_ab_eq_156 : ∃ a b : ℤ, (a * b = 156 ∧ a - b = -155) :=
by
  sorry

end NUMINAMATH_GPT_min_a_minus_b_when_ab_eq_156_l89_8962


namespace NUMINAMATH_GPT_find_parts_per_hour_find_min_A_machines_l89_8969

-- Conditions
variable (x y : ℕ) -- x is parts per hour by B, y is parts per hour by A

-- Definitions based on conditions
def machineA_speed_relation (x y : ℕ) : Prop :=
  y = x + 2

def time_relation (x y : ℕ) : Prop :=
  80 / y = 60 / x

def min_A_machines (x y : ℕ) (m : ℕ) : Prop :=
  8 * m + 6 * (10 - m) ≥ 70

-- Problem statements
theorem find_parts_per_hour (x y : ℕ) (h1 : machineA_speed_relation x y) (h2 : time_relation x y) :
  x = 6 ∧ y = 8 :=
sorry

theorem find_min_A_machines (m : ℕ) (h1 : machineA_speed_relation 6 8) (h2 : time_relation 6 8) (h3 : min_A_machines 6 8 m) :
  m ≥ 5 :=
sorry

end NUMINAMATH_GPT_find_parts_per_hour_find_min_A_machines_l89_8969


namespace NUMINAMATH_GPT_quadratic_has_non_real_roots_l89_8945

theorem quadratic_has_non_real_roots (c : ℝ) (h : c > 16) :
    ∃ (a b : ℂ), (x^2 - 8 * x + c = 0) = (a * a = -1) ∧ (b * b = -1) :=
sorry

end NUMINAMATH_GPT_quadratic_has_non_real_roots_l89_8945

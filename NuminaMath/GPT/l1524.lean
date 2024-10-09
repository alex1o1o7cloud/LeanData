import Mathlib

namespace fixed_point_for_all_k_l1524_152464

theorem fixed_point_for_all_k (k : ℝ) : (5, 225) ∈ { p : ℝ × ℝ | ∃ k : ℝ, p.snd = 9 * p.fst^2 + k * p.fst - 5 * k } :=
by
  sorry

end fixed_point_for_all_k_l1524_152464


namespace tree_initial_leaves_l1524_152426

theorem tree_initial_leaves (L : ℝ) (h1 : ∀ n : ℤ, 1 ≤ n ∧ n ≤ 4 → ∃ k : ℝ, L = k * (9/10)^n + k / 10^n)
                            (h2 : L * (9/10)^4 = 204) :
  L = 311 :=
by
  sorry

end tree_initial_leaves_l1524_152426


namespace distinct_schedules_l1524_152421

-- Define the problem setting and assumptions
def subjects := ["Chinese", "Mathematics", "Politics", "English", "Physical Education", "Art"]

-- Given conditions
def math_in_first_three_periods (schedule : List String) : Prop :=
  ∃ k, (k < 3) ∧ (schedule.get! k = "Mathematics")

def english_not_in_sixth_period (schedule : List String) : Prop :=
  schedule.get! 5 ≠ "English"

-- Define the proof problem
theorem distinct_schedules : 
  ∃! (schedules : List (List String)), 
  (∀ schedule ∈ schedules, 
    math_in_first_three_periods schedule ∧ 
    english_not_in_sixth_period schedule) ∧
  schedules.length = 288 :=
by
  sorry

end distinct_schedules_l1524_152421


namespace successive_product_4160_l1524_152429

theorem successive_product_4160 (n : ℕ) (h : n * (n + 1) = 4160) : n = 64 :=
sorry

end successive_product_4160_l1524_152429


namespace tulip_price_correct_l1524_152498

-- Initial conditions
def first_day_tulips : ℕ := 30
def first_day_roses : ℕ := 20
def second_day_tulips : ℕ := 60
def second_day_roses : ℕ := 40
def third_day_tulips : ℕ := 6
def third_day_roses : ℕ := 16
def rose_price : ℝ := 3
def total_revenue : ℝ := 420

-- Question: What is the price of one tulip?
def tulip_price (T : ℝ) : ℝ :=
    first_day_tulips * T + first_day_roses * rose_price +
    second_day_tulips * T + second_day_roses * rose_price +
    third_day_tulips * T + third_day_roses * rose_price

-- Proof problem statement
theorem tulip_price_correct (T : ℝ) : tulip_price T = total_revenue → T = 2 :=
by
  sorry

end tulip_price_correct_l1524_152498


namespace land_remaining_is_correct_l1524_152476

def lizzie_covered : ℕ := 250
def other_covered : ℕ := 265
def total_land : ℕ := 900
def land_remaining : ℕ := total_land - (lizzie_covered + other_covered)

theorem land_remaining_is_correct : land_remaining = 385 := 
by
  sorry

end land_remaining_is_correct_l1524_152476


namespace total_songs_time_l1524_152463

-- Definitions of durations for each radio show
def duration_show1 : ℕ := 180
def duration_show2 : ℕ := 240
def duration_show3 : ℕ := 120

-- Definitions of talking segments for each show
def talking_segments_show1 : ℕ := 3 * 10  -- 3 segments, 10 minutes each
def talking_segments_show2 : ℕ := 4 * 15  -- 4 segments, 15 minutes each
def talking_segments_show3 : ℕ := 2 * 8   -- 2 segments, 8 minutes each

-- Definitions of ad breaks for each show
def ad_breaks_show1 : ℕ := 5 * 5  -- 5 breaks, 5 minutes each
def ad_breaks_show2 : ℕ := 6 * 4  -- 6 breaks, 4 minutes each
def ad_breaks_show3 : ℕ := 3 * 6  -- 3 breaks, 6 minutes each

-- Function to calculate time spent on songs for a given show
def time_spent_on_songs (duration talking ad_breaks : ℕ) : ℕ :=
  duration - talking - ad_breaks

-- Total time spent on songs for all three shows
def total_time_spent_on_songs : ℕ :=
  time_spent_on_songs duration_show1 talking_segments_show1 ad_breaks_show1 +
  time_spent_on_songs duration_show2 talking_segments_show2 ad_breaks_show2 +
  time_spent_on_songs duration_show3 talking_segments_show3 ad_breaks_show3

-- The theorem we want to prove
theorem total_songs_time : total_time_spent_on_songs = 367 := 
  sorry

end total_songs_time_l1524_152463


namespace tap_B_fills_remaining_pool_l1524_152475

theorem tap_B_fills_remaining_pool :
  ∀ (flow_A flow_B : ℝ) (t_A t_B : ℕ),
  flow_A = 7.5 / 100 →  -- A fills 7.5% of the pool per hour
  flow_B = 5 / 100 →    -- B fills 5% of the pool per hour
  t_A = 2 →             -- A is open for 2 hours during the second phase
  t_A * flow_A = 15 / 100 →  -- A fills 15% of the pool in 2 hours
  4 * (flow_A + flow_B) = 50 / 100 →  -- A and B together fill 50% of the pool in 4 hours
  (100 / 100 - 50 / 100 - 15 / 100) / flow_B = t_B →  -- remaining pool filled only by B
  t_B = 7 := sorry    -- Prove that t_B is 7

end tap_B_fills_remaining_pool_l1524_152475


namespace adam_earning_per_lawn_l1524_152454

theorem adam_earning_per_lawn 
  (total_lawns : ℕ) 
  (forgotten_lawns : ℕ) 
  (total_earnings : ℕ) 
  (h1 : total_lawns = 12) 
  (h2 : forgotten_lawns = 8) 
  (h3 : total_earnings = 36) : 
  total_earnings / (total_lawns - forgotten_lawns) = 9 :=
by
  sorry

end adam_earning_per_lawn_l1524_152454


namespace operation_result_l1524_152473

theorem operation_result (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) (h_sum : a + b = 12) (h_prod : a * b = 32) 
: (1 / a : ℚ) + (1 / b) = 3 / 8 := by
  sorry

end operation_result_l1524_152473


namespace binom_n_plus_one_n_l1524_152467

theorem binom_n_plus_one_n (n : ℕ) (h : 0 < n) : Nat.choose (n + 1) n = n + 1 := 
sorry

end binom_n_plus_one_n_l1524_152467


namespace simplify_expression_l1524_152405

theorem simplify_expression (x : ℝ) : 3 * x + 6 * x + 9 * x + 12 * x + 15 * x + 18 = 45 * x + 18 :=
by
  sorry

end simplify_expression_l1524_152405


namespace max_value_of_m_l1524_152494

theorem max_value_of_m {a b c : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h_sum : a + b + c = 12) (h_prod : a * b + b * c + c * a = 20) :
  ∃ m, m = min (a * b) (min (b * c) (c * a)) ∧ m = 12 :=
by
  sorry

end max_value_of_m_l1524_152494


namespace car_speed_in_mph_l1524_152410

-- Defining the given conditions
def fuel_efficiency : ℚ := 56 -- kilometers per liter
def gallons_to_liters : ℚ := 3.8 -- liters per gallon
def kilometers_to_miles : ℚ := 1 / 1.6 -- miles per kilometer
def fuel_decrease_gallons : ℚ := 3.9 -- gallons
def time_hours : ℚ := 5.7 -- hours

-- Using definitions to compute the speed
theorem car_speed_in_mph :
  (fuel_decrease_gallons * gallons_to_liters * fuel_efficiency * kilometers_to_miles) / time_hours = 91 :=
sorry

end car_speed_in_mph_l1524_152410


namespace percentage_of_water_in_nectar_l1524_152491

-- Define the necessary conditions and variables
def weight_of_nectar : ℝ := 1.7 -- kg
def weight_of_honey : ℝ := 1 -- kg
def honey_water_percentage : ℝ := 0.15 -- 15%

noncomputable def water_in_honey : ℝ := weight_of_honey * honey_water_percentage -- Water content in 1 kg of honey

noncomputable def total_water_in_nectar : ℝ := water_in_honey + (weight_of_nectar - weight_of_honey) -- Total water content in nectar

-- The theorem to prove
theorem percentage_of_water_in_nectar :
    (total_water_in_nectar / weight_of_nectar) * 100 = 50 := 
by 
    -- Skipping the proof by using sorry as it is not required
    sorry

end percentage_of_water_in_nectar_l1524_152491


namespace f_3_equals_1000_l1524_152458

-- Define the function property f(lg x) = x
axiom f : ℝ → ℝ
axiom lg : ℝ → ℝ -- log function
axiom f_property : ∀ x : ℝ, f (lg x) = x

-- Prove that f(3) = 10^3
theorem f_3_equals_1000 : f 3 = 10^3 :=
by 
  -- Sorry to skip the proof
  sorry

end f_3_equals_1000_l1524_152458


namespace sum_of_cubes_l1524_152436

theorem sum_of_cubes (x y : ℝ) (h₁ : x + y = -1) (h₂ : x * y = -1) : x^3 + y^3 = -4 := by
  sorry

end sum_of_cubes_l1524_152436


namespace cone_radius_l1524_152400

noncomputable def radius_of_cone (V : ℝ) (h : ℝ) : ℝ := 
  3 / Real.sqrt (Real.pi)

theorem cone_radius :
  ∀ (V h : ℝ), V = 12 → h = 4 → radius_of_cone V h = 3 / Real.sqrt (Real.pi) :=
by
  intros V h hV hv
  sorry

end cone_radius_l1524_152400


namespace sin_double_angle_l1524_152474

theorem sin_double_angle (x : ℝ) (h : Real.cos (π / 4 - x) = 3 / 5) : Real.sin (2 * x) = -7 / 25 :=
by
  sorry

end sin_double_angle_l1524_152474


namespace nat_divides_power_difference_l1524_152446

theorem nat_divides_power_difference (n : ℕ) : n ∣ 2 ^ (2 * n.factorial) - 2 ^ n.factorial := by
  sorry

end nat_divides_power_difference_l1524_152446


namespace factorize_expression_l1524_152401

variable (a : ℝ)

theorem factorize_expression : a^3 - 2 * a^2 = a^2 * (a - 2) :=
by
  sorry

end factorize_expression_l1524_152401


namespace division_of_negatives_l1524_152449

theorem division_of_negatives : (-500 : ℤ) / (-50 : ℤ) = 10 := by
  sorry

end division_of_negatives_l1524_152449


namespace inequality1_solution_inequality2_solution_l1524_152414

variables (x a : ℝ)

theorem inequality1_solution : (∀ x : ℝ, (2 * x) / (x + 1) < 1 ↔ -1 < x ∧ x < 1) :=
by
  sorry

theorem inequality2_solution (a : ℝ) : 
  (∀ x : ℝ, x^2 + (2 - a) * x - 2 * a ≥ 0 ↔ 
    (a = -2 → true) ∧ 
    (a > -2 → (x ≤ -2 ∨ x ≥ a)) ∧ 
    (a < -2 → (x ≤ a ∨ x ≥ -2))) :=
by
  sorry

end inequality1_solution_inequality2_solution_l1524_152414


namespace point_K_outside_hexagon_and_length_KC_l1524_152447

theorem point_K_outside_hexagon_and_length_KC :
    ∀ (A B C K : ℝ × ℝ),
    A = (0, 0) →
    B = (3, 0) →
    C = (3 / 2, (3 * Real.sqrt 3) / 2) →
    K = (15 / 2, - (3 * Real.sqrt 3) / 2) →
    (¬ (0 ≤ K.1 ∧ K.1 ≤ 3 ∧ 0 ≤ K.2 ∧ K.2 ≤ 3 * Real.sqrt 3)) ∧
    Real.sqrt ((K.1 - C.1) ^ 2 + (K.2 - C.2) ^ 2) = 3 * Real.sqrt 7 :=
by
  intros A B C K hA hB hC hK
  sorry

end point_K_outside_hexagon_and_length_KC_l1524_152447


namespace geometric_series_first_term_l1524_152444

theorem geometric_series_first_term
  (a r : ℚ)
  (h1 : a / (1 - r) = 30)
  (h2 : a^2 / (1 - r^2) = 150) :
  a = 60 / 7 :=
by
  sorry

end geometric_series_first_term_l1524_152444


namespace rectangle_area_ratio_l1524_152471

-- Define points in complex plane or as tuples (for 2D geometry)
structure Point where
  x : ℝ
  y : ℝ

-- Rectangle vertices
def A : Point := {x := 0, y := 0}
def B : Point := {x := 1, y := 0}
def C : Point := {x := 1, y := 2}
def D : Point := {x := 0, y := 2}

-- Centroid of triangle BCD
def E : Point := {x := 1.0, y := 1.333}

-- Point F such that DF = 1/4 * DA
def F : Point := {x := 1.5, y := 0}

-- Calculate areas of triangles and quadrilateral
noncomputable def area_triangle (P Q R : Point) : ℝ :=
  0.5 * abs ((Q.x - P.x) * (R.y - P.y) - (Q.y - P.y) * (R.x - P.x))

noncomputable def area_rectangle : ℝ :=
  2.0  -- Area of rectangle ABCD (1 * 2)

noncomputable def problem_statement : Prop :=
  let area_DFE := area_triangle D F E
  let area_ABEF := area_rectangle - area_triangle A B F - area_triangle D A F
  area_DFE / area_ABEF = 1 / 10.5

theorem rectangle_area_ratio :
  problem_statement :=
by
  sorry

end rectangle_area_ratio_l1524_152471


namespace wicket_keeper_older_than_captain_l1524_152469

-- Define the team and various ages
def captain_age : ℕ := 28
def average_age_team : ℕ := 25
def number_of_players : ℕ := 11
def number_of_remaining_players : ℕ := number_of_players - 2
def average_age_remaining_players : ℕ := average_age_team - 1

theorem wicket_keeper_older_than_captain :
  ∃ (W : ℕ), W = captain_age + 3 ∧
  275 = number_of_players * average_age_team ∧
  216 = number_of_remaining_players * average_age_remaining_players ∧
  59 = 275 - 216 ∧
  W = 59 - captain_age :=
by
  sorry

end wicket_keeper_older_than_captain_l1524_152469


namespace black_balls_count_l1524_152427

theorem black_balls_count :
  ∀ (r k : ℕ), r = 10 -> (2 : ℚ) / 7 = r / (r + k : ℚ) -> k = 25 := by
  intros r k hr hprob
  sorry

end black_balls_count_l1524_152427


namespace water_left_after_four_hours_l1524_152490

def initial_water : ℕ := 40
def water_loss_per_hour : ℕ := 2
def water_added_hour3 : ℕ := 1
def water_added_hour4 : ℕ := 3

theorem water_left_after_four_hours :
    initial_water - 2 * 2 - 2 + water_added_hour3 - 2 + water_added_hour4 - 2 = 36 := 
by
    sorry

end water_left_after_four_hours_l1524_152490


namespace second_integer_is_66_l1524_152428

-- Define the conditions
def are_two_units_apart (a b c : ℤ) : Prop :=
  b = a + 2 ∧ c = a + 4

def sum_of_first_and_third_is_132 (a b c : ℤ) : Prop :=
  a + c = 132

-- State the theorem
theorem second_integer_is_66 (a b c : ℤ) 
  (H1 : are_two_units_apart a b c) 
  (H2 : sum_of_first_and_third_is_132 a b c) : b = 66 :=
by
  sorry -- Proof omitted

end second_integer_is_66_l1524_152428


namespace count_integers_congruent_to_7_mod_13_l1524_152438

theorem count_integers_congruent_to_7_mod_13 : 
  (∃ (n : ℕ), ∀ x, (1 ≤ x ∧ x < 500 ∧ x % 13 = 7) → x = 7 + 13 * n ∧ n < 38) :=
sorry

end count_integers_congruent_to_7_mod_13_l1524_152438


namespace tan_half_angle_third_quadrant_l1524_152465

theorem tan_half_angle_third_quadrant (α : ℝ) (hα : π < α ∧ α < 3 * π / 2) (h : Real.sin α = -24/25) :
  Real.tan (α / 2) = -4/3 := 
by 
  sorry

end tan_half_angle_third_quadrant_l1524_152465


namespace area_of_triangle_l1524_152495

-- Define the coordinates of the vertices
def A : ℝ × ℝ := (0, 3)
def B : ℝ × ℝ := (7, -1)
def C : ℝ × ℝ := (2, 6)

-- Define the function to calculate the area of the triangle formed by three vertices
def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

-- The theorem statement that the area of the triangle with given vertices is 14.5
theorem area_of_triangle : triangle_area A B C = 14.5 :=
by 
  -- Skipping the proof part
  sorry

end area_of_triangle_l1524_152495


namespace parallel_line_with_y_intercept_l1524_152433

theorem parallel_line_with_y_intercept (x y : ℝ) (m : ℝ) : 
  ((x + y + 4 = 0) → (x + y + m = 0)) ∧ (m = 1)
 := by sorry

end parallel_line_with_y_intercept_l1524_152433


namespace stable_state_exists_l1524_152417

-- Definition of the problem
theorem stable_state_exists 
(N : ℕ) (N_ge_3 : N ≥ 3) (letters : Fin N → Fin 3) 
(perform_operation : ∀ (letters : Fin N → Fin 3), Fin N → Fin 3)
(stable : ∀ (letters : Fin N → Fin 3), Prop)
(initial_state : Fin N → Fin 3):
  ∃ (state : Fin N → Fin 3), (∀ i, perform_operation state i = state i) ∧ stable state :=
sorry

end stable_state_exists_l1524_152417


namespace Mina_has_2_25_cent_coins_l1524_152402

def MinaCoinProblem : Prop :=
  ∃ (x y z : ℕ), -- number of 5-cent, 10-cent, and 25-cent coins
  x + y + z = 15 ∧
  (74 - 4 * x - 3 * y = 30) ∧ -- corresponds to 30 different values can be obtained
  z = 2

theorem Mina_has_2_25_cent_coins : MinaCoinProblem :=
by 
  sorry

end Mina_has_2_25_cent_coins_l1524_152402


namespace tank_filling_time_with_leaks_l1524_152404

theorem tank_filling_time_with_leaks (pump_time : ℝ) (leak1_time : ℝ) (leak2_time : ℝ) (leak3_time : ℝ) (fill_time : ℝ)
  (h1 : pump_time = 2)
  (h2 : fill_time = 3)
  (h3 : leak1_time = 6)
  (h4 : leak2_time = 8)
  (h5 : leak3_time = 12) :
  fill_time = 8 := 
sorry

end tank_filling_time_with_leaks_l1524_152404


namespace shaded_area_of_circles_l1524_152488

theorem shaded_area_of_circles :
  let R := 10
  let r1 := R / 2
  let r2 := R / 2
  (π * R^2 - (π * r1^2 + π * r1^2 + π * r2^2)) = 25 * π :=
by
  sorry

end shaded_area_of_circles_l1524_152488


namespace proof_problem_l1524_152440

variable {a b c d : ℝ}
variable {x1 y1 x2 y2 x3 y3 x4 y4 : ℝ}

-- Assume the conditions
variable (habcd_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d)
variable (unity_circle : x1^2 + y1^2 = 1 ∧ x2^2 + y2^2 = 1 ∧ x3^2 + y3^2 = 1 ∧ x4^2 + y4^2 = 1)
variable (unit_sum : a * b + c * d = 1)

-- Statement to prove
theorem proof_problem :
  (a * y1 + b * y2 + c * y3 + d * y4)^2 + (a * x4 + b * x3 + c * x2 + d * x1)^2
    ≤ 2 * ((a^2 + b^2) / (a * b) + (c^2 + d^2) / (c * d)) := 
  sorry

end proof_problem_l1524_152440


namespace elvins_fixed_monthly_charge_l1524_152439

-- Definition of the conditions
def january_bill (F C_J : ℝ) : Prop := F + C_J = 48
def february_bill (F C_J : ℝ) : Prop := F + 2 * C_J = 90

theorem elvins_fixed_monthly_charge (F C_J : ℝ) (h_jan : january_bill F C_J) (h_feb : february_bill F C_J) : F = 6 :=
by
  sorry

end elvins_fixed_monthly_charge_l1524_152439


namespace intersection_complement_l1524_152483

open Set

variable (U P Q : Set ℕ)
variable (H_U : U = {1, 2, 3, 4, 5, 6})
variable (H_P : P = {1, 2, 3, 4})
variable (H_Q : Q = {3, 4, 5})

theorem intersection_complement (hU : U = {1, 2, 3, 4, 5, 6}) (hP : P = {1, 2, 3, 4}) (hQ : Q = {3, 4, 5}) :
  P ∩ (U \ Q) = {1, 2} := by
  sorry

end intersection_complement_l1524_152483


namespace quadratic_equation_divisible_by_x_minus_one_l1524_152493

theorem quadratic_equation_divisible_by_x_minus_one (a b c : ℝ) (h1 : (x - 1) ∣ (a * x * x + b * x + c)) (h2 : c = 2) :
  (a = 1 ∧ b = -3 ∧ c = 2) → a * x * x + b * x + c = x^2 - 3 * x + 2 :=
by
  sorry

end quadratic_equation_divisible_by_x_minus_one_l1524_152493


namespace general_term_formula_l1524_152466

theorem general_term_formula (n : ℕ) : 
  ∀ (a : ℕ → ℕ), (a 1 = 1) → (∀ n > 1, a n - a (n-1) = 2^(n-1)) → (a n = 2^n - 1) :=
  by 
  intros a h1 hdif
  sorry

end general_term_formula_l1524_152466


namespace find_a_plus_b_l1524_152499

-- Given points A and B, where A(1, a) and B(b, -2) are symmetric with respect to the origin.
variables (a b : ℤ)

-- Definition for symmetry conditions
def symmetric_wrt_origin (x1 y1 x2 y2 : ℤ) :=
  x2 = -x1 ∧ y2 = -y1

-- The main theorem
theorem find_a_plus_b :
  symmetric_wrt_origin 1 a b (-2) → a + b = 1 :=
by
  sorry

end find_a_plus_b_l1524_152499


namespace perimeter_triangle_APR_l1524_152482

-- Define given lengths
def AB := 24
def AC := AB
def AP := 8
def AR := AP

-- Define lengths calculated from conditions 
def PB := AB - AP
def RC := AC - AR

-- Define properties from the tangent intersection at Q
def PQ := PB
def QR := RC
def PR := PQ + QR

-- Calculate the perimeter
def perimeter_APR := AP + PR + AR

-- Proof of the problem statement
theorem perimeter_triangle_APR : perimeter_APR = 48 :=
by
  -- Calculations already given in the statement
  sorry

end perimeter_triangle_APR_l1524_152482


namespace line_intersects_parabola_exactly_one_point_l1524_152481

theorem line_intersects_parabola_exactly_one_point (k : ℝ) :
  (∃ y : ℝ, -3 * y^2 - 4 * y + 10 = k) ∧
  (∀ y z : ℝ, -3 * y^2 - 4 * y + 10 = k ∧ -3 * z^2 - 4 * z + 10 = k → y = z) 
  → k = 34 / 3 :=
by
  sorry

end line_intersects_parabola_exactly_one_point_l1524_152481


namespace scientific_notation_l1524_152459

variables (n : ℕ) (h : n = 505000)

theorem scientific_notation : n = 505000 → "5.05 * 10^5" = "scientific notation of 505000" :=
by
  intro h
  sorry

end scientific_notation_l1524_152459


namespace minimize_f_sin_65_sin_40_l1524_152419

noncomputable def f (x : ℝ) (m : ℝ) (n : ℝ) : ℝ := m^2 * x^2 + (n + 1) * x + 1

theorem minimize_f_sin_65_sin_40 (m n : ℝ) (h₁ : m = Real.sin (65 * Real.pi / 180))
  (h₂ : n = Real.sin (40 * Real.pi / 180)) : 
  ∃ x, x = -1 ∧ (∀ y, f y m n ≥ f (-1) m n) :=
by
  -- Proof to be completed
  sorry

end minimize_f_sin_65_sin_40_l1524_152419


namespace subtraction_verification_l1524_152479

theorem subtraction_verification : 888888888888 - 111111111111 = 777777777777 :=
by
  sorry

end subtraction_verification_l1524_152479


namespace reach_any_composite_from_4_l1524_152416

/-- 
Prove that starting from the number \( 4 \), it is possible to reach any given composite number 
through repeatedly adding one of its divisors, different from itself and one. 
-/
theorem reach_any_composite_from_4:
  ∀ n : ℕ, Prime (n) → n ≥ 4 → (∃ k d : ℕ, d ∣ k ∧ k = k + d ∧ k = n) := 
by 
  sorry


end reach_any_composite_from_4_l1524_152416


namespace sqrt_20_19_18_17_plus_1_eq_341_l1524_152403

theorem sqrt_20_19_18_17_plus_1_eq_341 :
  Real.sqrt ((20: ℝ) * 19 * 18 * 17 + 1) = 341 := by
sorry

end sqrt_20_19_18_17_plus_1_eq_341_l1524_152403


namespace fraction_numerator_less_denominator_l1524_152457

theorem fraction_numerator_less_denominator (x : ℝ) (h : -3 ≤ x ∧ x ≤ 3) :
  (8 * x - 3 < 9 + 5 * x) ↔ (-3 ≤ x ∧ x < 3) :=
by sorry

end fraction_numerator_less_denominator_l1524_152457


namespace rice_mixture_ratio_l1524_152452

theorem rice_mixture_ratio
  (cost_variety1 : ℝ := 5) 
  (cost_variety2 : ℝ := 8.75) 
  (desired_cost_mixture : ℝ := 7.50) 
  (x y : ℝ) :
  5 * x + 8.75 * y = 7.50 * (x + y) → 
  y / x = 2 :=
by
  intro h
  sorry

end rice_mixture_ratio_l1524_152452


namespace cos_B_value_l1524_152451

theorem cos_B_value (A B C a b c : ℝ) (h₁ : b * Real.cos C + c * Real.cos B = Real.sqrt 3 * a * Real.cos B) :
  Real.cos B = Real.sqrt 3 / 3 := by
  sorry

end cos_B_value_l1524_152451


namespace count_valid_four_digit_numbers_l1524_152420

theorem count_valid_four_digit_numbers : 
  let valid_first_digits := (4*5 + 4*4)
  let valid_last_digits := (5*5 + 4*4)
  valid_first_digits * valid_last_digits = 1476 :=
by
  sorry

end count_valid_four_digit_numbers_l1524_152420


namespace permutations_five_three_eq_sixty_l1524_152424

theorem permutations_five_three_eq_sixty : (Nat.factorial 5) / (Nat.factorial (5 - 3)) = 60 := 
by
  sorry

end permutations_five_three_eq_sixty_l1524_152424


namespace cos_sin_identity_l1524_152437

theorem cos_sin_identity : 
  (Real.cos (14 * Real.pi / 180) * Real.cos (59 * Real.pi / 180) + 
   Real.sin (14 * Real.pi / 180) * Real.sin (121 * Real.pi / 180)) = (Real.sqrt 2 / 2) :=
by
  sorry

end cos_sin_identity_l1524_152437


namespace certain_number_value_l1524_152441

theorem certain_number_value
  (t b c x : ℝ)
  (h1 : (t + b + c + x + 15) / 5 = 12)
  (h2 : (t + b + c + 29) / 4 = 15) :
  x = 14 :=
by 
  sorry

end certain_number_value_l1524_152441


namespace rectangle_width_l1524_152422

theorem rectangle_width (x y : ℝ) (h1 : x * y = 12) (h2 : x^2 + y^2 = 25) : y = 3 := 
by 
  sorry

end rectangle_width_l1524_152422


namespace parallelogram_base_length_l1524_152462

theorem parallelogram_base_length (A H : ℝ) (base : ℝ) 
    (hA : A = 72) (hH : H = 6) (h_area : A = base * H) : base = 12 := 
by 
  sorry

end parallelogram_base_length_l1524_152462


namespace mike_total_spending_is_497_50_l1524_152489

def rose_bush_price : ℝ := 75
def rose_bush_count : ℕ := 6
def rose_bush_discount : ℝ := 0.10
def friend_rose_bushes : ℕ := 2
def tax_rose_bushes : ℝ := 0.05

def aloe_price : ℝ := 100
def aloe_count : ℕ := 2
def tax_aloe : ℝ := 0.07

def calculate_total_cost_for_mike : ℝ :=
  let total_rose_bush_cost := rose_bush_price * rose_bush_count
  let discount := total_rose_bush_cost * rose_bush_discount
  let cost_after_discount := total_rose_bush_cost - discount
  let sales_tax_rose_bushes := tax_rose_bushes * cost_after_discount
  let cost_rose_bushes_after_tax := cost_after_discount + sales_tax_rose_bushes

  let total_aloe_cost := aloe_price * aloe_count
  let sales_tax_aloe := tax_aloe * total_aloe_cost

  let total_cost_friend_rose_bushes := friend_rose_bushes * (rose_bush_price - (rose_bush_price * rose_bush_discount))
  let sales_tax_friend_rose_bushes := tax_rose_bushes * total_cost_friend_rose_bushes
  let total_cost_friend := total_cost_friend_rose_bushes + sales_tax_friend_rose_bushes

  let total_mike_rose_bushes := cost_rose_bushes_after_tax - total_cost_friend

  let total_cost_mike_aloe := total_aloe_cost + sales_tax_aloe

  total_mike_rose_bushes + total_cost_mike_aloe

theorem mike_total_spending_is_497_50 : calculate_total_cost_for_mike = 497.50 := by
  sorry

end mike_total_spending_is_497_50_l1524_152489


namespace breakfast_problem_probability_l1524_152456

def are_relatively_prime (m n : ℕ) : Prop :=
  Nat.gcd m n = 1

theorem breakfast_problem_probability : 
  ∃ m n : ℕ, are_relatively_prime m n ∧ 
  (1 / 1 * 9 / 11 * 6 / 10 * 1 / 3) * 1 = 9 / 55 ∧ m + n = 64 :=
by
  sorry

end breakfast_problem_probability_l1524_152456


namespace max_value_2x_minus_y_l1524_152415

theorem max_value_2x_minus_y (x y : ℝ) (h₁ : x + y - 1 < 0) (h₂ : x - y ≤ 0) (h₃ : 0 ≤ x) :
  ∃ z, (z = 2 * x - y) ∧ (z ≤ (1 / 2)) :=
sorry

end max_value_2x_minus_y_l1524_152415


namespace initial_quantity_of_gummy_worms_l1524_152492

theorem initial_quantity_of_gummy_worms (x : ℕ) (h : x / 2^4 = 4) : x = 64 :=
sorry

end initial_quantity_of_gummy_worms_l1524_152492


namespace marge_final_plant_count_l1524_152468

/-- Define the initial conditions of the garden -/
def initial_seeds : ℕ := 23
def marigold_seeds : ℕ := 10
def sunflower_seeds : ℕ := 8
def lavender_seeds : ℕ := 5
def seeds_without_growth : ℕ := 5

/-- Growth rates for each type of plant -/
def marigold_growth_rate : ℕ := 4
def sunflower_growth_rate : ℕ := 4
def lavender_growth_rate : ℕ := 3

/-- Impact of animals -/
def marigold_eaten_by_squirrels : ℕ := 2
def sunflower_eaten_by_rabbits : ℕ := 1

/-- Impact of pest control -/
def marigold_pest_control_reduction : ℕ := 0
def sunflower_pest_control_reduction : ℕ := 0
def lavender_pest_control_protected : ℕ := 2

/-- Impact of weeds -/
def weeds_strangled_plants : ℕ := 2

/-- Weeds left as plants -/
def weeds_kept_as_plants : ℕ := 1

/-- Marge's final number of plants -/
def survived_plants :=
  (marigold_growth_rate - marigold_eaten_by_squirrels - marigold_pest_control_reduction) +
  (sunflower_growth_rate - sunflower_eaten_by_rabbits - sunflower_pest_control_reduction) +
  (lavender_growth_rate - (lavender_growth_rate - lavender_pest_control_protected)) - weeds_strangled_plants

theorem marge_final_plant_count :
  survived_plants + weeds_kept_as_plants = 6 :=
by
  sorry

end marge_final_plant_count_l1524_152468


namespace lcm_18_20_25_l1524_152411

-- Lean 4 statement to prove the smallest positive integer divisible by 18, 20, and 25 is 900
theorem lcm_18_20_25 : Nat.lcm (Nat.lcm 18 20) 25 = 900 :=
by
  sorry

end lcm_18_20_25_l1524_152411


namespace wheel_distance_3_revolutions_l1524_152485

theorem wheel_distance_3_revolutions (r : ℝ) (n : ℝ) (circumference : ℝ) (total_distance : ℝ) :
  r = 2 →
  n = 3 →
  circumference = 2 * Real.pi * r →
  total_distance = n * circumference →
  total_distance = 12 * Real.pi := by
  intros
  sorry

end wheel_distance_3_revolutions_l1524_152485


namespace angle_sum_155_l1524_152418

theorem angle_sum_155
  (AB AC DE DF : ℝ)
  (h1 : AB = AC)
  (h2 : DE = DF)
  (angle_BAC angle_EDF : ℝ)
  (h3 : angle_BAC = 20)
  (h4 : angle_EDF = 30) :
  ∃ (angle_DAC angle_ADE : ℝ), angle_DAC + angle_ADE = 155 :=
by
  sorry

end angle_sum_155_l1524_152418


namespace sleepySquirrelNutsPerDay_l1524_152486

def twoBusySquirrelsNutsPerDay : ℕ := 2 * 30
def totalDays : ℕ := 40
def totalNuts : ℕ := 3200

theorem sleepySquirrelNutsPerDay 
  (s  : ℕ) 
  (h₁ : 2 * 30 * totalDays + s * totalDays = totalNuts) 
  : s = 20 := 
  sorry

end sleepySquirrelNutsPerDay_l1524_152486


namespace length_of_arc_l1524_152487

theorem length_of_arc (C : ℝ) (θ : ℝ) (DE : ℝ) (c_circ : C = 100) (angle : θ = 120) :
  DE = 100 / 3 :=
by
  -- Place the actual proof here.
  sorry

end length_of_arc_l1524_152487


namespace find_a_l1524_152406

variables {a b c : ℤ}

theorem find_a (h1 : a + b = c) (h2 : b + c = 7) (h3 : c = 5) : a = 3 :=
by
  sorry

end find_a_l1524_152406


namespace probability_of_losing_weight_l1524_152455

theorem probability_of_losing_weight (total_volunteers lost_weight : ℕ) (h_total : total_volunteers = 1000) (h_lost : lost_weight = 241) : 
    (lost_weight : ℚ) / total_volunteers = 0.24 := by
  sorry

end probability_of_losing_weight_l1524_152455


namespace integer_solutions_l1524_152413

-- Define the problem statement in Lean
theorem integer_solutions :
  {p : ℤ × ℤ | ∃ x y : ℤ, p = (x, y) ∧ x^2 + x = y^4 + y^3 + y^2 + y} =
  {(-1, -1), (0, -1), (-1, 0), (0, 0), (5, 2), (-6, 2)} :=
by
  sorry

end integer_solutions_l1524_152413


namespace cannot_determine_E1_l1524_152412

variable (a b c d : ℝ)

theorem cannot_determine_E1 (h1 : a + b - c - d = 5) (h2 : (b - d)^2 = 16) : 
  ¬ ∃ e : ℝ, e = a - b - c + d :=
by
  sorry

end cannot_determine_E1_l1524_152412


namespace marys_mother_bought_3_pounds_of_beef_l1524_152478

-- Define the variables and constants
def total_paid : ℝ := 16
def cost_of_chicken : ℝ := 2 * 1  -- 2 pounds of chicken
def cost_per_pound_beef : ℝ := 4
def cost_of_oil : ℝ := 1
def shares : ℝ := 3  -- Mary and her two friends

theorem marys_mother_bought_3_pounds_of_beef:
  total_paid - (cost_of_chicken / shares) - cost_of_oil = 3 * cost_per_pound_beef :=
by
  -- the proof goes here
  sorry

end marys_mother_bought_3_pounds_of_beef_l1524_152478


namespace line_x_intercept_l1524_152480

theorem line_x_intercept (P Q : ℝ × ℝ) (hP : P = (2, 3)) (hQ : Q = (6, 7)) :
  ∃ x, (x, 0) = (-1, 0) ∧ ∃ (m : ℝ), m = (Q.2 - P.2) / (Q.1 - P.1) ∧ ∀ (x y : ℝ), y = m * (x - P.1) + P.2 := 
  sorry

end line_x_intercept_l1524_152480


namespace increasing_interval_l1524_152470

-- Define the function f(x) = x^2 + 2*(a - 1)*x
def f (x : ℝ) (a : ℝ) : ℝ := x^2 + 2*(a - 1)*x

-- Define the condition for f(x) being increasing on [4, +∞)
def is_increasing_on_interval (a : ℝ) : Prop := 
  ∀ x y : ℝ, 4 ≤ x → x ≤ y → 
    f x a ≤ f y a

-- Define the main theorem that we need to prove
theorem increasing_interval (a : ℝ) (h : is_increasing_on_interval a) : -3 ≤ a :=
by 
  sorry -- proof is required, but omitted as per the instruction.

end increasing_interval_l1524_152470


namespace weight_of_dried_grapes_l1524_152434

def fresh_grapes_initial_weight : ℝ := 25
def fresh_grapes_water_percentage : ℝ := 0.90
def dried_grapes_water_percentage : ℝ := 0.20

theorem weight_of_dried_grapes :
  (fresh_grapes_initial_weight * (1 - fresh_grapes_water_percentage)) /
  (1 - dried_grapes_water_percentage) = 3.125 := by
  -- Proof omitted
  sorry

end weight_of_dried_grapes_l1524_152434


namespace GCD_40_48_l1524_152496

theorem GCD_40_48 : Int.gcd 40 48 = 8 :=
by sorry

end GCD_40_48_l1524_152496


namespace average_score_first_2_matches_l1524_152450

theorem average_score_first_2_matches (A : ℝ) 
  (h1 : 3 * 40 = 120) 
  (h2 : 5 * 36 = 180) 
  (h3 : 2 * A + 120 = 180) : 
  A = 30 := 
by 
  have hA : 2 * A = 60 := by linarith [h3]
  have hA2 : A = 30 := by linarith [hA]
  exact hA2

end average_score_first_2_matches_l1524_152450


namespace factorization_of_expression_l1524_152408

theorem factorization_of_expression
  (a b c : ℝ)
  (expansion : (b+c)*(c+a)*(a+b) + abc = (a+b+c)*(ab+ac+bc)) : 
  ∃ (m l : ℝ), (m = 0 ∧ l = a + b + c ∧ 
  (b+c)*(c+a)*(a+b) + abc = m*(a^2 + b^2 + c^2) + l*(ab + ac + bc)) :=
by
  sorry

end factorization_of_expression_l1524_152408


namespace ramu_profit_percent_l1524_152431

theorem ramu_profit_percent
  (cost_of_car : ℕ)
  (cost_of_repairs : ℕ)
  (selling_price : ℕ)
  (total_cost : ℕ := cost_of_car + cost_of_repairs)
  (profit : ℕ := selling_price - total_cost)
  (profit_percent : ℚ := ((profit : ℚ) / total_cost) * 100)
  (h1 : cost_of_car = 42000)
  (h2 : cost_of_repairs = 15000)
  (h3 : selling_price = 64900) :
  profit_percent = 13.86 :=
by
  sorry

end ramu_profit_percent_l1524_152431


namespace tan_of_alpha_l1524_152409

noncomputable def point_P : ℝ × ℝ := (1, -2)

theorem tan_of_alpha (α : ℝ) (h : ∃ (P : ℝ × ℝ), P = point_P ∧ P.2 / P.1 = -2) :
  Real.tan α = -2 :=
sorry

end tan_of_alpha_l1524_152409


namespace composite_sum_l1524_152443

theorem composite_sum (m n : ℕ) (h : 88 * m = 81 * n) : ∃ p q : ℕ, p > 1 ∧ q > 1 ∧ (m + n) = p * q :=
by sorry

end composite_sum_l1524_152443


namespace cubic_representation_l1524_152425

variable (a b : ℝ) (x : ℝ)
variable (v u w : ℝ)

axiom h1 : 6.266 * x^3 - 3 * a * x^2 + (3 * a^2 - b) * x - (a^3 - a * b) = 0
axiom h2 : b ≥ 0

theorem cubic_representation : v = a ∧ u = a ∧ w^2 = b → 
  6.266 * x^3 - 3 * v * x^2 + (3 * u^2 - w^2) * x - (u^3 - u * w^2) = 0 :=
by
  sorry

end cubic_representation_l1524_152425


namespace tangent_line_through_origin_l1524_152423

noncomputable def curve (x : ℝ) : ℝ := Real.exp (x - 1) + x

theorem tangent_line_through_origin :
  ∃ k : ℝ, k = 2 ∧ ∀ x y : ℝ, (y = k * x) ↔ (∃ m : ℝ, curve m = m + Real.exp (m - 1) ∧ (curve m) = (m + Real.exp (m - 1)) ∧ k = (Real.exp (m - 1) + 1) ∧ y = k * x ∧ y = 2*x) :=
by 
  sorry

end tangent_line_through_origin_l1524_152423


namespace gcd_polynomials_l1524_152461

-- Define a as a multiple of 1836
def is_multiple_of (a b : ℤ) : Prop := ∃ k : ℤ, a = k * b

-- Problem statement: gcd of the polynomial expressions given the condition
theorem gcd_polynomials (a : ℤ) (h : is_multiple_of a 1836) : Int.gcd (2 * a^2 + 11 * a + 40) (a + 4) = 4 :=
by
  sorry

end gcd_polynomials_l1524_152461


namespace part1_part2_l1524_152477

-- Define the complex number z in terms of m
def z (m : ℝ) : ℂ := (2 * m^2 - 3 * m - 2) + (m^2 - 3 * m + 2) * Complex.I

-- State the condition where z is a purely imaginary number
def purelyImaginary (m : ℝ) : Prop := 2 * m^2 - 3 * m - 2 = 0 ∧ m^2 - 3 * m + 2 ≠ 0

-- State the condition where z is in the second quadrant.
def inSecondQuadrant (m : ℝ) : Prop := 2 * m^2 - 3 * m - 2 < 0 ∧ m^2 - 3 * m + 2 > 0

-- Part 1: Prove that m = -1/2 given that z is purely imaginary.
theorem part1 : purelyImaginary m → m = -1/2 :=
sorry

-- Part 2: Prove the range of m for z in the second quadrant.
theorem part2 : inSecondQuadrant m → -1/2 < m ∧ m < 1 :=
sorry

end part1_part2_l1524_152477


namespace range_of_m_l1524_152432

theorem range_of_m (m : ℝ) (h1 : 0 < m) (h2 : ∀ x : ℝ, 0 < x ∧ x ≤ 1 → |m * x^3 - Real.log x| ≥ 1) : m ≥ (1 / 3) * Real.exp 2 :=
sorry

end range_of_m_l1524_152432


namespace megan_carrots_second_day_l1524_152460

theorem megan_carrots_second_day : 
  ∀ (initial : ℕ) (thrown : ℕ) (total : ℕ) (second_day : ℕ),
  initial = 19 →
  thrown = 4 →
  total = 61 →
  second_day = (total - (initial - thrown)) →
  second_day = 46 :=
by
  intros initial thrown total second_day h_initial h_thrown h_total h_second_day
  rw [h_initial, h_thrown, h_total] at h_second_day
  sorry

end megan_carrots_second_day_l1524_152460


namespace average_score_l1524_152445

theorem average_score (a_males : ℕ) (a_females : ℕ) (n_males : ℕ) (n_females : ℕ)
  (h_males : a_males = 85) (h_females : a_females = 92) (h_n_males : n_males = 8) (h_n_females : n_females = 20) :
  (a_males * n_males + a_females * n_females) / (n_males + n_females) = 90 :=
by
  sorry

end average_score_l1524_152445


namespace shaded_region_volume_l1524_152484

theorem shaded_region_volume :
  let r1 := 4   -- radius of the first cylinder
  let h1 := 2   -- height of the first cylinder
  let r2 := 1   -- radius of the second cylinder
  let h2 := 5   -- height of the second cylinder
  let V1 := π * r1^2 * h1 -- volume of the first cylinder
  let V2 := π * r2^2 * h2 -- volume of the second cylinder
  V1 + V2 = 37 * π :=
by
  sorry

end shaded_region_volume_l1524_152484


namespace rectangular_prism_faces_edges_vertices_sum_l1524_152430

theorem rectangular_prism_faces_edges_vertices_sum :
  let faces := 6
  let edges := 12
  let vertices := 8
  faces + edges + vertices = 26 := by
  let faces : ℕ := 6
  let edges : ℕ := 12
  let vertices : ℕ := 8
  sorry

end rectangular_prism_faces_edges_vertices_sum_l1524_152430


namespace valerie_initial_money_l1524_152472

theorem valerie_initial_money (n m C_s C_l L I : ℕ) 
  (h1 : n = 3) (h2 : m = 1) (h3 : C_s = 8) (h4 : C_l = 12) (h5 : L = 24) :
  I = (n * C_s) + (m * C_l) + L :=
  sorry

end valerie_initial_money_l1524_152472


namespace solve_x_l1524_152442

theorem solve_x (x : ℝ) : (x - 1)^2 = 4 → (x = 3 ∨ x = -1) := 
by 
  sorry

end solve_x_l1524_152442


namespace projection_of_orthogonal_vectors_l1524_152453

noncomputable def proj (u v : ℝ × ℝ) : ℝ × ℝ :=
  let scale := (u.1 * v.1 + u.2 * v.2) / (v.1 * v.1 + v.2 * v.2)
  (scale * v.1, scale * v.2)

theorem projection_of_orthogonal_vectors
  (a b : ℝ × ℝ)
  (h_orthogonal : a.1 * b.1 + a.2 * b.2 = 0)
  (h_proj : proj (4, -2) a = (4 / 5, 8 / 5)) :
  proj (4, -2) b = (16 / 5, -18 / 5) :=
sorry

end projection_of_orthogonal_vectors_l1524_152453


namespace monotonic_iff_a_range_l1524_152407

noncomputable def f (a x : ℝ) : ℝ := x^3 + a * x^2 + (a + 6) * x + 1

theorem monotonic_iff_a_range (a : ℝ) : 
  (∀ x1 x2, x1 ≤ x2 → f a x1 ≤ f a x2 ∨ f a x1 ≥ f a x2) ↔ (-3 < a ∧ a < 6) :=
by 
  sorry

end monotonic_iff_a_range_l1524_152407


namespace arithmetic_sqrt_of_9_l1524_152497

theorem arithmetic_sqrt_of_9 : ∃ y : ℝ, y ^ 2 = 9 ∧ y ≥ 0 ∧ y = 3 := by
  sorry

end arithmetic_sqrt_of_9_l1524_152497


namespace number_of_graphing_calculators_in_class_l1524_152448

-- Define a structure for the problem
structure ClassData where
  num_boys : ℕ
  num_girls : ℕ
  num_scientific_calculators : ℕ
  num_girls_with_calculators : ℕ
  num_graphing_calculators : ℕ
  no_overlap : Prop

-- Instantiate the problem using given conditions
def mrs_anderson_class : ClassData :=
{
  num_boys := 20,
  num_girls := 18,
  num_scientific_calculators := 30,
  num_girls_with_calculators := 15,
  num_graphing_calculators := 10,
  no_overlap := true
}

-- Lean statement for the proof problem
theorem number_of_graphing_calculators_in_class (data : ClassData) :
  data.num_graphing_calculators = 10 :=
by
  sorry

end number_of_graphing_calculators_in_class_l1524_152448


namespace totalBooksOnShelves_l1524_152435

-- Define the conditions
def numShelves : Nat := 150
def booksPerShelf : Nat := 15

-- Define the statement to be proved
theorem totalBooksOnShelves : numShelves * booksPerShelf = 2250 :=
by
  -- Skipping the proof
  sorry

end totalBooksOnShelves_l1524_152435

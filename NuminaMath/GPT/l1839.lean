import Mathlib

namespace range_of_m_l1839_183914

theorem range_of_m (m : ℝ) 
  (hp : ∀ x : ℝ, 2 * x > m * (x ^ 2 + 1)) 
  (hq : ∃ x0 : ℝ, x0 ^ 2 + 2 * x0 - m - 1 = 0) : 
  -2 ≤ m ∧ m < -1 :=
sorry

end range_of_m_l1839_183914


namespace integer_triplet_solution_l1839_183949

def circ (a b : ℤ) : ℤ := a + b - a * b

theorem integer_triplet_solution (x y z : ℤ) :
  circ (circ x y) z + circ (circ y z) x + circ (circ z x) y = 0 ↔
  (x = 0 ∧ y = 0 ∧ z = 2) ∨ (x = 0 ∧ y = 2 ∧ z = 0) ∨ (x = 2 ∧ y = 0 ∧ z = 0) :=
by
  sorry

end integer_triplet_solution_l1839_183949


namespace find_budget_l1839_183996

variable (B : ℝ)

-- Conditions provided
axiom cond1 : 0.30 * B = 300

theorem find_budget : B = 1000 :=
by
  -- Notes:
  -- The proof will go here.
  sorry

end find_budget_l1839_183996


namespace angle_C_in_triangle_l1839_183936

theorem angle_C_in_triangle {A B C : ℝ} 
  (h1 : A - B = 10) 
  (h2 : B = 0.5 * A) : 
  C = 150 :=
by
  -- Placeholder for proof
  sorry

end angle_C_in_triangle_l1839_183936


namespace area_ratio_of_squares_l1839_183929

theorem area_ratio_of_squares (s L : ℝ) 
  (H : 4 * L = 4 * 4 * s) : (L^2) = 16 * (s^2) :=
by
  -- assuming the utilization of the given condition
  sorry

end area_ratio_of_squares_l1839_183929


namespace loss_percentage_initially_l1839_183993

theorem loss_percentage_initially 
  (SP : ℝ) 
  (CP : ℝ := 400) 
  (h1 : SP + 100 = 1.05 * CP) : 
  (1 - SP / CP) * 100 = 20 := 
by 
  sorry

end loss_percentage_initially_l1839_183993


namespace max_value_fraction_l1839_183987

theorem max_value_fraction (a b : ℝ) 
  (h1 : a + b - 2 ≥ 0)
  (h2 : b - a - 1 ≤ 0)
  (h3 : a ≤ 1) :
  ∃ max_val, max_val = 7 / 5 ∧ ∀ (x y : ℝ), 
    (x + y - 2 ≥ 0) → (y - x - 1 ≤ 0) → (x ≤ 1) → (x + 2*y) / (2*x + y) ≤ max_val :=
sorry

end max_value_fraction_l1839_183987


namespace second_number_is_22_l1839_183913

noncomputable section

variables (x y : ℕ)

-- Definitions based on the conditions
-- Condition 1: The sum of two numbers is 33
def sum_condition : Prop := x + y = 33

-- Condition 2: The second number is twice the first number
def twice_condition : Prop := y = 2 * x

-- Theorem: Given the conditions, the second number y is 22.
theorem second_number_is_22 (h1 : sum_condition x y) (h2 : twice_condition x y) : y = 22 :=
by
  sorry

end second_number_is_22_l1839_183913


namespace remainder_3_pow_17_mod_5_l1839_183908

theorem remainder_3_pow_17_mod_5 :
  (3^17) % 5 = 3 :=
by
  have h : 3^4 % 5 = 1 := by norm_num
  sorry

end remainder_3_pow_17_mod_5_l1839_183908


namespace S8_is_255_l1839_183999

-- Definitions and hypotheses
def geometric_sequence_sum (a : ℕ → ℚ) (q : ℚ) (n : ℕ) : ℚ :=
  a 0 * (1 - q^n) / (1 - q)

variables (a : ℕ → ℚ) (q : ℚ)
variable (h_geo_seq : ∀ n, a (n + 1) = a n * q)
variable (h_S2 : geometric_sequence_sum a q 2 = 3)
variable (h_S4 : geometric_sequence_sum a q 4 = 15)

-- Goal
theorem S8_is_255 : geometric_sequence_sum a q 8 = 255 := 
by {
  -- skipping the proof
  sorry
}

end S8_is_255_l1839_183999


namespace Haley_boxes_needed_l1839_183961

theorem Haley_boxes_needed (TotalMagazines : ℕ) (MagazinesPerBox : ℕ) 
  (h1 : TotalMagazines = 63) (h2 : MagazinesPerBox = 9) : 
  TotalMagazines / MagazinesPerBox = 7 := by
sorry

end Haley_boxes_needed_l1839_183961


namespace entrance_sum_2_to_3_pm_exit_sum_2_to_3_pm_no_crowd_control_at_4_pm_l1839_183955

noncomputable def f : ℕ → ℕ
| n => if 1 ≤ n ∧ n ≤ 8 then 200 * n + 2000
       else if 9 ≤ n ∧ n ≤ 32 then 360 * (3 ^ ((n - 8) / 12)) + 3000
       else if 33 ≤ n ∧ n ≤ 45 then 32400 - 720 * n
       else 0

noncomputable def g : ℕ → ℕ
| n => if 1 ≤ n ∧ n ≤ 18 then 0
       else if 19 ≤ n ∧ n ≤ 32 then 500 * n - 9000
       else if 33 ≤ n ∧ n ≤ 45 then 8800
       else 0

theorem entrance_sum_2_to_3_pm : f 21 + f 22 + f 23 + f 24 = 17460 := by
  sorry

theorem exit_sum_2_to_3_pm : g 21 + g 22 + g 23 + g 24 = 9000 := by
  sorry

theorem no_crowd_control_at_4_pm : f 28 - g 28 < 80000 := by
  sorry

end entrance_sum_2_to_3_pm_exit_sum_2_to_3_pm_no_crowd_control_at_4_pm_l1839_183955


namespace find_a_l1839_183966

-- Define the conditions and the proof goal
theorem find_a (a : ℝ) (h1 : 0 < a) (h2 : a < 1) (h_eq : a + a⁻¹ = 5/2) :
  a = 1/2 :=
by
  sorry

end find_a_l1839_183966


namespace greatest_integer_solution_l1839_183931

theorem greatest_integer_solution :
  ∃ x : ℤ, (∀ y : ℤ, (6 * (y : ℝ)^2 + 5 * (y : ℝ) - 8) < (3 * (y : ℝ)^2 - 4 * (y : ℝ) + 1) → y ≤ x) 
  ∧ (6 * (x : ℝ)^2 + 5 * (x : ℝ) - 8) < (3 * (x : ℝ)^2 - 4 * (x : ℝ) + 1) ∧ x = 0 :=
by
  sorry

end greatest_integer_solution_l1839_183931


namespace profit_percentage_l1839_183982

theorem profit_percentage (C S : ℝ) (h : 30 * C = 24 * S) :
  (S - C) / C * 100 = 25 :=
by sorry

end profit_percentage_l1839_183982


namespace train_pass_man_in_16_seconds_l1839_183963

noncomputable def speed_km_per_hr := 54
noncomputable def speed_m_per_s := (speed_km_per_hr * 1000) / 3600
noncomputable def time_to_pass_platform := 16
noncomputable def length_platform := 90.0072
noncomputable def length_train := speed_m_per_s * time_to_pass_platform
noncomputable def time_to_pass_man := length_train / speed_m_per_s

theorem train_pass_man_in_16_seconds :
  time_to_pass_man = 16 :=
by sorry

end train_pass_man_in_16_seconds_l1839_183963


namespace johns_donation_is_correct_l1839_183995

/-
Conditions:
1. Alice, Bob, and Carol donated different amounts.
2. The ratio of Alice's, Bob's, and Carol's donations is 3:2:5.
3. The sum of Alice's and Bob's donations is $120.
4. The average contribution increases by 50% and reaches $75 per person after John donates.

The statement to prove:
John's donation is $240.
-/

def donations_ratio : ℕ × ℕ × ℕ := (3, 2, 5)
def sum_Alice_Bob : ℕ := 120
def new_avg_after_john : ℕ := 75
def num_people_before_john : ℕ := 3
def avg_increase_factor : ℚ := 1.5

theorem johns_donation_is_correct (A B C J : ℕ) 
  (h1 : A * 2 = B * 3) 
  (h2 : B * 5 = C * 2) 
  (h3 : A + B = sum_Alice_Bob) 
  (h4 : (A + B + C) / num_people_before_john = 80) 
  (h5 : ((A + B + C + J) / (num_people_before_john + 1)) = new_avg_after_john) :
  J = 240 := 
sorry

end johns_donation_is_correct_l1839_183995


namespace y_intercepts_parabola_l1839_183992

theorem y_intercepts_parabola : 
  ∀ (y : ℝ), ¬(0 = 3 * y^2 - 5 * y + 12) :=
by 
  -- Given x = 0, we have the equation 3 * y^2 - 5 * y + 12 = 0.
  -- The discriminant ∆ = b^2 - 4ac = (-5)^2 - 4 * 3 * 12 = 25 - 144 = -119 which is less than 0.
  -- Since the discriminant is negative, the quadratic equation has no real roots.
  sorry

end y_intercepts_parabola_l1839_183992


namespace polynomial_inequality_l1839_183968

-- Define the polynomial P and its condition
def P (a b c : ℝ) (x : ℝ) : ℝ := 12 * x^3 + a * x^2 + b * x + c
-- Define the polynomial Q and its condition
def Q (a b c : ℝ) (x : ℝ) : ℝ := (x^2 + x + 2001)^3 + a * (x^2 + x + 2001)^2 + b * (x^2 + x + 2001) + c

-- Assumptions
axiom P_has_distinct_roots (a b c : ℝ) : ∃ p q r : ℝ, p ≠ q ∧ q ≠ r ∧ p ≠ r ∧ P a b c p = 0 ∧ P a b c q = 0 ∧ P a b c r = 0
axiom Q_has_no_real_roots (a b c : ℝ) : ¬ ∃ x : ℝ, Q a b c x = 0

-- The goal to prove
theorem polynomial_inequality (a b c : ℝ) (h1 : ∃ p q r : ℝ, p ≠ q ∧ q ≠ r ∧ p ≠ r ∧ P a b c p = 0 ∧ P a b c q = 0 ∧ P a b c r = 0)
  (h2 : ¬ ∃ x : ℝ, Q a b c x = 0) : 2001^3 + a * 2001^2 + b * 2001 + c > 1 / 64 :=
by {
  -- sorry is added to skip the proof part
  sorry
}

end polynomial_inequality_l1839_183968


namespace sum_cubes_eq_neg_27_l1839_183911

theorem sum_cubes_eq_neg_27 (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
 (h_eq : (a^3 + 9) / a = (b^3 + 9) / b ∧ (b^3 + 9) / b = (c^3 + 9) / c) :
 a^3 + b^3 + c^3 = -27 :=
by
  sorry

end sum_cubes_eq_neg_27_l1839_183911


namespace garden_perimeter_l1839_183984

noncomputable def perimeter_of_garden (w l : ℝ) (h1 : l = 3 * w + 15) (h2 : w * l = 4050) : ℝ :=
  2 * l + 2 * w

theorem garden_perimeter (w l : ℝ) (h1 : l = 3 * w + 15) (h2 : w * l = 4050) :
  perimeter_of_garden w l h1 h2 = 304.64 :=
sorry

end garden_perimeter_l1839_183984


namespace average_weight_of_a_and_b_l1839_183960

-- Define the parameters in the conditions
variables (A B C : ℝ)

-- Conditions given in the problem
theorem average_weight_of_a_and_b (h1 : (A + B + C) / 3 = 45) 
                                 (h2 : (B + C) / 2 = 43) 
                                 (h3 : B = 33) : (A + B) / 2 = 41 := 
sorry

end average_weight_of_a_and_b_l1839_183960


namespace signup_ways_l1839_183985

theorem signup_ways (students groups : ℕ) (h_students : students = 5) (h_groups : groups = 3) :
  (groups ^ students = 243) :=
by
  have calculation : 3 ^ 5 = 243 := by norm_num
  rwa [h_students, h_groups]

end signup_ways_l1839_183985


namespace num_roses_given_l1839_183943

theorem num_roses_given (n : ℕ) (m : ℕ) (x : ℕ) :
  n = 28 → 
  (∀ (b g : ℕ), b + g = n → b * g = 45 * x) →
  (num_roses : ℕ) = 4 * x →
  (num_tulips : ℕ) = 10 * num_roses →
  (num_daffodils : ℕ) = x →
  num_roses = 16 :=
by
  sorry

end num_roses_given_l1839_183943


namespace tourists_speeds_l1839_183917

theorem tourists_speeds (x y : ℝ) :
  (20 / x + 2.5 = 20 / y) →
  (20 / (x - 2) = 20 / (1.5 * y)) →
  x = 8 ∧ y = 4 :=
by
  intros h1 h2
  -- The proof would go here
  sorry

end tourists_speeds_l1839_183917


namespace union_complement_eq_l1839_183928

def U : Set Nat := {0, 1, 2, 4, 6, 8}
def M : Set Nat := {0, 4, 6}
def N : Set Nat := {0, 1, 6}
def complement (u : Set α) (s : Set α) : Set α := {x ∈ u | x ∉ s}

theorem union_complement_eq :
  M ∪ (complement U N) = {0, 2, 4, 6, 8} :=
by sorry

end union_complement_eq_l1839_183928


namespace correct_parentheses_l1839_183912

theorem correct_parentheses : (1 * 2 * 3 + 4) * 5 = 50 := by
  sorry

end correct_parentheses_l1839_183912


namespace minor_premise_wrong_l1839_183909

noncomputable def is_even_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = f x

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = -f x

def f (x : ℝ) : ℝ := x^2 + x

theorem minor_premise_wrong : ¬ is_even_function f ∧ ¬ is_odd_function f := 
by
  sorry

end minor_premise_wrong_l1839_183909


namespace harry_total_travel_time_l1839_183997

def bus_time_already_sitting : Nat := 15
def bus_time_remaining : Nat := 25
def walk_fraction := 1 / 2

def bus_time_total : Nat := bus_time_already_sitting + bus_time_remaining
def walk_time : Nat := bus_time_total * walk_fraction

theorem harry_total_travel_time : bus_time_total + walk_time = 60 := by
  sorry

end harry_total_travel_time_l1839_183997


namespace f_one_zero_inequality_solution_l1839_183926

noncomputable def f : ℝ → ℝ := sorry

axiom increasing_f : ∀ x y, 0 < x → 0 < y → x < y → f x < f y
axiom functional_eq : ∀ x y, 0 < x → 0 < y → f (x / y) = f x - f y
axiom f_six : f 6 = 1

-- Part 1: Prove that f(1) = 0
theorem f_one_zero : f 1 = 0 := sorry

-- Part 2: Prove that ∀ x ∈ (0, (-3 + sqrt 153) / 2), f(x + 3) - f(1 / x) < 2
theorem inequality_solution : ∀ x, 0 < x → x < (-3 + Real.sqrt 153) / 2 → f (x + 3) - f (1 / x) < 2 := sorry

end f_one_zero_inequality_solution_l1839_183926


namespace cake_eaten_after_four_trips_l1839_183950

-- Define the fraction of the cake eaten on each trip
def fraction_eaten (n : Nat) : ℚ :=
  (1 / 3) ^ n

-- Define the total cake eaten after four trips
def total_eaten_after_four_trips : ℚ :=
  fraction_eaten 1 + fraction_eaten 2 + fraction_eaten 3 + fraction_eaten 4

-- The mathematical statement we want to prove
theorem cake_eaten_after_four_trips : total_eaten_after_four_trips = 40 / 81 := 
by
  sorry

end cake_eaten_after_four_trips_l1839_183950


namespace largest_base_b_digits_not_18_l1839_183921

-- Definition of the problem:
-- Let n = 12^3 in base 10
def n : ℕ := 12 ^ 3

-- Definition of the conditions:
-- In base 8, 1728 (12^3 in base 10) has its digits sum to 17
def sum_of_digits_base_8 (x : ℕ) : ℕ :=
  let digits := x.digits (8)
  digits.sum

-- Proof statement
theorem largest_base_b_digits_not_18 : ∃ b : ℕ, (max b) = 8 ∧ sum_of_digits_base_8 n ≠ 18 := by
  sorry

end largest_base_b_digits_not_18_l1839_183921


namespace fraction_expression_proof_l1839_183933

theorem fraction_expression_proof :
  (1 / 8 * 1 / 9 * 1 / 28 = 1 / 2016) ∨ ((1 / 8 - 1 / 9) * 1 / 28 = 1 / 2016) :=
by
  sorry

end fraction_expression_proof_l1839_183933


namespace ultramindmaster_secret_codes_count_l1839_183951

/-- 
In the game UltraMindmaster, we need to find the total number of possible secret codes 
formed by placing pegs of any of eight different colors into five slots.
Colors may be repeated, and each slot must be filled.
-/
theorem ultramindmaster_secret_codes_count :
  let colors := 8
  let slots := 5
  colors ^ slots = 32768 := by
    sorry

end ultramindmaster_secret_codes_count_l1839_183951


namespace horner_operations_count_l1839_183918

def polynomial (x : ℝ) : ℝ := 5*x^5 + 4*x^4 + 3*x^3 + 2*x^2 + x + 1

def horner_polynomial (x : ℝ) := (((((5*x + 4)*x + 3)*x + 2)*x + 1)*x + 1)

theorem horner_operations_count (x : ℝ) : 
    (polynomial x = horner_polynomial x) → 
    (x = 2) → 
    (mul_ops : ℕ) = 5 → 
    (add_ops : ℕ) = 5 := 
by 
  sorry

end horner_operations_count_l1839_183918


namespace penultimate_digit_odd_of_square_last_digit_six_l1839_183976

theorem penultimate_digit_odd_of_square_last_digit_six 
  (n : ℕ) 
  (h : (n * n) % 10 = 6) : 
  ((n * n) / 10) % 2 = 1 :=
sorry

end penultimate_digit_odd_of_square_last_digit_six_l1839_183976


namespace vector_subtraction_l1839_183900

theorem vector_subtraction (p q: ℝ × ℝ × ℝ) (hp: p = (5, -3, 2)) (hq: q = (-1, 4, -2)) :
  p - 2 • q = (7, -11, 6) :=
by
  sorry

end vector_subtraction_l1839_183900


namespace expected_value_of_game_l1839_183989

theorem expected_value_of_game :
  let heads_prob := 1 / 4
  let tails_prob := 1 / 2
  let edge_prob := 1 / 4
  let gain_heads := 4
  let loss_tails := -3
  let gain_edge := 0
  let expected_value := heads_prob * gain_heads + tails_prob * loss_tails + edge_prob * gain_edge
  expected_value = -0.5 :=
by
  sorry

end expected_value_of_game_l1839_183989


namespace exact_time_now_l1839_183941

/-- Given that it is between 9:00 and 10:00 o'clock,
and nine minutes from now, the minute hand of a watch
will be exactly opposite the place where the hour hand
was six minutes ago, show that the exact time now is 9:06
-/
theorem exact_time_now 
  (t : ℕ)
  (h1 : t < 60)
  (h2 : ∃ t, 6 * (t + 9) - (270 + 0.5 * (t - 6)) = 180 ∨ 6 * (t + 9) - (270 + 0.5 * (t - 6)) = -180) :
  t = 6 := 
sorry

end exact_time_now_l1839_183941


namespace find_number_l1839_183920

theorem find_number (n : ℕ) (h : (1 / 2 : ℝ) * n + 5 = 13) : n = 16 := 
by
  sorry

end find_number_l1839_183920


namespace right_triangles_not_1000_l1839_183990

-- Definitions based on the conditions
def numPoints := 100
def numDiametricallyOppositePairs := numPoints / 2
def rightTrianglesPerPair := numPoints - 2
def totalRightTriangles := numDiametricallyOppositePairs * rightTrianglesPerPair

-- Theorem stating the final evaluation of the problem
theorem right_triangles_not_1000 :
  totalRightTriangles ≠ 1000 :=
by
  -- calculation shows it's impossible
  sorry

end right_triangles_not_1000_l1839_183990


namespace negation_of_proposition_l1839_183957

-- Definitions based on given conditions
def is_not_divisible_by_2 (n : ℤ) := n % 2 ≠ 0
def is_odd (n : ℤ) := n % 2 = 1

-- The negation proposition to be proved
theorem negation_of_proposition : ∃ n : ℤ, is_not_divisible_by_2 n ∧ ¬ is_odd n := 
sorry

end negation_of_proposition_l1839_183957


namespace parabola_line_intersect_l1839_183959

theorem parabola_line_intersect (a : ℝ) (b : ℝ) (h1 : a ≠ 0) (h2 : ∀ x : ℝ, (y = a * x^2) ↔ (y = 2 * x - 3) → (x, y) = (1, -1)) :
  a = -1 ∧ b = -1 ∧ ((x, y) = (-3, -9) ∨ (x, y) = (1, -1)) := by
  sorry

end parabola_line_intersect_l1839_183959


namespace find_LCM_l1839_183919

-- Given conditions
def A := ℕ
def B := ℕ
def h := 22
def productAB := 45276

-- The theorem we want to prove
theorem find_LCM (a b lcm : ℕ) (hcf : ℕ) 
  (H_product : a * b = productAB) (H_hcf : hcf = h) : 
  (lcm = productAB / hcf) → 
  (a * b = hcf * lcm) :=
by
  intros H_lcm
  sorry

end find_LCM_l1839_183919


namespace luke_plays_14_rounds_l1839_183930

theorem luke_plays_14_rounds (total_points : ℕ) (points_per_round : ℕ)
  (h1 : total_points = 154) (h2 : points_per_round = 11) : 
  total_points / points_per_round = 14 := by
  sorry

end luke_plays_14_rounds_l1839_183930


namespace train_length_correct_l1839_183978

noncomputable def train_length (speed_kmh: ℝ) (time_s: ℝ) : ℝ :=
  let speed_ms := (speed_kmh * 1000) / 3600
  speed_ms * time_s

theorem train_length_correct :
  train_length 60 15 = 250.05 := 
by
  sorry

end train_length_correct_l1839_183978


namespace danielle_money_for_supplies_l1839_183962

-- Define the conditions
def cost_of_molds := 3
def cost_of_sticks_pack := 1
def sticks_in_pack := 100
def cost_of_juice_bottle := 2
def popsicles_per_bottle := 20
def remaining_sticks := 40
def used_sticks := sticks_in_pack - remaining_sticks

-- Define number of juice bottles used
def bottles_of_juice_used : ℕ := used_sticks / popsicles_per_bottle

-- Define the total cost
def total_cost : ℕ := cost_of_molds + cost_of_sticks_pack + bottles_of_juice_used * cost_of_juice_bottle

-- Prove that Danielle had $10 for supplies
theorem danielle_money_for_supplies : total_cost = 10 := by {
  sorry
}

end danielle_money_for_supplies_l1839_183962


namespace total_votes_cast_l1839_183956

theorem total_votes_cast (V: ℕ) (invalid_votes: ℕ) (diff_votes: ℕ) 
  (H1: invalid_votes = 200) 
  (H2: diff_votes = 700) 
  (H3: (0.01 : ℝ) * V = diff_votes) 
  : (V + invalid_votes = 70200) :=
by
  sorry

end total_votes_cast_l1839_183956


namespace solution_set_of_inequality_l1839_183947

-- Definition of the inequality and its transformation
def inequality (x : ℝ) : Prop :=
  (x - 2) / (x + 1) ≤ 0

noncomputable def transformed_inequality (x : ℝ) : Prop :=
  (x + 1) * (x - 2) ≤ 0 ∧ x + 1 ≠ 0

-- Statement of the theorem
theorem solution_set_of_inequality :
  {x : ℝ | inequality x} = {x : ℝ | -1 < x ∧ x ≤ 2} := 
sorry

end solution_set_of_inequality_l1839_183947


namespace problem_solution_l1839_183932

theorem problem_solution (k : ℤ) : k ≤ 0 ∧ -2 < k → k = -1 ∨ k = 0 :=
by
  sorry

end problem_solution_l1839_183932


namespace mary_characters_initials_l1839_183935

theorem mary_characters_initials :
  ∀ (total_A total_C total_D total_E : ℕ),
  total_A = 60 / 2 →
  total_C = total_A / 2 →
  total_D = 2 * total_E →
  total_A + total_C + total_D + total_E = 60 →
  total_D = 10 :=
by
  intros total_A total_C total_D total_E hA hC hDE hSum
  sorry

end mary_characters_initials_l1839_183935


namespace find_a_l1839_183945

noncomputable def binomial_coeff (n k : ℕ) := Nat.choose n k

theorem find_a (a : ℝ) 
  (h : ∃ (a : ℝ), a ^ 3 * binomial_coeff 8 3 = 56) : a = 1 :=
by
  sorry

end find_a_l1839_183945


namespace range_x1_x2_l1839_183948

noncomputable def g (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

def f (a b c : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x + c

theorem range_x1_x2 (a b c d x1 x2 : ℝ)
  (h1 : a ≠ 0)
  (h2 : a + 2 * b + 3 * c = 0)
  (h3 : f a b c 0 * f a b c 1 > 0)
  (hx1 : f a b c x1 = 0)
  (hx2 : f a b c x2 = 0) :
  abs (x1 - x2) ∈ Set.Ico 0 (2 / 3) :=
sorry

end range_x1_x2_l1839_183948


namespace largest_interior_angle_of_triangle_l1839_183904

theorem largest_interior_angle_of_triangle (a b c ext : ℝ)
    (h1 : a + b + c = 180)
    (h2 : a / 4 = b / 5)
    (h3 : a / 4 = c / 6)
    (h4 : c + 120 = a + 180) : c = 72 :=
by
  sorry

end largest_interior_angle_of_triangle_l1839_183904


namespace total_price_for_pizza_l1839_183903

-- Definitions based on conditions
def num_friends : ℕ := 5
def amount_per_person : ℕ := 8

-- The claim to be proven
theorem total_price_for_pizza : num_friends * amount_per_person = 40 := by
  -- Since the proof detail is not required, we use 'sorry' to skip the proof.
  sorry

end total_price_for_pizza_l1839_183903


namespace min_value_of_M_l1839_183975

theorem min_value_of_M (P : ℕ → ℝ) (n : ℕ) (M : ℝ):
  (P 1 = 9 / 11) →
  (∀ n ≥ 2, P n = (3 / 4) * (P (n - 1)) + (2 / 3) * (1 - P (n - 1))) →
  (∀ n ≥ 2, P n ≤ M) →
  (M = 97 / 132) := 
sorry

end min_value_of_M_l1839_183975


namespace find_m_l1839_183981

theorem find_m (m : ℝ) (x : ℝ) (h : x = 1) (h_eq : (m / (2 - x)) - (1 / (x - 2)) = 3) : m = 2 :=
sorry

end find_m_l1839_183981


namespace instantaneous_velocity_at_4_seconds_l1839_183988

-- Define the equation of motion
def s (t : ℝ) : ℝ := t^2 - 2 * t + 5

-- Define the velocity function as the derivative of s
def v (t : ℝ) : ℝ := 2 * t - 2

theorem instantaneous_velocity_at_4_seconds : v 4 = 6 := by
  -- Proof goes here
  sorry

end instantaneous_velocity_at_4_seconds_l1839_183988


namespace tangent_line_tangent_value_at_one_l1839_183942
noncomputable def f : ℝ → ℝ := sorry -- Placeholder for the function f

theorem tangent_line_tangent_value_at_one
  (f : ℝ → ℝ)
  (hf1 : f 1 = 3 - 1 / 2)
  (hf'1 : deriv f 1 = 1 / 2)
  (tangent_eq : ∀ x, f 1 + deriv f 1 * (x - 1) = 1 / 2 * x + 2) :
  f 1 + deriv f 1 = 3 :=
by sorry

end tangent_line_tangent_value_at_one_l1839_183942


namespace equation_of_line_passing_through_ellipse_midpoint_l1839_183910

noncomputable def ellipse (x y : ℝ) : Prop := (x^2 / 4) + (y^2 / 3) = 1

theorem equation_of_line_passing_through_ellipse_midpoint
  (x1 y1 x2 y2 : ℝ)
  (P : ℝ × ℝ)
  (hP : P = (1, 1))
  (hA : ellipse x1 y1)
  (hB : ellipse x2 y2)
  (midAB : (x1 + x2) / 2 = 1 ∧ (y1 + y2) / 2 = 1) :
  ∃ (a b c : ℝ), a = 4 ∧ b = 3 ∧ c = -7 ∧ a * P.2 + b * P.1 + c = 0 :=
sorry

end equation_of_line_passing_through_ellipse_midpoint_l1839_183910


namespace y_values_relation_l1839_183938

theorem y_values_relation :
  ∀ y1 y2 y3 : ℝ,
    (y1 = (-3 + 1) ^ 2 + 1) →
    (y2 = (0 + 1) ^ 2 + 1) →
    (y3 = (2 + 1) ^ 2 + 1) →
    y2 < y1 ∧ y1 < y3 :=
by
  sorry

end y_values_relation_l1839_183938


namespace scout_troop_profit_l1839_183923

noncomputable def buy_price_per_bar : ℚ := 3 / 4
noncomputable def sell_price_per_bar : ℚ := 2 / 3
noncomputable def num_candy_bars : ℕ := 800

theorem scout_troop_profit :
  num_candy_bars * (sell_price_per_bar : ℚ) - num_candy_bars * (buy_price_per_bar : ℚ) = -66.64 :=
by
  sorry

end scout_troop_profit_l1839_183923


namespace remainder_and_division_l1839_183925

theorem remainder_and_division (x y : ℕ) (h1 : x % y = 8) (h2 : (x / y : ℝ) = 76.4) : y = 20 :=
sorry

end remainder_and_division_l1839_183925


namespace johns_age_less_than_six_times_brothers_age_l1839_183922

theorem johns_age_less_than_six_times_brothers_age 
  (B J : ℕ) 
  (h1 : B = 8) 
  (h2 : J + B = 10) 
  (h3 : J = 6 * B - 46) : 
  6 * B - J = 46 :=
by
  rw [h1, h3]
  exact sorry

end johns_age_less_than_six_times_brothers_age_l1839_183922


namespace magnitude_2a_minus_b_l1839_183915

variables (a b : EuclideanSpace ℝ (Fin 2))
variables (θ : ℝ) (h_angle : θ = 5 * Real.pi / 6)
variables (h_mag_a : ‖a‖ = 4) (h_mag_b : ‖b‖ = Real.sqrt 3)

theorem magnitude_2a_minus_b :
  ‖2 • a - b‖ = Real.sqrt 91 := by
  -- Proof goes here.
  sorry

end magnitude_2a_minus_b_l1839_183915


namespace chord_length_invalid_l1839_183974

-- Define the circle radius
def radius : ℝ := 5

-- Define the maximum possible chord length in terms of the diameter
def max_chord_length (r : ℝ) : ℝ := 2 * r

-- The problem statement proving that 11 cannot be a chord length given the radius is 5
theorem chord_length_invalid : ¬ (11 ≤ max_chord_length radius) :=
by {
  sorry
}

end chord_length_invalid_l1839_183974


namespace peter_fraction_equiv_l1839_183927

def fraction_pizza_peter_ate (total_slices : ℕ) (slices_ate_alone : ℕ) (shared_slices_brother : ℚ) (shared_slices_sister : ℚ) : ℚ :=
  (slices_ate_alone / total_slices) + (shared_slices_brother / total_slices) + (shared_slices_sister / total_slices)

theorem peter_fraction_equiv :
  fraction_pizza_peter_ate 16 3 (1/2) (1/2) = 1/4 :=
by
  sorry

end peter_fraction_equiv_l1839_183927


namespace cos_alpha_minus_pi_l1839_183986

theorem cos_alpha_minus_pi (α : Real) (h : Real.sin (α / 2) = Real.sqrt 3 / 4) : 
  Real.cos (α - Real.pi) = -5 / 8 :=
sorry

end cos_alpha_minus_pi_l1839_183986


namespace gingerbread_price_today_is_5_l1839_183940

-- Given conditions
variables {x y a b k m : ℤ}

-- Price constraints
axiom price_constraint_yesterday : 9 * x + 7 * y < 100
axiom price_constraint_today1 : 9 * a + 7 * b > 100
axiom price_constraint_today2 : 2 * a + 11 * b < 100

-- Price change constraints
axiom price_change_gingerbread : a = x + k
axiom price_change_pastries : b = y + m
axiom gingerbread_change_range : |k| ≤ 1
axiom pastries_change_range : |m| ≤ 1

theorem gingerbread_price_today_is_5 : a = 5 :=
by
  sorry

end gingerbread_price_today_is_5_l1839_183940


namespace total_boys_in_school_l1839_183973

-- Define the total percentage of boys belonging to other communities
def percentage_other_communities := 100 - (44 + 28 + 10)

-- Total number of boys in the school, represented by a variable B
def total_boys (B : ℕ) : Prop :=
0.18 * (B : ℝ) = 117

-- The theorem states that the total number of boys B is 650
theorem total_boys_in_school : ∃ B : ℕ, total_boys B ∧ B = 650 :=
sorry

end total_boys_in_school_l1839_183973


namespace problem_statement_l1839_183972

def f(x : ℝ) : ℝ := 3 * x - 3
def g(x : ℝ) : ℝ := x^2 + 1

theorem problem_statement : f (1 + g 2) = 15 := by
  sorry

end problem_statement_l1839_183972


namespace projection_of_point_onto_xOy_plane_l1839_183979

def point := (ℝ × ℝ × ℝ)

def projection_onto_xOy_plane (P : point) : point :=
  let (x, y, z) := P
  (x, y, 0)

theorem projection_of_point_onto_xOy_plane : 
  projection_onto_xOy_plane (2, 3, 4) = (2, 3, 0) :=
by
  -- proof steps would go here
  sorry

end projection_of_point_onto_xOy_plane_l1839_183979


namespace find_abc_l1839_183906

theorem find_abc (a b c : ℤ) (h1 : a < b) (h2 : b < c) (h3 : c < 4)
  (h4 : a + b + c = a * b * c) : (a = 1 ∧ b = 2 ∧ c = 3) ∨ 
                                 (a = -3 ∧ b = -2 ∧ c = -1) ∨ 
                                 (a = -1 ∧ b = 0 ∧ c = 1) ∨ 
                                 (a = -2 ∧ b = 0 ∧ c = 2) ∨ 
                                 (a = -3 ∧ b = 0 ∧ c = 3) :=
sorry

end find_abc_l1839_183906


namespace combined_tax_rate_l1839_183902

theorem combined_tax_rate (Mork_income Mindy_income : ℝ) (Mork_tax_rate Mindy_tax_rate : ℝ)
  (h1 : Mork_tax_rate = 0.4) (h2 : Mindy_tax_rate = 0.3) (h3 : Mindy_income = 4 * Mork_income) :
  ((Mork_tax_rate * Mork_income + Mindy_tax_rate * Mindy_income) / (Mork_income + Mindy_income)) * 100 = 32 :=
by
  sorry

end combined_tax_rate_l1839_183902


namespace borrowed_amount_l1839_183964

theorem borrowed_amount (P : ℝ) (h1 : (9 / 100) * P - (8 / 100) * P = 200) : P = 20000 :=
  by sorry

end borrowed_amount_l1839_183964


namespace scores_greater_than_18_l1839_183980

theorem scores_greater_than_18 (scores : Fin 20 → ℝ) 
  (h_unique : Function.Injective scores)
  (h_sum : ∀ i j k : Fin 20, i ≠ j → i ≠ k → j ≠ k → scores i < scores j + scores k) :
  ∀ i : Fin 20, scores i > 18 := 
by
  sorry

end scores_greater_than_18_l1839_183980


namespace arithmetic_sequence_l1839_183958

theorem arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) (h₀ : a 1 = 2) (h₁ : a 2 + a 3 = 13)
    (h₂ : ∀ n, a n = a 1 + (n - 1) * d) : a 5 = 14 :=
by
  sorry

end arithmetic_sequence_l1839_183958


namespace geometric_sequence_sum_l1839_183994

variable {α : Type*} 
variable [LinearOrderedField α]

def is_geometric_sequence (a : ℕ → α) : Prop :=
  ∃ q : α, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → α) (h : is_geometric_sequence a) 
  (h1 : a 0 + a 1 = 20) 
  (h2 : a 2 + a 3 = 40) : 
  a 4 + a 5 = 80 :=
sorry

end geometric_sequence_sum_l1839_183994


namespace minimal_fraction_difference_l1839_183939

theorem minimal_fraction_difference (p q : ℕ) (hp : 0 < p) (hq : 0 < q) 
  (h1 : 3 / 5 < p / q) (h2 : p / q < 2 / 3) (hmin: ∀ r s : ℕ, (3 / 5 < r / s ∧ r / s < 2 / 3 ∧ s < q) → false) :
  q - p = 11 := 
sorry

end minimal_fraction_difference_l1839_183939


namespace find_original_number_l1839_183946

theorem find_original_number (x : ℝ) : 1.5 * x = 525 → x = 350 := by
  sorry

end find_original_number_l1839_183946


namespace geometric_sequence_sum_six_l1839_183977

theorem geometric_sequence_sum_six (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) 
  (h1 : 0 < q)
  (h2 : a 1 = 1)
  (h3 : a 3 * a 5 = 64)
  (h4 : ∀ n, a n = a 1 * q^(n-1))
  (h5 : ∀ n, S n = (a 1 * (1 - q^n)) / (1 - q)) :
  S 6 = 63 := 
sorry

end geometric_sequence_sum_six_l1839_183977


namespace washington_high_teacher_student_ratio_l1839_183967

theorem washington_high_teacher_student_ratio (students teachers : ℕ) (h_students : students = 1155) (h_teachers : teachers = 42) : (students / teachers : ℚ) = 27.5 :=
by
  sorry

end washington_high_teacher_student_ratio_l1839_183967


namespace steve_initial_amount_l1839_183905

theorem steve_initial_amount
  (P : ℝ) 
  (h : (1.1^2) * P = 121) : 
  P = 100 := 
by 
  sorry

end steve_initial_amount_l1839_183905


namespace circle_equation_and_range_of_a_l1839_183916

theorem circle_equation_and_range_of_a :
  (∃ m : ℤ, (x - m)^2 + y^2 = 25 ∧ (abs (4 * m - 29)) = 25) ∧
  (∀ a : ℝ, (a > 0 → (4 * (5 * a - 1)^2 - 4 * (a^2 + 1) > 0 → a > 5 / 12 ∨ a < 0))) :=
by
  sorry

end circle_equation_and_range_of_a_l1839_183916


namespace equation_value_l1839_183944

-- Define the expressions
def a := 10 + 3
def b := 7 - 5

-- State the theorem
theorem equation_value : a^2 + b^2 = 173 := by
  sorry

end equation_value_l1839_183944


namespace union_rational_irrational_is_real_intersection_rational_irrational_is_empty_l1839_183965

section
  def A : Set ℝ := {x : ℝ | ∃ q : ℚ, x = q}
  def B : Set ℝ := {x : ℝ | ¬ ∃ q : ℚ, x = q}

  theorem union_rational_irrational_is_real : A ∪ B = Set.univ :=
  by
    sorry

  theorem intersection_rational_irrational_is_empty : A ∩ B = ∅ :=
  by
    sorry
end

end union_rational_irrational_is_real_intersection_rational_irrational_is_empty_l1839_183965


namespace sin_double_angle_value_l1839_183937

theorem sin_double_angle_value (α : ℝ) (h1 : 0 < α ∧ α < π)
  (h2 : (1/2) * Real.cos (2 * α) = Real.sin (π/4 + α)) :
  Real.sin (2 * α) = -1 :=
by
  sorry

end sin_double_angle_value_l1839_183937


namespace quadrilateral_angle_contradiction_l1839_183969

theorem quadrilateral_angle_contradiction (a b c d : ℝ)
  (h : 0 < a ∧ a < 180 ∧ 0 < b ∧ b < 180 ∧ 0 < c ∧ c < 180 ∧ 0 < d ∧ d < 180)
  (sum_eq_360 : a + b + c + d = 360) :
  (¬ (a ≤ 90 ∨ b ≤ 90 ∨ c ≤ 90 ∨ d ≤ 90)) → (90 < a ∧ 90 < b ∧ 90 < c ∧ 90 < d) :=
sorry

end quadrilateral_angle_contradiction_l1839_183969


namespace sarah_ellie_total_reflections_l1839_183924

def sarah_tall_reflections : ℕ := 10
def sarah_wide_reflections : ℕ := 5
def sarah_narrow_reflections : ℕ := 8

def ellie_tall_reflections : ℕ := 6
def ellie_wide_reflections : ℕ := 3
def ellie_narrow_reflections : ℕ := 4

def tall_mirror_passages : ℕ := 3
def wide_mirror_passages : ℕ := 5
def narrow_mirror_passages : ℕ := 4

def total_reflections (sarah_tall sarah_wide sarah_narrow ellie_tall ellie_wide ellie_narrow
    tall_passages wide_passages narrow_passages : ℕ) : ℕ :=
  (sarah_tall * tall_passages + sarah_wide * wide_passages + sarah_narrow * narrow_passages) +
  (ellie_tall * tall_passages + ellie_wide * wide_passages + ellie_narrow * narrow_passages)

theorem sarah_ellie_total_reflections :
  total_reflections sarah_tall_reflections sarah_wide_reflections sarah_narrow_reflections
  ellie_tall_reflections ellie_wide_reflections ellie_narrow_reflections
  tall_mirror_passages wide_mirror_passages narrow_mirror_passages = 136 :=
by
  sorry

end sarah_ellie_total_reflections_l1839_183924


namespace number_of_skirts_l1839_183991

theorem number_of_skirts (T Ca Cs S : ℕ) (hT : T = 50) (hCa : Ca = 20) (hCs : Cs = 15) (hS : T - Ca = S * Cs) : S = 2 := by
  sorry

end number_of_skirts_l1839_183991


namespace alien_collected_95_units_l1839_183971

def convert_base_six_to_ten (n : ℕ) : ℕ :=
  match n with
  | 235 => 2 * 6^2 + 3 * 6^1 + 5 * 6^0
  | _ => 0

theorem alien_collected_95_units : convert_base_six_to_ten 235 = 95 := by
  sorry

end alien_collected_95_units_l1839_183971


namespace determine_d_iff_l1839_183998

theorem determine_d_iff (x : ℝ) : 
  (x ∈ Set.Ioo (-5/2) 3) ↔ (x * (2 * x + 3) < 15) :=
by
  sorry

end determine_d_iff_l1839_183998


namespace solve_arcsin_cos_eq_x_over_3_l1839_183970

noncomputable def arcsin (x : Real) : Real := sorry
noncomputable def cos (x : Real) : Real := sorry

theorem solve_arcsin_cos_eq_x_over_3 :
  ∀ x,
  - (3 * Real.pi / 2) ≤ x ∧ x ≤ 3 * Real.pi / 2 →
  arcsin (cos x) = x / 3 →
  x = 3 * Real.pi / 10 ∨ x = 3 * Real.pi / 8 :=
sorry

end solve_arcsin_cos_eq_x_over_3_l1839_183970


namespace jessies_weight_after_first_week_l1839_183952

-- Definitions from the conditions
def initial_weight : ℕ := 92
def first_week_weight_loss : ℕ := 56

-- The theorem statement
theorem jessies_weight_after_first_week : initial_weight - first_week_weight_loss = 36 := by
  -- Skip the proof
  sorry

end jessies_weight_after_first_week_l1839_183952


namespace sushi_father_lollipops_l1839_183953

-- Define the conditions
def lollipops_eaten : ℕ := 5
def lollipops_left : ℕ := 7

-- Define the total number of lollipops brought
def total_lollipops := lollipops_eaten + lollipops_left

-- Proof statement
theorem sushi_father_lollipops : total_lollipops = 12 := sorry

end sushi_father_lollipops_l1839_183953


namespace find_common_chord_l1839_183954

variable (x y : ℝ)

def circle1 (x y : ℝ) := x^2 + y^2 + 2*x + 3*y = 0
def circle2 (x y : ℝ) := x^2 + y^2 - 4*x + 2*y + 1 = 0
def common_chord (x y : ℝ) := 6*x + y - 1 = 0

theorem find_common_chord (x y : ℝ) (h1 : circle1 x y) (h2 : circle2 x y) : common_chord x y :=
by
  sorry

end find_common_chord_l1839_183954


namespace total_population_l1839_183934

theorem total_population (x T : ℝ) (h : 128 = (x / 100) * (50 / 100) * T) : T = 25600 / x :=
by
  sorry

end total_population_l1839_183934


namespace garden_perimeter_is_56_l1839_183983

-- Define the conditions
def garden_width : ℕ := 12
def playground_length : ℕ := 16
def playground_width : ℕ := 12
def playground_area : ℕ := playground_length * playground_width
def garden_length : ℕ := playground_area / garden_width
def garden_perimeter : ℕ := 2 * (garden_length + garden_width)

-- Statement to prove
theorem garden_perimeter_is_56 :
  garden_perimeter = 56 := by
sorry

end garden_perimeter_is_56_l1839_183983


namespace commission_rate_correct_l1839_183901

-- Define the given conditions
def base_pay := 190
def goal_earnings := 500
def required_sales := 7750

-- Define the commission rate function
def commission_rate (sales commission : ℕ) : ℚ := (commission : ℚ) / (sales : ℚ) * 100

-- The main statement to prove
theorem commission_rate_correct :
  commission_rate required_sales (goal_earnings - base_pay) = 4 :=
by
  sorry

end commission_rate_correct_l1839_183901


namespace polynomial_product_linear_term_zero_const_six_l1839_183907

theorem polynomial_product_linear_term_zero_const_six (a b : ℝ)
  (h1 : (a + 2 * b = 0)) 
  (h2 : b = 6) : (a + b = -6) :=
by
  sorry

end polynomial_product_linear_term_zero_const_six_l1839_183907

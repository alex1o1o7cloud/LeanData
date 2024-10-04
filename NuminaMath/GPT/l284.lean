import Mathlib

namespace cos_seven_pi_over_six_l284_284909

theorem cos_seven_pi_over_six : Real.cos (7 * Real.pi / 6) = -Real.sqrt 3 / 2 :=
by
  sorry

end cos_seven_pi_over_six_l284_284909


namespace line_parallel_l284_284825

theorem line_parallel (a : ℝ) : (∀ x y : ℝ, ax + y = 0) ↔ (x + ay + 1 = 0) → a = 1 ∨ a = -1 := 
sorry

end line_parallel_l284_284825


namespace complementary_angle_difference_l284_284272

def is_complementary (a b : ℝ) : Prop := a + b = 90

def in_ratio (a b : ℝ) (m n : ℝ) : Prop := a / b = m / n

theorem complementary_angle_difference (a b : ℝ) (h1 : is_complementary a b) (h2 : in_ratio a b 5 1) : abs (a - b) = 60 := 
by
  sorry

end complementary_angle_difference_l284_284272


namespace negation_of_forall_inequality_l284_284542

theorem negation_of_forall_inequality :
  (¬ (∀ x : ℝ, x > 0 → x * Real.sin x < 2^x - 1)) ↔ (∃ x : ℝ, x > 0 ∧ x * Real.sin x ≥ 2^x - 1) :=
by sorry

end negation_of_forall_inequality_l284_284542


namespace neg_half_to_fourth_power_eq_sixteenth_l284_284340

theorem neg_half_to_fourth_power_eq_sixteenth :
  (- (1 / 2 : ℚ)) ^ 4 = (1 / 16 : ℚ) :=
sorry

end neg_half_to_fourth_power_eq_sixteenth_l284_284340


namespace xyz_value_l284_284097

-- We define the constants from the problem
variables {x y z : ℂ}

-- Here's the theorem statement in Lean 4.
theorem xyz_value :
  (x * y + 5 * y = -20) →
  (y * z + 5 * z = -20) →
  (z * x + 5 * x = -20) →
  x * y * z = 100 :=
by
  intros h1 h2 h3
  sorry

end xyz_value_l284_284097


namespace tan_105_l284_284594

theorem tan_105 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  -- Definitions
  let tan45 := Real.tan (45 * Real.pi / 180)
  let tan60 := Real.tan (60 * Real.pi / 180)
  have h1 : tan45 = 1 := sorry
  have h2 : tan60 = Real.sqrt 3 := sorry
  have h3 : tan45 + tan60 = 1 + Real.sqrt 3 := sorry
  have h4 : 1 - tan45 * tan60 = 1 - 1 * Real.sqrt 3 := sorry
  
  -- Use tangent addition formula
  have tan_addition : Real.tan (105 * Real.pi / 180) = (1 + Real.sqrt 3) / (1 - Real.sqrt 3) := sorry
  
  -- Simplify and rationalize to prove the final result
  have tan_105_eq : (1 + Real.sqrt 3) / (1 - Real.sqrt 3) = -2 - Real.sqrt 3 := sorry
  
  exact tan_105_eq

end tan_105_l284_284594


namespace tan_105_eq_neg_2_sub_sqrt_3_l284_284618

-- Definitions
def angle105 : ℝ := 105 * (Math.pi / 180)
def angle45 : ℝ := 45 * (Math.pi / 180)
def angle60 : ℝ := 60 * (Math.pi / 180)

-- Theorem
theorem tan_105_eq_neg_2_sub_sqrt_3 :
  Real.tan angle105 = -2 - Real.sqrt 3 := by
  sorry

end tan_105_eq_neg_2_sub_sqrt_3_l284_284618


namespace lines_skew_iff_a_ne_20_l284_284029

variable {t u a : ℝ}
-- Definitions for the lines
def line1 (t : ℝ) (a : ℝ) := (2 + 3 * t, 3 + 4 * t, a + 5 * t)
def line2 (u : ℝ) := (3 + 6 * u, 2 + 5 * u, 1 + 2 * u)

-- Condition for lines to intersect
def lines_intersect (t u a : ℝ) :=
  2 + 3 * t = 3 + 6 * u ∧
  3 + 4 * t = 2 + 5 * u ∧
  a + 5 * t = 1 + 2 * u

-- The main theorem stating when lines are skew
theorem lines_skew_iff_a_ne_20 (a : ℝ) :
  (¬ ∃ t u : ℝ, lines_intersect t u a) ↔ a ≠ 20 := 
by 
  sorry

end lines_skew_iff_a_ne_20_l284_284029


namespace solve_for_y_l284_284232

noncomputable def find_angle_y : Prop :=
  let AB_CD_are_straight_lines : Prop := True
  let angle_AXB : ℕ := 70
  let angle_BXD : ℕ := 40
  let angle_CYX : ℕ := 100
  let angle_YXZ := 180 - angle_AXB - angle_BXD
  let angle_XYZ := 180 - angle_CYX
  let y := 180 - angle_YXZ - angle_XYZ
  y = 30

theorem solve_for_y : find_angle_y :=
by
  trivial

end solve_for_y_l284_284232


namespace diameter_of_lake_l284_284552

theorem diameter_of_lake (d : ℝ) (pi : ℝ) (h1 : pi = 3.14) 
  (h2 : 3.14 * d - d = 1.14) : d = 0.5327 :=
by
  sorry

end diameter_of_lake_l284_284552


namespace line_eq_489_l284_284008

theorem line_eq_489 (m b : ℤ) (h1 : m = 5) (h2 : 3 = m * 5 + b) : m + b^2 = 489 :=
by
  sorry

end line_eq_489_l284_284008


namespace max_value_10x_plus_3y_plus_12z_l284_284523

theorem max_value_10x_plus_3y_plus_12z (x y z : ℝ) 
  (h1 : 9 * x^2 + 4 * y^2 + 25 * z^2 = 1) 
  (h2 : z = 2 * y) : 
  10 * x + 3 * y + 12 * z ≤ Real.sqrt 253 :=
sorry

end max_value_10x_plus_3y_plus_12z_l284_284523


namespace sum_first_95_odds_equals_9025_l284_284888

-- Define the nth odd positive integer
def nth_odd (n : ℕ) : ℕ := 2 * n - 1

-- Define the sum of the first n odd positive integers
def sum_first_n_odds (n : ℕ) : ℕ := n^2

-- State the theorem to be proved
theorem sum_first_95_odds_equals_9025 : sum_first_n_odds 95 = 9025 :=
by
  -- We provide a placeholder for the proof
  sorry

end sum_first_95_odds_equals_9025_l284_284888


namespace tan_105_l284_284607

theorem tan_105 :
  tan 105 = -2 - sqrt 3 :=
by sorry

end tan_105_l284_284607


namespace mary_spent_total_amount_l284_284796

def cost_of_berries := 11.08
def cost_of_apples := 14.33
def cost_of_peaches := 9.31
def total_cost := 34.72

theorem mary_spent_total_amount :
  cost_of_berries + cost_of_apples + cost_of_peaches = total_cost :=
by
  sorry

end mary_spent_total_amount_l284_284796


namespace sunset_time_correct_l284_284800

theorem sunset_time_correct : 
  let sunrise := (6 * 60 + 43)       -- Sunrise time in minutes (6:43 AM)
  let daylight := (11 * 60 + 56)     -- Length of daylight in minutes (11:56)
  let sunset := (sunrise + daylight) % (24 * 60) -- Calculate sunset time considering 24-hour cycle
  let sunset_hour := sunset / 60     -- Convert sunset time back into hours
  let sunset_minute := sunset % 60   -- Calculate remaining minutes
  (sunset_hour - 12, sunset_minute) = (6, 39)    -- Convert to 12-hour format and check against 6:39 PM
:= by
  sorry

end sunset_time_correct_l284_284800


namespace percent_daisies_l284_284440

theorem percent_daisies 
    (total_flowers : ℕ)
    (yellow_flowers : ℕ)
    (yellow_tulips : ℕ)
    (blue_flowers : ℕ)
    (blue_daisies : ℕ)
    (h1 : 2 * yellow_tulips = yellow_flowers) 
    (h2 : 3 * blue_daisies = blue_flowers)
    (h3 : 10 * yellow_flowers = 7 * total_flowers) : 
    100 * (yellow_flowers / 2 + blue_daisies) = 45 * total_flowers :=
by
  sorry

end percent_daisies_l284_284440


namespace stratified_sampling_sophomores_selected_l284_284444

theorem stratified_sampling_sophomores_selected 
  (total_freshmen : ℕ) (total_sophomores : ℕ) (total_seniors : ℕ) 
  (freshmen_selected : ℕ) (selection_ratio : ℕ) :
  total_freshmen = 210 →
  total_sophomores = 270 →
  total_seniors = 300 →
  freshmen_selected = 7 →
  selection_ratio = total_freshmen / freshmen_selected →
  selection_ratio = 30 →
  total_sophomores / selection_ratio = 9 :=
by sorry

end stratified_sampling_sophomores_selected_l284_284444


namespace simplify_expression_l284_284259

theorem simplify_expression (x : ℝ) (h : x = 1) : (x - 1)^2 + (x + 1) * (x - 1) - 2 * x^2 = -2 :=
by
  sorry

end simplify_expression_l284_284259


namespace angle_of_inclination_range_l284_284731

theorem angle_of_inclination_range (a : ℝ) :
  (∃ m : ℝ, ax + (a + 1)*m + 2 = 0 ∧ (m < 0 ∨ m > 1)) ↔ (a < -1/2 ∨ a > 0) := sorry

end angle_of_inclination_range_l284_284731


namespace operation_example_l284_284353

def operation (a b : ℤ) : ℤ := 2 * a * b - b^2

theorem operation_example : operation 1 (-3) = -15 := by
  sorry

end operation_example_l284_284353


namespace number_of_tangents_l284_284529

-- Define the points and conditions
variable (A B : ℝ × ℝ)
variable (dist_AB : dist A B = 8)
variable (radius_A : ℝ := 3)
variable (radius_B : ℝ := 2)

-- The goal
theorem number_of_tangents (dist_condition : dist A B = 8) : 
  ∃ n, n = 2 :=
by
  -- skipping the proof
  sorry

end number_of_tangents_l284_284529


namespace sum_first_fifty_digits_of_decimal_of_one_over_1234_l284_284857

theorem sum_first_fifty_digits_of_decimal_of_one_over_1234 :
  let s := "00081037277147487844408427876817350238192918144683"
  let digits := s.data
  (4 * (list.sum (digits.map (λ c, (c.to_nat - '0'.to_nat)))) + (list.sum ((digits.take 6).map (λ c, (c.to_nat - '0'.to_nat)))) ) = 729 :=
by sorry

end sum_first_fifty_digits_of_decimal_of_one_over_1234_l284_284857


namespace cost_of_ground_school_l284_284537

theorem cost_of_ground_school (G : ℝ) (F : ℝ) (h1 : F = G + 625) (h2 : F = 950) :
  G = 325 :=
by
  sorry

end cost_of_ground_school_l284_284537


namespace positive_difference_in_x_coordinates_l284_284923

-- Define points for line l
def point_l1 : ℝ × ℝ := (0, 10)
def point_l2 : ℝ × ℝ := (2, 0)

-- Define points for line m
def point_m1 : ℝ × ℝ := (0, 3)
def point_m2 : ℝ × ℝ := (10, 0)

-- Define the proof statement with the given problem
theorem positive_difference_in_x_coordinates :
  let y := 20
  let slope_l := (point_l2.2 - point_l1.2) / (point_l2.1 - point_l1.1)
  let intersection_l_x := (y - point_l1.2) / slope_l + point_l1.1
  let slope_m := (point_m2.2 - point_m1.2) / (point_m2.1 - point_m1.1)
  let intersection_m_x := (y - point_m1.2) / slope_m + point_m1.1
  abs (intersection_l_x - intersection_m_x) = 54.67 := 
  sorry -- Proof goes here

end positive_difference_in_x_coordinates_l284_284923


namespace tenth_term_arithmetic_sequence_l284_284539

theorem tenth_term_arithmetic_sequence :
  ∀ (a : ℕ → ℚ), a 1 = 5/6 ∧ a 16 = 7/8 →
  a 10 = 103/120 :=
by
  sorry

end tenth_term_arithmetic_sequence_l284_284539


namespace length_of_LO_l284_284079

theorem length_of_LO (MN LO : ℝ) (alt_O_MN alt_N_LO : ℝ) (h_MN : MN = 15) 
  (h_alt_O_MN : alt_O_MN = 9) (h_alt_N_LO : alt_N_LO = 7) : 
  LO = 19 + 2 / 7 :=
by
  -- Sorry means to skip the proof.
  sorry

end length_of_LO_l284_284079


namespace smallest_odd_number_with_five_prime_factors_l284_284850

def is_prime_factor_of (p n : ℕ) : Prop :=
  p.prime ∧ p ∣ n

def is_odd (n : ℕ) : Prop :=
  ¬ 2 ∣ n

def has_five_distinct_prime_factors (n : ℕ) : Prop :=
  ∃ (p1 p2 p3 p4 p5 : ℕ), 
  p1.prime ∧ p2.prime ∧ p3.prime ∧ p4.prime ∧ p5.prime ∧ 
  p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p1 ≠ p5 ∧ 
  p2 ≠ p3 ∧ p2 ≠ p4 ∧ p2 ≠ p5 ∧ 
  p3 ≠ p4 ∧ p3 ≠ p5 ∧ 
  p4 ≠ p5 ∧ 
  p1 * p2 * p3 * p4 * p5 = n

theorem smallest_odd_number_with_five_prime_factors :
  is_odd 15015 ∧ has_five_distinct_prime_factors 15015 ∧ 
  (∀ n : ℕ, is_odd n ∧ has_five_distinct_prime_factors n → 15015 ≤ n) :=
by
  sorry

end smallest_odd_number_with_five_prime_factors_l284_284850


namespace function_value_at_minus_one_l284_284900

theorem function_value_at_minus_one :
  ( -(1:ℝ)^4 + -(1:ℝ)^3 + (1:ℝ) ) / ( -(1:ℝ)^2 + (1:ℝ) ) = 1 / 2 :=
by sorry

end function_value_at_minus_one_l284_284900


namespace domain_of_g_l284_284224

def f : ℝ → ℝ := sorry

theorem domain_of_g 
  (hf_dom : ∀ x, -2 ≤ x ∧ x ≤ 4 → f x = f x) : 
  ∀ x, -2 ≤ x ∧ x ≤ 2 ↔ (∃ y, y = f x + f (-x)) := 
by {
  sorry
}

end domain_of_g_l284_284224


namespace current_speed_correct_l284_284134

noncomputable def boat_upstream_speed : ℝ := (1 / 20) * 60
noncomputable def boat_downstream_speed : ℝ := (1 / 9) * 60
noncomputable def speed_of_current : ℝ := (boat_downstream_speed - boat_upstream_speed) / 2

theorem current_speed_correct :
  speed_of_current = 1.835 :=
by
  sorry

end current_speed_correct_l284_284134


namespace servings_of_popcorn_l284_284124

theorem servings_of_popcorn (popcorn_per_serving : ℕ) (jared_consumption : ℕ)
    (friend_consumption : ℕ) (num_friends : ℕ) :
    popcorn_per_serving = 30 →
    jared_consumption = 90 →
    friend_consumption = 60 →
    num_friends = 3 →
    (jared_consumption + num_friends * friend_consumption) / popcorn_per_serving = 9 := 
by
  intros h1 h2 h3 h4
  sorry

end servings_of_popcorn_l284_284124


namespace norris_money_left_l284_284968

-- Defining the conditions
def sept_savings : ℕ := 29
def oct_savings : ℕ := 25
def nov_savings : ℕ := 31
def dec_savings : ℕ := 35
def jan_savings : ℕ := 40

def initial_savings : ℕ := sept_savings + oct_savings + nov_savings + dec_savings + jan_savings
def interest_rate : ℝ := 0.02

def total_interest : ℝ :=
  sept_savings * interest_rate + 
  (sept_savings + oct_savings) * interest_rate + 
  (sept_savings + oct_savings + nov_savings) * interest_rate +
  (sept_savings + oct_savings + nov_savings + dec_savings) * interest_rate

def total_savings_with_interest : ℝ := initial_savings + total_interest
def hugo_owes_norris : ℕ := 20 - 10

-- The final statement to prove Norris' total amount of money
theorem norris_money_left : total_savings_with_interest + hugo_owes_norris = 175.76 := by
  sorry

end norris_money_left_l284_284968


namespace temperature_in_quebec_city_is_negative_8_l284_284548

def temperature_vancouver : ℝ := 22
def temperature_calgary (temperature_vancouver : ℝ) : ℝ := temperature_vancouver - 19
def temperature_quebec_city (temperature_calgary : ℝ) : ℝ := temperature_calgary - 11

theorem temperature_in_quebec_city_is_negative_8 :
  temperature_quebec_city (temperature_calgary temperature_vancouver) = -8 := by
  sorry

end temperature_in_quebec_city_is_negative_8_l284_284548


namespace PQ_value_l284_284242

theorem PQ_value (DE DF EF : ℕ) (CF : ℝ) (P Q : ℝ) 
  (h1 : DE = 996)
  (h2 : DF = 995)
  (h3 : EF = 994)
  (hCF :  CF = (995^2 - 4) / 1990)
  (hP : P = (1492.5 - EF))
  (hQ : Q = (s - DF)) :
  PQ = 1 ∧ m + n = 2 :=
by
  sorry

end PQ_value_l284_284242


namespace sum_of_integers_l284_284987

theorem sum_of_integers (a b : ℕ) (h1 : a * b + a + b = 255) (h2 : a < 30) (h3 : b < 30) (h4 : a % 2 = 1) :
  a + b = 30 := 
sorry

end sum_of_integers_l284_284987


namespace regular_polygon_properties_l284_284725

theorem regular_polygon_properties
  (n : ℕ)
  (h1 : (n - 2) * 180 = 3 * 360 + 180)
  (h2 : n > 2) :
  n = 9 ∧ (n - 2) * 180 / n = 140 := by
  sorry

end regular_polygon_properties_l284_284725


namespace smallest_odd_with_five_different_prime_factors_l284_284838

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

def has_five_distinct_prime_factors (n : ℕ) : Prop :=
  ∃ a b c d e : ℕ, 
    is_prime a ∧ is_prime b ∧ is_prime c ∧ is_prime d ∧ is_prime e ∧ 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e ∧ 
    n = a * b * c * d * e

theorem smallest_odd_with_five_different_prime_factors : ∃ n : ℕ, 
  is_odd n ∧ has_five_distinct_prime_factors n ∧ ∀ m : ℕ, 
  is_odd m ∧ has_five_distinct_prime_factors m → n ≤ m :=
exists.intro 15015 sorry

end smallest_odd_with_five_different_prime_factors_l284_284838


namespace parallel_and_perpendicular_implies_perpendicular_l284_284394

variables (l : Line) (α β : Plane)

axiom line_parallel_plane (l : Line) (π : Plane) : Prop
axiom line_perpendicular_plane (l : Line) (π : Plane) : Prop
axiom planes_are_perpendicular (π₁ π₂ : Plane) : Prop

theorem parallel_and_perpendicular_implies_perpendicular
  (h1 : line_parallel_plane l α)
  (h2 : line_perpendicular_plane l β) 
  : planes_are_perpendicular α β :=
sorry

end parallel_and_perpendicular_implies_perpendicular_l284_284394


namespace problem_solution_l284_284284

theorem problem_solution :
  (-2: ℤ)^2004 + 3 * (-2: ℤ)^2003 = -2^2003 := 
by
  sorry

end problem_solution_l284_284284


namespace smallest_odd_number_with_five_prime_factors_l284_284836

theorem smallest_odd_number_with_five_prime_factors :
  ∃ (n : ℕ), n = 3 * 5 * 7 * 11 * 13 ∧
  n % 2 ≠ 0 ∧
  ∃ (p1 p2 p3 p4 p5 : ℕ), 
    (prime p1) ∧ 
    (prime p2) ∧ 
    (prime p3) ∧ 
    (prime p4) ∧ 
    (prime p5) ∧ 
    p1 ≠ p2 ∧ 
    p2 ≠ p3 ∧ 
    p3 ≠ p4 ∧ 
    p4 ≠ p5 ∧ 
    p1 = 3 ∧ 
    p2 = 5 ∧ 
    p3 = 7 ∧ 
    p4 = 11 ∧ 
    p5 = 13 ∧ 
    n = p1 * p2 * p3 * p4 * p5 :=
sorry

end smallest_odd_number_with_five_prime_factors_l284_284836


namespace trig_expression_eval_l284_284476

open Real

-- Declare the main theorem
theorem trig_expression_eval (θ : ℝ) (k : ℤ) 
  (h : sin (θ + k * π) = -2 * cos (θ + k * π)) :
  (4 * sin θ - 2 * cos θ) / (5 * cos θ + 3 * sin θ) = 10 :=
  sorry

end trig_expression_eval_l284_284476


namespace round_310242_to_nearest_thousand_l284_284128

-- Define the conditions and the target statement
def round_to_nearest_thousand (n : ℕ) : ℕ :=
  if (n % 1000) < 500 then (n / 1000) * 1000 else (n / 1000 + 1) * 1000

theorem round_310242_to_nearest_thousand :
  round_to_nearest_thousand 310242 = 310000 :=
by
  sorry

end round_310242_to_nearest_thousand_l284_284128


namespace arithmetic_sequence_num_terms_l284_284355

theorem arithmetic_sequence_num_terms (a d l : ℕ) (h1 : a = 15) (h2 : d = 4) (h3 : l = 159) :
  ∃ n : ℕ, l = a + (n-1) * d ∧ n = 37 :=
by {
  sorry
}

end arithmetic_sequence_num_terms_l284_284355


namespace trigonometric_expression_l284_284366

theorem trigonometric_expression (θ : ℝ) (h : Real.tan θ = -3) :
    2 / (3 * (Real.sin θ) ^ 2 - (Real.cos θ) ^ 2) = 10 / 13 :=
by
  -- sorry to skip the proof
  sorry

end trigonometric_expression_l284_284366


namespace min_value_frac_ineq_l284_284368

theorem min_value_frac_ineq (m n : ℝ) (hm : m > 0) (hn : n > 0) (h_sum : m + n = 1) : 
  (9/m + 1/n) ≥ 16 :=
sorry

end min_value_frac_ineq_l284_284368


namespace smallest_odd_number_with_five_primes_proof_l284_284849

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m ∣ n, m = 1 ∨ m = n

def smallest_odd_primes : List ℕ := [3, 5, 7, 11, 13]

noncomputable def smallest_odd_number_with_five_primes : ℕ :=
  List.prod smallest_odd_primes

theorem smallest_odd_number_with_five_primes_proof : smallest_odd_number_with_five_primes = 15015 :=
by
  unfold smallest_odd_number_with_five_primes
  unfold smallest_odd_primes
  norm_num

end smallest_odd_number_with_five_primes_proof_l284_284849


namespace tan_105_degree_l284_284587

theorem tan_105_degree : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  have tan_add : ∀ (a b : ℝ), Real.tan (a + b) = (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b) :=
    sorry
  have tan_45 := Real.tan (45 * Real.pi / 180)
  have tan_60 := Real.tan (60 * Real.pi / 180)
  have tan_45_value : tan_45 = 1 := sorry
  have tan_60_value : tan_60 = Real.sqrt 3 := sorry
  sorry

end tan_105_degree_l284_284587


namespace joan_gave_28_seashells_to_sam_l284_284087

/-- 
Given:
- Joan found 70 seashells on the beach.
- After giving away some seashells, she has 27 left.
- She gave twice as many seashells to Sam as she gave to her friend Lily.

Show that:
- Joan gave 28 seashells to Sam.
-/
theorem joan_gave_28_seashells_to_sam (L S : ℕ) 
  (h1 : S = 2 * L) 
  (h2 : 70 - 27 = 43) 
  (h3 : L + S = 43) :
  S = 28 :=
by
  sorry

end joan_gave_28_seashells_to_sam_l284_284087


namespace A_can_complete_work_in_4_days_l284_284314

-- Definitions based on conditions
def work_done_in_one_day (days : ℕ) : ℚ := 1 / days

def combined_work_done_in_two_days (a b c : ℕ) : ℚ :=
  work_done_in_one_day a + work_done_in_one_day b + work_done_in_one_day c

-- Theorem statement based on the problem
theorem A_can_complete_work_in_4_days (A B C : ℕ) 
  (hB : B = 8) (hC : C = 8) 
  (h_combined : combined_work_done_in_two_days A B C = work_done_in_one_day 2) :
  A = 4 :=
sorry

end A_can_complete_work_in_4_days_l284_284314


namespace mr_brown_net_result_l284_284527

noncomputable def C1 := 1.50 / 1.3
noncomputable def C2 := 1.50 / 0.9
noncomputable def profit_from_first_pen := 1.50 - C1
noncomputable def tax := 0.05 * profit_from_first_pen
noncomputable def total_cost := C1 + C2
noncomputable def total_revenue := 3.00
noncomputable def net_result := total_revenue - total_cost - tax

theorem mr_brown_net_result : net_result = 0.16 :=
by
  sorry

end mr_brown_net_result_l284_284527


namespace order_numbers_l284_284104

theorem order_numbers (a b c : ℕ) (h1 : a = 8^10) (h2 : b = 4^15) (h3 : c = 2^31) : b = a ∧ a < c :=
by {
  sorry
}

end order_numbers_l284_284104


namespace tan_105_eq_neg2_sub_sqrt3_l284_284689

-- Define the main theorem to be proven
theorem tan_105_eq_neg2_sub_sqrt3 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_eq_neg2_sub_sqrt3_l284_284689


namespace goods_train_length_l284_284441

theorem goods_train_length (speed_kmph : ℕ) (platform_length_m : ℕ) (time_s : ℕ) 
    (h_speed : speed_kmph = 72) (h_platform : platform_length_m = 250) (h_time : time_s = 24) : 
    ∃ train_length_m : ℕ, train_length_m = 230 := 
by 
  sorry

end goods_train_length_l284_284441


namespace arithmetic_sequence_2023rd_term_l284_284978

theorem arithmetic_sequence_2023rd_term 
  (p q : ℤ)
  (h1 : 3 * p - q + 9 = 9)
  (h2 : 3 * (3 * p - q + 9) - q + 9 = 3 * p + q) :
  p + (2023 - 1) * (3 * p - q + 9) = 18189 := by
  sorry

end arithmetic_sequence_2023rd_term_l284_284978


namespace emily_quiz_probability_l284_284498

noncomputable def prob_at_least_two_correct : ℚ := 763 / 3888

theorem emily_quiz_probability :
  (let total_prob := 1 - ((5/6)^5 + 5 * (1/6) * (5/6)^4)
  in total_prob = prob_at_least_two_correct) :=
by sorry

end emily_quiz_probability_l284_284498


namespace find_n_l284_284171

def C (k : ℕ) : ℕ :=
  if k = 1 then 0
  else (Nat.factors k).eraseDup.foldr (· + ·) 0

theorem find_n (n : ℕ) : 
  (∀ n, (C (2 ^ n + 1) = C n) ↔ n = 3) := 
by
  sorry

end find_n_l284_284171


namespace sum_of_interior_angles_of_decagon_l284_284281

def sum_of_interior_angles_of_polygon (n : ℕ) : ℕ := (n - 2) * 180

theorem sum_of_interior_angles_of_decagon : sum_of_interior_angles_of_polygon 10 = 1440 :=
by
  -- Proof goes here
  sorry

end sum_of_interior_angles_of_decagon_l284_284281


namespace interval_where_decreasing_l284_284980

open Real

noncomputable def piecewise_function (x : ℝ) : ℝ :=
if x ≥ 0 then (x - 3) * x else -((x - 3) * x)

theorem interval_where_decreasing :
  ∃ a b : ℝ, (∀ x : ℝ, a ≤ x ∧ x ≤ b → (piecewise_function x)' < 0) ∧ a = 0 ∧ b = 3 / 2 :=
sorry

end interval_where_decreasing_l284_284980


namespace flour_needed_for_one_loaf_l284_284798

-- Define the conditions
def flour_needed_for_two_loaves : ℚ := 5 -- cups of flour needed for two loaves

-- Define the theorem to prove
theorem flour_needed_for_one_loaf : flour_needed_for_two_loaves / 2 = 2.5 :=
by 
  -- Skip the proof.
  sorry

end flour_needed_for_one_loaf_l284_284798


namespace max_soap_boxes_in_carton_l284_284561

def carton_volume (length width height : ℕ) : ℕ :=
  length * width * height

def soap_box_volume (length width height : ℕ) : ℕ :=
  length * width * height

def max_soap_boxes (carton_volume soap_box_volume : ℕ) : ℕ :=
  carton_volume / soap_box_volume

theorem max_soap_boxes_in_carton :
  max_soap_boxes (carton_volume 25 42 60) (soap_box_volume 7 6 6) = 250 :=
by
  sorry

end max_soap_boxes_in_carton_l284_284561


namespace parrot_seeds_consumed_l284_284403

theorem parrot_seeds_consumed (H1 : ∃ T : ℝ, 0.40 * T = 8) : 
  (∃ T : ℝ, 0.40 * T = 8 ∧ 2 * T = 40) :=
sorry

end parrot_seeds_consumed_l284_284403


namespace tan_105_eq_neg2_sub_sqrt3_l284_284629

theorem tan_105_eq_neg2_sub_sqrt3 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by 
  sorry

end tan_105_eq_neg2_sub_sqrt3_l284_284629


namespace completing_the_square_l284_284860

theorem completing_the_square (x : ℝ) : x^2 - 8 * x + 1 = 0 → (x - 4)^2 = 15 :=
by
  sorry

end completing_the_square_l284_284860


namespace average_angle_red_blue_vectors_l284_284188

theorem average_angle_red_blue_vectors 
  (N : ℕ) (hN : N > 1) -- N > 1 ensures there are at least two vectors
  (red blue : set ℕ) -- subsets red and blue representing indices of vectors
  (h : ∀ (i : ℕ), i ∈ red ∪ blue) -- all indices are either red or blue
  (disjoint_red_blue : disjoint red blue) -- no vector is both red and blue
  : ∑ (i in red) (j in blue), (360 / N) / (card red * card blue) = 180 :=
sorry

end average_angle_red_blue_vectors_l284_284188


namespace necessary_condition_l284_284879

theorem necessary_condition (x : ℝ) : x = 1 → x^2 = 1 :=
by
  sorry

end necessary_condition_l284_284879


namespace AllieMoreGrapes_l284_284878

-- Definitions based on conditions
def RobBowl : ℕ := 25
def TotalGrapes : ℕ := 83
def AllynBowl (A : ℕ) : ℕ := A + 4

-- The proof statement that must be shown.
theorem AllieMoreGrapes (A : ℕ) (h1 : A + (AllynBowl A) + RobBowl = TotalGrapes) : A - RobBowl = 2 :=
by {
  sorry
}

end AllieMoreGrapes_l284_284878


namespace hannahs_brothers_l284_284373

theorem hannahs_brothers (B : ℕ) (h1 : ∀ (b : ℕ), b = 8) (h2 : 48 = 2 * (8 * B)) : B = 3 :=
by
  sorry

end hannahs_brothers_l284_284373


namespace fourth_term_of_sequence_l284_284371

theorem fourth_term_of_sequence (x : ℤ) (h : x^2 - 2 * x - 3 < 0) (hx : x ∈ {n : ℤ | x^2 - 2 * x - 3 < 0}) :
  ∃ a_1 a_2 a_3 a_4 : ℤ, 
  (a_1 = x) ∧ (a_2 = x + 1) ∧ (a_3 = x + 2) ∧ (a_4 = x + 3) ∧ 
  (a_4 = 3 ∨ a_4 = -1) :=
by { sorry }

end fourth_term_of_sequence_l284_284371


namespace count_two_digit_primes_ending_in_3_l284_284741

def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100
def has_ones_digit_3 (n : ℕ) : Prop := n % 10 = 3
def is_prime (n : ℕ) : Prop := nat.prime n
def two_digit_primes_ending_in_3 (n : ℕ) : Prop :=
  is_two_digit n ∧ has_ones_digit_3 n ∧ is_prime n

theorem count_two_digit_primes_ending_in_3 :
  (nat.card { n : ℕ | two_digit_primes_ending_in_3 n } = 6) :=
sorry

end count_two_digit_primes_ending_in_3_l284_284741


namespace dig_site_date_l284_284330

theorem dig_site_date (S1 S2 S3 S4 : ℕ) (S2_bc : S2 = 852) 
  (h1 : S1 = S2 - 352) 
  (h2 : S3 = S1 + 3700) 
  (h3 : S4 = 2 * S3) : 
  S4 = 6400 :=
by sorry

end dig_site_date_l284_284330


namespace tan_105_eq_neg2_sub_sqrt3_l284_284665

theorem tan_105_eq_neg2_sub_sqrt3 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 := by
  sorry

end tan_105_eq_neg2_sub_sqrt3_l284_284665


namespace Jason_more_blue_marbles_l284_284951

theorem Jason_more_blue_marbles (Jason_blue_marbles Tom_blue_marbles : ℕ) 
  (hJ : Jason_blue_marbles = 44) (hT : Tom_blue_marbles = 24) :
  Jason_blue_marbles - Tom_blue_marbles = 20 :=
by
  sorry

end Jason_more_blue_marbles_l284_284951


namespace reggie_games_lost_l284_284408

-- Define the necessary conditions
def initial_marbles : ℕ := 100
def bet_per_game : ℕ := 10
def marbles_after_games : ℕ := 90
def total_games : ℕ := 9

-- Define the proof problem statement
theorem reggie_games_lost : (initial_marbles - marbles_after_games) / bet_per_game = 1 := by
  sorry

end reggie_games_lost_l284_284408


namespace representable_by_expression_l284_284920

theorem representable_by_expression (n : ℕ) :
  ∃ (x y z : ℕ), x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ (n = (x * y + y * z + z * x) / (x + y + z)) ↔ n ≠ 1 := by
  sorry

end representable_by_expression_l284_284920


namespace determine_hyperbola_eq_l284_284478

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

end determine_hyperbola_eq_l284_284478


namespace num_true_propositions_l284_284015

theorem num_true_propositions (x : ℝ) :
  (∀ x, x > -3 → x > -6) ∧
  (∀ x, x > -6 → x > -3 = false) ∧
  (∀ x, x ≤ -3 → x ≤ -6 = false) ∧
  (∀ x, x ≤ -6 → x ≤ -3) →
  2 = 2 :=
by
  sorry

end num_true_propositions_l284_284015


namespace no_distinct_positive_integers_l284_284789

noncomputable def P (x : ℕ) : ℕ := x^2000 - x^1000 + 1

theorem no_distinct_positive_integers (a : Fin 2001 → ℕ) (h_distinct : Function.Injective a) :
  ¬ (∀ i j, i ≠ j → a i * a j ∣ P (a i) * P (a j)) :=
sorry

end no_distinct_positive_integers_l284_284789


namespace two_digit_primes_ending_in_3_l284_284753

theorem two_digit_primes_ending_in_3 : ∃ n : ℕ, n = 7 ∧ (∀ x, 10 ≤ x ∧ x < 100 ∧ x % 10 = 3 → Prime x → (x = 13 ∨ x = 23 ∨ x = 43 ∨ x = 53 ∨ x = 73 ∨ x = 83)) :=
by {
  use 7,
  split,
  -- proof part is omitted for this example
  sorry,
}

end two_digit_primes_ending_in_3_l284_284753


namespace male_female_ratio_l284_284317

-- Definitions and constants
variable (M F : ℕ) -- Number of male and female members respectively
variable (h_avg_members : 66 * (M + F) = 58 * M + 70 * F) -- Average ticket sales condition

-- Statement of the theorem
theorem male_female_ratio (M F : ℕ) (h_avg_members : 66 * (M + F) = 58 * M + 70 * F) : M / F = 1 / 2 :=
sorry

end male_female_ratio_l284_284317


namespace exclusive_movies_count_l284_284156

-- Define the conditions
def shared_movies : Nat := 15
def andrew_movies : Nat := 25
def john_movies_exclusive : Nat := 8

-- Define the result calculation
def exclusive_movies (andrew_movies shared_movies john_movies_exclusive : Nat) : Nat :=
  (andrew_movies - shared_movies) + john_movies_exclusive

-- Statement to prove
theorem exclusive_movies_count : exclusive_movies andrew_movies shared_movies john_movies_exclusive = 18 := by
  sorry

end exclusive_movies_count_l284_284156


namespace problem1_problem2_l284_284370

-- Problem 1: Prove the solution set of the given inequality
theorem problem1 (x : ℝ) : (|x - 2| + 2 * |x - 1| > 5) ↔ (x < -1/3 ∨ x > 3) := 
sorry

-- Problem 2: Prove the range of values for 'a' such that the inequality holds
theorem problem2 (a : ℝ) : (∃ x : ℝ, |x - a| + |x - 1| ≤ |a - 2|) ↔ (a ≤ 3/2) :=
sorry

end problem1_problem2_l284_284370


namespace julio_lost_15_fish_l284_284090

def fish_caught_per_hour : ℕ := 7
def hours_fished : ℕ := 9
def fish_total_without_loss : ℕ := fish_caught_per_hour * hours_fished
def fish_total_actual : ℕ := 48
def fish_lost : ℕ := fish_total_without_loss - fish_total_actual

theorem julio_lost_15_fish : fish_lost = 15 := by
  sorry

end julio_lost_15_fish_l284_284090


namespace correct_statements_count_l284_284451

-- Definitions for each condition
def is_output_correct (stmt : String) : Prop :=
  stmt = "PRINT a, b, c"

def is_input_correct (stmt : String) : Prop :=
  stmt = "INPUT \"x=3\""

def is_assignment_correct_1 (stmt : String) : Prop :=
  stmt = "A=3"

def is_assignment_correct_2 (stmt : String) : Prop :=
  stmt = "A=B ∧ B=C"

-- The main theorem to be proven
theorem correct_statements_count (stmt1 stmt2 stmt3 stmt4 : String) :
  stmt1 = "INPUT a, b, c" → stmt2 = "INPUT x=3" → stmt3 = "3=A" → stmt4 = "A=B=C" →
  (¬ is_output_correct stmt1 ∧ ¬ is_input_correct stmt2 ∧ ¬ is_assignment_correct_1 stmt3 ∧ ¬ is_assignment_correct_2 stmt4) →
  0 = 0 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end correct_statements_count_l284_284451


namespace two_digit_integers_congruent_to_2_mod_4_l284_284065

theorem two_digit_integers_congruent_to_2_mod_4 :
  {n // 10 ≤ n ∧ n ≤ 99 ∧ n % 4 = 2}.card = 23 := 
sorry

end two_digit_integers_congruent_to_2_mod_4_l284_284065


namespace reggie_games_lost_l284_284407

-- Define the necessary conditions
def initial_marbles : ℕ := 100
def bet_per_game : ℕ := 10
def marbles_after_games : ℕ := 90
def total_games : ℕ := 9

-- Define the proof problem statement
theorem reggie_games_lost : (initial_marbles - marbles_after_games) / bet_per_game = 1 := by
  sorry

end reggie_games_lost_l284_284407


namespace simplify_expression_l284_284260

theorem simplify_expression (x : ℝ) (h : x = 1) : (x - 1)^2 + (x + 1) * (x - 1) - 2 * x^2 = -2 :=
by
  sorry

end simplify_expression_l284_284260


namespace quadratic_roots_condition_l284_284174

theorem quadratic_roots_condition (c : ℝ) : 
  (∀ x : ℝ, x^2 + 2 * x + 4 * c = 0 → (∆ := 2^2 - 4 * 1 * 4 * c, ∆ > 0)) ↔ c < 1/4 :=
by 
  sorry

end quadratic_roots_condition_l284_284174


namespace tan_105_eq_neg2_sub_sqrt3_l284_284630

theorem tan_105_eq_neg2_sub_sqrt3 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by 
  sorry

end tan_105_eq_neg2_sub_sqrt3_l284_284630


namespace parabola_equation_max_slope_OQ_l284_284487

theorem parabola_equation (p : ℝ) (hp : p = 2) :
    ∃ x y : ℝ, y^2 = 2 * p * x ↔ y^2 = 4 * x :=
by sorry

theorem max_slope_OQ (Q F : ℝ × ℝ) (hQF : ∀ P : ℝ × ℝ, P ∈ parabola_eq ↔ P.x = 10 * Q.x - 9 ∧ 
                                                         P.y = 10 * Q.y ∧ y^2 = 4 * P.x)
    (hPQ : (Q.x - P.x, Q.y - P.y) = 9 * (1 - Q.x, 0 - Q.y)) :
    ∃ n : ℝ, Q.y = n ∧ Q.x = (25 * n^2 + 9) / 10 ∧ 
        max (λ n, (10 * n) / (25 * n^2 + 9)) = 1 / 3 :=
by sorry

end parabola_equation_max_slope_OQ_l284_284487


namespace quadrilateral_pyramid_plane_intersection_l284_284480

-- Definitions:
-- Let MA, MB, MC, MD, MK, ML, MP, MN be lengths of respective segments
-- Let S_ABC, S_ABD, S_ACD, S_BCD be areas of respective triangles
variables {MA MB MC MD MK ML MP MN : ℝ}
variables {S_ABC S_ABD S_ACD S_BCD : ℝ}

-- Given a quadrilateral pyramid MABCD with a convex quadrilateral ABCD as base, and a plane intersecting edges MA, MB, MC, and MD at points K, L, P, and N respectively. Prove the following relation.
theorem quadrilateral_pyramid_plane_intersection :
  S_BCD * (MA / MK) + S_ADB * (MC / MP) = S_ABC * (MD / MN) + S_ACD * (MB / ML) :=
sorry

end quadrilateral_pyramid_plane_intersection_l284_284480


namespace original_people_in_room_l284_284785

theorem original_people_in_room (x : ℝ) (h1 : x / 3 * 2 / 2 = 18) : x = 54 :=
sorry

end original_people_in_room_l284_284785


namespace inequality_region_area_l284_284461

noncomputable def area_of_inequality_region : ℝ :=
  let region := {p : ℝ × ℝ | |p.fst - p.snd| + |2 * p.fst + 2 * p.snd| ≤ 8}
  let vertices := [(2, 2), (-2, 2), (-2, -2), (2, -2)]
  let d1 := 8
  let d2 := 8
  (1 / 2) * d1 * d2

theorem inequality_region_area :
  area_of_inequality_region = 32 :=
by
  sorry  -- Proof to be provided

end inequality_region_area_l284_284461


namespace fraction_simplified_to_p_l284_284821

theorem fraction_simplified_to_p (q : ℕ) (hq_pos : 0 < q) (gcd_cond : Nat.gcd 4047 q = 1) :
    (2024 / 2023) - (2023 / 2024) = 4047 / q := sorry

end fraction_simplified_to_p_l284_284821


namespace constant_t_exists_l284_284285

theorem constant_t_exists (c : ℝ) :
  ∃ t : ℝ, (∀ A B : ℝ × ℝ, (A.1^2 + A.2^2 = 1) ∧ (B.1^2 + B.2^2 = 1) ∧ (A.2 = A.1 * c + c) ∧ (B.2 = B.1 * c + c) → (t = -2)) :=
sorry

end constant_t_exists_l284_284285


namespace kristi_books_proof_l284_284454

variable (Bobby_books Kristi_books : ℕ)

def condition1 : Prop := Bobby_books = 142

def condition2 : Prop := Bobby_books = Kristi_books + 64

theorem kristi_books_proof (h1 : condition1 Bobby_books) (h2 : condition2 Bobby_books Kristi_books) : Kristi_books = 78 := 
by 
  sorry

end kristi_books_proof_l284_284454


namespace range_of_a_l284_284934

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, x^2 - a*x + 2*a > 0) : 0 < a ∧ a < 8 :=
by
  sorry

end range_of_a_l284_284934


namespace unique_primes_solution_l284_284028

theorem unique_primes_solution (p q r : ℕ) (hp : Prime p) (hq : Prime q) (hr : Prime r) : 
    p + q^2 = r^4 ↔ (p = 7 ∧ q = 3 ∧ r = 2) := 
by
  sorry

end unique_primes_solution_l284_284028


namespace tan_105_eq_neg_2_sub_sqrt_3_l284_284623

-- Definitions
def angle105 : ℝ := 105 * (Math.pi / 180)
def angle45 : ℝ := 45 * (Math.pi / 180)
def angle60 : ℝ := 60 * (Math.pi / 180)

-- Theorem
theorem tan_105_eq_neg_2_sub_sqrt_3 :
  Real.tan angle105 = -2 - Real.sqrt 3 := by
  sorry

end tan_105_eq_neg_2_sub_sqrt_3_l284_284623


namespace tan_105_eq_neg2_sub_sqrt3_l284_284644

theorem tan_105_eq_neg2_sub_sqrt3 :
  Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_eq_neg2_sub_sqrt3_l284_284644


namespace yearly_savings_l284_284010

-- Define the various constants given in the problem
def weeks_in_year : ℕ := 52
def months_in_year : ℕ := 12
def non_peak_weeks : ℕ := 16
def peak_weeks : ℕ := weeks_in_year - non_peak_weeks
def non_peak_months : ℕ := 4
def peak_months : ℕ := months_in_year - non_peak_months

-- Rates
def weekly_cost_non_peak_large : ℕ := 10
def weekly_cost_peak_large : ℕ := 12
def monthly_cost_non_peak_large : ℕ := 42
def monthly_cost_peak_large : ℕ := 48

-- Additional surcharge
def holiday_weeks : ℕ := 6
def holiday_surcharge : ℕ := 2

-- Compute the yearly costs
def yearly_weekly_cost : ℕ :=
  (non_peak_weeks * weekly_cost_non_peak_large) +
  (peak_weeks * weekly_cost_peak_large) +
  (holiday_weeks * (holiday_surcharge + weekly_cost_peak_large))

def yearly_monthly_cost : ℕ :=
  (non_peak_months * monthly_cost_non_peak_large) +
  (peak_months * monthly_cost_peak_large)

theorem yearly_savings : yearly_weekly_cost - yearly_monthly_cost = 124 := by
  sorry

end yearly_savings_l284_284010


namespace tan_105_degree_is_neg_sqrt3_minus_2_l284_284656

theorem tan_105_degree_is_neg_sqrt3_minus_2 :
  Real.tan (105 * Real.pi / 180) = -(Real.sqrt 3 + 2) := by
  sorry

end tan_105_degree_is_neg_sqrt3_minus_2_l284_284656


namespace mn_plus_one_unequal_pos_integers_l284_284052

theorem mn_plus_one_unequal_pos_integers (m n : ℕ) 
  (S : Finset ℕ) (h_card : S.card = m * n + 1) :
  (∃ (b : Fin (m + 1) → ℕ), (∀ i j : Fin (m + 1), i ≠ j → ¬(b i ∣ b j)) ∧ (∀ i : Fin (m + 1), b i ∈ S)) ∨ 
  (∃ (a : Fin (n + 1) → ℕ), (∀ i : Fin n, a i ∣ a (i + 1)) ∧ (∀ i : Fin (n + 1), a i ∈ S)) :=
sorry

end mn_plus_one_unequal_pos_integers_l284_284052


namespace min_shift_for_even_function_l284_284329

theorem min_shift_for_even_function :
  ∃ (m : ℝ), (m > 0) ∧ (∀ x : ℝ, (Real.sin (x + m) + Real.cos (x + m)) = (Real.sin (-x + m) + Real.cos (-x + m))) ∧ m = π / 4 :=
by
  sorry

end min_shift_for_even_function_l284_284329


namespace sum_of_integers_is_106_l284_284277

theorem sum_of_integers_is_106 (n m : ℕ) 
  (h1: n * (n + 1) = 1320) 
  (h2: m * (m + 1) * (m + 2) = 1320) : 
  n + (n + 1) + m + (m + 1) + (m + 2) = 106 :=
  sorry

end sum_of_integers_is_106_l284_284277


namespace area_of_triangle_l284_284446

theorem area_of_triangle : 
  let l : ℝ → ℝ → Prop := fun x y => 3 * x + 2 * y = 12 in
  let x_intercept := (4 : ℝ) in
  let y_intercept := (6 : ℝ) in
  ∃ x y : ℝ, l x 0 ∧ x = x_intercept ∧ l 0 y ∧ y = y_intercept ∧ (1 / 2) * x_intercept * y_intercept = 12 := 
by
  sorry

end area_of_triangle_l284_284446


namespace certain_number_condition_l284_284380

theorem certain_number_condition (x y z : ℤ) (N : ℤ)
  (hx : Even x) (hy : Odd y) (hz : Odd z)
  (hxy : x < y) (hyz : y < z)
  (h1 : y - x > N)
  (h2 : z - x = 7) :
  N < 3 := by
  sorry

end certain_number_condition_l284_284380


namespace quadratic_roots_condition_l284_284173

theorem quadratic_roots_condition (c : ℝ) : 
  (∀ x : ℝ, x^2 + 2 * x + 4 * c = 0 → (∆ := 2^2 - 4 * 1 * 4 * c, ∆ > 0)) ↔ c < 1/4 :=
by 
  sorry

end quadratic_roots_condition_l284_284173


namespace rhombus_area_l284_284359

-- Definitions
def side_length := 25 -- cm
def diagonal1 := 30 -- cm

-- Statement to prove
theorem rhombus_area (s : ℝ) (d1 : ℝ) (d2 : ℝ) 
  (h_s : s = 25) 
  (h_d1 : d1 = 30)
  (h_side : s^2 = (d1/2)^2 + (d2/2)^2) :
  (d1 * d2) / 2 = 600 :=
by sorry

end rhombus_area_l284_284359


namespace part1_part2_l284_284352

noncomputable def determinant (a b c d : ℤ) : ℤ :=
  a * d - b * c

-- Lean statement for Question (1)
theorem part1 :
  determinant 2022 2023 2021 2022 = 1 :=
by sorry

-- Lean statement for Question (2)
theorem part2 (m : ℤ) :
  determinant (m + 2) (m - 2) (m - 2) (m + 2) = 32 → m = 4 :=
by sorry

end part1_part2_l284_284352


namespace ratio_length_breadth_l284_284981

-- Define the conditions
def length := 135
def area := 6075

-- Define the breadth in terms of the area and length
def breadth := area / length

-- The problem statement as a Lean 4 theorem to prove the ratio
theorem ratio_length_breadth : length / breadth = 3 := 
by
  -- Proof goes here
  sorry

end ratio_length_breadth_l284_284981


namespace min_value_reciprocal_sum_l284_284793

theorem min_value_reciprocal_sum (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : Real.sqrt 3 = Real.sqrt (3^a * 3^b)) : 
  (1 / a + 1 / b) ≥ 4 :=
sorry

end min_value_reciprocal_sum_l284_284793


namespace total_phd_time_l284_284088

-- Definitions for the conditions
def acclimation_period : ℕ := 1
def basics_period : ℕ := 2
def research_period := basics_period + (3 * basics_period / 4)
def dissertation_period := acclimation_period / 2

-- Main statement to prove
theorem total_phd_time : acclimation_period + basics_period + research_period + dissertation_period = 7 := by
  -- Here should be the proof (skipped with sorry)
  sorry

end total_phd_time_l284_284088


namespace smallest_odd_with_five_prime_factors_l284_284835

theorem smallest_odd_with_five_prime_factors :
  ∃ n : ℕ, n = 3 * 5 * 7 * 11 * 13 ∧ ∀ m : ℕ, (m < n → (∃ p1 p2 p3 p4 p5 : ℕ,
  prime p1 ∧ odd p1 ∧ prime p2 ∧ odd p2 ∧ prime p3 ∧ odd p3 ∧
  prime p4 ∧ odd p4 ∧ prime p5 ∧ odd p5 ∧
  m = p1 * p2 * p3 * p4 * p5)) → m < 3 * 5 * 7 * 11 * 13 := 
by {
  use 3 * 5 * 7 * 11 * 13,
  split,
  norm_num,
  intros m hlt hexists,
  obtain ⟨p1, p2, p3, p4, p5, hp1, hodd1, hp2, hodd2, hp3, hodd3, hp4, hodd4, hp5, hodd5, hprod⟩ := hexists,
  sorry
}

end smallest_odd_with_five_prime_factors_l284_284835


namespace honda_day_shift_production_l284_284735

theorem honda_day_shift_production (S : ℕ) (day_shift_production : ℕ)
  (h1 : day_shift_production = 4 * S)
  (h2 : day_shift_production + S = 5500) :
  day_shift_production = 4400 :=
sorry

end honda_day_shift_production_l284_284735


namespace k4_min_bound_l284_284778

theorem k4_min_bound (n m X : ℕ) (hn : n ≥ 5) (hm : m ≤ n.choose 3) 
    (no_three_collinear : ∀ (P : Finset (Fin n)), P.card = 3 → ∀ (p : Fin n → Fin n → Fin n → Prop), 
                          ∃ u v w, u ≠ v ∧ v ≠ w ∧ u ≠ w ∧ P = {u,v,w} ∧ ¬ p u v w) :
    X ≥ m / 4 * (9 * m / n - 3 / 2 * n^2 + 11 / 2 * n - 6) :=
sorry

end k4_min_bound_l284_284778


namespace total_bones_in_graveyard_l284_284775

def total_skeletons : ℕ := 20

def adult_women : ℕ := total_skeletons / 2
def adult_men : ℕ := (total_skeletons - adult_women) / 2
def children : ℕ := (total_skeletons - adult_women) / 2

def bones_adult_woman : ℕ := 20
def bones_adult_man : ℕ := bones_adult_woman + 5
def bones_child : ℕ := bones_adult_woman / 2

def bones_graveyard : ℕ :=
  (adult_women * bones_adult_woman) +
  (adult_men * bones_adult_man) +
  (children * bones_child)

theorem total_bones_in_graveyard :
  bones_graveyard = 375 :=
sorry

end total_bones_in_graveyard_l284_284775


namespace regular_polygon_properties_l284_284726

theorem regular_polygon_properties
  (n : ℕ)
  (h1 : (n - 2) * 180 = 3 * 360 + 180)
  (h2 : n > 2) :
  n = 9 ∧ (n - 2) * 180 / n = 140 := by
  sorry

end regular_polygon_properties_l284_284726


namespace divide_segment_l284_284901

theorem divide_segment (a : ℝ) (n : ℕ) (h : 0 < n) : 
  ∃ P : ℝ, P = a / (n + 1) ∧ P > 0 :=
by
  sorry

end divide_segment_l284_284901


namespace arithmetic_sequence_common_difference_l284_284363

theorem arithmetic_sequence_common_difference
  (S : ℕ → ℤ)
  (a : ℕ → ℤ)
  (a₁ d : ℤ)
  (h1 : ∀ n, S n = n * (2 * a₁ + (n - 1) * d) / 2)
  (h2 : ∀ n, a n = a₁ + (n - 1) * d)
  (h3 : S 5 = 5 * (a 4) - 10) :
  d = 2 := sorry

end arithmetic_sequence_common_difference_l284_284363


namespace merchant_articles_l284_284872

theorem merchant_articles (N CP SP : ℝ) (h1 : N * CP = 16 * SP) (h2 : SP = CP * 1.0625) (h3 : CP ≠ 0) : N = 17 :=
by
  sorry

end merchant_articles_l284_284872


namespace previous_job_salary_is_correct_l284_284830

-- Define the base salary and commission structure.
def base_salary_new_job : ℝ := 45000
def commission_rate : ℝ := 0.15
def sale_amount : ℝ := 750
def minimum_sales : ℝ := 266.67

-- Define the total salary from the new job with the minimum sales.
def new_job_total_salary : ℝ :=
  base_salary_new_job + (commission_rate * sale_amount * minimum_sales)

-- Define Tom's previous job's salary.
def previous_job_salary : ℝ := 75000

-- Prove that Tom's previous job salary matches the new job total salary with the minimum sales.
theorem previous_job_salary_is_correct :
  (new_job_total_salary = previous_job_salary) :=
by
  -- This is where you would include the proof steps, but it's sufficient to put 'sorry' for now.
  sorry

end previous_job_salary_is_correct_l284_284830


namespace total_lives_l284_284565

-- Defining the number of lives for each animal according to the given conditions:
def cat_lives : ℕ := 9
def dog_lives : ℕ := cat_lives - 3
def mouse_lives : ℕ := dog_lives + 7
def elephant_lives : ℕ := 2 * cat_lives - 5
def fish_lives : ℕ := if (dog_lives + mouse_lives) < (elephant_lives / 2) then (dog_lives + mouse_lives) else elephant_lives / 2

-- The main statement we need to prove:
theorem total_lives :
  cat_lives + dog_lives + mouse_lives + elephant_lives + fish_lives = 47 :=
by
  sorry

end total_lives_l284_284565


namespace difference_in_tiles_l284_284152

-- Definition of side length of the nth square
def side_length (n : ℕ) : ℕ := n

-- Theorem stating the difference in tiles between the 10th and 9th squares
theorem difference_in_tiles : (side_length 10) ^ 2 - (side_length 9) ^ 2 = 19 := 
by {
  sorry
}

end difference_in_tiles_l284_284152


namespace radius_of_smaller_circle_l284_284500

theorem radius_of_smaller_circle (R : ℝ) (n : ℕ) (r : ℝ) 
  (hR : R = 10) 
  (hn : n = 7) 
  (condition : 2 * R = 2 * r * n) :
  r = 10 / 7 :=
by
  sorry

end radius_of_smaller_circle_l284_284500


namespace cos_alpha_plus_pi_over_2_l284_284049

theorem cos_alpha_plus_pi_over_2 (α : ℝ) (h : Real.sin α = 1/3) : 
    Real.cos (α + Real.pi / 2) = -(1/3) :=
by
  sorry

end cos_alpha_plus_pi_over_2_l284_284049


namespace circle_radius_l284_284426

theorem circle_radius (x y : ℝ) (h : x^2 + y^2 - 4*x + 6*y = 0) : ∃ r : ℝ, r = Real.sqrt 13 :=
by
  sorry

end circle_radius_l284_284426


namespace range_of_p_l284_284464

def p (x : ℝ) : ℝ := x^6 + 6 * x^3 + 9

theorem range_of_p : Set.Ici 9 = { y | ∃ x ≥ 0, p x = y } :=
by
  -- We skip the proof to only provide the statement as requested.
  sorry

end range_of_p_l284_284464


namespace catherine_bottle_caps_l284_284162

-- Definitions from conditions
def friends : ℕ := 6
def caps_per_friend : ℕ := 3

-- Theorem statement from question and correct answer
theorem catherine_bottle_caps : friends * caps_per_friend = 18 :=
by sorry

end catherine_bottle_caps_l284_284162


namespace hundredth_odd_integer_not_divisible_by_five_l284_284833

def odd_positive_integer (n : ℕ) : ℕ := 2 * n - 1

theorem hundredth_odd_integer_not_divisible_by_five :
  odd_positive_integer 100 = 199 ∧ ¬ (199 % 5 = 0) :=
by
  sorry

end hundredth_odd_integer_not_divisible_by_five_l284_284833


namespace pam_bags_l284_284528

-- Definitions
def gerald_bag_apples : ℕ := 40
def pam_bag_apples : ℕ := 3 * gerald_bag_apples
def pam_total_apples : ℕ := 1200

-- Theorem stating that the number of Pam's bags is 10
theorem pam_bags : pam_total_apples / pam_bag_apples = 10 := by
  sorry

end pam_bags_l284_284528


namespace line_through_point_parallel_l284_284708

theorem line_through_point_parallel (x y : ℝ) (h₁ : 2 * 2 + 4 * 3 + x = 0) (h₂ : x = -16) (h₃ : y = 8) :
  2 * x + 4 * y - 3 = 0 → x + 2 * y - 8 = 0 :=
by
  intro h₄
  sorry

end line_through_point_parallel_l284_284708


namespace sum_arithmetic_sequence_n_ge_52_l284_284921

theorem sum_arithmetic_sequence_n_ge_52 (n : ℕ) : 
  (∃ k, k = n) → 22 - 3 * (n - 1) = 22 - 3 * (n - 1) ∧ n ∈ { k | 3 ≤ k ∧ k ≤ 13 } :=
by
  sorry

end sum_arithmetic_sequence_n_ge_52_l284_284921


namespace scientific_notation_of_845_billion_l284_284233

/-- Express 845 billion yuan in scientific notation. -/
theorem scientific_notation_of_845_billion :
  (845 * (10^9 : ℝ)) / (10^9 : ℝ) = 8.45 * 10^3 :=
by
  sorry

end scientific_notation_of_845_billion_l284_284233


namespace toys_of_Jason_l284_284513

theorem toys_of_Jason (R J Jason : ℕ) 
  (hR : R = 1) 
  (hJ : J = R + 6) 
  (hJason : Jason = 3 * J) : 
  Jason = 21 :=
by
  sorry

end toys_of_Jason_l284_284513


namespace amusement_park_l284_284569

theorem amusement_park
  (A : ℕ)
  (adult_ticket_cost : ℕ := 22)
  (child_ticket_cost : ℕ := 7)
  (num_children : ℕ := 2)
  (total_cost : ℕ := 58)
  (cost_eq : adult_ticket_cost * A + child_ticket_cost * num_children = total_cost) :
  A = 2 :=
by {
  sorry
}

end amusement_park_l284_284569


namespace tan_105_eq_neg2_sub_sqrt3_l284_284628

theorem tan_105_eq_neg2_sub_sqrt3 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by 
  sorry

end tan_105_eq_neg2_sub_sqrt3_l284_284628


namespace conversion_problems_l284_284310

def decimal_to_binary (n : ℕ) : ℕ :=
  if n = 0 then 0 else n % 2 + 10 * decimal_to_binary (n / 2)

def largest_two_digit_octal : ℕ := 77

theorem conversion_problems :
  decimal_to_binary 111 = 1101111 ∧ (7 * 8 + 7) = 63 :=
by
  sorry

end conversion_problems_l284_284310


namespace problem_statement_l284_284047

noncomputable def polynomial_expansion (x : ℚ) : ℚ := (1 - 2 * x) ^ 8

theorem problem_statement :
  (8 * (1 - 2 * 1) ^ 7 * (-2)) = (a_1 : ℚ) + 2 * (a_2 : ℚ) + 3 * (a_3 : ℚ) + 4 * (a_4 : ℚ) +
  5 * (a_5 : ℚ) + 6 * (a_6 : ℚ) + 7 * (a_7 : ℚ) + 8 * (a_8 : ℚ) := by 
  sorry

end problem_statement_l284_284047


namespace find_n_after_folding_l284_284711

theorem find_n_after_folding (n : ℕ) (h : 2 ^ n = 128) : n = 7 := by
  sorry

end find_n_after_folding_l284_284711


namespace count_two_digit_primes_with_ones_digit_3_l284_284765

theorem count_two_digit_primes_with_ones_digit_3 :
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ n % 10 = 3 ∧ Nat.Prime n}.to_finset.card = 6 :=
by
  sorry

end count_two_digit_primes_with_ones_digit_3_l284_284765


namespace maximum_value_of_2x_plus_y_l284_284095

noncomputable def max_value_2x_plus_y (x y : ℝ) (h : 4 * x^2 + y^2 + x * y = 1) : ℝ :=
  (2 * x + y)

theorem maximum_value_of_2x_plus_y (x y : ℝ) (h : 4 * x^2 + y^2 + x * y = 1) :
  max_value_2x_plus_y x y h ≤ (2 * Real.sqrt 10) / 5 :=
sorry

end maximum_value_of_2x_plus_y_l284_284095


namespace tangent_line_ellipse_l284_284434

variable {a b x x0 y y0 : ℝ}

theorem tangent_line_ellipse (h : a * x0^2 + b * y0^2 = 1) :
  a * x0 * x + b * y0 * y = 1 :=
sorry

end tangent_line_ellipse_l284_284434


namespace tan_105_degree_is_neg_sqrt3_minus_2_l284_284654

theorem tan_105_degree_is_neg_sqrt3_minus_2 :
  Real.tan (105 * Real.pi / 180) = -(Real.sqrt 3 + 2) := by
  sorry

end tan_105_degree_is_neg_sqrt3_minus_2_l284_284654


namespace tan_add_tan_105_eq_l284_284657

noncomputable def tan : ℝ → ℝ := sorry -- Use the built-in library later for actual implementation

-- Given conditions
def tan_45_eq : tan 45 = 1 := by sorry
def tan_60_eq : tan 60 = Real.sqrt 3 := by sorry

-- Angle addition formula for tangent
theorem tan_add (a b : ℝ) :
  tan (a + b) = (tan a + tan b) / (1 - tan a * tan b) := by sorry

-- Main theorem to prove
theorem tan_105_eq :
  tan 105 = -2 - Real.sqrt 3 := by sorry

end tan_add_tan_105_eq_l284_284657


namespace quadratic_two_distinct_real_roots_l284_284186

theorem quadratic_two_distinct_real_roots (c : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2 * x₁ + 4 * c = 0 ∧ x₂^2 + 2 * x₂ + 4 * c = 0) ↔ c < 1 / 4 :=
sorry

end quadratic_two_distinct_real_roots_l284_284186


namespace abs_neg_four_squared_plus_six_l284_284810

theorem abs_neg_four_squared_plus_six : |(-4^2 + 6)| = 10 := by
  -- We skip the proof steps according to the instruction
  sorry

end abs_neg_four_squared_plus_six_l284_284810


namespace third_year_students_sampled_correct_l284_284108

-- The given conditions
def first_year_students := 700
def second_year_students := 670
def third_year_students := 630
def total_samples := 200
def total_students := first_year_students + second_year_students + third_year_students

-- The proportion of third-year students
def third_year_proportion := third_year_students / total_students

-- The number of third-year students to be selected
def samples_third_year := total_samples * third_year_proportion

theorem third_year_students_sampled_correct :
  samples_third_year = 63 :=
by
  -- We skip the actual proof for this statement with sorry
  sorry

end third_year_students_sampled_correct_l284_284108


namespace complementary_angle_difference_l284_284271

def is_complementary (a b : ℝ) : Prop := a + b = 90

def in_ratio (a b : ℝ) (m n : ℝ) : Prop := a / b = m / n

theorem complementary_angle_difference (a b : ℝ) (h1 : is_complementary a b) (h2 : in_ratio a b 5 1) : abs (a - b) = 60 := 
by
  sorry

end complementary_angle_difference_l284_284271


namespace total_gold_is_100_l284_284372

-- Definitions based on conditions
def GregsGold : ℕ := 20
def KatiesGold : ℕ := GregsGold * 4
def TotalGold : ℕ := GregsGold + KatiesGold

-- Theorem to prove
theorem total_gold_is_100 : TotalGold = 100 := by
  sorry

end total_gold_is_100_l284_284372


namespace tan_105_degree_is_neg_sqrt3_minus_2_l284_284655

theorem tan_105_degree_is_neg_sqrt3_minus_2 :
  Real.tan (105 * Real.pi / 180) = -(Real.sqrt 3 + 2) := by
  sorry

end tan_105_degree_is_neg_sqrt3_minus_2_l284_284655


namespace base7_to_base10_l284_284293

theorem base7_to_base10 : (2 * 7^4 + 3 * 7^3 + 4 * 7^2 + 5 * 7^1 + 6 * 7^0) = 6068 := by
  sorry

end base7_to_base10_l284_284293


namespace purchase_price_l284_284801

theorem purchase_price (P : ℝ)
  (down_payment : ℝ) (monthly_payment : ℝ) (number_of_payments : ℝ)
  (interest_rate : ℝ) (total_paid : ℝ)
  (h1 : down_payment = 12)
  (h2 : monthly_payment = 10)
  (h3 : number_of_payments = 12)
  (h4 : interest_rate = 0.10714285714285714)
  (h5 : total_paid = 132) :
  P = 132 / 1.1071428571428572 :=
by
  sorry

end purchase_price_l284_284801


namespace absolute_value_simplification_l284_284808

theorem absolute_value_simplification : abs(-4^2 + 6) = 10 := by
  sorry

end absolute_value_simplification_l284_284808


namespace other_candidate_votes_l284_284777

-- Define the constants according to the problem
variables (X Y Z : ℝ)
axiom h1 : X = Y + (1 / 2) * Y
axiom h2 : X = 22500
axiom h3 : Y = Z - (2 / 5) * Z

-- Define the goal
theorem other_candidate_votes : Z = 25000 :=
by
  sorry

end other_candidate_votes_l284_284777


namespace main_l284_284938

def M (x : ℝ) : Prop := x^2 - 5 * x ≤ 0
def N (x : ℝ) (p : ℝ) : Prop := p < x ∧ x < 6
def intersection (x : ℝ) (q : ℝ) : Prop := 2 < x ∧ x ≤ q

theorem main (p q : ℝ) (hM : ∀ x, M x → 0 ≤ x ∧ x ≤ 5) (hN : ∀ x, N x p → p < x ∧ x < 6) (hMN : ∀ x, (M x ∧ N x p) ↔ intersection x q) :
  p + q = 7 :=
by
  sorry

end main_l284_284938


namespace two_digit_primes_with_ones_digit_3_count_eq_7_l284_284745

def two_digit_numbers_with_ones_digit_3 : List ℕ :=
  [13, 23, 33, 43, 53, 63, 73, 83, 93]

def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def count_prime_numbers_with_ones_digit_3 : ℕ :=
  (two_digit_numbers_with_ones_digit_3.filter is_prime).length

theorem two_digit_primes_with_ones_digit_3_count_eq_7 : 
  count_prime_numbers_with_ones_digit_3 = 7 := 
  sorry

end two_digit_primes_with_ones_digit_3_count_eq_7_l284_284745


namespace geometric_sequence_sum_l284_284943

theorem geometric_sequence_sum (a_n : ℕ → ℝ) (q : ℝ) (h1 : q = 2) (h2 : a_n 1 + a_n 3 = 5) :
  a_n 3 + a_n 5 = 20 :=
by
  -- The proof would go here, but it is not required for this task.
  sorry

end geometric_sequence_sum_l284_284943


namespace yerema_can_pay_exactly_l284_284713

theorem yerema_can_pay_exactly (t k b m : ℤ) 
    (h_foma : 3 * t + 4 * k + 5 * b = 11 * m) : 
    ∃ n : ℤ, 9 * t + k + 4 * b = 11 * n := 
by 
    sorry

end yerema_can_pay_exactly_l284_284713


namespace triangle_area_l284_284448

theorem triangle_area :
  ∀ (x y : ℝ), (3 * x + 2 * y = 12 ∧ x ≥ 0 ∧ y ≥ 0) →
  (1 / 2) * 4 * 6 = 12 := by
  sorry

end triangle_area_l284_284448


namespace probability_ratio_l284_284167

noncomputable def p := 10 / (nat.choose 50 5)
noncomputable def q := 2250 / (nat.choose 50 5)

theorem probability_ratio : q / p = 225 := 
by
  sorry

end probability_ratio_l284_284167


namespace find_m_l284_284772

noncomputable def f (x m : ℝ) : ℝ := (x^2 + m*x) * Real.exp x

def is_monotonically_decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f y ≤ f x

theorem find_m (m : ℝ) :
  is_monotonically_decreasing (f (m := m)) (-3/2) 1 ∧
  (-3/2)^2 + (m + 2)*(-3/2) + m = 0 ∧
  1^2 + (m + 2)*1 + m = 0 →
  m = -3/2 :=
by
  sorry

end find_m_l284_284772


namespace toys_of_Jason_l284_284514

theorem toys_of_Jason (R J Jason : ℕ) 
  (hR : R = 1) 
  (hJ : J = R + 6) 
  (hJason : Jason = 3 * J) : 
  Jason = 21 :=
by
  sorry

end toys_of_Jason_l284_284514


namespace area_decreases_by_28_l284_284946

def decrease_in_area (s h : ℤ) (h_eq : h = s + 3) : ℤ :=
  let new_area := (s - 4) * (s + 7)
  let original_area := s * h
  new_area - original_area

theorem area_decreases_by_28 (s h : ℤ) (h_eq : h = s + 3) : decrease_in_area s h h_eq = -28 :=
sorry

end area_decreases_by_28_l284_284946


namespace fraction_pow_zero_is_one_l284_284455

theorem fraction_pow_zero_is_one (a b : ℤ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) : (a / (b : ℚ)) ^ 0 = 1 := by
  sorry

end fraction_pow_zero_is_one_l284_284455


namespace p_p_values_l284_284247

def p (x y : ℤ) : ℤ :=
if 0 ≤ x ∧ 0 ≤ y then x + 2*y
else if x < 0 ∧ y < 0 then x - 3*y
else 4*x + y

theorem p_p_values : p (p 2 (-2)) (p (-3) (-1)) = 6 :=
by
  sorry

end p_p_values_l284_284247


namespace stewarts_theorem_l284_284495

theorem stewarts_theorem 
  (a b b₁ a₁ d c : ℝ)
  (h₁ : b * b ≠ 0) 
  (h₂ : a * a ≠ 0) 
  (h₃ : b₁ * b₁ ≠ 0) 
  (h₄ : a₁ * a₁ ≠ 0) 
  (h₅ : d * d ≠ 0) 
  (h₆ : c = a₁ + b₁) :
  b * b * a₁ + a * a * b₁ - d * d * c = a₁ * b₁ * c :=
  sorry

end stewarts_theorem_l284_284495


namespace dane_daughters_initial_flowers_l284_284699

theorem dane_daughters_initial_flowers :
  (exists (x y : ℕ), x = y ∧ 5 * 4 = 20 ∧ x + y = 30) →
  (exists f : ℕ, f = 5 ∧ 10 = 30 - 20 + 10 ∧ x = f * 2) :=
by
  -- Lean proof needs to go here
  sorry

end dane_daughters_initial_flowers_l284_284699


namespace sabina_loan_l284_284255

-- Define the conditions
def tuition_per_year : ℕ := 30000
def living_expenses_per_year : ℕ := 12000
def duration : ℕ := 4
def sabina_savings : ℕ := 10000
def grant_first_two_years_percent : ℕ := 40
def grant_last_two_years_percent : ℕ := 30
def scholarship_percent : ℕ := 20

-- Calculate total tuition for 4 years
def total_tuition : ℕ := tuition_per_year * duration

-- Calculate total living expenses for 4 years
def total_living_expenses : ℕ := living_expenses_per_year * duration

-- Calculate total cost
def total_cost : ℕ := total_tuition + total_living_expenses

-- Calculate grant coverage
def grant_first_two_years : ℕ := (grant_first_two_years_percent * tuition_per_year / 100) * 2
def grant_last_two_years : ℕ := (grant_last_two_years_percent * tuition_per_year / 100) * 2
def total_grant_coverage : ℕ := grant_first_two_years + grant_last_two_years

-- Calculate scholarship savings
def annual_scholarship_savings : ℕ := living_expenses_per_year * scholarship_percent / 100
def total_scholarship_savings : ℕ := annual_scholarship_savings * (duration - 1)

-- Calculate total reductions
def total_reductions : ℕ := total_grant_coverage + total_scholarship_savings + sabina_savings

-- Calculate the total loan needed
def total_loan_needed : ℕ := total_cost - total_reductions

theorem sabina_loan : total_loan_needed = 108800 := by
  sorry

end sabina_loan_l284_284255


namespace number_of_weavers_is_4_l284_284110

theorem number_of_weavers_is_4
  (mats1 days1 weavers1 mats2 days2 weavers2 : ℕ)
  (h1 : mats1 = 4)
  (h2 : days1 = 4)
  (h3 : weavers2 = 10)
  (h4 : mats2 = 25)
  (h5 : days2 = 10)
  (h_rate_eq : (mats1 / (weavers1 * days1)) = (mats2 / (weavers2 * days2))) :
  weavers1 = 4 :=
by
  sorry

end number_of_weavers_is_4_l284_284110


namespace tan_105_l284_284685

-- Defining the necessary known values and functions
def tan_addition (a b : ℝ) : ℝ := (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b)
def tan_60 : ℝ := Real.sqrt 3
def tan_45 : ℝ := 1

-- Proof goal in Lean 4
theorem tan_105 : Real.tan (105 * Real.pi / 180) = 2 + Real.sqrt 3 :=
by
  let tan_105 := tan_addition (60 * Real.pi / 180) (45 * Real.pi / 180)
  have h1 : Real.tan (60 * Real.pi / 180) = tan_60 := by sorry
  have h2 : Real.tan (45 * Real.pi / 180) = tan_45 := by sorry
  show Real.tan (105 * Real.pi / 180) = 2 + Real.sqrt 3 from sorry

end tan_105_l284_284685


namespace g_f_neg2_l284_284790

def f (x : ℤ) : ℤ := x^3 + 3

def g (x : ℤ) : ℤ := 2*x^2 + 2*x + 1

theorem g_f_neg2 : g (f (-2)) = 41 :=
by {
  -- proof steps skipped
  sorry
}

end g_f_neg2_l284_284790


namespace tan_105_eq_neg2_sub_sqrt3_l284_284626

theorem tan_105_eq_neg2_sub_sqrt3 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by 
  sorry

end tan_105_eq_neg2_sub_sqrt3_l284_284626


namespace tan_105_eq_neg2_sub_sqrt3_l284_284627

theorem tan_105_eq_neg2_sub_sqrt3 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by 
  sorry

end tan_105_eq_neg2_sub_sqrt3_l284_284627


namespace circle_radius_l284_284005

theorem circle_radius (a c r : ℝ) (h₁ : a = π * r^2) (h₂ : c = 2 * π * r) (h₃ : a + c = 100 * π) : 
  r = 9.05 := 
sorry

end circle_radius_l284_284005


namespace find_b_l284_284896

def h (x : ℝ) : ℝ := 5 * x - 7

theorem find_b : ∃ b : ℝ, h(b) = 0 ∧ b = 7 / 5 :=
by
  sorry

end find_b_l284_284896


namespace tan_105_eq_minus_2_minus_sqrt_3_l284_284673

theorem tan_105_eq_minus_2_minus_sqrt_3 :
  Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_eq_minus_2_minus_sqrt_3_l284_284673


namespace ratio_of_areas_of_squares_l284_284279

open Real

theorem ratio_of_areas_of_squares :
  let side_length_C := 48
  let side_length_D := 60
  let area_C := side_length_C^2
  let area_D := side_length_D^2
  area_C / area_D = (16 : ℝ) / 25 :=
by
  sorry

end ratio_of_areas_of_squares_l284_284279


namespace ellipse_condition_l284_284783

theorem ellipse_condition (m : ℝ) : 
  (∃ x y : ℝ, m * (x^2 + y^2 + 2*y + 1) = (x - 2*y + 3)^2) → m > 5 :=
by
  intro h
  sorry

end ellipse_condition_l284_284783


namespace find_principal_amount_l284_284147

variable (P : ℝ)

def interestA_to_B (P : ℝ) : ℝ := P * 0.10 * 3
def interestB_from_C (P : ℝ) : ℝ := P * 0.115 * 3
def gain_B (P : ℝ) : ℝ := interestB_from_C P - interestA_to_B P

theorem find_principal_amount (h : gain_B P = 45) : P = 1000 := by
  sorry

end find_principal_amount_l284_284147


namespace tan_105_eq_minus_2_minus_sqrt_3_l284_284677

theorem tan_105_eq_minus_2_minus_sqrt_3 :
  Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_eq_minus_2_minus_sqrt_3_l284_284677


namespace polynomial_rewrite_l284_284820

theorem polynomial_rewrite :
  ∃ (a b c d e f : ℤ), 
  (2401 * x^4 + 16 = (a * x + b) * (c * x^3 + d * x^2 + e * x + f)) ∧
  (a + b + c + d + e + f = 274) :=
sorry

end polynomial_rewrite_l284_284820


namespace twentieth_fisherman_caught_l284_284995

-- Definitions based on conditions
def fishermen_count : ℕ := 20
def total_fish_caught : ℕ := 10000
def fish_per_nineteen_fishermen : ℕ := 400
def nineteen_count : ℕ := 19

-- Calculation based on the problem conditions
def total_fish_by_nineteen : ℕ := nineteen_count * fish_per_nineteen_fishermen

-- Prove the number of fish caught by the twentieth fisherman
theorem twentieth_fisherman_caught : 
  total_fish_caught - total_fish_by_nineteen = 2400 := 
by
  -- This is where the proof would go
  sorry

end twentieth_fisherman_caught_l284_284995


namespace polynomial_horner_v4_value_l284_284584

-- Define the polynomial f(x)
def f (x : ℤ) : ℤ := x^6 - 12*x^5 + 60*x^4 - 160*x^3 + 240*x^2 - 192*x + 64

-- Define Horner's Rule step by step for x = 2
def horner_eval (x : ℤ) : ℤ :=
  let v0 := 1
  let v1 := v0 * x - 12
  let v2 := v1 * x + 60
  let v3 := v2 * x - 160
  let v4 := v3 * x + 240
  v4

-- Prove that the value of v4 when x = 2 is 80
theorem polynomial_horner_v4_value : horner_eval 2 = 80 := by
  sorry

end polynomial_horner_v4_value_l284_284584


namespace return_trip_speed_l284_284003

theorem return_trip_speed (d xy_dist : ℝ) (s xy_speed : ℝ) (avg_speed : ℝ) (r return_speed : ℝ) :
  xy_dist = 150 →
  xy_speed = 75 →
  avg_speed = 50 →
  2 * xy_dist / ((xy_dist / xy_speed) + (xy_dist / return_speed)) = avg_speed →
  return_speed = 37.5 :=
by
  intros hxy_dist hxy_speed h_avg_speed h_avg_speed_eq
  sorry

end return_trip_speed_l284_284003


namespace log_equation_solution_l284_284864

theorem log_equation_solution (x : ℝ) (hpos : x > 0) (hneq : x ≠ 1) : (Real.log 8 / Real.log x) * (2 * Real.log x / Real.log 2) = 6 * Real.log 2 :=
by
  sorry

end log_equation_solution_l284_284864


namespace christine_needs_min_bottles_l284_284343

noncomputable def fluidOuncesToLiters (fl_oz : ℝ) : ℝ := fl_oz / 33.8

noncomputable def litersToMilliliters (liters : ℝ) : ℝ := liters * 1000

noncomputable def bottlesRequired (total_ml : ℝ) (bottle_size_ml : ℝ) : ℕ := 
  Nat.ceil (total_ml / bottle_size_ml)

theorem christine_needs_min_bottles (required_fl_oz : ℝ) (bottle_size_ml : ℝ) (fl_oz_per_l : ℝ) :
  required_fl_oz = 60 →
  bottle_size_ml = 250 →
  fl_oz_per_l = 33.8 →
  bottlesRequired (litersToMilliliters (fluidOuncesToLiters required_fl_oz)) bottle_size_ml = 8 := by
  intros h1 h2 h3
  rw [h1, h2, h3]
  -- We can leave the proof as sorry which signals it's not yet completed.
  sorry

end christine_needs_min_bottles_l284_284343


namespace root_expression_value_l284_284070

theorem root_expression_value (p m n : ℝ) 
  (h1 : m^2 + (p - 2) * m + 1 = 0) 
  (h2 : n^2 + (p - 2) * n + 1 = 0) : 
  (m^2 + p * m + 1) * (n^2 + p * n + 1) - 2 = 2 :=
by
  sorry

end root_expression_value_l284_284070


namespace total_legs_proof_l284_284957

def johnny_legs : Nat := 2
def son_legs : Nat := 2
def dog_legs : Nat := 4
def number_of_dogs : Nat := 2
def number_of_humans : Nat := 2

def total_legs : Nat :=
  (number_of_dogs * dog_legs) + (number_of_humans * johnny_legs)

theorem total_legs_proof : total_legs = 12 := by
  sorry

end total_legs_proof_l284_284957


namespace range_of_m_l284_284734

theorem range_of_m (x y m : ℝ) 
  (h1 : 3 * x + y = m - 1)
  (h2 : x - 3 * y = 2 * m)
  (h3 : x + 2 * y ≥ 0) : 
  m ≤ -1 := 
sorry

end range_of_m_l284_284734


namespace intersection_complement_eq_l284_284100

noncomputable def U : Set Int := {-3, -2, -1, 0, 1, 2, 3}
noncomputable def A : Set Int := {-1, 0, 1, 2}
noncomputable def B : Set Int := {-3, 0, 2, 3}

-- Complement of B with respect to U
noncomputable def U_complement_B : Set Int := U \ B

-- The statement we need to prove
theorem intersection_complement_eq :
  A ∩ U_complement_B = {-1, 1} :=
by
  sorry

end intersection_complement_eq_l284_284100


namespace necessary_but_not_sufficient_l284_284198

section geometric_progression

variables {a b c : ℝ}

def geometric_progression (a b c : ℝ) : Prop :=
  ∃ r : ℝ, a = b / r ∧ c = b * r

def necessary_condition (a b c : ℝ) : Prop :=
  a * c = b^2

theorem necessary_but_not_sufficient :
  (geometric_progression a b c → necessary_condition a b c) ∧
  (¬ (necessary_condition a b c → geometric_progression a b c)) :=
by sorry

end geometric_progression

end necessary_but_not_sufficient_l284_284198


namespace quadratic_distinct_roots_l284_284178

theorem quadratic_distinct_roots (c : ℝ) (h : c < 1 / 4) : 
  ∃ r1 r2 : ℝ, r1 ≠ r2 ∧ (r1 * r1 + 2 * r1 + 4 * c = 0) ∧ (r2 * r2 + 2 * r2 + 4 * c = 0) :=
by 
sorry

end quadratic_distinct_roots_l284_284178


namespace servings_correct_l284_284122

-- Define the pieces of popcorn in a serving
def pieces_per_serving := 30

-- Define the pieces of popcorn Jared can eat
def jared_pieces := 90

-- Define the pieces of popcorn each friend can eat
def friend_pieces := 60

-- Define the number of friends
def friends := 3

-- Calculate total pieces eaten by friends
def total_friend_pieces := friends * friend_pieces

-- Calculate total pieces eaten by everyone
def total_pieces := jared_pieces + total_friend_pieces

-- Calculate the number of servings needed
def servings_needed := total_pieces / pieces_per_serving

theorem servings_correct : servings_needed = 9 :=
by
  sorry

end servings_correct_l284_284122


namespace sequence_formula_l284_284540

def a (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 3
  | 2 => 6
  | 3 => 10
  | _ => sorry  -- The pattern is more general

theorem sequence_formula (n : ℕ) : a n = (n * (n + 1)) / 2 := 
  sorry

end sequence_formula_l284_284540


namespace proof_x_plus_y_equals_30_l284_284522

variable (x y : ℝ) (h_distinct : x ≠ y)
variable (h_det : Matrix.det ![
  ![2, 5, 10],
  ![4, x, y],
  ![4, y, x]
  ] = 0)

theorem proof_x_plus_y_equals_30 :
  x + y = 30 :=
sorry

end proof_x_plus_y_equals_30_l284_284522


namespace tan_105_l284_284603

theorem tan_105 :
  tan 105 = -2 - sqrt 3 :=
by sorry

end tan_105_l284_284603


namespace ellipse_properties_l284_284818

theorem ellipse_properties (h k a b : ℝ) (θ : ℝ)
  (h_def : h = -2)
  (k_def : k = 3)
  (a_def : a = 6)
  (b_def : b = 4)
  (θ_def : θ = 45) :
  h + k + a + b = 11 :=
by
  sorry

end ellipse_properties_l284_284818


namespace largest_five_digit_number_divisible_by_5_l284_284130

theorem largest_five_digit_number_divisible_by_5 : 
  ∃ n, (n % 5 = 0) ∧ (99990 ≤ n) ∧ (n ≤ 99995) ∧ (∀ m, (m % 5 = 0) → (99990 ≤ m) → (m ≤ 99995) → m ≤ n) :=
by
  -- The proof is omitted as per the instructions
  sorry

end largest_five_digit_number_divisible_by_5_l284_284130


namespace tan_105_degree_l284_284639

theorem tan_105_degree : Real.tan (Real.pi * 105 / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_degree_l284_284639


namespace tan_105_eq_neg2_sub_sqrt3_l284_284692

-- Define the main theorem to be proven
theorem tan_105_eq_neg2_sub_sqrt3 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_eq_neg2_sub_sqrt3_l284_284692


namespace Josh_pencils_left_l284_284390

theorem Josh_pencils_left (initial_pencils : ℕ) (given_pencils : ℕ) (remaining_pencils : ℕ) 
  (h_initial : initial_pencils = 142) 
  (h_given : given_pencils = 31) 
  (h_remaining : remaining_pencils = 111) : 
  initial_pencils - given_pencils = remaining_pencils :=
by
  sorry

end Josh_pencils_left_l284_284390


namespace average_multiples_of_10_l284_284432

theorem average_multiples_of_10 (a b : ℕ) (h₁ : a = 10) (h₂ : b = 400) :
  (∑ k in finset.range ((b - a) / 10 + 1), (a + k * 10)) / ((b - a) / 10 + 1) = 205 :=
by
  -- Using ∑ to denote the sum of all multiples of 10 in the given range
  sorry

end average_multiples_of_10_l284_284432


namespace intersection_nonempty_l284_284419

open Nat

theorem intersection_nonempty (a : ℕ) (ha : a ≥ 2) :
  ∃ (b : ℕ), b = 1 ∨ b = a ∧
  ∃ y, (∃ x, y = a^x ∧ x ≥ 1) ∧
       (∃ x, y = (a + 1)^x + b ∧ x ≥ 1) :=
by sorry

end intersection_nonempty_l284_284419


namespace original_denominator_is_nine_l284_284154

theorem original_denominator_is_nine (d : ℕ) : 
  (2 + 5) / (d + 5) = 1 / 2 → d = 9 := 
by sorry

end original_denominator_is_nine_l284_284154


namespace cindy_envelopes_l284_284345

theorem cindy_envelopes (h₁ : ℕ := 4) (h₂ : ℕ := 7) (h₃ : ℕ := 5) (h₄ : ℕ := 10) (h₅ : ℕ := 3) (initial : ℕ := 137) :
  initial - (h₁ + h₂ + h₃ + h₄ + h₅) = 108 :=
by
  sorry

end cindy_envelopes_l284_284345


namespace valentines_initial_l284_284251

theorem valentines_initial (gave_away : ℕ) (left_over : ℕ) (initial : ℕ) : 
  gave_away = 8 → left_over = 22 → initial = gave_away + left_over → initial = 30 :=
by
  intros h1 h2 h3
  sorry

end valentines_initial_l284_284251


namespace max_percent_liquid_X_l284_284525

theorem max_percent_liquid_X (wA wB wC : ℝ) (XA XB XC YA YB YC : ℝ)
  (hXA : XA = 0.8 / 100) (hXB : XB = 1.8 / 100) (hXC : XC = 3.0 / 100)
  (hYA : YA = 2.0 / 100) (hYB : YB = 1.0 / 100) (hYC : YC = 0.5 / 100)
  (hwA : wA = 500) (hwB : wB = 700) (hwC : wC = 300)
  (H_combined_limit : XA * wA + XB * wB + XC * wC + YA * wA + YB * wB + YC * wC ≤ 0.025 * (wA + wB + wC)) :
  XA * wA + XB * wB + XC * wC ≤ 0.0171 * (wA + wB + wC) :=
sorry

end max_percent_liquid_X_l284_284525


namespace polygon_properties_l284_284722

def interior_angle_sum (n : ℕ) : ℝ :=
  (n - 2) * 180

def exterior_angle_sum : ℝ :=
  360

theorem polygon_properties (n : ℕ) (h : interior_angle_sum n = 3 * exterior_angle_sum + 180) :
  n = 9 ∧ interior_angle_sum n / n = 140 :=
by
  sorry

end polygon_properties_l284_284722


namespace initial_percentage_of_grape_juice_l284_284221

theorem initial_percentage_of_grape_juice (P : ℝ) 
  (h₀ : 10 + 30 = 40)
  (h₁ : 40 * 0.325 = 13)
  (h₂ : 30 * P + 10 = 13) : 
  P = 0.1 :=
  by 
    sorry

end initial_percentage_of_grape_juice_l284_284221


namespace evaluate_neg_64_pow_two_thirds_l284_284033

theorem evaluate_neg_64_pow_two_thirds 
  (h : -64 = (-4)^3) : (-64)^(2/3) = 16 := 
by 
  -- sorry added to skip the proof.
  sorry  

end evaluate_neg_64_pow_two_thirds_l284_284033


namespace canoe_problem_l284_284002

-- Definitions:
variables (P_L P_R : ℝ)

-- Conditions:
def conditions := 
  (P_L = P_R) ∧ -- Condition that the probabilities for left and right oars working are the same
  (0 ≤ P_L) ∧ (P_L ≤ 1) ∧ -- Probability values must be between 0 and 1
  (1 - (1 - P_L) * (1 - P_R) = 0.84) -- Given the rowing probability is 0.84

-- Theorem that P_L = 0.6 given the conditions:
theorem canoe_problem : conditions P_L P_R → P_L = 0.6 :=
by
  sorry

end canoe_problem_l284_284002


namespace base7_to_base10_l284_284291

theorem base7_to_base10 : (2 * 7^4 + 3 * 7^3 + 4 * 7^2 + 5 * 7^1 + 6 * 7^0) = 6068 := by
  sorry

end base7_to_base10_l284_284291


namespace tan_105_degree_l284_284635

theorem tan_105_degree : Real.tan (Real.pi * 105 / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_degree_l284_284635


namespace barry_should_pay_l284_284337

def original_price : ℝ := 80
def discount_rate : ℝ := 0.15

theorem barry_should_pay:
  original_price * (1 - discount_rate) = 68 := 
by 
  -- Original price: 80
  -- Discount rate: 0.15
  -- Question: Final price after discount
  sorry

end barry_should_pay_l284_284337


namespace inequality_4th_power_l284_284405

theorem inequality_4th_power (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) (h_ineq : a ≥ b) :
  (a^4 + b^4) / 2 ≥ ((a + b) / 2)^4 :=
sorry

end inequality_4th_power_l284_284405


namespace product_of_two_larger_numbers_is_115_l284_284550

noncomputable def proofProblem : Prop :=
  ∃ (A B C : ℝ), B = 10 ∧ (C - B = B - A) ∧ (A * B = 85) ∧ (B * C = 115)

theorem product_of_two_larger_numbers_is_115 : proofProblem :=
by
  sorry

end product_of_two_larger_numbers_is_115_l284_284550


namespace jogger_usual_speed_l284_284443

theorem jogger_usual_speed (V T : ℝ) 
    (h_actual: 30 = V * T) 
    (h_condition: 40 = 16 * T) 
    (h_distance: T = 30 / V) :
  V = 12 := 
by
  sorry

end jogger_usual_speed_l284_284443


namespace tan_105_l284_284601

theorem tan_105 :
  tan 105 = -2 - sqrt 3 :=
by sorry

end tan_105_l284_284601


namespace inequality_not_always_true_l284_284062

theorem inequality_not_always_true
  (x y w : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x^2 > y^2) (hw : w ≠ 0) :
  ∃ w, w ≠ 0 ∧ x^2 * w ≤ y^2 * w :=
sorry

end inequality_not_always_true_l284_284062


namespace opposite_points_l284_284969

theorem opposite_points (A B : ℝ) (h1 : A = -B) (h2 : A < B) (h3 : abs (A - B) = 6.4) : A = -3.2 ∧ B = 3.2 :=
by
  sorry

end opposite_points_l284_284969


namespace kevin_distance_after_six_hops_l284_284516

def kevin_hops : ℚ := (1/2 : ℚ) + (1/4 : ℚ) + (1/8 : ℚ) + (1/16 : ℚ) + (1/32 : ℚ) + (1/64 : ℚ)

theorem kevin_distance_after_six_hops : kevin_hops = (63/64 : ℚ) :=
by {
  -- Proof place holder
  sorry
}

end kevin_distance_after_six_hops_l284_284516


namespace range_of_a_for_critical_points_l284_284771

noncomputable def f (a x : ℝ) : ℝ := x^3 - a * x^2 + a * x + 3

theorem range_of_a_for_critical_points : 
  ∀ a : ℝ, (∃ x : ℝ, deriv (f a) x = 0) ↔ (a < 0 ∨ a > 3) :=
by
  sorry

end range_of_a_for_critical_points_l284_284771


namespace find_smallest_n_l284_284780

-- defining the geometric sequence and its sum for the given conditions
def a_n (n : ℕ) := 3 * (4 ^ n)

def S_n (n : ℕ) := (a_n n - 1) / (4 - 1) -- simplification step

-- statement of the problem: finding the smallest natural number n such that S_n > 3000
theorem find_smallest_n :
  ∃ n : ℕ, S_n n > 3000 ∧ ∀ m : ℕ, m < n → S_n m ≤ 3000 := by
  sorry

end find_smallest_n_l284_284780


namespace position_2023_l284_284013

def initial_position := "ABCD"

def rotate_180 (pos : String) : String :=
  match pos with
  | "ABCD" => "CDAB"
  | "CDAB" => "ABCD"
  | "DCBA" => "BADC"
  | "BADC" => "DCBA"
  | _ => pos

def reflect_horizontal (pos : String) : String :=
  match pos with
  | "ABCD" => "ABCD"
  | "CDAB" => "DCBA"
  | "DCBA" => "CDAB"
  | "BADC" => "BADC"
  | _ => pos

def transformation (n : ℕ) : String :=
  let cnt := n % 4
  if cnt = 1 then rotate_180 initial_position
  else if cnt = 2 then rotate_180 (rotate_180 initial_position)
  else if cnt = 3 then rotate_180 (reflect_horizontal (rotate_180 initial_position))
  else reflect_horizontal initial_position

theorem position_2023 : transformation 2023 = "DCBA" := by
  sorry

end position_2023_l284_284013


namespace vertical_strips_count_l284_284508

theorem vertical_strips_count (a b x y : ℕ)
  (h_outer : 2 * a + 2 * b = 50)
  (h_inner : 2 * x + 2 * y = 32)
  (h_strips : a + x = 20) :
  b + y = 21 :=
by
  have h1 : a + b = 25 := by
    linarith
  have h2 : x + y = 16 := by
    linarith
  linarith


end vertical_strips_count_l284_284508


namespace tan_105_l284_284604

theorem tan_105 :
  tan 105 = -2 - sqrt 3 :=
by sorry

end tan_105_l284_284604


namespace number_of_ferns_is_six_l284_284238

def num_fronds_per_fern : Nat := 7
def num_leaves_per_frond : Nat := 30
def total_leaves : Nat := 1260

theorem number_of_ferns_is_six :
  total_leaves = num_fronds_per_fern * num_leaves_per_frond * 6 :=
by
  sorry

end number_of_ferns_is_six_l284_284238


namespace total_people_surveyed_l284_284515

-- Define the conditions
variable (total_surveyed : ℕ) (disease_believers : ℕ)
variable (rabies_believers : ℕ)

-- Condition 1: 75% of the people surveyed thought rats carried diseases
def condition1 (total_surveyed disease_believers : ℕ) : Prop :=
  disease_believers = (total_surveyed * 75) / 100

-- Condition 2: 50% of the people who thought rats carried diseases said rats frequently carried rabies
def condition2 (disease_believers rabies_believers : ℕ) : Prop :=
  rabies_believers = (disease_believers * 50) / 100

-- Condition 3: 18 people were mistaken in thinking rats frequently carry rabies
def condition3 (rabies_believers : ℕ) : Prop := rabies_believers = 18

-- The theorem to prove the total number of people surveyed given the conditions
theorem total_people_surveyed (total_surveyed disease_believers rabies_believers : ℕ) :
  condition1 total_surveyed disease_believers →
  condition2 disease_believers rabies_believers →
  condition3 rabies_believers →
  total_surveyed = 48 :=
by sorry

end total_people_surveyed_l284_284515


namespace p_sufficient_not_necessary_for_q_l284_284717

def p (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 ≤ 2
def q (x y : ℝ) : Prop := y ≥ x - 1 ∧ y ≥ 1 - x ∧ y ≤ 1

theorem p_sufficient_not_necessary_for_q :
  (∀ x y : ℝ, q x y → p x y) ∧ ¬(∀ x y : ℝ, p x y → q x y) := by
  sorry

end p_sufficient_not_necessary_for_q_l284_284717


namespace probability_of_selecting_cocaptains_l284_284187

open Nat

-- Define the sizes of the teams
def team_sizes : List ℕ := [6, 8, 9, 10]

-- Define the calculation of the probability of selecting two co-captains for a given team size
def probability_co_captains (n : ℕ) : ℚ :=
  1 / (choose n 2)

-- Define the overall probability
def total_probability : ℚ :=
  (1/4) * ((probability_co_captains 6) + (probability_co_captains 8) +
           (probability_co_captains 9) + (probability_co_captains 10))

theorem probability_of_selecting_cocaptains :
    total_probability = 131 / 5040 :=
  sorry

end probability_of_selecting_cocaptains_l284_284187


namespace num_two_digit_primes_with_ones_digit_eq_3_l284_284743

theorem num_two_digit_primes_with_ones_digit_eq_3 : 
  (finset.filter (λ n, n.mod 10 = 3 ∧ nat.prime n) (finset.range 100).filter (λ n, n >= 10)).card = 6 :=
by
  sorry

end num_two_digit_primes_with_ones_digit_eq_3_l284_284743


namespace statement_not_always_true_l284_284509

theorem statement_not_always_true 
  (a b c d : ℝ)
  (h1 : (a + b) / (3 * a - b) = (b + c) / (3 * b - c))
  (h2 : (b + c) / (3 * b - c) = (c + d) / (3 * c - d))
  (h3 : (c + d) / (3 * c - d) = (d + a) / (3 * d - a))
  (h4 : (d + a) / (3 * d - a) = (a + b) / (3 * a - b)) :
  a^2 + b^2 + c^2 + d^2 ≠ ab + bc + cd + da :=
by {
  sorry
}

end statement_not_always_true_l284_284509


namespace ratio_of_sheep_to_cow_l284_284450

noncomputable def sheep_to_cow_ratio 
  (S : ℕ) 
  (h1 : 12 + 4 * S = 108) 
  (h2 : S ≠ 0) : ℕ × ℕ := 
if h3 : 12 = 0 then (0, 0) else (2, 1)

theorem ratio_of_sheep_to_cow 
  (S : ℕ) 
  (h1 : 12 + 4 * S = 108) 
  (h2 : S ≠ 0) : sheep_to_cow_ratio S h1 h2 = (2, 1) := 
sorry

end ratio_of_sheep_to_cow_l284_284450


namespace distinct_solution_count_l284_284555

theorem distinct_solution_count
  (n : ℕ)
  (x y : ℕ)
  (h1 : x ≠ y)
  (h2 : x ≠ 2 * y)
  (h3 : y ≠ 2 * x)
  (h4 : x^2 - x * y + y^2 = n) :
  ∃ (pairs : Finset (ℕ × ℕ)), pairs.card ≥ 12 ∧ ∀ (a b : ℕ), (a, b) ∈ pairs → a^2 - a * b + b^2 = n :=
sorry

end distinct_solution_count_l284_284555


namespace tan_105_degree_l284_284614

theorem tan_105_degree : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 := by
  have h_add : ∀ a b : ℝ, Real.tan (a + b) = (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b) :=
    Real.tan_add

  have h_tan_60 : Real.tan (60 * Real.pi / 180) = Real.sqrt 3 := Real.tan_60
  have h_tan_45 : Real.tan (45 * Real.pi / 180) = 1 := Real.tan_45
  
  sorry

end tan_105_degree_l284_284614


namespace no_solution_l284_284243

def is_digit (B : ℕ) : Prop := B < 10

def divisible_by (n m : ℕ) : Prop := ∃ k, n = m * k

def satisfies_conditions (B : ℕ) : Prop :=
  is_digit B ∧
  divisible_by (12345670 + B) 2 ∧
  divisible_by (12345670 + B) 5 ∧
  divisible_by (12345670 + B) 11

theorem no_solution (B : ℕ) : ¬ satisfies_conditions B :=
sorry

end no_solution_l284_284243


namespace tan_105_eq_neg2_sub_sqrt3_l284_284696

-- Define the main theorem to be proven
theorem tan_105_eq_neg2_sub_sqrt3 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_eq_neg2_sub_sqrt3_l284_284696


namespace tan_105_l284_284681

-- Defining the necessary known values and functions
def tan_addition (a b : ℝ) : ℝ := (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b)
def tan_60 : ℝ := Real.sqrt 3
def tan_45 : ℝ := 1

-- Proof goal in Lean 4
theorem tan_105 : Real.tan (105 * Real.pi / 180) = 2 + Real.sqrt 3 :=
by
  let tan_105 := tan_addition (60 * Real.pi / 180) (45 * Real.pi / 180)
  have h1 : Real.tan (60 * Real.pi / 180) = tan_60 := by sorry
  have h2 : Real.tan (45 * Real.pi / 180) = tan_45 := by sorry
  show Real.tan (105 * Real.pi / 180) = 2 + Real.sqrt 3 from sorry

end tan_105_l284_284681


namespace amount_received_is_500_l284_284510

-- Define the conditions
def books_per_month : ℕ := 3
def months_per_year : ℕ := 12
def price_per_book : ℕ := 20
def loss : ℕ := 220

-- Calculate number of books bought in a year
def books_per_year : ℕ := books_per_month * months_per_year

-- Calculate total amount spent on books in a year
def total_spent : ℕ := books_per_year * price_per_book

-- Calculate the amount Jack got from selling the books based on the given loss
def amount_received : ℕ := total_spent - loss

-- Proving the amount received is $500
theorem amount_received_is_500 : amount_received = 500 := by
  sorry

end amount_received_is_500_l284_284510


namespace tan_105_l284_284605

theorem tan_105 :
  tan 105 = -2 - sqrt 3 :=
by sorry

end tan_105_l284_284605


namespace tan_105_eq_neg2_sub_sqrt3_l284_284695

-- Define the main theorem to be proven
theorem tan_105_eq_neg2_sub_sqrt3 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_eq_neg2_sub_sqrt3_l284_284695


namespace num_two_digit_primes_with_ones_digit_eq_3_l284_284744

theorem num_two_digit_primes_with_ones_digit_eq_3 : 
  (finset.filter (λ n, n.mod 10 = 3 ∧ nat.prime n) (finset.range 100).filter (λ n, n >= 10)).card = 6 :=
by
  sorry

end num_two_digit_primes_with_ones_digit_eq_3_l284_284744


namespace smallest_odd_number_with_five_prime_factors_is_15015_l284_284845

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m ∣ n, m = 1 ∨ m = n

def smallest_odd_primes : List ℕ := [3, 5, 7, 11, 13]

def has_five_distinct_prime_factors (n : ℕ) : Prop :=
  ∃ (factors : List ℕ), factors.nodup ∧ factors.length = 5 ∧ (∀ p ∈ factors, is_prime p) ∧ factors.prod = n

def is_odd (n : ℕ) : Prop := n % 2 = 1

def smallest_odd_number_with_five_prime_factors (n : ℕ) : Prop :=
  has_five_distinct_prime_factors n ∧ is_odd n

theorem smallest_odd_number_with_five_prime_factors_is_15015 :
  smallest_odd_number_with_five_prime_factors 15015 :=
by
  sorry

end smallest_odd_number_with_five_prime_factors_is_15015_l284_284845


namespace solution_existence_l284_284044

theorem solution_existence (m : ℤ) :
  (∀ x y : ℤ, 2 * x + (m - 1) * y = 3 ∧ (m + 1) * x + 4 * y = -3) ↔
  (m = -3 ∨ m = 3 → 
    (m = -3 → ∃ x y : ℤ, 2 * x + (m - 1) * y = 3 ∧ (m + 1) * x + 4 * y = -3) ∧
    (m = 3 → ¬∃ x y : ℤ, 2 * x + (m - 1) * y = 3 ∧ (m + 1) * x + 4 * y = -3)) := by
  sorry

end solution_existence_l284_284044


namespace base6_sum_eq_10_l284_284393

theorem base6_sum_eq_10 
  (A B C : ℕ) 
  (hA : 0 < A ∧ A < 6) 
  (hB : 0 < B ∧ B < 6) 
  (hC : 0 < C ∧ C < 6)
  (distinct : A ≠ B ∧ B ≠ C ∧ C ≠ A)
  (h_add : A*36 + B*6 + C + B*6 + C = A*36 + C*6 + A) :
  A + B + C = 10 := 
by
  sorry

end base6_sum_eq_10_l284_284393


namespace crayons_given_to_mary_l284_284580

theorem crayons_given_to_mary :
  let pack_crayons := 21 in
  let locker_crayons := 36 in
  let bobby_crayons := locker_crayons / 2 in
  let total_crayons := pack_crayons + locker_crayons + bobby_crayons in
  (total_crayons * (1 / 3) = 25) := by
rfl

end crayons_given_to_mary_l284_284580


namespace regular_polygon_sides_and_interior_angle_l284_284720

theorem regular_polygon_sides_and_interior_angle (n : ℕ) (H : (n - 2) * 180 = 3 * 360 + 180) :
  n = 9 ∧ (n - 2) * 180 / n = 140 :=
by
-- This marks the start of the proof, but the proof is omitted.
sorry

end regular_polygon_sides_and_interior_angle_l284_284720


namespace classroom_position_l284_284769

theorem classroom_position (a b c d : ℕ) (h : (1, 2) = (a, b)) : (3, 2) = (c, d) :=
by
  sorry

end classroom_position_l284_284769


namespace find_larger_number_l284_284538

theorem find_larger_number (L S : ℕ) 
  (h1 : L - S = 1000) 
  (h2 : L = 10 * S + 10) : 
  L = 1110 :=
sorry

end find_larger_number_l284_284538


namespace find_m_l284_284091

noncomputable def geometric_sequence_solution (S : ℕ → ℝ) (a : ℕ → ℝ) (q : ℝ) (m : ℕ) : Prop :=
  (S 3 + S 6 = 2 * S 9) ∧ (a 2 + a 5 = 2 * a m)

theorem find_m (S : ℕ → ℝ) (a : ℕ → ℝ) (q : ℝ) (m : ℕ) (h1 : S 3 + S 6 = 2 * S 9)
  (h2 : a 2 + a 5 = 2 * a m) : m = 8 :=
sorry

end find_m_l284_284091


namespace min_value_of_u_l284_284061

variable (a b : ℝ)
variable (h1 : 0 < a)
variable (h2 : 0 < b)
variable (h3 : a^2 - b + 4 ≤ 0)

theorem min_value_of_u : (∃ (u : ℝ), u = (2*a + 3*b) / (a + b) ∧ u ≥ 14/5) :=
sorry

end min_value_of_u_l284_284061


namespace q_f_digit_div_36_l284_284562

theorem q_f_digit_div_36 (q f : ℕ) (hq : q ≠ f) (hq_digit: q < 10) (hf_digit: f < 10) :
    (457 * 10000 + q * 1000 + 89 * 10 + f) % 36 = 0 → q + f = 6 :=
sorry

end q_f_digit_div_36_l284_284562


namespace complementary_angle_difference_l284_284273

def is_complementary (a b : ℝ) : Prop := a + b = 90

def in_ratio (a b : ℝ) (m n : ℝ) : Prop := a / b = m / n

theorem complementary_angle_difference (a b : ℝ) (h1 : is_complementary a b) (h2 : in_ratio a b 5 1) : abs (a - b) = 60 := 
by
  sorry

end complementary_angle_difference_l284_284273


namespace tan_105_eq_neg2_sub_sqrt3_l284_284625

theorem tan_105_eq_neg2_sub_sqrt3 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by 
  sorry

end tan_105_eq_neg2_sub_sqrt3_l284_284625


namespace abs_neg_four_squared_plus_six_l284_284811

theorem abs_neg_four_squared_plus_six : |(-4^2 + 6)| = 10 := by
  -- We skip the proof steps according to the instruction
  sorry

end abs_neg_four_squared_plus_six_l284_284811


namespace tan_105_eq_neg2_sub_sqrt3_l284_284631

theorem tan_105_eq_neg2_sub_sqrt3 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by 
  sorry

end tan_105_eq_neg2_sub_sqrt3_l284_284631


namespace base_7_to_10_of_23456_l284_284290

theorem base_7_to_10_of_23456 : 
  (2 * 7 ^ 4 + 3 * 7 ^ 3 + 4 * 7 ^ 2 + 5 * 7 ^ 1 + 6 * 7 ^ 0) = 6068 :=
by sorry

end base_7_to_10_of_23456_l284_284290


namespace barry_should_pay_l284_284336

def original_price : ℝ := 80
def discount_rate : ℝ := 0.15

theorem barry_should_pay:
  original_price * (1 - discount_rate) = 68 := 
by 
  -- Original price: 80
  -- Discount rate: 0.15
  -- Question: Final price after discount
  sorry

end barry_should_pay_l284_284336


namespace tan_add_tan_105_eq_l284_284661

noncomputable def tan : ℝ → ℝ := sorry -- Use the built-in library later for actual implementation

-- Given conditions
def tan_45_eq : tan 45 = 1 := by sorry
def tan_60_eq : tan 60 = Real.sqrt 3 := by sorry

-- Angle addition formula for tangent
theorem tan_add (a b : ℝ) :
  tan (a + b) = (tan a + tan b) / (1 - tan a * tan b) := by sorry

-- Main theorem to prove
theorem tan_105_eq :
  tan 105 = -2 - Real.sqrt 3 := by sorry

end tan_add_tan_105_eq_l284_284661


namespace tan_105_degree_l284_284640

theorem tan_105_degree : Real.tan (Real.pi * 105 / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_degree_l284_284640


namespace tan_105_eq_neg2_sub_sqrt3_l284_284671

theorem tan_105_eq_neg2_sub_sqrt3 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 := by
  sorry

end tan_105_eq_neg2_sub_sqrt3_l284_284671


namespace minimum_value_f_range_a_l284_284521

noncomputable def f (x : ℝ) : ℝ := (x + 1) * Real.log (x + 1)

theorem minimum_value_f :
  ∃ x : ℝ, f x = -(1 / Real.exp 1) :=
sorry

theorem range_a (a : ℝ) :
  (∀ x ≥ 0, f x ≥ a * x) ↔ a ∈ Set.Iic 1 :=
sorry

end minimum_value_f_range_a_l284_284521


namespace find_k_value_l284_284544

theorem find_k_value (k : ℚ) (h1 : (3, -5) ∈ {p : ℚ × ℚ | p.snd = k * p.fst}) (h2 : k ≠ 0) : k = -5 / 3 :=
sorry

end find_k_value_l284_284544


namespace change_after_buying_tickets_l284_284287

def cost_per_ticket := 8
def number_of_tickets := 2
def total_money := 25

theorem change_after_buying_tickets :
  total_money - number_of_tickets * cost_per_ticket = 9 := by
  sorry

end change_after_buying_tickets_l284_284287


namespace poly_div_l284_284459

theorem poly_div (A B : ℂ) :
  (∀ x : ℂ, x^3 + x^2 + 1 = 0 → x^202 + A * x + B = 0) → A + B = 0 :=
by
  intros h
  sorry

end poly_div_l284_284459


namespace seats_in_row_l284_284004

theorem seats_in_row (y : ℕ → ℕ) (k b : ℕ) :
  (∀ x, y x = k * x + b) →
  y 1 = 20 →
  y 19 = 56 →
  y 26 = 70 :=
by
  intro h1 h2 h3
  -- Additional constraints to prove the given requirements
  sorry

end seats_in_row_l284_284004


namespace rectangle_area_l284_284824

theorem rectangle_area :
  ∀ (width length : ℝ), (length = 3 * width) → (width = 5) → (length * width = 75) :=
by
  intros width length h1 h2
  rw [h2, h1]
  sorry

end rectangle_area_l284_284824


namespace cos_seven_pi_over_six_l284_284914

theorem cos_seven_pi_over_six : Real.cos (7 * Real.pi / 6) = -Real.sqrt 3 / 2 :=
by
  -- Place the proof here
  sorry

end cos_seven_pi_over_six_l284_284914


namespace max_M_range_a_l284_284048

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := a / x + x * Real.log x
def g (x : ℝ) : ℝ := x^3 - x^2 - 3

theorem max_M (x1 x2 : ℝ) (h1 : 0 ≤ x1) (h2 : x1 ≤ 2) (h3 : 0 ≤ x2) (h4 : x2 ≤ 2) : 
  4 ≤ g x1 - g x2 :=
sorry

theorem range_a (a : ℝ) (s t : ℝ) (h1 : 1 / 2 ≤ s) (h2 : s ≤ 2) (h3 : 1 / 2 ≤ t) (h4 : t ≤ 2) : 
  1 ≤ a ∧ f s a ≥ g t :=
sorry

end max_M_range_a_l284_284048


namespace rectangle_same_color_exists_l284_284815

theorem rectangle_same_color_exists (color : ℝ × ℝ → Prop) (red blue : Prop) (h : ∀ p : ℝ × ℝ, color p = red ∨ color p = blue) :
  ∃ (a b c d : ℝ × ℝ), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧ a ≠ c ∧ b ≠ d ∧
  (color a = color b ∧ color b = color c ∧ color c = color d) :=
sorry

end rectangle_same_color_exists_l284_284815


namespace max_value_of_polynomial_l284_284867

theorem max_value_of_polynomial :
  ∃ x : ℝ, (x = -1) ∧ ∀ y : ℝ, -3 * y^2 - 6 * y + 12 ≤ -3 * (-1)^2 - 6 * (-1) + 12 := by
  sorry

end max_value_of_polynomial_l284_284867


namespace two_digit_primes_end_in_3_l284_284760

theorem two_digit_primes_end_in_3 : 
  {n : ℕ | n ≥ 10 ∧ n < 100 ∧ n % 10 = 3 ∧ Prime n}.card = 6 := 
by
  sorry

end two_digit_primes_end_in_3_l284_284760


namespace discount_correct_l284_284874

def normal_cost : ℝ := 80
def discount_rate : ℝ := 0.45
def discounted_cost : ℝ := normal_cost - (discount_rate * normal_cost)

theorem discount_correct : discounted_cost = 44 := by
  -- By computation, 0.45 * 80 = 36 and 80 - 36 = 44
  sorry

end discount_correct_l284_284874


namespace two_digit_primes_with_ones_digit_3_count_eq_7_l284_284746

def two_digit_numbers_with_ones_digit_3 : List ℕ :=
  [13, 23, 33, 43, 53, 63, 73, 83, 93]

def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def count_prime_numbers_with_ones_digit_3 : ℕ :=
  (two_digit_numbers_with_ones_digit_3.filter is_prime).length

theorem two_digit_primes_with_ones_digit_3_count_eq_7 : 
  count_prime_numbers_with_ones_digit_3 = 7 := 
  sorry

end two_digit_primes_with_ones_digit_3_count_eq_7_l284_284746


namespace A_plays_D_third_day_l284_284714

section GoTournament

variables (Player : Type) (A B C D : Player) 

-- Define the condition that each player competes with every other player exactly once.
def each_plays_once (P : Player → Player → Prop) : Prop :=
  ∀ x y, x ≠ y → (P x y ∨ P y x)

-- Define the tournament setup and the play conditions.
variables (P : Player → Player → Prop)
variable [∀ x y, Decidable (P x y)] -- Assuming decidability for the play relation

-- The given conditions of the problem
axiom A_plays_C_first_day : P A C
axiom C_plays_D_second_day : P C D
axiom only_one_match_per_day : ∀ x, ∃! y, P x y

-- We aim to prove that A will play against D on the third day.
theorem A_plays_D_third_day : P A D :=
sorry

end GoTournament

end A_plays_D_third_day_l284_284714


namespace kiwi_count_l284_284319

theorem kiwi_count (o a b k : ℕ) (h1 : o + a + b + k = 540) (h2 : a = 3 * o) (h3 : b = 4 * a) (h4 : k = 5 * b) : k = 420 :=
sorry

end kiwi_count_l284_284319


namespace simplify_fraction_eq_l284_284807

theorem simplify_fraction_eq : (180 / 270 : ℚ) = 2 / 3 :=
by
  sorry

end simplify_fraction_eq_l284_284807


namespace no_values_of_g_g_x_eq_one_l284_284114

-- Define the function g and its properties based on the conditions
variable (g : ℝ → ℝ)
variable (h₁ : g (-4) = 1)
variable (h₂ : g (0) = 1)
variable (h₃ : g (4) = 3)
variable (h₄ : ∀ x, -4 ≤ x ∧ x ≤ 4 → g x ≥ 1)

-- Define the theorem to prove the number of values of x such that g(g(x)) = 1 is zero
theorem no_values_of_g_g_x_eq_one : ∃ n : ℕ, n = 0 ∧ (∀ x, -4 ≤ x ∧ x ≤ 4 → g (g x) = 1 → false) :=
by
  sorry -- proof to be provided later

end no_values_of_g_g_x_eq_one_l284_284114


namespace inequality_condition_l284_284204

noncomputable def f (a b x : ℝ) : ℝ := Real.exp x - (1 + a) * x - b

theorem inequality_condition (a b: ℝ) (h : ∀ x : ℝ, f a b x ≥ 0) : (b * (a + 1)) / 2 < 3 / 4 := 
sorry

end inequality_condition_l284_284204


namespace original_average_weight_l284_284829

theorem original_average_weight (W : ℝ) (h : (7 * W + 110 + 60) / 9 = 113) : W = 121 :=
by
  sorry

end original_average_weight_l284_284829


namespace number_of_arrangements_word_l284_284463

noncomputable def factorial (n : Nat) : Nat := 
  if n = 0 then 1 else n * factorial (n - 1)

theorem number_of_arrangements_word (letters : List Char) (n : Nat) (r1 r2 r3 : Nat) 
  (h1 : letters = ['M', 'A', 'T', 'H', 'E', 'M', 'A', 'T', 'I', 'C', 'S'])
  (h2 : 2 = r1) (h3 : 2 = r2) (h4 : 2 = r3) :
  n = 11 → 
  factorial n / (factorial r1 * factorial r2 * factorial r3) = 4989600 := 
by
  sorry

end number_of_arrangements_word_l284_284463


namespace asymptotes_of_hyperbola_l284_284267

theorem asymptotes_of_hyperbola :
  (∀ x y : ℝ, (x^2 / 16 - y^2 / 25 = 1) →
    (y = (5 / 4) * x ∨ y = -(5 / 4) * x)) :=
by
  intro x y h
  sorry

end asymptotes_of_hyperbola_l284_284267


namespace base_3_is_most_economical_l284_284222

theorem base_3_is_most_economical (m d : ℕ) (h : d ≥ 1) (h_m_div_d : m % d = 0) :
  3^(m / 3) ≥ d^(m / d) :=
sorry

end base_3_is_most_economical_l284_284222


namespace arithmetic_sequence_common_difference_l284_284924

theorem arithmetic_sequence_common_difference (a : ℕ → ℝ) (h : ∀ n, 4 * a (n + 1) - 4 * a n - 9 = 0) :
  ∃ d, (∀ n, a (n + 1) - a n = d) ∧ d = 9 / 4 := 
  sorry

end arithmetic_sequence_common_difference_l284_284924


namespace arithmetic_sequence_inequality_l284_284481

theorem arithmetic_sequence_inequality (a : ℕ → ℝ) (h : ∀ n : ℕ, n ≥ 2 → 2 * a n = a (n - 1) + a (n + 1)) :
  a 2 * a 4 ≤ a 3 ^ 2 :=
sorry

end arithmetic_sequence_inequality_l284_284481


namespace opposite_number_l284_284826

variable (a : ℝ)

theorem opposite_number (a : ℝ) : -(3 * a - 2) = -3 * a + 2 := by
  sorry

end opposite_number_l284_284826


namespace taxi_fare_charge_l284_284086

theorem taxi_fare_charge :
  let initial_fee := 2.25
  let total_distance := 3.6
  let total_charge := 4.95
  let increments := total_distance / (2 / 5)
  let distance_charge := total_charge - initial_fee
  let charge_per_increment := distance_charge / increments
  charge_per_increment = 0.30 :=
by
  sorry

end taxi_fare_charge_l284_284086


namespace total_bones_in_graveyard_l284_284776

def total_skeletons : ℕ := 20

def adult_women : ℕ := total_skeletons / 2
def adult_men : ℕ := (total_skeletons - adult_women) / 2
def children : ℕ := (total_skeletons - adult_women) / 2

def bones_adult_woman : ℕ := 20
def bones_adult_man : ℕ := bones_adult_woman + 5
def bones_child : ℕ := bones_adult_woman / 2

def bones_graveyard : ℕ :=
  (adult_women * bones_adult_woman) +
  (adult_men * bones_adult_man) +
  (children * bones_child)

theorem total_bones_in_graveyard :
  bones_graveyard = 375 :=
sorry

end total_bones_in_graveyard_l284_284776


namespace max_ratio_of_sequence_l284_284051

theorem max_ratio_of_sequence (S : ℕ → ℝ) (a : ℕ → ℝ)
  (hS : ∀ n : ℕ, S n = (n + 2) / 3 * a n) :
  ∃ n : ℕ, ∀ m : ℕ, (n = 2 → m ≠ 1) → (a n / a (n - 1)) ≤ (a m / a (m - 1)) :=
by
  sorry

end max_ratio_of_sequence_l284_284051


namespace two_digit_primes_ending_in_3_l284_284754

theorem two_digit_primes_ending_in_3 : ∃ n : ℕ, n = 7 ∧ (∀ x, 10 ≤ x ∧ x < 100 ∧ x % 10 = 3 → Prime x → (x = 13 ∨ x = 23 ∨ x = 43 ∨ x = 53 ∨ x = 73 ∨ x = 83)) :=
by {
  use 7,
  split,
  -- proof part is omitted for this example
  sorry,
}

end two_digit_primes_ending_in_3_l284_284754


namespace john_school_year_hours_l284_284237

noncomputable def requiredHoursPerWeek (summerHoursPerWeek : ℕ) (summerWeeks : ℕ) 
                                       (summerEarnings : ℕ) (schoolWeeks : ℕ) 
                                       (schoolEarnings : ℕ) : ℕ :=
    schoolEarnings * summerHoursPerWeek * summerWeeks / (summerEarnings * schoolWeeks)

theorem john_school_year_hours :
  ∀ (summerHoursPerWeek summerWeeks summerEarnings schoolWeeks schoolEarnings : ℕ),
    summerHoursPerWeek = 40 →
    summerWeeks = 10 →
    summerEarnings = 4000 →
    schoolWeeks = 50 →
    schoolEarnings = 4000 →
    requiredHoursPerWeek summerHoursPerWeek summerWeeks summerEarnings schoolWeeks schoolEarnings = 8 :=
by
  intros
  sorry

end john_school_year_hours_l284_284237


namespace simplify_expr_correct_l284_284258

-- Define the expression
def simplify_expr (z : ℝ) : ℝ := (3 - 5 * z^2) - (5 + 7 * z^2)

-- Prove the simplified form
theorem simplify_expr_correct (z : ℝ) : simplify_expr z = -2 - 12 * z^2 := by
  sorry

end simplify_expr_correct_l284_284258


namespace marks_in_mathematics_l284_284351

-- Definitions for the given conditions in the problem
def marks_in_english : ℝ := 86
def marks_in_physics : ℝ := 82
def marks_in_chemistry : ℝ := 87
def marks_in_biology : ℝ := 81
def average_marks : ℝ := 85
def number_of_subjects : ℕ := 5

-- Defining the total marks based on the provided conditions
def total_marks : ℝ := average_marks * number_of_subjects

-- Proving that the marks in mathematics are 89
theorem marks_in_mathematics : total_marks - (marks_in_english + marks_in_physics + marks_in_chemistry + marks_in_biology) = 89 :=
by
  sorry

end marks_in_mathematics_l284_284351


namespace describes_random_event_proof_l284_284133

def describes_random_event (phrase : String) : Prop :=
  match phrase with
  | "Winter turns into spring"  => False
  | "Fishing for the moon in the water" => False
  | "Seeking fish on a tree" => False
  | "Meeting unexpectedly" => True
  | _ => False

theorem describes_random_event_proof : describes_random_event "Meeting unexpectedly" = True :=
by
  sorry

end describes_random_event_proof_l284_284133


namespace sum_of_squares_and_product_l284_284992

theorem sum_of_squares_and_product
  (x y : ℕ) (h1 : x^2 + y^2 = 181) (h2 : x * y = 90) : x + y = 19 := by
  sorry

end sum_of_squares_and_product_l284_284992


namespace tan_105_degree_l284_284591

theorem tan_105_degree : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  have tan_add : ∀ (a b : ℝ), Real.tan (a + b) = (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b) :=
    sorry
  have tan_45 := Real.tan (45 * Real.pi / 180)
  have tan_60 := Real.tan (60 * Real.pi / 180)
  have tan_45_value : tan_45 = 1 := sorry
  have tan_60_value : tan_60 = Real.sqrt 3 := sorry
  sorry

end tan_105_degree_l284_284591


namespace renovation_project_total_l284_284151

def sand : ℝ := 0.17
def dirt : ℝ := 0.33
def cement : ℝ := 0.17

theorem renovation_project_total : sand + dirt + cement = 0.67 := 
by
  sorry

end renovation_project_total_l284_284151


namespace find_percentage_l284_284315

theorem find_percentage (P : ℝ) :
  (P / 100) * 1280 = ((0.20 * 650) + 190) ↔ P = 25 :=
by
  sorry

end find_percentage_l284_284315


namespace smallest_odd_number_with_five_prime_factors_l284_284837

theorem smallest_odd_number_with_five_prime_factors :
  ∃ (n : ℕ), n = 3 * 5 * 7 * 11 * 13 ∧
  n % 2 ≠ 0 ∧
  ∃ (p1 p2 p3 p4 p5 : ℕ), 
    (prime p1) ∧ 
    (prime p2) ∧ 
    (prime p3) ∧ 
    (prime p4) ∧ 
    (prime p5) ∧ 
    p1 ≠ p2 ∧ 
    p2 ≠ p3 ∧ 
    p3 ≠ p4 ∧ 
    p4 ≠ p5 ∧ 
    p1 = 3 ∧ 
    p2 = 5 ∧ 
    p3 = 7 ∧ 
    p4 = 11 ∧ 
    p5 = 13 ∧ 
    n = p1 * p2 * p3 * p4 * p5 :=
sorry

end smallest_odd_number_with_five_prime_factors_l284_284837


namespace find_weight_of_sausages_l284_284081

variable (packages : ℕ) (cost_per_pound : ℕ) (total_cost : ℕ) (total_weight : ℕ) (weight_per_package : ℕ)

-- Defining the given conditions
def jake_buys_packages (packages : ℕ) : Prop := packages = 3
def cost_of_sausages (cost_per_pound : ℕ) : Prop := cost_per_pound = 4
def amount_paid (total_cost : ℕ) : Prop := total_cost = 24

-- Derived condition to find total weight
def total_weight_of_sausages (total_cost : ℕ) (cost_per_pound : ℕ) : ℕ := total_cost / cost_per_pound

-- Derived condition to find weight per package
def weight_of_each_package (total_weight : ℕ) (packages : ℕ) : ℕ := total_weight / packages

-- The theorem statement
theorem find_weight_of_sausages
  (h1 : jake_buys_packages packages)
  (h2 : cost_of_sausages cost_per_pound)
  (h3 : amount_paid total_cost) :
  weight_of_each_package (total_weight_of_sausages total_cost cost_per_pound) packages = 2 :=
by
  sorry  -- Proof placeholder

end find_weight_of_sausages_l284_284081


namespace initial_coins_l284_284127

-- Define the condition for the initial number of coins
variable (x : Nat) -- x represents the initial number of coins

-- The main statement theorem that needs proof
theorem initial_coins (h : x + 8 = 29) : x = 21 := 
by { sorry } -- placeholder for the proof

end initial_coins_l284_284127


namespace time_with_walkway_l284_284573

theorem time_with_walkway (v w : ℝ) (t : ℕ) :
  (80 = 120 * (v - w)) → 
  (80 = 60 * v) → 
  t = 80 / (v + w) → 
  t = 40 :=
by
  sorry

end time_with_walkway_l284_284573


namespace same_function_C_l284_284132

theorem same_function_C (x : ℝ) (hx : x ≠ 0) : (x^0 = 1) ∧ ((1 / x^0) = 1) :=
by
  -- Definition for domain exclusion
  have h1 : x ^ 0 = 1 := by 
    sorry -- proof skipped
  have h2 : 1 / x ^ 0 = 1 := by 
    sorry -- proof skipped
  exact ⟨h1, h2⟩

end same_function_C_l284_284132


namespace length_of_segment_XY_l284_284430

noncomputable def rectangle_length (A B C D : ℝ) (BX DY : ℝ) : ℝ :=
  2 * BX + DY

theorem length_of_segment_XY (A B C D : ℝ) (BX DY : ℝ) (h1 : C = 2 * B) (h2 : BX = 4) (h3 : DY = 10) :
  rectangle_length A B C D BX DY = 13 :=
by
  rw [rectangle_length, h2, h3]
  sorry

end length_of_segment_XY_l284_284430


namespace calculate_seedlings_l284_284425

-- Define conditions
def condition_1 (x n : ℕ) : Prop :=
  x = 5 * n + 6

def condition_2 (x m : ℕ) : Prop :=
  x = 6 * m - 9

-- Define the main theorem based on these conditions
theorem calculate_seedlings (x : ℕ) : (∃ n, condition_1 x n) ∧ (∃ m, condition_2 x m) → x = 81 :=
by {
  sorry
}

end calculate_seedlings_l284_284425


namespace tan_105_eq_neg_2_sub_sqrt_3_l284_284620

-- Definitions
def angle105 : ℝ := 105 * (Math.pi / 180)
def angle45 : ℝ := 45 * (Math.pi / 180)
def angle60 : ℝ := 60 * (Math.pi / 180)

-- Theorem
theorem tan_105_eq_neg_2_sub_sqrt_3 :
  Real.tan angle105 = -2 - Real.sqrt 3 := by
  sorry

end tan_105_eq_neg_2_sub_sqrt_3_l284_284620


namespace problem_statement_l284_284418

variable (a b c : ℝ)

-- Conditions given in the problem
axiom h1 : (a * c) / (a + b) + (b * a) / (b + c) + (c * b) / (c + a) = -24
axiom h2 : (b * c) / (a + b) + (c * a) / (b + c) + (a * b) / (c + a) = 8

-- The Lean statement for the proof problem
theorem problem_statement (a b c : ℝ) (h1 : (a * c) / (a + b) + (b * a) / (b + c) + (c * b) / (c + a) = -24)
    (h2 : (b * c) / (a + b) + (c * a) / (b + c) + (a * b) / (c + a) = 8) :
    (b / (a + b) + c / (b + c) + a / (c + a)) = 19 / 2 :=
sorry

end problem_statement_l284_284418


namespace newspaper_spending_over_8_weeks_l284_284207

theorem newspaper_spending_over_8_weeks :
  (3 * 0.50 + 2.00) * 8 = 28 := by
  sorry

end newspaper_spending_over_8_weeks_l284_284207


namespace cost_of_pencils_l284_284970

open Nat

theorem cost_of_pencils (P : ℕ) : 
  (H : 20 * P + 80 * 3 = 360) → 
  P = 6 :=
by 
  sorry

end cost_of_pencils_l284_284970


namespace factorization_correct_l284_284357

theorem factorization_correct (x : ℤ) :
  (3 * (x + 3) * (x + 4) * (x + 7) * (x + 8) - 2 * x^2) =
  ((3 * x^2 + 35 * x + 72) * (x + 3) * (x + 6)) :=
by sorry

end factorization_correct_l284_284357


namespace cubic_product_of_roots_l284_284358

theorem cubic_product_of_roots (k : ℝ) :
  (∃ a b c : ℝ, a + b + c = 2 ∧ ab + bc + ca = 1 ∧ abc = -k ∧ -k = (max (max a b) c - min (min a b) c)^2) ↔ k = -2 :=
by
  sorry

end cubic_product_of_roots_l284_284358


namespace cos_seven_pi_over_six_l284_284910

theorem cos_seven_pi_over_six : Real.cos (7 * Real.pi / 6) = -Real.sqrt 3 / 2 :=
by
  sorry

end cos_seven_pi_over_six_l284_284910


namespace servings_correct_l284_284121

-- Define the pieces of popcorn in a serving
def pieces_per_serving := 30

-- Define the pieces of popcorn Jared can eat
def jared_pieces := 90

-- Define the pieces of popcorn each friend can eat
def friend_pieces := 60

-- Define the number of friends
def friends := 3

-- Calculate total pieces eaten by friends
def total_friend_pieces := friends * friend_pieces

-- Calculate total pieces eaten by everyone
def total_pieces := jared_pieces + total_friend_pieces

-- Calculate the number of servings needed
def servings_needed := total_pieces / pieces_per_serving

theorem servings_correct : servings_needed = 9 :=
by
  sorry

end servings_correct_l284_284121


namespace tan_105_eq_neg2_sub_sqrt3_l284_284643

theorem tan_105_eq_neg2_sub_sqrt3 :
  Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_eq_neg2_sub_sqrt3_l284_284643


namespace tan_add_tan_105_eq_l284_284663

noncomputable def tan : ℝ → ℝ := sorry -- Use the built-in library later for actual implementation

-- Given conditions
def tan_45_eq : tan 45 = 1 := by sorry
def tan_60_eq : tan 60 = Real.sqrt 3 := by sorry

-- Angle addition formula for tangent
theorem tan_add (a b : ℝ) :
  tan (a + b) = (tan a + tan b) / (1 - tan a * tan b) := by sorry

-- Main theorem to prove
theorem tan_105_eq :
  tan 105 = -2 - Real.sqrt 3 := by sorry

end tan_add_tan_105_eq_l284_284663


namespace unique_infinite_sequence_l284_284193

-- Defining conditions for the infinite sequence of negative integers
variable (a : ℕ → ℤ)
  
-- Condition 1: Elements in sequence are negative integers
def sequence_negative : Prop :=
  ∀ n, a n < 0 

-- Condition 2: For every positive integer n, the first n elements taken modulo n have n distinct remainders
def distinct_mod_remainders (n : ℕ) : Prop :=
  ∀ i j, i < n → j < n → i ≠ j → (a i % n ≠ a j % n) 

-- The main theorem statement
theorem unique_infinite_sequence (a : ℕ → ℤ) 
  (h1 : sequence_negative a) 
  (h2 : ∀ n, distinct_mod_remainders a n) :
  ∀ k : ℤ, ∃! n, a n = k :=
sorry

end unique_infinite_sequence_l284_284193


namespace Jerry_average_speed_l284_284236

variable (J : ℝ) -- Jerry's average speed in miles per hour
variable (C : ℝ) -- Carla's average speed in miles per hour
variable (T_J : ℝ) -- Time Jerry has been driving in hours
variable (T_C : ℝ) -- Time Carla has been driving in hours
variable (D : ℝ) -- Distance covered in miles

-- Given conditions
axiom Carla_speed : C = 35
axiom Carla_time : T_C = 3
axiom Jerry_time : T_J = T_C + 0.5

-- Distance covered by Carla in T_C hours at speed C
axiom Carla_distance : D = C * T_C

-- Distance covered by Jerry in T_J hours at speed J
axiom Jerry_distance : D = J * T_J

-- The goal to prove
theorem Jerry_average_speed : J = 30 :=
by
  sorry

end Jerry_average_speed_l284_284236


namespace intersection_eq_M_l284_284965

-- Define the sets M and N according to the given conditions
def M : Set ℝ := {x : ℝ | x^2 - x < 0}
def N : Set ℝ := {x : ℝ | |x| < 2}

-- The 'theorem' statement to prove M ∩ N = M
theorem intersection_eq_M : M ∩ N = M :=
  sorry

end intersection_eq_M_l284_284965


namespace roots_of_quadratic_l284_284094

open Real

theorem roots_of_quadratic (r s : ℝ) (h1 : r + s = 2 * sqrt 3) (h2 : r * s = 2) :
  r^6 + s^6 = 3104 :=
sorry

end roots_of_quadratic_l284_284094


namespace regular_polygon_properties_l284_284727

theorem regular_polygon_properties
  (n : ℕ)
  (h1 : (n - 2) * 180 = 3 * 360 + 180)
  (h2 : n > 2) :
  n = 9 ∧ (n - 2) * 180 / n = 140 := by
  sorry

end regular_polygon_properties_l284_284727


namespace kimberly_bought_skittles_l284_284392

-- Conditions
def initial_skittles : ℕ := 5
def total_skittles : ℕ := 12

-- Prove
theorem kimberly_bought_skittles : ∃ bought_skittles : ℕ, (total_skittles = initial_skittles + bought_skittles) ∧ bought_skittles = 7 :=
by
  sorry

end kimberly_bought_skittles_l284_284392


namespace number_of_two_digit_primes_with_ones_digit_three_l284_284739

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def has_ones_digit_three (n : ℕ) : Prop :=
  n % 10 = 3

def is_prime (n : ℕ) : Prop :=
  Nat.Prime n

theorem number_of_two_digit_primes_with_ones_digit_three :
  ∃! s : Finset ℕ, (∀ n ∈ s, is_two_digit n ∧ has_ones_digit_three n ∧ is_prime n) ∧ s.card = 6 :=
by
  sorry

end number_of_two_digit_primes_with_ones_digit_three_l284_284739


namespace speed_of_man_in_still_water_l284_284323

theorem speed_of_man_in_still_water (v_m v_s : ℝ) (h1 : v_m + v_s = 6.2) (h2 : v_m - v_s = 6) : v_m = 6.1 :=
by
  sorry

end speed_of_man_in_still_water_l284_284323


namespace simplify_and_find_ratio_l284_284059

theorem simplify_and_find_ratio (m : ℤ) : 
  let expr := (6 * m + 18) / 6 
  let c := 1
  let d := 3
  (c / d : ℚ) = 1 / 3 := 
by
  -- Conditions and transformations are stated here
  -- (6 * m + 18) / 6 can be simplified step-by-step
  sorry

end simplify_and_find_ratio_l284_284059


namespace mice_population_l284_284160

theorem mice_population :
  ∃ (mice_initial : ℕ) (pups_per_mouse : ℕ) (survival_rate_first_gen : ℕ → ℕ) 
    (survival_rate_second_gen : ℕ → ℕ) (num_dead_first_gen : ℕ) (pups_eaten_per_adult : ℕ)
    (total_mice : ℕ),
    mice_initial = 8 ∧ pups_per_mouse = 7 ∧
    (∀ n, survival_rate_first_gen n = (n * 80) / 100) ∧
    (∀ n, survival_rate_second_gen n = (n * 60) / 100) ∧
    num_dead_first_gen = 2 ∧ pups_eaten_per_adult = 3 ∧
    total_mice = mice_initial + (survival_rate_first_gen (mice_initial * pups_per_mouse)) - num_dead_first_gen + (survival_rate_second_gen ((mice_initial + (survival_rate_first_gen (mice_initial * pups_per_mouse))) * pups_per_mouse)) - ((mice_initial - num_dead_first_gen) * pups_eaten_per_adult) :=
  sorry

end mice_population_l284_284160


namespace largest_power_of_5_dividing_sum_l284_284903

theorem largest_power_of_5_dividing_sum :
  let s := 48! + 50! + 51!
  (∃ n : ℕ, (5^n ∣ s) ∧ ( ∀ m : ℕ, (5^m ∣ s → m ≤ n) )) :=
  ∃ n : ℕ, (5 ^ n ∣ (48! + 50! + 51!)) ∧ (∀ m : ℕ, 5 ^ m ∣ (48! + 50! + 51!) → m ≤ n) :=
        ∃ n : ℕ, n = 10 := sorry

end largest_power_of_5_dividing_sum_l284_284903


namespace gcd_of_ratios_l284_284378

noncomputable def gcd_of_two_ratios (A B : ℕ) : ℕ :=
  if h : A % B = 0 then B else gcd B (A % B)

theorem gcd_of_ratios (A B : ℕ) (k : ℕ) (h1 : Nat.lcm A B = 180) (h2 : A = 2 * k) (h3 : B = 3 * k) : gcd_of_two_ratios A B = 30 :=
  by
    sorry

end gcd_of_ratios_l284_284378


namespace sum_of_nonneg_numbers_ineq_l284_284428

theorem sum_of_nonneg_numbers_ineq
  (a b c d : ℝ)
  (h_nonneg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0)
  (h_sum : a + b + c + d = 4) :
  (a * b + c * d) * (a * c + b * d) * (a * d + b * c) ≤ 8 := sorry

end sum_of_nonneg_numbers_ineq_l284_284428


namespace tan_105_l284_284684

-- Defining the necessary known values and functions
def tan_addition (a b : ℝ) : ℝ := (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b)
def tan_60 : ℝ := Real.sqrt 3
def tan_45 : ℝ := 1

-- Proof goal in Lean 4
theorem tan_105 : Real.tan (105 * Real.pi / 180) = 2 + Real.sqrt 3 :=
by
  let tan_105 := tan_addition (60 * Real.pi / 180) (45 * Real.pi / 180)
  have h1 : Real.tan (60 * Real.pi / 180) = tan_60 := by sorry
  have h2 : Real.tan (45 * Real.pi / 180) = tan_45 := by sorry
  show Real.tan (105 * Real.pi / 180) = 2 + Real.sqrt 3 from sorry

end tan_105_l284_284684


namespace space_taken_by_files_l284_284249

-- Definitions/Conditions
def total_space : ℕ := 28
def space_left : ℕ := 2

-- Statement of the theorem
theorem space_taken_by_files : total_space - space_left = 26 := by sorry

end space_taken_by_files_l284_284249


namespace ratio_of_areas_l284_284876

theorem ratio_of_areas (s : ℝ) : (s^2) / ((3 * s)^2) = 1 / 9 := 
by
  sorry

end ratio_of_areas_l284_284876


namespace math_problem_l284_284930

theorem math_problem 
  (a b c : ℕ) 
  (h_primea : Nat.Prime a)
  (h_posa : 0 < a)
  (h_posb : 0 < b)
  (h_posc : 0 < c)
  (h_eq : a^2 + b^2 = c^2) :
  (b % 2 ≠ c % 2) ∧ (∃ k, 2 * (a + b + 1) = k^2) := 
sorry

end math_problem_l284_284930


namespace tan_105_eq_neg2_sub_sqrt3_l284_284666

theorem tan_105_eq_neg2_sub_sqrt3 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 := by
  sorry

end tan_105_eq_neg2_sub_sqrt3_l284_284666


namespace team_C_games_played_l284_284077

variable (x : ℕ)
variable (winC : ℕ := 5 * x / 7)
variable (loseC : ℕ := 2 * x / 7)
variable (winD : ℕ := 2 * x / 3)
variable (loseD : ℕ := x / 3)

theorem team_C_games_played :
  winD = winC - 5 →
  loseD = loseC - 5 →
  x = 105 := by
  sorry

end team_C_games_played_l284_284077


namespace largest_n_l284_284031

-- Define the condition that n, x, y, z are positive integers
def conditions (n x y z : ℕ) := (0 < x) ∧ (0 < y) ∧ (0 < z) ∧ (0 < n) 

-- Formulate the main theorem
theorem largest_n (x y z : ℕ) : 
  conditions 8 x y z →
  8^2 = x^2 + y^2 + z^2 + 2 * x * y + 2 * y * z + 2 * z * x + 4 * x + 4 * y + 4 * z - 10 :=
by 
  sorry

end largest_n_l284_284031


namespace smallest_odd_number_with_five_prime_factors_l284_284847

theorem smallest_odd_number_with_five_prime_factors : 
  ∃ n : ℕ, n = 15015 ∧ (∀ (p ∈ {3, 5, 7, 11, 13}), prime p) ∧ odd n :=
by
  sorry

end smallest_odd_number_with_five_prime_factors_l284_284847


namespace tan_105_degree_l284_284638

theorem tan_105_degree : Real.tan (Real.pi * 105 / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_degree_l284_284638


namespace tan_105_l284_284596

theorem tan_105 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  -- Definitions
  let tan45 := Real.tan (45 * Real.pi / 180)
  let tan60 := Real.tan (60 * Real.pi / 180)
  have h1 : tan45 = 1 := sorry
  have h2 : tan60 = Real.sqrt 3 := sorry
  have h3 : tan45 + tan60 = 1 + Real.sqrt 3 := sorry
  have h4 : 1 - tan45 * tan60 = 1 - 1 * Real.sqrt 3 := sorry
  
  -- Use tangent addition formula
  have tan_addition : Real.tan (105 * Real.pi / 180) = (1 + Real.sqrt 3) / (1 - Real.sqrt 3) := sorry
  
  -- Simplify and rationalize to prove the final result
  have tan_105_eq : (1 + Real.sqrt 3) / (1 - Real.sqrt 3) = -2 - Real.sqrt 3 := sorry
  
  exact tan_105_eq

end tan_105_l284_284596


namespace abc_equal_l284_284196

theorem abc_equal (a b c : ℝ)
  (h1 : ∀ x : ℝ, a * x^2 + b * x + c ≥ b * x^2 + c * x + a)
  (h2 : ∀ x : ℝ, b * x^2 + c * x + a ≥ c * x^2 + a * x + b) : a = b ∧ b = c :=
by
  sorry

end abc_equal_l284_284196


namespace reggie_games_lost_l284_284409

-- Given conditions:
def initial_marbles : ℕ := 100
def marbles_per_game : ℕ := 10
def games_played : ℕ := 9
def marbles_after_games : ℕ := 90

-- The statement to prove:
theorem reggie_games_lost : (initial_marbles - marbles_after_games) / marbles_per_game = 1 := 
sorry

end reggie_games_lost_l284_284409


namespace min_expression_l284_284520

theorem min_expression (a b c d e f : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f)
  (h_sum : a + b + c + d + e + f = 10) : 
  (1 / a + 9 / b + 16 / c + 25 / d + 36 / e + 49 / f) ≥ 67.6 :=
sorry

end min_expression_l284_284520


namespace Mike_and_Sarah_missed_days_l284_284129

theorem Mike_and_Sarah_missed_days :
  ∀ (V M S : ℕ), V + M + S = 17 → V + M = 14 → V = 5 → M + S = 12 :=
by
  intros V M S h1 h2 h3
  sorry

end Mike_and_Sarah_missed_days_l284_284129


namespace second_number_value_l284_284140

def first_number := ℚ
def second_number := ℚ

variables (x y : ℚ)

/-- Given conditions: 
      (1) \( \frac{1}{5}x = \frac{5}{8}y \)
      (2) \( x + 35 = 4y \)
    Prove that \( y = 40 \) 
-/
theorem second_number_value (h1 : (1/5 : ℚ) * x = (5/8 : ℚ) * y) (h2 : x + 35 = 4 * y) : 
  y = 40 :=
sorry

end second_number_value_l284_284140


namespace tan_add_tan_105_eq_l284_284660

noncomputable def tan : ℝ → ℝ := sorry -- Use the built-in library later for actual implementation

-- Given conditions
def tan_45_eq : tan 45 = 1 := by sorry
def tan_60_eq : tan 60 = Real.sqrt 3 := by sorry

-- Angle addition formula for tangent
theorem tan_add (a b : ℝ) :
  tan (a + b) = (tan a + tan b) / (1 - tan a * tan b) := by sorry

-- Main theorem to prove
theorem tan_105_eq :
  tan 105 = -2 - Real.sqrt 3 := by sorry

end tan_add_tan_105_eq_l284_284660


namespace distance_blown_by_storm_l284_284252

-- Definitions based on conditions
def speed : ℤ := 30
def time_travelled : ℤ := 20
def distance_travelled := speed * time_travelled
def total_distance := 2 * distance_travelled
def fractional_distance_left := total_distance / 3

-- Final statement to prove
theorem distance_blown_by_storm : distance_travelled - fractional_distance_left = 200 := by
  sorry

end distance_blown_by_storm_l284_284252


namespace total_tickets_sold_l284_284112

theorem total_tickets_sold
    (n₄₅ : ℕ) (n₆₀ : ℕ) (total_sales : ℝ) 
    (price₄₅ price₆₀ : ℝ)
    (h₁ : n₄₅ = 205)
    (h₂ : price₄₅ = 4.5)
    (h₃ : total_sales = 1972.5)
    (h₄ : price₆₀ = 6.0)
    (h₅ : total_sales = n₄₅ * price₄₅ + n₆₀ * price₆₀) :
    n₄₅ + n₆₀ = 380 := 
by
  sorry

end total_tickets_sold_l284_284112


namespace tan_add_tan_105_eq_l284_284658

noncomputable def tan : ℝ → ℝ := sorry -- Use the built-in library later for actual implementation

-- Given conditions
def tan_45_eq : tan 45 = 1 := by sorry
def tan_60_eq : tan 60 = Real.sqrt 3 := by sorry

-- Angle addition formula for tangent
theorem tan_add (a b : ℝ) :
  tan (a + b) = (tan a + tan b) / (1 - tan a * tan b) := by sorry

-- Main theorem to prove
theorem tan_105_eq :
  tan 105 = -2 - Real.sqrt 3 := by sorry

end tan_add_tan_105_eq_l284_284658


namespace tan_seventeen_pi_over_four_l284_284467

theorem tan_seventeen_pi_over_four : Real.tan (17 * Real.pi / 4) = 1 := by
  sorry

end tan_seventeen_pi_over_four_l284_284467


namespace base_7_to_10_of_23456_l284_284288

theorem base_7_to_10_of_23456 : 
  (2 * 7 ^ 4 + 3 * 7 ^ 3 + 4 * 7 ^ 2 + 5 * 7 ^ 1 + 6 * 7 ^ 0) = 6068 :=
by sorry

end base_7_to_10_of_23456_l284_284288


namespace quadratic_has_distinct_real_roots_l284_284179

theorem quadratic_has_distinct_real_roots {c : ℝ} (h : c < 1 / 4) :
  ∃ (r1 r2 : ℝ), r1 ≠ r2 ∧ (∃ f, f = (λ x : ℝ, x^2 + 2 * x + 4 * c)) ∧ f r1 = 0 ∧ f r2 = 0 :=
by
  sorry

end quadratic_has_distinct_real_roots_l284_284179


namespace fertilizer_prices_l284_284505

variables (x y : ℝ)

theorem fertilizer_prices :
  (x = y + 100) ∧ (2 * x + y = 1700) → (x = 600 ∧ y = 500) :=
by
  intros h
  cases h with h1 h2
  have h3 : y = 500 := by sorry
  have h4 : x = y + 100 := h1
  rw h3 at h4
  have h5 : x = 600 := by sorry
  exact ⟨h5, h3⟩

end fertilizer_prices_l284_284505


namespace new_parabola_after_shift_l284_284822

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := x^2 + 1

-- Define the transformation functions for shifting the parabola
def shift_left (x : ℝ) (shift : ℝ) : ℝ := x + shift
def shift_down (y : ℝ) (shift : ℝ) : ℝ := y - shift

-- Prove the transformation yields the correct new parabola equation
theorem new_parabola_after_shift : 
  (∀ x : ℝ, (shift_down (original_parabola (shift_left x 2)) 3) = (x + 2)^2 - 2) :=
by
  sorry

end new_parabola_after_shift_l284_284822


namespace min_employees_needed_l284_284880

-- Define the conditions
variable (W A : Finset ℕ)
variable (n_W n_A n_WA : ℕ)

-- Assume the given condition values
def sizeW := 95
def sizeA := 80
def sizeWA := 30

-- Define the proof problem
theorem min_employees_needed :
  (sizeW + sizeA - sizeWA) = 145 :=
by sorry

end min_employees_needed_l284_284880


namespace neg_of_exists_a_l284_284831

theorem neg_of_exists_a (a : ℝ) : ¬ (∃ a : ℝ, a^2 + 1 < 2 * a) :=
by
  sorry

end neg_of_exists_a_l284_284831


namespace combined_total_value_of_items_l284_284518

theorem combined_total_value_of_items :
  let V1 := 87.50 / 0.07
  let V2 := 144 / 0.12
  let V3 := 50 / 0.05
  let total1 := 1000 + V1
  let total2 := 1000 + V2
  let total3 := 1000 + V3
  total1 + total2 + total3 = 6450 := 
by
  sorry

end combined_total_value_of_items_l284_284518


namespace haley_candy_l284_284473

theorem haley_candy (X : ℕ) (h : X - 17 + 19 = 35) : X = 33 :=
by
  sorry

end haley_candy_l284_284473


namespace people_stools_chairs_l284_284501

def numberOfPeopleStoolsAndChairs (x y z : ℕ) : Prop :=
  2 * x + 3 * y + 4 * z = 32 ∧
  x > y ∧
  x > z ∧
  x < y + z

theorem people_stools_chairs :
  ∃ (x y z : ℕ), numberOfPeopleStoolsAndChairs x y z ∧ x = 5 ∧ y = 2 ∧ z = 4 :=
by
  sorry

end people_stools_chairs_l284_284501


namespace simplify_expr1_simplify_expr2_l284_284803

-- First expression
theorem simplify_expr1 (a b : ℝ) : a * (a - b) - (a + b) * (a - 2 * b) = 2 * b ^ 2 :=
by
  sorry

-- Second expression
theorem simplify_expr2 (x : ℝ) : 
  ( ( (4 * x - 9) / (3 - x) - x + 3 ) / ( (x ^ 2 - 4) / (x - 3) ) ) = - (x / (x + 2)) :=
by
  sorry

end simplify_expr1_simplify_expr2_l284_284803


namespace count_two_digit_integers_congruent_to_2_mod_4_l284_284068

theorem count_two_digit_integers_congruent_to_2_mod_4 : 
  let nums := {x : ℕ | 10 ≤ x ∧ x ≤ 99 ∧ x % 4 = 2}
  in nums.card = 23 :=
by
  sorry

end count_two_digit_integers_congruent_to_2_mod_4_l284_284068


namespace find_N_aN_bN_cN_dN_eN_l284_284246

theorem find_N_aN_bN_cN_dN_eN:
  ∃ (a b c d e : ℝ) (N : ℝ),
    (a > 0) ∧ (b > 0) ∧ (c > 0) ∧ (d > 0) ∧ (e > 0) ∧
    (a^2 + b^2 + c^2 + d^2 + e^2 = 1000) ∧
    (N = c * (a + 3 * b + 4 * d + 6 * e)) ∧
    (N + a + b + c + d + e = 150 + 250 * Real.sqrt 62 + 10 * Real.sqrt 50) := by
  sorry

end find_N_aN_bN_cN_dN_eN_l284_284246


namespace equal_roots_of_quadratic_l284_284944

theorem equal_roots_of_quadratic (k : ℝ) : (1 - 8 * k = 0) → (k = 1/8) :=
by
  intro h
  sorry

end equal_roots_of_quadratic_l284_284944


namespace sum_of_money_l284_284316

theorem sum_of_money (A B C : ℝ) (hB : B = 0.65 * A) (hC : C = 0.40 * A) (hC_value : C = 32) :
  A + B + C = 164 :=
by
  sorry

end sum_of_money_l284_284316


namespace presidency_meeting_ways_l284_284318

theorem presidency_meeting_ways :
  let total_schools := 4
  let members_per_school := 4
  let host_school_choices := total_schools
  let choose_3_from_4 := Nat.choose 4 3
  let choose_1_from_4 := Nat.choose 4 1
  let ways_per_host := choose_3_from_4 * choose_1_from_4 ^ 3
  let total_ways := host_school_choices * ways_per_host
  total_ways = 1024 := by
  sorry

end presidency_meeting_ways_l284_284318


namespace two_primes_equal_l284_284967

theorem two_primes_equal
  (a b c : ℕ)
  (p q r : ℕ)
  (hp : p = b^c + a ∧ Nat.Prime p)
  (hq : q = a^b + c ∧ Nat.Prime q)
  (hr : r = c^a + b ∧ Nat.Prime r) :
  p = q ∨ q = r ∨ r = p := 
sorry

end two_primes_equal_l284_284967


namespace cherry_pies_count_correct_l284_284412

def total_pies : ℕ := 36

def ratio_ap_bb_ch : (ℕ × ℕ × ℕ) := (2, 3, 4)

def total_ratio_parts : ℕ := 2 + 3 + 4

def pies_per_part (total_pies : ℕ) (total_ratio_parts : ℕ) : ℕ := total_pies / total_ratio_parts

def num_parts_ch : ℕ := 4

def num_cherry_pies (total_pies : ℕ) (total_ratio_parts : ℕ) (num_parts_ch : ℕ) : ℕ :=
  pies_per_part total_pies total_ratio_parts * num_parts_ch

theorem cherry_pies_count_correct : num_cherry_pies total_pies total_ratio_parts num_parts_ch = 16 := by
  sorry

end cherry_pies_count_correct_l284_284412


namespace number_minus_six_l284_284126

variable (x : ℤ)

theorem number_minus_six
  (h : x / 5 = 2) : x - 6 = 4 := 
sorry

end number_minus_six_l284_284126


namespace selling_price_eq_l284_284985

noncomputable def cost_price : ℝ := 1300
noncomputable def selling_price_loss : ℝ := 1280
noncomputable def selling_price_profit_25_percent : ℝ := 1625

theorem selling_price_eq (cp sp_loss sp_profit sp: ℝ) 
  (h1 : sp_profit = 1.25 * cp)
  (h2 : sp_loss = cp - 20)
  (h3 : sp = cp + 20) :
  sp = 1320 :=
sorry

end selling_price_eq_l284_284985


namespace total_selling_price_correct_l284_284322

def cost_price_1 := 750
def cost_price_2 := 1200
def cost_price_3 := 500

def loss_percent_1 := 10
def loss_percent_2 := 15
def loss_percent_3 := 5

noncomputable def selling_price_1 := cost_price_1 - ((loss_percent_1 / 100) * cost_price_1)
noncomputable def selling_price_2 := cost_price_2 - ((loss_percent_2 / 100) * cost_price_2)
noncomputable def selling_price_3 := cost_price_3 - ((loss_percent_3 / 100) * cost_price_3)

noncomputable def total_selling_price := selling_price_1 + selling_price_2 + selling_price_3

theorem total_selling_price_correct : total_selling_price = 2170 := by
  sorry

end total_selling_price_correct_l284_284322


namespace simplify_and_evaluate_expression_l284_284414

theorem simplify_and_evaluate_expression (m : ℝ) (h : m = Real.sqrt 2 - 3) : 
  (1 - (3 / (m + 3))) / (m / (m^2 + 6 * m + 9)) = Real.sqrt 2 := 
by
  rw [h]
  sorry

end simplify_and_evaluate_expression_l284_284414


namespace solve_inequality_l284_284263

theorem solve_inequality (x : ℝ) (h : x ≠ -1) : (2 - x) / (x + 1) ≥ 0 ↔ -1 < x ∧ x ≤ 2 :=
by
  sorry

end solve_inequality_l284_284263


namespace circle_constant_ratio_l284_284716

theorem circle_constant_ratio (b : ℝ) :
  (∀ (x y : ℝ), (x + 4)^2 + (y + b)^2 = 16 → 
    ∃ k : ℝ, 
      ∀ P : ℝ × ℝ, 
        P = (x, y) → 
        dist P (-2, 0) / dist P (4, 0) = k)
  → b = 0 :=
by
  intros h
  sorry

end circle_constant_ratio_l284_284716


namespace inverse_89_mod_91_l284_284707

theorem inverse_89_mod_91 : ∃ x ∈ set.Icc 0 90, (89 * x) % 91 = 1 :=
by
  use 45
  split
  · exact ⟨le_refl 45, le_of_lt (by norm_num)⟩
  · norm_num; sorry

end inverse_89_mod_91_l284_284707


namespace rectangular_solid_volume_l284_284040

variables {x y z : ℝ}

theorem rectangular_solid_volume :
  x * y = 15 ∧ y * z = 10 ∧ x * z = 6 ∧ x = 3 * y →
  x * y * z = 6 * Real.sqrt 5 :=
by
  intros h
  sorry

end rectangular_solid_volume_l284_284040


namespace savings_by_buying_gallon_l284_284082

def gallon_to_ounces : ℕ := 128
def bottle_volume_ounces : ℕ := 16
def cost_gallon : ℕ := 8
def cost_bottle : ℕ := 3

theorem savings_by_buying_gallon :
  (cost_bottle * (gallon_to_ounces / bottle_volume_ounces)) - cost_gallon = 16 := 
by
  sorry

end savings_by_buying_gallon_l284_284082


namespace problem_I_problem_II_l284_284563

-- Problem (I)
theorem problem_I (a : ℝ) (h : ∀ x : ℝ, x^2 - 3 * a * x + 9 > 0) : -2 ≤ a ∧ a ≤ 2 :=
sorry

-- Problem (II)
theorem problem_II (m : ℝ) 
  (h₁ : ∀ x : ℝ, x^2 + 2 * x - 8 < 0 → x - m > 0)
  (h₂ : ∃ x : ℝ, x^2 + 2 * x - 8 < 0) : m ≤ -4 :=
sorry

end problem_I_problem_II_l284_284563


namespace find_price_of_stock_A_l284_284007

-- Define conditions
def stock_investment_A (price_A : ℝ) : Prop := 
  ∃ (income_A: ℝ), income_A = 0.10 * 100

def stock_investment_B (price_B : ℝ) (investment_B : ℝ) : Prop := 
  price_B = 115.2 ∧ investment_B = 10 / 0.12

-- The main goal statement
theorem find_price_of_stock_A 
  (price_A : ℝ) (investment_B : ℝ) 
  (hA : stock_investment_A price_A) 
  (hB : stock_investment_B price_A investment_B) :
  price_A = 138.24 := 
sorry

end find_price_of_stock_A_l284_284007


namespace regression_line_l284_284057

theorem regression_line (m x1 y1 : ℝ) (h_slope : m = 1.23) (h_center : (x1, y1) = (4, 5)) :
  ∃ b : ℝ, (∀ x y : ℝ, y = m * x + b ↔ y = 1.23 * x + 0.08) :=
by
  use 0.08
  sorry

end regression_line_l284_284057


namespace regular_polygon_sides_and_interior_angle_l284_284721

theorem regular_polygon_sides_and_interior_angle (n : ℕ) (H : (n - 2) * 180 = 3 * 360 + 180) :
  n = 9 ∧ (n - 2) * 180 / n = 140 :=
by
-- This marks the start of the proof, but the proof is omitted.
sorry

end regular_polygon_sides_and_interior_angle_l284_284721


namespace evaluate_neg_64_pow_two_thirds_l284_284032

theorem evaluate_neg_64_pow_two_thirds 
  (h : -64 = (-4)^3) : (-64)^(2/3) = 16 := 
by 
  -- sorry added to skip the proof.
  sorry  

end evaluate_neg_64_pow_two_thirds_l284_284032


namespace sum_first_3k_plus_2_terms_l284_284575

variable (k : ℕ)

def first_term : ℕ := k^2 + 1

def sum_of_sequence (n : ℕ) : ℕ :=
  let a₁ := first_term k
  let aₙ := a₁ + (n - 1)
  n * (a₁ + aₙ) / 2

theorem sum_first_3k_plus_2_terms :
  sum_of_sequence k (3 * k + 2) = 3 * k^3 + 8 * k^2 + 6 * k + 3 :=
by
  -- Here we define the sequence and compute the sum
  sorry

end sum_first_3k_plus_2_terms_l284_284575


namespace smallest_natural_number_l284_284150

theorem smallest_natural_number (x : ℕ) : 
  (x % 5 = 2) ∧ (x % 6 = 2) ∧ (x % 7 = 3) → x = 122 := 
by
  sorry

end smallest_natural_number_l284_284150


namespace newspaper_spending_over_8_weeks_l284_284209

theorem newspaper_spending_over_8_weeks :
  (3 * 0.50 + 2.00) * 8 = 28 := by
  sorry

end newspaper_spending_over_8_weeks_l284_284209


namespace solve_for_k_l284_284223

theorem solve_for_k (k : ℝ) (h : 2 * (5:ℝ)^2 + 3 * (5:ℝ) - k = 0) : k = 65 := 
by
  sorry

end solve_for_k_l284_284223


namespace sum_of_valid_two_digit_numbers_l284_284700

theorem sum_of_valid_two_digit_numbers
  (a b : ℕ)
  (h1 : 1 ≤ a ∧ a ≤ 9)
  (h2 : 0 ≤ b ∧ b ≤ 9)
  (h3 : (a - b) ∣ (10 * a + b))
  (h4 : (a * b) ∣ (10 * a + b)) :
  (10 * a + b = 21) → (21 = 21) :=
sorry

end sum_of_valid_two_digit_numbers_l284_284700


namespace minimum_value_of_F_l284_284899

noncomputable def F (m n : ℝ) : ℝ := (m - n)^2 + (m^2 - n + 1)^2

theorem minimum_value_of_F : 
  (∀ m n : ℝ, F m n ≥ 9 / 32) ∧ (∃ m n : ℝ, F m n = 9 / 32) :=
by
  sorry

end minimum_value_of_F_l284_284899


namespace two_digit_primes_ending_in_3_eq_6_l284_284768

open Nat

def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def ends_in_digit_3 (n : ℕ) : Prop :=
  n % 10 = 3

def count_two_digit_primes_ending_in_3 : ℕ :=
  ([13, 23, 33, 43, 53, 63, 73, 83, 93].filter (λ n, is_prime n ∧ is_two_digit n ∧ ends_in_digit_3 n)).length

theorem two_digit_primes_ending_in_3_eq_6 : count_two_digit_primes_ending_in_3 = 6 :=
by
  sorry

end two_digit_primes_ending_in_3_eq_6_l284_284768


namespace find_new_ratio_l284_284101

def initial_ratio (H C : ℕ) : Prop := H = 6 * C

def transaction (H C : ℕ) : Prop :=
  H - 15 = (C + 15) + 70

def new_ratio (H C : ℕ) : Prop := H - 15 = 3 * (C + 15)

theorem find_new_ratio (H C : ℕ) (h1 : initial_ratio H C) (h2 : transaction H C) : 
  new_ratio H C :=
sorry

end find_new_ratio_l284_284101


namespace tan_105_l284_284598

theorem tan_105 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  -- Definitions
  let tan45 := Real.tan (45 * Real.pi / 180)
  let tan60 := Real.tan (60 * Real.pi / 180)
  have h1 : tan45 = 1 := sorry
  have h2 : tan60 = Real.sqrt 3 := sorry
  have h3 : tan45 + tan60 = 1 + Real.sqrt 3 := sorry
  have h4 : 1 - tan45 * tan60 = 1 - 1 * Real.sqrt 3 := sorry
  
  -- Use tangent addition formula
  have tan_addition : Real.tan (105 * Real.pi / 180) = (1 + Real.sqrt 3) / (1 - Real.sqrt 3) := sorry
  
  -- Simplify and rationalize to prove the final result
  have tan_105_eq : (1 + Real.sqrt 3) / (1 - Real.sqrt 3) = -2 - Real.sqrt 3 := sorry
  
  exact tan_105_eq

end tan_105_l284_284598


namespace five_by_five_rectangles_l284_284492

def choose (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem five_by_five_rectangles : (choose 5 2) * (choose 5 2) = 100 :=
by
  sorry

end five_by_five_rectangles_l284_284492


namespace sum_of_cuberoots_gt_two_l284_284116

theorem sum_of_cuberoots_gt_two {x₁ x₂ : ℝ} (h₁: x₁^3 = 6 / 5) (h₂: x₂^3 = 5 / 6) : x₁ + x₂ > 2 :=
sorry

end sum_of_cuberoots_gt_two_l284_284116


namespace small_panda_bears_count_l284_284814

theorem small_panda_bears_count :
  ∃ (S : ℕ), ∃ (B : ℕ),
    B = 5 ∧ 7 * (25 * S + 40 * B) = 2100 ∧ S = 4 :=
by
  exists 4
  exists 5
  repeat { sorry }

end small_panda_bears_count_l284_284814


namespace base_7_to_base_10_l284_284294

theorem base_7_to_base_10 (a b c d e : ℕ) (h : 23456 = e * 10000 + d * 1000 + c * 100 + b * 10 + a) :
  2 * 7^4 + 3 * 7^3 + 4 * 7^2 + 5 * 7^1 + 6 * 7^0 = 6068 :=
by
  sorry

end base_7_to_base_10_l284_284294


namespace ab_conditions_l284_284244

theorem ab_conditions (a b : ℝ) : ¬((a > b → a^2 > b^2) ∧ (a^2 > b^2 → a > b)) :=
by 
  sorry

end ab_conditions_l284_284244


namespace eval_expression_eq_2_l284_284854

theorem eval_expression_eq_2 :
  (10^2 + 11^2 + 12^2 + 13^2 + 14^2) / 365 = 2 :=
by
  sorry

end eval_expression_eq_2_l284_284854


namespace three_digit_numbers_count_l284_284217

theorem three_digit_numbers_count :
  let is_valid_number (h t u : ℕ) := 1 ≤ h ∧ h ≤ 9 ∧ 0 ≤ t ∧ t ≤ 9 ∧ 0 ≤ u ∧ u ≤ 9 ∧ h > t ∧ t > u
  in (∑ h in finset.range 10, ∑ t in finset.range h, ∑ u in finset.range t, if is_valid_number h t u then 1 else 0) = 84 :=
by
  let is_valid_number (h t u : ℕ) := 1 ≤ h ∧ h ≤ 9 ∧ 0 ≤ t ∧ t ≤ 9 ∧ 0 ≤ u ∧ u ≤ 9 ∧ h>t ∧ t>u
  have : ∑ h in finset.range 10, ∑ t in finset.range h, ∑ u in finset.range t, if is_valid_number h t u then 1 else 0 = 84 := by {
    -- skipping the proof
    sorry
  }
  exact this

end three_digit_numbers_count_l284_284217


namespace train_crosses_platform_in_39_seconds_l284_284001

theorem train_crosses_platform_in_39_seconds :
  ∀ (length_train length_platform : ℝ) (time_cross_signal : ℝ),
  length_train = 300 →
  length_platform = 25 →
  time_cross_signal = 36 →
  ((length_train + length_platform) / (length_train / time_cross_signal)) = 39 := by
  intros length_train length_platform time_cross_signal
  intros h_length_train h_length_platform h_time_cross_signal
  rw [h_length_train, h_length_platform, h_time_cross_signal]
  sorry

end train_crosses_platform_in_39_seconds_l284_284001


namespace annie_gives_mary_25_crayons_l284_284579

theorem annie_gives_mary_25_crayons :
  let initial_crayons_given := 21
  let initial_crayons_in_locker := 36
  let bobby_gift := initial_crayons_in_locker / 2
  let total_crayons := initial_crayons_given + initial_crayons_in_locker + bobby_gift
  let mary_share := total_crayons / 3
  mary_share = 25 := 
by
  sorry

end annie_gives_mary_25_crayons_l284_284579


namespace minimum_value_of_expression_l284_284963

open Real

noncomputable def f (x y z : ℝ) : ℝ := (x + 2 * y) / (x * y * z)

theorem minimum_value_of_expression :
  ∀ (x y z : ℝ),
    x > 0 → y > 0 → z > 0 →
    x + y + z = 1 →
    x = 2 * y →
    f x y z = 8 :=
by
  intro x y z x_pos y_pos z_pos h_sum h_xy
  sorry

end minimum_value_of_expression_l284_284963


namespace tan_105_eq_neg2_sub_sqrt3_l284_284669

theorem tan_105_eq_neg2_sub_sqrt3 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 := by
  sorry

end tan_105_eq_neg2_sub_sqrt3_l284_284669


namespace average_monthly_sales_booster_club_l284_284022

noncomputable def monthly_sales : List ℕ := [80, 100, 75, 95, 110, 180, 90, 115, 130, 200, 160, 140]

noncomputable def average_sales (sales : List ℕ) : ℝ :=
  (sales.foldr (λ x acc => x + acc) 0 : ℕ) / sales.length

theorem average_monthly_sales_booster_club : average_sales monthly_sales = 122.92 := by
  sorry

end average_monthly_sales_booster_club_l284_284022


namespace bob_speed_l284_284020

theorem bob_speed (j_speed : ℝ) (b_headstart : ℝ) (t : ℝ) (j_catches_up : t = 20 / 60 ∧ j_speed = 9 ∧ b_headstart = 1) : 
  ∃ b_speed : ℝ, b_speed = 6 := 
by
  sorry

end bob_speed_l284_284020


namespace initial_blue_balls_l284_284993

theorem initial_blue_balls (total_balls : ℕ) (remaining_balls : ℕ) (B : ℕ) :
  total_balls = 18 → remaining_balls = total_balls - 3 → (B - 3) / remaining_balls = 1 / 5 → B = 6 :=
by 
  intros htotal hremaining hprob
  sorry

end initial_blue_balls_l284_284993


namespace at_most_three_prime_divisors_l284_284195

-- Given a and b natural numbers such that a > 1, b is divisible by a^2,
-- and any divisor of b less than sqrt a is also a divisor of a, 
-- we need to prove that a has no more than three distinct prime divisors.

theorem at_most_three_prime_divisors
  (a b : ℕ) 
  (h1 : 1 < a)
  (h2 : a * a ∣ b)
  (h3 : ∀ d : ℕ, d ∣ b → d < Math.sqrt a → d ∣ a) 
  : finset.card (nat.factors a).to_finset ≤ 3 :=
sorry

end at_most_three_prime_divisors_l284_284195


namespace tan_105_eq_neg2_sub_sqrt3_l284_284642

theorem tan_105_eq_neg2_sub_sqrt3 :
  Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_eq_neg2_sub_sqrt3_l284_284642


namespace work_done_by_force_l284_284143

def F (x : ℝ) := 4 * x - 1

theorem work_done_by_force :
  let a := 1
  let b := 3
  (∫ x in a..b, F x) = 14 := by
  sorry

end work_done_by_force_l284_284143


namespace douglas_percent_votes_l284_284949

def percentageOfTotalVotesWon (votes_X votes_Y: ℕ) (percent_X percent_Y: ℕ) : ℕ :=
  let total_votes_Douglas : ℕ := (percent_X * 2 * votes_X + percent_Y * votes_Y)
  let total_votes_cast : ℕ := 3 * votes_Y
  (total_votes_Douglas * 100 / total_votes_cast)

theorem douglas_percent_votes (votes_X votes_Y : ℕ) (h_ratio : 2 * votes_X = votes_Y)
  (h_perc_X : percent_X = 64)
  (h_perc_Y : percent_Y = 46) :
  percentageOfTotalVotesWon votes_X votes_Y 64 46 = 58 := by
    sorry

end douglas_percent_votes_l284_284949


namespace smallest_sum_of_digits_l284_284356

noncomputable def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |>.sum

theorem smallest_sum_of_digits (n : ℕ) (h : sum_of_digits n = 2017) : sum_of_digits (n + 1) = 2 := 
sorry

end smallest_sum_of_digits_l284_284356


namespace perimeter_of_triangle_is_36_l284_284986

variable (inradius : ℝ)
variable (area : ℝ)
variable (P : ℝ)

theorem perimeter_of_triangle_is_36 (h1 : inradius = 2.5) (h2 : area = 45) : 
  P / 2 * inradius = area → P = 36 :=
sorry

end perimeter_of_triangle_is_36_l284_284986


namespace grid_area_l284_284541

theorem grid_area :
  let B := 10   -- Number of boundary points
  let I := 12   -- Number of interior points
  I + B / 2 - 1 = 16 :=
by
  sorry

end grid_area_l284_284541


namespace area_of_triangle_PQR_l284_284469

-- Define the problem conditions
def PQ : ℝ := 4
def PR : ℝ := 4
def angle_P : ℝ := 45 -- degrees

-- Define the main problem
theorem area_of_triangle_PQR : 
  (PQ = PR) ∧ (angle_P = 45) ∧ (PR = 4) → 
  ∃ A, A = 8 := 
by
  sorry

end area_of_triangle_PQR_l284_284469


namespace problem_statement_l284_284395

/-- Let x, y, z be nonzero real numbers such that x + y + z = 0.
    Prove that ∀ x y z : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 → x + y + z = 0 → (x^3 + y^3 + z^3) / (x * y * z) = 3. -/
theorem problem_statement (x y z : ℝ) (h : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0) (h₁ : x + y + z = 0) :
  (x^3 + y^3 + z^3) / (x * y * z) = 3 := 
by 
  sorry

end problem_statement_l284_284395


namespace ratio_of_socks_l284_284332

-- Conditions:
variable (B : ℕ) (W : ℕ) (L : ℕ)
-- B = number of black socks
-- W = initial number of white socks
-- L = number of white socks lost

-- Setting given conditions:
axiom hB : B = 6
axiom hL : L = W / 2
axiom hCond : W / 2 = B + 6

-- Prove the ratio of white socks to black socks is 4:1
theorem ratio_of_socks : B = 6 → W / 2 = B + 6 → (W / 2) + (W / 2) = 24 → (B : ℚ) / (W : ℚ) = 1 / 4 :=
by intros hB hCond hW
   sorry

end ratio_of_socks_l284_284332


namespace fair_dice_can_be_six_l284_284559

def fair_dice_outcomes : Set ℕ := {1, 2, 3, 4, 5, 6}

theorem fair_dice_can_be_six : 6 ∈ fair_dice_outcomes :=
by {
  -- This formally states that 6 is a possible outcome when throwing a fair dice
  sorry
}

end fair_dice_can_be_six_l284_284559


namespace newspaper_cost_over_8_weeks_l284_284215

def cost (day : String) : Real := 
  if day = "Sunday" then 2.00 
  else if day = "Wednesday" ∨ day = "Thursday" ∨ day = "Friday" then 0.50 
  else 0

theorem newspaper_cost_over_8_weeks : 
  (8 * ((cost "Wednesday" + cost "Thursday" + cost "Friday") + cost "Sunday")) = 28.00 :=
  by sorry

end newspaper_cost_over_8_weeks_l284_284215


namespace max_slope_of_line_OQ_l284_284486

-- Given conditions
variables {p : ℕ} (h_pos_p : p > 0)
def parabola_eq (p : ℕ) := ∀ (x y : ℝ), y^2 = 2 * p * x

-- Given distance from the focus to the directrix is 2
lemma distance_focus_directrix_eq_two : p = 2 :=
by sorry

-- Thus the equation of the parabola is:
def parabola : ∀ (x y : ℝ), y^2 = 4 * x :=
by sorry

-- Variables for point P and Q
variables {O P Q : ℝ × ℝ}
-- Point P lies on the parabola
variables (hP : ∃ (x y : ℝ), y^2 = 4 * x)
-- Condition relating vector PQ and QF
variables (hPQ_QF : ∀ (P Q F : ℝ × ℝ), (P - Q) = 9 * (Q - F))
-- Maximizing slope of line OQ
def max_slope (O Q : ℝ × ℝ) : ℝ := 
  ∀ (m n : ℝ), let slope := n / ((25 * n^2 + 9) / 10) in
  slope ≤ 1 / 3 := 
by sorry

-- Prove the theorem equivalent to solution part 2, maximum slope is 1/3
theorem max_slope_of_line_OQ : max_slope O Q = 1 / 3 :=
by sorry

end max_slope_of_line_OQ_l284_284486


namespace usual_time_to_reach_school_l284_284137

variable (R T : ℝ)
variable (h : T * R = (T - 4) * (7/6 * R))

theorem usual_time_to_reach_school (h : T * R = (T - 4) * (7/6 * R)) : T = 28 := by
  sorry

end usual_time_to_reach_school_l284_284137


namespace soda_count_l284_284885

theorem soda_count
  (W : ℕ) (S : ℕ) (B : ℕ) (T : ℕ)
  (hW : W = 26) (hB : B = 17) (hT : T = 31) :
  W + S - B = T → S = 22 :=
by
  sorry

end soda_count_l284_284885


namespace problem_statement_l284_284135

def operation (x y z : ℕ) : ℕ := (2 * x * y + y^2) * z^3

theorem problem_statement : operation 7 (operation 4 5 3) 2 = 24844760 :=
by
  sorry

end problem_statement_l284_284135


namespace correct_functional_relationship_water_amount_20th_minute_water_amount_supply_days_l284_284560

-- Define the constants k and b
variables (k b : ℝ)

-- Define the function y = k * t + b
def linear_func (t : ℝ) : ℝ := k * t + b

-- Define the data points as conditions
axiom data_point1 : linear_func k b 1 = 7
axiom data_point2 : linear_func k b 2 = 12
axiom data_point3 : linear_func k b 3 = 17
axiom data_point4 : linear_func k b 4 = 22
axiom data_point5 : linear_func k b 5 = 27

-- Define the water consumption rate and total minutes in a day
def daily_water_consumption : ℝ := 1500
def minutes_in_one_day : ℝ := 1440
def days_in_month : ℝ := 30

-- The expression y = 5t + 2
theorem correct_functional_relationship : (k = 5) ∧ (b = 2) :=
by
  sorry

-- Estimated water amount at the 20th minute
theorem water_amount_20th_minute (t : ℝ) (ht : t = 20) : linear_func 5 2 t = 102 :=
by
  sorry

-- The water leaked in a month (30 days) can supply the number of days
theorem water_amount_supply_days : (linear_func 5 2 (minutes_in_one_day * days_in_month)) / daily_water_consumption = 144 :=
by
  sorry

end correct_functional_relationship_water_amount_20th_minute_water_amount_supply_days_l284_284560


namespace num_two_digit_primes_with_ones_digit_3_l284_284738

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100
  
def ones_digit_is_3 (n : ℕ) : Prop :=
  n % 10 = 3

theorem num_two_digit_primes_with_ones_digit_3 : 
  (∃ n1 n2 n3 n4 n5 n6 : ℕ, 
   two_digit_number n1 ∧ ones_digit_is_3 n1 ∧ is_prime n1 ∧ 
   two_digit_number n2 ∧ ones_digit_is_3 n2 ∧ is_prime n2 ∧ 
   two_digit_number n3 ∧ ones_digit_is_3 n3 ∧ is_prime n3 ∧ 
   two_digit_number n4 ∧ ones_digit_is_3 n4 ∧ is_prime n4 ∧ 
   two_digit_number n5 ∧ ones_digit_is_3 n5 ∧ is_prime n5 ∧ 
   two_digit_number n6 ∧ ones_digit_is_3 n6 ∧ is_prime n6) ∧
  (∀ n : ℕ, two_digit_number n → ones_digit_is_3 n → is_prime n → 
  n = n1 ∨ n = n2 ∨ n = n3 ∨ n = n4 ∨ n = n5 ∨ n = n6) :=
sorry

end num_two_digit_primes_with_ones_digit_3_l284_284738


namespace most_likely_sitting_people_l284_284402

theorem most_likely_sitting_people :
  let num_people := 100
  let seats := 100
  let favorite_seats : Fin num_people → Fin seats := sorry
  -- Conditions related to people sitting behavior
  let sits_in_row (i : Fin num_people) : Prop :=
    ∀ j : Fin num_people, j < i → favorite_seats j ≠ favorite_seats i
  let num_sitting_in_row := Finset.card (Finset.filter sits_in_row (Finset.univ : Finset (Fin num_people)))
  -- Prove
  num_sitting_in_row = 10 := 
sorry

end most_likely_sitting_people_l284_284402


namespace probability_all_have_one_after_2020_rings_l284_284045

/--
Four friends – Alex, Bella, Charlie, and Dana – each start with $1. 
A bell rings every 10 seconds, and each of the players who currently have money independently 
chooses one of the other three players at random and gives $1 to that player. 
Prove that the probability that after the bell has rung 2020 times, each player will have $1 is 2/27.
-/
theorem probability_all_have_one_after_2020_rings : 
  let initial_state := (1, 1, 1, 1),
      bell_interval := 10,
      num_rings := 2020,
      probability_each_has_one := 2 / 27 in
  (∀ rings, rings = num_rings → 
    let final_state := ring_bell num_rings initial_state bell_interval in
    final_state = (1, 1, 1, 1) → 
    Pr(final_state) = probability_each_has_one) := sorry

end probability_all_have_one_after_2020_rings_l284_284045


namespace initial_total_quantity_l284_284873

theorem initial_total_quantity
  (x : ℝ)
  (milk_water_ratio : 5 / 9 = 5 * x / (3 * x + 12))
  (milk_juice_ratio : 5 / 8 = 5 * x / (4 * x + 6)) :
  5 * x + 3 * x + 4 * x = 24 :=
by
  sorry

end initial_total_quantity_l284_284873


namespace completing_the_square_l284_284861

theorem completing_the_square (x : ℝ) : x^2 - 8 * x + 1 = 0 → (x - 4)^2 = 15 :=
by
  sorry

end completing_the_square_l284_284861


namespace sum_of_integers_with_product_neg13_l284_284117

theorem sum_of_integers_with_product_neg13 (a b c : ℤ) (h : a * b * c = -13) : 
  a + b + c = 13 ∨ a + b + c = -11 := 
sorry

end sum_of_integers_with_product_neg13_l284_284117


namespace positive_difference_of_complementary_angles_l284_284270

-- Define the conditions
variables (a b : ℝ)
variable (h1 : a + b = 90)
variable (h2 : 5 * b = a)

-- Define the theorem we are proving
theorem positive_difference_of_complementary_angles (a b : ℝ) (h1 : a + b = 90) (h2 : 5 * b = a) :
  |a - b| = 60 :=
by
  sorry

end positive_difference_of_complementary_angles_l284_284270


namespace largest_value_of_x_l284_284709

theorem largest_value_of_x : 
  ∃ x, ( (15 * x^2 - 30 * x + 9) / (4 * x - 3) + 6 * x = 7 * x - 2 ) ∧ x = (19 + Real.sqrt 229) / 22 :=
sorry

end largest_value_of_x_l284_284709


namespace organic_fertilizer_prices_l284_284504

theorem organic_fertilizer_prices
  (x y : ℝ)
  (h1 : x - y = 100)
  (h2 : 2 * x + y = 1700) :
  x = 600 ∧ y = 500 :=
by {
  sorry
}

end organic_fertilizer_prices_l284_284504


namespace find_y_eq_l284_284165

theorem find_y_eq (y : ℝ) : (10 - y)^2 = 4 * y^2 → (y = 10 / 3 ∨ y = -10) :=
by
  intro h
  -- The detailed proof will be provided here
  sorry

end find_y_eq_l284_284165


namespace base_7_to_base_10_l284_284296

theorem base_7_to_base_10 (a b c d e : ℕ) (h : 23456 = e * 10000 + d * 1000 + c * 100 + b * 10 + a) :
  2 * 7^4 + 3 * 7^3 + 4 * 7^2 + 5 * 7^1 + 6 * 7^0 = 6068 :=
by
  sorry

end base_7_to_base_10_l284_284296


namespace number_of_people_l284_284338

-- Definitions based on the conditions
def total_cookies : ℕ := 420
def cookies_per_person : ℕ := 30

-- The goal is to prove the number of people is 14
theorem number_of_people : total_cookies / cookies_per_person = 14 :=
by
  sorry

end number_of_people_l284_284338


namespace count_two_digit_primes_ending_with_3_l284_284763

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m * m ≤ n → n % m ≠ 0

def two_digit (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100

def ends_with_3 (n : ℕ) : Prop :=
  n % 10 = 3

theorem count_two_digit_primes_ending_with_3 :
  {n : ℕ | two_digit n ∧ ends_with_3 n ∧ is_prime n}.to_finset.card = 6 := by
sorry

end count_two_digit_primes_ending_with_3_l284_284763


namespace rahim_books_l284_284406

/-- 
Rahim bought some books for Rs. 6500 from one shop and 35 books for Rs. 2000 from another. 
The average price he paid per book is Rs. 85. 
Prove that Rahim bought 65 books from the first shop. 
-/
theorem rahim_books (x : ℕ) 
  (h1 : 6500 + 2000 = 8500) 
  (h2 : 85 * (x + 35) = 8500) : 
  x = 65 := 
sorry

end rahim_books_l284_284406


namespace quadratic_inequality_solution_l284_284427

theorem quadratic_inequality_solution:
  ∀ x : ℝ, (x^2 + 2 * x < 3) ↔ (-3 < x ∧ x < 1) :=
by
  sorry

end quadratic_inequality_solution_l284_284427


namespace total_cost_over_8_weeks_l284_284210

def cost_per_weekday_edition : ℝ := 0.50
def cost_per_sunday_edition : ℝ := 2.00
def num_weekday_editions_per_week : ℕ := 3
def duration_in_weeks : ℕ := 8

theorem total_cost_over_8_weeks :
  (num_weekday_editions_per_week * cost_per_weekday_edition + cost_per_sunday_edition) * duration_in_weeks = 28.00 := by
  sorry

end total_cost_over_8_weeks_l284_284210


namespace zeros_in_interval_l284_284369

noncomputable def f (a x : ℝ) : ℝ := (1 / 2) * x^2 - a * Real.log(x)

theorem zeros_in_interval (a : ℝ) (h: a > 0) :
  (∀ x : ℝ, 1 < x ∧ x < Real.exp 1 → f a x = 0) ↔ (a ∈ Set.Ioo (Real.exp 1) ((1 / 2) * Real.exp 2)) :=
sorry

end zeros_in_interval_l284_284369


namespace tan_105_eq_neg2_sub_sqrt3_l284_284647

theorem tan_105_eq_neg2_sub_sqrt3 :
  Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_eq_neg2_sub_sqrt3_l284_284647


namespace Manu_takes_12_more_seconds_l284_284404

theorem Manu_takes_12_more_seconds (P M A : ℕ) 
  (hP : P = 60) 
  (hA1 : A = 36) 
  (hA2 : A = M / 2) : 
  M - P = 12 :=
by
  sorry

end Manu_takes_12_more_seconds_l284_284404


namespace tan_105_l284_284608

theorem tan_105 :
  tan 105 = -2 - sqrt 3 :=
by sorry

end tan_105_l284_284608


namespace carl_typing_hours_per_day_l284_284342

theorem carl_typing_hours_per_day (words_per_minute : ℕ) (total_words : ℕ) (days : ℕ) (hours_per_day : ℕ) :
  words_per_minute = 50 →
  total_words = 84000 →
  days = 7 →
  hours_per_day = (total_words / days) / (words_per_minute * 60) →
  hours_per_day = 4 :=
by
  intros h_word_rate h_total_words h_days h_hrs_formula
  rewrite [h_word_rate, h_total_words, h_days] at h_hrs_formula
  exact h_hrs_formula

end carl_typing_hours_per_day_l284_284342


namespace graveyard_bones_count_l284_284773

def total_skeletons : ℕ := 20
def half_total (n : ℕ) : ℕ := n / 2
def skeletons_adult_women : ℕ := half_total total_skeletons
def remaining_skeletons : ℕ := total_skeletons - skeletons_adult_women
def even_split (n : ℕ) : ℕ := n / 2
def skeletons_adult_men : ℕ := even_split remaining_skeletons
def skeletons_children : ℕ := even_split remaining_skeletons

def bones_per_woman : ℕ := 20
def bones_per_man : ℕ := bones_per_woman + 5
def bones_per_child : ℕ := bones_per_woman / 2

def total_bones_adult_women : ℕ := skeletons_adult_women * bones_per_woman
def total_bones_adult_men : ℕ := skeletons_adult_men * bones_per_man
def total_bones_children : ℕ := skeletons_children * bones_per_child

def total_bones_in_graveyard : ℕ := total_bones_adult_women + total_bones_adult_men + total_bones_children

theorem graveyard_bones_count : total_bones_in_graveyard = 375 := by
  sorry

end graveyard_bones_count_l284_284773


namespace remaining_yards_correct_l284_284871

-- Define the conversion constant
def yards_per_mile: ℕ := 1760

-- Define the conditions
def marathon_in_miles: ℕ := 26
def marathon_in_yards: ℕ := 395
def total_marathons: ℕ := 15

-- Define the function to calculate the remaining yards after conversion
def calculate_remaining_yards (marathon_in_miles marathon_in_yards total_marathons yards_per_mile: ℕ): ℕ :=
  let total_yards := total_marathons * marathon_in_yards
  total_yards % yards_per_mile

-- Statement to prove
theorem remaining_yards_correct :
  calculate_remaining_yards marathon_in_miles marathon_in_yards total_marathons yards_per_mile = 645 :=
  sorry

end remaining_yards_correct_l284_284871


namespace number_of_two_digit_primes_with_ones_digit_three_l284_284740

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def has_ones_digit_three (n : ℕ) : Prop :=
  n % 10 = 3

def is_prime (n : ℕ) : Prop :=
  Nat.Prime n

theorem number_of_two_digit_primes_with_ones_digit_three :
  ∃! s : Finset ℕ, (∀ n ∈ s, is_two_digit n ∧ has_ones_digit_three n ∧ is_prime n) ∧ s.card = 6 :=
by
  sorry

end number_of_two_digit_primes_with_ones_digit_three_l284_284740


namespace points_three_units_away_from_neg_two_on_number_line_l284_284103

theorem points_three_units_away_from_neg_two_on_number_line :
  ∃! p1 p2 : ℤ, |p1 + 2| = 3 ∧ |p2 + 2| = 3 ∧ p1 ≠ p2 ∧ (p1 = -5 ∨ p2 = -5) ∧ (p1 = 1 ∨ p2 = 1) :=
sorry

end points_three_units_away_from_neg_two_on_number_line_l284_284103


namespace part_a_2001_good_part_a_3001_not_good_part_b_product_of_good_is_good_part_c_product_good_then_one_good_l284_284325

-- Definition of a good number
def is_good (n : ℕ) : Prop := (n % 6 = 3)

-- Lean 4 statements

-- 1. 2001 is good
theorem part_a_2001_good : is_good 2001 :=
by sorry

-- 2. 3001 isn't good
theorem part_a_3001_not_good : ¬ is_good 3001 :=
by sorry

-- 3. The product of two good numbers is a good number
theorem part_b_product_of_good_is_good (x y : ℕ) (hx : is_good x) (hy : is_good y) : is_good (x * y) :=
by sorry

-- 4. If the product of two numbers is good, then at least one of the numbers is good
theorem part_c_product_good_then_one_good (x y : ℕ) (hxy : is_good (x * y)) : is_good x ∨ is_good y :=
by sorry

end part_a_2001_good_part_a_3001_not_good_part_b_product_of_good_is_good_part_c_product_good_then_one_good_l284_284325


namespace probability_coin_covers_black_region_l284_284445

open Real

noncomputable def coin_cover_black_region_probability : ℝ :=
  let side_length_square := 10
  let triangle_leg := 3
  let diamond_side_length := 3 * sqrt 2
  let smaller_square_side := 1
  let coin_diameter := 1
  -- The derived probability calculation
  (32 + 9 * sqrt 2 + π) / 81

theorem probability_coin_covers_black_region :
  coin_cover_black_region_probability = (32 + 9 * sqrt 2 + π) / 81 :=
by
  -- Proof goes here
  sorry

end probability_coin_covers_black_region_l284_284445


namespace number_of_ways_to_win_championships_l284_284940

-- Definitions for the problem
def num_athletes := 5
def num_events := 3

-- Proof statement
theorem number_of_ways_to_win_championships : 
  (num_athletes ^ num_events) = 125 := 
by 
  sorry

end number_of_ways_to_win_championships_l284_284940


namespace area_of_rectangle_l284_284421

-- Given conditions
def shadedSquareArea : ℝ := 4
def nonShadedSquareArea : ℝ := shadedSquareArea
def largerSquareArea : ℝ := 4 * 4  -- Since the side length is twice the previous squares

-- Problem statement
theorem area_of_rectangle (shadedSquareArea nonShadedSquareArea largerSquareArea : ℝ) :
  shadedSquareArea + nonShadedSquareArea + largerSquareArea = 24 :=
sorry

end area_of_rectangle_l284_284421


namespace part1_part2_l284_284485

-- Define the conditions that translate the quadratic equation having distinct real roots
def discriminant_condition (m : ℝ) : Prop :=
  let a := 1
  let b := -4
  let c := 3 - 2 * m
  b ^ 2 - 4 * a * c > 0

-- Define the root condition from Vieta's formulas and the additional given condition
def additional_condition (m : ℝ) : Prop :=
  let x1_plus_x2 := 4
  let x1_times_x2 := 3 - 2 * m
  x1_times_x2 + x1_plus_x2 - m^2 = 4

-- Prove the range of m for part 1
theorem part1 (m : ℝ) : discriminant_condition m → m ≥ -1/2 := by
  sorry

-- Prove the value of m for part 2 with the range condition
theorem part2 (m : ℝ) : discriminant_condition m → additional_condition m → m = 1 := by
  sorry

end part1_part2_l284_284485


namespace tan_105_l284_284600

theorem tan_105 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  -- Definitions
  let tan45 := Real.tan (45 * Real.pi / 180)
  let tan60 := Real.tan (60 * Real.pi / 180)
  have h1 : tan45 = 1 := sorry
  have h2 : tan60 = Real.sqrt 3 := sorry
  have h3 : tan45 + tan60 = 1 + Real.sqrt 3 := sorry
  have h4 : 1 - tan45 * tan60 = 1 - 1 * Real.sqrt 3 := sorry
  
  -- Use tangent addition formula
  have tan_addition : Real.tan (105 * Real.pi / 180) = (1 + Real.sqrt 3) / (1 - Real.sqrt 3) := sorry
  
  -- Simplify and rationalize to prove the final result
  have tan_105_eq : (1 + Real.sqrt 3) / (1 - Real.sqrt 3) = -2 - Real.sqrt 3 := sorry
  
  exact tan_105_eq

end tan_105_l284_284600


namespace complex_multiplication_l284_284021

variable (i : ℂ)
axiom imag_unit : i^2 = -1

theorem complex_multiplication : (3 + i) * i = -1 + 3 * i :=
by
  sorry

end complex_multiplication_l284_284021


namespace temperature_on_tuesday_l284_284868

theorem temperature_on_tuesday 
  (T W Th F : ℝ)
  (H1 : (T + W + Th) / 3 = 45)
  (H2 : (W + Th + F) / 3 = 50)
  (H3 : F = 53) :
  T = 38 :=
by 
  sorry

end temperature_on_tuesday_l284_284868


namespace minimum_bottles_needed_l284_284344

theorem minimum_bottles_needed (fl_oz_needed : ℝ) (bottle_size_ml : ℝ) (fl_oz_per_liter : ℝ) (ml_per_liter : ℝ)
  (h1 : fl_oz_needed = 60)
  (h2 : bottle_size_ml = 250)
  (h3 : fl_oz_per_liter = 33.8)
  (h4 : ml_per_liter = 1000) :
  ∃ n : ℕ, n = 8 ∧ fl_oz_needed * ml_per_liter / fl_oz_per_liter / bottle_size_ml ≤ n :=
by
  sorry

end minimum_bottles_needed_l284_284344


namespace tan_105_eq_neg_2_sub_sqrt_3_l284_284617

-- Definitions
def angle105 : ℝ := 105 * (Math.pi / 180)
def angle45 : ℝ := 45 * (Math.pi / 180)
def angle60 : ℝ := 60 * (Math.pi / 180)

-- Theorem
theorem tan_105_eq_neg_2_sub_sqrt_3 :
  Real.tan angle105 = -2 - Real.sqrt 3 := by
  sorry

end tan_105_eq_neg_2_sub_sqrt_3_l284_284617


namespace monotonically_decreasing_iff_l284_284202

noncomputable def f (a x : ℝ) : ℝ :=
if x < 1 then (3 * a - 2) * x + 6 * a - 1 else a^x

theorem monotonically_decreasing_iff (a : ℝ) : 
  (∀ x y : ℝ, x ≤ y → f a y ≤ f a x) ↔ (3/8 ≤ a ∧ a < 2/3) :=
sorry

end monotonically_decreasing_iff_l284_284202


namespace tan_105_l284_284595

theorem tan_105 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  -- Definitions
  let tan45 := Real.tan (45 * Real.pi / 180)
  let tan60 := Real.tan (60 * Real.pi / 180)
  have h1 : tan45 = 1 := sorry
  have h2 : tan60 = Real.sqrt 3 := sorry
  have h3 : tan45 + tan60 = 1 + Real.sqrt 3 := sorry
  have h4 : 1 - tan45 * tan60 = 1 - 1 * Real.sqrt 3 := sorry
  
  -- Use tangent addition formula
  have tan_addition : Real.tan (105 * Real.pi / 180) = (1 + Real.sqrt 3) / (1 - Real.sqrt 3) := sorry
  
  -- Simplify and rationalize to prove the final result
  have tan_105_eq : (1 + Real.sqrt 3) / (1 - Real.sqrt 3) = -2 - Real.sqrt 3 := sorry
  
  exact tan_105_eq

end tan_105_l284_284595


namespace sum_of_squares_and_product_l284_284991

theorem sum_of_squares_and_product
  (x y : ℕ) (h1 : x^2 + y^2 = 181) (h2 : x * y = 90) : x + y = 19 := by
  sorry

end sum_of_squares_and_product_l284_284991


namespace sum_of_fractions_l284_284557

theorem sum_of_fractions :
  (1 / 3) + (1 / 2) + (-5 / 6) + (1 / 5) + (1 / 4) + (-9 / 20) + (-9 / 20) = -9 / 20 := 
by
  sorry

end sum_of_fractions_l284_284557


namespace spherical_to_rectangular_coords_l284_284897

noncomputable def spherical_to_rectangular 
  (ρ θ φ : ℝ)  : ℝ × ℝ × ℝ :=
(ρ * sin φ * cos θ, ρ * sin φ * sin θ, ρ * cos φ)

theorem spherical_to_rectangular_coords 
  (hρ : ℝ := 10) (hθ : ℝ := 5 * Real.pi / 4) (hφ : ℝ := Real.pi / 4)
  (x y z : ℝ) :
  spherical_to_rectangular hρ hθ hφ = (-5, -5, 5 * Real.sqrt 2) :=
by
  sorry

end spherical_to_rectangular_coords_l284_284897


namespace teal_bakery_pumpkin_pie_l284_284230

theorem teal_bakery_pumpkin_pie (P : ℕ) 
    (pumpkin_price_per_slice : ℕ := 5)
    (custard_price_per_slice : ℕ := 6)
    (pumpkin_pies_sold : ℕ := 4)
    (custard_pies_sold : ℕ := 5)
    (custard_pieces_per_pie : ℕ := 6)
    (total_revenue : ℕ := 340) :
    4 * P * pumpkin_price_per_slice + custard_pies_sold * custard_pieces_per_pie * custard_price_per_slice = total_revenue → P = 8 := 
by
  sorry

end teal_bakery_pumpkin_pie_l284_284230


namespace eval_neg64_pow_two_thirds_l284_284035

theorem eval_neg64_pow_two_thirds : (-64 : Real)^(2/3) = 16 := 
by 
  sorry

end eval_neg64_pow_two_thirds_l284_284035


namespace at_least_one_zero_l284_284201

noncomputable def f (x : ℝ) (p q : ℝ) : ℝ := x^2 + p * x + q

theorem at_least_one_zero (p q : ℝ) (h_zero : ∃ m : ℝ, f m p q = 0 ∧ f (f (f m p q) p q) p q = 0) :
  f 0 p q = 0 ∨ f 1 p q = 0 :=
sorry

end at_least_one_zero_l284_284201


namespace max_possible_N_l284_284385

theorem max_possible_N (cities roads N : ℕ) (h1 : cities = 1000) (h2 : roads = 2017) (h3 : N = roads - (cities - 1 + 7 - 1)) :
  N = 1009 :=
by {
  sorry
}

end max_possible_N_l284_284385


namespace kelly_needs_more_apples_l284_284391

theorem kelly_needs_more_apples (initial_apples : ℕ) (total_apples : ℕ) (needed_apples : ℕ) :
  initial_apples = 128 → total_apples = 250 → needed_apples = total_apples - initial_apples → needed_apples = 122 :=
by
  intros h_initial h_total h_needed
  rw [h_initial, h_total] at h_needed
  exact h_needed

end kelly_needs_more_apples_l284_284391


namespace max_N_impassable_roads_l284_284386

def number_of_cities : ℕ := 1000
def number_of_roads : ℕ := 2017
def initial_connected_components : ℕ := 1
def target_connected_components : ℕ := 7

theorem max_N_impassable_roads 
    (h : ∀ (G : SimpleGraph (Fin 1000)), G.edgeCount = 2017 ∧ G.isConnected ∧ G.connectedComponents.card = target_connected_components) :
    ∃ N : ℕ, N = 993 :=
begin
  use 993,
  sorry
end

end max_N_impassable_roads_l284_284386


namespace neg_P_l284_284197

-- Define the proposition P
def P : Prop := ∃ x : ℝ, Real.exp x ≤ 0

-- State the negation of P
theorem neg_P : ¬P ↔ ∀ x : ℝ, Real.exp x > 0 := 
by 
  sorry

end neg_P_l284_284197


namespace additional_savings_is_297_l284_284009

-- Define initial order amount
def initial_order_amount : ℝ := 12000

-- Define the first set of discounts
def discount_scheme_1 (amount : ℝ) : ℝ :=
  let first_discount := amount * 0.75
  let second_discount := first_discount * 0.85
  let final_price := second_discount * 0.90
  final_price

-- Define the second set of discounts
def discount_scheme_2 (amount : ℝ) : ℝ :=
  let first_discount := amount * 0.70
  let second_discount := first_discount * 0.90
  let final_price := second_discount * 0.95
  final_price

-- Define the amount saved selecting the better discount scheme
def additional_savings : ℝ :=
  let final_price_1 := discount_scheme_1 initial_order_amount
  let final_price_2 := discount_scheme_2 initial_order_amount
  final_price_2 - final_price_1

-- Lean statement to prove the additional savings is $297
theorem additional_savings_is_297 : additional_savings = 297 := by
  sorry

end additional_savings_is_297_l284_284009


namespace tan_105_degree_l284_284615

theorem tan_105_degree : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 := by
  have h_add : ∀ a b : ℝ, Real.tan (a + b) = (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b) :=
    Real.tan_add

  have h_tan_60 : Real.tan (60 * Real.pi / 180) = Real.sqrt 3 := Real.tan_60
  have h_tan_45 : Real.tan (45 * Real.pi / 180) = 1 := Real.tan_45
  
  sorry

end tan_105_degree_l284_284615


namespace cows_problem_l284_284449

theorem cows_problem :
  ∃ (M X : ℕ), 
  (5 * M = X + 30) ∧ 
  (5 * M + X = 570) ∧ 
  M = 60 :=
by
  sorry

end cows_problem_l284_284449


namespace pages_in_first_chapter_l284_284564

theorem pages_in_first_chapter (x : ℕ) (h1 : x + 43 = 80) : x = 37 :=
by
  sorry

end pages_in_first_chapter_l284_284564


namespace steve_speed_back_l284_284266

theorem steve_speed_back :
  ∀ (d v_total : ℕ), d = 10 → v_total = 6 →
  (2 * (15 / 6)) = 5 :=
by
  intros d v_total d_eq v_total_eq
  sorry

end steve_speed_back_l284_284266


namespace tan_105_eq_neg2_sub_sqrt3_l284_284648

theorem tan_105_eq_neg2_sub_sqrt3 :
  Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_eq_neg2_sub_sqrt3_l284_284648


namespace negation_exists_l284_284983

theorem negation_exists (h : ∀ x : ℝ, 0 < x → Real.sin x < x) : ∃ x : ℝ, 0 < x ∧ Real.sin x ≥ x :=
by
  sorry

end negation_exists_l284_284983


namespace nina_running_distance_l284_284400

theorem nina_running_distance (x : ℝ) (hx : 2 * x + 0.67 = 0.83) : x = 0.08 := by
  sorry

end nina_running_distance_l284_284400


namespace ratio_cubed_eq_27_l284_284697

theorem ratio_cubed_eq_27 : (81000^3) / (27000^3) = 27 := 
by
  sorry

end ratio_cubed_eq_27_l284_284697


namespace reggie_games_lost_l284_284410

-- Given conditions:
def initial_marbles : ℕ := 100
def marbles_per_game : ℕ := 10
def games_played : ℕ := 9
def marbles_after_games : ℕ := 90

-- The statement to prove:
theorem reggie_games_lost : (initial_marbles - marbles_after_games) / marbles_per_game = 1 := 
sorry

end reggie_games_lost_l284_284410


namespace chosen_number_l284_284327

theorem chosen_number (x : ℕ) (h : 5 * x - 138 = 102) : x = 48 :=
sorry

end chosen_number_l284_284327


namespace acid_solution_replacement_percentage_l284_284109

theorem acid_solution_replacement_percentage 
  (original_concentration fraction_replaced final_concentration replaced_percentage : ℝ)
  (h₁ : original_concentration = 0.50)
  (h₂ : fraction_replaced = 0.5)
  (h₃ : final_concentration = 0.40)
  (h₄ : 0.25 + fraction_replaced * replaced_percentage = final_concentration) :
  replaced_percentage = 0.30 :=
by
  sorry

end acid_solution_replacement_percentage_l284_284109


namespace find_k_value_l284_284545

theorem find_k_value (k : ℚ) (h1 : (3, -5) ∈ {p : ℚ × ℚ | p.snd = k * p.fst}) (h2 : k ≠ 0) : k = -5 / 3 :=
sorry

end find_k_value_l284_284545


namespace max_sum_of_factors_l284_284138

theorem max_sum_of_factors (a b : ℕ) (h : a * b = 42) : a + b ≤ 43 :=
by
  -- sorry to skip the proof
  sorry

end max_sum_of_factors_l284_284138


namespace count_three_digit_numbers_l284_284216

open Finset

/-- The number of three-digit numbers where the hundreds digit is greater than the tens digit,
and the tens digit is greater than the ones digit. -/
theorem count_three_digit_numbers (U : set ℕ) (hU : U = {1, 2, 3, 4, 5, 6, 7, 8, 9}) :
  (U.to_finset.choose 3).card = 84 :=
by sorry

end count_three_digit_numbers_l284_284216


namespace time_to_carry_backpack_l284_284517

/-- 
Given:
1. Lara takes 73 seconds to crank open the door to the obstacle course.
2. Lara traverses the obstacle course the second time in 5 minutes and 58 seconds.
3. The total time to complete the obstacle course is 874 seconds.

Prove:
The time it took Lara to carry the backpack through the obstacle course the first time is 443 seconds.
-/
theorem time_to_carry_backpack (door_time : ℕ) (second_traversal_time : ℕ) (total_time : ℕ) : 
  (door_time + second_traversal_time + 443 = total_time) :=
by
  -- Given conditions
  let door_time := 73
  let second_traversal_time := 5 * 60 + 58 -- Convert 5 minutes 58 seconds to seconds
  let total_time := 874
  -- Calculate the time to carry the backpack
  sorry

end time_to_carry_backpack_l284_284517


namespace tan_105_degree_l284_284611

theorem tan_105_degree : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 := by
  have h_add : ∀ a b : ℝ, Real.tan (a + b) = (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b) :=
    Real.tan_add

  have h_tan_60 : Real.tan (60 * Real.pi / 180) = Real.sqrt 3 := Real.tan_60
  have h_tan_45 : Real.tan (45 * Real.pi / 180) = 1 := Real.tan_45
  
  sorry

end tan_105_degree_l284_284611


namespace tan_105_degree_l284_284634

theorem tan_105_degree : Real.tan (Real.pi * 105 / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_degree_l284_284634


namespace evaporation_rate_l284_284437

theorem evaporation_rate (initial_water_volume : ℕ) (days : ℕ) (percentage_evaporated : ℕ) (evaporated_fraction : ℚ)
  (h1 : initial_water_volume = 10)
  (h2 : days = 50)
  (h3 : percentage_evaporated = 3)
  (h4 : evaporated_fraction = percentage_evaporated / 100) :
  (initial_water_volume * evaporated_fraction) / days = 0.06 :=
by
  -- Proof goes here
  sorry

end evaporation_rate_l284_284437


namespace cos_seven_pi_six_eq_neg_sqrt_three_div_two_l284_284905

noncomputable def cos_seven_pi_six : Real :=
  Real.cos (7 * Real.pi / 6)

theorem cos_seven_pi_six_eq_neg_sqrt_three_div_two :
  cos_seven_pi_six = -Real.sqrt 3 / 2 :=
sorry

end cos_seven_pi_six_eq_neg_sqrt_three_div_two_l284_284905


namespace correct_option_l284_284367

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def is_decreasing_on_nonneg_real (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, 0 ≤ x₁ → 0 ≤ x₂ → x₁ < x₂ → f x₁ > f x₂

theorem correct_option (f : ℝ → ℝ) 
  (h_even : is_even_function f) 
  (h_decr : is_decreasing_on_nonneg_real f) :
  f 2 < f (-1) ∧ f (-1) < f 0 :=
by
  sorry

end correct_option_l284_284367


namespace cost_of_bread_l284_284526

-- Definition of the conditions
def total_purchase_amount : ℕ := 205  -- in cents
def amount_given_to_cashier : ℕ := 700  -- in cents
def quarter_value : ℕ := 25
def dime_value : ℕ := 10
def nickel_value : ℕ := 5
def num_nickels_received : ℕ := 8

-- Statement of the problem
theorem cost_of_bread :
  (∃ (B C : ℕ), B + C = total_purchase_amount ∧
                  amount_given_to_cashier - total_purchase_amount = 
                  (quarter_value + dime_value + num_nickels_received * nickel_value + 420) ∧
                  B = 125) :=
by
  -- Skipping the proof
  sorry

end cost_of_bread_l284_284526


namespace tangent_line_eq_k_2_f_is_decreasing_prove_an_condition_l284_284060

open Real

-- (I)
theorem tangent_line_eq_k_2 : 
  ∀ (f : ℝ → ℝ) (x : ℝ), 
    f x = ln (1 + x) - x + x^2 → 
    (∃ a b c, a = 3 ∧ b = -2 ∧ c = 2 * ln 2 - 3 ∧ ( ∀ y, f 1 = ln 2 → ((a * x + b * y + c) = 0))) :=
by sorry

-- (II)
theorem f_is_decreasing :
  ∀ k : ℝ,
    k ≥ 0 → k ≠ 1 →
    ((k = 0 → ∀ x > 0, deriv (λ x, ln (1 + x) - x + (k/2) * x^2) x < 0) ∧
    (0 < k ∧ k < 1 → ∀ x, x > 0 → x < (1 - k)/k → deriv (λ x, ln (1 + x) - x + (k/2) * x^2) x < 0) ∧
    (k > 1 → ∀ x, (1 - k)/k < x ∧ x < 0 → deriv (λ x, ln (1 + x) - x + (k/2) * x^2) x < 0)) :=
by sorry

-- (III)
theorem prove_an_condition :
  ∀ n : ℕ,
    n > 0 →
    ∀ (a_n b_n : ℝ),
      b_n = ln (1 + n) - n →
      a_n = ln (1 + n) - b_n →
      a_n = n →
      (∑ i in finset.range n, ∏ j in finset.range (2 * i + 1), a_n / (j + 1)) < sqrt (2 * a_n + 1) - 1 :=
by sorry

end tangent_line_eq_k_2_f_is_decreasing_prove_an_condition_l284_284060


namespace regular_15gon_symmetry_l284_284574

theorem regular_15gon_symmetry :
  ∀ (L R : ℕ),
  (L = 15) →
  (R = 24) →
  L + R = 39 :=
by
  intros L R hL hR
  exact sorry

end regular_15gon_symmetry_l284_284574


namespace additional_men_joined_l284_284311

noncomputable def solve_problem := 
  let M := 1000
  let days_initial := 17
  let days_new := 11.333333333333334
  let total_provisions := M * days_initial
  let additional_men := (total_provisions / days_new) - M
  additional_men

theorem additional_men_joined : solve_problem = 500 := by
  sorry

end additional_men_joined_l284_284311


namespace scientific_notation_l284_284453

def billion : ℝ := 10^9
def fifteenPointSeventyFiveBillion : ℝ := 15.75 * billion

theorem scientific_notation :
  fifteenPointSeventyFiveBillion = 1.575 * 10^10 :=
  sorry

end scientific_notation_l284_284453


namespace solution_to_system_l284_284718

theorem solution_to_system :
  (∀ (x y : ℚ), (y - x - 1 = 0) ∧ (y + x - 2 = 0) ↔ (x = 1/2 ∧ y = 3/2)) :=
by
  sorry

end solution_to_system_l284_284718


namespace tan_add_tan_105_eq_l284_284662

noncomputable def tan : ℝ → ℝ := sorry -- Use the built-in library later for actual implementation

-- Given conditions
def tan_45_eq : tan 45 = 1 := by sorry
def tan_60_eq : tan 60 = Real.sqrt 3 := by sorry

-- Angle addition formula for tangent
theorem tan_add (a b : ℝ) :
  tan (a + b) = (tan a + tan b) / (1 - tan a * tan b) := by sorry

-- Main theorem to prove
theorem tan_105_eq :
  tan 105 = -2 - Real.sqrt 3 := by sorry

end tan_add_tan_105_eq_l284_284662


namespace tan_105_eq_neg2_sub_sqrt3_l284_284641

theorem tan_105_eq_neg2_sub_sqrt3 :
  Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_eq_neg2_sub_sqrt3_l284_284641


namespace count_two_digit_primes_with_ones_digit_three_l284_284751

def is_prime (n : ℕ) : Prop := nat.prime n

def ones_digit_three (n : ℕ) : Prop := n % 10 = 3

def two_digit_number (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

theorem count_two_digit_primes_with_ones_digit_three : 
  {n : ℕ | two_digit_number n ∧ ones_digit_three n ∧ is_prime n}.to_finset.card = 6 :=
sorry

end count_two_digit_primes_with_ones_digit_three_l284_284751


namespace base7_to_base10_l284_284292

theorem base7_to_base10 : (2 * 7^4 + 3 * 7^3 + 4 * 7^2 + 5 * 7^1 + 6 * 7^0) = 6068 := by
  sorry

end base7_to_base10_l284_284292


namespace solve_for_y_l284_284975

-- Define the main theorem to be proven
theorem solve_for_y (y : ℤ) (h : 7 * (4 * y + 3) - 5 = -3 * (2 - 8 * y) + 5 * y) : y = 22 :=
by
  sorry

end solve_for_y_l284_284975


namespace h_at_7_over_5_eq_0_l284_284893

def h (x : ℝ) : ℝ := 5 * x - 7

theorem h_at_7_over_5_eq_0 : h (7 / 5) = 0 := 
by 
  sorry

end h_at_7_over_5_eq_0_l284_284893


namespace sam_more_than_avg_l284_284582

def bridget_count : ℕ := 14
def reginald_count : ℕ := bridget_count - 2
def sam_count : ℕ := reginald_count + 4
def average_count : ℕ := (bridget_count + reginald_count + sam_count) / 3

theorem sam_more_than_avg 
    (h1 : bridget_count = 14) 
    (h2 : reginald_count = bridget_count - 2) 
    (h3 : sam_count = reginald_count + 4) 
    (h4 : average_count = (bridget_count + reginald_count + sam_count) / 3): 
    sam_count - average_count = 2 := 
  sorry

end sam_more_than_avg_l284_284582


namespace total_cost_over_8_weeks_l284_284212

def cost_per_weekday_edition : ℝ := 0.50
def cost_per_sunday_edition : ℝ := 2.00
def num_weekday_editions_per_week : ℕ := 3
def duration_in_weeks : ℕ := 8

theorem total_cost_over_8_weeks :
  (num_weekday_editions_per_week * cost_per_weekday_edition + cost_per_sunday_edition) * duration_in_weeks = 28.00 := by
  sorry

end total_cost_over_8_weeks_l284_284212


namespace jim_saves_by_buying_gallon_l284_284084

-- Define the conditions as variables
def cost_per_gallon_costco : ℕ := 8
def ounces_per_gallon : ℕ := 128
def cost_per_16oz_bottle_store : ℕ := 3
def ounces_per_bottle : ℕ := 16

-- Define the theorem that needs to be proven
theorem jim_saves_by_buying_gallon (h1 : cost_per_gallon_costco = 8)
                                    (h2 : ounces_per_gallon = 128)
                                    (h3 : cost_per_16oz_bottle_store = 3)
                                    (h4 : ounces_per_bottle = 16) : 
  (8 * 3 - 8) = 16 :=
by sorry

end jim_saves_by_buying_gallon_l284_284084


namespace tan_105_eq_neg2_sub_sqrt3_l284_284693

-- Define the main theorem to be proven
theorem tan_105_eq_neg2_sub_sqrt3 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_eq_neg2_sub_sqrt3_l284_284693


namespace find_rate_l284_284158

def simple_interest_rate (P A T : ℕ) : ℕ :=
  ((A - P) * 100) / (P * T)

theorem find_rate :
  simple_interest_rate 750 1200 5 = 12 :=
by
  -- This is the statement of equality we need to prove
  sorry

end find_rate_l284_284158


namespace angle_733_in_first_quadrant_l284_284016

def in_first_quadrant (θ : ℝ) : Prop := 
  0 < θ ∧ θ < 90

theorem angle_733_in_first_quadrant :
  in_first_quadrant (733 % 360 : ℝ) :=
sorry

end angle_733_in_first_quadrant_l284_284016


namespace sum_of_roots_l284_284396

variable {h b : ℝ}
variable {x₁ x₂ : ℝ}

-- Definition of the distinct property
def distinct (x₁ x₂ : ℝ) : Prop := x₁ ≠ x₂

-- Definition of the original equations given the conditions
def satisfies_equation (x : ℝ) (h b : ℝ) : Prop := 3 * x^2 - h * x = b

-- Main theorem statement translating the given mathematical problem
theorem sum_of_roots (h b : ℝ) (x₁ x₂ : ℝ) (h₁ : satisfies_equation x₁ h b) 
  (h₂ : satisfies_equation x₂ h b) (h₃ : distinct x₁ x₂) : x₁ + x₂ = h / 3 :=
sorry

end sum_of_roots_l284_284396


namespace max_digit_sum_l284_284568

-- Define the condition for the hours and minutes digits
def is_valid_hour (h : ℕ) := 0 ≤ h ∧ h < 24
def is_valid_minute (m : ℕ) := 0 ≤ m ∧ m < 60

-- Define the function to calculate the sum of the digits of a two-digit number
def sum_of_digits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

-- Main statement: Prove that the maximum sum of the digits in the display is 24
theorem max_digit_sum : ∃ h m: ℕ, is_valid_hour h ∧ is_valid_minute m ∧ 
  sum_of_digits h + sum_of_digits m = 24 :=
sorry

end max_digit_sum_l284_284568


namespace paths_A_to_D_l284_284712

noncomputable def num_paths_from_A_to_D : ℕ := 
  2 * 2 * 2 + 1

theorem paths_A_to_D : num_paths_from_A_to_D = 9 := 
by
  sorry

end paths_A_to_D_l284_284712


namespace pascal_triangle_43rd_element_in_51_row_l284_284556

theorem pascal_triangle_43rd_element_in_51_row :
  (Nat.choose 50 42) = 10272278170 :=
  by
  -- proof construction here
  sorry

end pascal_triangle_43rd_element_in_51_row_l284_284556


namespace max_distinct_tangent_counts_l284_284401

-- Define the types and conditions for our circles and tangents
structure Circle where
  radius : ℝ

def circle1 : Circle := { radius := 3 }
def circle2 : Circle := { radius := 4 }

-- Define the statement to be proved
theorem max_distinct_tangent_counts :
  ∃ (k : ℕ), k = 5 :=
sorry

end max_distinct_tangent_counts_l284_284401


namespace carA_arrangements_l284_284705

-- Define the conditions of the problem
def students := {1, 2, 3, 4, 5, 6, 7, 8} -- 8 students
def grades := {1, 2, 3, 4} -- 4 grade levels
def carA := {a // a ∈ students} -- Car A
def carB := {b // b ∈ students} -- Car B

-- Define the freshman twin sisters
def twinSisters := {1, 2}

-- Number of ways to arrange students in Car A to have exactly 2 students from the same grade
def arrangements_in_carA_with_same_grade : ℕ :=
  let twinsInCarA := 3 * 4 -- Case 1
  let twinsNotInCarA := 3 * 4 -- Case 2
  twinsInCarA + twinsNotInCarA

-- Proof statement
theorem carA_arrangements : arrangements_in_carA_with_same_grade = 24 :=
  sorry

end carA_arrangements_l284_284705


namespace range_of_2_cos_sq_l284_284118

theorem range_of_2_cos_sq :
  ∀ x : ℝ, 0 ≤ 2 * (Real.cos x) ^ 2 ∧ 2 * (Real.cos x) ^ 2 ≤ 2 :=
by sorry

end range_of_2_cos_sq_l284_284118


namespace mn_value_l284_284819
open Real

-- Define the conditions
def L_1_scenario_1 (m n : ℝ) : Prop :=
  ∃ (θ₁ θ₂ : ℝ), θ₁ = 2 * θ₂ ∧ m = tan θ₁ ∧ n = tan θ₂ ∧ m = 4 * n

-- State the theorem
theorem mn_value (m n : ℝ) (hL1 : L_1_scenario_1 m n) (hm : m ≠ 0) : m * n = 2 :=
  sorry

end mn_value_l284_284819


namespace inf_arith_seq_contains_inf_geo_seq_l284_284971

-- Condition: Infinite arithmetic sequence of natural numbers
variable (a d : ℕ) (h : ∀ n : ℕ, n ≥ 1 → ∃ k : ℕ, k = a + (n - 1) * d)

-- Theorem: There exists an infinite geometric sequence within the arithmetic sequence
theorem inf_arith_seq_contains_inf_geo_seq :
  ∃ r : ℕ, ∀ n : ℕ, ∃ k : ℕ, k = a * r ^ (n - 1) := sorry

end inf_arith_seq_contains_inf_geo_seq_l284_284971


namespace value_of_a_minus_b_l284_284976

theorem value_of_a_minus_b (a b : ℤ) (h1 : 2020 * a + 2024 * b = 2040) (h2 : 2022 * a + 2026 * b = 2044) :
  a - b = 1002 :=
sorry

end value_of_a_minus_b_l284_284976


namespace parallel_lines_slope_l284_284379

theorem parallel_lines_slope (m : ℝ) :
  ((m + 2) * (2 * m - 1) = 3 * 1) →
  m = - (5 / 2) :=
by
  sorry

end parallel_lines_slope_l284_284379


namespace sum_of_integers_l284_284989

/-- Given two positive integers x and y such that the sum of their squares equals 181 
    and their product equals 90, prove that the sum of these two integers is 19. -/
theorem sum_of_integers (x y : ℤ) (h1 : x^2 + y^2 = 181) (h2 : x * y = 90) : x + y = 19 := by
  sorry

end sum_of_integers_l284_284989


namespace twentieth_fisherman_catch_l284_284999

theorem twentieth_fisherman_catch (total_fishermen : ℕ) (total_fish : ℕ) (fish_per_19 : ℕ) (fish_each_19 : ℕ) (h1 : total_fishermen = 20) (h2 : total_fish = 10000) (h3 : fish_per_19 = 19 * 400) (h4 : fish_each_19 = 400) : 
  fish_per_19 + fish_each_19 = total_fish := by
  sorry

end twentieth_fisherman_catch_l284_284999


namespace alice_bracelets_given_away_l284_284155

theorem alice_bracelets_given_away
    (total_bracelets : ℕ)
    (cost_of_materials : ℝ)
    (price_per_bracelet : ℝ)
    (profit : ℝ)
    (bracelets_given_away : ℕ)
    (bracelets_sold : ℕ)
    (total_revenue : ℝ)
    (h1 : total_bracelets = 52)
    (h2 : cost_of_materials = 3)
    (h3 : price_per_bracelet = 0.25)
    (h4 : profit = 8)
    (h5 : total_revenue = profit + cost_of_materials)
    (h6 : total_revenue = price_per_bracelet * bracelets_sold)
    (h7 : total_bracelets = bracelets_sold + bracelets_given_away) :
    bracelets_given_away = 8 :=
by
  sorry

end alice_bracelets_given_away_l284_284155


namespace find_k_l284_284360

theorem find_k (k : ℚ) :
  (5 + ∑' n : ℕ, (5 + 2*k*(n+1)) / 4^n) = 10 → k = 15/4 :=
by
  sorry

end find_k_l284_284360


namespace value_of_c_distinct_real_roots_l284_284181

-- Define the quadratic equation and the condition for having two distinct real roots
def quadratic_eqn (c : ℝ) : Prop :=
  ∀ (x : ℝ), x^2 + 2 * x + 4 * c = 0

def two_distinct_real_roots (c : ℝ) : Prop :=
  let Δ := 2^2 - 4 * 1 * (4 * c) in Δ > 0

-- The proof problem statement
theorem value_of_c_distinct_real_roots (c : ℝ) : c < 1 / 4 :=
by
  have h_discriminant : 4 - 16 * c > 0 :=
    calc
      4 - 16 * c = 4 - 16 * c : by ring
      ... > 0 : sorry
  have h_c_lt : c < 1 / 4 :=
    calc
      c < 1 / 4 : sorry
  exact h_c_lt

end value_of_c_distinct_real_roots_l284_284181


namespace two_pow_m_minus_one_not_divide_three_pow_n_minus_one_l284_284245

open Nat

theorem two_pow_m_minus_one_not_divide_three_pow_n_minus_one 
  (m n : ℕ) (hm : m ≥ 3) (hn : n ≥ 3) (hmo : Odd m) (hno : Odd n) : ¬ (∃ k : ℕ, 2^m - 1 = k * (3^n - 1)) := by
  sorry

end two_pow_m_minus_one_not_divide_three_pow_n_minus_one_l284_284245


namespace radius_increase_l284_284433

theorem radius_increase (C₁ C₂ : ℝ) (C₁_eq : C₁ = 30) (C₂_eq : C₂ = 40) :
  let r₁ := C₁ / (2 * Real.pi)
  let r₂ := C₂ / (2 * Real.pi)
  r₂ - r₁ = 5 / Real.pi :=
by
  simp [C₁_eq, C₂_eq]
  sorry

end radius_increase_l284_284433


namespace tan_105_eq_minus_2_minus_sqrt_3_l284_284679

theorem tan_105_eq_minus_2_minus_sqrt_3 :
  Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_eq_minus_2_minus_sqrt_3_l284_284679


namespace constant_term_binomial_expansion_l284_284536

theorem constant_term_binomial_expansion :
  ∀ (x : ℝ), ((2 / x) + x) ^ 4 = 24 :=
by
  sorry

end constant_term_binomial_expansion_l284_284536


namespace tan_105_degree_l284_284636

theorem tan_105_degree : Real.tan (Real.pi * 105 / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_degree_l284_284636


namespace simplify_fraction_l284_284804

theorem simplify_fraction (a b : ℕ) (h : a = 180) (k : b = 270) : 
  ∃ c d, c = 2 ∧ d = 3 ∧ (a / (Nat.gcd a b) = c) ∧ (b / (Nat.gcd a b) = d) :=
by
  sorry

end simplify_fraction_l284_284804


namespace f_five_l284_284099

noncomputable def f (x : ℝ) : ℝ := sorry

axiom odd_function : ∀ x : ℝ, f (-x) = - f x
axiom f_one : f 1 = 1 / 2
axiom functional_equation : ∀ x : ℝ, f (x + 2) = f x + f 2

theorem f_five : f 5 = 5 / 2 :=
by sorry

end f_five_l284_284099


namespace tan_105_degree_l284_284613

theorem tan_105_degree : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 := by
  have h_add : ∀ a b : ℝ, Real.tan (a + b) = (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b) :=
    Real.tan_add

  have h_tan_60 : Real.tan (60 * Real.pi / 180) = Real.sqrt 3 := Real.tan_60
  have h_tan_45 : Real.tan (45 * Real.pi / 180) = 1 := Real.tan_45
  
  sorry

end tan_105_degree_l284_284613


namespace towel_decrease_percentage_l284_284306

variable (L B : ℝ)
variable (h1 : 0.70 * L = L - (0.30 * L))
variable (h2 : 0.60 * B = B - (0.40 * B))

theorem towel_decrease_percentage (L B : ℝ) 
  (h1 : 0.70 * L = L - (0.30 * L))
  (h2 : 0.60 * B = B - (0.40 * B)) :
  ((L * B - (0.70 * L) * (0.60 * B)) / (L * B)) * 100 = 58 := 
by
  sorry

end towel_decrease_percentage_l284_284306


namespace geometric_inequality_l284_284264

-- Define the geometric setup
variables {A B C H B1 C1 N M Ob Oc : ℝ} -- representing points in ℝ

-- Conditions from the problem statement
def altitude_intersect (ABC : Triangle) (H : Point) :=
  is_altitude ABC B H ∧ is_altitude ABC C H

def circle_passing_points (O : Point) (A C1 N : Point) :=
  is_circle O [A, C1, N]

def midpoint (X Y M : Point) :=
  2 * (line_segment X Y) = line_segment X M + line_segment M Y

-- Main theorem statement
theorem geometric_inequality 
  (ABC : Triangle) 
  (H : Point) 
  (Ob : Point) 
  (Oc : Point)
  (B1 : Point) 
  (C1 : Point)
  (N : Point) 
  (M : Point) 
  (altitudes : altitude_intersect ABC H)
  (circle_ob : circle_passing_points Ob A C1 N)
  (circle_oc : circle_passing_points Oc A B1 M)
  (midpt_BH : midpoint B H N)
  (midpt_CH : midpoint C H M) :
  (distance B1 Ob + distance C1 Oc) > (distance (point_of_line_segment B C) * (4⁻¹)) :=
begin
  sorry,
end

end geometric_inequality_l284_284264


namespace example_function_not_power_function_l284_284422

-- Definition of a power function
def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, f x = x ^ α

-- Define the function y = 2x^(1/2)
def example_function (x : ℝ) : ℝ :=
  2 * x ^ (1 / 2)

-- The statement we want to prove
theorem example_function_not_power_function : ¬ is_power_function example_function := by
  sorry

end example_function_not_power_function_l284_284422


namespace lisa_needs_4_weeks_to_eat_all_candies_l284_284248

-- Define the number of candies Lisa has initially.
def candies_initial : ℕ := 72

-- Define the number of candies Lisa eats per week based on the given conditions.
def candies_per_week : ℕ := (3 * 2) + (2 * 2) + (4 * 2) + 1

-- Define the number of weeks it takes for Lisa to eat all the candies.
def weeks_to_eat_all_candies (candies : ℕ) (weekly_candies : ℕ) : ℕ := 
  (candies + weekly_candies - 1) / weekly_candies

-- The theorem statement that proves Lisa needs 4 weeks to eat all 72 candies.
theorem lisa_needs_4_weeks_to_eat_all_candies :
  weeks_to_eat_all_candies candies_initial candies_per_week = 4 :=
by
  sorry

end lisa_needs_4_weeks_to_eat_all_candies_l284_284248


namespace haley_fuel_consumption_ratio_l284_284063

theorem haley_fuel_consumption_ratio (gallons: ℕ) (miles: ℕ) (h_gallons: gallons = 44) (h_miles: miles = 77) :
  (gallons / Nat.gcd gallons miles) = 4 ∧ (miles / Nat.gcd gallons miles) = 7 :=
by
  sorry

end haley_fuel_consumption_ratio_l284_284063


namespace full_time_employees_l284_284324

theorem full_time_employees (total_employees part_time_employees number_full_time_employees : ℕ)
  (h1 : total_employees = 65134)
  (h2 : part_time_employees = 2041)
  (h3 : number_full_time_employees = total_employees - part_time_employees)
  : number_full_time_employees = 63093 :=
by {
  sorry
}

end full_time_employees_l284_284324


namespace smallest_N_satisfying_conditions_l284_284326

def is_divisible (n m : ℕ) : Prop :=
  m ∣ n

def satisfies_conditions (N : ℕ) : Prop :=
  (is_divisible N 10) ∧
  (is_divisible N 5) ∧
  (N > 15)

theorem smallest_N_satisfying_conditions : ∃ N, satisfies_conditions N ∧ N = 20 := 
  sorry

end smallest_N_satisfying_conditions_l284_284326


namespace shape_is_plane_l284_284170

noncomputable
def cylindrical_coordinates_shape (r θ z c : ℝ) := θ = 2 * c

theorem shape_is_plane (c : ℝ) : 
  ∀ (r : ℝ) (θ : ℝ) (z : ℝ), cylindrical_coordinates_shape r θ z c → (θ = 2 * c) :=
by
  sorry

end shape_is_plane_l284_284170


namespace F_2457_find_Q_l284_284361

-- Define the properties of a "rising number"
def is_rising_number (m : ℕ) : Prop :=
  ∃ (a b c d : ℕ),
    m = 1000 * a + 100 * b + 10 * c + d ∧
    a < b ∧ b < c ∧ c < d ∧
    a + d = b + c

-- Define F(m) as specified
def F (m : ℕ) : ℤ :=
  let a := m / 1000
  let b := (m / 100) % 10
  let c := (m / 10) % 10
  let d := m % 10
  let m' := 1000 * c + 100 * b + 10 * a + d
  (m' - m) / 99

-- Problem statement for F(2457)
theorem F_2457 : F 2457 = 30 := sorry

-- Properties given in the problem statement for P and Q
def is_specific_rising_number (P Q : ℕ) : Prop :=
  ∃ (x y z t : ℕ),
    P = 1000 + 100 * x + 10 * y + z ∧
    Q = 1000 * x + 100 * t + 60 + z ∧
    1 < x ∧ x < t ∧ t < 6 ∧ 6 < z ∧
    1 + z = x + y ∧
    x + z = t + 6 ∧
    F P + F Q % 7 = 0

-- Problem statement to find the value of Q
theorem find_Q (Q : ℕ) : 
  ∃ (P : ℕ), is_specific_rising_number P Q ∧ Q = 3467 := sorry

end F_2457_find_Q_l284_284361


namespace range_of_y_div_x_l284_284241

theorem range_of_y_div_x (x y : ℝ) (h : x^2 + y^2 + 4*x + 3 = 0) :
  - (Real.sqrt 3) / 3 <= y / x ∧ y / x <= (Real.sqrt 3) / 3 :=
sorry

end range_of_y_div_x_l284_284241


namespace number_of_two_digit_primes_with_ones_digit_3_l284_284747

-- Definition of two-digit numbers with a ones digit of 3
def two_digit_numbers_with_ones_digit_3 := [13, 23, 33, 43, 53, 63, 73, 83, 93]

-- Definition of prime predicate
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Proof statement
theorem number_of_two_digit_primes_with_ones_digit_3 : 
  let primes := (two_digit_numbers_with_ones_digit_3.filter is_prime) in
  primes.length = 7 :=
by
  sorry

end number_of_two_digit_primes_with_ones_digit_3_l284_284747


namespace number_of_paths_to_spell_MATH_l284_284506

-- Define the problem setting and conditions
def number_of_paths_M_to_H (adj: ℕ) (steps: ℕ): ℕ :=
  adj^(steps-1)

-- State the problem in Lean 4
theorem number_of_paths_to_spell_MATH : number_of_paths_M_to_H 8 4 = 512 := 
by 
  unfold number_of_paths_M_to_H 
  -- The needed steps are included:
  -- We calculate: 8^(4-1) = 8^3 which should be 512.
  sorry

end number_of_paths_to_spell_MATH_l284_284506


namespace cos_7pi_over_6_eq_neg_sqrt3_over_2_l284_284916

theorem cos_7pi_over_6_eq_neg_sqrt3_over_2 : Real.cos (7 * Real.pi / 6) = - (Real.sqrt 3) / 2 := by
  sorry

end cos_7pi_over_6_eq_neg_sqrt3_over_2_l284_284916


namespace tan_105_degree_l284_284609

theorem tan_105_degree : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 := by
  have h_add : ∀ a b : ℝ, Real.tan (a + b) = (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b) :=
    Real.tan_add

  have h_tan_60 : Real.tan (60 * Real.pi / 180) = Real.sqrt 3 := Real.tan_60
  have h_tan_45 : Real.tan (45 * Real.pi / 180) = 1 := Real.tan_45
  
  sorry

end tan_105_degree_l284_284609


namespace Raine_total_steps_l284_284254

-- Define the steps taken to and from school each day
def Monday_steps_to_school := 150
def Monday_steps_back := 170
def Tuesday_steps_to_school := 140
def Tuesday_steps_back := 140 + 30
def Wednesday_steps_to_school := 160
def Wednesday_steps_back := 210
def Thursday_steps_to_school := 150
def Thursday_steps_back := 140 + 30
def Friday_steps_to_school := 180
def Friday_steps_back := 200

-- Define total steps for each day
def Monday_total_steps := Monday_steps_to_school + Monday_steps_back
def Tuesday_total_steps := Tuesday_steps_to_school + Tuesday_steps_back
def Wednesday_total_steps := Wednesday_steps_to_school + Wednesday_steps_back
def Thursday_total_steps := Thursday_steps_to_school + Thursday_steps_back
def Friday_total_steps := Friday_steps_to_school + Friday_steps_back

-- Define the total steps for all five days
def total_steps :=
  Monday_total_steps +
  Tuesday_total_steps +
  Wednesday_total_steps +
  Thursday_total_steps +
  Friday_total_steps

-- Prove that the total steps equals 1700
theorem Raine_total_steps : total_steps = 1700 := 
by 
  unfold total_steps
  unfold Monday_total_steps Tuesday_total_steps Wednesday_total_steps Thursday_total_steps Friday_total_steps
  unfold Monday_steps_to_school Monday_steps_back
  unfold Tuesday_steps_to_school Tuesday_steps_back
  unfold Wednesday_steps_to_school Wednesday_steps_back
  unfold Thursday_steps_to_school Thursday_steps_back
  unfold Friday_steps_to_school Friday_steps_back
  sorry

end Raine_total_steps_l284_284254


namespace doug_initial_marbles_l284_284902

theorem doug_initial_marbles (ed_marbles : ℕ) (diff_ed_doug : ℕ) (final_ed_marbles : ed_marbles = 27) (diff : diff_ed_doug = 5) :
  ∃ doug_initial_marbles : ℕ, doug_initial_marbles = 22 :=
by
  sorry

end doug_initial_marbles_l284_284902


namespace cos_seven_pi_six_eq_neg_sqrt_three_div_two_l284_284904

noncomputable def cos_seven_pi_six : Real :=
  Real.cos (7 * Real.pi / 6)

theorem cos_seven_pi_six_eq_neg_sqrt_three_div_two :
  cos_seven_pi_six = -Real.sqrt 3 / 2 :=
sorry

end cos_seven_pi_six_eq_neg_sqrt_three_div_two_l284_284904


namespace tan_105_eq_neg_2_sub_sqrt_3_l284_284619

-- Definitions
def angle105 : ℝ := 105 * (Math.pi / 180)
def angle45 : ℝ := 45 * (Math.pi / 180)
def angle60 : ℝ := 60 * (Math.pi / 180)

-- Theorem
theorem tan_105_eq_neg_2_sub_sqrt_3 :
  Real.tan angle105 = -2 - Real.sqrt 3 := by
  sorry

end tan_105_eq_neg_2_sub_sqrt_3_l284_284619


namespace screamers_lineups_l284_284535

/-- The Screamers have 15 players. Bob, Yogi, and Zane refuse to play together in any combination.
    Determine the number of starting lineups of 6 players not including all of Bob, Yogi, and Zane. -/
theorem screamers_lineups (Bob Yogi Zane : Fin 15) :
  ¬ (Bob = Yogi ∧ Yogi = Zane) →
  (∑ i, if i = Bob ∨ i = Yogi ∨ i = Zane then 0 else 1) = 12 →
  (Finset.card {l : Finset (Fin 15) | ∃ b y z, 
    (b ∈ l → b = Bob) ∧ 
    (y ∈ l → y = Yogi) ∧
    (z ∈ l → z = Zane) ∧ 
    Finset.card l = 6 } = 3300) :=
by
  intros h1 h2
  sorry

end screamers_lineups_l284_284535


namespace find_x_l284_284076

theorem find_x (x : ℝ) (h1 : (x - 1) / (x + 2) = 0) (h2 : x ≠ -2) : x = 1 :=
sorry

end find_x_l284_284076


namespace average_earnings_per_minute_l284_284827

theorem average_earnings_per_minute (race_duration : ℕ) (lap_distance : ℕ) (certificate_rate : ℝ) (laps_run : ℕ) :
  race_duration = 12 → 
  lap_distance = 100 → 
  certificate_rate = 3.5 → 
  laps_run = 24 → 
  ((laps_run * lap_distance / 100) * certificate_rate) / race_duration = 7 :=
by
  intros hrace_duration hlap_distance hcertificate_rate hlaps_run
  rw [hrace_duration, hlap_distance, hcertificate_rate, hlaps_run]
  sorry

end average_earnings_per_minute_l284_284827


namespace tan_105_eq_neg2_sub_sqrt3_l284_284691

-- Define the main theorem to be proven
theorem tan_105_eq_neg2_sub_sqrt3 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_eq_neg2_sub_sqrt3_l284_284691


namespace antonio_correct_answers_l284_284226

theorem antonio_correct_answers :
  ∃ c w : ℕ, c + w = 15 ∧ 6 * c - 3 * w = 36 ∧ c = 9 :=
by
  sorry

end antonio_correct_answers_l284_284226


namespace total_birds_on_fence_l284_284120

theorem total_birds_on_fence (initial_pairs : ℕ) (birds_per_pair : ℕ) 
                             (new_pairs : ℕ) (new_birds_per_pair : ℕ)
                             (initial_birds : initial_pairs * birds_per_pair = 24)
                             (new_birds : new_pairs * new_birds_per_pair = 8) : 
                             ((initial_pairs * birds_per_pair) + (new_pairs * new_birds_per_pair) = 32) :=
sorry

end total_birds_on_fence_l284_284120


namespace area_bounded_by_graphs_l284_284457

noncomputable def compute_area : ℝ :=
  ∫ x in (0 : ℝ) .. 1, real.sqrt (4 - x^2)

theorem area_bounded_by_graphs :
  compute_area = (real.pi / 3) + (real.sqrt 3 / 2) :=
by
  sorry

end area_bounded_by_graphs_l284_284457


namespace Sam_dimes_remaining_l284_284256

-- Define the initial and borrowed dimes
def initial_dimes_count : Nat := 8
def borrowed_dimes_count : Nat := 4

-- State the theorem
theorem Sam_dimes_remaining : (initial_dimes_count - borrowed_dimes_count) = 4 := by
  sorry

end Sam_dimes_remaining_l284_284256


namespace twentieth_fisherman_caught_l284_284996

-- Definitions based on conditions
def fishermen_count : ℕ := 20
def total_fish_caught : ℕ := 10000
def fish_per_nineteen_fishermen : ℕ := 400
def nineteen_count : ℕ := 19

-- Calculation based on the problem conditions
def total_fish_by_nineteen : ℕ := nineteen_count * fish_per_nineteen_fishermen

-- Prove the number of fish caught by the twentieth fisherman
theorem twentieth_fisherman_caught : 
  total_fish_caught - total_fish_by_nineteen = 2400 := 
by
  -- This is where the proof would go
  sorry

end twentieth_fisherman_caught_l284_284996


namespace min_area_OBX_l284_284000

structure Point : Type :=
  (x : ℤ)
  (y : ℤ)

def O : Point := ⟨0, 0⟩
def B : Point := ⟨11, 8⟩

def area_triangle (A B C : Point) : ℚ :=
  1 / 2 * abs (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y))

def in_rectangle (X : Point) : Prop :=
  0 ≤ X.x ∧ X.x ≤ 11 ∧ 0 ≤ X.y ∧ X.y ≤ 8

theorem min_area_OBX : ∃ (X : Point), in_rectangle X ∧ area_triangle O B X = 1 / 2 :=
sorry

end min_area_OBX_l284_284000


namespace count_two_digit_primes_ending_with_3_l284_284764

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m * m ≤ n → n % m ≠ 0

def two_digit (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100

def ends_with_3 (n : ℕ) : Prop :=
  n % 10 = 3

theorem count_two_digit_primes_ending_with_3 :
  {n : ℕ | two_digit n ∧ ends_with_3 n ∧ is_prime n}.to_finset.card = 6 := by
sorry

end count_two_digit_primes_ending_with_3_l284_284764


namespace tan_105_degree_is_neg_sqrt3_minus_2_l284_284653

theorem tan_105_degree_is_neg_sqrt3_minus_2 :
  Real.tan (105 * Real.pi / 180) = -(Real.sqrt 3 + 2) := by
  sorry

end tan_105_degree_is_neg_sqrt3_minus_2_l284_284653


namespace car_b_speed_l284_284890

noncomputable def SpeedOfCarB (Speed_A Time_A Time_B d_ratio: ℝ) : ℝ :=
  let Distance_A := Speed_A * Time_A
  let Distance_B := Distance_A / d_ratio
  Distance_B / Time_B

theorem car_b_speed
  (Speed_A : ℝ) (Time_A : ℝ) (Time_B : ℝ) (d_ratio : ℝ)
  (h1 : Speed_A = 70) (h2 : Time_A = 10) (h3 : Time_B = 10) (h4 : d_ratio = 2) :
  SpeedOfCarB Speed_A Time_A Time_B d_ratio = 35 :=
by
  sorry

end car_b_speed_l284_284890


namespace range_of_a_l284_284933

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 - a*x + 2*a > 0) → 0 < a ∧ a < 8 := 
sorry

end range_of_a_l284_284933


namespace range_of_function_l284_284546

noncomputable def function_y (x : ℝ) : ℝ := -x^2 - 2 * x + 3

theorem range_of_function : 
  ∃ (a b : ℝ), a = -12 ∧ b = 4 ∧ 
  (∀ y, (∃ x, -5 ≤ x ∧ x ≤ 0 ∧ y = function_y x) ↔ a ≤ y ∧ y ≤ b) :=
sorry

end range_of_function_l284_284546


namespace fractions_zero_condition_l284_284802

variable {a b c : ℝ}

theorem fractions_zero_condition 
  (h : (a - b) / (1 + a * b) + (b - c) / (1 + b * c) + (c - a) / (1 + c * a) = 0) :
  (a - b) / (1 + a * b) = 0 ∨ (b - c) / (1 + b * c) = 0 ∨ (c - a) / (1 + c * a) = 0 := 
sorry

end fractions_zero_condition_l284_284802


namespace no_solution_fraction_eq_l284_284496

theorem no_solution_fraction_eq (m : ℝ) : 
  ¬(∃ x : ℝ, x ≠ -1 ∧ 3 * x / (x + 1) = m / (x + 1) + 2) ↔ m = -3 :=
by
  sorry

end no_solution_fraction_eq_l284_284496


namespace fewer_mpg_in_city_l284_284144

def city_mpg := 14
def city_distance := 336
def highway_distance := 480

def tank_size := city_distance / city_mpg
def highway_mpg := highway_distance / tank_size
def fewer_mpg := highway_mpg - city_mpg

theorem fewer_mpg_in_city : fewer_mpg = 6 := by
  sorry

end fewer_mpg_in_city_l284_284144


namespace days_in_month_l284_284567

theorem days_in_month
  (monthly_production : ℕ)
  (production_per_half_hour : ℚ)
  (hours_per_day : ℕ)
  (daily_production : ℚ)
  (days_in_month : ℚ) :
  monthly_production = 8400 ∧
  production_per_half_hour = 6.25 ∧
  hours_per_day = 24 ∧
  daily_production = production_per_half_hour * 2 * hours_per_day ∧
  days_in_month = monthly_production / daily_production
  → days_in_month = 28 :=
by
  sorry

end days_in_month_l284_284567


namespace michael_digging_time_equals_700_l284_284797

-- Conditions defined
def digging_rate := 4
def father_depth := digging_rate * 400
def michael_depth := 2 * father_depth - 400
def time_for_michael := michael_depth / digging_rate

-- Statement to prove
theorem michael_digging_time_equals_700 : time_for_michael = 700 :=
by
  -- Here we would provide the proof steps, but we use sorry for now
  sorry

end michael_digging_time_equals_700_l284_284797


namespace two_digit_primes_with_ones_digit_three_count_l284_284750

def is_two_digit (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100

def has_ones_digit_three (n : ℕ) : Prop :=
  n % 10 = 3

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def number_of_two_digit_primes_with_ones_digit_three : ℕ :=
  6

theorem two_digit_primes_with_ones_digit_three_count :
  number_of_two_digit_primes_with_ones_digit_three =
  (finset.filter (λ n, is_two_digit n ∧ has_ones_digit_three n ∧ is_prime n)
                 (finset.range 100)).card :=
by
  sorry

end two_digit_primes_with_ones_digit_three_count_l284_284750


namespace tan_105_eq_minus_2_minus_sqrt_3_l284_284680

theorem tan_105_eq_minus_2_minus_sqrt_3 :
  Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_eq_minus_2_minus_sqrt_3_l284_284680


namespace population_ratio_l284_284023

theorem population_ratio
  (P_A P_B P_C P_D P_E P_F : ℕ)
  (h1 : P_A = 8 * P_B)
  (h2 : P_B = 5 * P_C)
  (h3 : P_D = 3 * P_C)
  (h4 : P_D = P_E / 2)
  (h5 : P_F = P_A / 4) :
  P_E / P_B = 6 / 5 := by
    sorry

end population_ratio_l284_284023


namespace positive_difference_of_complementary_angles_l284_284268

-- Define the conditions
variables (a b : ℝ)
variable (h1 : a + b = 90)
variable (h2 : 5 * b = a)

-- Define the theorem we are proving
theorem positive_difference_of_complementary_angles (a b : ℝ) (h1 : a + b = 90) (h2 : 5 * b = a) :
  |a - b| = 60 :=
by
  sorry

end positive_difference_of_complementary_angles_l284_284268


namespace time_to_cross_second_platform_l284_284328

-- Definition of the conditions
variables (l_train l_platform1 l_platform2 t1 : ℕ)
variable (v : ℕ)

-- The conditions given in the problem
def conditions : Prop :=
  l_train = 190 ∧
  l_platform1 = 140 ∧
  l_platform2 = 250 ∧
  t1 = 15 ∧
  v = (l_train + l_platform1) / t1

-- The statement to prove
theorem time_to_cross_second_platform
    (l_train l_platform1 l_platform2 t1 : ℕ)
    (v : ℕ)
    (h : conditions l_train l_platform1 l_platform2 t1 v) :
    (l_train + l_platform2) / v = 20 :=
  sorry

end time_to_cross_second_platform_l284_284328


namespace proportion_correct_l284_284219

theorem proportion_correct (m n : ℤ) (h : 6 * m = 7 * n) (hn : n ≠ 0) : (m : ℚ) / 7 = n / 6 :=
by sorry

end proportion_correct_l284_284219


namespace quadratic_distinct_roots_l284_284183

theorem quadratic_distinct_roots (c : ℝ) : (∀ (x : ℝ), x^2 + 2 * x + 4 * c = 0 → x ∈ ℝ) ∧ (∃ x y : ℝ, x ≠ y) → c < 1 / 4 :=
by
  sorry

end quadratic_distinct_roots_l284_284183


namespace count_two_digit_primes_with_ones_3_l284_284757

open Nat

/-- Predicate to check if a number is a two-digit prime with ones digit 3. --/
def two_digit_prime_with_ones_3 (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ n % 10 = 3 ∧ Prime n

/-- Prove that there are exactly 6 two-digit primes with ones digit 3. --/
theorem count_two_digit_primes_with_ones_3 : 
  (Finset.filter two_digit_prime_with_ones_3 (Finset.range 100)).card = 6 := 
  by
  sorry

end count_two_digit_primes_with_ones_3_l284_284757


namespace smallest_odd_number_with_five_prime_factors_is_15015_l284_284844

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m ∣ n, m = 1 ∨ m = n

def smallest_odd_primes : List ℕ := [3, 5, 7, 11, 13]

def has_five_distinct_prime_factors (n : ℕ) : Prop :=
  ∃ (factors : List ℕ), factors.nodup ∧ factors.length = 5 ∧ (∀ p ∈ factors, is_prime p) ∧ factors.prod = n

def is_odd (n : ℕ) : Prop := n % 2 = 1

def smallest_odd_number_with_five_prime_factors (n : ℕ) : Prop :=
  has_five_distinct_prime_factors n ∧ is_odd n

theorem smallest_odd_number_with_five_prime_factors_is_15015 :
  smallest_odd_number_with_five_prime_factors 15015 :=
by
  sorry

end smallest_odd_number_with_five_prime_factors_is_15015_l284_284844


namespace greatest_possible_value_of_x_l284_284115

theorem greatest_possible_value_of_x (x : ℕ) (H : Nat.lcm (Nat.lcm x 12) 18 = 108) : x ≤ 108 := sorry

end greatest_possible_value_of_x_l284_284115


namespace time_per_lice_check_l284_284278

-- Define the number of students in each grade
def kindergartners := 26
def first_graders := 19
def second_graders := 20
def third_graders := 25

-- Define the total number of students
def total_students := kindergartners + first_graders + second_graders + third_graders

-- Define the total time in minutes
def hours := 3
def minutes_per_hour := 60
def total_minutes := hours * minutes_per_hour

-- Define the correct answer for time per check
def time_per_check := total_minutes / total_students

-- Prove that the time for each check is 2 minutes
theorem time_per_lice_check : time_per_check = 2 := 
by
  sorry

end time_per_lice_check_l284_284278


namespace sin2x_value_l284_284056

theorem sin2x_value (x : ℝ) (h : Real.sin (x + π / 4) = 3 / 5) : 
  Real.sin (2 * x) = 8 * Real.sqrt 2 / 25 := 
by sorry

end sin2x_value_l284_284056


namespace daily_evaporation_l284_284313

theorem daily_evaporation (initial_water: ℝ) (days: ℝ) (evap_percentage: ℝ) : 
  initial_water = 10 → days = 50 → evap_percentage = 2 →
  (initial_water * evap_percentage / 100) / days = 0.04 := 
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end daily_evaporation_l284_284313


namespace count_perfect_squares_lt_10_pow_9_multiple_36_l284_284494

theorem count_perfect_squares_lt_10_pow_9_multiple_36 : 
  ∃ N : ℕ, ∀ n < 31622, (n % 6 = 0 → n^2 < 10^9 ∧ 36 ∣ n^2 → n ≤ 31620 → N = 5270) :=
by
  sorry

end count_perfect_squares_lt_10_pow_9_multiple_36_l284_284494


namespace find_max_slope_of_OQ_l284_284488

noncomputable def parabola_C := {p : ℝ // p = 2}

def parabola_eq (p : ℝ) : Prop := 
  ∀ x y : ℝ, (y^2 = 2 * p * x) → (y^2 = 4 * x)

def max_slope (p : ℝ) (O Q : ℝ × ℝ) (F P Q' : ℝ × ℝ) : Prop := 
  ∀ K : ℝ, K = (Q.2) / (Q.1) → 
  ∀ n : ℝ, (K = (10 * n) / (25 * n^2 + 9)) →
  ∀ n : ℝ , n = (3 / 5) → 
  K = (1 / 3)

theorem find_max_slope_of_OQ : 
  ∀ pq: parabola_C,
  ∃ C : parabola_eq pq.val,
  ∃ O F P Q : (ℝ × ℝ),
  (F = (1, 0)) ∧
  (P.1 * P.1 = 4 * P.2) ∧
  (Q.1 - P.1, Q.2 - P.2) = 9 * -(F.1 - Q.1, Q.2) →
  max_slope pq.val O Q F P Q'.1 :=
sorry

end find_max_slope_of_OQ_l284_284488


namespace simplify_radicals_l284_284456

theorem simplify_radicals (q : ℝ) (hq : 0 < q) :
  (Real.sqrt (42 * q)) * (Real.sqrt (7 * q)) * (Real.sqrt (14 * q)) = 98 * q * Real.sqrt (3 * q) :=
by
  sorry

end simplify_radicals_l284_284456


namespace div_fraction_eq_l284_284493

theorem div_fraction_eq :
  (5 / 3) / (1 / 4) = 20 / 3 := 
by
  sorry

end div_fraction_eq_l284_284493


namespace positive_difference_of_complementary_ratio_5_1_l284_284274

-- Define angles satisfying the ratio condition and being complementary
def angle_pair (a b : ℝ) : Prop := (a + b = 90) ∧ (a = 5 * b ∨ b = 5 * a)

theorem positive_difference_of_complementary_ratio_5_1 :
  ∃ a b : ℝ, angle_pair a b ∧ abs (a - b) = 60 :=
by
  sorry

end positive_difference_of_complementary_ratio_5_1_l284_284274


namespace triangle_side_solution_l284_284477

/-- 
Given \( a \geq b \geq c > 0 \) and \( a < b + c \), a solution to the equation 
\( b \sqrt{x^{2} - c^{2}} + c \sqrt{x^{2} - b^{2}} = a x \) is provided by 
\( x = \frac{abc}{2 \sqrt{p(p-a)(p-b)(p-c)}} \) where \( p = \frac{1}{2}(a+b+c) \).
-/

theorem triangle_side_solution (a b c x : ℝ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c > 0) (h4 : a < b + c) :
  b * (Real.sqrt (x^2 - c^2)) + c * (Real.sqrt (x^2 - b^2)) = a * x → 
  x = (a * b * c) / (2 * Real.sqrt ((a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c))) :=
sorry

end triangle_side_solution_l284_284477


namespace complete_the_square_l284_284863

theorem complete_the_square (x : ℝ) : x^2 - 8 * x + 1 = 0 → (x - 4)^2 = 15 :=
by
  intro h
  sorry

end complete_the_square_l284_284863


namespace smallest_x_absolute_value_l284_284472

theorem smallest_x_absolute_value :
  ∃ x : ℝ, (|5 * x + 15| = 40) ∧ (∀ y : ℝ, |5 * y + 15| = 40 → x ≤ y) ∧ x = -11 :=
sorry

end smallest_x_absolute_value_l284_284472


namespace henri_total_time_l284_284939

variable (m1 m2 : ℝ) (r w : ℝ)

theorem henri_total_time (H1 : m1 = 3.5) (H2 : m2 = 1.5) (H3 : r = 10) (H4 : w = 1800) :
    m1 + m2 + w / r / 60 = 8 := by
  sorry

end henri_total_time_l284_284939


namespace tan_seventeen_pi_over_four_l284_284468

theorem tan_seventeen_pi_over_four : Real.tan (17 * Real.pi / 4) = 1 := by
  sorry

end tan_seventeen_pi_over_four_l284_284468


namespace min_fraction_value_l284_284375

noncomputable def min_value_fraction (x y z : ℝ) (h : x > 0 ∧ y > 0 ∧ z > 0) (h₁ : x^2 + y^2 + z^2 = 1) : ℝ :=
  (z+1)^2 / (2 * x * y * z)

theorem min_fraction_value (x y z : ℝ) (h : x > 0 ∧ y > 0 ∧ z > 0) (h₁ : x^2 + y^2 + z^2 = 1) :
  min_value_fraction x y z h h₁ = 3 + 2 * Real.sqrt 2 :=
  sorry

end min_fraction_value_l284_284375


namespace compare_negatives_l284_284346

theorem compare_negatives : -3 < -2 := 
by { sorry }

end compare_negatives_l284_284346


namespace num_two_digit_primes_with_ones_digit_3_l284_284737

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100
  
def ones_digit_is_3 (n : ℕ) : Prop :=
  n % 10 = 3

theorem num_two_digit_primes_with_ones_digit_3 : 
  (∃ n1 n2 n3 n4 n5 n6 : ℕ, 
   two_digit_number n1 ∧ ones_digit_is_3 n1 ∧ is_prime n1 ∧ 
   two_digit_number n2 ∧ ones_digit_is_3 n2 ∧ is_prime n2 ∧ 
   two_digit_number n3 ∧ ones_digit_is_3 n3 ∧ is_prime n3 ∧ 
   two_digit_number n4 ∧ ones_digit_is_3 n4 ∧ is_prime n4 ∧ 
   two_digit_number n5 ∧ ones_digit_is_3 n5 ∧ is_prime n5 ∧ 
   two_digit_number n6 ∧ ones_digit_is_3 n6 ∧ is_prime n6) ∧
  (∀ n : ℕ, two_digit_number n → ones_digit_is_3 n → is_prime n → 
  n = n1 ∨ n = n2 ∨ n = n3 ∨ n = n4 ∨ n = n5 ∨ n = n6) :=
sorry

end num_two_digit_primes_with_ones_digit_3_l284_284737


namespace larger_solution_of_quadratic_l284_284038

theorem larger_solution_of_quadratic :
  ∀ x y : ℝ, x^2 - 19 * x - 48 = 0 ∧ y^2 - 19 * y - 48 = 0 ∧ x ≠ y →
  max x y = 24 :=
by
  sorry

end larger_solution_of_quadratic_l284_284038


namespace parabola_line_intersection_l284_284148

/-- 
Given a parabola \( y^2 = 2x \), a line passing through the focus of 
the parabola intersects the parabola at points \( A \) and \( B \) where 
the sum of the x-coordinates of \( A \) and \( B \) is equal to 2. 
Prove that such a line exists and there are exactly 3 such lines.
--/
theorem parabola_line_intersection :
  ∃ l₁ l₂ l₃ : (ℝ × ℝ) → (ℝ × ℝ), 
    (∀ p, l₁ p = l₂ p ∧ l₁ p = l₃ p → false) ∧
    ∀ (A B : ℝ × ℝ), 
      (A.2 ^ 2 = 2 * A.1) ∧ 
      (B.2 ^ 2 = 2 * B.1) ∧ 
      (A.1 + B.1 = 2) →
      (∃ k : ℝ, 
        ∀ (x : ℝ), 
          ((A.2 = k * (A.1 - 1)) ∧ (B.2 = k * (B.1 - 1))) ∧ 
          (k * (A.1 - 1) = k * (B.1 - 1)) ∧ 
          (k ≠ 0)) :=
sorry

end parabola_line_intersection_l284_284148


namespace nina_total_spent_l284_284799

open Real

def toy_price : ℝ := 10
def toy_count : ℝ := 3
def toy_discount : ℝ := 0.15

def card_price : ℝ := 5
def card_count : ℝ := 2
def card_discount : ℝ := 0.10

def shirt_price : ℝ := 6
def shirt_count : ℝ := 5
def shirt_discount : ℝ := 0.20

def sales_tax_rate : ℝ := 0.07

noncomputable def discounted_price (price : ℝ) (count : ℝ) (discount : ℝ) : ℝ :=
  count * price * (1 - discount)

noncomputable def total_cost_before_tax : ℝ := 
  discounted_price toy_price toy_count toy_discount +
  discounted_price card_price card_count card_discount +
  discounted_price shirt_price shirt_count shirt_discount

noncomputable def total_cost_after_tax : ℝ :=
  total_cost_before_tax * (1 + sales_tax_rate)

theorem nina_total_spent : total_cost_after_tax = 62.60 :=
by
  sorry

end nina_total_spent_l284_284799


namespace percentage_of_stock_l284_284503

-- Definitions based on conditions
def income := 500  -- I
def investment := 1500  -- Inv
def price := 90  -- Price

-- Initiate the Lean 4 statement for the proof
theorem percentage_of_stock (P : ℝ) (h : income = (investment * P) / price) : P = 30 :=
by
  sorry

end percentage_of_stock_l284_284503


namespace polygon_properties_l284_284724

def interior_angle_sum (n : ℕ) : ℝ :=
  (n - 2) * 180

def exterior_angle_sum : ℝ :=
  360

theorem polygon_properties (n : ℕ) (h : interior_angle_sum n = 3 * exterior_angle_sum + 180) :
  n = 9 ∧ interior_angle_sum n / n = 140 :=
by
  sorry

end polygon_properties_l284_284724


namespace twentieth_fisherman_catch_l284_284998

theorem twentieth_fisherman_catch (total_fishermen : ℕ) (total_fish : ℕ) (fish_per_19 : ℕ) (fish_each_19 : ℕ) (h1 : total_fishermen = 20) (h2 : total_fish = 10000) (h3 : fish_per_19 = 19 * 400) (h4 : fish_each_19 = 400) : 
  fish_per_19 + fish_each_19 = total_fish := by
  sorry

end twentieth_fisherman_catch_l284_284998


namespace two_digit_primes_end_in_3_l284_284759

theorem two_digit_primes_end_in_3 : 
  {n : ℕ | n ≥ 10 ∧ n < 100 ∧ n % 10 = 3 ∧ Prime n}.card = 6 := 
by
  sorry

end two_digit_primes_end_in_3_l284_284759


namespace math_problem_l284_284884

theorem math_problem
  (x₁ x₂ x₃ x₄ x₅ x₆ x₇ : ℝ)
  (h1 : x₁ + 4 * x₂ + 9 * x₃ + 16 * x₄ + 25 * x₅ + 36 * x₆ + 49 * x₇ = 1)
  (h2 : 4 * x₁ + 9 * x₂ + 16 * x₃ + 25 * x₄ + 36 * x₅ + 49 * x₆ + 64 * x₇ = 12)
  (h3 : 9 * x₁ + 16 * x₂ + 25 * x₃ + 36 * x₄ + 49 * x₅ + 64 * x₆ + 81 * x₇ = 123) :
  16 * x₁ + 25 * x₂ + 36 * x₃ + 49 * x₄ + 64 * x₅ + 81 * x₆ + 100 * x₇ = 334 := by
  sorry

end math_problem_l284_284884


namespace parabola_normal_intersect_l284_284959

theorem parabola_normal_intersect {x y : ℝ} (h₁ : y = x^2) (A : ℝ × ℝ) (hA : A = (-1, 1)) :
  ∃ B : ℝ × ℝ, B = (1.5, 2.25) ∧ ∀ x : ℝ, (y - 1) = 1/2 * (x + 1) →
  ∀ x : ℝ, y = x^2 ∧ B = (1.5, 2.25) :=
sorry

end parabola_normal_intersect_l284_284959


namespace tan_105_degree_l284_284588

theorem tan_105_degree : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  have tan_add : ∀ (a b : ℝ), Real.tan (a + b) = (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b) :=
    sorry
  have tan_45 := Real.tan (45 * Real.pi / 180)
  have tan_60 := Real.tan (60 * Real.pi / 180)
  have tan_45_value : tan_45 = 1 := sorry
  have tan_60_value : tan_60 = Real.sqrt 3 := sorry
  sorry

end tan_105_degree_l284_284588


namespace tan_105_l284_284686

-- Defining the necessary known values and functions
def tan_addition (a b : ℝ) : ℝ := (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b)
def tan_60 : ℝ := Real.sqrt 3
def tan_45 : ℝ := 1

-- Proof goal in Lean 4
theorem tan_105 : Real.tan (105 * Real.pi / 180) = 2 + Real.sqrt 3 :=
by
  let tan_105 := tan_addition (60 * Real.pi / 180) (45 * Real.pi / 180)
  have h1 : Real.tan (60 * Real.pi / 180) = tan_60 := by sorry
  have h2 : Real.tan (45 * Real.pi / 180) = tan_45 := by sorry
  show Real.tan (105 * Real.pi / 180) = 2 + Real.sqrt 3 from sorry

end tan_105_l284_284686


namespace h_at_7_over_5_eq_0_l284_284894

def h (x : ℝ) : ℝ := 5 * x - 7

theorem h_at_7_over_5_eq_0 : h (7 / 5) = 0 := 
by 
  sorry

end h_at_7_over_5_eq_0_l284_284894


namespace percentage_goods_lost_l284_284006

theorem percentage_goods_lost
    (cost_price selling_price loss_price : ℝ)
    (profit_percent loss_percent : ℝ)
    (h_profit : selling_price = cost_price * (1 + profit_percent / 100))
    (h_loss_value : loss_price = selling_price * (loss_percent / 100))
    (cost_price_assumption : cost_price = 100)
    (profit_percent_assumption : profit_percent = 10)
    (loss_percent_assumption : loss_percent = 45) :
    (loss_price / cost_price * 100) = 49.5 :=
sorry

end percentage_goods_lost_l284_284006


namespace product_odd_primes_mod_32_l284_284339

open Nat

theorem product_odd_primes_mod_32 : 
  let primes := [3, 5, 7, 11, 13] 
  let product := primes.foldl (· * ·) 1 
  product % 32 = 7 := 
by
  sorry

end product_odd_primes_mod_32_l284_284339


namespace initial_boys_l284_284499

theorem initial_boys (p : ℝ) (initial_boys : ℝ) (final_boys : ℝ) (final_groupsize : ℝ) : 
  (initial_boys = 0.35 * p) ->
  (final_boys = 0.35 * p - 1) ->
  (final_groupsize = p + 3) ->
  (final_boys / final_groupsize = 0.3) ->
  initial_boys = 13 := 
by
  sorry

end initial_boys_l284_284499


namespace simplify_fraction_l284_284413

theorem simplify_fraction (a b c : ℕ) (h1 : 222 = 2 * 111) (h2 : 999 = 3 * 333) (h3 : 111 = 3 * 37) :
  (222 / 999 * 111) = 74 :=
by
  sorry

end simplify_fraction_l284_284413


namespace johnny_weekly_earnings_l284_284953

-- Define the conditions mentioned in the problem.
def number_of_dogs_at_once : ℕ := 3
def thirty_minute_walk_payment : ℝ := 15
def sixty_minute_walk_payment : ℝ := 20
def work_hours_per_day : ℝ := 4
def sixty_minute_walks_needed_per_day : ℕ := 6
def work_days_per_week : ℕ := 5

-- Prove Johnny's weekly earnings given the conditions
theorem johnny_weekly_earnings :
  let sixty_minute_walks_per_day := sixty_minute_walks_needed_per_day / number_of_dogs_at_once
  let sixty_minute_earnings_per_day := sixty_minute_walks_per_day * number_of_dogs_at_once * sixty_minute_walk_payment
  let remaining_hours_per_day := work_hours_per_day - sixty_minute_walks_per_day
  let thirty_minute_walks_per_day := remaining_hours_per_day * 2 -- each 30-minute walk takes 0.5 hours
  let thirty_minute_earnings_per_day := thirty_minute_walks_per_day * number_of_dogs_at_once * thirty_minute_walk_payment
  let daily_earnings := sixty_minute_earnings_per_day + thirty_minute_earnings_per_day
  let weekly_earnings := daily_earnings * work_days_per_week
  weekly_earnings = 1500 :=
by
  sorry

end johnny_weekly_earnings_l284_284953


namespace base_7_to_base_10_l284_284295

theorem base_7_to_base_10 (a b c d e : ℕ) (h : 23456 = e * 10000 + d * 1000 + c * 100 + b * 10 + a) :
  2 * 7^4 + 3 * 7^3 + 4 * 7^2 + 5 * 7^1 + 6 * 7^0 = 6068 :=
by
  sorry

end base_7_to_base_10_l284_284295


namespace square_area_from_diagonal_l284_284309

theorem square_area_from_diagonal (d : ℝ) (h : d = 28) : (∃ A : ℝ, A = 392) :=
by
  sorry

end square_area_from_diagonal_l284_284309


namespace total_potatoes_l284_284146

theorem total_potatoes (cooked_potatoes : ℕ) (time_per_potato : ℕ) (remaining_time : ℕ) (H1 : cooked_potatoes = 7) (H2 : time_per_potato = 5) (H3 : remaining_time = 45) : (cooked_potatoes + (remaining_time / time_per_potato) = 16) :=
by
  sorry

end total_potatoes_l284_284146


namespace exists_k_l284_284549

def satisfies_condition (a b : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, n ≥ 3 → (a n - a (n-1)) * (a n - a (n-2)) + (b n - b (n-1)) * (b n - b (n-2)) = 0

theorem exists_k (a b : ℕ → ℤ) 
  (h : satisfies_condition a b) : 
  ∃ k : ℕ, k > 0 ∧ a k = a (k + 2008) :=
sorry

end exists_k_l284_284549


namespace four_fold_application_of_f_l284_284398

def f (x : ℕ) : ℕ :=
  if x % 3 = 0 then
    x / 3
  else
    5 * x + 2

theorem four_fold_application_of_f : f (f (f (f 3))) = 187 := 
  by
    sorry

end four_fold_application_of_f_l284_284398


namespace age_difference_l284_284420

theorem age_difference (O Y : ℕ) (h₀ : O = 38) (h₁ : Y + O = 74) : O - Y = 2 := by
  sorry

end age_difference_l284_284420


namespace range_of_m_l284_284782

def is_ellipse (m : ℝ) : Prop :=
  ∀ x y : ℝ, m * (x^2 + y^2 + 2*y + 1) = (x - 2*y + 3)^2

theorem range_of_m (m : ℝ) (h : is_ellipse m) : m > 5 :=
sorry

end range_of_m_l284_284782


namespace base_seven_to_base_ten_l284_284297

theorem base_seven_to_base_ten : 
  let n := 23456 
  ∈ ℕ, nat : ℕ 
  in nat = 6068 := 
by 
  sorry

end base_seven_to_base_ten_l284_284297


namespace total_cost_over_8_weeks_l284_284211

def cost_per_weekday_edition : ℝ := 0.50
def cost_per_sunday_edition : ℝ := 2.00
def num_weekday_editions_per_week : ℕ := 3
def duration_in_weeks : ℕ := 8

theorem total_cost_over_8_weeks :
  (num_weekday_editions_per_week * cost_per_weekday_edition + cost_per_sunday_edition) * duration_in_weeks = 28.00 := by
  sorry

end total_cost_over_8_weeks_l284_284211


namespace A_inter_complement_RB_eq_l284_284377

def A : Set ℝ := {x | ∃ y, y = Real.sqrt (4 - x^2)}

def B : Set ℝ := {x | ∃ y, y = Real.log (1 - x)}

def complement_RB : Set ℝ := {x | x ≥ 1}

theorem A_inter_complement_RB_eq : A ∩ complement_RB = {x | 1 ≤ x ∧ x ≤ 2} := by
  sorry

end A_inter_complement_RB_eq_l284_284377


namespace compare_abc_l284_284189

variable (a b c : ℝ)

noncomputable def define_a : ℝ := (2/3)^(1/3)
noncomputable def define_b : ℝ := (2/3)^(1/2)
noncomputable def define_c : ℝ := (3/5)^(1/2)

theorem compare_abc (h₁ : a = define_a) (h₂ : b = define_b) (h₃ : c = define_c) :
  a > b ∧ b > c := by
  sorry

end compare_abc_l284_284189


namespace quadratic_distinct_roots_l284_284184

theorem quadratic_distinct_roots (c : ℝ) : (∀ (x : ℝ), x^2 + 2 * x + 4 * c = 0 → x ∈ ℝ) ∧ (∃ x y : ℝ, x ≠ y) → c < 1 / 4 :=
by
  sorry

end quadratic_distinct_roots_l284_284184


namespace eval_neg64_pow_two_thirds_l284_284034

theorem eval_neg64_pow_two_thirds : (-64 : Real)^(2/3) = 16 := 
by 
  sorry

end eval_neg64_pow_two_thirds_l284_284034


namespace volume_comparison_l284_284439

-- Define the properties for the cube and the cuboid.
def cube_side_length : ℕ := 1 -- in meters
def cuboid_width : ℕ := 50  -- in centimeters
def cuboid_length : ℕ := 50 -- in centimeters
def cuboid_height : ℕ := 20 -- in centimeters

-- Convert cube side length to centimeters.
def cube_side_length_cm := cube_side_length * 100 -- in centimeters

-- Calculate volumes.
def cube_volume : ℕ := cube_side_length_cm ^ 3 -- in cubic centimeters
def cuboid_volume : ℕ := cuboid_width * cuboid_length * cuboid_height -- in cubic centimeters

-- The theorem stating the problem.
theorem volume_comparison : cube_volume / cuboid_volume = 20 :=
by sorry

end volume_comparison_l284_284439


namespace ratio_of_areas_of_triangle_and_trapezoid_l284_284881

noncomputable def equilateral_triangle_area (s : ℝ) : ℝ := (s ^ 2 * Real.sqrt 3) / 4

theorem ratio_of_areas_of_triangle_and_trapezoid :
  let large_triangle_side := 10
  let small_triangle_side := 5
  let a_large := equilateral_triangle_area large_triangle_side
  let a_small := equilateral_triangle_area small_triangle_side
  let a_trapezoid := a_large - a_small
  (a_small / a_trapezoid) = (1 / 3) :=
by
  let large_triangle_side := 10
  let small_triangle_side := 5
  let a_large := equilateral_triangle_area large_triangle_side
  let a_small := equilateral_triangle_area small_triangle_side
  let a_trapezoid := a_large - a_small
  have h : (a_small / a_trapezoid) = (1 / 3) := 
    by sorry  -- Here would be the proof steps, but we're skipping
  exact h

end ratio_of_areas_of_triangle_and_trapezoid_l284_284881


namespace dice_even_probability_l284_284014

/-- Given six fair six-sided dice, each numbered from 1 to 6, where the probability of rolling an even number on a single die is 1/2, prove that the probability of exactly two dice showing an even number is 15/64. -/
theorem dice_even_probability :
  let probability_even : Rat := 1 / 2
  in let probability_odd : Rat := 1 / 2
  in (binomial 6 2 : Rat) * (probability_even^2) * (probability_odd^4) = 15 / 64 :=
by
  sorry

end dice_even_probability_l284_284014


namespace find_y_intersection_of_tangents_l284_284240

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2 - 2 * x - 3

-- Define the tangent slope at a point on the parabola
def tangent_slope (x : ℝ) : ℝ := 2 * (x - 1)

-- Define the perpendicular condition for tangents at points A and B
def perpendicular_condition (a b : ℝ) : Prop := (a - 1) * (b - 1) = -1 / 4

-- Define the y-coordinate of the intersection point P of the tangents at A and B
def y_coordinate_of_intersection (a b : ℝ) : ℝ := a * b - a - b + 2

-- Theorem to be proved
theorem find_y_intersection_of_tangents (a b : ℝ) 
  (ha : parabola a = a ^ 2 - 2 * a - 3) 
  (hb : parabola b = b ^ 2 - 2 * b - 3) 
  (hp : perpendicular_condition a b) :
  y_coordinate_of_intersection a b = -1 / 4 :=
sorry

end find_y_intersection_of_tangents_l284_284240


namespace general_form_of_line_l_l284_284570

-- Define the point
def pointA : ℝ × ℝ := (1, 2)

-- Define the normal vector
def normalVector : ℝ × ℝ := (1, -3)

-- Define the general form equation
def generalFormEq (x y : ℝ) : Prop := x - 3 * y + 5 = 0

-- Statement to prove
theorem general_form_of_line_l (x y : ℝ) (h_pointA : pointA = (1, 2)) (h_normalVector : normalVector = (1, -3)) :
  generalFormEq x y :=
sorry

end general_form_of_line_l_l284_284570


namespace widgets_per_shipping_box_l284_284703

theorem widgets_per_shipping_box :
  let widget_per_carton := 3
  let carton_width := 4
  let carton_length := 4
  let carton_height := 5
  let shipping_box_width := 20
  let shipping_box_length := 20
  let shipping_box_height := 20
  let carton_volume := carton_width * carton_length * carton_height
  let shipping_box_volume := shipping_box_width * shipping_box_length * shipping_box_height
  let cartons_per_shipping_box := shipping_box_volume / carton_volume
  cartons_per_shipping_box * widget_per_carton = 300 :=
by
  sorry

end widgets_per_shipping_box_l284_284703


namespace max_xy_ratio_proof_l284_284050

noncomputable def max_xy_ratio (x y a b : ℝ) (h1 : x ≥ y) (h2 : y > 0) (h3 : 0 ≤ a) (h4 : a ≤ x) (h5 : 0 ≤ b) (h6 : b ≤ y) (h7 : (x - a) ^ 2 + (y - b) ^ 2 = x ^ 2 + b ^ 2) (h8 : x ^ 2 + b ^ 2 = y ^ 2 + a ^ 2) : ℝ :=
  √(2) / 3

theorem max_xy_ratio_proof (x y a b : ℝ) (h1 : x ≥ y) (h2 : y > 0) (h3 : 0 ≤ a) (h4 : a ≤ x) (h5 : 0 ≤ b) (h6 : b ≤ y) (h7 : (x - a) ^ 2 + (y - b) ^ 2 = x ^ 2 + b ^ 2) (h8 : x ^ 2 + b ^ 2 = y ^ 2 + a ^ 2) : max_xy_ratio x y a b h1 h2 h3 h4 h5 h6 h7 h8 = 2 * √3 / 3 :=
  sorry

end max_xy_ratio_proof_l284_284050


namespace tan_identity_example_l284_284929

theorem tan_identity_example (α : ℝ) (h : Real.tan α = 3) :
  (Real.sin (α - Real.pi) + Real.cos (Real.pi - α)) /
  (Real.sin (Real.pi / 2 - α) + Real.cos (Real.pi / 2 + α)) = 2 :=
by
  sorry

end tan_identity_example_l284_284929


namespace tan_105_degree_l284_284612

theorem tan_105_degree : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 := by
  have h_add : ∀ a b : ℝ, Real.tan (a + b) = (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b) :=
    Real.tan_add

  have h_tan_60 : Real.tan (60 * Real.pi / 180) = Real.sqrt 3 := Real.tan_60
  have h_tan_45 : Real.tan (45 * Real.pi / 180) = 1 := Real.tan_45
  
  sorry

end tan_105_degree_l284_284612


namespace range_of_a_l284_284935

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, x^2 - a*x + 2*a > 0) : 0 < a ∧ a < 8 :=
by
  sorry

end range_of_a_l284_284935


namespace diameter_percentage_l284_284817

theorem diameter_percentage (d_R d_S : ℝ) (h : π * (d_R / 2)^2 = 0.25 * π * (d_S / 2)^2) : 
  d_R = 0.5 * d_S :=
by 
  sorry

end diameter_percentage_l284_284817


namespace probability_same_color_l284_284136

theorem probability_same_color (pairs : ℕ) (total_shoes : ℕ) (select_shoes : ℕ)
  (h_pairs : pairs = 6) 
  (h_total_shoes : total_shoes = 12) 
  (h_select_shoes : select_shoes = 2) : 
  (Nat.choose total_shoes select_shoes > 0) → 
  (Nat.div (pairs * (Nat.choose 2 2)) (Nat.choose total_shoes select_shoes) = 1/11) :=
by
  sorry

end probability_same_color_l284_284136


namespace find_a_l284_284073

-- Define the binomial coefficient function in Lean
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- Define the conditions and the proof problem statement
theorem find_a (a : ℝ) (h: (-a)^7 * binomial 10 7 = -120) : a = 1 :=
sorry

end find_a_l284_284073


namespace count_two_digit_integers_congruent_to_2_mod_4_l284_284066

theorem count_two_digit_integers_congruent_to_2_mod_4 : 
  ∃ n : ℕ, (∀ x : ℕ, 10 ≤ x ∧ x ≤ 99 → x % 4 = 2 → x = 4 * k + 2) ∧ n = 23 := 
by
  sorry

end count_two_digit_integers_congruent_to_2_mod_4_l284_284066


namespace count_two_digit_primes_with_ones_digit_3_l284_284766

theorem count_two_digit_primes_with_ones_digit_3 :
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ n % 10 = 3 ∧ Nat.Prime n}.to_finset.card = 6 :=
by
  sorry

end count_two_digit_primes_with_ones_digit_3_l284_284766


namespace interior_diagonal_length_l284_284282

variables (a b c : ℝ)

-- Conditions
def surface_area_eq : Prop := 2 * (a * b + b * c + c * a) = 22
def edge_length_eq : Prop := 4 * (a + b + c) = 24

-- Question to be proved
theorem interior_diagonal_length :
  surface_area_eq a b c → edge_length_eq a b c → (Real.sqrt (a^2 + b^2 + c^2) = Real.sqrt 14) :=
by
  intros h1 h2
  sorry

end interior_diagonal_length_l284_284282


namespace trapezoid_area_calc_l284_284228

noncomputable def isoscelesTrapezoidArea : ℝ :=
  let a := 1
  let b := 9
  let h := 2 * Real.sqrt 3
  0.5 * (a + b) * h

theorem trapezoid_area_calc : isoscelesTrapezoidArea = 20 * Real.sqrt 3 := by
  sorry

end trapezoid_area_calc_l284_284228


namespace total_legs_l284_284954

def human_legs : Nat := 2
def num_humans : Nat := 2
def dog_legs : Nat := 4
def num_dogs : Nat := 2

theorem total_legs :
  num_humans * human_legs + num_dogs * dog_legs = 12 := by
  sorry

end total_legs_l284_284954


namespace smallest_odd_number_with_five_primes_proof_l284_284848

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m ∣ n, m = 1 ∨ m = n

def smallest_odd_primes : List ℕ := [3, 5, 7, 11, 13]

noncomputable def smallest_odd_number_with_five_primes : ℕ :=
  List.prod smallest_odd_primes

theorem smallest_odd_number_with_five_primes_proof : smallest_odd_number_with_five_primes = 15015 :=
by
  unfold smallest_odd_number_with_five_primes
  unfold smallest_odd_primes
  norm_num

end smallest_odd_number_with_five_primes_proof_l284_284848


namespace sin_cos_ratio_value_sin_cos_expression_value_l284_284055

variable (α : ℝ)

-- Given condition
def tan_alpha_eq_3 := Real.tan α = 3

-- Goal (1)
theorem sin_cos_ratio_value 
  (h : tan_alpha_eq_3 α) : 
  (Real.sin α + Real.cos α) / (2 * Real.sin α - Real.cos α) = 4 / 5 := 
  sorry

-- Goal (2)
theorem sin_cos_expression_value
  (h : tan_alpha_eq_3 α) : 
  Real.sin α ^ 2 + Real.sin α * Real.cos α + 3 * Real.cos α ^ 2 = 15 := 
  sorry

end sin_cos_ratio_value_sin_cos_expression_value_l284_284055


namespace tan_105_degree_is_neg_sqrt3_minus_2_l284_284649

theorem tan_105_degree_is_neg_sqrt3_minus_2 :
  Real.tan (105 * Real.pi / 180) = -(Real.sqrt 3 + 2) := by
  sorry

end tan_105_degree_is_neg_sqrt3_minus_2_l284_284649


namespace probability_first_two_cards_black_l284_284948

theorem probability_first_two_cards_black :
  let deck := 52,
      suits := 4,
      cards_per_suit := 13,
      black_suits := 2
  in
  let total_black_cards := black_suits * cards_per_suit,
      total_ways := (deck * (deck - 1)) / 2,
      successful_ways := (total_black_cards * (total_black_cards - 1)) / 2
  in 
  (successful_ways / total_ways : ℚ) = 25 / 102 := by
  sorry

end probability_first_two_cards_black_l284_284948


namespace find_b_l284_284895

def h (x : ℝ) : ℝ := 5 * x - 7

theorem find_b : ∃ b : ℝ, h(b) = 0 ∧ b = 7 / 5 :=
by
  sorry

end find_b_l284_284895


namespace problem1_problem2_l284_284205

-- Definitions of the sets A, B, C
def A (a : ℝ) : Set ℝ := { x | x^2 - a*x + a^2 - 12 = 0 }
def B : Set ℝ := { x | x^2 - 2*x - 8 = 0 }
def C (m : ℝ) : Set ℝ := { x | m*x + 1 = 0 }

-- Problem 1: If A = B, then a = 2
theorem problem1 (a : ℝ) (h : A a = B) : a = 2 := sorry

-- Problem 2: If B ∪ C m = B, then m ∈ {-1/4, 0, 1/2}
theorem problem2 (m : ℝ) (h : B ∪ C m = B) : m = -1/4 ∨ m = 0 ∨ m = 1/2 := sorry

end problem1_problem2_l284_284205


namespace product_representation_count_l284_284384

theorem product_representation_count :
  let n := 1000000
  let distinct_ways := 139
  (∃ (a b c d e f : ℕ), 2^(a+b+c) * 5^(d+e+f) = n ∧ 
    a + b + c = 6 ∧ d + e + f = 6 ) → 
    139 = distinct_ways := 
by {
  sorry
}

end product_representation_count_l284_284384


namespace cube_faces_sum_l284_284530

open Nat

theorem cube_faces_sum (a b c d e f : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h5 : 0 < e) (h6 : 0 < f) 
    (h7 : (a + d) * (b + e) * (c + f) = 1386) : 
    a + b + c + d + e + f = 38 := 
sorry

end cube_faces_sum_l284_284530


namespace tan_105_l284_284602

theorem tan_105 :
  tan 105 = -2 - sqrt 3 :=
by sorry

end tan_105_l284_284602


namespace hiking_trip_distance_l284_284102

open Real

-- Define the given conditions
def distance_north : ℝ := 10
def distance_south : ℝ := 7
def distance_east1 : ℝ := 17
def distance_east2 : ℝ := 8

-- Define the net displacement conditions
def net_distance_north : ℝ := distance_north - distance_south
def net_distance_east : ℝ := distance_east1 + distance_east2

-- Prove the distance from the starting point
theorem hiking_trip_distance :
  sqrt ((net_distance_north)^2 + (net_distance_east)^2) = sqrt 634 := by
  sorry

end hiking_trip_distance_l284_284102


namespace sum_first_fifty_digits_of_decimal_of_one_over_1234_l284_284856

theorem sum_first_fifty_digits_of_decimal_of_one_over_1234 :
  let s := "00081037277147487844408427876817350238192918144683"
  let digits := s.data
  (4 * (list.sum (digits.map (λ c, (c.to_nat - '0'.to_nat)))) + (list.sum ((digits.take 6).map (λ c, (c.to_nat - '0'.to_nat)))) ) = 729 :=
by sorry

end sum_first_fifty_digits_of_decimal_of_one_over_1234_l284_284856


namespace largest_x_satisfying_abs_eq_largest_x_is_correct_l284_284168

theorem largest_x_satisfying_abs_eq (x : ℝ) (h : |x - 5| = 12) : x ≤ 17 :=
by
  sorry

noncomputable def largest_x : ℝ := 17

theorem largest_x_is_correct (x : ℝ) (h : |x - 5| = 12) : x ≤ largest_x :=
largest_x_satisfying_abs_eq x h

end largest_x_satisfying_abs_eq_largest_x_is_correct_l284_284168


namespace minimum_framing_needed_l284_284141

-- Definitions given the conditions
def original_width := 5
def original_height := 7
def enlargement_factor := 4
def border_width := 3
def inches_per_foot := 12

-- Conditions translated to definitions
def enlarged_width := original_width * enlargement_factor
def enlarged_height := original_height * enlargement_factor
def bordered_width := enlarged_width + 2 * border_width
def bordered_height := enlarged_height + 2 * border_width
def perimeter := 2 * (bordered_width + bordered_height)
def perimeter_in_feet := perimeter / inches_per_foot

-- Prove that the minimum number of linear feet of framing required is 10 feet
theorem minimum_framing_needed : perimeter_in_feet = 10 := 
by 
  sorry

end minimum_framing_needed_l284_284141


namespace factorize_expression_l284_284706

theorem factorize_expression (a b : ℝ) : 
  a^3 + 2 * a^2 * b + a * b^2 = a * (a + b)^2 := by sorry

end factorize_expression_l284_284706


namespace find_f_2547_l284_284239

theorem find_f_2547 (f : ℚ → ℚ)
  (h1 : ∀ x y : ℚ, f (x + y) = f x + f y + 2547)
  (h2 : f 2004 = 2547) :
  f 2547 = 2547 :=
sorry

end find_f_2547_l284_284239


namespace simplify_expression_l284_284106

theorem simplify_expression :
  ( ( (11 / 4) / (11 / 10 + 10 / 3) ) / ( 5 / 2 - ( 4 / 3 ) ) ) /
  ( ( 5 / 7 ) - ( ( (13 / 6 + 9 / 2) * 3 / 8 ) / (11 / 4 - 3 / 2) ) )
  = - (35 / 9) :=
by
  sorry

end simplify_expression_l284_284106


namespace value_of_m_l284_284962

def f (x m : ℝ) : ℝ := x^2 - 2 * x + m
def g (x m : ℝ) : ℝ := x^2 - 2 * x + 2 * m + 8

theorem value_of_m (m : ℝ) : (3 * f 5 m = g 5 m) → m = -22 :=
by
  intro h
  sorry

end value_of_m_l284_284962


namespace zeros_of_f_l284_284374

noncomputable def f (a b c x : ℝ) : ℝ := (x - a) * (x - b) + (x - b) * (x - c) + (x - c) * (x - a)

theorem zeros_of_f (a b c : ℝ) (h1 : a < b) (h2 : b < c) :
  (∃ x, a < x ∧ x < b ∧ f a b c x = 0) ∧ (∃ y, b < y ∧ y < c ∧ f a b c y = 0) :=
by
  sorry

end zeros_of_f_l284_284374


namespace total_money_is_twenty_l284_284206

-- Define Henry's initial money
def henry_initial_money : Nat := 5

-- Define the money Henry earned
def henry_earned_money : Nat := 2

-- Define Henry's total money
def henry_total_money : Nat := henry_initial_money + henry_earned_money

-- Define friend's money
def friend_money : Nat := 13

-- Define the total combined money
def total_combined_money : Nat := henry_total_money + friend_money

-- The main statement to prove
theorem total_money_is_twenty : total_combined_money = 20 := sorry

end total_money_is_twenty_l284_284206


namespace tan_105_l284_284683

-- Defining the necessary known values and functions
def tan_addition (a b : ℝ) : ℝ := (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b)
def tan_60 : ℝ := Real.sqrt 3
def tan_45 : ℝ := 1

-- Proof goal in Lean 4
theorem tan_105 : Real.tan (105 * Real.pi / 180) = 2 + Real.sqrt 3 :=
by
  let tan_105 := tan_addition (60 * Real.pi / 180) (45 * Real.pi / 180)
  have h1 : Real.tan (60 * Real.pi / 180) = tan_60 := by sorry
  have h2 : Real.tan (45 * Real.pi / 180) = tan_45 := by sorry
  show Real.tan (105 * Real.pi / 180) = 2 + Real.sqrt 3 from sorry

end tan_105_l284_284683


namespace inequality_proof_l284_284096

theorem inequality_proof (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (2 * x^2 / (y + z) + 2 * y^2 / (z + x) + 2 * z^2 / (x + y) ≥ x + y + z) :=
by
  sorry

end inequality_proof_l284_284096


namespace solve_system_equations_l284_284417

theorem solve_system_equations (x y : ℝ) :
  x + y = 0 ∧ 2 * x + 3 * y = 3 → x = -3 ∧ y = 3 :=
by {
  -- Leave the proof as a placeholder with "sorry".
  sorry
}

end solve_system_equations_l284_284417


namespace double_series_sum_l284_284164

theorem double_series_sum :
  (∑' j : ℕ, ∑' k : ℕ, (2 : ℝ) ^ (-(3 * k + 2 * j + (k + j) ^ 2))) = 4 / 3 :=
sorry

end double_series_sum_l284_284164


namespace tan_105_degree_is_neg_sqrt3_minus_2_l284_284651

theorem tan_105_degree_is_neg_sqrt3_minus_2 :
  Real.tan (105 * Real.pi / 180) = -(Real.sqrt 3 + 2) := by
  sorry

end tan_105_degree_is_neg_sqrt3_minus_2_l284_284651


namespace tan_105_degree_l284_284585

theorem tan_105_degree : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  have tan_add : ∀ (a b : ℝ), Real.tan (a + b) = (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b) :=
    sorry
  have tan_45 := Real.tan (45 * Real.pi / 180)
  have tan_60 := Real.tan (60 * Real.pi / 180)
  have tan_45_value : tan_45 = 1 := sorry
  have tan_60_value : tan_60 = Real.sqrt 3 := sorry
  sorry

end tan_105_degree_l284_284585


namespace twentieth_fisherman_caught_l284_284997

-- Definitions based on conditions
def fishermen_count : ℕ := 20
def total_fish_caught : ℕ := 10000
def fish_per_nineteen_fishermen : ℕ := 400
def nineteen_count : ℕ := 19

-- Calculation based on the problem conditions
def total_fish_by_nineteen : ℕ := nineteen_count * fish_per_nineteen_fishermen

-- Prove the number of fish caught by the twentieth fisherman
theorem twentieth_fisherman_caught : 
  total_fish_caught - total_fish_by_nineteen = 2400 := 
by
  -- This is where the proof would go
  sorry

end twentieth_fisherman_caught_l284_284997


namespace positive_integer_conditions_l284_284042

theorem positive_integer_conditions (p : ℕ) (hp : p > 0) :
  (∃ q : ℕ, q > 0 ∧ (5 * p + 36) = q * (2 * p - 9)) ↔ (p = 5 ∨ p = 6 ∨ p = 9 ∨ p = 18) :=
by sorry

end positive_integer_conditions_l284_284042


namespace area_of_triangle_l284_284447

open Real

-- Defining the line equation 3x + 2y = 12
def line_eq (x y : ℝ) : Prop := 3 * x + 2 * y = 12

-- Defining the vertices of the triangle
def vertex1 := (0, 0 : ℝ)
def vertex2 := (0, 6 : ℝ)
def vertex3 := (4, 0 : ℝ)

-- Define a function to calculate the area of the triangle
def triangle_area (v1 v2 v3 : ℝ × ℝ) : ℝ :=
  0.5 * abs ((v2.1 - v1.1) * (v3.2 - v1.2) - (v3.1 - v1.1) * (v2.2 - v1.2))

-- Prove that area of the triangle bounded by the line and coordinate axes is 12 square units
theorem area_of_triangle : triangle_area vertex1 vertex2 vertex3 = 12 :=
by
  sorry

end area_of_triangle_l284_284447


namespace cos_theta_sub_pi_div_3_value_l284_284054

open Real

noncomputable def problem_statement (θ : ℝ) : Prop :=
  sin (3 * π - θ) = (sqrt 5 / 2) * sin (π / 2 + θ)

theorem cos_theta_sub_pi_div_3_value (θ : ℝ) (hθ : problem_statement θ) :
  cos (θ - π / 3) = 1 / 3 + sqrt 15 / 6 ∨ cos (θ - π / 3) = - (1 / 3 + sqrt 15 / 6) :=
sorry

end cos_theta_sub_pi_div_3_value_l284_284054


namespace proof_problem_l284_284519

open Classical

variable (x y z : ℝ)

theorem proof_problem
  (cond1 : 0 < x ∧ x < 1)
  (cond2 : 0 < y ∧ y < 1)
  (cond3 : 0 < z ∧ z < 1)
  (cond4 : x * y * z = (1 - x) * (1 - y) * (1 - z)) :
  ((1 - x) * y ≥ 1/4) ∨ ((1 - y) * z ≥ 1/4) ∨ ((1 - z) * x ≥ 1/4) := by
  sorry

end proof_problem_l284_284519


namespace sum_max_min_a_l284_284972

theorem sum_max_min_a (a : ℝ) (h1 : ∀ x : ℝ, x^2 - a * x - 20 * a^2 < 0)
  (h2 : ∀ x1 x2 : ℝ, x1^2 - a * x1 - 20 * a^2 = 0 → x2^2 - a * x2 - 20 * a^2 = 0 → |x1 - x2| ≤ 9) :
    -1 ≤ a ∧ a ≤ 1 ∧ a ≠ 0 → (1 + -1) = 0 :=
by
  sorry

end sum_max_min_a_l284_284972


namespace probability_event_A_l284_284524

-- Defining the vectors
def a_m (m : ℕ) := (m, 1)
def b_n (n : ℕ) := (2, n)

-- Definitions of the set of m and n values
def valid_m : Finset ℕ := {1, 2, 3}
def valid_n : Finset ℕ := {1, 2, 3}

-- Condition for orthogonality
def orthogonal (m n : ℕ) : Prop := (m - 1) ^ 2 = n

-- Event A
def event_A : Finset (ℕ × ℕ) := 
  (valid_m.product valid_n).filter (λ ⟨m, n⟩, orthogonal m n)

-- Total number of possible pairs
def total_pairs : ℕ := (valid_m.product valid_n).card

theorem probability_event_A : 
  (event_A.card : ℚ) / total_pairs = 1 / 9 := 
sorry

end probability_event_A_l284_284524


namespace total_investment_with_interest_l284_284832

def principal : ℝ := 1000
def part3Percent : ℝ := 199.99999999999983
def rate3Percent : ℝ := 0.03
def rate5Percent : ℝ := 0.05
def interest3Percent : ℝ := part3Percent * rate3Percent
def part5Percent : ℝ := principal - part3Percent
def interest5Percent : ℝ := part5Percent * rate5Percent
def totalWithInterest : ℝ := principal + interest3Percent + interest5Percent

theorem total_investment_with_interest :
  totalWithInterest = 1046.00 :=
by
  unfold totalWithInterest interest5Percent part5Percent interest3Percent
  sorry

end total_investment_with_interest_l284_284832


namespace quadratic_has_distinct_real_roots_l284_284175

theorem quadratic_has_distinct_real_roots : ∀ c : ℝ, x^2 + 2 * x + 4 * c = 0 → 4 - 16 * c > 0 → c = 0 :=
begin
  intros c h_eq h_disc,
  sorry
end

end quadratic_has_distinct_real_roots_l284_284175


namespace tan_105_degree_l284_284590

theorem tan_105_degree : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  have tan_add : ∀ (a b : ℝ), Real.tan (a + b) = (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b) :=
    sorry
  have tan_45 := Real.tan (45 * Real.pi / 180)
  have tan_60 := Real.tan (60 * Real.pi / 180)
  have tan_45_value : tan_45 = 1 := sorry
  have tan_60_value : tan_60 = Real.sqrt 3 := sorry
  sorry

end tan_105_degree_l284_284590


namespace total_legs_l284_284955

def human_legs : Nat := 2
def num_humans : Nat := 2
def dog_legs : Nat := 4
def num_dogs : Nat := 2

theorem total_legs :
  num_humans * human_legs + num_dogs * dog_legs = 12 := by
  sorry

end total_legs_l284_284955


namespace solve_eq_l284_284262

open Real

noncomputable def solution : Set ℝ := { x | ∃ (n : ℤ), x = π / 12 + π * (n : ℝ) }

theorem solve_eq : { x : ℝ | ∃ (n : ℤ), x = π / 12 + π * (n : ℝ) } = solution := by sorry

end solve_eq_l284_284262


namespace no_five_coins_sum_to_43_l284_284710

def coin_values : Set ℕ := {1, 5, 10, 25}

theorem no_five_coins_sum_to_43 :
  ¬ ∃ (a b c d e : ℕ), a ∈ coin_values ∧ b ∈ coin_values ∧ c ∈ coin_values ∧ d ∈ coin_values ∧ e ∈ coin_values ∧ (a + b + c + d + e = 43) :=
sorry

end no_five_coins_sum_to_43_l284_284710


namespace servings_of_popcorn_l284_284123

theorem servings_of_popcorn (popcorn_per_serving : ℕ) (jared_consumption : ℕ)
    (friend_consumption : ℕ) (num_friends : ℕ) :
    popcorn_per_serving = 30 →
    jared_consumption = 90 →
    friend_consumption = 60 →
    num_friends = 3 →
    (jared_consumption + num_friends * friend_consumption) / popcorn_per_serving = 9 := 
by
  intros h1 h2 h3 h4
  sorry

end servings_of_popcorn_l284_284123


namespace sum_of_eight_terms_l284_284341

theorem sum_of_eight_terms :
  (5 ^ 8) + (5 ^ 8) + (5 ^ 8) + (5 ^ 8) + (5 ^ 8) + (5 ^ 8) + (5 ^ 8) + (5 ^ 8) = 3125000 :=
by
  sorry

end sum_of_eight_terms_l284_284341


namespace sum_of_integers_l284_284990

/-- Given two positive integers x and y such that the sum of their squares equals 181 
    and their product equals 90, prove that the sum of these two integers is 19. -/
theorem sum_of_integers (x y : ℤ) (h1 : x^2 + y^2 = 181) (h2 : x * y = 90) : x + y = 19 := by
  sorry

end sum_of_integers_l284_284990


namespace f_neg_one_l284_284931

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then x^2 + 1/x else - (x^2 + 1/(-x))

theorem f_neg_one : f (-1) = -2 :=
by
  -- This is where the proof would go, but it is left as a sorry
  sorry

end f_neg_one_l284_284931


namespace tan_105_degree_is_neg_sqrt3_minus_2_l284_284650

theorem tan_105_degree_is_neg_sqrt3_minus_2 :
  Real.tan (105 * Real.pi / 180) = -(Real.sqrt 3 + 2) := by
  sorry

end tan_105_degree_is_neg_sqrt3_minus_2_l284_284650


namespace carnival_total_cost_l284_284399

def morning_costs (under18_cost over18_cost : ℕ) : ℕ :=
  under18_cost + over18_cost

def afternoon_costs (under18_cost over18_cost : ℕ) : ℕ :=
  under18_cost + 1 + over18_cost + 1

noncomputable def mara_cost : ℕ :=
  let bumper_car_cost := morning_costs 2 0 + afternoon_costs 2 0
  let ferris_wheel_cost := morning_costs 5 5 + 5
  bumper_car_cost + ferris_wheel_cost

noncomputable def riley_cost : ℕ :=
  let space_shuttle_cost := morning_costs 0 5 + afternoon_costs 0 5
  let ferris_wheel_cost := morning_costs 0 6 + (6 + 1)
  space_shuttle_cost + ferris_wheel_cost

theorem carnival_total_cost :
  mara_cost + riley_cost = 61 := by
  sorry

end carnival_total_cost_l284_284399


namespace a1_plus_a9_l284_284715

def S (n : ℕ) : ℕ := n^2 + 1

def a (n : ℕ) : ℕ := S n - S (n - 1)

theorem a1_plus_a9 : (a 1) + (a 9) = 19 := by
  sorry

end a1_plus_a9_l284_284715


namespace largest_sum_achievable_l284_284053

noncomputable theory -- Declare noncomputable theory if necessary

-- Define the problem conditions and proof statement
theorem largest_sum_achievable {n : ℕ} (x : fin n → ℝ) (h : ∀ i j, i ≤ j → x i ≤ x j) (hn : 2 < n) :
  let seq_sum := λ (k : ℕ), x (k + 1) * (Nat.choose (n-2) (k-1)) in
  ∃ sums : Π (k : ℕ) (h : k < (n / 2) + 1), ℝ,
  (∑ k in finset.range (n/2 + 1), sums k sorry) = ∑ k in finset.range (n / 2 + 1), seq_sum k := 
sorry

end largest_sum_achievable_l284_284053


namespace express_in_scientific_notation_l284_284036

def scientific_notation (n : ℤ) (x : ℝ) :=
  ∃ (a : ℝ) (b : ℤ), 1 ≤ |a| ∧ |a| < 10 ∧ x = a * 10^b

theorem express_in_scientific_notation : scientific_notation (-8206000) (-8.206 * 10^6) :=
by
  sorry

end express_in_scientific_notation_l284_284036


namespace probability_two_roads_at_least_5_miles_long_l284_284348

-- Probabilities of roads being at least 5 miles long
def prob_A_B := 3 / 4
def prob_B_C := 2 / 3
def prob_C_D := 1 / 2

-- Theorem: Probability of at least two roads being at least 5 miles long
theorem probability_two_roads_at_least_5_miles_long :
  prob_A_B * prob_B_C * (1 - prob_C_D) +
  prob_A_B * prob_C_D * (1 - prob_B_C) +
  (1 - prob_A_B) * prob_B_C * prob_C_D +
  prob_A_B * prob_B_C * prob_C_D = 11 / 24 := 
by
  sorry -- Proof goes here

end probability_two_roads_at_least_5_miles_long_l284_284348


namespace gym_class_total_students_l284_284578

theorem gym_class_total_students (group1_members group2_members : ℕ) 
  (h1 : group1_members = 34) (h2 : group2_members = 37) :
  group1_members + group2_members = 71 :=
by
  sorry

end gym_class_total_students_l284_284578


namespace find_constants_monotonicity_range_of_k_l284_284191

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := (b - 2 ^ x) / (2 ^ (x + 1) + a)

theorem find_constants (h_odd : ∀ x : ℝ, f x a b = - f (-x) a b) :
  a = 2 ∧ b = 1 :=
sorry

theorem monotonicity (a : ℝ) (b : ℝ) (h_constants : a = 2 ∧ b = 1) :
  ∀ x y : ℝ, x < y → f y a b ≤ f x a b :=
sorry

theorem range_of_k (a : ℝ) (b : ℝ) (h_constants : a = 2 ∧ b = 1)
  (h_pos : ∀ x : ℝ, x ≥ 1 → f (k * 3^x) a b + f (3^x - 9^x + 2) a b > 0) :
  k < 4 / 3 :=
sorry

end find_constants_monotonicity_range_of_k_l284_284191


namespace smallest_odd_with_five_prime_factors_l284_284834

theorem smallest_odd_with_five_prime_factors :
  ∃ n : ℕ, n = 3 * 5 * 7 * 11 * 13 ∧ ∀ m : ℕ, (m < n → (∃ p1 p2 p3 p4 p5 : ℕ,
  prime p1 ∧ odd p1 ∧ prime p2 ∧ odd p2 ∧ prime p3 ∧ odd p3 ∧
  prime p4 ∧ odd p4 ∧ prime p5 ∧ odd p5 ∧
  m = p1 * p2 * p3 * p4 * p5)) → m < 3 * 5 * 7 * 11 * 13 := 
by {
  use 3 * 5 * 7 * 11 * 13,
  split,
  norm_num,
  intros m hlt hexists,
  obtain ⟨p1, p2, p3, p4, p5, hp1, hodd1, hp2, hodd2, hp3, hodd3, hp4, hodd4, hp5, hodd5, hprod⟩ := hexists,
  sorry
}

end smallest_odd_with_five_prime_factors_l284_284834


namespace positive_difference_of_complementary_angles_l284_284269

-- Define the conditions
variables (a b : ℝ)
variable (h1 : a + b = 90)
variable (h2 : 5 * b = a)

-- Define the theorem we are proving
theorem positive_difference_of_complementary_angles (a b : ℝ) (h1 : a + b = 90) (h2 : 5 * b = a) :
  |a - b| = 60 :=
by
  sorry

end positive_difference_of_complementary_angles_l284_284269


namespace tan_105_l284_284688

-- Defining the necessary known values and functions
def tan_addition (a b : ℝ) : ℝ := (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b)
def tan_60 : ℝ := Real.sqrt 3
def tan_45 : ℝ := 1

-- Proof goal in Lean 4
theorem tan_105 : Real.tan (105 * Real.pi / 180) = 2 + Real.sqrt 3 :=
by
  let tan_105 := tan_addition (60 * Real.pi / 180) (45 * Real.pi / 180)
  have h1 : Real.tan (60 * Real.pi / 180) = tan_60 := by sorry
  have h2 : Real.tan (45 * Real.pi / 180) = tan_45 := by sorry
  show Real.tan (105 * Real.pi / 180) = 2 + Real.sqrt 3 from sorry

end tan_105_l284_284688


namespace delta_solution_l284_284218

theorem delta_solution : ∃ Δ : ℤ, 4 * (-3) = Δ - 1 ∧ Δ = -11 :=
by
  -- Using the condition 4(-3) = Δ - 1, 
  -- we need to prove that Δ = -11
  sorry

end delta_solution_l284_284218


namespace tan_105_eq_neg2_sub_sqrt3_l284_284667

theorem tan_105_eq_neg2_sub_sqrt3 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 := by
  sorry

end tan_105_eq_neg2_sub_sqrt3_l284_284667


namespace number_of_two_digit_primes_with_ones_digit_3_l284_284748

-- Definition of two-digit numbers with a ones digit of 3
def two_digit_numbers_with_ones_digit_3 := [13, 23, 33, 43, 53, 63, 73, 83, 93]

-- Definition of prime predicate
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Proof statement
theorem number_of_two_digit_primes_with_ones_digit_3 : 
  let primes := (two_digit_numbers_with_ones_digit_3.filter is_prime) in
  primes.length = 7 :=
by
  sorry

end number_of_two_digit_primes_with_ones_digit_3_l284_284748


namespace radius_of_smaller_molds_l284_284442

noncomputable def hemisphere_volume (r : ℝ) : ℝ := (2/3) * Real.pi * r ^ 3

theorem radius_of_smaller_molds :
  (64 * hemisphere_volume (1/2)) = hemisphere_volume 2 :=
by
  sorry

end radius_of_smaller_molds_l284_284442


namespace problem_statement_l284_284376

variable {x y z : ℝ}

-- Lean 4 statement of the problem
theorem problem_statement (h₀ : 0 ≤ x) (h₁ : x ≤ 1) (h₂ : 0 ≤ y) (h₃ : y ≤ 1) (h₄ : 0 ≤ z) (h₅ : z ≤ 1) :
  (x / (y + z + 1) + y / (z + x + 1) + z / (x + y + 1)) ≤ 1 - (1 - x) * (1 - y) * (1 - z) := by
  sorry

end problem_statement_l284_284376


namespace dark_light_difference_9x9_grid_l284_284024

theorem dark_light_difference_9x9_grid : ∀ (n : ℕ),
  n = 9 → 
  let grid := (Finset.range (n * n)).image (λ i, if (i / n + i % n) % 2 = 0 then 'D' else 'L') in
  (grid.filter (λ s, s = 'D')).card -
  (grid.filter (λ s, s = 'L')).card = 1 :=
by
  intros n hn grid
  sorry

end dark_light_difference_9x9_grid_l284_284024


namespace distance_blown_westward_l284_284253

theorem distance_blown_westward
  (time_traveled_east : ℕ)
  (speed : ℕ)
  (travelled_halfway : Prop)
  (new_location_fraction : ℚ) :
  time_traveled_east = 20 →
  speed = 30 →
  travelled_halfway →
  new_location_fraction = 1 / 3 →
  let distance_traveled_east := speed * time_traveled_east,
      total_distance := 2 * distance_traveled_east,
      new_location_distance := new_location_fraction * total_distance in
  distance_traveled_east - new_location_distance = 200 :=
begin
  intros,
  sorry
end

end distance_blown_westward_l284_284253


namespace b100_mod_50_l284_284093

def b (n : ℕ) : ℕ := 7^n + 9^n

theorem b100_mod_50 : b 100 % 50 = 2 := by
  sorry

end b100_mod_50_l284_284093


namespace plane_through_A_perpendicular_to_BC_l284_284305

-- Define points A, B, and C
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def A : Point3D := ⟨-3, 6, 4⟩
def B : Point3D := ⟨8, -3, 5⟩
def C : Point3D := ⟨10, -3, 7⟩

-- Define the vector BC
def vectorBC (B C : Point3D) : Point3D :=
  ⟨C.x - B.x, C.y - B.y, C.z - B.z⟩

-- Equation of the plane
def planeEquation (p : Point3D) (n : Point3D) (x y z : ℝ) : ℝ :=
  n.x * (x - p.x) + n.y * (y - p.y) + n.z * (z - p.z)

theorem plane_through_A_perpendicular_to_BC : 
  planeEquation A (vectorBC B C) x y z = 0 ↔ x + z - 1 = 0 :=
by
  sorry

end plane_through_A_perpendicular_to_BC_l284_284305


namespace gumball_sharing_l284_284389

theorem gumball_sharing (init_j : ℕ) (init_jq : ℕ) (mult_j : ℕ) (mult_jq : ℕ) :
  init_j = 40 → init_jq = 60 → mult_j = 5 → mult_jq = 3 →
  (init_j + mult_j * init_j + init_jq + mult_jq * init_jq) / 2 = 240 :=
by
  intros h1 h2 h3 h4
  sorry

end gumball_sharing_l284_284389


namespace tan_105_eq_minus_2_minus_sqrt_3_l284_284678

theorem tan_105_eq_minus_2_minus_sqrt_3 :
  Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_eq_minus_2_minus_sqrt_3_l284_284678


namespace cos_seven_pi_over_six_l284_284915

theorem cos_seven_pi_over_six : Real.cos (7 * Real.pi / 6) = -Real.sqrt 3 / 2 :=
by
  -- Place the proof here
  sorry

end cos_seven_pi_over_six_l284_284915


namespace tan_add_tan_105_eq_l284_284659

noncomputable def tan : ℝ → ℝ := sorry -- Use the built-in library later for actual implementation

-- Given conditions
def tan_45_eq : tan 45 = 1 := by sorry
def tan_60_eq : tan 60 = Real.sqrt 3 := by sorry

-- Angle addition formula for tangent
theorem tan_add (a b : ℝ) :
  tan (a + b) = (tan a + tan b) / (1 - tan a * tan b) := by sorry

-- Main theorem to prove
theorem tan_105_eq :
  tan 105 = -2 - Real.sqrt 3 := by sorry

end tan_add_tan_105_eq_l284_284659


namespace tan_105_l284_284606

theorem tan_105 :
  tan 105 = -2 - sqrt 3 :=
by sorry

end tan_105_l284_284606


namespace algebraic_expression_value_l284_284200

theorem algebraic_expression_value (x y : ℝ) (h : x + 2 * y = 2) : 2 * x + 4 * y - 1 = 3 :=
sorry

end algebraic_expression_value_l284_284200


namespace james_total_fish_catch_l284_284950

-- Definitions based on conditions
def weight_trout : ℕ := 200
def weight_salmon : ℕ := weight_trout + (60 * weight_trout / 100)
def weight_tuna : ℕ := 2 * weight_trout
def weight_bass : ℕ := 3 * weight_salmon
def weight_catfish : ℚ := weight_tuna / 3

-- Total weight of the fish James caught
def total_weight_fish : ℚ := 
  weight_trout + weight_salmon + weight_tuna + weight_bass + weight_catfish 

-- The theorem statement
theorem james_total_fish_catch : total_weight_fish = 2013.33 := by
  sorry

end james_total_fish_catch_l284_284950


namespace quadratic_has_distinct_real_roots_l284_284180

theorem quadratic_has_distinct_real_roots {c : ℝ} (h : c < 1 / 4) :
  ∃ (r1 r2 : ℝ), r1 ≠ r2 ∧ (∃ f, f = (λ x : ℝ, x^2 + 2 * x + 4 * c)) ∧ f r1 = 0 ∧ f r2 = 0 :=
by
  sorry

end quadratic_has_distinct_real_roots_l284_284180


namespace find_n_range_l284_284041

theorem find_n_range (m n : ℝ) 
  (h_m : -Real.sqrt 3 ≤ m ∧ m ≤ Real.sqrt 3) :
  (∀ x y z : ℝ, 0 ≤ x^2 + 2 * y^2 + 3 * z^2 + 2 * x * y + 2 * m * z * x + 2 * n * y * z) ↔ 
  (m - Real.sqrt (3 - m^2) ≤ n ∧ n ≤ m + Real.sqrt (3 - m^2)) :=
by
  sorry

end find_n_range_l284_284041


namespace min_value_x_plus_y_l284_284365

theorem min_value_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 4 / y + 1 / x = 4) :
  x + y ≥ 9 / 4 :=
sorry

end min_value_x_plus_y_l284_284365


namespace final_bicycle_price_l284_284435

-- Define conditions 
def original_price : ℝ := 200
def first_discount : ℝ := 0.20
def second_discount : ℝ := 0.25
def price_after_first_discount := original_price * (1 - first_discount)
def final_price := price_after_first_discount * (1 - second_discount)

-- Define the Lean statement to be proven
theorem final_bicycle_price :
  final_price = 120 :=
by
  -- Proof goes here
  sorry

end final_bicycle_price_l284_284435


namespace selling_price_calculation_l284_284149

-- Given conditions
def cost_price : ℚ := 110
def gain_percent : ℚ := 13.636363636363626

-- Theorem Statement
theorem selling_price_calculation : 
  (cost_price * (1 + gain_percent / 100)) = 125 :=
by
  sorry

end selling_price_calculation_l284_284149


namespace tan_105_degree_l284_284616

theorem tan_105_degree : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 := by
  have h_add : ∀ a b : ℝ, Real.tan (a + b) = (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b) :=
    Real.tan_add

  have h_tan_60 : Real.tan (60 * Real.pi / 180) = Real.sqrt 3 := Real.tan_60
  have h_tan_45 : Real.tan (45 * Real.pi / 180) = 1 := Real.tan_45
  
  sorry

end tan_105_degree_l284_284616


namespace solve_for_a_l284_284928

open Set

theorem solve_for_a (a : ℝ) :
  let M := ({a^2, a + 1, -3} : Set ℝ)
  let P := ({a - 3, 2 * a - 1, a^2 + 1} : Set ℝ)
  M ∩ P = {-3} →
  a = -1 :=
by
  intros M P h
  have hM : M = {a^2, a + 1, -3} := rfl
  have hP : P = {a - 3, 2 * a - 1, a^2 + 1} := rfl
  rw [hM, hP] at h
  sorry

end solve_for_a_l284_284928


namespace range_of_a_l284_284932

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 - a*x + 2*a > 0) → 0 < a ∧ a < 8 := 
sorry

end range_of_a_l284_284932


namespace num_satisfying_inequality_l284_284736

theorem num_satisfying_inequality : ∃ (s : Finset ℤ), (∀ n ∈ s, (n + 4) * (n - 8) ≤ 0) ∧ s.card = 13 := by
  sorry

end num_satisfying_inequality_l284_284736


namespace cos_seven_pi_over_six_l284_284913

theorem cos_seven_pi_over_six : Real.cos (7 * Real.pi / 6) = -Real.sqrt 3 / 2 :=
by
  -- Place the proof here
  sorry

end cos_seven_pi_over_six_l284_284913


namespace probability_of_MATHEMATICS_letter_l284_284770

def unique_letters_in_mathematics : Finset Char := {'M', 'A', 'T', 'H', 'E', 'I', 'C', 'S'}

theorem probability_of_MATHEMATICS_letter :
  let total_letters := 26
  let unique_letters_count := unique_letters_in_mathematics.card
  (unique_letters_count / total_letters : ℝ) = 8 / 26 := by
  sorry

end probability_of_MATHEMATICS_letter_l284_284770


namespace tan_105_eq_neg2_sub_sqrt3_l284_284632

theorem tan_105_eq_neg2_sub_sqrt3 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by 
  sorry

end tan_105_eq_neg2_sub_sqrt3_l284_284632


namespace max_value_y_l284_284199

/-- Given x < 0, the maximum value of y = (1 + x^2) / x is -2 -/
theorem max_value_y {x : ℝ} (h : x < 0) : ∃ y, y = 1 + x^2 / x ∧ y ≤ -2 :=
sorry

end max_value_y_l284_284199


namespace tan_105_eq_neg2_sub_sqrt3_l284_284694

-- Define the main theorem to be proven
theorem tan_105_eq_neg2_sub_sqrt3 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_eq_neg2_sub_sqrt3_l284_284694


namespace correct_proposition_l284_284728

noncomputable def f (x : ℝ) : ℝ := 4 * Real.sin (2 * x + Real.pi / 3)

theorem correct_proposition :
  ¬ (∀ x : ℝ, f (x + 2 * Real.pi) = f x) ∧
  ¬ (∀ h : ℝ, f (-Real.pi / 6 + h) = f (-Real.pi / 6 - h)) ∧
  (∀ h : ℝ, f (-5 * Real.pi / 12 + h) = f (-5 * Real.pi / 12 - h)) :=
by sorry

end correct_proposition_l284_284728


namespace increased_amount_is_30_l284_284547

noncomputable def F : ℝ := (3 / 2) * 179.99999999999991
noncomputable def F' : ℝ := (5 / 3) * 179.99999999999991
noncomputable def J : ℝ := 179.99999999999991
noncomputable def increased_amount : ℝ := F' - F

theorem increased_amount_is_30 : increased_amount = 30 :=
by
  -- Placeholder for proof. Actual proof goes here.
  sorry

end increased_amount_is_30_l284_284547


namespace percentage_difference_l284_284381

theorem percentage_difference (x y z : ℝ) (h1 : y = 1.60 * x) (h2 : z = 0.60 * y) :
  abs ((z - x) / z * 100) = 4.17 :=
by
  sorry

end percentage_difference_l284_284381


namespace parabola_sum_l284_284979

theorem parabola_sum (a b c : ℝ)
  (h1 : 4 = a * 1^2 + b * 1 + c)
  (h2 : -1 = a * (-2)^2 + b * (-2) + c)
  (h3 : ∀ x : ℝ, a * x^2 + b * x + c = a * (x + 1)^2 - 2)
  : a + b + c = 5 := by
  sorry

end parabola_sum_l284_284979


namespace total_legs_proof_l284_284956

def johnny_legs : Nat := 2
def son_legs : Nat := 2
def dog_legs : Nat := 4
def number_of_dogs : Nat := 2
def number_of_humans : Nat := 2

def total_legs : Nat :=
  (number_of_dogs * dog_legs) + (number_of_humans * johnny_legs)

theorem total_legs_proof : total_legs = 12 := by
  sorry

end total_legs_proof_l284_284956


namespace problem_abc_value_l284_284792

theorem problem_abc_value 
  (a b c : ℤ)
  (h1 : a > b)
  (h2 : b > c)
  (h3 : c > 0)
  (h4 : Int.gcd b c = 1)
  (h5 : (b + c) % a = 0)
  (h6 : (a + c) % b = 0) :
  a * b * c = 6 :=
sorry

end problem_abc_value_l284_284792


namespace train_crosses_post_in_25_2_seconds_l284_284577

noncomputable def train_crossing_time (speed_kmph : ℝ) (length_m : ℝ) : ℝ :=
  length_m / (speed_kmph * 1000 / 3600)

theorem train_crosses_post_in_25_2_seconds :
  train_crossing_time 40 280.0224 = 25.2 :=
by 
  sorry

end train_crosses_post_in_25_2_seconds_l284_284577


namespace jim_saves_by_buying_gallon_l284_284085

-- Define the conditions as variables
def cost_per_gallon_costco : ℕ := 8
def ounces_per_gallon : ℕ := 128
def cost_per_16oz_bottle_store : ℕ := 3
def ounces_per_bottle : ℕ := 16

-- Define the theorem that needs to be proven
theorem jim_saves_by_buying_gallon (h1 : cost_per_gallon_costco = 8)
                                    (h2 : ounces_per_gallon = 128)
                                    (h3 : cost_per_16oz_bottle_store = 3)
                                    (h4 : ounces_per_bottle = 16) : 
  (8 * 3 - 8) = 16 :=
by sorry

end jim_saves_by_buying_gallon_l284_284085


namespace tan_105_eq_neg_2_sub_sqrt_3_l284_284621

-- Definitions
def angle105 : ℝ := 105 * (Math.pi / 180)
def angle45 : ℝ := 45 * (Math.pi / 180)
def angle60 : ℝ := 60 * (Math.pi / 180)

-- Theorem
theorem tan_105_eq_neg_2_sub_sqrt_3 :
  Real.tan angle105 = -2 - Real.sqrt 3 := by
  sorry

end tan_105_eq_neg_2_sub_sqrt_3_l284_284621


namespace ratio_of_e_to_l_l284_284145

-- Define the conditions
def e (S : ℕ) : ℕ := 4 * S
def l (S : ℕ) : ℕ := 8 * S

-- Prove the main statement
theorem ratio_of_e_to_l (S : ℕ) (h_e : e S = 4 * S) (h_l : l S = 8 * S) : e S / gcd (e S) (l S) / l S / gcd (e S) (l S) = 1 / 2 := by
  sorry

end ratio_of_e_to_l_l284_284145


namespace tan_105_degree_l284_284633

theorem tan_105_degree : Real.tan (Real.pi * 105 / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_degree_l284_284633


namespace count_two_digit_primes_ending_in_3_l284_284742

def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100
def has_ones_digit_3 (n : ℕ) : Prop := n % 10 = 3
def is_prime (n : ℕ) : Prop := nat.prime n
def two_digit_primes_ending_in_3 (n : ℕ) : Prop :=
  is_two_digit n ∧ has_ones_digit_3 n ∧ is_prime n

theorem count_two_digit_primes_ending_in_3 :
  (nat.card { n : ℕ | two_digit_primes_ending_in_3 n } = 6) :=
sorry

end count_two_digit_primes_ending_in_3_l284_284742


namespace inequality_range_l284_284139

noncomputable def f (x : ℝ) : ℝ := |x - 1| + |x - 2|

theorem inequality_range (a b x: ℝ) (h : a ≠ 0) :
  (|a + b| + |a - b|) ≥ |a| * f x → 1 ≤ x ∧ x ≤ 2 :=
by
  intro h1
  unfold f at h1
  sorry

end inequality_range_l284_284139


namespace tan_105_degree_l284_284589

theorem tan_105_degree : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  have tan_add : ∀ (a b : ℝ), Real.tan (a + b) = (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b) :=
    sorry
  have tan_45 := Real.tan (45 * Real.pi / 180)
  have tan_60 := Real.tan (60 * Real.pi / 180)
  have tan_45_value : tan_45 = 1 := sorry
  have tan_60_value : tan_60 = Real.sqrt 3 := sorry
  sorry

end tan_105_degree_l284_284589


namespace barbi_monthly_loss_l284_284333

variable (x : Real)

theorem barbi_monthly_loss : 
  (∃ x : Real, 12 * x = 99 - 81) → x = 1.5 :=
by
  sorry

end barbi_monthly_loss_l284_284333


namespace daphney_potatoes_l284_284026

theorem daphney_potatoes (cost_per_2kg : ℕ) (total_paid : ℕ) (amount_per_kg : ℕ) (kg_bought : ℕ) 
  (h1 : cost_per_2kg = 6) (h2 : total_paid = 15) (h3 : amount_per_kg = cost_per_2kg / 2) 
  (h4 : kg_bought = total_paid / amount_per_kg) : kg_bought = 5 :=
by
  sorry

end daphney_potatoes_l284_284026


namespace initial_number_18_l284_284855

theorem initial_number_18 (N : ℤ) (h : ∃ k : ℤ, N + 5 = 23 * k) : N = 18 := 
sorry

end initial_number_18_l284_284855


namespace number_of_ordered_pairs_l284_284039

theorem number_of_ordered_pairs : 
  ∃ n, n = 325 ∧ ∀ (a b : ℤ), 
    1 ≤ a ∧ a ≤ 50 ∧ a % 2 = 1 ∧ 
    0 ≤ b ∧ b % 2 = 0 ∧ 
    ∃ r s : ℤ, r + s = -a ∧ r * s = b :=
sorry

end number_of_ordered_pairs_l284_284039


namespace total_chocolates_distributed_l284_284438

theorem total_chocolates_distributed 
  (boys girls : ℕ)
  (chocolates_per_boy chocolates_per_girl : ℕ)
  (h_boys : boys = 60)
  (h_girls : girls = 60)
  (h_chocolates_per_boy : chocolates_per_boy = 2)
  (h_chocolates_per_girl : chocolates_per_girl = 3) : 
  boys * chocolates_per_boy + girls * chocolates_per_girl = 300 :=
by {
  sorry
}

end total_chocolates_distributed_l284_284438


namespace number_of_croutons_l284_284235

def lettuce_calories : ℕ := 30
def cucumber_calories : ℕ := 80
def crouton_calories : ℕ := 20
def total_salad_calories : ℕ := 350

theorem number_of_croutons : 
  ∃ n : ℕ, n * crouton_calories = total_salad_calories - (lettuce_calories + cucumber_calories) ∧ n = 12 :=
by
  sorry

end number_of_croutons_l284_284235


namespace two_digit_integers_congruent_to_2_mod_4_l284_284069

theorem two_digit_integers_congruent_to_2_mod_4 :
  let S := { n : ℕ | 10 ≤ n ∧ n ≤ 99 ∧ (n % 4 = 2) } in
  S.finite ∧ S.to_finset.card = 23 :=
by
  sorry

end two_digit_integers_congruent_to_2_mod_4_l284_284069


namespace Jason_toys_correct_l284_284511

variable (R Jn Js : ℕ)

def Rachel_toys : ℕ := 1

def John_toys (R : ℕ) : ℕ := R + 6

def Jason_toys (Jn : ℕ) : ℕ := 3 * Jn

theorem Jason_toys_correct (hR : R = 1) (hJn : Jn = John_toys R) (hJs : Js = Jason_toys Jn) : Js = 21 :=
by
  sorry

end Jason_toys_correct_l284_284511


namespace no_consecutive_integer_sum_to_36_l284_284064

theorem no_consecutive_integer_sum_to_36 :
  ∀ (a n : ℕ), n ≥ 2 → (n * a + n * (n - 1) / 2) = 36 → false :=
by
  sorry

end no_consecutive_integer_sum_to_36_l284_284064


namespace contrapositive_of_quadratic_l284_284265

theorem contrapositive_of_quadratic (m : ℝ) :
  (m > 0 → ∃ x : ℝ, x^2 + x - m = 0) ↔ (¬∃ x : ℝ, x^2 + x - m = 0 → m ≤ 0) :=
by
  sorry

end contrapositive_of_quadratic_l284_284265


namespace f_neg_two_l284_284823

noncomputable def f : ℝ → ℝ := sorry

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

variables (f_odd : is_odd_function f)
variables (f_two : f 2 = 2)

theorem f_neg_two : f (-2) = -2 :=
by
  -- Given that f is an odd function and f(2) = 2
  sorry

end f_neg_two_l284_284823


namespace derivative_is_even_then_b_eq_zero_l284_284349

variable (a b c : ℝ)
def f (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + 2

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x + c

-- The statement that the derivative is an even function
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g x = g (-x)

-- Our main theorem
theorem derivative_is_even_then_b_eq_zero : is_even (f' a b c) → b = 0 :=
by
  intro h
  have h1 := h 1
  have h2 := h (-1)
  sorry

end derivative_is_even_then_b_eq_zero_l284_284349


namespace shawn_red_pebbles_l284_284973

variable (Total : ℕ)
variable (B : ℕ)
variable (Y : ℕ)
variable (P : ℕ)
variable (G : ℕ)

theorem shawn_red_pebbles (h1 : Total = 40)
                          (h2 : B = 13)
                          (h3 : B - Y = 7)
                          (h4 : P = Y)
                          (h5 : G = Y)
                          (h6 : 3 * Y + B = Total)
                          : Total - (B + P + Y + G) = 9 :=
by
 sorry

end shawn_red_pebbles_l284_284973


namespace smallest_odd_number_with_five_different_prime_factors_l284_284852

theorem smallest_odd_number_with_five_different_prime_factors :
  ∃ (n : ℕ), (∀ p, prime p → p ∣ n → p ≠ 2) ∧ (nat.factors n).length = 5 ∧ ∀ m, (∀ p, prime p → p ∣ m → p ≠ 2) ∧ (nat.factors m).length = 5 → n ≤ m :=
  ⟨15015, 
  begin
    sorry
  end⟩

end smallest_odd_number_with_five_different_prime_factors_l284_284852


namespace newspaper_cost_over_8_weeks_l284_284214

def cost (day : String) : Real := 
  if day = "Sunday" then 2.00 
  else if day = "Wednesday" ∨ day = "Thursday" ∨ day = "Friday" then 0.50 
  else 0

theorem newspaper_cost_over_8_weeks : 
  (8 * ((cost "Wednesday" + cost "Thursday" + cost "Friday") + cost "Sunday")) = 28.00 :=
  by sorry

end newspaper_cost_over_8_weeks_l284_284214


namespace solution_set_inequality_l284_284497

theorem solution_set_inequality (a c : ℝ)
  (h : ∀ x : ℝ, (ax^2 + 2*x + c < 0) ↔ (x < -1/3 ∨ x > 1/2)) :
  (∀ x : ℝ, (cx^2 - 2*x + a ≤ 0) ↔ (-2 ≤ x ∧ x ≤ 3)) :=
sorry

end solution_set_inequality_l284_284497


namespace cos_seven_pi_over_six_l284_284908

theorem cos_seven_pi_over_six : Real.cos (7 * Real.pi / 6) = -Real.sqrt 3 / 2 :=
by
  sorry

end cos_seven_pi_over_six_l284_284908


namespace maria_baggies_count_l284_284795

def total_cookies (chocolate_chip : ℕ) (oatmeal : ℕ) : ℕ :=
  chocolate_chip + oatmeal

def baggies_count (total_cookies : ℕ) (cookies_per_bag : ℕ) : ℕ :=
  total_cookies / cookies_per_bag

theorem maria_baggies_count :
  let choco_chip := 2
  let oatmeal := 16
  let cookies_per_bag := 3
  baggies_count (total_cookies choco_chip oatmeal) cookies_per_bag = 6 :=
by
  sorry

end maria_baggies_count_l284_284795


namespace graph_is_point_l284_284030

theorem graph_is_point : ∀ x y : ℝ, x^2 + 3 * y^2 - 4 * x - 6 * y + 7 = 0 ↔ (x = 2 ∧ y = 1) :=
by
  sorry

end graph_is_point_l284_284030


namespace berry_saturday_reading_l284_284159

-- Given data
def sunday_pages := 43
def monday_pages := 65
def tuesday_pages := 28
def wednesday_pages := 0
def thursday_pages := 70
def friday_pages := 56
def average_goal := 50
def days_in_week := 7

-- Calculate total pages to meet the weekly goal
def weekly_goal := days_in_week * average_goal

-- Calculate pages read so far from Sunday to Friday
def pages_read := sunday_pages + monday_pages + tuesday_pages + wednesday_pages + thursday_pages + friday_pages

-- Calculate required pages to read on Saturday
def saturday_pages_required := weekly_goal - pages_read

-- The theorem statement: Berry needs to read 88 pages on Saturday.
theorem berry_saturday_reading : saturday_pages_required = 88 := 
by {
  -- The proof is omitted as per the instructions
  sorry
}

end berry_saturday_reading_l284_284159


namespace atLeastOneNotLessThanTwo_l284_284364

open Real

theorem atLeastOneNotLessThanTwo (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a + 1 / b < 2 ∧ b + 1 / c < 2 ∧ c + 1 / a < 2) → False := 
by
  sorry

end atLeastOneNotLessThanTwo_l284_284364


namespace combined_teaching_experience_l284_284786

def james_teaching_years : ℕ := 40
def partner_teaching_years : ℕ := james_teaching_years - 10

theorem combined_teaching_experience : james_teaching_years + partner_teaching_years = 70 :=
by
  sorry

end combined_teaching_experience_l284_284786


namespace min_dot_product_on_hyperbola_l284_284730

open Real

theorem min_dot_product_on_hyperbola :
  ∀ (P : ℝ × ℝ), (P.1 ≥ 1 ∧ P.1^2 - (P.2^2) / 3 = 1) →
  let PA1 := (P.1 + 1, P.2)
  let PF2 := (P.1 - 2, P.2)
  ∃ m : ℝ, m = -2 ∧ PA1.1 * PF2.1 + PA1.2 * PF2.2 = m :=
by
  intros P h
  let PA1 := (P.1 + 1, P.2)
  let PF2 := (P.1 - 2, P.2)
  use -2
  sorry

end min_dot_product_on_hyperbola_l284_284730


namespace fraction_meaningful_if_and_only_if_l284_284075

theorem fraction_meaningful_if_and_only_if {x : ℝ} : (2 * x - 1 ≠ 0) ↔ (x ≠ 1 / 2) :=
by
  sorry

end fraction_meaningful_if_and_only_if_l284_284075


namespace graveyard_bones_count_l284_284774

def total_skeletons : ℕ := 20
def half_total (n : ℕ) : ℕ := n / 2
def skeletons_adult_women : ℕ := half_total total_skeletons
def remaining_skeletons : ℕ := total_skeletons - skeletons_adult_women
def even_split (n : ℕ) : ℕ := n / 2
def skeletons_adult_men : ℕ := even_split remaining_skeletons
def skeletons_children : ℕ := even_split remaining_skeletons

def bones_per_woman : ℕ := 20
def bones_per_man : ℕ := bones_per_woman + 5
def bones_per_child : ℕ := bones_per_woman / 2

def total_bones_adult_women : ℕ := skeletons_adult_women * bones_per_woman
def total_bones_adult_men : ℕ := skeletons_adult_men * bones_per_man
def total_bones_children : ℕ := skeletons_children * bones_per_child

def total_bones_in_graveyard : ℕ := total_bones_adult_women + total_bones_adult_men + total_bones_children

theorem graveyard_bones_count : total_bones_in_graveyard = 375 := by
  sorry

end graveyard_bones_count_l284_284774


namespace arithmetic_progression_11th_term_l284_284502

theorem arithmetic_progression_11th_term:
  ∀ (a d : ℝ), (15 / 2) * (2 * a + 14 * d) = 56.25 → a + 6 * d = 3.25 → a + 10 * d = 5.25 :=
by
  intros a d h_sum h_7th
  sorry

end arithmetic_progression_11th_term_l284_284502


namespace michael_twenty_dollar_bills_l284_284966

theorem michael_twenty_dollar_bills (total_amount : ℕ) (denomination : ℕ) 
  (h_total : total_amount = 280) (h_denom : denomination = 20) : 
  total_amount / denomination = 14 := by
  sorry

end michael_twenty_dollar_bills_l284_284966


namespace total_population_l284_284312

theorem total_population (x T : ℝ) (h : 128 = (x / 100) * (50 / 100) * T) : T = 25600 / x :=
by
  sorry

end total_population_l284_284312


namespace range_of_m_l284_284781

def is_ellipse (m : ℝ) : Prop :=
  ∀ x y : ℝ, m * (x^2 + y^2 + 2*y + 1) = (x - 2*y + 3)^2

theorem range_of_m (m : ℝ) (h : is_ellipse m) : m > 5 :=
sorry

end range_of_m_l284_284781


namespace equivalence_of_statements_l284_284788

theorem equivalence_of_statements
  {G : Type*} [group G] [fintype G]
  (K : conjugacy_class G) (hK : group.closure (set_of (λ x, x ∈ K)) = ⊤) :
  (∃ m : ℕ, ∀ g : G, ∃ ks : fin m → G, (∀ i : fin m, ks i ∈ K) ∧ g = (list.of_fn ks).prod) ↔
  (G = commutator_subgroup G) :=
sorry

end equivalence_of_statements_l284_284788


namespace quadratic_two_distinct_real_roots_l284_284185

theorem quadratic_two_distinct_real_roots (c : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2 * x₁ + 4 * c = 0 ∧ x₂^2 + 2 * x₂ + 4 * c = 0) ↔ c < 1 / 4 :=
sorry

end quadratic_two_distinct_real_roots_l284_284185


namespace tan_105_degree_l284_284637

theorem tan_105_degree : Real.tan (Real.pi * 105 / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_degree_l284_284637


namespace cheese_pops_count_l284_284331

-- Define the number of hotdogs, chicken nuggets, and total portions
def hotdogs : ℕ := 30
def chicken_nuggets : ℕ := 40
def total_portions : ℕ := 90

-- Define the number of bite-sized cheese pops
def cheese_pops : ℕ := total_portions - hotdogs - chicken_nuggets

-- Theorem to prove that the number of bite-sized cheese pops Andrew brought is 20
theorem cheese_pops_count :
  cheese_pops = 20 :=
by
  -- The following proof is omitted
  sorry

end cheese_pops_count_l284_284331


namespace total_canoes_built_l284_284161

-- Definitions of conditions
def initial_canoes : ℕ := 8
def common_ratio : ℕ := 2
def number_of_months : ℕ := 6

-- Sum of a geometric sequence formula
-- Sₙ = a * (r^n - 1) / (r - 1)
def sum_of_geometric_sequence (a r n : ℕ) : ℕ := 
  a * (r^n - 1) / (r - 1)

-- Statement to prove
theorem total_canoes_built : 504 = sum_of_geometric_sequence initial_canoes common_ratio number_of_months := 
  by
  sorry

end total_canoes_built_l284_284161


namespace tan_105_eq_neg2_sub_sqrt3_l284_284668

theorem tan_105_eq_neg2_sub_sqrt3 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 := by
  sorry

end tan_105_eq_neg2_sub_sqrt3_l284_284668


namespace MrWillamTaxPercentage_l284_284466

-- Definitions
def TotalTaxCollected : ℝ := 3840
def MrWillamTax : ℝ := 480

-- Theorem Statement
theorem MrWillamTaxPercentage :
  (MrWillamTax / TotalTaxCollected) * 100 = 12.5 :=
by
  sorry

end MrWillamTaxPercentage_l284_284466


namespace least_integer_square_eq_double_plus_64_l284_284300

theorem least_integer_square_eq_double_plus_64 :
  ∃ x : ℤ, x^2 = 2 * x + 64 ∧ ∀ y : ℤ, y^2 = 2 * y + 64 → y ≥ x → x = -8 :=
by
  sorry

end least_integer_square_eq_double_plus_64_l284_284300


namespace solve_fractional_eq_l284_284107

theorem solve_fractional_eq (x : ℝ) (hx1 : x ≠ 1 / 3) (hx2 : x ≠ -3) :
  (3 * x + 2) / (3 * x * x + 8 * x - 3) = (3 * x) / (3 * x - 1) ↔ 
  (x = -1 + (Real.sqrt 15) / 3) ∨ (x = -1 - (Real.sqrt 15) / 3) := 
by 
  sorry

end solve_fractional_eq_l284_284107


namespace numbers_between_sixty_and_seventy_divide_2_pow_48_minus_1_l284_284984

theorem numbers_between_sixty_and_seventy_divide_2_pow_48_minus_1 :
  (63 ∣ 2^48 - 1) ∧ (65 ∣ 2^48 - 1) := 
by
  sorry

end numbers_between_sixty_and_seventy_divide_2_pow_48_minus_1_l284_284984


namespace valid_m_values_l284_284231

theorem valid_m_values (m : ℝ) :
  (∃ x y : ℝ, x^2 + y^2 - 2*m*x - 2*m*y + 2*m^2 + m - 1 = 0) → m < 1 :=
by
  sorry

end valid_m_values_l284_284231


namespace range_of_a_l284_284732

-- Define the function f(x) = x^2 - 3x
def f (x : ℝ) : ℝ := x^2 - 3 * x

-- Define the interval as a closed interval from -1 to 1
def interval : Set ℝ := Set.Icc (-1) (1)

-- State the main proposition
theorem range_of_a (a : ℝ) :
  (∃ x ∈ interval, -x^2 + 3 * x + a > 0) ↔ a > -2 :=
by
  sorry

end range_of_a_l284_284732


namespace smallest_odd_number_with_five_different_prime_factors_l284_284843

noncomputable def smallest_odd_with_five_prime_factors : Nat :=
  3 * 5 * 7 * 11 * 13

theorem smallest_odd_number_with_five_different_prime_factors :
  smallest_odd_with_five_prime_factors = 15015 :=
by
  sorry -- Proof to be filled in later

end smallest_odd_number_with_five_different_prime_factors_l284_284843


namespace range_of_m_l284_284190

noncomputable def P (x : ℝ) : Prop := x^2 - 8*x - 20 ≤ 0
noncomputable def Q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0 ∧ m > 0

theorem range_of_m (m : ℝ) : 
  (∀ x, P x → Q x m ∧ P x) ↔ (0 < m ∧ m ≤ 3) :=
sorry

end range_of_m_l284_284190


namespace probability_closer_to_center_radius6_eq_1_4_l284_284011

noncomputable def probability_closer_to_center (radius : ℝ) (r_inner : ℝ) :=
    let area_outer := Real.pi * radius ^ 2
    let area_inner := Real.pi * r_inner ^ 2
    area_inner / area_outer

theorem probability_closer_to_center_radius6_eq_1_4 :
    probability_closer_to_center 6 3 = 1 / 4 := by
    sorry

end probability_closer_to_center_radius6_eq_1_4_l284_284011


namespace table_price_l284_284153

theorem table_price (C T : ℝ) (h1 : 2 * C + T = 0.6 * (C + 2 * T)) (h2 : C + T = 96) : T = 84 := by
  sorry

end table_price_l284_284153


namespace cost_to_paint_floor_l284_284424

-- Define the conditions
def length_more_than_breadth_by_200_percent (L B : ℝ) : Prop :=
L = 3 * B

def length_of_floor := 23
def cost_per_sq_meter := 3

-- Prove the cost to paint the floor
theorem cost_to_paint_floor (B : ℝ) (L : ℝ) 
    (h1: length_more_than_breadth_by_200_percent L B) (h2: L = length_of_floor) 
    (rate: ℝ) (h3: rate = cost_per_sq_meter) :
    rate * (L * B) = 529.23 :=
by
  -- intermediate steps would go here
  sorry

end cost_to_paint_floor_l284_284424


namespace find_N_l284_284475

variables (k N : ℤ)

theorem find_N (h : ((k * N + N) / N - N) = k - 2021) : N = 2022 :=
by
  sorry

end find_N_l284_284475


namespace negation_proposition_l284_284982

theorem negation_proposition :
  (\neg (\forall x: ℝ, (0 < x → sin x < x)) = 
  (∃ x0: ℝ, (0 < x0 ∧ sin x0 ≥ x0))) := by
sorry

end negation_proposition_l284_284982


namespace savings_by_buying_gallon_l284_284083

def gallon_to_ounces : ℕ := 128
def bottle_volume_ounces : ℕ := 16
def cost_gallon : ℕ := 8
def cost_bottle : ℕ := 3

theorem savings_by_buying_gallon :
  (cost_bottle * (gallon_to_ounces / bottle_volume_ounces)) - cost_gallon = 16 := 
by
  sorry

end savings_by_buying_gallon_l284_284083


namespace speed_of_car_B_is_correct_l284_284889

def carB_speed : ℕ := 
  let speedA := 50 -- Car A's speed in km/hr
  let timeA := 6 -- Car A's travel time in hours
  let ratio := 3 -- The ratio of distances between Car A and Car B
  let distanceA := speedA * timeA -- Calculate Car A's distance
  let timeB := 1 -- Car B's travel time in hours
  let distanceB := distanceA / ratio -- Calculate Car B's distance
  distanceB / timeB -- Calculate Car B's speed

theorem speed_of_car_B_is_correct : carB_speed = 100 := by
  sorry

end speed_of_car_B_is_correct_l284_284889


namespace fuel_efficiency_l284_284025

noncomputable def gas_cost_per_gallon : ℝ := 4
noncomputable def money_spent_on_gas : ℝ := 42
noncomputable def miles_traveled : ℝ := 336

theorem fuel_efficiency : (miles_traveled / (money_spent_on_gas / gas_cost_per_gallon)) = 32 := by
  sorry

end fuel_efficiency_l284_284025


namespace spherical_to_rectangular_l284_284898

theorem spherical_to_rectangular
  (ρ θ φ : ℝ)
  (ρ_eq : ρ = 10)
  (θ_eq : θ = 5 * Real.pi / 4)
  (φ_eq : φ = Real.pi / 4) :
  let x := ρ * Real.sin φ * Real.cos θ
  let y := ρ * Real.sin φ * Real.sin θ
  let z := ρ * Real.cos φ
  (x, y, z) = (-5, -5, 5 * Real.sqrt 2) :=
by
  sorry

end spherical_to_rectangular_l284_284898


namespace desired_alcohol_percentage_is_18_l284_284261

noncomputable def final_alcohol_percentage (volume_x volume_y : ℕ) (percentage_x percentage_y : ℚ) : ℚ :=
  let total_volume := (volume_x + volume_y)
  let total_alcohol := (percentage_x * volume_x + percentage_y * volume_y)
  total_alcohol / total_volume * 100

theorem desired_alcohol_percentage_is_18 : 
  final_alcohol_percentage 300 200 0.10 0.30 = 18 := 
  sorry

end desired_alcohol_percentage_is_18_l284_284261


namespace cookie_price_ratio_l284_284491

theorem cookie_price_ratio (c b : ℝ) (h1 : 6 * c + 5 * b = 3 * (3 * c + 27 * b)) : c = (4 / 5) * b :=
sorry

end cookie_price_ratio_l284_284491


namespace positive_roots_of_x_pow_x_eq_one_over_sqrt_two_l284_284027

theorem positive_roots_of_x_pow_x_eq_one_over_sqrt_two (x : ℝ) (h : x > 0) : 
  (x^x = 1 / Real.sqrt 2) ↔ (x = 1 / 2 ∨ x = 1 / 4) := by
  sorry

end positive_roots_of_x_pow_x_eq_one_over_sqrt_two_l284_284027


namespace days_to_complete_work_l284_284532

variable {P W D : ℕ}

axiom condition_1 : 2 * P * 3 = W / 2
axiom condition_2 : P * D = W

theorem days_to_complete_work : D = 12 :=
by
  -- As an axiom or sorry is used, the proof is omitted.
  sorry

end days_to_complete_work_l284_284532


namespace percentage_increase_l284_284572

theorem percentage_increase (a : ℕ) (x : ℝ) (b : ℝ) (r : ℝ) 
    (h1 : a = 1500) 
    (h2 : r = 0.6) 
    (h3 : b = 1080) 
    (h4 : a * (1 + x / 100) * r = b) : 
    x = 20 := 
by 
  sorry

end percentage_increase_l284_284572


namespace newspaper_cost_over_8_weeks_l284_284213

def cost (day : String) : Real := 
  if day = "Sunday" then 2.00 
  else if day = "Wednesday" ∨ day = "Thursday" ∨ day = "Friday" then 0.50 
  else 0

theorem newspaper_cost_over_8_weeks : 
  (8 * ((cost "Wednesday" + cost "Thursday" + cost "Friday") + cost "Sunday")) = 28.00 :=
  by sorry

end newspaper_cost_over_8_weeks_l284_284213


namespace binom_2n_2_eq_n_2n_minus_1_l284_284431

theorem binom_2n_2_eq_n_2n_minus_1 (n : ℕ) (h : n > 0) : 
  (Nat.choose (2 * n) 2) = n * (2 * n - 1) := 
sorry

end binom_2n_2_eq_n_2n_minus_1_l284_284431


namespace white_roses_count_l284_284886

def total_flowers : ℕ := 6284
def red_roses : ℕ := 1491
def yellow_carnations : ℕ := 3025
def white_roses : ℕ := total_flowers - (red_roses + yellow_carnations)

theorem white_roses_count :
  white_roses = 1768 := by
  sorry

end white_roses_count_l284_284886


namespace correct_location_l284_284166

variable (A B C D : Prop)

axiom student_A_statement : ¬ A ∧ B
axiom student_B_statement : ¬ B ∧ C
axiom student_C_statement : ¬ B ∧ ¬ D
axiom ms_Hu_response : 
  ( (¬ A ∧ B = true) ∨ (¬ B ∧ C = true) ∨ (¬ B ∧ ¬ D = true) ) ∧ 
  ( (¬ A ∧ B = false) ∨ (¬ B ∧ C = false) ∨ (¬ B ∧ ¬ D = false) = false ) ∧ 
  ( (¬ A ∧ B ∨ ¬ B ∧ C ∨ ¬ B ∧ ¬ D) -> false )

theorem correct_location : B ∨ A := 
sorry

end correct_location_l284_284166


namespace MaireadRan40Miles_l284_284303

def MaireadRanMiles (R : ℝ) (W : ℝ) (J : ℝ) : Prop :=
  W = (3 / 5) * R ∧ J = 3 * R ∧ R + W + J = 184

theorem MaireadRan40Miles : ∃ R W J, MaireadRanMiles R W J ∧ R = 40 :=
by sorry

end MaireadRan40Miles_l284_284303


namespace area_CDM_l284_284554

noncomputable def AC := 8
noncomputable def BC := 15
noncomputable def AB := 17
noncomputable def M := (AC + BC) / 2
noncomputable def AD := 17
noncomputable def BD := 17

theorem area_CDM (h₁ : AC = 8)
                 (h₂ : BC = 15)
                 (h₃ : AB = 17)
                 (h₄ : AD = 17)
                 (h₅ : BD = 17)
                 : ∃ (m n p : ℕ),
                   m = 121 ∧
                   n = 867 ∧
                   p = 136 ∧
                   m + n + p = 1124 ∧
                   ∃ (area_CDM : ℚ), 
                   area_CDM = (121 * Real.sqrt 867) / 136 :=
by
  sorry

end area_CDM_l284_284554


namespace barry_shirt_discount_l284_284335

theorem barry_shirt_discount 
  (original_price : ℤ) 
  (discount_percent : ℤ) 
  (discounted_price : ℤ) 
  (h1 : original_price = 80) 
  (h2 : discount_percent = 15)
  (h3 : discounted_price = original_price - (discount_percent * original_price / 100)) : 
  discounted_price = 68 :=
sorry

end barry_shirt_discount_l284_284335


namespace total_spent_is_13_l284_284698

-- Let cost_cb represent the cost of the candy bar
def cost_cb : ℕ := 7

-- Let cost_ch represent the cost of the chocolate
def cost_ch : ℕ := 6

-- Define the total cost as the sum of cost_cb and cost_ch
def total_cost : ℕ := cost_cb + cost_ch

-- Theorem to prove the total cost equals $13
theorem total_spent_is_13 : total_cost = 13 := by
  sorry

end total_spent_is_13_l284_284698


namespace arithmetic_sequence_next_term_perfect_square_sequence_next_term_l284_284471

theorem arithmetic_sequence_next_term (a : ℕ → ℕ) (n : ℕ) (h₀ : a 0 = 0) (h₁ : ∀ n, a (n + 1) = a n + 3) :
  a 5 = 15 :=
by sorry

theorem perfect_square_sequence_next_term (b : ℕ → ℕ) (k : ℕ) (h₀ : ∀ k, b k = (k + 1) * (k + 1)) :
  b 5 = 36 :=
by sorry

end arithmetic_sequence_next_term_perfect_square_sequence_next_term_l284_284471


namespace general_terms_sum_c_n_l284_284362

noncomputable def a_n (n : ℕ) : ℝ := 2 * n - 1

def b_seq (T : ℕ → ℝ) : ℕ → ℝ
| 0     := 3
| (n+1) := 2 * T n + 3

noncomputable def b_n (n : ℕ) : ℝ := 3 ^ n

noncomputable def c_n (n : ℕ) : ℝ := (a_n n) / (b_n n)

noncomputable def M_n (n : ℕ) : ℝ := 1 - (n + 1) / (3 ^ n)

theorem general_terms :
  (∀ n, a_n n = 2 * n - 1) ∧
  (∀ n, b_n n = 3 ^ n) := sorry

theorem sum_c_n (n : ℕ) :
  (∑ i in finset.range n, c_n (i + 1)) = M_n n := sorry

end general_terms_sum_c_n_l284_284362


namespace quadratic_distinct_roots_l284_284177

theorem quadratic_distinct_roots (c : ℝ) (h : c < 1 / 4) : 
  ∃ r1 r2 : ℝ, r1 ≠ r2 ∧ (r1 * r1 + 2 * r1 + 4 * c = 0) ∧ (r2 * r2 + 2 * r2 + 4 * c = 0) :=
by 
sorry

end quadratic_distinct_roots_l284_284177


namespace simplify_fraction_eq_l284_284806

theorem simplify_fraction_eq : (180 / 270 : ℚ) = 2 / 3 :=
by
  sorry

end simplify_fraction_eq_l284_284806


namespace rohan_salary_l284_284307

variable (S : ℝ)

theorem rohan_salary (h₁ : (0.20 * S = 2500)) : S = 12500 :=
by
  sorry

end rohan_salary_l284_284307


namespace probability_of_four_card_success_l284_284869

example (cards : Fin 4) (pins : Fin 4) {attempts : ℕ}
  (h1 : ∀ (c : Fin 4) (p : Fin 4), attempts ≤ 3)
  (h2 : ∀ (c : Fin 4), ∃ (p : Fin 4), p ≠ c ∧ attempts ≤ 3) :
  ∃ (three_cards : Fin 3), attempts ≤ 3 :=
sorry

noncomputable def probability_success :
  ℚ := 23 / 24

theorem probability_of_four_card_success :
  probability_success = 23 / 24 :=
sorry

end probability_of_four_card_success_l284_284869


namespace geometric_sequence_is_alternating_l284_284080

theorem geometric_sequence_is_alternating (a : ℕ → ℝ) (q : ℝ)
  (h1 : a 1 + a 2 = -3 / 2)
  (h2 : a 4 + a 5 = 12)
  (hg : ∀ n, a (n + 1) = q * a n) :
  ∃ q, q < 0 ∧ ∀ n, a n * a (n + 1) ≤ 0 :=
by sorry

end geometric_sequence_is_alternating_l284_284080


namespace number_of_two_digit_integers_congruent_to_2_mod_4_l284_284067

theorem number_of_two_digit_integers_congruent_to_2_mod_4 : 
  let k_values := {k : ℤ | 2 ≤ k ∧ k ≤ 24} in 
  k_values.card = 23 :=
by
  let k_values := {k : ℤ | 2 ≤ k ∧ k ≤ 24}
  have : k_values = finset.Icc 2 24 := by sorry
  rw [this, finset.card_Icc]
  norm_num
  sorry

end number_of_two_digit_integers_congruent_to_2_mod_4_l284_284067


namespace lcm_is_600_l284_284470

def lcm_of_24_30_40_50_60 : ℕ :=
  Nat.lcm 24 (Nat.lcm 30 (Nat.lcm 40 (Nat.lcm 50 60)))

theorem lcm_is_600 : lcm_of_24_30_40_50_60 = 600 := by
  sorry

end lcm_is_600_l284_284470


namespace Jason_toys_correct_l284_284512

variable (R Jn Js : ℕ)

def Rachel_toys : ℕ := 1

def John_toys (R : ℕ) : ℕ := R + 6

def Jason_toys (Jn : ℕ) : ℕ := 3 * Jn

theorem Jason_toys_correct (hR : R = 1) (hJn : Jn = John_toys R) (hJs : Js = Jason_toys Jn) : Js = 21 :=
by
  sorry

end Jason_toys_correct_l284_284512


namespace barry_shirt_discount_l284_284334

theorem barry_shirt_discount 
  (original_price : ℤ) 
  (discount_percent : ℤ) 
  (discounted_price : ℤ) 
  (h1 : original_price = 80) 
  (h2 : discount_percent = 15)
  (h3 : discounted_price = original_price - (discount_percent * original_price / 100)) : 
  discounted_price = 68 :=
sorry

end barry_shirt_discount_l284_284334


namespace inverse_proportion_rises_left_to_right_l284_284225

theorem inverse_proportion_rises_left_to_right (k : ℝ) : 
  (∀ x : ℝ, x ≠ 0 → y = k / x → (x > 0 → y rises as x increases)) → k < 0 := 
begin
  sorry
end

end inverse_proportion_rises_left_to_right_l284_284225


namespace berries_per_bird_per_day_l284_284142

theorem berries_per_bird_per_day (birds : ℕ) (total_berries : ℕ) (days : ℕ) (berries_per_bird_per_day : ℕ) 
  (h_birds : birds = 5)
  (h_total_berries : total_berries = 140)
  (h_days : days = 4) :
  berries_per_bird_per_day = 7 :=
  sorry

end berries_per_bird_per_day_l284_284142


namespace three_topping_pizzas_l284_284875

theorem three_topping_pizzas : Nat.choose 8 3 = 56 := by
  sorry

end three_topping_pizzas_l284_284875


namespace second_group_members_l284_284566

theorem second_group_members (total first third : ℕ) (h1 : total = 70) (h2 : first = 25) (h3 : third = 15) :
  (total - first - third) = 30 :=
by
  sorry

end second_group_members_l284_284566


namespace neg_div_neg_eq_pos_division_of_negatives_example_l284_284892

theorem neg_div_neg_eq_pos (a b : Int) (hb : b ≠ 0) : (-a) / (-b) = a / b := by
  -- You can complete the proof here
  sorry

theorem division_of_negatives_example : (-81 : Int) / (-9) = 9 :=
  neg_div_neg_eq_pos 81 9 (by decide)

end neg_div_neg_eq_pos_division_of_negatives_example_l284_284892


namespace algebraic_expression_value_l284_284925

theorem algebraic_expression_value (x y : ℝ) (h : x - 2 * y = -1) : 6 + 2 * x - 4 * y = 4 := by
  sorry

end algebraic_expression_value_l284_284925


namespace B_visits_A_l284_284111

/-- Students A, B, and C were surveyed on whether they have visited cities A, B, and C -/
def student_visits_city (student : Type) (city : Type) : Prop := sorry -- assume there's a definition

variables (A_student B_student C_student : Type) (city_A city_B city_C : Type)

variables 
  -- A's statements
  (A_visits_more_than_B : student_visits_city A_student city_A → ¬ student_visits_city A_student city_B → ∃ city, student_visits_city B_student city ∧ ¬ student_visits_city A_student city)
  (A_not_visit_B : ¬ student_visits_city A_student city_B)
  -- B's statement
  (B_not_visit_C : ¬ student_visits_city B_student city_C)
  -- C's statement
  (all_three_same_city : student_visits_city A_student city_A → student_visits_city B_student city_A → student_visits_city C_student city_A)

theorem B_visits_A : student_visits_city B_student city_A :=
by
  sorry

end B_visits_A_l284_284111


namespace unique_rectangle_dimensions_l284_284926

theorem unique_rectangle_dimensions (a b : ℝ) (h_ab : a < b) :
  ∃! (x y : ℝ), x < a ∧ y < b ∧ x + y = (a + b) / 2 ∧ x * y = a * b / 4 :=
sorry

end unique_rectangle_dimensions_l284_284926


namespace solve_fraction_equation_l284_284531

theorem solve_fraction_equation (x : ℝ) (h : (x + 5) / (x - 3) = 4) : x = 17 / 3 :=
by
  sorry

end solve_fraction_equation_l284_284531


namespace num_of_int_solutions_l284_284941

/-- 
  The number of integer solutions to the equation 
  \((x^3 - x - 1)^{2015} = 1\) is 3.
-/
theorem num_of_int_solutions :
  ∃ n : ℕ, n = 3 ∧ ∀ x : ℤ, (x ^ 3 - x - 1) ^ 2015 = 1 ↔ x = 0 ∨ x = 1 ∨ x = -1 := 
sorry

end num_of_int_solutions_l284_284941


namespace fixed_point_of_parabola_l284_284791

theorem fixed_point_of_parabola (s : ℝ) : ∃ y : ℝ, y = 4 * 3^2 + s * 3 - 3 * s ∧ (3, y) = (3, 36) :=
by
  sorry

end fixed_point_of_parabola_l284_284791


namespace sally_quarters_total_l284_284105

/--
Sally originally had 760 quarters. She received 418 more quarters. 
Prove that the total number of quarters Sally has now is 1178.
-/
theorem sally_quarters_total : 
  let original_quarters := 760
  let additional_quarters := 418
  original_quarters + additional_quarters = 1178 :=
by
  let original_quarters := 760
  let additional_quarters := 418
  show original_quarters + additional_quarters = 1178
  sorry

end sally_quarters_total_l284_284105


namespace base_seven_to_base_ten_l284_284298

theorem base_seven_to_base_ten : 
  let n := 23456 
  ∈ ℕ, nat : ℕ 
  in nat = 6068 := 
by 
  sorry

end base_seven_to_base_ten_l284_284298


namespace positive_difference_of_complementary_ratio_5_1_l284_284275

-- Define angles satisfying the ratio condition and being complementary
def angle_pair (a b : ℝ) : Prop := (a + b = 90) ∧ (a = 5 * b ∨ b = 5 * a)

theorem positive_difference_of_complementary_ratio_5_1 :
  ∃ a b : ℝ, angle_pair a b ∧ abs (a - b) = 60 :=
by
  sorry

end positive_difference_of_complementary_ratio_5_1_l284_284275


namespace cos_seven_pi_over_six_l284_284911

theorem cos_seven_pi_over_six : Real.cos (7 * Real.pi / 6) = -Real.sqrt 3 / 2 :=
by
  sorry

end cos_seven_pi_over_six_l284_284911


namespace Billy_weight_l284_284019

variables (Billy Brad Carl Dave Edgar : ℝ)

-- Conditions
def conditions :=
  Carl = 145 ∧
  Dave = Carl + 8 ∧
  Brad = Dave / 2 ∧
  Billy = Brad + 9 ∧
  Edgar = 3 * Dave ∧
  Edgar = Billy + 20

-- The statement to prove
theorem Billy_weight (Billy Brad Carl Dave Edgar : ℝ) (h : conditions Billy Brad Carl Dave Edgar) : Billy = 85.5 :=
by
  -- Proof would go here
  sorry

end Billy_weight_l284_284019


namespace smallest_odd_number_with_five_different_prime_factors_l284_284853

theorem smallest_odd_number_with_five_different_prime_factors :
  ∃ (n : ℕ), (∀ p, prime p → p ∣ n → p ≠ 2) ∧ (nat.factors n).length = 5 ∧ ∀ m, (∀ p, prime p → p ∣ m → p ≠ 2) ∧ (nat.factors m).length = 5 → n ≤ m :=
  ⟨15015, 
  begin
    sorry
  end⟩

end smallest_odd_number_with_five_different_prime_factors_l284_284853


namespace ted_gathered_10_blue_mushrooms_l284_284887

noncomputable def blue_mushrooms_ted_gathered : ℕ :=
  let bill_red_mushrooms := 12
  let bill_brown_mushrooms := 6
  let ted_green_mushrooms := 14
  let total_white_spotted_mushrooms := 17
  
  let bill_white_spotted_red_mushrooms := bill_red_mushrooms / 2
  let bill_white_spotted_brown_mushrooms := bill_brown_mushrooms

  let total_bill_white_spotted_mushrooms := bill_white_spotted_red_mushrooms + bill_white_spotted_brown_mushrooms
  let ted_white_spotted_mushrooms := total_white_spotted_mushrooms - total_bill_white_spotted_mushrooms

  ted_white_spotted_mushrooms * 2

theorem ted_gathered_10_blue_mushrooms :
  blue_mushrooms_ted_gathered = 10 :=
by
  sorry

end ted_gathered_10_blue_mushrooms_l284_284887


namespace min_expr_value_l284_284169

theorem min_expr_value (a b c : ℝ) (h₀ : b > c) (h₁ : c > a) (h₂ : a > 0) (h₃ : b ≠ 0) :
  (∀ (a b c : ℝ), b > c → c > a → a > 0 → b ≠ 0 → 
   (2 + 6 * a^2 = (a+b)^3 / b^2 + (b-c)^2 / b^2 + (c-a)^3 / b^2) →
   2 <= (a + b)^3 / b^2 + (b - c)^2 / b^2 + (c - a)^3 / b^2) :=
by 
  sorry

end min_expr_value_l284_284169


namespace probability_25_cents_min_l284_284533

-- Define the five coins and their values
def penny := 0.01
def nickel := 0.05
def dime := 0.10
def quarter := 0.25
def halfDollar := 0.50

-- Define a function that computes the total value of heads up coins
def value_heads (results : (Bool × Bool × Bool × Bool × Bool)) : ℝ :=
  let (h₁, h₂, h₃, h₄, h₅) := results 
  (if h₁ then penny else 0) +
  (if h₂ then nickel else 0) +
  (if h₃ then dime else 0) +
  (if h₄ then quarter else 0) +
  (if h₅ then halfDollar else 0)

-- Define the main theorem statement
theorem probability_25_cents_min :
  (∑ results in (finset.univ : finset (Bool × Bool × Bool × Bool × Bool)),
    if value_heads results ≥ 0.25 then (1 : ℝ) else 0) / 32 = 13 / 16 := sorry

end probability_25_cents_min_l284_284533


namespace two_digit_primes_with_ones_digit_3_l284_284761

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def digits (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else let rec f (n : ℕ) : List ℕ :=
    if n = 0 then [] else (n % 10) :: f (n / 10)
  in List.reverse (f n)

def ends_with_3 (n : ℕ) : Prop :=
  digits n = (digits n).init ++ [3]

def two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem two_digit_primes_with_ones_digit_3 :
  (Finset.filter (λ n, is_prime n ∧ ends_with_3 n) (Finset.filter two_digit (Finset.range 100))).card = 6 := by
  sorry

end two_digit_primes_with_ones_digit_3_l284_284761


namespace probability_separation_event_l284_284058

noncomputable def probability_line_separates_circle : ℝ :=
  let c := (2 : ℝ)
  let r := 1
  let lower_bound := -1
  let upper_bound := 1 in
  let p := (1 - real.sqrt 3 / 3 - lower_bound + (1 - real.sqrt 3 / 3)) / (upper_bound - lower_bound) in
  (p : ℝ)

theorem probability_separation_event : 
  ∀ (k : ℝ), k ∈ Icc (-1 : ℝ) (1 : ℝ) →
  (∃ (P : ℝ), 
    (P = probability_line_separates_circle) ∧  
    P = (3 - real.sqrt 3) / 3) := 
by sorry

end probability_separation_event_l284_284058


namespace sum_of_bases_l284_284229

theorem sum_of_bases (R₁ R₂ : ℕ) 
    (h1 : (4 * R₁ + 5) / (R₁^2 - 1) = (3 * R₂ + 4) / (R₂^2 - 1))
    (h2 : (5 * R₁ + 4) / (R₁^2 - 1) = (4 * R₂ + 3) / (R₂^2 - 1)) : 
    R₁ + R₂ = 23 := 
sorry

end sum_of_bases_l284_284229


namespace tan_105_eq_neg2_sub_sqrt3_l284_284690

-- Define the main theorem to be proven
theorem tan_105_eq_neg2_sub_sqrt3 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_eq_neg2_sub_sqrt3_l284_284690


namespace two_digit_primes_with_ones_digit_3_l284_284762

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def digits (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else let rec f (n : ℕ) : List ℕ :=
    if n = 0 then [] else (n % 10) :: f (n / 10)
  in List.reverse (f n)

def ends_with_3 (n : ℕ) : Prop :=
  digits n = (digits n).init ++ [3]

def two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem two_digit_primes_with_ones_digit_3 :
  (Finset.filter (λ n, is_prime n ∧ ends_with_3 n) (Finset.filter two_digit (Finset.range 100))).card = 6 := by
  sorry

end two_digit_primes_with_ones_digit_3_l284_284762


namespace inequality_solution_set_l284_284945

theorem inequality_solution_set (a : ℤ) : 
  (∀ x : ℤ, (1 + a) * x > 1 + a → x < 1) → a < -1 :=
sorry

end inequality_solution_set_l284_284945


namespace count_two_digit_primes_with_ones_3_l284_284758

open Nat

/-- Predicate to check if a number is a two-digit prime with ones digit 3. --/
def two_digit_prime_with_ones_3 (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ n % 10 = 3 ∧ Prime n

/-- Prove that there are exactly 6 two-digit primes with ones digit 3. --/
theorem count_two_digit_primes_with_ones_3 : 
  (Finset.filter two_digit_prime_with_ones_3 (Finset.range 100)).card = 6 := 
  by
  sorry

end count_two_digit_primes_with_ones_3_l284_284758


namespace sum_of_solutions_l284_284779

theorem sum_of_solutions (y x : ℝ) (h1 : y = 7) (h2 : x^2 + y^2 = 100) : 
  x + -x = 0 :=
by
  sorry

end sum_of_solutions_l284_284779


namespace sum_possible_x_l284_284304

noncomputable def sum_of_x (x : ℝ) : ℝ :=
  let lst : List ℝ := [1, 2, 5, 2, 3, 2, x]
  let mean := (1 + 2 + 5 + 2 + 3 + 2 + x) / 7
  let median := 2
  let mode := 2
  if lst = List.reverse lst ∧ mean ≠ mode then
    mean
  else 
    0

theorem sum_possible_x : sum_of_x 1 + sum_of_x 5 = 6 :=
by 
  sorry

end sum_possible_x_l284_284304


namespace relationship_m_n_k_l_l284_284227

-- Definitions based on the conditions
variables (m n k l : ℕ)

-- Condition: Number of teachers (m), Number of students (n)
-- Each teacher teaches exactly k students
-- Any pair of students has exactly l common teachers

theorem relationship_m_n_k_l (h1 : 0 < m) (h2 : 0 < n) (h3 : 0 < k) (h4 : 0 < l)
  (hk : k * (k - 1) / 2 = k * (k - 1) / 2) (hl : n * (n - 1) / 2 = n * (n - 1) / 2) 
  (h5 : m * (k * (k - 1)) = (n * (n - 1)) * l) :
  m * k * (k - 1) = n * (n - 1) * l :=
by 
  sorry

end relationship_m_n_k_l_l284_284227


namespace max_integer_valued_fractions_l284_284037

-- Problem Statement:
-- Given a set of natural numbers from 1 to 22,
-- the maximum number of fractions that can be formed such that each fraction is an integer
-- (where an integer fraction is defined as a/b being an integer if and only if b divides a) is 10.

open Nat

theorem max_integer_valued_fractions : 
  ∀ (S : Finset ℕ), (∀ x, x ∈ S → 1 ≤ x ∧ x ≤ 22) →
  ∃ P : Finset (ℕ × ℕ), (∀ (a b : ℕ), (a, b) ∈ P → b ∣ a) ∧ P.card = 11 → 
  10 ≤ (P.filter (λ p => p.1 % p.2 = 0)).card :=
by
  -- proof goes here
  sorry

end max_integer_valued_fractions_l284_284037


namespace coloring_arithmetic_sequence_exists_l284_284465

open Finset

noncomputable def exists_coloring (A : Finset ℕ) (n : ℕ) : Prop :=
∃ c : ℕ → Prop, (∀ a b : ℕ, a ∈ A → b ∈ A → (a < b) → (b - a) % (n - 1) = 0 → c a ≠ c b)

theorem coloring_arithmetic_sequence_exists :
  ∃ c : ℕ → Prop, ∀ a b : ℕ, a ∈ (finset.range 2018).erase 0 →
    b ∈ (finset.range 2018).erase 0 →
    a < b →
    ((b - a) % (18 - 1) = 0) → (c a ≠ c b) :=
begin
  sorry
end

end coloring_arithmetic_sequence_exists_l284_284465


namespace sampling_probabilities_equal_l284_284192

noncomputable def populationSize (N : ℕ) := N
noncomputable def sampleSize (n : ℕ) := n

def P1 (N n : ℕ) : ℚ := (n : ℚ) / (N : ℚ)
def P2 (N n : ℕ) : ℚ := (n : ℚ) / (N : ℚ)
def P3 (N n : ℕ) : ℚ := (n : ℚ) / (N : ℚ)

theorem sampling_probabilities_equal (N n : ℕ) (hN : N > 0) (hn : n > 0) :
  P1 N n = P2 N n ∧ P2 N n = P3 N n :=
by
  -- Proof steps will go here
  sorry

end sampling_probabilities_equal_l284_284192


namespace tan_105_eq_minus_2_minus_sqrt_3_l284_284675

theorem tan_105_eq_minus_2_minus_sqrt_3 :
  Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_eq_minus_2_minus_sqrt_3_l284_284675


namespace smaller_interior_angle_of_parallelogram_l284_284423

theorem smaller_interior_angle_of_parallelogram (x : ℝ) 
  (h1 : ∃ l, l = x + 90 ∧ x + l = 180) :
  x = 45 :=
by
  obtain ⟨l, hl1, hl2⟩ := h1
  simp only [hl1] at hl2
  linarith

end smaller_interior_angle_of_parallelogram_l284_284423


namespace max_planes_determined_l284_284321

-- Definitions for conditions
variables (Point Line Plane : Type)
variables (l : Line) (A B C : Point)
variables (contains : Point → Line → Prop)
variables (plane_contains_points : Plane → Point → Point → Point → Prop)
variables (plane_contains_line_and_point : Plane → Line → Point → Prop)
variables (non_collinear : Point → Point → Point → Prop)
variables (not_on_line : Point → Line → Prop)

-- Hypotheses based on the conditions
axiom three_non_collinear_points : non_collinear A B C
axiom point_not_on_line (P : Point) : not_on_line P l

-- Goal: Prove that the number of planes is 4
theorem max_planes_determined : 
  ∃ total_planes : ℕ, total_planes = 4 :=
sorry

end max_planes_determined_l284_284321


namespace sum_of_non_solutions_l284_284397

theorem sum_of_non_solutions (A B C x : ℝ) 
  (h : ∀ x, ((x + B) * (A * x + 32)) = 4 * ((x + C) * (x + 8))) :
  (x = -B ∨ x = -8) → x ≠ -B → -B ≠ -8 → x ≠ -8 → x + 8 + B = 0 := 
sorry

end sum_of_non_solutions_l284_284397


namespace billy_unknown_lap_time_l284_284018

theorem billy_unknown_lap_time :
  ∀ (time_first_5_laps time_next_3_laps time_last_lap time_margaret total_time_billy : ℝ) (lap_time_unknown : ℝ),
    time_first_5_laps = 2 ∧
    time_next_3_laps = 4 ∧
    time_last_lap = 2.5 ∧
    time_margaret = 10 ∧
    total_time_billy = time_margaret - 0.5 →
    (time_first_5_laps + time_next_3_laps + time_last_lap + lap_time_unknown = total_time_billy) →
    lap_time_unknown = 1 :=
by
  sorry

end billy_unknown_lap_time_l284_284018


namespace tan_105_eq_minus_2_minus_sqrt_3_l284_284674

theorem tan_105_eq_minus_2_minus_sqrt_3 :
  Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_eq_minus_2_minus_sqrt_3_l284_284674


namespace ellipse_condition_l284_284784

theorem ellipse_condition (m : ℝ) : 
  (∃ x y : ℝ, m * (x^2 + y^2 + 2*y + 1) = (x - 2*y + 3)^2) → m > 5 :=
by
  intro h
  sorry

end ellipse_condition_l284_284784


namespace smallest_odd_with_five_different_prime_factors_l284_284839

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

def has_five_distinct_prime_factors (n : ℕ) : Prop :=
  ∃ a b c d e : ℕ, 
    is_prime a ∧ is_prime b ∧ is_prime c ∧ is_prime d ∧ is_prime e ∧ 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e ∧ 
    n = a * b * c * d * e

theorem smallest_odd_with_five_different_prime_factors : ∃ n : ℕ, 
  is_odd n ∧ has_five_distinct_prime_factors n ∧ ∀ m : ℕ, 
  is_odd m ∧ has_five_distinct_prime_factors m → n ≤ m :=
exists.intro 15015 sorry

end smallest_odd_with_five_different_prime_factors_l284_284839


namespace solve_system1_l284_284416

structure SystemOfEquations :=
  (a₁ a₂ b₁ b₂ c₁ c₂ : ℝ)

def system1 : SystemOfEquations :=
  { a₁ := 1, b₁ := -3, c₁ := 4,
    a₂ := 2, b₂ := -1, c₂ := 3 }

theorem solve_system1 :
  ∃ x y : ℝ, x - 3 * y = 4 ∧ 2 * x - y = 3 ∧ x = 1 ∧ y = -1 :=
by
  sorry

end solve_system1_l284_284416


namespace prob_axisymmetric_and_centrally_symmetric_l284_284125

theorem prob_axisymmetric_and_centrally_symmetric : 
  let card1 := "Line segment"
  let card2 := "Equilateral triangle"
  let card3 := "Parallelogram"
  let card4 := "Isosceles trapezoid"
  let card5 := "Circle"
  let cards := [card1, card2, card3, card4, card5]
  let symmetric_cards := [card1, card5]
  (symmetric_cards.length / cards.length : ℚ) = 2 / 5 :=
by sorry

end prob_axisymmetric_and_centrally_symmetric_l284_284125


namespace percentage_problem_l284_284072

theorem percentage_problem (p x : ℝ) (h1 : (p / 100) * x = 400) (h2 : (120 / 100) * x = 2400) : p = 20 := by
  sorry

end percentage_problem_l284_284072


namespace perfect_squares_of_nat_l284_284958

theorem perfect_squares_of_nat (a b c : ℕ) (h : a^2 + b^2 + c^2 = (a - b)^2 + (b - c)^2 + (c - a)^2) :
  ∃ m n p q : ℕ, ab = m^2 ∧ bc = n^2 ∧ ca = p^2 ∧ ab + bc + ca = q^2 :=
by sorry

end perfect_squares_of_nat_l284_284958


namespace cos_seven_pi_six_eq_neg_sqrt_three_div_two_l284_284907

noncomputable def cos_seven_pi_six : Real :=
  Real.cos (7 * Real.pi / 6)

theorem cos_seven_pi_six_eq_neg_sqrt_three_div_two :
  cos_seven_pi_six = -Real.sqrt 3 / 2 :=
sorry

end cos_seven_pi_six_eq_neg_sqrt_three_div_two_l284_284907


namespace tan_105_eq_neg_2_sub_sqrt_3_l284_284622

-- Definitions
def angle105 : ℝ := 105 * (Math.pi / 180)
def angle45 : ℝ := 45 * (Math.pi / 180)
def angle60 : ℝ := 60 * (Math.pi / 180)

-- Theorem
theorem tan_105_eq_neg_2_sub_sqrt_3 :
  Real.tan angle105 = -2 - Real.sqrt 3 := by
  sorry

end tan_105_eq_neg_2_sub_sqrt_3_l284_284622


namespace total_students_in_class_l284_284558

theorem total_students_in_class
    (students_in_front : ℕ)
    (students_in_back : ℕ)
    (lines : ℕ)
    (total_students_line : ℕ)
    (total_class : ℕ)
    (h_front: students_in_front = 2)
    (h_back: students_in_back = 5)
    (h_lines: lines = 3)
    (h_students_line : total_students_line = students_in_front + 1 + students_in_back)
    (h_total_class : total_class = lines * total_students_line) :
  total_class = 24 := by
  sorry

end total_students_in_class_l284_284558


namespace quadratic_roots_l284_284043

variable {a b c : ℝ}

theorem quadratic_roots (h₁ : a > 0) (h₂ : b > 0) (h₃ : c < 0) : 
  ∃ x₁ x₂ : ℝ, (a * x₁^2 + b * x₁ + c = 0) ∧ (a * x₂^2 + b * x₂ + c = 0) ∧ 
  (x₁ ≠ x₂) ∧ (x₁ > 0) ∧ (x₂ < 0) ∧ (|x₂| > |x₁|) := 
sorry

end quadratic_roots_l284_284043


namespace base_seven_to_base_ten_l284_284299

theorem base_seven_to_base_ten : 
  let n := 23456 
  ∈ ℕ, nat : ℕ 
  in nat = 6068 := 
by 
  sorry

end base_seven_to_base_ten_l284_284299


namespace find_multiple_l284_284576

theorem find_multiple :
  ∀ (total_questions correct_answers score : ℕ) (m : ℕ),
  total_questions = 100 →
  correct_answers = 90 →
  score = 70 →
  score = correct_answers - m * (total_questions - correct_answers) →
  m = 2 :=
by
  intros total_questions correct_answers score m h1 h2 h3 h4
  sorry

end find_multiple_l284_284576


namespace find_a_b_min_l284_284203

def f (x a b : ℝ) : ℝ := x^3 + a * x^2 + b * x + a^2

theorem find_a_b_min (a b : ℝ) :
  (∃ a b, f 1 a b = 10 ∧ deriv (f · a b) 1 = 0) →
  a = 4 ∧ b = -11 ∧ ∀ x ∈ Set.Icc (-4:ℝ) 3, f x a b ≥ f 1 4 (-11) := 
by
  -- Skipping the proof
  sorry

end find_a_b_min_l284_284203


namespace Jack_gave_Mike_six_notebooks_l284_284388

theorem Jack_gave_Mike_six_notebooks :
  ∀ (Gerald_notebooks Jack_notebooks_left notebooks_given_to_Paula total_notebooks_initial jack_notebooks_after_Paula notebooks_given_to_Mike : ℕ),
  Gerald_notebooks = 8 →
  Jack_notebooks_left = 10 →
  notebooks_given_to_Paula = 5 →
  total_notebooks_initial = Gerald_notebooks + 13 →
  jack_notebooks_after_Paula = total_notebooks_initial - notebooks_given_to_Paula →
  notebooks_given_to_Mike = jack_notebooks_after_Paula - Jack_notebooks_left →
  notebooks_given_to_Mike = 6 :=
by
  intros Gerald_notebooks Jack_notebooks_left notebooks_given_to_Paula total_notebooks_initial jack_notebooks_after_Paula notebooks_given_to_Mike
  intros Gerald_notebooks_eq Jack_notebooks_left_eq notebooks_given_to_Paula_eq total_notebooks_initial_eq jack_notebooks_after_Paula_eq notebooks_given_to_Mike_eq
  sorry

end Jack_gave_Mike_six_notebooks_l284_284388


namespace janet_more_cards_than_brenda_l284_284787

theorem janet_more_cards_than_brenda : ∀ (J B M : ℕ), M = 2 * J → J + B + M = 211 → M = 150 - 40 → J - B = 9 :=
by
  intros J B M h1 h2 h3
  sorry

end janet_more_cards_than_brenda_l284_284787


namespace intersection_equiv_l284_284733

def A : Set ℝ := { x : ℝ | x > 1 }
def B : Set ℝ := { x : ℝ | -1 < x ∧ x < 2 }
def C : Set ℝ := { x : ℝ | 1 < x ∧ x < 2 }

theorem intersection_equiv : A ∩ B = C :=
by
  sorry

end intersection_equiv_l284_284733


namespace find_couples_l284_284794

theorem find_couples (n p q : ℕ) (hn : 0 < n) (hp : 0 < p) (hq : 0 < q)
    (h_gcd : Nat.gcd p q = 1)
    (h_eq : p + q^2 = (n^2 + 1) * p^2 + q) : 
    (p = n + 1 ∧ q = n^2 + n + 1) :=
by 
  sorry

end find_couples_l284_284794


namespace number_of_students_only_taking_AMC8_l284_284017

def total_Germain := 13
def total_Newton := 10
def total_Young := 12

def olympiad_Germain := 3
def olympiad_Newton := 2
def olympiad_Young := 4

def number_only_AMC8 :=
  (total_Germain - olympiad_Germain) +
  (total_Newton - olympiad_Newton) +
  (total_Young - olympiad_Young)

theorem number_of_students_only_taking_AMC8 :
  number_only_AMC8 = 26 := by
  sorry

end number_of_students_only_taking_AMC8_l284_284017


namespace polygon_properties_l284_284723

def interior_angle_sum (n : ℕ) : ℝ :=
  (n - 2) * 180

def exterior_angle_sum : ℝ :=
  360

theorem polygon_properties (n : ℕ) (h : interior_angle_sum n = 3 * exterior_angle_sum + 180) :
  n = 9 ∧ interior_angle_sum n / n = 140 :=
by
  sorry

end polygon_properties_l284_284723


namespace num_two_digit_primes_with_ones_digit_three_is_seven_l284_284755

noncomputable def is_prime (n : ℕ) : Prop := sorry

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def has_ones_digit_three (n : ℕ) : Prop := n % 10 = 3

theorem num_two_digit_primes_with_ones_digit_three_is_seven :
  {n : ℕ | is_two_digit n ∧ has_ones_digit_three n ∧ is_prime n}.to_finset.card = 7 :=
by
  sorry

end num_two_digit_primes_with_ones_digit_three_is_seven_l284_284755


namespace smallest_gcd_qr_l284_284071

theorem smallest_gcd_qr {p q r : ℕ} (hpq : Int.gcd p q = 210) (hpr : Int.gcd p r = 770) :
  ∃ d, Int.gcd q r = d ∧ ∀ d', d' < d → ¬(Int.gcd q r = d') :=
sorry

end smallest_gcd_qr_l284_284071


namespace cos_seven_pi_over_six_l284_284912

theorem cos_seven_pi_over_six : Real.cos (7 * Real.pi / 6) = -Real.sqrt 3 / 2 :=
by
  -- Place the proof here
  sorry

end cos_seven_pi_over_six_l284_284912


namespace notebook_cost_l284_284382

open Nat

theorem notebook_cost
  (s : ℕ) (c : ℕ) (n : ℕ)
  (h_majority : s > 21)
  (h_notebooks : n > 2)
  (h_cost : c > n)
  (h_total : s * c * n = 2773) : c = 103 := 
sorry

end notebook_cost_l284_284382


namespace nonnegative_integer_pairs_solution_l284_284460

open Int

theorem nonnegative_integer_pairs_solution (x y : ℕ) : 
  3 * x ^ 2 + 2 * 9 ^ y = x * (4 ^ (y + 1) - 1) ↔ (x = 3 ∧ y = 1) ∨ (x = 2 ∧ y = 1) :=
by 
  sorry

end nonnegative_integer_pairs_solution_l284_284460


namespace base_7_to_10_of_23456_l284_284289

theorem base_7_to_10_of_23456 : 
  (2 * 7 ^ 4 + 3 * 7 ^ 3 + 4 * 7 ^ 2 + 5 * 7 ^ 1 + 6 * 7 ^ 0) = 6068 :=
by sorry

end base_7_to_10_of_23456_l284_284289


namespace remainder_sum_div_11_l284_284583

theorem remainder_sum_div_11 :
  ((100001 + 100002 + 100003 + 100004 + 100005 + 100006 + 100007 + 100008 + 100009 + 100010) % 11) = 10 :=
by
  sorry

end remainder_sum_div_11_l284_284583


namespace newspaper_spending_over_8_weeks_l284_284208

theorem newspaper_spending_over_8_weeks :
  (3 * 0.50 + 2.00) * 8 = 28 := by
  sorry

end newspaper_spending_over_8_weeks_l284_284208


namespace find_guest_sets_l284_284581

-- Definitions based on conditions
def cost_per_guest_set : ℝ := 32.0
def cost_per_master_set : ℝ := 40.0
def num_master_sets : ℕ := 4
def total_cost : ℝ := 224.0

-- The mathematical problem
theorem find_guest_sets (G : ℕ) (total_cost_eq : total_cost = cost_per_guest_set * G + cost_per_master_set * num_master_sets) : G = 2 :=
by
  sorry

end find_guest_sets_l284_284581


namespace geometric_sequence_expression_l284_284507

theorem geometric_sequence_expression (a : ℕ → ℝ) (q : ℝ) (n : ℕ) (h1 : a 2 = 1)
(h2 : a 3 * a 5 = 2 * a 7) : a n = 1 / 2 ^ (n - 2) :=
sorry

end geometric_sequence_expression_l284_284507


namespace problem_solution_l284_284194

variable (f : ℝ → ℝ)

-- Let f be an odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- f(x) = f(4 - x) for all x in ℝ
def satisfies_symmetry (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (4 - x)

-- f is increasing on [0, 2]
def is_increasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x < f y

theorem problem_solution :
  is_odd_function f →
  satisfies_symmetry f →
  is_increasing_on_interval f 0 2 →
  f 6 < f 4 ∧ f 4 < f 1 :=
by
  intros
  sorry

end problem_solution_l284_284194


namespace difference_of_one_third_and_five_l284_284113

theorem difference_of_one_third_and_five (n : ℕ) (h : n = 45) : (n / 3) - 5 = 10 :=
by
  sorry

end difference_of_one_third_and_five_l284_284113


namespace trajectory_of_midpoint_l284_284922

theorem trajectory_of_midpoint (Q : ℝ × ℝ) (P : ℝ × ℝ) (N : ℝ × ℝ)
  (h1 : Q.1^2 - Q.2^2 = 1)
  (h2 : N = (2 * P.1 - Q.1, 2 * P.2 - Q.2))
  (h3 : N.1 + N.2 = 2)
  (h4 : (P.2 - Q.2) / (P.1 - Q.1) = 1) :
  2 * P.1^2 - 2 * P.2^2 - 2 * P.1 + 2 * P.2 - 1 = 0 :=
  sorry

end trajectory_of_midpoint_l284_284922


namespace root_of_quadratic_eq_is_two_l284_284942

theorem root_of_quadratic_eq_is_two (k : ℝ) : (2^2 - 3 * 2 + k = 0) → k = 2 :=
by
  intro h
  sorry

end root_of_quadratic_eq_is_two_l284_284942


namespace cos_7pi_over_6_eq_neg_sqrt3_over_2_l284_284918

theorem cos_7pi_over_6_eq_neg_sqrt3_over_2 : Real.cos (7 * Real.pi / 6) = - (Real.sqrt 3) / 2 := by
  sorry

end cos_7pi_over_6_eq_neg_sqrt3_over_2_l284_284918


namespace greatest_integer_e_minus_5_l284_284961

theorem greatest_integer_e_minus_5 (e : ℝ) (h : 2 < e ∧ e < 3) : ⌊e - 5⌋ = -3 :=
by
  sorry

end greatest_integer_e_minus_5_l284_284961


namespace tan_105_eq_neg2_sub_sqrt3_l284_284646

theorem tan_105_eq_neg2_sub_sqrt3 :
  Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_eq_neg2_sub_sqrt3_l284_284646


namespace sequence_geometric_sequence_general_term_l284_284490

theorem sequence_geometric (a : ℕ → ℕ) (h₁ : a 1 = 2) (h₂ : ∀ n, a (n + 1) = 2 * a n + 1) :
  ∃ r : ℕ, (a 1 + 1) = 3 ∧ (∀ n, (a (n + 1) + 1) = r * (a n + 1)) := by
  sorry

theorem sequence_general_term (a : ℕ → ℕ) (h₁ : a 1 = 2) (h₂ : ∀ n, a (n + 1) = 2 * a n + 1) :
  ∀ n, a n = 3 * 2^(n-1) - 1 := by
  sorry

end sequence_geometric_sequence_general_term_l284_284490


namespace smallest_odd_number_with_five_different_prime_factors_l284_284842

noncomputable def smallest_odd_with_five_prime_factors : Nat :=
  3 * 5 * 7 * 11 * 13

theorem smallest_odd_number_with_five_different_prime_factors :
  smallest_odd_with_five_prime_factors = 15015 :=
by
  sorry -- Proof to be filled in later

end smallest_odd_number_with_five_different_prime_factors_l284_284842


namespace absolute_value_simplification_l284_284809

theorem absolute_value_simplification : abs(-4^2 + 6) = 10 := by
  sorry

end absolute_value_simplification_l284_284809


namespace two_digit_primes_ending_in_3_eq_6_l284_284767

open Nat

def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def ends_in_digit_3 (n : ℕ) : Prop :=
  n % 10 = 3

def count_two_digit_primes_ending_in_3 : ℕ :=
  ([13, 23, 33, 43, 53, 63, 73, 83, 93].filter (λ n, is_prime n ∧ is_two_digit n ∧ ends_in_digit_3 n)).length

theorem two_digit_primes_ending_in_3_eq_6 : count_two_digit_primes_ending_in_3 = 6 :=
by
  sorry

end two_digit_primes_ending_in_3_eq_6_l284_284767


namespace total_food_eaten_l284_284994

theorem total_food_eaten (num_puppies num_dogs : ℕ)
    (dog_food_per_meal dog_meals_per_day puppy_food_per_day : ℕ)
    (dog_food_mult puppy_meal_mult : ℕ)
    (h1 : num_puppies = 6)
    (h2 : num_dogs = 5)
    (h3 : dog_food_per_meal = 6)
    (h4 : dog_meals_per_day = 2)
    (h5 : dog_food_mult = 3)
    (h6 : puppy_meal_mult = 4)
    (h7 : puppy_food_per_day = (dog_food_per_meal / dog_food_mult) * puppy_meal_mult * dog_meals_per_day) :
    (num_dogs * dog_food_per_meal * dog_meals_per_day + num_puppies * puppy_food_per_day) = 108 := by
  -- conclude the theorem
  sorry

end total_food_eaten_l284_284994


namespace smallest_odd_with_five_prime_factors_is_15015_l284_284841

def is_odd (n : ℕ) : Prop := n % 2 = 1

def has_five_different_prime_factors (n : ℕ) : Prop :=
  ∃ (p1 p2 p3 p4 p5 : ℕ), nat.prime p1 ∧ nat.prime p2 ∧ nat.prime p3 ∧ nat.prime p4 ∧ nat.prime p5 ∧
  p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p1 ≠ p5 ∧
  p2 ≠ p3 ∧ p2 ≠ p4 ∧ p2 ≠ p5 ∧
  p3 ≠ p4 ∧ p3 ≠ p5 ∧
  p4 ≠ p5 ∧
  n = p1 * p2 * p3 * p4 * p5

def smallest_odd_number_with_five_different_prime_factors : ℕ :=
  15015

theorem smallest_odd_with_five_prime_factors_is_15015 :
  ∃ n, is_odd n ∧ has_five_different_prime_factors n ∧ n = 15015 :=
by exact ⟨15015, rfl, sorry⟩

end smallest_odd_with_five_prime_factors_is_15015_l284_284841


namespace inequality_solution_l284_284866

theorem inequality_solution {x : ℝ} : (1 / 2 - (x - 2) / 3 > 1) → (x < 1 / 2) :=
by {
  sorry
}

end inequality_solution_l284_284866


namespace operation_example_l284_284354

def operation (a b : ℤ) : ℤ := 2 * a * b - b^2

theorem operation_example : operation 1 (-3) = -15 := by
  sorry

end operation_example_l284_284354


namespace fuel_reduction_16km_temperature_drop_16km_l284_284883

-- Definition for fuel reduction condition
def fuel_reduction_rate (distance: ℕ) : ℕ := distance / 4 * 2

-- Definition for temperature drop condition
def temperature_drop_rate (distance: ℕ) : ℕ := distance / 8 * 1

-- Theorem to prove fuel reduction for 16 km
theorem fuel_reduction_16km : fuel_reduction_rate 16 = 8 := 
by
  -- proof will go here, but for now add sorry
  sorry

-- Theorem to prove temperature drop for 16 km
theorem temperature_drop_16km : temperature_drop_rate 16 = 2 := 
by
  -- proof will go here, but for now add sorry
  sorry

end fuel_reduction_16km_temperature_drop_16km_l284_284883


namespace Hilt_payment_l284_284250

def total_cost : ℝ := 2.05
def nickel_value : ℝ := 0.05
def dime_value : ℝ := 0.10

theorem Hilt_payment (n : ℕ) (h : n_n = n ∧ n_d = n) 
  (h_nickel : ℝ := n * nickel_value)
  (h_dime : ℝ := n * dime_value): 
  (n * nickel_value + n * dime_value = total_cost) 
  →  n = 14 :=
by {
  sorry
}

end Hilt_payment_l284_284250


namespace sunflower_packets_correct_l284_284974

namespace ShyneGarden

-- Define the given conditions
def eggplants_per_packet := 14
def sunflowers_per_packet := 10
def eggplant_packets_bought := 4
def total_plants := 116

-- Define the function to calculate the number of sunflower packets bought
def sunflower_packets_bought (eggplants_per_packet sunflowers_per_packet eggplant_packets_bought total_plants : ℕ) : ℕ :=
  (total_plants - (eggplant_packets_bought * eggplants_per_packet)) / sunflowers_per_packet

-- State the theorem to prove the number of sunflower packets
theorem sunflower_packets_correct :
  sunflower_packets_bought eggplants_per_packet sunflowers_per_packet eggplant_packets_bought total_plants = 6 :=
by
  sorry

end ShyneGarden

end sunflower_packets_correct_l284_284974


namespace range_of_k_l284_284816

noncomputable def f (x : ℝ) : ℝ := (x - 3) * Real.exp x

theorem range_of_k :
  (∀ x : ℝ, 2 < x → f x > k) →
  k ≤ -Real.exp 2 :=
by
  sorry

end range_of_k_l284_284816


namespace value_of_c_distinct_real_roots_l284_284182

-- Define the quadratic equation and the condition for having two distinct real roots
def quadratic_eqn (c : ℝ) : Prop :=
  ∀ (x : ℝ), x^2 + 2 * x + 4 * c = 0

def two_distinct_real_roots (c : ℝ) : Prop :=
  let Δ := 2^2 - 4 * 1 * (4 * c) in Δ > 0

-- The proof problem statement
theorem value_of_c_distinct_real_roots (c : ℝ) : c < 1 / 4 :=
by
  have h_discriminant : 4 - 16 * c > 0 :=
    calc
      4 - 16 * c = 4 - 16 * c : by ring
      ... > 0 : sorry
  have h_c_lt : c < 1 / 4 :=
    calc
      c < 1 / 4 : sorry
  exact h_c_lt

end value_of_c_distinct_real_roots_l284_284182


namespace hockey_league_num_games_l284_284308

theorem hockey_league_num_games :
  ∃ (num_teams : ℕ) (num_times : ℕ), 
    num_teams = 16 ∧ num_times = 10 ∧ 
    (num_teams * (num_teams - 1) / 2) * num_times = 2400 := by
  sorry

end hockey_league_num_games_l284_284308


namespace geometric_sequence_a3a5_l284_284383

theorem geometric_sequence_a3a5 :
  ∀ (a : ℕ → ℝ) (r : ℝ), (a 4 = 4) → (a 3 = a 0 * r ^ 3) → (a 5 = a 0 * r ^ 5) →
  a 3 * a 5 = 16 :=
by
  intros a r h1 h2 h3
  sorry

end geometric_sequence_a3a5_l284_284383


namespace positive_difference_of_complementary_ratio_5_1_l284_284276

-- Define angles satisfying the ratio condition and being complementary
def angle_pair (a b : ℝ) : Prop := (a + b = 90) ∧ (a = 5 * b ∨ b = 5 * a)

theorem positive_difference_of_complementary_ratio_5_1 :
  ∃ a b : ℝ, angle_pair a b ∧ abs (a - b) = 60 :=
by
  sorry

end positive_difference_of_complementary_ratio_5_1_l284_284276


namespace correct_average_marks_l284_284977

theorem correct_average_marks 
  (n : ℕ) (average initial_wrong new_correct : ℕ) 
  (h_num_students : n = 30)
  (h_average_marks : average = 100)
  (h_initial_wrong : initial_wrong = 70)
  (h_new_correct : new_correct = 10) :
  (average * n - (initial_wrong - new_correct)) / n = 98 := 
by
  sorry

end correct_average_marks_l284_284977


namespace q_at_4_l284_284350

def q (x : ℝ) : ℝ := |x - 3|^(1/3) + 3 * |x - 3|^(1/5) + 2 

theorem q_at_4 : q 4 = 6 := by
  sorry

end q_at_4_l284_284350


namespace intersection_M_N_l284_284964

def M : Set ℤ := {-1, 0, 1}
def N : Set ℤ := {x | x^2 ≠ x}

theorem intersection_M_N:
  M ∩ N = {-1} := by
  sorry

end intersection_M_N_l284_284964


namespace M1M2_product_l284_284092

theorem M1M2_product :
  ∀ (M1 M2 : ℝ),
  (∀ x : ℝ, x^2 - 5 * x + 6 ≠ 0 →
    (45 * x - 55) / (x^2 - 5 * x + 6) = M1 / (x - 2) + M2 / (x - 3)) →
  (M1 + M2 = 45) →
  (3 * M1 + 2 * M2 = 55) →
  M1 * M2 = 200 :=
by
  sorry

end M1M2_product_l284_284092


namespace tan_105_eq_neg2_sub_sqrt3_l284_284670

theorem tan_105_eq_neg2_sub_sqrt3 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 := by
  sorry

end tan_105_eq_neg2_sub_sqrt3_l284_284670


namespace sum_of_first_50_digits_of_one_over_1234_l284_284859

def first_n_digits_sum (x : ℚ) (n : ℕ) : ℕ :=
  sorry  -- This function should compute the sum of the first n digits after the decimal point of x

theorem sum_of_first_50_digits_of_one_over_1234 :
  first_n_digits_sum (1/1234) 50 = 275 :=
sorry

end sum_of_first_50_digits_of_one_over_1234_l284_284859


namespace widgets_per_shipping_box_l284_284704

theorem widgets_per_shipping_box :
  let widget_per_carton := 3
  let carton_width := 4
  let carton_length := 4
  let carton_height := 5
  let shipping_box_width := 20
  let shipping_box_length := 20
  let shipping_box_height := 20
  let carton_volume := carton_width * carton_length * carton_height
  let shipping_box_volume := shipping_box_width * shipping_box_length * shipping_box_height
  let cartons_per_shipping_box := shipping_box_volume / carton_volume
  cartons_per_shipping_box * widget_per_carton = 300 :=
by
  sorry

end widgets_per_shipping_box_l284_284704


namespace tan_105_eq_neg2_sub_sqrt3_l284_284645

theorem tan_105_eq_neg2_sub_sqrt3 :
  Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_eq_neg2_sub_sqrt3_l284_284645


namespace range_of_a_l284_284937

theorem range_of_a (a : ℝ) : (¬ ∀ x : ℝ, x^2 + a * x + 1 ≥ 0) ↔ a ∈ Set.Iio (-2) ∪ Set.Ioi 2 :=
by
  sorry

end range_of_a_l284_284937


namespace problem_mod_l284_284163

theorem problem_mod (a b c d : ℕ) (h1 : a = 2011) (h2 : b = 2012) (h3 : c = 2013) (h4 : d = 2014) :
  (a * b * c * d) % 5 = 4 :=
by
  sorry

end problem_mod_l284_284163


namespace leesburg_population_l284_284411

theorem leesburg_population (salem_population leesburg_population half_salem_population number_moved_out : ℕ)
  (h1 : half_salem_population * 2 = salem_population)
  (h2 : salem_population - number_moved_out = 754100)
  (h3 : salem_population = 15 * leesburg_population)
  (h4 : half_salem_population = 377050)
  (h5 : number_moved_out = 130000) :
  leesburg_population = 58940 :=
by
  sorry

end leesburg_population_l284_284411


namespace inequality_solution_set_l284_284280

theorem inequality_solution_set :
  {x : ℝ | (1/2 - x) * (x - 1/3) > 0} = {x : ℝ | 1/3 < x ∧ x < 1/2} := 
sorry

end inequality_solution_set_l284_284280


namespace equation_of_line_l284_284479

theorem equation_of_line (P : ℝ × ℝ) (A : ℝ) (m : ℝ) (hP : P = (-3, 4)) (hA : A = 3) (hm : m = 1) :
  ((2 * P.1 + 3 * P.2 - 6 = 0) ∨ (8 * P.1 + 3 * P.2 + 12 = 0)) :=
by 
  sorry

end equation_of_line_l284_284479


namespace tan_105_l284_284687

-- Defining the necessary known values and functions
def tan_addition (a b : ℝ) : ℝ := (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b)
def tan_60 : ℝ := Real.sqrt 3
def tan_45 : ℝ := 1

-- Proof goal in Lean 4
theorem tan_105 : Real.tan (105 * Real.pi / 180) = 2 + Real.sqrt 3 :=
by
  let tan_105 := tan_addition (60 * Real.pi / 180) (45 * Real.pi / 180)
  have h1 : Real.tan (60 * Real.pi / 180) = tan_60 := by sorry
  have h2 : Real.tan (45 * Real.pi / 180) = tan_45 := by sorry
  show Real.tan (105 * Real.pi / 180) = 2 + Real.sqrt 3 from sorry

end tan_105_l284_284687


namespace count_blanks_l284_284947

theorem count_blanks (B : ℝ) (h1 : 10 + B = T) (h2 : 0.7142857142857143 = B / T) : B = 25 :=
by
  -- The conditions are taken into account as definitions or parameters
  -- We skip the proof itself by using 'sorry'
  sorry

end count_blanks_l284_284947


namespace seats_per_bus_correct_l284_284988

-- Define the conditions given in the problem
def students : ℕ := 28
def buses : ℕ := 4

-- Define the number of seats per bus
def seats_per_bus : ℕ := students / buses

-- State the theorem that proves the number of seats per bus
theorem seats_per_bus_correct : seats_per_bus = 7 := by
  -- conditions are used as definitions, the goal is to prove seats_per_bus == 7
  sorry

end seats_per_bus_correct_l284_284988


namespace tan_105_degree_l284_284592

theorem tan_105_degree : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  have tan_add : ∀ (a b : ℝ), Real.tan (a + b) = (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b) :=
    sorry
  have tan_45 := Real.tan (45 * Real.pi / 180)
  have tan_60 := Real.tan (60 * Real.pi / 180)
  have tan_45_value : tan_45 = 1 := sorry
  have tan_60_value : tan_60 = Real.sqrt 3 := sorry
  sorry

end tan_105_degree_l284_284592


namespace infinite_sum_of_zeta_fractional_parts_l284_284474

open Real

noncomputable def fractional_part (x : ℝ) : ℝ := x - floor x

theorem infinite_sum_of_zeta_fractional_parts :
  (∑' k : ℕ, fractional_part (ζ (2 * (k + 1)))) = 1 / 2 :=
by
  sorry

end infinite_sum_of_zeta_fractional_parts_l284_284474


namespace tom_spent_correct_amount_l284_284553

-- Define the prices of the games
def batman_game_price : ℝ := 13.6
def superman_game_price : ℝ := 5.06

-- Define the total amount spent calculation
def total_spent := batman_game_price + superman_game_price

-- The main statement to prove
theorem tom_spent_correct_amount : total_spent = 18.66 := by
  -- Proof (intended)
  sorry

end tom_spent_correct_amount_l284_284553


namespace tan_105_l284_284593

theorem tan_105 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  -- Definitions
  let tan45 := Real.tan (45 * Real.pi / 180)
  let tan60 := Real.tan (60 * Real.pi / 180)
  have h1 : tan45 = 1 := sorry
  have h2 : tan60 = Real.sqrt 3 := sorry
  have h3 : tan45 + tan60 = 1 + Real.sqrt 3 := sorry
  have h4 : 1 - tan45 * tan60 = 1 - 1 * Real.sqrt 3 := sorry
  
  -- Use tangent addition formula
  have tan_addition : Real.tan (105 * Real.pi / 180) = (1 + Real.sqrt 3) / (1 - Real.sqrt 3) := sorry
  
  -- Simplify and rationalize to prove the final result
  have tan_105_eq : (1 + Real.sqrt 3) / (1 - Real.sqrt 3) = -2 - Real.sqrt 3 := sorry
  
  exact tan_105_eq

end tan_105_l284_284593


namespace tan_105_l284_284599

theorem tan_105 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  -- Definitions
  let tan45 := Real.tan (45 * Real.pi / 180)
  let tan60 := Real.tan (60 * Real.pi / 180)
  have h1 : tan45 = 1 := sorry
  have h2 : tan60 = Real.sqrt 3 := sorry
  have h3 : tan45 + tan60 = 1 + Real.sqrt 3 := sorry
  have h4 : 1 - tan45 * tan60 = 1 - 1 * Real.sqrt 3 := sorry
  
  -- Use tangent addition formula
  have tan_addition : Real.tan (105 * Real.pi / 180) = (1 + Real.sqrt 3) / (1 - Real.sqrt 3) := sorry
  
  -- Simplify and rationalize to prove the final result
  have tan_105_eq : (1 + Real.sqrt 3) / (1 - Real.sqrt 3) = -2 - Real.sqrt 3 := sorry
  
  exact tan_105_eq

end tan_105_l284_284599


namespace suff_but_not_nec_l284_284220

theorem suff_but_not_nec (a b : ℝ) (h : a > b ∧ b > 0) : a^2 > b^2 :=
by {
  sorry
}

end suff_but_not_nec_l284_284220


namespace tan_105_l284_284597

theorem tan_105 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  -- Definitions
  let tan45 := Real.tan (45 * Real.pi / 180)
  let tan60 := Real.tan (60 * Real.pi / 180)
  have h1 : tan45 = 1 := sorry
  have h2 : tan60 = Real.sqrt 3 := sorry
  have h3 : tan45 + tan60 = 1 + Real.sqrt 3 := sorry
  have h4 : 1 - tan45 * tan60 = 1 - 1 * Real.sqrt 3 := sorry
  
  -- Use tangent addition formula
  have tan_addition : Real.tan (105 * Real.pi / 180) = (1 + Real.sqrt 3) / (1 - Real.sqrt 3) := sorry
  
  -- Simplify and rationalize to prove the final result
  have tan_105_eq : (1 + Real.sqrt 3) / (1 - Real.sqrt 3) = -2 - Real.sqrt 3 := sorry
  
  exact tan_105_eq

end tan_105_l284_284597


namespace complete_the_square_l284_284862

theorem complete_the_square (x : ℝ) : x^2 - 8 * x + 1 = 0 → (x - 4)^2 = 15 :=
by
  intro h
  sorry

end complete_the_square_l284_284862


namespace students_in_all_three_workshops_l284_284828

-- Define the students counts and other conditions
def num_students : ℕ := 25
def num_dance : ℕ := 12
def num_chess : ℕ := 15
def num_robotics : ℕ := 11
def num_at_least_two : ℕ := 12

-- Define the proof statement
theorem students_in_all_three_workshops : 
  ∃ c : ℕ, c = 1 ∧ 
    (∃ a b d : ℕ, 
      a + b + c + d = num_at_least_two ∧
      num_students ≥ num_dance + num_chess + num_robotics - a - b - d - 2 * c
    ) := 
by
  sorry

end students_in_all_three_workshops_l284_284828


namespace thickness_of_stack_l284_284234

theorem thickness_of_stack (books : ℕ) (avg_pages_per_book : ℕ) (pages_per_inch : ℕ) (total_pages : ℕ) (thick_in_inches : ℕ)
    (h1 : books = 6)
    (h2 : avg_pages_per_book = 160)
    (h3 : pages_per_inch = 80)
    (h4 : total_pages = books * avg_pages_per_book)
    (h5 : thick_in_inches = total_pages / pages_per_inch) :
    thick_in_inches = 12 :=
by {
    -- statement without proof
    sorry
}

end thickness_of_stack_l284_284234


namespace interval_of_increase_l284_284936

-- Given conditions
def f (x : ℝ) : ℝ := (1/2) - cos(2 * x) * cos(2 * x)
def g (x : ℝ) : ℝ := 2 * sin(2 * x - (Real.pi / 8)) + 1

theorem interval_of_increase :
  let m : ℝ := Real.pi / 8 in
  is_monotonic_increasing_on (g) (Set.Icc π (5 * π / 4)) :=
sorry

end interval_of_increase_l284_284936


namespace simplify_and_evaluate_l284_284415

theorem simplify_and_evaluate :
  let x := (-1 : ℚ) / 2
  3 * x^2 - (5 * x - 3 * (2 * x - 1) + 7 * x^2) = -9 / 2 :=
by
  let x : ℚ := (-1 : ℚ) / 2
  sorry

end simplify_and_evaluate_l284_284415


namespace hypotenuse_length_l284_284882

noncomputable def hypotenuse_of_30_60_90_triangle (r : ℝ) : ℝ :=
  let a := (r * 3) / Real.sqrt 3
  2 * a

theorem hypotenuse_length (r : ℝ) (h : r = 3) : hypotenuse_of_30_60_90_triangle r = 6 * Real.sqrt 3 :=
  by sorry

end hypotenuse_length_l284_284882


namespace range_of_m_l284_284483

-- Definitions and conditions
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

def is_eccentricity (e a b : ℝ) : Prop :=
  e = Real.sqrt (1 - (b^2 / a^2))

def is_semi_latus_rectum (d a b : ℝ) : Prop :=
  d = 2 * b^2 / a

-- Main theorem statement
theorem range_of_m (a b m : ℝ) (x y : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0)
  (h3 : is_eccentricity (Real.sqrt (3) / 2) a b)
  (h4 : is_semi_latus_rectum 1 a b)
  (h_ellipse : ellipse a b x y) : 
  m ∈ Set.Ioo (-3 / 2 : ℝ) (3 / 2 : ℝ) := 
sorry

end range_of_m_l284_284483


namespace find_x_when_y_is_minus_21_l284_284543

variable (x y k : ℝ)

theorem find_x_when_y_is_minus_21
  (h1 : x * y = k)
  (h2 : x + y = 35)
  (h3 : y = 3 * x)
  (h4 : y = -21) :
  x = -10.9375 := by
  sorry

end find_x_when_y_is_minus_21_l284_284543


namespace min_value_reciprocals_l284_284960

open Real

theorem min_value_reciprocals (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : a + 3 * b = 1) :
  ∃ m : ℝ, (∀ x y : ℝ, 0 < x → 0 < y → x + 3 * y = 1 → (1 / x + 1 / y) ≥ m) ∧ m = 4 + 2 * sqrt 3 :=
sorry

end min_value_reciprocals_l284_284960


namespace johns_horses_l284_284089

theorem johns_horses 
  (feeding_per_day : ℕ := 2) 
  (food_per_feeding : ℝ := 20) 
  (bag_weight : ℝ := 1000) 
  (num_bags : ℕ := 60) 
  (days : ℕ := 60)
  (total_food : ℝ := num_bags * bag_weight) 
  (daily_food_consumption : ℝ := total_food / days) 
  (food_per_horse_per_day : ℝ := food_per_feeding * feeding_per_day) :
  ∀ H : ℝ, (daily_food_consumption / food_per_horse_per_day = H) → H = 25 := 
by
  intros H hH
  sorry

end johns_horses_l284_284089


namespace tan_alpha_eq_3_l284_284482

theorem tan_alpha_eq_3 (α : ℝ) (h1 : 0 < α ∧ α < (π / 2))
  (h2 : (Real.sin α)^2 + Real.cos ((π / 2) + 2 * α) = 3 / 10) : Real.tan α = 3 := by
  sorry

end tan_alpha_eq_3_l284_284482


namespace least_number_of_groups_l284_284012

theorem least_number_of_groups (total_players : ℕ) (max_per_group : ℕ) (h1 : total_players = 30) (h2 : max_per_group = 12) : ∃ (groups : ℕ), groups = 3 := 
by {
  -- Mathematical conditions and solution to be formalized here
  sorry
}

end least_number_of_groups_l284_284012


namespace evaluate_f_at_1_l284_284484

noncomputable def f (x : ℝ) : ℝ := 2^x + 2

theorem evaluate_f_at_1 : f 1 = 4 :=
by {
  -- proof goes here
  sorry
}

end evaluate_f_at_1_l284_284484


namespace tan_105_eq_minus_2_minus_sqrt_3_l284_284676

theorem tan_105_eq_minus_2_minus_sqrt_3 :
  Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_eq_minus_2_minus_sqrt_3_l284_284676


namespace smallest_odd_with_five_prime_factors_is_15015_l284_284840

def is_odd (n : ℕ) : Prop := n % 2 = 1

def has_five_different_prime_factors (n : ℕ) : Prop :=
  ∃ (p1 p2 p3 p4 p5 : ℕ), nat.prime p1 ∧ nat.prime p2 ∧ nat.prime p3 ∧ nat.prime p4 ∧ nat.prime p5 ∧
  p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p1 ≠ p5 ∧
  p2 ≠ p3 ∧ p2 ≠ p4 ∧ p2 ≠ p5 ∧
  p3 ≠ p4 ∧ p3 ≠ p5 ∧
  p4 ≠ p5 ∧
  n = p1 * p2 * p3 * p4 * p5

def smallest_odd_number_with_five_different_prime_factors : ℕ :=
  15015

theorem smallest_odd_with_five_prime_factors_is_15015 :
  ∃ n, is_odd n ∧ has_five_different_prime_factors n ∧ n = 15015 :=
by exact ⟨15015, rfl, sorry⟩

end smallest_odd_with_five_prime_factors_is_15015_l284_284840


namespace tan_105_eq_neg2_sub_sqrt3_l284_284672

theorem tan_105_eq_neg2_sub_sqrt3 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 := by
  sorry

end tan_105_eq_neg2_sub_sqrt3_l284_284672


namespace smallest_number_of_three_integers_l284_284551

theorem smallest_number_of_three_integers 
  (a b c : ℕ) 
  (hpos1 : 0 < a) (hpos2 : 0 < b) (hpos3 : 0 < c) 
  (hmean : (a + b + c) / 3 = 24)
  (hmed : b = 23)
  (hlargest : b + 4 = c) 
  : a = 22 :=
by
  sorry

end smallest_number_of_three_integers_l284_284551


namespace opposite_points_number_line_l284_284571

theorem opposite_points_number_line (a : ℤ) (h : a - 6 = -a) : a = 3 := by
  sorry

end opposite_points_number_line_l284_284571


namespace division_of_negatives_l284_284891

theorem division_of_negatives :
  (-81 : ℤ) / (-9) = 9 := 
  by
  -- Property of division with negative numbers
  have h1 : (-81 : ℤ) / (-9) = 81 / 9 := by sorry
  -- Perform the division
  have h2 : 81 / 9 = 9 := by sorry
  -- Combine the results
  rw h1
  exact h2

end division_of_negatives_l284_284891


namespace negation_of_proposition_l284_284489

-- Definitions from the problem conditions
def proposition (x : ℝ) := ∃ x < 1, x^2 ≤ 1

-- Reformulated proof problem
theorem negation_of_proposition : 
  ¬ (∃ x < 1, x^2 ≤ 1) ↔ ∀ x < 1, x^2 > 1 :=
by
  sorry

end negation_of_proposition_l284_284489


namespace tan_105_eq_neg_2_sub_sqrt_3_l284_284624

-- Definitions
def angle105 : ℝ := 105 * (Math.pi / 180)
def angle45 : ℝ := 45 * (Math.pi / 180)
def angle60 : ℝ := 60 * (Math.pi / 180)

-- Theorem
theorem tan_105_eq_neg_2_sub_sqrt_3 :
  Real.tan angle105 = -2 - Real.sqrt 3 := by
  sorry

end tan_105_eq_neg_2_sub_sqrt_3_l284_284624


namespace fourth_person_height_l284_284429

theorem fourth_person_height 
  (h : ℝ)
  (height_average : (h + (h + 2) + (h + 4) + (h + 10)) / 4 = 79)
  : h + 10 = 85 := 
by
  sorry

end fourth_person_height_l284_284429


namespace tims_total_earnings_l284_284286

theorem tims_total_earnings (days_of_week : ℕ) (tasks_per_day : ℕ) (tasks_40_rate : ℕ) (tasks_30_rate1 : ℕ) (tasks_30_rate2 : ℕ)
    (rate_40 : ℝ) (rate_30_1 : ℝ) (rate_30_2 : ℝ) (bonus_per_50 : ℝ) (performance_bonus : ℝ)
    (total_earnings : ℝ) :
  days_of_week = 6 →
  tasks_per_day = 100 →
  tasks_40_rate = 40 →
  tasks_30_rate1 = 30 →
  tasks_30_rate2 = 30 →
  rate_40 = 1.2 →
  rate_30_1 = 1.5 →
  rate_30_2 = 2.0 →
  bonus_per_50 = 10 →
  performance_bonus = 20 →
  total_earnings = 1058 :=
by
  intros
  sorry

end tims_total_earnings_l284_284286


namespace probability_of_25_cents_heads_l284_284534

/-- 
Considering the flipping of five specific coins: a penny, a nickel, a dime,
a quarter, and a half dollar, prove that the probability of getting at least
25 cents worth of heads is 3 / 4.
-/
theorem probability_of_25_cents_heads :
  let total_outcomes := 2^5
  let successful_outcomes_1 := 2^4
  let successful_outcomes_2 := 2^3
  let successful_outcomes := successful_outcomes_1 + successful_outcomes_2
  (successful_outcomes / total_outcomes : ℚ) = 3 / 4 :=
by
  sorry

end probability_of_25_cents_heads_l284_284534


namespace tan_105_l284_284682

-- Defining the necessary known values and functions
def tan_addition (a b : ℝ) : ℝ := (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b)
def tan_60 : ℝ := Real.sqrt 3
def tan_45 : ℝ := 1

-- Proof goal in Lean 4
theorem tan_105 : Real.tan (105 * Real.pi / 180) = 2 + Real.sqrt 3 :=
by
  let tan_105 := tan_addition (60 * Real.pi / 180) (45 * Real.pi / 180)
  have h1 : Real.tan (60 * Real.pi / 180) = tan_60 := by sorry
  have h2 : Real.tan (45 * Real.pi / 180) = tan_45 := by sorry
  show Real.tan (105 * Real.pi / 180) = 2 + Real.sqrt 3 from sorry

end tan_105_l284_284682


namespace regular_polygon_sides_and_interior_angle_l284_284719

theorem regular_polygon_sides_and_interior_angle (n : ℕ) (H : (n - 2) * 180 = 3 * 360 + 180) :
  n = 9 ∧ (n - 2) * 180 / n = 140 :=
by
-- This marks the start of the proof, but the proof is omitted.
sorry

end regular_polygon_sides_and_interior_angle_l284_284719


namespace lines_condition_l284_284157

-- Assume x and y are real numbers representing coordinates on the lines l1 and l2
variables (x y : ℝ)

-- Points on the lines l1 and l2 satisfy the condition |x| - |y| = 0.
theorem lines_condition (x y : ℝ) (h : abs x = abs y) : abs x - abs y = 0 :=
by
  sorry

end lines_condition_l284_284157


namespace sum_of_first_50_digits_of_one_over_1234_l284_284858

def first_n_digits_sum (x : ℚ) (n : ℕ) : ℕ :=
  sorry  -- This function should compute the sum of the first n digits after the decimal point of x

theorem sum_of_first_50_digits_of_one_over_1234 :
  first_n_digits_sum (1/1234) 50 = 275 :=
sorry

end sum_of_first_50_digits_of_one_over_1234_l284_284858


namespace scheme_A_yield_percentage_l284_284452

-- Define the initial investments and yields
def initial_investment_A : ℝ := 300
def initial_investment_B : ℝ := 200
def yield_B : ℝ := 0.5 -- 50% yield

-- Define the equation given in the problem
def yield_A_equation (P : ℝ) : Prop :=
  initial_investment_A + (initial_investment_A * (P / 100)) = initial_investment_B + (initial_investment_B * yield_B) + 90

-- The proof statement we need to prove
theorem scheme_A_yield_percentage : yield_A_equation 30 :=
by
  sorry -- Proof is omitted

end scheme_A_yield_percentage_l284_284452


namespace find_a_value_l284_284927

theorem find_a_value (a : ℝ) :
  {a^2, a + 1, -3} ∩ {a - 3, 2 * a - 1, a^2 + 1} = {-3} → a = -1 :=
by
  intro h
  sorry

end find_a_value_l284_284927


namespace Q_mod_m_l284_284172

open Nat

def Q (m : ℕ) : ℕ :=
  (List.range m).filter (fun x => gcd x m = 1)
  |> List.prod

theorem Q_mod_m {m : ℕ} (h: m ≥ 3) : Q m ≡ 1 [MOD m] ∨ Q m ≡ -1 [MOD m] :=
by sorry

end Q_mod_m_l284_284172


namespace negation_of_proposition_l284_284865

theorem negation_of_proposition :
  (∀ x : ℝ, x^2 + 1 ≥ 0) ↔ (¬ ∃ x : ℝ, x^2 + 1 < 0) :=
by
  sorry

end negation_of_proposition_l284_284865


namespace prove_a3_l284_284387

variable (a1 a2 a3 a4 : ℕ)
variable (q : ℕ)

-- Definition of the geometric sequence
def geom_seq (n : ℕ) : ℕ :=
  a1 * q^(n-1)

-- Given conditions
def cond1 := geom_seq 4 = 8
def cond2 := (geom_seq 2 + geom_seq 3) / (geom_seq 1 + geom_seq 2) = 2

-- Proving the required condition
theorem prove_a3 : cond1 ∧ cond2 → geom_seq 3 = 4 :=
by
sorry

end prove_a3_l284_284387


namespace tan_105_degree_is_neg_sqrt3_minus_2_l284_284652

theorem tan_105_degree_is_neg_sqrt3_minus_2 :
  Real.tan (105 * Real.pi / 180) = -(Real.sqrt 3 + 2) := by
  sorry

end tan_105_degree_is_neg_sqrt3_minus_2_l284_284652


namespace boat_stream_speed_l284_284436

theorem boat_stream_speed :
  ∀ (v : ℝ), (∀ (downstream_speed boat_speed : ℝ), boat_speed = 22 ∧ downstream_speed = 54/2 ∧ downstream_speed = boat_speed + v) -> v = 5 :=
by
  sorry

end boat_stream_speed_l284_284436


namespace trigonometric_product_identity_l284_284347

theorem trigonometric_product_identity : 
  let cos_40 : Real := Real.cos (Real.pi * 40 / 180)
  let sin_40 : Real := Real.sin (Real.pi * 40 / 180)
  let cos_50 : Real := Real.cos (Real.pi * 50 / 180)
  let sin_50 : Real := Real.sin (Real.pi * 50 / 180)
  (sin_50 = cos_40) → (cos_50 = sin_40) →
  (1 - cos_40⁻¹) * (1 + sin_50⁻¹) * (1 - sin_40⁻¹) * (1 + cos_50⁻¹) = 1 := by
  sorry

end trigonometric_product_identity_l284_284347


namespace widgets_per_shipping_box_l284_284702

theorem widgets_per_shipping_box 
  (widgets_per_carton : ℕ := 3)
  (carton_width : ℕ := 4)
  (carton_length : ℕ := 4)
  (carton_height : ℕ := 5)
  (box_width : ℕ := 20)
  (box_length : ℕ := 20)
  (box_height : ℕ := 20) :
  (widgets_per_carton * ((box_width * box_length * box_height) / (carton_width * carton_length * carton_height))) = 300 :=
by
  sorry

end widgets_per_shipping_box_l284_284702


namespace widgets_per_shipping_box_l284_284701

theorem widgets_per_shipping_box 
  (widgets_per_carton : ℕ := 3)
  (carton_width : ℕ := 4)
  (carton_length : ℕ := 4)
  (carton_height : ℕ := 5)
  (box_width : ℕ := 20)
  (box_length : ℕ := 20)
  (box_height : ℕ := 20) :
  (widgets_per_carton * ((box_width * box_length * box_height) / (carton_width * carton_length * carton_height))) = 300 :=
by
  sorry

end widgets_per_shipping_box_l284_284701


namespace Kyle_age_l284_284257

-- Let's define the variables for each person's age.
variables (Shelley Kyle Julian Frederick Tyson Casey Sandra David Fiona : ℕ) 

-- Defining conditions based on given problem.
axiom condition1 : Shelley = Kyle - 3
axiom condition2 : Shelley = Julian + 4
axiom condition3 : Julian = Frederick - 20
axiom condition4 : Julian = Fiona + 5
axiom condition5 : Frederick = 2 * Tyson
axiom condition6 : Tyson = 2 * Casey
axiom condition7 : Casey = Fiona - 2
axiom condition8 : Casey = Sandra / 2
axiom condition9 : Sandra = David + 4
axiom condition10 : David = 16

-- The goal is to prove Kyle's age is 23 years old.
theorem Kyle_age : Kyle = 23 :=
by sorry

end Kyle_age_l284_284257


namespace tan_105_degree_l284_284610

theorem tan_105_degree : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 := by
  have h_add : ∀ a b : ℝ, Real.tan (a + b) = (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b) :=
    Real.tan_add

  have h_tan_60 : Real.tan (60 * Real.pi / 180) = Real.sqrt 3 := Real.tan_60
  have h_tan_45 : Real.tan (45 * Real.pi / 180) = 1 := Real.tan_45
  
  sorry

end tan_105_degree_l284_284610


namespace num_two_digit_primes_with_ones_digit_three_is_seven_l284_284756

noncomputable def is_prime (n : ℕ) : Prop := sorry

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def has_ones_digit_three (n : ℕ) : Prop := n % 10 = 3

theorem num_two_digit_primes_with_ones_digit_three_is_seven :
  {n : ℕ | is_two_digit n ∧ has_ones_digit_three n ∧ is_prime n}.to_finset.card = 7 :=
by
  sorry

end num_two_digit_primes_with_ones_digit_three_is_seven_l284_284756


namespace unique_largest_negative_integer_l284_284283

theorem unique_largest_negative_integer :
  ∃! x : ℤ, x = -1 ∧ (∀ y : ℤ, y < 0 → x ≥ y) :=
by
  sorry

end unique_largest_negative_integer_l284_284283


namespace calcium_iodide_weight_l284_284301

theorem calcium_iodide_weight
  (atomic_weight_Ca : ℝ)
  (atomic_weight_I : ℝ)
  (moles : ℝ) :
  atomic_weight_Ca = 40.08 →
  atomic_weight_I = 126.90 →
  moles = 5 →
  (atomic_weight_Ca + 2 * atomic_weight_I) * moles = 1469.4 :=
by
  intros
  sorry

end calcium_iodide_weight_l284_284301


namespace cos_seven_pi_six_eq_neg_sqrt_three_div_two_l284_284906

noncomputable def cos_seven_pi_six : Real :=
  Real.cos (7 * Real.pi / 6)

theorem cos_seven_pi_six_eq_neg_sqrt_three_div_two :
  cos_seven_pi_six = -Real.sqrt 3 / 2 :=
sorry

end cos_seven_pi_six_eq_neg_sqrt_three_div_two_l284_284906


namespace calculation_result_l284_284458

theorem calculation_result :
  (10 * 19 * 20 * 53 * 100 + 601) / 13 = 1549277 :=
by 
  sorry

end calculation_result_l284_284458


namespace quadratic_has_distinct_real_roots_l284_284176

theorem quadratic_has_distinct_real_roots : ∀ c : ℝ, x^2 + 2 * x + 4 * c = 0 → 4 - 16 * c > 0 → c = 0 :=
begin
  intros c h_eq h_disc,
  sorry
end

end quadratic_has_distinct_real_roots_l284_284176


namespace cos_7pi_over_6_eq_neg_sqrt3_over_2_l284_284919

theorem cos_7pi_over_6_eq_neg_sqrt3_over_2 : Real.cos (7 * Real.pi / 6) = - (Real.sqrt 3) / 2 := by
  sorry

end cos_7pi_over_6_eq_neg_sqrt3_over_2_l284_284919


namespace simplify_abs_expr_l284_284813

theorem simplify_abs_expr : |(-4 ^ 2 + 6)| = 10 := by
  sorry

end simplify_abs_expr_l284_284813


namespace count_two_digit_primes_with_ones_digit_three_l284_284752

def is_prime (n : ℕ) : Prop := nat.prime n

def ones_digit_three (n : ℕ) : Prop := n % 10 = 3

def two_digit_number (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

theorem count_two_digit_primes_with_ones_digit_three : 
  {n : ℕ | two_digit_number n ∧ ones_digit_three n ∧ is_prime n}.to_finset.card = 6 :=
sorry

end count_two_digit_primes_with_ones_digit_three_l284_284752


namespace simplify_abs_expr_l284_284812

theorem simplify_abs_expr : |(-4 ^ 2 + 6)| = 10 := by
  sorry

end simplify_abs_expr_l284_284812


namespace smallest_odd_number_with_five_prime_factors_l284_284846

theorem smallest_odd_number_with_five_prime_factors : 
  ∃ n : ℕ, n = 15015 ∧ (∀ (p ∈ {3, 5, 7, 11, 13}), prime p) ∧ odd n :=
by
  sorry

end smallest_odd_number_with_five_prime_factors_l284_284846


namespace five_digit_number_divisible_by_B_is_multiple_of_1000_l284_284462

-- Definitions
def is_five_digit_number (A : ℕ) : Prop := 10000 ≤ A ∧ A < 100000
def B (A : ℕ) := (A / 1000 * 100) + (A % 100)
def is_four_digit_number (B : ℕ) : Prop := 1000 ≤ B ∧ B < 10000

-- Main theorem
theorem five_digit_number_divisible_by_B_is_multiple_of_1000
  (A : ℕ) (hA : is_five_digit_number A)
  (hAB : ∃ k : ℕ, B A = k) :
  A % 1000 = 0 := 
sorry

end five_digit_number_divisible_by_B_is_multiple_of_1000_l284_284462


namespace cos_7pi_over_6_eq_neg_sqrt3_over_2_l284_284917

theorem cos_7pi_over_6_eq_neg_sqrt3_over_2 : Real.cos (7 * Real.pi / 6) = - (Real.sqrt 3) / 2 := by
  sorry

end cos_7pi_over_6_eq_neg_sqrt3_over_2_l284_284917


namespace no_linear_term_in_expansion_l284_284074

theorem no_linear_term_in_expansion (a : ℤ) : 
  let p := (x^2 + a*x - 2) * (x - 1) in 
  ∀ (q : polynomial ℤ), q = p → 
  (q.coeff 1 = 0) →
  a = -2 :=
by
  intro a p q hq hcoeff1
  sorry

end no_linear_term_in_expansion_l284_284074


namespace tan_105_degree_l284_284586

theorem tan_105_degree : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  have tan_add : ∀ (a b : ℝ), Real.tan (a + b) = (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b) :=
    sorry
  have tan_45 := Real.tan (45 * Real.pi / 180)
  have tan_60 := Real.tan (60 * Real.pi / 180)
  have tan_45_value : tan_45 = 1 := sorry
  have tan_60_value : tan_60 = Real.sqrt 3 := sorry
  sorry

end tan_105_degree_l284_284586


namespace uncovered_area_l284_284320

theorem uncovered_area {s₁ s₂ : ℝ} (hs₁ : s₁ = 10) (hs₂ : s₂ = 4) : 
  (s₁^2 - 2 * s₂^2) = 68 := by
  sorry

end uncovered_area_l284_284320


namespace two_digit_primes_with_ones_digit_three_count_l284_284749

def is_two_digit (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100

def has_ones_digit_three (n : ℕ) : Prop :=
  n % 10 = 3

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def number_of_two_digit_primes_with_ones_digit_three : ℕ :=
  6

theorem two_digit_primes_with_ones_digit_three_count :
  number_of_two_digit_primes_with_ones_digit_three =
  (finset.filter (λ n, is_two_digit n ∧ has_ones_digit_three n ∧ is_prime n)
                 (finset.range 100)).card :=
by
  sorry

end two_digit_primes_with_ones_digit_three_count_l284_284749


namespace eq_radicals_same_type_l284_284119

theorem eq_radicals_same_type (a b : ℕ) (h1 : a - 1 = 2) (h2 : 3 * b - 1 = 7 - b) : a + b = 5 :=
by
  sorry

end eq_radicals_same_type_l284_284119


namespace rectangle_y_coordinate_l284_284078

theorem rectangle_y_coordinate (x1 x2 y1 A : ℝ) (h1 : x1 = -8) (h2 : x2 = 1) (h3 : y1 = 1) (h4 : A = 72)
    (hL : x2 - x1 = 9) (hA : A = 9 * (y - y1)) :
    (y = 9) :=
by
  sorry

end rectangle_y_coordinate_l284_284078


namespace smallest_odd_number_with_five_prime_factors_l284_284851

def is_prime_factor_of (p n : ℕ) : Prop :=
  p.prime ∧ p ∣ n

def is_odd (n : ℕ) : Prop :=
  ¬ 2 ∣ n

def has_five_distinct_prime_factors (n : ℕ) : Prop :=
  ∃ (p1 p2 p3 p4 p5 : ℕ), 
  p1.prime ∧ p2.prime ∧ p3.prime ∧ p4.prime ∧ p5.prime ∧ 
  p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p1 ≠ p5 ∧ 
  p2 ≠ p3 ∧ p2 ≠ p4 ∧ p2 ≠ p5 ∧ 
  p3 ≠ p4 ∧ p3 ≠ p5 ∧ 
  p4 ≠ p5 ∧ 
  p1 * p2 * p3 * p4 * p5 = n

theorem smallest_odd_number_with_five_prime_factors :
  is_odd 15015 ∧ has_five_distinct_prime_factors 15015 ∧ 
  (∀ n : ℕ, is_odd n ∧ has_five_distinct_prime_factors n → 15015 ≤ n) :=
by
  sorry

end smallest_odd_number_with_five_prime_factors_l284_284851


namespace find_three_digit_number_divisible_by_5_l284_284877

theorem find_three_digit_number_divisible_by_5 {n x : ℕ} (hx1 : 100 ≤ x) (hx2 : x < 1000) (hx3 : x % 5 = 0) (hx4 : x = n^3 + n^2) : x = 150 ∨ x = 810 := 
by
  sorry

end find_three_digit_number_divisible_by_5_l284_284877


namespace probability_boyA_or_girlB_selected_correct_l284_284046

noncomputable def probability_boyA_or_girlB_selected : ℚ :=
let total_ways := Nat.choose 4 2 * Nat.choose 6 2 in
let ways_neither_selected := Nat.choose 3 2 * Nat.choose 5 2 in
let ways_at_least_one_selected := total_ways - ways_neither_selected in
ways_at_least_one_selected / total_ways

theorem probability_boyA_or_girlB_selected_correct :
  probability_boyA_or_girlB_selected = 2 / 3 :=
by sorry

end probability_boyA_or_girlB_selected_correct_l284_284046


namespace sqrt_sum_eq_l284_284131

theorem sqrt_sum_eq : 
  (Real.sqrt (16 - 12 * Real.sqrt 3)) + (Real.sqrt (16 + 12 * Real.sqrt 3)) = 4 * Real.sqrt 6 :=
by
  sorry

end sqrt_sum_eq_l284_284131


namespace simplify_fraction_l284_284805

theorem simplify_fraction (a b : ℕ) (h : a = 180) (k : b = 270) : 
  ∃ c d, c = 2 ∧ d = 3 ∧ (a / (Nat.gcd a b) = c) ∧ (b / (Nat.gcd a b) = d) :=
by
  sorry

end simplify_fraction_l284_284805


namespace energy_soda_packs_l284_284870

-- Definitions and conditions
variables (total_bottles : ℕ) (regular_soda : ℕ) (diet_soda : ℕ) (pack_size : ℕ)
variables (complete_packs : ℕ) (remaining_regular : ℕ) (remaining_diet : ℕ) (remaining_energy : ℕ)

-- Conditions given in the problem
axiom h_total_bottles : total_bottles = 200
axiom h_regular_soda : regular_soda = 55
axiom h_diet_soda : diet_soda = 40
axiom h_pack_size : pack_size = 3

-- Proving the correct answer
theorem energy_soda_packs :
  complete_packs = (total_bottles - (regular_soda + diet_soda)) / pack_size ∧
  remaining_regular = regular_soda ∧
  remaining_diet = diet_soda ∧
  remaining_energy = (total_bottles - (regular_soda + diet_soda)) % pack_size :=
by
  sorry

end energy_soda_packs_l284_284870


namespace q1_q2_q3_l284_284729

noncomputable def quadratic_function (a x: ℝ) : ℝ := x^2 - 2 * a * x + a + 2

theorem q1 (a : ℝ) : (∀ {x : ℝ}, quadratic_function a x = 0 → x < 2) ∧ (quadratic_function a 2 > 0) ∧ (2 * a ≠ 0) → a < -1 := 
by 
  sorry

theorem q2 (a : ℝ) : (∀ x : ℝ, quadratic_function a x ≥ -1 - a * x) → -2 ≤ a ∧ a ≤ 6 := 
by 
  sorry
  
theorem q3 (a : ℝ) : (∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → quadratic_function a x ≤ 4) → a = 2 ∨ a = 2 / 3 := 
by 
  sorry

end q1_q2_q3_l284_284729


namespace find_a_l284_284098

def star (x y : ℤ × ℤ) : ℤ × ℤ := (x.1 - y.1, x.2 + y.2)

theorem find_a :
  ∃ (a b : ℤ), 
  star (5, 2) (1, 1) = (a, b) ∧
  star (a, b) (0, 1) = (2, 5) ∧
  a = 2 :=
sorry

end find_a_l284_284098


namespace exponential_simplification_l284_284302

theorem exponential_simplification : 
  (10^0.25) * (10^0.25) * (10^0.5) * (10^0.5) * (10^0.75) * (10^0.75) = 1000 := 
by 
  sorry

end exponential_simplification_l284_284302


namespace total_carrots_l284_284952

-- Definitions from conditions in a)
def JoanCarrots : ℕ := 29
def JessicaCarrots : ℕ := 11

-- Theorem that encapsulates the problem
theorem total_carrots : JoanCarrots + JessicaCarrots = 40 := by
  sorry

end total_carrots_l284_284952


namespace tan_add_tan_105_eq_l284_284664

noncomputable def tan : ℝ → ℝ := sorry -- Use the built-in library later for actual implementation

-- Given conditions
def tan_45_eq : tan 45 = 1 := by sorry
def tan_60_eq : tan 60 = Real.sqrt 3 := by sorry

-- Angle addition formula for tangent
theorem tan_add (a b : ℝ) :
  tan (a + b) = (tan a + tan b) / (1 - tan a * tan b) := by sorry

-- Main theorem to prove
theorem tan_105_eq :
  tan 105 = -2 - Real.sqrt 3 := by sorry

end tan_add_tan_105_eq_l284_284664

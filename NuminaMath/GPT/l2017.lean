import Mathlib

namespace lateral_surface_area_eq_total_surface_area_eq_l2017_201761

def r := 3
def h := 10

theorem lateral_surface_area_eq : 2 * Real.pi * r * h = 60 * Real.pi := by
  sorry

theorem total_surface_area_eq : 2 * Real.pi * r * h + 2 * Real.pi * r^2 = 78 * Real.pi := by
  sorry

end lateral_surface_area_eq_total_surface_area_eq_l2017_201761


namespace fair_share_of_bill_l2017_201767

noncomputable def total_bill : Real := 139.00
noncomputable def tip_percent : Real := 0.10
noncomputable def num_people : Real := 6
noncomputable def expected_amount_per_person : Real := 25.48

theorem fair_share_of_bill :
  (total_bill + (tip_percent * total_bill)) / num_people = expected_amount_per_person :=
by
  sorry

end fair_share_of_bill_l2017_201767


namespace fraction_simplification_l2017_201788

theorem fraction_simplification (a b c x y : ℝ) (m : ℝ) :
  (∀ (x y : ℝ), (y ≠ 0 → (y^2 / x^2) ≠ (y / x))) ∧
  (∀ (a b c : ℝ), (a + c^2) / (b + c^2) ≠ a / b) ∧
  (∀ (a b m : ℝ), ¬(m ≠ -1 → (a + b) / (m * a + m * b) = 1 / 2)) ∧
  (∃ a b : ℝ, (a - b) / (b - a) = -1) :=
  by
  sorry

end fraction_simplification_l2017_201788


namespace harriet_return_speed_l2017_201729

/-- Harriet's trip details: 
  - speed from A-ville to B-town is 100 km/h
  - the entire trip took 5 hours
  - time to drive from A-ville to B-town is 180 minutes (3 hours) 
  Prove the speed while driving back to A-ville is 150 km/h
--/
theorem harriet_return_speed:
  ∀ (t₁ t₂ : ℝ),
  (t₁ = 3) ∧ 
  (100 * t₁ = d) ∧ 
  (t₁ + t₂ = 5) ∧ 
  (t₂ = 2) →
  (d / t₂ = 150) :=
by
  intros t₁ t₂ h
  sorry

end harriet_return_speed_l2017_201729


namespace length_of_first_train_l2017_201706

theorem length_of_first_train 
  (speed_first_train_kmph : ℝ) 
  (speed_second_train_kmph : ℝ) 
  (crossing_time_s : ℝ) 
  (length_second_train_m : ℝ) 
  (hspeed_first : speed_first_train_kmph = 120) 
  (hspeed_second : speed_second_train_kmph = 80) 
  (htime : crossing_time_s = 9) 
  (hlength_second : length_second_train_m = 320.04) :
  ∃ (length_first_train_m : ℝ), abs (length_first_train_m - 180) < 0.1 :=
by
  sorry

end length_of_first_train_l2017_201706


namespace cube_edge_factor_l2017_201725

theorem cube_edge_factor (e f : ℝ) (h₁ : e > 0) (h₂ : (f * e) ^ 3 = 8 * e ^ 3) : f = 2 :=
by
  sorry

end cube_edge_factor_l2017_201725


namespace children_distribution_l2017_201704

theorem children_distribution (a b c d N : ℕ) 
  (h1 : a > b) (h2 : b > c) (h3 : c > d)
  (h4 : a + b + c + d < 18) 
  (h5 : a * b * c * d = N) : 
  N = 120 ∧ a = 5 ∧ b = 4 ∧ c = 3 ∧ d = 2 := 
by 
  sorry

end children_distribution_l2017_201704


namespace mod_remainder_l2017_201719

theorem mod_remainder (a b c x: ℤ):
    a = 9 → b = 5 → c = 3 → x = 7 →
    (a^6 + b^7 + c^8) % x = 4 :=
by
  intros
  sorry

end mod_remainder_l2017_201719


namespace bottles_difference_l2017_201737

noncomputable def Donald_drinks_bottles (P: ℕ): ℕ := 2 * P + 3
noncomputable def Paul_drinks_bottles: ℕ := 3
noncomputable def actual_Donald_bottles: ℕ := 9

theorem bottles_difference:
  actual_Donald_bottles - 2 * Paul_drinks_bottles = 3 :=
by 
  sorry

end bottles_difference_l2017_201737


namespace min_value_a_plus_3b_plus_9c_l2017_201787

theorem min_value_a_plus_3b_plus_9c {a b c : ℝ} (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a * b * c = 27) :
  a + 3*b + 9*c ≥ 27 :=
sorry

end min_value_a_plus_3b_plus_9c_l2017_201787


namespace least_n_divisibility_condition_l2017_201717

theorem least_n_divisibility_condition :
  ∃ n : ℕ, 0 < n ∧ ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → (k ∣ (n^2 - n + 1) ↔ (n = 5 ∧ k = 3)) := 
sorry

end least_n_divisibility_condition_l2017_201717


namespace bakery_problem_l2017_201741

theorem bakery_problem :
  let chocolate_chip := 154
  let oatmeal_raisin := 86
  let sugar := 52
  let capacity := 16
  let needed_chocolate_chip := capacity - (chocolate_chip % capacity)
  let needed_oatmeal_raisin := capacity - (oatmeal_raisin % capacity)
  let needed_sugar := capacity - (sugar % capacity)
  (needed_chocolate_chip = 6) ∧ (needed_oatmeal_raisin = 10) ∧ (needed_sugar = 12) :=
by
  sorry

end bakery_problem_l2017_201741


namespace shelves_used_l2017_201733

-- Definitions from conditions
def initial_bears : ℕ := 6
def shipment_bears : ℕ := 18
def bears_per_shelf : ℕ := 6

-- Theorem statement
theorem shelves_used : (initial_bears + shipment_bears) / bears_per_shelf = 4 := by
  sorry

end shelves_used_l2017_201733


namespace p_or_q_then_p_and_q_is_false_l2017_201782

theorem p_or_q_then_p_and_q_is_false (p q : Prop) (hpq : p ∨ q) : ¬(p ∧ q) :=
sorry

end p_or_q_then_p_and_q_is_false_l2017_201782


namespace center_of_conic_l2017_201745

-- Define the conic equation
def conic_equation (p q r α β γ : ℝ) : Prop :=
  p * α * β + q * α * γ + r * β * γ = 0

-- Define the barycentric coordinates of the center
def center_coordinates (p q r : ℝ) : ℝ × ℝ × ℝ :=
  (r * (p + q - r), q * (p + r - q), p * (r + q - p))

-- Theorem to prove that the barycentric coordinates of the center are as expected
theorem center_of_conic (p q r α β γ : ℝ) (h : conic_equation p q r α β γ) :
  center_coordinates p q r = (r * (p + q - r), q * (p + r - q), p * (r + q - p)) := 
sorry

end center_of_conic_l2017_201745


namespace math_problem_l2017_201749

noncomputable def parametric_equation_line (x y t : ℝ) : Prop :=
  x = 1 + (1/2) * t ∧ y = -5 + (Real.sqrt 3 / 2) * t

noncomputable def polar_equation_circle (ρ θ : ℝ) : Prop :=
  ρ = 8 * Real.sin θ

noncomputable def line_disjoint_circle (sqrt3 x y d : ℝ) : Prop :=
  sqrt3 = Real.sqrt 3 ∧ x = 0 ∧ y = 4 ∧ d = (9 + sqrt3) / 2 ∧ d > 4

theorem math_problem 
  (t θ x y ρ sqrt3 d : ℝ) :
  parametric_equation_line x y t ∧
  polar_equation_circle ρ θ ∧
  line_disjoint_circle sqrt3 x y d :=
by
  sorry

end math_problem_l2017_201749


namespace range_of_a_l2017_201746

noncomputable def f (x a : ℝ) := Real.log x + 1 / 2 * x^2 + a * x

theorem range_of_a
  (a : ℝ)
  (h : ∃ x : ℝ, x > 0 ∧ (1/x + x + a = 3)) :
  a ≤ 1 :=
by
  sorry

end range_of_a_l2017_201746


namespace probability_sum_even_for_three_cubes_l2017_201777

-- Define the probability function
def probability_even_sum (n: ℕ) : ℚ :=
  if n > 0 then 1 / 2 else 0

theorem probability_sum_even_for_three_cubes : probability_even_sum 3 = 1 / 2 :=
by
  sorry

end probability_sum_even_for_three_cubes_l2017_201777


namespace arithmetic_sequence_a17_l2017_201750

theorem arithmetic_sequence_a17 (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : S 13 = 78)
  (h2 : a 7 + a 12 = 10)
  (h_sum : ∀ n, S n = n * a 1 + (n * (n - 1) / 2) * (a 1 + (a 2 - a 1) / (2 - 1)))
  (h_term : ∀ n, a n = a 1 + (n - 1) * (a 2 - a 1) / (2 - 1)) :
  a 17 = 2 :=
by
  sorry

end arithmetic_sequence_a17_l2017_201750


namespace range_of_m_if_real_roots_specific_m_given_conditions_l2017_201760

open Real

-- Define the quadratic equation and its conditions
def quadratic_eq (m : ℝ) (x : ℝ) : Prop := x ^ 2 - x + 2 * m - 4 = 0
def has_real_roots (m : ℝ) : Prop := ∃ x1 x2 : ℝ, quadratic_eq m x1 ∧ quadratic_eq m x2

-- Proof that m ≤ 17/8 if the quadratic equation has real roots
theorem range_of_m_if_real_roots (m : ℝ) : has_real_roots m → m ≤ 17 / 8 := 
sorry

-- Define a condition on the roots
def roots_condition (x1 x2 m : ℝ) : Prop := (x1 - 3) * (x2 - 3) = m ^ 2 - 1

-- Proof of specific m when roots condition is given
theorem specific_m_given_conditions (m : ℝ) :
  (∃ x1 x2 : ℝ, quadratic_eq m x1 ∧ quadratic_eq m x2 ∧ roots_condition x1 x2 m) → m = -1 :=
sorry

end range_of_m_if_real_roots_specific_m_given_conditions_l2017_201760


namespace camila_weeks_needed_l2017_201792

/--
Camila has only gone hiking 7 times.
Amanda has gone on 8 times as many hikes as Camila.
Steven has gone on 15 more hikes than Amanda.
Camila plans to go on 4 hikes a week.

Prove that it will take Camila 16 weeks to achieve her goal of hiking as many times as Steven.
-/
noncomputable def hikes_needed_to_match_steven : ℕ :=
  let camila_hikes := 7
  let amanda_hikes := 8 * camila_hikes
  let steven_hikes := amanda_hikes + 15
  let additional_hikes_needed := steven_hikes - camila_hikes
  additional_hikes_needed / 4

theorem camila_weeks_needed : hikes_needed_to_match_steven = 16 := 
  sorry

end camila_weeks_needed_l2017_201792


namespace repeated_1991_mod_13_l2017_201722

theorem repeated_1991_mod_13 (k : ℕ) : 
  ((10^4 - 9) * (1991 * (10^(4*k) - 1)) / 9) % 13 = 8 :=
by
  sorry

end repeated_1991_mod_13_l2017_201722


namespace find_local_min_l2017_201798

def z (x y : ℝ) : ℝ := x^2 + 2 * y^2 - 2 * x * y - x - 2 * y

theorem find_local_min: ∃ (x y : ℝ), x = 2 ∧ y = 3/2 ∧ ∀ ⦃h : ℝ⦄, h ≠ 0 → z (2 + h) (3/2 + h) > z 2 (3/2) :=
by
  sorry

end find_local_min_l2017_201798


namespace homework_problems_l2017_201796

noncomputable def problems_solved (p t : ℕ) : ℕ := p * t

theorem homework_problems (p t : ℕ) (h_eq: p * t = (3 * p - 5) * (t - 3))
  (h_pos_p: p > 0) (h_pos_t: t > 0) (h_p_ge_15: p ≥ 15) 
  (h_friend_did_20: (3 * p - 5) * (t - 3) ≥ 20) : 
  problems_solved p t = 100 :=
by
  sorry

end homework_problems_l2017_201796


namespace swimming_speed_eq_l2017_201773

theorem swimming_speed_eq (S R H : ℝ) (h1 : R = 9) (h2 : H = 5) (h3 : H = (2 * S * R) / (S + R)) :
  S = 45 / 13 :=
by
  sorry

end swimming_speed_eq_l2017_201773


namespace regular_polygon_sides_l2017_201734

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ (k : ℕ), (k : ℕ) * 18 = 360) : n = 20 :=
by
  -- Proof body here
  sorry

end regular_polygon_sides_l2017_201734


namespace max_diameters_l2017_201700

theorem max_diameters (n : ℕ) (points : Finset (ℝ × ℝ)) (h : n ≥ 3) (hn : points.card = n)
  (d : ℝ) (h_d_max : ∀ {p q : ℝ × ℝ}, p ∈ points → q ∈ points → dist p q ≤ d) :
  ∃ m : ℕ, m ≤ n ∧ (∀ {p q : ℝ × ℝ}, p ∈ points → q ∈ points → dist p q = d → m ≤ n) := 
sorry

end max_diameters_l2017_201700


namespace problem_π_digit_sequence_l2017_201713

def f (n : ℕ) : ℕ :=
  match n with
  | 1  => 1
  | 2  => 4
  | 3  => 1
  | 4  => 5
  | 5  => 9
  | 6  => 2
  | 7  => 6
  | 8  => 5
  | 9  => 3
  | 10 => 5
  | _  => 0  -- for simplicity we define other cases arbitrarily

theorem problem_π_digit_sequence :
  ∃ n : ℕ, n > 0 ∧ f (f (f (f (f 10)))) = 1 := by
  sorry

end problem_π_digit_sequence_l2017_201713


namespace age_of_twin_brothers_l2017_201736

theorem age_of_twin_brothers (x : Nat) : (x + 1) * (x + 1) = x * x + 11 ↔ x = 5 :=
by
  sorry  -- Proof omitted.

end age_of_twin_brothers_l2017_201736


namespace proportionality_cube_and_fourth_root_l2017_201793

variables (x y z : ℝ) (k j m n : ℝ)

theorem proportionality_cube_and_fourth_root (h1 : x = k * y^3) (h2 : y = j * z^(1/4)) : 
  ∃ m : ℝ, ∃ n : ℝ, x = m * z^n ∧ n = 3/4 :=
by
  sorry

end proportionality_cube_and_fourth_root_l2017_201793


namespace find_x_l2017_201790

theorem find_x (x : ℝ) (h : x^29 * 4^15 = 2 * 10^29) : x = 5 := 
by 
  sorry

end find_x_l2017_201790


namespace students_opted_both_math_science_l2017_201786

def total_students : ℕ := 40
def not_opted_math : ℕ := 10
def not_opted_science : ℕ := 15
def not_opted_either : ℕ := 2

theorem students_opted_both_math_science :
  let T := total_students
  let M' := not_opted_math
  let S' := not_opted_science
  let E := not_opted_either
  let B := (T - M') + (T - S') - (T - E)
  B = 17 :=
by
  sorry

end students_opted_both_math_science_l2017_201786


namespace circle_rational_points_l2017_201785

theorem circle_rational_points :
  ( ∃ B : ℚ × ℚ, ∀ k : ℚ, B ∈ {p | p.1 ^ 2 + 2 * p.1 + p.2 ^ 2 = 1992} ) ∧ 
  ( (42 : ℤ)^2 + 2 * 42 + 12^2 = 1992 ) :=
by
  sorry

end circle_rational_points_l2017_201785


namespace water_volume_per_minute_l2017_201759

theorem water_volume_per_minute (depth width : ℝ) (flow_rate_kmph : ℝ) 
  (H_depth : depth = 5) 
  (H_width : width = 35) 
  (H_flow_rate_kmph : flow_rate_kmph = 2) : 
  (depth * width * (flow_rate_kmph * 1000 / 60)) = 5832.75 :=
by
  sorry

end water_volume_per_minute_l2017_201759


namespace least_number_to_subtract_l2017_201738

theorem least_number_to_subtract (n : ℕ) (h : n = 13294) : ∃ k : ℕ, n - 1 = k * 97 :=
by
  sorry

end least_number_to_subtract_l2017_201738


namespace geometric_sequence_increasing_iff_l2017_201753

variable {a : ℕ → ℝ} {q : ℝ}

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

def is_increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n < a (n + 1)

theorem geometric_sequence_increasing_iff 
  (ha : is_geometric_sequence a q) 
  (h : a 0 < a 1 ∧ a 1 < a 2) : 
  is_increasing_sequence a ↔ (a 0 < a 1 ∧ a 1 < a 2) := 
sorry

end geometric_sequence_increasing_iff_l2017_201753


namespace cost_of_three_stamps_is_correct_l2017_201728

-- Define the cost of one stamp
def cost_of_one_stamp : ℝ := 0.34

-- Define the number of stamps
def number_of_stamps : ℕ := 3

-- Define the expected total cost for three stamps
def expected_cost : ℝ := 1.02

-- Prove that the cost of three stamps is equal to the expected cost
theorem cost_of_three_stamps_is_correct : cost_of_one_stamp * number_of_stamps = expected_cost :=
by
  sorry

end cost_of_three_stamps_is_correct_l2017_201728


namespace length_of_faster_train_is_380_meters_l2017_201714

-- Defining the conditions
def speed_faster_train_kmph := 144
def speed_slower_train_kmph := 72
def time_seconds := 19

-- Conversion factor
def kmph_to_mps (speed : Nat) : Nat := speed * 1000 / 3600

-- Relative speed in m/s
def relative_speed_mps : Nat := kmph_to_mps (speed_faster_train_kmph - speed_slower_train_kmph)

-- Problem statement: Prove that the length of the faster train is 380 meters
theorem length_of_faster_train_is_380_meters :
  relative_speed_mps * time_seconds = 380 :=
sorry

end length_of_faster_train_is_380_meters_l2017_201714


namespace fruit_eating_problem_l2017_201707

theorem fruit_eating_problem (a₀ p₀ o₀ : ℕ) (h₀ : a₀ = 5) (h₁ : p₀ = 8) (h₂ : o₀ = 11) :
  ¬ ∃ (d : ℕ), (a₀ - d) = (p₀ - d) ∧ (p₀ - d) = (o₀ - d) ∧ ∀ k, k ≤ d → ((a₀ - k) + (p₀ - k) + (o₀ - k) = 24 - 2 * k ∧ a₀ - k ≥ 0 ∧ p₀ - k ≥ 0 ∧ o₀ - k ≥ 0) :=
by
  sorry

end fruit_eating_problem_l2017_201707


namespace solution_set_of_quadratic_inequality_l2017_201756

theorem solution_set_of_quadratic_inequality 
  (f : ℝ → ℝ) 
  (h₁ : ∀ x, f x < 0 ↔ x < -1 ∨ x > 1 / 3)
  (h₂ : ∀ x, f (Real.exp x) > 0 ↔ x < -Real.log 3) : 
  ∀ x, f (Real.exp x) > 0 ↔ x < -Real.log 3 := 
by
  intro x
  exact h₂ x

end solution_set_of_quadratic_inequality_l2017_201756


namespace impossible_to_transport_50_stones_l2017_201742

def arithmetic_sequence (a d n : ℕ) : List ℕ :=
  List.range n |>.map (fun i => a + i * d)

def can_transport (weights : List ℕ) (k : ℕ) (max_weight : ℕ) : Prop :=
  ∃ partition : List (List ℕ), partition.length = k ∧
    (∀ part ∈ partition, (part.sum ≤ max_weight))

theorem impossible_to_transport_50_stones :
  ¬ can_transport (arithmetic_sequence 370 2 50) 7 3000 :=
by
  sorry

end impossible_to_transport_50_stones_l2017_201742


namespace multiplication_pattern_correct_l2017_201752

theorem multiplication_pattern_correct :
  (1 * 9 + 2 = 11) ∧
  (12 * 9 + 3 = 111) ∧
  (123 * 9 + 4 = 1111) ∧
  (1234 * 9 + 5 = 11111) ∧
  (12345 * 9 + 6 = 111111) →
  123456 * 9 + 7 = 1111111 :=
by
  sorry

end multiplication_pattern_correct_l2017_201752


namespace number_of_red_balls_l2017_201715

-- Initial conditions
def num_black_balls : ℕ := 7
def num_white_balls : ℕ := 5
def freq_red_ball : ℝ := 0.4

-- Proving the number of red balls
theorem number_of_red_balls (total_balls : ℕ) (num_red_balls : ℕ) :
  total_balls = num_black_balls + num_white_balls + num_red_balls ∧
  (num_red_balls : ℝ) / total_balls = freq_red_ball →
  num_red_balls = 8 :=
by
  sorry

end number_of_red_balls_l2017_201715


namespace sum_of_a_b_either_1_or_neg1_l2017_201762

theorem sum_of_a_b_either_1_or_neg1 (a b : ℝ) (h1 : a + a = 0) (h2 : b * b = 1) : a + b = 1 ∨ a + b = -1 :=
by {
  sorry
}

end sum_of_a_b_either_1_or_neg1_l2017_201762


namespace farmers_acres_to_clean_l2017_201774

-- Definitions of the main quantities
variables (A D : ℕ)

-- Conditions
axiom condition1 : A = 80 * D
axiom condition2 : 90 * (D - 1) + 30 = A

-- Theorem asserting the total number of acres to be cleaned
theorem farmers_acres_to_clean : A = 480 :=
by
  -- The proof would go here, but is omitted as per instructions
  sorry

end farmers_acres_to_clean_l2017_201774


namespace find_f_l2017_201727

def f (x : ℝ) : ℝ := 3 * x + 2

theorem find_f (x : ℝ) : f x = 3 * x + 2 :=
  sorry

end find_f_l2017_201727


namespace find_num_boys_l2017_201778

-- Definitions for conditions
def num_children : ℕ := 13
def num_girls (num_boys : ℕ) : ℕ := num_children - num_boys

-- We will assume we have a predicate representing the truthfulness of statements.
-- boys tell the truth to boys and lie to girls
-- girls tell the truth to girls and lie to boys

theorem find_num_boys (boys_truth_to_boys : Prop) 
                      (boys_lie_to_girls : Prop) 
                      (girls_truth_to_girls : Prop) 
                      (girls_lie_to_boys : Prop)
                      (alternating_statements : Prop) : 
  ∃ (num_boys : ℕ), num_boys = 7 := 
  sorry

end find_num_boys_l2017_201778


namespace total_price_l2017_201799

theorem total_price (r w : ℕ) (hr : r = 4275) (hw : w = r - 1490) : r + w = 7060 :=
by
  sorry

end total_price_l2017_201799


namespace find_biology_marks_l2017_201748

variables (e m p c b : ℕ)
variable (a : ℝ)

def david_marks_in_biology : Prop :=
  e = 72 ∧
  m = 45 ∧
  p = 72 ∧
  c = 77 ∧
  a = 68.2 ∧
  (e + m + p + c + b) / 5 = a

theorem find_biology_marks (h : david_marks_in_biology e m p c b a) : b = 75 :=
sorry

end find_biology_marks_l2017_201748


namespace line_and_circle_separate_l2017_201712

theorem line_and_circle_separate
  (θ : ℝ) (hθ : ¬ ∃ k : ℤ, θ = k * Real.pi) :
  ¬ ∃ (x y : ℝ), (x^2 + y^2 = 1 / 2) ∧ (x * Real.cos θ + y - 1 = 0) :=
by
  sorry

end line_and_circle_separate_l2017_201712


namespace cobbler_works_fri_hours_l2017_201702

-- Conditions
def mending_rate : ℕ := 3  -- Pairs of shoes per hour
def mon_to_thu_days : ℕ := 4
def hours_per_day : ℕ := 8
def weekly_mended_pairs : ℕ := 105

-- Translate the conditions
def hours_mended_mon_to_thu : ℕ := mon_to_thu_days * hours_per_day
def pairs_mended_mon_to_thu : ℕ := mending_rate * hours_mended_mon_to_thu
def pairs_mended_fri : ℕ := weekly_mended_pairs - pairs_mended_mon_to_thu

-- Theorem statement to prove the desired question
theorem cobbler_works_fri_hours : (pairs_mended_fri / mending_rate) = 3 := by
  sorry

end cobbler_works_fri_hours_l2017_201702


namespace Keenan_essay_length_l2017_201723

-- Given conditions
def words_per_hour_first_two_hours : ℕ := 400
def first_two_hours : ℕ := 2
def words_per_hour_later : ℕ := 200
def later_hours : ℕ := 2

-- Total words written in 4 hours
def total_words : ℕ := words_per_hour_first_two_hours * first_two_hours + words_per_hour_later * later_hours

-- Theorem statement
theorem Keenan_essay_length : total_words = 1200 := by
  sorry

end Keenan_essay_length_l2017_201723


namespace infinitely_many_squares_of_form_l2017_201721

theorem infinitely_many_squares_of_form (k : ℕ) (hk : 0 < k) : 
  ∃ (n : ℕ), ∀ m : ℕ, ∃ n' > n, 2 * k * n' - 7 = m^2 :=
sorry

end infinitely_many_squares_of_form_l2017_201721


namespace triangle_property_proof_l2017_201769

noncomputable def triangleABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  a = 2 * Real.sqrt 2 ∧
  b = 5 ∧
  c = Real.sqrt 13 ∧
  C = Real.pi / 4 ∧
  ∃ sinA : ℝ, sinA = 2 * Real.sqrt 13 / 13 ∧
  ∃ sin_2A_plus_pi_4 : ℝ, sin_2A_plus_pi_4 = 17 * Real.sqrt 2 / 26

theorem triangle_property_proof :
  ∃ (A B C : ℝ), 
  triangleABC (2 * Real.sqrt 2) 5 (Real.sqrt 13) A B C
:= sorry

end triangle_property_proof_l2017_201769


namespace number_of_principals_in_oxford_high_school_l2017_201705

-- Define the conditions
def numberOfTeachers : ℕ := 48
def numberOfClasses : ℕ := 15
def studentsPerClass : ℕ := 20
def totalStudents : ℕ := numberOfClasses * studentsPerClass
def totalPeople : ℕ := 349
def numberOfPrincipals : ℕ := totalPeople - (numberOfTeachers + totalStudents)

-- Proposition: Prove the number of principals in Oxford High School
theorem number_of_principals_in_oxford_high_school :
  numberOfPrincipals = 1 := by sorry

end number_of_principals_in_oxford_high_school_l2017_201705


namespace height_after_16_minutes_l2017_201781

noncomputable def ferris_wheel_height (t : ℝ) : ℝ :=
  8 * Real.sin ((Real.pi / 6) * t - Real.pi / 2) + 10

theorem height_after_16_minutes : ferris_wheel_height 16 = 6 := by
  sorry

end height_after_16_minutes_l2017_201781


namespace decreased_cost_l2017_201754

theorem decreased_cost (original_cost : ℝ) (decrease_percentage : ℝ) (h1 : original_cost = 200) (h2 : decrease_percentage = 0.50) : 
  (original_cost - original_cost * decrease_percentage) = 100 :=
by
  -- This is the proof placeholder
  sorry

end decreased_cost_l2017_201754


namespace GPA_of_rest_of_classroom_l2017_201731

variable (n : ℕ) (x : ℝ)
variable (H1 : ∀ n, n > 0)
variable (H2 : (15 * n + 2 * n * x) / (3 * n) = 17)

theorem GPA_of_rest_of_classroom (n : ℕ) (H1 : ∀ n, n > 0) (H2 : (15 * n + 2 * n * x) / (3 * n) = 17) : x = 18 := by
  sorry

end GPA_of_rest_of_classroom_l2017_201731


namespace boys_of_other_communities_l2017_201744

axiom total_boys : ℕ
axiom muslim_percentage : ℝ
axiom hindu_percentage : ℝ
axiom sikh_percentage : ℝ

noncomputable def other_boy_count (total_boys : ℕ) 
                                   (muslim_percentage : ℝ) 
                                   (hindu_percentage : ℝ) 
                                   (sikh_percentage : ℝ) : ℝ :=
  let total_percentage := muslim_percentage + hindu_percentage + sikh_percentage
  let other_percentage := 1 - total_percentage
  other_percentage * total_boys

theorem boys_of_other_communities : 
    other_boy_count 850 0.44 0.32 0.10 = 119 :=
  by 
    sorry

end boys_of_other_communities_l2017_201744


namespace cycling_sequences_reappear_after_28_cycles_l2017_201735

/-- Cycling pattern of letters and digits. Letter cycle length is 7; digit cycle length is 4.
Prove that the LCM of 7 and 4 is 28, which is the first line on which both sequences will reappear -/
theorem cycling_sequences_reappear_after_28_cycles 
  (letters_cycle_length : ℕ) (digits_cycle_length : ℕ) 
  (h_letters : letters_cycle_length = 7) 
  (h_digits : digits_cycle_length = 4) 
  : Nat.lcm letters_cycle_length digits_cycle_length = 28 :=
by
  rw [h_letters, h_digits]
  sorry

end cycling_sequences_reappear_after_28_cycles_l2017_201735


namespace germination_probability_l2017_201780

open Nat

noncomputable def binomial_coeff (n k : ℕ) : ℕ :=
  Nat.choose n k

noncomputable def probability_of_success (p : ℚ) (k : ℕ) (n : ℕ) : ℚ :=
  (binomial_coeff n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem germination_probability :
  probability_of_success 0.9 5 7 = 0.124 := by
  sorry

end germination_probability_l2017_201780


namespace planting_trees_system_of_equations_l2017_201770

/-- This formalizes the problem where we have 20 young pioneers in total, 
each boy planted 3 trees, each girl planted 2 trees,
and together they planted a total of 52 tree seedlings.
We need to formalize proving that the system of linear equations is as follows:
x + y = 20
3x + 2y = 52
-/
theorem planting_trees_system_of_equations (x y : ℕ) (h1 : x + y = 20)
  (h2 : 3 * x + 2 * y = 52) : 
  (x + y = 20 ∧ 3 * x + 2 * y = 52) :=
by
  exact ⟨h1, h2⟩

end planting_trees_system_of_equations_l2017_201770


namespace jason_cutting_grass_time_l2017_201703

-- Conditions
def time_to_cut_one_lawn : ℕ := 30 -- in minutes
def lawns_cut_each_day : ℕ := 8
def days : ℕ := 2
def minutes_in_an_hour : ℕ := 60

-- Proof that the number of hours Jason spends cutting grass over the weekend is 8
theorem jason_cutting_grass_time:
  ((lawns_cut_each_day * days) * time_to_cut_one_lawn) / minutes_in_an_hour = 8 :=
by
  sorry

end jason_cutting_grass_time_l2017_201703


namespace son_time_to_complete_work_l2017_201716

noncomputable def man_work_rate : ℚ := 1 / 6
noncomputable def combined_work_rate : ℚ := 1 / 3

theorem son_time_to_complete_work :
  (1 / (combined_work_rate - man_work_rate)) = 6 := by
  sorry

end son_time_to_complete_work_l2017_201716


namespace cube_volume_is_8_l2017_201797

theorem cube_volume_is_8 (a : ℕ) 
  (h_cond : (a+2) * (a-2) * a = a^3 - 8) : 
  a^3 = 8 := 
by
  sorry

end cube_volume_is_8_l2017_201797


namespace rational_ordering_l2017_201732

theorem rational_ordering :
  (-3:ℚ)^2 < -1/3 ∧ (-1/3 < ((-3):ℚ)^2 ∧ ((-3:ℚ)^2 = |((-3:ℚ))^2|)) := 
by 
  sorry

end rational_ordering_l2017_201732


namespace two_pow_1000_mod_3_two_pow_1000_mod_5_two_pow_1000_mod_11_two_pow_1000_mod_13_l2017_201791

theorem two_pow_1000_mod_3 : 2^1000 % 3 = 1 := sorry
theorem two_pow_1000_mod_5 : 2^1000 % 5 = 1 := sorry
theorem two_pow_1000_mod_11 : 2^1000 % 11 = 1 := sorry
theorem two_pow_1000_mod_13 : 2^1000 % 13 = 3 := sorry

end two_pow_1000_mod_3_two_pow_1000_mod_5_two_pow_1000_mod_11_two_pow_1000_mod_13_l2017_201791


namespace felipe_total_time_l2017_201751

-- Given definitions
def combined_time_without_breaks := 126
def combined_time_with_breaks := 150
def felipe_break := 6
def emilio_break := 2 * felipe_break
def carlos_break := emilio_break / 2

theorem felipe_total_time (F E C : ℕ) 
(h1 : F = E / 2) 
(h2 : C = F + E)
(h3 : (F + E + C) = combined_time_without_breaks)
(h4 : (F + felipe_break) + (E + emilio_break) + (C + carlos_break) = combined_time_with_breaks) : 
F + felipe_break = 27 := 
sorry

end felipe_total_time_l2017_201751


namespace inequality_solution_l2017_201764

theorem inequality_solution (x : ℝ) :
  (x < -2 ∨ (-1 < x ∧ x < 2) ∨ (3 < x ∧ x < 4) ∨ (5 < x ∧ x < 7) ∨ (x > 8)) ↔
  (2 / (x - 2) - 3 / (x - 3) + 3 / (x - 4) - 2 / (x - 5) < 1 / 20) :=
sorry

end inequality_solution_l2017_201764


namespace race_head_start_l2017_201720

/-- A's speed is 22/19 times that of B. If A and B run a race, A should give B a head start of (3 / 22) of the race length so the race ends in a dead heat. -/
theorem race_head_start {Va Vb L H : ℝ} (hVa : Va = (22 / 19) * Vb) (hL_Va : L / Va = (L - H) / Vb) : 
  H = (3 / 22) * L :=
by
  sorry

end race_head_start_l2017_201720


namespace total_respondents_l2017_201726

theorem total_respondents (X Y : ℕ) 
  (hX : X = 60) 
  (hRatio : 3 * Y = X) : 
  X + Y = 80 := 
by
  sorry

end total_respondents_l2017_201726


namespace students_in_class_l2017_201747

theorem students_in_class (S : ℕ) (h1 : S / 3 + 2 * S / 5 + 12 = S) : S = 45 :=
sorry

end students_in_class_l2017_201747


namespace perfect_square_n_l2017_201784

open Nat

theorem perfect_square_n (n : ℕ) : 
  (∃ k : ℕ, 2 ^ (n + 1) * n = k ^ 2) ↔ 
  (∃ m : ℕ, n = 2 * m ^ 2) ∨ (∃ odd_k : ℕ, n = odd_k ^ 2 ∧ odd_k % 2 = 1) := 
sorry

end perfect_square_n_l2017_201784


namespace fill_tank_with_leak_l2017_201718

theorem fill_tank_with_leak (A L : ℝ) (h1 : A = 1 / 6) (h2 : L = 1 / 18) : (1 / (A - L)) = 9 :=
by
  sorry

end fill_tank_with_leak_l2017_201718


namespace no_solution_for_equation_l2017_201776

/-- The given equation expressed using letters as unique digits:
    ∑ (letters as digits) from БАРАНКА + БАРАБАН + КАРАБАС = ПАРАЗИТ
    We aim to prove that there are no valid digit assignments satisfying the equation. -/
theorem no_solution_for_equation :
  ∀ (b a r n k s p i t: ℕ),
  b ≠ a ∧ b ≠ r ∧ b ≠ n ∧ b ≠ k ∧ b ≠ s ∧ b ≠ p ∧ b ≠ i ∧ b ≠ t ∧
  a ≠ r ∧ a ≠ n ∧ a ≠ k ∧ a ≠ s ∧ a ≠ p ∧ a ≠ i ∧ a ≠ t ∧
  r ≠ n ∧ r ≠ k ∧ r ≠ s ∧ r ≠ p ∧ r ≠ i ∧ r ≠ t ∧
  n ≠ k ∧ n ≠ s ∧ n ≠ p ∧ n ≠ i ∧ n ≠ t ∧
  k ≠ s ∧ k ≠ p ∧ k ≠ i ∧ k ≠ t ∧
  s ≠ p ∧ s ≠ i ∧ s ≠ t ∧
  p ≠ i ∧ p ≠ t ∧
  i ≠ t →
  100000 * b + 10000 * a + 1000 * r + 100 * a + 10 * n + k +
  100000 * b + 10000 * a + 1000 * r + 100 * a + 10 * b + a + n +
  100000 * k + 10000 * a + 1000 * r + 100 * a + 10 * b + a + s ≠ 
  100000 * p + 10000 * a + 1000 * r + 100 * a + 10 * z + i + t :=
sorry

end no_solution_for_equation_l2017_201776


namespace min_x_y_l2017_201772

open Real

theorem min_x_y {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (h : 1/x + 9/y = 1) : x + y ≥ 16 := 
sorry

end min_x_y_l2017_201772


namespace odd_factors_of_360_l2017_201771

theorem odd_factors_of_360 : ∃ n : ℕ, n = 6 ∧ ∀ k : ℕ, k ∣ 360 → k % 2 = 1 ↔ k ∣ (3^2 * 5^1) := 
by
  sorry

end odd_factors_of_360_l2017_201771


namespace find_other_number_l2017_201758

theorem find_other_number (A B : ℕ) (HCF LCM : ℕ)
  (hA : A = 24)
  (hHCF: (HCF : ℚ) = 16)
  (hLCM: (LCM : ℚ) = 312)
  (hHCF_LCM: HCF * LCM = A * B) : 
  B = 208 :=
by
  sorry

end find_other_number_l2017_201758


namespace find_c_l2017_201766

def P (x : ℝ) (c : ℝ) : ℝ :=
  x^3 + 3*x^2 + c*x + 15

theorem find_c (c : ℝ) : (x - 3 = P x c → c = -23) := by
  sorry

end find_c_l2017_201766


namespace matrix_not_invertible_l2017_201724

noncomputable def determinant (a b c d : ℝ) : ℝ := a * d - b * c

theorem matrix_not_invertible (x : ℝ) :
  determinant (2*x + 1) 9 (4 - x) 10 = 0 ↔ x = 26/29 := by
  sorry

end matrix_not_invertible_l2017_201724


namespace acute_angle_sum_l2017_201740

theorem acute_angle_sum (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
    (h1 : Real.sin α = (2 * Real.sqrt 5) / 5) (h2 : Real.sin β = (3 * Real.sqrt 10) / 10) :
    α + β = 3 * Real.pi / 4 :=
sorry

end acute_angle_sum_l2017_201740


namespace not_odd_function_l2017_201795

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

noncomputable def f (x : ℝ) : ℝ :=
  Real.sin (x ^ 2 + 1)

theorem not_odd_function : ¬ is_odd_function f := by
  sorry

end not_odd_function_l2017_201795


namespace count_squares_with_center_55_25_l2017_201709

noncomputable def number_of_squares_with_natural_number_coordinates : ℕ :=
  600

theorem count_squares_with_center_55_25 :
  ∀ (x y : ℕ), (x = 55) ∧ (y = 25) → number_of_squares_with_natural_number_coordinates = 600 :=
by
  intros x y h
  cases h
  sorry

end count_squares_with_center_55_25_l2017_201709


namespace divide_composite_products_l2017_201743

theorem divide_composite_products :
  let first_three := [4, 6, 8]
  let next_three := [9, 10, 12]
  let prod_first_three := first_three.prod
  let prod_next_three := next_three.prod
  (prod_first_three : ℚ) / prod_next_three = 8 / 45 :=
by
  sorry

end divide_composite_products_l2017_201743


namespace correct_average_l2017_201783

theorem correct_average (initial_avg : ℝ) (n : ℕ) (error1 : ℝ) (wrong_num : ℝ) (correct_num : ℝ) :
  initial_avg = 40.2 → n = 10 → error1 = 19 → wrong_num = 13 → correct_num = 31 →
  (initial_avg * n - error1 - wrong_num + correct_num) / n = 40.1 :=
by
  intros
  sorry

end correct_average_l2017_201783


namespace triangle_side_length_l2017_201765

theorem triangle_side_length (A B C : ℝ) (h1 : AC = Real.sqrt 2) (h2: AB = 2)
  (h3 : (Real.sqrt 3 * Real.sin A + Real.cos A) / (Real.sqrt 3 * Real.cos A - Real.sin A) = Real.tan (5 * Real.pi / 12)) :
  BC = Real.sqrt 2 := 
sorry

end triangle_side_length_l2017_201765


namespace total_savings_l2017_201739

-- Define the given conditions
def number_of_tires : ℕ := 4
def sale_price : ℕ := 75
def original_price : ℕ := 84

-- State the proof problem
theorem total_savings : (original_price - sale_price) * number_of_tires = 36 :=
by
  -- Proof omitted
  sorry

end total_savings_l2017_201739


namespace find_value_at_frac_one_third_l2017_201730

theorem find_value_at_frac_one_third
  (f : ℝ → ℝ) 
  (a : ℝ)
  (h₁ : ∀ x, f x = x ^ a)
  (h₂ : f 2 = 1 / 4) :
  f (1 / 3) = 9 := 
  sorry

end find_value_at_frac_one_third_l2017_201730


namespace bus_stop_l2017_201775

theorem bus_stop (M H : ℕ) 
  (h1 : H = 2 * (M - 15))
  (h2 : M - 15 = 5 * (H - 45)) :
  M = 40 ∧ H = 50 := 
sorry

end bus_stop_l2017_201775


namespace coefficients_sum_l2017_201768

noncomputable def f (x : ℝ) : ℝ := (2 * x - 1) ^ 4

theorem coefficients_sum : 
  ∃ (a₀ a₁ a₂ a₃ a₄ : ℝ), 
  ((2 * x - 1) ^ 4 = a₀ + a₁ * x + a₂ * x ^ 2 + a₃ * x ^ 3 + a₄ * x ^ 4) 
  ∧ (a₁ + 2 * a₂ + 3 * a₃ + 4 * a₄ = 8) :=
sorry

end coefficients_sum_l2017_201768


namespace remainder_zero_division_l2017_201708

theorem remainder_zero_division :
  ∀ x : ℂ, (x^2 - x + 1 = 0) →
    ((x^5 + x^4 - x^3 - x^2 + 1) * (x^3 - 1)) % (x^2 - x + 1) = 0 :=
by sorry

end remainder_zero_division_l2017_201708


namespace proof_mn_squared_l2017_201779

theorem proof_mn_squared (m n : ℤ) (h1 : |m| = 3) (h2 : |n| = 2) (h3 : m < n) :
  m^2 + m * n + n^2 = 7 ∨ m^2 + m * n + n^2 = 19 :=
by
  sorry

end proof_mn_squared_l2017_201779


namespace milkman_profit_percentage_l2017_201711

noncomputable def profit_percentage (x : ℝ) : ℝ :=
  let cp_per_litre := x
  let sp_per_litre := 2 * x
  let mixture_litres := 8
  let milk_litres := 6
  let cost_price := milk_litres * cp_per_litre
  let selling_price := mixture_litres * sp_per_litre
  let profit := selling_price - cost_price
  let profit_percentage := (profit / cost_price) * 100
  profit_percentage

theorem milkman_profit_percentage (x : ℝ) 
  (h : x > 0) : 
  profit_percentage x = 166.67 :=
by
  sorry

end milkman_profit_percentage_l2017_201711


namespace length_of_first_train_is_correct_l2017_201710

noncomputable def length_of_first_train 
  (speed_first_train_kmph : ℝ)
  (length_second_train_m : ℝ)
  (speed_second_train_kmph : ℝ)
  (time_crossing_s : ℝ) : ℝ :=
  let speed_first_train_mps := (speed_first_train_kmph * 1000) / 3600
  let speed_second_train_mps := (speed_second_train_kmph * 1000) / 3600
  let relative_speed_mps := speed_first_train_mps + speed_second_train_mps
  let total_distance_m := relative_speed_mps * time_crossing_s
  total_distance_m - length_second_train_m

theorem length_of_first_train_is_correct :
  length_of_first_train 50 112 82 6 = 108.02 :=
by
  sorry

end length_of_first_train_is_correct_l2017_201710


namespace simplify_and_evaluate_expression_l2017_201757

theorem simplify_and_evaluate_expression (x : ℤ) (h1 : -2 < x) (h2 : x < 3) :
    (x ≠ 1) → (x ≠ -1) → (x ≠ 0) → 
    ((x / (x + 1) - (3 * x) / (x - 1)) / (x / (x^2 - 1))) = -8 :=
by 
  intro h3 h4 h5
  sorry

end simplify_and_evaluate_expression_l2017_201757


namespace range_of_m_l2017_201794

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, x^2 + 6 * m * x + m + 8 ≥ 0) ↔ (-8 / 9 ≤ m ∧ m ≤ 1) :=
sorry

end range_of_m_l2017_201794


namespace original_price_of_dish_l2017_201763

theorem original_price_of_dish (P : ℝ) (h1 : ∃ P, John's_payment = (0.9 * P) + (0.15 * P))
                               (h2 : ∃ P, Jane's_payment = (0.9 * P) + (0.135 * P))
                               (h3 : John's_payment = Jane's_payment + 0.51) : P = 34 := by
  -- John's Payment
  let John's_payment := (0.9 * P) + (0.15 * P)
  -- Jane's Payment
  let Jane's_payment := (0.9 * P) + (0.135 * P)
  -- Condition that John paid $0.51 more than Jane
  have h3 : John's_payment = Jane's_payment + 0.51 := sorry
  -- From the given conditions, we need to prove P = 34
  sorry

end original_price_of_dish_l2017_201763


namespace remainder_of_division_l2017_201701

theorem remainder_of_division :
  ∀ (L S R : ℕ), 
  L = 1575 → 
  L - S = 1365 → 
  S * 7 + R = L → 
  R = 105 :=
by
  intros L S R h1 h2 h3
  sorry

end remainder_of_division_l2017_201701


namespace remaining_tickets_l2017_201789

def initial_tickets : ℝ := 49.0
def lost_tickets : ℝ := 6.0
def spent_tickets : ℝ := 25.0

theorem remaining_tickets : initial_tickets - lost_tickets - spent_tickets = 18.0 := by
  sorry

end remaining_tickets_l2017_201789


namespace find_common_ratio_l2017_201755

-- Define the geometric sequence
def geometric_sequence (a₁ q : ℝ) (n : ℕ) : ℝ := a₁ * q^(n-1)

-- Given conditions
lemma a2_eq_8 (a₁ q : ℝ) : geometric_sequence a₁ q 2 = 8 :=
by sorry

lemma a5_eq_64 (a₁ q : ℝ) : geometric_sequence a₁ q 5 = 64 :=
by sorry

-- The common ratio q
theorem find_common_ratio (a₁ q : ℝ) (hq : 0 < q) :
  (geometric_sequence a₁ q 2 = 8) → (geometric_sequence a₁ q 5 = 64) → q = 2 :=
by sorry

end find_common_ratio_l2017_201755

import Mathlib

namespace angle_conversion_l117_11754

/--
 Given an angle in degrees, express it in degrees, minutes, and seconds.
 Theorem: 20.23 degrees can be converted to 20 degrees, 13 minutes, and 48 seconds.
-/
theorem angle_conversion : (20.23:ℝ) = 20 + (13/60 : ℝ) + (48/3600 : ℝ) :=
by
  sorry

end angle_conversion_l117_11754


namespace admission_price_for_adults_l117_11707

def total_people := 610
def num_adults := 350
def child_price := 1
def total_receipts := 960

theorem admission_price_for_adults (A : ℝ) (h1 : 350 * A + 260 = 960) : A = 2 :=
by {
  -- proof omitted
  sorry
}

end admission_price_for_adults_l117_11707


namespace M_inter_N_M_union_not_N_l117_11735

def M : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}
def N : Set ℝ := {x | x > 0}

theorem M_inter_N :
  M ∩ N = {x | 0 < x ∧ x ≤ 3} := 
sorry

theorem M_union_not_N :
  M ∪ {x | x ≤ 0} = {x | x ≤ 3} := 
sorry

end M_inter_N_M_union_not_N_l117_11735


namespace Camp_Cedar_number_of_counselors_l117_11716

theorem Camp_Cedar_number_of_counselors (boys : ℕ) (girls : ℕ) (total_children : ℕ) (counselors : ℕ)
  (h_boys : boys = 40)
  (h_girls : girls = 3 * boys)
  (h_total_children : total_children = boys + girls)
  (h_counselors : counselors = total_children / 8) :
  counselors = 20 :=
by
  -- this is a statement, so we conclude with sorry to skip the proof.
  sorry

end Camp_Cedar_number_of_counselors_l117_11716


namespace find_QS_l117_11753

theorem find_QS (cosR : ℝ) (RS QR QS : ℝ) (h1 : cosR = 3 / 5) (h2 : RS = 10) (h3 : cosR = QR / RS) (h4: QR ^ 2 + QS ^ 2 = RS ^ 2) : QS = 8 :=
by 
  sorry

end find_QS_l117_11753


namespace minimum_value_N_div_a4_possible_values_a4_l117_11734

noncomputable def lcm_10 (a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 : ℕ) : ℕ := 
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm a1 a2) a3) a4) a5) a6) a7) a8) a9) a10

theorem minimum_value_N_div_a4 {a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 : ℕ} 
  (h: a1 < a2 ∧ a2 < a3 ∧ a3 < a4 ∧ a4 < a5 ∧ a5 < a6 ∧ a6 < a7 ∧ a7 < a8 ∧ a8 < a9 ∧ a9 < a10) : 
  (lcm_10 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10) / a4 = 630 := sorry

theorem possible_values_a4 {a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 : ℕ} 
  (h: a1 < a2 ∧ a2 < a3 ∧ a3 < a4 ∧ a4 < a5 ∧ a5 < a6 ∧ a6 < a7 ∧ a7 < a8 ∧ a8 < a9 ∧ a9 < a10)
  (z: 1 ≤ a4 ∧ a4 ≤ 1300) :
  (lcm_10 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10) / a4 = 630 → a4 = 360 ∨ a4 = 720 ∨ a4 = 1080 := sorry

end minimum_value_N_div_a4_possible_values_a4_l117_11734


namespace peter_candles_l117_11751

theorem peter_candles (candles_rupert : ℕ) (ratio : ℝ) (candles_peter : ℕ) 
  (h1 : ratio = 3.5) (h2 : candles_rupert = 35) (h3 : candles_peter = candles_rupert / ratio) : 
  candles_peter = 10 := 
sorry

end peter_candles_l117_11751


namespace sphere_radius_l117_11742

/-- Given the curved surface area (CSA) of a sphere and its formula, 
    prove that the radius of the sphere is 4 cm.
    Conditions:
    - CSA = 4πr²
    - Curved surface area is 64π cm²
-/
theorem sphere_radius (r : ℝ) (h : 4 * Real.pi * r^2 = 64 * Real.pi) : r = 4 := by
  sorry

end sphere_radius_l117_11742


namespace sum_f_neg12_to_13_l117_11797

noncomputable def f (x : ℝ) := 1 / (3^x + Real.sqrt 3)

theorem sum_f_neg12_to_13 : 
  (f (-12) + f (-11) + f (-10) + f (-9) + f (-8) + f (-7) + f (-6)
  + f (-5) + f (-4) + f (-3) + f (-2) + f (-1) + f 0
  + f 1 + f 2 + f 3 + f 4 + f 5 + f 6 + f 7 + f 8 + f 9 + f 10
  + f 11 + f 12 + f 13) = (13 * Real.sqrt 3 / 3) :=
sorry

end sum_f_neg12_to_13_l117_11797


namespace original_square_area_is_144_square_centimeters_l117_11700

noncomputable def area_of_original_square (x : ℝ) : ℝ :=
  x^2 - (x - 3) * (x - 5)

theorem original_square_area_is_144_square_centimeters (x : ℝ) (h : area_of_original_square x = 81) :
  (x = 12) → (x^2 = 144) :=
by
  sorry

end original_square_area_is_144_square_centimeters_l117_11700


namespace coords_A_l117_11791

def A : ℝ × ℝ := (1, -2)

def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

def move_up (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ := (p.1, p.2 + d)

def A' : ℝ × ℝ := reflect_y_axis A

def A'' : ℝ × ℝ := move_up A' 3

theorem coords_A'' : A'' = (-1, 1) := by
  sorry

end coords_A_l117_11791


namespace fence_width_l117_11792

theorem fence_width (L W : ℝ) 
  (circumference_eq : 2 * (L + W) = 30)
  (width_eq : W = 2 * L) : 
  W = 10 :=
by 
  sorry

end fence_width_l117_11792


namespace find_smallest_k_satisfying_cos_square_l117_11759

theorem find_smallest_k_satisfying_cos_square (k : ℕ) (h : ∃ n : ℕ, k^2 = 180 * n - 64):
  k = 48 ∨ k = 53 :=
by sorry

end find_smallest_k_satisfying_cos_square_l117_11759


namespace total_tickets_needed_l117_11712

-- Define the conditions
def rollercoaster_rides (n : Nat) := 3
def catapult_rides (n : Nat) := 2
def ferris_wheel_rides (n : Nat) := 1
def rollercoaster_cost (n : Nat) := 4
def catapult_cost (n : Nat) := 4
def ferris_wheel_cost (n : Nat) := 1

-- Prove the total number of tickets needed
theorem total_tickets_needed : 
  rollercoaster_rides 0 * rollercoaster_cost 0 +
  catapult_rides 0 * catapult_cost 0 +
  ferris_wheel_rides 0 * ferris_wheel_cost 0 = 21 :=
by 
  sorry

end total_tickets_needed_l117_11712


namespace analytical_expression_f_min_value_f_range_of_k_l117_11726

noncomputable def max_real (a b : ℝ) : ℝ :=
  if a ≥ b then a else b

noncomputable def f (x : ℝ) : ℝ :=
  max_real (|x + 1|) (|x - 2|)

noncomputable def g (x k : ℝ) : ℝ :=
  x^2 - k * f x

-- Problem 1: Proving the analytical expression of f(x)
theorem analytical_expression_f (x : ℝ) :
  f x = if x < 0.5 then 2 - x else x + 1 :=
sorry

-- Problem 2: Proving the minimum value of f(x)
theorem min_value_f : ∃ x : ℝ, (∀ y : ℝ, f y ≥ f x) ∧ f x = 3 / 2 :=
sorry

-- Problem 3: Proving the range of k
theorem range_of_k (k : ℝ) :
  (∀ x : ℝ, x ≤ -1 → (g x k) ≤ (g (x - 1) k)) → k ≤ 2 :=
sorry

end analytical_expression_f_min_value_f_range_of_k_l117_11726


namespace maria_change_l117_11773

def cost_per_apple : ℝ := 0.75
def number_of_apples : ℕ := 5
def amount_paid : ℝ := 10.0
def total_cost := number_of_apples * cost_per_apple
def change_received := amount_paid - total_cost

theorem maria_change :
  change_received = 6.25 :=
sorry

end maria_change_l117_11773


namespace initial_goal_proof_l117_11770

def marys_collection (k : ℕ) : ℕ := 5 * k
def scotts_collection (m : ℕ) : ℕ := m / 3
def total_collected (k : ℕ) (m : ℕ) (s : ℕ) : ℕ := k + m + s
def initial_goal (total : ℕ) (excess : ℕ) : ℕ := total - excess

theorem initial_goal_proof : 
  initial_goal (total_collected 600 (marys_collection 600) (scotts_collection (marys_collection 600))) 600 = 4000 :=
by
  sorry

end initial_goal_proof_l117_11770


namespace find_length_of_room_l117_11720

noncomputable def cost_of_paving : ℝ := 21375
noncomputable def rate_per_sq_meter : ℝ := 900
noncomputable def width_of_room : ℝ := 4.75

theorem find_length_of_room :
  ∃ l : ℝ, l = (cost_of_paving / rate_per_sq_meter) / width_of_room ∧ l = 5 := by
  sorry

end find_length_of_room_l117_11720


namespace ratio_of_ages_l117_11747

theorem ratio_of_ages (S F : Nat) 
  (h1 : F = 3 * S) 
  (h2 : (S + 6) + (F + 6) = 156) : 
  (F + 6) / (S + 6) = 19 / 7 := 
by 
  sorry

end ratio_of_ages_l117_11747


namespace find_value_of_a_l117_11718

theorem find_value_of_a (a : ℝ) (h : 0.005 * a = 65) : a = 130 := 
by
  sorry

end find_value_of_a_l117_11718


namespace inequality_abc_l117_11765

theorem inequality_abc (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b * c = 1) :
  (a + b) * (b + c) * (c + a) ≥ 4 * (a + b + c - 1) :=
sorry

end inequality_abc_l117_11765


namespace ellipse_domain_l117_11775

theorem ellipse_domain (m : ℝ) :
  (-1 < m ∧ m < 2 ∧ m ≠ 1 / 2) -> 
  ∃ a b : ℝ, (a = 2 - m) ∧ (b = m + 1) ∧ a > 0 ∧ b > 0 ∧ a ≠ b :=
by
  sorry

end ellipse_domain_l117_11775


namespace find_N_l117_11777

theorem find_N (x N : ℝ) (h1 : x + 1 / x = N) (h2 : x^2 + 1 / x^2 = 2) : N = 2 :=
sorry

end find_N_l117_11777


namespace greatest_integer_x_l117_11711

theorem greatest_integer_x :
  ∃ (x : ℤ), (∀ (y : ℤ), (8 : ℝ) / 11 > (x : ℝ) / 15) ∧
    ¬ (8 / 11 > (x + 1 : ℝ) / 15) ∧
    x = 10 :=
by
  sorry

end greatest_integer_x_l117_11711


namespace max_candies_theorem_l117_11741

-- Defining constants: the number of students and the total number of candies.
def n : ℕ := 40
def T : ℕ := 200

-- Defining the condition that each student takes at least 2 candies.
def min_candies_per_student : ℕ := 2

-- Calculating the minimum total number of candies taken by 39 students.
def min_total_for_39_students := min_candies_per_student * (n - 1)

-- The maximum number of candies one student can take.
def max_candies_one_student_can_take := T - min_total_for_39_students

-- The statement to prove.
theorem max_candies_theorem : max_candies_one_student_can_take = 122 :=
by
  sorry

end max_candies_theorem_l117_11741


namespace smallest_n_equal_sums_l117_11745

def sum_first_n_arithmetic (a d n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

theorem smallest_n_equal_sums : ∀ (n : ℕ), 
  sum_first_n_arithmetic 7 4 n = sum_first_n_arithmetic 15 3 n → n ≠ 0 → n = 7 := by
  intros n h1 h2
  sorry

end smallest_n_equal_sums_l117_11745


namespace derivative_of_y_is_correct_l117_11769

noncomputable def y (x : ℝ) := x^2 * Real.sin x

theorem derivative_of_y_is_correct : (deriv y x = 2 * x * Real.sin x + x^2 * Real.cos x) :=
by
  sorry

end derivative_of_y_is_correct_l117_11769


namespace math_problem_l117_11778

-- Arithmetic sequence {a_n}
def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  a 2 = 8 ∧ a 3 + a 5 = 4 * a 2

-- General term of the arithmetic sequence {a_n}
def general_term (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a n = 4 * n

-- Geometric sequence {b_n}
def geometric_sequence (b : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  b 4 = a 1 ∧ b 6 = a 4

-- The sum S_n of the first n terms of the sequence {b_n - a_n}
def sum_sequence (b : ℕ → ℝ) (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = (2 ^ (n - 1) - 1 / 2 - 2 * n ^ 2 - 2 * n)

-- Full proof statement
theorem math_problem (a : ℕ → ℕ) (b : ℕ → ℝ) (S : ℕ → ℝ) :
  arithmetic_sequence a →
  general_term a →
  ∀ a_n : ℕ → ℝ, a_n 1 = 4 ∧ a_n 4 = 16 →
  geometric_sequence b a_n →
  sum_sequence b a_n S :=
by
  intros h_arith_seq h_gen_term h_a_n h_geom_seq
  sorry

end math_problem_l117_11778


namespace consecutive_even_numbers_l117_11785

theorem consecutive_even_numbers (n m : ℕ) (h : 52 * (2 * n - 1) = 100 * n) : n = 13 :=
by
  sorry

end consecutive_even_numbers_l117_11785


namespace geometric_seq_a5_value_l117_11798

theorem geometric_seq_a5_value 
  (a : ℕ → ℝ) (q : ℝ) 
  (h_seq : ∀ n : ℕ, a (n+1) = a n * q)
  (h_pos : ∀ n : ℕ, a n > 0) 
  (h1 : a 1 * a 8 = 4 * a 5)
  (h2 : (a 4 + 2 * a 6) / 2 = 18) 
  : a 5 = 16 := 
sorry

end geometric_seq_a5_value_l117_11798


namespace range_a_monotonically_increasing_l117_11799

def g (a x : ℝ) : ℝ := a * x^3 + a * x^2 + x

theorem range_a_monotonically_increasing (a : ℝ) : 
  (∀ x : ℝ, 3 * a * x^2 + 2 * a * x + 1 ≥ 0) ↔ (0 ≤ a ∧ a ≤ 3) := 
sorry

end range_a_monotonically_increasing_l117_11799


namespace value_of_fourth_set_l117_11714

def value_in_set (a b c d : ℕ) : ℕ :=
  (a * b * c * d) - (a + b + c + d)

theorem value_of_fourth_set :
  value_in_set 1 5 6 7 = 191 :=
by
  sorry

end value_of_fourth_set_l117_11714


namespace arithmetic_sequence_n_value_l117_11713

noncomputable def common_ratio (a₁ S₃ : ℕ) : ℕ := by sorry

theorem arithmetic_sequence_n_value:
  ∀ (a : ℕ → ℕ) (S : ℕ → ℕ),
  (∀ n, a n > 0) →
  a 1 = 3 →
  S 3 = 21 →
  (∃ q, q > 0 ∧ common_ratio 1 q = q ∧ a 5 = 48) →
  n = 5 :=
by
  intros
  sorry

end arithmetic_sequence_n_value_l117_11713


namespace cylinder_volume_l117_11750

theorem cylinder_volume (r h : ℝ) (radius_is_2 : r = 2) (height_is_3 : h = 3) :
  π * r^2 * h = 12 * π :=
by
  rw [radius_is_2, height_is_3]
  sorry

end cylinder_volume_l117_11750


namespace total_cost_correct_l117_11728

-- Define the costs for each day
def day1_rate : ℝ := 150
def day1_miles_cost : ℝ := 0.50 * 620
def gps_service_cost : ℝ := 10
def day1_total_cost : ℝ := day1_rate + day1_miles_cost + gps_service_cost

def day2_rate : ℝ := 100
def day2_miles_cost : ℝ := 0.40 * 744
def day2_total_cost : ℝ := day2_rate + day2_miles_cost + gps_service_cost

def day3_rate : ℝ := 75
def day3_miles_cost : ℝ := 0.30 * 510
def day3_total_cost : ℝ := day3_rate + day3_miles_cost + gps_service_cost

-- Define the total cost
def total_cost : ℝ := day1_total_cost + day2_total_cost + day3_total_cost

-- Prove that the total cost is equal to the calculated value
theorem total_cost_correct : total_cost = 1115.60 :=
by
  -- This is where the proof would go, but we leave it out for now
  sorry

end total_cost_correct_l117_11728


namespace total_area_rectABCD_l117_11794

theorem total_area_rectABCD (BF CF : ℝ) (X Y : ℝ)
  (h1 : BF = 3 * CF)
  (h2 : 3 * X - Y - (X - Y) = 96)
  (h3 : X + 3 * X = 192) :
  X + 3 * X = 192 :=
by
  sorry

end total_area_rectABCD_l117_11794


namespace directrix_of_parabola_l117_11758

theorem directrix_of_parabola : 
  (∀ (y x: ℝ), y^2 = 12 * x → x = -3) :=
sorry

end directrix_of_parabola_l117_11758


namespace roots_of_polynomial_l117_11702

theorem roots_of_polynomial :
  {r : ℝ | (10 * r^4 - 55 * r^3 + 96 * r^2 - 55 * r + 10 = 0)} = {2, 1, 1 / 2} :=
sorry

end roots_of_polynomial_l117_11702


namespace julia_tulip_count_l117_11731

def tulip_count (tulips daisies : ℕ) : Prop :=
  3 * daisies = 7 * tulips

theorem julia_tulip_count : 
  ∃ t, tulip_count t 65 ∧ t = 28 := 
by
  sorry

end julia_tulip_count_l117_11731


namespace find_x_perpendicular_l117_11760

-- Definitions used in the conditions
def a : ℝ × ℝ := (3, 2)
def b (x : ℝ) : ℝ × ℝ := (x, 4)

-- Condition: vectors a and b are perpendicular
def perpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

-- The theorem we want to prove
theorem find_x_perpendicular : ∀ x : ℝ, perpendicular a (b x) → x = -8 / 3 :=
by
  intros x h
  sorry

end find_x_perpendicular_l117_11760


namespace overall_ratio_men_women_l117_11723

variables (m_w_diff players_total beginners_m beginners_w intermediate_m intermediate_w advanced_m advanced_w : ℕ)

def total_men : ℕ := beginners_m + intermediate_m + advanced_m
def total_women : ℕ := beginners_w + intermediate_w + advanced_w

theorem overall_ratio_men_women 
  (h1 : beginners_m = 2) 
  (h2 : beginners_w = 4)
  (h3 : intermediate_m = 3) 
  (h4 : intermediate_w = 5) 
  (h5 : advanced_m = 1) 
  (h6 : advanced_w = 3) 
  (h7 : m_w_diff = 4)
  (h8 : total_men = 6)
  (h9 : total_women = 12)
  (h10 : players_total = 18) :
  total_men / total_women = 1 / 2 :=
by {
  sorry
}

end overall_ratio_men_women_l117_11723


namespace correct_option_is_D_l117_11768

-- Define the expressions to be checked
def exprA (x : ℝ) := 3 * x + 2 * x = 5 * x^2
def exprB (x : ℝ) := -2 * x^2 * x^3 = 2 * x^5
def exprC (x y : ℝ) := (y + 3 * x) * (3 * x - y) = y^2 - 9 * x^2
def exprD (x y : ℝ) := (-2 * x^2 * y)^3 = -8 * x^6 * y^3

theorem correct_option_is_D (x y : ℝ) : 
  ¬ exprA x ∧ 
  ¬ exprB x ∧ 
  ¬ exprC x y ∧ 
  exprD x y := by
  -- The proof would be provided here
  sorry

end correct_option_is_D_l117_11768


namespace range_of_b_l117_11705

theorem range_of_b (b : ℝ) :
  (∃ (x y : ℝ), 
    0 ≤ x ∧ x ≤ 4 ∧ 1 ≤ y ∧ y ≤ 3 ∧ 
    y = x + b ∧ (x - 2)^2 + (y - 3)^2 = 4) ↔ 
    1 - 2 * Real.sqrt 2 ≤ b ∧ b ≤ 3 :=
by
  sorry

end range_of_b_l117_11705


namespace max_lattice_points_in_unit_circle_l117_11746

-- Define a point with integer coordinates
structure LatticePoint :=
  (x : ℤ)
  (y : ℤ)

-- Define the condition for a lattice point to be strictly inside a given circle
def strictly_inside_circle (p : LatticePoint) (center : Prod ℤ ℤ) (r : ℝ) : Prop :=
  let dx := (p.x - center.fst : ℝ)
  let dy := (p.y - center.snd : ℝ)
  dx^2 + dy^2 < r^2

-- Define the problem statement
theorem max_lattice_points_in_unit_circle : ∀ (center : Prod ℤ ℤ) (r : ℝ),
  r = 1 → 
  ∃ (ps : Finset LatticePoint), 
    (∀ p ∈ ps, strictly_inside_circle p center r) ∧ 
    ps.card = 4 :=
by
  sorry

end max_lattice_points_in_unit_circle_l117_11746


namespace sum_of_three_largest_of_consecutive_numbers_l117_11790

theorem sum_of_three_largest_of_consecutive_numbers (n : ℕ) :
  (n + (n + 1) + (n + 2) = 60) → ((n + 2) + (n + 3) + (n + 4) = 66) :=
by
  -- Given the conditions and expected result, we can break down the proof as follows:
  intros h1
  sorry

end sum_of_three_largest_of_consecutive_numbers_l117_11790


namespace angle_bounds_find_configurations_l117_11706

/-- Given four points A, B, C, D on a plane, where α1 and α2 are the two smallest angles,
    and β1 and β2 are the two largest angles formed by these points, we aim to prove:
    1. 0 ≤ α2 ≤ 45 degrees,
    2. 72 degrees ≤ β2 ≤ 180 degrees,
    and to find configurations that achieve α2 = 45 degrees and β2 = 72 degrees. -/
theorem angle_bounds {A B C D : ℝ × ℝ} (α1 α2 β1 β2 : ℝ) 
  (h_angles : α1 ≤ α2 ∧ α2 ≤ β2 ∧ β2 ≤ β1 ∧ 
              0 ≤ α2 ∧ α2 ≤ 45 ∧ 
              72 ≤ β2 ∧ β2 ≤ 180) : 
  (0 ≤ α2 ∧ α2 ≤ 45 ∧ 72 ≤ β2 ∧ β2 ≤ 180) := 
by sorry

/-- Find configurations where α2 = 45 degrees and β2 = 72 degrees. -/
theorem find_configurations {A B C D : ℝ × ℝ} (α1 α2 β1 β2 : ℝ)
  (h_angles : α1 ≤ α2 ∧ α2 = 45 ∧ β2 = 72 ∧ β2 ≤ β1) :
  (α2 = 45 ∧ β2 = 72) := 
by sorry

end angle_bounds_find_configurations_l117_11706


namespace mike_total_cards_l117_11787

-- Given conditions
def mike_original_cards : ℕ := 87
def sam_given_cards : ℕ := 13

-- Question equivalence in Lean: Prove that Mike has 100 baseball cards now
theorem mike_total_cards : mike_original_cards + sam_given_cards = 100 :=
by 
  sorry

end mike_total_cards_l117_11787


namespace total_copies_to_save_40_each_l117_11710

-- Definitions for the conditions.
def cost_per_copy : ℝ := 0.02
def discount_rate : ℝ := 0.25
def min_copies_for_discount : ℕ := 100
def savings_required : ℝ := 0.40
def steve_copies : ℕ := 80
def dinley_copies : ℕ := 80

-- Lean 4 statement to prove the total number of copies 
-- to save $0.40 each.
theorem total_copies_to_save_40_each : 
  (steve_copies + dinley_copies) + 
  (savings_required / (cost_per_copy * discount_rate)) * 2 = 320 :=
by 
  sorry

end total_copies_to_save_40_each_l117_11710


namespace inconsistent_intercepts_l117_11781

-- Define the ellipse equation
def ellipse (x y : ℝ) (m : ℝ) : Prop :=
  x^2 / m + y^2 / 4 = 1

-- Define the line equations
def line1 (x k : ℝ) : ℝ := k * x + 1
def line2 (x : ℝ) (k : ℝ) : ℝ := - k * x - 2

-- Disc calculation for line1
def disc1 (m k : ℝ) : ℝ :=
  let a := 4 + m * k^2
  let b := 2 * m * k
  let c := -3 * m
  b^2 - 4 * a * c

-- Disc calculation for line2
def disc2 (m k : ℝ) : ℝ :=
  let bb := 4 * m * k
  bb^2

-- Statement of the problem
theorem inconsistent_intercepts (m k : ℝ) (hm_pos : 0 < m) :
  disc1 m k ≠ disc2 m k :=
by
  sorry

end inconsistent_intercepts_l117_11781


namespace only_integer_solution_is_trivial_l117_11755

theorem only_integer_solution_is_trivial (a b c : ℤ) (h : 5 * a^2 + 9 * b^2 = 13 * c^2) : a = 0 ∧ b = 0 ∧ c = 0 :=
by
  sorry

end only_integer_solution_is_trivial_l117_11755


namespace find_f_zero_l117_11761

variable (f : ℝ → ℝ)
variable (hf : ∀ x y : ℝ, f (x + y) = f x + f y + 1 / 2)

theorem find_f_zero : f 0 = -1 / 2 :=
by
  sorry

end find_f_zero_l117_11761


namespace avg_speed_trip_l117_11704

noncomputable def distance_travelled (speed time : ℕ) : ℕ := speed * time

noncomputable def average_speed (total_distance total_time : ℕ) : ℕ := total_distance / total_time

theorem avg_speed_trip :
  let first_leg_speed := 75
  let first_leg_time := 4
  let second_leg_speed := 60
  let second_leg_time := 2
  let total_time := first_leg_time + second_leg_time
  let first_leg_distance := distance_travelled first_leg_speed first_leg_time
  let second_leg_distance := distance_travelled second_leg_speed second_leg_time
  let total_distance := first_leg_distance + second_leg_distance
  average_speed total_distance total_time = 70 :=
by
  sorry

end avg_speed_trip_l117_11704


namespace average_velocity_of_particle_l117_11717

theorem average_velocity_of_particle (t : ℝ) (s : ℝ → ℝ) (h_s : ∀ t, s t = t^2 + 1) :
  (s 2 - s 1) / (2 - 1) = 3 :=
by {
  sorry
}

end average_velocity_of_particle_l117_11717


namespace integral_evaluation_l117_11727

noncomputable def integral_result : ℝ :=
  ∫ x in (0:ℝ)..(1:ℝ), (Real.sqrt (1 - x^2) - x)

theorem integral_evaluation :
  integral_result = (Real.pi - 2) / 4 :=
by
  sorry

end integral_evaluation_l117_11727


namespace first_term_of_geometric_series_l117_11733

-- Define the conditions and the question
theorem first_term_of_geometric_series (a r : ℝ) (h1 : a / (1 - r) = 30) (h2 : a^2 / (1 - r^2) = 180) : a = 10 :=
by sorry

end first_term_of_geometric_series_l117_11733


namespace minimum_value_of_expression_l117_11780

theorem minimum_value_of_expression (a b : ℝ) (h1 : a + 2 * b = 1) (h2 : a * b > 0) : 
  (1 / a) + (2 / b) = 5 :=
sorry

end minimum_value_of_expression_l117_11780


namespace largest_possible_d_l117_11788

theorem largest_possible_d (a b c d : ℝ) (h1 : a + b + c + d = 10) (h2 : ab + ac + ad + bc + bd + cd = 20) : 
    d ≤ (5 + Real.sqrt 105) / 2 :=
by
  sorry

end largest_possible_d_l117_11788


namespace ian_money_left_l117_11757

theorem ian_money_left
  (hours_worked : ℕ)
  (hourly_rate : ℕ)
  (spending_percentage : ℚ)
  (total_earnings : ℕ)
  (amount_spent : ℕ)
  (amount_left : ℕ)
  (h_worked : hours_worked = 8)
  (h_rate : hourly_rate = 18)
  (h_spending : spending_percentage = 0.5)
  (h_earnings : total_earnings = hours_worked * hourly_rate)
  (h_spent : amount_spent = total_earnings * spending_percentage)
  (h_left : amount_left = total_earnings - amount_spent) :
  amount_left = 72 := 
  sorry

end ian_money_left_l117_11757


namespace more_orange_pages_read_l117_11795

-- Define the conditions
def purple_pages_per_book : Nat := 230
def orange_pages_per_book : Nat := 510
def purple_books_read : Nat := 5
def orange_books_read : Nat := 4

-- Calculate the total pages read from purple and orange books respectively
def total_purple_pages_read : Nat := purple_pages_per_book * purple_books_read
def total_orange_pages_read : Nat := orange_pages_per_book * orange_books_read

-- State the theorem to be proved
theorem more_orange_pages_read : total_orange_pages_read - total_purple_pages_read = 890 :=
by
  -- This is where the proof steps would go, but we'll leave it as sorry to indicate the proof is not provided
  sorry

end more_orange_pages_read_l117_11795


namespace hexagon_AF_length_l117_11721

theorem hexagon_AF_length (BC CD DE EF : ℝ) (angleB angleC angleD angleE : ℝ) (angleF : ℝ) 
  (hBC : BC = 2) (hCD : CD = 2) (hDE : DE = 2) (hEF : EF = 2)
  (hangleB : angleB = 135) (hangleC : angleC = 135) (hangleD : angleD = 135) (hangleE : angleE = 135)
  (hangleF : angleF = 90) :
  ∃ (a b : ℝ), (AF = a + 2 * Real.sqrt b) ∧ (a + b = 6) :=
by
  sorry

end hexagon_AF_length_l117_11721


namespace alt_rep_of_set_l117_11772

def NatPos (x : ℕ) := x > 0

theorem alt_rep_of_set : {x : ℕ | NatPos x ∧ x - 3 < 2} = {1, 2, 3, 4} := by
  sorry

end alt_rep_of_set_l117_11772


namespace calc_sqrt_expr_l117_11740

theorem calc_sqrt_expr : (Real.sqrt 2 + 1) ^ 2 - Real.sqrt 18 + 2 * Real.sqrt (1 / 2) = 3 := by
  sorry

end calc_sqrt_expr_l117_11740


namespace exists_n_for_sin_l117_11730

theorem exists_n_for_sin (x : ℝ) (h : Real.sin x ≠ 0) :
  ∃ n : ℕ, |Real.sin (n * x)| ≥ Real.sqrt 3 / 2 :=
sorry

end exists_n_for_sin_l117_11730


namespace total_balls_l117_11771

theorem total_balls (S V B Total : ℕ) (hS : S = 68) (hV : S = V - 12) (hB : S = B + 23) : 
  Total = S + V + B := by
  sorry

end total_balls_l117_11771


namespace fraction_of_second_year_students_l117_11783

-- Define the fractions of first-year and second-year students
variables (F S f s: ℝ)

-- Conditions
axiom h1 : F + S = 1
axiom h2 : f = (1 / 5) * F
axiom h3 : s = 4 * f
axiom h4 : S - s = 0.2

-- The theorem statement to prove the fraction of second-year students is 2 / 3
theorem fraction_of_second_year_students (F S f s: ℝ) 
    (h1: F + S = 1) 
    (h2: f = (1 / 5) * F) 
    (h3: s = 4 * f) 
    (h4: S - s = 0.2) : 
    S = 2 / 3 :=
by 
    sorry

end fraction_of_second_year_students_l117_11783


namespace additional_vegetables_can_be_planted_l117_11715

-- Defines the garden's initial conditions.
def tomatoes_kinds := 3
def tomatoes_each := 5
def cucumbers_kinds := 5
def cucumbers_each := 4
def potatoes := 30
def rows := 10
def spaces_per_row := 15

-- The proof statement.
theorem additional_vegetables_can_be_planted (total_tomatoes : ℕ := tomatoes_kinds * tomatoes_each)
                                              (total_cucumbers : ℕ := cucumbers_kinds * cucumbers_each)
                                              (total_potatoes : ℕ := potatoes)
                                              (total_spaces : ℕ := rows * spaces_per_row) :
  total_spaces - (total_tomatoes + total_cucumbers + total_potatoes) = 85 := 
by 
  sorry

end additional_vegetables_can_be_planted_l117_11715


namespace max_difference_intersection_ycoords_l117_11766

theorem max_difference_intersection_ycoords :
  let f₁ (x : ℝ) := 5 - 2 * x^2 + x^3
  let f₂ (x : ℝ) := 1 + x^2 + x^3
  let x1 := (2 : ℝ) / Real.sqrt 3
  let x2 := - (2 : ℝ) / Real.sqrt 3
  let y1 := f₁ x1
  let y2 := f₂ x2
  (f₁ = f₂)
  → abs (y1 - y2) = (16 * Real.sqrt 3 / 9) :=
by
  sorry

end max_difference_intersection_ycoords_l117_11766


namespace combined_rainfall_is_23_l117_11737

-- Define the conditions
def monday_hours : ℕ := 7
def monday_rate : ℕ := 1
def tuesday_hours : ℕ := 4
def tuesday_rate : ℕ := 2
def wednesday_hours : ℕ := 2
def wednesday_rate (tuesday_rate : ℕ) : ℕ := 2 * tuesday_rate

-- Calculate the rainfalls
def monday_rainfall : ℕ := monday_hours * monday_rate
def tuesday_rainfall : ℕ := tuesday_hours * tuesday_rate
def wednesday_rainfall (wednesday_rate : ℕ) : ℕ := wednesday_hours * wednesday_rate

-- Define the total rainfall
def total_rainfall : ℕ :=
  monday_rainfall + tuesday_rainfall + wednesday_rainfall (wednesday_rate tuesday_rate)

theorem combined_rainfall_is_23 : total_rainfall = 23 := by
  -- Proof to be filled in
  sorry

end combined_rainfall_is_23_l117_11737


namespace decimal_properties_l117_11784

theorem decimal_properties :
  (3.00 : ℝ) = (3 : ℝ) :=
by sorry

end decimal_properties_l117_11784


namespace apple_trees_count_l117_11756

-- Conditions
def num_peach_trees : ℕ := 45
def kg_per_peach_tree : ℕ := 65
def total_mass_fruit : ℕ := 7425
def kg_per_apple_tree : ℕ := 150
variable (A : ℕ)

-- Proof goal
theorem apple_trees_count (h : A * kg_per_apple_tree + num_peach_trees * kg_per_peach_tree = total_mass_fruit) : A = 30 := 
sorry

end apple_trees_count_l117_11756


namespace smallest_odd_factors_gt_100_l117_11725

theorem smallest_odd_factors_gt_100 : ∃ n : ℕ, n > 100 ∧ (∀ d : ℕ, d ∣ n → (∃ m : ℕ, n = m * m)) ∧ (∀ m : ℕ, m > 100 ∧ (∀ d : ℕ, d ∣ m → (∃ k : ℕ, m = k * k)) → n ≤ m) :=
by
  sorry

end smallest_odd_factors_gt_100_l117_11725


namespace jake_time_to_row_lake_l117_11782

noncomputable def time_to_row_lake (side_length miles_per_side : ℝ) (swim_time_per_mile minutes_per_mile : ℝ) : ℝ :=
  let swim_speed := 60 / swim_time_per_mile -- miles per hour
  let row_speed := 2 * swim_speed          -- miles per hour
  let total_distance := 4 * side_length    -- miles
  total_distance / row_speed               -- hours

theorem jake_time_to_row_lake :
  time_to_row_lake 15 20 = 10 := sorry

end jake_time_to_row_lake_l117_11782


namespace find_b3_b17_l117_11722

variable {a : ℕ → ℤ} -- Arithmetic sequence
variable {b : ℕ → ℤ} -- Geometric sequence

axiom arith_seq {a : ℕ → ℤ} (d : ℤ) : ∀ (n : ℕ), a (n + 1) = a n + d
axiom geom_seq {b : ℕ → ℤ} (r : ℤ) : ∀ (n : ℕ), b (n + 1) = b n * r

theorem find_b3_b17 
  (h_arith : ∃ d, ∀ n, a (n + 1) = a n + d) 
  (h_geom : ∃ r, ∀ n, b (n + 1) = b n * r)
  (h_cond1 : 3 * a 1 - (a 8)^2 + 3 * a 15 = 0)
  (h_cond2 : a 8 = b 10) :
  b 3 * b 17 = 36 := 
sorry

end find_b3_b17_l117_11722


namespace min_heaviest_weight_l117_11744

theorem min_heaviest_weight : 
  ∃ (w : ℕ), ∀ (weights : Fin 8 → ℕ),
    (∀ i j, i ≠ j → weights i ≠ weights j) ∧
    (∀ (a b c d : Fin 8),
      a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d → 
      (weights a + weights b) ≠ (weights c + weights d) ∧ 
      max (max (weights a) (weights b)) (max (weights c) (weights d)) >= w) 
  → w = 34 := 
by
  sorry

end min_heaviest_weight_l117_11744


namespace solve_for_y_l117_11786

theorem solve_for_y (y : ℝ) (h : 6 * y^(1/4) - 3 * (y / y^(3/4)) = 12 + y^(1/4)) : y = 1296 := by
  sorry

end solve_for_y_l117_11786


namespace max_value_of_a2b3c2_l117_11796

theorem max_value_of_a2b3c2 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h_sum : a + b + c = 1) :
  a^2 * b^3 * c^2 ≤ 81 / 262144 :=
sorry

end max_value_of_a2b3c2_l117_11796


namespace blocks_differs_in_exactly_two_ways_correct_l117_11763

structure Block where
  material : Bool       -- material: false for plastic, true for wood
  size : Fin 3          -- sizes: 0 for small, 1 for medium, 2 for large
  color : Fin 4         -- colors: 0 for blue, 1 for green, 2 for red, 3 for yellow
  shape : Fin 4         -- shapes: 0 for circle, 1 for hexagon, 2 for square, 3 for triangle
  finish : Bool         -- finish: false for glossy, true for matte

def originalBlock : Block :=
  { material := false, size := 1, color := 2, shape := 0, finish := false }

def differsInExactlyTwoWays (b1 b2 : Block) : Bool :=
  (if b1.material ≠ b2.material then 1 else 0) +
  (if b1.size ≠ b2.size then 1 else 0) +
  (if b1.color ≠ b2.color then 1 else 0) +
  (if b1.shape ≠ b2.shape then 1 else 0) +
  (if b1.finish ≠ b2.finish then 1 else 0) == 2

def countBlocksDifferingInTwoWays : Nat :=
  let allBlocks := List.product
                  (List.product
                    (List.product
                      (List.product
                        [false, true]
                        ([0, 1, 2] : List (Fin 3)))
                      ([0, 1, 2, 3] : List (Fin 4)))
                    ([0, 1, 2, 3] : List (Fin 4)))
                  [false, true]
  (allBlocks.filter
    (λ b => differsInExactlyTwoWays originalBlock
                { material := b.1.1.1.1, size := b.1.1.1.2, color := b.1.1.2, shape := b.1.2, finish := b.2 })).length

theorem blocks_differs_in_exactly_two_ways_correct :
  countBlocksDifferingInTwoWays = 51 :=
  by
    sorry

end blocks_differs_in_exactly_two_ways_correct_l117_11763


namespace value_of_f_at_2_l117_11752

def f (x : ℝ) : ℝ := x^3 - x^2 - 1

theorem value_of_f_at_2 : f 2 = 3 := by
  sorry

end value_of_f_at_2_l117_11752


namespace min_AP_BP_l117_11762

-- Definitions based on conditions in the problem
def A : ℝ × ℝ := (2, 0)
def B : ℝ × ℝ := (7, 6)
def parabola (x y : ℝ) : Prop := y^2 = 8 * x

-- The theorem to prove the minimum value of AP + BP
theorem min_AP_BP
  (P : ℝ × ℝ)
  (hP_parabola : parabola P.1 P.2) :
  dist P A + dist P B ≥ 9 :=
sorry

end min_AP_BP_l117_11762


namespace area_of_white_square_l117_11793

theorem area_of_white_square
  (face_area : ℕ)
  (total_surface_area : ℕ)
  (blue_paint_area : ℕ)
  (faces : ℕ)
  (area_of_white_square : ℕ) :
  face_area = 12 * 12 →
  total_surface_area = 6 * face_area →
  blue_paint_area = 432 →
  faces = 6 →
  area_of_white_square = face_area - (blue_paint_area / faces) →
  area_of_white_square = 72 :=
by
  sorry

end area_of_white_square_l117_11793


namespace max_abs_value_of_quadratic_function_l117_11774

def quadratic_function (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

def point_in_band_region (y k l : ℝ) : Prop := k ≤ y ∧ y ≤ l

theorem max_abs_value_of_quadratic_function (a b c t : ℝ) (h1 : point_in_band_region (quadratic_function a b c (-2) + 2) 0 4)
                                             (h2 : point_in_band_region (quadratic_function a b c 0 + 2) 0 4)
                                             (h3 : point_in_band_region (quadratic_function a b c 2 + 2) 0 4)
                                             (h4 : point_in_band_region (t + 1) (-1) 3) :
  |quadratic_function a b c t| ≤ 5 / 2 :=
sorry

end max_abs_value_of_quadratic_function_l117_11774


namespace probability_top_king_of_hearts_l117_11738

def deck_size : ℕ := 52

def king_of_hearts_count : ℕ := 1

def probability_king_of_hearts_top_card (n : ℕ) (k : ℕ) : ℚ :=
  if n ≠ 0 then k / n else 0

theorem probability_top_king_of_hearts : 
  probability_king_of_hearts_top_card deck_size king_of_hearts_count = 1 / 52 :=
by
  -- Proof omitted
  sorry

end probability_top_king_of_hearts_l117_11738


namespace wrongly_noted_mark_is_90_l117_11708

-- Define the given conditions
def avg_marks (n : ℕ) (avg : ℚ) : ℚ := n * avg

def wrong_avg_marks : ℚ := avg_marks 10 100
def correct_avg_marks : ℚ := avg_marks 10 92

-- Equate the difference caused by the wrong mark
theorem wrongly_noted_mark_is_90 (x : ℚ) (h₁ : wrong_avg_marks = 1000) (h₂ : correct_avg_marks = 920) (h : x - 10 = 1000 - 920) : x = 90 := 
by {
  -- Proof goes here
  sorry
}

end wrongly_noted_mark_is_90_l117_11708


namespace problem_solution_l117_11701

noncomputable def expression_value : ℝ :=
  ((12.983 * 26) / 200) ^ 3 * Real.log 5 / Real.log 10

theorem problem_solution : expression_value = 3.361 := by
  sorry

end problem_solution_l117_11701


namespace females_over_30_prefer_webstream_l117_11767

-- Define the total number of survey participants
def total_participants : ℕ := 420

-- Define the number of participants who prefer WebStream
def prefer_webstream : ℕ := 200

-- Define the number of participants who do not prefer WebStream
def not_prefer_webstream : ℕ := 220

-- Define the number of males who prefer WebStream
def males_prefer : ℕ := 80

-- Define the number of females under 30 who do not prefer WebStream
def females_under_30_not_prefer : ℕ := 90

-- Define the number of females over 30 who do not prefer WebStream
def females_over_30_not_prefer : ℕ := 70

-- Define the total number of females under 30 who do not prefer WebStream
def females_not_prefer : ℕ := females_under_30_not_prefer + females_over_30_not_prefer

-- Define the total number of participants who do not prefer WebStream
def total_not_prefer : ℕ := 220

-- Define the number of males who do not prefer WebStream
def males_not_prefer : ℕ := total_not_prefer - females_not_prefer

-- Define the number of females who prefer WebStream
def females_prefer : ℕ := prefer_webstream - males_prefer

-- Define the total number of females under 30 who prefer WebStream
def females_under_30_prefer : ℕ := total_participants - prefer_webstream - females_under_30_not_prefer

-- Define the remaining females over 30 who prefer WebStream
def females_over_30_prefer : ℕ := females_prefer - females_under_30_prefer

-- The Lean statement to prove
theorem females_over_30_prefer_webstream : females_over_30_prefer = 110 := by
  sorry

end females_over_30_prefer_webstream_l117_11767


namespace second_discount_percentage_is_20_l117_11749

theorem second_discount_percentage_is_20 
    (normal_price : ℝ)
    (final_price : ℝ)
    (first_discount : ℝ)
    (first_discount_percentage : ℝ)
    (h1 : normal_price = 149.99999999999997)
    (h2 : final_price = 108)
    (h3 : first_discount_percentage = 10)
    (h4 : first_discount = normal_price * (first_discount_percentage / 100)) :
    (((normal_price - first_discount) - final_price) / (normal_price - first_discount)) * 100 = 20 := by
  sorry

end second_discount_percentage_is_20_l117_11749


namespace greatest_groups_of_stuffed_animals_l117_11703

def stuffed_animals_grouping : Prop :=
  let cats := 26
  let dogs := 14
  let bears := 18
  let giraffes := 22
  gcd (gcd (gcd cats dogs) bears) giraffes = 2

theorem greatest_groups_of_stuffed_animals : stuffed_animals_grouping :=
by sorry

end greatest_groups_of_stuffed_animals_l117_11703


namespace exists_k_for_any_n_l117_11732

theorem exists_k_for_any_n (n : ℕ) (hn : n > 0) : 
  ∃ k : ℕ, 2 * k^2 + 2001 * k + 3 ≡ 0 [MOD 2^n] :=
sorry

end exists_k_for_any_n_l117_11732


namespace tomorrowIsUncertain_l117_11736

-- Definitions as conditions
def isCertainEvent (e : Prop) : Prop := e = true
def isImpossibleEvent (e : Prop) : Prop := e = false
def isInevitableEvent (e : Prop) : Prop := e = true
def isUncertainEvent (e : Prop) : Prop := e ≠ true ∧ e ≠ false

-- Event: Tomorrow will be sunny
def tomorrowWillBeSunny : Prop := sorry -- Placeholder for the actual weather prediction model

-- Problem statement: Prove that "Tomorrow will be sunny" is an uncertain event
theorem tomorrowIsUncertain : isUncertainEvent tomorrowWillBeSunny := sorry

end tomorrowIsUncertain_l117_11736


namespace mashed_potatoes_count_l117_11743

theorem mashed_potatoes_count :
  ∀ (b s : ℕ), b = 489 → b = s + 10 → s = 479 :=
by
  intros b s h₁ h₂
  sorry

end mashed_potatoes_count_l117_11743


namespace airline_flights_increase_l117_11709

theorem airline_flights_increase (n k : ℕ) 
  (h : (n + k) * (n + k - 1) / 2 - n * (n - 1) / 2 = 76) :
  (n = 6 ∧ n + k = 14) ∨ (n = 76 ∧ n + k = 77) :=
by
  sorry

end airline_flights_increase_l117_11709


namespace correct_assignment_statements_l117_11748

-- Defining what constitutes an assignment statement in this context.
def is_assignment_statement (s : String) : Prop :=
  s ∈ ["x ← 1", "y ← 2", "z ← 3", "i ← i + 2"]

-- Given statements
def statements : List String :=
  ["x ← 1, y ← 2, z ← 3", "S^2 ← 4", "i ← i + 2", "x + 1 ← x"]

-- The Lean Theorem statement that these are correct assignment statements.
theorem correct_assignment_statements (s₁ s₃ : String) (h₁ : s₁ = "x ← 1, y ← 2, z ← 3") (h₃ : s₃ = "i ← i + 2") :
  is_assignment_statement s₁ ∧ is_assignment_statement s₃ :=
by
  sorry

end correct_assignment_statements_l117_11748


namespace solve_equation_l117_11764

theorem solve_equation : ∃ x : ℝ, (2 * x - 1) / 3 - (x - 2) / 6 = 2 ∧ x = 4 :=
by
  sorry

end solve_equation_l117_11764


namespace find_xy_l117_11779

theorem find_xy (x y : ℝ) (h : (x^2 + 6 * x + 12) * (5 * y^2 + 2 * y + 1) = 12 / 5) : 
    x * y = 3 / 5 :=
sorry

end find_xy_l117_11779


namespace power_mod_equiv_l117_11789

-- Define the main theorem
theorem power_mod_equiv {a n k : ℕ} (h₁ : a ≥ 2) (h₂ : n ≥ 1) :
  (a^k ≡ 1 [MOD (a^n - 1)]) ↔ (k % n = 0) :=
by sorry

end power_mod_equiv_l117_11789


namespace gcd_poly_l117_11724

-- Defining the conditions
def is_odd_multiple_of_17 (b : ℤ) : Prop := ∃ k : ℤ, b = 17 * (2 * k + 1)

theorem gcd_poly (b : ℤ) (h : is_odd_multiple_of_17 b) : 
  Int.gcd (12 * b^3 + 7 * b^2 + 49 * b + 106) 
          (3 * b + 7) = 1 :=
by sorry

end gcd_poly_l117_11724


namespace problem_1_problem_2_l117_11729

def f (x : ℝ) : ℝ := x^2 - 4 * x + 6

theorem problem_1 (m : ℝ) (h_mono : ∀ x y, m ≤ x → x ≤ y → y ≤ m + 1 → f y ≤ f x) : m ≤ 1 :=
  sorry

theorem problem_2 (a b : ℝ) (h_min : a < b) 
  (h_min_val : ∀ x, a ≤ x ∧ x ≤ b → f a ≤ f x)
  (h_max_val : ∀ x, a ≤ x ∧ x ≤ b → f x ≤ f b) 
  (h_fa_eq_a : f a = a) (h_fb_eq_b : f b = b) : a = 2 ∧ b = 3 :=
  sorry

end problem_1_problem_2_l117_11729


namespace odd_number_as_diff_of_squares_l117_11776

theorem odd_number_as_diff_of_squares (n : ℤ) : ∃ a b : ℤ, a^2 - b^2 = 2 * n + 1 :=
by
  use (n + 1), n
  sorry

end odd_number_as_diff_of_squares_l117_11776


namespace smallest_sum_Q_lt_7_9_l117_11739

def Q (N k : ℕ) : ℚ := (N + 1) / (N + k + 1)

theorem smallest_sum_Q_lt_7_9 : 
    ∃ N k : ℕ, (N + k) % 4 = 0 ∧ Q N k < 7 / 9 ∧ (∀ N' k' : ℕ, (N' + k') % 4 = 0 ∧ Q N' k' < 7 / 9 → N' + k' ≥ N + k) ∧ N + k = 4 :=
by
  sorry

end smallest_sum_Q_lt_7_9_l117_11739


namespace find_a_l117_11719

-- Given conditions
def div_by_3 (a : ℤ) : Prop :=
  (5 * a + 1) % 3 = 0 ∨ (3 * a + 2) % 3 = 0

def div_by_5 (a : ℤ) : Prop :=
  (5 * a + 1) % 5 = 0 ∨ (3 * a + 2) % 5 = 0

-- Proving the question 
theorem find_a (a : ℤ) : div_by_3 a ∧ div_by_5 a → a % 15 = 4 :=
by
  sorry

end find_a_l117_11719

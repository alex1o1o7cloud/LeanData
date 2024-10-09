import Mathlib

namespace minneapolis_st_louis_temperature_l317_31732

theorem minneapolis_st_louis_temperature (N M L : ℝ) (h1 : M = L + N)
                                         (h2 : M - 7 = L + N - 7)
                                         (h3 : L + 5 = L + 5)
                                         (h4 : (M - 7) - (L + 5) = |(L + N - 7) - (L + 5)|) :
  ∃ (N1 N2 : ℝ), (|N - 12| = 4) ∧ N1 = 16 ∧ N2 = 8 ∧ N1 * N2 = 128 :=
by {
  sorry
}

end minneapolis_st_louis_temperature_l317_31732


namespace number_minus_29_l317_31778

theorem number_minus_29 (x : ℕ) (h : x - 46 = 15) : x - 29 = 32 :=
sorry

end number_minus_29_l317_31778


namespace max_sum_arith_seq_l317_31719

theorem max_sum_arith_seq :
  let a1 := 29
  let d := 2
  let a_n (n : ℕ) := a1 + (n - 1) * d
  let S_n (n : ℕ) := n / 2 * (a1 + a_n n)
  S_n 10 = S_n 20 → S_n 20 = 960 := by
sorry

end max_sum_arith_seq_l317_31719


namespace kristin_runs_n_times_faster_l317_31724

theorem kristin_runs_n_times_faster (D K S : ℝ) (n : ℝ) 
  (h1 : K = n * S) 
  (h2 : 12 * D / K = 4 * D / S) : 
  n = 3 :=
by
  sorry

end kristin_runs_n_times_faster_l317_31724


namespace household_peak_consumption_l317_31770

theorem household_peak_consumption
  (p_orig p_peak p_offpeak : ℝ)
  (consumption : ℝ)
  (monthly_savings : ℝ)
  (x : ℝ)
  (h_orig : p_orig = 0.52)
  (h_peak : p_peak = 0.55)
  (h_offpeak : p_offpeak = 0.35)
  (h_consumption : consumption = 200)
  (h_savings : monthly_savings = 0.10) :
  (p_orig - p_peak) * x + (p_orig - p_offpeak) * (consumption - x) ≥ p_orig * consumption * monthly_savings → x ≤ 118 :=
sorry

end household_peak_consumption_l317_31770


namespace career_preference_representation_l317_31762

noncomputable def male_to_female_ratio : ℕ × ℕ := (2, 3)
noncomputable def total_students := male_to_female_ratio.1 + male_to_female_ratio.2
noncomputable def students_prefer_career := 2
noncomputable def full_circle_degrees := 360

theorem career_preference_representation :
  (students_prefer_career / total_students : ℚ) * full_circle_degrees = 144 := by
  sorry

end career_preference_representation_l317_31762


namespace marked_price_l317_31779

theorem marked_price (x : ℝ) (purchase_price : ℝ) (selling_price : ℝ) (profit_margin : ℝ) 
  (h_purchase_price : purchase_price = 100)
  (h_profit_margin : profit_margin = 0.2)
  (h_selling_price : selling_price = purchase_price * (1 + profit_margin))
  (h_price_relation : 0.8 * x = selling_price) : 
  x = 150 :=
by sorry

end marked_price_l317_31779


namespace find_a_for_parallel_lines_l317_31726

theorem find_a_for_parallel_lines (a : ℝ) :
  (∀ x y : ℝ, ax + 3 * y + 1 = 0 ↔ 2 * x + (a + 1) * y + 1 = 0) → a = -3 :=
by
  sorry

end find_a_for_parallel_lines_l317_31726


namespace pencils_remaining_in_drawer_l317_31703

-- Definitions of the conditions
def total_pencils_initially : ℕ := 34
def pencils_taken : ℕ := 22

-- The theorem statement with the correct answer
theorem pencils_remaining_in_drawer : total_pencils_initially - pencils_taken = 12 :=
by
  sorry

end pencils_remaining_in_drawer_l317_31703


namespace total_seeds_eaten_correct_l317_31769

-- Define the number of seeds each player ate
def seeds_first_player : ℕ := 78
def seeds_second_player : ℕ := 53
def seeds_third_player (seeds_second_player : ℕ) : ℕ := seeds_second_player + 30

-- Define the total seeds eaten
def total_seeds_eaten (seeds_first_player seeds_second_player seeds_third_player : ℕ) : ℕ :=
  seeds_first_player + seeds_second_player + seeds_third_player

-- Statement of the theorem
theorem total_seeds_eaten_correct : total_seeds_eaten seeds_first_player seeds_second_player (seeds_third_player seeds_second_player) = 214 :=
by
  sorry

end total_seeds_eaten_correct_l317_31769


namespace number_of_students_earning_B_l317_31715

variables (a b c : ℕ) -- since we assume we only deal with whole numbers

-- Given conditions:
-- 1. The probability of earning an A is twice the probability of earning a B.
axiom h1 : a = 2 * b
-- 2. The probability of earning a C is equal to the probability of earning a B.
axiom h2 : c = b
-- 3. The only grades are A, B, or C and there are 45 students in the class.
axiom h3 : a + b + c = 45

-- Prove that the number of students earning a B is 11.
theorem number_of_students_earning_B : b = 11 :=
by
    sorry

end number_of_students_earning_B_l317_31715


namespace sum_of_angles_subtended_by_arcs_l317_31736

theorem sum_of_angles_subtended_by_arcs
  (A B X Y C : Type)
  (arc_AX arc_XC : ℝ)
  (h1 : arc_AX = 58)
  (h2 : arc_XC = 62)
  (R S : ℝ)
  (hR : R = arc_AX / 2)
  (hS : S = arc_XC / 2) :
  R + S = 60 :=
by
  rw [hR, hS, h1, h2]
  norm_num

end sum_of_angles_subtended_by_arcs_l317_31736


namespace least_integer_x_l317_31721

theorem least_integer_x (x : ℤ) (h : 240 ∣ x^2) : x = 60 :=
sorry

end least_integer_x_l317_31721


namespace penny_purchase_exceeded_minimum_spend_l317_31723

theorem penny_purchase_exceeded_minimum_spend :
  let bulk_price_per_pound := 5
  let minimum_spend := 40
  let tax_per_pound := 1
  let total_paid := 240
  let total_cost_per_pound := bulk_price_per_pound + tax_per_pound
  let pounds_purchased := total_paid / total_cost_per_pound
  let minimum_pounds_to_spend := minimum_spend / bulk_price_per_pound
  pounds_purchased - minimum_pounds_to_spend = 32 :=
by
  -- The proof is omitted here as per the instructions.
  sorry

end penny_purchase_exceeded_minimum_spend_l317_31723


namespace distance_reflection_x_axis_l317_31720

/--
Given points C and its reflection over the x-axis C',
prove that the distance between C and C' is 6.
-/
theorem distance_reflection_x_axis :
  let C := (-2, 3)
  let C' := (-2, -3)
  dist C C' = 6 := by
  sorry

end distance_reflection_x_axis_l317_31720


namespace intersection_line_eq_l317_31782

-- Definitions of the circles
def circle1 (x y : ℝ) := x^2 + y^2 - 4*x - 6 = 0
def circle2 (x y : ℝ) := x^2 + y^2 - 4*y - 6 = 0

-- The theorem stating that the equation of the line passing through their intersection points is x = y
theorem intersection_line_eq (x y : ℝ) :
  (circle1 x y → circle2 x y → x = y) := 
by
  intro h1 h2
  sorry

end intersection_line_eq_l317_31782


namespace part_I_part_II_l317_31752

noncomputable def f (x a : ℝ) := 2 * |x - 1| - a
noncomputable def g (x m : ℝ) := - |x + m|

theorem part_I (a : ℝ) : 
  (∃! x : ℤ, x = -3 ∧ g x 3 > -1) → m = 3 := 
sorry

theorem part_II (m : ℝ) : 
  (∀ x : ℝ, f x a > g x m) → a < 4 := 
sorry

end part_I_part_II_l317_31752


namespace intersection_eq_l317_31784

def set_M : Set ℝ := { x : ℝ | (x + 3) * (x - 2) < 0 }
def set_N : Set ℝ := { x : ℝ | 1 ≤ x ∧ x ≤ 3 }

theorem intersection_eq : set_M ∩ set_N = { x : ℝ | 1 ≤ x ∧ x < 2 } := by
  sorry

end intersection_eq_l317_31784


namespace geometric_sequence_common_ratio_l317_31733

theorem geometric_sequence_common_ratio :
  ∀ (a : ℕ → ℝ) (q : ℝ),
  (∀ n, a (n + 1) = a n * q) →
  (∀ n m, n < m → a n < a m) →
  a 2 = 2 →
  a 4 - a 3 = 4 →
  q = 2 :=
by
  intros a q h_geo h_inc h_a2 h_a4_a3
  sorry

end geometric_sequence_common_ratio_l317_31733


namespace unique_peg_placement_l317_31755

noncomputable def peg_placement := 
  ∃! f : (Fin 6 → Fin 6 → Option (Fin 5)), 
    (∀ i j, f i j = some 0 → (∀ k, k ≠ i → f k j ≠ some 0) ∧ (∀ l, l ≠ j → f i l ≠ some 0)) ∧  -- Yellow pegs
    (∀ i j, f i j = some 1 → (∀ k, k ≠ i → f k j ≠ some 1) ∧ (∀ l, l ≠ j → f i l ≠ some 1)) ∧  -- Red pegs
    (∀ i j, f i j = some 2 → (∀ k, k ≠ i → f k j ≠ some 2) ∧ (∀ l, l ≠ j → f i l ≠ some 2)) ∧  -- Green pegs
    (∀ i j, f i j = some 3 → (∀ k, k ≠ i → f k j ≠ some 3) ∧ (∀ l, l ≠ j → f i l ≠ some 3)) ∧  -- Blue pegs
    (∀ i j, f i j = some 4 → (∀ k, k ≠ i → f k j ≠ some 4) ∧ (∀ l, l ≠ j → f i l ≠ some 4)) ∧  -- Orange pegs
    (∃! i j, f i j = some 0) ∧
    (∃! i j, f i j = some 1) ∧
    (∃! i j, f i j = some 2) ∧
    (∃! i j, f i j = some 3) ∧
    (∃! i j, f i j = some 4)
    
theorem unique_peg_placement : peg_placement :=
sorry

end unique_peg_placement_l317_31755


namespace carol_blocks_l317_31797

theorem carol_blocks (initial_blocks lost_blocks final_blocks : ℕ) 
  (h_initial : initial_blocks = 42) 
  (h_lost : lost_blocks = 25) : 
  final_blocks = initial_blocks - lost_blocks → final_blocks = 17 := by
  sorry

end carol_blocks_l317_31797


namespace log_expression_as_product_l317_31737

noncomputable def log (b : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log b

theorem log_expression_as_product (A m n p : ℝ) (hm : 0 < m) (hn : 0 < n) (hp : 0 < p) (hA : 0 < A) :
  log m A * log n A + log n A * log p A + log p A * log m A =
  log A (m * n * p) * log p A * log n A * log m A :=
by
  sorry

end log_expression_as_product_l317_31737


namespace circle_circumference_l317_31725

theorem circle_circumference (a b : ℝ) (h1 : a = 9) (h2 : b = 12) :
  ∃ c : ℝ, c = 15 * Real.pi :=
by
  sorry

end circle_circumference_l317_31725


namespace problem1_problem2_problem3_problem4_l317_31792

-- Problem 1
theorem problem1 : (-3 : ℝ) ^ 2 + (1 / 2) ^ (-1 : ℝ) + (Real.pi - 3) ^ 0 = 12 :=
by
  sorry

-- Problem 2
theorem problem2 (x : ℝ) : (8 * x ^ 4 + 4 * x ^ 3 - x ^ 2) / (-2 * x) ^ 2 = 2 * x ^ 2 + x - 1 / 4 :=
by
  sorry

-- Problem 3
theorem problem3 (x : ℝ) : (2 * x + 1) ^ 2 - (4 * x + 1) * (x + 1) = -x :=
by
  sorry

-- Problem 4
theorem problem4 (x y : ℝ) : (x + 2 * y - 3) * (x - 2 * y + 3) = x ^ 2 - 4 * y ^ 2 + 12 * y - 9 :=
by
  sorry

end problem1_problem2_problem3_problem4_l317_31792


namespace range_neg2a_plus_3_l317_31749

theorem range_neg2a_plus_3 (a : ℝ) (h : a < 1) : -2 * a + 3 > 1 :=
sorry

end range_neg2a_plus_3_l317_31749


namespace divides_a_square_minus_a_and_a_cube_minus_a_l317_31729

theorem divides_a_square_minus_a_and_a_cube_minus_a (a : ℤ) : 
  (2 ∣ a^2 - a) ∧ (3 ∣ a^3 - a) :=
by
  sorry

end divides_a_square_minus_a_and_a_cube_minus_a_l317_31729


namespace max_elements_set_M_l317_31750

theorem max_elements_set_M (n : ℕ) (hn : n ≥ 2) (M : Finset (ℕ × ℕ))
  (hM : ∀ {i k}, (i, k) ∈ M → i < k → ∀ {m}, k < m → (k, m) ∉ M) :
  M.card ≤ n^2 / 4 :=
sorry

end max_elements_set_M_l317_31750


namespace smallest_sum_arith_geo_sequence_l317_31758

theorem smallest_sum_arith_geo_sequence 
  (A B C D: ℕ) 
  (h1: A > 0) 
  (h2: B > 0) 
  (h3: C > 0) 
  (h4: D > 0)
  (h5: 2 * B = A + C)
  (h6: B * D = C * C)
  (h7: 3 * C = 4 * B) : 
  A + B + C + D = 43 := 
sorry

end smallest_sum_arith_geo_sequence_l317_31758


namespace meaning_of_probability_l317_31788

-- Definitions

def probability_of_winning (p : ℚ) : Prop :=
  p = 1 / 4

-- Theorem statement
theorem meaning_of_probability :
  probability_of_winning (1 / 4) →
  ∀ n : ℕ, (n ≠ 0) → (n / 4 * 4) = n :=
by
  -- Placeholder proof
  sorry

end meaning_of_probability_l317_31788


namespace find_range_of_m_l317_31783

noncomputable def range_of_m (m : ℝ) : Prop :=
  ((1 < m ∧ m ≤ 2) ∨ (3 ≤ m))

theorem find_range_of_m (m : ℝ) :
  (∃ x : ℝ, x^2 + m*x + 1 = 0 ∧ ∀ x1 x2 : ℝ, x1 ≠ x2 → x1 < 0 ∧ x2 < 0 ∧ x1^2 + m*x1 + 1 = 0 ∧ x2^2 + m*x2 + 1 = 0) ∨
  (¬ ∃ x : ℝ, 4 * x^2 + 4 * (m - 2) * x + 1 = 0 ∧ ∀ Δ, Δ < 0 ∧ Δ = 16 * (m^2 - 4 * m + 3)) ↔
  ¬((∃ x : ℝ, x^2 + m*x + 1 = 0 ∧ ∀ x1 x2 : ℝ, x1 ≠ x2 → x1 < 0 ∧ x2 < 0 ∧ x1^2 + m*x1 + 1 = 0 ∧ x2^2 + m*x2 + 1 = 0) ∧
  (¬ ∃ x : ℝ, 4 * x^2 + 4 * (m - 2) * x + 1 = 0 ∧ ∀ Δ, Δ < 0 ∧ Δ = 16 * (m^2 - 4 * m + 3))) →
  range_of_m m :=
sorry

end find_range_of_m_l317_31783


namespace impossible_odd_sum_l317_31747

theorem impossible_odd_sum (n m : ℤ) (h1 : (n^3 + m^3) % 2 = 0) (h2 : (n^3 + m^3) % 4 = 0) : (n + m) % 2 = 0 :=
sorry

end impossible_odd_sum_l317_31747


namespace diane_trip_length_l317_31709

-- Define constants and conditions
def first_segment_fraction : ℚ := 1 / 4
def middle_segment_length : ℚ := 24
def last_segment_fraction : ℚ := 1 / 3

def total_trip_length (x : ℚ) : Prop :=
  (1 - first_segment_fraction - last_segment_fraction) * x = middle_segment_length

theorem diane_trip_length : ∃ x : ℚ, total_trip_length x ∧ x = 57.6 := by
  sorry

end diane_trip_length_l317_31709


namespace oranges_after_selling_l317_31763

-- Definitions derived from the conditions
def oranges_picked := 37
def oranges_sold := 10
def oranges_left := 27

-- The theorem to prove that Joan is left with 27 oranges
theorem oranges_after_selling (h : oranges_picked - oranges_sold = oranges_left) : oranges_left = 27 :=
by
  -- Proof omitted
  sorry

end oranges_after_selling_l317_31763


namespace solve_system_of_equations_l317_31728

theorem solve_system_of_equations (x y z : ℝ) :
  (2 * x^2 / (1 + x^2) = y) →
  (2 * y^2 / (1 + y^2) = z) →
  (2 * z^2 / (1 + z^2) = x) →
  ((x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1 ∧ y = 1 ∧ z = 1)) :=
by
  sorry

end solve_system_of_equations_l317_31728


namespace gcd_98_63_l317_31713

-- Definition of gcd
def gcd_euclidean := ∀ (a b : ℕ), ∃ (g : ℕ), gcd a b = g

-- Statement of the problem using Lean
theorem gcd_98_63 : gcd 98 63 = 7 := by
  sorry

end gcd_98_63_l317_31713


namespace cannot_be_2009_l317_31738

theorem cannot_be_2009 (a b c : ℕ) (h : b * 1234^2 + c * 1234 + a = c * 1234^2 + a * 1234 + b) : (b * 1^2 + c * 1 + a ≠ 2009) :=
by
  sorry

end cannot_be_2009_l317_31738


namespace max_value_of_fraction_l317_31705

-- Define the problem statement:
theorem max_value_of_fraction (x y z : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100) (hz : 10 ≤ z ∧ z < 100) 
  (hmean : (x + y + z) / 3 = 60) : ∃ x y z, (∀ x y z, (10 ≤ x ∧ x < 100) ∧ (10 ≤ y ∧ y < 100) ∧ (10 ≤ z ∧ z < 100) ∧ (x + y + z) / 3 = 60 → 
  (x + y) / z ≤ 17) ∧ ((x + y) / z = 17) :=
by
  sorry

end max_value_of_fraction_l317_31705


namespace custom_op_value_l317_31766

-- Define the custom operation (a \$ b)
def custom_op (a b : Int) : Int := a * (b - 1) + a * b

-- Main theorem to prove the equivalence
theorem custom_op_value : custom_op 5 (-3) = -35 := by
  sorry

end custom_op_value_l317_31766


namespace curve_symmetry_l317_31740

-- Define the curve equation
def curve_eq (x y : ℝ) : Prop := x * y^2 - x^2 * y = -2

-- Define the symmetry condition about the line y = -x
def symmetry_about_y_equals_neg_x (x y : ℝ) : Prop :=
  curve_eq (-y) (-x)

-- Define the original curve equation
def original_curve (x y : ℝ) : Prop := curve_eq x y

-- Proof statement: The curve xy^2 - x^2y = -2 is symmetric about the line y = -x.
theorem curve_symmetry : ∀ (x y : ℝ), original_curve x y ↔ symmetry_about_y_equals_neg_x x y :=
by
  sorry

end curve_symmetry_l317_31740


namespace chocolate_bars_per_box_l317_31731

theorem chocolate_bars_per_box (total_chocolate_bars boxes : ℕ) (h1 : total_chocolate_bars = 710) (h2 : boxes = 142) : total_chocolate_bars / boxes = 5 := by
  sorry

end chocolate_bars_per_box_l317_31731


namespace find_divisor_l317_31786

theorem find_divisor (n x y z a b c : ℕ) (h1 : 63 = n * x + a) (h2 : 91 = n * y + b) (h3 : 130 = n * z + c) (h4 : a + b + c = 26) : n = 43 :=
sorry

end find_divisor_l317_31786


namespace total_distance_is_20_l317_31798

noncomputable def total_distance_walked (x : ℝ) : ℝ :=
  let flat_distance := 4 * x
  let uphill_time := (2 / 3) * (5 - x)
  let uphill_distance := 3 * uphill_time
  let downhill_time := (1 / 3) * (5 - x)
  let downhill_distance := 6 * downhill_time
  flat_distance + uphill_distance + downhill_distance

theorem total_distance_is_20 :
  ∃ x : ℝ, x >= 0 ∧ x <= 5 ∧ total_distance_walked x = 20 :=
by
  -- The existence proof is omitted (hence the sorry)
  sorry

end total_distance_is_20_l317_31798


namespace corrected_mean_is_45_55_l317_31744

-- Define the initial conditions
def mean_of_100_observations (mean : ℝ) : Prop :=
  mean = 45

def incorrect_observation : ℝ := 32
def correct_observation : ℝ := 87

-- Define the calculation of the corrected mean
noncomputable def corrected_mean (incorrect_mean : ℝ) (incorrect_obs : ℝ) (correct_obs : ℝ) (n : ℕ) : ℝ :=
  let sum_original := incorrect_mean * n
  let difference := correct_obs - incorrect_obs
  (sum_original + difference) / n

-- Theorem: The corrected new mean is 45.55
theorem corrected_mean_is_45_55 : corrected_mean 45 32 87 100 = 45.55 :=
by
  sorry

end corrected_mean_is_45_55_l317_31744


namespace jelly_cost_l317_31773

theorem jelly_cost (B J : ℕ) 
  (h1 : 15 * (6 * B + 7 * J) = 315) 
  (h2 : 0 ≤ B) 
  (h3 : 0 ≤ J) : 
  15 * J * 7 = 315 := 
sorry

end jelly_cost_l317_31773


namespace frog_probability_0_4_l317_31711

-- Definitions and conditions
def vertices : List (ℤ × ℤ) := [(1,1), (1,6), (5,6), (5,1)]
def start_position : ℤ × ℤ := (2,3)

-- Probabilities for transition, boundary definitions, this mimics the recursive nature described
def P : ℤ × ℤ → ℝ
| (x, 1) => 1   -- Boundary condition for horizontal sides
| (x, 6) => 1   -- Boundary condition for horizontal sides
| (1, y) => 0   -- Boundary condition for vertical sides
| (5, y) => 0   -- Boundary condition for vertical sides
| (x, y) => sorry  -- General case for other positions

-- The theorem to prove
theorem frog_probability_0_4 : P (2, 3) = 0.4 :=
by
  sorry

end frog_probability_0_4_l317_31711


namespace area_of_circle_above_below_lines_l317_31714

noncomputable def circle_area : ℝ :=
  40 * Real.pi

theorem area_of_circle_above_below_lines :
  ∃ (x y : ℝ), (x^2 + y^2 - 16*x - 8*y = 0) ∧ (y > x - 4) ∧ (y < -x + 4) ∧
  (circle_area = 40 * Real.pi) :=
  sorry

end area_of_circle_above_below_lines_l317_31714


namespace complement_union_l317_31794

open Set

def set_A : Set ℝ := {x | x ≤ 0}
def set_B : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}

theorem complement_union (A B : Set ℝ) (hA : A = set_A) (hB : B = set_B) :
  (univ \ (A ∪ B) = {x | 1 < x}) := by
  rw [hA, hB]
  sorry

end complement_union_l317_31794


namespace find_a_sq_plus_b_sq_l317_31700

theorem find_a_sq_plus_b_sq (a b : ℝ) (h1 : a - b = 3) (h2 : a * b = 10) :
  a^2 + b^2 = 29 := by
  sorry

end find_a_sq_plus_b_sq_l317_31700


namespace prop_neg_or_not_l317_31718

theorem prop_neg_or_not (p q : Prop) (h : ¬(p ∨ ¬ q)) : ¬ p ∧ q :=
by
  sorry

end prop_neg_or_not_l317_31718


namespace distance_between_A_and_B_l317_31799

theorem distance_between_A_and_B 
  (d : ℝ)
  (h1 : ∀ (t : ℝ), (t = 2 * (t / 2)) → t = 200) 
  (h2 : ∀ (t : ℝ), 100 = d - (t / 2 + 50))
  (h3 : ∀ (t : ℝ), d = 2 * (d - 60)): 
  d = 300 :=
sorry

end distance_between_A_and_B_l317_31799


namespace number_of_apples_l317_31765

theorem number_of_apples (A : ℝ) (h : 0.75 * A * 0.5 + 0.25 * A * 0.1 = 40) : A = 100 :=
by
  sorry

end number_of_apples_l317_31765


namespace star_equiv_zero_l317_31708

-- Define the new operation for real numbers a and b
def star (a b : ℝ) : ℝ := (a^2 - b^2)^2

-- Prove that (x^2 - y^2) star (y^2 - x^2) equals 0
theorem star_equiv_zero (x y : ℝ) : star (x^2 - y^2) (y^2 - x^2) = 0 := 
by sorry

end star_equiv_zero_l317_31708


namespace arithmetic_sqrt_of_4_eq_2_l317_31781

theorem arithmetic_sqrt_of_4_eq_2 (x : ℕ) (h : x^2 = 4) : x = 2 :=
sorry

end arithmetic_sqrt_of_4_eq_2_l317_31781


namespace otimes_eq_abs_m_leq_m_l317_31739

noncomputable def otimes (x y : ℝ) : ℝ :=
if x ≤ y then x else y

theorem otimes_eq_abs_m_leq_m' :
  ∀ (m : ℝ), otimes (abs (m - 1)) m = abs (m - 1) → m ∈ Set.Ici (1 / 2) := 
by
  sorry

end otimes_eq_abs_m_leq_m_l317_31739


namespace two_class_students_l317_31787

-- Define the types of students and total sum variables
variables (H M E HM HE ME HME : ℕ)
variable (Total_Students : ℕ)

-- Given conditions
axiom condition1 : Total_Students = 68
axiom condition2 : H = 19
axiom condition3 : M = 14
axiom condition4 : E = 26
axiom condition5 : HME = 3

-- Inclusion-Exclusion principle formula application
def exactly_two_classes : Prop := 
  Total_Students = H + M + E - (HM + HE + ME) + HME

-- Theorem to prove the number of students registered for exactly two classes is 6
theorem two_class_students : H + M + E - 2 * HME + HME - (HM + HE + ME) = 6 := by
  sorry

end two_class_students_l317_31787


namespace Doug_age_l317_31777

theorem Doug_age (Q J D : ℕ) (h1 : Q = J + 6) (h2 : J = D - 3) (h3 : Q = 19) : D = 16 := by
  sorry

end Doug_age_l317_31777


namespace zyka_expense_increase_l317_31734

theorem zyka_expense_increase (C_k C_c : ℝ) (h1 : 0.5 * C_k = 0.2 * C_c) : 
  (((1.2 * C_c) - C_c) / C_c) * 100 = 20 := by
  sorry

end zyka_expense_increase_l317_31734


namespace max_marks_l317_31751

variable (M : ℝ)

-- Conditions
def needed_to_pass (M : ℝ) := 0.20 * M
def pradeep_marks := 390
def marks_short := 25
def total_marks_needed := pradeep_marks + marks_short

-- Theorem statement
theorem max_marks : needed_to_pass M = total_marks_needed → M = 2075 := by
  sorry

end max_marks_l317_31751


namespace sum_of_ages_l317_31795

-- Definition of the ages based on the intervals and the youngest child's age.
def youngest_age : ℕ := 6
def second_youngest_age : ℕ := youngest_age + 2
def middle_age : ℕ := youngest_age + 4
def second_oldest_age : ℕ := youngest_age + 6
def oldest_age : ℕ := youngest_age + 8

-- The theorem stating the total sum of the ages of the children, given the conditions.
theorem sum_of_ages :
  youngest_age + second_youngest_age + middle_age + second_oldest_age + oldest_age = 50 :=
by sorry

end sum_of_ages_l317_31795


namespace four_digit_flippies_div_by_4_l317_31743

def is_flippy (n : ℕ) : Prop := 
  let digits := [4, 6]
  n / 1000 ∈ digits ∧
  (n / 100 % 10) ∈ digits ∧
  ((n / 10 % 10) = if (n / 100 % 10) = 4 then 6 else 4) ∧
  (n % 10) = if (n / 1000) = 4 then 6 else 4

def is_divisible_by_4 (n : ℕ) : Prop :=
  n % 4 = 0

theorem four_digit_flippies_div_by_4 : 
  ∃! n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ is_flippy n ∧ is_divisible_by_4 n :=
by
  sorry

end four_digit_flippies_div_by_4_l317_31743


namespace is_inverse_g1_is_inverse_g2_l317_31771

noncomputable def f (x : ℝ) := 3 + 2*x - x^2

noncomputable def g1 (x : ℝ) := -1 + Real.sqrt (4 - x)
noncomputable def g2 (x : ℝ) := -1 - Real.sqrt (4 - x)

theorem is_inverse_g1 : ∀ x, f (g1 x) = x :=
by
  intro x
  sorry

theorem is_inverse_g2 : ∀ x, f (g2 x) = x :=
by
  intro x
  sorry

end is_inverse_g1_is_inverse_g2_l317_31771


namespace find_scalars_l317_31716

noncomputable def B : Matrix (Fin 2) (Fin 2) ℝ :=
  !![1, 2;
    3, 1]

noncomputable def B4 : Matrix (Fin 2) (Fin 2) ℝ :=
  B * B * B * B

noncomputable def I : Matrix (Fin 2) (Fin 2) ℝ :=
  1

theorem find_scalars (r s : ℝ) (hB : B^4 = r • B + s • I) :
  (r, s) = (51, 52) :=
  sorry

end find_scalars_l317_31716


namespace problem_provable_l317_31722

noncomputable def given_expression (a : ℝ) : ℝ :=
  (1 / (a + 2)) / ((a^2 - 4 * a + 4) / (a^2 - 4)) - (2 / (a - 2))

theorem problem_provable : given_expression (Real.sqrt 5 + 2) = - (Real.sqrt 5 / 5) :=
by
  sorry

end problem_provable_l317_31722


namespace minimum_disks_needed_l317_31764

-- Define the conditions
def total_files : ℕ := 25
def disk_capacity : ℝ := 2.0
def files_06MB : ℕ := 5
def size_06MB_file : ℝ := 0.6
def files_10MB : ℕ := 10
def size_10MB_file : ℝ := 1.0
def files_03MB : ℕ := total_files - files_06MB - files_10MB
def size_03MB_file : ℝ := 0.3

-- Define the theorem that needs to be proved
theorem minimum_disks_needed : 
    ∃ (disks: ℕ), disks = 10 ∧ 
    (5 * size_06MB_file + 10 * size_10MB_file + 10 * size_03MB_file) ≤ disks * disk_capacity := 
by
  sorry

end minimum_disks_needed_l317_31764


namespace greenfield_academy_math_count_l317_31761

theorem greenfield_academy_math_count (total_players taking_physics both_subjects : ℕ) 
(h_total: total_players = 30) 
(h_physics: taking_physics = 15) 
(h_both: both_subjects = 3) : 
∃ taking_math : ℕ, taking_math = 21 :=
by
  sorry

end greenfield_academy_math_count_l317_31761


namespace discount_equivalence_l317_31759

variable (Original_Price : ℝ)

theorem discount_equivalence (h1 : Real) (h2 : Real) :
  (h1 = 0.5 * Original_Price) →
  (h2 = 0.7 * h1) →
  (Original_Price - h2) / Original_Price = 0.65 :=
by
  intros
  sorry

end discount_equivalence_l317_31759


namespace vectors_perpendicular_vector_combination_l317_31756

def vector_a : ℝ × ℝ := (2, -1)
def vector_b : ℝ × ℝ := (-3, 2)
def vector_c : ℝ × ℝ := (1, 1)

-- Auxiliary definition of vector addition
def vector_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 + v2.1, v1.2 + v2.2)

-- Auxiliary definition of dot product
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := (v1.1 * v2.1 + v1.2 * v2.2)

-- Auxiliary definition of scalar multiplication
def scalar_mul (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)

-- Proof that (vector_a + vector_b) is perpendicular to vector_c
theorem vectors_perpendicular : dot_product (vector_add vector_a vector_b) vector_c = 0 :=
by sorry

-- Proof that vector_c = 5 * vector_a + 3 * vector_b
theorem vector_combination : vector_c = vector_add (scalar_mul 5 vector_a) (scalar_mul 3 vector_b) :=
by sorry

end vectors_perpendicular_vector_combination_l317_31756


namespace smallest_term_of_sequence_l317_31735

-- Define the sequence a_n
def a (n : ℕ) : ℤ := 3 * n^2 - 28 * n

-- The statement that the 5th term is the smallest in the sequence
theorem smallest_term_of_sequence : ∀ n : ℕ, a 5 ≤ a n := by
  sorry

end smallest_term_of_sequence_l317_31735


namespace households_using_all_three_brands_correct_l317_31748

noncomputable def total_households : ℕ := 5000
noncomputable def non_users : ℕ := 1200
noncomputable def only_X : ℕ := 800
noncomputable def only_Y : ℕ := 600
noncomputable def only_Z : ℕ := 300

-- Let A be the number of households that used all three brands of soap
variable (A : ℕ)

-- For every household that used all three brands, 5 used only two brands and 10 used just one brand.
-- Number of households that used only two brands = 5 * A
-- Number of households that used only one brand = 10 * A

-- The equation for households that used just one brand:
def households_using_all_three_brands :=
10 * A = only_X + only_Y + only_Z

theorem households_using_all_three_brands_correct :
  (total_households - non_users = only_X + only_Y + only_Z + 5 * A + 10 * A) →
  (A = 170) := by
sorry

end households_using_all_three_brands_correct_l317_31748


namespace overall_average_correct_l317_31707

noncomputable def overall_average : ℝ :=
  let students1 := 60
  let students2 := 35
  let students3 := 45
  let students4 := 42
  let avgMarks1 := 50
  let avgMarks2 := 60
  let avgMarks3 := 55
  let avgMarks4 := 45
  let total_students := students1 + students2 + students3 + students4
  let total_marks := (students1 * avgMarks1) + (students2 * avgMarks2) + (students3 * avgMarks3) + (students4 * avgMarks4)
  total_marks / total_students

theorem overall_average_correct : overall_average = 52.00 := by
  sorry

end overall_average_correct_l317_31707


namespace rectangle_hall_length_l317_31780

variable (L B : ℝ)

theorem rectangle_hall_length (h1 : B = (2 / 3) * L) (h2 : L * B = 2400) : L = 60 :=
by sorry

end rectangle_hall_length_l317_31780


namespace joan_exam_time_difference_l317_31790

theorem joan_exam_time_difference :
  ∀ (E_time M_time E_questions M_questions : ℕ),
  E_time = 60 →
  M_time = 90 →
  E_questions = 30 →
  M_questions = 15 →
  (M_time / M_questions) - (E_time / E_questions) = 4 :=
by
  intros E_time M_time E_questions M_questions hE_time hM_time hE_questions hM_questions
  sorry

end joan_exam_time_difference_l317_31790


namespace intersection_distance_l317_31772

noncomputable def distance_between_intersections (l : ℝ → ℝ → Prop) (C : ℝ → ℝ → Prop) : Prop :=
  ∃ A B : ℝ × ℝ, 
    l A.1 A.2 ∧ C A.1 A.2 ∧ l B.1 B.2 ∧ C B.1 B.2 ∧ 
    dist A B = Real.sqrt 6

def line_l (x y : ℝ) : Prop :=
  x - y + 1 = 0

def curve_C (x y : ℝ) : Prop :=
  ∃ θ : ℝ, x = Real.sqrt 2 * Real.cos θ ∧ y = Real.sqrt 2 * Real.sin θ

theorem intersection_distance :
  distance_between_intersections line_l curve_C :=
sorry

end intersection_distance_l317_31772


namespace fraction_spent_on_furniture_l317_31706

variable (original_savings : ℕ)
variable (tv_cost : ℕ)
variable (f : ℚ)

-- Defining the conditions
def conditions := original_savings = 500 ∧ tv_cost = 100 ∧
  f = (original_savings - tv_cost) / original_savings

-- The theorem we want to prove
theorem fraction_spent_on_furniture : conditions original_savings tv_cost f → f = 4 / 5 := by
  sorry

end fraction_spent_on_furniture_l317_31706


namespace seq_general_form_l317_31785

theorem seq_general_form (p r : ℝ) (a : ℕ → ℝ)
  (hp : p > r)
  (hr : r > 0)
  (h_init : a 1 = r)
  (h_recurrence : ∀ n : ℕ, a (n+1) = p * a n + r^(n+1)) :
  ∀ n : ℕ, a n = r * (p^n - r^n) / (p - r) :=
by
  sorry

end seq_general_form_l317_31785


namespace find_multiple_of_numerator_l317_31768

theorem find_multiple_of_numerator
  (n d k : ℕ)
  (h1 : d = k * n - 1)
  (h2 : (n + 1) / (d + 1) = 3 / 5)
  (h3 : (n : ℚ) / d = 5 / 9) : k = 2 :=
sorry

end find_multiple_of_numerator_l317_31768


namespace maximal_points_coloring_l317_31730

/-- Given finitely many points in the plane where no three points are collinear,
which are colored either red or green, such that any monochromatic triangle
contains at least one point of the other color in its interior, the maximal number
of such points is 8. -/
theorem maximal_points_coloring (points : Finset (ℝ × ℝ))
  (h_no_three_collinear : ∀ (p1 p2 p3 : ℝ × ℝ), p1 ∈ points → p2 ∈ points → p3 ∈ points →
    p1 ≠ p2 → p2 ≠ p3 → p1 ≠ p3 →
    ¬ ∃ k b, ∀ p ∈ [p1, p2, p3], p.2 = k * p.1 + b)
  (colored : (ℝ × ℝ) → Prop)
  (h_coloring : ∀ (p1 p2 p3 : ℝ × ℝ), p1 ∈ points → p2 ∈ points → p3 ∈ points →
    colored p1 = colored p2 → colored p2 = colored p3 →
    ∃ p, p ∈ points ∧ colored p ≠ colored p1) :
  points.card ≤ 8 :=
sorry

end maximal_points_coloring_l317_31730


namespace perfect_score_l317_31746

theorem perfect_score (P : ℕ) (h : 3 * P = 63) : P = 21 :=
by
  -- Proof to be provided
  sorry

end perfect_score_l317_31746


namespace problem200_squared_minus_399_composite_l317_31774

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_composite (n : ℕ) : Prop :=
  ¬ is_prime n

theorem problem200_squared_minus_399_composite : is_composite (200^2 - 399) :=
sorry

end problem200_squared_minus_399_composite_l317_31774


namespace time_difference_l317_31741

-- Define the capacity of the tanks
def capacity : ℕ := 20

-- Define the inflow rates of tanks A and B in litres per hour
def inflow_rate_A : ℕ := 2
def inflow_rate_B : ℕ := 4

-- Define the times to fill tanks A and B
def time_A : ℕ := capacity / inflow_rate_A
def time_B : ℕ := capacity / inflow_rate_B

-- Proving the time difference between filling tanks A and B
theorem time_difference : (time_A - time_B) = 5 := by
  sorry

end time_difference_l317_31741


namespace sphere_hemisphere_radius_relationship_l317_31760

theorem sphere_hemisphere_radius_relationship (r : ℝ) (R : ℝ) (π : ℝ) (h : 0 < π):
  (4 / 3) * π * R^3 = (2 / 3) * π * r^3 →
  r = 3 * (2^(1/3 : ℝ)) →
  R = 3 :=
by
  sorry

end sphere_hemisphere_radius_relationship_l317_31760


namespace candy_days_l317_31796

theorem candy_days (neighbor_candy older_sister_candy candy_per_day : ℝ) 
  (h1 : neighbor_candy = 11.0) 
  (h2 : older_sister_candy = 5.0) 
  (h3 : candy_per_day = 8.0) : 
  ((neighbor_candy + older_sister_candy) / candy_per_day) = 2.0 := 
by 
  sorry

end candy_days_l317_31796


namespace max_men_with_all_amenities_marrried_l317_31754

theorem max_men_with_all_amenities_marrried :
  let total_men := 100
  let married_men := 85
  let men_with_TV := 75
  let men_with_radio := 85
  let men_with_AC := 70
  (∀ s : Finset ℕ, s.card ≤ total_men) →
  (∀ s : Finset ℕ, s.card ≤ married_men) →
  (∀ s : Finset ℕ, s.card ≤ men_with_TV) →
  (∀ s : Finset ℕ, s.card ≤ men_with_radio) →
  (∀ s : Finset ℕ, s.card ≤ men_with_AC) →
  (∀ s : Finset ℕ, s.card ≤ min married_men (min men_with_TV (min men_with_radio men_with_AC))) :=
by
  intros
  sorry

end max_men_with_all_amenities_marrried_l317_31754


namespace problem_l317_31710

theorem problem (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + (1/r^4) = 7 := 
by
  sorry

end problem_l317_31710


namespace length_of_each_song_l317_31753

-- Conditions
def first_side_songs : Nat := 6
def second_side_songs : Nat := 4
def total_length_of_tape : Nat := 40

-- Definition of length of each song
def total_songs := first_side_songs + second_side_songs

-- Question: Prove that each song is 4 minutes long
theorem length_of_each_song (h1 : first_side_songs = 6) 
                            (h2 : second_side_songs = 4) 
                            (h3 : total_length_of_tape = 40) 
                            (h4 : total_songs = first_side_songs + second_side_songs) : 
  total_length_of_tape / total_songs = 4 :=
by
  sorry

end length_of_each_song_l317_31753


namespace unique_valid_number_l317_31767

-- Define the form of the three-digit number.
def is_form_sixb5 (n : ℕ) : Prop :=
  ∃ b : ℕ, b < 10 ∧ n = 600 + 10 * b + 5

-- Define the condition for divisibility by 11.
def is_divisible_by_11 (n : ℕ) : Prop :=
  (n % 11 = 0)

-- Define the alternating sum property for our specific number format.
def alternating_sum_cond (b : ℕ) : Prop :=
  (11 - b) % 11 = 0

-- The final proposition to be proved.
theorem unique_valid_number : ∃ n, is_form_sixb5 n ∧ is_divisible_by_11 n ∧ n = 605 :=
by {
  sorry
}

end unique_valid_number_l317_31767


namespace sin2theta_cos2theta_sum_l317_31712

theorem sin2theta_cos2theta_sum (θ : ℝ) (h1 : Real.sin θ = 2 * Real.cos θ) (h2 : Real.sin θ ^ 2 + Real.cos θ ^ 2 = 1) : 
  Real.sin (2 * θ) + Real.cos (2 * θ) = 1 / 5 :=
by
  sorry

end sin2theta_cos2theta_sum_l317_31712


namespace integral_value_l317_31704

noncomputable def integral_sin_pi_over_2_to_pi : ℝ := ∫ x in (Real.pi / 2)..Real.pi, Real.sin x

theorem integral_value : integral_sin_pi_over_2_to_pi = 1 := by
  sorry

end integral_value_l317_31704


namespace betty_berries_july_five_l317_31727
open Nat

def betty_bear_berries : Prop :=
  ∃ (b : ℕ), (5 * b + 100 = 150) ∧ (b + 40 = 50)

theorem betty_berries_july_five : betty_bear_berries :=
  sorry

end betty_berries_july_five_l317_31727


namespace right_triangle_area_l317_31701

theorem right_triangle_area (a_square_area b_square_area hypotenuse_square_area : ℝ)
  (ha : a_square_area = 36) (hb : b_square_area = 64) (hc : hypotenuse_square_area = 100)
  (leg1 leg2 hypotenuse : ℝ)
  (hleg1 : leg1 * leg1 = a_square_area)
  (hleg2 : leg2 * leg2 = b_square_area)
  (hhyp : hypotenuse * hypotenuse = hypotenuse_square_area) :
  (1/2) * leg1 * leg2 = 24 :=
by
  sorry

end right_triangle_area_l317_31701


namespace probability_different_colors_is_correct_l317_31775

-- Definitions of chip counts
def blue_chips := 6
def red_chips := 5
def yellow_chips := 4
def green_chips := 3
def total_chips := blue_chips + red_chips + yellow_chips + green_chips

-- Definition of the probability calculation
def probability_different_colors := 
  ((blue_chips / total_chips) * ((red_chips + yellow_chips + green_chips) / total_chips)) +
  ((red_chips / total_chips) * ((blue_chips + yellow_chips + green_chips) / total_chips)) +
  ((yellow_chips / total_chips) * ((blue_chips + red_chips + green_chips) / total_chips)) +
  ((green_chips / total_chips) * ((blue_chips + red_chips + yellow_chips) / total_chips))

-- Given the problem conditions, we assert the correct answer
theorem probability_different_colors_is_correct :
  probability_different_colors = (119 / 162) := 
sorry

end probability_different_colors_is_correct_l317_31775


namespace geometric_sequence_sum_l317_31745

theorem geometric_sequence_sum (a : Nat → ℝ) (q : ℝ) (h_geom : ∀ n, a (n + 1) = q * a n)
  (hq : q > 1) (h2011_root : 4 * a 2011 ^ 2 - 8 * a 2011 + 3 = 0)
  (h2012_root : 4 * a 2012 ^ 2 - 8 * a 2012 + 3 = 0) :
  a 2013 + a 2014 = 18 :=
sorry

end geometric_sequence_sum_l317_31745


namespace selling_price_l317_31791

theorem selling_price (cost_price : ℝ) (loss_percentage : ℝ) : 
    cost_price = 1600 → loss_percentage = 0.15 → 
    (cost_price - (loss_percentage * cost_price)) = 1360 :=
by
  intros h_cp h_lp
  rw [h_cp, h_lp]
  norm_num

end selling_price_l317_31791


namespace fries_remaining_time_l317_31793

def recommendedTime : ℕ := 5 * 60
def timeInOven : ℕ := 45
def remainingTime : ℕ := recommendedTime - timeInOven

theorem fries_remaining_time : remainingTime = 255 :=
by
  sorry

end fries_remaining_time_l317_31793


namespace eval_expression_l317_31717

theorem eval_expression : 3 * (3 + 3) / 3 = 6 := by
  sorry

end eval_expression_l317_31717


namespace smallest_b_for_fraction_eq_l317_31702

theorem smallest_b_for_fraction_eq (a b : ℕ) (h1 : 1000 ≤ a ∧ a < 10000) (h2 : 100000 ≤ b ∧ b < 1000000)
(h3 : 1/2006 = 1/a + 1/b) : b = 120360 := sorry

end smallest_b_for_fraction_eq_l317_31702


namespace arithmetic_sequence_sum_l317_31776

theorem arithmetic_sequence_sum 
  (a : ℕ → ℤ) 
  (h_arith : ∀ n : ℕ, a (n+1) - a n = a 1 - a 0) 
  (h_sum : a 2 + a 4 + a 6 = 12) : 
  (a 0 + a 1 + a 2 + a 3 + a 4 + a 5 + a 6 = 28) :=
sorry

end arithmetic_sequence_sum_l317_31776


namespace find_m_of_odd_number_sequence_l317_31757

theorem find_m_of_odd_number_sequence : 
  ∃ m : ℕ, m > 1 ∧ (∃ a : ℕ, a = m * (m - 1) + 1 ∧ a = 2023) ↔ m = 45 :=
by
    sorry

end find_m_of_odd_number_sequence_l317_31757


namespace no_opposite_identical_numbers_l317_31742

open Finset

theorem no_opposite_identical_numbers : 
  ∀ (f g : Fin 20 → Fin 20), 
  (∀ i : Fin 20, ∃ j : Fin 20, f j = i ∧ g j = (i + j) % 20) → 
  ∃ k : ℤ, ∀ i : Fin 20, f (i + k) % 20 ≠ g i 
  := by
    sorry

end no_opposite_identical_numbers_l317_31742


namespace radius_of_circumscribed_circle_l317_31789

theorem radius_of_circumscribed_circle 
    (r : ℝ)
    (h : 8 * r = π * r^2) : 
    r = 8 / π :=
by
    sorry

end radius_of_circumscribed_circle_l317_31789

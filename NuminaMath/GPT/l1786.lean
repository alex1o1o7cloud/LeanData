import Mathlib

namespace curve_crosses_itself_l1786_178610

theorem curve_crosses_itself :
  ∃ t1 t2 : ℝ, t1 ≠ t2 ∧ (t1^2 - 3 = t2^2 - 3) ∧ (t1^3 - 6*t1 + 2 = t2^3 - 6*t2 + 2) ∧
  ((t1^2 - 3 = 3) ∧ (t1^3 - 6*t1 + 2 = 2)) :=
by
  sorry

end curve_crosses_itself_l1786_178610


namespace num_assignments_l1786_178619

/-- 
Mr. Wang originally planned to grade at a rate of 6 assignments per hour.
After grading for 2 hours, he increased his rate to 8 assignments per hour,
finishing 3 hours earlier than initially planned. 
Prove that the total number of assignments is 84. 
-/
theorem num_assignments (x : ℕ) (h : ℕ) (H1 : 6 * h = x) (H2 : 8 * (h - 5) = x - 12) : x = 84 :=
by
  sorry

end num_assignments_l1786_178619


namespace largest_value_l1786_178680

theorem largest_value (A B C D E : ℕ)
  (hA : A = (3 + 5 + 2 + 8))
  (hB : B = (3 * 5 + 2 + 8))
  (hC : C = (3 + 5 * 2 + 8))
  (hD : D = (3 + 5 + 2 * 8))
  (hE : E = (3 * 5 * 2 * 8)) :
  max (max (max (max A B) C) D) E = E := 
sorry

end largest_value_l1786_178680


namespace preservation_time_at_33_degrees_l1786_178628

noncomputable def preservation_time (x : ℝ) (k : ℝ) (b : ℝ) : ℝ :=
  Real.exp (k * x + b)

theorem preservation_time_at_33_degrees (k b : ℝ) 
  (h1 : Real.exp b = 192)
  (h2 : Real.exp (22 * k + b) = 48) :
  preservation_time 33 k b = 24 := by
  sorry

end preservation_time_at_33_degrees_l1786_178628


namespace triangle_QR_length_l1786_178664

noncomputable def length_PM : ℝ := 6 -- PM = 6 cm
noncomputable def length_MA : ℝ := 12 -- MA = 12 cm
noncomputable def length_NB : ℝ := 9 -- NB = 9 cm
def MN_parallel_PQ : Prop := true -- MN ∥ PQ

theorem triangle_QR_length 
  (h1 : MN_parallel_PQ)
  (h2 : length_PM = 6)
  (h3 : length_MA = 12)
  (h4 : length_NB = 9) : 
  length_QR = 27 :=
sorry

end triangle_QR_length_l1786_178664


namespace pine_saplings_in_sample_l1786_178600

-- Definitions based on conditions
def total_saplings : ℕ := 30000
def pine_saplings : ℕ := 4000
def sample_size : ℕ := 150

-- Main theorem to prove
theorem pine_saplings_in_sample : (pine_saplings * sample_size) / total_saplings = 20 :=
by sorry

end pine_saplings_in_sample_l1786_178600


namespace total_balloons_l1786_178653

-- Define the number of yellow balloons each person has
def tom_balloons : Nat := 18
def sara_balloons : Nat := 12
def alex_balloons : Nat := 7

-- Prove that the total number of balloons is 37
theorem total_balloons : tom_balloons + sara_balloons + alex_balloons = 37 := 
by 
  sorry

end total_balloons_l1786_178653


namespace part_I_part_II_part_III_l1786_178641

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 + a * x + 1
noncomputable def g (x : ℝ) : ℝ := Real.exp x

-- Part (I)
theorem part_I (a : ℝ) (h_a : a = 1) : 
  ∃ (x : ℝ), x ∈ Set.Icc (-2 : ℝ) 0 ∧ f x a * g x = 1 := sorry

-- Part (II)
theorem part_II (a : ℝ) (h_a : a = -1) (k : ℝ) :
  (∃ x : ℝ, f x a = k * g x ∧ ∀ y : ℝ, y ≠ x → f y a ≠ k * g y) ↔ 
  (k > 3 * Real.exp (-2) ∨ (0 < k ∧ k < 1 * Real.exp (-1))) := sorry

-- Part (III)
theorem part_III (a : ℝ) :
  (∀ (x₁ x₂ : ℝ), (x₁ ∈ Set.Icc 0 2 ∧ x₂ ∈ Set.Icc 0 2 ∧ x₁ ≠ x₂) →
  abs (f x₁ a - f x₂ a) < abs (g x₁ - g x₂)) ↔
  (-1 ≤ a ∧ a ≤ 2 - 2 * Real.log 2) := sorry

end part_I_part_II_part_III_l1786_178641


namespace min_cost_theater_tickets_l1786_178616

open Real

variable (x y : ℝ)

theorem min_cost_theater_tickets :
  (x + y = 140) →
  (y ≥ 2 * x) →
  ∀ x y, 60 * x + 100 * y ≥ 12160 :=
by
  sorry

end min_cost_theater_tickets_l1786_178616


namespace balls_remaining_l1786_178640

-- Define the initial number of balls in the box
def initial_balls := 10

-- Define the number of balls taken by Yoongi
def balls_taken := 3

-- Define the number of balls left after Yoongi took some balls
def balls_left := initial_balls - balls_taken

-- The theorem statement to be proven
theorem balls_remaining : balls_left = 7 :=
by
    -- Skipping the proof
    sorry

end balls_remaining_l1786_178640


namespace largest_square_tile_for_board_l1786_178625

theorem largest_square_tile_for_board (length width gcd_val : ℕ) (h1 : length = 16) (h2 : width = 24) 
  (h3 : gcd_val = Int.gcd length width) : gcd_val = 8 := by
  sorry

end largest_square_tile_for_board_l1786_178625


namespace prize_difference_l1786_178650

def mateo_hourly_rate : ℕ := 20
def sydney_daily_rate : ℕ := 400
def hours_in_a_week : ℕ := 24 * 7
def days_in_a_week : ℕ := 7

def mateo_total : ℕ := mateo_hourly_rate * hours_in_a_week
def sydney_total : ℕ := sydney_daily_rate * days_in_a_week

def difference_amount : ℕ := 560

theorem prize_difference : mateo_total - sydney_total = difference_amount := sorry

end prize_difference_l1786_178650


namespace part_a_part_b_l1786_178617

-- Define sum conditions for consecutive odd integers
def consecutive_odd_sum (N : ℕ) : Prop :=
  ∃ (n k : ℕ), n ≥ 2 ∧ N = n * (2 * k + n)

-- Part (a): Prove 2005 can be written as sum of consecutive odd positive integers
theorem part_a : consecutive_odd_sum 2005 :=
by
  sorry

-- Part (b): Prove 2006 cannot be written as sum of consecutive odd positive integers
theorem part_b : ¬consecutive_odd_sum 2006 :=
by
  sorry

end part_a_part_b_l1786_178617


namespace infinite_geometric_series_sum_l1786_178622

theorem infinite_geometric_series_sum :
  let a := (5 : ℚ) / 3
  let r := -(3 : ℚ) / 4
  |r| < 1 →
  (∀ S, S = a / (1 - r) → S = 20 / 21) :=
by
  intros a r h_abs_r S h_S
  sorry

end infinite_geometric_series_sum_l1786_178622


namespace max_xy_l1786_178693

theorem max_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 7 * x + 8 * y = 112) : xy ≤ 56 :=
sorry

end max_xy_l1786_178693


namespace probability_satisfies_inequality_l1786_178607

/-- Define the conditions for the points (x, y) -/
def within_rectangle (x y : ℝ) : Prop :=
  0 ≤ x ∧ x ≤ 4 ∧ 0 ≤ y ∧ y ≤ 5

def satisfies_inequality (x y : ℝ) : Prop :=
  x + 2 * y ≤ 6

/-- Compute the probability that a randomly selected point within the rectangle
also satisfies the inequality -/
theorem probability_satisfies_inequality : (∃ p : ℚ, p = 3 / 10) :=
sorry

end probability_satisfies_inequality_l1786_178607


namespace evaluate_expression_l1786_178659

theorem evaluate_expression : 
  |-2| + (1 / 4) - 1 - 4 * Real.cos (Real.pi / 4) + Real.sqrt 8 = 5 / 4 :=
by
  sorry

end evaluate_expression_l1786_178659


namespace david_cups_consumed_l1786_178690

noncomputable def cups_of_water (time_in_minutes : ℕ) : ℝ :=
  time_in_minutes / 20

theorem david_cups_consumed : cups_of_water 225 = 11.25 := by
  sorry

end david_cups_consumed_l1786_178690


namespace cheapest_shipping_option_l1786_178676

/-- Defines the cost options for shipping, given a weight of 5 pounds. -/
def cost_A (weight : ℕ) : ℝ := 5.00 + 0.80 * weight
def cost_B (weight : ℕ) : ℝ := 4.50 + 0.85 * weight
def cost_C (weight : ℕ) : ℝ := 3.00 + 0.95 * weight

/-- Proves that for a package weighing 5 pounds, the cheapest shipping option is Option C costing $7.75. -/
theorem cheapest_shipping_option : cost_C 5 < cost_A 5 ∧ cost_C 5 < cost_B 5 ∧ cost_C 5 = 7.75 :=
by
  -- Calculation is omitted
  sorry

end cheapest_shipping_option_l1786_178676


namespace parallel_lines_coefficient_l1786_178623

theorem parallel_lines_coefficient (a : ℝ) : 
  (∀ x y : ℝ, (a * x + 2 * y + 2 = 0) → (3 * x - y - 2 = 0)) → a = -6 :=
  by
    sorry

end parallel_lines_coefficient_l1786_178623


namespace total_people_l1786_178665

theorem total_people (N B : ℕ) (h1 : N = 4 * B + 10) (h2 : N = 5 * B + 1) : N = 46 := by
  -- The proof will follow from the conditions, but it is not required in this script.
  sorry

end total_people_l1786_178665


namespace bus_stops_for_4_minutes_per_hour_l1786_178657

theorem bus_stops_for_4_minutes_per_hour
  (V_excluding_stoppages V_including_stoppages : ℝ)
  (h1 : V_excluding_stoppages = 90)
  (h2 : V_including_stoppages = 84) :
  (60 * (V_excluding_stoppages - V_including_stoppages)) / V_excluding_stoppages = 4 :=
by
  sorry

end bus_stops_for_4_minutes_per_hour_l1786_178657


namespace amount_diana_owes_l1786_178652

-- Problem definitions
def principal : ℝ := 75
def rate : ℝ := 0.07
def time : ℝ := 1
def interest := principal * rate * time
def total_owed := principal + interest

-- Theorem to prove that the total amount owed is $80.25
theorem amount_diana_owes : total_owed = 80.25 := by
  sorry

end amount_diana_owes_l1786_178652


namespace find_y_l1786_178687

variable (x y z : ℕ)

-- Conditions
def condition1 : Prop := 100 + 200 + 300 + x = 1000
def condition2 : Prop := 300 + z + 100 + x + y = 1000

-- Theorem to be proven
theorem find_y (h1 : condition1 x) (h2 : condition2 x y z) : z + y = 200 :=
sorry

end find_y_l1786_178687


namespace geometric_mean_of_1_and_4_l1786_178651

theorem geometric_mean_of_1_and_4 :
  ∃ a : ℝ, a^2 = 4 ∧ (a = 2 ∨ a = -2) :=
by
  sorry

end geometric_mean_of_1_and_4_l1786_178651


namespace find_solutions_l1786_178678

theorem find_solutions (x y : ℝ) :
    (x * y^2 = 15 * x^2 + 17 * x * y + 15 * y^2 ∧ x^2 * y = 20 * x^2 + 3 * y^2) ↔ 
    (x = 0 ∧ y = 0) ∨ (x = -19 ∧ y = -2) :=
by sorry

end find_solutions_l1786_178678


namespace smallest_positive_integer_mod_l1786_178631

theorem smallest_positive_integer_mod (a : ℕ) (h1 : a ≡ 4 [MOD 5]) (h2 : a ≡ 6 [MOD 7]) : a = 34 :=
by
  sorry

end smallest_positive_integer_mod_l1786_178631


namespace cuboidal_box_area_l1786_178630

/-- Given conditions about a cuboidal box:
    - The area of one face is 72 cm²
    - The area of an adjacent face is 60 cm²
    - The volume of the cuboidal box is 720 cm³,
    Prove that the area of the third adjacent face is 120 cm². -/
theorem cuboidal_box_area (l w h : ℝ) (h1 : l * w = 72) (h2 : w * h = 60) (h3 : l * w * h = 720) :
  l * h = 120 :=
sorry

end cuboidal_box_area_l1786_178630


namespace minimum_path_proof_l1786_178669

noncomputable def minimum_path (r : ℝ) (h : ℝ) (d1 : ℝ) (d2 : ℝ) : ℝ :=
  let R := Real.sqrt (r^2 + h^2)
  let theta := 2 * Real.pi * (R / (2 * Real.pi * r))
  let A := (d1, 0)
  let B := (-d2 * Real.cos (theta / 2), -d2 * Real.sin (theta / 2))
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

theorem minimum_path_proof :
  minimum_path 800 (300 * Real.sqrt 3) 150 (450 * Real.sqrt 2) = 562.158 := 
by 
  sorry

end minimum_path_proof_l1786_178669


namespace solve_diamond_l1786_178644

theorem solve_diamond : 
  (∃ (Diamond : ℤ), Diamond * 5 + 3 = Diamond * 6 + 2) →
  (∃ (Diamond : ℤ), Diamond = 1) :=
by
  sorry

end solve_diamond_l1786_178644


namespace total_journey_distance_l1786_178648

/-- 
A woman completes a journey in 5 hours. She travels the first half of the journey 
at 21 km/hr and the second half at 24 km/hr. Find the total journey in km.
-/
theorem total_journey_distance :
  ∃ D : ℝ, (D / 2) / 21 + (D / 2) / 24 = 5 ∧ D = 112 :=
by
  use 112
  -- Please prove the following statements
  sorry

end total_journey_distance_l1786_178648


namespace simplify_expr_l1786_178611

variable (x : ℝ)

theorem simplify_expr : (2 * x^2 + 5 * x - 7) - (x^2 + 9 * x - 3) = x^2 - 4 * x - 4 :=
by
  sorry

end simplify_expr_l1786_178611


namespace find_n_from_binomial_term_l1786_178683

noncomputable def binomial_coefficient (n r : ℕ) : ℕ :=
  (Nat.factorial n) / ((Nat.factorial r) * (Nat.factorial (n - r)))

theorem find_n_from_binomial_term :
  (∃ n : ℕ, 3^2 * binomial_coefficient n 2 = 54) ↔ n = 4 :=
by
  sorry

end find_n_from_binomial_term_l1786_178683


namespace minimum_value_l1786_178685

open Real

-- Given the conditions
variables (a b c k : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hk : 0 < k)

-- The theorem
theorem minimum_value (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < k) : 
  ∃ x, x = (3 : ℝ) / k ∧ ∀ y, y = (a / (k * b) + b / (k * c) + c / (k * a)) → y ≥ x :=
sorry

end minimum_value_l1786_178685


namespace part1_part2_l1786_178613

open Real

variables (x a : ℝ)

def p (x a : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0 ∧ x^2 + 2 * x - 8 > 0

theorem part1 (h : a = 1) (h_pq : p x 1 ∧ q x) : 2 < x ∧ x < 3 :=
by sorry

theorem part2 (hpq : ∀ (a x : ℝ), ¬ p x a → ¬ q x) : 1 ≤ a ∧ a ≤ 2 :=
by sorry

end part1_part2_l1786_178613


namespace distinct_arrangements_balloon_l1786_178649

theorem distinct_arrangements_balloon : 
  let n := 7 
  let freq_l := 2 
  let freq_o := 2 
  let freq_b := 1 
  let freq_a := 1 
  let freq_n := 1 
  Nat.factorial n / (Nat.factorial freq_l * Nat.factorial freq_o * Nat.factorial freq_b * Nat.factorial freq_a * Nat.factorial freq_n) = 1260 :=
by
  sorry

end distinct_arrangements_balloon_l1786_178649


namespace sum_first_4_terms_of_arithmetic_sequence_eq_8_l1786_178674

def arithmetic_seq (a : ℕ → ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + (a 1 - a 0)

def S4 (a : ℕ → ℤ) : ℤ :=
  a 0 + a 1 + a 2 + a 3

theorem sum_first_4_terms_of_arithmetic_sequence_eq_8
  (a : ℕ → ℤ) 
  (h_seq : arithmetic_seq a) 
  (h_a2 : a 1 = 1) 
  (h_a3 : a 2 = 3) :
  S4 a = 8 :=
by
  sorry

end sum_first_4_terms_of_arithmetic_sequence_eq_8_l1786_178674


namespace complement_N_star_in_N_l1786_178654

-- The set of natural numbers
def N : Set ℕ := { n | true }

-- The set of positive integers
def N_star : Set ℕ := { n | n > 0 }

-- The complement of N_star in N is the set {0}
theorem complement_N_star_in_N : { n | n ∈ N ∧ n ∉ N_star } = {0} := by
  sorry

end complement_N_star_in_N_l1786_178654


namespace sector_area_l1786_178679

-- Define the properties and conditions
def perimeter_of_sector (r l : ℝ) : Prop :=
  l + 2 * r = 8

def central_angle_arc_length (r : ℝ) : ℝ :=
  2 * r

-- Theorem to prove the area of the sector
theorem sector_area (r : ℝ) (l : ℝ) 
  (h_perimeter : perimeter_of_sector r l) 
  (h_arc_length : l = central_angle_arc_length r) : 
  1 / 2 * l * r = 4 := 
by
  -- This is the place where the proof would go; we use sorry to indicate it's incomplete
  sorry

end sector_area_l1786_178679


namespace problem_sin_cos_k_l1786_178639

open Real

theorem problem_sin_cos_k {k : ℝ} :
  (∃ x : ℝ, sin x ^ 2 + cos x + k = 0) ↔ -2 ≤ k ∧ k ≤ 0 := by
  sorry

end problem_sin_cos_k_l1786_178639


namespace find_y_when_x_is_1_l1786_178689

theorem find_y_when_x_is_1 
  (k : ℝ) 
  (h1 : ∀ y, x = k / y^2) 
  (h2 : x = 1) 
  (h3 : x = 0.1111111111111111) 
  (y : ℝ) 
  (hy : y = 6) 
  (hx_k : k = 0.1111111111111111 * 36) :
  y = 2 := sorry

end find_y_when_x_is_1_l1786_178689


namespace range_of_x_l1786_178660

noncomputable def f (x : ℝ) : ℝ := (1 / 3) * x^3 - (1 / x) + 2 * Real.sin x

theorem range_of_x (x : ℝ) (h₀ : x > 0) (h₁ : f (1 - x) > f x) : x < (1 / 2) :=
by
  sorry

end range_of_x_l1786_178660


namespace parallel_lines_l1786_178634

theorem parallel_lines (a : ℝ) :
  (∀ x y : ℝ, 2 * x - a * y + 1 = 0) ∧ (∀ x y : ℝ, (a-1) * x - y + a = 0) →
  (a = 2 ↔ (∀ x1 y1 x2 y2 : ℝ, 2 * x1 - a * y1 + 1 = 0 ∧ (a-1) * x2 - y2 + a = 0 →
  (2 * x1 = (a * y1 - 1) ∧ (a-1) * x2 = y2 - a))) :=
sorry

end parallel_lines_l1786_178634


namespace problem1_l1786_178662

noncomputable def log6_7 : ℝ := Real.logb 6 7
noncomputable def log7_6 : ℝ := Real.logb 7 6

theorem problem1 : log6_7 > log7_6 := 
by
  sorry

end problem1_l1786_178662


namespace necessary_but_not_sufficient_not_sufficient_condition_l1786_178668

theorem necessary_but_not_sufficient (a b : ℝ) : (a > 2 ∧ b > 2) → (a + b > 4) :=
sorry

theorem not_sufficient_condition (a b : ℝ) : (a + b > 4) → ¬(a > 2 ∧ b > 2) :=
sorry

end necessary_but_not_sufficient_not_sufficient_condition_l1786_178668


namespace arithmetic_sequence_a3_is_8_l1786_178629

-- Define the arithmetic sequence
def arithmetic_sequence (a1 d n : ℕ) : ℕ :=
  a1 + (n - 1) * d

-- Theorem to prove a3 = 8 given a1 = 4 and d = 2
theorem arithmetic_sequence_a3_is_8 (a1 d : ℕ) (h1 : a1 = 4) (h2 : d = 2) : arithmetic_sequence a1 d 3 = 8 :=
by
  sorry -- Proof not required as per instruction

end arithmetic_sequence_a3_is_8_l1786_178629


namespace exists_sum_of_digits_div_11_l1786_178612

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem exists_sum_of_digits_div_11 (H : Finset ℕ) (h₁ : H.card = 39) :
  ∃ (a : ℕ) (h : a ∈ H), sum_of_digits a % 11 = 0 :=
by
  sorry

end exists_sum_of_digits_div_11_l1786_178612


namespace find_parabola_equation_l1786_178604

-- Define the problem conditions
def parabola_vertex_at_origin (f : ℝ → ℝ) : Prop :=
  f 0 = 0

def axis_of_symmetry_x_or_y (f : ℝ → ℝ) : Prop :=
  (∀ x, f x = 0) ∨ (∀ y, f 0 = y)

def passes_through_point (f : ℝ → ℝ) (pt : ℝ × ℝ) : Prop :=
  f pt.1 = pt.2

-- Define the specific forms we expect the equations of the parabola to take
def equation1 (x y : ℝ) : Prop :=
  y^2 = - (9 / 2) * x

def equation2 (x y : ℝ) : Prop :=
  x^2 = (4 / 3) * y

-- state the main theorem
theorem find_parabola_equation :
  ∃ f : ℝ → ℝ, parabola_vertex_at_origin f ∧ axis_of_symmetry_x_or_y f ∧ passes_through_point f (-2, 3) ∧
  (equation1 (-2) (f (-2)) ∨ equation2 (-2) (f (-2))) :=
sorry

end find_parabola_equation_l1786_178604


namespace non_degenerate_triangles_l1786_178638

theorem non_degenerate_triangles :
  let total_points := 16
  let collinear_points := 5
  let total_triangles := Nat.choose total_points 3
  let degenerate_triangles := 2 * Nat.choose collinear_points 3
  let nondegenerate_triangles := total_triangles - degenerate_triangles
  nondegenerate_triangles = 540 := 
by
  sorry

end non_degenerate_triangles_l1786_178638


namespace height_difference_l1786_178661

-- Define the heights of Eiffel Tower and Burj Khalifa as constants
def eiffelTowerHeight : ℕ := 324
def burjKhalifaHeight : ℕ := 830

-- Define the statement that needs to be proven
theorem height_difference : burjKhalifaHeight - eiffelTowerHeight = 506 := by
  sorry

end height_difference_l1786_178661


namespace hyperbola_range_m_l1786_178698

-- Define the condition that the equation represents a hyperbola
def isHyperbola (m : ℝ) : Prop := (2 + m) * (m + 1) < 0

-- The theorem stating the range of m given the condition
theorem hyperbola_range_m (m : ℝ) : isHyperbola m → -2 < m ∧ m < -1 := by
  sorry

end hyperbola_range_m_l1786_178698


namespace inscribed_circle_radius_l1786_178672

noncomputable def radius_of_inscribed_circle (AB AC BC : ℝ) : ℝ :=
  let s := (AB + AC + BC) / 2
  let K := Real.sqrt (s * (s - AB) * (s - AC) * (s - BC))
  K / s

theorem inscribed_circle_radius :
  radius_of_inscribed_circle 8 8 5 = 38 / 21 :=
by
  sorry

end inscribed_circle_radius_l1786_178672


namespace max_ratio_two_digit_mean_50_l1786_178643

theorem max_ratio_two_digit_mean_50 : 
  ∀ (x y : ℕ), (10 ≤ x ∧ x ≤ 99) ∧ (10 ≤ y ∧ y ≤ 99) ∧ (x + y = 100) → ( x / y ) ≤ 99 := 
by
  intros x y h
  obtain ⟨hx, hy, hsum⟩ := h
  sorry

end max_ratio_two_digit_mean_50_l1786_178643


namespace total_questions_attempted_l1786_178603

theorem total_questions_attempted (C W T : ℕ) (hC : C = 42) (h_score : 4 * C - W = 150) : T = C + W → T = 60 :=
by
  sorry

end total_questions_attempted_l1786_178603


namespace sequence_result_l1786_178618

theorem sequence_result (initial_value : ℕ) (total_steps : ℕ) 
    (net_effect_one_cycle : ℕ) (steps_per_cycle : ℕ) : 
    initial_value = 100 ∧ total_steps = 26 ∧ 
    net_effect_one_cycle = (15 - 12 + 3) ∧ steps_per_cycle = 3 
    → 
    ∀ (resulting_value : ℕ), resulting_value = 151 :=
by
  sorry

end sequence_result_l1786_178618


namespace which_is_lying_l1786_178614

-- Ben's statement
def ben_says (dan_truth cam_truth : Bool) : Bool :=
  (dan_truth ∧ ¬ cam_truth) ∨ (¬ dan_truth ∧ cam_truth)

-- Dan's statement
def dan_says (ben_truth cam_truth : Bool) : Bool :=
  (ben_truth ∧ ¬ cam_truth) ∨ (¬ ben_truth ∧ cam_truth)

-- Cam's statement
def cam_says (ben_truth dan_truth : Bool) : Bool :=
  ¬ ben_truth ∧ ¬ dan_truth

-- Lean statement to be proven
theorem which_is_lying :
  (∃ (ben_truth dan_truth cam_truth : Bool), 
    ben_says dan_truth cam_truth ∧ 
    dan_says ben_truth cam_truth ∧ 
    cam_says ben_truth dan_truth ∧
    ¬ ben_truth ∧ ¬ dan_truth ∧ cam_truth) ↔ (¬ ben_truth ∧ ¬ dan_truth ∧ cam_truth) :=
sorry

end which_is_lying_l1786_178614


namespace factorize_expr_l1786_178627

theorem factorize_expr (x : ℝ) : (x - 1) * (x + 3) + 4 = (x + 1) ^ 2 :=
by
  sorry

end factorize_expr_l1786_178627


namespace price_of_chips_l1786_178673

theorem price_of_chips (P : ℝ) (h1 : 1.5 = 1.5) (h2 : 45 = 45) (h3 : 15 = 15) (h4 : 10 = 10) :
  15 * P + 10 * 1.5 = 45 → P = 2 :=
by
  sorry

end price_of_chips_l1786_178673


namespace angie_pretzels_l1786_178699

theorem angie_pretzels (Barry_Shelly: ℕ) (Shelly_Angie: ℕ) :
  (Barry_Shelly = 12 / 2) → (Shelly_Angie = 3 * Barry_Shelly) → (Barry_Shelly = 6) → (Shelly_Angie = 18) :=
by
  intro h1 h2 h3
  sorry

end angie_pretzels_l1786_178699


namespace desktops_to_sell_l1786_178667

theorem desktops_to_sell (laptops desktops : ℕ) (ratio_laptops desktops_sold laptops_expected : ℕ) :
  ratio_laptops = 5 → desktops_sold = 3 → laptops_expected = 40 → 
  desktops = (desktops_sold * laptops_expected) / ratio_laptops :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  simp
  sorry -- This is where the proof would go, but it's not needed for this task

end desktops_to_sell_l1786_178667


namespace area_of_given_triangle_l1786_178675

def point := (ℝ × ℝ)

def area_of_triangle (A B C : point) : ℝ :=
  0.5 * (B.1 - A.1) * (C.2 - A.2)

theorem area_of_given_triangle :
  area_of_triangle (0, 0) (4, 0) (4, 6) = 12.0 :=
by 
  sorry

end area_of_given_triangle_l1786_178675


namespace triangle_area_change_l1786_178681

theorem triangle_area_change (B H : ℝ) (hB : B > 0) (hH : H > 0) :
  let A_original := (B * H) / 2
  let H_new := H * 0.60
  let B_new := B * 1.40
  let A_new := (B_new * H_new) / 2
  (A_new = A_original * 0.84) :=
by
  sorry

end triangle_area_change_l1786_178681


namespace curve_is_line_l1786_178608

theorem curve_is_line : ∀ (r θ : ℝ), r = 2 / (2 * Real.sin θ - Real.cos θ) → ∃ m b, ∀ (x y : ℝ), x = r * Real.cos θ → y = r * Real.sin θ → y = m * x + b :=
by
  intros r θ h
  sorry

end curve_is_line_l1786_178608


namespace calc_expr_value_l1786_178606

theorem calc_expr_value : (0.5 ^ 4) / (0.05 ^ 2.5) = 559.06 := 
by 
  sorry

end calc_expr_value_l1786_178606


namespace fencing_problem_l1786_178658

theorem fencing_problem (W L : ℝ) (hW : W = 40) (hArea : W * L = 320) : 
  2 * L + W = 56 :=
by
  sorry

end fencing_problem_l1786_178658


namespace TileD_in_AreaZ_l1786_178645

namespace Tiles

structure Tile :=
  (top : ℕ)
  (right : ℕ)
  (bottom : ℕ)
  (left : ℕ)

def TileA : Tile := {top := 5, right := 3, bottom := 2, left := 4}
def TileB : Tile := {top := 2, right := 4, bottom := 5, left := 3}
def TileC : Tile := {top := 3, right := 6, bottom := 1, left := 5}
def TileD : Tile := {top := 5, right := 2, bottom := 3, left := 6}

variables (X Y Z W : Tile)
variable (tiles : List Tile := [TileA, TileB, TileC, TileD])

noncomputable def areaZContains : Tile := sorry

theorem TileD_in_AreaZ  : areaZContains = TileD := sorry

end Tiles

end TileD_in_AreaZ_l1786_178645


namespace evaporation_fraction_l1786_178605

theorem evaporation_fraction (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 1)
  (h : (1 - x) * (3 / 4) = 1 / 6) : x = 7 / 9 :=
by
  sorry

end evaporation_fraction_l1786_178605


namespace integer_solutions_inequality_system_l1786_178671

noncomputable def check_inequality_system (x : ℤ) : Prop :=
  (3 * x + 1 < x - 3) ∧ ((1 + x) / 2 ≤ (1 + 2 * x) / 3 + 1)

theorem integer_solutions_inequality_system :
  {x : ℤ | check_inequality_system x} = {-5, -4, -3} :=
by
  sorry

end integer_solutions_inequality_system_l1786_178671


namespace fraction_of_emilys_coins_l1786_178691

theorem fraction_of_emilys_coins {total_states : ℕ} (h1 : total_states = 30)
    {states_from_1790_to_1799 : ℕ} (h2 : states_from_1790_to_1799 = 9) :
    (states_from_1790_to_1799 / total_states : ℚ) = 3 / 10 := by
  sorry

end fraction_of_emilys_coins_l1786_178691


namespace sequence_1005th_term_l1786_178626

-- Definitions based on conditions
def first_term : ℚ := sorry
def second_term : ℚ := 10
def third_term : ℚ := 4 * first_term - (1:ℚ)
def fourth_term : ℚ := 4 * first_term + (1:ℚ)

-- Common difference
def common_difference : ℚ := (fourth_term - third_term)

-- Arithmetic sequence term calculation
def nth_term (a d : ℚ) (n : ℕ) : ℚ :=
  a + (n-1) * d

-- Theorem statement
theorem sequence_1005th_term : nth_term first_term common_difference 1005 = 5480 := sorry

end sequence_1005th_term_l1786_178626


namespace cost_of_paintbrush_l1786_178601

noncomputable def cost_of_paints : ℝ := 4.35
noncomputable def cost_of_easel : ℝ := 12.65
noncomputable def amount_already_has : ℝ := 6.50
noncomputable def additional_amount_needed : ℝ := 12.00

-- Let's define the total cost needed and the total costs of items
noncomputable def total_cost_of_paints_and_easel : ℝ := cost_of_paints + cost_of_easel
noncomputable def total_amount_needed : ℝ := amount_already_has + additional_amount_needed

-- And now we can state our theorem that needs to be proved.
theorem cost_of_paintbrush : total_amount_needed - total_cost_of_paints_and_easel = 1.50 :=
by
  sorry

end cost_of_paintbrush_l1786_178601


namespace total_blue_balloons_l1786_178697

def joan_blue_balloons : ℕ := 60
def melanie_blue_balloons : ℕ := 85
def alex_blue_balloons : ℕ := 37
def gary_blue_balloons : ℕ := 48

theorem total_blue_balloons :
  joan_blue_balloons + melanie_blue_balloons + alex_blue_balloons + gary_blue_balloons = 230 :=
by simp [joan_blue_balloons, melanie_blue_balloons, alex_blue_balloons, gary_blue_balloons]

end total_blue_balloons_l1786_178697


namespace time_comparison_l1786_178688

-- Definitions from the conditions
def speed_first_trip (v : ℝ) : ℝ := v
def distance_first_trip : ℝ := 80
def distance_second_trip : ℝ := 240
def speed_second_trip (v : ℝ) : ℝ := 4 * v

-- Theorem to prove
theorem time_comparison (v : ℝ) (hv : v > 0) :
  (distance_second_trip / speed_second_trip v) = (3 / 4) * (distance_first_trip / speed_first_trip v) :=
by
  -- Outline of the proof, we skip the actual steps
  sorry

end time_comparison_l1786_178688


namespace tan_sin_equality_l1786_178692

theorem tan_sin_equality :
  (Real.tan (30 * Real.pi / 180))^2 + (Real.sin (45 * Real.pi / 180))^2 = 5 / 6 :=
by sorry

end tan_sin_equality_l1786_178692


namespace Michelangelo_ceiling_painting_l1786_178642

theorem Michelangelo_ceiling_painting (C : ℕ) : 
  ∃ C, (C + (1/4) * C = 15) ∧ (28 - (C + (1/4) * C) = 13) :=
sorry

end Michelangelo_ceiling_painting_l1786_178642


namespace minimum_sum_l1786_178663

open Matrix

noncomputable def a := 54
noncomputable def b := 40
noncomputable def c := 5
noncomputable def d := 4

theorem minimum_sum 
  (a b c d : ℕ) 
  (ha : 4 * a = 24 * a - 27 * b) 
  (hb : 4 * b = 15 * a - 17 * b) 
  (hc : 3 * c = 24 * c - 27 * d) 
  (hd : 3 * d = 15 * c - 17 * d) 
  (Hpos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) : 
  a + b + c + d = 103 :=
by
  sorry

end minimum_sum_l1786_178663


namespace problem_proof_l1786_178615

-- Define the given conditions and the target statement
theorem problem_proof (a b : ℝ) (h1 : a - b = 2) (h2 : a * b = 10.5) : a^2 + b^2 = 25 := 
by sorry

end problem_proof_l1786_178615


namespace eggs_per_basket_l1786_178609

theorem eggs_per_basket
  (kids : ℕ)
  (friends : ℕ)
  (adults : ℕ)
  (baskets : ℕ)
  (eggs_per_person : ℕ)
  (htotal : kids + friends + adults + 1 = 20)
  (eggs_total : (kids + friends + adults + 1) * eggs_per_person = 180)
  (baskets_count : baskets = 15)
  : (180 / 15) = 12 :=
by
  sorry

end eggs_per_basket_l1786_178609


namespace probability_of_qualified_product_l1786_178666

theorem probability_of_qualified_product :
  let p1 := 0.30   -- Proportion of the first batch
  let d1 := 0.05   -- Defect rate of the first batch
  let p2 := 0.70   -- Proportion of the second batch
  let d2 := 0.04   -- Defect rate of the second batch
  -- Probability of selecting a qualified product
  p1 * (1 - d1) + p2 * (1 - d2) = 0.957 :=
by
  sorry

end probability_of_qualified_product_l1786_178666


namespace statement_1_incorrect_statement_3_incorrect_statement_4_incorrect_l1786_178621

-- Define the notion of line and plane
def Line := Type
def Plane := Type

-- Define the relations: parallel, contained-in, and intersection
def parallel (a b : Line) : Prop := sorry
def contained_in (a : Line) (α : Plane) : Prop := sorry
def intersects_at (a : Line) (α : Plane) (P : Type) : Prop := sorry

-- Conditions translated into Lean
def cond1 (a : Line) (α : Plane) (b : Line) : Prop := parallel a α ∧ contained_in b α → parallel a b
def cond2 (a : Line) (α : Plane) (b : Line) {P : Type} : Prop := intersects_at a α P ∧ contained_in b α → ¬ parallel a b
def cond3 (a : Line) (α : Plane) : Prop := ¬ contained_in a α → parallel a α
def cond4 (a : Line) (α : Plane) (b : Line) : Prop := parallel a α ∧ parallel b α → parallel a b

-- The statements that need to be proved incorrect
theorem statement_1_incorrect (a : Line) (α : Plane) (b : Line) : ¬ (cond1 a α b) := sorry
theorem statement_3_incorrect (a : Line) (α : Plane) : ¬ (cond3 a α) := sorry
theorem statement_4_incorrect (a : Line) (α : Plane) (b : Line) : ¬ (cond4 a α b) := sorry

end statement_1_incorrect_statement_3_incorrect_statement_4_incorrect_l1786_178621


namespace machines_needed_l1786_178637

variables (R x m N : ℕ) (h1 : 4 * R * 6 = x)
           (h2 : N * R * 6 = m * x)

theorem machines_needed : N = m * 4 :=
by sorry

end machines_needed_l1786_178637


namespace min_A_div_B_l1786_178633

theorem min_A_div_B (x A B : ℝ) (hx_pos : 0 < x) (hA_pos : 0 < A) (hB_pos : 0 < B) 
  (h1 : x^2 + 1 / x^2 = A) (h2 : x - 1 / x = B + 3) : 
  (A / B) = 6 + 2 * Real.sqrt 11 :=
sorry

end min_A_div_B_l1786_178633


namespace lees_friend_initial_money_l1786_178670

theorem lees_friend_initial_money (lee_initial_money friend_initial_money total_cost change : ℕ) 
  (h1 : lee_initial_money = 10) 
  (h2 : total_cost = 15) 
  (h3 : change = 3) 
  (h4 : (lee_initial_money + friend_initial_money) - total_cost = change) : 
  friend_initial_money = 8 := by
  sorry

end lees_friend_initial_money_l1786_178670


namespace balance_balls_l1786_178646

variable (R O B P : ℝ)

-- Conditions based on the problem statement
axiom h1 : 4 * R = 8 * B
axiom h2 : 3 * O = 7.5 * B
axiom h3 : 8 * B = 6 * P

-- The theorem we need to prove
theorem balance_balls : 5 * R + 3 * O + 3 * P = 21.5 * B :=
by 
  sorry

end balance_balls_l1786_178646


namespace doris_needs_weeks_l1786_178695

noncomputable def average_weeks_to_cover_expenses (weekly_babysit_hours: ℝ) (saturday_hours: ℝ) : ℝ := 
  let weekday_income := weekly_babysit_hours * 20
  let saturday_income := saturday_hours * (if weekly_babysit_hours > 15 then 15 else 20)
  let teaching_income := 100
  let total_weekly_income := weekday_income + saturday_income + teaching_income
  let monthly_income_before_tax := total_weekly_income * 4
  let monthly_income_after_tax := monthly_income_before_tax * 0.85
  monthly_income_after_tax / 4 / 1200

theorem doris_needs_weeks (weekly_babysit_hours: ℝ) (saturday_hours: ℝ) :
  1200 ≤ (average_weeks_to_cover_expenses weekly_babysit_hours saturday_hours) * 4 * 1200 :=
  by
    sorry

end doris_needs_weeks_l1786_178695


namespace construction_company_total_weight_l1786_178682

noncomputable def total_weight_of_materials_in_pounds : ℝ :=
  let weight_of_concrete := 12568.3
  let weight_of_bricks := 2108 * 2.20462
  let weight_of_stone := 7099.5
  let weight_of_wood := 3778 * 2.20462
  let weight_of_steel := 5879 * (1 / 16)
  let weight_of_glass := 12.5 * 2000
  let weight_of_sand := 2114.8
  weight_of_concrete + weight_of_bricks + weight_of_stone + weight_of_wood + weight_of_steel + weight_of_glass + weight_of_sand

theorem construction_company_total_weight : total_weight_of_materials_in_pounds = 60129.72 :=
by
  sorry

end construction_company_total_weight_l1786_178682


namespace max_value_proof_l1786_178677

noncomputable def max_value_b_minus_a (a b : ℝ) : ℝ :=
  b - a

theorem max_value_proof (a b : ℝ) (h1 : a < 0) (h2 : ∀ x, (x^2 + 2017 * a) * (x + 2016 * b) ≥ 0) : max_value_b_minus_a a b ≤ 2017 :=
sorry

end max_value_proof_l1786_178677


namespace no_x_satisfies_inequality_l1786_178694

def f (x : ℝ) : ℝ := x^2 + x

theorem no_x_satisfies_inequality : ¬ ∃ x : ℝ, f (x - 2) + f x < 0 :=
by 
  unfold f 
  sorry

end no_x_satisfies_inequality_l1786_178694


namespace incorrect_option_D_l1786_178635

-- definition of geometric objects and their properties
def octahedron_faces : Nat := 8
def tetrahedron_can_be_cut_into_4_pyramids : Prop := True
def frustum_extension_lines_intersect_at_a_point : Prop := True
def rectangle_rotated_around_side_forms_cylinder : Prop := True

-- incorrect identification of incorrect statement
theorem incorrect_option_D : 
  (∃ statement : String, statement = "D" ∧ ¬rectangle_rotated_around_side_forms_cylinder)  → False :=
by
  -- Proof of incorrect identification is not required per problem instructions
  sorry

end incorrect_option_D_l1786_178635


namespace avg_of_first_21_multiples_l1786_178656

theorem avg_of_first_21_multiples (n : ℕ) (h : (21 * 11 * n / 21) = 88) : n = 8 :=
by
  sorry

end avg_of_first_21_multiples_l1786_178656


namespace find_first_number_l1786_178624

theorem find_first_number (a b : ℕ) (k : ℕ) (h1 : a = 3 * k) (h2 : b = 4 * k) (h3 : Nat.lcm a b = 84) : a = 21 := 
sorry

end find_first_number_l1786_178624


namespace circumscribed_circle_radius_l1786_178647

noncomputable def circumradius_of_right_triangle (a b c : ℕ) (h : a^2 + b^2 = c^2) : ℚ :=
  c / 2

theorem circumscribed_circle_radius :
  circumradius_of_right_triangle 30 40 50 (by norm_num : 30^2 + 40^2 = 50^2) = 25 := by
norm_num /- correct answer confirmed -/
sorry

end circumscribed_circle_radius_l1786_178647


namespace polygon_with_45_deg_exterior_angle_is_eight_gon_l1786_178620

theorem polygon_with_45_deg_exterior_angle_is_eight_gon
  (each_exterior_angle : ℝ) (h1 : each_exterior_angle = 45) 
  (sum_exterior_angles : ℝ) (h2 : sum_exterior_angles = 360) :
  ∃ (n : ℕ), n = 8 :=
by
  sorry

end polygon_with_45_deg_exterior_angle_is_eight_gon_l1786_178620


namespace train_speed_kmph_l1786_178684

theorem train_speed_kmph (len_train : ℝ) (len_platform : ℝ) (time_cross : ℝ) (total_distance : ℝ) (speed_mps : ℝ) (speed_kmph : ℝ) 
  (h1 : len_train = 250) 
  (h2 : len_platform = 150.03) 
  (h3 : time_cross = 20) 
  (h4 : total_distance = len_train + len_platform) 
  (h5 : speed_mps = total_distance / time_cross) 
  (h6 : speed_kmph = speed_mps * 3.6) : 
  speed_kmph = 72.0054 := 
by 
  -- This is where the proof would go
  sorry

end train_speed_kmph_l1786_178684


namespace off_road_vehicle_cost_l1786_178632

theorem off_road_vehicle_cost
  (dirt_bike_count : ℕ) (dirt_bike_cost : ℕ)
  (off_road_vehicle_count : ℕ) (register_cost : ℕ)
  (total_cost : ℕ) (off_road_vehicle_cost : ℕ) :
  dirt_bike_count = 3 → dirt_bike_cost = 150 →
  off_road_vehicle_count = 4 → register_cost = 25 →
  total_cost = 1825 →
  3 * dirt_bike_cost + 4 * off_road_vehicle_cost + 7 * register_cost = total_cost →
  off_road_vehicle_cost = 300 := by
  intros h1 h2 h3 h4 h5 h6
  sorry

end off_road_vehicle_cost_l1786_178632


namespace average_speed_distance_div_time_l1786_178636

theorem average_speed_distance_div_time (distance : ℕ) (time_minutes : ℕ) (average_speed : ℕ) : 
  distance = 8640 → time_minutes = 36 → average_speed = distance / (time_minutes * 60) → average_speed = 4 := 
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  simp at h3
  assumption

end average_speed_distance_div_time_l1786_178636


namespace three_digit_integers_sum_to_7_l1786_178696

theorem three_digit_integers_sum_to_7 : 
  ∃ n : ℕ, n = 28 ∧ (∃ a b c : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ a + b + c = 7) :=
sorry

end three_digit_integers_sum_to_7_l1786_178696


namespace age_problem_l1786_178602

theorem age_problem (A N : ℕ) (h₁: A = 18) (h₂: N * (A + 3) - N * (A - 3) = A) : N = 3 := by
  sorry

end age_problem_l1786_178602


namespace train_length_l1786_178655

theorem train_length (t_post t_platform l_platform : ℕ) (L : ℚ) : 
  t_post = 15 → t_platform = 25 → l_platform = 100 →
  (L / t_post) = (L + l_platform) / t_platform → 
  L = 150 :=
by 
  intros h1 h2 h3 h4
  -- Proof steps would go here
  sorry

end train_length_l1786_178655


namespace john_cakes_bought_l1786_178686

-- Conditions
def cake_price : ℕ := 12
def john_paid : ℕ := 18

-- Definition of the total cost
def total_cost : ℕ := 2 * john_paid

-- Calculate number of cakes
def num_cakes (total_cost cake_price : ℕ) : ℕ := total_cost / cake_price

-- Theorem to prove that the number of cakes John Smith bought is 3
theorem john_cakes_bought : num_cakes total_cost cake_price = 3 := by
  sorry

end john_cakes_bought_l1786_178686
